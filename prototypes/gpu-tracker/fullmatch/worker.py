"""A Full Match Mode worker process: python -B -m fullmatch.worker [--threads N], started by the scheduler (scheduler.py)
from the prototype folder, one fresh interpreter per worker (Windows spawn: nothing is forked, CUDA starts here).

It loads the detectors and the appearance model once (pipeline.Resources, as the server does for its own runs) and then
runs segment jobs one after another with the unchanged pipeline.run: a fresh tracker per job, the same models. Talking
to the scheduler, JSON lines:

    scheduler -> worker (stdin):   {"type": "job", "job": {...}}   {"type": "cancel"}   {"type": "shutdown"}
    worker -> scheduler (stdout):  ready, progress, done, failed, bye

stdout carries nothing but the protocol: the worker moves file descriptor 1 to stderr at start (Python prints, the
pipeline's event lines, library noise and C-level output all go to the worker log the scheduler keeps) and writes its
messages to a private duplicate of the original stdout. When stdin closes (the server stopped, was closed or crashed)
the worker cancels its job and exits: no worker outlives the app.

A job processes the segment's frames [proc0, proc1) of the source: the worker reads the timestamps of those two frames
(frame-exact seeking; the timestamps are irregular, so they are not computed from the frame index) and hands the
pipeline that start and length. The run writes into a temporary folder next to the final one; when it is complete, the
worker keeps its result compressed (manifest.compress_result: result.json.gz, about 11 times smaller, read back against
the original before the original goes; under a second), writes segment.json (manifest.sidecar: frames, first/last
frame, the stored file, its size and sha256) last and renames the folder into place, so a folder under its final name
is always a finished run. A cancelled or failed run's folder is deleted. A GPU out-of-memory error fails the job (kind
"oom": the scheduler retries it with fewer workers), never the process; the CUDA cache is emptied after every job.

A discovery job ({"kind": "discover"}, the calibration stage before tracking: calibration.py) solves a chunk of sampled
frames on the CPU and writes its rows and frame pairs to one JSON file (atomically). A light worker (--light) loads no
model and takes discovery jobs only: the scheduler adds a few while discovery runs, so that it uses more of the CPU
than the GPU workers alone would, and lets them go when it ends.
"""
import argparse
import gc
import json
import os
import queue
import shutil
import sys
import threading
import time
import traceback
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
THREAD_VARIABLES = ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS')


class Channel:
    """The protocol on the original stdout (fd 1 is redirected to stderr)."""

    def __init__(self):
        fd = os.dup(1)
        os.dup2(2, 1)
        sys.stdout = sys.stderr
        self.out = os.fdopen(fd, 'w', encoding='utf-8', buffering=1, newline='\n')
        self.lock = threading.Lock()

    def send(self, message):
        with self.lock:
            try:
                self.out.write(json.dumps(message, separators=(',', ':'), default=str)+'\n')
                self.out.flush()
            except (OSError, ValueError):
                pass   # the scheduler is gone (stdin closes too, and the worker exits)


class Commands:
    """The scheduler's commands, read on a thread: jobs and shutdown go to the queue, a cancel sets the flag at once,
    EOF means the scheduler is gone (cancel, then exit).

    The pipe is read through a private duplicate, and the process's standard input is pointed at NUL first: on Windows a
    blocking read pending on the standard-input pipe stalls every DLL that initialises its C runtime meanwhile (it asks
    the standard handles for their type, which waits for the read): numpy's import hung that way, and OpenCV's FFmpeg
    and CUDA's libraries load lazily during a run."""

    def __init__(self):
        self.jobs = queue.Queue()
        self.cancel = threading.Event()
        self.gone = threading.Event()
        fd = os.dup(0)
        null = os.open(os.devnull, os.O_RDONLY)
        os.dup2(null, 0)
        os.close(null)
        sys.stdin = open(os.devnull, encoding='utf-8')
        self.input = os.fdopen(fd, 'r', encoding='utf-8', newline='\n')
        threading.Thread(target=self._read, name='commands', daemon=True).start()

    def _read(self):
        for line in self.input:
            try:
                message = json.loads(line)
            except ValueError:
                continue
            kind = message.get('type')
            if kind == 'cancel':
                self.cancel.set()
            elif kind in ('job', 'shutdown'):
                self.jobs.put(message)
        self.gone.set()
        self.cancel.set()
        self.jobs.put({'type': 'shutdown'})


def window_times(source, proc0, proc1, frames, fps):
    """(start, seconds) for pipeline.run so that it processes frames [proc0, proc1): the decoder's own timestamps of the
    two frames (frame-exact seeking), the end of the video for the last segment."""
    import cv2
    capture = cv2.VideoCapture(str(source))
    try:
        def stamp(index):
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, _ = capture.read()
            if not ok:
                return None
            t = float(capture.get(cv2.CAP_PROP_POS_MSEC))/1000
            return t if t > 0 or not index else index/fps
        start = stamp(proc0)
        if start is None:
            raise RuntimeError(f'Frame {proc0} of the video could not be read.')
        end = stamp(proc1) if proc1 < frames else None
        seconds = (end-start) if end is not None else (frames-proc0)/fps+1.0
        return start, max(seconds, 1.0/fps)
    finally:
        capture.release()


def summarize(report):
    s = report.get('summary') or {}
    return {'pitchConfidence': (s.get('pitch') or {}).get('meanConfidence'), 'identifiedShare': s.get('identifiedShare'),
            'ballTracked': (s.get('ball') or {}).get('trackedShare'), 'smallSceneShare': (s.get('profile') or {}).get('smallSceneShare'),
            'autoLandmarks': (s.get('profile') or {}).get('autoLandmarks'), 'processingFPS': s.get('processingFPS'),
            'identities': s.get('globalIdentities'), 'timing': s.get('timing')}


def place(partial, final):
    """Rename the finished folder into place (an older artifact of the same segment is moved aside first, then deleted)."""
    final = Path(final)
    old = None
    if final.exists():
        old = final.with_name(f'{final.name}.old-{uuid.uuid4().hex[:6]}')
        os.replace(final, old)
    for attempt in range(20):
        try:
            os.replace(partial, final)
            break
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(.1*(attempt+1))
    if old is not None:
        shutil.rmtree(old, ignore_errors=True)


def run_job(job, resources, channel, commands, process):
    import torch
    import pipeline
    from fullmatch import manifest
    key = job['key']
    partial = Path(job['partial'])
    shutil.rmtree(partial, ignore_errors=True)
    commands.cancel.clear()
    peak = {'rss': process.memory_info().rss}
    started = time.perf_counter()
    last = {'t': 0.0}

    def progress(values):
        peak['rss'] = max(peak['rss'], process.memory_info().rss)
        # The offline passes report a stage without a frame count: the frames tracked so far still count.
        if values.get('frames') is not None:
            last['frames'] = values['frames']
        t = time.perf_counter()
        if t-last['t'] >= 1.0 or 'summary' in values or values.get('frames') is None:
            last['t'] = t
            channel.send({'type': 'progress', 'job': key, 'stage': values.get('stage'), 'progress': values.get('progress'),
                          'videoTime': values.get('videoTime'), 'frames': last.get('frames'),
                          'processingFPS': values.get('processingFPS'), 'rssGB': round(peak['rss']/2**30, 3)})

    try:
        start, seconds = window_times(job['source'], job['proc0'], job['proc1'], job['frames'], job['fps'])
        config = dict(job['config']) | {'start': start, 'seconds': seconds}
        report = pipeline.run(Path(job['source']), partial, config, progress, commands.cancel.is_set, job.get('previous'), resources=resources)
        if report.get('status') != 'complete':
            raise InterruptedError('cancelled')
        packing = time.perf_counter()
        stored, raw_bytes = manifest.compress_result(partial)
        facts = manifest.sidecar(report, stored) | {'matchId': job.get('matchId'), 'segment': job['index'],
                                                    'start': start, 'seconds': seconds, 'resultRawBytes': raw_bytes,
                                                    'compressSeconds': round(time.perf_counter()-packing, 2),
                                                    'elapsed': round(time.perf_counter()-started, 2)}
        manifest.write_json(partial/manifest.SIDECAR, facts)
        summary = summarize(report)
        del report
        place(partial, job['folder'])
        channel.send({'type': 'done', 'job': key, 'sidecar': facts, 'summary': summary,
                      'elapsed': round(time.perf_counter()-started, 2), 'peakRssGB': round(peak['rss']/2**30, 3),
                      'peakVramGB': round(torch.cuda.max_memory_allocated()/2**30, 3) if torch.cuda.is_available() else None})
    except BaseException as error:  # noqa: BLE001 - every failure is reported; the process lives on
        oom = isinstance(error, torch.cuda.OutOfMemoryError) or 'out of memory' in str(error).lower()
        kind = 'cancelled' if isinstance(error, InterruptedError) or commands.cancel.is_set() else 'oom' if oom else 'error'
        shutil.rmtree(partial, ignore_errors=True)
        channel.send({'type': 'failed', 'job': key, 'kind': kind, 'error': f'{type(error).__name__}: {error}',
                      'traceback': traceback.format_exc()[-4000:], 'elapsed': round(time.perf_counter()-started, 2)})
        if isinstance(error, (KeyboardInterrupt, SystemExit)):
            raise
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def run_discover(job, channel, commands):
    """One chunk of the calibration discovery: calibration.solve_chunk on the CPU, the rows and pairs written to
    job['out'] (atomically). No model is used: light workers run these too."""
    from fullmatch import calibration, manifest
    from soccer import field
    key = job['key']
    commands.cancel.clear()
    started = time.perf_counter()
    try:
        size = field.PitchSize.from_json(job['size']) if job.get('size') else None
        out = calibration.solve_chunk(job['source'], job['indices'], int(job['width']), int(job['height']), size=size,
                                      cancelled=commands.cancel.is_set)
        if commands.cancel.is_set():
            raise InterruptedError('cancelled')
        out = dict(out) | {'chunk': job['chunk'], 'indices': list(job['indices']), 'version': calibration.VERSION,
                           'seconds': round(time.perf_counter()-started, 2)}
        manifest.write_json(Path(job['out']), out)
        channel.send({'type': 'done', 'job': key, 'kind': 'discover', 'chunk': job['chunk'], 'frames': len(out.get('rows') or []),
                      'anchors': sum(1 for r in out.get('rows') or [] if r.get('ok')), 'elapsed': round(time.perf_counter()-started, 2)})
    except BaseException as error:  # noqa: BLE001 - reported; the process lives on
        kind = 'cancelled' if isinstance(error, InterruptedError) or commands.cancel.is_set() else 'error'
        channel.send({'type': 'failed', 'job': key, 'kind': kind, 'jobKind': 'discover', 'chunk': job.get('chunk'),
                      'error': f'{type(error).__name__}: {error}', 'traceback': traceback.format_exc()[-4000:],
                      'elapsed': round(time.perf_counter()-started, 2)})
        if isinstance(error, (KeyboardInterrupt, SystemExit)):
            raise
    finally:
        gc.collect()


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--threads', type=int, default=0, help='CPU threads for OpenCV and torch (0: library defaults)')
    parser.add_argument('--name', default='worker')
    parser.add_argument('--light', action='store_true', help='no model: calibration discovery jobs only')
    args = parser.parse_args(argv)
    if os.environ.get('FULLMATCH_STACKS'):
        # Diagnostics: the stacks of every thread to the worker log after that many seconds (and again every time).
        import faulthandler
        faulthandler.dump_traceback_later(float(os.environ['FULLMATCH_STACKS']), repeat=True, file=sys.stderr)
    channel = Channel()
    commands = Commands()
    sys.path.insert(0, str(ROOT))
    os.chdir(ROOT)
    loaded = time.perf_counter()
    if args.light:
        return light_main(args, channel, commands, loaded)
    try:
        import psutil
        import cv2
        import torch
        if args.threads > 0:
            cv2.setNumThreads(args.threads)
            torch.set_num_threads(args.threads)
        import pipeline
        from download_model import get_model
        gpu = pipeline.gpu_info()
        resources = pipeline.Resources().load(get_model())
    except BaseException as error:  # noqa: BLE001 - the scheduler sees why this worker never became ready
        channel.send({'type': 'failed', 'job': None, 'kind': 'startup', 'error': f'{type(error).__name__}: {error}',
                      'traceback': traceback.format_exc()[-4000:]})
        return 2
    process = psutil.Process()
    channel.send({'type': 'ready', 'pid': os.getpid(), 'name': args.name, 'gpu': gpu, 'threads': args.threads,
                  'loadSeconds': round(time.perf_counter()-loaded, 2), 'rssGB': round(process.memory_info().rss/2**30, 3),
                  'env': {k: os.environ.get(k) for k in THREAD_VARIABLES if os.environ.get(k)}})
    while True:
        message = commands.jobs.get()
        if message.get('type') == 'shutdown' or commands.gone.is_set():
            break
        if message['job'].get('kind') == 'discover':
            run_discover(message['job'], channel, commands)
        else:
            run_job(message['job'], resources, channel, commands, process)
        if commands.gone.is_set():
            break
    channel.send({'type': 'bye', 'pid': os.getpid()})
    return 0


def light_main(args, channel, commands, loaded):
    """A light worker: no torch, no model; discovery jobs only (a segment job is refused)."""
    try:
        import psutil
        import cv2
        if args.threads > 0:
            cv2.setNumThreads(args.threads)
        from fullmatch import calibration   # noqa: F401
    except BaseException as error:  # noqa: BLE001
        channel.send({'type': 'failed', 'job': None, 'kind': 'startup', 'error': f'{type(error).__name__}: {error}',
                      'traceback': traceback.format_exc()[-4000:]})
        return 2
    process = psutil.Process()
    channel.send({'type': 'ready', 'pid': os.getpid(), 'name': args.name, 'light': True, 'threads': args.threads,
                  'loadSeconds': round(time.perf_counter()-loaded, 2), 'rssGB': round(process.memory_info().rss/2**30, 3)})
    while True:
        message = commands.jobs.get()
        if message.get('type') == 'shutdown' or commands.gone.is_set():
            break
        job = message['job']
        if job.get('kind') == 'discover':
            run_discover(job, channel, commands)
        else:
            channel.send({'type': 'failed', 'job': job.get('key'), 'kind': 'error', 'error': 'a light worker runs no segment jobs'})
        if commands.gone.is_set():
            break
    channel.send({'type': 'bye', 'pid': os.getpid()})
    return 0


if __name__ == '__main__':
    sys.exit(main())
