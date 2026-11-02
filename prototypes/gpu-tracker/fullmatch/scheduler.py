"""The Full Match Mode scheduler: runs a match's segment jobs on a bounded pool of local worker processes (worker.py),
checkpoints every finished segment in the manifest, retries failures, and assembles the match at the end.

The worker count is decided before processing (system.auto_workers or the user's choice within system.allowed) and
held: workers start a few seconds apart (model loading at once spikes memory and the disk), each takes the next pending
segment in match order when it is idle. Every ~5 s the scheduler samples free memory, GPU memory and utilization, CPU
and the workers' memory. When starting another segment would be unsafe (system.unsafe) no new job starts until it
recovers; when it stays unsafe for UNSAFE_HOLD_S with several workers busy, one worker is retired after its current
job (no rapid adding and removing on short spikes). A GPU out-of-memory failure also retires one worker (never below
one) and puts the segment back in the queue.

Failures: a failed job is retried up to MAX_ATTEMPTS times (then the segment stays failed and the match stops as
'failed', its finished segments kept; resume retries the failed ones); a worker that dies mid-job fails that job and
is restarted (at most MAX_RESTARTS times per match). A finished job only counts once its artifact verifies
(manifest.verify_artifact, with the checksum); the manifest is saved after every change.

Controls: pause (no new jobs; the running ones finish and checkpoint; then the workers exit to free the GPU), stop
(running jobs are cancelled and go back to pending; finished segments are kept), resume (reconcile the manifest with
the artifacts on disk, then continue with the pending and failed segments only). Workers are children of the server:
they exit when it does (worker.py watches stdin), and stop()/close() wait for them and kill any that hang.

Calibration discovery (calibration.py) runs first when the manifest asks for it ('discovery' state pending or running):
the sampled frames of the match, in chunks, go to every ready worker and to a few light workers (worker.py --light, no
model) that start at once, up to DISCOVERY_STREAMS processes in all; the light ones leave when it ends. Each chunk's
result is a file under calibration/, so a stopped discovery resumes with the missing chunks only. calibration.discover()
then writes the match calibration index (calibration/discovery.json): the camera, the anchors that passed the rules and
the synthetic keyframes, which each segment job receives with the user's keyframes. A failed or refused discovery never
fails the match: the segments then calibrate as they would without it.

Progress covers the whole workflow without resetting: discovery is DISCOVERY_SHARE of the bar when it runs, tracking up
to TRACKING_SHARE (frames processed, the running jobs' partial frames included), assembly the rest. The ETA comes from
measured throughput: the frames finished over the last minutes (a sliding window), with a range that narrows as more of
the match is done; before anything is measured it comes from the benchmark table (system.estimate_seconds).
"""
import json
import os
import queue
import subprocess
import sys
import threading
import time
import traceback
import uuid
from collections import deque
from pathlib import Path

import psutil

from . import manifest as mf
from . import system

ROOT = Path(__file__).resolve().parents[1]
MAX_ATTEMPTS = 3
MAX_RESTARTS = 8
READY_TIMEOUT_S = 600.0     # a worker that has not loaded its models by then is killed and replaced
STAGGER_S = 3.0
RESOURCE_EVERY_S = 5.0
UNSAFE_HOLD_S = 30.0
STOP_WAIT_S = 90.0          # a cancelled job finishes its frame and the offline passes on the partial data, then exits
TRACKING_SHARE = .95
DISCOVERY_SHARE = .03       # of the bar, inside TRACKING_SHARE, when the calibration discovery runs
DISCOVERY_STREAMS = 8       # processes solving discovery chunks at once (GPU workers + light ones), at most cpus // 2
LIGHT_STAGGER_S = .5
RATE_WINDOW_S = 600.0       # throughput over the last ten minutes
ASSEMBLY_GUESS_S = (30.0, 4.0)   # assembly before it has been measured: seconds fixed + per segment


def default_command(threads=0, name='worker'):
    """The worker process: this interpreter, the prototype folder, isolated from the user's site (-B: no .pyc)."""
    return [sys.executable, '-B', '-m', 'fullmatch.worker', '--threads', str(int(threads)), '--name', name]


def light_command(name='light'):
    """A light worker: discovery jobs only, OpenCV and BLAS single-threaded (several run at once, one CPU each)."""
    return default_command(1, name)+['--light']


def thread_environment(threads):
    """The library thread pools of a worker capped at `threads` (0: the libraries' defaults, all logical CPUs)."""
    env = dict(os.environ)
    env.pop('PYTHONSTARTUP', None)
    if threads > 0:
        for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
            env[key] = str(int(threads))
    return env


class Worker:
    """One worker process and the thread reading its messages into the scheduler's event queue."""

    def __init__(self, number, command, env, log_path, events, cwd=ROOT, light=False):
        self.number = number
        self.light = light           # no model: discovery jobs only
        self.events = events
        self.log = open(log_path, 'a', encoding='utf-8', errors='replace')
        self.log.write(f'\n=== worker {number} started {mf.now()} ===\n')
        self.log.flush()
        flags = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
        self.process = subprocess.Popen(command, cwd=str(cwd), env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        stderr=self.log, text=True, encoding='utf-8', bufsize=1, creationflags=flags)
        self.state = 'starting'      # starting, idle, busy, retiring, gone
        self.started = time.time()
        self.info = {}
        self.job = None              # segment index (or discovery chunk) of the running job
        self.job_kind = None         # 'segment' or 'discover'
        self.job_started = None
        self.progress = {}
        self.retire = False
        threading.Thread(target=self._read, name=f'match-worker-{number}', daemon=True).start()

    @property
    def pid(self):
        return self.process.pid

    def _read(self):
        for line in self.process.stdout:
            try:
                message = json.loads(line)
            except ValueError:
                continue
            self.events.put((self, message))
        code = self.process.wait()
        self.events.put((self, {'type': 'exit', 'code': code}))

    def send(self, message):
        try:
            self.process.stdin.write(json.dumps(message)+'\n')
            self.process.stdin.flush()
            return True
        except (OSError, ValueError):
            return False

    def shutdown(self):
        self.send({'type': 'shutdown'})
        try:
            self.process.stdin.close()
        except OSError:
            pass

    def kill(self):
        try:
            self.process.kill()
        except OSError:
            pass

    def alive(self):
        return self.process.poll() is None

    def close_log(self):
        try:
            self.log.close()
        except OSError:
            pass


class MatchScheduler:
    """Runs one match (a manifest) to completion, a pause or a stop. start() returns at once; status() is cheap and
    thread-safe."""

    def __init__(self, manifest, workers, threads=0, command=None, assemble=None, before_start=None, after_stop=None,
                 seeds=None, measured=None, monitor_gpu=True, cwd=ROOT, light=None, streams=None):
        self.manifest = manifest
        self.target = max(1, int(workers))
        self.requested = self.target
        self.threads = int(threads)
        self.command = command or (lambda number: default_command(self.threads, f'worker-{number}'))
        self.light = light or (lambda number: light_command(f'light-{number}'))
        self.streams = int(streams) if streams is not None else min(DISCOVERY_STREAMS, max(1, (os.cpu_count() or 2)//2))
        self.assemble = assemble
        self.before_start, self.after_stop = before_start, after_stop
        self.seeds = seeds
        self.measured = measured or {}
        self.monitor_gpu = monitor_gpu
        self.cwd = cwd
        self.lock = threading.RLock()
        self.events = queue.Queue()
        self.workers = []
        self.queue = deque()
        self.mode = 'idle'           # idle, running, pausing, stopping, assembling, finished
        self.thread = None
        self.restarts = 0
        self.next_number = 1
        self.started_at = None
        self.resources = {}
        self.peaks = {'ramUsedGB': 0.0, 'workersRssGB': 0.0, 'gpuUsedGB': 0.0, 'gpuUtilization': 0.0, 'cpuPercent': 0.0}
        self.unsafe_since = None
        self.held = None
        self.samples = deque()       # (wall time, frames done including the running jobs' partial frames)
        self.assembly = {'state': 'pending', 'progress': 0.0, 'stage': None}
        self.errors = deque(maxlen=50)
        self.last_resource = 0.0
        self.last_start = 0.0
        self.last_light = 0.0
        self.light_failures = 0
        self.fps = float(manifest.data['video']['fps'])
        self.total_frames = sum(s.proc1-s.proc0 for s in manifest.segments())
        self.phase = 'tracking'      # 'discovery' while the calibration discovery runs first
        self.chunks = deque()        # discovery chunks still to solve
        self.calibration_index = None   # calibration/discovery.json once discovery ended (None: no synthetic keyframes)
        self.discovery_started = None
        self.tracking_started = None
        (manifest.folder/'logs').mkdir(parents=True, exist_ok=True)
        (manifest.folder/'segments').mkdir(parents=True, exist_ok=True)

    # ---- controls ---------------------------------------------------------------------------------------------
    def start(self, resume=False):
        with self.lock:
            if self.thread is not None and self.thread.is_alive():
                raise RuntimeError('This match is already being processed.')
            self.manifest.remove_partials()
            changed = self.manifest.reconcile(deep=resume)
            if changed:
                self.manifest.note('reconciled', segments=changed)
            if resume:
                for entry in self.manifest.data['segments']:
                    if entry['state'] == 'failed':
                        entry.update(state='pending', attempts=0)
                self.manifest.save()
            self.queue = deque(e['index'] for e in self.manifest.data['segments'] if e['state'] in ('pending', 'failed'))
            self.mode = 'running'
            self.started_at = time.time()
            self.samples.clear()
            self.held = None
            self.manifest.set_state('running', workers=self.manifest.data['workers'] | {'count': self.target},
                                    runtime=self.manifest.data.get('runtime') or {})
            self._prepare_discovery()
            self.tracking_started = None if self.phase == 'discovery' else self.started_at
            self._log(f"{'resume' if resume else 'start'}: {len(self.queue)} segments to process, {self.target} workers"
                      + (f'; calibration discovery first: {len(self.chunks)} chunks on up to {self.streams} processes' if self.phase == 'discovery' else ''))
            self.thread = threading.Thread(target=self._loop, name='match-scheduler', daemon=True)
            self.thread.start()

    def pause(self):
        with self.lock:
            if self.mode == 'running':
                self.mode = 'pausing'
                self._log('pause requested: no new segments; running ones finish')

    def stop(self):
        with self.lock:
            if self.mode in ('running', 'pausing'):
                self.mode = 'stopping'
                self._log('stop requested: cancelling running segments')
                for w in self.workers:
                    if w.state == 'busy':
                        w.send({'type': 'cancel'})

    def wait(self, timeout=None):
        if self.thread is not None:
            self.thread.join(timeout)
        return self.thread is None or not self.thread.is_alive()

    def close(self):
        """Stop and wait (the server is shutting down)."""
        self.stop()
        self.wait(STOP_WAIT_S+10)
        for w in list(self.workers):
            w.kill()

    # ---- the loop -----------------------------------------------------------------------------------------------
    def _loop(self):
        try:
            if self.before_start is not None:
                self.before_start()
            self._spawn_missing()
            while True:
                try:
                    worker, message = self.events.get(timeout=1.0)
                    self._handle(worker, message)
                except queue.Empty:
                    pass
                now = time.time()
                if now-self.last_resource >= RESOURCE_EVERY_S:
                    self._check_resources(now)
                self._check_starting(now)
                with self.lock:
                    mode = self.mode
                    busy = [w for w in self.workers if w.state == 'busy']
                if mode == 'running':
                    self._spawn_missing()
                    with self.lock:
                        stranded = self.restarts > MAX_RESTARTS and not any(w.state != 'gone' for w in self.workers)
                    if stranded:
                        self._end('failed', 'the worker processes could not start or kept failing (see logs/worker-*.log)')
                        break
                    self._dispatch()
                    if self.phase == 'discovery':
                        with self.lock:
                            solving = any(w.state == 'busy' and w.job_kind == 'discover' for w in self.workers)
                        if not self.chunks and not solving:
                            self._finish_discovery()
                        continue
                    with self.lock:
                        idle = not self.queue and not any(w.state == 'busy' for w in self.workers)
                    if idle and self._all_done():
                        self._finish_tracking()
                        break
                    if idle:
                        # Nothing queued, nothing running, and not every segment done: the rest failed MAX_ATTEMPTS times.
                        self._end('failed', 'some segments failed repeatedly; resume retries them')
                        break
                elif mode in ('pausing', 'stopping') and not busy:
                    self._end('paused' if mode == 'pausing' else 'stopped', None)
                    break
        except Exception as error:   # noqa: BLE001 - the match must not die silently
            self._log(f'scheduler error: {type(error).__name__}: {error}')
            self.errors.append({'at': mf.now(), 'error': f'scheduler: {error}'})
            self._end('failed', f'scheduler error: {error}')

    def _all_done(self):
        return all(e['state'] == 'done' for e in self.manifest.data['segments'])

    def _spawn_missing(self):
        with self.lock:
            if self.phase == 'discovery':
                self._spawn_light()
            live = [w for w in self.workers if w.state != 'gone' and not w.retire and not w.light]
            if len(live) >= self.target or self.mode != 'running' or self.held:
                return
            if time.time()-self.last_start < STAGGER_S:
                return
            if self.restarts > MAX_RESTARTS:
                return
            number = self.next_number
            self.next_number += 1
            log = self.manifest.folder/'logs'/f'worker-{number}.log'
            command = self.command(number) if callable(self.command) else list(self.command)
            worker = Worker(number, command, thread_environment(self.threads), log, self.events, cwd=self.cwd)
            self.workers.append(worker)
            self.last_start = time.time()
            self._log(f'worker {number} started (pid {worker.pid})')

    def _spawn_light(self):
        """Light workers while discovery runs: enough that the GPU workers and they make self.streams processes, never
        more than chunks are left, a few failed starts at most."""
        want = min(max(0, self.streams-self.target), len(self.chunks))
        light = [w for w in self.workers if w.light and w.state != 'gone' and not w.retire]
        if len(light) >= want or self.mode != 'running' or self.held or self.light_failures > 2*self.streams:
            return
        if time.time()-self.last_light < LIGHT_STAGGER_S:
            return
        number = self.next_number
        self.next_number += 1
        log = self.manifest.folder/'logs'/f'light-{number}.log'
        command = self.light(number) if callable(self.light) else list(self.light)
        worker = Worker(number, command, thread_environment(1), log, self.events, cwd=self.cwd, light=True)
        self.workers.append(worker)
        self.last_light = time.time()
        self._log(f'light worker {number} started (pid {worker.pid})')

    def _check_starting(self, now):
        with self.lock:
            for w in self.workers:
                if w.state == 'starting' and now-w.started > READY_TIMEOUT_S:
                    self._log(f'worker {w.number} did not become ready in {READY_TIMEOUT_S:.0f} s: killed')
                    w.kill()

    def _dispatch(self):
        with self.lock:
            if self.held or self.mode != 'running':
                return
            if self.phase == 'discovery':
                self._dispatch_chunks()
                return
            for w in self.workers:
                if not self.queue:
                    return
                if w.state != 'idle' or w.retire or w.light:
                    continue
                index = self.queue.popleft()
                job = self._job(index)
                if not w.send({'type': 'job', 'job': job}):
                    self.queue.appendleft(index)
                    continue
                w.state, w.job, w.job_kind, w.job_started, w.progress = 'busy', index, 'segment', time.time(), {}
                entry = self.manifest.entry(index)
                self.manifest.update_segment(index, state='running', attempts=entry['attempts']+1, worker=w.number,
                                             startedAt=mf.now(), error=None)
                self._log(f"segment {index} -> worker {w.number} (attempt {entry['attempts']})")

    def _job(self, index):
        entry = self.manifest.entry(index)
        folder = self.manifest.folder/'segments'/entry['key']
        partial = folder.with_name(f"{entry['key']}.partial-{uuid.uuid4().hex[:6]}")
        video = self.manifest.data['video']
        return {'key': entry['key'], 'index': index, 'matchId': self.manifest.data['matchId'],
                'source': self.manifest.data['source']['path'], 'frames': int(video['frames']), 'fps': float(video['fps']),
                'proc0': entry['proc0'], 'proc1': entry['proc1'], 'own0': entry['own0'], 'own1': entry['own1'],
                'config': segment_config(self.manifest.data['settings'], self.synthetic_keyframes(index)), 'previous': self.seeds,
                'folder': str(folder), 'partial': str(partial)}

    # ---- calibration discovery ----------------------------------------------------------------------------------
    def _chunk_path(self, k):
        return self.manifest.folder/'calibration'/'chunks'/f'c{int(k):03d}.json'

    def _index_path(self):
        return self.manifest.folder/'calibration'/'discovery.json'

    def _period_frames(self):
        """Each period's processed frames [first, last) as the planner laid the segments out."""
        out = {}
        for s in self.manifest.segments():
            a, b = out.get(s.period, (s.proc0, s.proc1))
            out[s.period] = (min(a, s.proc0), max(b, s.proc1))
        return [out[p] for p in sorted(out)]

    def _discovery_size(self):
        """The pitch size discovery solves with: the one the user's keyframes give (as every segment learns it), else the
        standard one."""
        from soccer import calibration as pitch_calibration, field
        video = self.manifest.data['video']
        manual = [c for c in self.manifest.data['settings'].get('calibrations') or [] if c.get('points')]
        return (pitch_calibration.learn_size(manual, int(video['width']), int(video['height'])) if manual else None) or field.PitchSize()

    def _prepare_discovery(self):
        """At start: run the discovery first (its state is pending or running) with the chunks not solved yet, or use its
        index when it already ended; nothing when the match has it off."""
        self.phase, self.chunks = 'tracking', deque()
        d = self.manifest.data.get('discovery')
        if not d or d.get('state') in (None, 'off'):
            return
        if d.get('state') in ('complete', 'refused', 'failed'):
            self.calibration_index = self._load_index()
            return
        if not d.get('chunks'):
            try:
                from . import calibration
                chunks = []
                for period in self._period_frames():
                    chunks += calibration.chunks(calibration.sample([period], self.fps), periods=[period])
                size = self._discovery_size().to_json()
                d.update(chunks=[{'index': k, 'indices': [int(i) for i in c], 'state': 'pending', 'attempts': 0} for k, c in enumerate(chunks)],
                         sampled=len({i for c in chunks for i in c}), size=size, version=calibration.VERSION)
            except Exception as error:   # noqa: BLE001 - discovery is an aid: the match goes on without it
                self._log(f'calibration discovery could not be planned: {type(error).__name__}: {error}')
                self.errors.append({'at': mf.now(), 'kind': 'discovery', 'error': f'calibration discovery could not be planned: {error}'})
                d.update(state='failed', summary={'error': f'{type(error).__name__}: {error}'}, finishedAt=mf.now())
                self.manifest.save()
                return
        for c in d['chunks']:
            if c['state'] == 'done' and not self._chunk_path(c['index']).exists():
                c['state'] = 'pending'
            elif c['state'] == 'running':
                c['state'] = 'pending'
        d['state'] = 'running'
        d.setdefault('startedAt', mf.now())
        self.manifest.save()
        self.chunks = deque(c['index'] for c in d['chunks'] if c['state'] == 'pending')
        self.phase = 'discovery'
        self.discovery_started = time.time()

    def _load_index(self):
        try:
            return json.loads(self._index_path().read_text(encoding='utf-8'))
        except (OSError, ValueError):
            return None

    def _dispatch_chunks(self):
        d = self.manifest.data['discovery']
        video = self.manifest.data['video']
        for w in self.workers:
            if not self.chunks:
                return
            if w.state != 'idle' or w.retire:
                continue
            k = self.chunks.popleft()
            entry = d['chunks'][k]
            job = {'kind': 'discover', 'key': f'c{k:03d}', 'chunk': k, 'indices': entry['indices'], 'source': self.manifest.data['source']['path'],
                   'width': int(video['width']), 'height': int(video['height']), 'size': d.get('size'), 'out': str(self._chunk_path(k))}
            self._chunk_path(k).parent.mkdir(parents=True, exist_ok=True)
            if not w.send({'type': 'job', 'job': job}):
                self.chunks.appendleft(k)
                continue
            w.state, w.job, w.job_kind, w.job_started, w.progress = 'busy', k, 'discover', time.time(), {}
            entry.update(state='running', attempts=entry['attempts']+1, worker=w.number)
        self.manifest.save()

    def _chunk_done(self, worker, message):
        k = worker.job
        worker.state, worker.job, worker.job_kind, worker.progress = 'idle', None, None, {}
        d = self.manifest.data['discovery']
        entry = d['chunks'][k]
        try:
            data = json.loads(self._chunk_path(k).read_text(encoding='utf-8'))
            ok = isinstance(data.get('rows'), list)
        except (OSError, ValueError):
            ok = False
        if ok:
            entry.update(state='done', frames=message.get('frames'), anchors=message.get('anchors'), seconds=message.get('elapsed'))
            self.manifest.save()
        else:
            self._chunk_retry(k, 'the chunk file is missing or unreadable')

    def _chunk_failed(self, worker, message):
        k = worker.job
        worker.state, worker.job, worker.job_kind, worker.progress = 'idle', None, None, {}
        if k is None:
            return
        if message.get('kind') == 'cancelled' or self.mode == 'stopping':
            self.manifest.data['discovery']['chunks'][k]['state'] = 'pending'
            if self.mode == 'running':
                self.chunks.appendleft(k)
            self.manifest.save()
            return
        self._chunk_retry(k, message.get('error') or 'failed')

    def _chunk_retry(self, k, error):
        entry = self.manifest.data['discovery']['chunks'][k]
        self.errors.append({'at': mf.now(), 'chunk': k, 'kind': 'discovery', 'error': error})
        if entry['attempts'] < MAX_ATTEMPTS:
            entry.update(state='pending', error=error)
            self.chunks.append(k)
            self._log(f'discovery chunk {k} failed: {error} - will retry')
        else:
            entry.update(state='failed', error=error)
            self._log(f"discovery chunk {k} failed {entry['attempts']} times: {error} - its frames are left out")
        self.manifest.save()

    def _finish_discovery(self):
        """Every chunk solved (or given up): the match calibration index from all of them, the light workers released,
        tracking next. A failure here leaves the segments to calibrate as they would without discovery."""
        from . import calibration
        d = self.manifest.data['discovery']
        video = self.manifest.data['video']
        rows, pairs, seen_rows, seen_pairs = [], [], set(), set()
        try:
            for c in d['chunks']:
                if c['state'] != 'done':
                    continue
                data = json.loads(self._chunk_path(c['index']).read_text(encoding='utf-8'))
                for r in data.get('rows') or []:
                    if r.get('index') not in seen_rows:
                        seen_rows.add(r.get('index'))
                        rows.append(r)
                for p in data.get('pairs') or []:
                    key = (p.get('a'), p.get('b'))
                    if key not in seen_pairs:
                        seen_pairs.add(key)
                        pairs.append(p)
            from soccer import field
            index = calibration.discover(rows, pairs, self.manifest.segments(), self.fps, int(video['width']), int(video['height']),
                                         size=field.PitchSize.from_json(d.get('size')), manual=self.manifest.data['settings'].get('calibrations') or [])
            mf.write_json(self._index_path(), index)
            state = 'refused' if index.get('refused') else 'complete'
            summary = {'sampled': index.get('sampled'), 'anchors': index.get('anchors'), 'kept': index.get('kept'),
                       'camera': index.get('camera'), 'refused': index.get('refused'), 'keyframes': len(index.get('keyframes') or [])}
        except Exception as error:   # noqa: BLE001 - discovery is an aid: the match goes on without it
            self._log(f'calibration discovery failed: {type(error).__name__}: {error}\n{traceback.format_exc()}')
            self.errors.append({'at': mf.now(), 'kind': 'discovery', 'error': f'calibration discovery failed: {error}'})
            index, state, summary = None, 'failed', {'error': f'{type(error).__name__}: {error}'}
        with self.lock:
            spent = round(time.time()-self.discovery_started, 1) if self.discovery_started else None
            d.update(state=state, summary=summary, finishedAt=mf.now(), seconds=spent)
            self.calibration_index = index
            self._calibration_status()
            self.manifest.save()
            self.phase = 'tracking'
            self.tracking_started = time.time()
            self.samples.clear()
            for w in self.workers:
                if w.light and w.state != 'gone':
                    w.retire = True
                    w.state = 'retiring' if w.state == 'idle' else w.state
                    w.shutdown()
        self._log(f"calibration discovery {state} in {spent} s: " + ', '.join(f'{k} {v}' for k, v in summary.items() if k != 'camera'))

    def synthetic_keyframes(self, index):
        """The discovered keyframes of segment `index` (none when discovery is off, refused or failed)."""
        if not self.calibration_index or self.calibration_index.get('refused'):
            return []
        from . import calibration
        return list(calibration.keyframes_for(self.calibration_index, self.manifest.segments()[index]))

    def _calibration_status(self):
        """The manifest's calibrationStatus after discovery: per segment the user's keyframes in reach and the discovered
        ones; a segment with neither relies on the automatic per-segment route."""
        status = dict(self.manifest.data.get('calibrationStatus') or {})
        anchored = set(status.get('anchoredSegments') or [])
        discovered = {}
        for s in self.manifest.segments():
            n = len(self.synthetic_keyframes(s.index))
            if n:
                discovered[s.index] = n
        both = anchored | set(discovered)
        total = len(self.manifest.data['segments'])
        status.update(discovered=sum(discovered.values()), discoveredSegments=sorted(discovered), covered=len(both),
                      uncovered=sorted(set(range(total))-both), automatic=total-len(both))
        d = self.manifest.data.get('discovery') or {}
        if d.get('state') == 'refused':
            status['note'] = f"Automatic calibration found no safe anchors: {(d.get('summary') or {}).get('refused')}. {total-len(both)} of {total} segments rely on the per-segment calibration; add keyframes in the stretches that matter."
        elif d.get('state') == 'failed':
            status['note'] = f'Automatic calibration failed ({(d.get("summary") or {}).get("error")}); the segments calibrate on their own.'
        else:
            status['note'] = (f'{sum(discovered.values())} automatic keyframes in {len(discovered)} segments; {len(both)} of {total} segments are anchored.'
                              + (f' {total-len(both)} rely on the per-segment calibration.' if total > len(both) else ''))
        self.manifest.data['calibrationStatus'] = status

    def _handle(self, worker, message):
        kind = message.get('type')
        with self.lock:
            if kind == 'ready':
                worker.state, worker.info = 'idle', message
                self._log(f"worker {worker.number} ready in {message.get('loadSeconds')} s")
            elif kind == 'progress':
                # Merged, not replaced: a stage message without a frame count keeps the frames tracked so far.
                worker.progress = {**worker.progress, **{k: v for k, v in message.items() if v is not None}}
            elif kind == 'done':
                self._done(worker, message)
            elif kind == 'failed':
                self._failed(worker, message)
            elif kind == 'exit':
                self._exited(worker, message.get('code'))
            elif kind == 'bye':
                worker.state = 'gone' if worker.state != 'busy' else worker.state

    def _done(self, worker, message):
        if worker.job_kind == 'discover':
            self._chunk_done(worker, message)
            return
        index = worker.job
        worker.state, worker.job, worker.job_kind, worker.progress = 'idle', None, None, {}
        if index is None:
            return
        segment = self.manifest.segments()[index]
        folder = self.manifest.artifact_folder(index)
        problem = mf.verify_artifact(folder, segment, deep=True)
        if problem:
            self._retry(index, f'the finished artifact does not verify: {problem}', 'error')
        else:
            sidecar = message.get('sidecar') or {}
            self.manifest.update_segment(index, state='done', error=None, artifact=str(folder.relative_to(self.manifest.folder)),
                                         finishedAt=mf.now(), elapsed=message.get('elapsed'), peakRssGB=message.get('peakRssGB'),
                                         peakVramGB=message.get('peakVramGB'), summary=message.get('summary'),
                                         frames=sidecar.get('frames'), first=sidecar.get('first'), last=sidecar.get('last'))
            self._log(f"segment {index} done by worker {worker.number} in {message.get('elapsed')} s")
        if worker.retire:
            worker.state = 'retiring'
            worker.shutdown()

    def _failed(self, worker, message):
        index = worker.job
        kind = message.get('kind')
        if kind == 'startup':
            self._log(f"{'light ' if worker.light else ''}worker {worker.number} could not start: {message.get('error')}")
            self.errors.append({'at': mf.now(), 'worker': worker.number, 'error': message.get('error')})
            worker.state = 'gone'
            if worker.light:
                self.light_failures += 1
            else:
                self.restarts += 1
            return
        if worker.job_kind == 'discover':
            self._chunk_failed(worker, message)
            return
        worker.state, worker.job, worker.job_kind, worker.progress = 'idle', None, None, {}
        if index is None:
            return
        if kind == 'cancelled' or self.mode == 'stopping':
            self.manifest.update_segment(index, state='pending', error='cancelled')
            if self.mode == 'running':
                self.queue.appendleft(index)
            return
        if kind == 'oom' and self.target > 1:
            self.target -= 1
            worker.retire = True
            self._log(f'GPU out of memory on segment {index}: continuing with {self.target} workers')
        self._retry(index, message.get('error') or 'failed', kind)
        if worker.retire:
            worker.state = 'retiring'
            worker.shutdown()

    def _retry(self, index, error, kind):
        entry = self.manifest.entry(index)
        self.errors.append({'at': mf.now(), 'segment': index, 'kind': kind, 'error': error})
        if entry['attempts'] < MAX_ATTEMPTS:
            self.manifest.update_segment(index, state='pending', error=error)
            self.queue.append(index)
            self._log(f"segment {index} failed ({kind}): {error} - will retry")
        else:
            self.manifest.update_segment(index, state='failed', error=error)
            self._log(f"segment {index} failed ({kind}) {entry['attempts']} times: {error}")

    def _exited(self, worker, code):
        index = worker.job
        was = worker.state
        kind = worker.job_kind
        worker.state, worker.job, worker.job_kind = 'gone', None, None
        worker.close_log()
        if kind == 'discover' and was == 'busy' and index is not None:
            # A chunk's worker ended mid-chunk: the chunk is retried by another (no GPU worker restart for a light one).
            if worker.light:
                self.light_failures += 1
            else:
                self.restarts += 1
            if self.mode == 'stopping':
                self.manifest.data['discovery']['chunks'][index]['state'] = 'pending'
                self.manifest.save()
            else:
                self._chunk_retry(index, f'the worker process ended (exit code {code})')
            self._log(f'worker {worker.number} died (exit code {code}) during discovery chunk {index}')
        elif worker.light:
            if was == 'starting' and not worker.retire:
                self.light_failures += 1
        elif was == 'busy' and index is not None:
            self.restarts += 1
            if self.mode == 'stopping':
                self.manifest.update_segment(index, state='pending', error='cancelled')
            else:
                self._retry(index, f'the worker process ended (exit code {code})', 'crash')
            self._log(f'worker {worker.number} died (exit code {code}) during segment {index}')
        elif was == 'starting':
            self.restarts += 1
            self._log(f'worker {worker.number} ended before it was ready (exit code {code})')
        self.workers = [w for w in self.workers if w is not worker]

    # ---- resources, progress ------------------------------------------------------------------------------------
    def _check_resources(self, now):
        self.last_resource = now
        snap = system.snapshot(gpu=self.monitor_gpu)
        rss = 0
        with self.lock:
            pids = [w.pid for w in self.workers if w.state != 'gone']
        for pid in pids:
            try:
                p = psutil.Process(pid)
                rss += p.memory_info().rss+sum(c.memory_info().rss for c in p.children(recursive=True))
            except psutil.Error:
                continue
        snap['workersRssGB'] = round(rss/2**30, 2)
        with self.lock:
            self.resources = snap
            self.peaks['workersRssGB'] = max(self.peaks['workersRssGB'], snap['workersRssGB'])
            self.peaks['ramUsedGB'] = max(self.peaks['ramUsedGB'], round(snap['ramTotalGB']-snap['ramAvailableGB'], 1))
            self.peaks['cpuPercent'] = max(self.peaks['cpuPercent'], snap['cpuPercent'])
            if snap.get('gpu'):
                self.peaks['gpuUsedGB'] = max(self.peaks['gpuUsedGB'], snap['gpu']['usedGB'])
                self.peaks['gpuUtilization'] = max(self.peaks['gpuUtilization'], snap['gpu']['utilization'])
            reason = system.unsafe(snap)
            if reason:
                if self.held is None:
                    self._log(f'holding new segments: {reason}')
                self.held = reason
                self.unsafe_since = self.unsafe_since or now
                busy = [w for w in self.workers if w.state == 'busy' and not w.retire and not w.light]
                if now-self.unsafe_since > UNSAFE_HOLD_S and len(busy) > 1 and self.target > 1:
                    self.target -= 1
                    busy[-1].retire = True
                    self.unsafe_since = now
                    self._log(f'resources stayed low ({reason}): continuing with {self.target} workers')
            else:
                if self.held:
                    self._log('resources recovered: starting segments again')
                self.held, self.unsafe_since = None, None
            done = self._frames_done()
            self.samples.append((now, done))
            while len(self.samples) > 2 and now-self.samples[0][0] > RATE_WINDOW_S:
                self.samples.popleft()

    def _frames_done(self):
        total = 0
        entries = self.manifest.data['segments']
        for e in entries:
            if e['state'] == 'done':
                total += e['proc1']-e['proc0']
        for w in self.workers:
            if w.state == 'busy' and w.job_kind != 'discover' and w.progress.get('frames'):
                total += int(w.progress['frames'])
        return total

    def rate(self):
        """Measured aggregate frames per second over the sliding window (None until two minutes of data)."""
        if len(self.samples) < 2:
            return None
        (t0, f0), (t1, f1) = self.samples[0], self.samples[-1]
        if t1-t0 < 120 or f1 <= f0:
            return None
        return (f1-f0)/(t1-t0)

    def simulate(self, rate=None):
        """Seconds until the last segment is tracked: the running jobs' remaining frames and the queue, run on `target`
        workers, each at the per-worker speed of that many concurrent workers (system.throughput / count), scaled by how
        this match's measured aggregate rate compares with the expected one. The tail is slower than the middle: when
        fewer segments remain than workers, fewer run at once (a lone last segment runs at one worker's speed)."""
        busy = [w for w in self.workers if w.state == 'busy' and w.job_kind != 'discover']
        active = max(1, len(busy))
        expected = system.throughput(active, self.measured)
        scale = (rate/expected) if rate else 1.0
        speed = lambda k: system.throughput(k, self.measured)/k*scale
        running = []
        for w in busy:
            e = self.manifest.entry(w.job)
            running.append(max(0.0, (e['proc1']-e['proc0'])-float(w.progress.get('frames') or 0)))
        pending = [self.manifest.entry(i)['proc1']-self.manifest.entry(i)['proc0'] for i in self.queue]
        t = 0.0
        while running or pending:
            while len(running) < self.target and pending:
                running.append(float(pending.pop(0)))
            k = len(running)
            v = max(speed(k), .05)
            step = min(running)/v
            t += step
            running = [r-step*v for r in running]
            running = [r for r in running if r > 1e-6]
        return t

    def assembly_guess(self):
        runtime = self.manifest.data.get('runtime') or {}
        if runtime.get('assemblySeconds'):
            return float(runtime['assemblySeconds'])
        a, b = ASSEMBLY_GUESS_S
        return a+b*len(self.manifest.data['segments'])

    def status(self):
        """The progress the page shows: state, segments, workers, video minutes, stage, elapsed, ETA, resources."""
        with self.lock:
            entries = self.manifest.data['segments']
            counts = self.manifest.counts()
            done_frames = self._frames_done()
            total = max(1, self.total_frames)
            owned_done = sum(e['own1']-e['own0'] for e in entries if e['state'] == 'done')
            owned_total = sum(e['own1']-e['own0'] for e in entries)
            active = [w for w in self.workers if w.state == 'busy' and w.job_kind != 'discover']
            elapsed = time.time()-self.started_at if self.started_at else 0.0
            rate = self.rate()
            remaining_frames = max(0, total-done_frames)
            assembly = self.assembly_guess()
            discovery = self._discovery_status()
            share = DISCOVERY_SHARE if discovery and discovery['state'] not in ('off',) else 0.0
            if self.mode in ('assembling',):
                eta = self._assembly_eta(elapsed)
            elif self.phase == 'discovery':
                left = discovery['etaSeconds'] if discovery else 0.0
                likely = left+self.simulate(None)+assembly
                eta = {'low': round(.8*likely), 'likely': round(likely), 'high': round(1.25*likely), 'measured': False}
            elif rate:
                fraction = done_frames/total
                spread = .25-.15*min(1.0, fraction/.5)
                likely = self.simulate(rate)+assembly
                eta = {'low': round(likely*(1-spread)), 'likely': round(likely), 'high': round(likely*(1+spread)), 'measured': True}
            else:
                likely = self.simulate(None)+assembly if (self.queue or active) else system.estimate_seconds(remaining_frames, max(1, self.target), self.measured, assembly)[1]
                eta = {'low': round(.8*likely), 'likely': round(likely), 'high': round(1.25*likely), 'measured': False}
            if self.mode == 'assembling':
                progress = TRACKING_SHARE+(1-TRACKING_SHARE)*float(self.assembly.get('progress') or 0)
                stage = self.assembly.get('stage') or 'Stitching'
            elif self.manifest.data['state'] == 'complete':
                progress, stage = 1.0, 'Ready'
            elif self.phase == 'discovery':
                progress = share*(discovery['done']/max(1, discovery['total']) if discovery else 0.0)
                stage = 'Calibration' if self.mode == 'running' else {'pausing': 'Pausing', 'stopping': 'Stopping'}.get(self.mode, self.mode.title())
            else:
                progress = share+(TRACKING_SHARE-share)*done_frames/total
                stage = 'Tracking' if self.mode == 'running' else {'pausing': 'Pausing', 'stopping': 'Stopping'}.get(self.mode, self.mode.title())
            return {'matchId': self.manifest.data['matchId'], 'state': self.manifest.data['state'], 'mode': self.mode,
                    'stage': stage, 'progress': round(progress, 4),
                    'segments': {'done': counts.get('done', 0), 'total': len(entries), 'running': len(active),
                                 'failed': counts.get('failed', 0), 'pending': counts.get('pending', 0)},
                    'video': {'doneSeconds': round(owned_done/self.fps, 1), 'totalSeconds': round(owned_total/self.fps, 1)},
                    'frames': {'done': done_frames, 'total': total},
                    'workers': {'target': self.target, 'requested': self.requested, 'threads': self.threads,
                                'active': len(active), 'list': [{'number': w.number, 'pid': w.pid, 'state': w.state, 'segment': w.job,
                                                                 'stage': w.progress.get('stage'), 'progress': w.progress.get('progress'),
                                                                 'fps': w.progress.get('processingFPS'), 'rssGB': w.progress.get('rssGB')}
                                                                for w in self.workers if w.state != 'gone' and not w.light and w.job_kind != 'discover'],
                                'light': sum(1 for w in self.workers if w.light and w.state != 'gone'),
                                'discovering': sum(1 for w in self.workers if w.state == 'busy' and w.job_kind == 'discover')},
                    'elapsed': round(elapsed), 'eta': eta, 'rate': round(rate, 2) if rate else None, 'held': self.held,
                    'resources': self.resources, 'peaks': dict(self.peaks), 'errors': list(self.errors)[-10:],
                    'assembly': dict(self.assembly), 'restarts': self.restarts, 'phase': self.phase,
                    'discovery': discovery, 'calibration': self.manifest.data.get('calibrationStatus')}

    def _discovery_status(self):
        """The discovery's state for the page: chunks done of total, sampled frames, the measured seconds per chunk and
        the remaining time on the processes solving them; the summary once it ended."""
        d = self.manifest.data.get('discovery')
        if not d:
            return None
        chunks = d.get('chunks') or []
        done = [c for c in chunks if c['state'] in ('done', 'failed')]
        solving = sum(1 for w in self.workers if w.state == 'busy' and w.job_kind == 'discover')
        streams = max(1, solving, min(self.streams, len(chunks)-len(done)))
        spent = [float(c['seconds']) for c in chunks if c['state'] == 'done' and c.get('seconds')]
        per_chunk = sum(spent)/len(spent) if spent else system.DISCOVERY_SECONDS_PER_FRAME*(len(chunks[0]['indices']) if chunks else 0)
        left = (len(chunks)-len(done))*per_chunk/streams
        return {'state': d.get('state'), 'done': len(done), 'total': len(chunks), 'sampled': d.get('sampled'),
                'solving': solving, 'etaSeconds': round(left, 1), 'seconds': d.get('seconds'), 'summary': d.get('summary')}

    def _assembly_eta(self, elapsed):
        started = self.assembly.get('startedAt') or time.time()
        guess = self.assembly_guess()
        p = max(.01, float(self.assembly.get('progress') or 0))
        spent = time.time()-started
        likely = max(0.0, spent/p-spent) if p > .1 else max(0.0, guess-spent)
        return {'low': round(likely*.7), 'likely': round(likely), 'high': round(likely*1.5), 'measured': p > .1}

    # ---- the end ------------------------------------------------------------------------------------------------
    def _finish_tracking(self):
        """All segments done: shut the workers down, record the throughput, assemble."""
        self._shutdown_workers()
        elapsed = time.time()-(self.tracking_started or self.started_at or time.time())
        runtime = self.manifest.data.get('runtime') or {}
        runtime.update(trackingSeconds=round(elapsed, 1), workers=self.target, threads=self.threads, peaks=dict(self.peaks),
                       restarts=self.restarts, errors=list(self.errors))
        if (self.manifest.data.get('discovery') or {}).get('seconds') is not None:
            runtime['discoverySeconds'] = self.manifest.data['discovery']['seconds']
        if elapsed > 0:
            runtime['aggregateFPS'] = round(self.total_frames/elapsed, 2)
        self.manifest.data['runtime'] = runtime
        self.manifest.save()
        if self.after_stop is not None:
            self.after_stop()
        if self.assemble is None:
            self._end('complete', None, notify=False)
            return
        with self.lock:
            self.mode = 'assembling'
            self.assembly = {'state': 'running', 'progress': 0.0, 'stage': 'Stitching', 'startedAt': time.time()}
        self.manifest.set_state('assembling')
        started = time.time()

        def progress(fraction, stage=None):
            with self.lock:
                self.assembly['progress'] = max(0.0, min(1.0, float(fraction)))
                if stage:
                    self.assembly['stage'] = stage
        try:
            self.assemble(self.manifest, progress)
            runtime['assemblySeconds'] = round(time.time()-started, 1)
            runtime['totalSeconds'] = round(time.time()-(self.started_at or started), 1)
            self.manifest.data['runtime'] = runtime
            with self.lock:
                self.assembly.update(state='complete', progress=1.0, stage='Ready')
            self._end('complete', None, notify=False)
        except Exception as error:   # noqa: BLE001
            import traceback
            self._log(f'assembly failed: {error}\n{traceback.format_exc()}')
            with self.lock:
                self.assembly.update(state='failed', error=str(error))
            self._end('failed', f'assembly failed: {error}', notify=False)

    def _shutdown_workers(self):
        with self.lock:
            workers = list(self.workers)
        for w in workers:
            if w.alive():
                w.shutdown()
        deadline = time.time()+STOP_WAIT_S
        for w in workers:
            try:
                w.process.wait(timeout=max(.1, deadline-time.time()))
            except subprocess.TimeoutExpired:
                self._log(f'worker {w.number} did not exit: killed')
                w.kill()
            w.close_log()
        # drain the exit events of the workers
        try:
            while True:
                worker, message = self.events.get_nowait()
                if message.get('type') in ('failed', 'done') and worker.job is not None:
                    self._handle(worker, message)
        except queue.Empty:
            pass
        with self.lock:
            self.workers = []

    def _end(self, state, reason, notify=True):
        if self.mode != 'finished' and self.workers:
            self._shutdown_workers()
        with self.lock:
            for entry in self.manifest.data['segments']:
                if entry['state'] == 'running':
                    entry.update(state='pending', error='interrupted')
            for entry in (self.manifest.data.get('discovery') or {}).get('chunks') or []:
                if entry['state'] == 'running':
                    entry['state'] = 'pending'
            self.mode = 'finished'
            fields = {'reason': reason} if reason else {'reason': None}
            self.manifest.set_state(state, **fields)
        self._log(f'match {state}' + (f': {reason}' if reason else ''))
        if notify and self.after_stop is not None:
            self.after_stop()

    def _log(self, text):
        try:
            with (self.manifest.folder/'logs'/'scheduler.log').open('a', encoding='utf-8') as out:
                out.write(f'[{mf.now()}] {text}\n')
        except OSError:
            pass


def segment_config(settings, synthetic=()):
    """The pipeline config of every segment job (start and seconds are the worker's): the page's processing settings,
    every calibration and boundary keyframe of the match (each is applied only at its own time; all calibrations
    together set the pitch size, so every segment shares one pitch), the segment's own discovered keyframes
    (calibration.py; solved with that pitch size, so they leave it as it is), no identity continuation (segments run
    in parallel; the stitching links them)."""
    out = {k: settings[k] for k in ('imageSize', 'ballTiles', 'fieldFilter', 'touchlineMargin', 'maxPerTeam', 'debug') if k in settings}
    out['boundaries'] = list(settings.get('boundaries') or [])
    out['calibrations'] = sorted(list(settings.get('calibrations') or [])+[dict(k) for k in synthetic], key=lambda c: float(c['time']))
    out['continueIdentities'] = False
    for key in ('detectBatch', 'ballScanEvery', 'nearPass'):
        if key in settings:
            out[key] = settings[key]
    return out
