"""Full Match Mode's scheduler (fullmatch/scheduler.py) with fake worker processes (fullmatch_fake_worker.py): real
subprocesses and the real protocol, no models or GPU.

What is checked: a match completes with 1, 2 and 3 workers, never more jobs at once than workers, each segment run
once, its verified artifact recorded and the assembly called once; a failed job is retried; a worker that dies mid-job
is replaced and its segment retried; a GPU out-of-memory failure drops one worker and retries the segment; a segment that
keeps failing leaves the match failed with its other segments kept, and resume retries it alone; pause lets running
jobs finish and starts no new ones, resume continues without re-running finished segments; stop cancels the running
job (back to pending) and resume completes the match; workers that cannot start fail the match instead of waiting
forever; the status() the page shows (progress that never goes backwards, segments, video seconds, ETA); a worker
exits when its stdin closes (no orphans). The calibration discovery stage (with a stand-in for calibration.py): its
chunks run before any segment, on the GPU workers and light ones that leave when it ends, a failed chunk is retried,
each segment job gets its own discovered keyframes with the user's, a refused or unplannable discovery lets the match
go on without them, and a stop during discovery resumes with the missing chunks only.

Run:  python -m unittest test_fullmatch_scheduler          (from this directory)
"""
import contextlib
import json
import os
import subprocess
import sys
import tempfile
import time
import types
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import fullmatch  # noqa: E402
from fullmatch import manifest as mf, plan, scheduler as sch  # noqa: E402
import fullmatch_journal  # noqa: E402

FAKE = Path(__file__).resolve().parent/'fullmatch_fake_worker.py'
VIDEO = {'name': 'm.mp4', 'width': 1920, 'height': 1080, 'fps': 25.0, 'frames': 6000, 'duration': 240.0}
SETTINGS = {'imageSize': 1280, 'ballTiles': False, 'boundaries': [], 'calibrations': [], 'fieldFilter': True,
            'touchlineMargin': .35, 'maxPerTeam': 11, 'debug': True}


class Harness:
    def __init__(self, outcomes=None, delay=.15, startup='ok', segment_s=20.0, discovery=False, chunk_outcomes=None,
                 chunk_delay=.05, settings=None):
        self.root = Path(tempfile.mkdtemp())
        self.script = self.root/'script.json'
        self.script.write_text(json.dumps({'delay': delay, 'startup': startup, 'outcomes': outcomes or {},
                                           'chunkOutcomes': chunk_outcomes or {}, 'chunkDelay': chunk_delay}), encoding='utf-8')
        periods = plan.periods(VIDEO['duration'], end=100.0)
        self.segments = plan.plan(VIDEO['frames'], VIDEO['fps'], periods, segment_s=segment_s, pad_s=1.0)
        extra = {'discovery': {'mode': 'auto', 'state': 'pending'}} if discovery else None
        self.manifest = mf.Manifest.create(self.root/'matches'/'m', {'key': 'k'*64, 'path': 'x.mp4'}, VIDEO, periods,
                                           self.segments, settings or SETTINGS, {'mode': 'auto'}, 1.0, segment_s, extra=extra)
        self.assembled = []

    def assemble(self, manifest, progress):
        progress(.5, 'Stitching')
        self.assembled.append(manifest.data['matchId'])
        progress(1.0, 'Finalizing')

    def scheduler(self, workers, streams=None):
        return sch.MatchScheduler(self.manifest, workers, command=lambda n: [sys.executable, '-B', str(FAKE), str(self.script), '--name', f'w{n}'],
                                  assemble=self.assemble, monitor_gpu=False, streams=streams,
                                  light=lambda n: [sys.executable, '-B', str(FAKE), str(self.script), '--name', f'light{n}', '--light'])

    def journal(self):
        return fullmatch_journal.read(self.root/'journal.jsonl')

    def starts(self, segment=None):
        return [e for e in self.journal() if e['event'] == 'start' and (segment is None or e['segment'] == segment)]


def max_concurrency(journal):
    spans = {}
    for e in journal:
        key = (e['worker'], e['segment'], e['pid'])
        if e['event'] == 'start':
            spans.setdefault(key, []).append([e['t'], None])
        elif spans.get(key):
            spans[key][-1][1] = e['t']
    edges = []
    for runs in spans.values():
        for a, b in runs:
            edges += [(a, 1), (b if b is not None else a+1e-3, -1)]
    best = cur = 0
    for _, d in sorted(edges):
        cur += d
        best = max(best, cur)
    return best


def run_to_end(test, s, timeout=60):
    test.assertTrue(s.wait(timeout), 'the scheduler did not finish in time')


@mock.patch.object(sch, 'STAGGER_S', .05)
@mock.patch.object(sch, 'RESOURCE_EVERY_S', .2)
class SchedulerTests(unittest.TestCase):
    def test_a_match_completes_with_one_two_and_three_workers(self):
        for workers in (1, 2, 3):
            with self.subTest(workers=workers):
                h = Harness()
                s = h.scheduler(workers)
                s.start()
                run_to_end(self, s)
                self.assertEqual(h.manifest.data['state'], 'complete')
                self.assertEqual(h.manifest.counts()['done'], len(h.segments))
                self.assertEqual(sorted(e['segment'] for e in h.starts()), [x.index for x in h.segments])
                self.assertLessEqual(max_concurrency(h.journal()), workers)
                if workers > 1:
                    self.assertGreater(max_concurrency(h.journal()), 1)
                for x in h.segments:
                    self.assertIsNone(mf.verify_artifact(h.manifest.artifact_folder(x.index), x))
                self.assertEqual(len(h.assembled), 1)
                status = s.status()
                self.assertEqual((status['state'], status['progress'], status['segments']['done']), ('complete', 1.0, len(h.segments)))
                self.assertEqual(status['video']['doneSeconds'], status['video']['totalSeconds'])
                self.assertIn('aggregateFPS', h.manifest.data['runtime'])

    def test_a_failed_job_is_retried(self):
        h = Harness(outcomes={'1': ['error']})
        s = h.scheduler(2)
        s.start()
        run_to_end(self, s)
        self.assertEqual(h.manifest.data['state'], 'complete')
        self.assertEqual(len(h.starts(1)), 2)
        self.assertEqual(h.manifest.entry(1)['attempts'], 2)

    def test_a_worker_that_dies_is_replaced_and_its_segment_retried(self):
        h = Harness(outcomes={'2': ['crash']})
        s = h.scheduler(2)
        s.start()
        run_to_end(self, s)
        self.assertEqual(h.manifest.data['state'], 'complete')
        self.assertEqual(len(h.starts(2)), 2)
        self.assertGreaterEqual(s.restarts, 1)
        self.assertGreaterEqual(len({e['pid'] for e in h.journal()}), 3)   # two workers and the replacement

    def test_gpu_out_of_memory_drops_a_worker_and_retries(self):
        h = Harness(outcomes={'0': ['oom']})
        s = h.scheduler(3)
        s.start()
        run_to_end(self, s)
        self.assertEqual(h.manifest.data['state'], 'complete')
        self.assertEqual(s.target, 2)
        self.assertEqual(len(h.starts(0)), 2)
        self.assertTrue(any(e.get('kind') == 'oom' for e in s.status()['errors']))

    def test_a_segment_that_keeps_failing_fails_the_match_and_resume_retries_it(self):
        h = Harness(outcomes={'3': ['error', 'error', 'error']})
        s = h.scheduler(2)
        s.start()
        run_to_end(self, s)
        self.assertEqual(h.manifest.data['state'], 'failed')
        self.assertEqual(h.manifest.entry(3)['state'], 'failed')
        done_before = h.manifest.counts()['done']
        self.assertEqual(done_before, len(h.segments)-1)
        runs_before = len(h.starts())
        s2 = h.scheduler(2)
        s2.start(resume=True)
        run_to_end(self, s2)
        self.assertEqual(h.manifest.data['state'], 'complete')
        self.assertEqual(len(h.starts())-runs_before, 1)    # only the failed segment ran again

    def test_pause_finishes_running_jobs_and_resume_continues(self):
        h = Harness(delay=.4)
        s = h.scheduler(2)
        s.start()
        while len(h.starts()) < 2:
            time.sleep(.02)
        s.pause()
        run_to_end(self, s)
        self.assertEqual(h.manifest.data['state'], 'paused')
        done = h.manifest.counts()['done']
        self.assertGreaterEqual(done, 2)
        self.assertLess(done, len(h.segments))
        self.assertEqual(len(h.starts()), done)          # every started job finished; nothing new started
        s2 = h.scheduler(2)
        s2.start(resume=True)
        run_to_end(self, s2)
        self.assertEqual(h.manifest.data['state'], 'complete')
        self.assertEqual(len(h.starts()), len(h.segments))   # nothing ran twice

    def test_stop_cancels_the_running_job_and_resume_completes(self):
        h = Harness(outcomes={'1': ['slow']}, delay=.2)
        s = h.scheduler(1)
        s.start()
        while not h.starts(1):
            time.sleep(.02)
        time.sleep(.2)
        s.stop()
        run_to_end(self, s)
        self.assertEqual(h.manifest.data['state'], 'stopped')
        self.assertEqual(h.manifest.entry(1)['state'], 'pending')
        self.assertEqual(h.manifest.entry(0)['state'], 'done')
        self.assertTrue(any(e['event'] == 'cancelled' for e in h.journal()))
        s2 = h.scheduler(1)
        s2.start(resume=True)
        run_to_end(self, s2)
        self.assertEqual(h.manifest.data['state'], 'complete')
        self.assertEqual(len(h.starts(0)), 1)

    def test_workers_that_cannot_start_fail_the_match(self):
        h = Harness(startup='fail')
        s = h.scheduler(2)
        s.start()
        run_to_end(self, s, timeout=90)
        self.assertEqual(h.manifest.data['state'], 'failed')
        self.assertIn('could not start', h.manifest.data['reason'])
        self.assertEqual(h.manifest.counts()['done'], 0)

    def test_status_progress_never_goes_backwards(self):
        h = Harness(delay=.1)
        s = h.scheduler(2)
        seen = []
        s.start()
        while not s.wait(.05):
            st = s.status()
            seen.append(st['progress'])
            self.assertIn(st['stage'], ('Tracking', 'Stitching', 'Finalizing', 'Pausing', 'Stopping', 'Finished', 'Assembling', 'Idle', 'Ready'))
            self.assertIsNotNone(st['eta']['likely'])
        seen.append(s.status()['progress'])
        self.assertTrue(all(b >= a-1e-9 for a, b in zip(seen, seen[1:])), seen)
        self.assertEqual(seen[-1], 1.0)


class FakeCalibration(types.ModuleType):
    """calibration.py's interface: frames every 2 s, chunks of 5 overlapping by one, and a discover() that puts one
    keyframe at the middle of every other segment (or refuses / fails, as a test asks)."""
    VERSION = 'fake-1'

    def __init__(self, refused=None, fail_plan=False, fail_discover=False):
        super().__init__('fullmatch.calibration')
        self.refused, self.fail_plan, self.fail_discover = refused, fail_plan, fail_discover
        self.rows = None

    def sample(self, periods, fps, step_s=2.0):
        if self.fail_plan:
            raise ValueError('fake planning failure')
        return [i for a, b in periods for i in range(a, b, int(round(step_s*fps)))]

    def chunks(self, indices, size=5, gaps=(1,), periods=None):
        return [indices[i:i+size+1] for i in range(0, len(indices), size)]

    def discover(self, rows, pairs, segments, fps, width, height, size=None, manual=()):
        if self.fail_discover:
            raise RuntimeError('fake discover failure')
        self.rows = sorted(r['index'] for r in rows)
        keyframes = [] if self.refused else [{'time': (s.proc0+s.proc1)//2/fps, 'frame': (s.proc0+s.proc1)//2, 'points': [],
                                               'source': 'discovery'} for s in segments if s.index % 2 == 0]
        return {'version': self.VERSION, 'sampled': len(rows), 'anchors': 0, 'kept': len(keyframes), 'camera': None,
                'refused': self.refused, 'keyframes': keyframes}

    def keyframes_for(self, index, segment):
        return [k for k in index['keyframes'] if segment.proc0 <= k['frame'] < segment.proc1]


@mock.patch.object(sch, 'STAGGER_S', .05)
@mock.patch.object(sch, 'LIGHT_STAGGER_S', .02)
@mock.patch.object(sch, 'RESOURCE_EVERY_S', .2)
class DiscoveryTests(unittest.TestCase):
    def patched(self, fake):
        # The package attribute is what `from . import calibration` takes. (Not mock.patch.dict(sys.modules): restoring
        # the whole dict drops cv2 when a test is the first to import it, and cv2 cannot be imported a second time.)
        return contextlib.nullcontext(), mock.patch.object(fullmatch, 'calibration', fake, create=True)

    def run_with(self, fake, h, workers=2, streams=4, watch=None):
        a, b = self.patched(fake)
        with a, b:
            s = h.scheduler(workers, streams=streams)
            s.start()
            while not s.wait(.03):
                if watch:
                    watch(s.status())
            return s

    def test_chunks_run_first_then_each_segment_gets_its_keyframes(self):
        manual = {'time': 30.0, 'points': []}
        h = Harness(delay=.1, discovery=True, settings=SETTINGS | {'calibrations': [manual]})
        fake, seen = FakeCalibration(), []
        self.run_with(fake, h, watch=lambda st: seen.append((st['stage'], st['progress'])))
        self.assertEqual(h.manifest.data['state'], 'complete')
        journal = h.journal()
        chunk_ends = [e['t'] for e in journal if e['event'] == 'chunk-end']
        starts = [e for e in journal if e['event'] == 'start']
        d = h.manifest.data['discovery']
        self.assertEqual(len(chunk_ends), len(d['chunks']))
        self.assertLess(max(chunk_ends), min(e['t'] for e in starts))          # every chunk before any segment
        self.assertTrue(any(e.get('light') for e in journal if e['event'] == 'chunk-start'))   # light workers helped
        self.assertTrue(any(not e.get('light') for e in journal if e['event'] == 'chunk-start'))
        self.assertEqual(fake.rows, sorted({i for c in d['chunks'] for i in c['indices']}))  # every sampled frame, once
        self.assertEqual(d['state'], 'complete')
        index = json.loads((h.manifest.folder/'calibration'/'discovery.json').read_text(encoding='utf-8'))
        for e in starts:
            mine = [k['time'] for k in fake.keyframes_for(index, h.segments[e['segment']])]
            self.assertEqual(sorted(e['calibrations']), sorted([30.0]+mine))  # the user's keyframe + its own discovered ones
        status = h.manifest.data['calibrationStatus']
        self.assertEqual(status['discoveredSegments'], [s.index for s in h.segments if s.index % 2 == 0])
        self.assertIn('Calibration', [st for st, _ in seen])
        progress = [p for _, p in seen]
        self.assertTrue(all(b >= a-1e-9 for a, b in zip(progress, progress[1:])), progress)
        self.assertIsNotNone(h.manifest.data['runtime'].get('discoverySeconds'))

    def test_a_failed_chunk_is_retried_and_a_refused_discovery_lets_the_match_go_on(self):
        h = Harness(delay=.05, discovery=True, chunk_outcomes={'1': ['error', 'ok'], '2': ['crash', 'ok']})
        fake = FakeCalibration(refused='no camera position most anchors agree on')
        self.run_with(fake, h)
        self.assertEqual(h.manifest.data['state'], 'complete')
        d = h.manifest.data['discovery']
        self.assertEqual(d['state'], 'refused')
        self.assertTrue(all(c['state'] == 'done' for c in d['chunks']))
        self.assertEqual(sum(1 for e in h.journal() if e['event'] == 'chunk-start' and e['chunk'] == 1), 2)
        self.assertTrue(all(e['calibrations'] == [] for e in h.journal() if e['event'] == 'start'))
        self.assertIn('no safe anchors', h.manifest.data['calibrationStatus']['note'])

    def test_discovery_that_cannot_be_planned_or_fails_never_fails_the_match(self):
        for fake in (FakeCalibration(fail_plan=True), FakeCalibration(fail_discover=True)):
            h = Harness(delay=.05, discovery=True)
            self.run_with(fake, h)
            self.assertEqual(h.manifest.data['state'], 'complete')
            self.assertEqual(h.manifest.data['discovery']['state'], 'failed')
            self.assertTrue(all(e['calibrations'] == [] for e in h.journal() if e['event'] == 'start'))

    def test_stop_during_discovery_then_resume_solves_the_missing_chunks_only(self):
        h = Harness(delay=.05, discovery=True, chunk_delay=.4)
        fake = FakeCalibration()
        a, b = self.patched(fake)
        with a, b:
            s = h.scheduler(1, streams=2)
            s.start()
            deadline = time.time()+20
            while time.time() < deadline and not any(c['state'] == 'done' for c in h.manifest.data['discovery']['chunks']):
                time.sleep(.02)
            s.stop()
            self.assertTrue(s.wait(30))
            self.assertEqual(h.manifest.data['state'], 'stopped')
            done = {c['index'] for c in h.manifest.data['discovery']['chunks'] if c['state'] == 'done'}
            self.assertTrue(done)
            self.assertFalse(any(e['event'] == 'start' for e in h.journal()))   # no segment before discovery ended
            s2 = h.scheduler(1, streams=2)
            s2.start(resume=True)
            self.assertTrue(s2.wait(60))
        self.assertEqual(h.manifest.data['state'], 'complete')
        solved = [e['chunk'] for e in h.journal() if e['event'] == 'chunk-end']
        self.assertEqual(len(solved), len(set(solved)))   # no chunk solved twice
        self.assertEqual(set(solved), {c['index'] for c in h.manifest.data['discovery']['chunks']})


class OrphanTests(unittest.TestCase):
    def test_a_worker_exits_when_its_stdin_closes(self):
        root = Path(tempfile.mkdtemp())
        script = root/'script.json'
        script.write_text(json.dumps({'delay': .1}), encoding='utf-8')
        p = subprocess.Popen([sys.executable, '-B', str(FAKE), str(script)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        self.assertEqual(json.loads(p.stdout.readline())['type'], 'ready')
        p.stdin.close()
        self.assertIsNotNone(p.wait(timeout=10))


class ConfigTests(unittest.TestCase):
    def test_every_segment_gets_every_keyframe_and_no_identity_continuation(self):
        settings = SETTINGS | {'calibrations': [{'time': 10.0, 'points': []}, {'time': 4000.0, 'points': []}],
                               'boundaries': [{'time': 5.0, 'points': [[0, 0], [1, 0], [1, 1]]}], 'continueIdentities': True,
                               'start': 1.0, 'seconds': 2.0}
        config = sch.segment_config(settings)
        self.assertEqual(len(config['calibrations']), 2)
        self.assertEqual(len(config['boundaries']), 1)
        self.assertFalse(config['continueIdentities'])
        self.assertNotIn('start', config)
        self.assertNotIn('seconds', config)

    def test_thread_caps_reach_the_worker_environment(self):
        env = sch.thread_environment(3)
        self.assertEqual((env['OMP_NUM_THREADS'], env['MKL_NUM_THREADS'], env['OPENBLAS_NUM_THREADS']), ('3', '3', '3'))
        self.assertEqual(sch.thread_environment(0).get('OMP_NUM_THREADS'), os.environ.get('OMP_NUM_THREADS'))


if __name__ == '__main__':
    unittest.main()
