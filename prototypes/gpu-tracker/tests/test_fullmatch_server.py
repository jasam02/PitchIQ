"""Full Match Mode's routes in server.py (fullmatch/routes.py) through Flask's test client, on a scratch folder: server.py
loaded with stand-ins for pipeline and download_model (as test_possession_server), the scheduler's workers replaced by the
fake worker (fullmatch_fake_worker.py: the real protocol and artifacts, no models), nvidia-smi by a fixed reading.

What is checked: the overview of a video with no matches; a plan (periods with a halftime break, segments, the AUTO
workers, an estimate with a range, disk, which segments a calibration keyframe anchors); bad ranges and bad keyframes are
refused like a Segment run's; a started match runs to completion, saves the dock's settings, refuses a second match and a
Segment run while it runs, and reports progress; pause and resume; stop and resume through the routes; a match the
server left running (closed mid-run) is interrupted at the next start and offered for resume; a match of another video
cannot be resumed here; a plan says the calibration discovery runs first where no keyframe is within reach (and not
when every segment has one, or the request turns it off), and the started match runs it before its segments; the
coaching routes answer a match (match= or a 'match-<id>' run) from the match, never from the video's Segment run, pass
the profile catalogue through and refuse plainly what a match has no data for.

Run:  python -m unittest test_fullmatch_server          (from this directory)
"""
import contextlib
import hashlib
import importlib.util
import io
import json
import sys
import tempfile
import threading
import time
import types
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fullmatch import manifest as mf, plan, scheduler as sch, system  # noqa: E402
from test_coaching_store import write_video  # noqa: E402

PORT = 8765
BASE = f'http://127.0.0.1:{PORT}'
FAKE = Path(__file__).resolve().parent/'fullmatch_fake_worker.py'
GPU = {'name': 'test GPU', 'utilization': 10.0, 'usedGB': 1.0, 'totalGB': 16.0, 'freeGB': 15.0}


def load_server(root):
    import cv2

    def metadata(path):
        capture = cv2.VideoCapture(str(path))
        try:
            fps, count = float(capture.get(cv2.CAP_PROP_FPS)), int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            return {'name': Path(path).name, 'width': int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    'height': int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)), 'fps': fps, 'frames': count, 'duration': count/fps}
        finally:
            capture.release()

    gate = threading.Event()

    def run(source, destination, config, progress, cancelled, previous):
        Path(destination).mkdir(parents=True, exist_ok=True)
        progress({'stage': 'Tracking'})
        gate.wait(20)
        return {'version': 3, 'status': 'complete', 'frames': [], 'match': {'players': []}}
    pipeline = types.ModuleType('pipeline')
    pipeline.ROOT, pipeline.metadata, pipeline.run = root, metadata, run
    pipeline.RESOURCES = types.SimpleNamespace(load=lambda *a: None, release=lambda: None)
    pipeline.gpu_info = lambda: {'name': 'test', 'torch': '-', 'cuda': '-', 'memoryGB': 0}
    download = types.ModuleType('download_model')
    download.get_model = lambda: None
    saved = {k: sys.modules.get(k) for k in ('pipeline', 'download_model')}
    sys.modules.update(pipeline=pipeline, download_model=download)
    try:
        spec = importlib.util.spec_from_file_location('fullmatch_test_server', ROOT/'server.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
    return module, gate


class Scratch:
    def __init__(self, outcomes=None, delay=.1):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root/'uploads').mkdir()
        self.video = write_video(self.root/'uploads'/'clip.avi', frames=3000, fps=25)   # 120 s
        self.key = hashlib.sha256(self.video.read_bytes()).hexdigest()
        self.folder = self.root/'runs'/self.key[:16]
        self.script = self.root/'script.json'
        self.script.write_text(json.dumps({'delay': delay, 'outcomes': outcomes or {}}), encoding='utf-8')
        self.patches = [mock.patch.object(sch, 'default_command', lambda threads=0, name='w': [sys.executable, '-B', str(FAKE), str(self.script), '--name', name]),
                        mock.patch.object(sch, 'STAGGER_S', .05), mock.patch.object(sch, 'RESOURCE_EVERY_S', .2),
                        mock.patch.object(system, 'gpu_status', lambda: dict(GPU)),
                        # A fixed reading of the computer (the tests must not depend on what else it is running).
                        mock.patch.object(system, 'snapshot', lambda gpu=True: {'ramTotalGB': 32.0, 'ramAvailableGB': 20.0, 'cpus': 16,
                                                                               'cpuPercent': 10.0, 'gpu': dict(GPU) if gpu else None})]
        for p in self.patches:
            p.start()
        self.server, self.gate = load_server(self.root)
        self.start_server()

    def start_server(self):
        captured = []
        argv = ['server.py', '--video', str(self.video), '--port', str(PORT)]
        with mock.patch.object(sys, 'argv', argv), mock.patch.object(self.server, 'serve', lambda app, **kw: captured.append(app)), \
                mock.patch.object(self.server, 'preload_models', lambda: None), mock.patch('atexit.register', lambda f: None), \
                contextlib.redirect_stdout(io.StringIO()):
            self.server.main()
        self.app = captured[0]
        self.client = self.app.test_client()

    def close(self):
        for p in self.patches:
            p.stop()
        self.gate.set()
        self.tmp.cleanup()

    def get(self, path, **params):
        return self.client.get(path, query_string=params, base_url=BASE)

    def post(self, path, body):
        return self.client.post(path, json=body, base_url=BASE)

    def wait_state(self, states, timeout=60):
        deadline = time.time()+timeout
        while time.time() < deadline:
            st = self.get('/api/match/status').get_json()
            if st.get('state') in states and st.get('mode') in ('finished', 'idle', None):
                return st
            time.sleep(.05)
        raise AssertionError(f'the match did not reach {states}: {st}')


BODY = {'segmentSeconds': 30, 'padSeconds': 1, 'workers': 'auto',
        'settings': {'imageSize': 1280, 'calibrations': [], 'boundaries': [], 'maxPerTeam': 11}}


class MatchRoutes(unittest.TestCase):
    def setUp(self):
        self.s = Scratch()
        self.addCleanup(self.s.close)

    def test_the_overview_of_a_video_without_matches(self):
        r = self.s.get('/api/match')
        self.assertEqual(r.status_code, 200)
        data = r.get_json()
        self.assertEqual((data['matches'], data['resumable'], data['busy']), ([], None, False))
        self.assertEqual(data['video']['key'], self.s.key)
        self.assertGreaterEqual(data['defaults']['autoWorkers'], 1)

    def test_a_plan_with_a_halftime_break(self):
        body = dict(BODY, start=5, end=115, halftime=[55, 65])
        r = self.s.post('/api/match/plan', body)
        self.assertEqual(r.status_code, 200, r.get_data(as_text=True))
        p = r.get_json()
        self.assertEqual([(x['start'], x['end'], x['name']) for x in p['periods']], [(5.0, 55.0, 'first half'), (65.0, 115.0, 'second half')])
        self.assertEqual(p['segments'], 4)
        self.assertEqual(p['matchSeconds'], 100.0)
        self.assertLessEqual(p['estimate']['low'], p['estimate']['likely'])
        self.assertLessEqual(p['estimate']['likely'], p['estimate']['high'])
        self.assertTrue(p['disk']['ok'])
        self.assertEqual(p['calibration']['anchored'], 0)
        self.assertIn('automatic', p['calibration']['note'])
        self.assertEqual(p['workers']['mode'], 'auto')

    def test_a_keyframe_anchors_the_segments_within_reach(self):
        # A real keyframe (CAN-ISL tactical wide at 530 s, 1920 x 1080 pixels, placed by hand): normalized, it solves on
        # any 16:9 frame.
        pixels = {'halfway-far': (1133, 430), 'centre-circle-far': (1129, 526), 'centre-circle-near': (1123, 631),
                  'centre-circle-left': (948, 578), 'centre-circle-right': (1310, 577), 'left-penalty-box-front-near': (253, 711),
                  'left-penalty-box-front-far': (576, 480), 'left-goal-box-front-near': (162, 624),
                  'left-goal-box-front-far': (338, 526), 'left-penalty-spot': (354, 570), 'right-penalty-box-front-far': (1734, 475)}
        points = [{'name': n, 'x': x/1920, 'y': y/1080} for n, (x, y) in pixels.items()]
        body = dict(BODY, settings=dict(BODY['settings'], calibrations=[{'time': 100.0, 'points': points}]))
        r = self.s.post('/api/match/plan', body)
        self.assertEqual(r.status_code, 200, r.get_data(as_text=True))
        p = r.get_json()
        self.assertEqual(p['calibration']['keyframes'], 1)
        self.assertEqual(p['calibration']['anchoredSegments'], [2, 3])   # 100 s is within 15 s of the last two (of 4)
        self.assertEqual(p['calibration']['automatic'], 2)

    def test_bad_ranges_and_keyframes_are_refused(self):
        self.assertEqual(self.s.post('/api/match/plan', dict(BODY, start=0, end=100, halftime=[110, 115])).status_code, 400)
        bad = dict(BODY, settings=dict(BODY['settings'], calibrations=[{'time': 10.0, 'points': [{'name': 'nowhere', 'x': .1, 'y': .1}]*4}]))
        r = self.s.post('/api/match/plan', bad)
        self.assertEqual(r.status_code, 400)
        self.assertIn('error', r.get_json())

    def test_a_match_runs_to_completion_and_blocks_other_runs(self):
        r = self.s.post('/api/match/start', dict(BODY, workers=2))
        self.assertEqual(r.status_code, 200, r.get_data(as_text=True))
        match_id = r.get_json()['matchId']
        self.assertEqual(self.s.post('/api/match/start', BODY).status_code, 400)
        refused = self.s.post('/api/run', {'start': 0, 'seconds': 10})
        self.assertEqual(refused.status_code, 400)
        self.assertIn('full match', refused.get_json()['error'])
        st = self.s.wait_state({'complete'})
        self.assertEqual(st['segments']['done'], st['segments']['total'])
        self.assertEqual(st['progress'], 1.0)
        manifest = mf.Manifest.load(self.s.folder/'matches'/match_id/'manifest.json')
        self.assertEqual(manifest.data['state'], 'complete')
        self.assertEqual(manifest.data['workers']['count'], 2)
        saved = json.loads((self.s.folder/'settings.json').read_text(encoding='utf-8'))
        self.assertEqual(saved['imageSize'], 1280)
        overview = self.s.get('/api/match').get_json()
        self.assertEqual(overview['matches'][0]['state'], 'complete')
        self.assertIsNone(overview['resumable'])

    def test_stop_and_resume_through_the_routes(self):
        self.s.script.write_text(json.dumps({'delay': .1, 'outcomes': {'1': ['slow']}}), encoding='utf-8')
        r = self.s.post('/api/match/start', dict(BODY, workers=1))
        match_id = r.get_json()['matchId']
        journal = self.s.root/'journal.jsonl'
        deadline = time.time()+30
        while time.time() < deadline and not (journal.exists() and '"segment": 1' in journal.read_text(encoding='utf-8')):
            time.sleep(.05)
        self.assertEqual(self.s.post('/api/match/stop', {}).status_code, 200)
        self.s.wait_state({'stopped'})
        overview = self.s.get('/api/match').get_json()
        self.assertEqual(overview['resumable']['matchId'], match_id)
        self.assertEqual(self.s.post('/api/match/resume', {'matchId': match_id}).status_code, 200)
        self.s.wait_state({'complete'})

    def test_pause_and_resume(self):
        self.s.script.write_text(json.dumps({'delay': .3}), encoding='utf-8')
        match_id = self.s.post('/api/match/start', dict(BODY, workers=1)).get_json()['matchId']
        time.sleep(.5)
        self.assertEqual(self.s.post('/api/match/pause', {}).status_code, 200)
        st = self.s.wait_state({'paused'})
        self.assertLess(st['segments']['done'], st['segments']['total'])
        self.assertEqual(self.s.post('/api/match/resume', {'matchId': match_id}).status_code, 200)
        self.s.wait_state({'complete'})

    def test_a_match_left_running_is_offered_for_resume_after_a_restart(self):
        meta = {'name': 'clip.avi', 'width': 64, 'height': 36, 'fps': 25.0, 'frames': 3000, 'duration': 120.0}
        segments = plan.plan(3000, 25.0, plan.periods(120.0), 30.0, 1.0)
        m = mf.Manifest.create(self.s.folder/'matches'/'20260101-000000-aaaaaa', {'key': self.s.key, 'path': str(self.s.video)},
                               meta, plan.periods(120.0), segments, BODY['settings'] | {'debug': True, 'ballTiles': False,
                                                                                         'fieldFilter': True, 'touchlineMargin': .35},
                               {'mode': 'auto'}, 1.0, 30.0)
        m.set_state('running')
        m.update_segment(0, state='running')
        self.s.start_server()     # the server starts again
        overview = self.s.get('/api/match').get_json()
        self.assertEqual(overview['resumable']['matchId'], '20260101-000000-aaaaaa')
        self.assertEqual(overview['resumable']['state'], 'stopped')

    def test_a_stopped_match_a_newer_complete_one_replaced_is_not_offered(self):
        meta = {'name': 'clip.avi', 'width': 64, 'height': 36, 'fps': 25.0, 'frames': 3000, 'duration': 120.0}
        segments = plan.plan(3000, 25.0, plan.periods(120.0), 30.0, 1.0)

        def make(name, state, created):
            m = mf.Manifest.create(self.s.folder/'matches'/name, {'key': self.s.key, 'path': str(self.s.video)}, meta,
                                   plan.periods(120.0), segments, {'imageSize': 1280}, {'mode': 'auto'}, 1.0, 30.0)
            m.data['created'] = created
            m.set_state(state)
        make('20260101-000000-cccccc', 'stopped', '2026-01-01T00:00:00')
        make('20260101-010000-dddddd', 'complete', '2026-01-01T01:00:00')
        self.assertIsNone(self.s.get('/api/match').get_json()['resumable'])
        make('20260101-020000-eeeeee', 'stopped', '2026-01-01T02:00:00')   # stopped after the complete one: offered
        self.assertEqual(self.s.get('/api/match').get_json()['resumable']['matchId'], '20260101-020000-eeeeee')

    def test_a_match_of_another_video_cannot_be_resumed_here(self):
        meta = {'name': 'x', 'width': 64, 'height': 36, 'fps': 25.0, 'frames': 3000, 'duration': 120.0}
        segments = plan.plan(3000, 25.0, plan.periods(120.0), 30.0, 1.0)
        mf.Manifest.create(self.s.folder/'matches'/'20260101-000000-bbbbbb', {'key': 'f'*64, 'path': 'other.mp4'}, meta,
                           plan.periods(120.0), segments, {'imageSize': 1280}, {'mode': 'auto'}, 1.0, 30.0).set_state('stopped')
        r = self.s.post('/api/match/resume', {'matchId': '20260101-000000-bbbbbb'})
        self.assertEqual(r.status_code, 400)
        self.assertIn('different file', r.get_json()['error'])
        self.assertEqual(self.s.post('/api/match/resume', {'matchId': '../../etc'}).status_code, 400)

    def test_the_plan_says_when_calibration_discovery_runs_and_the_match_runs_it_first(self):
        plan_out = self.s.post('/api/match/plan', BODY).get_json()
        self.assertTrue(plan_out['calibration']['discovery']['runs'])
        self.assertIn('looks for', plan_out['calibration']['note'])
        self.assertFalse(self.s.post('/api/match/plan', dict(BODY, discovery='off')).get_json()['calibration']['discovery']['runs'])
        r = self.s.post('/api/match/start', dict(BODY, workers=1))
        self.assertEqual(r.status_code, 200, r.get_data(as_text=True))
        self.s.wait_state({'complete'})
        manifest = mf.Manifest.load(self.s.folder/'matches'/r.get_json()['matchId']/'manifest.json')
        d = manifest.data['discovery']
        # The fake worker solves no anchor: discovery ran over every chunk, refused, and the match went on without it.
        self.assertTrue(d['chunks'] and all(c['state'] == 'done' for c in d['chunks']), d)
        self.assertIn(d['state'], ('refused', 'complete'))
        self.assertIsNotNone(manifest.data['runtime'].get('discoverySeconds'))

    def test_coaching_routes_answer_a_match_from_the_match(self):
        r = self.s.post('/api/match/start', dict(BODY, workers=2, discovery='off'))
        match_id = r.get_json()['matchId']
        self.s.wait_state({'complete'})
        answer = self.s.get('/api/coaching', match=match_id, team='A')
        self.assertEqual(answer.status_code, 200, answer.get_data(as_text=True))
        data = answer.get_json()
        self.assertEqual((data['engine'], data['run'], data['state']), (2, f'match-{match_id}', 'missing'))
        self.assertEqual(data['match']['matchId'], match_id)
        spans = self.s.get('/api/coaching/spans', run=f'match-{match_id}').get_json()
        self.assertEqual(spans['spans']['spans'], [])
        self.assertEqual(self.s.get('/api/coaching/profiles', match=match_id).status_code,
                         self.s.get('/api/coaching/profiles').status_code)      # the catalogue is the same for both
        refused = self.s.post(f'/api/coaching/feedback?match={match_id}', {'run': f'match-{match_id}'})
        self.assertEqual(refused.status_code, 404)
        self.assertIn('full match', refused.get_json()['error'])
        self.assertEqual(self.s.get('/api/coaching/frame', run=f'match-{match_id}', frame=5, w=640).status_code, 404)
        self.assertEqual(self.s.get('/api/coaching', match='20990101-000000-ffffff').status_code, 404)
        self.assertEqual(self.s.get('/api/coaching', match=match_id, team='C').status_code, 400)

    def test_the_key_moments_routes_build_and_serve_a_match_report(self):
        # The fake workers' match has people without pitch positions and no ball: the key-moments engine builds a
        # report with nothing to show (never a guess), on a background thread, and serves it from its cache after.
        r = self.s.post('/api/match/start', dict(BODY, workers=2, discovery='off'))
        match_id = r.get_json()['matchId']
        self.s.wait_state({'complete'})
        first = self.s.get('/api/coaching/moments', match=match_id, team='A')
        self.assertEqual(first.status_code, 200, first.get_data(as_text=True))
        self.assertEqual(first.get_json()['state'], 'missing')
        self.assertEqual(set(first.get_json()['teams']), {'A', 'B'})   # the team buttons are named before a report exists
        build = self.s.post(f'/api/coaching/moments/build?match={match_id}', {'team': 'A'})
        self.assertEqual(build.status_code, 200, build.get_data(as_text=True))
        self.assertIn(build.get_json()['state'], ('building', 'ready'))
        deadline, answer = time.time()+60, None
        while time.time() < deadline:
            answer = self.s.get('/api/coaching/moments', match=match_id, team='A').get_json()
            if answer['state'] != 'building':
                break
            time.sleep(.2)
        self.assertEqual(answer['state'], 'ready', answer)
        report = answer['report']
        self.assertEqual((report['kind'], report['team'], report['match']['matchId']), ('coaching-moments', 'A', match_id))
        self.assertEqual((report['strengths'], report['improvements']), ([], []))
        self.assertTrue(report['coverage']['items'])
        again = self.s.post(f'/api/coaching/moments/build?match={match_id}', {'team': 'A'}).get_json()
        self.assertEqual(again['state'], 'ready')                  # cached: no second build
        self.assertEqual(self.s.get('/api/coaching/moments', match=match_id, team='C').status_code, 400)
        self.assertEqual(self.s.get('/api/coaching/moments', match='20990101-000000-ffffff', team='A').status_code, 404)


if __name__ == '__main__':
    unittest.main()
