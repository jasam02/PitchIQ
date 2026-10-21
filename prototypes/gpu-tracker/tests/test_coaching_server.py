"""The coaching routes of server.py through Flask's test client, on a scratch folder: server.py is loaded with stand-ins
for pipeline (no torch, no models, no GPU) and download_model, its ROOT is a temporary folder (Video() makes its run
folders there, never under the repository's runs/), the source video is a small MJPG clip whose frame k is a flat grey
of (k % 32) * 8, and the report builder is the contract-shaped fixture (soccer.coaching.build_report_for is the engine's).

What is checked: a report is built once on a worker thread, cached under its fingerprint and served as ready; one build
runs at a time; nothing is built or served while a tracking run is processing or for a run the page does not show; a
team-shape change or a correction makes the stored report stale with the reasons, and undoing the correction finds it
again; screenshots are served only for evidence frames, and are those exact frames; the coach's feedback is stored per
video, replaced per finding and run, never written over an unreadable file; the PDF comes from the stored report with
the current display labels.

Run:  python -m unittest test_coaching_server          (from this directory)
"""
import contextlib
import importlib.util
import io
import json
import shutil
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
from soccer.coaching import store  # noqa: E402
from test_coaching_store import fixture_report, grey_of, synthetic_result, write_video  # noqa: E402

PORT = 8765
BASE = f'http://127.0.0.1:{PORT}'
RUN = '20261018-190000-abcdef'


def load_server(root, gate):
    """server.py as a private module: pipeline and download_model are stand-ins while it is imported (the real
    modules, if loaded, are put back), so nothing loads torch or a model."""
    import cv2

    def metadata(path):
        capture = cv2.VideoCapture(str(path))
        try:
            fps, count = float(capture.get(cv2.CAP_PROP_FPS)), int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            return {'name': Path(path).name, 'width': int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), 'height': int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                    'fps': fps, 'frames': count, 'duration': count/fps}
        finally:
            capture.release()

    def run(source, destination, config, progress, cancelled, previous):
        Path(destination).mkdir(parents=True, exist_ok=True)      # as pipeline.run does
        progress({'stage': 'Tracking'})
        gate.wait(20)
        return {'version': 3, 'status': 'complete', 'frames': [], 'match': {'players': []}}
    pipeline = types.ModuleType('pipeline')
    pipeline.ROOT, pipeline.RESOURCES, pipeline.metadata, pipeline.run = root, types.SimpleNamespace(load=lambda *a: None), metadata, run
    pipeline.gpu_info = lambda: {'name': 'test', 'torch': '-', 'cuda': '-', 'memoryGB': 0}
    download = types.ModuleType('download_model')
    download.get_model = lambda: None
    saved = {k: sys.modules.get(k) for k in ('pipeline', 'download_model')}
    sys.modules.update(pipeline=pipeline, download_model=download)
    try:
        spec = importlib.util.spec_from_file_location('coaching_test_server', ROOT/'server.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
    return module


class Scratch:
    """A scratch ROOT with an uploaded clip, one completed run with a synthetic result, and the server's app."""

    def __init__(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        shutil.copyfile(ROOT/'coaching.js', self.root/'coaching.js')
        (self.root/'uploads').mkdir()
        self.video = write_video(self.root/'uploads'/'clip.avi', frames=400)
        import hashlib
        self.key = hashlib.sha256(self.video.read_bytes()).hexdigest()
        self.folder = self.root/'runs'/self.key[:16]
        self.run_dir = self.folder/RUN
        self.run_dir.mkdir(parents=True)
        self.result = synthetic_result(seconds=12, first_frame=30, width=64, height=36)
        self.bytes = json.dumps(self.result, separators=(',', ':'), allow_nan=False).encode('utf-8')
        (self.run_dir/'result.json').write_bytes(self.bytes)
        (self.folder/'latest.json').write_text(json.dumps({'run': RUN}), encoding='utf-8')
        self.gate = threading.Event()
        self.server = load_server(self.root, self.gate)
        captured = []
        argv = ['server.py', '--video', str(self.video), '--port', str(PORT)]
        with mock.patch.object(sys, 'argv', argv), mock.patch.object(self.server, 'serve', lambda app, **kw: captured.append(app)), \
                mock.patch.object(self.server, 'preload_models', lambda: None), contextlib.redirect_stdout(io.StringIO()):
            self.server.main()
        self.app = captured[0]
        self.client = self.app.test_client()

    def get(self, path, **params):
        return self.client.get(path, query_string=params, base_url=BASE)

    def post(self, path, body):
        return self.client.post(path, json=body, base_url=BASE)

    def wait(self, timeout=20, **params):
        params = dict({'run': RUN, 'team': 'A'}, **params)
        end = time.time()+timeout
        while True:
            data = self.get('/api/coaching', **params).get_json()
            if data.get('state') != 'building' or time.time() > end:
                return data
            time.sleep(.02)

    def idle(self):
        """Wait until no build runs (a test's build must not leak into the next)."""
        end = time.time()+20
        while time.time() < end:
            data = self.get('/api/coaching', run=RUN, team='A').get_json() or {}
            if data.get('state') != 'building' and not data.get('busy'):
                return
            time.sleep(.02)


def fixture_builder(calls=None):
    def build(path, team, options):
        if calls is not None:
            calls.append((str(path), team, options))
        return fixture_report(json.loads(Path(path).read_text(encoding='utf-8')), team, options['context'])
    return build


class CoachingRoutes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.s = Scratch()

    @classmethod
    def tearDownClass(cls):
        cls.s.tmp.cleanup()

    def setUp(self):
        s = self.s
        s.idle()
        (s.run_dir/'result.json').write_bytes(s.bytes)
        shutil.rmtree(s.run_dir/'coaching', ignore_errors=True)
        shutil.rmtree(s.folder/'coaching', ignore_errors=True)
        for name in ('team_shape.json', 'coaching_feedback.json'):
            (s.folder/name).unlink(missing_ok=True)
        self.calls = []
        patcher = mock.patch.object(store, 'build_report_for', fixture_builder(self.calls))
        patcher.start()
        self.addCleanup(patcher.stop)

    def build(self, **params):
        body = dict({'run': RUN, 'team': 'A'}, **params)
        r = self.s.post('/api/coaching/build', body)
        self.assertEqual(r.status_code, 200, r.get_json())
        return self.s.wait(**{k: v for k, v in body.items() if k != 'run'})

    def test_the_script_is_served(self):
        with self.s.get('/coaching.js') as r:
            self.assertEqual((r.status_code, r.mimetype), (200, 'application/javascript'))
            self.assertEqual(r.data, (ROOT/'coaching.js').read_bytes())
            self.assertEqual(r.headers['Cache-Control'], 'no-store')

    def test_built_once_cached_and_served_ready(self):
        first = self.s.get('/api/coaching', run=RUN, team='A').get_json()
        self.assertEqual((first['state'], first['report'], first['fingerprint'], first['feedback']), ('missing', None, None, []))
        self.assertRegex(first['current'], r'^[0-9a-f]{64}$')
        started = self.s.post('/api/coaching/build', {'run': RUN, 'team': 'A'}).get_json()
        self.assertEqual((started['state'], started['fingerprint']), ('building', first['current']))
        data = self.s.wait()
        self.assertEqual((data['state'], data['fingerprint'], data['current'], data['reasons']), ('ready', first['current'], first['current'], []))
        self.assertEqual(data['report']['fingerprint'], first['current'])
        self.assertEqual(data['options'], {'team': 'A', 't0': None, 't1': None, 'context': 'competitive'})
        path = self.s.run_dir/'coaching'/f"report-A-{first['current'][:16]}.json"
        record = json.loads(path.read_text(encoding='utf-8'))
        self.assertEqual((record['fingerprint'], record['corrections']), (first['current'], []))
        self.assertEqual(record['inputs']['options'], data['options'])
        self.assertEqual(len(self.calls), 1)
        result_path, team, options = self.calls[0]
        self.assertEqual((Path(result_path), team), (self.s.run_dir/'result.json', 'A'))
        self.assertEqual({k: options[k] for k in ('t0', 't1', 'context', 'run', 'fingerprint')}, {'t0': None, 't1': None, 'context': 'competitive', 'run': RUN, 'fingerprint': first['current']})
        self.assertEqual(options['settings'], {'direction': {'A': None, 'switchAt': None}, 'lineup': {'A': None, 'B': None}, 'formations': []})
        self.assertEqual(Path(options['source']).resolve(), self.s.video.resolve())
        self.assertEqual(options['video']['key'], self.s.key[:16])
        self.assertIsNotNone(options['cache'], 'the server\'s parsed samples come from the same bytes')
        again = self.s.post('/api/coaching/build', {'run': RUN, 'team': 'A'}).get_json()
        self.assertEqual(again, {'ok': True, 'state': 'ready', 'fingerprint': first['current']})
        self.assertEqual(len(self.calls), 1, 'an up-to-date report is not built again')
        other = self.build(team='B', context='advanced', t0=3, t1=9)
        self.assertEqual((other['state'], other['options']), ('ready', {'team': 'B', 't0': 3.0, 't1': 9.0, 'context': 'advanced'}))
        self.assertEqual(self.s.wait()['state'], 'ready', 'other options are another report')

    def test_bad_requests(self):
        for params, status in [({'team': 'C'}, 400), ({'context': 'pro'}, 400), ({'t0': 2, 't1': 4}, 400), ({'t0': 'x', 't1': 9}, 400),
                               ({'run': 'another-run'}, 409)]:
            r = self.s.get('/api/coaching', **dict({'run': RUN, 'team': 'A'}, **params))
            self.assertEqual(r.status_code, status, params)
            self.assertIn('error', r.get_json())
        self.assertEqual(self.s.post('/api/coaching/build', ['A']).status_code, 400)
        self.assertEqual(self.s.post('/api/coaching/build', {'run': 'another-run', 'team': 'A'}).status_code, 409)
        # A run that is not a name is a bad request, not "an older result".
        for body in ({'run': [RUN], 'team': 'A'}, {'run': 5, 'team': 'A'}):
            r = self.s.post('/api/coaching/build', body)
            self.assertEqual((r.status_code, r.get_json()['error']), (400, 'run must be the name of the run the page shows.'), body)
        r = self.s.get('/api/coaching', run=RUN, team='A', t0=1, t1=3)
        self.assertIn('inside the analysed segment (00:01.00-00:12.97, source-video time)', r.get_json()['error'])
        self.assertEqual(self.s.client.post('/api/coaching/build', data='x', base_url=BASE).status_code, 415, 'JSON only (the request guard)')
        self.assertEqual(self.s.client.get('/api/coaching', base_url='http://evil.example').status_code, 403)
        self.assertEqual(self.calls, [])

    def test_one_build_at_a_time(self):
        release = threading.Event()
        slow = fixture_builder(self.calls)
        with mock.patch.object(store, 'build_report_for', lambda *a: (release.wait(20), slow(*a))[1]):
            self.assertEqual(self.s.post('/api/coaching/build', {'run': RUN, 'team': 'A'}).get_json()['state'], 'building')
            self.assertEqual(self.s.get('/api/coaching', run=RUN, team='A').get_json()['state'], 'building')
            other = self.s.get('/api/coaching', run=RUN, team='B').get_json()
            self.assertEqual((other['state'], other['busy']), ('missing', True))
            refused = self.s.post('/api/coaching/build', {'run': RUN, 'team': 'B'})
            self.assertEqual(refused.status_code, 409)
            self.assertIn('being built', refused.get_json()['error'])
            release.set()
            self.assertEqual(self.s.wait()['state'], 'ready')
        self.assertEqual(len(self.calls), 1)

    def test_failures_are_reported_and_nothing_is_stored(self):
        cases = [(mock.Mock(side_effect=RuntimeError('boom')), 'could not be built: boom'),
                 (mock.Mock(side_effect=store.EngineUnavailable('The coaching report engine is not available in this version yet.')), 'engine is not available'),
                 (mock.Mock(return_value=dict(fixture_report(self.s.result), summary=[float('nan')])), 'not finite')]

        def correcting(path, team, options):
            (self.s.run_dir/'result.json').write_bytes(self.s.bytes.replace(b'"corrections":[]', b'"corrections":[{"id":"c1","applied":true}]'))
            return fixture_report(self.s.result, team)
        cases.append((correcting, 'changed while the report was built'))
        for builder, message in cases:
            (self.s.run_dir/'result.json').write_bytes(self.s.bytes)
            with mock.patch.object(store, 'build_report_for', builder), contextlib.redirect_stderr(io.StringIO()):
                self.s.post('/api/coaching/build', {'run': RUN, 'team': 'A'})
                self.s.idle()
            data = self.s.get('/api/coaching', run=RUN, team='A').get_json()
            if message == 'changed while the report was built':
                # The result changed, so the failed build's fingerprint is not the current one: nothing to show.
                self.assertEqual(data['state'], 'missing')
            else:
                self.assertEqual(data['state'], 'error', message)
                self.assertIn(message, data['error'])
            self.assertIsNone(data['report'])
            self.assertEqual(list((self.s.run_dir/'coaching').glob('report-*.json')) if (self.s.run_dir/'coaching').exists() else [], [])
        (self.s.run_dir/'result.json').write_bytes(self.s.bytes)
        self.assertEqual(self.build()['state'], 'ready', 'a later build clears the error')

    def test_stale_with_reasons_and_an_undo_finds_the_report_again(self):
        ready = self.build()
        r = self.s.post('/api/team-shape/settings', {'direction': {'A': 'left'}})
        self.assertEqual(r.status_code, 200)
        stale = self.s.get('/api/coaching', run=RUN, team='A').get_json()
        self.assertEqual((stale['state'], stale['reasonCodes'], stale['fingerprint']), ('stale', ['settings'], ready['fingerprint']))
        self.assertNotEqual(stale['current'], ready['fingerprint'])
        self.assertIsNotNone(stale['report'], 'a stale report is still served')
        corrected = self.s.bytes.replace(b'"corrections":[]', b'"corrections":[{"id":"c20261018-190500-abcdef","applied":true}]')
        (self.s.run_dir/'result.json').write_bytes(corrected)
        stale = self.s.get('/api/coaching', run=RUN, team='A').get_json()
        self.assertEqual(stale['reasonCodes'], ['result', 'settings'])
        self.assertIn('1 correction made since this report', stale['reasons'][0])
        (self.s.run_dir/'result.json').write_bytes(self.s.bytes)
        (self.s.folder/'team_shape.json').unlink()
        back = self.s.get('/api/coaching', run=RUN, team='A').get_json()
        self.assertEqual((back['state'], back['fingerprint']), ('ready', ready['fingerprint']), 'the same bytes: the cached report again')
        self.assertEqual(len(self.calls), 1)

    def test_screenshots_only_of_evidence_frames(self):
        self.assertEqual(self.s.get('/api/coaching/frame', run=RUN, frame=100, w=640).status_code, 404, 'no report yet')
        data = self.build()
        frames = sorted({e['frame'] for e in data['report']['evidence']})
        for frame in frames:
            r = self.s.get('/api/coaching/frame', run=RUN, frame=frame, w=640)
            self.assertEqual((r.status_code, r.mimetype), (200, 'image/jpeg'))
            self.assertAlmostEqual(grey_of(r.data), (frame % 32)*8, delta=3, msg=f'the absolute source frame {frame}')
            self.assertTrue((self.s.folder/'coaching'/'frames'/f'f{frame}-w640.jpg').exists())
        self.assertEqual(self.s.get('/api/coaching/frame', run=RUN, frame=frames[0]+1, w=640).status_code, 404)
        self.assertEqual(self.s.get('/api/coaching/frame', run=RUN, frame=frames[0], w=999).status_code, 400)
        self.assertEqual(self.s.get('/api/coaching/frame', run=RUN, frame='x', w=640).status_code, 400)
        negative = self.s.get('/api/coaching/frame', run=RUN, frame=-1, w=640)
        self.assertEqual((negative.status_code, negative.get_json()['error']), (400, 'frame must be a source frame index (0 or more)'))
        self.assertEqual(self.s.get('/api/coaching/frame', run='another-run', frame=frames[0], w=640).status_code, 409)

    def test_feedback(self):
        data = self.build()
        gap = data['report']['improvements'][0]['observation']
        body = {'run': RUN, 'observation': gap, 'verdict': 'correct', 'text': 'Seen it too', 'fingerprint': data['fingerprint'], 'team': 'A'}
        r = self.s.post('/api/coaching/feedback', body)
        self.assertEqual(r.status_code, 200, r.get_json())
        entry = r.get_json()['feedback']
        self.assertEqual((entry['run'], entry['verdict'], entry['text'], entry['observation']['key'], entry['report']), (RUN, 'correct', 'Seen it too', gap, data['fingerprint'][:16]))
        served = self.s.get('/api/coaching', run=RUN, team='A').get_json()
        self.assertEqual(served['state'], 'ready', 'feedback never makes the report stale')
        self.assertEqual([(f['attached'], f['thisRun']) for f in served['feedback']], [(gap, True)])
        self.assertEqual([f['id'] for f in served['report']['feedback']], [entry['id']])
        replaced = self.s.post('/api/coaching/feedback', dict(body, verdict='partial', text='')).get_json()['feedback']
        stored = json.loads((self.s.folder/'coaching_feedback.json').read_text(encoding='utf-8'))['feedback']
        self.assertEqual([(f['id'], f['verdict']) for f in stored], [(replaced['id'], 'partial')], 'one verdict per finding and run')
        for change, status in [({'observation': 'obs-000000000000'}, 404), ({'observation': 'A-03'}, 400), ({'verdict': 'agree'}, 400),
                               ({'text': 'x'*(store.MAX_NOTE+1)}, 400), ({'text': 5}, 400), ({'run': 'another-run'}, 409),
                               ({'observation': gap+'\n'}, 400), ({'run': [RUN]}, 400)]:
            self.assertEqual(self.s.post('/api/coaching/feedback', dict(body, **change)).status_code, status, change)
        self.assertEqual(self.s.post('/api/coaching/feedback', dict(body, text=5)).get_json()['error'], 'text must be the note as a string.')
        self.assertEqual(self.s.post('/api/coaching/feedback', dict(body, verdict='agree')).get_json()['error'],
                         'verdict must be correct, partial, incorrect or insufficient.')
        self.assertEqual(self.s.post('/api/coaching/feedback/remove', {'id': replaced['id']}).get_json(), {'ok': True, 'removed': replaced['id']})
        self.assertEqual(self.s.post('/api/coaching/feedback/remove', {'id': replaced['id']}).status_code, 404)
        self.assertEqual(self.s.post('/api/coaching/feedback/remove', {'id': '../x'}).status_code, 400)
        self.assertEqual(self.s.post('/api/coaching/feedback/remove', {'id': replaced['id']+'\n'}).status_code, 400)
        self.assertEqual(self.s.get('/api/coaching', run=RUN, team='A').get_json()['feedback'], [])

    def test_feedback_from_another_run_is_listed_apart_never_on_a_finding(self):
        data = self.build()
        gap = data['report']['improvements'][0]['observation']
        mine = self.s.post('/api/coaching/feedback', {'run': RUN, 'observation': gap, 'verdict': 'correct', 'team': 'A'}).get_json()['feedback']
        # The same finding id given a verdict on an earlier run (ids leave the run out; letters are arbitrary per run).
        path = self.s.folder/'coaching_feedback.json'
        stored = json.loads(path.read_text(encoding='utf-8'))
        earlier = dict(json.loads(json.dumps(mine)), id='f20261018-120000-abcdef', run='20261017-120000-000000', verdict='incorrect',
                       text='Wrong team: this was the other side')
        # And this run's verdict on team B's report: it sits on no finding of team A's report.
        other_team = dict(json.loads(json.dumps(mine)), id='f20261018-120001-abcdef',
                          observation=dict(mine['observation'], key='obs-0123456789ab', team='B'))
        stored['feedback'] += [earlier, other_team]
        path.write_text(json.dumps(stored), encoding='utf-8')
        served = self.s.get('/api/coaching', run=RUN, team='A').get_json()
        by_id = {f['id']: f for f in served['feedback']}
        self.assertEqual((by_id[mine['id']]['attached'], by_id[mine['id']]['thisRun']), (gap, True))
        self.assertEqual((by_id[earlier['id']]['attached'], by_id[earlier['id']]['thisRun']), (None, False))
        self.assertEqual((by_id[other_team['id']]['attached'], by_id[other_team['id']]['thisRun']), (None, True))
        self.assertEqual([(f['id'], f['thisRun'], f['attached']) for f in served['report']['feedback']],
                         [(mine['id'], True, gap), (earlier['id'], False, None)],
                         'the report (and its PDF) carries the verdicts on its findings and, apart, those of earlier runs')
        seen = {}
        with mock.patch.object(store, 'render_pdf', lambda report, labels, frame, stale=None: (seen.update(feedback=report['feedback']), b'%PDF-1.4')[1]):
            self.assertEqual(self.s.get('/api/coaching/pdf', run=RUN, team='A').status_code, 200)
        self.assertEqual([f['id'] for f in seen['feedback']], [mine['id'], earlier['id']])

    def test_responses_never_carry_a_non_finite_number(self):
        data = self.build()
        gap = data['report']['improvements'][0]['observation']
        self.s.post('/api/coaching/feedback', {'run': RUN, 'observation': gap, 'verdict': 'correct', 'team': 'A'})
        path = self.s.folder/'coaching_feedback.json'
        text = path.read_text(encoding='utf-8')
        path.write_text(text.replace('"t0": ', '"t0": NaN, "was": ', 1).replace('"verdict"', '"score": Infinity, "verdict"', 1), encoding='utf-8')
        r = self.s.get('/api/coaching', run=RUN, team='A')
        self.assertEqual(r.status_code, 200)
        body = r.get_data(as_text=True)
        for token in ('NaN', 'Infinity'):
            self.assertNotIn(token, body)
        served = json.loads(body, parse_constant=lambda c: self.fail(f'{c} in the response'))
        self.assertEqual(served['feedback'][0]['observation']['t0'], None)
        self.assertEqual(served['feedback'][0]['attached'], gap, 'the entry itself is kept')

    def test_a_broken_coaching_package_fails_only_its_routes(self):
        source = (ROOT/'server.py').read_text(encoding='utf-8')
        self.assertNotRegex(source, r'(?m)^from soccer\.coaching|^import soccer\.coaching', 'not imported at start')
        with mock.patch.object(self.s.server, 'coaching_store', mock.Mock(side_effect=ImportError('broken coaching package'))), \
                contextlib.redirect_stderr(io.StringIO()):
            for path in ('/api/coaching', '/api/coaching/frame', '/api/coaching/pdf'):
                r = self.s.get(path, run=RUN, team='A', frame=1)
                self.assertEqual(r.status_code, 503, path)
                self.assertIn('broken coaching package', r.get_json()['error'])
            for path in ('/api/coaching/build', '/api/coaching/feedback', '/api/coaching/feedback/remove'):
                self.assertEqual(self.s.post(path, {'run': RUN, 'team': 'A'}).status_code, 503, path)
            for path in ('/api/info', '/api/status', '/api/result', '/coaching.js'):
                with self.s.get(path) as r:
                    self.assertEqual(r.status_code, 200, path)
        self.assertEqual(self.s.get('/api/coaching', run=RUN, team='A').status_code, 200, 'and back once the package loads')

    def test_an_unreadable_feedback_file_is_left_alone(self):
        data = self.build()
        path = self.s.folder/'coaching_feedback.json'
        path.write_text('{"version": 1, "feedback": [{"id"', encoding='utf-8')
        body = {'run': RUN, 'observation': data['report']['improvements'][0]['observation'], 'verdict': 'correct'}
        r = self.s.post('/api/coaching/feedback', body)
        self.assertEqual(r.status_code, 503)
        self.assertIn('nothing was changed', r.get_json()['error'])
        self.assertEqual(self.s.post('/api/coaching/feedback/remove', {'id': 'f20261018-190312-abcdef'}).status_code, 503)
        self.assertEqual(path.read_text(encoding='utf-8'), '{"version": 1, "feedback": [{"id"')
        self.assertEqual(self.s.get('/api/coaching', run=RUN, team='A').get_json()['feedback'], [])

    def test_pdf_of_the_stored_report(self):
        self.assertEqual(self.s.get('/api/coaching/pdf', run=RUN, team='A').status_code, 404, 'generate first')
        data = self.build()
        with mock.patch.object(store, 'render_pdf', mock.Mock(side_effect=store.PdfUnavailable('PDF export is not available in this version yet.'))):
            self.assertEqual(self.s.get('/api/coaching/pdf', run=RUN, team='A').status_code, 503)
        seen = {}

        def render(report, labels, frame, stale=None):
            evidence = report['evidence'][0]['frame']
            seen.update(fingerprint=report['fingerprint'], labels=labels, stale=stale, shot=frame(evidence), other=frame(evidence+1))
            return b'%PDF-1.4\n%fake\n'
        with mock.patch.object(store, 'render_pdf', render):
            r = self.s.get('/api/coaching/pdf', run=RUN, team='A')
        self.assertEqual((r.status_code, r.mimetype, r.data), (200, 'application/pdf', b'%PDF-1.4\n%fake\n'))
        self.assertIn('attachment', r.headers['Content-Disposition'])
        self.assertIn(f'pitchiq-coaching-clip-{RUN}-A-competitive.pdf', r.headers['Content-Disposition'], 'the level is in the name')
        self.assertEqual((seen['fingerprint'], seen['stale'], seen['other']), (data['fingerprint'], None, None))
        self.assertEqual((seen['labels']['B-07'], seen['labels']['A-03']), ('B-05', 'A-02'), 'the display labels of the result as it is now')
        self.assertEqual(seen['shot'][:2], b'\xff\xd8')
        self.s.post('/api/team-shape/settings', {'direction': {'A': 'left'}})
        with mock.patch.object(store, 'render_pdf', render):
            self.assertEqual(self.s.get('/api/coaching/pdf', run=RUN, team='A').status_code, 200)
        self.assertEqual(len(seen['stale']), 1, 'a stale report is exported with its reasons')


class WhileATrackingRunProcesses(unittest.TestCase):
    def test_refused_until_the_run_has_finished(self):
        s = Scratch()
        try:
            with mock.patch.object(store, 'build_report_for', fixture_builder()):
                self.assertEqual(s.post('/api/run', {'start': 0, 'seconds': 5}).status_code, 200)
                self.assertEqual(s.get('/api/status').get_json()['state'], 'running')
                for r in (s.post('/api/coaching/build', {'run': RUN, 'team': 'A'}), s.get('/api/coaching', run=RUN, team='A'),
                          s.post('/api/coaching/feedback', {'run': RUN, 'observation': 'obs-000000000000', 'verdict': 'correct'})):
                    self.assertEqual(r.status_code, 409)
                    self.assertIn('Processing is running', r.get_json()['error'])
                s.gate.set()
                end = time.time()+20
                while s.get('/api/status').get_json()['state'] == 'running' and time.time() < end:
                    time.sleep(.02)
                status = s.get('/api/status').get_json()
                self.assertNotEqual(status['resultRun'], RUN)
                self.assertEqual(s.get('/api/coaching', run=RUN, team='A').status_code, 409, 'the page must reload the newer run')
        finally:
            s.gate.set()
            s.tmp.cleanup()


if __name__ == '__main__':
    unittest.main()
