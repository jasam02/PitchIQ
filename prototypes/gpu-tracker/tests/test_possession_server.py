"""The possession route of server.py through Flask's test client, on a scratch folder (as test_coaching_server): server.py
is loaded with stand-ins for pipeline (no torch, no models, no GPU) and download_model, its ROOT is a temporary folder
(Video() makes its run folders there, never under the repository's runs/), and soccer/possession.py is a stand-in
engine whose compute() returns a record in the contract's shape and counts its calls (the real engine is tested by its
own tests; this file checks the route around it).

What is checked: /possession.js is served; the record is computed from the whole saved result once per version of the
file and served as compute() returned it; a correction (which rewrites result.json) makes the next request compute
again from the corrected result; one computation runs at a time, so two requests for the same version compute once; the
record is kept per run, nothing is served while a tracking run is processing or for a run the page does not show; a
possession module that cannot be imported, a computation that fails and a record that is not plain JSON (a NaN) answer
503 on this route only, are never cached, and every other route keeps working; the route has the Host and Origin
protections of the other API routes.

Run:  python -m unittest test_possession_server          (from this directory)
"""
import contextlib
import hashlib
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
import corrections  # noqa: E402
from test_coaching_store import synthetic_result, write_video  # noqa: E402

PORT = 8765
BASE = f'http://127.0.0.1:{PORT}'
RUN = '20261019-230000-abcdef'


def load_server(root, gate):
    """server.py as a private module: pipeline and download_model are stand-ins while it is imported (the real
    modules, if loaded, are put back), so nothing loads torch or a model. A tracking run waits for `gate`."""
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
        spec = importlib.util.spec_from_file_location('possession_test_server', ROOT/'server.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
    return module


class Engine:
    """A stand-in soccer/possession.py: compute() returns a contract-shaped record of the result it is given (one
    control run per second, alternating teams, the first frame's carrier the first player of that team) and keeps the
    results it was given. `delay` holds each computation that long; `fail` raises; `nan` puts a NaN in the record."""

    def __init__(self, delay=0.0, fail=None, nan=False):
        self.calls, self.delay, self.fail, self.nan = [], delay, fail, nan

    def compute(self, result, settings=None):
        self.calls.append(result)
        if self.delay:
            time.sleep(self.delay)
        if self.fail:
            raise self.fail
        frames = result.get('frames') or []
        if not frames:
            return {'version': 'stand-in', 'fps': 30.0, 't0': None, 't1': None, 'available': False, 'reason': 'No frames in this result.',
                    'quality': {'level': 'unavailable', 'coverage': 0.0, 'notes': []}, 'shares': {'A': 0, 'B': 0, 'dead': 0, 'unknown': 0},
                    'timeline': [], 'changes': [], 'deadSpells': []}
        first = {t: sorted(p['id'] for p in result['match']['players'] if p['team'] == t)[0] for t in ('A', 'B')}
        runs, k = [], 0
        while k < len(frames):
            end = min(len(frames), k+30)
            side = 'AB'[len(runs) % 2]
            runs.append({'t0': frames[k]['time'], 't1': frames[end-1]['time'], 'f0': frames[k]['frame'], 'f1': frames[end-1]['frame'], 'team': side,
                         'state': 'control', 'carrier': first[side], 'estimated': False, 'confidence': 'high'})
            k = end
        changes = [{'id': f'chg-{i:04d}', 'lostAt': runs[i-1]['t1'], 'wonAt': r['t0'], 'frame': r['f0'], 'fromTeam': runs[i-1]['team'], 'toTeam': r['team'],
                    'lostBy': runs[i-1]['carrier'], 'wonBy': r['carrier'], 'cause': 'tackle_or_duel', 'confidence': 'medium'} for i, r in enumerate(runs) if i]
        seconds = {t: sum(r['t1']-r['t0'] for r in runs if r['team'] == t) for t in ('A', 'B')}
        return {'version': 'stand-in', 'fps': 30.0, 't0': frames[0]['time'], 't1': frames[-1]['time'], 'available': True, 'reason': None,
                'quality': {'level': 'good', 'coverage': float('nan') if self.nan else 1.0, 'notes': []},
                'shares': {'A': seconds['A'], 'B': seconds['B'], 'dead': 0.0, 'unknown': 0.0}, 'timeline': runs, 'changes': changes, 'deadSpells': []}


class Scratch:
    """A scratch ROOT with an uploaded clip, one completed run with a synthetic result, and the server's app."""

    def __init__(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        for name in ('possession.js', 'index.html'):
            shutil.copyfile(ROOT/name, self.root/name)
        (self.root/'uploads').mkdir()
        self.video = write_video(self.root/'uploads'/'clip.avi', frames=400)
        self.key = hashlib.sha256(self.video.read_bytes()).hexdigest()
        self.folder = self.root/'runs'/self.key[:16]
        self.run_dir = self.folder/RUN
        self.run_dir.mkdir(parents=True)
        self.result = synthetic_result(seconds=6, first_frame=30, width=64, height=36)
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

    def get(self, path, client=None, **params):
        return (client or self.client).get(path, query_string=params, base_url=BASE)

    def engine(self, engine):
        """Use `engine` as soccer/possession.py until the test ends."""
        return mock.patch.object(self.server, 'possession_engine', lambda: engine)


class PossessionRoute(unittest.TestCase):
    def setUp(self):
        self.s = Scratch()
        self.addCleanup(self.s.tmp.cleanup)
        self.addCleanup(self.s.gate.set)

    def record(self, **params):
        r = self.s.get('/api/possession', **dict({'run': RUN}, **params))
        self.assertEqual(r.status_code, 200, r.get_data(as_text=True)[:400])
        return r.get_json()

    def test_the_script_is_served(self):
        with self.s.get('/possession.js') as r:
            self.assertEqual((r.status_code, r.mimetype), (200, 'application/javascript'))
            self.assertEqual(r.data, (ROOT/'possession.js').read_bytes())
            self.assertEqual(r.headers['Cache-Control'], 'no-store')

    def test_computed_once_from_the_whole_result_and_served_as_given(self):
        engine = Engine()
        with self.s.engine(engine):
            r = self.s.get('/api/possession', run=RUN)
            self.assertEqual((r.status_code, r.mimetype, r.headers['Cache-Control']), (200, 'application/json', 'no-store'))
            first = r.get_json()
            self.assertEqual(first, Engine().compute(self.s.result), 'the record exactly as compute() returned it')
            self.assertEqual(engine.calls[0], self.s.result, 'computed from the whole saved result (frames with people and ball)')
            self.assertTrue(engine.calls[0]['frames'][0]['people'])
            for _ in range(3):
                self.assertEqual(self.record(), first)
            self.assertEqual(len(engine.calls), 1, 'one computation per version of the result')
            self.assertEqual(self.s.get('/api/possession').get_json(), first, 'without a run: the shown run')
            self.assertEqual(len(engine.calls), 1)

    def test_a_correction_computes_again_from_the_corrected_result(self):
        engine = Engine()
        with self.s.engine(engine):
            before = self.record()
            # What a correction does to the run (corrections.rebuild): result.json rewritten with the correction applied,
            # here "B-07 is shown as B-04" and the correction listed, by the writer corrections use.
            corrected = json.loads(self.s.bytes)
            for e in corrected['match']['players']:
                if e['id'] == 'B-07':
                    e['display'] = 'B-04'
            corrected['corrections'] = [{'id': 'c20261019-231500-abcdef', 'applied': True}]
            corrections._write(self.s.run_dir/'result.json', corrected)
            after = self.record()
            self.assertEqual(len(engine.calls), 2)
            self.assertEqual(engine.calls[1]['corrections'], corrected['corrections'], 'the corrected result')
            self.assertEqual(next(e['display'] for e in engine.calls[1]['match']['players'] if e['id'] == 'B-07'), 'B-04')
            self.assertEqual(after, before, 'the stand-in keys on ids, so the record is the same; the page relabels it')
            self.record()
            self.assertEqual(len(engine.calls), 2, 'cached again')
            # Undoing the correction rewrites the original bytes: another version, computed again.
            corrections._write(self.s.run_dir/'result.json', json.loads(self.s.bytes))
            self.record()
            self.assertEqual((len(engine.calls), engine.calls[2]['corrections']), (3, []))

    def test_one_computation_at_a_time(self):
        engine = Engine(delay=.4)
        out = []

        def ask():
            client = self.s.app.test_client()
            r = self.s.get('/api/possession', client=client, run=RUN)
            out.append((r.status_code, r.get_data()))
        with self.s.engine(engine):
            threads = [threading.Thread(target=ask) for _ in range(3)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(20)
        self.assertEqual([code for code, _ in out], [200, 200, 200])
        self.assertEqual(len({body for _, body in out}), 1)
        self.assertEqual(len(engine.calls), 1, 'the second and third requests waited and read the cache')

    def test_kept_per_run_and_refused_while_processing_or_for_another_run(self):
        engine = Engine()
        with self.s.engine(engine):
            first = self.record()
            self.assertEqual(self.s.client.post('/api/run', json={'start': 0, 'seconds': 5}, base_url=BASE).status_code, 200)
            self.assertEqual(self.s.get('/api/status').get_json()['state'], 'running')
            r = self.s.get('/api/possession', run=RUN)
            self.assertEqual(r.status_code, 409)
            self.assertIn('Processing is running', r.get_json()['error'])
            self.s.gate.set()
            end = time.time()+20
            while self.s.get('/api/status').get_json()['state'] == 'running' and time.time() < end:
                time.sleep(.02)
            newer = self.s.get('/api/status').get_json()['resultRun']
            self.assertNotEqual(newer, RUN)
            old = self.s.get('/api/possession', run=RUN)
            self.assertEqual(old.status_code, 409)
            self.assertIn('older result than the latest run', old.get_json()['error'])
            record = self.record(run=newer)
            self.assertEqual((record['available'], record['reason']), (False, 'No frames in this result.'), 'the newer run, computed for its own result')
            self.assertEqual(len(engine.calls), 2)
            self.assertEqual(self.record(run=newer), record)
            self.assertEqual(len(engine.calls), 2)
        self.assertNotEqual(first, record)

    def test_bad_requests_and_the_request_guard(self):
        with self.s.engine(Engine()):
            r = self.s.get('/api/possession', run='another-run')
            self.assertEqual((r.status_code, r.get_json()['error']), (409, 'The page shows an older result than the latest run: reload the page.'))
            self.assertEqual(self.s.client.get('/api/possession', query_string={'run': RUN}, base_url='http://evil.example').status_code, 403)
            self.assertEqual(self.s.client.get('/api/possession', query_string={'run': RUN}, base_url=BASE, headers={'Origin': 'http://evil.example'}).status_code, 403)
            self.assertEqual(self.s.client.get('/api/possession', query_string={'run': RUN}, base_url=BASE, headers={'Origin': BASE}).status_code, 200)
            self.assertEqual(self.s.client.post('/api/possession', json={'run': RUN}, base_url=BASE).status_code, 405)
            (self.s.run_dir/'result.json').unlink()
            gone = self.s.get('/api/possession', run=RUN)
            self.assertEqual((gone.status_code, gone.get_json()['error']), (404, 'There is no result to judge possession on yet.'))

    def test_a_broken_possession_module_fails_only_its_route(self):
        source = (ROOT/'server.py').read_text(encoding='utf-8')
        self.assertNotRegex(source, r'(?m)^from soccer import .*\bpossession\b|^import soccer\.possession|^from soccer\.possession', 'not imported at start')
        others = ('/api/info', '/api/status', '/api/result', '/possession.js', '/')
        stderr = contextlib.redirect_stderr(io.StringIO())
        with stderr, mock.patch.object(self.s.server, 'possession_engine', mock.Mock(side_effect=ImportError('broken possession module'))):
            r = self.s.get('/api/possession', run=RUN)
            self.assertEqual(r.status_code, 503)
            self.assertEqual(r.get_json()['error'], 'Possession tracking is not available in this version: broken possession module')
            for path in others:
                with self.s.get(path) as ok:
                    self.assertEqual(ok.status_code, 200, path)
            self.assertEqual(self.s.get('/api/analytics', player='A-03').status_code, 200)
        # The real lazy import: a module that cannot be imported (None in sys.modules stops importlib) is the same 503.
        with contextlib.redirect_stderr(io.StringIO()), mock.patch.dict(sys.modules, {'soccer.possession': None}):
            r = self.s.get('/api/possession', run=RUN)
            self.assertEqual(r.status_code, 503)
            self.assertIn('Possession tracking is not available in this version', r.get_json()['error'])
        with contextlib.redirect_stderr(io.StringIO()), self.s.engine(Engine(fail=RuntimeError('boom'))):
            r = self.s.get('/api/possession', run=RUN)
            self.assertEqual((r.status_code, r.get_json()['error']), (503, 'The possession record could not be computed: boom'))
        nan = Engine(nan=True)
        with self.s.engine(nan):
            r = self.s.get('/api/possession', run=RUN)
            self.assertEqual(r.status_code, 503)
            self.assertIn('The possession record could not be served', r.get_json()['error'])
            for token in ('NaN', 'Infinity'):
                self.assertNotIn(token, r.get_data(as_text=True))
            self.s.get('/api/possession', run=RUN)
            self.assertEqual(len(nan.calls), 2, 'a record that could not be served is not cached')
        good = Engine()
        with self.s.engine(good):
            self.assertEqual(self.record(), Engine().compute(self.s.result), 'and served once the module works')
            self.assertEqual(len(good.calls), 1, 'no failure was cached')


if __name__ == '__main__':
    unittest.main()
