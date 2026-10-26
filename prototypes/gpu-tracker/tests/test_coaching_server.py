"""The coaching routes of server.py through Flask's test client, on a scratch folder: server.py is loaded with stand-ins
for pipeline (no torch, no models, no GPU) and download_model, its ROOT is a temporary folder (Video() makes its run
folders there, never under the repository's runs/), and the source video is a small MJPG clip whose frame k is a flat
grey of (k % 32) * 8.

v2 (adaptive coaching; a request without a v1 'context' level), with the REAL engine on EVAL's synthetic short_segment
scene (tools/coaching_synth.py: 60 s, 10 fps, both teams 11v11): the measurement tier is built once for both teams on
the worker thread and a level, age, style or role change is answered at once from the stored obs file (the obs bytes
never change: invariant 2); a format change or coach-marked spans make the measurement stale with the reasons, and
going back finds the stored tier again; the settings are validated (nothing that the cleaning would drop is saved, an
unreadable file is never overwritten); coach spans are cleaned against the segment, kept per run, reach the timeline
and leave every tier stale; verdicts and adjusted areas attach by finding key and range only; squads, the report's
progression section, the progress route and the match date; the PDF gets the served report, its feedback and the
team colours; screenshots only of key moments; one build at a time across both engines; a broken v2 module fails only
the v2 routes; tools/coaching_report.py gives the same fingerprints and report as the server.

v1 (kept while the page's legacy mode and the v1 tools use it; requests name a context level), with the
contract-shaped fixture report: a report is built once on a worker thread, cached under its fingerprint and served as
ready; nothing is built or served while a tracking run is processing or for a run the page does not show; a
team-shape change or a correction makes the stored report stale with the reasons, and undoing the correction finds it
again; screenshots only of evidence frames; feedback per video, replaced per finding and run, never written over an
unreadable file; the PDF from the stored report with the current display labels.

Run:  python -m unittest test_coaching_server          (from this directory)
"""
import contextlib
import copy
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
for _p in (str(ROOT), str(ROOT/'tools')):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from soccer.coaching import profiles as PR, schema as S, store  # noqa: E402
from test_coaching_store import fixture_report, grey_of, synthetic_result, write_video  # noqa: E402

PORT = 8765
BASE = f'http://127.0.0.1:{PORT}'
RUN = '20261018-190000-abcdef'
V1 = 'competitive'


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
    """A scratch ROOT with an uploaded clip, one completed run with a result, and the server's app."""

    def __init__(self, result=None, fps=30.0):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        shutil.copyfile(ROOT/'coaching.js', self.root/'coaching.js')
        (self.root/'uploads').mkdir()
        self.result = result if result is not None else synthetic_result(seconds=12, first_frame=30, width=64, height=36)
        last = max(int(f['frame']) for f in self.result['frames'])
        self.video = write_video(self.root/'uploads'/'clip.avi', frames=max(400, last+5), fps=fps)
        import hashlib
        self.key = hashlib.sha256(self.video.read_bytes()).hexdigest()
        self.folder = self.root/'runs'/self.key[:16]
        self.run_dir = self.folder/RUN
        self.run_dir.mkdir(parents=True)
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

    def wait(self, timeout=60, **params):
        params = dict({'run': RUN, 'team': 'A'}, **params)
        end = time.time()+timeout
        while True:
            data = self.get('/api/coaching', **params).get_json()
            if data.get('state') != 'building' or time.time() > end:
                return data
            time.sleep(.02)

    def idle(self):
        """Wait until no build runs (a test's build must not leak into the next); asked through the v1 answer, which
        builds nothing."""
        end = time.time()+60
        while time.time() < end:
            data = self.get('/api/coaching', run=RUN, team='A', context=V1).get_json() or {}
            if data.get('state') != 'building' and not data.get('busy'):
                return
            time.sleep(.02)

    def clear(self):
        """Every coaching file of the scratch video and the squads / progress store (the result restored)."""
        (self.run_dir/'result.json').write_bytes(self.bytes)
        shutil.rmtree(self.run_dir/'coaching', ignore_errors=True)
        shutil.rmtree(self.folder/'coaching', ignore_errors=True)
        shutil.rmtree(self.root/'runs'/'_coaching', ignore_errors=True)
        for name in ('team_shape.json', 'coaching_feedback.json', 'coaching_settings.json', 'coach_spans.json',
                     'coaching_match.json'):
            (self.folder/name).unlink(missing_ok=True)


def fixture_builder(calls=None):
    def build(path, team, options):
        if calls is not None:
            calls.append((str(path), team, options))
        return fixture_report(json.loads(Path(path).read_text(encoding='utf-8')), team, options['context'])
    return build


# ===================================================== v2 =============================================================
_SCENE = {}


def scene():
    """EVAL's short_segment scene (built once per process): 60 s at 10 fps from 600.0 s, frames 6000-6599."""
    if 'built' not in _SCENE:
        import coaching_synth
        _SCENE['built'] = coaching_synth.build('short_segment')
    return _SCENE['built']


def block(level=4, age='U14', **change):
    """A team settings block on the scene's 11v11 format (a level change that keeps the format: interpretation only)."""
    raw = dict({'ageGroup': age, 'playingLevel': level, 'format': {'players': 11, 'goalkeeper': True}}, **change)
    return PR.clean_team(raw, now='2026-10-22T10:00:00')


class CoachingV2Routes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.s = Scratch(scene().result, fps=scene().result['video']['fps'])

    @classmethod
    def tearDownClass(cls):
        cls.s.tmp.cleanup()

    def setUp(self):
        self.s.idle()
        self.s.clear()
        self.measured = []
        real = store.measure

        def counted(*a, **kw):
            self.measured.append(kw.get('team'))
            return real(*a, **kw)
        patcher = mock.patch.object(store, 'measure', counted)
        patcher.start()
        self.addCleanup(patcher.stop)

    # ---------- helpers ----------
    def answer(self, team='A', **params):
        r = self.s.get('/api/coaching', run=RUN, team=team, **params)
        self.assertEqual(r.status_code, 200, r.get_json())
        return r.get_json()

    def build(self, team='A', **params):
        r = self.s.post('/api/coaching/build', dict({'run': RUN, 'team': team}, **params))
        self.assertEqual(r.status_code, 200, r.get_json())
        q = {k: v for k, v in params.items() if k in ('t0', 't1')}
        data = self.s.wait(team=team, **q)
        self.assertEqual(data['state'], 'ready', data.get('error'))
        return data

    def save(self, team, settings, status=200):
        r = self.s.post('/api/coaching/settings', {'run': RUN, 'team': team, 'settings': settings})
        self.assertEqual(r.status_code, status, r.get_json())
        return r.get_json()

    def tier_files(self, kind, team='A'):
        return sorted(store.tier_dir(self.s.run_dir).glob(f'{kind}-{team}-*.json'))

    def findings(self, report):
        return [f for k in store.FINDING_SECTIONS for f in report[k]]

    def draw_area(self, data, evidence):
        """Draw a suggested area on key moment `evidence` of the stored report record of answer `data`, as
        alternatives.py draws one (a suggestedZone element on the moment's diagram): the short_segment scene's moments
        have none, and only a moment with one takes an adjusted area (CODE L-6)."""
        fp = data['fingerprints']
        path = store.tier_path(self.s.run_dir, 'report', {'team': data['options']['team'], 'mfp': fp['measurement'],
                                                          'rfp': fp['roles'], 'ifp': fp['interpretation']})
        record = json.loads(path.read_text(encoding='utf-8'))
        moment = next(e for e in record['report']['keyMoments'] if e['id'] == evidence)
        zone = {'type': 'suggestedZone', 'polygon': [[.40, .30], [.48, .30], [.48, .40], [.40, .40]], 'kind': 'zone',
                'alternative': 'alt-0123456789ab', 'mover': (moment['players'] or ['A-01'])[0], 'label': 'possible covering area'}
        moment['diagram'] = dict(moment['diagram'] or {}, elements=list((moment['diagram'] or {}).get('elements') or [])+[zone])
        path.write_text(json.dumps(record), encoding='utf-8')
        self.assertTrue(store.has_suggested_area(record['report'], evidence))

    # ---------- tests ----------
    def test_profiles_and_settings_before_anything_is_saved(self):
        r = self.s.get('/api/coaching/profiles')
        self.assertEqual((r.status_code, r.mimetype), (200, 'application/json'))
        self.assertEqual(r.get_json(), json.loads(store.json_text(PR.catalog())))
        got = self.s.get('/api/coaching/settings', run=RUN).get_json()
        self.assertEqual(got['settings'], {'version': 1, 'teams': {'A': None, 'B': None}, 'updated': None})
        self.assertEqual((got['inferred'], got['squads'], got['links'], got['matchDate']), ({'A': [], 'B': []}, [], [], None))
        first = self.answer()
        self.assertEqual((first['engine'], first['state'], first['report']), (2, 'missing', None))
        self.assertEqual(first['tiers'], {'measurement': 'missing', 'role': 'missing', 'interpretation': 'missing'})
        self.assertEqual(self.measured, [], 'asking builds nothing while the measurement is missing')

    def test_measured_once_for_both_teams_and_a_level_change_is_immediate(self):
        data = self.build()
        self.assertEqual(self.measured, [None], 'one measurement for both teams')
        self.assertEqual(data['tiers'], {'measurement': 'ready', 'role': 'ready', 'interpretation': 'ready'})
        rep = data['report']
        self.assertEqual((rep['kind'], rep['version'], rep['team'], rep['run']), ('coaching-report', 2, 'A', RUN))
        self.assertEqual(S.validate('report', rep), [])
        self.assertEqual((rep['profile']['ageGroup'], rep['profile']['level']), ('Adult', 6), 'nothing saved: the defaults')
        self.assertEqual(rep['fingerprints'], data['fingerprints'])
        self.assertEqual(rep['generated'], data['saved'])
        self.assertEqual(data['options'], {'team': 'A', 't0': None, 't1': None})
        self.assertEqual(len(self.tier_files('obs', 'B')), 1, 'the other team was measured in the same pass')
        other = self.answer('B')
        self.assertEqual((other['state'], self.measured), ('ready', [None]), 'team B is interpreted at once')
        obs_bytes = self.tier_files('obs')[0].read_bytes()
        saved = self.save('A', block(2, 'U10'))
        self.assertEqual(saved['stale'], ['interpretation'])
        t = time.perf_counter()
        level2 = self.answer()
        took = time.perf_counter()-t
        self.assertEqual((level2['state'], level2['report']['profile']['level'], level2['report']['profile']['ageGroup']),
                         ('ready', 2, 'U10'))
        self.assertLess(took, 2.0, 'a level change is answered from the stored tiers')
        self.assertEqual(self.measured, [None], 'no new measurement for a level change')
        self.assertEqual(self.tier_files('obs')[0].read_bytes(), obs_bytes, 'invariant 2: the observations never change')
        self.assertEqual(len(self.tier_files('roles')), 1, 'the role tier is reused')
        self.assertEqual(level2['fingerprints']['measurement'], data['fingerprints']['measurement'])
        # A confirmed role: the role tier and the interpretation are made again, from the same observations.
        roles = self.save('A', dict(block(2, 'U10'), roles={'A-02': [{'role': 'LB', 'from': None, 'to': None}]}))
        self.assertEqual(roles['stale'], ['role', 'interpretation'])
        self.assertEqual(roles['settings']['teams']['A']['roles']['A-02'][0]['set'], roles['settings']['updated'],
                         'the save time is stamped on a new role confirmation')
        self.assertEqual(self.answer()['state'], 'ready')
        self.assertEqual((len(self.tier_files('roles')), self.measured), (2, [None]))
        # A format change needs the measurement again: the report on screen is the earlier one, stale, with the reason.
        nine = self.save('A', block(4, 'U14', format={'players': 9, 'goalkeeper': True}))
        self.assertEqual(nine['stale'], ['measurement', 'role', 'interpretation'])
        stale = self.answer()
        self.assertEqual((stale['state'], stale['tiers']['measurement']), ('stale', 'missing'))
        self.assertIn('format', stale['reasonCodes'])
        self.assertIn('The match format or field size changed.', stale['reasons'])
        self.assertIsNotNone(stale['report'])
        self.assertEqual(self.build()['report']['profile']['format']['players'], 9)
        self.assertEqual(self.measured, [None, 'A'], 'only the team whose format changed is measured again')
        self.save('A', block(4, 'U14'))
        back = self.answer()
        self.assertEqual((back['state'], back['tiers']['measurement'], self.measured), ('ready', 'ready', [None, 'A']),
                         'going back finds the stored measurement')

    def test_settings_are_validated_and_an_unreadable_file_is_left_alone(self):
        path = self.s.folder/'coaching_settings.json'
        for team, settings, needle in [
                ('A', dict(block(), playingLevel=9), 'playingLevel'), ('A', dict(block(), ageGroup='U99'), 'ageGroup'),
                ('A', dict(block(), roles={'X-1': [{'role': 'LB'}]}), 'roles'),
                ('A', dict(block(), squad={'id': 'sq-000000000000', 'name': 'Nobody'}), 'not a known squad'),
                ('A', dict(block(), runLinks=[{'run': RUN, 'team': 'A', 'confirmed': True}]), 'needs a squad'),
                ('A', dict(block(), field={'length': 500, 'width': 40}), 'field')]:
            r = self.s.post('/api/coaching/settings', {'run': RUN, 'team': team, 'settings': settings})
            self.assertEqual(r.status_code, 400, needle)
            self.assertTrue(any(needle in p for p in r.get_json()['problems']), r.get_json())
        for body, status in [({'run': RUN, 'team': 'C', 'settings': block()}, 400), ({'run': RUN, 'team': 'A', 'settings': [1]}, 400),
                             ({'run': 'another-run', 'team': 'A', 'settings': block()}, 409)]:
            self.assertEqual(self.s.post('/api/coaching/settings', body).status_code, status, body)
        self.assertFalse(path.exists(), 'nothing refused was written')
        ok = self.save('A', block(3, 'U12'))
        self.assertEqual(json.loads(path.read_text(encoding='utf-8')), ok['settings'])
        self.assertEqual(S.validate('settings', ok['settings']), [])
        path.write_text('{"version": 1, "teams": {', encoding='utf-8')
        r = self.s.post('/api/coaching/settings', {'run': RUN, 'team': 'B', 'settings': block()})
        self.assertEqual(r.status_code, 503)
        self.assertEqual(path.read_text(encoding='utf-8'), '{"version": 1, "teams": {', 'left as it was')
        self.assertEqual(self.s.get('/api/coaching/settings', run=RUN).get_json()['settings']['teams'], {'A': None, 'B': None},
                         'served as nothing saved')

    def test_coach_spans_are_kept_per_run_and_make_every_tier_stale(self):
        seg = self.s.get('/api/coaching/spans', run=RUN).get_json()
        self.assertEqual((seg['spans'], seg['saved']), (store.empty_spans(), None))
        lo, hi = seg['segment']['t0'], seg['segment']['t1']
        self.assertEqual((lo, hi), (600.0, 659.9))
        engine_only = self.build()
        rec = dict(store.empty_spans(), spans=[{'t0': lo-.3, 't1': lo+20, 'state': 'A'}, {'t0': lo+20, 't1': lo+30, 'state': 'dead'},
                                               {'t0': hi+5, 't1': hi+9, 'state': 'B'}])
        for bad in (dict(rec, spans=[{'t0': lo+5, 't1': lo+1, 'state': 'A'}]), dict(rec, spans=[{'t0': lo, 't1': lo+3, 'state': 'C'}]),
                    dict(rec, kind='other'), [1], None):
            r = self.s.post('/api/coaching/spans', {'run': RUN, 'spans': bad})
            self.assertEqual(r.status_code, 400, bad)
        self.assertFalse((self.s.folder/'coach_spans.json').exists())
        r = self.s.post('/api/coaching/spans', {'run': RUN, 'spans': rec})
        self.assertEqual(r.status_code, 200, r.get_json())
        out = r.get_json()
        self.assertEqual(out['spans']['spans'], [{'t0': lo, 't1': lo+20, 'state': 'A'}, {'t0': lo+20, 't1': lo+30, 'state': 'dead'}],
                         'clipped to the segment within 0.5 s; one wholly after it dropped')
        self.assertEqual((out['dropped'], out['stale']), (1, ['measurement', 'role', 'interpretation']))
        self.assertEqual(self.s.post('/api/coaching/spans', {'run': RUN, 'spans': out['spans']}).get_json()['stale'], [],
                         'the same spans again change nothing')
        stored = json.loads((self.s.folder/'coach_spans.json').read_text(encoding='utf-8'))
        self.assertEqual(list(stored['runs']), [RUN])
        self.assertEqual(self.s.get('/api/coaching/spans', run=RUN).get_json()['spans'], out['spans'])
        stale = self.answer()
        self.assertEqual((stale['state'], stale['tiers']['measurement'], stale['coachSpans']), ('stale', 'missing', True))
        self.assertIn('coach_spans', stale['reasonCodes'])
        self.assertIn('The possession you marked changed.', stale['reasons'])
        marked = self.build()
        self.assertNotEqual(marked['fingerprints']['measurement'], engine_only['fingerprints']['measurement'])
        obs = store.load_tier(store.tier_path(self.s.run_dir, 'obs', {'team': 'A', 'mfp': marked['fingerprints']['measurement']}))
        self.assertIsNotNone(obs['timeline']['phaseQuality']['coach'], 'the spans reached the situation timeline')
        self.assertTrue(any(s.get('source') == 'coach' for s in obs['timeline']['phaseSpans']))
        # Spans of another run of the video are never used (team letters belong to a run).
        stored['runs']['20261017-120000-000000'] = stored['runs'][RUN]
        (self.s.folder/'coach_spans.json').write_text(json.dumps(stored), encoding='utf-8')
        r = self.s.post('/api/coaching/spans/remove', {'run': RUN})
        self.assertEqual((r.status_code, r.get_json()['spans'], r.get_json()['stale']), (200, store.empty_spans(), ['measurement', 'role', 'interpretation']))
        self.assertEqual(list(json.loads((self.s.folder/'coach_spans.json').read_text(encoding='utf-8'))['runs']), ['20261017-120000-000000'])
        back = self.answer()
        self.assertEqual((back['state'], back['coachSpans'], back['fingerprints']['measurement']),
                         ('ready', False, engine_only['fingerprints']['measurement']), 'the engine-only measurement is still stored')

    def test_verdicts_and_adjusted_areas_attach_by_key_and_range(self):
        self.save('A', block(4, 'U14'))
        data = self.build()
        rep = data['report']
        cited = [f for f in self.findings(rep) if f['timestamps']]
        for level, age in ((5, 'U16'), (3, 'U12'), (6, 'Adult'), (7, 'Adult')):   # whichever level the rules show one at
            if cited:
                break
            self.save('A', block(level, age))
            data = self.answer()
            rep = data['report']
            cited = [f for f in self.findings(rep) if f['timestamps']]
        self.assertTrue(cited, 'the scene gives a finding with key moments')
        f = cited[0]
        rng = {'t0': rep['range']['t0'], 't1': rep['range']['t1']}
        body = {'run': RUN, 'team': 'A', 'findingKey': f['key'], 'range': rng}
        r = self.s.post('/api/coaching/feedback', dict(body, verdict='correct', text='Seen it too'))
        self.assertEqual(r.status_code, 200, r.get_json())
        verdict = r.get_json()['feedback']
        self.assertEqual((verdict['findingKey'], verdict['attached'], verdict['thisRun'], verdict['verdict'], verdict['range']),
                         (f['key'], f['key'], True, 'correct', rng))
        self.assertRegex(verdict['id'], store.FEEDBACK_ID)
        again = self.s.post('/api/coaching/feedback', dict(body, verdict='partial')).get_json()['feedback']
        ev = f['timestamps'][0]['evidence']
        area = {'evidence': ev, 'polygon': [[.40, .30], [.48, .30], [.48, .40], [.40, .40]], 'mover': None, 'note': 'Stay goal side'}
        # CODE L-6 (round 2): an area was stored on a moment without a suggested area (a 'moments worth a look' review
        # has none by design; the short_segment scene's moments have none at all). Refused until the stored report
        # draws one there.
        self.assertFalse(store.has_suggested_area(rep, ev))
        r = self.s.post('/api/coaching/feedback', dict(body, adjustment=area))
        self.assertEqual(r.status_code, 400, r.get_json())
        self.assertIn('no suggested area', r.get_json()['error'])
        self.draw_area(data, ev)
        r = self.s.post('/api/coaching/feedback', dict(body, adjustment=area))
        self.assertEqual(r.status_code, 200, r.get_json())
        adj = r.get_json()['feedback']
        self.assertEqual((adj['verdict'], adj['attached'], adj['text']), (None, f['key'], 'Stay goal side'))
        self.assertRegex(adj['id'], store.ADJUSTMENT_ID)
        self.assertEqual(S.validate('adjustment', adj['adjustment']), [])
        for change, status in [({'verdict': 'agree'}, 400), ({'findingKey': 'D1|A|team|1.0'}, 400), ({'range': {'t0': 'x', 't1': 2}}, 400),
                               ({'verdict': 'correct', 'adjustment': area}, 400), ({}, 400), ({'team': 'C', 'verdict': 'correct'}, 400),
                               ({'range': {'t0': rng['t0']+10, 't1': rng['t1']}, 'verdict': 'correct'}, 404),
                               ({'findingKey': f['key'].replace('|A|', '|B|'), 'verdict': 'correct'}, 404),
                               ({'adjustment': dict(area, polygon=[[.4, .3], [2.0, .3], [.5, .5]])}, 400),
                               ({'adjustment': dict(area, evidence='ev-000000000000')}, 400),
                               ({'adjustment': dict(area, note='n\u00e4chste Woche')}, 400),
                               ({'adjustment': dict(area, mover='GK-A')}, 400), ({'run': 'another-run', 'verdict': 'correct'}, 409)]:
            self.assertEqual(self.s.post('/api/coaching/feedback', dict(body, **change)).status_code, status, change)
        doc = json.loads((self.s.folder/'coaching_feedback.json').read_text(encoding='utf-8'))
        self.assertEqual(doc['version'], 2)
        self.assertEqual([e['id'] for e in doc['feedback']], [again['id']], 'one verdict per finding, run and range')
        self.assertEqual([a['adjustment']['id'] for a in doc['adjustments']], [adj['id']])
        # A v1 verdict in the file is listed apart (no finding key), never on a finding.
        v1 = store.add_feedback(self.s.folder, RUN, {'fingerprint': 'f'*64, 'report': fixture_report(synthetic_result(seconds=6))},
                                fixture_report(synthetic_result(seconds=6))['improvements'][0]['observation'], 'correct')
        served = self.answer()
        by_id = {e['id']: e for e in served['feedback']}
        self.assertEqual((by_id[again['id']]['attached'], by_id[adj['id']]['attached']), (f['key'], f['key']))
        self.assertEqual((by_id[v1['id']]['findingKey'], by_id[v1['id']]['attached'], by_id[v1['id']]['thisRun']), (None, None, True))
        # Another range is another report: the verdicts are not on its findings.
        part = self.build(t0=rng['t0']+5, t1=rng['t1'])
        self.assertTrue(all(e['attached'] is None for e in part['feedback']))
        # A level change keeps the finding key, so the verdict stays on it when the finding is still shown.
        self.save('A', block(5, 'U16'))
        keys5 = {x['key'] for x in self.findings(self.answer()['report'])}
        on = {e['id']: e['attached'] for e in self.answer()['feedback']}
        self.assertEqual(on[again['id']], f['key'] if f['key'] in keys5 else None)
        self.assertEqual(self.s.post('/api/coaching/feedback/remove', {'id': adj['id']}).get_json(), {'ok': True, 'removed': adj['id']})
        self.assertEqual(self.s.post('/api/coaching/feedback/remove', {'id': adj['id']}).status_code, 404)
        self.assertEqual(self.s.post('/api/coaching/feedback/remove', {'id': 'adj-xyz'}).status_code, 400)

    def test_squads_progress_and_the_match_date(self):
        for bad in ({'name': '', 'ageGroup': 'U14', 'level': 4, 'format': {'players': 11, 'goalkeeper': True}},
                    {'name': 'Blue', 'ageGroup': 'U14', 'level': 9, 'format': {'players': 11, 'goalkeeper': True}}, [1]):
            self.assertEqual(self.s.post('/api/coaching/squads', bad).status_code, 400, bad)
        body = {'name': 'U14 Blue', 'ageGroup': 'U14', 'level': 4, 'format': {'players': 11, 'goalkeeper': True}}
        made = self.s.post('/api/coaching/squads', body).get_json()
        self.assertTrue(made['created'])
        sq = made['squad']
        same = self.s.post('/api/coaching/squads', dict(body, name='u14 blue')).get_json()
        self.assertEqual((same['created'], same['squad']['id']), (False, sq['id']), 'one squad per name')
        self.assertEqual([x['id'] for x in self.s.get('/api/coaching/squads').get_json()['squads']], [sq['id']])
        # Linked but not confirmed: the section asks to confirm, and no progress is written.
        self.save('A', block(4, 'U14', squad={'id': sq['id'], 'name': sq['name']}, runLinks=[{'run': RUN, 'team': 'A', 'confirmed': False}]))
        data = self.build()
        self.assertIn('Confirm which team', data['report']['progression']['note'])
        self.assertIsNone(data['report']['teamName'])
        self.assertFalse((self.s.root/'runs'/'_coaching'/'progress.json').exists())
        confirmed = self.save('A', block(4, 'U14', squad={'id': sq['id'], 'name': sq['name']}, runLinks=[{'run': RUN, 'team': 'A', 'confirmed': True}]))
        self.assertEqual(confirmed['stale'], [], 'the squad and its links are serve-time fields')
        data = self.answer()
        self.assertEqual((data['state'], data['report']['teamName'], data['report']['progression']['squad']), ('ready', 'U14 Blue', sq['id']))
        self.assertEqual(S.validate('report', data['report']), [])
        progress = self.s.root/'runs'/'_coaching'/'progress.json'
        entries = json.loads(progress.read_text(encoding='utf-8'))['entries']
        self.assertEqual([(e['squad'], e['run'], e['team'], e['dateSource']) for e in entries], [(sq['id'], RUN, 'A', 'file_mtime')],
                         'a report built before the link was confirmed still adds the match')
        stamp = progress.stat().st_mtime_ns
        self.answer()
        self.assertEqual(progress.stat().st_mtime_ns, stamp, 'served again: nothing rewritten')
        self.save('A', block(5, 'U14', squad={'id': sq['id'], 'name': sq['name']}, runLinks=[{'run': RUN, 'team': 'A', 'confirmed': True}]))
        self.assertEqual(self.answer()['report']['profile']['level'], 5)
        self.assertEqual(len(json.loads(progress.read_text(encoding='utf-8'))['entries']), 1, 'the same range replaces its entry')
        self.assertEqual(self.s.get('/api/coaching/squads').get_json()['squads'][0]['level'], 5, 'the squad follows the confirmed team')
        prog = self.s.get('/api/coaching/progress', squad=sq['id'], run=RUN, team='A').get_json()
        self.assertEqual((prog['squad']['id'], prog['entries'], prog['progress']['level']), (sq['id'], 1, 5))
        self.assertEqual(self.s.get('/api/coaching/progress', squad='sq-000000000000').status_code, 404)
        self.assertEqual(self.s.get('/api/coaching/progress', squad='../x').status_code, 400)
        self.assertEqual(self.s.post('/api/coaching/match-date', {'run': RUN, 'date': '2026-13-40'}).status_code, 400)
        r = self.s.post('/api/coaching/match-date', {'run': RUN, 'date': '2026-10-18'})
        self.assertEqual(r.get_json(), {'ok': True, 'date': '2026-10-18', 'updated': 1})
        entry = json.loads((self.s.root/'runs'/'_coaching'/'progress.json').read_text(encoding='utf-8'))['entries'][0]
        self.assertEqual((entry['matchDate'], entry['dateSource']), ('2026-10-18', 'coach'))
        self.assertEqual(self.s.get('/api/coaching/settings', run=RUN).get_json()['matchDate'], '2026-10-18')

    def test_the_pdf_gets_the_served_report_its_feedback_and_the_colours(self):
        self.assertEqual(self.s.get('/api/coaching/pdf', run=RUN, team='A').status_code, 404, 'generate first')
        self.save('A', block(4, 'U14'))
        data = self.build()
        f = next((x for x in self.findings(data['report']) if x['timestamps']), None)
        for level, age in ((5, 'U16'), (3, 'U12'), (6, 'Adult'), (7, 'Adult')):   # whichever level the rules show one at
            if f is not None:
                break
            self.save('A', block(level, age))
            data = self.answer()
            f = next((x for x in self.findings(data['report']) if x['timestamps']), None)
        self.assertIsNotNone(f, 'the scene gives a finding with key moments')
        level = data['report']['profile']['level']
        rng = {'t0': data['report']['range']['t0'], 't1': data['report']['range']['t1']}
        mine = self.s.post('/api/coaching/feedback', {'run': RUN, 'team': 'A', 'findingKey': f['key'], 'range': rng,
                                                      'verdict': 'correct'}).get_json()['feedback']
        seen = {}

        def render(report, labels, frame, stale=None, **options):
            moment = report['keyMoments'][0]['frame']
            seen.update(report=report, labels=labels, stale=stale, options=options, shot=frame(moment), other=frame(moment+1))
            return b'%PDF-1.4\n%fake\n'
        with mock.patch.object(store, 'render_pdf', render):
            r = self.s.get('/api/coaching/pdf', run=RUN, team='A', colourA='#b98cff', colourB='red')
        self.assertEqual((r.status_code, r.mimetype, r.data), (200, 'application/pdf', b'%PDF-1.4\n%fake\n'))
        self.assertIn(f'pitchiq-coaching-clip-{RUN}-A-level{level}.pdf', r.headers['Content-Disposition'])
        self.assertEqual((seen['report']['kind'], seen['report']['fingerprints'], seen['stale']),
                         ('coaching-report', data['fingerprints'], None))
        self.assertEqual(seen['options']['colours'], {'A': '#b98cff'}, 'only #rrggbb colours pass')
        self.assertEqual([e['id'] for e in seen['options']['feedback']], [mine['id']])
        self.assertEqual((seen['shot'][:2], seen['other']), (b'\xff\xd8', None), 'key-moment frames only')
        with mock.patch.object(store, 'render_pdf', mock.Mock(side_effect=store.PdfUnavailable('PDF export is not available.'))):
            self.assertEqual(self.s.get('/api/coaching/pdf', run=RUN, team='A').status_code, 503)

    def test_screenshots_of_key_moments_only(self):
        data = self.build()
        frames = sorted({e['frame'] for e in data['report']['keyMoments']})
        self.assertTrue(frames)
        r = self.s.get('/api/coaching/frame', run=RUN, frame=frames[0], w=640)
        self.assertEqual((r.status_code, r.mimetype), (200, 'image/jpeg'))
        self.assertAlmostEqual(grey_of(r.data), (frames[0] % 32)*8, delta=3, msg='the absolute source frame')
        other = next(k for k in range(6000, 6600) if k not in frames)
        self.assertEqual(self.s.get('/api/coaching/frame', run=RUN, frame=other, w=640).status_code, 404)

    def test_one_build_at_a_time_across_both_engines_and_failures_are_reported(self):
        release = threading.Event()
        real = store.measure

        def slow(*a, **kw):
            release.wait(30)
            return real(*a, **kw)
        with mock.patch.object(store, 'measure', slow):
            self.assertEqual(self.s.post('/api/coaching/build', {'run': RUN, 'team': 'A'}).get_json()['state'], 'building')
            busy = self.answer()
            self.assertEqual((busy['state'], busy['tiers']['measurement']), ('building', 'building'))
            self.assertEqual(self.answer('B')['state'], 'building', 'both teams are being measured')
            for body in ({'run': RUN, 'team': 'B'}, {'run': RUN, 'team': 'A', 'context': V1}):
                r = self.s.post('/api/coaching/build', body)
                self.assertEqual(r.status_code, 409, body)
                self.assertIn('being built', r.get_json()['error'])
            self.assertTrue(self.s.get('/api/coaching', run=RUN, team='A', context=V1).get_json()['busy'])
            release.set()
            self.assertEqual(self.s.wait()['state'], 'ready')
            # A format change of team A: only A is measured again, so B is ready (the build makes it only busy).
            release.clear()
            self.save('A', block(4, 'U14', format={'players': 9, 'goalkeeper': True}))
            self.assertEqual(self.s.post('/api/coaching/build', {'run': RUN, 'team': 'A'}).get_json()['state'], 'building')
            other = self.answer('B')
            self.assertEqual((other['state'], other['busy'], other['tiers']['measurement']), ('ready', True, 'ready'))
            self.assertEqual(self.answer()['state'], 'building')
            release.set()
            self.assertEqual(self.s.wait()['state'], 'ready')
        self.s.clear()
        with mock.patch.object(store, 'measure', mock.Mock(side_effect=RuntimeError('boom'))), contextlib.redirect_stderr(io.StringIO()):
            self.s.post('/api/coaching/build', {'run': RUN, 'team': 'A'})
            self.s.idle()
        failed = self.answer()
        self.assertEqual((failed['state'], failed['report']), ('error', None))
        self.assertIn('boom', failed['error'])
        self.assertEqual(self.tier_files('obs'), [], 'nothing stored')

        def correcting(*a, **kw):
            out = real(*a, **kw)
            (self.s.run_dir/'result.json').write_bytes(self.s.bytes.replace(b'"corrections":[]', b'"corrections":[{"id":"c1","applied":true}]', 1)
                                                      if b'"corrections":[]' in self.s.bytes else self.s.bytes+b' ')
            return out
        with mock.patch.object(store, 'measure', correcting):
            self.s.post('/api/coaching/build', {'run': RUN, 'team': 'A'})
            self.s.idle()
        self.assertEqual(self.tier_files('obs'), [], 'a result changed during the build is not stored under the old key')
        (self.s.run_dir/'result.json').write_bytes(self.s.bytes)
        self.assertEqual(self.build()['state'], 'ready', 'a later build clears the error')

    def test_bad_requests(self):
        for params, status in [({'team': 'C'}, 400), ({'t0': 601, 't1': 603}, 400), ({'t0': 'x', 't1': 650}, 400), ({'run': 'another-run'}, 409)]:
            r = self.s.get('/api/coaching', **dict({'run': RUN, 'team': 'A'}, **params))
            self.assertEqual(r.status_code, status, params)
            self.assertIn('error', r.get_json())
        for path in ('/api/coaching/build', '/api/coaching/settings', '/api/coaching/spans', '/api/coaching/spans/remove',
                     '/api/coaching/squads', '/api/coaching/match-date', '/api/coaching/feedback'):
            self.assertEqual(self.s.post(path, ['A']).status_code, 400, path)
        for path in ('/api/coaching/settings', '/api/coaching/spans'):
            self.assertEqual(self.s.get(path, run='another-run').status_code, 409, path)
        self.assertEqual(self.measured, [])

    def test_fix_round_1_requests(self):
        # CODE L6: a body nested deeper than the JSON parser's recursion limit was a 500 on every coaching POST route.
        deep = ('['*100000+']'*100000).encode('ascii')
        for path in ('/api/coaching/build', '/api/coaching/settings', '/api/coaching/spans', '/api/coaching/spans/remove',
                     '/api/coaching/squads', '/api/coaching/match-date', '/api/coaching/feedback', '/api/coaching/feedback/remove'):
            r = self.s.client.post(path, data=deep, content_type='application/json', base_url=BASE)
            self.assertEqual(r.status_code, 400, path)
        # CODE L7: a request another site's page makes (an <img> GET sends no Origin) is refused where the browser says
        # so; the page's own requests (same-origin) and tools without the header are answered.
        for site, status in (('cross-site', 403), ('same-site', 403), ('same-origin', 200), ('none', 200), (None, 200)):
            headers = {'Sec-Fetch-Site': site} if site else {}
            r = self.s.client.get('/api/coaching', query_string={'run': RUN, 'team': 'A'}, headers=headers, base_url=BASE)
            self.assertEqual(r.status_code, status, site)
        r = self.s.client.post('/api/coaching/build', json={'run': RUN, 'team': 'A'}, headers={'Sec-Fetch-Site': 'cross-site'}, base_url=BASE)
        self.assertEqual(r.status_code, 403)
        # CODE L5: the page's 0.2 s minimum and a count limit hold on the server too.
        lo = self.s.get('/api/coaching/spans', run=RUN).get_json()['segment']['t0']
        rec = dict(store.empty_spans(), spans=[{'t0': lo, 't1': lo+.001, 'state': 'A'}, {'t0': lo+1, 't1': lo+5, 'state': 'B'}])
        out = self.s.post('/api/coaching/spans', {'run': RUN, 'spans': rec}).get_json()
        self.assertEqual((out['spans']['spans'], out['dropped']), ([{'t0': lo+1, 't1': lo+5, 'state': 'B'}], 1))
        many = dict(store.empty_spans(), spans=[{'t0': lo+k*.01, 't1': lo+k*.01+.005, 'state': 'A'} for k in range(store.MAX_COACH_SPANS+1)])
        r = self.s.post('/api/coaching/spans', {'run': RUN, 'spans': many})
        self.assertEqual(r.status_code, 400)
        self.assertIn('at most', r.get_json()['error'])
        self.s.post('/api/coaching/spans/remove', {'run': RUN})
        self.assertEqual(self.measured, [])

    def test_the_answer_carries_the_measured_roles_once_the_role_tier_is_stored(self):
        # CODE M3: the roles editor read the measured roles once per run, before the first Generate, and showed "no
        # stable role" for every player until a reload; the report answer now carries them.
        first = self.answer()
        self.assertIsNone(first['inferred'], 'nothing measured yet')
        data = self.build()
        self.assertEqual(data['tiers']['role'], 'ready')
        got = self.s.get('/api/coaching/settings', run=RUN).get_json()['inferred']['A']
        self.assertTrue(data['inferred'], 'measured roles in the answer')
        self.assertEqual(data['inferred'], got, 'the same roles as the settings route gives')
        self.assertTrue(all(set(r) == {'id', 'role', 'source', 'confidence'} for r in data['inferred']))

    def test_a_coaching_table_that_does_not_validate_answers_json_503(self):
        # CODE M-1 (round 2): parameters.coverAngleDeg 70.0 (beyond its 60.0 measurement bound) made GET
        # /api/coaching/profiles answer Flask's HTML 500; the page could not parse it, took it for a server without the
        # v2 routes and built the v1 professional-reference report without a word.
        good = block(4, 'U14')
        data = json.loads(PR.PROFILES_PATH.read_text(encoding='utf-8'))
        data['parameters']['values']['coverAngleDeg']['value'] = 70.0
        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp)/'profiles.json'
            bad.write_text(json.dumps(data), encoding='utf-8')
            with mock.patch.object(PR, 'PROFILES_PATH', bad), contextlib.redirect_stderr(io.StringIO()):
                for r in (self.s.get('/api/coaching/profiles'), self.s.post('/api/coaching/settings', {'run': RUN, 'team': 'A', 'settings': good})):
                    self.assertEqual((r.status_code, r.mimetype), (503, 'application/json'))
                    got = r.get_json()
                    self.assertEqual(got['table'], 'profiles.json')
                    self.assertTrue(any('coverAngleDeg: 70.0' in p for p in got['problems']), got['problems'])
                    self.assertIn('profiles.json does not validate (parameters.coverAngleDeg: 70.0', got['error'])
                answer = self.s.get('/api/coaching', run=RUN, team='A')
                self.assertEqual((answer.status_code, answer.get_json()['engine']), (200, 2), 'the report route stays v2')
                with self.s.get('/api/info') as r:
                    self.assertEqual(r.status_code, 200)
        self.assertEqual(self.s.get('/api/coaching/profiles').status_code, 200, 'and back once the table validates')
        self.assertEqual(self.s.get('/api/coaching/profiles').get_json(), json.loads(store.json_text(PR.catalog())))

    def test_a_confirmed_player_keeps_his_measured_role(self):
        # CODE L-2 (round 2): a confirmed player's roles file holds only the coach's record, and the editor read
        # "Measured: no stable role" for him (BC60: A-07 RB, stable, high, before confirming him as RWB).
        self.save('A', block(4, 'U14'))
        before = {r['id']: r for r in self.build()['inferred']}
        pid = next(i for i, r in sorted(before.items()) if r['role'] and r['source'] != 'none')
        other = 'RWB' if before[pid]['role'] != 'RWB' else 'LWB'
        self.save('A', dict(block(4, 'U14'), roles={pid: [{'role': other, 'from': None, 'to': None}]}))
        answer = self.answer()
        self.assertEqual(answer['tiers']['role'], 'ready')
        keys = {'team': 'A', 'mfp': answer['fingerprints']['measurement'], 'rfp': answer['fingerprints']['roles']}
        roles = store.load_tier(store.tier_path(self.s.run_dir, 'roles', keys))
        self.assertTrue(any(r['id'] == pid and r['source'] == 'coach' for r in roles['rolebook']['records']), 'confirmed')
        self.assertNotIn(pid, {r['id'] for r in roles['rolebook']['records'] if r['source'] != 'coach'},
                         'the roles file holds only the coach record for him (the case of CODE L-2)')
        after = {r['id']: r for r in answer['inferred']}
        self.assertEqual(after[pid], before[pid], 'the measured reading, not the coach record and not missing')
        self.assertEqual(after, before, 'nobody else changes')
        settings = {r['id']: r for r in self.s.get('/api/coaching/settings', run=RUN).get_json()['inferred']['A']}
        self.assertEqual(settings, before)
        # Without the obs file the old reading is kept (the confirmed player left out), never an error.
        self.assertNotIn(pid, {r['id'] for r in store.measured_roles(dict(roles, fingerprint=None), obs=lambda: None)})
        obs = store.load_tier(store.tier_path(self.s.run_dir, 'obs', keys))
        self.assertEqual({r['id']: r for r in store.measured_roles(dict(roles, fingerprint=None), obs=obs)}, before)

    def test_the_pdf_is_told_whose_possession_the_coach_marked(self):
        # CODE M-2 (round 2): the PDF words the possession row as the page does, from the run's stored spans.
        self.build()
        seg = self.s.get('/api/coaching/spans', run=RUN).get_json()['segment']
        seen = {}

        def render(report, labels, frame, stale=None, **options):
            seen.update(options)
            return b'%PDF-1.4\n%fake\n'
        with mock.patch.object(store, 'render_pdf', render):
            self.assertEqual(self.s.get('/api/coaching/pdf', run=RUN, team='A').status_code, 200)
            self.assertEqual((seen['coach_spans'], seen['marked']), (False, None))
            self.s.post('/api/coaching/spans', {'run': RUN, 'spans': dict(store.empty_spans(), spans=[
                {'t0': seg['t0'], 't1': seg['t0']+20, 'state': 'A'}, {'t0': seg['t0']+20, 't1': seg['t0']+25, 'state': 'dead'}])})
            self.build()
            self.assertEqual(self.s.get('/api/coaching/pdf', run=RUN, team='A').status_code, 200)
            self.assertEqual((seen['coach_spans'], seen['marked']), (True, 'A'))

    def test_a_broken_v2_module_fails_only_the_v2_routes(self):
        broken = mock.Mock(side_effect=ImportError('broken profiles module'))
        with mock.patch.object(store, 'read_settings', broken), mock.patch.object(store, 'profiles_catalog_text', broken), \
                contextlib.redirect_stderr(io.StringIO()):
            for path in ('/api/coaching', '/api/coaching/settings', '/api/coaching/spans', '/api/coaching/profiles', '/api/coaching/pdf'):
                r = self.s.get(path, run=RUN, team='A')
                self.assertEqual(r.status_code, 503, path)
                self.assertIn('broken profiles module', r.get_json()['error'])
            for path in ('/api/coaching/build', '/api/coaching/spans'):
                self.assertEqual(self.s.post(path, {'run': RUN, 'team': 'A', 'spans': store.empty_spans()}).status_code, 503, path)
            self.assertEqual(self.s.get('/api/coaching', run=RUN, team='A', context=V1).status_code, 200, 'the v1 report still answers')
            for path in ('/api/info', '/api/status', '/coaching.js'):
                with self.s.get(path) as r:
                    self.assertEqual(r.status_code, 200, path)
        self.assertEqual(self.s.get('/api/coaching/settings', run=RUN).status_code, 200, 'and back once the module loads')

    def test_the_command_line_gives_the_same_fingerprints_and_report(self):
        self.save('A', block(3, 'U12', style='possession'))
        seg = self.s.get('/api/coaching/spans', run=RUN).get_json()['segment']
        self.s.post('/api/coaching/spans', {'run': RUN, 'spans': dict(store.empty_spans(), spans=[
            {'t0': seg['t0'], 't1': seg['t0']+30, 'state': 'A'}, {'t0': seg['t0']+30, 't1': seg['t1'], 'state': 'B'}])})
        served = self.build(t0=605, t1=655)
        spec = importlib.util.spec_from_file_location('coaching_report_cli', ROOT/'tools'/'coaching_report.py')
        cli = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cli)
        before = sorted(p.name for p in self.s.run_dir.iterdir())
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)/'cli.json'
            argv = [str(self.s.run_dir/'result.json'), '--team', 'A', '--t0', '605', '--t1', '655', '--out', str(out), '--quiet',
                    '--coaching', str(self.s.folder/'coaching_settings.json'), '--spans', str(self.s.folder/'coach_spans.json')]
            self.assertEqual(cli.main(argv), 0)
            mine = json.loads(out.read_text(encoding='utf-8'))
        self.assertEqual(sorted(p.name for p in self.s.run_dir.iterdir()), before, 'nothing written next to the result')
        self.assertEqual(mine['fingerprints']['A'], served['fingerprints'])
        a, b = copy.deepcopy(mine['reports']['A']), copy.deepcopy(served['report'])
        for r in (a, b):
            for k in ('generated', 'progression', 'teamName', 'video'):
                r.pop(k)
        self.assertEqual(a, b)
        # The v1 report stays one option away; the two engines' options never mix.
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)/'v1.json'
            self.assertEqual(cli.main([str(self.s.run_dir/'result.json'), '--team', 'A', '--no-reference', '--observations-only',
                                       '--out', str(out), '--quiet']), 0)
            self.assertIn('observations', json.loads(out.read_text(encoding='utf-8')), 'a v1-only option selects v1')
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                cli.main([str(self.s.run_dir/'result.json'), '--context', 'competitive', '--level', '4', '--out', str(out)])


class WhileATrackingRunProcesses(unittest.TestCase):
    def test_refused_until_the_run_has_finished(self):
        s = Scratch()
        try:
            with mock.patch.object(store, 'build_report_for', fixture_builder()):
                self.assertEqual(s.post('/api/run', {'start': 0, 'seconds': 5}).status_code, 200)
                self.assertEqual(s.get('/api/status').get_json()['state'], 'running')
                for r in (s.post('/api/coaching/build', {'run': RUN, 'team': 'A', 'context': V1}), s.get('/api/coaching', run=RUN, team='A', context=V1),
                          s.post('/api/coaching/feedback', {'run': RUN, 'observation': 'obs-000000000000', 'verdict': 'correct'}),
                          s.post('/api/coaching/build', {'run': RUN, 'team': 'A'}), s.get('/api/coaching', run=RUN, team='A'),
                          s.get('/api/coaching/settings', run=RUN), s.get('/api/coaching/spans', run=RUN),
                          s.post('/api/coaching/settings', {'run': RUN, 'team': 'A', 'settings': block()}),
                          s.post('/api/coaching/spans', {'run': RUN, 'spans': store.empty_spans()})):
                    self.assertEqual(r.status_code, 409)
                    self.assertIn('Processing is running', r.get_json()['error'])
                s.gate.set()
                end = time.time()+20
                while s.get('/api/status').get_json()['state'] == 'running' and time.time() < end:
                    time.sleep(.02)
                status = s.get('/api/status').get_json()
                self.assertNotEqual(status['resultRun'], RUN)
                self.assertEqual(s.get('/api/coaching', run=RUN, team='A', context=V1).status_code, 409, 'the page must reload the newer run')
                self.assertEqual(s.get('/api/coaching', run=RUN, team='A').status_code, 409)
        finally:
            s.gate.set()
            s.tmp.cleanup()


# ===================================================== v1 =============================================================
class CoachingRoutesV1(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.s = Scratch()

    @classmethod
    def tearDownClass(cls):
        cls.s.tmp.cleanup()

    def setUp(self):
        s = self.s
        s.idle()
        s.clear()
        self.calls = []
        patcher = mock.patch.object(store, 'build_report_for', fixture_builder(self.calls))
        patcher.start()
        self.addCleanup(patcher.stop)

    def get(self, path, **params):
        return self.s.get(path, **dict({'context': V1}, **params))

    def build(self, **params):
        body = dict({'run': RUN, 'team': 'A', 'context': V1}, **params)
        r = self.s.post('/api/coaching/build', body)
        self.assertEqual(r.status_code, 200, r.get_json())
        return self.s.wait(**{k: v for k, v in body.items() if k != 'run'})

    def test_the_script_is_served(self):
        with self.s.get('/coaching.js') as r:
            self.assertEqual((r.status_code, r.mimetype), (200, 'application/javascript'))
            self.assertEqual(r.data, (ROOT/'coaching.js').read_bytes())
            self.assertEqual(r.headers['Cache-Control'], 'no-store')

    def test_built_once_cached_and_served_ready(self):
        first = self.get('/api/coaching', run=RUN, team='A').get_json()
        self.assertEqual((first['state'], first['report'], first['fingerprint'], first['feedback']), ('missing', None, None, []))
        self.assertRegex(first['current'], r'^[0-9a-f]{64}$')
        started = self.s.post('/api/coaching/build', {'run': RUN, 'team': 'A', 'context': V1}).get_json()
        self.assertEqual((started['state'], started['fingerprint']), ('building', first['current']))
        data = self.s.wait(context=V1)
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
        again = self.s.post('/api/coaching/build', {'run': RUN, 'team': 'A', 'context': V1}).get_json()
        self.assertEqual(again, {'ok': True, 'state': 'ready', 'fingerprint': first['current']})
        self.assertEqual(len(self.calls), 1, 'an up-to-date report is not built again')
        other = self.build(team='B', context='advanced', t0=3, t1=9)
        self.assertEqual((other['state'], other['options']), ('ready', {'team': 'B', 't0': 3.0, 't1': 9.0, 'context': 'advanced'}))
        self.assertEqual(self.s.wait(context=V1)['state'], 'ready', 'other options are another report')

    def test_bad_requests(self):
        for params, status in [({'team': 'C'}, 400), ({'context': 'pro'}, 400), ({'t0': 2, 't1': 4}, 400), ({'t0': 'x', 't1': 9}, 400),
                               ({'run': 'another-run'}, 409)]:
            r = self.get('/api/coaching', **dict({'run': RUN, 'team': 'A'}, **params))
            self.assertEqual(r.status_code, status, params)
            self.assertIn('error', r.get_json())
        self.assertEqual(self.s.post('/api/coaching/build', ['A']).status_code, 400)
        self.assertEqual(self.s.post('/api/coaching/build', {'run': 'another-run', 'team': 'A', 'context': V1}).status_code, 409)
        # A run that is not a name is a bad request, not "an older result".
        for body in ({'run': [RUN], 'team': 'A', 'context': V1}, {'run': 5, 'team': 'A', 'context': V1}):
            r = self.s.post('/api/coaching/build', body)
            self.assertEqual((r.status_code, r.get_json()['error']), (400, 'run must be the name of the run the page shows.'), body)
        r = self.get('/api/coaching', run=RUN, team='A', t0=1, t1=3)
        self.assertIn('inside the analysed segment (00:01.00-00:12.97, source-video time)', r.get_json()['error'])
        self.assertEqual(self.s.client.post('/api/coaching/build', data='x', base_url=BASE).status_code, 415, 'JSON only (the request guard)')
        self.assertEqual(self.s.client.get('/api/coaching', base_url='http://evil.example').status_code, 403)
        self.assertEqual(self.calls, [])

    def test_one_build_at_a_time(self):
        release = threading.Event()
        slow = fixture_builder(self.calls)
        with mock.patch.object(store, 'build_report_for', lambda *a: (release.wait(20), slow(*a))[1]):
            self.assertEqual(self.s.post('/api/coaching/build', {'run': RUN, 'team': 'A', 'context': V1}).get_json()['state'], 'building')
            self.assertEqual(self.get('/api/coaching', run=RUN, team='A').get_json()['state'], 'building')
            other = self.get('/api/coaching', run=RUN, team='B').get_json()
            self.assertEqual((other['state'], other['busy']), ('missing', True))
            refused = self.s.post('/api/coaching/build', {'run': RUN, 'team': 'B', 'context': V1})
            self.assertEqual(refused.status_code, 409)
            self.assertIn('being built', refused.get_json()['error'])
            self.assertEqual(self.s.post('/api/coaching/build', {'run': RUN, 'team': 'B'}).status_code, 409, 'a v2 build waits too')
            release.set()
            self.assertEqual(self.s.wait(context=V1)['state'], 'ready')
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
                self.s.post('/api/coaching/build', {'run': RUN, 'team': 'A', 'context': V1})
                self.s.idle()
            data = self.get('/api/coaching', run=RUN, team='A').get_json()
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
        stale = self.get('/api/coaching', run=RUN, team='A').get_json()
        self.assertEqual((stale['state'], stale['reasonCodes'], stale['fingerprint']), ('stale', ['settings'], ready['fingerprint']))
        self.assertNotEqual(stale['current'], ready['fingerprint'])
        self.assertIsNotNone(stale['report'], 'a stale report is still served')
        corrected = self.s.bytes.replace(b'"corrections":[]', b'"corrections":[{"id":"c20261018-190500-abcdef","applied":true}]')
        (self.s.run_dir/'result.json').write_bytes(corrected)
        stale = self.get('/api/coaching', run=RUN, team='A').get_json()
        self.assertEqual(stale['reasonCodes'], ['result', 'settings'])
        self.assertIn('1 correction made since this report', stale['reasons'][0])
        (self.s.run_dir/'result.json').write_bytes(self.s.bytes)
        (self.s.folder/'team_shape.json').unlink()
        back = self.get('/api/coaching', run=RUN, team='A').get_json()
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
        served = self.get('/api/coaching', run=RUN, team='A').get_json()
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
        self.assertEqual(self.get('/api/coaching', run=RUN, team='A').get_json()['feedback'], [])

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
        served = self.get('/api/coaching', run=RUN, team='A').get_json()
        by_id = {f['id']: f for f in served['feedback']}
        self.assertEqual((by_id[mine['id']]['attached'], by_id[mine['id']]['thisRun']), (gap, True))
        self.assertEqual((by_id[earlier['id']]['attached'], by_id[earlier['id']]['thisRun']), (None, False))
        self.assertEqual((by_id[other_team['id']]['attached'], by_id[other_team['id']]['thisRun']), (None, True))
        self.assertEqual([(f['id'], f['thisRun'], f['attached']) for f in served['report']['feedback']],
                         [(mine['id'], True, gap), (earlier['id'], False, None)],
                         'the report (and its PDF) carries the verdicts on its findings and, apart, those of earlier runs')
        seen = {}
        with mock.patch.object(store, 'render_pdf', lambda report, labels, frame, stale=None: (seen.update(feedback=report['feedback']), b'%PDF-1.4')[1]):
            self.assertEqual(self.get('/api/coaching/pdf', run=RUN, team='A').status_code, 200)
        self.assertEqual([f['id'] for f in seen['feedback']], [mine['id'], earlier['id']])

    def test_responses_never_carry_a_non_finite_number(self):
        data = self.build()
        gap = data['report']['improvements'][0]['observation']
        self.s.post('/api/coaching/feedback', {'run': RUN, 'observation': gap, 'verdict': 'correct', 'team': 'A'})
        path = self.s.folder/'coaching_feedback.json'
        text = path.read_text(encoding='utf-8')
        path.write_text(text.replace('"t0": ', '"t0": NaN, "was": ', 1).replace('"verdict"', '"score": Infinity, "verdict"', 1), encoding='utf-8')
        for answer in (self.get('/api/coaching', run=RUN, team='A'), self.s.get('/api/coaching', run=RUN, team='A')):
            self.assertEqual(answer.status_code, 200)
            body = answer.get_data(as_text=True)
            for token in ('NaN', 'Infinity'):
                self.assertNotIn(token, body)
            served = json.loads(body, parse_constant=lambda c: self.fail(f'{c} in the response'))
            v1 = [e for e in served['feedback'] if 'observation' in e]
            self.assertEqual(v1[0]['observation']['t0'], None)
        self.assertEqual(json.loads(self.get('/api/coaching', run=RUN, team='A').get_data(as_text=True))['feedback'][0]['attached'], gap,
                         'the entry itself is kept')

    def test_a_broken_coaching_package_fails_only_its_routes(self):
        source = (ROOT/'server.py').read_text(encoding='utf-8')
        self.assertNotRegex(source, r'(?m)^from soccer\.coaching|^import soccer\.coaching', 'not imported at start')
        with mock.patch.object(self.s.server, 'coaching_store', mock.Mock(side_effect=ImportError('broken coaching package'))), \
                contextlib.redirect_stderr(io.StringIO()):
            for path in ('/api/coaching', '/api/coaching/frame', '/api/coaching/pdf', '/api/coaching/profiles', '/api/coaching/settings',
                         '/api/coaching/spans', '/api/coaching/squads', '/api/coaching/progress'):
                for context in (V1, None):
                    r = self.s.get(path, run=RUN, team='A', frame=1, **({'context': context} if context else {}))
                    self.assertEqual(r.status_code, 503, path)
                    self.assertIn('broken coaching package', r.get_json()['error'])
            for path in ('/api/coaching/build', '/api/coaching/feedback', '/api/coaching/feedback/remove', '/api/coaching/settings',
                         '/api/coaching/spans', '/api/coaching/spans/remove', '/api/coaching/squads', '/api/coaching/match-date'):
                self.assertEqual(self.s.post(path, {'run': RUN, 'team': 'A'}).status_code, 503, path)
            for path in ('/api/info', '/api/status', '/api/result', '/coaching.js'):
                with self.s.get(path) as r:
                    self.assertEqual(r.status_code, 200, path)
        self.assertEqual(self.get('/api/coaching', run=RUN, team='A').status_code, 200, 'and back once the package loads')

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
        self.assertEqual(self.get('/api/coaching', run=RUN, team='A').get_json()['feedback'], [])

    def test_pdf_of_the_stored_report(self):
        self.assertEqual(self.get('/api/coaching/pdf', run=RUN, team='A').status_code, 404, 'generate first')
        data = self.build()
        with mock.patch.object(store, 'render_pdf', mock.Mock(side_effect=store.PdfUnavailable('PDF export is not available in this version yet.'))):
            self.assertEqual(self.get('/api/coaching/pdf', run=RUN, team='A').status_code, 503)
        seen = {}

        def render(report, labels, frame, stale=None):
            evidence = report['evidence'][0]['frame']
            seen.update(fingerprint=report['fingerprint'], labels=labels, stale=stale, shot=frame(evidence), other=frame(evidence+1))
            return b'%PDF-1.4\n%fake\n'
        with mock.patch.object(store, 'render_pdf', render):
            r = self.get('/api/coaching/pdf', run=RUN, team='A')
        self.assertEqual((r.status_code, r.mimetype, r.data), (200, 'application/pdf', b'%PDF-1.4\n%fake\n'))
        self.assertIn('attachment', r.headers['Content-Disposition'])
        self.assertIn(f'pitchiq-coaching-clip-{RUN}-A-competitive.pdf', r.headers['Content-Disposition'], 'the level is in the name')
        self.assertEqual((seen['fingerprint'], seen['stale'], seen['other']), (data['fingerprint'], None, None))
        self.assertEqual((seen['labels']['B-07'], seen['labels']['A-03']), ('B-05', 'A-02'), 'the display labels of the result as it is now')
        self.assertEqual(seen['shot'][:2], b'\xff\xd8')
        self.s.post('/api/team-shape/settings', {'direction': {'A': 'left'}})
        with mock.patch.object(store, 'render_pdf', render):
            self.assertEqual(self.get('/api/coaching/pdf', run=RUN, team='A').status_code, 200)
        self.assertEqual(len(seen['stale']), 1, 'a stale report is exported with its reasons')


if __name__ == '__main__':
    unittest.main()
