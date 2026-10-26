"""The coaching report end to end: the parts (series, detectors, evidence, language, drills, report, store, frames,
render, pdf, the server's routes and tools/coaching_report.py) wired together on a synthetic match whose truth is
known, with the REAL engine everywhere (soccer.coaching.build_report_for; no fixture report stands in).

What is checked:
- server.py's coaching routes build the report on the worker thread with the engine, serve it ready under the
  fingerprint of its inputs, serve the screenshots of its evidence frames (the exact source frames), store the
  coach's feedback (a note outside WinAnsi included) and export the stored report as a PDF that parses;
- the served report has the contract's shape (section 7), every text of every finding passes the fact check against
  its own observation again, every evidence moment is a frame of the segment with seek = (frame + 0.5) / fps, and the
  three context levels change words only (same observations, evidence, metrics and priorities);
- tools/coaching_report.py --context gives the same v1 report and the same fingerprint as the server for the same
  result (the CLI's default is the v2 report since wave 7; its equality with the server is test_coaching_server's);
- the wording the integration run on the four test runs corrected (soccer/coaching/language.py, detectors.py,
  report.py): a strength is one-sided and states the professional median (never "within the professional range"
  or "upper level": broadcast 0-60 s team A's length had a median of 17.8 m against a professional p10 of 23.2 m),
  a lateral lag says it was measured in continuous stretches of at least the window (broadcast 42-222 s team B: 70 s
  measured, longest stretch 25 s) and reads "at about the same time" within one 5 Hz step (-0.15 s, 0.0 s re-measured
  independently), a formation reading is at most medium confidence (the same physical team read 3-4-3 'clear' and
  4-3-3 'ambiguous' in two overlapping runs), every reason code the series can give has words, and an evidence
  caption that fails the fact check is left out.

Opt-in on real runs: set PITCHIQ_COACHING_INTEGRATION to result.json paths joined with ';' and RealRuns builds both
teams at every level through build_report_for, checks the same properties and that each overlay's player feet sit on
the frame's box bottom centres (within 1 px).

Run:  python -m unittest test_coaching_integration          (from this directory)
"""
import contextlib
import hashlib
import io
import json
import math
import os
import re
import shutil
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for p in (str(ROOT), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)
import soccer.coaching as coaching  # noqa: E402
from soccer.coaching import detectors as det, drills, language, report as rp, series as cs, store  # noqa: E402
from test_coaching_detectors import build, gap_pushes, observe  # noqa: E402
from test_coaching_language import baseline, observation, render  # noqa: E402
from test_coaching_server import load_server  # noqa: E402
from test_coaching_store import grey_of, write_video  # noqa: E402

PORT = 8765
BASE = f'http://127.0.0.1:{PORT}'
RUN = '20261018-210000-c0ffee'
FPS = 5.0                                  # the synthetic match's frame rate (test_coaching_detectors.FPS)
SPELLS = [(20.0, 40.0), (80.0, 100.0), (140.0, 160.0), (200.0, 220.0)]   # the first midfield line pushed 10 m up
TEXT_FIELDS = ('headline', 'whatHappened', 'whyItMatters', 'whatToImprove', 'howToPractice', 'video', 'confidenceNote')
REPORT_KEYS = {'version', 'engineVersion', 'textEngine', 'fingerprint', 'generated', 'kind', 'context', 'team', 'teamName', 'video',
               'run', 'range', 'overview', 'summary', 'strengths', 'improvements', 'reviews', 'breakdown',
               'priorities', 'training', 'confidence', 'evidence', 'observations', 'feedback'}
SECTIONS = {'defensiveOrganization', 'spacing', 'formation', 'attackingStructure', 'movementSupport'}


def synthetic_match():
    """240 s, team A (4-4-2, defends the left goal) with its first midfield line pushed 10 m up in four spells."""
    result = build(seconds=240.0, overrides_a=gap_pushes(SPELLS))
    result['video'].update(name='clip.avi', fps=FPS)
    return result


def unchanged(rep):
    """What every context level must share: the measurements and the choices made from them."""
    findings = [(f['observation'], f['label'], f['metrics'], f['evidence'], f['drill'])
                for k in ('strengths', 'improvements', 'reviews') for f in rep[k]]
    return (rep['observations'], rep['evidence'], findings, [(p['finding'], p['drill']) for p in rep['priorities']],
            rep['kind'], rep['range'])


class ReportChecks:
    def check_report(self, rep, frames, fps):
        self.assertEqual(set(rep), REPORT_KEYS)
        self.assertEqual(set(rep['breakdown']), SECTIONS)
        self.assertFalse(rep['breakdown']['attackingStructure']['available'])
        self.assertLessEqual(len(rep['priorities']), rp.MAX_PRIORITIES)
        json.dumps(rep, allow_nan=False)
        obs = {o['id']: o for o in rep['observations']}
        evs = {e['id']: e for e in rep['evidence']}
        by_frame = {int(f['frame']): float(f['time']) for f in frames}
        for e in rep['evidence']:
            self.assertIn(e['frame'], by_frame, 'evidence must be a frame of the segment')
            self.assertAlmostEqual(e['time'], by_frame[e['frame']], places=3)
            self.assertAlmostEqual(e['seek'], (e['frame']+.5)/fps, places=3)
            self.assertIn(e['tier'], ('world', 'pinned', 'image'))
        library = drills.load_drills()
        for key in ('strengths', 'improvements', 'reviews'):
            for f in rep[key]:
                members = [obs[i] for i in f.get('observations') or [f['observation']]]   # a grouped strength
                for o in members:
                    self.assertIn(o['confidence']['overall'], ('high', 'medium'))
                drill = drills.drill(library, f['drill']) if f['drill'] else None
                facts = language.facts_for(None, [evs[e] for e in f['evidence']], drill, extra={'teamName': None})
                for o in members:
                    facts = facts.merge(language.facts_for(o))
                for field in TEXT_FIELDS:
                    self.assertEqual(language.validate_text(f.get(field), facts), [], (field, f.get(field)))
                self.assertNotIn('within the professional range', f['headline'])
                self.assertNotIn('upper level', f['headline'])


class ServerEndToEnd(ReportChecks, unittest.TestCase):
    """server.py with stand-ins for the tracking pipeline only: the report, its screenshots and its PDF are the real
    engine's, on a scratch ROOT (never the repository's runs/)."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = cls.root = Path(cls.tmp.name)
        shutil.copyfile(ROOT/'coaching.js', root/'coaching.js')
        (root/'uploads').mkdir()
        cls.result = synthetic_match()
        last = max(int(f['frame']) for f in cls.result['frames'])
        cls.video = write_video(root/'uploads'/'clip.avi', frames=last+10, fps=FPS)
        cls.key = hashlib.sha256(cls.video.read_bytes()).hexdigest()
        cls.folder = root/'runs'/cls.key[:16]
        cls.run_dir = cls.folder/RUN
        cls.run_dir.mkdir(parents=True)
        cls.bytes = json.dumps(cls.result, separators=(',', ':'), allow_nan=False).encode('utf-8')
        (cls.run_dir/'result.json').write_bytes(cls.bytes)
        (cls.folder/'latest.json').write_text(json.dumps({'run': RUN}), encoding='utf-8')
        cls.server = load_server(root, threading.Event())
        captured = []
        argv = ['server.py', '--video', str(cls.video), '--port', str(PORT)]
        with mock.patch.object(sys, 'argv', argv), \
                mock.patch.object(cls.server, 'serve', lambda app, **kw: captured.append(app)), \
                mock.patch.object(cls.server, 'preload_models', lambda: None), contextlib.redirect_stdout(io.StringIO()):
            cls.server.main()
        cls.client = captured[0].test_client()
        cls.reports = {}
        for context in language.LEVELS:
            cls.reports[context] = cls.build_report('A', context)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    @classmethod
    def get(cls, path, **params):
        return cls.client.get(path, query_string=params, base_url=BASE)

    @classmethod
    def build_report(cls, team, context):
        params = {'run': RUN, 'team': team, 'context': context}
        started = cls.client.post('/api/coaching/build', json=params, base_url=BASE).get_json()
        end = time.time()+120
        while True:
            data = cls.get('/api/coaching', **params).get_json()
            if data.get('state') != 'building' or time.time() > end:
                break
            time.sleep(.05)
        return {'started': started, 'answer': data}

    def test_the_engine_builds_a_report_the_page_can_show(self):
        a = self.reports['competitive']['answer']
        self.assertEqual((a['state'], a['error'], a['reasons']), ('ready', None, []), a.get('error'))
        rep = a['report']
        self.assertEqual(rep['fingerprint'], a['current'])
        self.assertEqual((rep['team'], rep['context'], rep['kind'], rep['run']), ('A', 'competitive', 'segment', RUN))
        self.assertEqual(rep['video'], {'name': 'clip.avi', 'key': self.key[:16]})
        self.assertNotIn('FIXTURE', json.dumps(rep), 'the engine, not the test fixture, made this report')
        self.assertEqual(rep['engineVersion'], coaching.ENGINE_VERSION)
        self.check_report(rep, self.result['frames'], FPS)
        gaps = [o for o in rep['observations'] if o['detector'] == 'unit_gap_dm']
        self.assertTrue(gaps and all(o['team'] == 'A' for o in gaps))
        self.assertTrue(any(o['occurrenceCount'] >= 2 for o in gaps), [o['occurrenceCount'] for o in gaps])
        stored = list((self.run_dir/'coaching').glob('report-A-*.json'))
        self.assertEqual(len(stored), 3, 'one stored report per context level')

    def test_context_levels_change_words_only(self):
        reps = {c: r['answer']['report'] for c, r in self.reports.items()}
        for context in ('development', 'advanced'):
            self.assertEqual(unchanged(reps[context]), unchanged(reps['competitive']), context)
        self.assertEqual(len({r['answer']['current'] for r in self.reports.values()}), 3)

    def test_screenshots_are_the_evidence_frames(self):
        rep = self.reports['competitive']['answer']['report']
        self.assertTrue(rep['evidence'])
        for e in rep['evidence'][:3]:
            r = self.get('/api/coaching/frame', run=RUN, frame=e['frame'], w=640)
            self.assertEqual((r.status_code, r.mimetype), (200, 'image/jpeg'), r.get_json(silent=True))
            self.assertAlmostEqual(grey_of(r.data), (e['frame'] % 32)*8, delta=2.0)
        frames = {e['frame'] for e in rep['evidence']}
        other = next(int(f['frame']) for f in self.result['frames'] if int(f['frame']) not in frames)
        self.assertEqual(self.get('/api/coaching/frame', run=RUN, frame=other, w=640).status_code, 404)

    def test_feedback_and_pdf_of_the_stored_report(self):
        rep = self.reports['competitive']['answer']['report']
        target = (rep['strengths']+rep['improvements']+rep['reviews'] or [{'observation': rep['observations'][0]['id']}])[0]
        body = {'run': RUN, 'team': 'A', 'observation': target['observation'], 'verdict': 'partial',
                'text': 'Gap right, but ball was dead \u2013 n\u00e4chste Woche \u2713', 'fingerprint': rep['fingerprint']}
        r = self.client.post('/api/coaching/feedback', json=body, base_url=BASE)
        self.assertEqual(r.status_code, 200, r.get_json())
        answer = self.get('/api/coaching', run=RUN, team='A', context='competitive').get_json()
        self.assertEqual([f['attached'] for f in answer['report']['feedback']], [target['observation']])
        r = self.get('/api/coaching/pdf', run=RUN, team='A', context='competitive')
        self.assertEqual((r.status_code, r.mimetype), (200, 'application/pdf'), r.get_json(silent=True))
        data = r.data
        self.assertTrue(data.startswith(b'%PDF-') and data.rstrip().endswith(b'%%EOF'))
        start = int(re.findall(rb'startxref\s+(\d+)', data)[-1])
        self.assertEqual(data[start:start+4], b'xref')
        self.assertIn(f'pitchiq-coaching-clip-{RUN}-A-competitive.pdf', r.headers['Content-Disposition'])   # the level is in the name

    def test_the_command_line_gives_the_same_report_and_fingerprint(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location('coaching_report_cli', ROOT/'tools'/'coaching_report.py')
        cli = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cli)
        out = Path(self.tmp.name)/'cli.json'
        # the CLI makes the v2 report by default since wave 7 (as the server does without a context level); this
        # class is the v1 engine end to end, so it asks for v1 by its level (test_coaching_server checks v2 CLI = v2
        # server)
        argv = [str(self.run_dir/'result.json'), '--team', 'A', '--context', 'competitive', '--out', str(out),
                '--quiet']
        self.assertEqual(cli.main(argv), 0)
        mine = json.loads(out.read_text(encoding='utf-8'))['reports']['A']
        served = dict(self.reports['competitive']['answer']['report'])
        self.assertEqual(mine['fingerprint'], served['fingerprint'])
        for r in (mine, served):
            r.pop('generated')
            r.pop('feedback')
        self.assertEqual(mine, served)


class Wording(unittest.TestCase):
    """The corrections the integration run made to the words (and to the formation confidence)."""

    def test_a_strength_is_one_sided(self):
        for level in language.LEVELS:
            for key, by_class in language.TEMPLATES.items():
                head = (by_class.get('strength') or {}).get('headline', {}).get(level, '')
                self.assertNotIn('professional range', head, (key, level))
                self.assertNotIn('reference range', head, (key, level))
        # fix round 1: the strength states the professional median beside the team's (17.8 m against 31.5 m), never
        # "within the professional upper level", which hid how far below the range the team sat
        depth = observation('compactness', 'strength', 'team_depth', measured=60.0,
                            metrics={'median': 17.8, 'shareAboveProfessionalP90': 0.0, 'professionalP50': 31.5,
                                     'professionalP75': 36.3, 'shareAboveProfessionalP75': 0.0},
                            base=dict(baseline(46.5, 43.1), threshold={'enter': 46.5, 'exit': 43.1, 'p90': 43.1}))
        out = render(depth)[0]
        self.assertEqual(out['headline'], 'Maintained a short team from back to front: median length 17.8 m (est.), '
                                          'professional median 31.5 m')

    def test_a_lateral_lag_names_its_stretches_and_its_resolution(self):
        lat = observation('lateral_shift', 'strength', measured=70.0, longest=25.0,
                          metrics={'lag': -0.15, 'corr': .953, 'windowSeconds': 20.0}, base=baseline(1.54, 1.04))
        happened = render(lat)[0]['whatHappened']
        self.assertIn('Over 70 s of measurement in continuous stretches of at least 20 s, the back line moved sideways '
                      'at about the same time as midfield on average.', happened)
        self.assertNotIn('continuous measurement', happened)
        lat['metrics']['lag'] = 0.6
        self.assertIn('moved sideways 0.6 s later than midfield', render(lat)[0]['whatHappened'])
        neutral = observation('lateral_shift', 'neutral', measured=40.0, metrics={'median': -0.1, 'windowSeconds': 30.0})
        note = render(neutral)[0]['note']
        self.assertIn('at about the same time as', note)
        self.assertNotIn('-0.1', note)

    def test_every_series_reason_code_has_words(self):
        codes = set(cs.REASONS) | {'ambiguous_formation', 'formation_unverified'}
        self.assertEqual(sorted(c for c in codes if c not in language.REASONS), [])
        self.assertEqual(language.wording_problems(language.REASONS['formation_unverified']), [])

    def test_a_formation_reading_is_at_most_medium_confidence(self):
        # 320 s: a stable formation is a strength only over 5 measured minutes (detectors.FORMATION_STRENGTH_SECONDS)
        obs, _ = observe(build(seconds=320.0))
        f = [o for o in obs if o['detector'] == 'formation_stability'][0]
        self.assertEqual((f['classification'], f['metrics']['label']), ('strength', '4-4-2'))
        self.assertEqual(f['confidence']['overall'], 'medium')
        self.assertIn('formation_unverified', f['confidence']['reasons'])
        self.assertEqual(det.FORMATION_MAX_CONFIDENCE, 'medium')

    def test_a_caption_that_fails_the_fact_check_is_left_out(self):
        gap = observation('unit_gap_dm', 'strength', measured=60.0, metrics={'median': 8.0,
                          'shareAboveProfessionalP90': 0.0}, base=baseline(22.6, 20.3))
        good = {'id': 'ev-000000000001', 'observation': gap['id'], 'occurrence': None, 'time': 12.0, 'frame': 360,
                'segmentTime': 12.0, 'seek': 12.0167, 'fps': 30.0, 'players': [], 'metrics': {'gapDM': 8.1},
                'confidence': 'high', 'tier': 'pinned', 'diagram': None, 'videoOverlay': None,
                'caption': 'Distance between back line and midfield here: 8.1 m (est.).'}
        bad = dict(good, id='ev-000000000002', time=20.0, frame=600, caption='The gap here was 31.4 m.')
        gap['evidence'] = [good['id'], bad['id']]
        meta = {'range': {'t0': 0.0, 't1': 59.97, 'seconds': 59.97}, 'segmentStart': 0.0, 'footage': 'broadcast',
                'pitch': {'length': 105.0, 'width': 68.0}}
        rep = rp.build_report([gap], [good, bad], meta, generated='2026-10-18T21:00:00')
        captions = {e['id']: e['caption'] for e in rep['evidence']}
        self.assertEqual(captions, {good['id']: good['caption'], bad['id']: None})
        self.assertIn('Some report text was left out because it could not be checked against the measurements.',
                      rep['confidence']['notes'])


@unittest.skipUnless(os.environ.get('PITCHIQ_COACHING_INTEGRATION'), 'set PITCHIQ_COACHING_INTEGRATION to result.json '
                                                                      'paths (;-separated) to run on real runs')
class RealRuns(ReportChecks, unittest.TestCase):
    def test_real_runs_end_to_end(self):
        for path in [p for p in os.environ['PITCHIQ_COACHING_INTEGRATION'].split(';') if p]:
            result = json.loads(Path(path).read_text(encoding='utf-8'))
            frames, fps = result['frames'], float(result['video']['fps'])
            W, H = result['video']['width'], result['video']['height']
            boxes = {int(f['frame']): {p['id']: p['box'] for p in f['people'] if p.get('id') and p.get('box')}
                     for f in frames}
            for team in ('A', 'B'):
                reps = {c: coaching.build_report_for(path, team, {'context': c}) for c in language.LEVELS}
                for c in ('development', 'advanced'):
                    self.assertEqual(unchanged(reps[c]), unchanged(reps['competitive']), (path, team, c))
                rep = reps['competitive']
                self.check_report(rep, frames, fps)
                for e in rep['evidence']:
                    for el in (e.get('videoOverlay') or {}).get('elements') or []:
                        box = boxes[e['frame']].get(el.get('id')) if el.get('type') == 'player' else None
                        if box:
                            foot = ((box[0]+box[2]/2)*W, (box[1]+box[3])*H)
                            self.assertLess(math.hypot(foot[0]-el['x'], foot[1]-el['y']), 1.0, (path, e['id']))


if __name__ == '__main__':
    unittest.main()
