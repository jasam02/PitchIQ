"""The coaching report's storage (soccer/coaching/store.py): the fingerprint keys a report on everything it is computed
from and on nothing else (the coach's feedback is not in it), a stored report goes stale with the reasons a coach can
act on, the feedback file is never wiped by a failed read, writes survive a Windows reader holding the old file, and a
screenshot is the exact source frame.

v2 (adaptive coaching, DESIGN v2): the per-video coaching settings are saved cleaned and validated with the tiers a change
makes stale, squad links never conflict, an unreadable file is never overwritten; coach spans are cleaned against the
segment as the situations stage cleans them, kept per run, and enter the measurement key only when there are some; each
setting changes only its own tier's fingerprint and those below; the role tier and the report are made from the stored
obs (never the result) and a stored report is never rebuilt; records are found fresh or stale with the reasons per tier;
verdicts and adjusted areas attach by finding key, run, team and range only; inferred roles; the squad of a team in a
run; a progress entry the rules cannot read is re-derived from its stored obs.

Also the shared fixtures of the coaching page and server tests: a synthetic result (two teams, absolute frame indices
that are not the array index, display labels that differ from the ids) and a report in the contract's shape built from
it (soccer/coaching/report.py makes the real one).

Run:  python -m unittest test_coaching_store          (from this directory)
"""
import json
import math
import os
import sys
import tempfile
import time
import types
import unittest
from datetime import datetime
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import soccer.coaching  # noqa: E402
from soccer.coaching import episodes, schema as S, store  # noqa: E402
import team_shape_synth as syn  # noqa: E402

FPS = 30.0


# ---------- shared fixtures ----------
def synthetic_result(seconds=12.0, fps=FPS, first_frame=30, width=1920, height=1080, seed=5):
    """Two teams holding 4-4-2 and 4-3-3 for `seconds`, the frames numbered from first_frame (absolute source frames,
    time = frame/fps), boxes on every person, a HIGH registration, and display labels that differ from the ids
    (A-02 and A-03 swapped, B-07 shown as B-05)."""
    times, teams, ends = syn.match('4-4-2', '4-3-3', seconds=seconds, fps=fps, seed=seed)
    result = syn.make_result(times, teams, ends, pitch_confidence=.8)
    shown = {'A-02': 'A-03', 'A-03': 'A-02', 'B-07': 'B-05'}
    for f, k in zip(result['frames'], range(len(result['frames']))):
        f['frame'] = first_frame+k
        f['time'] = round(f['frame']/fps, 6)
        f['calibration'] = {'confidence': .82, 'source': 'carried', 'disagreement': None, 'H': [1, 0, 0, 0, 1, 0, 0, 0, 1]}
        f['view'] = [[0, 0], [1, 0], [1, 1], [0, 1]]
        for p in f['people']:
            p['display'] = shown.get(p['id'], p['id'])
            x, y = p['pitch']
            p['box'] = [round(x-.006, 4), round(.25+.6*y-.05, 4), .012, .05]
    for e in result['match']['players'] + result['roster']['slots']:
        e['display'] = shown.get(e['id'], e['id'])
    result['video'] = {'name': 'clip.avi', 'width': width, 'height': height, 'fps': fps}
    result['corrections'] = []
    return result


def _evidence(result, obs, k, players, metrics, team):
    f = result['frames'][k]
    fps, t0, W, H = result['video']['fps'], result['frames'][0]['time'], result['video']['width'], result['video']['height']
    people = {p['id']: p for p in f['people']}
    pl = [{'type': 'player', 'id': p['id'], 'team': p['team'], 'x': p['pitch'][0], 'y': p['pitch'][1], 'kind': 'O',
           'role': 'D', 'emphasis': p['id'] in players} for p in f['people'] if p['role'] == 'PLAYER']
    a, b = people[players[0]]['pitch'], people[players[1]]['pitch']
    back = [result['frames'][max(0, k-j)] for j in (12, 8, 4, 0)]
    trail = [[q['pitch'][0], q['pitch'][1], fr['time']] for fr in back for q in fr['people'] if q['id'] == players[0]]
    diagram = {'preset': 'defensive_gap', 'title': f'Gap behind {players[0]} at {f["time"]:.2f} s', 't': f['time'], 'frame': f['frame'],
               'team': team, 'attack': 'right', 'metresEstimated': True, 'pitch': {'length': 105, 'width': 68},
               'elements': pl + [{'type': 'unit', 'unit': 'D', 'ids': [f'{team}-0{j}' for j in range(1, 5)]},
                                 {'type': 'measure', 'a': a, 'b': b, 'text': '21.4 m (est.)'},
                                 {'type': 'zone', 'polygon': [[.38, .3], [.52, .3], [.52, .7], [.38, .7]], 'style': 'gap'},
                                 {'type': 'band', 'axis': 'x', 'from': .38, 'to': .52, 'style': 'gap'},
                                 {'type': 'arrow', 'id': players[0], 'points': trail},
                                 {'type': 'suggested', 'id': players[0], 'x': a[0]+.03, 'y': a[1], 'label': 'COACHING ILLUSTRATION'},
                                 {'type': 'label', 'at': [.5, .04], 'text': f'{players[0]} and {players[1]}'}]}

    def foot(p):
        x, y, w, h = p['box']
        return [round((x+w/2)*W, 1), round((y+h)*H, 1)]
    overlay = {'tier': 'pinned', 'width': W, 'height': H,
               'elements': [{'type': 'player', 'id': i, 'team': people[i]['team'], 'x': foot(people[i])[0], 'y': foot(people[i])[1],
                             'box': [people[i]['box'][0]*W, people[i]['box'][1]*H, people[i]['box'][2]*W, people[i]['box'][3]*H], 'emphasis': True}
                            for i in players] + [{'type': 'measure', 'a': foot(people[players[0]]), 'b': foot(people[players[1]]), 'text': '21.4 m (est.)'}]}
    return {'id': episodes.stable_id('ev', obs, f['frame']), 'observation': obs, 'occurrence': episodes.stable_id('occ', obs, f['time']),
            'time': f['time'], 'frame': f['frame'], 'segmentTime': round(f['time']-t0, 3), 'seek': round((f['frame']+.5)/fps, 6), 'fps': fps,
            'players': list(players), 'metrics': metrics, 'confidence': 'high', 'tier': 'pinned',
            'calibration': {'confidence': .82, 'paintResidualPx': 4.1, 'disagreement': None}, 'diagram': diagram, 'videoOverlay': overlay,
            'caption': f'{players[0]} and {players[1]}: the gap at its widest.'}


def fixture_report(result, team='A', context='competitive', empty=False):
    """A report in the contract's shape (CONTRACT section 7) from a synthetic result: one strength, one improvement with
    two evidence frames, one review, one observation measured too briefly, a breakdown with unavailable sections, one
    priority with its drill. The text names players by id (A-03, B-07), as the template engine does."""
    frames = result['frames']
    t0, t1 = frames[0]['time'], frames[-1]['time']
    n = len(frames)
    other = 'B' if team == 'A' else 'A'
    p1, p2, opp = f'{team}-03', f'{team}-02', f'{other}-07'
    ids = {k: episodes.stable_id('obs', k, team, t0, t1) for k in ('compactness', 'unit_gap_dm', 'line_height_and_gap', 'lateral_shift')}
    ev_gap = [_evidence(result, ids['unit_gap_dm'], k, [p1, p2], {'gapDM': 21.4}, team) for k in (n//4, 3*n//4)]
    ev_review = [_evidence(result, ids['line_height_and_gap'], n//2, [p2, p1], {'lineHeight': 38.2, 'gapDM': 17.9}, team)]

    def observation(key, classification, severity, category, pattern, evidence, players, measured=float(t1-t0)):
        return {'id': ids[key], 'engineVersion': episodes.ENGINE_VERSION, 'team': team, 'detector': key, 'category': category, 'pattern': pattern,
                'classification': classification, 'severity': severity,
                'window': {'t0': t0, 't1': t1, 'segmentT0': 0.0, 'segmentT1': round(t1-t0, 2)},
                'measuredSeconds': measured, 'eligibleShare': .95, 'longestStretchSeconds': 9.0, 'occurrences': [], 'occurrenceCount': len(evidence),
                'ratePerMeasuredMinute': 3.1, 'baseline': {'reference': 'metrica-sample-game-1', 'rule': 'default', 'context': 'all'},
                'metrics': {}, 'players': players, 'evidence': [e['id'] for e in evidence],
                'confidence': {'tracking': 'high', 'calibration': 'medium', 'identity': 'high', 'ball': 'n/a', 'overall': 'medium', 'reasons': []},
                'recommendationKey': 'compact_lines', 'drillIds': ['compact_lines'],
                'status': 'suppressed' if classification == 'insufficient' else 'reported',
                'suppressionReason': 'measured 4.0 s, below the 20 s minimum' if classification == 'insufficient' else None}
    observations = [observation('compactness', 'strength', 'observed_strength', 'spacing', 'compact_width', [], [p1, p2]),
                    observation('unit_gap_dm', 'improvement', 'potential_improvement', 'spacing', 'space_between_defence_and_midfield', ev_gap, [p1, p2, opp]),
                    observation('line_height_and_gap', 'review', 'review_opportunity', 'defensive_line', 'high_line_with_gap', ev_review, [p2, p1]),
                    observation('lateral_shift', 'insufficient', None, 'movement', 'lateral_shift', [], [], measured=4.0)]
    finding = lambda key, label, headline, evidence, **kw: dict({'observation': ids[key], 'label': label, 'headline': headline,
                                                                 'whatHappened': kw.pop('what'), 'whyItMatters': kw.pop('why'),
                                                                 'whatToImprove': None, 'howToPractice': None, 'drill': None,
                                                                 'evidence': [e['id'] for e in evidence], 'metrics': kw.pop('metrics', [])}, **kw)
    strengths = [finding('compactness', 'Observed strength', 'Width maintained within the professional range', [],
                         what=f'Team width stayed inside the professional p10-p90 range for 11.9 s measured; {p1} and {p2} held the touchline side.',
                         why='A compact block leaves fewer gaps between the lines.', metrics=[{'name': 'width', 'value': 38.4, 'unit': 'm'}])]
    improvements = [finding('unit_gap_dm', 'Potential improvement', f'Space between the defence and midfield behind {p1}', ev_gap,
                            what=f'The gap between the back line and the first midfield line exceeded the professional p90 2 times; {p1} and {opp} were nearest.',
                            why='Space between the lines can be used by opponents receiving between them.',
                            whatToImprove='Keep the midfield line within about 12 m of the back line.', howToPractice='Compact lines practice.',
                            drill='compact_lines', metrics=[{'name': 'gapDM', 'value': 21.4, 'unit': 'm'}, {'name': 'Occurrences', 'value': 2, 'unit': ''}],
                            video=f'Watch the space behind {p1} when the ball is wide.', confidence='medium',
                            confidenceNote='Registration medium on part of the measured time.')]
    reviews = [finding('line_height_and_gap', 'Review opportunity', 'A high line together with a large gap', ev_review,
                       what=f'At one observed instance the back line stood 38.2 m from its goal with {p2} 17.9 m ahead of it.',
                       why='Context decides whether this was a deliberate high line.')]
    drill = {'id': 'compact_lines', 'name': 'Compact lines', 'purpose': 'Keep the distance between the units short.',
             'tacticalIssue': 'Large spaces between the back line and midfield.', 'players': {'min': 8, 'max': 12, 'note': 'Two lines of four.'},
             'durationMinutes': {'min': 15, 'max': 20, 'note': 'Four blocks.'}, 'setup': 'Two lines of four in a 40 x 30 m area.',
             'instructions': ['Shift with the ball'], 'coachingPoints': ['Move together', 'Talk across the line'], 'progressions': ['Narrow the area'],
             'safety': ['Warm up first'], 'variation': 'Play 7 v 7 at match speed.', 'level': context, 'source': 'Conventional unit shifting.'}
    report = {'version': 1, 'engineVersion': episodes.ENGINE_VERSION, 'fingerprint': None, 'generated': '2026-10-18T19:00:00',
              'kind': 'segment', 'context': context, 'team': team, 'teamName': None, 'video': {'name': 'clip.avi', 'key': '0'*16}, 'run': 'run',
              'range': {'t0': t0, 't1': t1, 'segmentT0': 0.0, 'segmentT1': round(t1-t0, 2), 'seconds': round(t1-t0, 2)},
              'overview': {'title': 'SEGMENT COACHING REPORT',
                           'coverage': {'rangeSeconds': 11.97, 'eligibleShare': .95, 'metresEstimated': True,
                                        'observations': {'total': 4, 'shown': 3}, 'measuredSeconds': {'spacing': 11.9, 'movement': 4.0}},
                           'limitations': ['Segment report: fewer than 20 minutes measured, so recurring patterns are not claimed.',
                                           'Lateral shift was measured for only 4.0 s, below the 20 s needed, so it is not assessed.'],
                           'context': ['The ball was in the middle third for most of the measured time.']},
              'summary': [f'The width held well; the space behind {p1} opened 2 times.'],
              'strengths': strengths, 'improvements': improvements, 'reviews': reviews,
              'breakdown': {'defensiveOrganization': {'available': True, 'reason': None, 'findings': [ids['line_height_and_gap']], 'notes': []},
                            'spacing': {'available': True, 'reason': None, 'findings': [ids['compactness'], ids['unit_gap_dm']], 'notes': []},
                            'formation': {'available': False, 'reason': 'Fewer than 60 s with every outfield player observed.', 'findings': [], 'notes': []},
                            'attackingStructure': {'available': False, 'reason': 'No possession record: attacking structure is not assessed.', 'findings': [], 'notes': []},
                            'movementSupport': {'available': True, 'reason': None, 'findings': [], 'notes': ['Lateral shift was measured for only 4.0 s.']}},
              'priorities': [{'title': 'Close the space between the lines', 'finding': ids['unit_gap_dm'], 'drill': 'compact_lines'}],
              'training': [{'drill': drill, 'findings': [ids['unit_gap_dm']]}],
              'confidence': {'overall': 'medium', 'notes': ['Registration MEDIUM on 40% of the measured time.']},
              'evidence': ev_gap+ev_review, 'observations': observations, 'feedback': []}
    if empty:
        report.update(strengths=[], improvements=[], reviews=[], summary=[], priorities=[], training=[], evidence=[],
                      observations=[observations[-1]],
                      breakdown={k: {'available': False, 'reason': 'Not enough measured time.', 'findings': [], 'notes': []}
                                 for k in report['breakdown']})
    return json.loads(json.dumps(report, allow_nan=False))


def write_video(path, frames=400, width=64, height=36, fps=FPS):
    """A small MJPG video whose frame k is a flat grey of (k % 32) * 8: adjacent frames differ by 8 grey levels."""
    import cv2
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'MJPG'), fps, (width, height))
    for k in range(frames):
        writer.write(np.full((height, width, 3), (k % 32)*8, np.uint8))
    writer.release()
    return path


def grey_of(jpeg):
    import cv2
    image = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_GRAYSCALE)
    return float(image.mean())


# ---------- tests ----------
OPTIONS = {'team': 'A', 't0': None, 't1': None, 'context': 'competitive'}


class OptionsAndFingerprint(unittest.TestCase):
    def test_options_whole_custom_and_refused(self):
        bounds = (1.0, 13.0)
        self.assertEqual(store.clean_options('A', bounds=bounds), OPTIONS)
        self.assertEqual(store.clean_options('B', '2.004', 9.996, 'advanced', bounds), {'team': 'B', 't0': 2.0, 't1': 10.0, 'context': 'advanced'})
        self.assertEqual(store.clean_options('A', 9, 2, None, bounds)['t0'], 2.0, 'the ends are ordered')
        self.assertEqual(store.clean_options('A', 0, 99, None, bounds), OPTIONS, 'a range covering the segment is the whole segment')
        self.assertEqual(store.clean_options('A', None, 8, None, bounds)['t0'], 1.0, 'a missing end is the segment end')
        self.assertEqual(store.clean_options('A', -5, 8, None, bounds)['t0'], 1.0, 'clamped to the segment')
        for bad in [('C',), ('A', 2, 4, None, bounds), ('A', 1, 5, 'pro', bounds), ('A', 'x', 5, None, bounds), ('A', math.nan, 5, None, bounds),
                    ('A', True, 9, None, bounds), ('A', 20, 30, None, bounds), (None,), ('A', [1], 5, None, bounds), (['A'],)]:
            with self.assertRaises(ValueError, msg=bad):
                store.clean_options(*bad)
        with self.assertRaises(ValueError) as outside:
            store.clean_options('A', 20, 30, None, (42.433, 222.4))
        self.assertIn('inside the analysed segment (00:42.43-03:42.40, source-video time)', str(outside.exception), 'the message names the segment')

    def test_hundredths_are_kept_as_the_page_sends_them(self):
        # The page rounds a typed time once (00:10.125 -> 10.13) and sends that value in both requests; the store keeps
        # it (round(10.125, 2) is 10.12: rounding here as well would have keyed the build on another range).
        for typed in (10.13, 10.12, 42.43, 1.0, 59.99, 123.45):
            self.assertEqual(store.clean_options('A', typed, typed+20, None, (0.0, 300.0))['t0'], typed)
        self.assertEqual(store.clean_options('A', '10.13', '30.13', None, (0.0, 300.0))['t0'], 10.13, 'the status request (a query string)')

    def test_pdf_filename_names_the_level(self):
        name = store.pdf_filename({'context': 'advanced'}, 'Match clip (1).mp4', '20261018-190000-abcdef', dict(OPTIONS, context='development'))
        self.assertEqual(name, 'pitchiq-coaching-Match-clip-1-20261018-190000-abcdef-A-development.pdf')
        self.assertEqual(store.pdf_filename({'context': 'advanced'}, None, 'r1', dict(OPTIONS, context=None)), 'pitchiq-coaching-video-r1-A-advanced.pdf')
        self.assertEqual(store.pdf_filename({}, None, 'r1', OPTIONS, '2bac432528d0e568'), 'pitchiq-coaching-2bac432528d0e568-r1-A-competitive.pdf')
        from soccer.coaching import pdf
        if hasattr(pdf, 'pdf_filename'):
            self.assertEqual(store.pdf_filename({}, 'clip.avi', 'r1', OPTIONS), pdf.pdf_filename('clip.avi', 'r1', 'A', 'competitive'), "the PDF writer's helper")
        with mock.patch.dict(sys.modules, {'soccer.coaching.pdf': types.ModuleType('soccer.coaching.pdf')}), \
                mock.patch.object(soccer.coaching, 'pdf', types.ModuleType('soccer.coaching.pdf'), create=True):
            self.assertEqual(store.pdf_filename({}, 'Match clip (1).mp4', 'r1', dict(OPTIONS, context='advanced')),
                             'pitchiq-coaching-Match-clip-1-r1-A-advanced.pdf', 'the same name without the helper')

    def test_fingerprint_keys_every_input_and_not_the_feedback(self):
        base = dict(result_sha='a'*64, settings={'direction': {'A': None}}, options=OPTIONS, engine_info={'version': '1', 'code': 'c'*16},
                    reference_sha='r'*64, drills_sha='d'*64)
        inputs = store.fingerprint_inputs(**base)
        fp = store.fingerprint(inputs)
        self.assertRegex(fp, r'^[0-9a-f]{64}$')
        self.assertEqual(fp, store.fingerprint(store.fingerprint_inputs(**base)), 'deterministic')
        self.assertNotIn('feedback', json.dumps(inputs))
        changes = {'result_sha': 'b'*64, 'settings': {'direction': {'A': 'left'}}, 'options': dict(OPTIONS, context='advanced'),
                   'engine_info': {'version': '2', 'code': 'c'*16}, 'reference_sha': 's'*64, 'drills_sha': 'e'*64}
        for key, value in changes.items():
            self.assertNotEqual(fp, store.fingerprint(store.fingerprint_inputs(**dict(base, **{key: value}))), key)
        # The settings are hashed canonically: key order does not matter.
        a = store.settings_sha({'lineup': {'A': 10, 'B': None}, 'direction': {'A': 'left', 'switchAt': None}})
        b = store.settings_sha({'direction': {'switchAt': None, 'A': 'left'}, 'lineup': {'B': None, 'A': 10}})
        self.assertEqual(a, b)
        # Missing reference or drill files are named, not hashed.
        with mock.patch.object(store, 'REFERENCE_FILE', Path(tempfile.gettempdir())/'no-such-reference.json'), \
                mock.patch.object(store, 'DRILLS_FILE', Path(tempfile.gettempdir())/'no-such-drills.json'):
            missing = store.fingerprint_inputs('a'*64, {}, OPTIONS)
        self.assertEqual((missing['reference'], missing['drills']), ('missing', 'missing'))

    def test_engine_names_the_running_code(self):
        e = store.engine()
        self.assertEqual(e['version'], episodes.ENGINE_VERSION)
        self.assertRegex(e['code'], r'^[0-9a-f]{16}$')
        self.assertEqual(store.code_sha(), store.code_sha())
        self.assertIn('store.py', store.CODE_EXCLUDED, 'storage changes never make a report stale')
        # Every soccer/ module the series and the evidence read is hashed (field.py decides the overlay tiers).
        names = {p.name for p in store.CODE_EXTRA}
        self.assertTrue({'team_shape.py', 'analytics.py', 'labels.py', 'pitch_model.py', 'registration.py', 'field.py', 'geometry.py'} <= names)
        self.assertTrue(all(p.exists() for p in store.CODE_EXTRA))
        import ast
        read = set()
        for name in ('series.py', 'evidence.py', 'detectors.py'):
            tree = ast.parse((store.PACKAGE/name).read_text(encoding='utf-8'))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.level == 2:
                    read.update(a.name for a in node.names)
        self.assertTrue(read, 'the coaching modules import soccer/ modules')
        self.assertLessEqual({f'{m}.py' for m in read}, names, 'every soccer/ module imported by the engine is in the hash')
        with mock.patch.dict(store._code, clear=True):
            before = store.code_sha()
            store._code.clear()
            with mock.patch.object(Path, 'read_bytes', lambda p, real=Path.read_bytes: real(p)+(b'#' if p.name == 'field.py' else b'')):
                self.assertNotEqual(store.code_sha(), before, 'an edit to field.py makes the reports stale')
            store._code.clear()

    def test_staleness_reasons(self):
        stored = store.fingerprint_inputs('a'*64, {}, OPTIONS, {'version': '1', 'code': 'c'*16}, 'r'*64, 'd'*64)
        self.assertEqual(store.staleness(stored, dict(stored)), [])
        now = dict(stored, result='b'*16)
        self.assertEqual([r['code'] for r in store.staleness(stored, now)], ['result'])
        made = store.staleness(stored, now, ['c1'], ['c1', 'c2', 'c3'])[0]['text']
        self.assertIn('2 corrections made since this report', made)
        undone = store.staleness(stored, now, ['c1', 'c2'], ['c2'])[0]['text']
        self.assertIn('1 correction undone', undone)
        both = dict(stored, settings='x'*16, engine={'version': '2', 'code': 'c'*16}, reference='q'*16, drills='p'*16)
        self.assertEqual([r['code'] for r in store.staleness(stored, both)], ['settings', 'engine', 'reference', 'drills'])
        self.assertIn('team-shape settings changed', store.staleness(stored, both)[0]['text'])
        self.assertEqual(store.applied_corrections([{'id': 'c2', 'applied': True}, {'id': 'c1', 'applied': False}, {'id': 'c0'}, 'x']), ['c0', 'c2'])


class ReportFiles(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.run = Path(self.tmp.name)/'video'/'run'
        self.run.mkdir(parents=True)
        self.result = synthetic_result(seconds=6)
        self.report = fixture_report(self.result)

    def tearDown(self):
        self.tmp.cleanup()

    def inputs(self, sha='a'*64, options=OPTIONS):
        return store.fingerprint_inputs(sha, {}, options, {'version': '1', 'code': 'c'*16}, 'r'*64, 'd'*64)

    def test_fresh_stale_and_kept(self):
        first = self.inputs()
        path = store.save_report(self.run, 'A', first, self.report, corrections=[])
        fp = store.fingerprint(first)
        self.assertEqual(path, self.run/'coaching'/f'report-A-{fp[:16]}.json')
        record = json.loads(path.read_text(encoding='utf-8'))
        self.assertEqual((record['fingerprint'], record['report']['fingerprint'], record['corrections']), (fp, fp, []))
        self.assertEqual(store.find_report(self.run, OPTIONS, fp)[1], True)
        later = self.inputs('b'*64)
        found, fresh = store.find_report(self.run, OPTIONS, store.fingerprint(later))
        self.assertEqual((found['fingerprint'], fresh), (fp, False), 'the earlier report, stale')
        self.assertEqual(store.find_report(self.run, dict(OPTIONS, context='advanced'), fp), (None, False), 'other options: no report')
        self.assertEqual(store.find_report(self.run, dict(OPTIONS, team='B'), fp), (None, False))
        self.assertEqual(store.evidence_frames(record), {e['frame'] for e in self.report['evidence']})
        for k in range(store.KEEP_REPORTS+2):
            os.utime(path, (1, 1)) if k == 0 else None
            store.save_report(self.run, 'A', self.inputs(f'{k:x}'*64), self.report)
        self.assertEqual(len(list((self.run/'coaching').glob('report-A-*.json'))), store.KEEP_REPORTS)
        store.save_report(self.run, 'B', self.inputs(options=dict(OPTIONS, team='B')), fixture_report(self.result, 'B'))
        self.assertEqual(len(list((self.run/'coaching').glob('report-*.json'))), store.KEEP_REPORTS+1, 'kept per team')

    def test_a_report_with_a_non_finite_number_is_not_stored(self):
        bad = dict(self.report, summary=[math.nan])
        with self.assertRaises(ValueError):
            store.save_report(self.run, 'A', self.inputs(), bad)
        self.assertEqual(list((self.run/'coaching').glob('*')), [], 'no file and no temporary file left')

    def test_records_follow_the_file(self):
        path = store.save_report(self.run, 'A', self.inputs(), self.report)
        first = store.load_record(path)
        data = json.loads(path.read_text(encoding='utf-8'))
        data['saved'] = '2030-01-01T00:00:00'
        path.write_text(json.dumps(data), encoding='utf-8')
        os.utime(path, ns=(path.stat().st_mtime_ns+10**9, path.stat().st_mtime_ns+10**9))
        self.assertEqual(store.load_record(path)['saved'], '2030-01-01T00:00:00')
        self.assertNotEqual(first['saved'], '2030-01-01T00:00:00')
        path.write_text('{not json', encoding='utf-8')
        self.assertIsNone(store.load_record(path))
        self.assertEqual(store.records(self.run), [])


class AtomicWrites(unittest.TestCase):
    def test_replace_retries_a_windows_reader_and_gives_up_cleanly(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'x.json'
            real, calls = os.replace, []

            def flaky(a, b):
                calls.append(1)
                if len(calls) < 3:
                    raise PermissionError('in use')
                real(a, b)
            with mock.patch.object(store.os, 'replace', flaky), mock.patch.object(store, 'REPLACE_WAIT', 0):
                store.write_json(path, {'a': 1})
            self.assertEqual((json.loads(path.read_text()), len(calls)), ({'a': 1}, 3))
            with mock.patch.object(store.os, 'replace', mock.Mock(side_effect=PermissionError('in use'))), mock.patch.object(store, 'REPLACE_WAIT', 0):
                with self.assertRaises(PermissionError):
                    store.write_json(path, {'a': 2})
            self.assertEqual(sorted(p.name for p in Path(tmp).iterdir()), ['x.json'], 'the temporary file is removed')
            self.assertEqual(json.loads(path.read_text()), {'a': 1})


class Feedback(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.video = Path(self.tmp.name)
        self.report = fixture_report(synthetic_result(seconds=6))
        self.record = {'fingerprint': 'f'*64, 'report': self.report, 'inputs': {}}
        self.gap = self.report['improvements'][0]['observation']

    def tearDown(self):
        self.tmp.cleanup()

    def stored(self):
        return json.loads(store.feedback_path(self.video).read_text(encoding='utf-8'))

    def test_add_replace_and_remove(self):
        e = store.add_feedback(self.video, 'run1', self.record, self.gap, 'correct', '  right call  ', now=datetime(2026, 10, 18, 19, 3, 12))
        self.assertRegex(e['id'], r'^f20261018-190312-[0-9a-f]{6}$')
        self.assertRegex(e['id'], store.FEEDBACK_ID)
        o = e['observation']
        self.assertEqual((e['run'], e['report'], e['verdict'], e['text']), ('run1', 'f'*16, 'correct', 'right call'))
        self.assertEqual((o['key'], o['detector'], o['team'], o['classification']), (self.gap, 'unit_gap_dm', 'A', 'improvement'))
        self.assertEqual(o['headline'], self.report['improvements'][0]['headline'])
        self.assertEqual((self.stored()['version'], self.stored()['adjustments']), (store.FEEDBACK_VERSION, []))
        again = store.add_feedback(self.video, 'run1', self.record, self.gap, 'partial')
        other_run = store.add_feedback(self.video, 'run2', self.record, self.gap, 'incorrect')
        entries = store.read_feedback(self.video)
        self.assertEqual([x['id'] for x in entries], [again['id'], other_run['id']], 'one verdict per finding and run')
        self.assertTrue(store.remove_feedback(self.video, again['id']))
        self.assertFalse(store.remove_feedback(self.video, again['id']))
        self.assertEqual([x['id'] for x in store.read_feedback(self.video)], [other_run['id']])

    def test_refused_without_writing(self):
        with self.assertRaises(KeyError):
            store.add_feedback(self.video, 'run1', self.record, 'obs-000000000000', 'correct')
        with self.assertRaises(ValueError):
            store.add_feedback(self.video, 'run1', self.record, self.gap, 'agree')
        with self.assertRaises(ValueError):
            store.add_feedback(self.video, 'run1', self.record, self.gap, 'correct', 'x'*(store.MAX_NOTE+1))
        self.assertFalse(store.feedback_path(self.video).exists())

    def test_an_unreadable_file_is_never_overwritten(self):
        path = store.feedback_path(self.video)
        path.write_text('{"version": 1, "feedback": [', encoding='utf-8')
        self.assertEqual(store.read_feedback(self.video), [], 'served as no feedback')
        with self.assertRaises(store.FeedbackUnreadable):
            store.add_feedback(self.video, 'run1', self.record, self.gap, 'correct')
        with self.assertRaises(store.FeedbackUnreadable):
            store.remove_feedback(self.video, 'f20261018-190312-abcdef')
        self.assertEqual(path.read_text(encoding='utf-8'), '{"version": 1, "feedback": [', 'left as it was')
        path.write_text('{"feedback": {"not": "a list"}}', encoding='utf-8')
        with self.assertRaises(store.FeedbackUnreadable):
            store.add_feedback(self.video, 'run1', self.record, self.gap, 'correct')
        # A read that keeps failing (the file held by another process) is an error before a write, not an empty list.
        path.write_text('{"version": 1, "feedback": []}', encoding='utf-8')
        with mock.patch.object(Path, 'read_text', side_effect=PermissionError('locked')), mock.patch.object(store, 'READ_WAIT', 0):
            self.assertEqual(store.read_feedback(self.video), [])
            with self.assertRaises(store.FeedbackUnreadable):
                store.read_feedback(self.video, strict=True)

    def test_entries_it_does_not_understand_are_kept(self):
        foreign = {'id': 'x1', 'something': 'from a later version'}
        store.write_json(store.feedback_path(self.video), {'version': 1, 'feedback': [foreign]})
        store.add_feedback(self.video, 'run1', self.record, self.gap, 'correct')
        self.assertEqual(self.stored()['feedback'][0], foreign)
        self.assertEqual(len(store.read_feedback(self.video)), 1, 'served: only well-formed entries')

    def test_a_verdict_keeps_what_identifies_its_finding(self):
        e = store.add_feedback(self.video, 'run1', self.record, self.gap, 'incorrect')
        obs = next(o for o in self.report['observations'] if o['id'] == self.gap)
        self.assertEqual({k: e['observation'][k] for k in ('pattern', 'context', 'players', 'detector', 'team')},
                         {'pattern': obs['pattern'], 'context': obs['baseline']['context'], 'players': sorted(obs['players']),
                          'detector': 'unit_gap_dm', 'team': 'A'})
        self.assertEqual((e['run'], e['level']), ('run1', 'competitive'))
        self.assertEqual(e['observation']['headline'], self.report['improvements'][0]['headline'], 'in id form: labels renumber')

    def test_attached_only_to_the_same_finding_of_the_same_run(self):
        e = store.add_feedback(self.video, 'run1', self.record, self.gap, 'correct')
        out = store.attach_feedback(self.report, store.read_feedback(self.video), 'run1')
        self.assertEqual((out[0]['attached'], out[0]['thisRun']), (self.gap, True))
        # Another run: ids leave the run out and team letters are arbitrary per run, so not even the same id attaches.
        other = store.attach_feedback(self.report, [e], 'run2')[0]
        self.assertEqual((other['attached'], other['thisRun']), (None, False))
        # The same run regenerated (another range or engine version): new ids, the same detector, team, pattern and
        # players over an overlapping period.
        moved = json.loads(json.dumps(self.report))
        for o in moved['observations']:
            o['id'] = o['id'].replace('obs-', 'obs-9')[:16]
            o['window']['t0'] += .5
        target = next(o['id'] for o in moved['observations'] if o['detector'] == 'unit_gap_dm')
        self.assertEqual(store.attach_feedback(moved, [e], 'run1')[0]['attached'], target)
        self.assertIsNone(store.attach_feedback(moved, [e], 'run2')[0]['attached'], 'never across runs')
        # Every observation of a detector shares the analysed range's window: another pattern (team area instead of
        # width), other players or another team is another finding.
        for change in ({'pattern': 'team_area'}, {'players': ['A-03']}, {'team': 'B'}):
            changed = json.loads(json.dumps(moved))
            for o in changed['observations']:
                if o['id'] == target:
                    o.update(change)
            self.assertIsNone(store.attach_feedback(changed, [e], 'run1')[0]['attached'], change)
        # Two observations that are the same finding but for the comparison context: the one with the stored context.
        twin = json.loads(json.dumps(moved))
        first = next(o for o in twin['observations'] if o['id'] == target)
        second = dict(json.loads(json.dumps(first)), id='obs-9aaaaaaaaaaa', baseline=dict(first['baseline'], context='third:middle'))
        twin['observations'].append(second)
        self.assertEqual(store.attach_feedback(twin, [e], 'run1')[0]['attached'], target)
        first['baseline']['context'] = 'third:own'
        self.assertIsNone(store.attach_feedback(twin, [e], 'run1')[0]['attached'], 'ambiguous: none')
        # An entry stored before patterns were kept attaches by its id only.
        old = json.loads(json.dumps(e))
        old['observation'].pop('pattern')
        self.assertEqual(store.attach_feedback(self.report, [old], 'run1')[0]['attached'], self.gap)
        self.assertIsNone(store.attach_feedback(moved, [old], 'run1')[0]['attached'])
        far = json.loads(json.dumps(moved))
        for o in far['observations']:
            o['window']['t0'], o['window']['t1'] = 500.0, 600.0
        self.assertIsNone(store.attach_feedback(far, [e], 'run1')[0]['attached'], 'no longer reported')
        self.assertIsNone(store.attach_feedback(None, [e], 'run1')[0]['attached'])

    def test_non_finite_numbers_never_come_out(self):
        path = store.feedback_path(self.video)
        e = store.add_feedback(self.video, 'run1', self.record, self.gap, 'correct')
        text = path.read_text(encoding='utf-8')
        path.write_text(text.replace('"t0": ' + json.dumps(e['observation']['t0']), '"t0": NaN').replace('"t1": ' + json.dumps(e['observation']['t1']), '"t1": 1e999'),
                        encoding='utf-8')
        self.assertIn('NaN', path.read_text(encoding='utf-8'))
        entries = store.read_feedback(self.video)
        self.assertEqual((entries[0]['id'], entries[0]['observation']['t0'], entries[0]['observation']['t1']), (e['id'], None, None),
                         'a hand-edited NaN or overflow reads as null; the entry is kept')
        json.dumps(store.attach_feedback(self.report, entries, 'run1'), allow_nan=False)
        self.assertEqual(json.loads(store.json_text({'a': [math.nan, 1.5, {'b': math.inf}]})), {'a': [None, 1.5, {'b': None}]})
        self.assertNotIn('NaN', store.json_text({'a': float('nan')}))
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)/'run'
            saved = store.save_report(run, 'A', store.fingerprint_inputs('a'*64, {}, OPTIONS, {'version': '1', 'code': 'c'*16}, 'r'*64, 'd'*64), self.report)
            self.assertIsNotNone(store.load_record(saved))
            saved.write_text(saved.read_text(encoding='utf-8').replace('"eligibleShare":0.95', '"eligibleShare":NaN', 1), encoding='utf-8')
            self.assertIsNone(store.load_record(saved), 'a stored report with a NaN is damaged: not served')

    def test_ids_are_matched_whole(self):
        self.assertTrue(store.is_observation_id(self.gap))
        self.assertTrue(store.is_feedback_id('f20261018-190312-abcdef'))
        for bad in (self.gap+'\n', ' '+self.gap, 'obs-xyz', ['obs-abcdef'], None):
            self.assertFalse(store.is_observation_id(bad), bad)
        for bad in ('f20261018-190312-abcdef\n', '../x', 5):
            self.assertFalse(store.is_feedback_id(bad), bad)


class LabelsAndScreenshots(unittest.TestCase):
    def test_labels_of_the_result_as_it_is_now(self):
        result = synthetic_result(seconds=2)
        result['match']['retired'] = [{'id': 'A-12', 'retiredInto': 'A-03'}]
        labels = store.labels_of(result)
        self.assertEqual((labels['A-03'], labels['A-02'], labels['B-07'], labels['A-01'], labels['A-12']), ('A-02', 'A-03', 'B-05', 'A-01', 'A-02'))
        self.assertEqual(store.labels_of(None), {})

    def test_an_excluded_identity_never_shows_as_a_label_that_is_someone_elses_now(self):
        # As corrections.apply_result leaves a result after "A-03 is not part of the match": the entry moves to
        # match.excluded keeping its display, and the labels renumber (A-04 shows as A-03).
        result = synthetic_result(seconds=2)
        match = result['match']
        gone = next(e for e in match['players'] if e['id'] == 'A-04')
        match['players'].remove(gone)
        match['excluded'] = [dict(gone, excluded=True, status='EXCLUDED'), {'id': None, 'track': 7, 'display': 'OUT'}]
        match['withdrawn'] = [{'id': 'B-12', 'display': 'B-10', 'status': 'WITHDRAWN'}, {'id': 'B-13', 'display': 'OUT'}]
        result['roster']['slots'] = [s for s in result['roster']['slots'] if s['id'] != 'A-04']
        for e in match['players']:
            if e['team'] == 'A' and e['id'] > 'A-04':
                e['display'] = f"A-{int(e['id'][2:])-1:02d}"
        labels = store.labels_of(result)
        self.assertEqual(labels['A-04'], 'excluded (was A-04)')
        self.assertEqual(labels['A-05'], 'A-04')
        self.assertEqual((labels['B-12'], labels['B-13']), ('withdrawn (was B-10)', 'withdrawn (was B-13)'))
        team_a = [v for k, v in labels.items() if k.startswith('A-')]
        self.assertEqual(len(set(team_a)), len(team_a), 'no two team A identities share a shown label')
        result['roster']['slots'].append({'id': 'A-04', 'display': 'A-04'})
        self.assertEqual(store.labels_of(result)['A-04'], 'excluded (was A-04)', 'over a roster slot left behind')

    def test_the_screenshot_is_the_exact_source_frame_and_cached(self):
        with tempfile.TemporaryDirectory() as tmp:
            video = write_video(Path(tmp)/'clip.avi', frames=80)
            folder = Path(tmp)/'runs'/'v'
            for frame in (0, 17, 18, 63):
                data = store.frame_jpeg(video, folder, frame, 640)
                self.assertEqual(data[:2], b'\xff\xd8')
                self.assertAlmostEqual(grey_of(data), (frame % 32)*8, delta=3, msg=frame)
            cached = store.frame_cache_path(folder, 17, 640)
            self.assertEqual(cached, folder/'coaching'/'frames'/'f17-w640.jpg')
            self.assertTrue(cached.exists())
            os.remove(video)
            self.assertAlmostEqual(grey_of(store.frame_jpeg(video, folder, 17, 640)), 17*8, delta=3, msg='served from the cache')
            with self.assertRaises(store.FrameUnavailable):
                store.frame_jpeg(video, folder, 5, 640)
            with self.assertRaises(ValueError):
                store.frame_jpeg(video, folder, 17, 800)

    def test_a_frame_past_the_end_or_at_another_time_is_unavailable(self):
        with tempfile.TemporaryDirectory() as tmp:
            video = write_video(Path(tmp)/'clip.avi', frames=10)
            with self.assertRaises(store.FrameUnavailable):
                store.frame_jpeg(video, Path(tmp)/'v', 50, 640)
            self.assertAlmostEqual(grey_of(store.frame_jpeg(video, Path(tmp)/'v', 4, 640, 4/FPS, FPS)), 32, delta=3)
            with self.assertRaises(store.FrameUnavailable, msg='the decoder must agree with the evidence time'):
                store.frame_jpeg(video, Path(tmp)/'v', 6, 640, 5/FPS, FPS)

    def test_the_pdf_gets_only_the_reports_evidence_frames(self):
        with tempfile.TemporaryDirectory() as tmp:
            video = write_video(Path(tmp)/'clip.avi', frames=60)
            folder = Path(tmp)/'v'
            report = {'evidence': [{'frame': 12, 'time': 12/FPS}, {'frame': 40, 'time': 40/FPS}, {'frame': -1}, {'frame': 'x'}]}
            frames = store.EvidenceFrames(video, folder, report, FPS)
            self.assertEqual(frames.allowed, {12, 40})
            got = frames.prefetch([12, 13, 40], [1280, 999])
            self.assertEqual(sorted(got['jpegs']), [(12, 1280), (40, 1280)])
            self.assertAlmostEqual(grey_of(frames(40, 1280)), 64, delta=3)
            self.assertIsNone(frames(13, 1280))
            self.assertIsNone(frames(12, 800))
            self.assertTrue(store.frame_cache_path(folder, 12, 1280).exists(), "shared with the page's screenshots")


class ProvidedByThePackage(unittest.TestCase):
    """The report and the PDF come from the rest of the package; until it provides them nothing stands in."""

    def test_no_engine_no_report(self):
        with mock.patch.object(soccer.coaching, 'build_report_for', None, create=True):
            with self.assertRaises(store.EngineUnavailable):
                store.build_report_for('result.json', 'A', {})
        with mock.patch.object(soccer.coaching, 'build_report_for', store.build_report_for, create=True):
            with self.assertRaises(store.EngineUnavailable):
                store.build_report_for('result.json', 'A', {})
        engine = mock.Mock(return_value={'version': 1})
        with mock.patch.object(soccer.coaching, 'build_report_for', engine, create=True):
            self.assertEqual(store.build_report_for('result.json', 'B', {'context': 'advanced'}), {'version': 1})
        engine.assert_called_once_with('result.json', 'B', {'context': 'advanced'})

    def test_pdf_writer(self):
        empty = types.ModuleType('soccer.coaching.pdf')
        with mock.patch.dict(sys.modules, {'soccer.coaching.pdf': empty}), mock.patch.object(soccer.coaching, 'pdf', empty, create=True):
            with self.assertRaises(store.PdfUnavailable):
                store.render_pdf({}, {}, lambda f, w=1280: None)
        writer = types.ModuleType('soccer.coaching.pdf')
        writer.report_pdf = mock.Mock(return_value=b'%PDF-1.4')
        frame = lambda f, w=1280: None
        with mock.patch.dict(sys.modules, {'soccer.coaching.pdf': writer}), mock.patch.object(soccer.coaching, 'pdf', writer, create=True):
            self.assertEqual(store.render_pdf({'team': 'A'}, {'A-01': 'A-01'}, frame, ['stale']), b'%PDF-1.4')
        writer.report_pdf.assert_called_once_with({'team': 'A'}, labels={'A-01': 'A-01'}, frame=frame, stale=['stale'])
        with mock.patch.dict(sys.modules, {'soccer.coaching.pdf': writer}), mock.patch.object(soccer.coaching, 'pdf', writer, create=True):
            store.render_pdf({'team': 'A'}, {}, frame, None, feedback=[], colours={'A': '#ffffff'})
        writer.report_pdf.assert_called_with({'team': 'A'}, labels={}, frame=frame, stale=None, feedback=[], colours={'A': '#ffffff'})


# =====================================================================================================================
# v2: the adaptive coaching store (DESIGN v2 1.2, 2.1, 8, 10)
# =====================================================================================================================
_V2 = {}


def v2_scene():
    """EVAL's short_segment scene (tools/coaching_synth.py) with its result bytes, sha, segment and default settings; the
    obs of both teams measured once per process through store.measure (about 1 s)."""
    if not _V2:
        tools = str(Path(__file__).resolve().parent.parent/'tools')
        if tools not in sys.path:
            sys.path.insert(0, tools)
        import coaching_synth
        b = coaching_synth.build('short_segment')
        data = json.dumps(b.result, separators=(',', ':'), allow_nan=False).encode('utf-8')
        from soccer import team_shape
        shape = team_shape.clean_settings({})
        settings = {'version': 1, 'teams': {'A': None, 'B': None}, 'updated': None}
        sha = store.sha256_bytes(data)
        _V2.update(built=b, data=data, sha=sha, shape=shape, settings=settings, segment=store.segment_of(b.result['frames']))
        _V2['obs'] = store.measure(json.loads(data), result_sha=sha, shape_settings=shape, settings=settings)
    return _V2


def team_block(level=4, age='U14', **change):
    from soccer.coaching import profiles
    return profiles.clean_team(dict({'ageGroup': age, 'playingLevel': level, 'format': {'players': 11, 'goalkeeper': True}},
                                    **change), now='2026-10-22T10:00:00')


def with_block(settings, team, block):
    out = json.loads(json.dumps(settings))
    out['teams'][team] = block
    return out


class SettingsV2(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.video = Path(self.tmp.name)
        self.squads = {'sq-0123456789ab'}

    def tearDown(self):
        self.tmp.cleanup()

    def save(self, team, raw, **kw):
        return store.save_team_settings(self.video, team, raw, now=datetime(2026, 10, 22, 10, 5, 0),
                                        squad_exists=lambda sq: sq in self.squads, **kw)

    def test_saved_cleaned_with_the_tiers_it_makes_stale(self):
        self.assertEqual(store.read_settings(self.video), {'version': 1, 'teams': {'A': None, 'B': None}, 'updated': None})
        first = self.save('A', {'ageGroup': 'u12', 'playingLevel': 3})
        self.assertEqual((first['new']['ageGroup'], first['new']['format'], first['settings']['updated']),
                         ('U12', {'players': 9, 'goalkeeper': True}, '2026-10-22T10:05:00'), 'filled as profiles.clean_team fills')
        self.assertEqual(first['stale'], ['measurement', 'role', 'interpretation'], '9v9 instead of the 11v11 default')
        self.assertEqual(store.read_settings(self.video), first['settings'])
        level = self.save('A', dict(first['new'], playingLevel=4))
        self.assertEqual(level['stale'], ['interpretation'])
        roles = self.save('A', dict(level['new'], roles={'A-02': [{'role': 'back-1/3', 'from': None, 'to': None}]}))
        self.assertEqual(roles['stale'], ['role', 'interpretation'])
        self.assertEqual(roles['new']['roles']['A-02'][0]['set'], '2026-10-22T10:05:00', 'the save time stamps a new confirmation')
        squad = self.save('A', dict(roles['new'], squad={'id': 'sq-0123456789ab', 'name': 'Blue'},
                                     runLinks=[{'run': 'r1', 'team': 'A', 'confirmed': True}]))
        self.assertEqual(squad['stale'], [], 'squad and run links are read at serve time only')
        self.assertIsNone(store.read_settings(self.video)['teams']['B'])

    def test_refused_when_anything_would_be_dropped_or_the_links_conflict(self):
        ok = self.save('A', team_block(squad={'id': 'sq-0123456789ab', 'name': 'Blue'},
                                       runLinks=[{'run': 'r1', 'team': 'A', 'confirmed': True}]))['new']
        for team, raw, needle in [
                ('A', dict(ok, playingLevel='4.5'), 'playingLevel'), ('A', dict(ok, style='tiki'), 'style'),
                ('A', dict(ok, formation={'intended': '1-2-1', 'periods': []}), 'formation'),
                ('A', dict(ok, squad={'id': 'sq-ffffffffffff', 'name': 'Red'}), 'not a known squad'),
                ('A', dict(ok, runLinks=[{'run': 'r1', 'team': 'A', 'confirmed': True}, {'run': 'r1', 'team': 'B', 'confirmed': False}]), 'one link per run'),
                ('B', team_block(squad={'id': 'sq-0123456789ab', 'name': 'Blue'}), 'already the other team'),
                ('C', team_block(), 'team must be A or B'), ('A', [1], 'not an object')]:
            with self.assertRaises(store.SettingsRefused, msg=needle) as caught:
                self.save(team, raw)
            self.assertTrue(any(needle in p for p in caught.exception.problems), (needle, caught.exception.problems))
        self.squads.add('sq-aaaaaaaaaaaa')
        with self.assertRaises(store.SettingsRefused) as caught:
            self.save('B', team_block(squad={'id': 'sq-aaaaaaaaaaaa', 'name': 'Red'}, runLinks=[{'run': 'r1', 'team': 'A', 'confirmed': True}]))
        self.assertIn('already confirmed as Blue', caught.exception.problems[0])
        self.assertEqual(store.read_settings(self.video)['teams']['A'], ok, 'nothing refused was written')

    def test_links_to_other_runs_are_kept_while_the_squad_stays(self):
        sq = {'id': 'sq-0123456789ab', 'name': 'Blue'}
        self.save('A', team_block(squad=sq, runLinks=[{'run': 'r1', 'team': 'A', 'confirmed': True}]))
        two = self.save('A', team_block(squad=sq, runLinks=[{'run': 'r2', 'team': 'B', 'confirmed': True}]))['new']
        self.assertEqual(two['runLinks'], [{'run': 'r2', 'team': 'B', 'confirmed': True}, {'run': 'r1', 'team': 'A', 'confirmed': True}])
        self.squads.add('sq-bbbbbbbbbbbb')
        other = self.save('A', team_block(squad={'id': 'sq-bbbbbbbbbbbb', 'name': 'Green'}, runLinks=[]))['new']
        self.assertEqual(other['runLinks'], [], 'another squad does not inherit the links')

    def test_an_unreadable_file_is_never_overwritten(self):
        path = store.settings_path(self.video)
        path.write_text('{"version": 1, "teams": ', encoding='utf-8')
        self.assertEqual(store.read_settings(self.video)['teams'], {'A': None, 'B': None}, 'served as nothing saved')
        with self.assertRaises(store.SettingsUnreadable):
            self.save('A', team_block())
        self.assertEqual(path.read_text(encoding='utf-8'), '{"version": 1, "teams": ')


class CoachSpansV2(unittest.TestCase):
    SEG = {'t0': 600.0, 't1': 659.9}

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.video = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def record(self, *spans):
        return dict(store.empty_spans(), spans=[{'t0': a, 't1': b, 'state': s} for a, b, s in spans])

    def test_cleaned_against_the_segment_and_kept_per_run(self):
        self.assertEqual(store.read_spans(self.video, 'r1'), (None, None))
        out = store.save_spans(self.video, 'r1', self.record((599.6, 610, 'A'), (610, 615, 'dead'), (660.5, 670, 'B')), self.SEG,
                               now=datetime(2026, 10, 22, 11, 0, 0))
        self.assertEqual(out['spans']['spans'], [{'t0': 600.0, 't1': 610.0, 'state': 'A'}, {'t0': 610.0, 't1': 615.0, 'state': 'dead'}])
        self.assertEqual((out['dropped'], out['changed'], out['saved']), (1, True, '2026-10-22T11:00:00'))
        self.assertEqual(S.validate('coach_spans', out['spans']), [])
        self.assertEqual(store.read_spans(self.video, 'r1'), (out['spans'], '2026-10-22T11:00:00'))
        self.assertEqual(store.read_spans(self.video, 'r2'), (None, None), 'team letters belong to a run')
        again = store.save_spans(self.video, 'r1', out['spans'], self.SEG, now=datetime(2026, 10, 22, 12, 0, 0))
        self.assertEqual((again['changed'], again['saved']), (False, '2026-10-22T11:00:00'))
        store.save_spans(self.video, 'r2', self.record((620, 640, 'B')), self.SEG)
        gone = store.save_spans(self.video, 'r1', store.empty_spans(), self.SEG)
        self.assertEqual((gone['changed'], gone['saved'], gone['spans']), (True, None, store.empty_spans()))
        self.assertEqual(list(json.loads(store.spans_path(self.video).read_text(encoding='utf-8'))['runs']), ['r2'])

    def test_refused_and_unreadable(self):
        for bad in (self.record((610, 605, 'A')), self.record((600, 610, 'C')), self.record((600, 610, 'A'), (605, 612, 'B')),
                    dict(self.record(), kind='spans'), None, [1]):
            with self.assertRaises(store.SettingsRefused, msg=bad):
                store.save_spans(self.video, 'r1', bad, self.SEG)
        with self.assertRaises(store.SettingsRefused):
            store.save_spans(self.video, '..', self.record((600, 610, 'A')), self.SEG)
        self.assertFalse(store.spans_path(self.video).exists())
        store.spans_path(self.video).write_text('{"version": 1, "runs": [', encoding='utf-8')
        self.assertEqual(store.read_spans(self.video, 'r1'), (None, None))
        with self.assertRaises(store.SettingsUnreadable):
            store.save_spans(self.video, 'r1', self.record((600, 610, 'A')), self.SEG)
        store.spans_path(self.video).write_text('{"version": 1, "runs": []}', encoding='utf-8')
        with self.assertRaises(store.SettingsUnreadable):
            store.save_spans(self.video, 'r1', self.record((600, 610, 'A')), self.SEG)

    def test_the_measurement_key_follows_the_spans_only_when_there_are_some(self):
        sha = 'a'*64
        self.assertIsNone(store.spans_sha(None))
        self.assertIsNone(store.spans_sha(store.empty_spans()))
        self.assertEqual(store.measured_result_sha(sha, None), sha)
        self.assertEqual(store.measured_result_sha(sha, store.empty_spans()), sha, 'no spans: the key it had before spans existed')
        one, two = self.record((600, 610, 'A')), self.record((600, 611, 'A'))
        self.assertNotEqual(store.measured_result_sha(sha, one), sha)
        self.assertNotEqual(store.measured_result_sha(sha, one), store.measured_result_sha(sha, two))
        self.assertEqual(store.measured_result_sha(sha, one), store.measured_result_sha(sha, json.loads(json.dumps(one))))


class TiersV2(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.v = v2_scene()

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.run = Path(self.tmp.name)/'video'/'run1'

    def tearDown(self):
        self.tmp.cleanup()

    def keys(self, settings=None, team='A', spans=None, shape=None, t0=None, t1=None, sha=None):
        v = self.v
        return store.tier_keys(sha or v['sha'], shape or v['shape'], settings or v['settings'], team, spans, v['segment'], t0, t1)

    def stored(self, team='A', settings=None):
        k = self.keys(settings, team)
        store.save_tier(self.run, 'obs', k, self.v['obs'][team])
        return k

    def test_each_input_changes_only_its_tier_and_those_below(self):
        base = self.keys()
        self.assertEqual(self.v['obs']['A']['fingerprint'], base['mfp'], 'measure() keys obs as tier_keys does')
        level = self.keys(with_block(self.v['settings'], 'A', team_block(2, 'U10')))
        self.assertEqual((level['mfp'], level['rfp']), (base['mfp'], base['rfp']))
        self.assertNotEqual(level['ifp'], base['ifp'])
        roles = self.keys(with_block(self.v['settings'], 'A', team_block(roles={'A-02': [{'role': 'LB', 'from': None, 'to': None}]})))
        self.assertEqual(roles['mfp'], base['mfp'])
        self.assertNotEqual(roles['rfp'], base['rfp'])
        for changed in (self.keys(with_block(self.v['settings'], 'A', team_block(format={'players': 9, 'goalkeeper': True}))),
                        self.keys(spans=dict(store.empty_spans(), spans=[{'t0': 600.0, 't1': 620.0, 'state': 'A'}])),
                        self.keys(shape={'direction': {'A': 'left', 'switchAt': None}, 'lineup': {'A': None, 'B': None}, 'formations': []}),
                        self.keys(sha='b'*64), self.keys(team='B')):
            self.assertNotEqual(changed['mfp'], base['mfp'])
        rng = self.keys(t0=605, t1=650)
        self.assertEqual((rng['mfp'], rng['rfp'], rng['range']), (base['mfp'], base['rfp'], {'t0': 605.0, 't1': 650.0, 'whole': False}))
        self.assertNotEqual(rng['ifp'], base['ifp'])
        self.assertEqual(base['range'], {'t0': None, 't1': None, 'whole': True})
        self.assertEqual(self.keys(team='B', settings=with_block(self.v['settings'], 'A', team_block(2, 'U10')))['ifp'], self.keys(team='B')['ifp'],
                         "one team's settings never reach the other's report")

    def test_segment_and_range_are_those_the_package_uses(self):
        from soccer import coaching
        result = self.v['built'].result
        self.assertEqual(store.segment_of(result['frames']), {k: coaching._segment(result)[k] for k in ('t0', 't1')})
        obs = {'segment': self.v['obs']['A']['segment']}
        for t0, t1 in [(None, None), (605.0, 650.0), (590.0, 700.0), (None, 640.0), (650.0, 605.0), (620.0, None)]:
            self.assertEqual(store.report_range(obs['segment'], t0, t1), coaching._report_range(obs, t0, t1), (t0, t1))

    def test_the_role_tier_and_the_report_are_made_from_the_stored_obs(self):
        k = self.keys()
        opts = {'team': 'A', 't0': None, 't1': None}
        self.assertIsNone(store.ensure_report(self.run, k, opts, self.v['settings']), 'no obs: the measurement comes first')
        self.stored()
        self.assertEqual(store.tier_states(self.run, k), {'measurement': 'ready', 'role': 'missing', 'interpretation': 'missing'})
        seen = []
        rec = store.ensure_report(self.run, k, opts, self.v['settings'], meta={'run': 'run1'}, corrections=['c1'],
                                  after=lambda r, o, ro: seen.append((r['keys'], o['fingerprint'], ro['fingerprint'])))
        self.assertEqual(store.tier_states(self.run, k), {'measurement': 'ready', 'role': 'ready', 'interpretation': 'ready'})
        self.assertEqual(rec['keys'], {'measurement': k['mfp'], 'roles': k['rfp'], 'interpretation': k['ifp']})
        self.assertEqual((rec['options'], rec['corrections'], rec['inputs']), (opts, ['c1'], k['inputs']))
        self.assertEqual(seen, [(rec['keys'], k['mfp'], k['rfp'])])
        self.assertEqual(rec['report']['fingerprints'], {'measurement': k['mfp'], 'roles': k['rfp'], 'interpretation': k['ifp']})
        again = store.ensure_report(self.run, k, opts, self.v['settings'], after=lambda *a: seen.append('again'))
        self.assertEqual((again['saved'], len(seen)), (rec['saved'], 1), 'a stored report is read, never rebuilt')
        roles_file = store.tier_path(self.run, 'roles', k)
        roles_bytes = roles_file.read_bytes()
        k2 = self.keys(with_block(self.v['settings'], 'A', team_block(7, 'Adult')))
        real_roles = store.build_role_tier
        with mock.patch.object(store, 'build_role_tier', side_effect=real_roles) as made:
            store.ensure_report(self.run, k2, opts, with_block(self.v['settings'], 'A', team_block(7, 'Adult')))
        self.assertEqual((made.call_count, roles_file.read_bytes()), (0, roles_bytes), 'the role tier is reused for a level change')
        with self.assertRaises(store.StoreError):
            store.build_interpretation(self.v['obs']['A'], store.load_tier(roles_file), self.v['settings'], dict(k, ifp='0'*64))
        wrong = self.keys(team='A', sha='c'*64)
        store.save_tier(self.run, 'obs', wrong, self.v['obs']['A'])
        with self.assertRaises(store.StoreError):
            store.ensure_report(self.run, wrong, opts, self.v['settings'])
        self.assertFalse(store.tier_path(self.run, 'roles', wrong).exists(), 'refused before any tier is built on it')

    def test_measure_merges_the_moments_pixel_anchors_and_passes_the_coach_spans(self):
        obs = self.v['obs']['A']
        self.assertEqual(set(obs['roleInputs']), {'context', 'detect', 'evidence'}, 'evidence.moments\' anchors merged (EVIDENCE request)')
        seen = {}
        from soccer import coaching
        real = coaching._stage('situations')

        def situations(cd, possession, team, measurement, coach_spans=None):
            seen[team] = coach_spans
            return real(cd, possession, team, measurement, coach_spans=coach_spans)
        spans = dict(store.empty_spans(), spans=[{'t0': 600.0, 't1': 630.0, 'state': 'A'}])
        stages = store.measurement_stages(spans)
        self.assertIn('situations', stages)
        self.assertNotIn('situations', store.measurement_stages(None), 'without spans the package\'s own stage runs')
        from soccer.coaching import measuring
        with mock.patch.object(measuring, 'stage', lambda name: situations if name == 'situations' else coaching._stage(name)):
            out = store.measure(json.loads(self.v['data']), result_sha=self.v['sha'], shape_settings=self.v['shape'],
                                settings=self.v['settings'], spans=spans, team='A')
        self.assertEqual(seen, {'A': spans})
        self.assertEqual(out['fingerprint'], self.keys(spans=spans)['mfp'])
        self.assertIsNotNone(out['timeline']['phaseQuality']['coach'])

    def test_tier_files_are_kept_per_kind_and_team(self):
        k = self.stored()
        self.assertEqual(store.load_tier(store.tier_path(self.run, 'obs', k)), self.v['obs']['A'])
        a, b = store.load_tier(store.tier_path(self.run, 'obs', k)), store.load_tier(store.tier_path(self.run, 'obs', k))
        self.assertIsNot(a, b, 'a fresh object for every reader')
        for n in range(store.KEEP_TIERS['obs']+2):
            store.save_tier(self.run, 'obs', dict(k, mfp=f'{n:x}'*64), {'n': n})
        names = sorted(p.name for p in store.tier_dir(self.run).glob('obs-A-*.json'))
        self.assertEqual(len(names), store.KEEP_TIERS['obs'])
        store.save_tier(self.run, 'obs', dict(k, team='B'), {'n': 'B'})
        self.assertEqual(len(list(store.tier_dir(self.run).glob('obs-*.json'))), store.KEEP_TIERS['obs']+1, 'kept per team')
        store.save_report(self.run, 'A', store.fingerprint_inputs('a'*64, {}, OPTIONS), fixture_report(synthetic_result(seconds=6)))
        self.assertEqual(len(list(store.tier_dir(self.run).glob('obs-A-*.json'))), store.KEEP_TIERS['obs'],
                         'v1 pruning never reaches the v2 folder')
        (store.tier_dir(self.run)/'obs-A-broken.json').write_text('{', encoding='utf-8')
        self.assertIsNone(store.load_tier(store.tier_dir(self.run)/'obs-A-broken.json'))


@unittest.skipUnless(hasattr(sys, 'monitoring'), 'needs sys.monitoring (Python 3.12)')
class TierCode(unittest.TestCase):
    """What each cache tier's fingerprint hashes is the code that tier runs (critic S-12 / CODE M5): every soccer/ file
    whose functions run while the measurement and role tiers are built is hashed by that tier (before fix round 1,
    store.py ran in both and was hashed by neither: an edit to its measurement plumbing served old obs files as fresh),
    and the text-only modules (words, drills, the v1 reference) never run there, so they are not followed (a wording
    edit made every run's measurement stale)."""

    @classmethod
    def setUpClass(cls):
        import importlib
        import pkgutil
        from soccer import coaching
        for m in pkgutil.iter_modules(coaching.__path__):      # every module imported first: only calls are seen
            importlib.import_module('soccer.coaching.'+m.name)
        v = v2_scene()
        mon = sys.monitoring
        tool = next((t for t in (4, 3) if mon.get_tool(t) is None), None)
        if tool is None:
            raise unittest.SkipTest('no free sys.monitoring tool id')
        soccer_dir = Path(coaching.__file__).resolve().parent.parent
        cls.ran = {'measurement': set(), 'role': set()}
        tier = [None]

        def started(code, offset):
            try:
                rel = Path(code.co_filename).resolve().relative_to(soccer_dir).as_posix()
            except ValueError:
                return mon.DISABLE
            if tier[0]:
                cls.ran[tier[0]].add(rel)
            return mon.DISABLE
        settings = with_block(v['settings'], 'A', team_block(4, formation={'intended': '4-4-2', 'periods': []},
                                                             roles={'A-02': [{'role': 'LB', 'from': None, 'to': None}]}))
        spans = dict(store.empty_spans(), spans=[{'t0': 600.0, 't1': 630.0, 'state': 'A'}])
        mon.use_tool_id(tool, 'test_coaching_store.TierCode')
        try:
            mon.register_callback(tool, mon.events.PY_START, started)
            for name, run in (('measurement', lambda: store.measure(json.loads(v['data']), result_sha=v['sha'],
                                                                     shape_settings=v['shape'], settings=settings,
                                                                     spans=spans)),
                              ('role', lambda: [store.build_role_tier(cls.obs[t], settings, t) for t in 'AB'])):
                tier[0] = name
                mon.restart_events()
                mon.set_events(tool, mon.events.PY_START)
                try:
                    out = run()
                finally:
                    mon.set_events(tool, 0)
                    tier[0] = None
                if name == 'measurement':
                    cls.obs = out
        finally:
            mon.register_callback(tool, mon.events.PY_START, None)
            mon.free_tool_id(tool)

    def test_every_file_a_tier_runs_is_hashed_and_no_text_only_module_runs(self):
        from soccer import coaching
        for tier in ('measurement', 'role'):
            hashed = set(coaching.code_files(tier))
            self.assertIn('coaching/contextual.py', self.ran[tier], 'the trace saw the tier run')
            self.assertEqual(self.ran[tier]-hashed, set(), f'{tier}: run but not in its fingerprint')
            self.assertEqual(self.ran[tier] & coaching.TEXT_ONLY, set(), f'{tier}: a text-only module ran')
        self.assertIn('coaching/measuring.py', self.ran['measurement'])

    def test_text_only_modules_are_hashed_by_the_interpretation_only(self):
        from soccer import coaching
        for tier in ('measurement', 'role'):
            files = set(coaching.code_files(tier))
            self.assertEqual(files & coaching.TEXT_ONLY, set(), tier)
            self.assertNotIn('coaching/store.py', files, 'the cache itself is in no fingerprint')
        self.assertLessEqual(set(coaching.TEXT_ONLY), set(coaching.code_files('interpretation')))
        for tier in ('measurement', 'role'):
            self.assertIn('coaching/measuring.py', coaching.code_files(tier))
        # A text-only module is reached from the two tiers only through the v1 detectors module: a new import of one
        # from a measurement or role module must be decided (made a root of that tier, or the call removed).
        for tier in ('measurement', 'role'):
            for rel in coaching.code_files(tier):
                edges = coaching._imports(rel) & coaching.TEXT_ONLY
                if rel != 'coaching/detectors.py':
                    self.assertEqual(edges, set(), f'{tier}: {rel} imports {sorted(edges)}')


class ReportRecordsV2(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.v = v2_scene()

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.run = Path(self.tmp.name)/'video'/'run1'
        self.opts = {'team': 'A', 't0': None, 't1': None}

    def tearDown(self):
        self.tmp.cleanup()

    def built(self, settings, opts=None, t0=None, t1=None):
        v = self.v
        k = store.tier_keys(v['sha'], v['shape'], settings, 'A', None, v['segment'], t0, t1)
        store.save_tier(self.run, 'obs', k, v['obs']['A'])
        return k, store.ensure_report(self.run, k, opts or self.opts, settings, meta={'run': 'run1'})

    def test_fresh_stale_by_options_and_the_reasons(self):
        s4 = with_block(self.v['settings'], 'A', team_block(4))
        k4, r4 = self.built(s4)
        self.assertEqual(store.find_report_v2(self.run, k4, self.opts), (r4, True))
        s5 = with_block(self.v['settings'], 'A', team_block(5, 'U16'))
        k5 = store.tier_keys(self.v['sha'], self.v['shape'], s5, 'A', None, self.v['segment'])
        found, fresh = store.find_report_v2(self.run, k5, self.opts)
        self.assertEqual((found['keys'], fresh), (r4['keys'], False), 'the newest report asked with the same options')
        self.assertEqual(store.find_report_v2(self.run, k5, {'team': 'A', 't0': 605.0, 't1': 650.0}), (None, False))
        self.assertEqual([c['code'] for c in store.staleness_v2(r4['inputs'], k5['inputs'])], ['profile'])
        spans = dict(store.empty_spans(), spans=[{'t0': 600.0, 't1': 610.0, 'state': 'A'}])
        nine = with_block(self.v['settings'], 'A', team_block(4, format={'players': 9, 'goalkeeper': True}))
        cases = [(store.tier_keys('b'*64, self.v['shape'], s4, 'A', None, self.v['segment']), ['result']),
                 (store.tier_keys(self.v['sha'], self.v['shape'], s4, 'A', spans, self.v['segment']), ['coach_spans']),
                 (store.tier_keys(self.v['sha'], self.v['shape'], nine, 'A', None, self.v['segment']), ['format']),
                 (store.tier_keys(self.v['sha'], {'direction': {'A': 'left', 'switchAt': None}, 'lineup': {'A': None, 'B': None},
                                                  'formations': []}, s4, 'A', None, self.v['segment']), ['settings']),
                 (store.tier_keys(self.v['sha'], self.v['shape'],
                                  with_block(self.v['settings'], 'A', team_block(4, roles={'A-02': [{'role': 'LB', 'from': None, 'to': None}]})),
                                  'A', None, self.v['segment']), ['roles'])]
        for keys, codes in cases:
            self.assertEqual([c['code'] for c in store.staleness_v2(r4['inputs'], keys['inputs'])], codes)
        changed = json.loads(json.dumps(k4['inputs']))
        changed['interpretation']['code'] = '0'*64
        self.assertEqual([c['code'] for c in store.staleness_v2(r4['inputs'], changed)], ['engine'])
        text = store.staleness_v2(r4['inputs'], cases[0][0]['inputs'], [], ['c1', 'c2'])[0]['text']
        self.assertIn('2 corrections made since this report', text)
        self.assertEqual([c['code'] for c in store.staleness_v2(r4['inputs'], k4['inputs'])], ['options'])

    def test_a_stale_report_is_the_closest_stored_one_not_the_newest(self):
        # CODE M2 through the page on BC60: level 4, level 2, level 4 again (a cache hit writes nothing), then only the
        # format changed: the newest file (level 2) was served under the level-4 setup with a 'profile' reason.
        s4 = with_block(self.v['settings'], 'A', team_block(4))
        k4, r4 = self.built(s4)
        time.sleep(.02)
        k2, r2 = self.built(with_block(self.v['settings'], 'A', team_block(2, 'U10')))
        self.assertEqual(store.ensure_report(self.run, k4, self.opts, s4)['saved'], r4['saved'], 'a cache hit')
        self.assertEqual(store.records_v2(self.run, 'A')[0]['keys'], r2['keys'], 'the level-2 file is the newest')
        nine4 = store.tier_keys(self.v['sha'], self.v['shape'], with_block(s4, 'A', team_block(4, format={'players': 9, 'goalkeeper': True})),
                                'A', None, self.v['segment'])
        found, fresh = store.find_report_v2(self.run, nine4, self.opts)
        self.assertEqual((found['keys'], fresh), (r4['keys'], False))
        self.assertEqual([c['code'] for c in store.staleness_v2(found['inputs'], nine4['inputs'])], ['format'])
        nine2 = store.tier_keys(self.v['sha'], self.v['shape'],
                                with_block(s4, 'A', team_block(2, 'U10', format={'players': 9, 'goalkeeper': True})), 'A', None, self.v['segment'])
        self.assertEqual(store.find_report_v2(self.run, nine2, self.opts)[0]['keys'], r2['keys'])
        far = store.tier_keys(self.v['sha'], self.v['shape'], with_block(s4, 'A', team_block(6, 'Adult')), 'A', None, self.v['segment'])
        self.assertEqual(store.find_report_v2(self.run, far, self.opts)[0]['keys'], r2['keys'], 'equally close: the newest')

    def test_a_roles_file_with_another_fingerprint_under_the_key_is_rebuilt(self):
        # CODE M4 survivor ensure_report_ignores_rfp: the roles file named by the key (16 hex of the rfp) must also
        # carry the key's full fingerprints, else it is built again, never used for another role setting.
        s4 = with_block(self.v['settings'], 'A', team_block(4))
        k4, r4 = self.built(s4)
        roles_path = store.tier_path(self.run, 'roles', k4)
        forged = dict(store.load_tier(roles_path), fingerprint='f'*64)
        store.write_json(roles_path, forged)
        store.tier_path(self.run, 'report', k4).unlink()
        again = store.ensure_report(self.run, k4, self.opts, s4, meta={'run': 'run1'})
        self.assertEqual(again['report']['fingerprints']['roles'], k4['rfp'])
        self.assertEqual(store.load_tier(roles_path)['fingerprint'], k4['rfp'], 'the roles file was built again')

    def test_records_newest_first_and_served_as_a_copy(self):
        k4, r4 = self.built(with_block(self.v['settings'], 'A', team_block(4)))
        time.sleep(.02)
        k6, r6 = self.built(with_block(self.v['settings'], 'A', team_block(6, 'Adult')))
        self.assertEqual([r['keys']['interpretation'] for r in store.records_v2(self.run, 'A')], [k6['ifp'], k4['ifp']])
        self.assertEqual(store.records_v2(self.run, 'B'), [])
        section = {'squad': 'sq-0123456789ab', 'skills': [], 'note': None}
        served = store.served_report(r4, progression=section, team_name='U14 Blue')
        self.assertEqual((served['generated'], served['progression'], served['teamName']), (r4['saved'], section, 'U14 Blue'))
        self.assertEqual(S.validate('report', served), [])
        self.assertIsNone(r4['report']['progression'], 'the stored record is not changed')
        self.assertIsNone(store.load_record_v2(store.tier_path(self.run, 'report', k4))['report']['generated'])
        self.assertEqual(store.evidence_times(r4), {e['frame']: e['time'] for e in r4['report']['keyMoments']})
        self.assertEqual(store.pdf_filename_v2(r4['report'], 'My Match.mp4', 'run1', '0'*16), 'pitchiq-coaching-My-Match-run1-A-level4.pdf')
        self.assertEqual(store.pdf_filename_v2(r4['report'], None, 'run1', 'abcdef0123456789'), 'pitchiq-coaching-abcdef0123456789-run1-A-level4.pdf')
        part = dict(r4['report'], range={'t0': 605.25, 't1': 650.5, 'seconds': 45.25, 'whole': False})
        self.assertEqual(store.pdf_filename_v2(part, 'm.mp4', 'run1'), 'pitchiq-coaching-m-run1-A-level4-605-651s.pdf',
                         'a custom range is in the name (critic R4)')
        from soccer.coaching import pdf
        for rep in (r4['report'], part):
            self.assertEqual(store.pdf_filename_v2(rep, rep['video']['name'] or 'x.mp4', rep['run']),
                             pdf.pdf_filename(rep['video']['name'] or 'x.mp4', rep['run'], 'A', None, None, 4,
                                              *((605.25, 650.5) if rep is part else (None, None))), "the PDF module's spelling")
        picks = {p['drill'] for p in r4['report']['drills']}
        self.assertEqual(set(store.drills_of(r4['report'])), picks)
        self.assertTrue(all(d['drill_id'] == k for k, d in store.drills_of(r4['report']).items()))


class FeedbackV2(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.v = v2_scene()
        cls.tmp = tempfile.TemporaryDirectory()
        run = Path(cls.tmp.name)/'video'/'run1'
        # The first profile whose report shows a finding with key moments (which one does depends on the rules: the
        # fix round changes verdicts, and this class is about where feedback attaches, not about a verdict).
        for level, age in ((4, 'U14'), (5, 'U16'), (3, 'U12'), (6, 'Adult'), (7, 'Adult'), (2, 'U10'), (1, 'U8')):
            settings = with_block(cls.v['settings'], 'A', team_block(level, age))
            k = store.tier_keys(cls.v['sha'], cls.v['shape'], settings, 'A', None, cls.v['segment'])
            store.save_tier(run, 'obs', k, cls.v['obs']['A'])
            cls.record = store.ensure_report(run, k, {'team': 'A', 't0': None, 't1': None}, settings)
            cls.finding = next((f for s in store.FINDING_SECTIONS for f in cls.record['report'][s] if f['timestamps']), None)
            if cls.finding is not None:
                break
        if cls.finding is None:
            raise AssertionError('the short_segment scene shows no finding with key moments at any level')
        # Only a key moment with a suggested area takes an adjusted one (CODE L-6, round 2), and the short_segment
        # scene's moments have none: the record gets one on the finding's first moment, as alternatives.py draws it
        # (a suggestedZone element on the moment's diagram); the stored files are not changed.
        cls.evidence = cls.finding['timestamps'][0]['evidence']
        cls.bare = json.loads(json.dumps(cls.record))
        cls.record = json.loads(json.dumps(cls.record))
        moment = next(e for e in cls.record['report']['keyMoments'] if e['id'] == cls.evidence)
        zone = {'type': 'suggestedZone', 'polygon': [[.40, .30], [.48, .30], [.48, .40], [.40, .40]], 'kind': 'zone',
                'alternative': 'alt-0123456789ab', 'mover': (moment['players'] or ['A-01'])[0], 'label': 'possible covering area'}
        moment['diagram'] = dict(moment['diagram'] or {}, elements=list((moment['diagram'] or {}).get('elements') or [])+[zone])
        cls.rng = {'t0': cls.record['report']['range']['t0'], 't1': cls.record['report']['range']['t1']}

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.video = Path(self.dir.name)

    def tearDown(self):
        self.dir.cleanup()

    def area(self, **change):
        return dict({'evidence': self.evidence, 'polygon': [[.4, .3], [.5, .3], [.5, .4]], 'mover': None,
                     'note': None}, **change)

    def test_verdicts_by_key_and_range_and_one_per_target(self):
        key = self.finding['key']
        e = store.add_verdict_v2(self.video, 'run1', self.record, key, 'correct', '  fine  ', now=datetime(2026, 10, 22, 9, 0, 0))
        self.assertRegex(e['id'], store.FEEDBACK_ID)
        self.assertEqual((e['findingKey'], e['range'], e['team'], e['text'], e['level'], e['engine']), (key, self.rng, 'A', 'fine', self.record['report']['profile']['level'], 2))
        self.assertEqual(e['finding']['detector'], self.finding['observation'])
        again = store.add_verdict_v2(self.video, 'run1', self.record, key, 'partial')
        other_run = store.add_verdict_v2(self.video, 'run2', self.record, key, 'incorrect')
        doc = store.read_feedback_doc(self.video)
        self.assertEqual(([x['id'] for x in doc['feedback']], doc['version']), ([again['id'], other_run['id']], store.FEEDBACK_VERSION))
        with self.assertRaises(KeyError):
            store.add_verdict_v2(self.video, 'run1', self.record, 'D1|A|team|1.0|2.0', 'correct')
        with self.assertRaises(ValueError):
            store.add_verdict_v2(self.video, 'run1', self.record, key, 'agree')
        with self.assertRaises(ValueError):
            store.add_verdict_v2(self.video, 'run1', self.record, key, 'correct', 'x'*(store.MAX_NOTE+1))

    def test_attached_only_to_a_shown_finding_of_this_run_team_and_range(self):
        key = self.finding['key']
        mine = store.add_verdict_v2(self.video, 'run1', self.record, key, 'correct')
        theirs = store.add_verdict_v2(self.video, 'run0', self.record, key, 'correct')
        doc = store.read_feedback_doc(self.video)
        other_range = dict(json.loads(json.dumps(doc['feedback'][0])), id='f20261022-090001-abcdef', range={'t0': 605.0, 't1': 650.0})
        other_team = dict(json.loads(json.dumps(doc['feedback'][0])), id='f20261022-090002-abcdef', team='B')
        withheld = dict(json.loads(json.dumps(doc['feedback'][0])), id='f20261022-090003-abcdef', findingKey='D2|A|team|600.0|601.0')
        doc['feedback'] += [other_range, other_team, withheld, {'id': 'x', 'from': 'a later version'}]
        store.write_json(store.feedback_path(self.video), doc)
        out = {e['id']: (e['attached'], e['thisRun']) for e in store.feedback_answer(self.record['report'], store.read_feedback_doc(self.video), 'run1')}
        self.assertEqual(out, {mine['id']: (key, True), theirs['id']: (None, False), other_range['id']: (None, True),
                               other_team['id']: (None, True), withheld['id']: (None, True)})
        self.assertEqual(store.feedback_answer(None, store.read_feedback_doc(self.video), 'run1')[0]['attached'], None)

    def test_adjusted_areas(self):
        key = self.finding['key']
        a = store.add_adjustment(self.video, 'run1', self.record, key, self.area(note='Hold   the\nline'), now=datetime(2026, 10, 22, 9, 0, 0))
        adj = a['adjustment']
        self.assertRegex(adj['id'], store.ADJUSTMENT_ID)
        self.assertEqual((adj['note'], adj['created'], adj['polygon']), ('Hold the line', '2026-10-22T09:00:00', [[.4, .3], [.5, .3], [.5, .4]]))
        self.assertEqual(S.validate('adjustment', adj), [])
        moved = store.add_adjustment(self.video, 'run1', self.record, key, self.area(polygon=[[.41, .3], [.5, .3], [.5, .4]]))
        self.assertEqual(moved['adjustment']['id'], adj['id'], 'one area per finding, moment, run, team and range')
        self.assertEqual(len(store.read_feedback_doc(self.video)['adjustments']), 1)
        for bad in (self.area(evidence='ev-000000000000'), self.area(polygon=[[.4, .3], [.5, .3]]), self.area(polygon=[[.4, .3], [1.3, .3], [.5, .5]]),
                    self.area(polygon=[[.4, .3], [True, .3], [.5, .5]]), self.area(mover='B-?'), self.area(note='caf\u00e9'), self.area(note=5), [1],
                    # CODE L4: no area (collinear points) and edges that cross (a bow tie) were stored with 200
                    self.area(polygon=[[.4, .3], [.5, .3], [.6, .3]]), self.area(polygon=[[.4, .3], [.5, .4], [.5, .3], [.4, .4]]),
                    self.area(polygon=[[.4, .3], [.6, .3], [.6, .5], [.5, .3], [.4, .5]])):
            with self.assertRaises(ValueError, msg=bad):
                store.add_adjustment(self.video, 'run1', self.record, key, bad)
        square = store.add_adjustment(self.video, 'run1', self.record, key, self.area(polygon=[[.4, .3], [.5, .3], [.5, .4], [.4, .4]]))
        self.assertEqual(len(square['adjustment']['polygon']), 4, 'a simple quadrilateral is stored')
        entry = store.feedback_answer(self.record['report'], store.read_feedback_doc(self.video), 'run1')[0]
        self.assertEqual((entry['id'], entry['verdict'], entry['attached'], entry['adjustment']['id']), (adj['id'], None, key, adj['id']))
        # A v1 verdict written afterwards keeps the area (the file is read strictly and both lists written back).
        report = fixture_report(synthetic_result(seconds=6))
        store.add_feedback(self.video, 'run1', {'fingerprint': 'f'*64, 'report': report}, report['improvements'][0]['observation'], 'correct')
        doc = store.read_feedback_doc(self.video)
        self.assertEqual((len(doc['feedback']), len(doc['adjustments'])), (1, 1))
        self.assertTrue(store.is_removable_id(adj['id']) and not store.is_removable_id('adj-12'))
        self.assertTrue(store.remove_feedback(self.video, adj['id']))
        self.assertFalse(store.remove_feedback(self.video, adj['id']))
        self.assertEqual(store.read_feedback_doc(self.video)['adjustments'], [])

    def test_only_a_moment_with_a_suggested_area_takes_an_adjusted_one(self):
        # CODE L-6 (round 2): an area was stored on a 'moments worth a look' review, which by design has none (the
        # route accepted any key moment of the finding).
        self.assertTrue(store.has_suggested_area(self.record['report'], self.evidence))
        self.assertFalse(store.has_suggested_area(self.bare['report'], self.evidence), 'the scene draws none there')
        with self.assertRaisesRegex(ValueError, 'no suggested area'):
            store.add_adjustment(self.video, 'run1', self.bare, self.finding['key'], self.area())
        self.assertFalse(store.feedback_path(self.video).exists(), 'nothing written')
        rep = json.loads(json.dumps(self.bare['report']))       # a listed alternative on the moment also counts
        moment = next(e for e in rep['keyMoments'] if e['id'] == self.evidence)
        rep['alternatives'] = rep['alternatives'] + [{'id': 'alt-0123456789ab'}]
        moment['alternative'] = 'alt-0123456789ab'
        self.assertTrue(store.has_suggested_area(rep, self.evidence))
        moment['alternative'] = 'alt-ffffffffffff'
        self.assertFalse(store.has_suggested_area(rep, self.evidence), 'an alternative the report does not list')
        self.assertFalse(store.has_suggested_area(rep, 'ev-000000000000'))

    def test_an_unreadable_file_is_never_overwritten(self):
        path = store.feedback_path(self.video)
        path.write_text('{"version": 2, "feedback": [], "adjustments": {}}', encoding='utf-8')
        self.assertEqual(store.read_feedback_doc(self.video)['adjustments'], [])
        for write in (lambda: store.add_verdict_v2(self.video, 'run1', self.record, self.finding['key'], 'correct'),
                      lambda: store.add_adjustment(self.video, 'run1', self.record, self.finding['key'], self.area())):
            with self.assertRaises(store.FeedbackUnreadable):
                write()
        self.assertEqual(path.read_text(encoding='utf-8'), '{"version": 2, "feedback": [], "adjustments": {}}')


class RolesAndProgressV2(unittest.TestCase):
    def test_inferred_roles_name_the_longest_role_and_its_most_trusted_source(self):
        def rec(pid, t0, t1, role, source, confidence='medium'):
            return {'id': pid, 't0': t0, 't1': t1, 'role': role, 'source': source, 'confidence': confidence}
        book = {'rolebook': {'records': [rec('A-02', 0, 30, 'LB', 'window', 'low'), rec('A-02', 30, 40, 'LB', 'stable', 'high'),
                                         rec('A-02', 40, 50, 'LWB', 'stable', 'high'), rec('A-05', 0, 10, None, 'none', None),
                                         rec('A-07', 0, 20, 'CB', 'carried'), rec('A-07', 20, 25, 'RB', 'coach', 'high'), 'junk']}}
        self.assertEqual(store.inferred_roles(book), [
            {'id': 'A-02', 'role': 'LB', 'source': 'stable', 'confidence': 'high'},
            {'id': 'A-05', 'role': None, 'source': 'none', 'confidence': None},
            {'id': 'A-07', 'role': 'CB', 'source': 'carried', 'confidence': 'medium'}])
        self.assertEqual(store.inferred_roles(None), [])

    def test_the_squad_of_a_team_in_this_run(self):
        sq = {'id': 'sq-0123456789ab', 'name': 'Blue'}
        settings = {'version': 1, 'teams': {'A': team_block(squad=sq, runLinks=[{'run': 'r1', 'team': 'A', 'confirmed': False},
                                                                                {'run': 'r2', 'team': 'B', 'confirmed': True}]),
                                            'B': None}, 'updated': None}
        link, block = store.progress_link(settings, 'r1', 'A')
        self.assertEqual((link['squad'], link['confirmed'], block['squad']), (sq, False, sq), 'unconfirmed: the section asks')
        self.assertEqual(store.progress_link(settings, 'r2', 'A'), (None, None), 'in r2 the squad is team B')
        link, block = store.progress_link(settings, 'r2', 'B')
        self.assertEqual((link['confirmed'], block['squad']), (True, sq), "read with the squad's own block")
        self.assertEqual(store.progress_link(settings, 'r3', 'B'), (None, None))
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(store.record_progress(tmp, settings, 'r1', '0'*16, 'A', {}, {}, {'t0': None, 't1': None}), [],
                             'an unconfirmed link writes nothing')
            self.assertEqual(store.mark_progress_stale(tmp, {'version': 1, 'teams': {'A': None, 'B': None}}, 'r1', '0'*16, 'A', 'a'*64), 0)
            self.assertFalse(Path(tmp, '_coaching').exists(), 'no squad: the progress store is not touched')

    def test_an_entry_the_rules_cannot_read_is_rederived_from_its_stored_obs(self):
        from soccer.coaching import progress as PG
        v = v2_scene()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sq, _ = PG.create_squad(root, 'Blue', 'U14', 4, {'players': 11, 'goalkeeper': True})
            settings = {'version': 1, 'teams': {'A': team_block(squad={'id': sq['id'], 'name': 'Blue'},
                                                                runLinks=[{'run': 'run1', 'team': 'A', 'confirmed': True}]), 'B': None},
                        'updated': None}
            run_dir = root/'video0000000000'/'run1'
            k = store.tier_keys(v['sha'], v['shape'], settings, 'A', None, v['segment'])
            store.save_tier(run_dir, 'obs', k, v['obs']['A'])
            rec = store.ensure_report(run_dir, k, {'team': 'A', 't0': None, 't1': None}, settings)
            roles = store.load_tier(store.tier_path(run_dir, 'roles', k))
            rows = store.record_progress(root, settings, 'run1', 'video0000000000', 'A', v['obs']['A'], roles, k['range'],
                                         coach_date='2026-10-18')
            self.assertEqual([r['written'] for r in rows], [True])
            self.assertEqual(store.record_progress_if_missing(root, settings, 'run1', 'video0000000000', run_dir, k, rec['report']), [],
                             'the entry exists: nothing written again')
            entry = PG.read_entries(root, sq['id'])[0]
            broken = json.loads(json.dumps(entry))
            opp = next(o for s in broken['skills'].values() for o in s['opportunities'])
            opp['rel'].pop(next(x for x in opp['rel'] if x.startswith('o')))
            PG.upsert(root, broken)
            self.assertEqual(PG.needs_rederive(PG.read_entries(root, sq['id'])[0]), 'rules')
            self.assertEqual(store.rederive_progress(root, sq['id']), 1)
            self.assertIsNone(PG.needs_rederive(PG.read_entries(root, sq['id'])[0]))
            PG.mark_stale(root, video='video0000000000', run='run1', team='A', measurement='f'*16)
            self.assertEqual(store.rederive_progress(root, sq['id']), 0, 'a stale entry waits for its run\'s next report')


class CatalogueV2(unittest.TestCase):
    def test_the_profiles_catalogue_is_made_once_per_version(self):
        from soccer.coaching import profiles
        text = store.profiles_catalog_text()
        self.assertEqual(json.loads(text), json.loads(store.json_text(profiles.catalog())))
        with mock.patch.object(profiles, 'catalog', side_effect=AssertionError('not again')):
            self.assertIs(store.profiles_catalog_text(), text)

    def test_a_table_swapped_in_under_another_path_is_read_again(self):
        # The text was remembered by the stat of the package's own profiles.json, so a table that profiles.load reads
        # from another path (profiles.PROFILES_PATH) was answered from the old text.
        from soccer.coaching import profiles
        store.profiles_catalog_text()
        data = json.loads(profiles.PROFILES_PATH.read_text(encoding='utf-8'))
        data['parameters']['values']['coverAngleDeg']['value'] = 70.0
        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp)/'profiles.json'
            bad.write_text(json.dumps(data), encoding='utf-8')
            with mock.patch.object(profiles, 'PROFILES_PATH', bad):
                with self.assertRaises(profiles.ProfileError) as caught:
                    store.profiles_catalog_text()
        failure = store.table_failure(caught.exception)
        self.assertEqual(failure['table'], 'profiles.json')
        self.assertEqual(failure['problems'], ['parameters.coverAngleDeg: 70.0 acts on nothing beyond the measurement bound (max 60.0)'])


class FixRound3Store(unittest.TestCase):
    """Verification round 2's CODE findings on the store: M-1 (which errors are a coaching table that does not validate)
    and M-2 (whose possession a run's coach spans name, for the PDF)."""

    def test_table_failures_are_told_apart_from_other_errors(self):
        from soccer.coaching import profiles, progress
        cases = ((profiles.ProfileError('sources.json invalid: sources[3].url: not https; sources[4].id: repeated'),
                  {'table': 'sources.json', 'problems': ['sources[3].url: not https', 'sources[4].id: repeated']}),
                 (progress.ProgressError('progress_rules.json invalid: interval.confidence: above 1'),
                  {'table': 'progress_rules.json', 'problems': ['interval.confidence: above 1']}),
                 (ValueError('drill library invalid: l4_cover: no source'), {'table': 'drills.json', 'problems': ['l4_cover: no source']}),
                 (progress.ProgressError('a squad name is printable ASCII, 1..60 characters'), None),
                 (S.SchemaError('report', ['x: missing']), None), (ValueError('team must be A or B'), None),
                 (KeyError('profiles.json invalid: x'), None), (RuntimeError('drill library invalid: x'), None))
        for error, want in cases:
            self.assertEqual(store.table_failure(error), want, repr(error))

    def test_whose_possession_the_spans_name(self):
        def spans(*states):
            return dict(store.empty_spans(), spans=[{'t0': k, 't1': k+1.0, 'state': s} for k, s in enumerate(states)])
        for rec, want in ((None, None), (store.empty_spans(), None), (spans('dead'), None), (spans('A', 'dead'), 'A'),
                          (spans('B'), 'B'), (spans('A', 'B', 'dead'), 'both')):
            self.assertEqual(store.marked_teams(rec), want, rec)


if __name__ == '__main__':
    unittest.main()
