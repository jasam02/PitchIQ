"""The coaching report's storage (soccer/coaching/store.py): the fingerprint keys a report on everything it is computed
from and on nothing else (the coach's feedback is not in it), a stored report goes stale with the reasons a coach can
act on, the feedback file is never wiped by a failed read, writes survive a Windows reader holding the old file, and a
screenshot is the exact source frame.

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
import types
import unittest
from datetime import datetime
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import soccer.coaching  # noqa: E402
from soccer.coaching import episodes, store  # noqa: E402
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
        self.assertEqual(self.stored()['version'], 1)
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


if __name__ == '__main__':
    unittest.main()
