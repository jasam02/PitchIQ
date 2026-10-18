"""The evaluation harness (tools/): the hold-out test rebuilds a keyframe from the other anchors only,
and the comparison of a run with itself is perfect agreement; tools/diagnose.py's exports; and the review page's
'Registered pitch (final)' overlay geometry, run in node against soccer/field.py (skipped without node)."""
import contextlib
import importlib.util
import io
import json
import math
import os
import re
import shutil
import subprocess
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import broadcast  # noqa: E402
from soccer import field  # noqa: E402

TOOLS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'tools')
PAGE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'index.html')
TACTICAL = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'tactical.js')
NODE = shutil.which('node')


def load_tool(name):
    spec = importlib.util.spec_from_file_location(f'tools_{name}', os.path.join(TOOLS, f'{name}.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


holdout = load_tool('holdout')
compare = load_tool('compare')
diagnose = load_tool('diagnose')
counts = load_tool('counts')
W, H = broadcast.W, broadcast.H
SIZE = field.PitchSize()
NAMES = ['centre-spot', 'centre-circle-far', 'centre-circle-near', 'halfway-far', 'halfway-near', 'right-penalty-box-front-far', 'corner-far-right']


def keyframe(P, time, index):
    marks = field.landmarks(SIZE)
    pts = field.project(P, [marks[n] for n in NAMES])
    return {'time': time, 'index': index, 'rms': 0.0, 'observability': .8,
            'points': [{'name': n, 'x': float(x/W), 'y': float(y/H)} for n, (x, y) in zip(NAMES, pts)]}


def record(n=30, bad_paint=False):
    P = field.orient(broadcast.camera(), W, H)
    frames = [{'t': i/25, 'dt': 0.0 if i == 0 else .04, 'M': np.eye(3).ravel().tolist(), 'reliable': True, 'cut': False, 'quality': 'flow'} for i in range(n)]
    anchors = [{'index': 0, 'time': 0.0, 'H': P.ravel().tolist(), 'observability': .8, 'source': 'landmarks', 'root': 0, 'details': {}},
               {'index': 20, 'time': .8, 'H': P.ravel().tolist(), 'observability': .8, 'source': 'landmarks', 'root': 20, 'details': {}}]
    if bad_paint:
        wrong = P.copy()
        wrong[0, 2] += 120   # a paint refinement rooted at keyframe 0 that shifted the view by 120 px
        anchors.append({'index': 10, 'time': .4, 'H': wrong.ravel().tolist(), 'observability': .8, 'source': 'paint', 'root': 0, 'details': {'score': .3}})
    return {'version': 1, 'width': W, 'height': H, 'fps': 25, 'offset': 0, 'size': SIZE.to_json(), 'frames': frames, 'anchors': anchors,
            'keyframes': [keyframe(P, 0.0, 0), keyframe(P, .8, 20)]}


class HoldoutTests(unittest.TestCase):
    def test_each_keyframe_is_rebuilt_from_the_other_one_alone(self):
        report = holdout.evaluate(record())
        self.assertEqual(len(report['keyframes']), 2)
        for k in report['keyframes']:
            self.assertEqual(k['status'], 'rebuilt')
            self.assertLess(k['pxMax'], .5)
            self.assertLess(k['metresMax'], .05)
            self.assertEqual(k['anchorsLeftOut'], 1)
            self.assertEqual(k['rebuiltFrom']['source'], 'carried')
        self.assertEqual(report['panStability']['maxPx'], 0.0)

    def test_a_derived_anchor_is_left_out_with_its_root(self):
        report = holdout.evaluate(record(bad_paint=True))
        first = report['keyframes'][0]
        self.assertEqual(first['anchorsLeftOut'], 2)       # the keyframe and the paint anchor rooted at it
        self.assertLess(first['pxMax'], .5)                 # rebuilt from keyframe 20 only: exact
        second = report['keyframes'][1]
        self.assertEqual(second['anchorsLeftOut'], 1)
        self.assertGreater(second['pxMax'], 50)             # the wrong paint anchor is part of the evidence for keyframe 20


class CompareTests(unittest.TestCase):
    def result(self):
        people = [{'track': 1, 'box': [.1, .2, .02, .06], 'state': 'confirmed', 'id': 'A-01', 'team': 'A', 'pitch': [.3, .4]},
                  {'track': 2, 'box': [.5, .5, .02, .06], 'state': 'uncertain', 'id': None, 'team': 'B', 'pitch': [.6, .7]},
                  {'track': 3, 'box': [.7, .5, .02, .06], 'state': 'candidate', 'id': None, 'team': None}]
        frames = [{'time': i/25, 'people': people, 'calibration': {'confidence': .8, 'disagreement': .5}} for i in range(5)]
        return {'video': {'width': 1920, 'height': 1080}, 'config': {'start': 0, 'seconds': .2}, 'frames': frames,
                'match': {'players': [{'id': 'A-01', 'team': 'A'}], 'goalkeepers': []},
                'summary': {'identifiedShare': .5, 'identity': {'idSwitches': 0}, 'pitch': {'meanConfidence': .8, 'registration': {'anchors': []}}, 'ball': {}},
                'tactical': {'pitchSize': {'length': 105, 'width': 68}}}

    def test_metrics_and_self_agreement(self):
        r = self.result()
        m = compare.metrics(r, 'a')
        self.assertEqual(m['perFrame'], {'tracked': 2.0, 'candidates': 1.0, 'drawnIdentified': 1.0, 'anonymousWithPosition': 1.0, 'trackedUndrawn': 1.0,
                                         'kitUndecided': 0.0, 'kitUndecidedWithPosition': 0.0})
        # No stored homography: nothing was checked against the final pitch, which is not the same as nothing inside it.
        self.assertEqual(m['fieldFilter'], {'rejectedPerFrame': 0.0, 'insideCheckedFrames': 0, 'rejectedInsideFinalPitch': None,
                                            'fieldFilterInsideFinalPitch': None, 'longOutsideInsideFinalPitch': None})
        self.assertEqual(m['identifiedByHeight']['60-90px'], .5)
        agree = compare.agreement(r, r)
        self.assertEqual(agree['boxAgreement'], 1.0)
        self.assertEqual(agree['labelAgreementExact'], 1.0)
        self.assertEqual(agree['labelAgreementMapped'], 1.0)
        self.assertEqual(agree['labelAgreementBothLabelled'], 1.0)
        self.assertEqual((agree['labelsOnlyInA'], agree['labelsOnlyInB']), (0, 0))
        self.assertEqual(agree['positionMeanMetres'], 0.0)
        self.assertIn('| perFrame.tracked | 2 |', compare.table([m]))


class DiagnoseTests(unittest.TestCase):
    """tools/diagnose.py on a small synthetic result whose homography is the broadcast test camera."""

    def setUp(self):
        self.P = field.orient(broadcast.camera(), W, H)

    def foot_box(self, world, height_px=30):
        fx, fy = field.project(self.P, [world])[0]
        h = height_px/H
        return [fx/W-.005, fy/H-h, .01, h]

    def normalized(self, a, b):
        (x0, y0), (x1, y1) = field.project(self.P, [a, b])
        return [x0/W, y0/H, x1/W, y1/H]

    def result(self):
        P = self.P
        tick = {'source': 'model', 'level': 'HIGH', 'confidence': .8, 'edges': {'far': 'line', 'near': 'predicted', 'left': 'off-screen', 'right': 'off-screen'},
                'curvature': .1, 'vanishing': None, 'markings': [self.normalized((20, 0), (85, 0)), self.normalized((52.5, 5), (52.5, 60))]}
        cal = {'confidence': .8, 'temporal': 1.0, 'observability': .8, 'source': 'landmarks', 'direction': None, 'anchor': 0, 'anchorAge': 0.0,
               'disagreement': None, 'H': P.ravel().tolist()}

        def people(track_a01):
            return [{'track': track_a01, 'id': 'A-01', 'display': 'A-01', 'team': 'A', 'state': 'confirmed', 'evidence': 'reidentified' if track_a01 == 5 else 'observed',
                     'box': self.foot_box((52.5, 34)), 'pitch': [.5, .5], 'pitchConfidence': .15, 'pitchSigma': [2.0, 3.0]},
                    {'track': 2, 'id': None, 'state': 'unknown', 'box': self.foot_box((40, 20), 60), 'pitch': [.38, .29], 'pitchConfidence': .7},
                    {'track': 3, 'id': None, 'state': 'confirmed', 'reason': 'long outside the pitch: not saved as a match observation',
                     'box': self.foot_box((30, 30), 60), 'pitch': [.2857, .4412], 'pitchConfidence': .7}]
        rejected = [{'box': self.foot_box((30, 30)), 'reason': 'audience'}, {'box': self.foot_box((-10, 30)), 'reason': 'outside-pitch'}]
        bx, by = field.project(P, [(60, 40)])[0]
        ball_box = [bx/W-.004, by/H-.004, .008, .008]
        ball = {'state': 'TRACKED', 'phase': 'PATH', 'origin': 'OBSERVATION', 'motion': 'MOVING', 'center': [bx/W, by/H], 'box': ball_box,
                'visual': .9, 'trajectory': 1.0, 'identity': .9, 'nearTrack': 1, 'firstPass': {'state': 'TRACKED', 'phase': 'LOCKED', 'confidence': .8, 'track': 1}}
        candidates = [{'index': 1, 'box': ball_box, 'det': .8, 'score': .9, 'visual': .9, 'status': 'ball', 'scores': {'attach': 0.0}, 'source': 0},
                      {'index': 2, 'box': [.1, .9, .008, .008], 'det': .5, 'score': .6, 'visual': .6, 'status': 'rejected', 'reason': 'OTHER_OBJECT',
                       'scores': {'attach': .4, 'otherObject': True}, 'player': {'track': 1, 'lower': True}, 'source': 2}]
        frames = []
        for t, track in ((0.0, 1), (.2, 1), (.4, 1), (1.2, 5), (1.4, 5)):
            f = {'time': t, 'people': people(track), 'rejected': [dict(r) for r in rejected], 'ball': dict(ball), 'ballCandidates': candidates,
                 'calibrated': True, 'calibration': dict(cal), 'mapping': {'imageTracks': 2, 'mapped': 2}, 'pitch': tick if t == 0.0 else None,
                 'pitchModel': {'level': 'HIGH', 'size': {'length': 105.0, 'width': 68.0, 'source': 'standard'}}}
            frames.append(f)
        frames[1]['calibration']['H'] = (-P).ravel().tolist()            # stored with the orientation lost
        frames[2]['calibration'].update(confidence=.1, source='carried', direction='forward', anchorAge=.4)
        events = [{'time': 1.3, 'kind': 'reid', 'track': 5, 'playerId': 'A-01', 'message': 'RE-ID EVENT'},
                  {'time': 0.0, 'kind': 'new-identity', 'track': 1, 'playerId': 'A-01', 'message': 'Track 1 becomes new identity'}]
        return {'video': {'name': 'synthetic.mp4', 'width': W, 'height': H}, 'frames': frames, 'events': events,
                'tactical': {'pitchSize': {'length': 105.0, 'width': 68.0, 'source': 'standard'}}, 'summary': {}}

    RECORD = {'offset': 0, 'anchors': [{'index': 0, 'time': 0.0, 'source': 'landmarks'}]}

    def test_calibration_export(self):
        rows = diagnose.calibration_rows(self.result(), self.RECORD)
        self.assertEqual([r['misoriented'] for r in rows], [0, 1, 0, 0, 0])
        self.assertEqual(rows[0]['paintMatched'], 2)
        self.assertEqual(rows[0]['paintNames'], 'far touchline;halfway line')
        self.assertLess(rows[0]['paintErrorPx'], 1.0)
        self.assertEqual(rows[0]['paintSides'], 'far')
        self.assertIsNone(rows[1].get('paintMatched'))                  # paint is checked on ticks only
        self.assertEqual((rows[2]['anchorSource'], rows[2]['anchorTime'], rows[2]['tickAge']), ('landmarks', 0.0, .4))
        self.assertEqual(rows[2]['edgeFar'], 'line')
        # Without the anchor list the anchor time follows from the age and the direction, and the stored index (a log
        # index that counts the pre-roll) is not reported as the anchor's segment frame.
        bare = diagnose.calibration_rows(self.result(), None)
        self.assertEqual(bare[2]['anchorTime'], 0.0)
        self.assertIsNone(bare[2].get('anchorFrame'))

    def test_without_registration_json_the_anchor_comes_from_the_summary(self):
        """A run that started after a 15 s pre-roll (450 logged frames): the frames store the log index 450; the summary's
        anchor list knows it as segment frame 0 at 0.0 s."""
        r = self.result()
        r['video']['fps'] = 25
        for f in r['frames']:
            f['calibration']['anchor'] = 450
        r['frames'][1]['calibration'].update(direction='forward', anchorAge=.2)
        r['summary'] = {'pitch': {'registration': {'anchors': [{'frame': 0, 'time': 0.0, 'source': 'landmarks'}, {'frame': 30, 'time': 1.2, 'source': 'paint'}]}}}
        rows = diagnose.calibration_rows(r, None)
        self.assertEqual([(x['anchorFrame'], x['anchorTime'], x['anchorSource']) for x in rows[:3]], [(0, 0.0, 'landmarks')]*3)
        self.assertEqual(diagnose.calibration_rows(r, self.RECORD | {'offset': 450})[2]['anchorFrame'], 0)
        # A blended frame looks both ways; an age that matches no listed anchor leaves the frame blank.
        r['frames'][3]['calibration'].update(direction='both', anchorAge=1.2)
        r['frames'][4]['calibration'].update(direction='forward', anchorAge=.7)
        rows = diagnose.calibration_rows(r, None)
        self.assertEqual((rows[3]['anchorFrame'], rows[3]['anchorSource']), (0, 'landmarks'))
        self.assertEqual((rows[4].get('anchorFrame'), rows[4].get('anchorSource'), rows[4]['anchorTime']), (None, None, .7))

    def test_counts_export(self):
        rows, reasons = diagnose.counts_rows(self.result())
        self.assertEqual(set(reasons), {'audience', 'outside-pitch'})
        first, weak = rows[0], rows[2]
        self.assertEqual((first['confirmed'], first['unknown'], first['mapped'], first['mappedLive'], first['unknownMapped']), (2, 1, 3, 2, 1))
        self.assertEqual((first['rejectedInside'], first['rejectedInsideFieldFilter'], first['rejectedInsideReasons']), (1, 1, 'audience:1'))
        self.assertEqual((first['longOutside'], first['longOutsideInside']), (1, 1))
        self.assertEqual((first['lowConfidence_20_40'], first['positioned_20_40'], first['positioned_ge40'], first['lowConfidenceSupportable']), (1, 1, 1, 1))
        self.assertEqual(rows[1]['rejectedInside'], 1)                  # the mis-oriented stored H is re-oriented first
        # Confidence .1: not checked against the final pitch (None, not 0) and no registration to support a position.
        self.assertEqual((first['insideChecked'], weak['insideChecked']), (1, 0))
        self.assertEqual((weak['rejectedInside'], weak['rejectedInsideFieldFilter'], weak['longOutsideInside'], weak['lowConfidenceSupportable']), (None, None, None, 0))
        summary = diagnose.summarize(self.result(), diagnose.calibration_rows(self.result()), rows, diagnose.identity_fragments(self.result()))
        self.assertEqual((summary['rejected']['insideCheckedFrames'], summary['rejected']['insideFinalPitch'], summary['longOutside']['insideFinalPitch']), (4, 4, 4))
        # The supportable count is imputation's small-scene rule: a registration of .3 positions people but does not support them.
        r = self.result()
        r['frames'][0]['calibration']['confidence'] = .3
        self.assertEqual(diagnose.counts_rows(r)[0][0]['lowConfidenceSupportable'], 0)
        # A run without a confident frame reports no inside counts at all.
        for f in r['frames']:
            f['calibration']['confidence'] = .3
        rows, _ = diagnose.counts_rows(r)
        summary = diagnose.summarize(r, diagnose.calibration_rows(r), rows, [])
        self.assertEqual((summary['rejected']['insideCheckedFrames'], summary['rejected']['insideFinalPitch'], summary['longOutside']['insideFinalPitch']), (0, None, None))

    def test_ball_window(self):
        lines = diagnose.ball_window_lines(self.result(), [(0.0, 0.1)])
        self.assertTrue(lines[0].startswith('== 0.00-0.10 s (1 frames)'))
        final = [l for l in lines if '<= FINAL' in l]
        self.assertEqual(len(final), 1)
        self.assertIn('#1', final[0])
        other = next(l for l in lines if 'OTHER_OBJECT' in l)
        self.assertIn('player T1 (A-01) lower', other)
        self.assertIn('source=2', other)
        self.assertIn('other yes', other)

    def test_only_the_candidate_the_final_ball_sits_on_is_marked(self):
        """Two candidates of one ball overlap (a tracklet's detection and a tile's): one mark, on the 'ball' candidate even
        when the other is nearer; without one, on the nearest."""
        r = self.result()
        frame = r['frames'][0]
        box = frame['ball']['box']
        shifted = [box[0]+.3*box[2], box[1], box[2], box[3]]            # .3 diameters off the final ball
        frame['ballCandidates'] = [{'index': 1, 'box': shifted, 'det': .8, 'status': 'ball', 'scores': {}},
                                   {'index': 2, 'box': list(box), 'det': .6, 'status': 'candidate', 'scores': {}},
                                   {'index': 3, 'box': [.1, .9, .008, .008], 'det': .5, 'status': 'rejected', 'scores': {}}]
        marked = lambda: [l.split()[0] for l in diagnose.ball_window_lines(r, [(0.0, 0.1)]) if '<= FINAL' in l]
        self.assertEqual(marked(), ['#1'])
        frame['ballCandidates'][0]['status'] = 'candidate'
        self.assertEqual(marked(), ['#2'])
        frame['ballCandidates'] = frame['ballCandidates'][2:]           # nothing near the final ball: no mark
        self.assertEqual(marked(), [])

    def test_identity_fragments(self):
        rows = diagnose.identity_fragments(self.result())
        self.assertEqual([(r['id'], r['segment'], r['track'], r['frames']) for r in rows], [('A-01', 1, 1, 3), ('A-01', 2, 5, 2)])
        self.assertEqual(rows[0]['gapEvents'], 'new-identity@0.00(T1)')
        self.assertEqual((rows[1]['gapBefore'], rows[1]['bridge'], rows[1]['firstEvidence']), (.8, 'reid', 'reidentified'))
        self.assertEqual(rows[1]['gapEvents'], 'reid@1.30(T5)')
        # A held team change (identity.py's 'team-hold', small scenes) is an identity event: listed in the gap, a bridge
        # after a team change, and counted in the summary.
        r = self.result()
        r['events'].append({'time': 1.25, 'kind': 'team-hold', 'track': 5, 'playerId': 'A-01', 'message': 'TEAM CHANGE IGNORED: small crops (drift)'})
        rows = diagnose.identity_fragments(r)
        self.assertEqual(rows[1]['gapEvents'], 'team-hold@1.25(T5);reid@1.30(T5)')
        self.assertEqual(rows[1]['bridge'], 'reid')
        r['events'] = [e for e in r['events'] if e['kind'] != 'reid']
        rows = diagnose.identity_fragments(r)
        self.assertEqual(rows[1]['bridge'], 'team-hold')
        summary = diagnose.summarize(r, diagnose.calibration_rows(r), diagnose.counts_rows(r)[0], rows)
        self.assertEqual(summary['identities']['events'], {'new-identity': 1, 'team-hold': 1})

    def test_compare_and_counts_report_undecided_kits_and_filtered_people(self):
        r = self.result()
        m = compare.metrics(r, 'synthetic')
        self.assertEqual((m['perFrame']['kitUndecided'], m['perFrame']['kitUndecidedWithPosition']), (1.0, 1.0))
        # Four of the five frames are confident enough to check (the third has confidence .1): the totals are over those.
        self.assertEqual(m['fieldFilter'], {'rejectedPerFrame': 2.0, 'insideCheckedFrames': 4, 'rejectedInsideFinalPitch': 4, 'fieldFilterInsideFinalPitch': 4,
                                            'longOutsideInsideFinalPitch': 4})
        report = counts.evaluate(r, {'synthetic': {'0.2': {'count': 4, 'uncertainty': 0}, '0.4': {'count': 4, 'uncertainty': 0}}}, 'synthetic')
        row, unchecked = report['rows']
        self.assertEqual((row['tracked'], row['missing'], row['kitUndecided'], row['rejectedInside']), (3, 1, 1, 1))
        self.assertEqual(row['rejected'], {'audience': 1, 'outside-pitch': 1})
        self.assertIsNone(unchecked['rejectedInside'])
        table = counts.table(report)
        self.assertIn('| 1 | audience 1, outside-pitch 1 | 1 (audience:1) | - |', table)
        self.assertIn('| 1 | audience 1, outside-pitch 1 | - | - |', table)          # the unchecked frame: '-', not 0

    def test_main_writes_outside_the_run_only(self):
        import json
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            run = os.path.join(tmp, 'runs', 'abc', 'run1')
            os.makedirs(run)
            with open(os.path.join(run, 'result.json'), 'w', encoding='utf-8') as handle:
                json.dump(self.result(), handle)
            out = os.path.join(tmp, 'out')
            with contextlib.redirect_stdout(io.StringIO()) as printed:
                self.assertEqual(diagnose.main([run, '--out', out, '--window', '0', '1']), 0)
            self.assertIn('stored H mis-oriented on 1 of 5 frames', printed.getvalue())
            self.assertEqual(sorted(os.listdir(out)), ['ball_window.txt', 'calibration.csv', 'counts.csv', 'identity_fragments.csv', 'summary.json'])
            self.assertEqual(sorted(os.listdir(run)), ['result.json'])
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                diagnose.main([run, '--out', os.path.join(run, 'diag')])


# ---------- the page's 'Registered pitch (final)' overlay, in node ----------
def page_script():
    with open(PAGE, encoding='utf-8') as handle:
        return '\n'.join(re.findall(r'<script>(.*?)</script>', handle.read(), re.S))


def js_function_span(src, name):
    """(start, end) of `function name(...){...}` in the page script (brace matching: the page keeps its braces balanced)."""
    start = src.index(f'function {name}(')
    depth = 0
    for k in range(src.index('{', start), len(src)):
        depth += {'{': 1, '}': -1}.get(src[k], 0)
        if depth == 0:
            return start, k+1
    raise ValueError(name)


def js_tactical():
    with open(TACTICAL, encoding='utf-8') as handle:
        return 'const Tactical={PITCH:' + re.search(r'const PITCH = (\{.*?\});', handle.read(), re.S).group(1) + '};\n'


def run_node(script, payload):
    done = subprocess.run([NODE, '-e', script], input=json.dumps(payload), capture_output=True, text=True, timeout=120)
    if done.returncode:
        raise AssertionError(done.stderr[-3000:])
    return json.loads(done.stdout)


@unittest.skipUnless(NODE, 'needs node to run the page script')
class RegisteredPitchOverlayTests(unittest.TestCase):
    """The overlay against soccer/field.py: the re-orientation and the projection, the paint matching that decides which
    sides are drawn as verified (field.match_segments mirrored), and that 'verified' is the tick's paint under the tick's
    FINAL homography, at most 0.2 s old, not the live model's edge verdict."""

    @classmethod
    def setUpClass(cls):
        cls.src = page_script()

    def cameras(self):
        import cv2
        src = np.float32([[0, 0], [105, 0], [105, 68], [0, 68]])
        dst = np.float32([[-.3*W, 1.4*H], [.52*W, .300*H], [.528*W, .303*H], [1.5*W, 1.6*H]])   # an end camera: the horizon in view
        return field.orient(broadcast.camera(), W, H), field.orient(cv2.getPerspectiveTransform(src, dst).astype(np.float64), W, H)

    @staticmethod
    def paint(P, rng):
        """Paint segments for camera P: on every visible marking a piece 1.5 px off its line, one 10 px off (inside the
        tolerance of 19.2 px at this width) and one 30 px off (beyond it), 60 px pieces turned by 5 and by 12 degrees about
        a point of the line (ends within the tolerance; the angle decides); and random segments."""
        out = []
        for _, a, b in field.lines(SIZE):
            seg = field.image_segment(P, a, b)
            if not field.visible(seg, W, H):
                continue
            o, d = np.array(seg[0]), np.subtract(seg[1], seg[0])
            u = d/max(float(np.hypot(*d)), 1e-9)
            n = np.array([-u[1], u[0]])
            for t0, t1, off in ((.2, .55, 1.5), (.6, .9, 10.0), (.3, .5, 30.0)):
                p, q = o+t0*d+off*n, o+t1*d+off*n
                out.append([p[0]/W, p[1]/H, q[0]/W, q[1]/H])
            for degrees in (5.0, 12.0):
                c, s = math.cos(math.radians(degrees)), math.sin(math.radians(degrees))
                r = np.array([c*u[0]-s*u[1], s*u[0]+c*u[1]])
                m = o+.45*d
                p, q = m-30*r, m+30*r
                out.append([p[0]/W, p[1]/H, q[0]/W, q[1]/H])
        out += rng.uniform(0, 1, (20, 4)).tolist()
        return [[float(v) for v in m] for m in out]

    def test_orientation_projection_and_paint_matching_mirror_field(self):
        src = self.src
        block = src[src.index('/* registered-pitch geometry:'):src.index('/* end of the registered-pitch geometry */')]
        script = js_tactical() + block + """
const input=JSON.parse(require('fs').readFileSync(0,'utf8'));
process.stdout.write(JSON.stringify({pitch:Tactical.PITCH,
  orient:input.matrices.map(h=>regOriented(h,input.W,input.H)),
  project:input.matrices.map(h=>{const o=regOriented(h,input.W,input.H);return input.points.map(([x,y])=>regProject(o,x,y));}),
  match:input.paint.map(c=>regMatch(c.H,input.L,input.Wd,input.W,input.H,c.markings))}));"""
        rng = np.random.default_rng(3)
        cams = self.cameras()
        matrices = [M for P in cams for M in (P, -P)]                   # -P: a stored matrix that lost its sign
        points = [(float(x), float(y)) for x in np.linspace(-30, 135, 12) for y in np.linspace(-40, 108, 9)]
        paint = [{'H': P.ravel().tolist(), 'markings': self.paint(P, rng)} for P in cams]
        out = run_node(script, {'W': W, 'H': H, 'L': SIZE.length, 'Wd': SIZE.width, 'matrices': [M.ravel().tolist() for M in matrices],
                                'points': points, 'paint': paint})
        # The page's pitch markings are the field's (the matching depends on them).
        p = out['pitch']
        self.assertEqual((p['penaltyDepth'], p['penaltyHalf'], p['goalDepth'], p['goalHalf'], p['spot'], p['circle']),
                         (field.PA, field.PA_HALF, field.GA, field.GA_HALF, field.SPOT, field.RADIUS))
        shown = 0
        for M, oriented, projected in zip(matrices, out['orient'], out['project']):
            Mo = field.orient(M, W, H)
            self.assertTrue(np.array_equal(np.asarray(oriented), Mo.ravel()))
            front = field.depth(Mo, points) > 1e-9
            for pt, ok, q in zip(points, front, projected):
                expected = field.project(Mo, [pt])[0] if ok else None
                if expected is not None and np.abs(expected).max() >= 1e5:
                    expected = None
                if expected is None:
                    self.assertIsNone(q, pt)
                else:
                    shown += 1
                    self.assertTrue(np.allclose(q, expected, rtol=1e-9, atol=1e-6), (pt, q, expected))
        self.assertGreater(shown, 100)
        self.assertLess(shown, len(points)*len(matrices))              # the end camera has world points behind it
        for case, js in zip(paint, out['match']):
            P = np.asarray(case['H']).reshape(3, 3)
            expected = field.match_segments(P, SIZE, [{'a': (m[0], m[1]), 'b': (m[2], m[3])} for m in case['markings']], W, H)
            self.assertEqual([m['name'] for m in js], [m['name'] for m in expected])
            self.assertTrue(np.allclose([m['error'] for m in js], [m['error'] for m in expected], atol=1e-6))
            self.assertIn('far touchline', {m['name'] for m in js})
            self.assertGreater(len(js), 6)

    def test_a_side_is_verified_by_the_ticks_paint_under_its_final_homography(self):
        """At broadcast 52 s the final H, carried 8.8 s, put the far touchline on the boards while the live model's tick said
        'line': the overlay drew it solid. Now a side is solid only where the tick's paint lies on it under the final H."""
        src = self.src
        start = src.index("/* Debug overlay 'Registered pitch (final)'")
        _, end = js_function_span(src, 'registeredPitch')
        helpers = [src[slice(*js_function_span(src, name))] for name in ('segment', 'calibrationAnchor')]
        stamp = src.index('const timestamp=')
        script = js_tactical() + """
let result=null,pitchIndex=[],info={video:{fps:25}};const COLORS={REJECTED:'#ff0000',PITCH:'#00ff00'},canvas={width:0,height:0},strokes=[];
const ctx={_dash:[],lineWidth:1,strokeStyle:'',fillStyle:'',beginPath(){},moveTo(){},lineTo(){},arc(){},fill(){},setLineDash(d){this._dash=d;},
  stroke(){strokes.push([this.lineWidth,this._dash.length]);}};
""" + src[stamp:src.index('\n', stamp)] + '\n' + '\n'.join(helpers) + '\n' + src[start:end] + """
const input=JSON.parse(require('fs').readFileSync(0,'utf8'));result=input.result;canvas.width=result.video.width;canvas.height=result.video.height;
let last=-1;pitchIndex=result.frames.map((f,i)=>{if(f.pitch)last=i;return last;});
process.stdout.write(JSON.stringify(result.frames.map((f,i)=>{strokes.length=0;const lines=registeredPitch(f,i).map(l=>l[0]);
  return {sides:lines.find(l=>l.includes(' · sides ')),verifiedStrokes:strokes.filter(s=>s[0]===3&&s[1]===0).length};})));"""
        P = field.orient(broadcast.camera(), W, H)
        drifted = np.array([[1, 0, 0], [0, 1, 60], [0, 0, 1]], np.float64) @ P      # the image 60 px off: beyond the 19.2 px tolerance
        markings = []
        for a, b in (((20, 0), (85, 0)), ((52.5, 5), (52.5, 60))):                   # far touchline and halfway line paint, under P
            (x0, y0), (x1, y1) = field.project(P, [a, b])
            markings.append([x0/W, y0/H, x1/W, y1/H])
        tick = {'source': 'model', 'level': 'HIGH', 'confidence': .8, 'markings': markings,
                'edges': {'far': 'line', 'near': 'predicted', 'left': 'off-screen', 'right': 'off-screen'}}
        frames = [{'time': t, 'calibration': {'H': M.ravel().tolist(), 'confidence': .8, 'source': 'carried'}, 'pitch': tick if has_tick else None}
                  for t, M, has_tick in ((0.0, P, True), (.04, P, False), (.2, drifted, True), (.24, drifted, False), (.7, P, False))]
        out = run_node(script, {'result': {'video': {'width': W, 'height': H, 'fps': 25}, 'frames': frames,
                                           'tactical': {'pitchSize': {'length': 105.0, 'width': 68.0, 'source': 'standard'}}, 'summary': {}}})
        self.assertIn('far verified', out[0]['sides'])
        self.assertGreaterEqual(out[0]['verifiedStrokes'], 1)
        self.assertIn('tick 0.04 s ago', out[1]['sides'])
        self.assertIn('far verified', out[1]['sides'])
        for k in (2, 3):                     # the live model's edge says 'line', the final H is off the paint: predicted, dashed
            self.assertIn('far predicted', out[k]['sides'])
            self.assertEqual(out[k]['verifiedStrokes'], 0)
        self.assertIn('not checked (the last paint tick is 0.5 s old)', out[4]['sides'])
        self.assertEqual(out[4]['verifiedStrokes'], 0)


if __name__ == '__main__':
    unittest.main()
