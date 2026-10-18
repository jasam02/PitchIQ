"""The evaluation harness (tools/): the hold-out test rebuilds a keyframe from the other anchors only,
and the comparison of a run with itself is perfect agreement; tools/diagnose.py's exports; the review page's
'Registered pitch (final)' overlay geometry, run in node against soccer/field.py (skipped without node); and the
2D view's measurement tools: tools/jitter.py (against the page in node), tools/trajectory_bench.py and
tools/check_foot_truth.py."""
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
import zlib

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



# ---------- tools/jitter.py, tools/trajectory_bench.py and tools/check_foot_truth.py: the 2D view's jitter and accuracy ----------
jitter = load_tool('jitter')
trajectory_bench = load_tool('trajectory_bench')
check_foot_truth = load_tool('check_foot_truth')
GIT = shutil.which('git')
BEFORE_STABILIZER = '88be58f'   # the commit whose tactical.js jitter.py's RAW stream reproduces
JITTER_IDS = ('A-01', 'A-02', 'A-03', 'B-01', 'B-02', 'B-03')


def jitter_result(n=90, fps=30.0, alternate=.3, noise=0.0, step_at=None, seed=0, estimates=False):
    """Six identities and one '?' person on a 30 fps grid: A-01 runs along the length at 3 m/s, the others stand. The raw
    positions (the frames' and the timeline's O samples) are the truth moved across by +-alternate metres on
    alternate frames, plus Gaussian noise; the stabilized block is the truth itself, with one rejected observation.
    step_at: from this frame on every raw position is 0.5 m further along the length (a registration step at an anchor
    that the summary lists). estimates: the timeline and the block carry estimates and holes too: A-02 is predicted
    (frozen at its last observation, sigma 1 m) on frames 30-74 in the timeline and unknown there in the block; A-03 is
    predicted with a sigma of 9 m (hidden) on frames 40-49 and interpolated on 50-54 in both; B-01 is unknown on frame 20
    in the block (a straight line across a missing frame); the '?' block bridges frames 10-14 (kept by the page: its
    observations 0.2 s apart) and 40-60 (0.73 s: left out)."""
    rng = np.random.default_rng(seed)
    times = np.arange(n)/fps
    truth = {pid: np.stack([np.full(n, 30.0+8*k), np.full(n, 10.0+9*k)], 1) for k, pid in enumerate(JITTER_IDS)}
    truth['A-01'][:, 0] = 20+3*times
    anon_truth = np.stack([np.full(n, 50.0), np.full(n, 30.0)], 1)
    sign = np.where(np.arange(n) % 2, -1.0, 1.0)

    def raw(path):
        out = path+np.stack([np.zeros(n), alternate*sign], 1)+rng.normal(0, noise, (n, 2))
        if step_at is not None:
            out[step_at:, 0] += .5
        return out
    raws = {pid: raw(p) for pid, p in truth.items()}
    anon_raw = raw(anon_truth)
    frames = []
    for i, t in enumerate(times):
        people = [{'track': k+1, 'id': pid, 'team': pid[0], 'state': 'confirmed', 'pitch': [raws[pid][i, 0]/105, raws[pid][i, 1]/68]}
                  for k, pid in enumerate(JITTER_IDS)]
        people.append({'track': 99, 'id': None, 'team': 'B', 'state': 'uncertain', 'pitch': [anon_raw[i, 0]/105, anon_raw[i, 1]/68]})
        anchor = step_at is not None and i == step_at
        frames.append({'time': float(t), 'frame': i, 'calibrated': True, 'people': people,
                       'calibration': {'source': 'landmarks' if anchor else 'carried', 'direction': None if anchor else 'forward', 'anchor': 0,
                                       'anchorAge': 0.0 if anchor else round(float(t), 3), 'confidence': .8}})
    column = lambda path, scale: [float(v)/scale for v in path]
    timeline = {'frameTimes': [round(float(t), 3) for t in times],
                'ids': {pid: {'team': pid[0], 'x': column(raws[pid][:, 0], 105), 'y': column(raws[pid][:, 1], 68), 'kind': ['O']*n, 'sigma': [[.4, .4]]*n}
                        for pid in JITTER_IDS}}
    weights = {pid: [1.0]*n for pid in JITTER_IDS}
    weights['A-02'][10] = 0.0
    stabilized = {'version': 1, 'method': 'test', 'frameTimes': list(timeline['frameTimes']),
                  'ids': {pid: {'team': pid[0], 'x': column(truth[pid][:, 0], 105), 'y': column(truth[pid][:, 1], 68), 'kind': ['O']*n,
                                'sigma': [[.1, .1]]*n, 'v': [[0.0, 0.0]]*n, 'w': weights[pid]} for pid in JITTER_IDS},
                  'anonymous': {'99': {'team': 'B', 'x': column(anon_truth[:, 0], 105), 'y': column(anon_truth[:, 1], 68), 'kind': ['O']*n, 'sigma': [[.1, .1]]*n}},
                  'flags': [{'id': 'A-02', 'track': 2, 'frame': 10, 'time': float(times[10]), 'reason': 'rejected: far off the trajectory', 'raw': [0, 0], 'used': None}],
                  'summary': {}, 'noise': {}}
    if estimates:
        tl, st = timeline['ids'], stabilized['ids']
        for i in range(30, 75):
            tl['A-02']['x'][i], tl['A-02']['y'][i], tl['A-02']['kind'][i], tl['A-02']['sigma'][i] = tl['A-02']['x'][29], tl['A-02']['y'][29], 'P', [1.0, 1.0]
            st['A-02']['x'][i], st['A-02']['y'][i], st['A-02']['kind'][i] = None, None, 'U'
        for block in (tl, st):
            for i in range(40, 55):
                block['A-03']['kind'][i] = 'P' if i < 50 else 'I'
                if i < 50:
                    block['A-03']['sigma'][i] = [9.0, 9.0]
        st['B-01']['x'][20], st['B-01']['y'][20], st['B-01']['kind'][20] = None, None, 'U'
        for i in list(range(10, 15))+list(range(40, 61)):
            stabilized['anonymous']['99']['kind'][i] = 'I'
    summary = {'pitch': {'registration': {'anchors': [{'frame': step_at, 'time': round(float(times[step_at]), 3), 'source': 'landmarks'}] if step_at is not None else []}}}
    return {'video': {'width': 1920, 'height': 1080, 'fps': fps}, 'frames': frames, 'timeline': timeline, 'stabilized': stabilized, 'summary': summary,
            'match': {'players': [{'id': pid, 'team': pid[0]} for pid in JITTER_IDS], 'goalkeepers': []}}


def weaving_result(delay=0, average=0, n=420, fps=30.0, noise=.05, seed=3):
    """One player (A-01) zigzagging at 4 m/s (a 90 degree turn every 2 s) for 11 s and then standing: the raw
    observations the path plus Gaussian noise, the stabilized block the path itself, delayed by delay frames or averaged
    over a centred window of average frames."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)/fps
    moving = t < 11
    vx = np.where(moving, 4/math.sqrt(2), 0.0)
    vy = np.where(moving, np.where((t//2) % 2 == 0, 1.0, -1.0)*4/math.sqrt(2), 0.0)
    path = np.stack([20+np.cumsum(vx)/fps, 34+np.cumsum(vy)/fps], 1)
    raw = path+rng.normal(0, noise, path.shape)
    stab = path.copy()
    if delay:
        stab[delay:], stab[:delay] = path[:-delay], path[0]
    if average:
        stab = np.array([path[max(0, i-average//2):i+average//2+1].mean(0) for i in range(n)])
    column = lambda a, scale: [float(v)/scale for v in a]
    frames = [{'time': float(ti), 'frame': i, 'calibrated': True,
               'people': [{'track': 1, 'id': 'A-01', 'team': 'A', 'state': 'confirmed', 'pitch': [raw[i, 0]/105, raw[i, 1]/68]}]} for i, ti in enumerate(t)]
    timeline = {'frameTimes': [float(v) for v in t], 'ids': {'A-01': {'team': 'A', 'x': column(raw[:, 0], 105), 'y': column(raw[:, 1], 68), 'kind': ['O']*n,
                                                                       'sigma': [[.1, .1]]*n}}}
    stabilized = {'version': 1, 'method': 'test', 'frameTimes': list(timeline['frameTimes']), 'anonymous': {}, 'flags': [], 'summary': {}, 'noise': {},
                  'ids': {'A-01': {'team': 'A', 'x': column(stab[:, 0], 105), 'y': column(stab[:, 1], 68), 'kind': ['O']*n, 'sigma': [[.05, .05]]*n, 'w': [1.0]*n}}}
    return {'video': {'width': 1920, 'height': 1080, 'fps': fps}, 'frames': frames, 'timeline': timeline, 'stabilized': stabilized, 'summary': {},
            'match': {'players': [{'id': 'A-01', 'team': 'A'}], 'goalkeepers': []}}


class JitterTests(unittest.TestCase):
    """tools/jitter.py: the port of the committed tactical.js and of the stabilized drawing, the two streams' metrics
    (observations and estimates apart), the responsiveness rows, the correction and coverage, the calibration-instability
    diagnostic and the rebuilt registration, on synthetic results."""

    def test_the_raw_port_smooths_observed_samples_and_snaps_like_the_page(self):
        dt = 1/30
        track = jitter.Track([0, dt, 2*dt, 3*dt, 4*dt], [0.0, 1.0, 2.0, 3.0, 10.0], [0.0]*5, [0, 0, 0, 1, 0], [None, None, None, 2.0, None])
        track.smooth_observed(.14)
        # +-0.07 s holds two neighbours a side at 30 fps; the I sample stops the window and is left as it is.
        self.assertEqual(track.x.tolist(), [1.0, 1.0, 1.0, 3.0, 10.0])
        x, _, drawn, observed = track.sample([2*dt+.2*dt, 2.5*dt, 4*dt+.5, 4*dt+.7], dt, .6, 1.0)
        self.assertEqual(x[0], 1.0)                      # within a quarter frame: the sample itself
        self.assertAlmostEqual(x[1], 2.0)                # halfway to the I sample: a straight line
        self.assertEqual(x[2], 10.0)                     # held 0.5 s after the last sample
        self.assertEqual(drawn.tolist(), [True, True, True, False])
        self.assertEqual(observed.tolist(), [True, False, False, False])   # the observation; a line to an estimate and a hold are not
        x, _, _, _ = track.sample([2*dt+.2*dt], dt, .6, 1.0, same=0.0)
        self.assertAlmostEqual(x[0], 1.4)                # no snap
        hidden = jitter.Track([0, dt], [0.0, 0.0], [0.0, 0.0], [2, 2], [9.0, 9.0])
        self.assertFalse(hidden.sample([0.0], dt, .6, 1.0)[2][0])   # a prediction less certain than uncertainSigma is not drawn

    def test_the_stabilized_sampler_is_sample_stable(self):
        dt = 1/30
        track = jitter.Track([0, dt, 2*dt, 5*dt, 6*dt], [0.0, 1.0, 2.0, 5.0, 6.0], [0.0]*5, [0, 0, 1, 0, 2], [np.nan, np.nan, np.nan, .5, 9.0], [0, 1, 2, 5, 6])
        x, _, drawn, observed = track.sample_stable([-.3*dt, -.2*dt, .5*dt, 1.5*dt, 3.5*dt, 5.5*dt], dt, .6, 1.0)
        self.assertEqual(drawn.tolist(), [False, True, True, True, True, True])   # the first sample a quarter frame early, not more
        self.assertTrue(np.allclose(x[1:], [0.0, .5, 1.5, 3.5, 5.5]))            # straight lines at every time, no snap
        # Observed between two observations of consecutive frames only: not towards the I sample, not across frames 2-5.
        self.assertEqual(observed.tolist(), [False, True, True, False, False, False])
        # Between an O (sigma 0.5 m) and a P (9 m) the sigma is the straight line (4.75 m: drawn); sampleAt's larger one hides it.
        self.assertFalse(track.sample([5.5*dt], dt, .6, 1.0, same=0.0)[2][0])
        # After a gap longer than the gap limit the next sample shows up a quarter frame early, and the hold ends at hold.
        far = jitter.Track([0.0, 2.0], [0.0, 1.0], [0.0, 0.0], [0, 0], f=[0, 60])
        x, _, drawn, _ = far.sample_stable([.5, 2.0-.2*dt, 2.0-.3*dt], dt, .6, 1.0)
        self.assertEqual(drawn.tolist(), [True, True, False])
        self.assertEqual(x[:2].tolist(), [0.0, 1.0])
        # Without frame indices two samples are of consecutive frames when at most 1.5 frame steps apart.
        loose = jitter.Track([0, dt, 3*dt], [0.0, 1.0, 3.0], [0.0]*3, [0, 0, 0])
        self.assertEqual(loose.sample_stable([.5*dt, 2*dt], dt, .6, 1.0)[3].tolist(), [True, False])

    def test_stable_tracks_keep_what_the_page_keeps(self):
        r = jitter_result(estimates=True)
        times = np.array([f['time'] for f in r['frames']])
        ident, anon = jitter._stable_tracks(r['stabilized'], times, jitter.final_roster(r))
        self.assertEqual(len(ident['A-02'].t), 45)                     # its U frames are holes
        self.assertEqual(anon['99'].f.tolist(), [i for i in range(90) if not 40 <= i <= 60])   # the 0.73 s bridge is left out, the 0.2 s one kept
        r['stabilized']['anonymous']['99']['team'] = None
        self.assertEqual(jitter._stable_tracks(r['stabilized'], times, jitter.final_roster(r))[1], {})   # a '?' needs a team

    def test_streams_correction_and_coverage(self):
        report = jitter.evaluate(jitter_result(), compute=False)
        self.assertEqual(report['stabilized']['source'], 'result')
        raw, stab = report['streams']['raw'], report['streams']['stabilized']
        self.assertEqual((raw['players']['markers'], raw['anonymous']['markers'], stab['anonymous']['markers']), (6, 1, 1))
        self.assertEqual((raw['players']['estimatedShare'], stab['players']['estimatedShare']), (0.0, 0.0))
        # The stabilized block is a straight run and still players, drawn by straight lines: no residual, no reversal.
        self.assertLess(stab['players']['display']['observed']['residualRms_cm'], .01)
        self.assertEqual(stab['players']['display']['observed']['reversals10PerMin'], 0.0)
        self.assertEqual(stab['players']['frames']['observed']['stationaryStep_cm']['p50'], 0.0)
        # The raw stream alternates +-0.3 m across: the page's 5-tap mean leaves +-6 cm on the frames, reversing every frame.
        self.assertGreater(raw['players']['display']['observed']['residualRms_cm'], 3.0)
        self.assertGreater(raw['players']['display']['observed']['reversals10PerMin'], 100)
        self.assertGreater(raw['anonymous']['display']['observed']['residualRms_cm'], 3.0)
        self.assertLess(stab['anonymous']['display']['observed']['residualRms_cm'], .01)
        self.assertEqual(raw['players']['display']['all'], raw['players']['display']['observed'])   # nothing estimated
        for kind in ('players', 'anonymous'):
            self.assertAlmostEqual(report['correction']['stabilized'][kind]['median_m'], .3, places=3)
            self.assertAlmostEqual(report['correction']['stabilized'][kind]['max_m'], .3, places=3)
        cov = report['coverage']['stabilized']
        self.assertEqual(cov['rawObservations'], 6*90)
        self.assertEqual(cov['stabilizedKindAtRawObservations'], {'O': 1.0})
        self.assertEqual(cov['weights'], {'n': 540, 'rejectedShare': round(1/540, 4), 'downWeightedShare': 0.0})
        self.assertEqual(cov['flags'], {'rejected: far off the trajectory': 1})
        text = jitter.table(report)
        self.assertIn('| residual RMS (cm) |', text)
        self.assertIn('| correction median/p90 m (n) | - | 0.3/0.3 (540) |', text)

    def test_estimates_are_measured_apart_from_observations(self):
        report = jitter.evaluate(jitter_result(estimates=True), compute=False)
        raw, stab = report['streams']['raw']['players'], report['streams']['stabilized']['players']
        self.assertGreater(raw['estimatedShare'], .05)
        self.assertGreater(stab['estimatedShare'], 0.0)
        for s in (raw, stab):
            d = s['display']
            self.assertAlmostEqual(d['observed']['minutes']+d['estimated']['minutes'], d['all']['minutes'], places=2)
        # A-02's frozen prediction stands perfectly still: pooled with the observations it would hide their vibration.
        self.assertEqual(raw['frames']['estimated']['stationaryStep_cm']['p50'], 0.0)
        self.assertGreater(raw['frames']['observed']['stationaryStep_cm']['p50'], 5.0)
        self.assertIn("### players, estimates (held, across a missing frame or gap, imputed); estimates' share of the drawn samples: RAW", jitter.table(report))

    def test_responsiveness_finds_a_late_or_a_sluggish_stream(self):
        resp = lambda r: jitter.evaluate(r, compute=False)['responsiveness']
        exact = resp(weaving_result())
        self.assertLess(abs(exact['stabilized']['lag_ms']), 10)
        self.assertLess(abs(exact['raw']['lag_ms']), 10)                   # the page's centred mean has no lag either
        self.assertEqual((exact['stabilized']['turns']['n'], exact['stabilized']['stops']['n']), (5, 1))
        self.assertLess(exact['stabilized']['turns']['median_m'], .1)
        late = resp(weaving_result(delay=3))['stabilized']
        self.assertAlmostEqual(late['lag_ms'], 100.0, delta=10.0)
        self.assertGreater(late['turns']['median_m'], 3*exact['stabilized']['turns']['median_m'])
        # A centred 1 s average has no lag but cuts the corners and slows down early: the jitter rows reward it, these do not.
        sluggish = jitter.evaluate(weaving_result(average=31), compute=False)
        self.assertLess(sluggish['streams']['stabilized']['players']['display']['observed']['residualRms_cm'],
                        jitter.evaluate(weaving_result(), compute=False)['streams']['raw']['players']['display']['observed']['residualRms_cm'])
        slow = sluggish['responsiveness']['stabilized']
        self.assertGreater(slow['turns']['median_m'], 4*exact['stabilized']['turns']['median_m'])
        self.assertGreater(slow['stops']['median_m'], 2*exact['stabilized']['stops']['median_m'])
        self.assertLess(slow['velocityCorrelation'], exact['stabilized']['velocityCorrelation'])
        self.assertIn('| turn rounding median/p90 m (n) |', jitter.table(sluggish))
        # A box-geometry flag of the block near a turn takes it out of the 'no box event' rows, for RAW too; a flag of the
        # trajectory fit does not (a sluggish smoother would hide its own rounding so).
        r = weaving_result()
        r['stabilized']['flags'] = [{'id': 'A-01', 'frame': 60, 'reason': 'box lost the legs: foot repaired from the top and the running height'},
                                    {'id': 'A-01', 'frame': 120, 'reason': 'down-weighted: off the trajectory'}]
        flagged = resp(r)
        for stream in ('raw', 'stabilized'):
            self.assertEqual((flagged[stream]['turns']['n'], flagged[stream]['turnsNoBoxEvent']['n'], flagged[stream]['stopsNoBoxEvent']['n']), (5, 4, 1))
        self.assertEqual(resp(weaving_result())['stabilized']['turnsNoBoxEvent']['n'], 5)
        r = weaving_result()
        del r['stabilized']
        self.assertEqual(resp(r)['raw']['turnsNoBoxEvent']['n'], 0)       # no block, no box events known: not given

    def test_a_registration_step_is_listed_at_its_anchor(self):
        report = jitter.evaluate(jitter_result(alternate=0.0, noise=.02, step_at=45), compute=False)
        cm = report['commonMode']['raw']
        self.assertEqual(cm['together']['pairs'], 1)
        self.assertEqual(cm['together']['atAnchors'], 1)
        self.assertLess(cm['together']['independentExpectation'], .05)
        top = cm['listed'][0]
        self.assertEqual((top['time'], top['players'], top['jumping']), (1.5, 6, 6))
        self.assertGreater(top['jumpMedian_m'], .4)
        self.assertGreater(top['commonStep_m'], .4)
        self.assertEqual(top['anchor'], {'time': 1.5, 'source': 'landmarks', 'frame': 'second'})
        self.assertEqual((top['before']['source'], top['after']['source'], top['after']['anchorAge']), ('carried', 'landmarks', 0.0))
        self.assertGreater(cm['commonShare'], cm['independentExpectation'])
        self.assertEqual(report['commonMode']['stabilized']['together']['pairs'], 0)   # the stabilized block has no step
        self.assertEqual(report['warnings'], [])
        clean = jitter.evaluate(jitter_result(alternate=0.0, noise=.02), compute=False)['commonMode']['raw']
        self.assertEqual(clean['together']['pairs'], 0)

    def test_a_stabilized_stream_that_still_steps_at_anchors_is_flagged(self):
        r = jitter_result(alternate=0.0, noise=.02, step_at=45)
        for pid in JITTER_IDS:      # the block built on homographies that snap: the step is in it too
            r['stabilized']['ids'][pid]['x'] = list(r['timeline']['ids'][pid]['x'])
            r['stabilized']['ids'][pid]['y'] = list(r['timeline']['ids'][pid]['y'])
        report = jitter.evaluate(r, compute=False)
        self.assertEqual(report['commonMode']['stabilized']['together']['atAnchors'], 1)
        self.assertIn('WARNING: STABILIZED still jumps together at 1 listed anchors', jitter.table(report))
        self.assertIn('--rebuild', report['warnings'][0])

    def test_missing_or_mismatched_blocks_measure_raw_only(self):
        r = jitter_result()
        del r['stabilized']
        report = jitter.evaluate(r, compute=False)
        self.assertEqual(list(report['streams']), ['raw'])
        self.assertEqual(report['stabilized']['missing'], 'no stabilized block in the result')
        r = jitter_result()
        r['stabilized']['frameTimes'] = r['stabilized']['frameTimes'][:-1]
        self.assertEqual(jitter.evaluate(r, compute=False)['stabilized']['missing'], 'the stabilized block does not match the frames')

    def test_window_ids_and_main(self):
        import tempfile
        report = jitter.evaluate(jitter_result(), window=(0.0, 1.5), ids=['A-01'], compute=False)
        self.assertEqual(report['streams']['raw']['players']['markers'], 1)
        self.assertNotIn('anonymous', report['streams']['raw'])
        self.assertLessEqual(report['streams']['raw']['players']['display']['all']['minutes'], 1.6/60)
        self.assertEqual(report['coverage']['stabilized']['rawObservations'], 46)
        # '?' alone: the markers only; no identity is measured, covered or corrected.
        anon = jitter.evaluate(jitter_result(), ids=['?'], compute=False)
        self.assertEqual(list(anon['streams']['raw']), ['anonymous'])
        self.assertEqual((anon['coverage'], anon['responsiveness'], anon['correction']['stabilized']['players']), ({}, {}, None))
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'result.json')
            with open(path, 'w', encoding='utf-8') as handle:
                json.dump(jitter_result(), handle)
            out = os.path.join(tmp, 'jitter.json')
            with contextlib.redirect_stdout(io.StringIO()) as printed:
                self.assertEqual(jitter.main([tmp, '--json', out, '--no-compute', '--rebuild']), 0)
            self.assertIn('stabilized: result', printed.getvalue())
            self.assertIn('WARNING: --rebuild: no registration.json', printed.getvalue())
            with open(out, encoding='utf-8') as handle:
                self.assertEqual(json.load(handle)['frames'], 90)
            self.assertEqual(sorted(os.listdir(tmp)), ['jitter.json', 'result.json'])

    def test_the_labels_follow_the_constants(self):
        saved = jitter.ACC_LEVELS
        try:
            jitter.ACC_LEVELS = (8.0, 20.0)
            report = jitter.evaluate(jitter_result(), compute=False)
            self.assertIn('accLagOver8Share', report['streams']['raw']['players']['display']['all'])
            self.assertIn('| accLag share > 8 m/s^2 |', jitter.table(report))
        finally:
            jitter.ACC_LEVELS = saved

    def test_rebuild_remaps_the_run_on_its_registration(self):
        """The registration rebuilt from a run's registration.json (two landmark anchors on a still camera): the people's
        stored positions, made on an older registration 1 m off, come back where their boxes' feet are, the timeline
        with them; the stored result is not touched and nothing is written."""
        import tempfile
        P = field.orient(broadcast.camera(), W, H)
        worlds = {'A-01': (40.0, 30.0), 'B-01': (60.0, 20.0)}
        frames = []
        for i in range(30):
            people = []
            for k, (pid, (x, y)) in enumerate(worlds.items()):
                u, v = field.project(P, [(x+.1*i, y)])[0]
                box = [(u-10)/W, (v-60)/H, 20/W, 60/H]
                people.append({'track': k+1, 'id': pid, 'team': pid[0], 'state': 'confirmed', 'box': box, 'score': .9,
                               'pitch': [(x+.1*i+1.0)/SIZE.length, y/SIZE.width], 'pitchConfidence': .8})
            frames.append({'time': i/25, 'frame': i, 'people': people, 'calibrated': True, 'calibration': {'H': P.ravel().tolist(), 'source': 'carried'}})
        result = {'video': {'width': W, 'height': H, 'fps': 25}, 'frames': frames, 'tactical': {'pitchSize': SIZE.to_json()},
                  'match': {'players': [{'id': pid, 'team': pid[0]} for pid in worlds], 'goalkeepers': []},
                  'roster': {'slots': [{'id': pid, 'team': pid[0], 'role': 'PLAYER', 'confirmed': True} for pid in worlds]}, 'summary': {}}
        stored = json.dumps(result, sort_keys=True)
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(jitter.rebuild(tmp, result), (None, f'no registration.json in {tmp}'))
            with open(os.path.join(tmp, 'registration.json'), 'w', encoding='utf-8') as handle:
                json.dump(record(), handle)
            rebuilt, _ = jitter.rebuild(tmp, result)
            self.assertEqual(sorted(os.listdir(tmp)), ['registration.json'])
        self.assertEqual(json.dumps(result, sort_keys=True), stored)
        for i, f in enumerate(rebuilt['frames']):
            for p in f['people']:
                x, y = worlds[p['id']]
                self.assertLess(abs(p['pitch'][0]*SIZE.length-(x+.1*i)), .02)
                self.assertLess(abs(p['pitch'][1]*SIZE.width-y), .02)
        self.assertEqual(len(rebuilt['timeline']['frameTimes']), 30)
        self.assertLess(abs(rebuilt['timeline']['ids']['A-01']['x'][10]*SIZE.length-41.0), .02)
        report = jitter.evaluate(result, compute=False, rebuilt=rebuilt)
        self.assertIn('registration', report['streams'])
        self.assertTrue(report['rebuilt'])


# The page's own drawing in node: every track of buildTimeline (tl.tracks with sampleAt, or tl.stable with sampleStable)
# at the given times, with the board's visible() rule (full lineup off); null where nothing is drawn.
PAGE_DRAW = r"""
const fs = require('fs'), vm = require('vm');
const input = JSON.parse(fs.readFileSync(0, 'utf8'));
const sandbox = {window: {}, console};
vm.createContext(sandbox);
vm.runInContext(fs.readFileSync(input.tactical, 'utf8'), sandbox, {filename: 'tactical.js'});
const T = sandbox.window.Tactical, R = T.RULES;
const tl = T.buildTimeline(input.result, {video: {fps: input.fps}});
const stable = input.mode === 'stable';
const tracks = stable ? tl.stable : tl.tracks, sampler = stable ? T.sampleStable : T.sampleAt;
const out = {};
if (tracks) for (const [key, track] of tracks) {
  const anonymous = !!track.info.anonymous;
  out[key] = input.times.map((t) => {
    const s = sampler(track, t, tl.frameStep, anonymous ? R.anonymousHold : R.holdAfter, anonymous ? R.anonymousGap : R.interpolateGap);
    return s && (s.kind !== 'P' || (Number.isFinite(s.sigma) ? s.sigma <= R.uncertainSigma : true)) ? [s.x, s.y] : null;
  });
}
process.stdout.write(JSON.stringify(out));
"""


@unittest.skipUnless(NODE, 'needs node to run the page')
class JitterAgainstThePageTests(unittest.TestCase):
    """tools/jitter.py draws what the page draws: RAW as the committed tactical.js before the stabilizer (read from git,
    skipped without it), STABILIZED as the working tree's sampleStable, sample for sample at display and frame times off
    the default phase, on a result with estimates, holes, hidden predictions and a '?' bridge."""

    def drawn(self, tactical, result, T, mode):
        return run_node(PAGE_DRAW, {'tactical': tactical, 'result': result, 'times': [float(t) for t in T], 'fps': result['video']['fps'], 'mode': mode})

    def mismatches(self, ident, anon, page, T, frame_step, stable):
        mine = {**ident, **{f'anon:{k}': v for k, v in anon.items()}}
        self.assertEqual(sorted(mine), sorted(page))
        bad = 0
        for key, track in mine.items():
            hold, gap = (jitter.RULES['anonymousHold'], jitter.RULES['anonymousGap']) if key.startswith('anon:') else (jitter.RULES['holdAfter'], jitter.RULES['interpolateGap'])
            x, y, drawn, _ = jitter._draw(track, T, frame_step, hold, gap, stable)
            shown = np.array([p is not None for p in page[key]])
            px = np.array([p[0] if p else np.nan for p in page[key]])
            py = np.array([p[1] if p else np.nan for p in page[key]])
            bad += int((shown != drawn).sum()+(shown & drawn & ((np.abs(px-x) > 1e-12) | (np.abs(py-y) > 1e-12))).sum())
        return bad

    def grid(self, result):
        times = np.array([f['time'] for f in result['frames']])
        step = 1/result['video']['fps']
        display, _ = jitter.display_times(times[0], times[-1])
        return times, step, np.concatenate([times, times-.24*step, times+.24*step, times-.26*step, display, times[0]+.37/144+np.arange(430)/144])

    def test_raw_is_the_page_before_the_stabilizer(self):
        import tempfile
        if not GIT:
            self.skipTest('needs git')
        done = subprocess.run([GIT, 'show', f'{BEFORE_STABILIZER}:prototypes/gpu-tracker/tactical.js'], capture_output=True, text=True,
                              cwd=os.path.dirname(TACTICAL))
        if done.returncode:
            self.skipTest(f'{BEFORE_STABILIZER} is not in this checkout')
        result = jitter_result(noise=.05, estimates=True)
        times, step, T = self.grid(result)
        with tempfile.TemporaryDirectory() as tmp:
            page = os.path.join(tmp, 'tactical.js')
            with open(page, 'w', encoding='utf-8') as handle:
                handle.write(done.stdout)
            drawn = self.drawn(page, result, T, 'raw')
        roster = jitter.final_roster(result)
        ident, anon = jitter.raw_stream(result, times, roster, jitter.people_samples(result, roster))
        self.assertEqual(self.mismatches(ident, anon, drawn, T, step, False), 0)

    def test_stabilized_is_the_pages_sample_stable(self):
        result = jitter_result(noise=.05, estimates=True)
        times, step, T = self.grid(result)
        ident, anon = jitter._stable_tracks(result['stabilized'], times, jitter.final_roster(result))
        self.assertEqual(self.mismatches(ident, anon, self.drawn(TACTICAL, result, T, 'stable'), T, step, True), 0)


def metrica_csv(path, team, times, positions, ball):
    """A tracking file in the Metrica sample layout (three header rows; period, frame, time, x/y per player normalized,
    the ball): synthetic data, for the bench's loader."""
    P = positions.shape[1]
    pairs = lambda cell: [c for k in range(P) for c in (cell(k), '')]
    with open(path, 'w', encoding='utf-8', newline='') as handle:
        for row in (['']*3+pairs(lambda k: team)+['']*2, ['']*3+pairs(lambda k: str(k+1))+['']*2,
                    ['Period', 'Frame', 'Time [s]']+pairs(lambda k: f'Player{k+1}')+['Ball', '']):
            handle.write(','.join(row)+'\n')
        for i, t in enumerate(times):
            cells = [str(1 if t < times[-1]/2 else 2), str(i+1), f'{t:.2f}']
            for p in range(P):
                cells += [f'{positions[i, p, 0]/105:.5f}', f'{positions[i, p, 1]/68:.5f}']
            cells += [f'{ball[i, 0]/105:.5f}', f'{ball[i, 1]/68:.5f}']
            handle.write(','.join(cells)+'\n')


def metrica_files(tmp, game=9, teleport=False):
    """Two synthetic teams of six for 60 s at 25 Hz in the Metrica layout under tmp (slow weaving players); teleport: the
    first home player jumps 3 m between two frames at 15 s and back at 15.4 s (a truth glitch)."""
    times = np.arange(0, 60, .04)
    rng = np.random.default_rng(4)
    P = 6
    start = np.c_[rng.uniform(20, 85, P), rng.uniform(8, 60, P)]
    phase = rng.uniform(0, 2*np.pi, P)
    teams = []
    for side in range(2):
        teams.append(start[None]+np.stack([6*np.sin(.3*times[:, None]+phase+side), 4*np.cos(.2*times[:, None]+phase)], -1))
    if teleport:
        teams[0][375:385, 0, 0] += 3.0
    ball = np.c_[52.5+20*np.sin(.1*times), np.full(len(times), 34.0)]
    for (side, team), pos in zip((('Home', 'Home'), ('Away', 'Away')), teams):
        metrica_csv(os.path.join(tmp, f'Sample_Game_{game}_RawTrackingData_{side}_Team.csv'), team, times, pos, ball)
    return times, teams


class TrajectoryBenchTests(unittest.TestCase):
    """tools/trajectory_bench.py on synthetic trajectories: the simulated cameras, the noise model and its scenarios, the
    methods and the metrics, the truth's glitches, the seeding, and whole runs on synthetic files in the Metrica layout."""
    dt = 1/30

    def test_the_cameras_are_plausible_views_that_round_trip(self):
        expected = {'pole': (.25, .4), 'broadcast': (.08, .14)}           # m per vertical pixel at the centre spot
        for name in trajectory_bench.CAMERAS:
            H_, R, C, K = trajectory_bench.camera_views(name, np.array([52.5]), self.dt)
            self.assertIsNone(field.sanity(H_[0], SIZE, 1920, 1080), name)   # not mirrored, the far side at the top
            ground = np.array([[[52.5, 34.0, 0.0], [10.0, 5.0, 0.0], [95.0, 60.0, 0.0], [30.0, 50.0, 0.0]]])
            px, depth = trajectory_bench.project(R, C, K, ground)
            self.assertTrue((depth > 0).all())
            back, J = trajectory_bench.back_project(H_, px)
            self.assertLess(np.abs(back-ground[..., :2]).max(), 1e-6)
            vertical = np.linalg.norm(J[0, 0, :, 1])
            self.assertTrue(expected[name][0] < vertical < expected[name][1], (name, vertical))
            self.assertGreater(vertical/np.linalg.norm(J[0, 0, :, 0]), 2.5)   # the depth direction's pixel is worth more

    def standing(self, seconds=120.0):
        T = int(seconds/self.dt)
        grid = np.array([(x, y) for x in np.linspace(15, 90, 6) for y in np.linspace(8, 60, 4)])[:22]
        truth = np.repeat(grid[None], T, axis=0)
        return truth, np.array(['A', 'B']*11), np.full(T, 52.5)

    def generators(self, camera, window=0):
        key = [0, 1, trajectory_bench.camera_key(camera), window]
        return np.random.default_rng(key), lambda kind: np.random.default_rng(key+[zlib.crc32(kind.encode('utf-8'))])

    def test_simulated_noise_follows_the_diagnosis(self):
        truth, teams, ball_x = self.standing()
        sim = trajectory_bench.simulate(truth, teams, ball_x, self.dt, 'pole', np.random.default_rng(1))
        r = sim['realism']
        self.assertTrue(.01 < r['eventShare'] < .07, r)
        self.assertGreater(r['eventGivenLowScore'], 4*r['eventGivenHighScore'])
        own = sim['who'] == np.arange(truth.shape[1])
        camera = np.array(trajectory_bench.CAMERAS['pole']['position'][:2])
        farther = np.linalg.norm(sim['obs']-camera, axis=-1)-np.linalg.norm(truth-camera, axis=-1)
        events = sim['observed'] & sim['event'] & own
        clean = sim['observed'] & ~sim['event'] & own
        self.assertGreater(np.median(farther[events]), 1.0)              # an event throws the player away from the camera
        self.assertLess(abs(np.median(farther[clean])), .05)
        # The core noise lies along the line of sight.
        ray = (truth-camera)/np.linalg.norm(truth-camera, axis=-1, keepdims=True)
        err = (sim['obs']-truth)[clean]
        along = (err*ray[clean]).sum(-1)
        across = err[:, 0]*ray[clean][:, 1]-err[:, 1]*ray[clean][:, 0]
        self.assertGreater(np.std(along), 5*np.std(across))
        boxes = sim['boxes'][sim['observed']]
        self.assertTrue((boxes[:, 2] > 0).all() and (boxes[:, 3] > 0).all())

    def test_the_scenarios_add_their_events_on_the_same_base_draws(self):
        truth, teams, ball_x = self.standing(60.0)
        rng, extra = self.generators('pole')
        base = trajectory_bench.simulate(truth, teams, ball_x, self.dt, 'pole', rng)
        rng, extra = self.generators('pole')
        top = trajectory_bench.simulate(truth, teams, ball_x, self.dt, 'pole', rng, ('top',), extra)
        # A top edge that moves leaves the foot point alone: the same observations, taller boxes in the events.
        np.testing.assert_array_equal(base['observed'], top['observed'])
        np.testing.assert_array_equal(np.nan_to_num(base['obs']), np.nan_to_num(top['obs']))
        self.assertTrue(.01 < top['realism']['extraShare']['top'] < .08, top['realism'])
        grown = top['observed'] & top['extra']
        self.assertTrue((top['boxes'][..., 3][grown] > base['boxes'][..., 3][grown]).all())
        self.assertFalse(base['extra'].any())
        # The other kinds move the foot point; a reach brings the player nearer the camera.
        camera = np.array(trajectory_bench.CAMERAS['pole']['position'][:2])
        for kind, sign in (('reach', -1), ('torso', 1)):
            rng, extra = self.generators('pole')
            sim = trajectory_bench.simulate(truth, teams, ball_x, self.dt, 'pole', rng, (kind,), extra)
            moved = sim['observed'] & sim['extra'] & ~sim['event'] & (sim['who'] == np.arange(22))
            farther = np.linalg.norm(sim['obs']-camera, axis=-1)-np.linalg.norm(truth-camera, axis=-1)
            self.assertGreater(sign*np.median(farther[moved]), .5, kind)

    def test_the_calibration_error_and_the_true_camera_motion(self):
        truth, teams, _ = self.standing(20.0)
        ball_x = 30+3*np.arange(len(truth))*self.dt                       # the camera pans after the ball
        rng, extra = self.generators('pole')
        base = trajectory_bench.simulate(truth, teams, ball_x, self.dt, 'pole', rng)
        # The motion row carries the previous frame's pixels to this frame's (the tracker's convention). An affine fit of
        # a panning wide lens: within a pixel in the middle of the view, where the stabilizer measures registration steps.
        H_, R, C, K = trajectory_bench.camera_views('pole', ball_x, self.dt)
        ground = np.concatenate([truth, np.zeros(truth.shape[:2]+(1,))], -1)
        px, _ = trajectory_bench.project(R, C, K, ground)
        M = base['motion']
        carried = np.einsum('tij,tpj->tpi', M[1:], np.concatenate([px[:-1], np.ones(px.shape[:2]+(1,))[:-1]], -1))
        error, moved = np.linalg.norm(carried-px[1:], axis=-1), np.linalg.norm(px[1:]-px[:-1], axis=-1)
        middle = (px[1:, :, 0] > .25*1920) & (px[1:, :, 0] < .75*1920) & (px[1:, :, 1] > .3*1080) & (px[1:, :, 1] < .95*1080)
        self.assertGreater(np.median(moved[middle]), 1.0)                  # the camera pans
        self.assertLess(error[middle].max(), 1.0)
        self.assertTrue(np.isnan(M[0]).all())
        together = {}
        for kind in ('steps', 'drift'):
            rng, extra = self.generators('pole')
            sim = trajectory_bench.simulate(truth, teams, ball_x, self.dt, 'pole', rng, (kind,), extra)
            self.assertGreater(sim['realism']['calibrationError_m']['rms'], .1, kind)
            np.testing.assert_array_equal(sim['motion'], base['motion'])   # the camera moved as before: only its registration is off
            # How far the error moves the players together from one frame to the next (the median over the players).
            together[kind] = np.nanmax(np.nanmedian(np.linalg.norm(np.diff(sim['obs']-base['obs'], axis=0), axis=-1), axis=1))
        self.assertGreater(together['steps'], .3)                          # the anchors' errors held: everybody steps at an anchor
        self.assertLess(together['drift'], together['steps']/3)            # joined by straight lines: a slow drift
        result = trajectory_bench.as_result(base, [f'A-{k:02d}' for k in range(22)], ['A']*22, np.arange(len(truth))*self.dt, 30.0)
        self.assertIsNone(result['frames'][0]['motion'])
        self.assertEqual(len(result['frames'][1]['motion']), 6)

    def test_glitches_leave_the_truth_unknown_around_a_teleport(self):
        times = np.arange(0, 10, .04)
        truth = np.zeros((len(times), 2, 2))
        truth[:, :, 0], truth[:, :, 1] = 30+2*times[:, None], 20.0
        truth[125:, 0, 0] += 3.0                       # 3 m between frames 124 and 125: 75 m/s
        bad = trajectory_bench.glitches(times, truth)
        self.assertFalse(bad[:, 1].any())
        hit = np.flatnonzero(bad[:, 0])
        # Every frame within GLITCH_HALF of the step, and only those.
        self.assertTrue(times[hit[0]] >= times[124]-trajectory_bench.GLITCH_HALF > times[hit[0]-1])
        self.assertTrue(times[hit[-1]] <= times[125]+trajectory_bench.GLITCH_HALF < times[hit[-1]+1])
        self.assertEqual(len(hit), hit[-1]-hit[0]+1)
        fast = truth.copy()
        fast[:, 1, 0] = 30+11*times                     # 11 m/s is a sprint, not a glitch
        self.assertFalse(trajectory_bench.glitches(times, fast)[:, 1].any())

    def straight_run(self):
        """One player at 5 m/s along the length for 10 s, observed with 0.3 m white noise across, a 0.5 s and a 2 s gap."""
        T = int(10/self.dt)
        t = np.arange(T)*self.dt
        truth = np.stack([10+5*t, np.full(T, 30.0)], 1)[:, None]
        observed = np.ones((T, 1), bool)
        observed[60:75] = False
        observed[150:210] = False
        obs = np.where(observed[..., None], truth+np.random.default_rng(2).normal(0, .3, truth.shape)*[0, 1], np.nan)
        return t, truth, obs, observed, np.tile(np.eye(2)*.09, (T, 1, 1, 1))

    def test_methods_reduce_white_noise_and_bridge_short_gaps_only(self):
        t, truth, obs, observed, R = self.straight_run()
        error = lambda est: float(np.sqrt(np.nanmean(np.sum((est-truth)**2, -1)[observed])))
        raw = error(obs)
        for est in (trajectory_bench.renderer(obs, observed, self.dt), trajectory_bench.savgol(obs, observed, self.dt, .3),
                    trajectory_bench.savgol(obs, observed, self.dt, .5), trajectory_bench.rts(obs, R, observed, self.dt, 1.0)):
            self.assertLess(error(est), .8*raw)
            self.assertTrue(np.isfinite(est[60:75, 0, 0]).all())         # 0.5 s gap: bridged
            self.assertTrue(np.isnan(est[150:210, 0, 0]).all())          # 2 s gap: a new segment
        self.assertLess(error(trajectory_bench.rts(obs, R, observed, self.dt, 1.0)), .4*raw)

    def test_the_renderer_method_is_the_pages_smoothing(self):
        """The bench's 'renderer' against tools/jitter.py's port of tactical.js before the stabilizer on the same samples
        (the timeline's I samples in the gaps: straight between the raw observations)."""
        t, truth, obs, observed, _ = self.straight_run()
        est = trajectory_bench.renderer(obs, observed, self.dt)[:, 0]
        bridged = trajectory_bench.bridgeable(observed, self.dt)[:, 0]
        keep = observed[:, 0] | bridged
        lines = trajectory_bench.bridge(obs, obs, observed, self.dt)[:, 0]
        track = jitter.Track(t[keep], lines[keep, 0], lines[keep, 1], np.where(observed[keep, 0], 0, 1)).smooth_observed(.14)
        self.assertTrue(np.allclose(track.x, est[keep, 0], atol=1e-9) and np.allclose(track.y, est[keep, 1], atol=1e-9))

    def test_lag_glide_and_stationary_vibration(self):
        T = int(8/self.dt)
        t = np.arange(T)*self.dt
        # 4 m/s along the length for 3 s, then standing.
        x = np.where(t < 3, 10+4*t, 22.0)
        truth = np.stack([x, np.full(T, 30.0)], 1)[:, None]
        speed = np.linalg.norm(trajectory_bench.central_velocity(truth, 3, self.dt), axis=-1)
        stops = trajectory_bench.stops(truth, speed, self.dt)
        self.assertEqual(len(stops), 1)
        self.assertTrue(np.allclose(stops[0][2], [1, 0]))
        overshoot = truth.copy()
        overshoot[stops[0][0]:, 0, 0] += np.minimum(.5, 4*(t[stops[0][0]:]-t[stops[0][0]]))   # glides on 0.5 m
        observed = np.ones((T, 1), bool)
        w = {'dt': self.dt, 'm': 3, 'ma': 8, 'speed': speed, 'velocity': trajectory_bench.central_velocity(truth, 3, self.dt),
             'turn': np.zeros((T, 1), bool), 'stops': stops, 'own': observed}
        score = trajectory_bench.Score(list(range(-15, 16)))
        score.add(overshoot, truth, observed, ~observed, w)
        s = score.summary(self.dt)
        self.assertAlmostEqual(s['stopGlide']['p50'], .5, places=2)
        self.assertEqual(s['stationaryStep_cm']['p50'], 0.0)
        self.assertEqual(s['coverage'], {'observed': 1.0, 'gaps': 0.0})
        # A copy three frames late on a weaving path: the velocity correlation peaks at +100 ms.
        y = 30+3*np.sin(2*np.pi*t/2.3)
        weave = np.stack([10+3*t, y], 1)[:, None]
        late = np.full_like(weave, np.nan)
        late[3:] = weave[:-3]
        speed = np.linalg.norm(trajectory_bench.central_velocity(weave, 3, self.dt), axis=-1)
        w.update(speed=speed, velocity=trajectory_bench.central_velocity(weave, 3, self.dt), stops=[])
        score = trajectory_bench.Score(list(range(-15, 16)))
        score.add(late, weave, observed, ~observed, w)
        self.assertAlmostEqual(score.summary(self.dt)['lag_ms'], 100.0, delta=5.0)

    def test_a_run_on_files_in_the_metrica_layout(self):
        import tempfile
        methods = ['raw', 'renderer', 'sg0.3', 'rts']
        try:
            from soccer import trajectory  # noqa: F401
            methods += ['smooth_tracks', 'stabilizer']
        except ImportError:
            pass
        with tempfile.TemporaryDirectory() as tmp:
            metrica_files(tmp)
            report = trajectory_bench.run(tmp, [9], ['pole', 'broadcast'], methods, n_windows=2, length=10.0, progress=lambda s: None,
                                          scenarios=['base', 'all'])
        game = report['games'][9]
        self.assertEqual(len(game['windows']), 2)
        self.assertTrue(game['windows'][0][1] <= 30.0 <= game['windows'][1][0])   # one window per period, none across half time
        self.assertEqual(list(game['scenarios']), ['base', 'all'])
        for camera, (raw_max, smooth_max) in (('pole', (1.0, .4)), ('broadcast', (.3, .15))):   # measured .44/.19-.07, .14/.10-.04
            out = game['scenarios']['base']['cameras'][camera]
            self.assertEqual(list(out['methods']), methods)
            self.assertEqual(out['missing'], {})
            for m, s in out['methods'].items():
                self.assertGreater(s['coverage']['observed'], .9, (camera, m))
                self.assertLess(s['rmse'], raw_max, (camera, m))
                if m != 'raw':
                    self.assertLess(s['rmse'], out['methods']['raw']['rmse'], (camera, m))
            for m in ('rts', 'stabilizer'):
                if m in out['methods']:
                    self.assertLess(out['methods'][m]['rmse'], smooth_max, (camera, m))
            self.assertEqual(out['realism']['truthGlitchShare'], 0.0)
            harsh = game['scenarios']['all']['cameras'][camera]['realism']
            self.assertTrue(set(harsh['extraShare']) == set(trajectory_bench.EXTRA_EVENTS) and harsh['calibrationError_m']['rms'] > 0)
            if 'stabilizer' in methods:
                self.assertIn('stabilizerRegistrationSteps', out['realism'])
        text = trajectory_bench.table(report)
        self.assertIn('### game 9, base scenario, pole camera', text)
        self.assertIn('### game 9, all scenario, broadcast camera', text)

    def test_glitches_are_left_out_and_the_seed_follows_the_camera_name(self):
        import tempfile
        kw = dict(n_windows=1, length=20.0, progress=lambda s: None, scenarios=['base'])
        with tempfile.TemporaryDirectory() as tmp:
            metrica_files(tmp, teleport=True)
            kept = trajectory_bench.run(tmp, [9], ['broadcast'], ['raw', 'rts'], clean=False, **kw)
            clean = trajectory_bench.run(tmp, [9], ['broadcast'], ['raw', 'rts'], **kw)
            both = trajectory_bench.run(tmp, [9], ['pole', 'broadcast'], ['raw', 'rts'], **kw)
            seeds = trajectory_bench.run(tmp, [9], ['broadcast'], ['raw', 'rts'], seeds=2, **kw)
        cams = lambda r: r['games'][9]['scenarios']['base']['cameras']
        # The teleport's 3 m in one frame is a 'sprint' of the truth when scored; left out, nothing sprints.
        self.assertIsNotNone(cams(kept)['broadcast']['methods']['raw']['byTrueSpeed']['sprinting'])
        self.assertIsNone(cams(clean)['broadcast']['methods']['raw']['byTrueSpeed']['sprinting'])
        self.assertGreater(cams(clean)['broadcast']['realism']['truthGlitchShare'], 0.0)
        self.assertEqual(cams(kept)['broadcast']['realism']['truthGlitchShare'], 0.0)
        # A camera's numbers do not depend on the cameras run with it.
        strip = lambda out: {m: {k: v for k, v in s.items() if k != 'seconds'} for m, s in out['methods'].items()}
        self.assertEqual(strip(cams(clean)['broadcast']), strip(cams(both)['broadcast']))
        spread = cams(seeds)['broadcast']['spread']['rts']
        self.assertEqual(spread['rmse']['n'], 2)
        self.assertLessEqual(spread['rmse']['min'], spread['rmse']['mean'])
        self.assertIn('spread over 2 seeds, rts:', trajectory_bench.table(seeds))


# ---------- tools/check_foot_truth.py ----------
class FootTruthRun:
    """A run directory (runs/<key>/<run> layout) with a result.json on the broadcast test camera, and a marks file for it
    in a temporary directory: check_foot_truth.score(run, annotations=marks) reads only these."""

    def __init__(self, people_by_frame, marks, stabilized=None):
        import tempfile
        self.tmp = tempfile.TemporaryDirectory()
        root = self.tmp.name
        self.marks = os.path.join(root, 'foot_positions.json')
        with open(self.marks, 'w', encoding='utf-8') as handle:
            json.dump({'note': 'test', 'clipkey': marks}, handle)
        self.run = os.path.join(root, 'runs', 'clipkey', 'run1')
        os.makedirs(self.run)
        frames = [{'time': round(i/25, 6), 'frame': 100+i, 'calibrated': True, 'calibration': {'H': FOOT_P.ravel().tolist(), 'source': 'landmarks'},
                   'people': people} for i, people in enumerate(people_by_frame)]
        result = {'video': {'width': W, 'height': H, 'fps': 25.0}, 'frames': frames, 'tactical': {'pitchSize': SIZE.to_json()}}
        if stabilized is not None:
            result['stabilized'] = stabilized
        with open(os.path.join(self.run, 'result.json'), 'w', encoding='utf-8') as handle:
            json.dump(result, handle)

    def score(self, computed=False):
        return check_foot_truth.score(self.run, computed, self.marks)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.tmp.cleanup()


FOOT_P = field.orient(broadcast.camera(), W, H)        # world metres -> pixels


def foot_box(world, height_px=60.0, legs_lost=0.0, past_feet=0.0):
    """A normalized [x, y, w, h] box whose bottom-centre is the world point's pixel (moved up by legs_lost or down by
    past_feet, in full box heights), its top a person's height above the feet."""
    u, v = field.project(FOOT_P, [world])[0]
    top = v-height_px
    bottom = v-legs_lost*height_px+past_feet*height_px
    w = .4*height_px
    return [(u-w/2)/W, top/H, w/W, (bottom-top)/H]


def foot_pitch(box):
    fx, fy = box[0]+box[2]/2, box[1]+box[3]
    p = field.unproject(FOOT_P, [(fx*W, fy*H)])[0]
    return [float(p[0]/SIZE.length), float(p[1]/SIZE.width)]


def foot_mark(world, frame, pid, box, sigma=2.0):
    u, v = field.project(FOOT_P, [world])[0]
    return {'time': (frame-100)/25, 'frame': frame, 'id': pid, 'display': pid, 'track': 1, 'box': box, 'foot': [float(u), float(v)], 'sigma_px': sigma,
            'window': 'w', 'note': ''}


class CheckFootTruthTests(unittest.TestCase):
    """tools/check_foot_truth.py on a synthetic run with a known camera: the reference projection through the frame's
    homography, the person found by the marked box (also on a rerun whose ids were handed out anew), the event classes
    against the track's reference height, and the readers of the stored block."""
    world = (60.0, 40.0)

    def person(self, box, pid='A-01', track=1, **extra):
        return {'id': pid, 'track': track, 'box': box, 'pitch': foot_pitch(box), **extra}

    def test_the_reference_is_the_marked_pixel_through_the_frames_homography(self):
        b = foot_box(self.world)
        with FootTruthRun([[self.person(b)]], [foot_mark(self.world, 100, 'A-01', b)]) as run:
            rec = run.score()[0][0]
        self.assertEqual((rec['state'], rec['match'], rec['class']), ('ok', 'id', 'normal'))
        self.assertAlmostEqual(rec['iou'], 1.0, places=9)
        self.assertLess(np.hypot(*(rec['ref']-self.world)), 1e-6)
        self.assertLess(check_foot_truth.error(rec, rec['raw'])[0], 1e-6)
        # 'away' points away from the camera: the pixel above the mark is farther.
        up = field.unproject(FOOT_P, [(rec['mark']['foot'][0], rec['mark']['foot'][1]-10)])[0]
        self.assertGreater(float((up-rec['ref']) @ rec['away']), 0)

    def test_events_against_the_tracks_reference_height(self):
        full = foot_box(self.world)
        # Ten frames of the track at full height around the marked one: h_ref is 60 px, the threshold 9 px.
        for lost, sigma, expected in ((.30, 2.0, 'event'), (.14, 2.0, 'ambiguous'), (.14, .5, 'normal'), (.10, 2.0, 'normal')):
            short = foot_box(self.world, legs_lost=lost)
            frames = [[self.person(full)] for _ in range(5)]+[[self.person(short)]]+[[self.person(full)] for _ in range(5)]
            with FootTruthRun(frames, [foot_mark(self.world, 105, 'A-01', short, sigma)]) as run:
                rec = run.score()[0][0]
            self.assertAlmostEqual(rec['h_ref'], 60.0, places=6)
            self.assertAlmostEqual(rec['threshold'], 9.0, places=6)
            self.assertEqual(rec['class'], expected, (lost, sigma))
        # The legs lost by 30%: the box ends above the feet and the player is thrown away from the camera.
        short = foot_box(self.world, legs_lost=.3)
        with FootTruthRun([[self.person(short)]], [foot_mark(self.world, 100, 'A-01', short)]) as run:
            rec = run.score()[0][0]
        self.assertLess(rec['dy'], -10)
        dist, away = check_foot_truth.error(rec, rec['raw'])
        self.assertGreater(away, .9*dist)
        # A box that runs past the feet is an event too, thrown towards the camera.
        long = foot_box(self.world, past_feet=.3)
        with FootTruthRun([[self.person(long)]], [foot_mark(self.world, 100, 'A-01', long)]) as run:
            rec = run.score()[0][0]
        self.assertTrue(rec['class'] == 'event' and rec['dy'] > 10 and check_foot_truth.error(rec, rec['raw'])[1] < 0)

    def test_the_person_is_found_by_the_marked_box(self):
        other = (63.0, 41.0)
        mine, theirs = foot_box(self.world), foot_box(other)
        # A run without the mark's id: the box.
        people = [self.person(theirs, 'A-07', 9), self.person(mine, 'A-03', 4)]
        with FootTruthRun([people], [foot_mark(self.world, 100, 'A-01', mine)]) as run:
            rec = run.score()[0][0]
        self.assertEqual(rec['match'], 'box')
        self.assertAlmostEqual(rec['iou'], 1.0, places=9)
        self.assertLess(check_foot_truth.error(rec, rec['raw'])[0], 1e-6)
        # A rerun that gave the mark's id to another player: still the box, never the id.
        far = foot_box((75.0, 20.0))
        people = [self.person(far, 'A-01', 9), self.person(mine, 'A-05', 4)]
        with FootTruthRun([people], [foot_mark(self.world, 100, 'A-01', mine)]) as run:
            rec = run.score()[0][0]
        self.assertEqual(rec['match'], 'box')
        self.assertLess(check_foot_truth.error(rec, rec['raw'])[0], 1e-6)
        # Two boxes on the marked player (a second track on the same person): the one with the mark's id.
        twin = list(mine)
        people = [self.person(twin, 'B-12', 7), self.person(mine, 'A-01', 4)]
        with FootTruthRun([people], [foot_mark(self.world, 100, 'A-01', mine)]) as run:
            rec = run.score()[0][0]
        self.assertEqual(rec['match'], 'id')
        # Nobody on the marked box: no person, with the best overlap reported.
        with FootTruthRun([[self.person(far, 'A-01', 9)]], [foot_mark(self.world, 100, 'A-01', mine)]) as run:
            rec = run.score()[0][0]
        self.assertEqual(rec['state'], 'no person')
        self.assertIn('best box overlap 0.00', check_foot_truth.row(rec, []))

    def test_marks_outside_the_run_and_without_registration(self):
        b = foot_box(self.world)
        people = [self.person(b)]
        with FootTruthRun([people, people], [foot_mark(self.world, 100, 'A-01', b), foot_mark(self.world, 250, 'A-01', b)]) as run:
            path = os.path.join(run.run, 'result.json')
            with open(path, encoding='utf-8') as handle:
                r = json.load(handle)
            r['frames'][0]['calibration'] = {}
            with open(path, 'w', encoding='utf-8') as handle:
                json.dump(r, handle)
            records, _, _ = run.score()
            lines = check_foot_truth.summary(records, [])
        self.assertEqual([rec['state'] for rec in records], ['no registration'])   # frame 250 is not in the run
        self.assertIn('marks scored 0 of 1', lines[-1])

    def test_the_stored_block_is_read_in_metres_and_anonymous_by_track(self):
        other = (30.0, 20.0)
        b, c = foot_box(self.world), foot_box(other)
        people = [self.person(b), {'id': None, 'track': 7, 'state': 'uncertain', 'team': 'B', 'box': c, 'pitch': foot_pitch(c)}]
        block = {'frameTimes': [0.0], 'ids': {'A-01': {'x': [(self.world[0]+.5)/SIZE.length], 'y': [self.world[1]/SIZE.width], 'kind': ['O']}},
                 'anonymous': {'7': {'x': [other[0]/SIZE.length], 'y': [(other[1]-.25)/SIZE.width], 'kind': ['O']}}}
        marks = [foot_mark(self.world, 100, 'A-01', b), dict(foot_mark(other, 100, None, c), id='?')]
        with FootTruthRun([people], marks, stabilized=block) as run:
            records, _, sources = run.score()
        self.assertEqual(sources, ['stored'])
        self.assertAlmostEqual(check_foot_truth.error(records[0], records[0]['stored'][0])[0], .5, places=6)
        self.assertEqual(records[1]['match'], 'box')
        self.assertAlmostEqual(check_foot_truth.error(records[1], records[1]['stored'][0])[0], .25, places=6)


if __name__ == '__main__':
    unittest.main()
