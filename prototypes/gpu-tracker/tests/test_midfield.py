"""The keyframe-less midfield anchor as a pinhole camera (midfield.solve allow_pinhole): the focal length from
the centre circle and its diameter, the two failure modes of a camera standing in the halfway line's plane
(a symmetric view, mirror-image solutions) and how people's heights and the other paint resolve them,
honest confidence (unconfirmed anchors stay below HIGH, flat ellipses lower the ceiling, capped chains stay
capped until paint independent of the anchor confirms them), the projective fallback and the refusals it
must not paper over, the touchlines found through the solved camera, the reported residual and field size,
the on-grass circle search, and the gate: runs with keyframes keep the shipped route."""
import math
import os
import unittest
from pathlib import Path
from unittest import mock

# Before anything imports ultralytics (the gate tests build BoT-SORT): its settings are the pipeline's own file
# (settings/Ultralytics), never the user's global one.
(Path(__file__).resolve().parents[1]/'settings'/'Ultralytics').mkdir(parents=True, exist_ok=True)
os.environ['YOLO_CONFIG_DIR'] = str(Path(__file__).resolve().parents[1]/'settings')

import cv2
import numpy as np

import broadcast
from scene import H, W
from soccer import field, midfield
from soccer.pitch import CIRCLE_MIN_DENSITY, PitchDetector, circle_points, find_circle
from soccer.pitch_model import HIGH, PitchTracker, combine

SIZE = field.PitchSize()


def camera(position, target, focal, roll=0.0):
    """A pinhole camera (square pixels, principal point at the image centre) at `position` (world metres,
    z down into the ground) looking at the ground point `target`: (H world -> pixels oriented, K, R, t)."""
    C = np.array(position, np.float64)
    z = np.array([target[0], target[1], 0.0])-C
    z /= np.linalg.norm(z)
    down = np.array([0.0, 0.0, 1.0])
    y = down-(down @ z)*z
    y /= np.linalg.norm(y)
    x = np.cross(y, z)
    if roll:
        c, s = math.cos(roll), math.sin(roll)
        x, y = c*x+s*y, -s*x+c*y
    R = np.vstack([x, y, z])
    t = -R @ C
    K = np.array([[focal, 0, W/2], [0, focal, H/2], [0, 0, 1.0]])
    P = K @ np.c_[R[:, 0], R[:, 1], t]
    return field.orient(P/np.linalg.norm(P), W, H), K, R, t


CAMERAS = {   # position, target, focal length (px at 1280 x 720)
    'broadcast-centre': ((52.5, 113, -22), (52.5, 34), 1400),   # in the halfway line's plane, looking along it
    'broadcast-pan': ((52.5, 113, -22), (62.5, 34), 1400),      # ... panned 10 m to the right
    'pole-centre': ((52.4, 81.5, -5.1), (52.5, 34), 900),       # amateur 3's pole (from the user's keyframes)
    'pole-left': ((52.4, 81.5, -5.1), (35, 34), 800),
    'pole-right': ((52.4, 81.5, -5.1), (70, 34), 800),
}


def exact_evidence(P):
    """The circle's image points (normalized, in frame) and the halfway line's image segment."""
    cx, cy = field.circle_centre(SIZE)
    pts = field.project(P, [(cx+field.RADIUS*math.cos(a), cy+field.RADIUS*math.sin(a)) for a in np.linspace(0, 2*math.pi, 240, endpoint=False)])
    a, b = field.image_segment(P, (cx, 0), (cx, SIZE.width))
    return ({'points': [(x/W, y/H) for x, y in pts if 0 <= x < W and 0 <= y < H]},
            {'a': (a[0]/W, a[1]/H), 'b': (b[0]/W, b[1]/H), 'support': .5, 'kind': 'v'})


def segment(P, a, b, support=.5):
    """A detected paint segment (normalized, clipped to the frame) of the ground line a-b (metres), or None."""
    q = field.image_segment(P, a, b)
    ok, p0, p1 = cv2.clipLine((0, 0, W, H), tuple(int(round(v)) for v in q[0]), tuple(int(round(v)) for v in q[1]))
    return {'a': (p0[0]/W, p0[1]/H), 'b': (p1[0]/W, p1[1]/H), 'support': support, 'kind': 'h'} if ok else None


def people(K, R, t, n=30, seed=0, height=1.75, jitter=.05, xs=(15, 90), ys=(5, 63)):
    """Normalized boxes of standing people (height +- jitter) spread over the ground area xs x ys (metres; the
    pitch by default), as a detector reports them."""
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < n:
        X = np.array([rng.uniform(*xs), rng.uniform(*ys), 0.0])
        tall = height*(1+jitter*rng.standard_normal())
        foot, head = K @ (R @ X+t), K @ (R @ (X+[0, 0, -tall])+t)
        if foot[2] <= 0 or head[2] <= 0:
            continue
        fx, fy, hy = foot[0]/foot[2], foot[1]/foot[2], head[1]/head[2]
        box = [(fx-.175*(fy-hy))/W, hy/H, .35*(fy-hy)/W, (fy-hy)/H]
        if box[0] > .01 and box[0]+box[2] < .99 and box[1] > .02 and box[1]+box[3] < .98:
            out.append(box)
    return out


def landmark_error(Ha, Hb):
    """(max, median) metres between where Ha puts the landmarks Hb shows in the frame and where they are."""
    marks = np.array(list(field.landmarks(SIZE).values()), np.float64)
    pb = field.project(Hb, marks)
    seen = (field.depth(Hb, marks) > 0) & (pb[:, 0] >= 0) & (pb[:, 0] < W) & (pb[:, 1] >= 0) & (pb[:, 1] < H)
    d = np.hypot(*(field.unproject(Ha, pb[seen])-marks[seen]).T)
    return float(d.max()), float(np.median(d))


def rendered(name):
    """A rendered frame of a camera, analysed: (P, detector, model, circle, halfway segment)."""
    P = camera(*CAMERAS[name])[0]
    img = broadcast.render(P, stands=False)
    detector = PitchDetector()
    model = detector.analyze(img, 0.0)
    circle, halfway = find_circle(detector.masks, model, on_grass=True)
    return P, detector, model, circle, halfway


class PinholeGeometryTests(unittest.TestCase):
    """The algebra: the circle's cone, the focal scan, the seeds."""

    def test_the_cone_gives_the_ground_plane_and_the_scan_the_focal_length(self):
        P, K, R, t = camera((40, 75, -8), (55, 30), 700, roll=.03)   # off the halfway line's plane
        circle, halfway = exact_evidence(P)
        T, s = midfield._normalizer(W, H)
        Ti = np.linalg.inv(T)
        C = Ti.T @ midfield.conic(midfield.ellipse_from_points([(x*W, y*H) for x, y in circle['points']])) @ Ti
        h = Ti.T @ midfield._line((halfway['a'][0]*W, halfway['a'][1]*H), (halfway['b'][0]*W, halfway['b'][1]*H))
        v = np.linalg.solve(C, h)
        phi = 700/s
        normals = [n for n, _ in midfield.circle_normals(np.diag([phi, phi, 1.0]) @ C @ np.diag([phi, phi, 1.0]))]
        truth = R @ [0, 0, 1.0]   # the ground's normal into the ground, in camera coordinates
        self.assertGreater(max(abs(n @ truth) for n in normals), 1-1e-6)
        seeds, spread = midfield.focal_seeds(C, v, W, s)
        self.assertGreater(spread, midfield.MIN_SPREAD)
        self.assertTrue(any(abs(seed*s-700) < 7 for seed in seeds), [seed*s for seed in seeds])
        poses = midfield.seed_poses(C, v, phi)
        centre = R @ [52.5, 34, 0]+t   # the circle's centre in camera coordinates
        self.assertTrue(any(np.linalg.norm(p-centre) < .05 for _, _, p in poses), (poses, centre))

    def test_a_view_along_the_halfway_line_cannot_tell_the_focal_length(self):
        circle, halfway = exact_evidence(camera(*CAMERAS['broadcast-centre'])[0])
        result = midfield.solve(circle, halfway, [], W, H, SIZE, allow_pinhole=True)
        self.assertNotIn('H', result)
        self.assertIn('symmetric about the halfway line', result['reason'])

    def test_the_people_heights_resolve_the_symmetric_view(self):
        P, K, R, t = camera(*CAMERAS['pole-centre'])
        circle, halfway = exact_evidence(P)
        result = midfield.solve(circle, halfway, [], W, H, SIZE, allow_pinhole=True, people=people(K, R, t))
        self.assertIn('H', result, result.get('reason'))
        d = result['details']
        self.assertAlmostEqual(d['focalPx'], 900, delta=45)
        self.assertLess(landmark_error(result['H'], P)[1], 1.0)
        self.assertIn('people heights', d['constraints'])
        self.assertAlmostEqual(d['peopleHeightMetres'], 1.75, delta=.15)
        self.assertAlmostEqual(d['cameraHeightMetres'], 5.1, delta=.6)

    def test_the_camera_never_stands_over_the_far_half(self):
        # Panned off the halfway line, the circle and its diameter fit two cameras, one of them over the far half
        # (a mirror image); only the near one is a camera whose far touchline is the one away from it.
        P = camera(*CAMERAS['broadcast-pan'])[0]
        circle, halfway = exact_evidence(P)
        result = midfield.solve(circle, halfway, [], W, H, SIZE, allow_pinhole=True)
        self.assertIn('H', result, result.get('reason'))
        self.assertLess(landmark_error(result['H'], P)[0], .05)
        self.assertGreater(result['details']['cameraPosition'][1], SIZE.width)


class HonestConfidenceTests(unittest.TestCase):
    """A circle, its diameter and an assumed camera model never make a HIGH anchor on their own."""

    def test_an_unconfirmed_anchor_stays_below_high_and_a_flat_ellipse_lowers_the_ceiling(self):
        P, K, R, t = camera(*CAMERAS['pole-left'])
        circle, halfway = exact_evidence(P)
        result = midfield.solve(circle, halfway, [], W, H, SIZE, allow_pinhole=True, people=people(K, R, t))
        self.assertIn('H', result, result.get('reason'))
        d = result['details']
        self.assertEqual(d['confirmed'], [])
        ratio = d['ellipseRatio']
        self.assertLess(ratio, midfield.FLAT_RATIO)
        self.assertAlmostEqual(d['cap'], round(midfield.PINHOLE_CAP*ratio/midfield.FLAT_RATIO, 3), places=3)
        self.assertLessEqual(result['observability'], result['cap']+1e-9)
        self.assertLess(combine(1.0, result['observability']), HIGH)
        self.assertEqual(d['size']['width']['source'], 'assumed')
        self.assertEqual(d['size']['length']['source'], 'assumed')
        self.assertIn('principal point at the image centre', d['intrinsics'])

    def test_paint_the_camera_predicts_confirms_it_and_lifts_the_ceiling(self):
        P, detector, model, circle, halfway = rendered('broadcast-pan')
        self.assertIsNotNone(circle)
        ellipse = midfield.ellipse_from_points([(x*W, y*H) for x, y in circle['points']])
        result = midfield.solve(circle, halfway, midfield.touchline_candidates(model, ellipse, W, H), W, H, SIZE, detector.masks,
                                allow_pinhole=True, markings=model.markings)
        self.assertIn('H', result, result.get('reason'))
        d = result['details']
        self.assertEqual(d['route'], 'pinhole')
        self.assertTrue(d['confirmed'], d)
        self.assertEqual(result['cap'], 1.0)
        worst, median = landmark_error(result['H'], P)
        self.assertLess(worst, 3.0)
        self.assertLess(median, 1.0)
        self.assertAlmostEqual(d['focalPx'], 1400, delta=140)

    def test_a_capped_chain_stays_capped_until_fine_paint_confirms_it(self):
        P = camera(*CAMERAS['broadcast-pan'])[0]
        tracker = PitchTracker(W, H, SIZE)
        self.assertTrue(tracker.anchor(np.linalg.inv(P), 0.0, observability=.25, source='auto-midfield', cap=.25))
        tracker.temporal = .8   # an aged anchor, so paint refines it
        cx, cy = field.circle_centre(SIZE)
        ring = field.project(P, [(cx+field.RADIUS*math.cos(a), cy+field.RADIUS*math.sin(a)) for a in np.linspace(0, 2*math.pi, 120)])
        stray = [(40.0+3*i, 60.0) for i in range(20)]   # paint the shipped search would fit (far from the circle)
        halfway = [{'kind': 'v', 'support': .5, 'a': tuple(field.project(P, [(52.5, 0)])[0]/[W, H]), 'b': tuple(field.project(P, [(52.5, 68)])[0]/[W, H])}]
        tracker.refine(halfway, .2, (), [tuple(p) for p in ring]+stray)
        self.assertEqual(tracker.state, 'refined')
        self.assertLessEqual(tracker.observability, .25+1e-9)
        self.assertLessEqual(tracker.refinement['own'], .25+1e-9)
        self.assertEqual(len(tracker._near_circle([tuple(p) for p in ring]+stray)), len(ring))
        near = [{'kind': 'h', 'support': .5, 'a': tuple(field.project(P, [(30, 68)])[0]/[W, H]), 'b': tuple(field.project(P, [(80, 68)])[0]/[W, H])}]
        tracker.temporal = .8
        tracker.refine(halfway+near, .4, (), [tuple(p) for p in ring])
        self.assertIsNone(tracker.cap)
        self.assertGreater(tracker.observability, .25)

    def test_the_anchors_own_lines_never_lift_its_ceiling(self):
        # The anchor was fitted to the halfway line and the near touchline: seeing them again is not independent
        # paint, however finely they match. A box line it did not use is, unless the detector barely saw it.
        P = camera(*CAMERAS['broadcast-pan'])[0]
        cx, cy = field.circle_centre(SIZE)
        ring = [tuple(p) for p in field.project(P, [(cx+field.RADIUS*math.cos(a), cy+field.RADIUS*math.sin(a)) for a in np.linspace(0, 2*math.pi, 120)])]
        own = [segment(P, (52.5, 0), (52.5, 68)), segment(P, (30, 68), (80, 68))]
        box = segment(P, (88.5, 13.84), (88.5, 54.16))
        self.assertTrue(all(own) and box)
        tracker = PitchTracker(W, H, SIZE)
        details = {'evidence': ['halfway line', 'near touchline', 'centre circle']}
        self.assertTrue(tracker.anchor(np.linalg.inv(P), 0.0, observability=.25, details=details, source='auto-midfield', cap=.25))
        self.assertEqual(tracker.fitted, {'halfway line', 'near touchline', 'centre circle'})
        for t, segs in ((.2, own), (.4, own+[dict(box, support=.03)])):
            tracker.temporal = .8
            tracker.refine(segs, t, (), ring)
            self.assertEqual(tracker.state, 'refined')
            self.assertIn('near touchline', {m['name'] for m in tracker.matched})
            self.assertEqual(tracker.cap, .25)
            self.assertLessEqual(tracker.observability, .25+1e-9)
            self.assertLessEqual(tracker.refinement['own'], .25+1e-9)
        tracker.temporal = .8
        tracker.refine(own+[box], .6, (), ring)
        self.assertIsNone(tracker.cap)
        self.assertGreater(tracker.observability, .25)
        self.assertTrue(any(k == 'confirmed' for _, k, _ in tracker.events))
        # An explicit list of fitted lines takes precedence over the details' evidence; a cap of 1 keeps none.
        tracker.anchor(np.linalg.inv(P), 1.0, observability=.25, details=details, source='auto-midfield', cap=.25, fitted=['halfway line'])
        self.assertEqual(tracker.fitted, {'halfway line'})
        tracker.anchor(np.linalg.inv(P), 2.0, observability=.25, details=details, source='auto-midfield', cap=1.0)
        self.assertEqual((tracker.cap, tracker.fitted), (None, frozenset()))

    def test_a_capped_chain_is_not_pulled_onto_a_line_off_its_prediction(self):
        # Amateur 3: the boards' foot 6 m beyond the far touchline is matched as the far touchline (a few pixels off
        # at metres per pixel); it must neither move the chain nor, once the chain sits on it, confirm it.
        # (Unfiltered, one refinement moves the far touchline 5 m onto it and the next matches it at 0.0 px.)
        P = camera(*CAMERAS['pole-left'])[0]
        halfway, boards = segment(P, (52.5, 0), (52.5, 68)), segment(P, (25, -6), (80, -6))
        self.assertEqual([m['name'] for m in field.match_segments(P, SIZE, [boards], W, H)], ['far touchline'])
        tracker = PitchTracker(W, H, SIZE)
        tracker.anchor(np.linalg.inv(P), 0.0, observability=.23, details={'evidence': ['halfway line', 'centre circle']}, source='auto-midfield', cap=.23)
        far = [(30, 0), (52.5, 0), (75, 0)]
        for t in (.2, .4, .6):
            tracker.temporal = .8
            tracker.refine([halfway, boards], t, (), None)   # no circle paint near the prediction on these frames
            self.assertEqual(tracker.state, 'refined')
            self.assertEqual([m['name'] for m in tracker.matched], ['halfway line'])
            self.assertEqual(tracker.cap, .23)
            self.assertLess(np.hypot(*(field.unproject(tracker.H, field.project(P, far))-far).T).max(), .3)
        # A line that agrees with the chain to CAP_METRES (the near touchline, metres from the camera) refines it
        # and lifts the ceiling.
        tracker.temporal = .8
        tracker.refine([halfway, boards, segment(P, (30, 68), (80, 68))], .8, (), None)
        self.assertEqual(sorted(m['name'] for m in tracker.matched), ['halfway line', 'near touchline'])
        self.assertIsNone(tracker.cap)

    def test_an_uncapped_anchor_refines_as_before(self):
        tracker = PitchTracker(W, H, SIZE)
        tracker.anchor(np.linalg.inv(camera(*CAMERAS['broadcast-pan'])[0]), 0.0, observability=.25)
        self.assertIsNone(tracker.cap)
        self.assertFalse(tracker.capped)
        self.assertEqual(tracker.fitted, frozenset())


class ProjectiveFallbackTests(unittest.TestCase):
    """When the pinhole route refuses for want of evidence, a touchline at the assumed width can still anchor the
    projective construction, capped like an unconfirmed pinhole anchor; never after a contradiction."""

    def test_the_fallback_is_capped_lower_for_a_flat_ellipse_and_stays_capped(self):
        # A pole camera 6 m off the halfway line's plane: the circle is grazing (ratio .11), and the pinhole camera
        # without people is unobservable (metres of uncertainty); the near touchline at 68 m makes the construction.
        P = camera((58.4, 81.5, -5.1), (52.5, 34), 900)[0]
        circle, halfway = exact_evidence(P)
        near = segment(P, (35, 68), (70, 68))
        result = midfield.solve(circle, halfway, [('near', near)], W, H, SIZE, allow_pinhole=True)
        self.assertIn('H', result, result.get('reason'))
        d = result['details']
        self.assertEqual(d['route'], 'projective')
        self.assertTrue(d['pinholeRefused'].startswith('pinhole camera: observability'), d['pinholeRefused'])
        self.assertLess(d['ellipseRatio'], midfield.FLAT_RATIO)
        self.assertAlmostEqual(result['cap'], midfield.PINHOLE_CAP*d['ellipseRatio']/midfield.FLAT_RATIO, delta=.005)
        self.assertLessEqual(result['observability'], result['cap']+1e-9)
        self.assertGreaterEqual(result['observability'], midfield.MIN_OBSERVABILITY)
        self.assertTrue(d['widthAssumed'])
        self.assertEqual(d['size'], SIZE.dimensions())
        self.assertEqual(result['fitted'], ['halfway line', 'near touchline'])
        self.assertLess(landmark_error(result['H'], P)[0], .5)
        # The chain it starts stays capped when it sees its own touchline again.
        tracker = PitchTracker(W, H, SIZE)
        self.assertTrue(tracker.anchor(np.linalg.inv(result['H']), 0.0, observability=result['observability'], details=d, source='auto-midfield',
                                       cap=result['cap'], fitted=result['fitted']))
        tracker.temporal = .8
        tracker.refine([halfway, near], .2, (), [(x*W, y*H) for x, y in circle['points']])
        self.assertEqual(tracker.state, 'refined')
        self.assertEqual(tracker.cap, result['cap'])
        self.assertLessEqual(tracker.observability, result['cap']+1e-9)

    def test_a_round_ellipse_keeps_the_full_ceiling(self):
        P = camera(*CAMERAS['broadcast-centre'])[0]
        circle, halfway = exact_evidence(P)
        result = midfield.solve(circle, halfway, [('far', segment(P, (30, 0), (75, 0)))], W, H, SIZE, allow_pinhole=True)
        self.assertIn('H', result, result.get('reason'))
        self.assertEqual(result['details']['route'], 'projective')
        self.assertGreater(result['details']['ellipseRatio'], midfield.FLAT_RATIO)
        self.assertEqual(result['cap'], midfield.PINHOLE_CAP)
        self.assertLess(combine(1.0, result['observability']), HIGH)

    def test_no_fallback_when_two_cameras_explain_the_paint(self):
        # Solutions that explain the paint alike (within AMBIGUOUS_CHI2) but stand metres apart make the frame
        # ambiguous: a contradiction the assumed width would only paper over by picking one. (The seeds of this
        # view converge to one camera; registration._disagreement is forced to say they are 10 m apart.)
        P, K, R, t = camera(*CAMERAS['pole-centre'])
        circle, halfway = exact_evidence(P)
        near = segment(P, (35, 68), (70, 68))
        with mock.patch.object(midfield.registration, '_disagreement', lambda *args: 10.0), \
                mock.patch.object(midfield, 'solve', wraps=midfield.solve) as solve:
            result = midfield.solve(circle, halfway, [('near', near)], W, H, SIZE, allow_pinhole=True, people=people(K, R, t))
        self.assertNotIn('H', result)
        self.assertIn('ambiguous', result['reason'])
        self.assertTrue(result['contradiction'])
        self.assertEqual(solve.call_count, 1)   # the projective construction was not tried
        # Unforced, the same evidence anchors (one camera, no ambiguity).
        self.assertIn('H', midfield.solve(circle, halfway, [('near', near)], W, H, SIZE, allow_pinhole=True, people=people(K, R, t)))

    def test_no_fallback_when_the_people_stand_off_the_solved_pitch(self):
        P, K, R, t = camera(*CAMERAS['pole-left'])
        circle, halfway = exact_evidence(P)
        crowd = people(K, R, t, xs=(-45, -12), ys=(0, 68))   # standing beyond the left goal line
        far = segment(P, (25, 0), (80, 0))
        for touchlines in ([], [('far', far)]):
            result = midfield.solve(circle, halfway, touchlines, W, H, SIZE, allow_pinhole=True, people=crowd)
            self.assertNotIn('H', result)
            self.assertIn('off the solved pitch', result['reason'])
            self.assertTrue(result['contradiction'])
            self.assertEqual(result['details']['peopleOnPitch'], 0.0)


class GroundTouchlineTests(unittest.TestCase):
    """Touchlines found through the solved camera: the one nearest the predicted touchline, and only while the
    joint fit holds it."""

    def test_a_straight_line_beyond_the_touchline_does_not_replace_it(self):
        for name in ('pole-left', 'pole-right', 'broadcast-pan'):
            P, K, R, t = camera(*CAMERAS[name])
            circle, halfway = exact_evidence(P)
            crowd = people(K, R, t)
            far = segment(P, (25, 0), (80, 0))
            alone = midfield.solve(circle, halfway, [], W, H, SIZE, allow_pinhole=True, people=crowd, markings=[far])
            self.assertIn('far touchline', alone['details']['constraints'], name)
            measured = alone['details']['size']['width']['measured']
            self.assertAlmostEqual(measured, SIZE.width, delta=1.0)
            for beyond in (6.0, 9.0):   # board feet, a running track's edge, the next pitch's line
                junk = segment(P, (25, -beyond), (80, -beyond))
                result = midfield.solve(circle, halfway, [], W, H, SIZE, allow_pinhole=True, people=crowd, markings=[junk, far])
                self.assertIn('far touchline', result['details']['constraints'], name)
                self.assertEqual(result['details']['size']['width']['measured'], measured, (name, beyond))
                self.assertAlmostEqual(landmark_error(result['H'], P)[0], landmark_error(alone['H'], P)[0], delta=.01)

    def test_a_parallel_line_beyond_an_undetected_touchline_is_not_one(self):
        # Only the line 6 m beyond the far touchline is detected (amateur 3 at 127 and 148 s: the boards' foot, 6-11 m
        # beyond the painted touchline the detector missed), found through the camera or slope-classified as the far
        # boundary: it is farther than TOUCHLINE_REACH from the predicted touchline, so the anchor is the one without it.
        for name in ('pole-left', 'pole-right', 'broadcast-pan'):
            P, K, R, t = camera(*CAMERAS[name])
            circle, halfway = exact_evidence(P)
            crowd = people(K, R, t)
            junk = segment(P, (25, -6), (80, -6))
            without = midfield.solve(circle, halfway, [], W, H, SIZE, allow_pinhole=True, people=crowd)
            for touchlines in ([], [('far', junk)]):
                result = midfield.solve(circle, halfway, touchlines, W, H, SIZE, allow_pinhole=True, people=crowd, markings=[junk])
                self.assertIn('H', result, (name, result.get('reason')))
                self.assertNotIn('far touchline', result['details']['constraints'], name)
                self.assertNotIn('measured', result['details']['size']['width'], name)
                self.assertAlmostEqual(landmark_error(result['H'], P)[0], landmark_error(without['H'], P)[0], delta=.01)

    def test_the_joint_fit_must_hold_its_touchlines(self):
        # The width a solution measures must stay within 2 TOUCHLINE_REACH of the assumed one ...
        T, _ = midfield._normalizer(W, H)
        solution = lambda half: {'layout': {'half': 7}, 'half': half, 'data': {'size_info': SIZE}}
        self.assertTrue(midfield._touchlines_hold(solution(SIZE.width/2+midfield.TOUCHLINE_REACH-.1), [], T))
        self.assertFalse(midfield._touchlines_hold(solution(SIZE.width/2+midfield.TOUCHLINE_REACH+.1), [], T))
        self.assertFalse(midfield._touchlines_hold(dict(solution(SIZE.width/2+midfield.TOUCHLINE_REACH+.1), data={'size_info': field.PitchSize(105, 60)}), [], T))
        self.assertTrue(midfield._touchlines_hold(dict(solution(37.0), data={'size_info': field.PitchSize(105, 76, 'landmarks')}), [], T))
        # ... and the joint fit of the joined touchlines is held to it: a joint fit that does not hold them drops them.
        P, K, R, t = camera(*CAMERAS['pole-left'])
        circle, halfway = exact_evidence(P)
        crowd, far = people(K, R, t), segment(P, (25, 0), (80, 0))
        joined = midfield.solve(circle, halfway, [], W, H, SIZE, allow_pinhole=True, people=crowd, markings=[far])
        self.assertIn('far touchline', joined['details']['constraints'])
        with mock.patch.object(midfield, '_touchlines_hold', lambda c, used, T: 'half' not in c['layout']):
            dropped = midfield.solve(circle, halfway, [], W, H, SIZE, allow_pinhole=True, people=crowd, markings=[far])
        self.assertIn('H', dropped, dropped.get('reason'))
        self.assertNotIn('far touchline', dropped['details']['constraints'])


class ReportTests(unittest.TestCase):
    """What the anchor reports: the true residual, and where the field size comes from."""

    def test_the_reported_residual_is_the_fit_residual_without_the_robust_loss(self):
        # Every 20th circle point pushed 20 px (10 noise sigmas) outwards: the robust fit discounts it, the report
        # must not (soft L1 counts 10 sigmas as 6.7; the robust value was 21 % lower here). The residual is the
        # circle points' Sampson distance to the solved circle's image, computed here from the returned H.
        P, K, R, t = camera((40, 75, -8), (55, 30), 700)   # off the halfway line's plane: a well-determined camera
        circle, halfway = exact_evidence(P)
        pts = np.array([(x*W, y*H) for x, y in circle['points']])
        centre = pts.mean(axis=0)
        out = (pts-centre)/np.hypot(*(pts-centre).T)[:, None]
        moved = pts+20*out*(np.arange(len(pts)) % 20 == 0)[:, None]
        result = midfield.solve({'points': [(x/W, y/H) for x, y in moved]}, halfway, [], W, H, SIZE, allow_pinhole=True, people=people(K, R, t))
        self.assertIn('H', result, result.get('reason'))
        cx, cy = field.circle_centre(SIZE)
        Q = np.array([[1, 0, -cx], [0, 1, -cy], [-cx, -cy, cx*cx+cy*cy-field.RADIUS**2]])
        Hi = np.linalg.inv(result['H'])
        C = Hi.T @ Q @ Hi
        p = np.c_[moved, np.ones(len(moved))]
        Cp = p @ C
        sampson = np.einsum('ij,ij->i', p, Cp)/(2*np.hypot(Cp[:, 0], Cp[:, 1]))
        expected = math.sqrt(float(np.sum(sampson**2))/(len(moved)+midfield.LINE_SAMPLES))   # the halfway line's samples lie on it
        self.assertAlmostEqual(result['rms_px'], expected, delta=.01*expected)
        self.assertEqual(result['details']['rmsPixels'], round(result['rms_px'], 1))
        self.assertAlmostEqual(result['details']['residual'], 1-result['rms_px']/(.02*W), delta=.001)

    def test_a_learned_field_size_is_reported_as_learned(self):
        P = camera(*CAMERAS['broadcast-pan'])[0]
        circle, halfway = exact_evidence(P)
        learned = field.PitchSize(100.0, 64.0, 'landmarks')
        result = midfield.solve(circle, halfway, [], W, H, learned, allow_pinhole=True)
        self.assertIn('H', result, result.get('reason'))
        self.assertEqual(result['details']['size'], learned.dimensions())
        self.assertEqual(result['details']['size']['width']['source'], 'landmarks')
        assumed = midfield.solve(circle, halfway, [], W, H, SIZE, allow_pinhole=True)
        self.assertEqual(assumed['details']['size']['length'], {'value': SIZE.length, 'source': 'assumed'})


class CircleDensityTests(unittest.TestCase):
    """The on-grass circle needs CIRCLE_MIN_DENSITY inlier pixels per pixel of arc seen, counted on every pixel of
    the search window (not on the max_points subsample, whose count shrinks with the clutter around the circle)."""

    def masks(self, step=1):
        w, h, angle = 640, 360, 60
        ridge = np.zeros((h, w), np.uint8)
        cv2.ellipse(ridge, (300, 180), (150, 22), angle, 0, 360, 1, 1)
        if step > 1:
            ys, xs = np.nonzero(ridge)
            keep = np.zeros_like(ridge)
            order = np.argsort(np.arctan2(ys-180, xs-300))
            keep[ys[order[::step]], xs[order[::step]]] = 1
            ridge = keep
        t = math.radians(angle)
        minor = np.array([-math.sin(t), math.cos(t)])
        a, b = np.array([300, 180])-120*minor, np.array([300, 180])+120*minor
        return {'w': w, 'h': h, 'ridge': ridge.astype(bool)}, {'a': (a[0]/w, a[1]/h), 'b': (b[0]/w, b[1]/h), 'support': .5, 'kind': 'v'}

    def test_a_sparse_scatter_on_an_ellipse_is_not_paint(self):
        masks, halfway = self.masks(step=3)
        self.assertIsNotNone(circle_points(masks, halfway, []))
        self.assertIsNone(circle_points(masks, halfway, [], min_density=CIRCLE_MIN_DENSITY))

    def test_the_density_is_counted_on_every_pixel_of_the_window(self):
        # A solid arc searched with a subsample of 120 of its pixels: the subsample's inliers alone fall short of
        # the density, every pixel of the window does not.
        masks, halfway = self.masks()
        found = circle_points(masks, halfway, [], max_points=120, min_density=CIRCLE_MIN_DENSITY)
        self.assertIsNotNone(found)
        (_, _), (ma, mi), _ = found['ellipse']
        a, b = ma*masks['w']/2, mi*masks['h']/2
        perimeter = math.pi*(3*(a+b)-math.sqrt((3*a+b)*(a+3*b)))
        self.assertLess(found['inliers'], CIRCLE_MIN_DENSITY*found['coverage']*perimeter)


class RenderedViewTests(unittest.TestCase):
    """Rendered frames of pinhole cameras through the detector, the on-grass circle search and the route."""

    def test_pole_camera_views_anchor_within_a_few_metres(self):
        for name in ('pole-left', 'pole-right'):
            P, detector, model, circle, halfway = rendered(name)
            self.assertIsNotNone(circle, name)
            ellipse = midfield.ellipse_from_points([(x*W, y*H) for x, y in circle['points']])
            result = midfield.solve(circle, halfway, midfield.touchline_candidates(model, ellipse, W, H), W, H, SIZE, detector.masks,
                                    allow_pinhole=True, markings=model.markings)
            self.assertIn('H', result, (name, result.get('reason')))
            worst, median = landmark_error(result['H'], P)
            self.assertLess(worst, 3.5, name)
            self.assertLess(median, 1.5, name)
            self.assertAlmostEqual(result['details']['cameraHeightMetres'], 5.1, delta=1.0)

    def test_the_far_flat_circle_is_found_on_the_grass(self):
        P = camera(*CAMERAS['pole-centre'])[0]
        detector = PitchDetector()
        model = detector.analyze(broadcast.render(P, stands=False), 0.0)
        circle, _ = find_circle(detector.masks, model, on_grass=True)
        self.assertIsNotNone(circle)
        cx, cy = field.circle_centre(SIZE)
        truth = midfield.ellipse_from_points(field.project(P, [(cx+field.RADIUS*math.cos(a), cy+field.RADIUS*math.sin(a)) for a in np.linspace(0, 2*math.pi, 400)]))
        found = midfield.ellipse_from_points([(x*W, y*H) for x, y in circle['points']])
        self.assertLess(math.hypot(found[0][0]-truth[0][0], found[0][1]-truth[0][1]), 3.0)
        self.assertAlmostEqual(max(found[1]), max(truth[1]), delta=.05*max(truth[1]))
        self.assertLess(min(truth[1])/max(truth[1]), .15)   # a grazing circle, as amateur 3's
        # The paint inside a person's box is not the circle's.
        box = (circle['points'][0][0]-.01, circle['points'][0][1]-.02, .02, .04)
        masked, _ = find_circle(detector.masks, model, on_grass=True, boxes=[box])
        self.assertFalse(any(box[0] <= x <= box[0]+box[2] and box[1] <= y <= box[1]+box[3] for x, y in (masked or {'points': []})['points']))


class CircleCentreTests(unittest.TestCase):
    """The on-grass search keeps an ellipse only when the halfway line passes near its centre in the
    ellipse's own coordinates (a circle's centre is on the line; perspective moves its image only a little)."""

    def test_an_ellipse_beside_the_halfway_line_is_not_the_circle(self):
        import cv2
        from soccer.pitch import CIRCLE_CENTRE_TOLERANCE, circle_points
        w, h, angle = 640, 360, 60
        ridge = np.zeros((h, w), np.uint8)
        cv2.ellipse(ridge, (300, 180), (150, 22), angle, 0, 360, 1, 1)
        masks = {'w': w, 'h': h, 'ridge': ridge.astype(bool)}
        t = math.radians(angle)
        minor, major = np.array([-math.sin(t), math.cos(t)]), np.array([math.cos(t), math.sin(t)])
        found = {}
        for offset in (0, 40):   # along the major axis: 0 and .27 of the semi-major axis
            c = np.array([300, 180])+offset*major
            a, b = c-120*minor, c+120*minor
            halfway = {'a': (a[0]/w, a[1]/h), 'b': (b[0]/w, b[1]/h), 'support': .5, 'kind': 'v'}
            found[offset] = (circle_points(masks, halfway, []) is not None,
                             circle_points(masks, halfway, [], centre_tolerance=CIRCLE_CENTRE_TOLERANCE) is not None)
        self.assertEqual(found[0], (True, True))
        self.assertEqual(found[40], (True, False))


class GateTests(unittest.TestCase):
    """Runs with keyframes keep the shipped construction; the tracker routes by auto_landmarks."""

    def test_the_shipped_route_is_unchanged_without_allow_pinhole(self):
        circle, halfway = exact_evidence(camera(*CAMERAS['broadcast-pan'])[0])
        result = midfield.solve(circle, halfway, [], W, H, SIZE)
        self.assertNotIn('H', result)
        self.assertEqual(result['reason'], 'the circle and the halfway line alone leave the pitch width unresolved: no touchline')

    def test_the_tracker_takes_the_pinhole_route_only_without_keyframes(self):
        from soccer.local import BotSortTracker
        from soccer.tracker import SoccerTracker, config_from
        # A pole view (no near touchline: the image-plane region is not HIGH, so the midfield anchor is tried).
        P, K, R, t = camera(*CAMERAS['pole-left'])
        frame = broadcast.render(P, stands=False)
        routes = {}
        for auto in (False, True):
            tracker = SoccerTracker(W, H, config_from({}), BotSortTracker(5), None)
            tracker.auto_landmarks = auto
            tracker.step(frame, 0.0, np.zeros((0, 6), np.float32))
            routes[auto] = (tracker.midfield or {}).get('route'), [a.source for a in tracker.anchors]
        self.assertIsNone(routes[False][0])
        self.assertEqual(routes[True][0], 'pinhole')
        self.assertEqual(routes[True][1], ['auto-midfield'])

    def test_with_keyframes_the_tracker_runs_the_shipped_search_and_solver(self):
        # auto_landmarks False: the shipped circle search and midfield.solve without allow_pinhole, on the same
        # inputs, decide what the tracker records (state, anchors, events), and no cap or fitted lines are kept.
        from soccer.local import BotSortTracker
        from soccer.tracker import SoccerTracker, config_from
        P = camera(*CAMERAS['pole-left'])[0]
        frame = broadcast.render(P, stands=False)
        with mock.patch.object(midfield, 'solve', wraps=midfield.solve) as solve:
            tracker = SoccerTracker(W, H, config_from({}), BotSortTracker(5), None)
            tracker.step(frame, 0.0, np.zeros((0, 6), np.float32))
        self.assertEqual(solve.call_count, 1)
        self.assertFalse(solve.call_args.kwargs.get('allow_pinhole', False))
        circle, halfway = find_circle(tracker.pitch_detector.masks, tracker.analysed)
        ellipse = midfield.ellipse_from_points([(x*W, y*H) for x, y in circle['points']])
        expected = midfield.solve(circle, halfway, midfield.touchline_candidates(tracker.analysed, ellipse, W, H), W, H, tracker.keyframes.size,
                                  tracker.pitch_detector.masks)
        self.assertNotIn('H', expected)
        self.assertEqual(tracker.midfield, {'time': 0.0, 'taken': False, 'reason': expected['reason']})
        self.assertEqual(tracker.anchors, [])
        self.assertEqual([m for _, k, m in tracker.keyframes.pitch.events if k == 'midfield'],
                         [f'centre circle seen ({circle["coverage"]:.0%} of the arc) but not used as an anchor: {expected["reason"]}'])
        self.assertIsNone(tracker.keyframes.pitch.cap)

    def test_the_automatic_anchor_carries_the_lines_it_was_fitted_to(self):
        from soccer.local import BotSortTracker
        from soccer.tracker import SoccerTracker, config_from
        tracker = SoccerTracker(W, H, config_from({}), BotSortTracker(5), None)
        tracker.auto_landmarks = True
        tracker.step(broadcast.render(camera(*CAMERAS['pole-left'])[0], stands=False), 0.0, np.zeros((0, 6), np.float32))
        model = tracker.keyframes.pitch
        self.assertIsNotNone(model.cap)
        self.assertEqual(model.fitted, set(tracker.midfield['evidence'])-{field.CENTER_CIRCLE})
        self.assertIn('halfway line', model.fitted)


if __name__ == '__main__':
    unittest.main()
