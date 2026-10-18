"""Low sideline cameras: calibration observability separate from temporal confidence, the vanishing-point
guard, pans that are not cuts, the offline registration passes (forward and backward from validated
anchors, retroactive remapping, mapping-loss warnings, ball bridging) and the automatic midfield anchor
from the centre circle, the halfway line and a touchline."""
import math
import unittest

import cv2
import numpy as np

import broadcast
from scene import H, W
from soccer import field, midfield, registration
from soccer.camera import COARSE_W, CameraMotion
from soccer.pitch import FieldFilter, PitchDetector, PitchModel, assess_confidence, converging_end, find_circle, zone_of
from soccer.pitch_model import MEDIUM, MIN_ANCHOR, PitchTracker, combine

SIZE = field.PitchSize()
MARKS = field.landmarks(SIZE)


def low_camera():
    """A low camera on the near touchline side near the halfway line: the near touchline is below the
    frame, the far touchline is a thin line a third of the way down, the touchlines converge to the right."""
    src = np.float32([[20, 0], [95, 0], [75, 68], [35, 68]])
    dst = np.float32([[0, .34*H], [.62*W, .30*H], [1.6*W, 1.5*H], [-.2*W, 1.25*H]])
    return cv2.getPerspectiveTransform(src, dst).astype(np.float64)


def end_camera():
    """A low camera behind one corner looking along the pitch: the far goal line shrinks to a few pixels
    where the touchlines meet (the horizon), the near goal line is below the frame."""
    src = np.float32([[0, 0], [105, 0], [105, 68], [0, 68]])
    dst = np.float32([[-.3*W, 1.4*H], [.52*W, .300*H], [.528*W, .303*H], [1.5*W, 1.6*H]])
    return cv2.getPerspectiveTransform(src, dst).astype(np.float64)


def apex_camera():
    """A low camera near a corner: the touchlines converge inside the frame, on the right."""
    src = np.float32([[20, 0], [95, 0], [75, 68], [35, 68]])
    dst = np.float32([[0, .34*H], [.70*W, .30*H], [.80*W, .33*H], [-.2*W, 1.25*H]])
    return cv2.getPerspectiveTransform(src, dst).astype(np.float64)


def landmarks(P, names):
    return [{'name': n, 'x': float(x), 'y': float(y)} for n, (x, y) in zip(names, field.project(P, [MARKS[n] for n in names]))]


def marking_lines(P, names):
    catalogue = {name: (a, b) for name, a, b in field.lines(SIZE)}
    out = []
    for name in names:
        a, b = catalogue[name]
        qa, qb = field.project(P, [a, b])
        out.append({'name': name, 'a': (float(qa[0]), float(qa[1])), 'b': (float(qb[0]), float(qb[1]))})
    return out


def circle_pixels(P, n=90):
    cx, cy = field.circle_centre(SIZE)
    pts = field.project(P, [(cx+field.RADIUS*math.cos(a), cy+field.RADIUS*math.sin(a)) for a in np.linspace(0, 2*math.pi, n)])
    return [(float(x), float(y)) for x, y in pts if 0 <= x < W and 0 <= y < H]


def spot_error(Ha, Hb):
    spots = [(11, 34), (94, 34), (52.5, 34), (52.5, 0), (0, 34), (105, 34)]
    return float(np.max(np.hypot(*(field.project(Ha, spots)-field.project(Hb, spots)).T)))


class ObservabilityTests(unittest.TestCase):
    """Geometric certainty comes from the evidence alone, never from how smoothly a model was carried."""

    def setUp(self):
        self.P = broadcast.camera()

    def test_spread_landmarks_beat_landmarks_along_one_line(self):
        spread = field.observability({'points': landmarks(self.P, ['corner-far-left', 'corner-far-right', 'halfway-far', 'halfway-near',
                                                                   'centre-spot', 'left-penalty-box-front-far', 'right-penalty-box-front-near'])}, W, H, SIZE, rms_m=.2)
        one_line = field.observability({'points': landmarks(self.P, ['halfway-far', 'centre-circle-far', 'centre-spot', 'centre-circle-near', 'halfway-near'])}, W, H, SIZE, rms_m=.2)
        self.assertGreater(spread['score'], .7, spread)
        self.assertLess(one_line['score'], .15, one_line)
        self.assertEqual(one_line['axes'], 0.0)

    def test_touchlines_alone_do_not_fix_the_pitch(self):
        touchlines = field.observability({'lines': marking_lines(self.P, ['far touchline', 'near touchline'])}, W, H, SIZE)
        both = field.observability({'lines': marking_lines(self.P, ['far touchline', 'near touchline', 'halfway line', 'left penalty area front'])}, W, H, SIZE)
        self.assertLessEqual(touchlines['score'], .1, touchlines)
        self.assertGreater(both['score'], .7, both)
        self.assertEqual(touchlines['transverse'], [])

    def test_centre_circle_with_the_halfway_line_adds_what_touchlines_lack(self):
        lines = marking_lines(self.P, ['far touchline', 'halfway line'])
        without = field.observability({'lines': lines}, W, H, SIZE)
        with_circle = field.observability({'lines': lines, 'circle': circle_pixels(self.P), 'H': self.P}, W, H, SIZE)
        self.assertGreater(with_circle['score'], without['score']+.3, (without, with_circle))
        self.assertGreaterEqual(with_circle['circleCoverage'], .9)

    def test_a_flat_jacobian_direction_lowers_the_score(self):
        good = np.eye(9)
        flat = np.diag([1, 1, 1, 1, 1, 1, 1, 1e-6, 1e-6])   # the gauge direction is always flat; a second one is unobserved
        lines = marking_lines(self.P, ['far touchline', 'near touchline', 'halfway line', 'left penalty area front'])
        self.assertGreater(field.observability({'lines': lines}, W, H, SIZE, jac=good)['score'], field.observability({'lines': lines}, W, H, SIZE, jac=flat)['score']+.5)

    def test_temporal_and_geometric_confidence_are_combined_not_confused(self):
        self.assertAlmostEqual(combine(1.0, 1.0), 1.0)
        self.assertLess(combine(1.0, .1), MEDIUM)         # smoothly carried but unconstrained: never confident
        self.assertLess(combine(.1, 1.0), combine(.9, 1.0))
        tracker = PitchTracker(W, H, SIZE)
        self.assertFalse(tracker.anchor(np.linalg.inv(self.P), 0.0, .1, observability=MIN_ANCHOR/2))
        self.assertIsNone(tracker.H)
        self.assertTrue(any(kind == 'refused' for _, kind, _ in tracker.events))
        self.assertTrue(tracker.anchor(np.linalg.inv(self.P), 0.0, .1, observability=.5))
        self.assertEqual(tracker.observability, .5)
        self.assertEqual(tracker.temporal, 1.0)
        self.assertLess(tracker.confidence, .7)


class VanishingPointTests(unittest.TestCase):
    """Two touchlines meet at a vanishing point, never at a corner of the pitch."""

    def setUp(self):
        self.far = {'a': (0.0, .30), 'b': (1.0, .36), 'side': 'far', 'support': .6, 'age': 0}
        self.near = {'a': (0.0, .95), 'b': (1.0, .30), 'side': 'near', 'support': .7, 'age': 0}   # meets the far line at x = .915

    def test_converging_touchlines_mark_the_side_unknown_unless_a_goal_line_was_seen(self):
        self.assertEqual(converging_end({'far': self.far, 'near': self.near}, []), 'right')
        self.assertIsNone(converging_end({'far': self.far, 'near': self.near}, [{'side': 'right', 'kind': 'goal-line'}]))
        parallel = {'far': self.far, 'near': {'a': (0.0, .95), 'b': (1.0, .98), 'side': 'near', 'support': .7, 'age': 0}}
        self.assertIsNone(converging_end(parallel, []))

    def test_a_region_ending_at_the_vanishing_point_is_never_confident_and_rejects_nobody_there(self):
        apex_x = .65/.71           # where the two lines meet, inside the frame
        polygon = [(0.0, .30), (apex_x, .30+.06*apex_x), (0.0, .95)]
        model = PitchModel(0.0, True, 16/9, polygon=polygon, lines=[self.far, self.near], coverage=.5, source='auto', vanishing='right')
        confidence, evidence = assess_confidence(model)
        self.assertLess(confidence, MEDIUM, evidence)
        self.assertTrue(any('vanishing' in e for e in evidence), evidence)
        self.assertIn('vanishing point', model.sources()['right'])
        config = FieldFilter(True, .02, .005, .05)
        zone, _ = zone_of(model, [.95, .26, .02, .1], config)    # feet just beyond the apex: not measured, so unknown
        self.assertEqual(zone, 'unknown')
        zone, _ = zone_of(model, [.3, .2, .02, .06], config)      # feet above the far touchline: really outside
        self.assertEqual(zone, 'outside')

    def test_the_tracked_model_reports_a_goal_line_at_the_horizon(self):
        # An under-constrained carry stretched the pitch along its length: the right goal line now projects
        # onto the touchlines' vanishing point. Such an end is reported as the horizon, not as a pitch end.
        P = field.orient(apex_camera() @ np.diag([30.0, 1.0, 1.0]), W, H)
        far, near = field.project(P, [(105, 0), (105, 68)])
        self.assertTrue(0 < far[0] < W and 0 < far[1] < H, far)          # the apex is inside the frame
        self.assertTrue(field.at_horizon(P, SIZE, W, H, 'right'))
        self.assertFalse(field.at_horizon(low_camera(), SIZE, W, H, 'right'))   # a side camera sees the goal line as a line
        tracker = PitchTracker(W, H, SIZE)
        self.assertTrue(tracker.anchor(np.linalg.inv(low_camera()), 0.0, .1, observability=.6))
        tracker.H = P
        tracker.refine([], .1)
        self.assertTrue(any(e.startswith('right goal line: AT THE HORIZON') for e in tracker.evidence), tracker.evidence)
        self.assertEqual(tracker.sides()['right'], 'horizon')


class PanTests(unittest.TestCase):
    def test_a_fast_pan_over_flat_grass_is_measured_coarsely_and_is_not_a_cut(self):
        rng = np.random.default_rng(3)
        wide = np.full((360, 1000), 110, np.uint8)
        wide[:] = (110+rng.normal(0, 6, size=wide.shape)).clip(0, 255)   # grass texture without trackable corners
        wide = cv2.GaussianBlur(wide, (5, 5), 0)
        before, after = wide[:, 0:640], wide[:, 120:760]                   # the camera panned 120 of 640 px
        motion = CameraMotion()
        small = lambda g: cv2.resize(g, (COARSE_W, round(g.shape[0]*COARSE_W/g.shape[1])), interpolation=cv2.INTER_AREA)
        motion.coarse = small(before)
        shift = motion.coarse_shift(small(after))
        self.assertIsNotNone(shift)
        self.assertAlmostEqual(shift[0]*640/COARSE_W, -120, delta=4)   # the picture moved left by the pan
        self.assertAlmostEqual(shift[1], 0, delta=2)
        motion.coarse = small(before)
        self.assertIsNone(motion.coarse_shift(small((rng.integers(0, 255, size=after.shape)).astype(np.uint8))))   # a different picture: no translation explains it


def motion_log(n, shift, cut_at=None, quality='flow'):
    return [{'t': .04*i, 'dt': .04, 'M': shift, 'reliable': True, 'cut': i == cut_at, 'quality': quality} for i in range(n)]


class TimelineTests(unittest.TestCase):
    """Pass 2: anchors carried both ways and blended; pass 3: positions remapped from the result."""

    def setUp(self):
        self.P = broadcast.camera()
        self.shift = np.array([[1, 0, 5], [0, 1, 0], [0, 0, 1]], float)
        names = ['corner-far-left', 'corner-far-right', 'halfway-far', 'halfway-near', 'centre-spot', 'left-penalty-box-front-far']
        self.obs = field.observability({'points': landmarks(self.P, names)}, W, H, SIZE, rms_m=.1)

    def anchor(self, index, log, observability=None, source='landmarks'):
        Hm = np.linalg.matrix_power(self.shift, index) @ self.P      # the true camera at that frame of this motion chain
        return registration.Anchor(index, log[index]['t'], Hm, self.obs['score'] if observability is None else observability, source, self.obs)

    def centre_at(self, entry, index):
        true = np.linalg.matrix_power(self.shift, index) @ self.P
        return field.to_pitch(entry['H'], SIZE, field.project(true, [MARKS['centre-spot']])[0])

    def test_one_anchor_is_carried_backwards_to_the_cut_and_forwards_to_the_end(self):
        log = motion_log(40, self.shift, cut_at=10)
        timeline = registration.build_timeline(log, [self.anchor(25, log)], W, H, SIZE)
        self.assertTrue(all(e['H'] is None and e['source'] == 'none' for e in timeline[:10]))
        self.assertEqual(timeline[25]['source'], 'landmarks')
        self.assertEqual((timeline[12]['source'], timeline[12]['direction']), ('carried', 'backward'))
        self.assertEqual((timeline[33]['source'], timeline[33]['direction']), ('carried', 'forward'))
        for i in (10, 18, 25, 31, 39):
            x, y = self.centre_at(timeline[i], i)
            self.assertAlmostEqual(x, .5, places=3)
            self.assertAlmostEqual(y, .5, places=3)
        self.assertGreater(timeline[25]['confidence'], timeline[10]['confidence'])
        self.assertEqual(timeline[10]['anchor'], 25)
        self.assertAlmostEqual(timeline[10]['age'], .6, places=3)

    def test_a_weak_anchor_starts_low_and_a_refused_one_never_exists(self):
        log = motion_log(20, self.shift)
        weak = registration.build_timeline(log, [self.anchor(10, log, observability=.2)], W, H, SIZE)
        strong = registration.build_timeline(log, [self.anchor(10, log, observability=.9)], W, H, SIZE)
        self.assertLess(weak[10]['confidence'], MEDIUM+.1)
        self.assertGreater(strong[10]['confidence'], .9)
        self.assertEqual(weak[10]['observability'], .2)
        tracker = PitchTracker(W, H, SIZE)
        self.assertFalse(tracker.anchor(np.linalg.inv(self.P), 0.0, observability=.05))

    def test_two_anchors_are_blended_and_their_disagreement_lowers_the_confidence(self):
        log = motion_log(40, self.shift)
        agree = registration.build_timeline(log, [self.anchor(5, log), self.anchor(35, log)], W, H, SIZE)
        wrong = registration.Anchor(35, log[35]['t'], np.array([[1, 0, 60], [0, 1, 0], [0, 0, 1]], float) @ np.linalg.matrix_power(self.shift, 35) @ self.P,
                                    .6, 'paint', {})
        disagree = registration.build_timeline(log, [self.anchor(5, log), wrong], W, H, SIZE)
        self.assertEqual((agree[20]['source'], agree[20]['direction']), ('interpolated', 'both'))
        self.assertLess(agree[20]['disagreement'], .3)
        self.assertGreater(disagree[20]['disagreement'], 2.0)
        self.assertLess(disagree[20]['confidence'], agree[20]['confidence']-.15)
        x, _ = self.centre_at(agree[20], 20)
        self.assertAlmostEqual(x, .5, places=2)

    def test_coarse_motion_fades_faster_and_no_motion_stops_the_chain(self):
        flow = registration.build_timeline(motion_log(60, self.shift), [self.anchor(0, motion_log(60, self.shift))], W, H, SIZE)
        coarse = registration.build_timeline(motion_log(60, self.shift, quality='coarse'), [self.anchor(0, motion_log(60, self.shift))], W, H, SIZE)
        self.assertLess(coarse[59]['temporal'], flow[59]['temporal'])
        log = motion_log(20, self.shift)
        for e in log[8:]:
            e['M'], e['reliable'], e['quality'] = None, False, 'none'
        stopped = registration.build_timeline(log, [self.anchor(0, log)], W, H, SIZE)
        self.assertIsNotNone(stopped[7]['H'])
        self.assertIsNone(stopped[15]['H'])

    def test_tracks_are_remapped_or_left_unresolved_never_dropped(self):
        log = motion_log(30, self.shift, cut_at=10)
        timeline = registration.build_timeline(log, [self.anchor(20, log)], W, H, SIZE)
        foot = field.project(self.P, [MARKS['right-penalty-spot']])[0]
        frames = []
        for i in range(30):
            fx, fy = (np.linalg.matrix_power(self.shift, i) @ np.array([foot[0], foot[1], 1.0]))[:2]
            box = [fx/W-.015, fy/H-.1, .03, .1]
            frames.append({'time': log[i]['t'], 'people': [{'track': k, 'state': 'confirmed', 'box': box} for k in range(7)],
                           'ball': {'state': 'TRACKED', 'center': [fx/W, fy/H-.01], 'box': [fx/W-.01, fy/H-.02, .02, .02], 'track': 1, 'confidence': .8}})
        metrics = registration.remap_frames(frames, timeline, W, H, SIZE)
        self.assertEqual(len(frames[5]['people']), 7)                       # nothing deleted where the registration is unknown
        self.assertEqual(frames[5]['people'][0]['pitchStatus'], 'unresolved')
        self.assertFalse(frames[5]['calibrated'])
        p = frames[25]['people'][0]
        self.assertAlmostEqual(p['pitch'][0], 94/105, places=2)
        self.assertAlmostEqual(p['pitch'][1], .5, places=2)
        self.assertGreater(p['pitchConfidence'], .4)
        self.assertEqual(p['pitchSource'], 'carried')
        self.assertEqual(frames[20]['pitchCoordinates'], 'calibration')
        self.assertEqual(frames[25]['pitchCoordinates'], 'predicted')
        self.assertIn('pitch', frames[25]['ball'])
        self.assertEqual(metrics['mappingLossFrames'], 10)
        self.assertEqual(metrics['medianMappedShare'], 1.0)
        far = registration.position(self.P, SIZE, tuple(field.project(self.P, [(52.5, 2.0)])[0]), 1.0)[1]
        near = registration.position(self.P, SIZE, tuple(field.project(self.P, [(52.5, 66.0)])[0]), 1.0)[1]
        self.assertLess(far, near)                                           # a few pixels are worth more metres far away

    def test_mapping_loss_is_a_calibration_warning_not_lost_players(self):
        frames = [{'time': .04*i, 'people': [{'track': k, 'state': 'confirmed', 'box': [.5, .5, .03, .1]} for k in range(8)], 'ball': {'state': 'UNKNOWN'}} for i in range(60)]
        timeline = [{'H': None, 'confidence': 0.0, 'temporal': 0.0, 'observability': 0.0, 'source': 'none', 'direction': None, 'anchor': None, 'age': None, 'disagreement': None}]*60
        registration.remap_frames(frames, timeline, W, H, SIZE)
        events = registration.warnings(frames)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]['kind'], 'calibration')
        self.assertIn('MAPPING LOSS', events[0]['message'])
        self.assertIn('image tracks are intact', events[0]['message'])
        self.assertTrue(all(len(f['people']) == 8 for f in frames))


class BallBridgingTests(unittest.TestCase):
    def frames(self, gap_frames, jump=.02):
        out = []
        for i in range(30):
            t = .04*i
            if 10 < i < 10+gap_frames+1:
                out.append({'time': t, 'ball': {'state': 'UNKNOWN', 'phase': 'RECOVERING', 'confidence': .1}})
            else:
                x = .3+jump*i
                out.append({'time': t, 'ball': {'state': 'TRACKED', 'center': [x, .5], 'box': [x-.01, .49, .02, .02], 'track': 1, 'confidence': .8}})
        return out

    def test_short_plausible_gaps_are_bridged_and_labelled(self):
        frames = self.frames(5)
        self.assertEqual(registration.bridge_ball(frames, W), 5)
        self.assertEqual(frames[13]['ball']['source'], 'INTERPOLATED')
        self.assertEqual(frames[13]['ball']['state'], 'MISSING')
        self.assertTrue(frames[13]['ball']['bridged'])
        self.assertAlmostEqual(frames[13]['ball']['center'][0], .3+.02*13, places=4)
        self.assertEqual(frames[5]['ball']['source'], 'DETECTED')

    def test_long_or_implausible_gaps_stay_unknown(self):
        frames = self.frames(16)                      # .64 s: longer than BALL_GAP
        self.assertEqual(registration.bridge_ball(frames, W), 0)
        self.assertEqual(frames[15]['ball']['state'], 'UNKNOWN')
        frames = self.frames(5, jump=.05)            # 1.25 frame widths per second across the gap: not the same ball
        self.assertEqual(registration.bridge_ball(frames, W), 0)


class MidfieldAnchorTests(unittest.TestCase):
    """The centre circle, the halfway line and one touchline register a low midfield view."""

    def setUp(self):
        self.P = low_camera()

    def test_pole_polar_construction_recovers_the_camera_from_exact_evidence(self):
        catalogue = {name: (a, b) for name, a, b in field.lines(SIZE)}
        seg = lambda name: {'a': tuple(field.project(self.P, [catalogue[name][0]])[0]/[W, H]), 'b': tuple(field.project(self.P, [catalogue[name][1]])[0]/[W, H]), 'support': .5, 'kind': 'v'}
        circle = {'points': [(x/W, y/H) for x, y in circle_pixels(self.P)]}
        result = midfield.solve(circle, seg('halfway line'), [('far', seg('far touchline'))], W, H, SIZE)
        self.assertIn('H', result, result.get('reason'))
        self.assertLess(spot_error(result['H'], self.P), 1.0)
        self.assertGreaterEqual(result['observability'], midfield.MIN_OBSERVABILITY)
        self.assertEqual(sorted(result['details']['evidence']), ['centre circle', 'far touchline', 'halfway line'])
        self.assertIn('centre-spot', result['details']['derived'])
        without = midfield.solve(circle, seg('halfway line'), [], W, H, SIZE)
        self.assertNotIn('H', without)
        self.assertIn('touchline', without['reason'])

    def test_detected_paint_of_a_low_view_anchors_the_model_and_the_touchline_apex_is_not_a_corner(self):
        img = broadcast.render(self.P, stands=False)
        detector = PitchDetector()
        model = detector.analyze(img, 0.0)
        circle, halfway = find_circle(detector.masks, model)
        self.assertIsNotNone(circle, 'the centre circle paint was not found')
        self.assertGreaterEqual(circle['coverage'], .5)
        ellipse = midfield.ellipse_from_points([(x*W, y*H) for x, y in circle['points']])
        touchlines = midfield.touchline_candidates(model, ellipse, W, H)
        self.assertEqual([s for s, _ in touchlines], ['far'])
        result = midfield.solve(circle, halfway, touchlines, W, H, SIZE, detector.masks)
        self.assertIn('H', result, result.get('reason'))
        self.assertLess(spot_error(result['H'], self.P), 6.0)
        self.assertGreaterEqual(min(result['details']['paint'].values()), .8, result['details']['paint'])
        tracker = PitchTracker(W, H, SIZE)
        self.assertTrue(tracker.anchor(np.linalg.inv(result['H']), 0.0, observability=result['observability'], details=result['details'], source='auto-midfield'))
        self.assertEqual(tracker.source, 'auto-midfield')
        self.assertIn('CENTRE CIRCLE', tracker.status_text())
        self.assertNotEqual(tracker.level, 'LOW')
        # The pitch end comes from the canonical length, not from where the touchlines meet in the image.
        far, near = field.project(result['H'], [(20, 0), (95, 0)]), field.project(result['H'], [(40, 68), (70, 68)])   # points in front of the camera
        d1, d2 = far[1]-far[0], near[1]-near[0]
        den = d1[0]*d2[1]-d1[1]*d2[0]
        t = ((near[0]-far[0])[0]*d2[1]-(near[0]-far[0])[1]*d2[0])/den
        apex_x = far[0][0]+t*d1[0]
        goal_x = field.project(result['H'], [(105, 34)])[0][0]
        self.assertGreater(abs(apex_x-goal_x), 150, (apex_x, goal_x))
        x, y = tracker.to_pitch(tuple(field.project(self.P, [(70, 40)])[0]))
        self.assertAlmostEqual(x, 70/105, delta=.02)
        self.assertAlmostEqual(y, 40/68, delta=.03)

    def test_a_straight_line_is_never_mistaken_for_the_circle(self):
        img = broadcast.render(self.P, stands=False)
        detector = PitchDetector()
        model = detector.analyze(img, 0.0)
        from soccer.pitch import circle_points
        touchline = max((s for s in model.markings if s['kind'] == 'h'), key=lambda s: s['support'])
        self.assertIsNone(circle_points(detector.masks, touchline, model.markings))


class HomographyJsonTests(unittest.TestCase):
    """The stored per-frame homography keeps the orientation the registration gave it."""

    def test_a_far_left_corner_behind_the_camera_keeps_the_orientation(self):
        # A camera at the near touchline beyond the halfway line looking at the right end: the far-left corner
        # (the world origin, whose depth H[2, 2] is) lies behind it.
        C = np.array([70.0, 72.0, -6.0])
        z = np.array([105.0, 55.0, 0.0])-C
        z /= np.linalg.norm(z)
        y = np.array([0.0, 0.0, 1.0])-z[2]*z
        y /= np.linalg.norm(y)
        R = np.vstack([np.cross(y, z), y, z])
        K = np.array([[900.0, 0, W/2], [0, 900.0, H/2], [0, 0, 1]])
        P = field.orient(K @ np.c_[R[:, 0], R[:, 1], -R @ C], W, H)
        self.assertLess(field.depth(P, [(0, 0)])[0], 0)
        self.assertLess(P[2, 2], 0)
        stored = np.array(registration.homography_json(P)).reshape(3, 3)
        self.assertAlmostEqual(abs(stored[2, 2]), 1.0)
        self.assertGreater(field.image_depth(stored, [(W/2, .95*H)])[0], 0)
        self.assertLess(spot_error(stored, P), 1e-6)
        upright = field.orient(broadcast.camera(), W, H)   # the usual side view: H[2, 2] > 0, stored as before
        self.assertAlmostEqual(np.array(registration.homography_json(upright)).reshape(3, 3)[2, 2], 1.0)


class PositionUncertaintyTests(unittest.TestCase):
    """Pass 3 (Phase 1): every position carries a sigma in metres; nothing is written below the confidence
    floor or above the horizon; the forward/backward disagreement is measured over the visible pitch."""

    def setUp(self):
        self.P = broadcast.camera()

    def test_sigma_grows_with_distance_and_with_a_clipped_box(self):
        P = low_camera()                                                  # a low side camera: strong perspective
        near_px = tuple(field.project(P, [(52.5, 50.0)])[0])
        far_px = tuple(field.project(P, [(52.5, 4.0)])[0])
        near = registration.position(P, SIZE, near_px, 1.0)
        far = registration.position(P, SIZE, far_px, 1.0)
        self.assertIsNotNone(near[2])
        self.assertGreater(far[2][1], 2*near[2][1])                       # far pixels are worth more metres across the pitch
        self.assertGreater(far[2][0], near[2][0])
        clipped = registration.position(P, SIZE, near_px, 1.0, (registration.FOOT_PIXELS, 30.0))
        self.assertGreater(clipped[2][1], 3*near[2][1])                   # a box cut by the frame's bottom: the feet are below it
        self.assertGreaterEqual(clipped[2][0], near[2][0])               # (a vertical pixel error also moves the point along the pitch)
        disagreeing = registration.position(P, SIZE, near_px, 1.0, None, 4.0)
        self.assertGreaterEqual(disagreeing[2][0], 4.0)                   # the registration's own disagreement adds in quadrature
        self.assertEqual(near[1], disagreeing[1])                         # the confidence is unchanged (compatibility)

    def test_positions_need_the_confidence_floor_and_stop_at_the_horizon(self):
        P = end_camera()
        grid = [(x, y) for x in np.linspace(5, 100, 20) for y in np.linspace(5, 63, 12)]
        spot = max(grid, key=lambda p: field.px_per_metre(P, p))             # the nearest spot in view: placed with a confidence above zero
        self.assertGreater(field.px_per_metre(P, spot), .7)
        foot = field.project(P, [spot])[0]
        above = None
        for y in np.linspace(0, .29*H, 30):
            if field.image_depth(P, [(W/2, y)])[0] <= 0:
                above = y
        self.assertIsNotNone(above, 'the end camera must have its horizon inside the frame')
        people = [{'track': 1, 'state': 'confirmed', 'box': [foot[0]/W-.01, foot[1]/H-.1, .02, .1]},
                  {'track': 2, 'state': 'confirmed', 'box': [.5-.01, above/H-.05, .02, .05]},
                  {'track': 3, 'state': 'confirmed', 'box': [.49, .9, .02, .1]},      # cut by the bottom edge (feet below it)
                  {'track': 4, 'state': 'confirmed', 'box': [.49, .89, .02, .1]}]     # the same person 7 px higher: whole
        entry = lambda c: {'H': P, 'confidence': c, 'temporal': c, 'observability': .8, 'source': 'carried', 'direction': 'forward', 'anchor': 0, 'age': 1.0, 'disagreement': 0.0}
        frames = [{'time': 0.0, 'people': [dict(p) for p in people], 'ball': {'state': 'UNKNOWN'}},
                  {'time': .04, 'people': [dict(p) for p in people], 'ball': {'state': 'UNKNOWN'}}]
        registration.remap_frames(frames, [entry(.6), entry(.1)], W, H, SIZE)
        ok, horizon, clipped, whole = frames[0]['people']
        self.assertIn('pitch', ok)
        self.assertIn('pitchSigma', ok)
        self.assertEqual(horizon.get('pitchStatus'), 'horizon')
        self.assertNotIn('pitch', horizon)
        self.assertTrue(clipped.get('pitchClipped'))
        self.assertFalse(whole.get('pitchClipped'))
        self.assertGreaterEqual(max(clipped['pitchSigma']), max(whole['pitchSigma']))   # (the size of the effect is tested above on a low camera)
        self.assertFalse(frames[1]['calibrated'])                        # below the floor: no positions at all
        self.assertIsNone(frames[1]['pitchCoordinates'])
        self.assertTrue(all(p.get('pitchStatus') == 'unresolved' and 'pitch' not in p for p in frames[1]['people']))

    def test_disagreement_is_measured_over_the_visible_pitch_and_is_finite_at_the_horizon(self):
        shifted = np.array([[1, 0, 60], [0, 1, 0], [0, 0, 1]], float) @ self.P
        self.assertLess(registration._disagreement(self.P, self.P, W, H, SIZE), 1e-6)
        self.assertGreater(registration._disagreement(self.P, shifted, W, H, SIZE), 2.0)
        pts = registration.ground_points(self.P, shifted, W, H, SIZE)
        self.assertGreaterEqual(len(pts), 10)
        self.assertGreater(float(np.ptp(pts[:, 1])), .4*H)                 # far and near rows both count
        end = end_camera()
        d = registration._disagreement(end, np.array([[1, 0, 20], [0, 1, 0], [0, 0, 1]], float) @ end, W, H, SIZE)
        self.assertTrue(math.isfinite(d))
        self.assertGreater(d, 0.0)


if __name__ == '__main__':
    unittest.main()
