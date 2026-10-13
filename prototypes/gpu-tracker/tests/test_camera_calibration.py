import unittest

import cv2
import numpy as np

import broadcast
from scene import H, W, pitch_frame
from soccer import calibration, field
from soccer.camera import WORK_W, AnchorViews, CameraMotion, CameraState, Keyframes
from soccer.geometry import transform_points
from soccer.pitch_model import PitchTracker


def textured(dx=0, seed=0):
    """A textured frame shifted left by dx pixels (camera pans right)."""
    rng = np.random.default_rng(seed)
    big = cv2.GaussianBlur(rng.integers(0, 255, size=(H, W+200, 3), dtype=np.uint8), (5, 5), 0)
    return np.ascontiguousarray(big[:, 50+dx:50+dx+W])


class CameraTests(unittest.TestCase):
    def test_pan_is_measured_and_compensated(self):
        cam, state = CameraMotion(), CameraState(W, H)
        M, reliable, cut = cam.estimate(textured(0), [])
        state.update(M, reliable, cut)
        M, reliable, cut = cam.estimate(textured(12), [])
        self.assertTrue(reliable)
        self.assertFalse(cut)
        self.assertAlmostEqual(M[0, 2], -12, delta=1.5)
        state.update(M, reliable, cut)
        # A player standing still on the pitch appears 12 px further left; stabilized position is unchanged.
        moved = [.5-12/W, .4, .02, .1]
        self.assertAlmostEqual(state.stabilize(moved)[0], .5+.01, delta=.002)
        self.assertEqual(state.segment, 0)

    def test_cut_starts_a_new_segment(self):
        cam, state = CameraMotion(), CameraState(W, H)
        for frame in (pitch_frame(), np.full((H, W, 3), 230, np.uint8)):
            M, reliable, cut = cam.estimate(frame, [])
            state.update(M, reliable, cut)
        self.assertTrue(cut)
        self.assertEqual(state.segment, 1)

    def test_boundary_keyframe_follows_the_camera(self):
        keys = Keyframes([{'time': 1.0, 'points': [[.1, .3], [.9, .3], [.9, .9], [.1, .9]]}], [], W, H)
        keys.update(0.5, np.eye(3), True)
        self.assertIsNone(keys.boundary())
        keys.update(1.0, np.eye(3), True)
        shift = np.array([[1, 0, -64], [0, 1, 0], [0, 0, 1]], float)
        keys.update(1.04, shift, True)
        self.assertAlmostEqual(keys.boundary()[0][0], .05, places=3)
        keys.update(1.08, np.eye(3), False)
        self.assertIsNone(keys.boundary())
        self.assertIn('lost', keys.warning)


class CalibrationTests(unittest.TestCase):
    def setUp(self):
        # A synthetic camera: pitch metres -> image pixels.
        src = np.float64([[0, 0], [105, 0], [105, 68], [0, 68]])
        dst = np.float64([[300, 200], [980, 200], [1250, 650], [30, 650]])
        self.P = cv2.getPerspectiveTransform(src.astype(np.float32), dst.astype(np.float32)).astype(np.float64)

    def image(self, name):
        x, y = transform_points(self.P, [calibration.LANDMARKS[name]])[0]
        return {'name': name, 'x': x/W, 'y': y/H}

    def test_landmarks_give_pitch_coordinates(self):
        names = ['halfway-far', 'halfway-near', 'left-penalty-box-front-far', 'left-penalty-box-front-near', 'centre-spot']
        Hm, rms = calibration.solve([self.image(n) for n in names], W, H)
        self.assertLess(rms, .05)
        spot = transform_points(self.P, [calibration.LANDMARKS['right-penalty-spot']])[0]
        x, y = calibration.to_pitch(Hm, spot)
        self.assertAlmostEqual(x, (105-11)/105, places=3)
        self.assertAlmostEqual(y, .5, places=3)
        outline = calibration.pitch_outline(Hm, W, H)
        self.assertGreater(len(outline), 20)

    def test_bad_landmark_sets_are_refused(self):
        with self.assertRaises(ValueError):
            calibration.solve([self.image('halfway-far'), self.image('centre-spot'), self.image('halfway-near')], W, H)
        three_on_a_line = ['corner-far-left', 'halfway-far', 'corner-far-right', 'centre-spot']
        with self.assertRaises(ValueError):
            calibration.solve([self.image(n) for n in three_on_a_line], W, H)
        mirrored = [dict(self.image(a), name=b) for a, b in (('corner-far-left', 'corner-far-right'), ('corner-far-right', 'corner-far-left'),
                                                             ('corner-near-right', 'corner-near-left'), ('corner-near-left', 'corner-near-right'))]
        with self.assertRaises(ValueError):
            calibration.solve(mirrored, W, H)

    def test_calibration_follows_the_camera(self):
        names = ['corner-far-left', 'corner-far-right', 'corner-near-right', 'corner-near-left']
        keys = Keyframes([], [{'time': 0.0, 'points': [self.image(n) for n in names]}], W, H)
        keys.update(0.0, np.eye(3), True)
        shift = np.array([[1, 0, -40], [0, 1, 0], [0, 0, 1]], float)
        keys.update(.04, shift, True)
        centre = transform_points(self.P, [calibration.LANDMARKS['centre-spot']])[0]-[40, 0]
        box = [centre[0]/W-.01, centre[1]/H-.1, .02, .1]
        x, y = keys.to_pitch(box)
        self.assertAlmostEqual(x, .5, places=3)
        self.assertAlmostEqual(y, .5, places=3)


class KeyframeRobustnessTests(unittest.TestCase):
    """Phase 1: one clearly wrong click among six or more landmarks is left out; a keyframe whose landmarks
    disagree moderately still anchors, at a lower observability, instead of being dropped silently."""

    def setUp(self):
        src = np.float64([[0, 0], [105, 0], [105, 68], [0, 68]])
        dst = np.float64([[300, 200], [980, 200], [1250, 650], [30, 650]])
        self.P = cv2.getPerspectiveTransform(src.astype(np.float32), dst.astype(np.float32)).astype(np.float64)
        self.names = ['corner-far-left', 'corner-far-right', 'halfway-far', 'halfway-near', 'centre-spot', 'left-penalty-box-front-far', 'right-penalty-spot']

    def image(self, name):
        x, y = transform_points(self.P, [calibration.LANDMARKS[name]])[0]
        return {'name': name, 'x': x/W, 'y': y/H}

    def test_a_clearly_wrong_click_is_left_out_only_among_enough_landmarks(self):
        points = [self.image(n) for n in self.names]
        self.assertEqual(calibration.solve_details(points, W, H)['outliers'], [])
        bad = [dict(p) for p in points]
        bad[4]['x'] += .12
        details = calibration.solve_details(bad, W, H, max_error=4.0)
        self.assertEqual(details['outliers'], ['centre-spot'])
        self.assertLess(details['rms'], .1)
        self.assertEqual(len(details['points']), 6)
        with self.assertRaises(ValueError):                 # five landmarks: no outlier search, the error shows
            calibration.solve(bad[:5], W, H)

    def test_moderate_disagreement_anchors_at_lower_observability(self):
        clean = [self.image(n) for n in self.names]
        keys = Keyframes([], [{'time': 0.0, 'points': clean}], W, H)
        keys.update(0.0, np.eye(3), True)
        reference = keys.pitch.observability
        self.assertIsNotNone(keys.H)
        warped = None
        for d in np.linspace(.002, .06, 60):                # every click a little off, alternately up and down, until they disagree by 1.5-4 m
            candidate = [dict(p, y=p['y']+d*(1 if i % 2 else -1)) for i, p in enumerate(clean)]
            rms = calibration.solve_details(candidate, W, H, max_error=10.0)['rms']
            if 1.6 < rms < 3.5:
                warped = candidate
                break
        self.assertIsNotNone(warped, 'could not build a moderately disagreeing keyframe')
        with self.assertRaises(ValueError):
            calibration.solve(warped, W, H)                  # the strict solver still refuses it
        keys = Keyframes([], [{'time': 0.0, 'points': warped}], W, H)
        keys.update(0.0, np.eye(3), True)
        self.assertIsNotNone(keys.H, keys.warning)           # ... but it anchors
        self.assertLess(keys.pitch.observability, reference-.1)
        self.assertGreater(keys.rms, 1.5)


class ReregistrationTests(unittest.TestCase):
    """Phase 1: a later frame is registered directly to the keyframe's view, so the model is re-tied to the
    keyframe's geometry instead of drifting with the chain of frame-to-frame motions."""

    @staticmethod
    def gray(image):
        return cv2.cvtColor(cv2.resize(image, (WORK_W, round(image.shape[0]*WORK_W/image.shape[1]))), cv2.COLOR_BGR2GRAY)

    def test_a_panned_frame_registers_directly_to_the_keyframe_view(self):
        P = field.orient(broadcast.camera(), W, H)
        frame = broadcast.render(P)
        shift = np.array([[1, 0, 90], [0, 1, 12], [0, 0, 1]], float)      # the picture moved right and down
        panned = cv2.warpPerspective(frame, shift, (W, H))
        views = AnchorViews(W, H)
        self.assertTrue(views.remember(0, 0.0, self.gray(frame), [], P, .8, 0))
        candidate = views.register(self.gray(panned), [], prefer=0)
        self.assertIsNotNone(candidate)
        self.assertGreaterEqual(candidate['matches'], 40)
        self.assertLess(candidate['rms'], 2.0)
        pts = np.float32([[300, 400], [900, 650], [640, 300]])
        self.assertLess(np.abs(transform_points(candidate['H'], pts)-transform_points(shift, pts)).max(), 3.0)
        # The model was carried by a motion estimate that overshot by 25 px; the direct registration corrects it.
        tracker = PitchTracker(W, H)
        self.assertTrue(tracker.anchor(np.linalg.inv(P), 0.0, .1, observability=.8, index=0))
        wrong = shift.copy()
        wrong[0, 2] += 25
        tracker.carry(wrong, True, False, 1.0, 1.0)
        chained = tracker.H.copy()
        moved = tracker.reregister(candidate['H'] @ P, 1.0, candidate['matches'], candidate['rms'], 'the keyframe view at 0.0s')
        self.assertIsNotNone(moved)
        self.assertGreater(moved, 15)
        truth = shift @ P
        spots = [(11, 34), (52.5, 34), (94, 34)]
        error = lambda Hm: float(np.max(np.hypot(*(field.project(Hm, spots)-field.project(truth, spots)).T)))
        self.assertGreater(error(chained), 20)
        self.assertLess(error(tracker.H), 4.0)
        self.assertEqual(tracker.state, 'reregistered')
        self.assertGreaterEqual(tracker.temporal, .85)
        self.assertTrue(any(kind == 'reregistered' for _, kind, _ in tracker.events))

    def test_a_frame_without_the_keyframe_view_is_not_registered_and_a_big_jump_is_refused(self):
        P = field.orient(broadcast.camera(), W, H)
        frame = broadcast.render(P)
        views = AnchorViews(W, H)
        self.assertTrue(views.remember(0, 0.0, self.gray(frame), [], P, .8, 0))
        rng = np.random.default_rng(1)
        noise = cv2.GaussianBlur(rng.integers(0, 255, size=(H, W, 3), dtype=np.uint8), (3, 3), 0)
        self.assertIsNone(views.register(self.gray(noise), [], prefer=0))
        tracker = PitchTracker(W, H)
        self.assertTrue(tracker.anchor(np.linalg.inv(P), 0.0, .1, observability=.8, index=0))
        far = np.array([[1, 0, .3*W], [0, 1, 0], [0, 0, 1]], float) @ P    # a registration that would move the view by 30% of the width
        self.assertIsNone(tracker.reregister(far, .2, 80, 1.0, 'a wrong view'))   # the chain is still confident: refused
        self.assertEqual(tracker.state, 'anchored')


if __name__ == '__main__':
    unittest.main()
