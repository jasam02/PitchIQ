import unittest

import cv2
import numpy as np

from scene import H, W, pitch_frame
from soccer import calibration
from soccer.camera import CameraMotion, CameraState, Keyframes
from soccer.geometry import transform_points


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


if __name__ == '__main__':
    unittest.main()
