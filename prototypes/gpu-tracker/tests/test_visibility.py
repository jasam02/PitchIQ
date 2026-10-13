"""What the camera sees of the pitch (soccer/visibility.py): the view polygon from the frame's homography, cut at
the horizon; unobserved roster slots classified as expected-in-frame, off-screen or unknown; nothing removed."""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import broadcast  # noqa: E402
from scene import H, W  # noqa: E402
from soccer import field, visibility  # noqa: E402
from soccer.geometry import point_in_polygon  # noqa: E402

SIZE = field.PitchSize()


def end_camera():
    import cv2
    src = np.float32([[0, 0], [105, 0], [105, 68], [0, 68]])
    dst = np.float32([[-.3*W, 1.4*H], [.52*W, .300*H], [.528*W, .303*H], [1.5*W, 1.6*H]])
    return field.orient(cv2.getPerspectiveTransform(src, dst).astype(np.float64), W, H)


class ViewTests(unittest.TestCase):
    def test_the_view_is_the_frame_on_the_ground_clipped_to_the_pitch(self):
        P = field.orient(broadcast.camera(), W, H)
        view = visibility.view_polygon(P, W, H, SIZE)
        self.assertIsNotNone(view)
        self.assertTrue(point_in_polygon((.5, .5), view))                          # the centre spot is in view
        self.assertTrue(all(-.05 <= x <= 1.05 and -.08 <= y <= 1.08 for x, y in view))   # clipped to the pitch plus the margin
        # The frame's own corners map onto the pitch inside the view; a point well beyond the right goal line does not.
        corner = field.unproject(P, [(W*.5, H*.9)])[0]/[SIZE.length, SIZE.width]
        self.assertTrue(point_in_polygon(tuple(corner), view))
        self.assertFalse(point_in_polygon((1.3, .5), view))

    def test_the_view_stops_at_the_horizon_and_a_frame_without_calibration_has_none(self):
        P = end_camera()
        self.assertTrue(any(field.image_depth(P, [(W/2, y)])[0] <= 0 for y in np.linspace(0, .29*H, 30)), 'the end camera has its horizon in the frame')
        view = visibility.view_polygon(P, W, H, SIZE)
        self.assertIsNotNone(view)
        self.assertTrue(all(np.isfinite(v) for p in view for v in p))
        self.assertFalse(point_in_polygon((1.0, .5), view))      # the far goal line, at the horizon, is not seen
        self.assertIsNone(visibility.view_polygon(None, W, H, SIZE))


class ClassificationTests(unittest.TestCase):
    def setUp(self):
        self.P = field.orient(broadcast.camera(), W, H)
        self.view = visibility.view_polygon(self.P, W, H, SIZE)

    def test_inside_outside_and_the_uncertain_edge(self):
        self.assertEqual(visibility.classify((.5, .5), 2.0, self.view, SIZE), 'C')
        self.assertEqual(visibility.classify((1.2, .5), 2.0, self.view, SIZE), 'S')
        self.assertEqual(visibility.classify((1.2, .5), 40.0, self.view, SIZE), 'U')   # too uncertain to say
        self.assertEqual(visibility.classify((.5, .5), 2.0, None, SIZE), 'U')
        self.assertEqual(visibility.classify(None, 2.0, self.view, SIZE), 'U')

    def test_annotate_classifies_every_unobserved_slot_from_its_last_position(self):
        frames = []
        for i in range(30):
            t = round(i*.2, 3)
            people = [{'track': 1, 'id': 'A-01', 'team': 'A', 'role': 'PLAYER', 'state': 'confirmed', 'pitch': [.5, .5], 'box': [.5, .4, .02, .1]}]
            if i < 10:
                people.append({'track': 2, 'id': 'A-02', 'team': 'A', 'role': 'PLAYER', 'state': 'confirmed', 'pitch': [.98, .5], 'box': [.9, .4, .02, .1]})
            frames.append({'time': t, 'people': people, 'calibrated': i < 25, 'calibration': {'H': self.P.ravel().tolist() if i < 25 else None, 'confidence': .8}})
        result = {'video': {'width': W, 'height': H}, 'frames': frames, 'tactical': {'pitchSize': SIZE.to_json()},
                  'roster': {'slots': [{'id': 'A-01', 'status': 'ACTIVE'}, {'id': 'A-02', 'status': 'UNKNOWN'}, {'id': 'A-03', 'status': 'UNKNOWN'}, {'id': 'A-09', 'status': 'RETIRED'}]}}
        summary = visibility.annotate(result)
        self.assertEqual(summary['estimates'], 'last-observation')
        self.assertEqual(summary['framesWithView'], 25)
        self.assertIsNotNone(frames[0]['view'])
        self.assertIsNone(frames[29]['view'])
        self.assertNotIn('A-01', frames[12]['visibility'])                      # observed: implied
        self.assertIn(frames[12]['visibility']['A-02'], ('C', 'S'))             # just after it vanished at the right edge: a clear call
        self.assertEqual(frames[12]['visibility']['A-03'], 'U')                # never seen: nothing to say
        self.assertEqual(frames[29]['visibility']['A-02'], 'U')                # no calibration here
        self.assertNotIn('A-09', frames[12]['visibility'])                      # retired slots are not classified
        self.assertAlmostEqual(sum(summary['shares'].values()), 1.0, places=2)
        self.assertEqual(len(frames[12]['people']), 1)                          # nothing removed
        late = frames[26]['visibility']['A-02']
        self.assertEqual(late, 'U')                                             # 5.2 s after the last observation and no view: unknown


if __name__ == '__main__':
    unittest.main()
