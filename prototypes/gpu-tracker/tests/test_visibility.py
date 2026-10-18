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

    def test_a_stored_homography_that_lost_its_sign_gives_the_same_view(self):
        """registration.homography_json divided by a negative H[2, 2] on frames whose far-left corner is behind the camera
        (the Veo regression window: 272 of 900 frames); the view is computed from the re-oriented matrix."""
        for P in (field.orient(broadcast.camera(), W, H), end_camera()):
            self.assertEqual(visibility.view_polygon(-P, W, H, SIZE), visibility.view_polygon(P, W, H, SIZE))

    def test_the_black_corners_of_a_dewarped_frame_are_not_seen(self):
        import cv2
        P = field.orient(broadcast.camera(), W, H)
        frame = np.full((H, W, 3), (60, 140, 50), np.uint8)
        cv2.ellipse(frame, (W, 0), (int(.45*W), int(.75*H)), 0, 0, 360, (0, 0, 0), -1)      # the dewarp's curved top-right corner
        area = visibility.image_area(frame)
        self.assertIsNotNone(area)
        self.assertTrue(point_in_polygon((.1, .9), area))
        self.assertFalse(point_in_polygon((.95, .05), area))
        hidden = field.unproject(P, [(.9*W, .4*H)])[0]/[SIZE.length, SIZE.width]          # ground seen only through the black corner
        plain, cut = visibility.view_polygon(P, W, H, SIZE), visibility.view_polygon(P, W, H, SIZE, area)
        self.assertTrue(point_in_polygon(tuple(hidden), plain))
        self.assertFalse(point_in_polygon(tuple(hidden), cut))
        self.assertTrue(point_in_polygon((.5, .5), cut))
        # A frame that is image everywhere, or nearly (a few dark pixels at a border), has no outline: the view is unchanged.
        self.assertIsNone(visibility.image_area(np.full((H, W, 3), (60, 140, 50), np.uint8)))
        frame[:] = (60, 140, 50)
        frame[:20, :20] = 0
        self.assertIsNone(visibility.image_area(frame))

    def test_an_outline_only_takes_area_away_also_where_the_horizon_cuts_the_view(self):
        """The end camera sees the horizon; an outline without the four corners (their deepest one included) must not move
        the horizon cut: every point of the outlined view is in the plain view (it used to admit a band at the horizon)."""
        import cv2
        P = end_camera()
        self.assertTrue(min(field.image_depth(P, [(0, 0), (W, 0), (W, H), (0, H)])) < 0)
        frame = np.full((H, W, 3), (60, 140, 50), np.uint8)
        for corner in ((0, 0), (W, 0), (W, H), (0, H)):
            cv2.circle(frame, corner, int(.30*W), (0, 0, 0), -1)
        area = visibility.image_area(frame)
        self.assertIsNotNone(area)
        plain, cut = visibility.view_polygon(P, W, H, SIZE), visibility.view_polygon(P, W, H, SIZE, area)
        self.assertIsNotNone(cut)
        points = np.random.default_rng(0).uniform([-.05, -.08], [1.05, 1.08], (20000, 2))
        inside_cut = [tuple(p) for p in points if point_in_polygon(tuple(p), cut)]
        self.assertGreater(len(inside_cut), 500)
        self.assertEqual([p for p in inside_cut if not point_in_polygon(p, plain)], [])
        self.assertLess(len(inside_cut), sum(point_in_polygon(tuple(p), plain) for p in points))   # and the corners are taken away
        # An outline that keeps only image beyond the cut leaves no view.
        self.assertIsNone(visibility.view_polygon(P, W, H, SIZE, [[.4, 0], [.6, 0], [.5, .05]]))

    def test_image_area_finds_the_black_wherever_it_touches_the_border_and_only_there(self):
        import cv2
        green = (60, 140, 50)
        # A virtual camera tilted down meets the edge of the lens image in the middle of the top border, no corner black
        # (amateur 3 at 150-158 s): found all the same.
        frame = np.full((H, W, 3), green, np.uint8)
        cv2.ellipse(frame, (W//2, -int(.25*H)), (int(.3*W), int(.45*H)), 0, 0, 360, (0, 0, 0), -1)
        self.assertTrue(frame[0, 0].any() and frame[0, -1].any())         # both top corners are image
        area = visibility.image_area(frame)
        self.assertIsNotNone(area)
        self.assertFalse(point_in_polygon((.5, .05), area))
        self.assertTrue(point_in_polygon((.05, .05), area) and point_in_polygon((.5, .6), area))
        # Grey frames work the same way.
        grey = visibility.image_area(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        self.assertIsNotNone(grey)
        self.assertFalse(point_in_polygon((.5, .05), grey))
        self.assertTrue(point_in_polygon((.05, .05), grey) and point_in_polygon((.5, .6), grey))
        # A dewarp's four black corners: one outline without any of them.
        frame[:] = green
        for corner in ((0, 0), (W, 0), (W, H), (0, H)):
            cv2.circle(frame, corner, int(.2*W), (0, 0, 0), -1)
        area = visibility.image_area(frame)
        self.assertTrue(point_in_polygon((.5, .5), area))
        self.assertFalse(any(point_in_polygon(p, area) for p in ((.02, .03), (.98, .03), (.98, .97), (.02, .97))))
        # Black that does not reach the border is dark image (a shadow, a dark kit), never 'no image'; the border
        # pre-check stops there without measuring the frame.
        frame[:] = green
        cv2.circle(frame, (W//2, H//2), int(.3*H), (0, 0, 0), -1)
        self.assertFalse(visibility._dark_border(frame, int(round(visibility.NO_IMAGE_WIDTH*H/W))))
        self.assertIsNone(visibility.image_area(frame))
        # A black frame or a fade (more than half without image) is not a dewarp: no outline.
        self.assertIsNone(visibility.image_area(np.zeros((H, W, 3), np.uint8)))
        frame[:] = 0
        frame[:, :int(.4*W)] = green
        self.assertIsNone(visibility.image_area(frame))


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
                  'roster': {'slots': [{'id': 'A-01', 'status': 'ACTIVE'}, {'id': 'A-02', 'status': 'OFF_SCREEN', 'exitEdge': 'right', 'lastSeen': 1.8},
                                       {'id': 'A-03', 'status': 'UNKNOWN'}, {'id': 'A-04', 'status': 'UNKNOWN', 'exitEdge': 'left', 'lastSeen': None},
                                       {'id': 'A-09', 'status': 'RETIRED'}]}}
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
        # Seen leaving through the right edge and not since: off-screen on every later frame with a view, even once the
        # estimate is too uncertain to place it (never 'U' while the view exists); a slot never seen stays unknown.
        self.assertTrue(all(frames[i]['visibility']['A-02'] in ('C', 'S') for i in range(11, 25)), [frames[i]['visibility']['A-02'] for i in range(11, 25)])
        self.assertEqual(frames[20]['visibility']['A-02'], 'S')
        self.assertEqual(frames[20]['visibility']['A-04'], 'U')

    def test_annotate_uses_the_latest_ticks_image_outline(self):
        """A slot last seen where the frame now has no image (a dewarp's black corner) is not 'expected in frame'."""
        hidden = [round(float(v), 4) for v in field.unproject(self.P, [(.85*W, .45*H)])[0]/[SIZE.length, SIZE.width]]
        area = [[0, 0], [.4, 0], [1, .8], [1, 1], [0, 1]]       # the image outline: the top-right corner is black

        def run(with_area):
            frames = []
            for i in range(10):
                people = [{'track': 1, 'id': 'A-01', 'state': 'confirmed', 'pitch': [.5, .5], 'box': [.5, .4, .02, .1]}]
                if i == 0:
                    people.append({'track': 2, 'id': 'A-02', 'state': 'confirmed', 'pitch': hidden, 'box': [.84, .35, .02, .1]})
                frames.append({'time': round(i*.2, 3), 'people': people, 'calibrated': True, 'calibration': {'H': self.P.ravel().tolist(), 'confidence': .8},
                               'pitch': ({'imageArea': area} if with_area else {}) if i % 5 == 0 else None})
            result = {'video': {'width': W, 'height': H}, 'frames': frames, 'tactical': {'pitchSize': SIZE.to_json()},
                      'roster': {'slots': [{'id': 'A-01', 'status': 'ACTIVE'}, {'id': 'A-02', 'status': 'ACTIVE'}]}}
            return visibility.annotate(result), frames
        summary, frames = run(False)
        self.assertEqual(frames[3]['visibility']['A-02'], 'C')
        self.assertNotIn('viewsWithoutBlackCorners', summary)
        summary, frames = run(True)
        self.assertEqual(summary['viewsWithoutBlackCorners'], 10)
        self.assertNotEqual(frames[3]['visibility']['A-02'], 'C')      # carried from the tick at frame 0 to frame 3
        self.assertFalse(point_in_polygon(tuple(hidden), frames[3]['view']))


if __name__ == '__main__':
    unittest.main()
