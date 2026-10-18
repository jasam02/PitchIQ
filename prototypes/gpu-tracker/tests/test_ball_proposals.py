"""The classical near-feet ball proposer (soccer/ball_proposals.py) on synthetic frames: a cream ball on grass is
proposed at its size for that row; paint, specks, oversized white things, shirts, white socks and what the detector
already reported are not."""
import unittest

import cv2
import numpy as np

import scene  # noqa: F401  (sys.path)
from soccer import ball_proposals as P

W, H = 640, 360
GRASS = (40, 120, 55)      # BGR, the olive turf of amateur 3 (darkest channel ~40-70)
CREAM = (215, 228, 232)    # BGR, a sunlit ball


def field():
    rng = np.random.default_rng(3)
    frame = np.empty((H, W, 3), np.uint8)
    frame[:] = GRASS
    noise = rng.integers(-12, 13, (H, W, 1))
    return np.clip(frame.astype(int)+noise, 0, 255).astype(np.uint8)


def expected(d):
    """A ball of d px expected everywhere (expected(y) -> d)."""
    return lambda y: d


def centres(found):
    return [((x1+x2)/2, (y1+y2)/2) for x1, y1, x2, y2, _ in found]


class ProposalTests(unittest.TestCase):
    def test_a_ball_on_the_grass_is_proposed_at_its_place(self):
        frame = field()
        cv2.circle(frame, (300, 200), 9, CREAM, -1)
        cv2.circle(frame, (297, 203), 2, (60, 60, 60), -1)   # a dark panel: closed before the blob is judged
        found = P.find(frame, [(0, 0, W, H)], expected(13.0))
        self.assertEqual(len(found), 1, found)
        (cx, cy), = centres(found)
        self.assertAlmostEqual(cx, 300, delta=2)
        self.assertAlmostEqual(cy, 200, delta=2)
        self.assertGreater(found[0][2]-found[0][0], 14)

    def test_paint_specks_and_oversized_white_things_are_not(self):
        frame = field()
        cv2.line(frame, (20, 100), (620, 120), (230, 235, 235), 6)      # a painted line
        cv2.circle(frame, (100, 250), 2, CREAM, -1)                        # a speck: the detector's business
        cv2.circle(frame, (400, 260), 30, CREAM, -1)                       # far larger than a ball at that row
        cv2.ellipse(frame, (520, 260), (16, 4), 0, 0, 360, CREAM, -1)      # elongated
        self.assertEqual(P.find(frame, [(0, 0, W, H)], expected(13.0)), [])

    def test_a_notched_or_hollow_blob_is_not_a_ball(self):
        # The letters of a burned-in watermark ('e', 'o' of Veo's logo) are round, white and ball-sized at the near rows.
        frame = field()
        cv2.circle(frame, (200, 200), 12, CREAM, -1)
        frame[196:203, 202:214] = GRASS                                    # an 'e'-like notch
        cv2.circle(frame, (400, 200), 12, CREAM, -1)
        cv2.circle(frame, (400, 200), 5, GRASS, -1)                        # an 'o'-like hole (a sixth of it, as Veo's)
        self.assertEqual(P.find(frame, [(0, 0, W, H)], expected(16.0)), [])

    def test_a_ball_next_to_a_line_is_still_proposed(self):
        frame = field()
        cv2.line(frame, (330, 0), (330, H), (230, 235, 235), 5)
        cv2.circle(frame, (310, 200), 9, CREAM, -1)
        self.assertEqual(len(P.find(frame, [(0, 0, W, H)], expected(13.0))), 1)

    def test_shirts_and_white_socks_are_not_but_the_ball_beside_the_feet_is(self):
        frame = field()
        box = [200/W, 100/H, 40/W, 110/H]   # a person 110 px tall, feet at y = 210
        cv2.circle(frame, (220, 140), 9, CREAM, -1)    # on the shirt (upper body)
        cv2.circle(frame, (220, 190), 9, CREAM, -1)    # on the shins, in the middle of the box: a white sock
        cv2.circle(frame, (275, 204), 9, CREAM, -1)    # beside the feet: the ball being dribbled
        people = [(7, box)]
        found = centres(P.find(frame, [(0, 0, W, H)], expected(13.0), people))
        self.assertEqual(len(found), 1, found)
        self.assertAlmostEqual(found[0][0], 275, delta=2)

    def test_nothing_is_proposed_where_the_detector_already_reported_or_outside_the_windows(self):
        frame = field()
        cv2.circle(frame, (300, 200), 9, CREAM, -1)
        self.assertEqual(P.find(frame, [(0, 0, W, H)], expected(13.0), taken=[(290, 190, 310, 210)]), [])
        self.assertEqual(P.find(frame, [(400, 0, W, H)], expected(13.0)), [])
        self.assertEqual(len(P.find(frame, [(250, 150, 350, 250), (260, 160, 360, 260)], expected(13.0))), 1)   # overlapping windows: once

    def test_a_window_without_grass_judges_nothing(self):
        frame = np.full((H, W, 3), 90, np.uint8)   # concrete, stands
        cv2.circle(frame, (300, 200), 9, CREAM, -1)
        self.assertEqual(P.find(frame, [(0, 0, W, H)], expected(13.0)), [])


if __name__ == '__main__':
    unittest.main()
