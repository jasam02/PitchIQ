"""Lens distortion as a measurement (soccer/lens.py): the division-model coefficient that straightens bent
markings is recovered from their paint, straight lines give no estimate, and the run summary is the median."""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from soccer import lens  # noqa: E402

W, H = 1280, 720
LINES = (((40, 150), (1240, 190)), ((100, 600), (1180, 560)), ((300, 100), (330, 700)))


def bent_lines(k):
    out = []
    for a, b in LINES:
        t = np.linspace(0, 1, 120)[:, None]
        straight = np.array(a, float)+(np.array(b, float)-np.array(a, float))*t
        out.append(lens.distort_points(straight, k, W, H))
    return out


class LensTests(unittest.TestCase):
    def test_the_coefficient_is_recovered_from_bent_lines(self):
        for k in (.15, -.2):
            estimate = lens.estimate_k(bent_lines(k), W, H)
            self.assertIsNotNone(estimate, k)
            self.assertAlmostEqual(estimate['k'], k, delta=.03)
            self.assertGreater(estimate['gain'], .8)
            self.assertEqual(estimate['lines'], 3)
            back = lens.undistort_points(lens.distort_points([(200.0, 300.0)], k, W, H), k, W, H)
            self.assertLess(float(np.abs(back-[200, 300]).max()), .05)

    def test_straight_lines_give_no_estimate_and_the_summary_is_the_median(self):
        self.assertIsNone(lens.estimate_k(bent_lines(0.0), W, H))
        self.assertIsNone(lens.estimate_k([[(0, 0)]*10], W, H))          # too few points
        self.assertIsNone(lens.summarize([None, None]))
        summary = lens.summarize([{'k': .1, 'gain': .5}, {'k': .12, 'gain': .6}, None, {'k': .14, 'gain': .4}])
        self.assertEqual(summary['frames'], 3)
        self.assertAlmostEqual(summary['k'], .12, places=3)
        self.assertEqual(summary['agreement'], 1.0)


if __name__ == '__main__':
    unittest.main()
