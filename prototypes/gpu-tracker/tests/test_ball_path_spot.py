"""The spot-lock rule of the offline ball decoder (soccer/ball_path._spot_locks) on hand-made sightings.

The CAN-ISL tactical-wide case it was written for: the causal pass locked onto the right penalty spot for 15 s (the
detector scores a painted spot like a ball) while the real ball was hidden in the midfield crowd; the painted-spot
discount spared the lock because the causal pass had taken it; the lock ended with the causal pass jumping 25 m to the
real ball. What is checked: such a stay (with the spot re-detected by short tracklets before and after it, as there) is
the spot: its sightings become a field marking without the acceptance bonus; a ball seen arriving at the spot (another
tracklet rolling in from metres away), a ball seen leaving it (kicked: another tracklet metres away soon after, at a
ball's speed), and a lock whose own tracklet rolls off the spot (a kickoff tap) all keep the exemption; a stay shorter
than SPOT_LOCK_S and a still thing off every painted spot are not touched.

Run:  python -m unittest test_ball_path_spot          (from this directory)
"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from soccer import ball_path as bp  # noqa: E402
from soccer.geometry import PAINTED_SPOTS  # noqa: E402

FPS = 25.0
SPOT = PAINTED_SPOTS[1]   # the right penalty spot (normalized pitch coordinates)


def node(i, tracklet, pitch, status='ball', det=.7):
    source = {'status': status, 'det': det, 'visual': .9, 'scores': {}}
    n = bp.Node(i, 1000.0, 500.0, 10.0, bp.quality(source), None, None, 0.0, 0.0, 0.0, 0.0, 30.0, tracklet, source)
    n.pitch = pitch
    return n


def scene(n_frames, sightings):
    """layers, times, T, Tinv for frames 0..n-1 (a still camera); sightings: [(frame, tracklet, pitch)]."""
    layers = [[] for _ in range(n_frames)]
    for i, k, p in sightings:
        layers[i].append(node(i, k, p))
    times = [i/FPS for i in range(n_frames)]
    T = [np.eye(3) for _ in range(n_frames)]
    return layers, times, T, [np.eye(3) for _ in range(n_frames)]


def on_spot(dx=0.0, dy=0.0):
    return (SPOT[0]+dx/105.0, SPOT[1]+dy/68.0)


class SpotLockTests(unittest.TestCase):
    def can_isl(self, before=(), after=()):
        """A 15 s lock (tracklet 360) on the spot from frame 100, re-detected by tracklet 359 just before and 396 after."""
        s = [(99, 359, on_spot(.5))] + [(i, 360, on_spot(.2*np.sin(i))) for i in range(100, 475)]
        s += [(i, 396, on_spot(.3)) for i in range(480, 484)]
        return scene(600, s+list(before)+list(after))

    def test_the_can_isl_stay_is_the_spot(self):
        far_ball = [(484, 380, (.75, .79))] + [(i, 388, (.75, .80+.002*(i-485))) for i in range(485, 520)]
        layers, times, T, Tinv = self.can_isl(after=far_ball)
        out = bp._spot_locks(layers, times, T, Tinv)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]['tracklets'], [359, 360, 396])
        # A short re-detection in the middle of the stay (CAN-ISL's tracklet 370) does not cut it: the end is the latest one.
        s = [(99, 359, on_spot(.5))] + [(i, 360, on_spot()) for i in range(100, 475)] + [(i, 370, on_spot(.1)) for i in range(250, 253)]
        s += [(i, 396, on_spot(.3)) for i in range(480, 484)]
        layers, times, T, Tinv = scene(600, s+far_ball)
        self.assertEqual(bp._spot_locks(layers, times, T, Tinv)[0]['tracklets'], [359, 360, 370, 396])

    def test_the_real_ball_far_away_soon_after_is_no_departure(self):
        # Without the re-detection after the stay: the real ball first seen 25 m away 0.84 s after the stay ended is
        # reachable at 30 m/s, but no ball kicked off a spot is first seen that far.
        s = [(i, 360, on_spot()) for i in range(100, 475)] + [(i, 388, (.75, .80)) for i in range(496, 520)]
        layers, times, T, Tinv = scene(600, s)
        self.assertEqual(len(bp._spot_locks(layers, times, T, Tinv)), 1)
        demoted = [m for l in layers for m in l if m.tracklet in (359, 360, 396)]
        self.assertTrue(all(m.marking for m in demoted))
        self.assertTrue(all(m.q <= 1e-3+1e-12 for m in demoted))   # never a frame of the path
        self.assertFalse(any(m.marking for l in layers for m in l if m.tracklet == 388))   # the real ball is untouched

    def test_a_ball_seen_arriving_keeps_the_exemption(self):
        rolling_in = [(i, 7, on_spot(-8+.4*(i-80))) for i in range(80, 99)]    # rolls in at 10 m/s from 8 m away
        layers, times, T, Tinv = self.can_isl(before=rolling_in)
        self.assertEqual(bp._spot_locks(layers, times, T, Tinv), [])
        self.assertFalse(any(m.marking for l in layers for m in l if m.tracklet == 360))

    def test_a_ball_seen_leaving_keeps_the_exemption(self):
        kicked = [(i, 8, on_spot(3+.9*(i-490))) for i in range(490, 500)]       # 3 m off the spot 0.24 s after, flying
        layers, times, T, Tinv = self.can_isl(after=kicked)
        self.assertEqual(bp._spot_locks(layers, times, T, Tinv), [])

    def test_a_lock_that_rolls_off_the_spot_is_a_ball(self):
        s = [(i, 50, on_spot()) for i in range(0, 200)] + [(i, 50, on_spot(.25*(i-199))) for i in range(200, 215)]
        layers, times, T, Tinv = scene(300, s)
        self.assertEqual(bp._spot_locks(layers, times, T, Tinv), [])

    def test_a_short_stay_and_a_still_thing_elsewhere_are_not_touched(self):
        layers, times, T, Tinv = scene(300, [(i, 60, on_spot()) for i in range(0, 40)])        # 1.6 s on the spot
        self.assertEqual(bp._spot_locks(layers, times, T, Tinv), [])
        layers, times, T, Tinv = scene(600, [(i, 61, (.4, .3)) for i in range(0, 400)])        # 16 s on no painted spot
        self.assertEqual(bp._spot_locks(layers, times, T, Tinv), [])
        self.assertFalse(any(m.marking for l in layers for m in l))


if __name__ == '__main__':
    unittest.main()
