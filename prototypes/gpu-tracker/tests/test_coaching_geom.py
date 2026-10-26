"""The tactical geometry of the coaching v2 detectors (soccer/coaching/tactics_geom.py): pure functions in a team's
attack frame (X from its own goal line, Y from its left touchline, own goal centre (0, W/2)), shared by the detectors
and the suggested-zone engine, so each relation is tested once here on hand-placed players whose answer is known.

Why these checks. A zone suggested by soccer/coaching/alternatives.py is accepted only when it restores the relation
the detector measured, so the point predicates (covers, covers_channel) must agree with the per-sample measurements
(cover_candidate, channel_occupied) on every scene: a property test draws random scenes and compares them. The scale
rules (a small field shortens the cover radius but a big one never stretches it; scaled distances follow L/105 and
W/68) decide whether a youth team is judged at its own size.

Run:  python -m unittest test_coaching_geom          (from this directory)
"""
import math
import unittest

import numpy as np

import scene  # noqa: F401  (adds the prototype directory to sys.path)
from soccer.coaching import tactics_geom as G  # noqa: E402

NAN = float('nan')
FULL = {'lengthM': 105.0, 'widthM': 68.0}
SMALL = {'lengthM': 52.5, 'widthM': 34.0}


class Scale(unittest.TestCase):
    def test_factors_and_scaled_distances(self):
        self.assertEqual(G.factors(FULL), (1.0, 1.0))
        self.assertEqual(G.factors(SMALL), (.5, .5))
        self.assertAlmostEqual(G.scaled(15.0, 'length', SMALL), 7.5)
        self.assertAlmostEqual(G.scaled(20.0, 'width', SMALL), 10.0)
        self.assertAlmostEqual(G.scaled(5.0, None, SMALL), 5.0)          # body scale: absolute metres

    def test_cover_radius_shrinks_with_the_field_but_never_stretches(self):
        self.assertAlmostEqual(G.cover_radius(FULL), 12.0)
        self.assertAlmostEqual(G.cover_radius({'lengthM': 120.0, 'widthM': 80.0}), 12.0)
        self.assertAlmostEqual(G.cover_radius({'lengthM': 70.0, 'widthM': 50.0}), 8.0)
        self.assertAlmostEqual(G.cover_radius({'lengthM': 30.0, 'widthM': 20.0}), 6.0)     # the floor

    def test_zones_and_channels(self):
        self.assertEqual([G.zone_of(x, 105.0) for x in (0.0, 34.9, 35.0, 69.9, 70.0)],
                         ['own_third', 'own_third', 'middle_third', 'middle_third', 'final_third'])
        self.assertIsNone(G.zone_of(NAN, 105.0))
        self.assertEqual([G.channel_of(y, 68.0) for y in (1.0, 22.6, 34.0, 45.4, 67.0)],
                         ['left', 'left', 'centre', 'right', 'right'])
        self.assertIsNone(G.channel_of(None, 68.0))
        self.assertTrue(G.in_outer_third(10.0, 'left', 68.0))
        self.assertFalse(G.in_outer_third(10.0, 'right', 68.0))


class PressureAndCover(unittest.TestCase):
    def test_first_defender_is_the_nearest_seen_player(self):
        X = np.array([10.0, 30.0, NAN, 22.0])
        Y = np.array([30.0, 30.0, 31.0, 34.0])
        j, d = G.first_defender(25.0, 34.0, X, Y)
        self.assertEqual(j, 3)
        self.assertAlmostEqual(d, 3.0)
        self.assertEqual(G.first_defender(25.0, 34.0, [NAN], [NAN])[0], None)

    def test_angle_at(self):
        self.assertAlmostEqual(G.angle_at(10.0, 34.0, 0.0, 34.0, 0.0, 34.0), 0.0)
        self.assertAlmostEqual(G.angle_at(10.0, 34.0, 10.0, 44.0, 0.0, 34.0), 90.0)
        self.assertTrue(math.isnan(G.angle_at(0.0, 34.0, 5.0, 5.0, 0.0, 34.0)))   # A at the goal: no direction

    def test_cover_candidate_prefers_the_goal_side_team_mate(self):
        # first defender at (30, 34); a team-mate 5 m beside him (90 deg off the goal line) and one 9 m goal-side
        X = np.array([30.0, 30.0, 21.0])
        Y = np.array([34.0, 39.0, 34.0])
        (cj, cd, ca), (sj, sd, sa) = G.cover_candidate(30.0, 34.0, X, Y, 68.0, exclude=[0])
        self.assertEqual((cj, round(cd, 3), round(ca, 3)), (2, 9.0, 0.0))
        self.assertEqual((sj, round(sd, 3)), (1, 5.0))
        self.assertAlmostEqual(sa, 90.0)
        self.assertTrue(G.covers((30.0, 34.0), (21.0, 34.0), 68.0, 12.0))
        self.assertFalse(G.covers((30.0, 34.0), (30.0, 39.0), 68.0, 12.0))         # beside him: no cover
        self.assertFalse(G.covers((30.0, 34.0), (16.0, 34.0), 68.0, 12.0))         # 14 m: too far

    def test_the_point_predicate_agrees_with_the_measurement_on_random_scenes(self):
        rng = np.random.default_rng(7)
        for _ in range(400):
            X = rng.uniform(0, 105, 10)
            Y = rng.uniform(0, 68, 10)
            X[rng.random(10) < .2] = NAN
            f = int(rng.integers(10))
            if not np.isfinite(X[f]):
                continue
            (cj, cd, _), _ = G.cover_candidate(X[f], Y[f], X, Y, 68.0, exclude=[f])
            measured = cj is not None and cd <= 12.0
            any_cover = any(G.covers((X[f], Y[f]), (X[k], Y[k]), 68.0, 12.0) for k in range(10)
                            if k != f and np.isfinite(X[k]))
            self.assertEqual(measured, any_cover)

    def test_closing_speed_and_goal_side_share(self):
        self.assertAlmostEqual(G.closing_speed(0.0, 0.0, 3.0, 0.0, 10.0, 0.0), 3.0)
        self.assertAlmostEqual(G.closing_speed(0.0, 0.0, -2.0, 0.0, 10.0, 0.0), -2.0)
        self.assertTrue(math.isnan(G.closing_speed(0.0, 0.0, NAN, 0.0, 10.0, 0.0)))
        self.assertAlmostEqual(G.goal_side_share([10.0, 20.0, 40.0, NAN], 30.0, exclude=[0]), .5)
        self.assertTrue(math.isnan(G.goal_side_share([NAN], 30.0)))


class WeakSideAndBehind(unittest.TestCase):
    def test_weak_side_counts_goal_side_defenders_in_the_far_half(self):
        # ball on our left (Y 10) at depth 40; defenders: two goal-side on the right half, one goal-side on the left,
        # one ahead of the ball on the right
        X = np.array([30.0, 35.0, 25.0, 50.0])
        Y = np.array([40.0, 60.0, 12.0, 55.0])
        w = G.weak_side(40.0, 10.0, X, Y, 68.0, FULL)
        self.assertEqual((w['ballSide'], w['weakCount'], w['centralCount']), ('left', 2, 1))
        self.assertAlmostEqual(w['openWidthM'], 8.0)                         # 68 - 60
        self.assertEqual(w['nearestWeak'], 1)
        self.assertIsNone(G.weak_side(40.0, 34.0, X, Y, 68.0))               # the ball in the centre: not wide
        w = G.weak_side(40.0, 10.0, X[2:3], Y[2:3], 68.0, side='left')
        self.assertEqual((w['weakCount'], w['openWidthM']), (0, 34.0))       # nobody there: half the width open
        self.assertEqual(G.weak_side(40.0, 30.0, X, Y, 68.0, side='right')['ballSide'], 'right')

    def test_goal_side_with_the_level_tolerance(self):
        """B-5 (COACH B3): BC180's geometry at 188.233 s, a wide ball at the box edge (17.25, 59.5): the centre-back
        dead centre 0.01 m upfield of it is goal-side by his distance to the goal centre; the far full-back 2.25 m
        upfield too; a player 0.5 m upfield and as far from the goal as the ball is level; one 15 m upfield is not."""
        bx, by = 17.25, 59.5
        X = np.array([17.26, 19.5, 17.75, 32.25, NAN])
        Y = np.array([34.9, 20.8, 59.5, 59.5, 30.0])
        goal, level = G.goal_side_masks(bx, by, X, Y, 68.0, 1.0)
        self.assertEqual(goal.tolist(), [True, True, False, False, False])
        self.assertEqual(level.tolist(), [False, False, True, False, False])
        goal, level = G.goal_side_masks(bx, by, X, Y, 68.0, np.array([1.0, 1.0, .1, 1.0, 1.0]))
        self.assertEqual(level.tolist(), [False, False, False, False, False])     # a tight sigma: not level
        f = G.far_side_protection(bx, by, X, Y, 68.0, FULL, side='right', tol=1.0,
                                  vx=np.array([0.0, 0.0, 0.0, -4.0, 0.0]))
        self.assertEqual(f, {'weakGoal': 1, 'weakLevel': 0, 'centralGoal': 1, 'weakRecovering': 0})
        self.assertEqual(G.weak_side(bx, by, X, Y, 68.0, FULL, side='right')['weakCount'], 0)   # the plain rule
        Y2 = np.array([50.0, 50.0, 50.0, 20.0, 30.0])
        f = G.far_side_protection(40.0, 10.0, np.array([35.0, 70.0, 40.5, 60.0, NAN]), Y2, 68.0, FULL, side='left',
                                  tol=1.0, vx=np.array([0.0, -4.0, 0.0, 0.0, 0.0]))
        # ball (40, 10), 46.6 m from the goal centre: (35, 50) is deeper (goal-side by depth), (40.5, 50) 1.5 m upfield
        # but 43.5 m from the goal centre (goal-side by distance), (70, 50) neither but running back at 4 m/s
        self.assertEqual((f['weakGoal'], f['weakLevel'], f['weakRecovering']), (2, 0, 1))
        self.assertIsNone(G.far_side_protection(40.0, 34.0, X, Y, 68.0))

    def test_behind_the_ball_and_the_rest_defence_margin(self):
        X = np.array([20.0, 35.0, 40.0, 60.0, NAN])
        n, m = G.behind_ball(X, 50.0, 10.0)
        self.assertEqual(n, 3)
        self.assertEqual(m.tolist(), [True, True, True, False, False])
        self.assertEqual(G.rest_margin(X, np.array([30.0, 45.0, 80.0]), 50.0, 10.0), (3, 1, 2))


class WideDefenders(unittest.TestCase):
    def test_vacated_channel(self):
        for a, b in zip(G.vacated_channel(30.0, 60.0, 'left', 68.0, 10.0), (25.0, 50.0, 0.0, 68.0/3)):
            self.assertAlmostEqual(a, b)
        self.assertEqual(G.vacated_channel(30.0, 31.0, 'right', 66.0, 10.0)[:2], (25.0, 25.0))   # never inverted
        for a, b in zip(G.vacated_channel(2.0, 60.0, 'right', 66.0, 10.0)[2:], (44.0, 66.0)):
            self.assertAlmostEqual(a, b)

    def test_channel_occupied_and_its_point_predicate(self):
        # left back (0) at depth 60 on the left; team-mates: in the left third 15 m behind (1), in the left third but
        # only 5 m behind (2), 15 m behind but central (3)
        X = np.array([60.0, 45.0, 55.0, 45.0])
        Y = np.array([5.0, 15.0, 10.0, 34.0])
        self.assertEqual(G.channel_occupied(X, Y, 0, 'left', 68.0, 10.0), (True, 1))
        self.assertEqual(G.channel_occupied(X[[0, 2, 3]], Y[[0, 2, 3]], 0, 'left', 68.0, 10.0), (False, None))
        self.assertTrue(G.covers_channel((45.0, 15.0), (60.0, 5.0), 'left', 68.0, 10.0))
        self.assertFalse(G.covers_channel((55.0, 10.0), (60.0, 5.0), 'left', 68.0, 10.0))
        self.assertEqual(G.channel_occupied(X, Y, None, 'left', 68.0, 10.0), (False, None))

    def test_cb_shift_is_signed_towards_the_touchline(self):
        self.assertAlmostEqual(G.cb_shift(20.0, 26.0, 'left'), 6.0)
        self.assertAlmostEqual(G.cb_shift(50.0, 44.0, 'right'), 6.0)
        self.assertAlmostEqual(G.cb_shift(30.0, 26.0, 'left'), -4.0)
        self.assertTrue(math.isnan(G.cb_shift(30.0, None, 'left')))

    def test_opponents_in_the_vacated_channel(self):
        OX = np.array([30.0, 30.0, 70.0, 45.0, NAN])
        OY = np.array([10.0, 40.0, 10.0, 5.0, 5.0])
        self.assertEqual(G.opponents_in_channel(OX, OY, 60.0, 'left', 68.0, 105.0), [0, 3])
        self.assertEqual(G.opponents_in_channel(OX, OY, 40.0, 'left', 68.0, 105.0), [0])    # goal-side of him only
        # a defender 80 m up: an opponent behind him at 60 m is in the other half, not in our vacated channel
        self.assertEqual(G.opponents_in_channel(np.array([60.0, 50.0]), np.array([10.0, 10.0]), 80.0, 'left', 68.0,
                                                105.0), [1])


class BackLineAndShape(unittest.TestCase):
    def test_lateral_gaps(self):
        gap, a, b, gy = G.lateral_gaps([10.0, 50.0, NAN, 22.0, 30.0])
        self.assertEqual((round(gap, 3), a, b, round(gy, 3)), (20.0, 4, 1, 40.0))
        self.assertTrue(math.isnan(G.lateral_gaps([10.0, NAN])[0]))

    def test_opponent_in_gap(self):
        OX = np.array([31.0, 31.0, 60.0, 26.0])     # the last one 4 m behind the line: inside the full-size band
        OY = np.array([40.0, 60.0, 40.0, 40.0])
        self.assertEqual(G.opponent_in_gap(OX, OY, (30.0, 30.0), (32.0, 50.0), FULL), [0, 3])
        self.assertEqual(G.opponent_in_gap(OX, OY, (30.0, 30.0), (32.0, 50.0), SMALL), [0])  # the band shrinks too

    def test_line_std_block_height_extent_and_occupation(self):
        self.assertAlmostEqual(G.line_std([10.0, 12.0, NAN]), 1.0)
        self.assertTrue(math.isnan(G.line_std([10.0])))
        self.assertEqual([G.block_height(h, 100.0) for h in (20.0, 30.0, 45.0)], ['low', 'mid', 'high'])
        self.assertIsNone(G.block_height(NAN, 100.0))
        w, d = G.team_extent([10.0, 50.0, NAN], [5.0, 40.0, 1.0])
        self.assertEqual((w, d), (35.0, 40.0))
        self.assertTrue(math.isnan(G.team_extent([1.0], [1.0])[0]))
        self.assertEqual(G.channels_occupied([5.0, 30.0, 33.0], 68.0), 2)
        self.assertEqual(G.channels_occupied([5.0, 30.0, 60.0, NAN], 68.0), 3)
        self.assertEqual(G.bands_occupied([5.0, 80.0], 105.0), 2)


if __name__ == '__main__':
    unittest.main()
