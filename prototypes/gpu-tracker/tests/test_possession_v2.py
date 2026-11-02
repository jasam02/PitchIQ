"""Possession v2 (soccer/possession_v2.py) on synthetic scenes (tests/possession_v2_synth.py): the spec's ten possession
cases, the ball's observation states, the time weighting and the shares, and that possession never touches the ball.

Run:  python -m unittest test_possession_v2      (from this directory; a few seconds)
"""
import copy
import json
import math
import os
import sys
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
from soccer import possession_v2 as pv  # noqa: E402
import possession_v2_synth as sy  # noqa: E402

A_, B_, CON, FREE, UNK, POSS, STOP, HALF = 0, 1, 2, 3, 4, 5, 6, 7


def codes_between(T, out, t0, t1, key='code'):
    sel = (T['t'] >= t0) & (T['t'] < t1)
    return out[key][sel]


def share(c, value):
    return float(np.mean(np.isin(c, value))) if len(c) else 0.0


def nowhere(t):
    return None


class PossessionCases(unittest.TestCase):
    """The spec's Part 15 cases 1-10 (case 7, halftime, and 8, a segment boundary, run through the match layer below)."""

    def test_1_a_brief_occlusion_keeps_the_team_and_says_inferred(self):
        b07 = sy.lerp([(0, 50, 30), (8, 58, 30)])
        b09 = sy.lerp([(0, 56, 33), (8, 60, 33)])
        a01 = sy.lerp([(0, 47, 36), (8, 53, 36)])
        sc = sy.Scene(8).player('B07', 'B', b07).player('B09', 'B', b09).player('A01', 'A', a01)
        sc.ball(sy.switch((4.0, sy.at(b07)), (4.8, nowhere), (9, sy.at(b09))))
        T, out, post, path, M, aux, bst = pv.analyse(sc.result())
        gap = codes_between(T, out, 4.05, 4.75)
        self.assertGreaterEqual(share(gap, B_), .9, 'B keeps the ball through the 0.8 s it is hidden')
        self.assertTrue(np.all(codes_between(T, out, 4.05, 4.75, 'design') == 2), 'and the gap is inferred, not observed')
        self.assertEqual(out['changes'], [])

    def test_2_a_duel_nobody_controls_is_not_a_confident_team(self):
        a05 = sy.lerp([(0, 40, 30), (3, 44, 30), (5, 44.5, 30), (8, 46, 31)])
        b03 = sy.lerp([(0, 47, 30), (3, 44.8, 30.2), (5, 45.0, 30.2), (8, 49, 30)])
        ball = sy.switch((3, sy.at(a05)), (5, lambda t: (44.4+.2*math.sin(6*t), 30.1)), (9, sy.at(b03)))
        sc = sy.Scene(8).player('A05', 'A', a05).player('B03', 'B', b03).ball(ball)
        T, out, post, path, M, aux, bst = pv.analyse(sc.result())
        duel = codes_between(T, out, 3.4, 5.0)
        self.assertLessEqual(share(duel, A_), .5, 'two players at a loose ball: not A for most of the duel')
        self.assertGreater(share(duel, [CON, UNK, FREE]), .4)
        ch = [c for c in out['changes'] if c['to'] == 'B']
        self.assertEqual(len(ch), 1)
        self.assertTrue(3.0 <= ch[0]['wonAt'] <= 6.0)

    def test_3_a_long_clearance_is_not_given_to_the_kicker(self):
        a02 = sy.lerp([(0, 20, 34), (10, 22, 34)])
        b06 = sy.lerp([(0, 70, 30), (7, 64, 32), (10, 66, 33)])
        b08 = sy.lerp([(0, 72, 38), (7, 66, 36), (10, 67, 36)])
        kick = sy.lerp([(3, 20.4, 34), (3.5, 30.4, 34)])
        ball = sy.switch((3, sy.at(a02)), (3.5, kick), (7, nowhere), (11, sy.at(b06)))
        sc = sy.Scene(10).player('A02', 'A', a02).player('B06', 'B', b06).player('B08', 'B', b08).ball(ball)
        T, out, post, path, M, aux, bst = pv.analyse(sc.result())
        flight = codes_between(T, out, 4.0, 7.0)
        self.assertLess(share(flight, A_), .5, 'the unseen seconds after a clearance are not all A\'s')
        ch = [c for c in out['changes'] if c['to'] == 'B']
        self.assertEqual(len(ch), 1)
        self.assertLess(abs(ch[0]['wonAt']-7.0), 1.0, 'B wins it where B is first seen with it')

    def test_4_a_pass_between_team_mates_stays_the_team(self):
        b07 = sy.lerp([(0, 40, 30), (8, 42, 30)])
        b09 = sy.lerp([(0, 52, 34), (8, 54, 34)])
        a04 = sy.lerp([(0, 46, 40), (8, 47, 40)])
        fl = sy.lerp([(3, 40.9, 30), (4.2, 52.8, 34)])
        ball = sy.switch((3, sy.at(b07)), (4.2, fl), (9, sy.at(b09)))
        state = lambda t: 'MISSING' if 3.4 <= t < 4.0 else 'TRACKED'
        sc = sy.Scene(8).player('B07', 'B', b07).player('B09', 'B', b09).player('A04', 'A', a04).ball(ball, state)
        T, out, post, path, M, aux, bst = pv.analyse(sc.result())
        flight = codes_between(T, out, 3.0, 4.2)
        self.assertEqual(share(flight, [A_, FREE, CON]), 0.0, 'a completed pass is never A\'s or a free ball')
        self.assertGreaterEqual(share(flight, B_), .7)
        self.assertEqual(out['changes'], [])

    def test_4b_a_slow_ground_pass_between_team_mates_stays_the_team(self):
        b07 = sy.lerp([(0, 40, 30), (9, 42, 30)])
        b09 = sy.lerp([(0, 52, 34), (9, 54, 34)])
        a04 = sy.lerp([(0, 46, 40), (9, 47, 40)])
        fl = sy.lerp([(3, 40.9, 30), (6, 52.8, 34)])
        ball = sy.switch((3, sy.at(b07)), (6, fl), (10, sy.at(b09)))
        sc = sy.Scene(9).player('B07', 'B', b07).player('B09', 'B', b09).player('A04', 'A', a04).ball(ball)
        T, out, post, path, M, aux, bst = pv.analyse(sc.result())
        flight = codes_between(T, out, 3.0, 6.0)
        self.assertEqual(share(flight, [A_, FREE, CON]), 0.0, 'a 3 s pass to a team-mate is not a free ball (CAN-ISL W8)')
        self.assertEqual(out['changes'], [], 'and no turnover is invented to explain it')

    def test_5_an_interception_changes_possession_when_the_other_team_controls(self):
        a04 = sy.lerp([(0, 30, 30), (8, 31, 30)])
        a08 = sy.lerp([(0, 45, 30), (8, 46, 30)])
        b02 = sy.lerp([(0, 40, 33), (4, 38.5, 30.3), (8, 41, 31)])
        fl = sy.lerp([(3, 30.9, 30), (4, 38.5, 30.2)])
        ball = sy.switch((3, sy.at(a04)), (4, fl), (9, sy.at(b02)))
        sc = sy.Scene(8).player('A04', 'A', a04).player('A08', 'A', a08).player('B02', 'B', b02).ball(ball)
        T, out, post, path, M, aux, bst = pv.analyse(sc.result())
        ch = out['changes']
        self.assertEqual([(c['from'], c['to']) for c in ch], [('A', 'B')])
        self.assertLess(abs(ch[0]['wonAt']-4.0), 1.0)
        self.assertGreaterEqual(share(codes_between(T, out, 5.0, 8.0), B_), .9)
        self.assertGreaterEqual(share(codes_between(T, out, .5, 2.8), A_), .9)

    def test_6_a_stoppage_is_left_out_and_the_restart_goes_on(self):
        a07 = sy.lerp([(0, 50, 60), (5, 55, 66), (16, 55, 66)])
        b11 = sy.lerp([(0, 60, 50), (8, 58, 68.5), (12, 58, 68.5), (16, 57, 60)])
        b05 = sy.lerp([(0, 62, 55), (12, 57, 62), (16, 56, 58)])
        ball = sy.switch((5, sy.at(a07)), (8, sy.lerp([(5, 55.4, 66), (8, 58, 69.5)])), (12.0, sy.at(b11, 0, .3)),
                         (17, sy.lerp([(12, 58, 68.5), (13, 57, 62), (16, 56.4, 58)])))
        sc = sy.Scene(16).player('A07', 'A', a07).player('B11', 'B', b11).player('B05', 'B', b05).ball(ball)
        result = sc.result()
        T, out, post, path, M, aux, bst = pv.analyse(result, stoppages=[{'t0': 5.2, 't1': 12.0}], restarts={12.0: 'B'})
        self.assertTrue(np.all(codes_between(T, out, 5.25, 11.95) == STOP), 'the throw-in\'s preparation is not possession')
        self.assertGreaterEqual(share(codes_between(T, out, 13.0, 16.0), B_), .9)
        rec = pv.compute(result, stoppages=[{'t0': 5.2, 't1': 12.0}], restarts={12.0: 'B'})
        s = rec['stats']['match']
        self.assertAlmostEqual(s['seconds']['stoppage'], 12.0-5.2, delta=.1)
        self.assertAlmostEqual(s['live']+s['seconds']['stoppage'], 16.0, delta=.05, msg='every frame counted once')

    def test_9_a_ball_on_a_far_spot_or_a_boot_does_not_switch_possession(self):
        a03 = sy.lerp([(0, 30, 30), (8, 38, 31)])
        b10 = sy.lerp([(0, 33, 33), (8, 40, 33)])
        spot = lambda t: (94.0, 34.0)
        ball = sy.switch((3, sy.at(a03)), (3.6, spot), (5, sy.at(a03)), (5.3, sy.at(b10, .2, .3)), (9, sy.at(a03)))
        sc = sy.Scene(8).player('A03', 'A', a03).player('B10', 'B', b10).ball(ball)
        T, out, post, path, M, aux, bst = pv.analyse(sc.result())
        self.assertEqual(out['changes'], [])
        self.assertEqual(share(out['code'], B_), 0.0)
        self.assertGreaterEqual(share(codes_between(T, out, .5, 8.0), A_), .85)

    def test_10_a_long_hidden_ball_between_two_teams_stays_unresolved(self):
        a09 = sy.lerp([(0, 50, 30), (3, 52, 30), (7, 53, 32), (11, 52, 31), (14, 51, 30)])
        b04 = sy.lerp([(0, 56, 31), (3, 53.5, 30.5), (7, 52.5, 31.5), (11, 52.6, 31), (14, 55, 31)])
        b12 = sy.lerp([(0, 60, 35), (14, 55, 34)])
        ball = sy.switch((3, sy.at(a09)), (11, nowhere), (15, sy.at(b04)))
        sc = sy.Scene(14).player('A09', 'A', a09).player('B04', 'B', b04).player('B12', 'B', b12).ball(ball)
        T, out, post, path, M, aux, bst = pv.analyse(sc.result())
        hidden = codes_between(T, out, 5.0, 10.5)
        self.assertLessEqual(share(hidden, [A_, B_]), .5, 'eight seconds unseen with an opponent at the carrier: mostly not named')
        self.assertGreaterEqual(share(codes_between(T, out, 11.5, 14.0), B_), .9)
        ch = [c for c in out['changes'] if c['to'] == 'B']
        self.assertEqual(len(ch), 1)
        self.assertEqual(ch[0]['confidence'], 'uncertain', 'nobody saw the turnover: it is not a confident change')


class MatchLayer(unittest.TestCase):
    """Cases 7 (halftime) and 8 (a segment boundary) through fullmatch/possession_match.py's decode."""

    def scene(self, seconds=16):
        a = sy.lerp([(0, 30, 30), (seconds, 45, 32)])
        b = sy.lerp([(0, 40, 40), (seconds, 50, 38)])
        sc = sy.Scene(seconds).player('A-01', 'A', a).player('B-01', 'B', b)
        sc.ball(sy.at(a))
        return sc

    def test_7_halftime_is_left_out_and_teams_keep_their_letters(self):
        from fullmatch import possession_match as pm
        sc = sy.Scene(20)
        a1 = sy.lerp([(0, 30, 30), (20, 40, 30)])
        b1 = sy.lerp([(0, 70, 30), (20, 60, 30)])
        sc.player('A-01', 'A', a1).player('B-01', 'B', b1)
        # first half: A has it; halftime 8-12 s (nothing counts); second half: the teams changed ends, B has it
        sc.ball(sy.switch((8, sy.at(a1)), (12, nowhere), (21, sy.at(b1, -.4))))
        T = pv.table(sc.result())
        state = {'timeline': [{'t0': 0.0, 't1': 8.0, 'state': 'IN_PLAY'}, {'t0': 8.0, 't1': 12.0, 'state': 'HALFTIME'},
                              {'t0': 12.0, 't1': 20.0, 'state': 'IN_PLAY'}], 'events': []}
        periods = [{'index': 0, 'name': 'first half', 'start': 0.0, 'end': 8.0}, {'index': 1, 'name': 'second half', 'start': 12.0, 'end': 20.0}]
        rec, R, out, post = pm.decode_match(T, state=state, periods=periods)
        self.assertTrue(np.all(codes_between(R, out, 8.05, 11.95) == HALF))
        first, second = rec['stats']['first half'], rec['stats']['second half']
        self.assertGreater(first['seconds']['A'], 6.0)
        self.assertEqual(first['seconds']['B'], 0.0)
        self.assertGreater(second['seconds']['B'], 6.0)
        self.assertEqual(second['seconds']['A'], 0.0)
        self.assertAlmostEqual(rec['stats']['match']['seconds']['halftime'], 4.0, delta=.05)
        self.assertLessEqual(first['seconds']['halftime']+second['seconds']['halftime'], .041,
                             'the break lies between the halves (a frame at a boundary spills half a frame each side)')

    def test_8_a_segment_boundary_is_not_a_break(self):
        sc = self.scene()
        result = sc.result()
        cut = 200   # frame 200 = 8 s: two views, each owning its half of the frames
        raw = []
        T1 = pv.table(result, owned=(0, cut), seg=0, raw_ids=raw)
        T2 = pv.table(result, owned=(cut, 10**9), seg=1, raw_ids=raw)
        T = pv.concat([T1, T2])
        self.assertFalse(T['cut'][T1['n']], 'abutting views do not cut the timeline')
        from fullmatch import possession_match as pm
        rec, R, out, post = pm.decode_match(T)
        around = codes_between(R, out, 7.0, 9.0)
        self.assertTrue(np.all(around == A_), 'no unknown spell and no turnover at the boundary')
        self.assertEqual(out['changes'], [])


class BallObservation(unittest.TestCase):
    """The ball's own states (Part 3) and that possession never changes what the tracker saw."""

    def test_states_and_provenance(self):
        a = sy.lerp([(0, 30, 30), (10, 35, 30)])
        b = sy.lerp([(0, 50, 40), (10, 52, 40)])
        fl = sy.lerp([(5, 35.4, 30), (6, 49.6, 40)])
        ball = sy.switch((2, sy.at(a)), (2.6, sy.at(a)), (5, sy.at(a)), (6, fl), (7.5, nowhere), (11, sy.at(b, -.4)))
        state = lambda t: 'MISSING' if (2.0 <= t < 2.6 or 5.3 <= t < 5.7) else 'TRACKED'
        sc = sy.Scene(10).player('A1', 'A', a).player('B1', 'B', b).ball(ball, state, near=lambda t: 'A1' if 2.0 <= t < 2.6 else None)
        T = pv.resolve(pv.table(sc.result()))
        st, conf, prov = pv.ball_states(T)
        at = lambda s: int(st[int(round(s*25))])
        self.assertEqual(pv.BALL_STATES[at(1.0)], 'OBSERVED')
        self.assertEqual(pv.BALL_STATES[at(2.3)], 'OCCLUDED', 'hidden at the player who carries it')
        self.assertEqual(pv.BALL_STATES[at(5.5)], 'PREDICTED', 'interpolated in flight')
        self.assertEqual(pv.BALL_STATES[at(7.0)], 'UNKNOWN')
        self.assertEqual(pv.BALL_STATES[at(7.6)], 'REACQUIRING', 'seen again after 1.5 s unplaced')
        self.assertEqual(pv.BALL_STATES[at(9.0)], 'OBSERVED')
        bridged = T['bst'] == 2
        self.assertFalse(np.any(st[bridged] == 0), 'an interpolated position is never OBSERVED')
        self.assertTrue(np.all(prov[bridged] == pv.PROVENANCE.index('interpolation')))

    def test_possession_never_changes_the_ball(self):
        a = sy.lerp([(0, 30, 30), (8, 38, 31)])
        sc = sy.Scene(8).player('A1', 'A', a).ball(sy.switch((3, sy.at(a)), (4, nowhere), (9, sy.at(a))))
        result = sc.result()
        before = copy.deepcopy(result)
        pv.compute(result)
        self.assertEqual(json.dumps(result, sort_keys=True), json.dumps(before, sort_keys=True))


class Shares(unittest.TestCase):
    """Part 10: time weighting, every frame once, no 50/50 for lack of data."""

    def test_weights_tile_the_frames_and_a_gap_in_the_frames_is_nobodys_time(self):
        a = sy.lerp([(0, 30, 30), (40, 60, 30)])
        sc = sy.Scene(40).player('A1', 'A', a).ball(sy.at(a))
        result = sc.result()
        result['frames'] = [f for f in result['frames'] if not (500 <= f['frame'] < 600)]   # 4 s of frames missing
        rec = pv.compute(result)
        s = rec['stats']['match']
        self.assertAlmostEqual(s['live'], 36.0, delta=.1, msg='36 s of frames, not 40')
        tl = rec['timeline']
        self.assertAlmostEqual(sum(b-a for a, b in zip(tl['t0'], tl['t1'])), 36.0, delta=.1)

    def test_no_percentages_without_enough_resolved_time(self):
        a = sy.lerp([(0, 30, 30), (10, 35, 30)])
        sc = sy.Scene(10).player('A1', 'A', a).ball(sy.at(a))
        rec = pv.compute(sc.result())
        self.assertIsNone(rec['stats']['match']['percent'])
        self.assertTrue(rec['stats']['match']['insufficient'])

    def test_percentages_are_resolved_controlled_time_only(self):
        runs = [(0.0, 30.0, 0, 1), (30.0, 40.0, 4, 0), (40.0, 50.0, 3, 0), (50.0, 70.0, 1, 2), (70.0, 80.0, 6, 0)]
        s = pv.stats_from_runs(runs)
        self.assertEqual(s['percent'], {'A': 60.0, 'B': 40.0})
        self.assertEqual(s['live'], 70.0)
        self.assertAlmostEqual(s['coverage'], 60/70, places=3)
        self.assertEqual(s['observed']['A'], 30.0)
        self.assertEqual(s['inferred']['B'], 20.0)
        self.assertAlmostEqual(sum(s['shareOfLive'].values()), 1.0, delta=.003)   # each share rounded to 3 places
        r = pv.stats_from_runs(runs, lo=20.0, hi=60.0)
        self.assertEqual(r['seconds']['A'], 10.0)
        self.assertEqual(r['seconds']['B'], 10.0)
        self.assertIsNone(r['percent'], '20 s resolved: below the 30 s floor')


if __name__ == '__main__':
    unittest.main()
