"""The key-moments detectors (soccer/moments/detectors) on synthetic scenes (moments_synth.Scene), including the spec's
cases where NO criticism may come out:

- a full back who advances while a midfielder rotates into the space behind: a positive moment, never an improvement;
  the same advance with nobody covering and an opponent in the channel: 'exposed_channel';
- the same exposed channel out of the camera's view, or on uncalibrated frames: nothing claimed;
- a defender deeper than the line because they mark the deepest attacker: no criticism; deep with a FREE runner in front
  and the ball in front of the line: 'dropped_off';
- a striker pressed and alone: no criticism (often the plan); a central midfielder in the same spot: a review;
- a gap in the back line with an attacker in it: flagged only when the ball's position is known;
- pressure with cover (positive), two pressing one with a free opponent (improvement), the same with the free
  opponent's lane screened (not free), pressure without cover near the opponents' goal (never criticised);
- the roles of a 4-4-2 inferred from pairwise positions, left and right per team direction.

Run: python -m unittest test_moments_detectors        (from this directory)
"""
import unittest

import moments_synth as ms
from soccer.moments import engine as en
from soccer.moments.detectors import back_line, pressure, support, wide_cover


def kinds(eps, team=None, category=None):
    return [e['kind'] for e in eps if (team is None or e['team'] == team) and (category is None or e['category'] == category)]


def fullback_scene(cover, view_rect=None, calibrated=True):
    """A attacks right. B has the ball for 40 s (A defends: roles), then A has it with the left back, who runs 23 m up
    the left. B's right midfielder moves into the channel behind; with `cover` A's left central midfielder rotates there."""
    s = ms.Scene(80, 'A')
    a, b = s.add_team('A'), s.add_team('B')
    s.possession('B', 10, 50, carrier=b['RCM'])
    s.ball_with(b['RCM'], 10, 50)
    s.possession('A', 50, 90, carrier=a['LB'])
    s.move(a['LB'], 50, 56, 55, 26)
    s.ball_with(a['LB'], 50, 90)
    s.move(b['RM'], 50, 56, 105-34, -21)                 # B's coordinates: depth 71 from B's goal = 34 from A's
    if cover:
        s.move(a['LCM'], 50, 56, 32, 20)
    if view_rect:
        s.see_only(50, 90, *view_rect)
    if not calibrated:
        s.calibrated[:] = False
    return s, a, b


class WideCoverTests(unittest.TestCase):
    def test_a_covered_advance_is_positive_never_an_improvement(self):
        s, a, b = fullback_scene(cover=True)
        eps = wide_cover.detect(en.Engine(s.build()))
        self.assertNotIn('exposed_channel', kinds(eps, 'A'))
        pos = [e for e in eps if e['team'] == 'A' and e['category'] == 'positive']
        self.assertTrue(pos, kinds(eps))
        self.assertEqual(pos[0]['kind'], 'rotated_to_cover')
        self.assertEqual(pos[0]['players']['focus'], [a['LB']])
        self.assertEqual(pos[0]['players']['support'], [a['LCM']])

    def test_an_uncovered_advance_with_an_opponent_in_the_channel_is_an_improvement(self):
        s, a, b = fullback_scene(cover=False)
        eps = wide_cover.detect(en.Engine(s.build()))
        imp = [e for e in eps if e['team'] == 'A' and e['kind'] == 'exposed_channel']
        self.assertEqual(len(imp), 1, kinds(eps))
        self.assertEqual(imp[0]['players']['focus'], [a['LB']])
        self.assertEqual(imp[0]['players']['opponents'], [b['RM']])
        self.assertEqual(imp[0]['phase']['state'], 'own')
        self.assertGreaterEqual(imp[0]['measures']['advanceM'], wide_cover.ADVANCE_M)

    def test_the_channel_out_of_view_is_not_judged(self):
        s, a, b = fullback_scene(cover=False, view_rect=(0, 40, 105, 68))     # the camera shows only the near side
        eps = wide_cover.detect(en.Engine(s.build()))
        self.assertNotIn('exposed_channel', kinds(eps))

    def test_uncalibrated_frames_give_nothing(self):
        s, a, b = fullback_scene(cover=False, calibrated=False)
        self.assertEqual(wide_cover.detect(en.Engine(s.build())), [])

    def test_the_player_on_the_ball_is_not_the_cover(self):
        # the centre-back carrying the ball in the channel neither covers it nor makes it 'exposed'
        s = ms.Scene(80, 'A')
        a, b = s.add_team('A'), s.add_team('B')
        s.possession('B', 10, 50, carrier=b['RCM'])
        s.ball_with(b['RCM'], 10, 50)
        s.possession('A', 50, 90, carrier=a['LCB'])
        s.move(a['LB'], 50, 56, 55, 26)
        s.move(a['LCB'], 50, 56, 33, 20)
        s.ball_with(a['LCB'], 50, 90)
        s.move(b['RM'], 50, 56, 105-38, -19)
        eps = wide_cover.detect(en.Engine(s.build()))
        self.assertEqual([e for e in eps if e['team'] == 'A' and e['category'] in ('positive', 'improvement')], [])


def back_scene():
    """A attacks right and defends from 10 to 90 s (B on the ball, with B's left central midfielder 50 m from A's goal)."""
    s = ms.Scene(80, 'A')
    a, b = s.add_team('A'), s.add_team('B')
    s.possession('B', 10, 90, carrier=b['LCM'])
    s.ball_with(b['LCM'], 10, 90)
    return s, a, b


class BackLineTests(unittest.TestCase):
    def test_a_defender_marking_the_deepest_attacker_is_not_criticised(self):
        s, a, b = back_scene()
        s.move(a['LCB'], 50, 54, 22, 8)
        s.move(b['RS'], 50, 54, 105-23.5, -8)          # 1.5 m in front of the deep centre-back, marked
        self.assertNotIn('dropped_off', kinds(back_line.detect(en.Engine(s.build()))))

    def test_a_deep_defender_with_a_free_runner_in_front_is_flagged(self):
        s, a, b = back_scene()
        s.move(a['LCB'], 50, 54, 22, 8)
        s.place(b['RS'], 54, 105-40, -14)
        s.move(b['RS'], 54, 59, 105-23, -14)          # runs 17 m towards A's goal (3.4 m/s), ending 6 m from the deep defender
        eps = back_line.detect(en.Engine(s.build()))
        self.assertIn('dropped_off', kinds(eps, 'A'), kinds(eps))
        e = [e for e in eps if e['kind'] == 'dropped_off'][0]
        self.assertEqual(e['players']['focus'], [a['LCB']])
        self.assertEqual(e['players']['opponents'], [b['RS']])
        self.assertIn(e['interpretation'], ('plausible', 'uncertain'))       # never 'clear' without the pass moment

    def gap_scene(self, with_ball):
        s, a, b = back_scene()
        s.place(a['LCB'], 50, 30, 12)
        s.place(a['RCB'], 50, 30, -10)
        s.place(b['LS'], 50, 105-31, -1)              # in the 22 m gap between the centre-backs, on the line
        if not with_ball:
            s.ball[:] = float('nan')
            s.ball_state[:] = 0
            s.spans = [{**sp, 'carrier': None} for sp in s.spans]
        return s, a, b

    def test_a_gap_with_an_attacker_needs_the_ball(self):
        s, a, b = self.gap_scene(with_ball=True)
        eps = back_line.detect(en.Engine(s.build()))
        gaps = [e for e in eps if e['kind'] == 'gap_attacker']
        self.assertEqual(len(gaps), 1, kinds(eps))
        self.assertEqual(sorted(gaps[0]['players']['focus']), sorted([a['LCB'], a['RCB']]))
        s, a, b = self.gap_scene(with_ball=False)
        self.assertNotIn('gap_attacker', kinds(back_line.detect(en.Engine(s.build()))))

    def test_the_line_stepping_up_together_is_positive(self):
        s, a, b = back_scene()
        for role in ('LB', 'LCB', 'RCB', 'RB'):
            d, l = ms.F442[role]
            s.move(a[role], 60, 63, 30+9, l)
        eps = back_line.detect(en.Engine(s.build()))
        ups = [e for e in eps if e['kind'] == 'stepped_up_together' and e['team'] == 'A']
        self.assertEqual(len(ups), 1, kinds(eps))
        self.assertEqual(ups[0]['category'], 'positive')
        self.assertGreaterEqual(ups[0]['measures']['riseM'], back_line.STEP_UP_M)


class SupportTests(unittest.TestCase):
    def isolated(self, carrier_role):
        s = ms.Scene(80, 'A')
        a, b = s.add_team('A'), s.add_team('B')
        s.possession('B', 10, 50, carrier=b['RCM'])
        s.ball_with(b['RCM'], 10, 50)
        c = a[carrier_role]
        s.possession('A', 50, 90, carrier=c)
        s.place(c, 50, 40, -30)                        # alone near the right touchline
        s.ball_with(c, 50, 90)
        for role in ('RB', 'RM', 'RCM', 'RS'):
            if role != carrier_role:
                s.place(a[role], 50, ms.F442[role][0], 10)    # the right side's support moved away
        s.place(b['LM'], 50, 105-41.5, 30.5)            # B's left midfielder presses at 1.6 m
        return s, a, b

    def test_a_pressed_midfielder_without_options_is_a_review(self):
        s, a, b = self.isolated('RCM')
        eps = support.detect(en.Engine(s.build()))
        self.assertIn('carrier_isolated', kinds(eps, 'A'), kinds(eps))

    def test_an_isolated_striker_is_not_criticised(self):
        s, a, b = self.isolated('RS')
        self.assertNotIn('carrier_isolated', kinds(support.detect(en.Engine(s.build())), 'A'))


class PressureTests(unittest.TestCase):
    def scene(self, carrier_depth=40.0):
        s, a, b = back_scene()
        s.spans = []
        s.possession('B', 10, 90, carrier=b['RCM'])
        s.place(b['RCM'], 50, 105-carrier_depth, 0)
        s.ball_with(b['RCM'], 10, 90)
        s.place(a['LCM'], 50, carrier_depth+1.5, 1)    # presses at about 1.8 m
        return s, a, b

    def test_pressure_with_cover_is_positive(self):
        s, a, b = self.scene()
        s.place(a['RCB'], 50, 32, -2)                  # 8 m goal-side of the carrier
        eps = pressure.detect(en.Engine(s.build()))
        pc = [e for e in eps if e['kind'] == 'press_and_cover' and e['team'] == 'A']
        self.assertTrue(pc, kinds(eps))
        self.assertEqual(pc[0]['players']['focus'], [a['LCM']])
        self.assertEqual(pc[0]['players']['support'], [a['RCB']])

    def test_two_pressing_one_with_a_free_opponent_is_an_improvement_unless_screened(self):
        s, a, b = self.scene()
        s.place(a['RCB'], 50, 32, -2)
        s.place(a['RCM'], 50, 40, 2.5)                # a second presser at 2.5 m, on the other side of the lane
        s.place(b['LS'], 50, 105-36, 14)               # B's striker free 14 m away, lane open
        s.place(a['LCB'], 50, 22, 20)                   # (A's left centre-back out of the way)
        eps = pressure.detect(en.Engine(s.build()))
        self.assertIn('double_press_free', kinds(eps, 'A'), kinds(eps))
        s2, a2, b2 = self.scene()
        s2.place(a2['RCB'], 50, 32, -2)
        s2.place(a2['RCM'], 50, 40, 2.5)
        s2.place(b2['LS'], 50, 105-36, 14)
        s2.place(a2['LCB'], 50, 38, -7)                  # standing in the lane to the 'free' striker
        self.assertNotIn('double_press_free', kinds(pressure.detect(en.Engine(s2.build()))))

    def test_pressure_without_cover_far_from_own_goal_is_not_criticised(self):
        s, a, b = self.scene(carrier_depth=75.0)
        for role in ('LB', 'LCB', 'RCB', 'RB', 'RCM', 'LM', 'RM'):
            d, l = ms.F442[role]
            s.place(a[role], 50, d, l)
        self.assertNotIn('press_no_cover', kinds(pressure.detect(en.Engine(s.build()))))


class RolesTests(unittest.TestCase):
    def test_a_442_is_named_line_by_line_and_side_by_side_for_both_directions(self):
        s, a, b = back_scene()
        e = en.Engine(s.build())
        k = s.k(40)
        for team, roles in (('A', a), ('B', b)):
            got = {name: (e.role(team, pid, k) or {}).get('role') for name, pid in roles.items() if name != 'GK'}
            self.assertEqual(got, {'LB': 'LB', 'LCB': 'LCB', 'RCB': 'RCB', 'RB': 'RB', 'LM': 'LM', 'LCM': 'CM', 'RCM': 'CM',
                                   'RM': 'RM', 'LS': 'LS', 'RS': 'RS'}, team)

    def test_roles_survive_a_partial_view(self):
        # the camera shows only A's defensive half for the whole scene: the pairwise solve still orders the back four
        s, a, b = back_scene()
        s.see_only(10, 90, 0, -5, 60, 73)
        for pid in (a['LS'], a['RS']):
            s.hide(pid, 10, 60)
        e = en.Engine(s.build())
        k = s.k(40)
        got = {name: (e.role('A', a[name], k) or {}).get('line') for name in ('LB', 'LCB', 'RCB', 'RB')}
        self.assertEqual(set(got.values()), {'D'}, got)


if __name__ == '__main__':
    unittest.main()
