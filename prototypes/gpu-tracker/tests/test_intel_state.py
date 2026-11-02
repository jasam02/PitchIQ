"""The match's state and events (fullmatch/gamestate.py) on synthetic matches (tests/intel_synth.py): the event cases
of the Full Match Intelligence spec (part 18): the ball over a touchline and a goal line, touching a line without
crossing it, missing while play goes on, still during play; corner setup and the corner taken; throw-in, goal kick,
free kick, penalty; a goal and its celebration and kick-off; a stoppage and its restart; halftime; a substitution
during a stoppage; an unknown restart; a camera cut without a stoppage. Then the masks and what each analysis counts,
the direction check and the coach's event corrections.

Run:  python -m unittest test_intel_state          (from this directory)
"""
import os
import sys
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
from fullmatch import gamestate as gs, halftime as ht, persons  # noqa: E402
import intel_synth as S  # noqa: E402


def match(seconds=120.0, **kw):
    m = S.Match(seconds=seconds, **kw)
    m.team('A')
    m.team('B')
    return m


def state_at(result, t):
    return next((r['state'] for r in result['timeline'] if r['t0'] <= t < r['t1']), None)


def events(result, kind=None):
    """The events, without the kick-off every period starts with (law 8; a synthetic clip starts a period)."""
    return [e for e in result['events'] if (kind is None or e['type'] == kind) and not e.get('periodStart')]


def mask_at(result, t):
    return gs.mask_at(gs.masks(result), [t])[0]


class OutOfPlay(unittest.TestCase):
    def test_touchline_and_throw_in(self):
        m = match()
        S.live(m, 10, 60)
        S.live(m, 75, 130)
        S.ball_path(m, 55, 60, (40.0, 60.0), (40.0, 69.5))           # out over the near touchline
        S.ball_at(m, 72, 75.4, 40.0, 68.4)                           # placed on the line for the throw
        S.dead(m, 60, 75, 'out_of_play', 'B')
        r, _, _ = S.state_of(m)
        throw = events(r, 'THROW_IN')
        self.assertEqual(len(throw), 1)
        self.assertAlmostEqual(throw[0]['t'], 75.0, delta=.3)
        self.assertEqual(state_at(r, 64.0), 'OUT_OF_PLAY')
        self.assertEqual(state_at(r, 73.0), 'RESTART_PREPARATION')
        self.assertEqual(state_at(r, 90.0), 'IN_PLAY')

    def test_goal_line_and_goal_kick(self):
        m = match()
        S.live(m, 10, 60)
        S.live(m, 75, 130)
        S.ball_path(m, 56, 60, (95.0, 30.0), (106.5, 30.0))
        S.ball_at(m, 72, 75.4, 100.0, 30.0)                          # in the goal area
        S.dead(m, 60, 75, 'out_of_play', 'B')
        r, _, _ = S.state_of(m)
        self.assertEqual([e['type'] for e in events(r) if e['type'] in gs.RESTARTS], ['GOAL_KICK'])
        self.assertEqual(state_at(r, 64.0), 'OUT_OF_PLAY')

    def test_touching_the_line_is_not_out(self):
        m = match()
        S.live(m, 10, 130)
        S.ball_path(m, 50, 55, (40.0, 60.0), (40.0, 67.9))            # on the line, never wholly over it
        S.ball_path(m, 55, 60, (40.0, 67.9), (45.0, 55.0))
        r, _, _ = S.state_of(m)
        self.assertEqual(state_at(r, 56.0), 'IN_PLAY')
        self.assertFalse([e for e in events(r) if e['type'] in gs.RESTARTS])

    def test_a_missing_ball_is_not_a_stoppage(self):
        m = match()
        S.live(m, 10, 130)
        S.ball_path(m, 20, 40, (30.0, 30.0), (60.0, 40.0))            # the ball seen, then not for 20 s
        r, _, _ = S.state_of(m)
        for t in (45.0, 50.0, 55.0):
            self.assertEqual(state_at(r, t), 'IN_PLAY')
        self.assertFalse(events(r))

    def test_a_still_ball_in_play_is_not_a_stoppage(self):
        m = match()
        S.live(m, 10, 130)
        S.ball_at(m, 50, 55, 60.0, 30.0)                              # still 5 s, players running, not dead
        r, _, _ = S.state_of(m)
        self.assertEqual(state_at(r, 52.0), 'IN_PLAY')
        self.assertFalse(events(r))


class SetPieces(unittest.TestCase):
    def test_corner_setup_and_taken(self):
        m = match()
        S.live(m, 10, 60)
        S.live(m, 80, 130)
        S.ball_at(m, 70, 80.4, 104.6, .6)
        S.dead(m, 60, 80, 'out_of_play', 'A')
        r, _, _ = S.state_of(m)
        corner = events(r, 'CORNER_KICK')
        self.assertEqual(len(corner), 1)
        self.assertEqual(state_at(r, 75.0), 'RESTART_PREPARATION')
        self.assertEqual(mask_at(r, 75.0), 'SET_PIECE_PREPARATION')    # never an open-play formation
        self.assertEqual(mask_at(r, 84.0), 'SET_PIECE_EXECUTION')      # taken: live, counted for set pieces
        self.assertEqual(mask_at(r, 95.0), 'ACTIVE_OPEN_PLAY')

    def test_free_kick(self):
        m = match()
        S.live(m, 10, 60)
        S.live(m, 75, 130)
        S.ball_at(m, 70, 75.4, 70.0, 30.0)
        S.dead(m, 60, 75, 'stoppage', 'A')
        r, _, _ = S.state_of(m)
        self.assertEqual(len(events(r, 'FREE_KICK')), 1)
        self.assertEqual(len(events(r, 'FOUL_OR_WHISTLE_STOPPAGE')), 1)

    def test_penalty_kick(self):
        m = match()
        S.live(m, 10, 60)
        S.live(m, 85, 130)
        S.ball_at(m, 78, 85.4, 94.0, 34.0)
        S.dead(m, 60, 85, 'stoppage', 'A')
        r, _, _ = S.state_of(m)
        self.assertEqual([e['type'] for e in events(r) if e['type'] in gs.RESTARTS], ['PENALTY_KICK'])

    def test_unknown_restart_is_not_named(self):
        m = match()
        S.live(m, 10, 60)
        S.live(m, 75, 130)
        S.dead(m, 60, 75, 'stoppage')                                 # no ball placed: the kind is not seen
        r, _, _ = S.state_of(m)
        kinds = [e['type'] for e in events(r) if e['type'] in gs.RESTARTS]
        self.assertEqual(kinds, ['UNKNOWN_RESTART'])
        self.assertEqual(state_at(r, 65.0), 'OUT_OF_PLAY')             # confirmed by the restart


class Goals(unittest.TestCase):
    def test_goal_celebration_and_kick_off(self):
        m = match(seconds=160.0)
        S.live(m, 10, 70)
        S.ball_path(m, 66, 70, (85.0, 33.0), (104.5, 34.0))           # into the right goal: team A attacks right
        S.live(m, 70, 78, speed=3.5)                                  # celebrating
        S.kickoff_layout(m, 88, 100, 'B')
        S.live(m, 100, 160)
        S.dead(m, 72, 99, 'stoppage', 'B')
        r, _, _ = S.state_of(m)
        goal, kick = events(r, 'GOAL'), [e for e in events(r, 'KICK_OFF') if not e.get('periodStart')]
        self.assertEqual(len(goal), 1)
        self.assertEqual(goal[0]['team'], 'A')
        self.assertAlmostEqual(goal[0]['t'], 69.6, delta=1.0)
        self.assertIsNone(goal[0]['scorer'])                           # never invented
        self.assertEqual(len(kick), 1)
        self.assertEqual(kick[0]['team'], 'B')
        for t in (74.0, 82.0, 95.0):                                   # the celebration is no open play
            self.assertNotEqual(mask_at(r, t), 'ACTIVE_OPEN_PLAY')
        self.assertEqual(mask_at(r, 120.0), 'ACTIVE_OPEN_PLAY')

    def test_the_goal_time_is_when_the_ball_went_in(self):
        # struck from 25 m at 25 m/s, in the net for three seconds, then picked out: the last sight of it in the goal
        # mouth is 3 s after it went in
        m = match(seconds=160.0)
        S.live(m, 10, 68)
        S.ball_path(m, 66, 68, (60.0, 34.0), (80.0, 34.0))
        S.ball_path(m, 68, 69, (80.0, 34.0), (105.0, 34.0))
        S.ball_path(m, 69, 72, (105.5, 34.5), (105.5, 34.5))
        S.ball_path(m, 72, 73, (105.5, 34.5), (102.0, 34.0))
        S.live(m, 70, 78, speed=3.5)                                  # celebrating
        S.kickoff_layout(m, 88, 100, 'B')
        S.live(m, 100, 160)
        S.dead(m, 72, 99, 'stoppage', 'B')
        r, _, _ = S.state_of(m)
        goal = events(r, 'GOAL')
        self.assertEqual(len(goal), 1)
        self.assertAlmostEqual(goal[0]['t'], 69.0, delta=.25)
        self.assertTrue(any(e.startswith('it went in at about 69 s') for e in goal[0]['evidence']), goal[0]['evidence'])

    def penalty_after_a_goal_kick(self, penalty=True):
        # the ball in the right goal mouth (a goal kick) at 60 s, the restart at 70 s and 10 s of play (too short for the
        # 'play went on' run), the foul at 80 s; the penalty set 108-115 s and scored, the ball never seen in the net
        m = match(seconds=220.0)
        s = m.scene
        S.live(m, 10, 60)
        S.ball_at(m, 59, 61, 103.0, 34.0)
        S.dead(m, 61, 70, 'stoppage', 'B')
        S.live(m, 70, 80)
        if penalty:
            taker = next(i for i, p in enumerate(s.people) if p['team'] == 'A' and p['role'] != 'GOALKEEPER')
            keeper = next(i for i, p in enumerate(s.people) if p['team'] == 'B' and p['role'] == 'GOALKEEPER')
            runner = next(i for i, p in enumerate(s.people) if p['team'] == 'A' and p['role'] != 'GOALKEEPER' and i != taker)
            s.pos[taker][s.k(108):s.k(115.2)] = [105.0-11.0-3.0, 34.0]      # 3 m behind the mark
            s.pos[keeper][s.k(108):s.k(116)] = [104.5, 34.0]                 # on the goal line
            s.pos[runner][s.k(115.2):s.k(117)] = [97.0, 40.0]                # into the box at the kick: the layout breaks
            waiting = [i for i, p in enumerate(s.people) if p['team'] == 'A' and p['role'] != 'GOALKEEPER' and i not in (taker, runner)][:3]
            for i, place in zip(waiting, ([86.0, 22.0], [86.0, 46.0], [84.0, 34.0])):
                s.pos[i][s.k(100):s.k(130)] = place                          # attackers waiting outside the box and the arc
        S.dead(m, 80, 139, 'stoppage', 'B')
        S.kickoff_layout(m, 130, 140, 'B')
        S.live(m, 140, 220)
        r, _, _ = S.state_of(m)
        return events(r, 'GOAL')

    def test_a_penalty_after_the_ball_was_in_a_goal_mouth_is_the_goal(self):
        goal = self.penalty_after_a_goal_kick()
        self.assertEqual(len(goal), 1)
        self.assertAlmostEqual(goal[0]['t'], 115.0, delta=.3)
        self.assertEqual((goal[0]['team'], goal[0]['timeUncertain']), ('A', False))
        self.assertTrue(any(e.startswith('a penalty at the right goal at about 115 s') for e in goal[0]['evidence']), goal[0]['evidence'])

    def test_no_goal_before_the_last_restart_of_play(self):
        goal = self.penalty_after_a_goal_kick(penalty=False)
        self.assertEqual(len(goal), 1)
        self.assertGreater(goal[0]['t'], 70.0, 'the ball in the goal mouth before the restart at 70 s was not the goal')
        self.assertTrue(goal[0]['timeUncertain'])

    def test_ball_near_the_goal_is_not_a_goal(self):
        m = match()
        S.live(m, 10, 130)
        S.ball_path(m, 50, 54, (90.0, 34.0), (104.0, 34.0))           # at the goal mouth, play goes on
        S.ball_path(m, 54, 60, (104.0, 34.0), (80.0, 40.0))
        r, _, _ = S.state_of(m)
        self.assertFalse(events(r, 'GOAL'))


class Halftime(unittest.TestCase):
    def test_halftime_interval_and_masks(self):
        m = match(seconds=200.0, halftime=(90.0, 110.0))
        S.live(m, 10, 90)
        S.live(m, 110, 210)
        data, index = m.build()
        settings = dict(ht.defaults(index), mode='break', firstHalfEnd=90.0, secondHalfStart=110.0)   # the break shown
        r, _, _ = S.state_of(m, settings)
        self.assertEqual(state_at(r, 100.0), 'HALFTIME')
        self.assertEqual(mask_at(r, 100.0), 'HALFTIME')
        names = gs.mask_at(gs.masks(r), [50.0, 100.0, 150.0])
        self.assertEqual(gs.eligible(names, 'open_play').tolist(), [True, False, True])
        self.assertEqual(gs.eligible(names, 'movement').tolist(), [True, False, True])

    def test_wrong_direction_setting_is_reported(self):
        m = match(seconds=120.0)
        S.kickoff_layout(m, 10, 20, 'A')
        S.live(m, 21, 120)
        data, index = m.build()
        right = ht.defaults(index)
        wrong = dict(right, teamAFirstHalf='left' if right['teamAFirstHalf'] == 'right' else 'right')
        ok, _, _ = S.state_of(m, right)
        bad, _, _ = S.state_of(m, wrong)
        self.assertEqual((ok['directionCheck']['agree'], ok['directionCheck']['disagree']), (1, 0))
        self.assertEqual(bad['directionCheck']['disagree'], 1)
        self.assertIn('note', bad['directionCheck'])


class Substitution(unittest.TestCase):
    def test_substitution_during_a_stoppage(self):
        m = S.Match(seconds=200.0)
        a = m.team('A')
        m.team('B')
        s = m.scene
        S.live(m, 10, 100)
        S.live(m, 125, 200)
        rm = a['RM']
        s.pos[rm][s.k(100.0):s.k(110.0)] = [55.0, 66.0]               # walks off by the halfway line
        s.hide(rm, 110.0, 1e9)
        new = m.person('A', 54.0, 66.5, 114.0, 120.0)                 # comes on there
        s.pos[new][s.k(120.0):] = s.pos[a['RCM']][s.k(120.0):]+np.array([2.0, 15.0])
        S.dead(m, 100, 124, 'stoppage', 'A')
        data, index = m.build()
        auto = persons.build_from(index, data, m.galleries(data), {})
        people = persons.resolve(auto, [])
        r, _, _ = S.state_of(m, people=people, places=people['places'])
        subs = events(r, 'SUBSTITUTION')
        self.assertEqual(len(subs), 1)
        self.assertEqual(subs[0]['team'], 'A')
        self.assertNotEqual(subs[0]['outPerson'], subs[0]['inPerson'])
        self.assertEqual(subs[0]['confidence'], 'low')                  # a candidate for the coach


class Camera(unittest.TestCase):
    def test_a_camera_cut_is_not_a_stoppage(self):
        m = match()
        S.live(m, 10, 130)
        for pid in range(len(m.scene.people)):
            m.scene.hide(pid, 60.0, 65.0)                              # no picture of the pitch for 5 s, no dead spell
        r, _, _ = S.state_of(m)
        self.assertEqual(state_at(r, 62.0), 'UNKNOWN')
        self.assertFalse([e for e in events(r) if e['type'] in gs.RESTARTS])

    def test_stoppage_without_restart_is_only_possible(self):
        m = match()
        S.live(m, 10, 60)
        S.dead(m, 60, 75, 'stoppage')                                   # dead, but nobody speeds up after it
        r, _, _ = S.state_of(m)
        self.assertEqual(state_at(r, 65.0), 'POSSIBLE_STOPPAGE')
        self.assertEqual(mask_at(r, 65.0), 'UNKNOWN')


class Corrections(unittest.TestCase):
    def test_confirm_retype_move_remove_add_state(self):
        m = match()
        S.live(m, 10, 60)
        S.live(m, 75, 130)
        S.ball_at(m, 70, 75.4, 70.0, 30.0)
        S.dead(m, 60, 75, 'stoppage', 'A')
        r, _, _ = S.state_of(m)
        fk = events(r, 'FREE_KICK')[0]
        items = [{'kind': 'retype', 'id': fk['id'], 'type': 'THROW_IN'},
                 {'kind': 'move', 'id': fk['id'], 't': 74.0},
                 {'kind': 'add', 'event': {'id': 'user-1', 'type': 'GOAL', 't': 100.0, 't0': 100.0, 't1': 100.0,
                                           'team': 'B', 'confidence': 'confirmed', 'evidence': ['added by you'],
                                           'source': 'you'}},
                 {'kind': 'state', 't0': 100.0, 't1': 110.0, 'state': 'OUT_OF_PLAY'}]
        out = gs.apply_corrections(r, items)
        e = next(x for x in out['events'] if x['id'] == fk['id'])
        self.assertEqual((e['type'], e['t'], e['status']), ('THROW_IN', 74.0, 'confirmed'))
        self.assertIn('user-1', {x['id'] for x in out['events']})
        self.assertEqual(state_at(out, 105.0), 'OUT_OF_PLAY')
        gone = gs.apply_corrections(r, [{'kind': 'remove', 'id': fk['id']}])
        self.assertEqual(next(x for x in gone['events'] if x['id'] == fk['id'])['status'], 'removed')
        self.assertNotIn('FREE_KICK', gone['stats']['events'])
        nothing = gs.apply_corrections(r, [])
        self.assertEqual(len(nothing['events']), len(r['events']))      # the automatic result is never lost


if __name__ == '__main__':
    unittest.main()
