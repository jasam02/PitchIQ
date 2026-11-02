"""The goal analysis (soccer/moments/goals.py) on synthetic scenes (moments_synth.Scene), with hand-made match states
(GOAL events, the timeline's stops, restart events) and possession v2 records:

- a counter-attack: the ball won in the middle third, carried 50 m and struck into the goal; the start is the gain, the
  end the strike (the goal time estimated from it), the scoring side gets where it won the ball and how far it moved
  it, the conceding side where it lost it; three attackers against the keeper in the box come out for both sides, and
  the keeper 10 m off the line when the ball was struck; the state's goal time 2 s late with the conceding team on
  the ball after the strike (picking it out of the net): that possession is not the start;
- the conceding team pushed up with two players goal-side: 'only 2 of the 10'; the same with three of them out of view,
  or with the attack starting near goal: nothing claimed (the camera does not show everyone, nobody can be goal-side
  near goal);
- the ball carrier not tracked (no attacker in the box): no box finding;
- a corner (set piece), a penalty (the ball still on the mark before the restart), a goal two seconds after a restart
  that ended a long stop with no ball placed (a penalty or a direct free kick, not told apart);
- a penalty told by the players' layout (law 14) with no ball, the goal time after the kick, the play that won it
  (the scoring team's last loss of the ball before it, in the penalty area); the layout held too short, or a defender
  inside the penalty area: no penalty;
- a goal with no positions: a note, no tracking findings; a goal whose scorer is unknown: not analysed;
- the conceding team on the ball one second before the goal after the scoring team's last gain: the start is that loss;
  the conceding team's possession followed by the scoring team's with no change recorded: a turnover there;
- a goal with nothing placed before it: kind 'unknown', the scoring team's last possession named;
- removed goals are left out, the signature follows the coach's corrections, for_team() splits scored and conceded,
  assemble() carries the section.

Run: python -m unittest test_moments_goals        (from this directory)
"""
import unittest

import moments_synth as ms
from soccer.moments import engine as en
from soccer.moments import goals as gl
from soccer.moments import language as lg
from soccer.moments import report as rp


def rec2(runs, changes=()):
    """A possession v2 record from (t0, t1, team or None, possessor id or None) runs."""
    people, idx = [], {}
    cols = {k: [] for k in ('t0', 't1', 'state', 'designation', 'confidence', 'possessor', 'ballSeen', 'decoded')}
    for t0, t1, team, who in runs:
        cols['t0'].append(float(t0))
        cols['t1'].append(float(t1))
        cols['state'].append({'A': 0, 'B': 1, None: 4}[team])
        cols['designation'].append(1 if team else 0)
        cols['confidence'].append(90 if team else -1)
        if who is not None and who not in idx:
            idx[who] = len(people)
            people.append({'id': who, 'label': who})
        cols['possessor'].append(idx[who] if who is not None else -1)
        cols['ballSeen'].append(100)
        cols['decoded'].append(-1)
    cols['people'] = people
    return {'timeline': cols, 'changes': list(changes)}


def change(frm, to, lost_at, won_at, lost_by=None, won_by=None):
    return {'from': frm, 'to': to, 'kind': 'open', 'lostAt': lost_at, 'wonAt': won_at, 'supported': True,
            'confidence': 'high', 'lostBy': lost_by, 'lostByLabel': lost_by, 'wonBy': won_by, 'wonByLabel': won_by}


def match_state(goals, timeline=None, events=()):
    """A match state: GOAL events [(t, team)] (a team of None: unknown), the timeline (IN_PLAY throughout by default)."""
    ev = [{'type': 'GOAL', 't': float(t), 'team': team, 'confidence': 'high', 'status': 'detected'} for t, team in goals]
    return {'events': ev + list(events), 'timeline': timeline if timeline is not None else
            [{'t0': ms.T0, 't1': ms.T0+120.0, 'state': 'IN_PLAY'}]}


def analyse(s, state, record=None, eps=()):
    data = s.build()
    engine = en.Engine(data)
    return gl.analyse(data, engine, list(eps), state, record), data


def words(goal, data, side):
    """The finding texts of one side ('scored' / 'conceded') of a goal."""
    names = lg.Names(data, goal.team, {'A': 'white', 'B': 'red'}, {})
    return [f['text'] for f in goal.record(names, lg.clock)[side]]


def counter_attack(pushed_up=False, hidden=(), carrier_tracked=True, keeper_out=False, won=(50, 0)):
    """A attacks right. B keeps the ball until 40 s; A's left central midfielder wins it in the middle of the pitch (A
    depth 50) and carries it 50 m in 6 s; A's strikers arrive in the box; the ball is in the goal at 47 s.
    pushed_up: B's outfield players high up the pitch when the ball is lost, two centre backs still goal-side."""
    s = ms.Scene(60, 'A')
    a, b = s.add_team('A'), s.add_team('B')
    s.place(a['LCM'], 39, won[0], won[1])
    s.place(b['RCM'], 39, 105-won[0], -won[1])           # B's coordinates: 55 m from B's goal is A's 50, B's right A's left
    if pushed_up:
        for role in ('LB', 'RB', 'LM', 'LCM', 'RM', 'LS', 'RS'):
            s.place(b[role], 39, 70+ms.F442[role][0]/4, ms.F442[role][1])
        s.place(b['LCB'], 39, 40, 8)
        s.place(b['RCB'], 39, 40, -8)
    s.possession('B', 10, 40, carrier=b['RCM'])
    s.ball_with(b['RCM'], 10, 40)
    s.possession('A', 40, 47, carrier=a['LCM'])
    s.move(a['LCM'], 40, 46, 100, 0)
    s.ball_with(a['LCM'], 40, 46.4)
    s.ball[s.k(46.4):s.k(48)] = [105.5, 34.0]
    s.move(a['LS'], 40, 46, 92, 6)
    s.move(a['RS'], 40, 46, 92, -6)
    if not carrier_tracked:
        s.hide(a['LCM'], 44, 48)
        s.hide(a['LS'], 40, 48)
        s.hide(a['RS'], 40, 48)
    for role in hidden:
        s.hide(b[role], 38, 48)
    if keeper_out:
        s.move(b['GK'], 43, 46, 10, 0)                  # B's keeper 10 m off the line when the ball is struck
    pid = {p: s.people[i]['id'] for p, i in (('a7', a['LCM']), ('b8', b['RCM']))}
    record = rec2([(10, 40, 'B', pid['b8']), (40, 47, 'A', pid['a7'])],
                  [change('B', 'A', 39.8, 40.0, lost_by=pid['b8'], won_by=pid['a7'])])
    return s, record, pid


class CounterAttackTests(unittest.TestCase):
    def test_the_gain_starts_the_attack_and_both_sides_are_told(self):
        s, record, pid = counter_attack()
        (g,), data = analyse(s, match_state([(47.0, 'A')]), record)
        self.assertEqual((g.kind, g.start_kind, g.start), ('open_play', 'turnover', 39.8))   # where B lost it
        self.assertAlmostEqual(g.t_strike, 46.2, places=3)        # the ball's last fast move towards goal (24.5 m/s)
        self.assertAlmostEqual(g.t_in, 46.4, places=3)            # it was in the goal at the next sample
        self.assertEqual(g.quality['placed'], 1.0)
        self.assertEqual((g.facts['startZone']['third'], g.facts['startZone']['channel']), ('middle third', 'centre'))
        self.assertGreater(g.facts['progressionM'], 45)
        self.assertEqual(g.facts['lastOnBall']['id'], pid['a7'])
        self.assertEqual(g.facts['box']['attackers'], 3)
        self.assertEqual(g.facts['box']['defenders'], 1)
        scored, conceded = words(g, data, 'scored'), words(g, data, 'conceded')
        self.assertTrue(any(t.startswith('Won the ball in the middle third (centre)') and 'scored 7 s later' in t for t in scored), scored)
        self.assertTrue(any(t.startswith('Moved the ball about') for t in scored), scored)
        self.assertTrue(any(t.startswith('3 Team A (white) players in and around the box') and '1 defender (keeper' in t
                            for t in scored), scored)
        self.assertTrue(any(t.startswith('Lost the ball in the middle third (centre), about 54 m from its own goal') and f"(red {pid['b8']} last had it)" in t
                            for t in conceded), conceded)
        self.assertTrue(any(t.startswith('Outnumbered or matched in the box: 1 defender') for t in conceded), conceded)
        # B's back four and three midfielders goal-side (7 of 10): no 'only N' claim
        self.assertEqual(g.facts['goalSideStart']['goalSide'], 7)
        self.assertFalse(any(t.startswith('Only ') for t in conceded), conceded)

    def test_each_side_hears_the_place_in_its_own_direction(self):
        s, record, pid = counter_attack(won=(75, 15))       # A's attacking third, on A's left: B's own third, on B's right
        (g,), data = analyse(s, match_state([(47.0, 'A')]), record)
        self.assertTrue(any(t.startswith('Won the ball in the attacking third (left), about 29 m from goal') for t in words(g, data, 'scored')))
        self.assertTrue(any(t.startswith('Lost the ball in its own third (right), about 29 m from its own goal') for t in words(g, data, 'conceded')),
                        words(g, data, 'conceded'))

    def test_a_record_carries_the_window_to_watch_and_the_sources(self):
        s, record, _ = counter_attack()
        (g,), data = analyse(s, match_state([(47.0, 'A')]), record)
        r = g.record(lg.Names(data, 'A', {}, {}), lg.clock)
        self.assertEqual((r['id'], r['team'], r['conceding'], r['kind'], r['startKind']), ('goal-47.0', 'A', 'B', 'open_play', 'turnover'))
        self.assertEqual(r['watch'], {'t0': 36.8, 't1': 49.0})
        self.assertEqual(r['tEstimated'], 46.4)
        for f in r['scored']+r['conceded']:
            self.assertIn(f['source'], ('tracking', 'possession estimate', 'match state', 'key moment'))
            self.assertIn(f['sure'], ('measured', 'estimated', 'fairly sure', 'clear', 'plausible', 'uncertain'))
            self.assertTrue(f['t'] is None or 36.8 <= f['t'] <= 49.0, f)
        self.assertNotIn('pid', str(r['facts']))

    def test_a_pushed_up_defence_is_named_only_when_enough_of_it_is_in_view(self):
        s, record, _ = counter_attack(pushed_up=True)
        (g,), data = analyse(s, match_state([(47.0, 'A')]), record)
        self.assertEqual((g.facts['goalSideStart']['goalSide'], g.facts['goalSideStart']['outfield']), (2, 10))
        conceded = words(g, data, 'conceded')
        self.assertTrue(any(t.startswith('Only 2 of the 10 outfield players in view were between the ball and goal') for t in conceded),
                        conceded)
        s, record, _ = counter_attack(pushed_up=True, hidden=('LM', 'RM', 'LS'))
        (g,), data = analyse(s, match_state([(47.0, 'A')]), record)
        self.assertEqual(g.facts['goalSideStart']['outfield'], 7)
        self.assertFalse(any(t.startswith('Only ') for t in words(g, data, 'conceded')))

    def test_no_goal_side_claim_when_the_attack_starts_near_goal(self):
        s, record, _ = counter_attack(pushed_up=True)
        late = rec2([(10, 45, 'B', None), (45, 47, 'A', None)], [change('B', 'A', 44.8, 45.0)])
        (g,), data = analyse(s, match_state([(47.0, 'A')]), late)
        self.assertEqual(g.start, 44.8)
        self.assertLess(g.facts['goalSideStart']['ballFromGoalM'], gl.GOAL_SIDE_FROM_M)
        self.assertFalse(any(t.startswith('Only ') for t in words(g, data, 'conceded')))

    def test_the_keeper_off_the_line_at_the_strike_is_said(self):
        s, record, _ = counter_attack(keeper_out=True)
        (g,), data = analyse(s, match_state([(47.0, 'A')]), record)
        self.assertEqual(g.facts['strike']['keeperOffLineM'], 10.0)
        self.assertGreater(g.facts['strike']['nearestDefenderM'], 20)
        conceded = words(g, data, 'conceded')
        self.assertIn('The keeper was about 10 m off the goal line when the ball was struck towards goal.', conceded)
        self.assertTrue(any(t.startswith('Nobody closed the ball down: no outfield player within') for t in conceded), conceded)
        s, record, _ = counter_attack()
        (g,), data = analyse(s, match_state([(47.0, 'A')]), record)
        self.assertFalse(any('off the goal line' in t for t in words(g, data, 'conceded')), 'a keeper on the line is not mentioned')

    def test_play_after_the_strike_is_not_the_attack(self):
        # the state's goal time 2 s late (the ball last seen in the net) and B on the ball after the strike (picking it out)
        s, _, pid = counter_attack()
        late = rec2([(10, 40, 'B', pid['b8']), (40, 46.3, 'A', pid['a7']), (47.0, 48.0, 'B', pid['b8'])],
                    [change('B', 'A', 39.8, 40.0, lost_by=pid['b8'], won_by=pid['a7'])])
        (g,), data = analyse(s, match_state([(48.5, 'A')]), late)
        self.assertEqual((g.start_kind, g.start), ('turnover', 39.8))
        self.assertAlmostEqual(g.t_in, 46.4, places=3)
        self.assertEqual(g.facts['seconds'], 6.6)

    def test_a_ball_loose_long_before_it_was_won_starts_the_attack_when_won(self):
        s, _, pid = counter_attack()
        loose = rec2([(10, 34, 'B', pid['b8']), (40, 47, 'A', pid['a7'])], [change('B', 'A', 34.0, 40.0, lost_by=pid['b8'], won_by=pid['a7'])])
        (g,), data = analyse(s, match_state([(47.0, 'A')]), loose)
        self.assertEqual((g.start_kind, g.start), ('turnover', 40.0))

    def test_the_conceding_teams_possession_then_the_scoring_teams_is_a_turnover(self):
        s, _, pid = counter_attack()
        implied = rec2([(10, 39.5, 'B', pid['b8']), (40, 47, 'A', pid['a7'])])    # no change recorded between them
        (g,), data = analyse(s, match_state([(47.0, 'A')]), implied)
        self.assertEqual((g.start_kind, g.start), ('turnover', 39.5))
        self.assertTrue(g.facts['gain']['implied'])
        self.assertEqual((g.facts['gain']['lostBy'], g.facts['gain']['wonBy']), (pid['b8'], pid['a7']))
        self.assertTrue(any(t.startswith('Won the ball in the middle third') for t in words(g, data, 'scored')))
        self.assertTrue(any(f"{pid['b8']} last had it)" in t for t in words(g, data, 'conceded')))

    def test_an_untracked_ball_carrier_gives_no_box_count(self):
        s, record, _ = counter_attack(carrier_tracked=False)
        (g,), data = analyse(s, match_state([(47.0, 'A')]), record)
        self.assertEqual(g.facts['box']['attackers'], 0)
        for side in ('scored', 'conceded'):
            self.assertFalse(any('in and around the box' in t or 'in the box' in t for t in words(g, data, side)))


def set_piece_scene():
    s = ms.Scene(60, 'A')
    a, b = s.add_team('A'), s.add_team('B')
    for role in ('LS', 'RS', 'LCB'):
        s.place(a[role], 49, 96, ms.F442[role][1])
    for role in ('LCB', 'RCB'):
        s.place(b[role], 49, 6, ms.F442[role][1])
    return s, a, b


class RestartTests(unittest.TestCase):
    def test_a_corner(self):
        s, a, b = set_piece_scene()
        s.ball[s.k(50):s.k(55)] = [104.0, 1.0]
        timeline = [{'t0': 10.0, 't1': 40.0, 'state': 'IN_PLAY'}, {'t0': 40.0, 't1': 50.0, 'state': 'OUT_OF_PLAY'},
                    {'t0': 50.0, 't1': 70.0, 'state': 'IN_PLAY'}]
        corner = {'type': 'CORNER_KICK', 't': 50.0, 'team': 'A', 'status': 'detected'}
        (g,), data = analyse(s, match_state([(55.0, 'A')], timeline, [corner]))
        self.assertEqual((g.kind, g.start_kind, g.restart['type']), ('set_piece', 'restart', 'CORNER_KICK'))
        self.assertIn('Scored 5 s after a corner.', words(g, data, 'scored'))
        self.assertIn('Conceded 5 s after a corner against it.', words(g, data, 'conceded'))

    def test_the_opponents_corner_before_a_goal_is_no_set_piece(self):
        s, a, b = set_piece_scene()
        timeline = [{'t0': 10.0, 't1': 40.0, 'state': 'IN_PLAY'}, {'t0': 40.0, 't1': 50.0, 'state': 'OUT_OF_PLAY'},
                    {'t0': 50.0, 't1': 70.0, 'state': 'IN_PLAY'}]
        corner = {'type': 'CORNER_KICK', 't': 50.0, 'team': 'B', 'status': 'detected'}
        (g,), data = analyse(s, match_state([(58.0, 'A')], timeline, [corner]))
        self.assertEqual((g.kind, g.start_kind), ('open_play', 'restart'))
        self.assertIn('The goal came 8 s after an opponent corner: the ball was won after it.', words(g, data, 'scored'))
        self.assertIn('Conceded 8 s after its own corner: the ball was lost after the restart.', words(g, data, 'conceded'))

    def test_a_penalty_is_the_ball_still_on_the_mark(self):
        s, a, b = set_piece_scene()
        s.ball[s.k(45):s.k(50)] = [105.0-gl.PENALTY_MARK_M, 34.0]
        timeline = [{'t0': 10.0, 't1': 30.0, 'state': 'IN_PLAY'}, {'t0': 30.0, 't1': 50.0, 'state': 'OUT_OF_PLAY'},
                    {'t0': 50.0, 't1': 70.0, 'state': 'IN_PLAY'}]
        restart = {'type': 'UNKNOWN_RESTART', 't': 50.0, 'team': None, 'status': 'detected'}
        (g,), data = analyse(s, match_state([(51.5, 'A')], timeline, [restart]))
        self.assertEqual((g.kind, g.restart['type']), ('penalty', 'PENALTY_KICK'))
        self.assertIn('Scored from the penalty spot (the ball still on the penalty mark).', words(g, data, 'scored'))
        self.assertTrue(words(g, data, 'conceded')[0].startswith('Gave away a penalty.'))
        # the ball still on the mark at the other end (the scoring team's own penalty area): no penalty
        s, a, b = set_piece_scene()
        s.ball[s.k(45):s.k(50)] = [gl.PENALTY_MARK_M, 34.0]
        (g,), data = analyse(s, match_state([(51.5, 'A')], timeline, [restart]))
        self.assertNotEqual(g.kind, 'penalty')

    def test_a_goal_straight_after_a_long_stop_with_no_ball_is_a_penalty_or_a_direct_free_kick(self):
        s, a, b = set_piece_scene()
        timeline = [{'t0': 10.0, 't1': 20.0, 'state': 'IN_PLAY'}, {'t0': 20.0, 't1': 50.0, 'state': 'OUT_OF_PLAY'},
                    {'t0': 50.0, 't1': 70.0, 'state': 'IN_PLAY'}]
        restart = {'type': 'UNKNOWN_RESTART', 't': 50.0, 'team': None, 'status': 'detected'}
        (g,), data = analyse(s, match_state([(52.0, 'B')], timeline, [restart]))
        self.assertEqual((g.kind, g.restart['stopSeconds']), ('direct_set_piece', 30.0))
        self.assertTrue(words(g, data, 'scored')[0].startswith('Scored 2 s after a restart that ended a long stop'))
        # the same restart six seconds before the goal: an ordinary restart, nothing about a penalty
        (g,), data = analyse(s, match_state([(56.0, 'B')], timeline, [restart]))
        self.assertNotEqual(g.kind, 'direct_set_piece')
        self.assertFalse(any('penalty' in t for t in words(g, data, 'scored')))


def penalty_scene(held=(40.0, 45.0), defender_in_box=False):
    """A attacks right. A's left striker stands 3 m behind the penalty mark of B's goal, B's keeper on the goal line,
    everyone else where the formation puts them (outside the penalty area) during `held`; then A's right striker runs
    into the box (the layout breaks). The ball is never placed."""
    s = ms.Scene(60, 'A')
    a, b = s.add_team('A'), s.add_team('B')
    s.place(a['LS'], held[0], 105-gl.PENALTY_MARK_M-3, 0)
    s.place(b['GK'], held[0], .5, 0)
    s.place(a['RS'], held[1]+.2, 97, 6)
    if defender_in_box:
        s.place(b['LCB'], held[0], 8, 12)
    return s, a, b


class PenaltyLayoutTests(unittest.TestCase):
    def test_the_layout_tells_a_penalty_and_the_goal_comes_after_the_kick(self):
        s, a, b = penalty_scene()
        (g,), data = analyse(s, match_state([(45.8, 'A')]))
        self.assertEqual((g.kind, g.restart['type'], g.restart['how']), ('penalty', 'PENALTY_KICK', 'layout'))
        self.assertEqual((g.start, g.restart['t'], g.t_in), (40.0, 45.0, 45.0))
        self.assertIn('Scored from the penalty spot (the players set out for a penalty).', words(g, data, 'scored'))
        self.assertTrue(words(g, data, 'conceded')[0].startswith('Gave away a penalty. The tracking did not place the play'))
        # the state's goal time can come before the kick (its stop began while the players stood for the penalty)
        (g,), data = analyse(s, match_state([(43.0, 'A')]))
        self.assertEqual((g.kind, g.t_in), ('penalty', 45.0))

    def test_the_play_that_won_it_is_the_scoring_teams_last_loss_of_the_ball(self):
        s, a, b = penalty_scene()
        s.ball[s.k(24):s.k(26)] = [100.0, 30.0]            # in B's penalty area, 5 m from the goal line
        taker = s.people[a['LS']]['id']
        record = rec2([(10, 25, 'A', taker)], [change('A', 'B', 25.0, 27.0, lost_by=taker)])
        (g,), data = analyse(s, match_state([(45.8, 'A')]), record)
        self.assertEqual(g.foul['t'], 25.0)
        self.assertEqual(g.foul['ball'], {'fromGoalM': 5.0, 'lateralM': 4.0, 'inBox': True})
        self.assertIn(f'Won a penalty: the attack that earned it stopped at 00:25 with {taker} on the ball in the penalty area, '
                      f'about 5 m from goal.', words(g, data, 'scored'))
        names = lg.Names(data, 'B', {'A': 'white', 'B': 'red'}, {})
        conceded = [f['text'] for f in g.record(names, lg.clock)['conceded']]
        self.assertEqual(conceded, [f'Gave away a penalty at 00:25: white {taker} had the ball in the penalty area, about 5 m from '
                                    f'goal when the attack stopped. Watch what happened there.'])

    def test_no_penalty_when_the_layout_is_short_or_a_defender_is_in_the_box(self):
        s, a, b = penalty_scene(held=(44.6, 45.0))
        (g,), data = analyse(s, match_state([(45.8, 'A')]))
        self.assertNotEqual(g.kind, 'penalty')
        s, a, b = penalty_scene(defender_in_box=True)
        (g,), data = analyse(s, match_state([(45.8, 'A')]))
        self.assertNotEqual(g.kind, 'penalty')


class LimitTests(unittest.TestCase):
    def test_no_positions_gives_a_note_and_no_tracking_findings(self):
        s = ms.Scene(60, 'A')
        a, b = s.add_team('A'), s.add_team('B')
        for pid in list(a.values())+list(b.values()):
            s.hide(pid, 15, 70)
        (g,), data = analyse(s, match_state([(50.0, 'B')]))
        self.assertEqual((g.start_kind, g.start), ('sustained', 20.0))
        self.assertEqual(g.quality['placed'], 0.0)
        self.assertTrue(any('too little to analyse' in n for n in g.notes), g.notes)
        self.assertEqual(words(g, data, 'scored'), [])
        self.assertEqual(words(g, data, 'conceded'), [])
        self.assertNotIn('box', g.facts)
        self.assertEqual(g.kind, 'unknown')

    def test_a_goal_not_seen_names_the_scoring_teams_last_possession(self):
        s = ms.Scene(60, 'A')
        a, b = s.add_team('A'), s.add_team('B')
        for pid in list(a.values())+list(b.values()):
            s.hide(pid, 15, 70)
        s.ball[s.k(13):s.k(14.4)] = [10.0, 34.0]            # B attacks left: 10 m from A's goal, in the penalty area
        carrier = s.people[b['LS']]['id']
        (g,), data = analyse(s, match_state([(50.0, 'B')]), rec2([(12, 14, 'B', carrier)]))
        self.assertEqual(g.kind, 'unknown')
        self.assertEqual((g.facts['lastSeen']['t'], g.facts['lastSeen']['gapS']), (14.0, 36.0))
        line = (f'The play before this goal was not seen (the tracking placed too little of it). Team B (red) last had the ball '
                f'at 00:14 ({carrier} in the penalty area, about 10 m from goal), 36 s before the goal.')
        names = lg.Names(data, 'B', {'A': 'white', 'B': 'red'}, {})
        r = g.record(names, lg.clock)
        self.assertEqual(([f['text'] for f in r['scored']], [f['text'] for f in r['conceded']]), ([line], [line]))

    def test_an_unknown_scorer_is_not_analysed(self):
        s, record, _ = counter_attack()
        (g,), data = analyse(s, match_state([(47.0, None)]), record)
        self.assertIsNone(g.team)
        self.assertTrue(g.notes[0].startswith('Which team scored is not known'))
        r = g.record(lg.Names(data, 'A', {}, {}), lg.clock)
        self.assertEqual((r['scored'], r['conceded'], r['kind']), ([], [], 'unknown'))

    def test_a_loss_after_the_last_gain_starts_the_attack(self):
        s, record, pid = counter_attack()
        loss = rec2([(10, 20, 'A', None), (20, 45, 'B', pid['b8']), (45, 47, None, None)], [change('A', 'B', 19.8, 20.0)])
        (g,), data = analyse(s, match_state([(47.0, 'A')]), loss)
        self.assertEqual((g.start_kind, g.start), ('loss', 45.0))
        self.assertTrue(words(g, data, 'scored')[0].startswith('Team A (white) scored 1 s after Team B (red) lost control'))
        self.assertTrue(words(g, data, 'conceded')[0].startswith('Lost control of the ball 1 s before the goal'))


class SectionTests(unittest.TestCase):
    def test_removed_goals_are_left_out_and_the_signature_follows_corrections(self):
        state = match_state([(47.0, 'A'), (60.0, 'B')])
        self.assertEqual([e['t'] for e in gl.goal_events(state)], [47.0, 60.0])
        before = gl.signature(state)
        state['events'][1]['status'] = 'removed'
        self.assertEqual([e['t'] for e in gl.goal_events(state)], [47.0])
        self.assertNotEqual(gl.signature(state), before)
        state['events'][0]['team'] = 'B'
        self.assertEqual(gl.signature(state), [(47.0, 'B', 'detected', False)])

    def test_for_team_splits_scored_and_conceded(self):
        s, record, _ = counter_attack()
        found, data = analyse(s, match_state([(47.0, 'A'), (65.0, 'B')]), record)
        names = lg.Names(data, 'A', {'A': 'white', 'B': 'red'}, {})
        sec = gl.for_team(found, 'A', names, lg.clock)
        self.assertEqual([g['t'] for g in sec['scored']], [47.0])
        self.assertEqual([g['t'] for g in sec['conceded']], [65.0])
        self.assertEqual(sec['scored'][0]['findings'], found[0].record(names, lg.clock)['scored'])
        self.assertNotIn('conceded', sec['scored'][0])
        self.assertNotIn('scored', sec['conceded'][0])
        other = gl.for_team(found, 'B', lg.Names(data, 'B', {}, {}), lg.clock)
        self.assertEqual(([g['t'] for g in other['scored']], [g['t'] for g in other['conceded']]), ([65.0], [47.0]))
        self.assertIn('no scorer or assist is named', sec['note'])

    def test_assemble_carries_the_section(self):
        s, record, _ = counter_attack()
        found, data = analyse(s, match_state([(47.0, 'A')]), record)
        engine = en.Engine(data)
        prof = rp._profile({'level': 4, 'ageGroup': 'Adult', 'format': '11v11'})
        sec = gl.for_team(found, 'A', lg.Names(data, 'A', {}, {}), lg.clock)
        rep = rp.assemble(data, engine, [], 'A', prof, {'A': 'white', 'B': 'red'}, {}, 'test', goals_section=sec)
        self.assertEqual(rep['goals'], sec)
        self.assertIsNone(rp.assemble(data, engine, [], 'A', prof, {}, {}, 'test')['goals'])


if __name__ == '__main__':
    unittest.main()
