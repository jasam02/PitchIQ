"""Team shape analytics: width and depth, defensive line height, formation recognition (soccer/team_shape.py)."""
import json
import math
import unittest

import numpy as np

from soccer import team_shape as ts
import team_shape_synth as syn

L, W = syn.LENGTH, syn.WIDTH


def frame_result(people_per_frame, times=None, goal_ends=None, size_source='standard', timeline=None, expected=11,
                 roster=None):
    """A result from explicit per-frame people: lists of (id, team, role, x_metres, y_metres, extra dict)."""
    times = list(times) if times is not None else [k*.2 for k in range(len(people_per_frame))]
    ids = {}
    frames = []
    for k, (t, people) in enumerate(zip(times, people_per_frame)):
        out = []
        for pid, team, role, xm, ym, *extra in people:
            ids[pid] = (team, role)
            rec = {'id': pid, 'team': team, 'role': role, 'state': 'confirmed', 'identityConfidence': .85,
                   'pitch': [xm/L, ym/W], 'pitchConfidence': .9, 'pitchSigma': [.4, .4]}
            if extra:
                rec.update(extra[0])
            out.append(rec)
        frames.append({'time': float(t), 'frame': k, 'people': out, 'calibrated': True})
    players = [{'id': pid, 'display': pid, 'team': team, 'confirmed': True} for pid, (team, role) in sorted(ids.items()) if role == 'PLAYER']
    gks = [{'id': pid, 'display': pid, 'team': team, 'confirmed': True} for pid, (team, role) in sorted(ids.items()) if role == 'GOALKEEPER']
    if roster:
        players += [{'id': pid, 'display': pid, 'team': team, 'confirmed': True} for pid, team in roster if pid not in ids]
    result = {'version': 3, 'video': {'fps': 5.0}, 'frames': frames, 'match': {'players': players, 'goalkeepers': gks},
              'roster': {'expected': {'A': expected, 'B': expected}, 'slots': []},
              'tactical': {'goalEnds': goal_ends if goal_ends is not None else {'left': 'A', 'right': 'B'},
                           'pitchSize': {'length': L, 'width': W, 'source': size_source}}}
    if timeline is not None:
        result['timeline'] = timeline
    return result


def outfield(team, xs, ys, start=1):
    return [(f'{team}-{start+j:02d}', team, 'PLAYER', x, y) for j, (x, y) in enumerate(zip(xs, ys))]


def series_for(result, team='A', mode='observed', settings=None):
    shape = ts.TeamShape(result, mode=mode, settings=settings)
    return shape, ts.width_depth_series(shape.data, team, shape.data.start, shape.data.end)


class WidthDepth(unittest.TestCase):
    def ten(self, xs=None, ys=None):
        xs = xs if xs is not None else [28, 40, 50, 60, 79, 45, 55, 35, 65, 50]
        ys = ys if ys is not None else [12, 20, 30, 40, 57, 25, 35, 45, 50, 15]
        return outfield('A', xs, ys)

    def test_width_is_lateral_max_minus_min(self):
        r = frame_result([self.ten()]*3)
        _, series = series_for(r)
        self.assertAlmostEqual(series[0]['width']['metres'], 45.0, places=6)
        self.assertEqual(series[0]['widthStatus'], 'complete')

    def test_depth_is_longitudinal_max_minus_min(self):
        r = frame_result([self.ten()]*3)
        _, series = series_for(r)
        self.assertAlmostEqual(series[0]['depth']['metres'], 51.0, places=6)
        self.assertEqual({series[0]['depth']['from'], series[0]['depth']['to']}, {'A-01', 'A-05'})

    def test_goalkeeper_is_excluded(self):
        people = self.ten()+[('GK-A', 'A', 'GOALKEEPER', 3.0, 34.0)]
        _, series = series_for(frame_result([people]*3))
        self.assertAlmostEqual(series[0]['depth']['metres'], 51.0, places=6)    # the keeper at 3 m does not deepen the team
        self.assertEqual(series[0]['players'], 10)

    def test_referees_and_unconfirmed_identities_are_excluded(self):
        people = self.ten()+[('REF-1', None, 'REFEREE', 100.0, 66.0), ('A-11', 'A', 'PLAYER', 100.0, 66.0)]
        r = frame_result([people]*3)
        for p in r['match']['players']:
            if p['id'] == 'A-11':
                p['confirmed'] = False     # an unconfirmed slot (a sideline person) is not a team player
        _, series = series_for(r)
        self.assertAlmostEqual(series[0]['width']['metres'], 45.0, places=6)
        self.assertAlmostEqual(series[0]['depth']['metres'], 51.0, places=6)

    def test_duplicate_identity_is_excluded(self):
        # The same person tracked twice (A-11 within 0.2 m of A-05, the widest player) counts once.
        people = self.ten()+[('A-11', 'A', 'PLAYER', 79.1, 57.1, {'identityConfidence': .4})]
        shape, series = series_for(frame_result([people]*3))
        self.assertEqual(series[0]['players'], 10)
        self.assertGreater(shape.data.teams['A']['excluded']['duplicateIdentity'], 0)

    def test_invalid_coordinates_are_rejected(self):
        people = self.ten()
        bad = [(f'A-{12+k:02d}', 'A', 'PLAYER', x, y, extra) for k, (x, y, extra) in enumerate([
            (140.0, 34.0, {}),                                    # off the pitch
            (50.0, 66.0, {'pitchConfidence': .2}),                # untrusted registration
            (50.0, 2.0, {'pitchSigma': [6.0, 6.0]}),              # a position too uncertain for a tactical measurement
            (60.0, 60.0, {'pitch': [float('nan'), .5]})])]        # not a number
        _, series = series_for(frame_result([people+bad]*3))
        self.assertEqual(series[0]['players'], 10)
        self.assertAlmostEqual(series[0]['width']['metres'], 45.0, places=6)

    def test_isolated_registration_spike_is_ignored(self):
        frames = []
        for k in range(40):
            people = self.ten()
            if k == 20:      # one sample where the widest player's mapped position jumps 40 m and back
                people = [p if p[0] != 'A-05' else ('A-05', 'A', 'PLAYER', 79.0, 17.0) for p in people]
                people = [p if p[0] != 'A-01' else ('A-01', 'A', 'PLAYER', 28.0, 52.0) for p in people]
            frames.append(people)
        r = frame_result(frames, times=[k/25 for k in range(40)])
        r['video']['fps'] = 25.0
        shape, series = series_for(r)
        # The spike is dropped (the sample then lacks two players and is only partial); no measured width is wrong.
        widths = {round(s['width']['metres'], 3) for s in series if s['widthStatus'] in ('complete', 'reliable')}
        self.assertEqual(widths, {45.0})
        self.assertGreater(shape.data.teams['A']['excluded']['registrationSpike'], 0)
        agg = ts.team_width_depth(shape.data, 'A', shape.data.start, shape.data.end)['width']
        self.assertEqual((agg['min'], agg['max']), (45.0, 45.0))

    def test_coverage_statuses(self):
        def status(n):
            people = self.ten()[:n]
            _, series = series_for(frame_result([people]*3))
            return series[0]['widthStatus'], series[0]['width']
        self.assertEqual(status(10)[0], 'complete')
        self.assertEqual(status(7)[0], 'partial')             # observed spread only
        self.assertEqual(status(6)[0], 'partial')
        st, value = status(5)
        self.assertEqual(st, 'unavailable')
        self.assertIsNone(value)

    def test_missing_extreme_player_keeps_partial(self):
        """Nine observed and the tenth's estimate outside the observed width: the measured width is not full-team."""
        people = self.ten()
        observed = [p for p in people if p[0] != 'A-05']                  # the widest (y 57) and most advanced (x 79) one
        frames = [observed]*5
        timeline = self.timeline(frames, {'A-05': (79/L, 57/W, 2.0)})
        _, series = series_for(frame_result(frames, timeline=timeline, roster=[('A-05', 'A')]))
        self.assertEqual(series[0]['widthStatus'], 'partial')
        self.assertEqual(series[0]['widthEdgeRisk'], 'possible')
        self.assertLess(series[0]['width']['metres'], 45.0)

    def test_missing_inner_player_is_reliable(self):
        people = self.ten()
        observed = [p for p in people if p[0] != 'A-06']                  # (45, 25): inside both extremes
        frames = [observed]*5
        timeline = self.timeline(frames, {'A-06': (45/L, 25/W, 2.0)})
        _, series = series_for(frame_result(frames, timeline=timeline, roster=[('A-06', 'A')]))
        self.assertEqual(series[0]['players'], 9)
        self.assertEqual(series[0]['widthStatus'], 'reliable')
        self.assertAlmostEqual(series[0]['width']['metres'], 45.0, places=6)

    def test_missing_player_without_estimate_is_partial(self):
        observed = [p for p in self.ten() if p[0] != 'A-06']
        _, series = series_for(frame_result([observed]*5, roster=[('A-06', 'A')]))
        self.assertEqual(series[0]['widthStatus'], 'partial')
        self.assertEqual(series[0]['widthEdgeRisk'], 'unknown')

    def test_estimated_mode_is_labelled(self):
        observed = [p for p in self.ten() if p[0] != 'A-05']
        frames = [observed]*5
        timeline = self.timeline(frames, {'A-05': (79/L, 57/W, 2.0)})
        r = frame_result(frames, timeline=timeline, roster=[('A-05', 'A')])
        _, obs = series_for(r)
        _, est = series_for(r, mode='estimated')
        self.assertEqual(obs[0]['estimated'], 0)
        self.assertEqual(est[0]['estimated'], 1)
        self.assertEqual(est[0]['observed'], 9)
        self.assertAlmostEqual(est[0]['width']['metres'], 45.0, places=4)

    def test_time_weighting(self):
        """Dense sampling of one stretch must not dominate: 2 s at width 45 m sampled at 5 Hz, then 2 s at 30 m."""
        wide = self.ten()
        narrow = outfield('A', [28, 40, 50, 60, 79, 45, 55, 35, 65, 50], [12, 20, 30, 40, 42, 25, 35, 38, 41, 15])
        frames, times = [], []
        for k in range(10):
            frames.append(wide); times.append(k*.2)
        for k in range(10):
            frames.append(narrow); times.append(2.0+k*.2)
        shape = ts.TeamShape(frame_result(frames, times=times))
        agg = ts.team_width_depth(shape.data, 'A', shape.data.start, shape.data.end)['width']
        self.assertAlmostEqual(agg['mean'], 37.5, delta=.8)
        self.assertEqual(agg['min'], 30.0)
        self.assertEqual(agg['max'], 45.0)

    def test_metres_flagged_estimated_without_measured_size(self):
        shape = ts.TeamShape(frame_result([self.ten()]*3))
        self.assertTrue(shape.report('A')['pitch']['estimated'])
        shape = ts.TeamShape(frame_result([self.ten()]*3, size_source='landmarks'))
        self.assertFalse(shape.report('A')['pitch']['estimated'])

    @staticmethod
    def timeline(frames, estimates):
        ids = {}
        for pid, (x, y, sigma) in estimates.items():
            ids[pid] = {'team': 'A', 'x': [x]*len(frames), 'y': [y]*len(frames), 'kind': ['P']*len(frames), 'sigma': [[sigma, sigma]]*len(frames)}
        return {'frameTimes': [k*.2 for k in range(len(frames))], 'ids': ids}


class Directions(unittest.TestCase):
    def result(self, goal_ends, flip_after=None, n=200):
        times = np.arange(n)*.2
        a, b = syn.LAYOUTS['4-4-2'], syn.LAYOUTS['4-4-2']
        frames = []
        for t in times:
            side_a = 'left' if flip_after is None or t < flip_after else 'right'
            side_b = 'right' if side_a == 'left' else 'left'
            people = []
            for team, layout, side in (('A', a, side_a), ('B', b, side_b)):
                for j, (X, Y) in enumerate(layout):
                    x, y = syn.to_pitch(X, Y, side)
                    people.append((f'{team}-{j+1:02d}', team, 'PLAYER', float(x)*L, float(y)*W))
                gx, gy = syn.to_pitch(6, 34, side)
                people.append((f'GK-{team}', team, 'GOALKEEPER', float(gx)*L, float(gy)*W))
            frames.append(people)
        return frame_result(frames, times=times, goal_ends=goal_ends)

    def test_tracking_goal_ends(self):
        shape = ts.TeamShape(self.result({'left': 'B', 'right': 'A'}))
        # The tracking says A defends the right goal, but every player stands as if A defended the left: contradicted.
        self.assertTrue((shape.directions['A']['status'] == 'contradicted').all())
        self.assertTrue((shape.directions['A']['side'] == '').all())
        shape = ts.TeamShape(self.result({'left': 'A', 'right': 'B'}))
        self.assertEqual(set(shape.directions['A']['side']), {'left'})
        self.assertEqual(set(shape.directions['B']['side']), {'right'})

    def test_one_named_end_names_the_other(self):
        shape = ts.TeamShape(self.result({'left': None, 'right': 'B'}))
        self.assertEqual(set(shape.directions['A']['side']), {'left'})
        self.assertEqual(shape.directions['A']['source'], 'inferred')

    def test_evidence_alone(self):
        shape = ts.TeamShape(self.result({}))
        self.assertEqual(set(shape.directions['A']['side']), {'left'})
        self.assertEqual(set(shape.directions['A']['status']), {'evidence'})

    def test_user_setting_with_side_switch(self):
        r = self.result({'left': 'A', 'right': 'B'}, flip_after=20.0)
        plain = ts.TeamShape(r)
        late = plain.data.t >= 30
        self.assertTrue((plain.directions['A']['status'][late] == 'contradicted').all())   # an unknown side change is not silently believed
        shape = ts.TeamShape(r, settings={'direction': {'A': 'left', 'switchAt': 20.0}})
        sides = shape.directions['A']['side']
        self.assertEqual(set(sides[shape.data.t < 20]), {'left'})
        self.assertEqual(set(sides[shape.data.t >= 20]), {'right'})
        self.assertFalse((shape.directions['A']['status'] == 'contradicted').any())

    def test_window_without_evidence_next_to_a_contradiction_is_not_believed(self):
        """The sides swap at 30 s unknown to the analysis; 60-90 s has no evidence at all (both keepers unseen, five
        players a team): it is not given the old side either."""
        r = self.result({'left': 'A', 'right': 'B'}, flip_after=30.0, n=600)
        for f in r['frames']:
            if 60 <= f['time'] < 90:
                f['people'] = [p for p in f['people'] if p['role'] == 'PLAYER' and int(p['id'][2:]) <= 5]
        shape = ts.TeamShape(r)
        quiet = (shape.data.t >= 60) & (shape.data.t < 90)
        self.assertTrue((shape.directions['A']['status'][quiet] == 'contradicted').all())
        self.assertTrue((shape.directions['A']['side'][quiet] == '').all())
        early = shape.data.t < 30
        self.assertEqual(set(shape.directions['A']['status'][early]), {'tracking'})

    def test_goalkeeper_decides_mixed_evidence(self):
        """30-60 s: the teams' outfield centres are in each other's half (an ordering that says A defends the right
        goal) while both keepers stay at their goals. The keeper's side decides: believed with A defending the left goal,
        contradicted with A defending the right one, although the ordering agrees with that."""
        def result(goal_ends):
            r = self.result(goal_ends, n=450)
            for f in r['frames']:
                if 30 <= f['time'] < 60:
                    for p in f['people']:
                        if p['role'] == 'PLAYER':
                            p['pitch'] = [.65 + .01*int(p['id'][2:]) if p['team'] == 'A' else .25 + .01*int(p['id'][2:]), p['pitch'][1]]
            return r
        window = lambda shape: (shape.data.t >= 30) & (shape.data.t < 60)
        right = ts.TeamShape(result({'left': 'A', 'right': 'B'}))
        self.assertEqual(set(right.directions['A']['status'][window(right)]), {'tracking'})
        self.assertTrue(any('A' in w.get('mixed', []) for w in right.directions['windows']))
        wrong = ts.TeamShape(result({'left': 'B', 'right': 'A'}))
        self.assertEqual(set(wrong.directions['A']['status'][window(wrong)]), {'contradicted'})

    def test_own_goal_distance(self):
        x = np.array([.2, .2])
        side = np.array(['left', 'right'])
        np.testing.assert_allclose(ts.own_goal_distance(x, side, L), [.2*L, .8*L])


class DefensiveLine(unittest.TestCase):
    def test_median_and_deepest(self):
        """Four defenders 31, 32, 34 and 33 m from their own (left) goal: line height 32.5 m, deepest defender 31 m."""
        defenders = [(31, 10), (32, 26), (34, 42), (33, 58)]
        layout = defenders+[(48, 9), (46, 26), (46, 42), (48, 59), (62, 28), (63, 41)]
        people = outfield('A', [p[0] for p in layout], [p[1] for p in layout])+[('GK-A', 'A', 'GOALKEEPER', 4.0, 34.0)]
        b = [(f'B-{j+1:02d}', 'B', 'PLAYER', L-X, W-Y) for j, (X, Y) in enumerate(syn.LAYOUTS['4-4-2'])]
        r = frame_result([people+b]*100)
        shape = ts.TeamShape(r)
        line = ts.defensive_line(shape.data, 'A', shape.directions, shape.shapes['A'], shape.data.start, shape.data.end)
        rec = line[len(line)//2]
        self.assertEqual(rec['status'], 'complete')
        self.assertEqual({d['id'] for d in rec['defenders']}, {'A-01', 'A-02', 'A-03', 'A-04'})   # the keeper is not one
        self.assertAlmostEqual(rec['height'], 32.5, places=6)
        self.assertAlmostEqual(rec['deepest'], 31.0, places=6)
        self.assertEqual(rec['quality'], 'coherent')

    def test_right_goal_is_measured_from_the_right_goal_line(self):
        times, teams, ends = syn.match('4-4-2', '4-4-2', seconds=40, side_a='right', seed=2)
        shape = ts.TeamShape(syn.make_result(times, teams, ends))
        rep = shape.report('A')
        self.assertEqual(rep['direction']['side'], 'right')
        # The layout's back four stands 24-28 m from its own goal (the block moves by up to 10 m).
        self.assertTrue(14 < rep['line']['summary']['height']['median'] < 38, rep['line']['summary'])

    def test_goalkeeper_never_lowers_the_line(self):
        times, teams, ends = syn.match('4-4-2', '4-4-2', seconds=40, seed=4)
        shape = ts.TeamShape(syn.make_result(times, teams, ends))
        line = ts.defensive_line(shape.data, 'A', shape.directions, shape.shapes['A'], 0, 40)
        self.assertTrue(all(not d['id'].startswith('GK') for r in line for d in r.get('defenders', [])))

    def test_insufficient_defenders(self):
        times, teams, ends = syn.match('4-4-2', '4-4-2', seconds=40, seed=5)
        vis = np.ones((len(times), 10), bool)
        vis[len(times)//2:, :3] = False          # three of the back four leave the view for the second half
        shape = ts.TeamShape(syn.make_result(times, teams, ends, visible={'A': vis}))
        line = ts.defensive_line(shape.data, 'A', shape.directions, shape.shapes['A'], 25, 40)
        self.assertTrue(all(r['status'] == 'unavailable' for r in line))

    def test_one_defender_missing_is_partial(self):
        times, teams, ends = syn.match('4-4-2', '4-4-2', seconds=40, seed=6)
        vis = np.ones((len(times), 10), bool)
        vis[len(times)//2:, 0] = False
        shape = ts.TeamShape(syn.make_result(times, teams, ends, visible={'A': vis}))
        line = ts.defensive_line(shape.data, 'A', shape.directions, shape.shapes['A'], 25, 40)
        self.assertTrue(all(r['status'] == 'partial' for r in line if r['status'] != 'displaced'), {r['status'] for r in line})

    def test_fullback_overlap_is_uneven_not_a_new_line(self):
        times = np.arange(0, 60, .1)

        def overlap(t, pos):
            pos = pos.copy()
            pos[(t >= 30) & (t < 36), 3, 0] += 20
            return pos
        a = syn.team_track(times, '4-4-2', 'left', 7, overrides=overlap, block=0, stretch=0)
        b = syn.team_track(times, '4-4-2', 'right', 8)
        r = syn.make_result(times, {'A': (a, syn.goalkeeper_track(times, 'left')), 'B': (b, syn.goalkeeper_track(times, 'right'))})
        shape = ts.TeamShape(r)
        line = ts.defensive_line(shape.data, 'A', shape.directions, shape.shapes['A'], 31, 35)
        self.assertTrue(all(len(r['defenders']) == 4 for r in line))       # still the same four defenders
        self.assertTrue(all(r['quality'] == 'uneven' for r in line), {r['quality'] for r in line})
        self.assertEqual([p['label'] for p in shape.shapes['A']['periods']], ['4-4-2'])

    def test_attacking_set_piece_is_displaced(self):
        """The centre-backs go forward into the opponents' box while midfielders stay back: no defensive line there."""
        times = np.arange(0, 60, .1)

        def corner(t, pos):
            pos = pos.copy()
            on = (t >= 30) & (t < 40)
            pos[on, :4, 0] = 95.0
            pos[on, 4:8, 0] = 60.0
            return pos
        a = syn.team_track(times, '4-4-2', 'left', 9, overrides=corner, block=0)
        b = syn.team_track(times, '4-4-2', 'right', 10)
        r = syn.make_result(times, {'A': (a, syn.goalkeeper_track(times, 'left')), 'B': (b, syn.goalkeeper_track(times, 'right'))})
        shape = ts.TeamShape(r)
        line = ts.defensive_line(shape.data, 'A', shape.directions, shape.shapes['A'], 32, 38)
        self.assertTrue(all(r['status'] == 'displaced' for r in line), {r['status'] for r in line})

    def test_direction_unknown_gives_no_line(self):
        times, teams, ends = syn.match('4-4-2', '4-4-2', seconds=30, seed=11)
        r = syn.make_result(times, teams, {'left': 'B', 'right': 'A'})      # contradicted by everything on the pitch
        shape = ts.TeamShape(r)
        line = ts.defensive_line(shape.data, 'A', shape.directions, shape.shapes['A'], 0, 30)
        self.assertTrue(all(r['status'] == 'unavailable' for r in line))

    def test_trend(self):
        times = np.arange(0, 90, .1)

        def advance(t, pos):
            return pos+np.stack([np.repeat((t/90*15)[:, None], 10, axis=1), np.zeros((len(t), 10))], axis=2)
        a = syn.team_track(times, '4-4-2', 'left', 12, overrides=advance, block=0)
        b = syn.team_track(times, '4-4-2', 'right', 13)
        r = syn.make_result(times, {'A': (a, syn.goalkeeper_track(times, 'left')), 'B': (b, syn.goalkeeper_track(times, 'right'))})
        rep = ts.TeamShape(r).report('A')
        self.assertEqual(rep['line']['summary']['trend']['label'], 'advancing')


class Formations(unittest.TestCase):
    def recognise(self, layout, seed=0, side='left', seconds=60, visible=None):
        times, teams, ends = syn.match(layout, '4-4-2', seconds=seconds, side_a=side, seed=seed)
        r = syn.make_result(times, teams, ends, visible=visible)
        shape = ts.TeamShape(r)
        return shape, shape.report('A')['formation']['periods']

    def test_clear_formations(self):
        for layout in ('4-4-2', '4-3-3', '4-2-3-1', '3-5-2'):
            for seed in (0, 1, 2):
                _, periods = self.recognise(layout, seed)
                self.assertEqual([p['label'] for p in periods], [layout], (layout, seed, periods))

    def test_roles_follow_the_lines(self):
        shape, periods = self.recognise('4-4-2', 3)
        roles = periods[0]['roles']
        for j, line in enumerate(syn.LINES['4-4-2']):
            self.assertEqual(roles[f'A-{j+1:02d}']['line'], line)
        # Left and right from the team's own point of view: A-01 (the left back) is LB.
        self.assertEqual(roles['A-01']['role'], 'LB')
        self.assertEqual(roles['A-04']['role'], 'RB')

    def test_orientation_does_not_matter(self):
        for layout in ('4-3-3', '3-5-2'):
            shape, periods = self.recognise(layout, 4, side='right')
            self.assertEqual([p['label'] for p in periods], [layout])
            self.assertEqual(periods[0]['roles']['A-01']['role'], 'LB' if layout.startswith('4') else 'LCB')

    def test_one_missing_player(self):
        vis = np.ones((600, 10), bool)
        vis[:, 9] = False
        shape, periods = self.recognise('4-3-3', 5, visible={'A': vis})
        self.assertEqual([p['label'] for p in periods], ['4-3-3'])
        self.assertEqual(periods[0]['players'], 9)

    def test_several_missing_players(self):
        vis = np.ones((600, 10), bool)
        vis[:, 6:] = False
        shape, periods = self.recognise('4-3-3', 6, visible={'A': vis})
        self.assertEqual([p['label'] for p in periods], ['insufficient'])
        self.assertTrue(all(w['status'] == 'coverage' for w in shape.shapes['A']['windows']))

    def test_two_missing_players_low_confidence(self):
        vis = np.ones((600, 10), bool)
        vis[:, 8:] = False
        shape, periods = self.recognise('4-4-2', 7, visible={'A': vis})
        self.assertNotEqual(periods[0].get('confidence'), 'clear')

    def test_swapping_players_is_not_a_formation_change(self):
        times = np.arange(0, 120, .1)

        def swap(t, pos):
            pos = pos.copy()
            on = (t >= 40) & (t < 80)
            pos[on, 5], pos[on, 6] = pos[on, 6].copy(), pos[on, 5].copy()
            return pos
        a = syn.team_track(times, '4-4-2', 'left', 8, overrides=swap)
        b = syn.team_track(times, '4-4-2', 'right', 9)
        r = syn.make_result(times, {'A': (a, syn.goalkeeper_track(times, 'left')), 'B': (b, syn.goalkeeper_track(times, 'right'))})
        shape = ts.TeamShape(r)
        self.assertEqual([p['label'] for p in shape.shapes['A']['periods']], ['4-4-2'])
        hist = shape.shapes['A']['roles']['A-06']
        role_at = lambda t: min(hist, key=lambda h: abs(h[0]-t))[1]
        self.assertIn(role_at(20), ('LCM', 'RCM'))
        self.assertIn(role_at(60), ('LCM', 'RCM'))
        self.assertNotEqual(role_at(20), role_at(60))     # the role changed with the swap while the formation did not

    def test_temporary_runs_do_not_change_the_formation(self):
        times = np.arange(0, 120, .1)

        def runs(t, pos):
            pos = pos.copy()
            pos[(t % 30 >= 10) & (t % 30 < 16), 3, 0] += 18        # the right back overlaps
            pos[(t % 40 >= 25) & (t % 40 < 33), 4:, 0] += 10       # midfield and forwards press
            return pos
        for layout in ('4-4-2', '4-3-3'):
            a = syn.team_track(times, layout, 'left', 10, overrides=runs)
            b = syn.team_track(times, '4-4-2', 'right', 11)
            r = syn.make_result(times, {'A': (a, syn.goalkeeper_track(times, 'left')), 'B': (b, syn.goalkeeper_track(times, 'right'))})
            labels = [p['label'] for p in ts.TeamShape(r).shapes['A']['periods']]
            self.assertEqual(labels, [layout])

    def test_sustained_change_is_detected(self):
        times = np.arange(0, 180, .1)
        a = np.concatenate([syn.team_track(times[times < 90], '4-4-2', 'left', 12), syn.team_track(times[times >= 90], '4-3-3', 'left', 13)])
        b = syn.team_track(times, '4-4-2', 'right', 14)
        r = syn.make_result(times, {'A': (a, syn.goalkeeper_track(times, 'left')), 'B': (b, syn.goalkeeper_track(times, 'right'))})
        periods = ts.TeamShape(r).shapes['A']['periods']
        named = [p for p in periods if p['label'] in ts.TEMPLATES]
        self.assertEqual([p['label'] for p in named], ['4-4-2', '4-3-3'])
        self.assertLess(abs(named[1]['t0']-90), 15)

    def test_ambiguous_shape_is_not_claimed(self):
        """Halfway between a 4-4-2 and a 4-2-3-1: the label must not be 'clear'."""
        mid = [((ax+bx)/2, (ay+by)/2) for (ax, ay), (bx, by) in zip(syn.LAYOUTS['4-4-2'],
               [(28, 7), (24, 25), (24, 43), (28, 61), (46, 10), (37, 27), (37, 41), (46, 58), (50, 34), (58, 34)])]
        times = np.arange(0, 60, .1)
        a = syn.team_track(times, mid, 'left', 15)
        b = syn.team_track(times, '4-4-2', 'right', 16)
        r = syn.make_result(times, {'A': (a, syn.goalkeeper_track(times, 'left')), 'B': (b, syn.goalkeeper_track(times, 'right'))})
        periods = ts.TeamShape(r).shapes['A']['periods']
        self.assertEqual(len(periods), 1)
        self.assertIn(periods[0].get('confidence'), ('ambiguous', 'likely'))
        self.assertIn({periods[0]['label'], periods[0]['alternative']} & {'4-4-2', '4-2-3-1'}, ({'4-4-2', '4-2-3-1'}, {'4-4-2'}, {'4-2-3-1'}))

    def test_halftime_flip(self):
        times = np.arange(0, 120, .1)
        first, second = times < 60, times >= 60
        a = np.concatenate([syn.team_track(times[first], '3-5-2', 'left', 17), syn.team_track(times[second], '3-5-2', 'right', 18)])
        b = np.concatenate([syn.team_track(times[first], '4-4-2', 'right', 19), syn.team_track(times[second], '4-4-2', 'left', 20)])
        ga = np.concatenate([syn.goalkeeper_track(times[first], 'left'), syn.goalkeeper_track(times[second], 'right')])
        gb = np.concatenate([syn.goalkeeper_track(times[first], 'right'), syn.goalkeeper_track(times[second], 'left')])
        r = syn.make_result(times, {'A': (a, ga), 'B': (b, gb)}, {'left': 'A', 'right': 'B'})
        shape = ts.TeamShape(r, settings={'direction': {'A': 'left', 'switchAt': 60.0}})
        for team, layout in (('A', '3-5-2'), ('B', '4-4-2')):
            named = [p['label'] for p in shape.shapes[team]['periods'] if p['label'] != 'direction']
            self.assertEqual(set(named), {layout}, (team, shape.shapes[team]['periods']))
        # Without the setting the second half is contradicted: no formation is claimed there.
        plain = ts.TeamShape(r)
        late = [p for p in plain.shapes['A']['periods'] if p['t1'] > 65]
        self.assertTrue(late and all(p['label'] == 'direction' for p in late), [(p['t0'], p['t1'], p['label']) for p in plain.shapes['A']['periods']])

    def test_template_fit_is_scale_and_shift_invariant(self):
        tpl = ts.TEMPLATES['4-3-3']
        P = tpl['xy']*[1.3, .8]+[5, -3]
        fit = ts.fit_template(P-P.mean(axis=0), tpl)
        self.assertLess(fit['relative'], 1e-6)
        self.assertEqual(sorted(fit['roles'].tolist()), list(range(10)))

    def test_nine_player_lineup(self):
        times = np.arange(0, 60, .1)
        nine = [(28, 7), (24, 24), (24, 44), (28, 61), (42, 9), (39, 26), (39, 42), (42, 59), (56, 34)]
        a = syn.team_track(times, nine, 'left', 21)
        b = syn.team_track(times, '4-4-2', 'right', 22)
        r = syn.make_result(times, {'A': (a, syn.goalkeeper_track(times, 'left')), 'B': (b, syn.goalkeeper_track(times, 'right'))})
        shape = ts.TeamShape(r, settings={'lineup': {'A': 9}})
        self.assertEqual([p['label'] for p in shape.shapes['A']['periods']], ['4-4-1'])
        default = ts.TeamShape(r)
        self.assertNotIn('4-4-1', [p['label'] for p in default.shapes['A']['periods']])   # with ten expected, never a 9-player shape

    def test_unsupported_lineup(self):
        times, teams, ends = syn.match('4-4-2', '4-4-2', seconds=30, seed=23)
        shape = ts.TeamShape(syn.make_result(times, teams, ends), settings={'lineup': {'A': 8}})
        self.assertEqual({p['label'] for p in shape.shapes['A']['periods']}, {'lineup'})


class ReportAndSettings(unittest.TestCase):
    def test_manual_correction_is_kept_apart(self):
        times, teams, ends = syn.match('4-3-3', '4-4-2', seconds=60, seed=24)
        r = syn.make_result(times, teams, ends)
        before = json.dumps(r['frames'][100])
        settings = {'formations': [{'team': 'A', 't0': 0, 't1': 60, 'formation': '4-2-3-1', 'predicted': '4-3-3', 'created': 'test'}]}
        shape = ts.TeamShape(r, settings=settings)
        rep = shape.report('A')
        period = rep['formation']['periods'][0]
        self.assertEqual(period['label'], '4-3-3')                    # the prediction is unchanged
        self.assertEqual(period['correction']['formation'], '4-2-3-1')  # the coach's label is shown with it
        self.assertEqual(json.dumps(r['frames'][100]), before)         # no coordinate moved

    def test_settings_are_cleaned(self):
        s = ts.clean_settings({'direction': {'A': 'up', 'switchAt': -1}, 'lineup': {'A': 3, 'B': 9},
                               'formations': [{'team': 'A', 't0': 5, 't1': 1, 'formation': '4-4-2'}, {'team': 'C', 't0': 0, 't1': 9, 'formation': '4-4-2'},
                                              {'team': 'B', 't0': 0, 't1': 9, 'formation': '9-9-9'}, {'team': 'B', 't0': 0, 't1': 9, 'formation': '5-4-1'}]})
        self.assertIsNone(s['direction']['A'])
        self.assertEqual(s['lineup'], {'A': None, 'B': 9})
        self.assertEqual([c['formation'] for c in s['formations']], ['5-4-1'])

    def test_report_is_json_and_ranges_clip(self):
        times, teams, ends = syn.match('4-4-2', '3-5-2', seconds=60, seed=25)
        shape = ts.TeamShape(syn.make_result(times, teams, ends))
        rep = shape.report('B', 10, 30)
        json.dumps(rep)
        self.assertEqual(rep['range']['t0'], 10)
        self.assertTrue(all(10-1e-6 <= r['t'] <= 30+1e-6 for r in rep['widthDepth']['series']))
        self.assertFalse(rep['formation']['phases']['available'])
        snap = shape.snapshot('B', 20.0)
        json.dumps(snap)
        self.assertEqual(len(snap['players']), 10)
        self.assertEqual(snap['formation']['label'], '3-5-2')

    def test_identity_corrections_change_the_analysis(self):
        """A player the user excluded (match.excluded instead of match.players) is gone from every measurement."""
        times, teams, ends = syn.match('4-4-2', '4-4-2', seconds=30, seed=26)
        r = syn.make_result(times, teams, ends)
        full = ts.TeamShape(r).report('A')['coverage']['identities']
        r['match']['excluded'] = [p for p in r['match']['players'] if p['id'] == 'A-05']
        r['match']['players'] = [p for p in r['match']['players'] if p['id'] != 'A-05']
        after = ts.TeamShape(r).report('A')
        self.assertEqual(after['coverage']['identities'], full-1)
        self.assertTrue(all(s['players'] <= 9 for s in after['widthDepth']['series']))


def two_teams(times, a, b=None, side_a='left', seed=40):
    """{'A': ..., 'B': ...} for make_result: team A's outfield positions `a`, team B a moving 4-4-2."""
    side_b = 'right' if side_a == 'left' else 'left'
    b = b if b is not None else syn.team_track(times, '4-4-2', side_b, seed)
    return {'A': (a, syn.goalkeeper_track(times, side_a)), 'B': (b, syn.goalkeeper_track(times, side_b))}


# A formless shape: ten spots no template fits (relative error above UNCLEAR_ERROR in every window).
FORMLESS = [(37, 44), (51, 60), (35, 46), (34, 53), (29, 56), (40, 47), (41, 23), (40, 8), (22, 36), (39, 53)]


class Coverage(unittest.TestCase):
    """Who counts, per sample: estimates, players who left, lineups, duplicates, glitches."""

    def test_estimated_mode_with_an_unpositioned_player_is_partial(self):
        observed = [p for p in WidthDepth().ten() if p[0] != 'A-06']
        _, est = series_for(frame_result([observed]*5, roster=[('A-06', 'A')]), mode='estimated')
        self.assertEqual(est[0]['widthStatus'], 'partial')        # A-06 has no position at all: an edge may be missing
        self.assertEqual(est[0]['widthEdgeRisk'], 'unknown')

    def test_mostly_estimated_is_never_complete(self):
        people = WidthDepth().ten()
        observed = people[:3]
        frames = [observed]*5
        estimates = {p[0]: (p[3]/L, p[4]/W, 2.0) for p in people[3:]}
        r = frame_result(frames, timeline=WidthDepth.timeline(frames, estimates), roster=[(p[0], 'A') for p in people[3:]])
        shape, est = series_for(r, mode='estimated')
        self.assertEqual((est[0]['observed'], est[0]['estimated']), (3, 7))
        self.assertEqual(est[0]['widthStatus'], 'estimated')
        agg = ts.team_width_depth(shape.data, 'A', shape.data.start, shape.data.end)['width']
        self.assertEqual(agg['reliableShare'], 0.0)
        self.assertEqual(agg['estimatedShare'], 1.0)
        self.assertIsNone(agg['mean'])

    def test_estimate_on_an_observed_teammate_is_not_an_extra_player(self):
        people = WidthDepth().ten()
        frames = [people]*5
        timeline = WidthDepth.timeline(frames, {'A-11': (79.2/L, 57.1/W, 2.0)})   # a stale prediction on top of A-05
        shape, est = series_for(frame_result(frames, timeline=timeline, roster=[('A-11', 'A')]), mode='estimated')
        self.assertEqual(est[0]['players'], 10)
        self.assertGreater(shape.data.teams['A']['excluded']['estimateOnObserved'], 0)

    def test_player_who_left_stops_counting(self):
        times = np.arange(0, 60, .1)

        def walk_off(t, pos):
            pos = pos.copy()
            pos[t >= 30, 4] = (60, 67.6)          # sent off at 30 s, still tracked along the touchline
            return pos
        r = syn.make_result(times, two_teams(times, syn.team_track(times, '4-4-2', 'left', 41, overrides=walk_off)))
        for slot in r['roster']['slots']:
            if slot['id'] == 'A-05':
                slot['left'] = {'time': 30.0, 'reason': 'substituted or sent off (marked by you)', 'correction': 'c1'}
        shape = ts.TeamShape(r, settings={'lineup': {'A': {'outfield': 9, 'from': 30}}})
        series = ts.width_depth_series(shape.data, 'A', 0, 60)
        late = [s for s in series if s['t'] >= 35]
        early = [s for s in series if s['t'] < 25]
        self.assertTrue(all(s['players'] == 9 and s['expected'] == 9 for s in late))
        self.assertTrue(all(s['expected'] == 10 for s in early))
        self.assertTrue(all(s['width']['to'] != 'A-05' and s['width']['from'] != 'A-05' for s in late))
        self.assertGreater(shape.data.teams['A']['excluded']['afterLeaving'], 0)
        self.assertEqual(shape.report('A')['coverage']['lineupChange'], {'from': 30.0, 'outfield': 9})

    def test_lineup_of_nine_with_ten_on_the_pitch_does_not_crash(self):
        times, teams, ends = syn.match('4-4-2', '4-4-2', seconds=40, seed=42)
        shape = ts.TeamShape(syn.make_result(times, teams, ends), settings={'lineup': {'A': 9}})
        rep = shape.report('A')
        json.dumps(rep)
        ok = [w for w in shape.shapes['A']['windows'] if w['status'] == 'ok']
        self.assertTrue(ok)
        self.assertTrue(all(w['players'] == 9 and w['excess'] == 1 for w in ok))
        json.dumps(shape.snapshot('A', 20))

    def test_dismissal_changes_the_candidates(self):
        times = np.arange(0, 150, .1)
        first, second = times < 70, times >= 70
        nine = [(28, 7), (24, 24), (24, 44), (28, 61), (42, 9), (39, 26), (39, 42), (42, 59), (56, 34), (50, 34)]
        a = np.concatenate([syn.team_track(times[first], '4-4-2', 'left', 43), syn.team_track(times[second], nine, 'left', 44)])
        a[second, 9] = np.nan                    # the tenth player is off from 70 s
        r = syn.make_result(times, two_teams(times, a))
        shape = ts.TeamShape(r, settings={'lineup': {'A': {'outfield': 9, 'from': 70}}})
        named = [p['label'] for p in shape.shapes['A']['periods'] if p['label'] in ts.TEMPLATES]
        self.assertEqual(named, ['4-4-2', '4-4-1'])
        self.assertTrue(all(w['status'] == 'lineup' for w in shape.shapes['A']['windows'] if w['t0'] < 70 < w['t1']))

    def test_duplicate_removal_spares_a_close_teammate(self):
        people = [p for p in WidthDepth().ten() if p[0] != 'A-06']+[('A-06', 'A', 'PLAYER', 79.6, 57.0),
                                                                     ('A-11', 'A', 'PLAYER', 79.3, 57.0, {'identityConfidence': .4})]
        shape, series = series_for(frame_result([people]*5))
        self.assertEqual(series[0]['players'], 10)         # A-11 is the duplicate; A-05 and A-06 (0.6 m apart) both stay
        ids = {series[0]['width']['to'], series[0]['depth']['to']}
        self.assertTrue(ids <= {'A-05', 'A-06'})

    def test_brief_closeness_is_not_a_duplicate(self):
        apart = [p if p[0] != 'A-06' else ('A-06', 'A', 'PLAYER', 80.5, 57.0) for p in WidthDepth().ten()]
        close = [p if p[0] != 'A-06' else ('A-06', 'A', 'PLAYER', 79.2, 57.0) for p in apart]      # 0.2 m from A-05
        _, series = series_for(frame_result([apart]*3+[close]*2+[apart]*3))       # two samples together: a challenge
        self.assertTrue(all(s['players'] == 10 for s in series))

    def test_spike_next_to_a_gap_or_track_edge_is_ignored(self):
        def run(frames_for):
            frames = [frames_for(k) for k in range(100)]
            r = frame_result(frames, times=[k/25 for k in range(100)])
            r['video']['fps'] = 25.0
            shape, series = series_for(r)
            widths = {round(s['width']['metres'], 3) for s in series if s['widthStatus'] in ('complete', 'reliable')}
            return shape, widths
        base = WidthDepth().ten()
        jumped = [p if p[0] != 'A-05' else ('A-05', 'A', 'PLAYER', 79.0, 17.0) for p in base]
        missing = [p for p in base if p[0] != 'A-05']
        # A 40 m jump for three frames, then the player is occluded for 1.6 s.
        shape, widths = run(lambda k: jumped if 50 <= k < 53 else (missing if 53 <= k < 93 else base))
        self.assertEqual(widths, {45.0})
        self.assertGreater(shape.data.teams['A']['excluded']['registrationSpike'], 0)
        # A track starting with three wrong frames before the position settles.
        shape, widths = run(lambda k: missing if k < 40 else (jumped if k < 43 else base))
        self.assertEqual(widths, {45.0})

    def test_samples_carry_geometric_and_identity_confidence(self):
        times, teams, ends = syn.match('4-4-2', '4-4-2', seconds=30, seed=45)
        shape = ts.TeamShape(syn.make_result(times, teams, ends))
        rep = shape.report('A')
        row = rep['widthDepth']['series'][10]
        self.assertAlmostEqual(row['geometric'], .9, places=3)
        self.assertAlmostEqual(row['identity'], .85, places=3)
        measured = [r for r in rep['line']['series'] if r['status'] in ('complete', 'partial')]
        self.assertTrue(measured and all(r['geometric'] is not None and r['identity'] is not None for r in measured))


class Stabilization(unittest.TestCase):
    """Formation periods around data gaps, lineups and confidence."""

    def test_formation_between_data_gaps_keeps_its_label(self):
        times = np.arange(0, 70, .1)
        vis = np.ones((len(times), 10), bool)
        vis[(times < 20) | (times >= 45), 4:] = False      # only four players observed outside 20-45 s
        r = syn.make_result(times, two_teams(times, syn.team_track(times, '4-3-3', 'left', 46)), visible={'A': vis})
        labels = [p['label'] for p in ts.TeamShape(r).shapes['A']['periods']]
        self.assertEqual(labels, ['insufficient', '4-3-3', 'insufficient'])

    def test_short_formation_between_formations_is_a_transition(self):
        times = np.arange(0, 180, .1)
        parts = [(times < 80, '4-4-2', 47), ((times >= 80) & (times < 100), '3-5-2', 48), (times >= 100, '4-4-2', 49)]
        a = np.concatenate([syn.team_track(times[m], lay, 'left', s) for m, lay, s in parts])
        labels = [p['label'] for p in ts.TeamShape(syn.make_result(times, two_teams(times, a))).shapes['A']['periods']]
        self.assertNotIn('3-5-2', labels)                  # 20 s is not a formation period

    def test_change_during_a_gap_starts_after_the_gap(self):
        times = np.arange(0, 140, .1)
        a = np.concatenate([syn.team_track(times[times < 68], '4-4-2', 'left', 50), syn.team_track(times[times >= 68], '4-3-3', 'left', 51)])
        vis = np.ones((len(times), 10), bool)
        vis[(times >= 62) & (times < 74), 4:] = False
        shape = ts.TeamShape(syn.make_result(times, two_teams(times, a), visible={'A': vis}))
        periods = shape.shapes['A']['periods']
        named = [p for p in periods if p['label'] in ts.TEMPLATES]
        self.assertEqual([p['label'] for p in named], ['4-4-2', '4-3-3'])
        self.assertLessEqual(named[1]['t0'], 80)          # the new run starts fresh after the gap
        self.assertTrue(all(p['label'] != '4-4-2' or p['t1'] <= 70 for p in periods))

    def test_nine_player_lineup_can_be_clear(self):
        times = np.arange(0, 60, .1)
        nine = [(28, 7), (24, 24), (24, 44), (28, 61), (42, 9), (39, 26), (39, 42), (42, 59), (56, 34)]
        r = syn.make_result(times, two_teams(times, syn.team_track(times, nine, 'left', 21)))
        period = ts.TeamShape(r, settings={'lineup': {'A': 9}}).shapes['A']['periods'][0]
        self.assertEqual((period['label'], period['confidence']), ('4-4-1', 'clear'))

    def test_low_identity_confidence_is_never_clear(self):
        times, teams, ends = syn.match('4-4-2', '4-4-2', seconds=60, seed=52)
        sure = ts.TeamShape(syn.make_result(times, teams, ends)).shapes['A']['periods'][0]
        unsure = ts.TeamShape(syn.make_result(times, teams, ends, identity_confidence=.45)).shapes['A']['periods'][0]
        self.assertEqual(sure['confidence'], 'clear')
        self.assertEqual(unsure['label'], '4-4-2')
        self.assertNotEqual(unsure['confidence'], 'clear')
        self.assertAlmostEqual(unsure['identity'], .45, places=2)

    def test_formation_needs_the_evidence_tail_direction(self):
        """No goal ends: the last seconds after the final 30 s evidence window still take the evidence's side."""
        for seconds in (62, 60.1):
            times, teams, _ = syn.match('4-3-3', '4-4-2', seconds=seconds, seed=53)
            shape = ts.TeamShape(syn.make_result(times, teams, {}))
            self.assertNotIn('unknown', set(shape.directions['A']['status']), seconds)
            self.assertEqual([p['label'] for p in shape.shapes['A']['periods']], ['4-3-3'], seconds)


class DefensiveUnit(unittest.TestCase):
    def test_coach_label_on_an_unclear_period_decides_the_unit(self):
        times = np.arange(0, 60, .1)
        r = syn.make_result(times, two_teams(times, syn.team_track(times, FORMLESS, 'left', 37)))
        plain = ts.TeamShape(r)
        self.assertEqual([p['label'] for p in plain.shapes['A']['periods']], ['unclear'])
        label = {'team': 'A', 't0': 0, 't1': 59.9, 'formation': '4-4-2', 'predicted': 'unclear'}
        shape = ts.TeamShape(r, settings={'formations': [label]})
        line = ts.defensive_line(shape.data, 'A', shape.directions, shape.shapes['A'], 20, 40)
        self.assertTrue(all(x['status'] in ('complete', 'partial') and x['source'] == 'formation' and x['unit'] == 4 for x in line))
        self.assertEqual(shape.shapes['A']['periods'][0]['label'], 'unclear')     # the prediction stays

    def test_sustained_swap_into_the_back_line_is_followed(self):
        times = np.arange(0, 140, .1)

        def swap(t, pos):
            pos = pos.copy()
            on = (t >= 40) & (t < 100)                      # the left centre-back and the holding midfielder swap places
            pos[on, 1], pos[on, 4] = pos[on, 4].copy(), pos[on, 1].copy()
            return pos
        a = syn.team_track(times, '4-1-4-1', 'left', 54, overrides=swap)
        shape = ts.TeamShape(syn.make_result(times, two_teams(times, a)))

        def unit(t0, t1):
            line = ts.defensive_line(shape.data, 'A', shape.directions, shape.shapes['A'], t0, t1)
            return {d['id'] for x in line for d in x.get('defenders', [])}
        self.assertIn('A-02', unit(15, 20))
        self.assertNotIn('A-05', unit(15, 20))
        self.assertIn('A-05', unit(68, 72))
        self.assertNotIn('A-02', unit(68, 72))

    def test_uneven_back_three(self):
        times = np.arange(0, 60, .1)

        def step_up(t, pos):
            pos = pos.copy()
            pos[(t >= 30) & (t < 36), 0, 0] += 14           # the left centre-back steps into midfield
            return pos
        a = syn.team_track(times, '3-5-2', 'left', 55, overrides=step_up, block=0, stretch=0)
        shape = ts.TeamShape(syn.make_result(times, two_teams(times, a)))
        line = ts.defensive_line(shape.data, 'A', shape.directions, shape.shapes['A'], 31, 35)
        self.assertTrue(all(x['unit'] == 3 and x['quality'] == 'uneven' for x in line), {(x.get('unit'), x.get('quality')) for x in line})

    def test_deepest_defender_only_from_complete_lines(self):
        times, teams, ends = syn.match('4-4-2', '4-4-2', seconds=40, seed=6)
        vis = np.ones((len(times), 10), bool)
        vis[len(times)//2:, 1] = False                      # a centre-back, often the deepest, leaves the view
        shape = ts.TeamShape(syn.make_result(times, teams, ends, visible={'A': vis}))
        late = ts.line_summary(ts.defensive_line(shape.data, 'A', shape.directions, shape.shapes['A'], 25, 40))
        self.assertIsNotNone(late['height'])
        self.assertIsNone(late['deepest'])                  # with one defender unmeasured, the deepest is unknown
        whole = ts.line_summary(ts.defensive_line(shape.data, 'A', shape.directions, shape.shapes['A'], 0, 40))
        self.assertIsNotNone(whole['deepest'])

    def test_depth_fallback_keeps_the_directions_apart(self):
        """Without a formation (an 8-player lineup) and a side swap inside a 30 s window, the deepest players are found
        per direction: on each side of the swap the line is the back four's."""
        r = Directions().result({'left': 'A', 'right': 'B'}, flip_after=45.0, n=450)
        shape = ts.TeamShape(r, settings={'direction': {'A': 'left', 'switchAt': 45.0}, 'lineup': {'A': 8}})
        line = ts.defensive_line(shape.data, 'A', shape.directions, shape.shapes['A'], 35, 55)
        measured = [x for x in line if x['status'] in ('complete', 'partial')]
        self.assertTrue(len(measured) > .8*len(line))
        self.assertTrue(all(x['source'] == 'depth rank' and 23 <= x['height'] <= 29 for x in measured), {x.get('height') for x in measured})


class SettingsAndRanges(unittest.TestCase):
    def test_malformed_settings_never_raise(self):
        for raw in ({'formations': 5}, {'formations': [{'team': 'A', 't0': 0, 't1': 9, 'formation': ['4-4-2']}]},
                    {'formations': [{'team': ['A'], 't0': 0, 't1': 9, 'formation': {'x': 1}}]}, {'direction': 'left'},
                    {'direction': {'A': ['left']}}, {'lineup': {'A': True, 'B': '9'}}, {'lineup': 4}, [], None, 'x'):
            s = ts.clean_settings(raw)
            self.assertEqual(s['formations'], [])
            self.assertIsNone(s['direction']['A'])
            self.assertEqual(s['lineup'], {'A': None, 'B': None})
        s = ts.clean_settings({'lineup': {'A': {'outfield': 9, 'from': 61.234}, 'B': {'outfield': 9}}, 'direction': {'A': 'right', 'switchAt': 2700}})
        self.assertEqual(s['lineup'], {'A': {'outfield': 9, 'from': 61.23}, 'B': 9})
        self.assertEqual(s['direction'], {'A': 'right', 'switchAt': 2700.0})

    def test_ranges_outside_or_reversed(self):
        times, teams, ends = syn.match('4-4-2', '4-4-2', seconds=60, seed=56)
        shape = ts.TeamShape(syn.make_result(times, teams, ends))
        for t0, t1 in ((-50, -10), (100, 200)):
            rep = shape.report('A', t0, t1)
            json.dumps(rep)
            self.assertEqual(rep['range']['note'], 'outside')
            self.assertEqual(rep['widthDepth']['series'], [])
            self.assertEqual(rep['widthDepth']['width']['reliableShare'], 0.0)
            self.assertIsNone(rep['line']['summary']['height'])
            self.assertEqual(rep['formation']['periods'], [])
        rep = shape.report('A', 30, 10)
        self.assertEqual((rep['range']['t0'], rep['range']['t1'], rep['range']['note']), (10, 30, 'reversed'))
        rep = shape.report('A', -5, 20)
        self.assertEqual((rep['range']['t0'], rep['range']['note']), (0, 'clipped'))

    def test_label_belongs_only_to_the_periods_it_covers(self):
        times = np.arange(0, 180, .1)
        a = np.concatenate([syn.team_track(times[times < 90], '4-4-2', 'left', 12), syn.team_track(times[times >= 90], '4-3-3', 'left', 13)])
        r = syn.make_result(times, two_teams(times, a, seed=14))
        shape = ts.TeamShape(r, settings={'formations': [{'team': 'A', 't0': 0, 't1': 95, 'formation': '4-2-3-1', 'predicted': '4-4-2'}]})
        named = [p for p in shape.shapes['A']['periods'] if p['label'] in ts.TEMPLATES]
        self.assertEqual([p['label'] for p in named], ['4-4-2', '4-3-3'])
        self.assertEqual(named[0]['correction']['formation'], '4-2-3-1')
        self.assertNotIn('correction', named[1])

    def test_chart_thinning_reports_its_step(self):
        rows, step = ts._thin(list(range(4000)))
        self.assertEqual((len(rows), step), (1334, 3))
        times, teams, ends = syn.match('4-4-2', '4-4-2', seconds=20, seed=57)
        rep = ts.TeamShape(syn.make_result(times, teams, ends)).report('A')
        self.assertAlmostEqual(rep['chartStep'], .2)

    def test_snapshot_uses_the_coach_label_and_the_lineup_at_its_time(self):
        times = np.arange(0, 60, .1)
        r = syn.make_result(times, two_teams(times, syn.team_track(times, FORMLESS, 'left', 37)))
        shape = ts.TeamShape(r, settings={'formations': [{'team': 'A', 't0': 0, 't1': 59.9, 'formation': '4-4-2'}],
                                          'lineup': {'A': {'outfield': 10, 'from': 0}}})
        snap = shape.snapshot('A', 30)
        self.assertEqual((snap['formation']['fitted'], snap['formation']['coach'], snap['expected']), ('4-4-2', '4-4-2', 10))
        self.assertEqual(snap['period']['label'], 'unclear')


# A back three: the 4-4-2's fullbacks push up as wing-backs and A-06 drops between the centre-backs.
L352 = [(41, 4), (25, 17), (25, 51), (41, 64), (38, 22), (23, 34), (35, 34), (38, 46), (55, 28), (55, 40)]


class Robustness(unittest.TestCase):
    """Split identities, ghosts, estimates and short periods: the cases a second review reproduced."""

    def three_back_spell(self):
        times = np.arange(0, 200, .1)

        def change(t, pos):
            pos = pos.copy()
            pos[(t >= 80) & (t < 110)] = np.array(L352, float)
            return pos
        a = syn.team_track(times, '4-4-2', 'left', 12, overrides=change)
        return syn.make_result(times, two_teams(times, a, seed=14))

    def test_a_short_period_keeps_its_own_back_line(self):
        r = self.three_back_spell()
        shape = ts.TeamShape(r)
        self.assertEqual([p['label'] for p in shape.shapes['A']['periods']], ['4-4-2', '3-5-2', '4-4-2'])
        line = ts.defensive_line(shape.data, 'A', shape.directions, shape.shapes['A'], 81, 109)
        self.assertTrue(all(x['status'] == 'complete' and {d['id'] for d in x['defenders']} == {'A-02', 'A-03', 'A-06'} for x in line))

    def test_coach_label_on_a_short_period_is_not_outvoted(self):
        r = self.three_back_spell()
        period = next(p for p in ts.TeamShape(r).shapes['A']['periods'] if p['label'] == '3-5-2')
        shape = ts.TeamShape(r, settings={'formations': [{'team': 'A', 't0': period['t0'], 't1': period['t1'], 'formation': '5-3-2'}]})
        line = ts.defensive_line(shape.data, 'A', shape.directions, shape.shapes['A'], period['t0']+1, period['t1']-1)
        self.assertTrue(all(x['unit'] == 5 and 'A-06' in {d['id'] for d in x['defenders']} for x in line))

    def test_no_missing_defender_at_a_sustained_swap(self):
        times = np.arange(0, 140, .1)

        def swap(t, pos):
            pos = pos.copy()
            on = (t >= 40) & (t < 100)
            pos[on, 1], pos[on, 4] = pos[on, 4].copy(), pos[on, 1].copy()
            return pos
        shape = ts.TeamShape(syn.make_result(times, two_teams(times, syn.team_track(times, '4-1-4-1', 'left', 54, overrides=swap))))
        line = ts.defensive_line(shape.data, 'A', shape.directions, shape.shapes['A'], 0, 140)
        self.assertTrue(all(x['status'] == 'complete' and len(x['defenders']) == 4 for x in line))   # also at 42-48 s and 102-108 s

    def test_short_label_does_not_take_over_a_long_period(self):
        times = np.arange(0, 180, .1)
        a = np.concatenate([syn.team_track(times[times < 90], '4-4-2', 'left', 12), syn.team_track(times[times >= 90], '4-3-3', 'left', 13)])
        r = syn.make_result(times, two_teams(times, a, seed=14))
        for t0, t1 in ((78, 100), (85, 95)):
            shape = ts.TeamShape(r, settings={'formations': [{'team': 'A', 't0': t0, 't1': t1, 'formation': '5-3-2'}]})
            self.assertFalse(any(p.get('correction') for p in shape.shapes['A']['periods']), (t0, t1))
            self.assertEqual([c['applied'] for c in shape.report('A')['formation']['corrections']], [False])

    def test_more_players_than_the_lineup_is_never_clear(self):
        times, teams, ends = syn.match('4-4-2', '4-4-2', seconds=60, seed=42)
        shape = ts.TeamShape(syn.make_result(times, teams, ends), settings={'lineup': {'A': 9}})
        ok = [w for w in shape.shapes['A']['windows'] if w['status'] == 'ok']
        left_out = {frozenset(f'A-{k:02d}' for k in range(1, 11))-frozenset(w['ids']) for w in ok}
        self.assertEqual(len(left_out), 1)                     # the same player is left out of every window
        self.assertTrue(all(p.get('confidence') != 'clear' for p in shape.shapes['A']['periods']))

    def test_flickering_duplicate_is_removed(self):
        base = [p for p in WidthDepth().ten() if p[0] != 'A-05']            # nine observed, the widest one unseen
        a03 = next(p for p in base if p[0] == 'A-03')
        ghost = ('A-11', 'A', 'PLAYER', a03[3]+.1, a03[4], {'identityConfidence': .5})
        frames = [base+[ghost] if k % 3 else base for k in range(30)]        # the split identity on 2 of every 3 samples
        shape, series = series_for(frame_result(frames, roster=[('A-05', 'A')]))
        self.assertTrue(all(s['players'] == 9 and s['widthStatus'] != 'complete' for s in series))
        self.assertGreater(shape.data.teams['A']['excluded']['duplicateIdentity'], 0)

    def test_ghost_of_a_departed_player_does_not_remove_the_substitute(self):
        times = np.arange(0, 60, .1)
        a = syn.team_track(times, '4-4-2', 'left', 60)
        sub = a[:, 4].copy()
        a = np.concatenate([a, sub[:, None]], axis=1)          # A-11 comes on for A-05 at 30 s, in the same spot
        a[times >= 30, 4] = np.nan
        a[times < 30, 10] = np.nan
        ghost = {k: [{'id': 'A-05', 'team': 'A', 'role': 'PLAYER', 'state': 'confirmed', 'identityConfidence': .95,
                      'pitch': [float(sub[k, 0])+.15/syn.LENGTH, float(sub[k, 1])], 'pitchConfidence': .9, 'pitchSigma': [.4, .4]}]
                 for k in range(len(times)) if times[k] >= 30}
        r = syn.make_result(times, two_teams(times, a), extra_people=ghost)
        for slot in r['roster']['slots']:
            if slot['id'] == 'A-05':
                slot['left'] = {'time': 30.0}
        shape = ts.TeamShape(r)
        late = [s for s in ts.width_depth_series(shape.data, 'A', 32, 60)]
        self.assertTrue(all(s['players'] == 10 for s in late))
        j = shape.data.teams['A']['ids'].index('A-11')
        self.assertTrue(np.isfinite(shape.data.teams['A']['x'][shape.data.t >= 32, j]).all())

    def test_duplicate_chain_does_not_depend_on_the_names(self):
        base = outfield('A', [28, 40, 50, 60, 45, 55, 35], [12, 20, 30, 40, 25, 35, 45])
        for p, g, q in (('A-08', 'A-10', 'A-09'), ('A-09', 'A-10', 'A-08')):
            people = base+[(p, 'A', 'PLAYER', 70.0, 50.0, {'identityConfidence': .4}), (g, 'A', 'PLAYER', 70.35, 50.0, {'identityConfidence': .6}),
                           (q, 'A', 'PLAYER', 70.7, 50.0, {'identityConfidence': .9})]
            shape, series = series_for(frame_result([people]*5))
            self.assertEqual(series[0]['players'], 9, (p, g, q))   # only the ghost between the two real players goes
            self.assertFalse(np.isfinite(shape.data.teams['A']['x'][:, shape.data.teams['A']['ids'].index(g)]).any())

    def test_split_identity_estimate_is_not_the_missing_player(self):
        observed = [p for p in WidthDepth().ten() if p[0] != 'A-05']
        a03 = next(p for p in observed if p[0] == 'A-03')
        frames = [observed]*5
        timeline = WidthDepth.timeline(frames, {'A-11': ((a03[3]+.14)/L, a03[4]/W, 2.0)})     # a stale prediction on A-03
        _, series = series_for(frame_result(frames, timeline=timeline, roster=[('A-05', 'A'), ('A-11', 'A')]))
        self.assertEqual((series[0]['widthStatus'], series[0]['widthEdgeRisk']), ('partial', 'unknown'))

    def test_two_estimates_of_one_person_count_once(self):
        observed = [p for p in WidthDepth().ten() if p[0] not in ('A-05', 'A-10')]
        frames = [observed]*5
        timeline = WidthDepth.timeline(frames, {'A-10': (50/L, 15/W, 2.0), 'A-11': (50.14/L, 15/W, 3.0)})
        shape, est = series_for(frame_result(frames, timeline=timeline, roster=[('A-05', 'A'), ('A-10', 'A'), ('A-11', 'A')]), mode='estimated')
        self.assertEqual((est[0]['players'], est[0]['estimated']), (9, 1))
        self.assertEqual(est[0]['widthStatus'], 'partial')
        self.assertGreater(shape.data.teams['A']['excluded']['estimateDuplicate'], 0)

    def test_spread_mostly_of_estimates_is_not_partial(self):
        people = WidthDepth().ten()
        frames = [people[:2]]*5
        estimates = {p[0]: (p[3]/L, p[4]/W, 2.0) for p in people[2:7]}
        r = frame_result(frames, timeline=WidthDepth.timeline(frames, estimates), roster=[(p[0], 'A') for p in people[2:]])
        shape, est = series_for(r, mode='estimated')
        self.assertEqual((est[0]['observed'], est[0]['estimated'], est[0]['widthStatus']), (2, 5, 'estimated'))
        agg = ts.team_width_depth(shape.data, 'A', shape.data.start, shape.data.end)['width']
        self.assertEqual(agg['partialShare'], 0.0)

    def test_small_lineup_fully_observed_is_complete_in_estimated_mode(self):
        people = WidthDepth().ten()[:7]
        _, est = series_for(frame_result([people]*5, expected=8), mode='estimated')     # eight a side: seven outfield
        self.assertEqual((est[0]['expected'], est[0]['widthStatus']), (7, 'complete'))

    def test_more_observed_than_the_lineup_is_excess(self):
        times = np.arange(0, 60, .1)

        def walk_off(t, pos):
            pos = pos.copy()
            pos[t >= 30, 4] = (60, 67.6)          # sent off at 30 s, still tracked along the touchline, not marked as left
            return pos
        r = syn.make_result(times, two_teams(times, syn.team_track(times, '4-4-2', 'left', 41, overrides=walk_off)))
        shape = ts.TeamShape(r, settings={'lineup': {'A': {'outfield': 9, 'from': 30}}})
        late = [s for s in ts.width_depth_series(shape.data, 'A', 35, 60) if s['observed'] == 10]
        self.assertTrue(late and all(s['widthStatus'] == 'excess' for s in late))
        agg = shape.report('A', 35, 60)['widthDepth']['width']
        self.assertGreater(agg['excessShare'], 0)

    def test_line_of_mostly_predicted_defenders_is_estimated(self):
        times, teams, ends = syn.match('4-4-2', '4-4-2', seconds=60, seed=61)
        vis = np.ones((len(times), 10), bool)
        hidden = (times >= 30) & (times < 36)
        vis[hidden, :3] = False
        out = teams['A'][0]
        ids = {f'A-{j+1:02d}': {'team': 'A', 'x': [float(v) for v in out[:, j, 0]], 'y': [float(v) for v in out[:, j, 1]],
                                'kind': ['P' if h else 'O' for h in hidden], 'sigma': [[2.0, 2.0]]*len(times)} for j in range(3)}
        r = syn.make_result(times, teams, ends, visible={'A': vis}, timeline={'frameTimes': [float(t) for t in times], 'ids': ids})
        line = ts.defensive_line(*(lambda s: (s.data, 'A', s.directions, s.shapes['A'], 31, 35))(ts.TeamShape(r, mode='estimated')))
        self.assertTrue(line and all(x['status'] == 'estimated' for x in line), {x['status'] for x in line})
        observed = ts.defensive_line(*(lambda s: (s.data, 'A', s.directions, s.shapes['A'], 31, 35))(ts.TeamShape(r)))
        self.assertTrue(all(x['status'] == 'unavailable' for x in observed))

    def test_report_uses_the_lineup_of_its_range(self):
        times, teams, ends = syn.match('4-4-2', '4-4-2', seconds=60, seed=62)
        shape = ts.TeamShape(syn.make_result(times, teams, ends), settings={'lineup': {'A': {'outfield': 9, 'from': 30}}})
        rep = shape.report('A', 35, 60)
        self.assertEqual((rep['formation']['expected'], rep['coverage']['expected'], rep['coverage']['default']), (9, 9, 10))
        self.assertEqual(rep['formation']['candidates'], list(ts.FORMATIONS_9))

    def test_numbers_too_large_for_a_float_are_dropped(self):
        s = ts.clean_settings({'direction': {'A': 'left', 'switchAt': 10**400}, 'lineup': {'A': {'outfield': 9, 'from': 10**400}},
                               'formations': [{'team': 'A', 't0': 0, 't1': 10**400, 'formation': '4-4-2'}]})
        self.assertEqual((s['direction'], s['lineup']['A'], s['formations']), ({'A': 'left', 'switchAt': None}, 9, []))


if __name__ == '__main__':
    unittest.main()
