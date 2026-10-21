"""The coaching series (soccer/coaching/series.py): every coaching number is measured on the team-shape grid from
observed positions only, and a sample that cannot be trusted is ineligible with a reason instead of a value: the
registration gates, complete teams for hull and spread, fully seen units under the template's line index, the voted
team centre, the certainty rule for the nearest teammate, the gated ball context with hysteresis, and the stabilized
flags as an evidence veto only. Fields a result does not have (a Metrica-shaped result) never veto.

Run:  python -m unittest test_coaching_series          (from this directory)
Real runs (optional): set PITCHIQ_COACHING_SMOKE to result.json paths separated by the path separator.
"""
import inspect
import json
import os
import sys
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (ROOT, os.path.join(ROOT, 'tools'), HERE):
    if p not in sys.path:
        sys.path.insert(0, p)
from soccer import team_shape as ts  # noqa: E402
from soccer.coaching import episodes as ep, series as cs  # noqa: E402
import team_shape_synth as syn  # noqa: E402

L, W = syn.LENGTH, syn.WIDTH


def build(la='4-4-2', lb='4-2-3-1', seconds=90.0, fps=10.0, seed=0, edit=None, **kw):
    """A synthetic match (team A defends the left goal), optionally edited frame by frame: edit(k, t, frame)."""
    times, teams, ends = syn.match(la, lb, seconds=seconds, fps=fps, seed=seed)
    result = syn.make_result(times, teams, ends, **kw)
    if edit is not None:
        for k, f in enumerate(result['frames']):
            edit(k, f['time'], f)
    return times, teams, result


def truth_attack(teams, team, side, frame_index):
    """True outfield positions of a team in its attacking frame (metres) on the grid's frames."""
    out, _ = teams[team]
    x, y = out[frame_index, :, 0], out[frame_index, :, 1]
    X, Y = (x, y) if side == 'left' else (1-x, 1-y)
    return X*L, Y*W


class Fixture(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.times, cls.teams, cls.result = build('4-4-2', '4-2-3-1')
        cls.cd = cs.build_series(cls.result)


class GridAndApi(Fixture):
    def test_the_signature_is_the_contract(self):
        params = inspect.signature(cs.build_series).parameters
        self.assertEqual(list(params), ['result', 'samples', 'shape', 'settings', 'mode'])
        self.assertEqual([params[k].default for k in ('samples', 'shape', 'settings', 'mode')], [None, None, None, 'observed'])
        with self.assertRaises(ValueError):
            cs.build_series(self.result, mode='guessed')

    def test_the_grid_is_the_team_shape_grid_with_absolute_frames(self):
        cd = self.cd
        data = cd.shape.observed
        np.testing.assert_array_equal(cd.t, data.t)
        frames = self.result['frames']
        np.testing.assert_array_equal(cd.frame, [frames[i]['frame'] for i in data.frame_index])
        for team in 'AB':
            s = cd.teams[team]
            self.assertIs(s.t, cd.t)
            np.testing.assert_array_equal(s.frame, cd.frame)
            self.assertEqual(s.ids, list(data.teams[team]['ids']))
            for metric in cs.TEAM_METRICS:
                self.assertEqual(s.eligible[metric].shape, (len(cd.t),), metric)
                self.assertEqual(s.reasons[metric].shape, (len(cd.t),), metric)
            self.assertEqual(s.eligible['nearestTeammate'].shape, (len(cd.t), len(s.ids)))
            np.testing.assert_allclose(s.segment_t, cd.t-frames[0]['time'])
        self.assertAlmostEqual(cd.seek(3), (cd.frame[3]+.5)/cd.fps)
        self.assertEqual(cd.sample_at(cd.t[7]+.01), 7)

    def test_a_given_shape_gives_the_same_series(self):
        shape = ts.TeamShape(self.result)
        again = cs.build_series(self.result, shape=shape)
        for team in 'AB':
            a, b = self.cd.teams[team], again.teams[team]
            for name in ('width', 'hull', 'gapDM', 'gapMF', 'lineStd', 'centreX', 'nearestTeammate'):
                np.testing.assert_array_equal(a.values[name], b.values[name], err_msg=name)
            for metric in cs.TEAM_METRICS+cs.PLAYER_METRICS:
                np.testing.assert_array_equal(a.reasons[metric], b.reasons[metric], err_msg=metric)

    def test_the_summary_is_json_safe(self):
        text = json.dumps(self.cd.summary(), allow_nan=False)
        back = json.loads(text)
        self.assertEqual(set(back['teams']), {'A', 'B'})
        self.assertEqual(set(back['teams']['A']), set(cs.TEAM_METRICS+cs.PLAYER_METRICS))


class FullCoverage(Fixture):
    def test_width_and_depth_are_team_shapes_complete_values(self):
        data = self.cd.shape.observed
        for team in 'AB':
            s = self.cd.teams[team]
            ref = ts.width_depth_series(data, team, data.start, data.end)
            want = np.array([r['width']['metres'] if r['widthStatus'] in ('complete', 'reliable') else np.nan for r in ref])
            np.testing.assert_array_equal(s.width, want)
            self.assertTrue(s.eligible['width'].all() and s.eligible['depth'].all())

    def test_hull_and_spread_on_complete_samples(self):
        s = self.cd.teams['A']
        self.assertTrue(s.eligible['hull'].all())
        k = 40
        P = np.array([(s.x[k, j]*L, s.y[k, j]*W) for j in range(len(s.ids))])
        c = P.mean(axis=0)
        self.assertAlmostEqual(s.spread[k], float(np.sqrt(((P-c)**2).sum(axis=1).mean())), places=6)
        self.assertGreater(s.hull[k], 400)
        square = cs._hull_area(np.array([0, 10, 10, 0, 5.0]), np.array([0, 0, 10, 10, 5.0]))
        self.assertAlmostEqual(square, 100.0)

    def test_units_follow_the_template_lines(self):
        cd = self.cd
        fi = cd.shape.observed.frame_index
        for team, layout, side in (('A', '4-4-2', 'left'), ('B', '4-2-3-1', 'right')):
            s = cd.teams[team]
            lines = np.array(syn.LINES[layout])
            X, _ = truth_attack(self.teams, team, side, fi)
            self.assertTrue((s.formation == layout).mean() > .95)
            ok = s.eligible['gapDM']
            self.assertGreater(ok.mean(), .95)
            # Back line to the FIRST midfield line (template line index 1), centre to centre.
            truth_dm = X[:, lines == 1].mean(axis=1)-X[:, lines == 0].mean(axis=1)
            np.testing.assert_allclose(s.gapDM[ok], truth_dm[ok], atol=1e-6)
            # Midfield (every middle line) to the forwards.
            middle = (lines > 0) & (lines < lines.max())
            truth_mf = X[:, lines == lines.max()].mean(axis=1)-X[:, middle].mean(axis=1)
            okf = s.eligible['gapMF']
            np.testing.assert_allclose(s.gapMF[okf], truth_mf[okf], atol=1e-6)
            u = s.units[int(np.flatnonzero(ok)[0])]
            self.assertEqual(u['D'], [s.ids[j] for j in np.flatnonzero(lines == 0)])
            self.assertEqual(u['M1'], [s.ids[j] for j in np.flatnonzero(lines == 1)])
        # In a 4-2-3-1 the middle group spans two lines: M1 is the holding pair, not the whole midfield.
        b = cd.teams['B']
        self.assertEqual(len(b.units[100]['M1']), 2)
        self.assertEqual(len(b.units[100]['M']), 5)
        self.assertLess(np.nanmedian(b.gapDM), np.nanmedian(b.MX-b.DX)-4)

    def test_the_line_is_the_defensive_lines_complete_samples(self):
        data = self.cd.shape.observed
        s = self.cd.teams['A']
        rows = ts.defensive_line(data, 'A', self.cd.shape.directions, self.cd.shape.shapes['A'], data.start, data.end)
        k = 60
        dd = np.array([v['distance'] for v in rows[k]['defenders']])
        self.assertAlmostEqual(s.lineStd[k], float(dd.std()))
        self.assertEqual(s.lineHeight[k], rows[k]['height'])
        self.assertTrue((s.lineSize[s.eligible['lineSize']] == 4).all())
        self.assertTrue(s.eligible['lineStd'].mean() > .95)

    def test_centre_and_third(self):
        s = self.cd.teams['A']
        X, Y = truth_attack(self.teams, 'A', 'left', self.cd.shape.observed.frame_index)
        np.testing.assert_allclose(s.centreX, X.mean(axis=1), atol=1e-6)
        np.testing.assert_allclose(s.centreY, Y.mean(axis=1), atol=1e-6)
        self.assertTrue((s.centreSource == 'complete').all())
        own = s.centreX < L/3
        final = s.centreX > 2*L/3
        self.assertTrue((s.third[own] == 'own').all() and (s.third[final] == 'final').all())
        self.assertTrue((s.third[~own & ~final] == 'middle').all())
        self.assertTrue(own.any() and (~own).any())

    def test_nearest_teammate_without_a_view_field_is_measured_everywhere(self):
        s = self.cd.teams['A']
        k = 30
        P = np.array([(s.x[k, j]*L, s.y[k, j]*W) for j in range(len(s.ids))])
        D = np.hypot(*(P[:, None, :]-P[None, :, :]).transpose(2, 0, 1))
        np.fill_diagonal(D, np.inf)
        np.testing.assert_allclose(s.nearestTeammate[k], D.min(axis=1))
        self.assertTrue(s.eligible['nearestTeammate'].all())

    def test_absent_ball_is_not_measured(self):
        self.assertFalse(self.cd.available['ball'])
        s = self.cd.teams['A']
        self.assertFalse(s.eligible['ball'].any())
        self.assertEqual(set(s.reasons['ball'].tolist()), {cs.NO_BALL})


class Coverage(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        times, teams, _ = syn.match('4-4-2', '4-4-2', seconds=90, fps=10)
        visible = {'A': np.ones((len(times), 10), bool)}
        cls.hidden = (times >= 40) & (times < 50)
        visible['A'][cls.hidden, 0] = False                  # a defender off camera for 10 s
        cls.result = syn.make_result(times, teams, {'left': 'A', 'right': 'B'}, visible=visible)
        cls.cd = cs.build_series(cls.result)

    def test_a_missing_defender_makes_the_team_and_its_units_unmeasured(self):
        s = self.cd.teams['A']
        hid = (s.t >= 40) & (s.t < 50)
        for metric in ('width', 'hull', 'spread'):
            self.assertFalse(s.eligible[metric][hid].any(), metric)
            self.assertEqual(set(s.reasons[metric][hid].tolist()), {cs.INCOMPLETE_TEAM}, metric)
        for metric in ('gapDM', 'lineStd'):
            self.assertFalse(s.eligible[metric][hid].any(), metric)
            self.assertTrue(set(s.reasons[metric][hid].tolist()) <= {cs.INCOMPLETE_UNITS, cs.NO_FORMATION}, metric)
        self.assertTrue(s.eligible['gapMF'][hid].mean() > .9)      # midfield and forwards are still fully seen
        self.assertTrue(np.isnan(s.width[hid]).all() and np.isnan(s.gapDM[hid]).all())
        self.assertTrue(s.eligible['width'][~hid].all())

    def test_values_exist_only_where_eligible(self):
        for team in 'AB':
            s = self.cd.teams[team]
            for name, metric in cs.VALUE_METRIC.items():
                v, e = s.values[name], s.eligible[metric]
                if v.dtype.kind == 'U':
                    self.assertTrue((v[~e] == '').all() and (v[e] != '').all(), (team, name))
                else:
                    self.assertTrue(np.isnan(v[~e]).all() and np.isfinite(v[e]).all(), (team, name))

    def test_coverage_counts_measured_time_and_stretches(self):
        s = self.cd.teams['A']
        c = s.coverage('hull')
        self.assertAlmostEqual(c['measuredSeconds'], 80.0, delta=.3)
        self.assertAlmostEqual(c['eligibleShare'], 80/90, delta=.005)
        self.assertAlmostEqual(c['longestStretchSeconds'], 40.0, delta=.3)
        self.assertEqual(c['measuredSeconds'], round(ep.measured_seconds(s.t, s.eligible['hull'], s.step), 2))
        not_measured = s.reason_seconds('hull')
        self.assertAlmostEqual(c['measuredSeconds']+sum(not_measured.values()), len(s.t)*s.step, places=6)
        self.assertEqual(set(not_measured), {cs.INCOMPLETE_TEAM})
        player = s.coverage('nearestTeammate', s.ids[0])
        self.assertAlmostEqual(player['measuredSeconds'], 80.0, delta=.3)
        self.assertEqual(s.reason_seconds('nearestTeammate', s.ids[0]), {cs.NOT_OBSERVED: 10.0})
        self.assertEqual(s.coverage('nearestTeammate', 'nobody')['measuredSeconds'], 0.0)


class Registration(unittest.TestCase):
    def test_the_final_registration_gates_every_metric(self):
        def edit(k, t, f):
            conf, dis = .8, 1.0
            if 20 <= t < 25:
                conf = .1
            elif 30 <= t < 32:
                conf = .3
            elif 60 <= t < 62:
                dis = 4.5
            f['calibration'] = {'confidence': conf, 'disagreement': dis}
            f['pitchConfidence'] = .99                       # the live value: never used
            if 70 <= t < 71:
                f['calibrated'] = False
        _, _, result = build(edit=edit)
        cd = cs.build_series(result)
        s = cd.teams['A']
        t = s.t
        expect = {cs.NOT_CALIBRATED: ((t >= 20) & (t < 25)) | ((t >= 70) & (t < 71)), cs.LOW_CALIBRATION: (t >= 30) & (t < 32),
                  cs.DISAGREEMENT: (t >= 60) & (t < 62)}
        for metric in ('width', 'hull', 'centre', 'gapDM', 'lineStd', 'ball'):
            for code, where in expect.items():
                self.assertEqual(set(s.reasons[metric][where].tolist()), {code}, (metric, code))
        bad = np.any(list(expect.values()), axis=0)
        self.assertTrue(s.eligible['width'][~bad].all())
        self.assertTrue(np.isnan(s.width[bad]).all() and np.isnan(s.gapDM[bad]).all() and (s.third[bad] == '').all())
        self.assertTrue((s.reasons['nearestTeammate'][expect[cs.DISAGREEMENT]] == cs.DISAGREEMENT).all())
        self.assertTrue(cd.available['calibration'])
        np.testing.assert_array_equal(cd.calibration['calibrated'], ~expect[cs.NOT_CALIBRATED])


class VotedCentre(unittest.TestCase):
    def test_the_voted_centre_is_not_dragged_by_the_visible_side(self):
        times, teams, ends = syn.match('4-4-2', '4-4-2', seconds=90, fps=10)
        full = cs.build_series(syn.make_result(times, teams, ends)).teams['A']
        visible = {'A': np.ones((len(times), 10), bool)}
        off = (times % 5) < 2                                 # the three left-sided players leave the picture 40% of the time
        for j in (0, 1, 4):
            visible['A'][off, j] = False
        part = cs.build_series(syn.make_result(times, teams, ends, visible=visible)).teams['A']
        voted = part.centreSource == 'voted'
        self.assertGreater(voted.sum(), .3*len(part.t))
        self.assertTrue(part.eligible['centre'].all())
        err_voted = np.abs(part.centreY-full.centreY)[voted]
        with np.errstate(all='ignore'):
            naive = np.nanmean(np.where(part.kind == 'O', part.y, np.nan), axis=1)*W
        err_naive = np.abs(naive-full.centreY)[voted]
        self.assertLess(np.percentile(err_voted, 90), 1.5)
        self.assertGreater(np.median(err_naive), 3*np.median(err_voted)+2)
        # Complete samples keep the centroid itself.
        np.testing.assert_allclose(part.centreX[part.complete], full.centreX[part.complete], atol=1e-9)


class SwapRule(unittest.TestCase):
    def test_a_player_dropped_from_the_back_line_takes_the_line_of_the_one_who_moved_in(self):
        D, M, F = ['d1', 'd2', 'd3', 'd4'], ['m1', 'm2', 'm3', 'm4'], ['f1', 'f2']
        usual = {**{p: 'D' for p in D}, **{p: 'M' for p in M}, **{p: 'F' for p in F}}
        swapped = dict(usual, d4='M', m1='D')                 # one window: d4 stepped up, m1 dropped in
        centres = [0.0, 5.0, 10.0, 15.0, 20.0]
        wr = [(c, '4-4-2', swapped if c == 10.0 else usual, 4, 0) for c in centres]
        line_of = {**{p: 0 for p in ('d1', 'd2', 'd3', 'm1')}, **{p: 1 for p in ('d4', 'm2', 'm3', 'm4')}, 'f1': 2, 'f2': 2}
        u = cs._unit_groups(10.0, wr, centres, 2, line_of)
        self.assertEqual(u['D'], D)                           # the majority keeps the usual back four
        self.assertEqual(u['M1'], M)                          # m1 takes d4's place in the midfield line
        self.assertEqual(u['F'], F)
        self.assertTrue(all(u['full'].values()))
        short = {p: line for p, line in line_of.items() if p != 'm4'}
        v = cs._unit_groups(10.0, wr, centres, 2, short)
        self.assertFalse(v['full']['M1'])
        self.assertTrue(v['full']['D'] and v['full']['F'])


class NearestTeammate(unittest.TestCase):
    def test_certain_only_when_the_disc_is_in_view(self):
        half = [[-5/L, -5/W], [.5, -5/W], [.5, 1+5/W], [-5/L, 1+5/W]]       # the left half of the pitch (+5 m margins)

        def edit(k, t, f):
            f['view'] = half
        _, _, result = build('4-4-2', '4-4-2', edit=edit)
        s = cs.build_series(result).teams['A']                # A defends the left goal: pitch x is its attack X
        obs = s.kind == 'O'
        x = np.where(obs, s.x*L, np.nan)
        want = obs & (x+s.nearestTeammate <= L/2+1e-9)        # the right edge is the only one inside the pitch
        np.testing.assert_array_equal(s.eligible['nearestTeammate'], want)
        self.assertTrue(want.any() and (obs & ~want).any())
        self.assertEqual(set(s.reasons['nearestTeammate'][obs & ~want].tolist()), {cs.OUT_OF_VIEW})
        poly = np.array(half)*[L, W]
        inside = cs._disc_inside(poly, np.array([10.0, 50.0, 60.0]), np.array([1.0, 30, 30]), np.array([8.0, 3.0, 1.0]), L, W)
        self.assertEqual(inside.tolist(), [True, False, False])   # a disc over the touchline is fine; beyond the view is not

    def test_unconfirmed_and_anonymous_kit_people_are_teammates(self):
        times, teams, ends = syn.match('4-4-2', '4-4-2', seconds=60, fps=10)
        out = teams['A'][0]
        extra = {}
        for k in range(len(times)):
            x, y = out[k, 8]                                   # next to a forward, 3 m further up
            ghost = out[k, 2]                                  # on top of a centre-back: the same person seen twice
            extra[k] = [{'id': None, 'team': 'A', 'role': 'PLAYER', 'state': 'uncertain', 'pitch': [float(x+3/L), float(y)], 'pitchConfidence': .8},
                        {'id': 'A-11', 'team': 'A', 'role': 'PLAYER', 'state': 'confirmed', 'pitch': [float(ghost[0]+.2/L), float(ghost[1])], 'pitchConfidence': .8},
                        {'id': None, 'team': 'A', 'role': 'NON_TARGET', 'state': 'nontarget', 'pitch': [float(out[k, 3, 0]), float(out[k, 3, 1])], 'pitchConfidence': .8}]
        base = cs.build_series(syn.make_result(times, teams, ends)).teams['A']
        s = cs.build_series(syn.make_result(times, teams, ends, extra_people=extra)).teams['A']
        np.testing.assert_allclose(s.nearestTeammate[:, 8], np.minimum(base.nearestTeammate[:, 8], 3.0), atol=1e-6)
        others = [j for j in range(10) if j != 8]
        np.testing.assert_allclose(np.minimum(s.nearestTeammate[:, others], 50), np.minimum(base.nearestTeammate[:, others], 50), atol=3.01)
        np.testing.assert_allclose(s.nearestTeammate[:, 2], base.nearestTeammate[:, 2])      # the ghost is ignored
        self.assertTrue((s.nearestExtras == 1).all())

    def test_someone_in_the_kit_without_a_position_makes_it_uncertain(self):
        def edit(k, t, f):
            if 20 <= t < 22:
                f['people'].append({'id': None, 'team': 'A', 'role': 'PLAYER', 'state': 'uncertain', 'pitchStatus': 'unresolved'})
        _, _, result = build('4-4-2', '4-4-2', edit=edit)
        s = cs.build_series(result).teams['A']
        where = (s.t >= 20) & (s.t < 22)
        self.assertEqual(set(s.reasons['nearestTeammate'][where].ravel().tolist()), {cs.UNPLACED_TEAMMATE})
        self.assertTrue(s.eligible['nearestTeammate'][~where].all())
        self.assertTrue(s.eligible['width'][where].all())

    def test_small_scene_runs_veto_isolation_and_the_ball_only(self):
        def edit(k, t, f):
            f['profile'] = {'smallScene': True}
            f['ball'] = {'state': 'TRACKED', 'pitch': [.5, .5], 'pitchConfidence': .9, 'pitchSigma': [.3, .3]}
        _, _, result = build('4-4-2', '4-4-2', edit=edit)
        cd = cs.build_series(result)
        self.assertTrue(cd.small_scene)
        s = cd.teams['A']
        self.assertEqual(set(s.reasons['nearestTeammate'].ravel().tolist()), {cs.SMALL_SCENE})
        self.assertEqual(set(s.reasons['ball'].tolist()), {cs.SMALL_SCENE})
        self.assertTrue(s.eligible['width'].all() and s.eligible['gapDM'].mean() > .95)


class Ball(unittest.TestCase):
    def ball_run(self, xs_m, extra=None, seconds=40.0):
        def edit(k, t, f):
            i = min(int(round(t*5)), len(xs_m)-1)
            f['ball'] = {'state': 'TRACKED', 'origin': 'OBSERVATION', 'pitch': [xs_m[i]/L, .5], 'pitchConfidence': .9, 'pitchSigma': [.3, .4]}
            if extra:
                extra(t, f['ball'])
        _, _, result = build('4-4-2', '4-4-2', seconds=seconds, edit=edit)
        return cs.build_series(result)

    def test_the_third_changes_only_well_inside_the_new_one(self):
        n = 200
        xs = np.full(n, 30.0)
        xs[25:100] = 35+np.where(np.arange(75) % 2 == 0, 1.0, -1.0)    # hovering on the 35 m line
        xs[100:] = 45.0
        cd = self.ball_run(xs)
        a, b = cd.teams['A'], cd.teams['B']
        seg = (a.t >= 5.2) & (a.t < 19.8)
        self.assertEqual(set(a.ballThird[seg].tolist()), {'own'})          # A defends the left goal: 35 m is its line
        self.assertEqual(set(a.ballThird[a.t >= 20.4].tolist()), {'middle'})
        self.assertEqual(set(b.ballThird[a.t < 20].tolist()), {'final'})    # 70 m from B's goal, hovering on its line
        np.testing.assert_allclose(a.ballX[a.eligible['ball']], xs[np.round(a.t*5).astype(int)][a.eligible['ball']], atol=1e-6)
        np.testing.assert_allclose(b.ballX, L-a.ballX, atol=1e-9)
        self.assertEqual(set(a.ballChannel[a.eligible['ballChannel']].tolist()), {'centre'})

    def test_after_an_unmeasured_second_the_zone_is_decided_afresh(self):
        n = 200
        xs = np.full(n, 30.0)
        xs[50:] = 36.0                                                      # 1 m inside the middle third

        def gap(t, b):
            if 9 <= t < 10.6:
                b['state'] = 'UNKNOWN'
        a = self.ball_run(xs, gap).teams['A']
        self.assertEqual(set(a.ballThird[(a.t >= 10.6)].tolist()), {'middle'})
        self.assertEqual(set(a.reasons['ball'][(a.t >= 9) & (a.t < 10.6)].tolist()), {cs.NO_BALL})
        steady = self.ball_run(xs).teams['A']
        self.assertEqual(set(steady.ballThird[steady.t >= 10.6].tolist()), {'own'})   # without the gap: hysteresis holds

    def test_ball_gates(self):
        def bad(t, b):
            if 5 <= t < 6:
                b['pitchConfidence'] = .2
            elif 7 <= t < 8:
                b['pitchSigma'] = [.5, 4.0]
            elif 9 <= t < 10:
                b['pitch'] = [1.05, .5]
            elif 11 <= t < 12:
                b['state'] = 'MISSING'
                b['bridged'] = True
        cd = self.ball_run(np.full(200, 50.0), bad, seconds=20)
        a = cd.teams['A']
        t = a.t
        self.assertEqual(set(a.reasons['ball'][(t >= 5) & (t < 6)].tolist()), {cs.BALL_UNCERTAIN})
        self.assertEqual(set(a.reasons['ball'][(t >= 7) & (t < 8)].tolist()), {cs.BALL_UNCERTAIN})
        self.assertEqual(set(a.reasons['ball'][(t >= 9) & (t < 10)].tolist()), {cs.BALL_OFF_PITCH})
        self.assertTrue(a.eligible['ball'][(t >= 11) & (t < 12)].all())          # a bridged ball is a ball
        self.assertFalse(a.ballObserved[(t >= 11) & (t < 12)].any())             # ... but not an observation
        self.assertTrue(a.ballObserved[(t >= 13) & (t < 14)].all())


class Stabilized(unittest.TestCase):
    def test_flags_veto_evidence_and_change_no_measurement(self):
        times, teams, ends = syn.match('4-4-2', '4-4-2', seconds=60, fps=10)
        plain = syn.make_result(times, teams, ends)
        flagged = syn.make_result(times, teams, ends)
        flagged['stabilized'] = {'flags': [{'id': 'A-01', 'track': 1, 'frame': 300, 'time': 30.0, 'reason': 'box lost the legs'}]}
        a, b = cs.build_series(plain), cs.build_series(flagged)
        sa, sb = a.teams['A'], b.teams['A']
        near = np.abs(sb.t-30.0) <= cs.FLAG_SECONDS+1e-9
        np.testing.assert_array_equal(sb.flagged[:, 0], near)
        self.assertFalse(sb.flagged[:, 1:].any())
        self.assertEqual(sb.flagged_ids(int(np.flatnonzero(near)[0])), {'A-01'})
        for name in ('width', 'hull', 'gapDM', 'lineStd', 'centreX', 'nearestTeammate'):
            np.testing.assert_array_equal(sa.values[name], sb.values[name], err_msg=name)
        self.assertTrue(b.available['stabilized'] and not a.available['stabilized'])


class MetricaShaped(unittest.TestCase):
    def test_a_metrica_shaped_result_runs_unvetoed(self):
        import team_shape_bench as tsb
        times = np.arange(0, 60, 1/25)
        teams = {}
        for team, layout, side, seed in (('A', '4-4-2', 'left', 0), ('B', '4-3-3', 'right', 1)):
            out = syn.team_track(times, layout, side, seed)
            gk = syn.goalkeeper_track(times, side, seed)
            pos = np.concatenate([gk[:, None, :], out], axis=1)*np.array([L, W])
            teams[team] = (times, pos, np.full((len(times), 2), np.nan), [f'p{j}' for j in range(pos.shape[1])])
        result, ends, switch = tsb.build_result(teams, np.ones(len(times), int), 5.0)
        settings = {'direction': {'A': 'left' if ends['left'] == 'A' else 'right', 'switchAt': switch}}
        cd = cs.build_series(result, settings=settings)
        self.assertEqual(cd.available, {'calibration': False, 'view': False, 'ball': False, 'stabilized': False})
        self.assertFalse(cd.small_scene or cd.metres_estimated)
        absent = {cs.NOT_CALIBRATED, cs.LOW_CALIBRATION, cs.DISAGREEMENT, cs.SMALL_SCENE, cs.OUT_OF_VIEW, cs.UNPLACED_TEAMMATE}
        for team in 'AB':
            s = cd.teams[team]
            for metric in cs.TEAM_METRICS+cs.PLAYER_METRICS:
                self.assertFalse(absent & set(s.reasons[metric].ravel().tolist()), (team, metric))
            for metric in ('width', 'hull', 'centre', 'gapDM', 'lineStd'):
                self.assertGreater(s.eligible[metric].mean(), .9, (team, metric))
            np.testing.assert_array_equal(s.eligible['nearestTeammate'], s.kind == 'O')
            self.assertEqual(s.frame[1]-s.frame[0], 5)                   # the 25 Hz source frame index
        json.dumps(cd.summary(), allow_nan=False)


class Degenerate(unittest.TestCase):
    def test_an_empty_result(self):
        cd = cs.build_series({'frames': [], 'match': {}, 'roster': {}, 'tactical': {}})
        self.assertEqual(len(cd.t), 0)
        for team in 'AB':
            self.assertEqual(cd.teams[team].coverage('gapDM'), {'measuredSeconds': 0.0, 'eligibleShare': 0.0, 'longestStretchSeconds': 0.0})
            self.assertEqual(cd.teams[team].coverage('nearestTeammate'), {'measuredSeconds': 0.0, 'eligibleShare': 0.0, 'longestStretchSeconds': 0.0})
        json.dumps(cd.summary(), allow_nan=False)
        self.assertIsNone(cd.sample_at(1.0))

    def test_a_team_nobody_identified(self):
        times, teams, ends = syn.match('4-4-2', '4-4-2', seconds=30, fps=10)
        result = syn.make_result(times, {'A': teams['A']}, ends)
        cd = cs.build_series(result)
        b = cd.teams['B']
        self.assertEqual(b.ids, [])
        self.assertFalse(b.mask('nearestTeammate').any() or b.eligible['width'].any())
        json.dumps(cd.summary(), allow_nan=False)


SMOKE = [p for p in os.environ.get('PITCHIQ_COACHING_SMOKE', '').split(os.pathsep) if p]


@unittest.skipUnless(SMOKE, 'set PITCHIQ_COACHING_SMOKE to result.json paths to run on real runs')
class RealRuns(unittest.TestCase):
    def test_real_runs(self):
        for path in SMOKE:
            with open(path, encoding='utf-8') as fh:
                result = json.load(fh)
            cd = cs.build_series(result)
            duration = len(cd.t)*cd.step
            for team, s in cd.teams.items():
                for metric in cs.TEAM_METRICS+cs.PLAYER_METRICS:
                    c = s.coverage(metric)
                    self.assertLessEqual(c['measuredSeconds'], duration+1e-6, (path, team, metric))
                    self.assertLessEqual(c['longestStretchSeconds'], c['measuredSeconds']+1e-6)
                for metric in ('width', 'hull', 'gapDM', 'gapMF', 'lineStd', 'centreX', 'nearestTeammate'):
                    v = s.values[metric]
                    e = s.eligible.get(metric, s.eligible.get('centre'))
                    self.assertTrue(np.isfinite(v[e]).all(), (path, team, metric))
            json.dumps(cd.summary(), allow_nan=False)


if __name__ == '__main__':
    unittest.main()
