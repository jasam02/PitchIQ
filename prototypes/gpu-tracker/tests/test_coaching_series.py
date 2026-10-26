"""The coaching series (soccer/coaching/series.py): every coaching number is measured on the team-shape grid from
observed positions only, and a sample that cannot be trusted is ineligible with a reason instead of a value: the
registration gates, complete teams for hull and spread, fully seen units under the template's line index, no gap
between interleaved units (critic B2), the voted team centre, the certainty rule for the nearest teammate and its
exemption when the whole lineup is observed (critic S8), each defender's offset from the rest of the back line, the
players' velocities (stretches, windows, jumps, gated frames), the gated ball context with hysteresis, and the
stabilized flags as an evidence veto only. Fields a result does not have (a Metrica-shaped result) never veto.
Each of these rules was undone once in a scratch copy (coaching2/build/context_series/mutate.py, 14 mutations) and
every mutation made a test here fail.

Run:  python -m unittest test_coaching_series          (from this directory)
Real runs (optional): set PITCHIQ_COACHING_SMOKE to result.json paths separated by the path separator.
"""
import inspect
import json
import math
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
H = int(round(cs.VEL_HALF_WINDOW/cs.SAMPLE_STEP))      # grid steps either side of a central velocity


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
        # The back line is incomplete: no defender's offset from the others is measured, for anyone.
        self.assertFalse(s.eligible['lineOffset'][hid].any())
        self.assertTrue(set(s.reasons['lineOffset'][hid].ravel().tolist()) <= {cs.INCOMPLETE_UNITS, cs.NO_FORMATION})

    def test_a_player_off_camera_has_no_velocity_and_his_edges_are_one_sided(self):
        s = self.cd.teams['A']
        hid = (s.t >= 40) & (s.t < 50)
        self.assertEqual(set(s.reasons['velocity'][hid, 0].tolist()), {cs.NOT_OBSERVED})
        self.assertTrue(s.eligible['velocity'][~hid].all())
        last, first = int(np.flatnonzero(s.t < 40)[-1]), int(np.flatnonzero(s.t >= 50)[0])
        X = s.x[:, 0]*L                                       # A defends the left goal: its attack X is pitch x
        self.assertAlmostEqual(s.velX[last, 0], (X[last]-X[last-H])/(s.t[last]-s.t[last-H]))
        self.assertAlmostEqual(s.velX[first, 0], (X[first+H]-X[first])/(s.t[first+H]-s.t[first]))

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
        for metric in ('lineOffset', 'velocity'):
            for code, where in expect.items():
                self.assertEqual(set(s.reasons[metric][where].ravel().tolist()), {code}, (metric, code))
        self.assertTrue(cd.available['calibration'])
        np.testing.assert_array_equal(cd.calibration['calibrated'], ~expect[cs.NOT_CALIBRATED])
        # A velocity never rests on a gated frame's position: right after the uncalibrated stretch it looks forward.
        first = int(np.flatnonzero(t >= 25)[0])
        X = s.x[:, 0]*L
        self.assertAlmostEqual(s.velX[first, 0], (X[first+H]-X[first])/(t[first+H]-t[first]))


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


def stepped_up(times, pos):
    """Team A's 4-4-2 layout with four spells (attack-frame metres): a left midfielder (A-06) well ahead of the
    forwards over [40, 44) s, a right midfielder (A-07) just ahead of their mean depth (55.5 m) over [50, 53) s, the
    left back (A-01) ahead of the midfield over [60, 64) s, both full-backs (A-01, A-04) ahead of it over [75, 79) s."""
    pos = pos.copy()
    pos[(times >= 40) & (times < 44), 5, 0] = 62.0
    pos[(times >= 50) & (times < 53), 6, 0] = 57.0
    pos[(times >= 60) & (times < 64), 0, 0] = 52.0
    both = (times >= 75) & (times < 79)
    pos[both, 0, 0] = 52.0
    pos[both, 3, 0] = 52.0
    return pos


class UnitsAndLine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        times = np.arange(0, 90, .1)
        teams = {'A': (syn.team_track(times, '4-4-2', 'left', 0, overrides=stepped_up), syn.goalkeeper_track(times, 'left', 0)),
                 'B': (syn.team_track(times, '4-4-2', 'right', 1), syn.goalkeeper_track(times, 'right', 1))}
        cls.cd = cs.build_series(syn.make_result(times, teams, {'left': 'A', 'right': 'B'}))
        cls.s = cls.cd.teams['A']

    def front(self, unit):
        """The most advanced observed member of a unit per sample (attack-frame metres; A defends the left goal)."""
        s = self.s
        X = np.where(s.kind == 'O', s.x*L, np.nan)
        out = np.full(len(s.t), -np.inf)
        for k, u in enumerate(s.units):
            cols = [s.col[p] for p in (u or {}).get(unit, ()) if p in s.col]
            if cols and np.isfinite(X[k, cols]).any():
                out[k] = np.nanmax(X[k, cols])
        return out

    def check_gate(self, gap, deep, ahead, ahead_centre):
        s = self.s
        both = s.eligible['unit'+deep] & s.eligible['unit'+ahead]
        with np.errstate(invalid='ignore'):
            mixed = both & (self.front(deep) > s.values[ahead_centre])
        np.testing.assert_array_equal(s.reasons[gap] == cs.UNITS_INTERLEAVED, mixed)
        np.testing.assert_array_equal(s.eligible[gap], both & ~mixed)
        self.assertTrue(np.isnan(s.values[gap][mixed]).all())
        return mixed

    def test_a_midfielder_ahead_of_the_forwards_makes_the_midfield_to_front_gap_unmeasured(self):
        # Critic B2: a template unit interleaved with the next one has no gap between them (BC60 B: a "midfielder"
        # ahead of the deepest "forward" in 98% of 244 frames, a 4.1 m template gap for 8.8 m by depth rank).
        s = self.s
        mixed = self.check_gate('gapMF', 'M', 'F', 'FX')
        spell = (s.t >= 40.4) & (s.t < 43.6)
        self.assertGreater(mixed[spell].mean(), .9)
        self.assertFalse(mixed[(s.t < 39) | ((s.t > 45) & (s.t < 49)) | (s.t > 54)].any())
        # Just ahead counts too: the rule is the next unit's mean depth, not a margin beyond it.
        near = (s.t >= 50.4) & (s.t < 52.6)
        with np.errstate(invalid='ignore'):
            ahead = self.front('M')-s.FX
        self.assertGreater(mixed[near].mean(), .5)
        self.assertTrue((mixed & near & (ahead < 2.0)).any())
        k = int(np.flatnonzero(spell)[0])
        self.assertIn('A-06', s.units[k]['M'])                 # the template's unit itself is unchanged
        self.assertTrue(s.eligible['unitM'][k] and s.eligible['unitF'][k])     # and so are the unit centres
        ok = s.eligible['gapMF']
        np.testing.assert_allclose(s.gapMF[ok], (s.FX-s.MX)[ok])

    def test_a_full_back_ahead_of_the_midfield_makes_the_back_to_midfield_gap_unmeasured(self):
        s = self.s
        mixed = self.check_gate('gapDM', 'D', 'M1', 'M1X')
        one, two = (s.t >= 60.4) & (s.t < 63.6), (s.t >= 75.4) & (s.t < 78.6)
        self.assertGreater(mixed[one].mean(), .9)
        self.assertGreater(mixed[two].mean(), .9)
        self.assertFalse(mixed[(s.t < 59) | ((s.t > 65) & (s.t < 74)) | (s.t > 80)].any())
        self.assertTrue(s.eligible['gapDM'][(s.t > 5) & (s.t < 35)].all())
        self.assertIn(cs.UNITS_INTERLEAVED, cs.REASONS)
        self.assertLess(cs.REASONS.index(cs.INCOMPLETE_UNITS), cs.REASONS.index(cs.UNITS_INTERLEAVED))

    def test_each_defenders_offset_from_the_median_of_the_others(self):
        cd, s = self.cd, self.s
        data = cd.shape.observed
        rows = ts.defensive_line(data, 'A', cd.shape.directions, cd.shape.shapes['A'], data.start, data.end)
        e = s.eligible['lineOffset']
        self.assertGreater(e.any(axis=1).mean(), .9)
        self.assertTrue((e.sum(axis=1)[e.any(axis=1)] == 4).all())       # a back four: four defenders measured
        checked = 0
        for k in np.flatnonzero(e.any(axis=1))[::7]:
            dd = rows[k]['defenders']
            dist = np.array([v['distance'] for v in dd])
            across = [v['y'] for v in dd]
            wide = {dd[int(np.argmin(across))]['id'], dd[int(np.argmax(across))]['id']}
            for i, v in enumerate(dd):
                j = s.col[v['id']]
                mid = float(np.median(np.delete(dist, i)))
                self.assertAlmostEqual(s.lineRef[k, j], mid)
                self.assertAlmostEqual(s.lineOffsetSigned[k, j], dist[i]-mid)
                self.assertAlmostEqual(s.lineOffset[k, j], abs(dist[i]-mid))
                self.assertEqual(s.lineRole[k, j], 'wide' if v['id'] in wide else 'central')
                checked += 1
            members = {s.col[v['id']] for v in dd}
            for j in set(range(len(s.ids)))-members:
                self.assertEqual(s.reasons['lineOffset'][k, j], cs.NOT_IN_LINE)
        self.assertGreater(checked, 100)

    def test_a_defender_out_of_the_line_is_measured_even_when_the_line_is_disrupted(self):
        s = self.s
        j1, j4 = s.col['A-01'], s.col['A-04']
        one = (s.t >= 60.4) & (s.t < 63.6)                   # the left back alone ahead: an uneven line
        self.assertTrue(s.eligible['lineStd'][one].all())
        self.assertTrue((s.lineOffsetSigned[one, j1] > 15).all() and (s.lineRole[one, j1] == 'wide').all())
        self.assertTrue((s.lineOffset[one, j4] < 8).all())
        two = (s.t >= 75.4) & (s.t < 78.6)                   # both full-backs ahead: disrupted, no line std
        self.assertEqual(set(s.reasons['lineStd'][two].tolist()), {cs.LINE_DISRUPTED})
        self.assertTrue(s.eligible['lineOffset'][two][:, [j1, j4]].all())
        self.assertTrue((s.lineOffsetSigned[two][:, [j1, j4]] > 10).all())


class Velocity(unittest.TestCase):
    def test_central_inside_one_sided_at_the_ends(self):
        t = np.arange(0, 6, .2)
        S = len(t)
        X = np.stack([t**2, 50-3.0*t], axis=1)
        Y = np.stack([10+0*t, 1.5*t], axis=1)
        vx, vy = cs.velocities(t, X, Y, np.ones((S, 2), bool), np.full(S, 'left'))
        for s in range(S):
            a, b = max(0, s-H), min(S-1, s+H)
            self.assertAlmostEqual(vx[s, 0], (X[b, 0]-X[a, 0])/(t[b]-t[a]))
        self.assertAlmostEqual(vx[10, 0], 2*t[10])                       # a central difference: exact on a parabola
        np.testing.assert_allclose(vx[:, 1], -3.0)
        np.testing.assert_allclose(vy[:, 0], 0.0, atol=1e-12)
        np.testing.assert_allclose(vy[:, 1], 1.5)

    def test_frame_times_off_the_exact_step(self):
        t = np.arange(40)*.2002                               # 29.97 fps: six frames are 0.2002 s
        X = (3.0*t)[:, None]
        vx, _ = cs.velocities(t, X, X*0, np.ones((40, 1), bool), np.full(40, 'right'))
        np.testing.assert_allclose(vx[:, 0], 3.0)
        self.assertTrue(np.isfinite(vx).all())
        t24 = np.arange(40)*(5/24)                            # 24 fps: five frames
        vx24, _ = cs.velocities(t24, (2.0*t24)[:, None], np.zeros((40, 1)), np.ones((40, 1), bool), np.full(40, 'left'))
        np.testing.assert_allclose(vx24[:, 0], 2.0)

    def test_short_tracks_gaps_and_side_changes(self):
        t = np.arange(0, 12, .2)
        S = len(t)
        X = np.stack([.3*t**2]*4, axis=1)                     # accelerating: every window gives its own value
        side = np.full(S, 'left')
        side[40:] = 'right'                                   # the halves: the attack frame flips at sample 40
        X[40:, 3] = L-X[40:, 3]
        usable = np.ones((S, 4), bool)
        usable[:, 0] = False
        usable[30, 0] = True                                  # one sighting: no baseline
        usable[:, 1] = False
        usable[30:32, 1] = True                               # two sightings 0.2 s apart: shorter than VEL_MIN_BASE
        usable[[20, 21, 22], 2] = False                       # three missing: 0.8 s between sightings
        vx, _ = cs.velocities(t, X, X*0, usable, side, half=1.0, jump_speed=np.inf)
        self.assertTrue(np.isnan(vx[:, 0]).all() and np.isnan(vx[:, 1]).all())
        # A gap longer than VEL_MAX_GAP is never spanned, however wide the window.
        self.assertAlmostEqual(vx[19, 2], (X[19, 2]-X[14, 2])/(t[19]-t[14]))
        self.assertAlmostEqual(vx[23, 2], (X[28, 2]-X[23, 2])/(t[28]-t[23]))
        self.assertAlmostEqual(vx[17, 2], (X[19, 2]-X[12, 2])/(t[19]-t[12]))
        # ... and neither is a side change (here the jump rule is off: the flip alone splits the stretch).
        self.assertAlmostEqual(vx[39, 3], (X[39, 3]-X[34, 3])/(t[39]-t[34]))
        self.assertAlmostEqual(vx[40, 3], (X[45, 3]-X[40, 3])/(t[45]-t[40]))
        self.assertLess(vx[40, 3], 0)                         # after the flip the player runs towards his own goal
        # A jump no player could run (the amateur clip's lone B-07 sighting 11.3 m from the next one 0.6 s later read
        # 16.6 m/s) splits the stretch: the odd position stands alone, both sides keep their own velocity.
        Xj = (2.0*t)[:, None].copy()
        uj = np.ones((S, 1), bool)
        uj[[11, 12], 0] = False
        Xj[10, 0] += 11.0
        vj, _ = cs.velocities(t, Xj, Xj*0, uj, np.full(S, 'left'))
        self.assertTrue(np.isnan(vj[10, 0]))
        np.testing.assert_allclose(vj[np.isfinite(vj[:, 0]), 0], 2.0)
        self.assertEqual(int(np.isnan(vj[:, 0]).sum()), 3)                 # the lone sighting and the two unusable
        self.assertAlmostEqual(cs.velocities(t, Xj*0+(2.0*t)[:, None], Xj*0, uj, np.full(S, 'left'))[0][10, 0], 2.0)
        # At the default window one missing sample is bridged: a central difference across it (exact on a parabola).
        u = np.ones((S, 1), bool)
        u[25, 0] = False
        v, _ = cs.velocities(t, X[:, :1], X[:, :1]*0, u, np.full(S, 'left'))
        self.assertTrue(np.isnan(v[25, 0]))
        self.assertAlmostEqual(v[24, 0], .6*t[24])
        self.assertAlmostEqual(v[26, 0], .6*t[26])

    def test_the_series_velocity_is_the_attack_frame_derivative(self):
        times, teams, ends = syn.match('4-4-2', '4-2-3-1', seconds=40, fps=10)
        visible = {'A': np.ones((len(times), 10), bool)}
        alone = (times >= 18) & (times < 22) & (np.abs(times-20.0) > .05)
        visible['A'][alone, 9] = False                        # A-10 seen at 20.0 s only within [18, 22)
        cd = cs.build_series(syn.make_result(times, teams, ends, visible=visible))
        fi = cd.shape.observed.frame_index
        for team, side in (('A', 'left'), ('B', 'right')):
            s = cd.teams[team]
            out = teams[team][0][fi]
            X = (out[:, :, 0] if side == 'left' else 1-out[:, :, 0])*L
            Y = (out[:, :, 1] if side == 'left' else 1-out[:, :, 1])*W
            S = len(s.t)
            checked = 0
            for k in range(H, S-H, 9):
                for j in range(len(s.ids)):
                    if not s.eligible['velocity'][k-H:k+H+1, j].all():
                        continue
                    self.assertAlmostEqual(s.velX[k, j], (X[k+H, j]-X[k-H, j])/(s.t[k+H]-s.t[k-H]), places=6)
                    self.assertAlmostEqual(s.velY[k, j], (Y[k+H, j]-Y[k-H, j])/(s.t[k+H]-s.t[k-H]), places=6)
                    checked += 1
                    self.assertAlmostEqual(s.speed[k, j], math.hypot(s.velX[k, j], s.velY[k, j]), places=9)
            self.assertGreater(checked, 100)
        a = cd.teams['A']
        k = int(np.argmin(np.abs(a.t-20.0)))
        self.assertEqual(a.reasons['velocity'][k, 9], cs.SHORT_TRACK)
        self.assertEqual(set(a.reasons['velocity'][(a.t >= 18) & (a.t < 22) & (np.abs(a.t-20) > .05), 9].tolist()), {cs.NOT_OBSERVED})
        self.assertTrue(a.eligible['velocity'][:, :9].all())
        # The velocity is a measurement of its own: no other value moves with it.
        self.assertEqual(cs.VALUE_METRIC['velX'], 'velocity')


class NearestTeammate(unittest.TestCase):
    HALF =[[-5/L, -5/W], [.5, -5/W], [.5, 1+5/W], [-5/L, 1+5/W]]       # the left half of the pitch (+5 m margins)

    def half_view_series(self, hide_from=40.0, hide_to=50.0):
        """Team A (defends the left goal: pitch x is its attack X) under a view of the left half, with its left back
        off camera over [hide_from, hide_to): the team is incomplete only there."""
        times, teams, ends = syn.match('4-4-2', '4-4-2', seconds=90, fps=10)
        visible = {'A': np.ones((len(times), 10), bool)}
        visible['A'][(times >= hide_from) & (times < hide_to), 0] = False
        result = syn.make_result(times, teams, ends, visible=visible)
        for f in result['frames']:
            f['view'] = self.HALF
        return cs.build_series(result).teams['A']

    def test_certain_only_when_the_disc_is_in_view(self):
        s = self.half_view_series()
        obs = s.kind == 'O'
        part = ~s.complete
        self.assertTrue(part.any() and s.complete.any())
        x = np.where(obs, s.x*L, np.nan)
        disc = obs & (x+s.nearestTeammate <= L/2+1e-9)        # the right edge is the only one inside the pitch
        # Incomplete samples (someone of the lineup unseen): the disc rule decides.
        np.testing.assert_array_equal(s.eligible['nearestTeammate'][part], disc[part])
        self.assertTrue(disc[part].any() and (obs[part] & ~disc[part]).any())
        self.assertEqual(set(s.reasons['nearestTeammate'][part][obs[part] & ~disc[part]].tolist()), {cs.OUT_OF_VIEW})
        poly = np.array(self.HALF)*[L, W]
        inside = cs._disc_inside(poly, np.array([10.0, 50.0, 60.0]), np.array([1.0, 30, 30]), np.array([8.0, 3.0, 1.0]), L, W)
        self.assertEqual(inside.tolist(), [True, False, False])   # a disc over the touchline is fine; beyond the view is not

    def test_the_whole_lineup_observed_lifts_the_view_veto(self):
        # Critic S8: with every outfield player of the lineup observed no unseen teammate can be nearer, so the view
        # does not matter (players outside the view polygon included); the other gates still apply.
        s = self.half_view_series()
        obs = s.kind == 'O'
        full = s.complete[:, None] & obs
        self.assertTrue(full.any())
        self.assertTrue(s.eligible['nearestTeammate'][full].all())
        x = np.where(obs, s.x*L, np.nan)
        self.assertTrue((full & (x+s.nearestTeammate > L/2)).any())      # discs beyond the view, measured all the same
        self.assertNotIn(cs.OUT_OF_VIEW, set(s.reasons['nearestTeammate'][s.complete].ravel().tolist()))
        # The value is the plain nearest distance there.
        k = int(np.flatnonzero(s.complete)[0])
        P = np.array([(s.x[k, j]*L, s.y[k, j]*W) for j in range(len(s.ids))])
        D = np.hypot(*(P[:, None, :]-P[None, :, :]).transpose(2, 0, 1))
        np.fill_diagonal(D, np.inf)
        np.testing.assert_allclose(s.nearestTeammate[k], D.min(axis=1))

    def test_a_view_without_a_polygon_still_vetoes_an_incomplete_team_only(self):
        times, teams, ends = syn.match('4-4-2', '4-4-2', seconds=60, fps=10)
        visible = {'A': np.ones((len(times), 10), bool)}
        visible['A'][(times >= 20) & (times < 30), 3] = False
        result = syn.make_result(times, teams, ends, visible=visible)
        for f in result['frames']:
            f['view'] = None                                   # the field exists but holds no usable polygon
        s = cs.build_series(result).teams['A']
        obs = s.kind == 'O'
        self.assertTrue(s.eligible['nearestTeammate'][s.complete[:, None] & obs].all())
        part = (~s.complete)[:, None] & obs
        self.assertTrue(part.any())
        self.assertEqual(set(s.reasons['nearestTeammate'][part].tolist()), {cs.OUT_OF_VIEW})

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


class FrameContextKept(unittest.TestCase):
    """The frame context the situation timeline reads (critic B4 data-quality counts, the view, the unplaced count)
    is kept on CoachingData and changes no measurement."""

    def test_in_kit_and_confirmed_counts(self):
        times, teams, ends = syn.match('4-4-2', '4-4-2', seconds=30, fps=10)
        base = cs.build_series(syn.make_result(times, teams, ends))
        np.testing.assert_array_equal(base.in_kit['A'], 10)
        np.testing.assert_array_equal(base.confirmed_in_kit['A'], 10)
        extra = {}
        for k in range(len(times)):
            extra[k] = [{'id': None, 'team': 'A', 'role': 'PLAYER', 'state': 'uncertain', 'pitch': [.4, .4],
                         'pitchConfidence': .8},                                      # in kit, not confirmed
                        {'id': 'A-12', 'team': 'A', 'role': 'PLAYER', 'state': 'confirmed'},     # no position: counts
                        {'id': 'A-13', 'team': 'A', 'role': 'PLAYER', 'state': 'confirmed', 'pitch': [1.3, .5],
                         'pitchConfidence': .8},                                      # beyond the lines: not on it
                        {'id': None, 'team': 'A', 'role': 'NON_TARGET', 'state': 'nontarget', 'pitch': [.5, .5],
                         'pitchConfidence': .8},                                      # not a player
                        {'id': 'A-14', 'team': 'A', 'role': 'PLAYER', 'state': 'rejected', 'pitch': [.5, .5],
                         'pitchConfidence': .8}]                                      # rejected
        cd = cs.build_series(syn.make_result(times, teams, ends, extra_people=extra))
        np.testing.assert_array_equal(cd.in_kit['A'], 12)
        np.testing.assert_array_equal(cd.confirmed_in_kit['A'], 11)
        np.testing.assert_array_equal(cd.unplaced['A'], 1)
        np.testing.assert_array_equal(cd.in_kit['B'], 10)
        for name, v in base.teams['A'].values.items():      # nothing else moved but what the extra people move
            if name not in ('nearestTeammate', 'nearestExtras'):
                np.testing.assert_array_equal(np.asarray(v), np.asarray(cd.teams['A'].values[name]), err_msg=name)

    def test_the_view_is_kept_per_sample(self):
        def edit(k, t, f):
            f['view'] = [[0, 0], [.5, 0], [.5, 1], [0, 1]]
        _, _, result = build('4-4-2', '4-4-2', seconds=20, edit=edit)
        cd = cs.build_series(result)
        self.assertEqual(len(cd.view), len(cd.t))
        np.testing.assert_allclose(cd.view[0], [[0, 0], [L/2, 0], [L/2, W], [0, W]])
        self.assertEqual(cs.build_series(build('4-4-2', '4-4-2', seconds=20)[2]).view[0], None)


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
                for metric in ('width', 'hull', 'gapDM', 'gapMF', 'lineStd', 'centreX', 'nearestTeammate', 'lineOffset',
                               'velX', 'speed'):
                    v = s.values[metric]
                    e = s.eligible[cs.VALUE_METRIC.get(metric, 'centre')]
                    self.assertTrue(np.isfinite(v[e]).all(), (path, team, metric))
                # Speeds a player can run (the jump rule removes identity jumps), interleaved gaps never measured.
                self.assertTrue((s.speed[s.eligible['velocity']] < 12).all(), (path, team))
                self.assertFalse((s.eligible['gapMF'] & (s.reasons['gapMF'] == cs.UNITS_INTERLEAVED)).any())
            json.dumps(cd.summary(), allow_nan=False)


if __name__ == '__main__':
    unittest.main()
