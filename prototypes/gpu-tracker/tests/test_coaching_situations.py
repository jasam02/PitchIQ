"""The situation timeline (soccer/coaching/situations.py, DESIGN v2 3.1-3.3, 3.5, 3.6): the phase follows possession-4
under the binding possession rules or the coach's marks, never a guess; the ball, the carrier, the players and the
gates say where they are unknown; completeness follows the coaching format and the scale the declared field; the
summary follows the schema and depends only on the measurement inputs.

Why these checks. The v1 engine judged every sample whatever the phase; v2's detectors trust this timeline for who had
the ball, so each binding rule is a test: the +-2 s exclusion around EVERY change (a removed rule shows up as a named
sample inside a change window), the engine's dead spells, no carrier that contradicts the phase, coach spans that
override the engine without an exclusion at their own boundaries and with the engine's windows still applied outside
them, no own/opp time named when possession-4 abstains and no coach span exists. The scenes are EVAL's synthetic
results (tools/coaching_synth.py, the real pipeline: analytics, team_shape, possession-4, the coaching series) whose
possession script, formations and roles are the truth; on them the named phase matched the script on 100% of the
named samples (covered_advance, professional, formation_change, sprint_back, small_field_scale, the coach-marked
amateur scene, the 4v4 with coach spans).
Each of 37 rules of situations.py and roles.py was undone once in a scratch copy (coaching2/build/context_w2/mutate.py)
and every mutation made a test of this file or its sibling fail.

Run:  python -m unittest test_coaching_situations          (from this directory)
"""
import copy
import json
import sys
import unittest
from pathlib import Path

import numpy as np

import scene  # noqa: F401  (adds the prototype directory to sys.path)

TOOLS = Path(__file__).resolve().parents[1]/'tools'
sys.path.insert(0, str(TOOLS))
import coaching_synth as cs  # noqa: E402
from soccer import analytics, possession, team_shape as ts  # noqa: E402
from soccer.coaching import schema as S, series, situations as SI  # noqa: E402

_PREPARED = {}


def prepared(scene_name, variant=None):
    """(built, settings, coaching series, possession-4 record) of a scene, cached for the module."""
    key = (scene_name, variant)
    if key not in _PREPARED:
        b = cs.build(scene_name, variant)
        st = cs.settings_for(b, level=4)
        samples = analytics.Samples(b.result)
        shape = ts.TeamShape(b.result, samples=samples)
        cd = series.build_series(b.result, samples=samples, shape=shape)
        _PREPARED[key] = (b, st, cd, possession.compute(b.result))
    return _PREPARED[key]


def timeline(scene_name, variant=None, team='A', coach=None, poss=None, measurement=None):
    b, st, cd, rec = prepared(scene_name, variant)
    meas = measurement or S.tier_settings(st['teams']['A'])['measurement']
    return SI.build_timeline(cd, rec if poss is None else poss, team, meas, coach_spans=coach)


def script_phase(truth, t, team):
    out = np.full(len(t), 'unknown', dtype='<U7')
    for sp in truth['possession']:
        m = (t >= sp['t0']) & (t <= sp['t1'])
        out[m] = 'dead' if sp['state'] == 'dead' else ('own' if sp['state'] == team else 'opp')
    return out


class Repetition(unittest.TestCase):
    def test_spells_merge_short_interruptions_and_keep_long_ones(self):
        t = 100.0+.2*np.arange(100)
        m = np.zeros(100, bool)
        m[10:20] = True                 # 2.0 s
        m[24:30] = True                 # 0.6 s after 0.6 s missing: merged with the first
        m[50:55] = True                 # 1.0 s alone: too short
        m[70:90] = True                 # 4.0 s
        found = SI.spells(t, m, min_s=2.0, step=.2)
        self.assertEqual([(s['t0'], s['t1']) for s in found], [(t[10], t[29]), (t[70], t[89])])
        self.assertEqual(found[0]['samples'], 16)
        self.assertEqual(SI.spells(t, m, eligible=np.zeros(100, bool)), [])
        self.assertEqual(len(SI.spells(t, m, min_s=2.0, merge_s=0.0, step=.2)), 2)    # 2.0 s and 4.0 s only

    def test_spells_on_a_2997_fps_grid_are_not_split_by_frame_times(self):
        t = 10.0+.2002*np.arange(50)
        self.assertEqual(len(SI.spells(t, np.ones(50, bool), merge_s=0.0, step=.2)), 1)

    def test_distinct_spells_start_ten_seconds_apart(self):
        sp = [{'t0': x, 't1': x+2} for x in (0.0, 4.0, 10.0, 12.0, 25.0)]
        self.assertEqual([s['t0'] for s in SI.distinct(sp)], [0.0, 10.0, 25.0])

    def test_rates_and_short_ranges(self):
        self.assertAlmostEqual(SI.rate_per_10min(3, 300.0), 6.0)
        self.assertIsNone(SI.rate_per_10min(3, 0.0))
        self.assertTrue(SI.short_range(179.9))
        self.assertFalse(SI.short_range(180.0))


class CoachSpanCleaning(unittest.TestCase):
    def test_kept_clipped_and_dropped(self):
        raw = {'version': 1, 'kind': 'coach-possession-spans', 'source': 'coach', 'spans': [
            {'t0': 99.8, 't1': 110.0, 'state': 'A'},            # inside with the half-second tolerance: clipped
            {'t0': 105.0, 't1': 115.0, 'state': 'B'},           # overlaps the first: dropped
            {'t0': 120.0, 't1': 118.0, 'state': 'B'},           # ends before it starts
            {'t0': 130.0, 't1': 140.0, 'state': 'C'},           # unknown state
            {'t0': 150.0, 't1': 210.0, 'state': 'dead'},        # beyond the segment: dropped, never clipped
            {'t0': 160.0, 't1': 170.0, 'state': 'B'}]}
        kept, dropped = SI.clean_coach_spans(raw, 100.0, 200.0)
        self.assertEqual(kept, [{'t0': 100.0, 't1': 110.0, 'state': 'A'}, {'t0': 160.0, 't1': 170.0, 'state': 'B'}])
        self.assertEqual(dropped, 4)
        self.assertEqual(SI.clean_coach_spans(raw['spans'][:1], 100.0, 200.0)[0], kept[:1])     # a plain list
        self.assertEqual(SI.clean_coach_spans(None, 0, 1), ([], 0))
        self.assertEqual(SI.clean_coach_spans('nonsense', 0, 1), ([], 1))

    def test_the_schema_record(self):
        good = S.example('coach_spans')
        self.assertEqual(S.validate('coach_spans', good), [])
        bad = copy.deepcopy(good)
        bad['spans'][1]['t0'] = 740.0                       # overlaps the first
        self.assertTrue(S.validate('coach_spans', bad))
        bad = copy.deepcopy(good)
        bad['spans'][0]['t1'] = bad['spans'][0]['t0']       # empty
        self.assertTrue(S.validate('coach_spans', bad))
        bad = copy.deepcopy(good)
        bad['spans'][0]['state'] = 'own'                    # absolute team letters only
        self.assertTrue(S.validate('coach_spans', bad))


class Broadcast(unittest.TestCase):
    """covered_advance: an 11v11 broadcast-like scene, possession-4 available."""

    @classmethod
    def setUpClass(cls):
        cls.b, cls.st, cls.cd, cls.poss = prepared('covered_advance')
        cls.tl = timeline('covered_advance')
        cls.tlB = timeline('covered_advance', team='B')

    def test_the_summary_follows_the_contract_and_only_its_inputs(self):
        for tl in (self.tl, self.tlB):
            self.assertEqual(S.validate('timeline', tl.summary()), [])
        again = timeline('covered_advance')
        self.assertEqual(json.dumps(again.summary(), sort_keys=True), json.dumps(self.tl.summary(), sort_keys=True))
        self.assertEqual(json.dumps(again.role_inputs(), sort_keys=True, allow_nan=False),
                         json.dumps(self.tl.role_inputs(), sort_keys=True, allow_nan=False))

    def test_the_named_phase_is_the_script(self):
        truth = script_phase(self.b.truth, self.tl.t, 'A')
        named = np.isin(self.tl.phase, ('own', 'opp'))
        self.assertGreater(named.mean(), .7)
        self.assertTrue((self.tl.phase[named] == truth[named]).all())
        self.assertTrue((self.tl.phaseSource[named] == 'possession-4').all())
        both = np.isin(self.tlB.phase, ('own', 'opp'))
        self.assertTrue((both == named).all())
        self.assertTrue(((self.tl.phase == 'own') == (self.tlB.phase == 'opp')).all())

    def test_every_change_window_is_excluded(self):
        t = self.tl.t
        changes = self.poss['changes']
        self.assertGreater(len(changes), 10)
        for c in changes:
            lo = min(c['lostAt'] if c.get('lostAt') is not None else c['wonAt'], c['wonAt']-2.0)
            m = (t >= lo) & (t <= c['wonAt']+2.0)
            self.assertTrue(m.any())
            self.assertTrue(self.tl.nearChange[m].all())
            self.assertFalse(np.isin(self.tl.phase[m], ('own', 'opp')).any(), c)
            self.assertTrue(set(self.tl.changeConfidence[m].tolist()) <= {'high', 'uncertain'})
        far = ~self.tl.nearChange
        self.assertTrue((self.tl.changeConfidence[far] == '').all())

    def test_the_engines_dead_spells_are_dead(self):
        t = self.tl.t
        for d in self.poss['deadSpells']:
            m = (t >= d['t0']) & (t <= d['t1'])
            self.assertTrue((self.tl.phase[m] == 'dead').all())
        self.assertGreater(self.tl.summary()['phaseSeconds']['dead'], 0)

    def test_phase_spans_and_their_lengths(self):
        tl = self.tl
        self.assertTrue((tl.phaseSpan[tl.phase == 'unknown'] == 0).all())
        for s in tl.summary()['phaseSpans']:
            i0 = int(np.flatnonzero(np.isclose(tl.t, s['t0']))[0])
            i1 = int(np.flatnonzero(np.isclose(tl.t, s['t1']))[0])
            self.assertTrue((tl.phase[i0:i1+1] == s['state']).all())
            self.assertAlmostEqual(tl.phaseSpan[i0], tl.t[i1]-tl.t[i0]+tl.step, places=3)
            self.assertEqual(s['source'], 'possession-4')
        long_ = tl.phaseSpan >= SI.MIN_SHAPE_SPAN_S
        self.assertTrue(long_[tl.phase == 'own'].mean() > .9)

    def test_the_carrier_never_contradicts_the_phase(self):
        tl = self.tl
        has = tl.carrier != ''
        self.assertTrue(has.any())
        self.assertTrue(((tl.phase[has] == 'own') & (tl.carrierTeam[has] == 'A') |
                         (tl.phase[has] == 'opp') & (tl.carrierTeam[has] == 'B')).all())
        # only non-estimated control runs name a carrier
        runs = sorted(self.poss['timeline'], key=lambda r: r['f0'])
        f0 = np.array([r['f0'] for r in runs])
        for s in np.flatnonzero(has)[::7]:
            r = runs[int(np.searchsorted(f0, tl.frame[s], side='right'))-1]
            self.assertEqual((r['state'], r['estimated'], r['carrier']), ('control', False, tl.carrier[s]))
        self.assertTrue(np.isfinite(tl.carrierX[has]).mean() > .95)
        sm = tl.summary()
        self.assertGreater(sm['carrierShare'], .5)
        est = tl.estimatedCarrier != ''           # an estimated run's carrier is kept apart, never in `carrier`
        self.assertFalse((est & has).any())
        for s in np.flatnonzero(est)[::5]:
            r = runs[int(np.searchsorted(f0, tl.frame[s], side='right'))-1]
            self.assertEqual((r['state'], r['estimated'], r['carrier']), ('control', True, tl.estimatedCarrier[s]))
            self.assertIn(tl.phase[s], ('own', 'opp'))

    def test_rough_shares_coverage_and_notes(self):
        q = self.tl.summary()['phaseQuality']
        self.assertTrue(q['available'])
        self.assertAlmostEqual(q['roughShares']['A']*20, round(q['roughShares']['A']*20))
        self.assertAlmostEqual(q['roughShares']['A']+q['roughShares']['B'], 1.0)
        self.assertIsNone(q['coach'])
        self.assertTrue(any('left out' in n for n in q['notes']))
        self.assertTrue(any('Stoppages' in n for n in q['notes']))
        for n in q['notes']:
            self.assertEqual(S.event_problems(n), [])

    def test_positions_in_this_teams_attack_frame(self):
        tl, tb = self.tl, self.tlB
        L, W = tl.scale['lengthM'], tl.scale['widthM']
        j = tl.col['A-01']
        ok = np.isfinite(tl.x[:, j])
        np.testing.assert_allclose(tl.X[ok, j], tl.x[ok, j]*L)                # A defends the left goal
        np.testing.assert_allclose(tb.oppX[ok, tb.oppCol['A-01']], (1-tl.x[ok, j])*L)
        np.testing.assert_allclose(tb.oppY[ok, tb.oppCol['A-01']], (1-tl.y[ok, j])*W)
        # the opponents' velocity turned into this team's frame
        k = tl.oppCol['B-02']
        v = tl.oppVelOk[:, k] & tb.velOk[:, tb.col['B-02']]
        np.testing.assert_allclose(tl.oppVelX[v, k], -tb.velX[v, tb.col['B-02']])
        self.assertTrue(tl.complete.mean() > .95 and not tl.excess.any())
        self.assertTrue((tl.ballThird[tl.ballReliable & np.isfinite(tl.ballAX)] != '').all())
        self.assertTrue(set(tl.ballThird.tolist()) <= set(S.ZONES) | {''})

    def test_the_players_and_the_data_quality(self):
        sm = self.tl.summary()
        self.assertEqual(sm['dataQuality']['lineup'], 10)
        self.assertIsNone(sm['dataQuality']['dominantCause'])
        self.assertEqual(sum(1 for p in sm['players'].values() if p['team'] == 'A' and not p['goalkeeper']), 10)
        self.assertTrue(all(p['p90SpeedMS'] is None or p['p90SpeedMS'] < 9.5 for p in sm['players'].values()))
        self.assertTrue(any(p['goalkeeper'] for p in sm['players'].values()))

    def test_role_inputs_are_compact_json(self):
        ri = self.tl.role_inputs()
        text = json.dumps(ri, allow_nan=False)
        self.assertLess(len(text), 400*len(self.tl.t))       # well under a kilobyte per second for 11v11
        self.assertEqual(ri['version'], SI.CONTEXT_VERSION)
        self.assertEqual(len(ri['track']['t']), len(ri['track']['phase']))
        self.assertEqual(len(ri['track']['x']), len(ri['ids']))
        self.assertTrue(ri['windows'] and all(w['label'] == '4-4-2' for w in ri['windows']))
        self.assertTrue(all(0 <= w['own'] <= 1 and 0 <= w['opp'] <= 1 for w in ri['windows']))


class EngineRules(unittest.TestCase):
    """The binding rules on an edited possession record: a change window and a dead spell injected, an unavailable
    record, no record."""

    def test_an_injected_change_and_dead_spell(self):
        b, st, cd, poss = prepared('covered_advance')
        edited = copy.deepcopy(poss)
        named = timeline('covered_advance')
        s = int(np.flatnonzero((named.phase == 'own') & (named.phaseSpan > 20))[0])+30
        won = float(named.t[s])
        edited['changes'].append({'id': 'x', 'lostAt': won-3.0, 'wonAt': won, 'frame': 0, 'fromTeam': 'B',
                                  'toTeam': 'A', 'lostBy': None, 'wonBy': None, 'cause': 'unclear',
                                  'confidence': 'high'})
        edited['deadSpells'].append({'t0': won+10.0, 't1': won+12.0, 'reason': 'stoppage', 'restartTeam': 'A'})
        tl = timeline('covered_advance', poss=edited)
        t = tl.t
        win = (t >= won-3.0) & (t <= won+2.0)
        self.assertTrue(tl.nearChange[win].all() and (tl.phase[win] == 'unknown').all())
        self.assertTrue((tl.changeConfidence[win] == 'high').all())
        dead = (t >= won+10.0) & (t <= won+12.0)
        self.assertTrue((tl.phase[dead] == 'dead').all() and (tl.carrier[dead] == '').all())
        self.assertEqual(S.validate('timeline', tl.summary()), [])

    def test_the_seconds_to_a_ball_between_the_teams(self):
        """Fix round 3 (LEAD decision (2)): contestS is the seconds from each sample to the nearest possession-4 run
        whose ball is 'loose' (between the teams) or 'contested', 0 inside one, NaN without any; a coach's marks
        never change it (a measurement of the ball, not of the phase)."""
        t = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        runs = [{'t0': 0.0, 't1': 1.4, 'state': 'control'}, {'t0': 1.5, 't1': 2.4, 'state': 'loose'},
                {'t0': 2.5, 't1': 4.4, 'state': 'flight'}, {'t0': 4.5, 't1': 4.6, 'state': 'contested'}]
        got = SI._contest_seconds({'available': True, 'timeline': runs}, t)
        np.testing.assert_allclose(got, [1.5, .5, 0.0, .6, .5, .4, 1.4])
        self.assertTrue(np.isnan(SI._contest_seconds({'available': True, 'timeline': runs[:1]}, t)).all())
        self.assertTrue(np.isnan(SI._contest_seconds({'available': False, 'timeline': runs}, t)).all())
        b, st, cd, poss = prepared('covered_advance')
        tl = timeline('covered_advance')
        loose = [r for r in poss['timeline'] if r['state'] in SI.CONTEST_STATES]
        inside = np.zeros(len(tl.t), bool)
        for r in loose:
            inside |= (tl.t >= r['t0']-1e-6) & (tl.t <= r['t1']+1e-6)
        if loose:
            self.assertTrue((tl.contestS[inside] == 0.0).all() and (tl.contestS[~inside] > 0.0).all())
        else:
            self.assertTrue(np.isnan(tl.contestS).all())
        spans = timeline('covered_advance', coach=[{'t0': float(tl.t[10]), 't1': float(tl.t[-10]), 'state': 'A'}])
        np.testing.assert_array_equal(spans.contestS, tl.contestS)          # the coach decides the phase only

    def test_an_estimated_control_run_names_no_carrier(self):
        b, st, cd, poss = prepared('covered_advance')
        edited = copy.deepcopy(poss)
        controls = [r for r in edited['timeline'] if r['state'] == 'control' and r['carrier']]
        for r in controls[::2]:
            r['estimated'] = True               # the ball was interpolated across this run
        tl = timeline('covered_advance', poss=edited)
        f = tl.frame
        for r in controls[:40]:
            m = (f >= r['f0']) & (f <= r['f1']) & np.isin(tl.phase, ('own', 'opp'))
            if not m.any():
                continue
            if r['estimated']:
                self.assertTrue((tl.carrier[m] == '').all())
                self.assertTrue((tl.estimatedCarrier[m] == r['carrier']).all())
            else:
                self.assertTrue((tl.carrier[m] == r['carrier']).all())
                self.assertTrue((tl.estimatedCarrier[m] == '').all())
        plain = timeline('covered_advance')
        self.assertLess(tl.summary()['carrierShare'], plain.summary()['carrierShare'])

    def test_an_unavailable_record_names_nothing(self):
        b, st, cd, poss = prepared('covered_advance')
        off = dict(copy.deepcopy(poss), available=False, reason='Not available on this footage.')
        for p in (off, {}):
            tl = timeline('covered_advance', poss=p)
            self.assertTrue((tl.phase == 'unknown').all() and (tl.phaseSource == 'none').all())
            self.assertTrue((tl.carrier == '').all())
            sm = tl.summary()
            self.assertEqual(S.validate('timeline', sm), [])
            self.assertFalse(sm['phaseQuality']['available'])
            self.assertIsNone(sm['phaseQuality']['roughShares'])
            self.assertIsNotNone(sm['phaseQuality']['reason'])
        tl = SI.build_timeline(cd, None, 'A', {'format': {'players': 11, 'goalkeeper': True}, 'field': None})
        self.assertEqual(tl.summary()['phaseQuality']['reason'], 'No possession record for this run.')

    def test_the_schema_refuses_named_time_without_the_engine_or_a_coach(self):
        sm = copy.deepcopy(timeline('covered_advance').summary())
        sm['phaseQuality']['available'] = False
        sm['phaseQuality']['roughShares'] = None
        self.assertTrue(any('not by the coach' in p for p in S.validate('timeline', sm)))


class CoachSpans(unittest.TestCase):
    def test_coach_spans_unlock_the_phase_where_possession_4_abstains(self):
        b, st, cd, poss = prepared('amateur_coach_spans', 'spans')
        plain = timeline('amateur_coach_spans', 'spans')
        self.assertFalse(poss['available'])
        self.assertTrue((plain.phase == 'unknown').all())
        tl = timeline('amateur_coach_spans', 'spans', coach=b.coach_spans)
        sm = tl.summary()
        self.assertEqual(S.validate('timeline', sm), [])
        named = np.isin(tl.phase, ('own', 'opp'))
        self.assertGreater(named.mean(), .9)
        truth = script_phase(b.truth, tl.t, 'A')
        self.assertTrue((tl.phase[named] == truth[named]).all())
        self.assertTrue((tl.phaseSource[named] == 'coach').all() and not tl.nearChange.any())
        coach = sm['phaseQuality']['coach']
        self.assertEqual(coach['spans'], len(b.coach_spans['spans']))
        self.assertEqual((coach['dropped'], coach['engineNamedSeconds'], coach['agreement']), (0, 0.0, None))
        self.assertAlmostEqual(coach['own']+coach['opp'], sm['phaseSeconds']['own']+sm['phaseSeconds']['opp'], 1)
        self.assertTrue(all(s['source'] == 'coach' for s in sm['phaseSpans']))
        self.assertFalse(sm['phaseQuality']['available'])
        self.assertTrue((tl.carrier == '').all())         # no engine, no carrier: D1 stays unavailable
        same = timeline('amateur_coach_spans', 'spans', measurement=dict(
            S.tier_settings(st['teams']['A'])['measurement'], coachSpans=b.coach_spans))
        self.assertEqual(json.dumps(same.summary(), sort_keys=True), json.dumps(sm, sort_keys=True))

    def test_coach_spans_override_the_engine_only_where_they_lie(self):
        b, st, cd, poss = prepared('covered_advance')
        t = timeline('covered_advance').t
        mid = float(t[len(t)//2])
        half = {'version': 1, 'kind': 'coach-possession-spans', 'source': 'coach',      # the script, as a coach marks it
                'spans': [{'t0': max(s['t0'], float(t[0])), 't1': s['t1'], 'state': s['state']}
                          for s in b.truth['possession'] if s['t1'] <= mid]}
        tl = timeline('covered_advance', coach=half)
        first = tl.t <= half['spans'][-1]['t1']
        inside = tl.coachMarked
        self.assertGreater(inside[first].mean(), .95)
        self.assertFalse(tl.nearChange[inside].any())             # no exclusion at the coach's own boundaries
        later = tl.t > mid+5
        self.assertTrue(tl.nearChange[later].any())               # the engine's windows still apply outside them
        self.assertTrue((tl.phaseSource[later & np.isin(tl.phase, ('own', 'opp'))] == 'possession-4').all())
        truth = script_phase(b.truth, tl.t, 'A')
        named = np.isin(tl.phase, ('own', 'opp'))
        self.assertTrue((tl.phase[named] == truth[named]).all())
        coach = tl.summary()['phaseQuality']['coach']
        self.assertGreater(coach['agreement'], .99)
        sources = {s['source'] for s in tl.summary()['phaseSpans']}
        self.assertEqual(sources, {'coach', 'possession-4'})
        self.assertEqual(S.validate('timeline', tl.summary()), [])

    def test_a_coach_marked_change_suppresses_two_seconds_either_side(self):
        """S-16 (FACTS L10): +-2 s around a coach-marked change between the two teams' spans the phase stays the
        coach's but the samples suppress (coachNear); a dead span between them is a restart, not a change."""
        b, st, cd, poss = prepared('covered_advance')
        t = timeline('covered_advance').t
        t0 = float(t[0])
        marks = {'version': 1, 'kind': 'coach-possession-spans', 'source': 'coach',
                 'spans': [{'t0': t0+10.0, 't1': t0+30.0, 'state': 'A'}, {'t0': t0+30.0, 't1': t0+50.0, 'state': 'B'},
                           {'t0': t0+50.0, 't1': t0+60.0, 'state': 'dead'}, {'t0': t0+60.0, 't1': t0+80.0, 'state': 'A'}]}
        tl = timeline('covered_advance', coach=marks)
        near = tl.coachNear
        around = np.abs(tl.t-(t0+30.0)) <= SI.CHANGE_EXCLUDE_S-.2
        self.assertTrue(near[around].all())
        self.assertTrue(np.isin(tl.phase[around], ('own', 'opp')).all())        # the phase stays named
        restart = np.abs(tl.t-(t0+60.0)) <= SI.CHANGE_EXCLUDE_S
        self.assertFalse(near[restart].any())                                      # dead -> A: no change
        self.assertFalse(near[np.abs(tl.t-(t0+30.0)) > SI.CHANGE_EXCLUDE_S+.2].any())
        self.assertFalse(tl.nearChange[tl.coachMarked].any())                      # the engine's windows untouched

    def test_shares_follow_the_coach_marks_and_one_side_is_explained(self):
        """B-11 (CODE H2, critic N6): with marks and the engine available the rough shares are the merged phase's;
        with the engine unavailable and only one team marked there are no shares and a note says what to mark."""
        b, st, cd, poss = prepared('covered_advance')
        t = timeline('covered_advance').t
        t0 = float(t[0])
        marks = {'version': 1, 'kind': 'coach-possession-spans', 'source': 'coach',
                 'spans': [{'t0': t0+5.0, 't1': t0+25.0, 'state': 'A'}, {'t0': t0+25.0, 't1': t0+33.5, 'state': 'B'}]}
        tl = timeline('covered_advance', coach=marks)
        sm = tl.summary()
        sec = sm['phaseSeconds']
        want = round(sec['own']/(sec['own']+sec['opp'])/SI.ROUGH_SHARE_STEP)*SI.ROUGH_SHARE_STEP
        self.assertEqual(sm['phaseQuality']['roughShares'], {'A': round(want, 2), 'B': round(1.0-want, 2)})
        self.assertEqual(S.validate('timeline', sm), [])
        b, st, cd, poss = prepared('amateur_coach_spans', 'spans')
        own = {'version': 1, 'kind': 'coach-possession-spans', 'source': 'coach',
               'spans': [x for x in b.coach_spans['spans'] if x['state'] == 'A']}
        for team, side in (('A', 'A'), ('B', 'A')):
            tl = timeline('amateur_coach_spans', 'spans', team=team, coach=own)
            q = tl.summary()['phaseQuality']
            self.assertIsNone(q['roughShares'])
            self.assertEqual(SI.one_sided(q['coach'], q['available'], team), side)
            self.assertTrue(any("marked only Team A's possession" in n for n in q['notes']), q['notes'])
            self.assertEqual(S.validate('timeline', tl.summary()), [])
        both = timeline('amateur_coach_spans', 'spans', coach=b.coach_spans).summary()['phaseQuality']
        self.assertIsNone(SI.one_sided(both['coach'], both['available'], 'A'))

    def test_the_ball_sigma_is_read_per_sample(self):
        tl = timeline('covered_advance')
        seen = np.isfinite(tl.ballSigma)
        self.assertGreater(seen.mean(), .5)
        np.testing.assert_allclose(tl.ballSigma[seen], .3, atol=1e-6)       # the scenes write pitchSigma .3 m

    def test_spans_outside_the_segment_are_not_used(self):
        b, st, cd, poss = prepared('amateur_coach_spans', 'spans')
        bad = {'version': 1, 'kind': 'coach-possession-spans', 'source': 'coach',
               'spans': [{'t0': 1.0, 't1': 5.0, 'state': 'A'}, {'t0': 2000.0, 't1': 2010.0, 'state': 'B'}]}
        tl = timeline('amateur_coach_spans', 'spans', coach=bad)
        q = tl.summary()['phaseQuality']
        self.assertEqual((q['coach']['spans'], q['coach']['dropped']), (0, 2))
        self.assertTrue((tl.phase == 'unknown').all())
        self.assertTrue(any('not used' in n for n in q['notes']))
        self.assertEqual(S.validate('timeline', tl.summary()), [])


class FormatAndScale(unittest.TestCase):
    def test_completeness_follows_the_coaching_format(self):
        b, st, cd, poss = prepared('kids_swarm', 'measured_size')
        tl = timeline('kids_swarm', 'measured_size')
        self.assertEqual(tl.format['outfield'], 4)
        self.assertTrue((tl.outfield == 4).all())
        self.assertTrue((np.asarray(cd.teams['A'].expected) == 3).all())    # team_shape's lineup: the roster minus one
        self.assertGreater(tl.complete.mean(), .9)
        self.assertEqual(tl.summary()['format'], '4v4')
        self.assertEqual(tl.summary()['formationPeriods'], [])
        self.assertTrue((tl.formationPeriod == -1).all() and not tl.formationChangeNear.any())

    def test_more_players_than_the_format_is_never_complete(self):
        tl = timeline('covered_advance', measurement={'format': {'players': 9, 'goalkeeper': True}, 'field': None})
        self.assertTrue((tl.outfield == 8).all())
        self.assertTrue(tl.excess[tl.observedCount == 10].all())
        self.assertFalse(tl.complete.any())          # ten seen for a 9v9: a wrong format or a duplicate identity
        self.assertEqual(S.validate('timeline', tl.summary()), [])

    def test_a_declared_small_field_rescales_and_an_assumed_one_is_not_usable(self):
        assumed = timeline('small_field_scale', 'assumed')
        declared = timeline('small_field_scale', 'declared')
        self.assertEqual(assumed.scale, {'lengthM': 105.0, 'widthM': 68.0, 'source': 'assumed', 'metresUsable': False})
        self.assertEqual(declared.scale, {'lengthM': 55.0, 'widthM': 36.0, 'source': 'declared', 'metresUsable': True})
        self.assertLessEqual(np.nanmax(declared.X), 55.0*1.03)
        ok = declared.velOk & assumed.velOk
        np.testing.assert_allclose(declared.velX[ok], assumed.velX[ok]*55.0/105.0)
        np.testing.assert_allclose(declared.velY[ok], assumed.velY[ok]*36.0/68.0)
        for tl in (assumed, declared):
            self.assertEqual(S.validate('timeline', tl.summary()), [])
        self.assertTrue(SI.scale_of(prepared('covered_advance')[2], {'players': 11}, None)['metresUsable'])

    def test_the_view_follows_the_scale(self):
        assumed = timeline('small_field_scale', 'assumed')
        declared = timeline('small_field_scale', 'declared')
        s = next(i for i, v in enumerate(assumed.view) if v is not None and v is not False)
        np.testing.assert_allclose(declared.view[s], assumed.view[s]*np.array([55.0/105.0, 36.0/68.0]))


class Formation(unittest.TestCase):
    def test_a_formation_change_marks_fifteen_seconds_either_side(self):
        tl = timeline('formation_change')
        labels = [p['label'] for p in tl.summary()['formationPeriods']]
        self.assertEqual(labels, ['4-4-2', '3-5-2'])
        boundary = tl.summary()['formationPeriods'][1]['t0']
        near = np.abs(tl.t-boundary) <= 15.0-.2
        far = np.abs(tl.t-boundary) > 15.0+.2
        self.assertTrue(tl.formationChangeNear[near].all())
        self.assertFalse(tl.formationChangeNear[far].any())
        self.assertEqual(set(tl.formationLabel.tolist()), {'4-4-2', '3-5-2'})

    def test_no_change_without_two_labels(self):
        self.assertFalse(timeline('covered_advance').formationChangeNear.any())

    def test_only_confident_lasting_changes_suppress(self):
        """S-3 (STATS 5): a change suppresses only between two readings that each last 60 s and read clear / likely
        (or coach-set), never when the labels change more than 6 times per 10 minutes."""
        class Shape:
            def __init__(self, periods):
                self.shapes = {'A': {'periods': periods, 'windows': [], 'labels': []}}

        def per(t0, t1, label, conf='clear'):
            return {'t0': t0, 't1': t1, 'label': label, 'confidence': conf}
        t = np.arange(0.0, 600.0, .2)
        fmt = {'players': 11}

        def near(periods):
            return SI._formation(Shape(periods), 'A', t, fmt)[3]
        n = near([per(0.0, 200.0, '4-4-2'), per(200.0, 600.0, '3-5-2')])
        self.assertTrue(n[np.abs(t-200.0) <= 15.0-.2].all() and not n[np.abs(t-200.0) > 15.2].any())
        self.assertFalse(near([per(0.0, 200.0, '4-4-2'), per(200.0, 240.0, '3-5-2'),
                               per(240.0, 600.0, '4-4-2')]).any())              # a 40 s reading: no change
        self.assertFalse(near([per(0.0, 200.0, '4-4-2'), per(200.0, 600.0, '3-5-2', 'unclear')]).any())
        self.assertTrue(near([per(0.0, 200.0, '4-4-2', 'likely'), per(200.0, 600.0, '3-5-2', 'likely')]).any())
        churn = [per(60.0*k, 60.0*(k+1), ('4-4-2', '3-5-2')[k % 2]) for k in range(10)]
        churn += [per(600.0, 700.0, '3-5-2')]
        t2 = np.arange(0.0, 700.0, .2)
        n2 = SI._formation(Shape(churn), 'A', t2, fmt)[3]
        self.assertFalse(n2.any())                                               # 10 changes in 11.7 minutes


class Gates(unittest.TestCase):
    def test_identity_taint_around_a_crossing(self):
        b, st, cd, poss = prepared('identity_crossing')
        tl = timeline('identity_crossing')
        j = tl.col['A-01']
        times = sorted({i['time'] for i in b.result['issues'] if i['id'] == 'A-01'})
        for tc in times:
            near = np.abs(tl.t-tc) <= 2.0-.1
            self.assertTrue(tl.tainted[near, j].all())
            s = int(np.argmin(np.abs(tl.t-tc)))
            self.assertFalse(tl.named_ok('A-01', s))
            self.assertEqual(tl.identity_level('A-01', np.flatnonzero(near)), 'low')
        far = np.min(np.abs(tl.t[:, None]-np.array(times)[None, :]), axis=1) > 2.1
        self.assertFalse(tl.tainted[far, j].any())
        self.assertTrue(tl.oppTainted[:, tl.oppCol['B-08']].any())
        self.assertEqual(tl.identity_level('A-02', np.arange(len(tl.t))), 'high')
        self.assertIsNone(tl.identity_level('X-1', np.arange(3)))

    def test_view_and_unplaced_gates(self):
        tl = timeline('three_visible')
        self.assertEqual(tl.summary()['completeShare'], 0.0)
        s = next(i for i in range(len(tl.t)) if tl.posOk[i].any() and tl.view[i] is not None)
        j = int(np.flatnonzero(tl.posOk[s])[0])
        self.assertTrue(tl.disc_in_view(s, tl.X[s, j], tl.Y[s, j], 0.5))
        self.assertFalse(tl.disc_in_view(s, tl.X[s, j], tl.Y[s, j], 40.0))
        full = timeline('covered_advance')
        self.assertTrue(full.local_context(0, 50.0, 30.0, 5.0))
        full.arrays['unplaced'] = full.unplaced.copy()
        full.arrays['unplaced'][0] = 1
        self.assertFalse(full.local_context(0, 50.0, 30.0, 5.0))
        self.assertTrue(full.local_context(0, 50.0, 30.0, 5.0, teams=('opp',)))

    def test_the_pitch_position_gate_follows_the_registration(self):
        tl = timeline('low_calibration')
        self.assertEqual(set(tl.regGate.tolist()), {''})       # .55 is above MEDIUM: positions stand
        self.assertTrue(tl.posOk.any())


class DataQuality(unittest.TestCase):
    def test_the_dominant_cause(self):
        self.assertIsNone(SI._dominant_cause(10, 10.1, 10.0, 9.5))
        self.assertEqual(SI._dominant_cause(10, 5.7, 2.9, 2.0), 'out_of_view')       # the Veo clip, team A
        self.assertEqual(SI._dominant_cause(10, 9.3, 7.9, 7.0), 'identity')
        self.assertEqual(SI._dominant_cause(10, 10.1, 9.8, 7.8), 'registration')     # broadcast 42-222 s, team A
        self.assertIsNone(SI._dominant_cause(None, 1, 1, 1))


class Degenerate(unittest.TestCase):
    def test_an_empty_result_and_a_team_nobody_identified(self):
        from soccer.coaching import roles as RO
        import team_shape_synth as syn
        meas7 = {'format': {'players': 7, 'goalkeeper': True}, 'field': None}
        cd = series.build_series({'frames': [], 'match': {}, 'roster': {}, 'tactical': {}})
        tl = SI.build_timeline(cd, None, 'A', meas7)
        self.assertEqual(S.validate('timeline', tl.summary()), [])
        json.dumps(tl.role_inputs(), allow_nan=False)
        book = RO.build_rolebook({'team': 'A', 'timeline': tl.summary(), 'roleInputs': {'context': tl.role_inputs()}},
                                 {'roles': {}, 'formation': {'intended': None, 'periods': []}})
        self.assertEqual((book['records'], S.validate('rolebook', book)), ([], []))
        times, teams, ends = syn.match('4-4-2', '4-4-2', seconds=30, fps=10)
        cd = series.build_series(syn.make_result(times, {'A': teams['A']}, ends))
        tl = SI.build_timeline(cd, None, 'B', {'format': {'players': 11, 'goalkeeper': True}, 'field': None})
        sm = tl.summary()
        self.assertEqual(S.validate('timeline', sm), [])
        self.assertEqual((sm['completeShare'], sm['dataQuality']['dominantCause']), (0.0, 'out_of_view'))
        self.assertRaises(ValueError, SI.build_timeline, cd, None, 'C', meas7)


class Orchestration(unittest.TestCase):
    """The stage in the measurement tier (coaching.build_observations) with stand-ins for the later stages: the obs
    file validates and its timeline does not change with the interpretation settings (invariant 2)."""

    def test_build_observations_with_the_real_situations_stage(self):
        import soccer.coaching as coaching
        b, st, cd, poss = prepared('covered_advance')

        def detect(tl):
            return {'opportunities': [], 'detectors': {}, 'roleInputs': {'context': tl.role_inputs()}}

        def moments(cd_, tl, opps):
            return {'moments': [], 'frameTiers': []}
        stages = {'detect': detect, 'moments': moments}
        outs = []
        for level in (1, 4, 7):
            obs = coaching.build_observations(b.result, 'A', coaching=cs.settings_for(b, level=level), series=cd,
                                              possession=poss, result_sha='0'*64, stages=stages)
            self.assertEqual(S.validate('obs', obs), [])
            outs.append(json.dumps(obs, sort_keys=True))
        self.assertEqual(len(set(outs)), 1)
        self.assertEqual(json.loads(outs[0])['roleInputs']['context']['team'], 'A')


if __name__ == '__main__':
    unittest.main()
