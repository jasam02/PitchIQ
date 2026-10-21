"""The coaching detectors (soccer/coaching/detectors.py): every classification rule is explicit and tested on
synthetic matches whose truth is known (tests/team_shape_synth.py layouts, moved by hand): a repeated spacing problem
is found against the professional window distribution and counted as distinct situations, one continuous situation is
one occurrence and never an improvement on its own, a short segment never makes a concern, low confidence never yields
an improvement, a one-team professional distribution never carries an improvement, a strength needs the favourable
side of the professional median, enough measurement for the clip and a bounded share beyond the professional p75, a
pooled strength quotes its strictest context, a neutral observation says which state it is in, a formation change
between unrelated shapes is a moment to review and one between related shapes is not, an isolated player is a review
(Holm-corrected over the players), the players' coverage decides who can be assessed, ids are stable and are internal
ids (a correction that relabels an identity changes the observation's players; display labels are never stored), and
without a professional reference nothing comparative is claimed ("no professional reference available").

The professional reference here is inline and synthetic (SYNTHETIC_REFERENCE), shaped like reference.json
(contract section 2, with the window distributions of tools/coaching_bench.py): the code under test must work from
whatever reference it is given.

Run:  python -m unittest test_coaching_detectors          (from this directory)
"""
import copy
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (ROOT, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)
import corrections as C  # noqa: E402
import soccer.coaching as coaching  # noqa: E402
import team_shape_synth as syn  # noqa: E402
from soccer.coaching import detectors as det, episodes as ep, reference as rf, series as cs  # noqa: E402

FPS = 5.0
GAP_PUSH = 10.0         # m the first midfield line moves forward in a pushed spell (normal gap about 15 m)
M1 = slice(4, 8)        # the 4-4-2 layout's first midfield line (team_shape_synth.LINES)


def quantiles(p50, p90, p95, n=20000):
    """A reference context Q shaped like reference.json's (contract section 2)."""
    return {'n': n, 'p05': p50*.5, 'p10': p50*.6, 'p25': p50*.8, 'p50': p50, 'p75': (p50+p90)/2, 'p90': p90,
            'p95': p95, 'p975': p95+(p95-p90)}


def rates(p10=.1, p50=.2, p90=.35, d50=7.0, d90=15.0):
    return {'default': {'perMinute': {'p10': p10, 'p50': p50, 'p90': p90}, 'durationSeconds': {'p50': d50, 'p90': d90},
                        'shareOfTime': .05, 'count': 40, 'measuredMinutes': 110.0}}


def window_levels(W):
    """The synthetic professional window levels at W measured minutes: p99 2 up to 3 minutes, 3 up to 5, then 0.6 W;
    the maximum one more; the median 0.3 W rounded down."""
    p99 = 2.0 if W <= 3 else (3.0 if W <= 5 else round(.6*W, 2))
    return {'p50': float(math.floor(.3*W)), 'p90': p99-.5, 'p95': p99-.25, 'p99': p99, 'p999': p99+1, 'max': p99+1}


def windows(levels=window_levels, hist=(900, 90, 10), both=True, minutes=rf.WINDOW_MINUTES):
    """A windowCounts block (tools/coaching_bench.py's shape): one entry per length, both teams unless `both` is
    False (then team B gave 5% of the windows)."""
    n = int(sum(hist))
    entries = []
    for W in minutes:
        e = dict(levels(W), minutes=float(W), n=n, nByTeam={'A': n//2, 'B': n-n//2} if both else {'A': n-n//20, 'B': n//20},
                 bothTeams=bool(both), independent=10.0, hist=list(hist))
        entries.append(e)
    return {'default': {'slideSeconds': 5.0, 'needSeconds': 5.0, 'minTeamShare': .2, 'entries': entries}}


def synthetic_reference(contexts=None, **override):
    """An inline reference: 'all' contexts only (a context missing falls back to 'all'), episode rates and window
    distributions for every metric and context, the default rule. contexts: {metric: {context: Q}} added; override:
    {metric: Q for 'all'}."""
    base = {'width': quantiles(55.0, 62.0, 64.0), 'depth': quantiles(33.0, 45.0, 48.0),
            'hull': quantiles(950.0, 1500.0, 1650.0), 'spread': quantiles(17.0, 22.0, 23.0),
            'gapDM': quantiles(16.0, 18.0, 20.0), 'gapMF': quantiles(14.0, 22.0, 25.0),
            'lineStd': quantiles(2.5, 7.0, 8.5), 'lineHeight': quantiles(35.0, 55.0, 60.0),
            'nearestTeammate': quantiles(14.0, 19.0, 21.0), 'lagDM': quantiles(.2, .8, 1.0)}
    base.update(override)
    metrics = {}
    for m, q in base.items():
        ctx = {'all': q}
        ctx.update((contexts or {}).get(m, {}))
        metrics[m] = {'unit': 'm', 'contexts': ctx, 'episodes': {c: rates() for c in ctx},
                      ('playerWindowCounts' if m == 'nearestTeammate' else 'windowCounts'): {c: windows() for c in ctx}}
    return {'version': 1, 'source': 'synthetic-test-reference', 'sampleRate': 5.0, 'metrics': metrics,
            'rules': {'default': {'enter': 'p95', 'exit': 'p90', 'minSeconds': 5.0, 'mergeSeconds': 3.0}}}


SYNTHETIC_REFERENCE = synthetic_reference()


def ramp(times, spells, amount, rise=2.0):
    """amount inside each (t0, t1) spell, ramped over `rise` seconds at both ends (no registration-like jump)."""
    out = np.zeros(len(times))
    for a, b in spells:
        up = np.clip((times-a)/rise, 0, 1)
        down = np.clip((b-times)/rise, 0, 1)
        out = np.maximum(out, amount*np.minimum(up, down))
    return out


def gap_pushes(spells, amount=GAP_PUSH):
    def overrides(times, pos):
        pos = pos.copy()
        pos[:, M1, 0] += ramp(times, spells, amount)[:, None]
        return pos
    return overrides


def build(seconds=240.0, start=0.0, layout_a='4-4-2', layout_b='4-2-3-1', overrides_a=None, track_a=None, seed=0,
          edit=None, **kw):
    """A synthetic result (team A defends the left goal) with absolute frame indices (the segment may start later in
    the video) and numbered goalkeepers as real runs have. edit(k, t, frame) changes frames in place."""
    times = start+np.arange(0, seconds, 1/FPS)
    out_a = track_a(times) if track_a is not None else syn.team_track(times, layout_a, 'left', seed, overrides=overrides_a)
    teams = {'A': (out_a, syn.goalkeeper_track(times, 'left', seed)),
             'B': (syn.team_track(times, layout_b, 'right', seed+1), syn.goalkeeper_track(times, 'right', seed+1))}
    result = syn.make_result(times, teams, {'left': 'A', 'right': 'B'}, **kw)
    result['video'].update(width=1920, height=1080)
    names = {'GK-A': 'GK-1', 'GK-B': 'GK-2'}
    for k, f in enumerate(result['frames']):
        f['frame'] = int(round(f['time']*FPS))
        for p in f['people']:
            p['id'] = names.get(p['id'], p['id'])
            p.setdefault('display', p['id'])
        if edit is not None:
            edit(k, f['time'], f)
    for g in result['match']['goalkeepers']:
        g['id'] = g['display'] = names[g['id']]
    return result


def observe(result, team='A', reference=SYNTHETIC_REFERENCE, **kw):
    """(observations without evidence hints, the series) of one team."""
    cd = cs.build_series(result)
    obs = det.detect(cd, team, reference, **kw)
    return [det.public(o) for o in obs], cd


def pick(observations, detector, pattern=None, cls=None):
    out = [o for o in observations if o['detector'] == detector and (pattern is None or o['pattern'] == pattern)
           and (cls is None or o['classification'] == cls)]
    return out


def occ(t0, t1, seconds=None, censored=False):
    return {'t0': t0, 't1': t1, 'seconds': seconds if seconds is not None else t1-t0, 'censored': censored}


RULE = dict(det.DEFAULT_RULE)


def win(minutes, reference=SYNTHETIC_REFERENCE, context='all'):
    return rf.window_baseline(reference, 'gapDM', context, minutes)


PASSES = {'passes': True, 'enough': True, 'favourable': True}


class ClassificationRules(unittest.TestCase):
    """detectors.classify, rule by rule (module docstring)."""

    def test_insufficient_without_reference_time_or_stretch(self):
        self.assertEqual(det.classify(300, 60, 0, RULE, win(5), PASSES, 'high', has_reference=False)[:3],
                         ('insufficient', None, ['no_reference']))
        self.assertEqual(det.classify(19.8, 19.8, 0, RULE, win(.33), PASSES, 'high')[2], ['short_measured'])
        self.assertEqual(det.classify(60, 4.8, 0, RULE, win(1), PASSES, 'high')[2], ['short_stretch'])

    def test_low_confidence_never_yields_an_improvement(self):
        cls, sev, why, _ = det.classify(300, 120, 8, RULE, win(5), None, 'low')
        self.assertEqual((cls, sev, why), ('insufficient', None, ['low_confidence']))

    def test_an_improvement_needs_two_and_more_than_the_professional_p99(self):
        w4 = win(4.0)                                    # next longer length 4 min: p99 3, maximum 4
        self.assertEqual((w4['p99'], w4['max'], w4['method']), (3.0, 4.0, 'exact'))
        self.assertEqual(det.classify(240, 100, 3, RULE, w4, None, 'high')[0], 'neutral')      # 3 is not beyond 3
        cls, sev, _, stats = det.classify(240, 100, 4, RULE, w4, None, 'high')
        self.assertEqual((cls, sev), ('improvement', 'potential_improvement'))                # 4 < 1.5 x 3
        self.assertEqual(stats['windowP99'], 3.0)
        self.assertEqual(det.classify(240, 100, 5, RULE, w4, None, 'high')[:2], ('improvement', 'high_confidence_concern'))
        self.assertEqual(det.classify(240, 100, 5, RULE, w4, None, 'medium')[:2], ('improvement', 'potential_improvement'))
        # never a single occurrence, whatever the professional level
        low = dict(w4, p99=.4, max=1.0)
        self.assertNotEqual(det.classify(240, 100, 1, RULE, low, None, 'high')[0], 'improvement')
        self.assertEqual(det.classify(240, 100, 2, RULE, low, None, 'high')[0], 'improvement')

    def test_the_next_longer_professional_length_is_compared(self):
        self.assertEqual((win(3.4)['from'], win(3.4)['method'], win(3.4)['p99']), ([4.0], 'next_longer', 3.0))
        self.assertEqual(win(.4)['from'], [1.0])                                              # the shortest length
        many = win(90.0)                                  # beyond the longest length: scaled from it
        self.assertEqual((many['method'], many['p99']), ('scaled', round(18.0*90/30, 3)))

    def test_a_one_team_distribution_never_carries_an_improvement(self):
        ref = synthetic_reference()
        ref['metrics']['gapDM']['windowCounts']['all'] = windows(both=False)
        w = rf.window_baseline(ref, 'gapDM', 'all', 4.0)
        self.assertEqual((w['available'], w['reason']), (False, 'one_team_baseline'))
        cls, _, _, stats = det.classify(240, 100, 20, RULE, w, None, 'high')
        self.assertEqual((cls, stats['improvementBlocked']), ('neutral', 'one_team_baseline'))
        self.assertEqual(det.classify(240, 100, 20, RULE, None, None, 'high')[3]['improvementBlocked'],
                         'no_window_baseline')

    def test_a_short_segment_never_makes_a_concern(self):
        w = win(2.5)
        cls, sev, _, _ = det.classify(150, 60, 10, RULE, w, None, 'high')
        self.assertEqual((cls, sev), ('improvement', 'potential_improvement'))
        self.assertEqual(det.classify(180, 60, 10, RULE, win(3.0), None, 'high')[1], 'high_confidence_concern')

    def test_a_strength_needs_the_comparison_confidence_and_no_excess(self):
        w10 = win(10.0)                                   # median 3 episodes in ten minutes
        self.assertEqual(det.classify(600, 60, 0, RULE, w10, PASSES, 'high')[:2], ('strength', 'observed_strength'))
        self.assertEqual(det.classify(600, 60, 0, RULE, w10, PASSES, 'medium')[0], 'strength')
        self.assertEqual(det.classify(600, 60, 3, RULE, w10, PASSES, 'high')[0], 'strength')
        self.assertEqual(det.classify(600, 60, 4, RULE, w10, PASSES, 'high')[0], 'neutral')     # beyond the median
        self.assertEqual(det.classify(600, 60, 0, RULE, w10, dict(PASSES, passes=False), 'high')[0], 'neutral')
        self.assertEqual(det.classify(600, 60, 1, RULE, None, PASSES, 'high')[0], 'neutral')   # no median to set it by

    def test_the_strength_comparison(self):
        ref = SYNTHETIC_REFERENCE                         # gapDM p50 16, p75 17, no tau: the 10 s fallback
        calm = np.full(300, 14.0)
        s = det.strength_check(ref, 'gapDM', calm, 60.0, 60.0, 'all')
        self.assertEqual((s['passes'], s['favourable'], s['shareAboveP75'], s['tauSeconds'], s['nEff']),
                         (True, True, 0.0, det.TAU_FALLBACK_SECONDS, 6.0))
        self.assertAlmostEqual(s['upperBound'], 1-.2**(1/6.0), places=3)
        wide = det.strength_check(ref, 'gapDM', np.full(300, 16.5), 60.0, 60.0, 'all')
        self.assertEqual((wide['passes'], wide['failed']), (False, ['unfavourable_side']))
        short = det.strength_check(ref, 'gapDM', calm[:200], 40.0, 60.0, 'all')
        self.assertIn('short_for_strength', short['failed'])
        self.assertIn('evidence_bound', short['failed'])  # 4 looks cannot bound the share below 25%
        # the measured time a strength needs is relative to the clip: max(30 s, min(60 s, 0.8 x range))
        self.assertEqual([det.strength_seconds(r) for r in (20, 45, 60, 600)], [30.0, 36.0, 48.0, 60.0])
        beyond = np.r_[np.full(240, 14.0), np.full(60, 17.5)]          # 20% of the time beyond the p75
        self.assertIn('evidence_bound', det.strength_check(ref, 'gapDM', beyond, 600.0, 600.0, 'all')['failed'])
        rare = np.r_[np.full(295, 14.0), np.full(5, 17.5)]             # 1.7% beyond over 60 looks
        self.assertTrue(det.strength_check(ref, 'gapDM', rare, 600.0, 600.0, 'all')['passes'])

    def test_neutral_states(self):
        one = [dict(occ(10.0, 25.0), peakT=20.0, segmentT0=10.0, segmentT1=25.0,
                    metrics={'width': {'peak': 62.3, 'mean': 60.0}})]
        state, spell = det.neutral_state(one, {'enough': True})
        self.assertEqual((state, spell['t0'], spell['t1'], spell['peak'], spell['peakT']),
                         ('one_spell', 10.0, 25.0, 62.3, 20.0))
        self.assertEqual(det.neutral_state(one*2, {'enough': True})[0], 'within_range')
        self.assertEqual(det.neutral_state([], {'enough': False})[0], 'under_measured')
        self.assertEqual(det.neutral_state([], {'enough': True})[0], 'absent')

    def test_tails(self):
        self.assertAlmostEqual(det.poisson_tail(0, 2.0), 1.0)
        self.assertAlmostEqual(det.poisson_tail(1, 2.0), 1-math.exp(-2.0))
        self.assertAlmostEqual(det.poisson_tail(2, .36), 1-math.exp(-.36)*1.36)
        self.assertAlmostEqual(det.binomial_tail(2, 2, .05), .0025)
        self.assertEqual(det.binomial_tail(3, 2, .05), 0.0)


class RepeatedSpacing(unittest.TestCase):
    """Team A's first midfield line steps 10 m away from the back line six times for about 10 s in 4 minutes: six
    against a professional p99 of 3 in 4 minutes (maximum 4) -> a high-confidence concern."""
    SPELLS = [(20.0, 32.0), (55.0, 67.0), (90.0, 102.0), (125.0, 137.0), (160.0, 172.0), (195.0, 207.0)]

    @classmethod
    def setUpClass(cls):
        cls.result = build(overrides_a=gap_pushes(cls.SPELLS))
        cls.obs, cls.cd = observe(cls.result)
        cls.gap = pick(cls.obs, 'unit_gap_dm')

    def test_a_repeated_spacing_problem_is_detected(self):
        self.assertEqual(len(self.gap), 1)
        o = self.gap[0]
        self.assertEqual((o['classification'], o['severity']), ('improvement', 'high_confidence_concern'))
        self.assertEqual(o['pattern'], 'space_between_defence_and_midfield')
        self.assertEqual((o['recommendationKey'], o['category'], o['metric']), ('compact_lines', 'spacing', 'gapDM'))
        self.assertGreater(o['occurrenceCount'], o['metrics']['professionalWindowP99'])
        w = o['baseline']['professionalWindow']
        self.assertEqual((w['available'], w['p99'], w['from']), (True, 3.0, [4.0]))
        self.assertIn(w['method'], ('exact', 'next_longer'))
        self.assertEqual(o['baseline']['threshold']['enter'], 20.0)
        self.assertEqual(o['baseline']['reference'], 'synthetic-test-reference@inline')
        self.assertIn('defensive_compactness', o['drillIds'])
        s = self.cd.teams['A']
        lines = np.array(syn.LINES['4-4-2'])
        ids = set(s.ids[j] for j in np.flatnonzero(lines <= 1))
        self.assertTrue(set(o['players']) <= ids and len(o['players']) == 8)
        for x in o['occurrences']:
            self.assertTrue(set(x['players']) == ids)
            self.assertGreaterEqual(x['metrics']['gapDM']['peak'], 20.0)

    def test_distinct_events_are_counted_once_each(self):
        o = self.gap[0]
        self.assertEqual(o['occurrenceCount'], len(self.SPELLS))
        for x, (a, b) in zip(o['occurrences'], self.SPELLS):
            self.assertLessEqual(a-1, x['t0'])
            self.assertLessEqual(x['t1'], b+1)
            self.assertGreaterEqual(x['seconds'], 5.0)
            self.assertFalse(x['censored'])
        self.assertAlmostEqual(o['ratePerMeasuredMinute'], len(self.SPELLS)/(o['measuredSeconds']/60), places=2)

    def test_measured_time_and_coverage(self):
        o = self.gap[0]
        cov = self.cd.teams['A'].coverage('gapDM')
        self.assertAlmostEqual(o['measuredSeconds'], cov['measuredSeconds'], places=1)
        self.assertAlmostEqual(o['eligibleShare'], cov['eligibleShare'], places=2)
        self.assertGreaterEqual(o['longestStretchSeconds'], 60.0)
        self.assertEqual(o['segmentKind'], 'long')
        self.assertEqual(o['confidence']['overall'], 'high')

    def test_a_short_segment_is_never_a_concern(self):
        obs, _ = observe(self.result, t0=0.0, t1=100.0)
        o = pick(obs, 'unit_gap_dm')[0]
        self.assertEqual(o['occurrenceCount'], 3)
        self.assertEqual(o['segmentKind'], 'short')
        self.assertEqual((o['classification'], o['severity']), ('improvement', 'potential_improvement'))
        self.assertEqual(o['window'], {'t0': 0.0, 't1': 100.0, 'segmentT0': 0.0, 'segmentT1': 100.0})

    def test_without_a_reference_nothing_comparative_is_claimed(self):
        obs, _ = observe(self.result, reference=False)
        for o in obs:
            if o['detector'] in ('formation_stability', 'ball_context'):
                continue
            self.assertEqual(o['classification'], 'insufficient', o['detector'])
            self.assertIn('no_reference', o['insufficientReasons'], o['detector'])
            self.assertEqual(o['occurrences'], [])
        f = pick(obs, 'formation_stability')[0]
        self.assertEqual((f['classification'], f['state']), ('neutral', 'under_measured'))   # 240 s: under 5 minutes

    def test_a_high_line_with_a_large_gap_is_a_review_never_a_mistake(self):
        # With this reference the line counts as high from 24 m (normal D height about 26 +- 10 m).
        ref = synthetic_reference(lineHeight=quantiles(20.0, 24.0, 26.0))
        obs, _ = observe(self.result, reference=ref)
        hl = pick(obs, 'line_height_and_gap', cls='review')
        self.assertTrue(hl)
        for o in hl:
            self.assertEqual((o['classification'], o['severity']), ('review', 'review_opportunity'))
            for x in o['occurrences']:
                self.assertGreaterEqual(x['metrics']['gapDM']['peak'], 20.0)
                self.assertGreaterEqual(x['metrics']['lineHeight']['peak'], 24.0)
        # with the default reference the line is never high: nothing to review, one neutral note of the block height
        note = pick(self.obs, 'line_height_and_gap')
        self.assertEqual([(o['classification'], o['state']) for o in note], [('neutral', 'absent')])
        m = note[0]['metrics']
        s, other = self.cd.teams['A'], self.cd.teams['B']
        self.assertAlmostEqual(m['lineHeight'], float(np.nanmedian(s.lineHeight[s.eligible['lineHeight']])), delta=.06)
        self.assertAlmostEqual(m['opponentLineHeight'],
                               float(np.nanmedian(other.lineHeight[other.eligible['lineHeight']])), delta=.06)
        self.assertGreater(m['opponentLineHeightSeconds'], 60.0)
        self.assertEqual(note[0]['occurrences'], [])

    def test_observations_are_json_safe(self):
        json.dumps(self.obs, allow_nan=False)


class OneSituation(unittest.TestCase):
    def test_a_continuous_ten_second_gap_is_one_event(self):
        obs, cd = observe(build(seconds=120.0, overrides_a=gap_pushes([(40.0, 54.0)])))
        o = pick(obs, 'unit_gap_dm')[0]
        self.assertEqual(o['occurrenceCount'], 1)
        x = o['occurrences'][0]
        self.assertTrue(8.0 <= x['seconds'] <= 16.0, x['seconds'])
        # sample by sample it would have been ~50 threshold crossings' worth of samples
        self.assertGreater(x['seconds']*FPS, 40)

    def test_a_brief_dip_inside_one_situation_does_not_split_it(self):
        def overrides(times, pos):
            pos = pos.copy()
            push = ramp(times, [(40.0, 60.0)], GAP_PUSH)
            dip = (times > 49.0) & (times < 50.6)                 # 1.6 s back to normal, < merge 3 s
            push[dip] = 0.0
            pos[:, M1, 0] += push[:, None]
            return pos
        obs, _ = observe(build(seconds=120.0, overrides_a=overrides))
        o = pick(obs, 'unit_gap_dm')[0]
        self.assertEqual(o['occurrenceCount'], 1)
        self.assertGreater(o['occurrences'][0]['seconds'], 15.0)

    def test_a_single_long_situation_is_one_spell_never_an_improvement(self):
        obs, _ = observe(build(seconds=120.0, overrides_a=gap_pushes([(30.0, 60.0)])))
        o = pick(obs, 'unit_gap_dm')[0]
        self.assertEqual((o['occurrenceCount'], o['classification'], o['state']), (1, 'neutral', 'one_spell'))
        spell = o['metrics']['spell']
        x = o['occurrences'][0]
        self.assertEqual((spell['t0'], spell['t1'], spell['peakT']), (x['t0'], x['t1'], x['peakT']))
        self.assertEqual(spell['peak'], x['metrics']['gapDM']['peak'])


class Maintained(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = build(seconds=120.0)
        cls.obs, cls.cd = observe(cls.result)

    def test_staying_within_the_professional_range_is_a_strength(self):
        o = pick(self.obs, 'unit_gap_dm')[0]
        self.assertEqual((o['classification'], o['severity'], o['occurrenceCount']), ('strength', 'observed_strength', 0))
        m = o['metrics']
        self.assertTrue(m['favourableSide'])
        self.assertLessEqual(m['median'], m['professionalP50'])
        self.assertLess(m['upperBoundShareAboveP75'], det.STRENGTH_PRO_SHARE)
        self.assertEqual(m['tauSeconds'], det.TAU_FALLBACK_SECONDS)
        self.assertEqual(o['drillIds'], [])
        self.assertNotIn('state', o)
        b = pick(self.obs, 'back_line_coordination')[0]
        self.assertEqual(b['classification'], 'strength')

    def test_a_wide_team_is_never_called_compact(self):
        # The synthetic team's area (about 1340 m2) is beyond this reference's professional median (950 m2).
        a = pick(self.obs, 'compactness', 'team_area')[0]
        self.assertEqual(a['classification'], 'neutral')
        self.assertFalse(a['metrics']['favourableSide'])
        self.assertIn('unfavourable_side', a['metrics']['strengthFailed'])
        self.assertEqual(a['state'], 'absent')

    def test_a_formation_held_for_two_minutes_is_described_not_a_strength(self):
        f = pick(self.obs, 'formation_stability')[0]
        self.assertEqual((f['classification'], f['pattern'], f['metrics']['label'], f['state']),
                         ('neutral', 'formation_stable', '4-4-2', 'under_measured'))
        self.assertEqual(f['metrics']['defenderSetChanges'], 0)
        self.assertEqual((f['metrics']['playersKeptRole'], f['metrics']['playersAssessed']), (10, 10))
        self.assertEqual(f['metrics']['roleStability'], 1.0)
        self.assertEqual(set(f['metrics']['roleShares']), set(self.cd.teams['A'].ids))
        long, _ = observe(build(seconds=320.0))
        f = pick(long, 'formation_stability')[0]
        self.assertEqual((f['classification'], f['metrics']['label']), ('strength', '4-4-2'))
        self.assertGreaterEqual(f['measuredSeconds'], det.FORMATION_STRENGTH_SECONDS)

    def test_a_strength_needs_enough_evidence_for_the_clip(self):
        obs, _ = observe(self.result, t0=0.0, t1=50.0)       # 50 s: 5 looks at the 10 s decorrelation time
        for o in obs:
            self.assertNotEqual(o['classification'], 'strength', o['detector'])
        gap = pick(obs, 'unit_gap_dm')[0]
        self.assertEqual((gap['classification'], gap['state']), ('neutral', 'absent'))
        self.assertAlmostEqual(gap['metrics']['strengthSeconds'], det.strength_seconds(50.2), places=1)
        self.assertIn('evidence_bound', gap['metrics']['strengthFailed'])

    def test_every_registry_detector_reports(self):
        self.assertEqual(list(det.DETECTORS), ['compactness', 'unit_gap_dm', 'unit_gap_mf', 'back_line_coordination',
                                               'line_height_and_gap', 'formation_stability', 'lateral_shift',
                                               'isolation', 'ball_context'])
        found = {o['detector'] for o in self.obs}
        self.assertEqual(found, set(det.DETECTORS))
        self.assertEqual({o['pattern'] for o in pick(self.obs, 'compactness')}, {'team_width', 'team_depth', 'team_area'})
        for o in self.obs:
            self.assertIn(o['classification'], ('strength', 'improvement', 'review', 'neutral', 'insufficient'))
            self.assertEqual(o['engineVersion'], ep.ENGINE_VERSION)
            if o['classification'] == 'insufficient':
                self.assertTrue(o['insufficientReasons'], o['detector'])
            if o['classification'] == 'neutral':
                self.assertIn(o['state'], ('absent', 'one_spell', 'within_range', 'under_measured', 'descriptive',
                                           'related_shapes'), o['detector'])


class Contexts(unittest.TestCase):
    def test_a_gap_normal_in_general_but_large_for_the_own_third_is_found_there(self):
        # Team A's centre is in its own third most of the time; a 15 m gap is normal pooled, beyond the own-third
        # p95 of this reference (12 m): an improvement in context third:own, and its text names that context.
        res = build(seconds=180.0, overrides_a=gap_pushes([(20.0, 32.0), (70.0, 82.0), (120.0, 132.0)], 3.0))
        ref = synthetic_reference(contexts={'gapDM': {'third:own': quantiles(9.0, 11.0, 12.0),
                                                      'third:middle': quantiles(15.0, 18.0, 20.0)}})
        obs, cd = observe(res, reference=ref)
        gap = pick(obs, 'unit_gap_dm')
        own = [o for o in gap if o['baseline']['context'] == 'third:own']
        self.assertTrue(own)
        self.assertEqual(own[0]['classification'], 'improvement')
        self.assertEqual(own[0]['baseline']['threshold']['enter'], 12.0)
        s = cd.teams['A']
        for x in own[0]['occurrences']:
            i = int(np.argmin(np.abs(cd.t-x['peakT'])))
            self.assertEqual(s.third[i], 'own')
        # pooled with the 'all' levels the same team is not flagged
        pooled, _ = observe(res)
        self.assertNotEqual(pick(pooled, 'unit_gap_dm')[0]['classification'], 'improvement')
        # a context whose professional windows come from one team carries no improvement, and says so
        ref['metrics']['gapDM']['windowCounts']['third:own'] = windows(both=False)
        obs, _ = observe(res, reference=ref)
        for o in pick(obs, 'unit_gap_dm'):
            self.assertNotEqual(o['classification'], 'improvement')
        pooled = [o for o in pick(obs, 'unit_gap_dm') if 'third:own' in (o['metrics'].get('contexts') or {})]
        self.assertTrue(pooled)
        self.assertIsNone(pooled[0]['metrics']['contexts']['third:own']['professionalWindowP99'])

    def test_a_pooled_strength_holds_in_every_context_and_quotes_the_strictest(self):
        # Team A's width (about 54 m) is in its own third 40% of the time and in the middle third 60%.
        res = build(seconds=240.0)
        easy = synthetic_reference(contexts={'width': {'third:own': quantiles(60.0, 66.0, 68.0),
                                                       'third:middle': quantiles(57.0, 63.0, 65.0)}})
        obs, _ = observe(res, reference=easy)
        w = pick(obs, 'compactness', 'team_width')[0]
        self.assertEqual((w['classification'], w['baseline']['context']), ('strength', 'third:middle'))
        m = w['metrics']
        self.assertEqual(m['strictestContext'], 'third:middle')          # nearer its own median than the own third
        self.assertEqual(m['professionalP50'], 57.0)
        self.assertEqual(m['median'], m['contexts']['third:middle']['median'])
        self.assertIn('pooledMedian', m)
        self.assertLess(m['comparisonSeconds'], w['measuredSeconds'])
        # the same pooled team against an own third whose median it exceeds: no strength, and that comparison quoted
        hard = synthetic_reference(contexts={'width': {'third:own': quantiles(50.0, 56.0, 58.0),
                                                       'third:middle': quantiles(57.0, 63.0, 65.0)}})
        obs, _ = observe(res, reference=hard)
        w = pick(obs, 'compactness', 'team_width')[0]
        self.assertEqual((w['classification'], w['baseline']['context']), ('neutral', 'third:own'))
        self.assertIn('unfavourable_side', w['metrics']['strengthFailed'])
        self.assertEqual(w['metrics']['professionalP50'], 50.0)

    def test_the_back_line_is_compared_by_its_size(self):
        ref = synthetic_reference(contexts={'lineStd': {'backline:4': quantiles(2.8, 7.8, 9.8),
                                                        'backline:3': quantiles(2.2, 4.9, 6.1)}})
        obs, _ = observe(build(seconds=120.0), reference=ref)
        b = pick(obs, 'back_line_coordination')[0]
        self.assertEqual(b['baseline']['context'], 'backline:4')
        self.assertEqual(b['metrics']['lineSize'], 4)
        self.assertEqual(b['baseline']['threshold']['p90'], 7.8)


class Confidence(unittest.TestCase):
    def test_unsupported_conclusions_are_suppressed(self):
        # The same repeated spacing problem, but every player of team A crosses a similar player every 3 s: the
        # identities are not trustworthy, so nothing is claimed about them.
        result = build(overrides_a=gap_pushes(RepeatedSpacing.SPELLS))
        result['issues'] = [{'id': f'A-{j:02d}', 'time': float(t), 'reason': 'Crossed a similar player: check identities.'}
                            for t in np.arange(0, 240, 3.0) for j in range(1, 11)]
        obs, _ = observe(result)
        for o in obs:
            self.assertNotIn(o['classification'], ('improvement',), o['detector'])
            self.assertNotEqual(o['severity'], 'high_confidence_concern')
        o = pick(obs, 'unit_gap_dm')[0]
        self.assertEqual(o['classification'], 'insufficient')
        self.assertEqual(o['confidence']['identity'], 'low')
        self.assertIn('low_confidence', o['insufficientReasons'])
        self.assertIn('crossings', o['confidence']['reasons'])

    def test_a_medium_registration_lowers_the_calibration_aspect(self):
        def edit(k, t, f):
            f['calibration'] = {'confidence': .5}
        obs, _ = observe(build(seconds=120.0, edit=edit))
        o = pick(obs, 'unit_gap_dm')[0]
        self.assertEqual(o['confidence']['calibration'], 'medium')
        self.assertIn('medium_registration', o['confidence']['reasons'])
        self.assertEqual(o['confidence']['overall'], 'medium')


class Formation(unittest.TestCase):
    @staticmethod
    def switch(first, second, at=120.0):
        def track(times):
            a = syn.team_track(times, first, 'left', 0)
            b = syn.team_track(times, second, 'left', 0)
            out = a.copy()
            out[times >= at] = b[times >= at]
            return out
        return track

    def test_a_formation_change_is_a_moment_to_review_not_a_mistake(self):
        obs, _ = observe(build(seconds=240.0, track_a=self.switch('4-4-2', '3-5-2')))
        f = pick(obs, 'formation_stability')
        self.assertEqual(len(f), 1)
        f = f[0]
        self.assertEqual((f['classification'], f['severity'], f['pattern']),
                         ('review', 'review_opportunity', 'formation_change'))
        self.assertEqual(f['metrics']['labels'], ['4-4-2', '3-5-2'])
        self.assertEqual(f['occurrenceCount'], 2)
        self.assertEqual([x['label'] for x in f['occurrences']], ['4-4-2', '3-5-2'])
        self.assertIn('formation_shadow_play', f['drillIds'])
        self.assertIsNone(f['ratePerMeasuredMinute'])
        for k in ('4-4-2', '3-5-2'):
            self.assertGreaterEqual(f['metrics']['labelSeconds'][k], det.FORMATION_SUSTAINED_SECONDS)

    def test_related_shapes_are_one_shape_read_two_ways(self):
        self.assertTrue(det._related('4-3-3', '4-1-4-1') and det._related('5-3-2', '3-5-2'))
        self.assertFalse(det._related('4-4-2', '3-5-2') or det._related('4-4-2', '4-4-2'))
        obs, _ = observe(build(seconds=240.0, track_a=self.switch('4-4-2', '4-2-3-1')))
        f = pick(obs, 'formation_stability')[0]
        if len(f['metrics']['labels']) >= 2:
            self.assertEqual((f['classification'], f['pattern'], f['state']),
                             ('neutral', 'formation_change', 'related_shapes'))
        else:
            self.assertNotEqual(f['classification'], 'review')
        self.assertEqual(f['drillIds'], [])

    def test_the_registration_gates_formation_time(self):
        # The registration chains disagree by 5 m before 200 s: team_shape still reads the formation there (it does
        # not look at the registration), the coaching layer does not count that time.
        def edit(k, t, f):
            f['calibration'] = {'confidence': .9, 'disagreement': 5.0 if t < 200.0 else 1.0}
        obs, _ = observe(build(seconds=240.0, edit=edit))
        f = pick(obs, 'formation_stability')[0]
        self.assertLessEqual(f['measuredSeconds'], 40.2)
        self.assertGreater(f['metrics']['unregisteredSeconds'], 150.0)
        self.assertNotEqual(f['classification'], 'strength')

    def test_back_line_changes_count_between_complete_windows_only(self):
        times = np.arange(0, 240.0, 1/FPS)
        visible = {'A': np.ones((len(times), 10), bool)}
        visible['A'][(times >= 100) & (times < 130), 1] = False      # a defender unseen for 30 s: 9-player windows
        obs, _ = observe(build(seconds=240.0, visible=visible))
        f = pick(obs, 'formation_stability')[0]
        self.assertEqual(f['metrics']['defenderSetChanges'], 0)
        self.assertLess(f['metrics']['completeWindows'], f['metrics']['classifiedWindows'])


class Players(unittest.TestCase):
    """Player A-10 (a forward) leaves his team-mates behind three times; A-09 is hardly ever seen."""

    @classmethod
    def setUpClass(cls):
        spells = [(30.0, 45.0), (100.0, 115.0), (170.0, 185.0)]

        def overrides(times, pos):
            pos = pos.copy()
            pos[:, 9, 0] += ramp(times, spells, 25.0)
            return pos
        times = np.arange(0, 240.0, 1/FPS)
        visible = {'A': np.ones((len(times), 10), bool)}
        visible['A'][:, 8] = (times % 60.0) < 3.0              # A-09: 3 s a minute
        cls.result = build(overrides_a=overrides, visible=visible)
        # Without A-09 the other forward's nearest team-mate is a midfielder 18-22 m away: levels above that, and
        # per-player windows in which no professional player had three spells (p = 0 for A-10).
        cls.ref = synthetic_reference(nearestTeammate=quantiles(14.0, 23.0, 25.0))
        cls.obs, cls.cd = observe(cls.result, reference=cls.ref)

    def test_an_isolated_player_is_a_role_check_review(self):
        iso = pick(self.obs, 'isolation')
        mine = [o for o in iso if o['players'] == ['A-10']]
        self.assertEqual(len(mine), 1)
        o = mine[0]
        self.assertEqual((o['classification'], o['severity'], o['pattern']), ('review', 'review_opportunity',
                                                                              'player_isolation'))
        self.assertEqual(o['occurrenceCount'], 3)
        self.assertEqual(o['drillIds'], [])                  # a role check, not a training need
        self.assertTrue(o['metrics']['roleCheck'])
        self.assertEqual(o['metrics']['pValue'], 0.0)
        self.assertLessEqual(o['metrics']['pValue'], o['metrics']['holmLevel'])
        self.assertTrue(o['baseline']['professionalWindow']['perPlayer'])
        cov = self.cd.teams['A'].coverage('nearestTeammate', 'A-10')
        self.assertLessEqual(o['measuredSeconds'], cov['measuredSeconds']+.01)
        for x in o['occurrences']:
            self.assertEqual(x['players'], ['A-10'])
        # A-09 is measured for far less than MIN_MEASURED_SECONDS: no observation of his own
        seen = self.cd.teams['A'].coverage('nearestTeammate', 'A-09')['measuredSeconds']
        self.assertLess(seen, det.MIN_MEASURED_SECONDS)
        self.assertFalse([o for o in iso if o['players'] == ['A-09']])
        # the reviews stand alone: no team-level "no finding" note contradicts them
        self.assertFalse([o for o in iso if o['classification'] == 'neutral'])
        self.assertFalse([o for o in iso if o['classification'] == 'improvement'])

    def test_the_players_are_corrected_for_multiple_testing(self):
        # Professional players reach three spells in 4 minutes 1% of the time: p .01 alone, but not after Holm over
        # the nine players tested (.01 / 9).
        ref = copy.deepcopy(self.ref)
        ref['metrics']['nearestTeammate']['playerWindowCounts'] = {'all': windows(hist=(900, 80, 10, 10)),
                                                                   'unit:F': windows(hist=(900, 80, 10, 10))}
        obs, _ = observe(self.result, reference=ref)
        iso = pick(obs, 'isolation')
        self.assertFalse([o for o in iso if o['classification'] == 'review'])
        team = [o for o in iso if o['players'] != ['A-10']]
        self.assertEqual(team[0]['metrics']['testedPlayers'], 9)
        self.assertEqual(rf.holm([.01, .5, None, .2], .01), [])
        self.assertEqual(rf.holm([.001, .5, None, .004], .01), [0, 3])
        self.assertEqual(rf.holm([.004, .003], .01), [1, 0])

    def test_isolation_is_vetoed_on_small_scene_runs(self):
        def edit(k, t, f):
            f['profile'] = {'smallScene': True}
        res = build(seconds=120.0, edit=edit)
        obs, _ = observe(res)
        iso = pick(obs, 'isolation')
        self.assertEqual(len(iso), 1)
        self.assertEqual(iso[0]['classification'], 'insufficient')
        self.assertIn('small_scene_veto', iso[0]['insufficientReasons'])
        self.assertIn('broadcast_only', iso[0]['insufficientReasons'])


class Lateral(unittest.TestCase):
    @staticmethod
    def track(delay):
        """Team A sliding across the pitch with period 20 s, its back line `delay` seconds behind the others."""
        def make(times):
            base = np.array(syn.LAYOUTS['4-4-2'], float)
            rng = np.random.default_rng(3)
            Y = base[None, :, 1]+8.0*np.sin(2*math.pi*times[:, None]/20.0)
            Y[:, :4] = base[None, :4, 1]+8.0*np.sin(2*math.pi*(times[:, None]-delay)/20.0)
            X = base[None, :, 0]+3.0*np.sin(2*math.pi*times[:, None]/37.0)
            X, Y = X+rng.normal(0, .15, X.shape), Y+rng.normal(0, .15, Y.shape)
            x, y = syn.to_pitch(X, Y, 'left')
            return np.stack([x, y], axis=2)
        return make

    def test_a_back_line_that_slides_late_is_found(self):
        obs, _ = observe(build(seconds=180.0, track_a=self.track(2.0)))
        o = pick(obs, 'lateral_shift')[0]
        self.assertEqual(o['classification'], 'improvement')
        self.assertAlmostEqual(o['metrics']['lag'], 2.0, delta=.4)
        self.assertAlmostEqual(o['metrics']['lag']/.2, round(o['metrics']['lag']/.2), places=6)   # 5 Hz resolution
        self.assertGreater(o['metrics']['corr'], .8)
        self.assertEqual(o['metrics']['windowSeconds'], 30.0)
        # late all along: one continuous situation, and every non-overlapping window beyond the professional level
        self.assertEqual(o['occurrenceCount'], 1)
        self.assertGreater(o['occurrences'][0]['seconds'], 120.0)
        self.assertEqual(o['metrics']['windowsBeyond'], o['metrics']['independentWindows'])
        self.assertGreaterEqual(o['metrics']['independentWindows'], 4)
        self.assertGreater(o['metrics']['windowsBeyond'], o['metrics']['professionalWindowP99'])
        # the reference level shown is the p90, with the share professionals spend beyond it (10%)
        self.assertEqual((o['baseline']['referenceLevel'], o['baseline']['professionalShareAbove']), ('p90', .1))

    def test_overlapping_windows_are_not_independent_trials(self):
        ws = [{'t0': 5.0*k, 't1': 5.0*k+29.8} for k in range(13)]
        self.assertEqual([w['t0'] for w in det._independent(ws, 30.0)], [0.0, 30.0, 60.0])

    def test_a_back_line_that_slides_with_midfield_is_not(self):
        obs, _ = observe(build(seconds=180.0, track_a=self.track(0.0)))
        o = pick(obs, 'lateral_shift')[0]
        self.assertIn(o['classification'], ('strength', 'neutral'))
        self.assertAlmostEqual(o['metrics']['lag'], 0.0, delta=.4)
        self.assertEqual(o['occurrenceCount'], 0)

    def test_a_lag_at_the_search_bound_is_no_measurement(self):
        step = .2
        self.assertTrue(rf.at_search_bound(4.0, step) and rf.at_search_bound(-3.8, step))
        self.assertFalse(rf.at_search_bound(3.7, step))
        t = np.arange(0, 120, step)
        lead = 10.0*np.sin(2*math.pi*t/40)
        follow = np.roll(lead, 25)                       # 5 s later: beyond the +-4 s search, the best is its edge
        ok = np.ones(len(t), bool)
        kept = rf.lateral_windows(t, lead, follow, ok, 30.0)
        every = rf.lateral_windows(t, lead, follow, ok, 30.0, drop_bound=False)
        self.assertTrue(every)
        self.assertTrue(all(rf.at_search_bound(w['lag'], step) for w in every if w not in kept))
        self.assertFalse(any(rf.at_search_bound(w['lag'], step) for w in kept))
        self.assertLess(len(kept), len(every))
        self.assertEqual([rf.quantize_lag(x, step) for x in (-.15, .09, .31, 1.06)], [-.2, 0.0, .4, 1.0])


class Ball(unittest.TestCase):
    def test_the_ball_context_is_descriptive(self):
        def edit(k, t, f):
            x = .2 if (k // 50) % 2 == 0 else .5                  # alternately in A's own and middle third
            f['ball'] = {'state': 'TRACKED', 'pitch': [x, .5], 'pitchConfidence': .9, 'pitchSigma': [.5, .5]}
        obs, _ = observe(build(seconds=120.0, edit=edit))
        b = pick(obs, 'ball_context')[0]
        self.assertEqual((b['classification'], b['state']), ('neutral', 'descriptive'))
        th = b['metrics']['thirds']
        self.assertAlmostEqual(sum(th.values()), 1.0, places=2)
        self.assertGreater(th['own'], .4)
        self.assertGreater(th['middle'], .4)
        self.assertEqual(b['occurrences'], [])
        self.assertIsNone(b['ratePerMeasuredMinute'])

    def test_without_a_ball_it_is_insufficient(self):
        obs, _ = observe(build(seconds=60.0))
        b = pick(obs, 'ball_context')[0]
        self.assertEqual(b['classification'], 'insufficient')
        self.assertIn('no_ball', b['insufficientReasons'])


class Identities(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = build(overrides_a=gap_pushes(RepeatedSpacing.SPELLS))

    def test_ids_are_stable(self):
        a, _ = observe(self.result)
        b, _ = observe(copy.deepcopy(self.result))
        self.assertEqual([o['id'] for o in a], [o['id'] for o in b])
        self.assertEqual([x['id'] for o in a for x in o['occurrences']], [x['id'] for o in b for x in o['occurrences']])
        for o in a:
            self.assertRegex(o['id'], r'^obs-[0-9a-f]{12}$')
            for x in o['occurrences']:
                self.assertRegex(x['id'], r'^occ-[0-9a-f]{12}$')
        self.assertEqual(len({o['id'] for o in a}), len(a))
        other, _ = observe(self.result, t0=10.0)
        self.assertFalse({o['id'] for o in a} & {o['id'] for o in other})

    def test_a_corrected_identity_changes_the_players_and_no_display_label_is_stored(self):
        corrected = C.apply_result(copy.deepcopy(self.result), [{
            'id': 'c1', 'run': 'run-1', 'made': None, 'time': 10.0, 'target': {'id': 'A-03', 'track': None},
            'before': {'role': 'player', 'team': 'A', 'id': 'A-03'},
            'after': {'role': 'player', 'team': 'A', 'id': 'A-15', 'into': None}, 'identity': None}], block=False)
        displays = {p['id']: p['display'] for p in corrected['match']['players']}
        self.assertEqual(displays['A-04'], 'A-03')            # labels renumbered: A-03 now shows on A-04
        before = coaching.build_observations(self.result, team='A', reference=SYNTHETIC_REFERENCE)
        after = coaching.build_observations(corrected, team='A', reference=SYNTHETIC_REFERENCE)
        gap_b = pick(before['observations'], 'unit_gap_dm')[0]
        gap_a = pick(after['observations'], 'unit_gap_dm')[0]
        self.assertIn('A-03', gap_b['players'])
        self.assertIn('A-15', gap_a['players'])
        self.assertNotIn('A-03', gap_a['players'])
        self.assertEqual(gap_a['occurrenceCount'], gap_b['occurrenceCount'])
        text = json.dumps({k: after[k] for k in ('observations', 'evidence')}, allow_nan=False)
        self.assertNotIn('"A-03"', text)                      # an id that no longer exists, nor a display label
        self.assertNotIn('display', text)


class Engine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.dir = Path(cls.tmp.name)
        cls.result = build(seconds=120.0, overrides_a=gap_pushes([(30.0, 42.0), (80.0, 92.0)]))
        (cls.dir/'result.json').write_text(json.dumps(cls.result), encoding='utf-8')
        (cls.dir/'reference.json').write_text(json.dumps(SYNTHETIC_REFERENCE), encoding='utf-8')

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_the_package_exports(self):
        for name in ('build_observations', 'build_report', 'build_report_for', 'ENGINE_VERSION'):
            self.assertTrue(callable(getattr(coaching, name)) or name == 'ENGINE_VERSION', name)
        out = coaching.build_observations(self.result, reference=SYNTHETIC_REFERENCE)
        self.assertEqual({o['team'] for o in out['observations']}, {'A', 'B'})
        self.assertFalse(any(k.startswith('_') for o in out['observations'] for k in o))
        meta = out['meta']
        self.assertEqual((meta['footage'], meta['range']['t0'], meta['segmentStart']), ('broadcast', 0.0, 0.0))
        self.assertIn('synthetic-test-reference', meta['reference'])
        self.assertEqual(coaching.build_observations(self.result, reference=False)['meta']['limitations'],
                         ['no_reference'])
        # the mean observed outfield players per team as numbers (report.py's data-quality note), the same as the
        # coverage text row
        self.assertEqual(set(meta['observedPlayers']), {'A', 'B'})
        for team, v in meta['observedPlayers'].items():
            self.assertIn(f'Team {team} {v:.1f}', meta['coverage']['observedOutfieldPlayers'])

    def test_the_opponents_block_height_needs_the_same_measured_time(self):
        # amateur 118-182 s: team B's back line was measured 18 s, so team B's report has no block height; team A's
        # report quoted B's 67.9 m anyway (over 18 s)
        from types import SimpleNamespace as NS
        n = 300                                                           # 60 s at 5 Hz

        def side(h, k):
            ok = np.zeros(n, bool)
            ok[:k] = True
            return NS(values={'lineHeight': np.full(n, h)}, eligible={'lineHeight': ok})
        for k_opp, quoted in ((90, False), (100, True), (300, True)):    # 18 s, 20 s, 60 s
            a = NS(rows=np.arange(n), s=side(15.1, 300), other=side(67.9, k_opp),
                   measured=lambda m: float(np.sum(m))*.2)
            b = det.block_heights(a)
            self.assertEqual(b['lineHeight'], 15.1)
            self.assertEqual(b['opponentLineHeight'], 67.9 if quoted else None, k_opp)
            self.assertAlmostEqual(b['opponentLineHeightSeconds'], k_opp*.2)

    def test_the_server_entry_builds_a_report(self):
        report = coaching.build_report_for(self.dir/'result.json', 'A', {
            'context': 'advanced', 'reference': SYNTHETIC_REFERENCE, 'run': 'run-1', 'fingerprint': 'f'*64,
            'video': {'key': 'k'*16, 'name': 'clip.mp4'}})
        self.assertEqual((report['team'], report['context'], report['kind'], report['run']),
                         ('A', 'advanced', 'segment', 'run-1'))
        self.assertTrue(all(o['team'] == 'A' for o in report['observations']))
        self.assertEqual(report['video'], {'name': 'clip.mp4', 'key': 'k'*16})
        json.dumps(report, allow_nan=False)

    def test_the_command_line_tool(self):
        out = self.dir/'out.json'
        run = subprocess.run([sys.executable, '-B', os.path.join(ROOT, 'tools', 'coaching_report.py'),
                              str(self.dir/'result.json'), '--team', 'A', '--reference', str(self.dir/'reference.json'),
                              '--out', str(out)], capture_output=True, text=True, timeout=300,
                             env=dict(os.environ, CUDA_VISIBLE_DEVICES='-1'))
        self.assertEqual(run.returncode, 0, run.stderr)
        data = json.loads(out.read_text(encoding='utf-8'))
        self.assertTrue(data['observations'])
        self.assertTrue(all(o['team'] == 'A' for o in data['observations']))
        self.assertEqual(set(data['reports']), {'A'})
        self.assertIn('unit_gap_dm', run.stderr)
        self.assertTrue(re.search(r'\d+ observations, \d+ evidence moments', run.stderr))


if __name__ == '__main__':
    unittest.main()
