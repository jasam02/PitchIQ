"""The coaching v2 opportunity finders (soccer/coaching/contextual.py, DESIGN v2 4.0-4.6): where a behaviour was
relevant and what was measured there, profile-independently, on EVAL's synthetic scenes (tools/coaching_synth.py:
the real analytics, team_shape, possession-4, series and situation timeline under them) whose planted behaviour is
the truth.

Why these checks. assess.py trusts these records for every level, so each gate DESIGN 4 names is a test that fails
when the gate is removed: no opportunity outside its phase (and none inside a change window), no first defender
without a non-estimated carrier and a view that rules out an unseen nearer defender, no team-level relation without
the whole team, no wide-defender opportunity without a coach / stable / carried role (or a medium relative slot on
small formats), never while he sprints back, the D-M gap left out when units interleave; abstention reasons and
seconds are always reported (spec Part 6), the role tier reads only the obs file, everything is JSON-safe and
deterministic, and the planted outcomes are what the relations say: pressure with / without cover, nobody pressing,
an empty far side, an open back-line gap with an opponent in it, covered / uncovered / exposed advances.

Run:  python -m unittest test_coaching_contextual          (from this directory)
"""
import copy
import json
import unittest
from collections import Counter

import numpy as np

import coaching_detect_fixtures as FX
from soccer.coaching import contextual as CX, roles as RO, schema as S, situations as SI  # noqa: E402

NAN = float('nan')


def opps(found, detector):
    return [o for o in found['opportunities'] if o['detector'] == detector]


def outcome_counts(found, detector, scale, params=None):
    c, m = Counter(), Counter()
    for o in opps(found, detector):
        outs, modes = CX.sample_outcomes(o, params, scale)
        c.update(x for x in outs if x is not None)
        m.update(x for x in modes if x)
    return c, m


class Segments(unittest.TestCase):
    T = 100.0+.2*np.arange(200)

    def test_runs_bridge_short_gaps_split_on_keys_and_cut_long_ones(self):
        e = np.zeros(200, bool)
        e[0:20] = True              # 4 s
        e[22:30] = True             # after 0.4 s missing: bridged
        e[40:44] = True             # 0.8 s alone: dropped (min 1 s)
        e[60:160] = True            # 20 s: cut at 10 s
        found = CX.segments(self.T, e, min_s=1.0, max_s=10.0, step=.2)
        self.assertEqual([(int(s[0]), int(s[-1])) for s in found], [(0, 29), (60, 109), (110, 159)])
        self.assertNotIn(20, found[0].tolist())             # the bridged gap's samples are left out
        keys = np.array(['a']*10+['b']*190, dtype=object)
        found = CX.segments(self.T, e, keys, min_s=1.0, max_s=10.0, step=.2)
        self.assertEqual([(int(s[0]), int(s[-1])) for s in found][:2], [(0, 9), (10, 29)])

    def test_a_short_tail_joins_the_piece_before_it(self):
        e = np.zeros(200, bool)
        e[0:54] = True              # 10.8 s: a 0.8 s tail
        found = CX.segments(self.T, e, min_s=1.0, max_s=10.0, step=.2)
        self.assertEqual([(int(s[0]), int(s[-1])) for s in found], [(0, 53)])

    def test_channel_in_view_by_station_mask(self):
        mask = sum(1 << k for k in range(2, 10))     # stations 10-45 m in view
        self.assertTrue(CX.channel_in_view(mask, 12.0, 40.0))
        self.assertFalse(CX.channel_in_view(mask, 5.0, 40.0))
        self.assertFalse(CX.channel_in_view(mask, 12.0, 47.0))
        self.assertTrue(CX.channel_in_view(None, 0.0, 100.0))      # no view field: nothing vetoes
        self.assertFalse(CX.channel_in_view(0, 10.0, 11.0))


class Coverage(unittest.TestCase):
    def test_dominant_reason_prefers_the_detectors_own_gates(self):
        o = {'relations': {'t': [1.0, 1.2]}, 'quality': {'samples': 2}, 't0': 1.0, 't1': 1.2}
        c = CX.coverage('weak_side_balance', 'measurement', [o], .2, 50.0,
                        {'near_change': 30.0, 'team_incomplete': 10.0, 'phase_unknown': 5.0})
        self.assertEqual(c['dominantReason'], 'team_incomplete')
        c = CX.coverage('weak_side_balance', 'measurement', [], .2, 0.0, {'phase_unknown': 30.0, 'near_change': 3.0})
        self.assertEqual((c['dominantReason'], c['eligibleSeconds']), ('phase_unknown', 0.0))
        self.assertEqual(S.validate('detector_coverage', c), [])

    def test_eligible_seconds_count_each_sample_time_once(self):
        a = {'relations': {'t': [1.0, 1.2, 1.4]}, 'quality': {'samples': 3}, 't0': 1.0, 't1': 1.4}
        b = {'relations': {'t': [1.2, 1.4, 1.6]}, 'quality': {'samples': 3}, 't0': 1.2, 't1': 1.6}
        self.assertAlmostEqual(CX.eligible_seconds([a, b], .2), .8)
        self.assertEqual(CX.coverage('wide_defender_advance', 'role', [a, b], .2, 10.0, {})['eligibleSpans'],
                         [[1.0, 1.6]])


class MeasurementTier(unittest.TestCase):
    def test_records_validate_and_respect_the_phase_rules(self):
        for scene, variant in (('unsupported_advance', 'none_behind'), ('defending', 'press_no_cover'),
                               ('amateur_coach_spans', 'spans'), ('kids_swarm', None), ('three_visible', None)):
            tl = FX.timeline(scene, variant)
            found = CX.detect(tl)
            json.dumps(found, allow_nan=False)
            for o in found['opportunities']:
                self.assertEqual(S.validate('opportunity', o), [], (scene, o['key']))
                self.assertEqual(o['tier'], 'measurement')
                rows = tl.rows(o['t0'], o['t1'])
                self.assertFalse(tl.nearChange[rows].any() and o['detector'] != 'attacking_shape')
                if o['detector'] in ('behind_ball', 'defensive_spacing', 'attacking_shape') and \
                        o['phase']['state'] != 'unknown':
                    self.assertGreaterEqual(o['phase']['spanSeconds'], SI.MIN_SHAPE_SPAN_S)
            for c in found['detectors'].values():
                self.assertEqual(S.validate('detector_coverage', c), [])
            self.assertIn('context', found['roleInputs'])
            self.assertEqual(found['roleInputs']['detect']['version'], CX.DETECT_VERSION)

    def test_fix_round_3_relations(self):
        """Fix round 3 (LEAD decision (2), critic N1): every measurement-tier opportunity carries contestS per sample
        (situations.contestS, None where none), D5 carries gapOppFreeM (the least-marked opponent in the gap: his
        distance to the nearest of ours goal-side of him, FREE_CAP_M when nobody is; None without an opponent there);
        the relations measured before are unchanged."""
        tl = FX.timeline('defending', 'gap_open')
        found = CX.detect(tl)
        for o in found['opportunities']:
            self.assertEqual(len(o['relations']['contestS']), len(o['relations']['t']), o['key'])
            k = [int(np.argmin(np.abs(tl.t-x))) for x in o['relations']['t']]
            want = [None if not np.isfinite(v) else round(float(v), 2) for v in tl.contestS[k]]
            self.assertEqual(o['relations']['contestS'], want)
        d5 = [o for o in found['opportunities'] if o['detector'] == 'defensive_spacing']
        self.assertTrue(d5)
        for o in d5:
            R = o['relations']
            for k, opp in enumerate(R['opponentInGap']):
                self.assertEqual(R['gapOppFreeM'][k] is None, not opp)
        OX, OY = np.array([30.0, 25.0]), np.array([20.0, 30.0])
        self.assertEqual(CX.gap_opponent_free_m(OX, OY, [0], np.array([25.0, 35.0]), np.array([22.0, 20.0])),
                         round(float(np.hypot(5.0, 2.0)), 2))          # only the goal-side one counts
        self.assertEqual(CX.gap_opponent_free_m(OX, OY, [0, 1], np.array([29.0, 50.0]), np.array([20.0, 30.0])),
                         CX.FREE_CAP_M)                                # nobody of ours goal-side of the second
        self.assertEqual(CX.gap_opponent_free_m(OX, OY, [], np.array([29.0]), np.array([20.0])), None)
        self.assertEqual(CX.gap_opponent_free_m(OX, OY, [0], None, None), None)

    def test_phase_reasons(self):
        phase = np.array(['own', 'opp', 'unknown', 'unknown', 'dead'])
        near = np.array([False, False, True, False, False])
        self.assertEqual(CX._phase_reasons(phase, near, 'opp').tolist(),
                         [CX.NA, '', 'near_change', 'phase_unknown', CX.NA])

    def test_shape_checks_need_phase_spans_of_three_seconds(self):
        b, st, cd, poss = FX.prepared('defending', 'press_cover')
        meas = S.tier_settings(st['teams']['A'])['measurement']
        base = SI.build_timeline(cd, poss, 'A', meas)
        opp = np.flatnonzero(base.phase == 'opp')
        t0 = float(base.t[opp[40]])
        for length, expect in ((2.0, False), (6.0, True)):     # a coach marks our ball for 2 s / 6 s
            tl = SI.build_timeline(cd, poss, 'A', meas, coach_spans=[{'t0': t0, 't1': t0+length, 'state': 'A'}])
            found = CX.detect(tl)
            inside = [o for o in opps(found, 'attacking_shape') if o['t0'] < t0+length and o['t1'] >= t0]
            self.assertEqual(bool(inside), expect, length)

    def test_deterministic(self):
        a = CX.detect(FX.timeline('defending', 'press_cover'))
        b = CX.detect(FX.timeline('defending', 'press_cover'))
        self.assertEqual(json.dumps(a, sort_keys=True), json.dumps(b, sort_keys=True))

    def test_d1_needs_a_carrier_of_the_other_team_in_its_phase(self):
        tl = FX.timeline('defending', 'press_cover')
        found = CX.detect(tl)
        d1 = opps(found, 'pressure_cover')
        self.assertGreater(len(d1), 10)
        for o in d1:
            rows = tl.rows(o['t0'], o['t1'])
            self.assertEqual(o['phase']['state'], 'opp')
            self.assertTrue(o['actors']['carrier'].startswith('B-'))
            self.assertTrue(all(tl.carrier[r] in ('', o['actors']['carrier']) for r in rows))
            self.assertEqual(o['quality']['visibility'], 'disc')

    def test_planted_pressure_and_cover(self):
        expect = {'press_cover': (1, None), 'press_no_cover': (-1, 'no_cover'), 'no_press': (-1, 'no_pressure')}
        for variant, (sign, mode) in expect.items():
            tl = FX.timeline('defending', variant)
            c, m = outcome_counts(CX.detect(tl), 'pressure_cover', tl.scale)
            self.assertGreater(c[sign], c[-sign], variant)
            if mode:
                self.assertEqual(m.most_common(1)[0][0], mode, variant)

    def test_planted_empty_far_side_and_open_gap(self):
        tl = FX.timeline('defending', 'weak_empty')
        c, _ = outcome_counts(CX.detect(tl), 'weak_side_balance', tl.scale)
        self.assertGreater(c[-1], c[1])
        tl = FX.timeline('defending', 'press_cover')
        c, _ = outcome_counts(CX.detect(tl), 'weak_side_balance', tl.scale)
        self.assertGreaterEqual(c[1], c[-1])
        tl = FX.timeline('defending', 'gap_open')
        c, m = outcome_counts(CX.detect(tl), 'defensive_spacing', tl.scale)
        self.assertGreater(m['lateral_gap'], 0)
        self.assertGreater(c[-1], 0)

    def test_the_first_defender_is_certain(self):
        # a zoomed camera: nobody unseen may be nearer the carrier than the first defender (the disc of radius d1 in
        # view, nobody of our kit unplaced); the samples where that fails are out_of_view, never an opportunity
        tl = FX.timeline('three_visible')
        found = CX.detect(tl)
        self.assertGreater(found['detectors']['pressure_cover']['abstain'].get('out_of_view', 0.0), 0.0)
        checked = 0
        for o in opps(found, 'pressure_cover'):
            for t, d1, cx, cy in zip(o['relations']['t'], o['relations']['d1'], o['relations']['carrierX'],
                                     o['relations']['carrierY']):
                s = int(np.argmin(np.abs(tl.t-t)))
                self.assertTrue(tl.local_context(s, cx, cy, d1-.01, teams=('own',)), t)
                checked += 1
        self.assertGreater(checked, 0)

    def test_team_level_detectors_need_the_whole_team(self):
        tl = FX.timeline('three_visible')
        found = CX.detect(tl)
        for d in ('weak_side_balance', 'behind_ball', 'attacking_shape'):
            self.assertEqual(found['detectors'][d]['opportunities'], 0, d)
            self.assertEqual(found['detectors'][d]['dominantReason'], 'team_incomplete', d)
        for o in opps(found, 'pressure_cover'):
            rows = tl.rows(o['t0'], o['t1'])
            self.assertEqual(int(tl.unplaced[rows].max()), 0)

    def test_no_phase_no_opportunity_but_describe_only_shape(self):
        tl = FX.timeline('possession_unknown')
        found = CX.detect(tl)
        for o in found['opportunities']:
            self.assertEqual((o['detector'], o['phase']['state'], o.get('variant')), ('attacking_shape', 'unknown',
                                                                                         'unknown'))
        for d, c in found['detectors'].items():
            self.assertEqual(c['dominantReason'], 'phase_unknown', d)
        self.assertGreater(found['detectors']['attacking_shape']['opportunities'], 0)

    def test_coach_marked_phase_reaches_the_opportunities(self):
        tl = FX.timeline('amateur_coach_spans', 'spans')
        found = CX.detect(tl)
        shapes = opps(found, 'attacking_shape')
        self.assertTrue(shapes)
        self.assertTrue(all(o['phase']['source'] == 'coach' and not o['phase']['estimated'] for o in shapes))
        self.assertEqual(found['detectors']['pressure_cover']['dominantReason'], 'no_carrier')
        self.assertEqual(found['detectors']['weak_side_balance']['dominantReason'], 'ball_unreliable')
        self.assertEqual(found['detectors']['behind_ball']['dominantReason'], 'ball_unreliable')
        self.assertNotIn('defensive_spacing', found['detectors'])      # small formats: the role tier
        bare = CX.detect(FX.timeline('amateur_coach_spans', 'spans', coach=False))
        self.assertTrue(all(o['phase']['state'] == 'unknown' for o in bare['opportunities']))

    def test_body_scale_needs_metres(self):
        found = CX.detect(FX.timeline('small_field_scale', 'assumed'))
        self.assertEqual(found['detectors']['pressure_cover']['opportunities'], 0)
        self.assertEqual(found['detectors']['pressure_cover']['dominantReason'], 'scale_unknown')
        found = CX.detect(FX.timeline('small_field_scale', 'declared'))
        self.assertGreater(found['detectors']['pressure_cover']['opportunities'], 0)

    def test_role_geometry_keeps_only_long_own_phase_spans_and_is_json(self):
        tl = FX.timeline('unsupported_advance', 'none_behind')
        geo = CX.role_geometry(tl)
        json.dumps(geo, allow_nan=False)
        s = geo['samples']
        self.assertTrue(set(s['phase']) <= {'own'})                  # 11v11: the other team's phase is not stored
        self.assertGreaterEqual(min(s['span']), SI.MIN_SHAPE_SPAN_S)
        self.assertEqual(len(s['X']), len(tl.ids))
        self.assertEqual(len(s['X'][0]), len(s['t']))
        self.assertEqual(s['oX'], [])                                 # opponents only near the channels on 11v11
        small = CX.role_geometry(FX.timeline('amateur_coach_spans', 'spans'))
        self.assertEqual(set(small['samples']['phase']), {'own', 'opp'})
        self.assertEqual(len(small['samples']['oX']), len(FX.timeline('amateur_coach_spans', 'spans').oppIds))


class RoleTier(unittest.TestCase):
    def role_tier(self, scene, variant=None, role=None, team='A'):
        obs, roles = FX.tiers(scene, variant, team=team)
        if role is not None:
            book = RO.build_rolebook(obs, role)
            return obs, CX.detect_roles(obs, book), book
        return obs, {'opportunities': roles['opportunities'], 'detectors': roles['detectors']}, roles['rolebook']

    def test_uncovered_and_covered_advances(self):
        obs, found, _ = self.role_tier('unsupported_advance', 'none_behind')
        lb = FX.prepared('unsupported_advance', 'none_behind')[0].truth['plants'][0]['player']
        d3 = opps(found, 'wide_defender_advance')
        self.assertEqual(len(d3), 16)                    # the 16 planted advances, one opportunity each
        for o in d3:
            self.assertEqual((o['tier'], o.get('variant'), o['actors']['primary']), ('role', 'left', [lb]))
            r = o['relations']
            self.assertFalse(any(r['channelOccupied']) or any(x for x in r['cbAdjusted'] if x))
            self.assertGreaterEqual(max(r['opponentsInChannel']), 1)
            self.assertIn(o['roles'][lb]['source'], ('coach', 'stable', 'carried'))
            self.assertEqual(S.validate('opportunity', o), [])
        obs, found, _ = self.role_tier('covered_advance', 'cb_shift')
        c, _ = outcome_counts(found, 'wide_defender_advance', obs['timeline']['scale'])
        self.assertGreater(c[1], 10*max(1, c[-1]))
        obs, found, _ = self.role_tier('covered_advance', 'channel')
        self.assertTrue(any(any(o['relations']['channelOccupied']) for o in opps(found, 'wide_defender_advance')))

    def test_never_while_the_defender_sprints_back(self):
        # sprint_back: every run back happens inside the change window, which is no phase at all (near_change)
        obs, found, _ = self.role_tier('sprint_back')
        tl = FX.timeline('sprint_back')
        for o in opps(found, 'wide_defender_advance'):
            self.assertTrue(all(v is None or v > -3.0 for v in o['relations']['wdVelX']))
            self.assertFalse(tl.nearChange[tl.rows(o['t0'], o['t1'])].any())
        self.assertGreater(found['detectors']['wide_defender_advance']['abstain'].get('near_change', 0.0), 0.0)
        # the planted advances' returns (3 m/s or more goalwards while still 15 m upfield) are 'recovering'
        obs, found, _ = self.role_tier('unsupported_advance', 'none_behind')
        for o in opps(found, 'wide_defender_advance'):
            self.assertTrue(all(v is None or v > -3.0 for v in o['relations']['wdVelX']))
        self.assertGreater(found['detectors']['wide_defender_advance']['abstain'].get('recovering', 0.0), 0.0)

    def test_a_window_role_is_not_enough(self):
        obs, _, book = self.role_tier('unsupported_advance', 'none_behind')
        weak = copy.deepcopy(book)
        for r in weak['records']:
            if r['source'] in ('stable', 'carried'):
                r['source'], r['confidence'] = 'window', 'low'
        found = CX.detect_roles(obs, weak)
        self.assertEqual(opps(found, 'wide_defender_advance'), [])
        self.assertEqual(found['detectors']['wide_defender_advance']['dominantReason'], 'role_unconfirmed')

    def test_a_coach_role_is_used_and_marked_high(self):
        obs, found, _ = self.role_tier('wrong_formation', 'coach_roles')
        d3 = opps(found, 'wide_defender_advance')
        self.assertTrue(d3)
        self.assertTrue(any(o['roles'].get('A-04', {}).get('source') == 'coach' for o in d3))

    def test_formation_change_is_flagged(self):
        obs, found, _ = self.role_tier('formation_change')
        self.assertTrue(any(o['intentSignals']['formationChangeNear'] for o in opps(found, 'wide_defender_advance')))

    def test_small_formats(self):
        obs, found, _ = self.role_tier('kids_swarm')
        for d in ('wide_defender_advance', 'defensive_spacing'):
            self.assertEqual(found['detectors'][d]['dominantReason'], 'format_not_supported')
            self.assertEqual(opps(found, d), [])
        obs, found, _ = self.role_tier('amateur_coach_spans', 'spans')
        d3 = opps(found, 'wide_defender_advance')
        self.assertTrue(d3)
        self.assertTrue(all(o['roles'][o['actors']['primary'][0]]['source'] == 'relative' for o in d3))
        self.assertTrue(opps(found, 'defensive_spacing'))
        for o in found['opportunities']:
            self.assertEqual(S.validate('opportunity', o), [])

    def test_the_role_tier_reads_only_the_obs_file(self):
        obs, roles = FX.tiers('unsupported_advance', 'none_behind')
        again = CX.detect_roles(json.loads(json.dumps(obs)), roles['rolebook'])
        self.assertEqual(json.dumps(again['opportunities'], sort_keys=True),
                         json.dumps(roles['opportunities'], sort_keys=True))
        stale = copy.deepcopy(obs)
        stale['roleInputs'].pop('detect')
        found = CX.detect_roles(stale, roles['rolebook'])
        self.assertEqual(found['opportunities'], [])


class StoredGeometry(unittest.TestCase):
    """The role tier on a hand-built roleInputs['detect']: a left back (A-02) 30 m beyond two centre-backs, the team
    not complete; the relaxed gate (DESIGN 3.5: the vacated channel in view, nobody unplaced) decides."""

    def obs(self, n=20, complete=False, view=None, unplaced=0, vx=0.5, shift=0.0):
        ids = ['A-02', 'A-04', 'A-05', 'A-06']
        X = {'A-02': 60.0, 'A-04': 30.0, 'A-05': 30.0, 'A-06': 50.0}
        Y = {'A-02': 5.0, 'A-04': 25.0-shift, 'A-05': 43.0, 'A-06': 34.0}
        full = sum(1 << k for k in range(22))
        s = {'t': [100.0+.2*k for k in range(n)], 'frame': [2500+5*k for k in range(n)], 'side': ['left']*n,
             'phase': ['own']*n, 'source': ['possession-4']*n, 'span': [20.0]*n, 'complete': [complete]*n,
             'oppComplete': [False]*n, 'unplaced': [unplaced]*n, 'oppUnplaced': [0]*n, 'fcn': [False]*n,
             'period': [0]*n, 'ballOk': [True]*n, 'ballX': [70.0]*n, 'ballY': [30.0]*n,
             'ballThird': ['final_third']*n, 'ballChannel': ['centre']*n, 'cal': [.9]*n,
             'viewL': [full if view is None else view]*n, 'viewR': [full]*n, 'ok': ['1111']*n, 'taint': ['0000']*n,
             'X': [[X[p]]*n for p in ids], 'Y': [[Y[p]]*n for p in ids],
             'vx': [[vx if p == 'A-02' else 0.0]*n for p in ids], 'threatN': [None]*n,
             'oppL': [[['B-09', 40.0, 6.0]]]*n, 'oppR': [[]]*n, 'oX': [], 'oY': []}
        geo = {'version': CX.DETECT_VERSION, 'team': 'A', 'format': '11v11', 'players': 11, 'outfield': 10,
               'scale': {'lengthM': 105.0, 'widthM': 68.0, 'source': 'assumed', 'metresUsable': True}, 'step': .2,
               'fps': 25.0, 'ids': ids, 'oppIds': ['B-09'], 'phaseSeconds': {'own': 4.0, 'opp': 0.0},
               'lost': {'own': {}, 'opp': {}}, 'medianY': {'A-04': {'0': 25.0}, 'A-05': {'0': 43.0}},
               'stationM': 5.0, 'lengthM': 105.0, 'samples': s}
        return {'team': 'A', 'timeline': {'format': '11v11', 'phaseSeconds': {'own': 4.0}},
                'roleInputs': {'detect': geo}}

    BOOK = {'team': 'A', 'format': '11v11', 'model': 'team_shape', 'interchanges': [], 'formationChecks': [],
            'records': [RO._record(pid, 0.0, 200.0, role, 'stable', 'high')
                        for pid, role in (('A-02', 'LB'), ('A-04', 'LCB'), ('A-05', 'RCB'), ('A-06', 'CM'))]}

    def test_relaxed_gate(self):
        found = CX.detect_roles(self.obs(), self.BOOK)
        d3 = opps(found, 'wide_defender_advance')
        self.assertEqual(len(d3), 1)
        r = d3[0]['relations']
        self.assertTrue(all(r['relaxed']) and not any(r['complete']))
        self.assertEqual((r['advanceM'][0], r['opponentsInChannel'][0], r['channelOccupied'][0], r['cbAdjusted'][0]),
                         (30.0, 1, False, False))
        self.assertEqual(d3[0]['quality']['visibility'], 'partial')
        self.assertEqual(S.validate('opportunity', d3[0]), [])
        for kw in ({'view': 0}, {'unplaced': 1}):
            found = CX.detect_roles(self.obs(**kw), self.BOOK)
            self.assertEqual(opps(found, 'wide_defender_advance'), [], kw)
            self.assertEqual(found['detectors']['wide_defender_advance']['dominantReason'], 'team_incomplete')
        found = CX.detect_roles(self.obs(view=0, complete=True), self.BOOK)
        self.assertFalse(any(opps(found, 'wide_defender_advance')[0]['relations']['relaxed']))

    def test_interchange_needs_the_slot_reoccupied(self):
        """S-4 (STATS 9): an interchange covers the advance only when the player who took the defender's slot stands on
        his lateral half at least 10 m behind him; a label swap alone is no cover and no intent signal."""
        book = copy.deepcopy(self.BOOK)
        book['interchanges'] = [{'t': 101.0, 'out': 'A-02', 'in': 'A-06', 'slot': 'LB'}]
        found = CX.detect_roles(self.obs(), book)          # A-06 at (50, 34): the left half, 10 m behind A-02
        o = opps(found, 'wide_defender_advance')[0]
        self.assertTrue(all(o['relations']['interchangeCover']))
        self.assertEqual(set(o['relations']['interchangeBy']), {'A-06'})
        self.assertEqual(o['intentSignals']['interchange']['in'], 'A-06')
        self.assertEqual(S.validate('opportunity', o), [])
        obs = self.obs()
        obs['roleInputs']['detect']['samples']['Y'][3] = [40.0]*20      # A-06 on the right half: a label swap only
        o = opps(CX.detect_roles(obs, book), 'wide_defender_advance')[0]
        self.assertTrue(all(o['relations']['interchange']) and not any(o['relations']['interchangeCover']))
        self.assertIsNone(o['intentSignals']['interchange'])
        swapped = copy.deepcopy(book)                                    # the defender took someone else's slot
        swapped['interchanges'] = [{'t': 101.0, 'out': 'A-06', 'in': 'A-02', 'slot': 'CM'}]
        o = opps(CX.detect_roles(self.obs(), swapped), 'wide_defender_advance')[0]
        self.assertFalse(any(o['relations']['interchangeCover']))

    def test_ball_depth_and_a_coach_marked_change_window(self):
        """B-1: the advance carries the ball's depth; S-16: samples within 2 s of a coach-marked change are lost to
        near_change, never part of an advance."""
        o = opps(CX.detect_roles(self.obs(), self.BOOK), 'wide_defender_advance')[0]
        self.assertEqual(set(o['relations']['ballX']), {70.0})
        obs = self.obs()
        obs['roleInputs']['detect']['samples']['near'] = [True]*5+[False]*15
        found = CX.detect_roles(obs, self.BOOK)
        o = opps(found, 'wide_defender_advance')[0]
        self.assertAlmostEqual(o['t0'], 101.0)
        self.assertAlmostEqual(found['detectors']['wide_defender_advance']['abstain']['near_change'], 1.0)

    def test_between_the_lines(self):
        """B-7: the D-M gap gate counts opponents between our lines on the ball's half, and those free (nobody of
        ours within 4 m) within 25 m of the ball."""
        OX = np.array([30.0, 30.0, 30.0, 50.0, 30.0])
        OY = np.array([10.0, 20.0, 50.0, 10.0, 12.0])
        X = np.array([20.0, 31.0, 40.0])
        Y = np.array([10.0, 20.5, 30.0])
        self.assertEqual(CX.between_lines(OX, OY, 20.0, 40.0, 35.0, 15.0, 68.0, X, Y), (3, 2))
        self.assertEqual(CX.between_lines(OX, OY, 20.0, 40.0, 70.0, 15.0, 68.0, X, Y), (3, 0))  # ball too far
        self.assertEqual(CX.between_lines(OX, OY, 20.0, 40.0, 35.0, 50.0, 68.0, X, Y), (1, 1))   # the other half
        self.assertEqual(CX.between_lines(OX, OY, 20.0, 40.0, float('nan'), 15.0, 68.0, X, Y), (None, None))

    def test_recovering_and_the_centre_back_shift(self):
        found = CX.detect_roles(self.obs(vx=-3.5), self.BOOK)
        self.assertEqual(opps(found, 'wide_defender_advance'), [])
        self.assertEqual(found['detectors']['wide_defender_advance']['dominantReason'], 'recovering')
        found = CX.detect_roles(self.obs(shift=5.0), self.BOOK)       # the left centre-back 5 m nearer his touchline
        r = opps(found, 'wide_defender_advance')[0]['relations']
        self.assertEqual((r['cbAdjusted'][0], r['nearCbShiftM'][0], r['nearCb'][0]), (True, 5.0, 'A-04'))


class Outcomes(unittest.TestCase):
    def opp(self, detector, **rel):
        n = len(next(iter(rel.values())))
        rel = dict(rel, t=[100.0+.2*k for k in range(n)])
        return {'detector': detector, 'relations': rel}

    def test_pressure_rules_by_level(self):
        """The three-state pressure call (fix round 1, S-2): pressed within 3 m, or within 5 m while closing at 1 m/s;
        not pressed beyond 5 m; 3-5 m without closing is unclear, never "pressed without cover"."""
        o = self.opp('pressure_cover', d1=[2.5, 4.0, 4.0, 7.0, 9.0, 2.5], closingMS=[0.0, 0.0, 1.2, 2.0, 2.0, 0.0],
                     coverDist=[8.0, None, None, None, None, 20.0], coverAngle=[30.0, None, None, None, None, 10.0],
                     coverVisible=[True]*6, carrierX=[20.0, 20.0, 20.0, 20.0, 20.0, 80.0])
        scale = {'lengthM': 105.0, 'widthM': 68.0}
        l1 = {'pressureRule': 'distance_or_go_to_ball', 'coverRequired': False, 'carrierZone': 'own_half',
              'unpressuredZone': 'own_half'}
        self.assertEqual(CX.sample_outcomes(o, l1, scale)[0], [1, 0, 1, 1, -1, None])   # goes to the ball at 7 m
        l4 = {'pressureRule': 'distance', 'coverRequired': True, 'carrierZone': 'any', 'unpressuredZone': 'own_third'}
        outs, modes = CX.sample_outcomes(o, l4, scale)
        self.assertEqual(outs, [1, 0, -1, -1, -1, -1])
        self.assertEqual(modes, [None, None, 'no_cover', 'no_pressure', 'no_pressure', 'no_cover'])
        self.assertEqual([CX.pressure_cover_flags(o['relations'], k, l4, scale)[0] for k in range(6)],
                         [True, False, True, False, False, True])
        hp = dict(l4, coverExemptZone='final_third')
        self.assertEqual(CX.sample_outcomes(o, hp, scale)[0][5], 0)                 # high press: no cover counted
        hidden = copy.deepcopy(o)
        hidden['relations']['coverVisible'][5] = False
        self.assertEqual(CX.sample_outcomes(hidden, l4, scale)[0][5], 0)            # cover not in view: unclear
        # the two-state reading of before (pressed anywhere within 5 m) is the certain distance set to 5 m
        self.assertEqual(CX.sample_outcomes(o, dict(l4, pressureCertainM=5.0), scale)[0][1], -1)
        # a closing speed nobody reaches: 3-5 m is always unclear
        self.assertEqual(CX.sample_outcomes(o, dict(l4, pressureClosingMS=99.0), scale)[0][2], 0)

    def test_cover_needs_the_angle(self):
        """S-15 (CODE M4 survivor 'cover_no_angle'): a team-mate inside the radius but 90 deg off the line to goal
        is no cover, under a profile that narrows the angle and on a hand-built relation alike."""
        o = self.opp('pressure_cover', d1=[2.0, 2.0], closingMS=[0.0, 0.0], coverDist=[5.0, 5.0],
                     coverAngle=[90.0, 40.0], coverVisible=[True, True], carrierX=[20.0, 20.0])
        scale = {'lengthM': 105.0, 'widthM': 68.0}
        l4 = {'pressureRule': 'distance', 'coverRequired': True, 'carrierZone': 'any', 'unpressuredZone': 'own_third'}
        self.assertEqual([CX.pressure_cover_flags(o['relations'], k, l4, scale)[1] for k in range(2)], [False, True])
        self.assertEqual(CX.sample_outcomes(o, l4, scale)[0], [-1, 1])
        self.assertEqual(CX.sample_outcomes(o, dict(l4, coverAngleDeg=30.0), scale)[0], [-1, -1])

    def test_wide_defender_rules_by_level(self):
        o = self.opp('wide_defender_advance', advanceM=[20.0, 20.0, 20.0, 10.0], beyondHalfM=[5.0, 5.0, 5.0, 0.0],
                     channelOccupied=[True, False, False, False], cbAdjusted=[None, False, False, False],
                     interchange=[False, False, False, False], dmDropped=[None, True, None, None],
                     opponentsInChannel=[0, 0, 2, 0], behindN=[1, 0, 1, 0], wdVelX=[0.0, 0.0, -1.0, 0.0],
                     ballX=[80.0, 80.0, 80.0, 80.0])
        scale = {'lengthM': 105.0, 'widthM': 68.0}
        self.assertEqual(CX.sample_outcomes(o, {'countOnly': 'uncovered'}, scale)[0], [1, -1, -1, None])
        self.assertEqual(CX.sample_outcomes(o, {'countOnly': 'exposed'}, scale)[0], [1, 0, -1, None])
        self.assertEqual(CX.sample_outcomes(o, {'useDmRotation': True}, scale)[0][1], 1)
        # level 2 (fix round 3, critic B-1): covered +1, exposed (an opponent in the vacated channel) -1, nobody
        # behind the ball in the final third -1, else 0
        self.assertEqual(CX.sample_outcomes(o, {'requireNobodyBehind': True}, scale)[0], [1, -1, -1, None])
        self.assertEqual(CX.sample_outcomes(o, {'recoveringMS': .5}, scale)[0][2], None)
        self.assertEqual(CX.sample_outcomes(o, {'advancedRule': 'halfway'}, scale)[0], [None]*4)   # 5 m < .1 L
        # ... outside the final third (build-up): covered +1 and exposed -1 wherever the ball is, the uncovered advance
        # with nobody of theirs in the channel unclear; nobody behind the ball is judged only in the final third (the
        # round-1 B-1 fix); an exposed sample needs no ball depth
        mid = copy.deepcopy(o)
        mid['relations']['ballX'] = [50.0, 50.0, None, 50.0]
        self.assertEqual(CX.sample_outcomes(mid, {'requireNobodyBehind': True}, scale)[0], [1, 0, -1, None])
        self.assertEqual(CX.sample_outcomes(mid, {'countOnly': 'uncovered'}, scale)[0], [1, -1, -1, None])
        nodepth = copy.deepcopy(mid)
        nodepth['relations']['opponentsInChannel'] = [0, 0, 0, 0]
        self.assertEqual(CX.sample_outcomes(nodepth, {'requireNobodyBehind': True}, scale)[0], [1, 0, None, None])
        old = copy.deepcopy(o)                      # a roles file from before fix round 1: never judged at level 2
        del old['relations']['ballX']
        self.assertEqual(CX.sample_outcomes(old, {'requireNobodyBehind': True}, scale)[0], [None]*4)

    def test_centre_back_shift_and_interchange_cover(self):
        """Test (c) on the measured shift (cbShiftM acts, S-14) and test (e) only with the slot re-occupied (S-4);
        the CB-shift-only cover decides on its own (S-15, CODE M4 survivor 'd3_cb_adjust_not_cover')."""
        o = self.opp('wide_defender_advance', advanceM=[20.0]*4, beyondHalfM=[5.0]*4,
                     channelOccupied=[False]*4, cbAdjusted=[True, False, False, None], nearCbShiftM=[5.0, 3.0, 3.0, None],
                     interchange=[False, True, True, False], interchangeCover=[False, False, True, False],
                     dmDropped=[None]*4, opponentsInChannel=[0]*4, behindN=[0]*4, wdVelX=[0.0]*4)
        scale = {'lengthM': 105.0, 'widthM': 68.0}
        self.assertEqual(CX.sample_outcomes(o, {'countOnly': 'uncovered'}, scale)[0], [1, -1, 1, -1])
        self.assertEqual(CX.sample_outcomes(o, {'countOnly': 'uncovered', 'cbShiftM': 2.5}, scale)[0], [1, 1, 1, -1])
        self.assertEqual(CX.sample_outcomes(o, {'countOnly': 'uncovered', 'cbShiftM': 6.0}, scale)[0], [-1, -1, 1, -1])
        old = copy.deepcopy(o)                      # a relation set without the measured shift: the stored flag
        del old['relations']['nearCbShiftM'], old['relations']['interchangeCover']
        self.assertEqual(CX.sample_outcomes(old, {'countOnly': 'uncovered'}, scale)[0], [1, -1, -1, -1])

    def test_level_one_reframe_final_third_and_cover(self):
        """Level 1 reads an advance with the behind_ball rule (critic B-1, N9; fix round 3, critic B-1 / COACH F-1): a
        covered advance is protected (+1) and an uncovered one with an opponent in the vacated channel is not (-1)
        wherever the ball is; otherwise the behind-the-ball count in the final third and unclear (0) outside it; a
        sample of an older roles file (no ball depth) is out of scope."""
        o = self.opp('wide_defender_advance', advanceM=[20.0]*6, beyondHalfM=[5.0]*6,
                     channelOccupied=[False, True, False, False, False, False],
                     cbAdjusted=[False, False, True, False, False, False],
                     interchange=[False]*6, interchangeCover=[False]*6, dmDropped=[None]*6,
                     opponentsInChannel=[0, 0, 0, 0, 0, 1], behindN=[0, 0, 0, 0, 2, 3], wdVelX=[0.0]*6,
                     restMargin=[None]*6, ballX=[80.0, 80.0, 80.0, 50.0, 80.0, 50.0])
        scale = {'lengthM': 105.0, 'widthM': 68.0}
        l1 = {'behindRule': 'count', 'minBehind': 1}
        self.assertEqual(CX.sample_outcomes(o, l1, scale, as_detector='behind_ball')[0], [-1, 1, 1, 0, 1, -1])
        mid = copy.deepcopy(o)                      # build-up: covered +1 and exposed -1 still, nothing else judged
        mid['relations']['ballX'] = [50.0]*6
        self.assertEqual(CX.sample_outcomes(mid, l1, scale, as_detector='behind_ball')[0], [0, 1, 1, 0, 0, -1])
        old = copy.deepcopy(o)
        del old['relations']['ballX']               # a roles file from before the fix: never judged
        self.assertEqual(CX.sample_outcomes(old, l1, scale, as_detector='behind_ball')[0], [None]*6)

    def test_far_side_level_tolerance_and_own_box(self):
        """D2 (critic B-5): a far-half defender level with the ball makes the sample unclear, one running back too;
        the ball within 20 m (x L/105) of the own goal line is not judged; without the tolerant relations the plain
        count decides."""
        o = self.opp('weak_side_balance', weakCount=[0, 0, 0, 0, 1], weakGoalCount=[0, 0, 0, 1, 1],
                     weakLevelCount=[1, 0, 0, 0, 0], weakRecovering=[0, 1, 0, 0, 0],
                     ballX=[40.0, 40.0, 40.0, 40.0, 15.0])
        scale = {'lengthM': 105.0, 'widthM': 68.0}
        self.assertEqual(CX.sample_outcomes(o, None, scale)[0], [0, 0, -1, 1, None])
        self.assertEqual(CX.sample_outcomes(o, {'excludeOwnGoalM': 10.0}, scale)[0][4], 1)
        small = {'lengthM': 52.5, 'widthM': 34.0}  # 20 m x L/105 = 10 m on a half-size field
        self.assertEqual(CX.sample_outcomes(o, None, small)[0][4], 1)
        old = copy.deepcopy(o)
        for k in ('weakGoalCount', 'weakLevelCount', 'weakRecovering'):
            del old['relations'][k]
        self.assertEqual(CX.sample_outcomes(old, None, scale)[0], [-1, -1, -1, -1, None])

    def test_spacing_and_shape(self):
        o = self.opp('defensive_spacing', lateralGapM=[25.0, 25.0, 12.0, 12.0, 12.0],
                     gapBallSide=[True, False, True, True, True], opponentInGap=[True, True, False, False, False],
                     dmGapM=[10.0, 10.0, 20.0, 11.0, 20.0], lineStdM=[1.0]*5, dmGapOpponents=[0, 0, 1, 0, 2],
                     dmGapFree=[0, 0, 1, 0, 0])
        scale = {'lengthM': 105.0, 'widthM': 68.0}
        outs, modes = CX.sample_outcomes(o, None, scale)
        # the D-M gap fails only with a FREE opponent between the lines on the ball side in reach of the ball (critic
        # B-7): the last sample has two between the lines, both marked
        self.assertEqual(outs, [-1, 1, -1, 1, 0])
        self.assertEqual(modes, ['lateral_gap', None, 'dm_gap', None, None])
        old = copy.deepcopy(o)
        del old['relations']['dmGapOpponents'], old['relations']['dmGapFree']
        self.assertEqual(CX.sample_outcomes(old, None, scale)[0], [-1, 1, 0, 1, 0])
        o = self.opp('attacking_shape', widthShare=[.6, .4, .6, .6], depthShare=[.35, .35, .2, .35],
                     channelsOccupied=[3, 3, 3, 2])
        self.assertEqual(CX.sample_outcomes(o, None, scale)[0], [1, -1, -1, -1])     # the last: a channel empty
        self.assertEqual(CX.sample_outcomes(o, None, scale)[1], [None, 'width', 'depth', 'channels'])
        self.assertEqual(CX.sample_outcomes(o, {'judgeWidth': False}, scale)[0], [1, 1, -1, -1])

    def test_depth_against_the_space_available(self):
        """D6 (critic B-7): a field share below depthShare fails only when the team also uses less than
        depthAvailShare of the length from its deepest outfield player to the opponents' last line; on 11v11 unclear
        where that cannot be measured, on small formats the field share decides there."""
        o = self.opp('attacking_shape', widthShare=[.8, .8, .8, .4, .8], depthShare=[.25, .25, .25, .25, .35],
                     channelsOccupied=[3]*5, depthAvailShare=[.95, .5, None, .95, .3])
        scale = {'lengthM': 105.0, 'widthM': 68.0}
        rule = {'depthRule': 'available', 'depthAvailShare': .75}
        outs, modes = CX.sample_outcomes(o, rule, scale)
        self.assertEqual(outs, [1, -1, 0, -1, 1])
        self.assertEqual(modes, [None, 'depth', None, 'width', None])
        self.assertEqual(CX.sample_outcomes(o, dict(rule, depthRule='available_or_field'), scale)[0], [1, -1, -1, -1, 1])
        self.assertEqual(CX.sample_outcomes(o, dict(rule, depthRule='field'), scale)[0], [-1, -1, -1, -1, 1])
        self.assertEqual(CX.sample_outcomes(o, dict(rule, depthAvailShare=.4), scale)[0][1], 1)   # the param acts

    def test_key_sample_is_inside_the_longest_failing_run(self):
        o = self.opp('weak_side_balance', weakCount=[1, 0, 1, 0, 0, 0, 1])
        self.assertEqual(CX.key_sample(o, {'lengthM': 105.0, 'widthM': 68.0}), 4)
        o = self.opp('weak_side_balance', weakCount=[1, 1, 1])
        self.assertEqual(CX.key_sample(o, {'lengthM': 105.0, 'widthM': 68.0}), 1)


if __name__ == '__main__':
    unittest.main()
