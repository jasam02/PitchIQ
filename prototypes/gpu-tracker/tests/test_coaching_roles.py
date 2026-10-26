"""The role book (soccer/coaching/roles.py, DESIGN v2 3.4): a role is a claim with a source and a confidence, never a
label read off an identity: the coach's confirmation first (with a contradiction flag), then a stable period role,
a role read when defending carried into own possession, a window role, relative slots for small formats; interchanges
only suppress; the measured formation is checked against the coach's plan.

Why these checks. The feasibility study measured that a possession team's full-back reads as a winger (0 of 229 s of
Barcelona's possession in a confident back four) and that holders of a role change on 4-25% of window steps, so each
rule that decides how far a criticism may lean on a role is tested on hand-made role inputs (stability share and
holder changes, the carried role and its 120 s bound, coach spans cutting derived roles, the lateral contradiction,
relative slots, their separation, interchanges held 3 s) and on EVAL's synthetic scenes against their planted roles:
the 4-3-3 'professional' scene and the 4-4-2 scenes read every outfield role right as 'stable' (high), the 7v7
scenes every relative slot right (medium), the 4v4 swarm only at low confidence.
Each of 37 rules of situations.py and roles.py was undone once in a scratch copy (coaching2/build/context_w2/mutate.py)
and every mutation made a test of this file or its sibling fail.

Run:  python -m unittest test_coaching_roles          (from this directory)
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
import soccer.coaching as coaching  # noqa: E402
from soccer import analytics, possession, team_shape as ts  # noqa: E402
from soccer.coaching import roles as RO, schema as S, series, situations as SI  # noqa: E402

IDS = [f'A-{k:02d}' for k in range(1, 11)]
R442 = dict(zip(IDS, ('LB', 'LCB', 'RCB', 'RB', 'LM', 'LCM', 'RCM', 'RM', 'LCF', 'RCF')))


def window(c, roles, period=0, label='4-4-2', own=0.0, opp=0.0, alt=None):
    return {'centre': float(c), 't0': float(c)-7.5, 't1': float(c)+7.5, 'period': period, 'label': label,
            'alternative': None, 'roles': dict(roles), 'alt': dict(alt or {}), 'own': own, 'opp': opp, 'samples': 75}


def period(index, t0, t1, label='4-4-2', confidence='likely', related=False):
    return {'index': index, 't0': t0, 't1': t1, 'label': label, 'rawLabel': label, 'confidence': confidence,
            'corrected': False, 'alternative': None, 'related': related, 'margin': .1, 'roleStability': .8}


def context(windows=(), periods=(), ids=IDS, track=None, fmt='11v11', outfield=10, seg=(0.0, 100.0), width=68.0):
    tr = track or {'t': [], 'x': [[] for _ in ids], 'y': [[] for _ in ids], 'phase': []}
    tr = dict(tr, step=1.0, complete=[True]*len(tr['t']))
    return {'version': SI.CONTEXT_VERSION, 'team': 'A', 'format': fmt, 'outfield': outfield,
            'scale': {'lengthM': 105.0, 'widthM': width, 'source': 'assumed', 'metresUsable': True}, 'ids': list(ids),
            'segment': {'t0': seg[0], 't1': seg[1]}, 'track': tr, 'windows': list(windows), 'periods': list(periods)}


def obs(ctx, players=None):
    return {'team': 'A', 'timeline': {'format': ctx['format'], 't0': ctx['segment']['t0'], 't1': ctx['segment']['t1'],
                                      'players': players or {}}, 'roleInputs': {'context': ctx}}


def book(ctx, roles=None, intended=None, periods=(), players=None):
    out = RO.build_rolebook(obs(ctx, players), {'roles': roles or {}, 'formation': {'intended': intended,
                                                                                    'periods': list(periods)}})
    assert S.validate('rolebook', out) == [], S.validate('rolebook', out)
    return out


def regular_windows(n=19, roles=R442, **kw):
    return [window(7.5+5*k, roles, **kw) for k in range(n)]


class Names(unittest.TestCase):
    def test_groups_and_sides(self):
        self.assertEqual([RO.role_group(r) for r in ('LB', 'LWB', 'CDM', 'CAM', 'RW', 'GK', 'back-2/3', 'front-1/1')],
                         ['D', 'D', 'M', 'M', 'F', 'GK', 'D', 'F'])
        self.assertEqual([RO.role_side(r) for r in ('LB', 'RCM', 'CB', 'CF', 'back-1/3', 'back-2/3', 'back-3/3',
                                                    'mid-1/1', 'GK')], ['L', 'R', 'C', 'C', 'L', 'C', 'R', 'C', 'C'])
        self.assertIsNone(RO.role_group(None))
        self.assertEqual(RO.blocks_11('4-2-3-1'), (4, 5, 1))
        self.assertIsNone(RO.blocks_11('2-2-2'))


class Stable(unittest.TestCase):
    def test_a_likely_period_gives_medium_stable_roles(self):
        b = book(context(regular_windows(), [period(0, 0.0, 100.0)]))
        self.assertEqual(b['model'], 'team_shape')
        self.assertEqual(len(b['records']), 10)
        for r in b['records']:
            self.assertEqual((r['source'], r['confidence'], r['role'], r['t0'], r['t1'], r['support']),
                             ('stable', 'medium', R442[r['id']], 0.0, 100.0, 1.0))
        clear = book(context(regular_windows(), [period(0, 0.0, 100.0, confidence='clear')]))
        self.assertTrue(all(r['confidence'] == 'high' for r in clear['records']))

    def test_an_ambiguous_period_gives_window_roles_only(self):
        b = book(context(regular_windows(), [period(0, 0.0, 100.0, confidence='ambiguous')]))
        self.assertEqual({r['source'] for r in b['records']}, {'window'})
        self.assertEqual({r['confidence'] for r in b['records']}, {'low'})

    def test_the_share_and_the_holder_change_rules(self):
        wins = regular_windows()
        for k in (3, 9, 15, 17, 18, 19):           # A-05 is LCM on 6 of 19 windows: share .68 < .7
            if k < len(wins):
                wins[k]['roles'] = dict(wins[k]['roles'], **{'A-05': 'LCM', 'A-06': 'LM'})
        b = book(context(wins, [period(0, 0.0, 100.0)]))
        src = {r['id']: r['source'] for r in b['records'] if r['t0'] == 0.0}
        self.assertEqual(src['A-05'], 'window')
        self.assertEqual(src['A-06'], 'window')     # LM changes holder on 9 of 18 steps
        self.assertEqual(src['A-01'], 'stable')
        wins = regular_windows()
        for k in (4, 8, 12):                        # the LB and the LCB swap on three windows: LB changes on 6 of 18
            wins[k]['roles'] = dict(wins[k]['roles'], **{'A-01': 'LCB', 'A-02': 'LB'})
        b = book(context(wins, [period(0, 0.0, 100.0)]))
        lb = [r for r in b['records'] if r['id'] == 'A-01']
        self.assertTrue(all(r['source'] == 'window' for r in lb))
        self.assertTrue(b['interchanges'])
        self.assertTrue(all(i['slot'] in ('LB', 'LCB') for i in b['interchanges']))

    def test_the_share_rule_alone(self):
        wins = regular_windows()
        for k in range(12, 19):                     # A-05 reads CAM (nobody else's role) on 7 of 19 windows: share .63
            wins[k]['roles'] = dict(wins[k]['roles'], **{'A-05': 'CAM'})
        b = book(context(wins, [period(0, 0.0, 100.0)]))
        a05 = [r for r in b['records'] if r['id'] == 'A-05']
        self.assertTrue(all(r['source'] == 'window' for r in a05))      # LM changes holder on 1 of 12 steps only
        self.assertEqual({r['source'] for r in b['records'] if r['id'] == 'A-06'}, {'stable'})

    def test_related_periods_name_the_alternative(self):
        alt = dict(R442, **{'A-01': 'LWB'})
        wins = [window(7.5+5*k, R442, alt=alt) for k in range(19)]
        b = book(context(wins, [period(0, 0.0, 100.0, related=True)]))
        lb = next(r for r in b['records'] if r['id'] == 'A-01')
        self.assertEqual(lb['ambiguousWith'], ['LWB'])


class Carried(unittest.TestCase):
    def test_the_role_read_when_defending_is_carried_into_own_possession(self):
        attack = dict(R442, **{'A-01': 'LM', 'A-05': 'LB'})       # in possession the left-back reads as a winger
        wins = []
        for k in range(19):
            c = 7.5+5*k
            if k < 6:
                wins.append(window(c, R442, opp=.9))
            else:
                wins.append(window(c, attack, own=.9))
        b = book(context(wins, [period(0, 0.0, 100.0, confidence='ambiguous')]))
        idx = RO.RoleIndex(b)
        r = idx.at('A-01', 60.0)
        self.assertEqual((r['role'], r['source'], r['confidence']), ('LB', 'carried', 'medium'))
        self.assertEqual(idx.at('A-01', 10.0)['source'], 'window')        # read while defending: a window role
        self.assertEqual(idx.at('A-05', 60.0)['role'], 'LM')

    def test_not_carried_beyond_120_s_or_into_another_period(self):
        attack = dict(R442, **{'A-01': 'LM', 'A-05': 'LB'})
        wins = [window(7.5+5*k, R442 if k < 3 else attack, opp=.9 if k < 3 else 0.0, own=0.0 if k < 3 else .9)
                for k in range(40)]
        b = book(context(wins, [period(0, 0.0, 205.0, confidence='ambiguous')], seg=(0.0, 205.0)))
        idx = RO.RoleIndex(b)
        self.assertEqual(idx.at('A-01', 100.0)['source'], 'carried')
        self.assertEqual(idx.at('A-01', 160.0)['source'], 'window')       # 150 s from the last defending window
        wins = [dict(w, period=1 if w['centre'] > 50 else 0) for w in wins[:19]]
        b = book(context(wins, [period(0, 0.0, 50.0, confidence='ambiguous'),
                                period(1, 50.0, 100.0, confidence='ambiguous')]))
        self.assertEqual(RO.RoleIndex(b).at('A-01', 80.0)['source'], 'window')


class Coach(unittest.TestCase):
    def track(self, y_of, seconds=100):
        t = [float(k) for k in range(seconds+1)]
        return {'t': t, 'x': [[30.0+j for _ in t] for j in range(10)],
                'y': [[y_of(j) for _ in t] for j in range(10)], 'phase': ['own']*len(t)}

    def test_a_coach_span_replaces_what_it_covers(self):
        ctx = context(regular_windows(), [period(0, 0.0, 100.0)])
        roles = {'A-01': [{'role': 'LWB', 'from': 20.0, 'to': 40.0, 'set': '2026-10-22T10:00:00'}],
                 'GK-1': [{'role': 'GK', 'from': None, 'to': None, 'set': '2026-10-22T10:00:00'}],
                 'B-01': [{'role': 'CB', 'from': None, 'to': None, 'set': '2026-10-22T10:00:00'}]}
        b = book(ctx, roles, players={'GK-1': {'team': 'A', 'goalkeeper': True, 'observedSeconds': 90.0,
                                               'p90SpeedMS': None}})
        a01 = [(r['t0'], r['t1'], r['role'], r['source']) for r in b['records'] if r['id'] == 'A-01']
        self.assertEqual(a01, [(0.0, 20.0, 'LB', 'stable'), (20.0, 40.0, 'LWB', 'coach'), (40.0, 100.0, 'LB', 'stable')])
        gk = [r for r in b['records'] if r['id'] == 'GK-1']
        self.assertEqual((gk[0]['role'], gk[0]['group'], gk[0]['confidence']), ('GK', 'GK', 'high'))
        self.assertFalse(any(r['id'] == 'B-01' for r in b['records']))      # another team's id is ignored

    def test_a_contradicted_confirmation_is_flagged_not_overruled(self):
        ctx = context(regular_windows(), [period(0, 0.0, 100.0)],
                      track=self.track(lambda j: 50.0 if j == 0 else 20.0))           # A-01 lives on the right
        roles = {'A-01': [{'role': 'LB', 'from': None, 'to': None, 'set': '2026-10-22T10:00:00'}],
                 'A-02': [{'role': 'CB', 'from': None, 'to': None, 'set': '2026-10-22T10:00:00'}],
                 'A-03': [{'role': 'LCB', 'from': None, 'to': None, 'set': '2026-10-22T10:00:00'}]}
        b = book(ctx, roles)
        by = {r['id']: r for r in b['records'] if r['source'] == 'coach'}
        self.assertEqual(by['A-01']['conflict'], RO.CONFLICT_CODE)
        self.assertEqual(by['A-01']['role'], 'LB')
        self.assertIsNone(by['A-02']['conflict'])               # a central role is never contradicted sideways
        self.assertIsNone(by['A-03']['conflict'])               # A-03 lives on the left
        short = context(regular_windows(), [period(0, 0.0, 100.0)], track=self.track(lambda j: 50.0, seconds=8))
        self.assertIsNone(book(short, roles)['records'][0]['conflict'])    # under 10 s seen: not judged


class Relative(unittest.TestCase):
    BLOCK = {'A-01': (10.0, 10.0), 'A-02': (10.0, 30.0), 'A-03': (10.0, 50.0), 'A-04': (25.0, 20.0),
             'A-05': (25.0, 40.0), 'A-06': (40.0, 30.0)}

    def ctx(self, swap=None, seconds=60, gap_m=None):
        ids = sorted(self.BLOCK)
        t = [float(k) for k in range(seconds+1)]
        xs, ys = [], []
        for pid in ids:
            x, y = self.BLOCK[pid]
            if gap_m is not None:
                x = {10.0: 20.0, 25.0: 20.0+gap_m, 40.0: 20.0+2*gap_m}[x]
            xs.append([x for _ in t])
            ys.append([y for _ in t])
        if swap:
            a, b2, t0, t1 = swap
            ia, ib = ids.index(a), ids.index(b2)
            for k, tt in enumerate(t):
                if t0 <= tt < t1:
                    ys[ia][k], ys[ib][k] = ys[ib][k], ys[ia][k]
        tr = {'t': t, 'x': xs, 'y': ys, 'phase': ['own']*len(t)}
        return context(ids=ids, track=tr, fmt='7v7', outfield=6, seg=(0.0, float(seconds)), width=36.0)

    def test_slots_from_depth_and_lateral_rank(self):
        b = book(self.ctx(), intended='1-3-2-1')
        self.assertEqual(b['model'], 'relative')
        idx = RO.RoleIndex(b)
        self.assertEqual({pid: idx.at(pid, 30.0)['role'] for pid in self.BLOCK},
                         {'A-01': 'back-1/3', 'A-02': 'back-2/3', 'A-03': 'back-3/3', 'A-04': 'mid-1/2',
                          'A-05': 'mid-2/2', 'A-06': 'front-1/1'})
        self.assertTrue(all(r['source'] == 'relative' and r['confidence'] == 'medium' for r in b['records']))
        self.assertEqual({r['t0'] for r in b['records']}, {0.0})
        self.assertEqual({r['t1'] for r in b['records']}, {60.0})
        self.assertEqual(b['interchanges'], [])
        self.assertEqual(b['formationChecks'], [])
        self.assertEqual(idx.at('A-01', 30.0)['side'], 'L')
        other = book(self.ctx(), intended='1-2-3-1')
        self.assertEqual(RO.RoleIndex(other).at('A-03', 30.0)['role'], 'mid-3/3')    # the right of a midfield three

    def test_close_blocks_are_low_confidence(self):
        b = book(self.ctx(gap_m=1.0), intended='1-3-2-1')
        self.assertEqual({r['confidence'] for r in b['records']}, {'low'})

    def test_an_interchange_needs_three_seconds_either_side(self):
        b = book(self.ctx(swap=('A-01', 'A-03', 20, 30)), intended='1-3-2-1')
        ins = b['interchanges']
        self.assertEqual({(i['slot'], i['out'], i['in']) for i in ins if i['t'] == 20.0},
                         {('back-1/3', 'A-01', 'A-03'), ('back-3/3', 'A-03', 'A-01')})
        self.assertTrue(any(i['t'] == 30.0 for i in ins))       # and back
        idx = RO.RoleIndex(b)
        self.assertEqual(len(idx.interchanges_near(21.0, pid='A-01')), 4)
        self.assertEqual(idx.interchanges_near(50.0), [])
        flicker = book(self.ctx(swap=('A-01', 'A-03', 20, 22)), intended='1-3-2-1')
        self.assertEqual(flicker['interchanges'], [])

    def test_an_incomplete_window_gets_no_slots(self):
        ctx = self.ctx()
        ctx['track']['x'][5] = [None]*len(ctx['track']['t'])
        b = book(ctx, intended='1-3-2-1')
        self.assertEqual(b['records'], [])
        ctx = self.ctx()
        ctx['track']['x'][5] = [v if k % 4 == 0 else None for k, v in enumerate(ctx['track']['x'][5])]   # 25% seen
        self.assertEqual(book(ctx, intended='1-3-2-1')['records'], [])


class FormationChecks(unittest.TestCase):
    def test_mismatch_only_on_a_clear_unrelated_reading(self):
        ctx = context(regular_windows(), [period(0, 0.0, 50.0, label='3-5-2', confidence='clear'),
                                          period(1, 50.0, 100.0, label='4-3-3', confidence='likely')])
        b = book(ctx, intended='4-4-2')
        self.assertEqual([(c['measured'], c['intended'], c['mismatch'], c['related']) for c in b['formationChecks']],
                         [('3-5-2', '4-4-2', True, False), ('4-3-3', '4-4-2', False, False)])
        rel = book(ctx, intended='5-3-2')
        self.assertEqual([(c['mismatch'], c['related']) for c in rel['formationChecks']], [(False, True), (False, False)])
        per = book(ctx, intended='4-4-2', periods=[{'t0': 0.0, 't1': 50.0, 'formation': '3-5-2'}])
        self.assertFalse(per['formationChecks'][0]['mismatch'])
        idx = RO.RoleIndex(b)
        self.assertEqual(idx.mismatch_at(20.0)['measured'], '3-5-2')
        self.assertIsNone(idx.mismatch_at(80.0))
        self.assertFalse(any(c['mismatch'] for c in book(ctx)['formationChecks']))      # no plan, no mismatch

    def test_an_11v11_window_without_a_template_role_takes_a_relative_slot_with_a_plan(self):
        track = {'t': [float(k) for k in range(101)],
                 'x': [[10.0+3*(j//4)+.5*(j % 4) for _ in range(101)] for j in range(10)],
                 'y': [[5.0+6*j for _ in range(101)] for j in range(10)], 'phase': ['own']*101}
        ctx = context([], [period(0, 0.0, 100.0, label=None, confidence='unclear')], track=track)
        self.assertEqual(book(ctx)['records'], [])
        b = book(ctx, intended='4-4-2')
        self.assertEqual({r['source'] for r in b['records']}, {'relative'})


def synthetic_roles(scene_name, variant=None, coach_roles=True):
    b = cs.build(scene_name, variant)
    st = cs.settings_for(b, level=4, roles=coach_roles)
    samples = analytics.Samples(b.result)
    shape = ts.TeamShape(b.result, samples=samples)
    cd = series.build_series(b.result, samples=samples, shape=shape)
    stages = {'detect': lambda tl: {'opportunities': [], 'detectors': {}, 'roleInputs': {'context': tl.role_inputs()}},
              'moments': lambda cd_, tl, o: {'moments': [], 'frameTiers': []}}
    ob = coaching.build_observations(b.result, 'A', coaching=st, series=cd, possession=possession.compute(b.result),
                                     result_sha='1'*64, stages=stages)
    role_stages = {'detect_roles': lambda o, bk: {'opportunities': [], 'detectors': {}},
                   'role_moments': lambda o, op: {'moments': []}}
    rf = coaching.build_roles(ob, st, 'A', stages=role_stages)
    return b, st, ob, rf


def agreement(b, rolebook):
    idx = RO.RoleIndex(rolebook)
    out = {}
    for pid, spans in b.truth['roles'].items():
        if not pid.startswith('A-'):
            continue
        for sp in spans:
            for t in np.arange(sp['t0']+.5, sp['t1'], 2.0):
                r = idx.at(pid, float(t))
                key = (r['source'], r['confidence'], r['role'] == sp['role']) if r else ('none', None, None)
                out[key] = out.get(key, 0)+1
    return out


class Scenes(unittest.TestCase):
    def test_a_clear_4_3_3_reads_every_role(self):
        b, st, ob, rf = synthetic_roles('professional')
        self.assertEqual(S.validate('roles', rf), [])
        self.assertEqual(set(agreement(b, rf['rolebook'])), {('stable', 'high', True)})

    def test_the_coachs_role_and_the_wrong_formation(self):
        b, st, ob, rf = synthetic_roles('wrong_formation', 'coach_roles')
        bk = rf['rolebook']
        a04 = [r for r in bk['records'] if r['id'] == 'A-04']
        self.assertEqual([(r['role'], r['source'], r['confidence'], r['conflict']) for r in a04],
                         [('LWB', 'coach', 'high', None)])
        self.assertTrue(any(c['mismatch'] for c in bk['formationChecks']))
        b2, st2, ob2, rf2 = synthetic_roles('wrong_formation', 'coach_roles', coach_roles=False)
        self.assertEqual(ob2['fingerprint'], ob['fingerprint'])                 # roles never reach the obs file
        self.assertNotEqual(rf2['fingerprint'], rf['fingerprint'])

    def test_small_formats_read_relative_slots(self):
        for scene_name, variant in (('small_field_scale', 'declared'), ('amateur_coach_spans', 'spans')):
            b, st, ob, rf = synthetic_roles(scene_name, variant)
            self.assertEqual(S.validate('roles', rf), [])
            self.assertEqual(rf['rolebook']['model'], 'relative')
            agree = agreement(b, rf['rolebook'])
            right = sum(v for k, v in agree.items() if k[2] is True)
            self.assertGreater(right/sum(agree.values()), .97, (scene_name, agree))
            self.assertTrue(all(k[0] == 'relative' for k in agree))

    def test_a_swarm_has_only_low_confidence_slots(self):
        b, st, ob, rf = synthetic_roles('kids_swarm', 'measured_size')
        confs = {r['confidence'] for r in rf['rolebook']['records']}
        self.assertEqual(confs, {'low'})


class Index(unittest.TestCase):
    def test_holders_and_precedence(self):
        b = book(context(regular_windows(), [period(0, 0.0, 100.0)]),
                 {'A-02': [{'role': 'LB', 'from': 50.0, 'to': 60.0, 'set': '2026-10-22T10:00:00'}]})
        idx = RO.RoleIndex(b)
        self.assertEqual(set(idx.holders(55.0, roles=('LB',))), {'A-01', 'A-02'})
        self.assertEqual(set(idx.holders(55.0, roles=('LB',), sources=('coach',))), {'A-02'})
        self.assertIsNone(idx.at('A-01', 500.0))
        self.assertIsNone(idx.at('A-99', 5.0))
        self.assertEqual(json.loads(json.dumps(b)), b)


if __name__ == '__main__':
    unittest.main()
