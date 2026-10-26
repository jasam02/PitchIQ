"""Tests that protect the values the game-3 freeze records (critic round 2, S-5; CODE round-2 L-1): each one goes
through the SHIPPED profile (profiles.resolve, so profiles.json as it is), not REFERENCE_PARAMS, so an edit of a frozen
number or the removal of a frozen guard fails here.

CODE's round-2 mutation pass found four rules no test protected (verify2/reports/code_regression_browser.md, L-1):
- j_pressure_closing_0 (profiles.json pressureClosingMS 0.0): a 0.1 m/s approach at 3-5 m became "pressed"; the three-
  state test of test_coaching_assess passes closing speeds of -1.0 and 1.5 only. Here: 4 m closing at 0.5 m/s stays
  unclear at level 4, at 1.2 m/s it is pressed (without cover: a failure), within 3 m it is pressed at any speed.
- o_drill_rule_not_excluded (assess.focus_exclusions without drills.demonstrated_checks): a check whose reported
  findings show only demonstrated situations while unreported families of it failed must not be the level's focus.
- mw_position_low_ok (the position-quality half of assess._clear_moments' opportunity gate): a high-identity,
  low-position failing opportunity is never a clear moment (0 of the 289 stored opportunities of round 2 had one).
- o_misfit_* (assess.format_misfit at its two call sites in level_focus): never fired on the library (0 of the
  FOCUS_GUIDANCE / GENERAL_OBJECTIVES items name a format); a synthetic "9v9" item for an 11v11 team keeps it alive.
Fix round 3's new values are pinned the same way, through the shipped profile at the levels the rules use: the
settled-play gates (momentsSettleS 2 s, momentsContestS 1 s), the D5 line (lineMinPlayers 4, lineStdMaxM 4 m) and the
marked opponent (markedM 4 m), the base-rate test (baseTestAlpha .05; the level-4 D5 base .048: 3 of 11 is above it,
tail .0136, 2 of 11 is not, tail .0950) and the praise base scale below level 5 (baseSuccessScale .5 / .6 / .7 / .8),
and the moments trigger's integers (momentsFromLevel 3, momentsMinSpells 3, momentsMaxPerReport 1), each checked on
ASSESS's implementation and on EVAL's independent re-implementation (tools/coaching_eval.py) where one exists.

Run:  python -m unittest test_coaching_frozen_values        (from this directory; about 2 s)
"""
import copy
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for p in (str(ROOT), str(HERE), str(ROOT/'tools')):
    if p not in sys.path:
        sys.path.insert(0, p)
import coaching_eval as E  # noqa: E402
from soccer.coaching import assess as AS, contextual as CX, profiles as PR, schema as S  # noqa: E402

STEP = .2
SCALE = {'lengthM': 105.0, 'widthM': 68.0, 'source': 'assumed', 'metresUsable': True}


def profile(level, age=None, players=11):
    return PR.resolve({'playingLevel': level, 'ageGroup': age or PR.default_age(level),
                       'format': {'players': players, 'goalkeeper': True}}, team='A')


def d1_opp(n=15, position='high', **rel):
    """A pressure-and-cover opportunity of n samples with constant relations (the carrier in our own third)."""
    base = {'d1': 4.0, 'closingMS': 0.0, 'coverDist': 18.0, 'coverAngle': 30.0, 'coverVisible': True,
            'coverId': 'A-06', 'carrierX': 20.0, 'goalSideShare': .7, 'primaryNamed': True, 'fd': 'A-03'}
    base.update(rel)
    t = [round(100.0+STEP*k, 3) for k in range(n)]
    key = S.opportunity_key('pressure_cover', 'A', t[0], t[-1], None)
    return {'id': S.record_id('opp', key), 'key': key, 'detector': 'pressure_cover', 'team': 'A',
            'tier': 'measurement', 't0': t[0], 't1': t[-1], 'frame0': 2500, 'frame1': 2500+n*5,
            'phase': {'state': 'opp', 'source': 'possession-4', 'estimated': True, 'spanSeconds': 12.0,
                      'nearChange': False},
            'ball': {'zone': 'own_third', 'channel': 'left', 'reliable': True},
            'actors': {'primary': ['A-03'], 'supporting': [], 'opponents': [], 'carrier': None}, 'roles': {},
            'relations': {'t': t, **{k: [v]*n for k, v in base.items()}},
            'quality': {'identity': 'high', 'position': position, 'visibility': 'complete', 'ball': 'reliable',
                        'samples': n},
            'intentSignals': {'recovering': False, 'interchange': None, 'formationChangeNear': False}, 'abstain': None}


def moment(o, k=7):
    t = o['relations']['t'][k]
    return {'id': S.record_id('ev', o['id'], k), 'opportunity': o['id'], 'sample': k, 't': t, 'frame': int(t*25)}


class PressureClosing(unittest.TestCase):
    """CODE L-1, j_pressure_closing_0: the shipped pressureClosingMS (1.0 m/s) through the level-4 profile."""

    def outcome(self, **rel):
        p = profile(4)['detectors']['pressure_cover']['params']
        outs, modes = CX.sample_outcomes(d1_opp(**rel), p, SCALE)
        return outs[0], modes[0]

    def test_a_slow_approach_at_four_metres_stays_unclear(self):
        self.assertEqual(self.outcome(d1=4.0, closingMS=0.5), (0, None))
        self.assertEqual(self.outcome(d1=4.0, closingMS=0.1), (0, None))

    def test_closing_at_the_frozen_speed_presses(self):
        self.assertEqual(self.outcome(d1=4.0, closingMS=1.2), (-1, 'no_cover'))
        self.assertEqual(self.outcome(d1=2.9, closingMS=-2.0), (-1, 'no_cover'))      # within 3 m: pressed anyway
        self.assertEqual(self.outcome(d1=4.0, closingMS=1.2, coverDist=4.0, coverAngle=10.0), (1, None))

    def test_the_frozen_numbers(self):
        p = profile(4)['detectors']['pressure_cover']['params']
        self.assertEqual((p['pressureM'], p['pressureCertainM'], p['pressureClosingMS']), (5.0, 3.0, 1.0))


class FocusExclusion(unittest.TestCase):
    """CODE L-1, o_drill_rule_not_excluded: assess.focus_exclusions' use of drills.demonstrated_checks."""

    def test_a_check_its_reported_findings_show_demonstrated_is_not_the_focus(self):
        prof = profile(4)
        d = 'pressure_cover'
        # the reported finding (a description) counts only demonstrated situations; families that were not reported
        # failed, so the shown counts alone (3 of 9 demonstrated, under the strength share) would not exclude it
        f = {'observation': d, 'verdict': 'describe', 'aggregate': {'demonstrated': 3, 'notDemonstrated': 0}}
        shown = {d: (3, 6)}
        self.assertLess(3/9, prof['evidence']['strengthShare'])
        self.assertNotIn(d, AS.focus_exclusions([], prof, shown))
        self.assertIn(d, AS.focus_exclusions([f], prof, shown))
        focus = AS.level_focus(prof, AS.focus_exclusions([f], prof, shown))
        self.assertIsNotNone(focus)
        self.assertNotEqual(S.SKILLS.get(focus['skill']), d)


class ClearMomentPosition(unittest.TestCase):
    """CODE L-1, mw_position_low_ok: a high-identity, LOW-position failing opportunity is no clear moment."""

    def clear(self, position):
        p = profile(4)['detectors']['pressure_cover']['params']
        o = d1_opp(position=position, d1=2.0)
        c = AS.classify(o, p, SCALE, STEP)
        self.assertEqual(c['outcome'], 'not_demonstrated')
        got = AS._clear_moments('pressure_cover', o, c, {o['id']: [moment(o)]}, None, 11, STEP, p, settle=None)
        mine = E._clear_opportunity(o, E.D1, 'opp', 11)
        return bool(got), mine

    def test_position_low_is_never_clear(self):
        self.assertEqual(self.clear('high'), (True, True))
        self.assertEqual(self.clear('medium'), (True, True))
        self.assertEqual(self.clear('low'), (False, False))


class FormatMisfit(unittest.TestCase):
    """CODE L-1, o_misfit_*: a guidance item naming another match format is never the team's focus (both call
    sites of assess.format_misfit in level_focus: the check-tied items and the general objectives)."""

    def test_the_function(self):
        prof = profile(4)
        self.assertEqual(AS.format_misfit('Play 9v9 with two lines.', prof), '9v9')
        self.assertEqual(AS.format_misfit('A 1-3-2-1 shape.', prof), '1-3-2-1')
        self.assertIsNone(AS.format_misfit('Play 11v11 in a 1-4-4-2.', prof))

    def test_a_check_item_naming_9v9_is_skipped(self):
        prof = profile(4)
        first = AS.level_focus(prof)
        self.assertIsNotNone(first)
        self.assertEqual(first['tag'], 'FEDERATION')
        bad = copy.deepcopy(prof)
        src = first['sources'][0]
        hit = [g for g in bad['principles'] if isinstance(g, dict) and (g.get('sources') or [None])[0] == src]
        self.assertTrue(hit)
        for g in hit:
            g['text'] = g['text'].rstrip('.')+' (as played 9v9).'
        again = AS.level_focus(bad)
        self.assertTrue(again is None or '9v9' not in again['text'], again)
        self.assertNotEqual((again or {}).get('sources'), first['sources'])

    def test_a_general_objective_naming_9v9_is_skipped(self):
        prof = profile(3)
        everything = set(S.DETECTORS)
        first = AS.level_focus(prof, everything)
        self.assertIsNotNone(first)
        self.assertIsNone(first['skill'])
        bad = copy.deepcopy(prof)
        src = first['sources'][0]
        for g in bad['objectives']:
            if isinstance(g, dict) and (g.get('sources') or [None])[0] == src and \
                    first['text'].find(str(g.get('text') or '').strip()[:30]) >= 0:
                g['text'] = g['text'].rstrip('.')+' in 9v9.'
        again = AS.level_focus(bad, everything)
        self.assertTrue(again is None or '9v9' not in again['text'], again)


class FixRound3Values(unittest.TestCase):
    """The fix-round-3 values through the shipped profile, on assess and on EVAL's re-implementation."""

    def test_the_settled_play_gate(self):
        p = profile(4)['detectors']['defensive_spacing']['params']
        self.assertEqual((p['momentsSettleS'], p['momentsContestS']), (2.0, 1.0))
        tl = {'step': STEP, 'phaseSpans': [{'t0': 0.0, 't1': 49.8, 'state': 'opp'},
                                           {'t0': 50.0, 't1': 60.0, 'state': 'dead'},
                                           {'t0': 60.2, 't1': 120.0, 'state': 'opp'}]}
        idx, mine = AS.settle_index(tl), E.unsettled_intervals(tl, p['momentsSettleS'])
        for t, ok in ((55.0, False), (61.9, False), (62.1, True), (40.0, True), (49.0, True)):
            self.assertEqual(AS.settled(idx, t, p['momentsSettleS']), ok, t)
            self.assertEqual(not E._unsettled(mine, t), ok, t)
        gap = {'step': STEP, 'phaseSpans': [{'t0': 0.0, 't1': 49.8, 'state': 'opp'},
                                            {'t0': 52.0, 't1': 120.0, 'state': 'opp'}]}
        idx, mine = AS.settle_index(gap), E.unsettled_intervals(gap, p['momentsSettleS'])
        for t, ok in ((47.7, True), (48.1, False), (51.0, False), (53.9, False), (54.1, True)):
            self.assertEqual(AS.settled(idx, t, p['momentsSettleS']), ok, t)
            self.assertEqual(not E._unsettled(mine, t), ok, t)

    def test_the_line_contest_and_marking_gates(self):
        p = profile(4)['detectors']['defensive_spacing']['params']
        self.assertEqual((p['lineMinPlayers'], p['lineStdMaxM'], p['markedM']), (4, 4.0, 4.0))
        o = {'relations': {'t': [10.0], 'contestS': [1.1], 'lineSize': [4], 'lineStdM': [3.9], 'gapOppFreeM': [4.1]}}
        self.assertTrue(AS._settled_sample('defensive_spacing', o, 0, 'lateral_gap', 11, p, None))
        self.assertTrue(E._sample_gates(E.D5, o['relations'], 0, 'lateral_gap', p, None, 11))
        for key, value in (('contestS', .9), ('lineSize', 3), ('lineStdM', 4.1), ('gapOppFreeM', 3.9)):
            o2 = {'relations': dict(o['relations'], **{key: [value]})}
            self.assertFalse(AS._settled_sample('defensive_spacing', o2, 0, 'lateral_gap', 11, p, None), key)
            self.assertFalse(E._sample_gates(E.D5, o2['relations'], 0, 'lateral_gap', p, None, 11), key)

    def test_the_base_rate_test(self):
        p4 = profile(4)['detectors']['defensive_spacing']['params']
        self.assertEqual(p4['baseTestAlpha'], .05)
        self.assertAlmostEqual(p4['baseFailShare'], .048, places=3)
        self.assertTrue(AS.above_base(3, 11, p4['baseFailShare'], p4['baseTestAlpha']))
        self.assertFalse(AS.above_base(2, 11, p4['baseFailShare'], p4['baseTestAlpha']))
        self.assertAlmostEqual(E.binomial_tail(3, 11, .048), .0136, places=4)
        self.assertAlmostEqual(E.binomial_tail(2, 11, .048), .0950, places=4)

    def test_the_praise_base_below_level_5(self):
        for lv, k in ((1, .5), (2, .6), (3, .7), (4, .8), (5, 1.0), (6, 1.0), (7, 1.0)):
            for d, dp in profile(lv)['detectors'].items():
                p = dp['params']
                if not dp['enabled'] or p.get('baseFailShare') is None:
                    continue
                self.assertAlmostEqual(AS.praise_base(p), 1.0-(1.0-float(p['baseFailShare']))*k, places=9, msg=(lv, d))

    def test_the_moments_integers(self):
        for lv in (1, 2, 3, 4, 5, 6, 7):
            for d, dp in profile(lv)['detectors'].items():
                if dp['enabled'] and dp['params'].get('momentsMinSpells') is not None:
                    p = dp['params']
                    self.assertEqual((p['momentsFromLevel'], p['momentsMinSpells'], p['momentsMaxPerReport']),
                                     (3, 3, 1), (lv, d))


if __name__ == '__main__':
    unittest.main()
