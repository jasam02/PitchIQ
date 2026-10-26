"""DESIGN 13.1 and 13.4 on the synthetic counterexamples: every scene of tools/coaching_synth.py (17 scenes, 30 variants)
through the REAL v2 engine (store.measure, the server's measurement path with the coach spans folded into the key and
the evidence anchors merged, store.build_role_tier, coaching.interpret with every real interpretation stage) at levels
1-7, every expectation the scene states checked by tools/coaching_eval.py, and the report invariants checked on every
report.

Why a separate file from test_coaching_synth.py: that one tests the generator (the scenes plant what they claim, on the
true positions); this one tests the engine against what was planted. The expectations were written in wave 1 before any
engine existed. The first wave-8 run on the full reports (Oct 22) found 15 failing checks of 395: 6 were conflicts
between DESIGN 13.1 and DESIGN rules other roles adopted (level 7 with the style unknown stays review, DESIGN 2.3/2.6;
level 1 judges D1 as "goes to the ball", DESIGN 4.1; D2/D5 are off at levels 1-2, DESIGN 2.3), settled in the scene file
with their DESIGN section; 7 were a scene that did not plant what it claimed (defending/gap_open opened the ball-side gap
for about half of each opportunity: 7 of 25 failed at level 4; it now keeps the ball on one wing per span); 2 were this
module's own over-strict reading ("the near centre-back" is not the left-back whose identity crossed). One is an open
engine defect recorded in coaching_eval.KNOWN_DEFECTS (DETECT, the level-1 reframing keeps no identity guardrail): the
test asserts it is still open, so the day it is repaired the test fails and the entry must go.

Criteria measured here (DESIGN 13.4): #3 level-1/2 caps and wording, #4 recall of planted unsupported advances at level 4
and none on the covered twins, #5 deeper reasoning at levels 5-7, #6 supported vs unsupported twins, #7 teammates behind
every work item, #8 annotations on the observed position of the key frame, #9 every zone's properties (the independent
check tests/coaching_zone_props.py), #10 declines when evidence is insufficient, #11 strengths grounded (and the mutation:
removing the demonstrations removes the strength), #12 drills and safety, #13 no generic or repeated language.

Final wave-8 run (Oct 22, `coaching_eval.py scenes`, 139 s): 380 checks pass, 0 fail, 1 known, 43 not applicable, 0
report invariants broken. #4: 6 of 6 planted scenes give a level-4 work item (16 failing spells each), the covered twins
and the professional-like scene 0 criticisms at any level. #5: relation + opponent claims of the planted finding L2 1,
L3 2, L4 3, L5-L7 4. #6: covered twins strength and never criticised at levels 3-6, unsupported twins the reverse. #12:
210 drill picks over every scene report, 0 for another check, 0 heading drills at U11 or younger.

Fix round 1 (Oct 23, `coaching_eval.py scenes` on the round's final code, 181 s): 39 variants (21 scenes), 451 checks
pass, 0 fail, 0 known, 65 not applicable, 10 vacuous; 0 report invariants broken, now including the round's rules
(coaching_eval.fix1_problems). KNOWN_DEFECTS is empty (the level-1 identity guardrail is repaired, critic B-6). New
scenes: channel_threat deep / high (COACH verifier scene 2) and stacking/reachable; low_calibration's channel opponent
waits upfield of the cover region, so its diagram-only zone exists again. Zones at the default profiles 47 -> 25 (D3
15, D1 6, D2 4): every D3 zone of wave 8 lay upfield of the planted channel opponent (critic B-8).

Fix round 3 (Oct 24, `coaching_eval.py scenes` on the round's final code, 177 s): 477 checks pass, 0 fail, 0 known, 65
not applicable, 10 vacuous; 0 report invariants broken, now with the amended level-1/2 rule (an exposed advance fails
wherever the ball is, critic B-1) and the fix-3 invariants (coaching_eval.fix3_problems, the recount with the
settled-play gates and the base-rate test). The planted exposed advances are criticised at levels 1-2 (16 of 16 not
demonstrated in none_behind, one_behind, low_calibration, stacking x2, channel_threat x2; test_6b), the covered twins,
the covered build-up and the professional scene are not; spells4 is a 'moments worth a look' review at levels 3-7,
single_advance 'too_few_situations'; wrong_formation/default's level 1 is a team-level try_this (no role or formation
named; the role downgrade of DESIGN 3.4 applies to levels 2-7).

Run:  python -m unittest test_coaching_counterexamples        (from this directory; about 3-4 minutes cold)
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
import coaching_detect_fixtures as FX  # noqa: E402  (shares the scenes' series and possession with the other tests)
import coaching_eval as E  # noqa: E402
import coaching_synth as cs  # noqa: E402
import coaching_zone_props as ZP  # noqa: E402
from soccer.coaching import alternatives as AL, assess as AS, profiles as PR, schema as S  # noqa: E402

# DESIGN 13.4 #4: the scenes whose planted unsupported advances (16 spells) must all pass every guardrail, so a level-4
# report must make a work item of them; the scenes DESIGN 13.1 designs to decline (one spell, 4 spells, an identity
# crossing, no possession, 3 team-mates in view, a wrong formation, a formation change, 60 s) are the #10 rows instead.
PLANTED_UNSUPPORTED = (('unsupported_advance', 'none_behind'), ('unsupported_advance', 'one_behind'),
                       ('low_calibration', 'default'), ('stacking', 'default'), ('stacking', 'reachable'),
                       ('channel_threat', 'deep'), ('channel_threat', 'high'), ('amateur_coach_spans', 'spans'),
                       ('amateur_coach_spans', 'veo'))
COVERED_TWINS = (('covered_advance', 'cb_shift'), ('covered_advance', 'channel'), ('professional', 'default'))
_ROWS = {}
_REPORTS = {}


def rows():
    if not _ROWS:
        for r in E.run_scenes(prepared=FX.prepared):
            _ROWS[r['scene']] = r
    return _ROWS


def report(scene, variant, level, style=None):
    key = (scene, variant, level, style)
    if key not in _REPORTS:
        _REPORTS[key] = E.scene_report(scene, variant, level, style=style, prepared=FX.prepared)
    return _REPORTS[key]


class Expectations(unittest.TestCase):
    """Every expectation of every scene at every level it names, on the real reports."""

    def test_every_scene_expectation_holds(self):
        for key, r in rows().items():
            with self.subTest(scene=key):
                bad = [c for c in r['checks'] if c['status'] in ('fail', 'fixed')]
                self.assertEqual(bad, [], key)
                self.assertTrue(any(c['status'] == 'pass' for c in r['checks']), key)

    def test_the_known_defects_are_still_open(self):
        found = {(k, c['level'], c['expect']): c['status'] for k, r in rows().items() for c in r['checks']
                 if (k, c['level'], c['expect']) in E.KNOWN_DEFECTS}
        self.assertEqual(set(found), set(E.KNOWN_DEFECTS), 'a known defect no longer appears in the scenes')
        for k, status in found.items():
            self.assertEqual(status, 'known', f'{k} now passes: remove it from coaching_eval.KNOWN_DEFECTS')

    def test_every_variant_is_covered(self):
        self.assertEqual(sorted(rows()), sorted(f'{s}/{v}' for s, v in cs.names()))
        checks = sum(c['status'] in ('pass', 'known') for r in rows().values() for c in r['checks'])
        self.assertGreaterEqual(checks, 370)

    def test_report_invariants_hold_on_every_scene_report(self):
        for key, r in rows().items():
            with self.subTest(scene=key):
                self.assertEqual(r['invariants'], [], key)

    def test_observations_do_not_change_with_the_level(self):
        # invariant 2 on the scene tiers: interpreting at every level never touched the obs / roles objects
        for key, r in rows().items():
            scene, _, variant = key.partition('/')
            b, st, obs, roles = E.scene_tiers(scene, variant, prepared=FX.prepared)
            self.assertEqual((E.sha_json(obs), E.sha_json(roles)), (r['obsSha'], r['rolesSha']), key)


class Criteria(unittest.TestCase):
    """DESIGN 13.4 criteria the scenes can measure (the counts are in the eval summary)."""

    def test_4_planted_unsupported_advances_become_work_items_at_level_4(self):
        hits = []
        for scene, variant in PLANTED_UNSUPPORTED:
            rep = report(scene, variant, 4)
            work = [f for f in rep['needsWork'] if f['observation'] == E.D3 and f['verdict'] == 'work_on']
            hits.append(bool(work))
            if work:
                self.assertGreaterEqual(work[0]['aggregate']['distinctFailingSpells'], 3, (scene, variant))
        self.assertGreaterEqual(sum(hits)/len(hits), .9, list(zip(PLANTED_UNSUPPORTED, hits)))
        for scene, variant in COVERED_TWINS:
            for lv in E.LEVELS:
                rep = report(scene, variant, lv)
                crit = [f['key'] for f in E.scoped(rep, (E.D3,), lv) if f['verdict'] in E.CRITICISM]
                self.assertEqual(crit, [], (scene, variant, lv))

    def test_5_levels_5_to_7_reason_with_more_relations(self):
        for variant in ('none_behind', 'one_behind'):
            reps = {lv: report('unsupported_advance', variant, lv) for lv in (2, 3, 5, 6)}
            reps[7] = report('unsupported_advance', variant, 7, style='possession')
            got = E.claims_by_level(reps)
            low = max(got.get(2, 0), got.get(3, 0))
            for lv in (5, 6, 7):
                self.assertGreaterEqual(got[lv], low+2, (variant, got))

    def test_6_supported_and_unsupported_twins(self):
        for lv in (3, 4, 5, 6):
            for scene, variant in (('covered_advance', 'cb_shift'), ('covered_advance', 'channel')):
                rep = report(scene, variant, lv)
                self.assertFalse([f for f in rep['needsWork']+rep['reviews'] if f['observation'] == E.D3], (variant, lv))
                self.assertTrue([f for f in rep['strengths'] if f['observation'] == E.D3], (variant, lv))
            for variant in ('none_behind', 'one_behind'):
                rep = report('unsupported_advance', variant, lv)
                self.assertTrue([f for f in rep['needsWork']+rep['reviews'] if f['observation'] == E.D3], (variant, lv))
                self.assertFalse([f for f in rep['strengths'] if f['observation'] == E.D3], (variant, lv))

    def test_6b_levels_1_2_tell_the_covered_from_the_exposed_advance(self):
        # fix round 3 (critic B-1 / N7, criterion 6 at levels 1-2): the covered twins, the covered build-up and the
        # professional-like scene are never criticised at levels 1-2 (level 1 may praise a covered advance); every
        # planted exposed advance is (16 of 16 failing samples' opportunities, levels 1 and 2)
        for scene, variant in COVERED_TWINS+(('covered_buildup', 'default'),):
            for lv in (1, 2):
                rep = report(scene, variant, lv)
                crit = [f['key'] for f in E.scoped(rep, (E.D3,), lv) if f['verdict'] in E.CRITICISM]
                self.assertEqual(crit, [], (scene, variant, lv))
        for scene, variant in PLANTED_UNSUPPORTED[:7]:
            for lv in (1, 2):
                rep = report(scene, variant, lv)
                crit = [f for f in E.scoped(rep, (E.D3,), lv) if f['verdict'] in E.CRITICISM]
                self.assertTrue(crit, (scene, variant, lv))
                self.assertEqual(crit[0]['aggregate']['notDemonstrated'], 16, (scene, variant, lv))

    def test_9_every_zone_holds_its_properties(self):
        # Fix round 1: 25 zones (D3 15, D1 6, D2 4) at the scenes' default profiles, against 47 before the round. Critic
        # B-8 removed the D3 zones that lay upfield of the planted channel opponent (all of them: COACH found 50 of 50
        # failing), S-2's three-state pressure call the D1 zones of the 3 m presser; the D3 zones are back on the
        # scenes whose opponent waits upfield of the cover region (low_calibration, stacking/reachable,
        # channel_threat/high). The floor is the measured 25 less a margin, with every detector that has a zone.
        zones, by = 0, {}
        for scene, variant in cs.names():
            b, st, obs, roles = E.scene_tiers(scene, variant, prepared=FX.prepared)
            moments = E.moment_index(obs, roles)
            for lv in E.LEVELS:
                rep = report(scene, variant, lv)
                prof = rep['profile']
                fs = {f['id']: f for f in E.findings_of(rep)}
                for alt in rep['alternatives']:
                    zones += 1
                    det = fs[alt['finding']]['observation']
                    by[det] = by.get(det, 0)+1
                    side = AL.attack_side(obs, moments[alt['evidence']])
                    probs = ZP.zone_problems(alt, fs[alt['finding']], obs, roles, prof, side)
                    self.assertEqual(probs, [], (scene, variant, lv, alt['id']))
                self.assertEqual(E.zone_side_problems(rep, obs, roles), [], (scene, variant, lv))
        self.assertGreaterEqual(zones, 20, by)
        self.assertEqual({E.D1, E.D2, E.D3} - set(by), set(), by)

    def test_9b_the_zone_side_check_catches_an_area_on_the_wrong_side(self):
        # the independent B-8 check (coaching_eval.zone_side_problems, game-3 G1+) has teeth: the channel_threat/high
        # zone moved 10 m upfield (the old engine's 5-11 m) is flagged, and so is a behind-the-ball area ahead of the ball
        b, st, obs, roles = E.scene_tiers('channel_threat', 'high', prepared=FX.prepared)
        rep = report('channel_threat', 'high', 4)
        alts = [a for a in rep['alternatives'] if a.get('zone')]
        self.assertTrue(alts)
        self.assertEqual(E.zone_side_problems(rep, obs, roles), [])
        L = obs['timeline']['scale']['lengthM']
        moments = E.moment_index(obs, roles)
        side = AL.attack_side(obs, moments[alts[0]['evidence']])
        self.assertEqual(side, 'left')                      # scene truth: A defends the left goal
        moved = copy.deepcopy(rep)
        for a in moved['alternatives']:
            if a.get('zone'):
                a['zone']['polygon'] = [[q[0]+10.0/L, q[1]] for q in a['zone']['polygon']]
        bad = E.zone_side_problems(moved, obs, roles)
        self.assertTrue(bad and 'upfield' in bad[0], bad)
        m = moments[alts[0]['evidence']]
        ball = (m.get('scene') or {}).get('ball')
        if ball is not None:
            ahead = copy.deepcopy(rep)
            a = next(x for x in ahead['alternatives'] if x.get('zone'))
            a['improves'] = dict(a.get('improves') or {}, relation='behind_ball')
            a['zone']['polygon'] = [[ball[0]+dx/L, ball[1]+dy] for dx, dy in ((1, 0), (6, 0), (6, .05), (1, .05))]
            self.assertTrue([p for p in E.zone_side_problems(ahead, obs, roles) if 'ahead of the ball' in p])

    def test_11_removing_the_demonstrations_removes_the_strength(self):
        b, st, obs, roles = E.scene_tiers('covered_advance', 'cb_shift', prepared=FX.prepared)
        rep = E.interpret(obs, roles, st, 'A', 4)
        strong = [f for f in rep['strengths'] if f['observation'] == E.D3]
        self.assertEqual(len(strong), 1)
        prof = rep['profile']
        need = prof['evidence']['strengthSpells']
        params = prof['detectors'][E.D3]['params']
        step = obs['timeline']['step']
        demo = [o['id'] for o in roles['opportunities'] if o['detector'] == E.D3 and
                (AS.classify(o, params, obs['timeline']['scale'], step) or {}).get('outcome') == 'demonstrated']
        self.assertGreaterEqual(len(demo), need)

        def keep(n):
            r2 = copy.deepcopy(roles)
            drop = set(demo[n:])
            r2['opportunities'] = [o for o in r2['opportunities'] if o['id'] not in drop]
            r2['moments'] = [m for m in r2['moments'] if m['opportunity'] not in drop]
            cov = r2['detectors'][E.D3]
            cov['opportunities'] = sum(1 for o in r2['opportunities'] if o['detector'] == E.D3)
            return [f for f in E.interpret(obs, r2, st, 'A', 4)['strengths'] if f['observation'] == E.D3]
        self.assertEqual(keep(need-1), [], 'fewer demonstrated spells than the level needs must not make a strength')
        self.assertEqual(len(keep(len(demo))), 1)
        # the twin with the demonstrations removed in the scene itself (DESIGN 13.4 #11 mutation twin)
        for lv in E.LEVELS:
            rep = report('professional', 'no_advances', lv)
            self.assertFalse([f for f in rep['strengths'] if f['observation'] == E.D3], lv)

    def test_10_declines_on_the_designed_scenes(self):
        declines = {('single_advance', 'default'): 'review_one_instance', ('unsupported_advance', 'spells4'): None,
                    ('identity_crossing', 'default'): None, ('possession_unknown', 'default'): None,
                    ('three_visible', 'default'): None, ('wrong_formation', 'default'): None,
                    ('formation_change', 'default'): None, ('short_segment', 'default'): None}
        for (scene, variant), _ in declines.items():
            for lv in range(2, 8):
                rep = report(scene, variant, lv)
                work = [f['key'] for f in rep['needsWork'] if f['observation'] == E.D3]
                self.assertEqual(work, [], (scene, variant, lv))

    def test_12_drills_follow_findings_and_never_head_young_players(self):
        from soccer.coaching import drills as DR
        lib = DR.load_library()
        base = cs.settings_for(cs.build_truth('unsupported_advance', 'none_behind'), level=4)['teams']['A']
        for age in ('U6', 'U8', 'U9', 'U10', 'U11'):
            for lv in E.LEVELS:
                block = dict(base, ageGroup=age, playingLevel=lv)
                prof = PR.resolve(block, team='A')
                for d in lib['drills']:
                    if not DR.safety_problems(d, age, prof.get('safety')):
                        self.assertEqual(d['safety']['heading'], 'none', (age, lv, d['drill_id']))
        rep = report('unsupported_advance', 'none_behind', 4)
        picks = [p for p in rep['drills'] if p['finding']]
        self.assertTrue(picks)
        for p in picks:
            f = next(x for x in E.findings_of(rep) if x['id'] == p['finding'])
            self.assertIn(f['observation'], DR.get(lib, p['drill'])['related_tactical_detector'])


if __name__ == '__main__':
    unittest.main()
