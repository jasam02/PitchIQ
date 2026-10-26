"""DESIGN 13.2: the same video at different levels. One measurement, many coaches: the observation and role files are
re-measured under every level, age band and style through the server's own path (store.measure, store.build_role_tier)
and must keep their bytes (invariant 2: the raw observations never change with the coaching profile); every cited moment
exists in them (invariant 3); a family's findings cite the same moments at every level; the enabled detectors and
verdicts follow DESIGN 2.3's table, written out in tools/coaching_eval.DESIGN_TABLE from the DESIGN text and NOT read
from profiles.json (so a profiles.json edit that drifts from the contract fails here); explanations of one family differ
between levels (word-trigram Jaccard < .5); and levels 1-2 respect their caps (<= 1 / <= 2 work items, no player named
in a challenge, no formation-discipline item, none of the level's banned words).

Synthetic scenes always run; real run copies run when PITCHIQ_COACHING_EVAL_RUNS names them
(NAME=path/to/result.json joined with ';'), as tools/coaching_eval.py runs them for the eval summary.

Measured on Oct 22 (wave 8, `coaching_eval.py levels` on all 30 scene variants x the 14 profiles of
coaching_eval.PROFILES: each level at its typical age, the two level-5 bands, an academy U19 and an adult team at level
6, three declared styles, an adult beginner at level 3, a U6 at level 1; 4 of them re-measured): the obs and roles
bytes were identical under every re-measured profile; no table, evidence or level-1/2 problem; profiles with the same
detector parameters always cited the same moments; the most similar explanations of one family at two different levels
scored a word-trigram Jaccard of .368 (limit .5) outside two open defects recorded in coaching_eval.KNOWN_LEVEL_DEFECTS,
which this test asserts are still seen (so a repair is noticed): TEXT's attacking-shape description repeats between
levels 5 and 7 (and 1/2, 4/6 when the phase is unknown; 184 pairs at .55-.76), and DETECT's findings re-pick their
moments when a level judges a different failing set (defending/no_press D1 at level 1 vs 2-7, small_field_scale D5
under the compact style). On the real run copies (`coaching_eval.py runs`, 196 reports) the same checks found 0
problems and 64 pairs of the D6 repetition. The first run of the table check also showed that DESIGN 2.3's rows alone
are not the contract: strengths come from DESIGN 4.3/4.8/4.9 and level-2 attacking shape's try_this from 4.6, so
DESIGN_TABLE merges 2.3 with the "Levels" rows of 4.1-4.6. This file runs 6 of the scenes (about 2-3 minutes).

Fix round 1 (Oct 23): both open defects are repaired (TEXT's per-level descriptions, critic S-6; DETECT's moments
picked once per family, S-5) and KNOWN_LEVEL_DEFECTS is empty, so test_the_known_defects_are_still_open asserts
nothing and any evidence or text hit is a plain problem. `coaching_eval.py levels` on all 39 scene variants: 0
problems, obs / roles identical under re-measurement, the worst level pair of one family .444 (807 s); on the 196
real-run reports 0 problems and 0 known defects (`runs`).

Fix round 3 (Oct 24): the level-1/2 reports of none_behind and amateur_coach_spans now criticise the exposed advances
(critic B-1), which the amended G1+someone_back invariant accepts; 0 problems on these six scenes (the full
`coaching_eval.py levels` run over every scene variant is in scratch build/eval_fix3/levels.log).

Run:  python -m unittest test_coaching_levels                 (from this directory)
"""
import os
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for p in (str(ROOT), str(HERE), str(ROOT/'tools')):
    if p not in sys.path:
        sys.path.insert(0, p)
import coaching_detect_fixtures as FX  # noqa: E402
import coaching_eval as E  # noqa: E402
from soccer.coaching import profiles as PR  # noqa: E402

SCENES = (('unsupported_advance', 'none_behind'), ('defending', 'press_no_cover'), ('defending', 'no_press'),
          ('amateur_coach_spans', 'spans'), ('kids_swarm', 'default'), ('professional', 'default'))
# Re-measured profiles (the measurement tier reads only format and field, so a few that differ in every interpretation
# input are enough to show the bytes do not move; all 14 are interpreted).
REMEASURED = E.REMEASURED


class SameVideoDifferentLevels(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.out = {}
        for scene, variant in SCENES:
            b, st, obs, roles = E.scene_tiers(scene, variant, prepared=FX.prepared)
            cls.out[(scene, variant)] = E.levels_check(
                obs, roles, st, 'A', remeasure=E.scene_remeasure(scene, variant, prepared=FX.prepared),
                remeasured=REMEASURED)

    def test_no_problem_on_any_scene(self):
        for key, res in self.out.items():
            with self.subTest(scene=key):
                self.assertEqual([p for p in res['problems'] if not p[0].startswith('known:')], [], key)

    def test_the_known_defects_are_still_open(self):
        seen = {p[0][6:] for res in self.out.values() for p in res['problems'] if p[0].startswith('known:')}
        for k in E.KNOWN_LEVEL_DEFECTS:
            self.assertIn(k, seen, f'{k} is no longer seen: remove it from coaching_eval.KNOWN_LEVEL_DEFECTS')

    def test_every_profile_was_interpreted(self):
        for key, res in self.out.items():
            self.assertEqual(len(res['reports']), len(E.PROFILES), key)
            self.assertEqual({r['profile']['level'] for r in res['reports'].values()}, set(E.LEVELS), key)

    def test_profiles_differ_where_the_table_differs(self):
        # DESIGN 13.4 #1: two levels whose table rows differ give different reports (statuses, verdicts or texts)
        for key, res in self.out.items():
            sig = {}
            for tag, rep in res['reports'].items():
                lv = rep['profile']['level']
                sig.setdefault(lv, (tuple(d['status'] for d in rep['overview']['detectors']),
                                    tuple(sorted((E.family(f), f['verdict'], (f['text'] or {}).get('explanation'))
                                                 for f in E.findings_of(rep))), rep['overview']['text']))
            for a in E.LEVELS:
                for b in E.LEVELS:
                    if a < b and E.DESIGN_TABLE[a] != E.DESIGN_TABLE[b]:
                        self.assertNotEqual(sig[a], sig[b], (key, a, b))

    def test_explanations_of_a_family_differ_between_levels(self):
        worst = max((max(res['pairs']['jaccard'] or [0.0]) for res in self.out.values()), default=0.0)
        self.assertLess(worst, E.TRIGRAM_LIMIT)

    def test_a_family_cites_the_same_moments_at_every_level(self):
        res = self.out[('unsupported_advance', 'none_behind')]
        cited = {}
        for tag, rep in res['reports'].items():
            for f in E.findings_of(rep):
                if f['observation'] == E.D3 and f['verdict'] in E.CRITICISM:
                    cited[tag] = tuple(t['evidence'] for t in f['timestamps'])
        self.assertGreaterEqual(len(cited), 10)
        self.assertEqual(len(set(cited.values())), 1, cited)

    def test_the_table_is_the_designs(self):
        # the profile of every level against DESIGN 2.3 on its own (no report): enabled detectors, verdicts, caps
        for lv in E.LEVELS:
            for age in {PR.default_age(lv), 'Adult'}:
                prof = PR.resolve({'ageGroup': age, 'playingLevel': lv}, team='A')
                self.assertEqual(E.profile_table_problems(prof), [], (lv, age))
                self.assertLessEqual(prof['evidence']['maxWorkItems'], E.MAX_WORK[lv])


@unittest.skipUnless(os.environ.get('PITCHIQ_COACHING_EVAL_RUNS'), 'set PITCHIQ_COACHING_EVAL_RUNS to NAME=result.json '
                                                                    'pairs (;-separated) to run on real run copies')
class RealRuns(unittest.TestCase):
    def test_real_runs_at_every_level(self):
        for item in [x for x in os.environ['PITCHIQ_COACHING_EVAL_RUNS'].split(';') if x]:
            name, _, path = item.partition('=')
            result, sha, obs, roles, _ = E.measure_run(path)
            for team in ('A', 'B'):
                def remeasure(settings, team=team, path=path, loaded=(result, sha)):
                    _, _, o2, r2, _ = E.measure_run(path, settings=settings, loaded=loaded)
                    return o2[team], r2[team]
                res = E.levels_check(obs[team], roles[team], E.run_settings(), team, remeasure=remeasure,
                                     remeasured=E.REMEASURED)
                self.assertEqual([p for p in res['problems'] if not p[0].startswith('known:')], [], (name, team))


if __name__ == '__main__':
    unittest.main()
