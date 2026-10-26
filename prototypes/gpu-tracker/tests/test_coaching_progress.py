"""Development progression across matches (soccer/coaching/progress.py, DESIGN v2 section 8; spec Part 13).

Why these checks. Spec Part 13 forbids three things a progression can easily do: base progress on report sentiment,
compare a 60 s analysis with a 90 minute one on totals, and fix its thresholds in code. So the tests pin that an entry
holds only opportunities (never a verdict), that its outcomes are exactly the report's (assess.classify) at every level
and are re-read when the level changes, that a short clip and a full match with the same per-opportunity behaviour read
the same (DESIGN 13.4 #14), and that every number lives in progress_rules.json marked PITCHIQ-PROPOSAL. The mastery
ladder (NOT ENOUGH EVIDENCE, LEARNING, DEVELOPING, CONSISTENT, ADAPTABLE) is tested at its boundaries on hand-built
histories; the store's rules (one entry per range, the longest of overlapping ranges, strict reads, stale entries
listed apart, unconfirmed team links writing nothing) on temporary folders; the outcome table's completeness by
scanning the classification code it caches. The synthetic scenes (EVAL's, through the real tiers) give the real
opportunities.

Run:  python -m unittest test_coaching_progress          (from this directory)
"""
import ast
import copy
import hashlib
import inspect
import json
import os
import re
import tempfile
import textwrap
import time
import unittest
from collections import Counter
from pathlib import Path

import coaching_detect_fixtures as FX
from soccer.coaching import assess as AS, contextual as CX, profiles as PR, progress as PG, schema as S  # noqa: E402

SQ = S.record_id('sq', 'u14 blue', '2026-10-01T10:00:00')
NOW = '2026-10-22T10:00:00+00:00'


def prof(level, style=None, players=11, age=None):
    block = {'playingLevel': level, 'ageGroup': age or PR.default_age(level), 'format': {'players': players}}
    if style:
        block['style'] = style
    return PR.resolve(block, team='A')


def block(level, style=None, players=11, age=None):
    return {'playingLevel': level, 'ageGroup': age or PR.default_age(level), 'format': {'players': players},
            'style': style or 'unknown'}


def entry(video, day, outs, *, skill='pressure_and_cover', profiles=None, phase=600.0, t0=0.0, length=5400.0,
          zone='own_third', channel='left', quality='high', run='run-1', team='A', squad=SQ, fmt='11v11', coach=False,
          stale=False, version='2', measurement='a'*16):
    """A hand-built entry: one opportunity per outcome code in outs (1, -1, 0, None) of one skill, with that outcome
    under every given profile's parameters (default: level 4). zone / channel may be callables of the index."""
    d = S.SKILLS[skill]
    keys = [PG.outcome_key(d, p['detectors'][d]['params']) for p in (profiles or [prof(4)])]
    opps = []
    for k, v in enumerate(outs):
        a = t0+1.0+k*(length-2.0)/max(1, len(outs))
        rel = {key: v for key in keys}
        rel.update({'samples': 10, 'coachPhase': coach})
        opps.append({'t0': round(a, 2), 't1': round(a+1.0, 2), 'zone': zone(k) if callable(zone) else zone,
                     'channel': channel(k) if callable(channel) else channel, 'rel': rel,
                     'quality': quality(k) if callable(quality) else quality})
    skills = {s: {'opportunities': [], 'phaseSeconds': phase, 'eligibleShare': None, 'confidence': 'low'}
              for s in S.SKILLS}
    skills[skill]['opportunities'] = opps
    e = {'id': S.record_id('pe', squad, video, run, team, t0, t0+length), 'squad': squad, 'video': video, 'run': run,
         'team': team, 'range': {'t0': t0, 't1': t0+length}, 'matchDate': day, 'dateSource': 'coach',
         'engine': {'measurement': measurement, 'version': version},
         'context': {'format': fmt, 'footage': 'broadcast', 'phaseAvailable': True, 'level': 4, 'ageGroup': 'U14'},
         'skills': skills, 'updated': NOW}
    if stale:
        e['stale'] = True
    return S.check('progress_entry', e)


def history(spec, **kw):
    """Entries for [(demonstrated, judged), ...] matches, one per video, dated one week apart."""
    out = []
    for k, (d, n) in enumerate(spec):
        outs = [1]*d+[-1]*(n-d)
        out.append(entry(f'v{k:03d}', f'2026-{1+k//28:02d}-{1+k % 28:02d}', outs, **kw))
    return out


def state(entries, level=4, skill='pressure_and_cover', **kw):
    return PG.mastery(entries, block(level, **kw), squad_id=SQ)['skills'][skill]


def scene_entry(scene, variant=None, *, t0=None, t1=None, coach=True, level=4):
    obs, roles = FX.tiers(scene, variant, coach=coach)
    return PG.build_entry(obs, roles, squad=SQ, video='vscene', run='run-1', match_date='2026-10-01',
                          date_source='coach', level=level, age_group=PR.default_age(level), t0=t0, t1=t1,
                          updated=NOW), obs, roles


class Rules(unittest.TestCase):
    def test_rules_file_validates_and_every_number_is_a_pitchiq_proposal(self):
        rules = PG.load_rules()
        self.assertEqual(PG.validate_rules({k: v for k, v in rules.items() if not k.startswith('_')}), [])
        self.assertEqual(rules['origin'], S.ORIGIN_P)
        self.assertIs(rules['validated'], False)
        self.assertIn('unvalidated', rules['label'].lower())
        self.assertIn('coach', rules['label'].lower())
        self.assertAlmostEqual(rules['_z'], 1.2816, places=3)

    def test_thresholds_are_design_8_4(self):
        th = PG.load_rules()['thresholds']
        want = {1: (.5, .65), 2: (.5, .65), 3: (.5, .65), 4: (.6, .75), 5: (.6, .75), 6: (.7, .85), 7: (.7, .85)}
        for lv, (s1, s2) in want.items():
            self.assertEqual((th[str(lv)]['s1'], th[str(lv)]['s2']), (s1, s2))
        ev = PG.load_rules()['evidence']
        self.assertEqual((ev['matchOpportunities'], ev['minOpportunities'], ev['minQualifyingMatches'],
                          ev['consistentMatches']), (6, 20, 2, 3))

    def test_broken_rules_are_refused(self):
        good = json.loads(PG.RULES_PATH.read_text(encoding='utf-8'))
        for path, value, needle in ((('thresholds', '4', 's1'), .9, 'thresholds.4'),
                                    (('evidence', 'origin'), 'FEDERATION', 'evidence'),
                                    (('adaptable', 'dimensions', 'pressure_and_cover'), ['weather'], 'dimensions'),
                                    (('trend', 'validated'), True, 'trend'),
                                    (('interval', 'confidence'), 1.5, 'interval'),
                                    (('evidence', 'minQuality'), 'great', 'minQuality')):
            doc = copy.deepcopy(good)
            node = doc
            for k in path[:-1]:
                node = node[k]
            node[path[-1]] = value
            problems = PG.validate_rules(doc)
            self.assertTrue(any(needle in p for p in problems), (path, problems))

    def test_a_number_outside_a_tagged_block_is_refused(self):
        doc = json.loads(PG.RULES_PATH.read_text(encoding='utf-8'))
        doc['origin'] = 'FEDERATION'
        self.assertTrue(PG.validate_rules(doc))


class OutcomeTable(unittest.TestCase):
    def _sources(self):
        return {(m, n): inspect.getsource(getattr(PG._MODULES[m], n)) for m, n in PG.CLASSIFY_CODE}

    def test_every_parameter_the_classification_reads_is_in_the_table_key(self):
        src = inspect.getsource(CX.sample_outcomes)+inspect.getsource(CX.pressure_cover_flags) + \
            inspect.getsource(AS.classify)
        keys = set(re.findall(r"""\b(?:p|params)(?:\[|\.get\()\s*['"](\w+)['"]""", src))
        for name in re.findall(r"""_scaled_param\(p,\s*['"](\w+)['"]""", src):
            keys |= {name, name[:-1]+'Scale'}
        self.assertTrue(keys)
        self.assertEqual(sorted(keys-set(PG.CLASSIFY_KEYS)), [])

    def test_the_hashed_code_calls_nothing_else_of_ours(self):
        """Every soccer function or constant the cached classification uses is in CLASSIFY_CODE / CLASSIFY_VALUES, so
        a change to any of them changes the outcome keys (and old entries are re-derived, never misread)."""
        listed_fn = {(PG._MODULES[m].__name__, n) for m, n in PG.CLASSIFY_CODE}
        listed_val = {(PG._MODULES[m].__name__, n) for m, n in PG.CLASSIFY_VALUES}
        missing = set()
        for m, n in PG.CLASSIFY_CODE:
            mod = PG._MODULES[m]
            tree = ast.parse(textwrap.dedent(inspect.getsource(getattr(mod, n))))
            for node in ast.walk(tree):
                target = None
                if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                    target = (mod, node.id)
                elif isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
                    other = getattr(mod, node.value.id, None)
                    if inspect.ismodule(other) and other.__name__.startswith('soccer.'):
                        target = (other, node.attr)
                if target is None or not hasattr(target[0], target[1]):
                    continue
                owner, name = target
                obj = getattr(owner, name)
                if inspect.ismodule(obj) or inspect.isclass(obj) or inspect.isbuiltin(obj):
                    continue
                if inspect.isfunction(obj):
                    if obj.__module__.startswith('soccer.') and (obj.__module__, obj.__name__) not in listed_fn:
                        missing.add(f'{obj.__module__}.{obj.__name__}')
                elif owner.__name__.startswith('soccer.') and name.isupper() and \
                        (owner.__name__, name) not in listed_val:
                    missing.add(f'{owner.__name__}.{name}')
        self.assertEqual(sorted(missing), [])

    def test_the_parameter_subset_classifies_like_the_whole_profile(self):
        obs, roles = FX.tiers('unsupported_advance', 'none_behind')
        opps = obs['opportunities']+roles['opportunities']
        scale, step = obs['timeline']['scale'], obs['timeline']['step']
        for level in range(1, 8):
            for style in ('unknown', 'counter', 'high_press'):
                p = prof(level, style)
                for o in opps:
                    dp = p['detectors'][o['detector']]
                    a = AS.classify(o, dp['params'], scale, step)
                    b = AS.classify(o, PG.classify_params(dp['params']), scale, step)
                    self.assertEqual(a and a['outcome'], b and b['outcome'], (level, style, o['key']))

    def test_the_key_follows_the_parameters_that_matter_only(self):
        p = copy.deepcopy(prof(4)['detectors']['pressure_cover']['params'])
        k0 = PG.outcome_key('pressure_cover', p)
        self.assertRegex(k0, PG.OUTCOME_KEY)
        p['relevance'] = 99.0
        p['objectiveRank'] = 1
        self.assertEqual(PG.outcome_key('pressure_cover', p), k0)
        p['pressureM'] = 6.0
        self.assertNotEqual(PG.outcome_key('pressure_cover', p), k0)
        self.assertNotEqual(PG.outcome_key('wide_defender_advance', p, 'behind_ball'),
                            PG.outcome_key('wide_defender_advance', p))
        q = {'minBehind': None}
        self.assertNotEqual(PG.outcome_key('behind_ball', q), PG.outcome_key('behind_ball', {}))

    def test_every_profile_of_a_format_has_its_outcome_in_the_table(self):
        for players in (4, 7, 11):
            sets = PG.param_sets(players)
            for level in range(1, 8):
                for style in S.STYLES:
                    p = prof(level, style, players=players)
                    for d in S.DETECTORS:
                        dp = p['detectors'][d]
                        if dp['enabled']:
                            self.assertIn(PG.outcome_key(d, dp['params']), sets[d], (players, level, style, d))
                    if PG.reframed(p):
                        self.assertIn(PG.outcome_key('wide_defender_advance', p['detectors']['behind_ball']['params'],
                                                     'behind_ball'), sets['wide_defender_advance'])
        n = {S.DETECTOR_CODE[d]: len(v) for d, v in PG.param_sets(11).items()}
        self.assertLessEqual(sum(n.values()), 40, n)      # a compact table (22 own + 2 reframed sets measured)

    def test_the_code_hash_ignores_docstrings_but_not_code(self):
        def f(x):
            """one."""
            return x+1

        def g(x):
            """two, longer."""
            return x+1

        def h(x):
            return x+2
        self.assertEqual(PG._semantic(f).replace("'f'", "'g'"), PG._semantic(g))
        self.assertNotEqual(PG._semantic(f).replace("'f'", "'h'"), PG._semantic(h))

    def test_a_change_of_the_classification_code_changes_every_key(self):
        params = prof(4)['detectors']['pressure_cover']['params']
        k0 = PG.outcome_key('pressure_cover', params)
        original = CX.pressure_cover_flags

        def pressure_cover_flags(R, k, params, scale):
            return original(R, k, params, scale)[::-1]
        saved = PG._cache.pop('code', None)
        CX.pressure_cover_flags = pressure_cover_flags
        try:
            self.assertNotEqual(PG.outcome_key('pressure_cover', params), k0)
        finally:
            CX.pressure_cover_flags = original
            PG._cache.pop('code', None)
            if saved is not None:
                PG._cache['code'] = saved
        self.assertEqual(PG.outcome_key('pressure_cover', params), k0)


class EntriesFromScenes(unittest.TestCase):
    """Real opportunities (EVAL's scenes through the measurement and role tiers) into entries."""

    def test_entry_validates_and_holds_only_counted_opportunities(self):
        e, obs, roles = scene_entry('unsupported_advance', 'none_behind')
        S.check('progress_entry', e)
        lo, hi = obs['segment']['t0'], obs['segment']['t1']
        opps = obs['opportunities']+roles['opportunities']
        for skill, d in S.SKILLS.items():
            want = sorted((o['t0'], o['t1']) for o in opps if o['detector'] == d and PG.counted(o, lo, hi))
            got = sorted((o['t0'], o['t1']) for o in e['skills'][skill]['opportunities'])
            self.assertEqual(got, want, skill)
        self.assertEqual(len(e['skills']['wide_defender_support']['opportunities']), 16)
        # No verdict, finding, text or level judgement is stored: only opportunities and their relations.
        text = json.dumps(e)
        for word in ('verdict', 'work_on', 'try_this', 'strength', 'headline', 'finding'):
            self.assertNotIn(word, text)
        self.assertIsNone(PG.needs_rederive(e))
        self.assertEqual(e['engine']['measurement'], obs['engine']['measurement'])
        self.assertTrue(e['context']['phaseAvailable'])

    def test_building_an_entry_never_changes_the_observations(self):
        obs, roles = FX.tiers('unsupported_advance', 'none_behind')
        before = [hashlib.sha256(json.dumps(x, sort_keys=True).encode()).hexdigest() for x in (obs, roles)]
        a = PG.build_entry(obs, roles, squad=SQ, video='v', run='r', match_date='2026-10-01', date_source='coach',
                           level=2, age_group='U10', updated=NOW)
        b = PG.build_entry(obs, roles, squad=SQ, video='v', run='r', match_date='2026-10-01', date_source='coach',
                           level=6, age_group='Adult', updated=NOW)
        after = [hashlib.sha256(json.dumps(x, sort_keys=True).encode()).hexdigest() for x in (obs, roles)]
        self.assertEqual(before, after)
        self.assertEqual(a['skills'], b['skills'])         # the level at analysis time changes nothing stored
        self.assertEqual(a['id'], b['id'])

    def test_outcomes_equal_the_report_at_every_level(self):
        """The progression reads each opportunity exactly as the report does (assess's aggregate counts per family),
        including level 1's reading of the advances as "someone ready to protect our goal"."""
        for scene, variant in (('unsupported_advance', 'none_behind'), ('defending', 'press_no_cover'),
                               ('covered_advance', 'cb_shift')):
            e, obs, roles = scene_entry(scene, variant)
            for level in range(1, 8):
                p = FX.profile(scene, variant, level=level)
                out = AS.assess(obs, roles, p)
                for f in out['findings']:
                    d, fam = f['observation'], f['key'].split('|')[2]
                    if fam == 'unknown':
                        continue
                    if fam == 'advance':
                        key = PG.outcome_key('wide_defender_advance', p['detectors'][d]['params'], 'behind_ball')
                        rows = e['skills']['wide_defender_support']['opportunities']
                    else:
                        key = PG.outcome_key(d, p['detectors'][d]['params'])
                        rows = e['skills'][PG.SKILL_OF[d]]['opportunities']
                        if fam in ('left', 'right'):
                            rows = [o for o in rows if o['rel'].get('leftSide') == (fam == 'left')]
                    c = Counter(o['rel'][key] for o in rows)
                    agg = f['aggregate']
                    self.assertEqual((c[1], c[-1], c[0]), (agg['demonstrated'], agg['notDemonstrated'],
                                                           agg['unclear']), (scene, level, f['key']))
                    if fam != 'advance':
                        self.assertAlmostEqual(e['skills'][PG.SKILL_OF[d]]['phaseSeconds'], agg['phaseSeconds'],
                                               places=1)

    def test_level_one_reads_unsupported_advances_as_someone_back(self):
        # fix round 3 (critic B-1): level 1 reads an advance as "someone ready to protect our goal": covered +1, left
        # exposed (one of theirs in the vacated channel, nobody of ours in it) -1 wherever the ball is, otherwise the
        # behind-the-ball rule in the final third. The 16 planted advances of unsupported_advance/none_behind (the
        # channel empty, B-10 in it, the ball at 36-42 m) were out of scope under fix round 1's final-third-only rule;
        # the progression now reads them as failing, as many as level 2 judges failing on this 11v11 scene, and their
        # covered twins as protected
        e, _, _ = scene_entry('unsupported_advance', 'none_behind')
        got1 = Counter(v for _, v in PG.outcomes(e, prof(1), 'someone_back_when_attacking'))
        got2 = Counter(v for _, v in PG.outcomes(e, prof(2), 'wide_defender_support'))
        self.assertEqual((got1[-1], got1[1]), (16, 0))
        self.assertEqual((got1[-1], got1[1]), (got2[-1], got2[1]))
        for variant in ('cb_shift', 'channel'):
            e, _, _ = scene_entry('covered_advance', variant)
            got = Counter(v for _, v in PG.outcomes(e, prof(1), 'someone_back_when_attacking'))
            self.assertEqual((got[-1], got[1]), (0, 16), variant)
        e, _, _ = scene_entry('unsupported_advance', 'final_third')
        got1 = PG.outcomes(e, prof(1), 'someone_back_when_attacking')
        got4 = PG.outcomes(e, prof(4), 'someone_back_when_attacking')
        self.assertGreaterEqual(Counter(v for _, v in got1)[-1], 8)     # the planted advances, nobody behind the ball
        self.assertEqual(PG.outcomes(e, prof(1), 'wide_defender_support'), [])     # D3 off at level 1
        self.assertTrue(got4)

    def test_a_reframed_advance_overlapping_a_judged_behind_ball_moment_counts_once(self):
        p1 = prof(1)
        k4 = PG.outcome_key('behind_ball', p1['detectors']['behind_ball']['params'])
        k3 = PG.outcome_key('wide_defender_advance', p1['detectors']['behind_ball']['params'], 'behind_ball')
        e = entry('v1', '2026-10-01', [1], skill='someone_back_when_attacking', profiles=[p1])
        o4 = e['skills']['someone_back_when_attacking']['opportunities'][0]
        o4['rel'] = {k4: 1}
        adv = [{'t0': o4['t0']+.5, 't1': o4['t1']+2, 'zone': 'middle_third', 'channel': 'left', 'rel': {k3: -1},
                'quality': 'high'},
               {'t0': o4['t1']+30, 't1': o4['t1']+34, 'zone': 'middle_third', 'channel': 'left', 'rel': {k3: -1},
                'quality': 'high'}]
        e['skills']['wide_defender_support']['opportunities'] = adv
        got = PG.outcomes(e, p1, 'someone_back_when_attacking')
        self.assertEqual(sorted(v for _, v in got), [-1, 1])

    def test_range_takes_opportunities_whose_middle_is_inside(self):
        full, obs, _ = scene_entry('unsupported_advance', 'none_behind')
        seg = obs['segment']
        mid = (seg['t0']+seg['t1'])/2.0
        half, _, _ = scene_entry('unsupported_advance', 'none_behind', t1=mid)
        for skill in S.SKILLS:
            want = [o for o in full['skills'][skill]['opportunities'] if (o['t0']+o['t1'])/2.0 <= mid+1e-6]
            self.assertEqual(len(half['skills'][skill]['opportunities']), len(want), skill)
        self.assertEqual(half['range'], {'t0': seg['t0'], 't1': round(mid, 3)})
        self.assertNotEqual(half['id'], full['id'])
        with self.assertRaises(PG.ProgressError):
            scene_entry('unsupported_advance', 'none_behind', t0=seg['t1']+10, t1=seg['t1']+20)

    def test_no_phase_means_no_progress_opportunity(self):
        e, obs, _ = scene_entry('possession_unknown')
        self.assertFalse(e['context']['phaseAvailable'])
        self.assertEqual(sum(len(r['opportunities']) for r in e['skills'].values()), 0)
        self.assertTrue(any(o['detector'] == 'attacking_shape' for o in obs['opportunities']))  # describe-only ones

    def test_coach_marked_phases_are_kept_and_shown(self):
        e, obs, _ = scene_entry('amateur_coach_spans', 'spans')
        self.assertTrue(e['context']['phaseAvailable'])
        coach = [o for r in e['skills'].values() for o in r['opportunities'] if o['rel']['coachPhase']]
        self.assertTrue(coach)
        self.assertEqual(e['context']['footage'], 'amateur')
        self.assertEqual(e['context']['format'], '7v7')
        entries = [copy.deepcopy(e) for _ in range(3)]
        for k, x in enumerate(entries):
            x['video'] = f'vid{k}'
            x['matchDate'] = f'2026-10-0{k+1}'
        det = PG.mastery(entries, block(4, players=7), squad_id=SQ)
        shares = [s['coachMarkedShare'] for s in det['skills'].values() if s['opportunities']]
        self.assertTrue(shares and all(x == 1.0 for x in shares), shares)
        self.assertIn('coach', det['note'])

    def test_a_short_scene_gives_few_opportunities_and_no_state(self):
        e, _, _ = scene_entry('short_segment')
        self.assertLessEqual(e['range']['t1']-e['range']['t0'], 61.0)
        det = PG.mastery([e], block(4), squad_id=SQ)
        for s in det['skills'].values():
            self.assertEqual(s['state'], 'NOT ENOUGH EVIDENCE')
        self.assertEqual(det['shortClips'], 1)


class Mastery(unittest.TestCase):
    def test_not_enough_evidence(self):
        self.assertEqual(state(history([(6, 6), (6, 6), (7, 7)]))['state'], 'NOT ENOUGH EVIDENCE')   # 19 judged
        s = state(history([(40, 40)]))                                          # one qualifying match
        self.assertEqual((s['state'], s['reason']), ('NOT ENOUGH EVIDENCE', 'few_matches'))
        s = state(history([(30, 30), (5, 5)]))                                  # the clip does not qualify
        self.assertEqual(s['qualifyingMatches'], 1)
        self.assertEqual(s['state'], 'NOT ENOUGH EVIDENCE')
        self.assertEqual(state(history([(6, 6), (6, 6), (8, 8)]))['state'], 'CONSISTENT')         # 20 judged

    def test_learning_and_developing_follow_the_lower_bound(self):
        s = state(history([(5, 10)]*3))                                        # 15 / 30: lower .386 < .6
        self.assertEqual(s['state'], 'LEARNING')
        self.assertAlmostEqual(s['lower'], .386, places=3)
        s = state(history([(15, 20)]*3))                                       # 45 / 60: .672, recent .672 < .75
        self.assertEqual(s['state'], 'DEVELOPING')

    def test_consistent_needs_three_recent_matches_each_above_s1(self):
        self.assertEqual(state(history([(27, 30)]*3))['state'], 'CONSISTENT')  # 81 / 90: .852 >= .75
        self.assertEqual(state(history([(27, 30)]*2))['state'], 'DEVELOPING')  # only two qualifying matches
        s = state(history([(40, 40), (40, 40), (3, 6)]))                       # pooled .93 but one match at .5
        self.assertEqual(s['state'], 'DEVELOPING')
        self.assertFalse(s['recent']['consistent'])
        # recent consistency wins over an older poor run inside the window
        s = state(history([(3, 10), (3, 10), (27, 30), (27, 30), (27, 30)]))
        self.assertEqual(s['state'], 'CONSISTENT')

    def test_adaptable_needs_two_contexts_each_consistent(self):
        one = state(history([(27, 30)]*3))
        self.assertEqual(one['state'], 'CONSISTENT')
        two = state(history([(27, 30)]*3, zone=lambda k: ('own_third', 'middle_third')[k % 2]))
        self.assertEqual(two['state'], 'ADAPTABLE')
        self.assertTrue(all(c['consistent'] for c in two['contexts']['zone'].values()))
        # a second context with too few moments does not count, even when all 5 demonstrate (lower bound .753)
        few = [entry(f'v{k}', f'2026-01-0{k+1}', [1]*28+[-1]*2,
                     zone=(lambda i, m=2-(k == 2): 'middle_third' if i < m else 'own_third')) for k in range(3)]
        s = state(few)
        self.assertEqual(s['contexts']['zone']['middle_third']['opportunities'], 5)
        self.assertGreaterEqual(s['contexts']['zone']['middle_third']['lower'], .75)
        self.assertEqual(s['state'], 'CONSISTENT')

    def test_thresholds_follow_the_level(self):
        h = history([(14, 20)]*3, profiles=[prof(2), prof(4), prof(6)])       # 42 / 60: lower .614
        self.assertEqual(state(h, level=2)['state'], 'DEVELOPING')             # s1 .50 (recent .614 < s2 .65)
        self.assertEqual(state(h, level=4)['state'], 'DEVELOPING')             # s1 .60
        self.assertEqual(state(h, level=6)['state'], 'LEARNING')               # s1 .70
        self.assertEqual(state(history([(14, 20)]*3), level=2)['state'], 'NOT ENOUGH EVIDENCE')   # no L2 outcome
        self.assertEqual(len(PG.mastery(history([(14, 20)]*3), block(2), squad_id=SQ)['separate']['needsRules']), 3)

    def test_unclear_and_out_of_scope_are_not_judged(self):
        e = entry('v1', '2026-10-01', [1]*10+[0]*10+[None]*10)
        s = state([e])
        self.assertEqual((s['opportunities'], s['unclear'], s['outOfScope']), (10, 10, 10))

    def test_low_quality_moments_are_left_out(self):
        h = history([(30, 30)]*3, quality=lambda k: 'low' if k % 3 == 0 else 'high')
        s = state(h)
        self.assertEqual(s['opportunities'], 60)
        self.assertEqual(s['lowQuality'], 30)

    def test_identity_counts_only_where_a_player_is_the_subject(self):
        q = {'position': 'high', 'identity': 'low', 'visibility': 'complete', 'ball': 'reliable', 'samples': 5}
        for d in S.DETECTORS:
            want = 'high' if d in AS.TEAM_LEVEL else 'low'
            self.assertEqual(PG._opp_quality({'detector': d, 'quality': q}), want, d)
        self.assertEqual(PG._opp_quality({'detector': 'behind_ball', 'quality': dict(q, position='medium')}), 'medium')

    def test_the_window_keeps_the_recent_qualifying_matches(self):
        h = history([(0, 30)]*4+[(15, 20)]*5)
        s = state(h)
        self.assertEqual(s['qualifyingMatches'], 5)
        self.assertEqual(s['opportunities'], 100)
        self.assertEqual(sum(1 for r in s['history'] if r['inWindow']), 5)
        self.assertEqual(s['state'], 'DEVELOPING')
        self.assertIn('most recent 5 qualifying matches', PG.mastery(h, block(4), squad_id=SQ)['note'])
        self.assertNotIn('most recent', PG.mastery(h[-5:], block(4), squad_id=SQ)['note'])

    def test_trend(self):
        self.assertEqual(state(history([(27, 30)]))['trend'], 'unknown')
        self.assertEqual(state(history([(10, 30), (10, 30), (27, 30)]))['trend'], 'up')
        self.assertEqual(state(history([(27, 30), (27, 30), (10, 30)]))['trend'], 'down')
        self.assertEqual(state(history([(20, 30), (20, 30), (22, 30)]))['trend'], 'flat')
        self.assertEqual(state(history([(7, 10), (6, 8)]))['trend'], 'flat')      # a 0.05 change is not a trend
        # .67 -> .83 on 30 + 30 moments: z 1.49, a change at 80% but not at the trend's 90%
        s = state(history([(20, 30), (25, 30)]))
        self.assertEqual(s['trend'], 'flat')
        self.assertAlmostEqual(s['trendDetail']['z'], 1.49, places=2)
        # .80 -> .85 on 1000 + 1000 moments: z 2.9, but a 0.05 change is below the minimum change of 0.1
        s = state(history([(800, 1000), (850, 1000)]))
        self.assertGreater(s['trendDetail']['z'], 2.5)
        self.assertEqual(s['trend'], 'flat')

    def test_a_level_change_re_reads_the_same_history(self):
        e, _, _ = scene_entry('defending', 'press_no_cover')
        entries = []
        for k in range(3):
            x = copy.deepcopy(e)
            x['video'], x['matchDate'] = f'v{k}', f'2026-10-0{k+1}'
            entries.append(x)
        before = json.dumps(entries, sort_keys=True)
        l1 = state(entries, level=1)
        l4 = state(entries, level=4)
        self.assertEqual(json.dumps(entries, sort_keys=True), before)        # reading never changes an entry
        self.assertGreater(l1['share'], l4['share'])     # level 1 asks "go to the ball", level 4 also cover
        self.assertEqual(l1['name'], 'Go to the ball and help a teammate')
        self.assertEqual(l4['name'], 'Pressure and cover')


class Normalisation(unittest.TestCase):
    """DESIGN 13.4 #14: a 60 s clip and a 90 minute match with the same per-opportunity behaviour read the same."""

    def test_equal_behaviour_gives_equal_shares_and_rates(self):
        clip = entry('clip', '2026-10-01', [1]*6+[-1]*2, phase=30.0, length=60.0)              # 8 in 30 s
        match = entry('match', '2026-10-08', [1]*540+[-1]*180, phase=2700.0, length=5400.0)   # 720 in 45 min
        a, b = state([clip]), state([match])
        self.assertEqual(a['share'], b['share'])
        self.assertEqual(a['rates']['opportunitiesPer10'], b['rates']['opportunitiesPer10'])
        self.assertEqual(a['rates']['notDemonstratedPer10'], b['rates']['notDemonstratedPer10'])
        self.assertEqual(a['rates']['opportunitiesPer10'], 160.0)        # 8 per 30 s of the relevant phase
        self.assertNotEqual(a['opportunities'], b['opportunities'])
        both = state([match, clip])
        self.assertEqual(both['trend'], 'flat')                    # 720 -> 8 moments is not a decline
        self.assertEqual(both['share'], .75)

    def test_clips_and_full_matches_reach_the_same_state(self):
        full = history([(27, 30)]*3)
        clips = []
        for k in range(3):                                         # each match cut into 60 s clips of 6 moments
            for c in range(5):
                outs = [1]*6 if c < 2 else ([1]*5+[-1] if c < 5 else [])
                clips.append(entry(f'v{k:03d}', f'2026-01-{1+k:02d}', outs, t0=60.0*c, length=60.0, phase=30.0))
        a, b = state(full), state(clips)
        self.assertEqual((a['opportunities'], a['demonstrated']), (b['opportunities'], b['demonstrated']))
        self.assertEqual(a['state'], b['state'])
        self.assertEqual(b['matches'], 3)                          # clips of one video are one match

    def test_a_short_clip_never_enters_a_comparison(self):
        h = history([(27, 30)]*3)+[entry('clip', '2026-12-01', [-1]*5, length=60.0, phase=30.0)]
        s = state(h)
        self.assertEqual(s['state'], 'CONSISTENT')                 # 5 failing moments of a clip: not a match
        self.assertEqual(s['trend'], 'flat')
        self.assertEqual(s['opportunities'], 95)                   # but its moments are counted


class Store(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)/'runs'

    def tearDown(self):
        self.tmp.cleanup()

    def stored(self):
        return PG.read_entries(self.root)

    def test_upsert_replaces_the_same_range_and_keeps_the_longest_overlap(self):
        a = entry('v1', '2026-10-01', [1]*8, t0=0.0, length=600.0)
        r = PG.upsert(self.root, a)
        self.assertTrue(r['written'])
        a2 = copy.deepcopy(a)
        a2['skills']['pressure_and_cover']['opportunities'] = a2['skills']['pressure_and_cover']['opportunities'][:3]
        r = PG.upsert(self.root, a2)                                # a regeneration replaces
        self.assertEqual(r['replaced'], [a['id']])
        self.assertEqual(len(self.stored()), 1)
        self.assertEqual(len(self.stored()[0]['skills']['pressure_and_cover']['opportunities']), 3)
        short = entry('v1', '2026-10-01', [1]*2, t0=100.0, length=60.0, run='run-2')
        r = PG.upsert(self.root, short)                             # inside a longer range of the same video
        self.assertEqual((r['written'], r['keptLonger']), (False, a['id']))
        longer = entry('v1', '2026-10-01', [1]*9, t0=300.0, length=900.0, run='run-3')
        r = PG.upsert(self.root, longer)
        self.assertEqual((r['written'], r['replaced']), (True, [a['id']]))
        touching = entry('v1', '2026-10-01', [1]*2, t0=1200.0, length=60.0)
        self.assertTrue(PG.upsert(self.root, touching)['written'])  # only touching: both kept
        other_video = entry('v2', '2026-10-02', [1]*2, t0=300.0, length=60.0)
        other_squad = entry('v1', '2026-10-01', [1]*2, t0=300.0, length=60.0,
                            squad=S.record_id('sq', 'other', NOW))
        self.assertTrue(PG.upsert(self.root, other_video)['written'])
        self.assertTrue(PG.upsert(self.root, other_squad)['written'])
        self.assertEqual(len(self.stored()), 4)
        self.assertEqual(len(PG.read_entries(self.root, SQ)), 3)
        self.assertEqual([p.name for p in PG.store_dir(self.root).iterdir()], ['progress.json'])   # no temp files

    def test_strict_read_before_a_write_never_wipes_the_file(self):
        PG.upsert(self.root, entry('v1', '2026-10-01', [1]*3))
        path = PG.progress_path(self.root)
        for garbage in ('{"version": 1, "entries": [', '{"version": 1, "entries": [{"t": NaN}]}',
                        '{"version": 7, "entries": []}', '[]'):
            path.write_text(garbage, encoding='utf-8')
            with self.assertRaises(PG.ProgressUnreadable):
                PG.upsert(self.root, entry('v2', '2026-10-02', [1]))
            self.assertEqual(path.read_text(encoding='utf-8'), garbage)
            self.assertEqual(PG.read_entries(self.root), [])        # a display read shows nothing, no error

    def test_a_malformed_entry_is_skipped_on_display_and_kept_on_write(self):
        good = entry('v1', '2026-10-01', [1]*3)
        PG.progress_path(self.root).parent.mkdir(parents=True)
        PG.progress_path(self.root).write_text(json.dumps({'version': 1, 'entries': [good, {'id': 'odd'}]}),
                                               encoding='utf-8')
        self.assertEqual([e['id'] for e in PG.read_entries(self.root)], [good['id']])
        PG.upsert(self.root, entry('v2', '2026-10-02', [1]))
        raw = json.loads(PG.progress_path(self.root).read_text(encoding='utf-8'))
        self.assertIn({'id': 'odd'}, raw['entries'])

    def test_the_parsed_file_is_reused_until_it_changes(self):
        a = entry('v1', '2026-10-01', [1]*3)
        PG.upsert(self.root, a)
        first = PG.read_entries(self.root)
        self.assertIs(PG.read_entries(self.root)[0], first[0])         # this process's own write: no re-parse
        PG.mark_stale(self.root, video='v1', run='run-1', team='A', measurement='f'*16)
        self.assertNotIn('stale', first[0])                             # a writer never changes a returned entry
        self.assertTrue(PG.read_entries(self.root)[0]['stale'])
        other = entry('v2', '2026-10-02', [1])                          # another process writes the file
        path = PG.progress_path(self.root)
        time.sleep(.01)
        path.write_text(json.dumps({'version': 1, 'entries': [other]}), encoding='utf-8')
        os.utime(path, ns=(time.time_ns(), time.time_ns()+10**9))
        self.assertEqual([e['video'] for e in PG.read_entries(self.root)], ['v2'])

    def test_a_non_finite_number_blocks_the_write(self):
        PG.upsert(self.root, entry('v1', '2026-10-01', [1]*3))
        path = PG.progress_path(self.root)
        text = path.read_text(encoding='utf-8').replace('"phaseSeconds":600.0', '"phaseSeconds":1e999', 1)
        path.write_text(text, encoding='utf-8')
        os.utime(path, ns=(time.time_ns(), time.time_ns()+2*10**9))
        self.assertEqual(PG.read_entries(self.root), [])                # not valid for display
        with self.assertRaises(PG.ProgressUnreadable):
            PG.upsert(self.root, entry('v2', '2026-10-02', [1]))
        self.assertEqual(path.read_text(encoding='utf-8'), text)

    def test_remove_mark_stale_and_dates(self):
        a = entry('v1', '2026-10-01', [1]*3, measurement='a'*16)
        b = entry('v1', '2026-10-01', [1]*3, t0=6000.0, measurement='b'*16)
        c = entry('v2', '2026-10-02', [1]*3)
        for e in (a, b, c):
            PG.upsert(self.root, e)
        n = PG.mark_stale(self.root, video='v1', run='run-1', team='A', measurement='b'*16)
        self.assertEqual(n, 1)
        st = {e['id']: e.get('stale', False) for e in self.stored()}
        self.assertEqual((st[a['id']], st[b['id']], st[c['id']]), (True, False, False))
        self.assertEqual(PG.mark_stale(self.root, video='v1', run='run-1', team='A', measurement='b'*16), 0)
        self.assertEqual(PG.set_match_date(self.root, squad_id=SQ, video='v2', day='2026-09-30'), 1)
        self.assertEqual([e['matchDate'] for e in self.stored() if e['video'] == 'v2'], ['2026-09-30'])
        with self.assertRaises(PG.ProgressError):
            PG.set_match_date(self.root, squad_id=SQ, video='v2', day='30/09/2026')
        self.assertEqual(PG.remove(self.root, squad_id=SQ, video='v1'), 2)
        self.assertEqual([e['video'] for e in self.stored()], ['v2'])

    def test_squads(self):
        s, made = PG.create_squad(self.root, 'U10 Blue', 'U10', 2, {'players': 7}, now=NOW)
        self.assertTrue(made)
        self.assertRegex(s['id'], r'sq-[0-9a-f]{12}')
        self.assertEqual(s['format'], {'players': 7, 'goalkeeper': True})
        again, made = PG.create_squad(self.root, ' u10 blue ', 'U12', 3, {'players': 9}, now=NOW)
        self.assertEqual((again['id'], made), (s['id'], False))
        for bad in (('', 'U10', 2, {'players': 7}), ('X', 'U99', 2, {'players': 7}), ('X', 'U10', 9, {'players': 7}),
                    ('X', 'U10', 2, {'players': 6}), ('Caf\u00e9', 'U10', 2, {'players': 7})):
            with self.assertRaises(PG.ProgressError, msg=bad):
                PG.create_squad(self.root, *bad, now=NOW)
        u = PG.update_squad(self.root, s['id'], level=3, age_group='U11')
        self.assertEqual((u['level'], u['ageGroup'], u['id']), (3, 'U11', s['id']))
        PG.create_squad(self.root, 'U12 Red', 'U12', 3, {'players': 9}, now=NOW)
        with self.assertRaises(PG.ProgressError):
            PG.update_squad(self.root, s['id'], name='u12 red')
        self.assertEqual(S.validate('squads', {'version': 1, 'squads': PG.read_squads(self.root)}), [])
        self.assertEqual(PG.squad(self.root, s['id'])['level'], 3)


class LinksAndRecord(unittest.TestCase):
    def settings(self, link=True, confirmed=True, team='A', run='run-1'):
        b = PR.clean_team({'ageGroup': 'U14', 'playingLevel': 4})
        b['squad'] = {'id': SQ, 'name': 'U14 Blue'}
        b['runLinks'] = [{'run': run, 'team': team, 'confirmed': confirmed}] if link else []
        st = {'version': 1, 'teams': {'A': b, 'B': None}, 'updated': None}
        self.assertEqual(S.validate('settings', st), [])
        return st

    def test_links(self):
        self.assertEqual(PG.confirmed_team(self.settings(), 'run-1', SQ), 'A')
        self.assertIsNone(PG.confirmed_team(self.settings(confirmed=False), 'run-1', SQ))
        self.assertIsNone(PG.confirmed_team(self.settings(), 'run-2', SQ))
        x = PG.links(self.settings(link=False), 'run-1')[0]
        self.assertEqual((x['team'], x['confirmed'], x['question']), (None, False,
                                                                      'Which team is U14 Blue in this run?'))
        self.assertEqual(PG.confirmed_team(self.settings(team='B'), 'run-1', SQ), 'B')

    def test_record_writes_only_confirmed_links(self):
        obs, roles = FX.tiers('defending', 'press_no_cover')
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out = PG.record(root, self.settings(confirmed=False), run='run-1', video='v1', tiers={'A': (obs, roles)},
                            coach_date='2026-10-01', now=NOW)
            self.assertEqual([(r['written'], r['reason']) for r in out], [(False, 'unconfirmed')])
            self.assertFalse(PG.progress_path(root).exists())
            out = PG.record(root, self.settings(team='B'), run='run-1', video='v1', tiers={'A': (obs, roles)},
                            coach_date='2026-10-01', now=NOW)
            self.assertEqual(out[0]['reason'], 'no_observations')
            video = root/'match.mp4'
            video.write_bytes(b'x')
            os.utime(video, (1760000000, 1760000000))
            out = PG.record(root, self.settings(), run='run-1', video='v1', tiers={'A': (obs, roles)},
                            video_path=video, now=NOW)
            self.assertTrue(out[0]['written'])
            e = PG.read_entries(root, SQ)[0]
            self.assertEqual((e['dateSource'], e['team'], e['context']['level']), ('file_mtime', 'A', 4))
            self.assertRegex(e['matchDate'], r'2025-10-0[89]')
            with self.assertRaises(PG.ProgressError):
                PG.record(root, self.settings(), run='run-1', video='v1', tiers={'A': (obs, roles)}, now=NOW)


class StaleAndRules(unittest.TestCase):
    def test_earlier_engines_are_listed_apart(self):
        h = history([(27, 30)]*3)
        old = entry('v-old', '2025-01-01', [-1]*30, stale=True)
        other = entry('v-ver', '2025-01-02', [-1]*30)
        other['engine']['version'] = '3'                 # (not a valid stored entry: read_entries skips it)
        det = PG.mastery(h+[old, other], block(4), squad_id=SQ)
        self.assertEqual(det['skills']['pressure_and_cover']['state'], 'CONSISTENT')
        self.assertEqual(sorted(x['video'] for x in det['separate']['earlierEngine']), ['v-old', 'v-ver'])
        self.assertIn('earlier engine', det['note'])
        self.assertEqual(PG.needs_rederive(old), 'engine')

    def test_missing_outcomes_are_re_derived_not_guessed(self):
        e, obs, roles = scene_entry('unsupported_advance', 'none_behind')
        broken = copy.deepcopy(e)
        o = broken['skills']['pressure_and_cover']['opportunities'][0]
        key = PG.outcome_key('pressure_cover', prof(4)['detectors']['pressure_cover']['params'])
        del o['rel'][key]
        self.assertEqual(PG.needs_rederive(broken), 'rules')
        self.assertIsNone(PG.outcomes(broken, prof(4), 'pressure_and_cover'))
        det = PG.mastery([broken], block(4), squad_id=SQ)
        self.assertEqual([x['id'] for x in det['separate']['needsRules']], [e['id']])
        again = PG.rederive(broken, obs, roles, updated=NOW)
        self.assertEqual(again, e)
        stale = copy.deepcopy(e)
        stale['stale'] = True
        self.assertNotIn('stale', PG.rederive(stale, obs, roles, updated=NOW))
        obs_b, roles_b = FX.tiers('unsupported_advance', 'none_behind', team='B')
        with self.assertRaises(PG.ProgressError):
            PG.rederive(e, obs_b, roles_b)


class Section(unittest.TestCase):
    def _report_with(self, progression):
        rep = S.example('report')
        rep['progression'] = progression
        return S.validate('report', rep)

    def test_section_follows_the_level_and_validates(self):
        e, _, _ = scene_entry('unsupported_advance', 'none_behind')
        entries = []
        for k in range(3):
            x = copy.deepcopy(e)
            x['video'], x['matchDate'] = f'v{k}', f'2026-10-0{k+1}'
            entries.append(x)
        ref = {'id': SQ, 'name': 'U14 Blue'}
        shown = {}
        for level in (1, 2, 4, 7):
            sec = PG.section(entries, ref, block(level))
            self.assertEqual(self._report_with(sec), [], level)
            self.assertLessEqual(len(sec['note']), PG.NOTE_MAX)
            self.assertEqual(S.event_problems(sec['note']), [])
            self.assertTrue(sec['note'].isascii())
            shown[level] = [r['skill'] for r in sec['skills']]
            for r in sec['skills']:
                self.assertEqual(r['name'], prof(level)['skills'][r['skill']])
        self.assertEqual(shown[1], ['pressure_and_cover', 'someone_back_when_attacking', 'spread_when_attacking'])
        self.assertIn('wide_defender_support', shown[4])
        self.assertNotIn('spread_when_attacking', shown[7])         # describe-only at level 7
        sec = PG.section(entries, ref, block(4), linked=False)
        self.assertIn('Confirm which team is U14 Blue', sec['note'])

    def test_rows_count_judged_moments_and_matches(self):
        sec = PG.section(history([(27, 30)]*3), {'id': SQ, 'name': 'U14 Blue'}, block(4))
        row = next(r for r in sec['skills'] if r['skill'] == 'pressure_and_cover')
        self.assertEqual((row['state'], row['opportunities'], row['matches'], row['trend']),
                         ('CONSISTENT', 90, 3, 'flat'))
        self.assertTrue(sec['note'].startswith(PG.load_rules()['label']))

    def test_read_block_prefers_the_viewed_runs_settings(self):
        sq = {'id': SQ, 'name': 'U14 Blue', 'ageGroup': 'U14', 'level': 4, 'format': {'players': 11,
                                                                                      'goalkeeper': True},
              'created': NOW}
        self.assertEqual(PG.read_block(sq)['playingLevel'], 4)
        b = PG.read_block(sq, {'ageGroup': 'U15', 'playingLevel': 5, 'style': 'counter'})
        self.assertEqual((b['ageGroup'], b['playingLevel'], b['style']), ('U15', 5, 'counter'))


class Cost(unittest.TestCase):
    def test_reading_a_season_is_fast(self):
        season = []
        for k in range(30):
            outs = [1]*150+[-1]*60+[0]*30
            season.append(entry(f'v{k:03d}', f'2026-{1+k//28:02d}-{1+k % 28:02d}', outs))
        t = time.perf_counter()
        PG.section(season, {'id': SQ, 'name': 'U14 Blue'}, block(4))
        self.assertLess(time.perf_counter()-t, 1.0)


if __name__ == '__main__':
    unittest.main()
