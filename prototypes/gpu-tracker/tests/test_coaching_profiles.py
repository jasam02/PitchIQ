"""The coaching profile layer (soccer/coaching/profiles.py, profiles.json, sources.json) and its sources build
(tools/build_coaching_sources.py).

What is pinned here and why:
- the sources registry is exactly what the build makes from the three research copies (a hand edit or a changed copy
  fails), every research id and every spelling DESIGN uses resolves to one canonical source, each URL belongs to one
  source, and the research copies are hashed independently of line endings;
- invariant 5 and the honesty rules of DESIGN 2: every number in profiles.json sits under a PITCHIQ-PROPOSAL /
  validated false section, every FEDERATION item cites a source that was read and names the research item it
  copies (verbatim unless marked edited), and the level rules hold (no work_on at levels 1-2, no try_this from 4);
  each rule is shown to fail on a mutated table, so the checks are not vacuous;
- the safety enumeration of DESIGN 2.4 over every age group x level x format x style: no heading at U11 or younger
  (stricter of U.S. Soccer and England), heading caps no looser than either federation's from U12, physical load and
  tackling within the age caps, no heading words in the coaching content of a U11-or-younger profile, and drills of
  the pool safe for the age once the v2 drill library exists;
- resolve: every combination is a valid schema profile; the level, age, format and style each change what they are
  meant to change (DESIGN 2.2-2.6) and nothing else; role-tier and serve-time settings never change the profile; the
  user's defaults (U.S. formats by age; nothing set = Adult, level 6, 11v11 with a notice);
- clean_settings never raises and always returns valid settings, drops what is malformed and says so, and leaves a
  valid file unchanged;
- the catalog for the setup form, the role names per format (no 11v11 names on small formats) and the helpers the
  other roles call (heading_rule, reach_speed_cap, unavailable_for, source_refs, canonical_source);
- the real profile stage runs inside coaching.interpret (the other stages are the schema tests' stand-ins);
- files are ASCII with CRLF line endings.

Run:  python -m unittest test_coaching_profiles          (from this directory)
"""
import copy
import json
import random
import re
import sys
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'tools'))
import build_coaching_sources as B  # noqa: E402

import soccer.coaching as coaching  # noqa: E402
from soccer.coaching import profiles as PR  # noqa: E402
from soccer.coaching import schema as S  # noqa: E402

FORMATS = [{'players': 4, 'goalkeeper': False}, {'players': 4, 'goalkeeper': True},
           {'players': 5, 'goalkeeper': True}, {'players': 7, 'goalkeeper': True},
           {'players': 9, 'goalkeeper': True}, {'players': 11, 'goalkeeper': True}]
HEAD_WORD = re.compile(r'\bhead(?:s|ing|er|ers|ed)?\b', re.I)
RESEARCH = B.load_research()


def table():
    """A deep copy of the validated profile table without the cache's '_sha' key."""
    data = copy.deepcopy(PR.load())
    data.pop('_sha', None)
    return data


def sources_doc():
    return PR.load_sources()['doc']


def problems(data, research=None):
    return PR.validate_profiles(data, sources_doc(), research=research)


def every_profile():
    for age in S.AGE_GROUPS:
        for level in range(1, 8):
            for fmt in FORMATS:
                for style in S.STYLES:
                    yield (age, level, fmt, style), PR.resolve({'ageGroup': age, 'playingLevel': level,
                                                                 'format': fmt, 'style': style})


class SourcesRegistryTests(unittest.TestCase):
    def test_sources_json_is_the_build_of_the_research_copies(self):
        self.assertEqual(json.loads(PR.SOURCES_PATH.read_bytes()), B.build(research=RESEARCH))
        self.assertEqual(B.check(), [])

    def test_merge_counts(self):
        doc = sources_doc()
        self.assertEqual(len(doc['sources']), 122)
        self.assertEqual(sum(len(d['sources']) for _n, d, _r in RESEARCH), 165)
        reads = [s['read'] for s in doc['sources']]
        self.assertEqual((reads.count('full'), reads.count('partial'), reads.count('not_read')), (90, 26, 6))

    def test_every_research_id_and_design_spelling_resolves(self):
        for name, data, _raw in RESEARCH:
            for s in data['sources']:
                self.assertIsNotNone(PR.canonical_source(s['id']), (name, s['id']))
        for spelling, cid in (('US_CG_U78_PDF', 'USS-CG-U78'), ('USS-GM17', 'USS-GM-U17'), ('CA-GAMES', 'CS-GAMES'),
                              ('EF_12_16_SERIES', 'EF-COVER-RECOVER-SESSION'), ('EF-CR26', 'EF-COVER-RECOVER-SESSION'),
                              ('ef_cover_recover', 'EF-COVER-RECOVER'), ('EF-PTC', 'EF-PTC'), ('UEFA-PRO', 'UEFA-PRO'),
                              ('US_F11_PAGE', 'USS-F-11V11'), ('USS-FORM11', 'USS-F-11V11')):
            self.assertEqual(PR.canonical_source(spelling), cid, spelling)
        self.assertIsNone(PR.canonical_source('NOT-A-SOURCE'))

    def test_design_ids_resolve(self):
        design = ('CA-GAMES CA_PRINCIPLES_PDF CA_STANDARDS_PDF CS-ACEP CS-CPRO CS-GCE CS-PDP CS-W2WC EF-11V11 '
                  'EF-COVER-RECOVER EF-COVER-RECOVER-SESSION EF-CRS EF-DLE-ROLES EF-DLE-TRANS EF-DLE-WIDE EF-PCB25 '
                  'EF-PHASES EF-PRACTISING-PRESS EF-PROG-7-11 EF-PROG-PRESS-COVER EF-PTC EF-PZ EF-SUPPORT-LINE '
                  'EF-SWIFT-SUPPORT EF_12_16_SERIES EF_CHALLENGE_RECOVER EF_COVER_RECOVER EF_COVER_RECOVER_SESSION '
                  'EF_PROG_7_11 EF_PROG_PRESS_COVER EF_SWIFT_SUPPORT FA_DNA_PRIORITIES UEFA-A UEFA-B UEFA-CONV '
                  'UEFA-FALSEFB UEFA-PRO UEFA-TR2223 UEFA-YB USS-AGG1516 USS-AGG17 USS-CG-U910 USS-F-4V4 USS-F-7V7 '
                  'USS-F-9V9 USS-GM-U78 USS-GM1516 USS-GM17 US_AGE_U78 US_CG_U1112_PDF US_CG_U1314_PDF US_CG_U78_PDF '
                  'US_CG_U910_PDF US_F11_PDF US_F4_PDF US_F7_PDF US_F9_PDF US_GM_U1112_PDF US_GM_U1314_PDF '
                  'US_GM_U78_PDF US_GM_U910_PDF').split()
        self.assertEqual([i for i in design if PR.canonical_source(i) is None], [])

    def test_each_url_belongs_to_one_source_and_read_is_the_best_reading(self):
        seen = {}
        for s in sources_doc()['sources']:
            for u in [s['url']]+s['extraUrls']:
                key = B.norm_url(u)
                self.assertNotIn(key, seen, f'{u} in {seen.get(key)} and {s["id"]}')
                seen[key] = s['id']
            best = max((B.STATUS[r['status']] for r in s['readings']), key=B.READ_ORDER.get)
            self.assertEqual(s['read'], best, s['id'])
            self.assertEqual(s['accessed'], '2026-10-21')

    def test_the_manual_merge_is_one_article(self):
        wide = PR.load_sources()['byId']['EF-DLE-WIDE']
        self.assertEqual({r['id'] for r in wide['readings']}, {'EF_DEFEND_WIDE', 'EF-DLE-WIDE'})
        self.assertEqual(len([wide['url']]+wide['extraUrls']), 2)

    def test_research_hash_ignores_line_endings(self):
        raw = RESEARCH[0][2]
        lf = raw.replace(b'\r\n', b'\n')
        self.assertEqual(B.canonical_sha(lf), B.canonical_sha(lf.replace(b'\n', b'\r\n')))
        inputs = {i['pass']: i['contentSha256'] for i in sources_doc()['inputs']}
        for name, _data, raw in RESEARCH:
            self.assertEqual(inputs[name], B.canonical_sha(raw))

    def test_a_hand_edit_or_a_conflict_fails(self):
        doc = copy.deepcopy(sources_doc())
        doc['sources'][0]['read'] = 'full' if doc['sources'][0]['read'] != 'full' else 'partial'
        self.assertNotEqual(doc, B.build(research=RESEARCH))
        bad = copy.deepcopy(sources_doc())
        bad['sources'][1]['aliases'].append(bad['sources'][0]['id'])
        self.assertTrue(PR.validate_sources(bad))
        saved = dict(B.CANON)
        try:
            B.CANON['US_F11_PAGE'] = 'USS-F-11V11-PAGE'      # two canonical ids for one merged source
            with self.assertRaises(ValueError):
                B.merge(RESEARCH)
        finally:
            B.CANON.clear()
            B.CANON.update(saved)

    def test_unreadable_gaps_are_recorded(self):
        gaps = sources_doc()['unreadable']
        self.assertEqual(len(gaps), 22)
        self.assertTrue(any('USS-FORMATS-PDF' in g['sources'] for g in gaps))
        for g in gaps:
            for sid in g['sources']:
                self.assertIn(sid, PR.load_sources()['byId'])

    def test_source_refs_are_report_records(self):
        refs = PR.source_refs(['US_GM_U78_PDF', 'USS-GM-U78', 'EF-PTC', 'nope'])
        self.assertEqual([r['id'] for r in refs], ['EF-PTC', 'USS-GM-U78'])
        report = S.example('report')
        report['sources'] = refs
        self.assertEqual([p for p in S.validate('report', report) if '.sources' in p], [])


class ProfileTableTests(unittest.TestCase):
    def test_the_table_validates_with_the_research_copies(self):
        self.assertEqual(problems(table(), RESEARCH), [])

    def test_every_number_is_a_pitchiq_proposal(self):
        data = table()
        data['formats']['7']['extraNumber'] = 3
        del data['formats']['origin']
        out = problems(data)
        self.assertTrue(any('formats.7.extraNumber' in p for p in out), out)
        data = table()
        data['levels']['4']['validated'] = True
        self.assertTrue(any('levels.4.evidence' in p and 'PITCHIQ-PROPOSAL' in p for p in problems(data)))

    def test_federation_items_cite_read_sources_and_their_research_item(self):
        data = table()
        fed = [g for lv in data['levels'].values() for part in PR.GUIDANCE_PARTS for g in lv[part]
               if g['tag'] == 'FEDERATION']
        self.assertGreater(len(fed), 100)
        read = PR.load_sources()['byId']
        for g in fed:
            self.assertTrue(g['sources'] and g['research'], g['text'][:60])
            for sid in g['sources']:
                self.assertNotEqual(read[sid]['read'], 'not_read', sid)
        mutations = [
            (lambda d: d['levels']['3']['principles'][0].update(sources=['USS-F-SELECT']), 'was not read'),
            (lambda d: d['levels']['3']['principles'][0].update(sources=[]), 'needs its source ids'),
            (lambda d: d['levels']['3']['principles'][0].update(sources=['UEFA-PRO']), "research item's sources"),
            (lambda d: d['levels']['1']['objectives'][0].update(text='Something else.'), 'differs from the research'),
            (lambda d: d['levels']['1']['objectives'][0].pop('research'), 'names the research item'),
            (lambda d: d['levels']['1']['objectives'][0].update(research='L1L2:levels.nowhere[0]'), 'not found'),
            (lambda d: d['levels']['2']['objectives'][0].update(tag='PITCHIQ-PROPOSAL'), 'the research item is'),
            (lambda d: d['levels']['2']['objectives'][0].update(text='A "'+'x'*61+'" quote.', edited=True),
             'quoted phrase'),
            (lambda d: d['levels']['2']['objectives'][0].update(text='caf'+chr(233), edited=True), 'printable ASCII'),
        ]
        for mutate, expect in mutations:
            data = table()
            mutate(data)
            out = problems(data, RESEARCH)
            self.assertTrue(any(expect in p for p in out), (expect, out))

    def test_edited_items_are_marked_and_short(self):
        for lv in table()['levels'].values():
            for part in PR.GUIDANCE_PARTS:
                for g in lv[part]:
                    if g.get('edited'):
                        self.assertTrue(g['research'])
                        self.assertLessEqual(len(g['text']), 400)

    def test_level_verdict_rules_are_enforced(self):
        for level, verdict, expect in ((1, 'work_on', 'never work_on'), (2, 'work_on', 'never work_on'),
                                       (4, 'try_this', 'level 1-3 framing'), (7, 'try_this', 'level 1-3 framing')):
            data = table()
            data['levels'][str(level)]['detectors']['behind_ball']['verdicts'].append(verdict)
            self.assertTrue(any(expect in p for p in problems(data)), (level, verdict))
        data = table()
        data['ages']['groups']['U11']['heading'] = 'u12_u13'
        self.assertTrue(any('no heading at U11' in p for p in problems(data)))
        data = table()
        data['formats']['7']['templates'] = True
        self.assertTrue(any('never use team_shape templates' in p for p in problems(data)))
        data = table()
        data['formats']['9']['formations'].pop()
        self.assertTrue(any('formations: must be exactly' in p for p in problems(data)))

    def test_formats_match_the_schema_and_federation_items_have_sources(self):
        data = table()
        for key in PR.FORMAT_KEYS:
            f = data['formats'][key]
            self.assertEqual(sorted(m['name'] for m in f['formations']), sorted(S.FORMATIONS[int(key)]))
            for m in f['formations']:
                if m['tag'] == 'FEDERATION':
                    self.assertTrue(m['sources'])
        self.assertEqual([m['name'] for m in data['formats']['7']['formations'] if m['tag'] == 'FEDERATION'],
                         ['1-3-2-1'])

    def test_heading_caps_are_the_stricter_of_both_federations(self):
        # U.S. Soccer: at most 30 min and 15-20 headers per player per week from U12; England: one session a week,
        # 5 light unopposed headers at U12-U13, 10 per session at U14-U16, contested from U18 (research items).
        england_per_session = {'U12': 5, 'U13': 5, 'U14': 10, 'U15': 10, 'U16': 10, 'U17': 10, 'U18': 10}
        for age in S.AGE_GROUPS:
            rule = PR.heading_rule(age)
            n = 99 if age == 'Adult' else int(age[1:])
            if n <= 11 or n >= 19:
                self.assertEqual((rule['heading'], rule['caps']), ('none', None), age)
                continue
            caps = rule['caps']
            self.assertEqual(rule['heading'], 'limited', age)
            self.assertLessEqual(caps['minutesPerWeek'], 30)
            self.assertLessEqual(caps['headersPerWeek'], 15)
            self.assertLessEqual(caps['sessionsPerWeek'], 1)
            self.assertLessEqual(caps['headersPerSession'], england_per_session[age])
            self.assertEqual(caps['contested'], n >= 18, age)
            self.assertEqual(caps['lightBall'], n <= 13, age)
            self.assertTrue(rule['sources'])

    def test_format_defaults_follow_the_user_decision(self):
        expect = {'U6': 4, 'U7': 4, 'U8': 4, 'U9': 7, 'U10': 7, 'U11': 9, 'U12': 9, 'U13': 11, 'U14': 11,
                  'U18': 11, 'Adult': 11}
        for age, players in expect.items():
            for fed in ('auto', 'US'):
                f = PR.default_format(age, fed)
                self.assertEqual(f['players'], players, (age, fed))
                self.assertEqual(f['goalkeeper'], players != 4)
        for fed in ('England', 'Canada'):
            self.assertEqual(PR.default_format('U11', fed)['players'], 7)
            self.assertEqual(PR.default_format('U13', fed)['players'], 9)
            self.assertEqual(PR.default_format('U12', fed)['players'], 9)
        self.assertEqual(PR.default_format(None), S.DEFAULT_FORMAT)

    def test_unavailable_records_and_focus(self):
        for level in range(1, 8):
            recs = PR.unavailable_for(level)
            self.assertIn('phase_amateur', [r['category'] for r in recs])
            report = S.example('report')
            report['overview']['unavailable'] = recs
            self.assertEqual([p for p in S.validate('report', report) if 'unavailable' in p], [], level)
            for f in PR.focus_for(level):
                if f['status'] == 'unavailable':
                    self.assertIn(f['unavailable'], [r['category'] for r in recs])
        self.assertIn('passing_support', [r['category'] for r in PR.unavailable_for(3)])
        self.assertNotIn('phase_amateur', [r['category'] for r in PR.unavailable_for(3, phase_available=True)])

    def test_profile_sha_ignores_line_endings_and_follows_content(self):
        data = table()
        self.assertEqual(PR.load()['_sha'], PR.canonical_sha(data)[:16])
        raw = PR.PROFILES_PATH.read_bytes()
        self.assertEqual(json.loads(raw.replace(b'\r\n', b'\n')), json.loads(raw))
        data['levels']['1']['evidence']['minOpportunities'] += 1
        self.assertNotEqual(PR.canonical_sha(data)[:16], PR.load()['_sha'])


class SafetyEnumerationTests(unittest.TestCase):
    """DESIGN 2.4: every (age group x level x format x style) profile."""

    @classmethod
    def setUpClass(cls):
        t = time.perf_counter()
        cls.profiles = list(every_profile())
        cls.seconds = time.perf_counter()-t

    def test_every_combination_is_a_valid_profile(self):
        self.assertEqual(len(self.profiles), len(S.AGE_GROUPS)*7*len(FORMATS)*len(S.STYLES))
        bad = [(k, S.validate('profile', p)[:2]) for k, p in self.profiles if S.validate('profile', p)]
        self.assertEqual(bad[:3], [])

    def test_resolve_is_fast(self):
        # interpretation tier target: < 300 ms for a whole report (DESIGN 1.2); the profile is a small part of it
        self.assertLess(self.seconds/len(self.profiles), .02)

    def test_no_heading_at_u11_or_younger(self):
        for (age, level, fmt, style), p in self.profiles:
            young = S.AGE_ORDER[age] <= S.AGE_ORDER['U11']
            if young:
                self.assertEqual(p['safety']['heading'], 'none', (age, level))
                content = [g['text'] for g in p['objectives']+p['principles']]
                content += [d['framing'] for d in p['detectors'].values() if d['framing']]
                content += p['register']['concepts']+[n['text'] for n in p['notices']]
                content += [v for v in p['skills'].values() if v]
                self.assertEqual([c for c in content if HEAD_WORD.search(c)], [], (age, level))

    def test_load_tackling_and_sprints_follow_the_age(self):
        caps = {'low': 0, 'moderate': 1, 'high': 2}
        for (age, level, fmt, style), p in self.profiles:
            n = 99 if age == 'Adult' else int(age[1:])
            limit = 'low' if n <= 10 else 'moderate' if n <= 14 else 'high'
            self.assertLessEqual(caps[p['safety']['maxLoad']], caps[limit], age)
            self.assertEqual(p['safety']['tackling'], n >= 9, age)
            self.assertEqual(p['safety']['sprintRaces'], n >= 13, age)
            self.assertEqual(p['detectors']['pressure_cover']['params'].get('reachSpeedCapMS'),
                             5.0 if n <= 10 else 6.0 if n <= 14 else 7.0)

    def test_small_formats_never_get_11v11_expectations(self):
        for (age, level, fmt, style), p in self.profiles:
            if fmt['players'] < 11:
                self.assertTrue(p['formation'] is None or p['formation'] in S.FORMATIONS[fmt['players']])
                d3 = p['detectors']['wide_defender_advance']['params']
                if level == 2 and fmt['players'] in (7, 9):
                    self.assertEqual(d3['advancedRule'], 'halfway')
                self.assertIn('small_format_registration', [n['code'] for n in p['notices']])

    def test_drill_pool_is_safe_for_the_age_once_the_v2_library_exists(self):
        lib = json.loads((ROOT/'soccer'/'coaching'/'drills.json').read_bytes())
        if lib.get('version') != 2:
            self.skipTest('drills.json is still the v1 library (DRILLS lands v2 in wave 1)')
        drills = {d['drill_id']: d for d in lib['drills']}
        loads = {'low': 0, 'moderate': 1, 'high': 2}
        for (age, level, fmt, style), p in self.profiles:
            for drill_id in p['drillPool']:
                self.assertIn(drill_id, drills, drill_id)
                safety = drills[drill_id]['safety']
                if S.AGE_ORDER[age] <= S.AGE_ORDER['U11']:
                    self.assertEqual(safety['heading'], 'none', (drill_id, age))


class ResolveTests(unittest.TestCase):
    def test_nothing_set_is_adult_level_6_11v11_with_a_notice(self):
        p = PR.resolve(None)
        self.assertEqual((p['ageGroup'], p['level'], p['format']['name'], p['style']), ('Adult', 6, '11v11',
                                                                                       'unknown'))
        notice = next(n for n in p['notices'] if n['code'] == 'defaults_used')
        self.assertIn('age group, playing level, match format', notice['text'])
        self.assertEqual(p['notices'][0]['code'], 'sources_disclaimer')
        self.assertEqual(p['notices'][0]['text'], sources_doc()['disclaimer'])

    def test_partial_blocks_fill_from_each_other(self):
        p = PR.resolve({'ageGroup': 'U10'})
        self.assertEqual((p['level'], p['format']['name']), (2, '7v7'))
        p = PR.resolve({'playingLevel': 3})
        self.assertEqual((p['ageGroup'], p['format']['name']), ('U12', '9v9'))
        p = PR.resolve({'format': {'players': 7, 'goalkeeper': True}})
        self.assertEqual((p['ageGroup'], p['level']), ('U10', 2))
        p = PR.resolve({'ageGroup': 'U13', 'federation': 'England'})
        self.assertEqual(p['format']['name'], '9v9')
        full = PR.resolve({'ageGroup': 'U14', 'playingLevel': 4, 'format': {'players': 11, 'goalkeeper': True}})
        self.assertNotIn('defaults_used', [n['code'] for n in full['notices']])

    def test_level_changes_checks_verdicts_evidence_and_words(self):
        ps = {lv: PR.resolve({'ageGroup': 'Adult', 'playingLevel': lv, 'style': 'possession'}) for lv in range(1, 8)}
        enabled = {lv: {d for d, v in p['detectors'].items() if v['enabled']} for lv, p in ps.items()}
        self.assertEqual(enabled[1], {'pressure_cover', 'behind_ball', 'attacking_shape'})
        self.assertNotIn('weak_side_balance', enabled[2])
        self.assertNotIn('defensive_spacing', enabled[2])
        for lv in range(3, 8):
            self.assertEqual(enabled[lv], set(S.DETECTORS), lv)
        for lv, p in ps.items():
            verdicts = {v for d in p['detectors'].values() for v in d['verdicts']}
            self.assertEqual('work_on' in verdicts, lv >= 3, lv)
            self.assertEqual('try_this' in verdicts, lv <= 3, lv)
        self.assertEqual([ps[lv]['evidence']['maxWorkItems'] for lv in range(1, 8)], [1, 2, 2, 3, 3, 3, 3])
        self.assertEqual([ps[lv]['evidence']['minOpportunities'] for lv in range(1, 8)], [8, 8, 10, 10, 12, 12, 15])
        self.assertIsNone(ps[7]['evidence']['minFailureShare'])
        self.assertEqual(ps[7]['detectors']['pressure_cover']['params']['minFailureShare'], .3)
        self.assertIn('balance', ps[1]['register']['banned'])
        self.assertIn('balance', ps[5]['register']['concepts'])
        self.assertIn('rest defence', ps[6]['register']['concepts'])
        self.assertNotIn('mistake', ps[6]['register']['banned'])
        self.assertIn('mistake', ps[5]['register']['banned'])
        names = [ps[lv]['skills']['someone_back_when_attacking'] for lv in (1, 4, 6)]
        self.assertEqual(names, ['Someone ready to protect our goal', 'Defensive shape behind the ball',
                                 'Rest-defence margin (PitchIQ)'])
        objectives = [tuple(g['text'] for g in ps[lv]['objectives']) for lv in range(1, 8)]
        self.assertEqual(len(set(objectives)), 7)
        pools = [tuple(ps[lv]['drillPool']) for lv in range(1, 8)]
        self.assertEqual(len(set(pools)), 7)

    def test_age_drives_naming_tone_and_bands(self):
        self.assertEqual(PR.resolve({'ageGroup': 'U9', 'playingLevel': 7})['evidence']['naming'], 'relative')
        self.assertEqual(PR.resolve({'ageGroup': 'Adult', 'playingLevel': 1})['evidence']['naming'], 'never')
        self.assertEqual(PR.resolve({'ageGroup': 'U14', 'playingLevel': 6})['evidence']['naming'], 'evidence_only')
        adult = PR.resolve({'ageGroup': 'Adult', 'playingLevel': 1})
        child = PR.resolve({'ageGroup': 'U8', 'playingLevel': 1})
        self.assertIn("'Next step'", adult['register']['tone'])
        self.assertIn("'Try this'", child['register']['tone'])
        self.assertIn('adult_low_level', [n['code'] for n in adult['notices']])
        u16 = PR.resolve({'ageGroup': 'U16', 'playingLevel': 5})
        u18 = PR.resolve({'ageGroup': 'U18', 'playingLevel': 5})
        self.assertEqual((u16['band'], u18['band']), ('U15-U16', 'U17-U18'))
        self.assertEqual(u16['detectors']['behind_ball']['verdicts'], ['describe'])
        self.assertEqual(u18['detectors']['behind_ball']['verdicts'], ['review', 'describe'])
        # interchangeFirst acted on nothing and was removed in fix round 1 (critic S-14)
        self.assertNotIn('interchangeFirst', u16['detectors']['wide_defender_advance']['params'])
        self.assertNotEqual([g['text'] for g in u16['objectives']], [g['text'] for g in u18['objectives']])
        academy = PR.resolve({'ageGroup': 'U19', 'playingLevel': 6, 'style': 'compact'})
        adult6 = PR.resolve({'ageGroup': 'Adult', 'playingLevel': 6, 'style': 'compact'})
        self.assertEqual((academy['band'], adult6['band']), ('academy', 'adult'))
        self.assertTrue(any('UEFA Youth B' in g['text'] for g in academy['notExpected']))
        self.assertFalse(any('UEFA Youth B' in g['text'] for g in adult6['notExpected']))
        above = PR.resolve({'ageGroup': 'U8', 'playingLevel': 5})
        notice = next(n for n in above['notices'] if n['code'] == 'level_above_age')
        self.assertIn('U8', notice['text'])
        self.assertIn('USS-CG-U78', notice['text'])
        self.assertNotIn('level_above_age', [n['code'] for n in PR.resolve({'ageGroup': 'U8',
                                                                           'playingLevel': 3})['notices']])

    def test_style_changes_only_its_declared_effects(self):
        base = PR.resolve({'ageGroup': 'Adult', 'playingLevel': 5, 'style': 'unknown'})   # no effect at level 5
        for style, changed in (('compact', {'defensive_spacing', 'weak_side_balance'}),
                               ('high_press', {'pressure_cover', 'defensive_spacing'}),
                               ('counter', {'behind_ball', 'attacking_shape', 'defensive_spacing'}),
                               ('possession', {'wide_defender_advance', 'attacking_shape'})):
            p = PR.resolve({'ageGroup': 'Adult', 'playingLevel': 5, 'style': style})
            diff = {d for d in S.DETECTORS if p['detectors'][d] != base['detectors'][d]}
            self.assertEqual(diff, changed, style)
            for d in changed:
                self.assertIsNotNone(p['detectors'][d]['params']['styleEffect'], (style, d))
            for k in ('objectives', 'principles', 'evidence', 'register', 'safety', 'drillPool', 'skills'):
                self.assertEqual(p[k], base[k], (style, k))
        compact = PR.resolve({'ageGroup': 'Adult', 'playingLevel': 5, 'style': 'compact'})
        d5 = compact['detectors']['defensive_spacing']['params']
        self.assertEqual((d5['lateralGapM'], d5['dmGapM']), (17.0, 15.3))
        self.assertTrue(compact['detectors']['weak_side_balance']['params']['acceptFarSideGap'])
        press = PR.resolve({'ageGroup': 'Adult', 'playingLevel': 5, 'style': 'high_press'})
        self.assertEqual(press['detectors']['pressure_cover']['params']['coverExemptZone'], 'final_third')
        self.assertEqual(press['detectors']['defensive_spacing']['params']['dmGapM'], 21.6)
        counter = PR.resolve({'ageGroup': 'Adult', 'playingLevel': 5, 'style': 'counter'})
        self.assertEqual(counter['detectors']['behind_ball']['params']['minBehind'],
                         base['detectors']['behind_ball']['params']['minBehind']+1)
        self.assertFalse(counter['detectors']['attacking_shape']['params']['judgeWidth'])
        self.assertEqual(counter['detectors']['behind_ball']['params']['styleEffect'], 'counter_more_behind')

    def test_unknown_style_at_levels_6_and_7(self):
        p7 = PR.resolve({'ageGroup': 'Adult', 'playingLevel': 7, 'style': 'unknown'})
        self.assertFalse(any('work_on' in d['verdicts'] for d in p7['detectors'].values()))
        self.assertTrue(all('review' in d['verdicts'] for d in p7['detectors'].values() if d['enabled']
                            and d['verdicts'] != ['describe']))
        p6 = PR.resolve({'ageGroup': 'Adult', 'playingLevel': 6, 'style': 'unknown'})
        with_work = {d for d, v in p6['detectors'].items() if 'work_on' in v['verdicts']}
        self.assertEqual(with_work, {'pressure_cover', 'wide_defender_advance'})
        self.assertIn('style_unknown_6', [n['code'] for n in p6['notices']])
        declared = PR.resolve({'ageGroup': 'Adult', 'playingLevel': 7, 'style': 'compact'})
        self.assertTrue(any('work_on' in d['verdicts'] for d in declared['detectors'].values()))
        p5 = PR.resolve({'ageGroup': 'Adult', 'playingLevel': 5, 'style': 'unknown'})
        self.assertTrue(any('work_on' in d['verdicts'] for d in p5['detectors'].values()))

    def test_format_sets_players_behind_the_ball(self):
        def rule(level, players):
            p = PR.resolve({'ageGroup': 'Adult', 'playingLevel': level, 'style': 'possession',
                            'format': {'players': players, 'goalkeeper': players != 4}})
            prm = p['detectors']['behind_ball']['params']
            return prm['behindRule'], prm['minBehind'], prm['minMargin']
        self.assertEqual(rule(1, 11), ('count', 1, None))
        self.assertEqual(rule(2, 7), ('count', 1, None))
        self.assertEqual(rule(3, 9), ('count', 2, None))
        self.assertEqual(rule(4, 7), ('count', 2, None))
        self.assertEqual(rule(5, 11), ('count', 2, None))
        self.assertEqual(rule(6, 11), ('margin', None, 1))
        self.assertEqual(rule(7, 4), ('count', 1, None))

    def test_eleven_a_side_depth_is_described_at_levels_1_to_3(self):
        """Fix round 2, LEAD decision (c): judgeDepth False on 11v11 at levels 1-3 only; the table's switch never
        reaches a resolved profile."""
        for level in range(1, 8):
            for players in (4, 7, 9, 11):
                p = PR.resolve({'playingLevel': level, 'format': {'players': players, 'goalkeeper': players != 4}})
                prm = p['detectors']['attacking_shape']['params']
                self.assertNotIn('judgeDepthOn11v11', prm)
                self.assertEqual(prm.get('judgeDepth'), False if (players == 11 and level <= 3) else None,
                                 (level, players))

    def test_moments_worth_a_look_numbers_are_pitchiq_proposals(self):
        """Fix round 2, LEAD decision (b): the first level, the clear spells and the cap per report are registry
        values (PITCHIQ-PROPOSAL, unvalidated, with a basis) shared by every enabled check."""
        data = table()
        reg = data['parameters']['values']
        self.assertEqual(data['parameters']['origin'], 'PITCHIQ-PROPOSAL')
        self.assertFalse(data['parameters']['validated'])
        want = {'momentsFromLevel': 3, 'momentsMinSpells': 3, 'momentsMaxPerReport': 1}
        for name, value in want.items():
            self.assertEqual((reg[name]['value'], reg[name]['scale']), (value, 'count'))
            self.assertTrue(reg[name]['basis'].startswith('[P]'))
            self.assertIn(name, data['parameters']['shared'])
        for level in range(1, 8):
            p = PR.resolve({'playingLevel': level})
            for d, dp in p['detectors'].items():
                if dp['enabled']:
                    self.assertEqual({k: dp['params'][k] for k in want}, want, (level, d))

    def test_fix_round_3_numbers(self):
        """Fix round 3 (LEAD decisions (2) and (3)): the base-rate test level, the settled-play and back-line values
        are PITCHIQ-PROPOSAL registry values with a basis; the praise test's level scale lives in levels 1-4 and
        reaches every enabled check; level 3's far-side check has a base share; a bad value is refused."""
        data = table()
        reg = data['parameters']['values']
        want = {'baseTestAlpha': (.05, 'share'), 'baseSuccessScale': (1.0, 'share'), 'momentsSettleS': (2.0, 'seconds'),
                'momentsContestS': (1.0, 'seconds'), 'lineMinPlayers': (4, 'count'), 'lineStdMaxM': (4.0, 'body'),
                'markedM': (4.0, 'body')}
        for name, (value, scale) in want.items():
            self.assertEqual((reg[name]['value'], reg[name]['scale']), (value, scale), name)
            self.assertTrue(reg[name]['basis'].startswith('[P]'), name)
        for name in ('baseTestAlpha', 'baseSuccessScale', 'momentsSettleS', 'momentsContestS'):
            self.assertIn(name, data['parameters']['shared'])
        for name in ('lineMinPlayers', 'lineStdMaxM', 'markedM'):
            self.assertIn(name, data['detectors']['defensive_spacing']['params'])
        for level, k in zip(range(1, 8), (.5, .6, .7, .8, 1.0, 1.0, 1.0)):
            p = PR.resolve({'playingLevel': level})
            for d, dp in p['detectors'].items():
                if dp['enabled']:
                    self.assertEqual((dp['params']['baseSuccessScale'], dp['params']['baseTestAlpha']), (k, .05),
                                     (level, d))
        p = PR.resolve({'playingLevel': 3})
        self.assertEqual(p['detectors']['weak_side_balance']['params']['baseFailShare'], .004)
        for path, value in ((('levels', '2', 'baseSuccessScale'), 0.0), (('levels', '2', 'baseSuccessScale'), 1.2),
                            (('parameters', 'values', 'baseTestAlpha', 'value'), .5)):
            bad = copy.deepcopy(data)
            node = bad
            for k in path[:-1]:
                node = node[k]
            node[path[-1]] = value
            self.assertTrue(problems(bad), (path, value))

    def test_scale_and_format_notices(self):
        codes = lambda b: [n['code'] for n in PR.resolve(b)['notices']]   # noqa: E731
        self.assertIn('scale_unknown', codes({'ageGroup': 'U10'}))
        self.assertNotIn('scale_unknown', codes({'ageGroup': 'U10', 'field': {'length': 50, 'width': 35}}))
        self.assertNotIn('scale_unknown', codes({'ageGroup': 'Adult'}))
        self.assertIn('no_back_block', codes({'ageGroup': 'U8'}))
        self.assertIn('format_above_default', codes({'ageGroup': 'U9', 'format': {'players': 11}}))
        self.assertNotIn('format_above_default', codes({'ageGroup': 'U14', 'format': {'players': 7}}))
        for code in ('scale_unknown', 'small_format_registration', 'no_back_block'):
            self.assertNotIn(code, codes({'ageGroup': 'U14'}))

    def test_role_tier_and_serve_time_settings_do_not_change_the_profile(self):
        base = S.example('team_settings')
        p = PR.resolve(base)
        other = copy.deepcopy(base)
        other['roles'] = {'A-09': [{'role': 'CF', 'from': 10.0, 'to': 20.0, 'set': '2026-10-21T15:00:00'}]}
        other['formation']['periods'] = [{'t0': 0.0, 't1': 30.0, 'formation': '4-4-2'}]
        other['squad'] = None
        other['runLinks'] = []
        self.assertEqual(PR.resolve(other), p)
        self.assertEqual(PR.resolve(base, team='B')['team'], 'B')

    def test_resolve_does_not_modify_its_input(self):
        block = S.example('team_settings')
        before = copy.deepcopy(block)
        PR.resolve(block)
        self.assertEqual(block, before)

    def test_the_real_stage_runs_inside_interpret(self):
        from test_coaching_schema import RESULT, Fakes, settings_with
        fakes = Fakes()
        stages = fakes.stages()
        del stages['profile']                 # coaching._stage imports soccer.coaching.profiles:resolve
        for level in (1, 4, 7):
            settings = settings_with(playingLevel=level)
            obs = coaching.build_observations(RESULT, team='A', coaching=settings, result_sha='f'*64, stages=stages)
            roles = coaching.build_roles(obs, settings, stages=stages)
            report = coaching.interpret(obs, roles, settings, stages=stages)
            self.assertEqual((report['profile']['level'], report['profile']['sha']), (level, PR.load()['_sha']))


class CleanSettingsTests(unittest.TestCase):
    def test_a_valid_file_is_unchanged(self):
        ex = S.example('settings')
        self.assertEqual(PR.clean_settings(ex), ex)
        self.assertEqual(PR.clean_settings(PR.clean_settings(ex)), ex)

    def test_garbage_never_raises_and_always_validates(self):
        rng = random.Random(7)
        atoms = [None, True, 0, -1, 3.5, float('nan'), 'x', 'U10', 'adult', [], {}, [1, 2], {'a': 1}, 7, '4', 11.0]

        def junk(depth=0):
            r = rng.random()
            if depth > 2 or r < .4:
                return rng.choice(atoms)
            if r < .7:
                return [junk(depth+1) for _ in range(rng.randint(0, 3))]
            return {rng.choice(['players', 'goalkeeper', 'length', 'width', 'intended', 'periods', 'role', 'from',
                                'to', 'set', 'id', 'name', 'run', 'team', 'confirmed', 'x']): junk(depth+1)
                    for _ in range(rng.randint(0, 4))}
        fields = ['squad', 'ageGroup', 'playingLevel', 'format', 'field', 'federation', 'formation', 'style', 'roles',
                  'runLinks']
        for _ in range(400):
            team = {f: junk() for f in fields if rng.random() < .7}
            raw = rng.choice([junk(), {'teams': {'A': team, 'B': rng.choice([None, team, junk()])},
                                       'updated': rng.choice([None, 'yesterday', '2026-10-21T15:00:00'])}])
            out = PR.clean_settings(raw, problems=[])
            self.assertEqual(S.validate('settings', out), [], raw)
            json.dumps(out, allow_nan=False)

    def test_malformed_parts_are_dropped_and_listed(self):
        raw = {'teams': {'A': {
            'ageGroup': 'u10', 'playingLevel': '3', 'format': {'players': 7, 'goalkeeper': False},
            'field': {'length': 35, 'width': 50}, 'federation': 'FIFA', 'style': 'tiki-taka',
            'formation': {'intended': '4-3-3', 'periods': [{'t0': 0, 't1': 50, 'formation': '1-3-2-1'},
                                                           {'t0': 40, 't1': 80, 'formation': '1-2-3-1'},
                                                           {'t0': 90, 't1': 95, 'formation': '1-3-2-3'}]},
            'roles': {'GK-A': [{'role': 'GK', 'set': '2026-10-21T15:00:00'}],
                      'A-02': [{'role': 'LB', 'set': '2026-10-21T15:00:00'},
                               {'role': 'back-1/3', 'from': 0, 'to': 30, 'set': '2026-10-21T15:00:00'},
                               {'role': 'back-2/3', 'from': 20, 'to': 40, 'set': '2026-10-21T15:00:00'},
                               {'role': 'mid-1/2', 'from': 50, 'to': 60}]},
            'squad': {'id': 'sq-zz', 'name': 'Blue'},
            'runLinks': [{'run': 'r1', 'team': 'A', 'confirmed': True}, {'run': 'r1', 'team': 'A', 'confirmed': True},
                         {'run': 'bad run!', 'team': 'A', 'confirmed': True}]}}}
        notes = []
        out = PR.clean_settings(raw, problems=notes)
        a = out['teams']['A']
        self.assertEqual((a['ageGroup'], a['playingLevel'], a['format']), ('U10', 3, {'players': 7,
                                                                                     'goalkeeper': True}))
        self.assertEqual(a['field'], {'length': 50.0, 'width': 35.0})
        self.assertEqual((a['federation'], a['style']), ('auto', 'unknown'))
        self.assertEqual(a['formation'], {'intended': None, 'periods': [{'t0': 0.0, 't1': 50.0,
                                                                         'formation': '1-3-2-1'}]})
        self.assertEqual(list(a['roles']), ['A-02'])
        self.assertEqual([s['role'] for s in a['roles']['A-02']], ['back-1/3'])
        self.assertIsNone(a['squad'])
        self.assertEqual(a['runLinks'], [{'run': 'r1', 'team': 'A', 'confirmed': True}])
        self.assertIsNone(out['teams']['B'])
        self.assertEqual(S.validate('settings', out), [])
        for expect in ('field: width larger', 'federation', 'style', 'formation.intended', 'periods',
                       'not an internal player id', 'LB', 'overlapping', 'no valid "set" time', 'squad', 'runLinks'):
            self.assertTrue(any(expect in n for n in notes), (expect, notes))

    def test_now_stamps_role_spans_without_a_time(self):
        raw = {'teams': {'A': {'ageGroup': 'U14', 'roles': {'A-05': [{'role': 'RB'}]}}}}
        self.assertEqual(PR.clean_settings(raw)['teams']['A']['roles'], {})
        out = PR.clean_settings(raw, now='2026-10-22T10:00:00')
        self.assertEqual(out['teams']['A']['roles']['A-05'][0]['set'], '2026-10-22T10:00:00')

    def test_levels_bools_and_ranges(self):
        for bad in (0, 8, True, 2.5, 'three', None):
            out = PR.clean_settings({'teams': {'A': {'ageGroup': 'U12', 'playingLevel': bad}}})
            self.assertEqual(out['teams']['A']['playingLevel'], 3, bad)
        out = PR.clean_settings({'teams': {'A': {'ageGroup': 'U12', 'field': {'length': 200, 'width': 50}}}})
        self.assertIsNone(out['teams']['A']['field'])
        out = PR.clean_settings({'teams': {'A': {'format': {'players': 4, 'goalkeeper': True},
                                                 'formation': {'intended': '1-2-1'}}}})
        self.assertIsNone(out['teams']['A']['formation']['intended'])       # 3 outfield cannot play a 1-2-1
        out = PR.clean_settings({'teams': {'A': {'format': {'players': 5}, 'formation': {'intended': '1-2-1'}}}})
        self.assertEqual((out['teams']['A']['format']['goalkeeper'], out['teams']['A']['formation']['intended']),
                         (True, '1-2-1'))

    def test_settings_changes_map_to_the_right_tiers(self):
        base = S.example('team_settings')
        older = PR.clean_team(dict(base, playingLevel=2))
        self.assertEqual(S.stale_tiers(base, older), ['interpretation'])
        bigger = PR.clean_team(dict(base, format={'players': 9, 'goalkeeper': True}))
        self.assertEqual(S.stale_tiers(base, bigger), ['measurement', 'role', 'interpretation'])

    def test_default_team_settings(self):
        ts = PR.default_team_settings('U9')
        self.assertEqual((ts['playingLevel'], ts['format']), (2, {'players': 7, 'goalkeeper': True}))
        self.assertEqual(S.validate('team_settings', ts), [])
        self.assertEqual(PR.default_team_settings(level=7)['ageGroup'], 'Adult')


class CatalogTests(unittest.TestCase):
    def test_catalog_is_complete_json(self):
        cat = PR.catalog()
        json.dumps(cat, allow_nan=False)
        self.assertEqual([lv['level'] for lv in cat['levels']], list(range(1, 8)))
        self.assertEqual([a['id'] for a in cat['ageGroups']], list(S.AGE_GROUPS))
        self.assertEqual([f['players'] for f in cat['formats']], [4, 5, 7, 9, 11])
        self.assertEqual([s['id'] for s in cat['styles']], list(S.STYLES))
        self.assertEqual(cat['disclaimer'], sources_doc()['disclaimer'])
        self.assertEqual(len(cat['sources']), 122)
        self.assertEqual(cat['formatDefaults']['U11']['England'], {'players': 7, 'goalkeeper': True})
        self.assertEqual(cat['levels'][4]['bands'], ['U15-U16', 'U17-U18'])

    def test_small_formats_offer_relative_roles_only(self):
        for f in PR.catalog()['formats']:
            for formation, roles in f['roles'].items():
                for r in roles:
                    if f['players'] < 11:
                        self.assertTrue(r == 'GK' or re.fullmatch(S.RELATIVE_ROLE, r), (f['name'], r))
                    rb = S.example('rolebook')
                    rb['records'][0]['role'] = r
                    self.assertEqual([p for p in S.validate('rolebook', rb) if '.role' in p], [], r)
        self.assertEqual(PR.roles_for({'players': 7, 'goalkeeper': True}, '1-3-2-1'),
                         ['GK', 'back-1/3', 'back-2/3', 'back-3/3', 'mid-1/2', 'mid-2/2', 'front-1/1'])
        self.assertEqual(PR.roles_for({'players': 4, 'goalkeeper': False}),
                         ['back-1/1', 'mid-1/2', 'mid-2/2', 'front-1/1'])
        self.assertEqual(PR.roles_for({'players': 11, 'goalkeeper': True}), list(S.TEMPLATE_ROLES))

    def test_helpers(self):
        self.assertEqual(PR.formation_outfield('1-3-2-1', 7), 6)
        self.assertEqual(PR.formation_outfield('1-2-1', 4), 4)
        self.assertEqual(PR.formation_outfield('4-4-1', 11), 9)
        self.assertEqual([PR.band_of(5, 'U15'), PR.band_of(5, 'Adult'), PR.band_of(6, 'U21'), PR.band_of(4, 'U14')],
                         ['U15-U16', 'U17-U18', 'academy', None])
        self.assertEqual([PR.reach_speed_cap(a) for a in ('U8', 'U12', 'Adult')], [5.0, 6.0, 7.0])
        self.assertEqual(PR.typical_level('U13'), 4)
        self.assertEqual(PR.default_age(5), 'U16')


class ToolAndFileTests(unittest.TestCase):
    def test_check_command(self):
        import contextlib
        import io
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            self.assertEqual(B.main(['--check']), 0)
        self.assertEqual(buf.getvalue().strip(), 'OK')

    def test_files_are_ascii_and_crlf(self):
        files = [ROOT/'soccer'/'coaching'/'profiles.py', ROOT/'soccer'/'coaching'/'profiles.json',
                 ROOT/'soccer'/'coaching'/'sources.json', ROOT/'tools'/'build_coaching_sources.py',
                 Path(__file__)] + sorted((ROOT/'tools'/'annotations'/'coaching'/'research').glob('*.json'))
        for f in files:
            raw = f.read_bytes()
            raw.decode('ascii')
            self.assertEqual(raw.count(b'\n'), raw.count(b'\r\n'), f.name)
            self.assertTrue(raw.endswith(b'\r\n'), f.name)

    def test_profiles_module_imports_only_the_standard_library_and_the_schema(self):
        text = (ROOT/'soccer'/'coaching'/'profiles.py').read_text(encoding='ascii')
        imports = set(re.findall(r'^(?:import|from) ([\w.]+)', text, re.M))
        self.assertEqual(imports-{'copy', 'hashlib', 'json', 'math', 're', 'pathlib', 'soccer.coaching'}, set())


if __name__ == '__main__':
    unittest.main()
