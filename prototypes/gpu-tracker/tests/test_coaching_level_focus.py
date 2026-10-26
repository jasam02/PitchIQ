"""The NEXT DEVELOPMENT OBJECTIVE when the match gives none (polish round, Oct 23): the level's suggested focus from its
guidance in profiles.json (assess.level_focus), on the page (coaching.js, run in node: skipped without node) and in the
PDF (soccer/coaching/pdf.py).

The spec asks for the section at every level. After fix round 2 (critic S-8) 182 of the 196 run-copy reports had no
objective at all ("No next objective comes from this match: ..."). Now, with no work item and no check judged often
enough with room to improve, the report shows:
- the label "Suggested focus for this level - from the level's guidance, not from this match", then
- the enabled check (lowest objectiveRank) that this match did not demonstrate, that the format can assess, with the
  FEDERATION guidance item of the level that states it (assess.FOCUS_GUIDANCE) and its source ids; else the level's
  general learning objective (assess.GENERAL_OBJECTIVES, no check); else the check as PitchIQ's own proposal;
- at most one optional drill, for that check (drills.py chooses it from assess's objective; report.py drops any other),
  and none for a general objective;
- never a weakness of this team, never a check the match demonstrated, never another match format or formation.

Run:  python -m unittest test_coaching_level_focus        (from this directory; about 40 s)
"""
import copy
import os
import re
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for _p in (ROOT, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import coaching_text_fixtures as TF  # noqa: E402
sys.path.insert(0, os.path.join(ROOT, 'tools'))
import coaching_eval as CE  # noqa: E402
from soccer.coaching import assess as AS, drills as DR, pdf, profiles as PR, report as R, schema as S  # noqa: E402

LABEL = AS.LEVEL_FOCUS_LABEL
# words that would make the focus a judgement of this team (it is guidance, not a finding)
WEAKNESS = re.compile(r"\b(?:did not|didn't|failed|fails?|weak(?:ness)?|struggl\w*|poor|mistakes?|needs? to|should|"
                      r"nobody|no one|lack\w*|your team|this team|the team (?:did|was|were|left|kept)|work on|"
                      r"improve)\b", re.I)
FORMATS = ({'players': 4, 'goalkeeper': False}, {'players': 5, 'goalkeeper': True}, {'players': 7, 'goalkeeper': True},
           {'players': 9, 'goalkeeper': True}, {'players': 11, 'goalkeeper': True})
# (level, band) without a general learning objective a team can take: U17-U18's fails the level-5 text gate ("game
# model"), level 6 adults' and level 7's are coach education (licences) or longer than the objective's 400 characters
NO_GENERAL = {(5, 'U17-U18'), (6, 'adult'), (7, None)}
LEVEL_AGES = {1: ('U6', 'U8'), 2: ('U9', 'U10'), 3: ('U11', 'U12'), 4: ('U13', 'U14'), 5: ('U15', 'U16', 'U17', 'U18'),
              6: ('U17', 'U19', 'U23', 'Adult'), 7: ('U21', 'Adult')}


def profiles():
    for lvl, ages in LEVEL_AGES.items():
        for age in ages:
            for fmt in FORMATS:
                yield PR.resolve({'playingLevel': lvl, 'ageGroup': age, 'format': dict(fmt)}, 'A')


def problems_of(focus, profile, excluded=()):
    """What the focus breaks of its rules ([] = none)."""
    out = []
    t = focus['text']
    if not t.startswith(LABEL+': '):
        out.append('label')
    if len(t) > 400 or not re.fullmatch(r'[ -~]+', t):
        out.append('length / ASCII')
    if WEAKNESS.search(t[len(LABEL):]):
        out.append(f'weakness wording "{WEAKNESS.search(t[len(LABEL):]).group(0)}"')
    if AS.format_misfit(t, profile):
        out.append(f'format {AS.format_misfit(t, profile)}')
    if focus['tag'] == 'FEDERATION' and not t.endswith('Source: '+', '.join(focus['sources'])+'.'):
        out.append('source ids')
    if focus['tag'] != 'FEDERATION' and (focus['sources'] or AS.LEVEL_FOCUS_OWN not in t):
        out.append('PitchIQ proposal not said')
    if focus['skill'] is not None:
        det = S.SKILLS[focus['skill']]
        if det in set(excluded):
            out.append(f'names the excluded {det}')
        if not profile['detectors'][det]['enabled'] or not DR.format_supports(profile, det):
            out.append(f'{det} off or not assessable on {profile["format"]["name"]}')
    # EVAL's text rules on every report text (coaching_eval.report_problems criterion 13 and invariant 4)
    banned = set(profile['register']['banned']) | set(CE.DESIGN_BANNED.get(profile['level'], ())) | {'mistake'}
    out += [f'EVAL generic "{g}"' for g in CE.generic_sentences(t)]
    out += [f'EVAL banned "{w}"' for w in CE.words_in(t, sorted(banned))]
    out += [f'event {p}' for p in S.event_problems(t)]
    out += AS.focus_problems(t, profile, [g['text'].strip() for p in ('objectives', 'principles')
                                          for g in profile[p]] + [s for p in ('objectives', 'principles')
                                                                  for g in profile[p] for s in g['sources']]
                             + [g['text'].strip()+'.' for p in ('objectives', 'principles') for g in profile[p]])
    return out


class TheTables(unittest.TestCase):
    def test_every_pointer_names_a_federation_item_of_its_level(self):
        data = PR.load()
        for table in (AS.FOCUS_GUIDANCE, {lvl: {'general': v} for lvl, v in AS.GENERAL_OBJECTIVES.items()}):
            for lvl, by_skill in table.items():
                items = [g for part in ('objectives', 'principles') for g in data['levels'][str(lvl)][part]]
                for skill, ptrs in by_skill.items():
                    if skill != 'general':
                        self.assertIn(skill, S.SKILLS)
                        self.assertTrue(data['levels'][str(lvl)]['skills'].get(skill), (lvl, skill))
                    for part, source, start in ptrs:
                        with self.subTest(level=lvl, skill=skill, source=source):
                            hit = [g for g in data['levels'][str(lvl)][part] if g['sources'][0] == source and
                                   g['text'].startswith(start)]
                            self.assertEqual(len(hit), 1, items)
                            self.assertEqual(hit[0]['tag'], 'FEDERATION')

    def test_the_label_is_the_same_on_the_page_and_in_the_pdf(self):
        self.assertEqual(pdf.LEVEL_FOCUS_LABEL, AS.LEVEL_FOCUS_LABEL)
        with open(os.path.join(ROOT, 'coaching.js'), encoding='utf-8') as fh:
            js = fh.read()
        m = re.search(r"const LEVEL_FOCUS_LABEL = '((?:[^'\\]|\\.)*)';", js)
        self.assertIsNotNone(m)
        self.assertEqual(m.group(1).replace("\\'", "'"), AS.LEVEL_FOCUS_LABEL)
        self.assertEqual(AS.LEVEL_FOCUS_LABEL, "Suggested focus for this level - from the level's guidance, not from "
                                               "this match")


class EveryProfile(unittest.TestCase):
    """Levels 1-7 at each age they take, on every match format: a focus always, by its rules."""

    def test_a_focus_for_every_level_age_and_format(self):
        kinds = {}
        for prof in profiles():
            focus = AS.level_focus(prof)
            key = (prof['level'], prof['ageGroup'], prof['format']['name'])
            with self.subTest(profile=key):
                self.assertIsNotNone(focus)
                self.assertEqual(problems_of(focus, prof), [], focus['text'])
            kinds[key] = 'check+source' if focus['skill'] and focus['tag'] == 'FEDERATION' else \
                'general' if focus['skill'] is None else 'check, PitchIQ'
        self.assertEqual(len(kinds), sum(len(v) for v in LEVEL_AGES.values())*len(FORMATS))
        # levels 2-7 have a check stated by a FEDERATION item of the level on 11v11 at their ages; level 1's items all
        # use words the level's text gate keeps out ("width", "pressing", "win the ball"): its general objective
        for (lvl, age, fmt), kind in kinds.items():
            if fmt == '11v11':
                self.assertEqual(kind, 'general' if lvl == 1 else 'check+source', (lvl, age, fmt))

    def test_never_an_excluded_check(self):
        for prof in profiles():
            if prof['format']['players'] != 11 and prof['ageGroup'] not in ('U8', 'U10', 'U12'):
                continue
            enabled = [d for d in S.DETECTORS if prof['detectors'][d]['enabled']]
            for k in range(len(enabled)+1):
                excluded = enabled[:k]
                focus = AS.level_focus(prof, excluded)
                with self.subTest(level=prof['level'], age=prof['ageGroup'], fmt=prof['format']['name'], k=k):
                    if focus is None:
                        # every check excluded where the level has no general learning objective a team can take
                        self.assertIn((prof['level'], prof['band']), NO_GENERAL)
                        self.assertEqual(k, len(enabled))
                        continue
                    self.assertEqual(problems_of(focus, prof, excluded), [], focus['text'])

    def test_general_objectives_only(self):
        for prof in profiles():
            focus = AS.level_focus(prof, checks=False)
            if focus is not None:
                self.assertIsNone(focus['skill'])
                self.assertEqual(problems_of(focus, prof), [])
            else:
                self.assertIn((prof['level'], prof['band']), NO_GENERAL)


class WhatTheMatchDemonstrated(unittest.TestCase):
    def setUp(self):
        self.prof = PR.resolve({'playingLevel': 4, 'ageGroup': 'U14'}, 'A')

    def test_shown_checks_are_excluded(self):
        bar = self.prof['evidence']['strengthShare']
        n = 10
        dem = int(round(bar*n+.5))
        shown = {'pressure_cover': (3, 0),                  # every decided one shown
                 'weak_side_balance': (dem, n-dem),         # at the strength bar
                 'defensive_spacing': (1, 9),               # shown once of ten: not demonstrated
                 'behind_ball': (0, 0)}                     # nothing decided
        ex = AS.focus_exclusions([], self.prof, shown, unsupported={'attacking_shape'})
        self.assertIn('pressure_cover', ex)
        self.assertIn('weak_side_balance', ex)
        self.assertIn('attacking_shape', ex)
        self.assertNotIn('defensive_spacing', ex)
        self.assertNotIn('behind_ball', ex)
        focus = AS.level_focus(self.prof, ex)
        # level 4's ranks: wide-defender support (no FEDERATION item), behind the ball (none), pressure and cover
        # (shown), back-line coordination: stated by EF-COVER-RECOVER-SESSION
        self.assertEqual(focus['skill'], 'back_line_coordination')
        self.assertIn('Keep the right spacing', focus['text'])
        self.assertTrue(focus['text'].endswith('Source: EF-COVER-RECOVER-SESSION.'))

    def test_a_strength_is_excluded(self):
        f = {'observation': 'defensive_spacing', 'verdict': 'strength', 'aggregate': {'demonstrated': 5,
                                                                                    'notDemonstrated': 1}}
        self.assertIn('defensive_spacing', AS.focus_exclusions([f], self.prof, {}))

    def test_small_formats_never_name_a_check_they_cannot_assess(self):
        prof = PR.resolve({'playingLevel': 4, 'ageGroup': 'U14', 'format': {'players': 5, 'goalkeeper': True}}, 'A')
        ex = AS.focus_exclusions([], prof, {})
        self.assertIn('wide_defender_advance', ex)
        self.assertIn('defensive_spacing', ex)
        focus = AS.level_focus(prof, ex)
        self.assertNotIn(focus['skill'], ('wide_defender_support', 'back_line_coordination'))

    def test_a_formation_or_format_of_another_size_is_a_misfit(self):
        p11 = PR.resolve({'playingLevel': 3, 'ageGroup': 'U12', 'format': {'players': 11, 'goalkeeper': True}}, 'A')
        p9 = PR.resolve({'playingLevel': 3, 'ageGroup': 'U12', 'format': {'players': 9, 'goalkeeper': True}}, 'A')
        self.assertEqual(AS.format_misfit('Introduce a 9v9 formation; the example given is 1-3-2-3.', p11), '9v9')
        self.assertIsNone(AS.format_misfit('Introduce a 9v9 formation; the example given is 1-3-2-3.', p9))
        self.assertEqual(AS.format_misfit('U.S. 1-3-2-1 diagram', p9), '1-3-2-1')
        self.assertIsNone(AS.format_misfit('1v1s and 2v2 duels at U15-U16', p9))


class ThroughTheEngine(unittest.TestCase):
    """Real reports (coaching.interpret on EVAL's synthetic scenes) at levels 1-7."""

    @classmethod
    def setUpClass(cls):
        cls.reps = {(sc, var, lvl): TF.report(sc, var, level=lvl)
                    for sc, var in (('professional', 'default'), ('defending', 'press_cover'), ('kids_swarm', 'default'),
                                    ('possession_unknown', 'default'))
                    for lvl in range(1, 8)}

    def test_the_objective_and_its_drill(self):
        focus_n = 0
        for key, rep in self.reps.items():
            no = rep['nextObjective']
            optional = [p for p in rep['drills'] if p['optional']]
            with self.subTest(report=key):
                self.assertEqual(S.validate('report', rep), [])
                self.assertLessEqual(len(optional), 1)
                if AS.is_level_focus(no):
                    focus_n += 1
                    self.assertEqual((no['fromMatch'], no['finding']), (False, None))
                    self.assertFalse(rep['needsWork'])
                    strong = {f['observation'] for f in rep['strengths']}
                    if no['skill'] is not None:
                        self.assertNotIn(S.SKILLS[no['skill']], strong)
                    self.assertFalse(WEAKNESS.search(no['text'][len(LABEL):]), no['text'])
                if optional:
                    lib = DR.load_library()
                    d = DR.get(lib, optional[0]['drill'])
                    self.assertIsNotNone(no['skill'])
                    self.assertIn(S.SKILLS[no['skill']], d['related_tactical_detector'])
                    self.assertEqual(DR.safety_problems(d, rep['profile']['ageGroup'], rep['profile']['safety']), [])
                    self.assertEqual(DR.format_problems(d, rep['profile']), [])
                elif no['skill'] is None:
                    self.assertEqual(optional, [])
        self.assertGreater(focus_n, 10)

    def test_no_repetition_words_under_three_minutes(self):
        # the level-6 focus's drill (l6_block_compactness_shadow) has the easier line "Repeat the shadow phase ...";
        # EVAL's short_segment scene asks for no "repeat" / "pattern" / "often" in any text of a range under 3 minutes
        for lvl in range(1, 8):
            rep = TF.report('short_segment', 'default', level=lvl)
            with self.subTest(level=lvl):
                self.assertLess(rep['range']['seconds'], 180)
                words = [(p, m.group(0)) for p, v, _ in CE.report_texts(rep) for m in CE.REPETITION.finditer(v)]
                self.assertEqual(words, [])
                self.assertTrue(AS.is_level_focus(rep['nextObjective']) or rep['nextObjective']['fromMatch'] or
                                rep['nextObjective']['skill'], rep['nextObjective'])

    def test_an_optional_drill_for_another_check_is_left_out(self):
        rep_key = next(k for k, r in self.reps.items() if AS.is_level_focus(r['nextObjective']) and
                       r['nextObjective']['skill'] and any(p['optional'] for p in r['drills']))
        sc, var, lvl = rep_key
        got = {}

        def cap(**kw):
            got.update(copy.deepcopy(kw))
            return R.build_report_v2(**kw)
        _, st, obs, roles = TF.tiers(sc, var)
        from soccer import coaching
        coaching.interpret(obs, roles, TF.settings_at(st, 'A', lvl), 'A', meta={'run': 'synthetic', 'teamName': None},
                           stages={'report': cap})
        kw = copy.deepcopy(got)
        lib = DR.load_library()
        want = S.SKILLS[kw['assessment']['nextObjective']['skill']]
        other = next(d['drill_id'] for d in lib['drills'] if want not in d['related_tactical_detector'])
        kw['drills'] = [dict(p, drill=other) if p['optional'] else p for p in kw['drills']]
        log = []
        rep = R.build_report_v2(**dict(kw, log=log))
        self.assertEqual([p for p in rep['drills'] if p['optional']], [])
        self.assertIn('not the one optional drill for the next objective', [x[2] for x in log if x[0] == 'drill'])


def focus_report():
    """professional/default at level 4 (team A): no work item, the level's focus (far-side protection) and its drill."""
    rep = TF.report('professional', 'default', level=4)
    assert AS.is_level_focus(rep['nextObjective']) and rep['nextObjective']['skill'], rep['nextObjective']
    return rep


class ThePdf(unittest.TestCase):
    def test_label_focus_and_sources(self):
        from test_coaching_pdf import text_of
        rep = focus_report()
        label, body, ids = pdf.split_focus(rep['nextObjective']['text'])
        text = ' '.join(text_of(pdf.build_report_pdf(rep, None, {})[0]).split())
        self.assertIn(label+'.', text)
        self.assertIn(' '.join(body.split()[:10]), text)
        titles = {s['id']: s['title'] for s in rep['sources']}
        for i in ids:
            self.assertIn(f'({i})', text)
            self.assertIn(' '.join(titles[i].split()[:4]), text)
        self.assertNotIn("From the level's objectives, not from this match.", text)


class ThePage(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from test_coaching_panel import NODE, run
        if not NODE:
            raise unittest.SkipTest('needs node to run the page scripts')
        from test_coaching_moments_view import job
        cls.rep = focus_report()
        cls.out = run(job(cls.rep))[0]

    def test_label_focus_and_sources_as_in_the_pdf(self):
        prog = next(s['text'] for s in self.out['first']['sections'] if s['key'] == 'progression')
        label, body, ids = pdf.split_focus(self.rep['nextObjective']['text'])
        self.assertIn(label+'.', prog.replace(chr(0x2019), "'"))
        self.assertIn(body, prog)
        titles = {s['id']: s['title'] for s in self.rep['sources']}
        for i in ids:
            self.assertIn(f'{titles[i]} ({i})', prog)
        self.assertNotIn('From the level\u2019s objectives, not from this match.', prog)

    def test_the_optional_drill_label_as_in_the_pdf(self):
        # pre-freeze (critic S-3): an objective chosen from this match's checks (kids_swarm at level 1) and the level's
        # suggested focus (professional at level 4) each read their own optional-drill label on the page
        from test_coaching_panel import run
        from test_coaching_moments_view import job
        for rep, want in ((TF.report('kids_swarm', 'default', level=1), 'match'), (self.rep, 'level')):
            out = run(job(rep))[0]
            drills = next(s['text'] for s in out['first']['sections'] if s['key'] == 'drills').replace(chr(0x2019), "'")
            with self.subTest(want=want):
                self.assertIn(pdf.OPTIONAL_LABELS[want], drills)
                self.assertNotIn(pdf.OPTIONAL_LABELS['level' if want == 'match' else 'match'], drills)


class PreFreeze(unittest.TestCase):
    """Pre-freeze (Oct 24, critic S-3, after the freeze gate): COACH F-13, the level-7 focus that read "is called the
    basis of a good counter-press ... (clip label 'Rest defence')" (32 of the coach verifier's 546 scene reports and
    39 of the 196 run copies); COACH F-9, an adult beginner given the U7-U8 objective "every player gets a turn in
    goal"; the optional drill's label "not from this match's evidence" on an objective chosen from this match's
    checks."""

    def test_no_guidance_text_reads_like_a_research_note(self):
        data = PR.load()
        for lvl, block in data['levels'].items():
            for part in PR.GUIDANCE_PARTS:
                for g in block.get(part) or ():
                    with self.subTest(level=lvl, part=part, text=g['text'][:40]):
                        self.assertNotIn('clip label', g['text'])
                        self.assertNotRegex(g['text'], r'\bis called\b')
        for age in ('U18', 'U19', 'U21', 'U23', 'Adult'):
            prof = PR.resolve({'playingLevel': 7, 'ageGroup': age}, 'A')
            focus = AS.level_focus(prof)
            with self.subTest(age=age):
                self.assertEqual(focus['skill'], 'someone_back_when_attacking')
                self.assertIn('A sound defensive shape during attacks is described as the basis of a good '
                              'counter-press', focus['text'])
                self.assertTrue(focus['text'].endswith('Source: UEFA-PI-1V1.'))
                self.assertEqual(problems_of(focus, prof), [], focus['text'])

    def test_no_general_objective_for_a_team_older_than_its_source(self):
        given = 0
        for lvl in range(1, 8):
            for age in S.AGE_GROUPS:
                prof = PR.resolve({'playingLevel': lvl, 'ageGroup': age}, 'A')
                for focus in (AS.level_focus(prof), AS.level_focus(prof, checks=False)):
                    for sid in (focus or {}).get('sources') or ():
                        top = AS.GENERAL_OBJECTIVE_TOP_AGE.get(sid)
                        if top is None:
                            continue
                        given += 1
                        with self.subTest(level=lvl, age=age, source=sid):
                            self.assertLessEqual(S.AGE_ORDER[age], S.AGE_ORDER[top])
        self.assertGreater(given, 0)
        adult = PR.resolve({'playingLevel': 1, 'ageGroup': 'Adult'}, 'A')
        self.assertNotIn('turn in goal', (AS.level_focus(adult) or {}).get('text') or '')
        self.assertIsNone(AS.level_focus(adult, checks=False))
        # the ages it is written for keep it
        for age in ('U6', 'U7', 'U8'):
            focus = AS.level_focus(PR.resolve({'playingLevel': 1, 'ageGroup': age}, 'A'), checks=False)
            self.assertIn('turn in goal', focus['text'], age)
        # the report-level invariant (G1) refuses the old objective for an adult team
        u8 = AS.level_focus(PR.resolve({'playingLevel': 1, 'ageGroup': 'U8'}, 'A'), checks=False)
        rep = {'profile': adult, 'nextObjective': {'skill': None, 'text': u8['text'], 'fromMatch': False,
                                                   'finding': None}}
        self.assertTrue(any('USS-CG-U78' in p for p in R.text_problems(rep)), R.text_problems(rep))

    def test_the_optional_drill_label_is_true_of_the_objective(self):
        self.assertIn(DR.MATCH_OBJECTIVE_LABEL, pdf.OPTIONAL_LABELS['match'])
        self.assertIn(DR.NEXT_OBJECTIVE_LABEL, pdf.OPTIONAL_LABELS['level'])
        with open(os.path.join(ROOT, 'coaching.js'), encoding='utf-8') as fh:
            js = fh.read()
        m = re.search(r"const OPTIONAL_LABELS = \{match: '([^']*)',\s*level: '([^']*)'\};", js)
        self.assertIsNotNone(m)
        self.assertEqual({'match': m.group(1).replace(chr(0x2019), "'"), 'level': m.group(2).replace(chr(0x2019), "'")},
                         pdf.OPTIONAL_LABELS)
        from test_coaching_pdf import text_of
        for sc, var, lvl, want in (('kids_swarm', 'default', 1, 'match'), ('kids_swarm', 'default', 3, 'match'),
                                   ('professional', 'default', 4, 'level')):
            rep = TF.report(sc, var, level=lvl)
            opt = [p for p in rep['drills'] if p['optional']]
            with self.subTest(scene=sc, level=lvl):
                self.assertEqual(len(opt), 1)
                self.assertEqual(pdf.optional_label(rep), pdf.OPTIONAL_LABELS[want])
                label = DR.MATCH_OBJECTIVE_LABEL if want == 'match' else DR.NEXT_OBJECTIVE_LABEL
                self.assertIn(label, opt[0]['fields']['why'])
                self.assertEqual(R.text_problems(rep), [])
                text = ' '.join(text_of(pdf.build_report_pdf(rep, None, {})[0]).split())
                self.assertIn(pdf.OPTIONAL_LABELS[want], text)
                self.assertNotIn(pdf.OPTIONAL_LABELS['level' if want == 'match' else 'match'], text)
        # the report-level invariant (G1) refuses the old label on an objective chosen from this match's checks
        bad = TF.report('kids_swarm', 'default', level=1)
        p = next(p for p in bad['drills'] if p['optional'])
        p['fields']['why'] = p['fields']['why'].replace(DR.MATCH_OBJECTIVE_LABEL, DR.NEXT_OBJECTIVE_LABEL)
        self.assertTrue(any('for an objective chosen' in x for x in R.text_problems(bad)), R.text_problems(bad))


if __name__ == '__main__':
    unittest.main()
