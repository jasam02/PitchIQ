"""The drill libraries (soccer/coaching/drills.py).

v2 (drills.json, DESIGN v2 section 7): 14 drills across the 7 playing levels, each a labelled PitchIQ adaptation or
original with the ids of the sources that informed it; the schema record plus the library rules (skills and detectors
agree, a success signal per detector naming a known relation and no number, every listed age passes the safety filter,
games fit the player range, no unmeasured event, heading, crossing, sprint or approval word, level wording); the safety
filter enumerated over every age x level x format x style x detector (no heading drill, no load above the age cap, no
contact drill below U9, no age outside the drill's range, the profile can only make it stricter); at the typical ages of
every level each detector the profile table lets produce a work item has an allowed drill; the candidate ranking.

v2 selection (drills.select / selection, the interpretation tier's 'drills' stage, DESIGN 7.3-7.4): up to three
priorities, one drill each (a second only for another failed relation), the failed relation read from the finding,
no drill without a work item but one optional drill for the next objective, the six fields through explain's gate at
the profile's level (numbers, words, phase qualifier, relation polarity; in a 60 s range too), the safety filter
enumerated through the stage over every age x level x format x detector, and EVAL's synthetic scenes through the real
tiers, assess and explain (and coaching.interpret with stand-ins for the later stages).

v1 (drills_v1.json, deprecated, read by the v1 engine until it retires): the six conventional exercises, unchanged.

Run:  python -m unittest test_coaching_drills          (from this directory)
"""
import copy
import importlib
import itertools
import json
import os
import re
import sys
import tempfile
import time
import unittest
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from soccer.coaching import drills, language, schema as S  # noqa: E402

# DESIGN 7.2: id -> (levels, detectors named there). The library may add detectors (DRILLS_ADDED, each needed for a
# level/detector pair that had no drill), never remove one.
DESIGN_TABLE = {
    'l1_help_your_teammate_2v2': ([1, 2], {'pressure_cover', 'behind_ball'}),
    'l1_spread_out_4v4': ([1, 2], {'attacking_shape'}),
    'l1_go_to_the_ball_corridor': ([1, 2], {'pressure_cover'}),
    'l2_pressure_and_cover_3v2': ([2, 3], {'pressure_cover'}),
    'l2_someone_stays_back_4v4': ([2, 3], {'behind_ball', 'wide_defender_advance'}),
    'l3_defensive_recovery_5v4': ([3, 4], {'pressure_cover', 'defensive_spacing'}),
    'l3_wide_and_deep_9v9_shape': ([3], {'attacking_shape', 'wide_defender_advance'}),
    'l4_defensive_unit_6v6': ([4, 5], {'wide_defender_advance', 'defensive_spacing'}),
    'l4_cover_the_line_3v3': ([4, 5], {'pressure_cover', 'weak_side_balance'}),
    'l5_transition_and_cover_8v8': ([5, 6], {'wide_defender_advance', 'behind_ball'}),
    'l5_shift_nine_zones': ([5, 6], {'weak_side_balance', 'defensive_spacing'}),
    'l6_rest_defence_9v9': ([6, 7], {'behind_ball', 'wide_defender_advance'}),
    'l6_block_compactness_shadow': ([6, 7], {'defensive_spacing'}),
    'l7_tactical_phase_11v11': ([7], {'wide_defender_advance', 'behind_ball', 'defensive_spacing'}),
}
DRILLS_ADDED = {'l4_defensive_unit_6v6': {'behind_ball'},                          # L4 D4 work_on had no drill
                'l5_shift_nine_zones': {'pressure_cover'},                         # L6 D1 had no drill
                'l6_block_compactness_shadow': {'pressure_cover', 'weak_side_balance'}}   # L7 D1, D2 had none
# DESIGN 2.3: the detectors whose verdicts include a work item (work_on or try_this) per level, and the typical ages
# of each level (DESIGN 2.2). Used when profiles.py (PROFILE) is not there yet; with it, the profiles are read.
WORK_ITEMS = {1: ('pressure_cover', 'behind_ball', 'attacking_shape'),
              2: ('pressure_cover', 'behind_ball', 'wide_defender_advance', 'attacking_shape'),   # D6: DESIGN 4.6
              3: ('pressure_cover', 'wide_defender_advance', 'behind_ball', 'attacking_shape'),
              4: ('pressure_cover', 'weak_side_balance', 'wide_defender_advance', 'behind_ball'),
              5: ('pressure_cover', 'weak_side_balance', 'wide_defender_advance', 'defensive_spacing'),
              6: ('pressure_cover', 'weak_side_balance', 'wide_defender_advance', 'behind_ball', 'defensive_spacing'),
              7: ('pressure_cover', 'weak_side_balance', 'wide_defender_advance', 'behind_ball', 'defensive_spacing')}
TYPICAL_AGES = {1: ('U7', 'U8'), 2: ('U9', 'U10'), 3: ('U11', 'U12'), 4: ('U13', 'U14'),
                5: ('U15', 'U16', 'U17', 'U18'), 6: ('U17', 'U18', 'U19', 'U21', 'U23', 'Adult'),
                7: ('U19', 'U21', 'U23', 'Adult')}
SPEC_EXAMPLES = (('Adult', 1), ('U13', 6), ('U17', 6), ('Adult', 7))   # spec Part 3 age x level combinations
FORMATS = {4: {'players': 4, 'goalkeeper': False}, 5: {'players': 5, 'goalkeeper': True},
           7: {'players': 7, 'goalkeeper': True}, 9: {'players': 9, 'goalkeeper': True},
           11: {'players': 11, 'goalkeeper': True}}
DEFAULT_FORMAT_BY_AGE = [('U6', 4), ('U9', 7), ('U11', 9), ('U13', 11)]   # U.S. formats by age (user decision 4)


def default_format(age):
    players = 4
    for start, n in DEFAULT_FORMAT_BY_AGE:
        if S.AGE_ORDER[age] >= S.AGE_ORDER[start]:
            players = n
    return FORMATS[players]


def profile(age, level, players=None, safety=None, style='unknown'):
    p = {'ageGroup': age, 'level': level, 'format': FORMATS[players] if players else default_format(age),
         'style': style}
    if safety is not None:
        p['safety'] = safety
    return p


def profiles_module():
    """PROFILE's profiles.py when it has landed (wave 1 runs in parallel), else None."""
    try:
        mod = importlib.import_module('soccer.coaching.profiles')
    except ModuleNotFoundError:
        return None
    return mod if callable(getattr(mod, 'resolve', None)) else None


def resolved(mod, age, level):
    block = {'ageGroup': age, 'playingLevel': level, 'format': default_format(age)}
    return mod.resolve(block, team='A')


def PR_default_age(level):
    from soccer.coaching import profiles as PR
    return PR.default_age(level)


def bare(library):
    return {k: copy.deepcopy(v) for k, v in library.items() if not k.startswith('_')}


# ================================ v2 ================================
class LibraryV2Tests(unittest.TestCase):
    def setUp(self):
        self.library = drills.load_library()
        self.by_id = {d['drill_id']: d for d in self.library['drills']}

    def test_library_is_valid_and_matches_the_design_table(self):
        self.assertEqual(drills.validate_library(self.library), [])
        self.assertEqual(S.validate('drill_library', bare(self.library)), [])
        self.assertEqual(self.library['version'], 2)
        self.assertEqual(sorted(self.by_id), sorted(DESIGN_TABLE))
        for did, (levels, detectors) in DESIGN_TABLE.items():
            d = self.by_id[did]
            self.assertEqual(d['development_level'], levels, did)
            self.assertEqual(set(d['related_tactical_detector']), detectors | DRILLS_ADDED.get(did, set()), did)
            self.assertEqual(d['origin'], S.ORIGIN_P)
            self.assertEqual(d['safety']['heading'], 'none', did)
            self.assertTrue(all(drills.CREDIT in r['note'] for r in d['source_reference']), did)
            self.assertTrue(any(r['relation'] in ('adapted_from', 'pitchiq_original') or 'adaptation' in r['note']
                                for r in d['source_reference']), did)
            for f in ('instructions', 'coaching_points', 'progression', 'regression', 'equipment'):
                self.assertGreaterEqual(len(d[f]), 2 if f != 'instructions' else 4, f'{did}.{f}')
        per_level = {lv: sum(lv in d['development_level'] for d in self.library['drills']) for lv in range(1, 8)}
        self.assertTrue(all(n >= 2 for n in per_level.values()), per_level)

    def test_file_is_crlf_ascii_json(self):
        for path in (drills.LIBRARY_PATH, drills.V1_LIBRARY_PATH):
            data = path.read_bytes()
            data.decode('ascii')
            lines = data.split(b'\n')
            self.assertTrue(all(line.endswith(b'\r') for line in lines[:-1]), path.name)
            json.loads(data)
        self.assertNotEqual(drills.LIBRARY_PATH, drills.V1_LIBRARY_PATH)

    def test_wording_is_safe_to_quote(self):
        n = 0
        for d in self.library['drills']:
            for path, text in drills.drill_texts(d):
                n += 1
                self.assertRegex(text, r'^[ -~]+$', f'{d["drill_id"]}.{path}')
                self.assertEqual(S.event_problems(text), [], f'{d["drill_id"]}.{path}: {text}')
                self.assertIsNone(re.search(r'\bhead|\bheader|\bcross(es|ing)?\b', text, re.I), text)
            for sig in d['success_signals_next_match']:
                self.assertIsNone(re.search(r'\d', sig['text']), sig['text'])
            lowest = min(d['development_level'])
            if lowest <= 1:          # level-1 words: helper, go to the ball, protect our goal
                for path, text in drills.instruction_texts(d):
                    self.assertIsNone(re.search(r'\b(cover\w*|press\w*|balanc\w*|transition\w*)', text, re.I),
                                      f'{d["drill_id"]}.{path}: {text}')
        self.assertGreater(n, 300)
        self.assertEqual(drills.text_problems(self.by_id['l1_help_your_teammate_2v2']), [])

    def test_skills_detectors_and_signals_agree(self):
        for d in self.library['drills']:
            skills = [d['primary_skill']]+d['secondary_skills']
            self.assertEqual({S.SKILLS[s] for s in skills}, set(d['related_tactical_detector']), d['drill_id'])
            self.assertEqual({s['detector'] for s in d['success_signals_next_match']},
                             set(d['related_tactical_detector']), d['drill_id'])
            for s in d['success_signals_next_match']:
                self.assertIn(s['relation'], drills.RELATIONS[s['detector']])
        self.assertEqual(set(drills.RELATIONS), set(S.DETECTORS))
        self.assertEqual(set(drills.SKILL_OF), set(S.DETECTORS))

    def test_games_fit_the_player_counts(self):
        """player_count counts every player, goalkeepers included (sources verifier F9): the name's game as the note
        plays it fits the range, the minimum can play at least one game of the note, no game of the drill's text
        needs more than the maximum, and every recommended format's game has room for the minimum."""
        self.assertEqual(drills.game_sizes('3v2, then 11 v 7 on half of an 11v11 pitch; a 9v9 field'), [5, 18])
        self.assertEqual(drills.game_headcounts('6 v 6 plus goalkeepers on half of an 11v11 pitch'), [(12, 14)])
        self.assertEqual(drills.game_headcounts('9 v 9 with goalkeepers; with 18 players use 8 v 8 plus goalkeepers.'),
                         [(18, 20), (16, 18)])
        self.assertEqual(drills.game_headcounts('4 v 4 with or without goalkeepers; 5 attackers v 4 defenders, plus a '
                                                'goalkeeper if available. Play 3 v 3.'), [(8, 8), (6, 6)])
        for d in self.library['drills']:
            pc = d['player_count']
            note = drills.game_headcounts(pc['note'])
            for n in drills.game_sizes(d['name']):
                n = next((h for g, h in note if g == n), n)
                self.assertTrue(pc['min'] <= n <= pc['max'], (d['drill_id'], n))
            if note:
                self.assertGreaterEqual(pc['min'], min(h for _, h in note), d['drill_id'])
            for _, text in drills.instruction_texts(d):
                for _, n in drills.game_headcounts(text):
                    self.assertLessEqual(n, pc['max'], d['drill_id'])
            for fmt in d['recommended_format']:
                self.assertLessEqual(pc['min'], 2*drills.FORMAT_PLAYERS[fmt], (d['drill_id'], fmt))
        self.assertEqual(drills.game_sizes(self.by_id['l2_pressure_and_cover_3v2']['name']), [5])
        # F9's three drills: the 6v6 and the 8v8 with goalkeepers need 14 and 18; the nine-zone game's note plays
        # 6 v 6 with fewer players, so 16 can run it
        mins = [self.by_id[i]['player_count']['min'] for i in
                ('l4_defensive_unit_6v6', 'l5_transition_and_cover_8v8', 'l5_shift_nine_zones')]
        self.assertEqual(mins, [14, 18, 16])

    def test_load_caches_hashes_and_refuses_an_invalid_file(self):
        self.assertIs(drills.load_library(), drills.load_library())
        self.assertRegex(drills.library_sha(self.library), r'^[0-9a-f]{16}$')
        good = bare(self.library)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'drills.json')
            bad = copy.deepcopy(good)
            bad['drills'][0]['safety']['heading'] = 'limited'
            with open(path, 'w', encoding='ascii', newline='\r\n') as f:
                json.dump(bad, f, indent=2)
            with self.assertRaises(ValueError) as ctx:
                drills.load_library(path)
            self.assertIn('heading', str(ctx.exception))
            with open(path, 'w', encoding='ascii', newline='\r\n') as f:
                json.dump(good, f, indent=2)
            os.utime(path, ns=(1, 2))
            self.assertEqual(len(drills.load_library(path)['drills']), 14)
        self.assertIs(drills.load_library(), drills.load_library())

    def test_quoted_numbers(self):
        d = self.by_id['l2_pressure_and_cover_3v2']
        self.assertEqual(drills.quoted_numbers(d), [2.0, 3.0, 5.0, 12.0, 15.0, 16.0, 22.0])

    def test_validation_catches_each_rule(self):
        good = bare(self.library)
        self.assertEqual(drills.validate_library(good), [])
        k1 = [d['drill_id'] for d in good['drills']].index('l1_help_your_teammate_2v2')
        k4 = [d['drill_id'] for d in good['drills']].index('l4_defensive_unit_6v6')
        k5 = [d['drill_id'] for d in good['drills']].index('l5_transition_and_cover_8v8')
        cases = {
            # the match format (fix round 1: sources F9, coach M3)
            'a minimum that cannot play its game with goalkeepers': (k4, lambda d: d['player_count'].update(min=12)),
            'a game with goalkeepers above the maximum': (k1, lambda d: d['progression'].append(
                'Play 6 v 6 with goalkeepers.')),
            'recommended for a format whose game is smaller than its minimum': (k5, lambda d: d.update(
                recommended_format=['7v7', '9v9', '11v11'])),
            'an 11v11 role name in a drill offered below 11v11': (k1, lambda d: d['coaching_points'].append(
                'The full-backs stay level with the helper.')),
            'a back four in a 9v9 drill': (k5, lambda d: d['instructions'].append('Defend with a back four.')),
            'an unknown key': (k1, lambda d: d.update(variations={})),
            'non-ASCII': (k1, lambda d: d['instructions'].append('Mark a 20 \u00d7 15 m area.')),
            'heading allowed': (k5, lambda d: d['safety'].update(heading='limited')),
            'load above the cap of its youngest age': (k1, lambda d: d['safety'].update(physicalLoad='moderate')),
            'contact below U9': (k1, lambda d: d['safety'].update(contact='normal')),
            'a skill that is not a progression skill': (k1, lambda d: d.update(primary_skill='help_teammate_defend')),
            'skills and detectors disagree': (k1, lambda d: d.update(secondary_skills=[])),
            'an unknown relation': (k1, lambda d: d['success_signals_next_match'][0].update(relation='closing')),
            'a detector without a signal': (k1, lambda d: d['success_signals_next_match'].pop()),
            'a number in a signal': (k1, lambda d: d['success_signals_next_match'][0].update(
                text='Someone goes to the ball in 5 more moments.')),
            'a level-1 drill says cover': (k1, lambda d: d['coaching_points'].append('Cover your teammate.')),
            'an unmeasured event': (k5, lambda d: d['coaching_points'].append('Win the ball back quickly.')),
            'a pass as an event': (k5, lambda d: d['regression'].append('Pass the ball twice before shooting.')),
            'heading words': (k5, lambda d: d['progression'].append('Finish crosses with headers.')),
            'a race': (k5, lambda d: d['progression'].append('Finish with a recovery race.')),
            'an approval claim': (k5, lambda d: d['source_reference'][0].update(
                note='PitchIQ adaptation of a session approved by England Football.')),
            'a source note without the credit': (k5, lambda d: d['source_reference'][0].update(
                note='Adapted from the England read and react session.')),
            'an adaptation not labelled as such': (k5, lambda d: d['source_reference'][0].update(
                note='PitchIQ version of the England read and react session.')),
            'a long quotation': (k5, lambda d: d['coaching_points'].append(
                'As the session says: "' + 'keep the shape behind the ball while the team attacks quickly' + '".')),
            'a game bigger than the group': (k1, lambda d: d['progression'].append('Play 7 v 7 on a big pitch.')),
            'the named game below the minimum': (k1, lambda d: d['player_count'].update(min=6)),
            'an age the filter refuses': (k5, lambda d: d['age'].update(min='U10')),
            'a federation name in the instructions': (k5, lambda d: d['instructions'].append(
                'Play it the UEFA way.')),
            'no success signals': (k1, lambda d: d.update(success_signals_next_match=[])),
            'a non-PitchIQ origin': (k1, lambda d: d.update(origin='FEDERATION')),
            # lines a report quotes (drill selection, wave 5): they must pass explain's gate in any report
            'an obligation word': (k5, lambda d: d['progression'].append('The back players must shift together.')),
            'a repeated-pattern word in a signal': (k1, lambda d: d['success_signals_next_match'][0].update(
                text='When the other team has the ball, someone goes to the ball more often.')),
            'a pattern word in a regression': (k5, lambda d: d['regression'].append('Repeat the patterns slowly.')),
            'a phase without its qualifier': (k5, lambda d: d['success_signals_next_match'][0].update(
                text='Out of possession, the channel behind is covered in more of these moments.')),
            'a number word in a signal': (k5, lambda d: d['success_signals_next_match'][1].update(
                text='When we attack, three players stay behind the ball in more of these moments.')),
            # fix round 3: the comparison with this match in one removable ending (coach F-8), no repetition word
            # the report refuses in a short range anywhere a coach reads it (coach F-7)
            'a signal without the comparison ending': (k1, lambda d: d['success_signals_next_match'][0].update(
                text='When the other team has the ball in our half, someone goes to the ball.')),
            'a second comparison with this match': (k1, lambda d: d['success_signals_next_match'][0].update(
                text='As in this match, someone goes to the ball in more of these moments than in this match.')),
            'a repetition word in a coaching point': (k5, lambda d: d['coaching_points'].append(
                'Switch the roles often.')),
        }
        for what, (k, change) in cases.items():
            lib = copy.deepcopy(good)
            change(lib['drills'][k])
            self.assertTrue(drills.validate_library(lib), what)
        # the format rules each name the planted defect alone
        expect = {'a minimum that cannot play its game with goalkeepers': 'cannot play any game of its note',
                  'a game with goalkeepers above the maximum': '(goalkeepers counted) does not fit',
                  'recommended for a format whose game is smaller than its minimum': 'recommended for 7v7',
                  'an 11v11 role name in a drill offered below 11v11': '11v11 role name "full-backs"',
                  'a back four in a 9v9 drill': '11v11 role name "back four"',
                  'a signal without the comparison ending': 'must end with "in more of these moments than in',
                  'a second comparison with this match': 'compare with this match nowhere else',
                  'a repetition word in a coaching point': '"often" is a repetition word the report refuses'}
        for what, words in expect.items():
            k, change = cases[what]
            lib = copy.deepcopy(good)
            change(lib['drills'][k])
            got = drills.validate_library(lib)
            self.assertEqual(len(got), 1, (what, got))
            self.assertIn(words, got[0], what)
        # an 11v11-only drill may name 11v11 roles (the level-7 phase training: "full-backs", "holding midfielder")
        self.assertTrue(drills.ELEVEN_ROLES.search(' '.join(
            t for _, t in drills.instruction_texts(self.by_id['l7_tactical_phase_11v11']))))
        lib = copy.deepcopy(good)
        lib['drills'].append(copy.deepcopy(lib['drills'][0]))
        self.assertIn('repeated drill_id', ' '.join(drills.validate_library(lib)))
        self.assertTrue(drills.validate_library({'version': 2, 'notes': 'x', 'drills': []}))
        self.assertTrue(drills.validate_library([]))

    def test_quoted_lines_pass_the_report_gate(self):
        """The lines drill selection quotes (success signals, regressions, progressions) pass explain's gate at every
        level the drill is offered at, in a range under 3 minutes too: level concepts, bans, events, hypotheticals,
        role words, phase wording, relation polarity (182 lines x 2 ranges). The library's own copy of the
        repeated-pattern words is language.SHORT_BANNED."""
        from soccer.coaching import explain
        self.assertEqual(drills.REPEAT_WORDS, tuple(language.SHORT_BANNED))
        n = 0
        for d in self.library['drills']:
            for level in d['development_level']:
                prof = resolve(PR_default_age(level), level)
                for path, line in drills.instruction_texts(d):
                    if not path.startswith(drills.QUOTED_PATHS):
                        continue
                    det = d['related_tactical_detector'][0]
                    if path.startswith('success'):
                        det = d['success_signals_next_match'][int(path.split('[')[1].split(']')[0])]['detector']
                    f = S.example('finding')
                    f['level'], f['observation'] = level, det
                    for short in (False, True):
                        self.assertEqual(explain.validate(
                            {'headline': drills.GATE_FILLER, 'explanation': line, 'short': drills.GATE_FILLER,
                             'action': None}, [], f, prof, short=short, checks=drills.LIBRARY_CHECKS), [],
                            (d['drill_id'], level, path))
                    n += 1
        self.assertEqual(n, 182)

    def test_report_repetition_words_and_signal_endings(self):
        """Fix round 3. Coach F-7: the report drops a level focus drill whose fields say "repeat", "pattern" or
        "often" in a range under 3 minutes (report.SHORT_REPEAT, EVAL's coaching_eval.REPETITION), and the level-6
        shadow drill's easier line "Repeat the shadow phase ..." kept level 6 from any focus drill on every clip of
        3 minutes or less (0 optional drills at level 6 in the 196 run copies); the library's copy of the words equals
        both, and no drill text a coach reads uses them. Coach F-8 = facts F-10: every success signal ends with the
        comparison with this match (drills.COMPARE) and nowhere else compares with it, so neutral_signal can leave it
        out when this match did not judge the check."""
        from soccer.coaching import report
        tools = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'tools')
        if tools not in sys.path:
            sys.path.insert(0, tools)
        import coaching_eval as CE
        self.assertEqual(drills.REPORT_REPEAT.pattern, report.SHORT_REPEAT.pattern)
        self.assertEqual(drills.REPORT_REPEAT.pattern, CE.REPETITION.pattern)
        for d in self.library['drills']:
            for path, text in drills.instruction_texts(d):
                self.assertIsNone(drills.REPORT_REPEAT.search(text), (d['drill_id'], path, text))
            for s in d['success_signals_next_match']:
                self.assertTrue(s['text'].endswith(drills.COMPARE), s['text'])
                neutral = drills.neutral_signal(s['text'])
                self.assertNotIn('this match', neutral)
                self.assertTrue(neutral.endswith('.') and not neutral.endswith(' .'), neutral)
        self.assertEqual(self.by_id['l6_block_compactness_shadow']['regression'][0],
                         'Run the shadow phase again with the coach calling out each picture.')
        self.assertEqual(drills.neutral_signal('Someone goes to the ball'+drills.COMPARE), 'Someone goes to the ball.')
        self.assertEqual(drills.neutral_signal('No comparison.'), 'No comparison.')

    def test_sources_registry_checks(self):
        cited = sorted({r['source'] for d in self.library['drills'] for r in d['source_reference']})
        self.assertEqual(len(cited), 21)
        reg = drills.sources_index({'sources': [{'id': s, 'read': 'full'} for s in cited]})
        self.assertEqual(drills.validate_library(self.library, sources=reg), [])
        reg = drills.sources_index([{'id': 'EF-PZ-CANON', 'aliases': ['EF-PZ'], 'read': 'partial'}] +
                                   [{'id': s, 'read': 'full'} for s in cited if s != 'EF-PZ'])
        self.assertEqual(drills.validate_library(self.library, sources=reg), [])          # an alias resolves
        reg['EF-PZ'] = dict(reg['EF-PZ'], read='not_read')
        self.assertIn('was not read', ' '.join(drills.validate_library(self.library, sources=reg)))
        del reg['EF-PZ']
        self.assertIn('not in the sources registry', ' '.join(drills.validate_library(self.library, sources=reg)))

    def test_sources_resolve_in_the_registry(self):
        reg = drills.load_sources()
        if reg is None:
            self.skipTest('soccer/coaching/sources.json is not built yet (PROFILE, wave 1)')
        self.assertEqual(drills.validate_library(self.library, sources=reg), [])


class SafetyFilterTests(unittest.TestCase):
    def setUp(self):
        self.library = drills.load_library()
        self.by_id = {d['drill_id']: d for d in self.library['drills']}

    def test_age_floor(self):
        self.assertEqual([drills.load_cap(a) for a in ('U6', 'U10', 'U11', 'U14', 'U15', 'Adult')],
                         ['low', 'low', 'moderate', 'moderate', 'high', 'high'])
        self.assertEqual(drills.effective_safety('U11'), {'heading': 'none', 'maxLoad': 'moderate', 'tackling': True})
        self.assertEqual(drills.effective_safety('U8'), {'heading': 'none', 'maxLoad': 'low', 'tackling': False})
        self.assertEqual(drills.effective_safety('U12')['heading'], 'limited')
        loose = {'heading': 'limited', 'maxLoad': 'high', 'tackling': True, 'sprintRaces': True, 'notes': []}
        self.assertEqual(drills.effective_safety('U8', loose), drills.effective_safety('U8'))   # never looser
        strict = {'heading': 'none', 'maxLoad': 'low', 'tackling': False, 'sprintRaces': False, 'notes': []}
        self.assertEqual(drills.effective_safety('Adult', strict),
                         {'heading': 'none', 'maxLoad': 'low', 'tackling': False})

    def test_heading_drills_are_refused_at_every_age(self):
        d = copy.deepcopy(self.by_id['l6_rest_defence_9v9'])
        d['safety']['heading'] = 'limited'
        d['age'] = {'min': 'U6', 'max': 'Adult'}
        d['safety']['physicalLoad'] = 'low'
        d['safety']['contact'] = 'no_tackle'
        reasons = {a: drills.safety_problems(d, a) for a in S.AGE_GROUPS}
        self.assertTrue(all(reasons.values()), reasons)
        self.assertEqual(reasons['U11'], ['heading_not_allowed_at_age'])
        self.assertEqual(reasons['U12'], ['heading_volume_undeclared'])
        self.assertEqual(reasons['U18'], ['heading_volume_undeclared'])
        self.assertEqual(reasons['Adult'], ['heading_guidance_not_researched'])

    def test_load_contact_and_age_range(self):
        d = self.by_id['l6_rest_defence_9v9']               # high load, contact, U15-Adult
        self.assertEqual(drills.safety_problems(d, 'U14'), ['load_above_cap', 'below_age_range'])
        self.assertEqual(drills.safety_problems(d, 'U15'), [])
        strict = {'heading': 'none', 'maxLoad': 'moderate', 'tackling': False, 'sprintRaces': False, 'notes': []}
        self.assertEqual(drills.safety_problems(d, 'Adult', strict), ['load_above_cap', 'tackling_not_allowed'])
        kid = copy.deepcopy(self.by_id['l2_pressure_and_cover_3v2'])
        self.assertEqual(drills.safety_problems(kid, 'U8'), [])
        kid['safety']['contact'] = 'normal'
        self.assertEqual(drills.safety_problems(kid, 'U8'), ['tackling_not_allowed'])
        self.assertEqual(drills.safety_problems(kid, 'U9'), [])
        self.assertEqual(drills.safety_problems(self.by_id['l1_spread_out_4v4'], 'U6'), [])
        narrow = copy.deepcopy(self.by_id['l1_spread_out_4v4'])
        narrow['age']['max'] = 'U10'
        self.assertEqual(drills.safety_problems(narrow, 'U11'), ['above_age_range'])
        prof = profile('U8', 2)
        older = [did for did, d in self.by_id.items() if S.AGE_ORDER[d['age']['min']] > S.AGE_ORDER['U8']]
        self.assertEqual(sorted(drills.refused(self.library, prof)), sorted(older))

    def test_enumeration_of_every_profile(self):
        """Every age x level x format x style x detector: candidates() never returns a drill the rules refuse, and
        the style never changes the drills."""
        start = time.perf_counter()
        profiles = checks = 0
        seen = set()
        for age, level, players, style in itertools.product(S.AGE_GROUPS, range(1, 8), FORMATS, S.STYLES):
            prof = profile(age, level, players, style=style)
            for det in S.DETECTORS:
                profiles += 1
                for c in drills.candidates(self.library, prof, det):
                    checks += 1
                    d = c['drill']
                    seen.add(d['drill_id'])
                    a = S.AGE_ORDER[age]
                    self.assertEqual(d['safety']['heading'], 'none')
                    self.assertLessEqual(drills.LOADS.index(d['safety']['physicalLoad']),
                                         drills.LOADS.index(drills.load_cap(age)))
                    if a < S.AGE_ORDER['U9']:
                        self.assertEqual(d['safety']['contact'], 'no_tackle')
                    self.assertTrue(S.AGE_ORDER[d['age']['min']] <= a <= S.AGE_ORDER[d['age']['max']])
                    self.assertIn(level, d['development_level'])
                    self.assertIn(det, d['related_tactical_detector'])
                    if a <= S.AGE_ORDER['U11']:
                        self.assertFalse(any(re.search(r'\bhead', t, re.I) for _, t in drills.drill_texts(d)))
        self.assertEqual(profiles, 17*7*5*5*6)
        self.assertEqual(seen, set(self.by_id))
        for age, level, players in itertools.product(S.AGE_GROUPS, range(1, 8), FORMATS):
            for det in S.DETECTORS:
                ids = {st: [c['drill_id'] for c in drills.candidates(self.library, profile(age, level, players,
                                                                                            style=st), det)]
                       for st in S.STYLES}
                self.assertEqual(len({tuple(v) for v in ids.values()}), 1, (age, level, players, det))
        self.assertLess(time.perf_counter()-start, 30.0)
        self.assertGreater(checks, 1000)

    def test_profile_safety_is_applied(self):
        strict = {'heading': 'none', 'maxLoad': 'low', 'tackling': False, 'sprintRaces': False, 'notes': []}
        ids = [c['drill_id'] for c in drills.candidates(self.library, profile('Adult', 6, 11, safety=strict),
                                                       'behind_ball')]
        self.assertEqual(ids, [])                     # the level-6 drills are contact drills of moderate/high load
        ids = [c['drill_id'] for c in drills.candidates(self.library, profile('Adult', 2, 7, safety=strict),
                                                       'pressure_cover')]
        self.assertEqual(sorted(ids), ['l1_go_to_the_ball_corridor', 'l1_help_your_teammate_2v2',
                                       'l2_pressure_and_cover_3v2'])


class CoverageAndRankingTests(unittest.TestCase):
    def setUp(self):
        self.library = drills.load_library()

    def work_items(self, age, level):
        mod = profiles_module()
        if mod is None:
            return WORK_ITEMS[level]
        p = resolved(mod, age, level)
        return tuple(d for d, rec in p['detectors'].items()
                     if rec['enabled'] and {'work_on', 'try_this'} & set(rec['verdicts']))

    def test_every_work_item_has_an_allowed_drill_at_typical_ages(self):
        pairs = set()
        gaps = []
        combos = [(age, lv) for lv, ages in TYPICAL_AGES.items() for age in ages] + list(SPEC_EXAMPLES)
        for age, level in combos:
            for det in self.work_items(age, level):
                pairs.add((level, det))
                if not drills.candidates(self.library, profile(age, level), det):
                    gaps.append((age, level, det))
        self.assertEqual(gaps, [])
        if profiles_module() is None:
            self.assertEqual(len(pairs), 29)

    def test_profile_drill_pools_exist_and_fit_their_level(self):
        mod = profiles_module()
        if mod is None:
            self.skipTest('soccer/coaching/profiles.py is not there yet (PROFILE, wave 1)')
        for level, ages in TYPICAL_AGES.items():
            p = resolved(mod, ages[0], level)
            for did in p['drillPool']:
                d = drills.get(self.library, did)
                self.assertIsNotNone(d, (level, did))
                self.assertIn(level, d['development_level'], did)
                self.assertEqual(drills.safety_problems(d, ages[0], p['safety']), [], (level, did))

    def test_ranking(self):
        ranked = drills.candidates(self.library, profile('U12', 3, 9), 'pressure_cover', failed_relations=['cover'])
        self.assertEqual([c['drill_id'] for c in ranked], ['l3_defensive_recovery_5v4', 'l2_pressure_and_cover_3v2'])
        self.assertTrue(all(c['relationMatch'] and c['skillMatch'] and c['formatMatch'] for c in ranked))
        # a failed relation only one drill trains comes first, whatever the other keys say
        ranked = drills.candidates(self.library, profile('U12', 3, 9), 'pressure_cover',
                                   failed_relations=['goal_side'])
        self.assertEqual(ranked[0]['drill_id'], 'l3_defensive_recovery_5v4')
        # primary skill: the nine-zone drill trains far-side protection first, the cover-the-line drill pressure
        ranked = drills.candidates(self.library, profile('U16', 5, 11), 'pressure_cover', failed_relations=['cover'])
        self.assertEqual([c['drill_id'] for c in ranked], ['l4_cover_the_line_3v3', 'l5_shift_nine_zones'])
        # format: at level 3 a 9v9 team gets the 9v9 shape drill, an 11v11 team still gets it (adaptable), flagged
        r9 = drills.candidates(self.library, profile('U12', 3, 9), 'attacking_shape')
        r11 = drills.candidates(self.library, profile('U13', 3, 11), 'attacking_shape')
        self.assertEqual([(c['drill_id'], c['formatMatch']) for c in r9], [('l3_wide_and_deep_9v9_shape', True)])
        self.assertEqual([(c['drill_id'], c['formatMatch']) for c in r11], [('l3_wide_and_deep_9v9_shape', False)])
        # 5v5 is assessed with the 4v4 rules
        self.assertTrue(drills.format_matches(drills.get(self.library, 'l1_spread_out_4v4'), '5v5'))
        # nothing for a detector the level has no drill for, or a level the drill is not offered at
        self.assertEqual(drills.candidates(self.library, profile('U8', 1), 'weak_side_balance'), [])
        self.assertEqual(drills.candidates(self.library, profile('Adult', 7), 'attacking_shape'), [])

    def test_levels_change_the_drill(self):
        """Same finding family (no cover behind the first defender; at level 1 nobody went to the ball), the
        typical age of each level: the first candidate changes with the level."""
        top = {}
        for level, ages in TYPICAL_AGES.items():
            failed = ['pressure'] if level == 1 else ['cover']
            ranked = drills.candidates(self.library, profile(ages[0], level), 'pressure_cover', failed)
            top[level] = ranked[0]['drill_id']
            self.assertIn(level, ranked[0]['drill']['development_level'])
        self.assertEqual(top, {1: 'l1_go_to_the_ball_corridor', 2: 'l2_pressure_and_cover_3v2',
                               3: 'l3_defensive_recovery_5v4', 4: 'l4_cover_the_line_3v3', 5: 'l4_cover_the_line_3v3',
                               6: 'l6_block_compactness_shadow', 7: 'l6_block_compactness_shadow'})


# ================================ v2 selection: the 'drills' stage (DESIGN 7.3-7.4) ================================
PHASE_OF = {'pressure_cover': 'opp', 'weak_side_balance': 'opp', 'defensive_spacing': 'opp',
            'wide_defender_advance': 'own', 'behind_ball': 'own', 'attacking_shape': 'own'}
FAILING = {   # a failing finding's measurements at its first key moment (explain.MEASUREMENT_KEYS)
    'pressure_cover': {'failureMode': 'no_cover', 'pressurePresent': True, 'coverPresent': False, 'd1M': 3.1,
                       'coverDistM': 14.0, 'metres': 'estimated'},
    'weak_side_balance': {'weakCount': 0, 'centralCount': 2, 'openWidthM': 25.0, 'metres': 'estimated'},
    'wide_defender_advance': {'side': 'left', 'advanceM': 18.0, 'behindN': 0, 'channelOccupied': False,
                              'cbAdjusted': False, 'dmDropped': False, 'interchange': False,
                              'opponentsInChannel': 1, 'threatN': 2, 'restMargin': 0, 'metres': 'estimated'},
    'behind_ball': {'behindN': 0, 'minBehind': 1, 'threatN': 2, 'restMargin': -1, 'metres': 'estimated'},
    'defensive_spacing': {'failureMode': 'lateral_gap', 'lateralGapM': 24.0, 'gapBallSide': True,
                          'opponentInGap': True, 'dmGapM': 12.0, 'lineStdM': 2.0, 'metres': 'estimated'},
    'attacking_shape': {'widthShare': .4, 'depthShare': .2, 'channelsOccupied': 2, 'metres': 'estimated'},
}
ALL_FORMATS = tuple(FORMATS.values())


def resolve(age, level, fmt=None, style=None):
    from soccer.coaching import profiles as PR
    block = {'ageGroup': age, 'playingLevel': level, 'format': fmt or default_format(age)}
    if style:
        block['style'] = style
    return PR.resolve(block, team='A')


def work_item(det, prof, priority=1, family='team', t0=760.0, verdict=None, claims=True, **measurements):
    """A failing finding of `det` at the profile's level, valid for schema 'finding' (S.example's evidence), with
    explain's claims (claims=True) or none (the stage then builds them)."""
    from soccer.coaching import explain
    f = S.example('finding')
    verdicts = prof['detectors'][det]['verdicts']
    f['verdict'] = verdict or ('work_on' if 'work_on' in verdicts else 'try_this')
    f['observation'], f['level'], f['priority'] = det, prof['level'], priority
    f['key'] = S.finding_key(det, 'A', family, t0, t0+115.6)
    f['id'] = S.record_id('fnd', f['key'])
    f['context']['phase']['state'] = PHASE_OF[det]
    f['measurements'] = dict(FAILING[det], **measurements)
    f['suggestedAction'], f['claims'], f['drills'] = None, [], []
    if f['verdict'] not in S.CRITICISM:
        f['priority'] = None
    if claims:
        f['claims'] = explain.build_claims(f, prof)
    return f


def gate_claims(f, drill):
    """What the stage lets a field use: the finding's claims (every number shown must be one of their facts) and the
    relations the drill restores."""
    out = list(f['claims'])+[{'type': 'drill_link', 'facts': {'drill': drill['drill_id']}}]
    out += [{'type': 'action', 'facts': {'relation': s['relation']}} for s in drill['success_signals_next_match']]
    return out


def gate(text, claims, f, prof, short, checks=None):
    from soccer.coaching import explain
    return explain.validate({'headline': drills.GATE_FILLER, 'explanation': text, 'short': drills.GATE_FILLER,
                             'action': None}, claims, f, prof, short=short, checks=checks)


class SelectionTests(unittest.TestCase):
    """drills.select / selection on synthetic findings (explain's claims) and resolved profiles."""

    @classmethod
    def setUpClass(cls):
        cls.library = drills.load_library()

    def check_pick(self, pk, f, prof, short):
        """The contract of one pick of a work item (schema, detector, level, safety, moments) and its six fields
        through explain's gate at the profile's level."""
        self.assertEqual(S.validate(S.DrillPick, pk), [])
        d = drills.get(self.library, pk['drill'])
        self.assertIn(f['observation'], d['related_tactical_detector'])
        self.assertEqual(drills.safety_problems(d, prof['ageGroup'], prof['safety']), [])
        self.assertEqual((pk['finding'], pk['optional']), (f['id'], False))
        self.assertEqual(pk['fields']['moments'], [ts['evidence'] for ts in f['timestamps']])
        claims = gate_claims(f, d)
        for field in ('trains', 'why', 'progress'):
            self.assertEqual(gate(pk['fields'][field], claims, f, prof, short), [], (field, pk['fields'][field]))
        for field in ('easier', 'harder'):
            self.assertEqual(gate(pk['fields'][field], claims, f, prof, short, drills.LIBRARY_CHECKS), [], field)
            self.assertTrue(pk['fields'][field].startswith(d['regression' if field == 'easier' else 'progression'][0]))
        text = ' '.join(pk['fields'][k] for k in ('trains', 'why', 'progress'))
        level = prof['level']
        if level == 1:
            self.assertIsNone(re.search(r'\d', text), text)          # level 1 shows no numbers
        if level < 5:
            self.assertNotIn('%', text)
        if level < 6:
            self.assertNotIn('per 10 minutes', text)
        self.assertIsNone(drills.REPEAT_RE.search(' '.join(pk['fields'][k] for k in drills.FIELD_LIMITS)))
        return d

    def test_every_work_item_at_typical_ages_gets_a_checked_drill(self):
        """DESIGN 13.4 #12: at the typical ages of every level (and the spec's age x level examples), a work item of
        every detector the level can criticise gets a drill of its detector and level that passes the safety filter,
        whose six fields pass explain's gate; in a 60 s range too."""
        combos = [(age, lv) for lv, ages in TYPICAL_AGES.items() for age in ages]+list(SPEC_EXAMPLES)
        n = 0
        for (age, level), rng in itertools.product(combos, (None, {'seconds': 60.0})):
            prof = resolve(age, level)
            for det, dp in prof['detectors'].items():
                if not {'work_on', 'try_this'} & set(dp['verdicts']):
                    continue
                f = work_item(det, prof)
                picks = drills.select([f], prof, [f['id']], range=rng)
                self.assertTrue(picks, (age, level, det))
                self.assertEqual(f['drills'], [pk['drill'] for pk in picks])
                for pk in picks:
                    d = self.check_pick(pk, f, prof, rng is not None)
                    self.assertIn(level, d['development_level'], (age, level, det))   # never the earlier fallback
                n += 1
        self.assertGreater(n, 100)

    def test_priorities_order_cap_and_verdicts(self):
        """Up to three priorities in the given order; a fourth gets nothing; reviews, strengths and descriptions never
        get a drill; an id that is not a reported work item is skipped as not_reported."""
        prof = resolve('U14', 4)
        items = [work_item(d, prof, priority=k+1, family=f'f{k}') for k, d in
                 enumerate(('weak_side_balance', 'pressure_cover', 'wide_defender_advance', 'behind_ball'))]
        review = work_item('defensive_spacing', prof, verdict='review', family='r')
        strength = work_item('attacking_shape', prof, verdict='strength', family='s')
        findings = [strength]+items+[review]
        sel = drills.selection(findings, prof, [f['id'] for f in items]+[review['id'], 'fnd-000000000000'])
        self.assertEqual([pk['finding'] for pk in sel['picks']], [f['id'] for f in items[:3]])
        self.assertEqual([pk['drill'] for pk in sel['picks']],
                         ['l4_cover_the_line_3v3', 'l3_defensive_recovery_5v4', 'l4_defensive_unit_6v6'])
        self.assertEqual(items[3]['drills'], [])
        self.assertEqual((review['drills'], strength['drills']), ([], []))
        self.assertEqual(sel['skipped'], [{'finding': review['id'], 'reason': 'not_reported'},
                                          {'finding': 'fnd-000000000000', 'reason': 'not_reported'}])
        # without a priority list the work items are taken by their priority
        again = copy.deepcopy(findings)
        self.assertEqual([pk['drill'] for pk in drills.select(again, prof)],
                         [pk['drill'] for pk in sel['picks']])
        # a drill a higher priority already got is not repeated when another ranks as high
        prof1 = resolve('U8', 1)
        a = work_item('pressure_cover', prof1, priority=1, family='a', failureMode='no_pressure')
        b = work_item('pressure_cover', prof1, priority=2, family='b', t0=900.0, failureMode='no_pressure')
        picks = drills.select([a, b], prof1, [a['id'], b['id']])
        self.assertEqual([pk['drill'] for pk in picks], ['l1_go_to_the_ball_corridor', 'l1_help_your_teammate_2v2'])
        # ... and repeated when nothing ranks as high (one drill then serves both priorities)
        prof2 = resolve('U10', 2)
        d3 = work_item('wide_defender_advance', prof2, priority=1, family='left')
        d4 = work_item('behind_ball', prof2, priority=2, family='team', t0=900.0)
        picks = drills.select([d3, d4], prof2, [d3['id'], d4['id']])
        self.assertEqual([(pk['finding'], pk['drill']) for pk in picks],
                         [(d3['id'], 'l2_someone_stays_back_4v4'), (d4['id'], 'l2_someone_stays_back_4v4')])

    def test_no_work_item_gives_one_optional_next_objective_drill(self):
        """DESIGN 7.3 step 4: without a work item no drill rests on this match; one optional drill for the next
        objective, labelled so, citing no moment."""
        prof = resolve('U14', 4)
        strength = work_item('pressure_cover', prof, verdict='strength', family='s')
        review = work_item('weak_side_balance', prof, verdict='review', family='r')
        sel = drills.selection([strength, review], prof, [])
        self.assertEqual(len(sel['picks']), 1)
        pk = sel['picks'][0]
        self.assertEqual((pk['finding'], pk['optional'], pk['fields']['moments']), (None, True, []))
        self.assertIn(drills.NEXT_OBJECTIVE_LABEL, pk['fields']['why'])
        self.assertEqual(S.validate(S.DrillPick, pk), [])
        # assess's rule: the enabled detector with the lowest objectiveRank that has a skill name and no strength
        ranks = sorted((dp['params'].get('objectiveRank') or 99, S.DETECTORS.index(d), d)
                       for d, dp in prof['detectors'].items()
                       if dp['enabled'] and prof['skills'].get(drills.SKILL_OF[d]) and d != 'pressure_cover')
        self.assertEqual(sel['objective']['detector'], ranks[0][2])
        self.assertIn(ranks[0][2], drills.get(self.library, pk['drill'])['related_tactical_detector'])
        self.assertEqual((strength['drills'], review['drills']), ([], []))
        # a strength on the level's first objective moves the objective to the next one
        everyone = sorted((dp['params'].get('objectiveRank') or 99, S.DETECTORS.index(d), d)
                          for d, dp in prof['detectors'].items()
                          if dp['enabled'] and prof['skills'].get(drills.SKILL_OF[d]))
        top = everyone[0][2]
        strong = work_item(top, prof, verdict='strength', family='t')
        self.assertEqual(drills.selection([strong], prof, [])['objective']['detector'], everyone[1][2])
        # assess's NextObjective resting on a finding the report does not show (withheld) is not used: the rule is
        other = next(r[2] for r in everyone if r[2] != sel['objective']['detector'])
        from_match = {'skill': drills.SKILL_OF[other], 'text': 'x', 'fromMatch': True, 'finding': 'fnd-'+'1'*12}
        self.assertEqual(drills.selection([strength, review], prof, [], next_objective=from_match)['objective'],
                         sel['objective'])
        # assess's NextObjective is used when it does not rest on a finding
        no = {'skill': 'back_line_coordination', 'text': 'x', 'fromMatch': False, 'finding': None}
        sel = drills.selection([], resolve('Adult', 6), [], next_objective=no)
        self.assertEqual(sel['objective'], {'skill': 'back_line_coordination', 'detector': 'defensive_spacing'})
        self.assertIn('defensive_spacing',
                      drills.get(self.library, sel['picks'][0]['drill'])['related_tactical_detector'])
        # pre-freeze (critic S-3): that objective (not from a work item, not the level's suggested focus) is the one
        # assess chose from this match's checks, and the drill says so; the level's focus and the level's own
        # objective keep "not from this match's evidence"
        self.assertIn(drills.MATCH_OBJECTIVE_LABEL, sel['picks'][0]['fields']['why'])
        self.assertNotIn(drills.NEXT_OBJECTIVE_LABEL, sel['picks'][0]['fields']['why'])
        from soccer.coaching import assess as AS
        focus = dict(no, text=AS.LEVEL_FOCUS_LABEL+': Back-line coordination. x')
        sel = drills.selection([], resolve('Adult', 6), [], next_objective=focus)
        self.assertIn(drills.NEXT_OBJECTIVE_LABEL, sel['picks'][0]['fields']['why'])
        obj = {'skill': 'back_line_coordination', 'detector': 'defensive_spacing'}
        self.assertEqual(drills.objective_label(obj, no), drills.MATCH_OBJECTIVE_LABEL)
        for other in (None, focus, from_match, dict(no, skill='pressure_and_cover')):
            self.assertEqual(drills.objective_label(obj, other), drills.NEXT_OBJECTIVE_LABEL, other)
        # nothing at all for an empty level table: no objective, no drill
        empty = copy.deepcopy(resolve('Adult', 6))
        for dp in empty['detectors'].values():
            dp['enabled'] = False
        self.assertEqual(drills.selection([], empty, [])['picks'], [])

    def test_second_drill_only_for_another_failed_relation(self):
        """DESIGN 7.3 step 4 on a library where two level-2 shape drills train different relations: both when width
        and the channels failed, one when only width failed; never a third."""
        lib = bare(self.library)
        base = drills.get(lib, 'l1_spread_out_4v4')
        variants = []
        for did, rel in (('aa_width_game', 'width'), ('ab_channels_game', 'channels'), ('ac_width_again', 'width')):
            d = copy.deepcopy(base)
            d['drill_id'] = did
            d['success_signals_next_match'] = [dict(base['success_signals_next_match'][0], relation=rel)]
            variants.append(d)
        lib['drills'] = variants
        self.assertEqual(drills.validate_library(lib), [])
        prof = resolve('U10', 2, fmt=FORMATS[7])
        both = work_item('attacking_shape', prof, widthShare=.4, depthShare=.5, channelsOccupied=2)
        self.assertEqual(drills.failed_relations(both, prof), ['width', 'channels'])
        picks = drills.select([both], prof, [both['id']], library=lib)
        self.assertEqual([pk['drill'] for pk in picks], ['aa_width_game', 'ab_channels_game'])
        self.assertTrue(picks[1]['fields']['why'].startswith('A second drill for the first priority'))
        one = work_item('attacking_shape', prof, widthShare=.4, depthShare=.5, channelsOccupied=3)
        self.assertEqual([pk['drill'] for pk in drills.select([one], prof, [one['id']], library=lib)],
                         ['aa_width_game'])

    def test_failed_relations(self):
        """The failed relation the +2 ranking reads, per detector (DESIGN 7.3 step 3)."""
        p4, p6 = resolve('U14', 4), resolve('Adult', 6)
        cases = [
            ('pressure_cover', p4, {}, ['cover']),
            ('pressure_cover', p4, {'failureMode': 'no_pressure'}, ['pressure']),
            ('weak_side_balance', p4, {}, ['weak_side']),
            ('wide_defender_advance', p4, {}, ['channel_cover', 'cb_shift']),
            ('wide_defender_advance', resolve('U16', 5), {}, ['channel_cover', 'cb_shift', 'dm_drop']),
            ('wide_defender_advance', p4, {'cbAdjusted': None}, ['channel_cover']),
            ('behind_ball', p4, {}, ['behind_ball']),
            ('behind_ball', p6, {}, ['rest_margin']),
            ('defensive_spacing', resolve('U16', 5), {}, ['lateral_gap']),
            ('defensive_spacing', resolve('U16', 5), {'failureMode': 'dm_gap'}, ['unit_gap']),
            ('attacking_shape', resolve('U12', 3), {}, ['width', 'depth', 'channels']),
            ('attacking_shape', resolve('U12', 3), {'widthShare': .7, 'channelsOccupied': 3}, ['depth']),
            ('attacking_shape', resolve('U12', 3, style='counter'), {}, ['depth', 'channels']),   # width not judged
        ]
        for det, prof, meas, expected in cases:
            works = {'work_on', 'try_this'} & set(prof['detectors'][det]['verdicts'])
            f = work_item(det, prof, verdict=None if works else 'review', **meas)
            self.assertEqual(drills.failed_relations(f, prof), expected, (det, meas))
        # from the claims when the failure mode is not measured; nothing for a strength
        f = work_item('pressure_cover', p4)
        f['measurements'].pop('failureMode')
        self.assertEqual(drills.failed_relations(f, p4), ['cover'])
        self.assertEqual(drills.failed_relations(work_item('pressure_cover', p4, verdict='strength'), p4), [])

    def test_unchecked_text_never_reaches_a_pick(self):
        """Every field goes through the gate: a library line the gate refuses is left out of 'easier' (a pattern word
        is refused in a 60 s range only); a pick whose field cannot pass at all is dropped with its reason."""
        prof = resolve('U14', 4)
        lib = bare(self.library)
        d = drills.get(lib, 'l4_defensive_unit_6v6')
        d['regression'] = ['Repeat it often until the shift holds.']+d['regression']
        f = work_item('wide_defender_advance', prof, family='left')
        long_pick = drills.select([f], prof, [f['id']], library=lib)[0]
        self.assertTrue(long_pick['fields']['easier'].startswith('Repeat it often'))
        short_pick = drills.select([f], prof, [f['id']], library=lib, range={'seconds': 60.0})[0]
        self.assertEqual(short_pick['fields']['easier'], ' '.join(d['regression'][1:3]))
        d['regression'] = ['The players must stay back.']
        sel = drills.selection([f], prof, [f['id']], library=lib)
        self.assertEqual([pk['drill'] for pk in sel['picks']], [])
        self.assertEqual(sel['skipped'], [{'finding': f['id'], 'reason': 'text_failed_validation'}])
        self.assertEqual(f['drills'], [])
        # a skill name the gate refuses (a banned word) cannot be shown in 'trains' or 'why': no pick
        bad = copy.deepcopy(prof)
        bad['skills'] = {k: 'Never leave the channel' for k in bad['skills']}
        sel = drills.selection([work_item('wide_defender_advance', bad, family='left')], bad, None)
        self.assertEqual((sel['picks'], [s['reason'] for s in sel['skipped']]), ([], ['text_failed_validation']))

    def test_texts_follow_the_level(self):
        """The same failing pattern worded by the level: no numbers at level 1, counts from 2, a share from 5, a rate
        per 10 minutes of the phase (estimated) from 6; the observed relation in explain's polarity phrases."""
        texts = {}
        for level, age in ((1, 'U8'), (2, 'U10'), (5, 'U16'), (6, 'Adult')):
            prof = resolve(age, level)
            f = work_item('pressure_cover', prof, failureMode='no_pressure' if level == 1 else 'no_cover')
            pk = drills.select([f], prof, [f['id']])[0]
            texts[level] = pk['fields']['why']
        self.assertIn("This drill is for this match's focus", texts[1])
        self.assertTrue(texts[2].startswith('This drill is for the first priority: '))
        self.assertIn('nobody went to the ball', texts[1])
        self.assertIn('In 5 of the 9 situations PitchIQ measured when the other team had the ball (estimated), no '
                      'teammate was covering behind the player who went to the ball (3 separate spells).', texts[2])
        self.assertIn('(about 56%)', texts[5])
        # the example's 88.2 s of the phase give no rate (fix round 3, coach F-11: explain's rule, a rate only from 5
        # minutes of the phase, a whole number below 10); test_counts_follow_explains_rules has the rates
        self.assertIn("(3 separate spells).", texts[6])
        self.assertNotIn('per 10 minutes', texts[6])
        # the words of the level: the vacated space at level 3 ("channel" is a level-4 concept), the channel from 4
        for age, level, words in (('U12', 3, 'a defender moved forward and the space they left was not covered'),
                                  ('U14', 4, 'nobody was in the channel behind a defender who moved forward')):
            prof = resolve(age, level)
            f = work_item('wide_defender_advance', prof, family='left')
            self.assertIn(words, drills.select([f], prof, [f['id']])[0]['fields']['why'])
        # the success signal of the failed relation: no pressure -> the pressure signal of the 3v2 (not its cover one)
        prof = resolve('U10', 2)
        f = work_item('pressure_cover', prof, failureMode='no_pressure', pressurePresent=False)
        pk = drills.select([f], prof, [f['id']])[0]
        self.assertEqual(pk['drill'], 'l2_pressure_and_cover_3v2')
        self.assertIn('someone gives pressure in more of these moments', pk['fields']['progress'])

    def test_counts_follow_explains_rules(self):
        """Fix round 3 (coach F-11: 60 drill 'why' lines printed "(about 100%)" and 51 a rate such as "about 42.5 per
        10 minutes" from 3.8 minutes of play): a drill line states a share and a rate on explain's rules for the same
        numbers: no share at the ends (SHARE_ALL; "6 of the 6" says it) or one that is not the share of the counts the
        sentence states; a rate only from RATE_MIN_PHASE_S of the phase, a whole number below RATE_DECIMAL_PHASE_S;
        the spells only from 2."""
        from soccer.coaching import explain

        def why(prof, rec_share=None, **agg):
            f = work_item('pressure_cover', prof, claims=False)
            f['aggregate'].update(agg)
            f['claims'] = explain.build_claims(f, prof)
            if rec_share is not None:
                next(c for c in f['claims'] if c['type'] == 'recurrence')['facts']['failShare'] = rec_share
            pk = drills.select([f], prof, [f['id']])[0]
            self.check_pick(pk, f, prof, False)
            return pk['fields']['why']
        p6 = resolve('Adult', 6)
        text = why(p6, opportunities=6, demonstrated=0, notDemonstrated=6, unclear=0, distinctFailingSpells=4)
        self.assertIn('In 6 of the 6 situations PitchIQ measured', text)
        self.assertNotIn('%', text)
        for phase_s, words in ((88.2, None), (299.0, None), (300.0, 'about 3 per 10 minutes'),
                               (599.0, 'about 3 per 10 minutes'), (600.0, 'about 3.4 per 10 minutes')):
            text = why(p6, phaseSeconds=phase_s)
            if words is None:
                self.assertNotIn('per 10 minutes', text, phase_s)
            else:
                self.assertIn(words+" of the other team's possession (estimated)", text)
        self.assertEqual(explain.RATE_MIN_PHASE_S, 300.0)
        self.assertEqual(explain.RATE_DECIMAL_PHASE_S, 600.0)
        text = why(p6, distinctFailingSpells=1)
        self.assertNotIn('separate spell', text)
        self.assertIn('(about 56%)', why(resolve('U16', 5)))
        self.assertNotIn('%', why(resolve('U16', 5), rec_share=.9))      # not the share of 5 of the 9

    def test_progress_compares_only_with_a_judged_check(self):
        """Fix round 3 (coach F-8 = facts F-10: 144 of 419 optional drills, 115 of 196 run copies, said "... in more
        of these moments than in this match" for a check this match never decided): the optional drill compares with
        this match only when a reported finding of the check counts a decided situation in the check's own phase
        (decided_situations); else it gives the signal without the comparison. A work item always failed, so its
        drill compares (test_texts_follow_the_level)."""
        prof = resolve('U14', 4)
        no = {'skill': 'spread_when_attacking', 'text': 'x', 'fromMatch': False, 'finding': None}

        def progress(findings):
            sel = drills.selection(findings, prof, [], next_objective=no)
            self.assertEqual(sel['objective']['detector'], 'attacking_shape')
            pk = sel['picks'][0]
            self.assertEqual(S.validate(S.DrillPick, pk), [])
            return pk['fields']['progress']

        def shape(state, dem, failed):
            f = work_item('attacking_shape', prof, verdict='describe', family=f'{state}{dem}{failed}')
            f['context']['phase']['state'] = state
            f['aggregate'].update(demonstrated=dem, notDemonstrated=failed, unclear=1)
            return f
        neutral = progress([])
        self.assertTrue(neutral.startswith('What you can look for next match: when we have the ball, '), neutral)
        self.assertNotIn('this match', neutral)
        self.assertIn('PitchIQ will look at the same kind of situations then.', neutral)
        self.assertIn('than in this match', progress([shape('own', 1, 2)]))
        self.assertNotIn('this match', progress([shape('unknown', 1, 2)]))   # no moment "when we have the ball"
        self.assertNotIn('this match', progress([shape('own', 0, 0)]))       # only unclear moments
        self.assertEqual(drills.decided_situations([shape('own', 1, 2), shape('unknown', 4, 0), shape('own', 2, 0)],
                                                   'attacking_shape'), 5)
        self.assertEqual(drills.decided_situations([shape('own', 1, 2)], 'pressure_cover'), 0)
        bare_f = shape('own', 1, 2)
        bare_f['aggregate'].pop('notDemonstrated')
        self.assertEqual(drills.decided_situations([bare_f], 'attacking_shape'), 0)
        # over every age x level x format, the optional drill of the level's own objective never compares when the
        # report has no finding
        for age, level, fmt in itertools.product(S.AGE_GROUPS, range(1, 8), ALL_FORMATS):
            for pk in drills.selection([], resolve(age, level, fmt), [])['picks']:
                self.assertNotIn('this match', pk['fields']['progress'], (age, level, fmt))

    def test_deterministic_inputs_untouched_and_style_free(self):
        prof = resolve('U14', 4)
        f = work_item('wide_defender_advance', prof, family='left')
        before_f, before_p = copy.deepcopy(f), copy.deepcopy(prof)
        one = drills.select([f], prof, [f['id']])
        self.assertEqual(prof, before_p)
        self.assertEqual({k: v for k, v in f.items() if k != 'drills'},
                         {k: v for k, v in before_f.items() if k != 'drills'})
        self.assertEqual(drills.select([copy.deepcopy(before_f)], prof, [f['id']]), one)
        ids = {style: [pk['drill'] for pk in drills.select([copy.deepcopy(before_f)], resolve('U14', 4, style=style),
                                                           [f['id']])]
               for style in ('unknown', 'possession', 'counter', 'high_press', 'compact')}
        self.assertEqual(len({tuple(v) for v in ids.values()}), 1, ids)
        # claims missing: the stage builds them (explain.build_claims) and the pick is the same drill
        bare_f = work_item('wide_defender_advance', prof, family='left', claims=False)
        self.assertEqual([pk['drill'] for pk in drills.select([bare_f], prof, [bare_f['id']])],
                         [pk['drill'] for pk in one])

    def test_safety_enumeration_through_selection(self):
        """DESIGN 7.4 through the stage: 17 age groups x 7 levels x 5 formats (profiles.resolve) x every detector the
        level can criticise and the format can assess. Every pick passes the safety filter at its age (no heading drill
        at all, nothing above the load cap, no contact drill below U9, the drill's age range) and the format filter (no
        drill only for larger formats, none needing more players than one game of the format, no 11v11 role name in
        its fields below 11v11), is a drill of the detector, offered at the level or (when no drill of the level passes
        both filters) at the nearest lower level down to drills.fallback_floor, saying why; a work item without a pick
        has no drill passing both at its level or below. Measured (fix round 1): 1581 work items -> 1476 picks; the 105
        without are all refused by the safety filter at every level up to theirs (none lost its drill to the format
        filter); 307 picks come from a lower level (207 for the age, 98 for the format, 2 for both). Fix round 3: the
        same 1476 picks (the 101 from two to four levels down are all for teams younger than their level's typical
        ages, whose age-fitting drills fallback_floor keeps)."""
        n = picks = 0
        earlier, missing = Counter(), Counter()
        for age, level, fmt in itertools.product(S.AGE_GROUPS, range(1, 8), ALL_FORMATS):
            prof = resolve(age, level, fmt)
            for det, dp in prof['detectors'].items():
                if not {'work_on', 'try_this'} & set(dp['verdicts']) or not drills.format_supports(prof, det):
                    continue
                f = work_item(det, prof, claims=False)
                sel = drills.selection([f], prof, [f['id']])
                n += 1
                if not sel['picks']:
                    self.assertEqual(drills.ranked_candidates(self.library, prof, det)[0], [], (age, level, det))
                    why = sel['skipped'][0]['reason']
                    self.assertIn(why, ('refused_by_safety', 'no_drill_at_level', 'no_drill_for_format'))
                    self.assertIn(drills.no_drill_reason(self.library, prof, det), ('refused_by_safety',
                                                                                    'no_drill_at_level'))
                    missing[why] += 1
                for pk in sel['picks']:
                    picks += 1
                    d = drills.get(self.library, pk['drill'])
                    self.assertEqual(d['safety']['heading'], 'none')
                    self.assertEqual(drills.safety_problems(d, age, prof['safety']), [], (age, level, pk['drill']))
                    self.assertEqual(drills.format_problems(d, prof), [], (age, level, fmt, pk['drill']))
                    if prof['format']['name'] != '11v11':
                        self.assertIsNone(drills.ELEVEN_ROLES.search(' '.join(pk['fields'][k] for k in
                                                                              drills.FIELD_LIMITS)), pk['drill'])
                    self.assertIn(det, d['related_tactical_detector'])
                    if level not in d['development_level']:
                        self.assertLess(d['development_level'][-1], level)
                        self.assertGreaterEqual(d['development_level'][-1], drills.fallback_floor(prof))
                        why = drills.level_gap(self.library, prof, det) or 'none_at_level'
                        self.assertIn(drills.EARLIER_NOTES[why], pk['fields']['why'])
                        earlier[why] += 1
        self.assertEqual((n, picks), (1581, 1476))
        self.assertEqual(dict(missing), {'refused_by_safety': 105})
        self.assertEqual(dict(earlier), {'age': 207, 'format': 98, 'age_and_format': 2})


class SelectionSceneTests(unittest.TestCase):
    """EVAL's synthetic scenes through the real tiers (tests/coaching_detect_fixtures), assess and explain, then the
    stage: the drills a coach of each level would get for the planted patterns."""

    @classmethod
    def setUpClass(cls):
        import coaching_detect_fixtures as F
        cls.F = F

    def run_level(self, scene, variant, level):
        from soccer.coaching import assess, explain
        obs, roles = self.F.tiers(scene, variant)
        prof = self.F.profile(scene, variant, level=level)
        out = assess.assess(obs, roles, prof)
        ex = explain.explain(out['findings'], prof, obs=obs, roles=roles)
        sel = drills.selection(ex['findings'], prof, out['priorities'], next_objective=out['nextObjective'])
        for pk in sel['picks']:
            self.assertEqual(S.validate(S.DrillPick, pk), [])
        return out, ex['findings'], sel

    def picks_for(self, sel, findings, detectors, family=None):
        """The picks of reported work items of these detectors (or, with family, of that finding family)."""
        by = {f['id']: f for f in findings}
        out = []
        for pk in sel['picks']:
            f = by.get(pk['finding'])
            if f is not None and (f['observation'] in detectors or
                                  (family and f['key'].split('|')[2] == family)):
                out.append((pk, f))
        return out

    def test_unsupported_advance_by_level(self):
        """DESIGN 13.1 row 2 (nobody behind the ball): wherever the tiers report the planted advances as a work item,
        it gets the level's drill, with the same evidence moments at every level from 2; levels 4-6 always do. Which
        levels criticise them is DETECT's rule (fix round 1, B-1: the level 1-2 reframe now needs the ball in the final
        third), not this stage's; a level with no work item gets only the optional drill, for assess's next objective
        (the fix round 1 measurement: levels 3-6 criticise, 1-2 and 7 do not)."""
        expected = {1: 'l1_help_your_teammate_2v2', 2: 'l2_someone_stays_back_4v4', 3: 'l3_wide_and_deep_9v9_shape',
                    4: 'l4_defensive_unit_6v6', 5: 'l5_transition_and_cover_8v8', 6: 'l5_transition_and_cover_8v8'}
        moments, levels = set(), []
        for level in range(1, 8):
            out, findings, sel = self.run_level('unsupported_advance', 'none_behind', level)
            adv = self.picks_for(sel, findings, ('wide_defender_advance',), family='advance')
            for pk, f in adv:
                self.assertEqual(pk['drill'], expected[level], level)
                self.assertIn(pk['drill'], f['drills'])
            if adv:
                levels.append(level)
                if level >= 2:
                    moments.add(tuple(adv[0][0]['fields']['moments']))
            if not any(pk['finding'] for pk in sel['picks']):
                self.assertTrue(all(pk['optional'] for pk in sel['picks']))
                no = out['nextObjective']
                if not no['fromMatch']:
                    self.assertEqual((sel['objective'] or {}).get('skill'), no['skill'], level)
        self.assertTrue({4, 5, 6} <= set(levels), levels)
        self.assertEqual(len(moments), 1)

    def test_pressure_without_cover_and_single_advance(self):
        """Pressure without cover: wherever the tiers report it as a work item, the level's pressure-and-cover drill
        for the failed relation 'cover' (DESIGN 7.3, wave 5's measured table), at level 2 worded "no teammate was
        covering behind the player who went to the ball". DESIGN 13.1 row 3 (truth 'no_drill'): one advance is never
        a work item, so no drill rests on it at any level."""
        expected = {2: 'l2_pressure_and_cover_3v2', 3: 'l3_defensive_recovery_5v4', 4: 'l4_cover_the_line_3v3',
                    5: 'l4_cover_the_line_3v3', 6: 'l6_block_compactness_shadow'}
        levels = []
        for level in range(2, 7):
            out, findings, sel = self.run_level('defending', 'press_no_cover', level)
            prof = self.F.profile('defending', 'press_no_cover', level=level)
            for pk, f in self.picks_for(sel, findings, ('pressure_cover',)):
                levels.append(level)
                self.assertEqual(pk['drill'], expected[level], level)
                self.assertEqual(drills.failed_relations(f, prof), ['cover'])
                if level == 2:
                    self.assertIn('no teammate was covering behind the player who went to the ball',
                                  pk['fields']['why'])
        self.assertTrue(levels)
        for level in range(1, 8):
            out, findings, sel = self.run_level('single_advance', None, level)
            self.assertEqual(self.picks_for(sel, findings, ('wide_defender_advance',), family='advance'), [], level)
            self.assertTrue(all(f['drills'] == [] for f in findings
                                if f['observation'] == 'wide_defender_advance' or f['key'].split('|')[2] == 'advance'))

    def test_kids_swarm_and_the_orchestrator(self):
        """DESIGN 13.1 row 12 (4v4 swarm, "at most one try_this" at levels 1-2): a reported shape work item gets the
        spread-out game, and every drill fits 4v4; the real stage inside coaching.interpret (later stages stood in)
        gets the explained findings and assess's priorities."""
        for level in (1, 2):
            out, findings, sel = self.run_level('kids_swarm', None, level)
            prof = self.F.profile('kids_swarm', None, level=level)
            self.assertLessEqual(len([f for f in findings if f['verdict'] in S.CRITICISM]), 1)
            for pk, f in self.picks_for(sel, findings, ('attacking_shape',)):
                self.assertEqual(pk['drill'], 'l1_spread_out_4v4')
            self.assertTrue(sel['picks'])
            for pk in sel['picks']:
                self.assertEqual(drills.format_problems(drills.get(drills.load_library(), pk['drill']), prof), [])
        from soccer import coaching
        obs, roles = self.F.tiers('unsupported_advance', 'none_behind')
        captured = {}

        def report(**kw):
            captured.update(kw)
            return {}
        stages = {'alternatives': lambda finding, obs, roles, profile: None,
                  'annotate': lambda findings, obs, roles, profile, alternatives: [], 'report': report}
        st = copy.deepcopy(self.F.prepared('unsupported_advance', 'none_behind')[1])
        st['teams']['A']['playingLevel'], st['teams']['A']['ageGroup'] = 4, 'U14'
        coaching.interpret(obs, roles, st, 'A', stages=stages, validate=False)
        self.assertEqual([pk['drill'] for pk in captured['drills']], ['l4_defensive_unit_6v6'])
        work = [f for f in captured['findings'] if f['verdict'] in S.CRITICISM]
        self.assertEqual([f['drills'] for f in work], [['l4_defensive_unit_6v6']])
        self.assertEqual(captured['drills'][0]['finding'], captured['assessment']['priorities'][0])

    def test_format_unsupported_matches_the_measurement_tier(self):
        """drills.FORMAT_UNSUPPORTED is a copy of the measurement tier's rule: on the 4v4 scene the checks whose
        coverage is format_not_supported are exactly D3 and D5; on the 7v7 scenes none is."""
        def unsupported(scene, variant):
            obs, roles = self.F.tiers(scene, variant)
            cov = dict(obs.get('detectors') or {})
            cov.update(roles.get('detectors') or {})
            return {d for d, c in cov.items() if (c or {}).get('dominantReason') == 'format_not_supported'}
        self.assertEqual(unsupported('kids_swarm', None),
                         {d for d, fmts in drills.FORMAT_UNSUPPORTED.items() if 4 in fmts})
        for scene, variant in (('small_field_scale', 'declared'), ('amateur_coach_spans', 'spans')):
            self.assertEqual(unsupported(scene, variant), set(), scene)
        self.assertTrue(all(7 not in fmts and 11 not in fmts for fmts in drills.FORMAT_UNSUPPORTED.values()))

    def test_kids_swarm_at_every_level_stays_in_its_format(self):
        """The 4v4 scene (U.S. 4v4, no keeper) through the real tiers at levels 1-7: every drill fits 4v4 (8
        players; before fix round 1, levels 3, 4, 6 and 7 got 9v9 / 6v6 / 8v8 / 11v11 drills), and the next
        objective is never the wide-defender check 4v4 cannot assess (it was at levels 4 and 6)."""
        for level in range(1, 8):
            out, findings, sel = self.run_level('kids_swarm', None, level)
            prof = self.F.profile('kids_swarm', None, level=level)
            for pk in sel['picks']:
                self.assertEqual(drills.format_problems(drills.get(drills.load_library(), pk['drill']), prof), [],
                                 (level, pk['drill']))
            if sel['objective'] is not None:
                self.assertTrue(drills.format_supports(prof, sel['objective']['detector']), level)


class FormatTests(unittest.TestCase):
    """Fix round 1 (critic S-9 drills part, coach verifier M3 and M5, sources verifier F9): drill choice and the next
    objective respect the match format and its player count; a demonstrated check is not the next objective."""

    @classmethod
    def setUpClass(cls):
        cls.library = drills.load_library()

    def get(self, did):
        return drills.get(self.library, did)

    def test_format_filter_rules(self):
        P = (lambda n: profile('Adult', 4, n))
        g = self.get
        self.assertEqual(drills.format_problems(g('l3_wide_and_deep_9v9_shape'), P(4)),
                         ['above_match_format', 'more_players_than_format'])
        self.assertEqual(drills.format_problems(g('l3_wide_and_deep_9v9_shape'), P(9)), [])
        self.assertEqual(drills.format_problems(g('l7_tactical_phase_11v11'), P(7)),
                         ['above_match_format', 'more_players_than_format'])
        self.assertEqual(drills.format_problems(g('l6_block_compactness_shadow'), P(9)), ['above_match_format'])
        self.assertEqual(drills.format_problems(g('l4_defensive_unit_6v6'), P(7)), [])    # 6 v 6 + keepers: 7-a-side
        self.assertEqual(drills.format_problems(g('l4_defensive_unit_6v6'), P(5)),
                         ['above_match_format', 'more_players_than_format'])
        self.assertEqual(drills.format_problems(g('l4_cover_the_line_3v3'), P(4)), [])    # a 3 v 3 game
        self.assertEqual(drills.format_problems(g('l1_spread_out_4v4'), P(5)), [])        # 5v5 plays 4v4 drills
        self.assertEqual(drills.format_problems(g('l1_help_your_teammate_2v2'), P(11)), [])   # a smaller game
        d = copy.deepcopy(g('l4_cover_the_line_3v3'))
        d['player_count']['min'] = 15
        self.assertEqual(drills.format_problems(d, P(7)), ['more_players_than_format'])
        self.assertEqual([d['drill_id'] for d in self.library['drills'] if drills.format_problems(d, P(11))], [])
        self.assertEqual([drills.team_players(P(n)) for n in FORMATS], [4, 5, 7, 9, 11])
        self.assertEqual(drills.team_players({}), 11)

    def test_candidates_never_above_the_format(self):
        """Every age x level x format x detector: the candidates are exactly the safety-allowed drills of the level
        that fit the format; at 11v11 the format filter removes nothing."""
        removed = 0
        for age, level, players in itertools.product(S.AGE_GROUPS, range(1, 8), FORMATS):
            prof = profile(age, level, players)
            for det in S.DETECTORS:
                got = sorted(c['drill_id'] for c in drills.candidates(self.library, prof, det))
                allowed = [d for d in self.library['drills'] if det in d['related_tactical_detector'] and
                           level in d['development_level'] and not drills.safety_problems(d, age)]
                fits = sorted(d['drill_id'] for d in allowed if not drills.format_problems(d, prof))
                self.assertEqual(got, fits, (age, level, players, det))
                if players == 11:
                    self.assertEqual(len(fits), len(allowed))
                removed += len(allowed)-len(fits)
        self.assertGreater(removed, 0)

    def test_small_formats_get_a_drill_of_their_format(self):
        """The coach verifier's M3 cases as synthetic work items: a 4v4 team at level 3 was given the 9v9 shape
        drill, a 7v7 team at levels 5-6 the 8v8 (16-22 players) and at level 7 the 11v11 phase training; now each gets
        a drill of its format, from an earlier level when its own has none, and the text says why. Fix round 3 (coach
        F-6): not from below drills.fallback_floor: the adult 7v7 team at level 7 no longer gets the level 4-5 6v6
        unit drill (two levels down); it gets none, and why is 'only_far_below' ("No drill in the library fits ... at
        this level" in the report's words)."""
        from soccer.coaching import report
        prof = resolve('Adult', 7, FORMATS[7])
        f = work_item('wide_defender_advance', prof, verdict='work_on')
        sel = drills.selection([f], prof, [f['id']])
        self.assertEqual((sel['picks'], sel['skipped']), ([], [{'finding': f['id'], 'reason': 'only_far_below'}]))
        self.assertEqual(drills.fallback_floor(prof), 6)
        self.assertEqual(drills.reachable_below(self.library, prof, 'wide_defender_advance')[0], 5)
        self.assertEqual(drills.no_drill_reason(self.library, prof, 'wide_defender_advance'), 'no_drill_at_level')
        self.assertIn('only_far_below', drills.SKIP_REASONS)
        self.assertIn(drills.no_drill_reason(self.library, prof, 'wide_defender_advance'), report.NO_DRILL_WORDS)
        cases = (('U12', 3, 4, 'attacking_shape', 'l1_spread_out_4v4', 'format'),
                 ('U16', 5, 7, 'wide_defender_advance', 'l4_defensive_unit_6v6', None),
                 ('Adult', 6, 7, 'wide_defender_advance', 'l4_defensive_unit_6v6', 'format'),
                 ('U8', 4, 5, 'behind_ball', 'l2_someone_stays_back_4v4', 'age'),
                 ('U11', 4, 4, 'pressure_cover', 'l2_pressure_and_cover_3v2', 'age_and_format'))
        for age, level, players, det, drill, why in cases:
            prof = resolve(age, level, FORMATS[players])
            f = work_item(det, prof, verdict='work_on' if level > 2 else 'try_this')
            sel = drills.selection([f], prof, [f['id']])
            self.assertEqual([pk['drill'] for pk in sel['picks']], [drill], (age, level, players, det))
            self.assertEqual(drills.format_problems(self.get(drill), prof), [])
            text = sel['picks'][0]['fields']['why']
            if why is None:
                self.assertFalse(any(n in text for n in drills.EARLIER_NOTES.values()), text)
            else:
                self.assertEqual(drills.level_gap(self.library, prof, det), why)
                self.assertIn(drills.EARLIER_NOTES[why], text)
            self.assertNotIn('more players than your match format', sel['picks'][0]['fields']['trains'])

    def test_no_drill_for_the_format_is_said(self):
        """Nothing at the level or below passes both filters: the work item keeps no drill, selection says why
        ('no_drill_for_format'), and no_drill_reason gives a code the report words (D5 at 4v4 as a synthetic work item:
        the measurement tier never makes one, the library has no D5 drill for 4v4)."""
        from soccer.coaching import report
        prof = resolve('U16', 5, FORMATS[4])
        f = work_item('defensive_spacing', prof)
        sel = drills.selection([f], prof, [f['id']])
        self.assertEqual((sel['picks'], sel['skipped']), ([], [{'finding': f['id'], 'reason': 'no_drill_for_format'}]))
        self.assertEqual(f['drills'], [])
        self.assertEqual(drills.level_gap(self.library, prof, 'defensive_spacing'), 'format')
        self.assertEqual(drills.no_drill_reason(self.library, prof, 'defensive_spacing'), 'no_drill_at_level')
        self.assertIn(drills.no_drill_reason(self.library, prof, 'defensive_spacing'), report.NO_DRILL_WORDS)
        self.assertIn('no_drill_for_format', drills.SKIP_REASONS)
        # the age alone: as before
        young = resolve('U10', 4, FORMATS[11])
        self.assertEqual(drills.no_drill_detail(self.library, young, 'weak_side_balance'), 'refused_by_safety')

    def test_next_objective_respects_the_format(self):
        """Coach verifier M3: 4v4 teams at levels 4 and 6 were sent to "Support behind advancing wide defenders",
        a check 4v4 cannot assess (D3 format_not_supported); the objective is the next check of the level, and its
        optional drill fits the format. A passed NextObjective naming such a check is not used. Over every age x level
        x format the objective is never off, never unnamed, never unsupported by the format."""
        for age, level, players in (('U8', 4, 4), ('Adult', 6, 4), ('U9', 4, 5), ('Adult', 6, 5)):
            prof = resolve(age, level, FORMATS[players])
            ranks = sorted(((dp['params'].get('objectiveRank') or 99), d) for d, dp in prof['detectors'].items()
                           if dp['enabled'])
            self.assertEqual(ranks[0][1], 'wide_defender_advance')          # the rank alone would pick D3
            self.assertEqual(drills.objective_exclusion(prof, 'wide_defender_advance'), 'format_not_supported')
            sel = drills.selection([], prof, [])
            self.assertEqual(sel['objective'], {'skill': 'someone_back_when_attacking', 'detector': 'behind_ball'})
            if age == 'Adult':
                # fix round 3 (coach F-6): the only 4v4 / 5v5 drill for it is the level 2-3 game, three levels below
                # an adult level-6 team: none is offered
                self.assertEqual(sel['picks'], [])
                self.assertEqual(drills.no_drill_detail(self.library, prof, 'behind_ball'), 'only_far_below')
            else:
                pk = sel['picks'][0]
                self.assertEqual(drills.format_problems(self.get(pk['drill']), prof), [])
                self.assertIn('behind_ball', self.get(pk['drill'])['related_tactical_detector'])
                # the drill's secondary skill on wide defenders is not named either (it was: "Also: Support behind
                # advancing wide defenders")
                self.assertNotIn('wide defender', pk['fields']['trains'].lower())
                self.assertIn('wide_defender_support', self.get(pk['drill'])['secondary_skills'])
            no = {'skill': 'wide_defender_support', 'text': 'x', 'fromMatch': False, 'finding': None}
            self.assertEqual(drills.selection([], prof, [], next_objective=no)['objective']['detector'],
                             'behind_ball')
            p11 = resolve(age, level, FORMATS[11])                          # 11v11 keeps the level's first check
            self.assertEqual(drills.selection([], p11, [])['objective']['detector'], 'wide_defender_advance')
        n = 0
        for age, level, fmt in itertools.product(S.AGE_GROUPS, range(1, 8), ALL_FORMATS):
            prof = resolve(age, level, fmt)
            o = drills.level_objective([], prof)
            if o is None:
                continue
            n += 1
            self.assertIsNone(drills.objective_exclusion(prof, o['detector']), (age, level, fmt))
            self.assertTrue(drills.format_supports(prof, o['detector']))
        self.assertEqual(n, 17*7*5)

    def test_a_demonstrated_check_is_not_the_next_objective(self):
        """Coach verifier M5 (praise-first): a check this match demonstrated is not named as the next objective: a
        reported strength (the old rule), reported findings counting at least one demonstrated situation and none
        not demonstrated (assess's rule, on the findings this stage sees: e.g. a check the level only describes), or
        a check the caller reports demonstrated (assess sees every classified family; this stage only the reported
        findings). A passed NextObjective naming a strength is not used either."""
        prof = resolve('U14', 4)
        top = drills.level_objective([], prof)['detector']
        strong = work_item(top, prof, verdict='strength', family='s')
        self.assertEqual(drills.demonstrated_checks([strong]), {top})
        self.assertNotEqual(drills.level_objective([strong], prof)['detector'], top)

        def desc(dem, failed, family='unknown'):
            f = work_item(top, prof, verdict='describe', family=family)
            f['aggregate'].update(demonstrated=dem, notDemonstrated=failed, distinctDemonstratedSpells=dem)
            return f
        self.assertEqual(drills.demonstrated_checks([desc(1, 0)], prof), {top})
        self.assertNotEqual(drills.level_objective([desc(3, 0)], prof)['detector'], top)
        self.assertEqual(drills.demonstrated_checks([desc(0, 0)], prof), set())         # nothing decided
        self.assertEqual(drills.demonstrated_checks([desc(3, 1)], prof), set())         # something failed
        self.assertEqual(drills.demonstrated_checks([desc(2, 0), desc(1, 1, 'other')], prof), set())
        review = work_item(top, prof, verdict='review', family='r')
        self.assertEqual(drills.demonstrated_checks([desc(3, 0), review], prof), set())
        self.assertEqual(drills.level_objective([desc(3, 0), review], prof)['detector'], top)
        no_counts = desc(3, 0)
        no_counts['aggregate'].pop('notDemonstrated')
        self.assertEqual(drills.demonstrated_checks([no_counts], prof), set())
        self.assertNotEqual(drills.level_objective([], prof, demonstrated={top})['detector'], top)
        self.assertEqual(drills.objective_exclusion(prof, top, {top}), 'demonstrated')
        passed = {'skill': drills.SKILL_OF[top], 'text': 'x', 'fromMatch': False, 'finding': None}
        self.assertNotEqual(drills.selection([strong], prof, [], next_objective=passed)['objective']['detector'], top)
        self.assertEqual(drills.selection([], prof, [], next_objective=passed)['objective']['detector'], top)
        # assess chose no objective (it sees every classified family, e.g. all demonstrated): no optional drill
        none = {'skill': None, 'text': 'x', 'fromMatch': False, 'finding': None}
        self.assertEqual(drills.selection([], prof, [], next_objective=none), {'picks': [], 'skipped': [],
                                                                                'objective': None})
        # the optional drill follows: it trains the next check, not the demonstrated one
        sel = drills.selection([desc(3, 0)], prof, [])
        self.assertNotEqual(sel['objective']['detector'], top)
        self.assertIn(sel['objective']['detector'], self.get(sel['picks'][0]['drill'])['related_tactical_detector'])
        # every check demonstrated or unavailable: no objective, no optional drill
        every = {d for d, dp in prof['detectors'].items() if dp['enabled']}
        self.assertIsNone(drills.level_objective([], prof, demonstrated=every))
        self.assertEqual(drills.selection([], dict(prof, detectors={d: dict(dp, enabled=False) for d, dp in
                                                                    prof['detectors'].items()}), [])['picks'], [])


class FallbackTests(unittest.TestCase):
    """Fix round 3 (coach F-6, critic S-3): when a level has no drill for a check, the fallback offered adult level
    4-7 teams the U11-U12 9v9 shape drill and the U7-U8 no-tackle 4v4 game (50 reports in the coach verifier's set),
    saying "No drill of this level is allowed at this age", which was false: those levels have no attacking-shape
    drill at all. Now a drill comes from the level just below, or no lower than the typical level of the team's age
    (drills.fallback_floor: a team younger than its level keeps an age-fitting drill), and the reason stated is the
    true one ('none_at_level' when the level has no drill of the check); otherwise none, and no_drill_detail says
    why in words true of the whole library."""

    @classmethod
    def setUpClass(cls):
        cls.library = drills.load_library()

    def optional(self, age, level, players, skill):
        prof = resolve(age, level, FORMATS[players])
        no = {'skill': skill, 'text': 'x', 'fromMatch': False, 'finding': None}
        return prof, drills.selection([], prof, [], next_objective=no)

    def test_the_coach_verifiers_cases(self):
        for age, level, players in (('Adult', 7, 11), ('Adult', 6, 11), ('Adult', 5, 11), ('U16', 5, 11),
                                    ('U18', 5, 11), ('Adult', 6, 4), ('Adult', 7, 4), ('Adult', 7, 7)):
            prof, sel = self.optional(age, level, players, 'spread_when_attacking')
            self.assertEqual(sel['objective']['detector'], 'attacking_shape')
            self.assertEqual(sel['picks'], [], (age, level, players))
            self.assertEqual(sel['skipped'], [{'finding': None, 'reason': 'only_far_below'}])
            self.assertEqual(drills.no_drill_detail(self.library, prof, 'attacking_shape'), 'only_far_below')
            self.assertEqual(drills.no_drill_reason(self.library, prof, 'attacking_shape'), 'no_drill_at_level')
        for age in ('U14', 'Adult'):            # one level below, with the true reason
            prof, sel = self.optional(age, 4, 11, 'spread_when_attacking')
            why = sel['picks'][0]['fields']['why']
            self.assertEqual(sel['picks'][0]['drill'], 'l3_wide_and_deep_9v9_shape')
            self.assertIsNone(drills.level_gap(self.library, prof, 'attacking_shape'))
            self.assertIn(drills.EARLIER_NOTES['none_at_level'], why)
            self.assertNotIn(drills.EARLIER_NOTE, why)
        # a team younger than its level keeps the age-fitting drill two levels down, with the true reason (the age)
        prof = resolve('U8', 5, FORMATS[4])
        f = work_item('pressure_cover', prof)
        sel = drills.selection([f], prof, [f['id']])
        self.assertEqual([pk['drill'] for pk in sel['picks']], ['l2_pressure_and_cover_3v2'])
        self.assertEqual(drills.level_gap(self.library, prof, 'pressure_cover'), 'age')
        self.assertIn(drills.EARLIER_NOTE, sel['picks'][0]['fields']['why'])
        floors = {(a, lv): drills.fallback_floor(profile(a, lv)) for a, lv in
                  (('Adult', 7), ('Adult', 4), ('U14', 6), ('U8', 4), ('U8', 1), ('U11', 5))}
        self.assertEqual(floors, {('Adult', 7): 6, ('Adult', 4): 3, ('U14', 6): 4, ('U8', 4): 1, ('U8', 1): 1,
                                  ('U11', 5): 3})

    def test_every_fallback_is_near_and_true(self):
        """Every age x level x format x check that can be the next objective (objective_exclusion): the first
        candidate is of the level, or of the nearest lower level that has one, never below fallback_floor, with the
        true reason (level_gap, else 'none_at_level' exactly when no drill of the level trains the check); with none,
        no_drill_detail's code is true of every drill of the check at the level and below. Measured (Oct 24): 2720
        profile/check pairs, 1345 of the level, 335 one level down and 430 further down for teams younger than their
        level (all at or above their age's typical level); 610 without a drill: 300 only further down than the floor
        allows, 225 refused by the safety filter, 85 with no drill at the level or below that passes both filters
        (before: 2410 drills, 730 of them two to five levels down, 340 of those with a false reason)."""
        tally = Counter()
        for age, level, fmt in itertools.product(S.AGE_GROUPS, range(1, 8), ALL_FORMATS):
            prof = resolve(age, level, fmt)
            floor = drills.fallback_floor(prof)
            for det in prof['detectors']:
                if drills.objective_exclusion(prof, det) is not None:
                    continue        # off, unnamed or not assessable on the format: never an objective or a finding
                cands, earlier = drills.ranked_candidates(self.library, prof, det)
                if not cands:
                    detail = drills.no_drill_detail(self.library, prof, det)
                    tally['none:'+detail] += 1
                    offered = [d for d in self.library['drills'] if det in d['related_tactical_detector'] and
                               min(d['development_level']) <= level]
                    reach = drills.reachable_below(self.library, prof, det)
                    if detail == 'only_far_below':
                        self.assertTrue(reach and max(reach) < floor, (age, level, fmt, det))
                    else:
                        self.assertEqual(reach, [], (age, level, fmt, det))
                    if detail == 'refused_by_safety':
                        self.assertTrue(offered and all(drills.safety_problems(d, age) for d in offered))
                    if detail == 'no_drill_for_format':
                        self.assertTrue(offered and all(drills.format_problems(d, prof) for d in offered))
                    continue
                top = max(cands[0]['drill']['development_level'])
                if level in cands[0]['drill']['development_level']:
                    self.assertFalse(earlier)
                    tally['level'] += 1
                    continue
                self.assertGreaterEqual(top, floor, (age, level, fmt, det))
                for lv in range(level-1, top, -1):
                    self.assertEqual(drills.candidates(self.library, dict(prof, level=lv), det), [])
                self.assertEqual(earlier, drills.level_gap(self.library, prof, det) or 'none_at_level')
                has_level = any(level in d['development_level'] and det in d['related_tactical_detector']
                                for d in self.library['drills'])
                self.assertEqual(earlier == 'none_at_level', not has_level)
                if top < level-drills.MAX_LEVELS_BELOW:
                    from soccer.coaching import profiles as PR
                    self.assertGreaterEqual(top, PR.typical_level(age), (age, level, fmt, det))   # age-fitting
                    tally['further'] += 1
                else:
                    tally['below1'] += 1
        self.assertEqual(dict(tally), {'level': 1345, 'below1': 335, 'further': 430, 'none:only_far_below': 300,
                                       'none:refused_by_safety': 225, 'none:no_drill_at_level': 85})


# ================================ v1 (deprecated; drills_v1.json) ================================
REQUIRED = {'defensive_compactness': 'compact_lines', 'lateral_shifting': 'lateral_shift',
            'back_line_coordination': 'back_line_coordination', 'support_angles': 'support_distances',
            'formation_shadow_play': 'formation_organisation', 'off_ball_movement': 'support_distances'}


class LibraryV1Tests(unittest.TestCase):
    def setUp(self):
        self.library = drills.load_drills()

    def test_library_validates_and_has_the_requested_exercises(self):
        self.assertEqual(drills.validate_drills(self.library), [])
        ids = [d['id'] for d in self.library['drills']]
        self.assertEqual(sorted(ids), sorted(REQUIRED))
        for did, key in REQUIRED.items():
            d = drills.drill(self.library, did)
            self.assertIn(key, d['recommendationKeys'], did)
            for f in ('name', 'purpose', 'tacticalIssue', 'area', 'setup', 'source'):
                self.assertGreater(len(d[f]), 10, f'{did}.{f}')
            for f in ('instructions', 'coachingPoints', 'safety', 'equipment', 'regressions', 'successCriteria'):
                self.assertGreaterEqual(len(d[f]), 2 if f != 'instructions' else 3, f'{did}.{f}')
            self.assertTrue(any(drills.REPORT_LINK in c for c in d['successCriteria']), did)
            self.assertTrue(any(re.search(r'\d', c) for c in d['successCriteria']), f'{did}: measurable')
            self.assertLessEqual(d['players']['min'], d['players']['max'])
            self.assertGreaterEqual(d['durationMinutes']['min'], 10)
            self.assertEqual(set(d['variations']), set(language.LEVELS))
            self.assertTrue(d['source'].startswith('Conventional'), did)

    def test_every_actionable_key_has_an_exercise(self):
        for key in language.RECOMMENDATION_KEYS:
            if key == 'ball_context':                  # descriptive only: never an improvement
                continue
            self.assertTrue(drills.drills_for(self.library, key), key)
        for key, by_cls in language.TEMPLATES.items():
            if 'improvement' in by_cls or 'review' in by_cls:
                self.assertTrue(drills.drills_for(self.library, key), key)

    def test_text_is_printable_ascii_and_reads_safely(self):
        strings = drills.all_text(self.library)
        self.assertGreater(len(strings), 100)
        for path, text in strings:
            self.assertRegex(text, r'^[ -~]*$', path)
            text.encode('cp1252')
            self.assertEqual(language.wording_problems(text, practice=True), [], f'{path}: {text}')
        for d in self.library['drills']:
            self.assertIsNone(re.search(r'\d', d['name']+d['purpose']), d['id'])
            self.assertTrue(any('warm up' in s.lower() for s in d['safety']), d['id'])
            self.assertTrue(any('age' in s.lower() for s in d['safety']), d['id'])

    def test_choose_respects_the_key(self):
        obs = {'recommendationKey': 'compact_lines', 'pattern': 'space_between_midfield_and_forwards',
               'drillIds': ['support_angles']}
        self.assertEqual(drills.choose(self.library, obs)['id'], 'off_ball_movement')   # pattern first
        obs['pattern'] = 'space_between_defence_and_midfield'
        self.assertEqual(drills.choose(self.library, obs)['id'], 'defensive_compactness')
        obs['drillIds'] = ['off_ball_movement']
        self.assertEqual(drills.choose(self.library, obs)['id'], 'off_ball_movement')   # the detector's choice
        self.assertIsNone(drills.choose(self.library, {'recommendationKey': 'ball_context'}))
        self.assertIsNone(drills.choose(self.library, {'recommendationKey': None}))
        self.assertIsNone(drills.choose({'drills': []}, obs))

    def test_drill_ref_carries_the_level_variation(self):
        d = drills.drill(self.library, 'support_angles')
        ref = drills.drill_ref(d, 'development')
        self.assertEqual(ref['variation'], d['variations']['development'])
        self.assertEqual(ref['level'], 'development')
        self.assertNotIn('variations', ref)
        ref['instructions'].append('changed')
        self.assertNotIn('changed', d['instructions'])
        self.assertEqual(drills.drill_numbers(d), [5.0, 14.0, 15.0, 20.0])
        for field in ('area', 'equipment', 'regressions', 'successCriteria'):
            self.assertEqual(ref[field], d[field], field)

    def test_every_game_fits_the_player_range(self):
        for d in self.library['drills']:
            texts = list(d['variations'].values())+[d['players'].get('note') or '']
            for n in drills.versus_counts(' '.join(texts)):
                self.assertTrue(d['players']['min'] <= n <= d['players']['max'], (d['id'], n))
        self.assertEqual(drills.versus_counts('4 v 1 or 5 v 2, then 4 v 4'), [5, 7, 8])
        self.assertIn('4 v 1', drills.drill(self.library, 'support_angles')['players']['note'])   # the variation's

    def test_validation_catches_problems(self):
        good = copy.deepcopy({k: v for k, v in self.library.items() if not k.startswith('_')})
        self.assertEqual(drills.validate_drills(good), [])
        cases = {
            'non-ASCII': lambda d: d.update(setup='Mark a 40 \u00d7 30 m area.'),
            'digits in a name': lambda d: d.update(name='Rondo 4v2'),
            'a missing field': lambda d: d.pop('safety'),
            'an unknown key': lambda d: d.update(recommendationKeys=['press_higher']),
            'bad players': lambda d: d.update(players={'min': 12, 'max': 8}),
            'a long duration': lambda d: d.update(durationMinutes={'min': 30, 'max': 90}),
            'a missing level': lambda d: d['variations'].pop('advanced'),
            'a bad id': lambda d: d.update(id='Compact Shape'),
            'no equipment': lambda d: d.pop('equipment'),
            'no regressions': lambda d: d.update(regressions=[]),
            'a game too small for the range': lambda d: d['variations'].update(development='Play 2 v 2.'),
            'no criterion in the report': lambda d: d.update(successCriteria=['Win the ball 3 times.']),
        }
        for what, change in cases.items():
            lib = copy.deepcopy(good)
            change(lib['drills'][0])
            self.assertTrue(drills.validate_drills(lib), what)
        lib = copy.deepcopy(good)
        lib['drills'].append(copy.deepcopy(lib['drills'][0]))
        self.assertIn('duplicate id', ' '.join(drills.validate_drills(lib)))
        self.assertTrue(drills.validate_drills({'version': 1, 'drills': []}))

    def test_load_refuses_an_invalid_file_and_hashes_a_valid_one(self):
        self.assertRegex(drills.library_sha(self.library), r'^[0-9a-f]{16}$')
        self.assertEqual(drills.library_sha({'drills': []}), 'inline')
        good = {k: v for k, v in self.library.items() if not k.startswith('_')}
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'drills.json')
            bad = copy.deepcopy(good)
            bad['drills'][0]['name'] = 'Rondo 4v2'
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(bad, f)
            with self.assertRaises(ValueError):
                drills.load_drills(path)
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(good, f)
            os.utime(path, ns=(1, 2))                  # a different stamp than the failed load
            self.assertEqual(len(drills.load_drills(path)['drills']), 6)
        self.assertIs(drills.load_drills(), drills.load_drills())   # cached per file stamp
        self.assertEqual(drills.DRILLS_PATH, drills.V1_LIBRARY_PATH)

    def test_v1_and_v2_caches_are_separate(self):
        v1, v2 = drills.load_drills(), drills.load_library()
        self.assertEqual(v1['version'], 1)
        self.assertEqual(v2['version'], 2)
        self.assertIs(drills.load_drills(), v1)
        self.assertIs(drills.load_library(), v2)


if __name__ == '__main__':
    unittest.main()
