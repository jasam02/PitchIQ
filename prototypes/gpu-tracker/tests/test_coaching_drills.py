"""The drill library (soccer/coaching/drills.json, drills.py): the six conventional exercises the report links to,
each complete (purpose, issue, area, equipment, set-up, instructions, player count, duration, coaching points,
progressions and regressions, a success criterion in the exercise and one in the coaching report's measurement,
variations per context level, safety), with every "N v M" game inside its player range (the first library asked for
4 v 4 in an exercise for at least 12 players), plain ASCII, linked only to recommendation keys that have wording, and
chosen for a finding only when written for its key.

Run:  python -m unittest test_coaching_drills          (from this directory)
"""
import copy
import json
import os
import re
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from soccer.coaching import drills, language  # noqa: E402

REQUIRED = {'defensive_compactness': 'compact_lines', 'lateral_shifting': 'lateral_shift',
            'back_line_coordination': 'back_line_coordination', 'support_angles': 'support_distances',
            'formation_shadow_play': 'formation_organisation', 'off_ball_movement': 'support_distances'}


class LibraryTests(unittest.TestCase):
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

    def test_file_is_crlf_ascii_json(self):
        data = drills.DRILLS_PATH.read_bytes()
        data.decode('ascii')
        lines = data.split(b'\n')
        self.assertTrue(all(line.endswith(b'\r') for line in lines[:-1]))
        json.loads(data)


if __name__ == '__main__':
    unittest.main()
