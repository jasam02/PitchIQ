"""tools/coaching_synth.py: the synthetic coaching scenes of DESIGN 13.1 (and the coach-marked possession scene of the
user's decision of 2026-10-21) are PitchIQ-shaped results that the real consumers load, and they plant what their
truth says.

Why these checks:
- The scenes are the ground truth of the coaching counterexample tests (wave 8): a scene that silently stops planting
  its situation would make every later test meaningless, so its planted relations are re-measured on the true
  positions (truth.json) and bounded here. Measured 2026-10-22: covered spells pass a cover test on 94-100% of their
  advanced own-possession samples, uncovered ones on 0% (the held wing-back spells up to 6%, the formation-change scene's
  first wing-back spell 20% during its ramps); the planted number behind the ball on 78-100% (median 100%); the
  opponent in the channel on 100%; 5.2-6.8 s advanced in own possession per short spell (28-30 s for a held wing-back),
  0.2 s in sprint_back, where the left-back then runs back at up to 5.8-7.2 m/s; crossings at .3 m; the defending
  variants press on 96% of the settled samples (1% for 'no press'), cover 95% of the pressed ones (12% for 'no cover'),
  empty the weak side on 100% and open the gap on 95% of the settled wide-ball samples.
- They must load through what the server runs on a result (analytics.Samples and its reports, TeamShape on the
  server's slim result, possession.compute serialized as the server does, the coaching series) without an error, and
  the consumers must see the scene as designed: possession-4 available with the scripted team on 99.7-100% of the
  frames it names on the broadcast-like scenes, abstaining on the sparse ball, the small-scene footage and the 4v4 field
  written as 105 x 68; team_shape reading the scripted formations; the evidence tier world on the broadcast scenes and
  pinned under the .55 registration.
- The catalogue (tools/annotations/coaching/synthetic_scenes.md) is generated from the registry and must not drift.
- Every scene's settings validate against soccer/coaching/schema.py at every level, and the results use internal ids.

The pipeline checks run on one variant of each kind of footage and camera; set PITCHIQ_SYNTH_ALL=1 to run them on every
variant. Run:  python -m unittest test_coaching_synth          (from this directory)
"""
import contextlib
import hashlib
import io
import json
import os
import re
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

import scene  # noqa: F401  (adds the prototype directory to sys.path)

TOOLS = Path(__file__).resolve().parents[1]/'tools'
sys.path.insert(0, str(TOOLS))
import coaching_synth as cs  # noqa: E402
from soccer.coaching import schema as S  # noqa: E402

ALL_VARIANTS = os.environ.get('PITCHIQ_SYNTH_ALL') == '1'
_BUILT, _TRUTH, _CHECKED = {}, {}, {}


def built(scene_name, variant=None):
    key = (scene_name, variant or cs.SCENES[scene_name][1][0])
    if key not in _BUILT:
        _BUILT[key] = cs.build(*key)
    return _BUILT[key]


def truth_of(scene_name, variant=None):
    """A Built with its truth (the result only when it is already built): the truth tests cover every variant."""
    key = (scene_name, variant or cs.SCENES[scene_name][1][0])
    if key in _BUILT:
        return _BUILT[key]
    if key not in _TRUTH:
        _TRUTH[key] = cs.build_truth(*key)
    return _TRUTH[key]


def result_keys():
    """The variants whose results are checked: the default of each scene, every variant with PITCHIQ_SYNTH_ALL=1."""
    return cs.names() if ALL_VARIANTS else [(s, v[0]) for s, (_, v, _) in cs.SCENES.items()]


def checked(scene_name, variant=None):
    b = built(scene_name, variant)
    if b.key not in _CHECKED:
        _CHECKED[b.key] = cs.check(b)
    return _CHECKED[b.key]


def sha(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('ascii')).hexdigest()


# Scenes whose consumers' readings are checked by default: one per footage kind and camera (all with PITCHIQ_SYNTH_ALL).
PIPELINE = [('covered_advance', 'cb_shift'), ('possession_unknown', 'default'), ('three_visible', 'default'),
            ('formation_change', 'default'), ('short_segment', 'default'), ('kids_swarm', 'default'),
            ('kids_swarm', 'measured_size'), ('amateur_coach_spans', 'spans')]
BROADCAST_FULL = {'covered_advance', 'unsupported_advance', 'single_advance', 'identity_crossing', 'low_calibration',
                  'sprint_back', 'wrong_formation', 'formation_change', 'short_segment', 'professional', 'stacking',
                  'defending', 'small_field_scale'}
SHAPES = {'covered_advance': ['4-4-2'], 'professional': ['4-3-3'], 'wrong_formation': ['3-5-2'],
          'formation_change': ['4-4-2', '3-5-2'], 'three_visible': ['insufficient'], 'small_field_scale': ['lineup']}


class Catalogue(unittest.TestCase):
    def test_every_design_row_and_the_coach_span_scene_has_a_scene(self):
        rows = {row for _, _, row in cs.SCENES.values()}
        for n in range(1, 16):
            self.assertTrue(any(r.startswith(f'DESIGN 13.1 row {n}') for r in rows), n)
        self.assertIn('amateur_coach_spans', cs.SCENES)
        self.assertGreaterEqual(len(cs.names()), 30)

    def test_expectations_use_the_documented_tokens(self):
        for scene_name, variant in cs.names():
            s = cs.spec(scene_name, variant)
            self.assertTrue(s.expected, s.scene)
            for e in s.expected:
                self.assertIn(e['expect'], cs.EXPECT_WORDS, (scene_name, variant))
                self.assertTrue(set(e['levels']) <= set(range(1, 8)) and e['levels'])
                self.assertTrue(e['detectors'] is None or set(e['detectors']) <= set(S.DETECTORS), e)
                self.assertTrue(e['source'])

    def test_the_catalogue_document_is_the_generated_one(self):
        text = cs.CATALOGUE.read_bytes().decode('ascii')
        self.assertNotIn('\n', text.replace('\r\n', ''), 'CRLF line endings')
        text = text.replace('\r\n', '\n')
        self.assertIn(cs.catalogue_markdown(), text, 'run: python tools/coaching_synth.py catalogue --write')

    def test_unknown_scene_or_variant_is_refused(self):
        with self.assertRaises(KeyError):
            cs.spec('no_such_scene')
        with self.assertRaises(KeyError):
            cs.spec('covered_advance', 'no_such_variant')


class Results(unittest.TestCase):
    """Every variant is a well-formed PitchIQ result (the fields the consumers read), deterministic, plausible."""

    def test_every_scene_is_a_pitchiq_shaped_result(self):
        pid_re = re.compile(rf'^(?:{S.PLAYER_ID})$')
        for scene_name, variant in result_keys():
            with self.subTest(scene=scene_name, variant=variant):
                b = built(scene_name, variant)
                r, s = b.result, b.spec
                for key in ('version', 'video', 'config', 'frames', 'match', 'events', 'issues', 'tactical', 'summary',
                            'roster', 'labels', 'status'):
                    self.assertIn(key, r)
                fps = r['video']['fps']
                frames = r['frames']
                self.assertEqual(len(frames), r['video']['frames'])
                idx = np.array([f['frame'] for f in frames])
                self.assertTrue(np.all(np.diff(idx) == 1))
                times = np.array([f['time'] for f in frames])
                self.assertTrue(np.allclose(times, idx/fps, atol=1e-5), 'time = frame / fps')
                self.assertAlmostEqual(times[0], cs.T_START)
                roster = {e['id']: e for key in ('players', 'goalkeepers') for e in r['match'][key]}
                self.assertEqual(len(roster), 2*s.players)
                for pid, e in roster.items():
                    self.assertRegex(pid, pid_re)
                    self.assertTrue(e['confirmed'])
                seen = set()
                for f in frames[::25]:
                    ids = [p['id'] for p in f['people']]
                    self.assertEqual(len(ids), len(set(ids)))
                    seen |= set(ids)
                    for p in f['people']:
                        self.assertIn(p['id'], roster)
                        self.assertTrue(all(-.03 <= v <= 1.03 for v in p['pitch']), p)
                        x, y, w, h = p['box']
                        self.assertTrue(0 <= x and 0 <= y and x+w <= 1+1e-6 and y+h <= 1+1e-6)
                    ball = f['ball']
                    self.assertIn(ball['state'], ('TRACKED', 'UNKNOWN'))
                    if ball['state'] == 'TRACKED':
                        self.assertEqual(ball['origin'], 'OBSERVATION')
                        self.assertEqual(len(ball['pitch']), 2)
                    self.assertEqual(len(f['calibration']['H']), 9)
                    self.assertEqual(len(f['view']), 4)
                    self.assertEqual(f['profile']['smallScene'], s.footage in ('amateur', 'veo'))
                self.assertTrue(seen)
                small = tuple(s.field) != (105.0, 68.0)
                size = r['tactical']['pitchSize']
                if s.measured:
                    self.assertEqual((size['length'], size['width'], size['source']), (s.field[0], s.field[1], 'landmarks'))
                else:
                    self.assertEqual((size['length'], size['width'], size['source']), (105.0, 68.0, 'standard'))
                self.assertEqual(r['roster']['expected']['A'], s.players)
                self.assertEqual(r['config']['outfieldPerTeam'], s.players-(1 if s.goalkeeper else 0))
                if small:
                    self.assertFalse(any(isinstance(f.get('pitch'), dict) for f in frames[:50]), 'no paint on a small field')
                json.dumps(r, allow_nan=False).encode('ascii')
                json.dumps(b.truth, allow_nan=False).encode('ascii')

    def test_speeds_are_plausible(self):
        for scene_name, variant in cs.names():
            with self.subTest(scene=scene_name, variant=variant):
                sp = truth_of(scene_name, variant).truth['speeds']
                self.assertLessEqual(max(sp['playersMaxMS'].values()), cs.LIMIT_PLAYER_SPEED)
                self.assertLessEqual(sp['ballMaxMS'], cs.LIMIT_BALL_SPEED)

    def test_deterministic(self):
        a = cs.build('short_segment')
        b = cs.build('short_segment')
        self.assertEqual(sha(a.result), sha(b.result))
        self.assertEqual(sha(a.truth), sha(b.truth))
        c = cs.build('short_segment', seed=1)
        self.assertNotEqual(sha(a.result), sha(c.result))

    def test_the_truth_follows_the_script(self):
        for scene_name, variant in cs.names():
            with self.subTest(scene=scene_name, variant=variant):
                b = truth_of(scene_name, variant)
                tr = b.truth
                spans = tr['possession']
                self.assertTrue(all(a['t1'] <= c['t0']+1e-6 for a, c in zip(spans, spans[1:])))
                self.assertEqual(len(tr['plants']), len(b.spec.plants))
                for c in tr['changes']:
                    self.assertLess(c['lostAt'], c['wonAt'])
                    self.assertNotEqual(c['from'], c['to'])
                for roles in tr['roles'].values():
                    self.assertTrue(roles)
                    for r in roles:
                        self.assertRegex(r['role'], rf'^(?:{"|".join(S.TEMPLATE_ROLES)}|{S.RELATIVE_ROLE})$')


class Plants(unittest.TestCase):
    """The planted relations hold on the true positions (truth.json), with the measured margins."""

    def plants(self, scene_name, variant=None):
        return list(truth_of(scene_name, variant).truth['plants'])

    def test_covered_advances_are_covered(self):
        for key in (('covered_advance', 'cb_shift'), ('covered_advance', 'channel'), ('professional', 'default')):
            ps = self.plants(*key)
            self.assertEqual(len(ps), 16)
            for p in ps:
                m = p['measured']
                self.assertGreaterEqual(m['advancedOwnSeconds'], 3.0, key)
                self.assertGreaterEqual(m['coverShare']['any'], .9, key)
                self.assertEqual(m['opponentShare'], 0.0)
        cb = self.plants('covered_advance', 'cb_shift')
        self.assertGreaterEqual(np.median([p['measured']['coverShare']['cbShift'] for p in cb]), .9)
        ch = self.plants('covered_advance', 'channel')
        self.assertEqual(max(p['measured']['coverShare']['cbShift'] for p in ch), 0.0)

    def test_unsupported_advances_are_uncovered_exposed_and_behind_as_planted(self):
        for scene_name, variant in cs.names():
            for p in truth_of(scene_name, variant).truth['plants']:
                if p['cover'] != 'none':
                    continue
                with self.subTest(scene=scene_name, variant=variant, t0=p['t0']):
                    m = p['measured']
                    self.assertLessEqual(m['coverShare']['any'] or 0.0, .25)
                    if p['opponent']:
                        self.assertGreaterEqual(m['opponentShare'], .95)
                    if p['behind'] is not None:
                        self.assertGreaterEqual(m['behindShare'], .75)
                    if p['recover']:
                        self.assertLess(m['advancedOwnSeconds'], 3.0)
                        self.assertGreaterEqual(m['maxGoalwardsMS'], 1.5*cs.RECOVER_SPEED)
                        self.assertGreaterEqual(m['recoveringShare'], .3)
                    else:
                        self.assertGreaterEqual(m['advancedOwnSeconds'], 3.0)
        none = self.plants('unsupported_advance', 'none_behind')
        self.assertEqual(len(none), 16)
        self.assertEqual(min(p['measured']['behindShare'] for p in none), 1.0)
        self.assertEqual(max(p['measured']['coverShare']['any'] for p in none), 0.0)
        one = self.plants('unsupported_advance', 'one_behind')
        self.assertGreaterEqual(min(p['measured']['behindShare'] for p in one), .9)
        self.assertEqual(len(self.plants('unsupported_advance', 'spells4')), 4)
        self.assertEqual(len(self.plants('single_advance')), 1)

    def test_distinct_spells_are_ten_seconds_apart(self):
        for key in (('unsupported_advance', 'none_behind'), ('covered_advance', 'cb_shift'), ('short_segment', 'default')):
            starts = sorted(p['t0'] for p in self.plants(*key))
            self.assertGreaterEqual(min(np.diff(starts)), 10.0, key)

    def test_identity_crossings_are_reported_at_each_spell(self):
        b = built('identity_crossing')
        ps = b.truth['plants']
        self.assertEqual(len(b.result['issues']), 2*len(ps))
        self.assertEqual(sum(e['kind'] == 'swap-uncertain' for e in b.result['events']), len(ps))
        for p in ps:
            self.assertLessEqual(p['measured']['crossingMinM'], .5)
            tc = p['crossing']['t']
            low = [pp['identityConfidence'] for f in b.result['frames'] if abs(f['time']-tc) < .5
                   for pp in f['people'] if pp['id'] == p['player']]
            self.assertTrue(low and max(low) < .5)

    def test_formation_change_flags_the_spells_near_it(self):
        ps = self.plants('formation_change')
        near = [p['measured']['advancedNearFormationChangeSeconds'] > 0 for p in ps]
        self.assertEqual(sum(near), 2)
        self.assertEqual({p['role'] for p in ps}, {'LB', 'LWB'})

    def test_the_zoomed_camera_sees_about_three_team_mates(self):
        v = truth_of('three_visible').truth['inView']['A']
        self.assertLessEqual(v['median'], 4)
        self.assertLessEqual(v['max'], 6)
        self.assertEqual(truth_of('covered_advance').truth['inView']['A']['min'], 10)

    def test_defending_variants_plant_their_outcome(self):
        got = {v: truth_of('defending', v).truth['defending']['A'] for v in cs.SCENES['defending'][1]}
        self.assertGreaterEqual(got['press_cover']['pressedShare'], .9)
        self.assertGreaterEqual(got['press_cover']['coveredShareOfPressed'], .9)
        self.assertGreaterEqual(got['press_no_cover']['pressedShare'], .9)
        self.assertLessEqual(got['press_no_cover']['coveredShareOfPressed'], .2)
        self.assertLessEqual(got['no_press']['pressedShare'], .05)
        self.assertGreaterEqual(got['weak_empty']['wideSamples'], 30)
        self.assertGreaterEqual(got['weak_empty']['weakEmptyShare'], .9)
        self.assertLessEqual(got['gap_open']['weakEmptyShare'], .1)
        self.assertGreaterEqual(got['gap_open']['gapOpenShare'], .9)
        self.assertLessEqual(got['weak_empty']['gapOpenShare'], .1)


class Pipeline(unittest.TestCase):
    """The scenes load through analytics, team shape, possession and the coaching series, and those see the scene."""

    def keys(self):
        return [(s, v) for s, v in cs.names()] if ALL_VARIANTS else PIPELINE

    def test_scenes_load_through_the_consumers(self):
        for scene_name, variant in self.keys():
            with self.subTest(scene=scene_name, variant=variant):
                got = checked(scene_name, variant)
                b = built(scene_name, variant)
                outfield = 2*(b.spec.players-(1 if b.spec.goalkeeper else 0))
                self.assertGreaterEqual(got['analytics']['identities'], min(outfield, 12) if b.spec.camera == 'follow' else outfield)
                self.assertEqual(got['series']['smallScene'], b.spec.footage in ('amateur', 'veo'))
                p = got['possession']
                should = (b.spec.ball == 'tracked' and b.spec.footage == 'broadcast'
                          and not (b.spec.scene == 'kids_swarm' and not b.spec.measured))
                self.assertEqual(p['available'], should, p)
                if should and b.spec.scene in BROADCAST_FULL:
                    self.assertGreaterEqual(p['agreement'], .95)
                    self.assertGreaterEqual(p['coverage'], .9)
                    self.assertLessEqual(abs(p['changes']-p['scriptChanges']), 1)
                elif should:
                    self.assertGreaterEqual(p['agreement'], .9)
                    self.assertGreaterEqual(p['coverage'], .6)
                if b.spec.scene in SHAPES and b.spec.variant == cs.SCENES[b.spec.scene][1][0]:
                    self.assertEqual(got['teamShape']['A'], SHAPES[b.spec.scene])

    def test_possession_abstains_for_the_stated_reasons(self):
        self.assertIn('placed', checked('possession_unknown')['possession']['reason'])
        self.assertIn('small', checked('amateur_coach_spans')['possession']['reason'])
        self.assertIn('disagree', checked('kids_swarm')['possession']['reason'])

    def test_evidence_tiers(self):
        from soccer.coaching import evidence, series
        if not hasattr(evidence, 'Frames'):
            self.skipTest('evidence.Frames changed: tiers are checked by the evidence tests')
        tiers = {}
        for key in (('covered_advance', 'cb_shift'), ('low_calibration', 'default')):
            cd = series.build_series(built(*key).result)
            fx = evidence.Frames(cd)
            tiers[key[0]] = {fx.tier(s)[0] for s in range(0, len(cd.t), 50)}
        self.assertEqual(tiers['covered_advance'], {'world'})
        self.assertEqual(tiers['low_calibration'], {'pinned'})

    def test_thirty_frames_a_second(self):
        b = cs.build('short_segment', fps=30.0)
        self.assertEqual(len(b.result['frames']), 1800)
        got = cs.check(b)
        self.assertTrue(got['possession']['available'])
        self.assertGreaterEqual(got['possession']['agreement'], .95)
        self.assertLessEqual(max(b.truth['speeds']['playersMaxMS'].values()), cs.LIMIT_PLAYER_SPEED)

    def test_v2_measurement_tier_when_its_stages_exist(self):
        import soccer.coaching as coaching
        b = built('covered_advance')
        try:
            obs = coaching.build_observations(b.result, 'A', coaching=cs.settings_for(b, level=4))
        except coaching.StageUnavailable as e:
            self.skipTest(f'v2 stages not built yet: {e}')
        self.assertEqual(S.validate('obs', obs), [])


class Settings(unittest.TestCase):
    def test_settings_validate_at_every_level(self):
        for scene_name, variant in cs.names():
            b = truth_of(scene_name, variant)
            for level in range(1, 8):
                with self.subTest(scene=scene_name, variant=variant, level=level):
                    st = cs.settings_for(b, level=level)
                    self.assertEqual(S.validate('settings', st), [])
                    block = st['teams']['A']
                    self.assertEqual(block['format'], {'players': b.spec.players, 'goalkeeper': b.spec.goalkeeper})
                    self.assertEqual(block['field'] is not None, b.spec.declared)

    def test_scene_specific_settings(self):
        st = cs.settings_for(truth_of('wrong_formation', 'coach_roles'))['teams']['A']
        self.assertEqual(st['formation']['intended'], '4-4-2')
        self.assertEqual(st['roles']['A-04'][0]['role'], 'LWB')
        self.assertEqual(cs.settings_for(truth_of('wrong_formation'))['teams']['A']['roles'], {})
        kid = cs.settings_for(truth_of('kids_swarm'), level=1)['teams']['A']
        self.assertEqual((kid['ageGroup'], kid['format']['goalkeeper'], kid['field']),
                         ('U8', False, {'length': 30.0, 'width': 20.0}))
        self.assertIsNone(cs.settings_for(truth_of('small_field_scale', 'assumed'))['teams']['A']['field'])
        st = cs.settings_for(truth_of('covered_advance'), level=6, style='possession')['teams']['A']
        self.assertEqual(st['style'], 'possession')


class CoachSpans(unittest.TestCase):
    def test_coach_spans_follow_the_script_inside_the_range(self):
        for scene_name, variant in cs.names():
            b = truth_of(scene_name, variant)
            with self.subTest(scene=scene_name, variant=variant):
                if not b.spec.coach_spans:
                    self.assertIsNone(b.coach_spans)
                    continue
                cspans = b.coach_spans['spans']
                first, last = b.truth['segment']['t0'], b.truth['segment']['t1']
                self.assertTrue(cspans)
                for sp in cspans:
                    self.assertIn(sp['state'], ('A', 'B', 'dead'))
                    self.assertTrue(first-1e-6 <= sp['t0'] < sp['t1'] <= last+1e-6)
                self.assertTrue(all(a['t1'] <= c['t0']+1e-6 for a, c in zip(cspans, cspans[1:])))
                own = sum(sp['t1']-sp['t0'] for sp in cspans if sp['state'] == 'A')
                script = sum(min(sp['t1'], last)-sp['t0'] for sp in b.truth['possession'] if sp['state'] == 'A')
                self.assertAlmostEqual(own, script, places=2)
        self.assertIsNotNone(truth_of('amateur_coach_spans', 'spans').coach_spans)
        self.assertIsNone(truth_of('amateur_coach_spans', 'no_spans').coach_spans)


class CommandLine(unittest.TestCase):
    def test_write_and_refuse_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(cs.main(['write', 'amateur_coach_spans/spans', '--out', tmp, '--level', '2']), 0)
            names = sorted(os.listdir(tmp))
            self.assertEqual(names, ['coach_spans.json', 'coaching_settings.json', 'result.json', 'truth.json'])
            st = json.loads(Path(tmp, 'coaching_settings.json').read_text(encoding='ascii'))
            self.assertEqual(st['teams']['A']['playingLevel'], 2)
            result = json.loads(Path(tmp, 'result.json').read_text(encoding='ascii'))
            self.assertEqual(sha(result), sha(built('amateur_coach_spans', 'spans').result))
            for bad in (Path(tmp, 'runs', 'x'), Path(tmp, 'uploads')):
                with self.assertRaises(SystemExit):
                    cs.main(['write', 'short_segment', '--out', str(bad)])
                self.assertFalse(bad.exists())


if __name__ == '__main__':
    unittest.main()
