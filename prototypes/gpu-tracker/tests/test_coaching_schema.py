"""The coaching v2 contract (soccer/coaching/schema.py) and the three-tier entry points (soccer/coaching/__init__.py:
build_observations with coaching=, build_roles, interpret).

What is pinned here and why:
- every example record validates and the example report passes cross_check against the example obs and roles files,
  so the examples other roles start their tests and mocks from are themselves inside the contract;
- strict JSON: NaN and Infinity, a bool where a number belongs, a numpy integer, an unknown key, non-ASCII text and a
  display label where an internal id belongs are refused (the v1 page broke on a NaN, invariant 6 and 8);
- the rules that carry DESIGN's guarantees: no work_on at levels 1-2 and no try_this from level 4, no criticism with
  a failed guardrail, no strength without a demonstration, shape detectors only on phase spans of >= 3 s, D3 only in
  the role tier, the annotation element budget and tier table (a suggested zone never on a pinned or image frame),
  seek = (frame + 0.5) / fps, no unmeasured event in a report's text, every cited moment measured (invariant 3);
- the entry points: v1 still runs, the v1 and v2 options are not mixed, a missing stage names its module, the
  measurement stages receive only the format and field size so the obs file and its fingerprint are the same for
  every level, age, style, role and range (invariant 2), a role change re-runs only the role tier, a level change only
  the interpretation tier, and a stage that breaks the contract fails where it ran.

Run:  python -m unittest test_coaching_schema          (from this directory)
"""
import copy
import importlib.util
import json
import os
import subprocess
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np  # noqa: E402

import soccer.coaching as coaching  # noqa: E402
from soccer import team_shape  # noqa: E402
from soccer.coaching import episodes, schema, store  # noqa: E402

S = schema


def problems(kind, value):
    return S.validate(kind, value)


class ExamplesTests(unittest.TestCase):
    def test_every_example_follows_the_contract(self):
        for kind in S.KINDS:
            with self.subTest(kind=kind):
                self.assertEqual(problems(kind, S.example(kind)), [])

    def test_the_example_files_belong_together(self):
        self.assertEqual(S.cross_check(S.example('report'), S.example('obs'), S.example('roles')), [])

    def test_examples_survive_a_json_round_trip(self):
        for kind in S.KINDS:
            value = json.loads(json.dumps(S.example(kind), allow_nan=False))
            self.assertEqual(problems(kind, value), [], kind)

    def test_examples_are_copies(self):
        a = S.example('finding')
        a['verdict'] = 'nonsense'
        self.assertEqual(S.example('finding')['verdict'], 'work_on')

    def test_vocabularies_match_the_code_they_describe(self):
        self.assertEqual(set(S.FORMATIONS_11), set(team_shape.FORMATION_LABELS))
        template_roles = {r for t in team_shape.TEMPLATES.values() for r in t['roles']}
        self.assertTrue(template_roles <= set(S.TEMPLATE_ROLES), template_roles-set(S.TEMPLATE_ROLES))
        self.assertEqual(set(S.DETECTOR_CODE), set(S.DETECTORS))
        self.assertEqual(set(S.SKILLS.values()), set(S.DETECTORS))
        self.assertEqual(set(S.DETECTOR_TIERS), set(S.DETECTORS))
        self.assertEqual(set(S.DETECTOR_PHASES), set(S.DETECTORS))
        self.assertEqual(sorted(sum(S.TIER_FIELDS.values(), ())+S.SERVE_TIME_FIELDS),
                         sorted(S.TeamSettings.__annotations__))
        for kind, spec in S.KINDS.items():
            self.assertTrue(spec is S.Element or hasattr(spec, '__required_keys__'), kind)

    def test_schema_needs_only_the_standard_library(self):
        path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'soccer', 'coaching',
                            'schema.py')
        code = ('import importlib.util, sys\n'
                f'spec = importlib.util.spec_from_file_location("schema_alone", r"{path}")\n'
                'm = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)\n'
                'print(sorted(k for k in ("numpy", "scipy", "cv2", "torch") if k in sys.modules))\n'
                'print(m.validate("finding", m.example("finding")))\n')
        out = subprocess.run([sys.executable, '-I', '-B', '-c', code], capture_output=True, text=True, timeout=60)
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertEqual(out.stdout.split(), ['[]', '[]'])
        self.assertIsNotNone(importlib.util.find_spec('soccer.coaching.schema'))


class StrictJsonTests(unittest.TestCase):
    def test_non_finite_numbers_are_refused_anywhere(self):
        o = S.example('opportunity')
        o['relations']['d1'][3] = float('nan')
        self.assertTrue(any('relations.d1[3]' in p and 'finite' in p for p in problems('opportunity', o)))
        c = S.example('claim')
        c['facts']['deep'] = {'x': [1.0, float('inf')]}
        self.assertTrue(any('facts.deep.x[1]' in p for p in problems('claim', c)))

    def test_types_are_exact(self):
        m = S.example('moment')
        m['frame'] = True
        self.assertTrue(any('moment.frame' in p for p in problems('moment', m)))
        m = S.example('moment')
        m['frame'] = np.int64(m['frame'])
        self.assertTrue(any('moment.frame' in p and 'int64' in p for p in problems('moment', m)))
        m = S.example('moment')
        m['t'] = np.float64(m['t'])                 # a float subclass: json.dumps writes it
        self.assertEqual(problems('moment', m), [])
        m['players'] = ('A-04', 'A-05', 'B-10')    # a tuple serializes as a list
        self.assertEqual(problems('moment', m), [])
        c = S.example('claim')
        c['facts']['when'] = {1, 2}
        self.assertTrue(any('set is not a JSON value' in p for p in problems('claim', c)))

    def test_records_are_closed_and_complete(self):
        f = S.example('finding')
        f['mistakes'] = 3
        self.assertTrue(any("unknown key 'mistakes'" in p for p in problems('finding', f)))
        f = S.example('finding')
        del f['guardrails']['intent']
        self.assertTrue(any("missing 'intent'" in p for p in problems('finding', f)))
        a = S.example('alternative')
        del a['from']
        self.assertTrue(any("missing 'from'" in p for p in problems('alternative', a)))

    def test_text_reaching_the_pdf_is_printable_ascii(self):
        e = S.example('evidence')
        e['annotation']['caption'] = 'Left back advanced \u2013 nobody covered.'
        self.assertTrue(any('printable ASCII' in p for p in problems('evidence', e)))
        e = S.example('evidence')
        e['annotation']['caption'] = 'x'*(S.CAPTION_MAX+1)
        self.assertTrue(any('annotation.caption' in p for p in problems('evidence', e)))
        f = S.example('finding')
        f['text']['headline'] = 'Line one\nline two'
        self.assertTrue(problems('finding', f))

    def test_stored_records_use_internal_ids(self):
        for bad in ('GK-A', 'B-?', 'Team A player 3', 'a-03'):
            f = S.example('finding')
            f['players']['primary'] = [bad]
            self.assertTrue(problems('finding', f), bad)
        for good in ('A-03', 'B-114', 'GK-1'):
            f = S.example('finding')
            f['players']['opponents'] = [good]
            self.assertEqual(problems('finding', f), [], good)

    def test_problem_lists_are_capped(self):
        o = S.example('opportunity')
        o['relations']['d1'] = ['x']*0+[float('nan')]*500
        out = problems('opportunity', o)
        self.assertLessEqual(len(out), S.MAX_PROBLEMS+1)
        with self.assertRaises(S.SchemaError) as ctx:
            S.check('opportunity', o)
        self.assertEqual(ctx.exception.kind, 'opportunity')
        self.assertIn('does not follow the coaching schema', str(ctx.exception))


class SettingsAndProfileRulesTests(unittest.TestCase):
    def team(self):
        return S.example('team_settings')

    def test_formats_and_formations(self):
        t = self.team()
        t['format'] = {'players': 7, 'goalkeeper': False}
        t['formation']['intended'] = None
        self.assertTrue(any('always has a goalkeeper' in p for p in problems('team_settings', t)))
        t = self.team()
        t['formation']['intended'] = '1-3-2-1'           # a 7v7 shape on an 11v11 team
        self.assertTrue(any('not offered for 11v11' in p for p in problems('team_settings', t)))
        t['format'] = {'players': 7, 'goalkeeper': True}
        self.assertEqual(problems('team_settings', t), [])
        t['format'] = {'players': 4, 'goalkeeper': False}
        t['formation']['intended'] = '1-2-1'
        self.assertEqual(problems('team_settings', t), [])
        t['format'] = {'players': 6, 'goalkeeper': True}
        self.assertTrue(problems('team_settings', t))

    def test_field_and_role_spans(self):
        t = self.team()
        t['field'] = {'length': 35.0, 'width': 50.0}
        self.assertTrue(any('width larger than length' in p for p in problems('team_settings', t)))
        t['field'] = {'length': 50.0, 'width': 35.0}
        self.assertEqual(problems('team_settings', t), [])
        t['roles']['A-03'] = [{'role': 'LB', 'from': 0.0, 'to': 100.0, 'set': '2026-10-21T15:00:00'},
                              {'role': 'back-1/3', 'from': 50.0, 'to': None, 'set': '2026-10-21T15:00:00'}]
        self.assertTrue(any('overlap' in p for p in problems('team_settings', t)))
        t['roles']['A-03'][1]['from'] = 100.0
        self.assertEqual(problems('team_settings', t), [])
        t['roles']['A-03'][1]['role'] = 'left back'
        self.assertTrue(problems('team_settings', t))

    def test_profile_verdicts_follow_the_level(self):
        p = S.example('profile')
        p['level'] = 2
        p['levelName'] = S.LEVEL_NAMES[2]
        self.assertTrue(any('try_this, never work_on' in x for x in problems('profile', p)))
        p = S.example('profile')
        p['detectors']['behind_ball']['verdicts'] = ['try_this']
        self.assertTrue(any('level 1-3 framing' in x for x in problems('profile', p)))
        p = S.example('profile')
        p['detectors']['behind_ball'].update(enabled=False)
        self.assertTrue(any('enabled must equal' in x for x in problems('profile', p)))
        p['detectors']['behind_ball'].update(verdicts=[])
        self.assertEqual(problems('profile', p), [])
        del p['detectors']['attacking_shape']
        self.assertTrue(any('missing attacking_shape' in x for x in problems('profile', p)))

    def test_profile_thresholds_are_pitchiq_proposals(self):
        p = S.example('profile')
        p['evidence']['validated'] = True
        self.assertTrue(problems('profile', p))
        p = S.example('profile')
        p['detectors']['pressure_cover']['origin'] = 'FEDERATION'
        self.assertTrue(problems('profile', p))
        p = S.example('profile')
        p['evidence']['minFailureShare'] = None            # style-relative is a level 7 rule
        self.assertTrue(any('only level 7' in x for x in problems('profile', p)))

    def test_profile_safety_and_guidance(self):
        p = S.example('profile')
        p['ageGroup'] = 'U10'
        self.assertTrue(any('no heading at U10' in x for x in problems('profile', p)))
        p = S.example('profile')
        p['objectives'][0]['text'] = 'The guide says "'+'a'*61+'".'
        self.assertTrue(any('quoted phrase of 61' in x for x in problems('profile', p)))
        p = S.example('profile')
        p['principles'][0]['sources'] = []
        self.assertTrue(any('FEDERATION item needs' in x for x in problems('profile', p)))
        p['principles'][0]['tag'] = 'PITCHIQ-PROPOSAL'
        self.assertEqual(problems('profile', p), [])


class SituationAndOpportunityRulesTests(unittest.TestCase):
    def test_timeline_follows_the_possession_rules(self):
        t = S.example('timeline')
        t['phaseQuality'].update(available=False, level='unavailable', roughShares=None, coverage=None)
        self.assertTrue(any('own/opp phase time is named' in p for p in problems('timeline', t)))
        t['phaseSeconds'].update(own=0.0, opp=0.0)
        t['phaseSpans'] = []
        self.assertEqual(problems('timeline', t), [])
        t = S.example('timeline')
        t['phaseQuality']['roughShares'] = {'A': .47, 'B': .53}
        self.assertTrue(any('rounded to 5%' in p for p in problems('timeline', t)))
        t = S.example('timeline')
        t['format'] = '7v7'
        self.assertTrue(any('assumed scale on a small format' in p for p in problems('timeline', t)))
        t['scale']['metresUsable'] = False
        self.assertEqual(problems('timeline', t), [])

    def test_opportunity_keys_tiers_and_phases(self):
        o = S.example('opportunity')
        o['key'] = 'D1|A|1.0|2.0'
        self.assertTrue(any('schema.opportunity_key' in p for p in problems('opportunity', o)))
        o = S.example('opportunity')
        o['detector'] = 'wide_defender_advance'
        o['key'] = S.opportunity_key('wide_defender_advance', 'A', o['t0'], o['t1'])
        self.assertTrue(any('role tier' in p for p in problems('opportunity', o)))
        o = S.example('opportunity')
        o['phase']['state'] = 'own'
        self.assertTrue(any('assessed only in phase opp' in p for p in problems('opportunity', o)))
        o['abstain'] = 'phase_unknown'                     # an abstaining opportunity may carry any phase
        self.assertEqual(problems('opportunity', o), [])

    def test_shape_detectors_need_three_second_spans(self):
        o = S.example('opportunity')
        o.update(detector='behind_ball', key=S.opportunity_key('behind_ball', 'A', o['t0'], o['t1']))
        o['phase'].update(state='own', spanSeconds=2.0)
        o['actors']['carrier'] = None
        self.assertTrue(any('>= 3 s' in p for p in problems('opportunity', o)))
        o['phase']['spanSeconds'] = 3.0
        self.assertEqual(problems('opportunity', o), [])

    def test_pressure_needs_a_carrier_and_d3_excludes_change_windows(self):
        o = S.example('opportunity')
        o['actors']['carrier'] = None
        self.assertTrue(any('carrier samples' in p for p in problems('opportunity', o)))
        o = S.example('opportunity')
        o.update(detector='wide_defender_advance', tier='role',
                 key=S.opportunity_key('wide_defender_advance', 'A', o['t0'], o['t1'], 'LB-side'), variant='LB-side')
        o['phase'].update(state='own', nearChange=True)
        self.assertTrue(any('exclude the change windows' in p for p in problems('opportunity', o)))
        o['phase']['nearChange'] = False
        self.assertEqual(problems('opportunity', o), [])

    def test_relations_are_per_sample(self):
        o = S.example('opportunity')
        o['relations']['coverDist'] = o['relations']['coverDist'][:-1]
        self.assertTrue(any('values for 19 samples' in p for p in problems('opportunity', o)))
        o = S.example('opportunity')
        o['relations']['t'] = list(reversed(o['relations']['t']))
        self.assertTrue(any('must increase' in p for p in problems('opportunity', o)))
        o = S.example('opportunity')
        del o['relations']['t']
        self.assertTrue(any('needs the sample times' in p for p in problems('opportunity', o)))
        o = S.example('opportunity')
        o['roles']['B-99'] = {'role': None, 'source': 'none', 'confidence': None}
        self.assertTrue(any('not one of the actors' in p for p in problems('opportunity', o)))

    def test_moments_seek_the_middle_of_their_frame(self):
        m = S.example('moment')
        self.assertEqual(m['seek'], round((m['frame']+.5)/m['fps'], 4))
        m['seek'] = m['t']
        self.assertTrue(any('(frame + 0.5) / fps' in p for p in problems('moment', m)))
        m = S.example('moment')
        m['scene']['frame'] += 1
        self.assertTrue(any("moment's frame" in p for p in problems('moment', m)))
        m = S.example('moment')
        m['players'].append('A-09')
        self.assertTrue(any('not observed in the scene' in p for p in problems('moment', m)))

    def test_obs_and_role_files_are_consistent(self):
        o = S.example('obs')
        o['engine']['measurement'] = '0'*16
        self.assertTrue(any("fingerprint's first 16 hex" in p for p in problems('obs', o)))
        o = S.example('obs')
        o['moments'][0]['opportunity'] = 'opp-'+'0'*12
        self.assertTrue(any('not in this file' in p for p in problems('obs', o)))
        o = S.example('obs')
        o['opportunities'].append(copy.deepcopy(o['opportunities'][0]))
        self.assertTrue(any('repeated ids' in p for p in problems('obs', o)))
        r = S.example('roles')
        r['rolebook'].update(format='7v7')
        self.assertTrue(any('never uses the 11v11' in p for p in problems('roles', r)))
        r['rolebook']['model'] = 'relative'
        self.assertTrue(any('11v11 templates only' in p for p in problems('roles', r)))
        r['rolebook']['records'][0].update(source='relative', role='back-1/3')
        self.assertEqual(problems('roles', r), [])


class FindingAndAlternativeRulesTests(unittest.TestCase):
    def test_level_framing(self):
        f = S.example('finding')
        f['level'] = 2
        self.assertTrue(any('level 2 frames work items as try_this' in p for p in problems('finding', f)))
        f['verdict'] = 'try_this'
        self.assertEqual(problems('finding', f), [])
        f['level'] = 4
        self.assertTrue(any('level 1-3 framing' in p for p in problems('finding', f)))
        f['level'] = 3
        self.assertEqual(problems('finding', f), [])

    def test_a_failed_guardrail_makes_a_review(self):
        f = S.example('finding')
        f['guardrails']['identity'] = 'fail'
        self.assertTrue(any('at most a review' in p for p in problems('finding', f)))
        f['verdict'] = 'review'
        self.assertEqual(problems('finding', f), [])

    def test_strengths_are_demonstrations(self):
        f = S.example('finding')
        f['verdict'] = 'strength'
        f['aggregate'].update(demonstrated=0, distinctDemonstratedSpells=0, unclear=4)
        self.assertTrue(any('never an absence of failures' in p for p in problems('finding', f)))

    def test_aggregates_add_up_to_the_listed_opportunities(self):
        f = S.example('finding')
        f['aggregate']['unclear'] = 2
        self.assertTrue(any('!= opportunities' in p for p in problems('finding', f)))
        f = S.example('finding')
        f['opportunities'] = f['opportunities'][:-1]
        self.assertTrue(any('the aggregate counts 9' in p for p in problems('finding', f)))
        f = S.example('finding')
        f['aggregate']['distinctFailingSpells'] = 6
        self.assertTrue(any('more distinct spells' in p for p in problems('finding', f)))
        f = S.example('finding')
        f['timestamps'][0]['opportunity'] = 'opp-'+'0'*12
        self.assertTrue(any("not one of the finding's opportunities" in p for p in problems('finding', f)))
        f = S.example('finding')
        f['key'] = 'D2|A|own_third|760.0|875.6'
        self.assertTrue(any('detector code and the team' in p for p in problems('finding', f)))
        f = S.example('finding')
        f['timestamps'] = []
        self.assertTrue(any('needs its evidence moments' in p for p in problems('finding', f)))

    def test_federation_claims_name_their_source(self):
        f = S.example('finding')
        f['claims'][-1]['source'] = None
        self.assertTrue(any('FEDERATION claim' in p for p in problems('finding', f)))

    def test_alternatives_are_validated_zones(self):
        a = S.example('alternative')
        a['checks']['noStacking'] = False
        self.assertTrue(any('noStacking failed' in p for p in problems('alternative', a)))
        a = S.example('alternative')
        a['improves']['after'] = False
        self.assertTrue(any('from false to true' in p for p in problems('alternative', a)))
        a = S.example('alternative')
        a['reach']['radiusM'] = 20.0
        self.assertTrue(any('horizonS x speedMS' in p for p in problems('alternative', a)))
        a = S.example('alternative')
        a['zone']['kind'] = 'point'
        self.assertTrue(any('small dashed circle' in p for p in problems('alternative', a)))
        a = S.example('alternative')
        a['rejected'].append({'mover': a['mover'], 'reason': 'mover_unseen'})
        self.assertTrue(any('also listed as rejected' in p for p in problems('alternative', a)))
        a = S.example('alternative')
        a['zone']['polygon'][0] = [1.5, .2]
        self.assertTrue(any('outside the normalized pitch' in p for p in problems('alternative', a)))
        n = S.example('no_alternative')
        n['reason'] = 'too_hard'
        self.assertTrue(problems('no_alternative', n))


class AnnotationRulesTests(unittest.TestCase):
    def test_tier_table(self):
        e = S.example('evidence')
        e['tier'] = e['videoOverlay']['tier'] = 'pinned'
        out = problems('evidence', e)
        self.assertTrue(any('suggestedZone is not drawn at tier pinned' in p for p in out), out)
        e['videoOverlay']['elements'] = [el for el in e['videoOverlay']['elements'] if el['type'] != 'suggestedZone']
        e['alternative'] = e['alternative']        # the diagram keeps its zone: the fallback of DESIGN 5.4
        self.assertEqual(problems('evidence', e), [])
        e['tier'] = e['videoOverlay']['tier'] = 'image'
        self.assertTrue(any('relation is not drawn at tier image' in p for p in problems('evidence', e)))
        self.assertEqual(set(S.TIER_ELEMENTS['suggestedZone']), {'world', 'diagram'})

    def test_element_budget(self):
        d = S.example('diagram')
        d['elements'].append(copy.deepcopy(next(el for el in d['elements'] if el['type'] == 'suggestedZone')))
        self.assertTrue(any('2 suggestedZone elements' in p for p in problems('diagram', d)))
        d = S.example('diagram')
        sup = next(el for el in d['elements'] if el['type'] == 'support')
        d['elements'] += [dict(sup, id=f'A-1{k}') for k in range(3)]
        self.assertTrue(any('4 support elements' in p for p in problems('diagram', d)))

    def test_v2_never_emits_the_dormant_circle(self):
        d = S.example('diagram')
        d['elements'].append({'type': 'suggested', 'id': 'A-05', 'x': .3, 'y': .4, 'label': 'COACHING ILLUSTRATION'})
        self.assertTrue(any('render-only for old reports' in p for p in problems('diagram', d)))
        self.assertEqual(problems('element', d['elements'][-1]), [])     # still a known element (old reports)

    def test_diagram_coordinates_are_normalized(self):
        d = S.example('diagram')
        d['elements'][1]['x'] = 1.4
        self.assertTrue(any('outside the normalized pitch' in p for p in problems('diagram', d)))
        d = S.example('diagram')
        d['elements'].append({'type': 'shadow', 'x': .1})
        self.assertTrue(any("unknown element type 'shadow'" in p for p in problems('diagram', d)))

    def test_annotation_window_and_links(self):
        e = S.example('evidence')
        e['annotation']['window'] = {'t0': e['time']+.5, 't1': e['time']+2.0}
        self.assertTrue(any('does not contain the key time' in p for p in problems('evidence', e)))
        e = S.example('evidence')
        e['annotation']['keyFrame'] += 3
        self.assertTrue(any('not the evidence frame' in p for p in problems('evidence', e)))
        e = S.example('evidence')
        e['seek'] = e['time']
        self.assertTrue(any('(frame + 0.5) / fps' in p for p in problems('evidence', e)))
        e = S.example('evidence')
        e['videoOverlay']['tier'] = 'pinned'
        e['videoOverlay']['elements'] = [el for el in e['videoOverlay']['elements'] if el['type'] != 'suggestedZone']
        self.assertTrue(any('not the evidence tier' in p for p in problems('evidence', e)))
        e = S.example('evidence')
        e['alternative'] = 'alt-'+'0'*12
        self.assertTrue(any('suggestedZone of' in p for p in problems('evidence', e)))


class DrillAndProgressRulesTests(unittest.TestCase):
    def test_drill_rules(self):
        d = S.example('drill')
        d['safety']['heading'] = 'limited'
        d['age']['min'] = 'U10'
        self.assertTrue(any('no heading for ages up to U11' in p for p in problems('drill', d)))
        d = S.example('drill')
        d['success_signals_next_match'][0]['detector'] = 'behind_ball'
        self.assertTrue(any('not a related detector' in p for p in problems('drill', d)))
        d = S.example('drill')
        d['age'] = {'min': 'U16', 'max': 'U12'}
        self.assertTrue(any('min after max' in p for p in problems('drill', d)))
        d = S.example('drill')
        d['duration']['minutes'] = 50
        self.assertTrue(problems('drill', d))
        d = S.example('drill')
        d['development_level'] = [5, 4]
        self.assertTrue(any('increasing order' in p for p in problems('drill', d)))
        d = S.example('drill')
        d['field_setup']['length'] = 30.0
        d['field_setup']['width'] = {'value': 20.0, 'unit': 'm', 'scaleWith': None}
        self.assertEqual(problems('drill', d), [])
        d['field_setup']['width'] = {'value': 20.0, 'unit': 'yd', 'scaleWith': None}
        self.assertTrue(problems('drill', d))
        lib = S.example('drill_library')
        lib['drills'].append(copy.deepcopy(lib['drills'][0]))
        self.assertTrue(any('repeated drill_id' in p for p in problems('drill_library', lib)))

    def test_progress_rules(self):
        p = S.example('progress_entry')
        p['skills']['balance'] = p['skills']['pressure_and_cover']
        self.assertTrue(problems('progress_entry', p))
        p = S.example('progress_entry')
        p['range'] = {'t0': 10.0, 't1': 5.0}
        self.assertTrue(any('ends before it starts' in x for x in problems('progress_entry', p)))


class ReportRulesTests(unittest.TestCase):
    def test_sections_hold_their_verdicts(self):
        r = S.example('report')
        r['strengths'] = r['needsWork']
        r['needsWork'] = []
        self.assertTrue(any('does not belong in strengths' in p for p in problems('report', r)))
        r = S.example('report')
        r['profile']['evidence']['maxWorkItems'] = 0
        self.assertTrue(any('the level allows 0' in p for p in problems('report', r)))
        r = S.example('report')
        r['needsWork'][0]['text'] = None
        self.assertTrue(any('a reported finding has its text' in p for p in problems('report', r)))

    def test_no_unmeasured_events_in_the_words(self):
        r = S.example('report')
        r['needsWork'][0]['text']['explanation'] = 'Team A lost the ball twice and the counter-attack followed.'
        out = problems('report', r)
        self.assertTrue(any('possession change' in p for p in out), out)
        self.assertTrue(any('counter-attack' in p for p in out), out)
        r['needsWork'][0]['text']['explanation'] = ('If the ball had been lost, the space behind could have been '
                                                    'exposed to a counter-attack.')
        self.assertEqual(problems('report', r), [])
        r['keyMoments'][0]['caption'] = 'Poor communication on the left.'
        self.assertTrue(any('communication' in p for p in problems('report', r)))

    def test_event_patterns(self):
        self.assertEqual(S.event_problems('Because you chose a counter-attacking style, a deep block is expected.'), [])
        self.assertTrue(S.event_problems('Three turnovers in our half.'))
        self.assertTrue(S.event_problems('After winning the ball, A-04 moved forward.'))
        self.assertTrue(S.event_problems('A-05 passed to A-08.'))
        self.assertEqual(S.event_problems('If A-05 had passed, the lane would have been open.'), [])
        self.assertTrue(S.event_problems('Even if it is hard, they must communicate.'))   # never allowed
        self.assertEqual(S.event_problems(None), [])

    def test_findings_cite_key_moments(self):
        r = S.example('report')
        r['keyMoments'] = r['keyMoments'][1:]
        self.assertTrue(any('which is not a key moment' in p for p in problems('report', r)))
        r = S.example('report')
        r['keyMoments'].reverse()
        self.assertTrue(any('chronological' in p for p in problems('report', r)))
        r = S.example('report')
        r['needsWork'][0]['timestamps'][0]['frame'] += 1
        self.assertTrue(any('differs from the key moment' in p for p in problems('report', r)))

    def test_drills_need_a_finding_or_are_optional(self):
        r = S.example('report')
        r['drills'][0]['finding'] = None
        self.assertTrue(any('no random exercises' in p for p in problems('report', r)))
        r['drills'][0]['optional'] = True
        self.assertEqual(problems('report', r), [])

    def test_engine_stamp_matches_the_fingerprints(self):
        r = S.example('report')
        r['engine']['interpretation'] = '0'*16
        self.assertTrue(any('three fingerprints' in p for p in problems('report', r)))

    def test_cross_check_is_invariant_three(self):
        obs, roles = S.example('obs'), S.example('roles')
        r = S.example('report')
        r['keyMoments'][1]['id'] = r['needsWork'][0]['timestamps'][1]['evidence'] = 'ev-'+'1'*12
        out = S.cross_check(r, obs, roles)
        self.assertTrue(any('is not a measured moment' in p for p in out), out)
        r = S.example('report')
        obs2 = S.example('obs')
        obs2['moments'][0]['frame'] += 1
        self.assertTrue(any('differs from the measured moment' in p for p in S.cross_check(r, obs2, roles)))
        r = S.example('report')
        r['needsWork'][0]['opportunities'][-1] = 'opp-'+'2'*12
        self.assertTrue(any('was not measured' in p for p in S.cross_check(r, obs, roles)))
        r = S.example('report')
        r['fingerprints']['roles'] = '3'*64
        self.assertTrue(any('role file' in p for p in S.cross_check(r, obs, roles)))
        roles2 = S.example('roles')
        roles2['measurement'] = '4'*64
        self.assertTrue(any('another observation file' in p for p in S.cross_check(S.example('report'), obs, roles2)))


class HelperTests(unittest.TestCase):
    def test_ids_and_keys_leave_the_engine_version_out(self):
        a = S.record_id('opp', 'D1|A|812.4|816.0', 812.41)
        self.assertRegex(a, r'^opp-[0-9a-f]{12}$')
        self.assertEqual(a, S.record_id('opp', 'D1|A|812.4|816.0', 812.44))
        with mock.patch.object(episodes, 'ENGINE_VERSION', '99'):
            self.assertEqual(a, S.record_id('opp', 'D1|A|812.4|816.0', 812.41))
        self.assertEqual(S.opportunity_key('pressure_cover', 'A', 812.43, 816.0), 'D1|A|812.4|816.0')
        self.assertEqual(S.opportunity_key('wide_defender_advance', 'B', 1, 2, 'LB-side'), 'D3|B|LB-side|1.0|2.0')
        self.assertEqual(S.finding_key('wide_defender_advance', 'A', 'LB-side', 812.4, 840.0), 'D3|A|LB-side|812.4|840.0')
        self.assertEqual(S.seek_of(733, 30.0), 24.45)
        self.assertEqual((S.format_name({'players': 7, 'goalkeeper': True}), S.outfield({'players': 7, 'goalkeeper': True}),
                          S.outfield({'players': 4, 'goalkeeper': False})), ('7v7', 6, 4))

    def test_settings_split_by_tier(self):
        team = S.example('team_settings')
        tiers = S.tier_settings(team)
        self.assertEqual(set(tiers['measurement']), {'format', 'field'})
        self.assertEqual(set(tiers['role']), {'roles', 'formation'})
        self.assertEqual(set(tiers['interpretation']), {'ageGroup', 'playingLevel', 'style', 'federation'})
        self.assertEqual(S.tier_settings(None)['measurement'], {'format': {'players': 11, 'goalkeeper': True},
                                                                'field': None})
        tiers['measurement']['format']['players'] = 7               # a copy: the settings are untouched
        self.assertEqual(team['format']['players'], 11)

    def test_stale_tiers_cascade(self):
        a = S.example('team_settings')
        cases = [('format', {'players': 9, 'goalkeeper': True}, ['measurement', 'role', 'interpretation']),
                 ('field', {'length': 70.0, 'width': 50.0}, ['measurement', 'role', 'interpretation']),
                 ('roles', {}, ['role', 'interpretation']),
                 ('formation', {'intended': '4-4-2', 'periods': []}, ['role', 'interpretation']),
                 ('playingLevel', 6, ['interpretation']), ('ageGroup', 'Adult', ['interpretation']),
                 ('style', 'compact', ['interpretation']), ('federation', 'England', ['interpretation']),
                 ('squad', {'id': 'sq-'+'a'*12, 'name': 'Renamed'}, []), ('runLinks', [], [])]
        for field, value, want in cases:
            b = copy.deepcopy(a)
            b[field] = value
            self.assertEqual(S.stale_tiers(a, b), want, field)

    def test_team_block(self):
        settings = S.example('settings')
        self.assertEqual(S.team_block(settings, 'A')['ageGroup'], 'U14')
        self.assertIsNone(S.team_block(settings, 'B'))
        self.assertIsNone(S.team_block({}, 'A'))
        self.assertIsNone(S.team_block(None, 'A'))
        with self.assertRaises(TypeError):
            S.team_block([], 'A')


# ---------- the entry points with stand-in stages ----------
RESULT = {'video': {'name': 'x.mp4', 'fps': 30.0}, 'frames': [{'time': 720.0}, {'time': 899.8}]}


class FakeTimeline:
    def __init__(self, team, measurement):
        self.team = team
        self.measurement = measurement

    def summary(self):
        t = S.example('timeline')
        t['team'] = self.team
        t['format'] = S.format_name(self.measurement['format'])
        if t['format'] != '11v11':
            t['scale']['metresUsable'] = False
        return t


class Fakes:
    """Stand-ins for every stage, built from the schema examples; they record what they were given."""

    def __init__(self):
        self.calls = []
        self.measurement_settings = []

    def stages(self, **override):
        names = ['series', 'possession', 'situations', 'detect', 'moments', 'rolebook', 'detect_roles',
                 'role_moments', 'profile', 'assess', 'alternatives', 'annotate', 'explain', 'drills', 'report']
        out = {n: getattr(self, n) for n in names}
        out.update(override)
        return out

    def _log(self, name):
        self.calls.append(name)

    def series(self, result, samples=None, shape=None, settings=None):
        self._log('series')
        return object()

    def possession(self, result):
        self._log('possession')
        return {'available': True}

    def situations(self, cd, poss, team, measurement):
        self._log('situations')
        self.measurement_settings.append(measurement)
        return FakeTimeline(team, measurement)

    def detect(self, timeline):
        self._log('detect')
        obs = S.example('obs')
        if timeline.team != 'A':
            return {'opportunities': [], 'detectors': {}, 'roleInputs': {}}
        return {'opportunities': obs['opportunities'], 'detectors': obs['detectors'], 'roleInputs': obs['roleInputs']}

    def moments(self, cd, timeline, opportunities):
        self._log('moments')
        ids = {o['id'] for o in opportunities}
        obs = S.example('obs')
        return {'moments': [m for m in obs['moments'] if m['opportunity'] in ids], 'frameTiers': obs['frameTiers']}

    def rolebook(self, obs, role):
        self._log('rolebook')
        self.role_settings = role
        book = S.example('rolebook')
        book['team'] = obs['team']
        return book

    def detect_roles(self, obs, book):
        self._log('detect_roles')
        return {'opportunities': [], 'detectors': {}}

    def role_moments(self, obs, opportunities):
        self._log('role_moments')
        return {'moments': []}

    def profile(self, block, team):
        self._log('profile')
        p = S.example('profile')
        level = (block or {}).get('playingLevel') or 4
        p.update(team=team, level=level, levelName=S.LEVEL_NAMES[level], ageGroup=(block or {}).get('ageGroup') or 'U14')
        for d in p['detectors'].values():
            if level <= 2:
                d['verdicts'] = ['strength', 'try_this', 'review']
        if level <= 2:
            p['evidence']['maxWorkItems'] = 1
        return p

    def assess(self, obs, roles, profile, t0, t1):
        self._log('assess')
        self.range = (t0, t1)
        f = S.example('finding')
        f['guardrails']['alternative'] = 'not_applicable'
        f['level'] = profile['level']
        f['verdict'] = 'try_this' if profile['level'] <= 2 else 'work_on'
        f['suggestedAction'] = None
        return {'findings': [f], 'detectors': [], 'priorities': [f['id']], 'nextObjective': None}

    def alternatives(self, finding, obs, roles, profile):
        self._log('alternatives')
        return S.example('alternative')

    def annotate(self, findings, obs, roles, profile, alternatives):
        self._log('annotate')
        return S.example('report')['keyMoments']

    def explain(self, findings, profile, obs, roles, evidence, alternatives):
        self._log('explain')
        text = S.example('finding')['text']
        for f in findings:
            f['text'] = copy.deepcopy(text)
        return {'findings': findings, 'withheld': []}

    def drills(self, findings, profile, priorities, next_objective=None):   # (assess's NextObjective, fix round 2)
        self._log('drills')
        return S.example('report')['drills']

    def report(self, obs, roles, profile, assessment, findings, evidence, alternatives, no_alternatives, drills,
               withheld, progress, meta, range):
        self._log('report')
        r = S.example('report')
        r.update(team=obs['team'], profile=profile, keyMoments=evidence, alternatives=alternatives,
                 noAlternatives=no_alternatives, drills=drills, withheld=withheld, progression=progress, range=range,
                 run=meta.get('run'), strengths=[f for f in findings if f['verdict'] == 'strength'],
                 needsWork=[f for f in findings if f['verdict'] in S.CRITICISM],
                 reviews=[f for f in findings if f['verdict'] == 'review'],
                 described=[f for f in findings if f['verdict'] == 'describe'])
        r['engine'] = r['fingerprints'] = None     # interpret stamps them
        return r


def settings_with(**team):
    base = S.example('settings')
    base['teams']['A'].update(team)
    return base


class EntryPointTests(unittest.TestCase):
    def setUp(self):
        self.fakes = Fakes()
        self.rsha = 'f'*64

    def observe(self, coaching_settings, fakes=None, **kw):
        fakes = fakes or self.fakes
        return coaching.build_observations(RESULT, team='A', coaching=coaching_settings, result_sha=self.rsha,
                                           stages=fakes.stages(), **kw)

    def full(self, coaching_settings, t0=None, t1=None):
        obs = self.observe(coaching_settings)
        roles = coaching.build_roles(obs, coaching_settings, stages=self.fakes.stages())
        report = coaching.interpret(obs, roles, coaching_settings, t0=t0, t1=t1, stages=self.fakes.stages())
        return obs, roles, report

    def test_v1_and_v2_options_are_not_mixed(self):
        with self.assertRaises(TypeError):
            coaching.build_observations(RESULT, possession={})
        with self.assertRaises(TypeError):
            coaching.build_observations(RESULT, stages={})
        for kw in ({'t0': 1.0}, {'t1': 2.0}, {'reference': False}, {'library': {}}, {'keep_series': True}):
            with self.assertRaises(TypeError, msg=kw):
                coaching.build_observations(RESULT, coaching={}, **kw)

    def test_a_missing_stage_names_its_module(self):
        with mock.patch.dict(coaching.STAGES, {'situations': 'soccer.coaching.no_such_module:build_timeline'}):
            stages = self.fakes.stages()
            del stages['situations']
            with self.assertRaises(coaching.StageUnavailable) as ctx:
                coaching.build_observations(RESULT, team='A', coaching={}, result_sha=self.rsha, stages=stages)
        self.assertIn('soccer.coaching.no_such_module', str(ctx.exception))
        self.assertIn('CONTEXT', str(ctx.exception))
        with mock.patch.dict(coaching.STAGES, {'drills': 'soccer.coaching.drills:no_such_function'}):
            with self.assertRaises(coaching.StageUnavailable):
                coaching._stage('drills')
        self.assertTrue(issubclass(coaching.StageUnavailable, NotImplementedError))

    def test_measurement_stages_see_only_format_and_field(self):
        obs = self.observe(settings_with(playingLevel=6, ageGroup='Adult', style='compact'))
        self.assertEqual(S.validate('obs', obs), [])
        self.assertEqual(self.fakes.measurement_settings, [{'format': {'players': 11, 'goalkeeper': True},
                                                            'field': None}])
        self.assertEqual(obs['inputs']['result'], self.rsha)
        self.assertEqual(obs['inputs']['shapeSettings'], store.settings_sha(team_shape.clean_settings({})))
        self.assertEqual(obs['segment'], {'t0': 720.0, 't1': 899.8, 'fps': 30.0, 'frames': 2})

    def test_observations_do_not_change_with_the_profile(self):
        """Invariant 2: the obs file and its fingerprint are identical for every interpretation and role setting."""
        base = canonical(self.observe(settings_with()))
        variants = [dict(playingLevel=level) for level in range(1, 8)] + \
                   [dict(ageGroup=age) for age in ('U8', 'U12', 'U16', 'Adult')] + \
                   [dict(style=style) for style in S.STYLES] + [dict(federation='Canada')] + \
                   [dict(roles={}), dict(formation={'intended': '4-4-2', 'periods': []})] + \
                   [dict(squad=None, runLinks=[])]
        for v in variants:
            self.assertEqual(canonical(self.observe(settings_with(**v))), base, v)
        self.assertEqual(canonical(self.observe(None)), base)   # defaults: 11v11, no declared field
        changed = self.observe(settings_with(field={'length': 100.0, 'width': 64.0}))
        self.assertNotEqual(changed['fingerprint'], json.loads(base)['fingerprint'])
        seven = self.observe(settings_with(format={'players': 7, 'goalkeeper': True},
                                           formation={'intended': None, 'periods': []}))
        self.assertNotEqual(seven['fingerprint'], json.loads(base)['fingerprint'])
        self.assertEqual(seven['timeline']['format'], '7v7')

    def test_the_measurement_fingerprint_follows_its_inputs(self):
        a = self.observe({})
        b = coaching.build_observations(RESULT, team='A', coaching={}, result_sha='e'*64, stages=self.fakes.stages())
        c = self.observe({}, settings={'direction': {'A': 'left'}})
        self.assertEqual(len({a['fingerprint'], b['fingerprint'], c['fingerprint']}), 3)
        self.assertEqual(a['engine']['measurement'], a['fingerprint'][:16])
        both = coaching.build_observations(RESULT, coaching={}, result_sha=self.rsha, stages=self.fakes.stages())
        self.assertEqual(set(both), {'A', 'B'})
        self.assertNotEqual(both['A']['fingerprint'], both['B']['fingerprint'])   # the team is an input
        self.assertEqual(both['B']['opportunities'], [])

    def test_given_series_and_possession_are_used(self):
        coaching.build_observations(RESULT, team='A', coaching={}, result_sha=self.rsha, series=object(),
                                    possession={'available': False}, stages=self.fakes.stages())
        self.assertNotIn('series', self.fakes.calls)
        self.assertNotIn('possession', self.fakes.calls)
        with self.assertRaises(ValueError):
            coaching.build_observations(RESULT, team='C', coaching={}, result_sha=self.rsha,
                                        stages=self.fakes.stages())

    def test_result_sha_defaults_to_the_canonical_json(self):
        obs = coaching.build_observations(RESULT, team='A', coaching={}, stages=self.fakes.stages())
        self.assertEqual(obs['inputs']['result'], coaching.result_sha_of(RESULT))

    def test_a_stage_that_breaks_the_contract_fails_where_it_ran(self):
        def bad_detect(timeline):
            out = self.fakes.detect(timeline)
            out['opportunities'][0]['relations']['d1'][0] = float('nan')
            return out
        with self.assertRaises(S.SchemaError) as ctx:
            coaching.build_observations(RESULT, team='A', coaching={}, result_sha=self.rsha,
                                        stages=self.fakes.stages(detect=bad_detect))
        self.assertEqual(ctx.exception.kind, 'obs')
        obs = coaching.build_observations(RESULT, team='A', coaching={}, result_sha=self.rsha,
                                          stages=self.fakes.stages(detect=bad_detect), validate=False)
        self.assertTrue(S.validate('obs', obs))

    def test_role_tier(self):
        obs = self.observe(settings_with())
        roles = coaching.build_roles(obs, settings_with(), stages=self.fakes.stages())
        self.assertEqual(S.validate('roles', roles), [])
        self.assertEqual(roles['measurement'], obs['fingerprint'])
        self.assertEqual(self.fakes.role_settings, {'roles': settings_with()['teams']['A']['roles'],
                                                    'formation': {'intended': '4-3-3', 'periods': []}})
        same = coaching.build_roles(obs, settings_with(playingLevel=1, style='counter'), stages=self.fakes.stages())
        self.assertEqual(canonical(same), canonical(roles))
        other = coaching.build_roles(obs, settings_with(formation={'intended': '4-4-2', 'periods': []}),
                                     stages=self.fakes.stages())
        self.assertNotEqual(other['fingerprint'], roles['fingerprint'])
        with self.assertRaises(ValueError):
            coaching.build_roles(obs, {}, team='B', stages=self.fakes.stages())

    def test_interpretation_tier(self):
        obs, roles, report = self.full(settings_with())
        self.assertEqual(S.validate('report', report), [])
        self.assertEqual(S.cross_check(report, obs, roles), [])
        self.assertEqual(report['fingerprints']['measurement'], obs['fingerprint'])
        self.assertEqual(report['fingerprints']['roles'], roles['fingerprint'])
        self.assertEqual(report['range'], {'t0': 720.0, 't1': 899.8, 'seconds': 179.8, 'whole': True})
        self.assertEqual(self.fakes.range, (720.0, 899.8))
        f = report['needsWork'][0]
        self.assertEqual(f['guardrails']['alternative'], 'pass')
        self.assertEqual(f['suggestedAction'], {'principle': 'cover_first_defender', 'mode': 'zone',
                                                'alternatives': [S.example('alternative')['id']]})
        self.assertIsNone(report['generated'])
        order = [c for c in self.fakes.calls if c in ('profile', 'assess', 'alternatives', 'annotate', 'explain',
                                                      'drills', 'report')]
        self.assertEqual(order, ['profile', 'assess', 'alternatives', 'annotate', 'explain', 'drills', 'report'])

    def test_a_level_change_reinterprets_only(self):
        obs, roles, report = self.full(settings_with())
        calls = len(self.fakes.calls)
        for level in (1, 2, 6):
            other = coaching.interpret(obs, roles, settings_with(playingLevel=level), stages=self.fakes.stages())
            self.assertEqual(S.validate('report', other), [])
            self.assertNotEqual(other['fingerprints']['interpretation'], report['fingerprints']['interpretation'])
            self.assertEqual(other['fingerprints']['measurement'], report['fingerprints']['measurement'])
        new = self.fakes.calls[calls:]
        self.assertFalse({'series', 'possession', 'situations', 'detect', 'moments', 'rolebook'} & set(new))
        low = coaching.interpret(obs, roles, settings_with(playingLevel=2), stages=self.fakes.stages())
        self.assertEqual(low['needsWork'][0]['verdict'], 'try_this')
        renamed = coaching.interpret(obs, roles, settings_with(squad={'id': 'sq-'+'b'*12, 'name': 'Other'}),
                                     stages=self.fakes.stages())
        self.assertEqual(renamed['fingerprints']['interpretation'], report['fingerprints']['interpretation'])

    def test_a_range_is_an_interpretation_option(self):
        obs, roles, whole = self.full(settings_with())
        part = coaching.interpret(obs, roles, settings_with(), t0=700.0, t1=800.0, stages=self.fakes.stages())
        self.assertEqual(part['range'], {'t0': 720.0, 't1': 800.0, 'seconds': 80.0, 'whole': False})
        self.assertEqual(self.fakes.range, (720.0, 800.0))
        self.assertNotEqual(part['fingerprints']['interpretation'], whole['fingerprints']['interpretation'])
        self.assertEqual(part['fingerprints']['measurement'], whole['fingerprints']['measurement'])

    def test_files_that_do_not_belong_together_are_refused(self):
        obs, roles, _ = self.full(settings_with())
        stale = dict(roles, measurement='0'*64)
        with self.assertRaises(ValueError):
            coaching.interpret(obs, stale, settings_with(), stages=self.fakes.stages())
        with self.assertRaises(ValueError):
            coaching.interpret(obs, roles, settings_with(), team='B', stages=self.fakes.stages())

    def test_a_report_citing_an_unmeasured_moment_is_refused(self):
        obs, roles, _ = self.full(settings_with())

        def invent(findings, obs, roles, profile, alternatives):
            ev = S.example('report')['keyMoments']
            ev[2]['id'] = 'ev-'+'9'*12
            findings[0]['timestamps'][2]['evidence'] = ev[2]['id']
            return ev
        with self.assertRaises(S.SchemaError) as ctx:
            coaching.interpret(obs, roles, settings_with(), stages=self.fakes.stages(annotate=invent))
        self.assertIn('not a measured moment', str(ctx.exception))

    def test_principle_only_alternatives(self):
        obs, roles, _ = self.full(settings_with())

        def no_zone(finding, obs, roles, profile):
            n = S.example('no_alternative')
            n['finding'] = finding['id']
            return n
        report = coaching.interpret(obs, roles, settings_with(), stages=self.fakes.stages(alternatives=no_zone))
        f = report['needsWork'][0]
        self.assertEqual(f['suggestedAction'], {'principle': 'cover_first_defender', 'mode': 'principle',
                                                'alternatives': []})
        self.assertEqual(report['noAlternatives'][0]['reason'], 'no_reachable_fix')

        def none(finding, obs, roles, profile):
            return None
        report = coaching.interpret(obs, roles, settings_with(), stages=self.fakes.stages(alternatives=none))
        self.assertEqual(report['needsWork'][0]['guardrails']['alternative'], 'pass')


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


class FingerprintTests(unittest.TestCase):
    def test_tier_code(self):
        m = coaching.code_files('measurement')
        for rel in ('possession.py', 'coaching/series.py', 'team_shape.py', 'analytics.py', 'pitch_model.py',
                    'registration.py', 'labels.py', 'field.py', 'geometry.py', 'ball_path.py',
                    'coaching/situations.py', 'coaching/contextual.py', 'coaching/schema.py', 'coaching/__init__.py'):
            self.assertIn(rel, m)
        self.assertNotIn('coaching/assess.py', m)
        self.assertNotIn('coaching/store.py', coaching.code_files('interpretation'))
        self.assertIn('coaching/roles.py', coaching.code_files('role'))
        for tier in ('measurement', 'role', 'interpretation'):
            self.assertRegex(coaching.code_sha(tier), r'^[0-9a-f]{64}$')
        self.assertEqual(len({coaching.code_sha(t) for t in ('measurement', 'role', 'interpretation')}), 3)

    def test_imports_are_followed_from_the_source(self):
        found = coaching._imports('coaching/series.py')
        self.assertTrue({'analytics.py', 'labels.py', 'pitch_model.py', 'registration.py', 'team_shape.py',
                         'coaching/episodes.py'} <= found, found)
        self.assertIn('ball_path.py', coaching._imports('possession.py'))
        self.assertEqual(coaching._imports('coaching/does_not_exist.py'), set())

    def test_inputs_and_names(self):
        team = S.example('team_settings')
        m = coaching.measurement_inputs('a'*64, 'b'*64, team, 'A')
        self.assertEqual(m['measurement'], {'format': {'players': 11, 'goalkeeper': True}, 'field': None})
        self.assertNotIn('playingLevel', json.dumps(m))
        r = coaching.role_inputs('c'*64, team, 'A')
        self.assertEqual(set(r['role']), {'roles', 'formation'})
        i = coaching.interpretation_inputs('c'*64, 'd'*64, team, 'A', 10.004, 20.0)
        self.assertEqual(i['range'], [10.0, 20.0])
        self.assertEqual(i['interpretation'], {'ageGroup': 'U14', 'playingLevel': 4, 'style': 'unknown',
                                               'federation': 'auto'})
        self.assertRegex(coaching.fingerprint(i), r'^[0-9a-f]{64}$')
        self.assertEqual(coaching.fingerprint(i), coaching.fingerprint(json.loads(json.dumps(i))))
        self.assertEqual(coaching.cache_name('obs', 'A', 'a'*64), 'obs-A-'+'a'*16+'.json')
        self.assertEqual(coaching.cache_name('roles', 'B', 'a'*64, 'b'*64), 'roles-B-'+'a'*16+'-'+'b'*16+'.json')
        self.assertEqual(coaching.cache_name('report', 'A', 'a'*64, 'b'*64, 'c'*64),
                         'report-A-'+'a'*16+'-'+'b'*16+'-'+'c'*16+'.json')
        with self.assertRaises(ValueError):
            coaching.cache_name('pdf', 'A', 'a'*64)

    def test_shape_settings_sha_matches_the_store(self):
        raw = {'direction': {'A': 'left', 'switchAt': 45.0}, 'lineup': {'A': 9}, 'formations': [], 'junk': 1}
        self.assertEqual(coaching.shape_settings_sha(raw), store.settings_sha(team_shape.clean_settings(raw)))
        self.assertEqual(coaching.shape_settings_sha(None), coaching.shape_settings_sha({}))


if __name__ == '__main__':
    unittest.main()
