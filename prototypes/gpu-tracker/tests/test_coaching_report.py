"""The coaching report (soccer/coaching/report.py) assembled from Observation records: nothing forced (a strength-only
report has no weaknesses, a limitations-only report no findings), insufficient and low-confidence observations
listed as limitations with their own reason (the run's data-quality limits once, with what would help), at most three
priorities drawn from the improvements then the reviews (a player's role check never one), drills linked only where
written for the finding's key, sections without a measurement marked unavailable with a pointer to the limitations,
strengths of one compact block grouped into one finding, the block height and ball location as the context paragraph,
every sentence fact-checked, ids rather than display labels, and context levels changing words only.

The fixtures come from test_coaching_language.py (shaped after the audited broadcast, amateur and Veo runs).

Run:  python -m unittest test_coaching_report          (from this directory)
"""
import json
import math
import os
import re
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from soccer.coaching import drills, language, report  # noqa: E402
import test_coaching_language as fx  # noqa: E402

PITCH = {'length': 105.0, 'width': 68.0, 'source': 'standard'}
GENERATED = '2026-10-18T20:00:00'


def meta_broadcast_42():
    return {'fingerprint': 'f'*16, 'video': {'name': '2bac432528d0e568.mp4', 'key': '2bac432528d0e568'},
            'run': '20261018-024712-3485a2', 'range': {'t0': 42.433, 't1': 222.4, 'seconds': 180.0},
            'segmentStart': 42.433, 'pitch': PITCH, 'footage': 'broadcast',
            'registration': {'high': .435, 'medium': .43, 'low': .135}}


def meta_broadcast_0():
    return {'range': {'t0': 0.0, 't1': 59.967, 'seconds': 60.0}, 'segmentStart': 0.0, 'pitch': PITCH,
            'footage': 'broadcast', 'registration': {'high': .854, 'medium': .146, 'low': 0.0}}


def meta_veo(observed=False):
    meta = {'range': {'t0': 1.0, 't1': 30.967, 'seconds': 30.0}, 'segmentStart': 1.0, 'pitch': PITCH,
            'footage': 'wide', 'smallScene': True, 'unavailable': {'movementSupport': 'few_players'},
            'limitations': ['few_players', 'small_scene', 'raw_code_9']}
    if observed:
        meta['coverage'] = {'observedOutfieldPlayers': 'Team A 2.0, Team B 2.6 (mean per sample)'}
    return meta


def meta_amateur():
    return {'range': {'t0': 118.2, 't1': 182.0, 'seconds': 63.83}, 'segmentStart': 118.2, 'pitch': PITCH,
            'footage': 'amateur', 'smallScene': True, 'keyless': True,
            'registration': {'high': 0.0, 'medium': 1.0, 'low': 0.0}, 'limitations': ['small_scene', 'keyless']}


def build(obs, evs, meta, level='competitive', team='A', **kw):
    return report.build_report(obs, evs, meta, level, team, generated=GENERATED, **kw)


def texts(rep):
    """Every coach-facing string of a report with its path (records excluded)."""
    out = []

    def walk(x, path):
        if isinstance(x, str):
            out.append((path, x))
        elif isinstance(x, dict):
            for k, v in x.items():
                walk(v, f'{path}.{k}')
        elif isinstance(x, list):
            for i, v in enumerate(x):
                walk(v, f'{path}[{i}]')
    for key in ('summary', 'strengths', 'improvements', 'reviews', 'priorities', 'overview', 'breakdown',
                'confidence'):
        walk(rep[key], key)
    return out


def report_facts(rep, meta=None):
    """What the report's text may state: its own records (observations, evidence, drills, range, counts) and the
    run description it was built from."""
    meta = meta or {}
    facts = language.facts_for(extra={k: meta.get(k) for k in ('pitch', 'registration', 'corrections')})
    ev = {e['id']: e for e in rep['evidence']}
    for o in rep['observations']:
        facts = facts.merge(language.facts_for(o, [ev[e] for e in o['evidence'] if e in ev]))
    counts = {'n': [len(rep['strengths']), len(rep['improvements']), len(rep['reviews']),
                    rep['overview']['coverage']['observations']['suppressed']]}
    facts = facts.merge(language.facts_for(extra={'range': rep['range'], 'counts': counts,
                                                  'advice': list(language.KEYFRAME_ADVICE),
                                                  'observed': report._observed(meta)}))
    for t in rep['training']:
        facts = facts.merge(language.facts_for(drill=t['drill']))
    return facts


class ReportShapeChecks:
    """Checks every report must pass, whatever it contains."""

    def check_report(self, rep, meta=None, facts_extra=None):
        json.loads(report.to_json(rep))                     # strict JSON: no NaN, ASCII
        for k in ('version', 'engineVersion', 'fingerprint', 'generated', 'kind', 'context', 'team', 'teamName',
                  'video', 'run', 'range', 'overview', 'summary', 'strengths', 'improvements', 'reviews',
                  'breakdown', 'priorities', 'training', 'confidence', 'evidence', 'observations', 'feedback'):
            self.assertIn(k, rep)
        self.assertEqual(set(rep['breakdown']), set(report.SECTIONS))
        self.assertLessEqual(len(rep['priorities']), report.MAX_PRIORITIES)
        facts = report_facts(rep, meta)
        if facts_extra:
            facts = facts.merge(language.facts_for(extra=facts_extra))
        short = rep['range']['seconds'] < language.SHORT_SEGMENT_SECONDS
        for path, text in texts(rep):
            if path.endswith(('.observation', '.finding', '.drill', '.label', '.confidence', '.unit', '.name',
                              '.overall')) or '.metrics' in path or '.findings[' in path or '.evidence[' in path \
                    or '.observations[' in path:
                continue
            self.assertEqual(language.validate_text(text, facts), [], f'{path}: {text}')
            self.assertEqual(language.wording_problems(text, short=short, practice='howToPractice' in path), [],
                             f'{path}: {text}')
            self.assertNotIn('{', text, path)
            self.assertNotIn('None', text, path)
            self.assertNotIn('upper level', text, path)
            self.assertNotRegex(text, r'\b[a-z]+_[a-z_]+\b', path)    # no raw codes
            text.encode('cp1252')
        for f in rep['strengths']+rep['improvements']+rep['reviews']:
            self.assertTrue(f['headline'] and f['whatHappened'] and f['whyItMatters'])
            self.assertIn(f['label'], language.LABELS.values())
            self.assertEqual(f['observations'][0], f['observation'])
        for f in rep['strengths']:
            self.assertEqual(f['label'], 'Observed strength')
            self.assertTrue(f['headline'].startswith('Maintained'))
            self.assertTrue(f['whatToImprove'].startswith('To build on it'), f['whatToImprove'])
            self.assertIsNone(f['drill'])
            self.assertIsNone(f['howToPractice'])
        for f in rep['reviews']:
            self.assertEqual(f['label'], 'Review opportunity')
            self.assertTrue(f['headline'].startswith('To review:'))
        for name, sec in rep['breakdown'].items():
            if sec['available']:
                self.assertIsNone(sec['reason'])
                self.assertTrue(sec['findings'] or sec['notes'], name)
            else:
                self.assertTrue(sec['reason'], name)
                self.assertEqual(sec['findings'], [])
        self.assertFalse(rep['breakdown']['attackingStructure']['available'])
        self.assertIn('which team had the ball', rep['breakdown']['attackingStructure']['reason'])
        # Each limitation line is in the limitations once; the breakdown points to it instead of repeating it.
        lims = rep['overview']['limitations']
        self.assertEqual(len(lims), len(set(lims)))
        for sec in rep['breakdown'].values():
            for line in lims:
                self.assertNotIn(line, ' '.join(sec['notes']+[sec['reason'] or '']))
        shown = {f['observation'] for f in rep['strengths']+rep['improvements']+rep['reviews']}
        members = {o for f in rep['strengths']+rep['improvements']+rep['reviews'] for o in f['observations']}
        by_id = {o['id']: o for o in rep['observations']}
        for o in rep['observations']:
            self.assertEqual(o['team'], rep['team'])
        for oid in members:
            self.assertNotEqual(by_id[oid]['classification'], 'insufficient')
            self.assertIn(by_id[oid]['confidence']['overall'], ('high', 'medium'))
        for t in rep['training']:
            for oid in t['findings']:
                self.assertIn(by_id[oid]['recommendationKey'], t['drill']['recommendationKeys'])
                self.assertIn(oid, shown)
        ev_ids = {e['id'] for e in rep['evidence']}
        for f in rep['strengths']+rep['improvements']+rep['reviews']:
            self.assertTrue(set(f['evidence']) <= ev_ids)
        return facts


class ReportTests(ReportShapeChecks, unittest.TestCase):
    def test_strength_only_report_forces_no_weakness(self):
        rep = build(fx.broadcast_strengths(), [], meta_broadcast_0())
        self.check_report(rep, meta_broadcast_0())
        self.assertEqual(rep['kind'], 'segment')
        self.assertEqual(rep['overview']['title'], 'SEGMENT COACHING REPORT')
        self.assertEqual(len(rep['strengths']), 4)
        self.assertEqual((rep['improvements'], rep['reviews'], rep['priorities'], rep['training']), ([], [], [], []))
        self.assertEqual(rep['summary'][1], '4 observed strengths met the evidence rules.')
        self.assertFalse(any('improve' in s.lower() for s in rep['summary']))
        self.assertIn('short segment', rep['summary'][-2])
        self.assertEqual(rep['confidence']['overall'], 'medium')       # a formation reading is at most medium
        self.assertTrue(rep['breakdown']['spacing']['available'])
        self.assertFalse(rep['breakdown']['movementSupport']['available'])
        self.assertEqual(rep['overview']['limitations'], [])
        self.assertEqual(rep['overview']['dataQualityLines'], 0)
        metrics = {m['name']: m for m in rep['strengths'][0]['metrics']}
        self.assertEqual(metrics['Measured time'], {'name': 'Measured time', 'value': 60.0, 'unit': 's'})
        # professional metres are real metres, and say so: a row without the flag reads as the report's (estimated)
        self.assertEqual(metrics['Professional median'], {'name': 'Professional median', 'value': 3.2, 'unit': 'm',
                                                          'estimated': False})
        self.assertTrue(metrics['Median']['estimated'])

    def test_insufficient_only_report_lists_limitations_and_nothing_else(self):
        rep = build(fx.veo_insufficient(), [], meta_veo())
        self.check_report(rep, meta_veo())
        self.assertEqual((rep['strengths'], rep['improvements'], rep['reviews']), ([], [], []))
        self.assertEqual((rep['priorities'], rep['training']), ([], []))
        self.assertEqual(rep['confidence']['overall'], 'low')
        self.assertIn('Team analysis was not possible on this footage', rep['summary'][2])
        for name, sec in rep['breakdown'].items():
            self.assertFalse(sec['available'], name)
        self.assertEqual(rep['breakdown']['spacing']['reason'], 'Not enough reliable measurement: team width and '
                                                                'back line to midfield distance could not be '
                                                                'assessed (see the limitations).')
        self.assertEqual(rep['breakdown']['movementSupport']['reason'], 'Only a few players of each team were in '
                                                                        'view.')

    def test_without_a_reference_the_reason_is_the_reference(self):
        # broadcast 0-60 s team A with no reference: 99-100% of the time measured, every comparative metric not
        # assessed; "too little of the play could be measured" and "not enough reliable measurement" were false
        obs = [fx.observation('compactness', 'insufficient', 'team_width', measured=59.6, eligible=.99, longest=42.6,
                              confidence='high', insufficientReasons=['no_reference']),
               fx.observation('back_line_coordination', 'insufficient', measured=60.0, eligible=1.0, longest=60.0,
                              confidence='high', insufficientReasons=['no_reference'])]
        meta = dict(meta_broadcast_0(), limitations=['no_reference'])
        rep = build(obs, [], meta)
        self.check_report(rep, meta)
        self.assertEqual(rep['overview']['limitations'][0], 'No professional reference was available, so nothing in '
                                                            'this report is compared with professional play.')
        self.assertIn('No professional reference was available, so the measurements could not be compared; see the '
                      'limitations.', rep['summary'])
        self.assertFalse(any('Too little of the play' in x for x in rep['summary']))
        self.assertEqual(rep['breakdown']['spacing']['reason'], 'No professional reference was available: team width '
                                                                'could not be compared (see the limitations).')
        self.assertFalse(any('Comparisons are with professional tracking data' in x
                             for x in rep['confidence']['notes']))
        with_ref = build(obs, [], meta_broadcast_0())
        self.assertTrue(any('Comparisons are with professional tracking data' in x
                            for x in with_ref['confidence']['notes']))

    def test_a_footage_with_few_players_in_view_gets_one_block(self):
        # Veo 1-31 s: the first report had 12 limitation lines (576 words), "only a few players of each team were in
        # view" 10 times.
        rep = build(fx.veo_insufficient(), [], meta_veo(observed=True))
        self.check_report(rep, meta_veo(observed=True))
        lim = rep['overview']['limitations']
        self.assertEqual(len(lim), 6)
        self.assertEqual(rep['overview']['dataQualityLines'], 2)
        self.assertTrue(lim[0].startswith('Team analysis was not possible: on average only 2.0 players of Team A and '
                                          '2.6 of Team B were in view'))
        self.assertIn('a camera-view limit, not a judgement on the team', lim[0])
        self.assertEqual(lim[1], "What would help: use a higher or wider camera view that keeps both teams' lines in "
                                 "the picture.")
        self.assertEqual(lim[2:], ['Team width: not assessed on this footage.',
                                   'Back line to midfield distance: not assessed on this footage.',
                                   'Back-line evenness: not assessed on this footage.',
                                   'Team organisation: not assessed on this footage.'])
        self.assertEqual({x['observation'] for x in rep['overview']['notAssessed']},
                         {o['id'] for o in rep['observations']})
        self.assertLess(sum(len(s.split()) for s in lim), 120)
        self.assertEqual(sum(t.count('only a few players') for _, t in texts(rep)), 1)
        self.assertFalse(any('raw_code' in s for s in lim))

    def test_mixed_report(self):
        obs, evs = fx.broadcast_mixed()
        rep = build(obs, evs, meta_broadcast_42())
        self.check_report(rep, meta_broadcast_42())
        self.assertEqual([f['label'] for f in rep['improvements']],
                         ['High-confidence concern', 'Potential improvement', 'Potential improvement'])
        self.assertEqual(len(rep['reviews']), 3)
        self.assertEqual(len(rep['priorities']), 3)
        imp_ids = [f['observation'] for f in rep['improvements']]
        self.assertEqual([p['finding'] for p in rep['priorities']], imp_ids[:3])
        by_id = {o['id']: o for o in rep['observations']}
        self.assertEqual(by_id[rep['priorities'][0]['finding']]['detector'], 'back_line_coordination')
        self.assertEqual([t['drill']['id'] for t in rep['training']],
                         ['back_line_coordination', 'defensive_compactness', 'lateral_shifting',
                          'formation_shadow_play'])
        self.assertEqual(len(rep['training'][0]['findings']), 2)          # the uneven line and the high-line review
        self.assertEqual(rep['training'][0]['drill']['variation'],
                         drills.drill(drills.load_drills(), 'back_line_coordination')['variations']['competitive'])
        self.assertEqual(rep['observations'][-1]['team'], 'A')             # the other team's record left out
        self.assertEqual(len(rep['observations']), 10)
        self.assertEqual(rep['overview']['coverage']['observations'],
                         {'total': 10, 'shown': 6, 'limitations': 1, 'suppressed': 1})
        self.assertEqual(len(rep['overview']['context']), 1)
        self.assertIn('Ball location, for context', rep['overview']['context'][0])
        notes = rep['breakdown']['spacing']['notes']
        self.assertEqual(notes[0], 'Measured for up to 2.2 minutes of the 3.0 minutes analysed.')
        self.assertTrue(any(n.startswith('Team width: median 55.1 m (est.) over 105 s') for n in notes))
        self.assertEqual(notes[-1], 'Not assessed here: midfield to front line distance (see the limitations).')
        self.assertEqual(rep['overview']['limitations'], [
            'To measure more: choose a spell where both lines of the team are in view.',
            'Midfield to front line distance: not assessed; measured 12 s (7.0% of the time), longest continuous '
            'stretch 4.4 s; the players of the lines involved were not all in view together for part of the time.'])
        self.assertEqual(rep['confidence']['overall'], 'medium')            # the formation review is medium
        self.assertIn('1 further observation was not reported (suppressed by the analysis).',
                      rep['confidence']['notes'])
        gap = next(f for f in rep['improvements'] if by_id[f['observation']]['detector'] == 'unit_gap_dm')
        self.assertEqual(gap['drill'], 'defensive_compactness')
        self.assertEqual(len(gap['evidence']), 2)
        self.assertTrue(gap['video'].startswith('Moments to watch: 01:04.80'))
        self.assertEqual(len(rep['evidence']), 4)
        self.assertEqual(rep['range'], {'t0': 42.433, 't1': 222.4, 'segmentT0': 0.0, 'segmentT1': 179.967,
                                        'seconds': 180.0})

    def test_a_players_support_review_is_a_role_check(self):
        obs, evs = fx.broadcast_mixed()
        iso = next(o for o in obs if o['detector'] == 'isolation')
        rep = build([iso], evs, meta_broadcast_42())
        self.check_report(rep, meta_broadcast_42())
        f = rep['reviews'][0]
        self.assertEqual(f['headline'], 'To review: distance between A-11 and the nearest team-mate (role check)')
        self.assertEqual((f['drill'], f['howToPractice'], rep['priorities'], rep['training']), (None, None, [], []))
        self.assertIn("Main point to review: Check A-11's role and support distances.", rep['summary'])
        rep = build([o for o in obs if o['classification'] == 'review'], evs, meta_broadcast_42())
        self.assertEqual(len(rep['priorities']), 2)                       # the two reviews, never the role check
        self.assertEqual(rep['reviews'][-1]['observation'], iso['id'])    # and it is listed last

    def test_priorities_take_improvements_first_then_reviews(self):
        obs, evs = fx.broadcast_mixed()
        keep = [o for o in obs if o['classification'] == 'review' or o['detector'] == 'unit_gap_dm']
        rep = build(keep, evs, meta_broadcast_42())
        kinds = ['improvement' if any(f['observation'] == p['finding'] for f in rep['improvements']) else 'review'
                 for p in rep['priorities']]
        self.assertEqual(kinds, ['improvement', 'review', 'review'])
        self.assertTrue(rep['priorities'][1]['title'].startswith('Review'))
        reviews_only = build([o for o in obs if o['classification'] == 'review'], evs, meta_broadcast_42())
        self.assertIn('Main point to review:', ' '.join(reviews_only['summary']))

    def test_strengths_of_one_compact_block_are_one_finding(self):
        # Broadcast 42-222 s team B: five strengths of one compact block, one "why it matters" sentence printed three
        # times. Width, length and area become one finding; both unit gaps another.
        obs, evs = fx.compact_block()
        meta = dict(meta_broadcast_42(), registration=None)
        for level in language.LEVELS:
            rep = build(obs, evs, meta, level, team='B')
            self.check_report(rep, meta)
            self.assertEqual(len(rep['strengths']), 2, level)
            shape = next(f for f in rep['strengths'] if len(f['observations']) == 3)
            lines = next(f for f in rep['strengths'] if len(f['observations']) == 2)
            self.assertEqual(shape['observations'], [o['id'] for o in obs[:3]])
            self.assertEqual(lines['observations'], [o['id'] for o in obs[3:]])
            self.assertEqual(len(shape['evidence']), 3)
            self.assertEqual(rep['overview']['coverage']['observations']['shown'], 5)
            self.assertEqual(rep['breakdown']['spacing']['findings'], [shape['observation'], lines['observation']]
                             if rep['strengths'][0] is shape else [lines['observation'], shape['observation']])
            text = ' '.join(t for _, t in texts(rep))
            self.assertEqual(text.count('This matters most when'), 2, level)       # once per finding
        rep = build(obs, evs, meta, team='B')
        shape = next(f for f in rep['strengths'] if len(f['observations']) == 3)
        self.assertEqual(shape['headline'], 'Maintained a compact team shape: width, length and area below the '
                                            'professional median')
        self.assertIn("Width: median 39.3 m (est.) over 2.4 minutes; in the strictest comparison, with the team's "
                      "centre in its own third (68 s), median 38.7 m (est.) against a professional median of 40.3 m, "
                      "beyond 48.0 m for 5.0% of that time.", shape['whatHappened'])
        self.assertIn('Professional teams go beyond each of those levels about a quarter of the time.',
                      shape['whatHappened'])
        self.assertEqual(shape['confidenceNote'], 'Medium confidence: the pitch registration was of medium '
                                                  'confidence.')
        names = [m['name'] for m in shape['metrics']]
        self.assertEqual(names, ['Measured time', 'Team width: median', 'Team width: professional median',
                                 'Team width: share of time beyond the professional upper quarter',
                                 'Team length: median', 'Team length: professional median',
                                 'Team length: share of time beyond the professional upper quarter',
                                 'Team area: median', 'Team area: professional median',
                                 'Team area: share of time beyond the professional upper quarter'])
        single = build(obs[:1]+obs[3:4], evs, meta, team='B')               # one of each: nothing to group
        self.assertEqual([len(f['observations']) for f in single['strengths']], [1, 1])

    def test_context_paragraph_names_both_block_heights_and_the_ball(self):
        obs = fx.every_finding()[0]
        keep = [o for o in obs if o['detector'] in ('line_height_and_gap', 'ball_context') and
                o['classification'] == 'neutral']
        rep = build(keep, [], meta_broadcast_42())
        self.check_report(rep, meta_broadcast_42())
        self.assertEqual(len(rep['overview']['context']), 2)
        self.assertTrue(rep['overview']['context'][0].startswith("Block height: Team A's back line stood a median "
                                                                 "30.6 m (est.)"))
        self.assertTrue(rep['overview']['context'][1].startswith('Ball location, for context'))
        self.assertFalse(rep['breakdown']['defensiveOrganization']['available'])

    def test_amateur_caveats_stated_once(self):
        obs, evs = fx.amateur_set()
        rep = build(obs, evs, meta_amateur())
        self.check_report(rep, meta_amateur())
        self.assertEqual(len(rep['improvements']), 1)
        self.assertEqual(rep['confidence']['overall'], 'medium')
        notes = ' '.join(rep['confidence']['notes'])
        self.assertIn('medium for 100%', notes)
        lim = rep['overview']['limitations']
        self.assertEqual(rep['overview']['dataQualityLines'], 3)
        self.assertTrue(lim[0].startswith('Players appear small in this footage'))
        self.assertIn('support distances and ball location are only measured on broadcast-style footage', lim[0])
        self.assertTrue(lim[1].startswith('No pitch keyframes were set'))
        self.assertTrue(lim[2].startswith('What would help: set 2-3 pitch keyframes'))
        everything = ' '.join(t for _, t in texts(rep)).lower()
        self.assertEqual(everything.count('keyframes were set'), 1)          # the first report said it 9 times
        self.assertEqual(everything.count('players appear small'), 1)
        f = rep['improvements'][0]
        self.assertEqual(f['confidenceNote'], 'Medium confidence because of the footage limits in the data-quality '
                                              'note.')
        self.assertTrue(any(s.startswith("Team organisation: not shown because the measurement confidence was low")
                            for s in lim))
        self.assertIn("Ball location: not measured on this footage; the ball's location is only used on "
                      "broadcast-style footage", ' '.join(lim))
        self.assertEqual(rep['reviews'], [])                                  # the low-confidence review withheld
        # High confidence on small-scene footage still reads medium overall.
        high = [dict(o, confidence=dict(o['confidence'], overall='high')) for o in obs[:1]]
        self.assertEqual(build(high, evs, meta_amateur())['confidence']['overall'], 'medium')

    def test_corrected_ids_are_tokens_rendered_at_view_time(self):
        obs, evs = fx.corrected_set()
        meta = {'range': {'t0': 0.0, 't1': 59.967, 'seconds': 60.0}, 'corrections': 2}
        rep = build(obs, evs, meta, team='B')
        self.check_report(rep, meta)
        blob = json.dumps({k: rep[k] for k in ('summary', 'improvements', 'priorities')})
        self.assertIn('B-12', blob)
        self.assertNotIn('B-09', blob)                                   # no display label is ever stored
        self.assertEqual(rep['observations'][0]['players'], ['B-12'])
        shown = language.resolve_report(rep, {'B-12': 'B-09'})
        self.assertEqual(shown['improvements'][0]['headline'], 'B-09 separated from the nearest team-mate')
        self.assertEqual(shown['priorities'][0]['title'], 'Shorten the support distances around B-09')
        self.assertEqual(shown['observations'][0]['players'], ['B-12'])        # records keep the id
        self.assertEqual(shown['priorities'][0]['finding'], rep['priorities'][0]['finding'])
        self.assertEqual(shown['context'], rep['context'])                      # 'competitive' is not text
        self.assertIn('This report reflects 2 manual corrections to the tracking.', rep['confidence']['notes'])
        self.assertTrue(any('labelled B here' in n for n in rep['confidence']['notes']))
        gone = language.resolve_report(rep, {'B-07': 'B-07'})                   # B-12 no longer in the roster
        self.assertEqual(gone['improvements'][0]['headline'], 'not in the roster now (was B-12) separated from the '
                                                              'nearest team-mate')

    def test_context_levels_change_wording_not_measurements(self):
        obs, evs = fx.broadcast_mixed()
        reps = {lv: build(obs, evs, meta_broadcast_42(), lv) for lv in language.LEVELS}
        base = reps['competitive']
        for lv, rep in reps.items():
            self.check_report(rep, meta_broadcast_42())
            self.assertEqual(rep['context'], lv)
            for key in ('strengths', 'improvements', 'reviews'):
                self.assertEqual([(f['observation'], f['label'], f['metrics'], f['drill'], f['evidence'],
                                   sorted(re.findall(r'\d+(?:\.\d+)?', f['whatHappened'])))
                                  for f in rep[key]],
                                 [(f['observation'], f['label'], f['metrics'], f['drill'], f['evidence'],
                                   sorted(re.findall(r'\d+(?:\.\d+)?', f['whatHappened'])))
                                  for f in base[key]], f'{lv} {key}')
            self.assertEqual([(p['finding'], p['drill']) for p in rep['priorities']],
                             [(p['finding'], p['drill']) for p in base['priorities']])
            self.assertEqual(rep['confidence']['overall'], base['confidence']['overall'])
            self.assertEqual([t['drill']['id'] for t in rep['training']], [t['drill']['id'] for t in base['training']])
            self.assertEqual(rep['training'][0]['drill']['level'], lv)
        heads = {lv: [f['headline'] for f in reps[lv]['improvements']] for lv in language.LEVELS}
        self.assertNotEqual(heads['development'], heads['advanced'])
        self.assertIn('the defenders', ' '.join(heads['development']).lower())
        self.assertIn('vertical compactness', reps['advanced']['improvements'][1]['whyItMatters'])
        with self.assertRaises(ValueError):
            build(obs, evs, meta_broadcast_42(), 'elite')

    def test_match_kind_from_twenty_minutes(self):
        obs = fx.broadcast_strengths()
        rep = build(obs, [], {'range': {'t0': 0.0, 't1': 1500.0}, 'pitch': PITCH})
        self.assertEqual(rep['kind'], 'match')
        self.assertEqual(rep['overview']['title'], 'MATCH COACHING REPORT')
        self.assertTrue(rep['summary'][0].startswith('Match report for Team A (as labelled in this run), 00:00.00 to '
                                                     '25:00.00 (25.0 minutes of video).'))
        self.assertFalse(any('short segment' in s for s in rep['summary']))
        self.assertEqual(build(obs, [], {'range': {'t0': 0.0, 't1': 1199.0}})['kind'], 'segment')

    def test_low_confidence_findings_are_withheld_as_limitations(self):
        obs, evs = fx.broadcast_mixed()
        gap = dict(obs[0], confidence=dict(obs[0]['confidence'], overall='low', reasons=['crossings']))
        rep = build([gap], evs, meta_broadcast_42())
        self.check_report(rep, meta_broadcast_42())
        self.assertEqual(rep['improvements'], [])
        self.assertEqual(rep['overview']['limitations'], [
            'Back line to midfield distance: not shown because the measurement confidence was low; players crossed '
            'each other many times, which makes identities less certain.'])
        self.assertFalse(rep['breakdown']['spacing']['available'])

    def test_high_confidence_concern_needs_high_confidence_three_occurrences_and_three_minutes(self):
        obs, evs = fx.broadcast_mixed()
        line = next(o for o in obs if o['detector'] == 'back_line_coordination')
        medium = dict(line, confidence=dict(line['confidence'], overall='medium'))
        rep = build([medium], evs, meta_broadcast_42())
        self.assertEqual(rep['improvements'][0]['label'], 'Potential improvement')
        self.assertNotIn('high-confidence concern', rep['improvements'][0]['whatHappened'])
        two = dict(line, occurrences=line['occurrences'][:2], occurrenceCount=2)
        self.assertEqual(build([two], evs, meta_broadcast_42())['improvements'][0]['label'], 'Potential improvement')
        short = dict(line, measuredSeconds=114.0)
        self.assertEqual(build([short], evs, meta_broadcast_42())['improvements'][0]['label'],
                         'Potential improvement')
        self.assertEqual(build([line], evs, meta_broadcast_42())['improvements'][0]['label'],
                         'High-confidence concern')

    def test_a_careless_engine_is_withheld_not_shown(self):
        obs, evs = fx.broadcast_mixed()

        class Careless(language.TemplateEngine):
            name = 'careless'

            def __init__(self, change):
                self.change = change

            def render(self, ctx):
                out = super().render(ctx)
                if 'headline' in out:
                    out = dict(out, **self.change(out))
                return out
        cases = {'a fabricated number': lambda o: {'whatHappened': o['whatHappened']+' The gap reached 31.7 m.'},
                 'a wrong player': lambda o: {'headline': o['headline']+' (A-99)'},
                 'blame': lambda o: {'whyItMatters': 'A terrible mistake by the defenders.'},
                 'a phase claim': lambda o: {'whatHappened': o['whatHappened']+' It came after a turnover.'},
                 'a recurring claim': lambda o: {'headline': 'A recurring weakness'}}
        for what, change in cases.items():
            rep = build(obs[:1], evs, meta_broadcast_42(), engine=Careless(change))
            self.assertEqual(rep['improvements'], [], what)
            self.assertEqual(rep['textEngine'], 'careless')
            self.assertEqual(rep['overview']['limitations'], [
                'Back line to midfield distance: not shown because its wording could not be checked against the '
                'measurements.'], what)

    def test_unknown_keys_and_classifications_are_never_invented(self):
        obs, evs = fx.broadcast_mixed()
        odd = dict(obs[0], recommendationKey='press_higher', id='obs-odd000000001')
        weird = dict(obs[1], classification='excellent', id='obs-odd000000002')
        rep = build([odd, weird], evs, meta_broadcast_42())
        self.check_report(rep, meta_broadcast_42())
        self.assertEqual(rep['improvements'], [])
        self.assertIn('no reviewed wording exists for this observation yet', rep['overview']['limitations'][0])

    def test_drills_link_only_where_written_for_the_key(self):
        obs, evs = fx.broadcast_mixed()
        gap = dict(obs[0], drillIds=['support_angles', 'no_such_drill'])     # neither fits compact_lines
        rep = build([gap], evs, meta_broadcast_42())
        self.assertEqual(rep['improvements'][0]['drill'], 'defensive_compactness')   # the library's own choice
        empty = build([gap], evs, meta_broadcast_42(), library={'drills': []})
        self.assertIsNone(empty['improvements'][0]['drill'])
        self.assertIsNone(empty['improvements'][0]['howToPractice'])
        self.assertEqual(empty['training'], [])
        self.assertIsNone(empty['priorities'][0]['drill'])

    def test_json_is_strict_and_rounded(self):
        obs, evs = fx.broadcast_mixed()
        obs[0]['metrics']['extra'] = float('nan')
        rep = build(obs, evs, meta_broadcast_42())
        text = report.to_json(rep)
        self.assertNotIn('NaN', text)
        text.encode('ascii')
        self.assertIsNone(next(o for o in rep['observations'] if o['id'] == obs[0]['id'])['metrics']['extra'])
        for f in rep['improvements']+rep['strengths']:
            for m in f['metrics']:
                if m['unit'] == 'm':
                    self.assertEqual(m['value'], round(m['value'], 1))
                    # the team's metres are estimates; a professional level is real metres
                    self.assertEqual(m.get('estimated', False), m['name'] not in ('Threshold', 'Professional median',
                                                                                  'Professional upper quarter'))
                if m['unit'] == 'share':
                    self.assertEqual(m['value'], round(m['value'], 3))
                self.assertFalse(isinstance(m['value'], float) and not math.isfinite(m['value']))
        self.assertNotIn('NaN', report.to_json(build(obs, evs, meta_broadcast_42(), 'advanced')))

    def test_team_name_and_validation(self):
        rep = build(fx.broadcast_strengths(), [], meta_broadcast_0(), team_name='U12 Lions')
        self.check_report(rep, meta_broadcast_0(), facts_extra={'teamName': 'U12 Lions'})
        self.assertEqual(rep['teamName'], 'U12 Lions')
        self.assertTrue(rep['summary'][0].startswith('Segment report for U12 Lions, 00:00.00'))
        with self.assertRaises(ValueError):
            build(fx.broadcast_strengths(), [], meta_broadcast_0(), team='C')
        empty = build([], [], {})
        self.assertEqual(empty['range']['seconds'], 0.0)
        self.assertEqual(empty['confidence']['overall'], 'low')
        json.loads(report.to_json(empty))


if __name__ == '__main__':
    unittest.main()
