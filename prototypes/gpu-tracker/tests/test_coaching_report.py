"""The coaching report: v2 (soccer/coaching/report.py, build_report_v2, DESIGN v2 9.2) and the deprecated v1
(soccer/coaching/report_v1.py, served until wave 7; its names are forwarded from report.py).

v2 (V2ReportTests, V2AssemblyTests, V2ForwardingTests), on EVAL's synthetic scenes through every real stage
(coaching.interpret: profile, assess, alternatives, annotate, explain, drills.select, the report; tests/
coaching_text_fixtures.py): at every level the report validates and cross-checks against its obs and roles files, no
real finding is withheld, every text passes explain's gate (findings) or validate_free (the overview, notices,
captions, the next objective), key moments are exactly the cited moments in time order, needsWork keeps the level's
cap and priority order; the raw observations are the same bytes after every level and style (invariant 2); levels 1-2
name nobody anywhere and level 1 shows no number; one family cites the same moments at levels 4-7 with explanations
below .5 trigram Jaccard; possession unknown is said, never filled; coach-marked spans read "(coach-marked)"; a clear
formation mismatch and a short range are notices; "nothing met the bar" says it is not proof; the shape and movement
diagrams are described only (own players, arrows of at least 1 m); a withheld finding's moments leave the report, a
finding whose moment was not made is withheld, ZONES' no-alternative reason reaches a work item (not a guardrail
review), drill picks the report cannot print are left out and logged; the v2 modules never import the v1 code.

v1 (ReportTests, unchanged behaviour): nothing forced (a strength-only
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
from collections import Counter

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


# =====================================================================================================================
# v2 (DESIGN v2 9.2): report.build_report_v2 through coaching.interpret on EVAL's scenes (every real stage)
# =====================================================================================================================
import copy as _copy  # noqa: E402
import time as _time  # noqa: E402

import coaching_text_fixtures as TX  # noqa: E402
from soccer import coaching  # noqa: E402
from soccer.coaching import explain as EX, language as L, language_v1, report_v1, schema as S  # noqa: E402
from soccer.coaching import assess as AS  # noqa: E402

V2_SCENES = (('unsupported_advance', 'none_behind'), ('defending', 'press_no_cover'), ('possession_unknown', None),
             ('amateur_coach_spans', 'spans'), ('wrong_formation', 'default'), ('short_segment', None))
LEVELS = range(1, 8)


def _ids_in(text):
    return language_v1.ID_TOKEN.findall(text or '')


class V2ReportTests(unittest.TestCase):
    """Every level on the scenes: what interpret returns is a valid, cross-checked report whose every text passes
    the gate, and the sections hold what assess and explain gave."""

    def test_every_scene_and_level_gives_a_valid_report(self):
        for scene, variant in V2_SCENES:
            b, st, obs, roles = TX.tiers(scene, variant)
            for lvl in LEVELS:
                with self.subTest(scene=scene, variant=variant, level=lvl):
                    rep = TX.report(scene, variant, lvl)
                    self.assertEqual(S.validate('report', rep), [])
                    self.assertEqual(S.cross_check(rep, obs, roles), [])
                    self.assertEqual(rep['withheld'], [], 'no real finding is withheld on these scenes')
                    found = TX.findings_of(rep)
                    self.assertEqual(EX.report_problems(found, rep['profile']), [])
                    self.assertLessEqual(len(rep['needsWork']), rep['profile']['evidence']['maxWorkItems'])
                    self.assertEqual([f['priority'] for f in rep['needsWork']],
                                     sorted(f['priority'] for f in rep['needsWork']))
                    times = [e['time'] for e in rep['keyMoments']]
                    self.assertEqual(times, sorted(times))
                    cited = {ts['evidence'] for f in found for ts in f['timestamps']}
                    self.assertEqual(cited, {e['id'] for e in rep['keyMoments']}, 'key moments = cited moments')
                    self.assertIsNotNone(rep['overview']['text'])
                    short = rep['range']['seconds'] < EX.SHORT_RANGE_S
                    for path, text in TX.texts_of(rep):
                        self.assertEqual(S.event_problems(text), [], path)
                        self.assertRegex(text, r'^[ -~]+$', path)
                        if short:
                            low = text.lower()
                            for w in language_v1.SHORT_BANNED:
                                self.assertNotRegex(low, r'(?<![a-z\-])'+re.escape(w)+r'(?![a-z\-])', (path, text))
                    ov = rep['overview']
                    self.assertEqual(EX.validate_free(ov['text'], rep['profile'], None, numbers=False,
                                                      check_facts=False, short=short), [])

    def test_raw_observations_never_change(self):
        """Invariant 2: the obs and roles files are the same bytes after every level's report."""
        b, st, obs, roles = TX.tiers('unsupported_advance', 'none_behind')
        before = json.dumps([obs, roles], sort_keys=True, allow_nan=False)
        for lvl in LEVELS:
            for style in ('unknown', 'possession', 'compact'):
                coaching.interpret(obs, roles, TX.settings_at(st, 'A', lvl, style=style), 'A')
        self.assertEqual(json.dumps([obs, roles], sort_keys=True, allow_nan=False), before)

    def test_levels_one_and_two_name_nobody_and_keep_their_caps(self):
        for scene, variant in V2_SCENES:
            for lvl in (1, 2):
                with self.subTest(scene=scene, level=lvl):
                    rep = TX.report(scene, variant, lvl)
                    for path, text in TX.texts_of(rep):
                        if '.caption' in path and path.startswith('dg-'):
                            continue        # diagram captions name nobody either; checked below
                        self.assertEqual(_ids_in(text), [], (path, text))
                    for d in rep['diagrams']:
                        self.assertEqual(_ids_in(d['caption']), [], d['caption'])
                    self.assertLessEqual(len(rep['needsWork']), lvl)
                    self.assertTrue(all(f['verdict'] == 'try_this' for f in rep['needsWork']))
                    if lvl == 1:
                        for f in TX.findings_of(rep):
                            body = ' '.join(v for v in f['text'].values() if v)
                            self.assertIsNone(re.search(r'\d', language_v1.TIME_TOKEN.sub(' ', body)), body)
                        self.assertIsNone(re.search(r'\d', rep['overview']['text']), rep['overview']['text'])

    def test_same_evidence_across_levels_for_one_family(self):
        """Spec Part 16: the wide-defender family cites the same moments at levels 4-7, with different words."""
        seen = {}
        for lvl in (4, 5, 6, 7):
            rep = TX.report('unsupported_advance', 'none_behind', lvl)
            fam = [f for f in TX.findings_of(rep) if f['observation'] == 'wide_defender_advance']
            self.assertEqual(len(fam), 1, lvl)
            seen[lvl] = fam[0]
        evs = {lvl: [ts['evidence'] for ts in f['timestamps']] for lvl, f in seen.items()}
        self.assertEqual(len({tuple(v) for v in evs.values()}), 1, evs)
        for a in (4, 5, 6):
            for b in range(a+1, 8):
                self.assertLess(EX.jaccard(seen[a]['text']['explanation'], seen[b]['text']['explanation']), .5)

    def test_possession_unknown_is_said_and_not_filled(self):
        rep = TX.report('possession_unknown', None, 4)
        po = rep['overview']['possession']
        self.assertFalse(po['available'])
        self.assertEqual(po['quality'], 'unavailable')
        self.assertEqual(po['reason'], report.UNAVAILABLE_REASON)
        self.assertIn('phase_unavailable', [n['code'] for n in rep['overview']['notices']])
        self.assertIn('could not be followed', rep['overview']['text'])
        for f in TX.findings_of(rep):
            self.assertEqual(f['context']['phase']['state'], 'unknown', 'only phase-free descriptions remain')
        self.assertEqual(rep['needsWork']+rep['reviews']+rep['strengths'], [])

    def test_coach_marked_spans_are_shown_as_coach_marked(self):
        rep = TX.report('amateur_coach_spans', 'spans', 4)
        po = rep['overview']['possession']
        self.assertTrue(po['available'])
        self.assertIn('coach-marked', po['caveat'])
        self.assertIn('spans you marked', rep['overview']['text'])
        phased = [f for f in TX.findings_of(rep) if f['context']['phase']['state'] in ('own', 'opp')]
        self.assertTrue(phased)
        for f in phased:
            self.assertIn('(coach-marked)', f['text']['explanation'])
            self.assertNotIn('(estimated)', f['text']['explanation'])

    def test_a_clear_formation_mismatch_is_a_notice(self):
        rep = TX.report('wrong_formation', 'default', 4)
        n = [x for x in rep['overview']['notices'] if x['code'] == 'formation_mismatch']
        self.assertEqual(len(n), 1)
        self.assertIn('3-5-2', n[0]['text'])
        self.assertIn('4-4-2', n[0]['text'])
        self.assertEqual(_ids_in(n[0]['text']), [])

    def test_short_range_notice_and_no_pattern_words(self):
        rep = TX.report('short_segment', None, 4)
        self.assertIn('short_range', [n['code'] for n in rep['overview']['notices']])
        b, st, obs, roles = TX.tiers('unsupported_advance', 'none_behind')
        t0 = obs['segment']['t0']+30.0
        rep = coaching.interpret(obs, roles, TX.settings_at(st, 'A', 6), 'A', t0=t0, t1=t0+90.0)
        self.assertIn('short_range', [n['code'] for n in rep['overview']['notices']])
        for path, text in TX.texts_of(rep):
            for w in ('repeated', 'pattern', 'often', 'regularly', 'consistently'):
                self.assertNotRegex(text.lower(), r'\b'+w+r'\b', (path, text))

    def test_nothing_found_says_it_is_not_proof(self):
        rep = TX.report('possession_unknown', None, 2)
        self.assertEqual(rep['strengths']+rep['needsWork']+rep['reviews'], [])
        self.assertIn('not proof', rep['overview']['text'])

    def test_diagrams_finding_shape_and_movement(self):
        rep = TX.report('unsupported_advance', 'none_behind', 4)
        kinds = Counter(d['kind'] for d in rep['diagrams'])
        self.assertEqual(kinds['finding'], sum(1 for e in rep['keyMoments'] if e['diagram']))
        self.assertGreaterEqual(kinds['formation'], 1)
        self.assertGreaterEqual(kinds['movement'], 1)
        for d in rep['diagrams']:
            self.assertEqual(S.validate('diagram', d['diagram']), [])
            if d['kind'] == 'formation':
                self.assertIsNone(d['finding'])
                self.assertRegex(d['caption'], r'\((?:estimated|coach-marked|estimated, partly coach-marked)\)')
                self.assertIn('Described, not judged', d['caption'])
                self.assertTrue(all(el['type'] == 'player' and el['team'] == 'A' for el in d['diagram']['elements']))
            if d['kind'] == 'movement':
                self.assertIn(d['evidence'], {e['id'] for e in rep['keyMoments']})
                arrows = [el for el in d['diagram']['elements'] if el['type'] == 'arrow']
                self.assertTrue(arrows)
                for a in arrows:
                    (x0, y0), (x1, y1) = a['points']
                    moved = math.hypot((x1-x0)*105.0, (y1-y0)*68.0)
                    self.assertGreaterEqual(moved, report.MOVEMENT_MIN_M-1e-6)
                    self.assertTrue(a['id'].startswith('A-'), 'arrows only for our own outfield players')
        self.assertLessEqual(kinds['formation'], report.MAX_FORMATION_DIAGRAMS)

    def test_next_objective_from_the_match(self):
        rep = TX.report('unsupported_advance', 'none_behind', 4)
        no = rep['nextObjective']
        self.assertTrue(no['fromMatch'])
        self.assertEqual(no['finding'], rep['needsWork'][0]['id'])
        self.assertIn('From this match', no['text'])
        self.assertEqual(EX.validate_free(no['text'], rep['profile'], None, check_facts=False), [])

    def test_drills_belong_to_reported_findings(self):
        for scene, variant in V2_SCENES:
            for lvl in (2, 4, 6):
                rep = TX.report(scene, variant, lvl)
                ids = {f['id'] for f in TX.findings_of(rep)}
                keys = {e['id'] for e in rep['keyMoments']}
                for p in rep['drills']:
                    with self.subTest(scene=scene, level=lvl, drill=p['drill']):
                        self.assertTrue(p['optional'] or p['finding'] in ids)
                        self.assertTrue(set(p['fields']['moments']) <= keys)
                        if p['finding']:
                            f = next(x for x in TX.findings_of(rep) if x['id'] == p['finding'])
                            self.assertIn(p['drill'], f['drills'])
                            self.assertIn(f['verdict'], S.CRITICISM)

    def test_fast_enough(self):
        """DESIGN 1.2: a level change rebuilds only the interpretation tier, target < 300 ms (3-min clip); the
        scene is 6 minutes."""
        b, st, obs, roles = TX.tiers('unsupported_advance', 'none_behind')
        took = []
        for lvl in LEVELS:
            a = _time.perf_counter()
            coaching.interpret(obs, roles, TX.settings_at(st, 'A', lvl), 'A')
            took.append(_time.perf_counter()-a)
        self.assertLess(sorted(took)[len(took)//2], .3, took)


class V2AssemblyTests(unittest.TestCase):
    """build_report_v2 called directly: what it does with withheld findings, missing moments, no-alternative records
    and drill picks it cannot print."""

    @classmethod
    def setUpClass(cls):
        cls.b, cls.st, cls.obs, cls.roles = TX.tiers('unsupported_advance', 'none_behind')

    def parts(self, level=4):
        """The stage inputs interpret hands the report stage, captured with a stand-in report stage."""
        got = {}

        def capture(**kw):
            got.update(_copy.deepcopy(kw))
            return report.build_report_v2(**kw)
        coaching.interpret(self.obs, self.roles, TX.settings_at(self.st, 'A', level), 'A',
                           stages={'report': capture})
        return got

    def test_a_withheld_findings_moments_are_dropped(self):
        kw = self.parts()
        gone = kw['findings'][0]
        kw['findings'] = kw['findings'][1:]
        kw['withheld'] = kw['withheld']+[{'finding': gone['id'], 'reason': 'text_failed_validation'}]
        log = []
        rep = report.build_report_v2(log=log, **kw)
        gone_evs = {ts['evidence'] for ts in gone['timestamps']}
        still = {ts['evidence'] for f in TX.findings_of(rep) for ts in f['timestamps']}
        self.assertFalse((gone_evs-still) & {e['id'] for e in rep['keyMoments']})
        self.assertTrue(all(e['finding'] != gone['id'] for e in rep['keyMoments']))
        self.assertTrue(all(d['finding'] != gone['id'] for d in rep['diagrams']))
        self.assertEqual(S.validate('report', rep), [])

    def test_a_finding_whose_moment_was_not_made_is_withheld(self):
        kw = self.parts()
        f = kw['findings'][0]
        kw['evidence'] = [e for e in kw['evidence'] if e['id'] != f['timestamps'][0]['evidence']]
        rep = report.build_report_v2(**kw)
        self.assertIn({'finding': f['id'], 'reason': 'moment_missing'}, rep['withheld'])
        self.assertNotIn(f['id'], {x['id'] for x in TX.findings_of(rep)})
        self.assertEqual(S.validate('report', rep), [])

    def test_no_alternative_reason_reaches_the_work_item(self):
        kw = self.parts()
        work = [f for f in kw['findings'] if f['verdict'] == 'work_on']
        self.assertTrue(work)
        f = work[0]
        kw['alternatives'] = [a for a in kw['alternatives'] if a['finding'] != f['id']]
        for fx_ in kw['findings']:
            if fx_['id'] == f['id']:
                fx_['claims'] = [c for c in fx_['claims'] if c['type'] != 'counterfactual']
                fx_['text']['action'] = 'Keep one teammate in the left channel behind an advancing defender.'
        kw['no_alternatives'] = [{'finding': f['id'], 'evidence': None, 't': None, 'principle': 'cover_behind_'
                                  'advancing_defender', 'reason': 'no_reachable_fix', 'rejected': []}]
        rep = report.build_report_v2(**kw)
        got = next(x for x in rep['needsWork'] if x['id'] == f['id'])
        # the wording key: no_reachable_fix, or (fix round 1, ZONES B-8) its channel-opponent variant
        words = EX.NO_ALT_WORDS[EX.no_alt_facts(f, kw['no_alternatives'][0])['wording']][1]
        self.assertTrue(got['text']['action'].startswith(words), got['text'])
        self.assertTrue(any(c['type'] == 'limitation' and c['facts'].get('code') == 'no_alternative'
                            for c in got['claims']))
        self.assertEqual(EX.validate(got['text'], got['claims'], got, rep['profile']), [])

    def test_a_guardrail_review_gets_no_second_why(self):
        kw = self.parts(7)                      # level 7, style unknown: the wide-defender family is a review
        rev = [f for f in kw['findings'] if f['verdict'] == 'review']
        self.assertTrue(rev)
        f = rev[0]
        before = f['text']['action']
        kw['no_alternatives'] = [{'finding': f['id'], 'evidence': None, 't': None, 'principle': 'cover_behind_'
                                  'advancing_defender', 'reason': 'guardrail_failed', 'rejected': []}]
        rep = report.build_report_v2(**kw)
        self.assertEqual(next(x for x in rep['reviews'] if x['id'] == f['id'])['text']['action'], before)

    def test_drill_picks_the_report_cannot_print_are_left_out(self):
        # DETECT fix round 1 (critic B-1): level 1 judges "someone back" with the ball in the final third only; the
        # none_behind scene's centre-back carries it in midfield (no level-1 criticism), so the level-1 try-this comes
        # from the same advances played high (variant final_third).
        got = {}

        def capture(**kw_):
            got.update(_copy.deepcopy(kw_))
            return report.build_report_v2(**kw_)
        _, st, obs, roles = TX.tiers('unsupported_advance', 'final_third')
        coaching.interpret(obs, roles, TX.settings_at(st, 'A', 1), 'A', stages={'report': capture})
        kw = got
        f = next(x for x in kw['findings'] if x['verdict'] == 'try_this')
        ev = f['timestamps'][0]['evidence']
        good = {'finding': f['id'], 'drill': 'l1_help_your_teammate_2v2', 'optional': False,
                'fields': {'trains': 'Someone ready to protect our goal.', 'why': 'For this match\'s focus.',
                           'moments': [ev], 'easier': 'Play 2 v 1.', 'harder': 'Add a defender.',
                           'progress': 'Someone stays back when we attack.'}}
        numbers = _copy.deepcopy(good)
        numbers['fields']['why'] = 'It happened in 16 of 16 attacks.'              # no numbers at level 1
        unknown = _copy.deepcopy(good)
        unknown['fields']['moments'] = ['ev-000000000000']
        loose = _copy.deepcopy(good)
        loose['finding'] = None                                                     # not optional: refused
        kw['drills'] = [good, numbers, unknown, loose]
        log = []
        rep = report.build_report_v2(log=log, **kw)
        self.assertEqual(len(rep['drills']), 1)
        self.assertEqual(rep['drills'][0]['fields']['why'], good['fields']['why'])
        self.assertEqual(len([x for x in log if x[0] == 'drill']), 3)
        self.assertEqual(next(x for x in rep['needsWork'] if x['id'] == f['id'])['drills'],
                         ['l1_help_your_teammate_2v2'])

    def test_a_work_item_without_a_drill_says_why(self):
        """DRILLS' request: a work item left without a drill gets a notice (library, safety, or not printable)."""
        kw = self.parts(4)
        work = [f for f in kw['findings'] if f['verdict'] in S.CRITICISM]
        self.assertTrue(work)
        shown = report.build_report_v2(**_copy.deepcopy(kw))
        self.assertNotIn('no_drill', [n['code'] for n in shown['overview']['notices']])
        none = _copy.deepcopy(kw)
        none['drills'] = []
        rep = report.build_report_v2(**none)
        texts = [n['text'] for n in rep['overview']['notices'] if n['code'] == 'no_drill']
        self.assertEqual(len(texts), len(work))
        self.assertIn(work[0]['text']['short'].lower(), texts[0].lower())
        bad = _copy.deepcopy(kw)
        for p in bad['drills']:
            p['fields']['why'] = 'It happened in 99 of 99 moments.'
        rep = report.build_report_v2(**bad)
        texts = [n['text'] for n in rep['overview']['notices'] if n['code'] == 'no_drill']
        self.assertTrue(texts and 'could not be worded' in texts[0])

    def test_the_fixed_report_sentences_pass_every_level(self):
        """A fixed sentence the gate refuses would silently disappear (a 'never' did): every level, every one."""
        from soccer.coaching import profiles as PR
        for lvl in LEVELS:
            prof = PR.resolve({'playingLevel': lvl, 'ageGroup': PR.default_age(lvl)})
            texts = [w.format(short='someone back') for w in report.NO_DRILL_WORDS.values()]
            texts += list(report.DATA_QUALITY_HELP.values())+[report.UNAVAILABLE_REASON]
            for text in texts:
                with self.subTest(level=lvl, text=text):
                    facts = L.facts_for(extra={'advice': list(report.KEYFRAME_ADVICE)})
                    self.assertEqual(EX.validate_free(text, prof, facts, numbers=False, short=True), [])

    def test_next_objective_skips_a_withheld_top_item(self):
        kw = self.parts()
        top = kw['assessment']['nextObjective']['finding']
        self.assertIsNotNone(top)
        kw['findings'] = [f for f in kw['findings'] if f['id'] != top]
        rep = report.build_report_v2(**kw)
        no = rep['nextObjective']
        self.assertNotEqual(no['finding'], top)
        if rep['needsWork']:
            self.assertEqual(no['finding'], rep['needsWork'][0]['id'])
        else:
            # fix round 2: no other work item gives no objective from the match (critic S-8), never the first
            # unpraised check; polish round: the level's general learning objective, labelled, which names no check
            # (this stage cannot see what the match demonstrated), else the no-objective sentence
            self.assertFalse(no['fromMatch'])
            self.assertIsNone(no['skill'])
            self.assertEqual(no, report._no_objective(rep['profile']))
        kw['findings'] = [f for f in kw['findings'] if f['verdict'] not in S.CRITICISM]
        rep = report.build_report_v2(**kw)
        self.assertEqual(rep['nextObjective'], report._no_objective(rep['profile']))
        focus = AS.level_focus(rep['profile'], checks=False)
        self.assertEqual(rep['nextObjective']['text'], focus['text'] if focus else
                         report.no_objective_text(rep['profile']))
        self.assertEqual([p for p in rep['drills'] if p['optional']], [])     # no check named: no optional drill

    def test_the_data_quality_notice_names_the_cause_and_the_help(self):
        """critic B4: the dominant data-quality cause and what would help (keyframes, identity corrections, a wider
        view); nothing when no cause dominates."""
        kw = self.parts()
        for cause, words in (('registration', 'keyframes'), ('identity', 'identity corrections'),
                             ('out_of_view', 'wider camera view'), (None, None)):
            k2 = _copy.deepcopy(kw)
            k2['obs']['timeline']['dataQuality']['dominantCause'] = cause
            rep = report.build_report_v2(**k2)
            got = [n['text'] for n in rep['overview']['notices'] if n['code'] == 'data_quality']
            with self.subTest(cause=cause):
                if cause is None:
                    self.assertEqual(got, [])
                else:
                    self.assertEqual(len(got), 1)
                    self.assertIn(words, got[0])

    def test_the_overview_drops_a_name_its_level_refuses(self):
        """The synthesis is gated like any text: a finding name the level's register bans is left out of the
        overview (the rest stays), never printed."""
        kw = self.parts(4)
        prof = _copy.deepcopy(kw['profile'])
        prof['register']['banned'] = list(prof['register']['banned'])+['cover']
        strong = [f for f in kw['findings'] if f['verdict'] == 'strength']
        self.assertTrue(strong)
        for f in strong:
            f['text']['short'] = 'Pressure with cover'
        sections = {'strengths': strong, 'needsWork': [], 'reviews': [], 'described': []}
        text = report._overview_text(prof, kw['assessment'], sections, kw['obs'], kw['range'])
        self.assertIsNotNone(text)
        self.assertNotRegex(text.lower(), r'(?<![a-z-])cover(?![a-z-])')
        self.assertIn('checks for this level', text)
        ok = report._overview_text(kw['profile'], kw['assessment'], sections, kw['obs'], kw['range'])
        self.assertIn('pressure with cover', ok)

    def test_work_items_follow_their_priority(self):
        kw = self.parts(4)
        work = [f for f in kw['findings'] if f['verdict'] == 'work_on']
        self.assertTrue(work)
        third, first = _copy.deepcopy(work[0]), _copy.deepcopy(work[0])
        third['id'], first['id'] = 'fnd-'+'e'*12, 'fnd-'+'f'*12
        work[0]['priority'], third['priority'], first['priority'] = 2, 3, 1
        kw['findings'] = kw['findings']+[third, first]      # input order 2, 3, 1 (neither sorted nor reversed)
        rep = report.build_report_v2(**kw)
        self.assertEqual([f['priority'] for f in rep['needsWork']], [1, 2, 3])
        self.assertEqual(rep['needsWork'][0]['id'], first['id'])

    def test_movement_arrows_are_ours_and_outfield_only(self):
        kw = self.parts(4)
        f = next(x for x in kw['findings'] if x['verdict'] != 'describe' and x['timestamps'])
        ev = next(e for e in kw['evidence'] if e['id'] == f['timestamps'][0]['evidence'])
        m = _copy.deepcopy(next(x for x in self.obs['moments']+self.roles['moments'] if x['id'] == ev['id']))
        self.assertIsNotNone(m['before'])
        m['scene']['players']['B-77'] = {'team': 'B', 'x': .5, 'y': .5, 'kind': 'O', 'sigma': None,
                                         'goalkeeper': False, 'box': None}
        m['before']['players']['B-77'] = {'team': 'B', 'x': .3, 'y': .3, 'kind': 'O', 'sigma': None,
                                          'goalkeeper': False, 'box': None}
        m['scene']['players']['GK-1'] = {'team': f['team'], 'x': .05, 'y': .5, 'kind': 'O', 'sigma': None,
                                         'goalkeeper': True, 'box': None}
        m['before']['players']['GK-1'] = {'team': f['team'], 'x': .15, 'y': .5, 'kind': 'O', 'sigma': None,
                                          'goalkeeper': True, 'box': None}
        d = report.movement_diagram(f, ev, m, self.obs, kw['profile'])
        self.assertIsNotNone(d)
        ids = {el['id'] for el in d['diagram']['elements'] if el['type'] == 'arrow'}
        self.assertNotIn('B-77', ids)
        self.assertNotIn('GK-1', ids)
        self.assertIn('B-77', {el['id'] for el in d['diagram']['elements'] if el['type'] == 'player'})

    def test_sources_cover_the_claims_and_the_drills(self):
        """report.sources = the profile's + the principle claims' + the picked drills'. Each part is checked alone
        (the others emptied): on this scene the drill and the claims cite the same two sources, all in the profile."""
        from soccer.coaching import drills as D
        from soccer.coaching import profiles as PR
        lib = D.load_library()
        kw = self.parts(4)
        self.assertTrue(kw['drills'])
        claim_src = {PR.canonical_source(c['source']) for f in kw['findings'] for c in f['claims'] if c.get('source')}
        drill_src = {PR.canonical_source(r['source']) for p in kw['drills']
                     for r in D.get(lib, p['drill'])['source_reference']}
        self.assertTrue(claim_src and drill_src)
        full = report.build_report_v2(**_copy.deepcopy(kw))
        ids = [s['id'] for s in full['sources']]
        self.assertEqual(ids, sorted(set(ids)))
        self.assertTrue(claim_src | drill_src | {PR.canonical_source(x) for x in kw['profile']['sources']} >= set(ids))
        claims_only = _copy.deepcopy(kw)
        claims_only['profile']['sources'], claims_only['drills'] = [], []
        self.assertEqual({s['id'] for s in report.build_report_v2(**claims_only)['sources']}, claim_src)
        drills_only = _copy.deepcopy(kw)
        drills_only['profile']['sources'] = []
        for f in drills_only['findings']:
            for c in f['claims']:
                if c.get('source'):
                    c['tag'], c['source'] = S.ORIGIN_P, None
        self.assertEqual({s['id'] for s in report.build_report_v2(**drills_only)['sources']}, drill_src)


class V2FixRoundOneTests(unittest.TestCase):
    """Fix round 1 (critic B-11 / N6, FACTS L11a, COACH L2 / L3): what the report itself now says."""

    def one_sided_obs(self, marked_own=True):
        b, st, obs, roles = TX.tiers('amateur_coach_spans', 'spans')
        obs = _copy.deepcopy(obs)
        pq = obs['timeline']['phaseQuality']
        pq['available'] = False
        pq['roughShares'] = None
        coach = dict(pq.get('coach') or {})
        coach.update(own=11.5 if marked_own else 0.0, opp=0.0 if marked_own else 8.0)
        pq['coach'] = coach
        return obs, roles, st

    def test_one_sided_coach_marks_give_no_share_and_say_why(self):
        """critic N6: Team A's two spells marked on the AM copy read "A 1.0, B 0.0"."""
        for own in (True, False):
            obs, roles, st = self.one_sided_obs(own)
            po = report._possession(obs)
            with self.subTest(marked_own=own):
                self.assertIsNone(po['roughShares'])
                self.assertEqual(po['source'], 'coach')
                self.assertIn('no share is given', po['caveat'])
                prof = PR_resolve(st, 4)
                notes = report._report_notices(obs, roles, prof, {'t0': 0.0, 't1': 1e6}, False)
                note = [n for n in notes if n['code'] == 'coach_marks_one_side']
                self.assertEqual(len(note), 1)
                marked = obs['team'] if own else ('B' if obs['team'] == 'A' else 'A')
                self.assertIn(f'only Team {marked}', note[0]['text'])
        b, st, obs, roles = TX.tiers('amateur_coach_spans', 'spans')     # both teams marked: a share, coach source
        po = report._possession(obs)
        self.assertEqual(po['source'], 'coach')
        self.assertIsNotNone(po['roughShares'])
        rep = TX.report('unsupported_advance', 'none_behind', 4)
        self.assertEqual(rep['overview']['possession']['source'], 'engine')
        self.assertEqual(S.validate('report', rep), [])

    def test_a_range_just_under_three_minutes_says_so(self):
        """FACTS L11a: 179.97 s read "This report covers 3.0 minutes" beside "shorter than three minutes"."""
        self.assertEqual(report.duration_words(179.967), 'just under 3 minutes')
        self.assertEqual(report.duration_words(180.0), '3.0 minutes')
        self.assertEqual(report.duration_words(365.0), '6.1 minutes')
        self.assertEqual(report.duration_words(59.0), '59 s')

    def test_team_level_challenges_circle_nobody(self):
        """COACH L2: levels 1-2 put the review outline on the advancing child; it is the support style now, and level
        4's evidence-only naming keeps the review ring."""
        for lvl, ring in ((1, False), (2, False), (4, True)):
            rep = TX.report('unsupported_advance', 'final_third', lvl)
            crit = {f['id'] for f in rep['needsWork']+rep['reviews']}
            moments = [e for e in rep['keyMoments'] if e['finding'] in crit]
            with self.subTest(level=lvl):
                self.assertTrue(moments)
                rings = [el for e in moments for spec in (e.get('diagram'), e.get('videoOverlay')) if spec
                         for el in spec['elements'] if el.get('type') == 'focus' and el.get('tone') == 'review']
                self.assertEqual(bool(rings), ring)
                self.assertEqual(S.validate('report', rep), [])

    def test_rest_defence_is_defined_once_wherever_the_report_uses_it(self):
        """COACH L3: 77 level-6 reports used "rest defence" without its PitchIQ definition (a headline, the next
        objective "Rest-defence margin (PitchIQ)", a drill's "Also:" line)."""
        rx = re.compile(r'\brest[- ]defen[cs]e\b', re.I)
        # fix round 3 (FACTS F-1): the definition carries the run's margin words; its start is said once
        definition = EX.DEFINITION_START['rest_defence']
        for scene, variant in (('box_edge', 'box'), ('unsupported_advance', 'none_behind'),
                               ('amateur_coach_spans', 'no_spans')):
            for lvl in (6, 7):
                rep = TX.report(scene, variant, lvl)
                allt = ' || '.join([rep['overview']['text'] or '', rep['nextObjective']['text']] +
                                   [v for f in TX.findings_of(rep) for v in f['text'].values() if v] +
                                   [v for p in rep['drills'] for v in p['fields'].values() if isinstance(v, str)])
                with self.subTest(scene=scene, level=lvl):
                    if rx.search(allt):
                        self.assertEqual(allt.count(definition), 1, allt)

    def test_the_no_drill_words_cover_the_format_reason(self):
        self.assertIn('no_drill_for_format', report.NO_DRILL_WORDS)


class V2MomentsTests(unittest.TestCase):
    """Fix round 2 (LEAD decision (b)): a 'moments worth a look' review sits in section 3 (WHAT NEEDS WORK) as its own
    part after the work items and the possible issues (REVIEW_PARTS: the page and the PDF print each part under its
    title), is never counted or named as a possible issue in the overview, gets no drill and no zone sentence; with no
    next objective the report says what that means (critic S-8 / ASSESS fix round 2)."""

    @classmethod
    def setUpClass(cls):
        cls.b, cls.st, cls.obs, cls.roles = TX.tiers('unsupported_advance', 'none_behind')

    def parts(self, level):
        got = {}

        def capture(**kw):
            got.update(_copy.deepcopy(kw))
            return report.build_report_v2(**kw)
        coaching.interpret(self.obs, self.roles, TX.settings_at(self.st, 'A', level), 'A', stages={'report': capture})
        return got

    def with_moments(self, level):
        """The report-stage inputs of the scene at a level with its first failing finding turned into a 'moments
        worth a look' review as assess fix round 2 makes one (the counts, no priority, every guardrail but evidence;
        one of its listed opportunities unclear) and worded by explain; its drill picks removed (drills attach to
        work items only)."""
        kw = self.parts(level)
        f0 = next(f for f in kw['findings'] if f['verdict'] in EX.FAILURES)
        f = _copy.deepcopy(f0)
        n = len(f['timestamps'])
        decided = len(f['opportunities'])-1
        self.assertGreater(decided, n)
        f.update(verdict='review', priority=None, limitations=[EX.MOMENTS], text=None, claims=[], drills=[])
        f['guardrails'] = {k: 'pass' for k in S.GUARDRAILS}
        f['guardrails']['evidence'] = 'fail'
        f['measurements'].update(clearMoments=n, clearSpells=n, decidedMoments=decided, failingMoments=n)
        f['aggregate'].update(opportunities=decided+1, demonstrated=decided-n, notDemonstrated=n, unclear=1,
                              distinctFailingSpells=n)
        out = EX.explain([f], kw['profile'], obs=kw['obs'], roles=kw['roles'])
        self.assertEqual(out['withheld'], [])
        kw['findings'] = [out['findings'][0] if x['id'] == f0['id'] else x for x in kw['findings']]
        kw['drills'] = [p for p in kw['drills'] if p.get('finding') != f0['id']]
        if kw['assessment']['nextObjective'].get('finding') == f0['id']:
            kw['assessment']['nextObjective'] = {'skill': None, 'text': '', 'fromMatch': False, 'finding': None}
        return kw, f0['id']

    def test_its_part_in_what_needs_work_and_the_overview(self):
        for lvl in (3, 4, 7):
            kw, fid = self.with_moments(lvl)
            rep = report.build_report_v2(**kw)
            with self.subTest(level=lvl):
                self.assertEqual(S.validate('report', rep), [])
                self.assertEqual(S.cross_check(rep, kw['obs'], kw['roles']), [])
                f = next(x for x in rep['reviews'] if x['id'] == fid)
                self.assertNotIn(fid, {x['id'] for x in rep['needsWork']})
                self.assertEqual(report.review_part(f), 'moments')
                parts = [p['part'] for p in report.REVIEW_PARTS]
                self.assertEqual([parts.index(report.review_part(x)) for x in rep['reviews']],
                                 sorted(parts.index(report.review_part(x)) for x in rep['reviews']))
                got = report.reviews_by_part(rep)
                self.assertEqual(got[-1][0]['title'], 'Moments worth a look')
                self.assertEqual([x['id'] for x in got[-1][1]], [fid])
                self.assertEqual(f['drills'], [])
                self.assertNotIn(fid, {p['finding'] for p in rep['drills']})
                self.assertEqual(EX.moments_problems(f), [])
                ov = rep['overview']['text']
                self.assertIn('Moments worth a look, not a verdict: '+f['text']['short'][:1].lower(), ov)
                n_issues = len([x for x in rep['reviews'] if report.review_part(x) == 'issues'])
                if not n_issues:
                    self.assertNotIn('possible issue', ov)
                for e in rep['keyMoments']:
                    if e['finding'] == fid:
                        self.assertEqual(e['caption'], f['text']['short'])

    def test_the_part_texts_pass_every_level(self):
        from soccer.coaching import profiles as PR
        for lvl in LEVELS:
            prof = PR.resolve({'playingLevel': lvl, 'ageGroup': PR.default_age(lvl)})
            for part in report.REVIEW_PARTS:
                for text in (part['title'], part['intro']):
                    with self.subTest(level=lvl, text=text):
                        self.assertEqual(EX.validate_free(text, prof, None, numbers=False, short=True), [])
            with self.subTest(level=lvl, text='no objective'):
                self.assertEqual(report.no_objective_text(prof), report._at(report.NO_OBJECTIVE_WORDS, lvl))

    def test_no_next_objective_is_said_plainly(self):
        """ASSESS fix round 2 (critic S-8): 182 of the 196 run-copy reports have no next objective; the text says what
        the rule found, not "No next objective could be chosen from this level's checks"."""
        for lvl in (1, 2, 4, 6, 7):
            kw = self.parts(lvl)
            kw['assessment']['nextObjective'] = {'skill': None, 'text': 'No next objective could be chosen from this '
                                                                       'level\'s checks.', 'fromMatch': False,
                                                 'finding': None}
            kw['findings'] = [f for f in kw['findings'] if f['verdict'] not in S.CRITICISM]
            rep = report.build_report_v2(**kw)
            no = rep['nextObjective']
            focus = AS.level_focus(rep['profile'], checks=False)
            with self.subTest(level=lvl):
                self.assertIsNone(no['skill'])
                # polish round: the level's general learning objective when it has one (levels 1-6 here), else the
                # sentence of fix round 2 (level 7 has none a team can take: its objectives are coach education)
                self.assertEqual(no['text'], focus['text'] if focus else
                                 report.NO_OBJECTIVE_WORDS[1 if lvl <= 2 else 3])
                self.assertEqual(focus is None, lvl == 7)
                self.assertNotIn('could be chosen', no['text'])
                self.assertEqual(S.validate('report', rep), [])


def PR_resolve(st, level):
    from soccer.coaching import profiles as PR
    return PR.resolve(TX.settings_at(st, 'A', level)['teams']['A'], team='A')


class V2ForwardingTests(unittest.TestCase):
    """The v1 report moved to report_v1.py (served until wave 7): the old names still work through report.py, and
    the v2 modules never import the v1 code (DESIGN 11.2)."""

    def test_v1_names_are_forwarded(self):
        self.assertIs(report.build_report, report_v1.build_report)
        self.assertEqual(report.SECTIONS, report_v1.SECTIONS)
        self.assertEqual(report.MAX_PRIORITIES, report_v1.MAX_PRIORITIES)
        self.assertEqual(report.language.DEFAULT_LEVEL, 'competitive')      # __init__.build_report_for reads it
        with self.assertRaises(AttributeError):
            report.no_such_name

    def test_v2_modules_never_import_the_v1_words(self):
        root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'soccer', 'coaching')
        for name in ('report.py', 'explain.py', 'language.py'):
            with open(os.path.join(root, name), encoding='ascii') as fh:
                src = fh.read()
            imports = [ln for ln in src.splitlines() if re.match(r'\s*(?:from|import)\s', ln)]
            self.assertFalse([ln for ln in imports if 'language_v1' in ln or 'report_v1' in ln], (name, imports))
        self.assertNotIn('report_v1', [p.split('/')[-1][:-3] for p in coaching.code_files('interpretation')])
        self.assertNotIn('language_v1', [p.split('/')[-1][:-3] for p in coaching.code_files('interpretation')])


class V2FixRoundThreeTests(unittest.TestCase):
    """Fix round 3 (Oct 24; critic S-3 / B-3, FACTS F-9, COACH F-4, F-11, F-12): the next objective says where it
    comes from, never asks for what the declared style accepts, starts with a capital; the overview's rest-defence
    definition carries the run's margin; report.text_problems (for coaching_eval.report_problems, game-3 G1+)
    refuses each old form. On the COACH verifier's 546 scene reports the S-8 objective read "From the level's
    objectives, not from this match" 209 times (0 after); the compact-style far-side objective 1 (0 after)."""

    def test_the_objective_from_this_match_s_checks_says_so(self):
        rep = TX.report('small_field_scale', 'declared', 4)
        no = rep['nextObjective']
        self.assertEqual((no['skill'], no['fromMatch']), ('back_line_coordination', False))
        self.assertTrue(no['text'].startswith(report.S8_LABEL), no['text'])
        self.assertNotIn('not from this match', no['text'])
        self.assertEqual(report.text_problems(rep), [])
        self.assertEqual(EX.validate_free(no['text'], rep['profile'], None, numbers=False, limit=400), [])
        old = _copy.deepcopy(rep)
        old['nextObjective']['text'] = "From the level's objectives, not from this match: Back-line coordination."
        self.assertEqual(len(report.text_problems(old)), 2)

    def test_no_objective_on_what_the_style_accepts(self):
        """COACH F-12: defending/weak_empty at level 7, compact: "Because the selected style is compact, open space
        on the far side is accepted", then "Defensive balance (far side)" as the next objective."""
        rep = TX.report('defending', 'weak_empty', 7, age='Adult', style='compact')
        d2 = [f for f in TX.findings_of(rep) if f['observation'] == 'weak_side_balance']
        self.assertTrue(any(EX.STYLE_EFFECT['compact_far_side_accepted'] in f['text']['explanation'] for f in d2))
        self.assertNotEqual(rep['nextObjective']['skill'], 'far_side_protection')
        self.assertEqual(report.style_accepted(TX.findings_of(rep), 'weak_side_balance'), 'compact')
        self.assertEqual(report.text_problems(rep), [])
        old = _copy.deepcopy(rep)
        old['nextObjective'] = {'skill': 'far_side_protection', 'fromMatch': False, 'finding': None,
                                'text': report.S8_LABEL+'Defensive balance (far side).'}
        self.assertTrue(any('style accepts' in p for p in report.text_problems(old)))

    def test_an_objective_from_a_work_item_starts_with_a_capital(self):
        rep = TX.report('channel_threat', 'deep', 2)
        no = rep['nextObjective']
        self.assertTrue(no['fromMatch'])
        self.assertTrue(no['text'][:1].isupper(), no['text'])
        old = _copy.deepcopy(rep)
        old['nextObjective']['text'] = no['text'][:1].lower()+no['text'][1:]
        self.assertTrue(report.text_problems(old))

    def test_the_overview_defines_rest_defence_with_the_margin(self):
        rep = TX.report('box_edge', 'box', 7)
        self.assertIn(EX.definition_text(EX.DEFINITIONS[1][3], 'at least 10 m'), rep['overview']['text'])
        self.assertEqual(report.margin_words({'timeline': {'scale': {'lengthM': 105.0, 'metresUsable': True}}},
                                             rep['profile']), 'at least 10 m')
        self.assertEqual(report.margin_words({'timeline': {'scale': {'lengthM': 105.0, 'metresUsable': False}}},
                                             rep['profile']), 'well')

    def test_professional_is_never_said_to_a_youth_team(self):
        rep = TX.report('defending', 'weak_empty', 5)
        self.assertEqual(report.text_problems(rep), [])
        old = _copy.deepcopy(rep)
        f = TX.findings_of(old)[0]
        f['text']['explanation'] += ' Professional teams do it too.'
        self.assertTrue(any('professional' in p for p in report.text_problems(old)))

    def test_the_far_below_drill_reason_is_worded(self):
        """DRILLS fix round 3: drills.no_drill_detail's 'only_far_below' (the only fitting drills are much younger
        levels') had no words and read as 'no_drill_at_level'."""
        for lvl in (4, 7):
            text = report.NO_DRILL_WORDS['only_far_below'].format(short='spreading the attack')
            prof = PR_resolve(TX.tiers('box_edge', 'box')[1], lvl)
            self.assertEqual(EX.validate_free(text, prof, None, numbers=False, limit=400), [], lvl)


if __name__ == '__main__':
    unittest.main()
