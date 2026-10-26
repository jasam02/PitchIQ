"""The contextual coaching assessment (soccer/coaching/assess.py, DESIGN v2 4.8-4.9): the level's verdicts on the
measured opportunities, with the spec Part 15 guardrails, from the resolved coaching profile (profiles.py).

Why these checks. Every rule of DESIGN 4.8 / 4.9 that could put a wrong sentence in front of a coach is a test on
hand-built opportunities whose outcome is known: a work item needs the level's opportunities, distinct failing spells
(10 s apart; 3 under a 3-minute range), failure share and relevant phase seconds; a strength needs demonstrated spells
(never the absence of failures: critic B1); a failed guardrail turns a criticism into a review (a window role, a
formation mismatch without a coach role, an identity crossing at the evidence moment); a declared style can explain
a pattern (describe); the level's cap keeps the most relevant work items; level 1-2 never say "work on", level 4+
never "try this", level 7 without a style only reviews; level 1 reframes an unsupported advance as "someone ready to
protect our goal". Then DESIGN 13.1's scenes end to end (EVAL's synthetic results through the real measurement and
role tiers): no criticism of covered advances, the planted unsupported ones found at the right levels with the same
evidence across levels, one instance only a review, identity crossings and wrong formations at most reviews, no
criticism near a formation change or of a sprint back, nothing on a 60 s clip or a professional-like team, no phase
without possession or coach marks, the planted defending outcomes; the observations are never changed by a profile
(invariant 2) and a level change costs milliseconds.

Run:  python -m unittest test_coaching_assess          (from this directory)
"""
import copy
import hashlib
import json
import time
import unittest

import coaching_detect_fixtures as FX
from soccer.coaching import assess as AS, profiles as PR, schema as S  # noqa: E402

STEP = .2
SCALE = {'lengthM': 105.0, 'widthM': 68.0, 'source': 'assumed', 'metresUsable': True}


def profile(level, style=None, age=None, players=11):
    block = {'playingLevel': level, 'ageGroup': age or PR.default_age(level),
             'format': {'players': players, 'goalkeeper': players != 4}}
    if style:
        block['style'] = style
    return PR.resolve(block, team='A')


def _phase(state='opp', source='possession-4'):
    return {'state': state, 'source': source, 'estimated': source == 'possession-4', 'spanSeconds': 12.0,
            'nearChange': False}


def make_opp(detector, t0, relations, n=15, *, state=None, variant=None, primary=('A-03',), roles=None,
             identity='high', visibility='complete', intent=None, tier=None):
    """One opportunity of n samples from t0 with the given per-sample relations (scalars are repeated)."""
    t = [round(t0+STEP*k, 3) for k in range(n)]
    rel = {'t': t}
    for k, v in relations.items():
        rel[k] = list(v) if isinstance(v, (list, tuple)) else [v]*n
    t1 = t[-1]
    key = S.opportunity_key(detector, 'A', t0, t1, variant)
    state = state or S.DETECTOR_PHASES[detector][0]
    o = {'id': S.record_id('opp', key), 'key': key, 'detector': detector, 'team': 'A',
         'tier': tier or S.DETECTOR_TIERS[detector][0], 't0': t0, 't1': t1, 'frame0': int(t0*25),
         'frame1': int(t1*25), 'phase': _phase(state),
         'ball': {'zone': 'own_third', 'channel': 'left', 'reliable': True},
         'actors': {'primary': list(primary), 'supporting': [], 'opponents': [], 'carrier': None},
         'roles': roles or {}, 'relations': rel,
         'quality': {'identity': identity, 'position': 'high', 'visibility': visibility, 'ball': 'reliable',
                     'samples': n},
         'intentSignals': intent or {'recovering': False, 'interchange': None, 'formationChangeNear': False},
         'abstain': None}
    if variant:
        o['variant'] = variant
    return o


def moment_of(o, sample=None):
    k = len(o['relations']['t'])//2 if sample is None else sample
    t = o['relations']['t'][k]
    frame = int(round(t*25))
    return {'id': S.record_id('ev', o['id'], k), 'opportunity': o['id'], 'detector': o['detector'], 'team': 'A',
            'sample': k, 't': t, 'frame': frame, 'seek': S.seek_of(frame, 25.0), 'fps': 25.0, 'tier': 'world',
            'calibration': None, 'players': [], 'flagged': [],
            'scene': {'t': t, 'frame': frame, 'players': {}, 'ball': None, 'ballReliable': False}, 'before': None,
            'image': None}


def files(opps, t0=0.0, t1=600.0, own=(0.0, 300.0), opp=(300.0, 600.0), rolebook=None, coverage=None):
    """(obs, roles) holding the opportunities (measurement tier in obs, role tier in roles) with a moment each and a
    timeline whose own / opp spans give the relevant phase seconds."""
    meas = [o for o in opps if o['tier'] == 'measurement']
    role = [o for o in opps if o['tier'] == 'role']
    spans = []
    if own:
        spans.append({'state': 'own', 't0': own[0], 't1': own[1]-STEP, 'frame0': 0, 'frame1': 1,
                      'source': 'possession-4'})
    if opp:
        spans.append({'state': 'opp', 't0': opp[0], 't1': opp[1]-STEP, 'frame0': 0, 'frame1': 1,
                      'source': 'possession-4'})
    timeline = {'team': 'A', 'step': STEP, 't0': t0, 't1': t1, 'format': '11v11', 'scale': dict(SCALE),
                'phaseSpans': spans, 'phaseQuality': {'coverage': .9}}
    cov = coverage or {}
    obs = {'team': 'A', 'segment': {'t0': t0, 't1': t1, 'fps': 25.0, 'frames': 1}, 'timeline': timeline,
           'opportunities': meas, 'moments': [moment_of(o) for o in meas],
           'detectors': {d: c for d, c in cov.items() if d != 'wide_defender_advance'}}
    roles = {'team': 'A', 'opportunities': role, 'moments': [moment_of(o) for o in role],
             'detectors': {d: c for d, c in cov.items() if d == 'wide_defender_advance'},
             'rolebook': rolebook or {'team': 'A', 'format': '11v11', 'model': 'team_shape', 'records': [],
                                      'interchanges': [], 'formationChecks': []}}
    return obs, roles


def weak(t0, failing):
    return make_opp('weak_side_balance', t0, {'weakCount': 0 if failing else 2, 'centralCount': 2,
                                              'openWidthM': 30.0, 'primaryNamed': True})


def advance(t0, covered=False, source='stable', confidence='high', named=True, opponents=1, behind=1,
            fcn=False, ball_x=80.0, cb_shift=0.5, identity='high'):
    roles = {'A-02': {'role': 'LB', 'source': source, 'confidence': confidence}}
    return make_opp('wide_defender_advance', t0, {
        'advanceM': 20.0, 'beyondHalfM': 5.0, 'channelOccupied': covered, 'channelBy': None, 'dmDropped': None,
        'cbAdjusted': cb_shift >= 4.0, 'nearCb': 'A-05', 'nearCbShiftM': cb_shift, 'behindN': behind,
        'interchange': False, 'interchangeCover': False, 'interchangeBy': None,
        'opponentsInChannel': opponents, 'threatN': None, 'restMargin': None, 'dmCbSpacingM': None, 'wdVelX': 0.5,
        'complete': True, 'relaxed': False, 'primaryNamed': named, 'wdSource': source, 'wdConfidence': confidence,
        'ballX': ball_x},
        variant='left', primary=('A-02',), roles=roles, identity=identity,
        intent={'recovering': False, 'interchange': None, 'formationChangeNear': fcn})


def spaced(n, start=10.0, every=20.0):
    return [start+every*k for k in range(n)]


class EvidenceTable(unittest.TestCase):
    def test_a_repeated_failure_is_a_work_item(self):
        opps = [weak(t, k % 2 == 0) for k, t in enumerate(spaced(12, 300.0))]
        obs, roles = files(opps)
        out = AS.assess(obs, roles, profile(4))
        f = FX.findings_of(out, 'weak_side_balance')
        self.assertEqual(len(f), 1)
        f = f[0]
        self.assertEqual(f['verdict'], 'work_on')
        self.assertEqual(S.validate('finding', f), [])
        a = f['aggregate']
        self.assertEqual((a['opportunities'], a['notDemonstrated'], a['demonstrated']), (12, 6, 6))
        self.assertEqual(a['distinctFailingSpells'], 6)
        self.assertEqual(out['priorities'], [f['id']])
        self.assertEqual(f['priority'], 1)
        self.assertEqual(out['nextObjective'], {'skill': 'far_side_protection', 'text': out['nextObjective']['text'],
                                                'fromMatch': True, 'finding': f['id']})
        self.assertEqual(f['guardrails']['identity'], 'not_applicable')     # team-level: nobody named
        self.assertEqual(f['measurements']['weakCount'], 0)
        self.assertEqual(f['measurements']['metres'], 'estimated')
        self.assertEqual(len(f['timestamps']), 3)
        self.assertTrue(all(t['opportunity'] in [o['id'] for o in opps if o['relations']['weakCount'][0] == 0]
                            for t in f['timestamps']))

    def test_too_little_phase_time_or_too_few_distinct_spells_is_a_review(self):
        opps = [weak(t, k % 2 == 0) for k, t in enumerate(spaced(12, 300.0))]
        obs, roles = files(opps, opp=(300.0, 360.0))                 # 60 s of the phase < 90 s at level 4
        f = FX.findings_of(AS.assess(obs, roles, profile(4)), 'weak_side_balance')[0]
        self.assertEqual((f['verdict'], f['guardrails']['evidence']), ('review', 'fail'))
        clustered = [weak(300.0+3.2*k, k < 6) for k in range(12)]   # the six failures within 16 s: 2 distinct spells
        obs, roles = files(clustered)
        f = FX.findings_of(AS.assess(obs, roles, profile(4)), 'weak_side_balance')[0]
        self.assertEqual(f['aggregate']['distinctFailingSpells'], 2)
        self.assertEqual(f['verdict'], 'work_on')                    # level 4 needs 2
        f = FX.findings_of(AS.assess(obs, roles, profile(5)), 'weak_side_balance')[0]
        self.assertEqual(f['verdict'], 'work_on')
        f = FX.findings_of(AS.assess(obs, roles, profile(3)), 'weak_side_balance')
        self.assertEqual([x['verdict'] for x in f], ['review'])      # level 3: review is its criticism

    def test_a_short_range_needs_three_distinct_spells(self):
        # 5 of 12 fail (.42 >= level 4's .4) in 2 distinct spells (level 4 needs 2), 170 s of phase (>= 90 s)
        opps = [weak(t, k < 5) for k, t in enumerate([300.0, 303.2, 306.4, 312.0, 315.2, 320.0, 330.0, 340.0, 350.0,
                                                       360.0, 370.0, 380.0])]
        obs, roles = files(opps)
        f = FX.findings_of(AS.assess(obs, roles, profile(4)), 'weak_side_balance')[0]
        self.assertEqual((f['verdict'], f['aggregate']['distinctFailingSpells']), ('work_on', 2))   # a long range
        out = AS.assess(obs, roles, profile(4), t0=300.0, t1=470.0)    # 170 s: under 3 minutes
        f = FX.findings_of(out, 'weak_side_balance')[0]
        self.assertEqual((f['verdict'], f['aggregate']['distinctFailingSpells']), ('review', 2))
        self.assertIn('short_range', f['limitations'])

    def test_a_failure_must_last_the_minimum_spell(self):
        p = profile(4)['detectors']['weak_side_balance']['params']
        short = make_opp('weak_side_balance', 300.0, {'weakCount': 0}, n=8)       # 1.6 s < failMinS 2 s
        long = make_opp('weak_side_balance', 320.0, {'weakCount': 0}, n=15)       # 3.0 s
        self.assertEqual(AS.classify(short, p, SCALE, STEP)['outcome'], 'unclear')
        self.assertEqual(AS.classify(long, p, SCALE, STEP)['outcome'], 'not_demonstrated')
        # a demonstration needs a run as long as a failure needs (critic B-2): 1.6 s is unclear, 2.0 s demonstrated
        self.assertEqual(AS.classify(make_opp('weak_side_balance', 340.0, {'weakCount': 2}, n=8), p, SCALE, STEP)
                         ['outcome'], 'unclear')
        c = AS.classify(make_opp('weak_side_balance', 360.0, {'weakCount': 2}, n=10), p, SCALE, STEP)
        self.assertEqual((c['outcome'], c['demoRun'], c['judgedSeconds']), ('demonstrated', 2.0, 2.0))

    def test_a_strength_needs_judged_runs_of_the_spell_minimum(self):
        """B-2 (FACTS H1): a behind-the-ball opportunity whose judged samples (the ball in the final third) last 0.2-
        2.8 s demonstrates nothing, however many of them there are; the strength needs 3 s runs at level 4."""
        def d4(t0, judged):
            bx = [80.0 if k < judged else 50.0 for k in range(40)]
            return make_opp('behind_ball', t0, {'behindN': 5, 'restMargin': None, 'threatN': None, 'ballX': bx},
                            n=40, primary=('A-05',))
        short = [d4(t, 14) for t in spaced(6, 300.0, 20.0)]        # 2.8 s judged each
        obs, roles = files(short, own=(300.0, 600.0))
        out = AS.assess(obs, roles, profile(4))
        self.assertEqual(FX.findings_of(out, 'behind_ball'), [])
        longer = [d4(t, 16) for t in spaced(6, 300.0, 20.0)]       # 3.2 s
        obs, roles = files(longer, own=(300.0, 600.0))
        f = FX.findings_of(AS.assess(obs, roles, profile(4)), 'behind_ball')
        self.assertEqual([x['verdict'] for x in f], ['strength'])
        self.assertEqual(f[0]['measurements']['judgedSeconds'], round(6*16*STEP, 2))     # the 6 cited

    def test_the_strength_share_is_over_decided_opportunities(self):
        """B-2 / S-15 (CODE M4 survivor 'strength_share_ignored'): enough demonstrated spells, but a demonstrated share
        of the decided opportunities below the level's strengthShare is no strength; unclear ones count for neither."""
        good = [weak(t, False) for t in spaced(6, 300.0)]
        bad = [weak(t, True) for t in spaced(5, 302.0, 20.0)]      # 6 / 11 = .55 < level 4's .65
        obs, roles = files(good+bad)
        out = AS.assess(obs, roles, profile(4))
        self.assertNotIn('strength', [f['verdict'] for f in FX.findings_of(out, 'weak_side_balance')])
        unclear = [make_opp('weak_side_balance', t, {'weakCount': [0, 2]*7+[2], 'primaryNamed': True})
                   for t in spaced(10, 303.0, 20.0)]
        obs, roles = files(good+unclear)                           # 6 demonstrated, 0 failing, 10 unclear
        f = FX.findings_of(AS.assess(obs, roles, profile(4)), 'weak_side_balance')
        self.assertEqual([x['verdict'] for x in f], ['strength'])
        # The share alone decides (the mutant above survived both cases: the first is a work item, the second has
        # no failure): 6 demonstrated, 4 failing, 10 unclear is not failing (4 / 20 = .2 < .4) and, with a base of
        # .05 (level 4's praise base 1 - .95 x .8 = .24: 4 of 10 has a tail of .22), not mixed; 6 / 10 = .6 < .65 is
        # no strength, 7 / 10 = .7 is. The 4 clear spells of 10 against .05 (tail .001) are 'moments worth a look'.
        prof = profile(4)
        prof['detectors']['weak_side_balance']['params']['baseFailShare'] = .05
        four = [weak(t, True) for t in spaced(4, 302.0, 20.0)]
        obs, roles = files(good+four+unclear)
        # (fix round 2) the 4 clear failures of the 10 decided are 'moments worth a look', never a strength; with the
        # review turned off the family says nothing
        f = FX.findings_of(AS.assess(obs, roles, prof), 'weak_side_balance')
        self.assertEqual([(x['verdict'], AS.MOMENTS in x['limitations']) for x in f], [('review', True)])
        off = copy.deepcopy(prof)
        off['detectors']['weak_side_balance']['params']['momentsFromLevel'] = 99
        self.assertEqual(FX.findings_of(AS.assess(obs, roles, off), 'weak_side_balance'), [])
        obs, roles = files(good+[weak(500.0, False)]+four[:3]+unclear)
        f = FX.findings_of(AS.assess(obs, roles, prof), 'weak_side_balance')
        self.assertEqual([x['verdict'] for x in f], ['strength'])

    def test_no_bare_strength_beside_failures_significantly_above_the_base(self):
        """B-2 (STATS 7) and fix round 3 (LEAD decision (3), COACH F-5): failures SIGNIFICANTLY above the level's
        base share (one-sided binomial tail at baseTestAlpha .05) leave the family described ('mixed_evidence'),
        never praised; the base is the professional one from level 5 and scaled below it (baseSuccessScale)."""
        good = [weak(t, False) for t in spaced(10, 300.0)]
        bad = [weak(t, True) for t in spaced(2, 302.0, 20.0)]
        obs, roles = files(good+bad)                # 2 failing of 12 decided
        prof = profile(5)
        self.assertEqual(prof['detectors']['weak_side_balance']['params']['baseFailShare'], .004)
        f = FX.findings_of(AS.assess(obs, roles, prof), 'weak_side_balance')
        self.assertEqual([(x['verdict'], 'mixed_evidence' in x['limitations']) for x in f], [('describe', True)])
        self.assertEqual((f[0]['measurements']['baseShare'], f[0]['measurements']['baseTail']),
                         (.004, round(AS.binom_tail(2, 12, .004), 4)))
        prof['detectors']['weak_side_balance']['params']['baseFailShare'] = .1     # P(X >= 2 | 12, .1) = .34
        f = FX.findings_of(AS.assess(obs, roles, prof), 'weak_side_balance')
        self.assertEqual([x['verdict'] for x in f], ['strength'])
        # level 4 (U14): the same 2 of 12 against the scaled base 1 - .996 x .8 = .203 is praised
        f = FX.findings_of(AS.assess(obs, roles, profile(4)), 'weak_side_balance')
        self.assertEqual([x['verdict'] for x in f], ['strength'])
        # BC180 Espanyol's far side, 10 of 11 (critic S-1 acceptance): praised at level 4, withheld at level 5 where
        # one lapse against the professional .004 has a tail of .043
        obs, roles = files(good+[weak(500.0, True)])
        self.assertEqual([x['verdict'] for x in FX.findings_of(AS.assess(obs, roles, profile(4)), 'weak_side_balance')],
                         ['strength'])
        self.assertEqual([x['verdict'] for x in FX.findings_of(AS.assess(obs, roles, profile(5)), 'weak_side_balance')],
                         ['describe'])

    def test_the_binomial_tail(self):
        self.assertAlmostEqual(AS.binom_tail(1, 10, .048), 1-(1-.048)**10, places=12)
        self.assertAlmostEqual(AS.binom_tail(3, 8, .5), (256-1-8-28)/256.0, places=12)
        self.assertEqual((AS.binom_tail(0, 5, .1), AS.binom_tail(6, 5, .1), AS.binom_tail(1, 5, 0.0)), (1.0, 0.0, 0.0))
        self.assertAlmostEqual(AS.binom_tail(400, 900, .4), .0037337, places=6)     # scipy binom.sf(399, 900, .4)
        self.assertEqual(AS.above_base(3, 9, None, .05), None)
        self.assertAlmostEqual(AS.praise_base({'baseFailShare': .004, 'baseSuccessScale': .8}), 1-.996*.8, places=12)
        self.assertAlmostEqual(AS.praise_base({'baseFailShare': .2}), .2, places=12)
        self.assertEqual([profile(lv)['detectors']['behind_ball']['params'].get('baseSuccessScale')
                          for lv in range(1, 8)], [.5, .6, .7, .8, 1.0, 1.0, 1.0])

    def test_one_instance_is_a_review_never_a_work_item(self):
        # one distinct failing spell (four failures within 10 s) in a family of eight: at least half the level's
        # minimum opportunities (the S-1 review floor; level 7 needs 15, half is 7.5)
        opps = [weak(320.0+2.5*k, True) for k in range(4)]+[weak(t, False) for t in spaced(4, 340.0)]
        obs, roles = files(opps)
        for level in (3, 4, 7):
            f = FX.findings_of(AS.assess(obs, roles, profile(level)), 'weak_side_balance')[0]
            self.assertEqual(f['verdict'], 'review')
            self.assertIn('one_instance', f['limitations'])
        obs, roles = files([weak(320.0, True)])         # one opportunity alone: too little play to judge (S-1) ...
        out = AS.assess(obs, roles, profile(4))
        self.assertEqual(FX.findings_of(out, 'weak_side_balance'), [])
        # ... with 300 s of the phase: too few situations, not a short range (fix round 3, critic S-2)
        self.assertEqual(FX.status_of(out, 'weak_side_balance')['reason'], 'too_few_situations')
        obs, roles = files([weak(320.0, True)], opp=(300.0, 330.0))     # 30 s of the phase < half of 90 s
        self.assertEqual(FX.status_of(AS.assess(obs, roles, profile(4)), 'weak_side_balance')['reason'],
                         'short_range')

    def test_a_strength_is_a_demonstration_not_an_absence_of_failures(self):
        good = [weak(t, False) for t in spaced(12, 300.0)]
        obs, roles = files(good)
        f = FX.findings_of(AS.assess(obs, roles, profile(4)), 'weak_side_balance')[0]
        self.assertEqual(f['verdict'], 'strength')
        self.assertGreaterEqual(f['aggregate']['distinctDemonstratedSpells'], 3)
        unclear = [make_opp('weak_side_balance', t, {'weakCount': [0, 2]*7+[2], 'primaryNamed': True})
                   for t in spaced(12, 300.0)]                          # neither outcome reaches 60%
        obs, roles = files(unclear)
        out = AS.assess(obs, roles, profile(4))
        self.assertEqual(FX.findings_of(out, 'weak_side_balance'), [])
        self.assertEqual(FX.status_of(out, 'weak_side_balance')['status'], 'assessed')

    def test_a_declared_style_explains_the_pattern(self):
        opps = [weak(t, k % 2 == 0) for k, t in enumerate(spaced(12, 300.0))]
        obs, roles = files(opps)
        f = FX.findings_of(AS.assess(obs, roles, profile(4, style='compact')), 'weak_side_balance')[0]
        self.assertEqual(f['verdict'], 'describe')
        self.assertEqual(f['context']['styleEffect'], 'compact_far_side_accepted')
        self.assertEqual(f['guardrails']['intent'], 'fail')


class Guardrails(unittest.TestCase):
    def test_role_sources(self):
        for source, conf, want in (('stable', 'high', 'work_on'), ('carried', 'medium', 'work_on'),
                                   ('window', 'low', 'review'), ('coach', 'high', 'work_on')):
            opps = [advance(t, source=source, confidence=conf) for t in spaced(12)]
            obs, roles = files(opps)
            f = FX.findings_of(AS.assess(obs, roles, profile(4)), 'wide_defender_advance')[0]
            self.assertEqual(f['verdict'], want, source)
            self.assertEqual(f['guardrails']['role'], 'fail' if want == 'review' else 'pass')
            self.assertEqual(f['key'].split('|')[2], 'left')
            self.assertEqual(S.validate('finding', f), [])

    def test_a_formation_mismatch_downgrades_players_without_a_coach_role(self):
        book = {'team': 'A', 'format': '11v11', 'model': 'team_shape', 'records': [], 'interchanges': [],
                'formationChecks': [{'period': 0, 't0': 0.0, 't1': 600.0, 'measured': '3-5-2',
                                     'measuredConfidence': 'clear', 'intended': '4-4-2', 'mismatch': True,
                                     'related': False}]}
        obs, roles = files([advance(t) for t in spaced(12)], rolebook=book)
        f = FX.findings_of(AS.assess(obs, roles, profile(4)), 'wide_defender_advance')[0]
        self.assertEqual((f['verdict'], f['guardrails']['role']), ('review', 'fail'))
        self.assertIn('formation_mismatch', f['limitations'])
        obs, roles = files([advance(t, source='coach') for t in spaced(12)], rolebook=book)
        f = FX.findings_of(AS.assess(obs, roles, profile(4)), 'wide_defender_advance')[0]
        self.assertEqual(f['verdict'], 'work_on')

    def test_an_identity_crossing_at_the_evidence_moment(self):
        obs, roles = files([advance(t, named=False, behind=0) for t in spaced(12)])
        for level in (2, 4, 6):
            f = FX.findings_of(AS.assess(obs, roles, profile(level, style='possession')), 'wide_defender_advance')[0]
            self.assertEqual((f['verdict'], f['guardrails']['identity']), ('review', 'fail'), level)
            self.assertIn('identity_uncertain', f['limitations'])
        # level 1 judges the advance with the behind_ball rule, and it still rests on who advanced (critic B-6)
        f = FX.findings_of(AS.assess(obs, roles, profile(1)), 'behind_ball')
        self.assertEqual([(x['verdict'], x['guardrails']['identity'], x['key'].split('|')[2]) for x in f],
                         [('review', 'fail', 'advance')])

    def test_low_identity_quality_fails_with_a_clean_moment(self):
        """S-15 (CODE M4 survivor 'identity_low_passes'): an opportunity whose identity quality is low fails the
        identity guardrail even when the evidence moment itself is clean."""
        obs, roles = files([advance(t, identity='low') for t in spaced(12)])
        f = FX.findings_of(AS.assess(obs, roles, profile(4)), 'wide_defender_advance')[0]
        self.assertEqual((f['verdict'], f['guardrails']['identity']), ('review', 'fail'))

    def test_a_centre_back_shift_alone_covers(self):
        """S-15 (CODE M4 survivor 'd3_cb_adjust_not_cover'): the channel empty, the near centre-back 5 m across: a
        supported advance (a strength at level 4), never a criticism."""
        obs, roles = files([advance(t, covered=False, cb_shift=5.0) for t in spaced(12)])
        f = FX.findings_of(AS.assess(obs, roles, profile(4)), 'wide_defender_advance')
        self.assertEqual([x['verdict'] for x in f], ['strength'])

    def test_role_based_criticism_is_suppressed_near_a_formation_change(self):
        obs, roles = files([advance(t, fcn=True) for t in spaced(12)])
        out = AS.assess(obs, roles, profile(4))
        self.assertEqual([f['verdict'] for f in FX.findings_of(out, 'wide_defender_advance')], [])

    def test_the_level_frames_the_criticism(self):
        opps = [advance(t) for t in spaced(16, every=18.0)]      # level 7 needs 15 opportunities
        obs, roles = files(opps)
        verdicts = {}
        for level in range(2, 8):
            out = AS.assess(obs, roles, profile(level))
            for f in out['findings']:
                self.assertEqual(S.validate('finding', f), [])
            verdicts[level] = [f['verdict'] for f in FX.findings_of(out, 'wide_defender_advance')]
        # level 2 (fix round 3, critic B-1): an uncovered advance with an opponent in the vacated channel fails
        # (someone was behind the ball, so the final-third rule alone would not see it)
        self.assertEqual(verdicts[2], ['try_this'])
        self.assertEqual(FX.findings_of(AS.assess(obs, roles, profile(2)), 'wide_defender_advance')[0]['measurements']
                         ['failureMode'], 'exposed')
        obs0, roles0 = files([advance(t, opponents=0) for t in spaced(16, every=18.0)])
        self.assertEqual(FX.findings_of(AS.assess(obs0, roles0, profile(2)), 'wide_defender_advance'), [])
        self.assertEqual(verdicts[3], ['work_on'])
        self.assertEqual(verdicts[4], ['work_on'])
        self.assertEqual(verdicts[7], ['review'])       # level 7 without a declared style: reviews only
        out = AS.assess(obs, roles, profile(7, style='possession'))
        self.assertEqual([f['verdict'] for f in FX.findings_of(out, 'wide_defender_advance')], ['work_on'])
        nobody = [advance(t, behind=0) for t in spaced(12)]
        obs, roles = files(nobody)
        out = AS.assess(obs, roles, profile(2))
        self.assertEqual([f['verdict'] for f in FX.findings_of(out, 'wide_defender_advance')], ['try_this'])
        # ... nobody behind the ball only with the ball in the final third (critic B-1): a build-up advance with no
        # opponent in the vacated channel is not judged at level 2; with one there it is (fix round 3)
        obs, roles = files([advance(t, behind=0, ball_x=50.0, opponents=0) for t in spaced(12)])
        self.assertEqual(FX.findings_of(AS.assess(obs, roles, profile(2)), 'wide_defender_advance'), [])
        obs, roles = files([advance(t, behind=0, ball_x=50.0) for t in spaced(12)])
        self.assertEqual([f['verdict'] for f in FX.findings_of(AS.assess(obs, roles, profile(2)),
                                                                'wide_defender_advance')], ['try_this'])
        exposed_only = [advance(t, opponents=0) for t in spaced(12)]
        obs, roles = files(exposed_only)
        out = AS.assess(obs, roles, profile(6, style='possession'))
        self.assertEqual(FX.findings_of(out, 'wide_defender_advance', S.CRITICISM), [])   # level 6: exposed only

    def test_level_one_reframes_an_advance_as_someone_ready_to_protect_our_goal(self):
        obs, roles = files([advance(t, behind=0) for t in spaced(12)])
        out = AS.assess(obs, roles, profile(1, players=11))
        self.assertEqual(FX.status_of(out, 'wide_defender_advance')['status'], 'off')
        f = FX.findings_of(out, 'behind_ball')
        self.assertEqual([(x['verdict'], x['key'].split('|')[2]) for x in f], [('try_this', 'advance')])
        self.assertEqual(f[0]['guardrails']['identity'], 'pass')          # who advanced is the measurement (B-6)
        self.assertEqual(S.validate('finding', f[0]), [])
        # critic B-1: never on build-up with the vacated channel empty of opponents (the ball outside the final
        # third), never on a covered advance ...
        for kw in ({'ball_x': 50.0, 'opponents': 0}, {'covered': True}, {'cb_shift': 5.0},
                   {'ball_x': None, 'opponents': 0}):
            obs, roles = files([advance(t, behind=0, **kw) for t in spaced(12)])
            out = AS.assess(obs, roles, profile(1, players=11))
            self.assertEqual(FX.findings_of(out, 'behind_ball', ('try_this', 'review')), [], kw)
        # ... but an uncovered advance with one of theirs in the channel we left is the level's question wherever the
        # ball is (fix round 3, critic B-1 / COACH F-1: the spec's "nobody stayed behind to cover"), said as such
        for kw in ({'ball_x': 50.0}, {'ball_x': None}, {'behind': 2}):
            obs, roles = files([advance(t, **dict({'behind': 0}, **kw)) for t in spaced(12)])
            f = FX.findings_of(AS.assess(obs, roles, profile(1, players=11)), 'behind_ball', ('try_this',))
            self.assertEqual([(x['key'].split('|')[2], x['measurements']['failureMode'],
                               x['measurements']['opponentsInChannel']) for x in f], [('advance', 'exposed', 1)], kw)


class Report(unittest.TestCase):
    def test_level_cap_and_priorities(self):
        shape = [make_opp('attacking_shape', t, {'widthShare': .4, 'depthShare': .2, 'channelsOccupied': 2,
                                                 'widthM': 27.0, 'depthM': 21.0}) for t in spaced(12)]
        back = [make_opp('behind_ball', t, {'behindN': 0, 'restMargin': None, 'threatN': None, 'ballX': 80.0},
                         primary=('A-05',)) for t in spaced(12, 15.0)]
        obs, roles = files(shape+back)
        out = AS.assess(obs, roles, profile(1, players=11))
        work = [f for f in out['findings'] if f['verdict'] == 'try_this']
        self.assertEqual([f['observation'] for f in work], ['attacking_shape'])     # level 1: one, the first objective
        capped = FX.findings_of(out, 'behind_ball')[0]
        self.assertEqual(capped['verdict'], 'review')
        self.assertIn('level_cap', capped['limitations'])
        out = AS.assess(obs, roles, profile(2, players=11))
        self.assertEqual(len([f for f in out['findings'] if f['verdict'] == 'try_this']), 2)
        self.assertEqual(len(out['priorities']), 2)

    def test_statuses_range_and_no_moment(self):
        opps = [weak(t, True) for t in spaced(12, 300.0)]
        cov = {'pressure_cover': {'detector': 'pressure_cover', 'tier': 'measurement', 'eligibleSeconds': 0.0,
                                  'phaseSeconds': 300.0, 'opportunities': 0, 'eligibleSpans': [],
                                  'abstain': {'no_carrier': 200.0}, 'dominantReason': 'no_carrier'}}
        obs, roles = files(opps, coverage=cov)
        out = AS.assess(obs, roles, profile(4))
        st = {d['detector']: d for d in out['detectors']}
        self.assertEqual(st['pressure_cover']['status'], 'not_assessed')
        self.assertEqual(st['pressure_cover']['reason'], 'no_carrier')
        self.assertEqual(st['weak_side_balance']['opportunities'], 12)
        self.assertEqual(len(out['detectors']), 6)
        out = AS.assess(obs, roles, profile(1))
        self.assertEqual(FX.status_of(out, 'weak_side_balance')['status'], 'off')
        out = AS.assess(obs, roles, profile(4), t0=300.0, t1=400.0)          # the first five only
        self.assertEqual(FX.status_of(out, 'weak_side_balance')['opportunities'], 5)
        self.assertAlmostEqual(FX.findings_of(out, 'weak_side_balance')[0]['aggregate']['phaseSeconds'], 100.2)
        bare = copy.deepcopy(obs)
        bare['moments'] = []
        f = FX.findings_of(AS.assess(bare, roles, profile(4)), 'weak_side_balance')[0]
        self.assertEqual((f['verdict'], f['timestamps']), ('describe', []))
        self.assertIn('no_moment', f['limitations'])

    def test_unknown_phase_shape_is_described_only(self):
        o = make_opp('attacking_shape', 50.0, {'widthShare': .4, 'depthShare': .2, 'channelsOccupied': 2},
                     state='unknown', variant='unknown')
        o['phase'] = {'state': 'unknown', 'source': 'none', 'estimated': False, 'spanSeconds': None,
                      'nearChange': False}
        obs, roles = files([o], own=None, opp=None)
        for level in (1, 4):
            f = FX.findings_of(AS.assess(obs, roles, profile(level)), 'attacking_shape')
            self.assertEqual([x['verdict'] for x in f], ['describe'])

    def test_measurements_explain_reads(self):
        d1 = [make_opp('pressure_cover', t, {'d1': 3.0, 'closingMS': 0.5, 'coverDist': 18.0, 'coverAngle': 30.0,
                                             'coverVisible': True, 'coverId': 'A-06', 'carrierX': 20.0,
                                             'goalSideShare': .7, 'primaryNamed': True, 'fd': 'A-03'})
              for t in spaced(12, 300.0)]
        obs, roles = files(d1)
        f = FX.findings_of(AS.assess(obs, roles, profile(4)), 'pressure_cover')[0]
        m = f['measurements']
        self.assertEqual(f['verdict'], 'work_on')
        self.assertEqual((m['failureMode'], m['pressurePresent'], m['coverPresent'], m['d1M'], m['coverDistM']),
                         ('no_cover', True, False, 3.0, 18.0))
        self.assertEqual(m['coverPresentShare'], 0.0)
        self.assertEqual((m['fdId'], f['players']['primary'], f['players']['supporting'][:1]),
                         ('A-03', ['A-03'], ['A-06']))
        out = AS.assess(obs, roles, profile(1))       # level 1: going to the ball is enough
        self.assertEqual(FX.findings_of(out, 'pressure_cover')[0]['verdict'], 'strength')

    def test_the_presser_is_the_moments_first_defender(self):
        """B-3 (FACTS H3): the family's modal first defender is A-10, but at the moment cited A-09 presses and A-10
        covers: the finding names A-09 as the presser and A-10 as the cover, never one player as both."""
        n = 15
        fd = ['A-10']*n
        fd[n//2] = 'A-09'
        cover = ['A-09']*n
        cover[n//2] = 'A-10'
        d1 = [make_opp('pressure_cover', t, {'d1': 2.0, 'closingMS': 0.5, 'coverDist': 18.0, 'coverAngle': 30.0,
                                             'coverVisible': True, 'coverId': cover, 'carrierX': 20.0,
                                             'goalSideShare': .7, 'primaryNamed': True, 'fd': fd},
                       primary=('A-10',)) for t in spaced(12, 300.0)]
        obs, roles = files(d1)
        f = FX.findings_of(AS.assess(obs, roles, profile(4)), 'pressure_cover')[0]
        self.assertEqual((f['measurements']['fdId'], f['measurements']['coverId']), ('A-09', 'A-10'))
        self.assertEqual(f['players']['primary'], ['A-09'])
        self.assertNotIn('A-09', f['players']['supporting'])

    def test_three_state_pressure(self):
        """S-2: a first defender 3-5 m away without closing is unclear, not pressing without cover."""
        d1 = [make_opp('pressure_cover', t, {'d1': 4.0, 'closingMS': -1.0, 'coverDist': 18.0, 'coverAngle': 30.0,
                                             'coverVisible': True, 'coverId': 'A-06', 'carrierX': 20.0,
                                             'goalSideShare': .7, 'primaryNamed': True, 'fd': 'A-03'})
              for t in spaced(12, 300.0)]
        obs, roles = files(d1)
        out = AS.assess(obs, roles, profile(4))
        self.assertEqual(FX.findings_of(out, 'pressure_cover', S.CRITICISM+('review',)), [])
        closing = copy.deepcopy(d1)
        for o in closing:
            o['relations']['closingMS'] = [1.5]*len(o['relations']['t'])
        obs, roles = files(closing)
        f = FX.findings_of(AS.assess(obs, roles, profile(4)), 'pressure_cover')[0]
        self.assertEqual((f['verdict'], f['measurements']['failureMode']), ('work_on', 'no_cover'))

    def test_every_not_assessed_check_names_why(self):
        """S-7: the phase never occurred (phase_absent), the coach marked only one team (coach_marks_one_side), no
        phase anywhere (phase_unknown); a check whose phase occurred and lost nothing to a gate is assessed with 0
        opportunities (the situation never arose)."""
        obs, roles = files([weak(t, True) for t in spaced(12, 300.0)], own=None)
        obs['timeline']['phaseQuality']['available'] = True
        out = AS.assess(obs, roles, profile(4))
        st = {d['detector']: d for d in out['detectors']}
        for d in ('wide_defender_advance', 'behind_ball', 'attacking_shape'):
            self.assertEqual((st[d]['status'], st[d]['reason']), ('not_assessed', 'phase_absent'), d)
        self.assertEqual((st['defensive_spacing']['status'], st['defensive_spacing']['reason']), ('assessed', None))
        one = copy.deepcopy(obs)
        one['timeline']['phaseQuality'] = {'available': False, 'coverage': None,
                                           'coach': {'spans': 1, 'dropped': 0, 'seconds': 300.0, 'own': 0.0,
                                                     'opp': 300.0, 'dead': 0.0, 'engineNamedSeconds': 0.0,
                                                     'agreement': None}}
        st = {d['detector']: d for d in AS.assess(one, roles, profile(4))['detectors']}
        self.assertEqual(st['behind_ball']['reason'], 'coach_marks_one_side')
        none = copy.deepcopy(obs)
        none['timeline']['phaseQuality'] = {'available': False, 'coverage': None}
        st = {d['detector']: d for d in AS.assess(none, roles, profile(4))['detectors']}
        self.assertEqual(st['behind_ball']['reason'], 'phase_unknown')
        for d in out['detectors']:
            self.assertTrue(d['status'] != 'not_assessed' or d['reason'], d)

    def test_a_formation_change_is_reported(self):
        """S-3: role-based criticism suppressed by a formation change says so (not assessed: formation_changing)."""
        obs, roles = files([advance(t, fcn=True) for t in spaced(12)])
        out = AS.assess(obs, roles, profile(4))
        self.assertEqual((FX.status_of(out, 'wide_defender_advance')['status'],
                          FX.status_of(out, 'wide_defender_advance')['reason']), ('not_assessed', 'formation_changing'))

    def test_the_review_floor(self):
        """S-1: below reviewFloorShare of the level's minimum opportunities or phase seconds a failing pattern is no
        review: the check reads not assessed, 'too_few_situations' when the phase lasted long enough but the
        situations were too few, 'short_range' when the phase itself was too short (fix round 3, critic S-2)."""
        obs, roles = files([weak(320.0, True), weak(340.0, True)])
        prof = profile(4)
        prof['detectors']['weak_side_balance']['params']['reviewFloorShare'] = .5      # 5 opportunities, 45 s
        out = AS.assess(obs, roles, prof)
        self.assertEqual(FX.findings_of(out, 'weak_side_balance'), [])
        self.assertEqual((FX.status_of(out, 'weak_side_balance')['status'], FX.status_of(out, 'weak_side_balance')
                          ['reason']), ('not_assessed', 'too_few_situations'))
        out = AS.assess(*files([weak(320.0, True), weak(340.0, True)], opp=(300.0, 340.0)), prof)
        self.assertEqual(FX.status_of(out, 'weak_side_balance')['reason'], 'short_range')    # 40 s of the phase
        prof['detectors']['weak_side_balance']['params']['reviewFloorShare'] = .1
        self.assertEqual([f['verdict'] for f in FX.findings_of(AS.assess(obs, roles, prof), 'weak_side_balance')],
                         ['review'])

    def test_the_next_objective_skips_what_was_shown(self):
        """S-8 (COACH M5): a check demonstrated without a failure is not the next objective."""
        back = [make_opp('behind_ball', t, {'behindN': 2, 'restMargin': None, 'threatN': None, 'ballX': 80.0},
                         primary=('A-05',)) for t in spaced(3)]
        obs, roles = files(back)
        out = AS.assess(obs, roles, profile(1))
        self.assertNotEqual(out['nextObjective']['skill'], 'someone_back_when_attacking')
        self.assertEqual([f['verdict'] for f in FX.findings_of(out, 'behind_ball')], ['strength'])   # L1 praise

    def test_compact_style_never_excuses_an_empty_middle(self):
        """S-8 (COACH M2): compact defending accepts far-side space only with the central corridor kept."""
        def w(t0, failing, central):
            return make_opp('weak_side_balance', t0, {'weakCount': 0 if failing else 2, 'centralCount': central,
                                                      'weakGoalCount': 0 if failing else 2,
                                                      'centralGoalCount': central, 'weakLevelCount': 0,
                                                      'weakRecovering': 0, 'openWidthM': 30.0, 'primaryNamed': True,
                                                      'ballX': 40.0})
        kept = [w(t, k % 2 == 0, 1) for k, t in enumerate(spaced(12, 300.0))]
        obs, roles = files(kept)
        f = FX.findings_of(AS.assess(obs, roles, profile(4, style='compact')), 'weak_side_balance')[0]
        self.assertEqual(f['verdict'], 'describe')
        empty = [w(t, k % 2 == 0, 0) for k, t in enumerate(spaced(12, 300.0))]
        obs, roles = files(empty)
        f = FX.findings_of(AS.assess(obs, roles, profile(4, style='compact')), 'weak_side_balance')[0]
        self.assertEqual(f['verdict'], 'work_on')

    def test_the_same_moments_at_every_level(self):
        """S-5: one family cites the same moments whatever the level judges (exposed advances first: they fail at
        levels 3-7, uncovered ones at levels 3-5 only)."""
        opps = [advance(t, opponents=1 if k % 3 == 0 else 0) for k, t in enumerate(spaced(16, every=18.0))]
        obs, roles = files(opps)
        cited = {}
        for level in (3, 4, 5, 6, 7):
            for f in FX.findings_of(AS.assess(obs, roles, profile(level, style='possession')),
                                    'wide_defender_advance', S.CRITICISM+('review',)):
                cited[level] = [t['evidence'] for t in f['timestamps']]
        self.assertEqual(len({tuple(v) for v in cited.values()}), 1, cited)

    def test_assess_never_changes_its_inputs(self):
        obs, roles = files([advance(t) for t in spaced(12)]+[weak(t, True) for t in spaced(12, 300.0)])
        before = json.dumps([obs, roles], sort_keys=True)
        for level in range(1, 8):
            for style in S.STYLES:
                AS.assess(obs, roles, profile(level, style=style))
        self.assertEqual(json.dumps([obs, roles], sort_keys=True), before)


def back(t0, behind, n=20, ball_x=80.0, **kw):
    """A behind-the-ball opportunity of n samples (4 s at 5 Hz) with the ball in the final third."""
    return make_opp('behind_ball', t0, {'behindN': behind, 'restMargin': None, 'threatN': None, 'ballX': ball_x}, n=n,
                    primary=('A-05',), **kw)


def mixed_family(fails=3, good=4, every=20.0, fail_kw=None, fail_behind=0, unclear=3):
    """good demonstrated, then `fails` failing behind-the-ball opportunities `every` s apart, then `unclear` ones
    (half their samples fail): 3 of 10 fail (under the level 3-4 failure shares), 4 of the 7 decided demonstrated
    (under the level 3-4 strength shares; a strength at levels 1-2), the 3 failures far above the level 3-4 base
    (.017: a tail of .0002)."""
    opps = [back(t, 3) for t in spaced(good, 10.0, every)]
    opps += [back(t, fail_behind, **(fail_kw or {})) for t in spaced(fails, 12.0+every*good, every)]
    opps += [back(t, [3, 0]*10) for t in spaced(unclear, 14.0+every*(good+fails), every)]
    return opps


def low_base(prof, detector, base=.05):
    """The profile with the detector's baseFailShare set to `base` (a unit test of the clear-moment mechanics, not of
    the base-rate trigger)."""
    prof['detectors'][detector]['params']['baseFailShare'] = base
    return prof


def mwl(out, detector):
    return [f for f in FX.findings_of(out, detector) if AS.MOMENTS in f['limitations']]


class MomentsWorthALook(unittest.TestCase):
    """Fix round 2, LEAD decision (b): the verdicts keep their strict majority rules; from level 3 a family with no
    verdict, review or strength whose failures are CLEAR (a run of failing samples as long as the failure spell with no
    unclear or out-of-scope sample, stable identity, the engine's or the coach's phase, every guardrail) in at least
    momentsMinSpells distinct spells is a 'moments worth a look' review: the counts, the clear moments, no verdict,
    at most momentsMaxPerReport per report; levels 1-2 unchanged."""

    def test_repeated_clear_failures_are_moments_worth_a_look_from_level_3(self):
        obs, roles = files(mixed_family(), own=(0.0, 300.0))
        fail_ids = {o['id'] for o in obs['opportunities'] if o['relations']['behindN'] == [0]*20}
        for level in (1, 2):
            # 4 of 7 shown (>= .5); 3 failures are no evidence against the scaled youth base (1 - 1 x .5 / .6: tails
            # .77 / .58), so praised (fix round 3, LEAD decision (3)); levels 1-2 never get the review
            f = FX.findings_of(AS.assess(obs, roles, profile(level)), 'behind_ball')
            self.assertEqual([(x['verdict'], 'mixed_evidence' in x['limitations']) for x in f], [('strength', False)])
        for level in (3, 4):
            out = AS.assess(obs, roles, profile(level))
            f = FX.findings_of(out, 'behind_ball')
            self.assertEqual([x['verdict'] for x in f], ['review'], level)
            f = f[0]
            self.assertEqual(S.validate('finding', f), [])
            self.assertIn(AS.MOMENTS, f['limitations'])
            self.assertNotIn('mixed_evidence', f['limitations'])
            self.assertNotIn('one_instance', f['limitations'])
            m = f['measurements']
            self.assertEqual((m['clearMoments'], m['clearSpells'], m['decidedMoments'], m['failingMoments']),
                             (3, 3, 7, 3))
            self.assertEqual((m['baseShare'], m['baseTail']), (.017, round(AS.binom_tail(3, 7, .017), 4)))
            self.assertEqual(len(f['timestamps']), 3)
            self.assertTrue({t['opportunity'] for t in f['timestamps']} <= fail_ids)
            self.assertEqual(f['priority'], None)
            self.assertEqual(out['priorities'], [])
            g = f['guardrails']
            self.assertEqual([k for k in ('identity', 'position', 'role', 'context', 'phase', 'intent')
                              if g[k] == 'fail'], [])
            self.assertEqual(g['evidence'], 'fail')                     # the level's verdict table was not met
            self.assertEqual(FX.status_of(out, 'behind_ball')['status'], 'assessed')
            # ... 4 of the 7 decided shown, under the strength share: the next objective from this match (S-8)
            self.assertEqual((out['nextObjective']['skill'], out['nextObjective']['fromMatch']),
                             ('someone_back_when_attacking', False))
            self.assertFalse(AS.is_level_focus(out['nextObjective']))
        # the trigger is relative (fix round 3, LEAD decision (2)): the same 3 clear spells of 7 against a base of
        # .25 (tail .36) are what professional play gives, never a review
        prof = profile(4)
        prof['detectors']['behind_ball']['params']['baseFailShare'] = .25
        self.assertEqual(mwl(AS.assess(obs, roles, prof), 'behind_ball'), [])
        del prof['detectors']['behind_ball']['params']['baseFailShare']    # no measured base: no review
        self.assertEqual(mwl(AS.assess(obs, roles, prof), 'behind_ball'), [])

    def test_only_clear_failures_count(self):
        """A failing run broken by an out-of-scope sample, a crossing taint, a phase near a change or with no source, a
        recovering run: the failure is not clear, and two clear spells are not enough."""
        gap = [0]*9+[None]+[0]*10                       # 1.8 s + 2.0 s pure runs under D4's 3 s spell
        cases = {'broken run': {'n': 20}, 'identity': {'identity': 'medium'},
                 'recovering': {'intent': {'recovering': True, 'interchange': None, 'formationChangeNear': False}}}
        for name, kw in cases.items():
            opps = mixed_family(fail_kw=kw, fail_behind=gap if name == 'broken run' else 0)
            obs, roles = files(opps)
            out = AS.assess(obs, roles, profile(4))
            self.assertEqual(mwl(out, 'behind_ball'), [], name)
            f = FX.findings_of(out, 'behind_ball')
            self.assertEqual([x['verdict'] for x in f], [], name)      # no verdict, no strength, nothing to say
        for name, change in (('near a change', {'nearChange': True}), ('no phase source', {'source': 'none'})):
            opps = mixed_family()
            for o in opps:
                if o['relations']['behindN'][0] == 0:
                    o['phase'].update(change)
            obs, roles = files(opps)
            self.assertEqual(mwl(AS.assess(obs, roles, profile(4)), 'behind_ball'), [], name)
        obs, roles = files(mixed_family(fails=2))
        self.assertEqual(mwl(AS.assess(obs, roles, profile(4)), 'behind_ball'), [])
        obs, roles = files(mixed_family(fails=3, every=4.5))   # three failing spells within 10 s: one distinct spell
        self.assertEqual(mwl(AS.assess(obs, roles, profile(4)), 'behind_ball'), [])

    def test_the_review_floor_and_the_first_level(self):
        obs, roles = files(mixed_family(), own=(0.0, 40.0))     # 40 s of own phase < half of level 4's 90 s
        self.assertEqual(mwl(AS.assess(obs, roles, profile(4)), 'behind_ball'), [])
        prof = profile(4)
        for dp in prof['detectors'].values():
            dp['params']['momentsFromLevel'] = 5
        obs, roles = files(mixed_family())
        self.assertEqual(mwl(AS.assess(obs, roles, prof), 'behind_ball'), [])
        for dp in prof['detectors'].values():
            dp['params']['momentsFromLevel'] = 4
        self.assertEqual(len(mwl(AS.assess(obs, roles, prof), 'behind_ball')), 1)

    def test_at_most_the_configured_number_per_report(self):
        """Two checks qualify; the cap (1) keeps the more relevant one and the other family gives what it gives
        without the review (here a mixed-evidence description, or nothing); with a cap of 2 both stay."""
        far = [weak(t, False) for t in spaced(4, 300.0)]+[weak(t, True) for t in spaced(3, 382.0)]
        far += [make_opp('weak_side_balance', t, {'weakCount': [0, 2]*7+[2], 'primaryNamed': True})
                for t in spaced(10, 445.0, 15.0)]               # 10 unclear: 3 of 17 fail, 4 of 7 decided shown
        obs, roles = files(mixed_family()+far)
        out = AS.assess(obs, roles, profile(4))
        got = [f['observation'] for f in out['findings'] if AS.MOMENTS in f['limitations']]
        self.assertEqual(len(got), 1)
        other = 'weak_side_balance' if got == ['behind_ball'] else 'behind_ball'
        rest = FX.findings_of(out, other)
        self.assertTrue(all(AS.MOMENTS not in f['limitations'] and f['verdict'] == 'describe' for f in rest), rest)
        self.assertEqual(FX.status_of(out, other)['status'], 'assessed')
        prof = profile(4)
        for dp in prof['detectors'].values():
            dp['params']['momentsMaxPerReport'] = 2
        out2 = AS.assess(obs, roles, prof)
        self.assertEqual(sorted(f['observation'] for f in out2['findings'] if AS.MOMENTS in f['limitations']),
                         ['behind_ball', 'weak_side_balance'])
        for f in out['findings']+out2['findings']:
            self.assertEqual(S.validate('finding', f), [])
            self.assertFalse(set(f) & {'_moments', '_fallback', '_score'})

    def test_the_presser_and_a_crossing_at_the_moment(self):
        def d1(t, failing, named=True):
            return make_opp('pressure_cover', t, {'d1': 2.0, 'closingMS': 0.5, 'coverDist': 18.0 if failing else 6.0,
                                                  'coverAngle': 30.0, 'coverVisible': True, 'coverId': 'A-06',
                                                  'carrierX': 20.0, 'goalSideShare': .7, 'primaryNamed': named,
                                                  'fd': 'A-03'})
        # 5 covered, 3 without cover: 3 of 8 fail (< .4), 5 of 8 shown (< .65): neither a verdict nor a strength;
        # against the shipped level-4 base (.654: professionals fail the cover check on most decided moments) 3 of 8
        # is no evidence (fix round 3, LEAD decision (2)); the clear-moment mechanics are shown with a base of .05
        opps = [d1(t, False) for t in spaced(5, 300.0)]+[d1(t, True) for t in spaced(3, 402.0)]
        obs, roles = files(opps)
        self.assertEqual(mwl(AS.assess(obs, roles, profile(4)), 'pressure_cover'), [])
        f = mwl(AS.assess(obs, roles, low_base(profile(4), 'pressure_cover')), 'pressure_cover')
        self.assertEqual(len(f), 1)
        self.assertEqual((f[0]['measurements']['fdId'], f[0]['players']['primary'], f[0]['measurements']['failureMode'],
                          f[0]['measurements']['decidedMoments']), ('A-03', ['A-03'], 'no_cover', 8))
        crossed = [d1(t, False) for t in spaced(5, 300.0)]+[d1(t, True, named=False) for t in spaced(3, 402.0)]
        obs, roles = files(crossed)
        self.assertEqual(mwl(AS.assess(obs, roles, low_base(profile(4), 'pressure_cover')), 'pressure_cover'), [])

    def test_nobody_pressed_is_clear_only_while_the_gap_is_not_closing(self):
        """A carrier in our third with our nearest player 8 m away is 'not pressed' under the three-state call; it is
        a clear moment only while the gap is not shrinking at 1 m/s or more (pressure arriving is not a carrier left
        alone: on Metrica game 1 every such moment of the rule's first version sat in a gap closing to 1.5-5 m)."""
        def d1(t, d, closing=.5):
            return make_opp('pressure_cover', t, {'d1': d, 'closingMS': closing, 'coverDist': 6.0, 'coverAngle': 30.0,
                                                  'coverVisible': True, 'coverId': 'A-06', 'carrierX': 20.0,
                                                  'goalSideShare': .7, 'primaryNamed': True, 'fd': 'A-03'})
        covered = [d1(t, 2.0) for t in spaced(5, 300.0)]
        gaps = {'steady': [8.0]*15, 'escaping': [8.0+.4*k for k in range(15)],
                'closing': [12.0-.4*k for k in range(15)]}         # 2 m/s, still 6.4 m at the end
        for name, gap, want in (('steady', gaps['steady'], 1), ('escaping', gaps['escaping'], 1),
                                ('closing', gaps['closing'], 0)):
            obs, roles = files(covered+[d1(t, gap, closing=3.0) for t in spaced(3, 402.0)])
            f = mwl(AS.assess(obs, roles, low_base(profile(4), 'pressure_cover')), 'pressure_cover')
            self.assertEqual(len(f), want, name)
            if f:
                self.assertEqual(f[0]['measurements']['failureMode'], 'no_pressure')


    def test_clear_moments_lie_in_settled_play(self):
        """Fix round 3 (LEAD decision (2), FACTS F-3): a failing sample is clear only at least momentsSettleS (2 s)
        after a dead ball and from any stretch whose team was not named, and momentsContestS (1 s) from a ball the
        possession engine calls loose or contested; one failing opportunity made unclear leaves 2 clear spells."""
        def spans_with(obs, extra):
            tl = obs['timeline']
            tl['phaseSpans'] = [s for s in tl['phaseSpans'] if s['state'] != 'own']+extra
        own = lambda a, b: {'state': 'own', 't0': a, 't1': b, 'frame0': 0, 'frame1': 1, 'source': 'possession-4'}
        dead = lambda a, b: {'state': 'dead', 't0': a, 't1': b, 'frame0': 0, 'frame1': 1, 'source': 'possession-4'}
        obs, roles = files(mixed_family())          # the second failing opportunity is 112.0-115.8 s
        self.assertEqual(len(mwl(AS.assess(obs, roles, profile(4)), 'behind_ball')), 1)
        cases = {
            # a throw-in's dead ball ending at 113 s: 113-115 s are its restart window, 1 s of failure stays
            'restart': [own(0.0, 109.8), dead(110.0, 113.0), own(113.2, 299.8)],
            # the engine names no team 109.8-114.0 s (a change window): 114-116 s are not settled either
            'unnamed': [own(0.0, 109.8), own(114.0, 299.8)],
            # a coach-marked change straight from their ball to ours at 112.5 s
            'switch': [{'state': 'opp', 't0': 100.0, 't1': 112.4, 'frame0': 0, 'frame1': 1, 'source': 'coach'},
                       own(112.6, 299.8)]}
        for name, extra in cases.items():
            obs, roles = files(mixed_family())
            spans_with(obs, extra)
            self.assertEqual(mwl(AS.assess(obs, roles, profile(4)), 'behind_ball'), [], name)
            prof = profile(4)
            for dp in prof['detectors'].values():
                dp['params']['momentsSettleS'] = 0.0
            self.assertEqual(len(mwl(AS.assess(obs, roles, prof), 'behind_ball')), 1, name)
        opps = mixed_family()
        second = [o for o in opps if o['t0'] == 112.0][0]
        second['relations']['contestS'] = [.4]*20                   # a contested ball 0.4 s away throughout
        obs, roles = files(opps)
        self.assertEqual(mwl(AS.assess(obs, roles, profile(4)), 'behind_ball'), [])
        second['relations']['contestS'] = [1.0]*20
        obs, roles = files(opps)
        self.assertEqual(len(mwl(AS.assess(obs, roles, profile(4)), 'behind_ball')), 1)
        idx = AS.settle_index({'step': .2, 'phaseSpans': cases['restart']})
        self.assertEqual([AS.settled(idx, t, 2.0) for t in (100.0, 108.0, 111.0, 114.9, 115.0, 200.0)],
                         [True, True, False, False, True, True])           # (before a dead ball is still play)

    def test_back_line_clear_moments_need_a_line_and_a_free_opponent(self):
        """Fix round 3 (FACTS F-3, critic N1): on 11v11 a lateral-gap failure is clear only with a recognised line of
        at least lineMinPlayers (4) with a depth spread of at most lineStdMaxM (4 m), and while the opponent in the gap
        is free: nobody of ours goal-side of him within markedM (4 m; gapOppFreeM)."""
        def d5(t, failing, **kw):
            rel = {'lineStdM': 1.0, 'lineSize': 4, 'lateralGapM': 25.0 if failing else 10.0, 'gapLeft': 'A-02',
                   'gapRight': 'A-03', 'gapBallSide': True, 'opponentInGap': failing, 'gapOpponents': int(failing),
                   'dmGapM': 10.0, 'dmGapFree': 0, 'gapOppFreeM': 8.0 if failing else None, 'primaryNamed': True}
            rel.update(kw)
            return make_opp('defensive_spacing', t, rel, primary=('A-03',),
                            roles={'A-03': {'role': 'back-2/4', 'source': 'relative', 'confidence': 'high'}})
        unclear = [make_opp('defensive_spacing', t, {'lineStdM': 1.0, 'lineSize': 4, 'lateralGapM': [25.0, 10.0]*7+[10.0],
                                                     'gapBallSide': True, 'opponentInGap': True, 'dmGapM': 10.0,
                                                     'dmGapFree': 0, 'gapOppFreeM': 8.0, 'primaryNamed': True})
                   for t in spaced(3, 460.0)]

        def family(**kw):
            return [d5(t, False) for t in spaced(4, 300.0)]+[d5(t, True, **kw) for t in spaced(3, 382.0)]+unclear
        obs, roles = files(family())
        f = mwl(AS.assess(obs, roles, profile(4)), 'defensive_spacing')
        self.assertEqual(len(f), 1)
        m = f[0]['measurements']
        self.assertEqual((m['clearSpells'], m['decidedMoments'], m['failureMode'], m['gapOppFreeM']),
                         (3, 7, 'lateral_gap', 8.0))
        for name, kw in (('three-player line', {'lineSize': 3}), ('a line 5 m deep', {'lineStdM': 5.0}),
                         ('the opponent marked', {'gapOppFreeM': 3.5})):
            obs, roles = files(family(**kw))
            self.assertEqual(mwl(AS.assess(obs, roles, profile(4)), 'defensive_spacing'), [], name)
        obs, roles = files(family(lineSize=3))
        obs['timeline']['format'] = '7v7'           # small formats: the back block is the format's, any size
        self.assertEqual(len(mwl(AS.assess(obs, roles, profile(4, players=7)), 'defensive_spacing')), 1)

    def test_wide_defender_clear_advances_are_their_own_floor(self):
        """Fix round 3 (critic S-2 / COACH F-14): each wide-defender opportunity is a whole advance, so 3 clear ones
        reach a 'moments worth a look' review below the level's opportunity floor (5 at level 4); other checks keep
        the floor (and the check reads 'too_few_situations', never 'short_range', with enough of the phase)."""
        obs, roles = files([advance(t) for t in spaced(3, 10.0, 60.0)])
        out = AS.assess(obs, roles, profile(4))
        f = mwl(out, 'wide_defender_advance')
        self.assertEqual([(x['verdict'], x['measurements']['clearSpells'], x['measurements']['decidedMoments'])
                          for x in f], [('review', 3, 3)])
        obs, roles = files([advance(t) for t in spaced(2, 10.0, 60.0)])
        out = AS.assess(obs, roles, profile(4))
        self.assertEqual(FX.findings_of(out, 'wide_defender_advance'), [])
        self.assertEqual(FX.status_of(out, 'wide_defender_advance')['reason'], 'too_few_situations')

    def test_the_moment_names_its_own_players(self):
        """Fix round 3 (FACTS F-2): the wide defender and the back-line pair named are the first cited moment's, not
        the family's most frequent (measurements actorId / gapLeftId / gapRightId; players.primary the same)."""
        opps = [advance(t) for t in spaced(12)]
        first = opps[0]
        first['actors']['primary'] = ['A-07']
        first['roles'] = {'A-07': {'role': 'LB', 'source': 'stable', 'confidence': 'high'}}
        obs, roles = files(opps)
        f = FX.findings_of(AS.assess(obs, roles, profile(4)), 'wide_defender_advance')[0]
        self.assertEqual(f['timestamps'][0]['opportunity'], first['id'])
        self.assertEqual((f['players']['primary'], f['measurements']['actorId'], f['roles']['A-07']['role']),
                         (['A-07'], 'A-07', 'LB'))
        self.assertEqual(f['measurements']['momentZone'], 'final_third')     # ballX 80 m at the moment

    def test_the_back_line_pair_is_the_moments(self):
        def d5(t, left, right, prim):
            return make_opp('defensive_spacing', t, {
                'lineStdM': 1.0, 'lineSize': 4, 'lateralGapM': 25.0, 'gapLeft': left, 'gapRight': right,
                'gapBallSide': True, 'opponentInGap': True, 'gapOpponents': 1, 'dmGapM': 10.0, 'dmGapFree': 0,
                'gapOppFreeM': 9.0, 'primaryNamed': True}, primary=(prim,))
        opps = [d5(300.0, 'A-04', 'A-02', 'A-02')]+[d5(t, 'A-02', 'A-03', 'A-03') for t in spaced(11, 320.0)]
        obs, roles = files(opps)
        f = FX.findings_of(AS.assess(obs, roles, profile(5)), 'defensive_spacing')[0]
        self.assertEqual(f['timestamps'][0]['opportunity'], opps[0]['id'])
        m = f['measurements']
        self.assertEqual((m['gapLeftId'], m['gapRightId'], m['actorId'], f['players']['primary'],
                          f['players']['supporting'][:1]), ('A-04', 'A-02', 'A-02', ['A-02'], ['A-04']))


class NextObjectiveEvidence(unittest.TestCase):
    """Fix round 2, critic S-8: the next objective (no work item) comes from a check this range decided often enough
    (the level's strengthSpells) and the team did not show; never from unclear runs."""

    def test_unclear_runs_never_choose_the_objective(self):
        # the professional-like scene after fix round 1: 3 behind-the-ball opportunities demonstrated for 2.4 s only
        short = [back(t, 3, n=12) for t in spaced(3)]
        obs, roles = files(short)
        for level in (1, 2, 3):
            out = AS.assess(obs, roles, profile(level))
            self.assertNotEqual(out['nextObjective']['skill'], 'someone_back_when_attacking', level)
        # no check decided enough: no objective from the match, the level's suggested focus instead (polish round)
        no = AS.assess(obs, roles, profile(2))['nextObjective']
        self.assertTrue(AS.is_level_focus(no), no)
        self.assertEqual((no['fromMatch'], no['finding']), (False, None))
        self.assertNotEqual(no['skill'], 'someone_back_when_attacking')

    def test_a_decided_check_the_team_did_not_show(self):
        def d1(t, failing):
            return make_opp('pressure_cover', t, {'d1': 2.0, 'closingMS': 0.5, 'coverDist': 18.0 if failing else 6.0,
                                                  'coverAngle': 30.0, 'coverVisible': True, 'coverId': 'A-06',
                                                  'carrierX': 20.0, 'goalSideShare': .7, 'primaryNamed': True,
                                                  'fd': 'A-03'})
        short = [back(t, 3, n=12) for t in spaced(3)]
        both = [d1(t, k % 2 == 0) for k, t in enumerate(spaced(4, 300.0))]+[d1(390.0, True)]
        obs, roles = files(short+both)                  # 2 of the 5 decided shown, under level 2's .5
        out = AS.assess(obs, roles, profile(2))
        self.assertEqual((out['nextObjective']['skill'], out['nextObjective']['fromMatch']),
                         ('pressure_and_cover', False))
        shown = [d1(t, False) for t in spaced(5, 300.0)]
        obs, roles = files(short+shown)
        no = AS.assess(obs, roles, profile(2))['nextObjective']
        # pressure and cover shown 5 of 5: never the focus (polish round: the level's general objective here)
        self.assertTrue(AS.is_level_focus(no), no)
        self.assertNotEqual(no['skill'], 'pressure_and_cover')

    def test_phase_unknown_shape_never_chooses_the_objective(self):
        # pre-freeze (FACTS F-9, second half): the objective "Chosen from this match's checks" read attacking shape
        # from moments when nobody was known to have the ball (AM A / B levels 1-5; here possession_unknown at levels
        # 4-5, every one of its attacking-shape opportunities the 'unknown' variant). It is chosen from the check's own
        # phase only; a phased shape check still chooses it (kids_swarm at levels 4-5).
        obs, roles = FX.tiers('possession_unknown', 'default')
        shape = [o for o in obs['opportunities'] if o['detector'] == 'attacking_shape' and o['abstain'] is None]
        self.assertTrue(shape)
        self.assertEqual({o.get('variant') for o in shape}, {'unknown'})
        for level in (4, 5):
            no = AS.assess(obs, roles, FX.profile('possession_unknown', 'default', level))['nextObjective']
            with self.subTest(scene='possession_unknown', level=level):
                self.assertNotEqual(no['skill'], 'spread_when_attacking', no)
        obs, roles = FX.tiers('kids_swarm', 'default')
        for level in (4, 5):
            no = AS.assess(obs, roles, FX.profile('kids_swarm', 'default', level))['nextObjective']
            with self.subTest(scene='kids_swarm', level=level):
                self.assertEqual((no['skill'], no['fromMatch'], AS.is_level_focus(no)),
                                 ('spread_when_attacking', False, False))


class DepthDescribedAtLevels1To3(unittest.TestCase):
    """Fix round 2, LEAD decision (c) (the coach's recommendation): 11v11 attacking depth is descriptive only at
    levels 1-3; width and the channels are still judged, small formats still judge depth."""

    @staticmethod
    def shape(t, width=.7, depth=.2):
        return make_opp('attacking_shape', t, {'widthShare': width, 'depthShare': depth, 'channelsOccupied': 3,
                                               'depthAvailShare': .5, 'widthM': 40.0, 'depthM': 20.0})

    def test_depth_alone_is_never_criticised_on_11v11(self):
        flat = [self.shape(t) for t in spaced(12)]
        obs, roles = files(flat)
        for level in (1, 2, 3):
            out = AS.assess(obs, roles, profile(level))
            self.assertEqual(FX.findings_of(out, 'attacking_shape', S.CRITICISM+('review',)), [], level)
            c = AS.classify(flat[0], profile(level)['detectors']['attacking_shape']['params'], SCALE, STEP)
            self.assertEqual(c['outcome'], 'unclear')
        self.assertEqual(AS.classify(flat[0], profile(4)['detectors']['attacking_shape']['params'], SCALE, STEP)
                         ['outcome'], 'not_demonstrated')
        small = AS.assess(obs, roles, profile(2, players=7))
        self.assertEqual([(f['verdict'], f['measurements']['failureMode'])
                          for f in FX.findings_of(small, 'attacking_shape')], [('try_this', 'depth')])

    def test_narrow_and_flat_is_criticised_for_width(self):
        obs, roles = files([self.shape(t, width=.3) for t in spaced(12)])
        f = FX.findings_of(AS.assess(obs, roles, profile(2)), 'attacking_shape')
        self.assertEqual([(x['verdict'], x['measurements']['failureMode']) for x in f], [('try_this', 'width')])


def sha(x):
    return hashlib.sha256(json.dumps(x, sort_keys=True, allow_nan=False).encode()).hexdigest()


class Scenes(unittest.TestCase):
    """DESIGN 13.1 rows end to end (tools/coaching_synth.py; levels 1-7, style unknown unless said)."""

    def run_levels(self, scene, variant=None, levels=range(1, 8), style=None):
        obs, roles = FX.tiers(scene, variant)
        out = {}
        for lv in levels:
            res = AS.assess(obs, roles, FX.profile(scene, variant, lv, style=style))
            for f in res['findings']:
                self.assertEqual(S.validate('finding', f), [], (scene, lv, f['key']))
            out[lv] = res
        return obs, roles, out

    def test_covered_advances_are_never_criticised(self):
        for variant in ('cb_shift', 'channel'):
            _, _, out = self.run_levels('covered_advance', variant)
            for lv, res in out.items():
                self.assertEqual(FX.findings_of(res, 'wide_defender_advance', ('work_on', 'try_this', 'review')), [])
                if lv >= 3:
                    self.assertEqual([f['verdict'] for f in FX.findings_of(res, 'wide_defender_advance')],
                                     ['strength'], (variant, lv))

    def test_unsupported_advances(self):
        obs, roles, out = self.run_levels('unsupported_advance', 'none_behind')
        # the planted advances happen with the near centre-back carrying the ball into midfield and B-10 standing in
        # the vacated channel: levels 1-2 judge "someone back" with the ball in the final third only (critic B-1), but
        # an uncovered advance with an opponent in the channel wherever the ball is (fix round 3, critic B-1 / COACH
        # F-1: the spec's EARLY DEVELOPMENT example) ...
        f1 = FX.findings_of(out[1], 'behind_ball', ('try_this',))
        self.assertEqual([(f['key'].split('|')[2], f['measurements']['failureMode']) for f in f1],
                         [('advance', 'exposed')])
        f2 = FX.findings_of(out[2], 'wide_defender_advance', ('try_this',))
        self.assertEqual([f['measurements']['failureMode'] for f in f2], ['exposed'])
        # ... and the same advances with the ball moved into the final third too
        high = copy.deepcopy(roles)
        for o in high['opportunities']:
            o['relations']['ballX'] = [80.0]*len(o['relations']['t'])
        f1 = FX.findings_of(AS.assess(obs, high, FX.profile('unsupported_advance', 'none_behind', 1)), 'behind_ball',
                            ('try_this',))
        self.assertTrue(any(f['key'].split('|')[2] == 'advance' for f in f1))       # L1: someone ready to protect
        self.assertEqual([f['verdict'] for f in FX.findings_of(
            AS.assess(obs, high, FX.profile('unsupported_advance', 'none_behind', 2)), 'wide_defender_advance')],
            ['try_this'])
        for lv in (4, 5, 6):
            self.assertEqual([f['verdict'] for f in FX.findings_of(out[lv], 'wide_defender_advance')], ['work_on'])
        evidence = {lv: [t['evidence'] for t in FX.findings_of(out[lv], 'wide_defender_advance')[0]['timestamps']]
                    for lv in range(3, 8)}
        self.assertEqual(len({tuple(v) for v in evidence.values()}), 1)              # same evidence at every level
        # 4 clear unsupported advances in 6.1 minutes: never a verdict, but from level 3 'moments worth a look' (its
        # clear advances are their own floor: critic S-2 / COACH F-14), never "the range is too short"
        _, _, out = self.run_levels('unsupported_advance', 'spells4')
        for lv, res in out.items():
            self.assertEqual(FX.findings_of(res, 'wide_defender_advance', S.CRITICISM), [], lv)
            if lv >= 3:
                f = mwl(res, 'wide_defender_advance')
                self.assertEqual(len(f), 1, lv)
                m = f[0]['measurements']
                self.assertEqual((m['clearSpells'], m['decidedMoments'], f[0]['verdict']), (4, 4, 'review'), lv)
                self.assertEqual(FX.status_of(res, 'wide_defender_advance')['status'], 'assessed')

    def test_one_instance_identity_crossing_and_sprint_back(self):
        _, _, out = self.run_levels('single_advance')
        for lv in range(2, 8):
            # one advance in 6.1 minutes: below half the level's minimum opportunities, too few situations to judge it
            # (S-1; fix round 3, critic S-2: the range is not short), and one clear advance is not three
            self.assertEqual(FX.findings_of(out[lv], 'wide_defender_advance'), [], lv)
            self.assertEqual((FX.status_of(out[lv], 'wide_defender_advance')['status'],
                              FX.status_of(out[lv], 'wide_defender_advance')['reason']),
                             ('not_assessed', 'too_few_situations'), lv)
        _, _, out = self.run_levels('identity_crossing')
        for lv, res in out.items():
            for f in FX.findings_of(res, 'wide_defender_advance'):
                self.assertNotIn(f['verdict'], S.CRITICISM)
                self.assertEqual(f['guardrails']['identity'], 'fail')
        _, _, out = self.run_levels('sprint_back')
        for res in out.values():
            self.assertEqual(FX.findings_of(res, 'wide_defender_advance', ('work_on', 'try_this', 'review')), [])

    def test_wrong_formation_and_formation_change(self):
        _, _, out = self.run_levels('wrong_formation')
        for res in out.values():
            for f in FX.findings_of(res, 'wide_defender_advance'):
                self.assertNotIn(f['verdict'], S.CRITICISM)
        _, _, out = self.run_levels('wrong_formation', 'coach_roles', levels=(4,))
        self.assertTrue(all(f['guardrails']['role'] == 'pass' for f in FX.findings_of(out[4], 'wide_defender_advance')))
        obs, roles, out = self.run_levels('formation_change')
        near = {o['id'] for o in roles['opportunities'] if o['intentSignals']['formationChangeNear']}
        self.assertTrue(near)
        for res in out.values():
            for f in FX.findings_of(res, 'wide_defender_advance', ('work_on', 'try_this', 'review')):
                self.assertFalse(near & {t['opportunity'] for t in f['timestamps']})

    def test_nothing_invented_on_a_short_clip_or_a_professional_like_team(self):
        for scene, variant in (('short_segment', None), ('professional', None), ('professional', 'no_advances')):
            _, _, out = self.run_levels(scene, variant)
            for lv, res in out.items():
                self.assertEqual([f['key'] for f in res['findings'] if f['verdict'] in S.CRITICISM], [],
                                 (scene, variant, lv))
        _, _, out = self.run_levels('professional', 'no_advances', levels=(4,))
        self.assertEqual(FX.findings_of(out[4], 'wide_defender_advance', ('strength',)), [])

    def test_unavailable_categories_follow_the_phase_source(self):
        cats = {}
        for scene, variant in (('possession_unknown', None), ('amateur_coach_spans', 'spans'),
                               ('amateur_coach_spans', 'no_spans'), ('covered_advance', 'cb_shift')):
            obs, _ = FX.tiers(scene, variant)
            got = AS.unavailable(obs, FX.profile(scene, variant, 4))
            for u in got:
                self.assertEqual(S.validate('unavailable', u) if 'unavailable' in S.KINDS else [], [])
            cats[(scene, variant)] = {u['category'] for u in got}
        self.assertIn('phase_amateur', cats[('possession_unknown', None)])
        self.assertIn('phase_amateur', cats[('amateur_coach_spans', 'no_spans')])
        self.assertNotIn('phase_amateur', cats[('amateur_coach_spans', 'spans')])     # the coach named the phase
        self.assertNotIn('phase_amateur', cats[('covered_advance', 'cb_shift')])

    def test_without_a_phase_nothing_is_assessed_but_the_shape_described(self):
        _, _, out = self.run_levels('possession_unknown', levels=(1, 4, 7))
        for res in out.values():
            for d in ('pressure_cover', 'wide_defender_advance', 'behind_ball'):
                st = FX.status_of(res, d)
                self.assertIn(st['status'], ('not_assessed', 'off'))
                if st['status'] == 'not_assessed':
                    self.assertEqual(st['reason'], 'phase_unknown')
            self.assertTrue(all(f['verdict'] == 'describe' for f in res['findings']))

    def test_coach_marked_amateur_clip(self):
        _, _, out = self.run_levels('amateur_coach_spans', 'spans', levels=(4,))
        st = {d['detector']: d for d in out[4]['detectors']}
        for d in ('wide_defender_advance', 'defensive_spacing', 'attacking_shape'):
            self.assertEqual(st[d]['status'], 'assessed', d)
        self.assertEqual(st['pressure_cover']['reason'], 'no_carrier')
        self.assertEqual((st['weak_side_balance']['reason'], st['behind_ball']['reason']),
                         ('ball_unreliable', 'ball_unreliable'))
        found = out[4]['findings']
        self.assertTrue(found)
        for f in found:                         # findings resting on coach-marked phases say so
            self.assertEqual((f['measurements']['phaseSource'], f['context']['phase']['estimated']), ('coach', False))

    def test_planted_defending_outcomes(self):
        want = {'press_cover': ('pressure_cover', 'demonstrated'), 'press_no_cover': ('pressure_cover', 'failing'),
                'no_press': ('pressure_cover', 'failing'), 'weak_empty': ('weak_side_balance', 'failing'),
                'gap_open': ('defensive_spacing', 'failing')}
        for variant, (det, outcome) in want.items():
            obs, roles, out = self.run_levels('defending', variant, levels=(4,))
            prof = FX.profile('defending', variant, 4)
            got = [AS.classify(o, prof['detectors'][det]['params'], obs['timeline']['scale'], .2)
                   for o in obs['opportunities'] if o['detector'] == det]
            d = sum(1 for c in got if c and c['outcome'] == 'demonstrated')
            nd = sum(1 for c in got if c and c['outcome'] == 'not_demonstrated')
            self.assertTrue(d > nd if outcome == 'demonstrated' else nd >= 5, (variant, d, nd))
            if det != 'defensive_spacing':          # the gap opens half of each spell: 7 of 25 opportunities fail,
                self.assertTrue(FX.findings_of(out[4], det), variant)  # below level 4's 40%: no finding there

    def test_profiles_never_change_the_observations_and_a_level_change_is_fast(self):
        obs, roles = FX.tiers('unsupported_advance', 'none_behind')
        s0, r0 = sha(obs), sha(roles)
        worst = 0.0
        for lv in range(1, 8):
            for style in S.STYLES:
                a = time.perf_counter()
                AS.assess(obs, roles, FX.profile('unsupported_advance', 'none_behind', lv, style=style))
                worst = max(worst, time.perf_counter()-a)
        self.assertEqual((sha(obs), sha(roles)), (s0, r0))
        self.assertLess(worst, .3)          # DESIGN 1.2: an interpretation rebuild under 300 ms (6-min scene here)


if __name__ == '__main__':
    unittest.main()
