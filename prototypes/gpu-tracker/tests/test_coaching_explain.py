"""Coaching explanations v2 (soccer/coaching/explain.py), against hand-made findings that follow schema.py and, from
wave 5, against the REAL findings of EVAL's scenes through every stage (RealFindingTests; tests/
coaching_text_fixtures.py). The first run on real findings found what the hand-made ones never produced: the text of
an attacking-shape finding said "narrow" when the depth had failed, counts of 0 read "0 of our players", values at one
moment were called averages, a description said "it stays a review", a team-level check named "our left defender",
"the centre-back line" read as a role word, "left" (a side) as a past-tense verb, and a metre-free level-4 text was
missing; each now has a test below (AttackingShapeModeTests, WaveFiveWordingTests).

What is pinned here and why:
- every detector x allowed verdict x level x child/adult age gives a schema-valid finding whose text passes the gate
  against its own claims (numbers, times, ids, roles, phase qualifier, number mode, concepts, bans, polarity,
  hypotheticals, unmeasured events), so a register can never print a word it is not allowed;
- the levels differ in substance, not only wording: level 1 has no numbers, no player named and no "cover"; level 2
  names no player and shows counts only; shares appear from level 5 and rates from level 6; the wide-defender family
  carries at least 2 more relation/opponent claims at levels 5-7 than at 2-3 (DESIGN 13.4 #5); explanations of one
  finding at two levels stay below .5 word-trigram Jaccard while citing the same evidence (Part 16);
- inside one report no two explanations exceed .5 Jaccard and no slot phrasing is used twice (DESIGN 6.4);
- the gate catches what a careless engine would write: a flipped relation ("covered" when the claim says not
  covered), an invented number, a player id at level 2, a role word without a confirmed role, "exposed" outside a
  hypothetical, "transition" at level 4, "if the ball had been lost" at level 2, a phase without its qualifier or with
  the wrong one, "repeated" in a short range, generic sentences, "mistake", "UEFA", "lost the ball";
- coach-marked phases (user decision Oct 21) read "(coach-marked)", mixed ones "(estimated, partly coach-marked)";
- an optional engine sees only the claims and the register, and its text is used only when it passes; when nothing
  passes the finding is withheld with a reason; a criticism without a teammate relation claim is withheld;
- principle claims cite a source only when the profile lists it (never an invented id);
- the spec's left-back examples are style references only: no explanation comes within .5 Jaccard of any of them.

Run:  python -m unittest test_coaching_explain          (from this directory)
"""
import copy
import itertools
import os
import re
import sys
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from soccer.coaching import explain as E  # noqa: E402
from soccer.coaching import language as L  # noqa: E402
from soccer.coaching import schema as S  # noqa: E402

LEVELS = range(1, 8)
TYPICAL_AGE = {1: 'U8', 2: 'U10', 3: 'U12', 4: 'U14', 5: 'U16', 6: 'Adult', 7: 'Adult'}
ADULT_AGE = 'Adult'
VERDICTS = {lvl: [v for v in S.FINDING_VERDICTS if not (v == 'work_on' and lvl <= 2) and
                  not (v == 'try_this' and lvl >= 4)] for lvl in LEVELS}
NAMING = {1: 'never', 2: 'relative', 3: 'role_words', 4: 'evidence_only', 5: 'yes', 6: 'yes', 7: 'yes'}
BANNED = {1: ['mistake', 'balance', 'cover', 'press', 'pressing', 'rest defence', 'transition'],
          2: ['mistake', 'balance', 'rest defence', 'transition', 'trigger'], 3: ['mistake', 'rest defence', 'trigger'],
          4: ['mistake', 'trigger'], 5: ['mistake'], 6: [], 7: []}
SOURCES = ['EF-COVER-RECOVER', 'US_GM_U1314_PDF', 'USS-GM1516', 'EF-PTC', 'US-GM-U78-PDF']
PHASE = {'pressure_cover': 'opp', 'weak_side_balance': 'opp', 'wide_defender_advance': 'own', 'behind_ball': 'own',
         'defensive_spacing': 'opp', 'attacking_shape': 'own'}
BALL = {'pressure_cover': ('own_third', 'left'), 'weak_side_balance': ('own_third', 'right'),
        'wide_defender_advance': ('middle_third', 'left'), 'behind_ball': ('final_third', 'centre'),
        'defensive_spacing': ('own_third', 'left'), 'attacking_shape': ('middle_third', 'centre')}
PLAYERS = {
    'pressure_cover': ({'primary': ['A-04'], 'supporting': ['A-05', 'A-08'], 'opponents': ['B-10']},
                       {'A-04': ('LCB', 'stable'), 'A-05': ('CDM', 'stable')}),
    'weak_side_balance': ({'primary': ['A-03'], 'supporting': ['A-05'], 'opponents': []},
                          {'A-03': ('RB', 'stable')}),
    'wide_defender_advance': ({'primary': ['A-02'], 'supporting': ['A-05', 'A-06'], 'opponents': ['B-11']},
                              {'A-02': ('LB', 'coach'), 'A-05': ('LCB', 'stable'), 'A-06': ('CDM', 'stable')}),
    'behind_ball': ({'primary': [], 'supporting': ['A-04', 'A-05'], 'opponents': ['B-09']}, {}),
    'defensive_spacing': ({'primary': ['A-04'], 'supporting': ['A-03'], 'opponents': ['B-07']},
                          {'A-04': ('LCB', 'stable'), 'A-03': ('LB', 'stable')}),
    'attacking_shape': ({'primary': [], 'supporting': [], 'opponents': []}, {}),
}
FAIL_M = {
    'pressure_cover': {'failureMode': 'no_cover', 'pressurePresent': True, 'coverPresent': False, 'd1M': 3.1,
                       'coverDistM': 14.2, 'coverAngleDeg': 72.0, 'coverPresentShare': .33, 'goalSideShare': .6,
                       'metres': 'estimated'},
    'weak_side_balance': {'weakCount': 0, 'centralCount': 2, 'openWidthM': 24.6, 'metres': 'estimated'},
    'wide_defender_advance': {'side': 'left', 'advanceM': 21.3, 'behindN': 2, 'channelOccupied': False,
                              'cbAdjusted': False, 'nearCbShiftM': 1.2, 'dmDropped': False, 'interchange': False,
                              'opponentsInChannel': 1, 'threatN': 3, 'restMargin': -1, 'dmCbSpacingM': 17.8,
                              'metres': 'estimated'},
    'behind_ball': {'behindN': 1, 'minBehind': 2, 'threatN': 2, 'restMargin': -1, 'metres': 'estimated'},
    'defensive_spacing': {'failureMode': 'lateral_gap', 'lateralGapM': 23.4, 'gapBallSide': True,
                          'opponentInGap': True, 'dmGapM': 19.2, 'lineStdM': 2.1, 'blockHeight': 'mid',
                          'metres': 'estimated'},
    'attacking_shape': {'widthShare': .48, 'depthShare': .27, 'widthM': 32.6, 'depthM': 28.4, 'channelsOccupied': 2,
                        'metres': 'estimated'},
}
OK_M = {
    'pressure_cover': {'pressurePresent': True, 'coverPresent': True, 'd1M': 2.6, 'coverDistM': 7.9,
                       'coverAngleDeg': 31.0, 'coverPresentShare': .78, 'metres': 'estimated'},
    'weak_side_balance': {'weakCount': 2, 'centralCount': 3, 'metres': 'estimated'},
    'wide_defender_advance': {'side': 'left', 'advanceM': 19.6, 'behindN': 3, 'channelOccupied': True,
                              'channelBy': 'A-06', 'cbAdjusted': True, 'nearCbShiftM': 6.1, 'dmDropped': True,
                              'interchange': False, 'opponentsInChannel': 0, 'restMargin': 1, 'threatN': 2,
                              'metres': 'estimated'},
    'behind_ball': {'behindN': 3, 'minBehind': 2, 'threatN': 2, 'restMargin': 1, 'metres': 'estimated'},
    'defensive_spacing': {'lineStdM': 1.6, 'dmGapM': 11.4, 'blockHeight': 'mid', 'metres': 'estimated'},
    'attacking_shape': {'widthShare': .72, 'depthShare': .36, 'widthM': 49.0, 'depthM': 37.8, 'channelsOccupied': 3,
                        'metres': 'estimated'},
}
FAMILY = {'pressure_cover': 'own_third', 'weak_side_balance': 'right', 'wide_defender_advance': 'LB-side',
          'behind_ball': 'team', 'defensive_spacing': 'own_third', 'attacking_shape': 'team'}
T0 = 760.0
FPS = 30.0


SMALL_FORMATS = {'4v4': (4, False, 4), '5v5': (5, True, 4), '7v7': (7, True, 6), '9v9': (9, True, 8)}


def profile_for(level, age=None, sources=SOURCES, fmt=None, **register):
    p = S.example('profile')
    p.update(level=level, levelName=S.LEVEL_NAMES[level], ageGroup=age or TYPICAL_AGE[level], sources=list(sources))
    if fmt:
        players, gk, outfield = SMALL_FORMATS[fmt]
        p['format'] = {'players': players, 'goalkeeper': gk, 'outfield': outfield, 'name': fmt}
    reg = E.REGISTERS[level]
    p['register'] = {'tone': reg['tone'], 'numbers': reg['numbers'], 'maxChallenges': reg['maxChallenges'],
                     'concepts': [], 'banned': list(BANNED[level])}
    p['register'].update(register)
    p['evidence']['naming'] = NAMING[level]
    return p


def finding_for(detector, verdict, level, *, one_spell=False, measurements=None, identity='pass', source=None,
                limitations=(), style='unknown', style_effect=None, roles=None, n=9, t0=T0, team='A',
                phase_seconds=188.2, fmt='11v11', phase=None):
    """A schema-valid finding of one detector, shaped after DESIGN 6.1 and the measurement keys of
    explain.MEASUREMENT_KEYS."""
    fail = verdict in ('try_this', 'work_on', 'review')
    m = copy.deepcopy(measurements if measurements is not None else (FAIL_M if fail else OK_M)[detector])
    if verdict == 'describe' and measurements is None:
        m = copy.deepcopy(OK_M[detector])
    if detector == 'pressure_cover' and level == 1 and fail and measurements is None:
        m.update(failureMode='no_pressure', pressurePresent=False)
    if detector == 'wide_defender_advance' and level <= 2 and fail and measurements is None:
        m['behindN'] = 0             # levels 1-2 judge the advance team-level: nobody behind the ball (DESIGN 4.3)
    players, role_map = PLAYERS[detector]
    if roles is not None:
        role_map = roles
    t1 = round(t0+14.0*(n-1)+3.6, 1)
    key = S.finding_key(detector, team, FAMILY[detector], t0, t1)
    fid = S.record_id('fnd', key, level)
    opps = [S.record_id('opp', key, k) for k in range(n)]
    stamps = []
    for k in range(min(3, n)):
        t = round(t0+14.0*k+1.0, 1)
        frame = int(round(t*FPS))
        stamps.append({'evidence': S.record_id('ev', opps[k], frame), 'opportunity': opps[k], 't': t, 'frame': frame,
                       'seek': S.seek_of(frame, FPS)})
    if verdict == 'strength':
        agg = {'opportunities': n, 'demonstrated': n-2, 'notDemonstrated': 1, 'unclear': 1, 'distinctFailingSpells': 1,
               'distinctDemonstratedSpells': 4}
    else:
        agg = {'opportunities': n, 'demonstrated': 3, 'notDemonstrated': n-4, 'unclear': 1,
               'distinctFailingSpells': 1 if one_spell else 3, 'distinctDemonstratedSpells': 2}
    agg.update(phaseSeconds=phase_seconds, eligibleShare=.39, ratePer10=3.4)
    zone, channel = BALL[detector]
    g = {k: 'pass' for k in S.GUARDRAILS}
    g['identity'] = identity
    if verdict == 'describe':
        g['alternative'] = 'not_applicable'
    f = {'id': fid, 'key': key, 'observation': detector, 'verdict': verdict, 'level': level, 'team': team,
         'priority': 1 if verdict in ('work_on', 'try_this') else None,
         'players': copy.deepcopy(players),
         'roles': {pid: {'role': r, 'source': src, 'confidence': 'high' if src == 'coach' else 'medium'}
                   for pid, (r, src) in role_map.items() if pid in players['primary']+players['supporting']},
         'timestamps': stamps, 'opportunities': opps,
         'context': {'phase': {'state': phase or PHASE[detector], 'estimated': source != 'coach', 'coverage': .79},
                     'ball': {'zone': zone, 'channel': channel}, 'style': style, 'styleEffect': style_effect,
                     'format': fmt},
         'measurements': m, 'aggregate': agg,
         'coverage': {'identity': 'high', 'position': 'high', 'visibility': 'complete', 'ball': 'reliable',
                      'calibrationTier': 'world', 'eligibleShareOfPhase': .39},
         'guardrails': g, 'limitations': list(limitations), 'suggestedAction': None, 'claims': [], 'text': None,
         'drills': []}
    if verdict == 'review' and identity == 'pass' and not limitations:
        f['guardrails']['role'] = 'fail'
    return f


def obs_for(findings, source='possession-4', mixed=False):
    """A minimal obs-like dict carrying the findings' opportunities with a phase source (coach-marked spans)."""
    opps = []
    for f in findings:
        for k, oid in enumerate(f['opportunities']):
            src = source if not (mixed and k % 2) else 'possession-4'
            opps.append({'id': oid, 'phase': {'state': f['context']['phase']['state'], 'source': src,
                                              'estimated': src != 'coach', 'spanSeconds': 9.0, 'nearChange': False}})
    return {'opportunities': opps, 'segment': {'t0': 700.0, 't1': 1000.0}}


def alternative_for(f, mover='A-05', display='zone'):
    ts = f['timestamps'][0]
    return {'id': S.record_id('alt', f['id'], ts['evidence'], mover), 'finding': f['id'], 'evidence': ts['evidence'],
            't': ts['t'], 'mover': mover, 'principle': E.DEFAULT_PRINCIPLE[f['observation']],
            'zone': {'polygon': [[.17, .36], [.21, .36], [.21, .40]], 'areaM2': 14.2, 'kind': 'zone', 'radiusM': None},
            'display': display, 'from': [.18, .45], 'reach': {'horizonS': 2.0, 'speedMS': 5.6, 'radiusM': 11.2},
            'improves': {'relation': 'cover', 'before': False, 'after': True, 'shareOfZone': .9},
            'checks': {'pressureKept': True, 'centralKept': True, 'gapOk': True, 'noNewFreeOpponent': True,
                       'noStacking': True}, 'assumption': 'single_mover', 'rejected': []}


def run(findings, profile, **kw):
    kw.setdefault('obs', obs_for(findings))
    kw.setdefault('range', {'t0': 700.0, 't1': 1000.0})
    return E.explain(findings, profile, **kw)


def all_cases(ages=('typical', 'adult')):
    for det in S.DETECTORS:
        for lvl in LEVELS:
            for verdict in VERDICTS[lvl]:
                for age in ages:
                    yield det, lvl, verdict, (TYPICAL_AGE[lvl] if age == 'typical' else ADULT_AGE)


class EveryCaseTests(unittest.TestCase):
    def test_every_detector_verdict_level_and_age_passes_the_gate(self):
        for det, lvl, verdict, age in all_cases():
            with self.subTest(detector=det, level=lvl, verdict=verdict, age=age):
                prof = profile_for(lvl, age)
                f = finding_for(det, verdict, lvl)
                alts = [alternative_for(f)] if verdict in ('work_on', 'try_this', 'review') else []
                log = []
                out = run([f], prof, alternatives=alts, log=log)
                self.assertEqual(out['withheld'], [], log)
                g = out['findings'][0]
                self.assertEqual(S.validate('finding', g), [])
                self.assertEqual(E.validate(g['text'], g['claims'], g, prof, short=False), [])
                self.assertEqual(log[0][1], 'planned', g['text'])
                self.assertLessEqual(len(g['text']['short']), S.CAPTION_MAX)
                for field in E.TEXT_FIELDS:
                    self.assertEqual(S.event_problems(g['text'][field] or ''), [])

    def test_one_spell_reviews_and_short_ranges_pass(self):
        for det in S.DETECTORS:
            for lvl in LEVELS:
                with self.subTest(detector=det, level=lvl):
                    prof = profile_for(lvl)
                    f = finding_for(det, 'review', lvl, one_spell=True)
                    out = run([f], prof, range={'t0': 700.0, 't1': 760.0})
                    self.assertEqual(out['withheld'], [])
                    text = out['findings'][0]['text']
                    for w in L.SHORT_BANNED:
                        self.assertNotRegex(' '.join(v for v in text.values() if v).lower(), r'\b'+re.escape(w)+r'\b')

    def test_the_input_finding_is_not_changed(self):
        f = finding_for('pressure_cover', 'work_on', 4)
        before = copy.deepcopy(f)
        run([f], profile_for(4))
        self.assertEqual(f, before)

    def test_deterministic(self):
        fs = [finding_for(d, 'work_on', 5) for d in S.DETECTORS]
        a = run(copy.deepcopy(fs), profile_for(5))
        b = run(copy.deepcopy(fs), profile_for(5))
        self.assertEqual(a, b)


class LevelTests(unittest.TestCase):
    def texts(self, det, verdict, lvl, **kw):
        f = finding_for(det, verdict, lvl, **kw)
        out = run([f], profile_for(lvl), alternatives=[alternative_for(f)])
        self.assertEqual(out['withheld'], [])
        return out['findings'][0]

    def test_level_one_has_no_numbers_no_names_no_cover(self):
        for det in S.DETECTORS:
            for verdict in VERDICTS[1]:
                g = self.texts(det, verdict, 1)
                body = ' '.join(v for v in g['text'].values() if v)
                with self.subTest(detector=det, verdict=verdict):
                    self.assertIsNone(re.search(r'\d', L.TIME_TOKEN.sub('', body)), body)
                    self.assertIsNone(L.ID_TOKEN.search(body), body)
                    self.assertNotRegex(body.lower(), r'\bcover|\bpress|\bbalanc|\btransition|\bchannel')
                    for pid in ('left back', 'centre-back', 'midfielder'):
                        self.assertNotIn(pid, body.lower())

    def test_level_one_try_this_is_one_question(self):
        for det in ('pressure_cover', 'behind_ball', 'attacking_shape', 'wide_defender_advance'):
            g = self.texts(det, 'try_this', 1)
            with self.subTest(detector=det):
                self.assertEqual(sum(v.count('?') for v in g['text'].values() if v), 1)
                self.assertTrue(g['text']['action'].endswith('?'))
                self.assertTrue(g['text']['headline'].startswith('Try this:'))

    def test_adult_register_says_next_step(self):
        f = finding_for('behind_ball', 'try_this', 1)
        g = run([f], profile_for(1, 'Adult'))['findings'][0]
        self.assertTrue(g['text']['headline'].startswith('Next step:'), g['text']['headline'])

    def test_level_two_counts_only_and_relative_words(self):
        for det in S.DETECTORS:
            for verdict in VERDICTS[2]:
                g = self.texts(det, verdict, 2)
                body = ' '.join(v for v in g['text'].values() if v)
                with self.subTest(detector=det, verdict=verdict):
                    self.assertIsNone(L.ID_TOKEN.search(body), body)
                    self.assertIsNone(E.METRE_RE.search(body), body)
                    self.assertNotIn('%', body)
        g = self.texts('wide_defender_advance', 'try_this', 2)
        self.assertIn('our left defender', g['text']['explanation'])

    def test_number_modes_by_level(self):
        for lvl in LEVELS:
            g = self.texts('wide_defender_advance', 'work_on' if lvl >= 3 else 'try_this', lvl)
            body = ' '.join(v for v in g['text'].values() if v)
            with self.subTest(level=lvl):
                if lvl < 5:
                    self.assertNotIn('%', body)
                if lvl < 6:
                    self.assertIsNone(E.RATE_RE.search(body), body)
                if lvl >= 3:
                    self.assertTrue(E.METRE_RE.search(body), body)
        # rates from 5 minutes of the phase only (fix round 1, COACH L5: false precision from 2.3 minutes)
        rates = [self.texts('pressure_cover', 'work_on', lvl, phase_seconds=420.0)['text']['explanation']
                 for lvl in (6, 7)]
        self.assertTrue(any(E.RATE_RE.search(t) for t in rates), rates)
        shares = self.texts('pressure_cover', 'work_on', 5)['text']['explanation']
        self.assertIn('%', shares)

    def test_estimated_metres_say_about(self):
        for lvl in (3, 4, 5, 6, 7):
            g = self.texts('pressure_cover', 'work_on', lvl)
            for m in E.METRE_RE.finditer(g['text']['explanation']):
                self.assertRegex(g['text']['explanation'][:m.start()], r'about\s*$')

    def test_ids_only_from_level_four_and_never_in_headlines(self):
        for lvl in LEVELS:
            g = self.texts('pressure_cover', 'work_on' if lvl >= 3 else 'try_this', lvl)
            with self.subTest(level=lvl):
                self.assertIsNone(L.ID_TOKEN.search(g['text']['headline']))
                self.assertIsNone(L.ID_TOKEN.search(g['text']['short']))
                if lvl <= 3:
                    self.assertIsNone(L.ID_TOKEN.search(g['text']['explanation']))
                else:
                    self.assertIn('A-04', g['text']['explanation'])
                if lvl == 4:
                    self.assertIsNone(L.ID_TOKEN.search(g['text']['action'] or ''))

    def test_role_words_at_level_three(self):
        g = self.texts('wide_defender_advance', 'work_on', 3)
        self.assertIn('the left back', g['text']['explanation'])
        self.assertIsNone(L.ID_TOKEN.search(g['text']['explanation']))

    def test_failed_identity_removes_names_and_role_words(self):
        for lvl in (3, 5, 7):
            f = finding_for('wide_defender_advance', 'review', lvl, identity='fail')
            g = run([f], profile_for(lvl))['findings'][0]
            body = ' '.join(v for v in g['text'].values() if v)
            with self.subTest(level=lvl):
                self.assertIsNone(L.ID_TOKEN.search(body), body)
                self.assertNotIn('left back', body)
                self.assertIn('identit' if lvl >= 4 else 'who the players were', body)

    def test_unconfirmed_roles_get_no_role_words(self):
        roles = {'A-02': ('LB', 'window'), 'A-05': ('LCB', 'window'), 'A-06': ('CDM', 'window')}
        f = finding_for('wide_defender_advance', 'review', 5, roles=roles)
        g = run([f], profile_for(5))['findings'][0]
        self.assertNotIn('left back', g['text']['explanation'])

    def test_more_relation_claims_at_higher_levels(self):
        """DESIGN 13.4 #5: deeper reasoning = at least 2 more relation/opponent claims for the same family. Holds
        from level 5 for D1-D3 and from level 6 for D4-D5 (their level 5 adds one: the opponents on our side of
        the ball, the distance to midfield); D6 is description from level 4 (DESIGN 4.6) and is not counted."""
        def count(det, lvl):
            f = finding_for(det, 'work_on' if lvl >= 3 else 'try_this', lvl)
            claims = E.build_claims(f, profile_for(lvl))
            return sum(1 for c in claims if c['type'] in ('relation', 'opponent_state', 'team_state') and
                       c['facts'].get('scope') != 'demonstrated')
        highs = {'pressure_cover': (5, 6, 7), 'weak_side_balance': (5, 6, 7), 'wide_defender_advance': (5, 6, 7),
                 'behind_ball': (6, 7), 'defensive_spacing': (6, 7)}
        for det, levels in highs.items():
            for low, high in itertools.product((2, 3), levels):
                with self.subTest(detector=det, low=low, high=high):
                    self.assertGreaterEqual(count(det, high), count(det, low)+2)

    def test_same_evidence_different_text_across_levels(self):
        for det in S.DETECTORS:
            texts, stamps = {}, set()
            for lvl in LEVELS:
                verdict = 'review'
                f = finding_for(det, verdict, lvl)
                g = run([f], profile_for(lvl))['findings'][0]
                texts[lvl] = g['text']['explanation']
                stamps.add(tuple(t['evidence'] for t in g['timestamps']))
            with self.subTest(detector=det):
                self.assertEqual(len(stamps), 1)
                for a, b in itertools.combinations(LEVELS, 2):
                    self.assertLess(E.jaccard(texts[a], texts[b]), E.JACCARD_MAX, (a, b, texts[a], texts[b]))

    def test_criticism_texts_differ_across_levels(self):
        for det in S.DETECTORS:
            texts = {}
            for lvl in LEVELS:
                verdict = 'work_on' if lvl >= 3 else 'try_this'
                texts[lvl] = self.texts(det, verdict, lvl)['text']['explanation']
            with self.subTest(detector=det):
                for a, b in itertools.combinations(LEVELS, 2):
                    self.assertLess(E.jaccard(texts[a], texts[b]), E.JACCARD_MAX, (a, b, texts[a], texts[b]))

    def test_rest_defence_defined_once_per_report(self):
        fs = [finding_for('behind_ball', 'work_on', 6), finding_for('wide_defender_advance', 'work_on', 6)]
        out = run(fs, profile_for(6))
        body = ' '.join(g['text']['explanation'] for g in out['findings'])
        self.assertEqual(body.count('PitchIQ measure'), 1, body)

    def test_pressure_and_cover_defined_once_at_level_two(self):
        fs = [finding_for('pressure_cover', 'try_this', 2), finding_for('pressure_cover', 'review', 2, t0=900.0)]
        out = run(fs, profile_for(2))
        body = ' '.join(g['text']['explanation'] for g in out['findings'])
        self.assertEqual(body.count('Pressure means'), 1, body)


class ReportTests(unittest.TestCase):
    def report(self, lvl, n_each=2):
        fs = []
        for det in S.DETECTORS:
            for k in range(n_each):
                verdict = VERDICTS[lvl][(k+S.DETECTORS.index(det)) % len(VERDICTS[lvl])]
                fs.append(finding_for(det, verdict, lvl, t0=T0+200.0*k))
        return fs, run(fs, profile_for(lvl), range={'t0': 700.0, 't1': 1300.0})

    def test_no_two_explanations_alike_and_no_slot_reused(self):
        for lvl in LEVELS:
            fs, out = self.report(lvl)
            with self.subTest(level=lvl):
                self.assertEqual(out['withheld'], [])
                self.assertEqual(E.report_problems(out['findings'], profile_for(lvl)), [])
                sents = [s for g in out['findings'] for s in E.SENTENCE_SPLIT.split(g['text']['explanation'])]
                moments = [s for s in sents if re.search(r'\d\d:\d\d', s)]
                patterns = [L.TIME_TOKEN.sub('T', s) for s in moments]
                self.assertEqual(len(patterns), len(set(patterns)), patterns)

    def test_fast(self):
        fs, _ = self.report(5)
        t = time.perf_counter()
        run(fs, profile_for(5))
        self.assertLess(time.perf_counter()-t, .3)


class PhaseWordingTests(unittest.TestCase):
    def test_estimated_coach_marked_and_mixed(self):
        f = finding_for('pressure_cover', 'work_on', 4)
        cases = {'possession-4': '(estimated)', 'coach': '(coach-marked)'}
        for src, qual in cases.items():
            g = run([f], profile_for(4), obs=obs_for([f], source=src))['findings'][0]
            self.assertIn('the other team had the ball '+qual, g['text']['explanation'])
            self.assertEqual(E._first(g['claims'], 'situation')['source'], src)
        g = run([f], profile_for(4), obs=obs_for([f], source='coach', mixed=True))['findings'][0]
        self.assertIn('(estimated, partly coach-marked)', g['text']['explanation'])

    def test_coach_marked_without_obs_from_the_estimated_flag(self):
        f = finding_for('behind_ball', 'work_on', 4, source='coach')
        g = E.explain([f], profile_for(4), range={'seconds': 600.0})['findings'][0]
        self.assertIn('(coach-marked)', g['text']['explanation'])

    def test_gate_refuses_a_wrong_or_missing_qualifier(self):
        f = finding_for('pressure_cover', 'work_on', 4)
        g = run([f], profile_for(4))['findings'][0]
        for bad in ('(coach-marked)', ''):
            text = dict(g['text'], explanation=g['text']['explanation'].replace(' (estimated)', ' '+bad if bad
                                                                                else ''))
            self.assertTrue(any('qualifier' in p for p in E.validate(text, g['claims'], g, profile_for(4))), bad)
        text = dict(g['text'], explanation='When we had the ball (estimated), A-04 stood still.')
        self.assertTrue(any('claimed phase' in p for p in E.validate(text, g['claims'], g, profile_for(4))))
        text = dict(g['text'], explanation='While they were attacking, A-04 stood still.')
        self.assertTrue(any('qualifier' in p for p in E.validate(text, g['claims'], g, profile_for(4))))


class GateTests(unittest.TestCase):
    def setUp(self):
        self.f = finding_for('pressure_cover', 'work_on', 4)
        self.p = profile_for(4)
        self.g = run([self.f], self.p, alternatives=[alternative_for(self.f)])['findings'][0]

    def problems(self, explanation=None, **fields):
        text = dict(self.g['text'], **fields)
        if explanation is not None:
            text['explanation'] = explanation
        return E.validate(text, self.g['claims'], self.g, self.p)

    def test_planned_text_passes(self):
        self.assertEqual(self.problems(), [])

    def test_flipped_polarity_is_refused(self):
        claims = copy.deepcopy(self.g['claims'])
        for c in claims:
            if c['facts'].get('relation') == 'cover' and c['type'] == 'relation':
                c['facts']['present'] = True
        self.assertTrue(any('polarity' in p for p in E.validate(self.g['text'], claims, self.g, self.p)))
        self.assertTrue(any('polarity' in p for p in self.problems(
            'When the other team had the ball (estimated), A-04 pressed the ball carrier and A-05 covered behind.')))

    def test_unreadable_relation_wording_is_refused(self):
        self.assertTrue(any('cannot read' in p for p in self.problems(
            'When the other team had the ball (estimated), A-04 was covered well.')))

    def test_invented_number_time_player_or_role(self):
        base = 'When the other team had the ball (estimated), A-04 pressed the ball carrier'
        self.assertTrue(self.problems(base+' to about 2 m.'))
        self.assertTrue(self.problems(base+' at 01:02.00.'))
        self.assertTrue(self.problems(base.replace('A-04', 'A-09')+'.'))
        self.assertTrue(any('role word' in p for p in self.problems(base+'; the right back watched.')))

    def test_naming_rules(self):
        self.assertTrue(any('names' in p for p in self.problems(headline='A-04 pressed alone')))
        p2 = profile_for(2)
        f2 = finding_for('pressure_cover', 'try_this', 2)
        g2 = run([f2], p2)['findings'][0]
        text = dict(g2['text'], explanation='When the other team had the ball (estimated), A-04 went to the ball.')
        self.assertTrue(any('names' in p for p in E.validate(text, g2['claims'], g2, p2)))

    def test_consequence_and_transition_gates(self):
        self.assertTrue(any('consequence' in p for p in self.problems(
            'When the other team had the ball (estimated), the left side was exposed.')))
        self.assertEqual([p for p in self.problems(
            'If the carrier had beaten the presser, our goal could have been exposed.') if 'consequence' in p], [])
        self.assertTrue(any('transition' in p for p in self.problems(
            'In a transition at that moment, nobody would have been there.')))
        p2 = profile_for(2)
        f2 = finding_for('behind_ball', 'try_this', 2)
        g2 = run([f2], p2)['findings'][0]
        text = dict(g2['text'], explanation='If the ball had been lost, nobody would have been there.')
        self.assertTrue(any('ball lost' in p for p in E.validate(text, g2['claims'], g2, p2)))

    def test_bans(self):
        for bad in ('That was a mistake by A-04.', 'The team needs to improve spacing.',
                    'The players should communicate better.', 'The players must stay in formation.',
                    'This follows the UEFA standard.', 'A-04 lost the ball.', 'A-04 failed to cover.'):
            with self.subTest(bad=bad):
                self.assertTrue(self.problems(bad), bad)

    def test_short_range_words(self):
        text = dict(self.g['text'], explanation=self.g['text']['explanation']+' This was a repeated pattern.')
        self.assertTrue(E.validate(text, self.g['claims'], self.g, self.p, short=True))

    def test_number_modes(self):
        p1 = profile_for(1)
        f1 = finding_for('behind_ball', 'try_this', 1)
        g1 = run([f1], p1)['findings'][0]
        text = dict(g1['text'], explanation=g1['text']['explanation']+' It happened 5 times.')
        self.assertTrue(any('numbers' in p for p in E.validate(text, g1['claims'], g1, p1)))
        self.assertTrue(any('percentages' in p for p in self.problems(
            self.g['text']['explanation']+' Cover was there in about 33% of them.')))

    def test_concept_gate(self):
        p1 = profile_for(1)
        f1 = finding_for('behind_ball', 'try_this', 1)
        g1 = run([f1], p1)['findings'][0]
        text = dict(g1['text'], explanation=g1['text']['explanation']+' Keep the balance and cover.')
        self.assertTrue(any('concept' in p for p in E.validate(text, g1['claims'], g1, p1)))

    def test_describe_does_not_judge(self):
        f = finding_for('attacking_shape', 'describe', 4)
        g = run([f], profile_for(4))['findings'][0]
        text = dict(g['text'], headline='Good attacking shape')
        self.assertTrue(any('judges' in p for p in E.validate(text, g['claims'], g, profile_for(4))))


class GateCheckTests(unittest.TestCase):
    """Each check of the gate is the one that catches its own case: with it the bad text is refused, without it
    (every other check still on) the same text passes. So no check is dead and none is covered by another."""

    def case(self, det, verdict, lvl, fmt=None, **fields):
        f = finding_for(det, verdict, lvl, fmt=fmt or '11v11')
        p = profile_for(lvl, fmt=fmt)
        g = run([f], p)['findings'][0]
        return dict(g['text'], **fields), g, p

    def cases(self):
        opp = 'When the other team had the ball (estimated), '
        d1 = ('pressure_cover', 'work_on', 4)
        return {
            'facts': self.case(*d1, explanation=opp+'A-09 watched the ball carrier.'),
            'units': self.case(*d1, explanation=opp+'A-04 pressed the ball carrier from about 2 m.'),
            'naming': self.case(*d1, headline='Cover for A-04'),
            'roles': self.case(*d1, explanation=opp+'the right back watched.'),
            'phase': self.case(*d1, explanation='When the other team had the ball (coach-marked), A-04 stood still.'),
            'numbers': self.case('pressure_cover', 'try_this', 2, explanation=opp+'one of our players went to the '
                                 'ball, and the nearest teammate was about 14 m away.'),
            'vocabulary': self.case(*d1, explanation=opp+'A-04 stood still. Keep the rest defence in mind.'),
            'events': self.case(*d1, explanation='A-04 lost the ball.'),
            'hypotheticals': self.case(*d1, explanation='The way to our goal was exposed.'),
            'polarity': self.case(*d1, explanation=opp+'A-04 pressed the ball carrier and A-05 covered behind.'),
            'questions': self.case('behind_ball', 'try_this', 1, action='Who stays back? Who helps?'),
            'describe': self.case('attacking_shape', 'describe', 4, headline='Good attacking shape'),
            # fix round 1: the cover (A-05) named as the presser (critic B-3); a full-back in a 7v7 report (S-9)
            'actors': self.case(*d1, explanation=opp+'A-05 pressed the ball carrier.'),
            'format_words': self.case(*d1, fmt='7v7', explanation=opp+'A-04 pressed the ball carrier near a '
                                                                      'full-back.'),
            # fix round 3 (LEAD decision (3), COACH F-5): a U14 team is not measured against professionals in words
            'professional': self.case(*d1, explanation=opp+'A-04 pressed the ball carrier, as professional teams do.'),
        }

    def test_each_check_catches_its_case_alone(self):
        names = set(E.FIELD_CHECKS) | set(E.TEXT_CHECKS)
        cases = self.cases()
        self.assertEqual(set(cases), names)
        for name, (text, g, p) in cases.items():
            with self.subTest(check=name):
                self.assertTrue(E.validate(text, g['claims'], g, p), name)
                self.assertEqual(E.validate(text, g['claims'], g, p, checks=names-{name}), [], name)


class FallbackTests(unittest.TestCase):
    class Engine:
        name = 'test'

        def __init__(self, text):
            self.text, self.requests = text, []

        def render(self, request):
            self.requests.append(request)
            return self.text

    def test_engine_sees_claims_and_register_only_and_valid_text_is_used(self):
        f = finding_for('pressure_cover', 'work_on', 4)
        good = {'headline': 'Pressure on the ball without covering support',
                'explanation': 'When the other team had the ball (estimated), A-04 pressed the ball carrier without '
                               'covering support.', 'short': 'Pressure without cover', 'action': None}
        eng = self.Engine(good)
        log = []
        g = run([f], profile_for(4), engine=eng, log=log)['findings'][0]
        self.assertEqual(log[0][1], 'engine')
        self.assertEqual(g['text'], good)
        self.assertEqual(set(eng.requests[0]), {'verdict', 'detector', 'level', 'register', 'short', 'claims',
                                                'limits', 'qualifier'})

    def test_invalid_engine_text_falls_back_to_the_plan(self):
        f = finding_for('pressure_cover', 'work_on', 4)
        bad = {'headline': 'A mistake', 'explanation': 'A-04 lost the ball.', 'short': 'x', 'action': None}
        log = []
        g = run([f], profile_for(4), engine=self.Engine(bad), log=log)['findings'][0]
        self.assertEqual(log[0][1], 'planned')
        self.assertNotEqual(g['text']['headline'], 'A mistake')

    def test_minimal_plan_then_withheld(self):
        f = finding_for('pressure_cover', 'work_on', 4)
        prof = profile_for(4, banned=['mistake', 'trigger', 'spells'])     # in every planned recurrence sentence
        log = []
        out = run([f], prof, log=log)
        self.assertEqual(out['withheld'], [])
        self.assertEqual(log[0][1], 'minimal')
        prof = profile_for(4, banned=['ball'])
        out = run([f], prof)
        self.assertEqual(out['findings'], [])
        self.assertEqual(out['withheld'], [{'finding': f['id'], 'reason': 'text_failed_validation'}])

    def test_criticism_without_a_relation_claim_is_withheld(self):
        f = finding_for('pressure_cover', 'work_on', 4)
        f['observation'] = 'weak_side_balance'
        f['key'] = f['key'].replace('D1|', 'D2|', 1)
        f['measurements'] = {}
        claims = E.build_claims(f, profile_for(4))
        self.assertTrue(any(c['type'] == 'team_state' for c in claims))   # weak_side is always claimed
        f2 = finding_for('attacking_shape', 'work_on', 4, measurements={})
        out = run([f2], profile_for(4))
        self.assertEqual(len(out['findings'])+len(out['withheld']), 1)

    def test_level_mismatch_is_withheld(self):
        f = finding_for('pressure_cover', 'work_on', 4)
        out = run([f], profile_for(5))
        self.assertEqual(out['withheld'], [{'finding': f['id'], 'reason': 'level_mismatch'}])


class ClaimTests(unittest.TestCase):
    def test_claims_are_schema_claims(self):
        for det, lvl, verdict, age in all_cases(('typical',)):
            f = finding_for(det, verdict, lvl)
            for c in E.build_claims(f, profile_for(lvl), alternative=alternative_for(f)):
                with self.subTest(detector=det, level=lvl, verdict=verdict, claim=c['type']):
                    self.assertEqual(S.validate('claim', c), [])

    def test_principle_sources_only_from_the_profile(self):
        f = finding_for('pressure_cover', 'work_on', 3)
        p = E._first(E.build_claims(f, profile_for(3)), 'principle')
        claims = E.build_claims(f, profile_for(3))
        pc = [c for c in claims if c['type'] == 'principle'][0]
        self.assertEqual((pc['tag'], pc['source']), ('FEDERATION', 'EF-COVER-RECOVER'))
        claims = E.build_claims(f, profile_for(3, sources=[]))
        pc = [c for c in claims if c['type'] == 'principle'][0]
        self.assertEqual((pc['tag'], pc['source']), (S.ORIGIN_P, None))
        self.assertEqual(p['principle'], 'cover_first_defender')
        f4 = finding_for('wide_defender_advance', 'work_on', 4)
        pc = [c for c in E.build_claims(f4, profile_for(4)) if c['type'] == 'principle'][0]
        self.assertEqual(pc['source'], 'US_GM_U1314_PDF')       # research id form, as the profile lists it
        f3 = finding_for('behind_ball', 'work_on', 3)           # no source is mapped for level 3: never borrowed
        pc = [c for c in E.build_claims(f3, profile_for(3)) if c['type'] == 'principle'][0]
        self.assertEqual((pc['tag'], pc['source']), (S.ORIGIN_P, None))

    def test_no_claim_invents_a_value(self):
        f = finding_for('wide_defender_advance', 'work_on', 7, measurements={'side': 'left', 'metres': 'estimated'})
        claims = E.build_claims(f, profile_for(7))
        self.assertFalse([c for c in claims if c['type'] == 'opponent_state'])
        self.assertIsNone(E._first(claims, 'actor_state')['distanceM'])

    def test_review_names_what_was_not_confirmed(self):
        f = finding_for('wide_defender_advance', 'review', 4, limitations=['recovering'])
        g = run([f], profile_for(4))['findings'][0]
        self.assertIn('recovering toward goal', g['text']['explanation'])
        self.assertTrue(g['text']['headline'].startswith('Possible issue to review: '))

    def test_one_observed_instance(self):
        f = finding_for('pressure_cover', 'review', 4, one_spell=True)
        g = run([f], profile_for(4))['findings'][0]
        self.assertRegex(g['text']['explanation'], r'one observed instance|One instance only|Only one instance')

    def test_style_effect_from_level_six(self):
        f = finding_for('wide_defender_advance', 'work_on', 6, style='possession',
                        style_effect='possession_advance_expected')
        g = run([f], profile_for(6))['findings'][0]
        self.assertIn('possession style', g['text']['explanation'])
        f5 = finding_for('wide_defender_advance', 'work_on', 5, style='possession',
                         style_effect='possession_advance_expected')
        g5 = run([f5], profile_for(5))['findings'][0]
        self.assertNotIn('style', g5['text']['explanation'])
        f6 = finding_for('wide_defender_advance', 'work_on', 6, style='possession', style_effect='no_such_effect')
        g6 = run([f6], profile_for(6))['findings'][0]
        self.assertNotIn('style', g6['text']['explanation'])            # an unknown effect is never worded

    def test_style_effects_cover_the_profile_codes(self):
        if PROFILES is None:
            self.skipTest('profiles has not landed')
        import json
        with open(os.path.join(os.path.dirname(PROFILES.__file__), 'profiles.json'), encoding='utf-8') as fh:
            data = json.loads(fh.read())
        codes = {e['code'] for name, st in data['styles'].items() if isinstance(st, dict)
                 for e in (st.get('effects') or {}).values()}
        self.assertEqual(sorted(codes - set(E.STYLE_EFFECT)), [])

    def test_every_style_effect_wording_passes_the_gate(self):
        for code, words in E.STYLE_EFFECT.items():
            style = next(s for s in ('high_press', 'possession', 'counter', 'compact') if code.startswith(s))
            for lvl in (6, 7):
                f = finding_for('wide_defender_advance', 'work_on', lvl, style=style, style_effect=code)
                log = []
                g = run([f], profile_for(lvl), log=log)['findings'][0]
                with self.subTest(code=code, level=lvl):
                    self.assertEqual(log[0][1], 'planned')
                    self.assertIn(words, g['text']['explanation'])

    def test_no_alternative_says_why_no_area_is_marked(self):
        f = finding_for('wide_defender_advance', 'work_on', 4)
        na = {'finding': f['id'], 'evidence': f['timestamps'][0]['evidence'], 't': f['timestamps'][0]['t'],
              'principle': 'cover_behind_advancing_defender', 'reason': 'would_open_new_gap', 'rejected': []}
        g = run([f], profile_for(4), no_alternatives=[na])['findings'][0]
        self.assertIn('would have opened another gap, so no area is marked', g['text']['action'])
        self.assertNotIn('marked on the diagram', g['text']['action'])
        self.assertNotIn('no area', g['text']['explanation'])

    def test_attacking_shape_never_gets_a_zone_sentence(self):
        for lvl in LEVELS:
            verdict = 'work_on' if lvl >= 3 else 'try_this'
            f = finding_for('attacking_shape', verdict, lvl)
            claims = E.build_claims(f, profile_for(lvl), alternative=alternative_for(f))
            self.assertFalse([c for c in claims if c['type'] == 'counterfactual'])

    def test_counterfactual_uses_the_alternative_mover(self):
        f = finding_for('pressure_cover', 'work_on', 5)
        g = run([f], profile_for(5), alternatives=[alternative_for(f, mover='A-05')])['findings'][0]
        self.assertIn('A-05', g['text']['action'])
        self.assertRegex(g['text']['action'], r'single-mover|only|held still|nobody else moved')   # the assumption
        g = run([f], profile_for(5))['findings'][0]
        self.assertNotIn('marked', g['text']['action'])


try:
    from soccer.coaching import profiles as PROFILES  # noqa: E402  (PROFILE's module; wave 1, may land later)
except ImportError:          # pragma: no cover
    PROFILES = None


@unittest.skipIf(PROFILES is None or not hasattr(PROFILES, 'resolve'), 'profiles.resolve has not landed')
class ResolvedProfileTests(unittest.TestCase):
    """The same cases with PROFILE's real profiles.resolve output (registers, banned words, naming, source ids)."""
    AGES = {1: ('U8', 'Adult'), 2: ('U10', 'Adult'), 3: ('U12', 'Adult'), 4: ('U14', 'Adult'), 5: ('U16', 'U18'),
            6: ('U19', 'Adult'), 7: ('Adult', 'U23')}

    def test_every_case_is_planned_and_cites_only_profile_sources(self):
        for lvl, ages in self.AGES.items():
            for age in ages:
                prof = PROFILES.resolve({'playingLevel': lvl, 'ageGroup': age}, team='A')
                for det in S.DETECTORS:
                    for verdict in VERDICTS[lvl]:
                        with self.subTest(level=lvl, age=age, detector=det, verdict=verdict):
                            f = finding_for(det, verdict, lvl)
                            log = []
                            alts = [alternative_for(f)] if verdict != 'strength' and verdict != 'describe' else []
                            out = run([f], prof, alternatives=alts, log=log)
                            self.assertEqual(log[0][1], 'planned', log)
                            for c in out['findings'][0]['claims']:
                                if c.get('source'):
                                    self.assertIn(c['source'], prof['sources'])


class SpecExampleTests(unittest.TestCase):
    """The spec's left-back examples (SPEC_v2 Part 5) are style references, never copied."""
    EXAMPLES = (
        'Two teammates moved forward together, leaving no one behind to help defend. Try keeping one player ready to '
        'protect the goal.',
        'Our left defender moved forward to help attack, but nobody stayed behind to cover. When one player goes '
        'forward, another can help protect the space behind.',
        'The left defender advanced while the nearby midfielder stayed high. This left limited cover on the left '
        'side if possession changed.',
        "At this moment, the left back was advanced beyond the defensive unit without adequate covering support. "
        "That reduced the team's defensive balance on the left.",
        "The left back's forward movement was not matched by a holding-midfielder rotation or center-back "
        "adjustment, leaving the wide defensive channel vulnerable during transition.",
        'The attacking structure lacked sufficient rest-defense balance behind the advanced fullback. A '
        'holding-midfielder rotation or back-line adjustment could have reduced exposure during possession loss.',
        "The left back's advanced position was not supported by a coordinated adjustment in the rest-defense "
        "structure. The spacing between the six and left center back potentially exposed a transition route into "
        "the vacated channel.",
    )

    def test_no_explanation_copies_an_example(self):
        for lvl in LEVELS:
            for verdict in VERDICTS[lvl]:
                f = finding_for('wide_defender_advance', verdict, lvl)
                g = run([f], profile_for(lvl), alternatives=[alternative_for(f)])['findings'][0]
                body = ' '.join(v for v in g['text'].values() if v)
                for ex in self.EXAMPLES:
                    with self.subTest(level=lvl, verdict=verdict):
                        self.assertLess(E.jaccard(body, ex), .2, (body, ex))


class BankTests(unittest.TestCase):
    def test_banks_are_ascii_and_have_variants(self):
        for det, slots in E.BANK.items():
            for slot, table in slots.items():
                for levels, items in table.items():
                    for t in (items if isinstance(items, list) else list(items)):
                        self.assertTrue(re.fullmatch(r'[ -~]+', t), (det, slot, t))
                    if slot.startswith(('open_', 'hypo', 'question', 'statement', 'headline_fail', 'headline_ok',
                                        'context_fail')) and items:      # [] = nothing said at those levels
                        self.assertGreaterEqual(len(items), 3, (det, slot, levels))

    def test_every_counterfactual_states_the_single_mover_assumption(self):
        rx = re.compile(r'only|nobody else|held still|single-mover|everyone else|stayed still|staying put|the others')
        for slot, table in E.BANK['*'].items():
            if slot.startswith('counterfactual'):
                for items in table.values():
                    for t in items:
                        self.assertRegex(t, rx, (slot, t))

    def test_principles_cover_every_level(self):
        for code, p in E.PRINCIPLES.items():
            for lvl in LEVELS:
                self.assertIsNotNone(E._at(p['text'], lvl), (code, lvl))

    def test_measurement_keys_documented(self):
        self.assertEqual(set(E.MEASUREMENT_KEYS), set(S.DETECTORS))


# =====================================================================================================================
# Wave 5: what the first run on REAL findings (DETECT's assess, EVIDENCE's moments, ZONES' zones) changed
# =====================================================================================================================
class RealFindingTests(unittest.TestCase):
    """EVAL's scenes through every real stage (tests/coaching_text_fixtures.py): at every level, every finding gets
    its planned text and none is withheld."""

    SCENES = (('unsupported_advance', 'none_behind'), ('unsupported_advance', 'one_behind'),
              ('identity_crossing', None), ('small_field_scale', 'assumed'), ('amateur_coach_spans', 'spans'),
              ('defending', 'press_no_cover'), ('defending', 'gap_open'))

    def test_every_real_finding_is_worded_at_every_level(self):
        import coaching_text_fixtures as TX
        from soccer import coaching
        for scene, variant in self.SCENES:
            b, st, obs, roles = TX.tiers(scene, variant)
            for lvl in LEVELS:
                with self.subTest(scene=scene, variant=variant, level=lvl):
                    got = {}

                    def capture(**kw):
                        got.update(kw)
                        from soccer.coaching import report
                        return report.build_report_v2(**kw)
                    coaching.interpret(obs, roles, TX.settings_at(st, 'A', lvl), 'A', stages={'report': capture})
                    self.assertEqual(got['withheld'], [])
                    self.assertEqual(E.report_problems(got['findings'], got['profile']), [])


class AttackingShapeModeTests(unittest.TestCase):
    """The relation that failed at the key moment is the one the text names (BC180 Barcelona at levels 1-2: depth
    share .25 < .30 with width .88 and three channels; the wave-1 text said "we stayed narrow")."""

    def text_of(self, lvl, m):
        f = finding_for('attacking_shape', 'try_this' if lvl <= 3 else 'work_on', lvl, measurements=m)
        out = run([f], profile_for(lvl))
        self.assertEqual(out['withheld'], [])
        return out['findings'][0]

    def test_depth_failure_never_says_narrow(self):
        m = {'widthShare': .88, 'depthShare': .25, 'widthM': 59.6, 'depthM': 26.1, 'channelsOccupied': 3,
             'metres': 'estimated'}
        for lvl in LEVELS:
            g = self.text_of(lvl, m)
            body = ' '.join(v for v in g['text'].values() if v).lower()
            with self.subTest(level=lvl):
                self.assertNotIn('narrow', body)
                self.assertNotRegex(body, r'one side|bunched')
                self.assertEqual(E._first(g['claims'], 'team_state', 'width')['present'], True)
                self.assertEqual(E._first(g['claims'], 'team_state', 'depth')['present'], False)
                self.assertEqual(E._first(g['claims'], 'situation')['mode'], 'depth')

    def test_modes(self):
        p = profile_for(4)
        cases = {'width': (.40, .35, 3), 'depth': (.70, .20, 3), 'shape': (.40, .20, 2), 'channels': (.70, .35, 2)}
        for mode, (w, d, c) in cases.items():
            f = finding_for('attacking_shape', 'work_on', 4, measurements={'widthShare': w, 'depthShare': d,
                                                                           'channelsOccupied': c,
                                                                           'metres': 'estimated'})
            self.assertEqual(E.failure_mode(f, 4, p), mode)

    def test_counter_style_does_not_judge_width(self):
        p = profile_for(6)
        p['detectors']['attacking_shape']['params'] = {'widthShare': .55, 'depthShare': .30, 'judgeWidth': False}
        f = finding_for('attacking_shape', 'work_on', 6, measurements={'widthShare': .30, 'depthShare': .20,
                                                                       'channelsOccupied': 3, 'metres': 'estimated'})
        self.assertIsNone(E.shape_relations(f, p)['width'])
        self.assertEqual(E.failure_mode(f, 6, p), 'depth')


class WaveFiveWordingTests(unittest.TestCase):
    def test_team_level_checks_name_nobody(self):
        """params.teamLevelOnly (levels 1-2 wide defenders, levels 1-3 behind the ball): no relative or role words,
        even where evidence.naming is 'relative' (EVAL's none_behind scene said "our left defender" at level 2)."""
        p = profile_for(2)
        p['detectors']['wide_defender_advance']['params'] = {'teamLevelOnly': True}
        f = finding_for('wide_defender_advance', 'try_this', 2)
        g = run([f], p)['findings'][0]
        body = ' '.join(v for v in g['text'].values() if v)
        self.assertNotRegex(body, r'\bour (?:left|right) defender\b|left back|A-0')
        self.assertIsNone(E._first(g['claims'], 'actor_state')['role'])
        self.assertEqual(E.register_for(p, f)['naming'], 'never')
        self.assertEqual(E.register_for(p)['naming'], 'relative')

    def test_zero_counts_read_none(self):
        m = dict(FAIL_M['behind_ball'], behindN=0, threatN=0, restMargin=0)
        for lvl in (3, 4, 5, 6, 7):
            f = finding_for('behind_ball', 'work_on' if lvl >= 3 else 'try_this', lvl, measurements=m)
            g = run([f], profile_for(lvl))['findings'][0]
            body = ' '.join(v for v in g['text'].values() if v)
            with self.subTest(level=lvl):
                self.assertNotRegex(body, r'\b0 of\b|\bonly 0\b|\+0\b|only none')

    def test_margin_sign(self):
        self.assertEqual(E._margin(0), '0')
        self.assertEqual(E._margin(2), '+2')
        self.assertEqual(E._margin(-1), '-1')

    def test_values_at_one_moment_are_not_averages(self):
        for det in ('attacking_shape', 'defensive_spacing', 'behind_ball', 'weak_side_balance'):
            for lvl in range(2, 8):
                g = run([finding_for(det, 'describe', lvl)], profile_for(lvl))['findings'][0]
                self.assertNotIn('on average', g['text']['explanation'], (det, lvl))

    def test_a_description_is_not_called_a_review(self):
        for lvl in LEVELS:
            f = finding_for('attacking_shape', 'describe', lvl, limitations=('short_range',))
            g = run([f], profile_for(lvl))['findings'][0]
            self.assertNotIn('review', g['text']['explanation'].lower(), lvl)

    def test_own_sentence_limitations(self):
        for lvl in (2, 4, 6):
            f = finding_for('pressure_cover', 'review', lvl, limitations=('level_cap',))
            f['guardrails']['role'] = 'pass'
            g = run([f], profile_for(lvl))['findings'][0]
            with self.subTest(code='level_cap', level=lvl):
                self.assertRegex(g['text']['explanation'], r'list (?:of things to try )?short|at most')
            f = finding_for('pressure_cover', 'describe', lvl, limitations=('no_moment',))
            f['timestamps'] = []
            g = run([f], profile_for(lvl))['findings'][0]
            with self.subTest(code='no_moment', level=lvl):
                self.assertIn('No single video moment', g['text']['explanation'])
            f = finding_for('weak_side_balance', 'review', lvl, one_spell=True, limitations=('one_instance',))
            f['guardrails']['role'] = 'pass'
            f['guardrails']['evidence'] = 'fail'
            g = run([f], profile_for(lvl))['findings'][0]
            with self.subTest(code='one_instance', level=lvl):
                self.assertNotIn(E.LIMITATION_WORDS['evidence'][0 if lvl <= 3 else 1], g['text']['explanation'])
                self.assertRegex(g['text']['explanation'], r'once|One instance|one observed instance|Only one')

    def test_level_one_reframed_advance(self):
        f = finding_for('behind_ball', 'try_this', 1, measurements=dict(FAIL_M['behind_ball'], behindN=0,
                                                                         minBehind=1))
        f['key'] = f['key'].replace('|team|', '|advance|')
        g = run([f], profile_for(1))['findings'][0]
        self.assertEqual(E.variant_of(g), 'advance')
        self.assertRegex(g['text']['explanation'], r'defender')
        self.assertNotRegex(L.TIME_TOKEN.sub(' ', g['text']['explanation']), r'\d')     # mm:ss times only

    def test_phase_source_from_the_measurements(self):
        f = finding_for('behind_ball', 'strength', 4)
        f['measurements']['phaseSource'] = 'coach'
        self.assertEqual(E.phase_source_of(f, {}), 'coach')
        f['measurements']['phaseSource'] = 'mixed'
        self.assertEqual(E.phase_source_of(f, {}), 'mixed')

    def test_short_range_from_assess_without_the_range(self):
        """interpret calls explain without the report range: assess's 'short_range' limitation decides."""
        f = finding_for('pressure_cover', 'work_on', 5, limitations=('short_range',))
        self.assertTrue(E.short_for(f, False))
        g = E.explain([f], profile_for(5), obs=obs_for([f]))['findings'][0]
        body = ' '.join(v for v in g['text'].values() if v).lower()
        for w in L.SHORT_BANNED:
            self.assertNotRegex(body, r'(?<![a-z\-])'+re.escape(w)+r'(?![a-z\-])')

    def test_a_metre_free_text_for_every_case(self):
        """A small format without a declared field has metresUsable false: every case still gets its planned text
        (small_field_scale/assumed level 4 was withheld before)."""
        for det, lvl, verdict, age in all_cases(ages=('typical',)):
            f = finding_for(det, verdict, lvl)
            f['measurements']['metres'] = 'unusable'
            log = []
            out = run([f], profile_for(lvl, age), log=log)
            with self.subTest(detector=det, level=lvl, verdict=verdict):
                self.assertEqual(out['withheld'], [])
                self.assertEqual(log[0][1], 'planned')
                self.assertNotRegex(out['findings'][0]['text']['explanation'], r'\d+ m\b')

    def test_the_centre_back_line_is_not_a_role_word(self):
        g = E._Gate([], finding_for('wide_defender_advance', 'review', 5, identity='fail'),
                    E.register_for(profile_for(5)), False)
        self.assertEqual(E._check_roles('explanation', 'He advanced beyond the centre-back line.', g), [])
        self.assertTrue(E._check_roles('explanation', 'The centre-back stayed central.', g))

    def test_left_is_a_side_in_a_statement(self):
        self.assertIsNone(E.PAST.search('a teammate stepping into the left channel behind them'))
        self.assertIsNotNone(E.PAST.search('the space he left'))

    def test_no_zone_for_detector_is_worded(self):
        self.assertIn('no_zone_for_detector', E.NO_ALT_WORDS)
        f = finding_for('defensive_spacing', 'work_on', 5, measurements=dict(FAIL_M['defensive_spacing'],
                                                                             failureMode='dm_gap'))
        na = {'finding': f['id'], 'evidence': None, 't': None, 'principle': 'unit_spacing',
              'reason': 'no_zone_for_detector', 'rejected': []}
        g = run([f], profile_for(5), no_alternatives=[na])['findings'][0]
        self.assertIn('no single area is marked', g['text']['action'])
        self.assertEqual(E._first(g['claims'], 'principle')['principle'], 'compact_lines')

    def test_with_no_alternative(self):
        f = finding_for('pressure_cover', 'work_on', 4)
        g = run([f], profile_for(4))['findings'][0]
        na = {'finding': f['id'], 'evidence': None, 't': None, 'principle': 'cover_first_defender',
              'reason': 'would_open_new_gap', 'rejected': []}
        h = E.with_no_alternative(g, na, profile_for(4))
        self.assertTrue(h['text']['action'].startswith(E.NO_ALT_WORDS['would_open_new_gap'][1]))
        self.assertEqual(E.validate(h['text'], h['claims'], h, profile_for(4)), [])
        self.assertIs(E.with_no_alternative(h, na, profile_for(4)), h, 'never twice')
        r = run([finding_for('pressure_cover', 'review', 4)], profile_for(4))['findings'][0]
        self.assertIs(E.with_no_alternative(r, dict(na, reason='guardrail_failed'), profile_for(4)), r)
        s = run([finding_for('pressure_cover', 'strength', 4)], profile_for(4))['findings'][0]
        self.assertIs(E.with_no_alternative(s, na, profile_for(4)), s)

    def test_the_level_six_margin_rule_claims_no_count(self):
        """Levels 6-7 on 9v9 / 11v11 judge the rest-defence margin: with 3 behind the ball and a margin of -1 the
        text must not claim a required count of 1 (it would contradict the measurement)."""
        p = profile_for(6)
        p['detectors']['behind_ball']['params'] = {'behindRule': 'margin', 'minMargin': 1}
        m = {'behindN': 3, 'threatN': 4, 'restMargin': -1, 'metres': 'estimated'}
        f = finding_for('behind_ball', 'work_on', 6, measurements=m)
        out = run([f], p)
        self.assertEqual(out['withheld'], [])
        rel = E._first(out['findings'][0]['claims'], 'team_state', 'behind_ball')
        self.assertEqual((rel['present'], rel['requiredMargin'], rel['margin']), (False, 1, -1))
        self.assertNotIn('required', rel)

    def test_every_wording_is_free_of_unmeasured_events(self):
        """schema.event_problems over every phrasing, principle, definition and fixed sentence: the wave-1
        'Because a guardrail did not pass' could never pass the gate ("pass" reads as an event)."""
        found = []

        def walk(x, path):
            if isinstance(x, str):
                found.extend((path, p) for p in S.event_problems(x))
            elif isinstance(x, dict):
                for k, v in x.items():
                    walk(v, f'{path}.{k}')
            elif isinstance(x, (list, tuple)):
                for i, v in enumerate(x):
                    walk(v, f'{path}[{i}]')
        walk(E.BANK, 'BANK')
        walk(E.NO_ALT_WORDS, 'NO_ALT_WORDS')
        walk(E.LIMITATION_WORDS, 'LIMITATION_WORDS')
        walk({c: p['text'] for c, p in E.PRINCIPLES.items()}, 'PRINCIPLES')
        walk([d[3] for d in E.DEFINITIONS], 'DEFINITIONS')
        walk(E.STYLE_EFFECT, 'STYLE_EFFECT')
        self.assertEqual(found, [])

    def test_validate_free(self):
        p1, p4 = profile_for(1), profile_for(4)
        f = L.facts_for(extra={'n': 3, 't': 610.0})
        self.assertEqual(E.validate_free('Three checks could be made at 10:10.00.', p4, f), [])
        self.assertTrue(E.validate_free('Four checks could be made.', p4, f))                  # not a fact
        self.assertTrue(E.validate_free('3 checks could be made.', p1, f))                     # no numbers at level 1
        self.assertEqual(E.validate_free('3 checks could be made.', p1, f, numbers=False), [])
        self.assertTrue(E.validate_free('The cover was poor.', p4, f))                         # banned word
        self.assertTrue(E.validate_free('We lost the ball twice.', p4, L.facts_for(extra={'n': 2})))   # event
        self.assertTrue(E.validate_free('When we had the ball, it went well.', p4, f))          # no qualifier
        self.assertEqual(E.validate_free('When we had the ball (estimated), it went well.', p4, f), [])
        self.assertTrue(E.validate_free('It happened repeatedly.', p4, f, short=True))
        self.assertTrue(E.validate_free('x'*50, p4, f, limit=40))
        self.assertEqual(E.validate_free('Play 2 v 1.', p1, f, numbers=False, check_facts=False), [])


# =====================================================================================================================
# Fix round 1 (critic B-3, B-4, S-6, S-9, S-10 and the coach's / facts verifier's low wording items)
# =====================================================================================================================
D6_FULL = {'widthShare': .72, 'depthShare': .36, 'widthM': 49.0, 'depthM': 37.8, 'channelsOccupied': 3,
           'depthAvailShare': .82, 'availableDepthM': 46.1, 'heightGapM': 3.4, 'deepN': 4, 'metres': 'estimated'}
D2_FULL = {'weakCount': 2, 'centralCount': 3, 'openWidthM': 11.6, 'weakLevelCount': 1, 'weakRecovering': 0,
           'ballX': 31.0, 'metres': 'estimated'}
D5_FULL = {'lateralGapM': 14.2, 'dmGapM': 12.6, 'lineStdM': 1.9, 'blockHeight': 'mid', 'dmGapOpponents': 1,
           'metres': 'estimated'}
D4_FULL = {'behindN': 4, 'minBehind': 2, 'threatN': 2, 'restMargin': 2, 'metres': 'estimated'}
DESCRIBE_M = {'attacking_shape': D6_FULL, 'weak_side_balance': D2_FULL, 'defensive_spacing': D5_FULL,
              'behind_ball': D4_FULL}


class FixRoundOneTests(unittest.TestCase):
    def body(self, g):
        return ' '.join(v for v in g['text'].values() if v)

    def one(self, f, p, **kw):
        log = []
        out = run([f], p, log=log, **kw)
        self.assertEqual(out['withheld'], [], log)
        self.assertEqual(log[0][1], 'planned', log)
        return out['findings'][0]

    # ---- B-3: the presser is the moment's first defender, never the cover ----
    def test_the_presser_is_the_first_defender_at_the_moment(self):
        """The family's modal first defender (players.primary) is not the presser at the linked moment; the text
        names measurements.fdId (critic spot check: 10 of 10 synthetic texts named the wrong presser), and the cover
        is never the presser even when coverId would say so."""
        for lvl in (4, 5, 6, 7):
            m = dict(FAIL_M['pressure_cover'], fdId='A-08', coverId='A-05')
            f = finding_for('pressure_cover', 'work_on', lvl, measurements=m)
            f['players']['supporting'] = ['A-05', 'A-08']
            g = self.one(f, profile_for(lvl))
            ids = re.findall(r'\b[AB]-\d+\b', g['text']['explanation'])
            with self.subTest(level=lvl):
                self.assertEqual(ids[0], 'A-08', g['text']['explanation'])
                self.assertNotIn('A-04', ids)
                self.assertEqual(E.invariant_problems(g), [])
        m = dict(FAIL_M['pressure_cover'], fdId='A-08', coverId='A-08')
        f = finding_for('pressure_cover', 'work_on', 5, measurements=m)
        f['players']['supporting'] = ['A-08', 'A-05']
        self.assertEqual((E.presser_of(f), E.cover_of(f)), ('A-08', 'A-05'))

    def test_the_gate_refuses_another_presser_and_the_presser_as_cover(self):
        m = dict(OK_M['pressure_cover'], fdId='A-04', coverId='A-05')
        f = finding_for('pressure_cover', 'strength', 5, measurements=m)
        p = profile_for(5)
        g = self.one(f, p)
        opp = 'When the other team had the ball (estimated), '
        for bad in (opp+'A-05 pressed the ball carrier with A-04 covering.',
                    opp+'A-04 pressed the ball carrier with A-04 covering.',
                    opp+'the ball carrier was pressed by A-05.'):
            text = dict(g['text'], explanation=bad)
            with self.subTest(text=bad):
                self.assertTrue([x for x in E.validate(text, g['claims'], g, p) if 'presser' in x or 'cover' in x])
                self.assertTrue(E.invariant_problems(dict(g, text=text)))
        good = dict(g['text'], explanation=opp+'A-04 pressed the ball carrier with A-05 covering behind.')
        self.assertEqual(E.validate(good, g['claims'], g, p), [])

    # ---- B-4: no phase claimed when it is unknown ----
    def test_phase_unknown_headlines_say_nothing_of_the_ball(self):
        for lvl in LEVELS:
            f = finding_for('attacking_shape', 'describe', lvl, measurements=D6_FULL, phase='unknown')
            f['key'] = f['key'].replace('|team|', '|unknown|')
            g = self.one(f, profile_for(lvl))
            with self.subTest(level=lvl):
                for k in ('headline', 'short'):
                    self.assertIsNone(E.PHASE_CLAIM_RE.search(g['text'][k]), g['text'][k])
                self.assertEqual(E.invariant_problems(g), [])
                bad = dict(g['text'], headline='Team shape with the ball')
                self.assertTrue(E.validate(bad, g['claims'], g, profile_for(lvl)))
                self.assertTrue(E.invariant_problems(dict(g, text=bad)))

    # ---- S-6: a description reads a different relation at every level ----
    def test_descriptions_differ_between_levels(self):
        """W8 miss 1: the attacking-shape description reused its sentence at levels 5 and 7 (.51-.80); the far-side
        and spacing descriptions now appear where a strength is withheld (BC180 B: .58-.74)."""
        for det, m in DESCRIBE_M.items():
            for phase in ((None, 'unknown') if det == 'attacking_shape' else (None,)):
                texts = {}
                for lvl in LEVELS:
                    f = finding_for(det, 'describe', lvl, measurements=m, phase=phase,
                                    limitations=('mixed_evidence',) if det != 'attacking_shape' else ())
                    if phase:
                        f['key'] = f['key'].replace('|team|', '|unknown|')
                    texts[lvl] = self.one(f, profile_for(lvl))['text']['explanation']
                with self.subTest(detector=det, phase=phase):
                    for a, b in itertools.combinations(LEVELS, 2):
                        self.assertLess(E.jaccard(texts[a], texts[b]), .4, (a, b, texts[a], texts[b]))

    def test_mixed_evidence_and_formation_changes_are_said(self):
        for lvl in LEVELS:
            f = finding_for('weak_side_balance', 'describe', lvl, measurements=D2_FULL,
                            limitations=('mixed_evidence', 'formation_changing'))
            f['aggregate'].update(demonstrated=5, notDemonstrated=4, unclear=0, opportunities=9)
            p = profile_for(lvl)
            p['detectors']['weak_side_balance']['params'] = {'baseFailShare': .004}
            g = self.one(f, p)
            with self.subTest(level=lvl):
                self.assertRegex(g['text']['explanation'], r'(?i)both ways|some did not|not showing it|not shown|did '
                                                           r'not show|Mixed evidence|lacked it|absent in')
                self.assertRegex(g['text']['explanation'], r'changing (?:its )?shape|formation change|change of '
                                                           r'formation')

    # ---- S-10: levels 5-7 state each relation they measured ----
    def test_levels_five_to_seven_state_the_measured_relations(self):
        roles = {'A-02': ('LB', 'coach'), 'A-05': ('LCB', 'coach'), 'A-06': ('CDM', 'coach')}
        for lvl in (5, 6, 7):
            for verdict in ('work_on', 'review'):
                f = finding_for('wide_defender_advance', verdict, lvl, roles=roles)
                g = self.one(f, profile_for(lvl))
                e = g['text']['explanation']
                with self.subTest(level=lvl, verdict=verdict):
                    self.assertIn('did not move across (a shift of about 1 m toward the ball)', e)
                    self.assertRegex(e, r'\b(?:holding midfielder|six) \(A-06\) did not drop')
                    if lvl >= 6:
                        self.assertRegex(e, r'margin (?:of )?-1')
                        # (the obs of these tests has no scale: the margin words are "well", fix round 3)
                        self.assertIn(E.definition_text(E.DEFINITIONS[1][3], 'well'), e)
                    if lvl == 7:
                        self.assertRegex(e, r'about 18 m')

    def test_rest_defence_is_defined_wherever_a_level_six_text_uses_it(self):
        for det in S.DETECTORS:
            for lvl in (6, 7):
                for verdict in VERDICTS[lvl]:
                    f = finding_for(det, verdict, lvl)
                    g = run([f], profile_for(lvl))['findings'][0]
                    body = self.body(g)
                    with self.subTest(detector=det, level=lvl, verdict=verdict):
                        if re.search(r'\brest[- ]defen[cs]e\b', body, re.I):
                            self.assertIn('Rest defence here is a PitchIQ measure', g['text']['explanation'])

    # ---- S-9: no 11v11 role names below 11v11 ----
    def test_small_formats_use_no_eleven_a_side_role_words(self):
        roles = {'A-02': ('LB', 'coach'), 'A-05': ('LCB', 'coach'), 'A-06': ('CDM', 'coach')}
        for fmt in ('4v4', '5v5', '7v7', '9v9'):
            for det in S.DETECTORS:
                for lvl in LEVELS:
                    for verdict in VERDICTS[lvl]:
                        f = finding_for(det, verdict, lvl, fmt=fmt,
                                        roles=roles if det == 'wide_defender_advance' else None)
                        out = run([f], profile_for(lvl, fmt=fmt))
                        with self.subTest(fmt=fmt, detector=det, level=lvl, verdict=verdict):
                            self.assertEqual(out['withheld'], [])
                            g = out['findings'][0]
                            self.assertIsNone(E.ROLE11_RE.search(self.body(g)), self.body(g))
                            self.assertEqual(E.invariant_problems(g), [])
        g = run([finding_for('wide_defender_advance', 'work_on', 5, fmt='7v7', roles=roles)],
                profile_for(5, fmt='7v7'))['findings'][0]
        self.assertIn('left defender (A-02)', g['text']['explanation'])

    # ---- low wording items ----
    def test_no_false_precision(self):
        """COACH L5: "about 100% of the 16 advances", "about 56.9 times per 10 minutes" from 2.3 minutes."""
        f = finding_for('wide_defender_advance', 'work_on', 5)
        f['aggregate'].update(opportunities=9, notDemonstrated=9, demonstrated=0, unclear=0)
        g = self.one(f, profile_for(5))
        self.assertNotRegex(self.body(g), r'about (?:100|0)%')
        for secs, rx in ((188.2, None), (420.0, r'about \d+ (?:times )?per 10 minutes'),
                         (700.0, r'about \d+\.\d (?:times )?per 10 minutes')):
            f = finding_for('pressure_cover', 'work_on', 6, phase_seconds=secs)
            body = self.one(f, profile_for(6))['text']['explanation']
            with self.subTest(seconds=secs):
                if rx is None:
                    self.assertIsNone(E.RATE_RE.search(body), body)
                else:
                    self.assertRegex(body, rx)

    def test_levels_one_and_two_never_explain_the_zone_engine(self):
        for lvl, verdict in ((1, 'try_this'), (2, 'try_this'), (3, 'work_on'), (4, 'work_on')):
            f = finding_for('wide_defender_advance', verdict, lvl)
            na = {'finding': f['id'], 'evidence': f['timestamps'][0]['evidence'], 't': f['timestamps'][0]['t'],
                  'principle': 'cover_behind_advancing_defender', 'reason': 'would_open_new_gap', 'rejected': []}
            g = self.one(f, profile_for(lvl), no_alternatives=[na])
            g2 = E.with_no_alternative(run([f], profile_for(lvl))['findings'][0], na, profile_for(lvl))
            with self.subTest(level=lvl):
                for x in (g, g2):
                    marked = re.search(r'is marked', x['text']['action'] or '')
                    self.assertEqual(bool(marked), lvl >= 3, x['text']['action'])

    def test_no_reachable_spot_goal_side_of_the_channel_opponent(self):
        f = finding_for('wide_defender_advance', 'work_on', 4)
        na = {'finding': f['id'], 'evidence': f['timestamps'][0]['evidence'], 't': f['timestamps'][0]['t'],
              'principle': 'cover_behind_advancing_defender', 'reason': 'no_reachable_fix', 'rejected': []}
        g = self.one(f, profile_for(4), no_alternatives=[na])
        self.assertIn('goal-side of the opponent in it', g['text']['action'])
        f['measurements']['opponentsInChannel'] = 0
        g = self.one(f, profile_for(4), no_alternatives=[na])
        self.assertIn('restores it in time', g['text']['action'])

    def test_level_one_strength_keeps_its_plan_without_cover(self):
        for cover in (False, True):
            m = dict(OK_M['pressure_cover'], coverPresent=cover)
            g = self.one(finding_for('pressure_cover', 'strength', 1, measurements=m), profile_for(1))
            with self.subTest(cover=cover):
                self.assertEqual('close behind to help' in g['text']['explanation'], cover)
                self.assertGreaterEqual(len(re.split(r'(?<=[.!?])\s+', g['text']['explanation'])), 3)
                self.assertNotIn('Keep it up', g['text']['action'])

    def test_behind_the_ball_is_placed_in_its_judged_third(self):
        """FACTS L8: D4 judges the final third only; "in the middle third" came from the opportunities' hysteresis
        zone. A description's moment may be anywhere: it names no place."""
        for lvl in (3, 4, 5):
            f = finding_for('behind_ball', 'strength', lvl)
            f['context']['ball']['zone'] = 'middle_third'
            g = self.one(f, profile_for(lvl))
            with self.subTest(level=lvl):
                self.assertNotRegex(self.body(g), r'middle third|middle of the pitch')
                self.assertIn('in the final third', g['text']['explanation'])
        f = finding_for('behind_ball', 'describe', 5, measurements=D4_FULL)
        f['context']['ball']['zone'] = 'middle_third'
        self.assertNotRegex(self.one(f, profile_for(5))['text']['explanation'], r'third|middle of the pitch')

    def test_far_side_words_follow_the_tolerant_count(self):
        for lvl in (4, 5, 6, 7):
            m = dict(FAIL_M['weak_side_balance'], weakCount=0, centralCount=1, openWidthM=34.0)
            g = self.one(finding_for('weak_side_balance', 'work_on' if lvl <= 6 else 'review', lvl, measurements=m),
                         profile_for(lvl))
            body = self.body(g)
            with self.subTest(level=lvl):
                self.assertNotRegex(body, r'[Nn]o defender stayed in the central corridor|held no defender')
                if lvl >= 5 and 'about 34 m' in body:
                    self.assertIn('whole far half', body)

    def test_one_player_covering_is_said_once(self):
        roles = {'A-02': ('LB', 'coach'), 'A-05': ('LCB', 'coach'), 'A-06': ('CDM', 'coach')}
        m = dict(OK_M['wide_defender_advance'], channelBy='A-05')
        for lvl in (5, 6, 7):
            g = self.one(finding_for('wide_defender_advance', 'strength', lvl, measurements=m, roles=roles),
                         profile_for(lvl))
            with self.subTest(level=lvl):
                self.assertNotIn('the near centre-back moved across to cover', g['text']['explanation'])

    def test_forward_runs_are_not_attacks(self):
        g = self.one(finding_for('wide_defender_advance', 'try_this', 2), profile_for(2))
        self.assertNotRegex(self.body(g), r'\battacks\b')
        self.assertIn('forward runs by our defenders', g['text']['explanation'])

    def test_depth_by_the_available_length(self):
        """critic B-7: against a low block the field share (.25) understates the depth; with depthRule 'available'
        the length used up to the opponents' last line decides (contextual.sample_outcomes' rule)."""
        p = profile_for(4)
        p['detectors']['attacking_shape']['params'] = {'widthShare': .55, 'depthShare': .30,
                                                       'depthAvailShare': .75, 'depthRule': 'available'}
        base = dict(D6_FULL, depthShare=.25)
        for avail, want in ((.95, True), (.5, False), (None, None)):
            m = dict(base, depthAvailShare=avail)
            self.assertEqual(E.shape_relations(finding_for('attacking_shape', 'strength', 4, measurements=m), p)
                             ['depth'], want, avail)
        p['detectors']['attacking_shape']['params']['depthRule'] = 'available_or_field'
        m = dict(base, depthAvailShare=None)
        self.assertFalse(E.shape_relations(finding_for('attacking_shape', 'strength', 4, measurements=m), p)['depth'])
        p['detectors']['attacking_shape']['params']['depthRule'] = 'field'
        self.assertFalse(E.shape_relations(finding_for('attacking_shape', 'strength', 4, measurements=base), p)
                         ['depth'])

    def test_a_description_the_style_explains_says_so(self):
        """COACH M2: under the compact style the far-side description named the intent limitation, not the style."""
        for lvl in (6, 7):
            f = finding_for('weak_side_balance', 'describe', lvl, measurements=D2_FULL, style='compact',
                            style_effect='compact_far_side_accepted')
            g = self.one(f, profile_for(lvl))
            self.assertIn(E.STYLE_EFFECT['compact_far_side_accepted'], g['text']['explanation'], lvl)

    def test_judged_play_is_stated_with_strengths(self):
        m = dict(OK_M['behind_ball'], judgedSeconds=12.4)
        for lvl, rx in ((3, None), (4, r'12 s'), (6, r'12 s')):
            g = self.one(finding_for('behind_ball', 'strength', lvl, measurements=m), profile_for(lvl))
            with self.subTest(level=lvl):
                if rx is None:
                    # the judged SECONDS are said from level 4 ("PitchIQ judged 12 s of play"); since fix round 3 a
                    # strength's count is of the moments PitchIQ could judge at every level from 2 (FACTS F-6)
                    self.assertNotRegex(g['text']['explanation'], r'judged \d|judged play|of play in these')
                else:
                    self.assertRegex(g['text']['explanation'], rx)


def unclear_rx(n):
    """The sentence stating n unclear opportunities, in any level's words."""
    return rf'\b{n}\b[^.]*(?:unclear|not be judged)|unclear: {n}\b'


def moments_finding(det, lvl, *, clear=3, decided=11, failing=3, spells=3, n=12, fmt='11v11', limitations=(),
                    measurements=None, **kw):
    """A 'moments worth a look' review as assess fix round 2 makes it: verdict review, limitation MOMENTS, every
    guardrail passed but evidence (the level's verdict table was not met), no priority, the counts in the
    measurements and an aggregate whose demonstrated = decided - failing and unclear = n - decided."""
    f = finding_for(det, 'review', lvl, n=n, fmt=fmt, limitations=list(limitations)+[E.MOMENTS],
                    measurements=measurements, **kw)
    f['guardrails'] = {k: 'pass' for k in S.GUARDRAILS}
    f['guardrails']['evidence'] = 'fail'
    f['measurements'].update(clearMoments=clear, clearSpells=spells, decidedMoments=decided, failingMoments=failing)
    f['aggregate'].update(demonstrated=decided-failing, notDemonstrated=failing, unclear=n-decided,
                          distinctFailingSpells=spells)
    return f


class MomentsWorthALookTests(unittest.TestCase):
    """Fix round 2 (LEAD decision (b); ASSESS's request): the words of a 'moments worth a look' review. Round 1 worded
    it as any review ("Possible issue to review: ...", "3 of 12 forward runs" over all opportunities with 1 unclear,
    "Not confirmed: enough evidence for a work item"); on the Metrica game-1 and planted caches (fix2/text, 41 of these
    reviews at levels 3-7) the text now says "Moments worth a look: ...", the clear moments of the opportunities
    PitchIQ could judge, the unclear ones, that it stays below the verdict bar, every moment, the first one, why the
    check matters and what to look for in the clips; no verdict wording, zone or drill."""

    VERDICT_RX = E.VERDICT_WORDS_RE

    def one(self, f, prof, **kw):
        log = []
        out = run([f], prof, log=log, **kw)
        self.assertEqual(out['withheld'], [], log)
        return out['findings'][0], log[0][1]

    def test_every_detector_level_age_and_format_is_worded_and_passes(self):
        """Also without any confirmed role: "the holding midfielder" in the level-5 action named a role the claims
        did not confirm and the gate withheld 2 Metrica game-1 reviews (window 2850 s, team B, L5 U16 / U18)."""
        for det in S.DETECTORS:
            for lvl in (3, 4, 5, 6, 7):
                for age, fmt, roles in ((None, '11v11', None), (ADULT_AGE, '11v11', None), (None, '7v7', None),
                                        (None, '11v11', {})):
                    prof = profile_for(lvl, age, fmt=None if fmt == '11v11' else fmt)
                    f = moments_finding(det, lvl, fmt=fmt, roles=roles)
                    with self.subTest(detector=det, level=lvl, age=age, fmt=fmt, roles=roles):
                        g, how = self.one(f, prof, alternatives=[alternative_for(f)])
                        self.assertEqual(how, 'planned', g['text'])
                        self.assertEqual(S.validate('finding', g), [])
                        self.assertEqual(E.validate(g['text'], g['claims'], g, prof), [])
                        self.assertEqual(E.invariant_problems(g), [])
                        t = g['text']
                        self.assertTrue(t['headline'].startswith('Moments worth a look: '), t['headline'])
                        self.assertRegex(t['explanation'], r'\b3 of (?:the )?11\b')
                        self.assertNotRegex(t['explanation'], r'\b\d+ of (?:the )?12\b')    # never all opportunities
                        self.assertRegex(t['explanation'], unclear_rx(r'(?:One more|1)'))     # 1 of the 12 unclear
                        self.assertRegex(t['explanation'], r'bar for a verdict|verdict bar')
                        # not held back by evidence it lacks: round 1's "Not confirmed: enough evidence for a work item"
                        self.assertNotRegex(t['explanation'], r'enough evidence|firm conclusion|enough moments')
                        self.assertIn('Why it matters: ', t['explanation'])
                        for ts in g['timestamps']:
                            self.assertIn(L.fmt_time(ts['t']), t['explanation'])
                        self.assertTrue(t['action'].startswith('In each clip, '), t['action'])
                        for k in E.TEXT_FIELDS:
                            self.assertNotRegex(t[k] or '', self.VERDICT_RX)
                        # what to look for, never a better position or why none is drawn
                        self.assertFalse([c for c in g['claims'] if c['type'] == 'counterfactual'])
                        self.assertNotIn('marked', t['action'])
                        self.assertEqual(g['drills'], [])

    def test_the_minimal_plan_passes_too(self):
        """The deterministic fallback when the planned text is refused: counts, the verdict bar, the moments and a
        plain look-for line, for every detector and level."""
        for det in S.DETECTORS:
            for lvl in (3, 4, 5, 6, 7):
                prof = profile_for(lvl)
                f = moments_finding(det, lvl, roles={})
                reg = E.register_for(prof, f)
                claims = E.build_claims(f, prof, register=reg)
                text = E._realize(f, claims, reg, E._ReportState(), minimal=True)
                with self.subTest(detector=det, level=lvl):
                    self.assertIsNotNone(text)
                    self.assertEqual(E.validate(text, claims, f, prof, register=reg), [])
                    g = dict(f, text=text, claims=claims)
                    self.assertEqual(E.moments_problems(g), [])

    def test_shares_from_level_five_counts_below(self):
        for lvl, share in ((3, False), (4, False), (5, True), (6, True), (7, True)):
            g, _ = self.one(moments_finding('wide_defender_advance', lvl), profile_for(lvl))
            with self.subTest(level=lvl):
                if share:
                    self.assertIn('(about 27%)', g['text']['explanation'])
                    self.assertIn('(about 73%)', g['text']['explanation'])
                else:
                    self.assertNotIn('%', g['text']['explanation'])

    def test_counts_said_plainly_at_the_ends(self):
        """3 of 3 judged (the planted D1 clips: 6 more unclear) reads "all 3", and with none judged as shown the
        sentence about the shown ones is left out rather than "0 of the 3"."""
        for lvl in (3, 4, 6):
            f = moments_finding('pressure_cover', lvl, clear=3, decided=3, failing=3, n=9)
            g, _ = self.one(f, profile_for(lvl))
            e = g['text']['explanation']
            with self.subTest(level=lvl):
                self.assertRegex(e, r'\b[Aa]ll 3\b')
                self.assertRegex(e, unclear_rx('6'))
                self.assertNotIn('did what the check asks', e)
                self.assertNotIn('met in', e)
                self.assertNotRegex(e, r'\b0 of\b')
                self.assertEqual(E.invariant_problems(g), [])
        g, _ = self.one(moments_finding('behind_ball', 4, decided=12, n=12), profile_for(4))
        self.assertNotRegex(g['text']['explanation'], unclear_rx(r'(?:One more|\d+)'))
        # 5 clear moments, 3 cited (planted D1 clips: "In all 5 ... The moments to watch are at" three times)
        for lvl in (3, 5, 6, 7):
            g, _ = self.one(moments_finding('pressure_cover', lvl, clear=5, decided=6, failing=5, spells=5),
                            profile_for(lvl))
            with self.subTest(level=lvl, shown=3):
                self.assertRegex(g['text']['explanation'], r'\b3 (?:of them|shown)\b')
        g, _ = self.one(moments_finding('pressure_cover', 4), profile_for(4))
        self.assertNotRegex(g['text']['explanation'], r'\b3 (?:of them|shown)\b')

    def test_pressure_and_cover_counts_hold_for_every_mode(self):
        """The cited moments of one review may fail either way (Metrica game 1, F/A/whole: nobody near the carrier at
        1153 s and 4603 s, a presser without cover at 3163 s); the count, the headline and every moment's caption are
        mode-free, and the first moment is described by its own flags, not the modal mode."""
        for lvl in (3, 4, 5, 6, 7):
            m = dict(FAIL_M['pressure_cover'], failureMode='no_cover', pressurePresent=False, coverPresent=None,
                     d1M=8.6)
            g, _ = self.one(moments_finding('pressure_cover', lvl, measurements=m), profile_for(lvl))
            t = g['text']
            with self.subTest(level=lvl):
                self.assertEqual(E.failure_mode(g, lvl), 'no_pressure')
                self.assertEqual(g['claims'][0]['facts']['mode'], 'no_pressure')
                self.assertNotRegex(t['short'].lower(), r'unpressed|without cover|no cover|covering')
                self.assertIn('pressure on the ball and cover behind', t['explanation'].lower())
                self.assertNotRegex(t['explanation'], r'without covering support|outside a covering position')

    def test_the_count_is_of_the_judged_opportunities(self):
        """ASSESS's request: round 1 printed "3 of 12" from aggregate.opportunities (12 with 1 unclear) where 3 of the
        11 judged were clear; the invariant refuses the old text."""
        f = moments_finding('wide_defender_advance', 4)
        g, _ = self.one(f, profile_for(4))
        old = copy.deepcopy(g)
        old['text']['explanation'] = 'Over the clip: 3 of 12 advances, in 3 separate spells. '+' '.join(
            L.fmt_time(ts['t']) for ts in g['timestamps'])
        probs = E.moments_problems(old)
        self.assertTrue(any('3 of 11' in p for p in probs), probs)
        self.assertTrue(any('of 12' in p for p in probs), probs)
        for change, want in ((lambda x: x['text'].update(headline='Possible issue to review: advance'),
                              'Moments worth a look'),
                             (lambda x: x.update(drills=['drill-1']), 'drill'),
                             (lambda x: x['text'].update(action='Work on covering the channel.'), 'verdict wording'),
                             (lambda x: x['text'].update(explanation=x['text']['explanation'].replace(
                                 L.fmt_time(x['timestamps'][1]['t']), 'later')), 'is not given'),
                             (lambda x: x.update(priority=1), 'priority')):
            bad = copy.deepcopy(g)
            change(bad)
            self.assertTrue(any(want in p for p in E.moments_problems(bad)), (want, E.moments_problems(bad)))
        self.assertEqual(E.moments_problems(g), [])
        self.assertEqual(E.moments_problems(run([finding_for('wide_defender_advance', 'review', 4)],
                                                profile_for(4))['findings'][0]), [])

    def test_other_reviews_keep_their_wording(self):
        """A review held back by what could not be confirmed is still "Possible issue to review" (spec Part 15)."""
        for lvl in (3, 5, 7):
            g = run([finding_for('pressure_cover', 'review', lvl)], profile_for(lvl))['findings'][0]
            self.assertTrue(g['text']['headline'].startswith('Possible issue to review: '))
            self.assertFalse(E.is_moments(g))
        f = moments_finding('pressure_cover', 4)
        del f['measurements']['decidedMoments']           # no counts: never invented, worded as any review
        g, _ = self.one(f, profile_for(4))
        self.assertTrue(g['text']['headline'].startswith('Possible issue to review: '))
        self.assertTrue(any('without its counts' in p for p in E.moments_problems(g)))

    def test_short_ranges_use_no_repeated_pattern_words(self):
        for det in S.DETECTORS:
            for lvl in (3, 4, 6):
                f = moments_finding(det, lvl, limitations=['short_range'])
                g, _ = self.one(f, profile_for(lvl), range={'t0': 700.0, 't1': 879.0})
                body = ' '.join(v for v in g['text'].values() if v).lower()
                with self.subTest(detector=det, level=lvl):
                    for w in L.SHORT_BANNED:
                        self.assertNotRegex(body, r'\b'+re.escape(w)+r'\b')
                    self.assertIn('long enough', body)
                    self.assertNotIn('stays a review because', body)

    def test_the_side_is_named_for_a_wide_defender(self):
        """Also beside the other side's strength in the same report (Metrica game 1, window 2850 s, team B: the
        strength's short label took the shared phrasing slot and the moments label lost "on the right")."""
        for lvl, want in ((3, 'Cover behind a forward run on the left'), (5, 'Advance without cover on the left')):
            g, _ = self.one(moments_finding('wide_defender_advance', lvl), profile_for(lvl))
            self.assertEqual(g['text']['short'], want)
            self.assertEqual(g['text']['headline'], E.MOMENTS_PREFIX+want[:1].lower()+want[1:])
            other = finding_for('wide_defender_advance', 'strength', lvl, t0=1300.0)
            out = run([other, moments_finding('wide_defender_advance', lvl)], profile_for(lvl))
            self.assertEqual(out['withheld'], [])
            self.assertEqual(out['findings'][1]['text']['short'], want)

    def test_no_federation_word_in_any_wording(self):
        """EVAL fix round 1: "Rest defence here is a PitchIQ measure, not a federation definition" put a federation
        word in 84 run-copy overview texts (fedscan.py, sources_safety verifier). No phrasing explain can emit names
        a federation or a governing body."""
        fed = re.compile(r'\b(U\.S\. Soccer|US Soccer|England Football|The FA|FA|Canada Soccer|UEFA|Concacaf|'
                         r'federation)\b')
        texts = [d[3] for d in E.DEFINITIONS]
        texts += [t for v in E.PRINCIPLES.values() for t in v['text'].values()]
        texts += [t for bank in E.BANK.values() for slot in bank.values() for v in slot.values()
                  for t in (v if isinstance(v, (list, tuple)) else [v])]
        texts += [w for pair in E.LIMITATION_WORDS.values() for w in pair]
        texts += [w for pair in E.NO_ALT_WORDS.values() for w in pair]
        self.assertGreater(len(texts), 700)
        self.assertEqual([t for t in texts if fed.search(t)], [])
        self.assertIn('Rest defence here is a PitchIQ measure', E.DEFINITIONS[1][3])


class WideDefenderCoverTests(unittest.TestCase):
    """Polish round (Oct 23; fix2/text.md's withheld findings): a wide defender's strength says who covered at its
    moment, as the level's rule judged it (contextual: the channel, the near centre-back's shift, a holding
    midfielder's drop where the level counts it, an interchange), and its label names the side. Before, every level
    4-7 strength said "{mate} was in the channel behind them": the gate withheld the strengths whose channel was empty
    while the centre-back covered (2 of the 3 findings withheld in 12 reports of the cached Metrica game-1 and planted
    tiers; on the planted D3 clip B-04 advanced 25 m with the channel empty and the near centre-back 8.4 m across)."""

    CB = {'channelOccupied': False, 'channelBy': None, 'cbAdjusted': True, 'nearCbShiftM': 8.4, 'dmDropped': False}
    CHANNEL = {'channelOccupied': True, 'channelBy': 'A-06', 'cbAdjusted': False, 'nearCbShiftM': 1.0,
               'dmDropped': False}
    # fix round 3 (FACTS F-8): the interchange that covers is the geometric one at the moment (interchangeCover)
    SWAP = {'channelOccupied': False, 'channelBy': None, 'cbAdjusted': False, 'nearCbShiftM': 1.2, 'dmDropped': False,
            'interchange': True, 'interchangeCover': True}
    CB_RX = re.compile(r'(?:centre-back|central defender) (?:moved|shifted) across')
    CHANNEL_RX = E.POLARITY_RE['channel_cover'][True]

    def strength(self, upd, lvl, fmt='11v11', **kw):
        m = copy.deepcopy(OK_M['wide_defender_advance'])
        m.update(upd)
        f = finding_for('wide_defender_advance', 'strength', lvl, measurements=m, fmt=fmt, **kw)
        log = []
        out = run([f], profile_for(lvl, fmt=None if fmt == '11v11' else fmt), log=log)
        self.assertEqual(out['withheld'], [], log)
        return out['findings'][0]

    def first_sentence(self, g):
        return re.split(r'(?<=\.)\s', g['text']['explanation'])[0]

    def test_the_centre_back_who_covered_is_named(self):
        for fmt in ('11v11', '9v9', '7v7'):
            for lvl in (3, 4, 5, 6, 7):
                with self.subTest(fmt=fmt, level=lvl):
                    g = self.strength(self.CB, lvl, fmt)
                    first = self.first_sentence(g)
                    self.assertRegex(first, self.CB_RX)
                    self.assertNotRegex(g['text']['explanation'], self.CHANNEL_RX)
                    cb = [c for c in g['claims'] if c['type'] == 'relation' and c['facts'].get('relation') ==
                          'cb_shift']
                    self.assertEqual([c['facts']['present'] for c in cb], [True])
                    self.assertEqual(E.validate(g['text'], g['claims'], g, profile_for(lvl, fmt=None if fmt == '11v11'
                                                                                         else fmt), short=False), [])
                    # the centre-back is not said twice (the levels 5-7 context sentence leaves it out)
                    self.assertEqual(len(self.CB_RX.findall(g['text']['explanation'])), 1)

    def test_the_teammate_in_the_channel_is_named(self):
        for lvl in (4, 5, 6, 7):
            with self.subTest(level=lvl):
                g = self.strength(self.CHANNEL, lvl)
                self.assertRegex(self.first_sentence(g), self.CHANNEL_RX)
                self.assertNotRegex(self.first_sentence(g), self.CB_RX)

    def test_an_interchange_is_said_right_after_the_advance(self):
        for lvl in (3, 4, 5, 6, 7):
            with self.subTest(level=lvl):
                g = self.strength(self.SWAP, lvl)
                second = re.split(r'(?<=\.)\s', g['text']['explanation'])[1]
                self.assertRegex(second, r'took (?:their place|the vacated spot)|was taken by a teammate')
                self.assertNotRegex(g['text']['explanation'], self.CHANNEL_RX)
                self.assertNotRegex(self.first_sentence(g), self.CB_RX)

    def test_the_cover_follows_the_levels_shift_rule(self):
        m = dict(self.CB, cbAdjusted=False, nearCbShiftM=5.0)       # the stored flag (4 m) and a level asking 6 m
        self.assertTrue(E.d3_cb_covered(dict(m, cbAdjusted=True), {'cbShiftM': 4.0}))
        self.assertFalse(E.d3_cb_covered(m, {'cbShiftM': 4.0}))
        self.assertFalse(E.d3_cb_covered(dict(m, cbAdjusted=True), {'cbShiftM': 6.0}))
        self.assertTrue(E.d3_cb_covered(m, {'cbShiftM': 4.5}))
        self.assertEqual(E.d3_cover_by(self.CHANNEL, {'cbShiftM': 4.0}), 'channel')
        self.assertEqual(E.d3_cover_by(self.CB, {'cbShiftM': 4.0}), 'cb')
        self.assertEqual(E.d3_cover_by(dict(self.SWAP, dmDropped=True), {'useDmRotation': True}), 'dm')
        self.assertEqual(E.d3_cover_by(dict(self.SWAP, dmDropped=True), {'useDmRotation': False}), 'swap')
        self.assertIsNone(E.d3_cover_by(dict(self.SWAP, interchangeCover=False), {}))
        # a role-label swap within 10 s alone is no cover (FACTS F-8: 17 of 54 interchange claims had nobody in the
        # vacated spot)
        self.assertIsNone(E.d3_cover_by(dict(self.SWAP, interchangeCover=None), {}))

    def test_the_label_names_the_side(self):
        for lvl, want in ((3, 'Covered forward run on the left'), (4, 'Supported advance on the left'),
                          (7, 'Supported advance on the left')):
            with self.subTest(level=lvl):
                g = self.strength(self.CB, lvl)
                self.assertEqual(g['text']['short'], want)
                self.assertIn('on the left', g['text']['headline'])
        for lvl in (1, 2):              # team-level at levels 1-2: "Someone back", no side
            self.assertEqual(self.strength(self.CB, lvl)['text']['short'], 'Someone back')

    def test_both_sides_keep_their_labels_in_one_report(self):
        for lvl in (3, 5):
            left = finding_for('wide_defender_advance', 'strength', lvl)
            m = dict(copy.deepcopy(OK_M['wide_defender_advance']), side='right', **self.CB)
            right = finding_for('wide_defender_advance', 'strength', lvl, measurements=m, t0=1300.0,
                                identity='fail')
            out = run([left, right], profile_for(lvl))
            with self.subTest(level=lvl):
                self.assertEqual(out['withheld'], [])
                shorts = sorted(g['text']['short'] for g in out['findings'])
                self.assertEqual([s.rsplit(' ', 1)[-1] for s in shorts], ['left', 'right'])


def obs_scaled(findings, length=105.0, usable=True):
    """obs_for with the run's coaching scale (explain reads the behind-the-ball margin from it: 10 m x L / 105)."""
    o = obs_for(findings)
    o['timeline'] = {'scale': {'lengthM': length, 'widthM': 68.0, 'metresUsable': usable, 'source': 'declared'}}
    return o


def advance_finding(verdict, lvl, **m):
    """Level 1's reframed advance (assess family 'advance' of behind_ball, fix round 3): the advance's relations."""
    meas = {'behindN': 2, 'minBehind': 1, 'threatN': 1, 'restMargin': 1, 'side': 'left', 'advanceM': 20.0,
            'channelOccupied': False, 'opponentsInChannel': 1, 'metres': 'estimated'}
    meas.update(m)
    f = finding_for('behind_ball', verdict, lvl, measurements=meas)
    f['key'] = f['key'].replace('|team|', '|advance|')
    return f


class FixRoundThreeTests(unittest.TestCase):
    """Fix round 3 (Oct 24; critic B-3 and the TEXT items of its fix list, FACTS F-1 to F-11, COACH F-3, F-10, F-11;
    ASSESS's requests (a)-(d)). Each rule has a test that it holds on the planned text and a test that its invariant
    (explain.fix3_problems / moments_problems, run by coaching_eval.report_problems for game-3 G1+) refuses the old
    false form. Measured before / after on scenes, run copies, the user's runs and Metrica game 1 (scratch
    coaching2/fix3/text)."""

    def one(self, f, prof, **kw):
        log = kw.pop('log', [])
        kw.setdefault('obs', obs_scaled([f]))
        out = run([f], prof, log=log, **kw)
        self.assertEqual(out['withheld'], [], log)
        # the planned text, never the minimal plan (the gate refused "At least 10 m behind the ball we had ..." as an
        # estimate without "about" until the threshold rule read a capital)
        self.assertEqual(log[0][1], 'planned', log)
        g = out['findings'][0]
        self.assertEqual(E.validate(g['text'], g['claims'], g, prof), [])
        self.assertEqual(E.invariant_problems(g), [])
        return g

    def bad(self, g, **text):
        h = copy.deepcopy(g)
        h['text'].update(text)
        return h

    def test_the_margin_is_tactics_geom_s(self):
        from soccer.coaching import tactics_geom as G
        self.assertEqual((E.BEHIND_MARGIN_M, E.BEHIND_REF_LENGTH), (G.BEHIND_BALL_M, G.REF_LENGTH))
        self.assertEqual(E.behind_margin(obs_scaled([], 105.0)), 10.0)
        self.assertEqual(E.behind_margin(obs_scaled([], 63.0)), 6.0)
        self.assertIsNone(E.behind_margin({}))

    def test_counts_behind_the_ball_state_their_margin(self):
        """FACTS F-1 = COACH F-2: "none of our players were behind the ball" was said with 3 defenders 4.2 m
        goal-side of it; the count is of players at least 10 m x L / 105 behind it."""
        seen = 0
        for det, verdict in (('wide_defender_advance', 'work_on'), ('wide_defender_advance', 'review'),
                             ('behind_ball', 'work_on'), ('behind_ball', 'strength'), ('behind_ball', 'describe')):
            for lvl in (3, 4, 5, 6, 7):
                g = self.one(finding_for(det, verdict, lvl), profile_for(lvl))
                e = g['text']['explanation']
                said = [s for s in re.split(r'(?<=\.)\s', e) if E.OBSERVED_RE.search(s)]   # (not the principle)
                with self.subTest(detector=det, verdict=verdict, level=lvl):
                    for s in said:
                        self.assertNotRegex(s, r'(?:ours|players?|we had \w+) (?:(?:were|was|stayed) )?behind the ball')
                    seen += e.count('at least 10 m behind the ball')
        self.assertGreater(seen, 20)
        # a smaller pitch scales the margin; unusable metres and levels below 3 say "well"
        g = self.one(finding_for('behind_ball', 'work_on', 4), profile_for(4), obs=obs_scaled([], 63.0))
        self.assertIn('at least 6 m behind the ball', g['text']['explanation'])
        m = dict(FAIL_M['behind_ball'], metres='unusable')
        g = self.one(finding_for('behind_ball', 'work_on', 4, measurements=m), profile_for(4))
        self.assertIn('well behind the ball', g['text']['explanation'])
        g = self.one(finding_for('behind_ball', 'describe', 2), profile_for(2))
        self.assertNotRegex(g['text']['explanation'], r'\d+ m')

    def test_the_margin_is_a_threshold_not_an_estimate(self):
        """"at least 10 m" is the rule's own distance: no "about", at the start of a sentence too; any other metre
        value of an estimated scale still needs it."""
        for lvl in (3, 5):
            p = profile_for(lvl)
            g = self.one(finding_for('wide_defender_advance', 'work_on', lvl), p)
            for extra, ok in (('At least 10 m behind the ball we had 2 outfield players.', True),
                              ('2 of our outfield players were at least 10 m behind the ball.', True),
                              ('2 of our outfield players were 10 m behind the ball.', False),
                              ('2 of our outfield players were at least 11 m behind the ball.', False)):
                text = dict(g['text'], explanation=g['text']['explanation']+' '+extra)
                with self.subTest(level=lvl, text=extra):
                    self.assertEqual(E.validate(text, g['claims'], g, p) == [], ok,
                                     E.validate(text, g['claims'], g, p))

    def test_the_old_behind_the_ball_count_is_refused(self):
        g = self.one(finding_for('wide_defender_advance', 'work_on', 3), profile_for(3))
        for old in ('None of our players were behind the ball at that moment.',
                    'Rest defence at that moment: 2 of ours behind the ball against 3 of theirs, a margin of -1.',
                    'Behind the ball we had 2 players.'):
            with self.subTest(text=old):
                h = self.bad(g, explanation=g['text']['explanation']+' '+old)
                self.assertTrue(any('margin' in p for p in E.fix3_problems(h)), old)

    def test_level_seven_shape_leaves_out_the_back_line_build_up(self):
        """The 5 real-clip level-7 shape descriptions said "none of ours behind the ball" with the ball at our own
        back line (deepN 0: our deepest outfielder within the margin of it): build-up, not missing support."""
        base = dict(OK_M['attacking_shape'], heightGapM=12.0)
        g = self.one(finding_for('attacking_shape', 'describe', 7, measurements=dict(base, deepN=0)), profile_for(7))
        self.assertNotRegex(g['text']['explanation'], r'of ours[^.]*behind the ball')
        g = self.one(finding_for('attacking_shape', 'describe', 7, measurements=dict(base, deepN=3)), profile_for(7))
        self.assertIn('3 of ours at least 10 m behind the ball', g['text']['explanation'])
        h = self.bad(g, explanation='At the moment shown our highest player was about 12 m short of their last line, '
                                    'with none of ours at least 10 m behind the ball.')
        h['measurements']['deepN'] = 0
        self.assertTrue(E.fix3_problems(h))

    def test_rest_defence_is_defined_with_the_margin(self):
        g = self.one(finding_for('wide_defender_advance', 'work_on', 6), profile_for(6))
        self.assertIn(E.definition_text(E.DEFINITIONS[1][3], 'at least 10 m'), g['text']['explanation'])
        self.assertIn('the opponents at least 10 m on our side of it', g['text']['explanation'])

    def test_the_moment_s_own_actor_and_gap_pair_are_named(self):
        """FACTS F-2: D3 named the family's modal wide defender, D5 the family's gap partner."""
        m = dict(FAIL_M['wide_defender_advance'], actorId='A-07')
        f = finding_for('wide_defender_advance', 'work_on', 5, measurements=m)
        f['players']['supporting'].append('A-07')
        f['roles']['A-07'] = {'role': 'LB', 'source': 'coach', 'confidence': 'high'}
        self.assertEqual(E.moment_actor(f), 'A-07')
        g = self.one(f, profile_for(5))
        self.assertEqual(E._first(g['claims'], 'actor_state')['player'], 'A-07')
        self.assertNotIn('(A-02)', g['text']['explanation'])
        h = copy.deepcopy(g)
        E._first(h['claims'], 'actor_state')['player'] = 'A-02'
        self.assertTrue(any('first moment' in p for p in E.fix3_problems(h)))
        m = dict(FAIL_M['defensive_spacing'], actorId='A-04', gapLeftId='A-04', gapRightId='A-08')
        f = finding_for('defensive_spacing', 'work_on', 5, measurements=m)
        self.assertEqual(E.gap_partner(f), 'A-08')
        g = self.one(f, profile_for(5))
        self.assertEqual(E._rel(g['claims'], 'lateral_gap')['player'], 'A-08')
        h = self.bad(g, explanation='The back line left a large gap on the ball side between A-04 and A-03.')
        self.assertTrue(any('pair' in p for p in E.fix3_problems(h)))

    def test_the_place_is_the_first_moment_s(self):
        """FACTS F-5: "in the final third, the right back pushed up" with the ball at X 36.5 m (the family's modal
        zone)."""
        m = dict(FAIL_M['pressure_cover'], momentZone='middle_third')
        g = self.one(finding_for('pressure_cover', 'work_on', 4, measurements=m), profile_for(4))
        self.assertIn('in the middle third', g['text']['explanation'])
        self.assertNotIn('in our own third', g['text']['explanation'])
        h = copy.deepcopy(g)
        E._first(h['claims'], 'situation')['zone'] = 'own_third'
        self.assertTrue(any('first moment was in' in p for p in E.fix3_problems(h)))
        # level 1's reframed advance is placed at its moment, not "near the other team's goal" by default
        g = self.one(advance_finding('try_this', 1, failureMode='exposed', momentZone='own_third'), profile_for(1))
        self.assertNotIn("near the other team's goal", g['text']['explanation'])

    def test_strengths_count_the_judged_moments(self):
        """FACTS F-6 = COACH F-3: "2 of the 15 moments with pressure on the ball looked like this" for a strength
        the rule decided on 2 of 3 (12 unclear)."""
        for lvl, want in ((2, '2 of the 3 defending moments'), (4, '2 of the 3'), (5, 'about 67%'),
                          (7, '2 of the 3')):
            f = finding_for('pressure_cover', 'strength', lvl)
            f['aggregate'].update(opportunities=15, demonstrated=2, notDemonstrated=1, unclear=12)
            g = self.one(f, profile_for(lvl))
            e = g['text']['explanation']
            with self.subTest(level=lvl):
                self.assertIn(want, e)
                self.assertIn('12 more could not be judged', e)
                self.assertNotRegex(e, r'\b2 (?:times )?(?:of|out of) (?:the )?15\b')
                self.assertNotRegex(e, r'moments with pressure on the ball|pressing moments')
                h = self.bad(g, explanation='Pressure and cover came 2 of the 15 moments with pressure on the ball.')
                self.assertTrue(E.fix3_problems(h))

    def test_the_cover_candidate_is_the_nearest_within_the_cover_angle(self):
        """FACTS F-7: "the nearest teammate, B-03, was about 13 m away" while B-07 stood nearer, 69.5 degrees off."""
        for lvl in (3, 4, 5, 6, 7):
            g = self.one(finding_for('pressure_cover', 'work_on', lvl), profile_for(lvl))
            said = [s for s in re.split(r'(?<=\.)\s', g['text']['explanation']) if E.OBSERVED_RE.search(s)]
            with self.subTest(level=lvl):
                self.assertNotRegex(' '.join(said), r'nearest teammate(?! within the cover angle)')
        h = self.bad(g, explanation='A-04 pressed the carrier at about 3 m, and the nearest teammate, A-05, was about '
                                    '14 m away, outside a covering position.')
        self.assertTrue(any('cover angle' in p for p in E.fix3_problems(h)))

    def test_an_interchange_is_said_only_when_it_covered_at_the_moment(self):
        """FACTS F-8: "a teammate took their place" rested on a role-label swap within 10 s (17 of 54)."""
        m = dict(OK_M['wide_defender_advance'], channelOccupied=False, channelBy=None, cbAdjusted=False,
                 nearCbShiftM=1.0, dmDropped=False, interchange=True, interchangeCover=False)
        for lvl in (3, 5, 7):
            g = self.one(finding_for('wide_defender_advance', 'strength', lvl, measurements=m), profile_for(lvl))
            with self.subTest(level=lvl):
                self.assertNotRegex(g['text']['explanation'], E.INTERCHANGE_RE)
        h = self.bad(g, explanation=g['text']['explanation']+' A teammate took their place on the left.')
        self.assertTrue(any('geometric' in p for p in E.fix3_problems(h)))

    def test_the_moments_counts_add_up(self):
        """FACTS F-4 = COACH F-10: "In 3 of the 17 ... The team did what the check asks in 9 of the 17" left 5
        failing moments unsaid."""
        for det in S.DETECTORS:
            for lvl in (3, 4, 5, 6, 7):
                f = moments_finding(det, lvl, clear=3, decided=17, failing=8, n=20)
                g = self.one(f, profile_for(lvl), alternatives=[alternative_for(f)])
                e = g['text']['explanation']
                with self.subTest(detector=det, level=lvl):
                    self.assertRegex(e, r'\b5\b[^.]*(?:fell short|shortfalls)|shortfalls[^.]*: 5\b')
                    self.assertEqual(E.moments_problems(g), [])
                    cut = ' '.join(s for s in re.split(r'(?<=\.)\s', e) if not re.search(r'fell short|shortfalls', s))
                    self.assertTrue(any('less clearly' in p for p in E.moments_problems(self.bad(g, explanation=cut))))

    def test_levels_one_two_word_the_exposed_advance(self):
        """Critic B-1 / ASSESS (a): levels 1-2 judge an advance left exposed wherever the ball is; with players
        behind the ball, "nobody stayed back" was false and the gate withheld it (one_behind, channel_threat,
        stacking at levels 1-2)."""
        g = self.one(advance_finding('try_this', 1, failureMode='exposed'), profile_for(1))
        e = g['text']['explanation']
        self.assertIn('one of theirs', e)
        self.assertNotRegex(e, r'nobody stayed back|no one stayed back|nobody back')
        m = dict(FAIL_M['wide_defender_advance'], behindN=2, failureMode='exposed', opponentsInChannel=2)
        g = self.one(finding_for('wide_defender_advance', 'try_this', 2, measurements=m), profile_for(2))
        e = g['text']['explanation']
        self.assertRegex(e, r'nobody covered the space they left|space they left was not covered')
        self.assertIn('2 of theirs', e)
        self.assertNotRegex(e, r'stayed back')

    def test_levels_one_two_praise_a_covered_advance(self):
        """ASSESS (b): a covered advance with nobody behind the ball was withheld at level 1 ("someone stayed
        back" against behindN 0): BC60 B and the user's run 636e03 B."""
        f = advance_finding('strength', 1, behindN=0, channelOccupied=True, opponentsInChannel=0)
        log = []
        g = self.one(f, profile_for(1), log=log)
        self.assertEqual(log[0][1], 'planned', g['text'])       # (BC60 B: the headline was refused, minimal plan)
        self.assertRegex(g['text']['explanation'], r'looked after the space|took care of the space|was there for')
        self.assertNotRegex(' '.join(v or '' for v in g['text'].values()), r'stayed back')
        m = dict(OK_M['wide_defender_advance'], behindN=0)
        log = []
        g = self.one(finding_for('wide_defender_advance', 'strength', 2, measurements=m), profile_for(2), log=log)
        self.assertEqual(log[0][1], 'planned', g['text'])
        self.assertRegex(g['text']['explanation'], r'space they left was covered|covered the space they left')
        self.assertNotRegex(' '.join(v or '' for v in g['text'].values()), r'stayed back')

    def test_the_mixed_evidence_base_is_never_professional_for_youth(self):
        """LEAD decision (3), COACH F-5: "at or above the professional base rate PitchIQ measured" at U16."""
        m = dict(D2_FULL, baseShare=.048, baseTail=.012)
        for lvl, age, pro in ((3, None, False), (4, None, False), (5, None, False), (5, 'Adult', True),
                              (6, 'U17', False), (7, None, True)):
            f = finding_for('weak_side_balance', 'describe', lvl, measurements=m, limitations=['mixed_evidence'])
            f['aggregate'].update(demonstrated=9, notDemonstrated=3, unclear=1)
            g = self.one(f, profile_for(lvl, age))
            with self.subTest(level=lvl, age=age):
                if not pro:
                    self.assertNotIn('professional', ' '.join(v or '' for v in g['text'].values()))
                self.assertNotIn('at or above', g['text']['explanation'])

    def test_whole_pitch_and_the_ball_s_side(self):
        """FACTS F-11: "we used the whole pitch: tall and deep" at 24% of the length; "2 of theirs standing in that
        space" counted on the ball's half."""
        for slot in ('open_ok', 'headline_ok'):
            for lvl in (1, 2):
                for t in E.BANK['attacking_shape'][slot][next(k for k in E.BANK['attacking_shape'][slot] if lvl in k)]:
                    self.assertNotRegex(t, r'whole pitch')
        m = dict(OK_M['defensive_spacing'], dmGapOpponents=2)
        g = self.one(finding_for('defensive_spacing', 'describe', 7, measurements=m), profile_for(7))
        self.assertIn("standing in that space on the ball's side", g['text']['explanation'])
        h = self.bad(g, explanation='Between our back line and midfield: 18 m, with 2 of theirs standing in that '
                                    'space.')
        self.assertTrue(E.fix3_problems(h))

    def test_level_one_says_roughly_how_often(self):
        """COACH F-11: "We did this a few times" for 24 of 24 moments."""
        for n, want in ((3, 'a few times'), (7, 'several times'), (24, 'many times')):
            f = finding_for('behind_ball', 'strength', 1, n=n+2)
            g = self.one(f, profile_for(1))
            with self.subTest(n=n):
                self.assertIn(want, g['text']['explanation'])
        h = self.bad(g, explanation='Someone stayed back. We did this a few times.')
        self.assertTrue(E.fix3_problems(h))


class PreFreezeRepeatTests(unittest.TestCase):
    """Freeze gate G-2 (Oct 24): on Metrica game 1 (F, team B, 1080.04-1260.04 s, level 5 at U16 and U18) the left and
    right wide-defender strengths of one 3-minute range were .5119 alike after every salted re-plan (criterion 13:
    two explanations of one report below .5): both stated the same principle, the same holding-midfielder sentence,
    the same counts and judged seconds. The pair is rebuilt here from that report's measurements and aggregates; it is
    .5253 alike with the explain.py of the gate (scratch coaching2/closer/probe_g2_synth.py on the before tree)."""

    M = {'advanceM': 21.1, 'behindN': 4, 'channelOccupied': True, 'cbAdjusted': False, 'nearCbShiftM': 1.8,
         'interchange': False, 'opponentsInChannel': 0, 'threatN': 2, 'restMargin': 2, 'dmCbSpacingM': 19.1,
         'dmDropped': False, 'metres': 'estimated', 'judgedSeconds': 24.8}
    AGG = {'opportunities': 4, 'demonstrated': 3, 'notDemonstrated': 0, 'unclear': 1, 'distinctFailingSpells': 0,
           'distinctDemonstratedSpells': 3, 'phaseSeconds': 48.4, 'eligibleShare': .715, 'ratePer10': 0.0}

    def pair(self, lvl, age=None):
        fs = []
        for side, mate, t0 in (('left', 'A-06', 760.0), ('right', 'A-05', 1300.0)):
            f = finding_for('wide_defender_advance', 'strength', lvl, measurements=dict(self.M, side=side,
                                                                                        channelBy=mate), t0=t0, n=4)
            f['aggregate'] = dict(self.AGG)
            f['context']['ball']['channel'] = 'left'
            fs.append(f)
        prof = profile_for(lvl, age)
        out = run(fs, prof)
        self.assertEqual(out['withheld'], [])
        return out['findings'], prof

    def test_two_sides_of_one_check_stay_below_the_limit(self):
        for lvl in (5, 6, 7):
            for age in (None, 'U18', 'Adult'):
                got, prof = self.pair(lvl, age)
                with self.subTest(level=lvl, age=age):
                    self.assertEqual(len(got), 2)
                    self.assertLess(E.jaccard(got[0]['text']['explanation'], got[1]['text']['explanation']),
                                    E.JACCARD_MAX)
                    self.assertEqual(E.report_problems(got, prof), [])
                    for g in got:
                        # what each side was judged on is kept: its count, the unclear one, its own key moment
                        self.assertIn('all 3 judged advances', g['text']['explanation'])
                        self.assertIn('one more could not be judged', g['text']['explanation'])
                        self.assertIn(L.fmt_time(g['timestamps'][0]['t']), g['text']['explanation'])

    def test_only_a_sentence_the_report_already_says_is_left_out(self):
        got, prof = self.pair(5)
        first, second = (g['text']['explanation'] for g in got)
        principle = E._principle_text(got[1]['claims'], 5)
        self.assertTrue(principle)
        self.assertIn(E._sentence(principle), first)          # the report still says why the check matters, once
        self.assertNotIn(E._sentence(principle), second)
        # nothing else is dropped: written alone, the second side differs from its text here only by that sentence
        alone = run([got[1]], prof)['findings'][0]['text']['explanation']
        self.assertIn(E._sentence(principle), alone)
        self.assertEqual(len(re.findall(r'[.?!](?:\s|$)', alone))-1, len(re.findall(r'[.?!](?:\s|$)', second)))

    def test_a_pair_already_below_the_limit_is_not_replanned(self):
        # the salted re-plans and the texts of every report already under the limit are unchanged: levels 6-7 of the
        # same pair (.42 / .33) keep the principle in both explanations
        for lvl in (6, 7):
            got, _ = self.pair(lvl)
            principle = E._sentence(E._principle_text(got[1]['claims'], lvl))
            with self.subTest(level=lvl):
                self.assertIn(principle, got[0]['text']['explanation'])
                self.assertIn(principle, got[1]['text']['explanation'])


if __name__ == '__main__':
    unittest.main()
