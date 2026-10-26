"""The contextual coaching assessment (DESIGN v2 sections 4.8-4.9): the level's verdicts on the measured
opportunities, with the spec Part 15 guardrails stored on every finding.

Why a separate stage. soccer/coaching/contextual.py finds the situations and measures them once per run, for every
level alike (invariant 2). This stage is the only place the coaching profile (soccer/coaching/profiles.py) meets the
measurements: the same relations are classified with the level's parameters (a four-year-old's "someone goes to the
ball" is closing at 1.5 m/s within 8 m; an adult's pressure is within 5 m with a covering team-mate), counted against
the level's evidence table and turned into a verdict:
  strength  the behaviour was demonstrated: at least the level's distinct demonstrated spells and share (never the
            absence of failures: critic B1 found the v1 "strength" counted time it never compared);
  work_on / try_this   a repeated failure: the level's minimum opportunities, distinct failing spells (10 s apart,
            DESIGN 3.6; under 3 minutes of range at least 3), failure share and relevant phase seconds, and every
            guardrail passed (try_this is the level 1-3 framing, a question);
  review    "possible issue to review": a criticism that failed a guardrail, or a failing pattern with too little
            evidence (one clear instance is a review, never a work item, at every level), or a level whose style is
            unknown (level 7: no work items);
  describe  context only: the level only describes this check, the declared style explains the pattern ("this looked
            intentional"), or no phase was known at all (attacking shape when the whole team was in view);
  none      nothing to say; the detector is still listed as assessed or not assessed with its reason.
Thresholds never adapt to fill a section (spec Part 10): a professional team may get work items and an amateur one
none.

Per opportunity: its in-scope samples (contextual.sample_outcomes with the level's parameters; e.g. carriers outside
the level's zone are out of scope) are demonstrated / not demonstrated / unclear; the opportunity is demonstrated when
at least outcomeShare (.6) of them demonstrate (defensive spacing: strengthTimeShare .7), not demonstrated when at
least .6 fail and the longest failing run (merged across 1 s) lasts spellMinS (2 s; 3 s for D3 / D4; weak side
failMinS), else unclear. Role-based criticism within 15 s of a formation change is suppressed (unclear), a counter-
example of DESIGN 13.1.

Guardrails (spec Part 15; DESIGN 4.8), stored per finding as pass / fail / not_applicable:
  identity   a named player passed the identity gate at the evidence moments (no crossing within 2 s) and the
             opportunities' identity quality is medium or high; team-level checks: not_applicable;
  position   the opportunities' registration quality is medium or high;
  role       role-based rules (D3; D5 on small formats) rest on a coach / stable / carried role or a medium relative
             slot, and the coach's selected formation was not contradicted by a clear reading for a player without a
             coach role (counterexample "selected formation is incorrect");
  context    the relationship's local gate and, for team-level claims, the whole team in view;
  phase      the phase was named (estimated or coach-marked) for every cited opportunity;
  intent     no recovering run, interchange or formation change near the failures, and not explained by the
             declared style;
  alternative  set by coaching.interpret after the zone stage (a validated zone or a principle-only phrasing);
  evidence   the level's evidence table was met.
Any failure on a criticism makes it a review; an identity failure makes explain.py drop names and individual role
words. Priorities rank the work items by the level's relevance weight x distinct failing spells x failure share x
quality and keep the level's cap (the rest become reviews); the next development objective is the top priority's
skill, else the level's next objective from profiles.json, stated as not from this match.

Every number here is a PITCHIQ-PROPOSAL (origin PITCHIQ-PROPOSAL, unvalidated); the level's values come from
profiles.json. Nothing here counts a change of possession or names one (possession docs rule 5).

Fix round 1 (2026-10-23, the critic's B-2, B-3, B-6, S-1, S-3, S-5, S-7, S-8): a demonstration needs a run as long as
a failure (spellMinS); the strength share is over decided opportunities and no bare strength is shown beside failures
at or above the professional base share (baseFailShare, Metrica game 1); the presser named is the moment's first
defender; the level-1 reframed advance keeps the identity guardrail; a review needs half the level's minimum
opportunities and phase seconds (reviewFloorShare), else "too little play" (short_range); a check silenced by a
formation change says so; moments are ranked by how many levels' rules fail or demonstrate them, so one family cites
the same moments at every level; every not-assessed check names why (phase_absent, coach_marks_one_side, ...); D4 may
be praised at levels 1-3, the next objective skips what was shown, the compact style excuses a far-side gap only with
the middle kept. Measured on Metrica game 1, share of team-clips with any criticism (try_this, work_on, review) at
levels 1-7 (typical ages, style unknown), before -> after: full tracking 3-minute clips (62) 6/7/11/16/14/12/12 ->
1/0/1/2/1/0/0, 15-minute clips (12) 0/1/3/2/2/0/0 -> 0/0/1/1/1/0/0; simulated camera 3-minute 0/2/5/5/2/2/2 ->
0/1/1/1/0/1/0, 15-minute 0/0/1/2/2/0/0 -> 0/1/1/1/0/0/0 (every level at or below .083 per 15-minute clip). An
interpretation of one 15-minute team-clip takes 11-20 ms.

Fix round 2 (2026-10-23, LEAD decisions (b) and (c), critic S-8). The verdicts keep their strict majority rules, which
left planted failures almost unseen (professional play fails some checks a third of the time, so 4-5 planted spells
among 20-30 natural opportunities never reach the failure share). (b) From momentsFromLevel (3) a family with no
verdict, review or strength becomes a 'moments worth a look' review (limitation 'moments_worth_a_look') when at least
momentsMinSpells (3) distinct failing spells are clear (_clear_moments: a run of failing samples as long as the failure
spell with no unclear, demonstrated or missing sample in it; for D1 'nobody pressed' also a gap to the carrier that
is not closing at 1 m/s; identity high; the phase from the engine or the coach; every guardrail) and the review floor
holds; it cites only clear moments, carries clearMoments / clearSpells / decidedMoments / failingMoments, gets no
drill, and at most momentsMaxPerReport (1) stay in a report (the others give what their family gives without one);
levels 1-2 are unchanged. (c) 11v11 attacking depth is described, never judged, at levels 1-3 (judgeDepth False).
S-8: the next objective needs a check with at least strengthSpells decided opportunities, a failure among them and a
demonstrated share under strengthShare; else none (the professional-like scene no longer gets "someone ready to
protect our goal" at levels 1-3 from three unclear opportunities). Measured on Metrica game 1 (each clip measured as
an upload, possession-4 on the clip; levels 1-7 at their typical age, style unknown), before -> after:
- team-clips with a verdict (work_on / try_this): unchanged; 0 at every level on full tracking (62 three-minute, 12
  fifteen-minute clips), on the simulated camera 0 at 3 minutes and 0/1/1/0/0/0/0 at 15 minutes (D6 width at levels
  2-3, not depth);
- team-clips with a 'moments worth a look' review: 0 -> 0/0/2/2/2/0/0 of the 12 fifteen-minute clips on full tracking
  (both team B's right wing-back advancing with the channel empty, 3 clear of 9-11 decided advances), 0 of 62 at 3
  minutes, 0 on the camera; reviews of any kind per fifteen-minute clip at most .083 -> .25 (levels 3-5);
- planted failures (10-minute clips, engine roles), clips criticising the planted check on a planted spell at levels
  2 / 4 / 6: the game-3 plan's G12 plants (at least 4 of 5 spells planted) D1 (presser 2 m) full 0/0/0 -> 0/2/2 of 6,
  D3x 0/3/3 -> 0/3/3 of 6, D5 0/0/1 -> 0/3/3 of 7, camera D1 0 -> 0 of 6, D3x 0/1/0 -> 0/1/0, D5 0/0/1 -> 0/0/1 of 7,
  covered twins 0 -> 0; fix round 1's plants with 4-5 spells planted: D1 (3 m) 0/1/0 -> 0/6/6 of 10, D1c (2 m) 0 ->
  0/4/4 of 10, D5 0/0/2 -> 0/6/6 of 11, D5dm 0 -> 0/4/4 of 11, D3x 0/3/3 -> 0/3/3 of 6; level 2 unchanged everywhere;
- the D1 gap rule: the rule's first version cited 6 'nobody pressed' moments (2 team-clips, levels 3-7), all in gaps
  closing from 7.8-14.6 m to 1.5-5.0 m within 2-3 s; with it none remain.
A cap of 2 or none gives the same counts everywhere above (no report had two candidates); requiring no unclear sample
anywhere in the opportunity, not only in the cited run, would cut the level-4 power (fix-1 plants, the 8 ten-minute
clips with 5 spells asked: D1 4 -> 2, D5 3 -> 1; G12 D5 3 -> 1 of 7) and change no professional count. An
interpretation of one 15-minute team-clip takes 14 ms (median of 84; 14 before).

Polish round (2026-10-23): the spec asks for a NEXT DEVELOPMENT OBJECTIVE at every level; after S-8, 182 of the 196
run-copy reports had none. With no work item and no S-8 check, the objective is now the level's suggested focus
(level_focus): "Suggested focus for this level - from the level's guidance, not from this match: <check>. <the
FEDERATION guidance item of the level that states it> Source: <ids>." (FOCUS_GUIDANCE), else the level's general
learning objective (GENERAL_OBJECTIVES, no check, no drill), else the check as PitchIQ's own proposal; never a check
this match demonstrated, the format cannot assess, or that is off (focus_exclusions), never a guidance item naming
another match format or formation (format_misfit), every text through explain.validate_free (focus_problems). drills.py
picks the one optional drill for that check from this objective, as before. Measured on ASSESS's cached Metrica game-1
and planted tiers at TEXT's nine level-3-7 profiles (8829 reports; scratch coaching2/polish, cached_gen.py and
cached_compare.py), before -> after: no objective 6643 -> 34 (levels 7 and 5 at U18, every check shown or excluded and
no general objective), level focus 0 -> 6609 (a check with its source 5381, PitchIQ's own 813, a general objective
415); verdict lists, work items and their drills unchanged; optional drills 2176 -> 8370 (at most one a report, each
for the objective's check); 0 focus texts naming a check with a strength in the report, 0 with words of a weakness;
report problems (coaching_eval.report_problems) 0 -> 0.

Fix round 3 (2026-10-24, LEAD decisions (2) and (3), critic B-1, B-2, S-1, S-2, N1, FACTS F-2 / F-3 / F-5). One
significance rule in both directions (binom_tail: the one-sided binomial tail at baseTestAlpha .05 against the
level's base share): a 'moments worth a look' review needs its clear failing spells (still at least momentsMinSpells
3) above what baseFailShare predicts for the decided opportunities (no measured base, no review; level 3's far-side
check got the level-4 rule's .004), and a strength is withheld ('mixed_evidence') only when its failures are above
praise_base (the professional base from level 5, 1 - (1 - base) x baseSuccessScale .5 / .6 / .7 / .8 at levels 1-4).
A failing sample is clear only in settled play (settle_index / settled: momentsSettleS 2 s after a dead ball and from
any stretch whose team was not named; contextual contestS: momentsContestS 1 s from a loose or contested ball), for
D5 on 11v11 with a line of at least lineMinPlayers 4 and a depth std of at most lineStdMaxM 4 m, and for a lateral gap
with the opponent in it free (gapOppFreeM above markedM 4 m). The wide-defender check's clear advances are their own
review floor, and a check below the floor with enough of its phase reads 'too_few_situations', not 'short_range'.
Levels 1-2 judge an uncovered advance with an opponent in the vacated channel wherever the ball is (contextual).
Findings name the first cited moment's own players and place (_moment_actors: actorId, gapLeftId / gapRightId,
momentZone, interchangeCover; D3 / D5 players.primary). Measured on Metrica game 1 (scratch coaching2/fix3/assess),
before -> after:
- whole-match professional team-reports with a 'moments worth a look' review, levels 3-7 (11 profiles x 2 teams): full
  tracking 22 -> 0 of 22, truth phase 17 -> 0, simulated camera 1 -> 0; halves 16 / 17 / 0 -> 0 of 44; the base test
  alone leaves 2 whole-match ones (both level-7 compact D5), the clear-moment gates alone 5 (team B's right wing-back,
  7 clear of 62 decided against a base of .212); 0 reviews remain whose failing share is at or below the base (7);
- verdicts unchanged everywhere: 0 on the 62 three-minute and 12 fifteen-minute full-tracking clips at every level;
  old-path reviews unchanged; 15-minute 'moments worth a look' at levels 3-5 2 -> 0 of 12 (3 of 9-11 decided against
  .212); levels 1-2: no criticism before or after on any clip, half or whole match, full or camera;
- withheld strengths ('mixed_evidence') on the clips, levels 1-7: full tracking 19/19/25/31/13/5/5 -> 0/0/0/0/1/1/1,
  camera 10/10/11/10/5/3/3 -> 0; whole match L3+ full tracking 28 -> 8;
- the 196 run-copy reports: 0 criticisms before and after; strengths 18 -> 28 (BC180 B's far side 10 of 11 and
  spacing 9 of 10 praised at level 4, spacing also at levels 5-7); the user's 9 runs (252 reports): reviews 1 -> 0 (the
  D5 review of critic N1: its opponents stood 2.83 and 2.89 m from our midfielder, goal-side), strengths 50 -> 64;
- the synthetic scenes: unsupported_advance none_behind / one_behind / final_third, channel_threat, stacking and
  low_calibration fail at levels 1-2 (16 of 16 advances exposed), the covered twins and the professional scene do
  not; spells4 (4 clear advances in 6.1 minutes) is a review at levels 3-7, single_advance 'too_few_situations'.

Pre-freeze (2026-10-24, critic S-3 after the freeze gate; scratch coaching2/closer, before / after on identical inputs):
- FACTS F-9, second half: the next objective and the focus count a check's decided opportunities in its own phase
  only; the phase-unknown attacking-shape family ('unknown': nobody was known to have the ball) no longer chooses
  "Chosen from this match's checks: <attacking shape>" (run copies: AM A / B levels 1-5, 9 of 196 -> 0; the coach
  verifier's 546 scene reports: possession_unknown levels 4-7, 9 -> 0; Metrica game 1: 0, its phase is always known).
- COACH F-9: a general learning objective (GENERAL_OBJECTIVES) is never given to a team older than its source's top
  age (GENERAL_OBJECTIVE_TOP_AGE, fits_age): scene reports with one written for younger players 24 -> 0 (19 level-1
  Adult "every player gets a turn in goal"); they now get the level's check as PitchIQ's own focus (19), or no
  objective where the match showed every check (5); run copies 0 -> 0; Metrica game 1: 7 level-3 Adult reports.
Verdict lists identical in every report (546 scenes, 196 run copies, 2128 game-1 reports).
"""
import bisect
import math
import re
from collections import Counter

from . import contextual as CX
from . import roles as RO
from . import schema as S
from . import situations as SI
from . import tactics_geom as G

MAX_TIMESTAMPS = 3              # evidence moments cited per finding (distinct, at least DISTINCT_S apart when possible)
QUALITY_FACTOR = {'high': 1.0, 'medium': .8, 'low': .5}
LEVEL_RANK = {'high': 0, 'medium': 1, 'low': 2}
VISIBILITY_RANK = {'complete': 0, 'disc': 1, 'partial': 2, 'none': 3}
TEAM_LEVEL = ('weak_side_balance', 'behind_ball', 'attacking_shape')   # no player is the subject of the claim
CONFIDENT = ('coach', 'stable', 'carried')
SKILL_OF = {d: k for k, d in S.SKILLS.items()}
UNKNOWN_STYLE = 'unknown_style_review_only'
DEFAULT_PRINCIPLE_D6 = 'height_width_depth'     # explain.DEFAULT_PRINCIPLE: D6 has no zones (DESIGN 5.2)
NO_OBJECTIVE = 'No next objective could be chosen from this level\'s checks.'
MOMENTS = 'moments_worth_a_look'    # the limitation code of a 'moments worth a look' review (fix round 2)

# ---------- the level's suggested focus (polish round, Oct 23) ----------
# The spec asks for a NEXT DEVELOPMENT OBJECTIVE at every level. When nothing in the match gives one (no work item and
# no check judged often enough with room to improve: critic S-8, fix round 2; 182 of the 196 run-copy reports), the
# report shows the level's own focus from its guidance in profiles.json, labelled as such, never as a weakness of this
# team and never a check this match demonstrated (level_focus). Each entry below points at one FEDERATION guidance
# item of the level (part, its first source id, the start of its text; tests check every pointer against
# profiles.json) that states what the check looks for at that level; a check with none here is offered as PitchIQ's
# own focus, after the level's general learning objectives (GENERAL_OBJECTIVES), which no check measures.
LEVEL_FOCUS_LABEL = "Suggested focus for this level - from the level's guidance, not from this match"
LEVEL_FOCUS_OWN = "This focus is PitchIQ's own proposal for this level; no source is quoted for it."
FOCUS_GUIDANCE = {
    2: {'pressure_and_cover': (('principles', 'EF-SWIFT-SUPPORT', 'FA 7-11 sessions: one player presses'),)},
    3: {'pressure_and_cover': (('principles', 'EF-COVER-RECOVER', 'Covering = taking up the right position'),),
        'spread_when_attacking': (('principles', 'USS-GM-U1112', 'Build an attacking shape with height, width'),)},
    4: {'pressure_and_cover': (('principles', 'EF-PROG-PRESS-COVER', 'Avoid two defenders pressing at once'),),
        'far_side_protection': (('principles', 'EF-COVER-RECOVER-SESSION', 'Protect the middle first'),),
        'back_line_coordination': (('principles', 'EF-COVER-RECOVER-SESSION', 'Keep the right spacing'),)},
    5: {'pressure_and_cover': (('principles', 'EF-PRINC', "England's defending principles"),),
        # (not "Shape: protect the middle first; ...; stay compact centrally; ...": EVAL's generic-sentence rule reads
        # "stay compact centrally" as the spec's generic "stay compact", 8 level-5 scene reports)
        'far_side_protection': (('principles', 'EF-PRINC', "England's defending principles"),),
        'spread_when_attacking': (('principles', 'USS-GM-U1516', 'Create an attacking shape with height, width'),)},
    6: {'pressure_and_cover': (('principles', 'EF-DLE-ROLES', 'Press, cover, balance roles'),),
        'far_side_protection': (('principles', 'EF-DLE-ROLES', 'Press, cover, balance roles'),)},
    7: {'someone_back_when_attacking': (('principles', 'UEFA-PI-1V1', 'A sound defensive shape during attacks'),)},
}
# The level's learning objectives (profiles.json 'objectives') a team can take as a focus: neither a coach-education
# programme nor a format the team may not play; tried in this order when no check-tied FEDERATION item is left.
GENERAL_OBJECTIVES = {
    1: (('objectives', 'USS-CG-U78', 'Every player experiences all positions'),),
    2: (('objectives', 'USS-CG-U910', 'Ball work: both feet'),),
    3: (('objectives', 'USS-CG-U1112', 'Players build an understanding of space and positioning'),
        ('objectives', 'USS-CG-U1112', 'Combination play in small groups')),
    4: (('objectives', 'USS-CG-U1314', 'Fundamental skills with and without the ball under high pressure'),),
    5: (('objectives', 'USS-AGG-U1516', 'U15-U16: keep playing players in different positions'),),
    6: (('objectives', 'USS-AGG-U17', 'U17+ (U.S. Soccer): develop tactical awareness'),),
    7: (),
}
# Pre-freeze (Oct 24, COACH F-9, critic S-3): each general learning objective is written for its source's ages, and a
# team older than the source's top age is never given it (it then gets the level's check as PitchIQ's own focus, or no
# objective). Before, level-1 Adult reports printed USS-CG-U78's "every player gets a turn in goal when a goalkeeper
# is used" (34 in the round-2 coach verifier's set), and over 17 ages x 7 levels 44 of the 96 profiles that reach a
# general objective got one written for younger players (DRILLS' fix-3 f9_general.py). None: no top age (U17+).
GENERAL_OBJECTIVE_TOP_AGE = {'USS-CG-U78': 'U8', 'USS-CG-U910': 'U10', 'USS-CG-U1112': 'U12', 'USS-CG-U1314': 'U14',
                             'USS-AGG-U1516': 'U16', 'USS-AGG-U17': None}
MATCH_FORMAT_RE = re.compile(r'\b(4|5|7|9|11)v\1\b')
FORMATION_RE = re.compile(r'\b1(?:-\d){2,5}\b')
FOCUS_SOURCE_RE = re.compile(r'\s*Source: ([A-Z0-9][A-Za-z0-9-]*(?:, [A-Z0-9][A-Za-z0-9-]*)*)\.$')   # pdf.py's


def _relevant_phase(detector):
    return S.DETECTOR_PHASES[detector][0]


# ---------- the base-rate test (fix round 3, LEAD decisions (2) and (3), critic N6) ----------
def binom_tail(k, n, p):
    """P(X >= k) for X ~ Binomial(n, p): the one-sided tail of the test both directions use (a 'moments worth a
    look' review's clear failing spells and a withheld strength's failures against the level's base share)."""
    k, n = int(k), int(n)
    if k <= 0:
        return 1.0
    if k > n:
        return 0.0
    p = float(p)
    if p <= 0.0:
        return 0.0
    if p >= 1.0:
        return 1.0
    lp, lq = math.log(p), math.log1p(-p)
    lc = math.lgamma(n+1)
    terms = [lc-math.lgamma(i+1)-math.lgamma(n-i+1)+i*lp+(n-i)*lq for i in range(k, n+1)]
    top = max(terms)
    return min(1.0, math.exp(top)*sum(math.exp(x-top) for x in terms))


def above_base(k, n, base, alpha):
    """Are k failures in n decided opportunities significantly above the base share (one-sided binomial tail at or
    below alpha)? None when there is no base."""
    if base is None:
        return None
    return binom_tail(k, n, base) <= float(alpha)+1e-12


def praise_base(params):
    """The base share a withheld strength is tested against (LEAD decision (3)): the level's professional
    baseFailShare, scaled for levels 1-4 by baseSuccessScale (k: the team is expected to show the behaviour k times
    as often as the professionals, base 1 - (1 - base) x k); None without a base."""
    base = params.get('baseFailShare')
    if base is None:
        return None
    k = float(params.get('baseSuccessScale') if params.get('baseSuccessScale') is not None else 1.0)
    return 1.0-(1.0-float(base))*k


# ---------- settled play (fix round 3, LEAD decision (2), FACTS F-3) ----------
def settle_index(timeline):
    """(the unnamed stretches as sorted (t0, t1) intervals, the ends of the dead-ball spans, sorted) of a timeline's
    phaseSpans: every gap between two consecutive named spans longer than 1.5 steps (the engine's change windows and
    the runs it could not name), and a direct switch between an 'own' and an 'opp' span (coach-marked) as a point."""
    step = float(timeline.get('step') or .2)
    spans = sorted((float(s['t0']), float(s['t1']), s['state']) for s in timeline.get('phaseSpans') or ()
                   if s.get('state') in ('own', 'opp', 'dead'))
    gaps, dead_ends = [], []
    for (a0, a1, sa), (b0, b1, sb) in zip(spans, spans[1:]):
        if b0-a1 > 1.5*step+1e-9:
            gaps.append((a1, b0))
        elif {sa, sb} == {'own', 'opp'}:
            mid = (a1+b0)/2.0
            gaps.append((mid, mid))
    gaps.sort()
    dead = [(a0, a1) for a0, a1, st in spans if st == 'dead']
    dead_ends = sorted(a1 for _, a1 in dead)
    return {'gaps': gaps, 'starts': [g[0] for g in gaps], 'deadEnds': dead_ends, 'dead': dead,
            'deadStarts': [a0 for a0, _ in dead]}


def settled(index, t, settle_s):
    """Is the time t in settled play: at least settle_s from every unnamed stretch (settle_index) and at least
    settle_s after the end of the last dead ball (a restart) before it?"""
    if index is None or settle_s <= 0:
        return True
    gaps, starts = index['gaps'], index['starts']
    k = bisect.bisect_right(starts, t+1e-9)-1
    for j in (k, k+1):
        if 0 <= j < len(gaps):
            a, b = gaps[j]
            d = a-t if t < a else (t-b if t > b else 0.0)
            if d < settle_s-1e-9:
                return False
    ends = index['deadEnds']
    j = bisect.bisect_right(ends, t+1e-9)-1
    if j >= 0 and t-ends[j] < settle_s-1e-9:
        return False
    j = bisect.bisect_right(index['deadStarts'], t+1e-9)-1
    if j >= 0 and t <= index['dead'][j][1]+1e-9:
        return False                # inside a dead ball
    return True


def _range(obs, t0, t1):
    seg = obs['segment']
    lo = float(seg['t0']) if t0 is None else float(t0)
    hi = float(seg['t1']) if t1 is None else float(t1)
    return min(lo, hi), max(lo, hi)


def _phase_seconds(timeline, state, lo, hi):
    """Seconds of a phase state inside [lo, hi] from the timeline's phase spans (each sample stands for one step)."""
    step = float(timeline.get('step') or .2)
    total = 0.0
    if state == 'unknown':
        named = sum(_phase_seconds(timeline, s, lo, hi) for s in ('own', 'opp', 'dead'))
        seg = max(0.0, min(hi, timeline['t1'])-max(lo, timeline['t0'])+step)
        return max(0.0, seg-named)
    for sp in timeline.get('phaseSpans') or ():
        if sp['state'] != state:
            continue
        a, b = max(float(sp['t0']), lo), min(float(sp['t1']), hi)
        if b >= a-1e-9:
            total += b-a+step
    return total


def _metres(scale):
    src = scale.get('source')
    if not scale.get('metresUsable'):
        return 'unusable'
    return {'measured': 'measured', 'declared': 'declared'}.get(src, 'estimated')


def _worst(levels, rank):
    levels = [x for x in levels if x in rank]
    return max(levels, key=lambda x: rank[x]) if levels else None


def _modal(values):
    c = Counter(v for v in values if v not in (None, ''))
    return c.most_common(1)[0][0] if c else None


def _first_code(effect):
    """The first style-effect code that is not the unknown-style rule (schema Code has no '+')."""
    for code in str(effect or '').split('+'):
        if code and code != UNKNOWN_STYLE:
            return code
    return None


# ---------- one opportunity ----------
def classify(opp, params, scale, step, as_detector=None):
    """The opportunity under the level's parameters: None when no sample is in scope, else {'outcome'
    'demonstrated' | 'not_demonstrated' | 'unclear', 'samples' (in scope), 'demoShare', 'failShare', 'failRun' (s),
    'mode' (modal failure mode), 'outs', 'modes'}. as_detector: judge with another detector's rule (the level-1
    reframing of D3 as behind_ball)."""
    det = as_detector or opp['detector']
    outs, modes = CX.sample_outcomes(opp, params, scale, as_detector=as_detector)
    if det == 'attacking_shape' and params.get('judgeDepth') is False:
        # Depth is described, never judged (fix round 2, LEAD decision (c), the coach's recommendation: 11v11 at
        # levels 1-3): a sample failing on depth alone is unclear, one failing on width and depth fails on width.
        # A demonstrated sample still needs the depth to have held, so nothing is praised that was not measured.
        for k, m in enumerate(modes):
            if m == 'depth':
                outs[k], modes[k] = 0, None
            elif m == 'shape':
                modes[k] = 'width'
    scope = [k for k, o in enumerate(outs) if o is not None]
    if not scope:
        return None
    t = [opp['relations']['t'][k] for k in scope]
    o = [outs[k] for k in scope]
    n = len(scope)
    demo, fail = o.count(1)/n, o.count(-1)/n
    runs = SI.spells(t, [x == -1 for x in o], min_s=0.0, merge_s=SI.SPELL_MERGE_S, step=step)
    fail_run = max((r['seconds'] for r in runs), default=0.0)
    druns = SI.spells(t, [x == 1 for x in o], min_s=0.0, merge_s=SI.SPELL_MERGE_S, step=step)
    demo_run = max((r['seconds'] for r in druns), default=0.0)
    share = float(params.get('outcomeShare', .6))
    demo_need = max(share, float(params.get('strengthTimeShare') or 0.0)) \
        if det == 'defensive_spacing' else share
    spell_min = float(params.get('spellMinS', SI.SPELL_MIN_S))
    if det == 'weak_side_balance':
        fail_min = float(params.get('failMinS', params.get('spellMinS', SI.SPELL_MIN_S)))
    else:
        fail_min = spell_min
    # A demonstration needs a run of the behaviour as long as a failure needs (fix round 1, critic B-2 / FACTS H1:
    # BC180 A's level-4 "3 of 3 attacking moments, 3 separate spells" rested on 0.2, 2.0 and 0.8 s of judged samples).
    if demo >= demo_need-1e-9 and demo_run >= spell_min-1e-9:
        outcome = 'demonstrated'
    elif fail >= share-1e-9 and fail_run >= fail_min-1e-9:
        outcome = 'not_demonstrated'
    else:
        outcome = 'unclear'
    # Counterexample "a formation deliberately changes": no role-based criticism within +-15 s of a change; the
    # level-1 reframed advance rests on the same role reading, so it is suppressed too (critic B-1).
    suppressed = False
    if outcome == 'not_demonstrated' and opp['detector'] == 'wide_defender_advance' and \
            opp['intentSignals']['formationChangeNear']:
        outcome, suppressed = 'unclear', True
    mode = Counter(m for m in modes if m).most_common(1)
    return {'outcome': outcome, 'samples': n, 'demoShare': demo, 'failShare': fail, 'failRun': fail_run,
            'failMinS': fail_min,
            'demoRun': demo_run, 'judgedSeconds': round(n*float(step), 2), 'suppressed': suppressed,
            'mode': mode[0][0] if mode else None, 'outs': outs, 'modes': modes}


def _family(opp, detector=None):
    if detector == 'behind_ball' and opp['detector'] == 'wide_defender_advance':
        return 'advance'                    # level 1: the advance reframed as "someone ready to protect our goal"
    if opp['detector'] == 'wide_defender_advance':
        return opp.get('variant') or 'team'
    if opp['detector'] == 'attacking_shape' and opp.get('variant') == 'unknown':
        return 'unknown'
    return 'team'


# ---------- guardrails ----------
def _moment_named(moment, opp):
    k = moment.get('sample')
    named = opp['relations'].get('primaryNamed')
    if named is None or k is None or not 0 <= k < len(named):
        return True
    return named[k] is not False


def _role_guard(detector, cited, index, fmt_players, limitations):
    if detector == 'wide_defender_advance' or (detector == 'defensive_spacing' and fmt_players != 11):
        ok = True
        for o in cited:
            prim = (o['actors']['primary'] or [None])[0]
            ref = o['roles'].get(prim) if prim else None
            if ref is None:
                ok = False
                continue
            relative_ok = ref['source'] == 'relative' and ref['confidence'] in ('medium', 'high')
            if not (ref['source'] in CONFIDENT or relative_ok):
                ok = False
            if index is not None and ref['source'] != 'coach' and index.mismatch_at((o['t0']+o['t1'])/2.0):
                ok = False
                if 'formation_mismatch' not in limitations:
                    limitations.append('formation_mismatch')
        return 'pass' if ok else 'fail'
    return 'not_applicable'


def _guardrails(detector, verdict, cited, moments, params, index, fmt_players, meets, style_explains, limitations):
    g = {k: 'not_applicable' for k in S.GUARDRAILS}
    # Team-level checks name nobody. A wide-defender check worded for the team still rests on WHO advanced: a crossing
    # at the evidence moment can make the measured advance another player's (counterexample "identity crossing").
    team_level = (detector in TEAM_LEVEL or bool(params.get('teamLevelOnly'))) and detector != 'wide_defender_advance'
    # ... also at level 1, where the advance is judged with the behind_ball rule (critic B-6 / CODE H1): any cited
    # opportunity found by the wide-defender detector rests on who advanced.
    team_level = team_level and not any(o['detector'] == 'wide_defender_advance' for o in cited)
    if not team_level:
        ok = all(o['quality']['identity'] in ('high', 'medium') for o in cited) and \
            all(_moment_named(m, o) for m, o in moments)
        g['identity'] = 'pass' if ok else 'fail'
        if not ok:
            limitations.append('identity_uncertain')
    g['position'] = 'pass' if all(o['quality']['position'] in ('high', 'medium') for o in cited) else 'fail'
    g['role'] = _role_guard(detector, cited, index, fmt_players, limitations)
    if detector in TEAM_LEVEL:
        g['context'] = 'pass' if all(o['quality']['visibility'] == 'complete' for o in cited) else 'fail'
    else:
        g['context'] = 'pass' if all(o['quality']['visibility'] != 'none' for o in cited) else 'fail'
    want = _relevant_phase(detector)
    g['phase'] = 'pass' if all(o['phase']['state'] == want and not o['phase']['nearChange'] for o in cited) else 'fail'
    if verdict in S.CRITICISM+('review',):
        intent = style_explains or any(o['intentSignals']['recovering'] or o['intentSignals']['interchange']
                                       or o['intentSignals']['formationChangeNear'] for o in cited)
        g['intent'] = 'fail' if intent else 'pass'
        g['alternative'] = 'pass'
        g['evidence'] = 'pass' if meets else 'fail'
    elif verdict == 'strength':
        g['evidence'] = 'pass'
    if style_explains:
        g['intent'] = 'fail'            # "this looked intentional because ..." (DESIGN 4.8: a describe)
    return g


# ---------- measurements explain.py reads (explain.MEASUREMENT_KEYS) ----------
def _measurements(detector, opp, k, family_items, params, scale, verdict, mode):
    R = opp['relations']
    out = {}

    def put(name, key, digits=None):
        v = R.get(key, [None]*(k+1))[k] if key in R else None
        if v is not None:
            out[name] = round(v, digits) if digits is not None and isinstance(v, float) else v
    if detector == 'pressure_cover':
        pressure, cover = CX.pressure_cover_flags(R, k, params, scale)
        out['pressurePresent'], out['coverPresent'] = pressure, cover
        if mode:
            out['failureMode'] = mode
        put('d1M', 'd1', 1)
        put('coverDistM', 'coverDist', 1)
        put('coverAngleDeg', 'coverAngle', 0)
        put('goalSideShare', 'goalSideShare', 2)
        put('coverId', 'coverId')
        put('fdId', 'fd')               # the presser AT the moment (critic B-3), never the family's modal one
        if out.get('coverId') == out.get('fdId'):
            out.pop('coverId', None)    # (cannot happen: the cover search excludes the first defender) never both
        pressed = covered = 0
        for o, c in family_items:
            for j, x in enumerate(c['outs']):
                if x is None:
                    continue
                p_, c_ = CX.pressure_cover_flags(o['relations'], j, params, scale)
                if p_:
                    pressed += 1
                    covered += int(c_)
        if pressed:
            out['coverPresentShare'] = round(covered/pressed, 3)
    elif detector == 'weak_side_balance':
        # The counts the verdict used: goal-side with the level tolerance where measured (critic B-5), so "no defender
        # in the central corridor" is never said with a centre-back level with the ball.
        if R.get('weakGoalCount') is not None and R['weakGoalCount'][k] is not None:
            put('weakCount', 'weakGoalCount')
            put('centralCount', 'centralGoalCount')
            put('weakLevelCount', 'weakLevelCount')
            put('weakRecovering', 'weakRecovering')
        else:
            put('weakCount', 'weakCount')
            put('centralCount', 'centralCount')
        put('openWidthM', 'openWidthM', 1)
        put('ballX', 'ballX', 1)
    elif detector == 'wide_defender_advance':
        if opp.get('variant') in ('left', 'right'):
            out['side'] = opp['variant']
        for name, key, dg in (('advanceM', 'advanceM', 1), ('behindN', 'behindN', None),
                              ('channelOccupied', 'channelOccupied', None), ('channelBy', 'channelBy', None),
                              ('cbAdjusted', 'cbAdjusted', None), ('nearCbShiftM', 'nearCbShiftM', 1),
                              ('interchange', 'interchange', None), ('opponentsInChannel', 'opponentsInChannel', None),
                              ('threatN', 'threatN', None), ('restMargin', 'restMargin', None),
                              ('dmCbSpacingM', 'dmCbSpacingM', 1)):
            put(name, key, dg)
        out['dmDropped'] = R['dmDropped'][k]          # None = not tested (no holding midfielder role)
        if mode:
            out['failureMode'] = mode   # 'exposed' / 'uncovered' / 'nobody_behind' (level 2) (fix round 3)
    elif detector == 'behind_ball':
        put('behindN', 'behindN')
        put('threatN', 'threatN')
        put('restMargin', 'restMargin')
        if params.get('minBehind') is not None:
            out['minBehind'] = int(params['minBehind'])
        if opp['detector'] == 'wide_defender_advance':
            # level 1's reframed advance (fix round 3, critic B-1): the advance's own relations, so the text can say
            # "one of theirs was in the space we left" ('exposed') or "nobody stayed back" ('nobody_behind')
            if opp.get('variant') in ('left', 'right'):
                out['side'] = opp['variant']
            put('advanceM', 'advanceM', 1)
            put('channelOccupied', 'channelOccupied')
            put('opponentsInChannel', 'opponentsInChannel')
            if mode:
                out['failureMode'] = mode
    elif detector == 'defensive_spacing':
        if mode:
            out['failureMode'] = mode
        for name, key, dg in (('lateralGapM', 'lateralGapM', 1), ('gapBallSide', 'gapBallSide', None),
                              ('opponentInGap', 'opponentInGap', None), ('dmGapM', 'dmGapM', 1),
                              ('lineStdM', 'lineStdM', 1), ('blockHeight', 'blockHeight', None),
                              ('dmGapOpponents', 'dmGapOpponents', None)):
            put(name, key, dg)
    elif detector == 'attacking_shape':
        if mode:
            out['failureMode'] = mode
        for name, key, dg in (('widthShare', 'widthShare', 2), ('depthShare', 'depthShare', 2), ('widthM', 'widthM', 1),
                              ('depthM', 'depthM', 1), ('channelsOccupied', 'channelsOccupied', None),
                              ('depthAvailShare', 'depthAvailShare', 2), ('availableDepthM', 'availableDepthM', 1),
                              ('heightGapM', 'heightGapM', 1), ('deepN', 'deepN', None)):
            put(name, key, dg)
    out['metres'] = _metres(scale)
    return out


def _moment_actors(detector, first, k, meas, scale):
    """Fix round 3 (FACTS F-2 / F-5, critic B-3; for TEXT): the first cited moment's own players and place, never the
    family's modal ones (the D1 presser has been the moment's since fix round 1). Sets in meas: momentZone (the ball's
    third at that sample where its depth is measured: D1 the carrier's, D2-D4 the ball's; else the moment's
    opportunity's zone), for D3 actorId (the advancing wide defender of that opportunity) and interchangeCover (the
    slot re-occupied at that sample), for D5 gapLeftId / gapRightId (the gap pair at that sample), actorId (the pair's
    member the opportunity names, else the left one) and gapOppFreeM. Returns (primary, partner or None) for D3 / D5,
    else None. Measured on Metrica game 1 (FACTS round 2): the family's player was not the moment's in 2 of 176 D3
    texts naming players and 1 of 1 D5 Metrica pair; 25 distinct texts placed the moment in the family's modal third
    more than 3 m away from its own."""
    if first is None:
        return None
    m, o = first
    R = o['relations']

    def at(key):
        v = R.get(key)
        return v[k] if v is not None and 0 <= k < len(v) else None
    L = float(scale['lengthM'])
    if detector == 'pressure_cover' and at('carrierZone'):
        zone = at('carrierZone')
    elif detector in ('weak_side_balance', 'wide_defender_advance', 'behind_ball') and at('ballX') is not None:
        zone = G.zone_of(at('ballX'), L)
    else:
        zone = (o.get('ball') or {}).get('zone')
    if zone:
        meas['momentZone'] = zone
    if detector == 'wide_defender_advance':
        actor = (o['actors']['primary'] or [None])[0]
        if at('interchangeCover') is not None:
            meas['interchangeCover'] = bool(at('interchangeCover'))
        if CX.valid_id(actor):
            meas['actorId'] = actor
            return actor, None
        return None
    if detector == 'defensive_spacing':
        gl, gr = at('gapLeft'), at('gapRight')
        if at('gapOppFreeM') is not None:
            meas['gapOppFreeM'] = round(float(at('gapOppFreeM')), 1)
        if not (CX.valid_id(gl) and CX.valid_id(gr)):
            return None
        meas['gapLeftId'], meas['gapRightId'] = gl, gr
        named = (o['actors']['primary'] or [None])[0]
        prim, other = (named, gr if named == gl else gl) if named in (gl, gr) else (gl, gr)
        meas['actorId'] = prim
        return prim, other
    return None


def _central_kept(o, c):
    """Did a failing far-side opportunity keep a goal-side defender in the central corridor on at least the
    outcomeShare of its failing samples (centralGoalCount, else the plain centralCount)?"""
    R = o['relations']
    key = 'centralGoalCount' if R.get('centralGoalCount') is not None else 'centralCount'
    vals = R.get(key) or []
    fail = [k for k, x in enumerate(c['outs']) if x == -1 and k < len(vals)]
    if not fail:
        return False
    kept = sum(1 for k in fail if (vals[k] or 0) >= 1)
    return kept >= .6*len(fail)-1e-9


def _clear_moments(detector, o, c, moments_of, index, players, step, params, settle=None):
    """The evidence moments of a not-demonstrated opportunity that show a CLEAR failure (fix round 2, LEAD decision
    (b)), [] when it is not one: a run of consecutive failing samples (no unclear, demonstrated or out-of-scope sample
    inside it, so D1's three-state call said 'pressed' or 'not pressed' throughout) lasting the level's failure spell;
    the identity high (no crossing taint) and the position medium or high; the whole team in view for a team-level
    check (else the relationship's local gate in view); the phase named by the possession engine or the coach's
    marks, the relevant one and away from a change; no recovering run, interchange or formation change; the role
    reading confident (D3, D5 on small formats); and the moment inside that run with its player named.
    A D1 'nobody pressed' sample is clear only while the gap from the carrier to our nearest player is not shrinking
    at pressureClosingMS (1 m/s) or more (its rate from the neighbouring samples, both measured): pressure arriving
    is not a carrier left alone. On Metrica game 1 (full tracking, 15-minute clips) the 6 level-4 'nobody pressed'
    moments of the rule's first version (2 team-clips, levels 3-7) all sat in gaps shrinking from 7.8-14.6 m to
    1.5-5.0 m within the 2-3 s of their opportunity; the defender's own closing speed (closingMS) did not tell them
    apart (-3.5 to 5.0 m/s: a carrier dribbling at a defender who holds closes the gap too).
    Fix round 3 (LEAD decision (2), FACTS F-3, critic N1): a failing sample is clear only in settled play, at least
    momentsSettleS (2 s) from any stretch whose team was not named (an engine change window or a run it could not
    name, a coach-marked change; settle_index on the timeline's phaseSpans) and after the end of a dead ball (a
    restart), and at least momentsContestS (1 s) from a ball the possession engine calls loose or contested (contestS:
    the ball between the teams); for D5 on 11v11 with a recognised line of at least lineMinPlayers (4) whose depth
    spread lineStdM is at most lineStdMaxM (4 m); and a lateral-gap failure only while the opponent in the gap is
    free: nobody of ours goal-side of him within markedM (4 m; gapOppFreeM). On Metrica game 1 the 6 disputed cited
    moments of round 2 each fail one of these (engine records at the moment): 4603.47 s 1.6 s after an engine run
    with no team named, 1153.27 s 1.4 s after a throw-in's dead ball, 369.67 s (truth phase) 1.2 s after one, 3217.07 s
    0.4-0.8 s from contested balls, 4557.67 s inside a loose ball (and a 5-player 'line' with a depth std of 8.1 m),
    750.87 s (camera) a 3-player line. A relation an older file lacks never makes a sample unclear."""
    if c is None or c['outcome'] != 'not_demonstrated':
        return []
    q = o['quality']
    if q['identity'] != 'high' or q['position'] not in ('high', 'medium'):
        return []
    in_view = q['visibility'] == 'complete' if detector in TEAM_LEVEL else q['visibility'] != 'none'
    if not in_view:
        return []
    ph = o['phase']
    if ph['state'] != _relevant_phase(detector) or ph['nearChange'] or ph['source'] not in ('possession-4', 'coach'):
        return []
    sig = o['intentSignals']
    if sig['recovering'] or sig['interchange'] or sig['formationChangeNear']:
        return []
    if _role_guard(detector, [o], index, players, []) == 'fail':
        return []
    fail = [x == -1 for x in c['outs']]
    if detector == 'pressure_cover':
        need = float(params.get('pressureClosingMS') or CX.REFERENCE_PARAMS['pressure_cover']['pressureClosingMS'])
        t, d1 = o['relations']['t'], o['relations']['d1']

        def left_alone(k):
            a, b = max(0, k-1), min(len(t)-1, k+1)
            if b == a or d1[a] is None or d1[b] is None or t[b]-t[a] > 2.5*step:
                return False
            return (d1[b]-d1[a])/(t[b]-t[a]) > -need+1e-9
        fail = [f and (m != 'no_pressure' or left_alone(k)) for k, (f, m) in enumerate(zip(fail, c['modes']))]
    fail = [f and _settled_sample(detector, o, k, c['modes'][k], players, params, settle)
            for k, f in enumerate(fail)]
    runs = SI.spells(o['relations']['t'], fail, min_s=c['failMinS'], merge_s=0.0, step=step)
    return [m for m in moments_of.get(o['id'], ()) if isinstance(m.get('sample'), int) and _moment_named(m, o) and
            any(r['i0'] <= m['sample'] <= r['i1'] for r in runs)]


def _settled_sample(detector, o, k, mode, players, params, settle):
    """Fix round 3 (_clear_moments): may the failing sample k of the opportunity count toward a clear failure?"""
    R = o['relations']
    settle_s = float(params.get('momentsSettleS') or 0.0)
    if settle is not None and not settled(settle, float(R['t'][k]), settle_s):
        return False
    contest_s = float(params.get('momentsContestS') or 0.0)
    cs_ = R['contestS'][k] if 'contestS' in R else None
    if contest_s > 0 and cs_ is not None and cs_ < contest_s-1e-9:
        return False
    if detector == 'defensive_spacing':
        if players == 11:
            size = R['lineSize'][k] if 'lineSize' in R else None
            std = R['lineStdM'][k] if 'lineStdM' in R else None
            need = params.get('lineMinPlayers')
            if need is not None and size is not None and size < int(need):
                return False
            top = params.get('lineStdMaxM')
            if top is not None and std is not None and std > float(top)+1e-9:
                return False
        marked = params.get('markedM')
        free = R['gapOppFreeM'][k] if 'gapOppFreeM' in R else None
        if mode == 'lateral_gap' and marked is not None and free is not None and free <= float(marked)+1e-9:
            return False
    return True


def _moments_cap(profile):
    """The level's momentsMaxPerReport (profiles.json, shared by every enabled check), 0 when unset."""
    for dp in profile['detectors'].values():
        if dp['enabled'] and dp['params'].get('momentsMaxPerReport') is not None:
            return int(dp['params']['momentsMaxPerReport'])
    return 0


def _cap_moments(findings, profile):
    """At most momentsMaxPerReport 'moments worth a look' reviews in one report (fix round 2): the highest scored
    (the level's relevance weight x clear spells x clear share of the decided opportunities x identity quality, then
    the level's objective rank) stay; each other one is replaced by the finding its family gives without the review
    (a description, a mixed-evidence note, or nothing)."""
    mw = sorted((f for f in findings if f.get('_moments') is not None),
                key=lambda f: (-f['_moments'][0], f['_moments'][1], f['key']))
    drop = {f['id'] for f in mw[_moments_cap(profile):]}
    out = []
    for f in findings:
        if f['id'] in drop:
            f = f.get('_fallback')
            if f is None:
                continue
        out.append(f)
    for f in out:
        f.pop('_moments', None)
        f.pop('_fallback', None)
    return out


# ---------- one family -> one finding ----------
def _pick_moments(cited, others, moments_of, items_rank, fallback=True):
    """Up to MAX_TIMESTAMPS moments of the cited opportunities (first by items_rank, then spread >= DISTINCT_S
    apart), falling back to the family's other opportunities only when `fallback` (never for a criticism or a review:
    CODE L12, a criticism must not cite a demonstrated moment); two moments at the same instant (a left and a right
    advance) count once (STATS 11); returned chronologically as (moment, opportunity) pairs."""
    pool = []
    for o in sorted(cited, key=items_rank):
        pool += [(m, o) for m in moments_of.get(o['id'], ())]
    if not pool and fallback:
        for o in others:
            pool += [(m, o) for m in moments_of.get(o['id'], ())]
    chosen = []
    for m, o in pool:
        if len(chosen) >= MAX_TIMESTAMPS:
            break
        if all(abs(m['t']-c[0]['t']) >= SI.DISTINCT_S-1e-9 for c in chosen):
            chosen.append((m, o))
    for m, o in pool:
        if len(chosen) >= MAX_TIMESTAMPS:
            break
        if all(m['id'] != c[0]['id'] and abs(m['t']-c[0]['t']) > 1e-6 for c in chosen):
            chosen.append((m, o))
    return sorted(chosen, key=lambda mo: mo[0]['t'])


_LADDER = {}


def _ladder(profile):
    """[(level, {detector: DetectorProfile})] of levels 1-7 at their typical age, style unknown, in this profile's
    match format (the profile-independent yardstick the moments are ranked by, S-5); cached per profiles sha and
    format."""
    from . import profiles as PR
    fmt = profile['format']
    key = (profile.get('sha'), int(fmt['players']), bool(fmt['goalkeeper']))
    hit = _LADDER.get(key)
    if hit is None:
        hit = []
        for lv in range(1, 8):
            block = {'playingLevel': lv, 'ageGroup': PR.default_age(lv), 'style': 'unknown',
                     'format': {'players': int(fmt['players']), 'goalkeeper': bool(fmt['goalkeeper'])}}
            hit.append((lv, PR.resolve(block)['detectors']))
        if len(_LADDER) > 32:
            _LADDER.clear()
        _LADDER[key] = hit
    return hit


def _level_counts(o, detector, ladder, scale, step):
    """(levels at which the opportunity is not demonstrated, levels at which it is demonstrated, the summed fail and
    demonstrated shares) over the ladder, for `detector`'s rule (the level-1 reframing where it applies)."""
    fails = demos = 0
    fsum = dsum = 0.0
    for _, dets in ladder:
        dp = dets[detector]
        if not dp['enabled']:
            continue
        if o['detector'] != detector:
            wide = dets['wide_defender_advance']
            if wide['enabled'] or wide['params'].get('reframedAs') != detector:
                continue
        c = classify(o, dp['params'], scale, step, as_detector=detector if o['detector'] != detector else None)
        if c is None:
            continue
        fails += c['outcome'] == 'not_demonstrated'
        demos += c['outcome'] == 'demonstrated'
        fsum += c['failShare']
        dsum += c['demoShare']
    return fails, demos, fsum, dsum


def _finding(detector, family, items, ctx, allow_moments=True):
    """The finding of one family (list of (opportunity, classification)) or None. allow_moments: a family with no
    verdict, review or strength may become a 'moments worth a look' review (the finding it gives without one is kept
    in '_fallback' for the report's cap)."""
    profile, team = ctx['profile'], ctx['team']
    dp = profile['detectors'][detector]
    p = dp['params']
    verdicts = dp['verdicts']
    ev = profile['evidence']
    n = len(items)
    dem = [o for o, c in items if c['outcome'] == 'demonstrated']
    nd = [o for o, c in items if c['outcome'] == 'not_demonstrated']
    unc = n-len(dem)-len(nd)
    distinct_s = float(p.get('distinctS', SI.DISTINCT_S))
    dfs = len(SI.distinct([{'t0': o['t0']} for o in nd], apart=distinct_s))
    dds = len(SI.distinct([{'t0': o['t0']} for o in dem], apart=distinct_s))
    state = 'unknown' if family == 'unknown' else _relevant_phase(detector)
    phase_s = _phase_seconds(ctx['timeline'], state, ctx['lo'], ctx['hi'])
    elig_s = CX.eligible_seconds([o for o, _ in items], ctx['step'])
    fail_share = len(nd)/n
    decided = len(dem)+len(nd)
    # The strength share is over the DECIDED opportunities (critic B-2 / COACH M4: unclear ones are evidence of
    # neither; the progression's mastery uses demonstrated / (demonstrated + not demonstrated) too), at least
    # strengthSpells of them.
    demo_share = len(dem)/decided if decided else 0.0
    min_fail = ev['minFailureShare'] if ev['minFailureShare'] is not None else float(p.get('minFailureShare', .3))
    short = (ctx['hi']-ctx['lo']) < float(p.get('shortRangeS', SI.SHORT_RANGE_S))
    need_spells = max(int(ev['minFailingSpells']), int(p.get('shortRangeMinFailingSpells', 3)) if short else 0)
    meets = (n >= ev['minOpportunities'] and dfs >= need_spells and fail_share >= min_fail-1e-9
             and phase_s >= ev['minPhaseSeconds']-1e-9)
    failing = len(nd) >= 1 and fail_share >= min_fail-1e-9
    crit = 'work_on' if 'work_on' in verdicts else ('try_this' if 'try_this' in verdicts else None)
    strength_ok = ('strength' in verdicts and dds >= ev['strengthSpells'] and decided >= ev['strengthSpells'] and
                   demo_share >= ev['strengthShare']-1e-9)
    # No bare strength beside failures significantly above the level's base (critic B-2 / STATS 7: a family with
    # planted 5 s "nobody behind the ball" spells read "what went well: someone stayed back"): baseFailShare is the
    # pooled not-demonstrated share of the decided opportunities on Metrica game 1 (full tracking) at the level's
    # rules; the family is then described with the note 'mixed_evidence', never praised. Fix round 3 (LEAD decision
    # (3), COACH F-5, critic N6): withheld only when the failures are significantly above that base by the one-sided
    # binomial tail at baseTestAlpha (.05), the base scaled below level 5 (praise_base, baseSuccessScale .5 / .6 / .7
    # / .8 at levels 1-4): before, one lapse in 10-11 (D2 base .004, D5 .048) withheld praise at every level from U14.
    alpha = float(p.get('baseTestAlpha') if p.get('baseTestAlpha') is not None else .05)
    base = p.get('baseFailShare')
    mixed = bool(nd) and bool(above_base(len(nd), decided, praise_base(p), alpha))
    # The compact style accepts space on the far side, never an empty middle (critic S-8 / COACH M2): it explains a
    # far-side gap only where every failing opportunity kept a goal-side defender in the central corridor.
    style_explains = bool(failing and detector == 'weak_side_balance' and p.get('acceptFarSideGap') and
                          all(_central_kept(o, ctx['by_id'][o['id']]) for o in nd))
    describe_only = not any(v in verdicts for v in ('strength',)+S.CRITICISM+('review',))
    floor = float(p.get('reviewFloorShare') or 0.0)
    floor_ok = n >= floor*ev['minOpportunities']-1e-9 and phase_s >= floor*ev['minPhaseSeconds']-1e-9
    limitations = []
    effect = p.get('styleEffect')
    if UNKNOWN_STYLE in str(effect or ''):
        limitations.append('style_unknown')
    if short:
        limitations.append('short_range')
    clear_instance = any(o['quality']['identity'] != 'low' and o['quality']['position'] != 'low' for o in nd)
    if family == 'unknown':
        verdict, cited = ('describe' if 'describe' in verdicts else None), [o for o, _ in items]
    elif style_explains:
        verdict, cited = ('describe' if 'describe' in verdicts else None), nd
    elif failing and meets and crit:
        verdict, cited = crit, nd
    elif failing and 'review' in verdicts and (meets or clear_instance) and floor_ok:
        verdict, cited = 'review', nd
    elif failing and 'review' in verdicts and (meets or clear_instance):
        # Below the review floor (S-1): too little play in this range to judge the check; the reason says which
        # part fell short (fix round 3, critic S-2 / COACH F-14: 4 clear advances in 6.1 minutes of a clip with 226 s
        # of our possession read "the range is too short"): too few relevant seconds -> 'short_range', enough of
        # them but too few situations -> 'too_few_situations'.
        short_s = phase_s < floor*ev['minPhaseSeconds']-1e-9
        if ctx['too_little'].get(detector) != 'short_range':
            ctx['too_little'][detector] = 'short_range' if short_s else 'too_few_situations'
        verdict, cited = None, []
    elif strength_ok and mixed:
        verdict, cited = ('describe' if 'describe' in verdicts else None), [o for o, _ in items]
        limitations.append('mixed_evidence')
    elif strength_ok:
        verdict, cited = 'strength', dem
    elif describe_only and 'describe' in verdicts:
        verdict, cited = 'describe', [o for o, _ in items]
    else:
        verdict, cited = None, []
    # 'Moments worth a look' (fix round 2, LEAD decision (b)): the verdicts keep their strict majority rules, but
    # professional play fails some checks a third of the time (cover: .654 of the decided level-4 opportunities on
    # Metrica game 1 are not demonstrated; planted failures among 20-30 natural opportunities never reach the 40%
    # share). From momentsFromLevel (3; levels 1-2 stay positive-first) a family with no verdict, review or
    # strength becomes a review when at least momentsMinSpells (3) distinct failing spells are CLEAR (_clear_moments)
    # above the review floor; it cites only those moments and carries its counts, never a verdict or a drill.
    # Fix round 3 (LEAD decision (2), STATS 2, critic N6): the trigger is relative. The clear failing spells must
    # also exceed what the level's baseFailShare predicts for the decided opportunities, by the one-sided binomial
    # tail at baseTestAlpha (a check with no measured base gives none). The clear spells are a subset of the failing
    # opportunities, so the test is conservative; a family failing at or below its base is never reviewed (round 2:
    # 7 of the 22 whole-match professional reviews were). The wide-defender check counts its clear advances against
    # their own floor (momentsMinSpells), not the opportunity floor (critic S-2 / COACH F-14: each opportunity is a
    # whole advance; 4 clear ones in 6.1 minutes were 'too little play').
    mwl = None
    from_lv = p.get('momentsFromLevel')
    floor_mw = floor_ok or detector == 'wide_defender_advance'
    if (allow_moments and verdict in (None, 'describe') and family != 'unknown' and not style_explains and nd and
            'review' in verdicts and from_lv is not None and int(profile['level']) >= int(from_lv) and floor_mw):
        clear_m = {}
        for o, c in items:
            if c['outcome'] == 'not_demonstrated':
                ms = _clear_moments(detector, o, c, ctx['moments'], ctx['index'], ctx['players'], ctx['step'], p,
                                    settle=ctx.get('settle'))
                if ms:
                    clear_m[o['id']] = ms
        clear = [o for o in nd if o['id'] in clear_m]
        spells_c = len(SI.distinct([{'t0': o['t0']} for o in clear], apart=distinct_s))
        tail = binom_tail(spells_c, decided, base) if base is not None else None
        if spells_c >= int(p.get('momentsMinSpells', 3)) and tail is not None and tail <= alpha+1e-12:
            fallback = _finding(detector, family, items, ctx, allow_moments=False)
            mwl = {'moments': clear_m, 'spells': spells_c, 'clear': len(clear), 'fallback': fallback, 'tail': tail}
            verdict, cited = 'review', clear
            limitations = [x for x in limitations if x != 'mixed_evidence']+[MOMENTS]
    if verdict is None:
        return None
    by_id = {o['id']: c for o, c in items}
    if any(c['suppressed'] for _, c in items):
        limitations.append('formation_changing')
    # The same moments at every level (critic S-5): the pool is ranked by how many levels' rules (1-7, typical ages,
    # style unknown, this format) fail it (criticisms, reviews and describes where failures dominate) or demonstrate it
    # (strengths, other describes), then by the summed shares, then by time; only this level's cited set differs.
    critical = verdict in S.CRITICISM+('review',)
    ranks = {}
    if len(cited) > MAX_TIMESTAMPS:
        ladder = _ladder(profile)
        for o in cited:
            ranks[o['id']] = _level_counts(o, detector, ladder, ctx['scale'], ctx['step'])
    if verdict == 'describe' and ranks:
        critical = sum(r[0] for r in ranks.values()) >= sum(r[1] for r in ranks.values()) and \
            any(r[0] for r in ranks.values())

    def item_rank(o):
        r = ranks.get(o['id'])
        if r is None:
            c = by_id[o['id']]
            return (-(c['failShare'] if critical else c['demoShare']), o['t0'])
        return (-r[0], -r[2], o['t0']) if critical else (-r[1], -r[3], o['t0'])
    chosen = _pick_moments(cited, [o for o, _ in items], mwl['moments'] if mwl else ctx['moments'], item_rank,
                           fallback=verdict not in S.CRITICISM+('review',))
    g = _guardrails(detector, verdict, cited, chosen, p, ctx['index'], ctx['players'], meets, style_explains,
                    limitations)
    if verdict in S.CRITICISM:
        failed = [k for k in ('identity', 'position', 'role', 'context', 'phase', 'intent') if g[k] == 'fail']
        weak_role = (detector == 'wide_defender_advance' and verdict == 'work_on' and p.get('workOnNeedsHighConfidence')
                     and not all(o['quality']['identity'] == 'high' and
                                 (o['roles'].get((o['actors']['primary'] or [None])[0]) or {}).get('confidence')
                                 == 'high' for o in cited))
        if failed or weak_role:
            if 'review' not in verdicts:
                return None
            verdict = 'review'
    if verdict == 'review' and not meets and dfs <= 1 and 'one_instance' not in limitations:
        limitations.append('one_instance')
    if verdict != 'describe' and not chosen:
        # No evidence moment exists for this family (EVIDENCE made none): never a claim without a moment.
        if 'describe' not in verdicts:
            return None
        verdict = 'describe'
        limitations.append('no_moment')
        g = _guardrails(detector, verdict, cited, chosen, p, ctx['index'], ctx['players'], meets, style_explains,
                        limitations)
    all_opps = sorted((o for o, _ in items), key=lambda o: o['t0'])
    t0, t1 = all_opps[0]['t0'], max(o['t1'] for o in all_opps)
    key = S.finding_key(detector, team, family, t0, t1)
    first = chosen[0] if chosen else None
    if first is not None:
        m, o = first
        k = int(m['sample']) if 0 <= int(m.get('sample', 0)) < len(o['relations']['t']) else 0
    else:
        o = cited[0] if cited else all_opps[0]
        k = CX.key_sample(o, ctx['scale'])
    mode = _modal(by_id[x['id']]['mode'] for x in cited) if verdict in S.CRITICISM+('review',) else None
    meas = _measurements(detector, o, k, items, p, ctx['scale'], verdict, mode)
    # The judged seconds behind the cited opportunities (critic B-2: counts and spells are of judged samples).
    meas['judgedSeconds'] = round(sum(by_id[x['id']]['judgedSeconds'] for x in cited), 2)
    if mwl:
        # what a 'moments worth a look' review reports ("in 4 of 13 defending moments the first defender had no
        # cover"): the clear failing opportunities and their distinct spells, of the decided ones; all failing ones
        meas.update(clearMoments=mwl['clear'], clearSpells=mwl['spells'], decidedMoments=decided,
                    failingMoments=len(nd))
        # the test that let it through (fix round 3): the level's base share and the binomial tail of the clear
        # spells against it (at most baseTestAlpha)
        meas.update(baseShare=round(float(base), 3), baseTail=round(float(mwl['tail']), 4))
    elif 'mixed_evidence' in limitations:
        pb = praise_base(p)             # the strength withheld: the base it was tested against and the tail
        meas.update(baseShare=round(pb, 3), baseTail=round(binom_tail(len(nd), decided, pb), 4))
    # Findings resting on coach-marked phases say so (user decision 1): the phase source of all its opportunities.
    srcs = {x['phase']['source'] for x in all_opps} - {'none'}
    if srcs:
        meas['phaseSource'] = 'coach' if srcs == {'coach'} else ('mixed' if 'coach' in srcs else 'possession-4')
    prim = _modal((x['actors']['primary'] or [None])[0] for x in cited)
    if detector == 'pressure_cover' and CX.valid_id(meas.get('fdId')):
        prim = meas['fdId']             # the presser at the moment the text links (critic B-3)
    partner = _moment_actors(detector, first, k, meas, ctx['scale'])
    if partner is not None:
        prim = partner[0]               # the moment's own players, never the family's (fix round 3, FACTS F-2)
    sup = []
    for x in cited:
        sup += [s for s in x['actors']['supporting'] if s != prim]
    sup = [s for s, _ in Counter(sup).most_common(3)]
    if detector == 'pressure_cover' and CX.valid_id(meas.get('coverId')) and meas['coverId'] != prim:
        sup = [meas['coverId']]+[s for s in sup if s != meas['coverId']][:2]
    if partner is not None and partner[1] is not None:
        sup = [partner[1]]+[s for s in sup if s != partner[1]][:2]
    opp_ids = []
    for x in cited:
        opp_ids += x['actors']['opponents']
    opp_ids = [s for s, _ in Counter(opp_ids).most_common(3)]
    players = {'primary': [prim] if prim else [], 'supporting': sup, 'opponents': opp_ids}
    refs = {}
    for pid in players['primary']+players['supporting']:
        for x in cited:
            if pid in x['roles']:
                refs[pid] = dict(x['roles'][pid])
                break
    zones = _modal(x['ball']['zone'] for x in cited)
    chans = _modal(x['ball']['channel'] for x in cited)
    estimated = any(x['phase']['source'] == 'possession-4' for x in cited) if cited else True
    quality = [x['quality'] for x in cited] or [x['quality'] for x in all_opps]
    vis = _worst([q['visibility'] for q in quality], VISIBILITY_RANK) or 'none'
    balls = {q['ball'] for q in quality}
    ball_q = 'reliable' if 'reliable' in balls else ('unreliable' if 'unreliable' in balls else 'not_needed')
    share_phase = round(min(1.0, elig_s/phase_s), 3) if phase_s > 0 else None
    pq = ctx['timeline'].get('phaseQuality') or {}
    finding = {
        'id': S.record_id('fnd', key), 'key': key, 'observation': detector, 'verdict': verdict,
        'level': profile['level'], 'team': team, 'priority': None, 'players': players, 'roles': refs,
        'timestamps': [{'evidence': m['id'], 'opportunity': m['opportunity'], 't': m['t'], 'frame': m['frame'],
                        'seek': m['seek']} for m, _ in chosen],
        'opportunities': [o['id'] for o in all_opps],
        'context': {'phase': {'state': state, 'estimated': bool(estimated),
                              'coverage': pq.get('coverage') if isinstance(pq.get('coverage'), float) else None},
                    'ball': {'zone': zones, 'channel': chans} if (zones or chans) else None,
                    'style': profile['style'], 'styleEffect': _first_code(effect),
                    'format': profile['format']['name']},
        'measurements': meas,
        'aggregate': {'opportunities': n, 'demonstrated': len(dem), 'notDemonstrated': len(nd), 'unclear': unc,
                      'distinctFailingSpells': dfs, 'distinctDemonstratedSpells': dds,
                      'phaseSeconds': round(phase_s, 2),
                      'eligibleShare': share_phase,
                      'ratePer10': round(SI.rate_per_10min(dfs, phase_s), 2) if phase_s > 0 else None},
        'coverage': {'identity': _worst([q['identity'] for q in quality], LEVEL_RANK) or 'high',
                     'position': _worst([q['position'] for q in quality], LEVEL_RANK) or 'high',
                     'visibility': vis, 'ball': ball_q,
                     'calibrationTier': first[0].get('tier') if first is not None else None,
                     'eligibleShareOfPhase': share_phase},
        'guardrails': g, 'limitations': list(dict.fromkeys(limitations)),
        'suggestedAction': ({'principle': DEFAULT_PRINCIPLE_D6, 'mode': 'principle', 'alternatives': []}
                            if detector == 'attacking_shape' and verdict in S.CRITICISM+('review',) else None),
        'claims': [], 'text': None, 'drills': [],
    }
    weight = QUALITY_FACTOR[finding['coverage']['identity']]
    finding['_score'] = (float(p.get('relevance') or 0.0)*dfs*fail_share*weight,
                         p.get('objectiveRank') or 99)
    if mwl:
        finding['_moments'] = (float(p.get('relevance') or 0.0)*mwl['spells']*mwl['clear']/max(1, decided)*weight,
                               p.get('objectiveRank') or 99)
        finding['_fallback'] = mwl['fallback']
    return finding


# ---------- priorities and the next objective ----------
def _prioritise(findings, profile):
    work = [f for f in findings if f['verdict'] in S.CRITICISM]
    work.sort(key=lambda f: (-f['_score'][0], f['_score'][1], f['key']))
    cap = int(profile['evidence']['maxWorkItems'])
    kept = []
    for f in work:
        if len(kept) < cap:
            kept.append(f)
        else:
            verdicts = profile['detectors'][f['observation']]['verdicts']
            f['verdict'] = 'review' if 'review' in verdicts else 'describe'
            f['limitations'].append('level_cap')
    for k, f in enumerate(kept):
        f['priority'] = min(10, k+1)
    return [f['id'] for f in kept]


def guidance_item(profile, pointer):
    """The resolved profile's guidance item a FOCUS_GUIDANCE / GENERAL_OBJECTIVES pointer names ({'text', 'tag',
    'sources'}), or None: the item may be filtered out by the band (levels 5-6), or not FEDERATION-tagged."""
    part, source, start = pointer
    for g in profile.get(part) or ():
        if isinstance(g, dict) and (g.get('sources') or [None])[0] == source and \
                str(g.get('text') or '').startswith(start) and g.get('tag') == 'FEDERATION':
            return g
    return None


def fits_age(pointer, profile):
    """May a team of the profile's age be given this GENERAL_OBJECTIVES item? False when it is older than the item's
    source's top age (GENERAL_OBJECTIVE_TOP_AGE); True without a top age or a known age group."""
    top = GENERAL_OBJECTIVE_TOP_AGE.get(pointer[1])
    age = profile.get('ageGroup')
    if top is None or age not in S.AGE_ORDER:
        return True
    return S.AGE_ORDER[age] <= S.AGE_ORDER[top]


def format_misfit(text, profile):
    """A match format or a formation a guidance text names that is not the team's (format-appropriate focus): '9v9'
    for an 11v11 team, '1-3-2-1' (7 players) for a 9v9 team; None when it names none or only the team's own."""
    players = int(((profile.get('format') or {}).get('players')) or 11)
    for m in MATCH_FORMAT_RE.finditer(text or ''):
        if int(m.group(1)) != players:
            return m.group(0)
    for m in FORMATION_RE.finditer(text or ''):
        if sum(int(x) for x in m.group(0).split('-')) != players:
            return m.group(0)
    return None


def focus_problems(text, profile, literals=()):
    """Why a suggested-focus text cannot be shown ([] = it can): explain.validate_free with the quoted guidance and
    its source ids as literal facts (a paraphrase of the source keeps its own numbers, like a drill's quoted lines),
    no number mode (the numbers are the source's, not measurements), at most 400 characters (schema NextObjective)."""
    from . import explain as EX
    from . import language as L
    facts = L.Facts()
    facts.literals.update(x for x in literals if x)
    if not isinstance(text, str):
        return ['empty']
    # the source ids are checked as ids of the profile's sources, not as words ("EF-SWIFT-SUPPORT" is no use of the
    # level-3 concept "support")
    m = FOCUS_SOURCE_RE.search(text)
    ids = m.group(1).split(', ') if m else []
    out = [f'{len(text)} characters (at most 400)'] if len(text) > 400 else []
    out += [f'source {i} is not one of the profile\'s' for i in ids if i not in set(profile.get('sources') or ())]
    return out+EX.validate_free(text[:m.start()] if m else text, profile, facts, numbers=False, limit=400)


def focus_text(profile, name, item):
    """The suggested focus's text: the label, the check's level name (None for a general learning objective), the
    guidance item's paraphrase with its source ids, or PitchIQ's own proposal when no item is quoted; None when no
    form of it passes focus_problems (a long item is tried without the check's name, which the drill still gives)."""
    if item is None:
        if not name:
            return None
        forms = [f'{LEVEL_FOCUS_LABEL}: {name}. {LEVEL_FOCUS_OWN}']
        literals = ()
    else:
        body = item['text'].strip()
        body = body if body.endswith(('.', '!', '?')) else body+'.'
        src = 'Source: '+', '.join(item['sources'])+'.'
        forms = ([f'{LEVEL_FOCUS_LABEL}: {name}. {body} {src}'] if name else []) + \
            [f'{LEVEL_FOCUS_LABEL}: {body} {src}']
        literals = [item['text'].strip(), body]+list(item['sources'])
    for text in forms:
        if not focus_problems(text, profile, literals):
            return text
    return None


def focus_exclusions(findings, profile, shown=None, unsupported=()):
    """The checks the level's suggested focus may not name: off, without a level name, not assessable on the match
    format (assess's format_not_supported, drills.format_supports), or demonstrated in this match: a strength, every
    decided opportunity demonstrated, a demonstrated share at the level's strength bar (shown: {detector:
    (demonstrated, not demonstrated)}), or drills.demonstrated_checks (the optional drill's own rule, so the drill and
    the focus name the same check)."""
    from . import drills as DR
    skills = profile.get('skills') or {}
    bar = float(profile['evidence']['strengthShare'])
    out = set(unsupported or ()) | {f['observation'] for f in findings if f['verdict'] == 'strength'}
    out |= DR.demonstrated_checks(findings, profile)
    for d, dp in profile['detectors'].items():
        dem, nd = (shown or {}).get(d, (0, 0))
        skill = SKILL_OF.get(d)
        if not dp.get('enabled') or not skill or not skills.get(skill) or not DR.format_supports(profile, d):
            out.add(d)
        elif dem >= 1 and (nd == 0 or dem/(dem+nd) >= bar-1e-9):
            out.add(d)
    return out


def level_focus(profile, excluded=(), checks=True):
    """The level's suggested focus when the match gives no objective (polish round): {'skill' (None for a general
    learning objective), 'text', 'tag' ('FEDERATION' | 'PITCHIQ-PROPOSAL'), 'sources'} or None. In this order: the
    enabled check (lowest objectiveRank) with a FEDERATION guidance item of the level that passes the text gate and
    names no other format; the level's general learning objective (GENERAL_OBJECTIVES), never for a team older than its
    source's ages (fits_age); the enabled check as PitchIQ's own focus. Checks in `excluded` (focus_exclusions) are never named; checks=False offers the general objectives
    only (report.py's fallback, which cannot see what the match demonstrated)."""
    lvl = int(profile['level'])
    skills = profile.get('skills') or {}
    ranked = []
    if checks:
        for d, dp in profile['detectors'].items():
            skill = SKILL_OF.get(d)
            if d in set(excluded or ()) or not dp.get('enabled') or not skill or not skills.get(skill):
                continue
            ranked.append((dp['params'].get('objectiveRank') or 99, S.DETECTORS.index(d), skill))
        ranked.sort()
    for _, _, skill in ranked:
        for ptr in FOCUS_GUIDANCE.get(lvl, {}).get(skill, ()):
            item = guidance_item(profile, ptr)
            if item is None or format_misfit(item['text'], profile):
                continue
            text = focus_text(profile, skills[skill], item)
            if text:
                return {'skill': skill, 'text': text, 'tag': 'FEDERATION', 'sources': list(item['sources'])}
    for ptr in GENERAL_OBJECTIVES.get(lvl, ()):
        item = guidance_item(profile, ptr)
        if item is None or format_misfit(item['text'], profile) or not fits_age(ptr, profile):
            continue
        text = focus_text(profile, None, item)
        if text:
            return {'skill': None, 'text': text, 'tag': 'FEDERATION', 'sources': list(item['sources'])}
    for _, _, skill in ranked:
        text = focus_text(profile, skills[skill], None)
        if text:
            return {'skill': skill, 'text': text, 'tag': S.ORIGIN_P, 'sources': []}
    return None


def is_level_focus(next_objective):
    """Is a NextObjective the level's suggested focus (level_focus), not an objective from the match?"""
    return isinstance(next_objective, dict) and not next_objective.get('fromMatch') and \
        str(next_objective.get('text') or '').startswith(LEVEL_FOCUS_LABEL)


def _next_objective(findings, priorities, profile, shown=None, unsupported=()):
    """The top work item's skill; else the level's next objective among the enabled checks the format supports that
    this range DECIDED often enough and the team did not show (shown: {detector: (demonstrated, not demonstrated)}):
    at least the level's strengthSpells decided opportunities (the count a strength needs), at least one not
    demonstrated, and a demonstrated share of the decided ones below the level's strengthShare, with no strength
    reported (critic S-8 / COACH M3, M5; fix round 2: a U8 team that kept someone back 3 of 3 times was told to work
    on "someone ready to protect our goal", and after fix round 1 the professional-like scene still was at levels
    1-3, on 3 behind-the-ball opportunities that were all unclear: demonstrated runs of 2.4, 0.4 and 0.6 s). No such
    check: no objective (NO_OBJECTIVE), never one the match gave no evidence for."""
    by_id = {f['id']: f for f in findings}
    skills = profile.get('skills') or {}
    if priorities:
        f = by_id[priorities[0]]
        skill = SKILL_OF[f['observation']]
        name = skills.get(skill) or skill.replace('_', ' ').capitalize()
        return {'skill': skill, 'text': f'{name}.'[:400], 'fromMatch': True, 'finding': f['id']}
    strong = {f['observation'] for f in findings if f['verdict'] == 'strength'}
    ev = profile['evidence']
    need, bar = int(ev['strengthSpells']), float(ev['strengthShare'])
    cands = []
    for d, dp in profile['detectors'].items():
        skill = SKILL_OF.get(d)
        dem, nd = (shown or {}).get(d, (0, 0))
        decided = dem+nd
        if decided < max(1, need) or nd == 0 or dem/decided >= bar-1e-9:
            continue                    # too little decided play to choose it, or the team showed it
        if dp['enabled'] and skill and skills.get(skill) and d not in strong and d not in unsupported:
            cands.append((dp['params'].get('objectiveRank') or 99, S.DETECTORS.index(d), skill))
    if not cands:
        # polish round: the level's suggested focus from its guidance, labelled as not from this match, never a
        # check this match demonstrated; NO_OBJECTIVE only when the level has none to offer
        focus = level_focus(profile, focus_exclusions(findings, profile, shown, unsupported))
        if focus:
            return {'skill': focus['skill'], 'text': focus['text'], 'fromMatch': False, 'finding': None}
        return {'skill': None, 'text': NO_OBJECTIVE, 'fromMatch': False, 'finding': None}
    _, _, skill = min(cands)
    return {'skill': skill, 'text': f'From the level\'s objectives, not from this match: {skills[skill]}.'[:400],
            'fromMatch': False, 'finding': None}


def phase_available(obs):
    """Did anything name who had the ball on this run: the possession estimate (possession-4 available) or coach-marked
    spans that name a team?"""
    pq = (obs.get('timeline') or {}).get('phaseQuality') or {}
    coach = pq.get('coach') or {}
    return bool(pq.get('available')) or float(coach.get('own') or 0.0)+float(coach.get('opp') or 0.0) > 0.0


def unavailable(obs, profile):
    """The report's MATCH OVERVIEW 'unavailable' records (DESIGN 4.7; schema Unavailable) for this run and level:
    the level's focus areas PitchIQ cannot measure (profiles.unavailable_for), plus every phase-dependent check when
    neither the possession estimate nor a coach-marked span named a team (phase_available)."""
    from . import profiles as PR
    return PR.unavailable_for(int(profile['level']), phase_available=phase_available(obs))


# ---------- the stage ----------
def assess(obs, roles, profile, t0=None, t1=None):
    """The 'assess' stage (INTERFACES.md LEAD 4): {'findings': [Finding (text None, drills [], suggestedAction None or
    a principle)], 'detectors': [DetectorStatus of the six detectors], 'priorities': [work-item finding ids, ranked],
    'nextObjective': NextObjective} of one team from its obs and roles files and the resolved profile, over the report
    range [t0, t1] (absolute seconds; an opportunity belongs to the range when its middle lies inside it)."""
    lo, hi = _range(obs, t0, t1)
    timeline = obs['timeline']
    scale = timeline['scale']
    step = float(timeline.get('step') or .2)
    team = obs['team']
    opps = list(obs['opportunities'])+list((roles or {}).get('opportunities') or [])
    cov = dict(obs.get('detectors') or {})
    cov.update((roles or {}).get('detectors') or {})
    moments = {}
    for m in list(obs.get('moments') or [])+list((roles or {}).get('moments') or []):
        moments.setdefault(m['opportunity'], []).append(m)
    for v in moments.values():
        v.sort(key=lambda m: m['t'])
    index = RO.RoleIndex(roles['rolebook']) if roles and roles.get('rolebook') else None
    ctx = {'profile': profile, 'team': team, 'timeline': timeline, 'scale': scale, 'step': step, 'lo': lo, 'hi': hi,
           'moments': moments, 'index': index, 'players': int(str(timeline.get('format') or '11v11').split('v')[0]),
           'too_little': {}, 'by_id': {}, 'settle': settle_index(timeline)}
    pq = timeline.get('phaseQuality') or {}
    marked = SI.one_sided(pq.get('coach'), bool(pq.get('available')), team)
    available = phase_available(obs)
    findings, statuses = [], []
    shown, unsupported = {}, set()
    for d in S.DETECTORS:
        dp = profile['detectors'][d]
        c = cov.get(d)
        wide = profile['detectors']['wide_defender_advance']
        reframed = d == 'behind_ball' and not wide['enabled'] and wide['params'].get('reframedAs') == 'behind_ball'
        in_range = [o for o in opps if (o['detector'] == d or (reframed and o['detector'] == 'wide_defender_advance'))
                    and o['abstain'] is None and lo-1e-6 <= (o['t0']+o['t1'])/2.0 <= hi+1e-6]
        state = _relevant_phase(d)
        phase_s = _phase_seconds(timeline, state, lo, hi)
        elig_s = CX.eligible_seconds(in_range, step)
        if not dp['enabled']:
            statuses.append({'detector': d, 'status': 'off', 'reason': None, 'eligibleSeconds': round(elig_s, 2),
                             'phaseSeconds': round(phase_s, 2), 'opportunities': len(in_range)})
            continue
        families = {}
        for o in in_range:
            cl = classify(o, dp['params'], scale, step, as_detector=d if o['detector'] != d else None)
            if cl is not None:
                families.setdefault(_family(o, d), []).append((o, cl))
                ctx['by_id'][o['id']] = cl
        # pre-freeze (FACTS F-9, second half): the next objective and the focus read the check in its own phase; the
        # phase-unknown attacking-shape family (nobody was known to have the ball) says nothing of "when we have the
        # ball" (AM A / B levels 1-5 were given "Spread out" from phase-unknown describes alone)
        phased = [v for fam, v in families.items() if fam != 'unknown']
        shown[d] = (sum(1 for v in phased for _, c in v if c['outcome'] == 'demonstrated'),
                    sum(1 for v in phased for _, c in v if c['outcome'] == 'not_demonstrated'))
        own = [o for o in in_range if o['detector'] == d]
        reason = _reason(d, c, phase_s, marked, available, ctx['players'])
        if reason == 'format_not_supported':
            unsupported.add(d)
        if not (own or families):
            # (level 1's reframed advances without a usable ball count as nothing assessed)
            if reason is None:
                # the phase occurred and nothing of it was lost to a gate: the situation never arose (assessed)
                statuses.append({'detector': d, 'status': 'assessed', 'reason': None, 'eligibleSeconds': 0.0,
                                 'phaseSeconds': round(phase_s, 2), 'opportunities': 0})
            else:
                statuses.append({'detector': d, 'status': 'not_assessed', 'reason': reason, 'eligibleSeconds': 0.0,
                                 'phaseSeconds': round(phase_s, 2), 'opportunities': 0})
            continue
        before = len(findings)
        for fam in sorted(families):
            f = _finding(d, fam, families[fam], ctx)
            if f is not None:
                findings.append(f)
        # decided once the report's cap on 'moments worth a look' reviews is known (a dropped one leaves the
        # finding its family gives without it, or nothing)
        statuses.append({'detector': d, '_made': findings[before:],
                         '_suppressed': any(c_['suppressed'] for v in families.values() for _, c_ in v),
                         'eligibleSeconds': round(elig_s, 2), 'phaseSeconds': round(phase_s, 2),
                         'opportunities': len(in_range)})
    findings = _cap_moments(findings, profile)
    final = {f['id'] for f in findings}
    for st in statuses:
        if '_made' not in st:
            continue
        made = any(f['id'] in final for f in st.pop('_made'))
        suppressed = st.pop('_suppressed')
        if not made and suppressed:
            # every criticism the check could have made was within a formation change (S-3): say so
            st.update(status='not_assessed', reason='formation_changing')
        elif not made and st['detector'] in ctx['too_little']:
            # a failing pattern below the review floor (S-1): too little play in this range to judge the check;
            # 'too_few_situations' when the relevant seconds were enough but the situations too few (fix round 3)
            st.update(status='not_assessed', reason=ctx['too_little'][st['detector']])
        else:
            st.update(status='assessed', reason=None)
    statuses = [{k: st[k] for k in ('detector', 'status', 'reason', 'eligibleSeconds', 'phaseSeconds',
                                    'opportunities')} for st in statuses]
    priorities = _prioritise(findings, profile)
    for f in findings:
        f.pop('_score', None)
    order = {'strength': 0, 'work_on': 1, 'try_this': 1, 'review': 2, 'describe': 3}
    findings.sort(key=lambda f: (order[f['verdict']], f['priority'] or 99, S.DETECTORS.index(f['observation']),
                                 f['key']))
    return {'findings': findings, 'detectors': statuses, 'priorities': priorities,
            'nextObjective': _next_objective(findings, priorities, profile, shown, unsupported)}


def _reason(detector, cov, phase_s, marked, available, players=11):
    """Why a check had nothing to assess (critic S-7: 76 of 666 real-run not-assessed statuses had no reason):
    'coach_marks_one_side' when the engine is unavailable and the coach marked only one team's possession while the
    check needs the other's; 'phase_absent' when the relevant phase never occurred although the phase is known;
    'phase_unknown' when no phase was known at all; else the coverage's dominant reason (None: the phase occurred and
    nothing was lost to a gate, so the situation itself never arose)."""
    reason = (cov or {}).get('dominantReason')
    if phase_s > 1e-9:
        if cov is None and (detector == 'wide_defender_advance' or (detector == 'defensive_spacing' and
                                                                   players != 11)):
            return 'role_unconfirmed'       # the role tier was not built: nothing role-based was measured
        return reason
    if marked is not None:
        return 'coach_marks_one_side'
    if available:
        return 'phase_absent'
    return reason or 'phase_unknown'


__all__ = ['assess', 'classify', 'unavailable', 'phase_available', 'MAX_TIMESTAMPS']
