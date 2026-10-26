"""The coaching report v2 (DESIGN v2 9.2): the seven sections a coach reads, assembled from the structured findings
of one team over one range, after the explanations, zones, evidence and drills have been made.

    1 MATCH OVERVIEW        overview: profile notices, footage, data quality and its cause, scale, who had the ball
                            (rough shares with coverage and caveats, or why it is not available), the shape measured
                            per period (described), each check assessed / not assessed with its reason, the categories
                            PitchIQ cannot measure (DESIGN 4.7), a synthesis (overview.text); described findings
    2 WHAT WENT WELL        strengths (demonstrated spells, never an absence of failures)
    3 WHAT NEEDS WORK       needsWork (work_on / try_this, by priority), then reviews in two parts (REVIEW_PARTS):
                            "Possible issues to review" (what could not be confirmed), then "Moments worth a look"
                            (repeated clear moments below the verdict bar: no verdict, no drill)
    4 KEY MOMENTS           keyMoments: the cited evidence, chronological, captions = the findings' validated short
    5 TACTICAL DIAGRAMS     diagrams: each moment's top-down diagram (suggested zones included), the average shape per
                            period and phase, and how our players moved in the 2 s before each finding's first moment
    6 TRAINING DRILLS       drills: drills.select's picks (only for reported findings; optional ones marked)
    7 DEVELOPMENT PROGRESSION   progression (merged by the server at serve time) and the NEXT DEVELOPMENT OBJECTIVE

Why the report is assembled here and worded nowhere else. Spec Part 12 asks for structured findings (observation,
team, players by id, timestamps, context, measurements, coverage, suggested action, level) translated into
coaching language with "a working deterministic fallback" and text validated against the evidence. The findings
keep every structured field (schema 'finding'); their words come from explain.py's claim plans and gate. What this
module writes itself (the overview synthesis, notices, the possession caveat, captions of the shape and movement
diagrams, the next objective) passes explain.validate_free: every number, time, id and formation must be a fact of
the report, no unmeasured event, no concept the level has not unlocked, no blame or generic sentence. A text that
fails is replaced by a shorter plain one or left out; a drill pick whose fields fail is left out (logged), never
reworded here. The v1 report printed bookkeeping as its summary ("N observed strengths met the evidence rules",
critic S9) and claimed things nobody had checked (B4); here the synthesis names what was checked, what was found and,
when nothing met the level's bar, that no finding is not proof of good organisation.

What the assembly guarantees (schema 'report' + cross_check, run by coaching.interpret):
- nothing is forced: sections hold exactly the verdicts assess gave and explain could word; an empty section stays
  empty (spec Part 10); needsWork never exceeds the level's maxWorkItems;
- every cited moment is a key moment of the obs / roles files at the same frame and seek (invariant 3); a key moment
  whose finding explain withheld is re-attached to a reported finding that cites it, or dropped; a finding citing a
  moment the evidence stage did not make is withheld ('moment_missing');
- no-alternative reasons reach the text: coaching.interpret calls explain without ZONES' NoAlternative records, so
  the report adds the validated "why no area is marked" sentence (explain.with_no_alternative) to work items and to
  reviews whose guardrails passed;
- ids stay internal (display labels are rendered at view time); times are absolute source seconds; ASCII only.

Measured (Oct 22, CPU; scratch coaching2/build/text_w5): coaching.interpret with every real stage on EVAL's 30 scene
variants and the five run copies (both teams, engine and coach-marked phases), levels 1-7: 322 reports, 0 schema or
cross_check problems, 0 findings withheld (592: 283 strengths, 49 work items, 79 reviews, 181 descriptions), an
overview text in every report, the most alike pair of explanations inside a report .25 word-trigram Jaccard (DESIGN
6.4 allows .5); 1703 finding, 408 movement and 469 shape diagrams; 49 drill picks for work items and 273 optional ones
for the next objective; 11 "why no area is marked" sentences added; notices: 119 short range, 112 data quality, 42
phase unavailable, 21 formation mismatch, 0 work items without a drill. The whole interpretation tier (profile to
report) takes 21 ms median, 34 ms p90 and 0.28 s at most (the first call of a process), the report stage 4-9 ms; a
report is 70 KB median, 171 KB at most (compact JSON).

Fix round 1 (Oct 23; scratch coaching2/fix1/text): with the coach's marks of one team only and PitchIQ unable to follow
the ball there is no possession share (critic N6: the AM copy read "A 1.0, B 0.0"), a caveat and the notice
'coach_marks_one_side' say what is missing, and PossessionOverview.source says where shares come from ('engine',
'coach', 'mixed'; SERVER + PAGE's request, schema additive); a range just under the 3-minute bar reads "just under 3
minutes" (FACTS L11a: "covers 3.0 minutes" beside "shorter than three minutes", 24 run-copy reports); a team-level
challenge (levels 1-2, or params.teamLevelOnly) draws its player in the neutral support style, never the review
outline (COACH L2: 27 scene moments circled one child); a concept the report uses without defining it (the level 6-7
rest defence in a headline, the next objective or a drill line) is defined in the overview; a work item left without a
drill because of its match format says so (DRILLS' no_drill_for_format).

Fix round 2 (Oct 23; LEAD decision (b), ASSESS's requests; scratch coaching2/fix2/text): the 'moments worth a look'
reviews (assess limitation MOMENTS) stay in section 3 as its own labelled part after the work items and the possible
issues (REVIEW_PARTS, review_part, reviews_by_part: one place for the page's and the PDF's headings and intros;
SERVER / PAGE / PDF via INTERFACES.md), never as a possible issue: the overview says "Moments worth a look, not a
verdict: <label>." apart from "There is one possible issue to review." (round 1 counted them as possible issues in 290
of the 290 reports carrying one on ASSESS's cached Metrica and planted tiers, 8829 reports at levels 3-7; now 0, and
290 name them). With no next objective (critic S-8 / ASSESS: 182 of the 196 run-copy reports, 6643 of the 8829 cached
tiers) the text says what that means, per level band ("No next objective comes from this match: no check of this
level had both enough judged moments and room to improve.") instead of "No next objective could be chosen from this
level's checks."; a withheld top work item with no other one reported now gives no objective too, never the first
unpraised check of the level (the rule S-8 removed from assess). On the 196 run copies only the next objective's
text changes (182); on the 624 scene reports verdicts, withheld and drills are identical, texts change for the 13
moments reviews and the new rest-defence definition (108 findings, 24 overviews).

Polish round (Oct 23): with no objective from the match the NEXT DEVELOPMENT OBJECTIVE is the level's suggested focus
assess chose (assess.level_focus: the label "Suggested focus for this level - from the level's guidance, not from this
match", the check, its FEDERATION guidance item and source ids), checked again here; a withheld top work item with no
other one, or a focus the gate refuses here, gives the level's general learning objective (_no_objective: no check,
since this stage cannot see what the match demonstrated), else no_objective_text. An optional drill is printed only
when it trains the objective's check, and at most one; the suggested focus's drill not under 3 minutes of play when a
line of it says "repeat", "pattern" or "often" (SHORT_REPEAT: EVAL's short_segment scene, level 6). The page and the
PDF print the label, the focus and each source's title with its id (coaching.js / pdf.py split_focus).

Fix round 3 (Oct 24; critic S-3 / B-3, FACTS F-9 = COACH F-4, COACH F-11, F-12; scratch coaching2/fix3/text, TEXT's
files before / after on identical inputs): assess's objective chosen from this match's checks (S-8: judged often
enough, room to improve, no work item) reads "Chosen from this match's checks (not a work item): <check>." (S8_LABEL),
never "From the level's objectives, not from this match" (209 / 14 / 12 / 293 / 187 -> 0 on the COACH verifier's 546
scene reports, the 196 run copies, the user's 252, Metrica game 1's 2072 clip and 252 whole-match reports); an
objective a shown finding says the declared style accepts is replaced by the level's general objective, else a sentence
that says so (STYLE_OBJECTIVE_WORDS; defending/weak_empty at level 7, compact: 1 -> 0); an objective from a work item
starts with a capital (3 -> 0); the overview's rest-defence definition carries the run's behind-the-ball margin
(margin_words: "at least 10 m"); DRILLS' 'only_far_below' reason has its words. text_problems is the report-level
invariant of these (for coaching_eval.report_problems / game-3 G1+), with "professional" never said to a youth team or
below level 5 (2 / 2 / 8 -> 0 on runs / clips / whole). Next objectives changed: 240 of 546 scene reports (208 S-8
relabels, 1 style, 13 capitals or a new short label, 18 where a level 1-2 work item the gate used to withhold is now
shown), every S-8 relabel on the other sets; drill picks change only with those newly shown work items (18) and the
style case (1).

The v1 report (overview / summary / strengths / improvements / reviews / breakdown / priorities / training /
confidence), still served until the server switches to v2 (wave 7), moved unchanged to report_v1.py; build_report,
SECTIONS, MAX_PRIORITIES and the other v1 names are forwarded from here (module __getattr__) so old callers keep
working, while the v2 interpretation fingerprint (coaching.code_files reads import statements) never hashes them.
"""
import copy
import hashlib
import importlib
import json
import math
import re
from collections import Counter

from soccer.coaching import evidence as EV
from soccer.coaching import explain as EX
from soccer.coaching import language as L
from soccer.coaching import schema as S
from soccer.coaching import situations as SI

REPORT_VERSION = 2
FORMATION_MIN_POINTS = 15       # 1 Hz track points (s) of one phase in one period for an average-shape diagram
MAX_FORMATION_DIAGRAMS = 6      # periods x phases shown (the longest first)
MOVEMENT_MIN_M = 1.0            # a player who moved less than this in the 2 s before a moment gets no arrow [P]
CAPTION_REF_MAX = 200           # DiagramRef.caption
SKILL_OF = {d: k for k, d in S.SKILLS.items()}
CRITICISM = S.CRITICISM
# Plain names of the checks for the overview synthesis when profiles.json has no level name for one (a check the
# level does not show as a skill); per level band, the nearest lower band applies. Validated like any other text.
CHECK_NAMES = {
    'pressure_cover': {1: 'going to the ball', 2: 'pressure and cover'},
    'weak_side_balance': {1: 'the far side when the ball is wide', 4: 'far-side protection'},
    'wide_defender_advance': {1: 'someone back when a defender goes forward', 3: 'cover behind a defender who goes '
                                                                                'forward',
                              4: 'support behind advancing wide defenders'},
    'behind_ball': {1: 'someone ready to protect our goal', 3: 'players behind the ball when attacking'},
    'defensive_spacing': {1: 'spacing of our defenders', 4: 'back-line spacing'},
    'attacking_shape': {1: 'spreading out with the ball', 3: 'width and depth with the ball'},
}
PHASE_WORDS = {'own': 'when we had the ball', 'opp': 'when the other team had the ball',
               'unknown': 'when the whole team was in view'}
UNAVAILABLE_REASON = 'The ball and who had it could not be followed reliably on this footage.'   # DESIGN 3.2
WORD_OF = {v: k for k, v in L.NUMBER_WORDS.items() if k not in ('twice', 'thrice')}   # 3 -> 'three' (a fact)
# Why a work item shows no drill (drills.no_drill_reason, or the report could not print drills.select's pick).
NO_DRILL_WORDS = {
    'no_drill_at_level': 'No drill in the library fits {short} at this level.',
    'refused_by_safety': 'No drill for {short} is allowed at this age at this level or below; the safety rules are not '
                         'relaxed to find one.',
    'not_printable': 'The drill proposed for {short} could not be worded for this level, so none is shown.',
    # DRILLS fix round 1 (critic S-9): every drill for the check is set up for a larger format or more players
    'no_drill_for_format': 'No drill for {short} fits this match format and its number of players.',
    # DRILLS fix round 3 (coach F-6): the only fitting drills are more than one level below (younger levels' games)
    'only_far_below': 'No drill in the library fits {short} at this level or close below it; drills meant for much '
                      'younger levels are not offered.',
}
# Fix round 3 (FACTS F-9 = COACH F-4): assess's objective chosen from this match's checks (critic S-8: a check this
# range judged often enough, with room to improve, and no work item) read "From the level's objectives, not from this
# match" on 213 of 742 coach-verifier reports, 96 of them beside a review of that very check. It is from this match:
# S8_LABEL says so (the page and the PDF add no origin line to a text that names one: coaching.js / pdf.py
# ORIGIN_NAMED).
S8_OLD = "From the level's objectives, not from this match: "
S8_LABEL = "Chosen from this match's checks (not a work item): "
# COACH F-12: the objective's check is what the declared style accepts in a finding this report shows ("Because the
# selected style is compact, open space on the far side is accepted", then "Defensive balance (far side)" as the
# objective): the level's general objective instead, else this
STYLE_OBJECTIVE_WORDS = ('No next objective comes from this match: the check with room to improve, {name}, showed '
                         'what the selected {style} style accepts.')
# The two parts of section 3's reviews (fix round 2, LEAD decision (b)), in this order; the page and the PDF print each
# non-empty part under its title and intro, after the work items (review_part(f) says which part a review is in):
# 'issues' = reviews held back by what PitchIQ could not confirm (spec Part 15 POSSIBLE ISSUE TO REVIEW), 'moments' =
# 'moments worth a look' (repeated clear failures below the level's verdict bar: no verdict, no drill).
REVIEW_PARTS = (
    {'part': 'issues', 'title': 'Possible issues to review',
     'intro': 'PitchIQ could not confirm everything about these: worth a look on the video, not a verdict.'},
    {'part': 'moments', 'title': 'Moments worth a look',
     'intro': "Clear moments, in separate spells, that stayed below this level's bar for a verdict: no verdict and "
              'no drill. Watch them with the players and decide whether they matter for how you want to play.'},
)
# The next objective when assess found none (critic S-8, ASSESS fix round 2: no enabled check of the level had both
# strengthSpells judged opportunities and a failure with a demonstrated share under strengthShare, with no strength):
# what the rule means, per level band, never an objective the match gave no evidence for.
NO_OBJECTIVE_WORDS = {
    1: 'No new goal is suggested from this clip: there was too little to judge, or the team already showed what this '
       'level looks for.',
    3: 'No next objective comes from this match: no check of this level had both enough judged moments and room to '
       'improve.',
}
# Repetition words a text may not use in a range under 3 minutes (polish round: coaching_eval.REPETITION, which the
# scenes' no_repetition_wording check applies to every text; language.SHORT_BANNED has no "repeat")
SHORT_REPEAT = re.compile(r'\b(repeated(?:ly)?|repeats?|patterns?|often)\b', re.I)
KEYFRAME_ADVICE = (2, 3)        # pitch keyframes the data-quality notice suggests (the calibration audit: two or three
                                # landmark frames anchor a keyless registration; language_v1.KEYFRAME_ADVICE)
# The dominant data-quality cause (situations.py: the largest loss of lineup -> in kit -> confirmed -> registered) and
# what would help (critic B4: the v1 Veo text named a camera limit when identities were the loss).
DATA_QUALITY_HELP = {
    'registration': 'Many players in view had no pitch position because the pitch registration was weak; two or three '
                    'pitch keyframes would help.',
    'identity': 'Many players in view could not be identified for sure; identity corrections or a longer clip would '
                'help.',
    'out_of_view': 'Many players were out of the camera view; a wider camera view would help.',
}


# ---------------------------------------------------------------------------------------------------------------
# small helpers

def _num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))


def _at(table, level):
    keys = [k for k in table if k <= level]
    return table[max(keys)] if keys else table[min(table)]


def _check_name(profile, detector):
    """The level's name of a check: profiles.json skills, else CHECK_NAMES."""
    skill = SKILL_OF.get(detector)
    name = (profile.get('skills') or {}).get(skill) if skill else None
    return name if isinstance(name, str) and name else _at(CHECK_NAMES[detector], int(profile['level']))


def _lower_first(s):
    return s[:1].lower()+s[1:] if s and not (len(s) > 1 and s[1].isupper()) else s


def _join(items):
    """'a and b'; with three or more items, or an item that has a comma itself ("even, compact back line"), the
    items are separated by semicolons so the list stays readable."""
    items = list(dict.fromkeys(i for i in items if i))
    if len(items) <= 1:
        return ''.join(items)
    if len(items) > 2 or any(',' in i or ' and ' in i for i in items):
        return '; '.join(items)
    return items[0]+' and '+items[1]


def _facts(*extra):
    """language.Facts of the given values (dicts / lists walked as in facts_for)."""
    return L.facts_for(extra={'v': list(extra)})


def _ok(text, profile, facts=None, **kw):
    """text when explain.validate_free accepts it, else None."""
    return text if text and not EX.validate_free(text, profile, facts, **kw) else None


def _phase_qualifier(obs):
    """'(estimated)', '(coach-marked)' or '(estimated, partly coach-marked)' for the run's phase source."""
    pq = (obs.get('timeline') or {}).get('phaseQuality') or {}
    coach = pq.get('coach') or {}
    coach_named = float(coach.get('own') or 0.0)+float(coach.get('opp') or 0.0) > 0.0
    if pq.get('available') and coach_named:
        return EX.QUALIFIER['mixed']
    return EX.QUALIFIER['coach'] if coach_named else EX.QUALIFIER['possession-4']


def _phase_available(obs):
    pq = (obs.get('timeline') or {}).get('phaseQuality') or {}
    coach = pq.get('coach') or {}
    return bool(pq.get('available')) or float(coach.get('own') or 0.0)+float(coach.get('opp') or 0.0) > 0.0


def _record_id(*parts):
    return S.record_id('dg', *parts)


def review_part(finding):
    """'moments' for a 'moments worth a look' review (assess limitation MOMENTS), else 'issues' (REVIEW_PARTS)."""
    return 'moments' if EX.MOMENTS in (finding.get('limitations') or ()) else 'issues'


def reviews_by_part(report):
    """[(part record of REVIEW_PARTS, its reviews)] in REVIEW_PARTS order, the empty parts left out: what the page and
    the PDF print under WHAT NEEDS WORK after the work items."""
    out = []
    for part in REVIEW_PARTS:
        fs = [f for f in report.get('reviews') or () if review_part(f) == part['part']]
        if fs:
            out.append((part, fs))
    return out


def _stand_in_fingerprint(obs, roles, profile, rng):
    """The interpretation fingerprint coaching.interpret overwrites (a direct call still gets a valid stamp)."""
    raw = json.dumps({'obs': obs.get('fingerprint'), 'roles': (roles or {}).get('fingerprint'),
                      'profile': profile.get('sha'), 'level': profile.get('level'), 'range': rng},
                     sort_keys=True, separators=(',', ':'), allow_nan=False)
    return hashlib.sha256(raw.encode('utf-8')).hexdigest()


# ---------------------------------------------------------------------------------------------------------------
# 1 MATCH OVERVIEW

def one_sided(obs):
    """The team whose possession alone the coach marked while PitchIQ could not follow the ball (situations.one_sided:
    engine unavailable, the coach's spans name one team only), else None."""
    pq = ((obs.get('timeline') or {}).get('phaseQuality')) or {}
    return SI.one_sided(pq.get('coach'), bool(pq.get('available')), obs['team'])


def _possession(obs):
    """PossessionOverview: the possession-4 estimate and / or the coach-marked spans, with plain caveats (possession
    binding rules 4, 6, 8: stoppages mostly missed, shares only roughly, the qualifier). `source` says where the shares
    come from: 'engine' (possession-4), 'coach' (the coach's spans only), 'mixed' (the coach's spans over the engine,
    situations.py's merged per-sample phase since fix round 1, critic B-11), None (nothing named a team). With the
    coach's marks of one team only and no engine there is no share (critic N6: Team A's two AM spells read "A 1.0,
    B 0.0") and the caveat says what is missing."""
    tl = obs.get('timeline') or {}
    pq = tl.get('phaseQuality') or {}
    coach = pq.get('coach') or {}
    own, opp = float(coach.get('own') or 0.0), float(coach.get('opp') or 0.0)
    coach_named = own+opp > 0.0
    engine = bool(pq.get('available'))
    team = obs['team']
    other = 'B' if team == 'A' else 'A'
    marked = one_sided(obs)
    shares = pq.get('roughShares') if engine else None
    if not engine and coach_named and marked is None:   # the coach's spans, rounded like the engine's (rule 6)
        a = round(round(own/(own+opp)*20.0)/20.0, 2)
        shares = {team: a, other: round(1.0-a, 2)}
    source = ('mixed' if coach_named else 'engine') if engine else ('coach' if coach_named else None)
    cov = pq.get('coverage') if engine and _num(pq.get('coverage')) else None
    if not engine and coach_named:
        seg = max(1e-9, float(tl.get('t1', 0.0))-float(tl.get('t0', 0.0)))
        cov = round(min(1.0, float(coach.get('seconds') or 0.0)/seg), 3)
    caveat = None
    broadcast = (tl.get('footage') or {}).get('kind') == 'broadcast'
    if engine:
        caveat = ('Estimated by PitchIQ from the tracking, so the shares are rough. The few seconds around each change '
                  'of who had the ball are left out')
        caveat += ('; stoppages are mostly missed on broadcast footage, so some dead-ball time may count as play.'
                   if broadcast else '.')
        if coach_named:
            caveat += ' Where you marked who had the ball, your spans are used (coach-marked).'
    elif coach_named:
        caveat = ('Who had the ball comes from the spans you marked (coach-marked); PitchIQ could not follow it on this '
                  'footage.')
        if marked is not None:
            caveat += (f' You marked only Team {marked}\'s possession, so no share is given; mark Team '
                       f'{"B" if marked == "A" else "A"}\'s spells too for one.')
    if engine:
        quality = pq.get('level') if pq.get('level') in ('good', 'limited') else 'limited'
    else:
        quality = 'limited' if coach_named else 'unavailable'
    return {'available': engine or coach_named, 'quality': quality,
            'reason': None if (engine or coach_named) else UNAVAILABLE_REASON,
            'roughShares': shares, 'coverage': cov, 'caveat': caveat, 'source': source}


def _report_notices(obs, roles, profile, rng, short):
    """Notices this run adds to the profile's (DESIGN 3.4, 3.2, 3.6): a formation reading that differs from the
    selected one, confirmed roles the positions contradict, a short range, the phase source. No player id (notices
    are plain text on the page)."""
    out = []
    lo, hi = rng['t0'], rng['t1']
    for c in ((roles or {}).get('rolebook') or {}).get('formationChecks') or ():
        if c.get('mismatch') and c.get('measured') and c.get('intended') and c['t1'] >= lo and c['t0'] <= hi:
            text = (f'From {L.fmt_time(max(c["t0"], lo))} to {L.fmt_time(min(c["t1"], hi))} the shape measured looked '
                    f'like {c["measured"]}; your selected {c["intended"]} is used as the plan, and role-based points '
                    'for players without a confirmed role stay possible issues to review.')
            ok = _ok(text, profile, _facts({'t': max(c['t0'], lo)}, {'t': min(c['t1'], hi)}, c['measured'],
                                           c['intended']), numbers=False, limit=400)
            if ok:
                out.append({'code': 'formation_mismatch', 'text': ok})
    conflicts = {r['id'] for r in ((roles or {}).get('rolebook') or {}).get('records') or ()
                 if r.get('conflict') and r.get('t1', hi) >= lo and r.get('t0', lo) <= hi}
    if conflicts:
        out.append({'code': 'role_conflict', 'text': 'Where a player stood did not match a role you confirmed for him '
                                                     'in this range; please check the player roles in the setup.'})
    if short:
        need = int(((profile.get('detectors') or {}).get('pressure_cover') or {}).get('params', {})
                   .get('shortRangeMinFailingSpells') or 3)
        text = (f'This range is shorter than three minutes: a short clip cannot show what the team usually does, '
                f'and a work item needs at least {WORD_OF.get(need, need)} separate moments.')
        ok = _ok(text, profile, _facts({'seconds': EX.SHORT_RANGE_S}, need), numbers=False, limit=400, short=True)
        if ok:
            out.append({'code': 'short_range', 'text': ok})
    cause = ((obs.get('timeline') or {}).get('dataQuality') or {}).get('dominantCause')
    if cause in DATA_QUALITY_HELP:           # critic B4: name the dominant cause and what would help
        ok = _ok(DATA_QUALITY_HELP[cause], profile, _facts(*KEYFRAME_ADVICE), numbers=False, limit=400)
        if ok:
            out.append({'code': 'data_quality', 'text': ok})
    if not _phase_available(obs):
        out.append({'code': 'phase_unavailable',
                    'text': 'Who had the ball could not be followed on this footage, so the checks that depend on it '
                            'were not made. Marking who had the ball in the possession strip (coach-marked spans) '
                            'would let them run.'})
    marked = one_sided(obs)
    if marked is not None:
        # critic N6: with one team's spans marked the phase notice is gone (the phase is known where marked), so the
        # coach is told which checks wait for the other team's spells (assess reason 'coach_marks_one_side')
        other = 'B' if marked == 'A' else 'A'
        text = (f'You marked only Team {marked}\'s possession and PitchIQ could not follow the ball on this footage, '
                f'so the checks played while Team {other} had the ball are not assessed until you mark Team '
                f'{other}\'s spells too.')
        ok = _ok(text, profile, None, numbers=False, limit=400)
        if ok:
            out.append({'code': 'coach_marks_one_side', 'text': ok})
    return out


def duration_words(seconds):
    """The report range as words: language.fmt_duration, except that a range under the 3-minute short-range bar that
    rounds to it reads "just under 3 minutes" (FACTS L11a: BC180's 179.97 s read "This report covers 3.0 minutes"
    beside "This range is shorter than three minutes")."""
    s = float(seconds)
    if s < EX.SHORT_RANGE_S and L.fmt_duration(s) == L.fmt_duration(EX.SHORT_RANGE_S):
        return f'just under {int(round(EX.SHORT_RANGE_S/60.0))} minutes'
    return L.fmt_duration(s)


def _overview_text(profile, assessment, sections, obs, rng):
    """The synthesis (critic S9): what was checked, what went well, what to work on, what to review, and when
    nothing met the level's bar, that no finding is not proof of good organisation; plus where the phase came from.
    Number-free at level 1."""
    lvl = int(profile['level'])
    counts = EX.register_for(profile)['numbers'] != 'none'
    statuses = [d for d in assessment.get('detectors') or () if d['status'] != 'off']
    assessed = [d for d in statuses if d['status'] == 'assessed']
    sents = []
    if counts and _num(rng.get('seconds')) and rng['seconds'] > 0:
        sents.append(f'This report covers {duration_words(rng["seconds"])} of play.')
    else:
        sents.append('This report looks at how the team played in this clip.')
    if statuses:
        if counts and not assessed:
            sents.append(f'None of the {len(statuses)} checks for this level could be made on this footage.')
        elif counts:
            sents.append(f'{len(assessed)} of the {len(statuses)} checks for this level could be made.')
        elif len(assessed) == len(statuses):
            sents.append('Every check for this level could be made.')
        elif not assessed:
            sents.append('None of the checks for this level could be made on this video.')
        else:
            sents.append('Some checks for this level could not be made on this video.')
    # the findings' own short texts (validated by explain, level-worded, specific: "Tall and deep" rather than the
    # skill's "Spread out: tall, wide, deep"), first letter lowered inside the sentence
    good = [_lower_first(f['text']['short']) for f in sections['strengths']]
    work = [_lower_first(f['text']['short']) for f in sections['needsWork']]
    if good:
        sents.append('What went well: '+_join(good)+'.')
    if work:
        sents.append(('Something to try next: ' if lvl <= 2 else 'To work on: ')+_join(work)+'.')
    # the possible issues (held back by what could not be confirmed) and the 'moments worth a look' (fix round 2:
    # clear, below the verdict bar) are said apart, the latter by name and never as an issue or a verdict
    issues = [f for f in sections['reviews'] if review_part(f) == 'issues']
    moments = [_lower_first(f['text']['short']) for f in sections['reviews'] if review_part(f) == 'moments']
    n_rev = len(issues)
    if n_rev:
        sents.append('There is also something to look at on the video.' if not counts else
                     'There is one possible issue to review.' if n_rev == 1 else
                     f'There are {n_rev} possible issues to review.')
    if not (good or work or n_rev):
        # plain at levels 1-3 (critic S10: no "evidence bar" wording in development reports)
        sents.append(('Nothing in this range was clear enough to list at this level.' if lvl <= 3 else
                      "Nothing met this level's evidence bar in this range.") +
                     ' That is not proof that all went well: PitchIQ could check only part of the play.')
    if moments:
        sents.append(('Moments worth a look, not a verdict: ' if len(moments) == 1 else
                      'Moments worth a look, not verdicts: ')+_join(moments)+'.')
    pq = (obs.get('timeline') or {}).get('phaseQuality') or {}
    if _phase_available(obs):
        q = _phase_qualifier(obs)
        sents.append({EX.QUALIFIER['possession-4']: 'Who had the ball is an estimate from the tracking.',
                      EX.QUALIFIER['coach']: 'Who had the ball comes from the spans you marked.',
                      EX.QUALIFIER['mixed']: 'Who had the ball is an estimate from the tracking, with your marked '
                                             'spans where you gave them.'}[q])
    elif pq:
        sents.append('Who had the ball could not be followed on this footage, so the checks that depend on it were '
                     'not made.')
    facts = _facts({'seconds': rng.get('seconds')}, len(assessed), len(statuses), n_rev, len(good), len(work))
    short = _num(rng.get('seconds')) and rng['seconds'] < EX.SHORT_RANGE_S
    text = ' '.join(sents)
    ok = _ok(text, profile, facts, short=short, limit=1500)
    if ok:
        return ok
    # a name a level's gate refuses (e.g. a word it has not unlocked) drops the names, never the facts' honesty
    plain = [s for s in sents if not s.startswith(('What went well', 'Something to try', 'To work on',
                                                   'Moments worth a look'))]
    return _ok(' '.join(plain), profile, facts, short=short, limit=1500)


def overview(obs, roles, profile, assessment, sections, rng, short, extra_notices=()):
    tl = obs['timeline']
    lo, hi = rng['t0'], rng['t1']
    formations = [dict(p) for p in tl.get('formationPeriods') or () if p['t1'] >= lo and p['t0'] <= hi]
    from soccer.coaching import assess as AS
    return {'notices': _report_notices(obs, roles, profile, rng, short)+list(extra_notices),
            'footage': copy.deepcopy(tl['footage']), 'dataQuality': copy.deepcopy(tl['dataQuality']),
            'scale': copy.deepcopy(tl['scale']), 'possession': _possession(obs), 'formations': formations,
            'detectors': copy.deepcopy(assessment.get('detectors') or []),
            'unavailable': AS.unavailable(obs, profile),
            'text': _overview_text(profile, assessment, sections, obs, rng)}


# ---------------------------------------------------------------------------------------------------------------
# 5 TACTICAL DIAGRAMS: the average shape per period and phase, and how our players moved before a moment

def _modal_roles(windows, period, phase):
    votes = {}
    for w in windows or ():
        if w.get('period') != period or float(w.get(phase) or 0.0) < .5:
            continue
        for pid, role in (w.get('roles') or {}).items():
            if role:
                votes.setdefault(pid, Counter())[role] += 1
    return {pid: c.most_common(1)[0][0] for pid, c in votes.items()}


def formation_diagrams(obs, profile, rng):
    """Average positions of our outfield players per formation period (11v11; the whole range on small formats) and
    per phase, from the 1 Hz track of obs.roleInputs.context (complete samples only, at least FORMATION_MIN_POINTS),
    in the attack frame drawn attacking right. Described, never judged (spec Part 7: a formation is not fixed dots;
    DESIGN 9.2 #5, critic N7)."""
    ctx = ((obs.get('roleInputs') or {}).get('context') or {})
    tr = ctx.get('track') or {}
    ids, ts = ctx.get('ids') or [], tr.get('t') or []
    if not ids or not ts:
        return []
    scale = obs['timeline']['scale']
    Lm, Wm = float(scale['lengthM']), float(scale['widthM'])
    lo, hi = rng['t0'], rng['t1']
    periods = [p for p in ctx.get('periods') or () if p['t1'] >= lo and p['t0'] <= hi] or \
        [{'index': -1, 't0': lo, 't1': hi, 'label': None, 'confidence': None}]
    phases = tr.get('phase') or ['unknown']*len(ts)
    named = any(ph in ('own', 'opp') for ph in phases)
    wanted = ('own', 'opp') if named else ('unknown',)
    complete = tr.get('complete') or [True]*len(ts)
    fps = float(obs['segment'].get('fps') or 25.0)
    qual = _phase_qualifier(obs)
    out = []
    for p in periods:
        a, b = max(lo, float(p['t0'])), min(hi, float(p['t1']))
        for ph in wanted:
            ks = [k for k, t in enumerate(ts) if a-1e-6 <= t <= b+1e-6 and phases[k] == ph and complete[k]]
            if len(ks) < FORMATION_MIN_POINTS:
                continue
            roles = _modal_roles(ctx.get('windows'), p.get('index'), ph) if ph in ('own', 'opp') else {}
            els = []
            for j, pid in enumerate(ids):
                xs = [tr['x'][j][k] for k in ks if tr['x'][j][k] is not None and tr['y'][j][k] is not None]
                ys = [tr['y'][j][k] for k in ks if tr['x'][j][k] is not None and tr['y'][j][k] is not None]
                if len(xs) < FORMATION_MIN_POINTS:
                    continue
                el = {'type': 'player', 'id': pid, 'team': obs['team'],
                      'x': round(min(max(sum(xs)/len(xs)/Lm, -S.NORM_MARGIN), 1+S.NORM_MARGIN), 4),
                      'y': round(min(max(sum(ys)/len(ys)/Wm, -S.NORM_MARGIN), 1+S.NORM_MARGIN), 4),
                      'kind': 'O', 'emphasis': True}
                if roles.get(pid) and len(roles[pid]) <= 8:
                    el['role'] = roles[pid]
                els.append(el)
            if len(els) < 3:
                continue
            t0, t1 = ts[ks[0]], ts[ks[-1]]
            phase_words = PHASE_WORDS[ph]+(f' {qual}' if ph in ('own', 'opp') else '')
            label = p.get('label') if p.get('label') and p.get('index', -1) >= 0 else None
            title = {'own': 'Average shape when we had the ball', 'opp': 'Average shape when the other team had the '
                                                                         'ball',
                     'unknown': 'Average shape when the whole team was in view'}[ph]
            caption = (f'Average positions from {L.fmt_time(t0)} to {L.fmt_time(t1)} {phase_words}'
                       + (f': the shape read as {label}' + (f' ({p["confidence"]})' if p.get('confidence') else '')
                          if label else '')+'. Described, not judged.')
            facts = _facts({'t': t0}, {'t': t1}, label or '')
            caption = _ok(caption, profile, facts, numbers=False, limit=CAPTION_REF_MAX) or \
                _ok(f'Average positions {phase_words}. Described, not judged.', profile, facts, numbers=False,
                    limit=CAPTION_REF_MAX)
            if not caption:
                continue
            diagram = {'preset': 'formation', 'title': title, 't': round(float(t0), 3),
                       'frame': int(round(float(t0)*fps)), 'team': obs['team'], 'attack': 'right',
                       'metresEstimated': scale.get('source') != 'measured' and scale.get('source') != 'declared',
                       'pitch': {'length': min(max(Lm, 15.0), 130.0), 'width': min(max(Wm, 10.0), 100.0)},
                       'elements': els}
            out.append((len(ks), {'id': _record_id('formation', obs['team'], p.get('index'), ph, t0, t1),
                                  'kind': 'formation', 'finding': None, 'evidence': None, 'caption': caption,
                                  'diagram': diagram}))
    out.sort(key=lambda x: -x[0])
    return sorted((d for _, d in out[:MAX_FORMATION_DIAGRAMS]), key=lambda d: d['diagram']['t'])


def movement_diagram(finding, ev, moment, obs, profile):
    """How our players moved in the REACH_H (2 s) before a finding's first key moment: an observed arrow from each
    own player's position then to his position at the moment (DESIGN 9.2 #5 "movement arrows", critic N7). Positions
    are the moment's scene (normalized pitch); None without the earlier scene."""
    before, scene = moment.get('before'), moment.get('scene')
    if not before or not scene or not ev.get('diagram'):
        return None
    scale = obs['timeline']['scale']
    Lm, Wm = float(scale['lengthM']), float(scale['widthM'])
    team = finding['team']
    els, arrows = [], []
    for pid, e in sorted(scene['players'].items()):
        el = {'type': 'player', 'id': pid, 'team': e['team'], 'x': e['x'], 'y': e['y'], 'kind': 'O',
              'emphasis': e['team'] == team}
        els.append(el)
        b = before['players'].get(pid)
        if e['team'] != team or not b or e.get('goalkeeper'):
            continue
        d = math.hypot((e['x']-b['x'])*Lm, (e['y']-b['y'])*Wm)
        if d >= MOVEMENT_MIN_M:
            arrows.append({'type': 'arrow', 'id': pid, 'points': [[round(b['x'], 4), round(b['y'], 4)],
                                                                   [round(e['x'], 4), round(e['y'], 4)]]})
    if not arrows:
        return None
    lvl = int(profile['level'])
    horizon = round(float(moment['t'])-float(before['t']), 1)
    t = float(moment['t'])
    caption = (f'How our players moved just before {L.fmt_time(t)}' if lvl == 1 else
               f'How our players moved in the {horizon:.1f} s before {L.fmt_time(t)}')+'. Observed positions only.'
    caption = _ok(caption, profile, _facts({'t': t}, horizon), numbers=False, limit=CAPTION_REF_MAX)
    if not caption:
        return None
    base = ev['diagram']
    diagram = {'preset': 'movement', 'title': 'How our players moved', 't': base['t'], 'frame': base['frame'],
               'team': base['team'], 'attack': base['attack'], 'metresEstimated': base['metresEstimated'],
               'pitch': dict(base['pitch']), 'elements': els+arrows}
    return {'id': _record_id('movement', finding['id'], ev['id']), 'kind': 'movement', 'finding': finding['id'],
            'evidence': ev['id'], 'caption': caption, 'diagram': diagram}


TEAM_LEVEL_WHY = {'wide_defender_advance': 'went forward', 'pressure_cover': 'nearest to the ball',
                  'weak_side_balance': 'on the far side', 'defensive_spacing': 'beside the gap'}


def team_level_rings(e, finding, profile):
    """A team-level challenge (levels 1-2, or a check the level judges team-level only: params.teamLevelOnly) never
    puts the review outline on one child (COACH L2: the level-1 "someone back" moments circled the advancing player
    with the review ring): its review-tone focus elements, in the diagram and in the video overlay, become the
    neutral support style (same player, position and box; DESIGN 5.4's support element, within its budget), so the
    picture shows the group and the empty space rather than a culprit. e is changed in place (a copy made by
    evidence.apply_captions); other findings' moments are left as they are."""
    if not finding or finding['verdict'] not in EX.FAILURES:
        return e
    reg = EX.register_for(profile, finding)
    if not (reg['level'] <= 2 or reg.get('teamLevel')):
        return e
    det = finding['observation']
    why = TEAM_LEVEL_WHY.get(det) or ('went forward' if EX.variant_of(finding) == 'advance' else 'one of the group')
    for spec in (e.get('diagram'), e.get('videoOverlay')):
        els = (spec or {}).get('elements')
        if not els:
            continue
        supports = sum(1 for x in els if x.get('type') == 'support')
        out = []
        for el in els:
            if el.get('type') == 'focus' and el.get('tone') == 'review':
                if supports >= S.ELEMENT_BUDGET['support']:
                    continue            # the budget is full: no outline at all rather than a review ring
                sup = {'type': 'support', 'id': el['id'], 'team': finding['team'], 'x': el['x'], 'y': el['y'],
                       'why': why}
                if 'box' in el:
                    sup['box'] = el['box']
                out.append(sup)
                supports += 1
            else:
                out.append(el)
        spec['elements'] = out
    return e


# ---------------------------------------------------------------------------------------------------------------
# 6 TRAINING DRILLS and 7 the NEXT DEVELOPMENT OBJECTIVE

def _drill_record(drill_id):
    try:
        from soccer.coaching import drills as D
        return D.get(D.load_library(), drill_id)
    except Exception:      # noqa: BLE001 - the library is optional here: without it no drill number is a fact
        return None


def _drill_facts(pick, finding, drill):
    extra = [c['facts'] for c in (finding or {}).get('claims') or ()]
    if finding is not None:
        extra += [finding.get('aggregate') or {}, finding.get('measurements') or {}]
    f = _facts(*extra)
    if drill:
        try:
            from soccer.coaching import drills as D
            for x in D.quoted_numbers(drill):
                f.add_number(x)
        except Exception:  # noqa: BLE001
            pass
        f.literals.add(drill.get('name') or '')
        f.literals.discard('')
    return f


def drill_problems(pick, finding, profile, key_ids, short=False):
    """Why a drill pick cannot be shown ([] = it can): its fields must pass explain.validate_free (facts from the
    finding's claims and the drill's own numbers; the level's number mode for why / progress, the match-derived
    fields; the quoted library lines easier / harder without the number checks, as drills.select writes them), and
    its moments must be key moments. drills.select already runs explain's gate; this is the report's own check of
    what it prints."""
    out = []
    drill = _drill_record(pick['drill'])
    facts = _drill_facts(pick, finding, drill)
    limits = {'trains': 300, 'why': 500, 'easier': 400, 'harder': 400, 'progress': 400}
    for field, lim in limits.items():
        text = (pick.get('fields') or {}).get(field)
        quoted = field in ('easier', 'harder')         # the library's own lines, with their setup numbers
        p = EX.validate_free(text, profile, facts, short=short, numbers=field in ('why', 'progress'), limit=lim,
                             check_facts=not quoted)
        out += [f'{field}: {x}' for x in p]
    for ev in (pick.get('fields') or {}).get('moments') or ():
        if ev not in key_ids:
            out.append(f'moments: {ev} is not a key moment')
    return out


def no_drill_notices(work, proposed, profile):
    """A notice per work item left without a drill (DRILLS' request): the library has none for its check at this
    level ('no_drill_at_level'), the safety filter refuses every one at this age and no lower level has an allowed one
    ('refused_by_safety', never relaxed), or the report could not print the one drills.select proposed."""
    out = []
    try:
        from soccer.coaching import drills as D
        lib = D.load_library()
    except Exception:      # noqa: BLE001 - without the library nothing can be said about it
        return out
    for f in work:
        if f['drills']:
            continue
        if f['id'] in proposed:
            why = 'not_printable'
        else:       # DRILLS' detail tells a format misfit from "nothing at this level" (fix round 1)
            detail = getattr(D, 'no_drill_detail', D.no_drill_reason)
            why = detail(lib, profile, f['observation'])
        if why not in NO_DRILL_WORDS:
            why = 'no_drill_at_level'
        text = NO_DRILL_WORDS[why].format(short=_lower_first(f['text']['short']))
        ok = _ok(text, profile, _facts(*[c['facts'] for c in f['claims']]), numbers=False, limit=400)
        if ok:
            out.append({'code': 'no_drill', 'text': ok})
    return out


def no_objective_text(profile):
    """The next objective's text when there is none (NO_OBJECTIVE_WORDS, by level band; validated)."""
    text = _at(NO_OBJECTIVE_WORDS, int(profile['level']))
    return _ok(text, profile, None, numbers=False, limit=400) or 'No next objective is given for this range.'


def _no_objective(profile):
    """The next objective when the match gives none and assess's suggested focus cannot be used (a withheld top work
    item with no other one reported; a focus text the gate refuses here): the level's general learning objective
    (assess.level_focus with checks=False: this stage cannot see what the match demonstrated, so it names no check),
    else no_objective_text."""
    from soccer.coaching import assess as AS
    focus = AS.level_focus(profile, checks=False)
    if focus:
        return {'skill': None, 'text': focus['text'], 'fromMatch': False, 'finding': None}
    return {'skill': None, 'text': no_objective_text(profile), 'fromMatch': False, 'finding': None}


def next_objective(assessment, kept, profile):
    """NEXT DEVELOPMENT OBJECTIVE (DESIGN 4.8 #7): the top reported work item's skill, with what the match showed;
    if explain withheld it, the next reported one; else assess's objective among the level's checks (said to come from
    the level and not from this match), or none: assess chooses one only from checks this range judged often enough
    and the team did not show (critic S-8, ASSESS fix round 2), and with none the text says what that means
    (no_objective_text: "no check of this level had both enough judged moments and room to improve"). Fix round 2:
    a withheld top work item with no other one reported gives no objective either (the earlier fallback chose the
    first unpraised check of the level, the S-8 rule assess no longer follows). Polish round: assess's "none" is now
    the level's suggested focus (assess.level_focus, kept when its text passes here); where this stage has to choose
    (the withheld top item), the level's general learning objective (_no_objective)."""
    from soccer.coaching import assess as AS
    no = dict(assessment.get('nextObjective') or {'skill': None, 'text': '', 'fromMatch': False, 'finding': None})
    by_id = {f['id']: f for f in kept}
    if AS.is_level_focus(no):
        # the level's suggested focus (polish round): assess chose it and checked its words; checked again here, as
        # every text this module prints
        literals = [g['text'].strip() for part in ('objectives', 'principles') for g in profile.get(part) or ()]
        literals += [s for part in ('objectives', 'principles') for g in profile.get(part) or ()
                     for s in g.get('sources') or ()]
        literals += [g['text'].strip()+'.' for part in ('objectives', 'principles') for g in profile.get(part) or ()]
        if not AS.focus_problems(no['text'], profile, literals):
            return {'skill': no.get('skill'), 'text': no['text'], 'fromMatch': False, 'finding': None}
        return _no_objective(profile)
    if no.get('fromMatch') and no.get('finding') not in by_id:
        work = sorted((f for f in kept if f['verdict'] in CRITICISM), key=lambda f: (f['priority'] or 99, f['key']))
        if work:
            f = work[0]
            no = {'skill': SKILL_OF[f['observation']], 'text': '', 'fromMatch': True, 'finding': f['id']}
        else:
            return _no_objective(profile)
    if no.get('skill') is None:
        return _no_objective(profile)
    if not no.get('fromMatch'):
        # assess's objective from this match's checks (S-8): never "not from this match" (fix round 3, FACTS F-9),
        # never the check a shown finding says the declared style accepts (COACH F-12)
        det = S.SKILLS.get(no['skill'])
        accepted = style_accepted(kept, det)
        if accepted:
            focus = _no_objective(profile)
            if focus['text'] != no_objective_text(profile):
                return focus
            text = STYLE_OBJECTIVE_WORDS.format(name=_lower_first(_check_name(profile, det)),
                                                style=EX.STYLE_WORD.get(accepted, accepted))
            return {'skill': None, 'text': _ok(text, profile, None, numbers=False, limit=400) or
                    no_objective_text(profile), 'fromMatch': False, 'finding': None}
        name = _cap_first(_check_name(profile, det)) if det else None
        text = no.get('text') or ''
        if name and (text.startswith(S8_OLD) or not text):
            no['text'] = _ok(f'{S8_LABEL}{name}.', profile, None, numbers=False, limit=400) or f'{name}.'[:400]
        return no
    if no.get('fromMatch') and no.get('finding') in by_id:
        f = by_id[no['finding']]
        # capitalised (COACH F-11: level 2's name "someone back when a defender goes forward" began the text)
        name = _cap_first(_check_name(profile, f['observation']))
        short = f['text']['short']
        t = f['timestamps'][0]['t'] if f['timestamps'] else None
        text = f'{name}. From this match: {short}' + (f' (first key moment {L.fmt_time(t)}).' if t is not None else
                                                       '.')
        facts = _facts(*[c['facts'] for c in f['claims']], {'t': t} if t is not None else {})
        no['text'] = _ok(text, profile, facts, short=EX.short_for(f, False), limit=400) or f'{name}.'[:400]
    return no


def _cap_first(s):
    return s[:1].upper()+s[1:] if s else s


def style_accepted(findings, detector):
    """The declared style (profiles style code) when a shown finding of `detector` states that the style accepts
    what it shows (its claims carry a style intent signal whose effect is one of explain.STYLE_ACCEPTS and its
    explanation says that effect), else None (fix round 3, COACH F-12)."""
    for f in findings or ():
        if f.get('observation') != detector:
            continue
        for c in f.get('claims') or ():
            fc = c.get('facts') or {}
            if c.get('type') == 'intent_signal' and fc.get('signal') == 'style' and fc.get('effect') in \
                    EX.STYLE_ACCEPTS and EX.STYLE_EFFECT[fc['effect']] in ((f.get('text') or {}).get('explanation')
                                                                        or ''):
                return fc.get('style')
    return None


def define_in_overview(report, profile, bm='well'):
    """The level's once-per-report definitions (explain.DEFINITIONS) are added to the overview synthesis when any text
    of the report uses a defined concept that no text of it defines (COACH L3: 77 level-6 reports said "rest
    defence" only in a headline, the next objective "Rest-defence margin (PitchIQ)" or a drill's "Also:" line and never
    said what it is). The overview is read first, so the concept is defined before it is used. bm: the behind-the-ball
    margin words of the run (explain.behind_words: "at least 10 m", fix round 3, FACTS F-1); a definition counts as
    given when its start (explain.DEFINITION_START) is in a text, whatever margin words a finding used."""
    lvl = int(profile['level'])
    ov = report['overview']
    texts = [t for f in (report['strengths']+report['needsWork']+report['reviews']+report['described'])
             for t in (f.get('text') or {}).values() if isinstance(t, str)]
    texts += [(report.get('nextObjective') or {}).get('text') or '', ov.get('text') or '']
    texts += [n['text'] for n in ov['notices']]
    texts += [v for p in report['drills'] for v in (p.get('fields') or {}).values() if isinstance(v, str)]
    joined = ' '.join(texts)
    for code, levels, rx, definition in EX.DEFINITIONS:
        text = EX.definition_text(definition, bm)
        if lvl in levels and rx.search(joined) and EX.DEFINITION_START[code] not in joined and ov.get('text') and \
                len(ov['text'])+1+len(text) <= 1500:
            ov['text'] = f'{ov["text"]} {text}'
            joined += ' '+text
    return report


def margin_words(obs, profile):
    """The run's behind-the-ball margin as the report's texts say it (explain.behind_words): "at least N m" with the
    level's rounded-metres register and usable metres (the scale's metresUsable, as assess's 'metres' qualifier), else
    "well"."""
    scale = ((obs or {}).get('timeline') or {}).get('scale') or {}
    return EX.behind_words(EX.behind_margin(obs), 'usable' if scale.get('metresUsable') else 'unusable',
                           EX.register_for(profile))


# ---------------------------------------------------------------------------------------------------------------
# the stage

def build_report_v2(*, obs, roles, profile, assessment, findings, evidence, alternatives, no_alternatives, drills,
                    withheld, progress, meta, range, log=None):
    """The 'report' stage (INTERFACES.md LEAD 4): schema 'report' of one team. findings: explain's (with text and
    claims; withheld ones already left out); evidence: annotate's records (one per cited moment); alternatives /
    no_alternatives: ZONES' records; drills: drills.select's picks; withheld: explain's; progress: the squad's
    Progression or None (merged at serve time); meta: {run, teamName, video {key, name}}; range: {t0, t1, seconds,
    whole}. engine / fingerprints / generated get stand-in values here and are overwritten by coaching.interpret.
    log: an optional list that receives what the report left out and why (drill picks, moments)."""
    meta = dict(meta or {})
    rng = dict(range)
    short = _num(rng.get('seconds')) and float(rng['seconds']) < EX.SHORT_RANGE_S
    withheld = [dict(w) for w in withheld or ()]
    ev_by_id = {e['id']: e for e in evidence or ()}
    nalts = {a['finding']: a for a in no_alternatives or () if isinstance(a, dict)}
    kept = []
    for f in findings:
        missing = [ts['evidence'] for ts in f['timestamps'] if ts['evidence'] not in ev_by_id]
        if missing:
            withheld.append({'finding': f['id'], 'reason': 'moment_missing'})
            if log is not None:
                log.append(('finding', f['id'], f'moments not made: {missing}'))
            continue
        kept.append(EX.with_no_alternative(f, nalts.get(f['id']), profile, short=short))
    ids = {f['id'] for f in kept}
    order = {f['id']: k for k, f in enumerate(kept)}
    sections = {'strengths': [f for f in kept if f['verdict'] == 'strength'],
                'needsWork': sorted((f for f in kept if f['verdict'] in CRITICISM),
                                    key=lambda f: (f['priority'] or 99, order[f['id']])),
                # the possible issues first, then the 'moments worth a look' (REVIEW_PARTS order, fix round 2)
                'reviews': sorted((f for f in kept if f['verdict'] == 'review'),
                                  key=lambda f: ([p['part'] for p in REVIEW_PARTS].index(review_part(f)),
                                                 order[f['id']])),
                'described': [f for f in kept if f['verdict'] == 'describe']}
    # 4 KEY MOMENTS: every cited moment, captions = the findings' validated short text
    cited = {}
    for f in kept:
        for ts in f['timestamps']:
            cited.setdefault(ts['evidence'], f['id'])
    key = []
    for e in EV.apply_captions(list(evidence or ()), kept):
        if e['id'] not in cited:
            if log is not None and e.get('finding') not in ids:
                log.append(('moment', e['id'], 'its finding was withheld'))
            continue
        if e.get('finding') not in ids:
            e['finding'] = cited[e['id']]
            if e.get('annotation'):
                e['annotation']['finding'] = cited[e['id']]
        key.append(e)
    key.sort(key=lambda e: (e['time'], e['id']))
    kept_by_id = {f['id']: f for f in kept}
    for e in key:
        team_level_rings(e, kept_by_id.get(e.get('finding')), profile)
    key_ids = {e['id'] for e in key}
    # 5 TACTICAL DIAGRAMS
    moments = {m['id']: m for m in list(obs.get('moments') or ())+list((roles or {}).get('moments') or ())}
    diagrams = []
    for e in key:
        if e.get('diagram'):
            diagrams.append({'id': _record_id('finding', e['id']), 'kind': 'finding', 'finding': e['finding'],
                             'evidence': e['id'], 'caption': e['caption'][:CAPTION_REF_MAX], 'diagram': e['diagram']})
    for f in kept:
        if f['verdict'] == 'describe' or not f['timestamps']:
            continue
        ev = next((e for e in key if e['id'] == f['timestamps'][0]['evidence']), None)
        m = moments.get(f['timestamps'][0]['evidence'])
        mv = movement_diagram(f, ev, m, obs, profile) if ev is not None and m is not None else None
        if mv:
            diagrams.append(mv)
    diagrams += formation_diagrams(obs, profile, rng)
    # 6 TRAINING DRILLS (7's objective first: an optional drill must train the objective's check, at most one)
    nobj = next_objective(assessment, kept, profile)
    want = S.SKILLS.get(nobj.get('skill')) if not nobj.get('fromMatch') else None
    from soccer.coaching import assess as AS
    focus_drill = AS.is_level_focus(nobj)
    by_id = {f['id']: f for f in kept}
    picks = []
    for pick in drills or ():
        f = by_id.get(pick.get('finding'))
        if pick.get('finding') is not None and f is None:
            if log is not None:
                log.append(('drill', pick.get('drill'), 'its finding was withheld'))
            continue
        if pick.get('finding') is None and not pick.get('optional'):
            if log is not None:
                log.append(('drill', pick.get('drill'), 'a drill without a finding must be optional'))
            continue
        if pick.get('finding') is None:
            # polish round: the optional drill is for the next objective shown, so it trains that check (drills.py
            # chose it from assess's objective; a different check, or a second one, is left out)
            trains = (_drill_record(pick['drill']) or {}).get('related_tactical_detector') or ()
            if want is None or want not in trains or any(p['finding'] is None for p in picks):
                if log is not None:
                    log.append(('drill', pick.get('drill'), 'not the one optional drill for the next objective'))
                continue
            # under 3 minutes of play no text says "repeat" / "pattern" / "often" (EVAL's no_repetition_wording):
            # the level-6 shadow drill's easier line "Repeat the shadow phase ..." on short_segment/default. Only
            # the suggested focus's drill is held to it here (the S-8 objective's drills are as fix round 2 left them)
            if short and focus_drill and any(SHORT_REPEAT.search(v) for v in (pick.get('fields') or {}).values()
                                          if isinstance(v, str)):
                if log is not None:
                    log.append(('drill', pick.get('drill'), 'repetition words in a range under 3 minutes'))
                continue
        p = drill_problems(pick, f, profile, key_ids, short=short)
        if p:
            if log is not None:
                log.append(('drill', pick.get('drill'), p))
            continue
        picks.append(copy.deepcopy(pick))
    for f in kept:
        f['drills'] = [p['drill'] for p in picks if p['finding'] == f['id']]
    proposed = {p.get('finding') for p in drills or ()}
    no_drill = no_drill_notices(sections['needsWork'], proposed, profile)
    # sources: the profile's, the principle claims', the drills'
    src_ids = list(profile.get('sources') or ())
    for f in kept:
        src_ids += [c['source'] for c in f['claims'] if c.get('source')]
    for p in picks:
        d = _drill_record(p['drill']) or {}
        src_ids += [r.get('source') for r in d.get('source_reference') or () if r.get('source')]
    from soccer.coaching import profiles as PR
    sources = PR.source_refs(src_ids)
    video = dict(obs.get('video') or {'name': None, 'key': None})
    for k in ('name', 'key'):
        v = (meta.get('video') or {}).get(k)
        if isinstance(v, str) and v:
            video[k] = v
    ifp = _stand_in_fingerprint(obs, roles, profile, rng)
    report = {
        'kind': 'coaching-report', 'version': REPORT_VERSION, 'team': obs['team'],
        'teamName': meta.get('teamName') if isinstance(meta.get('teamName'), str) and meta.get('teamName') else None,
        'generated': None,
        'engine': {'version': S.ENGINE_VERSION, 'measurement': obs['fingerprint'][:16],
                   'roles': roles['fingerprint'][:16], 'interpretation': ifp[:16]},
        'fingerprints': {'measurement': obs['fingerprint'], 'roles': roles['fingerprint'], 'interpretation': ifp},
        'run': meta.get('run') if isinstance(meta.get('run'), str) and meta.get('run') else None,
        'video': video, 'range': rng,
        'segment': {'t0': obs['segment']['t0'], 't1': obs['segment']['t1']},
        'profile': copy.deepcopy(profile),
        'overview': overview(obs, roles, profile, assessment, sections, rng, short, extra_notices=no_drill),
        'strengths': sections['strengths'], 'needsWork': sections['needsWork'], 'reviews': sections['reviews'],
        'described': sections['described'], 'keyMoments': key, 'diagrams': diagrams,
        'alternatives': [copy.deepcopy(a) for a in alternatives or () if a['finding'] in ids],
        'noAlternatives': [copy.deepcopy(a) for a in no_alternatives or () if a['finding'] in ids],
        'drills': picks, 'progression': copy.deepcopy(progress) if progress else None,
        'nextObjective': nobj,
        'sources': sources, 'withheld': withheld,
    }
    return define_in_overview(report, profile, margin_words(obs, profile))


def text_problems(report):
    """The fix-round-3 report-level text invariants (for EVAL's coaching_eval.report_problems and game-3 G1+; [] when
    the report holds them), each refusing the old false form:
    - FACTS F-9 = COACH F-4: an objective assess chose from this match's checks (a skill, not from a work item, not
      the level's suggested focus) never says "not from this match" and says it was chosen from this match's checks;
    - COACH F-12: the objective is never a check a shown finding says the declared style accepts;
    - COACH F-11: an objective from a work item starts with a capital;
    - LEAD decision (3) / COACH F-5: no finding text says "professional" for a youth team or below level 5;
    - pre-freeze (critic S-3, DRILLS / PAGE): an optional drill's label is true of the objective shown ("not from this
      match's evidence" never for an objective chosen from this match's checks, "chosen from this match's checks"
      only for one);
    - pre-freeze (COACH F-9): the objective quotes no general learning objective whose source's top age is below the
      team's age (assess.GENERAL_OBJECTIVE_TOP_AGE, fits_age)."""
    from soccer.coaching import assess as AS
    from soccer.coaching import drills as DR
    out = []
    prof = report.get('profile') or {}
    no = report.get('nextObjective') or {}
    text = str(no.get('text') or '')
    kept = [f for k in ('strengths', 'needsWork', 'reviews', 'described') for f in report.get(k) or ()]
    from_checks = bool(no.get('skill')) and not no.get('fromMatch') and not AS.is_level_focus(no)
    if from_checks:
        if 'not from this match' in text:
            out.append('nextObjective: "not from this match" on an objective chosen from this match\'s checks')
        if "this match's checks" not in text:
            out.append('nextObjective: an objective chosen from this match\'s checks does not say so')
        if style_accepted(kept, S.SKILLS.get(no['skill'])):
            out.append(f'nextObjective: {no["skill"]} is what a shown finding says the declared style accepts')
    for p in report.get('drills') or ():
        why = str(((p.get('fields') or {}).get('why')) or '') if p.get('optional') else ''
        if from_checks and DR.NEXT_OBJECTIVE_LABEL in why:
            out.append(f'drill {p.get("drill")}: "{DR.NEXT_OBJECTIVE_LABEL}" for an objective chosen from this '
                       'match\'s checks')
        if not from_checks and DR.MATCH_OBJECTIVE_LABEL in why:
            out.append(f'drill {p.get("drill")}: "{DR.MATCH_OBJECTIVE_LABEL}" for an objective not chosen from this '
                       'match\'s checks')
    m = AS.FOCUS_SOURCE_RE.search(text) if AS.is_level_focus(no) else None
    for sid in (m.group(1).split(', ') if m else ()):
        if sid in AS.GENERAL_OBJECTIVE_TOP_AGE and not AS.fits_age(('objectives', sid, ''), prof):
            out.append(f'nextObjective: the general objective of {sid} (up to '
                       f'{AS.GENERAL_OBJECTIVE_TOP_AGE[sid]}) for a team of {prof.get("ageGroup")}')
    if no.get('fromMatch') and text[:1].islower():
        out.append('nextObjective: starts in lower case')
    youth = prof.get('ageGroup') not in EX.ADULT_AGES or int(prof.get('level') or 0) < 5
    if youth:
        for f in kept:
            for k, v in (f.get('text') or {}).items():
                if isinstance(v, str) and EX.PROFESSIONAL_RE.search(v):
                    out.append(f'{f["key"]}.{k}: "professional" for a team of level {prof.get("level")} '
                               f'{prof.get("ageGroup")}')
    return out


# ---------------------------------------------------------------------------------------------------------------
# The v1 names (report_v1.py, deprecated): forwarded so the v1 report's callers keep working until wave 7

V1_MODULE = 'soccer.coaching.report_v1'


def __getattr__(name):
    """report.X for a v1 name (build_report, SECTIONS, MAX_PRIORITIES, to_json, ...): the attribute of report_v1."""
    if name.startswith('__'):
        raise AttributeError(name)
    v1 = importlib.import_module(V1_MODULE)
    try:
        return getattr(v1, name)
    except AttributeError:
        raise AttributeError(f'module {__name__!r} has no attribute {name!r}') from None


__all__ = ['build_report_v2', 'overview', 'formation_diagrams', 'movement_diagram', 'drill_problems',
           'no_drill_notices', 'next_objective', 'no_objective_text', 'review_part', 'reviews_by_part',
           'text_problems', 'style_accepted', 'margin_words', 'REPORT_VERSION', 'REVIEW_PARTS', 'NO_OBJECTIVE_WORDS',
           'S8_LABEL']
