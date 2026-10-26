"""Coaching explanations v2 (DESIGN 6.1-6.6): structured findings -> typed claims -> sentences in the register of the
selected playing level, checked by a deterministic gate before anything reaches the coach.

Why claims, registers and a gate instead of templates per detector (the v1 language.py):
- the v1 words were 555 fixed strings keyed by detector and classification, and the three wording levels changed only
  the vocabulary: the audit (coaching2/audit/AUDIT.md) measured identical findings, counts and drills at every level
  on all 8 team-runs. The spec (Part 12) asks for "measured state + context + relationships + profile + repeated
  evidence + alternatives -> finding", never "metric exceeds threshold -> canned response". Here a finding is first
  turned into claims (situation, actor_state, relation, team_state, opponent_state, recurrence, intent_signal,
  principle, counterfactual, hypothetical, action, drill_link, limitation: DESIGN 6.2) built only from its own
  measured fields, the alternative zone ZONES validated and the profile; the text may use nothing else;
- the claims are level-filtered (DESIGN 6.3): level 1 gets a team-level praise and one question with no numbers and
  no "cover"; level 2 introduces exactly "pressure" and "cover" (defined once per report); level 3 adds role words,
  rounded metres and "if the ball had been lost"; level 4 the channel and metres; level 5 the holding-midfielder and
  centre-back checks and shares; levels 6-7 rest defence (a PitchIQ definition, stated once), style and rates per
  10 minutes. So the same opportunity gives different claims, not only different words: on the same synthetic
  finding the wide-defender family has 2 relation/opponent/team claims at level 2, 3 at level 3, 6 at level 5 and 8
  at level 7 (pressure and cover 2 -> 4, far-side balance 1 -> 3, behind the ball 1 -> 3 at level 6);
- every sentence is checked (validate) against the claims of its finding, whatever wrote it: every number at its
  shown precision, time, player id, role word, formation and phase phrase must come from a claim (language.validate_text
  plus role and phase checks); relation polarity must match (critic N12: "covered" when the claim says not covered is
  refused); consequence words ("exposed", "risk") only inside a hypothetical sentence and from level 3, transition
  words from level 5; the level's concept gate, the global bans (blame, exaggeration, intent, "mistake", federation
  approval), the spec's generic sentences ("needs to improve spacing", "communicate better", "stay in formation"),
  unmeasured events (schema.event_problems: no turnover, no "lost the ball" outside a hypothetical) and, below 3
  minutes of range, the repeated-pattern words (language.SHORT_BANNED). The v1 report printed false sentences (critic
  B4); here a sentence that fails is never shown: the deterministic minimal plan is tried, then the finding is
  withheld with a reason;
- phase wording follows the possession binding rules (possession/v4/reports/docs.md rule 8): "when the other team had
  the ball (estimated)", because possession-4 names the team per frame .95-.96 right on held-out data but its change
  timing is not reliable (F1 .59-.69) and broadcast stoppages are mostly missed (about 16% of kept time was really
  dead on the fresh segment); a coach-marked phase (user decision Oct 21) reads "(coach-marked)", a finding built
  from both "(estimated, partly coach-marked)";
- variation is planned, not random: each claim type has several phrasings per register (758 phrasings, 49 principle
  paraphrases, 136 polarity phrases), chosen by a hash of the finding key and the claim type with the level as an
  offset, and a report never uses one phrasing twice for the same slot; after the report is written, any two
  explanations with a word-trigram Jaccard above .5 are re-planned (DESIGN 6.4). Measured on the synthetic findings
  of tests/test_coaching_explain.py: the worst pair inside a 12-finding report is .46, the worst pair of one
  finding's explanations at two levels .36, and the closest any wide-defender text comes to one of the spec's
  left-back examples (style references only, never copied) is .04;
- player naming follows the profile (DESIGN 4.9): never at level 1, relative words at level 2 ("our left defender"),
  role words at level 3, ids only inside the explanation at level 4, ids from level 5; headlines stay id-free at every
  level (display labels change with corrections; ids are rendered as labels at view time, language.resolve_players);
  a failed identity guardrail removes names and individual role words; a role word needs a coach, stable or carried
  role (DESIGN 3.4);
- an optional local model (TextEngine-like: render(request) -> text) receives only the claims and the register; its
  text must pass the same gate or the deterministic text is used (DESIGN 6.6, spec Part 12). No model ships.

Interfaces (INTERFACES.md, TEXT section): explain(findings, profile, obs=, roles=, evidence=, alternatives=, range=,
engine=) is the 'explain' stage; validate(text, claims, finding, profile, short=) is the gate; build_claims is public
so DETECT and EVAL can inspect what the text may say. The measurement keys explain reads per detector are listed in
MEASUREMENT_KEYS (DETECT fills them in Finding.measurements; a missing key drops its sentence, it is never defaulted).
text.short is at most 90 characters and passes the gate, so EVIDENCE and the report can use it as the annotation
caption (schema CAPTION_MAX).

The gate is a set of named checks (FIELD_CHECKS, TEXT_CHECKS); a test disables each in turn and shows it is the one
that refuses its own bad sentence, so no check is dead or shadowed. All 348 detector x verdict x level x child/adult
cases of the tests pass with the planned text (none needs the minimal plan).

Pure standard library plus language.py and schema.py (no numpy): a 12-finding report is explained in about 21 ms
(median of 35 runs, CPU, Oct 22), inside DESIGN's 300 ms for a whole interpretation-tier rebuild.

Wave 5, on REAL findings (Oct 22; DETECT's assess, EVIDENCE's moments and ZONES' zones on EVAL's 30 scene variants
and the five run copies, both teams, engine and coach-marked phases, levels 1-7: 592 findings; scratch
coaching2/build/text_w5). The wave-1 plans withheld 6 of them and printed sentences the hand-made findings never
produced: 4 attacking-shape texts named the width or an empty side when the depth had failed (BC180 Barcelona at
levels 1-2: depth share .25 with width .88 and three channels), 21 "0 of our players", 5 "a margin of +0", 60 "on
average" for a value at one key moment, 19 descriptions saying "it stays a review", 4 level-2 team-level texts naming
"our left defender". Now each relation's polarity comes from its measurement under the level's params
(shape_relations; the weak-side and behind-the-ball counts; the level 6-7 margin rule), a team-level check
(params.teamLevelOnly) names nobody (register_for(profile, finding)), a count of 0 reads "none", descriptions say "at
the moment shown" and "a description only", the limitation codes one_instance / level_cap / no_moment get their own
sentences, the distance between the lines (D5 dm_gap) has its own principle (compact_lines), question and statement,
every case has a metre-free phrasing (a small format without a declared field has no usable metres), "the centre-back
line" is not a role word and "left" before a side noun is not a past-tense verb. Result: 592 of 592 worded with the
planned text, 0 withheld, none of those defects left. Two inputs coaching.interpret does not pass are recovered: the
short range from assess's 'short_range' limitation (short_for) and ZONES' no-alternative reason, which report.py adds
through with_no_alternative. validate_free is the same gate for the report's own texts (overview, notices, captions,
the next objective, drill fields).

Fix round 1 (Oct 23; the critic's B-3, B-4, S-6, S-9, S-10 and the coach's / facts verifier's wording items; scratch
coaching2/fix1/text: gen.py runs the 36 scene variants x 22 profiles and the 7 run variants x 2 teams x 14 profiles
with TEXT's three files as they were (before/) and as they are, check.py counts; every other module is the round's
current code, so the differences are TEXT's alone):
- the presser is the first defender AT the moment linked (measurements.fdId, presser_of), the cover never him
  (cover_of), and a gate check ('actors') refuses an id in a presser phrase that is not the presser or the presser in
  a cover phrase (critic B-3; with DETECT's assess change 0 of 78 scene texts name a wrong presser, the check is the
  invariant for an engine or a later edit; invariant_problems gives it, the phase and the format rules to EVAL);
- a finding whose phase named no team says nothing of who had the ball in its headline or short label (B-4): 32 + 28
  phase-unknown findings were headed "Attacking shape" / "Team shape with the ball", now 0; the gate ('phase') refuses
  "with the ball", "attack..." there;
- descriptions read a different relation at every level (S-6): level 2 the lanes used, 3 rounded metres, 4 metres and
  channels, 5 shares, 6 the length used up to the opponents' last line (critic B-7's depthAvailShare), 7 the highest
  player against that line and the support behind the ball; the far-side and spacing descriptions likewise. Level
  pairs of one family at word-trigram Jaccard >= .5: 234 scene + 73 run-copy pairs before, 0 after;
- levels 5-7 of the wide-defender check state every relation they measured (S-10): the near centre-back's shift, the
  holding midfielder's drop, the rest-defence margin (defined once per report, also when only a headline, the next
  objective or a drill line names it) and the six-to-centre-back spacing at level 7 (the COACH verifier's
  claims_realised.py on its 11 profiles: level 6 centre-back 10 of 32 -> 32 of 32, holding-midfielder drop 0 of 8 ->
  8 of 8, behind the ball 6 of 28 -> 28 of 28; level 7 spacing 0 of 8 -> 8 of 8);
- below 11v11 no 11v11 role name (S-9): SMALL_ROLE_WORD for confirmed roles, the SMALL_WORDS lexicon over every field,
  the 'format_words' check (20 of 112 small-format scene reports used one before, 0 after);
- wording facts: far-side counts in the tolerant goal-side words of critic B-5 and the whole far half said as such;
  behind-the-ball verdicts placed in the final third they are judged in; shares of 0 / 1 and rates from under 5 minutes
  of the phase not given as estimates; forward runs not called attacks; levels 1-2 never explain the zone engine; the
  level-1 "teammate close behind" sentence only with that claim (18 level-1 strengths had fallen back to one
  sentence); one player covering said once; mixed_evidence and formation_changing worded; the judged seconds behind a
  strength stated from level 4.

Fix round 2 (Oct 23; LEAD decision (b), ASSESS's request; scratch coaching2/fix2/text: mw_gen.py interprets ASSESS's
cached Metrica game-1 tiers (15- and 3-minute clips, full tracking and simulated camera, and the planted clips of the
game-3 plan's G12 and of fix round 1) at levels 3-7 with TEXT's files as they were (before/) and as they are, 8829
reports each; mw_compare.py counts): a 'moments worth a look' review (assess limitation MOMENTS: repeated CLEAR
failures below the level's verdict bar) has its own words. Round 1 worded it as any review: "Possible issue to
review: ..." (290 of the 290 such reviews), the count over all opportunities ("3 of 12 forward runs" with 1 unclear:
272 of 290; 2 stated the clear of the judged ones), "Not confirmed: enough evidence for a work item" (290). Now 290 of
290 read "Moments worth a look: <a mode-free label>", give the clear moments of the opportunities PitchIQ could judge
("In 3 of the 11 forward runs PitchIQ could judge, ...", "all 4" when every judged one was clear; the share from
level 5), the unclear ones, that the check was met in the rest and that this stays below the level's bar for a
verdict (true of every such review: assess gives one only when the failing share did not reach minFailureShare), every
cited moment's time (and how many of the clear ones those are when fewer: 99 of the 290 cite 3 of 4 or more), the
first moment described by its own measurements, why the check matters (the level's
principle) and, as the action, what to look for in the clips ("In each clip, ..."); no zone, no question, no drill,
no verdict wording (0 of 290; VERDICT_WORDS_RE). The count, label and caption of a pressure-and-cover review hold for
both failure modes (the cited moments of one review mixed "nobody near the carrier" and "a presser without cover" on
ASSESS's whole-match list, F/A/whole 1153 / 3163 / 4603 s); its first moment is described by its own flags, not the
modal mode (it is a guard: in none of the 180 pressure-and-cover criticisms and reviews of the scenes, the run copies
and these moments reviews did the two differ). moments_problems
(in invariant_problems, so in coaching_eval.report_problems and game-3 G1+) refuses the round-1 text: 0 on the 290.
Verdict lists identical in 8829 of 8829, withheld 12 = 12 (pre-existing: three D3 strengths and one D2 description of
the planted and Metrica tiers, polarity); the first draft withheld 2 (a level-5 action said "the holding midfielder"
without a confirmed role) and read alike across levels 6-7 (.674 on a scene) and 4-5 (.488): each level now has its
own count, frame, moments and first-moment phrasings, worst level pair of one moments family .346 (.391 before). The
rest-defence definition no longer names a federation ("not a federation definition" tripped EVAL's federation-name
scan: 84 run-copy overview texts on the fix-1 code; with this round's other code 138 strings in 624 scene reports and
29 Metrica overview texts before, 0 after; no phrasing of this module matches the scan, tested). Interpretation of a
15-minute team-clip 39.1-39.2 ms median after, 38.9-39.7 before (three runs of 84 warm coaching.interpret calls,
every stage).

Polish round (Oct 23; fix2/text.md's withheld findings): a wide defender's strength says who covered at its moment,
by the level's rule (d3_cover_by: a teammate in the channel, the near centre-back's shift, a holding midfielder's drop
where the level counts it, an interchange said right after the advance), and its label and headline name the side
("Supported advance on the left"; the plain label only when the side is unknown). Before, every level 4-7 strength
said "{mate} was in the channel behind them" and the gate withheld those whose channel was empty while the
centre-back covered. On ASSESS's cached Metrica game-1 and planted tiers (8829 reports, levels 3-7; scratch
coaching2/polish): withheld findings 12 -> 1 (the D2 far-side description the gate cannot read stays), wide-defender
strengths 1451 -> 1462, the 13 covered by the centre-back (2 before, at level 3, said "a teammate covered") now name it
in their first sentence, the 1449 covered in the channel keep their sentence; 1462 of 1462 labels name the side;
no other finding's text changed (the strength still takes the plain label's phrasing slot, so a failure of the same
check keeps the label it had).

Fix round 3 (Oct 24; critic B-3 and the TEXT items of its fix list: FACTS F-1, F-2, F-4 to F-8, F-11, COACH F-3, F-10,
F-11; ASSESS's requests (a)-(d); LEAD decision (3) in words). Measured with TEXT's three files before / after this round
on identical inputs (scratch coaching2/fix3/text: gen.py, check.py, compare.py; every other module the frozen start of
the round), on the COACH verifier's 546 scene reports, the 196 run copies, the user's 9 runs (252 reports, read in
memory), and Metrica game 1 (ASSESS's fix-3 tiers: 2072 clip reports, 252 whole-match and half reports):
- F-1 = COACH F-2: a count of ours behind the ball (and of theirs on our side of it) carries the margin it was
  measured with, "at least 10 m" (10 m x L / 105, behind_words; "well" below level 3 or without usable metres): counts
  without it 285 / 18 / 14 / 695 / 360 -> 0 (scenes / runs / user runs / clips / whole); the level-7 shape clause
  "none of ours behind the ball" with the ball at our own back line (deepN 0) 24 / 15 / 6 / 96 / 6 -> 0; the
  rest-defence definition states the margin for both counts. The COACH verifier's none_behind2.py: 108 literally
  false texts -> 0 (its 6 matches now read "At least 10 m behind the ball at that moment: none of our outfield
  players"); the critic's spot_behind2.py 5 -> 0; FACTS's real_facts.py, its parser extended to the new words: 21
  unqualified counts differing from the plain goal-side count -> 0, and the 6 counts stated with the margin equal its
  own >= 10 m count. "At least N m" is the rule's threshold, not an estimate: it carries no "about" (_check_numbers).
- F-2: the actor and the gap pair are the first moment's own (measurements.actorId, gapLeftId / gapRightId:
  moment_actor, gap_partner); 0 wrong names before and after on these sets (ASSESS's players change already fixed the
  data), now an invariant (fix3_problems) rather than an accident of players.primary.
- F-5: the place is the first moment's third (momentZone): claims placed in another third 188 / 0 / 14 / 676 / 193
  -> 0; FACTS's metrica_extra.py zone words more than 3 m from the raw first moment 34 -> 0 (its 33 others are the
  level-2 principle "someone stays back near our goal", 14 within its 3 m hysteresis margin).
- F-6 = COACH F-3: a strength is counted over the moments PitchIQ could judge, the unclear ones said ("all 11 judged
  defending moments; 18 more could not be judged", not "11 of 29 pressing moments"): over all opportunities 284 / 10 /
  15 / 736 / 732 -> 0; COACH's strength_share_text.py 136 -> 0; FACTS's pressure-and-cover denominators 14 -> 0; the
  pressure-and-cover nouns are "defending moments" from level 2.
- F-4 = COACH F-10: a 'moments worth a look' review counts the judged moments that fell short less clearly
  (moments_more), so clear + less clear + met = judged; unaccounted 4 -> 0 of the scenes' 28 reviews (Metrica game 1
  has none since ASSESS's fix-3 trigger).
- F-7: "the nearest teammate within the cover angle" (coverId is the nearest teammate inside the 60-degree cone):
  unqualified 12 / 0 / 0 / 3 / 0 -> 0; an unnamed cover or ZONES' mover is "a teammate".
- F-8: "a teammate took their place" only with the geometric interchange at the moment (interchangeCover), never the
  10 s role-label swap: without it 0 / 0 / 0 / 5 / 13 -> 0 (FACTS's metrica_extra.py 7 -> 0).
- F-11: no "used the whole pitch" (35 / 5 / 2 / 124 / 28 -> 0); the opponents between the lines on the ball's side
  (0 / 0 / 0 / 2 / 16 -> 0). COACH F-11: level 1 counts in words, "a few / several / many times" (a few for 10 or more:
  30 / 0 / 0 / 28 / 64 -> 0); a description no longer states the possession style's weighting of strengths
  (76 / 18 / 12 / 276 / 36 -> 0).
- LEAD decision (3): a mixed-evidence description says the failures were more than the level's base explains; youth
  teams and levels 1-4 never read "professional" (2 / 2 / 8 -> 0 on runs / clips / whole; a gate check).
- ASSESS (a) / (b): level 1-2 words for an advance left exposed (one of theirs in the space, nobody of ours in it) and
  for a covered one with nobody behind the ball; findings withheld by the gate 26 / 2 / 2 / 24 / 0 -> 0, the minimal
  plan 0 -> 0 everywhere.
Verdicts unchanged (assess's findings identical in every report; only previously withheld ones are now shown), the most
alike pair of explanations in one report .145-.346 per set (.156-.350 before), one family's texts at two levels .453 at
worst (.444; 0 of 1499 pairs above .5). coaching.interpret on the six 15-minute full-tracking clips: 52.2 / 51.9 ms median
after, 52.0 / 51.9 before (two runs of 120 warm calls).

Pre-freeze (Oct 24, freeze gate G-2; scratch coaching2/closer): the left and right wide-defender strengths of one
3-minute range (Metrica game 1, F, team B, 1080.04-1260.04 s, level 5 at U16 and U18) stayed .5119 alike after every
salted re-plan of _spread: at levels 5-7 both state every relation, so they repeated the principle, the holding
midfielder's sentence, the counts and the judged seconds word for word. _drop_repeats now plans such a later
explanation once more without the optional sentences (REPEAT_SLOTS, the principle first) an earlier explanation of the
report already says word for word, only as far as needed and never into the minimal plan. On the gate's 2128 game-1
reports (F and C0, whole match, six 15-minute windows, 31 three-minute ranges, both teams, 14 profiles): criterion-13
problems 2 -> 0, the worst pair .5119 -> .4151 (the right side's text without the principle, which the left side
states); every other report's finding explanations identical, every verdict list identical. Tests:
PreFreezeRepeatTests (the pair rebuilt from that report's measurements is .5253 alike with the gate's code).
"""
import copy
import hashlib
import math
import re

from . import language as L
from . import schema as S

ENGINE_NAME = 'claims'
TEXT_FIELDS = ('headline', 'explanation', 'short', 'action')
LIMITS = {'headline': 160, 'explanation': 1500, 'short': 90, 'action': 500}   # short doubles as annotation caption
SHORT_RANGE_S = L.SHORT_SEGMENT_SECONDS    # 180 s: below it no repeated-pattern words (DESIGN 3.6)
JACCARD_MAX = .5                           # DESIGN 6.4: two explanations of one report must stay below this
RETRIES = 4                                # re-plans with shifted variants when two explanations are too alike
NUMBER_MODES = ('none', 'counts', 'rounded_metres', 'metres', 'shares', 'rates')
ADULT_AGES = ('U19', 'U21', 'U23', 'Adult')   # adult register for levels 1-3 ("Next step:", no child phrasing)
CONFIDENT_SOURCES = ('coach', 'stable', 'carried')   # role sources that may be named with role words (DESIGN 3.4)
NAMING = ('never', 'relative', 'role_words', 'evidence_only', 'yes')
CRITICISM = ('try_this', 'work_on')
FAILURES = ('try_this', 'work_on', 'review')
PER_MINUTES = 10                           # rates are per 10 minutes of the relevant phase (DESIGN 3.6)
CHANNELS_TOTAL = 3                         # lateral channels (D6)
# Rates are extrapolations: "about 56.9 times per 10 minutes" from 2.3 minutes of the phase was false precision
# (COACH L5). A rate is stated only from RATE_MIN_PHASE_S of the relevant phase (at most a twofold extrapolation), as
# a whole number below RATE_DECIMAL_PHASE_S; otherwise the counts are given over the phase time [PITCHIQ-PROPOSAL].
RATE_MIN_PHASE_S = 300.0
RATE_DECIMAL_PHASE_S = 600.0
# Shares at the ends are counts, not estimates: "about 100%" of 16 advances is "16 of 16" (COACH L5).
SHARE_ALL = .995
# Sentences the plan may drop, last first, when an explanation would pass the schema's 1500 characters (levels 6-7
# now state up to four more relations of the wide-defender check: critic S-10).
OPTIONAL_SLOTS = ('style', 'interchange', 'judged', 'opp_channel', 'behind_ctx', 'hypo', 'principle')
# Pre-freeze (Oct 24, freeze gate G-2): when the salted re-plans of _spread leave two explanations of one report at or
# above JACCARD_MAX, the later one is planned once more without the optional sentences an earlier explanation of the
# report already says word for word, dropped in this order and only as far as needed (the check's own principle
# first: two side-specific wide-defender strengths of one report repeat it by construction). On Metrica game 1, F,
# team B, 1080.04-1260.04 s at level 5 (U16 and U18) the left / right strengths were .5119 alike after every salt.
REPEAT_SLOTS = ('principle', 'behind_ctx', 'opp_channel', 'hypo', 'style', 'interchange', 'judged')
# Fix round 3 (FACTS F-1 = COACH F-2): "behind the ball" is measured as at least BEHIND_MARGIN_M x L/105 goal-side of
# the ball (tactics_geom.BEHIND_BALL_M / REF_LENGTH, a test keeps them equal; this module stays numpy-free), and the
# opponents "on our side of it" likewise. The texts said "none of ours behind the ball" with 1-4 outfielders 0-8 m
# goal-side of it (95 of 95 checkable texts, 5 of 5 level-7 run-copy describes), so every count now carries the
# margin: "at least 10 m behind the ball" where the level shows metres, "well behind the ball" below level 3 or
# without usable metres (behind_words).
BEHIND_MARGIN_M = 10.0
BEHIND_REF_LENGTH = 105.0
BEHIND_QUALIFIED_RE = re.compile(r'(?:\bwell|\bat least (?:about )?\d+ m) (?:behind the ball|on our side)', re.I)
# a count of ours (or of theirs on our side) at the ball's depth: the sentences the margin rule reads
BEHIND_COUNT_RE = re.compile(r"(?:\b(?:none|only \d+|\d+|one|two|three|four|five|six|seven|eight|nine|ten) of "
                             r"(?:ours|our (?:outfield )?players?|theirs|their (?:outfield )?players?)\b|"
                             r"\bBehind the ball (?:we had|at that moment)|\b(?:enough|fewer) (?:of our )?(?:outfield )?"
                             r"players\b|\bwe had \w+ (?:outfield )?players?)", re.I)

# Registers per level (DESIGN 6.3 [P]): the defaults when the profile's register block leaves a field out. The
# profile's register (numbers, maxChallenges, tone) and evidence.naming are the configurable data (profiles.json); the
# concept gate and the bans below are this module's (DESIGN 6.3 / 6.5) and always apply.
REGISTERS = {
    1: {'tone': 'praise first; one question; short sentences', 'numbers': 'none', 'maxChallenges': 1,
        'naming': 'never'},
    2: {'tone': 'praise first; up to two challenges as questions', 'numbers': 'counts', 'maxChallenges': 2,
        'naming': 'relative'},
    3: {'tone': 'strengths first; questions; hypotheticals', 'numbers': 'rounded_metres', 'maxChallenges': 2,
        'naming': 'role_words'},
    4: {'tone': 'neutral, specific', 'numbers': 'metres', 'maxChallenges': 3, 'naming': 'evidence_only'},
    5: {'tone': 'analytical', 'numbers': 'shares', 'maxChallenges': 3, 'naming': 'yes'},
    6: {'tone': 'tactical', 'numbers': 'rates', 'maxChallenges': 3, 'naming': 'yes'},
    7: {'tone': 'analyst', 'numbers': 'rates', 'maxChallenges': 3, 'naming': 'yes'},
}

# What explain reads from Finding.measurements per detector (DETECT fills them; scalars only). Values without a
# suffix are at the finding's first key moment (timestamps[0]); *Share over the opportunities; counts are integers.
# 'metres' ('measured' | 'declared' | 'estimated' | 'unusable') qualifies every metre value of the finding.
MEASUREMENT_KEYS = {
    # fdId: the presser AT the first key moment (fix round 1, critic B-3: the text named the family's modal first
    # defender while the distance and the cover were the moment's); coverId: the moment's nearest teammate
    'pressure_cover': ('failureMode', 'pressurePresent', 'coverPresent', 'd1M', 'coverDistM', 'coverAngleDeg',
                       'coverPresentShare', 'goalSideShare', 'fdId', 'coverId', 'metres', 'judgedSeconds'),
    # weakCount / centralCount: goal-side with the level tolerance (critic B-5); weakLevelCount: far-half defenders
    # level with the ball; weakRecovering: far-half defenders running back; ballX: attack-frame ball depth
    'weak_side_balance': ('weakCount', 'centralCount', 'openWidthM', 'weakLevelCount', 'weakRecovering', 'ballX',
                          'metres', 'judgedSeconds'),
    'wide_defender_advance': ('side', 'advanceM', 'behindN', 'channelOccupied', 'channelBy', 'cbAdjusted',
                              'nearCbShiftM', 'dmDropped', 'interchange', 'opponentsInChannel', 'threatN',
                              'restMargin', 'dmCbSpacingM', 'metres', 'judgedSeconds'),
    'behind_ball': ('behindN', 'minBehind', 'threatN', 'restMargin', 'metres', 'judgedSeconds'),
    'defensive_spacing': ('failureMode', 'lateralGapM', 'gapBallSide', 'opponentInGap', 'dmGapM', 'lineStdM',
                          'blockHeight', 'dmGapOpponents', 'metres', 'judgedSeconds'),
    # depthAvailShare: our length / the length from our deepest player to their last line (critic B-7);
    # availableDepthM, heightGapM (our highest player to their last line), deepN (ours 10 m behind the ball)
    'attacking_shape': ('failureMode', 'widthShare', 'depthShare', 'widthM', 'depthM', 'channelsOccupied',
                        'depthAvailShare', 'availableDepthM', 'heightGapM', 'deepN', 'metres', 'judgedSeconds'),
}
# Fix round 3 (ASSESS's _moment_actors, FACTS F-2 / F-5 / F-8): the first cited moment's own players and place, read
# before the family's modal ones: momentZone (every detector), actorId (D3, D5), gapLeftId / gapRightId (D5),
# interchangeCover (D3: the slot re-occupied, geometric), gapOppFreeM (D5); failureMode of D3 ('exposed',
# 'uncovered', 'nobody_behind') and of level 1's reframed advance (behind_ball: 'exposed', 'nobody_behind', with
# side, advanceM, channelOccupied and opponentsInChannel of the advance); baseShare / baseTail (the base a
# mixed-evidence description or a 'moments worth a look' review was tested against).
MOMENT_KEYS = ('momentZone', 'actorId', 'gapLeftId', 'gapRightId', 'interchangeCover', 'gapOppFreeM', 'baseShare',
               'baseTail')
FAILURE_MODES = {'pressure_cover': ('no_cover', 'no_pressure'), 'defensive_spacing': ('lateral_gap', 'dm_gap'),
                 'attacking_shape': ('width', 'depth', 'shape', 'channels')}
# Fix round 3 (ASSESS, critic B-1): the advance's failure modes, read from measurements.failureMode only (never
# inferred): 'exposed' (one of theirs in the space the defender left), 'uncovered' (levels 3-7) and 'nobody_behind'
# (levels 1-2: nobody behind the ball in the final third). Levels 1-2 word 'exposed' apart: the "nobody stayed back"
# wording is false there with players behind the ball (one_behind, channel_threat, stacking: withheld before).
ADVANCE_MODES = {'wide_defender_advance': ('exposed', 'uncovered', 'nobody_behind'),
                 'behind_ball': ('exposed', 'nobody_behind')}
# attacking shape at its key moment, judged as contextual.sample_outcomes judges it (widthShare / depthShare of the
# level's params, all three lateral channels): on Broadcast 42-222 s Barcelona's level 1-2 try_this failed on DEPTH
# (share .25 < .30) with width .88 and three channels, and the wave-1 text said "we stayed narrow" (a false sentence);
# the text now names the relation that failed (mode 'shape' = width and depth both).
SHAPE_DEFAULTS = {'widthShare': .55, 'depthShare': .30}

# The relations each detector's text may assert (polarity is checked only for these kinds).
DETECTOR_KINDS = {
    'pressure_cover': ('pressure', 'cover'),
    'weak_side_balance': ('weak_side',),
    'wide_defender_advance': ('advance_support', 'channel_cover', 'cb_shift', 'dm_drop', 'behind_ball'),
    'behind_ball': ('behind_ball',),
    'defensive_spacing': ('lateral_gap', 'unit_gap', 'line_evenness'),
    'attacking_shape': ('width', 'depth', 'channels'),
}

# ---------------------------------------------------------------------------------------------------------------
# Vocabulary gates (DESIGN 6.3, 6.5)

# Concepts and the level that unlocks them (cumulative). 'balance' is unlocked at level 4, not 5 as in DESIGN 6.3's
# concept column: DESIGN's own level 3-4 ban lists leave it out and the spec's INTERMEDIATE example uses it.
LOSS_HYPO = r'\b(?:the ball|possession) had (?:been )?(?:lost|changed hands|changed)\b'
CONCEPTS = (
    ('pressure', r'\bpressur\w*|\bpress(?:es|ed|ing|er|ers)?\b', 2),
    ('cover', r'\b(?:un)?cover(?:s|ed|ing)?\b', 2),
    ('support', r'\b(?:un)?support(?:s|ed|ing)?\b', 3),
    ('recover', r'\brecover\w*', 3),
    ('width', r'\bwidth\b', 3),
    ('depth', r'\bdepth\b', 3),
    ('ball lost (hypothetical)', LOSS_HYPO, 3),
    ('positional responsibility', r'\bpositional responsibilit\w*', 4),
    ('defensive line', r'\b(?:defensive|back) line\b', 4),
    ('compact', r'\bcompact\w*', 4),
    ('channel', r'\bchannels?\b', 4),
    ('unprotected space', r'\bunprotected\b', 4),
    ('balance', r'\b(?:im|un)?balanc\w*', 4),
    ('holding midfielder', r'\bholding midfielders?\b', 5),
    ('adjustment', r'\badjust\w*', 5),
    ('rotation', r'\brotat\w*', 5),
    ('transition', r'\btransitions?\b', 5),
    ('rest defence', r'\brest[- ]defen[cs]e\b', 6),
    ('block height', r'\b(?:low|mid|high)[- ]block\b', 6),
    ('style', r'\bstyles?\b', 6),
    ('the six', r'\bthe six\b', 7),
    ('route', r'\broutes?\b', 7),
    ('game model', r'\bgame[- ]model\b|\bgame plan\b', 7),
    ('trigger', r'\btriggers?\b', 8),          # pressing triggers are not measured (DESIGN 4.7): never
)
CONCEPT_RE = tuple((name, re.compile(rx, re.I), lvl) for name, rx, lvl in CONCEPTS)
GLOBAL_BANS = tuple(L.TONE_BANNED)+('uefa', 'certified', 'certification', 'endorsed', 'endorses', 'accredited',
                                    'approved by')
GENERIC_BANS = tuple(re.compile(rx, re.I) for rx in (
    r'\bneeds? to improve\b', r'\bimprove (?:the |their |our |its )?spacing\b', r'\bbetter spacing\b',
    r'\bstay(?:ed|s|ing)? in (?:formation|position|shape)\b', r'\bkeep (?:their|our|its) (?:formation|positions?)\b',
    r'\bwork harder\b', r'\bmore organi[sz]ed\b', r'\bconcentrat\w*', r'\bfocus more\b', r'\bbe more aware\b',
    r'\btalk more\b', r'\bcommunicat\w*'))
HYPOTHETICAL = S.HYPOTHETICAL               # if / could / would / might / potentially / possible / possibly
POLARITY_HYPO = re.compile(r'\b(?:if|could|would|might|potentially|possible|possibly|whether|can|try this|next '
                           r'step)\b', re.I)
TRANSITION = re.compile(r'\btransitions?\b|\bcounter[- ]?press\w*|\bcounter[- ]?attack\w*', re.I)
CONSEQUENCE = re.compile(r'\bexpos\w*|\bvulnerab\w*|\brisk\w*|\bdanger\w*|\bthreat\w*|\bin behind\b|\bbypass\w*',
                         re.I)
LOSS_RE = re.compile(LOSS_HYPO, re.I)
# Observed (past-tense) sentences. "left" is also a side ("the left channel" in a general statement made a level 5
# action unreadable on a real finding), so it counts only when no side noun follows it.
PAST = re.compile(r'\b(?:was|were|had|did|stayed|went|moved|kept|got|came|stood|held|dropped|shifted|spread)\b'
                  r'|\bleft\b(?!\s+(?:channel|back|side|centre|wing|winger|defender|midfielder|forward|touchline|half'
                  r'|third|flank|corridor|lane|holding|central))'
                  r'|\b[a-z]{3,}ed\b', re.I)
PHASE_RE = re.compile(r"\b(?:when|while) (we|the other team) had the ball\b(\s*\([^)]*\))?", re.I)
POSS_RE = re.compile(r"\b(our|the other team's) possession\b(\s*\([^)]*\))?", re.I)
PHASE_LOOSE = re.compile(r"\b(?:when|while) (?:we|they|the other team) (?:were )?"
                         r"(?:attack(?:ed|ing)|defend(?:ed|ing))\b"
                         r"|\b(?:in|out of) possession\b|\b(?:when|while) they had the ball\b", re.I)
QUALIFIER = {'possession-4': '(estimated)', 'coach': '(coach-marked)', 'mixed': '(estimated, partly coach-marked)'}
# A headline or short label of a finding whose phase is unknown never says who had the ball (critic B-4: "Attacking
# shape" / "Team shape with the ball" on 28 of 28 amateur findings while the report said the phase was unknown).
PHASE_CLAIM_RE = re.compile(r'\bwith the ball\b|\battack(?:s|ing|ed|ers?)?\b|\bin attack\b|\bin possession\b', re.I)
METRE_RE = re.compile(r'(?<![\w.])(\d+(?:\.\d+)?)\s?(?:m|metres|meters)\b')
DEGREE_RE = re.compile(r'(?<![\w.])\d+(?:\.\d+)?\s?degrees\b')
RATE_RE = re.compile(r'\bper \d+ minutes\b', re.I)
SENTENCE_SPLIT = re.compile(r'(?<=[.!?])\s+')

# ---------------------------------------------------------------------------------------------------------------
# Relation polarity (critic N12). The text states a relation only with these phrases; the gate finds them (negative
# first, so "no teammate was covering" is not read as "was covering"), checks each against the claims, and refuses an
# observed past-tense sentence that uses a relation verb outside them (a paraphrase the gate cannot read).
POLARITY = {
    'pressure': {
        False: (r'nobody went to the ball', r'nobody going to (?:them|it|the ball)', r'nobody put pressure on',
                r'no pressure on', r'had no pressure', r'(?:was|went) (?:left )?(?:not pressed|unpressed)',
                r'unpressed', r'nobody closed down', r'no defender engaged', r'nobody engaged',
                r'had no first-defender pressure', r'with nobody going to'),
        True: (r'went to the ball', r'got a player to the ball', r'put pressure on', r'pressure came from',
               r'pressed the (?:ball carrier|carrier|ball)', r'closed down the ball carrier',
               r'closed the ball carrier', r'engaged the (?:ball )?carrier', r'pressure on the ball came',
               r'had pressure on the ball', r'first-defender pressure from', r'was pressed by',
               r'pressed with cover', r'got pressure on')},
    'cover': {
        False: (r'(?:nobody|no teammate|no one) (?:was )?close behind to (?:help|cover)',
                r'no teammate (?:was )?covering', r'nobody (?:was )?covering', r'no one (?:was )?covering',
                r'no teammate (?:was )?in a covering position', r'without covering support',
                r'outside a covering position', r'no second defender (?:was )?covering', r'came without cover',
                r'(?:had|with) no cover', r'no covering teammate', r'without cover', r'without a covering',
                r'nobody (?:was )?behind to help'),
        True: (r'(?:a|the|one) teammate (?:was |stayed )?close behind to (?:cover|help)', r'cover close behind',
               r'covered behind', r'covering behind', r'with [^,.;]{1,60}? covering\b', r'in a covering position',
               r'gave covering support', r'covering support from', r'covered (?:at|by|from)', r'came with cover',
               r'held a covering position', r'with cover behind', r'stayed close behind to help',
               r'(?:someone|somebody|a teammate) (?:also )?stayed close behind', r'with cover\b')},
    'weak_side': {
        False: (r'nobody (?:was |stayed )?goal-side on the far side', r'no defender (?:was |stayed )?goal-side on '
                r'the far side', r'far side (?:was )?(?:left )?(?:open|empty|unprotected)',
                r'nobody (?:was )?(?:protecting|guarding) the far side', r'no far-side defender',
                # fix round 1 (critic B-5): the measured wording of the tolerant goal-side count
                r'(?:nobody|no defender|no one) on the far side (?:was|stood) (?:deeper than the ball|nearer our goal '
                r'than the ball|between the ball and our goal)'),
        True: (r'(?:defenders?|players?|someone|one of ours) (?:was |were |stayed )?goal-side on the far side',
               r'far side (?:was |stayed )?protected', r'protected the far side', r'(?:tucked|stayed) in on the far '
               r'side', r'(?:defenders?|players?|someone|one of ours) on the far side (?:was|were|stood) (?:deeper '
               r'than the ball|nearer our goal than the ball|between the ball and our goal)')},
    'advance_support': {
        False: (r'unsupported', r'advanced without (?:enough |any )?(?:cover|support)',
                r'(?:nobody|no teammate) covered the space (?:they|he|she) left',
                r'space (?:they|he|she) left was not covered', r'without support behind',
                r'not matched by'),
        True: (r'supported advance', r'advance was supported', r'advanced with (?:cover|support)',
               r'space (?:they|he|she) left was covered', r'covered the space (?:they|he|she) left')},
    'channel_cover': {
        False: (r'(?:nobody|no teammate|no one) (?:else )?(?:was )?in the (?:left |right |wide |vacated )?channel '
                r'behind', r'(?:left |right |wide |vacated )?channel behind (?:them|him|her)? ?(?:stayed |was )?empty',
                r'empty (?:left |right )?channel', r'nobody filled the (?:left |right )?channel'),
        True: (r'(?:was|stayed) in the (?:left |right |wide )?channel behind',
               r'with [^,.;]{1,60}? in the (?:left |right |wide )?channel behind',
               r'(?:moved|dropped|stepped) into the (?:left |right |wide )?channel behind',
               r'filled the (?:left |right )?channel', r'covered the (?:left |right |wide )?channel')},
    'cb_shift': {      # 'central defender': the small-format word (SMALL_WORDS)
        False: (r'(?:centre-back|central defender)(?: \([ab]-\d+\))? (?:did not|didn\'t) (?:move|shift) across',
                r'(?:centre-back|central defender)(?: \([ab]-\d+\))? stayed (?:central|inside)',
                r'no (?:centre-back|central defender) (?:moved|shifted) across'),
        True: (r'(?:centre-back|central defender)(?: \([ab]-\d+\))? (?:moved|shifted) (?:across|over)',)},
    'dm_drop': {
        False: (r'(?:midfielder|six)(?: \([ab]-\d+\))? (?:did not|didn\'t) drop',
                r'(?:midfielder|six)(?: \([ab]-\d+\))? stayed (?:high|ahead|up)',
                r'no (?:holding |deepest )?midfielder dropped'),
        True: (r'(?:midfielder|six)(?: \([ab]-\d+\))? dropped',)},
    # fix round 3 (FACTS F-1): the counts carry the margin ("well" / "at least 10 m") before "behind the ball"
    'behind_ball': {
        False: (r'(?:nobody|no one|no player) (?:was |stayed )?(?:back|(?:(?:well|at least \d+ m) )?behind the ball|'
                r'ready to protect our goal)',
                r'nobody back', r'only \d+ (?:of our )?(?:outfield )?players? (?:was |were |stayed )?'
                r'(?:(?:well|at least \d+ m) )?behind the ball',
                r'none of our (?:outfield )?players (?:was |were |stayed )?(?:(?:well|at least \d+ m) )?behind the '
                r'ball', r'nobody stayed back', r'no one stayed back',
                r'fewer players (?:(?:well|at least \d+ m) )?behind the ball than'),
        True: (r'(?:someone|somebody|a player|one of us) (?:was |stayed )?(?:back|ready to protect our goal)',
               r'enough players (?:stayed |were )?(?:(?:well|at least \d+ m) )?behind the ball',
               r'kept enough players (?:(?:well|at least \d+ m) )?behind the ball')},
    'lateral_gap': {
        False: (r'kept the (?:ball-side )?gaps? (?:between them )?small', r'(?:ball-side )?gaps? (?:between them )?'
                r'(?:stayed|were|was) small', r'no (?:large )?ball-side gap'),
        True: (r'gap of (?:about )?\d+ m (?:opened|was open)', r'left a (?:wide|large) gap',
               r'opened a (?:wide |large )?gap', r'(?:wide|large) gap opened')},
    'unit_gap': {
        False: (r'lines stayed close', r'kept the distance between (?:the )?lines (?:short|small)',
                r'stayed close to (?:our )?midfield'),
        True: (r'(?:large|big) space between (?:our |the )?(?:defenders|back line) and (?:our |the )?midfield',
               r'lines (?:were|stood) far apart', r'(?:large|big) space between the lines')},
    'line_evenness': {
        False: (r'line was uneven', r'line was not level'),
        True: (r'line (?:stayed|was) (?:even|level)', r'defenders stayed level', r'stayed level with each other')},
    'width': {
        False: (r'stayed narrow', r'was narrow', r'did not spread wide', r'bunched (?:up )?in the middle'),
        True: (r'spread out wide', r'spread wide', r'stayed wide', r'(?:the )?width was (?:there|used)',
               r'was wide enough')},
    'depth': {
        False: (r'stayed short', r'had little depth', r'little space from back to front', r'was short from back to '
                r'front', r'did not spread (?:out )?from back to front', r'was not deep'),
        True: (r'wide and deep', r'tall and deep', r'spread from back to front', r'deep enough')},
    'channels': {
        False: (r'(?:one|a) side (?:was |stayed )?empty', r'nobody on (?:the )?(?:left|right)'),
        True: (r'(?:players|someone) on the left, in the middle and on the right', r'(?:all|every) (?:three|\d) '
               r'(?:lateral )?channels', r'in all \d (?:lateral )?channels')},
}
POLARITY_RE = {k: {pol: re.compile('|'.join(f'(?:{p})' for p in pats), re.I) for pol, pats in v.items()}
               for k, v in POLARITY.items()}
# Relation verbs that, in an observed (past-tense) sentence, must be inside a recognised phrase.
POLARITY_GUARD = {
    'pressure': r'\bpressed\b|\bpressured\b|\bclosed down\b', 'cover': r'\bcovered\b|\bcovering\b',
    'weak_side': r'\bfar side (?:was|stayed)\b', 'advance_support': r'\bsupported\b',
    'channel_cover': r'\b(?:in|into) the (?:left |right |wide |vacated )?channel behind\b',
    'cb_shift': r'\bmoved across\b|\bshifted across\b', 'dm_drop': r'\bdropped\b',
    'behind_ball': r'\bstayed back\b|\bready to protect\b', 'lateral_gap': r'\bgap (?:opened|was (?:open|closed))\b',
    'unit_gap': r'\blines (?:stayed|were) (?:close|apart)\b', 'line_evenness': r'\buneven\b',
    'width': r'\bnarrow\b', 'depth': r'\bshort from back to front\b', 'channels': r'\bevery channel\b'}
POLARITY_GUARD_RE = {k: re.compile(v, re.I) for k, v in POLARITY_GUARD.items()}

# ---------------------------------------------------------------------------------------------------------------
# Players: role words (level 3+), relative words (level 2), and the phrases the gate reads back.
GROUP = {'LB': 'D', 'RB': 'D', 'LWB': 'D', 'RWB': 'D', 'LCB': 'D', 'RCB': 'D', 'CB': 'D', 'CDM': 'M', 'LDM': 'M',
         'RDM': 'M', 'LM': 'M', 'RM': 'M', 'LCM': 'M', 'CM': 'M', 'RCM': 'M', 'CAM': 'M', 'LW': 'F', 'RW': 'F',
         'CF': 'F', 'LCF': 'F', 'RCF': 'F', 'GK': 'GK'}
DM_ROLES = ('CDM', 'LDM', 'RDM')
CB_ROLES = ('CB', 'LCB', 'RCB')
WIDE_DEFENDERS = ('LB', 'RB', 'LWB', 'RWB')
ROLE_WORD = {'LB': 'left back', 'RB': 'right back', 'LWB': 'left wing-back', 'RWB': 'right wing-back',
             'LCB': 'left centre-back', 'RCB': 'right centre-back', 'CB': 'centre-back',
             'CDM': 'holding midfielder', 'LDM': 'left holding midfielder', 'RDM': 'right holding midfielder',
             'LM': 'left midfielder', 'RM': 'right midfielder', 'LCM': 'left central midfielder',
             'CM': 'central midfielder', 'RCM': 'right central midfielder', 'CAM': 'attacking midfielder',
             'LW': 'left winger', 'RW': 'right winger', 'CF': 'centre-forward', 'LCF': 'left centre-forward',
             'RCF': 'right centre-forward', 'GK': 'goalkeeper'}
RELATIVE_WORD = {'LB': 'left defender', 'LWB': 'left defender', 'RB': 'right defender', 'RWB': 'right defender',
                 'LM': 'left midfielder', 'RM': 'right midfielder', 'LW': 'left forward', 'RW': 'right forward',
                 'CDM': 'deepest midfielder', 'LDM': 'deepest midfielder', 'RDM': 'deepest midfielder',
                 'GK': 'goalkeeper'}
RELATIVE_BLOCK = {'back': 'defender', 'mid': 'midfielder', 'front': 'forward'}
# Below 11v11 a confirmed template role is worded relatively (DESIGN 2.5 "small formats never use ... 11v11 role
# names"; critic S-9 / COACH M3: 24 small-format reports said full-back, centre-back or holding midfielder).
SMALL_ROLE_WORD = {'LB': 'left defender', 'LWB': 'left defender', 'RB': 'right defender', 'RWB': 'right defender',
                   'LCB': 'left central defender', 'RCB': 'right central defender', 'CB': 'central defender',
                   'CDM': 'deepest midfielder', 'LDM': 'deepest midfielder', 'RDM': 'deepest midfielder',
                   'LM': 'left midfielder', 'RM': 'right midfielder', 'LCM': 'central midfielder',
                   'CM': 'central midfielder', 'RCM': 'central midfielder', 'CAM': 'attacking midfielder',
                   'LW': 'left forward', 'RW': 'right forward', 'CF': 'central forward', 'LCF': 'central forward',
                   'RCF': 'central forward', 'GK': 'goalkeeper'}
# The small-format lexicon: every field of a finding below 11v11 is passed through it before the gate (the banks and
# principles speak of full-backs and centre-backs, the 11v11 names); the format_words check then refuses any 11v11
# role name left (ROLE11_RE). Order matters (possessives and plurals first).
SMALL_WORDS = tuple((re.compile(a), b) for a, b in (
    (r"\bFull-back's\b", "Wide defender's"), (r"\bfull-back's\b", "wide defender's"),
    (r'\bFull-backs\b', 'Wide defenders'), (r'\bfull-backs\b', 'wide defenders'),
    (r'\bFull-back\b', 'Wide defender'), (r'\bfull-back\b', 'wide defender'),
    (r'\bWing-backs\b', 'Wide defenders'), (r'\bWing-back\b', 'Wide defender'),
    (r'\bwing-backs\b', 'wide defenders'), (r'\bwing-back\b', 'wide defender'),
    (r'\bcentre-back line\b', 'back line'),
    (r'\bCentre-backs\b', 'Central defenders'), (r'\bcentre-backs\b', 'central defenders'),
    (r'\bCentre-back\b', 'Central defender'), (r'\bcentre-back\b', 'central defender'),
    # "a holding midfielder" in a principle is a role in general (no player singled out): "a midfielder"
    (r'\bA holding midfielder\b', 'A midfielder'), (r'\ba holding midfielder\b', 'a midfielder'),
    (r'\bholding midfielders\b', 'deepest midfielders'), (r'\bHolding midfielder\b', 'Deepest midfielder'),
    (r'\bholding midfielder\b', 'deepest midfielder'),
    (r'\bThe six\b', 'The deepest midfielder'), (r'\bthe six\b', 'the deepest midfielder'),
    (r'\bleft back\b', 'left defender'), (r'\bright back\b', 'right defender')))
ROLE11_RE = re.compile(r"\bfull-backs?\b|\bcent(?:re|er)-backs?\b|\bholding midfielders?\b|\bwing-backs?\b|"
                       r"\b(?:left|right) backs?\b|\bthe six\b", re.I)
REL_ROLE_RE = re.compile(r'(back|mid|front)-([1-9])/([1-9])')
# phrase -> the role codes (regexes) it can name; the gate reads "the <phrase>" / "our <phrase>" (singular)
ROLE_PHRASES = (
    ('left wing-back', r'LWB'), ('right wing-back', r'RWB'), ('left back', r'LB'), ('right back', r'RB'),
    ('left centre-back', r'LCB'), ('right centre-back', r'RCB'), ('centre-back', r'CB|LCB|RCB'),
    ('left holding midfielder', r'LDM'), ('right holding midfielder', r'RDM'),
    ('holding midfielder', r'CDM|LDM|RDM'), ('deepest midfielder', r'CDM|LDM|RDM|mid-\d/\d'),
    ('left central midfielder', r'LCM'), ('right central midfielder', r'RCM'), ('central midfielder', r'CM|LCM|RCM'),
    ('attacking midfielder', r'CAM'), ('left midfielder', r'LM|mid-1/\d'), ('right midfielder', r'RM|mid-(\d)/\1'),
    ('left winger', r'LW'), ('right winger', r'RW'), ('left centre-forward', r'LCF'),
    ('right centre-forward', r'RCF'), ('centre-forward', r'CF|LCF|RCF'), ('left forward', r'LW|front-1/\d'),
    ('right forward', r'RW|front-(\d)/\1'), ('left defender', r'LB|LWB|back-1/\d'),
    ('right defender', r'RB|RWB|back-(\d)/\1'), ('leftmost defender', r'back-1/\d|LB|LWB'),
    ('rightmost defender', r'back-(\d)/\1|RB|RWB'), ('leftmost midfielder', r'mid-1/\d|LM'),
    ('rightmost midfielder', r'mid-(\d)/\1|RM'), ('leftmost forward', r'front-1/\d|LW'),
    ('rightmost forward', r'front-(\d)/\1|RW'), ('goalkeeper', r'GK'), ('six', r'CDM|LDM|RDM'),
    # the small-format words of SMALL_ROLE_WORD (fix round 1, critic S-9)
    ('left central defender', r'LCB'), ('right central defender', r'RCB'), ('central defender', r'CB|LCB|RCB'),
    ('central forward', r'CF|LCF|RCF'),
)
# "the centre-back line" names the team's line, not a player (a level-5 sentence with a failed identity guardrail
# was refused for it on the identity_crossing scene): a phrase followed by "line" is not a role word.
ROLE_PHRASE_RE = tuple((re.compile(r'\b(?:the|our) '+re.escape(p).replace(r'\ ', ' ')+r'\b(?![-\w])(?! lines?\b)',
                                   re.I),
                        re.compile(codes), p) for p, codes in sorted(ROLE_PHRASES, key=lambda x: -len(x[0])))

PLACE = {'own_third': ('near our goal', 'in our own third'), 'middle_third': ('in the middle of the pitch',
                                                                            'in the middle third'),
         'final_third': ("near the other team's goal", 'in the final third')}
SIDE_PLACE = {'left': 'on our left', 'centre': 'through the middle', 'right': 'on our right'}
OTHER_SIDE = {'left': 'right', 'right': 'left'}
STYLE_WORD = {'possession': 'possession', 'counter': 'counter', 'high_press': 'high press', 'compact': 'compact'}
STYLE_EFFECT = {    # context.styleEffect codes (profiles.json styles.*.effects.*.code) -> what the style changed;
    #                  a code without words here is not worded (never invented)
    'possession_advance_expected': 'advancing full-backs are expected, so only the uncovered spells count here',
    'possession_shape_emphasised': 'width and depth are weighed as strengths first',
    'counter_more_behind': 'one more player is expected behind the ball',
    'counter_width_not_judged': 'width with the ball is not judged',
    'counter_deep_block': 'a deeper block is not treated as a concern',
    'high_press_cover_exempt': 'missing cover in the opponents\' third is not counted',
    'high_press_gap_tolerated': 'a larger gap between midfield and defence is tolerated',
    'compact_tightened': 'the spacing is judged more tightly',
    'compact_far_side_accepted': 'open space on the far side is accepted',
}
# style effects a description does not state (fix round 3, COACH F-11: "For the possession style chosen in the setup:
# width and depth are weighed as strengths first." under a description, which weighs nothing)
STYLE_DESCRIBE_SILENT = ('possession_shape_emphasised',)
# style effects that ACCEPT what a description shows (assess style_explains): a next objective never asks for it
# (fix round 3, COACH F-12; report.next_objective)
STYLE_ACCEPTS = ('compact_far_side_accepted',)

# ---------------------------------------------------------------------------------------------------------------
# Principles (DESIGN 6.2 'principle' claims). Paraphrases in PitchIQ's own words; the candidate source ids are the
# research ids (DESIGN 2.7 canonical or research form). A principle claim is tagged FEDERATION only when one of its
# candidates is in the resolved profile's sources (PROFILE's sources.json decides the ids); otherwise it is a
# PITCHIQ-PROPOSAL without a source. Nothing here claims a federation endorses PitchIQ.
PRINCIPLES = {
    'pressure_on_ball': {'detector': 'pressure_cover', 'relation': 'pressure', 'text': {
        (1,): 'When the other team has the ball near our goal, someone goes to the ball quickly.',
        (2,): 'When the other team has the ball near our goal, the nearest player puts pressure on it.',
        (3,): 'When the ball carrier is in our own third, the nearest player closes them down to slow the attack.',
        (4,): 'In our own third the nearest defender engages the ball carrier so the attack has to slow down.',
        (5,): 'In our own third the first defender engages the carrier quickly, so the carrier has less time.',
        (6,): 'An unchallenged carrier near our goal has time to choose; first-defender pressure takes that time '
              'away.',
        (7,): 'First-defender pressure in the own third forces the carrier to act before choosing.'},
        'sources': {(1,): ('US-GM-U78-PDF', 'USS-GM-U78', 'US-GM-U78-PAGE'),
                    (2,): ('US-GM-U910-PDF', 'USS-GM-U910', 'US-GM-U910-PAGE'),
                    (3,): ('US-GM-U1112-PDF', 'USS-GM-U1112', 'US-GM-U1112-PAGE'),
                    (4,): ('US-GM-U1314-PDF', 'USS-GM1314', 'USS-GM-U1314', 'US-GM-U1314-PAGE'),
                    (5,): ('USS-GM1516', 'USS-GM-U1516', 'US-GM-U1516-PDF'),
                    (6, 7): ('USS-GM17', 'USS-GM-U17')}},
    'help_teammate': {'detector': 'pressure_cover', 'relation': 'cover', 'text': {
        (1,): 'Going to the ball together: one player goes, and a teammate stays close to help.'},
        'sources': {(1,): ('EF-PROG-7-11', 'EF-SWIFT-SUPPORT')}},
    'cover_first_defender': {'detector': 'pressure_cover', 'relation': 'cover', 'text': {
        (1,): 'Going to the ball together: one player goes, and a teammate stays close to help.',
        (2,): 'One player goes to the ball and the nearest teammate stays a little behind, ready to help.',
        (3,): 'Cover is standing behind the teammate who presses, at an angle and distance that make you the next '
              'one in the way.',
        (4,): 'The presser gets near enough to challenge while the nearest teammate covers at an angle, close enough '
              'to step in.',
        (5,): 'The first defender pressures the ball while the second defender covers at an angle behind, and the '
              'rest stay ball-side and goal-side.',
        (6,): 'Pressure on the ball holds up when the second defender covers the presser and the unit stays compact '
              'around the ball.',
        (7,): 'A press is only as strong as its second defender: the cover angle decides whether the carrier can go '
              'round the presser.'},
        'sources': {(1, 2): ('EF-PROG-7-11', 'EF-SWIFT-SUPPORT'), (3,): ('EF-COVER-RECOVER',),
                    (4,): ('EF-PROG-PRESS-COVER', 'EF-COVER-RECOVER-SESSION', 'EF-COVER-RECOVER'),
                    (5,): ('EF-PTC', 'EF-PCB25', 'EF-PZ', 'EF-COVER-RECOVER'), (6, 7): ('EF-DLE-ROLES', 'EF-PTC')}},
    'far_side_balance': {'detector': 'weak_side_balance', 'relation': 'weak_side', 'text': {
        (1, 2): 'When the ball is on one side, a defender on the far side stays between the ball and our goal.',
        (3,): 'With the ball on one side, the far-side defender stays goal-side, between the ball and our goal.',
        (4,): 'When the ball is wide, a far-side defender tucks in goal-side so the middle and the far side stay '
              'protected.',
        (5,): 'With the ball wide, balance means the far-side defenders stay goal-side to guard the middle and a '
              'switch of play.',
        (6,): 'Far-side defenders hold the balance against a switch while the unit stays compact around the ball.',
        (7,): 'The far-side structure is what makes a switch of play slow; its distance from the ball is traded '
              'against central protection.'},
        'sources': {(1, 2, 3, 4): ('EF-PRESS-COVER-BALANCE-2023', 'EF-PCB-2023', 'EF-PCB25', 'EF-PCB-2025'),
                    (5,): ('EF-PCB25', 'EF-PTC'), (6, 7): ('EF-DLE-ROLES',)}},
    'cover_behind_advancing_defender': {'detector': 'wide_defender_advance', 'relation': 'channel_cover', 'text': {
        (1,): 'When we attack, someone stays ready to protect our goal.',
        (2,): 'A defender who joins the attack needs a teammate holding back behind them.',
        (3,): 'A defender can join the attack when a teammate covers the space they leave behind.',
        (4,): 'Keeping a defensive shape behind the ball while attacking is what lets a full-back go forward.',
        (5,): 'A full-back\'s forward run is balanced when a centre-back moves across, a holding midfielder drops '
              'or a teammate fills the channel.',
        (6,): 'An advancing full-back is covered when the group behind the ball adjusts behind them.',
        (7,): 'An advanced full-back needs a matching adjustment in the rest-defence structure, judged against the '
              'team\'s game plan.'},
        'sources': {(3,): ('EF-COVER-RECOVER',), (4,): ('US-GM-U1314-PDF', 'USS-GM1314', 'USS-GM-U1314'),
                    (5,): ('USS-GM1516', 'USS-GM-U1516', 'US-GM-U1516-PDF')}},
    'someone_back': {'detector': 'behind_ball', 'relation': 'behind_ball', 'text': {
        (1,): 'When we attack, someone stays ready to protect our goal.',
        (2,): 'When we attack, someone stays back near our goal.',
        (3,): 'When we attack, a few players stay behind the ball so the other team cannot run straight at our goal.',
        (4,): 'While attacking, a defensive shape stays behind the ball so the team is ready if the ball changes '
              'hands.',
        (5,): 'A high defensive shape stays behind the ball while we attack, ready for the moment the ball changes '
              'hands.',
        (6,): 'Enough players behind the ball while attacking keep the team ready for the moment the ball changes '
              'hands.',
        (7,): 'Rest-defence occupation behind the ball is weighed against the opponents who stay forward.'},
        'sources': {(4,): ('US-GM-U1314-PDF', 'USS-GM1314', 'USS-GM-U1314'),
                    (5,): ('USS-GM1516', 'USS-GM-U1516', 'USS-GM17')}},
    'unit_spacing': {'detector': 'defensive_spacing', 'relation': 'lateral_gap', 'text': {
        (1,): 'Defenders who move across together leave no big spaces between them.',
        (2,): 'When the ball moves across, our defenders move across together.',
        (3,): 'Defenders move across together and keep the spaces between them small near the ball.',
        (4,): 'Defenders move across together and keep the gaps between them small on the side of the ball.',
        (5,): 'The back line shifts as a unit toward the ball, keeping the gaps small and the distance to midfield '
              'short.',
        (6,): 'A compact block keeps the ball-side gaps closed and the space between the lines short; space far '
              'from the ball is the accepted cost.',
        (7,): 'Block spacing is judged on the ball side: closed gaps and short distances between the lines matter '
              'more than the far touchline.'},
        'sources': {(4,): ('EF-PROG-PRESS-COVER', 'EF-PTC'), (5,): ('EF-PTC', 'EF-PZ'),
                    (6, 7): ('EF-DLE-WIDE', 'EF-PHASES')}},
    'compact_lines': {'detector': 'defensive_spacing', 'relation': 'unit_gap', 'text': {
        (1, 2, 3): 'Our defenders stay close behind our midfielders, so there is little space between them.',
        (4,): 'The back line stays close behind midfield, so the space between the lines stays small.',
        (5,): 'Compactness without the ball means a short distance from the back line to midfield as the block '
              'moves.',
        (6,): 'A short block keeps little space between the back line and midfield for an opponent to receive in.',
        (7,): 'The distance between the back line and midfield is weighed against the block height the game plan '
              'asks for.'},
        'sources': {(5,): ('USS-GM1516', 'USS-GM-U1516')}},
    'height_width_depth': {'detector': 'attacking_shape', 'relation': 'width', 'text': {
        (1,): 'When we have the ball, spreading out tall, wide and deep gives the player on the ball more choices.',
        (2,): 'Spreading out when we have the ball gives the player on the ball more space and more choices.',
        (3,): 'Width and depth when we have the ball stretch the other team and open space.',
        (4,): 'Height, width and depth when we have the ball stretch the opposing block.',
        (5,): 'Holding the width and the height with the ball keeps the opposing back line busy and opens space '
              'between its lines.',
        (6,): 'Height, width and depth with the ball pull the opposing block apart and open lanes between its lines.',
        (7,): 'The attacking structure is read by how far it stretches the block in both directions, judged against '
              'the game plan.'},
        'sources': {(1,): ('US-GM-U78-PDF', 'USS-GM-U78'), (2,): ('US-GM-U910-PDF', 'USS-GM-U910'),
                    (3,): ('US-GM-U1112-PDF', 'USS-GM-U1112'), (4,): ('US-GM-U1314-PDF', 'USS-GM1314'),
                    (5,): ('USS-GM1516',), (6, 7): ('USS-GM17',)}},
}
DEFAULT_PRINCIPLE = {'pressure_cover': 'cover_first_defender', 'weak_side_balance': 'far_side_balance',
                     'wide_defender_advance': 'cover_behind_advancing_defender', 'behind_ball': 'someone_back',
                     'defensive_spacing': 'unit_spacing', 'attacking_shape': 'height_width_depth'}
# Concepts defined once per report, the first time a sentence of that level uses them (DESIGN 6.3).
DEFINITIONS = (
    ('pressure_cover', (2,), re.compile(r'\bpressure\b|\bcover\b', re.I),
     'Pressure means getting close to the player with the ball; cover means a teammate a little behind, ready to '
     'help.'),
    # fix round 2: "not a federation definition" put a federation word in 84 run-copy overview texts (EVAL's
    # federation-name scan); the sentence now says the same without naming any body
    # fix round 3 (FACTS F-1): both counts are of players at least the behind-the-ball margin from the ball ({bm}:
    # behind_words, "at least 10 m" with usable metres, else "well"); definition_text fills it
    ('rest_defence', (6, 7), re.compile(r'\brest[- ]defen[cs]e\b', re.I),
     'Rest defence here is a PitchIQ measure of its own, not an official definition: our outfield players {bm} '
     'behind the ball minus the opponents {bm} on our side of it.'),
)
DEFINITION_START = {code: text.split(':')[0] for code, _, _, text in DEFINITIONS}


def definition_text(definition, bm='well'):
    """A DEFINITIONS text with the behind-the-ball margin words filled in (FACTS F-1): bm is behind_words' value."""
    return definition.replace('{bm}', bm or 'well')

# What could not be confirmed (limitation claims): guardrails that failed and the finding's limitation codes, as
# (plain words for levels 1-3, technical words for 4-7). Codes without wording are not stated (never invented).
LIMITATION_WORDS = {
    'identity': ('who the players were', "the players' identities"),
    'identity_uncertain': ('who the players were', "the players' identities"),
    'position': ('exactly where some players stood', 'some player positions'),
    'registration_low': ('the pitch lines on the video', 'the pitch registration'),
    'role': ("the player's position in the team", "the player's role"),
    'role_unconfirmed': ("the player's position in the team", "the player's role"),
    # the context guardrail reads the worst visibility over the cited spells, not the moments shown (FACTS L11b: all
    # 10 outfield players were observed at every cited BC60 / BC303 moment), so the words say "for the whole spell"
    'context': ('whether all nearby teammates were in view for the whole spell',
                'the team context (not every teammate was in view for the whole spell)'),
    'team_incomplete': ('the whole team (not everyone was in view)', 'the whole team (not all players were in '
                                                                      'view)'),
    'out_of_view': ('players outside the camera view', 'players outside the camera view'),
    'phase': ('who had the ball the whole time', 'the phase for every moment'),
    'phase_unknown': ('who had the ball', 'the phase (who had the ball)'),
    'possession_unavailable': ('who had the ball', 'the phase (who had the ball)'),
    'near_change': ('whether the play had settled', 'whether the phase had settled'),
    'ball_unreliable': ('where the ball was', 'the ball position'),
    'no_carrier': ('which player had the ball', 'the ball carrier'),
    'intent': ('whether the movement was on purpose', 'whether the movement was intentional (a recovery run, a '
                                                      'swap or a change of shape)'),
    'recovering': ('whether the player was running back', 'whether the player was recovering toward goal'),
    'alternative': ('a better position to show', 'a validated alternative position'),
    'evidence': ('enough moments for a firm conclusion', 'enough evidence for a work item at this level'),
    'scale_unknown': ('the field size (distances are left out)', 'the field size, so distances are left out'),
    'short_range': ('a long enough clip', 'a long enough stretch of play'),
    'units_interleaved': ('which players formed each line', 'the unit lines (players were interleaved)'),
    'style_unknown': ('the playing style', 'the playing style (none was declared)'),
    'formation_mismatch': ('the team shape', 'the formation (the measured shape differed from the one selected)'),
}
# Limitation codes assess.py sets that get a sentence of their own rather than a place in the "not confirmed" list
# (DETECT wave 3 asked for words): one_instance (one distinct failing spell, below the evidence table), level_cap (a
# work item over the level's cap, listed as a review), no_moment (no evidence moment existed: a description only).
OWN_SENTENCE_LIMITS = ('one_instance', 'level_cap', 'no_moment', 'mixed_evidence', 'formation_changing')
# Fix round 1 (DETECT's new codes): mixed_evidence = a strength withheld because the not-demonstrated share of the
# decided opportunities reached the professional base share (the family is described, never praised, critic B-2;
# fix round 3, LEAD decision (3): significantly above the level's base, scaled below level 5, by a binomial tail);
# formation_changing = an opportunity of the family fell within a change of formation and was left out (critic S-3).
# Before this round both were silent: 25 + 12 scene findings and 18 run-copy findings carried them with no word.
# Fix round 2 (LEAD decision (b)): a 'moments worth a look' review (assess.MOMENTS) is a review from repeated CLEAR
# failures that stayed below the level's verdict bar. Its text states the clear moments of the opportunities PitchIQ
# could judge (measurements.clearMoments / decidedMoments, never aggregate.opportunities: the round-1 text said "3 of
# 12 forward runs" for 3 clear of 11 judged), that it is not a verdict, the moments, why the check matters and what to
# look for in the clips; no verdict wording, no zone, no drill. MOMENTS_KEYS are the measurements it reads.
MOMENTS = 'moments_worth_a_look'
MOMENTS_KEYS = ('clearMoments', 'clearSpells', 'decidedMoments', 'failingMoments')
MOMENTS_PREFIX = 'Moments worth a look: '
# Words a 'moments worth a look' review never uses (invariant_problems): verdict framing, prescriptions, blame.
VERDICT_WORDS_RE = re.compile(r"\btry this\b|\bnext step\b|\bwork on\b|\bto work on\b|\bshould\b|\bmust\b|"
                              r"\bneeds? to\b|\bmistakes?\b|\berrors?\b|\bwrong\b|\bfail\w*|\bpoor\w*|\bbad(?:ly)?\b|"
                              r"\bpossible issue\b", re.I)

# ---------------------------------------------------------------------------------------------------------------
# Phrasing banks: {slot: {levels: [templates]}}. A template is used only when every {slot} it names has a value; the
# value None (a number this level does not show, a player this level does not name) skips the template. Slots ending
# in ':<mode>' are per failure mode. '{try}' is "Try this" (child) or "Next step" (adult register).
BANK = {}

BANK['pressure_cover'] = {
    'headline_fail:no_pressure': {
        (1,): ['{try}: someone goes to the ball near our goal', '{try}: getting to the ball quickly near our goal',
               '{try}: the first player to the ball near our goal'],
        (2,): ['{try}: pressure on the player with the ball', '{try}: someone goes to the ball quickly',
               '{try}: getting pressure on the ball near our goal'],
        (3,): ['Getting pressure on the ball in our own third', 'Closing down the ball carrier in our own third',
               '{try}: pressure on the ball carrier in our own third'],
        (4, 5): ['The ball carrier went unpressed in our own third', 'No pressure on the ball carrier in our own third',
                 'Pressure on the ball in our own third'],
        (6, 7): ['Unpressed ball carriers in our own third', 'No first-defender pressure in our own third',
                 'First-defender pressure in our own third']},
    'headline_fail:no_cover': {
        (1,): ['{try}: a teammate close behind to help', '{try}: going to the ball with a helper close behind',
               '{try}: one goes to the ball, one stays close to help'],
        (2,): ['{try}: pressure and cover together', '{try}: a teammate close behind to cover',
               '{try}: cover behind the player who goes to the ball'],
        (3,): ['Pressure on the ball came without cover', 'Cover behind the player pressing the ball',
               '{try}: cover behind the player pressing the ball'],
        (4, 5): ['Pressure on the ball without covering support', 'The presser had no cover behind',
                 'Cover behind the first defender'],
        (6, 7): ['First-defender pressure without a covering second defender', 'Pressure without cover behind the '
                 'first defender', 'Second-defender cover behind the press']},
    'headline_ok': {
        (1,): ['Going to the ball', 'Getting to the ball as a team', 'Someone went to the ball'],
        (2,): ['Pressure with cover', 'Pressure and cover together', 'Getting pressure and cover on the ball'],
        (3,): ['Pressure on the ball with cover behind', 'Pressing with a teammate covering',
               'Cover behind the player pressing'],
        (4, 5): ['Pressure on the ball with covering support', 'First defender pressed with cover behind',
                 'Near enough to challenge, covered behind'],
        (6, 7): ['Pressure and cover on the ball', 'First-defender pressure with cover', 'Second-defender cover behind '
                 'the first defender']},
    'headline_describe': {(1,): ['Going to the ball', 'Getting to the ball near our goal'],
                          (2, 3, 4, 5, 6, 7): ['Pressure on the ball', 'Pressure and cover in our half']},
    'short_fail:no_pressure': {(1, 2): ['Going to the ball', 'First to the ball'],
                               (3,): ['Pressure on the ball', 'Closing down the ball carrier'],
                               (4, 5, 6, 7): ['Unpressed ball carrier', 'No pressure on the ball']},
    'short_fail:no_cover': {(1,): ['A helper close behind', 'Someone close to help'],
                            (2,): ['Pressure and cover', 'Cover behind the pressure'],
                            (3,): ['Pressure without cover', 'Cover behind the presser'],
                            (4, 5, 6, 7): ['Pressure without cover', 'No covering second defender']},
    'short_ok': {(1,): ['Going to the ball', 'Getting to the ball'],
                 (2,): ['Pressure with cover', 'Pressure and cover'],
                 (3, 4, 5, 6, 7): ['Pressure with cover behind', 'Pressure and cover']},
    'short_describe': {(1,): ['Going to the ball'], (2, 3, 4, 5, 6, 7): ['Pressure on the ball']},
    'open_fail:no_pressure': {
        (1,): ['{Phase}, {place}, there were moments when nobody went to the ball.',
               '{Phase}, the player with the ball {place} sometimes had nobody going to them.',
               '{Phase}, {place}, the ball was sometimes left with nobody going to it.'],
        (2,): ['{Phase}, {place}, nobody put pressure on the player with the ball.',
               '{Phase}, the player with the ball {place} had no pressure on them.',
               '{Phase}, {place}, there was no pressure on the player with the ball.'],
        (3,): ['{Phase}, {place}, nobody closed down the ball carrier; the nearest player was {d1} away.',
               '{Phase}, the ball carrier {place} was not pressed: the nearest player stayed {d1} away.',
               '{Phase}, {place}, there was no pressure on the ball carrier.'],
        (4, 5): ['{Phase}, {place}, the ball carrier was left unpressed; the nearest defender, {actor}, was {d1} away.',
                 '{Phase}, {actor} was the nearest defender to the ball carrier {place} but stayed {d1} away, so '
                 'there was no pressure on the ball.',
                 '{Phase}, {place}, no defender engaged the ball carrier; {actor} was the nearest at {d1}.',
                 '{Phase}, {place}, no defender engaged the ball carrier.'],
        (6, 7): ['{Phase}, {place}, the ball carrier had no first-defender pressure: {actor} was nearest at {d1}.',
                 '{Phase}, {place}, nobody engaged the carrier; the nearest defender ({actor}) held {d1} off.',
                 '{Phase}, the carrier {place} went unpressed, with {actor} the nearest defender at {d1}.',
                 '{Phase}, {place}, nobody engaged the carrier as first defender.']},
    'open_fail:no_cover': {
        (1,): ['{Phase}, {place}, someone went to the ball, but nobody was close behind to help.',
               '{Phase}, one of us went to the ball {place}, with no teammate close behind to help.',
               '{Phase}, {place}, the player who went to the ball had nobody behind to help.'],
        (2,): ['{Phase}, {place}, {actor} went to the ball, but no teammate was close behind to cover.',
               '{Phase}, {actor} put pressure on the player with the ball {place}, with no teammate close behind '
               'to cover.',
               '{Phase}, {place}, pressure came from {actor}, and nobody was close behind to cover.'],
        # fix round 3 (FACTS F-7): {mate} / {cover_dist} are the nearest teammate WITHIN the cover angle (contextual:
        # coverId, tactics_geom.cover_candidate, 60 degrees off the presser's line to goal), not the nearest teammate
        # (on Metrica B-07 stood nearer, at 9.8 m, 69.5 degrees off, while "the nearest teammate, B-03" was 13.4 m)
        (3,): ['{Phase}, {place}, {actor} pressed the ball carrier, but nobody was covering behind; the nearest '
               'teammate within the cover angle was {cover_dist} away.',
               '{Phase}, {actor} closed down the ball carrier {place} with no teammate covering behind them.',
               '{Phase}, {place}, {actor} put pressure on the ball carrier while no teammate was covering behind.'],
        (4, 5): ['{Phase}, {place}, {actor} closed the ball carrier to {d1}, but no teammate was in a covering '
                 'position; {mate_named} was the nearest within the cover angle, {cover_dist} away.',
                 '{Phase}, {actor} pressed the ball carrier {place} without covering support: the nearest teammate '
                 'within the cover angle, {mate_named}, was {cover_dist} away.',
                 '{Phase}, {place}, {actor} pressed the ball carrier at {d1} while {mate_named}, the nearest teammate '
                 'within the cover angle, stood {cover_dist} away, outside a covering position.',
                 '{Phase}, {place}, {actor} pressed the ball carrier, but no teammate was in a covering position; the '
                 'nearest teammate within the cover angle was {cover_dist} away.',
                 '{Phase}, {place}, {actor} pressed the ball carrier, but no teammate was in a covering position.'],
        (6, 7): ['{Phase}, {place}, {actor} engaged the carrier as first defender with no second defender covering; '
                 '{mate_named} was nearest within the cover angle, at {cover_dist}.',
                 '{Phase}, {place}, first-defender pressure from {actor} came without cover: the second defender, '
                 '{mate_named}, held {cover_dist} away at {cover_angle}.',
                 '{Phase}, {actor} pressed the carrier {place} at {d1}, and the nearest teammate within the cover '
                 'angle, {mate_named}, was {cover_dist} away, outside a covering position.',
                 '{Phase}, {place}, {actor} engaged the carrier as first defender with no second defender covering; '
                 'the nearest teammate within the cover angle was {cover_dist} away.',
                 '{Phase}, {place}, {actor} engaged the carrier as first defender with no second defender covering.']},
    'open_ok': {
        (1,): ['{Phase}, {place}, someone from our team went to the ball.',
               '{Phase}, one of us went to the ball quickly {place}.',
               '{Phase}, {place}, we got a player to the ball.'],
        (2,): ['{Phase}, {place}, {actor} went to the ball and a teammate was close behind to cover.',
               '{Phase}, {actor} put pressure on the player with the ball {place}, with a teammate close behind to '
               'cover.', '{Phase}, {place}, we had pressure on the ball and cover close behind it.'],
        (3,): ['{Phase}, {place}, {actor} pressed the ball carrier and {mate} covered behind, {cover_dist} away.',
               '{Phase}, {actor} closed down the ball carrier {place} with {mate} covering behind them.',
               '{Phase}, {place}, pressure on the ball came with a teammate covering behind.'],
        (4, 5): ['{Phase}, {place}, {actor} closed the ball carrier to {d1} with {mate} covering at {cover_dist}.',
                 '{Phase}, {actor} pressed the ball carrier {place}, and {mate} gave covering support from '
                 '{cover_dist}.',
                 '{Phase}, {place}, the ball carrier was pressed by {actor} with {mate} in a covering position '
                 'behind.'],
        (6, 7): ['{Phase}, {place}, first defender {actor} engaged the carrier and second defender {mate_named} '
                 'covered at {cover_dist}.',
                 '{Phase}, {place}, pressure from {actor} came with cover: {mate} held a covering position '
                 '{cover_dist} behind.',
                 '{Phase}, {actor} pressed the carrier {place} at {d1}, covered by {mate} at {cover_angle}.',
                 '{Phase}, {place}, first defender {actor} engaged the carrier with a teammate covering behind.']},
    'help_ok': {(1,): ['A teammate also stayed close behind to help.', 'Someone stayed close behind to help, too.',
                       'There was also a teammate close behind to help.']},
    'describe': {(1,): ['{Phase}, {place}, the key moments show who went to the ball and who was nearby.'],
                 (2, 3, 4, 5, 6, 7): ['{Phase}, {place}, the moments with the ball carrier and the nearest defenders '
                                      'are listed below for context.']},
    'praise:no_pressure': {(1,): ['There were also moments when someone went to the ball quickly.',
                                  'Some of the time, someone went to the ball straight away.',
                                  'Well done for the moments when someone went to the ball.'],
                           (2, 3): ['We got pressure on the ball {dem_n} times.',
                                    '{dem_n} times, someone put pressure on the player with the ball.',
                                    'There were {dem_n} moments when we got pressure on the ball.']},
    'praise:no_cover': {(1,): ['There were also moments when a teammate stayed close behind to help.',
                               'Some of the time, a teammate stayed close behind to help.',
                               'Well done for the moments when someone stayed close behind to help.'],
                        (2, 3): ['{dem_n} times we had pressure on the ball and cover close behind it.',
                                 'We got pressure and cover together {dem_n} times.',
                                 'There were {dem_n} moments with a teammate close behind to cover.']},
    'hypo:no_cover': {
        (3,): ['If the ball carrier had got past the presser, nobody was in position to stop the next step.',
               'If the carrier had beaten the first player, there was no second player ready to step in.',
               'If the presser had been beaten, the way toward our goal would have been open.'],
        (4, 5): ['If the carrier had beaten the presser, nobody was placed to step in next.',
                 'One dribble past the presser would have left the way toward our goal open.',
                 'Had the presser been beaten, no teammate would have been ready to step in.'],
        (6, 7): ['A single dribble past the first defender would have met no second defender, so the press could '
                 'have been bypassed in one action.',
                 'Beating the first defender would have opened the way to goal, with no second defender to step in.',
                 'With the presser beaten, the next line would have had to step out unprepared.']},
    'hypo:no_pressure': {
        (3,): ['Without pressure, the ball carrier would have had time to pick the next action near our goal.',
               'With nobody going to the ball, the carrier could choose the next action calmly.',
               'Unpressed in our third, the carrier would have had time to look up and choose.'],
        (4, 5): ['Unpressed in our own third, the carrier could have picked the next action at leisure.',
                 'Without pressure, the carrier would have had time to choose the next action near our goal.',
                 'Left unpressed, the carrier could have taken the time to pick the next action.'],
        (6, 7): ['An unpressed carrier in our third could have dictated the next action.',
                 'Without first-defender pressure, the carrier would have had time to break our lines.',
                 'Unpressed, the carrier could have chosen the next action at leisure.']},
    'question:no_pressure': {(1, 2, 3): [
        'When the other team has the ball near our goal, who can go to the ball first?',
        'Who will be the first to go to the ball when the other team has it near our goal?',
        'When they have the ball near our goal, can the nearest player go to it straight away?']},
    'question:no_cover': {
        (1,): ['When one of us goes to the ball, who can stay close behind to help?',
               'Who can stand close behind the player who goes to the ball, ready to help?',
               'When a teammate goes to the ball, can the next player stay close behind to help?'],
        (2, 3): ['When one player goes to the ball, where could the nearest teammate stand to cover?',
                 'Who could stand a little behind the player who goes to the ball, ready to cover?',
                 'When a teammate puts pressure on the ball, where is the best place to cover them?']},
    'statement:no_pressure': {(4, 5, 6, 7): [
        'In our own third, the nearest defender engages the carrier first, and the others organise behind.',
        'The nearest defender closes the carrier down in our own third before the carrier can choose.',
        'Engage the carrier early in our own third; the nearest defender goes first.']},
    'statement:no_cover': {(4, 5, 6, 7): [
        'One player presses; the nearest teammate covers at an angle behind, close enough to step in.',
        'When the first defender presses, the second defender takes a covering position behind at an angle.',
        'Pressure and cover go together: the presser engages, the nearest teammate covers the space behind.']},
    # level 1: the praise is the explanation's; the action names one next step only (COACH L5.7: "Keep it up: going
    # to the ball together is the next step" praised and set a step in one breath)
    'build_on': {(1,): ['A next step: a teammate staying close behind the player who goes to the ball.',
                        'Next, a second player can stay close behind to help.',
                        'Something to add next time: one goes to the ball, and a teammate stays close to help.'],
                 (2, 3): ['To build on it, keep the covering player at an angle behind the one who presses.',
                          'To build on it, try the same pressure and cover when the ball is on the other side.',
                          'Next step: the same pressure and cover a little higher up the pitch.'],
                 (4, 5, 6, 7): ['To build on it, keep the second defender at an angle that also blocks the way to '
                                'goal.', 'To build on it, hold the same pressure and cover further up the pitch.',
                                'To build on it, keep the distance between presser and cover short as play moves.']},
    'restore': {(1,): ('stand close behind to help', 'stood close behind to help'),
                (2, 3, 4, 5, 6, 7): ('cover behind the presser', 'covered behind the presser')},
    'review_q:no_pressure': {(1, 2, 3, 4, 5, 6, 7): ['someone could have gone to the ball sooner']},
    'review_q:no_cover': {(1,): ['someone could have stayed close behind to help'],
                          (2, 3, 4, 5, 6, 7): ['a teammate could have covered behind the presser']},
    # fix round 3 (FACTS F-6 = COACH F-3): the counted moments are every opportunity of the check (carrier unpressed,
    # pressed, or 3-5 m without closing: unclear), not "moments with pressure on the ball" / "pressing moments" (of
    # 151 so called, 100 had any pressed sample); the moments the text counts are the other team's attacks
    'noun:no_pressure': {(1,): ['moments near our goal'], (2, 3, 4, 5, 6, 7): ['defending moments']},
    'noun:no_cover': {(1,): ['moments going to the ball'], (2, 3, 4, 5, 6, 7): ['defending moments']},
    'noun': {(1,): ['moments going to the ball'], (2, 3, 4, 5, 6, 7): ['defending moments']},
}

BANK['weak_side_balance'] = {
    'headline_fail': {
        (1, 2, 3): ['Protecting the far side when the ball is wide', 'The far side with the ball wide',
                    '{try}: someone goal-side on the far side'],
        (4, 5): ['No defender goal-side on the far side', 'Far-side protection with the ball wide',
                 'The far side was left open when the ball was wide'],
        (6, 7): ['Far-side balance with the ball wide', 'No far-side defender against a switch',
                 'Far-side balance against a switch']},
    'headline_ok': {(1, 2, 3): ['Protecting the far side', 'Someone goal-side on the far side',
                                'The far side stayed protected'],
                    (4, 5, 6, 7): ['Far-side protection with the ball wide', 'Balance on the far side',
                                   'The far side stayed protected']},
    'headline_describe': {(1, 2, 3, 4, 5, 6, 7): ['The far side with the ball wide', 'Far-side positions']},
    'short_fail': {(1, 2, 3): ['Far side open', 'Nobody on the far side'], (4, 5, 6, 7): ['Far side open',
                                                                                       'No far-side defender']},
    'short_ok': {(1, 2, 3, 4, 5, 6, 7): ['Far side protected', 'Far-side protection']},
    'short_describe': {(1, 2, 3, 4, 5, 6, 7): ['Far-side positions']},
    # Fix round 1 (critic B-5 / COACH B3, FACTS M7): goal-side is the tolerant count (deeper than the ball, or nearer
    # the own goal centre than it, beyond max(1 m, ball sigma + player sigma); a defender level with the ball is
    # unclear, not a failure), and the words say what was measured: "nobody on the far side was nearer our goal than
    # the ball", never "no defender stayed in the central corridor" (BC180's A-12 stood dead centre, 0.05 m short).
    'open_fail': {
        (1, 2, 3): ['{Phase}, with the ball {ball_side_place}, nobody on the far side was nearer our goal than the '
                    'ball.',
                    '{Phase}, the ball was {ball_side_place} and nobody on the far side was between the ball and our '
                    'goal.',
                    '{Phase}, with the ball {ball_side_place}, the far side was left open.'],
        (4, 5): ['{Phase}, with the ball {ball_side_place}, no defender on the far side was deeper than the ball or '
                 'nearer our goal than it.',
                 '{Phase}, the ball was {ball_side_place} and no defender on the far side was nearer our goal than the '
                 'ball, so the far side was left unprotected.',
                 '{Phase}, with the ball {ball_side_place}, nobody on the far side stood between the ball and our '
                 'goal.'],
        (6, 7): ['{Phase}, with the ball {ball_side_place}, there was no far-side defender goal-side of the ball.',
                 '{Phase}, the ball was {ball_side_place} and no defender on the far side was nearer our goal than the '
                 'ball.',
                 '{Phase}, with the ball {ball_side_place}, no defender on the far side was deeper than the ball or '
                 'nearer our goal than it.']},
    'open_ok': {
        (1, 2, 3): ['{Phase}, with the ball {ball_side_place}, someone stayed goal-side on the far side.',
                    '{Phase}, the ball was {ball_side_place} and {weak_n} defenders were goal-side on the far side.',
                    '{Phase}, with the ball {ball_side_place}, the far side stayed protected.'],
        (4, 5, 6, 7): ['{Phase}, with the ball {ball_side_place}, {weak_n} defenders were goal-side on the far side.',
                       '{Phase}, the ball was {ball_side_place} and the far side stayed protected.',
                       '{Phase}, with the ball {ball_side_place}, players tucked in on the far side.']},
    # descriptions (S-6): each level reads a different relation, not a paraphrase of one sentence
    'describe': {(1,): ['{Phase}, with the ball {ball_side_place}, the key moments show who was on the far side.'],
                 (2,): ['{Phase}, with the ball {ball_side_place}, {weak_count_words} stood on the far side between '
                        'the ball and our goal at the moment shown.'],
                 (3,): ['{Phase}, with the ball {ball_side_place}, the moment shown had {weak_count_words} on the far '
                        'side goal-side of the ball.',
                        '{Phase}, with the ball {ball_side_place}, the key moments show the far-side positions.'],
                 (4,): ['{Phase}, the ball was {ball_side_place}; at the moment shown {weak_count_words} on the far '
                        'side {weak_were} goal-side of it, and {central_def} in the central corridor.',
                        '{Phase}, the ball was {ball_side_place}; the key moments show the far-side positions.'],
                 (5,): ['{Phase}, with the ball {ball_side_place}, the far-side picture at the moment shown: '
                        '{weak_count_words} goal-side, and {open_space}.',
                        '{Phase}, with the ball {ball_side_place}, {weak_count_words} on the far side {weak_were} '
                        'goal-side at the moment shown.'],
                 (6,): ['{Phase}, with the ball {ball_side_place}, the far-side balance at the moment shown read '
                        '{weak_count_words} goal-side and {level_words} level with the ball.',
                        '{Phase}, with the ball {ball_side_place}, the far-side balance at the moment shown read '
                        '{weak_count_words} goal-side.'],
                 (7,): ['{Phase}, ball {ball_side_place}: against a switch the far-side structure at the moment shown '
                        'was {weak_count_words} goal-side, {level_words} level with the ball and {rec_words} running '
                        'back.',
                        '{Phase}, ball {ball_side_place}: against a switch the far-side structure at the moment shown '
                        'was {weak_count_words} goal-side.']},
    'context_fail': {(4,): ['{Central_def} in the central corridor {central_were} goal-side of the ball.',
                            'In the central corridor, {central_def} {central_were} nearer our goal than the ball.',
                            'Goal-side of the ball in the central corridor: {central_def}.'],
                     (5, 6, 7): ['{Central_def} in the central corridor {central_were} goal-side of the ball; on the '
                                 'far side, {open_space}.',
                                 'In the central corridor {central_def} {central_were} nearer our goal than the ball, '
                                 'and on the far side {open_space}.',
                                 'Goal-side in the central corridor: {central_def}; on the far side, {open_space}.']},
    'hypo': {(3,): ['If the ball had been switched to the far side, nobody was there to meet it.',
                    'A ball switched to the far side would have found open space.',
                    'If the ball had gone across, the far side would have been open.'],
             (4, 5): ['A switch of play to the far side would have found the space unprotected.',
                      'If the ball had been switched, the far side would have been unprotected.',
                      'A long ball to the far side could have found free space there.'],
             (6, 7): ['A switch at that moment could have exposed the far side.',
                      'If the ball had been switched, the far side would have been exposed with no defender to meet '
                      'it.',
                      'The far side could have been exposed to a switch.']},
    'question': {(1, 2, 3): ['When the ball is on one side, who could stay goal-side on the far side?',
                             'Who could tuck in on the far side when the ball is wide?',
                             'When the ball is wide, where could the far-side defender stand?']},
    'statement': {(4, 5, 6, 7): ['With the ball wide, the far-side defender tucks in goal-side.',
                                 'Keep a far-side defender goal-side when the ball is wide.',
                                 'The far-side defender moves in toward the middle, goal-side of the ball.']},
    'build_on': {(1, 2, 3, 4, 5, 6, 7): ['To build on it, keep the far-side defender goal-side as the ball moves.',
                                         'To build on it, keep that far-side position when the ball switches sides.',
                                         'To build on it, hold the same far-side position deeper in our half.']},
    'restore': {(1, 2, 3, 4, 5, 6, 7): ('protect the far side', 'protected the far side')},
    'review_q': {(1, 2, 3, 4, 5, 6, 7): ['a defender could have been goal-side on the far side']},
    'noun': {(1, 2, 3, 4, 5, 6, 7): ['moments with the ball wide']},
}

BANK['wide_defender_advance'] = {
    'headline_fail': {
        (1,): ['{try}: someone ready to protect our goal', '{try}: one player stays back when we attack',
               '{try}: someone back when we go forward'],
        (2,): ['{try}: someone stays back when a defender goes forward', '{try}: holding back when a defender '
               'attacks', '{try}: a teammate back when our defender moves up'],
        (3,): ['Covering the space when a defender goes forward', 'A defender went forward without cover behind',
               '{try}: covering behind a defender who goes forward'],
        (4,): ['Wide defender advanced without cover behind', 'The space behind an advancing full-back',
               'Unsupported advance by a wide defender'],
        (5,): ['Full-back advance not matched by cover behind', 'Unsupported full-back advance',
               'Cover behind an advancing full-back'],
        (6, 7): ['Full-back advance without rest-defence cover', 'Rest defence behind an advanced full-back',
                 'Unsupported full-back advance']},
    'headline_ok': {
        (1,): ['Someone ready to protect our goal', 'Someone stayed back, ready to protect our goal',
               'Ready to protect our goal'],
        (2,): ['Someone stayed back when a defender went forward', 'Holding back while a defender went forward',
               'Cover behind a defender going forward'],
        (3,): ['A defender went forward with cover behind', 'Covered forward runs by a defender',
               'Cover behind a defender going forward'],
        (4, 5, 6, 7): ['Supported advance by a wide defender', 'Wide defender advanced with cover behind',
                       'Covered full-back advance']},
    'headline_describe': {(1, 2, 3, 4, 5, 6, 7): ['A wide defender\'s forward runs', 'Forward runs on the flank']},
    'short_fail': {(1,): ['Someone back'], (2,): ['Someone stays back'], (3,): ['Cover behind a forward run'],
                   (4, 5, 6, 7): ['Unsupported advance', 'Advance without cover']},
    'short_ok': {(1, 2): ['Someone back'], (3,): ['Covered forward run'], (4, 5, 6, 7): ['Supported advance']},
    # Polish round (Oct 23): a strength's label and headline name the side when it is known (the other side's family
    # may be reviewed or praised in the same report); the plain ones above stay for a side PitchIQ could not read
    'short_ok_side': {(1, 2): [], (3,): ['Covered forward run on the {side}'],
                      (4, 5, 6, 7): ['Supported advance on the {side}']},
    'headline_ok_side': {
        (1, 2): [],
        (3,): ['A defender went forward on the {side} with cover behind', 'Covered forward runs on the {side}',
               'Cover behind a defender going forward on the {side}'],
        (4, 5, 6, 7): ['Supported advance by a wide defender on the {side}',
                       'Wide defender advanced on the {side} with cover behind', 'Covered full-back advance on the '
                       '{side}']},
    'short_describe': {(1, 2, 3, 4, 5, 6, 7): ['Wide defender forward runs']},
    'open_fail': {
        (1,): ['{Phase}, {place}, players moved forward together and nobody stayed back, ready to protect our goal.',
               '{Phase}, everyone went forward at times, and no one stayed back to protect our goal.',
               '{Phase}, {place}, we went forward with nobody back to protect our goal.'],
        (2,): ['{Phase}, {actor} moved up to help the attack, and nobody stayed back behind them.',
               '{Phase}, {place}, {actor} went forward, and no one stayed back behind them.',
               '{Phase}, {actor} joined the attack, and nobody stayed back to protect the space they left.'],
        (3,): ['{Phase}, {actor} moved {advance} ahead of our central defenders, and nobody covered the space they '
               'left on the {side}.',
               '{Phase}, {actor} went forward on the {side} while no teammate covered the space they left.',
               '{Phase}, {place}, {actor} advanced on the {side}, but the space they left was not covered.'],
        (4,): ['{Phase}, {actor} advanced {advance} beyond our central defenders while nobody was in the {side} '
               'channel behind them.',
               '{Phase}, {actor} was {advance} ahead of the back line with the {side} channel behind them empty.',
               '{Phase}, {place}, {actor} pushed up {advance} beyond the defensive line, and no teammate was in the '
               '{side} channel behind.',
               '{Phase}, {place}, {actor} pushed up beyond the back line, and no teammate was in the {side} channel '
               'behind.'],
        (5,): ['{Phase}, {actor} advanced {advance} beyond the centre-back line, and the {side} channel behind them '
               'stayed empty.',
               '{Phase}, {place}, {actor} pushed {advance} up the {side} with no teammate in the channel behind.',
               '{Phase}, {actor} advanced {advance} beyond the back line, and the run was not matched by cover in the '
               '{side} channel.',
               '{Phase}, {actor} advanced on the {side} beyond the back line, with no teammate in the channel behind.'],
        (6, 7): ['{Phase}, {actor} advanced {advance} beyond the back line and the {side} channel behind them stayed '
                 'empty.',
                 '{Phase}, {place}, {actor} held a position {advance} beyond the back line, with no teammate in the '
                 '{side} channel behind.',
                 '{Phase}, the advance by {actor} on the {side} was not matched by cover behind: nobody was in the '
                 'channel behind them.']},
    # Fix round 3 (critic B-1, ASSESS's request (a)): level 2 judges an advance left exposed (one of theirs in the
    # space the defender left, nobody covering it) wherever the ball is; with players behind the ball the "nobody
    # stayed back" wording was false and the gate withheld it (one_behind, channel_threat, stacking at level 2). The
    # level's own words: the space left, nobody covering it, one of theirs in it; never who should have covered.
    'open_fail:exposed': {
        (1,): [],
        (2,): ['{Phase}, {actor} went forward, nobody covered the space they left, and {opp_in} standing in it.',
               '{Phase}, {place}, {actor} joined the attack and the space they left was not covered: {opp_in} '
               'standing in it.',
               '{Phase}, {actor} moved up to help the attack; nobody covered the space they left, and {opp_in} there.'],
        (3, 4, 5, 6, 7): []},
    'headline_fail:exposed': {(1,): [],
                              (2,): ['{try}: covering the space when a defender goes forward',
                                     '{try}: a teammate in the space a defender leaves',
                                     '{try}: cover behind a defender who goes forward'],
                              (3, 4, 5, 6, 7): []},
    'short_fail:exposed': {(1,): [], (2,): ['Space behind a forward run'], (3, 4, 5, 6, 7): []},
    'question:exposed': {(1,): [],
                         (2,): ['When one of our defenders goes forward, who could cover the space they leave?',
                                'If our defender moves up, who can stand in the space behind them?',
                                'When a defender joins the attack, which teammate could fill the space they leave?'],
                         (3, 4, 5, 6, 7): []},
    'open_ok': {
        (1,): ['{Phase}, {place}, someone stayed back, ready to protect our goal.',
               '{Phase}, one of us stayed back while the others went forward, ready to protect our goal.',
               '{Phase}, {place}, someone was ready to protect our goal.'],
        (2,): ['{Phase}, {actor} went forward and someone stayed back behind them.',
               '{Phase}, {place}, {actor} joined the attack while a teammate stayed back.',
               '{Phase}, one of us stayed back as {actor} moved up.'],
        (3,): ['{Phase}, {actor} went forward on the {side} and {mate} covered the space they left.',
               '{Phase}, {place}, {actor} advanced and the space they left was covered.',
               '{Phase}, {actor} moved {advance} up the {side} while {mate} covered the space they left.'],
        (4, 5, 6, 7): ['{Phase}, {actor} advanced {advance} on the {side} and {mate} was in the channel behind them.',
                       '{Phase}, {place}, {actor} pushed up on the {side}; the advance was supported, with {mate} in '
                       'the {side} channel behind.',
                       '{Phase}, {actor} advanced on the {side} with cover behind: {mate} was in the channel behind.']},
    # Polish round (Oct 23): who provided the cover at the strength's first moment (d3_cover_by). 'open_ok' above is
    # the teammate in the channel; these say the near centre-back's shift, a midfielder's drop (levels 5-7, where
    # assess counts it as cover) or only that the space was covered (an interchange, said by the 'interchange' slot,
    # or a cover the measurements do not attribute). Before, every level 4-7 strength said "{mate} was in the
    # channel", and the gate withheld the strengths whose channel was empty while the centre-back covered (2 of the
    # 3 findings withheld in 12 reports of the cached Metrica game-1 and planted tiers, fix2/text).
    'open_ok:cb': {
        (1, 2): [],
        (3,): ['{Phase}, {actor} went forward on the {side} and the near centre-back moved across to cover the space '
               'they left.',
               '{Phase}, {place}, {actor} advanced on the {side}, and the near centre-back shifted across into the '
               'space they left.',
               '{Phase}, {actor} moved {advance} up the {side}; the near centre-back moved across to cover the space '
               'they left.'],
        (4, 5, 6, 7): ['{Phase}, {actor} advanced {advance} on the {side}, and the near centre-back moved across to '
                       'cover the space behind them.',
                       '{Phase}, {place}, {actor} pushed up on the {side}; the advance was supported, as the near '
                       'centre-back moved across behind them.',
                       '{Phase}, {actor} advanced on the {side} with cover behind: the near centre-back moved across '
                       'into the space they left.']},
    'open_ok:dm': {
        (1, 2, 3, 4): [],
        (5, 6, 7): ['{Phase}, {actor} advanced {advance} on the {side}, and {dm_mate} dropped toward the back line to '
                    'cover behind them.',
                    '{Phase}, {place}, {actor} pushed up on the {side}; the advance was supported, as {dm_mate} '
                    'dropped toward the back line.',
                    '{Phase}, {actor} advanced on the {side} with cover behind: {dm_mate} dropped toward the back '
                    'line.']},
    'headline_ok:covered': {(1,): [], (2,): ['Cover behind a defender going forward', 'A teammate covered the space a '
                                             'defender left', 'The space covered as a defender went forward'],
                            (3, 4, 5, 6, 7): []},
    'short_ok:covered': {(1,): [], (2,): ['Space behind a forward run covered'], (3, 4, 5, 6, 7): []},
    'open_ok:covered': {
        (1,): [],
        # fix round 3 (level 2 praises a covered advance wherever the ball is, critic B-1): "someone stayed back" only
        # with one of ours behind the ball at the moment (the claim), else these
        (2,): ['{Phase}, {actor} went forward and the space they left was covered.',
               '{Phase}, {place}, {actor} joined the attack, and a teammate covered the space they left.',
               '{Phase}, as {actor} moved up, the space they left was covered.'],
        (3,): ['{Phase}, {actor} went forward on the {side} and the space they left was covered.',
               '{Phase}, {place}, {actor} advanced on the {side} and the space they left was covered.',
               '{Phase}, {actor} moved {advance} up the {side}, and the space they left was covered.'],
        (4, 5, 6, 7): ['{Phase}, {actor} advanced {advance} on the {side}, and the advance was supported.',
                       '{Phase}, {place}, {actor} pushed up on the {side}, and the advance was supported.',
                       '{Phase}, {actor} advanced on the {side}; the advance was supported, with the space behind '
                       'them covered.']},
    'describe': {(1, 2, 3, 4, 5, 6, 7): ['{Phase}, {actor} advanced on the {side}; the moments are listed for '
                                         'context.']},
    # fix round 3 (FACTS F-1 = COACH F-2): {bm} the measured margin, "at least 10 m" ("well" without metres): "none
    # of our players were behind the ball" was said with 3 defenders 4.2 m goal-side of it
    'context_fail': {
        (2, 3): ['{behind} of our players {were} {bm} behind the ball at that moment.',
                 'At that moment {behind} of our players {were} {bm} behind the ball.',
                 '{Bm} behind the ball we had {behind} {players}.'],
        (4,): ['{behind} of our outfield players {were} {bm} behind the ball, and {opp_count} in the {side} channel '
               'in our half.', '{opp_count} in the {side} channel in our half at that moment.',
               '{Bm} behind the ball we had {behind} outfield {players}.'],
        (5,): ['{cb_dm_fail}',
               '{Cb} stayed central rather than moving across.',
               '{behind} of our outfield players {were} {bm} behind the ball, and {opp_count} in the {side} channel '
               'in our half.'],
        (6, 7): ['{cb_dm_fail}',
                 'Rest defence at that moment: {behind} of ours {bm} behind the ball against {threat} of theirs, a '
                 'margin of {margin}.',
                 '{opp_count} in the vacated {side} channel in our half.']},
    'context_ok': {(5, 6, 7): ['{cb_dm_ok}', 'Behind the run, {cb_dm_ok_low}', 'As the run went forward, '
                               '{cb_dm_ok_low}']},
    # Levels 5-7 state each relation they measured (critic S-10 / COACH M1: of 12 holding-midfielder drops carried at
    # level 6 none was said, of 12 six-to-centre-back spacings at level 7 none, of 42 rest-defence margins 9): the
    # centre-back / holding-midfielder sentence with the shift measured (cb_dm), the behind-the-ball count at level 5
    # (behind_ctx), the rest-defence margin at levels 6-7 (rest; its PitchIQ definition follows once per report), the
    # spacing at level 7 and the opponents in the vacated channel (opp_channel). Each slot is gated by level ([]).
    'cb_dm': {(1, 2, 3, 4): [], (5, 6, 7): ['{cb_dm_fail}']},
    # a strength at levels 5-7 also states the cover tests that did not provide the cover (S-10: 12 of 32 level-5
    # centre-back relations and 8 of 12 level-6 holding-midfielder drops of the covered scenes went unsaid)
    'context_also': {(1, 2, 3, 4): [], (5, 6, 7): ['{cb_dm_also}']},
    'behind_ctx': {(1, 2, 3, 4): [], (5,): ['At that moment {behind} of our outfield players {were} {bm} behind the '
                                           'ball.',
                                           '{Bm} behind the ball at that moment: {behind} of our outfield players.'],
                   (6, 7): []},
    'rest': {(1, 2, 3, 4, 5): [],
             (6, 7): ['Rest defence at that moment: {behind} of ours {bm} behind the ball against {threat} of theirs, '
                      'a margin of {margin}.',
                      'At that moment {behind} of ours {bm} behind the ball faced {threat} of theirs, a rest-defence '
                      'margin of {margin}.',
                      'Rest-defence count at that moment: {behind} of ours {bm} behind the ball, {threat} of theirs '
                      '{bm} on our side of it, margin {margin}.']},
    'opp_channel': {(1, 2, 3): [], (4, 5, 6, 7): ['{Opp_count} in the vacated {side} channel in our half.',
                                                  'In our half, {opp_count} in the {side} channel they had left.']},
    'interchange': {(3, 4, 5, 6, 7): ['A teammate took their place on the {side}.',
                                      'Someone else took the vacated spot on the {side}.',
                                      'Their spot on the {side} was taken by a teammate.']},
    'hypo': {
        (3,): ['If the ball had been lost at that moment, the space behind them on the {side} would have been open.',
               'If the ball had been lost there, nobody would have been ready on the {side}.',
               'Had the ball been lost, the {side} side behind them would have been open.'],
        (4,): ['If the ball had been lost, the {side} channel would have been unprotected.',
               'Had the ball changed hands there, the {side} channel behind would have been unprotected.',
               'If the ball had been lost at that moment, the space in the {side} channel would have been free.'],
        (5,): ['In a transition at that moment, the {side} channel could have been exposed.',
               'If possession had changed there, a transition into the {side} channel would have met no cover.',
               'A transition at that moment could have found the {side} channel exposed.'],
        (6,): ['If possession had changed there, the rest defence would have left the {side} channel exposed.',
               'In a transition, the {side} channel could have been exposed behind the advanced full-back.',
               'A transition at that moment could have exposed the {side} channel.'],
        (7,): ['A transition at that moment could potentially have used the vacated {side} channel as a route toward '
               'our goal.',
               'The vacated {side} channel was a potential transition route had possession changed.',
               'If possession had changed, the {side} channel could potentially have been the route into our half.']},
    'spacing': {(1, 2, 3, 4, 5, 6): [],
                # dm_holder: the confirmed holding midfielder's words, else "the nearest midfielder in a holding role"
                # (no role word singles out an unconfirmed player)
                (7,): ['The distance between {dm_holder} and the near centre-back was {dm_cb}.',
                       '{Dm_holder} and the near centre-back were {dm_cb} apart.',
                       'Spacing between {dm_holder} and the near centre-back: {dm_cb}.']},
    'question': {
        (1,): ['When we attack, who could stay back, ready to protect our goal?',
               'When everyone runs forward, who can stay ready to protect our goal?',
               'Who could stay back to protect our goal when we go forward?'],
        (2,): ['When one of our defenders goes forward, who could stay back?',
               'If our defender moves up, who can hold back behind them?',
               'When a defender joins the attack, which teammate could stay back?'],
        (3,): ['When our {side} defender goes forward, who could cover the space they leave?',
               'Who could fill the space behind a defender who goes forward?',
               'When a defender goes forward on the {side}, where could a teammate stand to cover?']},
    'statement': {(4, 5, 6, 7): [
        'Options: a later run by the full-back, or a teammate stepping into the {side} channel behind them.',
        'Either the run waits for cover, or a nearby teammate takes the {side} channel behind.',
        'The run is safer once someone holds the {side} channel behind: a centre-back across or a midfielder '
        'dropping.']},
    # levels 1-2: one concrete next step, not a generic "keep it up" (COACH L5 / criterion 13: 37 of the 396
    # verifier scene reports said "Keep it up")
    'build_on': {(1, 2): ['Next match, look for someone ready to protect our goal every time we attack.',
                          'A next step: one player staying back each time the others go forward.',
                          'Try it again next time: someone back whenever we go forward.'],
                 (3, 4, 5, 6, 7): ['To build on it, keep the cover behind as the run goes further.',
                                   'To build on it, use the same cover for runs on the other flank.',
                                   'To build on it, keep that cover when the ball moves inside.']},
    'restore': {(1, 2): ('stay back', 'stayed back'), (3,): ('cover the space behind', 'covered the space behind'),
                (4, 5, 6, 7): ('cover the {side} channel', 'covered the {side} channel')},
    'review_q': {(1, 2): ['someone could have stayed back'], (3,): ['a teammate could have covered the space they '
                                                                    'left'],
                 (4, 5, 6, 7): ['a teammate could have covered the {side} channel behind']},
    # the opportunities are a defender's forward runs, not attacks (COACH L5: "16 of the 16 attacks" were advances)
    'noun': {(1, 2): ['forward runs by our defenders'], (3,): ['forward runs'], (4, 5, 6, 7): ['advances']},
}

BANK['behind_ball'] = {
    'headline_fail': {
        (1,): ['{try}: someone ready to protect our goal', '{try}: one player stays back when we attack',
               '{try}: someone back when we go forward'],
        (2,): ['{try}: someone stays back when we attack', '{try}: holding someone back when we go forward',
               '{try}: a teammate back near our goal when we attack'],
        (3,): ['Players behind the ball when we attack', '{try}: enough players behind the ball',
               'Keeping players behind the ball'],
        (4, 5): ['Too few players behind the ball when attacking', 'Defensive shape behind the ball',
                 'Players behind the ball while attacking'],
        (6, 7): ['Rest defence behind the ball', 'Rest-defence numbers while attacking', 'Rest-defence margin']},
    'headline_ok': {(1,): ['Someone ready to protect our goal', 'Ready to protect our goal', 'Someone stayed back'],
                    (2,): ['Someone stayed back', 'Holding someone back', 'Someone back as we went forward'],
                    (3, 4, 5): ['Players behind the ball when attacking', 'Shape behind the ball',
                                'Enough players behind the ball'],
                    (6, 7): ['Rest defence behind the ball', 'Rest-defence margin held', 'Rest defence in place']},
    'headline_describe': {(1, 2, 3, 4, 5, 6, 7): ['Players behind the ball when attacking', 'Shape behind the ball']},
    'short_fail': {(1, 2): ['Someone back'], (3, 4, 5): ['Players behind the ball'], (6, 7): ['Rest defence']},
    'short_ok': {(1, 2): ['Someone back'], (3, 4, 5): ['Behind the ball'], (6, 7): ['Rest defence held']},
    'short_describe': {(1, 2, 3, 4, 5, 6, 7): ['Behind the ball']},
    'open_fail': {
        (1,): ['{Phase}, {place}, there were moments when nobody stayed back, ready to protect our goal.',
               '{Phase}, everyone went forward at times, and no one stayed back to protect our goal.',
               '{Phase}, {place}, we sometimes had nobody back to protect our goal.'],
        (2,): ['{Phase}, {place}, nobody stayed back near our goal.', '{Phase}, everyone went forward and no one '
               'stayed back.', '{Phase}, {place}, we had nobody back near our own goal.'],
        # fix round 3 (FACTS F-1): {bm} is the margin the count was measured with (behind_words)
        (3,): ['{Phase}, {place}, {only_behind} of our players {were} {bm} behind the ball; {required} is the aim at '
               'this level.', '{Phase}, {place}, there were fewer players {bm} behind the ball than the {required} '
               'this level asks for.', '{Phase}, {place}, {only_behind} of our players {were} {bm} behind the ball.'],
        (4, 5): ['{Phase}, {place}, {only_behind} of our outfield players {were} {bm} behind the ball, against the '
                 '{required} this level expects.',
                 '{Phase}, {place}, there were fewer players {bm} behind the ball than the {required} expected here.',
                 '{Phase}, {place}, {only_behind} of our outfield players {were} {bm} behind the ball.'],
        (6, 7): ['{Phase}, {place}, {only_behind} of our outfield players {were} {bm} behind the ball against '
                 '{threat} of theirs {bm} on our side, a rest-defence margin of {margin}.',
                 '{Phase}, {place}, the rest defence had a margin of {margin}: {behind} of ours {bm} behind the ball, '
                 '{threat} of theirs.',
                 '{Phase}, {place}, {only_behind} of our outfield players {were} {bm} behind the ball.']},
    'open_fail:advance': {
        (1, 2): ['{Phase}, {place}, one of our defenders went forward and nobody stayed back, ready to protect our '
                 'goal.',
                 '{Phase}, a defender joined the attack at times, and no one stayed back to protect our goal.',
                 '{Phase}, {place}, we went forward with a defender, and nobody stayed back to protect our goal.']},
    # Fix round 3 (critic B-1, the spec's EARLY DEVELOPMENT example; ASSESS's request (a)): level 1's reframed advance
    # fails wherever the ball is when one of theirs stood in the space the defender left and nobody of ours was in
    # it; "nobody stayed back" was false with players behind the ball (one_behind, channel_threat, stacking: the gate
    # withheld them). Team-level, no numbers, no "cover" (a level-2 concept).
    'open_fail:exposed': {
        (1,): ['{Phase}, one of our defenders went forward, nobody filled the space behind them, and {opp_in} '
               'standing in it.',
               '{Phase}, {place}, a defender joined the attack and none of our players was in the space behind them, '
               'while {opp_in} there.',
               '{Phase}, when one of our defenders went forward, {opp_in} in the space they left, and none of our '
               'players was there.'],
        (2, 3, 4, 5, 6, 7): []},
    'headline_fail:exposed': {(1,): ['{try}: filling the space when a defender goes forward',
                                     '{try}: a teammate in the space a defender leaves',
                                     '{try}: someone in the space behind a defender who goes forward'],
                              (2, 3, 4, 5, 6, 7): []},
    'short_fail:exposed': {(1,): ['Space behind a forward run'], (2, 3, 4, 5, 6, 7): []},
    'question:exposed': {(1,): ['When one of our defenders goes forward, who could fill the space they leave?',
                                'Who could stand in the space behind a defender who goes forward?',
                                'When a defender goes forward, where could a teammate stand to help?'],
                         (2, 3, 4, 5, 6, 7): []},
    'open_ok:advance': {
        (1, 2): ['{Phase}, {place}, when a defender went forward, someone stayed back, ready to protect our goal.',
                 '{Phase}, a defender joined the attack while one of us stayed back, ready to protect our goal.',
                 '{Phase}, {place}, someone stayed back as a defender went forward.']},
    # fix round 3 (ASSESS's request (b)): level 1 praises a covered advance wherever the ball is (a teammate in the
    # space left, the near central defender across, or a swap): "someone stayed back" only with one of ours behind
    # the ball at the moment, else these (BC60 B and the user's run 636e03 B were withheld: behindN 0, channel held)
    'headline_ok:advance_covered': {(1,): ['Looking after the space behind a defender', 'A teammate in the space a '
                                           'defender left', 'Someone in the space behind a defender'],
                                    (2, 3, 4, 5, 6, 7): []},
    'short_ok:advance_covered': {(1,): ['A teammate in the space left'], (2, 3, 4, 5, 6, 7): []},
    'open_ok:advance_covered': {
        (1,): ['{Phase}, {place}, when one of our defenders went forward, a teammate looked after the space they left.',
               '{Phase}, a defender joined the attack and a teammate took care of the space behind them.',
               '{Phase}, {place}, as one of our defenders went forward, a teammate was there for the space they left.'],
        (2, 3, 4, 5, 6, 7): []},
    'open_ok': {
        (1,): ['{Phase}, {place}, someone stayed back, ready to protect our goal.',
               '{Phase}, as players went forward, one of us stayed back, ready to protect our goal.',
               '{Phase}, {place}, someone was ready to protect our goal.'],
        # fix round 3 (FACTS F-1): one of ours at least the margin behind the ball is "back", not "near our goal" (the
        # ball is in the final third: such a player can stand 60 m from our goal)
        (2,): ['{Phase}, {place}, someone stayed back, {bm} behind the ball.', '{Phase}, one of us stayed back while '
               'the others attacked.', '{Phase}, {place}, a player stayed back.'],
        (3, 4, 5): ['{Phase}, {place}, {behind} of our players stayed {bm} behind the ball.',
                    '{Phase}, {place}, we kept enough players {bm} behind the ball ({behind}).',
                    '{Phase}, {place}, enough players stayed {bm} behind the ball.'],
        (6, 7): ['{Phase}, {place}, the rest defence held a margin of {margin}: {behind} of ours {bm} behind the '
                 'ball against {threat} of theirs.',
                 '{Phase}, {place}, we kept enough players {bm} behind the ball, a rest-defence margin of {margin}.',
                 '{Phase}, {place}, {behind} of our outfield players stayed {bm} behind the ball.']},
    # a description's moment may be anywhere (the verdicts are judged in the final third only), so it names no place
    # (FACTS L8); each level reads its own relation (S-6)
    'describe': {(1,): ['{Phase}, the key moments show who stayed near our goal.'],
                 (2, 3, 4): ['{Phase}, at the moment shown {behind} of our outfield players {were} {bm} behind the '
                             'ball.',
                             '{Phase}, the key moments show who was behind the ball.'],
                 (5,): ['{Phase}, at the moment shown {behind} of our outfield players {were} {bm} behind the ball, '
                        'with {threat} of theirs {bm} on our side of it.',
                        '{Phase}, the key moments show who was behind the ball.'],
                 (6,): ['{Phase}, the count at the moment shown was {behind} of ours {bm} behind the ball against '
                        '{threat} of theirs, a rest-defence margin of {margin}.',
                        '{Phase}, the key moments show the rest-defence numbers behind the ball.'],
                 (7,): ['{Phase}, rest-defence structure at the moment shown: margin {margin}, from {behind} {bm} '
                        'behind the ball and {threat} of theirs {bm} on our side.',
                        '{Phase}, the key moments show the structure behind the ball.']},
    'context_fail': {(5,): ['At that moment {threat} of their {threat_players} {threat_were} {bm} on our side of the '
                            'ball.',
                            '{Bm} on our side of the ball the other team had {threat} {threat_players}.',
                            'Their players {bm} on our side of the ball at that moment: {threat}.'],
                     (6, 7): []},
    'hypo': {(3,): ['If the ball had been lost at those moments, few players would have been ready near our goal.',
                    'Had the ball been lost, the way to our goal would have been open.',
                    'If the ball had been lost, there would have been little protection near our goal.'],
             (4,): ['If the ball had been lost, the team would not have been set behind the ball.',
                    'Had the ball changed hands there, our goal would have been protected by too few.',
                    'If the ball had been lost, the space behind would have been unprotected.'],
             (5,): ['In a transition at that moment, too few players would have been behind the ball.',
                    'If possession had changed, the team would have defended the transition short of numbers.',
                    'A transition then could have caught too few players behind the ball.'],
             (6, 7): ['A transition at that moment could have exposed the rest defence.',
                      'If possession had changed, the rest defence would have been short against their forwards.',
                      'Had possession changed, the margin could have exposed our goal.']},
    'question': {(1,): ['When we attack, who could stay back, ready to protect our goal?',
                        'When everyone runs forward, who can stay ready to protect our goal?',
                        'Who could stay back to protect our goal when we go forward?'],
                 (2, 3): ['When we attack, who could stay back?', 'Who can hold back near our goal while we attack?',
                          'When we go forward, which players could stay behind the ball?']},
    'statement': {(4, 5, 6, 7): ['Keep the planned number of players behind the ball while we attack.',
                                 'While attacking, hold enough players behind the ball for the moment it changes '
                                 'hands.', 'Leave enough players behind the ball before more join the attack.']},
    # levels 1-2: one concrete next step, not a generic "keep it up" (COACH L5 / criterion 13: 37 of the 396
    # verifier scene reports said "Keep it up")
    'build_on': {(1, 2): ['Next match, look for someone ready to protect our goal every time we attack.',
                          'A next step: one player staying back each time the others go forward.',
                          'Try it again next time: someone back whenever we go forward.'],
                 (3, 4, 5, 6, 7): ['To build on it, keep the same numbers behind the ball deeper into attacks.',
                                   'To build on it, hold the same shape behind the ball when we attack quickly.',
                                   'To build on it, keep it when we attack down either flank.']},
    'restore': {(1, 2): ('stay back', 'stayed back'),
                (3, 4, 5, 6, 7): ('add a player behind the ball', 'added a player behind the ball')},
    'review_q': {(1, 2): ['someone could have stayed back'], (3, 4, 5, 6, 7): ['more players could have stayed behind '
                                                                               'the ball']},
    'noun': {(1, 2, 3): ['attacks'], (4, 5, 6, 7): ['attacking moments']},
    'noun:advance': {(1, 2, 3, 4, 5, 6, 7): ['forward runs by our defenders']},     # level 1's reframed advances
}

BANK['defensive_spacing'] = {
    'headline_fail:lateral_gap': {(1, 2, 3): ['A gap between defenders on the ball side',
                                              'Space between our defenders near the ball',
                                              'Defenders spread apart near the ball'],
                                  (4, 5): ['Gaps in the back line near the ball', 'Ball-side gap in the back line',
                                           'A gap between defenders on the ball side'],
                                  (6, 7): ['Ball-side gap in the block', 'Gap in the back line on the ball side',
                                           'Block spacing on the ball side']},
    'headline_fail:dm_gap': {(1, 2, 3): ['Space between our defenders and midfield',
                                         'Distance from our defenders to midfield', 'Defenders far from midfield'],
                             (4, 5): ['Distance between the back line and midfield', 'Back line and midfield too '
                                      'far apart', 'Space between our defenders and midfield'],
                             (6, 7): ['Space between the lines', 'Distance between the back line and midfield',
                                      'Block length between the lines']},
    'headline_ok': {(1, 2, 3): ['Defenders moved as a unit', 'Defenders level and close to midfield',
                                'Defenders moving together'],
                    (4, 5, 6, 7): ['Defenders moved as a unit', 'An even back line close to midfield',
                                   'Compact spacing between the lines']},
    'headline_describe': {(1, 2, 3): ['Spacing of our defenders', 'Where our defenders stood'],
                          (4, 5, 6, 7): ['Spacing of our defenders', 'Back-line spacing']},
    'short_fail:lateral_gap': {(1, 2, 3, 4, 5, 6, 7): ['Ball-side gap', 'Gap near the ball']},
    'short_fail:dm_gap': {(1, 2, 3): ['Defenders far from midfield'],
                          (4, 5, 6, 7): ['Space between the lines', 'Lines too far apart']},
    'short_ok': {(1, 2, 3): ['Defenders moving together'], (4, 5, 6, 7): ['Even, compact back line', 'Unit spacing']},
    'short_describe': {(1, 2, 3): ['Spacing of our defenders'], (4, 5, 6, 7): ['Back-line spacing']},
    'open_fail:lateral_gap': {
        (1, 2, 3): ['{Phase}, a gap of {gap} opened between {actor} and {mate} on the ball side.',
                    '{Phase}, {place}, our defenders left a wide gap on the ball side.',
                    '{Phase}, {place}, a wide gap opened between our defenders on the side of the ball.'],
        (4, 5): ['{Phase}, a gap of {gap} opened between {actor} and {mate} on the ball side.',
                 '{Phase}, {place}, our defenders left a wide gap on the ball side, between {actor} and {mate}.',
                 '{Phase}, {place}, a wide gap opened in the back line on the side of the ball.'],
        (6, 7): ['{Phase}, {place}, a gap of {gap} opened in the block between {actor} and {mate}, on the ball side.',
                 '{Phase}, the back line left a large gap on the ball side between {actor} and {mate}.',
                 '{Phase}, {place}, a wide gap opened on the ball side of the block.']},
    'open_fail:dm_gap': {
        (1, 2, 3): ['{Phase}, {place}, our defenders stood {dm_gap} from our midfielders.',
                    '{Phase}, there was a large space between our defenders and midfield.',
                    '{Phase}, {place}, our defenders were {dm_gap} behind our midfielders.'],
        (4, 5): ['{Phase}, {place}, our back line was {dm_gap} from midfield.',
                 '{Phase}, there was a large space between our defenders and midfield.',
                 '{Phase}, {place}, the back line stood {dm_gap} behind midfield.'],
        (6, 7): ['{Phase}, {place}, the back line stood {dm_gap} behind midfield, a long block.',
                 '{Phase}, there was a large space between the back line and midfield.',
                 '{Phase}, {place}, our defenders were {dm_gap} from midfield.']},
    'open_ok': {(1, 2, 3): ['{Phase}, our defenders stayed level with each other, close to our midfielders.',
                            '{Phase}, {place}, our defenders stayed level with each other and moved together.',
                            '{Phase}, {place}, our defenders moved across together and stayed level with each '
                            'other.'],
                (4, 5, 6, 7): ['{Phase}, {place}, the back line stayed level with a spread of {line_std}, and the '
                               'lines stayed close.',
                               '{Phase}, our defenders stayed level with each other and the lines stayed close.',
                               '{Phase}, {place}, the back line was even and kept the distance between the lines '
                               'short.']},
    # descriptions (S-6): level 3 the distance to midfield, level 4 that and the widest gap, level 5 the line's
    # spread in depth, level 6 the block height, level 7 the opponents standing between the lines; the first
    # fillable phrasing of a level is used (BC180 B's D5 description repeated across levels at .48)
    'describe': {(1, 2): ['{Phase}, the key moments show where our defenders stood next to each other.'],
                 (3,): ['{Phase}, at the moment shown our defenders stood {dm_gap} from our midfielders.',
                        '{Phase}, at the moment shown the widest space between neighbouring defenders was '
                        '{gap_any}.',
                        '{Phase}, the key moments show where our defenders stood next to each other.'],
                 (4,): ['{Phase}, at the moment shown the back line was {dm_gap} from midfield, and its widest gap '
                        'measured {gap_any}.',
                        '{Phase}, the distance from the back line to midfield was {dm_gap} at the moment shown.',
                        '{Phase}, at the moment shown the widest gap in the back line was {gap_any}.',
                        '{Phase}, the key moments show where the back line stood.'],
                 (5,): ['{Phase}, the back line stood {dm_gap} behind midfield at the moment shown, its players '
                        'within {line_std} of each other in depth.',
                        '{Phase}, the back line stood {dm_gap} behind midfield at the moment shown.',
                        '{Phase}, at the moment shown the widest gap in the back line measured {gap_any}.',
                        '{Phase}, the key moments show the back line and midfield.'],
                 (6,): ['{Phase}, it was a {block} block at the moment shown, {dm_gap} long from the back line to '
                        'midfield.',
                        '{Phase}, the block from the back line to midfield measured {dm_gap} at the moment shown.',
                        '{Phase}, the key moments show the block from the back line to midfield.'],
                 # fix round 3 (FACTS F-11): dmGapOpponents counts the ball's lateral half only (contextual)
                 (7,): ['{Phase}, between our back line and midfield at the moment shown: {dm_gap}, with {between_n} '
                        'of theirs standing in that space on the ball\'s side.',
                        '{Phase}, between our back line and midfield at the moment shown: {dm_gap}.',
                        '{Phase}, the key moments show the space between our back line and midfield.']},
    'context_fail': {(1, 2, 3, 4, 5, 6, 7): ['An opponent was inside that gap.', 'One of their players stood in the '
                                             'gap.', 'The gap held an opponent.']},
    'block': {(6, 7): ['The team defended in a {block} block.', 'It was a {block} block at the time.',
                       'Block height at the time: a {block} block.']},
    'spacing': {(5, 6, 7): ['The back line was also {dm_gap} from midfield.',
                            'Midfield stood {dm_gap} ahead of the back line at the time.',
                            'The distance from the back line to midfield was {dm_gap}.']},
    'hypo': {(4, 5): ['A ball into that gap would have reached a free opponent between our defenders.',
                      'An opponent in that gap could have been found between our defenders.',
                      'If the ball had gone into the gap, our defenders would have been split.'],
             (6, 7): ['A ball into that gap could have split the back line.',
                      'The gap could have been exploited with a run between the defenders.',
                      'A runner into that gap would have been hard to pick up.']},
    'statement': {(4, 5, 6, 7): ['The defenders slide across together and close the gap nearest the ball.',
                                 'Keep the gaps nearest the ball small as the line shifts.',
                                 'Shift as a unit toward the ball; the gap nearest the ball closes first.']},
    'statement:dm_gap': {(4, 5, 6, 7): ['The back line steps up behind midfield, so the space between the lines '
                                        'stays small.',
                                        'Keep the back line and midfield close together as the block moves.',
                                        'When the ball goes forward, the back line follows midfield up the pitch.']},
    'question:dm_gap': {(1, 2, 3): ['How can our defenders stay close behind our midfielders?',
                                    'When the ball moves forward, can our defenders step up behind our midfielders?',
                                    'Who can help our defenders stay close to our midfielders when the ball goes '
                                    'forward?']},
    'hypo:dm_gap': {(4, 5): ['A ball into the space between the lines would have found room in front of our '
                             'defenders.',
                             'An opponent receiving between the lines would have had room to turn.',
                             'If the ball had been played between the lines, a receiver would have had room there.'],
                    (6, 7): ['A ball into the space between the lines could have found a free receiver.',
                             'The space between the lines could have let an opponent receive and turn.',
                             'A receiver between the lines could have turned toward our goal.']},
    'question': {(1, 2, 3): ['When the ball moves across, how can our defenders move across together?',
                             'Who closes the space next to the defender nearest the ball?',
                             'How can our defenders stay close to each other near the ball?']},
    'build_on': {(1, 2, 3, 4, 5, 6, 7): ['To build on it, keep the same spacing when the ball switches sides.',
                                         'To build on it, hold the spacing as the block steps up.',
                                         'To build on it, keep the line level as play speeds up.']},
    'restore': {(1, 2, 3, 4, 5, 6, 7): ('close the gap', 'closed the gap')},
    'review_q': {(1, 2, 3, 4, 5, 6, 7): ['the gap near the ball could have been closed']},
    'review_q:dm_gap': {(1, 2, 3): ['our defenders could have stayed closer to our midfielders'],
                        (4, 5, 6, 7): ['the back line could have stayed closer to midfield']},
    'noun': {(1, 2, 3, 4, 5, 6, 7): ['defending moments']},
}

BANK['attacking_shape'] = {
    'headline_fail': {(1,): ['{try}: spread out wide and deep', '{try}: spreading out when we have the ball',
                             '{try}: using the whole pitch when we have the ball'],
                      (2,): ['{try}: spreading out when we have the ball', '{try}: wide and deep when we attack',
                             '{try}: using the whole pitch when we attack'],
                      (3,): ['Width and depth when we have the ball', '{try}: width and depth in attack',
                             'Spreading out when we have the ball'],
                      (4, 5, 6, 7): ['Width and depth in attack', 'Attacking width and depth', 'Spreading the attack']},
    'headline_fail:width': {(1,): ['{try}: someone wide on each side', '{try}: spreading out wide',
                                   '{try}: using both sides of the pitch'],
                            (2,): ['{try}: players on both sides', '{try}: spreading out wide when we attack',
                                   '{try}: someone wide on each side'],
                            (3,): ['Using the width when we have the ball', '{try}: using the width in attack',
                                   'Spreading wide when we have the ball'],
                            (4, 5, 6, 7): ['Width in attack', 'Attacking width', 'Using the width of the pitch']},
    'headline_fail:depth': {(1,): ['{try}: tall and deep when we have the ball', '{try}: spreading out from back to '
                                   'front', '{try}: someone high and someone deep'],
                            (2,): ['{try}: spreading out from back to front', '{try}: tall and deep when we attack',
                                   '{try}: using the pitch from back to front'],
                            (3,): ['Depth when we have the ball', '{try}: depth in attack',
                                   'Spreading from back to front with the ball'],
                            (4, 5, 6, 7): ['Depth in attack', 'Attacking depth', 'Height and depth with the ball']},
    # fix round 3 (FACTS F-11): never "the whole pitch" for a shape judged by the length available (BC180 covered
    # 24-35% of the pitch's length against a low block)
    'headline_ok': {(1,): ['Spreading out', 'Spread out wide and deep', 'Spreading out as a team'],
                    (2,): ['Spreading out with the ball', 'Wide and deep in attack', 'Spread out wide and deep'],
                    (3,): ['Width and depth with the ball', 'A wide, deep attacking shape',
                           'Using the width and depth'],
                    (4, 5, 6, 7): ['Width and depth in attack', 'A stretched attacking shape', 'Height, width and '
                                   'depth']},
    'headline_describe': {(1, 2, 3, 4, 5, 6, 7): ['Attacking shape', 'Team shape with the ball']},
    # the phase unknown (amateur and Veo clips without marks): no word of who had the ball (critic B-4)
    'headline_describe:unknown': {(1, 2): ['How far the team spread out', 'How the team spread out'],
                                  (3, 4, 5, 6, 7): ['Team shape when the whole team was in view',
                                                    'How far the team spread out']},
    'short_fail': {(1, 2): ['Spreading out'], (3, 4, 5, 6, 7): ['Width and depth']},
    'short_fail:width': {(1, 2): ['Spreading out wide'], (3, 4, 5, 6, 7): ['Width in attack']},
    'short_fail:depth': {(1, 2): ['Tall and deep'], (3, 4, 5, 6, 7): ['Depth in attack']},
    'short_ok': {(1, 2): ['Spreading out'], (3, 4, 5, 6, 7): ['Width and depth']},
    'short_describe': {(1, 2, 3, 4, 5, 6, 7): ['Attacking shape']},
    'short_describe:unknown': {(1, 2, 3, 4, 5, 6, 7): ['Team spread', 'Team shape in view']},
    'open_fail': {(1, 2, 3, 4, 5, 6, 7): ['{Phase}, the team did not use the whole pitch at times.',
                                         '{Phase}, the team did not spread across the whole pitch at times.',
                                         '{Phase}, at times the team did not use all of the pitch.']},
    'open_fail:width': {
        (1,): ['{Phase}, we sometimes bunched up in the middle instead of spreading out wide.',
               '{Phase}, players sometimes came close together near the ball instead of spreading out wide.',
               '{Phase}, we sometimes stayed narrow, close together in the middle.'],
        (2,): ['{Phase}, we stayed narrow at times, with players close together near the ball.',
               '{Phase}, we bunched up in the middle at times and stayed narrow.',
               '{Phase}, at times the team was narrow, with players close together.'],
        (3,): ['{Phase}, the team stayed narrow, {width} wide.',
               '{Phase}, the team stayed narrow, with players close together near the ball.',
               '{Phase}, at times the team was narrow: {width} wide.'],
        (4, 5, 6, 7): ['{Phase}, the team was narrow, {width} wide.',
                       '{Phase}, the team stayed narrow, using {width_share} of the width.',
                       '{Phase}, the team stayed narrow at {width} across.',
                       '{Phase}, the team stayed narrow, close together near the ball.']},
    'open_fail:depth': {
        (1,): ['{Phase}, we sometimes stayed short, with little space from back to front.',
               '{Phase}, we did not spread out from back to front at times.',
               '{Phase}, at times we had little space from back to front.'],
        (2,): ['{Phase}, the team stayed short at times, with little space from back to front.',
               '{Phase}, we did not spread out from back to front at times.',
               '{Phase}, at times the team was short from back to front.'],
        (3,): ['{Phase}, the team stayed short, {depth} from back to front.',
               '{Phase}, the team was short from back to front, {depth} from the last to the first player.',
               '{Phase}, at times the team had little depth: {depth} from back to front.',
               '{Phase}, the team stayed short, with little space from back to front.'],
        (4, 5, 6, 7): ['{Phase}, the team was short from back to front: {depth} from the deepest to the highest '
                       'player.',
                       '{Phase}, the team stayed short, using {depth_share} of the pitch\'s length.',
                       '{Phase}, the shape stayed short at {depth} from back to front.',
                       '{Phase}, the team stayed short, with little space from back to front.']},
    'open_fail:shape': {
        (1,): ['{Phase}, we sometimes stayed narrow and close together, and the team stayed short.',
               '{Phase}, at times we stayed narrow, and we had little space from back to front.',
               '{Phase}, we sometimes bunched up in the middle and stayed short.'],
        (2,): ['{Phase}, we stayed narrow and short at times, with players close together near the ball.',
               '{Phase}, at times the team was narrow, with little space from back to front.',
               '{Phase}, we bunched up in the middle at times and stayed short.'],
        (3,): ['{Phase}, the team stayed narrow and short: {width} wide and {depth} from back to front.',
               '{Phase}, at times the team stayed narrow, {width} wide, with little space from back to front.',
               '{Phase}, the team was narrow and had little depth at times: {width} wide and {depth} long.',
               '{Phase}, at times the team stayed narrow and short, close together near the ball.'],
        (4, 5, 6, 7): ['{Phase}, the team was narrow and short: {width} wide and {depth} from back to front.',
                       '{Phase}, the shape stayed narrow and short, using {width_share} of the width and '
                       '{depth_share} of the length.',
                       '{Phase}, the team stayed narrow, {width} wide, and stayed short at {depth} from back to '
                       'front.',
                       '{Phase}, the team stayed narrow and short, close together near the ball.']},
    'open_fail:channels': {
        (1,): ['{Phase}, one side was sometimes empty.', '{Phase}, nobody stood on one side at times.',
               '{Phase}, at times nobody stood on one side of the pitch.'],
        (2,): ['{Phase}, one side was empty at times, with {channels} of the {channels_total} lanes used.',
               '{Phase}, one side was empty at times.', '{Phase}, at times one side stayed empty.'],
        (3,): ['{Phase}, one side was empty at times, with {channels} of the three lanes across the pitch used.',
               '{Phase}, one side was empty at times.',
               '{Phase}, at times one side stayed empty, and the team used only part of the width.'],
        (4, 5, 6, 7): ['{Phase}, one side was empty: the team used {channels} of the {channels_total} channels.',
                       '{Phase}, one side was empty at times.',
                       '{Phase}, one side stayed empty at times, with {channels} of the {channels_total} channels '
                       'used.']},
    'open_ok': {
        (1,): ['{Phase}, we spread out wide and deep, with players on the left, in the middle and on the right.',
               '{Phase}, we spread out wide, with someone on each side.',
               '{Phase}, we spread out tall and deep, with players on the left, in the middle and on the right.'],
        (2,): ['{Phase}, we spread wide and deep, with players on the left, in the middle and on the right.',
               '{Phase}, we spread out wide, using all {channels_total} lanes of the pitch.',
               '{Phase}, players spread out wide, and someone stayed wide on each side.'],
        (3,): ['{Phase}, the team spread out wide, {width} wide and {depth} long.',
               '{Phase}, the team was {width} wide, with players on the left, in the middle and on the right.',
               '{Phase}, we spread wide and deep, with players on the left, in the middle and on the right.'],
        (4, 5, 6, 7): ['{Phase}, the team was {width} wide and {depth} long, with players in all '
                       '{channels_total} channels.',
                       '{Phase}, the team used {width_share} of the pitch\'s width and {depth_share} of its length.',
                       '{Phase}, the team spread wide, using {width_share} of the pitch\'s width.',
                       '{Phase}, the team spread wide and deep, with players in all {channels_total} channels.']},
    # descriptions (critic S-6, W8 miss 1: levels 5 and 7 opened with the same sentence, Jaccard .51-.80 on 29 of 30
    # scene variants and every run copy): each level reads a different relation, the first fillable phrasing of the
    # level is used. Level 1 where to look, level 2 the lanes used (a count), level 3 rounded metres, level 4 metres
    # and channels, level 5 the shares of the pitch, level 6 the shape against the opponents' block (the length used
    # up to their last line, critic B-7), level 7 the height against their last line and the support behind the ball.
    'describe': {
        (1,): ['{Phase}, the key moments show how far we spread out.'],
        (2,): ['{Phase}, at the moment shown we had players in {channels_of} lanes across the '
               'pitch.',
               '{Phase}, the key moments show how wide and how far up the pitch we spread.'],
        (3,): ['{Phase}, at the moment shown the team stood {width} wide and {depth} from back to front.',
               '{Phase}, the key moments show how far the team spread out.'],
        (4,): ['{Phase}, the shape at the moment shown was {width} wide and {depth} long, with players in '
               '{channels_of} channels.',
               '{Phase}, the shape at the moment shown was {width} wide and {depth} long.',
               '{Phase}, the key moments show the width and length of the shape.'],
        (5,): ['{Phase}, at the moment shown the team occupied {width_share} of the pitch\'s width and '
               '{depth_share} of its length.',
               '{Phase}, the key moments show the share of the pitch the team occupied.'],
        (6,): ['{Phase}, measured against their block at the moment shown, our shape used {avail_used} up to their '
               'last line and {width_share} of the width.',
               '{Phase}, against their block at the moment shown our highest player was {height_gap_words}.',
               '{Phase}, at the moment shown the team occupied {width_share} of the width and {depth_share} of the '
               'pitch\'s length.'],
        # fix round 3 (FACTS F-1, critic spot check): {deep_n} is None when nobody of ours was the margin behind the
        # ball, i.e. the ball was with our own back line (build-up, not rest defence: "none of ours behind the ball"
        # was said with 1-4 outfielders 0-8 m goal-side of it on 5 of 5 real clips)
        (7,): ['{Phase}, at the moment shown our highest player was {height_gap_words}, with {deep_n} of ours {bm} '
               'behind the ball.',
               '{Phase}, at the moment shown our highest player was {height_gap_words}.',
               '{Phase}, height and support at the moment shown: {deep_n} of ours {bm} behind the ball.',
               '{Phase}, the moment shown spread over {width_share} of the width and {depth_share} of the length.']},
    'describe_unknown': {
        (1,): ['When the whole team was in view, the key moments show how far we spread out.'],
        (2,): ['When the whole team was in view, our players stood in {channels_of} lanes across '
               'the pitch.',
               'The key moments, with the whole team in view, show how wide and how far up the pitch we spread.'],
        (3,): ['When the whole team was in view, the shape was {width} wide and {depth} from back to front.',
               'The key moments, with the whole team in view, show how far the team spread out.'],
        (4,): ['With everyone in view at the moment shown, the team spread {width} across and {depth} long, in '
               '{channels_of} channels.',
               'With everyone in view at the moment shown, the team spread {width} across and {depth} long.'],
        (5,): ['With the whole team in view, the moment shown covered {width_share} of the width and {depth_share} '
               'of the length.',
               'The key moments, with the whole team in view, show the share of the pitch it covered.'],
        (6,): ['Whole team in view: against their block the shape used {avail_used} up to their last line.',
               'Whole team in view: the shape covered {width_share} of the width and {depth_share} of the pitch\'s '
               'length.'],
        (7,): ['Whole team in view: our highest player was {height_gap_words}.',
               'Whole team in view: the shape spread over {width_share} of the width, {depth_share} of the length.']},
    'question': {(1,): ['When we have the ball, can someone stay wide on each side?',
                        'When we have the ball, who can stand wide on the left and on the right?',
                        'Can we spread out so someone is wide on each side when we have the ball?'],
                 (2, 3): ['When we have the ball, who could stay wide on each side?',
                          'How could we spread out so there is a player on each side?',
                          'When we attack, where could the players stand to make the pitch big?']},
    'question:depth': {(1,): ['When we have the ball, can someone stay high and someone stay back?',
                              'When we have the ball, who can stand far up and who can stay back?',
                              'Can someone stay far up the pitch when we have the ball?'],
                       (2, 3): ['When we have the ball, who could stay high and who could stay deep?',
                                'How could we spread out from back to front when we attack?',
                                'When we have the ball, where could players stand to use the pitch from back to '
                                'front?']},
    'statement': {(4, 5, 6, 7): ['Hold one player wide on each side and one high, so the shape stretches the block.',
                                 'Keep the touchlines occupied and one player high when we have the ball.',
                                 'Stretch the shape: a wide player on each side, depth behind the ball.']},
    'statement:depth': {(4, 5, 6, 7): ['Keep one player high and the deepest players back, so the shape stretches '
                                       'the block from back to front.',
                                       'Stretch the shape from back to front: a high runner and support behind the '
                                       'ball.',
                                       'Use the length of the pitch: one player high, the deepest players behind the '
                                       'ball.']},
    'build_on': {(1, 2): ['Next match, look for us spreading out like this every time we get the ball.',
                          'A next step: someone wide on both sides as soon as we have the ball.',
                          'Try it again next time: spread out quickly once we have the ball.'],
                 (3, 4, 5, 6, 7): ['To build on it, keep the width when the ball comes back to the middle.',
                                   'To build on it, hold the width while adding depth behind the ball.',
                                   'To build on it, keep the shape when we attack quickly.']},
    'restore': {(1, 2, 3, 4, 5, 6, 7): ('spread the shape', 'spread the shape')},
    'review_q': {(1, 2, 3, 4, 5, 6, 7): ['the team could have spread wider']},
    'review_q:depth': {(1, 2, 3, 4, 5, 6, 7): ['the team could have spread further from back to front']},
    'review_q:shape': {(1, 2, 3, 4, 5, 6, 7): ['the team could have spread wider and further from back to front']},
    'review_q:channels': {(1, 2, 3, 4, 5, 6, 7): ['someone could have stood on the empty side']},
    'noun': {(1, 2, 3, 4, 5, 6, 7): ['attacking moments']},
}

# Generic slots (every detector). '{noun}' comes from the detector's bank.
BANK['*'] = {
    'recurrence_fail': {
        # fix round 3 (COACH F-11): level 1 has no numbers, but "a few times" was said for 10-24 moments: {often}
        # says roughly how many (often_words: once / a few / several / many times)
        (1,): ['This came up {often}.', 'It happened {often}.', 'We saw this {often}.'],
        (2,): ['This happened {fail_n} times out of {of_n} {noun}.', 'We saw it in {fail_n} of {of_n} {noun}.',
               '{fail_n} of the {of_n} {noun} looked like this.'],
        (3,): ['It happened in {fail_n} of {of_n} {noun}, in {spells} separate spells.',
               'Over the clip: {fail_n} of {of_n} {noun}, in {spells} separate spells.',
               '{fail_n} of {of_n} {noun} showed the same picture, in {spells} separate spells.'],
        (4,): ['In {phase_time} {phase_poss}, {fail_n} of {of_n} {noun} looked like this, in {spells} separate '
               'spells.', 'It showed in {fail_n} of {of_n} {noun} ({spells} separate spells).',
               'Across {phase_time} {phase_poss}: {fail_n} of {of_n} {noun}, {spells} separate spells.'],
        # a share of 0 or 1 and a rate from less than RATE_MIN_PHASE_S are not given (COACH L5: "about 100% of the
        # 16 advances", "about 42.5 per 10 minutes" from 3.8 minutes): the count phrasings then fill
        (5,): ['That was {fail_share} of the {of_n} {noun}, in {spells} separate spells.',
               'It applied to {fail_share} of the {of_n} {noun} ({spells} separate spells).',
               '{fail_n} of {of_n} {noun} ({fail_share}) looked like this, in {spells} separate spells.',
               'It applied to {fail_n} of the {of_n} {noun}, in {spells} separate spells.',
               '{fail_n} of {of_n} {noun} looked like this.'],
        (6, 7): ['That is about {rate} times per {per} minutes {poss_of}, in {spells} separate spells.',
                 'Rate: about {rate} per {per} minutes {poss_of}; {fail_n} of {of_n} {noun} over {spells} spells.',
                 '{fail_n} of {of_n} {noun}, about {rate} per {per} minutes {poss_of}.',
                 'Over {phase_time} {phase_poss}: {fail_n} of {of_n} {noun}, in {spells} separate spells.',
                 '{fail_n} of {of_n} {noun} in {phase_time} {phase_poss}.']},
    # Fix round 3 (FACTS F-6 = COACH F-3): a strength's count is of the moments PitchIQ could JUDGE (demonstrated +
    # not demonstrated), as assess's strength rule counts it, and the unclear ones are said ({unc_tail}): "2 of the
    # 15 moments with pressure on the ball looked like this" was a strength the rule decided on 2 of 3 (12 unclear),
    # "It held in 6 of 13 defending moments (about 46%)" one of 6 of 6. {dem_of}: "18 of the 26" / "all 3".
    'recurrence_ok': {
        (1,): ['It happened {often}.', 'We did this {often}.', 'We saw this {often}.'],
        (2,): ['We did this in {dem_of} {noun} PitchIQ could judge{unc_tail}.',
               'It showed in {dem_of} {noun} PitchIQ could judge{unc_tail}.',
               '{Dem_of} {noun} PitchIQ could judge looked like this{unc_tail}.'],
        (3, 4): ['It showed in {dem_of} {noun} PitchIQ could judge, in {dem_spells} separate spells{unc_tail}.',
                 '{Dem_of} judged {noun} looked like this, in {dem_spells} separate spells{unc_tail}.',
                 'Over the clip: {dem_of} judged {noun}, {dem_spells} separate spells{unc_tail}.'],
        (5,): ['That was {dem_share} of the {dec_n} {noun} PitchIQ could judge, in {dem_spells} separate '
               'spells{unc_tail}.',
               'It held in {dem_share} of the {dec_n} judged {noun} ({dem_spells} separate spells){unc_tail}.',
               '{Dem_of} judged {noun} ({dem_share}), in {dem_spells} separate spells{unc_tail}.',
               'It held in {dem_of} judged {noun}, in {dem_spells} separate spells{unc_tail}.'],
        (6, 7): ['It held in {dem_of} judged {noun}, over {dem_spells} separate spells{unc_tail}.',
                 '{Dem_of} judged {noun} ({dem_share}) showed it, in {dem_spells} separate spells{unc_tail}.',
                 'Shown in {dem_spells} separate spells: {dem_of} judged {noun}{unc_tail}.']},
    'one_instance': {
        (1, 2, 3): ['We saw this once, so it is something to check rather than a conclusion.',
                    'It was seen once only, so take it as something to look at.',
                    'This showed up once, so it is listed for a look only.'],
        (4, 5, 6, 7): ['This was one observed instance, so it is listed for review only.',
                       'Only one instance was seen; treat it as a moment to check, not a conclusion.',
                       'One instance only: worth a look on the video, not a verdict.']},
    'moment': {(1, 2, 3, 4, 5, 6, 7): [
        'The clearest example is at {t}.', 'See {t} for the clearest example.', 'The first key moment is at {t}.',
        'Watch the key moment at {t}.', 'It shows clearly at {t}.', 'Have a look at {t}.',
        'The key moment to watch is at {t}.', 'One example to watch is at {t}.', 'Start the video at {t}.',
        'The moment at {t} shows it best.', 'Jump to {t} to see it.', 'The video at {t} has an example.',
        'For an example, go to {t}.', 'You can see one at {t}.', 'The first instance is at {t}.',
        'Look at {t} first.', 'An example starts around {t}.', 'Check {t} on the video.']},
    'limits': {(1, 2, 3): ['Not sure here: {limits}.', 'What we could not check: {limits}.',
                           'This stays a look rather than a conclusion because we could not check {limits}.'],
               (4, 5, 6, 7): ['Not confirmed: {limits}.', 'What could not be confirmed: {limits}.',
                              'It stays a review because these were not confirmed: {limits}.']},
    # per level (S-6: the shared "this is a description, not a judgement" carried most of the cross-level overlap)
    'limits_describe': {(1, 2): ['We could not check {limits}, so this is a description only.',
                                 'This is a description only: we could not check {limits}.'],
                        (3,): ['A description only, because {limits} could not be checked.',
                               'Because {limits} could not be checked, it is only described.'],
                        (4,): ['Not confirmed: {limits}; this is a description, not a judgement.',
                               'This describes what was measured; not confirmed: {limits}.'],
                        (5,): ['It is described, not judged: {limits} could not be confirmed.',
                               'Described only, with {limits} unconfirmed.'],
                        (6,): ['It stays descriptive; unconfirmed: {limits}.',
                               'Descriptive only, as {limits} could not be confirmed.'],
                        (7,): ['Read as context, not as a verdict; unconfirmed: {limits}.',
                               'Context only, unconfirmed: {limits}.']},
    # a strength withheld because failures were as common as in professional play (assess 'mixed_evidence', critic
    # B-2): said, per level, so the coach knows why a shown behaviour is not praised
    # Fix round 3 (LEAD decision (3), COACH F-5): assess withholds a strength only when the failures are significantly
    # above the level's base (one-sided binomial tail at baseTestAlpha), the base scaled below level 5; "at or above
    # the professional base rate" is no longer the rule, and youth and levels 1-4 never read "professional"
    # ({base_words}: base_words_for). {base_share}: measurements.baseShare, the base tested.
    'mixed_evidence': {(1,): ['It went both ways in this clip, so it is described here rather than praised.'],
                       (2,): ['Some moments showed it and some did not, so it is only described.'],
                       (3,): ['It was shown in {dem_n} of the {decided} moments PitchIQ could judge and not shown in '
                              '{mixed_fail_n}, more often than this level expects, so it is described rather than '
                              'counted as a strength.'],
                       (4,): ['With {mixed_fail_n} of the {decided} judged moments not showing it, more than '
                              '{base_words} explains, it is described rather than counted as a strength.'],
                       (5,): ['Not shown in {mixed_fail_share} of the judged moments, more than {base_words} explains, '
                              'so it is not counted as a strength.',
                              'The moments that did not show it ({mixed_fail_n} of {decided}) are more than '
                              '{base_words} explains, so it is not counted as a strength.'],
                       (6,): ['Mixed evidence: not shown in {mixed_fail_share} of the judged moments, more than a base '
                              'of {base_share} ({base_src}) explains, so no strength is claimed.',
                              'Mixed evidence: {mixed_fail_n} of {decided} judged moments did not show it, more than '
                              '{base_words} explains, so no strength is claimed.'],
                       (7,): ['Absent in {mixed_fail_share} of the {decided} judged moments, more than a base of '
                              '{base_share} ({base_src}) explains: context, not a strength.',
                              'More than {base_words} explains, {mixed_fail_n} of {decided} judged moments lacked it: '
                              'context, not a strength.']},
    # an opportunity inside a change of formation was left out (assess 'formation_changing', critic S-3)
    'formation_changing': {(1, 2, 3): ['Moments while the team was changing its shape were left out.',
                                       'Some moments came while the team was changing shape and were left out.'],
                           (4, 5, 6, 7): ['Moments inside a change of formation were left out.',
                                          'Spells that fell within a formation change were not counted.']},
    # strengths from level 4: the judged play they rest on (critic B-2: "3 of 3 attacking moments" rested on 0.2, 2.0
    # and 0.8 s of judged play)
    'judged': {(1, 2, 3): [], (4,): ['PitchIQ judged {judged} of play in these spells.'],
               (5,): ['These spells give {judged} of judged play.'],
               (6, 7): ['Judged play behind it: {judged}.']},
    'level_cap': {(1, 2, 3): ['It is listed to look at because this level keeps the list of things to try short.',
                              'This level keeps the list of things to try short, so this one is listed to look at.'],
                  (4, 5, 6, 7): ['It met the bar for a work item, but this level shows at most {max_items}, so it is '
                                 'listed for review.',
                                 'This level shows at most {max_items} work items; this one met the bar and is listed '
                                 'for review.',
                                 'It met the bar for a work item; this level keeps the list short, so it is listed for '
                                 'review.']},
    'no_moment': {(1, 2, 3, 4, 5, 6, 7): ['No single video moment showed it clearly, so it is described without a '
                                          'moment to watch.']},
    'review_action': {(1, 2, 3): ['Watch the key moments and decide whether {q}.',
                                  'Have a look at the key moments to judge whether {q}.',
                                  'The key moments will show whether {q}.'],
                      (4, 5, 6, 7): ['Check the key moments to judge whether {q}.',
                                     'The key moments are the place to judge whether {q}.',
                                     'Worth checking on the video whether {q}.']},
    'counterfactual_zone': {
        (1, 2): ['A teammate moving to the spot marked on the diagram could have {restored}, if nobody else moved.',
                 'The diagram marks a spot where a teammate could have {restored}, with everyone else staying put.',
                 'If a teammate had moved to the marked spot, and only them, they could have {restored}.'],
        (3,): ['{Mover} could have moved into the area marked on the diagram to {restore}, assuming only they moved.',
               '{Mover} had time to reach the marked area, where they could have {restored}; this assumes everyone '
               'else stayed still.',
               'The diagram marks where {mover} could have stood to {restore}, with everyone else where they were.'],
        (4, 5): ['A possible area for {mover} is marked on the diagram; from there they could have {restored}, '
                 'assuming nobody else moved.',
                 '{Mover} could have reached the marked area in time and {restored}, a one-player estimate with the '
                 'others held still.',
                 'The marked area shows where {mover} could have {restored}; the estimate moves only that player.'],
        (6, 7): ['{Mover} reaching the marked zone would have {restored} (a single-mover estimate).',
                 'The marked zone was reachable for {mover} and would have {restored}, assuming the rest held still.',
                 'A single-mover estimate puts {mover} in the marked zone, where they would have {restored}.']},
    'counterfactual_direction': {
        (1, 2, 3): ['The arrow on the diagram shows a direction a teammate could have moved to {restore}, if nobody '
                    'else moved.',
                    'The arrow shows which way a teammate could have gone to {restore}, with everyone else staying '
                    'put.',
                    'Moving the way the arrow points, and only that player, a teammate could have {restored}.'],
        (4, 5, 6, 7): ['The arrow shows the direction {mover} could have moved to {restore}, assuming nobody else '
                       'moved; the area was too large to mark a spot.',
                       'The arrow points where {mover} could have gone to {restore} (a single-mover estimate; the area '
                       'was too large to pin down).',
                       'A move by {mover} in the direction of the arrow could have {restored}, with the others held '
                       'still.']},
    'counterfactual_self': {
        (3, 4, 5, 6, 7): ['{Actor} holding a deeper position, marked on the diagram, could have {restored}, assuming '
                          'nobody else moved.',
                          'A slightly deeper position for {actor}, marked on the diagram, could have {restored} with '
                          'the others held still.',
                          'The marked area shows where {actor} could have held back to {restore}; only that player is '
                          'moved in the estimate.']},
    'style': {(6, 7): ['With the {style} style you selected, {effect}.', 'Because the selected style is {style}, '
                       '{effect}.', 'For the {style} style chosen in the setup: {effect}.']},
    'review_prefix': {(1, 2, 3, 4, 5, 6, 7): ['Possible issue to review: ']},
    'no_alternative': {(1, 2, 3): ['{no_alt_plain}'], (4, 5, 6, 7): ['{no_alt}']},
    # 'Moments worth a look' (fix round 2, LEAD decision (b); levels 3-7, assess gives none at levels 1-2): the clear
    # moments among the opportunities PitchIQ could judge, stated as a count of them (levels 5-7 also as a share),
    # then the judged ones that met the check and that this stays below the level's verdict bar (true of every such
    # review: assess gives one only when the failing share of the opportunities did not reach minFailureShare). The
    # first phrasing that fills is used (ordered slots), so the share is shown whenever the level shows shares.
    # {mw_what} is the detector's mode-free 'moments_what' (the cited moments may differ in failure mode).
    # {mw_of}: "3 of the 11" ("all 3" when every judged one was a clear moment); {mw_of_short}: "3 of 11" / "All 3"
    # Fix round 3 (FACTS F-4 = COACH F-10): the count is of CLEAR moments, and says so at every level ("In 3 of the
    # 17 defending moments PitchIQ could judge, pressure ... were not both there" read as if 3 failed, while 8 had);
    # the failures that were not clear are counted by moments_more, so clear + less clear + met = judged.
    'moments_count': {
        (1, 2, 3): ['In {mw_of} {mw_noun} PitchIQ could judge, it was clear that {mw_what}.',
                    'In a few of the {mw_noun} PitchIQ could judge, it was clear that {mw_what}.'],  # number-free
        (4,): ['PitchIQ found {mw_of} judged {mw_noun} to be clear cases, in {mw_spells} separate spells: {mw_what}.',
               'PitchIQ found {mw_of} judged {mw_noun} to be clear cases: {mw_what}.'],
        # level 5 apart from level 4 too (the D5 level 4 / 5 pair of one planted family read .488 with a shared start)
        (5,): ['PitchIQ could judge {mw_decided} {mw_noun}; {mw_of} ({mw_share}) were clear moments, in {mw_spells} '
               'separate spells: {mw_what}.',
               'PitchIQ could judge {mw_decided} {mw_noun}; {mw_of} were clear moments, in {mw_spells} separate '
               'spells: {mw_what}.',
               'PitchIQ could judge {mw_decided} {mw_noun}; {mw_of} were clear moments: {mw_what}.'],
        (6,): ['{mw_of_short} judged {mw_noun} were clear ({mw_share}), over {mw_spells} separate spells: {mw_what}.',
               '{mw_of_short} judged {mw_noun} were clear, over {mw_spells} separate spells: {mw_what}.',
               '{mw_of_short} judged {mw_noun} were clear: {mw_what}.'],
        # level 7 is worded apart from level 6 (S-6: one family's texts at two levels stay under .5 trigram Jaccard;
        # the shared level 6-7 phrasings gave .674 on the low_block scene's behind-the-ball review)
        (7,): ['Clear in {mw_of} {mw_noun} PitchIQ could judge ({mw_share}), across {mw_spells} distinct spells: '
               '{mw_what}.',
               'Clear in {mw_of} {mw_noun} PitchIQ could judge, across {mw_spells} distinct spells: {mw_what}.',
               'Clear in {mw_of} {mw_noun} PitchIQ could judge: {mw_what}.']},
    # the judged moments that fell short without being clear (failingMoments - clearMoments), so the counts add up
    'moments_more': {(1, 2, 3): ['{Mw_more} more fell short, less clearly, and {mw_more_are} not cited.'],
                     (4,): ['In {mw_more} more it fell short less clearly; {mw_more_those} not cited.'],
                     (5,): ['{Mw_more} more fell short less clearly and {mw_more_are} not among the moments shown.'],
                     (6,): ['A further {mw_more} fell short less clearly and {mw_more_are} left out of the clips.'],
                     (7,): ['Less clear shortfalls, not cited: {mw_more}.']},
    'moments_unclear': {(1, 2, 3, 4): ['{Mw_unclear_more} could not be judged clearly.'],
                        (5,): ['{Mw_unclear_more} {mw_unclear_was} too unclear to judge.'],
                        (6,): ['{Mw_unclear_more} stayed unclear and {mw_unclear_is} left out of the count.'],
                        (7,): ['Left out as unclear: {mw_unclear_n}.']},
    'moments_frame': {
        (1, 2, 3): ['The team did what the check asks in {mw_dem} of the {mw_decided}, and this stays below the bar '
                    'for a verdict at this level: these are moments to look at together.',
                    'This stays below the bar for a verdict at this level: these are moments to look at together.'],
        (4,): ['The team did what the check asks in {mw_dem} of the {mw_decided}; this stays below the level\'s bar '
               'for a verdict, so these are moments worth a look rather than a conclusion.',
               'This stays below the level\'s bar for a verdict, so these are moments worth a look rather than a '
               'conclusion.'],
        (5,): ['The check was met in {mw_dem} of the {mw_decided} ({mw_dem_share}); below the level\'s bar for a '
               'verdict, these are moments worth a look, not a conclusion.',
               'The check was met in {mw_dem} of the {mw_decided}; below the level\'s bar for a verdict, these are '
               'moments worth a look, not a conclusion.',
               'Below the level\'s bar for a verdict, these are moments worth a look, not a conclusion.'],
        (6,): ['The check was met in {mw_dem} of the {mw_decided} ({mw_dem_share}): below the bar for a verdict at '
               'this level, so these moments are for the video session, with no conclusion drawn.',
               'The check was met in {mw_dem} of the {mw_decided}: below the bar for a verdict at this level, so '
               'these moments are for the video session, with no conclusion drawn.',
               'Below the bar for a verdict at this level, these moments are for the video session, with no '
               'conclusion drawn.'],
        (7,): ['With the check met in {mw_dem} of the {mw_decided} ({mw_dem_share}), this sits under the level\'s '
               'verdict bar: context for the analysis, not a conclusion.',
               'With the check met in {mw_dem} of the {mw_decided}, this sits under the level\'s verdict bar: context '
               'for the analysis, not a conclusion.',
               'This sits under the level\'s verdict bar: context for the analysis, not a conclusion.']},
    # {mw_shown}: how many of the clear moments are cited, only when fewer than all (at most MAX_TIMESTAMPS: "In all 5
    # ... The moments to watch are at" three times read as if those three were all of them)
    'moments_times': {(1, 2, 3, 4): ['{Mw_shown} of them are shown, at {mw_times}.',
                                     'The moments to watch are at {mw_times}.'],
                      (5,): ['Shown here: {mw_shown} of them, at {mw_times}.', 'They are at {mw_times}.'],
                      (6,): ['Clips to review ({mw_shown} of them): {mw_times}.', 'Clips to review: {mw_times}.'],
                      (7,): ['The clips ({mw_shown} shown): {mw_times}.', 'The clips: {mw_times}.']},
    'moments_first': {(1, 2, 3, 4): ['In the first of them, '], (5,): ['The first of them: '],
                      (6,): ['In the first clip, '], (7,): ['First clip: ']},
    # the minimal plan's action (no role, relation or concept word the gate could refuse)
    'look_for_plain': {(1, 2, 3, 4, 5, 6, 7): ['In each clip, look at where our nearest players stood around the '
                                               'ball and the space behind them.']},
    'moments_why': {(1, 2, 3, 4, 5, 6, 7): ['Why it matters: ']},
    'limits_moments': {(1, 2, 3): ['What we could not check: {limits}.'], (4, 5, 6, 7): ['Not confirmed: {limits}.']},
}
# Per detector, for 'moments worth a look' reviews: moments_what says what the clear moments had in common in words
# that hold whatever the failure mode of each (contextual.sample_outcomes: a pressure-and-cover sample fails with
# nobody near the carrier OR a presser without cover, so "the ball carrier was left unpressed" would be false for some
# cited moments; the cited moments of one review mixed both modes on Metrica game 1, F/A/whole 1153 / 3163 / 4603 s),
# and passes the polarity gate (no relation phrase it cannot read); short_moments the caption of every cited moment
# on the same rule (detectors with failure modes); noun_moments the opportunities counted (all of the family's, of
# every mode); look_for the action: what to look at in each clip, at the level's concepts, never a prescription.
MOMENTS_BANK = {
    'pressure_cover': {
        'noun_moments': {(1, 2, 3, 4, 5, 6, 7): ['defending moments']},
        'short_moments': {(1, 2, 3): ['Pressure and cover'], (4, 5, 6, 7): ['Pressure and cover on the ball']},
        'moments_what': {(1, 2, 3): ['pressure on the ball and cover behind it were not both there'],
                         (4, 5, 6, 7): ['pressure on the ball and cover behind the first defender were not both in '
                                        'place']},
        'look_for': {
            (1, 2, 3): ['In each clip, look at who was nearest to the ball carrier, how close they got, and where the '
                        'next teammate stood behind them.'],
            (4,): ['In each clip, watch the first defender and the nearest teammate: how close the first defender got '
                   'to the ball carrier, and how far behind, and at what angle, the teammate stood.'],
            (5,): ['In each clip, check the first defender\'s distance to the ball carrier and where the second '
                   'defender stood: near enough behind, and at an angle, to step in.'],
            (6, 7): ['In each clip, read the first defender\'s distance to the carrier, the second defender\'s distance '
                     'and angle behind, and how the rest of the unit stood around the ball.']}},
    'weak_side_balance': {
        'short_moments': {(1, 2, 3, 4, 5, 6, 7): ['Far-side defenders']},
        'moments_what': {(1, 2, 3, 4, 5, 6, 7): ['the far side had fewer defenders between the ball and our goal '
                                                 'than the check asks for']},
        'look_for': {
            (1, 2, 3): ['In each clip, look at the far side when the ball was wide: how many of our defenders stood '
                        'between the ball and our goal, and how much space was open toward the far touchline.'],
            (4, 5): ['In each clip, count the far-side defenders between the ball and our goal, and the open space '
                     'toward the far touchline.'],
            (6, 7): ['In each clip, read the far-side defenders\' depth and distance from the ball, and how much space '
                     'a switch of play would find.']}},
    'wide_defender_advance': {
        # the side named: the other side's family may be praised in the same report ("covered forward run")
        'short_moments': {(1, 2, 3): ['Cover behind a forward run on the {side}', 'Cover behind a forward run'],
                          (4, 5, 6, 7): ['Advance without cover on the {side}', 'Advance without cover']},
        'moments_what': {(1, 2, 3): ['the space behind the defender who went forward was left without cover'],
                         (4, 5, 6, 7): ['the space behind the advancing defender was left without cover']},
        'look_for': {
            (1, 2, 3): ['In each clip, look at the space the defender left when going forward: who was nearest to it, '
                        'and whether a central defender or a midfielder could have moved across to cover it.'],
            (4,): ['In each clip, watch the {side} channel behind the advancing defender: who was nearest to it, and '
                   'whether the near centre-back or a midfielder could have covered it in time.',
                   'In each clip, watch the channel behind the advancing defender: who was nearest to it, and whether '
                   'the near centre-back or a midfielder could have covered it in time.'],
            # "a holding midfielder" (a role in general): "the holding midfielder" names a role the claims may not
            # have confirmed (the gate refused it on Metrica game 1, window 2850 s, team B at level 5)
            (5,): ['In each clip, check who could have filled the {side} channel: the near centre-back moving across, '
                   'a holding midfielder dropping, or a teammate already there.',
                   'In each clip, check who could have filled the channel: the near centre-back moving across, a '
                   'holding midfielder dropping, or a teammate already there.'],
            (6, 7): ['In each clip, read how the {side} channel was covered behind the advance (the near centre-back '
                     'across, a holding midfielder dropping or a teammate in it) and the rest-defence margin left '
                     'behind the ball.',
                     'In each clip, read how the channel was covered behind the advance (the near centre-back across, '
                     'a holding midfielder dropping or a teammate in it) and the rest-defence margin left behind the '
                     'ball.']}},
    'behind_ball': {
        # fix round 3 (FACTS F-1): {bm} the margin the count was measured with ("at least 10 m" / "well")
        'moments_what': {(1, 2, 3, 4, 5, 6, 7): ['fewer than {mw_required} of our players were {bm} behind the ball',
                                                 'fewer of our players were {bm} behind the ball than the check asks '
                                                 'for']},
        # the level 6-7 margin rule (params.behindRule 'margin' on 9v9 / 11v11): never a required count
        'moments_what:margin': {(1, 2, 3, 4, 5): ['our players {bm} behind the ball, against their forwards, were '
                                                  'fewer than the check asks for'],
                                (6, 7): ['the rest-defence margin was below what the check asks for']},
        'look_for': {
            (1, 2, 3): ['In each clip, count how many of our players were {bm} behind the ball, and look at where the '
                        'other team\'s forwards were.'],
            (4, 5): ['In each clip, count our players {bm} behind the ball against the opponents who stayed forward, '
                     'and look at the space behind our last defender.'],
            (6, 7): ['In each clip, read the rest-defence margin: our players {bm} behind the ball against their '
                     'forwards, and the space behind our last line.']}},
    'defensive_spacing': {
        'noun_moments': {(1, 2, 3, 4, 5, 6, 7): ['defending moments']},
        'short_moments': {(1, 2, 3): ['Spacing of our defenders'], (4, 5, 6, 7): ['Back-line spacing']},
        'moments_what': {(1, 2, 3): ['an opponent stood in a large space between our defenders, or between our '
                                     'defenders and midfielders'],
                         (4, 5, 6, 7): ['an opponent stood in a large gap in our back line, or between it and '
                                        'midfield']},
        'look_for': {
            (1, 2, 3): ['In each clip, look at the space between the defenders nearest the ball, the distance from our '
                        'defenders to our midfielders, and where the nearest opponent stood.'],
            (4, 5): ['In each clip, watch the gaps in the back line near the ball and the distance from the back line '
                     'to midfield as the ball moved, and who was free in that space.'],
            (6, 7): ['In each clip, read the back line\'s spacing near the ball and its distance to midfield against '
                     'the block height you want, and which opponents were free between the lines.']}},
    'attacking_shape': {
        'noun_moments': {(1, 2, 3, 4, 5, 6, 7): ['attacking moments']},
        'short_moments': {(1, 2, 3, 4, 5, 6, 7): ['Width and depth']},
        'moments_what': {(1, 2, 3, 4, 5, 6, 7): ['the team was less spread out than the check asks for']},
        'look_for': {
            (1, 2, 3): ['In each clip, look at how wide and how deep the team stood, and whether each side of the '
                        'pitch had a player.'],
            (4, 5): ['In each clip, check the width and the length of our shape, and whether each of the three '
                     'channels had a player.'],
            (6, 7): ['In each clip, read how far our shape stretched the opposing block in width and length, and how '
                     'high our highest player stood against their last line.']}},
}
for _det, _slots_of in MOMENTS_BANK.items():
    BANK[_det].update(_slots_of)
del _det, _slots_of
# Why no suggested area is drawn (NoAlternative.reason, ZONES), plain and technical; spec Part 8: the annotation then
# shows the observed problem and the principle only.
NO_ALT_MIN_LEVEL = 3        # levels 1-2 never explain why no area is drawn (COACH L5: engine internals for a U8 coach)
NO_ALT_WORDS = {
    'no_reachable_fix': ('No teammate could have reached a better spot in time, so no spot is marked.',
                         'No teammate could have reached a position that restores it in time, so no area is marked.'),
    # ZONES fix round 1 (critic B-8): a wide-defender zone must stay goal-side of the deepest opponent in the vacated
    # channel, so no_reachable_fix there also means "nothing reachable on our goal's side of him"
    'no_reachable_fix:channel': ('No teammate could have got back in time to a spot that covers the space and stays '
                                 'between the opponents and our goal, so no spot is marked.',
                                 'No teammate could have reached, in time, a position that covers the channel and '
                                 'stays goal-side of the opponent in it, so no area is marked.'),
    'would_open_new_gap': ('Any spot a teammate could have moved to would have left another space open, so no spot is '
                           'marked.',
                           'Every reachable position would have opened another gap, so no area is marked.'),
    'mover_unseen': ('The nearby teammates were not in view a moment earlier, so no area is marked.',
                     'The candidate teammates were not observed two seconds earlier, so no area is marked.'),
    'context_missing': ('Not enough of the scene was in view to mark an area.',
                        'Too little of the scene was observed to validate an area.'),
    'scale_unknown': ('Without a known field size no area is marked.',
                      'Without a declared or measured field size no area is marked.'),
    'guardrail_failed': ('Because some context could not be confirmed, no area is marked.',
                         # not "did not pass": "pass" is an unmeasured event to schema.event_problems (the wave-1
                         # wording could never be shown; found by the wave-5 mutation check)
                         'Because part of the context could not be confirmed, no precise area is marked.'),
    'no_zone_for_detector': ('The space between the lines closes when the whole group moves together, so no single '
                             'spot is marked.',
                             'The gap between the lines is closed by the whole unit moving, so no single area is '
                             'marked.'),
}
_SLOT = re.compile(r'\{(\w+)\}')


def no_alt_facts(finding, no_alternative):
    """The facts of a no-alternative limitation claim: ZONES' reason and the wording key (NO_ALT_WORDS): a
    wide-defender finding with an opponent in the vacated channel words no_reachable_fix as "nothing reachable on our
    goal's side of him" (ZONES fix round 1: the zone stops 2 m upfield of the deepest channel opponent)."""
    reason = no_alternative.get('reason')
    m = finding.get('measurements') or {}
    wording = reason
    if reason == 'no_reachable_fix' and finding['observation'] == 'wide_defender_advance' and \
            _num(m.get('opponentsInChannel')) and m['opponentsInChannel'] >= 1:
        wording = 'no_reachable_fix:channel'
    return {'reason': reason, 'wording': wording}


# ---------------------------------------------------------------------------------------------------------------
# Small helpers

def _num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))


def _at(table, level):
    """The entry of a {levels tuple: value} table for a level: the tuple containing it, else the nearest lower
    level's entry, else the nearest higher one."""
    if not table:
        return None
    for keys, v in table.items():
        if level in keys:
            return v
    below = [(max(k), v) for k, v in table.items() if max(k) < level]
    if below:
        return max(below, key=lambda kv: kv[0])[1]
    return min(((min(k), v) for k, v in table.items()), key=lambda kv: kv[0])[1]


def _bank(detector, slot, level):
    return _at((BANK.get(detector) or {}).get(slot), level)


def _cap(s):
    return s[:1].upper()+s[1:] if s else s


_PLAIN_FIRST = re.compile(r'[A-Z][a-z]+(?=[\s,-])|A(?=[\s,])')


def _lower_word(s):
    """s with its first word lowered when it is a plain capitalised word ("When", "The", "A"), never an id ("A-05"),
    a name or an acronym, so a sentence can follow "In the first of them, "."""
    return s[:1].lower()+s[1:] if s and _PLAIN_FIRST.match(s) else s


def _hash(*parts):
    return int(hashlib.sha1('|'.join(str(p) for p in parts).encode()).hexdigest()[:8], 16)


def _fill(template, slots):
    """The template filled from slots, or None when a slot it names has no value."""
    names = _SLOT.findall(template)
    if any(slots.get(n) is None for n in names):
        return None
    return template.format(**{n: slots[n] for n in names})


def _sentence(s):
    s = re.sub(r'\s+', ' ', s).strip()
    s = s[:1].upper()+s[1:]
    return s if s[-1:] in '.?!' else s+'.'


def _trigrams(text):
    w = re.findall(r"[a-z0-9']+", (text or '').lower())
    return {tuple(w[k:k+3]) for k in range(len(w)-2)}


def jaccard(a, b):
    """Word-trigram Jaccard similarity of two texts (DESIGN 6.4's repetition measure)."""
    ta, tb = _trigrams(a), _trigrams(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb)/len(ta | tb)


def _norm_source(sid):
    return str(sid).upper().replace('_', '-')


# ---------------------------------------------------------------------------------------------------------------
# Register

def _params(profile, detector):
    return (((profile.get('detectors') or {}).get(detector) or {}).get('params') or {})


CB_SHIFT_STORED_M = 4.0     # tactics_geom.CB_SHIFT_M: the shift the stored cbAdjusted flag was measured against


def d3_cb_covered(m, params):
    """Did the near centre-back's shift cover the advance at this moment, by the level's rule (contextual's
    sample_outcomes test (c))? The stored cbAdjusted flag, unless the level's cbShiftM differs from the threshold it
    was stored with and the shift itself was measured (nearCbShiftM); None when neither was measured."""
    shift, need, flag = m.get('nearCbShiftM'), params.get('cbShiftM'), m.get('cbAdjusted')
    if isinstance(flag, bool) and (not _num(need) or abs(float(need)-CB_SHIFT_STORED_M) < 1e-9 or not _num(shift)):
        return flag
    if _num(shift) and _num(need):
        return float(shift) >= float(need)
    return flag if isinstance(flag, bool) else None


def d3_cover_by(m, params):
    """Who covered a wide defender's advance at a strength's moment, in contextual's order of tests (polish round,
    Oct 23): 'channel' (a teammate in the channel behind: test a), 'cb' (the near centre-back moved across: test c),
    'dm' (a holding midfielder dropped, where the level counts it: useDmRotation), 'swap' (an interchange: a teammate
    took the position) or None (the measurements of the moment name no cover). The strength's text says that one."""
    if m.get('channelOccupied') is True:
        return 'channel'
    if d3_cb_covered(m, params) is True:
        return 'cb'
    if params.get('useDmRotation') and m.get('dmDropped') is True:
        return 'dm'
    if m.get('interchangeCover') is True:
        return 'swap'           # fix round 3 (FACTS F-8): the geometric interchange at the moment, not a label swap
    return None


def register_for(profile, finding=None):
    """The effective register of a resolved profile: DESIGN 6.3's defaults for its level, overridden by the
    profile's register block (numbers, maxChallenges, tone, banned, concepts) and evidence.naming, plus the adult flag
    (age group U19 and older: "Next step:" instead of "Try this:"). With a finding whose detector the level judges
    team-level only (params.teamLevelOnly: levels 1-2 pressure and cover / wide defenders, levels 1-3 behind the ball),
    nobody is named or singled out by a role or relative word (naming 'never'): on EVAL's unsupported_advance scene
    the level-2 text said "our left defender went forward" although DESIGN 4.3 frames level 2 team-level."""
    reg = _register(profile)
    if finding is not None and _params(profile, finding['observation']).get('teamLevelOnly'):
        reg = dict(reg, naming='never', teamLevel=True)
    return reg


def _register(profile):
    level = int(profile['level'])
    reg = dict(REGISTERS[level])
    pr = profile.get('register') or {}
    for k in ('numbers', 'maxChallenges', 'tone'):
        if pr.get(k) is not None:
            reg[k] = pr[k]
    if reg['numbers'] not in NUMBER_MODES:
        reg['numbers'] = REGISTERS[level]['numbers']
    reg['banned'] = tuple(w.lower() for w in pr.get('banned') or ())
    reg['concepts'] = tuple(pr.get('concepts') or ())
    naming = (profile.get('evidence') or {}).get('naming')
    if naming in NAMING:
        reg['naming'] = naming
    reg['level'] = level
    reg['adult'] = profile.get('ageGroup') in ADULT_AGES
    reg['teamLevel'] = False
    # small formats never use 11v11 role names (DESIGN 2.5; critic S-9: 24 small-format reports said full-back,
    # centre-back or holding midfielder): the small-format lexicon (SMALL_WORDS) and the format_words check apply
    fmt = (profile.get('format') or {}).get('name') if isinstance(profile.get('format'), dict) else None
    reg['format'] = fmt if isinstance(fmt, str) else '11v11'
    reg['small'] = reg['format'] != '11v11'
    return reg


def _numbers_at_least(reg, mode):
    return NUMBER_MODES.index(reg['numbers']) >= NUMBER_MODES.index(mode)


# ---------------------------------------------------------------------------------------------------------------
# Claims (DESIGN 6.2): the only material the text may use

def _role_of(finding, pid):
    r = (finding.get('roles') or {}).get(pid) if pid else None
    return r if isinstance(r, dict) else None


def _confident(role):
    if not role or not role.get('role'):
        return False
    if role.get('source') in CONFIDENT_SOURCES:
        return True
    return role.get('source') == 'relative' and role.get('confidence') in ('high', 'medium')


def _person(finding, pid, reg, identity_ok):
    """The facts a claim keeps about one of our players: the id only when the level may name players (and the
    identity guardrail passed), the role only when it is confident enough for role words (DESIGN 3.4)."""
    if not pid:
        return {'player': None, 'role': None, 'roleSource': None}
    role = _role_of(finding, pid)
    named = identity_ok and reg['naming'] in ('evidence_only', 'yes')
    worded = identity_ok and reg['naming'] != 'never' and _confident(role)
    return {'player': pid if named else None, 'role': role['role'] if worded else None,
            'roleSource': role.get('source') if worded else None}


def _supporting(finding, roles_wanted):
    for pid in finding['players'].get('supporting') or ():
        r = _role_of(finding, pid)
        if r and r.get('role') in roles_wanted:
            return pid
    return None


def _source_for(code, level, profile):
    """The source a principle claim may cite: a candidate listed for exactly this level (never a neighbouring
    level's source) that the resolved profile also lists."""
    have = {_norm_source(s): s for s in profile.get('sources') or ()}
    cands = next((v for k, v in PRINCIPLES[code]['sources'].items() if level in k), ())
    for cand in cands:
        if _norm_source(cand) in have:
            return have[_norm_source(cand)]
    return None


def phase_source_of(finding, opportunities=None):
    """Where the finding's phase came from: 'possession-4' (estimated), 'coach' (coach-marked spans), 'mixed', or
    'none' when no phase is named. Read from its opportunities (obs/roles files) when given, else from
    context.phase.estimated (a coach-marked phase is not an estimate)."""
    state = finding['context']['phase']['state']
    if state not in ('own', 'opp'):
        return 'none'
    srcs = set()
    for oid in finding.get('opportunities') or ():
        o = (opportunities or {}).get(oid)
        if o:
            srcs.add(o['phase']['source'])
    srcs.discard('none')
    if not srcs:
        # without the obs / roles files: assess's measurements.phaseSource (DETECT wave 3), else the estimated flag
        src = (finding.get('measurements') or {}).get('phaseSource')
        if src in QUALIFIER:
            return src
        return 'possession-4' if finding['context']['phase']['estimated'] else 'coach'
    if srcs == {'coach'}:
        return 'coach'
    return 'mixed' if 'coach' in srcs else 'possession-4'


def shape_relations(finding, profile=None):
    """attacking_shape at the finding's key moment: {'width': bool | None (not judged), 'depth': bool | None,
    'channels': bool | None}, judged with the level's params as contextual.sample_outcomes does (None where the
    measurement is missing). Depth (fix round 1, critic B-7): the field share holds, or, with the level's depthRule
    'available' / 'available_or_field', the length used up to the opponents' last line (depthAvailShare against
    params.depthAvailShare, .75); 'available' with that length unmeasured is not judged (None), 'available_or_field'
    then falls back to the field share. On BC180 Barcelona against Espanyol's low block the field share was .25 and
    the available share about 1.0: the wave-5 text still called that "short from back to front"."""
    m = finding.get('measurements') or {}
    pr = _params(profile or {}, 'attacking_shape')
    p = dict(SHAPE_DEFAULTS, depthAvailShare=.75)
    p.update({k: v for k, v in pr.items() if _num(v)})
    judge_width = pr.get('judgeWidth') is not False
    rule = pr.get('depthRule') or 'field'
    ws, ds, ch = m.get('widthShare'), m.get('depthShare'), m.get('channelsOccupied')
    da = m.get('depthAvailShare')
    if not _num(ds):
        depth = None
    elif ds >= p['depthShare']-1e-9 or rule == 'field':
        depth = ds >= p['depthShare']-1e-9
    elif _num(da):
        depth = da >= p['depthAvailShare']-1e-9
    else:
        depth = False if rule == 'available_or_field' else None
    return {'width': (ws >= p['widthShare']-1e-9 if _num(ws) else None) if judge_width else None,
            'depth': depth,
            'channels': (int(ch) >= CHANNELS_TOTAL) if _num(ch) else None}


def failure_mode(finding, level, profile=None):
    """The failure mode of a criticised or reviewed finding: D1 no_cover / no_pressure and D5 lateral_gap / dm_gap
    (measurements.failureMode, else inferred), D6 the relation that failed at the key moment (width, depth, shape =
    both, channels); None for other detectors and verdicts."""
    det = finding['observation']
    m = finding.get('measurements') or {}
    if det in ADVANCE_MODES:
        # fix round 3 (ASSESS): the advance's mode as measured, never inferred (None without it)
        mode = m.get('failureMode') if finding['verdict'] in FAILURES else None
        return mode if mode in ADVANCE_MODES[det] else None
    if det not in FAILURE_MODES or finding['verdict'] not in FAILURES:
        return None
    if det == 'pressure_cover' and is_moments(finding) and isinstance(m.get('pressurePresent'), bool):
        # a 'moments worth a look' review describes its FIRST moment (a clear failing sample, so its own flags say
        # which way it failed), not the modal mode of moments that may differ (fix round 2)
        if m['pressurePresent'] is False:
            return 'no_pressure'
        if m.get('coverPresent') is False:
            return 'no_cover'
    mode = m.get('failureMode')
    if mode in FAILURE_MODES[det]:
        return mode
    if det == 'pressure_cover':
        if level == 1 or m.get('pressurePresent') is False:
            return 'no_pressure'
        return 'no_cover'
    if det == 'attacking_shape':
        rel = shape_relations(finding, profile)
        if rel['width'] is False and rel['depth'] is False:
            return 'shape'
        if rel['width'] is False:
            return 'width'
        if rel['depth'] is False:
            return 'depth'
        if rel['channels'] is False:
            return 'channels'
        return None
    return 'dm_gap' if _num(m.get('dmGapM')) and not _num(m.get('lateralGapM')) else 'lateral_gap'


def presser_of(finding):
    """The pressure-cover finding's presser: measurements.fdId (the first defender at the first key moment, critic
    B-3), else players.primary[0] (assess sets it to fdId since fix round 1)."""
    m = finding.get('measurements') or {}
    if isinstance(m.get('fdId'), str) and L.ID_RE.fullmatch(m['fdId']):
        return m['fdId']
    return (finding['players'].get('primary') or [None])[0]


def cover_of(finding):
    """The pressure-cover finding's covering teammate at the moment: measurements.coverId, else the first supporting
    player; never the presser (None then)."""
    m = finding.get('measurements') or {}
    presser = presser_of(finding)
    cands = [m.get('coverId')]+list(finding['players'].get('supporting') or ())
    return next((c for c in cands if isinstance(c, str) and c and c != presser), None)


def m_channel_by(finding):
    """The player in the vacated channel at the moment (wide_defender_advance measurements.channelBy), or None."""
    v = (finding.get('measurements') or {}).get('channelBy')
    return v if isinstance(v, str) and v else None


def phase_known(finding):
    """Did the finding's phase name a team (own / opp)? With 'unknown' no sentence may say who had the ball (critic
    B-4: 28 of 28 phase-unknown findings on the amateur clip were headed "Attacking shape" / "Team shape with the
    ball")."""
    return finding['context']['phase']['state'] in ('own', 'opp')


def is_moments(finding):
    """Is this a 'moments worth a look' review (assess fix round 2: verdict 'review' with the limitation MOMENTS and
    its counts in the measurements)? Without the counts it is worded as any other review (never invented)."""
    m = finding.get('measurements') or {}
    return finding.get('verdict') == 'review' and MOMENTS in (finding.get('limitations') or ()) and \
        all(_num(m.get(k)) for k in MOMENTS_KEYS) and m['decidedMoments'] >= 1


def variant_of(finding):
    """A wording variant every verdict shares: 'advance' for level 1's reframed wide-defender advances (assess family
    'advance', observation behind_ball: "someone ready to protect our goal" while a defender pushes up)."""
    parts = str(finding.get('key') or '').split('|')
    if finding['observation'] == 'behind_ball' and len(parts) > 2 and parts[2] == 'advance':
        return 'advance'
    return None


def _valid_id(x):
    return isinstance(x, str) and bool(L.ID_RE.fullmatch(x))


def moment_actor(finding):
    """The player a wide-defender or spacing finding's text names as its actor (FACTS F-2, fix round 3): the first
    cited moment's own (measurements.actorId: the advancing defender of that opportunity, the gap player the moment's
    opportunity names), else players.primary[0]. Before, D3 and D5 named the family's modal player: "the right back
    (B-03) advanced on the right" while the moment's wide defender was B-04 on the other flank."""
    m = finding.get('measurements') or {}
    if finding['observation'] in ('wide_defender_advance', 'defensive_spacing') and _valid_id(m.get('actorId')):
        return m['actorId']
    return (finding['players'].get('primary') or [None])[0]


def gap_partner(finding):
    """The other player of a spacing finding's gap at its first moment (measurements.gapLeftId / gapRightId, FACTS
    F-2: the text named A-02 / A-03 while the moment's pair was A-04 / A-02), else players.supporting[0]."""
    m = finding.get('measurements') or {}
    pair = [m.get('gapLeftId'), m.get('gapRightId')]
    actor = moment_actor(finding)
    if all(_valid_id(p) for p in pair) and pair[0] != pair[1]:
        return pair[1] if actor == pair[0] else pair[0]
    return (finding['players'].get('supporting') or [None])[0]


def behind_margin(obs):
    """The behind-the-ball margin of a run in metres: BEHIND_MARGIN_M x L / 105 of its coaching scale (tactics_geom's
    rule), None without a scale."""
    scale = ((obs or {}).get('timeline') or {}).get('scale') or {}
    length = scale.get('lengthM')
    return round(BEHIND_MARGIN_M*float(length)/BEHIND_REF_LENGTH, 2) if _num(length) and length > 0 else None


def behind_words(margin, metres, reg):
    """How a text states the behind-the-ball margin (FACTS F-1): "at least 10 m" from the level's rounded-metres
    register with usable metres (the margin is the threshold the count used, so it carries no "about"), else
    "well"."""
    if _num(margin) and metres != 'unusable' and _numbers_at_least(reg, 'rounded_metres'):
        return f'at least {int(round(float(margin)))} m'
    return 'well'


def often_words(n):
    """A number-free count for level 1 (COACH F-11: "We did this a few times" for 24 of 24 moments)."""
    if not _num(n) or n < 2:
        return 'once'
    return 'a few times' if n < 5 else ('several times' if n < 10 else 'many times')


def base_words_for(reg, level):
    """The base a mixed-evidence description was tested against, in words (LEAD decision (3), COACH F-5): the
    professional base only for an adult team at levels 5-7 (the base is the professional rate there, unscaled);
    else the level's own base (scaled below level 5). (words, source note)."""
    if reg.get('adult') and level >= 5:
        return 'the professional base rate PitchIQ measured', 'measured in professional play'
    return 'the base rate PitchIQ uses for this level', "PitchIQ's base for this level"


def build_claims(finding, profile, *, phase_source=None, alternative=None, no_alternative=None, register=None,
                 short=False, behind_margin_m=None):
    """The typed claims of one finding at the profile's level (schema 'claim' records): what the text may say.
    Built only from the finding's own fields, the validated alternative (ZONES) and the profile; level-filtered
    (DESIGN 6.3): a concept the level does not use gives no claim. Every claim's facts carry the numbers, times and
    ids its sentence may show. behind_margin_m: the run's behind-the-ball margin (behind_margin(obs)); with it the
    wide-defender, behind-the-ball and attacking-shape claims carry it (situation behindMarginM) and the counts say
    "at least N m behind the ball"."""
    reg = register or register_for(profile)
    lvl = reg['level']
    det = finding['observation']
    verdict = finding['verdict']
    fail = verdict in FAILURES
    m = finding.get('measurements') or {}
    a = finding['aggregate']
    ctx = finding['context']
    ident = finding['guardrails'].get('identity') != 'fail'
    psrc = phase_source or phase_source_of(finding)
    out = []

    def add(kind, facts, tag=None, source=None):
        c = {'type': kind, 'facts': {k: v for k, v in facts.items()}}
        if tag:
            c['tag'] = tag
            c['source'] = source
        out.append(c)

    ts = finding.get('timestamps') or []
    ball = ctx.get('ball') or {}
    mode = failure_mode(finding, lvl, profile)
    # fix round 3 (FACTS F-5): the place is the first cited moment's own third (measurements.momentZone), the one
    # the explanation describes; the family's modal zone put 25 distinct Metrica texts in the wrong third ("in the
    # final third, the right back pushed up" with the ball at X 36.5 m)
    zone = m.get('momentZone') if m.get('momentZone') in PLACE else ball.get('zone')
    if det == 'behind_ball':
        # behind the ball is JUDGED with the ball in the final third only (contextual.D4_JUDGED_ZONE): the place of a
        # judged verdict is that zone, never the 3 m hysteresis zone of the opportunities (FACTS L8: "in the middle
        # third" with the ball at 70.4-72.3 m); a description's moment may be anywhere, so it names no place. Fix
        # round 3: level 1's reframed advance is judged wherever the ball is (covered or exposed), so it is placed
        # at its moment (momentZone), never "near the other team's goal" by default
        judged = _params(profile, 'behind_ball').get('behindBallZone') or 'final_third'
        if verdict == 'describe':
            zone = None
        elif m.get('momentZone') in PLACE:
            zone = m['momentZone']
        elif variant_of(finding) == 'advance':
            zone = None
        else:
            zone = judged if judged in PLACE else None
    sit = {'detector': det, 'phase': ctx['phase']['state'], 'source': psrc,
           'estimated': bool(ctx['phase']['estimated']), 'coverage': ctx['phase'].get('coverage'),
           'zone': zone, 'channel': ball.get('channel'), 'format': ctx.get('format'),
           't': ts[0]['t'] if ts else None, 'moments': len(ts), 'mode': mode, 'variant': variant_of(finding)}
    if det in ('wide_defender_advance', 'behind_ball', 'attacking_shape') and _num(behind_margin_m):
        # the margin every count of ours (and of theirs on our side) at the ball's depth was measured with (FACTS F-1);
        # behindMetres: the finding's metres qualifier ('unusable': the words say "well", never a number)
        sit.update(behindMarginM=float(behind_margin_m), behindMetres=m.get('metres', 'estimated'))
    add('situation', sit)
    primary = (finding['players'].get('primary') or [None])[0]
    if det == 'pressure_cover':
        # the presser AT the moment the text links (critic B-3: the family's modal first defender was named in 10 of
        # 10 checked synthetic texts while the distance and the cover were the moment's), never the cover
        primary = presser_of(finding)
    elif det in ('wide_defender_advance', 'defensive_spacing'):
        primary = moment_actor(finding)       # the moment's own advancing defender / gap player (FACTS F-2)
    actor = _person(finding, primary, reg, ident)
    metres = m.get('metres', 'estimated')
    params = _params(profile, det)
    rel = []                                   # (kind, present, facts)

    if det == 'pressure_cover':
        mate = cover_of(finding)
        add('actor_state', dict(actor, state='first_defender', distanceM=m.get('d1M'), metres=metres))
        pressed = mode != 'no_pressure'
        rel.append(('pressure', pressed, {'distanceM': m.get('d1M')}))
        if mode == 'no_cover' or (verdict == 'strength' and m.get('coverPresent') is not False and lvl >= 2):
            rel.append(('cover', mode != 'no_cover', dict(_person(finding, mate, reg, ident),
                                                          distanceM=m.get('coverDistM'),
                                                          angleDeg=m.get('coverAngleDeg'), metres=metres)))
        elif verdict == 'strength' and lvl == 1 and m.get('coverPresent') is True:
            rel.append(('cover', True, dict(_person(finding, mate, reg, ident), scope='help')))
        if lvl >= 5 and _num(m.get('goalSideShare')):
            add('team_state', {'state': 'goal_side', 'share': m['goalSideShare']})
        if lvl >= 5 and _num(m.get('coverPresentShare')):
            add('team_state', {'state': 'cover_share', 'share': m['coverPresentShare']})
    elif det == 'weak_side_balance':
        need = params.get('minWeakCount') if _num(params.get('minWeakCount')) else 1
        weak = m.get('weakCount')
        present = (weak >= need) if _num(weak) and verdict != 'describe' else (None if verdict == 'describe'
                                                                                else not fail)
        rel.append(('weak_side', present, {'count': weak, 'ballChannel': ball.get('channel'), 'metres': metres}))
        if lvl >= 4 and _num(m.get('centralCount')):
            add('team_state', {'state': 'central', 'count': m['centralCount']})
        # the far touchline's distance from the widest goal-side far-half defender; with none it is the whole far
        # half (tactics_geom.weak_side's W/2), said as such (FACTS M7: "about 34 m" read as a measured gap)
        if lvl >= 5 and (fail or verdict == 'describe') and _num(m.get('openWidthM')):
            add('relation', {'relation': 'open_width', 'present': None, 'distanceM': m['openWidthM'],
                             'metres': metres, 'wholeHalf': not (_num(weak) and weak >= 1)})
        # levels 6-7: the far-half defenders level with the ball (unclear to the measurement) and running back
        if lvl >= 6 and _num(m.get('weakLevelCount')):
            add('team_state', {'state': 'level', 'count': int(m['weakLevelCount'])})
        if lvl >= 6 and _num(m.get('weakRecovering')):
            add('team_state', {'state': 'recovering', 'count': int(m['weakRecovering'])})
    elif det == 'wide_defender_advance':
        side = m.get('side') if m.get('side') in ('left', 'right') else None
        if side is None and actor['role']:
            side = 'left' if actor['role'][:1] == 'L' else 'right' if actor['role'][:1] == 'R' else None
        # polish round: a strength from level 3 says who covered at its moment (d3_cover_by), so its cover claim is
        # the one the level's rule used; the centre-back's shift (and a midfielder's drop) is claimed below level 5
        # only when it was the cover
        cover_by = d3_cover_by(m, params) if verdict == 'strength' and lvl >= 3 else None
        state = dict(actor, state='advanced', side=side, distanceM=m.get('advanceM'), metres=metres)
        if cover_by:
            state['coverBy'] = cover_by
        add('actor_state', state)
        rel.append(('advance_support', not fail, {}))
        if lvl >= 3 and isinstance(m.get('channelOccupied'), bool):
            filler = m.get('channelBy') if isinstance(m.get('channelBy'), str) else None
            rel.append(('channel_cover', m['channelOccupied'], dict(_person(finding, filler, reg, ident), side=side)))
        cb_present = (cover_by == 'cb' or (cover_by == 'channel' and d3_cb_covered(m, params) is True)) \
            if verdict == 'strength' and lvl >= 3 else m.get('cbAdjusted')
        if (lvl >= 5 and isinstance(m.get('cbAdjusted'), bool)) or (lvl >= 3 and cover_by == 'cb'):
            cb = _supporting(finding, CB_ROLES)
            rel.append(('cb_shift', bool(cb_present), dict(_person(finding, cb, reg, ident),
                                                           distanceM=m.get('nearCbShiftM'), metres=metres)))
        if (lvl >= 5 and isinstance(m.get('dmDropped'), bool)) or (lvl >= 3 and cover_by == 'dm'):
            dm = _supporting(finding, DM_ROLES)
            rel.append(('dm_drop', bool(m['dmDropped']), _person(finding, dm, reg, ident)))
        # fix round 3 (FACTS F-8): "a teammate took their place" only with the geometric interchange AT the moment
        # (measurements.interchangeCover: a teammate re-occupied the slot), never the +-10 s role-label swap
        # (interchange), which was the claim of 17 of 54 such sentences with nobody in the vacated spot
        if lvl >= 3 and m.get('interchangeCover') is True:
            add('intent_signal', {'signal': 'interchange', 'side': side})
        if _num(m.get('behindN')):
            rel.append(('behind_ball', m['behindN'] >= 1, {'count': int(m['behindN']), 'required': 1}))
        # the opponents in the vacated channel: from level 4 as a count, and at level 2 for an exposed advance (the
        # failure is that one of theirs stood in the space left; fix round 3, critic B-1); level 3 does not state it
        if _num(m.get('opponentsInChannel')) and m['opponentsInChannel'] >= 1 and \
                (lvl >= 4 or (lvl <= 2 and mode == 'exposed')):
            add('opponent_state', {'state': 'in_channel', 'count': int(m['opponentsInChannel']), 'side': side})
        if lvl >= 6 and _num(m.get('restMargin')):
            add('team_state', {'state': 'rest_margin', 'margin': m['restMargin'], 'behind': m.get('behindN'),
                               'threat': m.get('threatN')})
        if lvl >= 7 and _num(m.get('dmCbSpacingM')):
            dm = _supporting(finding, DM_ROLES)
            add('relation', dict(_person(finding, dm, reg, ident), relation='dm_cb_spacing', present=None,
                                 distanceM=m['dmCbSpacingM'], metres=metres))
    elif det == 'behind_ball':
        cnt = m.get('behindN')
        if params.get('behindRule') == 'margin' and variant_of(finding) is None:
            # levels 6-7 on 9v9 / 11v11 judge the PitchIQ rest-defence margin, not a count: no required count is
            # claimed (a required 1 against 3 behind the ball would contradict the measurement)
            need = params.get('minMargin') if _num(params.get('minMargin')) else None
            marg = m.get('restMargin')
            present = None if verdict == 'describe' else ((marg >= need) if _num(marg) and need is not None
                                                          else not fail)
            rel.append(('behind_ball', present, {'count': int(cnt) if _num(cnt) else None, 'margin': marg,
                                                 'requiredMargin': need}))
        else:
            req = m.get('minBehind') if _num(m.get('minBehind')) else (params.get('minBehind') if
                                                                          _num(params.get('minBehind')) else 1)
            present = not fail if verdict != 'describe' else (cnt >= req if _num(cnt) else None)
            if verdict != 'describe' and _num(cnt):
                present = cnt >= req
            rel.append(('behind_ball', present, {'count': int(cnt) if _num(cnt) else None, 'required': int(req)}))
        if variant_of(finding) == 'advance':
            # level 1's reframed advance (fix round 3, critic B-1): what assess judged it by, the space the defender
            # left (covered or not) and the opponents in it; its words never claim a count behind the ball they did
            # not measure (the gate reads these relations: DETECTOR_KINDS_ADVANCE)
            rel.append(('advance_support', not fail if verdict != 'describe' else None, {}))
            if isinstance(m.get('channelOccupied'), bool):
                rel.append(('channel_cover', m['channelOccupied'], {'side': m.get('side')}))
            if _num(m.get('opponentsInChannel')) and m['opponentsInChannel'] >= 1:
                add('opponent_state', {'state': 'in_channel', 'count': int(m['opponentsInChannel']),
                                       'side': m.get('side')})
        if lvl >= 5 and _num(m.get('threatN')):
            add('opponent_state', {'state': 'threat', 'count': int(m['threatN'])})
        if lvl >= 6 and _num(m.get('restMargin')):
            add('team_state', {'state': 'rest_margin', 'margin': m['restMargin'], 'behind': cnt,
                               'threat': m.get('threatN')})
    elif det == 'defensive_spacing':
        sup = gap_partner(finding)          # the moment's own gap pair (FACTS F-2)
        two = dict(_person(finding, sup, reg, ident))
        if mode == 'lateral_gap':
            rel.append(('lateral_gap', True, dict(two, distanceM=m.get('lateralGapM'), metres=metres)))
            if m.get('opponentInGap') is True:
                add('opponent_state', {'state': 'in_gap', 'count': 1})
            if lvl >= 5 and _num(m.get('dmGapM')):         # the distance to midfield beside the gap (context)
                rel.append(('unit_gap', None, {'distanceM': m['dmGapM'], 'metres': metres}))
        elif mode == 'dm_gap':
            rel.append(('unit_gap', True, {'distanceM': m.get('dmGapM'), 'metres': metres}))
        elif verdict == 'strength':
            rel.append(('line_evenness', True, {'distanceM': m.get('lineStdM'), 'metres': metres}))
            rel.append(('unit_gap', False, {'distanceM': m.get('dmGapM'), 'metres': metres}))
        else:
            rel.append(('unit_gap', None, {'distanceM': m.get('dmGapM'), 'metres': metres}))
            if _num(m.get('lateralGapM')):     # a description may give the widest gap when no line distance exists
                add('relation', {'relation': 'lateral_gap_context', 'present': None, 'distanceM': m['lateralGapM'],
                                 'metres': metres})
            # a description's further context by level (S-6: per-level reasoning, not a paraphrase): the line's
            # evenness from level 5, the opponents standing between our back line and midfield at level 7
            if lvl >= 5 and _num(m.get('lineStdM')):
                rel.append(('line_evenness', None, {'distanceM': m['lineStdM'], 'metres': metres}))
            if lvl >= 7 and _num(m.get('dmGapOpponents')):
                # the ball's lateral half only (contextual's dmGapOpponents; FACTS F-11: "2 of theirs standing in that
                # space" with 3 across the full width)
                add('opponent_state', {'state': 'between_lines', 'count': int(m['dmGapOpponents']),
                                       'scope': 'ball_side'})
        if lvl >= 6 and m.get('blockHeight') in ('low', 'mid', 'high'):
            add('team_state', {'state': 'block', 'height': m['blockHeight']})
        add('actor_state', dict(actor, state='defender'))
    elif det == 'attacking_shape':
        # each relation as measured at the key moment under the level's params (shape_relations), never the
        # verdict's polarity copied onto all three
        sr = shape_relations(finding, profile)

        def shown(k):
            return None if verdict == 'describe' else sr[k]
        rel.append(('width', shown('width'), {'share': m.get('widthShare'), 'distanceM': m.get('widthM'),
                                              'metres': metres}))
        depth = {'share': m.get('depthShare'), 'distanceM': m.get('depthM'), 'metres': metres}
        # the length actually available, from our deepest player to their last line (critic B-7): from level 4 in
        # metres, from level 5 as the share of it our shape used
        if lvl >= 4 and _num(m.get('availableDepthM')):
            depth['availM'] = m['availableDepthM']
        if lvl >= 5 and _num(m.get('depthAvailShare')):
            depth['availShare'] = m['depthAvailShare']
        rel.append(('depth', shown('depth'), depth))
        if _num(m.get('channelsOccupied')):
            rel.append(('channels', shown('channels'), {'count': int(m['channelsOccupied']), 'of': CHANNELS_TOTAL}))
        # levels 6-7 read the shape against the opponents' block: how far our highest player stood from their last
        # line; level 7 also the support behind the ball (context, never a verdict)
        if lvl >= 6 and _num(m.get('heightGapM')):
            add('relation', {'relation': 'height_gap', 'present': None, 'distanceM': m['heightGapM'],
                             'metres': metres})
        if lvl >= 7 and _num(m.get('deepN')):
            add('team_state', {'state': 'deep_support', 'count': int(m['deepN'])})

    team_kinds = ('behind_ball', 'width', 'depth', 'channels', 'weak_side')
    for kind, present, facts in rel:
        add('team_state' if kind in team_kinds else 'relation', dict(facts, relation=kind, present=present))
    # praise first at levels 1-2 (DESIGN 6.3): the demonstrated part of a criticised behaviour, from its count
    if verdict in FAILURES and lvl <= 2 and a['demonstrated'] > 0:
        pk = {'pressure_cover': 'pressure' if mode == 'no_pressure' else 'cover'}.get(det) or \
            {'wide_defender_advance': 'behind_ball', 'behind_ball': 'behind_ball', 'attacking_shape': 'width',
             'weak_side_balance': 'weak_side', 'defensive_spacing': 'line_evenness'}[det]
        add('team_state' if pk in team_kinds else 'relation',
            {'relation': pk, 'present': True, 'scope': 'demonstrated', 'count': a['demonstrated']})
    rec = {'opportunities': a['opportunities'], 'failing': a['notDemonstrated'], 'demonstrated': a['demonstrated'],
           'unclear': a['unclear'], 'failingSpells': a['distinctFailingSpells'],
           'demonstratedSpells': a['distinctDemonstratedSpells'], 'phaseSeconds': a['phaseSeconds'],
           'ratePer10': a.get('ratePer10'), 'perMinutes': PER_MINUTES, 'short': bool(short),
           # the judged play behind the cited opportunities (assess, critic B-2), stated with strengths from level 4
           'judgedSeconds': m.get('judgedSeconds') if _num(m.get('judgedSeconds')) else None}
    if a['opportunities']:
        rec['failShare'] = round(a['notDemonstrated']/a['opportunities'], 4)
        rec['demonstratedShare'] = round(a['demonstrated']/a['opportunities'], 4)
    # fix round 3 (FACTS F-6 = COACH F-3): a strength is stated over the moments PitchIQ could judge, as assess's rule
    # counts it (demonstrated / (demonstrated + not demonstrated)); demonstratedShare (of all) stays for drills.py
    decided = a['demonstrated']+a['notDemonstrated']
    rec['decided'] = decided
    if decided:
        rec['decidedShare'] = round(a['demonstrated']/decided, 4)
    add('recurrence', rec)
    effect = next((e for e in str(ctx.get('styleEffect') or '').split('+') if e in STYLE_EFFECT), None)
    if lvl >= 6 and effect and ctx.get('style') in STYLE_WORD:
        add('intent_signal', {'signal': 'style', 'style': ctx['style'], 'effect': effect})
    # what could not be confirmed
    seen = set()
    lim_codes = list(finding.get('limitations') or ())
    moments = is_moments(finding)
    if verdict == 'review' or verdict == 'describe':
        for g in S.GUARDRAILS:
            if g == 'evidence' and ('one_instance' in lim_codes or moments):
                continue                 # "seen once" has its own sentence; "not enough evidence" would repeat it;
                #                          a 'moments worth a look' review says why it is no verdict in its own words
            if finding['guardrails'].get(g) == 'fail' and g in LIMITATION_WORDS:
                # a phase that named no team at all is "who had the ball", not "the phase for every moment"
                code = 'phase_unknown' if g == 'phase' and not phase_known(finding) else g
                seen.add(g)
                seen.add(code)
                add('limitation', {'code': code, 'guardrail': True})
    for code in lim_codes:
        alias = {'identity_uncertain': 'identity', 'role_unconfirmed': 'role', 'phase_unknown': 'phase'}.get(code)
        if code == MOMENTS:
            if moments and code not in seen:
                # the counts a 'moments worth a look' review states (assess fix round 2): the clear failing
                # opportunities and their distinct spells, of the opportunities PitchIQ could judge (demonstrated +
                # not demonstrated), the judged ones that met the check, the unclear ones, and every cited moment
                seen.add(code)
                clear, dec = int(m['clearMoments']), int(m['decidedMoments'])
                dem = max(0, dec-int(m['failingMoments']))
                # more: the judged ones that fell short without being clear (fix round 3, FACTS F-4 = COACH F-10:
                # 24 of 73 count sentences left them out, "3 of the 17 ... met in 9 of the 17")
                add('limitation', {'code': code, 'guardrail': False, 'clear': clear, 'decided': dec,
                                   'spells': int(m['clearSpells']), 'failing': int(m['failingMoments']),
                                   'demonstrated': dem, 'unclear': int(a.get('unclear') or 0),
                                   'more': max(0, int(m['failingMoments'])-clear),
                                   'clearShare': round(clear/dec, 4), 'demonstratedShare': round(dem/dec, 4),
                                   'moments': [{'t': x['t']} for x in ts], 'shown': len(ts)})
            continue
        if code in OWN_SENTENCE_LIMITS and code not in seen:
            seen.add(code)
            facts = {'code': code, 'guardrail': False}
            if code == 'level_cap' and _num((profile.get('evidence') or {}).get('maxWorkItems')):
                facts['maxWorkItems'] = int(profile['evidence']['maxWorkItems'])
            if code == 'mixed_evidence':
                # a strength withheld (assess, critic B-2): the decided opportunities both ways, and the base share
                # it was tested against: measurements.baseShare (fix round 3: assess's praise base, the level's
                # baseFailShare scaled below level 5, and baseTail, the binomial tail that withheld it), else the
                # level's params.baseFailShare (Metrica game 1 at this level's rules)
                dec = a['demonstrated']+a['notDemonstrated']
                facts.update(demonstrated=a['demonstrated'], failing=a['notDemonstrated'], decided=dec)
                if dec:
                    facts['failShare'] = round(a['notDemonstrated']/dec, 4)
                if _num(m.get('baseShare')):
                    facts['baseShare'] = float(m['baseShare'])
                elif _num(params.get('baseFailShare')):
                    facts['baseShare'] = float(params['baseFailShare'])
                if _num(m.get('baseTail')):
                    facts['baseTail'] = float(m['baseTail'])
            add('limitation', facts)
        elif code in LIMITATION_WORDS and code not in seen and alias not in seen:
            seen.add(code)
            add('limitation', {'code': code, 'guardrail': False})
    # principle, hypothetical, counterfactual, action
    sa = finding.get('suggestedAction') or {}
    pcode = sa.get('principle') if sa.get('principle') in PRINCIPLES else DEFAULT_PRINCIPLE[det]
    if det == 'pressure_cover':
        pcode = 'pressure_on_ball' if mode == 'no_pressure' else ('help_teammate' if lvl == 1 else
                                                                 'cover_first_defender')
    if det == 'wide_defender_advance' and lvl <= 2:
        # level 2's exposed advance (fix round 3): the cover principle, the space left needs a teammate behind
        pcode = 'cover_behind_advancing_defender' if mode == 'exposed' and lvl == 2 else 'someone_back'
    if det == 'defensive_spacing' and mode == 'dm_gap':
        pcode = 'compact_lines'          # the distance between the lines, not the gaps inside the back line
    prel = 'behind_ball' if pcode == 'someone_back' else PRINCIPLES[pcode]['relation']
    if det == 'pressure_cover' and pcode == 'pressure_on_ball':
        prel = 'pressure'
    if det == 'attacking_shape' and mode in ('depth', 'channels'):
        prel = mode
    src = _source_for(pcode, lvl, profile)
    add('principle', {'principle': pcode, 'detector': det, 'relation': prel},
        tag='FEDERATION' if src else S.ORIGIN_P, source=src)
    if fail and lvl >= 3 and det != 'attacking_shape':
        add('hypothetical', {'consequence': {3: 'ball_lost', 4: 'ball_lost', 5: 'transition', 6: 'rest_defence',
                                             7: 'route'}[lvl] if det in ('wide_defender_advance', 'behind_ball')
                             else {'pressure_cover': 'presser_beaten' if mode == 'no_cover' else 'carrier_free',
                                   'weak_side_balance': 'switch',
                                   'defensive_spacing': 'line_gap' if mode == 'dm_gap' else 'gap_run'}[det],
                             'relation': prel})
    if moments:
        pass        # a 'moments worth a look' review says what to look for, never a better position (fix round 2)
    elif alternative and fail and det != 'attacking_shape':      # no zones for attacking shape (DESIGN 5.2)
        mover = alternative.get('mover')
        add('counterfactual', dict(_person(finding, mover, reg, ident), relation=prel,
                                   display=alternative.get('display'), assumption='single_mover',
                                   alternative=alternative.get('id'), evidence=alternative.get('evidence'),
                                   moverIsActor=bool(mover and mover == primary)))
    elif no_alternative and fail and no_alternative.get('reason') in NO_ALT_WORDS and lvl >= NO_ALT_MIN_LEVEL and \
            not (verdict == 'review' and no_alternative['reason'] == 'guardrail_failed'):
        # a review whose guardrail failed already says what could not be confirmed; levels 1-2 never explain the
        # zone engine (COACH L5: "Every spot a teammate could reach would have opened another space" for a U8 coach)
        add('limitation', dict(no_alt_facts(finding, no_alternative), code='no_alternative', guardrail=False))
    if fail:
        add('action', {'principle': pcode, 'relation': prel,
                       'mode': 'question' if verdict == 'try_this' or (lvl <= 3 and verdict == 'work_on')
                       else ('review' if verdict == 'review' else 'statement')})
    for d in finding.get('drills') or ():
        add('drill_link', {'drill': d})
    return out


# ---------------------------------------------------------------------------------------------------------------
# Slots: the words each claim fact becomes at the level

def _ref_text(facts, reg, field, generic):
    """How a player is referred to in a field: an id (with a role word when one is known), role words, relative
    words or the team-level generic phrase."""
    pid, role = facts.get('player'), facts.get('role')
    lvl = reg['level']
    word = None
    if role:
        mm = REL_ROLE_RE.fullmatch(role)
        if mm:
            k, n = int(mm.group(2)), int(mm.group(3))
            pos = 'leftmost ' if k == 1 else 'rightmost ' if k == n else None
            word = pos+RELATIVE_BLOCK[mm.group(1)] if pos else None
        elif lvl <= 2:
            word = RELATIVE_WORD.get(role)
        elif reg.get('small'):
            word = SMALL_ROLE_WORD.get(role)     # relative words below 11v11 (DESIGN 2.5, critic S-9)
        else:
            word = ROLE_WORD.get(role)
            if role in DM_ROLES and lvl <= 4:
                word = 'deepest midfielder'
            elif role in DM_ROLES and lvl == 7:
                word = 'six'             # the analyst's name for the holding midfielder (DESIGN 6.3, level 7)
    headline_like = field in ('headline', 'short')
    if headline_like:
        return generic
    if reg['naming'] in ('never',):
        return generic
    if pid and reg['naming'] in ('evidence_only', 'yes') and field == 'explanation':
        return f'the {word} ({pid})' if word else pid
    if pid and reg['naming'] == 'yes' and field == 'action':
        return f'the {word} ({pid})' if word else pid
    if word:
        return ('our ' if lvl <= 2 else 'the ')+word
    return generic


def _metres(x, reg, metres):
    if not _num(x) or not _numbers_at_least(reg, 'rounded_metres') or metres == 'unusable':
        return None
    v = int(round(float(x)))
    return f'about {v} m' if metres == 'estimated' or reg['numbers'] == 'rounded_metres' else f'{v} m'


def _count(x, reg):
    if not _num(x) or not _numbers_at_least(reg, 'counts'):
        return None
    return str(int(round(float(x))))


def _count_word(x, reg):
    """A count of players, with 0 read as "none" ("none of our players were behind the ball")."""
    c = _count(x, reg)
    return 'none' if c == '0' else c


def _margin(m):
    """The rest-defence margin with its sign (+1, -2); 0 stays 0 (the wave-1 text printed "+0")."""
    m = int(round(float(m)))
    return f'{m:+d}' if m else '0'


def _share(x, reg, whole=None):
    """A share as "about N%" from the shares register. At the ends a share is a count, not an estimate (COACH L5:
    "about 100% of the 16 advances"): 1 reads `whole` ("all") when given, else None so a count phrasing fills; 0 is
    None."""
    if not _num(x) or not _numbers_at_least(reg, 'shares'):
        return None
    if float(x) >= SHARE_ALL:
        return whole
    if float(x) <= 1.0-SHARE_ALL:
        return None
    return f'about {int(round(100*float(x)))}%'


def _players_words(x, reg, noun):
    """A count of players for a description: 'no defender' / 'one defender' / '3 defenders' (noun None: 'nobody' /
    'one' / '3'); None where the level shows no numbers."""
    if not _num(x) or not _numbers_at_least(reg, 'counts'):
        return None
    n = int(round(float(x)))
    if noun is None:
        return 'nobody' if n == 0 else ('one' if n == 1 else str(n))
    return f'no {noun}' if n == 0 else (f'one {noun}' if n == 1 else f'{n} {noun}s')


def _claims_of(claims, kind=None, relation=None):
    return [c for c in claims if (kind is None or c['type'] == kind) and
            (relation is None or c['facts'].get('relation') == relation)]


def _first(claims, kind, relation=None, **match):
    for c in _claims_of(claims, kind, relation):
        if all(c['facts'].get(k) == v for k, v in match.items()):
            return c['facts']
    return None


def _rel(claims, relation):
    for c in claims:
        if c['type'] in ('relation', 'team_state') and c['facts'].get('relation') == relation and \
                c['facts'].get('scope') != 'demonstrated':
            return c['facts']
    return None


def _slots(finding, claims, reg, field):
    """Slot values for the templates of one field."""
    lvl = reg['level']
    det = finding['observation']
    sit = _first(claims, 'situation')
    rec = _first(claims, 'recurrence') or {}
    s = {'try': 'Next step' if reg['adult'] else 'Try this', 'per': str(PER_MINUTES),
         'channels_total': None}
    if sit['phase'] in ('own', 'opp') and sit['source'] in QUALIFIER:
        who = 'we had the ball' if sit['phase'] == 'own' else 'the other team had the ball'
        lead = ('when', 'while')[_hash(finding['key'], 'phase', lvl) % 2]
        s['phase'] = f'{lead} {who} {QUALIFIER[sit["source"]]}'
        s['Phase'] = _cap(s['phase'])
        poss = 'our possession' if sit['phase'] == 'own' else "the other team's possession"
        s['poss_of'] = f'of {poss} {QUALIFIER[sit["source"]]}'
        s['phase_poss'] = s['poss_of']
    if sit.get('zone') in PLACE:
        s['place'] = PLACE[sit['zone']][0 if lvl <= 2 else 1]
    if sit.get('channel') in SIDE_PLACE:
        s['side_place'] = SIDE_PLACE[sit['channel']]
        if sit['channel'] in OTHER_SIDE:
            s['ball_side_place'] = SIDE_PLACE[sit['channel']]
    if _num(sit.get('t')):
        s['t'] = L.fmt_time(sit['t'])
    actor = _first(claims, 'actor_state') or {}
    generic_actor = {'pressure_cover': 'one of our players', 'wide_defender_advance': 'one of our defenders',
                     'defensive_spacing': 'one of our defenders'}.get(det, 'one of our players')
    s['actor'] = _ref_text(actor, reg, field, generic_actor)
    s['Actor'] = _cap(s['actor'])
    if actor.get('side') in ('left', 'right'):
        s['side'] = actor['side']
        s['side_place'] = SIDE_PLACE[actor['side']]
    metres = actor.get('metres', 'estimated')
    # fix round 3 (FACTS F-1): the margin every count of ours behind the ball was measured with ("at least 10 m",
    # "well" below the rounded-metres register or without usable metres)
    s['bm'] = behind_words(sit.get('behindMarginM'), sit.get('behindMetres'), reg)
    s['Bm'] = _cap(s['bm'])
    opp_ch = _first(claims, 'opponent_state', state='in_channel')
    if opp_ch and _num(opp_ch.get('count')):
        # the opponents in the space an advancing defender left (levels 1-2, an exposed advance): "one of theirs" at
        # level 1 (no numbers), the count from level 2
        n = int(opp_ch['count'])
        s['opp_in'] = 'one of theirs was' if n == 1 else (f'{n} of theirs were' if _numbers_at_least(reg, 'counts')
                                                          else 'some of theirs were')
    if det == 'pressure_cover':
        cov = _rel(claims, 'cover') or {}
        # fix round 3 (FACTS F-7): the cover candidate is the nearest teammate WITHIN the cover angle; unnamed it is
        # "a teammate" (never "the nearest teammate"), and the appositive phrasings need the named one (mate_named)
        s['mate'] = _ref_text(cov, reg, field, 'a teammate')
        s['mate_named'] = _ref_text(cov, reg, field, None)
        s['d1'] = _metres(actor.get('distanceM'), reg, metres)
        s['cover_dist'] = _metres(cov.get('distanceM'), reg, metres)
        if _num(cov.get('angleDeg')) and _numbers_at_least(reg, 'metres'):
            s['cover_angle'] = f'about {int(round(cov["angleDeg"]))} degrees'
    elif det == 'weak_side_balance':
        ws = _rel(claims, 'weak_side') or {}
        s['weak_n'] = _count(ws.get('count'), reg) if (ws.get('count') or 0) >= 2 else None
        s['weak_count_words'] = _players_words(ws.get('count'), reg, 'defender')
        if _num(ws.get('count')):
            s['weak_were'] = 'was' if int(ws['count']) <= 1 else 'were'
        cen = _first(claims, 'team_state', state='central') or {}
        s['central_n'] = _count(cen.get('count'), reg)
        ow = _first(claims, 'relation', 'open_width') or {}
        s['open_width'] = _metres(ow.get('distanceM'), reg, ow.get('metres', metres))
        if s['open_width']:
            # with nobody goal-side the measure is the whole far half (W/2), said as such (FACTS M7)
            s['open_space'] = (f'nobody was goal-side across the whole far half, {s["open_width"]} wide'
                               if ow.get('wholeHalf') else
                               f'the far touchline was {s["open_width"]} from the widest goal-side defender')
        lev = _first(claims, 'team_state', state='level')
        s['level_words'] = _players_words(lev['count'], reg, None) if lev else None
        recov = _first(claims, 'team_state', state='recovering')
        s['rec_words'] = _players_words(recov['count'], reg, None) if recov else None
    elif det == 'wide_defender_advance':
        s['advance'] = _metres(actor.get('distanceM'), reg, metres)
        ch = _rel(claims, 'channel_cover') or {}
        s['mate'] = _ref_text(ch, reg, field, 'a teammate')
        # the near centre-back and the holding midfielder are named (role words, ids) only with a confirmed role;
        # otherwise the sentence speaks of the group ("no midfielder dropped")
        cb = _rel(claims, 'cb_shift') or {}
        s['Cb'] = _cap(_ref_text(cb, reg, field, '')) if cb.get('role') else None
        dm = _rel(claims, 'dm_drop') or _first(claims, 'relation', 'dm_cb_spacing') or {}
        s['dm_ref'] = _ref_text(dm, reg, field, '') if dm.get('role') else None
        s['dm_ref_cap'] = _cap(s['dm_ref']) if s['dm_ref'] else None
        # the shift the near centre-back made, when measured (levels 5-7 state it: critic S-10)
        shift = _metres(cb.get('distanceM'), reg, cb.get('metres', metres)) \
            if cb.get('present') is False and _num(cb.get('distanceM')) and cb['distanceM'] >= .5 else None
        cb_no = 'The near centre-back did not move across' + (f' (a shift of {shift} toward the ball)' if shift
                                                               else '')
        if cb.get('present') is False and dm.get('present') is False:
            s['cb_dm_fail'] = (cb_no+', and '+(s['dm_ref'] or 'no midfielder') +
                               (' did not drop.' if s['dm_ref'] else ' dropped toward the back line.'))
        elif cb.get('present') is False:
            s['cb_dm_fail'] = cb_no+'.'
        elif dm.get('present') is False:
            s['cb_dm_fail'] = (_cap(s['dm_ref'])+' did not drop.') if s['dm_ref'] else \
                'No midfielder dropped toward the back line.'
        s['dm_mate'] = s['dm_ref'] or 'a midfielder'
        # who covered at a strength's moment (polish round): the opening sentence says it, so the context sentence
        # does not say it again
        cover_by = actor.get('coverBy')
        ok = []
        # the channel held by a centre-back IS the near centre-back moving across: one player, said once (COACH
        # L5.4: "the left centre-back (A-02) in the channel behind. The near centre-back moved across" read as two)
        same = cb.get('present') is True and ch.get('present') is True and \
            (_role_of(finding, m_channel_by(finding)) or {}).get('role') in CB_ROLES
        if cb.get('present') is True and cover_by != 'cb':
            ok.append('the near centre-back moved across and was the one in the channel' if same else
                      'the near centre-back moved across to cover')
        if dm.get('present') is True and cover_by != 'dm':
            ok.append((s['dm_ref'] or 'a midfielder')+' dropped toward the back line')
        neg = []
        if (_rel(claims, 'advance_support') or {}).get('present') is True:
            if cb.get('present') is False:
                neg.append('the near centre-back stayed central' + (f' (a shift of {shift})' if shift else ''))
            if dm.get('present') is False:
                neg.append((s['dm_ref']+' did not drop') if s['dm_ref'] else
                           'no midfielder dropped toward the back line')
        if neg:
            s['cb_dm_also'] = 'Besides that cover, '+' and '.join(neg)+'.'
        if ok:
            s['cb_dm_ok_low'] = ' and '.join(ok)+'.'
            s['cb_dm_ok'] = _cap(s['cb_dm_ok_low'])
        beh = _rel(claims, 'behind_ball') or {}
        s['behind'] = _count_word(beh.get('count'), reg)
        opp = _first(claims, 'opponent_state', state='in_channel')
        if opp and _numbers_at_least(reg, 'counts'):
            s['opp_count'] = 'one opponent was' if opp['count'] == 1 else f'{opp["count"]} opponents were'
            s['Opp_count'] = _cap(s['opp_count'])
        rm = _first(claims, 'team_state', state='rest_margin')
        if rm and _numbers_at_least(reg, 'counts'):
            s['margin'] = _margin(rm['margin'])
            s['threat'] = _count_word(rm.get('threat'), reg)
        spc = _first(claims, 'relation', 'dm_cb_spacing')
        if spc:
            s['dm_cb'] = _metres(spc.get('distanceM'), reg, spc.get('metres', metres))
            s['dm_holder'] = s['dm_ref'] or 'the nearest midfielder in a holding role'
            s['Dm_holder'] = _cap(s['dm_holder'])
    elif det == 'behind_ball':
        beh = _rel(claims, 'behind_ball') or {}
        s['behind'] = _count_word(beh.get('count'), reg)
        s['required'] = _count(beh.get('required'), reg)
        thr = _first(claims, 'opponent_state', state='threat')
        if thr and _numbers_at_least(reg, 'counts'):
            s['threat'] = _count_word(thr['count'], reg)
            s['threat_were'] = 'was' if thr['count'] == 1 else 'were'
            s['threat_players'] = 'player' if thr['count'] == 1 else 'players'
        rm = _first(claims, 'team_state', state='rest_margin')
        if rm and _numbers_at_least(reg, 'counts'):
            s['margin'] = _margin(rm['margin'])
            s['threat'] = _count_word(rm.get('threat'), reg)
    elif det == 'defensive_spacing':
        gap = _rel(claims, 'lateral_gap') or {}
        s['mate'] = _ref_text(gap, reg, field, 'the next defender') if gap else None
        s['gap'] = _metres(gap.get('distanceM'), reg, gap.get('metres', metres))
        ctxgap = _first(claims, 'relation', 'lateral_gap_context') or {}
        s['gap_any'] = _metres(ctxgap.get('distanceM'), reg, ctxgap.get('metres', metres))
        lg = _rel(claims, 'unit_gap') or {}
        s['dm_gap'] = _metres(lg.get('distanceM'), reg, lg.get('metres', metres))
        ev = _rel(claims, 'line_evenness') or {}
        s['line_std'] = _metres(ev.get('distanceM'), reg, ev.get('metres', metres))
        blk = _first(claims, 'team_state', state='block')
        s['block'] = blk['height'] if blk else None
        btw = _first(claims, 'opponent_state', state='between_lines')
        s['between_n'] = _count_word(btw['count'], reg) if btw else None
    elif det == 'attacking_shape':
        w = _rel(claims, 'width') or {}
        d = _rel(claims, 'depth') or {}
        s['width'] = _metres(w.get('distanceM'), reg, w.get('metres', metres))
        s['depth'] = _metres(d.get('distanceM'), reg, d.get('metres', metres))
        s['width_share'] = _share(w.get('share'), reg, whole='all')
        s['depth_share'] = _share(d.get('share'), reg, whole='all')
        # the length used up to the opponents' last line (critic B-7) and our highest player against that line
        if _num(d.get('availShare')) and _numbers_at_least(reg, 'shares'):
            s['avail_used'] = ('the whole length' if d['availShare'] >= SHARE_ALL else
                               f'{_share(d["availShare"], reg)} of the length')
        hg = _first(claims, 'relation', 'height_gap')
        hm = _metres(abs(hg['distanceM']), reg, hg.get('metres', metres)) if hg else None
        if hm:
            s['height_gap_words'] = ('level with their last line' if abs(hg['distanceM']) < .5 else
                                     f'{hm} short of their last line' if hg['distanceM'] > 0 else
                                     f'{hm} beyond their last line')
        dp = _first(claims, 'team_state', state='deep_support')
        # fix round 3: none of ours the margin behind the ball means the ball was with our own back line (our deepest
        # outfield player within the margin of it): build-up, not missing support, so the clause is left out
        s['deep_n'] = _count_word(dp['count'], reg) if dp and int(dp['count']) >= 1 else None
        ch = _rel(claims, 'channels') or {}
        s['channels'] = _count(ch.get('count'), reg)
        s['channels_total'] = _count(CHANNELS_TOTAL, reg) if ch else None
        if lvl >= 3 and s['channels_total']:
            s['channels_total'] = 'three'
        if s['channels'] and s['channels_total']:
            # "each of the three channels" / "2 of the three channels" (not "3 of the three"; "all three channels"
            # is the channels relation's polarity phrase, which a description does not state)
            s['channels_of'] = (f"each of the {s['channels_total']}" if ch.get('count') == CHANNELS_TOTAL else
                                f"{s['channels']} of the {s['channels_total']}")
    # number agreement for counted players; a count of 0 reads "none of our players" (the wave-1 text said "0 of
    # our players were behind the ball" on real findings) and the "only N" phrasing becomes "none"
    beh = _rel(claims, 'behind_ball') or {}
    if _num(beh.get('count')) and s.get('behind'):
        n = int(beh['count'])
        s['were'] = 'was' if n == 1 else 'were'
        s['players'] = 'player' if n == 1 else ('players' if n else None)
        s['only_behind'] = 'none' if n == 0 else f'only {s["behind"]}'
    cen = _first(claims, 'team_state', state='central')
    if cen and _num(cen.get('count')) and _numbers_at_least(reg, 'counts'):
        n = int(cen['count'])
        s['central_def'] = 'no defender' if n == 0 else f'{n} defender'+('' if n == 1 else 's')
        s['Central_def'] = _cap(s['central_def'])
        s['central_were'] = 'was' if n <= 1 else 'were'
    # recurrence
    if rec:
        s['fail_n'] = _count(rec.get('failing'), reg)
        s['dem_n'] = _count(rec.get('demonstrated'), reg)
        s['of_n'] = _count(rec.get('opportunities'), reg)
        s['spells'] = _count(rec.get('failingSpells'), reg) if rec.get('failingSpells', 0) >= 2 else None
        s['dem_spells'] = _count(rec.get('demonstratedSpells'), reg) if rec.get('demonstratedSpells', 0) >= 2 \
            else None
        s['fail_share'] = _share(rec.get('failShare'), reg)
        # fix round 3 (FACTS F-6 = COACH F-3): a strength's count and share are of the moments PitchIQ could judge
        # (decided), "all N" when every one showed it, the unclear ones said after it; level 1 has no numbers, so
        # its count is said as words (often: COACH F-11)
        dec = rec.get('decided') if _num(rec.get('decided')) else None
        s['dem_share'] = _share(rec.get('decidedShare'), reg) if dec else _share(rec.get('demonstratedShare'), reg)
        if dec and _numbers_at_least(reg, 'counts'):
            dem, dec_n = int(rec.get('demonstrated') or 0), int(dec)
            s['dec_n'] = _count(dec_n, reg)
            s['dem_of'] = f'all {dec_n}' if dem >= dec_n else f'{dem} of the {dec_n}'
            s['Dem_of'] = _cap(s['dem_of'])
            unc = int(rec.get('unclear') or 0)
            s['unc_tail'] = '' if unc <= 0 else ('; one more could not be judged' if unc == 1 else
                                                 f'; {unc} more could not be judged')
        s['often'] = often_words(rec.get('demonstrated') if finding['verdict'] == 'strength' else rec.get('failing'))
        phase_s = rec.get('phaseSeconds') if _num(rec.get('phaseSeconds')) else 0.0
        if _num(rec.get('ratePer10')) and _numbers_at_least(reg, 'rates') and phase_s >= RATE_MIN_PHASE_S-1e-9:
            # from at least 5 minutes of the phase only; a whole number below 10 minutes (COACH L5)
            r = float(rec['ratePer10'])
            s['rate'] = f'{r:.1f}' if phase_s >= RATE_DECIMAL_PHASE_S-1e-9 else f'{int(round(r))}'
        if phase_s > 0 and _numbers_at_least(reg, 'metres'):
            s['phase_time'] = L.fmt_duration(phase_s)
        if _num(rec.get('judgedSeconds')) and rec['judgedSeconds'] > 0 and _numbers_at_least(reg, 'metres'):
            s['judged'] = L.fmt_duration(rec['judgedSeconds'])
    mix = _first(claims, 'limitation', code='mixed_evidence')
    if mix and _numbers_at_least(reg, 'counts'):
        s['decided'] = _count(mix.get('decided'), reg)
        s['mixed_fail_n'] = _count(mix.get('failing'), reg)
        s['mixed_fail_share'] = _share(mix.get('failShare'), reg)
        s['base_share'] = _share(mix.get('baseShare'), reg)
    s['base_words'], s['base_src'] = base_words_for(reg, lvl)
    mode = _claim_mode(claims)
    variant = (_first(claims, 'situation') or {}).get('variant')
    nouns = (_bank(det, 'noun:'+mode, lvl) if mode else None) or \
        (_bank(det, 'noun:'+variant, lvl) if variant else None) or _bank(det, 'noun', lvl)
    s['noun'] = nouns[0] if nouns else None
    # counterfactual
    cf = _first(claims, 'counterfactual')
    if cf:
        # fix round 3 (FACTS F-7's class): ZONES' mover is the least-effort one of the 2-3 teammates nearest the
        # region, not necessarily the nearest teammate, so unnamed he is "a teammate" at every level
        s['mover'] = _ref_text(cf, reg, field, 'a teammate')
        s['Mover'] = _cap(s['mover'])
    restore = _bank(det, 'restore', lvl)
    if restore:
        s['restore'] = restore[0].format(side=s.get('side', 'wide'))
        s['restored'] = restore[1].format(side=s.get('side', 'wide'))
    # style
    st = _first(claims, 'intent_signal', signal='style')
    if st:
        s['style'] = STYLE_WORD[st['style']]
        s['effect'] = STYLE_EFFECT[st['effect']]
    # limitations
    na = _first(claims, 'limitation', code='no_alternative')
    if na:
        s['no_alt_plain'], s['no_alt'] = NO_ALT_WORDS[na.get('wording') if na.get('wording') in NO_ALT_WORDS
                                                      else na['reason']]
    lims = [c['facts']['code'] for c in claims if c['type'] == 'limitation' and c['facts']['code'] in LIMITATION_WORDS]
    if lims:
        words = [LIMITATION_WORDS[c][0 if lvl <= 3 else 1] for c in lims]
        words = list(dict.fromkeys(words))
        s['limits'] = words[0] if len(words) == 1 else ', '.join(words[:-1])+' and '+words[-1]
    cap = _first(claims, 'limitation', code='level_cap')
    if cap:
        s['max_items'] = _count(cap.get('maxWorkItems'), reg)
    mw = _first(claims, 'limitation', code=MOMENTS)
    if mw:
        s.update(_moments_slots(det, mw, claims, reg, s))
    return s


def _times_words(times):
    """'48:18.77, 49:11.77 and 49:32.77' (language.fmt_time of each, in the order given)."""
    words = [L.fmt_time(t) for t in times]
    return words[0] if len(words) == 1 else ', '.join(words[:-1])+' and '+words[-1]


def _moments_slots(det, mw, claims, reg, s):
    """The slots of a 'moments worth a look' review (fix round 2): its counts in the level's number mode, the
    opportunities counted (every mode of the family), the cited moments, and what the clear moments had in common
    (moments_what, the D4 margin rule and a required count of one worded apart). Counts need the counts register."""
    lvl = reg['level']
    out = {}
    nouns = _bank(det, 'noun_moments', lvl) or _bank(det, 'noun', lvl)
    out['mw_noun'] = nouns[0] if nouns else None
    times = [x['t'] for x in mw.get('moments') or () if _num(x.get('t'))]
    out['mw_times'] = _times_words(times) if times else None
    shown = len(times) if times and len(times) < int(mw['clear']) else None
    out['mw_shown'] = _count(shown, reg) if shown else None
    out['Mw_shown'] = out['mw_shown']
    if _numbers_at_least(reg, 'counts'):
        out['mw_clear'] = _count(mw['clear'], reg)
        out['mw_decided'] = _count(mw['decided'], reg)
        every = int(mw['clear']) >= int(mw['decided'])
        out['mw_of'] = f'all {out["mw_decided"]}' if every else f'{out["mw_clear"]} of the {out["mw_decided"]}'
        out['mw_of_short'] = f'All {out["mw_decided"]}' if every else f'{out["mw_clear"]} of {out["mw_decided"]}'
        out['mw_spells'] = _count(mw['spells'], reg)
        out['mw_share'] = _share(mw.get('clearShare'), reg)
        out['mw_dem'] = _count(mw['demonstrated'], reg) if mw.get('demonstrated') else None
        out['mw_dem_share'] = _share(mw.get('demonstratedShare'), reg) if mw.get('demonstrated') else None
        unc = int(mw.get('unclear') or 0)
        out['Mw_unclear_more'] = None if unc <= 0 else ('One more' if unc == 1 else f'{unc} more')
        out['mw_unclear_n'] = _count(unc, reg) if unc > 0 else None
        out['mw_unclear_is'] = 'is' if unc == 1 else 'are'
        out['mw_unclear_was'] = 'was' if unc == 1 else 'were'
        # the judged ones that fell short without being clear (fix round 3, FACTS F-4 = COACH F-10)
        more = int(mw.get('more') or 0)
        out['mw_more'] = None if more <= 0 else ('one' if more == 1 else _count(more, reg))
        out['Mw_more'] = _cap(out['mw_more']) if out['mw_more'] else None
        out['mw_more_are'] = 'is' if more == 1 else 'are'
        out['mw_more_those'] = 'that one is' if more == 1 else 'those are'
    slot = 'moments_what'
    beh = _rel(claims, 'behind_ball') or {}
    if det == 'behind_ball':
        if 'requiredMargin' in beh:
            slot = 'moments_what:margin'
        elif _num(beh.get('required')):
            req = int(beh['required'])
            out['mw_required'] = _count(req, reg) if req >= 2 else None
    cands = _bank(det, slot, lvl) or []
    out['mw_what'] = next((x for x in (_fill(c, dict(s, **out)) for c in cands) if x), None)
    out['Mw_what'] = _cap(out['mw_what']) if out['mw_what'] else None
    return out


# ---------------------------------------------------------------------------------------------------------------
# Planning and realisation (DESIGN 6.4)

class _ReportState:
    """Per-report memory: the phrasings already used (a report never repeats one for the same slot), the concepts
    already defined, and a salt that shifts every choice when two explanations came out too alike."""

    def __init__(self):
        self.used = set()
        self.defined = set()
        self.current = []            # the phrasings the finding being written has taken

    def snapshot(self):
        return set(self.used), set(self.defined)

    def restore(self, snap):
        self.used, self.defined = set(snap[0]), set(snap[1])


def _pick(rep, finding, det, slot, level, slots, salt=0, minimal=False, used_key=None):
    """One sentence for a slot: the first unused phrasing (starting at a hash of the finding key and the slot,
    offset by the level) whose slots all have values. minimal: always the first phrasing that fills. A try_this
    headline is framed "Try this:" / "Next step:" whenever the bank has such a phrasing; other verdicts never are."""
    cands = _bank(det, slot, level) if det else None
    if cands is None:
        cands = _bank('*', slot, level)
        det_key = '*'
    else:
        det_key = det
    if not cands:
        return None
    if slot.startswith('headline_fail'):
        framed = [c for c in cands if '{try}' in c]
        cands = (framed or cands) if finding['verdict'] == 'try_this' else [c for c in cands if '{try}' not in c]
        if not cands:
            return None
    n = len(cands)
    # a description reads its level's own relation: the first phrasing that fills is the level's (S-6), the later
    # ones are fallbacks for missing measurements, so the hash does not pick among them
    # (fix round 2: likewise the 'moments worth a look' slots, whose first phrasing shows the most the level allows)
    ordered = slot.startswith(('describe', 'moments_', 'look_for', 'limits_moments', 'short_moments'))
    start = 0 if minimal or ordered else (_hash(finding['key'], slot, salt)+level) % n
    key = used_key or (det_key if '{noun}' not in ' '.join(cands) else det, slot)
    fallback = None
    for i in range(n):
        idx = (start+i) % n
        text = _fill(cands[idx], slots)
        if text is None:
            continue
        if minimal:
            return text
        if (key, idx) not in rep.used:
            rep.used.add((key, idx))
            rep.current.append((key, idx))
            return text
        if fallback is None:
            fallback = text
    return fallback


def _plan(finding, claims, reg, minimal=False):
    """The slot order of each field for the verdict and level (DESIGN 6.4)."""
    verdict = finding['verdict']
    lvl = reg['level']
    det = finding['observation']
    rec = _first(claims, 'recurrence') or {}
    one = verdict == 'review' and (rec.get('failingSpells', 0) <= 1 or
                                   _first(claims, 'limitation', code='one_instance') is not None)
    moments = is_moments(finding)
    if minimal:
        if moments:
            return ['moments_count', 'moments_more', 'moments_frame', 'moments_times'], ['look_for_plain']
        exp = {'strength': ['open_ok'], 'describe': ['describe'], 'review': ['open_fail', 'limits'],
               'try_this': ['open_fail'], 'work_on': ['open_fail']}[verdict]
        act = {'strength': [], 'describe': [], 'review': ['review_action'], 'try_this': ['question'],
               'work_on': ['principle']}[verdict]
        return exp, act
    # levels 5-7 of the wide-defender check state every relation they measured, one sentence each (critic S-10)
    d3_deep = det == 'wide_defender_advance' and lvl >= 5
    context = ['cb_dm', 'behind_ctx', 'rest', 'spacing', 'opp_channel'] if d3_deep else ['context_fail']
    if verdict == 'strength':
        exp = ['open_ok', 'help_ok' if lvl == 1 else None, 'context_ok', 'context_also' if d3_deep else None,
               'behind_ctx' if d3_deep else None, 'rest' if d3_deep else None, 'spacing' if d3_deep else None,
               'interchange', 'recurrence_ok', 'judged', 'formation_changing', 'moment', 'principle']
        if (_first(claims, 'actor_state') or {}).get('coverBy') == 'swap':
            # the interchange was the cover (polish round): said right after the advance, as the one who covered
            exp.remove('interchange')
            exp.insert(1, 'interchange')
        act = ['build_on']
    elif verdict == 'describe':
        # 'style': a description the declared style explains says so (COACH M2: BC180 A at L7-compact named the
        # intent limitation, not the style claim "open space on the far side is accepted")
        exp = ['describe', 'style', 'mixed_evidence', 'formation_changing', 'interchange', 'limits', 'no_moment',
               'moment']
        act = []
    elif verdict == 'try_this':
        exp = ['praise', 'open_fail']+(context if lvl >= 3 else [])+['recurrence_fail', 'formation_changing',
                                                                    'moment', 'principle' if lvl >= 2 else None]
        act = ['no_alternative', 'question']
    elif verdict == 'work_on':
        exp = ['open_fail']+context+['block', None if d3_deep else 'spacing', 'recurrence_fail',
                                     'formation_changing', 'moment', 'hypo', 'principle', 'style']
        act = ['counterfactual', 'no_alternative', 'question' if lvl <= 3 else 'statement']
    elif moments:
        # 'moments worth a look' (fix round 2): the clear moments of the judged ones, the unclear ones, why it is no
        # verdict, the moments, the first one described, its context, why the check matters; the action says what
        # to look for in the clips (no zone, no question, no drill)
        exp = ['moments_count', 'moments_more', 'moments_unclear', 'moments_frame', 'moments_times', 'open_fail'] + \
            context+['formation_changing', 'limits', 'why']
        act = ['look_for']
    else:   # review
        exp = ['open_fail']+context+['limits', 'level_cap', 'one_instance' if one else 'recurrence_fail',
                                     'formation_changing', 'moment', 'principle']
        act = ['review_action', 'no_alternative']
    return [x for x in exp if x], act


def _claim_mode(claims):
    """The failure mode build_claims fixed (situation claim), or None."""
    return (_first(claims, 'situation') or {}).get('mode')


def _mode_slot(finding, claims, slot, level):
    """slot:mode (the failure mode) or slot:variant (level 1's reframed advance) when the bank has it, else slot."""
    sit = _first(claims, 'situation') or {}
    det = finding['observation']
    # 'unknown': the phase named no team (critic B-4: phase-free headlines and shorts)
    unknown = 'unknown' if sit.get('phase') not in ('own', 'opp') else None
    for mode in (sit.get('mode'), sit.get('variant'), unknown):
        # a mode bank empty at this level ([]: fix round 3, the level 1-2 'exposed' words) falls through
        if mode and _bank(det, f'{slot}:{mode}', level):
            return f'{slot}:{mode}'
    return slot


def _open_ok_slot(finding, claims, level):
    """The opening sentence's slot of a strength: a wide defender's says who covered at its moment (polish round:
    'open_ok' the teammate in the channel, 'open_ok:cb' / ':dm' the centre-back's shift / a midfielder's drop,
    ':covered' a cover the claims do not attribute, an interchange said by its own slot), else _mode_slot's."""
    if finding['observation'] == 'wide_defender_advance' and level >= 3:
        cover = (_first(claims, 'actor_state') or {}).get('coverBy')
        name = {'channel': None, 'cb': 'open_ok:cb', 'dm': 'open_ok:dm'}.get(cover, 'open_ok:covered')
        if cover != 'channel' and (_rel(claims, 'channel_cover') or {}).get('present') is True:
            name = None                 # (a channel claim present wins: the teammate there is the cover)
        if name and _bank('wide_defender_advance', name, level):
            return name
    # fix round 3 (critic B-1, ASSESS's request (b)): levels 1-2 praise a covered advance wherever the ball is; "someone
    # stayed back" is said only with one of ours behind the ball at the moment (the behind-the-ball claim), else the
    # cover itself ("the space they left was covered" at level 2, "a teammate looked after the space" at level 1)
    behind = (_rel(claims, 'behind_ball') or {}).get('present') is True
    if finding['observation'] == 'wide_defender_advance' and level == 2 and not behind and \
            _bank('wide_defender_advance', 'open_ok:covered', level):
        return 'open_ok:covered'
    if finding['observation'] == 'behind_ball' and variant_of(finding) == 'advance' and level == 1 and \
            (not behind or (_rel(claims, 'channel_cover') or {}).get('present') is True) and \
            _bank('behind_ball', 'open_ok:advance_covered', level):
        return 'open_ok:advance_covered'
    return _mode_slot(finding, claims, 'open_ok', level)


def _principle_text(claims, level):
    p = _first(claims, 'principle')
    if not p:
        return None
    return _at(PRINCIPLES[p['principle']]['text'], level)


def _realize(finding, claims, reg, rep, salt=0, minimal=False, repeats=None):
    """Text fields for one finding from its claims, or None when a required sentence cannot be filled. repeats:
    (earlier explanations of the report, slots): a sentence of one of those slots that an earlier explanation already
    contains word for word is left out (_drop_repeats; every slot named there is in OPTIONAL_SLOTS)."""
    lvl = reg['level']
    det = finding['observation']
    verdict = finding['verdict']
    exp_plan, act_plan = _plan(finding, claims, reg, minimal)
    slots_e = _slots(finding, claims, reg, 'explanation')
    slots_a = _slots(finding, claims, reg, 'action')
    slots_h = _slots(finding, claims, reg, 'headline')
    pick = lambda slot, slots, **kw: _pick(rep, finding, det if _bank(det, slot, lvl) is not None else None,  # noqa
                                           slot, lvl, slots, salt=salt, minimal=minimal, **kw)
    sentences = []
    optional = set()          # sentences the length guard may drop (OPTIONAL_SLOTS)
    moments = is_moments(finding)
    for slot in exp_plan:
        if slot == 'principle':
            text = _principle_text(claims, lvl)
        elif slot in ('moments_count', 'moments_more', 'moments_unclear', 'moments_frame', 'moments_times'):
            text = pick(slot, slots_e)
            if text is None and slot == 'moments_count':
                return None             # a 'moments worth a look' review always says how many, of how many
            if text is None and slot == 'moments_more' and slots_e.get('mw_more'):
                return None             # ... and the judged ones that fell short less clearly (fix round 3)
        elif slot == 'why':
            # why the check matters: the level's principle, introduced as such ("Why it matters: ...")
            p = _principle_text(claims, lvl)
            text = _bank('*', 'moments_why', lvl)[0]+_lower_word(p.replace(': ', '; ')) if p else None
        elif slot == 'praise':
            if not _claims_of(claims, None, None) or not [c for c in claims if c['facts'].get('scope') ==
                                                           'demonstrated']:
                continue
            text = pick(_mode_slot(finding, claims, 'praise', lvl), slots_e)
        elif slot == 'limits':
            # a description is not a review: its own wording ("this is a description only"); nor is a 'moments worth
            # a look' review held back by what was not confirmed ("It stays a review because ..." would be false)
            name = 'limits_describe' if verdict == 'describe' else ('limits_moments' if moments else 'limits')
            text = pick(name, slots_e) if slots_e.get('limits') else None
        elif slot in ('level_cap', 'no_moment', 'mixed_evidence', 'formation_changing'):
            if not _first(claims, 'limitation', code=slot):
                continue
            text = pick(slot, slots_e)
        elif slot == 'help_ok':
            # only with the level-1 "a teammate close behind" claim (the wave-5 plan said it whatever the cover was:
            # 18 level-1 scene strengths then fell back to one sentence, the gate refusing the planned text)
            if not _first(claims, 'relation', 'cover', scope='help'):
                continue
            text = pick('help_ok', slots_e)
        elif slot == 'context_fail':
            if det == 'defensive_spacing' and not _first(claims, 'opponent_state', state='in_gap'):
                continue
            text = pick('context_fail', slots_e)
        elif slot == 'context_ok':
            if det != 'wide_defender_advance' or not (_rel(claims, 'cb_shift') or {}).get('present') and not \
                    (_rel(claims, 'dm_drop') or {}).get('present'):
                continue
            text = pick('context_ok', slots_e)
        elif slot == 'interchange':
            if not _first(claims, 'intent_signal', signal='interchange'):
                continue
            text = pick('interchange', slots_e)
        elif slot == 'hypo':
            if not _first(claims, 'hypothetical'):
                continue
            text = pick(_mode_slot(finding, claims, 'hypo', lvl), slots_e)
        elif slot == 'style':
            st = _first(claims, 'intent_signal', signal='style') or {}
            if verdict == 'describe' and st.get('effect') in STYLE_DESCRIBE_SILENT:
                continue        # fix round 3 (COACH F-11): a weighting of strengths changes nothing in a description
            text = pick('style', slots_e) if slots_e.get('style') else None
        elif slot in ('block', 'spacing'):
            if det == 'defensive_spacing' and slot == 'spacing' and _claim_mode(claims) != 'lateral_gap':
                continue            # the D-M distance is the finding itself in the dm_gap mode
            text = pick(slot, slots_e)
        elif slot in ('open_fail', 'describe'):
            name = _mode_slot(finding, claims, slot, lvl)
            if slot == 'describe' and det == 'attacking_shape' and 'phase' not in slots_e:
                name = 'describe_unknown'
            text = pick(name, slots_e)
            if text is None:
                return None
            if moments and slot == 'open_fail':
                # the first cited moment (the measurements are its own: DETECT section 4), after the moments list
                text = _bank('*', 'moments_first', lvl)[0]+_lower_word(text)
        elif slot == 'open_ok':
            text = pick(_open_ok_slot(finding, claims, lvl), slots_e)
            if text is None:
                return None
        else:
            text = pick(_mode_slot(finding, claims, slot, lvl), slots_e)
        if text:
            sentence = _sentence(text)
            if repeats and slot in repeats[1] and slot in OPTIONAL_SLOTS and any(sentence in e for e in repeats[0]):
                continue            # said word for word by an earlier explanation of the report (_drop_repeats)
            sentences.append(sentence)
            if slot in OPTIONAL_SLOTS:
                optional.add(sentences[-1])
    if not sentences:
        return None
    actions = []
    for slot in act_plan:
        if slot == 'counterfactual':
            cf = _first(claims, 'counterfactual')
            if not cf:
                continue
            name = 'counterfactual_self' if cf.get('moverIsActor') and lvl >= 3 else \
                'counterfactual_direction' if cf.get('display') == 'direction' else 'counterfactual_zone'
            text = pick(name, slots_a)
        elif slot in ('look_for', 'look_for_plain'):
            text = pick(slot, slots_a)
        elif slot == 'review_action':
            qs = _bank(det, _mode_slot(finding, claims, 'review_q', lvl), lvl)
            q = qs[0].format(side=slots_a.get('side', 'wide')) if qs else None
            text = pick('review_action', dict(slots_a, q=q)) if q else None
        elif slot == 'principle':
            text = _principle_text(claims, lvl)
        elif slot == 'no_alternative':
            text = pick('no_alternative', slots_a) if slots_a.get('no_alt') else None
        else:
            text = pick(_mode_slot(finding, claims, slot, lvl), slots_a)
        if text:
            actions.append(_sentence(text))
        if actions and slot in ('counterfactual',):
            break
    if moments:
        # "Moments worth a look: pressure and cover on the ball": the caption of every cited moment is mode-free
        # where the detector has failure modes (short_moments), else the detector's failing short label
        name = 'short_moments' if _bank(det, 'short_moments', lvl) else _mode_slot(finding, claims, 'short_fail', lvl)
        # its own used-key: the other side's strength took index 0 of ('short', det) on Metrica game 1 (window
        # 2850 s, team B) and the label lost its side
        short = pick(name, slots_h, used_key=('short_moments', det))
        headline = MOMENTS_PREFIX+(short[:1].lower()+short[1:] if short else 'clear moments to watch')
    elif verdict == 'review':
        prefix = _bank('*', 'review_prefix', lvl)[0]
        short = pick(_mode_slot(finding, claims, 'short_fail', lvl), slots_h, used_key=('short', det))
        headline = prefix+(short[:1].lower()+short[1:] if short else 'a moment to check')
    elif verdict == 'strength':
        # a wide defender's strength names its side (polish round): its own used-key per side, so a report praising
        # both sides keeps both sides' labels
        sided = bool(_bank(det, 'headline_ok_side', lvl)) and slots_h.get('side') in ('left', 'right')
        # fix round 3: a level 1-2 covered advance with nobody behind the ball is labelled by its cover, never
        # "Someone stayed back" (the gate refused that headline and BC60 B got the one-sentence minimal text)
        covered = {'open_ok:advance_covered': 'advance_covered', 'open_ok:covered': 'covered'}.get(
            _open_ok_slot(finding, claims, lvl)) if lvl <= 2 else None
        if covered and not _bank(det, f'headline_ok:{covered}', lvl):
            covered = None
        headline = (pick(f'headline_ok:{covered}', slots_h) if covered else None) or \
            (pick('headline_ok_side', slots_h) if sided else None) or pick('headline_ok', slots_h)
        plain = pick('short_ok', slots_h, used_key=('short', det))   # (taken as before: the other findings of the
        # check keep the short phrasing they had, 32 of 8829 cached-tier reports otherwise changed a failure label)
        short = (pick(f'short_ok:{covered}', slots_h, used_key=('short_covered', det)) if covered else None) or \
            (pick('short_ok_side', slots_h, used_key=('short_side', det, slots_h['side'])) if sided else None) \
            or plain
    elif verdict == 'describe':
        headline = pick(_mode_slot(finding, claims, 'headline_describe', lvl), slots_h)
        short = pick(_mode_slot(finding, claims, 'short_describe', lvl), slots_h, used_key=('short', det))
    else:
        headline = pick(_mode_slot(finding, claims, 'headline_fail', lvl), slots_h)
        short = pick(_mode_slot(finding, claims, 'short_fail', lvl), slots_h, used_key=('short', det))
    if not headline or not short:
        return None
    action = ' '.join(actions) if actions else None
    if action is None and verdict in CRITICISM:
        action = _principle_text(claims, lvl)
    # one definition per concept per report, after the first explanation sentence that uses it; a concept used only
    # in the headline, short or action is defined at the end of the explanation (COACH L3: 77 level-6 reports said
    # "rest defence" only in a headline such as "Rest defence behind an advanced full-back" and never defined it)
    for code, levels, rx, definition in DEFINITIONS:
        if lvl in levels and code not in rep.defined:
            at = next((k for k, sent in enumerate(sentences) if rx.search(sent)), None)
            if at is None and any(rx.search(x or '') for x in (headline, short, action)):
                at = len(sentences)-1
            if at is not None:
                # the margin both counts are measured with (fix round 3, FACTS F-1)
                sentences.insert(at+1, definition_text(definition, slots_e.get('bm')))
                rep.defined.add(code)
                rep.current.append(('def', code))
    out = {'headline': headline, 'explanation': ' '.join(sentences), 'short': short, 'action': action}
    if reg.get('small'):
        out = {k: _small_words(v) for k, v in out.items()}
    # the schema's length: the sentences the plan marks optional are dropped, last first, before the gate sees it
    while len(out['explanation']) > LIMITS['explanation'] and len(sentences) > 2:
        drop = next((k for k in range(len(sentences)-1, 0, -1) if sentences[k] in optional), None)
        if drop is None:
            break
        sentences.pop(drop)
        out['explanation'] = _small_words(' '.join(sentences)) if reg.get('small') else ' '.join(sentences)
    return out


def _small_words(text):
    """A text with the 11v11 role names replaced by the small-format words (SMALL_WORDS)."""
    if not text:
        return text
    for rx, word in SMALL_WORDS:
        text = rx.sub(word, text)
    return text


# ---------------------------------------------------------------------------------------------------------------
# The gate (DESIGN 6.5)

def _facts(claims):
    f = L.facts_for(extra={'claims': [c['facts'] for c in claims]})
    if any(c['facts'].get('role') in DM_ROLES for c in claims):
        f.literals |= {'the six', 'The six'}        # a role name, not the number word (level 7)
    return f


UNIT_KEYS = {'m': lambda k: k.endswith('M'), '%': lambda k: k.lower().endswith('share') or k == 'coverage',
             'deg': lambda k: k == 'angleDeg', 'rate': lambda k: k == 'ratePer10'}
PERCENT_RE = re.compile(r'(?<![\w.])(\d+(?:\.\d+)?)\s?%')
RATE_VALUE_RE = re.compile(r'(?<![\w.])(\d+(?:\.\d+)?)(?: times)? per \d+ minutes', re.I)
DEGREE_VALUE_RE = re.compile(r'(?<![\w.])(\d+(?:\.\d+)?)\s?degrees\b')


def _unit_problems(text, claims):
    """language.validate_text accepts a number when any fact has it; a value with a unit must come from a fact of
    that unit (a count of 2 must not validate "2 m"): metres from *M facts, percentages from shares, degrees from
    angleDeg, rates from ratePer10."""
    pools = {u: [] for u in UNIT_KEYS}
    for c in claims:
        for k, v in c['facts'].items():
            if not _num(v):
                continue
            for u, test in UNIT_KEYS.items():
                if test(k):
                    pools[u].append(abs(float(v))*(100.0 if u == '%' else 1.0))
    out = []
    bare = L.TIME_TOKEN.sub(' ', text)
    for u, rx in (('m', METRE_RE), ('%', PERCENT_RE), ('deg', DEGREE_VALUE_RE), ('rate', RATE_VALUE_RE)):
        for m in rx.finditer(bare):
            v = m.group(1)
            d = len(v.split('.')[1]) if '.' in v else 0
            if not any(abs(p-float(v)) <= .5*10**-d+1e-9 for p in pools[u]):
                out.append(f'"{m.group(0)}" is not a {u} value of the claims')
    return out


def _polarity_index(claims):
    observed, restoring = set(), set()
    for c in claims:
        f = c['facts']
        if c['type'] in ('relation', 'team_state') and f.get('relation') and isinstance(f.get('present'), bool):
            observed.add((f['relation'], f['present']))
        if c['type'] in ('counterfactual', 'action', 'principle') and f.get('relation'):
            restoring.add(f['relation'])
            if f['relation'] == 'channel_cover':
                restoring.add('advance_support')
            if f['relation'] == 'cover':
                restoring.add('pressure')
    return observed, restoring


def _polarity_problems(sentence, kinds, observed, restoring):
    """Relation polarity of one sentence against the claims. A sentence about the match (past tense, a player id or
    a phase phrase) must state each relation as its claim says; a hypothetical or question may also state the
    relation a counterfactual, action or principle claim restores; a general present-tense statement (a principle)
    is not an observation and is not read for polarity."""
    out = []
    hypo = bool(POLARITY_HYPO.search(sentence)) or sentence.rstrip().endswith('?')
    if not hypo and not (PAST.search(sentence) or L.ID_TOKEN.search(sentence) or PHASE_RE.search(sentence)):
        return out
    s = sentence
    found = set()
    for kind in kinds:
        for pol in (False, True):
            rx = POLARITY_RE[kind][pol]

            def hit(m, kind=kind, pol=pol):
                found.add((kind, pol))
                return ' '*len(m.group(0))
            s = rx.sub(hit, s)
    for kind, pol in sorted(found):
        if (kind, pol) in observed:
            continue
        if hypo and kind in restoring:
            continue
        out.append(f'polarity: "{kind}" stated as {"present" if pol else "absent"}, which no claim says')
    if not hypo and PAST.search(sentence):
        for kind in kinds:
            if POLARITY_GUARD_RE[kind].search(s) and not any(k == kind for k, _ in found):
                out.append(f'polarity: "{kind}" worded in a way the gate cannot read ("{sentence[:60]}")')
    return out


def _roles_known(claims):
    known = set()
    for c in claims:
        f = c['facts']
        if f.get('role') and (f.get('roleSource') in CONFIDENT_SOURCES or f.get('roleSource') == 'relative'):
            known.add(f['role'])
    return known


class _Gate:
    """What the checks of one text need, computed once from the claims, the finding and the register."""

    def __init__(self, claims, finding, reg, short):
        self.claims, self.finding, self.reg, self.short = claims, finding, reg, short
        self.level = reg['level']
        self.facts = _facts(claims)
        self.sit = _first(claims, 'situation') or {}
        self.identity_ok = finding['guardrails'].get('identity') != 'fail'
        self.kinds = DETECTOR_KINDS.get(finding['observation'], ())
        if variant_of(finding) == 'advance':
            # level 1's reframed advance also states the space the defender left (fix round 3)
            self.kinds = tuple(self.kinds)+('advance_support', 'channel_cover')
        self.observed, self.restoring = _polarity_index(claims)
        self.known_roles = _roles_known(claims)
        self.metres_est = any(c['facts'].get('metres') == 'estimated' for c in claims)
        # thresholds the text states as "at least N m" (fix round 3: the behind-the-ball margin is the rule's own
        # distance, not an estimate, so it carries no "about")
        self.thresholds = [float(c['facts']['behindMarginM']) for c in claims
                           if _num(c['facts'].get('behindMarginM'))]


def _check_facts(field, v, g):
    """Every number (at its shown precision), time, player id and formation is in the claims' facts."""
    return L.validate_text(v, g.facts)


def _check_units(field, v, g):
    """A value with a unit comes from a fact of that unit (metres, shares, degrees, rates)."""
    return _unit_problems(v, g.claims)


def _check_naming(field, v, g):
    """Player ids only where the level names players (DESIGN 4.9), never in headlines, never with a failed
    identity guardrail."""
    ids = L.ID_TOKEN.findall(v)
    if not ids:
        return []
    if not g.identity_ok:
        return [f'names {ids[0]} although the identity guardrail failed']
    if g.reg['naming'] in ('never', 'relative', 'role_words') or field in ('headline', 'short') or \
            (field == 'action' and g.reg['naming'] != 'yes'):
        return [f'names {ids[0]} (naming "{g.reg["naming"]}" at level {g.level})']
    return []


def _check_roles(field, v, g):
    """A role word singling out a player ("the left back") needs a confirmed role in the claims (DESIGN 3.4), and
    none at level 1, with a failed identity guardrail, or in a headline up to level 4."""
    out, s = [], v
    for rx, codes, phrase in ROLE_PHRASE_RE:
        for m in rx.finditer(s):
            if g.reg['naming'] == 'never' or not g.identity_ok or (field in ('headline', 'short') and g.level <= 4):
                out.append(f'role word "{m.group(0)}" singles out a player at this level')
            elif not any(codes.fullmatch(r) for r in g.known_roles):
                out.append(f'role word "{m.group(0)}" is not a confirmed role in the claims')
        s = rx.sub(lambda m: ' '*len(m.group(0)), s)
    return out


def _check_phase(field, v, g):
    """The phase is the claimed one and carries its qualifier: (estimated), (coach-marked) or (estimated, partly
    coach-marked) (possession binding rule 8; user decision Oct 21)."""
    out = []
    for m in list(PHASE_RE.finditer(v))+list(POSS_RE.finditer(v)):
        state = 'own' if m.group(1).lower() in ('we', 'our') else 'opp'
        qual = (m.group(2) or '').strip()
        if g.sit.get('phase') != state:
            out.append(f'phase "{m.group(0)}" is not the claimed phase')
        elif qual != QUALIFIER.get(g.sit.get('source'), '(?)'):
            out.append(f'phase "{m.group(0)}" needs the qualifier {QUALIFIER.get(g.sit.get("source"))}')
    loose = PHASE_LOOSE.search(v)
    if loose:
        out.append(f'phase stated without its qualifier ("{loose.group(0)}")')
    if g.sit.get('phase') not in ('own', 'opp') and field in ('headline', 'short'):
        m = PHASE_CLAIM_RE.search(v)
        if m:
            out.append(f'"{m.group(0)}" claims who had the ball, which is unknown here')
    return out


def _check_numbers(field, v, g):
    """The level's number mode (DESIGN 6.3): none at level 1, counts at 2, rounded metres at 3, metres at 4,
    shares from 5, rates from 6; estimated metres say "about"."""
    out, reg = [], g.reg
    bare = L.TIME_TOKEN.sub(' ', v)
    if reg['numbers'] == 'none':
        if re.search(r'\d', bare) or any(w in L.NUMBER_WORDS for w in re.findall(r'[a-z]+', bare.lower())):
            out.append('numbers at a level that shows none')
    if not _numbers_at_least(reg, 'rounded_metres') and METRE_RE.search(bare):
        out.append('metres at a level that shows counts only')
    if not _numbers_at_least(reg, 'metres') and DEGREE_RE.search(bare):
        out.append('angles before level 4')
    if not _numbers_at_least(reg, 'shares') and '%' in bare:
        out.append('percentages before the shares register')
    if not _numbers_at_least(reg, 'rates') and RATE_RE.search(bare):
        out.append('rates before the rates register')
    if g.metres_est or reg['numbers'] == 'rounded_metres':
        thresholds = getattr(g, 'thresholds', ())
        for m in METRE_RE.finditer(bare):
            before = bare[max(0, m.start()-9):m.start()]
            if re.search(r'at least\s*$', before, re.I) and any(abs(t-float(m.group(1))) <= .5+1e-9 for t in
                                                                 thresholds):
                continue            # a rule's threshold ("at least 10 m behind the ball"), not an estimate
            if not re.search(r'(?:about|around|roughly)\s*$', before) and \
                    not bare[m.end():m.end()+7].startswith(' (est.)'):
                out.append(f'estimated metres "{m.group(0)}" without "about"')
    return out


def _check_vocabulary(field, v, g):
    """Concepts the level has not unlocked, the global and profile bans, the spec's generic sentences, and the
    repeated-pattern words in a range under 3 minutes."""
    out = []
    for name, rx, unlock in CONCEPT_RE:
        if g.level < unlock and rx.search(v):
            out.append(f'concept "{name}" is not used before level {unlock}')
    words = ' '+re.sub(r'[^a-z\-\' ]+', ' ', v.lower())+' '
    for w in GLOBAL_BANS+g.reg['banned']:
        if re.search(r'(?<![a-z\-])'+re.escape(w)+r'(?![a-z\-])', words):
            out.append(f'uses "{w}"')
    for rx in GENERIC_BANS:
        m = rx.search(v)
        if m:
            out.append(f'generic sentence ("{m.group(0)}")')
    if g.short:
        for w in L.SHORT_BANNED:
            if re.search(r'(?<![a-z\-])'+re.escape(w)+r'(?![a-z\-])', words):
                out.append(f'"{w}" in a range under {int(SHORT_RANGE_S)} s')
    return out


def _check_events(field, v, g):
    """No unmeasured event (invariant 4): turnovers, communication, possession changes, counter-attacks or passes
    outside a hypothetical sentence."""
    return [f'unmeasured event {p}' for p in S.event_problems(v)]


def _check_hypotheticals(field, v, g):
    """Consequence words only inside a hypothetical sentence and from level 3; transition words from level 5;
    "if the ball had been lost" only as a hypothetical (its level is the concept gate's)."""
    out = []
    for sentence in SENTENCE_SPLIT.split(v):
        hypo = HYPOTHETICAL.search(sentence) is not None
        m = CONSEQUENCE.search(sentence)
        if m and (not hypo or g.level < 3):
            out.append(f'consequence "{m.group(0)}" outside a hypothetical'+(' (level 3+)' if g.level < 3 else ''))
        m = TRANSITION.search(sentence)
        if m and (not hypo or g.level < 5):
            out.append(f'transition wording "{m.group(0)}" outside a level 5+ hypothetical')
        m = LOSS_RE.search(sentence)
        if m and not hypo:
            out.append(f'"{m.group(0)}" outside a hypothetical')
    return out


def _check_polarity(field, v, g):
    """Every stated relation has the polarity its claim has (critic N12)."""
    out = []
    for sentence in SENTENCE_SPLIT.split(v):
        out += _polarity_problems(sentence, g.kinds, g.observed, g.restoring)
    return out


_PID = r'([AB]-\d{1,3})'
_ROLE_PAREN = re.compile(r'\b[Tt]he [a-z][a-z -]{0,40}? \(([AB]-\d{1,3})\)')
PRESSER_MENTION = tuple(re.compile(p, re.I) for p in (
    _PID+r' (?:pressed|closed|put pressure|went to the ball|engaged)\b', r'\bpressed by '+_PID,
    r'\bpressure (?:came )?from '+_PID, r'\bfirst defender,? '+_PID, r'\bnearest defender,? '+_PID,
    r'\bnearest defender \('+_PID+r'\)', _PID+r' was the nearest defender'))
COVER_MENTION = tuple(re.compile(p, re.I) for p in (
    r'\bwith '+_PID+r' (?:covering|in a covering position)', _PID+r' (?:gave covering support|held a covering '
    r'position|covered\b)', r'\bcovered by '+_PID, r'\bsecond defender,? '+_PID, r'\bnearest teammate,? '+_PID,
    _PID+r', the nearest teammate'))


def _check_actors(field, v, g):
    """Pressure and cover (critic B-3): the presser named is the first defender AT the moment linked
    (measurements.fdId), and the presser is never named as the cover; both in the claims and in the text (an id in a
    presser phrase must be the presser claim's player, an id in a cover phrase must not be the presser and must be the
    cover claim's player when one is named)."""
    if g.finding['observation'] != 'pressure_cover':
        return []
    out = []
    fd = (g.finding.get('measurements') or {}).get('fdId')
    actor = (_first(g.claims, 'actor_state') or {}).get('player')
    cover = (_rel(g.claims, 'cover') or {}).get('player')
    if field == 'headline':           # the claim set once per text, on its first field
        if actor and isinstance(fd, str) and actor != fd:
            out.append(f'presser claim {actor} is not the first defender at the moment ({fd})')
        if actor and cover and actor == cover:
            out.append(f'{actor} is claimed as presser and cover')
    s = _ROLE_PAREN.sub(r'\1', v)
    for rx in PRESSER_MENTION:
        for m in rx.finditer(s):
            if m.group(1) != actor:
                out.append(f'{m.group(1)} named as the presser; the moment\'s presser is {actor or fd}')
    for rx in COVER_MENTION:
        for m in rx.finditer(s):
            if m.group(1) == actor:
                out.append(f'{m.group(1)} named as the presser and as the cover')
            elif cover and m.group(1) != cover:
                out.append(f'{m.group(1)} named as the cover; the moment\'s cover is {cover}')
    return out


def _check_format_words(field, v, g):
    """Below 11v11 no 11v11 role name (DESIGN 2.5; critic S-9: full-back, centre-back, holding midfielder, wing-back,
    left / right back, the six)."""
    if not g.reg.get('small'):
        return []
    return [f'11v11 role word "{m.group(0)}" in a {g.reg.get("format")} report' for m in ROLE11_RE.finditer(v)]


PROFESSIONAL_RE = re.compile(r'\bprofessional', re.I)


def _check_professional(field, v, g):
    """A youth team, or any team below level 5, is never measured against professionals in words (fix round 3, LEAD
    decision (3), COACH F-5: "at or above the professional base rate" at U16; the base below level 5 is scaled)."""
    if g.reg.get('adult') and g.level >= 5:
        return []
    m = PROFESSIONAL_RE.search(v)
    return [f'"{m.group(0)}" at level {g.level} for a {"youth" if not g.reg.get("adult") else "adult"} team'] if m \
        else []


def _check_questions(text, g):
    """At most the level's challenges as questions (level 1: one)."""
    n = sum((text.get(k) or '').count('?') for k in TEXT_FIELDS)
    lim = max(1, int(g.reg['maxChallenges']))
    return [f'{n} questions (at most {lim} at level {g.level})'] if n > lim else []


def _check_describe(text, g):
    """A description never judges."""
    if g.finding['verdict'] != 'describe':
        return []
    out = []
    for field in ('headline', 'explanation'):
        m = re.search(r'\b(?:well done|good|needs?|improve\w*|try this|next step|could have)\b', text.get(field) or '',
                      re.I)
        if m:
            out.append(f'{field}: a description judges ("{m.group(0)}")')
    return out


# The gate's checks, in order; tests disable each in turn and show it is the one that catches its case.
FIELD_CHECKS = {'facts': _check_facts, 'units': _check_units, 'naming': _check_naming, 'roles': _check_roles,
                'phase': _check_phase, 'numbers': _check_numbers, 'vocabulary': _check_vocabulary,
                'events': _check_events, 'hypotheticals': _check_hypotheticals, 'polarity': _check_polarity,
                'actors': _check_actors, 'format_words': _check_format_words, 'professional': _check_professional}
TEXT_CHECKS = {'questions': _check_questions, 'describe': _check_describe}


def _check_shape(text):
    """Required fields, printable ASCII (the PDF's WinAnsi subset) and the schema's lengths."""
    out = []
    if not isinstance(text, dict):
        return ['text: not a dict']
    for k in TEXT_FIELDS:
        v = text.get(k)
        if v is None:
            if k != 'action':
                out.append(f'{k}: missing')
            continue
        if not isinstance(v, str) or not v.strip():
            out.append(f'{k}: empty')
            continue
        if len(v) > LIMITS[k]:
            out.append(f'{k}: {len(v)} characters (at most {LIMITS[k]})')
        if not re.fullmatch(r'[ -~]+', v):
            out.append(f'{k}: not printable ASCII')
    return out


def validate(text, claims, finding, profile, *, short=False, register=None, checks=None):
    """The problems of a finding's text against its claims ([] when it may be shown). Works on any engine's output.
    Field shape first (required fields, printable ASCII, lengths), then per field: facts (numbers at their shown
    precision, times, ids, formations, units), naming, role words, phase and qualifier, the level's number mode,
    vocabulary (concepts, bans, generic sentences, short-range words), unmeasured events, hypotheticals (consequence,
    transition, ball lost) and relation polarity; then the number of questions and, for a description, no judgement.
    checks: the names of FIELD_CHECKS / TEXT_CHECKS to run (all by default; tests use it to show each one matters)."""
    out = _check_shape(text)
    if out:
        return out
    g = _Gate(claims, finding, register or register_for(profile, finding), short)
    for field in TEXT_FIELDS:
        v = text.get(field)
        if v is None:
            continue
        for name, check in FIELD_CHECKS.items():
            if checks is None or name in checks:
                out += [f'{field}: {p}' for p in check(field, v, g)]
    for name, check in TEXT_CHECKS.items():
        if checks is None or name in checks:
            out += check(text, g)
    return out


class _FreeGate:
    """What the field checks read for a text that belongs to no finding (validate_free)."""

    def __init__(self, reg, short, metres_est):
        self.reg, self.level, self.short, self.metres_est = reg, reg['level'], short, metres_est


def validate_free(text, profile, facts=None, *, short=False, numbers=True, metres_estimated=True, limit=None,
                  check_facts=True):
    """The gate for report text that is not a finding's (report.py: the overview synthesis, the next objective, the
    notices, the possession caveat, captions, drill fields): printable ASCII within limit characters; every number,
    time, id and formation in facts (a language.Facts; None = no number may appear); with numbers=True the level's
    number mode (no digits at level 1, metres from level 3, ...); concepts the level has not unlocked, the global and
    profile bans, the spec's generic sentences and, with short=True, the repeated-pattern words; unmeasured events;
    consequence and transition words outside a hypothetical; a phase phrase only with its qualifier. [] = may be
    shown. check_facts=False skips the facts (a quoted drill-library line keeps its own setup numbers, "2 v 1")."""
    out = []
    if not isinstance(text, str) or not text.strip():
        return ['empty']
    if limit is not None and len(text) > limit:
        out.append(f'{len(text)} characters (at most {limit})')
    if not re.fullmatch(r'[ -~]+', text):
        out.append('not printable ASCII')
    if out:
        return out
    g = _FreeGate(_register(profile), short, metres_estimated)
    if check_facts:
        out += L.validate_text(text, facts if facts is not None else L.Facts())
    if numbers:
        out += _check_numbers('text', text, g)
    out += _check_vocabulary('text', text, g)
    out += _check_events('text', text, g)
    out += _check_hypotheticals('text', text, g)
    for m in PHASE_RE.finditer(text):
        if (m.group(2) or '').strip() not in QUALIFIER.values():
            out.append(f'phase "{m.group(0)}" without its qualifier')
    loose = PHASE_LOOSE.search(text)
    if loose:
        out.append(f'phase stated without its qualifier ("{loose.group(0)}")')
    return out


# ---------------------------------------------------------------------------------------------------------------
# The stage

def short_for(finding, short):
    """The short-range rule for one finding: the report range's (short), or assess's 'short_range' limitation, which
    it sets on every finding of a range under its params.shortRangeS (180 s). coaching.interpret calls this stage
    without the range (INTERFACES, TEXT request to LEAD), so a 60 s range inside a 10-minute segment is still short."""
    return bool(short) or 'short_range' in (finding.get('limitations') or ())


def _short(range_, obs):
    if isinstance(range_, dict):
        if _num(range_.get('seconds')):
            return float(range_['seconds']) < SHORT_RANGE_S
        if _num(range_.get('t0')) and _num(range_.get('t1')):
            return float(range_['t1'])-float(range_['t0']) < SHORT_RANGE_S
    seg = (obs or {}).get('segment') or {}
    if _num(seg.get('t0')) and _num(seg.get('t1')):
        return float(seg['t1'])-float(seg['t0']) < SHORT_RANGE_S
    return True        # unknown range: the cautious side (no repeated-pattern words)


def _opportunity_index(*files):
    idx = {}
    for f in files:
        for o in (f or {}).get('opportunities') or ():
            idx[o['id']] = o
    return idx


def request_for(finding, claims, reg, short):
    """What an optional text engine receives (DESIGN 6.6): the claims and the register, nothing else."""
    return {'verdict': finding['verdict'], 'detector': finding['observation'], 'level': reg['level'],
            'register': {k: reg[k] for k in ('tone', 'numbers', 'maxChallenges', 'naming', 'adult', 'banned')},
            'short': bool(short), 'claims': copy.deepcopy(claims),
            'limits': dict(LIMITS), 'qualifier': QUALIFIER}


def _write(finding, claims, profile, reg, rep, short, engine=None, salt=0, repeats=None):
    """(text, source) for one finding: an engine's text when it passes the gate, else the planned text, else the
    minimal plan; (None, reason) when nothing passes. repeats: _realize's, for the planned text only."""
    problems = []
    if engine is not None:
        try:
            text = engine.render(request_for(finding, claims, reg, short))
            text = {k: (text or {}).get(k) for k in TEXT_FIELDS}
            p = validate(text, claims, finding, profile, short=short, register=reg)
            if not p:
                return text, 'engine'
            problems += p
        except Exception:      # noqa: BLE001 - an optional engine may fail; the deterministic text is used
            pass
    snap = rep.snapshot()
    rep.current = []
    text = _realize(finding, claims, reg, rep, salt=salt, repeats=repeats)
    if text is not None:
        p = validate(text, claims, finding, profile, short=short, register=reg)
        if not p:
            return text, 'planned'
        problems += p
    rep.restore(snap)
    rep.current = []
    text = _realize(finding, claims, reg, rep, salt=salt, minimal=True)
    if text is not None:
        p = validate(text, claims, finding, profile, short=short, register=reg)
        if not p:
            return text, 'minimal'
        problems += p
    rep.restore(snap)
    rep.current = []
    return None, problems


def explain(findings, profile, obs=None, roles=None, evidence=None, alternatives=None, *, range=None,
            no_alternatives=None, engine=None, log=None):
    """The 'explain' stage (INTERFACES.md stage table): each finding gets its claims (build_claims) and text
    (headline <= 160, explanation <= 1500, short <= 90, action <= 500 or None) in the profile's register; findings
    whose text cannot pass the gate are returned in 'withheld' with a reason and left out of 'findings'.
    range: the report range ({t0, t1} or {seconds}) for the short-range rule; without it the obs segment is used.
    no_alternatives: ZONES' NoAlternative records; a criticism without a zone then says why no area is marked.
    evidence: accepted for the stage contract; the text does not need it (the moments are cited by the findings).
    engine: an optional text engine (render(request) -> text dict); its text is used only when it passes the gate.
    log: an optional list that receives (finding id, source of the text or the gate's problems)."""
    reg0 = register_for(profile)
    short0 = _short(range, obs)
    margin = behind_margin(obs)         # fix round 3 (FACTS F-1): the behind-the-ball margin of this run's scale
    opps = _opportunity_index(obs, roles)
    alts = {a['finding']: a for a in alternatives or () if isinstance(a, dict) and 'zone' in a}
    nalts = {a['finding']: a for a in no_alternatives or () if isinstance(a, dict) and 'zone' not in a}
    rep = _ReportState()
    done, withheld, picks = [], [], {}
    for f0 in findings:
        f = copy.deepcopy(f0)
        if f['level'] != reg0['level']:
            withheld.append({'finding': f['id'], 'reason': 'level_mismatch'})
            continue
        reg = register_for(profile, f)
        short = short_for(f, short0)
        claims = build_claims(f, profile, phase_source=phase_source_of(f, opps), alternative=alts.get(f['id']),
                              no_alternative=nalts.get(f['id']), register=reg, short=short, behind_margin_m=margin)
        if f['verdict'] in CRITICISM and not [c for c in claims if c['type'] in ('relation', 'team_state') and
                                              c['facts'].get('scope') != 'demonstrated']:
            withheld.append({'finding': f['id'], 'reason': 'no_relation_claim'})
            continue
        text, how = _write(f, claims, profile, reg, rep, short, engine=engine)
        if text is None:
            withheld.append({'finding': f['id'], 'reason': 'text_failed_validation'})
            if log is not None:
                log.append((f['id'], how))
            continue
        f['claims'] = claims
        f['text'] = text
        picks[f['id']] = list(rep.current)
        done.append(f)
        if log is not None:
            log.append((f['id'], how))
    _spread(done, profile, rep, short0, picks)
    return {'findings': done, 'withheld': withheld}


def _spread(done, profile, rep, short0, picks):
    """Re-plan the later of any two explanations whose word-trigram Jaccard exceeds JACCARD_MAX (DESIGN 6.4): its
    phrasings are released and other variants (a salted start) are tried; the deterministic text of an engine-written
    finding is not re-planned."""
    for k in range(len(done)):
        for salt in range(1, RETRIES+1):
            worst = max((jaccard(done[k]['text']['explanation'], done[j]['text']['explanation'])
                         for j in range(k)), default=0.0)
            if worst <= JACCARD_MAX:
                break
            fid = done[k]['id']
            old = picks.get(fid) or ()
            rep.used -= {p for p in old if p[0] != 'def'}
            rep.defined -= {p[1] for p in old if p[0] == 'def'}
            text, how = _write(done[k], done[k]['claims'], profile, register_for(profile, done[k]), rep,
                               short_for(done[k], short0), engine=None, salt=salt)
            if text is not None:
                done[k]['text'] = text
                picks[fid] = list(rep.current)
            else:
                rep.used |= {p for p in old if p[0] != 'def'}
                rep.defined |= {p[1] for p in old if p[0] == 'def'}
        if _worst_alike(done, k) >= JACCARD_MAX:
            _drop_repeats(done, k, profile, rep, short0, picks)


def _worst_alike(done, k):
    """The largest word-trigram Jaccard of explanation k against the earlier explanations of the report."""
    return max((jaccard(done[k]['text']['explanation'], done[j]['text']['explanation']) for j in range(k)),
               default=0.0)


def _drop_repeats(done, k, profile, rep, short0, picks):
    """Pre-freeze (freeze gate G-2): explanation k is still at least JACCARD_MAX alike an earlier one after the salted
    re-plans (coaching_eval's criterion 13 refuses a pair at .5 or more): plan it again (salt 0) without the optional
    sentences an earlier explanation of the report already says word for word, the slots of REPEAT_SLOTS added one at
    a time, and keep the first planned text (never the minimal one) that comes under JACCARD_MAX; otherwise the text
    stays as the re-plans left it. Only sentences OPTIONAL_SLOTS lets the length guard drop are ever left out."""
    fid = done[k]['id']
    earlier = tuple(done[j]['text']['explanation'] for j in range(k))
    for n in range(1, len(REPEAT_SLOTS)+1):
        snap = rep.snapshot()
        old = picks.get(fid) or ()
        rep.used -= {p for p in old if p[0] != 'def'}
        rep.defined -= {p[1] for p in old if p[0] == 'def'}
        text, how = _write(done[k], done[k]['claims'], profile, register_for(profile, done[k]), rep,
                           short_for(done[k], short0), engine=None, salt=0, repeats=(earlier, REPEAT_SLOTS[:n]))
        if text is not None and how == 'planned' and \
                max(jaccard(text['explanation'], e) for e in earlier) < JACCARD_MAX:
            done[k]['text'] = text
            picks[fid] = list(rep.current)
            return True
        rep.restore(snap)
    return False


def with_no_alternative(finding, no_alternative, profile, *, short=False):
    """The finding with ZONES' reason for drawing no area added to its action (DESIGN 5.3: "if no mover passes ... the
    annotation shows the observed problem and the principle"), validated like the rest of its text; the finding
    itself (unchanged) when there is nothing to add or the result would not pass the gate. coaching.interpret runs
    this stage without the NoAlternative records, so the report stage calls it (TEXT request to LEAD, INTERFACES.md).
    The plan's place: before the question / statement of a work item, after the review action of a review; a review
    whose guardrail failed already says what could not be confirmed (no sentence for 'guardrail_failed')."""
    if not no_alternative or finding.get('text') is None or finding['verdict'] not in FAILURES:
        return finding
    if is_moments(finding):
        return finding          # what to look for, never why no better position is drawn (fix round 2)
    reason = no_alternative.get('reason')
    if reason not in NO_ALT_WORDS or (finding['verdict'] == 'review' and reason == 'guardrail_failed'):
        return finding
    if _first(finding['claims'], 'limitation', code='no_alternative') or _first(finding['claims'], 'counterfactual'):
        return finding
    reg = register_for(profile, finding)
    if reg['level'] < NO_ALT_MIN_LEVEL:
        return finding          # levels 1-2 never explain the zone engine (COACH L5)
    facts = dict(no_alt_facts(finding, no_alternative), code='no_alternative', guardrail=False)
    claims = list(finding['claims'])+[{'type': 'limitation', 'facts': facts}]
    sentence = NO_ALT_WORDS[facts['wording']][0 if reg['level'] <= 3 else 1]
    if reg.get('small'):
        sentence = _small_words(sentence)
    action = finding['text'].get('action')
    if finding['verdict'] == 'review':
        new = f'{action} {sentence}' if action else sentence
    else:
        new = f'{sentence} {action}' if action else sentence
    text = dict(finding['text'], action=new)
    if validate(text, claims, finding, profile, short=short_for(finding, short), register=reg):
        return finding
    out = copy.deepcopy(finding)
    out['claims'], out['text'] = claims, text
    return out


def invariant_problems(finding):
    """The fix-round-1 text invariants of one worded finding, without its profile (for EVAL's report_problems and the
    game-3 G1+ check, critic section 6): the presser named is the moment's first defender (measurements.fdId) and is
    never named as the cover (B-3); a finding whose phase is unknown says nothing of who had the ball in its headline
    or short label (B-4); below 11v11 no 11v11 role name appears (S-9, the format read from context.format)."""
    out = []
    text = finding.get('text') or {}
    fields = [(k, text.get(k)) for k in TEXT_FIELDS if isinstance(text.get(k), str)]
    if finding['observation'] == 'pressure_cover':
        fd = (finding.get('measurements') or {}).get('fdId')
        claims = finding.get('claims') or []
        actor = (_first(claims, 'actor_state') or {}).get('player')
        cover = (_rel(claims, 'cover') or {}).get('player')
        if actor and isinstance(fd, str) and actor != fd:
            out.append(f'presser {actor} is not the first defender at the moment ({fd})')
        if actor and cover and actor == cover:
            out.append(f'{actor} is named as presser and cover')
        for k, v in fields:
            s = _ROLE_PAREN.sub(r'\1', v)
            for rx in PRESSER_MENTION:
                for m in rx.finditer(s):
                    if isinstance(fd, str) and m.group(1) != fd:
                        out.append(f'{k}: {m.group(1)} named as the presser; the first defender at the moment is {fd}')
            for rx in COVER_MENTION:
                for m in rx.finditer(s):
                    if m.group(1) in (fd, actor):
                        out.append(f'{k}: {m.group(1)} named as the presser and as the cover')
    if not phase_known(finding):
        for k, v in fields:
            if k in ('headline', 'short') and PHASE_CLAIM_RE.search(v):
                out.append(f'{k}: "{PHASE_CLAIM_RE.search(v).group(0)}" with the phase unknown')
    if (finding.get('context') or {}).get('format') not in (None, '11v11'):
        for k, v in fields:
            m = ROLE11_RE.search(v)
            if m:
                out.append(f'{k}: 11v11 role word "{m.group(0)}" in a {finding["context"]["format"]} report')
    out += moments_problems(finding)
    out += fix3_problems(finding)
    return out


INTERCHANGE_RE = re.compile(r'\btook their place\b|\btook the vacated spot\b|\bwas taken by a teammate\b', re.I)
NEAREST_MATE_RE = re.compile(r'\bnearest teammate\b(?! within the cover angle)', re.I)
WHOLE_PITCH_RE = re.compile(r'\b(?:used|using|uses) the whole pitch\b', re.I)
D1_PRESSURE_NOUN_RE = re.compile(r'\bmoments with pressure on the ball\b|\bpressing moments\b', re.I)
MARGIN_BEFORE_RE = re.compile(r'(?:\bwell|\bat least (?:about )?\d+ m)\s*$', re.I)
OBSERVED_RE = re.compile(r'\b(?:was|were|had|did|stayed|went|moved|kept|got|came|stood|held|dropped|shifted|spread|'
                         r'faced)\b', re.I)


def fix3_problems(finding):
    """The fix-round-3 text invariants of one worded finding (critic B-3, FACTS F-1, F-2, F-5 to F-8, F-11, COACH
    F-3 / F-11; in invariant_problems, so coaching_eval.report_problems and game-3 G1+ enforce them), each refusing
    the old false form:
    - F-1: a count of ours behind the ball (or of theirs on our side of it) states the margin it was measured with
      ("at least 10 m" / "well"), and a level-7 shape description says no "none of ours ... behind the ball" when
      nobody of ours was the margin behind it (the ball with our own back line);
    - F-2: a wide-defender or spacing finding names the first moment's own actor (measurements.actorId) and, for a
      gap, the moment's own pair (gapLeftId / gapRightId);
    - F-5: the finding is placed in its first moment's third (measurements.momentZone);
    - F-6 / COACH F-3: a strength's count is of the judged moments ("N of the D", "all D"), never of all its
      opportunities, says how many more could not be judged, and pressure-and-cover never counts "moments with
      pressure on the ball" / "pressing moments";
    - F-7: the pressure-and-cover cover candidate is "the nearest teammate within the cover angle", never "the
      nearest teammate";
    - F-8: "a teammate took their place" only with the geometric interchange at the moment (interchangeCover);
    - F-11: no "used the whole pitch" for a shape; the opponents between the lines are said on the ball's side;
    - COACH F-11: level 1 never says "a few" of five or more moments."""
    out = []
    det = finding['observation']
    text = finding.get('text') or {}
    m = finding.get('measurements') or {}
    a = finding.get('aggregate') or {}
    claims = finding.get('claims') or []
    exp = text.get('explanation') if isinstance(text.get('explanation'), str) else ''
    allt = ' '.join(v for k, v in text.items() if k in TEXT_FIELDS and isinstance(v, str))
    # F-1: the margin on every count at the ball's depth
    if det in ('wide_defender_advance', 'behind_ball', 'attacking_shape'):
        for k in ('explanation', 'action'):
            for s in SENTENCE_SPLIT.split(text.get(k) if isinstance(text.get(k), str) else ''):
                if not BEHIND_COUNT_RE.search(s) or not (OBSERVED_RE.search(s) or re.search(r'\d', s)):
                    continue
                for mm in re.finditer(r'behind the ball|on our side', s, re.I):
                    if not MARGIN_BEFORE_RE.search(s[:mm.start()]):
                        out.append(f'{k}: a count "{mm.group(0)}" without the margin it was measured with ("{s[:80]}")')
        if det == 'attacking_shape' and m.get('deepN') == 0 and re.search(r'\bnone of ours\b[^.]*\bbehind the ball',
                                                                         exp):
            out.append('explanation: "none of ours ... behind the ball" with the ball at our own back line')
    # F-2: the moment's own players
    if det in ('wide_defender_advance', 'defensive_spacing') and _valid_id(m.get('actorId')):
        act = (_first(claims, 'actor_state') or {}).get('player')
        if act and act != m['actorId']:
            out.append(f'actor {act} is not the first moment\'s ({m["actorId"]})')
        pair = (m.get('gapLeftId'), m.get('gapRightId'))
        if det == 'defensive_spacing' and all(_valid_id(p) for p in pair):
            mate = (_rel(claims, 'lateral_gap') or {}).get('player')
            if mate and (mate not in pair or mate == act):
                out.append(f'gap partner {mate} is not of the first moment\'s pair {pair[0]} / {pair[1]}')
            s = _ROLE_PAREN.sub(r'\1', exp)
            for mm in re.finditer(r'\bbetween ([AB]-\d{1,3}) and ([AB]-\d{1,3})\b', s):
                if {mm.group(1), mm.group(2)} != set(pair):
                    out.append(f'explanation: the gap "{mm.group(0)}" is not the first moment\'s pair')
    # F-5: the moment's own third
    sit = _first(claims, 'situation') or {}
    if m.get('momentZone') in PLACE and sit.get('zone') not in (None, m['momentZone']):
        out.append(f'placed in {sit["zone"]}, the first moment was in {m["momentZone"]}')
    # F-6 / COACH F-3: a strength over the judged moments
    if finding.get('verdict') == 'strength' and _num(a.get('opportunities')):
        dem, nd, unc, n = (int(a.get(k) or 0) for k in ('demonstrated', 'notDemonstrated', 'unclear', 'opportunities'))
        dec = dem+nd
        if n != dec and re.search(rf'(?<![\d.]){dem} (?:times )?(?:of|out of) (?:the )?{n}(?![\d.])', exp):
            out.append(f'explanation: a strength counted over all {n} opportunities, not the {dec} judged')
        if unc > 0 and re.search(rf'(?<![\d.]){dem} of the {dec}(?![\d.])|\b[Aa]ll {dec}(?![\d.])', exp) and \
                not re.search(r'could not be judged|unclear', exp):
            out.append(f'explanation: {unc} unclear moments of the strength are not said')
        if det == 'pressure_cover' and D1_PRESSURE_NOUN_RE.search(exp):
            out.append(f'explanation: "{D1_PRESSURE_NOUN_RE.search(exp).group(0)}" as the counted moments')
    # F-7: the cover candidate, in the explanation's sentences about the match (a principle's "the nearest teammate
    # covers at an angle" is general advice; the action's "In each clip, watch ... the nearest teammate" asks the
    # coach to look at whoever that was)
    if det == 'pressure_cover':
        said = [s for s in SENTENCE_SPLIT.split(exp) if OBSERVED_RE.search(s) and NEAREST_MATE_RE.search(s)]
        if said:
            out.append(f'"nearest teammate" without "within the cover angle" ("{said[0][:70]}")')
        mate = (_rel(claims, 'cover') or {}).get('player')
        if mate and re.search(rf'\b{re.escape(mate)}\)? was (?:the )?nearest\b(?! within the cover angle)',
                              _ROLE_PAREN.sub(r'\1', allt)):
            out.append(f'{mate} called the nearest without "within the cover angle"')
    # F-8: the interchange said only with the geometric one
    if INTERCHANGE_RE.search(allt) and m.get('interchangeCover') is not True:
        out.append(f'"{INTERCHANGE_RE.search(allt).group(0)}" without the geometric interchange at the moment')
    # F-11
    if det == 'attacking_shape' and finding.get('verdict') in ('strength', 'describe') and WHOLE_PITCH_RE.search(allt):
        out.append(f'"{WHOLE_PITCH_RE.search(allt).group(0)}" for a shape')
    if det == 'defensive_spacing' and re.search(r'standing in that space\b(?! on the ball\'s side)', allt):
        out.append('opponents "standing in that space" counted on the ball\'s side only')
    # COACH F-11: level 1's number-free counts
    if int(finding.get('level') or 0) == 1 and re.search(r'\ba few (?:times|of these moments)\b', exp):
        n_ = a.get('demonstrated') if finding.get('verdict') == 'strength' else a.get('notDemonstrated')
        if _num(n_) and n_ >= 5:
            out.append(f'explanation: "a few" for {int(n_)} moments')
    return out


def moments_problems(finding):
    """The fix-round-2 invariants of a worded 'moments worth a look' review (LEAD decision (b); [] for any other
    finding): it is a review with no drill and no priority; the headline says "Moments worth a look"; the
    explanation states the clear moments of the JUDGED opportunities ("3 of the 11", never the "3 of 12" of all
    opportunities the round-1 text printed) and every cited moment's time; no text field uses verdict wording
    (VERDICT_WORDS_RE: "try this", "work on", "should", "mistake", "possible issue", ...)."""
    if MOMENTS not in (finding.get('limitations') or ()):
        return []
    out = []
    if finding.get('verdict') != 'review':
        out.append(f'moments worth a look with verdict {finding.get("verdict")}')
    if finding.get('drills'):
        out.append('moments worth a look with a drill')
    if finding.get('priority') is not None:
        out.append('moments worth a look with a priority')
    text = finding.get('text') or {}
    if not is_moments(finding) or not isinstance(text.get('explanation'), str):
        return out+(['moments worth a look without its counts'] if not is_moments(finding) else [])
    m = finding['measurements']
    if not str(text.get('headline') or '').startswith(MOMENTS_PREFIX):
        out.append('headline: does not say "Moments worth a look"')
    exp = text['explanation']
    clear, dec = int(m['clearMoments']), int(m['decidedMoments'])
    said = rf'(?<!\d){clear} of (?:the )?{dec}(?!\d)' if clear < dec else rf'\b[Aa]ll {dec}(?!\d)'
    if not re.search(said, exp):
        out.append(f'explanation: "{clear} of {dec}" (clear of judged) is not stated')
    # fix round 3 (FACTS F-4 = COACH F-10): the counts add up, the judged ones that fell short less clearly are said
    more = int(m['failingMoments'])-clear
    if more > 0:
        n = rf'(?:{more}|one)' if more == 1 else str(more)
        if not re.search(rf'(?<![\d.])\b{n}\b[^.]*\b(?:fell short|shortfalls)\b|\bshortfalls[^.]*:\s*{n}\b', exp, re.I):
            out.append(f'explanation: the {more} judged moments that fell short less clearly are not said')
    opp = (finding.get('aggregate') or {}).get('opportunities')
    if _num(opp) and int(opp) != dec and re.search(rf'(?<!\d)\d+ of (?:the )?{int(opp)}(?!\d)', exp):
        out.append(f'explanation: a count "of {int(opp)}" (all opportunities, not the judged ones)')
    for ts in finding.get('timestamps') or ():
        if L.fmt_time(ts['t']) not in exp:
            out.append(f'explanation: the moment {L.fmt_time(ts["t"])} is not given')
    for k in TEXT_FIELDS:
        v = text.get(k)
        if isinstance(v, str) and VERDICT_WORDS_RE.search(v):
            out.append(f'{k}: verdict wording "{VERDICT_WORDS_RE.search(v).group(0)}"')
    return out


def report_problems(findings, profile, *, short=False):
    """Report-level checks a test or the report stage can run: every finding's text passes the gate against its
    claims (each in its own register and short-range rule: short_for), and no two explanations are more alike than
    JACCARD_MAX."""
    out = []
    for f in findings:
        out += [f'{f["id"]}: {p}' for p in validate(f['text'], f['claims'], f, profile, short=short_for(f, short),
                                                     register=register_for(profile, f))]
        out += [f'{f["id"]}: {p}' for p in invariant_problems(f)]
    for i in range(len(findings)):
        for j in range(i):
            jac = jaccard(findings[i]['text']['explanation'], findings[j]['text']['explanation'])
            if jac > JACCARD_MAX:
                out.append(f'{findings[i]["id"]} and {findings[j]["id"]}: explanations {jac:.2f} alike')
    return out


__all__ = ['explain', 'validate', 'build_claims', 'register_for', 'phase_source_of', 'failure_mode', 'jaccard',
           'is_moments', 'moments_problems', 'fix3_problems', 'MOMENTS', 'MOMENTS_KEYS', 'MOMENTS_PREFIX',
           'behind_margin', 'behind_words', 'definition_text', 'moment_actor', 'gap_partner', 'STYLE_ACCEPTS',
           'STYLE_EFFECT', 'BEHIND_MARGIN_M', 'DEFINITIONS',
           'invariant_problems', 'presser_of', 'cover_of', 'phase_known', 'no_alt_facts',
           'report_problems', 'request_for', 'short_for', 'shape_relations', 'variant_of', 'validate_free',
           'with_no_alternative', 'REGISTERS', 'MEASUREMENT_KEYS', 'PRINCIPLES', 'BANK', 'POLARITY', 'CONCEPTS',
           'QUALIFIER', 'LIMITS', 'JACCARD_MAX']
