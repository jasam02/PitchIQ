"""The training drills a coaching report links to its findings: the v2 library (drills.json), its validator, the
safety filter and the candidate step of drill selection (DESIGN v2 section 7; spec Part 11).

Why a new library. The v1 library (now drills_v1.json, kept only for the v1 engine until it retires) had 6 drills,
3 wording variations (development / competitive / advanced), a generic "conventional session" source line and no
heading or age rule; drills were chosen by recommendation key, so the audit found identical drills at every level, and
the round-2 critic counted 0 drills on 24 real reports (M1). The spec asks for drills per playing level (1-7), tied to
the tactical detector a finding comes from, credited to real coaching resources, age-appropriate, and never with
heading where the governing rules forbid it. The v2 library has 14 drills across the 7 levels (3 to 5 per level),
each a PitchIQ adaptation or PitchIQ original labelled as such, with the ids of the sources that informed it.

What is checked, and why each rule exists (validate_library; a broken library must not reach a report):
- the schema record (schema.Drill, closed: an unknown key is a problem);
- skills: primary and secondary skills are progression skill ids (schema.SKILLS), and their detectors are exactly the
  drill's related detectors, so selection, progression and the "what it trains" text read one vocabulary;
- success signals: every related detector has at least one, each names a relation of RELATIONS for its detector (the
  failed relation a finding reports, ranked +2 in selection, DESIGN 7.3) and carries no digits (the report fills in
  its own numbers; a number in the library would be invented);
- safety: every age inside a drill's age range passes the safety filter, so the library never lists an age it would
  refuse (heading drills are refused at every age, see below);
- player counts: the game in the name ("3v2") fits the player range, and every other "N v M" game fits under the
  maximum (the v1 library once asked for 4 v 4 in an exercise for at least 12 players);
- wording: printable ASCII (the PDF writer is base-14 Helvetica / WinAnsi); no unmeasured event (schema.event_problems:
  no "pass", "won/lost the ball", counter-attack, turnover, communication), because TEXT quotes regressions,
  progressions and success signals in the report; no heading, crossing, sprint or race words; no approval claims; the
  level vocabulary bans of DESIGN 6.3 for every level the drill is offered at (a level-1 drill says "helper", not
  "cover"); quoted phrases of at most 60 characters (sources are paraphrased, never copied);
- credit: every source note says "PitchIQ" (adaptation or original), and an adapted_from note also says "not a
  verbatim" federation session (spec Part 11); with a sources registry, every id must resolve and must not be a source
  that was not read.

The safety filter (safety_problems) is a hard filter that runs before any ranking (DESIGN 7.4). It applies the
stricter of an age floor kept here (DESIGN 2.4) and the profile's safety block (profiles.json), so a mis-set profile can
never loosen it:
- heading: none up to U11 in practice in both federations researched [F] (U.S. Soccer coaching guides U7-U8 to
  U11-U12: none up to U11; England Football heading guidance EF-HEADING: none in training U7-U11). From U12 a
  heading drill would have to declare a volume within the stricter U.S./England cap, and the v2 drill schema has no
  volume field, so every heading drill is refused at every age (and U19+ guidance was not researched). The library
  has none;
- physical load [P]: low up to U10, moderate U11-U14, high from U15;
- contact [P]: no tackling below U9 (the England 2023 4v2 pressing game is a no-tackle game [F]); a profile with
  tackling false also refuses contact drills;
- the drill's age range.
Measured on this library (tests/test_coaching_drills.py): candidates() enumerated over 17 age groups x 7 levels x 5
formats x 5 styles x 6 detectors (17,850 profile/detector pairs, 249,900 drill examinations, 15,150 candidates
returned) offers no drill with heading, no load above the age cap, no contact drill below U9 and no age outside the
drill's range, and the style never changes the drills; at every typical age of each level (DESIGN 2.2) and in the
spec's four age x level examples, every detector the profile table lets produce a work item (29 level/detector
pairs: DESIGN 2.3, plus the level-2 D6 try_this of DESIGN 4.6) has at least one allowed drill. The validator catches
each of 38 planted defects (30 in wave 5, 5 in fix round 1, 3 in fix round 3).

Candidates (DESIGN 7.3 steps 2-3): drills of the finding's detector, offered at the profile's level, that pass the
safety filter and the format filter (fix round 1, below), ranked by: the finding's failed relation is one of the
drill's success-signal relations (+2), the drill's primary skill is the detector's skill (+1), then the team's match
format among recommended_format (a 5v5 team also matches 4v4 drills: DESIGN 2.2 assesses 5v5 with the 4v4 rules),
then the distance from the drill's middle level to the profile level, then the drill introduced at the higher level
(with two-level drills the distance ties: at level 3 the level-3 drill comes before the level-2 drill, which DESIGN
2.3 lists as that level's regression), then the id.

Selection (select / selection, the interpretation tier's 'drills' stage; DESIGN 7.3 steps 1, 4, 5; spec Part 11):
- only work items get drills: up to three priorities in assess's order, one drill each, a second only when it trains a
  failed relation the first does not (no random exercises: a review, strength or description never gets one). The
  failed relation is read from the finding, never guessed (failed_relations: the failure mode, the measured centre-back
  shift / midfielder drop, the level's margin rule, the width / depth / channel shares against the level's values);
- a drill a higher priority already got is not repeated when another ranks as high; when none does, the same drill
  serves both priorities (two picks, one per finding);
- no work item reported: one optional drill for the next objective (assess's), labelled "for the next objective, not
  from this match's evidence", citing no moment;
- when the safety filter leaves no drill of the level at the team's age (a level above the age's typical level: a
  U11 team at level 4 has no allowed level-4 drill for D2, D3 and D4), the nearest lower level's allowed drill is
  offered and the text says so (D3 and D4 there; D2 has none below level 4); the filter is never relaxed (the
  same for the format filter, fix round 1); since fix round 3 never from below fallback_floor (see below);
- the six fields are written from the finding's claims (explain.build_claims) and each passes explain's gate at the
  profile's level before it is shown: numbers only as the level shows them (none at level 1, counts from 2, a share
  from 5, a rate per 10 minutes of the phase from 6), the phase with its qualifier ("(estimated)" / "(coach-marked)"),
  the observed relation in explain's polarity phrases (so the gate confirms it against TEXT's claims), no level
  concept too early, no banned or repeated-pattern word. A field that fails falls back to a shorter plan; a pick
  whose field cannot pass is dropped. Quoted library lines ('easier', 'harder') keep the drill's own setup numbers
  ("2 v 1"): the gate's number checks are not asked of them, every other check is.
Why the library was reworded for this (wave 5): explain's gate refused 37 of the 182 lines a report quotes (success
signals, regressions, progressions) at some level they are offered at: "more often" is a repeated-pattern word the
report may not use in a range under 3 minutes, "must" is a banned tone word, "Out of possession" names a phase
without its qualifier, "width" is a level-3 concept in a level-1 signal, "three lanes" a number word. All 182 now pass
at every offered level, in a short range too, and the validator refuses those words in quoted lines.
Measured (CPU, Oct 22): through the stage, 1785 synthetic work items (17 age groups x 7 levels x 5 formats x every
detector the level can criticise) got 1650 picks, each passing the safety filter at its age; the 135 without a
pick have no drill of their detector allowed at that age at the level or below (D3 under U8, D2 under U12, D5 under
U11, at levels above the age's typical one); 235 picks come from a lower level. The stage takes 4.3-4.4 ms median per
report (18 ms worst, the first call; two runs) on the 16 team-runs x 7 levels of the five run copies, where the only
work items are Barcelona's attacking depth at levels 1-2 (BC180, engine and coach-marked phases: the spread-out game);
the other 108 reports get the optional next-objective drill.

Fix round 1 (Oct 23; critic S-9 drills part, coach verifier M3 and M5, sources verifier F9). The match format used to
be only a ranking key, so a 4v4 team at level 3 got the 9v9 shape drill (14-20 players), 7v7 teams at levels 5-7 the
8v8 and 11v11 sessions, and 4v4 teams at levels 4 and 6 the next objective "Support behind advancing wide defenders", a
check 4v4 cannot assess. Now:
- the format filter (format_problems) runs with the safety filter, before any ranking, and is never relaxed: no drill
  whose every recommended format is larger than the team's, none needing more players (goalkeepers counted) than one
  game of the team's format (2 x players a side [P]: the settings carry no squad size). A smaller game stays allowed
  ("It is played as a smaller game than your match format"). When the level has no drill passing both filters, the
  nearest lower level's is offered and 'why' says why (EARLIER_NOTES: the age, the format, or both); when no level
  has one, selection says 'no_drill_for_format' (no_drill_detail; no_drill_reason keeps the report's two codes);
- player_count counts goalkeepers ("6 v 6 plus goalkeepers" = 14; F9): the validator checks the name's and the
  note's games with their keepers, that the minimum can play a game of its note, that every recommended format's
  game has room for the minimum, and that a drill offered below 11v11 uses no 11v11 role name (DESIGN 2.5). Library
  data changed with it: l4_defensive_unit_6v6 min 12 -> 14 and also 7v7 (its 6 v 6 plus goalkeepers is a 7-a-side
  game, three at the back as the U.S. 1-3-2-1); l5_transition_and_cover_8v8 min 16 -> 18; l4_cover_the_line_3v3 also
  4v4 / 5v5 (a 3 v 3 game for 6-12 players); the 9v9-and-up l5_transition_and_cover_8v8 and l6_rest_defence_9v9 say
  "wide defender", "central midfielder", "central defender" instead of full-back, holding midfielder, back four, wing
  backs. The wave-1 library fails the new rules with 9 problems;
- the next objective without a work item (level_objective, objective_exclusion): never off, unnamed, a check the
  format cannot assess (FORMAT_UNSUPPORTED, kept equal to the measurement tier by a test) or one this match
  demonstrated (demonstrated_checks: a strength, or reported findings with at least one demonstrated situation and
  none failed, assess's rule on what this stage sees); a passed NextObjective is used unless excluded, and assess's
  "no objective" (skill None) gives no optional drill.
Measured (CPU, Oct 23): the 1785 synthetic work items of the enumeration above: picks above the format 339 -> 0, needing
more players than the format 220 -> 0; of the 1581 the format can assess, every one that had a drill keeps one (1476;
the other 105 are refused by the safety filter, as before), 154 picks change, all at 4v4 / 5v5 / 7v7; the optional
drill over the 595 age x level x format profiles: objectives the format cannot assess 68 -> 0, drills above the
format 123 -> 0, more players 87 -> 0. Inside coaching.interpret on identical inputs (the old and the new stage side
by side; scratch coaching2/fix1/drills/ab.py): EVAL's 30 scene variants (330 reports, 77 small-format): picks above
the format 46 -> 0, more players 34 -> 0, 11v11 role words in a pick's fields 14 -> 0, unsupported objectives 8 -> 0;
the five run copies (110 reports, 40 with 4v4 / 7v7 declared): 40 -> 0, 30 -> 0, 20 -> 0, 10 -> 0; at 11v11 every
report picks the same drills with the same fields (253 of 253 scene, 70 of 70 run, 434 of 434 Metrica game 1 reports
of 3-minute clips).

Fix round 3 (Oct 24; coach verifier F-6, F-7, F-8, F-11 drill parts; facts verifier F-10; critic S-3):
- F-6, the fallback. It went down any number of levels, so adult level 4-7 teams got the U11-U12 9v9 shape drill and
  the U7-U8 no-tackle 4v4 game, saying "No drill of this level is allowed at this age" where the level had no drill
  of the check at all. Now a lower level's drill comes from no lower than fallback_floor: the level just below
  (MAX_LEVELS_BELOW 1 [P]), or the typical level of the team's age when that is lower (a team younger than its level
  keeps an age-fitting drill); the reason is the true one ('none_at_level': "The library has no drill of this level
  for this skill, ..."); without a drill, no_drill_detail's code is true of every drill of the check at the level and
  below ('only_far_below' when the nearest is under the floor; the report words it as 'no_drill_at_level'), and an
  objective without a drill is in selection's 'skipped' (finding None).
- F-7: the level-6 shadow drill's easier line said "Repeat the shadow phase ...", a word report.SHORT_REPEAT refuses
  under 3 minutes, so the report dropped it as the focus drill of levels 6-7 on every clip of 3 minutes or less (0
  optional drills at level 6 in the 196 run copies). The line now says
  "Run the shadow phase again ..."; the validator refuses those words (REPORT_REPEAT, equal to the report's and
  EVAL's by a test) in every drill text a coach reads.
- F-8 = facts F-10: the optional drill's progress line said "... in more of these moments than in this match" for
  checks the match never decided. Every success signal now ends with COMPARE (4 reworded: the goal-side and the three
  ball-side gap signals), and the optional drill uses it only when a reported finding of the check counts a decided
  situation in the check's own phase (decided_situations; phase-unknown shape moments do not count), else the
  signal without it (neutral_signal). A work item's drill compares as before (it failed in this match).
- F-11: a drill line's share and rate follow explain's rules for the same numbers (_counts): no share at the ends or
  one that is not the share of the stated counts, a rate only from 5 minutes of the phase and a whole number below
  10, the spells only from 2.
Measured (CPU, Oct 24; old and new module side by side on identical inputs, scratch coaching2/fix3/drills):
- synthetic enumeration (17 ages x 7 levels x 5 formats; enum3.py): work items 1476 of 1581 with a drill before and
  after (the 101 from two to four levels down are all for teams younger than their level, kept by the floor); every
  check as the objective (2720 pairs): drills 2410 -> 2110, from two or more levels down 730 -> 430 (all for younger
  teams, at or above their age's typical level), false reasons 340 -> 0, progress lines comparing with a match that
  reported nothing 2246 -> 0;
- inside coaching.interpret (ab3.py), verdicts and next objectives identical in every report: the coach verifier's
  39 scene variants x 14 profiles (546 reports): drills from two or more levels down 56 -> 0 (52 optional, 4 work
  items: adult 7v7 at level 7, now "No drill in the library fits ... at this level"), false reasons 47 -> 0,
  "(about 100%)" 60 -> 0, rates from under 5 minutes 51 -> 0, repetition words 33 -> 0, comparisons with an
  undecided check 114 -> 0; the 7 run variants x 2 teams x 14 profiles (196 reports): optional drills 129 -> 169
  (level 6 under 3 minutes: 0 -> 39 of 42), from two levels down 2 -> 0, false reasons 3 -> 0, comparisons with an
  undecided check 124 -> 0; Metrica game 1 3-minute clips x 2 teams x 9 profiles (558 reports): optional drills 339
  -> 381 (level 6: 5 -> 45), from two or more levels down 19 -> 0, false reasons 27 -> 0, comparisons with an
  undecided check 204 -> 0. The comparison is now also left out where the match decided the check but reported no
  finding of it (5 run-copy, 58 scene, 127 Metrica optional drills): what the report does not show is not compared.

Physical load classes [P] (used in the safety block): low = short rounds (at most 3 min) with rests at least half as
long, small areas, no repeated maximal running; moderate = game blocks of 3-5 min, repeated short recovery runs with the
ball in play; high = larger areas or longer blocks with repeated high-speed runs.
"""
import copy
import hashlib
import json
import re
from pathlib import Path

from soccer.coaching import schema as S

LIBRARY_PATH = Path(__file__).with_name('drills.json')           # v2 (this module's library)
SOURCES_PATH = Path(__file__).with_name('sources.json')          # PROFILE's registry (may not exist yet)
V1_LIBRARY_PATH = Path(__file__).with_name('drills_v1.json')     # v1, read only by the v1 engine
DRILLS_PATH = V1_LIBRARY_PATH                                     # the v1 name, kept for old callers

# ---------- safety floor (DESIGN 2.4); the profile's safety block can only make it stricter ----------
HEADING_NONE_UP_TO = 'U11'          # [F] U.S. Soccer age guides (none to U11); EF-HEADING (none in training U7-U11)
HEADING_RESEARCHED_UP_TO = 'U18'    # [F] caps researched U12-U18 only; U19+ / Adult guidance was not researched
NO_TACKLE_BELOW = 'U9'              # [P] no tackling progressions below U9 (DESIGN 2.4)
LOADS = ('low', 'moderate', 'high')
LOAD_CAP_FROM = (('U6', 'low'), ('U11', 'moderate'), ('U15', 'high'))   # [P] cap from each age on (DESIGN 2.4)

# ---------- the match format (fix round 1, critic S-9 / coach M3 / sources F9; see the module docstring) ----------
FORMAT_PLAYERS = {'4v4': 4, '5v5': 5, '7v7': 7, '9v9': 9, '11v11': 11}   # players a side, the keeper included
# The checks a format cannot assess: the measurement tier returns format_not_supported for D3 and D5 at 4v4 and its
# 5v5 alias (contextual._d3 and _d5_small: no back block of three; DESIGN 2.5 "4v4 has no D5", DESIGN 4.3 abstain
# "format_not_supported (4v4: no back block)"). A copy, kept equal to the tier by a test (kids_swarm, 4v4).
FORMAT_UNSUPPORTED = {'wide_defender_advance': (4, 5), 'defensive_spacing': (4, 5)}
# 11v11 role names (DESIGN 2.5: small formats never use them); a drill offered below 11v11 uses relative words
# ("wide defender", "central midfielder", "the nearest central defender").
ELEVEN_ROLES = re.compile(r'\b(?:full-?backs?|cent(?:re|er)[- ]backs?|holding midfielders?|wing[- ]?backs?|'
                          r'back (?:four|five)|sweepers?)\b', re.I)
# "6 v 6 plus goalkeepers": the clause after a game (up to ';' or '.') adds two keepers to its outfield players.
KEEPERS_CLAUSE = re.compile(r'^[^;.]*?\b(?:with|plus) goalkeepers\b', re.I)

# ---------- relations a success signal (and a finding's failed relation) may name, per detector ----------
# DESIGN 4.1-4.6 outcomes and 5.1 step 1 ("the failed relation"). Proposed in INTERFACES.md as the shared vocabulary
# for Claim.facts.relation, Alternative.improves.relation and the relation element's kind.
RELATIONS = {
    'pressure_cover': ('pressure', 'cover', 'goal_side'),
    'weak_side_balance': ('weak_side', 'central_protection'),
    'wide_defender_advance': ('channel_cover', 'cb_shift', 'dm_drop', 'interchange'),
    'behind_ball': ('behind_ball', 'rest_margin', 'behind_spread'),
    'defensive_spacing': ('lateral_gap', 'unit_gap', 'line_evenness'),
    'attacking_shape': ('width', 'depth', 'channels'),
}
SKILL_OF = {d: s for s, d in S.SKILLS.items()}     # detector -> progression skill

# ---------- wording ----------
# DESIGN 6.3 "banned additions" per level, applied to the text of every drill offered at that level (profiles.json
# holds the runtime register; this copy guards the library when it is loaded).
WORDS = {
    'mistake': r'\bmistakes?\b', 'balance': r'\bbalanc\w*', 'cover': r'\bcover\w*', 'press': r'\bpress\w*',
    'rest defence': r'\brest[- ]defen[cs]e\b', 'transition': r'\btransition\w*', 'trigger': r'\btrigger\w*',
}
BANNED_BY_LEVEL = {1: ('mistake', 'balance', 'cover', 'press', 'rest defence', 'transition'),
                   2: ('mistake', 'balance', 'rest defence', 'transition', 'trigger'),
                   3: ('mistake', 'rest defence', 'trigger'), 4: ('mistake', 'trigger'), 5: ('mistake',),
                   6: (), 7: ()}
# Everywhere in a drill (source notes included): words a report may not use (DESIGN 6.3 global bans), heading and the
# crosses that invite it, sprint races (DESIGN 2.4 keeps them out of the v2 library), approval claims.
GLOBAL_BANNED = (
    ('failure word', re.compile(r'\bfail(?:s|ed|ing|ure)?\b', re.I)),
    ('should have', re.compile(r'\bshould have\b', re.I)),
    ('poor', re.compile(r'\bpoor(?:ly)?\b', re.I)),
    ('always/never', re.compile(r'\b(?:always|never)\b', re.I)),
    ('stay in formation', re.compile(r'\bstay in formation\b', re.I)),
    ('heading', re.compile(r'\bhead(?:s|ed|ing|er|ers)?\b', re.I)),
    ('crossing', re.compile(r'\bcross(?:es|ed|ing)?\b', re.I)),
    ('sprint or race', re.compile(r'\b(?:sprint\w*|races?|racing)\b', re.I)),
    ('approval claim', re.compile(r'\b(?:approved|endorsed|certified|accredited|licensed)\b', re.I)),
    ('obligation word', re.compile(r'\bmust\b', re.I)),          # language.TONE_BANNED: the report never says "must"
)
# The lines a report quotes (success signals in 'progress', regressions in 'easier', progressions in 'harder': drill
# selection, DESIGN 7.3 step 5) pass explain's gate in any report: no repeated-pattern words (refused in a range under
# 3 minutes; a copy of language.SHORT_BANNED, kept equal by a test), no phase named without its qualifier ("out of
# possession": the report says "when the other team had the ball (estimated)"), and in a success signal no number
# word (a signal carries no numbers: the report fills in its own).
QUOTED_PATHS = ('progression[', 'regression[', 'success_signals_next_match[')
REPEAT_WORDS = ('recurring', 'recurred', 'recur', 'repeated', 'repeatedly', 'pattern', 'patterns', 'habit', 'habits',
                'habitually', 'regularly', 'consistently', 'often', 'frequently', 'tends to', 'tendency', 'routinely',
                'persistent', 'persistently')
REPEAT_RE = re.compile(r'(?<![a-z\-])(?:'+'|'.join(re.escape(w) for w in REPEAT_WORDS)+r')(?![a-z\-])', re.I)
LOOSE_PHASE = re.compile(r'\b(?:in|out of) possession\b', re.I)
NUMBER_WORD = re.compile(r'\b(?:two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|twenty|twice|thrice)\b',
                         re.I)
INSTRUCTION_BANNED = (('federation name in instructions', re.compile(r'\bUEFA\b')),)
# Fix round 3 (coach F-7): the words the report (report.SHORT_REPEAT) and EVAL (coaching_eval.REPETITION) refuse in
# any text of a range under 3 minutes; a copy, kept equal to both by a test. The page and the PDF print every
# instruction line of a drill card, and the report drops a focus drill whose fields carry one of them, so no drill
# text may (the level-6 shadow drill's "Repeat the shadow phase ..." kept level 6 from any focus drill on clips of
# 3 minutes or less: 0 optional drills at level 6 in the 196 run copies).
REPORT_REPEAT = re.compile(r'\b(repeated(?:ly)?|repeats?|patterns?|often)\b', re.I)
# Fix round 3 (coach F-8 = facts F-10): every success signal ends with this comparison, so the comparison-free form
# (neutral_signal) is always at hand for a check this match did not judge ("... in more of these moments than in
# this match" was printed for checks with 0 decided situations: 144 of 419 optional drills, 115 of 196 run copies).
COMPARE = ' in more of these moments than in this match.'
QUOTED = re.compile(r'"([^"]*)"')
MAX_QUOTE = 60                      # a quoted phrase from a source stays short (DESIGN 2.7)
# An 'N v M' game ('3v2', '11 v 7'); 'half of an 11v11 pitch' names a size, not a game.
GAME = re.compile(r'\b(\d{1,2})\s?v\s?(\d{1,2})\b(?!\s+(?:pitch|pitches|field|fields|format|team|teams)\b)')
CREDIT = 'PitchIQ'                  # every source note says it is a PitchIQ adaptation or original
NOT_VERBATIM = 'not a verbatim'     # ... and an adaptation says it is not a verbatim federation session

_cache = {}


# ================================ v2 ================================
def load_library(path=None):
    """The validated v2 library {'version': 2, 'notes', 'drills', '_sha'} (cached per file and modification time).
    Raises ValueError listing the problems when the file does not validate."""
    p = Path(path) if path is not None else LIBRARY_PATH
    st = p.stat()
    key = (str(p.resolve()), st.st_mtime_ns, st.st_size)
    hit = _cache.get('v2')
    if hit is None or hit[0] != key:
        data = p.read_bytes()
        library = json.loads(data.decode('ascii'))
        problems = validate_library(library)
        if problems:
            raise ValueError('drill library invalid: '+'; '.join(problems[:20]))
        library['_sha'] = hashlib.sha256(data).hexdigest()[:16]
        _cache['v2'] = (key, library)
    return _cache['v2'][1]


def load_sources(path=None):
    """The sources registry as {id: record}, aliases included, or None when the file does not exist yet. Accepts a
    list of records or {'sources': [...]}; each record has 'id' and optionally 'aliases' and 'read'."""
    p = Path(path) if path is not None else SOURCES_PATH
    if not p.exists():
        return None
    raw = json.loads(p.read_text(encoding='utf-8'))
    return sources_index(raw)


def sources_index(raw):
    """{id or alias: record} from a registry (a list of records, {'sources': [...]} or an {id: record} map)."""
    if isinstance(raw, dict) and isinstance(raw.get('sources'), list):
        raw = raw['sources']
    if isinstance(raw, dict):
        raw = [dict(rec, id=k) if isinstance(rec, dict) and 'id' not in rec else rec for k, rec in raw.items()]
    out = {}
    for rec in raw or ():
        if isinstance(rec, dict) and isinstance(rec.get('id'), str):
            out[rec['id']] = rec
            for alias in rec.get('aliases') or ():
                if isinstance(alias, str):
                    out.setdefault(alias, rec)
    return out


def validate_library(library, sources=None):
    """Every problem with a v2 library, as readable strings ([] when valid). sources: an optional registry index
    (sources_index / load_sources) every cited id must resolve in, and not as a source that was not read."""
    if not isinstance(library, dict):
        return ['the library is not an object']
    clean = {k: v for k, v in library.items() if not str(k).startswith('_')}
    problems = S.validate('drill_library', clean)
    if problems:
        return problems                  # the library rules below assume the schema holds
    for d in clean['drills']:
        problems += [f'drill {d["drill_id"]}: {p}' for p in drill_problems(d, sources)]
    return problems


def drill_problems(d, sources=None):
    """The library rules for one schema-valid drill (see the module docstring)."""
    out = []
    skills = [d['primary_skill']]+list(d['secondary_skills'])
    unknown = [s for s in skills if s not in S.SKILLS]
    if unknown:
        out.append(f'skills {unknown} are not progression skills ({", ".join(S.SKILLS)})')
    elif d['primary_skill'] in d['secondary_skills'] or len(set(skills)) != len(skills):
        out.append('skills repeat')
    elif {S.SKILLS[s] for s in skills} != set(d['related_tactical_detector']):
        out.append(f'the skills train {sorted({S.SKILLS[s] for s in skills})} but the related detectors are '
                   f'{sorted(d["related_tactical_detector"])}')
    signalled = set()
    for k, sig in enumerate(d['success_signals_next_match']):
        allowed = RELATIONS.get(sig['detector'], ())
        if sig['relation'] not in allowed:
            out.append(f'success_signals_next_match[{k}].relation {sig["relation"]!r} is not one of {allowed}')
        if (sig['detector'], sig['relation']) in signalled:
            out.append(f'success_signals_next_match[{k}] repeats {sig["detector"]}/{sig["relation"]}')
        signalled.add((sig['detector'], sig['relation']))
        if re.search(r'\d', sig['text']):
            out.append(f'success_signals_next_match[{k}].text has a number (the report fills in its own)')
        if not sig['text'].endswith(COMPARE) or 'this match' in neutral_signal(sig['text']):
            out.append(f'success_signals_next_match[{k}].text must end with "{COMPARE.strip()}" and compare with '
                       'this match nowhere else (the comparison is left out when this match did not judge the check)')
    missing = sorted(set(d['related_tactical_detector'])-{det for det, _ in signalled})
    if missing:
        out.append(f'no success signal for {missing}')
    lo, hi = S.AGE_ORDER[d['age']['min']], S.AGE_ORDER[d['age']['max']]
    for age in S.AGE_GROUPS[lo:hi+1]:
        refused = safety_problems(d, age)
        if refused:
            out.append(f'listed for {age} but refused there ({", ".join(refused)})')
            break
    pc = d['player_count']
    note_games = game_headcounts(pc['note'])
    for n in game_sizes(d['name']):
        n = next((h for g, h in note_games if g == n), n)      # the name's game as the note plays it (keepers)
        if not pc['min'] <= n <= pc['max']:
            out.append(f'the {n}-player game in the name does not fit player_count {pc["min"]}-{pc["max"]}')
    if note_games and pc['min'] < min(h for _, h in note_games):
        out.append(f'player_count min {pc["min"]} cannot play any game of its note (the smallest needs '
                   f'{min(h for _, h in note_games)} players, goalkeepers counted)')
    for path, text in instruction_texts(d):
        for _, n in game_headcounts(text):
            if n > pc['max']:
                out.append(f'{path}: a {n}-player game (goalkeepers counted) does not fit player_count max '
                           f'{pc["max"]}')
    for fmt in d['recommended_format']:
        if pc['min'] > 2*FORMAT_PLAYERS[fmt]:
            out.append(f'recommended for {fmt}, but player_count min {pc["min"]} is more than a {fmt} game has '
                       f'({2*FORMAT_PLAYERS[fmt]})')
    if min(FORMAT_PLAYERS[f] for f in d['recommended_format']) < 11:
        for path, text in instruction_texts(d):
            m = ELEVEN_ROLES.search(text)
            if m:
                out.append(f'{path}: 11v11 role name "{m.group(0)}" in a drill offered below 11v11 (use relative '
                           'words)')
    for k, ref in enumerate(d['source_reference']):
        if CREDIT not in ref['note']:
            out.append(f'source_reference[{k}].note must say it is a {CREDIT} adaptation or original')
        if ref['relation'] == 'adapted_from' and NOT_VERBATIM not in ref['note']:
            out.append(f'source_reference[{k}].note must say it is {NOT_VERBATIM} federation session')
        if sources is not None:
            rec = sources.get(ref['source'])
            if rec is None:
                out.append(f'source_reference[{k}].source {ref["source"]} is not in the sources registry')
            elif str(rec.get('read', '')).replace('-', '_') in ('not_read', 'not_fetched'):
                out.append(f'source_reference[{k}].source {ref["source"]} was not read, so it cannot be cited')
    out += text_problems(d)
    return out


def text_problems(d):
    """Wording problems of one drill: unmeasured events, global bans, the level bans of every level it is offered at,
    long quotations."""
    out = []
    level_words = sorted({w for lv in d['development_level'] for w in BANNED_BY_LEVEL.get(lv, ())})
    level_res = [(w, re.compile(WORDS[w], re.I)) for w in level_words]
    instruction_paths = {p for p, _ in instruction_texts(d)}
    for path, text in drill_texts(d):
        for p in S.event_problems(text):
            out.append(f'{path}: names an unmeasured event ({p})')
        for name, pat in GLOBAL_BANNED:
            m = pat.search(text)
            if m:
                out.append(f'{path}: {name} ("{m.group(0)}")')
        for q in QUOTED.findall(text):
            if len(q) > MAX_QUOTE:
                out.append(f'{path}: a quoted phrase is longer than {MAX_QUOTE} characters')
        if path.startswith(QUOTED_PATHS):
            m = REPEAT_RE.search(text) or LOOSE_PHASE.search(text)
            if m:
                out.append(f'{path}: "{m.group(0)}" cannot be quoted in every report (repeated-pattern word or a '
                           'phase without its qualifier)')
            m = NUMBER_WORD.search(text) if path.startswith('success_signals_next_match[') else None
            if m:
                out.append(f'{path}: number word "{m.group(0)}" (the report fills in its own numbers)')
        if path in instruction_paths:
            m = REPORT_REPEAT.search(text)
            if m:
                out.append(f'{path}: "{m.group(0)}" is a repetition word the report refuses under 3 minutes of play')
            for word, pat in level_res:
                m = pat.search(text)
                if m:
                    out.append(f'{path}: "{m.group(0)}" is not level-{min(d["development_level"])} wording '
                               f'(banned: {word})')
            for name, pat in INSTRUCTION_BANNED:
                if pat.search(text):
                    out.append(f'{path}: {name}')
    return out


# ---------- the safety filter (DESIGN 2.4, 7.4) ----------
def load_cap(age_group):
    """The physical-load cap of an age group [P]."""
    cap = LOADS[0]
    for start, load in LOAD_CAP_FROM:
        if S.AGE_ORDER[age_group] >= S.AGE_ORDER[start]:
            cap = load
    return cap


def effective_safety(age_group, profile_safety=None):
    """The rules applied at an age: the stricter of the age floor and the profile's safety block (schema
    ProfileSafety; None = the floor only). {'heading': 'none'|'limited', 'maxLoad', 'tackling'}."""
    age = S.AGE_ORDER[age_group]
    heading = 'none' if age <= S.AGE_ORDER[HEADING_NONE_UP_TO] else 'limited'
    max_load = load_cap(age_group)
    tackling = age >= S.AGE_ORDER[NO_TACKLE_BELOW]
    ps = profile_safety or {}
    if ps.get('heading') == 'none':
        heading = 'none'
    if ps.get('maxLoad') in LOADS and LOADS.index(ps['maxLoad']) < LOADS.index(max_load):
        max_load = ps['maxLoad']
    if ps.get('tackling') is False:
        tackling = False
    return {'heading': heading, 'maxLoad': max_load, 'tackling': tackling}


def safety_problems(drill, age_group, profile_safety=None):
    """Why a drill may not be offered at an age (codes; [] = allowed). Runs before any ranking."""
    out = []
    rules = effective_safety(age_group, profile_safety)
    age = S.AGE_ORDER[age_group]
    sf = drill['safety']
    if sf['heading'] != 'none':
        if rules['heading'] == 'none':
            out.append('heading_not_allowed_at_age')
        elif age > S.AGE_ORDER[HEADING_RESEARCHED_UP_TO]:
            out.append('heading_guidance_not_researched')
        else:
            out.append('heading_volume_undeclared')    # the schema has no volume field to check against the cap
    if LOADS.index(sf['physicalLoad']) > LOADS.index(rules['maxLoad']):
        out.append('load_above_cap')
    if sf['contact'] != 'no_tackle' and not rules['tackling']:
        out.append('tackling_not_allowed')
    if age < S.AGE_ORDER[drill['age']['min']]:
        out.append('below_age_range')
    if age > S.AGE_ORDER[drill['age']['max']]:
        out.append('above_age_range')
    return out


def allowed(drill, age_group, profile_safety=None):
    """True when the safety filter lets the drill be offered at that age."""
    return not safety_problems(drill, age_group, profile_safety)


def refused(library, profile):
    """{drill_id: codes} of the drills the safety filter removes for a profile (for the report's notes)."""
    out = {}
    for d in library['drills']:
        codes = safety_problems(d, profile['ageGroup'], profile.get('safety'))
        if codes:
            out[d['drill_id']] = codes
    return out


# ---------- candidates (DESIGN 7.3 steps 2-3) ----------
def format_name(profile):
    fmt = profile.get('format') or S.DEFAULT_FORMAT
    return fmt.get('name') or S.format_name(fmt)


def format_matches(drill, name):
    rec = drill['recommended_format']
    return name in rec or (name == '5v5' and '4v4' in rec)


def candidates(library, profile, detector, failed_relations=()):
    """The drills a finding of `detector` may get for this profile, best first. profile needs ageGroup, level,
    format and (optionally) safety, as profiles.resolve returns it. The safety filter and the format filter
    (format_problems) run first; then the ranking. Each item: {'drill_id', 'drill' (the record), 'relationMatch',
    'skillMatch', 'formatMatch', 'levelDistance'}."""
    level = profile['level']
    fmt = format_name(profile)
    skill = SKILL_OF.get(detector)
    failed = set(failed_relations or ())
    out = []
    for d in library['drills']:
        if detector not in d['related_tactical_detector'] or level not in d['development_level']:
            continue
        if safety_problems(d, profile['ageGroup'], profile.get('safety')) or format_problems(d, profile):
            continue
        rel = any(s['detector'] == detector and s['relation'] in failed for s in d['success_signals_next_match'])
        levels = d['development_level']
        out.append({'drill_id': d['drill_id'], 'drill': d, 'relationMatch': rel,
                    'skillMatch': d['primary_skill'] == skill, 'formatMatch': format_matches(d, fmt),
                    'levelDistance': abs((levels[0]+levels[-1])/2-level)})
    out.sort(key=lambda c: (-(2*c['relationMatch']+c['skillMatch']), not c['formatMatch'], c['levelDistance'],
                            -c['drill']['development_level'][0], c['drill_id']))
    return out


# ---------- selection: the 'drills' stage (DESIGN 7.3 steps 1, 4, 5; spec Part 11 drill_selection) ----------
MAX_PRIORITIES = 3          # [P] up to three priorities get drills (DESIGN 7.3 step 1; spec Part 11)
MAX_PER_PRIORITY = 2        # [P] one drill; a second only when it covers another failed relation (DESIGN 7.3 step 4)
MAX_LINES = 2               # [P] regression / progression lines quoted as 'easier' / 'harder'
FIELD_LIMITS = {'trains': 300, 'why': 500, 'easier': 400, 'harder': 400, 'progress': 400}   # schema DrillFields
SHARE_TOLERANCE = .001      # a stated share must be the share of the stated counts (recurrence shares carry 4 decimals)
# The gate (explain.validate) checks every field against the finding's claims. Quoted library lines ('easier',
# 'harder') carry the drill's own setup numbers ('2 v 1', 'about 8 m'): drill content, not measurements, so the
# facts, units and number-register checks are not asked of them; everything else is (level concepts, bans, events,
# phase wording, role words, relation polarity).
LIBRARY_CHECKS = ('naming', 'roles', 'phase', 'vocabulary', 'events', 'hypotheticals', 'polarity')
GATE_FILLER = 'Drill.'      # the headline / short the gate needs beside the checked field (passes every check)
NEXT_OBJECTIVE_LABEL = "for the next objective, not from this match's evidence"   # DESIGN 7.3 step 4
# Pre-freeze (Oct 24, critic S-3: DRILLS / PAGE / TEXT): NEXT_OBJECTIVE_LABEL was printed for every optional drill,
# also for an objective assess chose from this match's checks (critic S-8; report.S8_LABEL "Chosen from this match's
# checks (not a work item): ..."), where it was false. That drill now says so (objective_label); the old label stays for
# the level's suggested focus and the level's own objective, which do not come from this match.
MATCH_OBJECTIVE_LABEL = "for the next objective, chosen from this match's checks (not a work item)"
SKIP_REASONS = ('not_reported', 'no_drill_at_level', 'refused_by_safety', 'no_drill_for_format',
                'only_far_below', 'text_failed_validation')
ORDINAL = {1: 'first', 2: 'second', 3: 'third'}     # the priority's rank in 'why' (MAX_PRIORITIES)
# Fix round 3 (coach F-6): a drill from a lower level is offered only from the level just below, or from a level no
# lower than the typical level of the team's age (profiles.typical_level: a U8 team declared at level 4, whose
# level-4 drills the safety filter refuses, may still get an age-fitting level-1 or level-2 drill) [PITCHIQ-PROPOSAL].
# Unlimited, the fallback gave adult level 4-7 teams the U11-U12 9v9 shape drill and the U7-U8 no-tackle 4v4 game
# (50 reports in the coach verifier's set), with a reason that was false for them ("No drill of this level is allowed
# at this age": the level had no attacking-shape drill at all).
MAX_LEVELS_BELOW = 1
EARLIER_NOTE = 'No drill of this level is allowed at this age, so this one comes from an earlier level.'
EARLIER_NOTES = {'age': EARLIER_NOTE,
                 'format': 'No drill of this level fits your match format, so this one comes from an earlier level.',
                 'age_and_format': 'No drill of this level fits both this age and your match format, so this one '
                                   'comes from an earlier level.',
                 # fix round 3: the level has no drill of the check at all (level_gap None)
                 'none_at_level': 'The library has no drill of this level for this skill, so this one comes from an '
                                  'earlier level.'}

# How a failed relation is stated as observed (past tense). Each entry: (claim relation, claim polarity, first level,
# phrase); the first entry whose relation and polarity the finding's claims state is used, so the gate's polarity
# check confirms the sentence against TEXT's claims (explain.POLARITY phrases). None: worded by _behind_phrase.
OBSERVED = {
    'pressure': (('pressure', False, 1, 'nobody went to the ball'),),
    'cover': (('cover', False, 2, 'no teammate was covering behind the player who went to the ball'),),
    'weak_side': (('weak_side', False, 1, 'nobody was goal-side on the far side'),),
    'channel_cover': (('channel_cover', False, 4, 'nobody was in the channel behind a defender who moved forward'),
                      ('advance_support', False, 3, 'a defender moved forward and the space they left was not covered'),
                      ('behind_ball', False, 1, 'a defender moved forward and nobody stayed back')),
    'behind_ball': (('behind_ball', False, 1, None),),
    'rest_margin': (('behind_ball', False, 6, 'the rest-defence margin (a PitchIQ measure) was below what this level '
                                              'asks for'),),
    'lateral_gap': (('lateral_gap', True, 1, 'a large gap opened between our back players on the ball side'),),
    'unit_gap': (('unit_gap', True, 1, 'the lines were far apart'),),
    'width': (('width', False, 1, 'the team stayed narrow'),),
    'depth': (('depth', False, 1, 'the team stayed short from back to front'),),
    'channels': (('channels', False, 1, 'one side was empty'),),
}
# What a drill works on, per relation (present tense, level-safe words; the gate checks them anyway).
NOUN = {
    'pressure': 'going to the ball', 'cover': 'a teammate covering behind the player who goes to the ball',
    'goal_side': 'staying goal-side of the ball', 'weak_side': 'protecting the far side',
    'central_protection': 'protecting the middle',
    'channel_cover': ('filling the space behind a defender who moves forward',
                      'filling the channel behind a defender who moves forward'),
    'cb_shift': 'the nearest central defender moving across', 'dm_drop': 'a midfielder dropping in',
    'interchange': 'swapping places',
    'behind_ball': ('someone staying ready to protect our goal', 'keeping players behind the ball'),
    'rest_margin': 'the rest-defence margin (a PitchIQ measure)',
    'behind_spread': 'spreading the players behind the ball',
    'lateral_gap': 'keeping the ball-side gaps between back players small',
    'unit_gap': 'keeping the back players and midfield close', 'line_evenness': 'keeping the back players level',
    'width': 'using the field from one side to the other', 'depth': 'spreading from deep to tall',
    'channels': 'using the left, the middle and the right',
}
NOUN_LEVEL = {'channel_cover': 4, 'behind_ball': 3}     # the level from which the second wording of NOUN is used


def failed_relations(finding, profile=None):
    """The relations of RELATIONS[detector] a criticised or reviewed finding reports as missing, most specific first:
    what DESIGN 7.3 step 3 ranks +2. Read from the finding (measurements at its first key moment, then its claims) and
    the level's parameters, never invented: pressure and cover from the failure mode; far side; for an unsupported
    advance the vacated channel (and the centre-back shift / midfielder drop when measured as not done); behind the
    ball, or the rest-defence margin where the level judges a margin; the ball-side gap or the unit gap; width, depth
    and the three channels against the level's shares. [] for any other verdict."""
    if finding.get('verdict') not in S.CRITICISM+('review',):
        return []
    det = finding['observation']
    m = finding.get('measurements') or {}
    params = (((profile or {}).get('detectors') or {}).get(det) or {}).get('params') or {}
    absent, present = _claimed(finding.get('claims') or ())
    out = []
    if det == 'pressure_cover':
        mode = m.get('failureMode')
        if mode not in ('no_cover', 'no_pressure'):
            mode = 'no_pressure' if 'pressure' in absent else ('no_cover' if 'cover' in absent else None)
        out = {'no_cover': ['cover'], 'no_pressure': ['pressure']}.get(mode, [])
    elif det == 'weak_side_balance':
        out = ['weak_side']
    elif det == 'wide_defender_advance':
        out = ['channel_cover']
        if m.get('cbAdjusted') is False:
            out.append('cb_shift')
        if params.get('useDmRotation') and m.get('dmDropped') is False:
            out.append('dm_drop')
    elif det == 'behind_ball':
        out = ['rest_margin'] if params.get('behindRule') == 'margin' else ['behind_ball']
    elif det == 'defensive_spacing':
        mode = m.get('failureMode')
        if mode not in ('lateral_gap', 'dm_gap'):
            mode = 'lateral_gap' if 'lateral_gap' in present else ('dm_gap' if 'unit_gap' in present else None)
        out = {'lateral_gap': ['lateral_gap'], 'dm_gap': ['unit_gap']}.get(mode, [])
    elif det == 'attacking_shape':
        ws, ds, ch = m.get('widthShare'), m.get('depthShare'), m.get('channelsOccupied')
        if _num(ws) and _num(params.get('widthShare')) and ws < params['widthShare'] and \
                params.get('judgeWidth') is not False:
            out.append('width')
        if _num(ds) and _num(params.get('depthShare')) and ds < params['depthShare']:
            out.append('depth')
        if _num(ch) and ch < 3:
            out.append('channels')
        if not out:
            out = [r for r in ('width', 'depth', 'channels') if r in absent]
    return [r for r in out if r in RELATIONS.get(det, ())]


def select(findings, profile, priorities=None, *, next_objective=None, range=None, library=None):
    """The 'drills' stage (INTERFACES.md LEAD 4): [schema DrillPick] for a report, and each finding's 'drills' filled
    in place with the ids picked for it ([] for every other finding). See selection() for the rules."""
    return selection(findings, profile, priorities, next_objective=next_objective, range=range,
                     library=library)['picks']


def selection(findings, profile, priorities=None, *, next_objective=None, range=None, library=None):
    """Drill selection (DESIGN 7.3) with its diagnostics: {'picks': [DrillPick], 'skipped': [{'finding', 'reason'}],
    'objective': {'skill', 'detector'} | None}.
    - Priorities: the first MAX_PRIORITIES of `priorities` (assess's ranked work items) that are reported work items
      (try_this / work_on) in `findings`; without `priorities`, the work items by their priority. Reviews, strengths
      and descriptions never get a drill (no random exercises).
    - Per priority: candidates() (detector, level, safety filter first, then the failed relation +2, the primary skill
      +1, format, level centre); the first pick prefers a drill no higher priority already got when one ranks as high;
      a second pick only when it covers a failed relation the first does not.
    - No work item reported: one optional drill for the next objective (`next_objective`, assess's NextObjective, when
      it does not rest on a finding and objective_exclusion lets it through; else level_objective: the enabled check
      with the lowest objectiveRank that has a skill name, that the match format can assess and that this match did
      not demonstrate), labelled "for the next objective, not from this match's evidence", or "..., chosen from this
      match's checks (not a work item)" for assess's objective from this match's checks (objective_label); when the
      objective gets none, 'skipped' says why with finding None (no_drill_detail's code, or 'text_failed_validation').
    - The six fields are built from the finding's claims (explain's; built here when missing) and pass explain's gate
      at the profile's level (quoted library lines without the number checks, see LIBRARY_CHECKS); a field that
      fails falls back to a shorter plan, and a pick whose field cannot pass is dropped ('text_failed_validation').
    range: the report range {'seconds'} or {'t0', 't1'} (under 3 minutes no repeated-pattern words); without it each
    finding's recurrence claim decides, else the cautious side. library: a validated v2 library (default: drills.json).
    Deterministic; inputs are not modified except each finding's 'drills'."""
    from . import explain as E
    lib = library if library is not None else load_library()
    reg = E.register_for(profile)
    by_id = {f['id']: f for f in findings}
    work = [f for f in findings if f.get('verdict') in S.CRITICISM]
    if priorities is None:
        order = [f['id'] for f in sorted(work, key=lambda f: (f.get('priority') or 99, f['key']))]
    else:
        order = list(dict.fromkeys(priorities))
    picks, skipped, used = [], [], []
    chosen = {f['id']: [] for f in findings}
    ranked = []
    for fid in order:
        f = by_id.get(fid)
        if f is None or f.get('verdict') not in S.CRITICISM:
            skipped.append({'finding': fid, 'reason': 'not_reported'})
            continue
        ranked.append(f)
    for rank, f in enumerate(ranked[:MAX_PRIORITIES], start=1):
        det = f['observation']
        failed = failed_relations(f, profile)
        cands, earlier = ranked_candidates(lib, profile, det, failed)
        if not cands:
            skipped.append({'finding': f['id'], 'reason': no_drill_detail(lib, profile, det)})
            continue
        first = next((c for c in cands if _score(c) == _score(cands[0]) and c['drill_id'] not in used), cands[0])
        got = [first]
        covered = _covers(first, det, failed)
        for c in cands:
            if len(got) >= MAX_PER_PRIORITY:
                break
            if c is first or c['drill_id'] in used:
                continue
            if _covers(c, det, failed)-covered:
                got.append(c)
                covered |= _covers(c, det, failed)
        priority = f.get('priority') if isinstance(f.get('priority'), int) else rank
        for k, c in enumerate(got):
            fields = _pick_fields(f, c, failed, profile, reg, E, priority, second=k > 0, range_=range,
                                  earlier=earlier)
            if fields is None:
                skipped.append({'finding': f['id'], 'reason': 'text_failed_validation'})
                continue
            picks.append({'finding': f['id'], 'drill': c['drill_id'], 'optional': False, 'fields': fields})
            chosen[f['id']].append(c['drill_id'])
            used.append(c['drill_id'])
    objective = None
    if not ranked:
        objective = _objective(findings, profile, next_objective)
        if objective is not None:
            cands, earlier = ranked_candidates(lib, profile, objective['detector'])
            fields = None
            if cands:
                fields = _objective_fields(cands[0], objective, profile, reg, E, findings, range, earlier,
                                           label=objective_label(objective, next_objective))
                if fields is not None:
                    picks.append({'finding': None, 'drill': cands[0]['drill_id'], 'optional': True,
                                  'fields': fields})
            if fields is None:      # fix round 3: why the objective has no drill (finding None)
                skipped.append({'finding': None, 'reason': 'text_failed_validation' if cands else
                                no_drill_detail(lib, profile, objective['detector'])})
    for f in findings:
        f['drills'] = list(chosen.get(f['id'], []))
    return {'picks': picks, 'skipped': skipped, 'objective': objective}


def _score(c):
    return 2*c['relationMatch']+c['skillMatch']


def _covers(c, detector, failed):
    return {s['relation'] for s in c['drill']['success_signals_next_match']
            if s['detector'] == detector and s['relation'] in failed}


def ranked_candidates(library, profile, detector, failed_relations=()):
    """(candidates, earlier): candidates() at the profile's level; when the safety filter or the format filter leaves
    none there (a level above the age's typical level: e.g. U11 at level 4 has no allowed level-4 drill for D2, D3,
    D4; a small format: a 4v4 team at level 3 has no level-3 shape drill for its format), or the level has no drill
    of the check at all, the candidates of the nearest lower level that has any, down to fallback_floor (fix round
    3), with earlier = why the level had none (level_gap: 'age', 'format' or 'age_and_format'; 'none_at_level' when
    the level offers no drill of the check; truthy), else False. Neither filter is ever relaxed."""
    cands = candidates(library, profile, detector, failed_relations)
    if cands:
        return cands, False
    for lower in range(int(profile['level'])-1, fallback_floor(profile)-1, -1):
        cands = candidates(library, dict(profile, level=lower), detector, failed_relations)
        if cands:
            return cands, level_gap(library, profile, detector) or 'none_at_level'
    return [], False


def fallback_floor(profile):
    """The lowest level a fallback drill may come from (fix round 3, coach F-6): the level just below the profile's
    (MAX_LEVELS_BELOW), or the typical level of the team's age when that is lower (a team younger than its level's
    typical ages keeps age-fitting drills: U8 at level 4 -> level 1); never below 1. Adult (typical level 6) at level
    7 -> 6, at level 4 -> 3; U14 (typical 4) at level 6 -> 4."""
    from . import profiles as PR
    floor = int(profile['level'])-MAX_LEVELS_BELOW
    typical = PR.typical_level(profile['ageGroup'])
    if isinstance(typical, int) and not isinstance(typical, bool):
        floor = min(floor, typical)
    return max(1, floor)


def reachable_below(library, profile, detector):
    """The lower levels (nearest first) with a drill of `detector` passing both filters at this profile's age and
    format, without fallback_floor's limit (no_drill_detail tells 'only_far_below' with it)."""
    return [lv for lv in range(int(profile['level'])-1, 0, -1)
            if candidates(library, dict(profile, level=lv), detector)]


def level_gap(library, profile, detector):
    """Why the profile's level offers no drill of `detector` (None when it offers one, or has none at all): 'age'
    (the safety filter refuses every one at this age), 'format' (the age allows some, the format fits none) or
    'age_and_format' (each passes one filter, none both)."""
    offered = [d for d in library['drills'] if detector in d['related_tactical_detector'] and
               profile['level'] in d['development_level']]
    if not offered:
        return None
    safe = [d for d in offered if not safety_problems(d, profile['ageGroup'], profile.get('safety'))]
    fits = [d for d in offered if not format_problems(d, profile)]
    if any(d in fits for d in safe):
        return None
    return 'age' if not safe else ('format' if not fits else 'age_and_format')


def no_drill_detail(library, profile, detector):
    """Why a work item of `detector` got no drill (ranked_candidates found none down to fallback_floor), as
    selection's 'skipped' reason. Each code is true of the whole library at this level and below (fix round 3: with
    the fallback limited, the old 'refused_by_safety' could be said while a drill further down was allowed):
    - 'only_far_below': a drill passing both filters exists, but only below fallback_floor;
    - 'refused_by_safety': drills of the check exist at the level or below, and the safety filter refuses every one
      at this age;
    - 'no_drill_for_format': some pass the safety filter, and no drill at the level or below fits the match format;
    - 'no_drill_at_level': the level and those below have no drill of the check, or each drill fails one filter."""
    level = int(profile['level'])
    reach = reachable_below(library, profile, detector)
    if reach and max(reach) < fallback_floor(profile):
        return 'only_far_below'
    offered = [d for d in library['drills'] if detector in d['related_tactical_detector'] and
               min(d['development_level']) <= level]
    if not offered:
        return 'no_drill_at_level'
    if all(safety_problems(d, profile['ageGroup'], profile.get('safety')) for d in offered):
        return 'refused_by_safety'
    if all(format_problems(d, profile) for d in offered):
        return 'no_drill_for_format'
    return 'no_drill_at_level'


def no_drill_reason(library, profile, detector):
    """Why a work item of `detector` got no drill, in the codes report.NO_DRILL_WORDS words: 'refused_by_safety' or
    'no_drill_at_level' ("No drill in the library fits ... at this level", also said when no drill fits the match
    format or when the nearest fitting drill is more than one level down; no_drill_detail tells them apart:
    'no_drill_for_format', 'only_far_below')."""
    why = no_drill_detail(library, profile, detector)
    return 'no_drill_at_level' if why in ('no_drill_for_format', 'only_far_below') else why


def _num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and x == x and abs(x) != float('inf')


def _claimed(claims):
    """(relations claimed absent, relations claimed present) of relation / team_state claims, without the praise-first
    'demonstrated' claims of levels 1-2."""
    absent, present = set(), set()
    for c in claims:
        fa = c.get('facts') or {}
        if c.get('type') in ('relation', 'team_state') and fa.get('scope') != 'demonstrated' and fa.get('relation'):
            if fa.get('present') is False:
                absent.add(fa['relation'])
            elif fa.get('present') is True:
                present.add(fa['relation'])
    return absent, present


def _skill_name(profile, skill):
    name = (profile.get('skills') or {}).get(skill)
    return name if isinstance(name, str) and name else None


def _recurrence(claims):
    return next((c['facts'] for c in claims if c.get('type') == 'recurrence'), None)


def _situation(claims):
    return next((c['facts'] for c in claims if c.get('type') == 'situation'), None) or {}


def _is_short(claims, range_, E, finding=None):
    """Is the report range under 3 minutes (no repeated-pattern words)? The range when given; else the finding's
    recurrence claim (explain's range or obs segment) or assess's 'short_range' limitation (a short range inside a
    long segment, as explain.short_for); else the cautious side."""
    if isinstance(range_, dict):
        if _num(range_.get('seconds')):
            return float(range_['seconds']) < E.SHORT_RANGE_S
        if _num(range_.get('t0')) and _num(range_.get('t1')):
            return float(range_['t1'])-float(range_['t0']) < E.SHORT_RANGE_S
    if 'short_range' in ((finding or {}).get('limitations') or ()):
        return True
    rec = _recurrence(claims) or {}
    if isinstance(rec.get('short'), bool):
        return rec['short']
    return True                     # unknown range: the cautious side (as explain)


def _gate(text, claims, finding, profile, short, E, checks=None):
    """[] when explain's gate lets the text be shown as a finding's explanation at this level."""
    if not text or len(text) > max(FIELD_LIMITS.values()):
        return ['empty or too long']
    return E.validate({'headline': GATE_FILLER, 'explanation': text, 'short': GATE_FILLER, 'action': None},
                      claims, finding, profile, short=short, checks=checks)


def _first_passing(plans, limit, claims, finding, profile, short, E, checks=None):
    for text in plans:
        if text and len(text) <= limit and not _gate(text, claims, finding, profile, short, E, checks):
            return text
    return None


def _lines(lines, limit, claims, finding, profile, short, E):
    """Up to MAX_LINES library lines that pass the gate (LIBRARY_CHECKS), joined, within limit."""
    ok = [x for x in lines if not _gate(x, claims, finding, profile, short, E, LIBRARY_CHECKS)]
    out = ''
    for x in ok[:MAX_LINES]:
        nxt = (out+' '+x).strip()
        if len(nxt) > limit:
            break
        out = nxt
    return out or None


def _signal(drill, detector, relations):
    sigs = [s for s in drill['success_signals_next_match'] if s['detector'] == detector]
    for rel in relations:
        for s in sigs:
            if s['relation'] == rel:
                return s
    return sigs[0] if sigs else drill['success_signals_next_match'][0]


def _noun(relation, level):
    v = NOUN.get(relation)
    if isinstance(v, tuple):
        return v[1] if level >= NOUN_LEVEL.get(relation, 99) else v[0]
    return v


def _phase_clause(sit, E):
    state, source = sit.get('phase'), sit.get('source')
    qual = E.QUALIFIER.get(source)
    if state not in ('own', 'opp') or qual is None:
        return None, None
    who = 'we' if state == 'own' else 'the other team'
    poss = 'our possession' if state == 'own' else "the other team's possession"
    return f'when {who} had the ball {qual}', f'{poss} {qual}'


def _behind_phrase(level, required, family):
    if level <= 2:
        phrase = 'nobody was ready to protect our goal'
    elif not _num(required) or required <= 1:
        phrase = 'nobody stayed back'
    else:
        phrase = 'there were fewer players behind the ball than this level asks for'
    return ('a defender moved forward and '+phrase) if family == 'advance' else phrase


def _observed_phrase(finding, failed, level, claims):
    """(relation, phrase) of the first failed relation whose observed state the claims state (OBSERVED)."""
    absent, present = _claimed(claims)
    family = str(finding.get('key') or '').split('|')[2] if str(finding.get('key') or '').count('|') >= 4 else None
    for rel in failed:
        for kind, polarity, first_level, phrase in OBSERVED.get(rel, ()):
            if level < first_level or kind not in (present if polarity else absent):
                continue
            if phrase is None:
                req = next((c['facts'].get('required') for c in claims if c.get('type') in ('relation', 'team_state')
                            and (c.get('facts') or {}).get('relation') == 'behind_ball'), None)
                phrase = _behind_phrase(level, req, family)
            return rel, phrase
    return None, None


def _counts(rec, reg, poss_phrase, E, failing=True):
    """' (about N%)' share and ' (S separate spells, about R per 10 minutes of ...)' pieces by the number register,
    on explain's rules for the same numbers (fix round 3, coach F-11: 60 drill lines printed "(about 100%)" and 51 a
    rate like "about 42.5 per 10 minutes" from 3.8 minutes of play):
    - the share only strictly inside (1 - SHARE_ALL, SHARE_ALL) ("16 of the 16" says it), and only when it is the
      share of the two counts the sentence states (a share over other moments would contradict them);
    - the spells only from 2 (explain's rule; "1 separate spells" is no pattern);
    - the rate only from RATE_MIN_PHASE_S of the phase (at most a twofold extrapolation), a whole number below
      RATE_DECIMAL_PHASE_S."""
    share = rec.get('failShare' if failing else 'demonstratedShare')
    part, whole = rec.get('failing' if failing else 'demonstrated'), rec.get('opportunities')
    mode = reg['numbers']
    top = float(getattr(E, 'SHARE_ALL', .995))
    pct = ''
    if mode in ('shares', 'rates') and _num(share) and _num(part) and _num(whole) and whole > 0 and \
            abs(float(share)-float(part)/float(whole)) <= SHARE_TOLERANCE and 1.0-top < float(share) < top:
        pct = f' (about {int(round(100*float(share)))}%)'
    spells = rec.get('failingSpells')
    rate = rec.get('ratePer10')
    phase_s = rec.get('phaseSeconds')
    tail = ''
    if failing and _num(spells) and spells >= 2:
        tail = f'{int(spells)} separate spells'
        lo = float(getattr(E, 'RATE_MIN_PHASE_S', 300.0))
        dec = float(getattr(E, 'RATE_DECIMAL_PHASE_S', 600.0))
        if mode == 'rates' and _num(rate) and poss_phrase and _num(phase_s) and phase_s >= lo-1e-9:
            r = f'{float(rate):.1f}' if phase_s >= dec-1e-9 else f'{int(round(float(rate)))}'
            tail += f', about {r} per {int(rec.get("perMinutes") or 10)} minutes of {poss_phrase}'
        tail = f' ({tail})'
    return pct, tail


def _format_note(cand, profile):
    """The 'trains' note when the team's format is not one the drill is recommended for. A drill set up for more
    players than the format never gets here (format_problems refuses it before ranking)."""
    if cand['formatMatch']:
        return ''
    team = team_players(profile)
    sizes = [FORMAT_PLAYERS[f] for f in cand['drill']['recommended_format']]
    if max(sizes) < team:
        return ' It is played as a smaller game than your match format.'
    return ' Fit the pitch to your match format.'


def _trains(cand, profile, claims, finding, short, E):
    d = cand['drill']
    main = _skill_name(profile, d['primary_skill']) or d['primary_skill'].replace('_', ' ').capitalize()
    # a secondary skill of a check the format cannot assess is not named (a 4v4 team is never told the game also
    # trains support behind advancing wide defenders)
    also = [n for n in (_skill_name(profile, s) for s in d['secondary_skills']
                        if format_supports(profile, S.SKILLS.get(s))) if n]
    base = f'Main skill: {main}.'
    full = base+(f' Also: {"; ".join(also)}.' if also else '')
    return _first_passing([full+_format_note(cand, profile), full, base], FIELD_LIMITS['trains'], claims, finding,
                          profile, short, E)


def _validation_claims(claims, drill, relations, signal):
    """The finding's claims plus what this pick adds: a drill_link claim (drill id, failed relations; no number, so
    every number shown is a fact of the finding itself) and an action claim with the success signal's relation (the
    relation the drill aims to restore, so a hypothetical 'what you can look for' sentence may state it)."""
    return list(claims)+[{'type': 'drill_link', 'facts': {'drill': drill['drill_id'], 'relations': list(relations)}},
                         {'type': 'action', 'facts': {'principle': None, 'relation': signal['relation'],
                                                      'mode': 'statement'}}]


def _pick_fields(finding, cand, failed, profile, reg, E, priority, second, range_, earlier=False):
    d = cand['drill']
    det = finding['observation']
    level = reg['level']
    claims = finding.get('claims') or E.build_claims(finding, profile, register=reg)
    short = _is_short(claims, range_, E, finding)
    covered = [r for r in failed if r in _covers(cand, det, failed)]
    sig = _signal(d, det, covered or failed)
    vclaims = _validation_claims(claims, d, failed, sig)
    skill = SKILL_OF.get(det)
    sname = _skill_name(profile, skill) or _skill_name(profile, d['primary_skill']) or \
        (skill or d['primary_skill']).replace('_', ' ').capitalize()
    numbers = reg['numbers'] != 'none'
    # no past-tense verb here: a skill name ("Someone ready to protect our goal") is a goal, not an observation, and
    # the gate reads a past-tense sentence for relation polarity. The rank is a word ("the second priority"): a digit
    # would have to be a fact of the finding wherever the text is checked again (report.drill_problems)
    rank = ORDINAL.get(priority)
    lead = ('A second drill for ' if second else 'This drill is for ')+(f'the {rank} priority' if numbers and rank
                                                                       else "this match's focus")
    s1 = f'{lead}: {sname}.'
    rec = _recurrence(claims) or {}
    phase, poss = _phase_clause(_situation(claims), E)
    _, phrase = _observed_phrase(finding, covered+[r for r in failed if r not in covered], level, claims)
    where = f' {phase}' if phase else ''
    s2 = s2_plain = None
    if numbers and _num(rec.get('failing')) and _num(rec.get('opportunities')) and rec['opportunities'] > 0:
        pct, tail = _counts(rec, reg, poss, E)
        head = f'In {int(rec["failing"])} of the {int(rec["opportunities"])} situations{pct} PitchIQ measured{where}'
        if phrase:
            s2 = f'{head}, {phrase}{tail}.'
        s2_plain = f'{head}, this was not demonstrated{tail}.'
    elif phrase:
        s2 = f'In some situations{where}, {phrase}.'
    if cand['relationMatch']:
        target = next(iter(covered), None)
        s3 = f'Its game works directly on {_noun(target, level)}.' if target else ''
    elif cand['skillMatch']:
        s3 = 'No drill at this level works on that part directly; this one trains the same skill.'
    else:
        s3 = 'It is the closest drill this level offers for this skill.'
    if earlier:
        s3 = (s3+' ' if s3 else '')+EARLIER_NOTES.get(earlier, EARLIER_NOTE)
    why = _first_passing([' '.join(x for x in (s1, s2, s3) if x), ' '.join(x for x in (s1, s2_plain, s3) if x),
                          ' '.join(x for x in (s1, s3) if x), s1], FIELD_LIMITS['why'], vclaims, finding, profile,
                         short, E)
    trains = _trains(cand, profile, vclaims, finding, short, E)
    easier = _lines(d['regression'], FIELD_LIMITS['easier'], vclaims, finding, profile, short, E)
    harder = _lines(d['progression'], FIELD_LIMITS['harder'], vclaims, finding, profile, short, E)
    progress = _progress(sig, rec, reg, vclaims, finding, profile, short, E)
    if None in (why, trains, easier, harder, progress):
        return None
    moments = list(dict.fromkeys(ts['evidence'] for ts in finding.get('timestamps') or ()))
    return {'trains': trains, 'why': why, 'moments': moments, 'easier': easier, 'harder': harder,
            'progress': progress}


def neutral_signal(text):
    """A success signal without its comparison with this match (COMPARE): what to look for next match when this
    match did not judge the check. The validator makes every signal end with COMPARE."""
    return text[:-len(COMPARE)]+'.' if text.endswith(COMPARE) else text


def _progress(sig, rec, reg, claims, finding, profile, short, E, compare=True):
    """'progress': the success signal as what to look for next match, compared with this match only when this match
    judged the check (compare; fix round 3, coach F-8 = facts F-10), then this match's counts when the level shows
    numbers."""
    text = sig['text'] if compare else neutral_signal(sig['text'])
    head = f'What you can look for next match: {text[:1].lower()}{text[1:]}'
    later = ' PitchIQ will look at the same kind of situations then.'
    plans = []
    n, dem = rec.get('opportunities'), rec.get('demonstrated')
    if compare and reg['numbers'] != 'none' and _num(n) and _num(dem) and n > 0:
        n, dem = int(n), int(dem)
        if dem == 0:
            plans.append(f'{head} In this match none of the {n} situations PitchIQ measured showed it, so any of '
                         'them next match would be progress.')
        else:
            pct, _ = _counts(rec, reg, None, E, failing=False)
            plans.append(f'{head} In this match {dem} of the {n} situations PitchIQ measured showed it{pct}, so more '
                         f'than {dem} of every {n} next match would be progress.')
    plans += [head+later, head, 'PitchIQ will look at the same kind of situations in the next match.']
    return _first_passing(plans, FIELD_LIMITS['progress'], claims, finding, profile, short, E)


def decided_situations(findings, detector):
    """The situations of `detector` this match decided (demonstrated + not demonstrated) in the check's own phase,
    summed over the reported findings of the check whose aggregate counts both and whose phase is the one its success
    signals speak of (S.DETECTOR_PHASES[detector][0]: "when we have the ball" / "when the other team has the ball");
    0 when no reported finding counts any. The optional drill compares its success signal with this match only from
    1: a check with only unclear moments, or none reported, was not judged here (facts F-10: BC60 A at level 4 had 0
    decided pressure-and-cover situations), and phase-unknown shape moments (the amateur clip's attacking-shape
    describes) are no moments "when we have the ball" (facts F-9)."""
    phase = S.DETECTOR_PHASES.get(detector, (None,))[0]
    n = 0
    for f in findings:
        a = f.get('aggregate') or {}
        state = (((f.get('context') or {}).get('phase')) or {}).get('state')
        if f.get('observation') == detector and state == phase and _num(a.get('demonstrated')) and \
                _num(a.get('notDemonstrated')):
            n += int(a['demonstrated'])+int(a['notDemonstrated'])
    return n


def demonstrated_checks(findings, profile=None):
    """The checks this match demonstrated, as its reported findings show it (coach verifier M5: praise-first, a
    team is not sent to work on what it already did): a check with a reported strength, or a check whose reported
    findings together count at least one demonstrated situation and none not demonstrated (their aggregates; e.g. a
    check the level describes only). It is assess's rule (assess._next_objective, DETECT fix round 1: every
    classified opportunity of the check in the range, demonstrated > 0 and not demonstrated == 0) on what this stage
    sees: a check with no reported finding is invisible here (level_objective's `demonstrated` carries it). A check
    with a criticism or a review counts a failed situation, so it is never 'demonstrated'; a finding without counts
    makes its check unknown (not demonstrated). profile: unused, kept for callers that pass it."""
    strong, counts = set(), {}
    for f in findings:
        det = f.get('observation')
        if f.get('verdict') == 'strength':
            strong.add(det)
        a = f.get('aggregate') or {}
        d, nd = a.get('demonstrated'), a.get('notDemonstrated')
        got = counts.setdefault(det, [0, 0, True])
        if _num(d) and _num(nd) and f.get('verdict') not in S.CRITICISM+('review',):
            got[0] += int(d)
            got[1] += int(nd)
        else:
            got[2] = False
    return strong | {det for det, (d, nd, known) in counts.items() if known and d >= 1 and nd == 0}


def objective_exclusion(profile, detector, demonstrated=()):
    """Why a check cannot be the level's next objective (None = it can): 'off' (not enabled at the level),
    'no_skill_name' (the level names no skill for it), 'format_not_supported' (the match format cannot assess it:
    no wide-defender objective at 4v4 / 5v5) or 'demonstrated' (this match demonstrated it: coach verifier M5)."""
    dp = (profile.get('detectors') or {}).get(detector) or {}
    if not dp.get('enabled'):
        return 'off'
    if not _skill_name(profile, SKILL_OF.get(detector)):
        return 'no_skill_name'
    if not format_supports(profile, detector):
        return 'format_not_supported'
    if detector in set(demonstrated or ()):
        return 'demonstrated'
    return None


def level_objective(findings, profile, demonstrated=None):
    """{'skill', 'detector'} of the next development objective taken from the level when no work item is reported
    (DESIGN 4.8 #7, assess's rule, with fix round 1's exclusions): the check with the lowest objectiveRank (then the
    detector order) that objective_exclusion lets through, the demonstrated checks being demonstrated_checks(findings,
    profile) plus `demonstrated` (checks the caller knows this match demonstrated at the level's strength bar though no
    finding reports them: assess sees every classified family, this stage only the reported findings). None when no
    check qualifies. Public so that assess._next_objective and report.next_objective can name the same objective as
    the optional drill (INTERFACES.md, DRILLS fix round 1). Measured numbers: the module docstring."""
    dem = demonstrated_checks(findings, profile) | set(demonstrated or ())
    cands = []
    for det, dp in (profile.get('detectors') or {}).items():
        if det in SKILL_OF and objective_exclusion(profile, det, dem) is None:
            cands.append(((dp.get('params') or {}).get('objectiveRank') or 99, S.DETECTORS.index(det), det))
    if not cands:
        return None
    det = min(cands)[2]
    return {'skill': SKILL_OF[det], 'detector': det}


def _objective(findings, profile, next_objective):
    """{'skill', 'detector'} of the next objective when no work item is reported: assess's NextObjective when it does
    not rest on a finding and objective_exclusion lets it through; None when assess chose none (skill None: every
    check of the level was demonstrated or excluded, which assess sees and this stage cannot: on Metrica game 1, 12
    of 434 reports); else level_objective."""
    if isinstance(next_objective, dict) and not next_objective.get('fromMatch'):
        if next_objective.get('skill') is None:
            return None
        if next_objective.get('skill') in S.SKILLS:
            det = S.SKILLS[next_objective['skill']]
            if objective_exclusion(profile, det, demonstrated_checks(findings, profile)) is None:
                return {'skill': next_objective['skill'], 'detector': det}
    return level_objective(findings, profile)


def objective_label(objective, next_objective):
    """The optional drill's label: MATCH_OBJECTIVE_LABEL when its objective is assess's NextObjective chosen from this
    match's checks (not from a work item, not the level's suggested focus, the same skill), else NEXT_OBJECTIVE_LABEL
    (the level's suggested focus, or level_objective when assess gave none this stage can use)."""
    from . import assess as AS
    no = next_objective if isinstance(next_objective, dict) else {}
    if objective and no.get('skill') is not None and no.get('skill') == objective.get('skill') and \
            not no.get('fromMatch') and not AS.is_level_focus(no):
        return MATCH_OBJECTIVE_LABEL
    return NEXT_OBJECTIVE_LABEL


def _objective_fields(cand, objective, profile, reg, E, findings, range_, earlier=False,
                      label=NEXT_OBJECTIVE_LABEL):
    """The fields of the optional next-objective drill: nothing measured is quoted (no finding, no moments); its
    success signal compares with this match only when a reported finding of the check counts a decided situation
    (decided_situations). label: objective_label's, in the 'why' line."""
    d = cand['drill']
    det = objective['detector']
    finding = {'id': None, 'key': None, 'observation': det, 'verdict': 'try_this', 'guardrails': {}}
    sig = _signal(d, det, ())
    claims = _validation_claims([], d, (), sig)
    rec_claims = next((f.get('claims') for f in findings if _recurrence(f.get('claims') or ())), ())
    short = _is_short(rec_claims or (), range_, E)
    sname = _skill_name(profile, objective['skill'])
    note = ' '+EARLIER_NOTES.get(earlier, EARLIER_NOTE) if earlier else ''
    why = _first_passing([f'This report shows no work item at this level. This drill is {label}: '
                          f'{sname}.{note}', f'This drill is {label}: {sname}.{note}'],
                         FIELD_LIMITS['why'], claims, finding, profile, short, E)
    trains = _trains(cand, profile, claims, finding, short, E)
    easier = _lines(d['regression'], FIELD_LIMITS['easier'], claims, finding, profile, short, E)
    harder = _lines(d['progression'], FIELD_LIMITS['harder'], claims, finding, profile, short, E)
    progress = _progress(sig, {}, reg, claims, finding, profile, short, E,
                         compare=decided_situations(findings, det) >= 1)
    if None in (why, trains, easier, harder, progress):
        return None
    return {'trains': trains, 'why': why, 'moments': [], 'easier': easier, 'harder': harder, 'progress': progress}


# ---------- helpers ----------
def get(library, drill_id):
    """The v2 drill with that id, or None."""
    for d in library.get('drills') or ():
        if d.get('drill_id') == drill_id:
            return d
    return None


def game_sizes(text):
    """Player totals of every 'N v M' game in a text ('3v2' -> 5, '11 v 7' -> 18)."""
    return [int(a)+int(b) for a, b in GAME.findall(text or '')]


def game_headcounts(text):
    """(outfield total, head count) of every 'N v M' game in a text: the head count adds two goalkeepers when the
    game's clause (up to the next ';' or '.') says "with goalkeepers" or "plus goalkeepers" ('6 v 6 plus goalkeepers'
    -> (12, 14); 'with or without goalkeepers' and 'plus a goalkeeper if available' add none). player_count counts
    every player a drill uses, goalkeepers included (sources verifier F9)."""
    text = text or ''
    out = []
    for m in GAME.finditer(text):
        n = int(m.group(1))+int(m.group(2))
        out.append((n, n+2 if KEEPERS_CLAUSE.search(text[m.end():]) else n))
    return out


def team_players(profile):
    """Players a side of the profile's match format (the keeper included; 11 without a format)."""
    fmt = profile.get('format') or S.DEFAULT_FORMAT
    name = fmt.get('name') or S.format_name(fmt)
    return FORMAT_PLAYERS.get(name, int(fmt.get('players') or 11))


def format_problems(drill, profile):
    """Why a drill does not fit the team's match format (codes; [] = it fits). A hard filter, like the safety
    filter, run before any ranking and never relaxed:
    - 'above_match_format': every format the drill is recommended for has more players a side than the team's
      (no 9v9 drill for a 4v4 team, no 11v11 drill for a 7v7 team); a smaller game is fine (a 2v2 for an 11v11
      team; 'trains' says it is played as a smaller game). A 5v5 team (4 outfield, DESIGN 2.2) fits 4v4 drills;
    - 'more_players_than_format': the drill needs more players (player_count.min, goalkeepers counted) than one game
      of the team's own format has (2 x players a side: 8 at 4v4, 10 at 5v5, 14 at 7v7, 18 at 9v9, 22 at 11v11)
      [P]: the settings carry no squad size, and a squad that plays NvN is not assumed to bring more than two teams
      of N to training."""
    n = team_players(profile)
    out = []
    if min(FORMAT_PLAYERS[f] for f in drill['recommended_format']) > n:
        out.append('above_match_format')
    if drill['player_count']['min'] > 2*n:
        out.append('more_players_than_format')
    return out


def format_supports(profile, detector):
    """False when the team's match format cannot assess the check at all (FORMAT_UNSUPPORTED: D3 and D5 at 4v4 /
    5v5): such a check is never the next objective (it can have no finding either)."""
    return team_players(profile) not in FORMAT_UNSUPPORTED.get(detector, ())


def quoted_numbers(d):
    """The numbers a report may quote from a v2 drill (TEXT adds them to its drill_link facts): the numbers in its
    name, its player range, its duration and its field size."""
    out = [float(x) for x in re.findall(r'\d+', d['name'])]
    out += [float(d['player_count']['min']), float(d['player_count']['max']), float(d['duration']['minutes'])]
    for side in ('length', 'width'):
        v = d['field_setup'][side]
        out.append(float(v['value'] if isinstance(v, dict) else v))
    return sorted(set(out))


def instruction_texts(d):
    """(path, text) of the text a coach reads as the drill (everything except the source credit)."""
    out = [('name', d['name']), ('player_count.note', d['player_count']['note']),
           ('field_setup.notes', d['field_setup']['notes']), ('duration.blocks', d['duration']['blocks'])]
    for f in ('equipment', 'instructions', 'coaching_points', 'progression', 'regression'):
        out += [(f'{f}[{k}]', t) for k, t in enumerate(d[f])]
    out += [(f'success_signals_next_match[{k}].text', s['text'])
            for k, s in enumerate(d['success_signals_next_match'])]
    return out


def drill_texts(d):
    """(path, text) of every string a drill shows: its instructions and its source notes."""
    return instruction_texts(d)+[(f'source_reference[{k}].note', r['note'])
                                 for k, r in enumerate(d['source_reference'])]


def library_sha(library):
    """The first 16 hex of the library file's sha256, 'inline' for a library that did not come from a file (v1 and
    v2)."""
    return library.get('_sha', 'inline')


# ================================ v1 (deprecated) ================================
# The v1 engine (build_observations without coaching=..., report.build_report) still reads the v1 library until it
# retires (DESIGN 11.2, wave 7). Behaviour unchanged; only the file moved to drills_v1.json.
LEVELS = ('development', 'competitive', 'advanced')   # the v1 report's context levels (language.LEVELS)
TEXT_FIELDS = ('name', 'purpose', 'tacticalIssue', 'area', 'setup', 'source')
LIST_FIELDS = ('instructions', 'coachingPoints', 'safety', 'equipment', 'regressions', 'successCriteria')   # non-empty
REPORT_LINK = 'coaching report'   # a success criterion names the report measurement it should move
VERSUS = re.compile(r'(\d+) v (\d+)')
OPTIONAL_LIST_FIELDS = ('progressions', 'patterns')
MAX_PLAYERS = 30            # a session for one squad; anything larger is a typo
MIN_MINUTES, MAX_MINUTES = 5, 45   # one exercise, not a whole session
ID_RE = re.compile(r'[a-z][a-z0-9_]{2,40}')
PRINTABLE = re.compile(r'[ -~]*')   # printable ASCII: a strict subset of WinAnsi


def load_drills(path=None):
    """The validated v1 library {'version', 'notes', 'drills': [...], '_sha'} (cached per file and modification
    time). Raises ValueError listing every problem when the file does not validate."""
    p = Path(path) if path is not None else V1_LIBRARY_PATH
    st = p.stat()
    key = (str(p.resolve()), st.st_mtime_ns, st.st_size)
    hit = _cache.get('v1')
    if hit is None or hit[0] != key:
        data = p.read_bytes()
        library = json.loads(data.decode('utf-8'))
        problems = validate_drills(library)
        if problems:
            raise ValueError('drill library invalid: '+'; '.join(problems))
        library['_sha'] = hashlib.sha256(data).hexdigest()[:16]
        _cache['v1'] = (key, library)
    return _cache['v1'][1]


def validate_drills(library, known_keys=None):
    """Every problem with a v1 library, as readable strings (empty when it is valid). known_keys defaults to the
    recommendation keys language.py has templates for (not checked when language.py no longer lists them)."""
    if known_keys is None:
        from soccer.coaching import language
        known_keys = getattr(language, 'RECOMMENDATION_KEYS', None)
    problems = []
    if not isinstance(library, dict):
        return ['the library is not an object']
    if not isinstance(library.get('version'), int):
        problems.append('version must be an integer')
    drills = library.get('drills')
    if not isinstance(drills, list) or not drills:
        return problems+['drills must be a non-empty list']
    seen = set()
    for i, d in enumerate(drills):
        where = f'drill {i}'
        if not isinstance(d, dict):
            problems.append(f'{where} is not an object')
            continue
        did = d.get('id')
        if not isinstance(did, str) or not ID_RE.fullmatch(did):
            problems.append(f'{where}: id must be lower_snake_case')
        else:
            where = f'drill {did}'
            if did in seen:
                problems.append(f'{where}: duplicate id')
            seen.add(did)
        for f in TEXT_FIELDS:
            if not _sentence(d.get(f)):
                problems.append(f'{where}: {f} must be non-empty text')
        for f in ('name', 'purpose'):
            if isinstance(d.get(f), str) and re.search(r'\d', d[f]):
                problems.append(f'{where}: {f} must not contain digits')
        for f in LIST_FIELDS:
            v = d.get(f)
            if not isinstance(v, list) or not v or not all(_sentence(s) for s in v):
                problems.append(f'{where}: {f} must be a non-empty list of text')
        for f in OPTIONAL_LIST_FIELDS:
            v = d.get(f)
            if v is not None and (not isinstance(v, list) or not all(_sentence(s) for s in v)):
                problems.append(f'{where}: {f} must be a list of text')
        keys = d.get('recommendationKeys')
        if not isinstance(keys, list) or not keys or not all(isinstance(k, str) for k in keys):
            problems.append(f'{where}: recommendationKeys must be a non-empty list')
        elif known_keys is not None:
            unknown = sorted(set(keys)-set(known_keys))
            if unknown:
                problems.append(f'{where}: unknown recommendationKeys {unknown}')
        problems += _range(d.get('players'), 1, MAX_PLAYERS, f'{where}: players')
        problems += _range(d.get('durationMinutes'), MIN_MINUTES, MAX_MINUTES, f'{where}: durationMinutes')
        var = d.get('variations')
        if not isinstance(var, dict) or any(not _sentence(var.get(lv)) for lv in LEVELS):
            problems.append(f'{where}: variations needs text for each of {", ".join(LEVELS)}')
        crit = d.get('successCriteria')
        if isinstance(crit, list) and crit and not any(isinstance(c, str) and REPORT_LINK in c for c in crit):
            problems.append(f'{where}: a success criterion must name the {REPORT_LINK} measurement it should move')
        pl = d.get('players')
        if isinstance(pl, dict) and isinstance(pl.get('min'), int) and isinstance(pl.get('max'), int):
            texts = list((var or {}).values()) if isinstance(var, dict) else []
            texts.append(pl.get('note') or '')
            for n in versus_counts(' '.join(t for t in texts if isinstance(t, str))):
                if not pl['min'] <= n <= pl['max']:
                    problems.append(f'{where}: a {n}-player game does not fit players {pl["min"]}-{pl["max"]}')
        for path, text in _strings(d):
            if not PRINTABLE.fullmatch(text):
                bad = sorted({c for c in text if not PRINTABLE.fullmatch(c)})
                problems.append(f'{where}: {path} has characters outside printable ASCII {bad!r}')
    return problems


def versus_counts(text):
    """The player totals of every 'N v M' in a text (4 v 2 -> 6); v1 spelling with spaces."""
    return [int(a)+int(b) for a, b in VERSUS.findall(text or '')]


def drill(library, drill_id):
    """The v1 drill with that id, or None."""
    for d in library.get('drills') or ():
        if d.get('id') == drill_id:
            return d
    return None


def drills_for(library, recommendation_key, pattern=None):
    """The v1 drills written for a recommendation key, in library order; those that also name the pattern first."""
    found = [d for d in library.get('drills') or () if recommendation_key in (d.get('recommendationKeys') or ())]
    if pattern:
        found.sort(key=lambda d: pattern not in (d.get('patterns') or ()))
    return found


def choose(library, observation):
    """The v1 drill for an observation: the first of its drillIds that exists and is written for its recommendation
    key, else the library's first drill for that key (pattern first). None when nothing fits."""
    key = observation.get('recommendationKey')
    if not key:
        return None
    for did in observation.get('drillIds') or ():
        d = drill(library, did)
        if d is not None and key in (d.get('recommendationKeys') or ()):
            return d
    found = drills_for(library, key, observation.get('pattern'))
    return found[0] if found else None


def drill_ref(d, level='competitive'):
    """The v1 drill as a report carries it: the library record (without internal fields) with the variation for the
    report's context level as 'variation'."""
    out = {k: copy.deepcopy(v) for k, v in d.items() if not k.startswith('_') and k != 'variations'}
    out['variation'] = (d.get('variations') or {}).get(level)
    out['level'] = level
    return out


def all_text(library):
    """Every string in a v1 library with its path, for the ASCII and wording checks."""
    return list(_strings(library))


def drill_numbers(d):
    """The numbers a v1 finding may quote from a drill: its player count and duration bounds."""
    out = []
    for f in ('players', 'durationMinutes'):
        r = d.get(f) or {}
        out += [float(r[k]) for k in ('min', 'max') if isinstance(r.get(k), (int, float))]
    return out


def _sentence(x):
    return isinstance(x, str) and bool(x.strip())


def _range(r, lo, hi, where):
    if not isinstance(r, dict):
        return [f'{where} must be {{min, max}}']
    a, b = r.get('min'), r.get('max')
    if not (isinstance(a, int) and isinstance(b, int)) or isinstance(a, bool) or isinstance(b, bool):
        return [f'{where} min and max must be integers']
    if not lo <= a <= b <= hi:
        return [f'{where} must satisfy {lo} <= min <= max <= {hi}']
    if 'note' in r and not _sentence(r['note']):
        return [f'{where} note must be text']
    return []


def _strings(x, path=''):
    if isinstance(x, str):
        yield path, x
    elif isinstance(x, dict):
        for k, v in x.items():
            if not str(k).startswith('_'):
                yield from _strings(v, f'{path}.{k}' if path else str(k))
    elif isinstance(x, list):
        for i, v in enumerate(x):
            yield from _strings(v, f'{path}[{i}]')
