"""The data contract of the coaching v2 engine (DESIGN v2, sections 1.2-1.3, 2-10): typed records and strict JSON
validators for every file and record the roles exchange, so a wrong shape is caught where it is made and not three
modules later in the page or the PDF.

Why a contract module and strict validators. The v1 layer has no schema, and the critic of round 2
(coaching/verify2/reports/critic.md) and the audit found defects that are shape defects:
- a NaN written by Flask's jsonify broke the page's JSON.parse (store.py keeps a last-resort guard for it); here every
  number must be finite and every value plain JSON (a numpy integer is refused: json.dumps cannot write it);
- the renderers accept a 'suggested' element that no producer ever made (0 of 63 evidence records, 0 of 24 reports in
  the audit), so a whole feature was dead without any test failing; here every element type is declared, the new
  annotation elements have a budget and a tier table, and a v2 record may not carry the dormant 'suggested' element;
- v1 ids include ENGINE_VERSION, so a coach's verdict was lost on every engine update; v2 keys (opportunity_key,
  finding_key) and record_id leave the engine version out;
- all 8 team-runs of the audit gave identical findings at every level: the v2 records carry the level and the
  verdict, and the rules below refuse the level/verdict pairs the profile table forbids (no "work_on" at levels 1-2,
  no "try_this" from level 4, no criticism with a failed guardrail, no strength without a demonstration).
The record layouts are DESIGN's JSON examples made exact; where DESIGN shows "..." the record is an open map with typed
values (relations, measurements, claim facts, detector parameters, roleInputs).

How it is used. The TypedDicts are the single source of truth: the validator reads their annotations (Annotated
constraints, Literal enumerations, NotRequired keys) and walks a value against them, then applies the cross-field rules
registered per record type. Records are closed: an unknown key is a problem (a field the contract does not name must
be added here first; INTERFACES.md says how). validate(kind, value) returns readable problems ('finding.verdict: ...'),
check(kind, value) raises SchemaError, example(kind) returns a valid record to start tests and mocks from.

What the rules enforce from the DESIGN invariants (1.3): 3 (cross_check: every evidence id a report cites exists in
the observation or role file, at the same frame and seek), 4 (event_problems: no turnover, no "lost/won the ball", no
counter-attack or pass as an event, no communication, outside a hypothetical sentence), 5 (profile thresholds carry
origin PITCHIQ-PROPOSAL and validated false), 7 (seek = (frame + 0.5) / fps rounded to 4 decimals) and 8 (text that
reaches the PDF is printable ASCII, a strict subset of WinAnsi; numbers finite). Invariant 2 (observations do not
change with the profile) is structural: tier_settings() hands the measurement tier only the format and the field size.

Pure standard library: importing it costs no numpy and no scipy (the package import stays light).
"""
import copy
import hashlib
import json
import math
import re
import types
import typing
from typing import Annotated, Any, Literal, NotRequired, TypedDict, get_args, get_origin

SCHEMA_VERSION = 2          # coaching v2 records (a v1 report has version 1 and is validated by nothing here)
ENGINE_VERSION = '2'        # the v2 engine's version in EngineStamp; never part of a key or an id
MAX_PROBLEMS = 100          # problems listed per validation (the rest are counted)
SEEK_DECIMALS = 4           # seek = round((frame + 0.5) / fps, 4): evidence.build's convention since v1


# ---------- constraint markers (used inside Annotated[...]) ----------
class Range:
    """Inclusive numeric bounds."""

    def __init__(self, lo=None, hi=None):
        self.lo, self.hi = lo, hi

    def check(self, v, path, out):
        if self.lo is not None and v < self.lo:
            out.append(f'{path}: {v!r} is below {self.lo}')
        if self.hi is not None and v > self.hi:
            out.append(f'{path}: {v!r} is above {self.hi}')


class Len:
    """Inclusive length bounds of a string, list or object."""

    def __init__(self, lo=0, hi=None):
        self.lo, self.hi = lo, hi

    def check(self, v, path, out):
        n = len(v)
        if n < self.lo or (self.hi is not None and n > self.hi):
            out.append(f'{path}: length {n} outside {self.lo}..{self.hi if self.hi is not None else "any"}')


class Pattern:
    """A string matching the regular expression in full."""

    def __init__(self, regex):
        self.regex = re.compile(regex)

    def check(self, v, path, out):
        if self.regex.fullmatch(v) is None:
            out.append(f'{path}: {v!r} does not match {self.regex.pattern}')


class Text:
    """Text that may reach the PDF: printable ASCII (a strict subset of WinAnsi, base-14 Helvetica), 1..n characters."""
    PRINTABLE = re.compile(r'[ -~]+')

    def __init__(self, n):
        self.n = n

    def check(self, v, path, out):
        if not v or len(v) > self.n:
            out.append(f'{path}: text length {len(v)} outside 1..{self.n}')
        elif self.PRINTABLE.fullmatch(v) is None:
            bad = sorted({c for c in v if not (' ' <= c <= '~')})
            out.append(f'{path}: not printable ASCII ({", ".join(repr(c) for c in bad[:5])})')


class Unique:
    """A list without repeated items."""

    def check(self, v, path, out):
        seen = [json.dumps(x, sort_keys=True) for x in v]
        if len(set(seen)) != len(seen):
            out.append(f'{path}: repeated items')


class Increasing:
    """A list of numbers in increasing order (strictly when strict)."""

    def __init__(self, strict=False):
        self.strict = strict

    def check(self, v, path, out):
        for a, b in zip(v, v[1:]):
            if isinstance(a, (int, float)) and isinstance(b, (int, float)) and (b < a or (self.strict and b == a)):
                out.append(f'{path}: not in increasing order ({a} then {b})')
                return


# ---------- vocabularies ----------
TEAMS = ('A', 'B')
AGE_GROUPS = ('U6', 'U7', 'U8', 'U9', 'U10', 'U11', 'U12', 'U13', 'U14', 'U15', 'U16', 'U17', 'U18', 'U19', 'U21',
              'U23', 'Adult')
AGE_ORDER = {a: k for k, a in enumerate(AGE_GROUPS)}
LEVEL_NAMES = {1: 'FOUNDATION', 2: 'EARLY DEVELOPMENT', 3: 'DEVELOPING', 4: 'INTERMEDIATE', 5: 'ADVANCED YOUTH',
               6: 'COMPETITIVE ADULT / ACADEMY', 7: 'PROFESSIONAL / ELITE'}
FORMAT_PLAYERS = (4, 5, 7, 9, 11)       # 5 = the 5v5 alias (England U7-U8, Canada U8-U9), assessed with the 4v4 rules
FORMAT_NAMES = ('4v4', '5v5', '7v7', '9v9', '11v11')
# Intended formations per format (DESIGN 2.5). Small formats count the keeper in front ('1-3-2-1' is 7), except the
# U.S. 4v4 diamond '1-2-1', which has no keeper; 11v11 uses team_shape's outfield names (its 10- and 9-outfield
# templates; a test asserts this list equals team_shape.FORMATION_LABELS). null = "Other / not sure".
FORMATIONS_11 = ('4-4-2', '4-3-3', '4-2-3-1', '4-1-4-1', '3-5-2', '3-4-3', '5-3-2', '5-4-1', '4-4-1', '4-3-2', '5-3-1')
FORMATIONS = {4: ('1-2-1',), 5: ('1-2-1',), 7: ('1-3-2-1', '1-2-3-1', '1-3-1-2'), 9: ('1-3-2-3', '1-3-3-2', '1-4-3-1'),
              11: FORMATIONS_11}
STYLES = ('possession', 'counter', 'high_press', 'compact', 'unknown')
FEDERATIONS = ('auto', 'US', 'England', 'Canada')
DETECTORS = ('pressure_cover', 'weak_side_balance', 'wide_defender_advance', 'behind_ball', 'defensive_spacing',
             'attacking_shape')
DETECTOR_CODE = {'pressure_cover': 'D1', 'weak_side_balance': 'D2', 'wide_defender_advance': 'D3', 'behind_ball': 'D4',
                 'defensive_spacing': 'D5', 'attacking_shape': 'D6'}
CODE_DETECTOR = {v: k for k, v in DETECTOR_CODE.items()}
# The tier an opportunity of each detector is made in (DESIGN 1.2): D3 needs the wide defender and the centre-backs
# (roles), D5 needs the relative back block on small formats; the others are role-free.
DETECTOR_TIERS = {'pressure_cover': ('measurement',), 'weak_side_balance': ('measurement',),
                  'wide_defender_advance': ('role',), 'behind_ball': ('measurement',),
                  'defensive_spacing': ('measurement', 'role'), 'attacking_shape': ('measurement',)}
# The phase each detector's opportunity needs when it is assessed (DESIGN 4.1-4.6; D6 falls back to 'unknown' with a
# describe-only reading).
DETECTOR_PHASES = {'pressure_cover': ('opp',), 'weak_side_balance': ('opp',), 'wide_defender_advance': ('own',),
                   'behind_ball': ('own',), 'defensive_spacing': ('opp',), 'attacking_shape': ('own', 'unknown')}
SHAPE_DETECTORS = ('wide_defender_advance', 'behind_ball', 'defensive_spacing', 'attacking_shape')   # spans >= 3 s
MIN_SHAPE_SPAN_S = 3.0      # possession rule 5: shape and spacing metrics use only phase spans of >= 3 s
SKILLS = {'pressure_and_cover': 'pressure_cover', 'someone_back_when_attacking': 'behind_ball',
          'wide_defender_support': 'wide_defender_advance', 'far_side_protection': 'weak_side_balance',
          'back_line_coordination': 'defensive_spacing', 'spread_when_attacking': 'attacking_shape'}
FINDING_VERDICTS = ('strength', 'try_this', 'work_on', 'review', 'describe')
CRITICISM = ('try_this', 'work_on')     # verdicts that need every guardrail to pass (DESIGN 4.8)
ABSTAIN_REASONS = ('phase_unknown', 'possession_unavailable', 'ball_unreliable', 'no_carrier', 'out_of_view',
                   'team_incomplete', 'identity_uncertain', 'role_unconfirmed', 'scale_unknown', 'format_not_supported',
                   'registration_low', 'short_range', 'recovering', 'units_interleaved', 'near_change',
                   # fix round 1 (DETECT, additive; critic S-7, S-3): the relevant phase never occurred although the
                   # phase is known; the coach marked only one team's possession (no engine); every criticism the
                   # check could make lay within a formation change
                   'phase_absent', 'coach_marks_one_side', 'formation_changing',
                   # fix round 3 (ASSESS, additive; critic S-2): below the review floor with enough of the relevant
                   # phase but too few situations (short_range keeps 'too little of the phase')
                   'too_few_situations')
PHASES = ('own', 'opp', 'dead', 'unknown')
ZONES = ('own_third', 'middle_third', 'final_third')
CHANNELS = ('left', 'centre', 'right')
CONFIDENCE = ('high', 'medium', 'low')
TIERS = ('world', 'pinned', 'image')
ROLE_SOURCES = ('coach', 'stable', 'carried', 'window', 'relative', 'none')
TEMPLATE_SOURCES = ('stable', 'carried', 'window')      # derived from team_shape templates: 11v11 only (DESIGN 2.5)
# Tactical role names: team_shape's template roles (11v11), plus GK, CB and CM; small formats use relative slots
# 'back-2/3' (block, k-th from the left of n) or a coach-confirmed name.
TEMPLATE_ROLES = ('LB', 'LCB', 'CB', 'RCB', 'RB', 'LWB', 'RWB', 'LM', 'LCM', 'CM', 'RCM', 'RM', 'CDM', 'LDM', 'RDM',
                  'CAM', 'LW', 'RW', 'CF', 'LCF', 'RCF', 'GK')
RELATIVE_ROLE = r'(?:back|mid|front)-[1-9]/[1-9]'
CLAIM_TYPES = ('situation', 'actor_state', 'relation', 'team_state', 'opponent_state', 'recurrence', 'intent_signal',
               'principle', 'counterfactual', 'hypothetical', 'action', 'drill_link', 'limitation')
GUARDRAILS = ('identity', 'position', 'role', 'context', 'phase', 'intent', 'alternative', 'evidence')
ALT_REASONS = ('no_reachable_fix', 'would_open_new_gap', 'mover_unseen', 'context_missing', 'scale_unknown',
               'guardrail_failed', 'no_zone_for_detector')
MASTERY_STATES = ('NOT ENOUGH EVIDENCE', 'LEARNING', 'DEVELOPING', 'CONSISTENT', 'ADAPTABLE')
ORIGIN_P = 'PITCHIQ-PROPOSAL'
TAGS = ('FEDERATION', ORIGIN_P)

# Which settings fields feed which cache tier (DESIGN 1.2). squad and runLinks are read at serve time (progression,
# the team link) and are in no fingerprint, like the coach's feedback.
TIER_FIELDS = {'measurement': ('format', 'field'), 'role': ('roles', 'formation'),
               'interpretation': ('ageGroup', 'playingLevel', 'style', 'federation')}
SERVE_TIME_FIELDS = ('squad', 'runLinks')
TIER_ORDER = ('measurement', 'role', 'interpretation')
DEFAULT_FORMAT = {'players': 11, 'goalkeeper': True}     # the runs PitchIQ has today are 11v11 broadcast or camera
TIER_DEFAULTS = {'format': DEFAULT_FORMAT, 'field': None, 'roles': {}, 'formation': {'intended': None, 'periods': []},
                 'ageGroup': None, 'playingLevel': None, 'style': 'unknown', 'federation': 'auto'}

# Annotation elements (DESIGN 5.4, 5.5): where each type may be drawn (video tiers and the top-down diagram), and how
# many of the new types one annotation may carry. The old types keep the CONTRACT section 6 rules (pitch geometry
# only at world; unit lines and measures between feet from pinned; boxes, labels and arrows between feet at image).
TIER_ELEMENTS = {
    'focus': ('world', 'pinned', 'image', 'diagram'), 'support': ('world', 'pinned', 'image', 'diagram'),
    'relation': ('world', 'pinned', 'diagram'), 'channel': ('world', 'diagram'),
    'suggestedZone': ('world', 'diagram'), 'moveArrow': ('world', 'diagram'),
    'caption': ('world', 'pinned', 'image', 'diagram'),
    'player': ('world', 'pinned', 'image', 'diagram'), 'unit': ('world', 'pinned', 'diagram'),
    'measure': ('world', 'pinned', 'diagram'), 'zone': ('world', 'diagram'), 'band': ('world', 'diagram'),
    'arrow': ('world', 'pinned', 'image', 'diagram'), 'label': ('world', 'pinned', 'image', 'diagram'),
    'suggested': ('diagram',)}
ELEMENT_BUDGET = {'focus': 2, 'support': 3, 'relation': 3, 'channel': 1, 'suggestedZone': 1, 'moveArrow': 1,
                  'caption': 1}
CAPTION_MAX = 90
NORM_MARGIN = .1            # diagram coordinates may reach this far outside the pitch (a player on the touchline)

# ---------- primitive types ----------
PLAYER_ID = r'[AB]-\d{1,3}|GK-\d{1,3}'   # internal ids (A-07, GK-1); displays (GK-A, B-?) are refused
Num = float                                             # a JSON number: int or float, finite, never a bool
Seconds = Annotated[float, Range(0.0, 1e6)]             # absolute source-video seconds
Duration = Annotated[float, Range(0.0, 1e6)]
Frame = Annotated[int, Range(0, 10**9)]
Count = Annotated[int, Range(0, 10**8)]
Share = Annotated[float, Range(0.0, 1.0)]
Metres = Annotated[float, Range(0.0, 1000.0)]
Norm = Annotated[float, Range(-NORM_MARGIN, 1+NORM_MARGIN)]
Level = Literal[1, 2, 3, 4, 5, 6, 7]
Team = Literal['A', 'B']
AgeGroup = Literal['U6', 'U7', 'U8', 'U9', 'U10', 'U11', 'U12', 'U13', 'U14', 'U15', 'U16', 'U17', 'U18', 'U19', 'U21',
                   'U23', 'Adult']
FormatName = Literal['4v4', '5v5', '7v7', '9v9', '11v11']
Style = Literal['possession', 'counter', 'high_press', 'compact', 'unknown']
Federation = Literal['auto', 'US', 'England', 'Canada']
DetectorName = Literal['pressure_cover', 'weak_side_balance', 'wide_defender_advance', 'behind_ball',
                       'defensive_spacing', 'attacking_shape']
SkillId = Literal['pressure_and_cover', 'someone_back_when_attacking', 'wide_defender_support', 'far_side_protection',
                  'back_line_coordination', 'spread_when_attacking']
Verdict = Literal['strength', 'try_this', 'work_on', 'review', 'describe']
AbstainReason = Literal['phase_unknown', 'possession_unavailable', 'ball_unreliable', 'no_carrier', 'out_of_view',
                        'team_incomplete', 'identity_uncertain', 'role_unconfirmed', 'scale_unknown',
                        'format_not_supported', 'registration_low', 'short_range', 'recovering', 'units_interleaved',
                        'near_change', 'phase_absent', 'coach_marks_one_side', 'formation_changing',
                        'too_few_situations']
Phase = Literal['own', 'opp', 'dead', 'unknown']
ZoneName = Literal['own_third', 'middle_third', 'final_third']
Channel = Literal['left', 'centre', 'right']
Conf = Literal['high', 'medium', 'low']
Tier = Literal['world', 'pinned', 'image']
RoleSource = Literal['coach', 'stable', 'carried', 'window', 'relative', 'none']
ClaimType = Literal['situation', 'actor_state', 'relation', 'team_state', 'opponent_state', 'recurrence',
                    'intent_signal', 'principle', 'counterfactual', 'hypothetical', 'action', 'drill_link',
                    'limitation']
GuardrailState = Literal['pass', 'fail', 'not_applicable']
AltReason = Literal['no_reachable_fix', 'would_open_new_gap', 'mover_unseen', 'context_missing', 'scale_unknown',
                    'guardrail_failed', 'no_zone_for_detector']
Tag = Literal['FEDERATION', 'PITCHIQ-PROPOSAL']
PlayerId = Annotated[str, Pattern(PLAYER_ID)]
RoleName = Annotated[str, Pattern('|'.join(TEMPLATE_ROLES)+'|'+RELATIVE_ROLE)]
FormationName = Annotated[str, Pattern(r'\d(?:-\d){1,4}')]
OppId = Annotated[str, Pattern(r'opp-[0-9a-f]{12}')]
EvId = Annotated[str, Pattern(r'ev-[0-9a-f]{12}')]
FndId = Annotated[str, Pattern(r'fnd-[0-9a-f]{12}')]
AltId = Annotated[str, Pattern(r'alt-[0-9a-f]{12}')]
PeId = Annotated[str, Pattern(r'pe-[0-9a-f]{12}')]
SqId = Annotated[str, Pattern(r'sq-[0-9a-f]{12}')]
AdjId = Annotated[str, Pattern(r'adj-[0-9a-f]{12}')]
DiagramId = Annotated[str, Pattern(r'dg-[0-9a-f]{12}')]
DrillId = Annotated[str, Pattern(r'[a-z][a-z0-9_]{2,60}')]
SourceId = Annotated[str, Pattern(r'[A-Z][A-Z0-9]*(?:[-_][A-Z0-9]+)*')]
Code = Annotated[str, Pattern(r'[a-z][a-z0-9_]{0,60}')]
FieldName = Annotated[str, Pattern(r'[A-Za-z][A-Za-z0-9_]{0,40}')]   # a key of an open map (camelCase)
Word = Annotated[str, Pattern(r'[a-z][a-z \-]{0,40}')]
Hex16 = Annotated[str, Pattern(r'[0-9a-f]{16}')]
Hex64 = Annotated[str, Pattern(r'[0-9a-f]{64}')]
ISO = Annotated[str, Pattern(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}(?::\d{2}(?:\.\d{1,6})?)?(?:Z|[+-]\d{2}:?\d{2})?')]
Date = Annotated[str, Pattern(r'\d{4}-\d{2}-\d{2}')]
RunName = Annotated[str, Pattern(r'[A-Za-z0-9_.\-]{1,120}')]
FindingKey = Annotated[str, Pattern(r'D[1-6]\|[AB]\|[A-Za-z0-9_.\-]{1,40}\|\d+\.\d\|\d+\.\d')]
OppKey = Annotated[str, Pattern(r'D[1-6]\|[AB]\|(?:[A-Za-z0-9_.\-]{1,40}\|)?\d+\.\d\|\d+\.\d')]
Url = Annotated[str, Pattern(r'https?://[!-~]{3,500}')]
Point = Annotated[list[float], Len(2, 2)]               # [x, y]: normalized pitch (diagrams) or pixels (overlays)
Polygon = Annotated[list[Point], Len(3, 64)]
Span = Annotated[list[Seconds], Len(2, 2), Increasing()]
Scalar = float | int | bool | str | None


# ================= settings (DESIGN 2.1): runs/<video>/coaching_settings.json =================
class SquadRef(TypedDict):
    id: SqId
    name: Annotated[str, Text(60)]


class FormatSpec(TypedDict):
    players: Literal[4, 5, 7, 9, 11]
    goalkeeper: bool


class FieldSize(TypedDict):
    length: Annotated[float, Range(15.0, 130.0)]
    width: Annotated[float, Range(10.0, 100.0)]


FormationPeriod = TypedDict('FormationPeriod', {'t0': Seconds, 't1': Seconds, 'formation': FormationName})


class FormationSetting(TypedDict):
    intended: FormationName | None
    periods: list[FormationPeriod]


RoleSpan = TypedDict('RoleSpan', {'role': RoleName, 'from': Seconds | None, 'to': Seconds | None, 'set': ISO})


class RunLink(TypedDict):
    run: RunName
    team: Team
    confirmed: bool


class TeamSettings(TypedDict):
    squad: SquadRef | None
    ageGroup: AgeGroup
    playingLevel: Level
    format: FormatSpec
    field: FieldSize | None
    federation: Federation
    formation: FormationSetting
    style: Style
    roles: dict[PlayerId, list[RoleSpan]]
    runLinks: list[RunLink]


class TeamsSettings(TypedDict):
    A: TeamSettings | None
    B: TeamSettings | None


class CoachingSettings(TypedDict):
    version: Literal[1]
    teams: TeamsSettings
    updated: ISO | None


# ================= profile (DESIGN 2.2-2.6): profiles.resolve(team settings) =================
class Guidance(TypedDict):
    text: Annotated[str, Text(400)]      # a paraphrase; quoted phrases at most 60 characters
    tag: Tag
    sources: list[SourceId]


class DetectorProfile(TypedDict):
    enabled: bool
    verdicts: Annotated[list[Verdict], Unique()]   # the verdicts this level may give for the detector
    framing: Annotated[str, Text(200)] | None      # the level's framing of the behaviour (e.g. "go to the ball")
    params: dict[FieldName, Scalar]                # thresholds and options [P]
    origin: Literal['PITCHIQ-PROPOSAL']
    validated: Literal[False]


class EvidenceRules(TypedDict):
    minOpportunities: Count
    minFailingSpells: Count
    minFailureShare: Share | None        # None: style-relative (level 7)
    minPhaseSeconds: Duration
    maxWorkItems: Annotated[int, Range(0, 10)]
    strengthSpells: Count
    strengthShare: Share
    naming: Literal['never', 'relative', 'role_words', 'evidence_only', 'yes']
    origin: Literal['PITCHIQ-PROPOSAL']
    validated: Literal[False]


class Register(TypedDict):
    tone: Annotated[str, Text(120)]
    numbers: Literal['none', 'counts', 'rounded_metres', 'metres', 'shares', 'rates']
    maxChallenges: Annotated[int, Range(0, 10)]
    concepts: list[Word]
    banned: list[Word]


class ProfileSafety(TypedDict):
    heading: Literal['none', 'limited']
    maxLoad: Literal['low', 'moderate', 'high']
    tackling: bool
    sprintRaces: bool
    notes: list[Annotated[str, Text(300)]]


class Notice(TypedDict):
    code: Code
    text: Annotated[str, Text(400)]


class ResolvedFormat(TypedDict):
    players: Literal[4, 5, 7, 9, 11]
    goalkeeper: bool
    outfield: Annotated[int, Range(3, 10)]
    name: FormatName


class Profile(TypedDict):
    version: Annotated[int, Range(1, 1000)]          # profiles.json version
    sha: Hex16                                       # profiles.json sha (first 16 hex)
    team: Team
    ageGroup: AgeGroup
    level: Level
    levelName: Annotated[str, Text(60)]
    band: Annotated[str, Text(40)] | None            # sub-band: 'U15-U16', 'U17-U18', 'academy', 'adult'
    format: ResolvedFormat
    field: FieldSize | None                          # declared by the coach; the run's scale is in the timeline
    federation: Federation
    style: Style
    formation: FormationName | None
    objectives: list[Guidance]
    principles: list[Guidance]
    notExpected: list[Guidance]
    detectors: dict[DetectorName, DetectorProfile]
    evidence: EvidenceRules
    register: Register
    skills: dict[SkillId, Annotated[str, Text(80)] | None]   # level-specific display names; None = not shown
    drillPool: list[DrillId]
    safety: ProfileSafety
    notices: list[Notice]
    sources: list[SourceId]


# ================= situations (DESIGN 3.1-3.3): the SituationTimeline summary in obs-*.json =================
# Coach-marked possession spans (user decision 1, 2026-10-21; added by CONTEXT wave 2 through the INTERFACES.md change
# procedure, the layout EVAL-synth proposed): "team A had the ball from t0 to t1", "dead ball from t0 to t1", in
# absolute source seconds, per video. Where present they decide the phase instead of possession-4 (situations.py).
class CoachSpan(TypedDict):
    t0: Seconds
    t1: Seconds
    state: Literal['A', 'B', 'dead']


class CoachSpans(TypedDict):
    version: Literal[1]
    kind: Literal['coach-possession-spans']
    source: Literal['coach']
    spans: list[CoachSpan]               # sorted, t0 < t1, no overlap (inside the analysed segment: situations.py)


class CoachPhase(TypedDict):
    spans: Count                         # coach spans used
    dropped: Count                       # coach spans refused (outside the segment, overlapping, malformed)
    seconds: Duration                    # timeline seconds the coach spans decide
    own: Duration
    opp: Duration
    dead: Duration
    engineNamedSeconds: Duration         # of `seconds`, where possession-4 (after its rules) had named a team
    agreement: Share | None              # share of engineNamedSeconds where it named the coach's team


class RoughShares(TypedDict):
    A: Share
    B: Share


class PhaseQuality(TypedDict):
    available: bool                      # possession-4's record (the engine), whatever the coach marked
    reason: Annotated[str, Text(300)] | None
    level: Literal['good', 'limited', 'unavailable']
    coverage: Share | None
    roughShares: RoughShares | None      # rounded to 5% (possession rule 7)
    notes: list[Annotated[str, Text(300)]]
    coach: NotRequired[CoachPhase | None]   # coach-marked spans merged into the phase (None: none given)


class Footage(TypedDict):
    kind: Literal['broadcast', 'amateur', 'veo', 'unknown']
    smallSceneShare: Share
    keyless: bool
    lensBent: bool


class DataQuality(TypedDict):
    inKit: Annotated[float, Range(0, 30)] | None
    confirmed: Annotated[float, Range(0, 30)] | None
    positioned: Annotated[float, Range(0, 30)] | None
    lineup: Annotated[int, Range(1, 11)] | None
    dominantCause: Literal['registration', 'identity', 'out_of_view'] | None


class Scale(TypedDict):
    lengthM: Annotated[float, Range(15.0, 130.0)]
    widthM: Annotated[float, Range(10.0, 100.0)]
    source: Literal['measured', 'declared', 'assumed']
    metresUsable: bool                   # body-scale distances may be judged (DESIGN 2.5)


class PhaseSpan(TypedDict):
    state: Phase
    t0: Seconds
    t1: Seconds
    frame0: Frame
    frame1: Frame
    source: NotRequired[Literal['possession-4', 'coach']]   # who named it (situations.py splits spans by source)


class PeriodSummary(TypedDict):
    index: Annotated[int, Range(0, 10000)]
    t0: Seconds
    t1: Seconds
    label: FormationName | None
    confidence: Literal['clear', 'likely', 'ambiguous', 'unclear'] | None


class PhaseSeconds(TypedDict):
    own: Duration
    opp: Duration
    dead: Duration
    unknown: Duration
    nearChange: Duration


class PlayerSummary(TypedDict):
    team: Team
    goalkeeper: bool
    observedSeconds: Duration
    p90SpeedMS: Annotated[float, Range(0.0, 15.0)] | None   # the player's own observed p90 speed (DESIGN 5.1 step 5)


class TimelineSummary(TypedDict):
    team: Team
    samples: Count
    step: Annotated[float, Range(0.01, 5.0)]
    t0: Seconds
    t1: Seconds
    frame0: Frame
    frame1: Frame
    fps: Annotated[float, Range(0.0, 1000.0)]
    format: FormatName
    phaseSeconds: PhaseSeconds
    phaseSpans: list[PhaseSpan]          # named spans after the possession rules (excluded change windows removed)
    phaseQuality: PhaseQuality
    footage: Footage
    dataQuality: DataQuality
    scale: Scale
    completeShare: Share
    observedMean: Annotated[float, Range(0, 30)]
    ballReliableShare: Share
    carrierShare: Share | None           # share of the opponents' possession with a known carrier
    formationPeriods: list[PeriodSummary]
    players: dict[PlayerId, PlayerSummary]


# ================= opportunities (DESIGN 4.0) =================
class OppPhase(TypedDict):
    state: Phase
    source: Literal['possession-4', 'coach', 'none']
    estimated: bool
    spanSeconds: Duration | None
    nearChange: bool


class OppBall(TypedDict):
    zone: ZoneName | None
    channel: Channel | None
    reliable: bool


class Actors(TypedDict):
    primary: list[PlayerId]
    supporting: list[PlayerId]
    opponents: list[PlayerId]
    carrier: PlayerId | None


class RoleRef(TypedDict):
    role: RoleName | None
    source: RoleSource
    confidence: Conf | None


Interchange = TypedDict('Interchange', {'t': Seconds, 'out': PlayerId, 'in': PlayerId, 'slot': Annotated[str, Text(40)]})


class IntentSignals(TypedDict):
    recovering: bool
    interchange: Interchange | None
    formationChangeNear: bool


class OppQuality(TypedDict):
    identity: Conf
    position: Conf
    visibility: Literal['complete', 'disc', 'partial', 'none']
    ball: Literal['reliable', 'unreliable', 'not_needed']
    samples: Annotated[int, Range(1, 100000)]


class Opportunity(TypedDict):
    id: OppId
    key: OppKey
    detector: DetectorName
    team: Team
    tier: Literal['measurement', 'role']
    variant: NotRequired[Annotated[str, Pattern(r'[A-Za-z0-9_.\-]{1,40}')] | None]
    t0: Seconds
    t1: Seconds
    frame0: Frame
    frame1: Frame
    phase: OppPhase
    ball: OppBall
    actors: Actors
    roles: dict[PlayerId, RoleRef]
    relations: dict[FieldName, list[Scalar]]   # per sample; 't' required
    quality: OppQuality
    intentSignals: IntentSignals
    abstain: AbstainReason | None


# ================= evidence moments (in obs-*.json / roles-*.json; profile-independent) =================
class SceneEntry(TypedDict):
    team: Team
    x: Norm
    y: Norm
    kind: Annotated[str, Pattern(r'[A-Z]{1,2}')]     # 'O' observed (the only kind drawn as a fact)
    sigma: Annotated[float, Range(0.0, 100.0)] | None
    goalkeeper: bool
    box: Annotated[list[float], Len(4, 4)] | None    # normalized image box [x, y, w, h] at this frame


class Scene(TypedDict):
    t: Seconds
    frame: Frame
    players: dict[PlayerId, SceneEntry]
    ball: Point | None
    ballReliable: bool


class ImageSize(TypedDict):
    width: Annotated[int, Range(1, 100000)]
    height: Annotated[int, Range(1, 100000)]


class Moment(TypedDict):
    id: EvId
    opportunity: OppId
    detector: DetectorName
    team: Team
    sample: Count                        # index into the opportunity's relations arrays
    t: Seconds
    frame: Frame
    seek: Seconds
    fps: Annotated[float, Range(0.0, 1000.0)]
    tier: Tier
    calibration: dict[FieldName, Scalar] | None
    players: list[PlayerId]
    flagged: list[PlayerId]              # stabilizer flags within +-0.5 s (never used as evidence)
    scene: Scene
    before: Scene | None                 # the scene REACH_H seconds earlier (alternatives' reach), None when unseen
    image: ImageSize | None


class FrameTier(TypedDict):
    t0: Seconds
    t1: Seconds
    frame0: Frame
    frame1: Frame
    tier: Tier
    confidence: Share | None


class DetectorCoverage(TypedDict):
    detector: DetectorName
    tier: Literal['measurement', 'role']
    eligibleSeconds: Duration
    phaseSeconds: Duration | None        # the detector's relevant phase seconds
    opportunities: Count
    eligibleSpans: list[Span]
    abstain: dict[AbstainReason, Duration]   # seconds of the relevant phase lost per reason
    dominantReason: AbstainReason | None


class EngineStamp(TypedDict):
    version: Literal['2']
    measurement: Hex16
    roles: NotRequired[Hex16]
    interpretation: NotRequired[Hex16]


class MeasurementSettings(TypedDict):
    format: FormatSpec
    field: FieldSize | None


class ObsInputs(TypedDict):
    result: Hex64
    shapeSettings: Hex64
    measurement: MeasurementSettings


class SegmentInfo(TypedDict):
    t0: Seconds
    t1: Seconds
    fps: Annotated[float, Range(0.0, 1000.0)]
    frames: Count


class VideoInfo(TypedDict):
    name: Annotated[str, Len(1, 300)] | None
    key: Annotated[str, Len(1, 120)] | None


class ObsFile(TypedDict):
    kind: Literal['coaching-observations']
    version: Literal[2]
    team: Team
    engine: EngineStamp
    fingerprint: Hex64                   # mfp
    inputs: ObsInputs
    segment: SegmentInfo
    video: VideoInfo
    timeline: TimelineSummary
    detectors: dict[DetectorName, DetectorCoverage]
    opportunities: list[Opportunity]
    moments: list[Moment]
    frameTiers: list[FrameTier]
    roleInputs: dict[FieldName, Any]   # open: CONTEXT/DETECT document it


# ================= roles (DESIGN 3.4): roles-*.json =================
class RoleRecord(TypedDict):
    id: PlayerId
    t0: Seconds
    t1: Seconds
    role: RoleName | None
    group: Literal['GK', 'D', 'M', 'F'] | None
    side: Literal['L', 'C', 'R'] | None
    source: RoleSource
    confidence: Conf | None
    support: Share | None
    period: Annotated[int, Range(0, 10000)] | None
    ambiguousWith: list[RoleName]
    conflict: Code | None                # e.g. 'confirmed_role_contradicted'


class FormationCheck(TypedDict):
    period: Annotated[int, Range(0, 10000)]
    t0: Seconds
    t1: Seconds
    measured: FormationName | None
    measuredConfidence: Literal['clear', 'likely', 'ambiguous', 'unclear'] | None
    intended: FormationName | None
    mismatch: bool                       # clear reading, differs from the plan and not RELATED
    related: bool


class RoleBook(TypedDict):
    team: Team
    format: FormatName
    model: Literal['team_shape', 'relative']
    records: list[RoleRecord]
    interchanges: list[Interchange]
    formationChecks: list[FormationCheck]


class RoleInputsStamp(TypedDict):
    roles: dict[PlayerId, list[RoleSpan]]
    formation: FormationSetting


class RolesFile(TypedDict):
    kind: Literal['coaching-roles']
    version: Literal[2]
    team: Team
    engine: EngineStamp
    measurement: Hex64                   # mfp of the obs file it was built from
    fingerprint: Hex64                   # rfp
    inputs: RoleInputsStamp
    rolebook: RoleBook
    detectors: dict[DetectorName, DetectorCoverage]
    opportunities: list[Opportunity]
    moments: list[Moment]


# ================= findings (DESIGN 4.8, 6.1, 6.2) =================
class Claim(TypedDict):
    type: ClaimType
    facts: dict[FieldName, Any]
    tag: NotRequired[Tag]
    source: NotRequired[SourceId | None]


class FindingTimestamp(TypedDict):
    evidence: EvId
    opportunity: OppId
    t: Seconds
    frame: Frame
    seek: Seconds


class FindingPlayers(TypedDict):
    primary: list[PlayerId]
    supporting: list[PlayerId]
    opponents: list[PlayerId]


class PhaseContext(TypedDict):
    state: Phase
    estimated: bool
    coverage: Share | None


class BallContext(TypedDict):
    zone: ZoneName | None
    channel: Channel | None


class FindingContext(TypedDict):
    phase: PhaseContext
    ball: BallContext | None
    style: Style
    styleEffect: Code | None
    format: FormatName


class Aggregate(TypedDict):
    opportunities: Count
    demonstrated: Count
    notDemonstrated: Count
    unclear: Count
    distinctFailingSpells: Count
    distinctDemonstratedSpells: Count
    phaseSeconds: Duration
    eligibleShare: Share | None
    ratePer10: Annotated[float, Range(0.0, 10000.0)] | None


class FindingCoverage(TypedDict):
    identity: Conf
    position: Conf
    visibility: Literal['complete', 'disc', 'partial', 'none']
    ball: Literal['reliable', 'unreliable', 'not_needed']
    calibrationTier: Tier | None
    eligibleShareOfPhase: Share | None


Guardrails = TypedDict('Guardrails', {g: GuardrailState for g in GUARDRAILS})


class SuggestedAction(TypedDict):
    principle: Code
    mode: Literal['zone', 'point', 'direction', 'principle']
    alternatives: list[AltId]


class FindingText(TypedDict):
    headline: Annotated[str, Text(160)]
    explanation: Annotated[str, Text(1500)]
    short: Annotated[str, Text(120)]
    action: Annotated[str, Text(500)] | None


class Finding(TypedDict):
    id: FndId
    key: FindingKey
    observation: DetectorName
    verdict: Verdict
    level: Level
    team: Team
    priority: Annotated[int, Range(1, 10)] | None
    players: FindingPlayers
    roles: dict[PlayerId, RoleRef]
    timestamps: list[FindingTimestamp]
    opportunities: Annotated[list[OppId], Unique()]
    context: FindingContext
    measurements: dict[FieldName, Scalar]
    aggregate: Aggregate
    coverage: FindingCoverage
    guardrails: Guardrails
    limitations: list[Code]
    suggestedAction: SuggestedAction | None
    claims: list[Claim]
    text: FindingText | None             # None until explain has run; a report requires it
    drills: list[DrillId]


# ================= alternatives (DESIGN 5.1-5.3) =================
class AltZone(TypedDict):
    polygon: Polygon                     # normalized pitch coordinates
    areaM2: Annotated[float, Range(0.0, 10000.0)]
    kind: Literal['zone', 'point']
    radiusM: Annotated[float, Range(0.0, 50.0)] | None


class Reach(TypedDict):
    horizonS: Annotated[float, Range(0.1, 10.0)]
    speedMS: Annotated[float, Range(0.1, 12.0)]
    radiusM: Annotated[float, Range(0.0, 120.0)]


class Improves(TypedDict):
    relation: Code
    before: bool
    after: bool
    shareOfZone: Share


class AltChecks(TypedDict):
    pressureKept: bool
    centralKept: bool
    gapOk: bool
    noNewFreeOpponent: bool
    noStacking: bool


class Rejected(TypedDict):
    mover: PlayerId
    reason: AltReason


Alternative = TypedDict('Alternative', {
    'id': AltId, 'finding': FndId, 'evidence': EvId, 't': Seconds, 'mover': PlayerId, 'principle': Code,
    'zone': AltZone, 'display': Literal['zone', 'point', 'direction'], 'from': Point, 'reach': Reach,
    'improves': Improves, 'checks': AltChecks, 'assumption': Literal['single_mover'], 'rejected': list[Rejected]})


class NoAlternative(TypedDict):
    finding: FndId
    evidence: EvId | None
    t: Seconds | None
    principle: Code
    reason: AltReason
    rejected: list[Rejected]


# ================= evidence, diagrams, overlays, annotation elements (DESIGN 5.4-5.6; CONTRACT section 6) =========
class FocusEl(TypedDict):
    type: Literal['focus']
    id: PlayerId
    x: float
    y: float
    tone: Literal['review', 'strength']
    label: bool
    secondary: NotRequired[bool]
    box: NotRequired[Annotated[list[float], Len(4, 4)]]


class SupportEl(TypedDict):
    type: Literal['support']
    id: PlayerId
    team: Team
    x: float
    y: float
    why: Annotated[str, Text(60)]
    box: NotRequired[Annotated[list[float], Len(4, 4)]]


class RelationEl(TypedDict):
    type: Literal['relation']
    a: PlayerId
    b: PlayerId | Literal['goal', 'ball', 'zone']
    kind: Literal['cover', 'gap', 'channel', 'pressure', 'behind_ball']
    ok: bool
    text: NotRequired[Annotated[str, Text(40)]]
    pa: NotRequired[Point]
    pb: NotRequired[Point]


class ChannelEl(TypedDict):
    type: Literal['channel']
    polygon: Polygon
    kind: Literal['vacated', 'gap', 'weak_side']


class SuggestedZoneEl(TypedDict):
    type: Literal['suggestedZone']
    polygon: Polygon
    kind: Literal['zone', 'point']
    alternative: AltId
    mover: PlayerId
    label: Annotated[str, Text(40)]


MoveArrowEl = TypedDict('MoveArrowEl', {'type': Literal['moveArrow'], 'id': PlayerId, 'from': Point, 'to': Point})


class CaptionEl(TypedDict):
    type: Literal['caption']
    text: Annotated[str, Text(CAPTION_MAX)]
    anchor: Literal['top-left', 'top-right', 'bottom-left', 'bottom-right']


class PlayerEl(TypedDict):
    type: Literal['player']
    id: PlayerId
    team: Team
    x: float
    y: float
    kind: NotRequired[Annotated[str, Pattern(r'[A-Z]{0,2}')]]
    role: NotRequired[Annotated[str, Len(0, 8)] | None]
    emphasis: NotRequired[bool]
    box: NotRequired[Annotated[list[float], Len(4, 4)]]
    pitch: NotRequired[Point]


class UnitEl(TypedDict):
    type: Literal['unit']
    unit: Annotated[str, Pattern(r'[A-Z][A-Z0-9]{0,3}')]
    ids: list[PlayerId]


class MeasureEl(TypedDict):
    type: Literal['measure']
    a: Point
    b: Point
    text: NotRequired[Annotated[str, Text(60)]]
    bracket: NotRequired[bool]
    ids: NotRequired[list[PlayerId]]


class ZoneEl(TypedDict):
    type: Literal['zone']
    polygon: Polygon
    style: Literal['gap', 'space']


BandEl = TypedDict('BandEl', {'type': Literal['band'], 'axis': Literal['x', 'y'], 'from': float, 'to': float,
                              'style': Literal['gap', 'space']})


class ArrowEl(TypedDict):
    type: Literal['arrow']
    id: PlayerId
    points: Annotated[list[Annotated[list[float], Len(2, 3)]], Len(2, 400)]


class LabelEl(TypedDict):
    type: Literal['label']
    at: Point
    text: Annotated[str, Text(120)]


class SuggestedEl(TypedDict):             # v1 'COACHING ILLUSTRATION' circle: render-only for old reports
    type: Literal['suggested']
    id: PlayerId
    x: float
    y: float
    label: NotRequired[Annotated[str, Text(60)]]


Element = (FocusEl | SupportEl | RelationEl | ChannelEl | SuggestedZoneEl | MoveArrowEl | CaptionEl | PlayerEl | UnitEl
           | MeasureEl | ZoneEl | BandEl | ArrowEl | LabelEl | SuggestedEl)


class PitchSize(TypedDict):
    length: Annotated[float, Range(15.0, 130.0)]
    width: Annotated[float, Range(10.0, 100.0)]


class Diagram(TypedDict):
    preset: Annotated[str, Pattern(r'[a-z][a-z0-9_]{1,40}')]
    title: Annotated[str, Text(120)]
    t: Seconds
    frame: Frame
    team: Team
    attack: Literal['left', 'right'] | None
    metresEstimated: bool
    pitch: PitchSize
    elements: list[Element]


class VideoOverlay(TypedDict):
    tier: Tier
    width: Annotated[int, Range(1, 100000)]
    height: Annotated[int, Range(1, 100000)]
    elements: list[Element]


class Window(TypedDict):
    t0: Seconds
    t1: Seconds


class Annotation(TypedDict):
    finding: FndId
    window: Window                       # DESIGN 5.6: key time - 3.0 s to + 2.0 s [P]
    keyFrame: Frame
    keyTime: Seconds
    caption: Annotated[str, Text(CAPTION_MAX)]


class Evidence(TypedDict):
    id: EvId                             # the moment's id in obs-*.json / roles-*.json (invariant 3)
    opportunity: OppId
    finding: FndId | None
    detector: DetectorName
    team: Team
    time: Seconds
    frame: Frame
    seek: Seconds
    fps: Annotated[float, Range(0.0, 1000.0)]
    segmentTime: Annotated[float, Range(0.0, 1e6)]
    tier: Tier
    calibration: dict[FieldName, Scalar] | None
    players: list[PlayerId]
    flagged: list[PlayerId]
    confidence: Conf
    alternative: AltId | None
    diagram: Diagram | None
    videoOverlay: VideoOverlay | None
    annotation: Annotation | None
    caption: Annotated[str, Text(200)]


# ================= drills v2 (DESIGN 7.1) =================
class AgeRange(TypedDict):
    min: AgeGroup
    max: AgeGroup


class PlayerCount(TypedDict):
    min: Annotated[int, Range(2, 30)]
    max: Annotated[int, Range(2, 30)]
    note: Annotated[str, Text(200)]


class Dimension(TypedDict):
    value: Annotated[float, Range(1.0, 120.0)]
    unit: Literal['m']
    scaleWith: Literal['age', 'format'] | None


class FieldSetup(TypedDict):
    length: Annotated[float, Range(1.0, 120.0)] | Dimension
    width: Annotated[float, Range(1.0, 120.0)] | Dimension
    notes: Annotated[str, Text(400)]


class DrillDuration(TypedDict):
    minutes: Annotated[int, Range(5, 45)]
    blocks: Annotated[str, Text(120)]


class SuccessSignal(TypedDict):
    detector: DetectorName
    relation: Code
    text: Annotated[str, Text(300)]


class DrillSafety(TypedDict):
    heading: Literal['none', 'limited']
    physicalLoad: Literal['low', 'moderate', 'high']
    contact: Literal['normal', 'no_tackle']


class SourceReference(TypedDict):
    source: SourceId
    relation: Literal['adapted_from', 'inspired_by', 'pitchiq_original']
    note: Annotated[str, Text(300)]


DrillLines = Annotated[list[Annotated[str, Text(400)]], Len(1, 20)]


class Drill(TypedDict):
    drill_id: DrillId
    name: Annotated[str, Text(80)]
    development_level: Annotated[list[Level], Len(1, 7), Unique(), Increasing(strict=True)]
    age: AgeRange
    primary_skill: Code
    secondary_skills: list[Code]
    related_tactical_detector: Annotated[list[DetectorName], Len(1, 6), Unique()]
    recommended_format: Annotated[list[FormatName], Len(1, 5), Unique()]
    player_count: PlayerCount
    equipment: Annotated[list[Annotated[str, Text(80)]], Len(1, 20)]
    field_setup: FieldSetup
    duration: DrillDuration
    instructions: DrillLines
    coaching_points: DrillLines
    progression: DrillLines
    regression: DrillLines
    success_signals_next_match: Annotated[list[SuccessSignal], Len(1, 6)]
    safety: DrillSafety
    source_reference: Annotated[list[SourceReference], Len(1, 6)]
    origin: Literal['PITCHIQ-PROPOSAL']


class DrillLibrary(TypedDict):
    version: Literal[2]
    notes: Annotated[str, Text(2000)]
    drills: Annotated[list[Drill], Len(1, 200)]


# ================= progression (DESIGN 8) =================
class ProgressOpp(TypedDict):
    t0: Seconds
    t1: Seconds
    zone: ZoneName | None
    channel: Channel | None
    rel: dict[FieldName, float | bool | None]
    quality: Conf


class SkillRecord(TypedDict):
    opportunities: list[ProgressOpp]
    phaseSeconds: Duration
    eligibleShare: Share | None
    confidence: Conf


class ProgressEngine(TypedDict):
    measurement: Hex16
    version: Literal['2']


class ProgressContext(TypedDict):
    format: FormatName
    footage: Literal['broadcast', 'amateur', 'veo', 'unknown']
    phaseAvailable: bool
    level: Level
    ageGroup: AgeGroup


class ProgressEntry(TypedDict):
    id: PeId
    squad: SqId
    video: Annotated[str, Pattern(r'[A-Za-z0-9_.\-]{1,80}')]
    run: RunName
    team: Team
    range: Window
    matchDate: Date
    dateSource: Literal['coach', 'file_mtime']
    engine: ProgressEngine
    context: ProgressContext
    skills: dict[SkillId, SkillRecord]
    stale: NotRequired[bool]
    updated: ISO


class ProgressFile(TypedDict):
    version: Literal[1]
    entries: list[ProgressEntry]


class Squad(TypedDict):
    id: SqId
    name: Annotated[str, Text(60)]
    ageGroup: AgeGroup
    level: Level
    format: FormatSpec
    created: ISO


class SquadsFile(TypedDict):
    version: Literal[1]
    squads: list[Squad]


class Adjustment(TypedDict):             # a coach-adjusted suggested zone (DESIGN 10); never changes measurements
    id: AdjId
    findingKey: FindingKey
    evidence: EvId
    polygon: Polygon
    mover: PlayerId | None
    note: Annotated[str, Text(2000)] | None
    created: ISO


# ================= the v2 report (DESIGN 9.2: seven sections) =================
class ReportRange(TypedDict):
    t0: Seconds
    t1: Seconds
    seconds: Duration
    whole: bool


class Fingerprints(TypedDict):
    measurement: Hex64
    roles: Hex64
    interpretation: Hex64


class DetectorStatus(TypedDict):
    detector: DetectorName
    status: Literal['assessed', 'not_assessed', 'off']
    reason: AbstainReason | None
    eligibleSeconds: Duration
    phaseSeconds: Duration | None
    opportunities: Count


class Unavailable(TypedDict):
    category: Code
    reason: Annotated[str, Text(400)]
    unlock: Annotated[str, Text(300)] | None


class PossessionOverview(TypedDict):
    available: bool
    quality: Literal['good', 'limited', 'unavailable']
    reason: Annotated[str, Text(300)] | None
    roughShares: RoughShares | None
    coverage: Share | None
    caveat: Annotated[str, Text(400)] | None
    # fix round 1 (TEXT, additive; critic B-11 / N6, requested by SERVER + PAGE): where the shares come from
    source: NotRequired[Literal['engine', 'coach', 'mixed'] | None]


class Overview(TypedDict):
    notices: list[Notice]
    footage: Footage
    dataQuality: DataQuality
    scale: Scale
    possession: PossessionOverview
    formations: list[PeriodSummary]
    detectors: list[DetectorStatus]
    unavailable: list[Unavailable]
    text: Annotated[str, Text(1500)] | None


class DiagramRef(TypedDict):
    id: DiagramId
    kind: Literal['finding', 'formation', 'movement']
    finding: FndId | None
    evidence: EvId | None
    caption: Annotated[str, Text(200)]
    diagram: Diagram


class DrillFields(TypedDict):
    trains: Annotated[str, Text(300)]
    why: Annotated[str, Text(500)]
    moments: list[EvId]
    easier: Annotated[str, Text(400)]
    harder: Annotated[str, Text(400)]
    progress: Annotated[str, Text(400)]


class DrillPick(TypedDict):
    finding: FndId | None
    drill: DrillId
    optional: bool                       # True: for the next objective, no finding (labelled by drills.objective_label)
    fields: DrillFields


class SkillRow(TypedDict):
    skill: SkillId
    name: Annotated[str, Text(80)]
    state: Literal['NOT ENOUGH EVIDENCE', 'LEARNING', 'DEVELOPING', 'CONSISTENT', 'ADAPTABLE']
    opportunities: Count
    matches: Count
    trend: Literal['up', 'down', 'flat', 'unknown']


class Progression(TypedDict):
    squad: SqId | None
    skills: list[SkillRow]
    note: Annotated[str, Text(400)] | None


class NextObjective(TypedDict):
    skill: SkillId | None
    text: Annotated[str, Text(400)]
    fromMatch: bool
    finding: FndId | None


class SourceRef(TypedDict):
    id: SourceId
    title: Annotated[str, Text(300)]
    url: Url
    publisher: Annotated[str, Text(120)] | None


class Withheld(TypedDict):
    finding: FndId
    reason: Code


class Report(TypedDict):
    kind: Literal['coaching-report']
    version: Literal[2]
    team: Team
    teamName: Annotated[str, Text(80)] | None
    generated: ISO | None
    engine: EngineStamp
    fingerprints: Fingerprints
    run: RunName | None
    video: VideoInfo
    range: ReportRange
    segment: Window
    profile: Profile
    overview: Overview                   # 1 MATCH OVERVIEW
    strengths: list[Finding]             # 2 WHAT WENT WELL
    needsWork: list[Finding]             # 3 WHAT NEEDS WORK: work_on / try_this ...
    reviews: list[Finding]               # ... then "Possible issue to review"
    described: list[Finding]             # context only (verdict describe), shown in the overview
    keyMoments: list[Evidence]           # 4 KEY MOMENTS (chronological)
    diagrams: list[DiagramRef]           # 5 TACTICAL DIAGRAMS
    alternatives: list[Alternative]
    noAlternatives: list[NoAlternative]
    drills: list[DrillPick]              # 6 TRAINING DRILLS
    progression: Progression | None      # 7 DEVELOPMENT PROGRESSION (filled at serve time; None in the cache)
    nextObjective: NextObjective
    sources: list[SourceRef]
    withheld: list[Withheld]


KINDS = {
    'settings': CoachingSettings, 'team_settings': TeamSettings, 'profile': Profile, 'timeline': TimelineSummary,
    'opportunity': Opportunity, 'moment': Moment, 'detector_coverage': DetectorCoverage, 'obs': ObsFile,
    'role_record': RoleRecord, 'rolebook': RoleBook, 'roles': RolesFile, 'claim': Claim, 'finding': Finding,
    'alternative': Alternative, 'no_alternative': NoAlternative, 'element': Element, 'diagram': Diagram,
    'overlay': VideoOverlay, 'annotation': Annotation, 'evidence': Evidence, 'drill': Drill,
    'drill_library': DrillLibrary, 'progress_entry': ProgressEntry, 'progress_file': ProgressFile,
    'squads': SquadsFile, 'adjustment': Adjustment, 'report': Report, 'coach_spans': CoachSpans}


# ---------- the validator ----------
class SchemaError(ValueError):
    """A value does not follow the contract; .problems lists why."""

    def __init__(self, kind, problems):
        self.kind = kind
        self.problems = list(problems)
        head = '; '.join(self.problems[:8])
        more = f' (+{len(self.problems)-8} more)' if len(self.problems) > 8 else ''
        super().__init__(f'{kind} does not follow the coaching schema: {head}{more}')


_RULES = {}
_HINTS = {}
_UNION = (typing.Union, types.UnionType)


def rule(*record_types):
    """Register a cross-field rule fn(value, path, out) for record types; it runs only when the record's own fields
    passed (so it may index them freely)."""
    def wrap(fn):
        for t in record_types:
            _RULES.setdefault(t, []).append(fn)
        return fn
    return wrap


def _hints(td):
    h = _HINTS.get(td)
    if h is None:
        h = {}
        for k, v in typing.get_type_hints(td, include_extras=True).items():
            while get_origin(v) in (NotRequired, typing.Required):
                v = get_args(v)[0]
            h[k] = v
        _HINTS[td] = h
    return h


def _discriminator(td):
    t = _hints(td).get('type')
    return get_args(t) if get_origin(t) is Literal else ()


def _is_number(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _json(value, path, out, depth=0):
    """Any plain JSON value with finite numbers."""
    if depth > 40:
        out.append(f'{path}: nested too deeply')
    elif value is None or isinstance(value, (bool, str)):
        return
    elif isinstance(value, int):
        return
    elif isinstance(value, float):
        if not math.isfinite(value):
            out.append(f'{path}: {value!r} is not a finite number')
    elif isinstance(value, (list, tuple)):
        for i, v in enumerate(value):
            _json(v, f'{path}[{i}]', out, depth+1)
    elif isinstance(value, dict):
        for k, v in value.items():
            if not isinstance(k, str):
                out.append(f'{path}: key {k!r} is not a string')
            else:
                _json(v, f'{path}.{k}', out, depth+1)
    else:
        out.append(f'{path}: {type(value).__name__} is not a JSON value')


def _type_name(spec):
    if isinstance(spec, type):
        return spec.__name__
    return str(spec).replace('typing.', '')


def _check(spec, value, path, out):
    if len(out) > MAX_PROBLEMS:
        return
    origin = get_origin(spec)
    if origin is Annotated:
        base, *marks = get_args(spec)
        n = len(out)
        _check(base, value, path, out)
        if len(out) == n:
            for m in marks:
                m.check(value, path, out)
        return
    if origin in (NotRequired, typing.Required):
        _check(get_args(spec)[0], value, path, out)
        return
    if spec is Any:
        _json(value, path, out)
        return
    if spec is type(None):
        if value is not None:
            out.append(f'{path}: expected null, got {type(value).__name__}')
        return
    if origin is Literal:
        if not any(value == a and type(value) is type(a) for a in get_args(spec)):
            out.append(f'{path}: {value!r} is not one of {", ".join(repr(a) for a in get_args(spec))}')
        return
    if origin in _UNION:
        args = get_args(spec)
        if value is None:
            if type(None) not in args:
                out.append(f'{path}: null is not allowed')
            return
        cands = [a for a in args if a is not type(None)]
        if isinstance(value, dict) and 'type' in value and cands and all(typing.is_typeddict(a) for a in cands):
            for a in cands:
                if value['type'] in _discriminator(a):
                    _check(a, value, path, out)
                    return
            out.append(f'{path}.type: unknown element type {value.get("type")!r}')
            return
        best = None
        for a in cands:
            sub = []
            _check(a, value, path, sub)
            if not sub:
                return
            if best is None or len(sub) < len(best):
                best = sub
        out.extend(best or [f'{path}: no alternative of {_type_name(spec)} matched'])
        return
    if origin in (list, tuple):
        if not isinstance(value, (list, tuple)):
            out.append(f'{path}: expected a list, got {type(value).__name__}')
            return
        (item,) = get_args(spec)[:1]
        for i, v in enumerate(value):
            _check(item, v, f'{path}[{i}]', out)
        return
    if origin is dict:
        if not isinstance(value, dict):
            out.append(f'{path}: expected an object, got {type(value).__name__}')
            return
        kspec, vspec = get_args(spec)
        for k, v in value.items():
            if not isinstance(k, str):
                out.append(f'{path}: key {k!r} is not a string')
                continue
            _check(kspec, k, f'{path} key', out)
            _check(vspec, v, f'{path}.{k}', out)
        return
    if typing.is_typeddict(spec):
        if not isinstance(value, dict):
            out.append(f'{path}: expected an object ({spec.__name__}), got {type(value).__name__}')
            return
        hints = _hints(spec)
        n = len(out)
        for k in sorted(spec.__required_keys__):
            if k not in value:
                out.append(f'{path}: missing {k!r}')
        for k, v in value.items():
            if k not in hints:
                out.append(f'{path}: unknown key {k!r} ({spec.__name__} is closed)')
            else:
                _check(hints[k], v, f'{path}.{k}', out)
        if len(out) == n:
            for fn in _RULES.get(spec, ()):
                fn(value, path, out)
        return
    if spec is bool:
        if not isinstance(value, bool):
            out.append(f'{path}: expected true/false, got {type(value).__name__}')
        return
    if spec is int:
        if not isinstance(value, int) or isinstance(value, bool):
            out.append(f'{path}: expected an integer, got {type(value).__name__}')
        return
    if spec is float:
        if not _is_number(value):
            out.append(f'{path}: expected a number, got {type(value).__name__}')
        elif isinstance(value, float) and not math.isfinite(value):
            out.append(f'{path}: {value!r} is not a finite number')
        return
    if spec is str:
        if not isinstance(value, str):
            out.append(f'{path}: expected a string, got {type(value).__name__}')
        return
    raise TypeError(f'schema has no check for {spec!r}')


def validate(kind, value):
    """Every problem with value as a record of kind (a name of KINDS or a record type): readable strings, [] when it
    follows the contract."""
    spec = KINDS[kind] if isinstance(kind, str) else kind
    name = kind if isinstance(kind, str) else getattr(kind, '__name__', 'value')
    out = []
    _check(spec, value, name, out)
    if len(out) > MAX_PROBLEMS:
        out = out[:MAX_PROBLEMS]+[f'... and more problems (stopped after {MAX_PROBLEMS})']
    return out


def check(kind, value):
    """value, or SchemaError listing its problems."""
    problems = validate(kind, value)
    if problems:
        raise SchemaError(kind if isinstance(kind, str) else getattr(kind, '__name__', 'value'), problems)
    return value


def is_valid(kind, value):
    return not validate(kind, value)


# ---------- helpers every role uses ----------
def format_name(fmt):
    """'7v7' from {'players': 7, 'goalkeeper': True}."""
    return f'{int(fmt["players"])}v{int(fmt["players"])}'


def outfield(fmt):
    """Outfield players of a format: players minus the keeper (4v4 has none by default)."""
    return int(fmt['players'])-(1 if fmt.get('goalkeeper') else 0)


def _norm_part(p):
    if isinstance(p, float):
        return f'{round(p, 1):.1f}'
    return str(p)


def record_id(prefix, *parts):
    """prefix-<12 hex of sha1('|'.join(parts))>, floats rounded to 0.1 (a regeneration that moves an edge by a sample
    keeps the id). Unlike episodes.stable_id, the engine version is not in it (DESIGN 6.1)."""
    text = '|'.join(_norm_part(p) for p in parts)
    return f'{prefix}-{hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]}'


def opportunity_key(detector, team, t0, t1, variant=None):
    """'D1|A|812.4|816.0' (or 'D3|A|LB-side|812.4|840.0' with a variant): engine-independent."""
    mid = f'{variant}|' if variant else ''
    return f'{DETECTOR_CODE[detector]}|{team}|{mid}{float(t0):.1f}|{float(t1):.1f}'


def finding_key(detector, team, family, t0, t1):
    """'D3|A|LB-side|812.4|840.0': detector, team, the finding's family (a side, a zone or 'team') and the span of its
    opportunities; feedback and progression attach by it, so it never contains the engine version or the level."""
    return f'{DETECTOR_CODE[detector]}|{team}|{family}|{float(t0):.1f}|{float(t1):.1f}'


def seek_of(frame, fps):
    """The seek time of a frame: its middle, (frame + 0.5) / fps, rounded to 4 decimals (invariant 7)."""
    return round((int(frame)+.5)/float(fps), SEEK_DECIMALS)


def tier_settings(team_settings):
    """The settings of one team split by cache tier (DESIGN 1.2): {'measurement': {format, field}, 'role': {roles,
    formation}, 'interpretation': {ageGroup, playingLevel, style, federation}}, missing fields at TIER_DEFAULTS. The
    measurement tier is handed only its part, so a profile change can never reach the observations (invariant 2)."""
    ts = team_settings if isinstance(team_settings, dict) else {}
    out = {}
    for tier, fields in TIER_FIELDS.items():
        out[tier] = {f: copy.deepcopy(ts[f]) if f in ts else copy.deepcopy(TIER_DEFAULTS[f]) for f in fields}
    return out


def stale_tiers(old_team_settings, new_team_settings):
    """The tiers a settings change makes stale, cascading downwards: a measurement change rebuilds everything, a role
    change roles and interpretation, an interpretation change only the interpretation; [] when nothing a fingerprint
    reads changed (e.g. only the squad name)."""
    a, b = tier_settings(old_team_settings), tier_settings(new_team_settings)
    for k, tier in enumerate(TIER_ORDER):
        if json.dumps(a[tier], sort_keys=True) != json.dumps(b[tier], sort_keys=True):
            return list(TIER_ORDER[k:])
    return []


def team_block(coaching, team):
    """The team's block of a CoachingSettings dict (None when absent); coaching may be None or {}."""
    if coaching is None:
        return None
    if not isinstance(coaching, dict):
        raise TypeError('coaching settings must be a dict like {"version": 1, "teams": {"A": ..., "B": ...}}')
    teams = coaching.get('teams') or {}
    if not isinstance(teams, dict):
        raise TypeError('coaching settings "teams" must be an object')
    return teams.get(team)


# ---------- invariant 4: events PitchIQ does not measure ----------
HYPOTHETICAL = re.compile(r'\b(?:if|could|would|might|potentially|possible|possibly)\b', re.I)
EVENT_PATTERNS = (
    ('turnover', re.compile(r'\bturnovers?\b', re.I), False),
    ('communication', re.compile(r'\bcommunicat\w*', re.I), False),
    ('possession change', re.compile(r'\b(?:lost|lose|loses|losing|won|win|wins|winning|regain(?:s|ed|ing)?)\s+'
                                     r'(?:the\s+)?(?:ball|possession)\b', re.I), True),
    ('counter-attack', re.compile(r'\bcounter[- ]?attack(?:s|ed)?\b(?!-?ing)', re.I), True),
    ('pass', re.compile(r'\b(?:pass|passes|passed)\b', re.I), True),
)


def event_problems(text):
    """The unmeasured events a text states (invariant 4; possession docs rule 5; spec Part 10): turnovers and
    communication anywhere; a change of possession, a counter-attack or a pass outside a hypothetical sentence
    ("if the ball had been lost ..." is allowed). [] when clean. TEXT's validator is the full gate; this is the
    contract's safety net run on every finding text and caption of a report."""
    out = []
    for sentence in re.split(r'(?<=[.!?])\s+', text or ''):
        hypothetical = HYPOTHETICAL.search(sentence) is not None
        for name, pat, allowed_if_hypothetical in EVENT_PATTERNS:
            m = pat.search(sentence)
            if m and not (allowed_if_hypothetical and hypothetical):
                out.append(f'{name}: "{m.group(0)}"')
    return out


# ---------- cross-field rules ----------
def _spans_ok(spans, path, out, what='spans'):
    prev = None
    for k, (a, b) in enumerate(spans):
        if b < a:
            out.append(f'{path}[{k}]: {what} end {b} before start {a}')
        if prev is not None and a < prev-1e-6:
            out.append(f'{path}[{k}]: {what} overlap or are out of order')
        prev = b


@rule(TeamSettings)
def _team_settings(v, path, out):
    fmt = v['format']
    if fmt['players'] in (7, 9, 11) and not fmt['goalkeeper']:
        out.append(f'{path}.format: {fmt["players"]}v{fmt["players"]} always has a goalkeeper')
    if v['field'] is not None and v['field']['width'] > v['field']['length']:
        out.append(f'{path}.field: width larger than length')
    allowed = FORMATIONS[fmt['players']]
    if v['formation']['intended'] is not None and v['formation']['intended'] not in allowed:
        out.append(f'{path}.formation.intended: {v["formation"]["intended"]} is not offered for '
                   f'{format_name(fmt)} ({", ".join(allowed)})')
    for k, p in enumerate(v['formation']['periods']):
        if p['formation'] not in allowed:
            out.append(f'{path}.formation.periods[{k}]: {p["formation"]} is not offered for {format_name(fmt)}')
    _spans_ok([(p['t0'], p['t1']) for p in v['formation']['periods']], f'{path}.formation.periods', out, 'periods')
    for pid, spans in v['roles'].items():
        lo = [(-1.0 if s['from'] is None else s['from'], 1e7 if s['to'] is None else s['to']) for s in spans]
        _spans_ok(lo, f'{path}.roles.{pid}', out, 'role spans')


@rule(Profile)
def _profile(v, path, out):
    missing = [d for d in DETECTORS if d not in v['detectors']]
    if missing:
        out.append(f'{path}.detectors: missing {", ".join(missing)}')
    for d, p in v['detectors'].items():
        if p['enabled'] != bool(p['verdicts']):
            out.append(f'{path}.detectors.{d}: enabled must equal "has verdicts"')
        if v['level'] <= 2 and 'work_on' in p['verdicts']:
            out.append(f'{path}.detectors.{d}: level {v["level"]} frames work items as try_this, never work_on')
        if v['level'] >= 4 and 'try_this' in p['verdicts']:
            out.append(f'{path}.detectors.{d}: try_this is a level 1-3 framing')
    f = v['format']
    if f['outfield'] != outfield(f) or f['name'] != format_name(f):
        out.append(f'{path}.format: outfield/name do not match players and goalkeeper')
    if v['evidence']['minFailureShare'] is None and v['level'] != 7:
        out.append(f'{path}.evidence.minFailureShare: only level 7 is style-relative (null)')
    if AGE_ORDER[v['ageGroup']] <= AGE_ORDER['U11'] and v['safety']['heading'] != 'none':
        out.append(f'{path}.safety.heading: no heading at {v["ageGroup"]} (DESIGN 2.4)')
    for part in ('objectives', 'principles', 'notExpected'):
        for k, g in enumerate(v[part]):
            _guidance(g, f'{path}.{part}[{k}]', out)


def _guidance(g, path, out):
    for q in re.findall(r'"([^"]*)"', g['text']):
        if len(q) > 60:
            out.append(f'{path}.text: a quoted phrase of {len(q)} characters (at most 60: paraphrase, never copy)')
    if g['tag'] == 'FEDERATION' and not g['sources']:
        out.append(f'{path}: a FEDERATION item needs its source ids')


@rule(TimelineSummary)
def _timeline(v, path, out):
    if v['t1'] < v['t0'] or v['frame1'] < v['frame0']:
        out.append(f'{path}: segment ends before it starts')
    _spans_ok([(s['t0'], s['t1']) for s in v['phaseSpans']], f'{path}.phaseSpans', out)
    rs = v['phaseQuality']['roughShares']
    if rs is not None:
        for team, x in rs.items():
            if abs(x*20-round(x*20)) > 1e-6:
                out.append(f'{path}.phaseQuality.roughShares.{team}: {x} is not rounded to 5% (possession rule 7)')
    coach = v['phaseQuality'].get('coach')
    if not v['phaseQuality']['available']:
        # Without the engine only coach-marked spans may name own/opp time (user decision 1).
        named = v['phaseSeconds']['own']+v['phaseSeconds']['opp']
        by_coach = (coach['own']+coach['opp']) if coach else 0.0
        if named > by_coach+1e-6 or \
                any(s['state'] in ('own', 'opp') and s.get('source') != 'coach' for s in v['phaseSpans']):
            out.append(f'{path}: possession is unavailable but own/opp phase time is named (not by the coach)')
        if rs is not None:
            out.append(f'{path}.phaseQuality.roughShares: shares without an available possession record')
    if coach is None and any(s.get('source') == 'coach' for s in v['phaseSpans']):
        out.append(f'{path}.phaseSpans: coach spans named without phaseQuality.coach')
    if v['format'] != '11v11' and v['scale']['source'] == 'assumed' and v['scale']['metresUsable']:
        out.append(f'{path}.scale: an assumed scale on a small format is not usable for body-scale distances')


@rule(CoachSpans)
def _coach_spans(v, path, out):
    for k, s in enumerate(v['spans']):
        if not s['t1'] > s['t0']:
            out.append(f'{path}.spans[{k}]: ends at {s["t1"]}, not after its start {s["t0"]}')
    _spans_ok([(s['t0'], s['t1']) for s in v['spans']], f'{path}.spans', out)


@rule(Opportunity)
def _opportunity(v, path, out):
    if v['t1'] < v['t0'] or v['frame1'] < v['frame0']:
        out.append(f'{path}: ends before it starts')
    key = opportunity_key(v['detector'], v['team'], v['t0'], v['t1'], v.get('variant'))
    if v['key'] != key:
        out.append(f'{path}.key: {v["key"]!r} is not {key!r} (schema.opportunity_key)')
    if v['tier'] not in DETECTOR_TIERS[v['detector']]:
        out.append(f'{path}.tier: {v["detector"]} opportunities are made in the {"/".join(DETECTOR_TIERS[v["detector"]])}'
                   ' tier')
    if v['abstain'] is None:
        if v['phase']['state'] not in DETECTOR_PHASES[v['detector']]:
            out.append(f'{path}.phase.state: {v["detector"]} is assessed only in phase '
                       f'{"/".join(DETECTOR_PHASES[v["detector"]])}')
        if v['detector'] in SHAPE_DETECTORS and v['phase']['state'] in ('own', 'opp') and \
                (v['phase']['spanSeconds'] is None or v['phase']['spanSeconds'] < MIN_SHAPE_SPAN_S):
            out.append(f'{path}.phase.spanSeconds: shape rules need a phase span of >= {MIN_SHAPE_SPAN_S:g} s')
        if v['detector'] == 'wide_defender_advance' and v['phase']['nearChange']:
            out.append(f'{path}.phase.nearChange: D3 opportunities exclude the change windows')
        if v['detector'] == 'pressure_cover' and v['actors']['carrier'] is None:
            out.append(f'{path}.actors.carrier: D1 runs only on carrier samples')
    rel = v['relations']
    if 't' not in rel:
        out.append(f'{path}.relations: needs the sample times "t"')
        return
    n = v['quality']['samples']
    for name, arr in rel.items():
        if len(arr) != n:
            out.append(f'{path}.relations.{name}: {len(arr)} values for {n} samples')
    t = rel['t']
    if not all(_is_number(x) for x in t):
        out.append(f'{path}.relations.t: sample times must be numbers')
    else:
        if any(b <= a for a, b in zip(t, t[1:])):
            out.append(f'{path}.relations.t: sample times must increase')
        if t and (t[0] < v['t0']-.01 or t[-1] > v['t1']+.01):
            out.append(f'{path}.relations.t: sample times outside [t0, t1]')
    ids = v['actors']['primary']+v['actors']['supporting']
    if len(set(ids)) != len(ids):
        out.append(f'{path}.actors: a player is both primary and supporting (or repeated)')
    known = set(ids)|set(v['actors']['opponents'])
    for pid in v['roles']:
        if pid not in known:
            out.append(f'{path}.roles.{pid}: not one of the actors')


@rule(Moment)
def _moment(v, path, out):
    if v['fps'] > 0 and abs(v['seek']-seek_of(v['frame'], v['fps'])) > 1e-4:
        out.append(f'{path}.seek: {v["seek"]} is not (frame + 0.5) / fps = {seek_of(v["frame"], v["fps"])}')
    if v['scene']['frame'] != v['frame']:
        out.append(f'{path}.scene.frame: the scene must be the moment\'s frame')
    missing = [p for p in v['players'] if p not in v['scene']['players']]
    if missing:
        out.append(f'{path}.players: {", ".join(missing)} not observed in the scene')
    if v['before'] is not None and v['before']['t'] >= v['t']:
        out.append(f'{path}.before: must be earlier than the moment')


def _tier_file(v, path, out, tier):
    ids = [o['id'] for o in v['opportunities']]
    if len(set(ids)) != len(ids):
        out.append(f'{path}.opportunities: repeated ids')
    for k, o in enumerate(v['opportunities']):
        if o['team'] != v['team']:
            out.append(f'{path}.opportunities[{k}]: team {o["team"]} in a team {v["team"]} file')
        if o['tier'] != tier:
            out.append(f'{path}.opportunities[{k}]: a {o["tier"]}-tier opportunity in the {tier} file')
    opp = set(ids)
    mids = [m['id'] for m in v['moments']]
    if len(set(mids)) != len(mids):
        out.append(f'{path}.moments: repeated ids')
    for k, m in enumerate(v['moments']):
        if m['opportunity'] not in opp:
            out.append(f'{path}.moments[{k}]: opportunity {m["opportunity"]} is not in this file')
    for d, c in v['detectors'].items():
        if c['detector'] != d:
            out.append(f'{path}.detectors.{d}: detector field says {c["detector"]}')


@rule(ObsFile)
def _obs(v, path, out):
    if v['engine']['measurement'] != v['fingerprint'][:16]:
        out.append(f'{path}.engine.measurement: not the fingerprint\'s first 16 hex')
    if v['timeline']['team'] != v['team']:
        out.append(f'{path}.timeline.team: not the file\'s team')
    _tier_file(v, path, out, 'measurement')
    _spans_ok([(s['t0'], s['t1']) for s in v['frameTiers']], f'{path}.frameTiers', out)


@rule(RolesFile)
def _roles_file(v, path, out):
    if v['engine']['measurement'] != v['measurement'][:16] or v['engine'].get('roles') != v['fingerprint'][:16]:
        out.append(f'{path}.engine: measurement/roles must be the first 16 hex of measurement/fingerprint')
    if v['rolebook']['team'] != v['team']:
        out.append(f'{path}.rolebook.team: not the file\'s team')
    _tier_file(v, path, out, 'role')


@rule(RoleRecord)
def _role_record(v, path, out):
    if v['t1'] < v['t0']:
        out.append(f'{path}: ends before it starts')
    if v['source'] == 'none' and v['role'] is not None:
        out.append(f'{path}.role: source none carries no role')


@rule(RoleBook)
def _rolebook(v, path, out):
    if v['format'] != '11v11':
        if v['model'] != 'relative':
            out.append(f'{path}.model: {v["format"]} never uses the 11v11 team_shape templates (DESIGN 2.5)')
        for k, r in enumerate(v['records']):
            if r['source'] in TEMPLATE_SOURCES:
                out.append(f'{path}.records[{k}].source: {r["source"]} roles come from 11v11 templates only')


@rule(Finding)
def _finding(v, path, out):
    verdict, level = v['verdict'], v['level']
    if verdict == 'work_on' and level <= 2:
        out.append(f'{path}.verdict: level {level} frames work items as try_this')
    if verdict == 'try_this' and level >= 4:
        out.append(f'{path}.verdict: try_this is a level 1-3 framing')
    if verdict in CRITICISM:
        failed = [g for g in GUARDRAILS if v['guardrails'][g] == 'fail']
        if failed:
            out.append(f'{path}.guardrails: {", ".join(failed)} failed, so the finding is at most a review (DESIGN 4.8)')
    a = v['aggregate']
    if verdict == 'strength' and (a['demonstrated'] < 1 or a['distinctDemonstratedSpells'] < 1):
        out.append(f'{path}.aggregate: a strength needs demonstrated spells, never an absence of failures')
    if a['demonstrated']+a['notDemonstrated']+a['unclear'] != a['opportunities']:
        out.append(f'{path}.aggregate: demonstrated + notDemonstrated + unclear != opportunities')
    if a['distinctFailingSpells'] > a['notDemonstrated'] or a['distinctDemonstratedSpells'] > a['demonstrated']:
        out.append(f'{path}.aggregate: more distinct spells than opportunities with that outcome')
    if len(v['opportunities']) != a['opportunities']:
        out.append(f'{path}.opportunities: {len(v["opportunities"])} listed, the aggregate counts {a["opportunities"]}')
    if verdict != 'describe' and not v['timestamps']:
        out.append(f'{path}.timestamps: a {verdict} needs its evidence moments')
    evs = [ts['evidence'] for ts in v['timestamps']]
    if len(set(evs)) != len(evs):
        out.append(f'{path}.timestamps: repeated evidence')
    for k, ts in enumerate(v['timestamps']):
        if ts['opportunity'] not in v['opportunities']:
            out.append(f'{path}.timestamps[{k}].opportunity: not one of the finding\'s opportunities')
    if not v['key'].startswith(DETECTOR_CODE[v['observation']]+'|'+v['team']+'|'):
        out.append(f'{path}.key: must start with the detector code and the team')
    for k, c in enumerate(v['claims']):
        if c.get('tag') == 'FEDERATION' and not c.get('source'):
            out.append(f'{path}.claims[{k}]: a FEDERATION claim names its source id')


@rule(Alternative)
def _alternative(v, path, out):
    if not v['improves']['after'] or v['improves']['before']:
        out.append(f'{path}.improves: an alternative turns the relation from false to true')
    bad = [k for k, ok in v['checks'].items() if not ok]
    if bad:
        out.append(f'{path}.checks: {", ".join(bad)} failed, so no zone may be drawn (DESIGN 5.1 step 7)')
    r = v['reach']
    if abs(r['radiusM']-r['horizonS']*r['speedMS']) > .05:
        out.append(f'{path}.reach.radiusM: must be horizonS x speedMS')
    if v['zone']['kind'] == 'point' and v['zone']['radiusM'] is None:
        out.append(f'{path}.zone.radiusM: a point is a small dashed circle with a radius')
    if v['display'] == 'point' and v['zone']['kind'] != 'point':
        out.append(f'{path}.display: point display needs a point zone')
    if any(r_['mover'] == v['mover'] for r_ in v['rejected']):
        out.append(f'{path}.rejected: the chosen mover is also listed as rejected')
    _in_unit([v['from']]+v['zone']['polygon'], f'{path}', out)


def _in_unit(points, path, out):
    for p in points:
        if not all(-NORM_MARGIN <= c <= 1+NORM_MARGIN for c in p):
            out.append(f'{path}: point {p} is outside the normalized pitch')
            return


def _element_points(el):
    pts = []
    for k in ('a', 'b', 'at', 'from', 'to', 'pa', 'pb', 'pitch'):
        if isinstance(el.get(k), list):
            pts.append(el[k][:2])
    if 'x' in el and 'y' in el:
        pts.append([el['x'], el['y']])
    pts += [p[:2] for p in el.get('polygon') or ()]
    pts += [p[:2] for p in el.get('points') or ()]
    return pts


def _budget(elements, path, out):
    counts = {}
    for el in elements:
        counts[el['type']] = counts.get(el['type'], 0)+1
    for t, n in counts.items():
        if n > ELEMENT_BUDGET.get(t, 10**6):
            out.append(f'{path}: {n} {t} elements (at most {ELEMENT_BUDGET[t]}, DESIGN 5.5)')
    if counts.get('suggested'):
        out.append(f'{path}: the v1 "suggested" circle is render-only for old reports; v2 draws suggestedZone')


@rule(Diagram)
def _diagram(v, path, out):
    _budget(v['elements'], f'{path}.elements', out)
    for k, el in enumerate(v['elements']):
        if 'diagram' not in TIER_ELEMENTS[el['type']]:
            out.append(f'{path}.elements[{k}]: {el["type"]} is not drawn on diagrams')
        if el['type'] != 'measure' or not el.get('bracket'):
            _in_unit(_element_points(el), f'{path}.elements[{k}]', out)


@rule(VideoOverlay)
def _overlay(v, path, out):
    _budget(v['elements'], f'{path}.elements', out)
    for k, el in enumerate(v['elements']):
        if v['tier'] not in TIER_ELEMENTS[el['type']]:
            out.append(f'{path}.elements[{k}]: {el["type"]} is not drawn at tier {v["tier"]} (DESIGN 5.4)')


@rule(Evidence)
def _evidence(v, path, out):
    if v['fps'] > 0 and abs(v['seek']-seek_of(v['frame'], v['fps'])) > 1e-4:
        out.append(f'{path}.seek: {v["seek"]} is not (frame + 0.5) / fps = {seek_of(v["frame"], v["fps"])}')
    if v['videoOverlay'] is not None and v['videoOverlay']['tier'] != v['tier']:
        out.append(f'{path}.videoOverlay.tier: not the evidence tier')
    a = v['annotation']
    if a is not None:
        if not a['window']['t0'] <= a['keyTime'] <= a['window']['t1']:
            out.append(f'{path}.annotation.window: does not contain the key time')
        if a['keyFrame'] != v['frame']:
            out.append(f'{path}.annotation.keyFrame: not the evidence frame')
        if v['finding'] is not None and a['finding'] != v['finding']:
            out.append(f'{path}.annotation.finding: not the evidence finding')
    zones = [el for spec in (v['diagram'], v['videoOverlay']) if spec for el in spec['elements']
             if el['type'] == 'suggestedZone']
    for el in zones:
        if el['alternative'] != v['alternative']:
            out.append(f'{path}: a suggestedZone of {el["alternative"]} on evidence of alternative {v["alternative"]}')


@rule(Drill)
def _drill(v, path, out):
    if AGE_ORDER[v['age']['min']] > AGE_ORDER[v['age']['max']]:
        out.append(f'{path}.age: min after max')
    if v['player_count']['min'] > v['player_count']['max']:
        out.append(f'{path}.player_count: min above max')
    for k, s in enumerate(v['success_signals_next_match']):
        if s['detector'] not in v['related_tactical_detector']:
            out.append(f'{path}.success_signals_next_match[{k}]: {s["detector"]} is not a related detector')
    if v['safety']['heading'] != 'none' and AGE_ORDER[v['age']['min']] <= AGE_ORDER['U11']:
        out.append(f'{path}.safety.heading: no heading for ages up to U11 (DESIGN 2.4)')


@rule(DrillLibrary)
def _library(v, path, out):
    ids = [d['drill_id'] for d in v['drills']]
    if len(set(ids)) != len(ids):
        out.append(f'{path}.drills: repeated drill_id')


@rule(ProgressEntry)
def _progress(v, path, out):
    if v['range']['t1'] < v['range']['t0']:
        out.append(f'{path}.range: ends before it starts')
    for skill, rec in v['skills'].items():
        for k, o in enumerate(rec['opportunities']):
            if o['t1'] < o['t0']:
                out.append(f'{path}.skills.{skill}.opportunities[{k}]: ends before it starts')


@rule(Report)
def _report(v, path, out):
    e, fp = v['engine'], v['fingerprints']
    if e['measurement'] != fp['measurement'][:16] or e.get('roles') != fp['roles'][:16] or \
            e.get('interpretation') != fp['interpretation'][:16]:
        out.append(f'{path}.engine: must carry the first 16 hex of the three fingerprints')
    level = v['profile']['level']
    if v['profile']['team'] != v['team']:
        out.append(f'{path}.profile.team: not the report\'s team')
    sections = {'strengths': ('strength',), 'needsWork': CRITICISM, 'reviews': ('review',), 'described': ('describe',)}
    findings = []
    for name, verdicts in sections.items():
        for k, f in enumerate(v[name]):
            fpath = f'{path}.{name}[{k}]'
            findings.append(f)
            if f['verdict'] not in verdicts:
                out.append(f'{fpath}.verdict: {f["verdict"]} does not belong in {name}')
            if f['level'] != level or f['team'] != v['team']:
                out.append(f'{fpath}: level/team differ from the report\'s profile')
            if f['text'] is None:
                out.append(f'{fpath}.text: a reported finding has its text')
            else:
                for field in ('headline', 'explanation', 'short', 'action'):
                    for p in event_problems(f['text'][field]):
                        out.append(f'{fpath}.text.{field}: names an unmeasured event ({p})')
    if len(v['needsWork']) > v['profile']['evidence']['maxWorkItems']:
        out.append(f'{path}.needsWork: {len(v["needsWork"])} work items, the level allows '
                   f'{v["profile"]["evidence"]["maxWorkItems"]}')
    ids = [f['id'] for f in findings]
    if len(set(ids)) != len(ids):
        out.append(f'{path}: a finding appears twice')
    fids = set(ids)
    moments = {m['id']: m for m in v['keyMoments']}
    if len(moments) != len(v['keyMoments']):
        out.append(f'{path}.keyMoments: repeated evidence ids')
    times = [m['time'] for m in v['keyMoments']]
    if times != sorted(times):
        out.append(f'{path}.keyMoments: not in chronological order')
    for f in findings:
        for ts in f['timestamps']:
            m = moments.get(ts['evidence'])
            if m is None:
                out.append(f'{path}: finding {f["id"]} cites {ts["evidence"]}, which is not a key moment')
            elif m['frame'] != ts['frame'] or abs(m['seek']-ts['seek']) > 1e-4 or m['opportunity'] != ts['opportunity']:
                out.append(f'{path}: finding {f["id"]} timestamp of {ts["evidence"]} differs from the key moment')
    for k, m in enumerate(v['keyMoments']):
        if m['finding'] is not None and m['finding'] not in fids:
            out.append(f'{path}.keyMoments[{k}].finding: not a reported finding')
        for field in ('caption',):
            for p in event_problems(m[field]):
                out.append(f'{path}.keyMoments[{k}].caption: names an unmeasured event ({p})')
        if m['annotation'] is not None:
            for p in event_problems(m['annotation']['caption']):
                out.append(f'{path}.keyMoments[{k}].annotation.caption: names an unmeasured event ({p})')
    for k, d in enumerate(v['drills']):
        if d['finding'] is None and not d['optional']:
            out.append(f'{path}.drills[{k}]: a drill without a finding is optional (no random exercises)')
        if d['finding'] is not None and d['finding'] not in fids:
            out.append(f'{path}.drills[{k}].finding: not a reported finding')
        for ev in d['fields']['moments']:
            if ev not in moments:
                out.append(f'{path}.drills[{k}].fields.moments: {ev} is not a key moment')
    for k, a in enumerate(v['alternatives']):
        if a['finding'] not in fids:
            out.append(f'{path}.alternatives[{k}].finding: not a reported finding')
    for k, d in enumerate(v['diagrams']):
        if d['finding'] is not None and d['finding'] not in fids:
            out.append(f'{path}.diagrams[{k}].finding: not a reported finding')


def cross_check(report, obs, roles=None):
    """Invariant 3 across files: every key moment of the report is a moment of the observation file (or of the role
    file, for role-tier opportunities) at the same frame and seek, of the same opportunity; every opportunity a finding
    aggregates exists there; the report's fingerprints are those of the files. [] when consistent."""
    out = []
    moments = {m['id']: m for m in obs['moments']}
    opps = {o['id'] for o in obs['opportunities']}
    if roles is not None:
        moments.update({m['id']: m for m in roles['moments']})
        opps |= {o['id'] for o in roles['opportunities']}
        if report['fingerprints']['roles'] != roles['fingerprint']:
            out.append('report.fingerprints.roles: not the role file\'s fingerprint')
        if roles['measurement'] != obs['fingerprint']:
            out.append('roles.measurement: the role file was built from another observation file')
    if report['fingerprints']['measurement'] != obs['fingerprint']:
        out.append('report.fingerprints.measurement: not the observation file\'s fingerprint')
    for k, ev in enumerate(report['keyMoments']):
        m = moments.get(ev['id'])
        if m is None:
            out.append(f'report.keyMoments[{k}]: {ev["id"]} is not a measured moment (invariant 3)')
            continue
        if (m['frame'], m['opportunity']) != (ev['frame'], ev['opportunity']) or abs(m['seek']-ev['seek']) > 1e-4:
            out.append(f'report.keyMoments[{k}]: {ev["id"]} differs from the measured moment (frame, seek, opportunity)')
    for name in ('strengths', 'needsWork', 'reviews', 'described'):
        for f in report[name]:
            for o in f['opportunities']:
                if o not in opps:
                    out.append(f'report.{name}: finding {f["id"]} aggregates {o}, which was not measured')
    return out


# ---------- examples: valid records to start tests and mocks from ----------
def _examples():
    team, fps = 'A', 30.0
    mfp = hashlib.sha256(b'example-measurement').hexdigest()
    rfp = hashlib.sha256(b'example-roles').hexdigest()
    ifp = hashlib.sha256(b'example-interpretation').hexdigest()
    n = 19                                   # samples of one opportunity (3.6 s at 5 Hz)

    def make_opp(a):
        b = round(a+.2*(n-1), 1)
        f0 = int(round(a*fps))
        key = opportunity_key('pressure_cover', team, a, b)
        return {'id': record_id('opp', key), 'key': key, 'detector': 'pressure_cover', 'team': team,
                'tier': 'measurement', 't0': a, 't1': b, 'frame0': f0, 'frame1': f0+108,
                'phase': {'state': 'opp', 'source': 'possession-4', 'estimated': True, 'spanSeconds': 9.2,
                          'nearChange': False},
                'ball': {'zone': 'own_third', 'channel': 'left', 'reliable': True},
                'actors': {'primary': ['A-04'], 'supporting': ['A-05', 'A-08'], 'opponents': ['B-10'],
                           'carrier': 'B-10'},
                'roles': {'A-04': {'role': 'LCB', 'source': 'stable', 'confidence': 'medium'}},
                'relations': {'t': [round(a+.2*k, 1) for k in range(n)], 'd1': [3.1]*n, 'fdGoalSide': [True]*n,
                              'coverDist': [14.0]*n, 'coverAngle': [70.0]*n, 'secondId': ['A-05']*n},
                'quality': {'identity': 'high', 'position': 'high', 'visibility': 'disc', 'ball': 'reliable',
                            'samples': n},
                'intentSignals': {'recovering': False, 'interchange': None, 'formationChangeNear': False},
                'abstain': None}

    def make_moment(o):
        frame = o['frame0']+30
        t = round(o['t0']+1.0, 1)
        scene = {'t': t, 'frame': frame, 'ball': [.2, .3], 'ballReliable': True,
                 'players': {'A-04': {'team': 'A', 'x': .22, 'y': .31, 'kind': 'O', 'sigma': .8, 'goalkeeper': False,
                                      'box': [.40, .50, .02, .08]},
                             'A-05': {'team': 'A', 'x': .18, 'y': .45, 'kind': 'O', 'sigma': .9, 'goalkeeper': False,
                                      'box': [.30, .52, .02, .08]},
                             'B-10': {'team': 'B', 'x': .21, 'y': .29, 'kind': 'O', 'sigma': .7, 'goalkeeper': False,
                                      'box': [.42, .49, .02, .08]}}}
        before = copy.deepcopy(scene)
        before.update(t=round(t-2.0, 1), frame=frame-60)
        return {'id': record_id('ev', o['id'], frame), 'opportunity': o['id'], 'detector': 'pressure_cover',
                'team': team, 'sample': 5, 't': t, 'frame': frame, 'seek': seek_of(frame, fps), 'fps': fps,
                'tier': 'world', 'calibration': {'confidence': .82, 'paintResidualPx': 4.1, 'disagreement': None},
                'players': ['A-04', 'A-05', 'B-10'], 'flagged': [], 'scene': scene, 'before': before,
                'image': {'width': 1920, 'height': 1080}}
    opps = [make_opp(round(760.0+14.0*k, 1)) for k in range(9)]
    moments = [make_moment(o) for o in opps[:3]]
    opp, moment = opps[0], moments[0]
    frame = moment['frame']
    coverage = {'detector': 'pressure_cover', 'tier': 'measurement', 'eligibleSeconds': 34.0, 'phaseSeconds': 88.2,
                'opportunities': len(opps), 'eligibleSpans': [[o['t0'], o['t1']] for o in opps],
                'abstain': {'no_carrier': 30.2}, 'dominantReason': 'no_carrier'}
    timeline = {'team': team, 'samples': 900, 'step': .2, 't0': 720.0, 't1': 899.8, 'frame0': 21600, 'frame1': 26994,
                'fps': fps, 'format': '11v11',
                'phaseSeconds': {'own': 30.0, 'opp': 128.2, 'dead': 4.0, 'unknown': 17.6, 'nearChange': 12.0},
                'phaseSpans': [{'state': 'own', 't0': 720.0, 't1': 750.0, 'frame0': 21600, 'frame1': 22500},
                               {'state': 'opp', 't0': 758.0, 't1': 886.2, 'frame0': 22740, 'frame1': 26586}],
                'phaseQuality': {'available': True, 'reason': None, 'level': 'good', 'coverage': .79,
                                 'roughShares': {'A': .45, 'B': .55},
                                 'notes': ['Broadcast stoppages are mostly missed.']},
                'footage': {'kind': 'broadcast', 'smallSceneShare': 0.0, 'keyless': False, 'lensBent': False},
                'dataQuality': {'inKit': 9.6, 'confirmed': 9.4, 'positioned': 9.1, 'lineup': 10,
                                'dominantCause': None},
                'scale': {'lengthM': 105.0, 'widthM': 68.0, 'source': 'assumed', 'metresUsable': True},
                'completeShare': .62, 'observedMean': 9.1, 'ballReliableShare': .71, 'carrierShare': .45,
                'formationPeriods': [{'index': 0, 't0': 720.0, 't1': 899.8, 'label': '4-3-3', 'confidence': 'likely'}],
                'players': {'A-04': {'team': 'A', 'goalkeeper': False, 'observedSeconds': 170.0, 'p90SpeedMS': 5.6},
                            'A-05': {'team': 'A', 'goalkeeper': False, 'observedSeconds': 168.0, 'p90SpeedMS': 5.9}}}
    obs = {'kind': 'coaching-observations', 'version': 2, 'team': team,
           'engine': {'version': ENGINE_VERSION, 'measurement': mfp[:16]}, 'fingerprint': mfp,
           'inputs': {'result': hashlib.sha256(b'result').hexdigest(), 'shapeSettings': hashlib.sha256(b'{}').hexdigest(),
                      'measurement': {'format': dict(DEFAULT_FORMAT), 'field': None}},
           'segment': {'t0': 720.0, 't1': 899.8, 'fps': fps, 'frames': 5395},
           'video': {'name': 'example.mp4', 'key': None}, 'timeline': timeline,
           'detectors': {'pressure_cover': coverage}, 'opportunities': opps, 'moments': moments,
           'frameTiers': [{'t0': 720.0, 't1': 899.8, 'frame0': 21600, 'frame1': 26994, 'tier': 'world',
                           'confidence': .8}],
           'roleInputs': {'t': [720.0, 720.2], 'frame': [21600, 21606]}}
    rolebook = {'team': team, 'format': '11v11', 'model': 'team_shape',
                'records': [{'id': 'A-04', 't0': 720.0, 't1': 899.8, 'role': 'LCB', 'group': 'D', 'side': 'L',
                             'source': 'stable', 'confidence': 'medium', 'support': .83, 'period': 0,
                             'ambiguousWith': [], 'conflict': None}],
                'interchanges': [], 'formationChecks': [
                    {'period': 0, 't0': 720.0, 't1': 899.8, 'measured': '4-3-3', 'measuredConfidence': 'likely',
                     'intended': None, 'mismatch': False, 'related': False}]}
    roles = {'kind': 'coaching-roles', 'version': 2, 'team': team,
             'engine': {'version': ENGINE_VERSION, 'measurement': mfp[:16], 'roles': rfp[:16]},
             'measurement': mfp, 'fingerprint': rfp,
             'inputs': {'roles': {}, 'formation': {'intended': None, 'periods': []}}, 'rolebook': rolebook,
             'detectors': {}, 'opportunities': [], 'moments': []}
    guide = {'text': 'Cover is the angle and distance that make you the next protection behind the presser.',
             'tag': 'FEDERATION', 'sources': ['EF-COVER-RECOVER']}
    dp = {'enabled': True, 'verdicts': ['strength', 'work_on', 'review'], 'framing': 'close enough to engage, far '
          'enough to support', 'params': {'pressureM': 5.0, 'coverAngleDeg': 60.0}, 'origin': ORIGIN_P,
          'validated': False}
    profile = {'version': 1, 'sha': mfp[:16], 'team': team, 'ageGroup': 'U14', 'level': 4,
               'levelName': LEVEL_NAMES[4], 'band': None,
               'format': {'players': 11, 'goalkeeper': True, 'outfield': 10, 'name': '11v11'}, 'field': None,
               'federation': 'auto', 'style': 'unknown', 'formation': '4-3-3', 'objectives': [guide],
               'principles': [guide], 'notExpected': [],
               'detectors': {d: copy.deepcopy(dp) for d in DETECTORS},
               'evidence': {'minOpportunities': 10, 'minFailingSpells': 2, 'minFailureShare': .4,
                            'minPhaseSeconds': 90.0, 'maxWorkItems': 3, 'strengthSpells': 3, 'strengthShare': .65,
                            'naming': 'evidence_only', 'origin': ORIGIN_P, 'validated': False},
               'register': {'tone': 'neutral, specific', 'numbers': 'metres', 'maxChallenges': 3,
                            'concepts': ['pressure', 'cover', 'defensive line'], 'banned': ['mistake', 'trigger']},
               'skills': {s: s.replace('_', ' ').capitalize() for s in SKILLS},
               'drillPool': ['l4_defensive_unit_6v6', 'l4_cover_the_line_3v3'],
               'safety': {'heading': 'limited', 'maxLoad': 'moderate', 'tackling': True, 'sprintRaces': True,
                          'notes': ['Growth spurts: keep sprint work low in volume.']},
               'notices': [], 'sources': ['EF-COVER-RECOVER']}
    fkey = finding_key('pressure_cover', team, 'own_third', opps[0]['t0'], opps[-1]['t1'])
    fid = record_id('fnd', fkey, 4)
    alt_id = record_id('alt', fid, moment['id'], 'A-05')
    finding = {'id': fid, 'key': fkey, 'observation': 'pressure_cover', 'verdict': 'work_on', 'level': 4,
               'team': team, 'priority': 1,
               'players': {'primary': ['A-04'], 'supporting': ['A-05'], 'opponents': ['B-10']},
               'roles': {'A-04': {'role': 'LCB', 'source': 'stable', 'confidence': 'medium'}},
               'timestamps': [{'evidence': m['id'], 'opportunity': m['opportunity'], 't': m['t'],
                               'frame': m['frame'], 'seek': m['seek']} for m in moments],
               'opportunities': [o['id'] for o in opps],
               'context': {'phase': {'state': 'opp', 'estimated': True, 'coverage': .79},
                           'ball': {'zone': 'own_third', 'channel': 'left'}, 'style': 'unknown', 'styleEffect': None,
                           'format': '11v11'},
               'measurements': {'d1Median': 3.1, 'coverPresentShare': .33, 'metres': 'estimated'},
               'aggregate': {'opportunities': 9, 'demonstrated': 3, 'notDemonstrated': 5, 'unclear': 1,
                             'distinctFailingSpells': 3, 'distinctDemonstratedSpells': 2, 'phaseSeconds': 88.2,
                             'eligibleShare': .39, 'ratePer10': 3.4},
               'coverage': {'identity': 'high', 'position': 'high', 'visibility': 'disc', 'ball': 'reliable',
                            'calibrationTier': 'world', 'eligibleShareOfPhase': .39},
               'guardrails': {g: 'pass' for g in GUARDRAILS}, 'limitations': [],
               'suggestedAction': {'principle': 'cover_first_defender', 'mode': 'zone', 'alternatives': [alt_id]},
               'claims': [{'type': 'situation', 'facts': {'phase': 'opp', 't': 813.4}},
                          {'type': 'relation', 'facts': {'relation': 'cover', 'present': False, 'with': 'A-05',
                                                         'distanceM': 14.0}},
                          {'type': 'recurrence', 'facts': {'failing': 5, 'of': 9, 'distinctSpells': 3}},
                          {'type': 'principle', 'facts': {'principle': 'cover_first_defender'}, 'tag': 'FEDERATION',
                           'source': 'EF-COVER-RECOVER'}],
               'text': {'headline': 'Pressure on the ball came without cover behind it',
                        'explanation': 'When the other team had the ball (estimated) in our third, A-04 closed the '
                                       'carrier to about 3 m while the nearest teammate stayed about 14 m away at a '
                                       'wide angle.',
                        'short': 'Pressure without cover', 'action': 'One presses, the nearest teammate covers.'},
               'drills': ['l4_cover_the_line_3v3']}
    alternative = {'id': alt_id, 'finding': fid, 'evidence': moment['id'], 't': moment['t'], 'mover': 'A-05',
                   'principle': 'cover_first_defender',
                   'zone': {'polygon': [[.17, .36], [.21, .36], [.21, .40], [.17, .40]], 'areaM2': 14.2, 'kind': 'zone',
                            'radiusM': None},
                   'display': 'zone', 'from': [.18, .45], 'reach': {'horizonS': 2.0, 'speedMS': 5.6, 'radiusM': 11.2},
                   'improves': {'relation': 'cover', 'before': False, 'after': True, 'shareOfZone': .92},
                   'checks': {'pressureKept': True, 'centralKept': True, 'gapOk': True, 'noNewFreeOpponent': True,
                              'noStacking': True},
                   'assumption': 'single_mover', 'rejected': [{'mover': 'A-08', 'reason': 'would_open_new_gap'}]}
    diagram = {'preset': 'pressure_cover', 'title': 'Pressure and cover', 't': moment['t'], 'frame': frame,
               'team': team, 'attack': 'right', 'metresEstimated': True, 'pitch': {'length': 105.0, 'width': 68.0},
               'elements': [
                   {'type': 'player', 'id': 'B-10', 'team': 'B', 'x': .21, 'y': .29, 'kind': 'O', 'emphasis': False},
                   {'type': 'focus', 'id': 'A-04', 'x': .22, 'y': .31, 'tone': 'review', 'label': True},
                   {'type': 'support', 'id': 'A-05', 'team': 'A', 'x': .18, 'y': .45, 'why': 'nearest teammate'},
                   {'type': 'relation', 'a': 'A-04', 'b': 'A-05', 'kind': 'cover', 'ok': False, 'text': '14 m'},
                   {'type': 'suggestedZone', 'polygon': alternative['zone']['polygon'], 'kind': 'zone',
                    'alternative': alt_id, 'mover': 'A-05', 'label': 'possible covering area'},
                   {'type': 'moveArrow', 'id': 'A-05', 'from': [.18, .45], 'to': [.19, .38]},
                   {'type': 'caption', 'text': 'Pressure on the ball; nobody covered behind it.',
                    'anchor': 'top-left'}]}
    overlay = {'tier': 'world', 'width': 1920, 'height': 1080, 'elements': [
        {'type': 'focus', 'id': 'A-04', 'x': 787.2, 'y': 626.4, 'tone': 'review', 'label': True,
         'box': [768.0, 540.0, 38.4, 86.4]},
        {'type': 'relation', 'a': 'A-04', 'b': 'A-05', 'kind': 'cover', 'ok': False, 'text': '14 m'},
        {'type': 'suggestedZone', 'polygon': [[600.0, 700.0], [680.0, 700.0], [690.0, 760.0], [590.0, 760.0]],
         'kind': 'zone', 'alternative': alt_id, 'mover': 'A-05', 'label': 'possible covering area'},
        {'type': 'caption', 'text': 'Pressure on the ball; nobody covered behind it.', 'anchor': 'top-left'}]}
    def make_evidence(m, first):
        return {'id': m['id'], 'opportunity': m['opportunity'], 'finding': fid, 'detector': 'pressure_cover',
                'team': team, 'time': m['t'], 'frame': m['frame'], 'seek': m['seek'], 'fps': fps,
                'segmentTime': round(m['t']-720.0, 3), 'tier': 'world', 'calibration': m['calibration'],
                'players': ['A-04', 'A-05', 'B-10'], 'flagged': [], 'confidence': 'high',
                'alternative': alt_id if first else None, 'diagram': diagram if first else None,
                'videoOverlay': overlay if first else None,
                'annotation': {'finding': fid, 'window': {'t0': round(m['t']-3.0, 3), 't1': round(m['t']+2.0, 3)},
                               'keyFrame': m['frame'], 'keyTime': m['t'],
                               'caption': 'Pressure on the ball; nobody covered behind it.'},
                'caption': 'A-04 pressed the carrier; the nearest teammate was about 14 m away.'}
    key_moments = [make_evidence(m, k == 0) for k, m in enumerate(moments)]
    evidence = key_moments[0]
    drill = {'drill_id': 'l4_cover_the_line_3v3', 'name': '3v3 Cover the Line', 'development_level': [4, 5],
             'age': {'min': 'U12', 'max': 'U16'}, 'primary_skill': 'pressure_and_cover',
             'secondary_skills': ['far_side_protection'],
             'related_tactical_detector': ['pressure_cover', 'weak_side_balance'],
             'recommended_format': ['9v9', '11v11'],
             'player_count': {'min': 6, 'max': 12, 'note': '3 attackers v 3 defenders, rotate'},
             'equipment': ['cones', 'bibs', 'balls'],
             'field_setup': {'length': {'value': 30.0, 'unit': 'm', 'scaleWith': 'age'}, 'width': 25.0,
                             'notes': 'A covering line 10 m in front of the end line.'},
             'duration': {'minutes': 15, 'blocks': '4 x 3 min with rests'},
             'instructions': ['Attackers try to dribble over the end line.'],
             'coaching_points': ['One presses, the nearest teammate covers at an angle.'],
             'progression': ['Add a fourth attacker.'], 'regression': ['Play 2v2.'],
             'success_signals_next_match': [{'detector': 'pressure_cover', 'relation': 'cover',
                                             'text': 'Cover present in more pressured moments.'}],
             'safety': {'heading': 'none', 'physicalLoad': 'moderate', 'contact': 'normal'},
             'source_reference': [{'source': 'EF-PTC', 'relation': 'adapted_from',
                                   'note': 'PitchIQ adaptation; not a verbatim federation session'}],
             'origin': ORIGIN_P}
    settings = {'version': 1, 'updated': '2026-10-21T15:00:00',
                'teams': {'A': {'squad': {'id': record_id('sq', 'U14 Blue'), 'name': 'U14 Blue'}, 'ageGroup': 'U14',
                                'playingLevel': 4, 'format': {'players': 11, 'goalkeeper': True}, 'field': None,
                                'federation': 'auto', 'formation': {'intended': '4-3-3', 'periods': []},
                                'style': 'unknown',
                                'roles': {'A-03': [{'role': 'LB', 'from': None, 'to': None,
                                                    'set': '2026-10-21T15:00:00'}]},
                                'runLinks': [{'run': 'run-20261021', 'team': 'A', 'confirmed': True}]},
                          'B': None}}
    progress = {'id': record_id('pe', 'sq', 'video', 'run', 'A'), 'squad': settings['teams']['A']['squad']['id'],
                'video': 'f597d73d53f7b1fd', 'run': 'run-20261021', 'team': team, 'range': {'t0': 720.0, 't1': 899.8},
                'matchDate': '2026-10-18', 'dateSource': 'coach',
                'engine': {'measurement': mfp[:16], 'version': '2'},
                'context': {'format': '11v11', 'footage': 'broadcast', 'phaseAvailable': True, 'level': 4,
                            'ageGroup': 'U14'},
                'skills': {'pressure_and_cover': {'opportunities': [
                    {'t0': o['t0'], 't1': o['t1'], 'zone': 'own_third', 'channel': 'left',
                     'rel': {'d1Med': 3.1, 'coverShare': 0.0, 'closingMax': 2.4}, 'quality': 'high'} for o in opps],
                    'phaseSeconds': 88.2, 'eligibleShare': .39, 'confidence': 'medium'}},
                'updated': '2026-10-21T15:00:00'}
    report = {'kind': 'coaching-report', 'version': 2, 'team': team, 'teamName': None,
              'generated': '2026-10-21T15:00:00',
              'engine': {'version': ENGINE_VERSION, 'measurement': mfp[:16], 'roles': rfp[:16],
                         'interpretation': ifp[:16]},
              'fingerprints': {'measurement': mfp, 'roles': rfp, 'interpretation': ifp}, 'run': 'run-20261021',
              'video': {'name': 'example.mp4', 'key': None},
              'range': {'t0': 720.0, 't1': 899.8, 'seconds': 179.8, 'whole': True},
              'segment': {'t0': 720.0, 't1': 899.8}, 'profile': profile,
              'overview': {'notices': [], 'footage': timeline['footage'], 'dataQuality': timeline['dataQuality'],
                           'scale': timeline['scale'],
                           'possession': {'available': True, 'quality': 'good', 'reason': None,
                                          'roughShares': {'A': .45, 'B': .55}, 'coverage': .79,
                                          'caveat': 'Shares are estimates; broadcast stoppages are mostly missed.'},
                           'formations': timeline['formationPeriods'],
                           'detectors': [{'detector': 'pressure_cover', 'status': 'assessed', 'reason': None,
                                          'eligibleSeconds': 34.0, 'phaseSeconds': 88.2, 'opportunities': 9},
                                         {'detector': 'wide_defender_advance', 'status': 'not_assessed',
                                          'reason': 'role_unconfirmed', 'eligibleSeconds': 0.0, 'phaseSeconds': 70.0,
                                          'opportunities': 0}],
                           'unavailable': [{'category': 'passing_support',
                                            'reason': 'No pass-lane or body-orientation model.',
                                            'unlock': 'A lane model and labels.'}],
                           'text': None},
              'strengths': [], 'needsWork': [finding], 'reviews': [], 'described': [], 'keyMoments': key_moments,
              'diagrams': [{'id': record_id('dg', fid, moment['id']), 'kind': 'finding', 'finding': fid,
                            'evidence': moment['id'], 'caption': 'Pressure without cover', 'diagram': diagram}],
              'alternatives': [alternative], 'noAlternatives': [],
              'drills': [{'finding': fid, 'drill': 'l4_cover_the_line_3v3', 'optional': False,
                          'fields': {'trains': 'Pressure and cover.', 'why': 'Cover was missing in 5 of 9 moments.',
                                     'moments': [moment['id']], 'easier': 'Play 2v2.',
                                     'harder': 'Add a fourth attacker.',
                                     'progress': 'Cover present in more than 5 of 9 pressured moments.'}}],
              'progression': None,
              'nextObjective': {'skill': 'pressure_and_cover', 'text': 'Pressure and cover.', 'fromMatch': True,
                                'finding': fid},
              'sources': [{'id': 'EF-COVER-RECOVER', 'title': 'Cover and recover', 'url':
                           'https://learn.englandfootball.com/', 'publisher': 'England Football'}],
              'withheld': []}
    return {'settings': settings, 'team_settings': settings['teams']['A'], 'profile': profile, 'timeline': timeline,
            'opportunity': opp, 'moment': moment, 'detector_coverage': coverage, 'obs': obs,
            'role_record': rolebook['records'][0], 'rolebook': rolebook, 'roles': roles,
            'claim': finding['claims'][0], 'finding': finding, 'alternative': alternative,
            'no_alternative': {'finding': fid, 'evidence': moment['id'], 't': moment['t'],
                               'principle': 'cover_first_defender', 'reason': 'no_reachable_fix', 'rejected': []},
            'element': diagram['elements'][1], 'diagram': diagram, 'overlay': overlay,
            'annotation': evidence['annotation'], 'evidence': evidence, 'drill': drill,
            'drill_library': {'version': 2, 'notes': 'Example library.', 'drills': [drill]},
            'progress_entry': progress, 'progress_file': {'version': 1, 'entries': [progress]},
            'squads': {'version': 1, 'squads': [{'id': settings['teams']['A']['squad']['id'], 'name': 'U14 Blue',
                                                 'ageGroup': 'U14', 'level': 4,
                                                 'format': {'players': 11, 'goalkeeper': True},
                                                 'created': '2026-10-21T15:00:00'}]},
            'adjustment': {'id': record_id('adj', fkey, moment['id']), 'findingKey': fkey, 'evidence': moment['id'],
                           'polygon': [[.16, .35], [.22, .35], [.22, .41]], 'mover': 'A-05', 'note': None,
                           'created': '2026-10-21T15:00:00'},
            'report': report,
            'coach_spans': {'version': 1, 'kind': 'coach-possession-spans', 'source': 'coach',
                            'spans': [{'t0': 720.0, 't1': 750.0, 'state': 'A'},
                                      {'t0': 752.4, 't1': 756.0, 'state': 'dead'},
                                      {'t0': 758.0, 't1': 886.2, 'state': 'B'}]}}


_EXAMPLES = None


def example(kind):
    """A valid record of kind (a deep copy: mutate freely). The examples form one consistent mini-run: the
    'report' example passes cross_check(example('report'), example('obs'), example('roles'))."""
    global _EXAMPLES
    if _EXAMPLES is None:
        _EXAMPLES = _examples()
    return copy.deepcopy(_EXAMPLES[kind])


__all__ = ['SCHEMA_VERSION', 'ENGINE_VERSION', 'KINDS', 'SchemaError', 'validate', 'check', 'is_valid',
           'cross_check', 'event_problems', 'example', 'record_id', 'opportunity_key', 'finding_key', 'seek_of',
           'format_name', 'outfield', 'tier_settings', 'stale_tiers', 'team_block', 'TIER_FIELDS', 'TIER_ELEMENTS',
           'ELEMENT_BUDGET', 'DETECTORS', 'DETECTOR_CODE', 'SKILLS', 'FINDING_VERDICTS', 'ABSTAIN_REASONS',
           'FORMATIONS', 'AGE_GROUPS', 'LEVEL_NAMES', 'STYLES', 'FEDERATIONS', 'GUARDRAILS', 'CLAIM_TYPES']
