"""Development progression across matches (DESIGN v2 section 8; spec Part 13): a squad's coaching skills followed over
the matches PitchIQ analysed, from the opportunities where each skill mattered, never from match totals or from the
wording of a report.

Why opportunities and not report verdicts. Spec Part 13 asks to "base progress on observed opportunities and
successful demonstrations" and never "on arbitrary report sentiment", and not to compare a 60 s analysis with a 90
minute one on totals. A report verdict is the wrong unit twice over: it depends on how much footage there was (a
work item needs 8-15 opportunities and 2-3 distinct failing spells, so a short clip never gets one), and on the level
(the same moment is "go to the ball" at level 1 and "pressure with cover" at level 4). So an entry stores, per skill,
the opportunities of the analysed range with level-independent relations, and every reading (outcome, mastery state,
trend, rate) is made at read time with the squad's CURRENT profile: a level change re-reads the whole history the same
way the report reads this match.

Why each opportunity carries an outcome table. DESIGN 8.2 asks for "compact relations" with outcomes recomputed at
read time. The report's outcome of an opportunity (assess.classify) is a per-sample rule (60% of the in-scope samples
demonstrate or fail, the failing run lasts the spell minimum) that a few scalar summaries cannot reproduce, and the
schema allows only scalars in ProgressOpp.rel. Keeping every sample would cost what the obs file costs (about 6.5 MB
per team for Metrica game 1's 90 minutes without the role geometry, DETECT's measurement) for every match of a season.
So rel holds, besides the compact descriptive relations (d1Med, coverDistMed, ...), the outcome of the opportunity
under every classification parameter set profiles.json can resolve for the match format (all 7 levels x 17 age groups
x 5 styles), keyed 'o' + 10 hex of a hash of (the classification code, the detector, the parameters): 1 demonstrated,
-1 not demonstrated, 0 unclear, None out of the level's scope. Measured: 11v11 has 22 distinct sets (D1 8, D2 1, D3 4,
D4 4, D5 3, D6 2; 7v7 21, 4v4 20) plus level 1's 2 readings of D3 as "someone back"; the 595 profiles of a format
resolve in about 0.25 s, once per process. The outcomes are exactly the report's: on the five run copies (both teams,
engine and coach phases) and Metrica game 1 every entry equals assess.classify at all 7 levels x 5 styles (185 checks
per team-run, 0 differences), and the table is still level-independent: every level's reading is in it. When
profiles.json or the classification code changes, a key is missing and the entry is re-derived from its cached obs
file (rederive), else listed apart, never misread.

Measured (Oct 22, CPU). A 90-minute Metrica entry is 233-270 KB per team (856-963 opportunities) and builds in
0.12-0.37 s; the broadcast run copies 1-12 KB in under 6 ms. A season of 30 such matches for two squads is a 15 MB
progress.json: parsed and validated once per process (0.87 s for one squad's 30 entries; nothing after this process's
own writes), one more upsert 0.13 s, the report section 12 ms. Cut into 60 s clips, Metrica game 1 gives the same
judged counts as the whole match and per-10-minute rates within 0.2% (the phase seconds of a span cut at a clip edge
gain one 0.2 s step), while single clips range from 0 to 50 failing moments per 10 minutes (D1, level 4, p90): that
spread is why matches are compared only per opportunity and only when they qualify.

Mastery (progress_rules.json, all PITCHIQ-PROPOSAL, unvalidated; DESIGN 8.4). Per skill, over the squad's matches (a
match = one video; its entries are non-overlapping ranges), on the judged opportunities (demonstrated or not; unclear
and out of scope are not judged) whose identity and position quality is at least medium:
  NOT ENOUGH EVIDENCE  fewer than 20 judged opportunities or fewer than 2 qualifying matches (>= 6 judged each);
  LEARNING             the Wilson 80% lower bound of the demonstrated share is below s1;
  DEVELOPING           the lower bound is at least s1;
  CONSISTENT           over the last 3 qualifying matches the pooled lower bound is at least s2 and every one of them
                       has a share of at least s1 (checked before DEVELOPING: recent consistency wins);
  ADAPTABLE            CONSISTENT, and in one context dimension (ball zone, ball channel, wide defender's side, match
                       format) at least 2 values with >= 6 judged opportunities each have a lower bound of at least s2.
s1 / s2: levels 1-3 .50 / .65, 4-5 .60 / .75, 6-7 .70 / .85. The evidence is the most recent 5 qualifying matches and
every match after the first of them (a PitchIQ addition: a season-old run of poor matches does not hold a team down).
The trend compares the latest qualifying match with the 3 before it per opportunity (a change of 0.1 that a
two-proportion test finds at 90%). Orientation only (one professional match, Metrica game 1 cut into six 15-minute
"matches", style unknown): no skill of either team reads LEARNING at levels 1-5; at levels 6-7 team B's back-line
coordination does (share .77, lower bound .696 < s1 .70), the D-M gap DETECT measured as its failure; at 80% the trend
read 'down' at 7 of the 14 team x level readings of pressure and cover, at 90% at 1.
Normalisation (spec Part 13; DESIGN 13.4 #14): matches are compared only through per-opportunity shares, only when
they qualify, and rates are per 10 minutes of the skill's relevant phase; a 60 s clip adds its few opportunities to
the pooled share and is never set against a 90 minute match on totals (tests: equal per-opportunity behaviour gives
equal shares and rates, a flat trend and the same state whatever the clip lengths).

Team identity across runs (DESIGN 8.1). A squad (runs/_coaching/squads.json) is linked per video in
coaching_settings.json: teams[letter].squad and runLinks [{run, team, confirmed}]. Team letters can change between runs,
so a run without a confirmed link asks "Which team is <squad> in this run?" and writes nothing.

The store (runs/_coaching/progress.json): one entry per (squad, video, run, team, range); a regeneration replaces it,
and overlapping ranges of the same video and squad keep only the longest, so a moment is never counted twice. A newer
measurement engine marks older entries of the run stale (mark_stale); stale entries are re-derived from their cached
obs file when it exists, else listed apart ("analysed with an earlier engine"). Writes are atomic (a uuid temporary
file and os.replace, retried while a Windows reader holds the file) under a process lock; the file is read strictly
before every write (an unreadable file is an error, never an empty list, so a failed read cannot wipe a season).
The atomic writer is this module's own, not store.py's: progress.py is hashed into the interpretation tier
(coaching.TIER_ROOTS) and importing store.py would put the storage code into every report fingerprint.

Possession rules (possession v4 docs): the opportunities already carry them (team per frame only, +-2 s around every
change and dead spells removed, shape checks on spans of 3 s or more); nothing here counts a change of possession or
says who lost or won the ball. Coach-marked phases (user decision 1) are kept per opportunity (rel.coachPhase) and
shown as coach-entered in the detail and the note.
"""
import ast
import copy
import hashlib
import inspect
import json
import math
import os
import re
import statistics
import textwrap
import threading
import time
import uuid
from datetime import date, datetime, timezone
from pathlib import Path

from . import assess as AS
from . import contextual as CX
from . import episodes as EP
from . import profiles as PR
from . import schema as S
from . import situations as SI
from . import tactics_geom as G

RULES_PATH = Path(__file__).with_name('progress_rules.json')
RULES_KIND = 'coaching-progress-rules'
STORE_DIR = '_coaching'             # runs/_coaching/ holds the squads and the progress entries (DESIGN 8.1-8.2)
SQUADS_FILE = 'squads.json'
PROGRESS_FILE = 'progress.json'
FILE_VERSION = 1                    # schema 'progress_file' / 'squads' version
OUTCOME_PREFIX = 'o'                # rel key of an outcome: 'o' + OUTCOME_HEX hex digits
OUTCOME_HEX = 10
OUTCOME_KEY = re.compile(r'o[0-9a-f]{10}')
OUTCOME_CODE = {'demonstrated': 1, 'not_demonstrated': -1, 'unclear': 0}
QUALITY_RANK = {'high': 0, 'medium': 1, 'low': 2}
TEAM_LEVEL = AS.TEAM_LEVEL          # identity is not part of these checks' quality (assess's identity guardrail)
CONTEXT_DIMENSIONS = ('zone', 'channel', 'side', 'format')
STATES = S.MASTERY_STATES
NOTE_MAX = 400                      # schema Progression.note
REPLACE_TRIES, REPLACE_WAIT = 20, .1    # os.replace retries against a Windows reader (store.py's values)
READ_TRIES, READ_WAIT = 8, .05          # read retries while the file is being replaced
OVERLAP_EPS_S = 1e-6                # s: ranges that only touch do not overlap
RANGE_DIGITS = 3

# Every profile parameter the outcome of an opportunity depends on: what contextual.sample_outcomes,
# contextual.pressure_cover_flags (and _scaled_param) and assess.classify read from the parameters. A test scans their
# source and fails when they read a key that is not here (an outcome table missing a parameter would be wrong).
CLASSIFY_KEYS = ('advancedHalfwayShare', 'advancedM', 'advancedRule', 'advancedScale', 'behindBallZone', 'behindRule',
                 'carrierZone', 'cbShiftM', 'countOnly', 'coverAngleDeg', 'coverExemptZone', 'coverRadiusM',
                 'coverRadiusMinM', 'coverRequired', 'depthAvailShare', 'depthRule', 'depthShare', 'dmGapM',
                 'dmGapScale', 'dmGapStrengthHiM', 'dmGapStrengthHiScale', 'dmGapStrengthLoM', 'dmGapStrengthLoScale',
                 'excludeOwnGoalM', 'excludeOwnGoalScale', 'failMinS', 'goToBallClosingMS', 'goToBallMaxM',
                 'judgeDepth', 'judgeWidth', 'lateralGapM', 'lateralGapScale', 'lineStdStrengthM', 'minBehind',
                 'minMargin', 'minWeakCount', 'outcomeShare', 'pressureCertainM', 'pressureClosingMS', 'pressureM',
                 'pressureRule', 'recoveringMS', 'requireNobodyBehind', 'spellMinS', 'strengthTimeShare',
                 'unpressuredZone', 'useDmRotation', 'widthShare')
# The code an outcome is computed with: its functions (AST without docstrings) and constants are hashed into every
# outcome key, so a change of the classification code makes the stored outcomes unreadable (re-derived from obs)
# instead of silently different from the report's. A test checks that these functions call nothing else of ours.
_MODULES = {'contextual': CX, 'assess': AS, 'situations': SI, 'tactics_geom': G, 'episodes': EP}
CLASSIFY_CODE = (('contextual', 'sample_outcomes'), ('contextual', 'pressure_cover_flags'),
                 ('contextual', '_scaled_param'), ('assess', 'classify'), ('situations', 'spells'),
                 ('episodes', 'sample_step'), ('tactics_geom', 'factors'), ('tactics_geom', 'scaled'),
                 ('tactics_geom', 'cover_radius'))
CLASSIFY_VALUES = (('contextual', 'REFERENCE_PARAMS'), ('contextual', 'D4_JUDGED_ZONE'),
                   ('situations', 'SPELL_MERGE_S'), ('situations', 'SPELL_MIN_S'), ('tactics_geom', 'COVER_ANGLE_DEG'),
                   ('tactics_geom', 'RECOVERING_MS'), ('tactics_geom', 'REF_LENGTH'), ('tactics_geom', 'REF_WIDTH'),
                   ('tactics_geom', 'COVER_RADIUS_M'), ('tactics_geom', 'COVER_RADIUS_MIN_M'))

# Level-independent compact relations kept per opportunity (descriptive: what the moment looked like; the outcome
# table decides): (stored name, relation, summary, digits). Shares are over the samples where the relation is known.
SUMMARY = {
    'pressure_cover': (('d1Med', 'd1', 'median', 1), ('coverDistMed', 'coverDist', 'median', 1),
                       ('closingMax', 'closingMS', 'max', 1), ('coverVisibleShare', 'coverVisible', 'share', 2)),
    'weak_side_balance': (('weakMin', 'weakCount', 'min', None), ('weakMed', 'weakCount', 'median', 1),
                          ('openWidthMed', 'openWidthM', 'median', 1), ('centralMin', 'centralCount', 'min', None)),
    'wide_defender_advance': (('advanceMed', 'advanceM', 'median', 1),
                              ('channelOccupiedShare', 'channelOccupied', 'share', 2),
                              ('cbAdjustedShare', 'cbAdjusted', 'share', 2),
                              ('dmDroppedShare', 'dmDropped', 'share', 2),
                              ('opponentsMax', 'opponentsInChannel', 'max', None),
                              ('behindMin', 'behindN', 'min', None)),
    'behind_ball': (('behindMin', 'behindN', 'min', None), ('behindMed', 'behindN', 'median', 1),
                    ('restMarginMin', 'restMargin', 'min', None), ('threatMax', 'threatN', 'max', None),
                    ('ballXMed', 'ballX', 'median', 1)),
    'defensive_spacing': (('lineStdMed', 'lineStdM', 'median', 2), ('lateralGapMax', 'lateralGapM', 'max', 1),
                          ('dmGapMed', 'dmGapM', 'median', 1), ('opponentInGapShare', 'opponentInGap', 'share', 2)),
    'attacking_shape': (('widthShareMed', 'widthShare', 'median', 2), ('depthShareMed', 'depthShare', 'median', 2),
                        ('channelsMin', 'channelsOccupied', 'min', None), ('bandsMin', 'bandsOccupied', 'min', None)),
}
SKILL_OF = {d: k for k, d in S.SKILLS.items()}

_LOCK = threading.RLock()           # one read-modify-write of the store at a time in this process
_cache = {}
_docs = {}                          # store file -> (version key, document, row validation flags)


class ProgressError(ValueError):
    """A request the progression store refuses (a malformed squad, entry or rules file); the message says why."""


class ProgressUnreadable(ProgressError):
    """The squads or progress file exists but cannot be read or understood: nothing was written."""


# ---------- rules (progress_rules.json) ----------
def _canonical_sha(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True, separators=(',', ':'), ensure_ascii=True,
                                     allow_nan=False).encode('ascii')).hexdigest()


def _is_num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _numbers_tagged(node, path, origin, validated, out):
    """Invariant 5: every number sits in a block marked origin PITCHIQ-PROPOSAL, validated false."""
    if isinstance(node, dict):
        origin = node.get('origin', origin)
        validated = node.get('validated', validated)
        for k, v in node.items():
            if k in ('version',):
                continue
            _numbers_tagged(v, f'{path}.{k}', origin, validated, out)
    elif isinstance(node, list):
        for i, v in enumerate(node):
            _numbers_tagged(v, f'{path}[{i}]', origin, validated, out)
    elif _is_num(node) and (origin != S.ORIGIN_P or validated is not False):
        out.append(f'{path}: a number outside a PITCHIQ-PROPOSAL / validated false block')


def _ascii_text(v, n):
    return isinstance(v, str) and 0 < len(v) <= n and re.fullmatch(r'[ -~]+', v) is not None


def validate_rules(doc):
    """The problems of a progress rules document ([] when it is usable)."""
    out = []
    if not isinstance(doc, dict):
        return ['rules: not an object']
    if doc.get('kind') != RULES_KIND:
        out.append(f'rules.kind: must be {RULES_KIND!r}')
    if doc.get('version') != 1:
        out.append('rules.version: must be 1')
    if not _ascii_text(doc.get('label'), 120):
        out.append('rules.label: printable ASCII, 1..120 characters')
    if doc.get('origin') != S.ORIGIN_P or doc.get('validated') is not False:
        out.append('rules: origin PITCHIQ-PROPOSAL and validated false')
    blocks = ('interval', 'evidence', 'thresholds', 'adaptable', 'trend', 'rates')
    for b in blocks:
        blk = doc.get(b)
        if not isinstance(blk, dict):
            out.append(f'rules.{b}: missing')
            continue
        if blk.get('origin') != S.ORIGIN_P or blk.get('validated') is not False:
            out.append(f'rules.{b}: origin PITCHIQ-PROPOSAL and validated false')
        if not _ascii_text(blk.get('basis'), 1000):
            out.append(f'rules.{b}.basis: printable ASCII, 1..1000 characters')
    if out:
        return out
    iv = doc['interval']
    if iv.get('method') != 'wilson' or not _is_num(iv.get('confidence')) or not .5 <= iv['confidence'] < 1:
        out.append('rules.interval: method wilson, confidence in [0.5, 1)')
    ev = doc['evidence']
    for k in ('matchOpportunities', 'minOpportunities', 'minQualifyingMatches', 'consistentMatches'):
        if not isinstance(ev.get(k), int) or isinstance(ev.get(k), bool) or ev[k] < 1:
            out.append(f'rules.evidence.{k}: an integer >= 1')
    w = ev.get('windowQualifyingMatches')
    if w is not None and (not isinstance(w, int) or isinstance(w, bool) or w < ev.get('consistentMatches', 1)):
        out.append('rules.evidence.windowQualifyingMatches: null or an integer >= consistentMatches')
    if ev.get('minQuality') not in QUALITY_RANK:
        out.append('rules.evidence.minQuality: high, medium or low')
    th = doc['thresholds']
    for lv in range(1, 8):
        t = th.get(str(lv))
        if not (isinstance(t, dict) and _is_num(t.get('s1')) and _is_num(t.get('s2')) and 0 < t['s1'] < t['s2'] < 1):
            out.append(f'rules.thresholds.{lv}: {{s1, s2}} with 0 < s1 < s2 < 1')
    ad = doc['adaptable']
    for k in ('minContexts', 'contextOpportunities'):
        if not isinstance(ad.get(k), int) or isinstance(ad.get(k), bool) or ad[k] < 1:
            out.append(f'rules.adaptable.{k}: an integer >= 1')
    dims = ad.get('dimensions')
    if not isinstance(dims, dict) or set(dims) != set(S.SKILLS):
        out.append('rules.adaptable.dimensions: one list per skill')
    else:
        for skill, v in dims.items():
            if not isinstance(v, list) or not v or any(x not in CONTEXT_DIMENSIONS for x in v):
                out.append(f'rules.adaptable.dimensions.{skill}: a non-empty list of {CONTEXT_DIMENSIONS}')
    tr = doc['trend']
    if not isinstance(tr.get('previousMatches'), int) or isinstance(tr.get('previousMatches'), bool) or \
            tr['previousMatches'] < 1:
        out.append('rules.trend.previousMatches: an integer >= 1')
    if not _is_num(tr.get('minDelta')) or not 0 <= tr['minDelta'] < 1:
        out.append('rules.trend.minDelta: in [0, 1)')
    if not _is_num(tr.get('confidence')) or not .5 <= tr['confidence'] < 1:
        out.append('rules.trend.confidence: in [0.5, 1)')
    rt = doc['rates']
    if not _is_num(rt.get('perSeconds')) or rt['perSeconds'] <= 0:
        out.append('rules.rates.perSeconds: a positive number')
    if not _is_num(rt.get('shortRangeSeconds')) or rt['shortRangeSeconds'] < 0:
        out.append('rules.rates.shortRangeSeconds: a number >= 0')
    _numbers_tagged(doc, 'rules', None, None, out)
    return out


def load_rules(path=None):
    """The validated progress rules (cached per file and modification time) with '_sha' (16 hex of the canonical
    sha256) and '_z' (the interval's z). Raises ProgressError listing the problems."""
    p = Path(path or RULES_PATH)
    st = p.stat()
    key = ('rules', str(p.resolve()), st.st_mtime_ns, st.st_size)
    hit = _cache.get('rules')
    if hit is not None and hit[0] == key:
        return hit[1]
    doc = json.loads(p.read_bytes().decode('utf-8'), parse_constant=_refuse_constant)
    problems = validate_rules(doc)
    if problems:
        raise ProgressError('progress_rules.json invalid: '+'; '.join(problems[:20]))
    doc['_sha'] = _canonical_sha({k: v for k, v in doc.items() if not k.startswith('_')})[:16]
    doc['_z'] = statistics.NormalDist().inv_cdf(.5+float(doc['interval']['confidence'])/2.0)
    doc['_zTrend'] = statistics.NormalDist().inv_cdf(.5+float(doc['trend']['confidence'])/2.0)
    _cache['rules'] = (key, doc)
    return doc


# ---------- the outcome table ----------
def _semantic(fn):
    """The AST of a function without its docstring and positions: what it computes, not how it is written down."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
    for node in ast.walk(tree):
        body = getattr(node, 'body', None)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and body and isinstance(body[0], ast.Expr) and \
                isinstance(getattr(body[0], 'value', None), ast.Constant) and isinstance(body[0].value.value, str):
            node.body = body[1:] or [ast.Pass()]
    return ast.dump(tree, annotate_fields=False, include_attributes=False)


def classify_code_sha():
    """12 hex of the classification code and constants (CLASSIFY_CODE, CLASSIFY_VALUES), cached per process."""
    hit = _cache.get('code')
    if hit is None:
        parts = [f'{m}.{n}:{_semantic(getattr(_MODULES[m], n))}' for m, n in CLASSIFY_CODE]
        parts += [f'{m}.{n}={json.dumps(getattr(_MODULES[m], n), sort_keys=True)}' for m, n in CLASSIFY_VALUES]
        hit = hashlib.sha1('\n'.join(parts).encode('utf-8')).hexdigest()[:12]
        _cache['code'] = hit
    return hit


def classify_params(params):
    """The part of a detector profile's params an outcome depends on (CLASSIFY_KEYS present in it; a key set to None
    is kept: for minBehind / minMargin None and missing differ)."""
    return {k: params[k] for k in CLASSIFY_KEYS if k in (params or {})}


def outcome_key(detector, params, as_detector=None):
    """The rel key of an opportunity's outcome under a detector profile's params: 'o' + 10 hex of (the classification
    code, the opportunity's detector, the parameters it reads, and as_detector when the opportunity is judged with
    another detector's rule: level 1 reads a wide defender's advance with the behind_ball rule, DESIGN 4.3)."""
    doc = {'c': classify_code_sha(), 'd': detector, 'p': classify_params(params)}
    if as_detector:
        doc['a'] = as_detector
    text = json.dumps(doc, sort_keys=True, separators=(',', ':'), allow_nan=False)
    return OUTCOME_PREFIX+hashlib.sha1(text.encode('utf-8')).hexdigest()[:OUTCOME_HEX]


def _players(fmt):
    if isinstance(fmt, dict):
        return int(fmt['players'])
    return int(str(fmt).split('v')[0])


def reframed(profile):
    """Does the profile read wide-defender advances as "someone ready to protect our goal" (DESIGN 4.3 level 1: D3
    off with reframedAs behind_ball, D4 on), as assess.assess does?"""
    wide, back = profile['detectors']['wide_defender_advance'], profile['detectors']['behind_ball']
    return back['enabled'] and not wide['enabled'] and wide['params'].get('reframedAs') == 'behind_ball'


def param_sets(players, data=None):
    """{detector: {outcome key: (params, as_detector)}} over every profile profiles.json resolves for a match
    format (7 levels x the age groups x the styles; disabled detectors skipped; a level that reframes D3 adds the
    behind_ball reading of D3's opportunities), cached per (profiles sha, players, code)."""
    data = data or PR.load()
    players = _players(players)
    ck = ('sets', data['_sha'], players, classify_code_sha())
    hit = _cache.get(ck)
    if hit is not None:
        return hit
    out = {d: {} for d in S.DETECTORS}
    for level in range(1, 8):
        for age in S.AGE_GROUPS:
            for style in S.STYLES:
                prof = PR.resolve({'ageGroup': age, 'playingLevel': level, 'style': style,
                                   'format': {'players': players}}, data=data)
                for d in S.DETECTORS:
                    dp = prof['detectors'][d]
                    if dp['enabled']:
                        out[d].setdefault(outcome_key(d, dp['params']), (classify_params(dp['params']), None))
                if reframed(prof):
                    back = prof['detectors']['behind_ball']['params']
                    out['wide_defender_advance'].setdefault(
                        outcome_key('wide_defender_advance', back, 'behind_ball'),
                        (classify_params(back), 'behind_ball'))
    _cache[ck] = out
    return out


# ---------- entries ----------
def _summary(values, how, digits):
    vals = [v for v in (values or ()) if v is not None and not (isinstance(v, float) and not math.isfinite(v))]
    if not vals:
        return None
    if how == 'share':
        return round(sum(1 for v in vals if v is True or (not isinstance(v, bool) and v))/len(vals), digits)
    nums = [v for v in vals if _is_num(v)]
    if not nums:
        return None
    if how == 'median':
        v = statistics.median(nums)
    elif how == 'min':
        v = min(nums)
    else:
        v = max(nums)
    if digits is None and float(v).is_integer():
        return int(v)
    return round(float(v), digits if digits is not None else 2)


def _opp_quality(o):
    q = [o['quality']['position']]
    if o['detector'] not in TEAM_LEVEL:
        q.append(o['quality']['identity'])
    return max(q, key=lambda x: QUALITY_RANK[x])


def _progress_opp(o, sets, scale, step):
    d = o['detector']
    R = o['relations']
    rel = {}
    for name, key, how, digits in SUMMARY[d]:
        v = _summary(R.get(key), how, digits)
        if v is not None:
            rel[name] = v
    rel['samples'] = len(R['t'])
    rel['coachPhase'] = o['phase']['source'] == 'coach'
    if d == 'wide_defender_advance' and o.get('variant') in ('left', 'right'):
        rel['leftSide'] = o['variant'] == 'left'
    for key, (params, as_detector) in sets.items():
        c = AS.classify(o, params, scale, step, as_detector=as_detector)
        rel[key] = None if c is None else OUTCOME_CODE[c['outcome']]
    return {'t0': float(o['t0']), 't1': float(o['t1']), 'zone': o['ball']['zone'], 'channel': o['ball']['channel'],
            'rel': rel, 'quality': _opp_quality(o)}


def _range(obs, t0, t1):
    seg = obs['segment']
    lo = float(seg['t0']) if t0 is None else max(float(seg['t0']), float(t0))
    hi = float(seg['t1']) if t1 is None else min(float(seg['t1']), float(t1))
    if hi < lo:
        raise ProgressError(f'the range {t0}-{t1} lies outside the analysed segment {seg["t0"]}-{seg["t1"]}')
    return lo, hi


def counted(o, lo, hi):
    """Does an opportunity enter progression for the range: not abstained, in the detector's relevant phase (a D6
    describe-only opportunity without a known phase never does), its middle inside [lo, hi] (assess's range rule)."""
    if o['abstain'] is not None or o['phase']['state'] != S.DETECTOR_PHASES[o['detector']][0]:
        return False
    return lo-1e-6 <= (o['t0']+o['t1'])/2.0 <= hi+1e-6


def _confidence(items, timeline):
    if not items:
        return 'low'
    ranks = sorted(QUALITY_RANK[p['quality']] for p in items)
    conf = ('high', 'medium', 'low')[ranks[len(ranks)//2]]
    pq = timeline.get('phaseQuality') or {}
    engine_only = not any(p['rel'].get('coachPhase') for p in items)
    if conf == 'high' and engine_only and pq.get('level') != 'good':
        conf = 'medium'             # the phase rests on a possession estimate of limited quality
    return conf


def build_entry(obs, roles=None, *, squad, video, run, match_date, date_source, level, age_group, t0=None, t1=None,
                updated=None, data=None):
    """The progress entry (schema 'progress_entry') of one team's analysed range: per skill its opportunities with
    level-independent relations and the outcome table, the relevant phase seconds, the eligible share and a
    confidence; obs / roles are the cached observation and role files (never modified). level / age_group are the
    team's settings when it was analysed (context only: readings use the squad's current profile)."""
    if not (isinstance(squad, str) and re.fullmatch(r'sq-[0-9a-f]{12}', squad)):
        raise ProgressError(f'squad {squad!r} is not a squad id')
    lo, hi = _range(obs, t0, t1)
    tl = obs['timeline']
    scale = tl['scale']
    step = float(tl.get('step') or .2)
    fmt = (obs.get('inputs') or {}).get('measurement', {}).get('format') or tl['format']
    players = _players(fmt)
    sets = param_sets(players, data)
    opps = [o for o in list(obs.get('opportunities') or [])+list((roles or {}).get('opportunities') or [])
            if counted(o, lo, hi)]
    skills = {}
    for skill, d in S.SKILLS.items():
        items = sorted((o for o in opps if o['detector'] == d), key=lambda o: (o['t0'], o['t1'], o['id']))
        rows = [_progress_opp(o, sets[d], scale, step) for o in items]
        phase_s = AS._phase_seconds(tl, S.DETECTOR_PHASES[d][0], lo, hi)
        elig = CX.eligible_seconds(items, step)
        skills[skill] = {'opportunities': rows, 'phaseSeconds': round(phase_s, 2),
                         'eligibleShare': round(min(1.0, elig/phase_s), 3) if phase_s > 0 else None,
                         'confidence': _confidence(rows, tl)}
    team = obs['team']
    entry = {'id': S.record_id('pe', squad, video, run, team, lo, hi), 'squad': squad, 'video': video, 'run': run,
             'team': team, 'range': {'t0': round(lo, RANGE_DIGITS), 't1': round(hi, RANGE_DIGITS)},
             'matchDate': match_date, 'dateSource': date_source,
             'engine': {'measurement': obs['engine']['measurement'], 'version': S.ENGINE_VERSION},
             'context': {'format': S.format_name({'players': players}), 'footage': tl['footage']['kind'],
                         'phaseAvailable': AS.phase_available(obs), 'level': int(level), 'ageGroup': age_group},
             'skills': skills, 'updated': updated or now_iso()}
    return S.check('progress_entry', entry)


def rederive(entry, obs, roles=None, data=None, updated=None):
    """The entry made again from the run's current obs / roles files (same squad, video, run, team, range, date and
    context): for a stale entry or one whose outcome table misses the current rules."""
    if obs['team'] != entry['team']:
        raise ProgressError('the observation file is another team\'s')
    ctx = entry['context']
    return build_entry(obs, roles, squad=entry['squad'], video=entry['video'], run=entry['run'],
                       match_date=entry['matchDate'], date_source=entry['dateSource'], level=ctx['level'],
                       age_group=ctx['ageGroup'], t0=entry['range']['t0'], t1=entry['range']['t1'], updated=updated,
                       data=data)


def needs_rederive(entry, data=None):
    """None when the entry can be read with the current engine and rules, else why not: 'engine' (an earlier
    measurement engine: stale, or another engine version) or 'rules' (its outcome table misses a parameter set of
    the current profiles.json or classification code)."""
    if entry.get('stale') or entry['engine']['version'] != S.ENGINE_VERSION:
        return 'engine'
    sets = param_sets(entry['context']['format'], data)
    for skill, rec in entry['skills'].items():
        want = set(sets[S.SKILLS[skill]])
        for o in rec['opportunities']:
            if not want <= set(o['rel']):
                return 'rules'
    return None


def now_iso():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def match_date_of(coach_date=None, video_path=None):
    """(matchDate, dateSource): the coach's date (YYYY-MM-DD) when given, else the local date of the video file's
    modification time. Raises ProgressError when neither is available (a progress entry never invents a date)."""
    if coach_date is not None:
        try:
            return date.fromisoformat(str(coach_date)).isoformat(), 'coach'
        except ValueError:
            raise ProgressError(f'match date {coach_date!r} is not YYYY-MM-DD') from None
    if video_path is not None and Path(video_path).exists():
        return datetime.fromtimestamp(Path(video_path).stat().st_mtime).date().isoformat(), 'file_mtime'
    raise ProgressError('no match date: give the coach\'s date or the video file')


# ---------- squads and links (DESIGN 8.1) ----------
def links(settings, run):
    """The squads named in a video's coaching settings and their team in this run: [{'block' (the settings letter),
    'squad' {id, name}, 'team' (this run's letter) | None, 'confirmed', 'question' | None}]. A run without a confirmed
    link writes no progress; the question is what the page asks ("Which team is U10 Blue in this run?")."""
    out, seen = [], set()
    teams = (settings or {}).get('teams') or {}
    for letter in S.TEAMS:
        block = teams.get(letter)
        sq = (block or {}).get('squad')
        if not sq or sq.get('id') in seen:
            continue
        seen.add(sq['id'])
        link = next((x for x in (block.get('runLinks') or ()) if x.get('run') == run), None)
        confirmed = bool(link and link.get('confirmed') and link.get('team') in S.TEAMS)
        out.append({'block': letter, 'squad': {'id': sq['id'], 'name': sq['name']},
                    'team': link['team'] if link and link.get('team') in S.TEAMS else None, 'confirmed': confirmed,
                    'question': None if confirmed else f'Which team is {sq["name"]} in this run?'[:200]})
    return out


def confirmed_team(settings, run, squad_id):
    """The team letter a squad is confirmed as in this run, else None."""
    for x in links(settings, run):
        if x['squad']['id'] == squad_id and x['confirmed']:
            return x['team']
    return None


# ---------- files ----------
def store_dir(root):
    return Path(root)/STORE_DIR


def squads_path(root):
    return store_dir(root)/SQUADS_FILE


def progress_path(root):
    return store_dir(root)/PROGRESS_FILE


def _refuse_constant(token):
    raise ValueError(f'{token} is not a finite number')


def _replace(tmp, path):
    for attempt in range(REPLACE_TRIES):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if attempt == REPLACE_TRIES-1:
                Path(tmp).unlink(missing_ok=True)
                raise
            time.sleep(REPLACE_WAIT)


def _write(path, doc):
    path = Path(path)
    try:
        text = json.dumps(doc, allow_nan=False, separators=(',', ':'), ensure_ascii=True)
    except ValueError:
        raise ProgressUnreadable(f'{path.name} holds a number that is not finite; nothing was changed.') from None
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f'.{path.name}.{uuid.uuid4().hex}.tmp')
    tmp.write_text(text, encoding='utf-8')
    _replace(tmp, path)


def _stat_key(path):
    try:
        st = os.stat(path)
    except FileNotFoundError:
        return None
    return st.st_mtime_ns, st.st_size


def _parse(path, list_key, strict):
    """The document of a store file, or None (missing; or, not strict, unreadable). Strict: an unreadable or foreign
    file raises ProgressUnreadable, never reads as empty."""
    for attempt in range(READ_TRIES):
        try:
            if not path.exists():
                return None
            doc = json.loads(path.read_text(encoding='utf-8'), parse_constant=_refuse_constant)
            break
        except (ValueError, TypeError, RecursionError):
            if strict:
                raise ProgressUnreadable(f'{path.name} could not be read; nothing was changed.') from None
            return None
        except OSError:
            if attempt == READ_TRIES-1:
                if strict:
                    raise ProgressUnreadable(f'{path.name} could not be read; nothing was changed.') from None
                return None
            time.sleep(READ_WAIT)
    if not (isinstance(doc, dict) and doc.get('version') == FILE_VERSION and isinstance(doc.get(list_key), list)):
        if strict:
            raise ProgressUnreadable(f'{path.name} is not in the expected format; nothing was changed.')
        return None
    return doc


def _load(path, list_key, strict):
    """(document, validation flags) of a store file, (None, None) when missing. A file version (modification time,
    size) is parsed once per process and its rows validated lazily: a 30-match season of two squads (60 entries of
    90 minutes, 15 MB) takes about a second to parse and validate, and nothing after this process's own writes."""
    path = Path(path)
    key = _stat_key(path)
    if key is None:
        return None, None
    hit = _docs.get(str(path))
    if hit is not None and hit[0] == key:
        return hit[1], hit[2]
    doc = _parse(path, list_key, strict)
    if doc is None:
        return None, None
    flags = [None]*len(doc[list_key])
    _docs[str(path)] = (key, doc, flags)
    return doc, flags


def _keep(path, doc, flags):
    """Write a new document version and remember it with its flags (the cached documents are never changed in
    place: writers build new lists and copy the rows they change)."""
    try:
        _write(path, doc)
    except BaseException:
        _docs.pop(str(path), None)
        raise
    key = _stat_key(path)
    if key is not None:
        _docs[str(path)] = (key, doc, flags)


def _valid_rows(rows, flags, check, select=None):
    out = []
    for i, x in enumerate(rows):
        if select is not None and not (isinstance(x, dict) and select(x)):
            continue
        if flags[i] is None:
            flags[i] = isinstance(x, dict) and check(x)
        if flags[i]:
            out.append(x)
    return out


def _squad_ok(s):
    return not S.validate('squads', {'version': FILE_VERSION, 'squads': [s]})


def _entry_ok(e):
    return not S.validate('progress_entry', e)


def read_squads(root, strict=False):
    """The squads (schema 'squads' records); a missing file is []. Not strict: malformed squads are skipped."""
    doc, flags = _load(squads_path(root), 'squads', strict)
    if doc is None:
        return []
    if strict:
        return [s for s in doc['squads'] if isinstance(s, dict)]
    return _valid_rows(doc['squads'], flags, _squad_ok)


def squad(root, squad_id):
    """One squad by id, or None."""
    return next((s for s in read_squads(root) if s['id'] == squad_id), None)


def _squad_record(name, age_group, level, fmt, created, squad_id=None):
    if not _ascii_text(name, 60):
        raise ProgressError('a squad name is printable ASCII, 1..60 characters')
    problems = []
    clean = PR.clean_team({'format': fmt, 'ageGroup': age_group, 'playingLevel': level}, problems=problems)
    if fmt is None or any('.format' in x or '.ageGroup' in x or '.playingLevel' in x for x in problems):
        raise ProgressError(f'a squad needs a format of 4, 5, 7, 9 or 11 players, an age group and a level 1-7 '
                            f'({"; ".join(problems) or "no format"})')
    clean = clean['format']
    rec = {'id': squad_id or S.record_id('sq', name.strip().casefold(), created), 'name': name.strip(),
           'ageGroup': age_group, 'level': level, 'format': clean, 'created': created}
    problems = S.validate('squads', {'version': 1, 'squads': [rec]})
    if problems:
        raise ProgressError('; '.join(problems[:5]))
    return rec


def create_squad(root, name, age_group, level, fmt, now=None):
    """(squad, created): a new squad, or the existing one with the same name (case-insensitive) and created False."""
    created = now or now_iso()
    rec = _squad_record(name, age_group, level, fmt, created)
    with _LOCK:
        path = squads_path(root)
        doc, flags = _load(path, 'squads', strict=True)
        rows, flags = (list(doc['squads']), list(flags)) if doc else ([], [])
        for s in rows:
            if isinstance(s, dict) and str(s.get('name', '')).strip().casefold() == rec['name'].casefold():
                return copy.deepcopy(s), False
        _keep(path, {'version': FILE_VERSION, 'squads': rows+[rec]}, flags+[True])
    return copy.deepcopy(rec), True


def update_squad(root, squad_id, *, name=None, age_group=None, level=None, fmt=None):
    """The squad with its current name, age group, level or format changed (the profile its progress is read
    with); SERVER keeps it in step with the latest settings save of a linked team."""
    with _LOCK:
        path = squads_path(root)
        doc, flags = _load(path, 'squads', strict=True)
        rows, flags = (list(doc['squads']), list(flags)) if doc else ([], [])
        for k, s in enumerate(rows):
            if isinstance(s, dict) and s.get('id') == squad_id:
                new_name = name if name is not None else s['name']
                if any(isinstance(o, dict) and o.get('id') != squad_id and
                       str(o.get('name', '')).strip().casefold() == str(new_name).strip().casefold() for o in rows):
                    raise ProgressError(f'another squad is called {new_name!r}')
                rec = _squad_record(new_name, age_group or s['ageGroup'], level or s['level'],
                                    fmt or s['format'], s['created'], squad_id=squad_id)
                rows[k], flags[k] = rec, True
                _keep(path, {'version': FILE_VERSION, 'squads': rows}, flags)
                return copy.deepcopy(rec)
    raise ProgressError(f'no squad {squad_id!r}')


def read_entries(root, squad_id=None, strict=False):
    """The progress entries (of one squad when squad_id is given); a missing file is []. Not strict: entries that do
    not validate are skipped. The entries are the cache's own objects: read them, copy before changing them."""
    doc, flags = _load(progress_path(root), 'entries', strict)
    if doc is None:
        return []
    select = None if squad_id is None else (lambda e: e.get('squad') == squad_id)
    if strict:
        return [e for e in doc['entries'] if isinstance(e, dict) and (select is None or select(e))]
    return _valid_rows(doc['entries'], flags, _entry_ok, select)


def _overlap(a, b):
    return min(a['t1'], b['t1'])-max(a['t0'], b['t0']) > OVERLAP_EPS_S


def _length(e):
    return float(e['range']['t1'])-float(e['range']['t0'])


def _entries_for_write(root):
    doc, flags = _load(progress_path(root), 'entries', strict=True)
    return (list(doc['entries']), list(flags)) if doc else ([], [])


def upsert(root, entry):
    """Store an entry: the same (squad, video, run, team, range) is replaced; an overlapping range of the same squad
    and video keeps only the longest (a tie: the new one), so a moment is never counted twice. Returns {'written',
    'id', 'replaced' [ids], 'keptLonger' id | None}."""
    entry = copy.deepcopy(S.check('progress_entry', entry))
    with _LOCK:
        rows, flags = _entries_for_write(root)
        same = [e for e in rows if isinstance(e, dict) and e.get('id') == entry['id']]
        over = [e for e in rows if isinstance(e, dict) and e.get('id') != entry['id'] and
                e.get('squad') == entry['squad'] and e.get('video') == entry['video'] and
                isinstance(e.get('range'), dict) and _overlap(e['range'], entry['range'])]
        longer = [e for e in over if _length(e) > _length(entry)+1e-9]
        if longer:
            return {'written': False, 'id': entry['id'], 'replaced': [], 'keptLonger': longer[0]['id']}
        drop = {id(e) for e in same+over}
        kept = [(e, f) for e, f in zip(rows, flags) if id(e) not in drop]+[(entry, True)]
        _keep(progress_path(root), {'version': FILE_VERSION, 'entries': [e for e, _ in kept]}, [f for _, f in kept])
    return {'written': True, 'id': entry['id'], 'replaced': [e['id'] for e in same+over], 'keptLonger': None}


def _edit(root, match, change):
    """Apply change(entry) -> bool (changed) to copies of the stored entries match(entry) selects; returns how many
    changed."""
    with _LOCK:
        rows, flags = _entries_for_write(root)
        n = 0
        for i, e in enumerate(rows):
            if isinstance(e, dict) and match(e):
                e2 = copy.deepcopy(e)
                if change(e2):
                    rows[i], flags[i] = e2, None
                    n += 1
        if n:
            _keep(progress_path(root), {'version': FILE_VERSION, 'entries': rows}, flags)
    return n


def remove(root, *, squad_id, video=None, run=None, team=None):
    """Remove a squad's entries (of one video / run / team when given), e.g. when a coach unlinks a wrong team.
    Returns how many were removed."""
    def hit(e):
        return isinstance(e, dict) and e.get('squad') == squad_id and \
            all(v is None or e.get(k) == v for k, v in (('video', video), ('run', run), ('team', team)))
    with _LOCK:
        rows, flags = _entries_for_write(root)
        kept = [(e, f) for e, f in zip(rows, flags) if not hit(e)]
        n = len(rows)-len(kept)
        if n:
            _keep(progress_path(root), {'version': FILE_VERSION, 'entries': [e for e, _ in kept]},
                  [f for _, f in kept])
    return n


def mark_stale(root, *, video, run, team, measurement):
    """Entries of this video, run and team made with another measurement fingerprint (16 hex) become stale (and
    those made with it are not); returns how many changed. SERVER calls it whenever it builds or serves obs."""
    def change(e):
        stale = (e.get('engine') or {}).get('measurement') != measurement
        if bool(e.get('stale')) == stale:
            return False
        if stale:
            e['stale'] = True
        else:
            e.pop('stale', None)
        return True
    return _edit(root, lambda e: e.get('video') == video and e.get('run') == run and e.get('team') == team, change)


def set_match_date(root, *, squad_id, video, day):
    """The coach's date (YYYY-MM-DD) for every entry of a squad's video (dateSource 'coach'); returns how many
    changed."""
    day, _ = match_date_of(day)

    def change(e):
        if e.get('matchDate') == day and e.get('dateSource') == 'coach':
            return False
        e['matchDate'], e['dateSource'] = day, 'coach'
        return True
    return _edit(root, lambda e: e.get('squad') == squad_id and e.get('video') == video, change)


def record(root, settings, *, run, video, tiers, t0=None, t1=None, coach_date=None, video_path=None, now=None,
           data=None):
    """Write the progress of every squad confirmed in this run (DESIGN 8.1): tiers = {team letter: (obs, roles)} of
    the analysed range. Returns per linked squad {'squad', 'team', 'written', 'id', 'replaced', 'keptLonger',
    'reason'}; an unconfirmed link or a team without observations writes nothing (reason 'unconfirmed' /
    'no_observations')."""
    out = []
    for link in links(settings, run):
        row = {'squad': link['squad']['id'], 'team': link['team'], 'written': False, 'id': None, 'replaced': [],
               'keptLonger': None, 'reason': None}
        if not link['confirmed']:
            row['reason'] = 'unconfirmed'
        elif link['team'] not in (tiers or {}):
            row['reason'] = 'no_observations'
        else:
            block = S.team_block(settings, link['block']) or {}
            obs, roles = tiers[link['team']]
            day, source = match_date_of(coach_date, video_path)
            clean = PR.clean_team(block)
            entry = build_entry(obs, roles, squad=link['squad']['id'], video=video, run=run, t0=t0, t1=t1,
                                match_date=day, date_source=source, level=clean['playingLevel'],
                                age_group=clean['ageGroup'], updated=now, data=data)
            row.update(upsert(root, entry))
        out.append(row)
    return out


# ---------- reading: outcomes, mastery, the report section ----------
def wilson(demonstrated, n, z):
    """(lower, upper) Wilson score interval of demonstrated / n at the normal quantile z; (0, 1) without data."""
    if n <= 0:
        return 0.0, 1.0
    p = demonstrated/n
    z2 = z*z
    den = 1.0+z2/n
    centre = (p+z2/(2.0*n))/den
    half = z*math.sqrt(p*(1.0-p)/n+z2/(4.0*n*n))/den
    return max(0.0, centre-half), min(1.0, centre+half)


def read_block(squad_rec=None, block=None):
    """The settings a squad's progress is read with: a team settings block (the viewed run's; its age group, level and
    style) when given, else the squad record's age group, level and format with style unknown."""
    if block:
        b = {k: block[k] for k in ('ageGroup', 'playingLevel', 'style', 'format') if block.get(k) is not None}
    else:
        b = {}
    if squad_rec:
        b.setdefault('ageGroup', squad_rec['ageGroup'])
        b.setdefault('playingLevel', squad_rec['level'])
        b.setdefault('format', squad_rec['format'])
    return PR.clean_team(b)


class _Reader:
    """Profiles of the squad's current settings per match format (an entry is read with its own format's rules)."""

    def __init__(self, block, data):
        self.block, self.data, self.cache = block, data, {}

    def profile(self, players):
        players = _players(players)
        if players not in self.cache:
            b = {'ageGroup': self.block['ageGroup'], 'playingLevel': self.block['playingLevel'],
                 'style': self.block.get('style') or 'unknown', 'format': {'players': players}}
            self.cache[players] = PR.resolve(b, data=self.data)
        return self.cache[players]


def outcomes(entry, profile, skill):
    """[(ProgressOpp, outcome code 1 / -1 / 0 / None)] of one skill of an entry under a profile (that of the entry's
    format), or None when the entry has no outcome for the profile's parameters (needs_rederive 'rules'); [] when
    the profile does not judge the skill. At a level that reframes D3 (reframed()), "someone back when attacking"
    also reads the wide defenders' advances with the behind_ball rule, as the report does (its 'advance' family),
    except an advance overlapping a judged behind_ball opportunity: a moment is never counted twice."""
    d = S.SKILLS[skill]
    dp = profile['detectors'][d]
    if not dp['enabled']:
        return []
    out = _read_outcomes(entry, skill, outcome_key(d, dp['params']))
    if out is None or d != 'behind_ball' or not reframed(profile):
        return out
    extra = _read_outcomes(entry, SKILL_OF['wide_defender_advance'],
                           outcome_key('wide_defender_advance', dp['params'], 'behind_ball'))
    if extra is None:
        return None
    judged = [o for o, v in out if v is not None]
    return out+[(o, v) for o, v in extra if not (v is not None and any(_overlap(o, j) for j in judged))]


def _read_outcomes(entry, skill, key):
    rows = entry['skills'].get(skill, {}).get('opportunities') or []
    out = []
    for o in rows:
        if key not in o['rel']:
            return None
        out.append((o, o['rel'][key]))
    return out


def _context_value(dim, o, entry):
    if dim == 'zone':
        return o['zone']
    if dim == 'channel':
        return o['channel']
    if dim == 'side':
        side = o['rel'].get('leftSide')
        return None if side is None else ('left' if side else 'right')
    return entry['context']['format']


def _shown(profile, skill):
    """A skill row is shown when the level names it and can say more than a description about it."""
    dp = profile['detectors'][S.SKILLS[skill]]
    return bool(profile['skills'].get(skill)) and dp['enabled'] and any(v != 'describe' for v in dp['verdicts'])


def _match_rows(entries, reader, skill, min_rank):
    """Per match (video) of usable entries: judged counts, phase seconds and the judged opportunities with their
    contexts; and the entries that could not be read with the current rules."""
    by_video, unreadable = {}, []
    for e in entries:
        prof = reader.profile(e['context']['format'])
        got = outcomes(e, prof, skill)
        if got is None:
            unreadable.append(e)
            continue
        rec = e['skills'].get(skill) or {'phaseSeconds': 0.0, 'opportunities': []}
        m = by_video.setdefault(e['video'], {'video': e['video'], 'date': e['matchDate'], 'entries': [],
                                             'demonstrated': 0, 'notDemonstrated': 0, 'unclear': 0, 'outOfScope': 0,
                                             'lowQuality': 0, 'coachMarked': 0, 'phaseSeconds': 0.0, 'seconds': 0.0,
                                             'judged': []})
        m['date'] = min(m['date'], e['matchDate'])
        m['entries'].append(e)
        m['phaseSeconds'] += float(rec['phaseSeconds'])
        m['seconds'] += _length(e)
        for o, v in got:
            if v is None:
                m['outOfScope'] += 1
            elif QUALITY_RANK[o['quality']] > min_rank:
                m['lowQuality'] += 1
            elif v == 0:
                m['unclear'] += 1
            else:
                m['demonstrated' if v == 1 else 'notDemonstrated'] += 1
                m['judged'].append((o, e, v))
                m['coachMarked'] += int(bool(o['rel'].get('coachPhase')))
    rows = sorted(by_video.values(), key=lambda m: (m['date'], m['video']))
    for m in rows:
        m['n'] = m['demonstrated']+m['notDemonstrated']
        m['share'] = m['demonstrated']/m['n'] if m['n'] else None
    return rows, unreadable


def _pooled(matches):
    d = sum(m['demonstrated'] for m in matches)
    n = sum(m['n'] for m in matches)
    return d, n


def _trend(qual, rules, z):
    """'up' / 'down' / 'flat' between the latest qualifying match and the previous ones (per opportunity), 'unknown'
    with fewer than two qualifying matches."""
    if len(qual) < 2:
        return 'unknown', None
    tr = rules['trend']
    last = qual[-1]
    before = qual[-1-int(tr['previousMatches']):-1]
    d1, n1 = _pooled(before)
    d2, n2 = last['demonstrated'], last['n']
    p1, p2 = d1/n1, d2/n2
    pooled = (d1+d2)/(n1+n2)
    se = math.sqrt(pooled*(1.0-pooled)*(1.0/n1+1.0/n2))
    zval = (p2-p1)/se if se > 0 else (0.0 if p2 == p1 else math.copysign(math.inf, p2-p1))
    detail = {'previousShare': round(p1, 3), 'previousOpportunities': n1, 'latestShare': round(p2, 3),
              'latestOpportunities': n2, 'z': round(zval, 2) if math.isfinite(zval) else None}
    if p2-p1 >= float(tr['minDelta'])-1e-9 and zval >= z:
        return 'up', detail
    if p1-p2 >= float(tr['minDelta'])-1e-9 and -zval >= z:
        return 'down', detail
    return 'flat', detail


def _contexts(judged, dims, rules, z, s2):
    ad = rules['adaptable']
    out = {}
    passing = 0
    for dim in dims:
        vals = {}
        for o, e, v in judged:
            c = _context_value(dim, o, e)
            if c is None:
                continue
            cell = vals.setdefault(c, [0, 0])
            cell[0] += int(v == 1)
            cell[1] += 1
        rows = {}
        ok = 0
        for c, (d, n) in sorted(vals.items()):
            lo, _ = wilson(d, n, z)
            good = n >= int(ad['contextOpportunities']) and lo >= s2-1e-9
            ok += int(good)
            rows[c] = {'opportunities': n, 'demonstrated': d, 'lower': round(lo, 3), 'consistent': good}
        out[dim] = rows
        passing = max(passing, ok)
    return out, passing >= int(ad['minContexts'])


def _rate(count, seconds, per):
    return round(float(count)*per/seconds, 2) if seconds > 0 else None


def _skill(entries, reader, skill, profile, rules):
    z = rules['_z']
    ev, th = rules['evidence'], rules['thresholds'][str(profile['level'])]
    s1, s2 = float(th['s1']), float(th['s2'])
    rows, unreadable = _match_rows(entries, reader, skill, QUALITY_RANK[ev['minQuality']])
    need = int(ev['matchOpportunities'])
    for m in rows:
        m['qualifies'] = m['n'] >= need
    qual_all = [m for m in rows if m['qualifies']]
    k = ev.get('windowQualifyingMatches')
    start = 0
    if k is not None and len(qual_all) > int(k):
        start = next(i for i, m in enumerate(rows) if m is qual_all[-int(k)])
    window = rows[start:]
    qual = [m for m in window if m['qualifies']]
    d, n = _pooled(window)
    lo, hi = wilson(d, n, z)
    per = float(rules['rates']['perSeconds'])
    phase_s = sum(m['phaseSeconds'] for m in window)
    last = qual[-int(ev['consistentMatches']):] if len(qual) >= int(ev['consistentMatches']) else []
    d3, n3 = _pooled(last)
    lo3 = wilson(d3, n3, z)[0] if last else None
    consistent = bool(last) and lo3 >= s2-1e-9 and all(m['share'] >= s1-1e-9 for m in last)
    judged = [x for m in window for x in m['judged']]
    contexts, adaptable = _contexts(judged, rules['adaptable']['dimensions'][skill], rules, z, s2)
    if n < int(ev['minOpportunities']) or len(qual) < int(ev['minQualifyingMatches']):
        state, reason = 'NOT ENOUGH EVIDENCE', ('few_opportunities' if n < int(ev['minOpportunities'])
                                                else 'few_matches')
    elif consistent:
        state, reason = ('ADAPTABLE', 'contexts') if adaptable else ('CONSISTENT', 'recent_matches')
    elif lo >= s1-1e-9:
        state, reason = 'DEVELOPING', 'lower_bound'
    else:
        state, reason = 'LEARNING', 'lower_bound'
    trend, trend_detail = _trend(qual, rules, rules['_zTrend'])
    coach = sum(m['coachMarked'] for m in window)
    history = [{'video': m['video'], 'matchDate': m['date'], 'runs': sorted({e['run'] for e in m['entries']}),
                'seconds': round(m['seconds'], 1), 'phaseSeconds': round(m['phaseSeconds'], 1),
                'opportunities': m['n'], 'demonstrated': m['demonstrated'], 'notDemonstrated': m['notDemonstrated'],
                'unclear': m['unclear'], 'outOfScope': m['outOfScope'], 'lowQuality': m['lowQuality'],
                'share': round(m['share'], 3) if m['share'] is not None else None, 'qualifies': m['qualifies'],
                'inWindow': i >= start, 'opportunitiesPer10': _rate(m['n'], m['phaseSeconds'], per)}
               for i, m in enumerate(rows)]
    return {'skill': skill, 'detector': S.SKILLS[skill], 'name': profile['skills'].get(skill),
            'shown': _shown(profile, skill), 'state': state, 'reason': reason,
            'opportunities': n, 'demonstrated': d, 'notDemonstrated': n-d,
            'unclear': sum(m['unclear'] for m in window), 'outOfScope': sum(m['outOfScope'] for m in window),
            'lowQuality': sum(m['lowQuality'] for m in window),
            'share': round(d/n, 3) if n else None, 'lower': round(lo, 3), 'upper': round(hi, 3), 's1': s1, 's2': s2,
            'matches': sum(1 for m in window if m['n'] > 0), 'qualifyingMatches': len(qual),
            'recent': {'matches': len(last), 'opportunities': n3, 'lower': round(lo3, 3) if lo3 is not None else None,
                       'consistent': consistent},
            'contexts': contexts, 'trend': trend, 'trendDetail': trend_detail,
            'rates': {'phaseSeconds': round(phase_s, 1), 'opportunitiesPer10': _rate(n, phase_s, per),
                      'demonstratedPer10': _rate(d, phase_s, per), 'notDemonstratedPer10': _rate(n-d, phase_s, per)},
            'coachMarkedShare': round(coach/n, 3) if n else None,
            'history': history, 'unreadable': [e['id'] for e in unreadable]}


def mastery(entries, block, *, squad_id=None, rules=None, data=None):
    """The squad's progression under its current settings block (read_block): {'squad', 'level', 'rules' {version,
    sha, label}, 'skills' {skill: detail}, 'separate' {'earlierEngine': [...], 'needsRules': [...]}, 'matches',
    'shortClips', 'note'}. Entries of an earlier engine (stale, another version) and entries without an outcome for
    the current rules are never read into a state: they are listed apart."""
    rules = rules or load_rules()
    data = data or PR.load()
    block = PR.clean_team(block if isinstance(block, dict) else {})
    reader = _Reader(block, data)
    current = reader.profile(block.get('format') or {'players': 11})
    usable, earlier = [], []
    for e in entries:
        if squad_id is not None and e['squad'] != squad_id:
            continue
        (earlier if (e.get('stale') or e['engine']['version'] != S.ENGINE_VERSION) else usable).append(e)
    skills = {s: _skill(usable, reader, s, current, rules) for s in S.SKILLS}
    needs = sorted({i for s in skills.values() for i in s['unreadable']})
    short = float(rules['rates']['shortRangeSeconds'])
    videos = {}
    for e in usable:
        if e['id'] not in needs:
            videos[e['video']] = videos.get(e['video'], 0.0)+_length(e)
    detail = {'squad': squad_id, 'level': current['level'],
              'rules': {'version': rules['version'], 'sha': rules['_sha'], 'label': rules['label']},
              'skills': skills,
              'separate': {'earlierEngine': [_brief(e) for e in earlier],
                           'needsRules': [_brief(e) for e in usable if e['id'] in needs]},
              'matches': len(videos), 'shortClips': sum(1 for s in videos.values() if s < short)}
    detail['note'] = _note(detail, rules)
    return detail


def _brief(e):
    return {'id': e['id'], 'video': e['video'], 'run': e['run'], 'team': e['team'], 'matchDate': e['matchDate'],
            'range': dict(e['range'])}


def _note(detail, rules, linked=True, squad_name=None):
    parts = [rules['label']]
    m, clips = detail['matches'], detail['shortClips']
    if m:
        what = f'{m} match' + ('es' if m != 1 else '')
        if clips:
            what += f', {clips} of them short clip' + ('s' if clips != 1 else '')
        parts.append(f'Read per moment where each skill was needed, never on match totals ({what}).')
    k = rules['evidence'].get('windowQualifyingMatches')
    if k and any(not h['inWindow'] for s in detail['skills'].values() if s['shown'] for h in s['history']):
        parts.append(f'States use the most recent {k} qualifying matches of each skill.')
    n_old = len(detail['separate']['earlierEngine'])
    if n_old:
        parts.append(f'{n_old} analys' + ('es' if n_old != 1 else 'is') + ' from an earlier engine listed apart.')
    n_rules = len(detail['separate']['needsRules'])
    if n_rules:
        parts.append(f'{n_rules} analys' + ('es' if n_rules != 1 else 'is') + ' to re-read with the current rules.')
    if any((s['coachMarkedShare'] or 0) > 0 for s in detail['skills'].values() if s['shown']):
        parts.append('Some moments use possession marked by the coach.')
    if not linked:
        who = squad_name or 'the squad'
        parts.append(f'Confirm which team is {who} in this run to add this match.')
    text = ' '.join(parts)
    while len(text) > NOTE_MAX and len(parts) > 1:
        parts.pop(1)
        text = ' '.join(parts)
    return text[:NOTE_MAX]


def section(entries, squad_ref, block, *, linked=True, rules=None, data=None):
    """The report's DEVELOPMENT PROGRESSION (schema Progression {squad, skills [SkillRow], note}) of a squad, merged
    by SERVER at serve time: the rows the level shows (profile skill names; a describe-only check has none), each
    with its state, judged opportunities, matches and trend. linked: this run is confirmed as the squad's (else the
    note asks to confirm it)."""
    rules = rules or load_rules()
    detail = mastery(entries, block, squad_id=squad_ref['id'], rules=rules, data=data)
    rows = [{'skill': s, 'name': v['name'], 'state': v['state'], 'opportunities': v['opportunities'],
             'matches': v['matches'], 'trend': v['trend']} for s, v in detail['skills'].items() if v['shown']]
    return {'squad': squad_ref['id'], 'skills': rows,
            'note': _note(detail, rules, linked=linked, squad_name=squad_ref.get('name'))}


__all__ = ['build_entry', 'rederive', 'needs_rederive', 'match_date_of', 'links', 'confirmed_team', 'read_squads',
           'squad', 'create_squad', 'update_squad', 'read_entries', 'upsert', 'remove', 'mark_stale',
           'set_match_date', 'record', 'mastery', 'section', 'outcomes', 'outcome_key', 'param_sets',
           'classify_params', 'classify_code_sha', 'wilson', 'read_block', 'load_rules', 'validate_rules',
           'counted', 'ProgressError', 'ProgressUnreadable', 'CLASSIFY_KEYS', 'SUMMARY']
