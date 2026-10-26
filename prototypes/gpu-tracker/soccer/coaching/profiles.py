"""The coaching profile (DESIGN v2 section 2): what a team's age group, playing level, match format and declared style
mean for the coaching report, read from profiles.json, and the coach's settings cleaned before they are stored.

Why a profile table and not wording levels. The committed engine judged every team against one professional match
and changed only its wording by level: the audit measured identical observation ids, counts and drills at all three
levels on all 8 team-runs. The brief asks for genuinely different reasoning for a U9 beginner, a U14 team and a
professional squad. So the level decides which checks run and which verdicts they may give (no "work on" at levels
1-2, no "try this" from level 4), how much evidence a criticism needs (8 opportunities and 3 distinct failing spells
at level 1, 15 and 2 at level 7), how players may be named, the vocabulary and the drills; the age decides safety
(heading, load, tackling), tone, the naming cap and the level 5/6 sub-bands; the format decides the role model and
the minimum number of players behind the ball; the declared style adjusts thresholds and, at levels 6-7 with no style,
turns criticisms into reviews. The measurements never change with any of this (invariant 2): resolve() only reads the
interpretation-tier settings plus the format, field and formation the profile shows.

Why every number is a PitchIQ proposal. The three research passes (122 sources after the merge, 6 of them not read)
found no federation source that gives a positional distance, count or time a report could use as a threshold
(research levels 5-7, finding 1). So every threshold in profiles.json sits under "origin": "PITCHIQ-PROPOSAL",
"validated": false, and is an initial value (DESIGN Appendix A) with its measured basis where one exists, e.g. the
5 m pressure distance (labeller notes: pressed median 2.6 m, 83% within 5 m). Federation content is text: paraphrased
items tagged FEDERATION that cite canonical source ids from sources.json (built by tools/build_coaching_sources.py);
the verbatim ones keep a pointer to the research item they copy, and validate_profiles() checks the copy, the tag and
that no unread source is cited. Heading caps quote federation guidance in text; the numbers PitchIQ applies are the
stricter of the U.S. Soccer and England caps (a PitchIQ rule, user decision 2026-10-21): no heading at U11 or younger.

Defaults (user decisions 2026-10-21). The match format defaults to the U.S. Soccer maximum for the age (4v4 without a
keeper to U8, 7v7 to U10, 9v9 to U12, 11v11 from U13) for auto and US. With England or Canada chosen it is that
federation's own format where the research has it (England 5v5 at U7-U8, Canada 5v5 at U8-U9, both 7v7 at U11 and 9v9
at U13), else PitchIQ's nearest one, which the notices never call the federation's (formatDefaults.basis; fix round 1,
SOURCES F1: England U7-U8 and Canada U8-U9 had been 4v4 without a keeper). The coach can choose any format. With
nothing set, a team is resolved as Adult, level 6, 11v11 (the footage PitchIQ has today) and the profile says so in a
notice.

clean_settings() follows team_shape's pattern: malformed parts are dropped (and listed when a problems list is
passed), never raised, and the result always validates against schema 'settings'. It keeps only the fields the
schema declares: a new settings field must be taught here.

Fix round 2 (2026-10-23, LEAD decisions (b) and (c)): the 'moments worth a look' numbers (momentsFromLevel 3,
momentsMinSpells 3, momentsMaxPerReport 1) are shared registry values every enabled check carries, and the level table's
judgeDepthOn11v11 (false at levels 1-3) resolves to the attacking-shape parameter judgeDepth False on 11v11 only;
assess.py's docstring has what they changed on Metrica game 1.

Fix round 3 (2026-10-24, LEAD decisions (2) and (3)): the shared registry values baseTestAlpha (.05, the one-sided
binomial test level), baseSuccessScale (1.0), momentsSettleS (2 s) and momentsContestS (1 s), the back-line values
lineMinPlayers (4), lineStdMaxM (4 m) and markedM (4 m), and a level-object baseSuccessScale (levels 1-4: .5 / .6 /
.7 / .8, copied into every detector's params; validate_profiles refuses one outside (0, 1], and a baseTestAlpha
outside (0, .5)); level 3's far-side check carries the level-4 rule's baseFailShare .004 (1 of 242 decided on the whole
of Metrica game 1, full tracking). assess.py's docstring has what they changed.

profiles.py imports only the standard library and schema.py (the package import itself still loads numpy
through episodes.py, as before). One resolve() takes about 1.5 ms (3570 profiles, each validated against the
schema, in 5.5 s), well inside the 300 ms the interpretation tier has for a level change (DESIGN 1.2).
"""
import copy
import hashlib
import json
import math
import re
from pathlib import Path

from soccer.coaching import schema as S

PROFILES_PATH = Path(__file__).with_name('profiles.json')
SOURCES_PATH = Path(__file__).with_name('sources.json')
SETTINGS_VERSION = 1
NAMING_ORDER = ('never', 'relative', 'role_words', 'evidence_only', 'yes')    # strictest first
READ_VALUES = ('full', 'partial', 'not_read')
FOCUS_STATUS = ('assessed', 'described', 'language_only', 'suppressor', 'unavailable')
FORMAT_KEYS = ('4', '5', '7', '9', '11')
SMALL_FORMATS = (4, 5, 7, 9)
DEFAULT_AGE_BY_FORMAT = {4: 'U8', 5: 'U8', 7: 'U10', 9: 'U12', 11: 'Adult'}   # nothing set: the format's typical age
FALLBACK_AGE = 'Adult'          # nothing set at all: the 11v11 broadcast footage PitchIQ has today
OUTFIELD_BLOCKS = {3: (1, 1, 1), 4: (1, 2, 1), 6: (3, 2, 1), 8: (3, 2, 3)}   # relative blocks without a formation
FIELD_LIMITS = {'length': (15.0, 130.0), 'width': (10.0, 100.0)}             # schema FieldSize
QUOTE = re.compile(r'"([^"]*)"')
PRINTABLE = re.compile(r'[ -~]*')
WORD = re.compile(r'[a-z][a-z \-]{0,40}')
CODE = re.compile(r'[a-z][a-z0-9_]{0,60}')
DRILL_ID = re.compile(r'[a-z][a-z0-9_]{2,60}')
FIELD_NAMES = {'ageGroup': 'age group', 'playingLevel': 'playing level', 'format': 'match format'}
FEDERATION_LABEL = {'auto': 'U.S. Soccer', 'US': 'U.S. Soccer', 'England': 'England', 'Canada': 'Canada Soccer'}
GUIDANCE_PARTS = ('objectives', 'principles', 'notExpected')
LEVEL_BANDS = {5: ('U15-U16', 'U17-U18'), 6: ('academy', 'adult')}
# Profile values a measurement gate already bounds (fix round 1, critic S-14 / STATS 8): beyond the bound an edit acts
# on nothing, so the table refuses it. Copies of tactics_geom's constants (this module imports no numpy; a test checks
# they agree): detection emits only advances of at least ADVANCED_M (15 m x L/105), stores the cover within
# COVER_ANGLE_DEG (60 deg) and drops samples running back at RECOVERING_MS (3 m/s) or more.
MEASUREMENT_BOUNDS = {'advancedM': ('min', 15.0), 'coverAngleDeg': ('max', 60.0), 'recoveringMS': ('max', 3.0)}

_cache = {}


class ProfileError(ValueError):
    """profiles.json or sources.json does not validate (the problems are in the message)."""


# ---------- files ----------
def read_json(path):
    return json.loads(Path(path).read_bytes().decode('utf-8'))


def canonical_sha(data):
    """sha256 hex of a JSON value's canonical form (sorted keys, no whitespace): the same on CRLF and LF checkouts."""
    return hashlib.sha256(json.dumps(data, sort_keys=True, separators=(',', ':'), ensure_ascii=True,
                                     allow_nan=False).encode('ascii')).hexdigest()


def _cached(name, path, loader):
    p = Path(path)
    st = p.stat()
    key = (str(p.resolve()), st.st_mtime_ns, st.st_size)
    hit = _cache.get(name)
    if hit is None or hit[0] != key:
        _cache[name] = (key, loader(p))
    return _cache[name][1]


def load_sources(path=None):
    """The sources registry {'doc', 'byId', 'alias'} (cached per file and modification time)."""
    def loader(p):
        doc = read_json(p)
        problems = validate_sources(doc)
        if problems:
            raise ProfileError('sources.json invalid: '+'; '.join(problems[:20]))
        by_id = {s['id']: s for s in doc['sources']}
        alias = {}
        for s in doc['sources']:
            for a in s['aliases']+[s['id']]:
                alias[_alias_key(a)] = s['id']
        return {'doc': doc, 'byId': by_id, 'alias': alias}
    return _cached('sources', path or SOURCES_PATH, loader)


def load(path=None, sources_path=None):
    """The validated profile table (cached per file and modification time) with '_sha' (first 16 hex of its
    canonical sha256). Raises ProfileError listing the problems: a broken table must not reach a report."""
    src = load_sources(sources_path)

    def loader(p):
        data = read_json(p)
        problems = validate_profiles(data, src['doc'])
        if problems:
            raise ProfileError('profiles.json invalid: '+'; '.join(problems[:20]))
        data['_sha'] = canonical_sha(data)[:16]
        return data
    return _cached('profiles', path or PROFILES_PATH, loader)


def _alias_key(source_id):
    return str(source_id).strip().upper().replace('_', '-')


def canonical_source(source_id, sources_path=None):
    """The canonical id of a source written with any research id or spelling ('US_GM_U78_PDF', 'USS-GM1516',
    'CA-GAMES'), or None when it is not in the registry."""
    return load_sources(sources_path)['alias'].get(_alias_key(source_id))


def source_refs(ids, sources_path=None):
    """Report SourceRef records {id, title, url, publisher} for source ids (canonicalised, de-duplicated, sorted);
    unknown ids are skipped."""
    src = load_sources(sources_path)
    out, seen = [], set()
    for i in ids:
        cid = src['alias'].get(_alias_key(i))
        if cid and cid not in seen:
            seen.add(cid)
            s = src['byId'][cid]
            out.append({'id': cid, 'title': s['title'][:300], 'url': s['url'], 'publisher': s['publisher']})
    return sorted(out, key=lambda r: r['id'])


# ---------- age, level, format helpers ----------
def _age_number(age):
    return 99 if age == 'Adult' else int(age[1:])


def age_info(age, data=None):
    """The age group's row: typicalLevel, register, naming cap, maxLoad, tackling, sprintRaces, reachSpeedCapMS,
    heading rule key."""
    data = data or load()
    return data['ages']['groups'][age]


def typical_level(age, data=None):
    return age_info(age, data)['typicalLevel']


def default_age(level, data=None):
    data = data or load()
    return data['levels'][str(level)]['defaultAge']


def default_format(age, federation='auto', data=None):
    """The default match format {'players', 'goalkeeper'} for an age group: U.S. Soccer formats for auto and US;
    England's and Canada's own where the research has them (England 5v5 at U7-U8, 7v7 to U11, 9v9 at U12-U13; Canada
    5v5 at U8-U9, 7v7 at U10-U11, 9v9 at U12-U13), else the nearest PitchIQ format (default_format_basis says why).
    None age -> 11v11. Fix round 1 (SOURCES F1 / critic S-13): England U7-U8 and Canada U8-U9 were 4v4 without a
    keeper and Canada U9 7v7, so an England U8 coach choosing England's own 5v5 was told it was above England's
    format."""
    data = data or load()
    if age is None:
        return dict(S.DEFAULT_FORMAT)
    fed = 'US' if federation in (None, 'auto') else federation
    return dict(data['formatDefaults']['byAge'][age][fed])


def default_format_basis(age, federation='auto', data=None):
    """Why default_format is not the federation's own format at this age (a short phrase: "the nearest PitchIQ format:
    Canada plays 3v3 at U6-U7 ..."), or None when it is (always None for auto and US)."""
    data = data or load()
    if age is None or federation in (None, 'auto', 'US'):
        return None
    return ((data['formatDefaults'].get('basis') or {}).get(federation) or {}).get(age)


def heading_rule(age, data=None):
    """The heading rule PitchIQ applies at this age: {key, heading 'none'|'limited', caps {sessionsPerWeek,
    minutesPerWeek, headersPerSession, headersPerWeek, contested, lightBall, unopposed} | None, text, sources}. The
    caps are the stricter of the U.S. Soccer and England figures; DRILLS filters any drill with heading content by
    them."""
    data = data or load()
    key = age_info(age, data)['heading']
    rule = copy.deepcopy(data['heading']['rules'][key])
    rule['key'] = key
    return rule


def reach_speed_cap(age, data=None):
    """The top speed (m/s) a suggested area may assume for a player of this age (DESIGN 5.1)."""
    return age_info(age, data)['reachSpeedCapMS']


def band_of(level, age):
    """The sub-band of levels 5 and 6 (DESIGN 2.2): 'U15-U16' / 'U17-U18' at level 5, 'academy' / 'adult' at level 6;
    None elsewhere."""
    n = _age_number(age)
    if level == 5:
        return 'U15-U16' if n <= 16 else 'U17-U18'
    if level == 6:
        return 'adult' if age == 'Adult' else 'academy'
    return None


def formation_outfield(name, players):
    """Outfield players a formation describes: small formats count the keeper first ('1-3-2-1' = 6 outfield) except
    the 4v4 / 5v5 diamond '1-2-1' (4 outfield); 11v11 names are outfield only ('4-3-3' = 10, '4-4-1' = 9)."""
    parts = [int(x) for x in name.split('-')]
    if players in (7, 9):
        return sum(parts)-1
    return sum(parts)


def blocks_of(fmt, formation=None):
    """The relative blocks (back, mid, front) of a small format from its formation, or the generic blocks for its
    outfield count."""
    players, out = int(fmt['players']), S.outfield(fmt)
    if formation and formation in S.FORMATIONS.get(players, ()) and formation_outfield(formation, players) == out:
        parts = [int(x) for x in formation.split('-')]
        blocks = parts[1:] if players in (7, 9) else parts
        if len(blocks) == 3:
            return tuple(blocks)
    return OUTFIELD_BLOCKS.get(out, (max(1, out//3), max(1, out-2*max(1, out//3)), max(1, out//3)))


def roles_for(fmt, formation=None):
    """The role names a coach may confirm for a format: team_shape's role names for 11v11; relative slots
    ('back-2/3', ...) from the formation's blocks plus GK when there is a keeper for smaller formats (DESIGN 2.5:
    never 11v11 role names on a small format)."""
    if int(fmt['players']) == 11:
        return list(S.TEMPLATE_ROLES)
    names = ['GK'] if fmt.get('goalkeeper') else []
    for block, n in zip(('back', 'mid', 'front'), blocks_of(fmt, formation)):
        names += [f'{block}-{k}/{n}' for k in range(1, n+1)]
    return names


# ---------- settings cleaning (DESIGN 2.1) ----------
def _note(problems, text):
    if problems is not None:
        problems.append(text)


def _is_num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _clean_age(v):
    if not isinstance(v, str):
        return None
    t = v.strip()
    if t.lower() == 'adult':
        return 'Adult'
    t = t.upper()
    return t if t in S.AGE_GROUPS else None


def _clean_level(v):
    if isinstance(v, bool):
        return None
    if isinstance(v, str) and v.strip().isdigit():
        v = int(v.strip())
    if isinstance(v, float) and v.is_integer():
        v = int(v)
    return v if isinstance(v, int) and 1 <= v <= 7 else None


def _clean_format(v):
    if not isinstance(v, dict):
        return None
    players = v.get('players')
    if isinstance(players, float) and players.is_integer():
        players = int(players)
    if isinstance(players, bool) or players not in S.FORMAT_PLAYERS:
        return None
    gk = v.get('goalkeeper')
    if players in (7, 9, 11):
        gk = True
    elif not isinstance(gk, bool):
        gk = players == 5           # 4v4 has no keeper by default (U.S. 4v4); the 5v5 alias has one
    return {'players': players, 'goalkeeper': gk}


def _clean_field(v, path, problems):
    if v is None:
        return None
    if not isinstance(v, dict) or not _is_num(v.get('length')) or not _is_num(v.get('width')):
        _note(problems, f'{path}.field: needs numeric length and width in metres, dropped')
        return None
    length, width = float(v['length']), float(v['width'])
    if width > length:
        length, width = width, length
        _note(problems, f'{path}.field: width larger than length, swapped')
    lo, hi = FIELD_LIMITS['length']
    wlo, whi = FIELD_LIMITS['width']
    if not (lo <= length <= hi and wlo <= width <= whi):
        _note(problems, f'{path}.field: {length:g} x {width:g} m is outside {lo:g}-{hi:g} x {wlo:g}-{whi:g} m, dropped')
        return None
    return {'length': round(length, 2), 'width': round(width, 2)}


def _formation_ok(name, fmt):
    players = fmt['players']
    if not isinstance(name, str) or name not in S.FORMATIONS[players]:
        return False
    if players == 11:
        return True
    return formation_outfield(name, players) == S.outfield(fmt)


def _clean_formation(v, fmt, path, problems):
    out = {'intended': None, 'periods': []}
    if v is None:
        return out
    if not isinstance(v, dict):
        _note(problems, f'{path}.formation: not an object, dropped')
        return out
    intended = v.get('intended')
    if intended is not None:
        if _formation_ok(intended, fmt):
            out['intended'] = intended
        else:
            _note(problems, f'{path}.formation.intended: {intended!r} does not fit {S.format_name(fmt)}, dropped')
    periods = v.get('periods') or []
    if not isinstance(periods, list):
        periods = []
    kept = []
    for k, p in enumerate(periods):
        if (isinstance(p, dict) and _is_num(p.get('t0')) and _is_num(p.get('t1')) and 0 <= p['t0'] <= p['t1']
                and _formation_ok(p.get('formation'), fmt)):
            kept.append({'t0': float(p['t0']), 't1': float(p['t1']), 'formation': p['formation']})
        else:
            _note(problems, f'{path}.formation.periods[{k}]: malformed or not for {S.format_name(fmt)}, dropped')
    kept.sort(key=lambda p: (p['t0'], p['t1']))
    for p in kept:
        if out['periods'] and p['t0'] < out['periods'][-1]['t1']:
            _note(problems, f'{path}.formation.periods: a period overlapping the previous one was dropped')
            continue
        out['periods'].append(p)
    return out


_ISO = re.compile(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}(?::\d{2}(?:\.\d{1,6})?)?(?:Z|[+-]\d{2}:?\d{2})?')
_PID = re.compile(S.PLAYER_ID)
_SQ = re.compile(r'sq-[0-9a-f]{12}')
_RUN = re.compile(r'[A-Za-z0-9_.\-]{1,120}')


def _clean_roles(v, fmt, path, problems, now):
    out = {}
    if not isinstance(v, dict):
        if v is not None:
            _note(problems, f'{path}.roles: not an object, dropped')
        return out
    for pid in sorted(v):
        spans = v[pid]
        if not isinstance(pid, str) or not _PID.fullmatch(pid):
            _note(problems, f'{path}.roles: {pid!r} is not an internal player id, dropped')
            continue
        if isinstance(spans, dict):
            spans = [spans]
        if not isinstance(spans, list):
            _note(problems, f'{path}.roles.{pid}: not a list of spans, dropped')
            continue
        kept = []
        for k, s in enumerate(spans):
            if not isinstance(s, dict):
                _note(problems, f'{path}.roles.{pid}[{k}]: not an object, dropped')
                continue
            role, a, b, when = s.get('role'), s.get('from'), s.get('to'), s.get('set', now)
            if not (isinstance(role, str) and (re.fullmatch(S.RELATIVE_ROLE, role) or
                                               (role in S.TEMPLATE_ROLES and (fmt['players'] == 11 or role == 'GK')))):
                _note(problems, f'{path}.roles.{pid}[{k}]: role {role!r} is not offered for {S.format_name(fmt)}, '
                                'dropped')
                continue
            if (a is not None and not (_is_num(a) and a >= 0)) or (b is not None and not (_is_num(b) and b >= 0)) \
                    or (a is not None and b is not None and b < a):
                _note(problems, f'{path}.roles.{pid}[{k}]: from/to malformed, dropped')
                continue
            if not (isinstance(when, str) and _ISO.fullmatch(when)):
                _note(problems, f'{path}.roles.{pid}[{k}]: no valid "set" time, dropped')
                continue
            kept.append({'role': role, 'from': None if a is None else float(a), 'to': None if b is None else float(b),
                         'set': when})
        kept.sort(key=lambda s: (-1.0 if s['from'] is None else s['from']))
        clean, end = [], None
        for s in kept:
            lo = -1.0 if s['from'] is None else s['from']
            if end is not None and lo < end:
                _note(problems, f'{path}.roles.{pid}: a span overlapping the previous one was dropped')
                continue
            clean.append(s)
            end = 1e7 if s['to'] is None else s['to']
        if clean:
            out[pid] = clean
    return out


def _clean_squad(v, path, problems):
    if v is None:
        return None
    if (isinstance(v, dict) and isinstance(v.get('id'), str) and _SQ.fullmatch(v['id'])
            and isinstance(v.get('name'), str) and 1 <= len(v['name'].strip()) <= 60
            and PRINTABLE.fullmatch(v['name'].strip())):
        return {'id': v['id'], 'name': v['name'].strip()}
    _note(problems, f'{path}.squad: needs an sq- id and a printable name of at most 60 characters, dropped')
    return None


def _clean_links(v, path, problems):
    out, seen = [], set()
    for k, link in enumerate(v if isinstance(v, list) else []):
        if (isinstance(link, dict) and isinstance(link.get('run'), str) and _RUN.fullmatch(link['run'])
                and link.get('team') in S.TEAMS and isinstance(link.get('confirmed'), bool)):
            key = (link['run'], link['team'])
            if key not in seen:
                seen.add(key)
                out.append({'run': link['run'], 'team': link['team'], 'confirmed': link['confirmed']})
        else:
            _note(problems, f'{path}.runLinks[{k}]: malformed, dropped')
    return out


def clean_team(raw, *, now=None, problems=None, defaulted=None, path='team'):
    """One team's settings block, cleaned and completed (schema 'team_settings'), or None for a missing or
    non-object block. Missing age group / level / format are filled: level from the age (its typical level), age from
    the level (its default age) or the format, format from the age and federation (U.S. formats by default).
    defaulted (a list) receives the names of the fields that were filled; problems (a list) what was dropped. now (an
    ISO time) stamps role spans without a 'set' time; without it such spans are dropped."""
    if raw is None:
        return None
    if not isinstance(raw, dict):
        _note(problems, f'{path}: not an object, dropped')
        return None
    data = load()
    age = _clean_age(raw.get('ageGroup'))
    if raw.get('ageGroup') is not None and age is None:
        _note(problems, f'{path}.ageGroup: {raw.get("ageGroup")!r} unknown, defaulted')
    level = _clean_level(raw.get('playingLevel'))
    if raw.get('playingLevel') is not None and level is None:
        _note(problems, f'{path}.playingLevel: {raw.get("playingLevel")!r} not 1-7, defaulted')
    federation = raw.get('federation')
    if federation not in S.FEDERATIONS:
        if federation is not None:
            _note(problems, f'{path}.federation: {federation!r} unknown, auto used')
        federation = 'auto'
    fmt = _clean_format(raw.get('format'))
    if raw.get('format') is not None and fmt is None:
        _note(problems, f'{path}.format: {raw.get("format")!r} is not 4, 5, 7, 9 or 11 players, defaulted')
    filled = []
    if age is None:
        filled.append('ageGroup')
        age = (default_age(level, data) if level is not None
               else DEFAULT_AGE_BY_FORMAT[fmt['players']] if fmt is not None else FALLBACK_AGE)
    if level is None:
        filled.append('playingLevel')
        level = typical_level(age, data)
    if fmt is None:
        filled.append('format')
        fmt = default_format(age, federation, data)
    if defaulted is not None:
        defaulted.extend(filled)
    style = raw.get('style')
    if style not in S.STYLES:
        if style is not None:
            _note(problems, f'{path}.style: {style!r} unknown, unknown used')
        style = 'unknown'
    return {'squad': _clean_squad(raw.get('squad'), path, problems), 'ageGroup': age, 'playingLevel': level,
            'format': fmt, 'field': _clean_field(raw.get('field'), path, problems), 'federation': federation,
            'formation': _clean_formation(raw.get('formation'), fmt, path, problems), 'style': style,
            'roles': _clean_roles(raw.get('roles'), fmt, path, problems, now),
            'runLinks': _clean_links(raw.get('runLinks'), path, problems)}


def clean_settings(raw, *, now=None, problems=None):
    """coaching_settings.json content cleaned (schema 'settings'): {'version': 1, 'teams': {'A', 'B'}, 'updated'}.
    Malformed parts are dropped, never raised (problems, a list, receives what was dropped); a team block is completed
    with defaults (clean_team)."""
    raw = raw if isinstance(raw, dict) else {}
    if raw and not isinstance(raw.get('teams'), dict):
        _note(problems, 'teams: not an object, dropped')
    teams = raw.get('teams') if isinstance(raw.get('teams'), dict) else {}
    updated = raw.get('updated')
    if updated is not None and not (isinstance(updated, str) and _ISO.fullmatch(updated)):
        _note(problems, 'updated: not an ISO time, dropped')
        updated = None
    return {'version': SETTINGS_VERSION,
            'teams': {t: clean_team(teams.get(t), now=now, problems=problems, path=f'teams.{t}') for t in S.TEAMS},
            'updated': updated}


def default_team_settings(age_group=None, level=None, fmt=None, federation='auto'):
    """A complete team block with defaults for whatever is not given (the Coaching setup form's starting values)."""
    raw = {'ageGroup': age_group, 'playingLevel': level, 'format': fmt, 'federation': federation}
    return clean_team({k: v for k, v in raw.items() if v is not None})


# ---------- resolve (DESIGN 2.2-2.6) ----------
def _guidance(items, band):
    out = []
    for g in items:
        if band is not None and g.get('bands') and band not in g['bands']:
            continue
        out.append({'text': g['text'], 'tag': g['tag'], 'sources': list(g['sources'])})
    return out


def _stricter(a, b):
    return a if NAMING_ORDER.index(a) <= NAMING_ORDER.index(b) else b


def _scale_key(name):
    return (name[:-1] if name.endswith('M') else name)+'Scale'


def _detector(data, level, d, age, band, fmt, style):
    lv = data['levels'][str(level)]
    spec = lv['detectors'][d]
    meta = data['detectors'][d]
    verdicts = list(spec['verdicts'])
    if band is not None and spec.get('verdictsByBand'):
        verdicts = list(spec['verdictsByBand'][band])
    effects = []
    limits = data['styles']['unknownLimits'].get(str(level)) if style == 'unknown' else None
    if limits is not None and 'work_on' in verdicts and d not in limits['workOnDetectors']:
        verdicts = [v for v in verdicts if v != 'work_on']
        if 'review' not in verdicts:
            verdicts.insert(0, 'review')
        effects.append('unknown_style_review_only')
    enabled = bool(verdicts)
    if not enabled:
        return {'enabled': False, 'verdicts': [], 'framing': None, 'params': dict(spec['params']),
                'origin': S.ORIGIN_P, 'validated': False}
    reg = data['parameters']['values']
    params = {}
    for name in list(meta['params'])+list(data['parameters']['shared']):
        p = reg[name]
        params[name] = p['value']
        if p['scale'] in ('length', 'width'):
            params[_scale_key(name)] = p['scale']
    if 'spellMinS' in meta:
        params['spellMinS'] = meta['spellMinS']
    if lv.get('baseSuccessScale') is not None:
        # the praise test's level scale (fix round 3, LEAD decision (3)): one value per level, levels 1-4
        params['baseSuccessScale'] = lv['baseSuccessScale']
    params.update(spec['params'])
    params['needsMetres'] = bool(meta['needsMetres'])
    if meta['zones']:
        params['reachHorizonS'] = reg['reachHorizonS']['value']
        params['reachSpeedCapMS'] = age_info(age, data)['reachSpeedCapMS']
        params['zonePoint'] = level >= 4           # DESIGN 5.3: a point-like zone only from level 4
    order = lv['priorities']['order']
    params['relevance'] = lv['priorities']['relevance'].get(d, 0.0)
    params['objectiveRank'] = order.index(meta['skill'])+1 if meta['skill'] in order else None
    players = int(fmt['players'])
    if d == 'wide_defender_advance':
        params['advancedRule'] = 'halfway' if level == 2 and players in (7, 9) else 'cb_line'
    if d == 'attacking_shape':
        # depth against the space to the opponents' last line (critic B-7); 11v11: unclear where it cannot be
        # measured, small formats: the field share there
        params['depthRule'] = 'available' if players == 11 else 'available_or_field'
        # 11v11 depth descriptive only at levels 1-3 (fix round 2, LEAD decision (c), the coach's recommendation):
        # assess.classify then never fails a sample on depth alone; small formats keep judging it
        if params.pop('judgeDepthOn11v11', True) is False and players == 11:
            params['judgeDepth'] = False
    if d == 'behind_ball':
        rule = lv['behindBall'][str(players)]
        params['behindRule'] = rule['rule']
        params['minBehind'] = rule.get('minBehind')
        params['minMargin'] = rule.get('minMargin')
    eff = data['styles'][style]['effects'].get(d)
    if eff:
        for k, v in eff.get('set', {}).items():
            params[k] = v
        for k, v in eff.get('add', {}).items():
            if params.get(k) is not None:
                params[k] = params[k]+v
        for k, v in eff.get('multiply', {}).items():
            if params.get(k) is not None:
                params[k] = round(params[k]*v, 3)
        effects.append(eff['code'])
    params['styleEffect'] = '+'.join(effects) if effects else None
    return {'enabled': True, 'verdicts': verdicts, 'framing': spec['framing'], 'params': params,
            'origin': S.ORIGIN_P, 'validated': False}


def _safety(data, age, level):
    info = age_info(age, data)
    rule = heading_rule(age, data)
    notes = [rule['text']]
    sources = list(rule['sources'])
    for n in data['safetyNotes']['notes']:
        if age in n['ages']:
            notes.append(n['text'])
            sources += n['sources']
    return ({'heading': rule['heading'], 'maxLoad': info['maxLoad'], 'tackling': info['tackling'],
             'sprintRaces': info['sprintRaces'], 'notes': notes}, sources)


def _notices(data, team_settings, level, age, band, filled):
    out = [{'code': 'sources_disclaimer', 'text': data['notices']['sources_disclaimer']}]
    sources = []
    t = data['notices']
    fmt, style, fed = team_settings['format'], team_settings['style'], team_settings['federation']
    fname = S.format_name(fmt)
    if filled:
        values = {'ageGroup': team_settings['ageGroup'], 'playingLevel': f'level {level}', 'format': fname}
        out.append({'code': 'defaults_used',
                    'text': t['defaults_used'].format(fields=', '.join(FIELD_NAMES[f] for f in filled),
                                                      values=', '.join(values[f] for f in filled))})
    typ = typical_level(age, data)
    if level-typ > 2:
        summ = data['levels'][str(typ)]['ageSummary']
        sources += summ['sources']
        out.append({'code': 'level_above_age',
                    'text': t['level_above_age'].format(level=level, age=age, summary=summ['text']+' ('
                                                        + ', '.join(summ['sources'])+').')})
    if age_info(age, data)['register'] == 'adult' and level <= 3:
        sources += ['CS-W2WC', 'CS-GCE']
        out.append({'code': 'adult_low_level', 'text': t['adult_low_level'].format(level=level)})
    if band is not None:
        code = 'band_'+band.lower().replace('-', '_')
        out.append({'code': code, 'text': t[code]})
        sources += {'band_u15_u16': ['USS-AGG-U1516'], 'band_u17_u18': ['USS-AGG-U17'],
                    'band_academy': ['UEFA-YB']}.get(code, [])
    if style == 'unknown' and str(level) in data['styles']['unknownLimits']:
        code = f'style_unknown_{level}'
        out.append({'code': code, 'text': t[code]})
    players = int(fmt['players'])
    if players in SMALL_FORMATS:
        if team_settings['field'] is None:
            out.append({'code': 'scale_unknown', 'text': t['scale_unknown'].format(format=fname)})
        out.append({'code': 'small_format_registration',
                    'text': t['small_format_registration'].format(format=fname)})
    if players in (4, 5):
        out.append({'code': 'no_back_block', 'text': t['no_back_block'].format(format=fname)})
    dflt = default_format(age, fed, data)
    if players > dflt['players']:
        # A row that is not the federation's own format is never called the federation's (SOURCES F1).
        basis = default_format_basis(age, fed, data)
        text = (t['format_above_nearest'].format(federation=FEDERATION_LABEL[fed], age=age, default=S.format_name(dflt),
                                                 basis=basis, format=fname) if basis and 'format_above_nearest' in t
                else t['format_above_default'].format(federation=FEDERATION_LABEL[fed], age=age,
                                                      default=S.format_name(dflt), format=fname))
        out.append({'code': 'format_above_default', 'text': text})
    return out, sources


def resolve(block=None, team='A', *, data=None):
    """The CoachingProfile (schema 'profile') of one team from its settings block (partial or None: defaults are
    filled and named in a 'defaults_used' notice). Only age group, playing level, style and federation change what
    it decides; format, field and formation are shown and used for format rules (they are measurement- and
    role-tier inputs, so a change there rebuilds those tiers first)."""
    data = data or load()
    filled = []
    ts = clean_team(block if isinstance(block, dict) else {}, defaulted=filled)
    age, level, fmt, style = ts['ageGroup'], ts['playingLevel'], ts['format'], ts['style']
    lv = data['levels'][str(level)]
    band = band_of(level, age)
    info = age_info(age, data)
    guidance = {part: _guidance(lv[part], band) for part in GUIDANCE_PARTS}
    detectors = {d: _detector(data, level, d, age, band, fmt, style) for d in S.DETECTORS}
    ev = lv['evidence']
    evidence = {'minOpportunities': ev['minOpportunities'], 'minFailingSpells': ev['minFailingSpells'],
                'minFailureShare': ev['minFailureShare'], 'minPhaseSeconds': ev['minPhaseSeconds'],
                'maxWorkItems': ev['maxWorkItems'], 'strengthSpells': ev['strengthSpells'],
                'strengthShare': ev['strengthShare'], 'naming': _stricter(ev['naming'], info['naming']),
                'origin': S.ORIGIN_P, 'validated': False}
    if ev['minFailureShare'] is None:
        for d in detectors.values():
            if d['enabled']:
                d['params']['minFailureShare'] = ev['minFailureShareStyleBase']
    reg = lv['register']
    tone = reg['tone']
    if info['register'] == 'adult':
        tone = tone.replace("'Try this'", "'Next step'")+'; adult register, no child phrasing'
    elif info['register'] == 'child':
        tone += '; simple words for children'
    else:
        tone += '; plain words for teenagers'
    register = {'tone': tone[:120], 'numbers': reg['numbers'], 'maxChallenges': reg['maxChallenges'],
                'concepts': list(reg['concepts']), 'banned': list(reg['banned'])}
    safety, safety_sources = _safety(data, age, level)
    notices, notice_sources = _notices(data, ts, level, age, band, filled)
    formation = ts['formation']['intended']
    fsources = []
    for f in data['formats'][str(fmt['players'])]['formations']:
        if f['name'] == formation:
            fsources = f['sources']
    sources = set(safety_sources) | set(notice_sources) | set(fsources)
    for part in guidance.values():
        for g in part:
            sources.update(g['sources'])
    return {'version': data['version'], 'sha': data['_sha'], 'team': team, 'ageGroup': age, 'level': level,
            'levelName': lv['name'], 'band': band,
            'format': {'players': fmt['players'], 'goalkeeper': fmt['goalkeeper'], 'outfield': S.outfield(fmt),
                       'name': S.format_name(fmt)},
            'field': ts['field'], 'federation': ts['federation'], 'style': style, 'formation': formation,
            'objectives': guidance['objectives'], 'principles': guidance['principles'],
            'notExpected': guidance['notExpected'], 'detectors': detectors, 'evidence': evidence,
            'register': register, 'skills': dict(lv['skills']), 'drillPool': list(lv['drillPool']),
            'safety': safety, 'notices': notices, 'sources': sorted(sources)}


# ---------- what the report and the page show ----------
def focus_for(level, data=None):
    """The brief's focus areas for a level and what PitchIQ does with each: [{spec, status, detectors, note,
    unavailable}] (status assessed | described | language_only | suppressor | unavailable)."""
    data = data or load()
    return copy.deepcopy(data['levels'][str(level)]['focus'])


def unavailable_for(level, phase_available=None, data=None):
    """The report's MATCH OVERVIEW 'unavailable' records (schema Unavailable {category, reason, unlock}) for the focus
    areas of a level that PitchIQ cannot measure (DESIGN 4.7), de-duplicated, plus 'phase_amateur' (every check that
    needs to know who had the ball) unless phase_available is True (the run's possession estimate, or coach-marked
    spans, name a team)."""
    data = data or load()
    cats = data['unavailable']['categories']
    codes = []
    for f in data['levels'][str(level)]['focus']:
        if f['unavailable'] and f['unavailable'] not in codes:
            codes.append(f['unavailable'])
    if phase_available is not True and 'phase_amateur' not in codes:
        codes.append('phase_amateur')
    return [{'category': c, 'reason': cats[c]['reason'], 'unlock': cats[c]['unlock']} for c in codes]


def catalog(data=None, sources_path=None):
    """The static description the setup form and the profile panel need (GET /api/coaching/profiles): levels with
    their guidance, focus, skills and verdicts; age groups with their heading rule; formats with goalkeeper defaults,
    formations (tag and sources) and the roles a coach may confirm; format defaults by age and federation; styles;
    unavailable categories; and every source with its title, link and read status."""
    data = data or load()
    src = load_sources(sources_path)
    levels = []
    for lvl in range(1, 8):
        lv = data['levels'][str(lvl)]
        dets = {}
        for d in S.DETECTORS:
            spec = lv['detectors'][d]
            dets[d] = {'verdicts': spec['verdicts'], 'framing': spec['framing']}
            if spec.get('verdictsByBand'):
                dets[d]['verdictsByBand'] = spec['verdictsByBand']
        levels.append({'level': lvl, 'name': lv['name'], 'summary': lv['summary'], 'typicalAges': lv['typicalAges'],
                       'defaultAge': lv['defaultAge'], 'bands': list(LEVEL_BANDS.get(lvl, ())),
                       'objectives': [_strip_g(g) for g in lv['objectives']],
                       'principles': [_strip_g(g) for g in lv['principles']],
                       'notExpected': [_strip_g(g) for g in lv['notExpected']],
                       'focus': focus_for(lvl, data), 'skills': lv['skills'], 'detectors': dets,
                       'evidence': {k: v for k, v in lv['evidence'].items() if k != 'minFailureShareStyleBase'},
                       'register': lv['register'], 'drillPool': lv['drillPool']})
    ages = [{'id': a, 'typicalLevel': typical_level(a, data), 'heading': heading_rule(a, data),
             'maxLoad': age_info(a, data)['maxLoad']} for a in S.AGE_GROUPS]
    formats = []
    for key in FORMAT_KEYS:
        f = data['formats'][key]
        players = int(key)
        gk = f['goalkeeperDefault']
        fmt = {'players': players, 'goalkeeper': gk}
        formats.append({'players': players, 'name': f['name'], 'goalkeeperDefault': gk,
                        'goalkeeperChoice': f['goalkeeperChoice'], 'outfield': S.outfield(fmt),
                        'roleModel': f['roleModel'], 'note': f['note'], 'formations': f['formations'],
                        'roles': {fm['name']: roles_for(fmt, fm['name']) for fm in f['formations']},
                        'aliasOf': int(f['aliasOf']) if f.get('aliasOf') else None})
    styles = [{'id': s, 'name': data['styles'][s]['name'], 'description': data['styles'][s]['description']}
              for s in S.STYLES]
    defaults = {a: {'auto': default_format(a, 'auto', data), 'US': default_format(a, 'US', data),
                    'England': default_format(a, 'England', data), 'Canada': default_format(a, 'Canada', data)}
                for a in S.AGE_GROUPS}
    # Whose format each default is, for the setup's note (S-13): the federation's own unless a basis says otherwise.
    basis = {a: {f: default_format_basis(a, f, data) for f in ('England', 'Canada') if default_format_basis(a, f, data)}
             for a in S.AGE_GROUPS}
    sources = [{'id': s['id'], 'title': s['title'], 'url': s['url'], 'publisher': s['publisher'], 'read': s['read']}
               for s in src['doc']['sources']]
    return {'version': data['version'], 'sha': data['_sha'], 'disclaimer': data['disclaimer'], 'tags': data['tags'],
            'levels': levels, 'ageGroups': ages, 'formats': formats, 'formatDefaults': defaults,
            'formatDefaultBasis': {a: b for a, b in basis.items() if b}, 'styles': styles,
            'federations': list(S.FEDERATIONS),
            'unavailable': [dict(v, code=k) for k, v in data['unavailable']['categories'].items()],
            'sources': sources}


def _strip_g(g):
    out = {'text': g['text'], 'tag': g['tag'], 'sources': list(g['sources'])}
    if g.get('bands'):
        out['bands'] = list(g['bands'])
    return out


# ---------- validation ----------
def validate_sources(doc):
    """Problems of a sources.json document (shape only; tools/build_coaching_sources.py --check compares it with a
    fresh build)."""
    out = []
    if not isinstance(doc, dict) or doc.get('kind') != 'coaching-sources' or not isinstance(doc.get('sources'), list):
        return ['not a coaching-sources document']
    seen, aliases = set(), {}
    for k, s in enumerate(doc['sources']):
        path = f'sources[{k}]'
        if not isinstance(s, dict):
            out.append(f'{path}: not an object')
            continue
        sid = s.get('id')
        if not isinstance(sid, str) or not re.fullmatch(r'[A-Z][A-Z0-9]*(?:-[A-Z0-9]+)*', sid):
            out.append(f'{path}.id: {sid!r} is not a canonical id')
        if sid in seen:
            out.append(f'{path}.id: {sid} repeated')
        seen.add(sid)
        if s.get('read') not in READ_VALUES:
            out.append(f'{path}.read: {s.get("read")!r}')
        if not isinstance(s.get('url'), str) or not re.fullmatch(r'https?://[!-~]{3,500}', s['url']):
            out.append(f'{path}.url: not a link')
        for field in ('title', 'publisher'):
            if not isinstance(s.get(field), str) or not s[field] or not PRINTABLE.fullmatch(s[field]):
                out.append(f'{path}.{field}: not printable ASCII')
        for a in s.get('aliases') or []:
            key = _alias_key(a)
            if aliases.get(key, sid) != sid:
                out.append(f'{path}.aliases: {a} names two sources')
            aliases[key] = sid
    for s in doc['sources']:
        if isinstance(s, dict) and aliases.get(_alias_key(s.get('id', '')), s.get('id')) != s.get('id'):
            out.append(f'{s.get("id")}: also an alias of another source')
    return out


def _text_ok(text, n, path, out):
    if not isinstance(text, str) or not 1 <= len(text) <= n or not PRINTABLE.fullmatch(text):
        out.append(f'{path}: text must be 1-{n} printable ASCII characters')
        return False
    return True


def _research_check(g, path, out, research, alias):
    item = _find_research(research, g['research'])
    if item is None:
        out.append(f'{path}.research: {g["research"]!r} not found in the research copies')
        return
    if item.get('tag') != g.get('tag'):
        out.append(f'{path}.tag: the research item is {item.get("tag")}')
    allowed = {alias.get(_alias_key(i)) for i in item.get('sources') or []}
    for sid in g.get('sources') or []:
        if sid not in allowed:
            out.append(f'{path}.sources: {sid} is not among the research item\'s sources')
    if not g.get('edited') and g.get('text') != item.get('text'):
        out.append(f'{path}.text: differs from the research item (mark "edited": true for a shortened paraphrase)')


def _find_research(research, pointer):
    if not isinstance(pointer, str) or ':' not in pointer:
        return None
    name, path = pointer.split(':', 1)
    for pname, doc, *_ in research:
        if pname != name:
            continue
        node = doc
        for part in re.findall(r'[^.\[\]]+|\[\d+\]', path):
            if part.startswith('['):
                idx = int(part[1:-1])
                if not isinstance(node, list) or idx >= len(node):
                    return None
                node = node[idx]
            else:
                if not isinstance(node, dict) or part not in node:
                    return None
                node = node[part]
        return node if isinstance(node, dict) and isinstance(node.get('text'), str) else None
    return None


def _numbers_tagged(node, path, origin, validated, out):
    """Invariant 5: every number sits under an object that declares origin PITCHIQ-PROPOSAL and validated false."""
    if isinstance(node, dict):
        if 'origin' in node:
            origin, validated = node['origin'], node.get('validated')
        for k, v in node.items():
            if k == 'version' and path == '':
                continue
            _numbers_tagged(v, f'{path}.{k}' if path else k, origin, validated, out)
    elif isinstance(node, list):
        for i, v in enumerate(node):
            _numbers_tagged(v, f'{path}[{i}]', origin, validated, out)
    elif isinstance(node, (int, float)) and not isinstance(node, bool):
        if origin != S.ORIGIN_P or validated is not False:
            out.append(f'{path}: a number outside a PITCHIQ-PROPOSAL / validated false section')
        if isinstance(node, float) and not math.isfinite(node):
            out.append(f'{path}: not finite')


def validate_profiles(data, sources_doc, research=None):
    """Every problem of a profiles.json document against the sources registry (and, when research is given as
    [(pass, document, ...)], against the research copies its verbatim items point to). [] when valid."""
    out = []
    if not isinstance(data, dict) or data.get('kind') != 'coaching-profiles' or not isinstance(data.get('version'),
                                                                                            int):
        return ['not a coaching-profiles document']
    srcs = {s['id']: s for s in sources_doc.get('sources', [])} if isinstance(sources_doc, dict) else {}
    alias = {}
    for s in srcs.values():
        for a in s.get('aliases', [])+[s['id']]:
            alias[_alias_key(a)] = s['id']
    if data.get('disclaimer') != sources_doc.get('disclaimer'):
        out.append('disclaimer: differs from sources.json')
    _numbers_tagged(data, '', None, None, out)

    def guide(g, path, bands=(), need_research=True):
        _check_guidance_core(g, path, out, srcs, bands)
        if need_research and isinstance(g, dict) and g.get('tag') == 'FEDERATION' and not g.get('research'):
            out.append(f'{path}: a FEDERATION item names the research item it paraphrases')
        if research is not None and isinstance(g, dict) and g.get('research'):
            _research_check(g, path, out, research, alias)

    reg = data.get('parameters', {}).get('values', {})
    for name, p in reg.items():
        if not _is_num(p.get('value')) or p.get('scale') not in ('body', 'length', 'width', 'length_share', 'angle',
                                                                 'share', 'count', 'seconds'):
            out.append(f'parameters.{name}: needs a finite value and a known scale')
    for name in data.get('parameters', {}).get('shared', []):
        if name not in reg:
            out.append(f'parameters.shared: {name} not in the registry')
    a_ = (reg.get('baseTestAlpha') or {}).get('value')
    if a_ is not None and not (_is_num(a_) and 0.0 < a_ < .5):
        out.append('parameters.baseTestAlpha: a one-sided test level in (0, .5)')
    bounded = [(f'parameters.{n}', n, (p or {}).get('value')) for n, p in reg.items()]
    for key, lv in (data.get('levels') or {}).items():
        for d, spec in ((lv or {}).get('detectors') or {}).items():
            for n, v in ((spec or {}).get('params') or {}).items():
                bounded.append((f'levels.{key}.detectors.{d}.params.{n}', n, v))
    for path, n, v in bounded:
        if n in MEASUREMENT_BOUNDS and _is_num(v):
            side, bound = MEASUREMENT_BOUNDS[n]
            if (side == 'min' and v < bound-1e-9) or (side == 'max' and v > bound+1e-9):
                out.append(f'{path}: {v} acts on nothing beyond the measurement bound ({side} {bound})')
    dets = data.get('detectors', {})
    if set(dets) != set(S.DETECTORS):
        out.append('detectors: must list the six detectors')
    for d, meta in dets.items():
        for name in meta.get('params', []):
            if name not in reg:
                out.append(f'detectors.{d}.params: {name} not in the registry')
        if meta.get('skill') not in S.SKILLS or S.SKILLS[meta['skill']] != d:
            out.append(f'detectors.{d}.skill: does not match schema SKILLS')
        if meta.get('code') != S.DETECTOR_CODE.get(d):
            out.append(f'detectors.{d}.code: not {S.DETECTOR_CODE.get(d)}')
    levels = data.get('levels', {})
    if sorted(levels) != [str(k) for k in range(1, 8)]:
        out.append('levels: must be "1".."7"')
    cats = data.get('unavailable', {}).get('categories', {})
    for key, lv in levels.items():
        level = int(key)
        path = f'levels.{key}'
        bands = LEVEL_BANDS.get(level, ())
        if lv.get('name') != S.LEVEL_NAMES[level]:
            out.append(f'{path}.name: not {S.LEVEL_NAMES[level]}')
        for a in lv.get('typicalAges', [])+[lv.get('defaultAge')]:
            if a not in S.AGE_GROUPS:
                out.append(f'{path}: age {a!r} unknown')
        _text_ok(lv.get('summary'), 200, f'{path}.summary', out)
        # the age summary (shown when a team is placed far above its age's level) condenses the level's objectives:
        # it may cite only sources its objectives cite
        summary = lv.get('ageSummary')
        guide(summary, f'{path}.ageSummary', need_research=False)
        cited = {s for g in lv.get('objectives') or [] if isinstance(g, dict) for s in g.get('sources') or []}
        if isinstance(summary, dict) and not set(summary.get('sources') or []) <= cited:
            out.append(f'{path}.ageSummary.sources: only sources the level\'s objectives cite')
        for part in GUIDANCE_PARTS:
            items = lv.get(part)
            if not isinstance(items, list) or not items:
                out.append(f'{path}.{part}: needs items')
                continue
            for k, g in enumerate(items):
                guide(g, f'{path}.{part}[{k}]', bands)
            for b in bands:
                if not [g for g in items if not g.get('bands') or b in g['bands']]:
                    out.append(f'{path}.{part}: nothing for band {b}')
        ld = lv.get('detectors', {})
        if set(ld) != set(S.DETECTORS):
            out.append(f'{path}.detectors: must list the six detectors')
        for d, spec in ld.items():
            dp = f'{path}.detectors.{d}'
            lists = [spec.get('verdicts', [])]+list((spec.get('verdictsByBand') or {}).values())
            for b in spec.get('verdictsByBand') or {}:
                if b not in bands:
                    out.append(f'{dp}.verdictsByBand: {b!r} is not a band of this level')
            for vs in lists:
                if not isinstance(vs, list) or len(set(vs)) != len(vs) or any(v not in S.FINDING_VERDICTS
                                                                               for v in vs):
                    out.append(f'{dp}.verdicts: {vs!r}')
                    continue
                if level <= 2 and 'work_on' in vs:
                    out.append(f'{dp}: level {level} frames work items as try_this, never work_on')
                if level >= 4 and 'try_this' in vs:
                    out.append(f'{dp}: try_this is a level 1-3 framing')
            if spec.get('verdicts') and spec.get('framing') is not None:
                _text_ok(spec['framing'], 200, f'{dp}.framing', out)
            elif spec.get('verdicts'):
                out.append(f'{dp}.framing: an enabled detector names its framing')
            for k, v in (spec.get('params') or {}).items():
                if not (v is None or isinstance(v, (bool, str)) or _is_num(v)):
                    out.append(f'{dp}.params.{k}: not a scalar')
            if spec.get('verdicts') and d not in lv.get('priorities', {}).get('relevance', {}):
                out.append(f'{dp}: an enabled detector needs a relevance weight')
        pr = lv.get('priorities', {})
        for s in pr.get('order', []):
            if s not in S.SKILLS:
                out.append(f'{path}.priorities.order: {s!r} is not a skill')
        for d, w in pr.get('relevance', {}).items():
            if d not in S.DETECTORS or not (_is_num(w) and 0 <= w <= 1):
                out.append(f'{path}.priorities.relevance.{d}: {w!r}')
        bb = lv.get('behindBall', {})
        if sorted(bb) != sorted(FORMAT_KEYS):
            out.append(f'{path}.behindBall: needs every format')
        for f, r in bb.items():
            if r.get('rule') == 'count':
                if not (isinstance(r.get('minBehind'), int) and r['minBehind'] >= 1):
                    out.append(f'{path}.behindBall.{f}: minBehind >= 1')
            elif r.get('rule') == 'margin':
                if not isinstance(r.get('minMargin'), int):
                    out.append(f'{path}.behindBall.{f}: minMargin')
            else:
                out.append(f'{path}.behindBall.{f}: rule count or margin')
        ev = lv.get('evidence', {})
        rec = {k: v for k, v in ev.items() if k != 'minFailureShareStyleBase'}
        rec.update(origin=S.ORIGIN_P, validated=False)
        for p in S.validate('profile', _probe_profile(level, rec)):
            if '.evidence' in p:
                out.append(f'{path}.evidence: {p}')
        if ev.get('minFailureShare') is None and not _is_num(ev.get('minFailureShareStyleBase')):
            out.append(f'{path}.evidence: a style-relative share needs its base value')
        reg_ = lv.get('register', {})
        if reg_.get('numbers') not in ('none', 'counts', 'rounded_metres', 'metres', 'shares', 'rates'):
            out.append(f'{path}.register.numbers')
        if reg_.get('maxChallenges') != ev.get('maxWorkItems'):
            out.append(f'{path}.register.maxChallenges: must equal evidence.maxWorkItems')
        _text_ok(reg_.get('tone'), 80, f'{path}.register.tone', out)
        for w in reg_.get('concepts', [])+reg_.get('banned', []):
            if not isinstance(w, str) or not WORD.fullmatch(w):
                out.append(f'{path}.register: {w!r} is not a lower-case word or phrase')
        clash = set(reg_.get('concepts', [])) & set(reg_.get('banned', []))
        if clash:
            out.append(f'{path}.register: both a concept and banned: {sorted(clash)}')
        if set(lv.get('skills', {})) != set(S.SKILLS):
            out.append(f'{path}.skills: must name the six skills')
        for s, name in lv.get('skills', {}).items():
            if name is not None:
                _text_ok(name, 80, f'{path}.skills.{s}', out)
            elif ld.get(S.SKILLS.get(s), {}).get('verdicts') and level > 2:
                out.append(f'{path}.skills.{s}: an enabled detector from level 3 shows its skill')
        for i in lv.get('drillPool', []):
            if not isinstance(i, str) or not DRILL_ID.fullmatch(i):
                out.append(f'{path}.drillPool: {i!r}')
        for k, f in enumerate(lv.get('focus', [])):
            fp = f'{path}.focus[{k}]'
            _text_ok(f.get('spec'), 120, f'{fp}.spec', out)
            if f.get('status') not in FOCUS_STATUS:
                out.append(f'{fp}.status: {f.get("status")!r}')
            for d in f.get('detectors', []):
                if d not in S.DETECTORS:
                    out.append(f'{fp}.detectors: {d!r}')
            if f.get('unavailable') is not None and f['unavailable'] not in cats:
                out.append(f'{fp}.unavailable: {f["unavailable"]!r} is not a category')
            if f.get('status') == 'unavailable' and not f.get('unavailable'):
                out.append(f'{fp}: an unavailable focus names its category')
            if f.get('note') is not None:
                _text_ok(f['note'], 200, f'{fp}.note', out)
        if lv.get('origin') != S.ORIGIN_P or lv.get('validated') is not False:
            out.append(f'{path}: origin PITCHIQ-PROPOSAL, validated false')
        k_ = lv.get('baseSuccessScale')
        if k_ is not None and not (_is_num(k_) and 0.0 < k_ <= 1.0):
            out.append(f'{path}.baseSuccessScale: a share in (0, 1]')
    for code, c in cats.items():
        if not CODE.fullmatch(code):
            out.append(f'unavailable.{code}: not a code')
        _text_ok(c.get('category'), 200, f'unavailable.{code}.category', out)
        _text_ok(c.get('reason'), 400, f'unavailable.{code}.reason', out)
        if c.get('unlock') is not None:
            _text_ok(c['unlock'], 300, f'unavailable.{code}.unlock', out)
    groups = data.get('ages', {}).get('groups', {})
    rules = data.get('heading', {}).get('rules', {})
    if set(groups) != set(S.AGE_GROUPS):
        out.append('ages.groups: must list every age group')
    for a, row in groups.items():
        ap = f'ages.groups.{a}'
        if row.get('typicalLevel') not in range(1, 8):
            out.append(f'{ap}.typicalLevel')
        if row.get('naming') not in NAMING_ORDER or row.get('register') not in ('child', 'teen', 'adult'):
            out.append(f'{ap}: naming or register unknown')
        if row.get('maxLoad') not in ('low', 'moderate', 'high'):
            out.append(f'{ap}.maxLoad')
        if not isinstance(row.get('tackling'), bool) or not isinstance(row.get('sprintRaces'), bool):
            out.append(f'{ap}: tackling and sprintRaces are booleans')
        if not _is_num(row.get('reachSpeedCapMS')) or not 2 <= row['reachSpeedCapMS'] <= 10:
            out.append(f'{ap}.reachSpeedCapMS')
        rule = rules.get(row.get('heading'))
        if rule is None:
            out.append(f'{ap}.heading: no rule {row.get("heading")!r}')
        elif a in S.AGE_GROUPS and S.AGE_ORDER[a] <= S.AGE_ORDER['U11'] and rule.get('heading') != 'none':
            out.append(f'{ap}.heading: no heading at U11 or younger (DESIGN 2.4)')
    for key, rule in rules.items():
        rp = f'heading.rules.{key}'
        if rule.get('heading') not in ('none', 'limited'):
            out.append(f'{rp}.heading')
        if (rule.get('heading') == 'limited') != isinstance(rule.get('caps'), dict):
            out.append(f'{rp}.caps: limited heading needs caps, none needs null')
        for k, v in (rule.get('caps') or {}).items():
            if k in ('contested', 'lightBall', 'unopposed'):
                if not isinstance(v, bool):
                    out.append(f'{rp}.caps.{k}: boolean')
            elif not (isinstance(v, int) and not isinstance(v, bool) and v > 0):
                out.append(f'{rp}.caps.{k}: positive integer')
        _text_ok(rule.get('text'), 300, f'{rp}.text', out)
        for sid in rule.get('sources', []):
            if sid not in srcs or srcs[sid]['read'] == 'not_read':
                out.append(f'{rp}.sources: {sid} not a read source')
    for k, g in enumerate(data.get('heading', {}).get('guidance', [])):
        guide(g, f'heading.guidance[{k}]')
    for k, n in enumerate(data.get('safetyNotes', {}).get('notes', [])):
        np_ = f'safetyNotes.notes[{k}]'
        _text_ok(n.get('text'), 300, f'{np_}.text', out)
        if n.get('tag') not in S.TAGS or (n.get('tag') == 'FEDERATION' and not n.get('sources')):
            out.append(f'{np_}: tag and sources')
        for a in n.get('ages', []):
            if a not in S.AGE_GROUPS:
                out.append(f'{np_}.ages: {a!r}')
        for sid in n.get('sources', []):
            if sid not in srcs or srcs[sid]['read'] == 'not_read':
                out.append(f'{np_}.sources: {sid} not a read source')
    fmts = data.get('formats', {})
    for key in FORMAT_KEYS:
        f = fmts.get(key)
        fp = f'formats.{key}'
        if not isinstance(f, dict):
            out.append(f'{fp}: missing')
            continue
        players = int(key)
        names = [m.get('name') for m in f.get('formations', [])]
        if sorted(names) != sorted(S.FORMATIONS[players]):
            out.append(f'{fp}.formations: must be exactly {S.FORMATIONS[players]}')
        for m in f.get('formations', []):
            if m.get('tag') not in S.TAGS or (m.get('tag') == 'FEDERATION' and not m.get('sources')):
                out.append(f'{fp}.formations.{m.get("name")}: tag and sources')
            for sid in m.get('sources', []):
                if sid not in srcs or srcs[sid]['read'] == 'not_read':
                    out.append(f'{fp}.formations.{m.get("name")}.sources: {sid} not a read source')
            _text_ok(m.get('note'), 120, f'{fp}.formations.{m.get("name")}.note', out)
        if players in (7, 9, 11) and (f.get('goalkeeperDefault') is not True or f.get('goalkeeperChoice')):
            out.append(f'{fp}: {players}v{players} always has a goalkeeper')
        if players < 11 and (f.get('templates') or f.get('roleModel') != 'relative'):
            out.append(f'{fp}: small formats never use team_shape templates (DESIGN 2.5)')
        _text_ok(f.get('note'), 300, f'{fp}.note', out)
    by_age = data.get('formatDefaults', {}).get('byAge', {})
    if set(by_age) != set(S.AGE_GROUPS):
        out.append('formatDefaults.byAge: must list every age group')
    for a, row in by_age.items():
        for fed in ('US', 'England', 'Canada'):
            spec = row.get(fed)
            if _clean_format(spec) != spec:
                out.append(f'formatDefaults.byAge.{a}.{fed}: {spec!r} is not a format')
    styles = data.get('styles', {})
    for s in S.STYLES:
        st = styles.get(s)
        if not isinstance(st, dict):
            out.append(f'styles.{s}: missing')
            continue
        _text_ok(st.get('name'), 60, f'styles.{s}.name', out)
        _text_ok(st.get('description'), 300, f'styles.{s}.description', out)
        for d, eff in st.get('effects', {}).items():
            if d not in S.DETECTORS or not CODE.fullmatch(str(eff.get('code', ''))):
                out.append(f'styles.{s}.effects.{d}: detector and code')
            for op in ('add', 'multiply'):
                for k, v in eff.get(op, {}).items():
                    if not _is_num(v):
                        out.append(f'styles.{s}.effects.{d}.{op}.{k}: a number')
    for lvl, lim in styles.get('unknownLimits', {}).items():
        if lvl not in levels or any(d not in S.DETECTORS for d in lim.get('workOnDetectors', [])):
            out.append(f'styles.unknownLimits.{lvl}')
    notices = data.get('notices', {})
    for code, text in notices.items():
        if not CODE.fullmatch(code):
            out.append(f'notices.{code}: not a code')
        _text_ok(text, 400, f'notices.{code}', out)
    return out


def _check_guidance_core(g, path, out, srcs, bands):
    if not isinstance(g, dict):
        out.append(f'{path}: not an object')
        return
    extra = set(g)-{'text', 'tag', 'sources', 'research', 'edited', 'bands'}
    if extra:
        out.append(f'{path}: unknown keys {sorted(extra)}')
    if _text_ok(g.get('text'), 400, f'{path}.text', out):
        for q in QUOTE.findall(g['text']):
            if len(q) > 60:
                out.append(f'{path}.text: a quoted phrase of {len(q)} characters (at most 60)')
    if g.get('tag') not in S.TAGS:
        out.append(f'{path}.tag: {g.get("tag")!r}')
    sources = g.get('sources')
    if not isinstance(sources, list):
        out.append(f'{path}.sources: not a list')
        sources = []
    if g.get('tag') == 'FEDERATION' and not sources:
        out.append(f'{path}: a FEDERATION item needs its source ids')
    for sid in sources:
        s = srcs.get(sid)
        if s is None:
            out.append(f'{path}.sources: {sid} is not a canonical id in sources.json')
        elif s['read'] == 'not_read':
            out.append(f'{path}.sources: {sid} was not read and cannot be cited for content')
    for b in g.get('bands') or []:
        if b not in bands:
            out.append(f'{path}.bands: {b!r} is not a band of this level')
    if g.get('edited') is not None and g.get('edited') is not True:
        out.append(f'{path}.edited: true or absent')


def _probe_profile(level, evidence):
    """A minimal profile record carrying an evidence block, so the schema's EvidenceRules checks can be reused."""
    p = S.example('profile')
    p['level'] = level
    p['levelName'] = S.LEVEL_NAMES[level]
    p['evidence'] = evidence
    for d in p['detectors'].values():
        d['verdicts'] = ['strength', 'review']
    return p
