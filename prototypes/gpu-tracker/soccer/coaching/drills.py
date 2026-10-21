"""The training exercises a coaching report links to its findings: a small reviewed library (drills.json), loaded and
checked here.

Why a fixed library and not generated exercises: a report that pairs a measured finding with an invented or unsafe
session is worse than one that pairs it with nothing. So every exercise in drills.json is a conventional session
format (small-sided game with a zone condition, unit shifting against servers, back-line stepping and dropping,
rondo-style keep-ball, shadow play, unit pattern play) with its purpose, the tactical issue it addresses, area,
equipment, set-up, instructions, player count, duration, coaching points, progressions and regressions, a measurable
success criterion in the exercise and one back in the coaching report's own measurement (REPORT_LINK), a variation per
context level and safety notes, and it names the recommendation keys (soccer/coaching/language.py) it serves. A
finding only gets an exercise whose keys include the finding's key: an exercise is never attached to an issue it was
not written for.

The player counts hold together: every "N v M" in an exercise's variations and player note fits its player range
(versus_counts). The first library asked for 4 v 4 at development level in an exercise for at least 12 players, and
for 4 v 1 in one whose note said 4 v 2.

The text is plain printable ASCII (the PDF writer uses base-14 Helvetica with WinAnsi encoding, where anything else
prints as '?'), and names and purposes carry no digits: the report quotes them inside a finding's text, and every
number in that text must come from the facts (language.validate_text); the drill's own player count and duration are
added to those facts from the library record.
"""
import copy
import hashlib
import json
import re
from pathlib import Path

DRILLS_PATH = Path(__file__).with_name('drills.json')
LEVELS = ('development', 'competitive', 'advanced')   # the report's context levels (language.LEVELS)
TEXT_FIELDS = ('name', 'purpose', 'tacticalIssue', 'area', 'setup', 'source')
LIST_FIELDS = ('instructions', 'coachingPoints', 'safety', 'equipment', 'regressions', 'successCriteria')   # non-empty
REPORT_LINK = 'coaching report'   # a success criterion names the report measurement it should move
VERSUS = re.compile(r'(\d+) v (\d+)')
OPTIONAL_LIST_FIELDS = ('progressions', 'patterns')
MAX_PLAYERS = 30            # a session for one squad; anything larger is a typo
MIN_MINUTES, MAX_MINUTES = 5, 45   # one exercise, not a whole session
ID_RE = re.compile(r'[a-z][a-z0-9_]{2,40}')
PRINTABLE = re.compile(r'[ -~]*')   # printable ASCII: a strict subset of WinAnsi

_cache = {}


def load_drills(path=None):
    """The validated library {'version', 'notes', 'drills': [...], '_sha'} (cached per file and modification time).
    Raises ValueError listing every problem when the file does not validate: a broken library must not reach a
    report."""
    p = Path(path) if path is not None else DRILLS_PATH
    st = p.stat()
    key = (str(p.resolve()), st.st_mtime_ns, st.st_size)
    if _cache.get('key') != key:
        data = p.read_bytes()
        library = json.loads(data.decode('utf-8'))
        problems = validate_drills(library)
        if problems:
            raise ValueError('drill library invalid: '+'; '.join(problems))
        library['_sha'] = hashlib.sha256(data).hexdigest()[:16]
        _cache.update(key=key, library=library)
    return _cache['library']


def library_sha(library):
    """The first 16 hex of the library file's sha256 (part of the report fingerprint), 'inline' for a library that
    did not come from a file."""
    return library.get('_sha', 'inline')


def validate_drills(library, known_keys=None):
    """Every problem with the library, as readable strings (empty when it is valid). known_keys defaults to the
    recommendation keys language.py has templates for."""
    if known_keys is None:
        from soccer.coaching import language
        known_keys = set(language.RECOMMENDATION_KEYS)
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
        else:
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
    """The player totals of every 'N v M' in a text (4 v 2 -> 6)."""
    return [int(a)+int(b) for a, b in VERSUS.findall(text or '')]


def drill(library, drill_id):
    """The drill with that id, or None."""
    for d in library.get('drills') or ():
        if d.get('id') == drill_id:
            return d
    return None


def drills_for(library, recommendation_key, pattern=None):
    """The drills written for a recommendation key, in library order; those that also name the pattern first."""
    found = [d for d in library.get('drills') or () if recommendation_key in (d.get('recommendationKeys') or ())]
    if pattern:
        found.sort(key=lambda d: pattern not in (d.get('patterns') or ()))
    return found


def choose(library, observation):
    """The drill for an observation: the first of its drillIds that exists and is written for its recommendation
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
    """The drill as a report carries it: the library record (without internal fields) with the variation for the
    report's context level as 'variation'."""
    out = {k: copy.deepcopy(v) for k, v in d.items() if not k.startswith('_') and k != 'variations'}
    out['variation'] = (d.get('variations') or {}).get(level)
    out['level'] = level
    return out


def drill_numbers(d):
    """The numbers a finding may quote from a drill: its player count and duration bounds."""
    out = []
    for f in ('players', 'durationMinutes'):
        r = d.get(f) or {}
        out += [float(r[k]) for k in ('min', 'max') if isinstance(r.get(k), (int, float))]
    return out


def all_text(library):
    """Every string in the library with its path, for the ASCII and wording checks."""
    return list(_strings(library))


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
