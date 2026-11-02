"""Full Match Intelligence over an assembled match: the stages, their caches and one snapshot the routes read.

    1. data       the 5 Hz match data (soccer/moments/data.py; assembly/coaching/moments/data-<fp>.npz, ~1 min once)
    2. persons    the automatic people and places (fullmatch/persons.py; assembly/intel/persons-<fp>.json, seconds)
    3. people     the people under the coach's corrections (persons.resolve; in memory, milliseconds)
    4. halftime   the halftime and the first-half direction (fullmatch/halftime.py; assembly/halftime.json)
    5. state      the state timeline and events (fullmatch/gamestate.py; assembly/intel/state-<fp>.json, keyed by the
                  data, the people's teams and places, the halftime), under the coach's event corrections
                  (assembly/intel/events.json)
    6. masks      the analytics masks from the state (gamestate.masks)
Nothing earlier is re-run by a later stage: a halftime change recomputes stage 5 and what reads the directions; a
roster correction stages 3 and 5; an event correction only the corrected view of stage 5. No detection, tracking or
calibration is ever touched.

snapshot(folder) answers from the caches only (None when stage 1 or 2 is missing: build() makes them, on a thread the
routes start). The snapshot is memoised per folder on the stamps of every file it read.
"""
import hashlib
import json
import threading
from pathlib import Path

from . import corrections as match_corrections
from . import gamestate
from . import halftime as ht
from . import persons

_memo = {}
_lock = threading.Lock()


def _stamp(path):
    try:
        st = Path(path).stat()
        return (st.st_mtime_ns, st.st_size)
    except OSError:
        return None


def _stamps(folder):
    asm = Path(folder)/'assembly'
    files = [asm/'match.json', asm/'possession.json', asm/match_corrections.FILE, asm/ht.FILE, asm/'intel'/gamestate.FILE]
    files += sorted((asm/'intel').glob('persons-*.json'))
    return tuple((str(p), _stamp(p)) for p in files)


def load_data(folder, progress=None):
    """The match's MatchData (built and cached when missing)."""
    from soccer.moments import data as md
    return md.match_data(folder, progress=progress)


def has_data(folder):
    from soccer.moments import data as md
    try:
        fp = md.source_fingerprint(folder)
    except (OSError, ValueError, KeyError):
        return False
    return (Path(folder)/'assembly'/'coaching'/'moments'/f'data-{fp[:16]}.npz').is_file()


def people_fingerprint(auto, items):
    """What the state depends on of the people: the automatic result and the corrections that change teams, keepers,
    exclusions or places."""
    keep = [c for c in items or [] if c.get('kind') in persons.PLACE_KINDS+('confirm',)]
    return hashlib.sha256(json.dumps([auto.get('fingerprint'), keep], sort_keys=True).encode()).hexdigest()[:16]


def build(folder, video=None, progress=None):
    """Every stage made and cached; returns the snapshot."""
    folder = Path(folder)
    say = progress or (lambda f, text: None)
    say(.02, 'Reading the match data (about a minute the first time)')
    data = load_data(folder, progress=lambda i, n: say(.02+.5*i/max(1, n), 'Reading the match data'))
    say(.55, 'Finding the people and their places')
    persons.cached(folder, data=data)
    say(.8, 'Recognising stoppages, restarts and goals')
    snap = snapshot(folder, video=video, data=data, make_state=True)
    say(1.0, 'Ready')
    return snap


def snapshot(folder, video=None, data=None, make_state=False):
    """{'ready', 'index', 'auto', 'people', 'items', 'halftime', 'directionPeriods', 'state', 'masks'} from the caches
    (the state is computed when missing and make_state, or when the match data is at hand); None when the automatic
    people are not built yet."""
    folder = Path(folder)
    key = (str(folder), _stamps(folder), str(video) if video else None)
    with _lock:
        hit = _memo.get(str(folder))
        if hit and hit[0] == key and hit[1].get('state') is not None:
            return hit[1]
    auto = persons.latest(folder)
    if auto is None:
        return None
    index = json.loads((folder/'assembly'/'match.json').read_text(encoding='utf-8'))
    items = match_corrections.load(folder)
    people = persons.resolve(auto, items)
    settings = ht.load(folder, index)
    out = {'ready': True, 'index': index, 'auto': auto, 'people': people, 'items': items, 'halftime': settings,
           'directionPeriods': ht.periods(index, settings), 'state': None, 'masks': None}
    pfp = people_fingerprint(auto, items)
    state = _cached_state(folder, settings, pfp)
    if state is None and (make_state or data is not None):
        if data is None:
            data = load_data(folder)
        possession = _read_json(folder/'assembly'/'possession.json') or {}
        from . import gamestate as gs
        dcopy = data
        ht.apply_to_data(dcopy, index, settings)
        state = gs.cached(folder, dcopy, possession, settings, index, people, pfp, places=people['places'], video=video)
    if state is not None:
        state = gamestate.apply_corrections(state, gamestate.load_corrections(folder))
        out['state'] = state
        out['masks'] = gamestate.masks(state)
    with _lock:
        _memo[str(folder)] = (key, out)
    return out


def _cached_state(folder, settings, pfp):
    """The newest cached state of this halftime and these people (no match data needed to find it)."""
    want = (ht.fingerprint(settings), pfp)
    best = None
    for path in (Path(folder)/'assembly'/'intel').glob('state-*.json'):
        out = _read_json(path)
        if not out or out.get('version') != gamestate.VERSION:
            continue
        if tuple(out.get('keys') or ()) == want:
            st = path.stat().st_mtime_ns
            if best is None or st > best[0]:
                best = (st, out)
    return best[1] if best else None


def _read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None


def effective(folder):
    """(aliases {fragment: person}, excluded {fragment}, teams {person: team}) for whatever reads the match's people,
    or None when the automatic people are not built (then the plain corrections apply)."""
    snap = snapshot(folder)
    if snap is None:
        return None
    return aliases_of(snap['people'])


def aliases_of(people):
    aliases = {f: pid for f, pid in people['person'].items() if pid != f}
    teams = {pid: p['team'] for pid, p in people['people'].items() if p.get('team') in ('A', 'B')}
    return aliases, set(people['excluded']), teams


def masks_fingerprint(snap):
    if not snap or not snap.get('masks'):
        return 'none'
    return hashlib.sha256(json.dumps(snap['masks'], sort_keys=True).encode()).hexdigest()[:16]


def keep_function(snap, kind):
    """A function times -> bool array: the samples `kind` of analysis counts (gamestate.ELIGIBLE), or None."""
    if not snap or not snap.get('masks'):
        return None
    runs = snap['masks']

    def keep(times):
        return gamestate.eligible(gamestate.mask_at(runs, times), kind)
    return keep


def dead_spans(snap):
    """The intervals ordinary tactical analysis leaves out (everything but ACTIVE_OPEN_PLAY and UNKNOWN), as the coach
    possession spans of the match coaching report (state 'dead')."""
    if not snap or not snap.get('masks'):
        return None
    out = []
    for m in snap['masks']:
        if m['mask'] in ('ACTIVE_OPEN_PLAY', 'UNKNOWN') or m['t1'] <= m['t0']:
            continue
        if out and m['t0']-out[-1]['t1'] <= .5:
            out[-1]['t1'] = m['t1']
        else:
            out.append({'t0': m['t0'], 't1': m['t1'], 'state': 'dead'})
    return {'version': 1, 'kind': 'coach-possession-spans', 'source': 'match-state', 'spans': out}
