"""Possession v2 over a whole assembled match (soccer/possession_v2.py on one continuous timeline).

    1. table     every segment view's owned frames read once into one raw frame table (the ball, the people near it, every
                 player's position): assembly/intel/possession2-table-<fp>.npz, keyed by the views (about a minute once)
    2. record    the table under the match's people (fullmatch/persons.py: persons, teams, exclusions), decoded over the
                 whole match with the game state's stoppages (fullmatch/gamestate.py) and the halftime; intervals,
                 stats for the match and both halves: assembly/intel/possession2-<fp>.json, keyed by the table, the
                 people, the state and the halftime (seconds)
Segment boundaries are not breaks: the views' owned frames join at the cuts, so a possession runs across them. Confirmed
stoppages (OUT_OF_PLAY, RESTART_PREPARATION), HALFTIME and NON_MATCH_FOOTAGE break the sequence and are left out of the
shares; POSSIBLE_STOPPAGE is decoded (play may be live) but shown apart and left out of the shares; IN_PLAY and the
state engine's UNKNOWN are live. A restart the state engine typed with a team starts the next sequence with that team
as a prior. Nothing is re-run and nothing earlier is changed: the old possession record (assembly/possession.json) stays
what the game state reads.
"""
import gzip
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from soccer import possession_v2 as pv

TABLE_VERSION = 1
STATE_CODE = {'IN_PLAY': -1, 'UNKNOWN': -1, 'POSSIBLE_STOPPAGE': pv.POSSIBLE, 'OUT_OF_PLAY': pv.STOPPAGE,
              'RESTART_PREPARATION': pv.STOPPAGE, 'HALFTIME': pv.HALFTIME, 'NON_MATCH_FOOTAGE': pv.NON_MATCH}
RESTARTS = ('KICK_OFF', 'CORNER_KICK', 'GOAL_KICK', 'THROW_IN', 'FREE_KICK', 'PENALTY_KICK', 'UNKNOWN_RESTART')
RESTART_NEAR_S = 2.0      # a live sequence starting this close to a typed restart takes its team as the prior


def _intel_dir(folder):
    d = Path(folder)/'assembly'/'intel'
    d.mkdir(parents=True, exist_ok=True)
    return d


def table_fingerprint(index):
    """The raw table depends on the views only (and this code's table version)."""
    views = [(s.get('view'), s.get('viewBytes'), s.get('cut0'), s.get('cut1')) for s in index.get('segments') or []]
    return hashlib.sha256(json.dumps([TABLE_VERSION, views], sort_keys=True).encode()).hexdigest()[:16]


def _save_table(path, T):
    arrays = {k: T[k] for k in pv.FRAME_KEYS+pv.NEAR_KEYS+pv.ROW_KEYS+('nr_start', 'rw_start')}
    meta = {'n': T['n'], 'raw_ids': T['raw_ids'], 'length': T['length'], 'width': T['width'], 'fps': T['fps'],
            'segments': T.get('segments')}
    tmp = path.with_suffix('.tmp.npz')
    np.savez_compressed(tmp, __meta__=np.frombuffer(json.dumps(meta).encode('utf-8'), np.uint8), **arrays)
    tmp.replace(path)


def _load_table(path):
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(bytes(z['__meta__']).decode('utf-8'))
        T = {k: z[k] for k in z.files if k != '__meta__'}
    T.update(n=int(meta['n']), raw_ids=list(meta['raw_ids']), length=float(meta['length']), width=float(meta['width']),
             fps=float(meta['fps']), segments=meta.get('segments'))
    return T


def frame_table(folder, index=None, progress=None):
    """The match's raw frame table (cached): every view's owned frames in match order."""
    folder = Path(folder)
    if index is None:
        index = json.loads((folder/'assembly'/'match.json').read_text(encoding='utf-8'))
    fp = table_fingerprint(index)
    path = _intel_dir(folder)/f'possession2-table-{fp}.npz'
    if path.exists():
        try:
            return _load_table(path), fp
        except (OSError, ValueError, KeyError):
            pass
    say = progress or (lambda f, text: None)
    raw_ids, parts, segs = [], [], []
    segments = index.get('segments') or []
    for k, s in enumerate(segments):
        say(k/max(1, len(segments)), f'Reading segment {k+1} of {len(segments)}')
        with gzip.open(folder/'assembly'/s['view'], 'rt', encoding='utf-8') as fh:
            view = json.load(fh)
        T = pv.table(view, owned=(int(s['cut0']), int(s['cut1'])), seg=int(s.get('index', k)), raw_ids=raw_ids)
        parts.append(T)
        if T['n']:
            segs.append({'index': int(s.get('index', k)), 't0': float(T['t'][0]), 't1': float(T['t'][-1])})
        del view
    T = pv.concat(parts)
    T['segments'] = segs
    _save_table(path, T)
    for old in path.parent.glob('possession2-table-*.npz'):
        if old != path:
            try:
                old.unlink()
            except OSError:
                pass
    return T, fp


def live_codes(T, state):
    """Per frame the code the game state gives (-1 live; POSSIBLE / STOPPAGE / HALFTIME / NON_MATCH otherwise)."""
    t = T['t']
    out = np.full(T['n'], -1, np.int8)
    for r in (state or {}).get('timeline') or []:
        code = STATE_CODE.get(r.get('state'), -1)
        if code < 0:
            continue
        a = int(np.searchsorted(t, float(r['t0'])-1e-9))
        b = int(np.searchsorted(t, float(r['t1'])+1e-9, side='right'))
        out[a:b] = code
    return out


def restart_starts(T, decode_live, state):
    """{first frame of each decoded sequence: team index of a typed restart within RESTART_NEAR_S, else None}."""
    t = T['t']
    events = [e for e in (state or {}).get('events') or []
              if e.get('type') in RESTARTS and e.get('status') != 'removed' and e.get('team') in pv.TEAMS]
    times = np.array([float(e['t']) for e in events]) if events else np.zeros(0)
    out = {}
    for a, b in pv._runs(decode_live):
        out[a] = None
        if len(times):
            j = int(np.argmin(np.abs(times-t[a])))
            if abs(times[j]-t[a]) <= RESTART_NEAR_S:
                out[a] = pv.TEAMS.index(events[j]['team'])
    return out


def people_maps(people):
    """(person_of {raw id: person}, team_of {person: team}, excluded {raw id}, names {person: {display, team, keeper}})."""
    if not people:
        return {}, {}, set(), {}
    person_of = dict(people.get('person') or {})
    team_of = {pid: p.get('team') for pid, p in (people.get('people') or {}).items() if p.get('team') in pv.TEAMS}
    excluded = set(people.get('excluded') or ())
    names = {pid: {'display': p.get('display'), 'team': p.get('team'), 'keeper': bool(p.get('keeper'))}
             for pid, p in (people.get('people') or {}).items()}
    return person_of, team_of, excluded, names


def halves(periods):
    out = []
    for p in periods or []:
        if p.get('start') is None or p.get('end') is None:
            continue
        out.append({'name': p.get('name') or f"period {p.get('index', len(out))+1}", 't0': float(p['start']),
                    't1': float(p['end'])})
    return out


def decode_match(T, people=None, state=None, periods=None, ends=None, settings=None, params=None):
    """The record of a raw table: people applied, decoded with the state's stoppages, intervals and stats."""
    t0 = time.perf_counter()
    person_of, team_of, excluded, names = people_maps(people)
    R = pv.resolve(T, person_of=person_of, team_of=team_of, excluded=excluded, ends=ends, S=settings)
    M, aux = pv.classes(R, settings)
    codes = live_codes(R, state)
    decode_live = (codes == -1) | (codes == pv.POSSIBLE)
    starts = restart_starts(R, decode_live, state)
    mod = pv.model(params)
    post, path = pv.decode(R, M, decode_live, mod, starts=starts, S=settings)
    out = pv.label(R, post, path, M, aux, codes, settings)
    bst, bconf, bprov = pv.ball_states(R)
    rec = pv.record(R, out, post, path, M, bst, halves=halves(periods), names=names, settings=settings)
    rec['seconds'] = round(time.perf_counter()-t0, 2)
    return rec, R, out, post


def record_fingerprint(table_fp, people_fp, state, halftime_fp, settings=None):
    st = state or {}
    key = [pv.VERSION, pv.model_id(), table_fp, people_fp, st.get('fingerprint'), st.get('corrections'), halftime_fp,
           sorted((settings or {}).items())]
    return hashlib.sha256(json.dumps(key, sort_keys=True, default=str).encode()).hexdigest()[:16]


def cached_record(folder, fp):
    path = Path(folder)/'assembly'/'intel'/f'possession2-{fp}.json'
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None


def build(folder, snap, progress=None, settings=None):
    """The match's possession v2 record (cached): the frame table (read once), then the decode under the snapshot's
    people, state and halftime."""
    from . import halftime as ht, intel
    folder = Path(folder)
    say = progress or (lambda f, text: None)
    index = snap['index'] if snap else json.loads((folder/'assembly'/'match.json').read_text(encoding='utf-8'))
    t0 = time.perf_counter()
    T, tfp = frame_table(folder, index, progress=lambda f, text: say(.05+.75*f, text))
    table_s = time.perf_counter()-t0
    people = snap['people'] if snap else None
    state = snap['state'] if snap else None
    pfp = intel.people_fingerprint(snap['auto'], snap['items']) if snap else 'none'
    hfp = ht.fingerprint(snap['halftime']) if snap else 'none'
    fp = record_fingerprint(tfp, pfp, state, hfp, settings)
    hit = cached_record(folder, fp)
    if hit is not None:
        return hit
    say(.85, 'Working out possession over the whole match')
    periods = snap['directionPeriods'] if snap else index.get('periods')
    rec, R, out, post = decode_match(T, people=people, state=state, periods=periods, settings=settings)
    rec['fingerprint'] = fp
    rec['tableSeconds'] = round(table_s, 2)
    rec['source'] = {'kind': 'match', 'segments': len(index.get('segments') or []),
                     'state': 'match state' if state else 'none (no stoppages known)',
                     'people': 'match people' if people else 'segment identities'}
    if state is None:
        rec['notes'].insert(0, 'The match state is not built: no stoppage, halftime or restart is known, so every frame '
                               'counts as live play. Build the match state for the in-play shares.')
    path = _intel_dir(folder)/f'possession2-{fp}.json'
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(rec, separators=(',', ':'), allow_nan=False), encoding='utf-8')
    tmp.replace(path)
    for old in path.parent.glob('possession2-*.json'):
        if old != path and not old.name.startswith('possession2-table'):
            try:
                old.unlink()
            except OSError:
                pass
    say(1.0, 'Ready')
    return rec


def latest(folder, snap, settings=None):
    """The cached record for the snapshot's people, state and halftime, or None (no reading of views)."""
    from . import halftime as ht, intel
    folder = Path(folder)
    index = snap['index'] if snap else json.loads((folder/'assembly'/'match.json').read_text(encoding='utf-8'))
    tfp = table_fingerprint(index)
    if not (folder/'assembly'/'intel'/f'possession2-table-{tfp}.npz').exists():
        return None
    pfp = intel.people_fingerprint(snap['auto'], snap['items']) if snap else 'none'
    hfp = ht.fingerprint(snap['halftime']) if snap else 'none'
    return cached_record(folder, record_fingerprint(tfp, pfp, snap['state'] if snap else None, hfp, settings))
