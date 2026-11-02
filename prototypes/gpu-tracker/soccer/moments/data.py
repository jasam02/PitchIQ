"""MatchData: a whole match (or one Segment run) as compact arrays at 5 Hz for the key-moments engine.

Built once from what the analysis saved, never re-running anything: the assembled views of a full match over the frames
each segment OWNS (cut0 <= frame < cut1, so a frame is counted once), the match's possession record and its periods
with their attacking directions. A Segment run gives the same from its result.

Per sample (every GRID-th owned frame):
    t, frame, segment, period             the source time (s), the absolute frame, the segment and period it belongs to
    calibrated, calibConf, pitchLevel     the frame's pitch registration (pitchLevel 3 HIGH .. 0 none)
    view                                  the camera's footprint on the pitch (6 corners, metres; NaN when unknown)
    ball, ballState, ballConf, ballPConf  the ball on the pitch (metres) when the run placed it, its state code
    possTeam, possState, possCarrier,     the possession record at that time: team (1 A, 2 B, 0 none), state code,
    possConf, possSettled, possRun        the carrier (person index or -1), confidence (0 low .. 2 high), settled = well
                                          inside one team's spell (POSS_SETTLE_S from either end, the spell at least
                                          POSS_MIN_S long), the index of that spell in possRuns (-1)
Per observation (CSR by sample: obsPtr[s]..obsPtr[s+1]):
    pid, x, y, vx, vy, sigma, idConf      the person, the position (metres, smoothed over +-SMOOTH_HALF frames of one
                                          unbroken run), the velocity (m/s, over +-VEL_HALF frames; NaN near a run's
                                          ends), the position's own uncertainty (m), the identity confidence
    box                                   the person's box in the frame (normalized x, y, w, h), for drawing
Per sample also H (the frame's pitch-metres -> pixels homography, 9 values, NaN when none) and the video's size in meta,
so a moment can be drawn over the video without reading the views again.
People: index -> {id, display, team, role}; corrections (merged / excluded / team) are applied on load (apply_corrections),
so the cache does not depend on them.

Only seen positions are used: a person the tracker carried without seeing them (evidence in UNSEEN), without a pitch
position, with a pitch confidence under MIN_PITCH_CONF or an uncertainty over MAX_SIGMA_M is left out at that frame. A
position whose identity was linked offline ('connected offline', 'reidentified', ...) was seen and is kept.
Coordinates: x along the pitch (0 .. length, metres), y across (0 .. width), as the pitch registration names them.
"""
import gzip
import hashlib
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

VERSION = 2
GRID = 5                 # one sample every 5th owned frame (5 Hz at 25 fps)
SMOOTH_HALF = 2          # positions averaged over +-2 frames within one unbroken run (0.2 s at 25 fps)
VEL_HALF = 5             # velocity from the smoothed positions +-5 frames apart (0.4 s)
RUN_GAP = 2              # frames: a person missing longer than this starts a new run (no smoothing across it)
MAX_SIGMA_M = 3.0        # a position less certain than this (its larger axis) is not used
MIN_PITCH_CONF = .4
UNSEEN = ('predicted', 'carried', 'held', 'estimated', 'interpolated')   # person evidence that is not a sighting
POSS_SETTLE_S = 1.0      # a possession sample is 'settled' this far inside its team's spell
POSS_MIN_S = 2.0         # and the spell is at least this long
POSS_JOIN_S = .5         # spans of one team closer than this are one spell
BALL_CODES = {'TRACKED': 1, 'BRIDGED': 2, 'PREDICTED': 2, 'HELD': 2, 'UNKNOWN': 3}            # anything else: 0
POSS_CODES = {'control': 1, 'flight': 2, 'contested': 3, 'dead': 4}                           # anything else: 0
POSS_STATE_NAMES = ('unknown', 'control', 'flight', 'contested', 'dead')
CONF_CODES = {'low': 0, 'medium': 1, 'high': 2}
LEVEL_CODES = {'LOW': 1, 'MEDIUM': 2, 'HIGH': 3}
TEAM_CODES = {'A': 1, 'B': 2}
TEAMS = ('', 'A', 'B')

SAMPLE_FIELDS = ('t', 'frame', 'segment', 'period', 'calibrated', 'calibConf', 'pitchLevel', 'view', 'H', 'ball', 'ballState',
                 'ballConf', 'ballPConf', 'possTeam', 'possState', 'possCarrier', 'possConf', 'possSettled', 'possRun')
OBS_FIELDS = ('obsPtr', 'pid', 'x', 'y', 'vx', 'vy', 'sigma', 'idConf', 'box')


class MatchData:
    """The arrays (see the module doc) plus people, periods, possession spells and changes, and where they came from."""

    def __init__(self, arrays, people, meta):
        for k in SAMPLE_FIELDS+OBS_FIELDS:
            setattr(self, k, arrays[k])
        self.people = people                    # [{'id', 'display', 'team', 'role'}]
        self.meta = meta                        # length, width, fps, periods, possRuns, changes, source, fingerprint
        self.length = float(meta['length'])
        self.width = float(meta['width'])
        self.fps = float(meta['fps'])
        self.periods = meta['periods']          # [{'index', 'start', 'end', 'name', 'attacksRight'}]
        self.possRuns = meta.get('possRuns') or []   # [{'team', 't0', 't1'}]
        self.changes = meta.get('changes') or []
        self.index_of = {p['id']: i for i, p in enumerate(people)}
        self.team_of = np.array([TEAM_CODES.get(p.get('team'), 0) for p in people], dtype=np.int8)
        self.is_keeper = np.array([p.get('role') == 'GOALKEEPER' for p in people], dtype=bool)

    @property
    def samples(self):
        return len(self.t)

    def rows(self, s):
        """The observation rows of sample s (a slice)."""
        return slice(int(self.obsPtr[s]), int(self.obsPtr[s+1]))

    def sample_at(self, t):
        """The sample nearest to time t."""
        i = int(np.searchsorted(self.t, t))
        if i <= 0:
            return 0
        if i >= len(self.t):
            return len(self.t)-1
        return i if abs(self.t[i]-t) < abs(self.t[i-1]-t) else i-1

    def obs_sample(self):
        """The sample of every observation row."""
        return np.repeat(np.arange(len(self.t)), np.diff(self.obsPtr))

    def direction(self, team, s):
        """+1 when `team` attacks towards x = length at sample s, -1 towards x = 0, 0 when unknown."""
        p = self.periods[int(self.period[s])] if 0 <= int(self.period[s]) < len(self.periods) else None
        if not p or p.get('attacksRight') not in ('A', 'B'):
            return 0
        return 1 if p['attacksRight'] == team else -1


# ---------------------------------------------------------------- building
def _possession_tables(possession, ids):
    """Spans and spells of the possession record, as arrays: (t0, t1, team, state, carrier, conf, estimated) and the
    team spells [{'team', 't0', 't1'}] (consecutive spans of one team in control / flight / contested)."""
    spans = sorted((s for s in (possession or {}).get('timeline') or [] if s.get('t1', 0) > s.get('t0', 0)),
                   key=lambda s: s['t0'])
    t0 = np.array([float(s['t0']) for s in spans]) if spans else np.zeros(0)
    t1 = np.array([float(s['t1']) for s in spans]) if spans else np.zeros(0)
    team = np.array([TEAM_CODES.get(s.get('team'), 0) for s in spans], dtype=np.int8)
    state = np.array([POSS_CODES.get(s.get('state'), 0) for s in spans], dtype=np.int8)
    carrier = np.array([ids.get(s.get('carrier'), -1) if s.get('carrier') else -1 for s in spans], dtype=np.int32)
    conf = np.array([CONF_CODES.get(s.get('confidence'), 0) for s in spans], dtype=np.int8)
    runs = []
    for s in spans:
        code = TEAM_CODES.get(s.get('team'), 0)
        if not code or POSS_CODES.get(s.get('state'), 0) not in (1, 2, 3):
            continue
        if runs and runs[-1]['team'] == s['team'] and float(s['t0'])-runs[-1]['t1'] <= POSS_JOIN_S:
            runs[-1]['t1'] = max(runs[-1]['t1'], float(s['t1']))
        else:
            runs.append({'team': s['team'], 't0': float(s['t0']), 't1': float(s['t1'])})
    return (t0, t1, team, state, carrier, conf), runs


def _possession_samples(t, tables, runs):
    t0, t1, team, state, carrier, conf = tables
    n = len(t)
    out = {'possTeam': np.zeros(n, np.int8), 'possState': np.zeros(n, np.int8), 'possCarrier': np.full(n, -1, np.int32),
           'possConf': np.zeros(n, np.int8), 'possSettled': np.zeros(n, bool), 'possRun': np.full(n, -1, np.int32)}
    if len(t0):
        i = np.searchsorted(t0, t, side='right')-1
        ok = (i >= 0) & (t < t1[np.clip(i, 0, None)])
        j = np.where(ok, i, 0)
        out['possTeam'] = np.where(ok, team[j], 0).astype(np.int8)
        out['possState'] = np.where(ok, state[j], 0).astype(np.int8)
        out['possCarrier'] = np.where(ok, carrier[j], -1).astype(np.int32)
        out['possConf'] = np.where(ok, conf[j], 0).astype(np.int8)
    if runs:
        r0 = np.array([r['t0'] for r in runs])
        r1 = np.array([r['t1'] for r in runs])
        k = np.searchsorted(r0, t, side='right')-1
        inside = (k >= 0) & (t < r1[np.clip(k, 0, None)])
        kk = np.where(inside, k, 0)
        long_enough = (r1[kk]-r0[kk]) >= POSS_MIN_S
        settled = inside & long_enough & (t-r0[kk] >= POSS_SETTLE_S) & (r1[kk]-t >= POSS_SETTLE_S)
        out['possRun'] = np.where(inside, k, -1).astype(np.int32)
        out['possSettled'] = settled
    return out


class _Collector:
    """Accumulates owned frames (people, ball, calibration, view) and turns them into the arrays."""

    def __init__(self, length, width, roster):
        self.length, self.width = float(length), float(width)
        self.ids = {}
        self.people = []
        self.roster = roster or {}
        self.frames = []          # (frame, time, segment, calibrated, calibConf, level, view, ball..., )
        self.H = []               # the frame's homography (9 values) or None
        self.obs = []             # (frame_index, pid, x, y, sigma, idConf, box x, y, w, h)
        self.evidence = defaultdict(int)

    def pid(self, person):
        key = person.get('id')
        i = self.ids.get(key)
        if i is None:
            i = self.ids[key] = len(self.people)
            known = self.roster.get(key) or {}
            self.people.append({'id': key, 'display': known.get('display') or person.get('display') or key,
                                'team': known.get('team') if known.get('team') in ('A', 'B') else person.get('team'),
                                'role': known.get('role') or person.get('role')})
        return i

    def add_frame(self, f, segment):
        k = len(self.frames)
        pitch = f.get('pitch') or {}
        ball = f.get('ball') or {}
        bp = ball.get('pitch')
        view = f.get('view')
        cal = f.get('calibration') if isinstance(f.get('calibration'), dict) else {}
        h = cal.get('H')
        self.H.append(h if isinstance(h, list) and len(h) == 9 else None)
        self.frames.append((int(f['frame']), float(f['time']), int(segment), bool(f.get('calibrated')),
                            float(f.get('pitchConfidence') or 0.0), LEVEL_CODES.get(pitch.get('level'), 0),
                            view if isinstance(view, list) and len(view) >= 3 else None,
                            (bp[0]*self.length, bp[1]*self.width) if isinstance(bp, list) and len(bp) == 2 else None,
                            BALL_CODES.get(ball.get('state'), 0), float(ball.get('confidence') or 0.0),
                            float(ball.get('pitchConfidence') or 0.0)))
        for p in f.get('people') or []:
            if p.get('id') is None or p.get('role') not in ('PLAYER', 'GOALKEEPER'):
                continue
            self.evidence[p.get('evidence')] += 1
            if p.get('evidence') in UNSEEN:
                continue
            pp = p.get('pitch')
            if not (isinstance(pp, list) and len(pp) == 2) or (p.get('pitchConfidence') or 0) < MIN_PITCH_CONF:
                continue
            sg = p.get('pitchSigma') or [MAX_SIGMA_M+1, MAX_SIGMA_M+1]
            sigma = float(max(sg)) if isinstance(sg, list) and sg else float(sg)
            if sigma > MAX_SIGMA_M:
                continue
            box = p.get('box') if isinstance(p.get('box'), list) and len(p['box']) == 4 else (np.nan,)*4
            self.obs.append((k, self.pid(p), pp[0]*self.length, pp[1]*self.width, sigma,
                             float(p.get('identityConfidence') or 0.0), *box))

    def finish(self, fps, periods, possession):
        n = len(self.frames)
        frame = np.array([f[0] for f in self.frames], dtype=np.int64)
        order = np.argsort(frame, kind='stable')
        if np.any(order != np.arange(n)):
            raise ValueError('owned frames must arrive in order')
        tsec = np.array([f[1] for f in self.frames])
        obs = np.array(self.obs, dtype=np.float64) if self.obs else np.zeros((0, 10))
        fk = obs[:, 0].astype(np.int64)
        pid = obs[:, 1].astype(np.int32)
        x, y, sigma, idc = obs[:, 2], obs[:, 3], obs[:, 4], obs[:, 5]
        fabs = frame[fk]
        # smoothing and velocity per person, within unbroken runs (consecutive absolute frames, gaps <= RUN_GAP)
        sx, sy = x.copy(), y.copy()
        vx, vy = np.full(len(x), np.nan), np.full(len(x), np.nan)
        by_person = np.lexsort((fabs, pid))
        bounds = np.flatnonzero(np.diff(pid[by_person]))+1
        for idx in np.split(by_person, bounds):
            if not len(idx):
                continue
            fa = fabs[idx]
            run = np.concatenate(([0], np.cumsum(np.diff(fa) > RUN_GAP)))
            key = run*10_000_000+fa          # frames of different runs never fall in one window
            cx, cy = np.concatenate(([0.], np.cumsum(x[idx]))), np.concatenate(([0.], np.cumsum(y[idx])))
            lo = np.searchsorted(key, key-SMOOTH_HALF, side='left')
            hi = np.searchsorted(key, key+SMOOTH_HALF, side='right')
            cnt = hi-lo
            mx, my = (cx[hi]-cx[lo])/cnt, (cy[hi]-cy[lo])/cnt
            sx[idx], sy[idx] = mx, my
            a = np.searchsorted(key, key-VEL_HALF, side='left')
            b = np.searchsorted(key, key+VEL_HALF, side='right')-1
            span = fa[b]-fa[a]
            ok = (span >= VEL_HALF) & (run[a] == run) & (run[b] == run)
            dt = np.where(ok, span/fps, np.nan)
            vx[idx] = np.where(ok, (mx[b]-mx[a])/dt, np.nan)
            vy[idx] = np.where(ok, (my[b]-my[a])/dt, np.nan)
        # the grid: every GRID-th owned frame
        grid = np.arange(0, n, GRID)
        pos_in_grid = np.full(n, -1, np.int64)
        pos_in_grid[grid] = np.arange(len(grid))
        g = pos_in_grid[fk]
        keep = g >= 0
        rows = np.lexsort((pid[keep], g[keep]))
        gk = g[keep][rows]
        counts = np.bincount(gk, minlength=len(grid))
        arrays = {'obsPtr': np.concatenate(([0], np.cumsum(counts))).astype(np.int64),
                  'pid': pid[keep][rows].astype(np.int32),
                  'x': sx[keep][rows].astype(np.float32), 'y': sy[keep][rows].astype(np.float32),
                  'vx': vx[keep][rows].astype(np.float32), 'vy': vy[keep][rows].astype(np.float32),
                  'sigma': sigma[keep][rows].astype(np.float32), 'idConf': idc[keep][rows].astype(np.float32),
                  'box': obs[:, 6:10][keep][rows].astype(np.float32)}
        fr = [self.frames[i] for i in grid]
        t = tsec[grid]
        arrays['t'] = t
        arrays['frame'] = frame[grid]
        arrays['segment'] = np.array([f[2] for f in fr], dtype=np.int16)
        period = np.full(len(grid), -1, np.int8)
        for p in periods:
            period[(t >= float(p['start'])-1e-6) & (t <= float(p['end'])+1e-6)] = int(p['index'])
        arrays['period'] = period
        arrays['calibrated'] = np.array([f[3] for f in fr], dtype=bool)
        arrays['calibConf'] = np.array([f[4] for f in fr], dtype=np.float32)
        arrays['pitchLevel'] = np.array([f[5] for f in fr], dtype=np.int8)
        view = np.full((len(grid), 6, 2), np.nan, np.float32)
        for i, f in enumerate(fr):
            if f[6] is not None:
                pts = np.asarray(f[6], float)[:6]
                view[i, :len(pts)] = pts*[self.length, self.width]
                if len(pts) < 6:
                    view[i, len(pts):] = view[i, len(pts)-1]
        arrays['view'] = view
        arrays['H'] = np.array([self.H[i] if self.H[i] is not None else [np.nan]*9 for i in grid], dtype=np.float64)
        arrays['ball'] = np.array([f[7] if f[7] is not None else (np.nan, np.nan) for f in fr], dtype=np.float32)
        arrays['ballState'] = np.array([f[8] for f in fr], dtype=np.int8)
        arrays['ballConf'] = np.array([f[9] for f in fr], dtype=np.float32)
        arrays['ballPConf'] = np.array([f[10] for f in fr], dtype=np.float32)
        tables, runs = _possession_tables(possession, self.ids)
        arrays.update(_possession_samples(t, tables, runs))
        changes = []
        for c in (possession or {}).get('changes') or []:
            changes.append({k: c.get(k) for k in ('id', 'lostAt', 'wonAt', 'frame', 'fromTeam', 'toTeam', 'lostBy', 'wonBy',
                                                   'cause', 'confidence')})
        return arrays, runs, changes


def build_match(folder, progress=None):
    """MatchData of an assembled full match (runs/<video>/matches/<id>), from its views over the owned frames."""
    started = time.perf_counter()
    folder = Path(folder)
    asm = folder/'assembly'
    index = json.loads((asm/'match.json').read_text(encoding='utf-8'))
    possession = json.loads((asm/'possession.json').read_text(encoding='utf-8')) if (asm/'possession.json').is_file() else {}
    size = index.get('pitchSize') or {}
    length, width = float(size.get('length') or 105.0), float(size.get('width') or 68.0)
    roster = {r['matchId']: {'display': r.get('display'), 'team': r.get('team'), 'role': r.get('role')}
              for r in index.get('roster') or []}
    col = _Collector(length, width, roster)
    for seg in index['segments']:
        view = json.loads(gzip.decompress((asm/seg['view']).read_bytes()).decode('utf-8'))
        for f in view.get('frames') or []:
            if seg['cut0'] <= f['frame'] < seg['cut1']:
                col.add_frame(f, seg['index'])
        del view
        if progress:
            progress(seg['index']+1, len(index['segments']))
    fps = float(index['video']['fps'])
    periods = [{'index': p['index'], 'start': p['start'], 'end': p['end'], 'name': p.get('name'),
                'attacksRight': p.get('attacksRight'), 'directionAgreement': p.get('directionAgreement')}
               for p in index['periods']]
    arrays, runs, changes = col.finish(fps, periods, possession)
    meta = {'version': VERSION, 'kind': 'match', 'matchId': index.get('matchId'), 'length': length, 'width': width,
            'fps': fps, 'periods': periods, 'possRuns': runs, 'changes': changes,
            'videoSize': [int(index['video'].get('width') or 1920), int(index['video'].get('height') or 1080)],
            'videoName': index['video'].get('name'), 'sourceKey': (index.get('source') or {}).get('key'),
            'possession': {'version': possession.get('version'), 'quality': possession.get('quality'),
                           'shares': possession.get('shares'), 'engines': possession.get('engines')},
            'fingerprint': source_fingerprint(folder), 'evidence': dict(col.evidence),
            'buildSeconds': round(time.perf_counter()-started, 1)}
    return MatchData(arrays, col.people, meta)


def source_fingerprint(folder):
    """What a match's MatchData depends on: the assembled index, views and possession (their bytes' digests), this
    module's version. Corrections are applied on load and are not part of it."""
    asm = Path(folder)/'assembly'
    h = hashlib.sha256(f'moments-data-{VERSION}-{GRID}-{SMOOTH_HALF}-{VEL_HALF}-{MAX_SIGMA_M}-{MIN_PITCH_CONF}'.encode())
    index_bytes = (asm/'match.json').read_bytes()
    h.update(hashlib.sha256(index_bytes).digest())
    index = json.loads(index_bytes.decode('utf-8'))
    for seg in index['segments']:
        p = asm/seg['view']
        st = p.stat()
        h.update(f"{seg['view']}|{st.st_size}|{seg['cut0']}|{seg['cut1']}".encode())
    if (asm/'possession.json').is_file():
        h.update(hashlib.sha256((asm/'possession.json').read_bytes()).digest())
    return h.hexdigest()


# ---------------------------------------------------------------- cache
def save(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {k: getattr(data, k) for k in SAMPLE_FIELDS+OBS_FIELDS}
    arrays['__people__'] = np.frombuffer(json.dumps(data.people).encode('utf-8'), dtype=np.uint8)
    arrays['__meta__'] = np.frombuffer(json.dumps(data.meta).encode('utf-8'), dtype=np.uint8)
    temporary = path.with_name(f'.{path.name}.tmp.npz')
    np.savez_compressed(temporary, **arrays)
    for attempt in range(20):
        try:
            temporary.replace(path)
            return path
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(.05*(attempt+1))


def load(path):
    with np.load(path, allow_pickle=False) as z:
        arrays = {k: z[k] for k in SAMPLE_FIELDS+OBS_FIELDS}
        people = json.loads(bytes(z['__people__']).decode('utf-8'))
        meta = json.loads(bytes(z['__meta__']).decode('utf-8'))
    return MatchData(arrays, people, meta)


def match_data(folder, cache_dir=None, progress=None):
    """The match's MatchData from the cache (assembly/coaching/moments/data-<fp16>.npz), built when missing."""
    folder = Path(folder)
    cache_dir = Path(cache_dir) if cache_dir else folder/'assembly'/'coaching'/'moments'
    fp = source_fingerprint(folder)
    path = cache_dir/f'data-{fp[:16]}.npz'
    if path.is_file():
        try:
            data = load(path)
            if data.meta.get('fingerprint') == fp and data.meta.get('version') == VERSION:
                return data
        except (OSError, ValueError, KeyError):
            pass
    data = build_match(folder, progress=progress)
    save(data, path)
    return data


# ---------------------------------------------------------------- corrections
def apply_possession_v2(data, record, mode='agree'):
    """The match's possession v2 record (soccer/possession_v2.py) applied to the samples' possession, in place (after
    apply_corrections: its possessors are the match's people).

    mode 'agree' (what the key-moments report uses): a sample keeps its phase only where possession v2 names the same
    team as the segments' possession record, and is dead where either calls the ball dead (v2: a stoppage, halftime,
    footage that is not the match, a possible stoppage); elsewhere no team. Two independent estimates agreeing is the
    'sufficiently reliable' timing the coaching findings need: this can only take phases away, never add one, so no
    finding appears because of v2 alone (measured on CAN-ISL: the replace mode below changed 153 of 188 episodes).
    mode 'replace': v2's own phases ('control' where its team was seen on the ball, 'flight' where inferred; contested,
    free and unresolved time no team), for experiments only.
    The report's possession note reads the shares of the samples' phases either way."""
    tl = record.get('timeline') or {}
    people = tl.get('people') or []
    spans = []
    for t0, t1, code, des, conf, poss in zip(tl.get('t0') or [], tl.get('t1') or [], tl.get('state') or [],
                                            tl.get('designation') or [], tl.get('confidence') or [], tl.get('possessor') or []):
        team = 'A' if code == 0 else 'B' if code == 1 else None
        state = ('control' if des == 1 else 'flight') if team else ('dead' if code >= 5 else 'unknown')
        c = conf/100.0 if conf is not None and conf >= 0 else None
        spans.append({'t0': t0, 't1': t1, 'team': team, 'state': state,
                      'carrier': people[poss]['id'] if team and poss is not None and 0 <= poss < len(people) else None,
                      'confidence': 'high' if c is not None and c >= .95 else 'medium' if c is not None and c >= .8 else 'low'})
    tables, runs = _possession_tables({'timeline': spans}, data.index_of)
    v2 = _possession_samples(data.t, tables, runs)
    if mode == 'agree':
        dead = POSS_CODES['dead']
        old_team, old_state = data.possTeam.copy(), data.possState.copy()
        agree = (v2['possTeam'] == old_team) & (old_team != 0)
        either_dead = (old_state == dead) | (v2['possState'] == dead)
        team = np.where(agree & ~either_dead, old_team, 0).astype(np.int8)
        state = np.where(either_dead, dead, np.where(agree, old_state, 0)).astype(np.int8)
        data.possCarrier = np.where(team != 0, data.possCarrier, -1).astype(np.int32)
        data.possConf = np.where(team != 0, np.minimum(data.possConf, v2['possConf']), 0).astype(np.int8)
        data.possTeam, data.possState = team, state
        runs = _team_runs(data.t, team)
        settled = _possession_samples(data.t, (np.zeros(0),)*6, runs)
        data.possSettled, data.possRun = settled['possSettled'], settled['possRun']
    else:
        for k, v in v2.items():
            setattr(data, k, v)
    data.meta['possRuns'] = runs
    sec = (record.get('stats') or {}).get('match', {}).get('seconds') or {}
    data.meta['possession'] = {'version': record.get('version'), 'source': f'possession-v2 ({mode})',
                               'shares': {'A': sec.get('A', 0.0), 'B': sec.get('B', 0.0),
                                          'dead': sum(sec.get(k, 0.0) for k in ('possible_stoppage', 'stoppage', 'halftime', 'non_match')),
                                          'unknown': sum(sec.get(k, 0.0) for k in ('contested', 'free', 'unknown'))}}
    return data


def _team_runs(t, team):
    """[{'team', 't0', 't1'}] of consecutive samples naming one team (gaps up to POSS_JOIN_S joined)."""
    runs = []
    names = {v: k for k, v in TEAM_CODES.items()}
    for i in range(len(t)):
        code = int(team[i])
        if not code:
            continue
        if runs and runs[-1]['team'] == names[code] and float(t[i])-runs[-1]['t1'] <= POSS_JOIN_S+1e-9:
            runs[-1]['t1'] = float(t[i])
        else:
            runs.append({'team': names[code], 't0': float(t[i]), 't1': float(t[i])})
    return runs


def apply_corrections(data, aliases=None, excluded=None, teams=None):
    """The coach's roster corrections (fullmatch.corrections.resolve_aliases: merged id -> final id, excluded ids, team
    overrides) applied in place: merged people share one index, excluded people's rows are dropped, teams overridden.
    Two merged fragments never share a frame (the merge refuses it), so no sample gets one person twice."""
    aliases, excluded, teams = aliases or {}, set(excluded or ()), teams or {}
    if not (aliases or excluded or teams):
        return data
    final = {}
    people = []
    for i, p in enumerate(data.people):
        target = aliases.get(p['id'], p['id'])
        if target not in final:
            final[target] = len(people)
            base = next((q for q in data.people if q['id'] == target), p)
            people.append(dict(base, id=target))
    remap = np.array([final[aliases.get(p['id'], p['id'])] for p in data.people], dtype=np.int32)
    for pid_new, p in enumerate(people):
        if p['id'] in teams:
            p['team'] = teams[p['id']]
    drop = np.array([p['id'] in excluded or aliases.get(p['id'], p['id']) in excluded for p in data.people], dtype=bool)
    keep = ~drop[data.pid]
    sample_of = data.obs_sample()[keep]
    data.pid = remap[data.pid[keep]]
    for k in ('x', 'y', 'vx', 'vy', 'sigma', 'idConf', 'box'):
        setattr(data, k, getattr(data, k)[keep])
    counts = np.bincount(sample_of, minlength=len(data.t))
    data.obsPtr = np.concatenate(([0], np.cumsum(counts))).astype(np.int64)
    data.possCarrier = np.where(data.possCarrier >= 0, remap[np.clip(data.possCarrier, 0, None)], -1).astype(np.int32)
    data.people = people
    data.index_of = {p['id']: i for i, p in enumerate(people)}
    data.team_of = np.array([TEAM_CODES.get(p.get('team'), 0) for p in people], dtype=np.int8)
    data.is_keeper = np.array([p.get('role') == 'GOALKEEPER' for p in people], dtype=bool)
    return data


def footprint_contains(view, px, py):
    """Whether the points (px, py) lie inside the camera footprint `view` (6 x 2, metres; NaN = unknown -> False)."""
    if view is None or not np.all(np.isfinite(view)):
        return np.zeros(np.shape(px), bool)
    from matplotlib.path import Path as MPath
    return MPath(view).contains_points(np.column_stack([np.ravel(px), np.ravel(py)])).reshape(np.shape(px))


def view_share(view, length, width, step=3.0):
    """The share of the pitch inside the camera footprint (a grid of points every `step` metres)."""
    if view is None or not np.all(np.isfinite(view)):
        return 0.0
    gx, gy = np.meshgrid(np.arange(step/2, length, step), np.arange(step/2, width, step))
    return float(footprint_contains(view, gx, gy).mean())


def sha_of_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode('utf-8')).hexdigest()


def round_time(t):
    return None if t is None or not math.isfinite(t) else round(float(t), 3)
