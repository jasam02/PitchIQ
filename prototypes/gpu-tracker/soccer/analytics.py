"""Player analytics from a finished result: heatmaps, average positions and movement trails, computed from the
finalized global identities and their observed pitch positions. Nothing here runs a model or reads the video.

Pipeline: result.json -> extract(): per identity (the final roster: team players and goalkeepers with a confirmed
slot, so corrections, exclusions and relabelling apply) the OBSERVED samples with a valid, trusted pitch position,
each carrying the seconds of video it stands for -> select a player and a time range -> heatmap(), average(),
averages(), trail(), coverage() -> JSON the page renders on the same pitch as the 2D match view.

Conventions, shared with tactical.js: positions are normalized pitch coordinates of the feet, x 0..1 from the left
goal line to the right goal line, y 0..1 from the far touchline to the near touchline, on a 105 x 68 m pitch;
observed samples are smoothed with the same centred 0.14 s window the 2D view uses, so the trail runs through the
same points the board shows. Time is video time: a sample stands for half the distance to its neighbours on each
side, never more than SAMPLE_CAP/2 seconds a side, so denser sampling never means more occupancy and a long absence
is never counted as standing still at the last known place. Coverage is the union of the seconds the samples stand
for over the selected interval, not a frame count.

Excluded, never clamped onto the pitch: positions off the pitch (beyond OUTSIDE), a registration below
MIN_PITCH_CONFIDENCE, a person whose box is cut by the frame edge (the feet unseen), an identity below
MIN_IDENTITY_CONFIDENCE or not confirmed on that frame, anyone who is not a confirmed team player or goalkeeper of
the final roster (referees, staff, excluded, retired, unconfirmed A-? identities), and every estimate (the timeline's
interpolated and predicted positions are not observations).
"""
import math
from bisect import bisect_left, bisect_right

import numpy as np

LENGTH, WIDTH = 105.0, 68.0          # metres; the registration's size is used when the result carries one
OUTSIDE = .02                        # normalized: a position this far beyond a line is still the pitch (a foot on the touchline)
MIN_PITCH_CONFIDENCE = .5            # the registration's confidence at the sample, times its local projection certainty
MIN_IDENTITY_CONFIDENCE = .3
SMOOTH_WINDOW = .14                  # s: the 2D view's centred window over observed samples
SAMPLE_CAP = .5                      # s: a gap between consecutive samples up to this is covered (half to each); of a longer gap, a
                                     # sample stands for only SAMPLE_CAP/2: the rest is unseen time, not occupancy
HEAT_CELL = 1.0                      # m: heatmap grid cell
HEAT_SIGMA = 2.5                     # m: Gaussian smoothing of the occupancy grid
TRAIL_GAP = 1.0                      # s: consecutive samples further apart than this are not joined (the 2D view's interpolateGap)
TRAIL_MAX_SPEED = 12.0               # m/s: a faster step between consecutive samples, over at least JUMP_MIN metres, is a registration
JUMP_MIN = 2.0                       # m: glitch, not a run (a far player's position jitters by half a metre a frame: that is not a jump)
LOW_COVERAGE = .5                    # coverage below this is flagged
CURRENT_WINDOW = 300.0               # s: "current 5 minutes" is the five-minute block the playhead is in


class Samples:
    """The observed samples of every analysable identity of a result."""

    def __init__(self, result, min_pitch_confidence=MIN_PITCH_CONFIDENCE, min_identity_confidence=MIN_IDENTITY_CONFIDENCE):
        frames = result.get('frames') or []
        video = result.get('video') or {}
        size = ((result.get('tactical') or {}).get('pitchSize') or {})
        self.length = float(size.get('length') or LENGTH)
        self.width = float(size.get('width') or WIDTH)
        self.fps = float(video.get('fps') or 0.0) or 25.0
        self.cap = max(SAMPLE_CAP, 2.5/self.fps)
        self.start = frames[0]['time'] if frames else 0.0
        self.end = frames[-1]['time'] if frames else 0.0
        self.people = roster(result)
        self.excluded = {'offPitch': 0, 'lowPitchConfidence': 0, 'clipped': 0, 'lowIdentityConfidence': 0, 'notConfirmed': 0, 'notInRoster': 0}
        raw = {pid: ([], [], []) for pid in self.people}
        for f in frames:
            t = float(f['time'])
            seen = set()
            for p in f.get('people') or []:
                pid = p.get('id')
                if not pid:
                    continue
                if pid not in raw:
                    self.excluded['notInRoster'] += 1
                    continue
                if p.get('state') != 'confirmed':
                    self.excluded['notConfirmed'] += 1
                    continue
                if (p.get('identityConfidence') or 0.0) < min_identity_confidence:
                    self.excluded['lowIdentityConfidence'] += 1
                    continue
                pos = p.get('pitch')
                if not _valid(pos):
                    self.excluded['offPitch'] += 1
                    continue
                if (p.get('pitchConfidence') or 0.0) < min_pitch_confidence:
                    self.excluded['lowPitchConfidence'] += 1
                    continue
                if p.get('pitchClipped'):
                    self.excluded['clipped'] += 1
                    continue
                if pid in seen:
                    continue   # one identity, one position per frame
                seen.add(pid)
                ts, xs, ys = raw[pid]
                ts.append(t)
                xs.append(float(pos[0]))
                ys.append(float(pos[1]))
        self.tracks = {}
        for pid, (ts, xs, ys) in raw.items():
            if not ts:
                continue
            t = np.array(ts, np.float64)
            order = np.argsort(t, kind='stable')
            t, x, y = t[order], np.array(xs, np.float64)[order], np.array(ys, np.float64)[order]
            # Runs of continuous observation: a break where the samples are further apart than TRAIL_GAP or where the
            # step is faster than any player (a registration glitch). Smoothing never bleeds across a break.
            why = _breaks(t, x, y, self.length, self.width)
            run = np.cumsum(why != '')
            sx, sy = x.copy(), y.copy()
            for k in np.unique(run):
                m = run == k
                sx[m], sy[m] = _smooth(t[m], x[m], y[m], SMOOTH_WINDOW)
            self.tracks[pid] = {'t': t, 'x': sx, 'y': sy, 'w': _weights(t, self.cap, 1/self.fps), 'run': run, 'why': why}

    def window(self, pid, t0, t1):
        """The samples of an identity within [t0, t1]: (t, x, y, w, run, why) arrays, empty when none."""
        tr = self.tracks.get(pid)
        if tr is None:
            return tuple(np.zeros(0) for _ in range(4))+(np.zeros(0, int), np.zeros(0, dtype=object))
        a, b = bisect_left(tr['t'].tolist(), t0), bisect_right(tr['t'].tolist(), t1)
        return tr['t'][a:b], tr['x'][a:b], tr['y'][a:b], tr['w'][a:b], tr['run'][a:b], tr['why'][a:b]


def _breaks(t, x, y, length, width, gap=TRAIL_GAP, max_speed=TRAIL_MAX_SPEED, jump_min=JUMP_MIN):
    """Per sample, why a break precedes it ('' when none): 'unseen' when the previous sample is more than `gap` seconds
    earlier, 'jump' when the step from it is faster than `max_speed` over at least `jump_min` metres."""
    why = np.full(len(t), '', dtype=object)
    if len(t) > 1:
        dt = np.diff(t)
        metres = np.hypot(np.diff(x)*length, np.diff(y)*width)
        speed = np.divide(metres, dt, out=np.zeros_like(metres), where=dt > 0)
        why[1:][dt > gap] = 'unseen'
        why[1:][(dt <= gap) & (speed > max_speed) & (metres >= jump_min)] = 'jump'
    return why


def roster(result):
    """{id: {'id', 'display', 'team', 'role'}} of the identities the analytics describe: the final roster's team players
    and goalkeepers of team A or B with a confirmed slot. Everything else (referees, excluded, retired, unconfirmed
    A-? identities) is left out, so corrections apply."""
    match = result.get('match') or {}
    out = {}
    for key, role in (('players', 'PLAYER'), ('goalkeepers', 'GOALKEEPER')):
        for e in match.get(key) or []:
            pid = e.get('id')
            if not pid or e.get('team') not in ('A', 'B') or e.get('confirmed') is False:
                continue
            out[pid] = {'id': pid, 'display': e.get('display') or pid, 'team': e['team'], 'role': role}
    return out


def _valid(p):
    return isinstance(p, (list, tuple)) and len(p) == 2 and all(isinstance(v, (int, float)) and math.isfinite(v) for v in p) \
        and -OUTSIDE <= p[0] <= 1+OUTSIDE and -OUTSIDE <= p[1] <= 1+OUTSIDE


def _smooth(t, x, y, window):
    """The 2D view's smoothing: every sample becomes the mean of the samples within +-window/2 of it."""
    if len(t) < 2:
        return x, y
    lo = np.searchsorted(t, t-window/2, side='left')
    hi = np.searchsorted(t, t+window/2, side='right')
    cx, cy = np.concatenate([[0.0], np.cumsum(x)]), np.concatenate([[0.0], np.cumsum(y)])
    n = hi-lo
    return (cx[hi]-cx[lo])/n, (cy[hi]-cy[lo])/n


def _weights(t, cap, step):
    """Seconds each sample stands for: half the distance to each neighbour, at most cap/2 a side, and half a frame step
    on an open side (before the first sample, after the last). Weights of consecutive samples never overlap, so their
    sum is the covered time, whatever the sampling rate."""
    n = len(t)
    if n == 0:
        return np.zeros(0)
    w = np.zeros(n)
    if n > 1:
        half = np.minimum(np.diff(t), cap)/2
        w[:-1] += half
        w[1:] += half
    w[0] += step/2
    w[-1] += step/2
    return w


def coverage(samples, pid, t0, t1):
    """Observed seconds (the union of what the samples stand for, clipped to the interval), the interval, its share,
    and the sample count."""
    t, x, y, w, _, _ = samples.window(pid, t0, t1)
    interval = max(0.0, t1-t0)
    seconds = float(min(w.sum(), interval)) if len(w) else 0.0
    share = seconds/interval if interval > 0 else 0.0
    return {'observedSeconds': round(seconds, 2), 'intervalSeconds': round(interval, 2), 'coverage': round(min(1.0, share), 3),
            'samples': int(len(t)), 'low': bool(len(t)) and share < LOW_COVERAGE}


def heatmap(samples, pid, t0, t1, cell=HEAT_CELL, sigma=HEAT_SIGMA):
    """Observed time per cell of a metres grid over the pitch, Gaussian-smoothed (seconds per cell; the total stays the
    observed time). {'cols', 'rows', 'cell', 'sigma', 'max', 'seconds', 'grid' (rows of cols, far touchline first)}."""
    t, x, y, w, _, _ = samples.window(pid, t0, t1)
    cols, rows = int(math.ceil(samples.length/cell)), int(math.ceil(samples.width/cell))
    grid = np.zeros((rows, cols))
    if len(t):
        cx = np.clip((x*samples.length/cell).astype(int), 0, cols-1)   # samples are already on the pitch (OUTSIDE at most)
        cy = np.clip((y*samples.width/cell).astype(int), 0, rows-1)
        np.add.at(grid, (cy, cx), w)
        grid = _blur(grid, sigma/cell)
    return {'cols': cols, 'rows': rows, 'cell': cell, 'sigma': sigma, 'max': round(float(grid.max()), 4) if len(t) else 0.0,
            'seconds': round(float(w.sum()), 2), 'grid': [[round(float(v), 4) for v in row] for row in grid]}


def _blur(grid, sigma_cells):
    if sigma_cells <= 0:
        return grid
    r = int(math.ceil(3*sigma_cells))
    k = np.exp(-.5*(np.arange(-r, r+1)/sigma_cells)**2)
    k /= k.sum()
    out = np.apply_along_axis(lambda v: np.convolve(v, k, mode='same'), 1, grid)
    out = np.apply_along_axis(lambda v: np.convolve(v, k, mode='same'), 0, out)
    total, kept = grid.sum(), out.sum()
    return out*(total/kept) if kept > 0 else out   # what the kernel pushed past the edges comes back: the total is the observed time


def average(samples, pid, t0, t1):
    """The time-weighted mean of the observed positions in the interval, with its coverage; None when there is none."""
    t, x, y, w, _, _ = samples.window(pid, t0, t1)
    cov = coverage(samples, pid, t0, t1)
    info = samples.people.get(pid) or {'id': pid, 'display': pid, 'team': None, 'role': 'PLAYER'}
    if not len(t) or w.sum() <= 0:
        return dict(info, position=None, **cov)
    return dict(info, position=[round(float((x*w).sum()/w.sum()), 4), round(float((y*w).sum()/w.sum()), 4)], **cov)


def averages(samples, t0, t1, teams=('A', 'B')):
    """Average positions of every analysable identity of the given teams in the interval (those with samples)."""
    out = []
    for pid, info in samples.people.items():
        if info['team'] not in teams:
            continue
        a = average(samples, pid, t0, t1)
        if a['position'] is not None:
            out.append(a)
    out.sort(key=lambda a: (a['team'], a['role'] != 'PLAYER', a['display']))
    return out


def trail(samples, pid, t0, t1):
    """The observed path in the interval as segments of consecutive samples joined only across gaps of at most TRAIL_GAP
    seconds and steps slower than TRAIL_MAX_SPEED; the start and end samples; the breaks (with their seconds)."""
    t, x, y, w, run, why = samples.window(pid, t0, t1)
    segments, breaks = [], []
    if len(t):
        cut = [0]
        for i in range(1, len(t)):
            if run[i] != run[i-1]:
                cut.append(i)
                breaks.append({'from': round(float(t[i-1]), 3), 'to': round(float(t[i]), 3), 'seconds': round(float(t[i]-t[i-1]), 2), 'why': str(why[i])})
        cut.append(len(t))
        for a, b in zip(cut[:-1], cut[1:]):
            segments.append({'t': [round(float(v), 3) for v in t[a:b]], 'x': [round(float(v), 4) for v in x[a:b]], 'y': [round(float(v), 4) for v in y[a:b]]})
    point = lambda i: {'t': round(float(t[i]), 3), 'x': round(float(x[i]), 4), 'y': round(float(y[i]), 4)}
    return {'segments': segments, 'start': point(0) if len(t) else None, 'end': point(len(t)-1) if len(t) else None,
            'breaks': breaks, 'unseenSeconds': round(float(sum(b['seconds'] for b in breaks)), 2),
            'metres': round(float(sum(_length(s, samples) for s in segments)), 1)}


def _length(segment, samples):
    xs, ys = np.array(segment['x'])*samples.length, np.array(segment['y'])*samples.width
    return float(np.hypot(np.diff(xs), np.diff(ys)).sum()) if len(xs) > 1 else 0.0


def clamp_range(samples, t0=None, t1=None):
    """The requested interval clipped to the processed segment; the whole segment when not given."""
    clip = lambda v: min(samples.end, max(samples.start, float(v)))
    a = samples.start if t0 is None else clip(t0)
    b = samples.end if t1 is None else clip(t1)
    return (a, b) if b >= a else (a, a)


def current_block(samples, at, window=CURRENT_WINDOW):
    """The five-minute block of the segment that contains the playhead."""
    k = math.floor(max(0.0, float(at))/window)
    return clamp_range(samples, k*window, (k+1)*window)


def report(samples, pid, t0, t1, scope='selected'):
    """Everything the page needs for one selection: coverage, heatmap, average (plus the scope's averages) and trail of
    the player, over [t0, t1]."""
    teams = {'selected': (), 'A': ('A',), 'B': ('B',), 'both': ('A', 'B')}.get(scope, ())
    info = samples.people.get(pid)
    out = {'player': info, 'range': {'t0': round(t0, 3), 't1': round(t1, 3), 'segment': [round(samples.start, 3), round(samples.end, 3)]},
           'rules': {'minPitchConfidence': MIN_PITCH_CONFIDENCE, 'minIdentityConfidence': MIN_IDENTITY_CONFIDENCE, 'sampleCap': samples.cap,
                     'trailGap': TRAIL_GAP, 'trailMaxSpeed': TRAIL_MAX_SPEED, 'heatSigma': HEAT_SIGMA, 'lowCoverage': LOW_COVERAGE, 'smoothWindow': SMOOTH_WINDOW},
           'excluded': samples.excluded, 'pitch': {'length': samples.length, 'width': samples.width}}
    if info is not None:
        out['coverage'] = coverage(samples, pid, t0, t1)
        out['heatmap'] = heatmap(samples, pid, t0, t1)
        out['average'] = average(samples, pid, t0, t1)
        out['trail'] = trail(samples, pid, t0, t1)
    out['averages'] = averages(samples, t0, t1, teams) if teams else ([out['average']] if info is not None and out['average']['position'] else [])
    return out
