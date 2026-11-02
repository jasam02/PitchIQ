"""Tactical roles per player and time window, from where teammates stand relative to each other.

The camera shows part of the pitch, so a team's centroid at one instant is biased towards what is in view. Relative
positions are not: for every pair of teammates seen together, the difference of their depths (and of their lateral
positions) is a direct measurement. Over a window (WINDOW_S, every STEP_S) the pair differences are averaged and each
player's relative depth z and relative lateral position w are solved by least squares on the co-presence graph
(sum z = 0); a player seen with others only part of the time still gets a place.

The window uses the samples that show the team's DEFENSIVE organisation (the opponents had the ball or nobody was named,
never a dead ball) when there are enough of them: a full back pushed on in possession is still the left back. When a
window has too few such samples it uses every live sample.

Lines from z (deepest first): 3-5 defenders, then midfielders and 1-3 forwards, the split with the smallest within-line
spread (natural breaks); sides from w within each line (left = positive w). Names:
    defence of 3  LCB CB RCB         of 4  LB LCB RCB RB         of 5  LWB LCB CB RCB RWB
    midfield      LM / RM for the widest players 10 m or more off the line's middle (LWB / RWB behind a back three),
                  DM for a central midfielder 3 m or more deeper than the other central ones, else CM
    attack        ST, or LS RS, or LW ST RW
Confidence (high / medium / low): the lower of presence (the share of the window's samples the player was in) and the
margin to the nearest line boundary (metres between the player's depth and the middle of the gap to the next line).
Roles are an inference from positions: the report states role-based criticism more softly at medium and drops it at low.
"""
from collections import defaultdict

import numpy as np

from . import context as cx

WINDOW_S = 60.0
STEP_S = 20.0
MIN_SAMPLES = 50          # informative samples in a window (10 s at 5 Hz)
MIN_PRESENCE = .3         # share of a window's samples a player must be in to get a role there
MIN_PLAYERS = 8           # players with a role needed to split the lines
NEAR_S = 40.0             # a role applies up to this far from its window's centre
DEFENDING_NEAR_S = 90.0   # a window of the defensive organisation is preferred up to this far away
WIDE_M = 10.0             # a midfielder this far off the line's middle is a wide one
DM_M = 3.0
DEF_NAMES = {3: ('LCB', 'CB', 'RCB'), 4: ('LB', 'LCB', 'RCB', 'RB'), 5: ('LWB', 'LCB', 'CB', 'RCB', 'RWB')}
FWD_NAMES = {1: ('ST',), 2: ('LS', 'RS'), 3: ('LW', 'ST', 'RW')}
FULL_BACKS = ('LB', 'RB', 'LWB', 'RWB')
ROLE_WORDS = {'LB': 'left back', 'RB': 'right back', 'LWB': 'left wing-back', 'RWB': 'right wing-back',
              'LCB': 'left centre-back', 'CB': 'centre-back', 'RCB': 'right centre-back', 'DM': 'holding midfielder',
              'CM': 'central midfielder', 'LM': 'left midfielder', 'RM': 'right midfielder', 'ST': 'striker',
              'LS': 'left striker', 'RS': 'right striker', 'LW': 'left winger', 'RW': 'right winger'}


def _breaks(z):
    """(a, c): the deepest a are defenders, the last c forwards; the split of sorted z with the smallest within-line sum
    of squares, 3 <= a <= 5, 1 <= c <= 3, at least 2 midfielders."""
    n = len(z)
    best = None
    for a in range(3, 6):
        for c in range(1, 4):
            m = n-a-c
            if m < 2:
                continue
            parts = (z[:a], z[a:a+m], z[a+m:])
            cost = sum(float(((p-p.mean())**2).sum()) for p in parts)
            if best is None or cost < best[0]:
                best = (cost, a, c)
    return (best[1], best[2]) if best else (None, None)


def _level(value, high, medium):
    return 'high' if value >= high else 'medium' if value >= medium else 'low'


def _lower(a, b):
    order = {'low': 0, 'medium': 1, 'high': 2}
    return a if order[a] <= order[b] else b


def window_roles(data, ctx, samples):
    """{pid: role record} for one window's samples (indices), or {} when the lines cannot be split."""
    if len(samples) < MIN_SAMPLES:
        return {}
    local = {}
    pairs_d, pairs_l, pairs_n = [], [], []
    seen = defaultdict(int)
    obs = []
    for s in samples:
        idx, pids = ctx.players(int(s))
        if len(idx) < 3:
            continue
        d = ctx.depth(data.x[idx], int(s))
        lat = ctx.lateral(data.y[idx], int(s))
        obs.append((pids, d, lat))
        for p in pids:
            seen[int(p)] += 1
    used = len(obs)
    if used < MIN_SAMPLES:
        return {}
    keep = [p for p, k in seen.items() if k >= MIN_PRESENCE*used]
    if len(keep) < MIN_PLAYERS:
        return {}
    for i, p in enumerate(sorted(keep)):
        local[p] = i
    n = len(local)
    Sd, Sl, N = np.zeros((n, n)), np.zeros((n, n)), np.zeros((n, n))
    for pids, d, lat in obs:
        sel = np.array([local.get(int(p), -1) for p in pids])
        ok = sel >= 0
        if ok.sum() < 2:
            continue
        k, dd, ll = sel[ok], d[ok], lat[ok]
        Sd[np.ix_(k, k)] += dd[:, None]-dd[None, :]
        Sl[np.ix_(k, k)] += ll[:, None]-ll[None, :]
        N[np.ix_(k, k)] += 1
    np.fill_diagonal(N, 0)
    W = N
    lap = np.diag(W.sum(1))-W
    z = np.linalg.lstsq(lap, Sd.sum(1), rcond=None)[0]
    w = np.linalg.lstsq(lap, Sl.sum(1), rcond=None)[0]
    pids = np.array(sorted(local, key=local.get))
    order = np.argsort(z)
    zs = z[order]
    a, c = _breaks(zs)
    if a is None:
        return {}
    m = n-a-c
    lines = ['D']*a+['M']*m+['F']*c
    gaps = {('D', 'M'): (zs[a-1]+zs[a])/2, ('M', 'F'): (zs[a+m-1]+zs[a+m])/2}
    out = {}
    groups = {'D': [], 'M': [], 'F': []}
    for rank, i in enumerate(order):
        groups[lines[rank]].append(i)
    names = {}
    d_sorted = sorted(groups['D'], key=lambda i: -w[i])
    for i, name in zip(d_sorted, DEF_NAMES[len(d_sorted)]):
        names[i] = name
    mids = groups['M']
    if mids:
        mw = np.median([w[i] for i in mids])
        wide_l = max(mids, key=lambda i: w[i])
        wide_r = min(mids, key=lambda i: w[i])
        back_three = len(groups['D']) == 3
        for i in mids:
            names[i] = 'CM'
        if w[wide_l]-mw >= WIDE_M:
            names[wide_l] = 'LWB' if back_three else 'LM'
        if mw-w[wide_r] >= WIDE_M and wide_r != wide_l:
            names[wide_r] = 'RWB' if back_three else 'RM'
        central = [i for i in mids if names[i] == 'CM']
        if len(central) >= 2:
            deep = min(central, key=lambda i: z[i])
            others = [z[i] for i in central if i != deep]
            if min(others)-z[deep] >= DM_M:
                names[deep] = 'DM'
    f_sorted = sorted(groups['F'], key=lambda i: -w[i])
    for i, name in zip(f_sorted, FWD_NAMES[len(f_sorted)]):
        names[i] = name
    for rank, i in enumerate(order):
        line = lines[rank]
        bounds = []
        if line in ('D', 'M'):
            bounds.append(gaps[('D', 'M')] if line == 'D' else gaps[('D', 'M')])
        if line in ('M', 'F'):
            bounds.append(gaps[('M', 'F')])
        margin = min(abs(z[i]-b) for b in bounds)
        presence = seen[int(pids[i])]/used
        conf = _lower(_level(margin, 4.0, 2.0), _level(presence, .6, MIN_PRESENCE))
        out[int(pids[i])] = {'role': names[i], 'line': line, 'z': round(float(z[i]), 2), 'w': round(float(w[i]), 2),
                             'presence': round(presence, 2), 'margin': round(float(margin), 2), 'confidence': conf}
    out['__shape__'] = {'defenders': len(groups['D']), 'midfielders': m, 'forwards': c, 'players': n, 'samples': used}
    return out


class Roles:
    """Role records of one team over the match: windows every STEP_S, looked up by person and time."""

    def __init__(self, data, ctx):
        self.data, self.ctx = data, ctx
        self.windows = []                     # (t_centre, {pid: record}, shape)
        self.by_pid = defaultdict(list)       # pid -> [(t_centre, record)]
        for p in data.periods:
            t = float(p['start'])+WINDOW_S/2
            end = float(p['end'])
            while t-WINDOW_S/2 < end:
                a = np.searchsorted(data.t, t-WINDOW_S/2)
                b = np.searchsorted(data.t, t+WINDOW_S/2)
                samples = np.arange(a, b)
                live = samples[(ctx.phase[samples] != cx.DEAD) & data.calibrated[samples]]
                defending = live[ctx.phase[live] != cx.OWN]
                chosen, basis = (defending, 'defending') if len(defending) >= MIN_SAMPLES else (live, 'live')
                recs = window_roles(data, ctx, chosen)
                if recs:
                    shape = recs.pop('__shape__')
                    shape['basis'] = basis
                    self.windows.append((t, recs, shape))
                    for pid, rec in recs.items():
                        rec['basis'] = basis
                        self.by_pid[pid].append((t, rec))
                t += STEP_S
        for pid, items in self.by_pid.items():
            names = [r['role'] for _, r in items]
            for _, r in items:
                r['stability'] = round(names.count(r['role'])/len(names), 2)

    def role_at(self, pid, t):
        """The role record of person `pid` at time t: the nearest window within DEFENDING_NEAR_S that saw the team's
        defensive organisation, else the nearest window within NEAR_S. A window of the team's own possession places a
        full back who pushed on among the midfielders (on a synthetic scene a left midfielder became a 'wing-back' and
        a striker who dropped deep a 'midfielder'), so it is only the fallback."""
        items = self.by_pid.get(int(pid))
        if not items:
            return None
        defending = [it for it in items if it[1].get('basis') == 'defending' and abs(it[0]-t) <= DEFENDING_NEAR_S]
        if defending:
            return min(defending, key=lambda it: abs(it[0]-t))[1]
        best = min(items, key=lambda it: abs(it[0]-t))
        return best[1] if abs(best[0]-t) <= NEAR_S else None

    def shape_at(self, t):
        if not self.windows:
            return None
        best = min(self.windows, key=lambda w: abs(w[0]-t))
        return best[2] if abs(best[0]-t) <= NEAR_S else None

    def shapes(self):
        """Counts of (defenders, midfielders, forwards) over the windows that saw 10 outfield players."""
        out = defaultdict(int)
        for _, _, shape in self.windows:
            if shape['players'] >= 10:
                out[f"{shape['defenders']}-{shape['midfielders']}-{shape['forwards']}"] += 1
        return dict(out)


def role_word(role):
    return ROLE_WORDS.get(role, 'player')
