"""Team shape analytics from a finished result: team width and depth, defensive line height and formation recognition,
computed from the finalized global identities and their observed pitch positions. Nothing here runs a model or reads
the video, and nothing here changes a position: the measurements are built on the same validated samples as the player
analytics (soccer/analytics.py), so corrections, exclusions and relabelling apply to both.

Pipeline (one TeamShapeData per result and position mode, shared by the three analyses):
    result.json -> analytics.Samples (observed, validated, smoothed samples per final-roster identity)
      -> a regular sample grid (SAMPLE_RATE per second, real frames, each carrying the seconds it stands for)
      -> per team: outfield players and goalkeepers apart; registration glitches, uncertain positions and duplicate
         identities removed; the lineup per sample; coverage per sample (measured outfield players against the lineup)
      -> width / depth per sample (+ aggregates)              team_width_depth()
      -> attacking direction per sample                        team_directions()
      -> formation windows, stabilized periods, roles          formations()
      -> defensive unit from the roles, line height            defensive_line()
      -> one JSON report per team and range                    TeamShape.report()

Coordinates are the result's normalized pitch coordinates: x 0..1 from the left goal line to the right goal line (the
goal-to-goal axis, so depth is measured along x), y 0..1 from the far touchline to the near touchline (the lateral axis,
so width is measured along y). Metres use the pitch size the registration reports; when that size is the standard
assumption (tactical.pitchSize.source == 'standard') every metre value is flagged as estimated, and the normalized
share of the pitch is reported next to it.

Positions: OBSERVED by default (the samples of analytics.Samples). The 'estimated' mode adds the timeline's
interpolated and predicted positions (kinds I and P, sigma at most ESTIMATE_MAX_SIGMA metres), always labelled, never
silently mixed: every per-sample value carries how many of its players were observed and how many estimated, and a
value mostly made of estimates is never called complete or reliable. Formation recognition never uses estimated
positions (the estimate follows the team, so it would confirm the shape it came from).
"""
import bisect
import math
from collections import Counter

import numpy as np
from scipy.optimize import linear_sum_assignment

from . import analytics

SAMPLE_RATE = 5.0             # samples per second evaluated (each a real frame; the rest of the frames add nothing to a shape)
SAMPLE_CAP = 1.0              # s: a sample stands for half the gap to each neighbour, at most SAMPLE_CAP/2 a side
MAX_SIGMA = 3.0               # m: an observed position less certain than this is not a tactical measurement
SPIKE_SECONDS = .5            # s: a run of samples shorter than this next to a registration jump is a glitch (see _fill)
DUPLICATE_METRES = .4         # m: two identities of one team this close ...
DUPLICATE_SAMPLES = 3         # ... on this many consecutive samples are one person seen twice; the less certain goes
ESTIMATE_MAX_SIGMA = 8.0      # m: estimated positions (estimated mode only) at most this uncertain, as the 2D view draws them
EDGE_MARGIN = 1.0             # m: a missing player's estimate this far beyond an observed extreme may be the real extreme
RELIABLE_MIN = 8              # observed outfield players for a full-team width / depth, subject to the edge check
PARTIAL_MIN = 6               # ... for a partial (observed-spread) value; below it nothing is reported
ROBUST_PERCENTILES = (10, 90) # the separately labelled central-spread measurement
DIRECTION_WINDOW = 30.0       # s: evidence window for the attacking direction (goalkeeper side, goal-side ordering of the teams)
DIRECTION_MIN_GK_SECONDS = 3.0
DIRECTION_MIN_ORDER_SECONDS = 5.0


def _round(v, n=2):
    return None if v is None or not np.isfinite(v) else round(float(v), n)


def _number(v):
    """A finite real number (not a bool); an integer too large for a float is not one."""
    if not isinstance(v, (int, float)) or isinstance(v, bool):
        return False
    try:
        return math.isfinite(v)
    except OverflowError:
        return False


def reliable_min(expected):
    """Observed outfield players for a reliable (full-team) width or depth: RELIABLE_MIN, or all but one of a smaller
    lineup (an 8-a-side team), never fewer than PARTIAL_MIN."""
    return min(RELIABLE_MIN, max(PARTIAL_MIN, expected-1))


def lineup_of(setting, default):
    """The outfield lineup a setting describes: None (the default for the whole video), a number of outfield players,
    or {'outfield': n, 'from': seconds} (n from that time on, e.g. after a dismissal). Returns (before, after, from)."""
    if isinstance(setting, int) and not isinstance(setting, bool):
        return setting, setting, None
    if isinstance(setting, dict) and isinstance(setting.get('outfield'), int):
        start = setting.get('from')
        if _number(start):
            return default, int(setting['outfield']), float(start)
        return int(setting['outfield']), int(setting['outfield']), None
    return default, default, None


# ---------- the shared team shape data ----------
class TeamShapeData:
    """The validated outfield and goalkeeper positions of both teams on a regular grid of real frames.

    Per team: `ids` (outfield identities of the final roster), arrays (S x N) `x`, `y` (normalized, NaN when the player
    has no usable position on that sample), `kind` ('O' observed, 'E' estimated, '' none), `identity`, `quality`
    (pitch confidence), `sigma` (m); `gk` (S x 2, NaN when no goalkeeper of the team is observed); `expected_t` (outfield
    players on the pitch per sample) and `expected` (at the start); `active` (S x N: the slot can be on the pitch);
    `excluded` (reasons, counted per sample)."""

    def __init__(self, result, samples=None, mode='observed', rate=SAMPLE_RATE, lineup=None):
        if mode not in ('observed', 'estimated'):
            raise ValueError('mode must be observed or estimated')
        self.mode = mode
        self.samples = samples if samples is not None else analytics.Samples(result)
        self.length, self.width = self.samples.length, self.samples.width
        size = (result.get('tactical') or {}).get('pitchSize') or {}
        self.size_source = size.get('source') or 'standard'
        self.metres_estimated = self.size_source == 'standard'
        frames = result.get('frames') or []
        times = np.array([float(f['time']) for f in frames], np.float64)
        self.fps = self.samples.fps
        if len(times):
            bins = np.floor((times-times[0])*rate+1e-9).astype(np.int64)
            keep = np.r_[True, bins[1:] != bins[:-1]]
        else:
            keep = np.zeros(0, bool)
        self.t = times[keep]
        self.frame_index = np.flatnonzero(keep)
        self.w = analytics._weights(self.t, SAMPLE_CAP, 1/rate) if len(self.t) else np.zeros(0)
        self.rate = rate
        self.start = float(times[0]) if len(times) else 0.0
        self.end = float(times[-1]) if len(times) else 0.0
        roster = self.samples.people
        expected = ((result.get('roster') or {}).get('expected') or {})
        slots = {s.get('id'): s for s in (result.get('roster') or {}).get('slots') or []}
        block = result.get('timeline') or {}
        block_ok = bool(block.get('ids')) and len(block.get('frameTimes') or []) == len(frames)
        lineup = lineup or {}
        self.teams = {}
        for team in ('A', 'B'):
            ids = sorted(pid for pid, info in roster.items() if info['team'] == team and info['role'] == 'PLAYER')
            gks = sorted(pid for pid, info in roster.items() if info['team'] == team and info['role'] == 'GOALKEEPER')
            S, N = len(self.t), len(ids)
            d = {'ids': ids, 'display': {pid: roster[pid]['display'] for pid in ids+gks}, 'gks': gks,
                 'x': np.full((S, N), np.nan), 'y': np.full((S, N), np.nan), 'kind': np.full((S, N), '', dtype='<U1'),
                 'identity': np.zeros((S, N)), 'quality': np.zeros((S, N)), 'sigma': np.full((S, N), np.nan),
                 'ex': np.full((S, N), np.nan), 'ey': np.full((S, N), np.nan), 'esigma': np.full((S, N), np.nan),
                 'gk': np.full((S, 2), np.nan), 'excluded': Counter()}
            for j, pid in enumerate(ids):
                self._fill(d, j, pid)
            # Slots that cannot be on the pitch at a sample: not yet seen (a substitute) or recorded as having left (a
            # substitution or dismissal you marked: roster slot 'left' {'time': ...}).
            d['active'] = np.ones((S, N), bool)
            for j, pid in enumerate(ids):
                slot = slots.get(pid) or {}
                first = slot.get('firstSeen')
                left = slot.get('left')
                left = left.get('time') if isinstance(left, dict) else left
                if isinstance(first, (int, float)):
                    d['active'][:, j] &= self.t >= float(first)-1.0
                if isinstance(left, (int, float)) and not isinstance(left, bool):
                    d['active'][:, j] &= self.t <= float(left)+1.0
            # A player who has left is not an outfield player any more, even while still tracked walking off; this comes
            # before the duplicates, so a departed identity carried on onto his substitute cannot remove the substitute.
            gone = (d['kind'] == 'O') & ~d['active']
            if gone.any():
                d['x'][gone] = d['y'][gone] = np.nan
                d['kind'][gone] = ''
                d['excluded']['afterLeaving'] += int(gone.sum())
            self._remove_duplicates(d)
            if block_ok:
                for j, pid in enumerate(ids):
                    self._fill_estimate(d, j, block['ids'].get(pid))
            for pid in gks:
                self._fill_goalkeeper(d, pid)
            default = max(1, min(int(expected.get(team) or 11)-1, 10))
            before, after, change = lineup_of(lineup.get(team), default)
            d['default'] = default
            d['expected_t'] = np.full(S, max(1, min(before, 10)), np.int64)
            if change is not None:
                d['expected_t'][self.t >= change] = max(1, min(after, 10))
            d['expected'] = int(d['expected_t'][0]) if S else default
            d['lineupChange'] = {'from': change, 'outfield': after} if change is not None else None
            if mode == 'estimated':
                use = (d['kind'] == '') & np.isfinite(d['ex']) & d['active']
                d['x'][use], d['y'][use], d['kind'][use], d['sigma'][use] = d['ex'][use], d['ey'][use], 'E', d['esigma'][use]
                self._drop_estimates_on_observed(d)
            self.teams[team] = d

    def _fill(self, d, j, pid):
        """One identity's observed samples onto the grid, without registration glitches and uncertain positions. A
        glitch is a run of samples shorter than SPIKE_SECONDS that a registration jump (a step faster than any player,
        analytics._breaks) separates from the rest of the track on at least one side, while its other side is a jump, a
        gap or the end of the track: a mapped position that leaps away and back, or leaps and then disappears, or appears
        wrong for a moment before the track settles. A short genuine sighting between two gaps has no jump and stays."""
        tr = self.samples.tracks.get(pid)
        if tr is None or not len(self.t):
            return
        t = tr['t']
        k = np.clip(np.searchsorted(t, self.t), 0, len(t)-1)
        hit = np.abs(t[k]-self.t) <= .25/self.fps
        run, why = tr['run'], tr['why']
        spike = np.zeros(len(t), bool)
        starts = np.flatnonzero(np.r_[True, run[1:] != run[:-1]])
        ends = np.r_[starts[1:], len(t)]
        for a, b in zip(starts, ends):
            if t[b-1]-t[a] >= SPIKE_SECONDS:
                continue
            before = str(why[a]) if a > 0 else 'edge'
            after = str(why[b]) if b < len(t) else 'edge'
            sides = (before, after)
            if 'jump' in sides and all(s in ('jump', 'unseen', 'edge') for s in sides) and not (a == 0 and b == len(t)):
                spike[a:b] = True
        rows = np.flatnonzero(hit)
        src = k[rows]
        bad_spike = spike[src]
        bad_sigma = np.isfinite(tr['sigma'][src]) & (tr['sigma'][src] > MAX_SIGMA)
        d['excluded']['registrationSpike'] += int(bad_spike.sum())
        d['excluded']['uncertainPosition'] += int((bad_sigma & ~bad_spike).sum())
        ok = ~bad_spike & ~bad_sigma
        rows, src = rows[ok], src[ok]
        d['x'][rows, j] = tr['x'][src]
        d['y'][rows, j] = tr['y'][src]
        d['kind'][rows, j] = 'O'
        d['identity'][rows, j] = tr['identity'][src]
        d['quality'][rows, j] = tr['quality'][src]
        d['sigma'][rows, j] = tr['sigma'][src]

    def _remove_duplicates(self, d):
        """Two identities of a team within DUPLICATE_METRES of each other on DUPLICATE_SAMPLES or more consecutive samples
        on which both are seen are one person seen twice (a split track the identity manager did not merge). Samples
        where one of them is unseen neither break nor extend the stretch, so a second detection that flickers on and off
        is caught too. Over the stretch the identity with the lower identity confidence is dropped. Brief closeness of
        two teammates seen together (a wall, a challenge) is not a duplicate. The most certain identities claim their
        stretches first and a position already dropped is never compared again, so a ghost between two real teammates
        removes only itself, whatever the identities are called."""
        x, y = d['x'], d['y']
        S, N = x.shape
        if N < 2 or not S:
            return
        seen = np.isfinite(x)
        dist = np.hypot((x[:, :, None]-x[:, None, :])*self.length, (y[:, :, None]-y[:, None, :])*self.width)
        close = dist < DUPLICATE_METRES
        stretches = []
        for i, j in zip(*np.nonzero(np.triu(close.any(axis=0), k=1))):
            both = np.flatnonzero(seen[:, i] & seen[:, j])
            col = np.r_[False, close[both, i, j], False]
            edges = np.flatnonzero(col[1:] != col[:-1])
            for a, b in zip(edges[::2], edges[1::2]):
                if b-a >= DUPLICATE_SAMPLES:
                    rows = both[a:b]
                    ci, cj = float(d['identity'][rows, i].mean()), float(d['identity'][rows, j].mean())
                    stretches.append((-max(ci, cj), -(b-a), int(i), int(j), rows, ci, cj))
        for _, _, i, j, rows, ci, cj in sorted(stretches, key=lambda st: st[:4]):
            live = rows[np.isfinite(x[rows, i]) & np.isfinite(x[rows, j])]
            if not len(live):
                continue
            drop = i if (ci, -i) < (cj, -j) else j
            x[live, drop] = y[live, drop] = np.nan
            d['kind'][live, drop] = ''
            d['excluded']['duplicateIdentity'] += int(len(live))

    def _drop_estimates_on_observed(self, d):
        """Estimated mode. An estimated position on top of an observed teammate is most likely the same person (a stale
        prediction of a player observed under another identity), and of two estimates on top of each other only the
        more certain one is kept: neither counts as an extra player. A sample never has more positions than its lineup:
        the least certain estimates go first."""
        est = d['kind'] == 'E'
        if not est.any():
            return
        near = 1.25*DUPLICATE_METRES
        for s in np.flatnonzero(est.any(axis=1)):
            x, y = d['x'][s], d['y'][s]
            oj = np.flatnonzero(d['kind'][s] == 'O')
            ej = np.flatnonzero(d['kind'][s] == 'E')
            ej = ej[np.argsort(np.nan_to_num(d['sigma'][s, ej], nan=np.inf), kind='stable')]    # the most certain first
            kept = []
            for j in ej:
                on_observed = len(oj) and (np.hypot((x[oj]-x[j])*self.length, (y[oj]-y[j])*self.width) < near).any()
                on_estimate = kept and (np.hypot((x[kept]-x[j])*self.length, (y[kept]-y[j])*self.width) < near).any()
                if on_observed or on_estimate:
                    d['x'][s, j] = d['y'][s, j] = np.nan
                    d['kind'][s, j] = ''
                    d['excluded']['estimateOnObserved' if on_observed else 'estimateDuplicate'] += 1
                else:
                    kept.append(j)
            room = max(0, int(d['expected_t'][s])-len(oj))
            for j in kept[room:]:
                d['x'][s, j] = d['y'][s, j] = np.nan
                d['kind'][s, j] = ''
                d['excluded']['estimateBeyondLineup'] += 1

    def _fill_estimate(self, d, j, cols):
        if not cols or not len(self.t):
            return
        kind = cols.get('kind') or []
        xs, ys, sig = cols.get('x') or [], cols.get('y') or [], cols.get('sigma') or []
        for s, f in enumerate(self.frame_index):
            if f >= len(kind) or kind[f] not in ('I', 'P') or f >= len(xs) or f >= len(ys) or xs[f] is None or ys[f] is None:
                continue
            sg = sig[f] if f < len(sig) else None
            sg = float(max(sg)) if isinstance(sg, (list, tuple)) and sg else (float(sg) if isinstance(sg, (int, float)) else math.nan)
            if not analytics._valid([xs[f], ys[f]]) or (np.isfinite(sg) and sg > ESTIMATE_MAX_SIGMA):
                continue
            d['ex'][s, j], d['ey'][s, j], d['esigma'][s, j] = float(xs[f]), float(ys[f]), sg

    def _fill_goalkeeper(self, d, pid):
        tr = self.samples.tracks.get(pid)
        if tr is None or not len(self.t):
            return
        t = tr['t']
        k = np.clip(np.searchsorted(t, self.t), 0, len(t)-1)
        hit = np.abs(t[k]-self.t) <= .25/self.fps
        free = hit & ~np.isfinite(d['gk'][:, 0])
        d['gk'][free, 0] = tr['x'][k[free]]
        d['gk'][free, 1] = tr['y'][k[free]]

    # ----- helpers -----
    def rows(self, t0, t1):
        """Sample indices within [t0, t1]."""
        return np.flatnonzero((self.t >= t0-1e-9) & (self.t <= t1+1e-9))

    def metres(self, axis):
        return self.length if axis == 'x' else self.width


# ---------- attacking direction ----------
def team_directions(data, result, overrides=None):
    """Which goal each team defends at every sample. Sources, in order: the user's setting (overrides = {'A': 'left' or
    'right', 'switchAt': seconds or None}: from switchAt on the sides are swapped, e.g. the second half), the tracking's
    goal ends (tactical.goalEnds, decided from which team defends which side; when only one end is named the other team
    defends the other end), else the evidence below when it is unanimous. The evidence, per DIRECTION_WINDOW (windows
    also split at switchAt): the side the team's goalkeeper stands on, and the goal-side ordering of the two teams'
    outfield centroids (a team's centroid lies between the opponents' centroid and its own goal). The goalkeeper decides
    when he is seen (a keeper stays in his own half; on Metrica the two kinds disagreed in 1-3% of the windows, as often
    under the right direction as under a wrong one), the ordering otherwise. A window whose evidence contradicts the base
    direction is 'contradicted' (a side change the setting does not know of, or a mistaken base): no
    direction-dependent value is reported there, nor in a window without evidence next to a contradicted one (the side
    change may have happened before it). Without any base, a stretch without evidence between (or within one window of)
    windows whose evidence agrees takes their side. Returns {'A': {...}, 'B': {...}, 'windows': [...]} with
    per-sample arrays 'side' ('left', 'right' or '') and 'status' ('set', 'tracking', 'inferred', 'evidence',
    'contradicted', 'unknown')."""
    S = len(data.t)
    out = {'windows': []}
    base, source = {}, None
    overrides = overrides or {}
    if overrides.get('A') in ('left', 'right'):
        base = {'A': overrides['A'], 'B': 'right' if overrides['A'] == 'left' else 'left'}
        source = 'set'
    else:
        ends = ((result.get('tactical') or {}).get('goalEnds') or {})
        left, right = ends.get('left'), ends.get('right')
        if left in ('A', 'B') and right in ('A', 'B') and left != right:
            base, source = {left: 'left', right: 'right'}, 'tracking'
        elif left in ('A', 'B') and right in (None, ''):
            base, source = {left: 'left', ('B' if left == 'A' else 'A'): 'right'}, 'inferred'
        elif right in ('A', 'B') and left in (None, ''):
            base, source = {right: 'right', ('B' if right == 'A' else 'A'): 'left'}, 'inferred'
    switch = overrides.get('switchAt') if source == 'set' else None
    flip = lambda side: 'right' if side == 'left' else 'left'
    sides = {team: np.full(S, '', dtype='<U5') for team in ('A', 'B')}
    status = {team: np.full(S, 'unknown', dtype='<U12') for team in ('A', 'B')}
    if base:
        for team in ('A', 'B'):
            sides[team][:] = base[team]
            status[team][:] = source
            if isinstance(switch, (int, float)):
                sides[team][data.t >= float(switch)] = flip(base[team])
    checks = {'A': [], 'B': []}      # per evidence window: (rows, window, 'none' | 'ok' | 'bad')
    if S:
        edges = list(np.arange(data.start, data.end, DIRECTION_WINDOW))+[data.end+1e-6]
        if isinstance(switch, (int, float)) and data.start < switch < data.end:
            edges = sorted({round(e, 6) for e in edges if abs(e-switch) >= 1.0} | {float(switch)})
        for a, b in zip(edges[:-1], edges[1:]):
            rows = np.flatnonzero((data.t >= a-1e-9) & (data.t < b-1e-9))
            if not len(rows):
                continue
            w = data.w[rows]
            evidence = {}
            for team in ('A', 'B'):
                gk = data.teams[team]['gk'][rows]
                seen = np.isfinite(gk[:, 0])
                votes = []
                if w[seen].sum() >= DIRECTION_MIN_GK_SECONDS:
                    votes.append(('goalkeeper', 'left' if np.median(gk[seen, 0]) < .5 else 'right', float(w[seen].sum())))
                evidence[team] = votes
            xa, xb = data.teams['A']['x'][rows], data.teams['B']['x'][rows]
            na, nb = np.isfinite(xa).sum(axis=1), np.isfinite(xb).sum(axis=1)
            both = (na >= PARTIAL_MIN) & (nb >= PARTIAL_MIN)
            if w[both].sum() >= DIRECTION_MIN_ORDER_SECONDS:
                ma, mb = np.nanmean(xa[both], axis=1), np.nanmean(xb[both], axis=1)
                a_left = float((w[both]*(ma < mb)).sum()/w[both].sum())
                if a_left >= .75 or a_left <= .25:
                    evidence['A'].append(('ordering', 'left' if a_left >= .75 else 'right', float(w[both].sum())))
                    evidence['B'].append(('ordering', 'right' if a_left >= .75 else 'left', float(w[both].sum())))
            window = {'t0': _round(a, 2), 't1': _round(min(b, data.end), 2),
                      'evidence': {t: [{'kind': k, 'side': s, 'seconds': _round(sec, 1)} for k, s, sec in v] for t, v in evidence.items()}}
            for team in ('A', 'B'):
                said = {s for _, s, _ in evidence[team]}
                if base:
                    current = str(sides[team][rows[0]])        # uniform inside the window: windows split at switchAt
                    keeper = {s for k, s, _ in evidence[team] if k == 'goalkeeper'}
                    decisive = keeper or said
                    if len(said) > 1:
                        window.setdefault('mixed', []).append(team)
                    bad = bool(decisive) and current not in decisive
                    checks[team].append((rows, window, 'bad' if bad else ('ok' if said else 'none')))
                    if bad:
                        status[team][rows] = 'contradicted'
                        window.setdefault('contradicted', []).append(team)
                elif len(said) == 1:
                    side = next(iter(said))
                    other = {s for _, s, _ in evidence['B' if team == 'A' else 'A']}
                    if not other or other == {flip(side)}:
                        sides[team][rows] = side
                        status[team][rows] = 'evidence'
            out['windows'].append(window)
        if base:
            # A window without evidence next to a contradicted one is not believed either.
            for team in ('A', 'B'):
                states = [c[2] for c in checks[team]]
                for i, (rows, window, state) in enumerate(checks[team]):
                    if state != 'none':
                        continue
                    before = next((states[k] for k in range(i-1, -1, -1) if states[k] != 'none'), None)
                    after = next((states[k] for k in range(i+1, len(states)) if states[k] != 'none'), None)
                    if 'bad' in (before, after):
                        status[team][rows] = 'contradicted'
                        window.setdefault('contradicted', []).append(team)
                        window.setdefault('unverified', []).append(team)
        else:
            for team in ('A', 'B'):
                _fill_direction_gaps(data.t, sides[team], status[team])
    for team in ('A', 'B'):
        bad = status[team] == 'contradicted'
        out[team] = {'side': np.where(bad, '', sides[team]), 'status': status[team], 'base': base.get(team), 'source': source}
    out['switchAt'] = switch
    return out


def _fill_direction_gaps(t, sides, status):
    """Without a base direction: samples without evidence between two stretches with the same evidence side (at most two
    windows apart), or within one window of the only evidence at either end, take that side."""
    known = np.flatnonzero(sides != '')
    if not len(known):
        return
    for s in np.flatnonzero(sides == ''):
        k = bisect.bisect_left(known.tolist(), s)
        before = known[k-1] if k > 0 else None
        after = known[k] if k < len(known) else None
        if before is not None and after is not None:
            if sides[before] == sides[after] and t[after]-t[before] <= 2*DIRECTION_WINDOW:
                sides[s], status[s] = sides[before], 'evidence'
        else:
            near = before if before is not None else after
            if abs(t[s]-t[near]) <= DIRECTION_WINDOW:
                sides[s], status[s] = sides[near], 'evidence'


def own_goal_distance(x, side, length):
    """Metres from the team's own goal line along the goal-to-goal axis: x * L when defending the left goal (x = 0),
    (1 - x) * L when defending the right goal (x = 1)."""
    return np.where(side == 'left', x*length, np.where(side == 'right', (1-x)*length, np.nan))


def attack_frame(x, y, side):
    """Attacking-direction-relative coordinates: X' 0 at the own goal line, 1 at the opponents' goal line; Y' 0 at the
    team's left touchline (looking towards the goal it attacks), 1 at its right. Defending the left goal the team looks
    towards +x and its left is the far touchline (y = 0); defending the right goal both flip."""
    X = np.where(side == 'left', x, np.where(side == 'right', 1-x, np.nan))
    Y = np.where(side == 'left', y, np.where(side == 'right', 1-y, np.nan))
    return X, Y


# ---------- width and depth ----------
def _weighted_median(values, weights):
    if not len(values):
        return None
    order = np.argsort(values)
    v, w = np.asarray(values)[order], np.asarray(weights)[order]
    c = np.cumsum(w)
    if c[-1] <= 0:
        return None
    k = int(np.searchsorted(c, c[-1]/2))
    if abs(c[k]-c[-1]/2) < 1e-12 and k+1 < len(v):
        return float((v[k]+v[k+1])/2)
    return float(v[k])


def _edge_risk(data, d, s, axis, lo, hi, expected):
    """Observed mode: could the true extreme on this axis lie outside the observed [lo, hi] (normalized) at sample s?
    'none' when every missing player of the lineup has an estimate inside the observed range (with its sigma and
    EDGE_MARGIN), 'possible' when some estimate lies beyond it, 'unknown' when some missing player has no estimate.
    Only estimates of distinct people count: one on top of an observed teammate, or on top of a more certain estimate,
    is a split identity's stale prediction, not the missing player."""
    observed = d['kind'][s] == 'O'
    missing = expected-int(observed.sum())
    if missing <= 0:
        return 'none'
    ex, ey, sig = d['ex'][s], d['ey'][s], d['esigma'][s]
    pool = np.flatnonzero(~observed & d['active'][s] & np.isfinite(ex))
    pool = pool[np.argsort(np.nan_to_num(sig[pool], nan=np.inf), kind='stable')]
    taken_x, taken_y = list(d['x'][s, observed]), list(d['y'][s, observed])
    candidates = []
    near = 1.25*DUPLICATE_METRES
    for j in pool:
        if taken_x and (np.hypot((np.array(taken_x)-ex[j])*data.length, (np.array(taken_y)-ey[j])*data.width) < near).any():
            continue
        candidates.append(j)
        taken_x.append(ex[j])
        taken_y.append(ey[j])
    if len(candidates) < missing:
        return 'unknown'
    candidates = np.array(candidates)
    pos = ex if axis == 'x' else ey
    margin = (np.nan_to_num(sig[candidates], nan=0.0)+EDGE_MARGIN)/data.metres(axis)
    beyond = (pos[candidates]-margin < lo) | (pos[candidates]+margin > hi)
    return 'possible' if beyond.any() else 'none'


def width_depth_series(data, team, t0, t1):
    """Per sample in [t0, t1]: the measured outfield count (observed, and estimated in estimated mode), the team's width
    (y extent) and depth (x extent) by the standard max - min definition in normalized units and metres, the separately
    labelled 10th-90th percentile spreads, the players at the extremes, the observed players' mean pitch confidence
    (geometric) and lowest identity confidence, and a status per dimension:
        complete     every outfield player of the lineup measured, enough of them observed (reliable_min: 8, or all but
                     one of a smaller lineup);
        reliable     (observed mode) reliable_min observed and every missing player's (distinct) estimate lies inside
                     the measured extremes;
        estimated    (estimated mode) mostly estimates: the lineup filled with too few observed, or fewer than
                     PARTIAL_MIN observed;
        partial      at least PARTIAL_MIN measured (observed, in estimated mode) but an extreme may be missing: the
                     observed spread, a lower bound;
        excess       more observed players than the lineup (a duplicate identity, or a player who left still tracked):
                     an extreme may be someone who is not in the team's shape;
        unavailable  fewer than PARTIAL_MIN measured: no value.
    Only complete and reliable values enter the averages."""
    d = data.teams[team]
    rows = data.rows(t0, t1)
    out = []
    lo_p, hi_p = ROBUST_PERCENTILES
    for s in rows:
        have = np.isfinite(d['x'][s])
        obs = d['kind'][s] == 'O'
        n = int(have.sum())
        observed = int(obs.sum())
        estimated = int((d['kind'][s] == 'E').sum())
        expected = int(d['expected_t'][s])
        rec = {'t': round(float(data.t[s]), 3), 'w': round(float(data.w[s]), 4), 'players': n, 'observed': observed, 'estimated': estimated,
               'expected': expected,
               'geometric': round(float(d['quality'][s, obs].mean()), 3) if observed else None,
               'identity': round(float(d['identity'][s, obs].min()), 3) if observed else None}
        if n < PARTIAL_MIN:
            rec.update(width=None, depth=None, widthStatus='unavailable', depthStatus='unavailable', widthEdgeRisk='unknown', depthEdgeRisk='unknown')
            out.append(rec)
            continue
        xs, ys, ids = d['x'][s, have], d['y'][s, have], [d['ids'][j] for j in np.flatnonzero(have)]
        width_n, depth_n = float(ys.max()-ys.min()), float(xs.max()-xs.min())
        rec['width'] = {'normalized': round(width_n, 4), 'metres': round(width_n*data.width, 2),
                        'from': ids[int(np.argmin(ys))], 'to': ids[int(np.argmax(ys))]}
        rec['depth'] = {'normalized': round(depth_n, 4), 'metres': round(depth_n*data.length, 2),
                        'from': ids[int(np.argmin(xs))], 'to': ids[int(np.argmax(xs))]}
        rec['robustWidth'] = round(float(np.percentile(ys, hi_p)-np.percentile(ys, lo_p))*data.width, 2)
        rec['robustDepth'] = round(float(np.percentile(xs, hi_p)-np.percentile(xs, lo_p))*data.length, 2)
        enough = reliable_min(expected)
        for axis, key, vals in (('y', 'width', ys), ('x', 'depth', xs)):
            if observed > expected:
                status, risk = 'excess', 'none'
            elif data.mode == 'observed':
                if observed >= expected:
                    status, risk = 'complete', 'none'
                else:
                    risk = _edge_risk(data, d, s, axis, float(vals.min()), float(vals.max()), expected)
                    status = 'reliable' if observed >= enough and risk == 'none' else 'partial'
            else:
                # Estimated mode: every available estimate is already among the measured players, so a lineup player
                # still without a position has an unknown edge; a filled lineup is complete only if mostly observed,
                # and a spread is partial only if its observed players alone would be.
                if n >= expected:
                    risk = 'none'
                    status = 'complete' if observed >= min(enough, expected) else 'estimated'
                else:
                    risk, status = 'unknown', ('partial' if observed >= PARTIAL_MIN else 'estimated')
            rec[key+'Status'] = status
            rec[key+'EdgeRisk'] = risk
        out.append(rec)
    return out


def _aggregate(series, key):
    """Time-weighted mean, weighted median, observed min / max of a series value over its reliable samples (complete or
    reliable status), with the shares of the interval that were reliable, mostly estimated, partial and unavailable."""
    status = key+'Status'
    good = [r for r in series if r.get(status) in ('complete', 'reliable')]
    part = [r for r in series if r.get(status) == 'partial']
    est = [r for r in series if r.get(status) == 'estimated']
    exc = [r for r in series if r.get(status) == 'excess']
    total = sum(r['w'] for r in series)   # the time the samples of the interval stand for
    share = lambda rs: _round(sum(r['w'] for r in rs)/total, 3) if total else 0.0
    agg = {'reliableShare': share(good), 'partialShare': share(part), 'estimatedShare': share(est), 'excessShare': share(exc),
           'samples': len(good)}
    known = sum((agg[k] or 0) for k in ('reliableShare', 'partialShare', 'estimatedShare', 'excessShare'))
    agg['unavailableShare'] = _round(max(0.0, 1-known), 3) if total else None
    if good:
        v = np.array([r[key]['metres'] for r in good])
        n = np.array([r[key]['normalized'] for r in good])
        w = np.array([r['w'] for r in good])
        agg.update(mean=_round((v*w).sum()/w.sum()), median=_round(_weighted_median(v, w)), min=_round(v.min()), max=_round(v.max()),
                   meanNormalized=_round((n*w).sum()/w.sum(), 4))
    else:
        agg.update(mean=None, median=None, min=None, max=None, meanNormalized=None)
    if part:
        v = np.array([r[key]['metres'] for r in part])
        w = np.array([r['w'] for r in part])
        agg['partialMean'] = _round((v*w).sum()/w.sum())   # the observed spread only: a lower bound, never mixed into the mean above
    return agg


def team_width_depth(data, team, t0, t1):
    series = width_depth_series(data, team, t0, t1)
    return {'series': series, 'width': _aggregate(series, 'width'), 'depth': _aggregate(series, 'depth')}


# ---------- formation templates ----------
# Role anchors in metres relative to the team's outfield centroid, in the attacking frame: dx along the attacking
# direction (+ towards the opponents' goal), dy across it (+ towards the team's RIGHT touchline). The formations in
# _EMPIRICAL are the mean role locations of SoccerCPD's 864 formation periods (Kim et al., KDD 2022, its
# data/role_records.csv: centroid-relative, both possession phases mixed), mirrored left/right and averaged so that the
# templates are symmetric; those in _COMPOSED and _NINE are built from the same lines (the same back four, back three,
# midfield and forward depths), so that every template is made the same way. A formation is its units from the back, as
# its name counts them (4-3-3: the back four, the midfield three, the front three), members ordered left to right; the
# first unit is the defensive unit, the last the forwards.
_EMPIRICAL = {
    '4-4-2': [[('LB', -7.5, -17.2), ('LCB', -13.4, -6.0), ('RCB', -13.4, 6.0), ('RB', -7.5, 17.2)],
              [('LM', 8.2, -16.7), ('LCM', -1.6, -5.6), ('RCM', -1.6, 5.6), ('RM', 8.2, 16.7)],
              [('LCF', 14.3, -4.6), ('RCF', 14.3, 4.6)]],
    '4-2-3-1': [[('LB', -7.1, -17.4), ('LCB', -13.5, -6.1), ('RCB', -13.5, 6.1), ('RB', -7.1, 17.4)],
                [('LDM', -1.9, -5.5), ('RDM', -1.9, 5.5)],
                [('LM', 9.2, -16.5), ('CAM', 9.1, 0.0), ('RM', 9.2, 16.5)],
                [('CF', 17.5, 0.0)]],
    '4-3-3': [[('LB', -7.0, -17.0), ('LCB', -13.5, -6.3), ('RCB', -13.5, 6.3), ('RB', -7.0, 17.0)],
              [('LCM', 4.5, -6.7), ('CDM', -4.8, 0.0), ('RCM', 4.5, 6.7)],
              [('LW', 9.9, -16.3), ('CF', 17.0, 0.0), ('RW', 9.9, 16.3)]],
    '3-5-2': [[('LCB', -11.1, -11.3), ('CB', -13.3, 0.0), ('RCB', -11.1, 11.3)],
              [('LWB', -1.4, -20.0), ('LCM', 4.7, -9.8), ('CDM', -2.3, 0.0), ('RCM', 4.7, 9.8), ('RWB', -1.4, 20.0)],
              [('LCF', 15.6, -6.4), ('RCF', 15.6, 6.4)]],
    '3-4-3': [[('LCB', -11.1, -10.7), ('CB', -13.1, 0.0), ('RCB', -11.1, 10.7)],
              [('LWB', -2.2, -19.2), ('LCM', 0.4, -5.3), ('RCM', 0.4, 5.3), ('RWB', -2.2, 19.2)],
              [('LW', 11.4, -11.8), ('CF', 15.9, 0.0), ('RW', 11.4, 11.8)]],
}
_COMPOSED = {
    '4-1-4-1': [[('LB', -7.0, -17.0), ('LCB', -13.5, -6.3), ('RCB', -13.5, 6.3), ('RB', -7.0, 17.0)],
                [('CDM', -4.8, 0.0)],
                [('LM', 5.5, -16.5), ('LCM', 4.0, -6.5), ('RCM', 4.0, 6.5), ('RM', 5.5, 16.5)],
                [('CF', 17.0, 0.0)]],
    '5-3-2': [[('LWB', -8.5, -20.0), ('LCB', -11.1, -11.3), ('CB', -13.3, 0.0), ('RCB', -11.1, 11.3), ('RWB', -8.5, 20.0)],
              [('LCM', 4.7, -9.8), ('CDM', -2.3, 0.0), ('RCM', 4.7, 9.8)],
              [('LCF', 15.6, -6.4), ('RCF', 15.6, 6.4)]],
    '5-4-1': [[('LWB', -8.5, -20.0), ('LCB', -11.1, -11.3), ('CB', -13.3, 0.0), ('RCB', -11.1, 11.3), ('RWB', -8.5, 20.0)],
              [('LM', 3.5, -16.5), ('LCM', -0.5, -5.6), ('RCM', -0.5, 5.6), ('RM', 3.5, 16.5)],
              [('CF', 15.0, 0.0)]],
}
# Nine outfield players (one dismissed): the same lines with one player fewer.
_NINE = {
    '4-4-1': [[('LB', -7.5, -17.2), ('LCB', -13.4, -6.0), ('RCB', -13.4, 6.0), ('RB', -7.5, 17.2)],
              [('LM', 6.5, -16.7), ('LCM', -1.6, -5.6), ('RCM', -1.6, 5.6), ('RM', 6.5, 16.7)], [('CF', 15.0, 0.0)]],
    '4-3-2': [[('LB', -7.0, -17.0), ('LCB', -13.5, -6.3), ('RCB', -13.5, 6.3), ('RB', -7.0, 17.0)],
              [('LCM', 1.5, -9.0), ('CM', -1.5, 0.0), ('RCM', 1.5, 9.0)], [('LCF', 14.3, -4.6), ('RCF', 14.3, 4.6)]],
    '5-3-1': [[('LWB', -8.5, -20.0), ('LCB', -11.1, -11.3), ('CB', -13.3, 0.0), ('RCB', -11.1, 11.3), ('RWB', -8.5, 20.0)],
              [('LCM', 2.0, -9.8), ('CM', -1.0, 0.0), ('RCM', 2.0, 9.8)], [('CF', 15.0, 0.0)]],
}
FORMATIONS_10 = ('4-4-2', '4-3-3', '4-2-3-1', '4-1-4-1', '3-5-2', '3-4-3', '5-3-2', '5-4-1')
FORMATIONS_9 = ('4-4-1', '4-3-2', '5-3-1')
# Pairs that differ in one detail: often both fair descriptions of one shape.
# (No 4-4-1-1: built from the empirical lines, where wide midfielders sit about 10 m above the central pair, it is the
# 4-2-3-1 again and could only produce artificial ambiguity.)
RELATED = {frozenset(p) for p in (('4-3-3', '4-1-4-1'), ('3-5-2', '5-3-2'), ('3-4-3', '5-4-1'), ('4-4-2', '4-2-3-1'))}

WINDOW_SECONDS = 15.0         # s: a formation window (the team's average shape over it)
WINDOW_STEP = 5.0             # s: windows start this far apart (they overlap)
MIN_PRESENT = .5              # share of a window a player must be observed in to take part
FORMATION_MAX_MISSING = 1     # outfield players of the lineup a window may miss and still be classified (9 of 10: on Metrica
                              # with a simulated camera, windows of 8 agreed with the full-coverage label only 25-56% of the
                              # time); a 'clear' formation needs all of them
SCALE_BOUNDS = (.6, 1.8)      # the elastic template's stretch per axis, relative to its nominal size
UNCLEAR_ERROR = .42           # relative fit error above which no formation describes the window ('unclear')
SWITCH_PENALTY = .45          # change-point cost: a new formation must fit better by this much, summed over windows (tuned on
                              # Metrica game 1: 250 raw window-to-window changes an hour became 5 and 13; a clear change still
                              # takes 15-30 s, a slight one over a minute)
MIN_SEGMENT_SECONDS = 30.0    # a stabilized formation period shorter than this between two others is a transition
AMBIGUOUS_MARGIN = .04        # relative-error margin to the runner-up below which the label is ambiguous
CLEAR_MARGIN = .10            # ... at or above which (with support and coverage) it is clear
ROLE_CONTINUITY = .15         # assignment penalty (share of the team spread, squared) for leaving the previous window's line
ROLE_MEMORY = 60.0            # s: a player's last role carries this long through windows without a formation
ROLE_HORIZON = 30.0           # s: the defensive unit at a time is the back line by majority over the windows this close to it
CLEAR_IDENTITY = .6           # a 'clear' formation also needs this mean identity confidence ...
CLEAR_ROLE_STABILITY = .7     # ... and this share of its players holding one role for 75% of its windows


def template(name):
    lines = _EMPIRICAL.get(name) or _COMPOSED.get(name) or _NINE.get(name)
    if lines is None:
        raise KeyError(name)
    roles, xy, line = [], [], []
    for k, members in enumerate(lines):
        for role, dx, dy in members:
            roles.append(role)
            xy.append((dx, dy))
            line.append(k)
    xy = np.array(xy, np.float64)
    xy -= xy.mean(axis=0)
    groups = ['D' if k == 0 else ('F' if k == len(lines)-1 else 'M') for k in line]
    return {'name': name, 'roles': roles, 'xy': xy, 'line': np.array(line), 'group': groups, 'lines': [len(m) for m in lines],
            'source': 'empirical' if name in _EMPIRICAL else 'composed'}


TEMPLATES = {name: template(name) for name in FORMATIONS_10+FORMATIONS_9}


def candidates(expected):
    if expected == 10:
        return FORMATIONS_10
    if expected == 9:
        return FORMATIONS_9
    return ()




def fit_template(P, tpl, prev_group=None):
    """Elastic template match of an observed shape P (m x 2 metres, centroid-relative, attacking frame) to a template
    with at least m roles: the template is shifted and stretched per axis (within SCALE_BOUNDS of its nominal size) and
    the players are assigned to its roles one to one by the Hungarian algorithm, alternating until the assignment is
    stable (from two starting scales, the better kept). A player whose previous line (prev_group[i]: 'D', 'M', 'F' or '')
    is the back line while the role is not, or the other way round, pays ROLE_CONTINUITY in the assignment (not in the
    reported error). Returns the RMS residual in metres, the relative error (RMS residual over the shape's RMS spread),
    the role index per player, the per-axis scale and the shift."""
    T = tpl['xy']
    m = len(P)
    if m > len(T):
        raise ValueError(f'{m} players for a {len(T)}-role template')
    spread = float(np.sqrt((np.sum((P-P.mean(axis=0))**2, axis=1)).mean())) or 1.0
    t_std = T.std(axis=0)
    inits = [np.clip(P.std(axis=0)/np.where(t_std > 0, t_std, 1.0), *SCALE_BOUNDS), np.ones(2)]
    penalty = None
    if prev_group is not None and any(prev_group):
        back = np.array([g == 'D' for g in tpl['group']])
        penalty = np.zeros((m, len(T)))
        for i, g in enumerate(prev_group):
            if g:
                penalty[i] = np.where(back != (g == 'D'), ROLE_CONTINUITY*spread**2, 0.0)
    best = None
    for scale in inits:
        s = scale.copy()
        shift = P.mean(axis=0)-(T*s).mean(axis=0)
        cols = rows = None
        for _ in range(12):
            A = T*s+shift
            C = ((P[:, None, :]-A[None, :, :])**2).sum(axis=2)
            if penalty is not None:
                C = C+penalty
            r, c = linear_sum_assignment(C)
            if cols is not None and np.array_equal(c, cols):
                break
            rows, cols = r, c
            Tm, Pm = T[c], P[r]
            for axis in (0, 1):
                tv = Tm[:, axis]-Tm[:, axis].mean()
                var = float((tv**2).sum())
                if var > 1e-9:
                    s[axis] = float(np.clip((tv*(Pm[:, axis]-Pm[:, axis].mean())).sum()/var, *SCALE_BOUNDS))
                shift[axis] = Pm[:, axis].mean()-s[axis]*Tm[:, axis].mean()
        A = T*s+shift
        res = P[rows]-A[cols]
        rms = float(np.sqrt((res**2).sum(axis=1).mean()))
        if best is None or rms < best['rms']:
            order = np.empty(m, int)
            order[rows] = cols
            best = {'rms': rms, 'relative': rms/spread, 'roles': order, 'scale': s.copy(), 'shift': shift.copy(), 'spread': spread}
    return best


# ---------- formation windows ----------
def _window_shape(data, team, rows, side, expected, prefer=()):
    """The team's average shape over one window, from OBSERVED positions only: the outfield players observed in at
    least MIN_PRESENT of the window (the `expected` most present of them when more are, e.g. around a substitution),
    each represented by the mean of their offset from the team centre. A sample's centre is voted (each visible player's
    position minus his own mean offset), not the mean of whoever is visible, so a camera showing one side of the team
    does not drag the shape. Returns the ids, the offsets P (m x 2 metres, attacking frame), the mean centre (attacking
    frame metres), each player's observed share, the players' mean identity confidence and how many were left out."""
    d = data.teams[team]
    w = data.w[rows]
    total = float(w.sum())
    observed = d['kind'][rows] == 'O'
    present = (observed*w[:, None]).sum(axis=0)/max(total, 1e-9)
    cols = np.flatnonzero(present >= MIN_PRESENT)
    excess = max(0, len(cols)-expected)
    if excess:
        # The previous window's players first (the same subset from window to window), then the most present.
        keep = np.array([d['ids'][j] in prefer for j in cols], bool)
        cols = np.sort(cols[np.lexsort((-present[cols], ~keep))[:expected]])
    if not len(cols):
        return None
    X, Y = attack_frame(d['x'][np.ix_(rows, cols)], d['y'][np.ix_(rows, cols)], side)
    pos = np.stack([X*data.length, Y*data.width], axis=2)          # (rows, m, 2) metres
    vis = observed[:, cols] & np.isfinite(pos[..., 0])
    count = vis.sum(axis=1)
    use = count >= max(2, int(np.ceil(.6*len(cols))))
    if w[use].sum() < .3*total:
        return None
    pos, vis, w = pos[use], vis[use], w[use]
    count = vis.sum(axis=1)
    centre = np.where(vis[..., None], pos, 0.0).sum(axis=1)/np.maximum(count, 1)[:, None]
    full = count == count.max()
    offsets = np.zeros((len(cols), 2))
    for i in range(len(cols)):
        sel = vis[:, i] & full if (vis[:, i] & full).any() else vis[:, i]
        offsets[i] = (pos[sel, i]-centre[sel]).mean(axis=0)
    for _ in range(2):
        centre = np.where(vis[..., None], pos-offsets[None], 0.0).sum(axis=1)/np.maximum(count, 1)[:, None]
        for i in range(len(cols)):
            sel = vis[:, i]
            offsets[i] = ((pos[sel, i]-centre[sel])*w[sel, None]).sum(axis=0)/max(w[sel].sum(), 1e-9)
    offsets -= offsets.mean(axis=0)
    centre_mean = (centre*w[:, None]).sum(axis=0)/w.sum()
    ident = d['identity'][rows][:, cols][observed[:, cols]]
    return {'ids': [d['ids'][j] for j in cols], 'cols': cols, 'P': offsets, 'centre': centre_mean, 'present': present[cols],
            'players': len(cols), 'excess': excess, 'identity': float(ident.mean()) if ident.size else 0.0}


def formation_windows(data, team, directions):
    """Every WINDOW_SECONDS window (WINDOW_STEP apart) of the segment: its lineup, coverage and direction and, for each
    candidate formation of that lineup, the elastic template fit; the best and the runner-up. Role continuity runs from
    window to window. A window is classified only with one direction and one lineup throughout, and with at most
    FORMATION_MAX_MISSING of the lineup's players missing."""
    d = data.teams[team]
    out = []
    if not len(data.t):
        return out
    side_all = directions[team]['side']
    prev_group, prev_ids = {}, set()
    t = data.start
    while True:
        a, b = t, min(t+WINDOW_SECONDS, data.end)
        rows = data.rows(a, b)
        win = {'t0': round(a, 2), 't1': round(b, 2), 'centre': round((a+b)/2, 2), 'status': 'ok', '_rows': rows}
        sides = set(side_all[rows].tolist()) if len(rows) else set()
        lineups = set(d['expected_t'][rows].tolist()) if len(rows) else set()
        expected = next(iter(lineups)) if len(lineups) == 1 else None
        win['expected'] = expected
        names = candidates(expected) if expected is not None else ()
        if not len(rows):
            win['status'] = 'empty'
        elif not names:
            win['status'] = 'lineup'
        elif len(sides) != 1 or '' in sides:
            win['status'] = 'direction'
        else:
            side = next(iter(sides))
            shape = _window_shape(data, team, rows, side, expected, prev_ids)
            if shape is None or shape['players'] < expected-FORMATION_MAX_MISSING:
                win['status'] = 'coverage'
                win['players'] = 0 if shape is None else shape['players']
            else:
                win.update(players=shape['players'], ids=shape['ids'], side=side, centre_xy=shape['centre'].round(3).tolist(),
                           present=[round(float(v), 2) for v in shape['present']], excess=shape['excess'], identity=round(shape['identity'], 3))
                pg = [prev_group.get(pid, '') for pid in shape['ids']]
                fits = {name: fit_template(shape['P'], TEMPLATES[name], pg) for name in names}
                ranked = sorted(fits, key=lambda n: fits[n]['relative'])
                win['fits'] = {n: {'relative': round(fits[n]['relative'], 4), 'rms': round(fits[n]['rms'], 3)} for n in names}
                win['best'], win['second'] = ranked[0], (ranked[1] if len(ranked) > 1 else None)
                best = fits[ranked[0]]
                win['margin'] = round((fits[ranked[1]]['relative']-best['relative'])/max(best['relative'], 1e-9), 4) if len(ranked) > 1 else None
                win['spread'] = round(best['spread'], 2)
                win['_fits'] = fits
                win['_P'] = shape['P']
                tpl = TEMPLATES[ranked[0]]
                for pid, role in zip(shape['ids'], best['roles']):
                    prev_group[pid] = tpl['group'][role]
                prev_ids = set(shape['ids'])
        out.append(win)
        if b >= data.end-1e-6:
            break
        t += WINDOW_STEP
    return out


DATA_GAPS = ('insufficient', 'direction', 'lineup')


def stabilize(windows):
    """Change-point detection over the windows: a Viterbi path over the lineup's candidate formations plus an 'unclear'
    state. A window costs its relative fit error for a formation (UNCLEAR_ERROR for 'unclear'), and every change of state
    costs SWITCH_PENALTY, so a new formation must fit better over a sustained stretch before it replaces the current one;
    a fullback's run or a pressing moment inside one window does not. A path runs over consecutive classified windows of
    one lineup only: any window without the data for a formation (coverage, direction, lineup) ends it, and a stretch of
    them is reported as such. A formation period shorter than MIN_SEGMENT_SECONDS between two periods of evidence (a
    formation or 'unclear') becomes a transition; next to a data gap it keeps its label. Returns the per-window states and
    the periods."""
    labels = [None]*len(windows)
    runs, run = [], []
    for i, w in enumerate(windows):
        ok = w.get('status') == 'ok'
        if ok and run and i == run[-1]+1 and windows[run[-1]]['expected'] == w['expected']:
            run.append(i)
        else:
            if run:
                runs.append(run)
            run = [i] if ok else []
    if run:
        runs.append(run)
    for run in runs:
        states = list(candidates(windows[run[0]]['expected']))+['unclear']
        E = np.array([[windows[i]['fits'][n]['relative'] if n != 'unclear' else UNCLEAR_ERROR for n in states] for i in run])
        n, k = E.shape
        cost = E[0].copy()
        back = np.zeros((n, k), int)
        for r in range(1, n):
            trans = cost[:, None]+SWITCH_PENALTY*(1-np.eye(k))
            back[r] = np.argmin(trans, axis=0)
            cost = trans[back[r], np.arange(k)]+E[r]
        path = [int(np.argmin(cost))]
        for r in range(n-1, 0, -1):
            path.append(int(back[r][path[-1]]))
        path.reverse()
        for i, s in zip(run, path):
            labels[i] = states[s]
    reasons = {'coverage': 'insufficient', 'direction': 'direction', 'lineup': 'lineup', 'empty': 'insufficient'}
    periods = []
    for i, w in enumerate(windows):
        label = labels[i] if labels[i] is not None else reasons.get(w.get('status'), 'insufficient')
        # Windows overlap: a period starts where the previous one ended (the first at its window's start).
        if periods and periods[-1]['label'] == label:
            periods[-1]['windows'].append(i)
            periods[-1]['t1'] = w['t1']
        else:
            start = w['t0'] if not periods else max(periods[-1]['t0'], min(periods[-1]['t1'], w['centre']-WINDOW_STEP/2))
            if periods:
                periods[-1]['t1'] = start
            periods.append({'label': label, 'windows': [i], 't0': start, 't1': w['t1']})
    for k, p in enumerate(periods):
        if 0 < k < len(periods)-1 and p['label'] in TEMPLATES and p['t1']-p['t0'] < MIN_SEGMENT_SECONDS \
                and periods[k-1]['label'] not in DATA_GAPS and periods[k+1]['label'] not in DATA_GAPS:
            p['label'] = 'transition'
    merged = []
    for p in periods:
        if merged and merged[-1]['label'] == p['label']:
            merged[-1]['windows'] += p['windows']
            merged[-1]['t1'] = p['t1']
        else:
            merged.append(dict(p, windows=list(p['windows'])))
    for p in merged:
        p['t0'], p['t1'] = round(float(p['t0']), 2), round(float(p['t1']), 2)
    return labels, merged


def _from_attack(X, Y, side, data):
    """Attacking-frame metres back to normalized pitch coordinates."""
    x, y = X/data.length, Y/data.width
    return (x, y) if side == 'left' else (1-x, 1-y)


def _anchors(ws, label, data):
    """The fitted template's role anchors for the period on the pitch: the template stretched and shifted as fitted,
    placed at the period's mean centre. Idealized positions for a faint overlay, never moved onto the players."""
    tpl = TEMPLATES[label]
    side = ws[0]['side']
    s = np.mean([w['_fits'][label]['scale'] for w in ws], axis=0)
    shift = np.mean([w['_fits'][label]['shift'] for w in ws], axis=0)
    centre = np.mean([w['centre_xy'] for w in ws], axis=0)
    A = tpl['xy']*s+shift+centre
    out = []
    for (X, Y), role, line in zip(A, tpl['roles'], tpl['line']):
        x, y = _from_attack(X, Y, side, data)
        out.append({'role': role, 'line': int(line), 'x': round(float(x), 4), 'y': round(float(y), 4)})
    return out


def _average_shape(ws, data, team):
    """Each player's average position over the period's windows (the mean centre plus his mean offset), on the pitch,
    and the goalkeeper's mean observed position over the same samples; observed positions only."""
    if not ws:
        return [], None
    side = ws[0]['side']
    centre = np.mean([w['centre_xy'] for w in ws], axis=0)
    acc = {}
    for w in ws:
        for pid, off in zip(w['ids'], w['_P']):
            acc.setdefault(pid, []).append(off)
    out = []
    for pid, offs in acc.items():
        X, Y = centre+np.mean(offs, axis=0)
        x, y = _from_attack(X, Y, side, data)
        out.append({'id': pid, 'display': data.teams[team]['display'].get(pid, pid), 'x': round(float(x), 4), 'y': round(float(y), 4),
                    'windows': len(offs)})
    rows = np.unique(np.concatenate([w['_rows'] for w in ws]))
    gk = data.teams[team]['gk'][rows]
    seen = np.isfinite(gk[:, 0])
    keeper = {'x': round(float(gk[seen, 0].mean()), 4), 'y': round(float(gk[seen, 1].mean()), 4)} if seen.sum() >= 5 else None
    return out, keeper


def _overlaps(c, p):
    """A coach's label belongs to a period when they share at least half of each: a label saved for a period still
    finds it after a re-run moves its boundaries by a few seconds, a label of a period that is now split in two belongs
    to both halves, and a short label never takes over a long neighbouring period. A label no period matches any more
    is kept, reported as not applied."""
    a, b = max(float(c['t0']), p['t0']), min(float(c['t1']), p['t1'])
    if b <= a:
        return False
    return (b-a) >= .5*max(float(c['t1'])-float(c['t0']), 1e-6) and (b-a) >= .5*max(p['t1']-p['t0'], 1e-6)


def _period_summary(windows, period, data, team, corrections):
    """A formation period: its label, how well the evidence supports it, the alternatives, the roles, the shape."""
    label = period['label']
    ws = [windows[i] for i in period['windows'] if windows[i].get('status') == 'ok']
    expected = ws[0]['expected'] if ws else next((windows[i]['expected'] for i in period['windows'] if windows[i].get('expected')), None)
    names = candidates(expected) if expected else ()
    out = {'t0': period['t0'], 't1': period['t1'], 'label': label, 'windows': len(period['windows']), 'evidenceWindows': len(ws),
           'expected': expected, 'candidates': list(names)}
    if ws and names:
        mean_err = {n: float(np.mean([w['fits'][n]['relative'] for w in ws])) for n in names}
        ranked = sorted(names, key=lambda n: mean_err[n])
        out['ranking'] = [{'formation': n, 'relativeError': round(mean_err[n], 4),
                           'rms': round(float(np.mean([w['fits'][n]['rms'] for w in ws])), 2)} for n in ranked]
        out['players'] = round(float(np.mean([w['players'] for w in ws])), 1)
        out['minPlayers'] = int(min(w['players'] for w in ws))
        out['identity'] = round(float(np.mean([w['identity'] for w in ws])), 3)
        out['excess'] = int(max(w.get('excess', 0) for w in ws))
        out['side'] = ws[0]['side']
        if label in names:
            support = sum(1 for w in ws if w['best'] == label)/len(ws)
            others = [n for n in ranked if n != label]
            runner = others[0] if others else None
            margin = (mean_err[runner]-mean_err[label])/max(mean_err[label], 1e-9) if runner else None
            tpl = TEMPLATES[label]
            votes = {}
            for w in ws:
                for pid, r in zip(w['ids'], w['_fits'][label]['roles']):
                    votes.setdefault(pid, Counter())[int(r)] += 1
            roles = {}
            for pid, c in votes.items():
                r, n = c.most_common(1)[0]
                roles[pid] = {'role': tpl['roles'][r], 'line': int(tpl['line'][r]), 'group': tpl['group'][r], 'share': round(n/sum(c.values()), 2)}
            stability = sum(1 for r in roles.values() if r['share'] >= .75)/max(1, len(roles))
            out.update(support=round(support, 3), alternative=runner, margin=round(margin, 4) if margin is not None else None,
                       roles=roles, roleStability=round(stability, 3), anchors=_anchors(ws, label, data), lines=tpl['lines'])
            if margin is None or margin < AMBIGUOUS_MARGIN:
                out['confidence'] = 'ambiguous'
            elif margin >= CLEAR_MARGIN and support >= .6 and out['minPlayers'] >= expected and len(ws) >= 3 \
                    and out['identity'] >= CLEAR_IDENTITY and stability >= CLEAR_ROLE_STABILITY and not out['excess']:
                out['confidence'] = 'clear'
            else:
                out['confidence'] = 'likely'
            out['related'] = bool(runner and frozenset((label, runner)) in RELATED)
    out['shape'], out['goalkeeper'] = _average_shape(ws, data, team)
    corr = [c for c in corrections if c.get('team') == team and _overlaps(c, period)]
    if corr:
        out['correction'] = corr[-1]
        label_c = corr[-1].get('formation')
        if ws and label_c in names:
            # The roles under the coach's formation (they decide this period's defensive unit); the prediction stays.
            tpl = TEMPLATES[label_c]
            votes = {}
            for w in ws:
                for pid, r in zip(w['ids'], w['_fits'][label_c]['roles']):
                    votes.setdefault(pid, Counter())[int(r)] += 1
            out['correctedRoles'] = {pid: {'role': tpl['roles'][c.most_common(1)[0][0]], 'line': int(tpl['line'][c.most_common(1)[0][0]]),
                                           'group': tpl['group'][c.most_common(1)[0][0]]} for pid, c in votes.items()}
    return out


def formations(data, team, directions, corrections=()):
    """Windows, stabilized periods with their summaries, and per window the roles of the formation that decides that
    window (the coach's label of its period when there is one, else the stabilized formation; none in an unclear
    window): `windowRoles` [(centre, formation, {id: group}, back line size)], sorted by time, for the defensive unit."""
    windows = formation_windows(data, team, directions)
    labels, periods = stabilize(windows)
    summaries = [_period_summary(windows, p, data, team, corrections) for p in periods]
    roles, window_roles = {}, []
    for k, (p, summary) in enumerate(zip(periods, summaries)):
        corrected = (summary.get('correction') or {}).get('formation') if summary.get('correctedRoles') else None
        for i in p['windows']:
            w = windows[i]
            if w.get('status') != 'ok':
                continue
            label = labels[i]
            decides = corrected if corrected in w['_fits'] else (label if label in w['_fits'] else None)
            if decides is None:
                continue
            tpl = TEMPLATES[decides]
            fit = w['_fits'][decides]
            groups = {pid: tpl['group'][r] for pid, r in zip(w['ids'], fit['roles'])}
            window_roles.append((w['centre'], decides, groups, tpl['lines'][0], k))
            if label in TEMPLATES:
                for pid, r in zip(w['ids'], w['_fits'][label]['roles']):
                    roles.setdefault(pid, []).append((w['centre'], TEMPLATES[label]['roles'][r], TEMPLATES[label]['group'][r], label))
    window_roles.sort(key=lambda wr: wr[0])
    mine = [c for c in corrections if c.get('team') == team]
    applied = [any(_overlaps(c, p) for p in periods) for c in mine]
    return {'windows': windows, 'labels': labels, 'periods': summaries, 'roles': roles, 'windowRoles': window_roles,
            'expected': data.teams[team]['expected'], 'applied': applied}


# ---------- defensive line ----------
LINE_COHERENT = 8.0           # m: longitudinal spread of the measured defenders up to which the line is coherent
LINE_DISRUPTED = 15.0         # m: ... beyond which (after setting aside one advanced defender) it is disrupted (not in the summary)
DISPLACED_SHARE = .5          # the unit is displaced (a set piece or an all-out attack) when its height is beyond this share of
                              # the team's other outfield players: a defensive line is behind most of its team
TREND_METRES = 3.0            # m over the range: a fitted change at least this large is advancing / retreating
DEPTH_FALLBACK_GAP = 4.0      # m: without a formation, the deepest k players form the unit only if a gap this large follows them


def _defenders_at(t, window_roles, centres=None, memory=ROLE_MEMORY):
    """The defensive unit at time t and its size. The formation window nearest to t decides the period and the
    formation (the coach's label of the period when there is one, else the stabilized formation). Over that period's
    windows of that formation within ROLE_HORIZON seconds of t, the players most often in its back line form the unit,
    exactly as many as the back line has (ties go to the nearest window's back line): a set piece or a run of a few
    seconds does not change who the defenders are, a swap inside the back line keeps both defenders, a sustained swap (a
    centre-back and the holding midfielder exchanging places for half a minute) is followed, and the neighbouring
    periods never outvote a short period or the coach's label. Without a window within ROLE_HORIZON, the nearest one
    within `memory` seconds ('role memory')."""
    if not window_roles:
        return set(), None, None
    centres = centres if centres is not None else [wr[0] for wr in window_roles]
    k = bisect.bisect_left(centres, t)
    j0 = min((j for j in (k-1, k) if 0 <= j < len(window_roles)), key=lambda j: abs(centres[j]-t))
    gap = abs(centres[j0]-t)
    if gap > memory:
        return set(), None, None
    _, formation, nearest, size, period = window_roles[j0]
    if gap > ROLE_HORIZON:
        return {pid for pid, g in nearest.items() if g == 'D'}, size, 'role memory'
    lo = bisect.bisect_left(centres, t-ROLE_HORIZON)
    hi = bisect.bisect_right(centres, t+ROLE_HORIZON)
    votes = {}
    for _, f, groups, _, q in window_roles[lo:hi]:
        if q != period or f != formation:
            continue
        for pid, g in groups.items():
            dv, tv = votes.get(pid, (0, 0))
            votes[pid] = (dv+(g == 'D'), tv+1)
    ranked = sorted((pid for pid, (dv, _) in votes.items() if dv > 0),
                    key=lambda pid: (-votes[pid][0]/votes[pid][1], nearest.get(pid) != 'D', pid))
    return set(ranked[:size]), size, 'formation'


def _depth_fallback(data, team, side, rows):
    """Without any recognised formation: the deepest players over a window (by their mean distance from the own goal),
    3, 4 or 5 of them, where the largest gap behind them is at least DEPTH_FALLBACK_GAP. Only when at least RELIABLE_MIN
    outfield players are present in the window: with fewer, the deepest players on screen need not be the back line
    (on the amateur clip, with five of ten visible, they were not). Low confidence by design."""
    d = data.teams[team]
    if not len(rows):
        return set(), None
    w = data.w[rows]
    observed = d['kind'][rows] == 'O'
    present = (observed*w[:, None]).sum(axis=0)/max(float(w.sum()), 1e-9)
    cols = np.flatnonzero(present >= MIN_PRESENT)
    if len(cols) < RELIABLE_MIN:
        return set(), None
    dist = own_goal_distance(d['x'][np.ix_(rows, cols)], np.full((len(rows), len(cols)), side), data.length)
    mean = np.array([np.average(dist[observed[:, j], k], weights=w[observed[:, j]]) for k, j in enumerate(cols)])
    order = np.argsort(mean)
    gaps = {k: float(mean[order[k]]-mean[order[k-1]]) for k in (3, 4, 5) if k < len(order)}
    if not gaps:
        return set(), None
    k = max(gaps, key=gaps.get)
    if gaps[k] < DEPTH_FALLBACK_GAP:
        return set(), None
    return {d['ids'][cols[i]] for i in order[:k]}, k


def defensive_line(data, team, directions, shape, t0, t1):
    """Per sample in [t0, t1]: the defensive unit (see _defenders_at; without any, the depth-rank fallback over the
    30 s window and the same direction), the measured defenders' distances from the team's own goal line, the
    representative LINE HEIGHT (their median), the DEEPEST DEFENDER height (their minimum), the unit's lateral extent,
    its longitudinal spread, the defenders' mean pitch confidence and lowest identity confidence, and a status:
        complete     every defender of the unit measured;
        partial      one defender missing (at least two measured): the median of the others;
        unavailable  fewer measured, no direction, or no defensive unit known;
        displaced    the unit is not behind most of its team (an attacking set piece, an all-out attack): no line.
    Quality, from the defenders' spread along the pitch: coherent (all within LINE_COHERENT m), uneven (one defender
    stepped up, the others within LINE_COHERENT m: a fullback's overlap), stretched (the others within LINE_DISRUPTED m),
    disrupted (beyond: low confidence, left out of the summary). Goalkeepers never count."""
    d = data.teams[team]
    side_all = directions[team]['side']
    status_all = directions[team]['status']
    rows = data.rows(t0, t1)
    window_roles = shape.get('windowRoles') or []
    centres = [wr[0] for wr in window_roles]
    fallback = {}
    out = []
    for s in rows:
        t = float(data.t[s])
        side = str(side_all[s])
        rec = {'t': round(t, 3), 'w': round(float(data.w[s]), 4)}
        if side not in ('left', 'right'):
            rec.update(status='unavailable', reason='direction contradicted' if status_all[s] == 'contradicted' else 'direction unknown')
            out.append(rec)
            continue
        unit, size, source = _defenders_at(t, window_roles, centres)
        if not unit:
            key = (int((t-data.start)//DIRECTION_WINDOW), side)
            if key not in fallback:
                a = data.start+key[0]*DIRECTION_WINDOW
                win = data.rows(a, a+DIRECTION_WINDOW)
                fallback[key] = _depth_fallback(data, team, side, win[side_all[win] == side])
            unit, size = fallback[key]
            source = 'depth rank' if unit else None
        if not unit or not size:
            rec.update(status='unavailable', reason='no defensive unit')
            out.append(rec)
            continue
        cols = [j for j, pid in enumerate(d['ids']) if pid in unit and np.isfinite(d['x'][s, j])]
        observed = [j for j in cols if d['kind'][s, j] == 'O']
        dist = own_goal_distance(d['x'][s, cols], np.full(len(cols), side), data.length) if cols else np.zeros(0)
        rec.update(unit=int(size), measured=len(cols), observed=len(observed), estimated=len(cols)-len(observed), source=source, side=side,
                   geometric=round(float(d['quality'][s, observed].mean()), 3) if observed else None,
                   identity=round(float(d['identity'][s, observed].min()), 3) if observed else None,
                   defenders=[{'id': d['ids'][j], 'x': round(float(d['x'][s, j]), 4), 'y': round(float(d['y'][s, j]), 4),
                               'distance': round(float(v), 2), 'kind': str(d['kind'][s, j])} for j, v in zip(cols, dist)])
        if len(cols) < 2 or len(cols) < size-1:
            rec.update(status='unavailable', reason=f'{len(cols)} of {size} defenders measured')
            out.append(rec)
            continue
        height = float(np.median(dist))
        others = [j for j, pid in enumerate(d['ids']) if pid not in unit and np.isfinite(d['x'][s, j])]
        if len(others) >= 3:
            ahead = own_goal_distance(d['x'][s, others], np.full(len(others), side), data.length)
            if float((ahead < height).mean()) > DISPLACED_SHARE:
                rec.update(status='displaced', reason='defensive unit ahead of most of its team (set piece or attack)', height=round(height, 2))
                out.append(rec)
                continue
        rec['status'] = 'complete' if len(cols) >= size else 'partial'
        if len(observed) < max(2, size-1):
            rec['status'] = 'estimated'      # estimated mode: the line stands mostly on predicted defenders
        rec['height'] = round(height, 2)
        rec['deepest'] = round(float(dist.min()), 2)
        spread = float(dist.max()-dist.min())
        core = float(np.ptp(np.sort(dist)[:-1])) if len(dist) >= 3 else spread
        rec['spread'] = round(spread, 2)
        rec['advanced'] = int((dist > height+LINE_COHERENT).sum())
        rec['quality'] = ('coherent' if spread <= LINE_COHERENT else 'uneven' if core <= LINE_COHERENT else
                          'stretched' if core <= LINE_DISRUPTED else 'disrupted')
        ys = d['y'][s, cols]
        rec['lateral'] = [round(float(ys.min()), 4), round(float(ys.max()), 4)]
        out.append(rec)
    return out


def line_summary(series):
    """The line over a range: the time-weighted mean and median of the representative height (measured, non-disrupted
    samples) and of the deepest defender (complete samples only: with one defender unmeasured he may be the deepest),
    the shares of time measured, complete, disrupted, displaced and mostly estimated, the quality shares, and the trend (a
    weighted linear fit of the height against time: advancing, retreating or steady)."""
    good = [r for r in series if r.get('status') in ('complete', 'partial') and r.get('quality') != 'disrupted']
    full = [r for r in good if r['status'] == 'complete']
    total = sum(r['w'] for r in series) or 1.0
    out = {'measuredShare': round(sum(r['w'] for r in good)/total, 3),
           'completeShare': round(sum(r['w'] for r in full)/total, 3),
           'disruptedShare': round(sum(r['w'] for r in series if r.get('quality') == 'disrupted')/total, 3),
           'displacedShare': round(sum(r['w'] for r in series if r.get('status') == 'displaced')/total, 3),
           'estimatedShare': round(sum(r['w'] for r in series if r.get('status') == 'estimated')/total, 3),
           'quality': {q: round(sum(r['w'] for r in good if r.get('quality') == q)/total, 3) for q in ('coherent', 'uneven', 'stretched')},
           'samples': len(good)}
    if not good:
        return dict(out, height=None, deepest=None, trend=None)
    h = np.array([r['height'] for r in good])
    w = np.array([r['w'] for r in good])
    t = np.array([r['t'] for r in good])
    out['height'] = {'mean': _round((h*w).sum()/w.sum()), 'median': _round(_weighted_median(h, w)), 'min': _round(h.min()), 'max': _round(h.max())}
    if full:
        dp = np.array([r['deepest'] for r in full])
        wf = np.array([r['w'] for r in full])
        out['deepest'] = {'mean': _round((dp*wf).sum()/wf.sum()), 'median': _round(_weighted_median(dp, wf)), 'min': _round(dp.min())}
    else:
        out['deepest'] = None
    out['trend'] = None
    if len(good) >= 10 and t[-1]-t[0] >= 10:
        tm = (t*w).sum()/w.sum()
        hm = (h*w).sum()/w.sum()
        slope = float((w*(t-tm)*(h-hm)).sum()/max(float((w*(t-tm)**2).sum()), 1e-9))
        change = slope*(t[-1]-t[0])
        out['trend'] = {'metresPerMinute': round(float(slope*60), 2), 'change': round(float(change), 2),
                        'label': 'advancing' if change >= TREND_METRES else ('retreating' if change <= -TREND_METRES else 'steady')}
    return out


# ---------- the report ----------
CHART_POINTS = 1500           # series sent for charts are thinned to at most this many samples (aggregates use them all)
FORMATION_LABELS = frozenset(FORMATIONS_10+FORMATIONS_9)


def clean_settings(raw):
    """The team-shape settings of a video (stored apart from the tracking, in the video's run folder): the defending
    direction set by the user, the outfield lineup per team (a number for the whole video, or {'outfield': n, 'from': t}
    from a time on, e.g. a dismissal), and formation labels set by the user for periods. Anything malformed is dropped,
    never raised."""
    raw = raw if isinstance(raw, dict) else {}
    out = {'direction': {'A': None, 'switchAt': None}, 'lineup': {'A': None, 'B': None}, 'formations': []}
    direction = raw.get('direction') if isinstance(raw.get('direction'), dict) else {}
    if isinstance(direction.get('A'), str) and direction['A'] in ('left', 'right'):
        out['direction']['A'] = direction['A']
        sw = direction.get('switchAt')
        if _number(sw) and sw >= 0:
            out['direction']['switchAt'] = round(float(sw), 2)
    lineup = raw.get('lineup') if isinstance(raw.get('lineup'), dict) else {}
    for team in ('A', 'B'):
        v = lineup.get(team)
        if isinstance(v, int) and not isinstance(v, bool) and 7 <= v <= 10:
            out['lineup'][team] = v
        elif isinstance(v, dict) and isinstance(v.get('outfield'), int) and not isinstance(v.get('outfield'), bool) and 7 <= v['outfield'] <= 10:
            if _number(v.get('from')) and v['from'] >= 0:
                out['lineup'][team] = {'outfield': v['outfield'], 'from': round(float(v['from']), 2)}
            else:
                out['lineup'][team] = v['outfield']
    labels = raw.get('formations') if isinstance(raw.get('formations'), list) else []
    for c in labels:
        if not isinstance(c, dict) or not isinstance(c.get('team'), str) or c['team'] not in ('A', 'B') \
                or not isinstance(c.get('formation'), str) or c['formation'] not in FORMATION_LABELS:
            continue
        t0, t1 = c.get('t0'), c.get('t1')
        if not (_number(t0) and _number(t1)) or t1 <= t0:
            continue
        out['formations'].append({'team': c['team'], 't0': round(float(t0), 2), 't1': round(float(t1), 2), 'formation': c['formation'],
                                  'predicted': c.get('predicted') if isinstance(c.get('predicted'), str) else None,
                                  'phase': 'observed', 'created': str(c.get('created') or '')[:40]})
    return out


def _thin(series, limit=CHART_POINTS):
    """At most `limit` rows for a chart, and the step used (1 = every sample)."""
    if len(series) <= limit:
        return series, 1
    step = int(math.ceil(len(series)/limit))
    return series[::step], step


class TeamShape:
    """Team shape of one result for one position mode and the video's settings: computed once (the formation windows and
    the directions over the whole segment), reported for any team and range."""

    def __init__(self, result, samples=None, mode='observed', settings=None):
        self.settings = clean_settings(settings)
        samples = samples if samples is not None else analytics.Samples(result)
        lineup = self.settings['lineup']
        self.observed = TeamShapeData(result, samples, 'observed', lineup=lineup)
        self.data = self.observed if mode == 'observed' else TeamShapeData(result, samples, mode, lineup=lineup)
        self.mode = mode
        self.directions = team_directions(self.observed, result, self.settings['direction'])
        corrections = self.settings['formations']
        # Formations are always recognised from observed positions only (no circular confirmation by estimates).
        self.shapes = {team: formations(self.observed, team, self.directions, corrections) for team in ('A', 'B')}
        self.samples_excluded = dict(samples.excluded)

    def clamp(self, t0=None, t1=None):
        """The requested range within the analysed segment: (t0, t1, note). A reversed range is swapped ('reversed'),
        a range reaching beyond the segment is cut to it ('clipped'), and a range entirely outside it gives no samples
        at all ('outside', t0 > t1 so no row matches), never a boundary sample standing for it."""
        start, end = self.data.start, self.data.end
        a = start if t0 is None or not _number(t0) else float(t0)
        b = end if t1 is None or not _number(t1) else float(t1)
        note = None
        if b < a:
            a, b, note = b, a, 'reversed'
        if b < start-1e-6 or a > end+1e-6:
            return a, b, 'outside'
        if a < start-1e-6 or b > end+1e-6:
            note = note or 'clipped'
        return max(a, start), min(b, end), note

    def _direction_report(self, team, rows):
        dr = self.directions[team]
        st = Counter(dr['status'][rows].tolist()) if len(rows) else Counter()
        sides = Counter(s for s in dr['side'][rows].tolist() if s) if len(rows) else Counter()
        w = self.data.w[rows] if len(rows) else np.zeros(0)
        total = float(w.sum()) or 1.0
        known = float(w[dr['side'][rows] != ''].sum()) if len(rows) else 0.0
        return {'side': sides.most_common(1)[0][0] if sides else None, 'sides': dict(sides), 'source': dr['source'], 'base': dr['base'],
                'knownShare': round(known/total, 3) if len(rows) else 0.0, 'status': dict(st), 'switchAt': self.directions.get('switchAt'),
                'contradicted': [{'t0': wd['t0'], 't1': wd['t1']} for wd in self.directions['windows'] if team in wd.get('contradicted', [])],
                'windows': self.directions['windows']}

    def report(self, team, t0=None, t1=None):
        a, b, note = self.clamp(t0, t1)
        data, d = self.data, self.data.teams[team]
        if note == 'outside':
            rows = np.zeros(0, np.int64)
            wd = {'series': [], 'width': _aggregate([], 'width'), 'depth': _aggregate([], 'depth')}
            line = []
        else:
            rows = data.rows(a, b)
            wd = team_width_depth(data, team, a, b)
            line = defensive_line(data, team, self.directions, self.shapes[team], a, b)
        shape = self.shapes[team]
        periods = []
        if note != 'outside':
            for p in shape['periods']:
                if p['t1'] < a or p['t0'] > b:
                    continue
                q = dict(p)
                q['clipped'] = [round(max(a, p['t0']), 2), round(min(b, p['t1']), 2)]
                periods.append(q)
        keys = ('t0', 't1', 'centre', 'status', 'players', 'expected', 'best', 'second', 'margin', 'spread', 'identity', 'excess')
        windows = [{k: w.get(k) for k in keys} | {'label': shape['labels'][i]}
                   for i, w in enumerate(shape['windows']) if note != 'outside' and w['t1'] >= a and w['t0'] <= b]
        observed_counts = (d['kind'][rows] == 'O').sum(axis=1) if len(rows) else np.zeros(0)
        expected_rows = d['expected_t'][rows] if len(rows) else np.zeros(0)
        w = data.w[rows]
        total = float(w.sum()) or 1.0
        excluded = dict(self.samples_excluded)
        for k, v in d['excluded'].items():
            excluded[k] = excluded.get(k, 0)+int(v)
        wd_rows, wd_step = _thin(wd['series'])
        line_rows, line_step = _thin(line)
        lineup_now = sorted(set(expected_rows.tolist())) if len(rows) else [d['expected']]
        range_lineup = int(expected_rows[0]) if len(rows) else d['expected']
        team_labels = [c for c in self.settings['formations'] if c['team'] == team]
        return {
            'team': team, 'mode': self.mode,
            'range': {'t0': round(a, 3), 't1': round(b, 3), 'segment': [round(data.start, 3), round(data.end, 3)], 'note': note,
                      'requested': [t0 if _number(t0) else None, t1 if _number(t1) else None], 'samples': int(len(rows))},
            'pitch': {'length': data.length, 'width': data.width, 'source': data.size_source, 'estimated': data.metres_estimated},
            'direction': self._direction_report(team, rows),
            'coverage': {'expected': range_lineup, 'lineups': lineup_now, 'lineupChange': d['lineupChange'], 'default': d['default'],
                         'identities': len(d['ids']),
                         'meanObserved': round(float((observed_counts*w).sum()/total), 2) if len(rows) else 0.0,
                         'fullShare': round(float(w[observed_counts >= expected_rows].sum()/total), 3) if len(rows) else 0.0,
                         'reliableShare': round(float(w[observed_counts >= RELIABLE_MIN].sum()/total), 3) if len(rows) else 0.0,
                         'goalkeeperShare': round(float(w[np.isfinite(d['gk'][rows, 0])].sum()/total), 3) if len(rows) else 0.0},
            'chartStep': round(max(wd_step, line_step)/data.rate, 3),
            'widthDepth': {'width': wd['width'], 'depth': wd['depth'], 'step': round(wd_step/data.rate, 3),
                           'series': [{k: r.get(k) for k in ('t', 'players', 'observed', 'estimated', 'expected', 'widthStatus', 'depthStatus',
                                                              'robustWidth', 'robustDepth', 'geometric', 'identity')}
                                      | {'width': r['width']['metres'] if r.get('width') else None, 'depth': r['depth']['metres'] if r.get('depth') else None}
                                      for r in wd_rows]},
            'line': {'summary': line_summary(line), 'step': round(line_step/data.rate, 3),
                     'series': [{k: r.get(k) for k in ('t', 'status', 'height', 'deepest', 'spread', 'quality', 'advanced', 'measured', 'observed',
                                                        'estimated', 'unit', 'source', 'reason', 'geometric', 'identity')} for r in line_rows],
                     'sources': dict(Counter(r.get('source') for r in line if r.get('source')))},
            'formation': {'expected': range_lineup, 'candidates': list(candidates(range_lineup)), 'periods': periods, 'windows': windows,
                          'phases': {'available': False, 'reason': 'This result has no reliable ball-possession record, so shapes are '
                                                                       'observed positional structures over both phases.'},
                          'corrections': [dict(c, applied=ok) for c, ok in zip(team_labels, shape['applied'])]},
            'excluded': excluded,
            'rules': {'sampleRate': data.rate, 'reliableMin': RELIABLE_MIN, 'partialMin': PARTIAL_MIN, 'maxSigma': MAX_SIGMA,
                      'duplicateMetres': DUPLICATE_METRES, 'duplicateSamples': DUPLICATE_SAMPLES, 'windowSeconds': WINDOW_SECONDS,
                      'windowStep': WINDOW_STEP, 'formationMaxMissing': FORMATION_MAX_MISSING, 'minPresent': MIN_PRESENT,
                      'unclearError': UNCLEAR_ERROR, 'switchPenalty': SWITCH_PENALTY, 'minSegmentSeconds': MIN_SEGMENT_SECONDS,
                      'ambiguousMargin': AMBIGUOUS_MARGIN, 'clearMargin': CLEAR_MARGIN, 'clearIdentity': CLEAR_IDENTITY,
                      'clearRoleStability': CLEAR_ROLE_STABILITY, 'roleHorizon': ROLE_HORIZON, 'roleMemory': ROLE_MEMORY,
                      'lineCoherent': LINE_COHERENT, 'lineDisrupted': LINE_DISRUPTED, 'robustPercentiles': list(ROBUST_PERCENTILES),
                      'minPitchConfidence': analytics.MIN_PITCH_CONFIDENCE, 'minIdentityConfidence': analytics.MIN_IDENTITY_CONFIDENCE},
            'settings': {'direction': self.settings['direction'], 'lineup': self.settings['lineup']},
        }

    def _period_at(self, team, t):
        for p in self.shapes[team]['periods']:
            if p['t0'] <= t <= p['t1']:
                return p
        return None

    def snapshot(self, team, t):
        """Everything the pitch shows at one sample: the team's measured players (kind O or E), its goalkeeper, the
        opponents' measured outfield players (context), the width / depth extremes, the defensive line, and the roles of
        the formation window around t (under the coach's label of its period when there is one)."""
        data = self.data
        if not len(data.t) or not _number(t):
            return {'t': None}
        s = int(np.argmin(np.abs(data.t-float(t))))
        ts = float(data.t[s])
        d = data.teams[team]
        other = data.teams['B' if team == 'A' else 'A']
        players = [{'id': pid, 'display': d['display'].get(pid, pid), 'x': round(float(d['x'][s, j]), 4), 'y': round(float(d['y'][s, j]), 4),
                    'kind': str(d['kind'][s, j])} for j, pid in enumerate(d['ids']) if np.isfinite(d['x'][s, j])]
        opponents = [{'id': pid, 'x': round(float(other['x'][s, j]), 4), 'y': round(float(other['y'][s, j]), 4), 'kind': str(other['kind'][s, j])}
                     for j, pid in enumerate(other['ids']) if np.isfinite(other['x'][s, j])]
        gk = d['gk'][s]
        wd = width_depth_series(data, team, ts, ts)
        line = defensive_line(data, team, self.directions, self.shapes[team], ts, ts)
        shape = self.shapes[team]
        window = None
        for i, w in enumerate(shape['windows']):
            if w['t0'] <= ts <= w['t1'] and w.get('status') == 'ok' and (window is None or abs(w['centre']-ts) < abs(window[1]['centre']-ts)):
                window = (i, w)
        roles = {}
        formation = None
        period = self._period_at(team, ts)
        if window is not None:
            i, w = window
            label = shape['labels'][i]
            coach = ((period or {}).get('correction') or {}).get('formation')
            name = coach if coach in w['_fits'] else (label if label in w['_fits'] else w['best'])
            fit = w['_fits'][name]
            tpl = TEMPLATES[name]
            roles = {pid: {'role': tpl['roles'][r], 'line': int(tpl['line'][r]), 'group': tpl['group'][r]} for pid, r in zip(w['ids'], fit['roles'])}
            formation = {'label': label, 'fitted': name, 'coach': coach, 'best': w['best'], 'second': w['second'], 'margin': w['margin'],
                         'players': w['players'], 'expected': w['expected'], 't0': w['t0'], 't1': w['t1'],
                         'relativeError': round(fit['relative'], 4), 'rms': round(fit['rms'], 2)}
        side = str(self.directions[team]['side'][s]) or None
        return {'t': round(ts, 3), 'team': team, 'side': side, 'directionStatus': str(self.directions[team]['status'][s]),
                'players': players, 'opponents': opponents,
                'goalkeeper': {'x': round(float(gk[0]), 4), 'y': round(float(gk[1]), 4)} if np.isfinite(gk[0]) else None,
                'widthDepth': wd[0] if wd else None, 'line': line[0] if line else None, 'formation': formation, 'roles': roles,
                'period': {'t0': period['t0'], 't1': period['t1'], 'label': period['label']} if period else None,
                'expected': int(d['expected_t'][s])}
