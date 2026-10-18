"""How much the 2D Match view's player markers jitter, before and after the trajectory stabilizer, and how responsive
they stay, measured from a saved result (read-only: nothing but --json is written).

    python tools/jitter.py RESULT_JSON|RUN_DIR [--window S E] [--json OUT] [--ids A-01 B-07 ? ...] [--hz 60] [--rebuild]

Streams of the same run, each measured for the identified players and, separately, for the '?' markers (tracked
people with a team and no identity, state 'uncertain'):

  RAW         what tactical.js draws without a stabilized block, frozen here as it was before the stabilizer (88be58f,
              checked against that page in node by tests/test_tools.py): the result's timeline block (O = the frames'
              raw pitch positions, I/P/U imputation's estimates), its O samples averaged over a centred 0.14 s window
              that stops at the first non-O neighbour (Track.smoothObserved), sampled by sampleAt (within a quarter frame
              of a sample the sample itself, straight lines inside gaps of at most interpolateGap, the last sample held
              holdAfter, then nothing) with P hidden above uncertainSigma. The '?' markers come from the frames' people,
              averaged over 0.14 s on all their samples (Track.smooth), joined across anonymousGap and held
              anonymousHold. Without a timeline block the identities are built from the frames' people the same way.
  STABILIZED  result.stabilized as the working tree's page draws it: its tracks as stableTracks keeps them (U and
              off-board samples left out; a '?' needs a team, and its bridged samples are kept only between two of its
              observations at most anonymousGap apart) and sampled by sampleStable (no extra smoothing, no quarter-frame
              snap: straight lines between frame samples at every time; only the first sample of a run, and the first
              after a gap longer than the gap limit, shows up to a quarter frame early; between two samples the pair's
              kind and the linear interpolation of their sigma). A result without the block (or whose frameTimes do not
              match the frames) gets one built by soccer/stabilized.py (pipeline.stabilized_block in older trees) when
              soccer/trajectory.py has it; otherwise the stream is reported missing and only RAW is measured. On a
              stored result the block is built on the result's own homographies: STABILIZED is the stabilizer alone.
  With --rebuild (a run directory: its registration.json holds the tracker's frame log and anchors) two more, in memory:
  the registration rebuilt by the working tree's soccer/registration.py (build_timeline, remap_frames: what pass 3 of a
  re-run computes, without the detector), the timeline estimated again (imputation.estimate) and the block built on it.
  REGISTRATION is RAW drawn from those positions, STABILIZED+REGISTRATION the block drawn as above: what a re-run of
  the clip would draw with the working tree's registration, for trying a registration change without the GPU (with
  today's registration, which is the committed one, they reproduce RAW and STABILIZED). The registration snaps at
  anchor frames; the stabilizer spreads those steps (soccer/trajectory.py), and a stabilized stream that still jumps
  together at listed anchors is flagged in the table.

Both streams are sampled at display times (--hz, 1x playback, the first display time a quarter period after the first
frame) and at frame times; the tables show the display times and the stationary step at both.

Every drawn sample is either observed (the observation itself, or between two observations of consecutive frames) or an
estimate (held after the last sample, a straight line across a missing frame or a gap, imputation's I and P). The two
are measured apart, each on its own segments (a cut wherever the part changes); the header gives the estimates' share of
the drawn samples. 'Standing still looks still' is the observed part: an estimate held or predicted is still by
construction.

Definitions (the renderer investigation's, so the numbers compare with its report; positions in board metres:
normalized x*105, y*68, what the board draws)
  series      one marker's positions on a uniform time grid, cut into segments wherever it is not drawn (or its part
              changes); nothing is measured across a cut. Rates are per minute of marker time inside segments.
  step        |p[i] - p[i-1]| between consecutive samples of a segment.
  true speed  |d/dt| of a Savitzky-Golay fit (order 2, the odd number of samples nearest SG_SECONDS); 'stationary' =
              fitted speed < STATIONARY at both ends of a step. Segments shorter than the fit are left out of the
              residual and stationary statistics (kept in the others).
  residual    |p[i] - fit[i]|: the high-frequency part of the path. Reported as RMS, all samples and stationary ones.
  accLag      |p[i+m] - 2 p[i] + p[i-m]| / (m dt)^2, m = round(LAG / dt): acceleration over a fixed lag (comparable
              across rates); the share of samples above each ACC_LEVELS. accLag250: the same with LAG_LONG, share
              above ACC_LONG_LEVEL (white noise of sigma s gives about 39 s m/s^2 per metre there).
  reversal    u = p[i] - p[i-m], w = p[i+m] - p[i] (m as accLag): u.w < 0 with |u|, |w| >= thr (REVERSAL); consecutive
              flagged samples are one event.
  lag speed   |p[i+m] - p[i]| / (m dt) > SPEED_MAX: events (consecutive samples are one).
  jump        a step > JUMP.
Every one of those rows improves as a smoother gets more sluggish (a 1 s moving average of the stabilized path beats the
stabilizer on all of them on the broadcast run): they show how still the markers are, not whether the motion survived,
and cannot rank smoothers alone. Against the raw observations (players, at frame times, observed and estimated alike):
  lag         the shift maximising the correlation of the stream's velocity with the raw observations' (central
              differences over +-DIFF, both less their moving mean over +-LAG_HIGH_PASS, the raw's changes above
              SPEED_MAX left out as box events): positive = the stream is late. A centred smoother has none; a
              stream whose velocity changes correlate less than LAG_MIN_CORRELATION with the raw ones has lost them
              (the lag is then not shown: see the rounding).
  rounding    at the turns and stops of the raw observations (found on their SG_SECONDS fit: a direction change over
              TURN_ANGLE between the TURN_HALF before and after, both faster than TURN_SPEED, at its sharpest; a fall
              from STOP_FROM within the second before to under STOP_AT, staying under STOP_HOLD for a second) the RMS
              distance, over +-CHANGE_SPAN, between the stream and the raw observations' local mean (+-LOCAL, at least
              LOCAL_MIN of them): a stream that rounds turns or slows early departs from it there. RAW sits near zero by
              construction, an exact smoother at the local mean's own noise (noise below: the raw residual RMS over
              the square root of the local mean's samples), a sluggish one above it. Box events in the raw make turns
              of their own and move the local mean (a smoother that rightly ignores them departs from it there), so
              the rounding is also given at the turns and stops with no box-geometry flag (BOX_EVENTS) of the run's
              stabilized block within BOX_EVENT_NEAR, the same ones for RAW and STABILIZED (and for REGISTRATION and
              STABILIZED+REGISTRATION). The rows compare streams on the same footage; the accuracy itself is
              tools/trajectory_bench.py's.
  correction  |stabilized - raw observation| on the frames where the timeline (or, for '?', the frame) has a raw
              observation and the stabilized stream a position: median, p90, max.
  coverage    drawn marker-minutes of each stream; what the stabilized stream made of the raw observations (its kind
              there); the share of its observation weights that were rejected (w = 0) or down-weighted (0 < w < 1),
              and its flags by reason.
  common mode the calibration-instability diagnostic, on each stream's observed samples at frame times: per player the
              residual displacement e = r[i] - r[i-1] of consecutive frames (r: the residual above); per frame pair
              with at least COMMON_MIN_PLAYERS players the common part c = mean(e), its share N |c|^2 / sum |e|^2 of the
              pair's residual displacement power (1/N for independent noise), pooled over pairs, and the median |c| on
              the pairs at a listed anchor against the others. A homography step is not a translation (it moves the
              players by different amounts across the pitch), so the jumps together are found per player instead: a
              player jumps on a pair when |e| exceeds JUMP_FACTOR times its own median |e| and JUMP_FLOOR; a pair where
              at least TOGETHER of its players jump is listed with the frames' calibration (source, direction, anchor,
              age) and the listed anchor it sits on, if any, and counted against the number independent jumps at the
              run's pooled jump rate would give (binomial).
"""
import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from scipy.signal import savgol_filter
from scipy.stats import binom

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# tactical.js RULES as committed before the stabilizer (88be58f): RAW is drawn with these, so a later change to the
# page does not move the 'before' numbers.
RULES = {'interpolateGap': 1.0, 'holdAfter': .6, 'anonymousGap': .3, 'anonymousHold': .3, 'smoothWindow': .14, 'outside': .1, 'uncertainSigma': 8.0}
SAME = .25               # frame steps: sampleAt shows a sample as is this close to its time (RAW) ...
EARLY = .25              # ... and sampleStable shows the first sample of a run this early (STABILIZED)
CONSECUTIVE = 1.5        # frame steps: without frame indices, two samples this close are of consecutive frames (the page's rule)
L_BOARD, W_BOARD = 105.0, 68.0   # m: the board the normalized positions are drawn on
HZ = 60.0                # display refresh rate of the drawn positions (1x playback)
PHASE = .25              # display periods: the first display time after the first frame
SG_SECONDS = 1.0         # s: the Savitzky-Golay fit (order 2) the residual is measured against
STATIONARY = 1.0         # m/s: 'stationary' below this fitted speed
LAG = .1                 # s: the lag of accLag, lag speed and reversals
LAG_LONG = .25           # s: the lag of accLag250
REVERSAL = (.10, .25)    # m: reversal thresholds (1 and 2.5 m/s apparent speed over the lag)
JUMP = 1.0               # m: a jump between consecutive samples
SPEED_MAX = 10.0         # m/s: lag speeds above this are events (faster than a sprint)
ACC_LEVELS = (20.0, 50.0)  # m/s^2: accLag shares reported (8 saturates: over half the samples on every run, broadcast included)
ACC_LONG_LEVEL = 8.0     # m/s^2: accLag250 share reported (physically meaningful over 0.25 s)
# Responsiveness against the raw observations (tools/trajectory_bench.py's lag, turn and stop definitions).
DIFF = .1                # s: half lag of the central differences of the velocity
LAG_MAX = .5             # s: shifts searched for the lag
LAG_HIGH_PASS = .5       # s: half window of the moving mean taken off both velocities (the changes of velocity peak sharply)
# The lag is reported where the velocity changes still correlate this much: a 1 s moving average of the stabilized path
# keeps 0.09-0.23 (amateur, broadcast run), its peak then lies anywhere in +-LAG_MAX; the stabilizer keeps 0.38-0.61.
LAG_MIN_CORRELATION = .25
TURN_ANGLE, TURN_SPEED, TURN_HALF = 60.0, 2.0, .5        # degrees, m/s, s
STOP_FROM, STOP_AT, STOP_HOLD, STOP_SECONDS = 2.0, .5, 1.0, 1.0   # m/s, m/s, m/s, s
LOCAL = .1               # s: half window of the raw observations' local mean ...
LOCAL_MIN = 5            # ... over at least this many of them
CHANGE_SPAN = .3         # s: the rounding is measured over this either side of a turn's or stop's frame
# The block's flags that are box geometry (the box against its track's running height), not the trajectory: a turn or stop
# with one of them within BOX_EVENT_NEAR is a box event of the raw rather than a manoeuvre (on the amateur run 71 of the 80
# raw-detected turns, the stabilized path a median 0.65 m off the raw local mean there, 0.19 m at the other 9). A flag of
# the trajectory fit (down-weighted, rejected) never excludes one: a sluggish smoother would hide its own rounding so.
BOX_EVENTS = ('box lost the legs', 'box reaches into a nearer player')
BOX_EVENT_NEAR = .5      # s
# The calibration-instability diagnostic.
COMMON_MIN_PLAYERS = 4   # players observed on both frames of a pair
# A player's jump: for white Gaussian noise |e| is Rayleigh, P(|e| > 3 medians) = exp(-6.2) = 0.2%; the floor keeps a
# broadcast's centimetre noise from counting. The Veo run's keyframe disagreement (13.3 s, 18.8 s) moved 5 of 7 and 3 of 5
# players by over a metre, by 0.5-3.6 m each: no common translation passes a z-test there.
JUMP_FACTOR = 3.0
JUMP_FLOOR = .10         # m (3 m/s apparent at 30 fps)
JUMP_SCALE_MIN = 30      # residual displacements a player needs for its own median
TOGETHER = .6            # share of a pair's players jumping that makes it a jump together
ANCHOR_SLACK = .5        # frame steps: a frame this close to a listed anchor's time is that anchor's frame (times are rounded to 1 ms)
MAX_LISTED = 40          # jumps together listed (largest first); all are counted
KINDS = {'O': 0, 'I': 1, 'P': 2}
PARTS = ('all', 'observed', 'estimated')
LABELS = {'raw': 'RAW', 'stabilized': 'STABILIZED', 'registration': 'REGISTRATION', 'stabilized+registration': 'STABILIZED+REGISTRATION'}


# ---------- tactical.js: Track, buildTimeline, sampleAt, stableTracks, sampleStable ----------
class Track:
    """One marker's samples, as tactical.js keeps them: times, normalized x / y, kind codes (KINDS), sigma (m, NaN
    without one) and the frame index of each sample (None: two samples are of consecutive frames when CONSECUTIVE
    frame steps apart or closer). Samples that do not come after the last one are dropped (Track.push)."""

    def __init__(self, t, x, y, k, s=None, f=None):
        keep, last = [], -math.inf
        for i, ti in enumerate(t):
            if ti > last:
                keep.append(i)
                last = ti
        self.t = np.asarray(t, np.float64)[keep]
        self.x = np.asarray(x, np.float64)[keep]
        self.y = np.asarray(y, np.float64)[keep]
        self.k = np.asarray(k, np.int8)[keep]
        self.s = np.full(len(keep), np.nan) if s is None else np.asarray(s, np.float64)[keep]
        self.f = None if f is None else np.asarray(f, np.int64)[keep]

    def smooth(self, window):
        """Track.smooth: the centred average of all samples within +-window/2 (the summation order of the page)."""
        t, x, y = self.t.tolist(), self.x.tolist(), self.y.tolist()
        n = len(t)
        xs, ys = [0.0]*n, [0.0]*n
        lo = hi = 0
        for i in range(n):
            while t[i]-t[lo] > window/2:
                lo += 1
            hi = max(hi, i)
            while hi+1 < n and t[hi+1]-t[i] <= window/2:
                hi += 1
            sx = sy = 0.0
            for j in range(lo, hi+1):
                sx += x[j]
                sy += y[j]
            xs[i], ys[i] = sx/(hi-lo+1), sy/(hi-lo+1)
        self.x, self.y = np.asarray(xs), np.asarray(ys)
        return self

    def smooth_observed(self, window):
        """Track.smoothObserved: the same average over O samples only, stopping at the first non-O neighbour."""
        t, x, y, k = self.t.tolist(), self.x.tolist(), self.y.tolist(), self.k.tolist()
        n = len(t)
        xs, ys = list(x), list(y)
        for i in range(n):
            if k[i] != 0:
                continue
            sx = sy = 0.0
            m = 0
            j = i
            while j >= 0 and t[i]-t[j] <= window/2 and k[j] == 0:
                sx += x[j]
                sy += y[j]
                m += 1
                j -= 1
            j = i+1
            while j < n and t[j]-t[i] <= window/2 and k[j] == 0:
                sx += x[j]
                sy += y[j]
                m += 1
                j += 1
            xs[i], ys[i] = sx/m, sy/m
        self.x, self.y = np.asarray(xs), np.asarray(ys)
        return self

    def _consecutive(self, a, b, frame_step):
        """Per pair of sample indices: the two samples are of consecutive frames."""
        if self.f is not None:
            return self.f[b] == self.f[a]+1
        return (self.t[b]-self.t[a]) <= CONSECUTIVE*frame_step

    def sample(self, T, frame_step, hold, gap_limit, same=SAME):
        """sampleAt at the times T, then TacticalView.visible with the full lineup off: (x, y, drawn, observed) arrays,
        NaN where nothing is drawn; observed: drawn from observations only (an O sample, or between two O samples of
        consecutive frames). same: the snap to a sample, in frame steps (0 for straight lines everywhere)."""
        T = np.asarray(T, np.float64)
        N, n = len(T), len(self.t)
        x, y = np.full(N, np.nan), np.full(N, np.nan)
        if not n:
            return x, y, np.zeros(N, bool), np.zeros(N, bool)
        kind, sigma, observed = np.full(N, -1, np.int8), np.full(N, np.nan), np.zeros(N, bool)
        t, X, Y, K, S = self.t, self.x, self.y, self.k, self.s
        snap = frame_step*same
        k = np.clip(np.searchsorted(t, T, side='right')-1, 0, n-1)
        dt = T-t[k]
        before = T < t[0]-snap
        at_k = ~before & ((dt < 0) | (dt <= snap))
        rest = ~before & ~at_k
        k1 = np.minimum(k+1, n-1)
        has_next = (k+1) < n
        at_k1 = rest & has_next & (t[k1]-T <= snap)
        gap = t[k1]-t[k]
        lerp = rest & has_next & ~at_k1 & (gap <= gap_limit)
        held = rest & ~at_k1 & ~lerp & (dt <= hold)
        for mask, idx in ((at_k, k), (at_k1, k1), (held, k)):
            i = idx[mask]
            x[mask], y[mask], kind[mask], sigma[mask] = X[i], Y[i], K[i], S[i]
        observed[at_k] = K[k[at_k]] == 0
        observed[at_k1] = K[k1[at_k1]] == 0
        if lerp.any():
            a, b = k[lerp], k1[lerp]
            u = dt[lerp]/gap[lerp]
            x[lerp] = X[a]+(X[b]-X[a])*u
            y[lerp] = Y[a]+(Y[b]-Y[a])*u
            kind[lerp] = np.where((K[a] == 2) | (K[b] == 2), 2, 1)
            sg = np.maximum(np.nan_to_num(S[a], nan=0.0), np.nan_to_num(S[b], nan=0.0))
            sigma[lerp] = np.where(sg == 0, np.nan, sg)
            observed[lerp] = (K[a] == 0) & (K[b] == 0) & self._consecutive(a, b, frame_step)
        drawn = (kind >= 0) & ((kind != 2) | ~np.isfinite(sigma) | (sigma <= RULES['uncertainSigma']))
        x[~drawn], y[~drawn] = np.nan, np.nan
        return x, y, drawn, observed & drawn

    def sample_stable(self, T, frame_step, hold, gap_limit, early=EARLY):
        """sampleStable at the times T, then TacticalView.visible with the full lineup off: (x, y, drawn, observed) as
        sample(). From a quarter frame (early) before the first sample that sample; at a sample's time that sample;
        between two samples at most gap_limit apart the straight line, its kind P when either is P, O when both are O
        and of consecutive frames, else I, its sigma the straight line between theirs; after a longer gap the next
        sample from a quarter frame before it; else the last sample held for hold."""
        T = np.asarray(T, np.float64)
        N, n = len(T), len(self.t)
        x, y = np.full(N, np.nan), np.full(N, np.nan)
        if not n:
            return x, y, np.zeros(N, bool), np.zeros(N, bool)
        kind, sigma, observed = np.full(N, -1, np.int8), np.full(N, np.nan), np.zeros(N, bool)
        t, X, Y, K, S = self.t, self.x, self.y, self.k, self.s
        lead = frame_step*early
        first = (T >= t[0]-lead) & (T <= t[0])
        k = np.clip(np.searchsorted(t, T, side='right')-1, 0, n-1)
        dt = T-t[k]
        after = T > t[0]
        at_k = after & (dt == 0)
        rest = after & ~at_k
        k1 = np.minimum(k+1, n-1)
        has_next = (k+1) < n
        gap = t[k1]-t[k]
        lerp = rest & has_next & (gap <= gap_limit)
        next_early = rest & has_next & ~lerp & (t[k1]-T <= lead)
        held = rest & ~lerp & ~next_early & (dt <= hold)
        for mask, idx in ((first, np.zeros(N, np.int64)), (at_k, k), (next_early, k1), (held, k)):
            i = idx[mask]
            x[mask], y[mask], kind[mask], sigma[mask] = X[i], Y[i], K[i], S[i]
            if mask is not held:
                observed[mask] = K[i] == 0
        if lerp.any():
            a, b = k[lerp], k1[lerp]
            u = dt[lerp]/gap[lerp]
            x[lerp] = X[a]+(X[b]-X[a])*u
            y[lerp] = Y[a]+(Y[b]-Y[a])*u
            both_o = (K[a] == 0) & (K[b] == 0) & self._consecutive(a, b, frame_step)
            kind[lerp] = np.where((K[a] == 2) | (K[b] == 2), 2, np.where(both_o, 0, 1))
            sa, sb = S[a], S[b]
            sigma[lerp] = np.where(np.isfinite(sa) & np.isfinite(sb), sa+(sb-sa)*u, np.where(np.isfinite(sa), sa, sb))
            observed[lerp] = kind[lerp] == 0
        drawn = (kind >= 0) & ((kind != 2) | ~np.isfinite(sigma) | (sigma <= RULES['uncertainSigma']))
        x[~drawn], y[~drawn] = np.nan, np.nan
        return x, y, drawn, observed & drawn


def final_roster(result):
    """tactical.js finalRoster: the identities the board draws (team players and goalkeepers of the final registry)."""
    match = result.get('match') or {}
    ids = {p['id'] for p in match.get('players') or [] if p.get('team') in ('A', 'B')}
    return ids | {p['id'] for p in match.get('goalkeepers') or []}


def _column(values):
    return np.array([np.nan if v is None else v for v in values], np.float64)


def _valid(x, y):
    o = RULES['outside']
    return np.isfinite(x) & np.isfinite(y) & (x >= -o) & (x <= 1+o) & (y >= -o) & (y <= 1+o)


def _sigmas(cols, n):
    """The page's sigma of each sample: the larger of [along, across] (NaN without one)."""
    out = np.full(n, np.nan)
    for i, s in enumerate(cols.get('sigma') or [None]*n):
        if isinstance(s, list):
            got = [v for v in s if v is not None]
            if got:
                out[i] = max(got)
    return out


def _block_tracks(block, times, roster, only=None):
    """Identity tracks from a timeline-shaped block ({'ids': {pid: x, y, kind, sigma}}): what buildTimeline pushes."""
    tracks = {}
    for pid, cols in block['ids'].items():
        if pid not in roster or (only is not None and pid not in only):
            continue
        x, y = _column(cols['x']), _column(cols['y'])
        kind = np.array([KINDS.get(k, -1) for k in cols['kind']], np.int8)
        sigma = _sigmas(cols, len(x))
        ok = (kind >= 0) & _valid(x, y)
        if ok.any():
            frames = np.flatnonzero(ok)
            tracks[pid] = Track(times[ok], x[ok], y[ok], kind[ok], sigma[ok], frames)
    return tracks


def _stable_track(cols, times, anonymous):
    """stableTracks' track of one block column: its frames with a position (no U, on the board); a '?' keeps a bridged
    stretch only between two of its observations at most anonymousGap apart. None when nothing is left."""
    x, y = _column(cols.get('x') or []), _column(cols.get('y') or [])
    kinds = list(cols.get('kind') or [])
    n = min(len(times), len(x), len(y), len(kinds))
    x, y, kinds = x[:n], y[:n], kinds[:n]
    shown = [i for i in range(n) if kinds[i] and kinds[i] != 'U' and kinds[i] in KINDS and _valid(x[i], y[i])]
    if anonymous:
        kept, j, last = [], 0, -1
        while j < len(shown):
            if kinds[shown[j]] == 'O':
                last = shown[j]
                kept.append(shown[j])
                j += 1
                continue
            nxt = j
            while nxt < len(shown) and kinds[shown[nxt]] != 'O':
                nxt += 1
            if last >= 0 and nxt < len(shown) and times[shown[nxt]]-times[last] <= RULES['anonymousGap']:
                kept += shown[j:nxt]
            j = nxt
        shown = kept
    if not shown:
        return None
    i = np.array(shown, np.int64)
    kind = np.array([KINDS[kinds[j]] for j in shown], np.int8)
    return Track(times[i], x[i], y[i], kind, _sigmas(cols, len(times))[i] if cols.get('sigma') else None, i)


def _stable_tracks(block, times, roster, only=None):
    """STABILIZED: (identity tracks, '?' tracks) of a stabilized block as stableTracks keeps them."""
    ident, anon = {}, {}
    for pid, cols in (block.get('ids') or {}).items():
        if pid in roster and (only is None or pid in only):
            track = _stable_track(cols, times, False)
            if track is not None:
                ident[pid] = track
    for key, cols in (block.get('anonymous') or {}).items():
        if cols.get('team') in ('A', 'B'):
            track = _stable_track(cols, times, True)
            if track is not None:
                anon[str(key)] = track
    return ident, anon


def people_samples(result, roster):
    """The frames' raw positions: ({pid: (frame indices, x, y)} of roster identities, {track: ...} of the '?' people)."""
    ident, anon = {}, {}
    for i, f in enumerate(result.get('frames') or []):
        for p in f.get('people') or []:
            pitch = p.get('pitch')
            if not (isinstance(pitch, list) and len(pitch) == 2 and all(isinstance(v, (int, float)) for v in pitch)):
                continue
            if not _valid(np.float64(pitch[0]), np.float64(pitch[1])):
                continue
            if not p.get('id'):
                if p.get('state') == 'uncertain' and p.get('team') in ('A', 'B'):
                    anon.setdefault(str(p.get('track')), []).append((i, pitch[0], pitch[1]))
            elif p['id'] in roster:
                ident.setdefault(p['id'], []).append((i, pitch[0], pitch[1]))
    as_arrays = lambda d: {k: (np.array([r[0] for r in v]), np.array([r[1] for r in v], np.float64), np.array([r[2] for r in v], np.float64))
                           for k, v in d.items()}
    return as_arrays(ident), as_arrays(anon)


def _timeline_block(result):
    frames = result.get('frames') or []
    tl = result.get('timeline') or {}
    return tl if tl.get('ids') and tl.get('frameTimes') and len(tl['frameTimes']) == len(frames) else None


def raw_stream(result, times, roster, people, only=None):
    """RAW: (identity tracks, '?' tracks) as the committed tactical.js builds and smooths them."""
    block = _timeline_block(result)
    if block:
        ident = {pid: tr.smooth_observed(RULES['smoothWindow']) for pid, tr in _block_tracks(block, times, roster, only).items()}
    else:
        ident = {pid: Track(times[i], x, y, np.zeros(len(i), np.int8), f=i).smooth(RULES['smoothWindow'])
                 for pid, (i, x, y) in people[0].items() if only is None or pid in only}
    anon = {key: Track(times[i], x, y, np.zeros(len(i), np.int8), f=i).smooth(RULES['smoothWindow']) for key, (i, x, y) in people[1].items()}
    return ident, anon


def raw_observations(result, roster, people, only=None):
    """The identities' raw observations {pid: (frame indices, x, y)} normalized: the timeline's O samples (the frames'
    pitch positions), else the frames' people."""
    block = _timeline_block(result)
    if not block:
        return {pid: v for pid, v in people[0].items() if only is None or pid in only}
    out = {}
    for pid, cols in block['ids'].items():
        if pid not in roster or (only is not None and pid not in only):
            continue
        i = np.flatnonzero(np.array([k == 'O' for k in cols['kind']], bool))
        x, y = _column(cols['x'])[i], _column(cols['y'])[i]
        ok = _valid(x, y)
        out[pid] = (i[ok], x[ok], y[ok])
    return out


# ---------- the stabilized block ----------
def _builder():
    """(build, None) with build(result, width, height) -> (block, summary): soccer/stabilized.py's assembly, else the
    pipeline's older name for it; (None, why) when neither is importable."""
    try:
        from soccer import stabilized
        return stabilized.block, None
    except ImportError as error:
        first = error
    try:
        import pipeline                # imports torch, does not touch the GPU
    except ImportError as error:
        return None, f'the stabilizer is not importable ({first}; {error})'
    build = getattr(pipeline, 'stabilized_block', None)
    return (build, None) if build is not None else (None, f'neither soccer/stabilized.py ({first}) nor pipeline.stabilized_block')


def stabilized_block(result, compute=True):
    """(block, source): result.stabilized when its frameTimes match the frames ('result'), else the block the
    assembly builds from soccer/trajectory.py ('computed'), else (None, why it is missing)."""
    frames = result.get('frames') or []
    block = result.get('stabilized')
    if isinstance(block, dict) and isinstance(block.get('ids'), dict) and len(block.get('frameTimes') or []) == len(frames):
        return block, 'result'
    why = 'no stabilized block in the result' if not block else 'the stabilized block does not match the frames'
    if not compute:
        return None, why
    build, missing = _builder()
    if build is None:
        return None, f'{why}; {missing}'
    video = result.get('video') or {}
    try:
        block, _ = build(result, int(video.get('width') or 1920), int(video.get('height') or 1080))
    except Exception as error:   # noqa: BLE001 - as in the pipeline: a failing stabilizer leaves RAW to measure
        return None, f'{why}; building one failed ({type(error).__name__}: {error})'
    return block, 'computed'


def rebuild(run, result):
    """(rebuilt result, None), (rebuilt result without a block, why) or (None, why): a copy of the result with its
    registration rebuilt from the run's registration.json by the working tree's soccer/registration.py (build_timeline
    over the whole log, the run's slice of it remapped onto its frames: people's and the ball's pitch positions), its
    timeline estimated again and its stabilized block built on that. In memory: nothing is written. The identities are
    not touched."""
    path = Path(run)/'registration.json'
    if not path.exists():
        return None, f'no registration.json in {run}'
    sys.path.insert(0, str(ROOT/'tools'))
    try:
        import holdout                 # the hold-out tool's readers of the saved frame log and anchors
        from soccer import field, imputation, registration
    except ImportError as error:
        return None, f'the registration is not importable ({error})'
    record = json.loads(path.read_text(encoding='utf-8'))
    out = copy.deepcopy(result)
    out.pop('stabilized', None)
    frames = out.get('frames') or []
    W, H = int(record['width']), int(record['height'])
    size = field.PitchSize.from_json(record.get('size'))
    offset = int(record.get('offset') or 0)
    timeline = registration.build_timeline(holdout.log_of(record), holdout.anchors_of(record), W, H, size)[offset:offset+len(frames)]
    if len(timeline) != len(frames):
        return None, f'the registration log covers {len(timeline)} of the {len(frames)} frames'
    registration.remap_frames(frames, timeline, W, H, size)
    pitch = (out.get('tactical') or {}).get('pitchSize') or {}
    out['timeline'] = imputation.estimate(out, float(pitch.get('length', 105.0)), float(pitch.get('width', 68.0)))
    block, source = stabilized_block(out, True)
    if block is None:
        return out, f'the rebuilt result has no stabilized block: {source}'
    out['stabilized'] = block
    return out, None


# ---------- metrics ----------
def segments(valid):
    """[a, b) ranges of consecutive True samples."""
    v = np.concatenate([[False], np.asarray(valid, bool), [False]]).astype(np.int8)
    d = np.diff(v)
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def _events(flag):
    """Runs of consecutive True samples."""
    return int(np.count_nonzero(np.diff(np.concatenate([[0], np.asarray(flag, np.int8)])) == 1))


def _fit_window(dt):
    return int(round(SG_SECONDS/dt)) | 1


def _acc_key(level, long=False):
    return f"accLag{'250' if long else ''}Over{level:g}Share"


class Pool:
    """Metric samples pooled over the segments of a group of markers."""

    def __init__(self):
        self.seconds = 0.0
        self.lists = {k: [] for k in ('resid', 'residStat', 'stepStat', 'accLag', 'accLong')}
        self.counts = dict.fromkeys(('jumps', 'lagSpeedEvents', 'rev10', 'rev25'), 0)

    def add(self, P, dt):
        n = len(P)
        self.seconds += n*dt
        if n < 2:
            return
        d = np.linalg.norm(np.diff(P, axis=0), axis=1)
        self.counts['jumps'] += int(np.count_nonzero(d > JUMP))
        m = max(1, int(round(LAG/dt)))
        if n > 2*m:
            self.lists['accLag'].append(np.linalg.norm(P[2*m:]-2*P[m:-m]+P[:-2*m], axis=1)/(m*dt)**2)
            u, w = P[m:-m]-P[:-2*m], P[2*m:]-P[m:-m]
            back = (u*w).sum(1) < 0
            nu, nw = np.linalg.norm(u, axis=1), np.linalg.norm(w, axis=1)
            self.counts['rev10'] += _events(back & (nu >= REVERSAL[0]) & (nw >= REVERSAL[0]))
            self.counts['rev25'] += _events(back & (nu >= REVERSAL[1]) & (nw >= REVERSAL[1]))
        if n > m:
            self.counts['lagSpeedEvents'] += _events(np.linalg.norm(P[m:]-P[:-m], axis=1)/(m*dt) > SPEED_MAX)
        m4 = max(1, int(round(LAG_LONG/dt)))
        if n > 2*m4:
            self.lists['accLong'].append(np.linalg.norm(P[2*m4:]-2*P[m4:-m4]+P[:-2*m4], axis=1)/(m4*dt)**2)
        W = _fit_window(dt)
        if n >= W >= 5:
            fit = savgol_filter(P, W, 2, axis=0, mode='interp')
            speed = np.linalg.norm(savgol_filter(P, W, 2, deriv=1, delta=dt, axis=0, mode='interp'), axis=1)
            resid = np.linalg.norm(P-fit, axis=1)
            still = speed < STATIONARY
            self.lists['resid'].append(resid)
            self.lists['residStat'].append(resid[still])
            self.lists['stepStat'].append(d[still[1:] & still[:-1]])

    def summary(self):
        cat = {k: (np.concatenate(v) if v else np.zeros(0)) for k, v in self.lists.items()}
        minutes = self.seconds/60
        per_minute = lambda c: round(c/minutes, 2) if minutes else None
        rms = lambda a: round(float(np.sqrt(np.mean(a**2)))*100, 2) if len(a) else None
        pct = lambda a, p: round(float(np.percentile(a, p))*100, 2) if len(a) else None
        share = lambda a, level: round(float(np.mean(a > level)), 4) if len(a) else None
        c = self.counts
        return {'minutes': round(minutes, 3), 'residualRms_cm': rms(cat['resid']), 'residualStationaryRms_cm': rms(cat['residStat']),
                'reversals25PerMin': per_minute(c['rev25']), 'reversals10PerMin': per_minute(c['rev10']),
                'jumpsOver1mPerMin': per_minute(c['jumps']), 'lagSpeedOver10EventsPerMin': per_minute(c['lagSpeedEvents']),
                **{_acc_key(level): share(cat['accLag'], level) for level in ACC_LEVELS},
                _acc_key(ACC_LONG_LEVEL, True): share(cat['accLong'], ACC_LONG_LEVEL),
                'stationaryStep_cm': {'n': int(len(cat['stepStat'])), 'p50': pct(cat['stepStat'], 50), 'p90': pct(cat['stepStat'], 90)}}


def _draw(track, T, frame_step, hold, gap_limit, stable):
    return track.sample_stable(T, frame_step, hold, gap_limit) if stable else track.sample(T, frame_step, hold, gap_limit)


def measure(tracks, T, dt, frame_step, hold, gap_limit, stable, window=None):
    """({part: pooled summary}, {key: summary of all its drawn samples}, the estimates' share of the drawn samples) of the
    markers drawn at the times T. stable: drawn by sampleStable (else sampleAt)."""
    pools, per = {part: Pool() for part in PARTS}, {}
    inside = np.ones(len(T), bool) if window is None else (T >= window[0]) & (T <= window[1])
    drawn_n = estimated_n = 0
    for key, tr in tracks.items():
        x, y, drawn, observed = _draw(tr, T, frame_step, hold, gap_limit, stable)
        drawn &= inside
        drawn_n += int(drawn.sum())
        estimated_n += int((drawn & ~observed).sum())
        own = Pool()
        P = np.stack([x*L_BOARD, y*W_BOARD], 1)
        for part, mask in (('all', drawn), ('observed', drawn & observed), ('estimated', drawn & ~observed)):
            for a, b in segments(mask):
                pools[part].add(P[a:b], dt)
                if part == 'all':
                    own.add(P[a:b], dt)
        if own.seconds:
            per[key] = own.summary()
    return {part: pool.summary() for part, pool in pools.items()}, per, round(estimated_n/drawn_n, 4) if drawn_n else None


# ---------- responsiveness against the raw observations ----------
def _velocity(P, m, dt):
    v = np.full_like(P, np.nan)
    if len(P) > 2*m:
        v[m:-m] = (P[2*m:]-P[:-2*m])/(2*m*dt)
    return v


def _moving_mean(P, half, need):
    """(T, 2) centred mean over +-half samples where at least need of them are finite, else NaN."""
    ok = np.isfinite(P[:, 0])
    c = np.concatenate([np.zeros((1, 2)), np.cumsum(np.where(ok[:, None], P, 0.0), axis=0)])
    n = np.concatenate([[0], np.cumsum(ok)])
    T = len(P)
    lo, hi = np.clip(np.arange(T)-half, 0, T), np.clip(np.arange(T)+half+1, 0, T)
    count = n[hi]-n[lo]
    return np.where((count >= need)[:, None], (c[hi]-c[lo])/np.maximum(count, 1)[:, None], np.nan), count, hi-lo


def _high_pass(v, half):
    """Velocities minus their centred mean over +-half samples, NaN where the window is not whole."""
    mean, count, width = _moving_mean(v, half, 1)
    return np.where((np.isfinite(v[:, 0]) & (count == width))[:, None], v-mean, np.nan)


def motion_changes(R, frame_step):
    """(turn frames, stop frames) of one identity's raw observations R (T, 2) metres, NaN where unobserved: on the
    SG_SECONDS fit of each contiguous run (see the module docstring)."""
    W = _fit_window(frame_step)
    h, before, after = int(round(TURN_HALF/frame_step)), int(round(1.0/frame_step)), int(round(STOP_SECONDS/frame_step))
    turns, stops = [], []
    for a, b in segments(np.isfinite(R[:, 0])):
        if b-a < max(W, 2*h+1, before+after+1):
            continue
        fit = savgol_filter(R[a:b], W, 2, axis=0, mode='interp')
        speed = np.linalg.norm(savgol_filter(R[a:b], W, 2, deriv=1, delta=frame_step, axis=0, mode='interp'), axis=1)
        u, w = fit[h:-h]-fit[:-2*h], fit[2*h:]-fit[h:-h]
        nu, nw = np.linalg.norm(u, axis=1), np.linalg.norm(w, axis=1)
        angle = np.zeros(b-a)
        cos = (u*w).sum(1)/np.maximum(nu*nw, 1e-9)
        fast = (nu/(h*frame_step) > TURN_SPEED) & (nw/(h*frame_step) > TURN_SPEED)
        angle[h:-h] = np.where(fast, np.degrees(np.arccos(np.clip(cos, -1, 1))), 0.0)
        for t in np.flatnonzero(angle > TURN_ANGLE):
            lo, hi = max(0, t-h), min(b-a, t+h+1)
            if angle[lo:hi].argmax()+lo == t:
                turns.append(a+int(t))
        for t in np.flatnonzero((speed[1:] < STOP_AT) & (speed[:-1] >= STOP_AT))+1:
            if t >= before and t+after < b-a and (speed[t-before:t] >= STOP_FROM).any() and speed[t:t+after+1].max() < STOP_HOLD:
                stops.append(a+int(t))
    return turns, stops


def box_events(block):
    """{pid: sorted frame indices} of a stabilized block's box-geometry flags (BOX_EVENTS); None without a block."""
    if block is None:
        return None
    out = {}
    for f in (block or {}).get('flags') or []:
        if str(f.get('reason') or '').split(':')[0] in BOX_EVENTS and f.get('id') is not None and f.get('frame') is not None:
            out.setdefault(f['id'], []).append(int(f['frame']))
    return {pid: np.array(sorted(v)) for pid, v in out.items()}


def responsiveness(tracks, raw_obs, times, frame_step, hold, gap_limit, stable, inside, events=None):
    """The stream's lag and rounding against the raw observations (see the module docstring), players at frame times.
    events: {pid: frame indices} of box events (box_events): the rounding is also given without the turns and stops
    near one."""
    T = len(times)
    m = max(1, int(round(DIFF/frame_step)))
    half, L = int(round(LAG_HIGH_PASS/frame_step)), int(round(LAG_MAX/frame_step))
    local, span = int(round(LOCAL/frame_step)), int(round(CHANGE_SPAN/frame_step))
    shifts = list(range(-L, L+1))
    corr = np.zeros((len(shifts), 3))
    rounding = {'turns': [], 'stops': [], 'turnsNoBoxEvent': [], 'stopsNoBoxEvent': []}
    near = int(round(BOX_EVENT_NEAR/frame_step))
    noise = []
    for pid, tr in tracks.items():
        if pid not in raw_obs:
            continue
        i, rx, ry = raw_obs[pid]
        R = np.full((T, 2), np.nan)
        R[i, 0], R[i, 1] = rx*L_BOARD, ry*W_BOARD
        R[~inside] = np.nan
        x, y, drawn, _ = _draw(tr, times, frame_step, hold, gap_limit, stable)
        S = np.stack([x*L_BOARD, y*W_BOARD], 1)
        S[~(drawn & inside)] = np.nan
        vS, vR = _high_pass(_velocity(S, m, frame_step), half), _high_pass(_velocity(R, m, frame_step), half)
        ref_ok = np.isfinite(vR[:, 0]) & (np.linalg.norm(np.nan_to_num(vR), axis=1) <= SPEED_MAX)
        for k, s in enumerate(shifts):
            a = vS[max(0, s):T+min(0, s)]
            b, ok_b = vR[max(0, -s):T-max(0, s)], ref_ok[max(0, -s):T-max(0, s)]
            ok = ok_b & np.isfinite(a[:, 0])
            corr[k] += [(a[ok]*b[ok]).sum(), (a[ok]**2).sum(), (b[ok]**2).sum()]
        ref, _, _ = _moving_mean(R, local, LOCAL_MIN)
        W = _fit_window(frame_step)
        for a0, b0 in segments(np.isfinite(R[:, 0])):
            if b0-a0 >= W:
                noise.append(np.linalg.norm(R[a0:b0]-savgol_filter(R[a0:b0], W, 2, axis=0, mode='interp'), axis=1))
        turns, stops = motion_changes(R, frame_step)
        for name, at in (('turns', turns), ('stops', stops)):
            for t in at:
                lo, hi = t-span, t+span+1
                if lo < 0 or hi > T:
                    continue
                d = np.linalg.norm(S[lo:hi]-ref[lo:hi], axis=1)
                if np.isfinite(d).all():
                    rounding[name].append(float(np.sqrt(np.mean(d**2))))
                    flagged = None if events is None else events.get(pid, np.zeros(0, int))
                    if flagged is not None and not ((flagged >= lo-near) & (flagged < hi+near)).any():
                        rounding[f'{name}NoBoxEvent'].append(rounding[name][-1])
    r = corr[:, 0]/np.sqrt(np.maximum(corr[:, 1]*corr[:, 2], 1e-12))
    out = {'lag_ms': None, 'velocityCorrelation': None}
    if corr[:, 1].any() and corr[:, 2].any():
        k = int(np.argmax(r))
        frac = 0.0
        if 0 < k < len(r)-1:
            den = r[k-1]-2*r[k]+r[k+1]
            frac = .5*(r[k-1]-r[k+1])/den if den else 0.0
        out = {'lag_ms': round(float((shifts[k]+frac)*frame_step*1000), 1) if r[k] >= LAG_MIN_CORRELATION else None,
               'velocityCorrelation': round(float(r[k]), 4)}
    resid = np.concatenate(noise) if noise else np.zeros(0)
    out['localMeanNoise_m'] = round(float(np.sqrt(np.mean(resid**2))/math.sqrt(2*local+1)), 3) if len(resid) else None
    for name, values in rounding.items():
        v = np.asarray(values)
        out[name] = {'n': int(len(v)), 'median_m': round(float(np.median(v)), 3) if len(v) else None,
                     'p90_m': round(float(np.percentile(v, 90)), 3) if len(v) else None}
    return out


# ---------- correction, coverage, common mode ----------
def _quantiles(a):
    a = np.asarray(a, np.float64)
    if not len(a):
        return None
    return {'n': int(len(a)), 'median_m': round(float(np.median(a)), 3), 'p90_m': round(float(np.percentile(a, 90)), 3), 'max_m': round(float(a.max()), 3)}


def corrections(raw_obs, stab_cols, inside):
    """|stabilized - raw observation| (board metres) on the raw observations' frames where the stabilized stream has a
    position. raw_obs: {key: (frame indices, x, y)} normalized; stab_cols: {key: cols} of the stabilized block."""
    out = []
    for key, (i, x, y) in raw_obs.items():
        cols = stab_cols.get(key)
        if cols is None:
            continue
        keep = inside[i]
        i, x, y = i[keep], x[keep], y[keep]
        sx, sy = _column(cols['x'])[i], _column(cols['y'])[i]
        ok = np.isfinite(sx) & np.isfinite(sy) & np.array([cols['kind'][j] not in (None, 'U') for j in i], bool)
        out.append(np.hypot((sx[ok]-x[ok])*L_BOARD, (sy[ok]-y[ok])*W_BOARD))
    return _quantiles(np.concatenate(out) if out else [])


def coverage(raw_obs, block, roster, inside, only=None):
    """What the stabilized stream made of the raw identity observations, and its observation weights and flags."""
    kinds, weights = {}, []
    total = 0
    for pid, (i, _, _) in raw_obs.items():
        cols = block['ids'].get(pid)
        i = i[inside[i]]
        total += len(i)
        for j in i:
            k = cols['kind'][j] if cols is not None else None
            kinds[k or 'none'] = kinds.get(k or 'none', 0)+1
    for pid, cols in block['ids'].items():
        if pid in roster and (only is None or pid in only) and cols.get('w') is not None:
            weights += [w for w, keep in zip(cols['w'], inside) if keep and w is not None]
    w = np.asarray(weights, np.float64)
    flags = {}
    for f in block.get('flags') or []:
        if only is None or f.get('id') in only:
            flags[f.get('reason') or '?'] = flags.get(f.get('reason') or '?', 0)+1
    return {'rawObservations': total, 'stabilizedKindAtRawObservations': {k: round(v/max(1, total), 4) for k, v in sorted(kinds.items())},
            'weights': {'n': int(len(w)), 'rejectedShare': round(float(np.mean(w == 0)), 4) if len(w) else None,
                        'downWeightedShare': round(float(np.mean((w > 0) & (w < 1))), 4) if len(w) else None},
            'flags': dict(sorted(flags.items(), key=lambda kv: -kv[1]))}


def observed_positions(cols_by_key, n_frames, observed_kind='O'):
    """(P, T, 2) board metres of each key's observed samples (NaN elsewhere), from timeline-shaped columns."""
    keys = list(cols_by_key)
    out = np.full((len(keys), n_frames, 2), np.nan)
    for j, key in enumerate(keys):
        cols = cols_by_key[key]
        x, y = _column(cols['x']), _column(cols['y'])
        ok = np.array([k == observed_kind for k in cols['kind']]) & _valid(x, y)
        out[j, ok, 0], out[j, ok, 1] = x[ok]*L_BOARD, y[ok]*W_BOARD
    return out


def anchor_list(result):
    """[(time, source)] of the validated anchors the result's summary lists."""
    reg = ((result.get('summary') or {}).get('pitch') or {}).get('registration') or {}
    return [(float(a['time']), a.get('source')) for a in reg.get('anchors', []) if a.get('time') is not None]


def _calibration(frame):
    cal = frame.get('calibration') or {}
    return {k: cal.get(k) for k in ('source', 'direction', 'anchor', 'anchorAge', 'confidence')}


def common_mode(P, result, times, frame_step, inside):
    """The calibration-instability diagnostic on (players, frames, 2) observed positions in metres (see the module
    docstring)."""
    n_players, T, _ = P.shape
    W = _fit_window(frame_step)
    E = np.full((n_players, T, 2), np.nan)
    for j in range(n_players):
        for a, b in segments(np.isfinite(P[j, :, 0]) & inside):
            if b-a < W:
                continue
            r = P[j, a:b]-savgol_filter(P[j, a:b], W, 2, axis=0, mode='interp')
            E[j, a+1:b] = np.diff(r, axis=0)
    ok = np.isfinite(E[:, :, 0])
    count = ok.sum(0)
    use = count >= COMMON_MIN_PLAYERS
    if not use.any():
        return None
    Z = np.where(ok[:, :, None], E, 0.0)
    c = Z.sum(0)/np.maximum(count, 1)[:, None]
    power = (Z**2).sum((0, 2))
    common = count*(c**2).sum(1)
    step = np.linalg.norm(c, axis=1)
    # Jumps per player against its own scale; a pair's share of jumping players among those with a scale.
    size = np.linalg.norm(Z, axis=2)
    scaled = ok.sum(1) >= JUMP_SCALE_MIN
    median = np.array([np.median(size[j, ok[j]]) if scaled[j] else np.nan for j in range(n_players)])
    judged = ok & scaled[:, None]
    jumps = judged & (size > np.maximum(JUMP_FACTOR*median[:, None], JUMP_FLOOR))
    n_judged, n_jumps = judged.sum(0), jumps.sum(0)
    need = np.ceil(TOGETHER*n_judged).astype(int)
    together = use & (n_judged >= COMMON_MIN_PLAYERS) & (n_jumps >= need)
    rate = float(jumps[:, use].sum()/max(1, judged[:, use].sum()))
    eligible = use & (n_judged >= COMMON_MIN_PLAYERS)
    expected = float(binom.sf(need[eligible]-1, n_judged[eligible], rate).sum()) if eligible.any() else 0.0
    anchors = anchor_list(result)
    anchor_times = np.array([a[0] for a in anchors]) if anchors else np.zeros(0)

    def at_anchor(i):
        if not len(anchor_times):
            return None
        for k in (i, i-1):
            d = np.abs(anchor_times-times[k])
            if d.min() <= ANCHOR_SLACK*frame_step:
                a = anchors[int(d.argmin())]
                return {'time': a[0], 'source': a[1], 'frame': 'second' if k == i else 'first'}
        return None
    frames = result['frames']
    pairs = np.flatnonzero(use)
    anchored = np.array([at_anchor(i) is not None for i in pairs]) if len(anchor_times) else np.zeros(len(pairs), bool)
    listed = []
    jump_size = np.where(jumps, size, np.nan)
    for i in sorted(np.flatnonzero(together), key=lambda i: -np.nanmedian(jump_size[:, i]))[:MAX_LISTED]:
        listed.append({'time': round(float(times[i]), 3), 'frame': frames[i].get('frame', int(i)), 'players': int(n_judged[i]), 'jumping': int(n_jumps[i]),
                       'jumpMedian_m': round(float(np.nanmedian(jump_size[:, i])), 3), 'commonStep_m': round(float(step[i]), 3),
                       'commonVector_m': [round(float(v), 3) for v in c[i]], 'share': round(float(common[i]/power[i]), 3) if power[i] else None,
                       'before': _calibration(frames[i-1]), 'after': _calibration(frames[i]), 'anchor': at_anchor(i)})
    minutes = len(pairs)*frame_step/60
    med = lambda a: round(float(np.median(a)), 3) if len(a) else None
    return {'pairs': int(len(pairs)), 'meanPlayers': round(float(count[use].mean()), 1),
            'commonShare': round(float(common[use].sum()/power[use].sum()), 4) if power[use].sum() else None,
            'independentExpectation': round(float(use.sum()/count[use].sum()), 4),
            'commonStepMedian_m': {'atAnchors': med(step[pairs][anchored]), 'elsewhere': med(step[pairs][~anchored]), 'anchorPairs': int(anchored.sum())},
            'together': {'pairs': int(together.sum()), 'independentExpectation': round(expected, 2), 'playerJumpRate': round(rate, 4),
                         'perMinute': round(float(together.sum())/minutes, 2) if minutes else None,
                         'atAnchors': int(sum(at_anchor(i) is not None for i in np.flatnonzero(together))),
                         'powerShare': round(float(power[together].sum()/power[use].sum()), 4) if power[use].sum() else None},
            'listed': listed}


# ---------- the report ----------
def display_times(t0, t1, hz=HZ):
    step = 1.0/hz
    n = int(math.floor((t1-t0-PHASE*step)/step))+1
    return t0+PHASE*step+np.arange(max(n, 0))*step, step


def evaluate(result, window=None, ids=None, hz=HZ, compute=True, rebuilt=None):
    """The report of one result (see the module docstring). ids: identities to measure (None: all); '?' among them keeps
    the '?' markers (they are measured whenever ids is None). rebuilt: the result with its registration rebuilt
    (rebuild()), measured as the REGISTRATION and STABILIZED+REGISTRATION streams."""
    started = time.perf_counter()
    frames = result.get('frames') or []
    times = np.array([f['time'] for f in frames], np.float64)
    fps = float((result.get('video') or {}).get('fps') or 0)
    frame_step = 1/fps if fps > 0 else ((times[-1]-times[0])/(len(times)-1) if len(times) > 1 else .04)
    only = None if ids is None else set(ids)-{'?'}
    players = only is None or bool(only)
    with_anonymous = ids is None or '?' in ids
    roster = final_roster(result)
    block, source = stabilized_block(result, compute)
    T_disp, disp_step = display_times(float(times[0]), float(times[-1]), hz)
    inside = np.ones(len(times), bool) if window is None else (times >= window[0]) & (times <= window[1])
    report = {'frames': len(frames), 'time': [round(float(times[0]), 3), round(float(times[-1]), 3)], 'window': window, 'fps': fps, 'displayHz': hz,
              'stabilized': {'source': source if block is not None else None, 'missing': None if block is not None else source,
                             'method': block.get('method') if block else None},
              'rebuilt': rebuilt is not None, 'rules': RULES, 'streams': {}, 'responsiveness': {}, 'commonMode': {}, 'correction': {}, 'coverage': {},
              'warnings': []}
    # (stream, its result, its raw observations, its tracks, drawn by sampleStable, the block for corrections)
    sources = [('raw', result, None)]
    if block is not None:
        sources.append(('stabilized', result, block))
    if rebuilt is not None:
        sources.append(('registration', rebuilt, None))
        if rebuilt.get('stabilized'):
            sources.append(('stabilized+registration', rebuilt, rebuilt['stabilized']))
    for name, res, blk in sources:
        people = people_samples(res, roster)
        raw_obs = raw_observations(res, roster, people, only)
        if blk is None:
            ident, anon = raw_stream(res, times, roster, people, only)
        else:
            ident, anon = _stable_tracks(blk, times, roster, only)
        stable = blk is not None
        groups = {'players': (ident, RULES['holdAfter'], RULES['interpolateGap'])} if players else {}
        if with_anonymous:
            groups['anonymous'] = (anon, RULES['anonymousHold'], RULES['anonymousGap'])
        report['streams'][name] = {}
        for group, (tracks, hold, gap) in groups.items():
            display, per, share = measure(tracks, T_disp, disp_step, frame_step, hold, gap, stable, window)
            at_frames, _, _ = measure(tracks, times, frame_step, frame_step, hold, gap, stable, window)
            report['streams'][name][group] = {'markers': len(tracks), 'estimatedShare': share, 'display': display, 'frames': at_frames, 'perMarker': per}
        if players:
            events = box_events(blk if blk is not None else block if res is result else (rebuilt or {}).get('stabilized'))
            report['responsiveness'][name] = responsiveness(ident, raw_obs, times, frame_step, RULES['holdAfter'], RULES['interpolateGap'], stable, inside,
                                                            events)
        if blk is None:
            tl = _timeline_block(res) or {}
            cols = {pid: c for pid, c in (tl.get('ids') or {}).items() if pid in roster and (only is None or pid in only)}
        else:
            cols = {pid: c for pid, c in blk['ids'].items() if pid in roster and (only is None or pid in only)}
            report['correction'][name] = {'players': corrections(raw_obs, cols, inside) if players else None}
            if with_anonymous:
                report['correction'][name]['anonymous'] = corrections(people[1], {str(k): v for k, v in (blk.get('anonymous') or {}).items()}, inside)
            if players:
                report['coverage'][name] = coverage(raw_obs, blk, roster, inside, only)
        cm = common_mode(observed_positions(cols, len(frames)), res, times, frame_step, inside) if cols else None
        report['commonMode'][name] = cm
        if blk is not None and cm and cm['together']['atAnchors']:
            report['warnings'].append(f"{LABELS[name]} still jumps together at {cm['together']['atAnchors']} listed anchors ({cm['together']['powerShare']} of "
                                      'its residual displacement power): the homographies snap there'
                                      + ('' if rebuilt is not None else '; --rebuild measures the run on the rebuilt registration'))
    if block is not None:
        report['stabilized'].update(summary=block.get('summary'), noise=block.get('noise'))
    report['seconds'] = round(time.perf_counter()-started, 1)
    return report


def _pair(q):
    return '-' if q is None or q['p50'] is None else f"{q['p50']}/{q['p90']}"


def _rows():
    """(label, getter of a group's part) of the jitter tables, labelled from the constants."""
    d = lambda key: (lambda s, part: s['display'][part][key])
    return (('marker-minutes', d('minutes')),
            ('residual RMS (cm)', d('residualRms_cm')),
            ('stationary residual RMS (cm)', d('residualStationaryRms_cm')),
            (f'reversals >= {REVERSAL[1]*100:g} cm /min', d('reversals25PerMin')),
            (f'reversals >= {REVERSAL[0]*100:g} cm /min', d('reversals10PerMin')),
            (f'jumps > {JUMP:g} m /min', d('jumpsOver1mPerMin')),
            (f'lag speed > {SPEED_MAX:g} m/s events /min', d('lagSpeedOver10EventsPerMin')),
            *((f'accLag share > {level:g} m/s^2', d(_acc_key(level))) for level in ACC_LEVELS),
            (f'accLag250 share > {ACC_LONG_LEVEL:g} m/s^2', d(_acc_key(ACC_LONG_LEVEL, True))),
            ('stationary step p50/p90 (cm), display', lambda s, part: _pair(s['display'][part]['stationaryStep_cm'])),
            ('stationary step p50/p90 (cm), frames', lambda s, part: _pair(s['frames'][part]['stationaryStep_cm'])))


def _q(v, key='median_m'):
    return '-' if not v or v.get(key) is None else f"{v[key]}/{v['p90_m']} ({v['n']})"


def table(report, name=''):
    streams = list(report['streams'])
    rebuilt = 'rebuilt registration' if report.get('rebuilt') else 'stored registration only'
    head = f"== {name} {report['frames']} frames {report['time'][0]}-{report['time'][1]} s" + (f", window {report['window']}" if report['window'] else '') + \
        f", display {report['displayHz']:g} Hz; stabilized: " + (report['stabilized']['source'] or f"missing ({report['stabilized']['missing']})") + f'; {rebuilt}'
    lines = [head]
    columns = '| metric | '+' | '.join(LABELS.get(s, s) for s in streams)+' |'
    rows = _rows()
    for group, part, title in (('players', 'observed', 'players, observed (the observation, or between two of consecutive frames)'),
                               ('players', 'estimated', 'players, estimates (held, across a missing frame or gap, imputed)'),
                               ('anonymous', 'observed', "'?' markers, observed"), ('anonymous', 'estimated', "'?' markers, estimates")):
        have = [s for s in streams if group in report['streams'][s]]
        if not have:
            continue
        shares = ', '.join(f"{LABELS.get(s, s)} {report['streams'][s][group]['estimatedShare']}" for s in have)
        lines += ['', f"### {title}; estimates' share of the drawn samples: {shares}", columns, '|---|'+'---|'*len(streams)]
        for label, get in rows:
            cells = []
            for s in streams:
                v = get(report['streams'][s][group], part) if group in report['streams'][s] else None
                cells.append('-' if v is None else str(v))
            lines.append(f'| {label} | '+' | '.join(cells)+' |')
    if report['responsiveness']:
        lines += ['', '### players, against the raw observations (at frame times; the rows above cannot tell a sluggish smoother)', columns,
                  '|---|'+'---|'*len(streams)]
        resp = [report['responsiveness'].get(s) or {} for s in streams]
        cell = lambda v: '-' if v is None else str(v)
        lines.append('| lag vs raw (ms) | '+' | '.join(cell(r.get('lag_ms')) for r in resp)+' |')
        lines.append('| velocity correlation | '+' | '.join(cell(r.get('velocityCorrelation')) for r in resp)+' |')
        lines.append('| turn rounding median/p90 m (n) | '+' | '.join(_q(r.get('turns')) for r in resp)+' |')
        lines.append('| stop rounding median/p90 m (n) | '+' | '.join(_q(r.get('stops')) for r in resp)+' |')
        lines.append('| turn rounding, no box event near | '+' | '.join(_q(r.get('turnsNoBoxEvent')) for r in resp)+' |')
        lines.append('| stop rounding, no box event near | '+' | '.join(_q(r.get('stopsNoBoxEvent')) for r in resp)+' |')
        lines.append("| raw local mean's noise (m) | "+' | '.join(cell(r.get('localMeanNoise_m')) for r in resp)+' |')
        corr = [(report['correction'].get(s) or {}).get('players') for s in streams]
        lines.append('| correction median/p90 m (n) | '+' | '.join(_q(c) for c in corr)+' |')
    lines.append('')
    for stream, kinds in report['correction'].items():
        for kind, value in kinds.items():
            if value:
                lines.append(f"correction {LABELS.get(stream, stream)} {kind}: median {value['median_m']} m, p90 {value['p90_m']} m, max {value['max_m']} m (n {value['n']})")
    for stream, c in report['coverage'].items():
        lines.append(f"coverage {LABELS.get(stream, stream)}: {c['rawObservations']} raw observations -> stabilized kinds {c['stabilizedKindAtRawObservations']}; "
                     f"weights rejected {c['weights']['rejectedShare']}, down-weighted {c['weights']['downWeightedShare']}; flags {c['flags']}")
    for stream, cm in report['commonMode'].items():
        if not cm:
            continue
        lines.append(f"common mode {LABELS.get(stream, stream)}: share {cm['commonShare']} (independent {cm['independentExpectation']}, {cm['meanPlayers']} players); "
                     f"median common step at anchors {cm['commonStepMedian_m']['atAnchors']} m ({cm['commonStepMedian_m']['anchorPairs']} pairs), elsewhere "
                     f"{cm['commonStepMedian_m']['elsewhere']} m; jumps together {cm['together']['pairs']} (independent {cm['together']['independentExpectation']}; "
                     f"{cm['together']['perMinute']}/min, {cm['together']['atAnchors']} at anchors, {cm['together']['powerShare']} of the power)")
        for e in cm['listed'][:8]:
            lines.append(f"   {e['time']:.3f} s  {e['jumping']}/{e['players']} players jump {e['jumpMedian_m']} m (common {e['commonStep_m']} m)  "
                         f"{e['before']['source']}->{e['after']['source']} "
                         f"age {e['before']['anchorAge']}->{e['after']['anchorAge']}  anchor {e['anchor']}")
    for warning in report['warnings']:
        lines.append(f'WARNING: {warning}')
    return '\n'.join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('result', help='result.json, or a run directory holding one')
    ap.add_argument('--window', type=float, nargs=2, metavar=('S', 'E'), help='measure only this source-time window')
    ap.add_argument('--json', help='write the full report here')
    ap.add_argument('--ids', nargs='+', help="identities to measure ('?' for the unidentified markers)")
    ap.add_argument('--hz', type=float, default=HZ, help='display refresh rate')
    ap.add_argument('--no-compute', action='store_true', help='do not build a stabilized block the result does not have')
    ap.add_argument('--rebuild', action='store_true', help="also measure the run on its registration rebuilt from the run's registration.json")
    a = ap.parse_args(argv)
    path = Path(a.result)
    path = path/'result.json' if path.is_dir() else path
    with open(path, encoding='utf-8') as handle:
        result = json.load(handle)
    rebuilt, why = rebuild(path.parent, result) if a.rebuild else (None, None)
    report = evaluate(result, a.window, a.ids, a.hz, not a.no_compute, rebuilt)
    report['input'] = str(path)
    if why:
        report['warnings'].append(f'--rebuild: {why}')
    print(table(report, path.parent.name))
    if a.json:
        Path(a.json).write_text(json.dumps(report, indent=1), encoding='utf-8')
    return 0


if __name__ == '__main__':
    sys.exit(main())
