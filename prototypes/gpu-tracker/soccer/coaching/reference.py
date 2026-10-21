"""The professional reference the coaching detectors compare a team with: distributions of team width, depth,
compactness, the gaps between the units, back-line evenness, isolation and lateral co-movement, and how many episodes
beyond them professional teams show in a stretch of a given length, measured on real professional tracking (the Metrica
Sports sample game 1, both teams, 5 samples a second, every outfield player observed) by tools/coaching_bench.py into
reference.json.

Why a reference and not fixed ideals: there is no universal right width or gap. On that game a team's defence to
midfield gap is 15.7 m at the p90 with its centre in its own third but 24.9 m with it in the final third, and each
team spends about 10% of its time above the game's own p90 of every metric by construction: "a large gap happened
three times" is routine in professional play (the default rule, above the p95 for 5 s, fires 3-11 times per team and
half for that gap). A PitchIQ count is therefore always compared with what professional teams showed under the same
rule, in the same context, over the same measured time, never read on its own. The reference has no possession phase
(PitchIQ records none), so the all-phase distributions are the ones used.

Dead ball included. A PitchIQ run has no in-play mask (nothing in the result says when the ball is dead), so the
distributions a run is compared with are measured the way a run measures: over all the time with the team's 10
outfield players observed, dead ball included. The in-play distributions stay in the file as documentation
(metrics[m].inPlay); see tools/coaching_bench.py for what the dead ball changes.

Window distributions. Episodes are not spread evenly: they cluster within a half and differ between the teams (game 1
team B's second half reads 0.44 width episodes a minute overall, yet one 10-minute stretch of it holds 1.12 a minute),
so a Poisson test at a professional rate calls professional play a weakness (1.3-1.4 improvement findings per 10
team-minutes on game 1 itself). The reference instead keeps, per metric and context, the distribution of the episode
count over professional team-windows of W measured minutes (W in WINDOW_MINUTES and a whole half; windows slide by
WINDOW_SLIDE_SECONDS through each team-half of both teams), with its p50/p90/p95/p99/p999, maximum, histogram and the
windows per team. window_baseline gives those levels for any measured time M (those of the next longer window
length), window_tail the share of professional windows with at least a count: both are upper estimates, so a
comparison never favours a finding. A context whose windows do not come from both teams (each at least
WINDOW_MIN_TEAM_SHARE of them) is marked one-team (bothTeams false): no improvement can rest on one professional
team's windows. A strength compares sample quantiles, which are one team's too in such a context (game 1's
back-line sizes: a back four is team A's, a back three team B's), so a back-line strength is set against one
professional team's line; the strength rule's favourable side keeps it to medians below that team's own median.

One professional match is a narrow base: the reference is a sanity range, not a standard. Its source, sample sizes
and rules travel with every observation built on it (soccer/coaching/detectors.py).

The lateral shift is the one reference metric that is not a value per sample: it is the lag and the correlation of
two units' sideways velocities over a window. Its computation lives here (player_velocity, unit_velocity,
lateral_lag, lateral_windows, dm_velocities, late_windows, independent_windows), so that the professional distribution
(tools/coaching_bench.py) and a team's measurement (the lateral_shift detector) are the same code. A unit's sideways
velocity is the mean of its members' own velocities, not the velocity of the unit's centre: when a player changes unit
the centre jumps (a back four's centre moves 2.5 m when a full-back 10 m wide is swapped for a midfielder), the members
do not. On Metrica game 1 the centre's velocity widens team A's lag p10-p90 from -0.3..0.9 s to -0.7..1.5 s (B:
-0.8..0.7 to -1.2..0.9 s). A window whose best lag sits at the edge of the search (|lag| >= LAG_MAX_SECONDS - one
sample step) is no measurement: the true best shift may lie beyond the search, and game 1 had 11 such 20 s windows of
team B at exactly +-4 s, 6 of them with a correlation above .5. Such windows are left out of the reference and of every
detection. Lags are stated at the 5 Hz resolution (quantize_lag: multiples of one sample step), never finer.

Identity. A reference is named '<source>@<sha16>' (reference_id) by content_sha: the sha256 of its canonical JSON
without the 'validation' block, which is a record about the frozen reference written after it (tools/coaching_bench.py
--validate --record): recording a validation changes no threshold, so it leaves the name every observation carries
and the name the record gives the same. Until 2026-10-19 the name was the sha of the file's bytes (the validation
history's e8410fd48fca5ddc and e8dc19cdfecc8219 are those).
"""
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from . import episodes as ep

REFERENCE_PATH = Path(__file__).with_name('reference.json')
MIN_SAMPLES = 300           # a context quantile from fewer samples (60 s at 5 Hz) falls back to the metric's 'all'
QUANTILES = ('p05', 'p10', 'p25', 'p50', 'p75', 'p90', 'p95', 'p975')

# Window distributions of the episode count (tools/coaching_bench.py writes them as metrics[m].windowCounts and, for
# the per-player nearest-teammate distance, metrics[m].playerWindowCounts: {context: {rule: {'entries': [...]}}}).
WINDOW_MINUTES = (1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 15.0, 20.0, 25.0, 30.0)
                                # measured minutes of a professional window (plus a whole half): the lengths 1, 2, 3, 5,
                                # 10 and 20 with steps between them, so the next longer length is at most 1.5x (above one
                                # minute 1.33x) a measured time (window_baseline compares with that next longer length)
WINDOW_SLIDE_SECONDS = 5.0      # s of measured time between the starts of two professional windows (they overlap:
                                # the distribution is that of a window placed anywhere, its n is no count of
                                # independent windows; 'independent' says how many disjoint ones fit)
WINDOW_MIN_TEAM_SHARE = .2      # each team must give at least this share of a length's windows for the distribution to
                                # count as both teams' (game 1's back four: team A 891 one-minute windows, team B 52:
                                # 5.5%, team B's back four is a few minutes of its back three's wing-backs dropping)
WINDOW_LEVELS = ('p50', 'p90', 'p95', 'p99', 'p999', 'max')

# The lateral shift of the defence against the midfield (Metrica game 1, both units complete, both teams, dead ball
# included): 30 s windows give a lag of the defence behind the midfield of p10 -0.8 s, p50 0.2 s, p90 1.0 s (in play
# -0.6 / 0.2 / 0.8 s) and a peak correlation p50 .85; 20 s windows are wider, p10 -1.0 s, p90 1.0 s, p95 1.4 s.
LAG_WINDOW_SECONDS = 30.0   # s: one lag and one correlation per window this long (the reference's 'all' context)
LAG_STEP_SECONDS = 5.0      # s: windows start this far apart inside a continuous measured stretch (the formation
                            # windows' step: a unit's membership changes no faster)
LAG_MAX_SECONDS = 4.0       # s: the lag is searched within +-this (ball to forwards reaches 4 s; defence to midfield
                            # stays within +-1.3 s from p10 to p90)
LAG_BOUND_STEPS = 1         # a window whose |lag| is within this many sample steps of LAG_MAX_SECONDS is dropped
LAG_SMOOTH_SECONDS = 1.0    # s: each player's sideways position is averaged over this before it is differenced (5 samples
                            # at 5 Hz: the frame-to-frame registration jitter of 0.4-0.6 m per step would dominate otherwise)
LAG_MIN_SPEED = .3          # m/s: both units' sideways velocity must vary at least this much over a window (its std), or
                            # there is no movement to line up and the lag is noise (it removed none of game 1's 393
                            # in-play windows: it is there for a team walking through a stoppage)
LAG_MIN_CORR = .5           # a window whose best correlation is lower is never counted as late (professional p10 .62:
                            # its best lag is not a clear one)

_cache = {}
IDENTITY_EXCLUDES = ('validation',)     # blocks of reference.json that describe the reference without being part of it


def load_reference(path=None):
    """The reference dict (cached per file and modification time). Raises FileNotFoundError when it is missing:
    callers say "no professional reference available" instead of comparing with nothing."""
    p = Path(path) if path is not None else REFERENCE_PATH
    st = p.stat()
    key = (str(p.resolve()), st.st_mtime_ns, st.st_size)
    if _cache.get('key') != key:
        data = p.read_bytes()
        ref = json.loads(data.decode('utf-8'))
        ref['_sha'] = content_sha(ref)
        ref['_fileSha'] = hashlib.sha256(data).hexdigest()[:16]
        _cache.update(key=key, ref=ref)
    return _cache['ref']


def content_sha(reference):
    """The reference's identity (16 hex, module docstring "Identity"): the sha256 of its canonical JSON (sorted keys,
    no spaces, ASCII) without IDENTITY_EXCLUDES and without the loader's own '_' keys."""
    body = {k: v for k, v in reference.items() if k not in IDENTITY_EXCLUDES and not str(k).startswith('_')}
    text = json.dumps(body, sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False)
    return hashlib.sha256(text.encode('ascii')).hexdigest()[:16]


def reference_id(reference):
    """'<source>@<sha16>': which reference an observation was compared with."""
    return f"{reference.get('source', 'unknown')}@{reference.get('_sha', 'inline')}"


def quantile(reference, metric, q='p90', context='all'):
    """The metric's quantile q in context ('all', 'third:own', 'backline:4', ...), falling back to 'all' when the
    context is missing or rests on fewer than MIN_SAMPLES samples. Returns (value, used_context) or (None, None)."""
    m = (reference.get('metrics') or {}).get(metric) or {}
    contexts = m.get('contexts') or {}
    for c in (context, 'all'):
        qd = contexts.get(c)
        if qd and (qd.get('n') or 0) >= MIN_SAMPLES and _finite(qd.get(q)):
            return float(qd[q]), c
    return None, None


def episode_baseline(reference, metric, rule='default', context='all'):
    """The professional episode rate for a metric under a rule: {'perMinute': {p10, p50, p90}, 'perHalf',
    'durationSeconds', 'shareOfTime', 'context'} for the context, else 'all', else None. reference.json keeps them as
    metrics[metric]['episodes'][context][rule]. Descriptive (how often per minute professional teams showed one); the
    improvement test is window_baseline's."""
    eps = (((reference.get('metrics') or {}).get(metric) or {}).get('episodes') or {})
    for c in (context, 'all'):
        r = (eps.get(c) or {}).get(rule)
        if r:
            return dict(r, context=c)
    return None


def rule(reference, name='default'):
    """The episode rule: {'enter': 'p95', 'exit': 'p90', 'minSeconds': 5.0, 'mergeSeconds': 3.0}."""
    rules = reference.get('rules') or {}
    r = rules.get(name)
    if r is None:
        raise KeyError(f'no episode rule {name!r} in the reference')
    return dict(r)


def tau_seconds(reference, metric):
    """The metric's decorrelation time in seconds (metrics[metric].tau.seconds: the e-folding time of the professional
    beyond-p75 indicator, tools/coaching_bench.py), or None when the reference has none."""
    t = (((reference.get('metrics') or {}).get(metric) or {}).get('tau') or {}).get('seconds')
    return float(t) if _finite(t) and t > 0 else None


def _finite(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


# ---------- window distributions ----------
def window_entries(reference, metric, context, rule='default', per_player=False):
    """The window entries of a metric in a context (sorted by minutes), [] when the reference has none. Exactly the
    context: a window distribution never falls back to another context (its counts are that context's)."""
    key = 'playerWindowCounts' if per_player else 'windowCounts'
    block = ((((reference.get('metrics') or {}).get(metric) or {}).get(key) or {}).get(context) or {}).get(rule) or {}
    entries = [e for e in block.get('entries') or () if isinstance(e, dict) and _finite(e.get('minutes'))
               and (e.get('n') or 0) > 0]
    return sorted(entries, key=lambda e: float(e['minutes']))


def _usable(entries):
    """The leading entries whose windows come from both teams (a longer window has fewer of them, so the both-team
    lengths are a prefix)."""
    out = []
    for e in entries:
        if not e.get('bothTeams'):
            break
        out.append(e)
    return out


def _tail(entry, count):
    """The share of the entry's windows with at least `count` episodes (its histogram)."""
    hist = entry.get('hist') or []
    n = float(sum(hist))
    if n <= 0:
        return None
    c = max(0, int(math.ceil(count-1e-9)))
    return float(sum(hist[c:]))/n


def _pick(usable, entries, M):
    """(entry, method) the levels at M measured minutes are read from: the shortest both-team length at least M
    ('exact' or 'next_longer'); beyond the longest length computed (the whole half, or the longest any professional
    team spent in the context in one half), that length scaled ('scaled'); (None, reason) when the both-team lengths
    end before M while longer one-team lengths exist."""
    hi = [e for e in usable if float(e['minutes']) >= M-1e-9]
    if hi:
        return hi[0], ('exact' if abs(float(hi[0]['minutes'])-M) < 1e-9 else 'next_longer')
    if usable and len(usable) == len(entries):
        return usable[-1], 'scaled'
    return None, 'one_team_baseline'


def window_baseline(reference, metric, context, minutes, rule='default', per_player=False):
    """The professional window levels for a measured time of `minutes` in a context: {'available', 'minutes',
    'method', 'from' (the window length used), 'p50' ... 'p999', 'max', 'n', 'nByTeam', 'bothTeams', 'reason'}.

    The levels are those of the shortest professional window length at least as long as M ('next_longer'): a window
    holds every episode of any stretch inside it, so its count quantiles are at least those of M, and the comparison
    can only be stricter than at M itself. Each level is the largest of that length and every shorter one (a
    quantile grows with the length; the measured long lengths rest on few windows, so game 1's back-line height read
    a p99 of 13 episodes at 30 minutes but 9 over a whole half). The nearest length not longer than M is not used:
    counts are small integers, and a team at a professional rate shows two spells in 1.28 minutes more often than in
    one (on game 1 the one-minute level scaled to 1.28 minutes let in-sample professional windows through as
    improvements). Shorter than the shortest length: that length. Longer than the longest length computed (the whole
    half, 22-45 measured minutes, or the longest stretch any professional team spent in the context in one half): its
    levels scaled by M / its length ('scaled'; the counts grow less than in proportion from 10 minutes on, so this
    errs on the strict side). available False with reason 'no_window_baseline' (the reference has none for the metric
    and context) or 'one_team_baseline' (no window length as long as M has windows of both teams: game 1's back-line
    sizes, and the final third beyond a few minutes)."""
    entries = window_entries(reference, metric, context, rule, per_player)
    usable = _usable(entries)
    out = {'available': False, 'minutes': round(float(minutes), 3), 'context': context, 'perPlayer': bool(per_player)}
    if not entries:
        return dict(out, reason='no_window_baseline')
    M = max(float(minutes), 0.0)
    e, method = _pick(usable, entries, M)
    if e is None:
        return dict(out, reason=method, nByTeam=(usable[-1] if usable else entries[0]).get('nByTeam'),
                    longestBothTeams=float(usable[-1]['minutes']) if usable else None)
    scale = M/float(e['minutes']) if method == 'scaled' else 1.0
    upto = [x for x in usable if float(x['minutes']) <= float(e['minutes'])+1e-9]
    levels = {}
    for k in WINDOW_LEVELS:
        vals = [float(x[k]) for x in upto if _finite(x.get(k))]
        levels[k] = round(max(vals)*scale, 3) if vals else None
    return dict(out, available=True, method=method, reason=None, **levels,
                **{'from': [float(e['minutes'])], 'n': int(e['n']), 'nByTeam': dict(e.get('nByTeam') or {}),
                   'bothTeams': True})


def window_tail(reference, metric, context, minutes, count, rule='default', per_player=False):
    """An upper estimate of the share of professional windows of `minutes` measured minutes in the context with at
    least `count` episodes (an empirical p-value), or None without a both-team distribution as long: the largest share
    over the length window_baseline reads (the next longer one: a share at least that at M) and every shorter one;
    beyond the longest length, its share of at least floor(count / scale) episodes."""
    entries = window_entries(reference, metric, context, rule, per_player)
    usable = _usable(entries)
    e, method = _pick(usable, entries, max(float(minutes), 0.0))
    if e is None:
        return None
    if method == 'scaled':
        count = math.floor(float(count)*float(e['minutes'])/max(float(minutes), 1e-9)+1e-9)
    tails = [_tail(x, count) for x in usable if float(x['minutes']) <= float(e['minutes'])+1e-9]
    tails = [x for x in tails if x is not None]
    return max(tails) if tails else None


def holm(pvalues, alpha):
    """Holm's step-down procedure: the indices (into pvalues) rejected at family-wise level alpha. None entries are
    not tested (they neither count in the family nor can be rejected)."""
    tested = sorted((p, i) for i, p in enumerate(pvalues) if p is not None)
    k = len(tested)
    out = []
    for rank, (p, i) in enumerate(tested):
        if p > alpha/(k-rank):
            break
        out.append(i)
    return out


def share_upper_bound(share, n_eff, confidence=.8):
    """The one-sided upper confidence bound of a share observed over n_eff independent looks (Clopper-Pearson, with
    the share's look count k = share * n_eff allowed to be fractional): 1 - (1 - confidence)^(1/n) with nothing beyond,
    so 0% over n_eff looks is below 25% at 80% only from n_eff = ln .2 / ln .75 = 5.6 looks."""
    if not _finite(share) or not _finite(n_eff) or n_eff <= 0:
        return 1.0
    from scipy.special import betaincinv
    k = min(max(float(share), 0.0), 1.0)*float(n_eff)
    if k >= n_eff-1e-12:
        return 1.0
    return float(betaincinv(k+1.0, float(n_eff)-k, float(confidence)))


# ---------- lateral shift: the same computation for the reference and for a team ----------
def quantize_lag(x, step):
    """A lag in seconds at the sampling's resolution: the nearest multiple of one sample step (0.2 s at 5 Hz), so no
    lag is ever stated finer than the data can resolve. None stays None."""
    if x is None or not _finite(float(x)) or not step or step <= 0:
        return None
    q = round(float(x)/float(step))*float(step)
    return round(q+0.0, 6)


def player_velocity(t, Y, smooth_seconds=LAG_SMOOTH_SECONDS):
    """Sideways velocity (m/s) per sample and player. Y: S x N sideways metres (NaN where the player is not observed).
    Each player's position is averaged over smooth_seconds (an odd number of samples, centred) and differenced across
    the two neighbouring samples; NaN unless every sample of that span is observed and the span has no time gap (a
    step longer than 1.5 sample steps), so nothing is ever interpolated."""
    t = np.asarray(t, float)
    Y = np.asarray(Y, float)
    if Y.ndim == 1:
        Y = Y[:, None]
    S, N = Y.shape
    V = np.full((S, N), np.nan)
    step = ep.sample_step(t)
    k = max(1, int(round(smooth_seconds/step)))
    k += 1-k % 2                                     # odd: centred on the sample
    h = k//2
    if S < k+2 or not N:
        return V
    fin = np.isfinite(Y)
    c = np.vstack([np.zeros((1, N)), np.cumsum(np.where(fin, Y, 0.0), axis=0)])
    smooth = np.full((S, N), np.nan)
    smooth[h:S-h] = (c[k:]-c[:S-k+1])/k              # mean of Y[s-h .. s+h]
    bad = np.vstack([np.zeros((1, N), np.int64), np.cumsum(~fin, axis=0)])
    gaps = np.r_[0, np.cumsum(np.diff(t) > 1.5*step)]          # gaps[i]: time gaps between samples 0..i
    s = np.arange(h+1, S-1-h)
    a, b = s-1-h, s+1+h                              # the span the difference rests on
    ok = ((bad[b+1]-bad[a]) == 0) & ((gaps[b]-gaps[a]) == 0)[:, None]
    with np.errstate(invalid='ignore'):
        V[s] = np.where(ok, (smooth[s+1]-smooth[s-1])/(t[s+1]-t[s-1])[:, None], np.nan)
    return V


def unit_velocity(t, Y, members, smooth_seconds=LAG_SMOOTH_SECONDS):
    """A unit's sideways velocity (m/s) per sample: the mean of its members' velocities (members: S x N bool, the
    unit's membership per sample), NaN where it has no member or a member's velocity is unknown."""
    V = player_velocity(t, Y, smooth_seconds)
    M = np.asarray(members, bool).reshape(V.shape)
    n = M.sum(axis=1)
    out = np.where(M, np.nan_to_num(V), 0.0).sum(axis=1)/np.maximum(n, 1)
    out[(M & ~np.isfinite(V)).any(axis=1) | (n == 0)] = np.nan
    return out


def lateral_lag(lead, follow, step, max_seconds=LAG_MAX_SECONDS):
    """(lag seconds, peak correlation) of the velocity `follow` against `lead` (equal lengths, finite): the time shift
    within +-max_seconds that lines the two up best, positive when `follow` moves later than `lead`, refined between
    samples by a parabola through the peak; the correlation is the normalized cross-correlation at the best whole-sample
    shift (over the overlapping part)."""
    a = np.asarray(lead, float)
    b = np.asarray(follow, float)
    a, b = a-a.mean(), b-b.mean()
    n = len(a)
    max_lag = max(0, min(int(round(max_seconds/step)), n-2))
    rs = []
    for shift in range(-max_lag, max_lag+1):
        x, y = (a[:n-shift], b[shift:]) if shift >= 0 else (a[-shift:], b[:n+shift])
        den = math.sqrt(float((x*x).sum())*float((y*y).sum()))
        rs.append(float((x*y).sum())/den if den > 0 else 0.0)
    rs = np.array(rs)
    k = int(np.argmax(rs))
    lag = float(k-max_lag)
    if 0 < k < len(rs)-1:
        den = rs[k-1]-2*rs[k]+rs[k+1]
        if den < 0:
            lag += .5*(rs[k-1]-rs[k+1])/den
    return lag*step, float(rs[k])


def at_search_bound(lag, step, max_seconds=LAG_MAX_SECONDS):
    """True when a window's lag sits at the edge of the search (|lag| >= max_seconds - LAG_BOUND_STEPS steps)."""
    return abs(float(lag)) >= float(max_seconds)-LAG_BOUND_STEPS*float(step)-1e-9


def lateral_windows(t, lead, follow, eligible, window_seconds=LAG_WINDOW_SECONDS, step_seconds=LAG_STEP_SECONDS,
                    max_seconds=LAG_MAX_SECONDS, min_speed=LAG_MIN_SPEED, drop_bound=True):
    """The lag and correlation of `follow` against `lead` (sideways velocities, S) over every window of
    window_seconds that lies inside one continuous stretch where `eligible` holds and both velocities are known; the
    windows start step_seconds apart from the stretch's start. A window in which either unit hardly moves sideways
    (velocity std below min_speed) is left out, and so (drop_bound) is one whose lag sits at the search bound
    (at_search_bound). Returns [{'t0', 't1', 'i0', 'lag', 'corr'}] in time order."""
    t = np.asarray(t, float)
    lead = np.asarray(lead, float)
    follow = np.asarray(follow, float)
    ok = np.asarray(eligible, bool) & np.isfinite(lead) & np.isfinite(follow)
    step = ep.sample_step(t)
    n = max(2, int(round(window_seconds/step)))
    every = max(1, int(round(step_seconds/step)))
    idx = np.flatnonzero(ok)
    if not len(idx):
        return []
    cut = np.flatnonzero((np.diff(idx) > 1) | (np.diff(t[idx]) > 1.5*step))
    out = []
    for run in np.split(idx, cut+1):
        for w0 in range(int(run[0]), int(run[-1])-n+2, every):
            va, vb = lead[w0:w0+n], follow[w0:w0+n]
            if va.std() < min_speed or vb.std() < min_speed:
                continue
            lag, r = lateral_lag(va, vb, step, max_seconds)
            if drop_bound and at_search_bound(lag, step, max_seconds):
                continue
            out.append({'t0': float(t[w0]), 't1': float(t[w0+n-1]), 'i0': w0, 'lag': lag, 'corr': r})
    return out


def late_windows(windows, enter, min_corr=LAG_MIN_CORR):
    """The windows beyond the enter level whose lag is a clear one (correlation at least min_corr)."""
    return [w for w in windows if w['lag'] >= enter and w['corr'] >= min_corr]


def independent_windows(windows, seconds):
    """The windows that do not overlap (each starting at least `seconds` after the previous one kept), in time
    order: the windows start every 5 s, so neighbours share most of their samples and are not independent looks."""
    out, last = [], None
    for w in windows:
        if last is None or w['t0'] >= last+seconds-1e-6:
            out.append(w)
            last = w['t0']
    return out


def dm_velocities(data, team):
    """The defence's and the midfield's sideways velocities of one team of a coaching CoachingData
    (soccer/coaching/series.build_series): (t, defence, midfield, eligible), the units the series' per-sample groups
    (D: the back line, M: every middle line), eligible where both units are complete and every member observed. The
    velocities (m/s) are differenced on the pitch and then turned into the team's attacking frame sample by sample
    (towards the team's right positive), so a side swap at half time never reads as a 68 m jump. The lagDM of the
    reference is lateral_windows(t, midfield, defence, eligible): positive when the defence moves later."""
    s = data.teams[team]
    y = np.asarray(s.y, float)*float(data.pitch['width'])
    sign = np.where(s.side == 'left', 1.0, np.where(s.side == 'right', -1.0, np.nan))
    defence = unit_velocity(s.t, y, s.group == 'D')*sign
    midfield = unit_velocity(s.t, y, s.group == 'M')*sign
    return s.t, defence, midfield, s.eligible['unitD'] & s.eligible['unitM']
