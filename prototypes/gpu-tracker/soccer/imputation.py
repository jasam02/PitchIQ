"""Where the players the camera does not see probably are: an estimate per roster slot and frame, with
its uncertainty and its kind, from the observed positions of the whole segment.

Kinds (never mixed up with observations):
    O  observed on this frame
    I  interpolated: between two observations at most INTERPOLATE_GAP apart (linear)
    P  predicted: an estimate with a sigma in metres
    U  unknown: nothing supportable (never observed, too long hidden, too uncertain); the slot stays, no
       coordinates are invented

The shipped estimate (estimate(), method 'hybrid'): between two observations, the straight line between
them (the sigma of both sides plus a term growing with the gap); where only one side is observed, the last
observed position moved with the team (TEAM SHIFT): the visible teammates' displacement since that
observation, each weighted by its uncertainties, at least MIN_VOTERS of them, ramped in over TAU_SHIFT so
that the first seconds hold the position, the sigma growing at SHIFT_SIGMA_RATE. An observation always
wins and the estimate meets both observations exactly (no teleporting when the player reappears).

Also implemented, for the benchmark (tools/metrica_bench.py) and as alternatives: holding the last
position, constant velocity (CV_SIGMA_RATE), a fixed anchor, the offset from the raw visible mean, and
ROLE-ANCHORED CENTROID VOTING: each player's offset from the team centre is an exponentially weighted mean
over the frames the player was observed (TAU_FORMATION: a dynamic formation, no fixed system); the team
centre on a frame is never the raw mean of the visible players (the camera shows one side of the team:
that mean is biased) but the weighted average of their votes (centre = position - own offset, at least
MIN_VOTERS), and the offsets are re-learned against the voted centre in BOOTSTRAP_PASSES passes so the bias
cancels in the learning too. On the Metrica sample games with simulated camera windows the voting lost to
holding the last position on every bucket of time hidden, and the team shift won from 1 s on
(PROGRESS.md), which is why the shipped estimate is the one above.

All positions here are in metres on the pitch (x along the length, y across). The module is pure numpy
and is the same code the Metrica benchmark (tools/metrica_bench.py) measures.
"""
import math

import numpy as np

TAU_FORMATION = 30.0     # s: time constant of a player's role offset (dynamic formation)
MIN_OFFSET_SECONDS = 10.0  # s of observations before a player has an offset
MIN_VOTERS = 4           # visible players with offsets needed for a voted centre
BOOTSTRAP_PASSES = 3
TAU_MOTION = 2.5         # s: the blend from the player's own motion to the formation estimate
VELOCITY_WINDOW = 1.0    # s: observations this recent give the velocity
VELOCITY_DECAY = 2.0     # s: the carried velocity fades with this time constant
CV_SIGMA_RATE = 1.8      # m/s: growth of the motion estimate's uncertainty (calibrated on Metrica game 1)
CV_ONLY_MAX = 3.0        # s: without a formation estimate, motion alone this long, then unknown
INTERPOLATE_GAP = 1.0    # s: gaps up to this with both ends observed are interpolated (I)
MAX_SIGMA = 15.0         # m: an estimate less certain than this is unknown
MAX_GAP = 25.0           # s: hidden longer than this (to the nearest observation) is unknown
CENTRE_FLOOR = 1.0       # m: the voted centre is never more certain than this
DEFAULT_SIGMA = 1.5      # m: observation uncertainty when none is given
TAU_ROLE = 20.0          # s: how fast a hidden player's offset relaxes from where they were last seen to their role offset
TAU_SHIFT = 2.0          # s: the team's movement is applied to the last position only gradually (holding wins in the first seconds)
SHIFT_SIGMA_RATE = .8    # m/s: growth of the team-shift estimate's uncertainty (the player moves with the team, plus this)
HYBRID_BASE = 'shift'    # the one-sided estimate used where only one side is observed (chosen on the benchmark)
KINDS = {'O': 0, 'I': 1, 'P': 2, 'U': 3}
KIND_NAMES = ['O', 'I', 'P', 'U']


# ---------- formation: offsets and the voted centre ----------
def learn_offsets(pos, sig, times, tau=TAU_FORMATION, passes=BOOTSTRAP_PASSES, min_seconds=MIN_OFFSET_SECONDS, method='voting'):
    """pos: (T, P, 2) metres with NaN where unobserved; sig: (T, P) observation sigma (NaN where unobserved);
    times: (T,). Returns (offsets (T, P, 2): each player's role offset as known at that frame, NaN before it
    exists; offset_var (T, P): its variance; centre (T, 2): the team centre per frame, NaN where unknown;
    centre_var (T,)). method 'voting' bootstraps the voted centre; 'mean' keeps the raw visible mean (the
    biased baseline, for the benchmark)."""
    T, P, _ = pos.shape
    visible = np.isfinite(pos[:, :, 0])
    count = visible.sum(axis=1)
    # Pass 0: the raw visible mean where most of the team is in view (a nearly unbiased start).
    centre = np.full((T, 2), np.nan)
    centre_var = np.full(T, np.nan)
    enough = count >= max(MIN_VOTERS, 7)
    for t in np.flatnonzero(enough):
        centre[t] = np.nanmean(pos[t], axis=0)
        centre_var[t] = CENTRE_FLOOR**2
    offsets = np.full((T, P, 2), np.nan)
    offset_var = np.full((T, P), np.nan)
    for _ in range(passes):
        offsets, offset_var = _ew_offsets(pos, sig, times, centre, tau, min_seconds)
        if method != 'voting':
            # The raw visible mean on every frame with any visible player: the biased reference.
            centre = np.full((T, 2), np.nan)
            centre_var = np.full(T, np.nan)
            for t in np.flatnonzero(count >= 1):
                centre[t] = np.nanmean(pos[t], axis=0)
                centre_var[t] = CENTRE_FLOOR**2
            continue
        centre, centre_var = vote_centres(pos, sig, offsets, offset_var)
    return offsets, offset_var, centre, centre_var


def _ew_offsets(pos, sig, times, centre, tau, min_seconds):
    T, P, _ = pos.shape
    offsets = np.full((T, P, 2), np.nan)
    offset_var = np.full((T, P), np.nan)
    for i in range(P):
        o = None
        var = 25.0
        seen = 0.0
        last_t = None
        for t in range(T):
            if np.isfinite(pos[t, i, 0]) and np.isfinite(centre[t, 0]):
                r = pos[t, i]-centre[t]
                if o is None:
                    o, var, last_t = r.copy(), 25.0, times[t]
                    seen = 0.0
                else:
                    dt = max(0.0, times[t]-last_t)
                    a = 1-math.exp(-dt/tau)
                    o = o+a*(r-o)
                    var = (1-a)*var+a*float(np.sum((r-o)**2))/2
                    seen += dt
                    last_t = times[t]
            if o is not None and seen >= min_seconds:
                offsets[t, i] = o
                offset_var[t, i] = max(var, 1.0)
    return offsets, offset_var


def vote_centres(pos, sig, offsets, offset_var, min_voters=MIN_VOTERS):
    """Per frame: the team centre as voted by the visible players with offsets (position - offset, weighted
    by 1/(offset variance + observation variance)); NaN with fewer than min_voters."""
    T = pos.shape[0]
    centre = np.full((T, 2), np.nan)
    centre_var = np.full(T, np.nan)
    for t in range(T):
        ok = np.isfinite(pos[t, :, 0]) & np.isfinite(offsets[t, :, 0])
        if ok.sum() < min_voters:
            continue
        s = sig[t, ok]
        s = np.where(np.isfinite(s), s, DEFAULT_SIGMA)
        w = 1.0/(offset_var[t, ok]+s*s)
        votes = pos[t, ok]-offsets[t, ok]
        centre[t] = (votes*w[:, None]).sum(axis=0)/w.sum()
        centre_var[t] = max(CENTRE_FLOOR**2, 1.0/w.sum())
    return centre, centre_var


# ---------- the estimate ----------
def _velocity(pos, sig, times, i, t, forward=True, window=VELOCITY_WINDOW):
    """Velocity of player i at an observed frame t from the observations within the window before (or after) it."""
    k = t
    step = -1 if forward else 1
    j = t+step
    while 0 <= j < len(times) and abs(times[j]-times[t]) <= window:
        if np.isfinite(pos[j, i, 0]):
            k = j
        j += step
    if k == t or abs(times[k]-times[t]) < 1e-9:
        return np.zeros(2)
    return (pos[t, i]-pos[k, i])/(times[t]-times[k])


def _one_direction(pos, sig, times, offsets, offset_var, centre, centre_var, forward=True, method='voting'):
    """Per player and frame, the estimate carried from the nearest observation on one side: (est (T, P, 2),
    sigma (T, P), age (T, P) seconds since that observation; NaN where none)."""
    T, P, _ = pos.shape
    est = np.full((T, P, 2), np.nan)
    sigma = np.full((T, P), np.nan)
    age = np.full((T, P), np.nan)
    order = range(T) if forward else range(T-1, -1, -1)
    for i in range(P):
        last = None   # (time, position, sigma, velocity)
        last_centre = None   # (centre, centre variance) at the last observation, for the team-shift estimate
        for t in order:
            if np.isfinite(pos[t, i, 0]):
                s = sig[t, i] if np.isfinite(sig[t, i]) else DEFAULT_SIGMA
                last = (times[t], pos[t, i].copy(), s, _velocity(pos, sig, times, i, t, forward))
                last_centre = (centre[t].copy(), centre_var[t]) if np.isfinite(centre[t, 0]) else None
                est[t, i], sigma[t, i], age[t, i] = pos[t, i], s, 0.0
                continue
            if last is None:
                continue
            tau = abs(times[t]-last[0])
            if tau > MAX_GAP:
                continue
            sign = 1.0 if forward else -1.0
            if method in ('shift', 'shiftrole'):
                # The player moves with the team: the last position shifted by the voted centre's movement since,
                # the offset relaxing towards the role offset as the time hidden grows ('shiftrole').
                if last_centre is not None and np.isfinite(centre[t, 0]):
                    o_last = last[1]-last_centre[0]
                    o = o_last
                    if method == 'shiftrole' and np.isfinite(offsets[t, i, 0]):
                        lam = math.exp(-tau/TAU_ROLE)
                        o = lam*o_last+(1-lam)*offsets[t, i]
                    ramp = 1-math.exp(-tau/TAU_SHIFT)
                    est[t, i] = last[1]+ramp*((centre[t]+o)-last[1])
                    sigma[t, i] = math.sqrt(last[2]**2+ramp*(centre_var[t]+last_centre[1])+(SHIFT_SIGMA_RATE*tau)**2)
                else:
                    est[t, i], sigma[t, i] = last[1], math.sqrt(last[2]**2+(CV_SIGMA_RATE*tau)**2)
                age[t, i] = tau
                continue
            if method == 'last':
                p_cv, s_cv = last[1], math.sqrt(last[2]**2+(CV_SIGMA_RATE*tau)**2)
            else:
                decay = VELOCITY_DECAY*(1-math.exp(-tau/VELOCITY_DECAY))   # integral of the fading velocity
                p_cv = last[1]+sign*last[3]*decay
                s_cv = math.sqrt(last[2]**2+(CV_SIGMA_RATE*tau)**2)
            p_f = None
            if method in ('voting', 'anchor', 'mean') and np.isfinite(centre[t, 0]) and np.isfinite(offsets[t, i, 0]):
                p_f = centre[t]+offsets[t, i]
                s_f = math.sqrt(centre_var[t]+offset_var[t, i])
            if method in ('last', 'cv') or p_f is None:
                if tau > CV_ONLY_MAX and method not in ('last', 'cv'):
                    continue
                est[t, i], sigma[t, i], age[t, i] = p_cv, s_cv, tau
                continue
            w = math.exp(-tau/TAU_MOTION)
            est[t, i] = w*p_cv+(1-w)*p_f
            sigma[t, i] = math.sqrt((w*s_cv)**2+((1-w)*s_f)**2)
            age[t, i] = tau
    return est, sigma, age


def estimate_team(pos, sig, times, method='voting', bidirectional=True, learned=None):
    """pos (T, P, 2) metres with NaN where unobserved, sig (T, P), times (T,). Returns (est (T, P, 2), sigma
    (T, P), kind (T, P) as KINDS codes, age (T, P) seconds to the nearest observation). method: 'voting' (the
    full method), 'anchor' (offsets without time weighting: a fixed formation), 'mean' (offsets against the
    raw visible mean: the biased reference), 'cv' (motion only), 'last' (last position only), 'linear'
    (interpolation between observations only, offline)."""
    T, P, _ = pos.shape
    if learned is None:
        if method in ('voting', 'mean', 'shift', 'shiftrole') or method.startswith('hybrid'):
            learned = learn_offsets(pos, sig, times, method='mean' if method == 'mean' else 'voting')
        elif method == 'anchor':
            learned = learn_offsets(pos, sig, times, tau=1e9, method='voting')
        else:
            learned = (np.full((T, P, 2), np.nan), np.full((T, P), np.nan), np.full((T, 2), np.nan), np.full(T, np.nan))
    offsets, offset_var, centre, centre_var = learned
    kind = np.full((T, P), KINDS['U'], np.int8)
    est = np.full((T, P, 2), np.nan)
    sigma = np.full((T, P), np.nan)
    age = np.full((T, P), np.nan)
    observed = np.isfinite(pos[:, :, 0])
    if method.startswith('hybrid'):
        # Offline: a gap with observations at both ends is interpolated straight between them (the benchmark's
        # best); a stretch with an observation on one side only takes the one-sided estimate of the base method.
        base = method.split(':', 1)[1] if ':' in method else HYBRID_BASE
        l_est, l_sig, l_age = _one_direction(pos, sig, times, offsets, offset_var, centre, centre_var, True, 'last')
        r_est, r_sig, r_age = _one_direction(pos, sig, times, offsets, offset_var, centre, centre_var, False, 'last')
        f_est, f_sig, f_age = _one_direction(pos, sig, times, offsets, offset_var, centre, centre_var, True, base)
        b_est, b_sig, b_age = _one_direction(pos, sig, times, offsets, offset_var, centre, centre_var, False, base)
        both = np.isfinite(l_est[:, :, 0]) & np.isfinite(r_est[:, :, 0])
        f_est = np.where(both[:, :, None], l_est, f_est)
        f_sig = np.where(both, l_sig, f_sig)
        b_est = np.where(both[:, :, None], r_est, b_est)
        b_sig = np.where(both, r_sig, b_sig)
        f_age, b_age = np.where(both, l_age, f_age), np.where(both, r_age, b_age)
    elif method == 'linear':
        f_est, f_sig, f_age = _one_direction(pos, sig, times, offsets, offset_var, centre, centre_var, True, 'last')
        b_est, b_sig, b_age = _one_direction(pos, sig, times, offsets, offset_var, centre, centre_var, False, 'last')
    else:
        f_est, f_sig, f_age = _one_direction(pos, sig, times, offsets, offset_var, centre, centre_var, True, method)
        b_est, b_sig, b_age = _one_direction(pos, sig, times, offsets, offset_var, centre, centre_var, False, method) if bidirectional else (None, None, None)
    for i in range(P):
        for t in range(T):
            if observed[t, i]:
                est[t, i], sigma[t, i], kind[t, i], age[t, i] = pos[t, i], (sig[t, i] if np.isfinite(sig[t, i]) else DEFAULT_SIGMA), KINDS['O'], 0.0
                continue
            have_f = np.isfinite(f_est[t, i, 0])
            have_b = b_est is not None and np.isfinite(b_est[t, i, 0])
            if have_f and have_b:
                gap = f_age[t, i]+b_age[t, i]
                u = f_age[t, i]/gap if gap > 0 else .5
                if method == 'linear' or gap <= INTERPOLATE_GAP:
                    # Straight between the two observations (the endpoints are the observations themselves).
                    est[t, i] = (1-u)*f_est[t, i]+u*b_est[t, i] if method == 'linear' else (1-u)*f_est[t, i]+u*b_est[t, i]
                    sigma[t, i] = math.sqrt(((1-u)*f_sig[t, i])**2+(u*b_sig[t, i])**2)+(0.0 if method == 'linear' else .5*math.sqrt(gap))
                    kind[t, i] = KINDS['I'] if gap <= INTERPOLATE_GAP else KINDS['P']
                else:
                    est[t, i] = (1-u)*f_est[t, i]+u*b_est[t, i]
                    sigma[t, i] = math.sqrt(((1-u)*f_sig[t, i])**2+(u*b_sig[t, i])**2)
                    kind[t, i] = KINDS['P']
                age[t, i] = min(f_age[t, i], b_age[t, i])
            elif have_f or have_b:
                if method == 'linear':
                    continue
                src_est, src_sig, src_age = (f_est, f_sig, f_age) if have_f else (b_est, b_sig, b_age)
                est[t, i], sigma[t, i], kind[t, i], age[t, i] = src_est[t, i], src_sig[t, i], KINDS['P'], src_age[t, i]
            else:
                continue
            if kind[t, i] == KINDS['P'] and (sigma[t, i] > MAX_SIGMA or age[t, i] > MAX_GAP):
                est[t, i], sigma[t, i], kind[t, i] = np.nan, np.nan, KINDS['U']
    return est, sigma, kind, age


# ---------- the result's timeline block ----------
def estimate(result, size_length=105.0, size_width=68.0, min_confidence=.2):
    """The `timeline` block of a finished result: per roster slot (team players and goalkeepers), per frame,
    the estimated normalized position, its kind and its sigma (metres), from the observed positions of the
    segment; None where unknown. Pure: the same result gives the same timeline."""
    frames = result.get('frames', [])
    roster = result.get('roster') or {}
    slots = [s for s in roster.get('slots', []) if s.get('status') not in ('RETIRED', 'LEFT') and s.get('team') in ('A', 'B')]
    times = np.array([f['time'] for f in frames], np.float64)
    T = len(frames)
    block = {'frameTimes': [round(float(t), 3) for t in times], 'ids': {}, 'method': f'hybrid-v1 (straight between observations; one-sided: {HYBRID_BASE})',
             'constants': {'tauFormation': TAU_FORMATION, 'tauRole': TAU_ROLE, 'minVoters': MIN_VOTERS, 'maxSigma': MAX_SIGMA, 'maxGap': MAX_GAP,
                           'interpolateGap': INTERPOLATE_GAP, 'shiftSigmaRate': SHIFT_SIGMA_RATE, 'cvSigmaRate': CV_SIGMA_RATE}, 'kinds': {}}
    if not T or not slots:
        return block
    index = {s['id']: k for k, s in enumerate(slots)}
    pos = np.full((T, len(slots), 2), np.nan)
    sig = np.full((T, len(slots)), np.nan)
    calibrated = np.array([bool(f.get('calibrated')) for f in frames])
    for t, f in enumerate(frames):
        if not calibrated[t]:
            continue
        for p in f.get('people', []):
            k = index.get(p.get('id'))
            if k is None or not isinstance(p.get('pitch'), list) or (p.get('pitchConfidence') or 0.0) < min_confidence:
                continue
            pos[t, k] = (p['pitch'][0]*size_length, p['pitch'][1]*size_width)
            s = p.get('pitchSigma')
            sig[t, k] = max(s) if isinstance(s, list) and s else DEFAULT_SIGMA
    counts = {k: 0 for k in KIND_NAMES}
    for team in ('A', 'B'):
        members = [k for k, s in enumerate(slots) if s['team'] == team]
        if not members:
            continue
        est, sigma, kind, _ = estimate_team(pos[:, members], sig[:, members], times, method='hybrid')
        # A frame without a trusted registration places nobody: no observation can be checked against it, so the
        # slots are unknown there (the estimate resumes on the next calibrated frame).
        kind[~calibrated] = KINDS['U']
        for j, k in enumerate(members):
            if not slots[k].get('confirmed', True):
                # An unconfirmed identity (beyond the team's slots: probably not a fielded player) keeps its observed
                # positions and gets no estimate where it was not seen.
                kind[np.isnan(pos[:, k, 0]), j] = KINDS['U']
        for j, k in enumerate(members):
            pid = slots[k]['id']
            xs, ys, kinds, sigmas = [], [], [], []
            for t in range(T):
                code = KIND_NAMES[kind[t, j]]
                counts[code] += 1
                kinds.append(code)
                if code == 'U':
                    xs.append(None)
                    ys.append(None)
                    sigmas.append(None)
                else:
                    xs.append(round(float(est[t, j, 0])/size_length, 4))
                    ys.append(round(float(est[t, j, 1])/size_width, 4))
                    sv = round(float(sigma[t, j]), 1)
                    sigmas.append([sv, sv])
            block['ids'][pid] = {'team': team, 'x': xs, 'y': ys, 'kind': kinds, 'sigma': sigmas}
    total = max(1, sum(counts.values()))
    block['kinds'] = {k: round(v/total, 3) for k, v in counts.items()}
    return block
