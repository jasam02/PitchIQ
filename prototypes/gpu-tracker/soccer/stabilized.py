"""The 2D replay's stabilized positions as a block of their own on a finished result, result['stabilized'] (README, "2D
replay stabilization"): soccer/trajectory.py's stabilized observations per roster slot and anonymous '?' track, and
imputation.estimate's estimates run on them. pipeline.run attaches it to the finished report, once the summary lists
the registration's anchors, so that it is the block built from the saved result.json; corrections.apply_result builds
it again after a correction. The raw positions (frames[].people[].pitch) and result.timeline are never
touched: the analytics, team shape, visibility and the tools keep reading them.

Light on purpose: numpy and the soccer modules, the stabilizer imported only when a block is built, so corrections and
the offline tools build the block without the detector stack (pipeline.py keeps the older names stabilized_block and
attach_stabilized for the callers that look for them there).

What a block guarantees; otherwise attach() leaves it out and the 2D view draws the raw positions, as for a run without
one:
- it keeps the timeline's observations. A frame the timeline counts as observed (a slot's O; an anonymous track's
  observation, by the same rule) that the stabilizer leaves unplaced (no position, not finite, or beyond the drawable
  pitch) is drawn as the raw observation and counted (summary rawFrames; rawSlots names a slot it did not return at
  all). More than MAX_RAW_SHARE of them, or stabilized observations a median of more than MAX_MEDIAN_MOVE away from the
  raw ones, and the stabilizer is taken to have failed;
- a frame without a trusted registration places nobody, as in the timeline; a result without one gets no block;
- a correction moves only what it corrects, when the stabilizer takes a noise model: the corrected result's block is
  built with the original run's fitted noise model (noise_of: the pixel noise, both passes of the geometric height's
  fit, the footage's edge-fit level), and the registration's snaps are measured at every person on the frame, so the
  identities the correction did not touch keep their stabilized observations exactly (eleven exclusions on the three
  regression runs: none moved; imputation's estimates for the corrected team follow, as in the timeline) (summary
  noise: 'kept'; 'refitted' where the stabilizer has no noise parameter and fitted it again);
- it serializes with allow_nan=False and without a default for numpy numbers (the server writes the result again).
"""
import copy
import inspect
import json
import math
import time
import traceback
from collections import Counter

import numpy as np

from . import field, imputation

VERSION = 1
OUTSIDE = .1             # normalized: how far beyond the lines a stabilized position may lie (tactical.js RULES.outside: the
                         # board draws nothing further out; no timeline observation of the Oct 18 runs lies there)
MAX_RAW_SHARE = .01      # the share of the timeline's observations the stabilizer may leave to be drawn raw (0 on the
                         # amateur, Veo and broadcast runs of Oct 18) before its block is left out
MAX_MEDIAN_MOVE = 2.0    # m: the median distance of the stabilized observations from the raw ones beyond which the block is
                         # left out (a stabilizer that moved everybody: units, axes, pitch size); 0.17 / 0.29 / 0.10 m on the
                         # amateur, Veo and broadcast runs
FLAG_WEIGHT = .5         # weights under this count as down-weighted (the stabilizer's constants.flagWeight when it gives one)


class NothingToPlace(ValueError):
    """A result no block can place anyone on (no frame with a trusted registration): attach() skips it."""


def _rounded(value, digits):
    """A finite number as a plain float rounded to digits, else None: the block goes into result.json, which is written
    with allow_nan=False (and written again by the server without a default for numpy numbers)."""
    if value is None or isinstance(value, bool):
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return round(value, digits) if math.isfinite(value) else None


def _plain(value):
    """The stabilizer's own blocks (noise, summary, constants, flags) as plain JSON: numpy numbers and arrays as Python
    ones, NaN and infinities as None."""
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_plain(v) for v in value]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else None
    return value


def _pair(a, b, digits):
    a, b = _rounded(a, digits), _rounded(b, digits)
    return None if a is None or b is None else [a, b]


def _weight(value):
    """An observation's weight to 2 decimals; a whole one (1 normal, 0 rejected: most of them) as an int."""
    w = _rounded(value, 2)
    return int(w) if w is not None and w == int(w) else w


def _takes_noise(stabilize):
    """Whether the stabilizer has a noise parameter (a fitted noise model to use instead of fitting one)."""
    try:
        return 'noise' in inspect.signature(stabilize).parameters
    except (TypeError, ValueError):
        return False


def noise_of(result):
    """The noise model the stabilizer fitted for a result's block (result.stabilized.noise), for building the block of
    the same run corrected; None without a block or a model."""
    made = result.get('stabilized')
    noise = made.get('noise') if isinstance(made, dict) else None
    return copy.deepcopy(noise) if isinstance(noise, dict) and noise else None


def block(result, width, height, size=None, stabilize=None, noise=None):
    """(result['stabilized'], the run summary's 'stabilizer' entry) of a finished result. Per roster slot (the keys of
    result.timeline.ids) and frame: the normalized x / y, the kind (O/I/P/U, as in the timeline), sigma [along, across]
    in metres and w, the stabilizer's weight of the observation (1 normal, under 1 down-weighted, 0 rejected); per
    anonymous '?' track its team, x / y, kind and sigma; the stabilizer's flags, noise model, summary and constants.
    O and I are the stabilizer's (its smoothed observations and the short gaps it bridges, also on an unconfirmed slot,
    where the timeline's raw sampler bridged the same gaps); P, U and imputation's own I come from imputation.estimate
    run on the stabilized observations with the raw observation's sigma at least, so a prediction starts where the
    stabilized player was and turns P or U where the timeline's does. See the module docstring for what is kept.
    size: the pitch (field.PitchSize), the result's own when None. stabilize: the stabilizer, trajectory.stabilize when
    None (the tests hand in a stub). noise: a noise model to use instead of fitting one (noise_of the uncorrected
    result), handed on when the stabilizer takes one. Raises what the stabilizer raises, ValueError for a block that
    would not keep the timeline's observations, NothingToPlace without a trusted registration (attach keeps them all
    from the run)."""
    started = time.perf_counter()
    frames = result.get('frames') or []
    timeline = result.get('timeline') or {}
    T = len(frames)
    if not T or len(timeline.get('frameTimes') or []) != T:
        raise ValueError('the result has no timeline for its frames')
    calibrated = [bool(f.get('calibrated')) for f in frames]
    if not any(calibrated):
        raise NothingToPlace('no frame has a trusted registration: nobody to place')
    if stabilize is None:
        # Imported here: an error in the stabilizer costs the 2D replay its block, never its callers their import.
        from soccer import trajectory
        stabilize = trajectory.stabilize
    size = size or field.PitchSize.from_json((result.get('tactical') or {}).get('pitchSize'))
    L, W = float(size.length), float(size.width)
    if noise is not None and _takes_noise(stabilize):
        out, noise_source = stabilize(result, width, height, size, noise=copy.deepcopy(noise)), 'kept'
    else:
        out, noise_source = stabilize(result, width, height, size), 'fitted' if noise is None else 'refitted'
    stabilized_at = time.perf_counter()
    if len(out.get('frameTimes') or []) != T:
        raise ValueError(f"the stabilizer gave {len(out.get('frameTimes') or [])} frame times for {T} frames")
    smoothed = out.get('ids') or {}
    smoothed_anonymous = {str(k): v for k, v in (out.get('anonymous') or {}).items()}
    constants = _plain(out.get('constants') or {})
    flag_weight = constants.get('flagWeight') if isinstance(constants.get('flagWeight'), (int, float)) else FLAG_WEIGHT

    def at(cols, key, t):
        values = cols.get(key) if cols else None
        return values[t] if values is not None and t < len(values) else None

    def placed(cols, t):
        """(kind, x m, y m) of the stabilizer's O or I on frame t where the board can draw it, else None."""
        kind = at(cols, 'kind', t)
        if kind not in ('O', 'I') or not calibrated[t]:
            return None
        x, y = _rounded(at(cols, 'x', t), 6), _rounded(at(cols, 'y', t), 6)
        if x is None or y is None or not (-OUTSIDE <= x/L <= 1+OUTSIDE and -OUTSIDE <= y/W <= 1+OUTSIDE):
            return None
        return kind, x, y

    # The raw observations the block must keep: per slot the people's positions on the frames the timeline marks O (the
    # one imputation took: imputation.observation's rule, the later of two people with one id); per anonymous track the
    # positions the stabilizer observes by the same rule (an unidentified team person with a box). And the anonymous
    # tracks' teams, from their unidentified observations.
    raw_seen = {pid: {} for pid in timeline.get('ids') or {}}
    anonymous_seen, teams = {}, {}
    for t, f in enumerate(frames):
        for p in f.get('people') or []:
            pid = p.get('id')
            if not pid and p.get('state') == 'uncertain' and p.get('team') in ('A', 'B'):
                teams.setdefault(str(p.get('track')), Counter())[p['team']] += 1
            if not calibrated[t]:
                continue
            if pid is not None:
                if pid not in raw_seen or timeline['ids'][pid]['kind'][t] != 'O':
                    continue
                seen = imputation.observation(p, f, L, W)
                if seen is not None:
                    raw_seen[pid][t] = seen
            elif p.get('state') == 'uncertain' and p.get('team') in ('A', 'B') and p.get('track') is not None and isinstance(p.get('box'), list):
                seen = imputation.observation(p, f, L, W)
                if seen is not None:
                    anonymous_seen.setdefault(str(p['track']), {})[t] = seen

    # The stabilized observations are imputation's observations, with the raw observation's sigma at least (its
    # predictions then grow from the timeline's sigma and turn U where the timeline's do: the stabilized position is
    # better than the raw one, the gap after it is not). A timeline observation the stabilizer left unplaced is the raw
    # observation; a slot it did not return keeps its people's positions (imputation reads them itself).
    observations, fallback, raw_slots = {}, set(), []
    for pid, seen in raw_seen.items():
        cols = smoothed.get(pid)
        if cols is None:
            raw_slots.append(pid)
            continue
        xs, ys, sigmas = [None]*T, [None]*T, [None]*T
        for t in range(T):
            p, r = placed(cols, t), seen.get(t)
            if p is not None and p[0] == 'O':
                xs[t], ys[t] = p[1], p[2]
                larger = [s for s in (_rounded(at(cols, 'sx', t), 3), _rounded(at(cols, 'sy', t), 3), r[2] if r else None) if s is not None]
                sigmas[t] = max(larger) if larger else None
            elif p is None and r is not None:
                xs[t], ys[t], sigmas[t] = r
                fallback.add((pid, t))
        observations[pid] = (xs, ys, sigmas)
    estimated = imputation.estimate(result, L, W, observations=observations)['ids']
    unknown = {'kind': ['U']*T, 'x': [None]*T, 'y': [None]*T, 'sigma': [None]*T}
    ids, kinds = {}, Counter()
    weights = {'observations': 0, 'rejected': 0, 'downWeighted': 0}
    moved = []   # metres between the raw observation and the stabilized one on the same frame (roster slots)
    for pid, raw in timeline.get('ids', {}).items():
        cols, est = smoothed.get(pid), estimated.get(pid) or unknown
        entry = {'team': raw.get('team'), 'x': [], 'y': [], 'kind': [], 'sigma': [], 'w': []}
        for t in range(T):
            p = placed(cols, t)
            if p is not None:
                kind = p[0]
                entry['x'].append(round(p[1]/L, 4))
                entry['y'].append(round(p[2]/W, 4))
                entry['sigma'].append(_pair(at(cols, 'sx', t), at(cols, 'sy', t), 1))
                r = raw_seen[pid].get(t)
                if kind == 'O' and r is not None:
                    moved.append(math.hypot(p[1]-r[0], p[2]-r[1]))
            else:
                kind = est['kind'][t]
                entry['x'].append(est['x'][t])
                entry['y'].append(est['y'][t])
                entry['sigma'].append(est['sigma'][t])
            w = _weight(at(cols, 'w', t)) if p is not None else None
            entry['w'].append(w)
            entry['kind'].append(kind)
            kinds[kind] += 1
            if w is not None:
                weights['observations'] += 1
                weights['rejected'] += w <= 0
                weights['downWeighted'] += 0 < w < flag_weight
        ids[pid] = entry
    # The anonymous '?' tracks: the stabilizer's, then any track it observed none of; an observation it left unplaced is
    # the raw one.
    anonymous, anonymous_raw = {}, 0
    for key in list(smoothed_anonymous)+[k for k in anonymous_seen if k not in smoothed_anonymous]:
        cols, seen, votes = smoothed_anonymous.get(key), anonymous_seen.get(key, {}), teams.get(key)
        entry = {'team': votes.most_common(1)[0][0] if votes else None, 'x': [], 'y': [], 'kind': [], 'sigma': []}
        for t in range(T):
            p, r = placed(cols, t), seen.get(t)
            if p is not None:
                kind, x, y, sigma = p[0], round(p[1]/L, 4), round(p[2]/W, 4), _pair(at(cols, 'sx', t), at(cols, 'sy', t), 1)
            elif r is not None:
                kind, x, y, sigma = 'O', round(r[0]/L, 4), round(r[1]/W, 4), [round(r[2], 1)]*2
                anonymous_raw += 1
            else:
                kind, x, y, sigma = 'U', None, None, None
            entry['kind'].append(kind)
            entry['x'].append(x)
            entry['y'].append(y)
            entry['sigma'].append(sigma)
        if any(k != 'U' for k in entry['kind']):
            anonymous[key] = entry
    # Does the block keep the timeline's observations?
    observed = sum(len(seen) for seen in raw_seen.values())+sum(len(seen) for seen in anonymous_seen.values())
    raw_frames = len(fallback)+sum(len(raw_seen[pid]) for pid in raw_slots)+anonymous_raw
    if raw_frames > MAX_RAW_SHARE*observed:
        raise ValueError(f'the stabilizer left {raw_frames} of the {observed} observations unplaced'
                         + (f' (slots it did not return: {", ".join(sorted(raw_slots))})' if raw_slots else ''))
    moved.sort()

    def quantile(q):
        return round(moved[min(len(moved)-1, int(q*len(moved)))], 2) if moved else None
    if moved and moved[len(moved)//2] > MAX_MEDIAN_MOVE:
        raise ValueError(f'the stabilized observations are a median {moved[len(moved)//2]:.1f} m from the raw ones')
    flags = []
    for f in out.get('flags') or []:
        flag = _plain(f)
        flag['time'] = _rounded(f.get('time'), 3)
        for k in ('raw', 'used'):
            v = f.get(k)
            flag[k] = _pair(v[0], v[1], 2) if isinstance(v, (list, tuple, np.ndarray)) and len(v) == 2 else None
        flags.append(flag)
    own = _plain(out.get('summary') or {})
    own.pop('runtimeSeconds', None)   # the time is the run summary's: the same result gives the same block
    made = {'version': VERSION, 'method': str(out.get('method') or 'stabilizer'), 'frameTimes': list(timeline['frameTimes']), 'ids': ids,
            'anonymous': anonymous, 'flags': flags, 'summary': own, 'noise': _plain(out.get('noise') or {}), 'constants': constants}
    total = max(1, sum(kinds.values()))
    # The run summary's entry. kinds and slots cover the roster slots (kinds as summary.timeline's; slots: the weighted
    # observations, rejected 0, down-weighted under flagWeight, and how far the stabilized observations are from the
    # raw ones); allTracks is the stabilizer's own count over every track, anonymous ones included, and the flags.
    summary = {'status': 'ok', 'noise': noise_source, 'ids': len(ids), 'anonymous': len(anonymous),
               'rawSlots': sorted(raw_slots), 'rawFrames': raw_frames, 'rawShare': round(raw_frames/max(1, observed), 4),
               'kinds': {k: round(kinds[k]/total, 3) for k in imputation.KIND_NAMES},
               'slots': {'observations': weights['observations'],
                         'rejectedShare': round(weights['rejected']/max(1, weights['observations']), 4),
                         'downweightedShare': round(weights['downWeighted']/max(1, weights['observations']), 4),
                         'correctionMetres': {'samples': len(moved), 'median': quantile(.5), 'p90': quantile(.9),
                                              'max': round(moved[-1], 2) if moved else None}},
               'allTracks': {**{k: own.get(k) for k in ('observations', 'rejectedShare', 'downweightedShare', 'repairedShare', 'splits')},
                             'flags': len(flags), 'flagReasons': dict(Counter(str(f.get('reason')) for f in flags))},
               'seconds': {'stabilize': round(stabilized_at-started, 2), 'total': round(time.perf_counter()-started, 2)}}
    return made, summary


def attach(result, width, height, size=None, stabilize=None, noise=None):
    """Put block()'s block into result['stabilized'] (replacing an older one) and return the run summary's 'stabilizer'
    entry. A stabilizer that fails, a block that would not keep the timeline's observations or would not serialize,
    never fails the run or the correction: the block is left out, which the 2D view reads as "draw the raw positions"
    (tactical.js), and the entry is {'status': 'failed', 'error', 'seconds'} (the console has the traceback), or
    {'status': 'skipped', 'reason', 'seconds'} for a result nobody can be placed on."""
    result.pop('stabilized', None)
    started = time.perf_counter()
    try:
        made, summary = block(result, width, height, size, stabilize, noise)
        summary['bytes'] = len(json.dumps(made, separators=(',', ':'), allow_nan=False))
    except NothingToPlace as why:
        return {'status': 'skipped', 'reason': str(why), 'seconds': {'total': round(time.perf_counter()-started, 2)}}
    except Exception as error:   # noqa: BLE001 - the stabilized block is optional; the raw positions stay
        traceback.print_exc()
        print(f'2D replay stabilization failed; the 2D view keeps the raw positions: {type(error).__name__}: {error}', flush=True)
        return {'status': 'failed', 'error': f'{type(error).__name__}: {error}'[:300], 'seconds': {'total': round(time.perf_counter()-started, 2)}}
    result['stabilized'] = made
    return summary
