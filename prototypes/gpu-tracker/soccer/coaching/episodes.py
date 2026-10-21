"""Episodes of a coaching metric: one continuous tactical situation is one occurrence.

A coach counts situations, not frames. On the Metrica professional sample (game 1, 5 samples a second, both teams)
a threshold crossed sample by sample fires about 25% more often than with hysteresis, in shorter pieces, and a
10 s spell of a large gap between defence and midfield would otherwise be fifty "events". So an episode starts when
the value crosses ENTER, lasts while it stays beyond EXIT (hysteresis: a value hovering at the threshold is one
episode, not many), and ends when it comes back past EXIT. Episodes closer than MERGE seconds are one (the same
situation, briefly interrupted), and an episode shorter than MIN seconds is not counted (a passing moment). Where the
metric cannot be measured (a player off camera, no registration) for longer than MERGE seconds the episode ends there
and is marked censored: it was not seen to end, and a censored episode at the start of the data was not seen to begin.
Nothing is ever bridged across an unmeasured stretch longer than MERGE seconds.

The same function computes the professional baseline (tools/coaching_bench.py on Metrica game 1) and the detections
on a PitchIQ run, so a count and its baseline follow one rule. Times are absolute seconds; each sample stands for one
sample step (the median spacing of the samples), so an episode's duration is last - first + step.
"""
import hashlib
import math

import numpy as np

ENGINE_VERSION = '1'        # bump on any change to a detector, a constant or a template (part of every stable id)


def stable_id(prefix, *parts):
    """A short id that is the same whenever the same thing is found again: prefix-<12 hex of sha1(parts|version)>.
    Floats are rounded to 0.1 s so a regeneration that moves an edge by a sample keeps the id."""
    def norm(p):
        if isinstance(p, float):
            return f'{round(p, 1):.1f}'
        return str(p)
    text = '|'.join(norm(p) for p in parts)+'|'+ENGINE_VERSION
    return f'{prefix}-{hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]}'


def sample_step(t):
    """The samples' typical spacing (median of the positive steps); 0.2 s (5 Hz) when it cannot be measured."""
    t = np.asarray(t, float)
    d = np.diff(t)
    d = d[np.isfinite(d) & (d > 1e-9)]
    return float(np.median(d)) if len(d) else .2


def measured_seconds(t, eligible, step=None):
    """Seconds of measured (eligible) time: each eligible sample stands for one sample step."""
    step = sample_step(t) if step is None else step
    return float(np.count_nonzero(np.asarray(eligible, bool)))*step


def stretches(t, eligible, merge_seconds=0.0, step=None):
    """Continuous eligible stretches [(t0, t1)], gaps up to merge_seconds bridged: the measurable spells."""
    t = np.asarray(t, float)
    ok = np.asarray(eligible, bool)
    step = sample_step(t) if step is None else step
    out = []
    idx = np.flatnonzero(ok)
    if not len(idx):
        return out
    start = prev = idx[0]
    for i in idx[1:]:
        if t[i]-t[prev] > step*1.5+merge_seconds:
            out.append((float(t[start]), float(t[prev])))
            start = i
        prev = i
    out.append((float(t[start]), float(t[prev])))
    return out


def longest_stretch(t, eligible, step=None):
    """The longest continuous eligible stretch in seconds (last - first + step)."""
    step = sample_step(t) if step is None else step
    s = stretches(t, eligible, 0.0, step)
    return max((b-a+step for a, b in s), default=0.0)


def episodes(t, value, eligible, enter, exit, min_seconds=5.0, merge_seconds=3.0, direction='above', step=None):
    """The episodes of value beyond enter (direction 'above': value >= enter starts one, value < exit ends it;
    'below' mirrors it), over the samples where eligible is true and value is finite. Returns, in time order,
    [{'t0', 't1', 'i0', 'i1' (sample indices), 'peakT', 'peakIndex', 'peak', 'mean', 'samples', 'seconds',
    'censoredStart', 'censoredEnd'}]. A gap in eligibility longer than merge_seconds ends an episode (censoredEnd);
    an episode that begins on the first measured sample after such a gap (or of the series) already beyond enter is
    censoredStart. Episodes at most merge_seconds apart are merged; shorter than min_seconds are dropped."""
    t = np.asarray(t, float)
    v = np.asarray(value, float)
    ok = np.asarray(eligible, bool) & np.isfinite(v)
    if direction not in ('above', 'below'):
        raise ValueError('direction must be above or below')
    if exit is None:
        exit = enter
    sign = 1.0 if direction == 'above' else -1.0
    if sign*exit > sign*enter:
        raise ValueError('exit must not be beyond enter (hysteresis)')
    step = sample_step(t) if step is None else step
    s = sign*v
    e_in, e_out = sign*enter, sign*exit
    raw = []
    cur = None
    last_ok = None
    for i in range(len(t)):
        if not ok[i]:
            continue
        gap = last_ok is not None and t[i]-t[last_ok] > step*1.5+merge_seconds
        if cur is not None and gap:
            cur['censoredEnd'] = True
            raw.append(cur)
            cur = None
        fresh = last_ok is None or gap          # first measured sample of a stretch
        if cur is None:
            if s[i] >= e_in:
                cur = {'i0': i, 'i1': i, 'censoredStart': bool(fresh), 'censoredEnd': False, 'idx': [i]}
        else:
            if s[i] >= e_out:
                cur['i1'] = i
                cur['idx'].append(i)
            else:
                raw.append(cur)
                cur = None
                if s[i] >= e_in:                # cannot happen with exit <= enter, kept for clarity
                    cur = {'i0': i, 'i1': i, 'censoredStart': False, 'censoredEnd': False, 'idx': [i]}
        last_ok = i
    if cur is not None:
        cur['censoredEnd'] = True               # the data ended inside it
        raw.append(cur)
    # Merge episodes at most merge_seconds apart (the same situation, briefly interrupted).
    merged = []
    for ep in raw:
        if merged and t[ep['i0']]-t[merged[-1]['i1']] <= merge_seconds+1e-9 and not merged[-1]['censoredEnd']:
            prev = merged[-1]
            prev['i1'] = ep['i1']
            prev['idx'] += ep['idx']
            prev['censoredEnd'] = ep['censoredEnd']
        else:
            merged.append(dict(ep))
    out = []
    for ep in merged:
        idx = np.asarray(ep['idx'], int)
        seconds = float(t[ep['i1']]-t[ep['i0']]+step)
        if seconds < min_seconds-1e-9:
            continue
        k = idx[int(np.argmax(s[idx]))]
        out.append({'t0': float(t[ep['i0']]), 't1': float(t[ep['i1']]), 'i0': int(ep['i0']), 'i1': int(ep['i1']),
                    'peakT': float(t[k]), 'peakIndex': int(k), 'peak': float(v[k]), 'mean': float(np.mean(v[idx])),
                    'samples': int(len(idx)), 'seconds': seconds,
                    'censoredStart': bool(ep['censoredStart']), 'censoredEnd': bool(ep['censoredEnd'])})
    return out


def rate_per_minute(count, seconds):
    """Occurrences per measured minute (None without measured time)."""
    return None if not seconds or seconds <= 0 else float(count)/(seconds/60.0)


def share_beyond(value, eligible, threshold, direction='above'):
    """The share of the measured samples beyond threshold (None without any)."""
    v = np.asarray(value, float)
    ok = np.asarray(eligible, bool) & np.isfinite(v)
    if not ok.any():
        return None
    beyond = v[ok] >= threshold if direction == 'above' else v[ok] <= threshold
    return float(np.mean(beyond))


def finite_or_none(x, digits=3):
    """A JSON-safe rounded float, or None for NaN/inf/None."""
    if x is None:
        return None
    x = float(x)
    return round(x, digits) if math.isfinite(x) else None
