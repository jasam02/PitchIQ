"""The halftime of a full match: where the first half ends, where the second begins, and which way team A attacks in the
first half. Everything that depends on the attacking direction reads it from here; nothing else changes.

The coach sets it on the match timeline (a marker, or two handles when the recording includes the break) and says which
way team A attacks in the first half; until then it is the match plan's break (the periods the match was processed in:
one marker at the first half's end when the gap between them is shorter than BREAK_MIN_S, as when the recording left
the break out; the break's two times when it is longer) and the direction the segments' goal ends voted for
(fullmatch/identities.attacking_directions). Source-video seconds throughout: the match clock is not the video's.

Saved per match in assembly/halftime.json:
    mode             'marker' (one transition time) or 'break' (first half end .. second half start, the interval between
                     them is HALFTIME)
    firstHalfEnd     s
    secondHalfStart  s (= firstHalfEnd in marker mode)
    teamAFirstHalf   'right' | 'left': where team A attacks in the first half (the second half is the opposite)
    confirmed        the coach saved it (False: the defaults)

What it changes: the periods used for directions (periods()), the HALFTIME interval (interval()), the per-sample period
of the key-moments data (apply_to_data()). What it never changes: positions (no coordinate is flipped; views that want
'attacking to the right' turn their own copy), identities, detections, the segments and the periods they were processed
in (index['periods'] stays the processing plan; the direction periods are a separate list). A halftime inside a
segment is fine: everything here is by time.
"""
import hashlib
import json
from pathlib import Path

FILE = 'halftime.json'
BREAK_MIN_S = 120.0       # a gap between the plan's halves this long is a recorded break; shorter, one marker
MODES = ('marker', 'break')
SIDES = ('left', 'right')


def match_range(index):
    periods = index.get('periods') or []
    if not periods:
        return None
    return float(periods[0]['start']), float(periods[-1]['end'])


def defaults(index):
    """The plan's halftime and the inferred first-half direction (see the module doc)."""
    periods = index.get('periods') or []
    rng = match_range(index)
    first_attacker = periods[0].get('attacksRight') if periods else None
    side = None if first_attacker not in ('A', 'B') else ('right' if first_attacker == 'A' else 'left')
    if len(periods) >= 2:
        end, start = float(periods[0]['end']), float(periods[1]['start'])
    elif rng:
        end = start = None
    else:
        end = start = None
    second = periods[1].get('attacksRight') if len(periods) >= 2 else None
    inferred = {'firstHalf': first_attacker, 'secondHalf': second,
                'agreement': [p.get('directionAgreement') for p in periods[:2]],
                'consistent': first_attacker in ('A', 'B') and second in ('A', 'B') and first_attacker != second}
    mode = 'break' if end is not None and start is not None and start-end >= BREAK_MIN_S else 'marker'
    if mode == 'marker' and end is not None:
        start = end           # a gap the recording left out (no frames in it): one transition time
    return {'mode': mode, 'firstHalfEnd': end, 'secondHalfStart': start, 'teamAFirstHalf': side, 'confirmed': False,
            'inferred': inferred, 'range': list(rng) if rng else None}


def load(folder, index):
    """The match's halftime: the saved one over the defaults."""
    out = defaults(index)
    try:
        saved = json.loads((Path(folder)/'assembly'/FILE).read_text(encoding='utf-8'))
    except (OSError, ValueError):
        saved = None
    if isinstance(saved, dict):
        for k in ('mode', 'firstHalfEnd', 'secondHalfStart', 'teamAFirstHalf', 'confirmed', 'at'):
            if k in saved:
                out[k] = saved[k]
    return out


def clean(index, body):
    """Validated settings from a request body, or ValueError with the reason."""
    rng = match_range(index)
    if rng is None:
        raise ValueError('This match has no periods.')
    mode = body.get('mode', 'marker')
    if mode not in MODES:
        raise ValueError('mode must be marker or break.')
    try:
        end = float(body['firstHalfEnd'])
        start = float(body['secondHalfStart']) if mode == 'break' else end
    except (KeyError, TypeError, ValueError):
        raise ValueError('firstHalfEnd (and secondHalfStart for a break) must be times in seconds.') from None
    lo, hi = rng
    if not (lo < end < hi) or not (lo < start < hi):
        raise ValueError(f'Halftime must lie inside the match ({lo:.0f}-{hi:.0f} s of the video).')
    if start < end:
        raise ValueError('The second half cannot start before the first half ends.')
    side = body.get('teamAFirstHalf')
    if side not in SIDES:
        raise ValueError('teamAFirstHalf must be left or right.')
    return {'mode': mode, 'firstHalfEnd': round(end, 3), 'secondHalfStart': round(start, 3), 'teamAFirstHalf': side,
            'confirmed': True}


def save(folder, settings):
    from . import manifest as mf
    mf.write_json(Path(folder)/'assembly'/FILE, dict(settings, at=mf.now()))


def reset(folder):
    try:
        (Path(folder)/'assembly'/FILE).unlink()
    except OSError:
        pass


def interval(settings):
    """(t0, t1) of HALFTIME (empty in marker mode: t0 == t1), or None when the match has no halftime."""
    end, start = settings.get('firstHalfEnd'), settings.get('secondHalfStart')
    if end is None or start is None:
        return None
    return float(end), float(start)


def periods(index, settings):
    """The periods for directions: [{'index', 'name', 'start', 'end', 'attacksRight', 'defendsLeft'}] - the first half
    from the match's start to firstHalfEnd, the second from secondHalfStart to the end (later periods of the plan, extra
    time, kept with their own directions). Without a halftime, the plan's periods with the first-half direction."""
    plan = index.get('periods') or []
    rng = match_range(index)
    if rng is None:
        return []
    side = settings.get('teamAFirstHalf')
    first = None if side not in SIDES else ('A' if side == 'right' else 'B')
    second = None if first is None else ('B' if first == 'A' else 'A')
    half = interval(settings)
    if half is None:
        return [dict(p, attacksRight=first, defendsLeft=first) for p in plan]
    end_regular = float(plan[1]['end']) if len(plan) >= 2 else rng[1]
    out = [{'index': 0, 'name': 'first half', 'start': rng[0], 'end': half[0], 'attacksRight': first, 'defendsLeft': first},
           {'index': 1, 'name': 'second half', 'start': half[1], 'end': end_regular, 'attacksRight': second,
            'defendsLeft': second}]
    for p in plan[2:]:
        out.append(dict(p, index=len(out)))
    return out


def direction_settings(index, settings):
    """team_shape's direction setting: {'A': side team A attacks in the first half... as team_shape names it (the side it
    DEFENDS: team_shape.clean_settings 'A' = left|right goal), 'switchAt': second half start}."""
    side = settings.get('teamAFirstHalf')
    half = interval(settings)
    if side not in SIDES:
        return {'A': None, 'switchAt': None}
    defends = 'left' if side == 'right' else 'right'
    return {'A': defends, 'switchAt': None if half is None else float(half[1])}


def fingerprint(settings):
    keys = ('mode', 'firstHalfEnd', 'secondHalfStart', 'teamAFirstHalf')
    return hashlib.sha256(json.dumps({k: settings.get(k) for k in keys}, sort_keys=True).encode()).hexdigest()[:12]


def apply_to_data(data, index, settings):
    """In place: a MatchData's per-sample period and its periods from the halftime (the key-moments engine reads the
    attacking direction from them): samples inside the HALFTIME interval get period -1 (no direction)."""
    import numpy as np
    plist = periods(index, settings)
    if not plist:
        return data
    t = data.t
    period = np.full(len(t), -1, np.int8)
    for p in plist:
        period[(t >= float(p['start'])-1e-6) & (t <= float(p['end'])+1e-6)] = int(p['index'])
    half = interval(settings)
    if half is not None and half[1] > half[0]:
        period[(t > half[0]) & (t < half[1])] = -1
    data.period = period
    data.periods = [{'index': p['index'], 'start': p['start'], 'end': p['end'], 'name': p.get('name'),
                     'attacksRight': p.get('attacksRight')} for p in plist]
    data.meta = dict(data.meta, periods=data.periods, halftime=fingerprint(settings))
    return data
