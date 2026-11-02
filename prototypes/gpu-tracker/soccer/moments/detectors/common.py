"""Helpers the detectors share: people at a sample, the moment window, reliability and phase records."""
import numpy as np

from .. import context as cx
from .. import engine as en

BEFORE_S = 8.0          # lead-up shown before the key time, by default
AFTER_S = 5.0           # follow-up shown after it
MIN_BEFORE_S, MAX_BEFORE_S = 5.0, 12.0


def row_of(data, s, pid):
    r = data.rows(s)
    hit = np.flatnonzero(data.pid[r] == pid)
    return int(r.start+hit[0]) if len(hit) else -1


def where(data, s, pid):
    """(x, y, vx, vy) of person pid at sample s, or None."""
    i = row_of(data, s, pid)
    if i < 0:
        return None
    return float(data.x[i]), float(data.y[i]), float(data.vx[i]), float(data.vy[i])


def track(data, pid, s0, s1):
    """[(t, x, y)] of person pid over samples s0..s1 (inclusive), where seen."""
    out = []
    for s in range(max(0, s0), min(data.samples-1, s1)+1):
        p = where(data, s, pid)
        if p:
            out.append((float(data.t[s]), p[0], p[1]))
    return out


def ball_at(data, s):
    """(x, y, source) of the ball at sample s: the tracked ball on the pitch ('ball'), else the named carrier ('carrier'),
    else None. Defensive judgements need it: a gap 35 m from the ball is not the same situation as one next to it."""
    b = data.ball[s]
    if np.all(np.isfinite(b)) and data.ballState[s] == 1:
        return float(b[0]), float(b[1]), 'ball'
    c = int(data.possCarrier[s])
    if c >= 0:
        p = where(data, s, c)
        if p:
            return p[0], p[1], 'carrier'
    return None


def window(t_key, start=None, before=BEFORE_S, after=AFTER_S):
    """(before, after) seconds for the moment player: the default lead-up, longer when the situation built up longer
    (from `start`, plus 3 s), never under MIN_BEFORE_S or over MAX_BEFORE_S."""
    if start is not None:
        before = max(before, t_key-start+3.0)
    return round(float(np.clip(before, MIN_BEFORE_S, MAX_BEFORE_S)), 1), round(float(after), 1)


def presence(data, pids, s0, s1):
    """Share of samples s0..s1 in which every person of pids is seen."""
    n = ok = 0
    for s in range(max(0, s0), min(data.samples-1, s1)+1):
        n += 1
        r = data.rows(s)
        here = set(int(p) for p in data.pid[r])
        ok += all(int(p) in here for p in pids)
    return ok/n if n else 0.0


def reliability(data, s, pids, view_share, span=(10, 5)):
    """('high' / 'medium' / 'low', reasons): identity confidence of the people at the key sample, their presence around
    it (span = samples before / after), the frame's calibration and the share of the relevant area in view."""
    idc = []
    sig = []
    for p in pids:
        i = row_of(data, s, p)
        if i < 0:
            return 'low', ['person_not_seen_at_key']
        idc.append(float(data.idConf[i]))
        sig.append(float(data.sigma[i]))
    pres = presence(data, pids, s-span[0], s+span[1])
    parts = {'identity': en.level_of(min(idc) if idc else 0, .8, .6),
             'presence': en.level_of(pres, .9, .7),
             'calibration': en.level_of(float(data.calibConf[s]), .8, .5),
             'view': en.level_of(view_share, .95, .8),
             'position': 'high' if (max(sig) if sig else 9) <= 1.0 else 'medium' if max(sig) <= 2.0 else 'low'}
    level = en.lower(*parts.values())
    reasons = [f'{k}_{v}' for k, v in parts.items() if v != 'high']
    return level, reasons


FOLLOW_UP_S = 5.0
RECEIVE_M = 2.0


def follow_up(engine, team, s, opponent=None, receiver=None, seconds=FOLLOW_UP_S):
    """What was observed in the `seconds` after sample s (never used to judge the decision): whether the possession record
    shows the ball changing team ('wonBack' for a team defending, 'lost' for a team on the ball; None when the record
    names nobody), and whether the tracked ball came within RECEIVE_M of `opponent` ('opponentReceived') or of the
    team's `receiver` ('received'); None when the ball was not placed near them."""
    data, ctx = engine.data, engine.ctx[team]
    s1 = data.sample_at(float(data.t[s])+seconds)
    span = slice(s, s1+1)
    out = {}
    if ctx.phase[s] == cx.OPP:
        out['wonBack'] = bool(np.any(ctx.phase[span] == cx.OWN)) if np.any(ctx.phase[span] != cx.UNKNOWN) else None
    elif ctx.phase[s] == cx.OWN:
        out['lost'] = bool(np.any(ctx.phase[span] == cx.OPP)) if np.any(ctx.phase[span] != cx.UNKNOWN) else None
    for key, who in (('opponentReceived', opponent), ('received', receiver)):
        if who is None or who < 0:
            continue
        got, seen = False, False
        for k in range(s, s1+1):
            b = data.ball[k]
            if not (np.all(np.isfinite(b)) and data.ballState[k] == 1):
                continue
            p = where(data, k, who)
            if p is None:
                continue
            seen = True
            if np.hypot(b[0]-p[0], b[1]-p[1]) <= RECEIVE_M:
                got = True
                break
        out[key] = got if seen else None
    return out


def phase_info(ctx, s):
    return {'state': cx.PHASE_NAMES[int(ctx.phase[s])], 'settled': bool(ctx.settled[s]),
            'confidence': ['low', 'medium', 'high'][int(ctx.data.possConf[s])] if ctx.phase[s] in (cx.OWN, cx.OPP) else None}


def roles_of(engine, team, s, pids):
    out = {}
    for p in pids:
        rec = engine.role(team, p, s)
        if rec:
            out[int(p)] = {'role': rec['role'], 'confidence': rec['confidence']}
    return out


def score_reliability(level):
    return {'high': 1.0, 'medium': .8, 'low': .45}[level]


def score_interpretation(level):
    return {'clear': 1.0, 'plausible': .8, 'uncertain': .5}[level]
