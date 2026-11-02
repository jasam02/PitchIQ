"""Alternative positions for an improvement moment: who could have moved where, and whether they could have got there.

Never "back to the formation spot". For each kind of situation the candidate answers are tactical (a teammate rotates into
the space, the player holds a deeper position, a neighbour narrows the gap), each a ZONE (a disc of ZONE_R metres, or a
band) around a point chosen from the situation's own geometry. A candidate is kept only when:
    reach      the mover gets to the zone (geometry.time_to_reach) within the time the situation allowed: the nearest
               opponent's time to the same point plus REACH_SLACK_S (or AVAILABLE_S when no opponent is involved); for
               the cover behind an advancing full back, plus the time since the advance began (up to LEAD_S): an
               'exposed' channel means nobody could get there from where they stood at the key moment, so the useful
               question is whether a rotation started WITH the run would have (the text says so)
    no new gap the mover was not the only player of the team within GUARD_M of another opponent (moving them would free
               that opponent)
    not on top the zone's centre is not within CROWD_M of another teammate (someone is already there: then the answer is
               a different one)
Every suggestion holds the others where they were (one mover at a time), and its text says so.
A DECISION-level option (the advanced player holding a deeper position, the defender holding the line, stepping up level)
is about the choice made before the key moment, not a movement at it: it is not refused for the player being on the ball
or for the time left, only for freeing an opponent or standing on a teammate.

Options are tagged by complexity so the level picks: 'simple' (one player, one instruction: levels 1-3 see only these),
'unit' (a rotation or a shift: from level 4), 'structure' (a team-shape answer with its trade-off: levels 6-7 also see
the alternatives).
"""
import math

import numpy as np

from . import geometry as geo
from .detectors import common as cm

ZONE_R = 4.0
REACH_SLACK_S = .5
LEAD_S = 4.0
AVAILABLE_S = 2.5
GUARD_M = 8.0
CROWD_M = 2.5
ON_BALL_M = 2.5


def _disc(x, y, r=ZONE_R, n=12):
    return [(float(x+r*math.cos(2*math.pi*k/n)), float(y+r*math.sin(2*math.pi*k/n))) for k in range(n)]


def _people(engine, team, s):
    data, ctx = engine.data, engine.ctx[team]
    idx, pids = ctx.players(s)
    oidx, opids = ctx.opponents(s)
    return idx, pids, oidx, opids


def _reach(data, i, x, y):
    return float(geo.time_to_reach(data.x[i], data.y[i], data.vx[i], data.vy[i], x, y))


def _opp_time(data, oidx, x, y):
    if not len(oidx):
        return np.inf
    return float(geo.time_to_reach(data.x[oidx], data.y[oidx], data.vx[oidx], data.vy[oidx], x, y).min())


def _check(engine, team, s, mover, x, y, budget, decision=False):
    """(ok, reasons, reach seconds) for moving `mover` (a person index) to (x, y) at sample s."""
    data = engine.data
    idx, pids, oidx, opids = _people(engine, team, s)
    rows = {int(p): int(i) for i, p in zip(idx, pids)}
    if mover not in rows:
        return False, ['mover_not_seen'], None
    if not decision and mover in on_ball(data, s):
        return False, ['mover_on_the_ball'], None
    i = rows[mover]
    t = _reach(data, i, x, y)
    reasons = []
    if t > budget and not decision:
        reasons.append('cannot_reach')
    for j, q in zip(oidx, opids):
        d_me = math.hypot(data.x[i]-data.x[j], data.y[i]-data.y[j])
        if d_me > GUARD_M:
            continue
        others = [r for p, r in rows.items() if p != mover and math.hypot(data.x[r]-data.x[j], data.y[r]-data.y[j]) <= GUARD_M]
        if not others and math.hypot(x-data.x[j], y-data.y[j]) > GUARD_M:
            reasons.append('frees_an_opponent')
            break
    for p, r in rows.items():
        if p != mover and math.hypot(data.x[r]-x, data.y[r]-y) <= CROWD_M:
            reasons.append('already_occupied')
            break
    return not reasons, reasons, round(t, 2)


DECISIONS = ('hold_deeper', 'hold_the_line', 'step_up_level')


def _option(engine, team, s, mover, x, y, budget, label, complexity, extra=None):
    ok, reasons, t = _check(engine, team, s, mover, x, y, budget, decision=label in DECISIONS)
    data = engine.data
    p = cm.where(data, s, mover)
    return {'label': label, 'complexity': complexity, 'mover': int(mover), 'point': (float(x), float(y)),
            'zone': _disc(x, y), 'from': (p[0], p[1]) if p else None, 'reachS': t, 'budgetS': round(float(budget), 2),
            'valid': ok, 'rejected': reasons} | (extra or {})


def on_ball(data, s):
    """Person indices on the ball at sample s (the named carrier, anyone within ON_BALL_M of any estimate of the ball):
    never suggested as the one to move (checked on video: at 27:38 on CAN-ISL the suggestion asked the player carrying
    the ball to move into the channel)."""
    out = set()
    if int(data.possCarrier[s]) >= 0:
        out.add(int(data.possCarrier[s]))
    b = data.ball[s]
    if np.all(np.isfinite(b)):
        r = data.rows(s)
        near = np.hypot(data.x[r]-b[0], data.y[r]-b[1]) <= ON_BALL_M
        out.update(int(p) for p in data.pid[r][near])
    return out


def _nearest(engine, team, s, x, y, exclude=(), roles=None):
    data, ctx = engine.data, engine.ctx[team]
    idx, pids = ctx.players(s)
    best = None
    exclude = set(exclude) | on_ball(data, s)
    for i, p in zip(idx, pids):
        if int(p) in exclude:
            continue
        if roles:
            r = engine.role(team, p, s)
            if not r or r['role'] not in roles:
                continue
        t = _reach(data, i, x, y)
        if best is None or t < best[1]:
            best = (int(p), t)
    return best


def options(engine, ep):
    """Candidate alternatives for an improvement episode, best first (valid ones before rejected ones)."""
    data, team = engine.data, ep['team']
    ctx = engine.ctx[team]
    s = ep['sKey']
    oidx = ctx.opponents(s)[0]
    kind = ep['kind']
    focus = ep['players']['focus']
    out = []
    if kind == 'exposed_channel':
        kx, ky = ep['zones']['coverPoint']
        lead = min(LEAD_S, max(0.0, float(ep['tKey'])-float(ep['t0'])))
        budget = _opp_time(data, oidx, kx, ky)+REACH_SLACK_S+lead
        best = _nearest(engine, team, s, kx, ky, exclude=set(focus))
        if best:
            out.append(_option(engine, team, s, best[0], kx, ky, budget, 'rotate_cover', 'unit'))
        line = ep['measures']['lineDepthM']
        side = 1.0 if ep['measures']['side'] == 'left' else -1.0
        hx, hy = ctx.to_pitch(line+8.0, side*22.0, s)
        out.append(_option(engine, team, s, focus[0], hx, hy, AVAILABLE_S*2, 'hold_deeper', 'simple'))
        cb = _nearest(engine, team, s, kx, ky, exclude=set(focus)|({best[0]} if best else set()), roles=('LCB', 'CB', 'RCB'))
        if cb:
            p = cm.where(data, s, cb[0])
            if p:
                mx, my = (p[0]+kx)/2, (p[1]+ky)/2
                out.append(_option(engine, team, s, cb[0], mx, my, budget, 'shift_across', 'structure'))
    elif kind in ('stepped_out_open',):
        px, py = ep['zones']['point']
        budget = _opp_time(data, oidx, px, py)+REACH_SLACK_S
        best = _nearest(engine, team, s, px, py, exclude=set(focus))
        if best:
            role = engine.role(team, best[0], s)
            label = 'midfielder_drops' if role and role['line'] == 'M' else 'neighbour_narrows'
            out.append(_option(engine, team, s, best[0], px, py, budget, label, 'unit'))
        p = cm.where(data, s, focus[0])
        if p:
            out.append(_option(engine, team, s, focus[0], px, py, AVAILABLE_S*2, 'hold_the_line', 'simple'))
    elif kind == 'gap_attacker':
        gx, gy = ep['zones']['point']
        budget = _opp_time(data, oidx, gx, gy)+REACH_SLACK_S
        for pid in focus:
            p = cm.where(data, s, pid)
            if p:
                mx, my = (2*gx+p[0])/3, (2*gy+p[1])/3
                out.append(_option(engine, team, s, pid, mx, my, budget, 'narrow_the_gap', 'simple'))
        dm = _nearest(engine, team, s, gx, gy, exclude=set(focus), roles=('DM', 'CM'))
        if dm:
            out.append(_option(engine, team, s, dm[0], gx, gy, budget, 'midfielder_drops', 'structure'))
    elif kind == 'dropped_off':
        line = ep['measures']['lineDepthM']
        p = cm.where(data, s, focus[0])
        if p:
            lat = float(ctx.lateral(p[1], s))
            hx, hy = ctx.to_pitch(line, lat, s)
            out.append(_option(engine, team, s, focus[0], hx, hy, AVAILABLE_S, 'step_up_level', 'simple'))
    elif kind == 'free_between_lines':
        ox, oy = ep['zones']['opponent']
        ball = ep['zones'].get('ball')
        if ball:
            vx, vy = ball[0]-ox, ball[1]-oy
            n = math.hypot(vx, vy) or 1.0
            zx, zy = ox+vx/n*2.5, oy+vy/n*2.5
        else:
            zx, zy = ox, oy
        budget = AVAILABLE_S
        # The midfielder dropping onto the receiver is the first answer; a centre-back stepping out of the line is the
        # riskier alternative (shown at levels 6-7 only).
        for pid, label, cx_ in ((ep['players']['focus'][0], 'midfielder_drops', 'unit'),
                                (ep['players']['support'][0] if ep['players']['support'] else None, 'defender_steps_in', 'structure')):
            if pid is not None:
                out.append(_option(engine, team, s, pid, zx, zy, budget, label, cx_))
    elif kind == 'double_press_free':
        free = ep['players']['opponents'][1] if len(ep['players']['opponents']) > 1 else None
        carrier = ep['players']['opponents'][0]
        c, f = cm.where(data, s, carrier), cm.where(data, s, free) if free is not None else None
        if c and f and len(focus) > 1:
            zx, zy = (c[0]+2*f[0])/3, (c[1]+2*f[1])/3
            out.append(_option(engine, team, s, focus[1], zx, zy, AVAILABLE_S, 'second_presser_covers', 'simple'))
    elif kind == 'press_no_cover':
        carrier = ep['players']['opponents'][0]
        c = cm.where(data, s, carrier)
        if c:
            gx, gy = ctx.own_goal(s)
            vx, vy = gx-c[0], gy-c[1]
            n = math.hypot(vx, vy) or 1.0
            zx, zy = c[0]+vx/n*7.0, c[1]+vy/n*7.0
            best = _nearest(engine, team, s, zx, zy, exclude=set(focus))
            if best:
                out.append(_option(engine, team, s, best[0], zx, zy, AVAILABLE_S, 'cover_the_presser', 'simple'))
    elif kind == 'slow_reorganisation':
        ball = ep['zones'].get('ball')
        if ball:
            bd = float(ctx.depth(ball[0], s))
            zx, zy = ctx.to_pitch(max(8.0, bd-8.0), 0.0, s)
            for pid in focus[:1]:
                out.append(_option(engine, team, s, pid, zx, zy, 4.0, 'recover_central', 'simple'))
    elif kind == 'carrier_isolated':
        c = cm.where(data, s, focus[0])
        if c:
            idx, pids, oidx2, _ = _people(engine, team, s)
            opp_xy = np.column_stack([data.x[oidx2], data.y[oidx2]]) if len(oidx2) else np.zeros((0, 2))
            best = None
            for ang in range(0, 360, 20):
                for r in (10.0, 14.0):
                    x, y = c[0]+r*math.cos(math.radians(ang)), c[1]+r*math.sin(math.radians(ang))
                    if not (0 <= x <= data.length and 0 <= y <= data.width):
                        continue
                    if len(opp_xy) and geo.lane_blocked(opp_xy[:, 0], opp_xy[:, 1], c[0], c[1], x, y, width=2.5):
                        continue
                    near = _nearest(engine, team, s, x, y, exclude=set(focus))
                    if near and (best is None or near[1] < best[2]):
                        best = (x, y, near[1], near[0])
            if best:
                out.append(_option(engine, team, s, best[3], best[0], best[1], AVAILABLE_S, 'offer_an_angle', 'simple'))
    out.sort(key=lambda o: (not o['valid'], {'simple': 0, 'unit': 1, 'structure': 2}[o['complexity']]))
    return out


def choose(opts, level):
    """(main option or None, alternatives) for a coaching level 1-7: levels 1-3 take the simplest valid option (one
    player, one instruction; never a team-structure answer), 4-5 prefer a unit answer (a rotation or a shift) when one
    is valid, 6-7 take the unit answer and list the other valid ones."""
    valid = [o for o in opts if o['valid']]
    if not valid:
        return None, []
    if level <= 3:
        simple = [o for o in valid if o['complexity'] in ('simple', 'unit')]
        return (simple[0] if simple else None), []
    unit = [o for o in valid if o['complexity'] in ('unit', 'structure')]
    main = unit[0] if unit else valid[0]
    alts = [o for o in valid if o is not main] if level >= 6 else []
    return main, alts
