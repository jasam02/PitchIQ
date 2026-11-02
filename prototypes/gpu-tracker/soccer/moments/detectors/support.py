"""Detector 5: support for the team's own ball carrier.

During the team's settled possession with a named carrier of the team, seen, and (when the ball is placed) the ball
within BALL_M of them. An OPTION is a teammate OPTION_MIN_M..OPTION_MAX_M from the carrier with an open lane (no opponent
within LANE_M of the straight line from the carrier to them).

    carrier_isolated   IMPROVEMENT (a review: possibly a planned risk)  the carrier under pressure (an opponent within
                       PRESSED_M) with at most one option within NEAR_M, for ISOLATED_S or more, outside the final third
                       (depth < FINAL_THIRD_M) and not a forward (an isolated striker is often the plan)
    offered_support    POSITIVE  a teammate who, while the carrier was under pressure, moved MOVED_M or more in the last
                       LOOK_BACK_S into an open lane within NEAR_M of the carrier that was not open LOOK_BACK_S earlier
The lane test is geometric (straight line, current positions): it does not know the carrier's body shape or the pass that
was played; it is evidence of an available angle, not of a pass.
"""
import math

import numpy as np

from .. import context as cx
from .. import engine as en
from .. import geometry as geo
from . import common as cm

NAME = 'support'
BALL_M = 3.0
OPTION_MIN_M, OPTION_MAX_M, NEAR_M = 6.0, 30.0, 20.0
LANE_M = 2.5
PRESSED_M = 3.0
ISOLATED_S = 1.0
FINAL_THIRD_M = 70.0
MOVED_M, LOOK_BACK_S = 3.0, 2.0
FORWARD_ROLES = ('ST', 'LS', 'RS', 'LW', 'RW')
VIEW_RADIUS_M, VIEW_SHARE = 20.0, .8


def _options(data, cx_, cy_, mates_xy, opp_xy):
    """Open-lane options among teammates (indices into mates_xy) and their distances."""
    out = []
    for k, (mx, my) in enumerate(mates_xy):
        d = math.hypot(mx-cx_, my-cy_)
        if not (OPTION_MIN_M <= d <= OPTION_MAX_M):
            continue
        if len(opp_xy) and geo.lane_blocked(opp_xy[:, 0], opp_xy[:, 1], cx_, cy_, mx, my, width=LANE_M):
            continue
        out.append((k, d))
    return out


def _state(engine, team, s):
    """(carrier xy, pressed by, [(pid, xy)], opp xy array, open option pids within NEAR_M, view) or None."""
    data, ctx = engine.data, engine.ctx[team]
    c = int(data.possCarrier[s])
    if c < 0 or data.team_of[c] != (1 if team == 'A' else 2):
        return None
    p = cm.where(data, s, c)
    if not p:
        return None
    b = data.ball[s]
    if np.all(np.isfinite(b)) and data.ballState[s] == 1 and math.hypot(b[0]-p[0], b[1]-p[1]) > BALL_M:
        return None
    idx, pids = ctx.players(s)
    oidx, opids = ctx.opponents(s)
    mates = [(int(q), (float(data.x[i]), float(data.y[i]))) for i, q in zip(idx, pids) if int(q) != c]
    opp_xy = np.column_stack([data.x[oidx], data.y[oidx]]) if len(oidx) else np.zeros((0, 2))
    dpress = np.hypot(opp_xy[:, 0]-p[0], opp_xy[:, 1]-p[1]) if len(opp_xy) else np.array([np.inf])
    q = int(np.argmin(dpress))
    pressed_by = int(opids[q]) if len(oidx) and dpress[q] <= PRESSED_M else -1
    opts = _options(data, p[0], p[1], [m[1] for m in mates], opp_xy)
    near = [mates[k][0] for k, d in opts if d <= NEAR_M]
    pts = geo.disc_points(p[0], p[1], VIEW_RADIUS_M)
    view = geo.visible_share(data.view[s], pts[0], pts[1], data.length, data.width)
    return {'carrier': c, 'xy': (p[0], p[1]), 'pressedBy': pressed_by, 'mates': mates, 'near': near, 'view': view,
            'depth': float(ctx.depth(p[0], s))}


def _items(engine, team):
    data, ctx = engine.data, engine.ctx[team]
    items = []
    back = int(round(LOOK_BACK_S*5))
    for s in engine.live(team):
        s = int(s)
        if ctx.phase[s] != cx.OWN or not ctx.settled[s] or data.possCarrier[s] < 0:
            continue
        st = _state(engine, team, s)
        if not st or st['view'] < VIEW_SHARE or st['pressedBy'] < 0:
            continue
        role = engine.role(team, st['carrier'], s)
        if len(st['near']) <= 1 and st['depth'] < FINAL_THIRD_M and not (role and role['role'] in FORWARD_ROLES):
            items.append(((team, 'carrier_isolated', st['carrier']), s, float(data.t[s]), dict(st, s=s, t=float(data.t[s]),
                                                                                             kind='carrier_isolated')))
        prev = _state(engine, team, s-back) if s-back >= 0 and data.period[s-back] == data.period[s] else None
        if prev and prev['carrier'] == st['carrier']:
            for q in st['near']:
                if q in prev['near']:
                    continue
                a = cm.where(data, s-back, q)
                b = cm.where(data, s, q)
                if a and b and math.hypot(b[0]-a[0], b[1]-a[1]) >= MOVED_M:
                    items.append(((team, 'offered_support', q), s, float(data.t[s]),
                                  dict(st, s=s, t=float(data.t[s]), kind='offered_support', mover=q,
                                       moved=math.hypot(b[0]-a[0], b[1]-a[1]))))
    return items


def _episode(engine, team, key, recs):
    data, ctx = engine.data, engine.ctx[team]
    kind = key[1]
    if kind == 'carrier_isolated' and recs[-1]['t']-recs[0]['t']+.2 < ISOLATED_S:
        return None
    k = recs[0] if kind == 'offered_support' else recs[len(recs)//2]
    s = k['s']
    if kind == 'carrier_isolated':
        dists = sorted(((math.hypot(xy[0]-k['xy'][0], xy[1]-k['xy'][1]), q) for q, xy in k['mates']))
        focus, support = [k['carrier']], [q for _, q in dists[:2]]
        category, base = 'improvement', .8
    else:
        focus, support = [k['mover']], [k['carrier']]
        category, base = 'positive', .85
    opponents = [k['pressedBy']]
    phase = cm.phase_info(ctx, s)
    level, reasons = cm.reliability(data, s, focus+opponents, k['view'])
    interp = 'plausible' if kind == 'carrier_isolated' else 'clear'
    if phase['confidence'] not in ('medium', 'high'):
        interp = 'plausible' if interp == 'clear' else 'uncertain'
        reasons = reasons+['possession_low_confidence']
    before, after = cm.window(k['t'], start=recs[0]['t'], before=6.0, after=4.0)
    danger = 1.2 if k['depth'] < 35 else 1.0
    score = base*danger*cm.score_reliability(level)*cm.score_interpretation(interp)
    return {'detector': NAME, 'kind': kind, 'team': team, 'category': category,
            't0': recs[0]['t'], 't1': recs[-1]['t'], 'tKey': k['t'], 'sKey': s, 'before': before, 'after': after,
            'players': {'focus': focus, 'support': support, 'opponents': opponents},
            'roles': cm.roles_of(engine, team, s, focus+support) | cm.roles_of(engine, ctx.other, s, opponents),
            'measures': {'optionsNear': len(k['near']), 'carrierDepthM': round(k['depth'], 1),
                         'durationS': round(recs[-1]['t']-recs[0]['t'], 1), 'movedM': round(k.get('moved', 0.0), 1),
                         'viewShare': round(k['view'], 2)},
            'zones': {'carrier': k['xy']}, 'phase': phase,
            'outcome': cm.follow_up(engine, team, s, receiver=k['mover'] if kind == 'offered_support' else None),
            'reliability': level, 'interpretation': interp, 'reasons': reasons, 'score': round(float(score), 3)}


def detect(engine):
    out = []
    for team in 'AB':
        for key, recs in en.runs(_items(engine, team), max_gap_s=.6, min_len_s=0.0):
            ep = _episode(engine, team, key, recs)
            if ep:
                out.append(ep)
    return out
