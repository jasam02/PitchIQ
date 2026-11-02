"""Detector 2: pressure on the ball carrier, cover behind it, balance around it.

Only where the possession record names the OPPONENTS' ball carrier, the spell is settled, and the carrier is seen; when
the ball itself is placed on the pitch it must be within BALL_M of that carrier (else the sample is skipped: who had the
ball would be a guess). Without a named carrier this detector says nothing (ball-carrier pressure is never invented).

    pressure      the team's nearest player within PRESS_M of the carrier, or within CLOSING_M and closing at CLOSING_V
    cover         another player GOAL-SIDE of the carrier (within COVER_ANGLE of the line from the carrier to the team's
                  goal), COVER_MIN_M..COVER_MAX_M from the carrier and deeper than the presser
    double        a second player within PRESS_M of the carrier
    free option   one of the carrier's teammates within FREE_RANGE_M of the carrier with no defender within FREE_M of them,
                  not farther from the team's goal than the carrier + 5 m (a forward or sideways outlet), and the line
                  from the ball to them open (no defender within LANE_M of it)

    press_and_cover   POSITIVE     pressure with cover
    double_press_free IMPROVEMENT  two pressers while a carrier's teammate stands free nearby: one could have covered
    press_no_cover    IMPROVEMENT  pressure, no cover, and nobody else of the team within the cone towards goal within
                                   COVER_MAX_M, within NO_COVER_MAX_DEPTH_M of the team's own goal: one pass or dribble
                                   past the presser and the way to goal is open
No pressure is not judged here (whether to press is the team's plan).
"""
import math

import numpy as np

from .. import context as cx
from .. import engine as en
from .. import geometry as geo
from . import common as cm

NAME = 'pressure'
BALL_M = 3.0
PRESS_M, CLOSING_M, CLOSING_V = 3.5, 5.5, 1.5
COVER_ANGLE, COVER_MIN_M, COVER_MAX_M = 50.0, 4.0, 15.0
FREE_RANGE_M, FREE_M, LANE_M = 18.0, 5.0, 2.5
NO_COVER_MAX_DEPTH_M = 55.0   # 'pressure without cover' only within this of the team's own goal (checked on video: a lone
#                               striker pressing the opposing centre-back 73 m from her own goal is normal pressing)
VIEW_RADIUS_M, VIEW_SHARE = 15.0, .8
MIN_S = {'press_and_cover': 1.0, 'double_press_free': .8, 'press_no_cover': 1.0}
CATEGORY = {'press_and_cover': 'positive', 'double_press_free': 'improvement', 'press_no_cover': 'improvement'}


def _sample(engine, team, s):
    data, ctx = engine.data, engine.ctx[team]
    c = int(data.possCarrier[s])
    if c < 0 or data.team_of[c] != md_team(ctx.other) or not ctx.settled[s] or ctx.phase[s] != cx.OPP:
        return []
    ci = cm.row_of(data, s, c)
    if ci < 0:
        return []
    cxp, cyp = float(data.x[ci]), float(data.y[ci])
    b = data.ball[s]
    if np.all(np.isfinite(b)) and data.ballState[s] == 1 and math.hypot(b[0]-cxp, b[1]-cyp) > BALL_M:
        return []
    idx, pids = ctx.players(s)
    if len(idx) < 4:
        return []
    px, py = data.x[idx], data.y[idx]
    d = np.hypot(px-cxp, py-cyp)
    order = np.argsort(d)
    vx, vy = np.nan_to_num(data.vx[idx]), np.nan_to_num(data.vy[idx])
    ux, uy = (cxp-px)/np.maximum(d, 1e-6), (cyp-py)/np.maximum(d, 1e-6)
    closing = vx*ux+vy*uy
    p0 = int(order[0])
    pressing = d[p0] <= PRESS_M or (d[p0] <= CLOSING_M and closing[p0] >= CLOSING_V)
    if not pressing:
        return []
    gx, gy = ctx.own_goal(s)
    to_goal = (gx-cxp, gy-cyp)
    depth = ctx.depth(px, s)
    cover = []
    for k in range(len(idx)):
        if k == p0 or not (COVER_MIN_M <= d[k] <= COVER_MAX_M) or depth[k] >= depth[p0]:
            continue
        if geo.angle_between(px[k]-cxp, py[k]-cyp, *to_goal) <= COVER_ANGLE:
            cover.append(k)
    second = [k for k in order[1:3] if d[k] <= PRESS_M]
    oidx, opids = ctx.opponents(s)
    free = []
    cdepth = float(ctx.depth(cxp, s))
    for k in range(len(oidx)):
        if int(opids[k]) == c:
            continue
        ox, oy = float(data.x[oidx[k]]), float(data.y[oidx[k]])
        if math.hypot(ox-cxp, oy-cyp) > FREE_RANGE_M or float(ctx.depth(ox, s)) > cdepth+5:
            continue
        if np.hypot(px-ox, py-oy).min() <= FREE_M:
            continue
        # A teammate screening the line from the ball to that opponent makes them not free (checked on video: at 12:32
        # on CAN-ISL a third player stood in that line).
        if geo.lane_blocked(px, py, cxp, cyp, ox, oy, width=LANE_M):
            continue
        free.append(k)
    pts = geo.disc_points(cxp, cyp, VIEW_RADIUS_M)
    view = geo.visible_share(data.view[s], pts[0], pts[1], data.length, data.width)
    if view < VIEW_SHARE:
        return []
    cover.sort(key=lambda k: d[k])       # the nearest cover player is the one named (checked on video: the first in list
    #                                      order was named at 16:07 on CAN-ISL, 13 m away, while the cover 7 m away went unnamed)
    rec = {'s': int(s), 't': float(data.t[s]), 'carrier': c, 'presser': int(pids[p0]), 'pressDist': float(d[p0]),
           'cover': [int(pids[k]) for k in cover], 'coverDist': float(d[cover[0]]) if cover else None,
           'second': [int(pids[k]) for k in second], 'free': [int(opids[k]) for k in free], 'carrierDepth': cdepth,
           'carrierXY': (cxp, cyp), 'view': view}
    if second and free:
        kind = 'double_press_free'
    elif cover:
        kind = 'press_and_cover'
    else:
        cone = [k for k in range(len(idx)) if k != p0 and d[k] <= COVER_MAX_M and depth[k] < cdepth
                and geo.angle_between(px[k]-cxp, py[k]-cyp, *to_goal) <= 70]
        if cone or cdepth > NO_COVER_MAX_DEPTH_M:
            return []
        kind = 'press_no_cover'
    rec['kind'] = kind
    return [((team, kind, c), rec)]


def md_team(letter):
    return 1 if letter == 'A' else 2


def _episode(engine, team, key, recs):
    data, ctx = engine.data, engine.ctx[team]
    kind = key[1]
    if recs[-1]['t']-recs[0]['t']+.2 < MIN_S[kind]:
        return None
    k = recs[len(recs)//2] if kind == 'press_and_cover' else min(recs, key=lambda r: r['carrierDepth'])
    s = k['s']
    focus = [k['presser']]
    if kind == 'double_press_free':
        focus = [k['presser']]+k['second'][:1]
        support = []
    elif kind == 'press_and_cover':
        support = k['cover'][:1]
    else:
        support = []
    opponents = [k['carrier']]+(k['free'][:1] if kind == 'double_press_free' else [])
    phase = cm.phase_info(ctx, s)
    level, reasons = cm.reliability(data, s, focus+support[:1]+opponents[:1], k['view'])
    interp = 'clear' if phase['confidence'] in ('medium', 'high') else 'plausible'
    if phase['confidence'] not in ('medium', 'high'):
        reasons = reasons+['possession_low_confidence']
    before, after = cm.window(k['t'], start=recs[0]['t'], before=6.0, after=4.0)
    danger = 1.3 if k['carrierDepth'] < 35 else 1.0 if k['carrierDepth'] < 60 else .75
    base = {'press_and_cover': .9, 'double_press_free': 1.0, 'press_no_cover': .85}[kind]
    score = base*danger*cm.score_reliability(level)*cm.score_interpretation(interp)*min(1.2, .8+(recs[-1]['t']-recs[0]['t'])/5)
    return {'detector': NAME, 'kind': kind, 'team': team, 'category': CATEGORY[kind],
            't0': recs[0]['t'], 't1': recs[-1]['t'], 'tKey': k['t'], 'sKey': s, 'before': before, 'after': after,
            'players': {'focus': focus, 'support': support, 'opponents': opponents},
            'roles': cm.roles_of(engine, team, s, focus+support) | cm.roles_of(engine, ctx.other, s, opponents),
            'measures': {'pressDistanceM': round(k['pressDist'], 1), 'coverDistanceM': None if k['coverDist'] is None else round(k['coverDist'], 1),
                         'freeOptions': len(k['free']), 'carrierDepthM': round(k['carrierDepth'], 1),
                         'durationS': round(recs[-1]['t']-recs[0]['t'], 1), 'viewShare': round(k['view'], 2)},
            'zones': {'carrier': k['carrierXY']}, 'phase': phase,
            'outcome': cm.follow_up(engine, team, s, opponent=opponents[1] if len(opponents) > 1 else None),
            'reliability': level, 'interpretation': interp, 'reasons': reasons, 'score': round(float(score), 3)}


def detect(engine):
    out = []
    data = engine.data
    for team in 'AB':
        ctx = engine.ctx[team]
        items = []
        for s in engine.live(team):
            if data.possCarrier[s] < 0 or ctx.phase[s] != cx.OPP:
                continue
            for key, rec in _sample(engine, team, int(s)):
                items.append((key, rec['s'], rec['t'], rec))
        for key, recs in en.runs(items, max_gap_s=.6, min_len_s=0.0):
            ep = _episode(engine, team, key, recs)
            if ep:
                out.append(ep)
    return out
