"""Detector 6b: an opponent free between the team's defence and midfield.

During the opponents' settled possession, with the team's defenders and at least two central midfielders in view: the
defence line = median depth of the defenders, the midfield line = median depth of the central midfielders (DM / CM). When
the lines are LINES_GAP_M or more apart, the ball (tracked, or the named carrier) is in front of the midfield line (by
BALL_AHEAD_M: the space between the lines is a target only while the ball is still in front of them; checked on CAN-ISL
video, where an 'opponent between the lines' with the ball already in the penalty area meant nothing), and an opponent
stands between the lines (more than INSIDE_M from either), in the central zone (|lateral| <= CENTRAL_M), with no player
of the team within FREE_M, the opponents have a receiver between the lines:

    free_between_lines   IMPROVEMENT  for FREE_S or more; the players named are the nearest midfielder and the nearest
                                      defender (either could have closed the space; the report does not pick one)
"""
import numpy as np

from .. import context as cx
from .. import engine as en
from .. import geometry as geo
from . import common as cm

NAME = 'between_lines'
LINES_GAP_M, INSIDE_M, CENTRAL_M, FREE_M, FREE_S, BALL_AHEAD_M, BALL_MAX_M = 12.0, 3.0, 15.0, 5.0, 1.0, 3.0, 30.0
VIEW_RADIUS_M, VIEW_SHARE = 15.0, .8


def _sample(engine, team, s):
    data, ctx = engine.data, engine.ctx[team]
    idx, pids = ctx.players(s)
    if len(idx) < 7:
        return []
    recs = [engine.role(team, p, s) for p in pids]
    d = [k for k, r in enumerate(recs) if r and r['line'] == 'D' and r['confidence'] != 'low']
    m = [k for k, r in enumerate(recs) if r and r['role'] in ('DM', 'CM') and r['confidence'] != 'low']
    if len(d) < 3 or len(m) < 2:
        return []
    depth = ctx.depth(data.x[idx], s)
    lat = ctx.lateral(data.y[idx], s)
    dl, ml = float(np.median(depth[d])), float(np.median(depth[m]))
    if ml-dl < LINES_GAP_M:
        return []
    ball = cm.ball_at(data, s)
    if ball is None or float(ctx.depth(ball[0], s)) < ml+BALL_AHEAD_M:
        return []
    oidx, opids = ctx.opponents(s)
    out = []
    for i, q in zip(oidx, opids):
        od, ol = float(ctx.depth(data.x[i], s)), float(ctx.lateral(data.y[i], s))
        if not (dl+INSIDE_M < od < ml-INSIDE_M) or abs(ol) > CENTRAL_M:
            continue
        dist = np.hypot(depth-od, lat-ol)
        if dist.min() <= FREE_M:
            continue
        if np.hypot(ball[0]-data.x[i], ball[1]-data.y[i]) > BALL_MAX_M:
            continue                     # too far from the ball for a pass now (03:52 on CAN-ISL: 45 m away)
        pts = geo.disc_points(data.x[i], data.y[i], VIEW_RADIUS_M)
        view = geo.visible_share(data.view[s], pts[0], pts[1], data.length, data.width)
        if view < VIEW_SHARE:
            continue
        near_m = m[int(np.argmin(dist[m]))]
        near_d = d[int(np.argmin(dist[d]))]
        out.append(((team, int(q)), {'s': int(s), 't': float(data.t[s]), 'opponent': int(q), 'mid': int(pids[near_m]),
                                     'def': int(pids[near_d]), 'gap': ml-dl, 'free': float(dist.min()),
                                     'xy': (float(data.x[i]), float(data.y[i])), 'line': dl, 'midLine': ml, 'view': view,
                                     'ball': ball[:2], 'ballSource': ball[2]}))
    return out


def _episode(engine, team, key, recs):
    data, ctx = engine.data, engine.ctx[team]
    if recs[-1]['t']-recs[0]['t']+.2 < FREE_S:
        return None
    k = max(recs, key=lambda r: r['free'])
    s = k['s']
    focus, support, opponents = [k['mid']], [k['def']], [k['opponent']]
    phase = cm.phase_info(ctx, s)
    level, reasons = cm.reliability(data, s, focus+opponents, k['view'])
    interp = 'clear' if phase['confidence'] in ('medium', 'high') else 'plausible'
    if interp != 'clear':
        reasons = reasons+['possession_low_confidence']
    before, after = cm.window(k['t'], start=recs[0]['t'], before=6.0, after=4.0)
    danger = 1.25 if k['line'] < 35 else 1.0
    score = .95*danger*cm.score_reliability(level)*cm.score_interpretation(interp)*min(1.3, .8+(recs[-1]['t']-recs[0]['t'])/4)
    return {'detector': NAME, 'kind': 'free_between_lines', 'team': team, 'category': 'improvement',
            't0': recs[0]['t'], 't1': recs[-1]['t'], 'tKey': k['t'], 'sKey': s, 'before': before, 'after': after,
            'players': {'focus': focus, 'support': support, 'opponents': opponents},
            'roles': cm.roles_of(engine, team, s, focus+support) | cm.roles_of(engine, ctx.other, s, opponents),
            'measures': {'linesGapM': round(k['gap'], 1), 'freeM': round(k['free'], 1), 'lineDepthM': round(k['line'], 1),
                         'durationS': round(recs[-1]['t']-recs[0]['t'], 1), 'viewShare': round(k['view'], 2)},
            'zones': {'opponent': k['xy'], 'ball': k['ball']}, 'phase': phase,
            'outcome': cm.follow_up(engine, team, s, opponent=k['opponent']),
            'reliability': level, 'interpretation': interp, 'reasons': reasons, 'score': round(float(score), 3)}


def detect(engine):
    out = []
    for team in 'AB':
        ctx = engine.ctx[team]
        items = []
        for s in engine.live(team):
            if ctx.phase[s] != cx.OPP or not ctx.settled[s]:
                continue
            for key, rec in _sample(engine, team, int(s)):
                items.append((key, rec['s'], rec['t'], rec))
        for key, recs in en.runs(items, max_gap_s=.6, min_len_s=0.0):
            ep = _episode(engine, team, key, recs)
            if ep:
                out.append(ep)
    return out
