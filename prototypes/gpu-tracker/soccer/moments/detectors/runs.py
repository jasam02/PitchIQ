"""Detector 4: useful movement off the ball while the team has it.

During the team's settled possession, a RUN is a player's spell at RUN_SPEED or faster with a forward component of at
least FORWARD_SPEED, lasting RUN_S or more, by a player who is not the named ball carrier at the time (a dribble is not
an off-ball run). A run counts when it changed something observable at its end:

    run_in_behind         the runner ended beyond the opponents' last outfield player (deeper than all of them towards
                          their goal), in the opponents' half
    created_separation    the nearest opponent was within TIGHT_M at the start and FREE_M or more away at the end
A run is described as an option it CREATED, never as a pass that should have been played: whether the ball could reach
the runner is stated only when the ball is placed on the pitch at the end and no opponent stands within LANE_M of the
line from it to the runner ('lane open').
"""
import numpy as np

from .. import context as cx
from .. import engine as en
from .. import geometry as geo
from . import common as cm

NAME = 'runs'
RUN_SPEED, FORWARD_SPEED, RUN_S = 5.0, 3.0, 1.2
TIGHT_M, FREE_M = 3.5, 6.0
LANE_M = 2.5
VIEW_RADIUS_M, VIEW_SHARE = 15.0, .8


def _items(engine, team):
    data, ctx = engine.data, engine.ctx[team]
    items = []
    for s in engine.live(team):
        if ctx.phase[s] != cx.OWN or not ctx.settled[s]:
            continue
        idx, pids = ctx.players(int(s))
        if not len(idx):
            continue
        vx, vy = data.vx[idx], data.vy[idx]
        speed = np.hypot(vx, vy)
        fwd = ctx.dir[s]*vx
        for k in np.flatnonzero((speed >= RUN_SPEED) & (fwd >= FORWARD_SPEED)):
            p = int(pids[k])
            if p == int(data.possCarrier[s]):
                continue
            items.append(((team, p), int(s), float(data.t[s]), {'s': int(s), 't': float(data.t[s]), 'pid': p}))
    return items


def _episode(engine, team, key, recs):
    data, ctx = engine.data, engine.ctx[team]
    p = key[1]
    s_a, s_b = recs[0]['s'], recs[-1]['s']
    a, b = cm.where(data, s_a, p), cm.where(data, s_b, p)
    if not a or not b:
        return None
    end_depth = float(ctx.depth(b[0], s_b))
    oidx, opids = ctx.opponents(s_b)
    if len(oidx) < 4:
        return None
    odepth = ctx.depth(data.x[oidx], s_b)
    last = float(odepth.max())
    d_start = np.hypot(data.x[ctx.opponents(s_a)[0]]-a[0], data.y[ctx.opponents(s_a)[0]]-a[1])
    d_end = np.hypot(data.x[oidx]-b[0], data.y[oidx]-b[1])
    in_behind = end_depth > last+.5 and end_depth > data.length/2
    separated = len(d_start) and d_start.min() <= TIGHT_M and d_end.min() >= FREE_M
    if not (in_behind or separated):
        return None
    kind = 'run_in_behind' if in_behind else 'created_separation'
    pts = geo.disc_points(b[0], b[1], VIEW_RADIUS_M)
    view = geo.visible_share(data.view[s_b], pts[0], pts[1], data.length, data.width)
    if view < VIEW_SHARE:
        return None
    lane = None
    ball = data.ball[s_b]
    if np.all(np.isfinite(ball)) and data.ballState[s_b] == 1:
        lane = not geo.lane_blocked(data.x[oidx], data.y[oidx], float(ball[0]), float(ball[1]), b[0], b[1], width=LANE_M)
    nearest = int(opids[int(np.argmin(d_end))])
    marker = int(opids[int(np.argmin(np.hypot(data.x[ctx.opponents(s_a)[0]]-a[0], data.y[ctx.opponents(s_a)[0]]-a[1])))]) if len(d_start) else nearest
    phase = cm.phase_info(ctx, s_b)
    level, reasons = cm.reliability(data, s_b, [p, marker], view, span=(int((s_b-s_a)), 3))
    interp = 'clear' if lane else 'plausible'
    if lane is None:
        reasons = reasons+['ball_not_placed']
    t_key = float(data.t[s_b])
    before, after = cm.window(t_key, start=recs[0]['t'], before=6.0, after=4.0)
    danger = 1.3 if end_depth > 88.5 else 1.1 if end_depth > 70 else .9
    base = 1.0 if in_behind else .85
    score = base*danger*cm.score_reliability(level)*cm.score_interpretation(interp)*(1.1 if lane else 1.0)
    return {'detector': NAME, 'kind': kind, 'team': team, 'category': 'positive',
            't0': recs[0]['t'], 't1': t_key, 'tKey': t_key, 'sKey': int(s_b), 'before': before, 'after': after,
            'players': {'focus': [p], 'support': [], 'opponents': [marker]},
            'roles': cm.roles_of(engine, team, s_b, [p]) | cm.roles_of(engine, ctx.other, s_b, [marker]),
            'measures': {'runM': round(float(np.hypot(b[0]-a[0], b[1]-a[1])), 1), 'runS': round(t_key-recs[0]['t'], 1),
                         'endDepthM': round(end_depth, 1), 'beyondLastM': round(end_depth-last, 1),
                         'separationStartM': round(float(d_start.min()), 1) if len(d_start) else None,
                         'separationEndM': round(float(d_end.min()), 1), 'laneOpen': lane, 'viewShare': round(view, 2)},
            'zones': {'runStart': (a[0], a[1]), 'runEnd': (b[0], b[1])}, 'phase': phase,
            'outcome': cm.follow_up(engine, team, int(s_b), receiver=p),
            'reliability': level, 'interpretation': interp, 'reasons': reasons, 'score': round(float(score), 3)}


def detect(engine):
    out = []
    for team in 'AB':
        for key, recs in en.runs(_items(engine, team), max_gap_s=.4, min_len_s=RUN_S):
            ep = _episode(engine, team, key, recs)
            if ep:
                out.append(ep)
    return out
