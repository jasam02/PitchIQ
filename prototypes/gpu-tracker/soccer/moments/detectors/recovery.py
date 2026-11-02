"""Detector 3: the team's movement in the seconds after it lost the ball.

Only at changes of possession the record names (from the team to the opponents) after which the opponents' spell lasts
(OPP on at least SETTLED_SHARE of the SETTLE_S after the change) and where the ball's position at the loss is known (the
ball placed on the pitch, else the player who lost it or the one who won it). Over RECOVER_S after the loss:

    recovery_run          POSITIVE     a player who was ahead of the ball at the loss ran RUN_M or more back towards
                                       their own goal, at RUN_SPEED or faster for at least RUN_S, and ended goal-side of
                                       the ball in the central corridor (|lateral| <= CORRIDOR_M)
    slow_reorganisation   IMPROVEMENT  STAY_N or more players ahead of the ball moved back less than STAY_M while two or
                                       more opponents advanced ADVANCE_M or more towards the goal, and at the end the team
                                       had GOALSIDE_MAX or fewer outfield players goal-side of the ball
Movement only: nothing here is about effort or intent (a player may have had a reason the camera does not show), and the
wording says what was observed.
"""
import numpy as np

from .. import context as cx
from .. import geometry as geo
from . import common as cm

NAME = 'recovery'
SETTLE_S, SETTLED_SHARE = 3.0, .7
RECOVER_S = 6.0
AHEAD_M = 2.0
RUN_M, RUN_SPEED, RUN_S = 12.0, 5.0, 1.5
CORRIDOR_M = 20.0
STAY_N, STAY_M = 3, 4.0
ADVANCE_M = 8.0
GOALSIDE_MAX = 5
VIEW_RADIUS_M, VIEW_SHARE = 25.0, .7


def _ball_at(engine, s, change, which):
    data = engine.data
    b = data.ball[s]
    if np.all(np.isfinite(b)) and data.ballState[s] == 1:
        return float(b[0]), float(b[1]), 'ball'
    pid = data.index_of.get(change.get('lostBy' if which == 'lost' else 'wonBy'))
    if pid is not None:
        p = cm.where(data, s, pid)
        if p:
            return p[0], p[1], 'player'
    return None


def _change(engine, team, change):
    data, ctx = engine.data, engine.ctx[team]
    t_loss = float(change.get('lostAt') or change.get('wonAt') or 0)
    t_won = float(change.get('wonAt') or t_loss)
    s0 = data.sample_at(t_loss)
    s1 = data.sample_at(t_loss+RECOVER_S)
    sw0, sw1 = data.sample_at(t_won), data.sample_at(t_won+SETTLE_S)
    if data.period[s0] != data.period[s1] or not data.calibrated[s0] or ctx.dir[s0] == 0:
        return []
    if np.mean(ctx.phase[sw0:sw1+1] == cx.OPP) < SETTLED_SHARE:
        return []
    ball0 = _ball_at(engine, s0, change, 'lost') or _ball_at(engine, data.sample_at(t_won), change, 'won')
    if not ball0:
        return []
    bd0 = float(ctx.depth(ball0[0], s0))
    ball1 = None
    b = data.ball[s1]
    if np.all(np.isfinite(b)) and data.ballState[s1] == 1:
        ball1 = (float(b[0]), float(b[1]))
    elif data.possCarrier[s1] >= 0:
        p = cm.where(data, s1, int(data.possCarrier[s1]))
        ball1 = (p[0], p[1]) if p else None
    pts = geo.disc_points(ball0[0], ball0[1], VIEW_RADIUS_M)
    view = geo.visible_share(data.view[s0], pts[0], pts[1], data.length, data.width)
    if view < VIEW_SHARE:
        return []
    idx0, pids0 = ctx.players(s0)
    ahead = [int(p) for i, p in zip(idx0, pids0) if float(ctx.depth(data.x[i], s0)) > bd0+AHEAD_M]
    if not ahead:
        return []
    moves = {}
    for p in ahead:
        tr = cm.track(data, p, s0, s1)
        if len(tr) < 0.6*(s1-s0+1):
            continue
        ds = np.array([float(ctx.depth(x, data.sample_at(t))) for t, x, y in tr])
        ts = np.array([t for t, x, y in tr])
        back = float(ds[0]-ds[-1])
        speed = -np.gradient(ds, ts) if len(ds) > 2 else np.zeros(len(ds))
        fast = speed >= RUN_SPEED
        longest, cur = 0.0, 0.0
        for k in range(1, len(ts)):
            cur = cur+(ts[k]-ts[k-1]) if fast[k] and fast[k-1] else 0.0
            longest = max(longest, cur)
        end = cm.where(data, s1, p)
        moves[p] = {'back': back, 'fastS': longest, 'peak': float(speed.max()) if len(speed) else 0.0,
                    'end': end, 'endDepth': float(ds[-1]), 'startDepth': float(ds[0])}
    if not moves:
        return []
    oidx0, opids0 = ctx.opponents(s0)
    advancing = []
    for i, p in zip(oidx0, opids0):
        e = cm.where(data, s1, int(p))
        if e:
            gain = float(ctx.depth(data.x[i], s0)-ctx.depth(e[0], s1))
            if gain >= ADVANCE_M:
                advancing.append((int(p), gain))
    advancing.sort(key=lambda a: -a[1])
    idx1, pids1 = ctx.players(s1)
    bd1 = float(ctx.depth(ball1[0], s1)) if ball1 else None
    goalside_end = None if bd1 is None else int(sum(float(ctx.depth(data.x[i], s1)) < bd1 for i in idx1))
    out = []
    runners = []
    for p, m in moves.items():
        if m['back'] < RUN_M or m['fastS'] < RUN_S or not m['end'] or bd1 is None or m['endDepth'] >= bd1:
            continue
        if abs(float(ctx.lateral(m['end'][1], s1))) > CORRIDOR_M:
            continue
        runners.append((p, m))
    common = {'tLoss': t_loss, 's0': s0, 's1': s1, 'ballDepth': bd0, 'ballEnd': ball1, 'goalsideEnd': goalside_end,
              'advancing': advancing, 'view': view, 'change': change, 'ballSource': ball0[2], 'aheadN': len(ahead)}
    if runners:
        p, m = max(runners, key=lambda r: r[1]['back'])
        out.append(('recovery_run', p, m, common))
    stayers = [p for p, m in moves.items() if m['back'] < STAY_M]
    if len(stayers) >= STAY_N and len(advancing) >= 2 and goalside_end is not None and goalside_end <= GOALSIDE_MAX:
        out.append(('slow_reorganisation', stayers, None, common))
    return out


def _episode(engine, team, kind, who, move, c):
    data, ctx = engine.data, engine.ctx[team]
    # The key frame of a recovery run is where it ENDS (goal-side, in the middle): at 3 s after the loss the run had not
    # shown its result yet (71:41 on CAN-ISL). A slow reorganisation is keyed at the loss itself.
    s_key = c['s0'] if kind == 'slow_reorganisation' else c['s1']
    if kind == 'recovery_run':
        focus, support = [who], []
        nearest = None
        if move['end'] and c['advancing']:
            nearest = min(c['advancing'], key=lambda a: (lambda e: 1e9 if not e else np.hypot(e[0]-move['end'][0], e[1]-move['end'][1]))(cm.where(data, c['s1'], a[0])))
        opponents = [nearest[0]] if nearest else []
        category = 'positive'
    else:
        focus, support = list(who[:2]), list(who[2:4])
        opponents = [a[0] for a in c['advancing'][:2]]
        category = 'improvement'
    phase = cm.phase_info(ctx, s_key)
    level, reasons = cm.reliability(data, c['s0'], focus[:2], c['view'], span=(0, 10))
    conf = c['change'].get('confidence') or 'low'
    if conf == 'low' or c['ballSource'] != 'ball':
        interp = 'plausible' if category == 'positive' else 'uncertain'
        reasons = reasons+(['change_low_confidence'] if conf == 'low' else [])+(['ball_from_player'] if c['ballSource'] != 'ball' else [])
    else:
        interp = 'clear' if category == 'positive' else 'plausible'
    t_key = float(data.t[s_key])
    before, after = cm.window(t_key, start=c['tLoss']-3.0, before=6.0, after=5.0)
    measures = {'aheadOfBall': c['aheadN'], 'goalsideAtEnd': c['goalsideEnd'], 'opponentsAdvancing': len(c['advancing']),
                'secondsAfterLoss': round(t_key-c['tLoss'], 1), 'ballDepthM': round(c['ballDepth'], 1),
                'viewShare': round(c['view'], 2)}
    if move:
        measures.update(backM=round(move['back'], 1), sprintS=round(move['fastS'], 1), peakSpeed=round(move['peak'], 1))
    danger = 1.25 if c['ballDepth'] > 52.5 else 1.0      # lost in the opponents' half: more players ahead of the ball
    base = .95 if category == 'positive' else 1.0
    score = base*danger*cm.score_reliability(level)*cm.score_interpretation(interp)
    zones = {}
    if c['ballEnd']:
        zones['ball'] = c['ballEnd']
    return {'detector': NAME, 'kind': kind, 'team': team, 'category': category,
            't0': c['tLoss'], 't1': c['tLoss']+RECOVER_S, 'tKey': t_key, 'sKey': int(s_key), 'before': before, 'after': after,
            'players': {'focus': focus, 'support': support, 'opponents': opponents},
            'roles': cm.roles_of(engine, team, s_key, focus+support) | cm.roles_of(engine, ctx.other, s_key, opponents),
            'measures': measures, 'zones': zones, 'phase': phase,
            'outcome': {'changeId': c['change'].get('id'), 'cause': c['change'].get('cause')},
            'reliability': level, 'interpretation': interp, 'reasons': reasons, 'score': round(float(score), 3)}


def detect(engine):
    out = []
    for team in 'AB':
        for change in engine.data.changes:
            if change.get('fromTeam') != team or change.get('toTeam') not in ('A', 'B'):
                continue
            for kind, who, move, c in _change(engine, team, change):
                out.append(_episode(engine, team, kind, who, move, c))
    return out
