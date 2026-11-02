"""Detectors 6 and 7a: the back line, defender by defender, and as a unit.

Per live sample with the opponents on the ball or nobody named (never the team's own possession: the space behind a
team that has the ball is wide_cover's rest-defence reading; never a dead ball), the team's defenders in view (role line
'D', not low confidence) sorted from left to right. All the defenders the role window counted must be in view: a
defender off-screen could be standing in the very gap being judged. The defender-by-defender readings also need the
BALL (tracked, or the named carrier): a gap 35 m from the ball is not the situation it is next to it. Checked on CAN-ISL
video: without the ball a striker waiting between two centre-backs while the ball was at the other end came out as a
'gap', and a defender marking the deepest attacker as 'dropped off'.

    stepped_out_open      IMPROVEMENT  one defender STEP_M or more higher than the others' median depth, the ball
                                       within STEP_BALL_M of them, and the point they left (their lateral position at
                                       the others' depth) has no teammate within FILL_M while an opponent is within
                                       THREAT_M of it
    stepped_out_covered   POSITIVE     the same step, engaging an opponent (within ENGAGE_M) near the ball, with a
                                       teammate within FILL_M of the point left or the neighbours closed to CLOSED_GAP_M
    gap_attacker          IMPROVEMENT  two neighbouring defenders farther apart than GAP_M (CENTRAL_GAP_M between two
                                       centre-backs), an opponent within ATTACK_M of the gap's middle, level with or
                                       behind the line, no defender within MARK_M of that opponent, the ball in front
                                       of the line and within GAP_BALL_M of the gap
    dropped_off           IMPROVEMENT  one defender DROP_M or more deeper than the others, an opponent between their
                                       depth and the line who is FREE (no defender within MARK_M) and running towards
                                       goal (RUNNER_SPEED), the ball in front of the line. A defender marking the
                                       opponent closest to them is never criticised for being deep. Without the moment
                                       of a pass the offside line cannot be judged: at most 'plausible'.
    stepped_up_together   POSITIVE     over STEP_UP_S the same 4+ defenders all moved STEP_EACH_M or more forward, the
                                       line's median depth rose STEP_UP_M or more, and the line was level (offsets'
                                       standard deviation <= LEVEL_M) at both ends
Real defences stagger: a single defender a few metres off the line is not judged; only the relationships above are.
"""
import numpy as np

from .. import context as cx
from .. import engine as en
from .. import geometry as geo
from . import common as cm

NAME = 'back_line'
STEP_M, FILL_M, THREAT_M, ENGAGE_M, CLOSED_GAP_M, STEP_BALL_M = 8.0, 6.0, 10.0, 4.0, 14.0, 12.0
GAP_M, CENTRAL_GAP_M, ATTACK_M, MARK_M, GAP_BALL_M = 22.0, 18.0, 6.0, 5.0, 35.0
DROP_M, DROP_LAT_M = 6.0, 15.0
RUNNER_SPEED = 2.0
STEP_UP_S, STEP_UP_M, STEP_EACH_M, LEVEL_M = 3.0, 6.0, 3.0, 2.5
VIEW_RADIUS_M, VIEW_SHARE = 12.0, .8
MIN_S = {'stepped_out_open': 1.0, 'stepped_out_covered': 1.0, 'gap_attacker': 1.0, 'dropped_off': 1.0,
         'stepped_up_together': .4}
CATEGORY = {'stepped_out_open': 'improvement', 'stepped_out_covered': 'positive', 'gap_attacker': 'improvement',
            'dropped_off': 'improvement', 'stepped_up_together': 'positive'}


def _defenders(engine, team, s):
    """(rows, pids, depth, lateral, roles) of the team's defenders in view at s, left to right, or None when the role
    window counted more defenders than are in view."""
    data, ctx = engine.data, engine.ctx[team]
    idx, pids = ctx.players(s)
    if len(idx) < 4:
        return None
    recs = [engine.role(team, p, s) for p in pids]
    keep = [k for k, r in enumerate(recs) if r and r['line'] == 'D' and r['confidence'] != 'low']
    shape = engine.roles[team].shape_at(float(data.t[s]))
    if len(keep) < 3 or (shape and len(keep) < shape['defenders']):
        return None
    idx, pids = idx[keep], pids[keep]
    depth = ctx.depth(data.x[idx], s)
    lat = ctx.lateral(data.y[idx], s)
    order = np.argsort(-lat)
    return idx[order], pids[order], depth[order], lat[order], [recs[keep[k]]['role'] for k in order]


def _view(engine, s, x, y):
    data = engine.data
    px, py = geo.disc_points(x, y, VIEW_RADIUS_M)
    return geo.visible_share(data.view[s], px, py, data.length, data.width)


def _sample(engine, team, s):
    data, ctx = engine.data, engine.ctx[team]
    ball = cm.ball_at(data, s)
    if ball is None:
        return []
    got = _defenders(engine, team, s)
    if got is None:
        return []
    idx, pids, depth, lat, names = got
    oidx, opids = ctx.opponents(s)
    if not len(oidx):
        return []
    bd, bl = float(ctx.depth(ball[0], s)), float(ctx.lateral(ball[1], s))
    odepth = ctx.depth(data.x[oidx], s)
    olat = ctx.lateral(data.y[oidx], s)
    midx, mpids = ctx.players(s)
    mdepth, mlat = ctx.depth(data.x[midx], s), ctx.lateral(data.y[midx], s)
    dir_ = ctx.dir[s] or 1
    out = []
    n = len(idx)
    for i in range(n):
        others = np.delete(np.arange(n), i)
        line = float(np.median(depth[others]))
        off = float(depth[i]-line)
        if off >= STEP_M:
            if np.hypot(bd-depth[i], bl-lat[i]) > STEP_BALL_M:
                continue
            px, py = ctx.to_pitch(line, lat[i], s)
            dmates = np.hypot(mdepth-line, mlat-lat[i])
            dmates[mpids == pids[i]] = np.inf
            filled = bool(dmates.min() <= FILL_M)
            nb = [j for j in (i-1, i+1) if 0 <= j < n]
            closed = len(nb) == 2 and abs(lat[nb[0]]-lat[nb[1]]) <= CLOSED_GAP_M
            dopp_self = np.hypot(odepth-depth[i], olat-lat[i])
            dopp_point = np.hypot(odepth-line, olat-lat[i])
            q = int(np.argmin(dopp_point))
            engaged = bool(dopp_self.min() <= ENGAGE_M)
            view = _view(engine, s, px, py)
            if view < VIEW_SHARE:
                continue
            if not (filled or closed) and dopp_point[q] <= THREAT_M:
                kind = 'stepped_out_open'
            elif engaged and (filled or closed):
                kind = 'stepped_out_covered'
            else:
                continue
            j = int(np.argmin(dmates))
            out.append(((team, kind, int(pids[i])), {
                's': int(s), 't': float(data.t[s]), 'kind': kind, 'focus': int(pids[i]), 'role': names[i],
                'support': [int(mpids[j])] if filled else [int(pids[k]) for k in nb],
                'opponent': int(opids[q]) if kind == 'stepped_out_open' else int(opids[int(np.argmin(dopp_self))]),
                'offset': off, 'line': line, 'point': (float(px), float(py)), 'fillM': float(dmates.min()),
                'oppToPointM': float(dopp_point[q]), 'view': view, 'severity': THREAT_M-float(dopp_point[q]),
                'ball': ball[:2], 'ballSource': ball[2]}))
        elif -off >= DROP_M:
            if bd <= line:
                continue                                         # the ball is already behind the line
            between = (odepth >= depth[i]) & (odepth <= line) & (np.abs(olat-lat[i]) <= DROP_LAT_M)
            free = []
            for q in np.flatnonzero(between):
                if np.hypot(depth-odepth[q], lat-olat[q]).min() <= MARK_M:
                    continue                                     # marked (by this defender or another)
                if -dir_*np.nan_to_num(data.vx[oidx[q]]) < RUNNER_SPEED:
                    continue                                     # not running towards goal
                free.append(int(q))
            if not free:
                continue
            px, py = ctx.to_pitch(depth[i], lat[i], s)
            view = _view(engine, s, px, py)
            if view < VIEW_SHARE:
                continue
            b = max(free, key=lambda q: odepth[q])
            out.append(((team, 'dropped_off', int(pids[i])), {
                's': int(s), 't': float(data.t[s]), 'kind': 'dropped_off', 'focus': int(pids[i]), 'role': names[i],
                'support': [int(pids[k]) for k in (i-1, i+1) if 0 <= k < n], 'opponent': int(opids[b]),
                'offset': off, 'line': line, 'point': (float(px), float(py)), 'view': view, 'severity': -off,
                'ball': ball[:2], 'ballSource': ball[2]}))
    line_all = float(np.median(depth))
    if bd < line_all:
        return out                                               # gaps matter while the ball is in front of the line
    for i in range(n-1):
        gap = float(lat[i]-lat[i+1])
        central = names[i] in ('LCB', 'CB', 'RCB') and names[i+1] in ('LCB', 'CB', 'RCB')
        if gap < (CENTRAL_GAP_M if central else GAP_M):
            continue
        gd, gl = float((depth[i]+depth[i+1])/2), float((lat[i]+lat[i+1])/2)
        if np.hypot(bd-gd, bl-gl) > GAP_BALL_M:
            continue
        d = np.hypot(odepth-gd, olat-gl)
        cand = np.flatnonzero((d <= ATTACK_M) & (np.abs(olat-gl) <= gap/2-2) & (odepth <= line_all+3))
        if not len(cand):
            continue
        q = int(cand[np.argmin(d[cand])])
        if np.hypot(depth-odepth[q], lat-olat[q]).min() <= MARK_M:
            continue
        px, py = ctx.to_pitch(gd, gl, s)
        view = _view(engine, s, px, py)
        if view < VIEW_SHARE:
            continue
        out.append(((team, 'gap_attacker', (int(pids[i]), int(pids[i+1]))), {
            's': int(s), 't': float(data.t[s]), 'kind': 'gap_attacker', 'focus': int(pids[i]), 'pair': [int(pids[i]), int(pids[i+1])],
            'roles2': [names[i], names[i+1]], 'role': names[i], 'support': [int(pids[i+1])], 'opponent': int(opids[q]),
            'gap': gap, 'line': line_all, 'point': (float(px), float(py)), 'oppToGapM': float(d[q]),
            'view': view, 'severity': gap, 'ball': ball[:2], 'ballSource': ball[2]}))
    return out


def _step_up(engine, team, samples):
    """'stepped_up_together' records: the same 4+ defenders all forward STEP_EACH_M+, median STEP_UP_M+, level at both ends."""
    data, ctx = engine.data, engine.ctx[team]
    back = int(round(STEP_UP_S*5))
    cache = {}

    def line(s):
        if s not in cache:
            got = _defenders(engine, team, s)
            cache[s] = None if got is None else {int(p): (float(d), float(l)) for p, d, l in zip(got[1], got[2], got[3])}
        return cache[s]
    out = []
    live = set(int(s) for s in samples)
    for s in samples:
        s = int(s)
        a = s-back
        if a not in live or data.period[a] != data.period[s]:
            continue
        now, before = line(s), line(a)
        if not now or not before:
            continue
        common = sorted(set(now) & set(before))
        if len(common) < 4:
            continue
        d1 = np.array([now[p][0] for p in common])
        d0 = np.array([before[p][0] for p in common])
        moved = d1-d0
        if moved.min() < STEP_EACH_M or np.median(moved) < STEP_UP_M:
            continue
        if np.std(d1-np.median(d1)) > LEVEL_M or np.std(d0-np.median(d0)) > LEVEL_M:
            continue
        mid = np.median(d1)
        lat = np.median([now[p][1] for p in common])
        px, py = ctx.to_pitch(mid, lat, s)
        view = _view(engine, s, px, py)
        if view < VIEW_SHARE:
            continue
        ball = cm.ball_at(data, s)
        out.append(((team, 'stepped_up_together', 0), {
            's': s, 't': float(data.t[s]), 'kind': 'stepped_up_together', 'focus': common[0], 'members': common,
            'role': None, 'support': common[1:], 'opponent': -1, 'rise': float(np.median(moved)), 'line': float(mid),
            'lineBefore': float(np.median(d0)), 'point': (float(px), float(py)), 'view': view,
            'severity': float(np.median(moved)), 'ball': ball[:2] if ball else None, 'ballSource': ball[2] if ball else None}))
    return out


def _episode(engine, team, key, recs):
    data, ctx = engine.data, engine.ctx[team]
    kind = key[1]
    if (recs[-1]['t']-recs[0]['t']) + .2 < MIN_S[kind]:
        return None
    k = max(recs, key=lambda r: r['severity'])
    s = k['s']
    category = CATEGORY[kind]
    focus = k.get('members') or [k['focus']]
    support = [p for p in k['support'] if p not in focus][:3]
    opponents = [k['opponent']] if k['opponent'] >= 0 else []
    phase = cm.phase_info(ctx, s)
    level, reasons = cm.reliability(data, s, focus[:2]+opponents, k['view'])
    if phase['state'] != 'opp' or not phase['settled']:
        interp = 'uncertain'
        reasons = reasons+['phase_unknown' if phase['state'] == 'unknown' else 'phase_unsettled']
    else:
        interp = 'plausible' if kind == 'dropped_off' else 'clear'
    if k.get('ballSource') == 'carrier' and interp == 'clear':
        interp = 'plausible'
        reasons = reasons+['ball_from_carrier']
    before, after = cm.window(k['t'], start=recs[0]['t'])
    zones = {'point': k['point']}
    if k.get('ball'):
        zones['ball'] = k['ball']
    measures = {'durationS': round(recs[-1]['t']-recs[0]['t'], 1), 'lineDepthM': round(k['line'], 1),
                'viewShare': round(k['view'], 2)}
    if k.get('ball'):
        measures['ballToPointM'] = round(float(np.hypot(k['ball'][0]-k['point'][0], k['ball'][1]-k['point'][1])), 1)
    if kind.startswith('stepped_out'):
        measures.update(stepM=round(k['offset'], 1), fillM=round(k['fillM'], 1), opponentToPointM=round(k['oppToPointM'], 1))
    elif kind == 'gap_attacker':
        measures.update(gapM=round(k['gap'], 1), opponentToGapM=round(k['oppToGapM'], 1))
        focus = k['pair']
        support = []
    elif kind == 'dropped_off':
        measures.update(dropM=round(-k['offset'], 1))
    elif kind == 'stepped_up_together':
        measures.update(riseM=round(k['rise'], 1), lineBeforeM=round(k['lineBefore'], 1), defenders=len(k['members']))
        focus, support = k['members'][:2], k['members'][2:5]
    danger = 1.25 if k['line'] < 35 else 1.0 if k['line'] < 52.5 else .8
    phase_w = {'opp': 1.0, 'own': .6, 'unknown': .55}[phase['state']]
    base = {'stepped_out_open': 1.0, 'gap_attacker': 1.0, 'dropped_off': .75, 'stepped_out_covered': .85,
            'stepped_up_together': .8}[kind]
    dur = min(1.3, .8+(recs[-1]['t']-recs[0]['t'])/4)
    score = base*danger*phase_w*dur*cm.score_reliability(level)*cm.score_interpretation(interp)
    return {'detector': NAME, 'kind': kind, 'team': team, 'category': category,
            't0': recs[0]['t'], 't1': recs[-1]['t'], 'tKey': k['t'], 'sKey': s, 'before': before, 'after': after,
            'players': {'focus': focus, 'support': support, 'opponents': opponents},
            'roles': cm.roles_of(engine, team, s, focus+support) | cm.roles_of(engine, ctx.other, s, opponents),
            'measures': measures, 'zones': zones, 'phase': phase,
            'outcome': cm.follow_up(engine, team, s, opponent=opponents[0] if opponents else None),
            'reliability': level, 'interpretation': interp, 'reasons': reasons, 'score': round(float(score), 3)}


def detect(engine):
    out = []
    for team in 'AB':
        ctx = engine.ctx[team]
        samples = [int(s) for s in engine.live(team) if ctx.phase[s] in (cx.OPP, cx.UNKNOWN)]
        items = []
        for s in samples:
            for key, rec in _sample(engine, team, s):
                items.append((key, s, rec['t'], rec))
        for key, rec in _step_up(engine, team, samples):
            items.append((key, rec['s'], rec['t'], rec))
        for key, recs in en.runs(items, max_gap_s=.6, min_len_s=0.0):
            ep = _episode(engine, team, key, recs)
            if ep:
                out.append(ep)
    return out
