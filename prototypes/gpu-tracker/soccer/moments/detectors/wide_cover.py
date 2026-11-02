"""Detector 1: a wide defender high up the pitch, and the space behind them.

A full back or wing-back (role LB / RB / LWB / RWB, inferred from the team's defensive organisation, at least medium
confidence) is ADVANCE_M or more higher than the median depth of the team's other defenders, on their own side. The
space they left is the wide channel behind them, around the other defenders' depth; its COVER POINT is KEY_LAT metres to
that side at the defenders' depth + KEY_DEPTH.

Who would get to the cover point first (geometry.time_to_reach, the same model for both teams):
    covered      a teammate (anyone but the advanced player) no later than the nearest opponent + COVER_MARGIN_S
    exposed      the nearest opponent at least EXPOSED_MARGIN_S earlier than any teammate, and within THREAT_M of it
    no threat    no opponent within THREAT_M of the cover point (nothing to cover against right now)
    unseen       the camera did not show VIEW_SHARE of the disc of VIEW_RADIUS_M around the cover point: a teammate or an
                 opponent could be there unseen, so nothing is claimed
    no ball / ball far   the ball (tracked, or the named carrier) unknown, or not near enough for the space to matter
                 (BALL_OPP_M from the cover point while the opponents have it or nobody is named; BALL_OWN_M from the
                 advanced player while the team has it): not judged
An advanced full back is never a fault by itself: covered spells with an opponent near and the coverer within
COVER_NEAR_M of the cover point are a POSITIVE moment (stronger when the coverer moved there: 'rotated_to_cover'),
exposed spells an IMPROVEMENT, the rest neutral or insufficient.
The phase only changes the reading: with the opponents on the ball the risk is immediate; with the team on the ball it is
the space a counter-attack would use if the ball were lost (rest defence); unknown, the reading is uncertain.
"""
import numpy as np

from .. import context as cx
from .. import engine as en
from .. import geometry as geo
from .. import roles as rl
from . import common as cm

NAME = 'wide_cover'
ADVANCE_M = 15.0
WIDE_M = 8.0
CHANNEL_DEPTH = (-6.0, 14.0)
CHANNEL_LAT = (12.0, 34.0)
KEY_DEPTH, KEY_LAT = 2.0, 22.0
THREAT_M = 22.0
COVER_MARGIN_S = .3
EXPOSED_MARGIN_S = 1.0
VIEW_RADIUS_M, VIEW_SHARE = 15.0, .8
MIN_EXPOSED_S, MIN_COVERED_S = 1.0, 1.5
ROTATE_M, ROTATE_S = 4.0, 4.0
OUTCOME_S = 5.0
ON_BALL_M = 2.5           # a teammate this close to the ball (or the named carrier) is not counted as cover
BALL_IN_SPACE_M = 12.0    # own ball carried in the space behind the full back: the space is not empty, not judged
#                           (checked on video: 93:08 and 27:38 on CAN-ISL read 'nobody covered' with the centre-back on the
#                           ball in that very channel)
COVER_NEAR_M = 8.0        # a positive 'covered' moment needs the coverer IN the space (checked on video: at 43:37 on
#                           CAN-ISL a midfielder 11 m away was 'covering' only by the time-to-reach model)
BALL_OPP_M = 40.0         # opponents on the ball: the ball within this of the cover point
BALL_OWN_M = 30.0         # own ball (rest defence): the ball within this of the advanced player (lost there, the space matters)


def _sample(engine, team, s):
    """[(key, record)] for every advanced wide defender of `team` at sample s."""
    data, ctx = engine.data, engine.ctx[team]
    idx, pids = ctx.players(s)
    if len(idx) < 5:
        return []
    depth = ctx.depth(data.x[idx], s)
    lat = ctx.lateral(data.y[idx], s)
    recs = [engine.role(team, p, s) for p in pids]
    centre = [k for k, r in enumerate(recs) if r and r['line'] == 'D' and r['role'] not in rl.FULL_BACKS]
    if not centre:
        return []
    line_depth = float(np.median(depth[centre]))
    out = []
    oidx, opids = ctx.opponents(s)
    for k, r in enumerate(recs):
        if not r or r['role'] not in rl.FULL_BACKS or r['confidence'] == 'low':
            continue
        side = 1.0 if r['role'].startswith('L') else -1.0
        adv = float(depth[k]-line_depth)
        if adv < ADVANCE_M or side*lat[k] < WIDE_M:
            continue
        kd, kl = line_depth+KEY_DEPTH, side*KEY_LAT
        kx, ky = ctx.to_pitch(kd, kl, s)
        ball = cm.ball_at(data, s)
        carrier = int(data.possCarrier[s])
        # The player on the ball covers nothing if it is lost (checked on video: at 87:15 on CAN-ISL the centre-back
        # 'covering' the channel was the one with the ball).
        # Any estimate of the ball will do here (excluding a coverer is the cautious side): at 87:15 the ball was only
        # predicted, not tracked.
        any_ball = data.ball[s] if np.all(np.isfinite(data.ball[s])) else (ball[:2] if ball else None)
        on_ball = {j for j in range(len(idx)) if int(pids[j]) == carrier or
                   (any_ball is not None and np.hypot(data.x[idx[j]]-any_ball[0], data.y[idx[j]]-any_ball[1]) <= ON_BALL_M)}
        mates = [j for j in range(len(idx)) if j != k and j not in on_ball]
        if not mates:
            continue
        tt = geo.time_to_reach(data.x[idx[mates]], data.y[idx[mates]], data.vx[idx[mates]], data.vy[idx[mates]], kx, ky)
        j = int(np.argmin(tt)) if len(tt) else -1
        rec = {'s': int(s), 't': float(data.t[s]), 'fb': int(pids[k]), 'role': r['role'], 'roleConf': r['confidence'],
               'advance': adv, 'lineDepth': line_depth, 'fbDepth': float(depth[k]), 'fbLat': float(lat[k]), 'side': side,
               'key': (float(kx), float(ky)), 'centre': [int(pids[c]) for c in centre],
               'coverer': int(pids[mates[j]]) if j >= 0 else -1, 'tTeam': float(tt[j]) if j >= 0 else np.inf,
               'dCover': float(np.hypot(data.x[idx[mates[j]]]-kx, data.y[idx[mates[j]]]-ky)) if j >= 0 else np.inf}
        if len(oidx):
            to = geo.time_to_reach(data.x[oidx], data.y[oidx], data.vx[oidx], data.vy[oidx], kx, ky)
            q = int(np.argmin(to))
            rec.update(threat=int(opids[q]), tOpp=float(to[q]), dOpp=float(np.hypot(data.x[oidx[q]]-kx, data.y[oidx[q]]-ky)))
        else:
            rec.update(threat=-1, tOpp=np.inf, dOpp=np.inf)
        px, py = geo.disc_points(kx, ky, VIEW_RADIUS_M)
        rec['view'] = geo.visible_share(data.view[s], px, py, data.length, data.width)
        rec['ball'] = ball[:2] if ball else None
        rec['ballSource'] = ball[2] if ball else None
        own = ctx.phase[s] == cx.OWN
        if ball:
            near = (np.hypot(ball[0]-data.x[idx[k]], ball[1]-data.y[idx[k]]) <= BALL_OWN_M if own
                    else np.hypot(ball[0]-kx, ball[1]-ky) <= BALL_OPP_M)
        in_space = own and any(np.hypot(data.x[idx[j]]-kx, data.y[idx[j]]-ky) <= BALL_IN_SPACE_M for j in on_ball)
        if rec['view'] < VIEW_SHARE:
            status = 'unseen'
        elif not ball:
            status = 'no_ball'
        elif not near:
            status = 'ball_far'
        elif in_space:
            status = 'ball_in_space'
        elif rec['dOpp'] > THREAT_M:
            status = 'no_threat'
        elif rec['tTeam'] <= rec['tOpp']+COVER_MARGIN_S:
            status = 'covered'
        elif rec['tTeam'] >= rec['tOpp']+EXPOSED_MARGIN_S:
            status = 'exposed'
        else:
            status = 'unclear'
        rec['status'] = status
        out.append(((team, int(pids[k])), rec))
    return out


def _outcome(engine, team, ep_rec, s_key):
    """What followed within OUTCOME_S: did the nearest opponent get near the cover point, did the full back come back,
    did the other team get the ball (as the possession record estimates it)."""
    data, ctx = engine.data, engine.ctx[team]
    s_end = data.sample_at(float(data.t[s_key])+OUTCOME_S)
    kx, ky = ep_rec['key']
    reached = None
    if ep_rec.get('threat', -1) >= 0:
        dmin = None
        for s in range(s_key, s_end+1):
            p = cm.where(data, s, ep_rec['threat'])
            if p:
                d = float(np.hypot(p[0]-kx, p[1]-ky))
                dmin = d if dmin is None else min(dmin, d)
        reached = None if dmin is None else bool(dmin <= 6.0)
    back = None
    p0, p1 = cm.where(data, s_key, ep_rec['fb']), cm.where(data, s_end, ep_rec['fb'])
    if p0 and p1:
        back = float(ctx.depth(p0[0], s_key)-ctx.depth(p1[0], s_end))
    lost = bool(np.any(ctx.phase[s_key:s_end+1] == cx.OPP)) if ctx.phase[s_key] == cx.OWN else None
    return {'opponentReachedSpace': reached, 'fullBackRecoveredM': None if back is None else round(back, 1),
            'ballLostAfter': lost}


def _episode(engine, team, key, recs):
    data, ctx = engine.data, engine.ctx[team]
    dt = 1.0/5
    exposed = [r for r in recs if r['status'] == 'exposed']
    covered = [r for r in recs if r['status'] == 'covered' and r['dOpp'] <= THREAT_M and r['dCover'] <= COVER_NEAR_M]
    unseen = sum(r['status'] == 'unseen' for r in recs)
    if len(exposed)*dt >= MIN_EXPOSED_S:
        category = 'improvement'
        k = max(exposed, key=lambda r: r['tTeam']-r['tOpp'])
    elif len(covered)*dt >= MIN_COVERED_S:
        category = 'positive'
        k = min(covered, key=lambda r: r['dOpp'])
    elif unseen > len(recs)/2:
        category = 'insufficient'
        k = recs[len(recs)//2]
    else:
        return None
    s = k['s']
    kind = 'exposed_channel' if category == 'improvement' else 'covered_advance' if category == 'positive' else 'unseen_channel'
    coverer_moved = None
    if category == 'positive' and k['coverer'] >= 0:
        s0 = data.sample_at(k['t']-ROTATE_S)
        p0 = cm.where(data, s0, k['coverer'])
        if p0:
            coverer_moved = float(np.hypot(p0[0]-k['key'][0], p0[1]-k['key'][1])-k['dCover'])
            if coverer_moved >= ROTATE_M:
                kind = 'rotated_to_cover'
    if kind == 'covered_advance':
        # A defender who was already there is the structure doing its job (a back three's wide centre-back stands in
        # that channel by default: 87:15 on CAN-ISL), not a decision worth a moment; a midfielder in the space is one.
        rec_cov = engine.role(team, k['coverer'], s) if k['coverer'] >= 0 else None
        if not rec_cov or rec_cov['line'] != 'M':
            return None
    focus = [k['fb']]
    support = [k['coverer']] if k['coverer'] >= 0 else []
    if category == 'improvement':
        support = [c for c in k['centre'][:2]]
    opponents = [k['threat']] if k['threat'] >= 0 else []
    phase = cm.phase_info(ctx, s)
    level, reasons = cm.reliability(data, s, focus+support[:1]+opponents, k['view'])
    if phase['state'] == 'unknown' or not phase['settled']:
        interp = 'uncertain'
        reasons = reasons+['phase_unknown' if phase['state'] == 'unknown' else 'phase_unsettled']
    elif k['roleConf'] != 'high':
        interp = 'plausible'
        reasons = reasons+['role_medium']
    else:
        interp = 'clear' if (category != 'improvement' or k['tTeam']-k['tOpp'] >= 1.5) else 'plausible'
    if category == 'improvement' and phase['state'] == 'own' and interp == 'clear':
        interp = 'plausible'     # rest defence: the risk exists only if the ball is lost
    before, after = cm.window(k['t'], start=recs[0]['t'])
    zone = geo.rect_in_team(ctx, s, k['lineDepth']+CHANNEL_DEPTH[0], k['lineDepth']+CHANNEL_DEPTH[1],
                            k['side']*CHANNEL_LAT[0], k['side']*CHANNEL_LAT[1])
    margin = (k['tTeam']-k['tOpp']) if np.isfinite(k['tOpp']) and np.isfinite(k['tTeam']) else None
    danger = 1.25 if k['lineDepth'] < 35 else 1.0 if k['lineDepth'] < 52.5 else .8
    phase_w = {'opp': 1.0, 'own': .75, 'unknown': .5}[phase['state']]
    if category == 'improvement':
        base = 1.0*min(1.5, max(.6, (margin or 1.0)/1.0))*min(1.3, .7+len(exposed)*dt/3)
    elif category == 'positive':
        base = .9*(1.2 if kind == 'rotated_to_cover' else 1.0)*(1.15 if k['dOpp'] <= 10 else 1.0)
    else:
        base = .3
    score = base*danger*phase_w*cm.score_reliability(level)*cm.score_interpretation(interp)
    return {'detector': NAME, 'kind': kind, 'team': team, 'category': category,
            't0': recs[0]['t'], 't1': recs[-1]['t'], 'tKey': k['t'], 'sKey': s, 'before': before, 'after': after,
            'players': {'focus': focus, 'support': support, 'opponents': opponents},
            'roles': cm.roles_of(engine, team, s, focus+support) | cm.roles_of(engine, ctx.other, s, opponents),
            'measures': {'advanceM': round(k['advance'], 1), 'lineDepthM': round(k['lineDepth'], 1),
                         'fullBackDepthM': round(k['fbDepth'], 1), 'opponentToSpaceM': round(k['dOpp'], 1),
                         'teammateToSpaceM': round(k['dCover'], 1), 'timeMarginS': None if margin is None else round(margin, 2),
                         'exposedS': round(len(exposed)*dt, 1), 'coveredS': round(len(covered)*dt, 1),
                         'durationS': round(recs[-1]['t']-recs[0]['t'], 1), 'viewShare': round(k['view'], 2),
                         'covererClosedM': None if coverer_moved is None else round(coverer_moved, 1),
                         'side': 'left' if k['side'] > 0 else 'right'},
            'zones': {'channel': zone, 'coverPoint': k['key']} | ({'ball': k['ball']} if k.get('ball') else {}),
            'phase': phase, 'outcome': _outcome(engine, team, k, s) | cm.follow_up(engine, team, s),
            'reliability': level, 'interpretation': interp, 'reasons': reasons, 'score': round(float(score), 3)}


def detect(engine):
    out = []
    for team in 'AB':
        items = []
        for s in engine.live(team):
            for key, rec in _sample(engine, team, s):
                items.append((key, int(s), rec['t'], rec))
        for key, recs in en.runs(items, min_len_s=1.0):
            ep = _episode(engine, team, key, recs)
            if ep:
                out.append(ep)
    return out
