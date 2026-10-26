"""The DESIGN 13.1 property check of a suggested zone, written independently of soccer/coaching/alternatives.py's
vectorized code: per point, plain loops over the tactics_geom functions the detectors use, judged with the same
contextual.sample_outcomes rule and the level's parameters.

zone_problems(alt, finding, obs, roles, profile, side) -> [problem text] for one Alternative:
  reach      every polygon vertex lies within reach.radiusM of the mover's position in the moment's 'before' scene
             (so the whole polygon does: the disc is convex);
  stacking   no other team-mate (keepers included) inside the polygon or within STACK_M of it;
  restore    of the improvement grid's points inside the polygon (spacing max(0.5, L/105) m, anchored at the attack
             frame's origin; at least MIN_POINTS of them) at least ZONE_KEEP_SHARE restore the failed relation with
             the mover moved there;
  checks     every one of those points keeps pressure on the carrier, the central protection, the back line without a
             new ball-side gap, every marked opponent in our half marked;
  region     at least ZONE_KEEP_SHARE of those points lie in the detector's principle region (DESIGN 5.2: D1 cover the
             sector 4 m to R behind the first defender within the cover angle, goal-side of him; D1 pressure within
             the pressure distance of the carrier and 1 m goal-side of him; D2 the far half goal-side of the ball near
             the back line's median depth; D3 the vacated channel; D4 goal-side of the ball in the central two-thirds,
             also for a wide-defender advance judged as "someone stays back" (level 1's reframing and level 2's
             teamLevelOnly rule); D5 between the gap's defenders at the line's depth +-3 m);
  goal-side  (critic B-8) on a wide-defender advance, no vertex of the zone lies more than LEVEL_M (2 m) upfield of
             the deepest opponent in the vacated channel (an opponent outfield player in the defender's outer third,
             goal-side of him and in our half, at the moment); a zone upfield of him does not stand between him and
             our goal;
  behind the ball  a "someone stays back" zone (D4, level 1's reframed and level 2's advance) has no vertex ahead of
             the ball;
  the record itself: schema 'alternative' valid, improves before False -> after True, the five checks true.
fine_share(...) gives the restoring share on a finer grid (0.25 m) as a discretisation cross-check.
side: the side the team defends at the moment (the test knows it; alternatives.attack_side is tested separately).
"""
import math

import numpy as np

from soccer.coaching import contextual as CX, schema as S, tactics_geom as G

STACK_M = 3.0
KEEP = .8
MIN_POINTS = 4
TOL = .02
LINE_BAND_M = 8.0
MARK_M = 10.0
PRESSURE_M = 5.0
ON_BALL_M = 1.5
LEVEL_M = 2.0


def to_attack(x, y, side, L, W):
    return (x*L, y*W) if side == 'left' else ((1-x)*L, (1-y)*W)


def inside(poly, q):
    """Even-odd point in polygon, boundary (within 1e-6) counted inside."""
    x, y = q
    n = len(poly)
    c = False
    for i in range(n):
        x0, y0 = poly[i]
        x1, y1 = poly[(i+1) % n]
        if (y0 > y) != (y1 > y) and x < (x1-x0)*(y-y0)/(y1-y0)+x0:
            c = not c
    return c or edge_distance(poly, q) <= 1e-6


def edge_distance(poly, q):
    best = math.inf
    n = len(poly)
    for i in range(n):
        ax, ay = poly[i]
        bx, by = poly[(i+1) % n]
        dx, dy = bx-ax, by-ay
        ll = dx*dx+dy*dy
        t = 0.0 if ll == 0 else max(0.0, min(1.0, ((q[0]-ax)*dx+(q[1]-ay)*dy)/ll))
        best = min(best, math.hypot(q[0]-(ax+t*dx), q[1]-(ay+t*dy)))
    return best


class Case:
    """The moment of an Alternative in the attack frame, with everything the per-point re-measurement needs."""

    def __init__(self, alt, finding, obs, roles, profile, side):
        moments = {m['id']: m for m in (obs.get('moments') or [])+((roles or {}).get('moments') or [])}
        opps = {o['id']: o for o in (obs.get('opportunities') or [])+((roles or {}).get('opportunities') or [])}
        self.m = moments[alt['evidence']]
        self.o = opps[self.m['opportunity']]
        self.k = int(self.m['sample'])
        self.scale = obs['timeline']['scale']
        self.L, self.W = float(self.scale['lengthM']), float(self.scale['widthM'])
        self.side = side
        self.team = finding['team']
        self.det = finding['observation']
        self.params = dict(profile['detectors'][self.det]['params'])
        self.as_detector = self.det if self.o['detector'] != self.det else None
        # level 2's wide-defender rule (teamLevelOnly): judged and fixed as "someone stays back"
        self.team_level = (self.det == 'wide_defender_advance' and not self.as_detector and
                           bool(self.params.get('teamLevelOnly')))
        self.obs, self.roles, self.profile, self.alt = obs, roles, profile, alt
        own, gk, opp = {}, {}, {}
        for pid, e in self.m['scene']['players'].items():
            p = to_attack(e['x'], e['y'], side, self.L, self.W)
            if e['team'] == self.team:
                (gk if e['goalkeeper'] else own)[pid] = p
            elif not e['goalkeeper']:
                opp[pid] = p
        self.ids = sorted(own)
        self.own, self.gk, self.op = own, gk, opp
        b = self.m['scene'].get('ball')
        self.ball = to_attack(b[0], b[1], side, self.L, self.W) if self.m['scene'].get('ballReliable') and b else None
        bs = self.m['before']['players'][alt['mover']]
        self.before = to_attack(bs['x'], bs['y'], side, self.L, self.W)
        self.poly = [to_attack(p[0], p[1], side, self.L, self.W) for p in alt['zone']['polygon']]
        R = self.o['relations']
        self.rel = {key: R[key][self.k] for key in R}

    # ----- the relation, re-measured with the mover at q -----
    def arrays(self, q):
        X = np.array([q[0] if pid == self.alt['mover'] else self.own[pid][0] for pid in self.ids], float)
        Y = np.array([q[1] if pid == self.alt['mover'] else self.own[pid][1] for pid in self.ids], float)
        return X, Y

    def ball_xy(self):
        if self.ball is not None:
            return self.ball
        if self.rel.get('ballX') is not None:
            return self.rel['ballX'], self.rel.get('ballY')
        if self.o['detector'] == 'pressure_cover':
            return self.carrier()
        return None

    def carrier(self):
        c = self.o['actors'].get('carrier')
        if c in self.opp_pos():
            return self.opp_pos()[c]
        if self.rel.get('carrierX') is not None:
            return self.rel['carrierX'], self.rel['carrierY']
        return None

    def opp_pos(self):
        return self.op

    def outcome(self, q):
        """sample_outcomes' value with the mover at q (None = not judged)."""
        X, Y = self.arrays(q)
        d = self.o['detector']
        R = {'t': [0.0]}
        if d == 'pressure_cover':
            cx, cy = self.carrier()
            j, d1 = G.first_defender(cx, cy, X, Y)
            (cj, cd, ca), _ = G.cover_candidate(X[j], Y[j], X, Y, self.W, exclude=[j])
            same = self.ids[j] == self.rel['fd']
            R.update(d1=[d1], closingMS=[self.rel['closingMS'] if same else None],
                     coverDist=[None if cj is None else cd], coverAngle=[None if cj is None else ca],
                     coverVisible=[self.rel['coverVisible']], carrierX=[cx])
        elif d == 'weak_side_balance':
            bx, by = self.ball_xy()
            w = G.weak_side(bx, by, X, Y, self.W, self.scale, side=self.rel['ballSide'])
            R['weakCount'] = [w['weakCount']]
        elif d == 'behind_ball':
            bx = self.ball_xy()[0]
            n, _ = G.behind_ball(X, bx, G.BEHIND_BALL_M*self.L/105.0)
            thr = self.rel.get('threatN')
            R.update(ballX=[bx], behindN=[n], restMargin=[None if thr is None else n-thr])
        elif d == 'wide_defender_advance':
            R.update(self.d3_relations(X, Y))
        elif d == 'defensive_spacing':
            R.update(self.d5_relations(X, Y))
        out, _ = CX.sample_outcomes({'detector': d, 'relations': R}, self.params, self.scale,
                                    as_detector=self.as_detector)
        return out[0], R

    def roles_now(self):
        R, Sr, C = CX.roles_at(self.roles['rolebook'], self.ids, [self.m['t']])
        return list(R[0]), list(Sr[0]), list(C[0])

    def d3_parts(self):
        side = self.o['variant']
        w = self.ids.index(self.o['actors']['primary'][0])
        R, Sr, C = self.roles_now()
        if int(str(self.obs['timeline']['format']).split('v')[0]) == 11:
            cbs = [j for j, r in enumerate(R) if r in CX.CB_ROLES and j != w]
            dms = [j for j, (r, s) in enumerate(zip(R, Sr)) if r in CX.DM_ROLES and s in ('coach', 'stable')]
        else:
            back = []
            for j, (r, s, c) in enumerate(zip(R, Sr, C)):
                if r.startswith('back-') and (s == 'coach' or (s == 'relative' and c in ('medium', 'high'))):
                    a, b = r[5:].split('/')
                    back.append((j, int(a), int(b)))
            n = max((b for _, _, b in back), default=0)
            cbs = [j for j, a, _ in back if 1 < a < n and j != w]
            dms = []
        return side, w, cbs, dms

    def d3_relations(self, X, Y):
        side, w, cbs, dms = self.d3_parts()
        kl = self.L/105.0
        cbx = float(np.mean(X[cbs]))
        occ, _ = G.channel_occupied(X, Y, w, side, self.W, G.CHANNEL_BEHIND_M*kl)
        near = min(cbs, key=lambda c: Y[c]) if side == 'left' else max(cbs, key=lambda c: Y[c])
        geo = self.obs['roleInputs'].get('detect') or {}
        ts = geo.get('samples', {}).get('t') or []
        period = '-1'
        if ts:
            i = int(np.argmin(np.abs(np.asarray(ts)-self.m['t'])))
            if abs(ts[i]-self.m['t']) <= .11:
                period = str(geo['samples']['period'][i])
        med = (geo.get('medianY') or {}).get(self.ids[near]) or {}
        mv = med.get(period, med.get('-1'))
        sh = G.cb_shift(Y[near], mv if mv is not None else math.nan, side)
        dm_seen = [d for d in dms if np.isfinite(X[d])]
        och = 0
        for o in self.op.values():
            if G.in_outer_third(o[1], side, self.W) and o[0] < X[w] and o[0] < self.L/2:
                och += 1
        bxy = self.ball_xy()
        behind = self.rel.get('behindN')
        if bxy is not None and behind is not None:
            behind = G.behind_ball(X, bxy[0], G.BEHIND_BALL_M*kl)[0]
        thr = self.rel.get('threatN')
        out = {'advanceM': [X[w]-cbx], 'beyondHalfM': [X[w]-self.L/2], 'channelOccupied': [occ],
               'cbAdjusted': [None if not math.isfinite(sh) else sh >= G.CB_SHIFT_M],
               'interchange': [self.rel.get('interchange')],
               'dmDropped': [any(X[d] <= cbx+G.DM_DROP_M for d in dm_seen) if dm_seen else None],
               'opponentsInChannel': [och],
               'wdVelX': [None if self.ids[w] == self.alt['mover'] else self.rel.get('wdVelX')],
               'behindN': [behind], 'threatN': [thr],
               'restMargin': [None if (behind is None or thr is None) else behind-thr]}
        if 'ballX' in self.o['relations']:          # the detector's relations carry the ball depth: so does this one
            out['ballX'] = [None if bxy is None else bxy[0]]
        return out

    def channel_threat(self):
        """(id, depth) of the deepest opponent in the vacated channel at the moment (outer third of the advance's
        side, goal-side of the wide defender, in our half), None when there is none or this is not an advance."""
        if self.o['detector'] != 'wide_defender_advance':
            return None
        side = self.o['variant']
        xw = self.own[self.o['actors']['primary'][0]][0]
        best = None
        for pid in sorted(self.op):
            x, y = self.op[pid]
            if G.in_outer_third(y, side, self.W) and x < xw and x < self.L/2 and (best is None or x < best[1]):
                best = (pid, x)
        return best

    def someone_back(self):
        """Is the zone a "someone stays back" zone (the D4 region)?"""
        d = self.o['detector']
        return d == 'behind_ball' or (d == 'wide_defender_advance' and (self.as_detector == 'behind_ball' or
                                                                         self.team_level))

    def d5_members(self):
        a, b = self.rel['gapLeft'], self.rel['gapRight']
        others = sorted((pid for pid in self.ids if pid not in (a, b)),
                        key=lambda p: (abs(self.own[p][0]-self.rel['lineHeightM']), p))
        return [a, b]+others[:max(0, int(self.rel['lineSize'])-2)]

    def d5_relations(self, X, Y):
        mem = [self.ids.index(p) for p in self.d5_members()]
        gap, li, ri, gy = G.lateral_gaps(Y[mem])
        left, right = (X[mem][li], Y[mem][li]), (X[mem][ri], Y[mem][ri])
        OX = np.array([o[0] for o in self.op.values()], float)
        OY = np.array([o[1] for o in self.op.values()], float)
        inn = G.opponent_in_gap(OX, OY, left, right, self.scale) if len(OX) else []
        side = self.rel.get('ballSide')
        mean0 = float(np.mean([self.own[p][0] for p in self.d5_members()]))
        dm0 = self.rel.get('dmGapM')
        return {'lateralGapM': [gap], 'gapBallSide': [None if side is None else (gy < self.W/2) == (side == 'left')],
                'opponentInGap': [bool(inn)], 'lineStdM': [G.line_std(X[mem])],
                'dmGapM': [None if dm0 is None else dm0+mean0-float(np.mean(X[mem]))]}

    def in_region(self, q):
        """DESIGN 5.2's principle region of the moment's problem (module docstring), at the scene's positions."""
        d = self.o['detector']
        kl, kw = self.L/105.0, self.W/68.0
        x, y = q
        if d == 'pressure_cover':
            cx, cy = self.carrier()
            if self.alt['improves']['relation'] == 'pressure':
                pm = float(self.params.get('pressureM') or PRESSURE_M)
                return math.hypot(x-cx, y-cy) <= pm and x <= cx-1.0
            fx, fy = self.own[self.rel['fd']]
            r = G.cover_radius(self.scale, self.params.get('coverRadiusM', 12.0), self.params.get('coverRadiusMinM', 6.0))
            dd = math.hypot(x-fx, y-fy)
            if dd < 4.0 or dd > r or x >= fx:
                return False
            ang = G.angle_at(fx, fy, x, y, 0.0, self.W/2)
            return ang <= min(float(self.params.get('coverAngleDeg') or 60.0), 60.0)
        if d == 'weak_side_balance':
            bx, by = self.ball_xy()
            X = sorted(p[0] for p in self.own.values())
            line = [v for v in X if v <= X[0]+LINE_BAND_M*kl]
            med = float(np.median(line))
            far = y > self.W/2 if self.rel['ballSide'] == 'left' else y < self.W/2
            return max(0.0, med-8.0*kl) <= x <= min(bx, med+8.0*kl) and far
        if self.someone_back():
            bx = self.ball_xy()[0]
            return x <= bx-10.0*kl and self.W/6 <= y <= 5*self.W/6
        if d == 'wide_defender_advance':
            side, w, cbs, _ = self.d3_parts()
            X0 = np.array([self.own[p][0] for p in self.ids])
            cbx = float(np.mean(X0[cbs]))
            return (cbx-5.0 <= x <= X0[w]-10.0*kl) and bool(G.in_outer_third(y, side, self.W))
        if d == 'defensive_spacing':
            mem = self.d5_members()
            lx = float(np.mean([self.own[p][0] for p in mem]))
            ya, yb = sorted((self.own[self.rel['gapLeft']][1], self.own[self.rel['gapRight']][1]))
            return ya < y < yb and abs(x-lx) <= 3.0
        return False

    def restored(self, q):
        out, R = self.outcome(q)
        d = self.o['detector']
        if d == 'defensive_spacing':
            return out in (0, 1)
        if self.team_level:         # level 2: the rule stops failing and someone now stands behind the ball
            return out != -1 and (R['behindN'][0] or 0) >= 1
        if d == 'wide_defender_advance' and not self.as_detector:
            if out == 1:
                return True
            if self.params.get('advancedRule') == 'halfway':
                return R['beyondHalfM'][0] < float(self.params.get('advancedHalfwayShare') or .1)*self.L
            adv = G.scaled(float(self.params.get('advancedM') or 15.0), self.params.get('advancedScale', 'length'),
                           self.scale)
            return R['advanceM'][0] < adv
        return out == 1

    # ----- the five checks -----
    def mates(self):
        pts = [p for pid, p in self.own.items() if pid != self.alt['mover']]
        return pts+list(self.gk.values())

    def checks_ok(self, q):
        X, Y = self.arrays(q)
        X0 = np.array([self.own[p][0] for p in self.ids])
        Y0 = np.array([self.own[p][1] for p in self.ids])
        pm = float(self.params.get('pressureM') or PRESSURE_M) if self.o['detector'] == 'pressure_cover' \
            else PRESSURE_M
        c = None
        if self.o['detector'] == 'pressure_cover':
            c = self.carrier()
        elif self.o['detector'] in ('weak_side_balance', 'defensive_spacing'):
            cid = self.o['actors'].get('carrier')
            c = self.opp_pos().get(cid) if cid else None
        bad = []
        if c is not None and np.any(np.hypot(X0-c[0], Y0-c[1]) <= pm) and not np.any(np.hypot(X-c[0], Y-c[1]) <= pm):
            bad.append('pressureKept')
        bxy = self.ball_xy()
        band = G.CENTRAL_BAND_M*self.W/68.0
        if bxy is not None:
            def central(XX, YY):
                return int(np.sum((XX <= bxy[0]) & (np.abs(YY-self.W/2) <= band)))
            if central(X0, Y0) >= 1 and central(X, Y) < 1:
                bad.append('centralKept')
        thr = G.scaled(float((self.profile['detectors']['defensive_spacing']['params'].get('lateralGapM') or 20.0)
                             if self.profile['detectors']['defensive_spacing']['enabled'] else 20.0), 'width',
                       self.scale)
        if not self.line_bad(X0, Y0, thr) and self.line_bad(X, Y, thr):
            bad.append('gapOk')
        mark = MARK_M*self.L/105.0
        for o in self.op.values():
            if o[0] >= self.L/2:
                continue
            was = np.any(np.hypot(X0-o[0], Y0-o[1]) <= mark)
            now = np.any(np.hypot(X-o[0], Y-o[1]) <= mark)
            if was and not now:
                bad.append('noNewFreeOpponent')
                break
        return bad

    def ball_side(self):
        if self.o['detector'] in ('weak_side_balance', 'defensive_spacing') and self.rel.get('ballSide'):
            return self.rel['ballSide']
        b = self.ball_xy()
        if b is None or b[1] is None:
            return None
        return 'left' if b[1] < self.W/2 else 'right'

    def line_bad(self, X, Y, thr):
        if self.o['detector'] == 'defensive_spacing':
            mem = [self.ids.index(p) for p in self.d5_members()]
        else:
            deep = np.nanmin(X)
            mem = [j for j in range(len(X)) if X[j] <= deep+LINE_BAND_M*self.L/105.0]
        if len(mem) < 3:
            return False
        gap, _, _, gy = G.lateral_gaps(Y[mem])
        side = self.ball_side()
        on = True if side is None else ((gy < self.W/2) if side == 'left' else (gy > self.W/2))
        return bool(gap > thr and on)

    # ----- the grid inside the polygon -----
    def grid_inside(self, h):
        xs = [p[0] for p in self.poly]
        ys = [p[1] for p in self.poly]
        out = []
        for i in range(int(math.floor(min(xs)/h)), int(math.ceil(max(xs)/h))+1):
            for j in range(int(math.floor(min(ys)/h)), int(math.ceil(max(ys)/h))+1):
                q = (i*h, j*h)
                if inside(self.poly, q):
                    out.append(q)
        return out


def zone_problems(alt, finding, obs, roles, profile, side):
    """[problem] of one Alternative against the DESIGN 13.1 properties (module docstring); [] when it holds."""
    probs = [f'schema: {p}' for p in S.validate('alternative', alt)]
    c = Case(alt, finding, obs, roles, profile, side)
    r = alt['reach']['radiusM']
    for v in c.poly:
        if math.hypot(v[0]-c.before[0], v[1]-c.before[1]) > r+TOL:
            probs.append(f'reach: vertex {v} is {math.hypot(v[0]-c.before[0], v[1]-c.before[1]):.2f} m > {r} m')
            break
    for q in c.mates():
        if inside(c.poly, q) or edge_distance(c.poly, q) < STACK_M-TOL:
            probs.append(f'stacking: a team-mate at {q} is {edge_distance(c.poly, q):.2f} m from the zone')
            break
    h = max(.5, c.L/105.0)
    pts = c.grid_inside(h)
    if len(pts) < MIN_POINTS:
        probs.append(f'restore: {len(pts)} grid points inside the zone, fewer than {MIN_POINTS}')
        if not pts:
            return probs
    ok = [c.restored(q) for q in pts]
    share = sum(ok)/len(pts)
    if share < KEEP-1e-9:
        probs.append(f'restore: {share:.2f} of {len(pts)} grid points restore the relation')
    for q in pts:
        bad = c.checks_ok(q)
        if bad:
            probs.append(f'checks: {bad} fail at {q}')
            break
    inreg = sum(c.in_region(q) for q in pts)/len(pts)
    if inreg < KEEP-1e-9:
        probs.append(f'region: {inreg:.2f} of {len(pts)} grid points in the principle region')
    top = max(v[0] for v in c.poly)
    threat = c.channel_threat()
    if threat is not None and top > threat[1]+LEVEL_M+TOL:
        probs.append(f'goal-side: the zone reaches {top:.2f} m, upfield of the channel opponent {threat[0]} at '
                     f'{threat[1]:.2f} m (+{LEVEL_M} m level)')
    if c.someone_back():
        b = c.ball_xy()
        if b is not None and top > b[0]+TOL:
            probs.append(f'behind the ball: the zone reaches {top:.2f} m, ahead of the ball at {b[0]:.2f} m')
    if alt['improves']['before'] or not alt['improves']['after']:
        probs.append('improves: must go from false to true')
    if not all(alt['checks'].values()):
        probs.append('checks: a check is false in the record')
    if alt['mover'] in (c.o['actors'].get('carrier'),):
        probs.append('mover: the carrier')
    return probs


def fine_share(alt, finding, obs, roles, profile, side, h=.25):
    """The restoring share on a finer grid (discretisation cross-check)."""
    c = Case(alt, finding, obs, roles, profile, side)
    pts = c.grid_inside(h)
    return sum(c.restored(q) for q in pts)/len(pts) if pts else 0.0


def on_ball_ids(alt, finding, obs, roles, profile, side):
    c = Case(alt, finding, obs, roles, profile, side)
    b = c.ball
    if b is None:
        return set()
    return {pid for pid, p in c.own.items() if math.hypot(p[0]-b[0], p[1]-b[1]) <= ON_BALL_M}
