"""One persistent match ball.

The detector proposes ball candidates; most small white things on a pitch are not the ball. Each
candidate is scored on several properties at once (detector confidence, size against the expected
ball size at that depth, compact round shape, isolation on the grass, location on the pitch) and
hard-rejected when it is clearly something else:

- FIELD_LINE   part of a long thin white structure (touchline, halfway line, box lines, circle)
- STATIONARY   a white spot that stays at the same camera-compensated (or pitch) position
- SIZE         far too large or too small for a ball at that depth
- OUTSIDE_PITCH far outside the playable field
- PLAYER_PART  white socks, shoes or kit inside a person's body (only when no track explains it)

A single Kalman-filtered track (constant velocity, camera-compensated) follows the ball. Detections
close to its predicted path are preferred; a detection elsewhere has to build its own evidence as a
tentative track (several hits, real motion) before it can replace a lost ball. When the ball is
hidden the track is MISSING and its position is predicted for a short time (following the player it
disappeared next to); after that the ball is UNKNOWN rather than a guess.
"""
import math
from collections import Counter, deque
from dataclasses import dataclass, field

import cv2
import numpy as np

from .geometry import clamp, signed_distance

BALL_DIAMETER = .22      # metres
PLAYER_HEIGHT = 1.8      # metres
MAX_SPEED = 40.0         # m/s: a hard shot; faster jumps are not the same ball
ACCEL_NOISE = 25.0       # m/s^2 process noise for the Kalman filter (kicks, bounces)
MAX_MISSING = 1.0        # seconds a free ball is predicted while unseen
MAX_CARRIED = 2.5        # ... while it is next to (hidden by) a player
STATIC_SECONDS = 2.0     # a candidate seen this long at one spot without moving is a painted/static spot
STATIC_RADIUS = .6       # metres
SHOW_CONFIDENCE = .35    # below this the ball is reported UNKNOWN
REASONS = {'FIELD_LINE': 'BALL REJECTED: FIELD LINE', 'STATIONARY': 'BALL REJECTED: STATIONARY', 'SIZE': 'BALL REJECTED: SIZE',
           'OUTSIDE_PITCH': 'BALL REJECTED: OUTSIDE PITCH', 'PLAYER_PART': 'BALL REJECTED: PLAYER PART',
           'TRAJECTORY': 'BALL REJECTED: TRAJECTORY', 'LOW_SCORE': 'BALL REJECTED: LOW CONFIDENCE'}


class Kalman:
    """Constant-velocity Kalman filter in image pixels; state [x, y, vx, vy] (px, px/s)."""

    def __init__(self, x, y, size_px, px_per_m):
        self.x = np.array([x, y, 0.0, 0.0])
        vel = 10*px_per_m
        self.P = np.diag([size_px**2, size_px**2, vel**2, vel**2])
        self.px_per_m = px_per_m

    def compensate(self, M):
        """Move the state with the camera: positions through M, velocities through its linear part."""
        p = M @ np.array([self.x[0], self.x[1], 1.0])
        A = M[:2, :2]
        J = np.zeros((4, 4))
        J[:2, :2] = A
        J[2:, 2:] = A
        self.x = np.r_[p[:2]/p[2], A @ self.x[2:]]
        self.P = J @ self.P @ J.T

    def predict(self, dt):
        F = np.eye(4)
        F[0, 2] = F[1, 3] = dt
        q = (ACCEL_NOISE*self.px_per_m)**2
        G = np.array([[dt*dt/2, 0], [0, dt*dt/2], [dt, 0], [0, dt]])
        self.x = F @ self.x
        self.P = F @ self.P @ F.T+G @ G.T*q

    def innovation(self, z, sigma):
        H = np.eye(2, 4)
        y = np.asarray(z)-H @ self.x
        S = H @ self.P @ H.T+np.eye(2)*sigma**2
        return y, S

    def update(self, z, sigma):
        H = np.eye(2, 4)
        y, S = self.innovation(z, sigma)
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x+K @ y
        self.P = (np.eye(4)-K @ H) @ self.P

    def mahalanobis(self, z, sigma):
        y, S = self.innovation(z, sigma)
        return float(y @ np.linalg.inv(S) @ y)


@dataclass
class Candidate:
    box: list            # pixels [x1, y1, x2, y2]
    det: float
    cx: float
    cy: float
    d: float             # diameter, px
    scores: dict = field(default_factory=dict)
    base: float = 0.0
    reason: str = ''     # hard rejection
    status: str = 'candidate'
    stab: tuple = None
    traj: float = None
    final: float = 0.0


@dataclass
class Tentative:
    kf: Kalman
    first: float
    last: float
    hits: deque = field(default_factory=lambda: deque(maxlen=12))   # (time, base, det, stab)


def _component_stats(mask, cx, cy, d):
    """Connected bright component at the candidate centre: (area, major, minor, touches_border, circularity)."""
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if count <= 1:
        return None
    h, w = mask.shape
    r = max(1, int(round(d/2)))
    ys, xs = np.ogrid[:h, :w]
    disk = (xs-cx)**2+(ys-cy)**2 <= r*r
    touching = np.unique(labels[disk & (labels > 0)])
    if not len(touching):
        return None
    label = max(touching, key=lambda k: stats[k, cv2.CC_STAT_AREA])
    comp = labels == label
    py, px = np.nonzero(comp)
    area = len(px)
    if area < 3:
        return {'area': area, 'major': 1.0, 'minor': 1.0, 'border': False, 'circularity': 1.0, 'bulge': True}
    cov = np.cov(np.vstack([px, py]))
    evals, evecs = np.linalg.eigh(cov)
    major, minor = 4*math.sqrt(max(evals[1], 1e-6)), 4*math.sqrt(max(evals[0], 1e-6))
    x0, y0, bw, bh = stats[label, :4]
    border = x0 == 0 or y0 == 0 or x0+bw >= w or y0+bh >= h
    contours, _ = cv2.findContours(comp.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    perimeter = max(cv2.arcLength(contours[0], True), 1.0) if contours else 1.0
    # Thickness across the structure at the candidate vs further along it: a ball lying on a painted
    # line makes a bulge, a plain line keeps its width.
    along = (px-cx)*evecs[0, 1]+(py-cy)*evecs[1, 1]
    across = (px-cx)*evecs[0, 0]+(py-cy)*evecs[1, 0]
    here, far = np.abs(along) <= max(1.0, d/2), np.abs(along) >= max(2.0, d)
    thick_here = float(np.ptp(across[here])+1) if here.sum() > 1 else 0.0
    thick_far = float(np.ptp(across[far])+1) if far.sum() > 3 else thick_here
    bulge = thick_here >= max(.7*d, 1.8*thick_far)
    return {'area': area, 'major': major, 'minor': minor, 'border': bool(border), 'bulge': bulge,
            'circularity': clamp(4*math.pi*area/perimeter**2)}


def analyse_patch(frame, cx, cy, d):
    """Shape, isolation and field-line evidence from the pixels around a candidate."""
    H, W = frame.shape[:2]
    R = int(max(10, 3*d))
    x0, y0, x1, y1 = max(0, int(cx)-R), max(0, int(cy)-R), min(W, int(cx)+R+1), min(H, int(cy)+R+1)
    patch = frame[y0:y1, x0:x1]
    if patch.size == 0:
        return {'shape': .5, 'isolation': .5, 'line': 0.0, 'large': False}
    s = patch.astype(np.int16)
    b, g, r = s[..., 0], s[..., 1], s[..., 2]
    grass = (g > 30) & (g > r*.95) & (g > b*1.12) & (g-b > 12)
    low = patch.min(axis=2).astype(np.int16)
    grass_low = float(np.median(low[grass])) if grass.sum() > 10 else float(np.median(low))
    bright = (low >= max(110, grass_low+55)) & ((patch.max(axis=2).astype(np.int16)-low) <= 110)
    lx, ly = cx-x0, cy-y0
    yy, xx = np.ogrid[:patch.shape[0], :patch.shape[1]]
    dist = np.sqrt((xx-lx)**2+(yy-ly)**2)
    ring = (dist >= 1.2*max(d, 3)/2+1) & (dist <= 1.25*max(d, 3)+2)
    isolation = float(grass[ring].mean()) if ring.any() else .5
    comp = _component_stats(bright, lx, ly, d)
    line = 0.0
    large = False
    if comp is not None:
        elongation = comp['major']/max(comp['minor'], 1.0)
        # A long thin bright structure through the candidate: painted line or arc (or a ball on one: bulge).
        if comp['major'] >= 2.5*max(d, 3) and elongation >= 3:
            line = .3 if comp['bulge'] else 1.0
        elif comp['major'] >= 1.8*max(d, 3) and elongation >= 2.2 and comp['border']:
            line = .3 if comp['bulge'] else .6
        large = comp['area'] >= 6*math.pi*(max(d, 3)/2)**2 and elongation < 3
    if d < 7 or comp is None:
        shape = .6  # too small to judge shape reliably
    else:
        aspect = comp['minor']/max(comp['major'], 1e-6)
        shape = clamp((comp['circularity']-.35)/.5)*clamp((aspect-.35)/.4)
    return {'shape': round(shape, 3), 'isolation': round(isolation, 3), 'line': line, 'large': large}


class BallTracker:
    def __init__(self, width, height, fps):
        self.W, self.H, self.fps = width, height, max(1.0, fps)
        self.kf = None
        self.state = 'UNKNOWN'
        self.confidence = 0.0
        self.track = 0
        self.last_seen = -math.inf
        self.last_time = None
        self.last_d = 8.0
        self.carrier = None          # (person track id, offset) when lost next to a player
        self.lost_at = None          # (time, x, y) of the last lost ball, for re-acquisition
        self.trajectory = deque(maxlen=int(self.fps*2))   # (time, x, y, state) of the shown ball, for previews
        self.path = deque(maxlen=int(self.fps*4))   # (time, stabilized point) of the active track
        self.started = -math.inf
        self.recent = deque(maxlen=int(self.fps))   # trajectory scores of recent updates
        self.tentative = []
        self.spots = []              # static-spot memory: [{'x','y','first','last','hits','segment'}]
        self.rejections = Counter()
        self.since_log = Counter()
        self.last_log = -math.inf
        self.events = []

    # ---------- helpers ----------
    def _px_per_m(self, y_norm, scale):
        height = scale(y_norm) if scale else None
        if not height or height <= 0:
            height = .09  # unknown: a typical broadcast player is ~9% of the frame height
        return height*self.H/PLAYER_HEIGHT

    def _event(self, time, title, lines):
        self.events.append({'time': round(float(time), 3), 'kind': 'ball', 'track': None, 'playerId': None,
                            'message': '\n'.join([title]+lines)})

    def _spot(self, stab, segment):
        aspect = self.W/self.H
        for s in self.spots:
            if s['segment'] == segment and math.hypot((stab[0]-s['x'])*aspect, stab[1]-s['y']) <= s['r']:
                return s
        return None

    def _static(self, spot):
        return spot is not None and spot['hits'] >= 6 and spot['last']-spot['first'] >= STATIC_SECONDS

    # ---------- main ----------
    def step(self, frame, time, rows, M, reliable, cut, pitch, people, scale, stabilize, segment, to_pitch=None):
        """rows: (N, 6) ball detections in pixels. people: [(track id, normalized box)].
        scale(y) -> expected player height (normalized) at foot height y, or None.
        stabilize(x, y) -> camera-compensated normalized point. Returns the frame's ball dict and
        debug candidates."""
        dt = 1/self.fps if self.last_time is None else max(1e-3, time-self.last_time)
        self.last_time = time
        if cut and self.kf is not None:
            self._lose(time, 'camera cut')
        if cut:
            self.tentative = []
        for kf in ([self.kf] if self.kf is not None else [])+[t.kf for t in self.tentative]:
            if reliable:
                kf.compensate(M)
            else:
                kf.P[:2, :2] += np.eye(2)*(self.last_d*3)**2
        if self.kf is not None:
            self.kf.predict(dt)
        cands = self._candidates(frame, rows, pitch, people, scale, stabilize, segment, to_pitch, time)
        chosen = self._associate(cands, time, scale)
        promoted = self._tentatives(cands, chosen, time, dt, scale)
        chosen = chosen or promoted
        if chosen is None and self.kf is not None:
            self._missing(time, people, scale)
        elif chosen is not None:
            self._check_parked(chosen, time, segment)
        self._remember_spots(cands, chosen, time, segment, scale)
        for c in cands:
            if c.reason and c.status != 'ball':
                self.rejections[c.reason] += 1
                self.since_log[c.reason] += 1
        self._log(time, chosen)
        out = self._output(time, chosen, to_pitch)
        if out['state'] in ('TRACKED', 'MISSING'):
            self.trajectory.append((round(time, 3), out['center'][0], out['center'][1], out['state']))
        return out, [self._debug(c) for c in cands[:12]]

    def _candidates(self, frame, rows, pitch, people, scale, stabilize, segment, to_pitch, time):
        cands = []
        for x1, y1, x2, y2, conf, _ in np.asarray(rows, np.float64).reshape(-1, 6):
            w, h = x2-x1, y2-y1
            if w <= 0 or h <= 0:
                continue
            c = Candidate([x1, y1, x2, y2], float(conf), (x1+x2)/2, (y1+y2)/2, math.sqrt(w*h))
            ground = (c.cy+c.d/2)/self.H
            px_m = self._px_per_m(ground, scale)
            expected = BALL_DIAMETER*px_m
            ratio = c.d/max(expected, 1e-6)
            size = math.exp(-.5*(math.log(max(ratio, 1e-3))/math.log(1.9))**2)
            patch = analyse_patch(frame, c.cx, c.cy, c.d)
            nx, ny = c.cx/self.W, c.cy/self.H
            # Field location (the ball can fly above the pitch in the image, so only far-away is rejected).
            if pitch is not None and pitch.reliable and len(pitch.polygon) >= 3:
                out = signed_distance((nx, ny), pitch.polygon, pitch.aspect)
                fieldscore = 1.0 if out <= 0 else math.exp(-out/.04)
            else:
                out, fieldscore = 0.0, .7
            # Inside a person's body (not at the feet): white socks, shoes, shorts, sleeves.
            body = any(bx+.15*bw < nx < bx+.85*bw and by+.05*bh < ny < by+.88*bh for _, (bx, by, bw, bh) in people)
            c.stab = stabilize(nx, ny)
            c.scores = {'det': round(clamp((c.det-.08)/.45), 3), 'size': round(size, 3), 'shape': patch['shape'],
                        'isolation': patch['isolation'], 'field': round(fieldscore, 3), 'line': patch['line']}
            c.base = round((.35*c.scores['det']+.2*size+.15*patch['shape']+.15*patch['isolation']+.15*fieldscore)
                           * (.5 if body else 1.0)*(1-.5*patch['line']), 3)
            if patch['line'] >= 1.0:
                c.reason = 'FIELD_LINE'
            elif ratio > 3.0 or ratio < .3 or patch['large']:
                c.reason = 'SIZE'
            elif out > .12:
                c.reason = 'OUTSIDE_PITCH'
            else:
                spot = self._spot(c.stab, segment)
                known = to_pitch is not None and _painted_spot(to_pitch((nx, ny+c.d/2/self.H)))
                # A white spot that was visible at the same time as the tracked ball, elsewhere, is not the ball.
                if self._static(spot) or known or (spot is not None and spot['coexisted'] and spot['hits'] >= 3):
                    c.reason = 'STATIONARY'
                elif body:
                    c.reason = 'PLAYER_PART'
            cands.append(c)
        cands.sort(key=lambda c: -c.base)
        return cands

    def _associate(self, cands, time, scale):
        """Best candidate on the active track's predicted path, or None."""
        if self.kf is None:
            return None
        px_m = self._px_per_m(self.kf.x[1]/self.H, scale)
        gap = max(1/self.fps, time-self.last_seen) if self.last_seen > -math.inf else 1/self.fps
        if self.state == 'MISSING':
            # Reappearing near the predicted position only; a ball that travelled far while hidden is found
            # again by a tentative track with real motion, never by jumping to whatever is nearby.
            allowed = (1.5+12*gap)*px_m+1.5*self.last_d
        else:
            allowed = MAX_SPEED*gap*px_m+1.5*self.last_d
        best = None
        recent = time-self.last_seen <= max(3/self.fps, .15)
        for c in cands:
            # A track that is following the ball may follow it across a painted line or to rest, but only
            # right on its predicted path; sizes never fit, and lines or spots never feed a lost track.
            if c.reason == 'SIZE':
                continue
            if c.reason in ('FIELD_LINE', 'STATIONARY') and not recent:
                continue
            err = math.hypot(c.cx-self.kf.x[0], c.cy-self.kf.x[1])
            if err > allowed:
                if c.base >= .45 and not c.reason:
                    c.reason = 'TRAJECTORY'
                continue
            m2 = self.kf.mahalanobis((c.cx, c.cy), max(1.0, c.d/4))
            c.traj = round(math.exp(-m2/18), 3)
            if c.reason in ('FIELD_LINE', 'STATIONARY') and (c.traj < .6 or err > 1.5*max(c.d, self.last_d)+.5*MAX_SPEED*gap*px_m*.25):
                continue
            c.final = round(.55*c.base+.45*c.traj+(.1 if c.reason == 'PLAYER_PART' else 0), 3)
            if c.final >= .3 and (best is None or c.final > best.final):
                best = c
        if best is not None:
            for c in cands:
                if c is not best and not c.reason and c.base >= .3:
                    c.reason = 'TRAJECTORY'
            best.reason, best.status = '', 'ball'
            self.kf.update((best.cx, best.cy), max(1.0, best.d/4))
            self.state, self.last_seen, self.last_d = 'TRACKED', time, best.d
            self.confidence = clamp(.6*self.confidence+.4*best.final, 0, .99)
            self.recent.append(best.traj)
            self.carrier = None
        return best

    def _tentatives(self, cands, chosen, time, dt, scale):
        """Unexplained, plausible candidates build tentative tracks; one may become the ball when the
        active track is lost (or missing for a while)."""
        for t in self.tentative:
            t.kf.predict(dt)
        free = [c for c in cands if c is not chosen and c.reason in ('', 'TRAJECTORY') and c.base >= .3]
        for c in free:
            best, best_err = None, math.inf
            for t in self.tentative:
                allowed = MAX_SPEED*max(dt, time-t.last)*t.kf.px_per_m+1.5*c.d
                err = math.hypot(c.cx-t.kf.x[0], c.cy-t.kf.x[1])
                if err <= allowed and err < best_err:
                    best, best_err = t, err
            if best is None:
                if len(self.tentative) >= 6:
                    self.tentative.sort(key=lambda t: (len(t.hits), t.last))
                    self.tentative.pop(0)
                best = Tentative(Kalman(c.cx, c.cy, max(c.d, 4), self._px_per_m(c.cy/self.H, scale)), time, time)
                self.tentative.append(best)
            else:
                best.kf.update((c.cx, c.cy), max(1.0, c.d/4))
            best.last = time
            best.hits.append((time, c.base, c.det, c.stab, c))
        self.tentative = [t for t in self.tentative if time-t.last <= .4]
        if self.kf is not None and not (self.state == 'MISSING' and time-self.last_seen > .3):
            return None
        ready = []
        for t in self.tentative:
            hits = [h for h in t.hits if time-h[0] <= .6]
            if len(hits) < min(3, max(2, int(self.fps*.12))):
                continue
            mean = sum(h[1] for h in hits)/len(hits)
            moved = math.hypot(hits[-1][3][0]-hits[0][3][0], hits[-1][3][1]-hits[0][3][1])
            px_m = t.kf.px_per_m
            moved_m = moved*self.H/px_m if px_m > 0 else 0
            near_lost = self.lost_at is not None and time-self.lost_at[0] <= 4 and \
                math.hypot(t.kf.x[0]-self.lost_at[1], t.kf.x[1]-self.lost_at[2]) <= (2+MAX_SPEED*(time-self.lost_at[0])*.5)*px_m
            if mean >= .45 and (moved_m >= .6 or (near_lost and mean >= .5)):
                ready.append((mean*len(hits), t, hits, moved_m, near_lost))
        if not ready:
            return None
        _, t, hits, moved_m, near_lost = max(ready, key=lambda r: r[0])
        title = 'BALL REACQUIRED' if near_lost else 'BALL TRACK SWITCH' if self.kf is not None else 'BALL ACQUIRED'
        self.kf = t.kf
        self.tentative.remove(t)
        last = hits[-1][4]
        self.state, self.last_seen, self.last_d = 'TRACKED', time, last.d
        self.confidence = clamp(sum(h[1] for h in hits)/len(hits), 0, .99)
        if title != 'BALL REACQUIRED':
            self.track += 1
            self.trajectory.clear()
        self.started = time
        self.last_log = time
        self.path.clear()
        self.recent.clear()
        self.carrier = None
        self.lost_at = None
        last.status, last.reason = 'ball', ''
        last.traj, last.final = 1.0, round(self.confidence, 3)
        self._event(time, title, [f'Candidate: x = {last.cx:.0f}, y = {last.cy:.0f} px', f'Detector confidence: {last.det:.2f}',
                                  f"Size score: {last.scores['size']:.2f}", f"Shape score: {last.scores['shape']:.2f}",
                                  f'Motion consistency: {len(hits)} hits in {hits[-1][0]-hits[0][0]:.2f} s, moved {moved_m:.1f} m',
                                  f'Final confidence: {self.confidence:.2f}'])
        return last

    def _missing(self, time, people, scale):
        """No detection on the predicted path: the ball is hidden (or gone)."""
        x, y = self.kf.x[0], self.kf.x[1]
        if self.state == 'TRACKED':
            # Lost right next to a player: it is probably at their feet, hidden by their body.
            px_m = self._px_per_m(y/self.H, scale)
            best = None
            for pid, (bx, by, bw, bh) in people:
                fx, fy = (bx+bw/2)*self.W, (by+bh)*self.H
                dist = math.hypot(fx-x, fy-y)
                if dist <= 1.2*px_m+bw*self.W/2 and (best is None or dist < best[0]):
                    best = (dist, pid, (x-fx, y-fy))
            self.carrier = (best[1], best[2]) if best else None
            self.state = 'MISSING'
        if self.carrier is not None:
            box = next((b for pid, b in people if pid == self.carrier[0]), None)
            if box is not None:
                fx, fy = (box[0]+box[2]/2)*self.W, (box[1]+box[3])*self.H
                self.kf.x[:2] = (fx+self.carrier[1][0]*.8, fy+self.carrier[1][1]*.8)
                self.kf.x[2:] *= .5
            else:
                self.carrier = None
        elif time-self.last_seen > .2:
            self.kf.x[2:] *= .9  # rolling friction while unseen
        missing = time-self.last_seen
        self.confidence *= math.exp(-(1/self.fps)/.8)
        limit = MAX_CARRIED if self.carrier is not None else MAX_MISSING
        if missing > limit:
            self._lose(time, f'not seen for {missing:.1f} s')

    def _check_parked(self, chosen, time, segment):
        """A track that never moves and sits on a white spot known from before it started has latched
        onto paint (a penalty or centre spot), not the ball: let it go. A ball that comes to rest after
        being followed (a free kick) keeps its track."""
        self.path.append((time, chosen.stab))
        if time-self.path[0][0] < 3:
            return
        old = next((p for t, p in self.path if time-t <= 3), self.path[0][1])
        spread = math.hypot(chosen.stab[0]-old[0], chosen.stab[1]-old[1])
        spot = self._spot(chosen.stab, segment)
        if spot is not None and spot['first'] < self.started-.5 and spread <= spot['r']:
            self._lose(time, 'stationary on a white spot that was there before the track started')

    def _lose(self, time, why):
        if self.kf is not None:
            self.lost_at = (time, float(self.kf.x[0]), float(self.kf.x[1]))
            self._event(time, 'BALL LOST', [f'Last position: x = {self.kf.x[0]:.0f}, y = {self.kf.x[1]:.0f} px', f'Reason: {why}',
                                             'The ball is UNKNOWN until a candidate shows consistent motion again.'])
        self.kf, self.state, self.confidence, self.carrier = None, 'UNKNOWN', 0.0, None
        self.recent.clear()
        self.path.clear()
        self.trajectory.clear()

    def _remember_spots(self, cands, chosen, time, segment, scale):
        """White spots that do not move are remembered (penalty and centre spots, debris, stickers)."""
        moving = chosen is not None and self.kf is not None and math.hypot(*self.kf.x[2:]) > 1.5*self.kf.px_per_m
        seen_with_ball = chosen is not None and self.state == 'TRACKED'
        for c in cands:
            if c is chosen and moving:
                continue
            spot = self._spot(c.stab, segment)
            if spot is None:
                r = STATIC_RADIUS*self._px_per_m((c.cy+c.d/2)/self.H, scale)/self.H
                spot = {'x': c.stab[0], 'y': c.stab[1], 'r': max(.004, r), 'first': time, 'last': time, 'hits': 1, 'segment': segment, 'coexisted': False}
                self.spots.append(spot)
            else:
                spot['hits'] += 1
                spot['last'] = time
            if spot['hits'] > 1:
                spot['x'] += (c.stab[0]-spot['x'])*.2
                spot['y'] += (c.stab[1]-spot['y'])*.2
            if seen_with_ball and c is not chosen:
                spot['coexisted'] = True
        self.spots = [s for s in self.spots if time-s['last'] <= 10 and s['segment'] == segment][-200:]

    def _log(self, time, chosen):
        if chosen is not None and time-self.last_log >= 2:
            rejected = ', '.join(f'{k} {v}' for k, v in self.since_log.most_common()) or 'none'
            motion = sum(self.recent)/len(self.recent) if self.recent else 0.0
            self._event(time, 'BALL TRACK UPDATE', [f'Candidate: x = {chosen.cx:.0f}, y = {chosen.cy:.0f} px',
                                                     f'Detector confidence: {chosen.det:.2f}', f'Motion consistency: {motion:.2f}',
                                                     f"Shape score: {chosen.scores['shape']:.2f}", f'Trajectory score: {chosen.traj:.2f}',
                                                     f'Final confidence: {self.confidence:.2f}', f'Rejected candidates since last update: {rejected}'])
            self.last_log = time
            self.since_log.clear()

    def _output(self, time, chosen, to_pitch):
        if self.kf is None or self.state == 'UNKNOWN' or self.confidence < SHOW_CONFIDENCE:
            return {'state': 'UNKNOWN', 'confidence': round(self.confidence, 3)}
        x, y = self.kf.x[0], self.kf.x[1]
        d = chosen.d if chosen is not None else self.last_d
        box = [x-d/2, y-d/2, x+d/2, y+d/2] if chosen is None else chosen.box
        out = {'state': self.state, 'confidence': round(self.confidence, 3), 'track': self.track,
               'box': [round(box[0]/self.W, 5), round(box[1]/self.H, 5), round((box[2]-box[0])/self.W, 5), round((box[3]-box[1])/self.H, 5)],
               'center': [round(x/self.W, 5), round(y/self.H, 5)],
               'velocity': [round(self.kf.x[2]/self.W, 4), round(self.kf.x[3]/self.H, 4)]}
        if self.state == 'MISSING':
            out['missingFor'] = round(time-self.last_seen, 2)
            if self.carrier is not None:
                out['nearTrack'] = self.carrier[0]
        if chosen is not None:
            out['detector'] = round(chosen.det, 3)
        if to_pitch is not None:
            p = to_pitch((x/self.W, (y+d/2)/self.H))
            if p is not None:
                out['pitch'] = [round(p[0], 4), round(p[1], 4)]
        return out

    def _debug(self, c):
        out = {'box': [round(c.box[0]/self.W, 5), round(c.box[1]/self.H, 5), round((c.box[2]-c.box[0])/self.W, 5), round((c.box[3]-c.box[1])/self.H, 5)],
               'det': round(c.det, 3), 'score': c.final if c.status == 'ball' else c.base, 'status': c.status}
        if c.reason and c.status != 'ball':
            out['reason'] = c.reason
            out['text'] = REASONS[c.reason]
        return out


PAINTED_SPOTS = ((11/105, .5), (94/105, .5), (.5, .5))


def _painted_spot(p):
    """Within ~1 m of a penalty spot or the centre spot on the calibrated pitch."""
    return p is not None and any(math.hypot((p[0]-x)*105, (p[1]-y)*68) <= 1.0 for x, y in PAINTED_SPOTS)


