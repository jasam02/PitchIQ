"""One persistent match ball, followed through time.

The tracker never asks "which white blob looks most like a ball in this frame?" but "which detection
is the ball I am already following?". Each frame:

    detector ball candidates
    -> evidence per candidate: size for that depth, compact shape, grass around it, location on
       the pitch, overlap with the painted lines, position inside a person's body
    -> candidate TRACKLETS (every candidate, not only the ball) add temporal evidence: independent
       motion in camera-compensated or pitch coordinates, attachment to a player (the same place
       inside one person's box for many frames: wrist tape, gloves, boots, socks), standing still
       for seconds (penalty spot, debris) and having been seen at the same time as the confidently
       tracked ball (there is only one ball)
    -> the ACTIVE BALL: predict with a camera-compensated Kalman filter, gate around the prediction
       (tight while LOCKED, wider while OCCLUDED, growing while RECOVERING, physically wide when a
       player is within reach because a kick changes the velocity), score the candidates with
       trajectory consistency as the strongest term, and resist switching to another object
    -> only when the ball is genuinely lost does a tracklet that stayed plausible for several frames
       become the (re)acquired ball

States: SEARCHING (no ball) -> LOCKED (observed this frame) -> OCCLUDED (short gap: predicted, kept
at the player that hides it) -> RECOVERING (longer gap: expanding search, confirmation required)
-> LOST -> SEARCHING. The output keeps one ball with one track id; while hidden the position is
predicted (state MISSING); without a confident ball it is UNKNOWN, never a guess.
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
KICK_SPEED = 15.0        # m/s: velocity change allowed while a player is within reach (kick, deflection)
ACCEL_NOISE = 30.0       # m/s^2 process noise of the Kalman filter (bounces, friction, spin)
GATE_FLOOR = .45         # metres: the gate around the prediction is never tighter than this
CONTACT_REACH = 1.5      # metres from a player's feet within which a touch, deflection or occlusion can happen
OCCLUDED_MAX = .4        # seconds of a short gap with immediate reconnection near the prediction
MAX_MISSING = 1.5        # seconds a free ball is predicted before it is LOST
MAX_CARRIED = 3.0        # ... while it is at (hidden by) a player
PARKED_MAX = 8.0         # seconds a ball may rest before the track waits for motion again
STATIC_SECONDS = 2.0     # a candidate seen this long at one spot without moving is a static false positive
STATIC_RADIUS = .6       # metres
LINE_RUN = .4            # seconds a moving track may be supported by candidates lying on a painted line
SHOW_CONFIDENCE = .35    # below this the ball is reported UNKNOWN
SWITCH_MARGIN = .15      # another candidate must beat the one on the predicted path by this much
FLAG_SECONDS = 1.5       # how long "seen at the same time as the ball" stays with a moving tracklet
ACCEPT = {'LOCKED': .45, 'OCCLUDED': .5, 'RECOVERING': .55}
HARD = ('SIZE', 'OUTSIDE_PITCH', 'OTHER_OBJECT', 'PLAYER_ATTACHED')
REASONS = {'FIELD_LINE': 'BALL REJECTED: FIELD LINE', 'STATIC': 'BALL REJECTED: STATIC FALSE POSITIVE',
           'PLAYER_ATTACHED': 'BALL REJECTED: PLAYER ATTACHED OBJECT', 'OTHER_OBJECT': 'BALL REJECTED: SEEN WHILE THE BALL WAS ELSEWHERE',
           'SIZE': 'BALL REJECTED: SIZE', 'OUTSIDE_PITCH': 'BALL REJECTED: OUTSIDE PITCH',
           'TRAJECTORY': 'BALL REJECTED: TRAJECTORY', 'LOW_SCORE': 'BALL REJECTED: LOW SCORE'}
OUTPUT_STATE = {'LOCKED': 'TRACKED', 'OCCLUDED': 'MISSING', 'RECOVERING': 'MISSING', 'SEARCHING': 'UNKNOWN'}


class Kalman:
    """Constant-velocity Kalman filter in image pixels; state [x, y, vx, vy] (px, px/s)."""

    def __init__(self, x, y, size_px, px_per_m):
        self.x = np.array([x, y, 0.0, 0.0])
        vel = 10*px_per_m
        self.P = np.diag([size_px**2, size_px**2, vel**2, vel**2]).astype(float)
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

    def inflate(self, pos_px=None, vel_px=None):
        """Floor the uncertainty: an unknown position while hidden, a possible kick for the velocity."""
        if pos_px:
            self.P[0, 0], self.P[1, 1] = max(self.P[0, 0], pos_px**2), max(self.P[1, 1], pos_px**2)
        if vel_px:
            self.P[2, 2], self.P[3, 3] = max(self.P[2, 2], vel_px**2), max(self.P[3, 3], vel_px**2)

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
class Contact:
    """A candidate's relation to the nearest person: where it lies inside their box (u, v in box
    units) and how far it is from their feet."""
    pid: object
    u: float
    v: float
    box: tuple
    foot_m: float
    inside: bool

    @property
    def upper_body(self):
        return self.inside and .05 <= self.v <= .72


@dataclass
class Candidate:
    index: int
    box: list            # pixels [x1, y1, x2, y2]
    det: float
    cx: float
    cy: float
    d: float             # diameter, px
    px_m: float          # pixels per metre at this depth
    stab: tuple          # camera-compensated normalized position
    pitch: tuple | None  # calibrated pitch position
    person: Contact | None
    segment: int = 0
    scores: dict = field(default_factory=dict)
    visual: float = 0.0  # appearance-only quality (no temporal evidence)
    penalty: float = 0.0
    reason: str = ''
    status: str = 'candidate'    # ball | candidate | rejected
    tracklet: object = None
    own_spot: bool = False
    err: float | None = None     # pixels from the ball prediction
    m2: float | None = None      # Mahalanobis distance from the ball prediction
    traj: float | None = None
    final: float | None = None


class Tracklet:
    """A short-lived chain of candidate detections (any white thing, not only the ball)."""

    def __init__(self, tid, kf, time, c, fps):
        self.id, self.kf, self.first, self.last, self.last_d = tid, kf, time, time, c.d
        self.hits = deque(maxlen=int(fps*3))
        self.unreliable = 0
        self.flag_until, self.flag_track = -math.inf, None

    def flagged(self, time):
        return time < self.flag_until

    def window(self, time, seconds):
        return [h for h in self.hits if time-h[0] <= seconds]


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
        return {'shape': .5, 'isolation': .5, 'line': 0.0, 'large': False, 'bulge': False, 'stretch': 1.0}
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
    stretch = 1.0
    if comp is not None:
        stretch = comp['major']/max(d, 3)
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
    if large:
        isolation = min(isolation, .3)  # part of a big white blob (kit, boards): not a ball on the grass
    return {'shape': round(shape, 3), 'isolation': round(isolation, 3), 'line': line, 'large': large,
            'bulge': bool(comp and comp['bulge']), 'stretch': round(stretch, 2)}


PAINTED_SPOTS = ((11/105, .5), (94/105, .5), (.5, .5))


def _painted_spot(p):
    """Within ~1 m of a penalty spot or the centre spot on the calibrated pitch."""
    return p is not None and any(math.hypot((p[0]-x)*105, (p[1]-y)*68) <= 1.0 for x, y in PAINTED_SPOTS)


def _segment_px_distance(px, py, a, b):
    ax, ay, bx, by = a[0], a[1], b[0], b[1]
    dx, dy = bx-ax, by-ay
    length2 = dx*dx+dy*dy
    t = 0.0 if length2 <= 1e-9 else clamp(((px-ax)*dx+(py-ay)*dy)/length2)
    return math.hypot(px-(ax+t*dx), py-(ay+t*dy))


class BallTracker:
    def __init__(self, width, height, fps):
        self.W, self.H, self.fps = width, height, max(1.0, fps)
        self.confirm = max(3, int(round(self.fps*.14)))   # consecutive plausible frames before (re)acquisition
        # The active ball.
        self.kf = None
        self.phase = 'SEARCHING'
        self.track = 0
        self.confidence = 0.0
        self.born = -math.inf
        self.last_seen = -math.inf
        self.last_time = None
        self.last_d = 8.0
        self.missing_frames = 0
        self.carrier = None          # {'id', 'offset', 'foot'}: the player the ball disappeared at
        self.lost_at = None          # {'time', 'x', 'y', 'track'} of the last lost ball, for re-acquisition
        self.last_confident = None   # (time, x, y) of the last observation with a high score
        self.history = deque(maxlen=int(self.fps*3))  # (time, x, y, observed, stab, pitch, px_m)
        self.trajectory = deque(maxlen=int(self.fps*2))  # (time, x, y, state) of the shown ball, for previews
        self.speeds = deque(maxlen=int(self.fps*.3))     # (time, m/s) recent speeds, for the acceleration estimate
        self.line_run = 0.0
        self.parked_since = None
        self.occluded_at = None
        self.own_tracklet = None     # the tracklet the ball's own detections continue
        # Everything else that was ever a candidate.
        self.tracklets = []
        self.next_tracklet = 1
        self.spots = []              # static-spot memory: [{'stab','pitch','r','first','last','hits','segment', ...}]
        self.rejections = Counter()
        self.since_log = Counter()
        self.last_log = -math.inf
        self.limits = {}
        self.events = []

    # ---------- helpers ----------
    def _px_per_m(self, y_norm, scale):
        height = scale(y_norm) if scale else None
        if not height or height <= 0:
            height = .09  # unknown: a typical broadcast player is ~9% of the frame height
        return height*self.H/PLAYER_HEIGHT

    def _event(self, time, title, lines):
        self.events.append({'time': round(float(time), 3), 'kind': 'ball', 'track': None, 'playerId': None,
                            'message': '\n'.join([title]+list(lines))})

    def _allow(self, key, time, every):
        last = self.limits.get(key)
        if last is not None and time-last < every and time >= last:
            return False
        self.limits[key] = time
        return True

    def _metres(self, a, b):
        """Distance in metres between two candidates / hits (pitch coordinates when both are calibrated,
        else camera-compensated image coordinates at their depth)."""
        if a.pitch is not None and b.pitch is not None:
            return math.hypot((a.pitch[0]-b.pitch[0])*105, (a.pitch[1]-b.pitch[1])*68)
        px_m = max(1e-3, (a.px_m+b.px_m)/2)
        return math.hypot((a.stab[0]-b.stab[0])*self.W, (a.stab[1]-b.stab[1])*self.H)/px_m

    def _spot(self, c):
        for s in self.spots:
            if c.pitch is not None and s['pitch'] is not None:
                if math.hypot((c.pitch[0]-s['pitch'][0])*105, (c.pitch[1]-s['pitch'][1])*68) <= STATIC_RADIUS:
                    return s
            elif s['segment'] == c.segment and math.hypot((c.stab[0]-s['stab'][0])*self.W, (c.stab[1]-s['stab'][1])*self.H)/c.px_m <= STATIC_RADIUS:
                return s
        return None

    @staticmethod
    def _static(spot):
        return spot is not None and (spot['false_positive'] or (spot['hits'] >= 6 and spot['last']-spot['first'] >= STATIC_SECONDS))

    def _nearest_person(self, nx, ny, people, px_m):
        best = None
        for pid, (bx, by, bw, bh) in people:
            if bw <= 0 or bh <= 0:
                continue
            foot_m = math.hypot((nx-bx-bw/2)*self.W, 2.0*(ny-by-bh)*self.H)/px_m  # image rows are foreshortened along the pitch
            inside = bx-.25*bw <= nx <= bx+1.25*bw and by-.05*bh <= ny <= by+1.05*bh  # arms swing beyond the torso box
            if inside or foot_m <= CONTACT_REACH:
                key = (0 if inside else 1, foot_m)
                if best is None or key < best[0]:
                    best = (key, Contact(pid, (nx-bx)/bw, (ny-by)/bh, (bx, by, bw, bh), foot_m, inside))
        return best[1] if best else None

    def _contact(self, x, y, people, px_m):
        """The player within reach of an image point (a touch, deflection or occlusion is possible)."""
        return self._nearest_person(x/self.W, y/self.H, people, px_m)

    # ---------- main ----------
    def step(self, frame, time, rows, M, reliable, cut, pitch, people, scale, stabilize, segment, to_pitch=None):
        """rows: (N, 6) ball detections in pixels. people: [(track id or None, normalized box)].
        scale(y) -> expected player height (normalized) at foot height y, or None.
        stabilize(x, y) -> camera-compensated normalized point. Returns the frame's ball dict and
        debug candidates."""
        dt = 1/self.fps if self.last_time is None else max(1e-3, time-self.last_time)
        self.last_time = time
        if cut:
            if self.kf is not None:
                self._lose(time, 'camera cut')
            self.tracklets = []
        for kf in ([self.kf] if self.kf is not None else [])+[t.kf for t in self.tracklets]:
            if reliable:
                kf.compensate(M)
            else:
                kf.P[:2, :2] += np.eye(2)*(self.last_d*3)**2
            kf.predict(dt)
        cands = self._candidates(frame, rows, pitch, people, scale, stabilize, segment, to_pitch)
        self._prelink(cands, time)
        for c in cands:
            self._classify(c, time, reliable)
        chosen = self._associate(cands, time, dt, people) if self.kf is not None else None
        if chosen is not None:
            self._update(chosen, time, dt)
        elif self.kf is not None:
            self._missing(time, dt, people)
        self._link(cands, time, reliable)
        if chosen is not None:
            self.own_tracklet = chosen.tracklet
        if self.kf is None:
            chosen = self._promote(time)
        self._remember_spots(cands, chosen, time, segment)
        self._mark_others(cands, chosen, time)
        for c in cands:
            if c.reason and c.status != 'ball':
                c.status = 'rejected'
                self.rejections[c.reason] += 1
                self.since_log[c.reason] += 1
        self._log(time, cands, chosen)
        out = self._output(time, chosen, to_pitch)
        if out['state'] in ('TRACKED', 'MISSING'):
            self.trajectory.append((round(time, 3), out['center'][0], out['center'][1], out['state']))
        return out, [self._debug(c) for c in cands[:12]]

    # ---------- per-frame evidence ----------
    def _candidates(self, frame, rows, pitch, people, scale, stabilize, segment, to_pitch):
        cands = []
        markings = []
        if pitch is not None and pitch.reliable:
            markings = [((m['a'][0]*self.W, m['a'][1]*self.H), (m['b'][0]*self.W, m['b'][1]*self.H)) for m in pitch.markings]
        for x1, y1, x2, y2, conf, _ in np.asarray(rows, np.float64).reshape(-1, 6):
            w, h = x2-x1, y2-y1
            if w <= 0 or h <= 0:
                continue
            cx, cy, d = (x1+x2)/2, (y1+y2)/2, math.sqrt(w*h)
            ground = (cy+d/2)/self.H
            px_m = self._px_per_m(ground, scale)
            ratio = d/max(BALL_DIAMETER*px_m, 1e-6)
            size = math.exp(-.5*(math.log(max(ratio, 1e-3))/math.log(1.9))**2)
            patch = analyse_patch(frame, cx, cy, d)
            nx, ny = cx/self.W, cy/self.H
            # Field location (the ball can fly above the pitch in the image, so only far-away is rejected).
            if pitch is not None and pitch.reliable and len(pitch.polygon) >= 3:
                out = signed_distance((nx, ny), pitch.polygon, pitch.aspect)
                fieldscore = 1.0 if out <= 0 else math.exp(-out/.04)
            else:
                out, fieldscore = 0.0, .7
            # On a fitted painted line of the pitch model: counts when the bright structure here is itself
            # stretched along something (a fit may run through a ball and a player's socks), and a ball
            # lying on a line makes a bulge.
            nearest = min((_segment_px_distance(cx, cy, a, b) for a, b in markings), default=math.inf)
            marking = clamp(1-(nearest-.5*d)/max(d, 2.0))*clamp((patch['stretch']-1.2)/1.3)
            line = max(patch['line'], marking*(.3 if patch['bulge'] else 1.0))
            c = Candidate(0, [x1, y1, x2, y2], float(conf), cx, cy, d, px_m, stabilize(nx, ny),
                          to_pitch((nx, ground)) if to_pitch is not None else None, self._nearest_person(nx, ny, people, px_m))
            c.segment = segment
            c.scores = {'det': round(clamp((c.det-.08)/.45), 3), 'size': round(size, 3), 'shape': patch['shape'],
                        'isolation': patch['isolation'], 'field': round(fieldscore, 3), 'line': round(line, 3)}
            c.visual = round(.4*c.scores['det']+.2*size+.15*patch['shape']+.15*patch['isolation']+.1*fieldscore, 3)
            if ratio > 3.0 or ratio < .3:
                c.reason = 'SIZE'
            elif out > .12:
                c.reason = 'OUTSIDE_PITCH'
            cands.append(c)
        cands.sort(key=lambda c: -c.visual)
        for k, c in enumerate(cands):
            c.index = k+1
        return cands

    def _link_gate(self, t, c, gap, time):
        """Pixels within which candidate c may continue tracklet t. A young tracklet (velocity unknown)
        reaches as far as a fast ball moves in two frames; an established one predicts its position;
        one flagged as another object never reaches far (it must not swallow the reappearing ball)."""
        if len(t.hits) < 3 and not t.flagged(time):
            return MAX_SPEED*min(gap, 2/self.fps)*c.px_m+2*max(c.d, t.last_d)
        speed = math.hypot(t.kf.x[2], t.kf.x[3])
        return max(.7*c.px_m, 2.5*max(c.d, t.last_d))+.5*speed*gap+2*c.d

    def _prelink(self, cands, time):
        """Which tracklet each candidate would continue (greedy, nearest prediction first): its
        temporal evidence is read before the ball decides."""
        pairs = []
        for c in cands:
            if c.reason:
                continue
            for t in self.tracklets:
                err = math.hypot(c.cx-t.kf.x[0], c.cy-t.kf.x[1])
                if err <= self._link_gate(t, c, time-t.last, time):
                    pairs.append((err, c, t))
        taken_c, taken_t = set(), set()
        for err, c, t in sorted(pairs, key=lambda p: p[0]):
            if id(c) in taken_c or t.id in taken_t:
                continue
            c.tracklet = t
            taken_c.add(id(c))
            taken_t.add(t.id)

    def _attachment(self, c, time):
        """How much the candidate behaves like something fixed to a player's body (wrist tape, gloves,
        boots, socks): the same place inside one person's box for many frames. A single frame inside
        the upper body is only a prior; the ball lives at the feet and passes through bodies in flight."""
        p, t = c.person, c.tracklet
        prior = .5 if p is not None and p.upper_body else 0.0
        if t is None and (p is None or p.pid is None):
            return prior, 0.0, None
        pid = p.pid if p is not None else None
        if pid is None and t is not None:
            # Outside every box this frame (an arm swing): still the person it was attached to just before.
            recent = [h.person.pid for _, h in t.window(time, .6) if h.person is not None and h.person.pid is not None]
            if len(recent) >= 3 and recent.count(recent[-1]) >= .8*len(recent):
                pid = recent[-1]
        if pid is None or t is None:
            return prior, 0.0, None
        hits = [(when, h.person) for when, h in t.hits if time-when <= 1.5 and h.person is not None and h.person.pid == pid]
        if p is not None and p.pid == pid:
            hits.append((time, p))
        if len(hits) < 3:
            return prior, 0.0, None
        us, vs = np.array([q.u for _, q in hits]), np.array([q.v for _, q in hits])
        span = time-hits[0][0]
        if vs.mean() <= .75:
            stable = us.std() <= .22 and vs.std() <= .12
            temporal = clamp((span-.2)/.6) if stable else 0.0
        else:
            stable = us.std() <= .2 and vs.std() <= .06   # at the feet the ball is plausible: needs long stability
            temporal = clamp((span-1.0)/1.0) if stable else 0.0
        return max(prior, temporal), round(float(span), 2), pid

    def _classify(self, c, time, reliable):
        t = c.tracklet
        attach, attached_for, attached_to = self._attachment(c, time)
        speed = moved = span = 0.0
        settled = 0.0
        if t is not None:
            hits = t.window(time, .6)
            if hits:
                first = hits[0][1]
                moved, span = self._metres(first, c), time-hits[0][0]
                speed = moved/span if span > 1e-3 else 0.0
            long = t.window(time, STATIC_SECONDS+.5)
            if len(long) >= 3:
                drift = max(self._metres(h[1], c) for h in long)
                age = time-long[0][0]
                if drift <= STATIC_RADIUS and t.unreliable <= .2*len(t.hits):
                    settled = clamp(age/STATIC_SECONDS)
        spot = self._spot(c)
        # The tracked ball came to rest here: its own spot (and its own stillness) is not a false positive.
        c.own_spot = self.kf is not None and ((spot is not None and spot['ball_track'] == self.track) or (spot is None and t is not None and t is self.own_tracklet))
        known = _painted_spot(c.pitch)
        static = 0.0 if c.own_spot else 1.0 if self._static(spot) or known or settled >= 1.0 else 0.0
        still = t is not None and len(t.hits) >= 3 and speed < 1.0
        flagged = (t is not None and t.flagged(time)) or (spot is not None and time < spot['flag_until'] and still)
        c.scores.update(attach=round(attach, 3), attachedFor=attached_for, attachedTo=attached_to, motion=round(clamp(speed/3), 3), speed=round(speed, 2),
                        static=static, settled=round(settled, 2), otherObject=bool(flagged),
                        person=None if c.person is None else {'track': c.person.pid, 'u': round(c.person.u, 2), 'v': round(c.person.v, 2), 'feet': round(c.person.foot_m, 2)},
                        tracklet=None if t is None else t.id)
        c.penalty = round(min(1.0, .5*c.scores['line']+.6*attach+.5*static+.3*settled*(1-static)), 3)
        if c.reason:
            return
        if c.scores['line'] >= .99:
            c.reason = 'FIELD_LINE'
        elif attach >= .7:
            c.reason = 'PLAYER_ATTACHED'
        elif flagged:
            c.reason = 'OTHER_OBJECT'
        elif static:
            c.reason = 'STATIC'

    # ---------- the active ball ----------
    def _associate(self, cands, time, dt, people):
        """The candidate that is the ball already being followed, or None."""
        kf, phase = self.kf, self.phase
        x, y = kf.x[0], kf.x[1]
        px_m = kf.px_per_m
        gap = max(dt, time-self.last_seen)
        speed_px = math.hypot(kf.x[2], kf.x[3])
        contact = self._contact(x, y, people, px_m)
        if contact is not None:
            kf.inflate(pos_px=.2*px_m, vel_px=.7*KICK_SPEED*px_m)   # a touch may change the velocity at once
        r_tight = max(GATE_FLOOR*px_m, 2*self.last_d)+.3*speed_px*dt
        if phase != 'LOCKED':
            r_tight = max(r_tight, (GATE_FLOOR+6*gap)*px_m)
        r_wide = MAX_SPEED*(min(gap, OCCLUDED_MAX) if contact is not None else gap)*px_m+2*self.last_d
        moving = speed_px/px_m >= 1.5
        threshold = max(ACCEPT[phase], .5 if contact is not None else 0)
        eligible = []
        for c in cands:
            c.err = err = math.hypot(c.cx-x, c.cy-y)
            c.m2 = m2 = kf.mahalanobis((c.cx, c.cy), max(1.0, c.d/4))
            if c.reason in HARD:
                continue
            near = err <= r_tight or m2 <= 9
            previous = c.tracklet.window(time, .3) if c.tracklet is not None else []  # the same thing seen just before
            if phase == 'RECOVERING':
                # Far from the last sighting: only a chain of consistent detections can be the ball again.
                ok = err <= r_wide and len(previous) >= max(2, self.confirm-1) and sum(h[1].visual for h in previous)/len(previous) >= .45
            else:
                ok = near or (contact is not None and err <= r_wide)
            if not ok:
                if c.visual >= .45 and not c.reason:
                    c.reason = 'TRAJECTORY'
                continue
            # One detection is never enough to move the ball off its predicted path or to bring it back
            # after a gap: outside the tight gate it must have been seen there the frame before; after a
            # gap it needs a confident detector or that previous sighting.
            on_body = c.person is not None and c.person.inside and c.person.v <= .72
            if (err > r_tight and (not previous or c.scores['motion'] < .3 or on_body)) or (self.missing_frames >= 3 and c.det < .3 and not previous):
                if not c.reason:
                    c.reason = 'LOW_SCORE'
                continue
            if c.reason in ('FIELD_LINE', 'STATIC') and not c.own_spot:
                # A moving track may cross a line or a spot, right on its predicted path, briefly.
                if not (phase == 'LOCKED' and moving and m2 <= 6 and self.line_run < LINE_RUN):
                    continue
            if c.person is not None and c.person.upper_body and m2 > 2:
                continue  # inside a body: only exactly where the ball was predicted (a flight through, a chest trap)
            traj = math.exp(-m2/18)*(.7 if err > r_tight else 1.0)
            prox = math.exp(-(err/max(r_tight, 1.0))**2)
            interaction = 1.0 if c.person is not None and not c.person.upper_body and c.person.foot_m <= .8 else 0.0
            penalty = c.penalty-.35*c.scores['line']-.4*c.scores['static'] if moving and m2 <= 6 else c.penalty  # the track explains the paint or spot under it
            c.traj = round(traj, 3)
            c.final = round(.5*traj+.1*prox+.4*c.visual+.05*interaction-max(0.0, penalty), 3)
            c.scores['interaction'] = interaction
            eligible.append(c)
        if not eligible:
            return None
        incumbent = min(eligible, key=lambda c: c.m2)
        best = max(eligible, key=lambda c: c.final)
        chosen = best
        if best is not incumbent and incumbent.final >= threshold-.1 and best.final < incumbent.final+SWITCH_MARGIN:
            chosen = incumbent  # switching resistance: stay on the predicted path unless clearly better
        need = threshold if chosen.err <= r_tight else max(threshold, .6)
        if chosen.final < need:
            for c in eligible:
                if not c.reason:
                    c.reason = 'LOW_SCORE'
            return None
        if chosen is not incumbent and self._allow('switch', time, 2):
            self._event(time, 'BALL CANDIDATE PREFERRED', [f'The candidate on the predicted path scored {incumbent.final:.2f}; candidate #{chosen.index} scored {chosen.final:.2f} and was taken instead.']
                        + self._lines(incumbent) + self._lines(chosen))
        return chosen

    def _update(self, chosen, time, dt):
        kf = self.kf
        chosen.status, chosen.reason = 'ball', ''
        kf.update((chosen.cx, chosen.cy), max(1.0, chosen.d/4))
        kf.px_per_m = chosen.px_m
        gap = time-self.last_seen
        if self.phase != 'LOCKED' and gap >= 3/self.fps-1e-6:
            self._event(time, 'BALL REACQUIRED', [f'Hidden for {gap:.2f} s ({self.missing_frames} frames), found again {chosen.err:.0f} px from the predicted position.'] + self._lines(chosen))
        if self.phase != 'LOCKED':
            self.confidence = max(self.confidence, min(chosen.final, .6))
        self.phase, self.last_seen, self.last_d, self.missing_frames, self.carrier = 'LOCKED', time, chosen.d, 0, None
        self.confidence = clamp(.7*self.confidence+.3*chosen.final, 0, .99)
        if chosen.final >= .6:
            self.last_confident = (time, chosen.cx, chosen.cy)
        on_paint = chosen.scores['line'] >= .99 or (chosen.scores['static'] and not chosen.own_spot)
        self.line_run = self.line_run+dt if on_paint else 0.0
        self.history.append((time, chosen.cx, chosen.cy, True, chosen.stab, chosen.pitch, chosen.px_m))
        self.speeds.append((time, math.hypot(kf.x[2], kf.x[3])/kf.px_per_m))
        self._check_parked(chosen, time)

    def _missing(self, time, dt, people):
        """No detection is the ball: it is hidden (behind or under a player), or gone."""
        kf = self.kf
        self.missing_frames += 1
        px_m = kf.px_per_m
        x, y = kf.x[0], kf.x[1]
        if self.phase == 'LOCKED':
            self.phase, self.occluded_at = 'OCCLUDED', time
            contact = self._contact(x, y, people, px_m)
            if contact is not None and contact.pid is not None and (contact.inside or not self._receding(contact, px_m)):  # this player hides it or has it
                bx, by, bw, bh = contact.box
                fx, fy = (bx+bw/2)*self.W, (by+bh)*self.H
                self.carrier = {'id': contact.pid, 'offset': (x-fx, y-fy), 'foot': (fx, fy)}
        if self.carrier is not None:
            box = next((b for pid, b in people if pid == self.carrier['id']), None)
            if box is not None:
                fx, fy = (box[0]+box[2]/2)*self.W, (box[1]+box[3])*self.H
                ox, oy = self.carrier['offset'][0]*.85, self.carrier['offset'][1]*.85
                vx, vy = (fx-self.carrier['foot'][0])/dt, (fy-self.carrier['foot'][1])/dt
                kf.x[:2] = (fx+ox, fy+oy)
                kf.x[2:] = (.6*kf.x[2]+.4*vx, .6*kf.x[3]+.4*vy)  # it moves with the player that has it
                kf.inflate(pos_px=.5*px_m, vel_px=.7*KICK_SPEED*px_m)
                self.carrier.update(offset=(ox, oy), foot=(fx, fy))
            else:
                self.carrier = None
        elif time-self.last_seen > .2:
            kf.x[2:] *= .9  # rolling friction while unseen
        gap = time-self.last_seen
        self.confidence *= math.exp(-dt/(2.5 if self.carrier is not None else .8))  # known to be at a player: it is still there
        if self.phase == 'OCCLUDED' and self.carrier is None and gap > OCCLUDED_MAX:
            self.phase = 'RECOVERING'
            if self._allow('recovering', time, 2):
                self._event(time, 'BALL RECOVERING', [f'Not seen for {gap:.2f} s near x = {x:.0f}, y = {y:.0f} px; the search area grows with time and a candidate must stay plausible for {self.confirm} frames.'])
        self.history.append((time, kf.x[0], kf.x[1], False, None, None, px_m))
        if gap > (MAX_CARRIED if self.carrier is not None else MAX_MISSING):
            self._lose(time, f'not seen for {gap:.1f} s')

    def _receding(self, contact, px_m):
        """The ball has been moving away from this player's feet over the last observed frames: kicked
        away or rolling past, not dribbled or trapped."""
        observed = [h for h in self.history if h[3]]
        if len(observed) < 4:
            return False
        fx, fy = (contact.box[0]+contact.box[2]/2)*self.W, (contact.box[1]+contact.box[3])*self.H
        metres = lambda h: math.hypot(h[1]-fx, 2.0*(h[2]-fy))/px_m
        return metres(observed[-1])-metres(observed[-4]) > .3

    def _check_parked(self, chosen, time):
        """A ball that stops is kept (a free kick is coming) with lower confidence; a track that never
        moved and rests on a spot known from before it started had latched onto paint or debris."""
        old = next((h for h in self.history if h[3] and time-h[0] >= 2.5), None)
        if old is None:
            return
        moved = self._metres(Candidate(0, [], 0, old[1], old[2], 1, old[6], old[4], old[5], None), chosen)
        if moved > STATIC_RADIUS:
            self.parked_since = None
            return
        if self.parked_since is None:
            self.parked_since = time
        self.confidence = min(self.confidence, .5)
        spot = self._spot(chosen)
        if spot is not None and (spot['first'] < self.born-.5 or spot['false_positive']):
            spot['false_positive'] = True
            self._lose(time, 'never moved, on a white spot that was there before the track started (static false positive)')
        elif time-self.parked_since > PARKED_MAX:
            self._lose(time, f'at rest for {time-self.parked_since:.0f} s; the track waits for motion')

    def _lose(self, time, why):
        if self.kf is not None:
            self.lost_at = {'time': time, 'x': float(self.kf.x[0]), 'y': float(self.kf.x[1]), 'track': self.track}
            self._event(time, 'BALL LOST', [f'Last position: x = {self.kf.x[0]:.0f}, y = {self.kf.x[1]:.0f} px', f'Reason: {why}',
                                             'The ball is UNKNOWN until a candidate shows consistent motion again.'])
        for t in self.tracklets:
            if t.flag_track == self.track:
                t.flag_until = -math.inf  # the lock may have been wrong: reconsider everything it excluded
        for s in self.spots:
            if s['flag_track'] == self.track:
                s['flag_until'] = -math.inf
        self.kf, self.phase, self.confidence, self.carrier = None, 'SEARCHING', 0.0, None
        self.line_run, self.parked_since, self.missing_frames = 0.0, None, 0
        self.history.clear()
        self.speeds.clear()
        self.trajectory.clear()

    # ---------- tracklets and (re)acquisition ----------
    def _link(self, cands, time, reliable):
        """Continue tracklets with this frame's candidates (the ball's own candidate included) and
        start new ones; stale tracklets expire."""
        for c in cands:
            if c.reason in ('SIZE', 'OUTSIDE_PITCH'):
                continue
            t = c.tracklet
            if t is None:
                t = Tracklet(self.next_tracklet, Kalman(c.cx, c.cy, max(c.d, 4), c.px_m), time, c, self.fps)
                self.next_tracklet += 1
                self.tracklets.append(t)
                c.tracklet = t
                c.scores['fit'] = 0.0
            else:
                c.scores['fit'] = round(t.kf.mahalanobis((c.cx, c.cy), max(1.0, c.d/4)), 1)  # consistency with the tracklet's motion
                t.kf.update((c.cx, c.cy), max(1.0, c.d/4))
                t.kf.px_per_m = c.px_m
            t.hits.append((time, c))
            t.last, t.last_d = time, c.d
            t.unreliable += not reliable
        self.tracklets = [t for t in self.tracklets if time-t.last <= .5][-40:]

    def _promote(self, time):
        """SEARCHING: a tracklet that stayed plausible for several frames and moved like a ball becomes
        the ball (with the old track id again when it reappears near where the ball was lost). A ball
        at rest is acquired once it moves: without motion nothing tells it from a painted spot."""
        ready = []
        for t in self.tracklets:
            if t.flagged(time) or t.hits[-1][0] < time-1e-6:
                continue
            hits = t.window(time, .6)
            if len(hits) < self.confirm:
                continue
            cs = [h[1] for h in hits]
            visual = sum(c.visual for c in cs)/len(cs)
            bad = sum(1 for c in cs if c.reason in ('FIELD_LINE', 'STATIC', 'PLAYER_ATTACHED', 'OTHER_OBJECT'))
            if visual < .45 or bad > len(cs)//4 or max(c.scores['attach'] for c in cs) >= .35:
                continue
            moved = self._metres(cs[0], cs[-1])
            fit = sum(c.scores.get('fit', 0.0) for c in cs[1:])/max(1, len(cs)-1)
            calibrated = all(c.pitch is not None for c in cs)
            if moved >= .5 and fit <= 16 and (calibrated or t.unreliable <= .2*len(t.hits)):
                ready.append((visual*len(hits), t, hits, moved))
        if not ready:
            return None
        _, t, hits, moved = max(ready, key=lambda r: r[0])
        cs = [h[1] for h in hits]
        last = cs[-1]
        lost = self.lost_at
        reacquired = lost is not None and time-lost['time'] <= 4 and \
            math.hypot(t.kf.x[0]-lost['x'], t.kf.x[1]-lost['y']) <= (2+.5*MAX_SPEED*(time-lost['time']))*last.px_m
        self.kf = t.kf
        self.tracklets.remove(t)
        if reacquired:
            self.track = lost['track']
        else:
            self.track += 1
            self.trajectory.clear()
        self.phase, self.born, self.last_seen, self.last_d, self.missing_frames, self.carrier = 'LOCKED', time, time, last.d, 0, None
        self.confidence = clamp(sum(c.visual for c in cs)/len(cs), 0, .99)
        self.line_run, self.parked_since, self.lost_at = 0.0, None, None
        self.last_log = time
        self.history.clear()
        for when, c in hits:
            self.history.append((when, c.cx, c.cy, True, c.stab, c.pitch, c.px_m))
        last.status, last.reason, last.traj, last.final = 'ball', '', 1.0, round(self.confidence, 3)
        title = 'BALL REACQUIRED' if reacquired else 'BALL TRACK SWITCH' if lost is not None and time-lost['time'] <= 4 else 'BALL ACQUIRED'
        self._event(time, title, [f'Track BALL-{self.track}: {len(cs)} consistent detections in {time-hits[0][0]:.2f} s, moved {moved:.1f} m',
                                  f'Final confidence: {self.confidence:.2f}'] + self._lines(last))
        return last

    # ---------- memories ----------
    def _remember_spots(self, cands, chosen, time, segment):
        """White things that do not move are remembered (penalty and centre spots, debris, stickers). A spot
        is anchored where it was first seen, so something that slides along (tape on a running player)
        never becomes a static spot."""
        moving = chosen is not None and self.kf is not None and math.hypot(*self.kf.x[2:]) > 1.5*self.kf.px_per_m
        for c in cands:
            if c.reason in ('SIZE', 'OUTSIDE_PITCH') or (c is chosen and moving):
                continue
            spot = self._spot(c)
            if spot is None:
                spot = {'stab': c.stab, 'pitch': c.pitch, 'first': time, 'last': time, 'hits': 1, 'segment': segment,
                        'false_positive': False, 'flag_until': -math.inf, 'flag_track': None, 'ball_track': self.track if c is chosen else None}
                self.spots.append(spot)
            else:
                if c is chosen:
                    spot['ball_track'] = self.track
                spot['hits'] += 1
                spot['last'] = time
        self.spots = [s for s in self.spots if time-s['last'] <= 60 and (s['segment'] == segment or s['pitch'] is not None)][-300:]

    def _mark_others(self, cands, chosen, time):
        """There is one ball: whatever is seen elsewhere while it is confidently tracked is not it."""
        if chosen is None or self.phase != 'LOCKED' or self.confidence < .5:
            return
        for c in cands:
            if c is chosen or math.hypot(c.cx-chosen.cx, c.cy-chosen.cy) <= 3*max(c.d, chosen.d):
                continue
            if c.tracklet is not None:
                c.tracklet.flag_until, c.tracklet.flag_track = time+FLAG_SECONDS, self.track
            spot = self._spot(c)
            if spot is not None:
                spot['flag_until'], spot['flag_track'] = time+10, self.track

    # ---------- logging and output ----------
    def _lines(self, c):
        s = c.scores
        lines = [f'BALL CANDIDATE #{c.index}', f'Detector confidence: {c.det:.2f}']
        if c.traj is not None:
            lines.append(f'Trajectory consistency: {c.traj:.2f}')
        if c.err is not None:
            lines.append(f'Distance from prediction: {c.err:.0f} px')
        lines += [f"Independent motion: {s['motion']:.2f} ({s['speed']:.1f} m/s)", f"Field line overlap: {s['line']:.2f}",
                  f"Player attachment score: {s['attach']:.2f}" + (f" (same place on track {s['attachedTo']} for {s['attachedFor']:.1f} s)" if s['attachedFor'] else ''),
                  f"Static object score: {s['static']:.2f}", f"Size / shape / isolation / on pitch: {s['size']:.2f} / {s['shape']:.2f} / {s['isolation']:.2f} / {s['field']:.2f}",
                  f'FINAL SCORE: {c.final if c.final is not None else c.visual:.2f}',
                  'ACCEPTED' if c.status == 'ball' else f'REJECTED: {c.reason}' if c.reason else 'CANDIDATE']
        return lines

    def _log(self, time, cands, chosen):
        if self.kf is None:
            return
        for c in cands:
            if c.status != 'ball' and c.det >= .6 and c.err is not None and c.reason in ('FIELD_LINE', 'STATIC', 'PLAYER_ATTACHED', 'OTHER_OBJECT') and \
                    self._allow('reject-'+c.reason, time, 3):
                self._event(time, 'BALL CANDIDATE REJECTED', self._lines(c))
        if time-self.last_log < 1:
            return
        self.last_log = time
        rejected = ', '.join(f'{k} {v}' for k, v in self.since_log.most_common()) or 'none'
        self.since_log.clear()
        lines = [f'Track BALL-{self.track}: {self.phase}, confidence {self.confidence:.2f}, speed {self._speed():.1f} m/s, age {time-self.born:.1f} s'
                 + (f', hidden for {time-self.last_seen:.2f} s' if self.phase != 'LOCKED' else '')]
        if chosen is not None:
            lines += self._lines(chosen)
        others = sorted((c for c in cands if c is not chosen and c.det >= .4), key=lambda c: -c.visual)[:2]
        for c in others:
            lines += self._lines(c)
        lines.append(f'Rejected candidates since the last update: {rejected}')
        self._event(time, 'BALL TRACK UPDATE', lines)

    def _speed(self):
        return math.hypot(self.kf.x[2], self.kf.x[3])/self.kf.px_per_m if self.kf is not None else 0.0

    def _output(self, time, chosen, to_pitch):
        if self.kf is None or self.confidence < (SHOW_CONFIDENCE if self.phase == 'LOCKED' else .25):
            return {'state': 'UNKNOWN', 'phase': self.phase, 'confidence': round(self.confidence, 3)}
        kf = self.kf
        x, y = kf.x[0], kf.x[1]
        d = chosen.d if chosen is not None else self.last_d
        box = [x-d/2, y-d/2, x+d/2, y+d/2] if chosen is None else chosen.box
        speed = self._speed()
        accel = 0.0
        if len(self.speeds) >= 2 and self.speeds[-1][0] > self.speeds[0][0]:
            accel = (self.speeds[-1][1]-self.speeds[0][1])/(self.speeds[-1][0]-self.speeds[0][0])
        out = {'state': OUTPUT_STATE[self.phase], 'phase': self.phase, 'observed': chosen is not None, 'confidence': round(self.confidence, 3),
               'track': self.track, 'age': round(time-self.born, 2), 'missingFrames': self.missing_frames,
               'box': [round(box[0]/self.W, 5), round(box[1]/self.H, 5), round((box[2]-box[0])/self.W, 5), round((box[3]-box[1])/self.H, 5)],
               'center': [round(x/self.W, 5), round(y/self.H, 5)],
               'predicted': [round((x+kf.x[2]/self.fps)/self.W, 5), round((y+kf.x[3]/self.fps)/self.H, 5)],
               'velocity': [round(kf.x[2]/self.W, 4), round(kf.x[3]/self.H, 4)], 'speed': round(speed, 2),
               'direction': round(math.degrees(math.atan2(kf.x[3], kf.x[2])), 1) if speed >= .5 else None,
               'acceleration': round(accel, 2), 'lastSeen': round(self.last_seen, 3)}
        if self.last_confident is not None:
            out['lastConfident'] = [round(self.last_confident[1]/self.W, 5), round(self.last_confident[2]/self.H, 5)]
        if self.phase != 'LOCKED':
            out['missingFor'] = round(time-self.last_seen, 2)
            if self.carrier is not None:
                out['nearTrack'] = self.carrier['id']
        if chosen is not None:
            out['detector'] = round(chosen.det, 3)
            out['score'] = chosen.final
        if to_pitch is not None:
            p = to_pitch((x/self.W, (y+d/2)/self.H))
            if p is not None:
                out['pitch'] = [round(p[0], 4), round(p[1], 4)]
        return out

    def _debug(self, c):
        s = {k: v for k, v in c.scores.items() if k not in ('person', 'tracklet', 'attachedFor', 'attachedTo')}
        s['penalty'] = c.penalty
        if c.traj is not None:
            s['trajectory'] = c.traj
        if c.err is not None:
            s['distance'] = round(c.err, 1)
        out = {'index': c.index, 'box': [round(c.box[0]/self.W, 5), round(c.box[1]/self.H, 5), round((c.box[2]-c.box[0])/self.W, 5), round((c.box[3]-c.box[1])/self.H, 5)],
               'det': round(c.det, 3), 'score': c.final if c.final is not None else c.visual, 'status': c.status, 'scores': s}
        if c.reason and c.status != 'ball':
            out['reason'] = c.reason
            out['text'] = REASONS[c.reason]
        return out
