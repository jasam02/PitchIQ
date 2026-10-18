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
    -> a detection never takes the ball over on its own. Beyond the gate only a CHAIN of consistent
       detections that the hidden ball could have reached in the unseen time is considered (the
       entry rule), and anything sitting on another player's lower body is a boot until its own
       motion relative to that player, or the ball's trajectory arriving there, says otherwise;
       every such decision is logged as a BALL SWITCH WARNING
    -> only when the ball is genuinely lost does a tracklet that stayed plausible for several frames
       become the (re)acquired ball

This is the first, causal pass. ball_path.py decodes the final trajectory offline over the whole
segment (every frame's candidates plus a "not visible" node, dynamic programming), which is what the
result, the 2D view and the metrics use; the states here drive the live preview and the search.

States: SEARCHING (no ball) -> LOCKED (observed this frame) -> OCCLUDED (short gap: predicted, kept
at the player that hides it) -> RECOVERING (longer gap: expanding search, confirmation required)
-> LOST -> SEARCHING. The output keeps one ball with one track id; while hidden the position is
predicted (state MISSING); without a confident ball it is UNKNOWN, never a guess.

Small scenes (`small_scene`, set by SoccerTracker before every step: amateur footage whose far players are tiny, or a
low-resolution source; never open on broadcast footage) add rules for what broadcast never shows, each off without the
gate: classical near-feet proposals where the detector is blind (soccer/ball_proposals.py), the hidden-ball gate at the
candidate's own depth, no lone detection taken through the grown uncertainty after a long hide, the entry rule measured
from the last sighting as well as the drifting prediction, a PROVISIONAL lock after a reacquisition beyond the mean
prediction (fresh motion history, no "not the ball" flags until it has moved and met a player), and a size prior
centred on the measured box size.
"""
import math
from collections import Counter, deque
from dataclasses import dataclass, field

import cv2
import numpy as np

from . import ball_proposals
from .field import DEFAULT_LENGTH, DEFAULT_WIDTH
from .geometry import PAINTED_SPOTS, clamp, signed_distance
from .pitch import nearest_boundary

BALL_DIAMETER = .22      # metres
PLAYER_HEIGHT = 1.8      # metres
MAX_SPEED = 40.0         # m/s: a hard shot; faster jumps are not the same ball
KICK_SPEED = 15.0        # m/s: velocity change allowed while a player is within reach (kick, deflection)
ACCEL_NOISE = 30.0       # m/s^2 process noise of the Kalman filter (bounces, friction, spin)
GATE_FLOOR = .45         # metres: the gate around the prediction is never tighter than this
CONTACT_REACH = 1.5      # metres from a player's feet within which a touch, deflection or occlusion can happen
OCCLUDED_MAX = .4        # seconds of a short gap with immediate reconnection near the prediction
MAX_MISSING = 2.5        # seconds a free ball is predicted before it is LOST (detectors miss a rolling ball for a while)
MAX_CARRIED = 3.0        # ... while it is at (hidden by) a player
PARKED_MAX = 8.0         # seconds a ball may rest before the track waits for motion again
STATIC_SECONDS = 2.0     # a candidate seen this long at one spot without moving is a static false positive
STATIC_RADIUS = .6       # metres
LINE_RUN = .4            # seconds a moving track may be supported by candidates lying on a painted line
SHOW_CONFIDENCE = .35    # below this the ball is reported UNKNOWN
SWITCH_MARGIN = .25      # another candidate must beat the one on the predicted path by this much
FLAG_SECONDS = 1.5       # how long "seen at the same time as the ball" stays with a moving tracklet
RELATIVE_FREE = .6       # metres a candidate must move relative to the player it sits on to count as free of them
ENTRY_SLOW = 8.0         # m/s: a hidden ball may have rolled this fast unseen without any further evidence
FAST_SPEED = 8.0         # m/s: from here the motion state is FAST rather than ROLLING
REST_WINDOW = .4         # seconds of sightings that show the ball at rest (STATIONARY) when none of them ...
REST_RADIUS = .25        # ... lies further than this (metres) from the oldest: no velocity is carried into the next prediction
ACCEPT = {'LOCKED': .45, 'OCCLUDED': .5, 'RECOVERING': .55}
HARD = ('SIZE', 'OUTSIDE_PITCH', 'OTHER_OBJECT', 'PLAYER_ATTACHED')
REASONS = {'FIELD_LINE': 'BALL REJECTED: FIELD LINE', 'STATIC': 'BALL REJECTED: STATIC FALSE POSITIVE',
           'PLAYER_ATTACHED': 'BALL REJECTED: PLAYER ATTACHED OBJECT', 'OTHER_OBJECT': 'BALL REJECTED: SEEN WHILE THE BALL WAS ELSEWHERE',
           'SIZE': 'BALL REJECTED: SIZE', 'OUTSIDE_PITCH': 'BALL REJECTED: OUTSIDE PITCH',
           'TRAJECTORY': 'BALL REJECTED: TRAJECTORY', 'LOW_SCORE': 'BALL REJECTED: LOW SCORE',
           'JUMP': 'BALL REJECTED: IMPOSSIBLE JUMP', 'NO_ENTRY': 'BALL REJECTED: NO TRAJECTORY INTO THIS PLAYER',
           'UNCONFIRMED': 'BALL REJECTED: NOT YET A CONSISTENT CHAIN'}
OUTPUT_STATE = {'LOCKED': 'TRACKED', 'OCCLUDED': 'MISSING', 'RECOVERING': 'MISSING', 'SEARCHING': 'UNKNOWN'}
SOURCES = ('detector', 'tiles', 'crop', 'proposal')   # the pass behind a candidate (row column 6), for the debug output only ...
PROPOSAL = 3              # ... except the classical proposals of a small scene, which are weak evidence
# Small scene only:
KEEP_DET = .4             # candidates ranked past the 12 written per frame are still written from this detector score
PROPOSE_AFTER = .2        # s unseen before the proposer looks for the ball (it also keeps looking while proposals alone carry the lock)
PROPOSE_LOST_S = 2.0      # s after a loss during which it still looks around the last sighting
PROPOSE_RADIUS = .25      # the proposal windows' half-size is capped at this share of the frame's longer side (amateur 3, 146-149 s:
                          # the ball stayed within ~450 px of its last sighting moved with the camera; ~7 ms per megapixel looked at)
PROPOSAL_CHAIN = 2        # earlier sightings within .3 s a proposal's tracklet needs before it can be the ball, even on the predicted path
MIN_BALL_PX = 3.0         # px: the size prior never expects a ball smaller than the smallest boxes the detector draws
SIZE_CENTRE = 1.4         # the size score peaks at the measured box / prior ratio of real balls (broadcast 1.36, Veo 1.54, amateur 1.43):
                          # centred on 1.0 it scored 3-4 px specks 1.0 and the real ball .5-.75 on amateur 3
SIZE_RATIO_MAX = 4.5      # hard SIZE bound on that ratio (3.0 otherwise): real 9-11.5 px balls at amateur 3 152.3-153.4 s (detector up
                          # to .74, people ~20 px tall) were rejected at 3.3-4.2
FLAG_STRONGER = .2        # a lock never rules out a detection whose score exceeds the mean of its own over the last .6 s by more than
                          # this (amateur 3, 152.97 s: a lock on a player's sock, detector .2-.5, ruled out the real ball at .6-.8)
PROVEN_SECONDS = .5       # a provisional lock proves itself after this long, by moving PROVEN_METRES (camera-compensated image and
PROVEN_METRES = 1.0       # pitch distance both) and by passing within CONTACT_REACH of a player: on amateur 3 the causal camera
                          # compensation drifted 1.6-3.4 m/s on a still speck (149-152 s), so motion alone proves nothing there


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
    relative: float | None = None   # motion relative to the player it is on (0: moves with them, 1: free)
    required: float | None = None   # m/s the hidden ball would have needed to reach this candidate's chain
    entry: str = ''                 # why the chain cannot be the ball's continuation ('' when it can)
    far: bool = False               # beyond the gate around the prediction
    source: int = 0                 # the pass that produced it (SOURCES)
    gate: tuple | None = None       # (tight, path, wide) gate radii in pixels it was judged with
    flagged_by: int | None = None   # the tracklet of the lock that ruled it "seen while the ball was elsewhere" (diagnostics)
    continues: bool = False         # small scene: it continues the ball's own tracklet, which predicted it (_associate)


class Tracklet:
    """A short-lived chain of candidate detections (any white thing, not only the ball)."""

    def __init__(self, tid, kf, time, c, fps):
        self.id, self.kf, self.first, self.last, self.last_d = tid, kf, time, time, c.d
        self.hits = deque(maxlen=int(fps*3))
        self.unreliable = 0
        self.flag_until, self.flag_track, self.flag_by = -math.inf, None, None

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


def _painted_spot(p):
    """Within ~1 m of a penalty spot or the centre spot on the calibrated pitch."""
    return p is not None and any(math.hypot((p[0]-x)*DEFAULT_LENGTH, (p[1]-y)*DEFAULT_WIDTH) <= 1.0 for x, y in PAINTED_SPOTS)


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
        self.unflagged = False       # the current hidden spell has released the lock's "not the ball" flags
        self.small_scene = False     # the small-scene gate, set by SoccerTracker before every step (every "small scene" rule needs it)
        self.anchor = None           # small scene: time of a provisional reacquisition, until the lock proves itself (PROVEN_*)
        self.anchor_touched = False  # ... a player has been within reach of the lock since then
        self.seen_at = None          # (x, y, px per metre) of the last observation, moved with the camera since
        self.lock_source = 0         # the source of the ball's latest observation
        self.gates = {}              # this frame's gate radii at the prediction's depth (diagnostics)
        self.frame_debug = None      # this frame's diagnostics (gates, the lock's tracklet, provisional, proposals): SoccerTracker's ballDebug

    @property
    def scan_locked(self):
        """The ball is followed well enough for the pipeline to scan the whole frame (the ball tiles) only as a safeguard:
        LOCKED, not on a provisional lock, and seen by a detector pass (while classical proposals alone carry the lock the
        detector is blind on the ball, and the tiles are how it finds it again). Both exceptions exist only in a small
        scene: elsewhere this is phase == 'LOCKED'."""
        return self.phase == 'LOCKED' and self.anchor is None and self.lock_source != PROPOSAL

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
            return math.hypot((a.pitch[0]-b.pitch[0])*DEFAULT_LENGTH, (a.pitch[1]-b.pitch[1])*DEFAULT_WIDTH)
        px_m = max(1e-3, (a.px_m+b.px_m)/2)
        return math.hypot((a.stab[0]-b.stab[0])*self.W, (a.stab[1]-b.stab[1])*self.H)/px_m

    def _metres_image(self, a, b):
        """Metres between two candidates in camera-compensated image coordinates, whatever the calibration says."""
        px_m = max(1e-3, (a.px_m+b.px_m)/2)
        return math.hypot((a.stab[0]-b.stab[0])*self.W, (a.stab[1]-b.stab[1])*self.H)/px_m

    def _metres_since(self, entry, c):
        """Metres between an observed history entry (time, x, y, observed, stab, pitch, px_m) and a candidate."""
        return self._metres(Candidate(0, [], 0, entry[1], entry[2], 1, entry[6], entry[4], entry[5], None), c)

    def _resting(self, time):
        """The ball has been observed standing still: at least three sightings over the last REST_WINDOW
        seconds, spanning a quarter second, none further than REST_RADIUS metres from the oldest of them.
        Confirmed stillness, never the absence of evidence (an unseen ball is OCCLUDED, not STATIONARY)."""
        recent = [h for h in self.history if h[3] and time-h[0] <= REST_WINDOW]
        if len(recent) < 3 or recent[-1][0]-recent[0][0] < .25:
            return False
        oldest = recent[0]
        return all(self._metres_since(oldest, Candidate(0, [], 0, h[1], h[2], 1, h[6], h[4], h[5], None)) <= REST_RADIUS for h in recent[1:])

    def _spot(self, c):
        for s in self.spots:
            if c.pitch is not None and s['pitch'] is not None:
                if math.hypot((c.pitch[0]-s['pitch'][0])*DEFAULT_LENGTH, (c.pitch[1]-s['pitch'][1])*DEFAULT_WIDTH) <= STATIC_RADIUS:
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
    def step(self, frame, time, rows, M, reliable, cut, pitch, people, scale, stabilize, segment, to_pitch=None, calibration=None):
        """rows: (N, 6) ball detections in pixels [x1, y1, x2, y2, score, class], or (N, 7) with the pass that produced
        each (column 6: SOURCES; diagnostic only, a missing column means the whole-frame detector, and so does a debug
        candidate without a 'source').
        people: [(track id or None, normalized box)].
        scale(y) -> expected player height (normalized) at foot height y, or None.
        stabilize(x, y) -> camera-compensated normalized point. calibration: confidence of the pitch
        registration this frame (None: trusted whenever to_pitch is given): penalties that need a pitch
        position (the painted spots) are only applied under a trusted registration, so a wrong or
        missing one cannot reject the ball. Returns the frame's ball dict and debug candidates."""
        dt = 1/self.fps if self.last_time is None else max(1e-3, time-self.last_time)
        self.last_time = time
        self.calibration = clamp(float(calibration)) if calibration is not None else (1.0 if to_pitch is not None else 0.0)
        rows = np.asarray(rows, np.float64)
        columns = 7 if rows.ndim == 2 and rows.shape[1] == 7 else 6
        rows = rows.reshape(-1, columns)
        sources = rows[:, 6].astype(int) if columns == 7 else np.zeros(len(rows), int)
        rows, self.gates = rows[:, :6], {}
        if not self.small_scene:
            # The gate closed: the lock is judged like any other from here on, and no classical proposal carries it.
            self.anchor = None
            if self.lock_source == PROPOSAL:
                self.lock_source = 0
        if cut:
            if self.kf is not None:
                self._lose(time, 'camera cut')
            self.tracklets = []
            self.seen_at = None
        for kf in ([self.kf] if self.kf is not None else [])+[t.kf for t in self.tracklets]:
            if reliable:
                kf.compensate(M)
            else:
                kf.P[:2, :2] += np.eye(2)*(self.last_d*3)**2
            kf.predict(dt)
        if self.seen_at is not None and reliable and not cut:
            p = np.asarray(M, np.float64) @ np.array([self.seen_at[0], self.seen_at[1], 1.0])
            self.seen_at = (float(p[0]/p[2]), float(p[1]/p[2]), self.seen_at[2])
        proposals = self._propose(frame, rows, people, scale, time) if self.small_scene else []
        if proposals:
            rows = np.vstack([rows, [[x1, y1, x2, y2, ball_proposals.PROPOSAL_DET, 0.0] for x1, y1, x2, y2, _ in proposals]])
            sources = np.r_[sources, np.full(len(proposals), PROPOSAL)]
        cands = self._candidates(frame, rows, sources, pitch, people, scale, stabilize, segment, to_pitch)
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
        if self.anchor is not None and chosen is not None and self._contact(chosen.cx, chosen.cy, people, chosen.px_m) is not None:
            self.anchor_touched = True
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
        self.frame_debug = self._frame_debug(len(proposals))
        shown = cands[:12]
        if self.small_scene:
            # Appearance ranks a large near ball (size score .5) below specks (1.0): a strong one is never left unwritten.
            shown = shown+[c for c in cands[12:] if c.status == 'ball' or c.det >= KEEP_DET]
        return out, [self._debug(c) for c in shown]

    def _frame_debug(self, proposals):
        """What decided this frame, for the debug output (no decision reads it): the gate radii at the prediction's depth,
        the tracklet the ball's own sightings continue, and in a small scene the provisional state and the proposals."""
        out = {'phase': self.phase, **self.gates}
        if self.kf is not None and self.own_tracklet is not None:
            out['lockTracklet'] = self.own_tracklet.id
        if self.small_scene:
            out['smallScene'] = True
            if self.anchor is not None:
                out['provisional'] = {'since': round(self.anchor, 3), 'touched': self.anchor_touched}
            if proposals:
                out['proposals'] = proposals
        return out

    # ---------- per-frame evidence ----------
    def _candidates(self, frame, rows, sources, pitch, people, scale, stabilize, segment, to_pitch):
        cands = []
        markings = []
        if pitch is not None and pitch.reliable:
            markings = [((m['a'][0]*self.W, m['a'][1]*self.H), (m['b'][0]*self.W, m['b'][1]*self.H)) for m in pitch.markings]
        for (x1, y1, x2, y2, conf, _), source in zip(rows, sources):
            w, h = x2-x1, y2-y1
            if w <= 0 or h <= 0:
                continue
            cx, cy, d = (x1+x2)/2, (y1+y2)/2, math.sqrt(w*h)
            ground = (cy+d/2)/self.H
            px_m = self._px_per_m(ground, scale)
            if self.small_scene:
                # Small scene: the prior is floored at the smallest box the detector draws and peaks at the measured box size.
                ratio = d/max(BALL_DIAMETER*px_m, MIN_BALL_PX)
                size = math.exp(-.5*(math.log(max(ratio/SIZE_CENTRE, 1e-3))/math.log(1.9))**2)
                oversized = ratio > SIZE_RATIO_MAX
            else:
                ratio = d/max(BALL_DIAMETER*px_m, 1e-6)
                size = math.exp(-.5*(math.log(max(ratio, 1e-3))/math.log(1.9))**2)
                oversized = ratio > 3.0
            patch = analyse_patch(frame, cx, cy, d)
            nx, ny = cx/self.W, cy/self.H
            # Field location (the ball can fly above the pitch in the image, so only far-away is rejected).
            # The playable region is trusted as far as its own confidence: a LOW region cannot reject, and
            # beyond a converging end (a vanishing point, not a measured pitch end) nothing is outside.
            trust = clamp(float(getattr(pitch, 'confidence', 1.0))) if pitch is not None else 0.0
            if pitch is not None and pitch.reliable and len(pitch.polygon) >= 3:
                out = signed_distance((nx, ny), pitch.polygon, pitch.aspect)
                if out > 0 and getattr(pitch, 'vanishing', None):
                    edge = nearest_boundary(pitch, (nx, ny))
                    if edge is not None and edge['side'] == pitch.vanishing:
                        out = 0.0
                fieldscore = 1.0 if out <= 0 else max(.7*(1-trust), math.exp(-out/.04))
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
            c.segment, c.source = segment, int(source)
            c.scores = {'det': round(clamp((c.det-.08)/.45), 3), 'size': round(size, 3), 'shape': patch['shape'],
                        'isolation': patch['isolation'], 'field': round(fieldscore, 3), 'line': round(line, 3)}
            c.visual = round(.4*c.scores['det']+.2*size+.15*patch['shape']+.15*patch['isolation']+.1*fieldscore, 3)
            if oversized or ratio < .3:
                c.reason = 'SIZE'
            elif out > .12 and trust >= .5:
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
        relative, sightings = self._relative_motion(c, time)
        c.relative = None if relative is None else round(clamp(relative/RELATIVE_FREE), 3)
        if c.person is not None and c.person.inside and c.person.v > .72 and sightings >= 3 and c.relative is not None and c.relative < .3:
            attach = max(attach, .55)   # a young chain glued to one player's lower body: boot-like, short of a hard rejection
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
                # A thing fixed to the field drifts in pitch coordinates while a carried calibration drifts; it does not
                # drift in the camera-compensated image. Either distance saying "still" is enough.
                drift = max(min(self._metres(h[1], c), self._metres_image(h[1], c)) for h in long)
                age = time-long[0][0]
                if drift <= STATIC_RADIUS and t.unreliable <= .2*len(t.hits):
                    settled = clamp(age/STATIC_SECONDS)
        spot = self._spot(c)
        # The tracked ball came to rest here: its own spot (and its own stillness) is not a false positive.
        c.own_spot = self.kf is not None and ((spot is not None and spot['ball_track'] == self.track) or (spot is None and t is not None and t is self.own_tracklet))
        known = _painted_spot(c.pitch) and self.calibration >= .5   # a painted spot is only where a trusted registration says
        static = 0.0 if c.own_spot else 1.0 if self._static(spot) or known or settled >= 1.0 else 0.0
        still = t is not None and len(t.hits) >= 3 and speed < 1.0
        flagged = (t is not None and t.flagged(time)) or (spot is not None and time < spot['flag_until'] and still)
        if flagged:
            c.flagged_by = t.flag_by if t is not None and t.flagged(time) else spot['flag_by']
        c.scores.update(attach=round(attach, 3), attachedFor=attached_for, attachedTo=attached_to, motion=round(clamp(speed/3), 3), speed=round(speed, 2),
                        relative=c.relative,
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

    def _relative_motion(self, c, time):
        """How far (metres) the candidate has moved relative to the player it is on over the last 0.6 s,
        and from how many sightings: a boot or sock moves with the player and stays near zero, a dribbled
        or passing ball does not."""
        p, t = c.person, c.tracklet
        if p is None or p.pid is None or t is None:
            return None, 0
        hits = [h for _, h in t.window(time, .6) if h.person is not None and h.person.pid == p.pid]
        if len(hits) < 2:
            return None, len(hits)
        bw, bh = p.box[2]*self.W, p.box[3]*self.H
        moved = max(math.hypot((h.person.u-p.u)*bw, (h.person.v-p.v)*bh) for h in hits)/max(c.px_m, 1e-3)
        return moved, len(hits)

    def _entry(self, c, previous, time):
        """Why a candidate beyond the gate cannot be the hidden ball's continuation (None when it can): it
        must already be a chain of consistent sightings, and the ball must have been able to get from
        where it is believed to be to where that chain started, in the time it was unseen. A fast
        unseen flight is plausible only if the chain still moves fast and does not sit at a foot."""
        if len(previous) < max(2, self.confirm-1) or sum(h[1].visual for h in previous)/len(previous) < .45:
            return 'UNCONFIRMED'
        first_t, first = previous[0]
        elapsed = max(2/self.fps, first_t-self.last_seen)
        c.required = math.hypot(first.cx-self.kf.x[0], first.cy-self.kf.x[1])/max(c.px_m, 1e-3)/elapsed
        if self.small_scene and self.seen_at is not None:
            # Small scene: the prediction drifts while the ball is hidden (friction, a carrier box that slid off, a dribbler the
            # detector lost); where the ball was last seen, moved with the camera, bounds the speed it needed as well.
            c.required = min(c.required, math.hypot(first.cx-self.seen_at[0], first.cy-self.seen_at[1])/max(c.px_m, 1e-3)/elapsed)
        lower = c.person is not None and c.person.inside and c.person.v > .72
        if c.required > MAX_SPEED:
            return 'JUMP'
        if c.required > ENTRY_SLOW and (c.scores['speed'] < .5*c.required or lower):
            return 'JUMP'
        return None

    def _warn_switch(self, c, time, gap, decision):
        kf = self.kf
        required = c.required if c.required is not None else (c.err or 0.0)/max(c.px_m, 1e-3)/max(gap, 1/self.fps)
        static = c.scores.get('static', 0.0)
        settled = c.scores.get('settled', 0.0)
        spot = _painted_spot(c.pitch) and self.calibration >= .5
        marking = 'HIGH' if static >= 1.0 or spot else 'MEDIUM' if settled >= .5 else 'LOW'
        lines = [f'Old position: x = {kf.x[0]:.0f}, y = {kf.x[1]:.0f} px ({self.phase}, hidden for {gap:.2f} s' + (f', at track {self.carrier["id"]}' if self.carrier else '') + ')',
                 f'New position: x = {c.cx:.0f}, y = {c.cy:.0f} px ({c.err:.0f} px from the prediction)',
                 f'Elapsed time: {max(gap, 1/self.fps):.3f} s', f'Required velocity: {required:.1f} m/s',
                 f'Physical plausibility: {c.traj if c.traj is not None else 0.0:.2f} (trajectory consistency)', f'Previous trajectory confidence: {self.confidence:.2f}',
                 f"Near penalty spot: {'YES' if spot else 'NO'}", f'Static field marking evidence: {marking} (static {static:.2f}, settled {settled:.2f})',
                 f'Candidate detector confidence: {c.det:.2f}', f"Player attachment score: {c.scores['attach']:.2f}"
                 + (f" (on track {c.person.pid}, motion relative to them {c.relative if c.relative is not None else 0:.2f})" if c.person is not None and c.person.pid is not None else ''),
                 f'Decision: {decision}' + (f' ({c.entry})' if c.entry else '') + (' / FIELD_MARKING' if decision != 'ACCEPTED' and marking == 'HIGH' else '')]
        self._event(time, 'BALL SWITCH WARNING', lines)

    def _unflag(self):
        """What a lock excluded as "seen elsewhere while the ball was tracked" is only as good as that
        lock: once the ball has been hidden for long, or lost, those objects are reconsidered (a wrong
        lock on a boot must not keep excluding the real ball)."""
        for t in self.tracklets:
            if t.flag_track == self.track:
                t.flag_until = -math.inf
        for s in self.spots:
            if s['flag_track'] == self.track:
                s['flag_until'] = -math.inf

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
        r_path = r_tight   # the mean prediction itself, without the uncertainty that grows while hidden
        if phase != 'LOCKED':
            r_tight = max(r_tight, (GATE_FLOOR+6*gap)*px_m)
        r_wide = MAX_SPEED*(min(gap, OCCLUDED_MAX) if contact is not None else gap)*px_m+2*self.last_d
        self.gates = {'pxm': round(px_m, 2), 'tight': round(r_tight, 1), 'path': round(r_path, 1), 'wide': round(r_wide, 1)}
        moving = speed_px/px_m >= 1.5
        threshold = max(ACCEPT[phase], .5 if contact is not None else 0)
        eligible = []
        warned = False
        for c in cands:
            c.err = err = math.hypot(c.cx-x, c.cy-y)
            c.m2 = m2 = kf.mahalanobis((c.cx, c.cy), max(1.0, c.d/4))
            if self.small_scene:
                # Small scene: the gate in metres is converted to pixels at the candidate's depth when it lies further away
                # than the last sighting (a near-field ball's gate was 2-3x too wide at the far side: amateur 3, 147.33 s).
                k = min(1.0, c.px_m/max(px_m, 1e-6))
                c.gate = (r_tight*k, r_path*k, r_wide*k)
            else:
                c.gate = (r_tight, r_path, r_wide)
            tight, path, wide = c.gate
            if c.reason in HARD:
                continue
            previous = c.tracklet.window(time, .3) if c.tracklet is not None else []  # the same thing seen just before
            # Consistent with the prediction: within the gate, or statistically close AND not far beyond it
            # (the uncertainty grows without bound while the ball is hidden; it must not make everything "near").
            # Small scene: after a long hide the grown uncertainty never makes a lone detection consistent.
            lone = self.small_scene and gap > OCCLUDED_MAX and not previous
            # Small scene: the ball's own tracklet, seen just before and predicting this sighting by its own motion, is
            # the ball whatever the ball's filter expected: it follows a kick by a player the detector does not see
            # (amateur 3, 148.6 s: the dribbler, 60-120 px tall, was found only by a 640 px pass).
            t = c.tracklet
            if self.small_scene and t is not None and t is self.own_tracklet and time-t.last <= OCCLUDED_MAX:
                own = t.kf.mahalanobis((c.cx, c.cy), max(1.0, c.d/4))
                if own <= 9:
                    c.continues, c.m2, m2 = True, min(m2, own), min(m2, own)
            near = err <= tight or (m2 <= 9 and err <= 1.5*tight and not lone) or c.continues
            lower = c.person is not None and c.person.inside and c.person.v > .72
            other = lower and c.person.pid is not None and (self.carrier is None or c.person.pid != self.carrier['id'])
            if other and err > path and not c.continues and (c.relative is None or c.relative < .4):
                # On another player's lower body, with neither the ball's trajectory arriving there nor any
                # motion of its own relative to that player: a boot or sock until proven otherwise. The
                # uncertainty that grows while the ball is hidden never makes such a thing "consistent".
                c.entry = 'NO_ENTRY'
                if not c.reason:
                    c.reason = 'NO_ENTRY'
                if c.visual >= .45 and not warned and self._allow('switch-warning', time, 1.0):
                    warned = True
                    self._warn_switch(c, time, gap, 'REJECTED')
                continue
            if c.source == PROPOSAL and len(previous) < PROPOSAL_CHAIN:
                if not c.reason:
                    c.reason = 'UNCONFIRMED'   # a classical proposal is the ball only as part of a chain, even on the path
                continue
            if not near:
                c.far = True
                if not (err <= wide and (contact is not None or phase == 'RECOVERING')):
                    if c.visual >= .45 and not c.reason:
                        c.reason = 'TRAJECTORY'
                    continue
                # Beyond the gate only a chain of consistent sightings that the hidden ball could have
                # reached is the ball: one detection never is, whatever its confidence.
                verdict = self._entry(c, previous, time)
                if verdict is not None:
                    c.entry = verdict
                    if not c.reason:
                        c.reason = verdict
                    if c.visual >= .45 and not warned and self._allow('switch-warning', time, 1.0):
                        warned = True
                        self._warn_switch(c, time, gap, 'REJECTED')
                    continue
            elif self.missing_frames >= 3 and c.det < .3 and not previous:
                if not c.reason:
                    c.reason = 'LOW_SCORE'
                continue
            if c.reason in ('FIELD_LINE', 'STATIC') and not c.own_spot:
                # A moving track may cross a line or a spot, right on its predicted path, briefly.
                if not (phase == 'LOCKED' and moving and m2 <= 6 and self.line_run < LINE_RUN):
                    continue
            if c.person is not None and c.person.upper_body and m2 > 2:
                continue  # inside a body: only exactly where the ball was predicted (a flight through, a chest trap)
            traj = math.exp(-m2/18)*(.7 if err > tight else 1.0)
            prox = math.exp(-(err/max(tight, 1.0))**2)
            interaction = 1.0 if c.person is not None and not c.person.upper_body and c.person.foot_m <= .8 else 0.0
            penalty = c.penalty-.35*c.scores['line']-.4*c.scores['static'] if moving and m2 <= 6 else c.penalty  # the track explains the paint or spot under it
            c.traj = round(traj, 3)
            c.final = round(.5*traj+.1*prox+.4*c.visual+.05*interaction-max(0.0, penalty), 3)
            c.scores['interaction'] = interaction
            eligible.append(c)
        if not eligible:
            return None
        # A classical proposal never beats a real detector candidate on the same frame: proposals are judged only when no
        # detector candidate is good enough (they exist in small scenes alone).
        detected = [c for c in eligible if c.source != PROPOSAL]
        pools = [detected, eligible] if detected and len(detected) < len(eligible) else [eligible]
        for pool in pools:
            chosen, incumbent = self._select(pool, threshold)
            if chosen is not None:
                break
        if chosen is None:
            for c in eligible:
                if not c.reason:
                    c.reason = 'LOW_SCORE'
            return None
        if chosen is not incumbent and self._allow('switch', time, 2):
            self._event(time, 'BALL CANDIDATE PREFERRED', [f'The candidate on the predicted path scored {incumbent.final:.2f}; candidate #{chosen.index} scored {chosen.final:.2f} and was taken instead.']
                        + self._lines(incumbent) + self._lines(chosen))
        return chosen

    @staticmethod
    def _select(eligible, threshold):
        """(the chosen candidate, or None when it scores too low; the one on the predicted path) among eligible ones."""
        incumbent = min(eligible, key=lambda c: c.m2)
        best = max(eligible, key=lambda c: c.final)
        chosen = best
        if best is not incumbent and incumbent.final >= threshold-.1 and best.final < incumbent.final+SWITCH_MARGIN:
            chosen = incumbent  # switching resistance: stay on the predicted path unless clearly better
        need = threshold if chosen.err <= chosen.gate[0] else max(threshold, .6)
        return (chosen if chosen.final >= need else None), incumbent

    def _update(self, chosen, time, dt):
        gap = time-self.last_seen
        provisional = self.small_scene and gap > OCCLUDED_MAX and chosen.err > chosen.gate[1] and not chosen.continues
        if provisional:
            self._provisional(chosen, time)
        elif chosen.continues and chosen.err > chosen.gate[1]:
            # Its own tracklet's motion is the better estimate: the ball's filter takes it over (velocity included).
            self.kf.x, self.kf.P = chosen.tracklet.kf.x.copy(), chosen.tracklet.kf.P.copy()
        kf = self.kf
        chosen.status, chosen.reason = 'ball', ''
        kf.update((chosen.cx, chosen.cy), max(1.0, chosen.d/4))
        kf.px_per_m = chosen.px_m
        if self.phase != 'LOCKED' and gap >= 3/self.fps-1e-6:
            self._event(time, 'BALL REACQUIRED', [f'Hidden for {gap:.2f} s ({self.missing_frames} frames), found again {chosen.err:.0f} px from the predicted position.']
                        + (['PROVISIONAL LOCK: taken beyond the predicted path after a long hide; it rules nothing else out until it has moved '
                            f'{PROVEN_METRES:.0f} m and met a player'] if provisional else []) + self._lines(chosen))
        if chosen.far:
            self._warn_switch(chosen, time, gap, 'ACCEPTED')
        self.unflagged = False
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
        self.seen_at, self.lock_source = (chosen.cx, chosen.cy, chosen.px_m), chosen.source
        self._check_parked(chosen, time)
        if self._resting(time):
            kf.x[2:] = 0.0   # observed at rest: no stale velocity is carried on (a kick is caught by the uncertainty a player within reach adds)

    def _provisional(self, chosen, time):
        """Small scene: a reacquisition beyond the mean prediction after a long hide may be anything the grown gate let in
        (amateur 3: a painted arc at 147.33 s, then a still speck at 148.9 s). The lock starts afresh from this sighting,
        with no velocity, history or rest carried over from the object it replaces, and stays PROVISIONAL until it proves
        itself (_proven): until then it rules nothing out as "seen while the ball was elsewhere", it is judged parked
        against its own sightings, and the pipeline keeps scanning the whole frame for the ball (scan_locked)."""
        self.anchor, self.anchor_touched = time, False
        self.kf = Kalman(chosen.cx, chosen.cy, max(chosen.d, 4), chosen.px_m)
        self.history.clear()
        self.speeds.clear()
        self.line_run, self.parked_since = 0.0, None

    def _proven(self, chosen, time):
        """A provisional lock has proven itself: PROVEN_SECONDS since it was taken, PROVEN_METRES moved since then in both the
        camera-compensated image and (when calibrated) on the pitch, and a player within reach of it on the way."""
        observed = [h for h in self.history if h[3]]
        if time-self.anchor < PROVEN_SECONDS or not self.anchor_touched or len(observed) < 3:
            return False
        h = observed[0]
        first = Candidate(0, [], 0, h[1], h[2], 1, h[6], h[4], h[5], None)
        return min(self._metres(first, chosen), self._metres_image(first, chosen)) >= PROVEN_METRES

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
        if gap > OCCLUDED_MAX and not self.unflagged:
            self._unflag()
            self.unflagged = True
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
        moved and rests on a spot known from before it started had latched onto paint or debris (for a
        provisional lock: from before it was taken)."""
        old = next((h for h in self.history if h[3] and time-h[0] >= 2.5), None)
        if old is None:
            return
        if self._metres_since(old, chosen) > STATIC_RADIUS:
            self.parked_since = None
            return
        if self.parked_since is None:
            self.parked_since = time
        self.confidence = min(self.confidence, .5)
        spot = self._spot(chosen)
        started = self.anchor if self.anchor is not None else self.born
        if spot is not None and (spot['first'] < started-.5 or spot['false_positive']):
            spot['false_positive'] = True
            self._lose(time, 'never moved, on a white spot that was there before the track started (static false positive)')
        elif time-self.parked_since > PARKED_MAX:
            self._lose(time, f'at rest for {time-self.parked_since:.0f} s; the track waits for motion')

    def _lose(self, time, why):
        if self.kf is not None:
            self.lost_at = {'time': time, 'x': float(self.kf.x[0]), 'y': float(self.kf.x[1]), 'track': self.track}
            self._event(time, 'BALL LOST', [f'Last position: x = {self.kf.x[0]:.0f}, y = {self.kf.x[1]:.0f} px', f'Reason: {why}',
                                             'The ball is UNKNOWN until a candidate shows consistent motion again.'])
        self._unflag()  # the lock may have been wrong: reconsider everything it excluded
        self.kf, self.phase, self.confidence, self.carrier = None, 'SEARCHING', 0.0, None
        self.line_run, self.parked_since, self.missing_frames, self.anchor = 0.0, None, 0, None
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
        at rest is acquired once it moves: without motion nothing tells it from a painted spot. A tracklet
        whose sighting this frame is a classical proposal (small scenes only) is taken only when no ready
        tracklet was seen by the detector this frame: a proposal never beats a detector candidate."""
        ready = []
        for t in self.tracklets:
            if t.flagged(time) or t.hits[-1][0] < time-1e-6:
                continue
            hits = t.window(time, .8)
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
            # Every detection inside one person's lower box moves with that person: boots, socks, or a ball
            # glued to the feet. Only a long history (the attachment test) tells them apart; a dribbled
            # ball pops in and out of the box and qualifies at once.
            on_feet = all(c.person is not None and c.person.inside and c.person.v >= .7 for c in cs) and \
                len({c.person.pid for c in cs}) == 1 and (time-t.first < 1.5 or len(hits) < 2*self.confirm)
            if moved >= .5 and fit <= 16 and not on_feet and (calibrated or t.unreliable <= .2*len(t.hits)):
                ready.append((visual*len(hits), t, hits, moved))
        if not ready:
            return None
        detected = [r for r in ready if r[2][-1][1].source != PROPOSAL]   # without proposals: all of them, in order
        _, t, hits, moved = max(detected or ready, key=lambda r: r[0])
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
        self.line_run, self.parked_since, self.lost_at, self.anchor = 0.0, None, None, None
        self.seen_at, self.lock_source = (last.cx, last.cy, last.px_m), last.source
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
                        'false_positive': False, 'flag_until': -math.inf, 'flag_track': None, 'flag_by': None, 'ball_track': self.track if c is chosen else None}
                self.spots.append(spot)
            else:
                if c is chosen:
                    spot['ball_track'] = self.track
                spot['hits'] += 1
                spot['last'] = time
        self.spots = [s for s in self.spots if time-s['last'] <= 60 and (s['segment'] == segment or s['pitch'] is not None)][-300:]

    def _mark_others(self, cands, chosen, time):
        """There is one ball: whatever is seen elsewhere while it is confidently tracked is not it. Only a
        mature lock may say so (old enough, confident, and it has moved like a ball), and never about
        things close to the ball: a wrong lock on a boot must not disqualify the real ball next to it.
        Small scene: neither may a provisional lock that has not proven itself, nor a lock whose sighting this frame is
        a classical proposal (weak evidence never rules out the detector's candidates), and no lock rules out a detection
        clearly stronger than its own recent ones (FLAG_STRONGER)."""
        if chosen is None or self.phase != 'LOCKED' or self.confidence < .6 or time-self.born < .5:
            return
        if self.small_scene:
            if chosen.source == PROPOSAL:
                return
            if self.anchor is not None:
                if not self._proven(chosen, time):
                    return
                self.anchor = None   # proven: an ordinary lock from now on
        observed = [h for h in self.history if h[3]]
        if len(observed) < 3 or self._metres_since(observed[0], chosen) < 1.0:
            return
        strength = math.inf
        if self.small_scene and chosen.tracklet is not None:
            recent = [h.det for _, h in chosen.tracklet.window(time, .6)]   # this frame's sighting included (linked above)
            strength = sum(recent)/len(recent)+FLAG_STRONGER
        by = chosen.tracklet.id if chosen.tracklet is not None else None
        for c in cands:
            if c is chosen or math.hypot(c.cx-chosen.cx, c.cy-chosen.cy) <= max(3*max(c.d, chosen.d), 1.5*chosen.px_m):
                continue
            if c.det > strength:
                continue
            if c.tracklet is not None:
                c.tracklet.flag_until, c.tracklet.flag_track, c.tracklet.flag_by = time+FLAG_SECONDS, self.track, by
            spot = self._spot(c)
            if spot is not None:
                spot['flag_until'], spot['flag_track'], spot['flag_by'] = time+10, self.track, by

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
                  f"Static object score: {s['static']:.2f}", f"Size / shape / isolation / on pitch: {s['size']:.2f} / {s['shape']:.2f} / {s['isolation']:.2f} / {s['field']:.2f}"]
        if c.person is not None and c.person.pid is not None:
            lines.append(f"On track {c.person.pid} ({'lower body' if c.person.v > .72 else 'upper body' if c.person.inside else 'within reach'})"
                         + (f", motion relative to them: {c.relative:.2f}" if c.relative is not None else ''))
        if c.required is not None:
            lines.append(f'Required speed from the last known position: {c.required:.1f} m/s')
        lines += [f'FINAL SCORE: {c.final if c.final is not None else c.visual:.2f}',
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

    def _propose(self, frame, rows, people, scale, time):
        """Small scene: classical ball proposals (soccer/ball_proposals.py) where the hidden ball can be, once it has
        been unseen for PROPOSE_AFTER (or while proposals alone carry the lock): around its last sighting, moved with
        the camera, and around the prediction, as far as the hidden-ball gate reaches (capped at PROPOSE_RADIUS), and
        for PROPOSE_LOST_S after a loss around the last sighting. Nothing is proposed where the detector already
        reported something."""
        if self.seen_at is None:
            return []
        gap = time-self.last_seen
        if self.kf is not None:
            if gap < PROPOSE_AFTER and self.lock_source != PROPOSAL:
                return []
            px_m = self.kf.px_per_m
            centres = [(self.seen_at[0], self.seen_at[1]), (self.kf.x[0], self.kf.x[1])]
        elif self.lost_at is not None and time-self.lost_at['time'] <= PROPOSE_LOST_S:
            px_m = self.seen_at[2]
            centres = [(self.seen_at[0], self.seen_at[1])]
        else:
            return []
        half = min((GATE_FLOOR+6*gap)*px_m+3*self.last_d, PROPOSE_RADIUS*max(self.W, self.H))
        windows = [(x-half, y-half, x+half, y+half) for x, y in centres]
        if len(windows) == 2 and abs(centres[0][0]-centres[1][0]) < 2*half and abs(centres[0][1]-centres[1][1]) < 2*half:
            (ax0, ay0, ax1, ay1), (bx0, by0, bx1, by1) = windows   # overlapping: one pass over their union
            windows = [(min(ax0, bx0), min(ay0, by0), max(ax1, bx1), max(ay1, by1))]
        expected = lambda y: BALL_DIAMETER*self._px_per_m(y/self.H, scale)
        return ball_proposals.find(frame, windows, expected, people, [tuple(r[:4]) for r in rows])

    def search_window(self, dt=None):
        """Where to look for the ball at the original resolution: a square (x0, y0, x1, y1 pixels) around
        the predicted position while the ball is followed or recently lost, sized for the gate (wider the
        longer it has been missing), or None. The whole-frame detector sees a far ball as two or three
        pixels; a crop at full resolution does not."""
        if self.kf is None or self.phase == 'SEARCHING':
            return None
        dt = 1/self.fps if dt is None else dt
        x, y = self.kf.x[0]+self.kf.x[2]*dt, self.kf.x[1]+self.kf.x[3]*dt
        px_m = self.kf.px_per_m
        gap = max(dt, (self.last_time-self.last_seen) if self.last_time is not None else 0.0)
        reach = (GATE_FLOOR+6*gap)*px_m+.5*math.hypot(self.kf.x[2], self.kf.x[3])*dt+3*self.last_d
        half = float(min(max(reach, 24*self.last_d, 96.0), .35*max(self.W, self.H)))
        x0, y0 = int(max(0, x-half)), int(max(0, y-half))
        x1, y1 = int(min(self.W, x+half)), int(min(self.H, y+half))
        if x1-x0 < 32 or y1-y0 < 32:
            return None
        return x0, y0, x1, y1

    def _output(self, time, chosen, to_pitch):
        if self.kf is None or self.confidence < (SHOW_CONFIDENCE if self.phase == 'LOCKED' else .25):
            return {'state': 'UNKNOWN', 'phase': self.phase, 'confidence': round(self.confidence, 3), 'motion': 'UNKNOWN', 'origin': 'UNKNOWN'}
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
        # Detector confidence is not identity confidence: the track-level identity, the appearance of
        # the observation and its trajectory consistency are reported apart.
        out['identity'] = round(self.confidence, 3)
        out['visual'] = chosen.visual if chosen is not None else None
        out['trajectory'] = chosen.traj if chosen is not None else None
        # Where this position comes from, and the motion state the evidence supports: a hidden ball is OCCLUDED,
        # never assumed still; STATIONARY needs sightings that show it at rest.
        out['origin'] = 'OBSERVATION' if chosen is not None else 'PREDICTION'
        out['motion'] = 'OCCLUDED' if self.phase in ('OCCLUDED', 'RECOVERING') else 'STATIONARY' if self._resting(time) else 'ROLLING' if speed < FAST_SPEED else 'FAST'
        if self.phase != 'LOCKED':
            out['missingFor'] = round(time-self.last_seen, 2)
            if self.carrier is not None:
                out['nearTrack'] = self.carrier['id']
        if self.phase == 'RECOVERING':
            out['candidatePaths'] = sum(1 for t in self.tracklets if not t.flagged(time) and len(t.hits) >= 2)
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
               'det': round(c.det, 3), 'score': c.final if c.final is not None else c.visual, 'visual': c.visual, 'status': c.status, 'scores': s,
               'pxm': round(c.px_m, 2), 'tracklet': c.tracklet.id if c.tracklet is not None else None}
        if c.person is not None and c.person.pid is not None:
            out['player'] = {'track': c.person.pid, 'u': round(c.person.u, 2), 'v': round(c.person.v, 2), 'lower': bool(c.person.inside and c.person.v > .72)}
        if c.entry:
            out['entry'] = c.entry
        if c.required is not None:
            out['required'] = round(c.required, 1)
        if c.reason and c.status != 'ball':
            out['reason'] = c.reason
            out['text'] = REASONS[c.reason]
        # Diagnostics only (nothing reads them back): the pass behind the candidate unless it is the whole-frame detector
        # (most candidates: a missing 'source' means the detector, which keeps match-length results small), the lock
        # that ruled it "seen elsewhere", and in a small scene the gate radii at its own depth.
        if c.source != 0:
            out['source'] = SOURCES[c.source] if 0 <= c.source < len(SOURCES) else str(c.source)
        if c.flagged_by is not None:
            out['flaggedBy'] = c.flagged_by
        if self.small_scene and c.gate is not None:
            out['gate'] = [round(r, 1) for r in c.gate]
        return out
