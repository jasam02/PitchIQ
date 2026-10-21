"""A Metrica game as a PitchIQ-shaped result, under (a) full tracking or (b) a simulated broadcast camera, so the
possession engine can be evaluated on both against tools/possession_truth.py's truth.

    import sys; sys.path.insert(0, '<repo>/tools')
    import possession_truth as mt, possession_sim as cs
    g = mt.load_game(1, TRACKING, EVENTS)                    # tools/possession_truth.py
    sim = cs.simulate(g, mode='camera', fps=30, seed=0)     # the whole game as arrays (about 15 s)
    result = sim.result(600.0, 900.0)                        # a PitchIQ result.json dict for 10:00-15:00
    code, k = mt.truth_at(g, [f['time'] for f in result['frames']])   # truth per frame: 0 dead, 1 Home (A), 2 Away (B)
    cs.calibration(sim, g)                                   # the ball audit's measures on the simulation (AUDIT)

Run under python -I -B (the Metrica files are untrusted data). frames[i].time is the tracking clock in seconds and
frames[i].frame = round(time*fps); team A is Home and team B Away, ids GK-A, A-01 ... by tracking column
(tools/team_shape_bench.py); tactical.goalEnds from the goalkeepers in period 1. result.simulation says what was
simulated; an engine must not read it.

(a) mode='full': every player on the pitch at his exact position (pitchConfidence .9, pitchSigma .3 m), the ball
TRACKED / OBSERVATION wherever Metrica tracks it and UNKNOWN where it does not (Metrica leaves the ball untracked
while it is out of play, tools/possession_truth.py), motion from its speed, nearTrack from the image rule below, boxes from
the camera of (b) (they may lie outside the image).

(b) mode='camera': a main camera on the halfway line, CAM_BACK m behind the near touchline and CAM_HEIGHT m up,
pans and zooms so that a look-at point that follows the ball with a LAG s first-order lag stays in the middle of the
image with VIEW_WIDTH m across it (about 30 m wide at the near touchline, 90 m at the far one). People whose feet
project inside the 1920 x 1080 image are visible (a small dropout rate aside); boxes come from the projection (1.8 m
tall, .4 of the height wide). Positions are what PitchIQ computes: the feet's pixel with AR(1) detection noise,
back-projected through a registration that is the true camera with AR(1) pan / tilt / zoom errors. So the error is
shared by everything near one spot and grows with depth, and an airborne ball (a height is simulated for long fast
flights) lands too far from the camera, as the audit found. The ball is detected frame by frame by a two-state
Markov chain whose persistence depends on where it is in the image (hidden in a player's lower box, free, fast,
airborne, off screen). The sightings are assembled as soccer/ball_path.py's final path is: gaps up to MAX_SKIP_S
(MAX_STILL_S for a ball that stayed put) are bridged (MISSING, INTERPOLATION, position interpolated), longer gaps are
UNKNOWN, a stretch with fewer than MIN_RUN sightings is dropped. Speeds are taken over +-SPEED_WINDOW s of sightings.
DETECT and the noise constants were fitted on game 1 to the ball audit's broadcast numbers
(AUDIT, at the end); calibration() measures the same on a simulation.
Where Metrica blanks a dead ball, dead_fill='hold' (default) lets the ball roll on for DEAD_ROLL_S and shows it
still at the restart spot (where Metrica's ball reappears, with the taker) for DEAD_HOLD_S before the restart, as a
broadcast shows a ball placed for a free kick or held for a throw-in; the rest of the stoppage is UNKNOWN. That makes a
long UNKNOWN spell a dead-ball cue the real footage does not give as reliably (real broadcasts keep the ball in view
through short stoppages and cut to replays in long ones): dead_fill='visible' shows the ball at the restart spot for
the whole stoppage after the roll, dead_fill='none' leaves all of it UNKNOWN; an engine should hold up under all three.
noise='independent' replaces the camera's position errors by fp_common.py's: an AR(1) error per player with a sigma
drawn from U(.5, 2) m and a 1 s correlation, independent of everyone else (a harder stress test than the camera).

Not simulated: identity switches and team errors, extra people (one referee distractor aside), replays and cuts,
registration collapses (BC42 190-215 s), painted marks taken for the ball, small-scene footage.

Game 3 (Metrica's EPTS sample: g = mt.load_game(3, DIR, DIR), with DIR holding Sample_Game_3_metadata.xml,
_tracking.txt and _events.json) goes through this module unchanged. tools/possession_truth.py gives it the CSV
games' arrays, and corrects its tracking's Player6 / Player13 swap over the second half before anything here sees
it. Its 17 Home and 18 Away columns (13 substitutions) become 35 roster slots with their firstSeen / left, with ids
GK-A, A-01 ... by column as in games 1-2, and tactical.goalEnds comes from period 1's goalkeepers (in game 3 Home
defends the right goal first). Checked on game 3 (2026-10-20, no engine run): the full and camera results have the
same top-level, video, frame and people fields as game 1's, and the same ball fields per ball state. Every
wonById / lostById of the truth is a simulated id. frames[].frame = round(time*fps) holds. A whole game in camera mode
takes about 9 s and has 172512 frames at 30 fps. No constant was refitted on game 3.
"""
import math

import numpy as np
from scipy.signal import lfilter

import possession_truth as mt

LENGTH, WIDTH = mt.LENGTH, mt.WIDTH
IMG_W, IMG_H = 1920, 1080
CAM_BACK = 25.0          # m behind the near touchline
CAM_HEIGHT = 25.0        # m above the pitch
VIEW_WIDTH = 60.0        # m across the image at the look-at point (the task's "~60 m view")
LAG = 1.0                # s: the look-at point's first-order lag behind the ball
MAX_PAN = 20.0           # m/s: the look-at point moves no faster
LOOK_Y = (18.0, 50.0)    # the look-at point's y stays in this band (m from the far touchline)
PLAYER_H = 1.8           # m: box height
BOX_ASPECT = .4          # box width / height
BALL_D = .22             # m
# Registration error (the estimated camera minus the true one): AR(1) with REG_TAU s correlation.
REG_TAU = 2.0
PAN_SIGMA = .25          # degrees
TILT_SIGMA = .22         # degrees
ZOOM_SIGMA = .025        # relative focal length
FOOT_SIGMA = 5.0         # px: AR(1) foot-point detection noise per person (FOOT_TAU s)
FOOT_TAU = .3
BALL_PX_SIGMA = .4       # px: white noise on the ball's box
SIGMA_REG = .2           # m added to the reported pitchSigma (3 px of foot-point error) for the registration's own
                         # uncertainty (PitchIQ adds the registration disagreement): people p10/50/90 .35/.44/.57 m
                         # against BC60's .2/.4/.9
DROP_P, DROP_KEEP = .01, .8   # a visible person drops out with this probability per frame and stays out with DROP_KEEP
INDEPENDENT_SIGMA = (.5, 2.0)  # noise='independent': per-player sigma range (m), 1 s AR(1) (fp_common.py)
# Ball detection: (P(seen | seen on the previous frame), P(missed | missed on the previous frame)) per image context.
DETECT = {'feet': (.921, .983), 'free': (.928, .39), 'fast': (.779, .548), 'air': (.736, .482)}
CONTEXTS = ('feet', 'free', 'fast', 'air')
MISS_DECAY = 140.0       # frames: see detect_ball
FEET_BOX = (-.1, 1.1, .6, 1.08)   # u0, u1, v0, v1 in box units: ball centre in a player's lower box (ball_path.lower_body)
AIR_Z = 1.5              # m: a ball higher than this is 'air'
FEET_HOLD_S = .2         # s: a ball in a player's lower box this recently is still at his feet (context 'feet')
WEAK_SHARE = .04         # share of sightings labelled WEAK_DETECTION (audit: .021-.026 of frames)
NEAR_P = .9              # a frame missed in a player's lower box carries that player's track as nearTrack (the first
                         # pass saw the ball go into him); otherwise none
# soccer/ball_path.py's final path.
OCCLUDED_MAX = .4        # s: soccer/ball.py's first pass calls a gap OCCLUDED this long, or while a player hides the ball
MAX_SKIP_S = 1.2
MAX_STILL_S = 3.0
STILL_SPEED = 1.0
TRAVEL_SPEED = 8.0
MIN_RUN = 4
SPEED_WINDOW = .15       # s either side: sightings this close give a sighting's speed (ball_path NEIGHBOUR_S .3 s)
# Simulated ball height (Metrica has none): a flight is a run of true speed >= FLIGHT_SPEED lasting FLIGHT_MIN_S; one
# longer than LOFT_MIN_M is lofted with probability LOFT_P, apex g T^2 / 8 capped at LOFT_MAX m.
FLIGHT_SPEED, FLIGHT_MIN_S, LOFT_MIN_M, LOFT_P, LOFT_MAX = 10.0, .5, 25.0, .2, 5.0
DEAD_ROLL_S = 1.0
DEAD_HOLD_S = 4.0
CONFIDENCE = (.78, .06)  # frame registration confidence: mean, sd (AR(1), clipped to [.5, .95]); BC60 p10/50/90 .63/.79/.89
REF_OFFSET = (-9.0, 7.0) # the referee distractor stands about this far from the ball (2 s lag)


# ----------------------------------------------------------------------------------------------- camera geometry

def rotation(pan, tilt):
    """World-to-camera rows (right, down, forward) for a camera looking towards -y, panned by `pan` (rad, positive
    towards +x) and tilted down by `tilt` (rad). Arrays (T,) -> (T, 3, 3)."""
    f = np.stack([np.sin(pan)*np.cos(tilt), -np.cos(pan)*np.cos(tilt), -np.sin(tilt)], -1)
    r = np.cross(np.array([0.0, 0.0, 1.0]), f)
    r /= np.linalg.norm(r, axis=-1, keepdims=True)
    d = np.cross(r, f)
    return np.stack([r, d, f], -2)


def project(C, R, foc, P):
    """Pixels (T, N, 2) and depth (T, N) of world points P (T, N, 3) for a camera at C (3,), R (T, 3, 3), focal (T,)."""
    q = np.einsum('tij,tnj->tni', R, P-C)
    z = q[..., 2]
    with np.errstate(divide='ignore', invalid='ignore'):
        u = IMG_W/2+foc[:, None]*q[..., 0]/z
        v = IMG_H/2+foc[:, None]*q[..., 1]/z
    return np.stack([u, v], -1), z


def ground(C, R, foc, uv):
    """Pitch-plane (x, y) of pixels uv (T, N, 2); NaN for rays that miss it."""
    ray = np.stack([(uv[..., 0]-IMG_W/2)/foc[:, None], (uv[..., 1]-IMG_H/2)/foc[:, None], np.ones(uv.shape[:-1])], -1)
    w = np.einsum('tji,tnj->tni', R, ray)
    with np.errstate(divide='ignore', invalid='ignore'):
        t = -C[2]/w[..., 2]
    xy = C[:2]+t[..., None]*w[..., :2]
    return np.where((t > 0)[..., None], xy, np.nan)


def _metres_per_px(C, R, foc, uv):
    """Ground metres per pixel along x and y at the pixels uv (T, N, 2), at least 5 cm."""
    base = ground(C, R, foc, uv)
    du = ground(C, R, foc, uv+np.array([1.0, 0.0]))
    dv = ground(C, R, foc, uv+np.array([0.0, 1.0]))
    sx = np.maximum(np.abs(du[..., 0]-base[..., 0]), np.abs(dv[..., 0]-base[..., 0]))
    sy = np.maximum(np.abs(du[..., 1]-base[..., 1]), np.abs(dv[..., 1]-base[..., 1]))
    return np.stack([np.maximum(sx, .05), np.maximum(sy, .05)], -1)


def _ar1(shape, tau, fps, sigma, rng):
    """Stationary AR(1) series along axis 0: correlation time tau (s), marginal sd sigma (scalar or broadcastable)."""
    phi = math.exp(-1.0/(tau*fps))
    e = rng.normal(size=shape)*math.sqrt(1-phi*phi)
    e[0] = rng.normal(size=shape[1:])
    return lfilter([1.0], [1.0, -phi], e, axis=0)*sigma


def _resample(times, values, t):
    """Linear interpolation of values (T, ...) sampled at times onto t; NaN where a neighbour is NaN."""
    x = np.interp(t, times, np.arange(len(times)))
    k0 = np.clip(np.floor(x).astype(np.int64), 0, len(times)-2)
    w = (x-k0).reshape((-1,)+(1,)*(values.ndim-1))
    return (1-w)*values[k0]+w*values[k0+1]


def _runs(mask):
    """[(first, last)] index runs where mask is True."""
    m = np.asarray(mask, bool)
    if not m.any():
        return []
    d = np.diff(np.concatenate([[0], m.astype(np.int8), [0]]))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)-1))


def _motion(speed):
    if speed is None or not np.isfinite(speed):
        return 'UNKNOWN'
    return 'STATIONARY' if speed < STILL_SPEED else 'ROLLING' if speed < TRAVEL_SPEED else 'FAST'


# ----------------------------------------------------------------------------------------------- the result

class Sim:
    """Per-frame arrays of one simulated game. result(start, end) builds the PitchIQ dict for a window."""

    def result(self, start=None, end=None, name=None):
        t = self.t
        lo = 0 if start is None else int(np.searchsorted(t, start-1e-9))
        hi = len(t) if end is None else int(np.searchsorted(t, end+1e-9, side='right'))
        frames = [self._frame(i) for i in range(lo, hi)]
        s = self.state[lo:hi]
        n = max(1, len(s))
        return {'version': 3, 'video': {'name': name or f'metrica_game{self.game}_{self.mode}.sim', 'width': IMG_W, 'height': IMG_H,
                                        'fps': float(self.fps), 'frames': len(frames), 'duration': len(frames)/self.fps},
                'frames': frames, 'match': self.match, 'roster': self.roster,
                'tactical': {'goalEnds': self.ends, 'pitchSize': {'length': LENGTH, 'width': WIDTH, 'source': 'metrica'}, 'calibratedShare': 1.0},
                'summary': {'ball': {'trackedShare': round(float((s == 1).sum()/n), 3), 'missingShare': round(float((s == 2).sum()/n), 3),
                                     'unknownShare': round(float((s == 0).sum()/n), 3), 'phases': {'PATH': 1.0}}},
                'limitations': ['Simulated from Metrica tracking by camera_sim.py: no identity or team errors, no replays, no registration collapse.'],
                'simulation': {'game': self.game, 'mode': self.mode, 'fps': self.fps, 'seed': self.seed, 'params': self.params,
                               'window': [float(t[lo]), float(t[hi-1])] if hi > lo else None}}

    def _frame(self, i):
        t = float(self.t[i])
        conf = round(float(self.conf[i]), 3)
        people = []
        for j in np.flatnonzero(self.visible[i]):
            x, y = self.est[i, j]
            if not (np.isfinite(x) and np.isfinite(y)):
                continue
            pid, team = self.ids[j], self.team[j]
            role = 'GOALKEEPER' if pid.startswith('GK') else 'PLAYER'
            people.append({'time': t, 'track': int(self.track[i, j]), 'box': [round(float(v), 5) for v in self.box[i, j]], 'score': .85, 'cls': 'player',
                           'zone': 'inside', 'evidence': 'observed', 'id': pid, 'display': pid, 'team': team, 'role': role,
                           'label': f'{role}_TEAM_{team}', 'state': 'confirmed', 'identityConfidence': .85, 'roleConfidence': 1.0,
                           'pitch': [round(float(x/LENGTH), 5), round(float(y/WIDTH), 5)], 'pitchConfidence': conf, 'pitchSource': 'observed',
                           'pitchSigma': [round(float(v), 2) for v in self.sigma[i, j]]})
        if self.ref_visible is not None and self.ref_visible[i]:
            rx, ry = self.ref[i]
            people.append({'time': t, 'track': 9000, 'box': [round(float(v), 5) for v in self.ref_box[i]], 'score': .8, 'cls': 'player',
                           'zone': 'inside', 'evidence': 'observed', 'id': None, 'display': 'REF', 'team': None, 'role': 'NON_TARGET',
                           'label': 'NON_TARGET', 'state': 'nontarget', 'identityConfidence': 0.0, 'roleConfidence': .9,
                           'pitch': [round(float(rx/LENGTH), 5), round(float(ry/WIDTH), 5)], 'pitchConfidence': conf, 'pitchSource': 'observed',
                           'pitchSigma': [.5, .5]})
        return {'time': round(t, 6), 'frame': int(self.frame[i]), 'people': people, 'ball': self._ball(i), 'calibrated': True,
                'cameraReliable': True, 'cut': False, 'pitchConfidence': conf, 'pitchMode': 'simulated', 'profile': {'smallScene': False},
                'view': self.view[i], 'calibration': {'confidence': conf, 'source': 'simulated', 'disagreement': None}}

    def _ball(self, i):
        st = int(self.state[i])
        if st == 0:
            return {'state': 'UNKNOWN', 'phase': 'PATH', 'confidence': 0.0, 'identity': 0.0, 'source': 'PATH', 'origin': 'UNKNOWN', 'motion': 'UNKNOWN',
                    'firstPass': {'state': 'UNKNOWN', 'phase': 'SEARCHING', 'confidence': 0.0, 'track': None}}
        cx, cy = self.ball_px[i]
        d = max(2.0, float(self.ball_d[i]))
        out = {'state': 'TRACKED' if st == 1 else 'MISSING', 'phase': 'PATH', 'observed': st == 1, 'confidence': .94 if st == 1 else .75,
               'identity': .94, 'visual': .8 if st == 1 else None, 'trajectory': 1.0, 'track': 1, 'source': 'PATH',
               'box': [round((cx-d/2)/IMG_W, 5), round((cy-d/2)/IMG_H, 5), round(d/IMG_W, 5), round(d/IMG_H, 5)],
               'center': [round(cx/IMG_W, 5), round(cy/IMG_H, 5)]}
        sp = self.ball_speed[i]
        speed = None if not np.isfinite(sp) else round(float(sp), 2)
        if st == 1:
            out.update(origin='WEAK_DETECTION' if self.weak[i] else 'OBSERVATION', motion=_motion(sp), speed=speed)
            fp = {'state': 'TRACKED', 'phase': 'LOCKED', 'confidence': .9, 'track': 1, 'center': out['center']}
        else:
            out.update(bridged=True, origin='INTERPOLATION', motion='OCCLUDED', missingFor=round(float(self.missing_for[i]), 2), speed=speed)
            occluded = self.holder_carried[i] >= 0 or self.missing_for[i] <= OCCLUDED_MAX
            fp = {'state': 'MISSING', 'phase': 'OCCLUDED' if occluded else 'RECOVERING', 'confidence': .4, 'track': 1, 'motion': 'OCCLUDED'}
        if self.near[i] >= 0:
            out['nearTrack'] = int(self.near[i])
        x, y = self.ball_est[i]
        if np.isfinite(x) and np.isfinite(y):
            out['pitch'] = [round(float(x/LENGTH), 4), round(float(y/WIDTH), 4)]
            out['pitchConfidence'] = round(float(self.conf[i]), 3)
            out['pitchSigma'] = [round(float(v), 2) for v in self.ball_sigma[i]]
        out['firstPass'] = fp
        return out


# ----------------------------------------------------------------------------------------------- simulation

def simulate(g, mode='camera', fps=30.0, seed=0, width=VIEW_WIDTH, dead_fill='hold', referee=True, detect=None, noise='camera'):
    """Simulate a game loaded by possession_truth.load_game (or load_tracking): a Sim with per-frame arrays."""
    if mode not in ('full', 'camera') or noise not in ('camera', 'independent'):
        raise ValueError((mode, noise))
    sim = scene(g, mode, fps, seed, width, dead_fill, referee)
    observe(sim, detect, noise)
    return sim


def scene(g, mode='camera', fps=30.0, seed=0, width=VIEW_WIDTH, dead_fill='hold', referee=True):
    """The deterministic part: resampled truth, the dead-ball fill, ball heights, the camera, projections, contexts."""
    rng = np.random.default_rng(seed)
    times = g['time']
    j0, j1 = int(math.ceil(times[0]*fps-1e-9)), int(math.floor(times[-1]*fps+1e-9))
    frame = np.arange(j0, j1+1)
    t = frame/fps
    T = len(t)
    players = np.concatenate([g['Home'], g['Away']], axis=1)
    ids = list(g['ids']['Home'])+list(g['ids']['Away'])
    team = ['A']*g['Home'].shape[1]+['B']*g['Away'].shape[1]
    P = players.shape[1]
    pos = _resample(times, players, t)
    ball = _resample(times, g['ball'], t)
    sim = Sim()
    sim.g = g
    sim.game, sim.mode, sim.fps, sim.seed = int(g['game']), mode, float(fps), int(seed)
    sim.params = {'width': width, 'deadFill': dead_fill, 'camBack': CAM_BACK, 'camHeight': CAM_HEIGHT, 'lag': LAG}
    sim.t, sim.frame, sim.ids, sim.team, sim.rng = t, frame, ids, team, rng
    sim.true_pos, sim.true_ball = pos, ball
    shown = ball.copy()
    if dead_fill in ('hold', 'visible') and mode == 'camera':
        roll, hold = int(round(DEAD_ROLL_S*fps)), int(round(DEAD_HOLD_S*fps))
        if dead_fill == 'visible':
            hold = T
        for a, b in _runs(~np.isfinite(ball[:, 0])):
            if b-a+1 < fps:
                continue
            if a >= 2 and np.isfinite(ball[a-1]).all() and np.isfinite(ball[a-2]).all():
                v = ball[a-1]-ball[a-2]
                for k in range(min(roll, b-a+1)):
                    shown[a+k] = ball[a-1]+v*(k+1)*max(0.0, 1-k/(2.0*max(roll, 1)))
            if b+1 < T and np.isfinite(ball[b+1]).all():
                shown[max(a+roll, b-hold+1):b+1] = ball[b+1]
    sim.shown_ball = shown
    true_speed = np.full(T, np.nan)
    true_speed[1:] = np.hypot(*np.diff(shown, axis=0).T)*fps
    sim.true_speed = true_speed
    z = np.zeros(T)
    if mode == 'camera':
        for a, b in _runs(np.nan_to_num(true_speed, nan=0.0) >= FLIGHT_SPEED):
            dur = (b-a+1)/fps
            dist = float(np.nansum(true_speed[a:b+1])/fps)
            if dur >= FLIGHT_MIN_S and dist >= LOFT_MIN_M and rng.random() < LOFT_P:
                apex = min(LOFT_MAX, 9.81*dur*dur/8)
                s = (np.arange(a, b+1)-a+.5)/(b-a+1)
                z[a:b+1] = 4*apex*s*(1-s)
    sim.ball_z = z
    C = np.array([LENGTH/2, WIDTH+CAM_BACK, CAM_HEIGHT])
    look = np.zeros((T, 2))
    cur = np.array([LENGTH/2, WIDTH/2])
    alpha = 1-math.exp(-1.0/(LAG*fps))
    lo_x, hi_x = width/2-15, LENGTH-width/2+15
    for i in range(T):
        tg = shown[i]
        if tg[0] == tg[0] and tg[1] == tg[1]:
            gx = min(max(tg[0], lo_x), hi_x)
            gy = min(max(tg[1], LOOK_Y[0]), LOOK_Y[1])
            sx, sy = alpha*(gx-cur[0]), alpha*(gy-cur[1])
            n = math.hypot(sx, sy)
            if n > MAX_PAN/fps:
                sx, sy = sx*MAX_PAN/fps/n, sy*MAX_PAN/fps/n
            cur = np.array([cur[0]+sx, cur[1]+sy])
        look[i] = cur
    dx, dy = look[:, 0]-C[0], C[1]-look[:, 1]
    pan = np.arctan2(dx, dy)
    tilt = np.arctan2(C[2], np.hypot(dx, dy))
    foc = (IMG_W/2)/((width/2)/np.sqrt(dx*dx+dy*dy+C[2]**2))
    R = rotation(pan, tilt)
    sim.C, sim.R, sim.foc, sim.pan, sim.tilt, sim.look = C, R, foc, pan, tilt, look
    P3 = np.concatenate([pos, np.zeros((T, P, 1))], -1)
    feet_px, depth = project(C, R, foc, P3)
    head_px, _ = project(C, R, foc, P3+np.array([0, 0, PLAYER_H]))
    h = np.abs(feet_px[..., 1]-head_px[..., 1])
    in_img = (depth > 0) & (feet_px[..., 0] >= 0) & (feet_px[..., 0] < IMG_W) & (feet_px[..., 1] >= 0) & (feet_px[..., 1] < IMG_H)
    on = np.isfinite(pos[..., 0])
    if mode == 'full':
        visible = on.copy()
    else:
        drop = np.zeros((T, P), bool)
        u = rng.random((T, P))
        for i in range(1, T):
            drop[i] = np.where(drop[i-1], u[i] < DROP_KEEP, u[i] < DROP_P)
        visible = on & in_img & ~drop
    sim.feet_px, sim.h, sim.visible, sim.on = feet_px, h, visible, on
    sim.box = _boxes(feet_px, h)
    track = np.zeros((T, P), np.int64)
    nxt = 1
    for j in range(P):
        for a, b in _runs(visible[:, j]):
            track[a:b+1, j] = nxt
            nxt += 1
    sim.track = track
    B3 = np.concatenate([shown, z[:, None]], -1)[:, None, :]
    bpx, bdepth = project(C, R, foc, B3)
    tpx, _ = project(C, R, foc, B3+np.array([0, 0, BALL_D]))
    bpx, bdepth = bpx[:, 0], bdepth[:, 0]
    sim.ball_d = np.abs(tpx[:, 0, 1]-bpx[:, 1])
    have = np.isfinite(shown[:, 0])
    sim.ball_in = have & (bdepth > 0) & (bpx[:, 0] >= 0) & (bpx[:, 0] < IMG_W) & (bpx[:, 1] >= 0) & (bpx[:, 1] < IMG_H) if mode == 'camera' else have
    sim.ball_ground_px = bpx                               # the image point under the ball (its box bottom on the ground)
    centre = bpx-np.stack([np.zeros(T), sim.ball_d/2], -1)
    sim.ball_centre_px = centre
    sim.holder = _lower_box(centre, sim.box, visible, track)
    # A ball that was in a player's lower box within FEET_HOLD_S is still at his feet (between two touches of a dribble).
    k = int(round(FEET_HOLD_S*fps))
    held = sim.holder >= 0
    if k > 0:
        held = np.convolve(held.astype(np.int64), np.ones(2*k+1, np.int64), 'same') > 0
        idx = np.where(sim.holder >= 0, np.arange(T), -1)
        last = np.maximum.accumulate(idx)
        carry = np.where(last >= 0, sim.holder[np.maximum(last, 0)], -1)
        sim.holder_carried = np.where(held & (np.arange(T)-last <= k) & (last >= 0), carry, -1)
    else:
        sim.holder_carried = sim.holder
    ctx = np.where(held, 0, np.where(z > AIR_Z, 3, np.where(np.nan_to_num(true_speed) >= TRAVEL_SPEED, 2, 1)))
    sim.ctx = ctx                                          # 0 feet, 1 free, 2 fast, 3 air (CONTEXTS)
    sim.ref = sim.ref_box = sim.ref_visible = None
    if referee:
        ref = np.full((T, 2), np.nan)
        curr = None
        a2 = 1-math.exp(-1.0/(2.0*fps))
        off = np.array(REF_OFFSET)
        for i in range(T):
            if np.isfinite(shown[i]).all():
                goal = np.clip(shown[i]+off, [2, 2], [LENGTH-2, WIDTH-2])
                curr = goal if curr is None else curr+a2*(goal-curr)
            if curr is not None:
                ref[i] = curr
        R3 = np.concatenate([ref, np.zeros((T, 1))], -1)[:, None, :]
        rpx, rdepth = project(C, R, foc, R3)
        rh, _ = project(C, R, foc, R3+np.array([0, 0, PLAYER_H]))
        hh = np.abs(rpx[:, 0, 1]-rh[:, 0, 1])
        sim.ref = ref
        sim.ref_box = np.stack([(rpx[:, 0, 0]-BOX_ASPECT*hh/2)/IMG_W, (rpx[:, 0, 1]-hh)/IMG_H, BOX_ASPECT*hh/IMG_W, hh/IMG_H], -1)
        vis = np.isfinite(ref[:, 0])
        if mode == 'camera':
            vis &= (rdepth[:, 0] > 0) & (rpx[:, 0, 0] >= 0) & (rpx[:, 0, 0] < IMG_W) & (rpx[:, 0, 1] >= 0) & (rpx[:, 0, 1] < IMG_H)
        sim.ref_visible = vis
    period = np.asarray(g['period'])
    gk_x = np.nanmean(g['Home'][period == 1, 0, 0])
    sim.ends = {'left': 'A', 'right': 'B'} if gk_x < LENGTH/2 else {'left': 'B', 'right': 'A'}
    players_, gks, slots = [], [], []
    for j, pid in enumerate(ids):
        if not on[:, j].any():
            continue
        entry = {'id': pid, 'display': pid, 'team': team[j], 'confirmed': True}
        (gks if pid.startswith('GK') else players_).append(entry)
        present = np.flatnonzero(on[:, j])
        slots.append({'id': pid, 'team': team[j], 'role': 'GOALKEEPER' if pid.startswith('GK') else 'PLAYER', 'confirmed': True,
                      'firstSeen': float(t[present[0]]), 'left': float(t[present[-1]]) if present[-1] < T-1 else None})
    sim.match = {'players': players_, 'goalkeepers': gks}
    sim.roster = {'expected': {'A': 11, 'B': 11}, 'slots': slots}
    return sim


def observe(sim, detect=None, noise='camera', seed=None):
    """The random part: registration and detection noise, positions, the ball's detection and final path."""
    rng = np.random.default_rng(sim.seed+1 if seed is None else seed)
    fps, T = sim.fps, len(sim.t)
    P = sim.true_pos.shape[1]
    detect = dict(DETECT if detect is None else detect)
    sim.params.update(detect=detect, noise=noise)
    C, R, foc = sim.C, sim.R, sim.foc
    if sim.mode == 'camera' and noise == 'camera':
        R_est = rotation(sim.pan+_ar1((T,), REG_TAU, fps, math.radians(PAN_SIGMA), rng), sim.tilt+_ar1((T,), REG_TAU, fps, math.radians(TILT_SIGMA), rng))
        foc_est = foc*(1+_ar1((T,), REG_TAU, fps, ZOOM_SIGMA, rng))
        foot = sim.feet_px+_ar1((T, P, 2), FOOT_TAU, fps, FOOT_SIGMA, rng)
        sim.box = _boxes(foot, sim.h)                     # detected boxes: the noisy foot point, the true height
        sim.est = ground(C, R_est, foc_est, foot)
        sim.sigma = _metres_per_px(C, R_est, foc_est, foot)*3.0+SIGMA_REG
        bpx = sim.ball_ground_px+rng.normal(size=(T, 2))*BALL_PX_SIGMA
        b_est = ground(C, R_est, foc_est, bpx[:, None, :])[:, 0]
        b_sig = _metres_per_px(C, R_est, foc_est, bpx[:, None, :])[:, 0]*3.0+SIGMA_REG
        sim.view = [_view(v) for v in ground(C, R_est, foc_est, np.broadcast_to(np.array([[0, 0], [IMG_W, 0], [IMG_W, IMG_H], [0, IMG_H]], float),
                                                                                    (T, 4, 2)).copy())]
        sim.conf = np.clip(CONFIDENCE[0]+_ar1((T,), 5.0, fps, CONFIDENCE[1], rng), .5, .95)
    elif sim.mode == 'camera':
        sig = rng.uniform(*INDEPENDENT_SIGMA, size=(1, P, 1))
        sim.est = sim.true_pos+_ar1((T, P, 2), 1.0, fps, 1.0, rng)*sig
        sim.sigma = np.broadcast_to(sig[..., 0][..., None]*np.ones((1, 1, 2)), (T, P, 2)).copy()
        bpx = sim.ball_ground_px.copy()
        b_est = sim.shown_ball+rng.normal(size=(T, 2))*.3
        b_sig = np.full((T, 2), .3)
        sim.view = [_view(v) for v in ground(C, R, foc, np.broadcast_to(np.array([[0, 0], [IMG_W, 0], [IMG_W, IMG_H], [0, IMG_H]], float),
                                                                            (T, 4, 2)).copy())]
        sim.conf = np.full(T, .8)
    else:
        sim.est = sim.true_pos.copy()
        sim.sigma = np.full((T, P, 2), .3)
        bpx = sim.ball_ground_px.copy()
        b_est = sim.shown_ball.copy()
        b_sig = np.full((T, 2), .3)
        sim.view = [[[-.05, -.05], [1.05, -.05], [1.05, 1.05], [-.05, 1.05]]]*T
        sim.conf = np.full(T, .9)
    if sim.mode == 'full':
        seen = sim.ball_in.copy()
    else:
        seen = detect_ball(sim.ctx, sim.ball_in, detect, rng.random(T))
    sim.sightable = (b_est, bpx-np.stack([np.zeros(T), sim.ball_d/2], -1), b_sig)   # kept for re-running the detection
    assemble(sim, seen, *sim.sightable)
    sim.weak = (rng.random(T) < WEAK_SHARE) & (sim.state == 1) & (sim.mode == 'camera')
    near = np.full(T, -1, np.int64)
    tr = sim.state == 1
    near[tr] = sim.holder[tr]
    br = (sim.state == 2) & (sim.holder_carried >= 0) & (rng.random(T) < NEAR_P)
    near[br] = sim.holder_carried[br]
    sim.near = near
    return sim


def detect_ball(ctx, ball_in, detect, u, decay=None):
    """Per frame: is the ball seen? A two-state chain whose transition probabilities depend on the context; a miss
    continues with P(missed | missed) * exp(-L / decay) after L missed frames (the detector finds a hidden ball again
    within about a second: the audit's bridged spells reach 1-2.3 s while UNKNOWN stays at .1%)."""
    stay = [detect[k][0] for k in CONTEXTS]
    miss = [detect[k][1] for k in CONTEXTS]
    tau = float(MISS_DECAY if decay is None else decay)
    T = len(ctx)
    seen = np.zeros(T, bool)
    prev = False
    run = 0
    ctx_l = ctx.tolist()
    in_l = ball_in.tolist()
    u_l = u.tolist()
    ex = math.exp
    for i in range(T):
        if not in_l[i]:
            prev, run = False, 0
            continue
        c = ctx_l[i]
        if prev:
            prev = u_l[i] < stay[c]
        else:
            prev = u_l[i] >= miss[c]*ex(-run/tau)
        run = 0 if prev else run+1
        seen[i] = prev
    return seen


def assemble(sim, seen, b_est, b_px, b_sig):
    """soccer/ball_path.py's final path from the sightings: bridged gaps (MISSING), dropped short stretches, speeds."""
    fps, T = sim.fps, len(seen)
    state = np.zeros(T, np.int8)
    missing_for = np.zeros(T)
    est = np.full((T, 2), np.nan)
    px = np.full((T, 2), np.nan)
    sig = np.full((T, 2), np.nan)
    speed = np.full(T, np.nan)
    sight = np.flatnonzero(seen)
    skip, still = int(round(MAX_SKIP_S*fps)), int(round(MAX_STILL_S*fps))
    if len(sight):
        gap = np.diff(sight)-1
        d = np.hypot(*(b_est[sight[1:]]-b_est[sight[:-1]]).T)
        with np.errstate(invalid='ignore'):
            ok = (gap <= skip) | ((gap <= still) & (d/((gap+1)/fps) < STILL_SPEED))
        cuts = np.flatnonzero(~ok)+1
        groups = np.split(sight, cuts)
    else:
        groups = []
    win = max(1, int(round(SPEED_WINDOW*fps)))
    for grp in groups:
        if len(grp) < MIN_RUN:
            continue
        state[grp] = 1
        est[grp], px[grp], sig[grp] = b_est[grp], b_px[grp], b_sig[grp]
        tg = grp/fps
        lo = np.searchsorted(grp, grp-win)
        hi = np.searchsorted(grp, grp+win, side='right')-1
        dt = tg[hi]-tg[lo]
        with np.errstate(invalid='ignore', divide='ignore'):
            sp = np.hypot(*(b_est[grp[hi]]-b_est[grp[lo]]).T)/dt
        link = np.full(len(grp), np.nan)
        link[1:] = np.hypot(*np.diff(b_est[grp], axis=0).T)/(np.diff(grp)/fps)
        sp = np.where(dt > 0, sp, link)
        speed[grp] = sp
        for k in np.flatnonzero(np.diff(grp) > 1):
            a, b = grp[k], grp[k+1]
            m = np.arange(a+1, b)
            w = ((m-a)/(b-a))[:, None]
            state[m] = 2
            est[m] = (1-w)*b_est[a]+w*b_est[b]
            px[m] = (1-w)*b_px[a]+w*b_px[b]
            sig[m] = np.maximum(b_sig[a], b_sig[b])
            missing_for[m] = (m-a)/fps
            speed[m] = link[k+1]
    sim.state, sim.ball_est, sim.ball_px, sim.ball_sigma, sim.ball_speed, sim.missing_for = state, est, px, sig, speed, missing_for
    sim.seen = seen


def _boxes(feet, h):
    """Normalized [x, y, w, h] boxes standing on the feet pixels (T, P, 2) with heights h (T, P) px."""
    return np.stack([(feet[..., 0]-BOX_ASPECT*h/2)/IMG_W, (feet[..., 1]-h)/IMG_H, BOX_ASPECT*h/IMG_W, h/IMG_H], -1)


def _lower_box(centre, box, visible, track):
    """Per frame, the track of the visible player whose lower box (FEET_BOX) holds the ball centre, the one whose box
    centre is horizontally closest (soccer/ball_path.py lower_body); -1 if none."""
    T, P = visible.shape
    bx, by = box[..., 0]*IMG_W, box[..., 1]*IMG_H
    bw, bh = box[..., 2]*IMG_W, box[..., 3]*IMG_H
    with np.errstate(invalid='ignore', divide='ignore'):
        u = (centre[:, None, 0]-bx)/bw
        v = (centre[:, None, 1]-by)/bh
    u0, u1, v0, v1 = FEET_BOX
    inside = visible & np.isfinite(u) & np.isfinite(v) & (u >= u0) & (u <= u1) & (v >= v0) & (v <= v1)
    score = np.where(inside, np.abs(u-.5), np.inf)
    j = np.argmin(score, axis=1)
    ok = np.isfinite(score[np.arange(T), j])
    return np.where(ok, track[np.arange(T), j], -1)


def _view(pts):
    """The image footprint on the pitch, clipped to the pitch +-5 m, normalized (frames[].view)."""
    poly = [p for p in pts if np.isfinite(p).all()]
    if len(poly) < 3:
        return None
    for axis, bound, keep_le in ((0, -5.0, False), (0, LENGTH+5, True), (1, -5.0, False), (1, WIDTH+5, True)):
        out = []
        for k in range(len(poly)):
            a, b = poly[k-1], poly[k]
            ina = a[axis] <= bound if keep_le else a[axis] >= bound
            inb = b[axis] <= bound if keep_le else b[axis] >= bound
            if inb:
                if not ina:
                    out.append(a+(bound-a[axis])/(b[axis]-a[axis])*(b-a))
                out.append(b)
            elif ina:
                out.append(a+(bound-a[axis])/(b[axis]-a[axis])*(b-a))
        poly = out
        if not poly:
            return None
    return [[round(float(p[0]/LENGTH), 4), round(float(p[1]/WIDTH), 4)] for p in poly]


# ----------------------------------------------------------------------------------------------- calibration report

def _spells(mask, fps):
    r = _runs(mask)
    return np.array([(b-a+1)/fps for a, b in r]) if r else np.zeros(0)


def _q(v, q):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return round(float(np.percentile(v, q)), 2) if len(v) else None


def calibration(sim, g=None, in_play_only=True):
    """The ball audit's broadcast measures on the simulation (coaching ball audit sections 2-3; the people stats of BC60):
    ball state shares, bridged / tracked spells, why frames are bridged, origin and motion shares, sigma, the ball's
    distance to the carrier it sits on, the players' errors, visible players, the one-team-within-r shares (pitch
    metric, observed + bridged), pitch-vs-image agreement (observed, 1.5 m), consecutive-sighting speeds."""
    g = g or sim.g
    fps = sim.fps
    code, _ = mt.truth_at(g, sim.t)
    play = code != mt.DEAD if in_play_only else np.ones(len(sim.t), bool)
    st = sim.state
    n = max(1, int(play.sum()))
    out = {'mode': sim.mode, 'frames': int(play.sum()), 'inPlayOnly': bool(in_play_only)}
    out['ball'] = {k: round(float(((st == v) & play).sum()/n), 3) for k, v in (('TRACKED', 1), ('MISSING', 2), ('UNKNOWN', 0))}
    bridged, tracked, unknown = (_spells((st == v) & play, fps) for v in (2, 1, 0))
    out['bridgedSpells'] = {'count': int(len(bridged)), 'perMinute': round(len(bridged)/(n/fps/60), 1), 'median': _q(bridged, 50), 'p90': _q(bridged, 90),
                            'max': round(float(bridged.max()), 2) if len(bridged) else None, 'atLeast1sPerMinute': round(float((bridged >= 1).sum())/(n/fps/60), 2)}
    out['trackedSpells'] = {'median': _q(tracked, 50), 'p90': _q(tracked, 90)}
    out['unknownSpells'] = {'count': int(len(unknown)), 'longest': round(float(unknown.max()), 2) if len(unknown) else 0.0}
    br, tr = (st == 2) & play, (st == 1) & play
    out['bridgedAtFeet'] = round(float((sim.ctx[br] == 0).mean()), 3) if br.any() else None
    occ = (sim.holder_carried >= 0) | (sim.missing_for <= OCCLUDED_MAX)
    out['bridgedFirstPassOccluded'] = round(float(occ[br].mean()), 3) if br.any() else None
    out['bridgedWithNearTrack'] = round(float((sim.near[br] >= 0).mean()), 3) if br.any() else None
    out['contextShare'] = {c: round(float(((sim.ctx == k) & play & sim.ball_in).sum()/n), 3) for k, c in enumerate(CONTEXTS)}
    out['origin'] = {'OBSERVATION': round(float((tr & ~sim.weak).sum()/n), 3), 'WEAK_DETECTION': round(float((tr & sim.weak).sum()/n), 3)}
    sp = sim.ball_speed
    with np.errstate(invalid='ignore'):
        out['motion'] = {'STATIONARY': round(float((tr & (sp < STILL_SPEED)).sum()/n), 3),
                         'ROLLING': round(float((tr & (sp >= STILL_SPEED) & (sp < TRAVEL_SPEED)).sum()/n), 3),
                         'FAST': round(float((tr & (sp >= TRAVEL_SPEED)).sum()/n), 3)}
    pos = (st > 0) & play & np.isfinite(sim.ball_est[:, 0])
    out['ballSigma'] = {'median': _q(np.nanmax(sim.ball_sigma[pos], axis=1), 50), 'p90': _q(np.nanmax(sim.ball_sigma[pos], axis=1), 90)}
    e_tr = np.hypot(*(sim.ball_est[tr & pos]-sim.shown_ball[tr & pos]).T)
    e_br = np.hypot(*(sim.ball_est[br & pos]-sim.shown_ball[br & pos]).T)
    out['ballPositionError'] = {'trackedMedian': _q(e_tr, 50), 'trackedP90': _q(e_tr, 90), 'bridgedMedian': _q(e_br, 50), 'bridgedP90': _q(e_br, 90)}
    with np.errstate(invalid='ignore'):
        beyond = np.maximum.reduce([-sim.ball_est[:, 0], sim.ball_est[:, 0]-LENGTH, -sim.ball_est[:, 1], sim.ball_est[:, 1]-WIDTH])
    out['ballOutsidePitch'] = round(float((pos & (beyond > .02*LENGTH)).sum()/max(1, pos.sum())), 4)
    gap = []
    for i in np.flatnonzero(tr & (sim.near >= 0) & pos):
        j = np.flatnonzero((sim.track[i] == sim.near[i]) & sim.visible[i])
        if len(j) and np.isfinite(sim.est[i, j[0]]).all():
            gap.append(float(np.hypot(*(sim.ball_est[i]-sim.est[i, j[0]]))))
    out['ballToCarrierFeet'] = {'n': len(gap), 'median': _q(gap, 50), 'p90': _q(gap, 90)}
    vis = sim.visible & play[:, None]
    e = np.hypot(*(sim.est[vis]-sim.true_pos[vis]).T)
    y = sim.true_pos[vis][:, 1]
    out['playerError'] = {'median': _q(e, 50), 'p90': _q(e, 90), 'p99': _q(e, 99), 'farThirdMedian': _q(e[y < WIDTH/3], 50),
                          'nearThirdMedian': _q(e[y > 2*WIDTH/3], 50)}
    rms = []
    for j in range(vis.shape[1]):
        m = vis[:, j]
        if m.sum() > 100:
            rms.append(float(np.sqrt(np.nanmean(np.sum((sim.est[m, j]-sim.true_pos[m, j])**2, axis=1)/2))))
    out['playerErrorRmsPerAxis'] = {'p10': _q(rms, 10), 'median': _q(rms, 50), 'p90': _q(rms, 90)}
    s = np.nanmax(sim.sigma[vis], axis=1)
    out['playerSigma'] = {'p10': _q(s, 10), 'median': _q(s, 50), 'p90': _q(s, 90)}
    cnt = vis.sum(axis=1)[play]
    out['visiblePlayers'] = {'p10': _q(cnt, 10), 'median': _q(cnt, 50), 'p90': _q(cnt, 90)}
    teamA = np.array([tm == 'A' for tm in sim.team])
    d = np.hypot(*(sim.est-sim.ball_est[:, None, :]).transpose(2, 0, 1))
    d = np.where(sim.visible & np.isfinite(d), d, np.inf)
    dA = d[:, teamA].min(axis=1)
    dB = d[:, ~teamA].min(axis=1)
    out['oneTeamWithin'] = {f'{r:g}m': round(float((((dA <= r) ^ (dB <= r)) & pos).sum()/max(1, pos.sum())), 3) for r in (1.0, 1.5, 2.0, 3.0)}
    out['bothTeamsWithin1.5m'] = round(float(((dA <= 1.5) & (dB <= 1.5) & pos).sum()/max(1, pos.sum())), 3)
    # Pitch vs image labels on observed frames at 1.5 m (the audit's within_reach geometry for the image metric).
    lab_p = np.where((dA <= 1.5) & (dB <= 1.5), 3, np.where(dA <= 1.5, 1, np.where(dB <= 1.5, 2, 0)))
    fx = (sim.box[..., 0]+sim.box[..., 2]/2)*IMG_W
    fy = (sim.box[..., 1]+sim.box[..., 3])*IMG_H
    cx = sim.ball_px[:, 0]
    cy = sim.ball_px[:, 1]+sim.ball_d/2
    hpx = sim.box[..., 3]*IMG_H
    near_rows = sim.visible & (np.abs(fy-cy[:, None]) < .25*IMG_H)
    hh = np.where(near_rows, hpx, np.nan)
    with np.errstate(invalid='ignore'):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            pxm = np.nanmedian(hh, axis=1)/1.8
    pxm = np.where(np.isfinite(pxm), pxm, .09*IMG_H/1.8)
    di = np.hypot(cx[:, None]-fx, 2.0*(cy[:, None]-fy))/pxm[:, None]
    di = np.where(sim.visible & np.isfinite(di), di, np.inf)
    iA, iB = di[:, teamA].min(axis=1), di[:, ~teamA].min(axis=1)
    lab_i = np.where((iA <= 1.5) & (iB <= 1.5), 3, np.where(iA <= 1.5, 1, np.where(iB <= 1.5, 2, 0)))
    obs = tr & pos & np.isfinite(cx)
    agree = (lab_p == lab_i) & obs
    one = obs & ((lab_p == 1) | (lab_p == 2) | (lab_i == 1) | (lab_i == 2))
    out['pitchVsImage'] = {'agree': round(float(agree.sum()/max(1, obs.sum())), 3),
                           'oneTeamAgree': round(float((agree & ((lab_p == 1) | (lab_p == 2))).sum()/max(1, one.sum())), 3),
                           'imageOneTeamShare': round(float((((iA <= 1.5) ^ (iB <= 1.5)) & pos & np.isfinite(cx)).sum()/max(1, pos.sum())), 3)}
    # Consecutive TRACKED frames: pitch speed (the audit's pitchStepSpeed).
    k = np.flatnonzero(tr[1:] & tr[:-1])
    step = np.hypot(*(sim.ball_est[k+1]-sim.ball_est[k]).T)*fps
    step = step[np.isfinite(step)]
    out['consecutiveSpeedOver'] = {'25': round(float((step > 25).mean()), 4) if len(step) else None, '40': round(float((step > 40).mean()), 4) if len(step) else None}
    return out


# The coaching ball audit's broadcast numbers, the calibration targets: BC0 (0-60 s) and BC42 (42.4-222.4 s) of the
# La Liga clip; BC60's people statistics measured by the possession truth role (2026-10-20).
AUDIT = {
    'ball': {'TRACKED': (.561, .641), 'MISSING': (.438, .358), 'UNKNOWN': (.001, .001)},
    'bridgedSpells': {'median': (.10, .10), 'p90': (.69, .63), 'max': (2.33, 1.80), 'atLeast1sPerMinute': (3.0, 3.0)},
    'trackedSpells': {'median': (.10, .17), 'p90': (.97, 1.10)},
    'bridgedFirstPassOccluded': (.88, .79), 'bridgedWithNearTrack': (.84, .69),
    'origin': {'OBSERVATION': (.539, .616), 'WEAK_DETECTION': (.021, .026)},
    'motion': {'STATIONARY': (.127, .172), 'ROLLING': (.331, .339), 'FAST': (.103, .130)},
    'ballSigma': {'median': (.45, .5), 'p90': (.9, 13.6)},
    'ballToCarrierFeet': {'median': (.47, .48), 'p90': (.79, .92)},
    'ballOutsidePitch': (.016, .038),
    'oneTeamWithin': {'1m': (.45, .41), '1.5m': (.591, .553), '2m': (.64, .59), '3m': (.66, .60)},
    'pitchVsImage': {'agree': (.924, .925), 'oneTeamAgree': (.875, .870), 'imageOneTeamShare': (.625, .520)},
    'consecutiveSpeedOver': {'25': (.011, .039), '40': (.001, .006)},
    'bothTeamsWithin1.5m': (.001, .028),
    'visiblePlayers': {'median': (21, None)},
    'playerSigma': {'p10': (.2, None), 'median': (.4, None), 'p90': (.9, None)},
}
