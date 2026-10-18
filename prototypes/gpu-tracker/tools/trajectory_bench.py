"""Accuracy of the player trajectory smoothers on real movement: the Metrica Sports sample games as ground truth, seen
through a simulated camera with the measurement noise measured on the user's runs (the bouncy-marker investigation of
Oct 18), projected back to the pitch, smoothed by every method and compared with the truth.

    python -I tools/trajectory_bench.py --data DIR --game 2 [--camera pole] [--camera broadcast] [--windows 4] [--length 300]
                                        [--rate 30] [--methods raw,renderer,sg0.3,sg0.5,rts,smooth_tracks,stabilizer] [--rts-q Q]
                                        [--scenario base] [--scenario all] [--seed 0] [--seeds 1] [--keep-glitches] [--json OUT]

DIR holds the Metrica sample data (tools/metrica_bench.py; downloaded data: run with python -I). Game 1 tunes the
constants, game 2 is held out for the reported numbers. That protocol was broken once: during the stabilizer's tuning
on Oct 18 two sprint probes ran on game 2 before the last edit of soccer/trajectory.py (pole sprint RMSE moved 0.673 ->
0.666 m between the two game-2 reports). Keep game 2 for reporting; a tuning that needs a second sample splits game 1
(its first half against its second). --windows windows of --length seconds are spread evenly over the two periods
(never across half time, where the players jump to their kick-off positions); both teams are seen together (they
occlude each other), resampled from 25 Hz to --rate by linear interpolation (the runs are 30 fps).

The truth has tracking glitches of its own: one-frame steps faster than any player (GLITCH_SPEED; game 1 456 and game 2
1095 of them at 25 Hz, typically a run of ~6 frames at ~20 m/s that moves the player a few metres, up to 2500 m/s at
half time). Scored as motion they dominated the sprint and turn buckets and the lag, for every method alike (game 2,
broadcast camera, Oct 18: the stabilizer's sprint RMSE 0.53 m with them, 0.11 m without, the raw observations' 0.20 and
0.22; its turn RMSE 0.52 and 0.20 m; its lag -16 and +3 ms), although 0.2% of the samples lie near one. The truth
is now unknown within GLITCH_HALF of each (NaN: the player is not seen there and nothing is scored); the realism block
gives the share (--keep-glitches restores the old scoring, for comparison).

The camera (CAMERAS) is a pinhole at a known position whose aim follows the ball along the length with a lag: 'pole'
is the amateur camera (5 m high beside the halfway line, a wide lens: ground seen at ~8 degrees, a vertical pixel worth
~0.3 m at the centre of the pitch), 'broadcast' a camera 25 m high in the stand (~20 degrees, ~0.11 m). Each true
position gets a foot pixel and a box (PLAYER_HEIGHT tall, BOX_ASPECT wide); people smaller than MIN_BOX or off the
image are not seen. The measurement noise is in PIXELS, as diagnosed: the bottom and top edges each with sigma
sqrt(CORE_PX^2 + (CORE_REL h)^2) and the box centre with that over sqrt(2), white; occlusion from the boxes themselves
(a nearer person covering more than OCCLUDED of the box) multiplies it by OCCLUDED_GAIN per episode and the box-extent
event rate by OCCLUDED_EVENTS, and hides the person above HIDDEN; box-extent events (EVENT_RATE, EVENT_SECONDS) move the
bottom edge up by EVENT_SHIFT of the box height while the top stays (legs cut off: the player appears farther from the
camera); detection scores are low (< .5) mostly in events (LOW_SCORE); short missed detections (MISS_RATE,
MISS_FRAMES); and occasionally two team-mates crossing swap identities (SWITCH_*). That is the 'base' scenario: the
noisy foot pixel projected back through the true camera, and the one box event the stabilizer's top + running-height
repair is built for. The other SCENARIOS add what it is not built for: box events that move the top edge too or
instead (EXTRA_EVENTS: top, shift, torso, reach), and a calibration error (CAL_*: the anchors' errors held between
anchors, 'steps', as the registration snaps at anchor frames; or joined by straight lines, 'drift', as a registration
that crossfades between anchors would; both with a slow wobble) applied to the frames' homographies, through which the observations are projected back
and the stabilizer works. 'all' is every event kind with the drift. Each scenario draws the base processes as 'base'
does (paired comparisons) and its own from a generator of its own. The result handed to the stabilizer carries the
true camera motion between frames (frames[].motion: an affine fit, as the tracker's similarity estimate, within a
pixel in the middle of a panning pole camera's view), so its registration-step handling runs. --seeds N repeats
everything with N seeds (seed, seed+1, ...): the metrics are pooled and their spread across the seeds reported (game 2,
3 seeds: the lag of the raw observations, which have none, spreads by 13 ms on the pole camera and 2 ms on the
broadcast one, the size of the differences it is used to rank). Each camera's seed comes from its name (camera_key),
so its numbers do not depend on the cameras run with it.

Methods (each gives positions on the observed frames and, except raw, inside gaps of at most BRIDGE_GAP):
  raw        the observations
  renderer   tactical.js before the stabilizer (88be58f, still its raw comparison): the centred 0.14 s mean of the
             observations, stopping at a gap, and straight lines between the raw observations across a gap (the
             timeline's I samples)
  sg0.3/0.5  Savitzky-Golay, order 2, windows of 0.3 / 0.5 s on each contiguous run of observations, straight lines
             between the smoothed ends across a gap
  rts        constant-velocity Kalman filter and Rauch-Tung-Striebel smoother, fixed process noise RTS_Q, the
             measurement covariance from the core pixel noise through the camera's Jacobian (no outlier handling)
  smooth_tracks  soccer/trajectory.smooth_tracks (the stabilizer's robust adaptive smoother) on the same observations
             and covariance as rts: what its reweighting and adaptive process noise add
  stabilizer soccer/trajectory.stabilize on the simulated window written as a PitchIQ result (frames with the camera's
             homography and true camera motion, people with box, score, local track, identity and pitch): the whole
             stabilizer, its noise model fitted on the window, its box-event repair and registration-step handling
             included

Metrics against the truth (metres), on the samples whose observation is the identity's own (after a switch another
player's observations are on it for 0.5-3 s: nothing can fix those, their RMSE is reported apart): on the observed
frames RMSE, median and p95 and the RMSE by true speed (SPEED_BUCKETS); on the bridged gaps the RMSE; speed error
(central differences over +-DIFF, mean |error| and the bias); lag (the time shift maximising the correlation of the
estimated with the true velocity, both less their moving mean over LAG_HIGH_PASS each side: positive = late); stop
glide (true stops: from >= STOP_FROM m/s within the second before to < STOP_AT, then under STOP_HOLD for a second; how
far beyond the true stop point the estimate gets along the run direction in that second, minus how far the truth
itself drifts); turn rounding (RMSE and median within TURN_NEAR of a direction change > TURN_ANGLE between the half
seconds before and after, both faster than TURN_SPEED); stationary vibration (the frame-to-frame step while the truth
is slower than STILL); acceleration (second difference over a lag of ACC_LAG) p95 against the truth's. Also per
camera and scenario the realism of the simulation: visible, occluded, event, missed and switched shares, the score
cue, the box height, the metres per vertical pixel, the truth's glitch share, the extra events' shares, the calibration
error, and the stabilizer's registration steps and flags.
"""
import argparse
import json
import math
import sys
import time
import zlib
from pathlib import Path

import numpy as np
from scipy.signal import savgol_filter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'tools'))
from metrica_bench import LENGTH, WIDTH, load_team  # noqa: E402

IMAGE = (1920, 1080)
# Pinhole cameras: position (x along the length, y across: the near touchline is y = 68, height above the ground) and
# focal length in pixels. 'pole' from the amateur run's automatic calibration (camera 4.96 m high, focal 884 px);
# 'broadcast' placed so the ground is seen at ~20 degrees and a vertical pixel at the centre spot is worth 0.11 m, the
# broadcast run's value.
CAMERAS = {'pole': {'position': (52.5, WIDTH+3.0, 5.0), 'focal': 900.0},
           'broadcast': {'position': (52.5, WIDTH+35.0, 25.0), 'focal': 1960.0}}
PAN_TAU = 1.0            # s: the aim follows the ball's x with this first-order lag (across: the middle of the pitch) ...
AIM_RANGE = (10.0, 95.0)  # m: ... and stays over this part of the length
# Assumed, not measured: the box top above the foot point, the box's width over its height, the smallest box detected.
PLAYER_HEIGHT = 1.8      # m
BOX_ASPECT = .4
MIN_BOX = 8.0            # px
# Pixel noise (the diagnosis' robust fit of the bottom edge on the three runs): sigma = sqrt(CORE_PX^2 + (CORE_REL h)^2).
CORE_PX = .4
CORE_REL = .011
# A nearer person covering more than OCCLUDED of the box multiplied the bottom edge's noise by 2-4 and the event rate by ~10
# on the runs (the diagnosis); HIDDEN is assumed.
OCCLUDED = .2
OCCLUDED_GAIN = (2.0, 4.0)   # the noise factor of an occlusion episode (uniform)
OCCLUDED_EVENTS = 10.0
HIDDEN = .7              # covered this much: not detected
# Box-extent events: 1.7-3.7% of the samples on the runs, 0.1-0.7 s. EVENT_RATE .04 gives 3.3% (pole) and 1.7% (broadcast:
# less occlusion) on game 1 (the realism block); .05 gave 4.1% and 2.1%.
EVENT_RATE = .04         # /s per visible person outside occlusion
EVENT_SECONDS = (.1, .7)
EVENT_SHIFT = (.12, .30)  # of the box height the bottom edge moves up (uniform: mean .21)
# P(score < .5), chosen so that an event is ~10x likelier on a low score than on a confident one, the runs' ratio (the realism
# block reports the simulated one).
LOW_SCORE = {'event': .5, 'occluded': .2, 'clean': .05}
# Assumed: short missed detections and identity switches.
MISS_RATE = .2           # /s: short missed detections start at this rate per visible person ...
MISS_FRAMES = (1, 10)    # ... and last this many frames (uniform)
SWITCH_DISTANCE = 1.5    # m: two team-mates coming this close ...
SWITCH_PROB = .05        # ... swap identities with this probability ...
SWITCH_SECONDS = (.5, 3.0)  # ... for this long
# Extra box events (SCENARIOS), each a per-person process like the base one (OCCLUDED_EVENTS times likelier under
# occlusion, EVENT_SECONDS long, EXTRA_LOW_SCORE of its samples with a low score): (rate /s, the share of the box height
# it moves, uniform). Assumed, a stress test of the repair rather than a measurement; the reach rate is the runs' ratio of
# the stabilizer's 'box reaches into a nearer player' to 'box lost the legs' flags (broadcast run 67 to 170) times
# EVENT_RATE.
EXTRA_EVENTS = {'top': (.04, (.10, .30)),     # the top edge moves up (arms raised, a header, a farther head merged in)
                'shift': (.02, (.10, .30)),   # the whole box moves up or down (and sideways by up to half its width): on a neighbour
                'torso': (.02, (.12, .30)),   # the legs lost and the top cut down by half as much: a box on the torso
                'reach': (.015, (.12, .30))}  # the bottom edge moves DOWN, the top stays: the box reaches into a nearer player
EXTRA_LOW_SCORE = .5
# The calibration error (SCENARIOS 'steps', 'drift'): a tilt and (half as large, independent) a pan of the camera in the
# frames' homographies. An anchor every CAL_ANCHORS seconds with an error of CAL_PX pixels (CAL_BIG_SHARE of them
# CAL_BIG_PX), held until the next one ('steps') or joined to it by a straight line ('drift'), plus an Ornstein-Uhlenbeck
# wobble of CAL_WOBBLE_PX with CAL_TAU. Assumed from the stored runs' registrations: the broadcast run has 23 anchors in
# 60 s, 10 of which moved every player by 0.36-0.56 m (3-5 px) and the others by a median 0.03 m (0.3 px); the amateur
# run 65 in 64 s, 4 of them 0.34-0.96 m (1-3 px).
CAL_ANCHORS = (1.0, 3.0)  # s (uniform)
CAL_PX, CAL_BIG_PX, CAL_BIG_SHARE = .4, 2.5, .2
CAL_WOBBLE_PX, CAL_TAU = .3, 1.0
SCENARIOS = {'base': (), 'top': ('top',), 'shift': ('shift',), 'torso': ('torso',), 'reach': ('reach',), 'steps': ('steps',), 'drift': ('drift',),
             'all': ('top', 'shift', 'torso', 'reach', 'drift')}
DEFAULT_SCENARIOS = ('base', 'all')
# The truth's glitches (module docstring): a step faster than this over one truth frame ...
GLITCH_SPEED = 12.0      # m/s (no footballer: top speeds are ~10 m/s)
GLITCH_HALF = .5         # s: ... leaves the truth unknown this far either side of it
MOTION_GRID = (np.linspace(.1, .9, 5), np.linspace(.1, .9, 5))   # normalized pixels the camera motion's affine fit uses
BRIDGE_GAP = 1.0         # s: gaps up to this between two observations are bridged (tactical.js interpolateGap)
RENDERER_WINDOW = .14    # s: tactical.js smoothWindow
SG_WINDOWS = {'sg0.3': .3, 'sg0.5': .5}
# m^2/s^3: white-acceleration spectral density of the fixed-q RTS, chosen on game 1 (4 x 300 s, both cameras, the truth's
# glitches left out): the RMSE falls all the way down the sweep (q .3/1/3/10/30: pole .368/.425/.479/.537/.583 m, broadcast
# .135/.146/.154/.161/.166: a low q averages the box events away) while the error at turns rises (median: pole
# .185/.139/.110/.100/.113 m, broadcast .056/.039/.036/.037/.037); 1 is the knee (with the glitches scored, as first
# tuned: pole turn median .266/.192/.145/.125/.128, the same knee).
RTS_Q = 1.0
RTS_V0 = 4.0             # m/s: velocity sigma of a new segment
METHODS = ('raw', 'renderer', 'sg0.3', 'sg0.5', 'rts', 'smooth_tracks', 'stabilizer')
SPEED_BUCKETS = (('standing', 0.0, .5), ('walking', .5, 2.0), ('jogging', 2.0, 5.0), ('running', 5.0, 7.0), ('sprinting', 7.0, math.inf))
DIFF = .1                # s: half lag of the central differences (speed, velocity)
LAG_MAX = .5             # s: velocity shifts searched for the lag
# s: half window of the moving mean taken off both velocities before the lag's correlation. A player's velocity changes
# slowly, so the plain correlation peaks broadly (0.998 one frame off on game 1) and the raw observations, which have no
# lag, measured +14 ms on the broadcast camera; the changes of velocity peak sharply.
LAG_HIGH_PASS = .5
# m/s: footballers seldom come to rest (game 1: 2-3 stops under .3 m/s per 5 minutes of 28 players, 25-37 under .5 that
# stay under 1 for a second)
STOP_FROM, STOP_AT, STOP_HOLD = 2.0, .5, 1.0
STOP_SECONDS = 1.0       # s after a stop that are measured
TURN_ANGLE, TURN_SPEED, TURN_HALF, TURN_NEAR = 60.0, 2.0, .5, 1.0   # degrees, m/s, s, s
STILL = .3               # m/s: stationary vibration below this true speed
# s: lag of the acceleration's second difference. The truth's own p95 on game 1 is 3.9-4.7 m/s^2 over one frame, 2.8-3.3
# over 0.1 s and 2.5-2.9 over 0.25 s: the short lags add the 25 Hz data's noise and the resampling's kinks.
ACC_LAG = .25


# ---------- the camera ----------
def rotation(position, target):
    """World-to-camera rotations (T, 3, 3): rows right, down, forward. World axes: x along the length, y towards the
    near touchline, z DOWN (so x, y, z is right-handed and the view is not mirrored)."""
    forward = target-position
    forward /= np.linalg.norm(forward, axis=-1, keepdims=True)
    right = np.cross(forward, [0.0, 0.0, -1.0])
    right /= np.linalg.norm(right, axis=-1, keepdims=True)
    down = np.cross(forward, right)
    return np.stack([right, down, forward], axis=-2)


def camera_views(name, ball_x, dt):
    """Per frame homographies (T, 3, 3) pitch metres -> pixels, rotations and the camera centre (z down)."""
    cam = CAMERAS[name]
    C = np.array([cam['position'][0], cam['position'][1], -cam['position'][2]])
    # The aim follows the ball (held while it is unknown), kept over the pitch.
    x = np.clip(ball_x, AIM_RANGE[0], AIM_RANGE[1])
    aim = np.empty(len(x))
    aim[0] = x[0] if np.isfinite(x[0]) else LENGTH/2
    a = 1-math.exp(-dt/PAN_TAU)
    for t in range(1, len(x)):
        aim[t] = aim[t-1]+a*(x[t]-aim[t-1]) if np.isfinite(x[t]) else aim[t-1]
    target = np.stack([aim, np.full_like(aim, WIDTH/2), np.zeros_like(aim)], 1)
    R = rotation(C, target)
    K = np.array([[cam['focal'], 0, IMAGE[0]/2], [0, cam['focal'], IMAGE[1]/2], [0, 0, 1.0]])
    t = -(R @ C)
    H = K @ np.stack([R[:, :, 0], R[:, :, 1], t], axis=-1)
    return H/H[:, 2:3, 2:3], R, C, K


def project(R, C, K, points):
    """Pixels (T, P, 2) and depth (T, P) of world points (T, P, 3)."""
    cam = np.einsum('tij,tpj->tpi', R, points-C)
    z = cam[..., 2]
    with np.errstate(divide='ignore', invalid='ignore'):
        u = K[0, 0]*cam[..., 0]/z+K[0, 2]
        v = K[1, 1]*cam[..., 1]/z+K[1, 2]
    return np.stack([u, v], -1), z


def back_project(H, pixels):
    """Pitch metres (T, P, 2) of pixels through the inverse homographies (NaN behind the camera) and the Jacobian
    d(pitch)/d(pixel) (T, P, 2, 2)."""
    Hi = np.linalg.inv(H)
    q = np.einsum('tij,tpj->tpi', Hi, np.concatenate([pixels, np.ones(pixels.shape[:2]+(1,))], -1))
    w = q[..., 2]
    with np.errstate(divide='ignore', invalid='ignore'):
        p = q[..., :2]/w[..., None]
        J = (Hi[:, None, :2, :2]-p[..., :, None]*Hi[:, None, 2:3, :2])/w[..., None, None]
    p[~(w > 1e-12)] = np.nan
    return p, J


# ---------- the simulation ----------
def runs(mask):
    """[a, b) ranges of consecutive True along axis 0 of a 1-D mask."""
    v = np.concatenate([[False], np.asarray(mask, bool), [False]]).astype(np.int8)
    d = np.diff(v)
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def occlusion(pixels, height, depth, visible, chunk=1500):
    """Per frame and person the largest share of the box covered by a nearer visible person."""
    T, P = visible.shape
    cover = np.zeros((T, P))
    w = BOX_ASPECT*height
    x0, x1 = pixels[..., 0]-w/2, pixels[..., 0]+w/2
    y0, y1 = pixels[..., 1]-height, pixels[..., 1]
    for a in range(0, T, chunk):
        s = slice(a, a+chunk)
        ix = np.clip(np.minimum(x1[s, :, None], x1[s, None, :])-np.maximum(x0[s, :, None], x0[s, None, :]), 0, None)
        iy = np.clip(np.minimum(y1[s, :, None], y1[s, None, :])-np.maximum(y0[s, :, None], y0[s, None, :]), 0, None)
        nearer = (depth[s, None, :] < depth[s, :, None]) & visible[s, None, :] & visible[s, :, None]
        share = np.where(nearer, ix*iy, 0.0)/np.maximum((w*height)[s, :, None], 1e-9)
        cover[s] = np.nan_to_num(share).max(axis=2)
    return cover


def switches(truth, teams, visible, dt, rng):
    """Identity map (T, P): whose observation each identity carries (itself unless two team-mates crossed and swapped)."""
    T, P, _ = truth.shape
    who = np.tile(np.arange(P), (T, 1))
    close_before = np.zeros((P, P), bool)
    busy_until = np.full(P, -1)
    events = 0
    for t in range(T):
        d = np.linalg.norm(truth[t, :, None]-truth[t, None, :], axis=-1)
        close = (d < SWITCH_DISTANCE) & (teams[:, None] == teams[None, :]) & visible[t, :, None] & visible[t, None, :]
        np.fill_diagonal(close, False)
        for i, j in zip(*np.nonzero(np.triu(close & ~close_before))):
            if busy_until[i] < t and busy_until[j] < t and rng.random() < SWITCH_PROB:
                n = int(round(rng.uniform(*SWITCH_SECONDS)/dt))
                who[t:t+n, i], who[t:t+n, j] = j, i
                busy_until[[i, j]] = t+n
                events += 1
        close_before = close
    return who, events


def episodes(visible, occluded, dt, rate, rng, draws=1):
    """A per-person event process (the base events' rule: OCCLUDED_EVENTS times likelier under occlusion, EVENT_SECONDS
    long): (on (T, P), draws (T, P, draws) uniform numbers fixed per episode)."""
    T, P = visible.shape
    on = np.zeros((T, P), bool)
    u = np.zeros((T, P, draws))
    left = np.zeros(P, int)
    current = np.zeros((P, draws))
    for t in range(T):
        start = visible[t] & (left == 0) & (rng.random(P) < rate*dt*np.where(occluded[t], OCCLUDED_EVENTS, 1.0))
        left[start] = np.maximum(1, np.round(rng.uniform(*EVENT_SECONDS, P)/dt)).astype(int)[start]
        current[start] = rng.random((P, draws))[start]
        live = left > 0
        on[t, live] = True
        u[t, live] = current[live]
        left[live] -= 1
    return on, u


def calibration_error(T, dt, smooth, rng):
    """(T,) pixels: the anchors' errors (CAL_*), held until the next anchor or joined to it by a straight line (smooth),
    plus the Ornstein-Uhlenbeck wobble."""
    starts = [0]
    while starts[-1] < T:
        starts.append(starts[-1]+max(1, int(round(rng.uniform(*CAL_ANCHORS)/dt))))
    starts = np.array(starts)
    levels = rng.normal(0, 1, len(starts))*np.where(rng.random(len(starts)) < CAL_BIG_SHARE, CAL_BIG_PX, CAL_PX)
    t = np.arange(T)
    held = np.interp(t, starts, levels) if smooth else levels[np.searchsorted(starts, t, side='right')-1]
    a = math.exp(-dt/CAL_TAU)
    kicks = rng.normal(0, CAL_WOBBLE_PX*math.sqrt(1-a*a), T)
    wobble = np.empty(T)
    wobble[0] = rng.normal(0, CAL_WOBBLE_PX)
    for k in range(1, T):
        wobble[k] = a*wobble[k-1]+kicks[k]
    return held+wobble


def miscalibrated(H, K, dt, smooth, rng):
    """The frames' homographies (T, 3, 3) with a calibration error: the camera tilted by calibration_error pixels over
    the focal length, and panned by half as much (an independent draw)."""
    T = len(H)
    tilt = calibration_error(T, dt, smooth, rng)/K[0, 0]
    pan = .5*calibration_error(T, dt, smooth, rng)/K[0, 0]
    Rx, Ry = np.zeros((T, 3, 3)), np.zeros((T, 3, 3))
    Rx[:, 0, 0], Rx[:, 1, 1], Rx[:, 1, 2], Rx[:, 2, 1], Rx[:, 2, 2] = 1, np.cos(tilt), -np.sin(tilt), np.sin(tilt), np.cos(tilt)
    Ry[:, 0, 0], Ry[:, 0, 2], Ry[:, 1, 1], Ry[:, 2, 0], Ry[:, 2, 2] = np.cos(pan), np.sin(pan), 1, -np.sin(pan), np.cos(pan)
    out = K @ Ry @ Rx @ np.linalg.inv(K) @ H
    return out/out[:, 2:3, 2:3]


def image_motion(H):
    """(T, 2, 3) the camera's true image motion from the previous frame to each frame (pixels to pixels), the affine part
    as the tracker writes it (frames[].motion): a least-squares fit over MOTION_GRID. Row 0 is NaN (no previous frame)."""
    gx, gy = np.meshgrid(MOTION_GRID[0]*IMAGE[0], MOTION_GRID[1]*IMAGE[1])
    pts = np.c_[gx.ravel(), gy.ravel(), np.ones(gx.size)]                     # (N, 3)
    M = H[1:] @ np.linalg.inv(H[:-1])                                          # the previous frame's pixels -> this frame's
    q = np.einsum('tij,nj->tni', M, pts)
    q = q[..., :2]/q[..., 2:]
    out = np.full((len(H), 2, 3), np.nan)
    out[1:] = np.swapaxes(np.linalg.pinv(pts) @ q, 1, 2)
    return out


def simulate(truth, teams, ball_x, dt, camera, rng, scenario=(), extra=None):
    """One window seen through a camera: the observations and everything the result and the realism block need.
    scenario: the extra event kinds and calibration error of SCENARIOS, each drawn from extra(kind), a generator of its
    own (the base draws are the same in every scenario)."""
    T, P, _ = truth.shape
    H, R, C, K = camera_views(camera, ball_x, dt)
    ground = np.concatenate([np.nan_to_num(truth), np.zeros((T, P, 1))], -1)
    head = ground-[0.0, 0.0, PLAYER_HEIGHT]
    foot_px, depth = project(R, C, K, ground)
    head_px, _ = project(R, C, K, head)
    height = foot_px[..., 1]-head_px[..., 1]
    on_pitch = np.isfinite(truth[..., 0])
    in_image = (foot_px[..., 0] >= 0) & (foot_px[..., 0] < IMAGE[0]) & (foot_px[..., 1] >= 0) & (foot_px[..., 1] < IMAGE[1])
    visible = on_pitch & (depth > 0) & in_image & (height >= MIN_BOX)
    cover = occlusion(foot_px, height, depth, visible)
    occluded = visible & (cover > OCCLUDED)
    # Per occlusion episode one noise factor; events and misses as per-person processes.
    gain = np.ones((T, P))
    for p in range(P):
        for a, b in runs(occluded[:, p]):
            gain[a:b, p] = rng.uniform(*OCCLUDED_GAIN)
    event = np.zeros((T, P))          # the bottom edge's upward shift in box heights (0: none)
    missed = np.zeros((T, P), bool)
    left_event, left_miss = np.zeros(P, int), np.zeros(P, int)
    shift = np.zeros(P)
    for t in range(T):
        rate = EVENT_RATE*dt*np.where(occluded[t], OCCLUDED_EVENTS, 1.0)
        start = visible[t] & (left_event == 0) & (rng.random(P) < rate)
        left_event[start] = np.maximum(1, np.round(rng.uniform(*EVENT_SECONDS, P)/dt)).astype(int)[start]
        shift[start] = rng.uniform(*EVENT_SHIFT, P)[start]
        on = left_event > 0
        event[t, on] = shift[on]
        left_event[on] -= 1
        start = visible[t] & (left_miss == 0) & (rng.random(P) < MISS_RATE*dt)
        left_miss[start] = rng.integers(MISS_FRAMES[0], MISS_FRAMES[1]+1, P)[start]
        on = left_miss > 0
        missed[t, on] = True
        left_miss[on] -= 1
    in_event = visible & (event > 0)
    p_low = np.where(in_event, LOW_SCORE['event'], np.where(occluded, LOW_SCORE['occluded'], LOW_SCORE['clean']))
    score = np.where(rng.random((T, P)) < p_low, rng.uniform(.25, .5, (T, P)), rng.uniform(.5, .95, (T, P)))
    sigma = np.sqrt(CORE_PX**2+(CORE_REL*height)**2)*gain
    bottom = foot_px[..., 1]+rng.normal(size=(T, P))*sigma-event*height
    top = head_px[..., 1]+rng.normal(size=(T, P))*sigma
    centre = foot_px[..., 0]+rng.normal(size=(T, P))*sigma/math.sqrt(2)
    low = visible & (score < .5)
    score_cue = (round(float((in_event & low).sum()/max(1, low.sum())), 3), round(float((in_event & ~low).sum()/max(1, (visible & ~low).sum())), 3))
    # The scenario's extra box events (EXTRA_EVENTS) and calibration error.
    in_extra, shares = np.zeros((T, P), bool), {}
    for kind in scenario:
        if kind not in EXTRA_EVENTS:
            continue
        g = extra(kind)
        rate, (lo, hi) = EXTRA_EVENTS[kind]
        on, u = episodes(visible, occluded, dt, rate, g, 3)
        move = np.where(on, lo+(hi-lo)*u[..., 0], 0.0)*height
        if kind == 'top':
            top = top-move
        elif kind == 'shift':
            move = np.where(u[..., 1] < .5, -move, move)
            top, bottom = top+move, bottom+move
            centre = centre+np.where(on, u[..., 2]-.5, 0.0)*BOX_ASPECT*height
        elif kind == 'torso':
            bottom, top = bottom-move, top+move/2
        else:
            bottom = bottom+move
        score = np.where(on & (g.random((T, P)) < EXTRA_LOW_SCORE), g.uniform(.25, .5, (T, P)), score)
        in_extra |= visible & on
        shares[kind] = round(float((visible & on).sum()/max(1, visible.sum())), 4)
    calibration = next((k for k in scenario if k in ('steps', 'drift')), None)
    H_used = H if calibration is None else miscalibrated(H, K, dt, calibration == 'drift', extra(calibration))
    detected = visible & ~missed & (cover <= HIDDEN) & (bottom-top >= 2)
    who, n_switch = switches(np.nan_to_num(truth, nan=1e6), teams, detected, dt, rng)
    take = lambda a: np.take_along_axis(a, who, axis=1)
    centre, bottom, top, score, detected = take(centre), take(bottom), take(top), take(score), take(detected)
    sigma_core, width = take(np.sqrt(CORE_PX**2+(CORE_REL*height)**2)), take(BOX_ASPECT*height)
    pitch, J = back_project(H_used, np.stack([centre, bottom], -1))
    inside = (pitch[..., 0] >= -.15*LENGTH) & (pitch[..., 0] <= 1.15*LENGTH) & (pitch[..., 1] >= -.15*WIDTH) & (pitch[..., 1] <= 1.15*WIDTH)
    observed = detected & np.isfinite(pitch[..., 0]) & inside
    obs = np.where(observed[..., None], pitch, np.nan)
    # The measurement covariance a smoother may know: the core pixel noise through the Jacobian.
    S_px = np.zeros((T, P, 2, 2))
    S_px[..., 0, 0], S_px[..., 1, 1] = (sigma_core/math.sqrt(2))**2, sigma_core**2
    Rm = J @ S_px @ np.swapaxes(J, -1, -2)
    n_vis = max(1, int(visible.sum()))
    m_per_px = np.linalg.norm(J[..., :, 1], axis=-1)[observed]
    realism = {'visibleShare': round(float(visible.sum()/max(1, on_pitch.sum())), 3), 'observedShare': round(float(observed.sum()/n_vis), 3),
               'occludedShare': round(float(occluded.sum()/n_vis), 4), 'eventShare': round(float(in_event.sum()/n_vis), 4),
               'missedShare': round(float((visible & missed).sum()/n_vis), 4), 'switches': n_switch,
               'switchedShare': round(float(((who != np.arange(P)) & observed).sum()/max(1, observed.sum())), 4),
               'eventGivenLowScore': score_cue[0], 'eventGivenHighScore': score_cue[1],
               'boxHeightMedian_px': round(float(np.median(height[visible])), 1) if visible.any() else None,
               'metresPerVerticalPixelMedian': round(float(np.median(m_per_px)), 3) if len(m_per_px) else None,
               'extraShare': shares}
    if calibration is not None:
        # How far the miscalibrated homographies put a noiseless foot point (metres, visible people).
        off = np.linalg.norm(back_project(H_used, foot_px)[0]-np.nan_to_num(truth), axis=-1)[visible]
        off = off[np.isfinite(off)]
        realism['calibrationError_m'] = {'rms': round(float(np.sqrt(np.mean(off**2))), 3), 'p95': round(float(np.percentile(off, 95)), 3)} if len(off) else None
    # The box keeps its width when an event takes its legs.
    boxes = np.stack([(centre-width/2)/IMAGE[0], top/IMAGE[1], width/IMAGE[0], (bottom-top)/IMAGE[1]], -1)
    return {'obs': obs, 'R': Rm, 'observed': observed, 'who': who, 'event': take(in_event), 'extra': take(in_extra), 'H': H_used, 'motion': image_motion(H),
            'boxes': boxes, 'score': score, 'realism': realism}


# ---------- the methods ----------
def bridgeable(observed, dt):
    """Frames (T, P) inside a gap of at most BRIDGE_GAP between two observations."""
    T, P = observed.shape
    idx = np.arange(T)[:, None]
    prev = np.maximum.accumulate(np.where(observed, idx, -1), axis=0)
    nxt = np.minimum.accumulate(np.where(observed, idx, T)[::-1], axis=0)[::-1]
    return ~observed & (prev >= 0) & (nxt < T) & ((nxt-prev)*dt <= BRIDGE_GAP+1e-9)


def bridge(est, edges, observed, dt):
    """est with the bridgeable gaps filled by straight lines between the edge values (edges: (T, P, 2) at observations)."""
    out = est.copy()
    gaps = bridgeable(observed, dt)
    t = np.arange(len(observed))
    for p in range(observed.shape[1]):
        if not gaps[:, p].any():
            continue
        k = np.flatnonzero(observed[:, p])
        g = np.flatnonzero(gaps[:, p])
        for d in range(2):
            out[g, p, d] = np.interp(t[g], k, edges[k, p, d])
    return out


def renderer(obs, observed, dt):
    """tactical.js today: Track.smoothObserved on the observations (+-RENDERER_WINDOW/2, stopping at the first gap), the
    gaps straight between the raw observations."""
    half = int(math.floor(RENDERER_WINDOW/2/dt+1e-9))
    filled = np.nan_to_num(obs)
    total = np.where(observed[..., None], filled, 0.0)
    count = observed.astype(np.float64)
    for later in (False, True):
        chain = observed.copy()
        for o in range(1, half+1):
            # The observation o frames earlier (later), counted while every one in between was observed too.
            neighbour, values = np.zeros_like(observed), np.zeros_like(filled)
            if later:
                neighbour[:-o], values[:-o] = observed[o:], filled[o:]
            else:
                neighbour[o:], values[o:] = observed[:-o], filled[:-o]
            chain &= neighbour
            total += np.where(chain[..., None], values, 0.0)
            count += chain
    est = np.where(observed[..., None], total/np.maximum(count, 1)[..., None], np.nan)
    return bridge(est, obs, observed, dt)


def savgol(obs, observed, dt, seconds):
    """Savitzky-Golay (order 2) on each contiguous run of observations, the gaps straight between the smoothed ends."""
    est = np.full_like(obs, np.nan)
    window = int(round(seconds/dt)) | 1
    for p in range(observed.shape[1]):
        for a, b in runs(observed[:, p]):
            n = b-a
            w = min(window, n if n % 2 else n-1)
            est[a:b, p] = savgol_filter(obs[a:b, p], w, 2, axis=0, mode='interp') if w >= 3 else obs[a:b, p]
    return bridge(est, est, observed, dt)


def rts(obs, R, observed, dt, q=RTS_Q):
    """Constant-velocity Kalman filter + RTS smoother on every identity at once, fixed process noise q; a gap longer
    than BRIDGE_GAP starts a new segment. Positions where observed or bridgeable."""
    T, P, _ = obs.shape
    F = np.eye(4)
    F[0, 2] = F[1, 3] = dt
    Q1 = q*np.array([[dt**3/3, dt**2/2], [dt**2/2, dt]])
    Q = np.zeros((4, 4))
    Q[np.ix_([0, 2], [0, 2])] = Q1
    Q[np.ix_([1, 3], [1, 3])] = Q1
    xf, Pf = np.zeros((T, P, 4)), np.zeros((T, P, 4, 4))
    xp, Pp = np.zeros((T, P, 4)), np.zeros((T, P, 4, 4))
    start = np.zeros((T, P), bool)     # a segment starts here: the smoother does not look across
    x, Pc = np.zeros((P, 4)), np.tile(np.eye(4), (P, 1, 1))
    alive = np.zeros(P, bool)
    last = np.full(P, -10**9)
    for t in range(T):
        x = x @ F.T
        Pc = F @ Pc @ F.T+Q
        xp[t], Pp[t] = x, Pc
        o = observed[t]
        new = o & (~alive | ((t-last)*dt > BRIDGE_GAP+1e-9))
        if new.any():
            x[new] = np.c_[obs[t, new], np.zeros((int(new.sum()), 2))]
            P0 = np.zeros((int(new.sum()), 4, 4))
            P0[:, :2, :2] = R[t, new]
            P0[:, 2, 2] = P0[:, 3, 3] = RTS_V0**2
            Pc[new] = P0
            start[t, new] = True
            alive |= new
        upd = o & ~new
        if upd.any():
            S = Pc[upd][:, :2, :2]+R[t, upd]
            Kg = Pc[upd][:, :, :2] @ np.linalg.inv(S)
            y = obs[t, upd]-x[upd, :2]
            x[upd] = x[upd]+np.einsum('pij,pj->pi', Kg, y)
            Pc[upd] = Pc[upd]-Kg @ Pc[upd][:, :2, :]
        last = np.where(o, t, last)
        xf[t], Pf[t] = x, Pc
    xs = xf.copy()
    for t in range(T-2, -1, -1):
        cont = ~start[t+1]
        if not cont.any():
            continue
        G = Pf[t, cont] @ F.T @ np.linalg.inv(Pp[t+1, cont])
        xs[t, cont] = xf[t, cont]+np.einsum('pij,pj->pi', G, xs[t+1, cont]-xp[t+1, cont])
    keep = observed | bridgeable(observed, dt)
    return np.where(keep[..., None], xs[..., :2], np.nan)


def as_result(sim, names, teams, times, rate):
    """The simulated window as a PitchIQ result for soccer/trajectory.py: frames with the homography the observations
    were projected through, the true camera motion from the previous frame (the tracker's motion row) and the observed
    people (box, score, local track, identity, pitch), the roster."""
    frames = []
    obs, observed, boxes, score, H = sim['obs'], sim['observed'], sim['boxes'], sim['score'], sim['H']
    motion = sim.get('motion')
    for t in range(len(times)):
        people = []
        for p in np.flatnonzero(observed[t]):
            people.append({'track': int(p+1), 'id': names[p], 'display': names[p], 'team': teams[p], 'role': 'PLAYER', 'state': 'confirmed',
                           'cls': 'player', 'evidence': 'observed', 'identityConfidence': .9, 'box': [round(float(v), 5) for v in boxes[t, p]],
                           'score': round(float(score[t, p]), 3), 'pitch': [round(float(obs[t, p, 0]/LENGTH), 4), round(float(obs[t, p, 1]/WIDTH), 4)],
                           'pitchConfidence': .8})
        moved = motion is not None and np.isfinite(motion[t]).all()
        frames.append({'time': round(float(times[t]), 4), 'frame': t, 'people': people, 'calibrated': True, 'cameraReliable': True, 'cut': False,
                       'motion': [round(float(v), 6) for v in motion[t].ravel()] if moved else None,
                       'calibration': {'H': H[t].ravel().tolist(), 'confidence': .8, 'source': 'landmarks'}})
    players = [{'id': n, 'display': n, 'team': tm, 'role': 'PLAYER'} for n, tm in zip(names, teams)]
    return {'version': 3, 'video': {'width': IMAGE[0], 'height': IMAGE[1], 'fps': rate}, 'frames': frames,
            'match': {'players': players, 'goalkeepers': []},
            'roster': {'slots': [{'id': n, 'team': tm, 'role': 'PLAYER', 'confirmed': True} for n, tm in zip(names, teams)]},
            'tactical': {'pitchSize': {'length': LENGTH, 'width': WIDTH, 'source': 'metrica'}}}


def stabilizer(sim, names, teams, times, rate):
    """soccer/trajectory.stabilize on the window written as a result: (T, P, 2) metres, NaN where it gives no position.
    What it found is left in sim['stabilizer']: its registration steps and its flags by reason."""
    from soccer import field, trajectory
    out = trajectory.stabilize(as_result(sim, names, teams, times, rate), IMAGE[0], IMAGE[1], field.PitchSize(LENGTH, WIDTH))
    flags = {}
    for f in out.get('flags') or []:
        reason = str(f.get('reason') or '?').split(':')[0]
        flags[reason] = flags.get(reason, 0)+1
    sim['stabilizer'] = {'registrationSteps': (out.get('noise') or {}).get('registrationSteps'), 'flags': flags}
    est = np.full(sim['obs'].shape, np.nan)
    for p, name in enumerate(names):
        cols = (out.get('ids') or {}).get(name)
        if cols is None:
            continue
        est[:, p, 0] = [np.nan if v is None else v for v in cols['x']]
        est[:, p, 1] = [np.nan if v is None else v for v in cols['y']]
    return est


def smooth_tracks(sim, times):
    """soccer/trajectory.smooth_tracks, the stabilizer's array-level smoother, on the observations with the rts baseline's
    measurement covariance (no box-event repair, no fitted noise model): what its robust adaptive smoothing adds."""
    from soccer import trajectory
    out = trajectory.smooth_tracks(times, sim['obs'], sim['R'])
    return np.where((out['kind'] != '')[..., None], out['x'], np.nan)


# ---------- the metrics ----------
def central_velocity(p, m, dt):
    """(T, P, 2) central differences over +-m frames, NaN at the ends and wherever a side is missing."""
    v = np.full_like(p, np.nan)
    if len(p) > 2*m:
        v[m:-m] = (p[2*m:]-p[:-2*m])/(2*m*dt)
    return v


def high_pass(v, half):
    """(T, P, 2) velocities minus their centred mean over +-half frames: the changes of velocity. NaN where the window
    is not whole."""
    T = len(v)
    ok = np.isfinite(v[..., 0])
    c = np.concatenate([np.zeros((1,)+v.shape[1:]), np.cumsum(np.where(ok[..., None], v, 0.0), axis=0)])
    n = np.concatenate([np.zeros((1, v.shape[1])), np.cumsum(ok, axis=0)])
    lo, hi = np.clip(np.arange(T)-half, 0, T), np.clip(np.arange(T)+half+1, 0, T)
    count = n[hi]-n[lo]
    whole = ok & (count == (hi-lo)[:, None])
    return np.where(whole[..., None], v-(c[hi]-c[lo])/np.maximum(count, 1)[..., None], np.nan)


def second_difference(p, m, dt):
    a = np.full(p.shape[:2], np.nan)
    if len(p) > 2*m:
        a[m:-m] = np.linalg.norm(p[2*m:]-2*p[m:-m]+p[:-2*m], axis=-1)/(m*dt)**2
    return a


def stops(truth, speed, dt):
    """(frame, person, run direction) of the true stops: the speed falls under STOP_AT after reaching STOP_FROM within
    the second before, and stays under STOP_HOLD for STOP_SECONDS."""
    T = len(truth)
    before, after = int(round(1.0/dt)), int(round(STOP_SECONDS/dt))
    out = []
    if T <= before+after+1:
        return out
    fast = np.nan_to_num(speed) >= STOP_FROM
    for t, p in zip(*np.nonzero((speed < STOP_AT) & (np.roll(speed, 1, axis=0) >= STOP_AT))):
        if t < before or t+after >= T or not fast[t-before:t, p].any() or not (np.nanmax(speed[t:t+after+1, p]) < STOP_HOLD):
            continue
        d = truth[t, p]-truth[t-before, p]
        n = np.linalg.norm(d)
        if np.isfinite(n) and n > 0:
            out.append((int(t), int(p), d/n))
    return out


def turns(truth, dt):
    """(T, P) True within TURN_NEAR of a direction change."""
    T, P, _ = truth.shape
    h, near = int(round(TURN_HALF/dt)), int(round(TURN_NEAR/dt))
    angle = np.full((T, P), np.nan)
    if T > 2*h:
        a = (truth[h:-h]-truth[:-2*h])/(h*dt)
        b = (truth[2*h:]-truth[h:-h])/(h*dt)
        na, nb = np.linalg.norm(a, axis=-1), np.linalg.norm(b, axis=-1)
        cos = (a*b).sum(-1)/np.maximum(na*nb, 1e-9)
        angle[h:-h] = np.where((na > TURN_SPEED) & (nb > TURN_SPEED), np.degrees(np.arccos(np.clip(cos, -1, 1))), np.nan)
    turn = np.nan_to_num(angle) > TURN_ANGLE
    out = np.zeros((T, P), bool)
    for t, p in zip(*np.nonzero(turn)):
        lo, hi = max(0, t-h), min(T, t+h+1)
        if np.nan_to_num(angle[lo:hi, p]).argmax()+lo == t:
            out[max(0, t-near):t+near+1, p] = True
    return out


class Score:
    """One method's errors pooled over the windows of a camera."""

    def __init__(self, shifts):
        self.lists = {k: [] for k in ('err', 'gapErr', 'switchedErr', 'speedErr', 'turnErr', 'stillStep', 'acc', 'glide')}
        self.buckets = {name: [] for name, _, _ in SPEED_BUCKETS}
        self.corr = np.zeros((len(shifts), 3))
        self.shifts = shifts
        self.covered = np.zeros(2)
        self.total = np.zeros(2)

    def add(self, est, truth, observed, gaps, w):
        """One window. w: the truth's speed, velocity, turns and stops, the lags, and 'own' (T, P): the identity carries
        its own observation (not another player's after a switch)."""
        e = np.linalg.norm(est-truth, axis=-1)
        valid = np.isfinite(truth[..., 0]) & w['own']
        on, between = observed & valid, gaps & valid
        has = np.isfinite(est[..., 0])
        self.total += [on.sum(), between.sum()]
        self.covered += [(on & has).sum(), (between & has).sum()]
        self.lists['err'].append(e[on & has])
        self.lists['gapErr'].append(e[between & has])
        self.lists['switchedErr'].append(e[observed & has & np.isfinite(truth[..., 0]) & ~w['own']])
        for name, lo, hi in SPEED_BUCKETS:
            self.buckets[name].append(e[on & has & (w['speed'] >= lo) & (w['speed'] < hi)])
        self.lists['turnErr'].append(e[on & has & w['turn']])
        v = central_velocity(est, w['m'], w['dt'])
        s = np.linalg.norm(v, axis=-1)
        ok = on & np.isfinite(s) & np.isfinite(w['speed'])
        self.lists['speedErr'].append((s-w['speed'])[ok])
        still = (w['speed'] < STILL) & np.roll(w['speed'] < STILL, 1, axis=0) & valid & np.roll(valid, 1, axis=0)
        step = np.full(e.shape, np.nan)
        step[1:] = np.linalg.norm(est[1:]-est[:-1], axis=-1)
        self.lists['stillStep'].append(step[still & np.isfinite(step)])
        a = second_difference(est, w['ma'], w['dt'])
        self.lists['acc'].append(a[on & np.isfinite(a)])
        half = int(round(LAG_HIGH_PASS/w['dt']))
        vt = high_pass(np.where(valid[..., None], w['velocity'], np.nan), half)
        vh = high_pass(v, half)
        for k, s_ in enumerate(self.shifts):
            ve = vh[max(0, s_):len(vh)+min(0, s_)]
            vv = vt[max(0, -s_):len(vt)-max(0, s_)]
            ok = np.isfinite(ve[..., 0]) & np.isfinite(vv[..., 0])
            self.corr[k] += [(ve[ok]*vv[ok]).sum(), (ve[ok]**2).sum(), (vv[ok]**2).sum()]
        after = int(round(STOP_SECONDS/w['dt']))
        for t, p, d in w['stops']:
            seg = est[t:t+after+1, p]
            if np.isfinite(seg[:, 0]).all() and valid[t:t+after+1, p].all():
                self.lists['glide'].append(float(((seg-truth[t, p])@d).max()-((truth[t:t+after+1, p]-truth[t, p])@d).max()))

    def summary(self, dt):
        cat = {k: (np.concatenate([np.atleast_1d(x) for x in v]) if v else np.zeros(0)) for k, v in self.lists.items()}
        rmse = lambda a: round(float(np.sqrt(np.mean(a**2))), 3) if len(a) else None
        pct = lambda a, q, scale=1.0: round(float(np.percentile(a, q))*scale, 3) if len(a) else None
        r = self.corr[:, 0]/np.sqrt(np.maximum(self.corr[:, 1]*self.corr[:, 2], 1e-12))
        k = int(np.argmax(r))
        frac = 0.0
        if 0 < k < len(r)-1:
            den = r[k-1]-2*r[k]+r[k+1]
            frac = .5*(r[k-1]-r[k+1])/den if den else 0.0
        speed = cat['speedErr']
        return {'coverage': {'observed': round(float(self.covered[0]/max(1, self.total[0])), 4), 'gaps': round(float(self.covered[1]/max(1, self.total[1])), 4)},
                'rmse': rmse(cat['err']), 'median': pct(cat['err'], 50), 'p95': pct(cat['err'], 95),
                'byTrueSpeed': {n: rmse(np.concatenate(v)) for n, v in self.buckets.items()}, 'gapRmse': rmse(cat['gapErr']), 'switchedRmse': rmse(cat['switchedErr']),
                'speedErrorMean': round(float(np.mean(np.abs(speed))), 3) if len(speed) else None, 'speedBias': round(float(np.mean(speed)), 3) if len(speed) else None,
                'lag_ms': round(float((self.shifts[k]+frac)*dt*1000), 1), 'velocityCorrelation': round(float(r[k]), 4),
                'stopGlide': {'n': int(len(cat['glide'])), 'p50': pct(cat['glide'], 50), 'p90': pct(cat['glide'], 90)},
                'turnRmse': rmse(cat['turnErr']), 'turnMedian': pct(cat['turnErr'], 50),
                'stationaryStep_cm': {'p50': pct(cat['stillStep'], 50, 100), 'p90': pct(cat['stillStep'], 90, 100)}, 'accelerationP95': pct(cat['acc'], 95)}


# ---------- the bench ----------
def periods(path):
    """The period column of a Metrica tracking CSV (rows after the three header lines)."""
    with open(path, encoding='utf-8') as fh:
        return np.array([int(line.split(',', 1)[0]) for k, line in enumerate(fh) if k >= 3 and line.strip()])


def load_game(data, game):
    """Both teams stacked: (times, truth (T, P, 2) metres, team per column, names, ball x, period per row)."""
    parts = []
    for team, side in (('A', 'Home'), ('B', 'Away')):
        parts.append((team, load_team(Path(data)/f'Sample_Game_{game}_RawTrackingData_{side}_Team.csv')))
    times, _, ball, _ = parts[0][1]
    truth = np.concatenate([p[1][1] for p in parts], axis=1)
    teams = np.array([team for team, p in parts for _ in range(p[1].shape[1])])
    names = [f'{team}-{k+1:02d}' for team, p in parts for k in range(p[1].shape[1])]
    period = periods(Path(data)/f'Sample_Game_{game}_RawTrackingData_Home_Team.csv')
    if len(period) != len(times):
        period = np.ones(len(times), int)
    return times, truth, teams, names, ball[:, 0], period


def windows(times, period, count, length):
    """[(t0, t1)] spread evenly over the periods, never across one's end."""
    out = []
    spans = [(times[period == k][0], times[period == k][-1]) for k in np.unique(period)]
    per = [count//len(spans)+(1 if k < count % len(spans) else 0) for k in range(len(spans))]
    for (a, b), n in zip(spans, per):
        for k in range(n):
            centre = a+(k+.5)*(b-a)/n
            t0 = float(np.clip(centre-length/2, a, max(a, b-length)))
            out.append((t0, min(b, t0+length)))
    return out


def resample(times, values, grid):
    """Linear interpolation of (T, ...) values onto the grid (NaN where either neighbour is NaN)."""
    k = np.clip(np.searchsorted(times, grid, side='right')-1, 0, len(times)-2)
    u = ((grid-times[k])/(times[k+1]-times[k])).reshape((-1,)+(1,)*(values.ndim-1))
    return values[k]*(1-u)+values[k+1]*u


def glitches(times, truth):
    """(T, P) True within GLITCH_HALF of a one-frame step of the truth faster than GLITCH_SPEED (the module docstring)."""
    T, P, _ = truth.shape
    bad = np.zeros((T, P), bool)
    if T < 2:
        return bad
    with np.errstate(invalid='ignore', divide='ignore'):
        step = np.linalg.norm(np.diff(truth, axis=0), axis=-1)/np.diff(times)[:, None]
    for t, p in zip(*np.nonzero(np.nan_to_num(step) > GLITCH_SPEED)):
        lo = np.searchsorted(times, times[t]-GLITCH_HALF, side='left')
        hi = np.searchsorted(times, times[t+1]+GLITCH_HALF, side='right')
        bad[lo:hi, p] = True
    return bad


def camera_key(camera):
    """The camera's part of the seed: fixed by its name, not by its place in --camera (a camera's numbers do not depend
    on which others run with it)."""
    return zlib.crc32(camera.encode('utf-8'))


SPREAD = (('rmse', ('rmse',)), ('sprinting', ('byTrueSpeed', 'sprinting')), ('turnRmse', ('turnRmse',)), ('lag_ms', ('lag_ms',)),
          ('stopGlideP50', ('stopGlide', 'p50')), ('stationaryStepP50_cm', ('stationaryStep_cm', 'p50')))


def spread(summaries):
    """Mean, standard deviation, min and max over the seeds' summaries of the SPREAD metrics."""
    out = {}
    for name, path in SPREAD:
        values = []
        for s in summaries:
            for key in path:
                s = s.get(key) if isinstance(s, dict) else None
            if s is not None:
                values.append(float(s))
        if values:
            v = np.asarray(values)
            out[name] = {'n': len(v), 'mean': round(float(v.mean()), 4), 'sd': round(float(v.std(ddof=1)), 4) if len(v) > 1 else 0.0,
                         'min': round(float(v.min()), 4), 'max': round(float(v.max()), 4)}
    return out


def _mean_realism(realism):
    """The windows' (and seeds') realism blocks averaged: numbers by their mean, the extra shares per kind, the
    calibration error's rms and p95, the stabilizer's registration-step frames summed and its flags added up."""
    out = {}
    for k in realism[0]:
        if k in ('stabilizerRegistrationSteps', 'stabilizerFlags'):
            continue
        values = [r.get(k) for r in realism if r.get(k) is not None]
        if not values:
            out[k] = None
        elif isinstance(values[0], dict):
            keys = {key for v in values for key in v}
            out[k] = {key: round(float(np.mean([v[key] for v in values if v.get(key) is not None])), 4) for key in sorted(keys)}
        else:
            out[k] = round(float(np.mean(values)), 4)
    out['switches'] = int(sum(r['switches'] for r in realism))
    steps = [r.get('stabilizerRegistrationSteps') for r in realism if r.get('stabilizerRegistrationSteps')]
    if steps:
        out['stabilizerRegistrationSteps'] = {'frames': int(sum(s.get('frames') or 0 for s in steps)),
                                              'largestMetres': max((s.get('largestMetres') or 0.0 for s in steps), default=None)}
    flags = {}
    for r in realism:
        for key, n in (r.get('stabilizerFlags') or {}).items():
            flags[key] = flags.get(key, 0)+n
    if flags:
        out['stabilizerFlags'] = dict(sorted(flags.items(), key=lambda kv: -kv[1]))
    return out


def run(data, games, cameras, methods, n_windows=4, length=300.0, rate=30.0, seed=0, q=RTS_Q, progress=print, scenarios=DEFAULT_SCENARIOS, seeds=1,
        clean=True):
    """The report: per game, scenario and camera the simulation's realism and every method's metrics (Score.summary),
    pooled over the windows and seeds (seed, seed+1, ... seed+seeds-1), with their spread across the seeds when there are
    several. clean: the truth's glitches are left out (glitches()). A method whose module cannot be imported is reported
    missing."""
    report = {'settings': {'cameras': cameras, 'methods': methods, 'windows': n_windows, 'length': length, 'rate': rate, 'seed': seed, 'seeds': seeds,
                           'rtsQ': q, 'scenarios': list(scenarios), 'glitchesLeftOut': clean, 'glitchSpeed': GLITCH_SPEED, 'glitchHalf': GLITCH_HALF},
              'games': {}}
    dt = 1.0/rate
    shifts = list(range(-int(round(LAG_MAX/dt)), int(round(LAG_MAX/dt))+1))
    for game in games:
        times, truth_raw, teams, names, ball_all, period = load_game(data, game)
        bad = glitches(times, truth_raw) if clean else np.zeros(truth_raw.shape[:2], bool)
        truth_all = np.where(bad[..., None], np.nan, truth_raw)
        spans = windows(times, period, n_windows, length)
        entry = {'windows': [[round(a, 2), round(b, 2)] for a, b in spans], 'scenarios': {}}
        for scenario in scenarios:
            kinds = SCENARIOS[scenario]
            entry['scenarios'][scenario] = {'cameras': {}}
            for camera in cameras:
                pooled = {m: Score(shifts) for m in methods}
                per_seed = {m: [] for m in methods}
                realism, missing, seconds = [], {}, dict.fromkeys(methods, 0.0)
                for s in range(seed, seed+seeds):
                    scores = {m: Score(shifts) for m in methods} if seeds > 1 else {}
                    for wi, (a, b) in enumerate(spans):
                        grid = np.arange(a, b, dt)
                        truth = resample(times, truth_all, grid)
                        ball_x = resample(times, ball_all, grid)
                        key = [s, game, camera_key(camera), wi]
                        rng = np.random.default_rng(key)
                        sim = simulate(truth, teams, ball_x, dt, camera, rng, kinds, lambda kind, key=key: np.random.default_rng(key+[zlib.crc32(kind.encode('utf-8'))]))
                        realism.append(sim['realism'])
                        known = np.isfinite(resample(times, truth_raw, grid)[..., 0])
                        realism[-1]['truthGlitchShare'] = round(float((known & ~np.isfinite(truth[..., 0])).sum()/max(1, known.sum())), 5)
                        m = int(round(DIFF/dt))
                        velocity = central_velocity(truth, m, dt)
                        speed = np.linalg.norm(velocity, axis=-1)
                        w = {'dt': dt, 'm': m, 'ma': int(round(ACC_LAG/dt)), 'speed': speed, 'velocity': velocity, 'turn': turns(truth, dt),
                             'stops': stops(truth, speed, dt), 'own': sim['who'] == np.arange(truth.shape[1])}
                        gaps = bridgeable(sim['observed'], dt)
                        for method in methods:
                            if method in missing:
                                continue
                            started = time.perf_counter()
                            try:
                                est = estimate(method, sim, names, teams, grid, rate, dt, q)
                            except ImportError as error:
                                missing[method] = f'soccer/trajectory.py not importable ({error})'
                                continue
                            seconds[method] += time.perf_counter()-started
                            pooled[method].add(est, truth, sim['observed'], gaps, w)
                            if scores:
                                scores[method].add(est, truth, sim['observed'], gaps, w)
                        if 'stabilizer' in sim:
                            realism[-1]['stabilizerRegistrationSteps'] = sim['stabilizer']['registrationSteps']
                            realism[-1]['stabilizerFlags'] = sim['stabilizer']['flags']
                        t_still = np.linalg.norm(truth[1:]-truth[:-1], axis=-1)[(speed[1:] < STILL) & (speed[:-1] < STILL)]
                        realism[-1]['truthStationaryStep_cm_p50'] = round(float(np.median(t_still))*100, 2) if len(t_still) else None
                        acc_t = second_difference(truth, w['ma'], dt)
                        realism[-1]['truthAccelerationP95'] = round(float(np.nanpercentile(acc_t, 95)), 3)
                        progress(f'game {game} {scenario} {camera} seed {s} window {wi+1}/{len(spans)} {a:.0f}-{b:.0f} s: {sim["realism"]}')
                    for method, score in scores.items():
                        if method not in missing:
                            per_seed[method].append(score.summary(dt))
                out = {'realism': _mean_realism(realism), 'methods': {}, 'missing': missing}
                for m in methods:
                    if m not in missing:
                        out['methods'][m] = pooled[m].summary(dt) | {'seconds': round(seconds[m], 1)}
                if seeds > 1:
                    out['spread'] = {m: spread(per_seed[m]) for m in methods if m not in missing}
                entry['scenarios'][scenario]['cameras'][camera] = out
        report['games'][game] = entry
    return report


def estimate(method, sim, names, teams, grid, rate, dt, q):
    """One method's positions (T, P, 2) metres on a simulated window, NaN where it gives none."""
    obs, observed = sim['obs'], sim['observed']
    if method == 'raw':
        return obs
    if method == 'renderer':
        return renderer(obs, observed, dt)
    if method in SG_WINDOWS:
        return savgol(obs, observed, dt, SG_WINDOWS[method])
    if method == 'rts':
        return rts(obs, sim['R'], observed, dt, q)
    if method == 'smooth_tracks':
        return smooth_tracks(sim, grid)
    if method == 'stabilizer':
        return stabilizer(sim, names, [str(t) for t in teams], grid, rate)
    raise ValueError(method)


def table(report):
    lines = []
    for game, entry in report['games'].items():
        for scenario, block in entry['scenarios'].items():
            for camera, out in block['cameras'].items():
                r = out['realism']
                lines.append(f"\n### game {game}, {scenario} scenario, {camera} camera: visible {r['visibleShare']}, observed {r['observedShare']} of visible, "
                             f"occluded {r['occludedShare']}, events {r['eventShare']}, extra events {r.get('extraShare') or {}}, missed {r['missedShare']}, "
                             f"switched {r['switchedShare']}, box {r['boxHeightMedian_px']} px, {r['metresPerVerticalPixelMedian']} m per vertical px"
                             + (f", calibration error {r['calibrationError_m']} m" if r.get('calibrationError_m') else '')
                             + f"; truth: glitch share {r.get('truthGlitchShare')}, stationary step {r['truthStationaryStep_cm_p50']} cm, acceleration p95 "
                             f"{r['truthAccelerationP95']} m/s^2")
                if r.get('stabilizerRegistrationSteps') or r.get('stabilizerFlags'):
                    lines.append(f"stabilizer: registration steps {r.get('stabilizerRegistrationSteps')}; flags {r.get('stabilizerFlags')}")
                buckets = [n for n, _, _ in SPEED_BUCKETS]
                lines.append('| method | coverage obs/gaps | RMSE | median | p95 | '+' | '.join(buckets)+' | gaps RMSE | switched RMSE | speed err (bias) '
                             '| lag ms | stop glide p50/p90 (n) | turn RMSE/median | still step p50/p90 cm | acc p95 | s |')
                lines.append('|---|'+'---|'*(len(buckets)+13))
                for m, s in out['methods'].items():
                    glide, still = s['stopGlide'], s['stationaryStep_cm']
                    lines.append(f"| {m} | {s['coverage']['observed']}/{s['coverage']['gaps']} | {s['rmse']} | {s['median']} | {s['p95']} | "
                                 + ' | '.join(str(s['byTrueSpeed'][b]) for b in buckets)
                                 + f" | {s['gapRmse']} | {s['switchedRmse']} | {s['speedErrorMean']} ({s['speedBias']}) | {s['lag_ms']} | "
                                 f"{glide['p50']}/{glide['p90']} ({glide['n']}) | {s['turnRmse']}/{s['turnMedian']} | {still['p50']}/{still['p90']} | "
                                 f"{s['accelerationP95']} | {s['seconds']} |")
                for m, sp in (out.get('spread') or {}).items():
                    lines.append(f"spread over {report['settings']['seeds']} seeds, {m}: "
                                 + ', '.join(f"{k} {v['mean']} +- {v['sd']} ({v['min']}..{v['max']})" for k, v in sp.items()))
                for m, why in out['missing'].items():
                    lines.append(f'{m}: missing ({why})')
    return '\n'.join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', required=True)
    ap.add_argument('--game', type=int, action='append', required=True)
    ap.add_argument('--camera', action='append', choices=sorted(CAMERAS))
    ap.add_argument('--windows', type=int, default=4)
    ap.add_argument('--length', type=float, default=300.0)
    ap.add_argument('--rate', type=float, default=30.0)
    ap.add_argument('--methods', default=','.join(METHODS))
    ap.add_argument('--rts-q', type=float, default=RTS_Q)
    ap.add_argument('--scenario', action='append', choices=list(SCENARIOS), help=f"default: {', '.join(DEFAULT_SCENARIOS)}")
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--seeds', type=int, default=1, help='repeat with this many seeds and report the spread')
    ap.add_argument('--keep-glitches', action='store_true', help="score the truth's glitches as motion (the old scoring)")
    ap.add_argument('--json')
    a = ap.parse_args(argv)
    report = run(a.data, a.game, a.camera or ['pole', 'broadcast'], a.methods.split(','), a.windows, a.length, a.rate, a.seed, a.rts_q,
                 progress=lambda s: print(s, file=sys.stderr, flush=True), scenarios=a.scenario or list(DEFAULT_SCENARIOS), seeds=max(1, a.seeds),
                 clean=not a.keep_glitches)
    print(table(report))
    if a.json:
        Path(a.json).write_text(json.dumps(report, indent=1), encoding='utf-8')
    return 0


if __name__ == '__main__':
    sys.exit(main())
