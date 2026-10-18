"""The stabilized 2D trajectories (soccer/trajectory.py): synthetic results seen through a low pinhole camera (an amateur
pole: a vertical pixel is worth about half a metre at the centre of the pitch, the boxes' edges carry the measured pixel
noise), so the foot pixel, the Jacobian and the box events are exercised as on a real run. A standing player settles, a
runner is not delayed, sprints, stops, cuts and reversals are followed, single-frame outliers and box-extent events are
dealt with and flagged, gaps are bridged only when short, identities are never mixed, and an unreliable projection keeps
its uncertainty."""
import copy
import importlib.util
import json
import math
import os
import sys
import tracemalloc
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from soccer import field, trajectory  # noqa: E402

W, HH = 1920, 1080
SIZE = field.PitchSize()
FPS = 30.0


def camera(height=6.0, back=25.0, target=(52.5, 30.0), focal=1400.0, side_x=52.5):
    """3x4 pinhole projection of a camera behind the near touchline looking at `target` on the ground."""
    C = np.array([side_x, SIZE.width+back, height])
    f = np.array([target[0], target[1], 0.0])-C
    f /= np.linalg.norm(f)
    right = np.cross([0, 0, 1.0], f)
    right /= np.linalg.norm(right)
    down = np.cross(right, f)
    R = np.stack([right, down, f])
    K = np.array([[focal, 0, W/2], [0, focal, HH/2], [0, 0, 1.0]])
    return K @ np.c_[R, -R @ C]


def ground_homography(P):
    H = P[:, [0, 1, 3]]
    return field.orient(H/abs(H[2, 2]), W, HH)


def project(P, x, y, z=0.0):
    p = P @ np.array([x, y, z, 1.0])
    return p[:2]/p[2]


def box_px(P, x, y, tall=1.8):
    """[x, y, w, h] pixels of a person standing at (x, y): from the head's image down to the foot's."""
    fx, fy = project(P, x, y)
    _, hy = project(P, x, y, tall)
    h = fy-hy
    return np.array([fx-.2*h, hy, .4*h, h])


def noisy(box, rng, a=.45, b=.011):
    """The box with the diagnosed pixel noise on its top and bottom edges and its centre."""
    x, y, w, h = box
    s = math.hypot(a, b*h)
    top, bottom, cx = y+rng.normal()*s, y+h+rng.normal()*s, x+w/2+rng.normal()*s
    return np.array([cx-w/2, top, w, bottom-top])


def make_result(paths, times, P=None, seed=0, noise=(.45, .011), homographies=None, edit=None, tracks=None, anonymous=(), keep=None,
                person=None, extra=None):
    """A finished result: paths {pid: (T, 2) metres, NaN where the person is not seen}; homographies: per-frame stored
    homographies (a calibration error: the boxes stay the truth, the positions follow the stored H); edit(pid, i, box)
    returns the box actually detected (box events); tracks {pid: (T,) local track}; anonymous: pids that are
    unidentified team people (state 'uncertain', no id); keep(pid, i): False leaves the person out of that frame;
    person(pid, i, person) may change a person's fields (a low score); extra(i): more people on frame i (boxes in
    pixels, no team: a split-off legs box)."""
    rng = np.random.default_rng(seed)
    P = camera() if P is None else P
    truth_H = ground_homography(P)
    frames = []
    for i, t in enumerate(times):
        Hs = truth_H if homographies is None else homographies[i]
        people = []
        for n, (pid, path) in enumerate(paths.items()):
            x, y = path[i]
            if not np.isfinite(x) or (keep is not None and not keep(pid, i)):
                continue
            b = box_px(P, x, y)
            b = noisy(b, rng, *noise) if noise else b
            if edit is not None:
                b = edit(pid, i, b)
            foot = (b[0]+b[2]/2, b[1]+b[3])
            p = field.to_pitch(Hs, SIZE, foot)
            if p is None:
                continue
            ident = pid not in anonymous
            people.append({'track': int(tracks[pid][i]) if tracks and pid in tracks else 100+n, 'id': pid if ident else None,
                           'display': pid if ident else None, 'team': 'A', 'state': 'confirmed' if ident else 'uncertain',
                           'box': [b[0]/W, b[1]/HH, b[2]/W, b[3]/HH], 'score': .9, 'pitch': [p[0], p[1]],
                           'pitchConfidence': .8, 'pitchSigma': [.3, .9]})
            if person is not None:
                person(pid, i, people[-1])
        for b in (extra(i) if extra is not None else ()):
            people.append({'track': 900, 'id': None, 'display': None, 'team': None, 'state': 'unknown', 'box': [b[0]/W, b[1]/HH, b[2]/W, b[3]/HH],
                           'score': .6, 'pitch': None, 'pitchConfidence': 0.0})
        frames.append({'time': float(t), 'frame': i, 'people': people, 'calibrated': True, 'motion': [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                       'calibration': {'H': [float(v) for v in Hs.ravel()], 'confidence': .9}, 'profile': {'smallScene': False}})
    slots = [{'id': pid, 'team': 'A', 'status': 'ACTIVE', 'confirmed': True} for pid in paths if pid not in anonymous]
    return {'video': {'width': W, 'height': HH, 'fps': FPS}, 'frames': frames, 'roster': {'slots': slots},
            'tactical': {'pitchSize': SIZE.to_json()}}


def series(track, key):
    return np.array([np.nan if v is None else v for v in track[key]], np.float64)


def xy(track):
    return np.c_[series(track, 'x'), series(track, 'y')]


def raw_xy(result, pid):
    out = np.full((len(result['frames']), 2), np.nan)
    for i, f in enumerate(result['frames']):
        for p in f['people']:
            if p['id'] == pid:
                out[i] = (p['pitch'][0]*SIZE.length, p['pitch'][1]*SIZE.width)
    return out


def integrate(t, v):
    return np.concatenate([[0.0], np.cumsum((v[1:]+v[:-1])/2*np.diff(t))])


def still(t, x=52.5, y=30.0):
    return np.tile([x, y], (len(t), 1)).astype(np.float64)


BROADCAST = dict(height=25.0, back=35.0, focal=1960.0)   # a broadcast-like camera: the ground at about 19 degrees


def raw_track(result, pid='A-01'):
    """The stored positions (T, 1, 2) and their covariance from the true pixel noise through each frame's Jacobian
    (T, 1, 2, 2): what a fixed-q smoother gets."""
    frames = result['frames']
    z = np.full((len(frames), 1, 2), np.nan)
    R = np.tile(np.eye(2), (len(frames), 1, 1, 1))
    for i, f in enumerate(frames):
        p = next((q for q in f['people'] if q['id'] == pid), None)
        if p is None:
            continue
        G = np.linalg.inv(np.array(f['calibration']['H']).reshape(3, 3))[None]
        b = p['box']
        fx, fy, h = (b[0]+b[2]/2)*W, (b[1]+b[3])*HH, b[3]*HH
        J = trajectory.jacobians(G, np.array([fx]), np.array([fy]))[0]
        s = math.hypot(.45, .011*h)
        R[i, 0] = J @ np.diag([s*s, s*s]) @ J.T+np.eye(2)*trajectory.REGISTRATION_FLOOR**2
        z[i, 0] = (p['pitch'][0]*SIZE.length, p['pitch'][1]*SIZE.width)
    return z, R


def fixed_q(times, z, R, q):
    """A plain constant-velocity RTS smoother with a fixed process noise q on the same observations (the baseline the
    stabilizer must not lag behind)."""
    return trajectory.smooth_tracks(times, z, R, options={'adaptive': False, 'robust': False, 'switch': False, 'qLow': q})['x'][:, 0]


def depth(t, s, base):
    """A path along the camera's line of sight (y, towards the camera as s grows) at x = 50."""
    return np.c_[np.full_like(t, 50.0), base+s]


def cut_path(t, speed, seconds, along_depth=False):
    """speed m/s, then a 90-degree cut within `seconds` at 4 s."""
    turn = np.clip((t-4)/seconds, 0, 1)*np.pi/2
    c = np.c_[20+integrate(t, speed*np.cos(turn)), 15+integrate(t, speed*np.sin(turn))]
    return np.c_[c[:, 1]+30, c[:, 0]-10] if along_depth else c


def shifted(offset, H=None):
    """Per frame a stored homography whose positions are the truth plus offset (T, 2) metres: a static camera whose
    registration moves by the offset (the motion rows stay the identity, so the step between two frames is the offset's
    change)."""
    H = ground_homography(camera()) if H is None else H
    return [H @ np.array([[1, 0, -dx], [0, 1, -dy], [0, 0, 1.0]]) for dx, dy in offset]


def warped(maps, H=None):
    """Per frame a stored homography whose positions are the frame's world -> world map S (3x3) of the truth: the
    registration of a static camera reads a ground point X at S X (the motion rows stay the identity)."""
    H = ground_homography(camera()) if H is None else H
    return [H @ np.linalg.inv(S) for S in maps]


def about(point, angle, scale=1.0, shift=(0.0, 0.0)):
    """The world map turning by angle (radians) and scaling about point, then shifting."""
    c, s = math.cos(angle)*scale, math.sin(angle)*scale
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    move = np.array([[1, 0, point[0]], [0, 1, point[1]], [0, 0, 1.0]])
    return np.array([[1, 0, shift[0]], [0, 1, shift[1]], [0, 0, 1.0]]) @ move @ R @ np.linalg.inv(move)


def chain_reads(maps, steps, ground):
    """A static ground point read through the registration whose only changes are the step frames' maps: (T, 2)."""
    read = np.empty((len(maps), 2))
    p = np.asarray(ground, np.float64)
    for i in range(len(maps)):
        if i in steps:
            p = trajectory._carry(maps[i][None], p[None])[0]
        read[i] = p
    return read


def team(t, n=7):
    """n players across the pitch, every other one running along the length at 1.5 m/s."""
    return {f'A-{k:02d}': np.c_[25+8*k+(1.5*t if k % 2 else 0*t), np.full_like(t, 12.0+5*k)] for k in range(1, n+1)}


def ramp(t, at, step, seconds, after=False):
    """The display correction of a step `step` (2,) at time `at` spread over `seconds`: before it linear from 0 at the
    ramp's start to the whole step on the frame before `at`, none from `at` on; after it (after=True) none before `at`,
    minus the whole step on its frame, back to 0 linearly over the ramp."""
    if after:
        share = -np.clip(1-(t-at)/seconds, 0, 1)*(t >= at-1e-3)       # (the summary's times are rounded to 1 ms)
    else:
        share = np.clip((t-(at-seconds))/seconds, 0, 1)*(t < at-1e-3)
    return share[:, None]*np.asarray(step, np.float64)[None]


def event_ramp(t, event, step):
    """ramp() of a step event as noise.registrationSteps reports it."""
    return ramp(t, event['from'], step, event['rampSeconds'], event['spread'] == 'after')


def need(step):
    """The ramp a step of `step` metres at every player asks for."""
    return float(np.clip(float(np.hypot(*step))/trajectory.GLIDE_SPEED, trajectory.RAMP_MIN, trajectory.RAMP_MAX))


def anchored(result, frame, source='reregistered', listed=False, age=0.0, root=None):
    """Mark an anchor on a frame: its calibration's source and anchorAge (age 0: the frame's registration is the
    anchor's), and with listed the summary's registration anchors list it too (with root, when given: the frame of the
    anchor its chain started from)."""
    f = result['frames'][frame]
    f['calibration'].update(source=source if age == 0 else 'interpolated', anchorAge=age)
    if listed:
        anchors = result.setdefault('summary', {}).setdefault('pitch', {}).setdefault('registration', {}).setdefault('anchors', [])
        anchors.append({'frame': frame, 'time': f['time'], 'source': source})
        if root is not None:
            anchors[-1]['root'] = root
    return result


def correction_rate(times, events, X):
    """The display correction's speed at fixed positions X (P, 2) on every frame: (T,) the fastest of them, m/s, the
    spread step frames left out (the correction cancels the raw jump there)."""
    T = len(times)
    out = np.zeros(T)
    stepped = {u for e in events if e['spread'] for u in e['steps']}
    for f in range(1, T):
        if f in stepped:
            continue
        d1 = trajectory.spread_steps(np.full(len(X), f), X, times, events)[0]
        d0 = trajectory.spread_steps(np.full(len(X), f-1), X, times, events)[0]
        out[f] = np.hypot(*(d1-d0).T).max()/(times[f]-times[f-1])
    return out


def shown(out, pids):
    """(players, T, 2) stabilized positions of the identities pids."""
    return np.stack([xy(out['ids'][pid]) for pid in pids])


class SettlingTests(unittest.TestCase):
    def test_a_standing_player_settles_on_the_point(self):
        t = np.arange(0, 6, 1/FPS)
        result = make_result({'A-01': still(t)}, t)
        out = trajectory.stabilize(result, W, HH)
        est, raw = xy(out['ids']['A-01']), raw_xy(result, 'A-01')
        err = np.hypot(*(est-still(t)).T)
        self.assertLess(np.median(err), .12, err)                               # near the point (raw: about 0.3 m in depth)
        self.assertLess(np.median(np.hypot(*(raw-still(t)).T)), .5)
        step, raw_step = np.hypot(*np.diff(est, axis=0).T), np.hypot(*np.diff(raw, axis=0).T)
        self.assertLess(np.median(step), .01)                                   # no vibration frame to frame
        self.assertLess(np.median(step), np.median(raw_step)/20)
        self.assertEqual(out['ids']['A-01']['kind'], ['O']*len(t))
        weights = np.array(out['ids']['A-01']['w'])
        self.assertGreaterEqual(np.mean(weights == 1.0), .97)                  # white noise keeps its weight (chi-square 99%)
        self.assertTrue((weights > 0).all())
        # The uncertainty is the depth direction's: across the line of sight (here x) it is several times smaller.
        sx, sy = np.nanmedian(series(out['ids']['A-01'], 'sx')), np.nanmedian(series(out['ids']['A-01'], 'sy'))
        self.assertGreater(sy, 3*sx)

    def test_a_runner_is_not_delayed(self):
        t = np.arange(0, 6, 1/FPS)
        truth = np.c_[35+5*t, np.full_like(t, 25.0)]                             # 5 m/s along the length
        result = make_result({'A-01': truth}, t, seed=1)
        out = trajectory.stabilize(result, W, HH)
        est, raw = xy(out['ids']['A-01']), raw_xy(result, 'A-01')
        along = est[:, 0]-truth[:, 0]
        self.assertLess(abs(np.mean(along[15:-15])), .03)                      # no lag behind the runner
        self.assertLess(np.sqrt(np.mean(np.sum((est-truth)**2, axis=1))), np.sqrt(np.mean(np.sum((raw-truth)**2, axis=1)))/2.5)
        speed = np.hypot(series(out['ids']['A-01'], 'vx'), series(out['ids']['A-01'], 'vy'))
        self.assertLess(abs(np.median(speed[15:-15])-5.0), .15)


class ManoeuvreTests(unittest.TestCase):
    def test_a_sprint_and_a_stop_without_gliding_past_the_stop(self):
        t = np.arange(0, 8, 1/FPS)
        v = np.interp(t, [0, 2, 3.5, 5, 5.5, 8], [0, 0, 8, 8, 0, 0])            # 8 m/s, then stopped within 0.5 s
        truth = np.c_[30+integrate(t, v), np.full_like(t, 30.0)]
        glides, worst = [], []
        for seed in range(3):
            out = trajectory.stabilize(make_result({'A-01': truth}, t, seed=seed), W, HH)
            est = xy(out['ids']['A-01'])
            after = t >= 5.5
            glides.append(max(0.0, float(np.max(est[after, 0]-truth[after, 0]))))
            worst.append(float(np.max(np.hypot(*(est-truth)[(t >= 1.5) & (t <= 6.5)].T))))
        self.assertLess(max(glides), .06, glides)                               # 0.03 m (0.04 without the raised q)
        self.assertLess(max(worst), .3, worst)                                  # 0.19 m

    def test_a_sharp_cut_and_a_reversal_are_rounded_only_a_little(self):
        t = np.arange(0, 8, 1/FPS)
        turn = np.clip((t-4)/.25, 0, 1)*np.pi/2                                  # 5 m/s, a 90-degree cut within 0.25 s
        cut = np.c_[30+integrate(t, 5*np.cos(turn)), 15+integrate(t, 5*np.sin(turn))]
        v = np.interp(t, [0, 3.8, 4.2, 8], [5, 5, -5, -5])                      # 5 m/s, back the other way within 0.4 s
        reverse = np.c_[40+integrate(t, v), np.full_like(t, 25.0)]
        for name, truth in (('cut', cut), ('reverse', reverse)):
            worst = []
            for seed in range(3):
                out = trajectory.stabilize(make_result({'A-01': truth}, t, seed=seed), W, HH)
                err = np.hypot(*(xy(out['ids']['A-01'])-truth).T)
                worst.append(float(err[(t >= 3.5) & (t <= 5)].max()))
                self.assertLess(np.sqrt(np.mean(err**2)), .11, name)           # 0.08 / 0.07 m (0.13 without the raised q)
            # Bounded rounding at 25-31 m/s^2: 0.19 / 0.15 m; 0.3 m with qHigh 5, 0.52-0.57 m without the raised q.
            self.assertLess(max(worst), .25, (name, worst))

    def test_stops_and_reversals_in_depth_are_followed(self):
        # Along the pole camera's line of sight, where a vertical pixel is worth about 0.4 m: a stop from 8 m/s within
        # 0.3 s and a reversal at 5 m/s within 0.4 s. A fixed-q smoother is 0.34 m off at its worst with q = 1 and 0.19 m
        # with q = 10 (and noisier where the player is steady); the stabilizer 0.14 m, 0.27-0.54 m without the raised q
        # or with a plausibility veto at 6 m/s or 10 m/s^2.
        t = np.arange(0, 8, 1/FPS)
        stop = depth(t, integrate(t, np.interp(t, [0, 4, 4.3, 8], [8, 8, 0, 0])), 5.0)
        reverse = depth(t, integrate(t, np.interp(t, [0, 3.8, 4.2, 8], [5, 5, -5, -5])), 20.0)
        for name, truth in (('stop', stop), ('reverse', reverse)):
            worst, slow = [], []
            for seed in range(3):
                result = make_result({'A-01': truth}, t, seed=seed)
                est = xy(trajectory.stabilize(result, W, HH)['ids']['A-01'])
                z, R = raw_track(result)
                near = (t >= 3.0) & (t <= 5.5)
                worst.append(float(np.max(np.hypot(*(est-truth)[near].T))))
                slow.append(float(np.max(np.hypot(*(fixed_q(t, z, R, 1.0)-truth)[near].T))))
            self.assertLess(max(worst), .16, (name, worst))                       # 0.14 m (0.17 with a 6 m/s veto)
            self.assertLess(max(worst), max(slow), (name, worst, slow))

    def test_a_broadcast_camera_stays_responsive(self):
        # Professional footage (the ground at 19 degrees, 0.06-0.09 m a pixel): the stabilizer must not be more sluggish
        # than a plain smoother. Within 1.3 times a fixed-q (q = 10) RTS's worst error on a cut, a reversal and a stop
        # (0.7-0.85 of it now), and well under the raw positions'. A 6 m/s cut within 0.2 s (42 m/s^2, past the
        # plausibility veto) included: a veto that fell back to Q_LOW left it 0.33 m off (seed 3), 0.12 m now. The sprint
        # start from standing is the adaptive q's weak spot: q rises only once the residuals lean, about 0.1 m behind for
        # 0.2 s at 16 m/s^2 (1.3 times q = 10).
        t = np.arange(0, 8, 1/FPS)
        P = camera(**BROADCAST)
        cases = (('cut', cut_path(t, 5, .25), 1.3, 3), ('cut in depth', cut_path(t, 5, .25, True), 1.3, 3),
                 ('hard cut', cut_path(t, 6, .2), 1.3, 4),
                 ('reverse', np.c_[40+integrate(t, np.interp(t, [0, 3.8, 4.2, 8], [5, 5, -5, -5])), np.full_like(t, 30.0)], 1.3, 3),
                 ('stop in depth', depth(t, integrate(t, np.interp(t, [0, 4, 4.3, 8], [8, 8, 0, 0])), 5.0), 1.3, 3),
                 ('sprint start', np.c_[20+integrate(t, np.interp(t, [0, 3.5, 4.0, 8], [0, 0, 8, 8])), np.full_like(t, 30.0)], 1.5, 3))
        near = (t >= 3.0) & (t <= 5.5)
        for name, truth, factor, seeds in cases:
            worst, plain, raw = [], [], []
            for seed in range(seeds):
                result = make_result({'A-01': truth}, t, P=P, seed=seed)
                est = xy(trajectory.stabilize(result, W, HH)['ids']['A-01'])
                z, R = raw_track(result)
                worst.append(float(np.max(np.hypot(*(est-truth)[near].T))))
                plain.append(float(np.max(np.hypot(*(fixed_q(t, z, R, 10.0)-truth)[near].T))))
                raw.append(float(np.max(np.hypot(*(z[:, 0]-truth)[near].T))))
            self.assertLess(max(worst), factor*max(plain), (name, worst, plain))
            self.assertLess(max(worst), .5*max(raw), (name, worst, raw))


class OutlierTests(unittest.TestCase):
    def test_a_single_frame_outlier_is_rejected_and_flagged(self):
        t = np.arange(0, 6, 1/FPS)
        truth = np.c_[40+2*t, np.full_like(t, 30.0)]
        k = 90

        def jump(pid, i, b):
            return b+[0, 14, 0, 0] if i == k else b                             # the whole box 14 px lower: about 7 m nearer
        result = make_result({'A-01': truth}, t, edit=jump)
        raw = raw_xy(result, 'A-01')
        self.assertGreater(np.hypot(*(raw[k]-truth[k])), 4.0)
        out = trajectory.stabilize(result, W, HH)
        track = out['ids']['A-01']
        self.assertLess(track['w'][k], .1)                                      # rejected (a whole-box shift is not repaired)
        self.assertLess(np.hypot(*(xy(track)[k]-truth[k])), .3)
        self.assertEqual(track['kind'][k], 'O')                                 # still an observed frame, with its stabilized position
        flags = [f for f in out['flags'] if f['frame'] == k]
        self.assertEqual(len(flags), 1, out['flags'])
        self.assertEqual(flags[0]['id'], 'A-01')
        self.assertTrue(flags[0]['reason'].startswith(('rejected', 'down-weighted')))
        self.assertAlmostEqual(flags[0]['raw'][1], raw[k, 1], places=1)
        others = [i for i, w in enumerate(track['w']) if w is not None and w < trajectory.FLAG_WEIGHT and abs(i-k) > 2]
        self.assertLessEqual(len(others), 2)

    def test_a_box_that_loses_the_legs_is_repaired_from_its_top(self):
        t = np.arange(0, 6, 1/FPS)
        truth = np.c_[45+1.5*t, np.full_like(t, 28.0)]
        event = (t >= 3.0) & (t < 3.3)

        def truncate(pid, i, b):
            return b*[1, 1, 1, .7] if event[i] else b                            # top steady, bottom 30% higher for 0.3 s
        result = make_result({'A-01': truth}, t, edit=truncate, seed=2)
        raw = raw_xy(result, 'A-01')
        self.assertGreater(np.median(np.hypot(*(raw[event]-truth[event]).T)), 4.0)   # the player thrown away from the camera
        out = trajectory.stabilize(result, W, HH)
        est = xy(out['ids']['A-01'])
        self.assertLess(np.max(np.hypot(*(est[event]-truth[event]).T)), .5)
        repairs = {f['frame']: f for f in out['flags'] if f['reason'].startswith(('box lost the legs', 'box reaches'))}
        inside = set(np.flatnonzero(event))
        self.assertGreaterEqual(len(set(repairs) & inside), int(event.sum())-2, out['flags'])
        for k in set(repairs)-inside:
            # Next to the event's edges the local fit still leans towards it and may flag a normal box (its repair, the
            # top plus the running height, carries a sigma of about 3 px): the stabilized position stays right. Nowhere
            # else.
            self.assertLessEqual(min(abs(k-i) for i in inside), 2, k)
            self.assertLess(np.hypot(*(est[k]-truth[k])), .5)
        repaired = {k for k, f in repairs.items() if f['reason'].startswith('box lost the legs')}
        used = np.array([repairs[k]['used'] for k in sorted(set(repairs) & inside)])
        self.assertLess(np.median(np.hypot(*(used-truth[sorted(set(repairs) & inside)]).T)), .8)   # each within its ~3 px sigma
        self.assertEqual(out['noise']['events']['legsLost'], len(repaired))
        self.assertGreater(out['summary']['repairedShare'], 0)

    def test_a_box_cut_for_a_second_and_a_half_keeps_its_full_boxes(self):
        # Veo A-10 at 6.6-9.5 s: the box ends at the hips for over a second (the legs a box of their own), a few full
        # boxes in between. A running median of the box height follows the cut boxes; the full boxes were then taken for
        # boxes reaching into a nearer player and 'repaired' to the cut boxes' foot, 9 m off with a sigma of 0.1 m. The
        # geometric height from the top keeps them, repairs or doubts the cut ones, and the sigma stays honest.
        t = np.arange(0, 8, 1/FPS)
        truth = np.c_[45+1.5*t, np.full_like(t, 28.0)]
        cut = (t >= 3.0) & (t < 4.5)
        full = {int(np.argmin(np.abs(t-s))) for s in (3.5, 3.8, 4.1)}
        P = camera()

        def hips(pid, i, b):
            return b*[1, 1, 1, .6] if cut[i] and i not in full else b

        def legs(i):
            if not cut[i] or i in full:
                return []
            x, y, w, h = box_px(P, *truth[i])
            return [[x, y+.6*h-2, w, .4*h+2]]
        result = make_result({'A-01': truth}, t, edit=hips, extra=legs, seed=3)
        raw = raw_xy(result, 'A-01')
        lost = cut & ~np.isin(np.arange(len(t)), sorted(full))
        self.assertGreater(np.median(np.hypot(*(raw[lost]-truth[lost]).T)), 6.0)    # the cut boxes are thrown 9 m away
        out = trajectory.stabilize(result, W, HH)
        track = out['ids']['A-01']
        err = np.hypot(*(xy(track)-truth).T)
        self.assertLess(err[cut].max(), .4, err[cut])                               # 0.18 m (9 m before)
        flagged = {f['frame'] for f in out['flags'] if f['reason'].startswith('box')}
        self.assertFalse(flagged & full, out['flags'])                              # the full boxes are no events ...
        for i in full:
            self.assertLess(err[i], .3)                                             # ... and are where they are
        sy = series(track, 'sy')
        self.assertLess(np.max(err[cut]/sy[cut]), 4.0)                              # 1.9 (119 before: 9 m at 0.08 m)
        self.assertGreater(out['noise']['events']['doubtful']+out['noise']['events']['legsLost'], int(lost.sum())//2)

    def test_a_camera_wide_offset_moves_every_player_alike(self):
        # The registration wobbles and steps by 2 m at 4 s (a re-anchored calibration): every stored position moves by
        # it. Per-player smoothing does not remove it (that is the calibration's to fix) and must not turn it into
        # per-player artifacts: every player's stabilized trajectory moves by about the same amount, nothing splits.
        # The wobble (up to 0.03 m a frame) is a gradual change and stays; the step is spread over a ramp (after it here:
        # the segment's start leaves 4 s before it, and the 2 m step asks for more).
        t = np.arange(0, 8, 1/FPS)
        paths = team(t)
        offset = np.c_[.5*np.sin(2*np.pi*.3*t)+np.where(t >= 4, 2.0, 0.0), .3*np.cos(2*np.pi*.2*t)]
        clean = trajectory.stabilize(make_result(paths, t, seed=4), W, HH)
        moved = trajectory.stabilize(make_result(paths, t, seed=4, homographies=shifted(offset)), W, HH)
        steps = moved['noise']['registrationSteps']
        self.assertGreaterEqual(steps['frames'], 1)                                  # the step is seen as the calibration's
        self.assertEqual((steps['spreadEvents'], steps['spreadFrames'], steps['jumpFrames']), (1, 1, 0))   # ... and spread
        self.assertEqual(moved['summary']['splits'], 0)
        diff = shown(moved, paths)-shown(clean, paths)                               # (players, T, 2)
        expected = offset+event_ramp(t, steps['events'][0], (2.0, 0.0))
        self.assertLess(np.max(np.abs(diff.mean(axis=0)-expected)), .15)            # the offset is kept, the step spread
        self.assertLess(np.max(diff.std(axis=0)), .1)                                # ... and alike for every player
        self.assertLessEqual(sum(1 for f in moved['flags'] if f['reason'].startswith('rejected')),
                             sum(1 for f in clean['flags'] if f['reason'].startswith('rejected'))+len(paths))


class GapTests(unittest.TestCase):
    def test_a_short_gap_is_bridged_and_a_long_one_is_not(self):
        t = np.arange(0, 9, 1/FPS)
        truth = np.c_[30+3*t, np.full_like(t, 20.0)]
        short = (t >= 2.0) & (t < 2.6)                                           # 0.6 s unseen
        long_ = (t >= 5.0) & (t < 7.0)                                           # 2 s unseen
        result = make_result({'A-01': truth}, t, keep=lambda pid, i: not (short[i] or long_[i]), seed=5)
        out = trajectory.stabilize(result, W, HH)
        track = out['ids']['A-01']
        kinds = np.array(track['kind'], object)
        self.assertTrue((kinds[short] == 'I').all())
        est = xy(track)
        self.assertLess(np.max(np.hypot(*(est[short]-truth[short]).T)), .5)
        sy = series(track, 'sy')
        gap = np.flatnonzero(short)
        middle = gap[len(gap)//2]
        observed = np.array([k == 'O' for k in track['kind']])
        self.assertGreater(sy[middle], 1.25*np.median(sy[observed]))           # the sigma grows into the gap ...
        self.assertGreaterEqual(sy[middle], np.max(sy[gap])-.002)              # ... most in its middle
        self.assertTrue(all(k is None for k in kinds[long_]))                   # nothing fabricated across 2 s
        self.assertTrue(np.isnan(est[long_]).all())
        self.assertEqual(kinds[np.flatnonzero(long_)[0]-1], 'O')
        self.assertEqual(kinds[np.flatnonzero(long_)[-1]+1], 'O')
        self.assertEqual(out['summary']['segments'], 2)

    def test_one_identity_on_two_local_tracks_is_one_trajectory(self):
        t = np.arange(0, 6, 1/FPS)
        truth = np.c_[40+2*t, np.full_like(t, 32.0)]
        tracks = {'A-01': np.where(t < 3.0, 7, 19)}                             # the tracker handed the player to a new track
        out = trajectory.stabilize(make_result({'A-01': truth}, t, tracks=tracks, seed=6), W, HH)
        track = out['ids']['A-01']
        self.assertEqual(track['kind'], ['O']*len(t))
        self.assertEqual(out['summary']['segments'], 1)
        est = xy(track)
        k = int(np.argmax(t >= 3.0))
        self.assertLess(np.hypot(*(est[k]-est[k-1])), .15)                     # no break where the track changed
        self.assertFalse(out['anonymous'])

    def test_irregular_timestamps(self):
        rng = np.random.default_rng(7)
        t = np.cumsum(rng.uniform(.02, .07, 160))                               # frames 20-70 ms apart
        truth = np.c_[30+4*t, 25+1.0*t]
        out = trajectory.stabilize(make_result({'A-01': truth}, t, seed=7), W, HH)
        track = out['ids']['A-01']
        est = xy(track)
        inner = slice(10, -10)
        self.assertLess(np.sqrt(np.mean(np.sum((est-truth)[inner]**2, axis=1))), .15)
        self.assertLess(abs(np.mean((est-truth)[inner, 0])), .05)                # no lag
        self.assertLess(abs(np.median(series(track, 'vx')[inner])-4.0), .25)    # the velocity uses the actual steps
        self.assertLess(abs(np.median(series(track, 'vy')[inner])-1.0), .25)


class IdentityTests(unittest.TestCase):
    def test_an_unexplained_jump_splits_the_trajectory(self):
        # A-01 runs at 2 m/s; at 3 s its observations continue on a player 15 m away (an unresolved identity switch).
        t = np.arange(0, 6, 1/FPS)
        truth = np.c_[40+2*t, np.where(t < 3, 30.0, 45.0)]
        out = trajectory.stabilize(make_result({'A-01': truth}, t, seed=8), W, HH)
        k = int(np.argmax(t >= 3))
        splits = [f for f in out['flags'] if f['reason'].startswith('unexplained jump')]
        self.assertEqual([f['frame'] for f in splits], [k])
        est = xy(out['ids']['A-01'])
        self.assertLess(np.hypot(*(est[k-1]-truth[k-1])), .5)                   # not glided across: each side keeps its own
        self.assertLess(np.hypot(*(est[k]-truth[k])), .5)
        self.assertEqual(out['summary']['segments'], 2)

    def test_the_new_segment_starts_at_the_new_players_first_box(self):
        # The switch's first three boxes are uncertain detections (no evidence of a switch themselves): they belong to
        # the player after the jump and start his segment, not drawn at the old player's place (amateur A-03 at 160.4 s).
        t = np.arange(0, 6, 1/FPS)
        truth = np.c_[40+2*t, np.where(t < 3, 30.0, 45.0)]
        k = int(np.argmax(t >= 3))

        def uncertain(pid, i, p):
            if k <= i < k+3:
                p['score'] = .3
        out = trajectory.stabilize(make_result({'A-01': truth}, t, seed=8, person=uncertain), W, HH)
        self.assertEqual([f['frame'] for f in out['flags'] if f['reason'].startswith('unexplained jump')], [k])
        est = xy(out['ids']['A-01'])
        self.assertLess(np.max(np.hypot(*(est[k:k+3]-truth[k:k+3]).T)), .5)       # 15 m off before
        self.assertLess(np.hypot(*(est[k-1]-truth[k-1])), .5)

    def test_a_switch_shortly_before_the_track_ends_is_split(self):
        # The other player is seen 0.4 s only (less than SWITCH_PERSIST) before the track ends: the end persists. Not
        # split, the smoother pulled the old player's last half second 8.5 m towards the other one.
        t = np.arange(0, 6, 1/FPS)
        truth = np.c_[40+2*t, np.where(t < 3, 30.0, 45.0)]
        truth[t >= 3.4] = np.nan
        k = int(np.argmax(t >= 3))
        out = trajectory.stabilize(make_result({'A-01': truth}, t, seed=8), W, HH)
        self.assertEqual([f['frame'] for f in out['flags'] if f['reason'].startswith('unexplained jump')], [k])
        est = xy(out['ids']['A-01'])
        before, after = (t >= 2.5) & (t < 3), (t >= 3) & (t < 3.4)
        self.assertLess(np.max(np.hypot(*(est[before]-truth[before]).T)), .3)
        self.assertLess(np.max(np.hypot(*(est[after]-truth[after]).T)), .5)

    def test_a_far_players_box_step_is_no_switch(self):
        # Near the far touchline a pixel is about 0.9 m in depth: the box settling 5 px higher moves the stored position
        # 4.6 m for good. That is no switch (the jump is within SWITCH_NOISE sigmas of the two observations): smoothed,
        # not split into a hard jump on the board.
        t = np.arange(0, 6, 1/FPS)
        truth = np.c_[50+.5*t, np.full_like(t, 6.0)]
        k = int(np.argmax(t >= 3))
        result = make_result({'A-01': truth}, t, edit=lambda pid, i, b: b-[0, 5, 0, 0] if i >= k else b, seed=13)
        raw = raw_xy(result, 'A-01')
        self.assertGreater(np.hypot(*(raw[k:].mean(axis=0)-raw[:k].mean(axis=0))), 4.0)
        self.assertGreater(np.hypot(*(raw[k]-raw[k-1])), trajectory.SWITCH_MIN+trajectory.SWITCH_SPEED/FPS)
        out = trajectory.stabilize(result, W, HH)
        self.assertEqual(out['summary']['splits'], 0)
        self.assertLess(np.nanmax(np.hypot(*np.diff(xy(out['ids']['A-01']), axis=0).T)), 1.0)

    def test_identities_are_never_mixed_and_anonymous_tracks_are_kept_by_their_track(self):
        t = np.arange(0, 4, 1/FPS)
        paths = {'A-01': np.c_[40+3*t, np.full_like(t, 30.0)], 'A-02': np.c_[52-3*t, np.full_like(t, 30.6)],   # they cross
                 'ghost': np.c_[60+t, np.full_like(t, 20.0)]}
        tracks = {'A-01': np.full(len(t), 3), 'A-02': np.full(len(t), 4), 'ghost': np.full(len(t), 77)}
        result = make_result(paths, t, tracks=tracks, anonymous=('ghost',), seed=9)
        out = trajectory.stabilize(result, W, HH)
        self.assertEqual(set(out['ids']), {'A-01', 'A-02'})
        self.assertEqual(set(out['anonymous']), {'77'})
        for pid in ('A-01', 'A-02'):
            self.assertLess(np.max(np.hypot(*(xy(out['ids'][pid])-paths[pid]).T)), .6, pid)
        self.assertLess(np.median(np.hypot(*(xy(out['anonymous']['77'])-paths['ghost']).T)), .15)


class CalibrationTests(unittest.TestCase):
    def test_a_camera_cut_is_a_calibration_step(self):
        # At a cut the registration starts again (no motion row: its step is not measured) and every stored position
        # jumps by its offset, here 2.5 m. Each player takes it at once: no glide across it, nothing rejected or split.
        t = np.arange(0, 8, 1/FPS)
        paths = {f'A-{n:02d}': np.c_[25+8*n+(1.5*t if n % 2 else 0*t), np.full_like(t, 12.0+5*n)] for n in range(1, 8)}
        offset = np.c_[np.where(t >= 4, 2.5, 0.0), np.zeros_like(t)]
        H = ground_homography(camera())
        result = make_result(paths, t, seed=4, homographies=[H @ np.array([[1, 0, -dx], [0, 1, -dy], [0, 0, 1.0]]) for dx, dy in offset])
        k = int(np.argmax(t >= 4))
        result['frames'][k]['cut'], result['frames'][k]['motion'] = True, None
        out = trajectory.stabilize(result, W, HH)
        self.assertEqual(out['noise']['registrationSteps']['cuts'], 1)
        self.assertEqual(out['summary']['splits'], 0)
        near = (t >= 3.5) & (t <= 4.5)
        for pid, truth in paths.items():
            self.assertLess(np.max(np.hypot(*(xy(out['ids'][pid])-truth-offset)[near].T)), .3, pid)   # 1.56 m before
        self.assertFalse([f for f in out['flags'] if f['reason'].startswith(('rejected', 'down-weighted'))])

    def test_a_flipped_homography_is_oriented(self):
        # A result written before the stored homographies were oriented holds some with the opposite sign (the same
        # mapping, the ground behind the camera): they still give the Jacobian and the box repairs.
        t = np.arange(0, 6, 1/FPS)
        truth = np.c_[45+1.5*t, np.full_like(t, 28.0)]
        result = make_result({'A-01': truth}, t, edit=lambda pid, i, b: b*[1, 1, 1, .7] if 90 <= i < 99 else b, seed=2)
        flipped = make_result({'A-01': truth}, t, edit=lambda pid, i, b: b*[1, 1, 1, .7] if 90 <= i < 99 else b, seed=2)
        for f in flipped['frames'][::2]:
            f['calibration']['H'] = [-v for v in f['calibration']['H']]
        a, b = trajectory.stabilize(result, W, HH), trajectory.stabilize(flipped, W, HH)
        self.assertEqual(a['ids']['A-01'], b['ids']['A-01'])
        self.assertEqual(a['noise']['events'], b['noise']['events'])
        self.assertGreater(a['noise']['events']['legsLost'], 0)


class StepSpreadingTests(unittest.TestCase):
    """A registration snap never makes every player jump in the 2D view: the display spreads it over a ramp as long as
    the step needs to glide at GLIDE_SPEED, cut short only by a user keyframe (pinned: shown exactly as registered), a
    camera cut or another registration start; overlapping ramps add. A keyframe's arriving step is spread before it, its
    leaving step after it, and so are a trusted re-registration's (rooted at a keyframe), which other ramps cross and
    whose squeezed ramps turn, unless it is an outlier (its two steps cancel); a step right after a keyframe or a
    trusted anchor leaves it, and on an automatic anchor's frame it takes the automatic steps after it (an outlier there
    disappears). An automatic anchor's steps and the steps near them are one event spread before it: the event's frames
    show the registration after it, an outlier anchor disappears. A ramp its room squeezes is spread the other way when
    it may, or not at all (the jump freedom), never into a fast glide; an event of several steps that cancels is spread
    its first way when neither has the room. The raw positions are not changed."""

    def test_a_step_is_spread_before_it_and_the_registration_holds_from_its_frame_on(self):
        t = np.arange(0, 8, 1/FPS)
        paths = team(t)
        step = np.array([.6, .3])
        offset = np.where((t >= 4)[:, None], step[None], 0.0)
        result = make_result(paths, t, seed=21, homographies=shifted(offset))
        raw_before = json.dumps(result['frames'])
        clean = trajectory.stabilize(make_result(paths, t, seed=21), W, HH)
        moved = trajectory.stabilize(result, W, HH)
        self.assertEqual(json.dumps(result['frames']), raw_before)                  # the raw positions are untouched
        steps = moved['noise']['registrationSteps']
        self.assertEqual((steps['spreadEvents'], steps['jumpFrames'], steps['notSpread'], steps['anchorFrames']), (1, 0, 0, 0))
        event = steps['events'][0]
        self.assertEqual((event['reason'], event['spread']), ('blip', 'before'))
        self.assertAlmostEqual(event['from'], 4.0, places=2)
        self.assertAlmostEqual(event['rampSeconds'], need(step), places=2)          # as long as the step asks for
        self.assertAlmostEqual(event['stepMetres']['median'], float(np.hypot(*step)), places=2)
        diff = shown(moved, paths)-shown(clean, paths)
        expected = event_ramp(t, event, step)+offset
        self.assertLess(np.max(np.abs(diff-expected)), .05)                          # the ramp, then the raw from the step on
        jump = np.hypot(*np.diff(shown(moved, paths), axis=1).transpose(2, 0, 1))   # (players, T-1) frame-to-frame steps
        calm = np.hypot(*np.diff(shown(clean, paths), axis=1).transpose(2, 0, 1))
        k = int(np.argmax(t >= 4))
        self.assertLess(np.max(jump[:, k-1]-calm[:, k-1]), .03)                      # no jump at the step (0.67 m raw)
        self.assertEqual(moved['summary']['splits'], 0)
        self.assertFalse([f for f in moved['flags'] if f['reason'].startswith('rejected')])
        self.assertGreater(moved['summary']['spreadMetres']['observations'], 0)
        self.assertLess(moved['summary']['spreadMetres']['max'], float(np.hypot(*step))+.01)

    def test_a_spike_shows_its_neighbours_registration(self):
        # One frame's registration jumps 1.1 m away and the next one back (no anchor there): the two steps are one event
        # whose map is the identity, the spike frame is carried to its neighbours' registration, nothing ramps and nothing
        # jumps.
        t = np.arange(0, 8, 1/FPS)
        paths = team(t)
        k = int(np.argmax(t >= 4))
        offset = np.zeros((len(t), 2))
        offset[k] = (1.0, .5)
        clean = trajectory.stabilize(make_result(paths, t, seed=22), W, HH)
        moved = trajectory.stabilize(make_result(paths, t, seed=22, homographies=shifted(offset)), W, HH)
        steps = moved['noise']['registrationSteps']
        self.assertEqual((steps['spreadEvents'], steps['spreadFrames'], steps['jumpFrames']), (1, 2, 0))
        self.assertLess(steps['events'][0]['stepMetres']['max'], .01)               # it composes to the identity
        diff = shown(moved, paths)-shown(clean, paths)
        self.assertLess(np.max(np.hypot(*diff.transpose(2, 0, 1))), .03)             # no spike (1.1 m on the frame raw)
        self.assertEqual(moved['summary']['spreadMetres']['observations'], len(paths))   # only the spike frame moved

    def test_a_keyframes_leaving_snap_keeps_the_keyframe_and_eases_out_after_it(self):
        # A user keyframe at 4 s; the frame after it jumps by 0.67 m into the registration's blend towards the next
        # anchor. The step leaves the keyframe: it is spread after it, so the keyframe's frame and every frame before it
        # stay as registered and the frames from the step on ease from the keyframe's registration into the blend's.
        t = np.arange(0, 8, 1/FPS)
        paths = team(t)
        k = int(np.argmax(t >= 4))
        step = np.array([.6, -.3])
        offset = np.where((np.arange(len(t)) > k)[:, None], step[None], 0.0)
        clean = trajectory.stabilize(make_result(paths, t, seed=28), W, HH)
        moved = trajectory.stabilize(anchored(make_result(paths, t, seed=28, homographies=shifted(offset)), k, 'landmarks'), W, HH)
        steps = moved['noise']['registrationSteps']
        self.assertEqual((steps['spreadEvents'], steps['jumpFrames'], steps['anchorFrames'], steps['pinnedFrames']), (1, 0, 1, 1))
        self.assertEqual(steps['pinnedCorrectionMetres'], 0.0)
        event = steps['events'][0]
        self.assertEqual((event['reason'], event['keyframe'], event['spread'], event['squeezed']), ('leaving', True, 'after', None))
        self.assertAlmostEqual(event['from'], t[k+1], places=3)
        self.assertAlmostEqual(event['rampSeconds'], need(step), places=2)
        diff = shown(moved, paths)-shown(clean, paths)
        expected = event_ramp(t, event, step)+offset                                  # 0 up to the keyframe, then easing in
        self.assertLess(np.max(np.abs(diff-expected)), .05)
        self.assertLess(np.max(np.abs(diff[:, :k+1])), .03)                          # the keyframe's frame and before: raw
        jump = np.hypot(*np.diff(shown(moved, paths), axis=1).transpose(2, 0, 1))
        calm = np.hypot(*np.diff(shown(clean, paths), axis=1).transpose(2, 0, 1))
        self.assertLess(np.max(jump[:, k]-calm[:, k]), .03)                          # no jump on the step's frame

    def test_an_automatic_anchors_steps_are_one_event_spread_before_it(self):
        # An automatic (re-registered) anchor at 4 s. With only a leaving step (the frame after it jumps by 0.67 m into
        # the blend) the step is spread before the anchor, as a blip is: the anchor's frame shows the blend's
        # registration, the frames after the step are as registered, and the anchor's registration is never held after
        # it (spread after it, the anchor's registration held for seconds where the blend was better: BC's 43.2 s paint
        # anchor). With an arriving step too, the same way (a registration that snaps twice in two frames), the two are
        # one event: its net step is spread before it over the ramp the net step asks for, and the anchor's frame
        # shows the registration after it.
        t = np.arange(0, 8, 1/FPS)
        paths = team(t)
        n = len(t)
        k = int(np.argmax(t >= 4))
        clean = trajectory.stabilize(make_result(paths, t, seed=28), W, HH)
        leave, arrive = np.array([.4, -.2]), np.array([.2, -.1])
        for offset, total, reason in ((np.where((np.arange(n) > k)[:, None], leave[None], 0.0), leave, 'leaving'),
                                      (np.where((np.arange(n) >= k)[:, None], arrive[None], 0.0)+np.where((np.arange(n) > k)[:, None], leave[None], 0.0),
                                       arrive+leave, 'anchor')):
            moved = trajectory.stabilize(anchored(make_result(paths, t, seed=28, homographies=shifted(offset)), k), W, HH)
            steps = moved['noise']['registrationSteps']
            self.assertEqual((steps['spreadEvents'], steps['jumpFrames'], steps['anchorFrames'], steps['pinnedFrames']), (1, 0, 1, 0), reason)
            event = steps['events'][0]
            self.assertEqual((event['reason'], event['keyframe'], event['spread'], event['squeezed']), (reason, False, 'before', None))
            self.assertAlmostEqual(event['stepMetres']['median'], float(np.hypot(*total)), places=2)   # the net step
            self.assertAlmostEqual(event['rampSeconds'], need(total), places=2)
            self.assertAlmostEqual(event['from'], t[k] if reason == 'anchor' else t[k+1], places=3)
            diff = shown(moved, paths)-shown(clean, paths)
            expected = ramp(t, t[k], total, event['rampSeconds'])+offset               # (at the anchor's frame: the whole step)
            expected[k] = total
            self.assertLess(np.max(np.abs(diff-expected)), .05, reason)
            self.assertLess(np.max(np.abs(diff[:, k]-total)), .05, reason)             # the anchor's frame: the blend's
            self.assertLess(np.max(np.abs(diff[:, k+1:]-total)), .05, reason)          # after it: as registered
            jump = np.hypot(*np.diff(shown(moved, paths), axis=1).transpose(2, 0, 1))
            calm = np.hypot(*np.diff(shown(clean, paths), axis=1).transpose(2, 0, 1))
            self.assertLess(np.max(jump[:, k-1:k+1]-calm[:, k-1:k+1]), .03, reason)   # no jump at either step
            self.assertFalse([f for f in moved['flags'] if f['reason'].startswith('rejected')])

    def test_an_outlier_automatic_anchor_leaves_no_drift(self):
        # An automatic anchor whose registration is 1.1 m off both neighbours' (an arriving step and a leaving step
        # back): one event that composes to the identity. The anchor's frame is shown in its neighbours' registration
        # and nothing ramps (spread as an arriving and a leaving step, every player drifted towards the outlier for 5 s
        # and back). The same with an outlier turned by 0.1 rad about the pitch's centre (2-4 m at the players): the
        # event's two steps are carried through each other, so they cancel exactly (added each in the frame's own
        # registration, they would leave 0.01 of the distance from the centre: up to 0.3 m).
        t = np.arange(0, 8, 1/FPS)
        paths = team(t)
        k = int(np.argmax(t >= 4))
        clean = trajectory.stabilize(make_result(paths, t, seed=22), W, HH)
        shift = np.zeros((len(t), 2))
        shift[k] = (1.0, .5)
        turn = [np.eye(3)]*len(t)
        turn[k] = about((52.5, 34.0), .1)
        for name, homographies in (('shifted', shifted(shift)), ('turned', warped(turn))):
            moved = trajectory.stabilize(anchored(make_result(paths, t, seed=22, homographies=homographies), k), W, HH)
            steps = moved['noise']['registrationSteps']
            self.assertEqual((steps['spreadEvents'], steps['spreadFrames'], steps['jumpFrames']), (1, 2, 0), name)
            event = steps['events'][0]
            self.assertEqual((event['reason'], event['spread'], event['grid'][0] > .5), ('anchor', 'before', True), name)
            self.assertLess(event['stepMetres']['max'], .01, name)                    # it composes to the identity
            diff = shown(moved, paths)-shown(clean, paths)
            self.assertLess(np.max(np.hypot(*diff.transpose(2, 0, 1))), .03, name)    # no outlier, no drift
            self.assertEqual(moved['summary']['spreadMetres']['observations'], len(paths), name)   # only its frame moved
        times, events, _ = trajectory.display_events(anchored(make_result(paths, t, seed=22, homographies=warped(turn)), k), W, HH)
        X = np.array([p[k-1] for p in paths.values()])
        for f in range(1, k):                                                        # no drift before it at fixed positions
            np.testing.assert_allclose(trajectory.spread_steps(np.full(len(X), f), X, times, events)[0], X, atol=1e-9)

    def test_a_relabelled_person_changes_no_ramp(self):
        # A correction relabels people and never moves them: the step of a registration that turns by 0.03 rad about the
        # pitch's centre (0.1-0.6 m at the players) is measured at every person on the frame, whoever they are, so taking
        # the farthest player out of the match (no identity, rejected) leaves the ramps, and with the original noise model
        # every other player, exactly where they were (measured at the identities' observations only, the ramp
        # changed, and with it everybody's correction on it: excluding B-03 on the amateur run changed 7 ramps and moved
        # 8048 samples of the others by up to 0.23 m). The ramp's step is the median at the people, not the pitch grid.
        t = np.arange(0, 8, 1/FPS)
        paths = team(t)
        k = int(np.argmax(t >= 4))
        turn = [np.eye(3) if i <= k else about((52.5, 34.0), .03) for i in range(len(t))]
        result = anchored(make_result(paths, t, seed=29, homographies=warped(turn)), k)
        out = copy.deepcopy(result)
        for f in out['frames']:
            for p in f['people']:
                if p.get('id') == 'A-07':
                    p.update(id=None, display=None, team=None, state='rejected', role='UNKNOWN')
        before, after = trajectory.display_events(result, W, HH)[1], trajectory.display_events(out, W, HH)[1]
        self.assertEqual(len(before), 1)
        self.assertEqual([(e['steps'], e['spread'], e['ramp'], e['reach']) for e in after],
                         [(e['steps'], e['spread'], e['ramp'], e['reach']) for e in before])
        X = np.array([p['pitch'] for p in result['frames'][before[0]['start']-1]['people']])*[SIZE.length, SIZE.width]
        moved = (before[0]['map'][:2, :2] @ X.T).T+before[0]['map'][:2, 2]                  # where the turn takes them
        self.assertAlmostEqual(before[0]['reach'], float(np.median(np.hypot(*(moved-X).T))), places=9)   # at the people
        first = trajectory.stabilize(result, W, HH)
        again = trajectory.stabilize(out, W, HH, noise=json.loads(json.dumps(first['noise'])))
        self.assertNotIn('A-07', [k for k, track in again['ids'].items() if any(v is not None for v in track['x'])])
        for pid in paths:
            if pid != 'A-07':
                self.assertEqual(again['ids'][pid], first['ids'][pid], pid)

    def test_a_squeezed_ramp_is_spread_the_other_way_or_not_at_all(self):
        # A step 0.2 s before a cut, leaving an anchor. An automatic anchor's is spread before it: its full ramp, no fast
        # glide. A keyframe's can only be spread after it, into 0.17 s (every player gliding at 4 m/s): it is not spread
        # but snaps, every player taking it at once with the jump freedom, the keyframe exact, nothing rejected. An
        # automatic anchor's arriving step 0.4 s after a cut has no room before it (1.7 m/s): it is spread after it.
        t = np.arange(0, 8, 1/FPS)
        paths = team(t)
        n = len(t)
        c = int(np.argmax(t >= 5))
        k = c-6
        step = np.array([.6, .3])
        cut = np.where((np.arange(n) >= c)[:, None], [2.5, 0.0], 0.0)
        offset = np.where((np.arange(n) > k)[:, None], step[None], 0.0)+cut
        X = np.array([p[0] for p in paths.values()])

        def run(offset, frame, source, seed=43):
            result = anchored(make_result(paths, t, seed=seed, homographies=shifted(offset)), frame, source)
            result['frames'][c]['cut'], result['frames'][c]['motion'] = True, None
            return result, trajectory.stabilize(result, W, HH)
        only_cut = run(cut, k, 'reregistered')[1]
        result, moved = run(offset, k, 'reregistered')
        steps = moved['noise']['registrationSteps']
        event = steps['events'][0]
        self.assertEqual((event['reason'], event['spread'], event['squeezed']), ('leaving', 'before', None))
        self.assertAlmostEqual(event['rampSeconds'], need(step), places=2)
        times, events, _ = trajectory.display_events(result, W, HH)
        self.assertLess(correction_rate(times, events, X).max(), 1.6*trajectory.GLIDE_SPEED)   # 4 m/s spread after it
        self.assertFalse([f for f in moved['flags'] if f['reason'].startswith(('rejected', 'down-weighted'))])
        diff = shown(moved, paths)-shown(only_cut, paths)
        self.assertLess(np.max(np.abs(diff[:, k+1:]-step)), .06)                    # from the step on: as registered
        result, snapped = run(offset, k, 'landmarks')
        steps = snapped['noise']['registrationSteps']
        event = steps['events'][0]
        self.assertEqual((event['reason'], event['keyframe'], event['spread'], event['squeezed']), ('leaving', True, None, 'snapped'))
        self.assertEqual((steps['spreadEvents'], steps['notSpread'], steps['jumpFrames'], steps['squeezed']), (0, 1, 1, {'turned': 0, 'snapped': 1}))
        self.assertIsNone(snapped['summary']['spreadMetres'])
        self.assertEqual((snapped['summary']['splits'], steps['pinnedCorrectionMetres']), (0, 0.0))
        self.assertFalse([f for f in snapped['flags'] if f['reason'].startswith(('rejected', 'down-weighted'))])
        near = (t >= 4.2) & (t < 5)
        for pid, truth in paths.items():                                             # each takes the step at once
            self.assertLess(np.max(np.hypot(*(xy(snapped['ids'][pid])-truth-offset)[near].T)), .3, pid)
        times, events, _ = trajectory.display_events(result, W, HH)
        self.assertEqual(correction_rate(times, events, X).max(), 0.0)
        j = int(np.argmax(t >= 3))+12
        offset = np.where((np.arange(n) >= int(np.argmax(t >= 3)))[:, None], [2.5, 0.0], 0.0)+np.where((np.arange(n) >= j)[:, None], step[None], 0.0)
        result = anchored(make_result(paths, t, seed=42, homographies=shifted(offset)), j, 'paint')
        result['frames'][j-12]['cut'], result['frames'][j-12]['motion'] = True, None
        turned = trajectory.stabilize(result, W, HH)
        steps = turned['noise']['registrationSteps']
        event = steps['events'][0]
        self.assertEqual((event['reason'], event['spread'], event['squeezed'], steps['squeezed']), ('arriving', 'after', 'turned', {'turned': 1, 'snapped': 0}))
        self.assertAlmostEqual(event['rampSeconds'], need(step), places=2)
        times, events, _ = trajectory.display_events(result, W, HH)
        self.assertLess(correction_rate(times, events, X).max(), 1.6*trajectory.GLIDE_SPEED)   # 1.7 m/s spread before it
        self.assertFalse([f for f in turned['flags'] if f['reason'].startswith(('rejected', 'down-weighted'))])

    def test_a_pinned_keyframe_is_shown_exactly_as_registered(self):
        # A user keyframe at 4 s whose registration differs from both neighbours the same way (an arriving and a leaving
        # step that cancel: a keyframe is honoured all the same), a blip 0.4 s after it whose ramp before it would reach
        # across the keyframe, and a second keyframe at 6 s that only the summary's anchors list (the registration did not
        # take it on its frame) with a step arriving there: the display corrects nothing on either keyframe's frame, every
        # ramp ends there, and nothing jumps.
        t = np.arange(0, 8, 1/FPS)
        paths = team(t)
        k, j, m = (int(np.argmax(t >= s-1e-9)) for s in (4.0, 4.4, 6.0))
        offset = np.zeros((len(t), 2))
        offset[k] = (.5, .2)
        offset[j:] += (.6, 0.0)
        offset[m:] += (-.4, .3)
        clean = trajectory.stabilize(make_result(paths, t, seed=29), W, HH)
        result = anchored(make_result(paths, t, seed=29, homographies=shifted(offset)), k, 'landmarks')
        anchored(result, m, 'landmarks', listed=True, age=1.2)
        moved = trajectory.stabilize(result, W, HH)
        steps = moved['noise']['registrationSteps']
        self.assertEqual((steps['anchorFrames'], steps['pinnedFrames'], steps['trustedFrames'], steps['trustedEvents'], steps['notSpread'],
                          steps['jumpFrames']), (2, 2, 0, 0, 0, 0))
        self.assertEqual(steps['pinnedCorrectionMetres'], 0.0)                     # exactly as registered
        self.assertEqual([(e['reason'], e['keyframe'], e['trusted'], e['spread'], e['squeezed']) for e in steps['events']],
                         [('arriving', True, False, 'before', None), ('leaving', True, False, 'after', None),
                          ('blip', False, False, 'after', 'turned'),        # (the blip's 0.4 s before it would glide at 1.5 m/s)
                          ('arriving', True, False, 'before', None)])         # (the 4 s keyframe, an outlier, is honoured)
        for e in steps['events'][1:]:                                               # every ramp after 4 s ends at 6 s
            self.assertLessEqual(e['from']+e['rampSeconds'] if e['spread'] == 'after' else e['from'], 6.0+1e-6)
        self.assertAlmostEqual(steps['events'][3]['rampSeconds'], 2.0, places=2)     # back to the keyframe at 4 s
        times, events, pinned = trajectory.display_events(result, W, HH)
        self.assertEqual(np.flatnonzero(pinned).tolist(), [k, m])
        X = np.array([[30.0, 20.0], [70.0, 45.0], [52.5, 34.0]])
        for f in (k, m):
            np.testing.assert_array_equal(trajectory.spread_steps(np.full(3, f), X, times, events)[0], X)
        diff = shown(moved, paths)-shown(clean, paths)
        for f in (k, m):
            self.assertLess(np.max(np.abs(diff[:, f]-offset[f])), .05)              # the keyframe's own registration
        jump = np.hypot(*np.diff(shown(moved, paths), axis=1).transpose(2, 0, 1))
        calm = np.hypot(*np.diff(shown(clean, paths), axis=1).transpose(2, 0, 1))
        self.assertLess(np.max((jump-calm)[:, [k-1, k, j-1, m-1]]), .04)             # no jump at any step (0.5-0.6 m raw)

    def test_two_steps_close_together_keep_their_ramps_and_glide_slowly(self):
        # Two blips 0.4 s apart, 0.6 m each the same way. The second's ramp used to end at the first (0.4 s: every
        # player glided at 1.5 m/s); now each keeps the ramp its step asks for, the two add, and no position moves faster
        # than the two glides together.
        t = np.arange(0, 10, 1/FPS)
        paths = team(t)
        step = np.array([.6, 0.0])
        offset = np.where((t >= 5)[:, None], step[None], 0.0)+np.where((t >= 5.4)[:, None], step[None], 0.0)
        result = make_result(paths, t, seed=30, homographies=shifted(offset))
        moved = trajectory.stabilize(result, W, HH)
        clean = trajectory.stabilize(make_result(paths, t, seed=30), W, HH)
        steps = moved['noise']['registrationSteps']
        self.assertEqual(steps['spreadEvents'], 2)
        for e in steps['events']:
            self.assertEqual(e['spread'], 'before')
            self.assertAlmostEqual(e['rampSeconds'], need(step), places=2)
        times, events, _ = trajectory.display_events(result, W, HH)
        X = np.array([p[0] for p in paths.values()])
        rate = correction_rate(times, events, X)
        self.assertLess(rate.max(), 2*float(np.hypot(*step))/need(step)+.01)        # 2 x GLIDE_SPEED at most
        self.assertGreater(rate.max(), 1.5*float(np.hypot(*step))/need(step))       # ... where the two overlap
        diff = shown(moved, paths)-shown(clean, paths)
        expected = offset+sum(event_ramp(t, e, step) for e in steps['events'])
        self.assertLess(np.max(np.abs(diff-expected)), .05)
        standing = [n for n, pid in enumerate(paths) if int(pid[2:]) % 2 == 0]
        speed = np.hypot(*np.diff(diff[standing], axis=1).transpose(2, 0, 1))*FPS
        self.assertLess(np.max(speed[:, (t[1:] > 1) & (t[1:] < 9)]), 2*trajectory.GLIDE_SPEED+.25)

    def test_overlapping_ramps_add_and_every_step_is_cancelled_exactly(self):
        # On arrays: two steps 0.4 s apart whose ramps overlap. Translations: the correction is the two shares' sum.
        # A turn and a scale of the registration about a point plus a shift (projective maps that do not commute): a
        # static ground point P, read through each frame's registration S_t P, is shown at exactly S_t P + sum_v s_v(t)
        # (S_v P - S_{v-1} P) (each step's share of its own jump, the position carried through the other step first:
        # added in the frame's own registration instead, 2-5 mm off), so continuously across both steps, moving no
        # faster than the two ramps' glides together.
        T = 80
        times = np.arange(T)/10.0
        level = np.zeros(T)
        level[[50, 54]] = .5
        maps = np.tile(np.eye(3), (T, 1, 1))
        maps[50, 0, 2], maps[54, 1, 2] = .5, .5
        starts = np.zeros(T, bool)
        starts[0] = True
        events = trajectory.step_events(times, level, maps, starts)
        self.assertEqual([(e['start'], e['spread'], tuple(np.round(e['ramp'], 9))) for e in events], [(50, 'before', (2.5, 5.0)), (54, 'before', (2.9, 5.4))])
        X = np.tile([40.0, 30.0], (T, 1))
        frame = np.arange(T)
        got = trajectory.spread_steps(frame, X, times, events)[0]-X
        s1 = np.clip((times-2.5)/2.5, 0, 1)*(frame < 50)
        s2 = np.clip((times-2.9)/2.5, 0, 1)*(frame < 54)
        np.testing.assert_allclose(got, np.c_[.5*s1, .5*s2], atol=1e-12)
        columns, shares = trajectory.spread_shares(times, events)
        self.assertEqual(columns, [(0, 50), (1, 54)])
        np.testing.assert_allclose(shares, np.c_[s1, s2], atol=1e-12)
        maps[50], maps[54] = about((52.5, 34.0), .01, 1.006, (.3, -.2)), about((20.0, 60.0), -.012, .995, (-.2, .4))
        maps[54, 2, :2] = (1e-5, -2e-5)                                              # a little perspective
        events = trajectory.step_events(times, level, maps, starts)
        self.assertEqual([e['spread'] for e in events], ['before', 'before'])
        columns, shares = trajectory.spread_shares(times, events)
        for ground in ([10.0, 5.0], [52.5, 34.0], [95.0, 60.0]):
            read = chain_reads(maps, (50, 54), ground)
            shown_at = trajectory.spread_steps(frame, read, times, events)[0]
            jumps = {v: read[v]-read[v-1] for _, v in columns}                       # S_v P - S_{v-1} P
            np.testing.assert_allclose(shown_at, read+sum(shares[:, [c]]*jumps[v][None] for c, (_, v) in enumerate(columns)), atol=1e-9)
            speed = np.hypot(*np.diff(shown_at, axis=0).T)/.1
            fastest = sum(np.hypot(*(trajectory._carry(e['map'][None], read[e['start']-1][None])[0]-read[e['start']-1]))/e['rampSeconds']
                          for e in events)
            self.assertLess(speed.max(), fastest*1.01+1e-9)                          # the raw jumps (0.4-0.8 m) are gone
            np.testing.assert_allclose(shown_at[-1], read[-1], atol=1e-12)          # after both: the registration's own
            untouched = times <= min(e['ramp'][0] for e in events)+1e-9
            np.testing.assert_allclose(shown_at[untouched], read[untouched], atol=1e-12)   # before the ramps: as read

    def test_a_trusted_reregistration_is_spread_as_a_keyframe(self):
        # A re-registration at 5 s rooted at the user keyframe at 1 s (the summary lists it with the keyframe's frame as
        # its root): it carries the keyframe's geometry. Its registration differs from both neighbours (an arriving and a
        # leaving step): each is an event of its own, the arriving one spread before it, the leaving one after it, so its
        # frame is shown exactly as registered and nothing jumps (as an automatic anchor's the two were one event spread
        # before it, its frame shown in the registration after it: BC's 24.6 and 34.4 s, 0.12-0.13 m from the keyframes'
        # landmarks as registered, 0.41-0.45 m so).
        t = np.arange(0, 10, 1/FPS)
        paths = team(t)
        n = len(t)
        p, k = int(np.argmax(t >= 1)), int(np.argmax(t >= 5))
        arrive, leave = np.array([.3, .2]), np.array([.4, -.2])
        offset = np.where((np.arange(n) >= k)[:, None], arrive[None], 0.0)+np.where((np.arange(n) > k)[:, None], leave[None], 0.0)
        clean = trajectory.stabilize(make_result(paths, t, seed=61), W, HH)
        result = anchored(make_result(paths, t, seed=61, homographies=shifted(offset)), p, 'landmarks', listed=True, root=p)
        anchored(result, k, 'reregistered', listed=True, root=p)
        moved = trajectory.stabilize(result, W, HH)
        steps = moved['noise']['registrationSteps']
        self.assertEqual((steps['spreadEvents'], steps['jumpFrames'], steps['anchorFrames'], steps['pinnedFrames'], steps['trustedFrames'],
                          steps['trustedEvents']), (2, 0, 2, 1, 1, 2))
        self.assertEqual([(e['reason'], e['keyframe'], e['trusted'], e['spread'], e['squeezed']) for e in steps['events']],
                         [('arriving', False, True, 'before', None), ('leaving', False, True, 'after', None)])
        first, second = steps['events']
        self.assertAlmostEqual(first['rampSeconds'], need(arrive), places=2)
        self.assertAlmostEqual(second['rampSeconds'], need(leave), places=2)
        times, events, _ = trajectory.display_events(result, W, HH)
        X = np.array([[30.0, 20.0], [70.0, 45.0], [52.5, 34.0]])
        np.testing.assert_array_equal(trajectory.spread_steps(np.full(3, k), X, times, events)[0], X)   # no ramp crosses it
        diff = shown(moved, paths)-shown(clean, paths)
        expected = offset+event_ramp(t, first, arrive)+event_ramp(t, second, leave)
        self.assertLess(np.max(np.abs(diff-expected)), .05)
        self.assertLess(np.max(np.abs(diff[:, k]-arrive)), .05)                     # its own registration
        jump = np.hypot(*np.diff(shown(moved, paths), axis=1).transpose(2, 0, 1))
        calm = np.hypot(*np.diff(shown(clean, paths), axis=1).transpose(2, 0, 1))
        self.assertLess(np.max(jump[:, k-1:k+1]-calm[:, k-1:k+1]), .03)             # no jump at either step
        self.assertFalse([f for f in moved['flags'] if f['reason'].startswith('rejected')])

    def test_trusted_anchors_bound_no_ramp_and_turn_when_squeezed(self):
        # On arrays. Veo's case: a keyframe's 2.3 m arriving step 5 s after another keyframe, with re-registrations rooted
        # at that one 1.1-3.1 s after it: the ramp runs back across them to the keyframe (5 s, 0.46 m/s, not squeezed;
        # bounded at the last of them it would have 1.9 s for its 6 s, and every player would snap 2.3 m), each moved by
        # its share of it. (A keyframe also passed as trusted stays a keyframe.)
        times = np.arange(120)/10.0
        level, maps = np.zeros(120), np.tile(np.eye(3), (120, 1, 1))
        starts, anchors, pinned, trusted = (np.zeros(120, bool) for _ in range(4))
        starts[0] = anchors[[15, 26, 36, 46, 65]] = pinned[[15, 65]] = trusted[[26, 36, 46, 65]] = True
        level[65], maps[65, 0, 2] = 2.3, 2.3
        events = trajectory.step_events(times, level, maps, starts, anchors, pinned, None, trusted)
        self.assertEqual([(e['start'], e['reason'], e['keyframe'], e['trusted'], e['spread'], e['squeezed'], tuple(np.round(e['ramp'], 9)))
                          for e in events], [(65, 'arriving', True, False, 'before', None, (1.5, 6.5))])
        X = np.array([[40.0, 30.0]])
        for f in (26, 36, 46):
            np.testing.assert_allclose(trajectory.spread_steps(np.full(1, f), X, times, events)[0]-X, [[2.3*(times[f]-1.5)/5, 0.0]], atol=1e-9)
        # A trusted anchor's leaving step 0.1 s before a registration start (a cut) is squeezed after it: it turns, spread
        # before it back to the keyframe 1.4 s earlier (its frame moved to the registration after the step, the keyframe's
        # not at all). A keyframe's leaving step there snaps. A trusted anchor's arriving step 0.1 s after a cut turns
        # after it (its frame shown in the registration before the step).
        times = np.arange(150)/30.0
        X = np.array([[30.0, 20.0], [70.0, 45.0]])
        for kind, cut, step, expected, shift in (
                ('trusted', 104, 101, (101, 'leaving', False, True, 'before', 'turned', (2.0, round(101/30, 9))), .4*(100/30-2)/(101/30-2)),
                ('keyframe', 104, 101, (101, 'leaving', True, False, None, 'snapped', None), 0.0),
                ('trusted', 97, 100, (100, 'arriving', False, True, 'after', 'turned', (round(100/30, 9), round(100/30+2, 9))), -.4)):
            level, maps = np.zeros(150), np.tile(np.eye(3), (150, 1, 1))
            starts, anchors, pinned, trusted = (np.zeros(150, bool) for _ in range(4))
            starts[[0, cut]] = anchors[[60, 100]] = pinned[60] = True
            (trusted if kind == 'trusted' else pinned)[100] = True
            level[step], maps[step, 0, 2] = .4, .4
            events = trajectory.step_events(times, level, maps, starts, anchors, pinned, None, trusted)
            self.assertEqual([(e['start'], e['reason'], e['keyframe'], e['trusted'], e['spread'], e['squeezed'],
                               None if e['ramp'] is None else tuple(np.round(e['ramp'], 9))) for e in events], [expected], kind)
            np.testing.assert_array_equal(trajectory.spread_steps(np.full(2, 60), X, times, events)[0], X)   # the keyframe: never
            moved = trajectory.spread_steps(np.full(2, 100), X, times, events)[0]-X
            np.testing.assert_allclose(moved, np.tile([shift, 0.0], (2, 1)), atol=1e-9)   # the trusted frame: moved when turned
        # Leaving ramps spread after run across trusted frames too: a keyframe's 0.6 m leaving step and a trusted anchor's,
        # each over its full 3 s across the trusted anchors 0.5-1.5 s after it, each moved by its share (bounded at the
        # first, 0.4 s would glide at 1.5 m/s: the keyframe's step would snap, the trusted one's turn before it).
        times = np.arange(120)/10.0
        level, maps = np.zeros(120), np.tile(np.eye(3), (120, 1, 1))
        starts, anchors, pinned, trusted = (np.zeros(120, bool) for _ in range(4))
        starts[0] = anchors[[15, 20, 30, 50, 55, 65]] = pinned[15] = trusted[[20, 30, 50, 55, 65]] = True
        level[[16, 51]] = .6
        maps[16, 0, 2], maps[51, 1, 2] = .6, .6
        events = trajectory.step_events(times, level, maps, starts, anchors, pinned, None, trusted)
        self.assertEqual([(e['start'], e['reason'], e['keyframe'], e['trusted'], e['spread'], e['squeezed'], tuple(np.round(e['ramp'], 9)))
                          for e in events], [(16, 'leaving', True, False, 'after', None, (1.6, 4.6)),
                                             (51, 'leaving', False, True, 'after', None, (5.1, 8.1))])
        X = np.array([[40.0, 30.0]])
        for f, axis, at in ((20, 0, 1.6), (30, 0, 1.6), (55, 1, 5.1), (65, 1, 5.1)):
            expected = np.zeros((1, 2))
            expected[0, axis] = -.6*(1-(times[f]-at)/3)
            np.testing.assert_allclose(trajectory.spread_steps(np.full(1, f), X, times, events)[0]-X, expected, atol=1e-9, err_msg=str(f))

    def test_only_reregistrations_rooted_at_a_keyframe_are_trusted(self):
        # anchor_frames: a re-registration the summary lists with a keyframe's frame as its root is trusted, also when the
        # keyframe is listed after it, pinned by its frame's calibration only or on the first frame (root 0), when the
        # root or the listed frame is a numpy integer, and when the root names a listed keyframe whose frame index
        # disagrees with its time (the listing is resolved by its time, and a root naming its index with it: listed at
        # 115 at frame 110's time, and at 115 again at its own; a re-registration listed at 68 at frame 65's time is
        # frame 65's, trusted there; and the other way round, listed at 55 at its own time as a paint anchor and at 55
        # again at frame 52's time as a keyframe: any pinned frame its listings resolve to roots it). Automatic: a
        # re-registration rooted at an automatic anchor (all of the amateur run's, rooted at its auto-midfield anchor),
        # or at a listed index that resolves to one (114, listed at frame 112's time, though frame 114 is a keyframe by
        # its calibration), a paint anchor rooted at a keyframe, a re-registration only a frame's calibration shows (no
        # root known), and a root that is missing, a bool (numpy's too), no integer (a float, a string), negative,
        # outside the frames, its own frame or no anchor's; a keyframe's frame is never trusted.
        t = np.arange(0, 4, 1/FPS)
        result = make_result({'A-01': still(t)}, t, noise=None)
        anchored(anchored(anchored(result, 1, 'landmarks'), 90, 'reregistered'), 114, 'landmarks')
        listed = [(20, 'reregistered', 1), (30, 'reregistered', 50), (50, 'landmarks', 50), (50, 'reregistered', 1), (40, 'paint', 1),
                  (60, 'auto-midfield', 60), (70, 'reregistered', 60), (75, 'reregistered', None), (76, 'reregistered', True),
                  (77, 'reregistered', 1.0), (78, 'reregistered', '1'), (79, 'reregistered', -119), (80, 'reregistered', 500),
                  (81, 'reregistered', 81), (82, 'reregistered', 5), (0, 'landmarks', 0), (25, 'reregistered', 0),
                  (35, 'reregistered', np.int64(1)), (36, 'reregistered', np.True_), (37, 'reregistered', np.float64(1.0)),
                  (115, 'landmarks', 115, 110), (115, 'paint', None), (105, 'reregistered', 115), (68, 'reregistered', 1, 65),
                  (np.int64(85), 'reregistered', 1, None), (114, 'reregistered', 1, 112), (100, 'reregistered', 114),
                  (55, 'paint', None), (55, 'landmarks', 55, 52), (106, 'reregistered', 55)]

        def listing(row):            # (frame, source, root[, the frame whose time it has: None for none])
            when = row[3] if len(row) > 3 else row[0]
            return dict({'frame': row[0], 'source': row[1]}, **({} if when is None else {'time': result['frames'][when]['time']}),
                        **({} if row[2] is None else {'root': row[2]}))
        result['summary'] = {'pitch': {'registration': {'anchors': [listing(row) for row in listed]}}}
        anchors, pinned, trusted = trajectory.anchor_frames(result)
        self.assertEqual(np.flatnonzero(anchors).tolist(), [0, 1, 20, 25, 30, 35, 36, 37, 40, 50, 52, 55, 60, 65, 70, 75, 76, 77, 78, 79, 80,
                                                            81, 82, 85, 90, 100, 105, 106, 110, 112, 114, 115])
        self.assertEqual(np.flatnonzero(pinned).tolist(), [0, 1, 50, 52, 110, 114])
        self.assertEqual(np.flatnonzero(trusted).tolist(), [20, 25, 30, 35, 65, 85, 105, 106, 112])
        # In the display: an anchor's leaving step is spread after it only when it is trusted; before it otherwise.
        t = np.arange(0, 6, 1/FPS)
        paths = team(t, 3)
        p, k = int(np.argmax(t >= 1)), int(np.argmax(t >= 3))
        offset = np.where((np.arange(len(t)) > k)[:, None], [.4, -.2], 0.0)
        base = anchored(make_result(paths, t, seed=62, homographies=shifted(offset)), p, 'landmarks', listed=True, root=p)
        anchored(base, 0, 'auto-midfield', listed=True, root=0)
        for source, root in (('reregistered', p), ('reregistered', 0), ('paint', p), ('reregistered', k), ('reregistered', None),
                             ('reregistered', float(p)), ('reregistered', -1), ('reregistered', np.int64(p))):
            result = anchored(json.loads(json.dumps(base)), k, source, listed=True, root=root)
            trusted = source == 'reregistered' and isinstance(root, (int, np.integer)) and root == p
            _, events, _ = trajectory.display_events(result, W, HH)
            self.assertEqual([(e['reason'], e['keyframe'], e['trusted'], e['spread']) for e in events],
                             [('leaving', False, trusted, 'after' if trusted else 'before')], (source, root))

    def test_a_trusted_outlier_is_not_honoured(self):
        # A re-registration rooted at the keyframe at 1 s whose registration is 0.6 m off both neighbours' the same way (an
        # outlier: its arriving and leaving steps cancel) is not honoured: its two steps are an automatic anchor's, one
        # event that composes to about the identity, its frame shown in its neighbours' registration, and nothing ramps
        # (honoured, every player moved by more than 0.3 m for 3 s, towards it and back). The same between two cuts 0.2 s
        # apart (honoured, both its steps snapped: a 1.18 m spike on its frame).
        t = np.arange(0, 10, 1/FPS)
        paths = team(t)
        n = len(t)
        p, k = int(np.argmax(t >= 1)), int(np.argmax(t >= 5))
        offset = np.zeros((n, 2))
        offset[k] = (.6, 0.0)
        clean = trajectory.stabilize(make_result(paths, t, seed=63), W, HH)
        result = anchored(make_result(paths, t, seed=63, homographies=shifted(offset)), p, 'landmarks', listed=True, root=p)
        anchored(result, k, 'reregistered', listed=True, root=p)
        moved = trajectory.stabilize(result, W, HH)
        steps = moved['noise']['registrationSteps']
        self.assertEqual((steps['pinnedFrames'], steps['trustedFrames'], steps['trustedEvents'], steps['spreadEvents'], steps['jumpFrames']),
                         (1, 1, 0, 1, 0))
        self.assertEqual([(e['reason'], e['keyframe'], e['trusted'], e['spread'], e['squeezed']) for e in steps['events']],
                         [('anchor', False, False, 'before', None)])
        self.assertLess(steps['events'][0]['stepMetres']['max'], .01)               # it composes to the identity
        diff = shown(moved, paths)-shown(clean, paths)
        self.assertLess(np.max(np.hypot(*diff.transpose(2, 0, 1))), .03)             # no outlier, no drift
        self.assertEqual(moved['summary']['spreadMetres']['observations'], len(paths))   # only its frame moved
        c = int(np.argmax(t >= 4))
        k = c+3
        cuts = np.where((np.arange(n) >= c)[:, None], [2.5, 0.0], 0.0)+np.where((np.arange(n) >= k+3)[:, None], [1.5, 0.0], 0.0)
        offset = cuts.copy()
        offset[k] += (1.0, .5)

        def run(offset, mark):
            result = make_result(paths, t, seed=64, homographies=shifted(offset))
            for f in (c, k+3):
                result['frames'][f]['cut'], result['frames'][f]['motion'] = True, None
            if mark:
                anchored(anchored(result, p, 'landmarks', listed=True, root=p), k, 'reregistered', listed=True, root=p)
            return trajectory.stabilize(result, W, HH)
        moved, clean = run(offset, True), run(cuts, False)
        steps = moved['noise']['registrationSteps']
        self.assertEqual([(e['reason'], e['trusted'], e['spread'], e['squeezed']) for e in steps['events']], [('anchor', False, 'before', None)])
        self.assertEqual((steps['trustedFrames'], steps['jumpFrames'], steps['notSpread']), (1, 0, 0))
        self.assertLess(np.nanmax(np.hypot(*(shown(moved, paths)-shown(clean, paths)).transpose(2, 0, 1))), .03)   # no spike
        # On arrays: the composite's largest step at the players on the frame before it decides. Out 0.5 m and back 0.48 m
        # (0.02 m net): one automatic event, the trusted frame shown in the registration after it. Back 0.46 m (0.04 m
        # net), or a pair whose net turn about the players' centre is 0.018 m at the median player and 0.42 m at the
        # farthest: honoured, two events, its frame exactly as registered. The same turn with the players around its
        # centre on the frame before it (0.018 m at the farthest; far ones on the other frames, far points on the pitch):
        # an outlier. A keyframe's outlier is honoured always. Right after a keyframe the outlier's arriving step is still
        # its own, not the keyframe's leaving step: one event, spread after it (the room before it is the keyframe's), its
        # frame held at the keyframe's registration. A step back on the next frame that arrives at a keyframe is no
        # leaving step (two arrivals, each honoured); one that arrives at an automatic anchor is, and the outlier's two
        # steps are then two automatic anchors' arrivals, one event.
        times = np.arange(90)/30.0
        X = np.array([[48.0, 33.0], [50.0, 34.0], [52.0, 35.0], [49.0, 34.5], [51.0, 33.5], [95.0, 60.0], [10.0, 5.0]])
        shift = lambda dx: about((0.0, 0.0), 0.0, 1.0, (dx, 0.0))
        c = (50.0, 34.0)
        honoured = [([50], 'arriving', False, True, 'before', None), ([51], 'leaving', False, True, 'after', None)]
        outlier = [([50, 51], 'anchor', False, False, 'before', None)]
        everywhere, close = (lambda i: X), (lambda i: X[:5] if i == 49 else X)
        for name, kind, first, second, points, expected, held in (     # held: the registration frame 50 is shown in
                ('0.02 m net', 'trusted', shift(.5), shift(-.48), everywhere, outlier, 51),
                ('0.04 m net', 'trusted', shift(.5), shift(-.46), everywhere, honoured, 50),
                ('turned', 'trusted', about(c, .01), about(c, -.002), everywhere, honoured, 50),
                ('turned, players close', 'trusted', about(c, .01), about(c, -.002), close, outlier, 51),
                ('keyframe', 'keyframe', shift(.5), shift(-.48), everywhere, [([50], 'arriving', True, False, 'before', None),
                                                                             ([51], 'leaving', True, False, 'after', None)], 50),
                ('after a keyframe', 'keyframe+trusted', shift(.5), shift(-.48), everywhere, [([50, 51], 'anchor', False, False, 'after', 'turned')], 49),
                ('next a keyframe', 'trusted+keyframe', shift(.5), shift(-.48), everywhere, [([50], 'arriving', False, True, 'before', None),
                                                                                            ([51], 'arriving', True, False, 'before', None)], None),
                ('next automatic', 'trusted+automatic', shift(.5), shift(-.48), everywhere, [([50, 51], 'arriving', False, False, 'before', None)], 51)):
            level, maps = np.zeros(90), np.tile(np.eye(3), (90, 1, 1))
            starts, anchors, pinned, trusted = (np.zeros(90, bool) for _ in range(4))
            starts[0] = anchors[[10, 50]] = pinned[10] = True
            (pinned if kind == 'keyframe' else trusted)[50] = True
            anchors[49] = pinned[49] = kind == 'keyframe+trusted'
            anchors[51] = pinned[51] = kind == 'trusted+keyframe'
            anchors[51] |= kind == 'trusted+automatic'
            level[[50, 51]] = .5
            maps[50], maps[51] = first, second
            events = trajectory.step_events(times, level, maps, starts, anchors, pinned, points, trusted)
            self.assertEqual([(e['steps'], e['reason'], e['keyframe'], e['trusted'], e['spread'], e['squeezed']) for e in events], expected, name)
            if held is not None:        # (None: the trusted frame is on the keyframe's arriving ramp)
                read = chain_reads(maps, (50, 51), [52.0, 35.0])
                got = trajectory.spread_steps(np.full(1, 50), read[50][None], times, events)[0][0]
                np.testing.assert_allclose(got, read[held], atol=1e-9, err_msg=name)
        # A blip 0.1 s before a keyframe that has no step of its own, and a trusted outlier right after the keyframe
        # (its return leaves 0.02 m: demoted, its arriving step its own, no leaving step of the keyframe): the blip and
        # the outlier are never grouped across the keyframe, whose frame moves by nothing (grouped, the event's net step
        # spread before the blip moved it 0.02 m).
        level, maps = np.zeros(90), np.tile(np.eye(3), (90, 1, 1))
        starts, anchors, pinned, trusted = (np.zeros(90, bool) for _ in range(4))
        starts[0] = anchors[[10, 49, 50]] = pinned[[10, 49]] = trusted[50] = True
        level[[46, 50, 51]] = .5
        maps[46, 1, 2], maps[50, 0, 2], maps[51, 0, 2] = .3, .5, -.48
        events = trajectory.step_events(times, level, maps, starts, anchors, pinned, None, trusted)
        self.assertEqual([(e['steps'], e['reason'], e['trusted'], e['spread'], e['squeezed']) for e in events],
                         [([46], 'blip', False, 'before', None), ([50, 51], 'anchor', False, 'after', 'turned')])
        np.testing.assert_array_equal(trajectory.spread_steps(np.full(len(X), 49), X, times, events)[0], X)
        # _cancels itself: more than one step, and every point's step finite and under SPREAD_STEP.
        self.assertEqual([trajectory._cancels(np.array(r), n) for r, n in (([.01, .02], 2), ([.01, .02], 1), ([.01, .04], 2), ([.01, np.nan], 2),
                                                                          ([], 2))], [True, False, False, False, False])

    def test_a_step_right_after_an_honoured_anchor_leaves_it(self):
        # A keyframe or a trusted anchor at 5.9 s and an automatic anchor on the next frame whose registration differs
        # from it: the step there is the honoured anchor's leaving step, an event of its own spread after it, and the
        # honoured frame is shown exactly as registered (as an arrival at the automatic anchor, spread before it, it ran
        # across the trusted frame, 0.34 m off its registration, and the keyframe's was squeezed into a turn). A trusted
        # anchor right after a keyframe is no automatic one: the step arrives at it (squeezed before it by the keyframe,
        # spread after it), the keyframe exact as well.
        times = np.arange(90)/10.0
        X = np.array([[30.0, 20.0], [70.0, 45.0]])
        for kind in ('trusted', 'keyframe', 'both'):
            level, maps = np.zeros(90), np.tile(np.eye(3), (90, 1, 1))
            starts, anchors, pinned, trusted = (np.zeros(90, bool) for _ in range(4))
            starts[0] = anchors[[15, 59, 60]] = pinned[15] = True
            (trusted if kind == 'trusted' else pinned)[59] = True
            trusted[60] = kind == 'both'
            level[[59, 60]] = .4, .36
            maps[59, 0, 2] = .4
            maps[60, 0, 2], maps[60, 1, 2] = -.3, .2
            events = trajectory.step_events(times, level, maps, starts, anchors, pinned, None, trusted)
            mark = (kind != 'trusted', kind == 'trusted')
            second = ([60], 'arriving', False, True, 'after', 'turned') if kind == 'both' else ([60], 'leaving', *mark, 'after', None)
            self.assertEqual([(e['steps'], e['reason'], e['keyframe'], e['trusted'], e['spread'], e['squeezed']) for e in events],
                             [([59], 'arriving', *mark, 'before', None), second], kind)
            np.testing.assert_array_equal(trajectory.spread_steps(np.full(2, 59), X, times, events)[0], X)   # as registered
            moved = trajectory.spread_steps(np.full(2, 60), X, times, events)[0]-X
            np.testing.assert_allclose(moved, np.tile([.3, -.2], (2, 1)), atol=1e-9)    # frame 60: held at frame 59's registration
        # The other way round, an automatic anchor on the frame before a keyframe and a registration start 0.2 s before
        # it: the keyframe's step arrives at the keyframe (it leaves no automatic anchor), and with no room both snap,
        # the keyframe exactly as registered (taken for the automatic anchor's leaving step, it was grouped with that
        # one's arrival and the event spread after it: the keyframe's frame 0.22 m off).
        level, maps = np.zeros(90), np.tile(np.eye(3), (90, 1, 1))
        starts, anchors, pinned = np.zeros(90, bool), np.zeros(90, bool), np.zeros(90, bool)
        starts[[0, 57]] = anchors[[15, 59, 60]] = pinned[[15, 60]] = True
        level[[59, 60]] = .4, .36
        maps[59, 0, 2] = .4
        maps[60, 0, 2], maps[60, 1, 2] = -.3, .2
        events = trajectory.step_events(times, level, maps, starts, anchors, pinned)
        self.assertEqual([(e['steps'], e['reason'], e['keyframe'], e['spread'], e['squeezed']) for e in events],
                         [([59], 'arriving', False, None, 'snapped'), ([60], 'arriving', True, None, 'snapped')])
        np.testing.assert_array_equal(trajectory.spread_steps(np.full(2, 60), X, times, events)[0], X)

    def test_an_automatic_outlier_right_after_an_honoured_anchor_disappears(self):
        # A keyframe or a trusted anchor at 5.9 s that agrees with both neighbours, and an automatic anchor on the next
        # frame 0.5 m off them, back on the frame after it: the honoured anchor's leaving step on the automatic anchor's
        # frame takes the automatic anchor's leaving step into its event, spread after it, which composes to the
        # identity: every frame, the honoured one exactly, is shown in the neighbours' registration (two events, the
        # honoured anchor's leaving step spread after it and the automatic anchor's before it, back across the trusted
        # frame, did not cancel: the trusted frame 0.46 m off, the display up to 0.48 m). Out 0.6 m and back to 0.3 m
        # (BC's 37.6 s anchor's shape): one event, the honoured frame exact, the display easing from its registration
        # into the one after the outlier, never beyond either (two events: the trusted frame 0.26 m off, the display
        # 0.28 m beyond the registrations). With a registration start 0.1 s after the outlier (no room after it) the
        # event still cancels: a keyframe's is spread after it all the same (its one way), a trusted anchor's turned
        # before it (the room is there), moving its frame by nothing. An outlier over two frames, back 0.2 s after the
        # automatic anchor (a blip): the same, the blip taken in too. A keyframe's leaving step on a frame that is no
        # anchor's takes nothing: a blip on the next frame is an event of its own. Back on another automatic anchor
        # 0.2 s after the first, the event takes that one's arrival and the blip after it, 0.4 s in all, for either kind
        # (an 'anchor' event, the honoured anchor's, spread after it: its frame exact, the display never back).
        times = np.arange(90)/10.0
        X = np.array([[30.0, 20.0], [70.0, 45.0]])
        ground = np.array([52.0, 35.0])
        for kind in ('trusted', 'keyframe'):
            for name, out, back, b, start, way in (
                    ('outlier', .5, -.5, 61, None, ('after', None)), ('out 0.6, back to 0.3', .6, -.3, 61, None, ('after', None)),
                    ('no room after', .5, -.5, 61, 62, ('after', None) if kind == 'keyframe' else ('before', 'turned')),
                    ('two frames', .5, -.5, 62, None, ('after', None))):
                level, maps = np.zeros(90), np.tile(np.eye(3), (90, 1, 1))
                starts, anchors, pinned, trusted = (np.zeros(90, bool) for _ in range(4))
                starts[0] = anchors[[15, 59, 60]] = pinned[15] = True
                starts[start or 0] = True
                (trusted if kind == 'trusted' else pinned)[59] = True
                level[[60, b]] = out, -back
                maps[60, 0, 2], maps[b, 0, 2] = out, back
                events = trajectory.step_events(times, level, maps, starts, anchors, pinned, None, trusted)
                self.assertEqual([(e['steps'], e['reason'], e['keyframe'], e['trusted'], e['spread'], e['squeezed']) for e in events],
                                 [([60, b], 'leaving', kind == 'keyframe', kind == 'trusted', *way)], (kind, name))
                np.testing.assert_array_equal(trajectory.spread_steps(np.full(2, 59), X, times, events)[0], X)   # as registered
                shown_at = trajectory.spread_steps(np.arange(90), chain_reads(maps, (60, b), ground), times, events)[0]-ground
                np.testing.assert_allclose(shown_at[:b+1], 0.0, atol=1e-9, err_msg=name)    # the outlier's frames held
                np.testing.assert_allclose(shown_at[:, 1], 0.0, atol=1e-9)
                self.assertTrue(np.all(np.diff(shown_at[:, 0]) >= -1e-12), (kind, name))      # never back, never beyond
                self.assertAlmostEqual(float(shown_at[-1, 0]), out+back, places=9)
        level, maps = np.zeros(90), np.tile(np.eye(3), (90, 1, 1))
        starts, anchors, pinned = np.zeros(90, bool), np.zeros(90, bool), np.zeros(90, bool)
        starts[0] = anchors[[15, 59]] = pinned[[15, 59]] = True
        level[[60, 61]] = .4, .3
        maps[60, 0, 2], maps[61, 1, 2] = .4, .3
        events = trajectory.step_events(times, level, maps, starts, anchors, pinned)
        self.assertEqual([(e['steps'], e['reason'], e['keyframe'], e['spread'], e['squeezed']) for e in events],
                         [([60], 'leaving', True, 'after', None), ([61], 'blip', False, 'after', 'turned')])
        for kind in ('trusted', 'keyframe'):
            level, maps = np.zeros(90), np.tile(np.eye(3), (90, 1, 1))
            starts, anchors, pinned, trusted = (np.zeros(90, bool) for _ in range(4))
            starts[0] = anchors[[15, 59, 60, 62]] = pinned[15] = True
            (trusted if kind == 'trusted' else pinned)[59] = True
            level[[60, 62, 64]] = .6, .3, .1
            maps[60, 0, 2], maps[62, 0, 2], maps[64, 0, 2] = .6, -.3, .1
            events = trajectory.step_events(times, level, maps, starts, anchors, pinned, None, trusted)
            self.assertEqual([(e['steps'], e['reason'], e['keyframe'], e['trusted'], e['spread'], e['squeezed']) for e in events],
                             [([60, 62, 64], 'anchor', kind == 'keyframe', kind == 'trusted', 'after', None)], kind)
            shown_at = trajectory.spread_steps(np.arange(90), chain_reads(maps, (60, 62, 64), ground), times, events)[0]-ground
            np.testing.assert_allclose(shown_at[:65], 0.0, atol=1e-9)
            self.assertTrue(np.all(np.diff(shown_at[:, 0]) >= -1e-12), kind)
        # Never taken (30 fps: SPREAD_GAP is six frames): a keyframe's arrival two frames later (taken, the keyframe was
        # carried back 0.2 m), a step 0.23 s later, one after a registration start, one after a trusted frame.
        times = np.arange(90)/30.0
        for name, j in (('keyframe', 62), ('beyond SPREAD_GAP', 67), ('after a start', 62), ('after a trusted frame', 63)):
            level, maps = np.zeros(90), np.tile(np.eye(3), (90, 1, 1))
            starts, anchors, pinned, trusted = (np.zeros(90, bool) for _ in range(4))
            starts[0] = anchors[[15, 59, 60]] = pinned[[15, 59]] = True
            anchors[j] = pinned[j] = name == 'keyframe'
            starts[61] = name == 'after a start'
            anchors[61] = trusted[61] = name == 'after a trusted frame'
            level[[60, j]] = .5
            maps[60, 0, 2], maps[j, 0, 2] = .5, -.3
            events = trajectory.step_events(times, level, maps, starts, anchors, pinned, None, trusted)
            self.assertEqual([e['steps'] for e in events], [[60], [j]], name)
            for f in np.flatnonzero(pinned[50:70])+50:
                np.testing.assert_array_equal(trajectory.spread_steps(np.full(2, f), X, times, events)[0], X)

    def test_an_outlier_between_two_cuts_is_spread_whatever_its_room(self):
        # An automatic anchor 0.1 s after a cut whose registration is 1.1 m off both neighbours', 0.1 s before another
        # cut: its two steps compose to about the identity (under SPREAD_STEP at every player), so the event cancels: with
        # room neither way it is spread its first way, before it into the 0.1 s it has, the anchor's frame carried to its
        # neighbours' registration (left to snap, squeezed both ways, the anchor's frame was a 1.16 m spike).
        t = np.arange(0, 8, 1/FPS)
        paths = team(t)
        n = len(t)
        c = int(np.argmax(t >= 4))
        k = c+3
        cuts = np.where((np.arange(n) >= c)[:, None], [2.5, 0.0], 0.0)+np.where((np.arange(n) >= k+3)[:, None], [1.5, 0.0], 0.0)
        offset = cuts.copy()
        offset[k] += (1.0, .5)

        def run(offset, mark):
            result = make_result(paths, t, seed=51, homographies=shifted(offset))
            for f in (c, k+3):
                result['frames'][f]['cut'], result['frames'][f]['motion'] = True, None
            return trajectory.stabilize(anchored(result, k, 'paint') if mark else result, W, HH)
        moved, clean = run(offset, True), run(cuts, False)
        steps = moved['noise']['registrationSteps']
        self.assertEqual([(e['reason'], e['spread'], e['squeezed']) for e in steps['events']], [('anchor', 'before', None)])
        self.assertEqual((steps['jumpFrames'], steps['notSpread'], steps['squeezed']), (0, 0, {'turned': 0, 'snapped': 0}))
        self.assertLess(steps['events'][0]['stepMetres']['median'], trajectory.SPREAD_STEP)
        self.assertLess(np.nanmax(np.hypot(*(shown(moved, paths)-shown(clean, paths)).transpose(2, 0, 1))), .03)   # no spike
        # On arrays, between two registration starts: one frame of room before it, and none (the start at the event's own
        # time), two frames after it: spread before it. It is a fallback only: with room after it (no second start) the
        # event takes the first way that has the room, as any other does (turned), its frames carried all the same.
        for room, after in ((1/30, 14), (0.0, 14), (1/30, None)):
            times = np.arange(30)/30.0
            times[10] = times[11]-room
            level, maps = np.zeros(30), np.tile(np.eye(3), (30, 1, 1))
            starts, anchors = np.zeros(30, bool), np.zeros(30, bool)
            starts[[0, 10]] = anchors[11] = True
            starts[after or 0] = True
            level[[11, 12]] = 1.0
            maps[11, 0, 2], maps[12, 0, 2] = 1.0, -1.0
            with np.errstate(all='raise'):
                events = trajectory.step_events(times, level, maps, starts, anchors)
                self.assertEqual([(e['steps'], e['reason'], e['spread'], e['squeezed']) for e in events],
                                 [([11, 12], 'anchor', 'before', None) if after else ([11, 12], 'anchor', 'after', 'turned')], (room, after))
                self.assertTrue(np.isfinite(trajectory.spread_shares(times, events)[1]).all())
                shown_at = trajectory.spread_steps(np.arange(30), chain_reads(maps, (11, 12), [40.0, 30.0]), times, events)[0]
            np.testing.assert_allclose(shown_at, np.tile([40.0, 30.0], (30, 1)), atol=1e-9)   # the outlier frame in its neighbours'
        # The largest step at the players decides, not the median: between the same starts, two steps whose net turn about
        # the players' centre is 0.018 m at the median player and 0.42 m at the farthest do not cancel: they snap
        # (spread into the frame of room, a far point jumped 0.36 m at once). A single step never cancels, however small:
        # a keyframe's 0.02 m leaving step 0.1 s before a cut snaps (spread after it into the 0.1 s, it would keep no
        # jump freedom).
        times = np.arange(60)/30.0
        level, maps = np.zeros(60), np.tile(np.eye(3), (60, 1, 1))
        starts, anchors = np.zeros(60, bool), np.zeros(60, bool)
        starts[[0, 20, 24]] = anchors[21] = True
        level[[21, 22]] = .5
        maps[21], maps[22] = about((50.0, 34.0), .01), about((50.0, 34.0), -.002)
        X = np.array([[48.0, 33.0], [50.0, 34.0], [52.0, 35.0], [49.0, 34.5], [51.0, 33.5], [95.0, 60.0], [10.0, 5.0]])
        events = trajectory.step_events(times, level, maps, starts, anchors, points=lambda i: X)
        self.assertEqual([(e['steps'], e['reason'], e['spread'], e['squeezed']) for e in events], [([21, 22], 'anchor', None, 'snapped')])
        self.assertLess(events[0]['reach'], trajectory.SPREAD_STEP)
        self.assertGreater(events[0]['reachMax'], .4)
        times = np.arange(40)/10.0
        level, maps = np.zeros(40), np.tile(np.eye(3), (40, 1, 1))
        starts, anchors = np.zeros(40, bool), np.zeros(40, bool)
        starts[[0, 12]] = anchors[10] = True
        level[11], maps[11, 0, 2] = .05, .02
        events = trajectory.step_events(times, level, maps, starts, anchors, anchors.copy())
        self.assertEqual([(e['start'], e['reason'], e['keyframe'], e['spread'], e['squeezed'], e['ramp']) for e in events],
                         [(11, 'leaving', True, None, 'snapped', None)])
        # Measured at the players, not on the pitch: between two starts as above, an outlier whose return leaves a
        # 0.001 rad turn about the players (0.0016 m at the players, 0.065 m on the pitch grid) cancels: spread before
        # it, no player moves by more than 2 mm at once (judged on the pitch grid it snapped: every player jumped
        # 0.5 m).
        times = np.arange(60)/30.0
        level, maps = np.zeros(60), np.tile(np.eye(3), (60, 1, 1))
        starts, anchors = np.zeros(60, bool), np.zeros(60, bool)
        starts[[0, 20, 24]] = anchors[21] = True
        level[[21, 22]] = .5
        maps[21], maps[22] = about((50.0, 34.0), .01, 1.0, (.5, 0.0)), np.linalg.inv(about((50.0, 34.0), .009, 1.0, (.5, 0.0)))
        X = np.array([[49.0, 33.0], [51.0, 35.0], [50.0, 34.0], [48.5, 34.5]])
        events = trajectory.step_events(times, level, maps, starts, anchors, points=lambda i: X)
        self.assertEqual([(e['steps'], e['reason'], e['spread'], e['squeezed']) for e in events], [([21, 22], 'anchor', 'before', None)])
        for p in X:
            read = chain_reads(maps, (21, 22), p)
            shown_at = trajectory.spread_steps(np.arange(60), read, times, events)[0]
            self.assertGreater(np.hypot(*np.diff(read, axis=0).T).max(), .45)                   # the raw jump
            self.assertLess(np.hypot(*np.diff(shown_at, axis=0).T).max(), .002)

    def test_event_steps_and_frames_on_arrays(self):
        # An automatic anchor two frames after a registration start with an arriving and a leaving step: no room before
        # it, so the event is spread after it, both its frames held at the registration before it (the frame between its
        # steps included), then easing into the registration after it. The same with steps that turn and scale the
        # registration about two points plus a shift (maps that do not commute): each step's share is of its own jump,
        # the position carried to the registration before the step through the other step and its own (a static ground
        # point P is shown at S_t P + sum_v s_v(t) (S_v P - S_{v-1} P) exactly), and the event's map is A_23 A_22.
        times = np.arange(90)/30.0
        shift = lambda dx, dy: about((0.0, 0.0), 0.0, 1.0, (dx, dy))
        for name, first, second in (('shifted', shift(.3, 0.0), shift(0.0, .3)),
                                    ('turned', about((52.5, 34.0), .01, 1.006, (.3, -.2)), about((20.0, 60.0), -.012, .995, (-.2, .4)))):
            level, maps = np.zeros(90), np.tile(np.eye(3), (90, 1, 1))
            starts, anchors = np.zeros(90, bool), np.zeros(90, bool)
            starts[[0, 20]] = anchors[22] = True
            level[[22, 23]] = .3
            maps[22], maps[23] = first, second
            events = trajectory.step_events(times, level, maps, starts, anchors)
            self.assertEqual([(e['steps'], e['reason'], e['spread'], e['squeezed']) for e in events], [([22, 23], 'anchor', 'after', 'turned')], name)
            np.testing.assert_allclose(events[0]['map'], trajectory._normalized(maps[23] @ maps[22]), atol=1e-12, err_msg=name)
            for ground in ([40.0, 30.0], [10.0, 5.0], [95.0, 60.0]):
                read = chain_reads(maps, (22, 23), ground)
                shown_at = trajectory.spread_steps(np.arange(90), read, times, events)[0]
                np.testing.assert_allclose(shown_at[20:24], np.tile(read[21], (4, 1)), atol=1e-9, err_msg=name)
                share = np.clip(1-(times-times[23])/events[0]['rampSeconds'], 0, 1)[:, None]
                np.testing.assert_allclose(shown_at[24:], read[24:]-share[24:]*(read[23]-read[21]), atol=1e-9, err_msg=name)
        # The step is measured at the players on the frame before it: turned by 0.005 rad about the pitch's corner,
        # players around (90, 60) move by 0.54 m and the pitch grid by about 0.3 m (its median): the ramp is the one the
        # players' step asks for (on the other frames the players are near the corner, where the turn moves nothing).
        times = np.arange(100)/10.0
        level, maps = np.zeros(100), np.tile(np.eye(3), (100, 1, 1))
        starts = np.zeros(100, bool)
        starts[0] = True
        level[50], maps[50] = .3, about((0.0, 0.0), .005)
        X = np.array([[85.0, 55.0], [90.0, 60.0], [95.0, 62.0]])
        events = trajectory.step_events(times, level, maps, starts, points=lambda i: X if i == 49 else X/20)
        at = float(np.median(np.hypot(*(trajectory._carry(np.repeat(maps[50][None], len(X), axis=0), X)-X).T)))
        grid = trajectory._pitch_grid()
        on_grid = float(np.median(np.hypot(*(trajectory._carry(np.repeat(maps[50][None], len(grid), axis=0), grid)-grid).T)))
        self.assertGreater(at, 1.5*on_grid)
        self.assertAlmostEqual(events[0]['reach'], at, places=9)
        self.assertAlmostEqual(events[0]['needSeconds'], at/trajectory.GLIDE_SPEED, places=9)
        # A keyframe's or a trusted anchor's arriving step 0.1 s after a blip, no pinned or trusted frame between them: two
        # events (as one, the keyframe's step would be spread with the blip). Two blips 0.1 s apart with a trusted
        # frame between them: two events too (as one, the trusted frame would be carried through the second blip).
        times = np.arange(90)/30.0
        for kind in ('keyframe', 'trusted', 'between'):
            level, maps = np.zeros(90), np.tile(np.eye(3), (90, 1, 1))
            starts, anchors, pinned, trusted = (np.zeros(90, bool) for _ in range(4))
            starts[0] = anchors[60] = True
            {'keyframe': pinned, 'trusted': trusted, 'between': trusted}[kind][60] = True
            at = (57, 60) if kind != 'between' else (59, 62)
            level[list(at)], maps[list(at), 0, 2] = .2, .2
            events = trajectory.step_events(times, level, maps, starts, anchors, pinned, None, trusted)
            self.assertEqual([e['steps'] for e in events], [[at[0]], [at[1]]], kind)
            self.assertEqual([(e['reason'], e['keyframe'], e['trusted']) for e in events],
                             [('blip', False, False), ('arriving', kind == 'keyframe', kind == 'trusted') if kind != 'between' else
                              ('blip', False, False)], kind)

    def test_squeezes_and_event_bounds_on_arrays(self):
        # A keyframe's 2.4 m arriving step 5 s after another keyframe: its full ramp (RAMP_MAX, 6 s) does not fit, but 5 s
        # glides at 0.48 m/s, under twice the full ramp's 0.4 m/s: spread before it over the 5 s, not squeezed (twice
        # GLIDE_SPEED alone would snap it, as it would Veo's 18.8 s keyframe: 2.3 m with room for 5.5 s).
        times = np.arange(120)/10.0
        level, maps = np.zeros(120), np.tile(np.eye(3), (120, 1, 1))
        starts, anchors = np.zeros(120, bool), np.zeros(120, bool)
        starts[0] = anchors[[15, 65]] = True
        level[65], maps[65, 0, 2] = 2.4, 2.4
        events = trajectory.step_events(times, level, maps, starts, anchors, anchors.copy())
        self.assertEqual([(e['start'], e['reason'], e['keyframe'], e['spread'], e['squeezed'], tuple(np.round(e['ramp'], 9))) for e in events],
                         [(65, 'arriving', True, 'before', None, (1.5, 6.5))])
        # Two blips 0.1 s apart with a keyframe between them are two events: as one, the keyframe's frame inside it
        # would be carried through the second blip. The second has 0.07 s of room before it (the keyframe): after it.
        times = np.arange(120)/30.0
        level, maps = np.zeros(120), np.tile(np.eye(3), (120, 1, 1))
        starts, anchors = np.zeros(120, bool), np.zeros(120, bool)
        starts[0] = anchors[60] = True
        level[[59, 62]], maps[[59, 62], 0, 2] = .1, .1
        events = trajectory.step_events(times, level, maps, starts, anchors, anchors.copy())
        self.assertEqual([(e['steps'], e['reason'], e['spread'], e['squeezed']) for e in events], [([59], 'blip', 'before', None), ([62], 'blip', 'after', 'turned')])
        X = np.array([[30.0, 20.0], [70.0, 45.0]])
        np.testing.assert_array_equal(trajectory.spread_steps(np.full(2, 60), X, times, events)[0], X)
        # A 0.035 m blip 0.1 s after a registration start and 0.1 s before a keyframe: no ramp shorter than RAMP_FLOOR (the
        # 0.1 s would glide at 0.35 m/s, under twice GLIDE_SPEED). Nor for a 0.02 m one (0.2 m/s), under SPREAD_STEP at
        # every player but a single step, which never cancels (spread before it into the 0.1 s, it moved no frame and
        # lost its jump freedom).
        times = np.arange(40)/10.0
        for size, expected in ((.035, (11, None, 'snapped', None)), (.02, (11, None, 'snapped', None))):
            level, maps = np.zeros(40), np.tile(np.eye(3), (40, 1, 1))
            starts, anchors = np.zeros(40, bool), np.zeros(40, bool)
            starts[[0, 10]] = anchors[12] = True
            level[11], maps[11, 0, 2] = .05, size
            events = trajectory.step_events(times, level, maps, starts, anchors, anchors.copy())
            self.assertEqual([(e['start'], e['spread'], e['squeezed'], None if e['ramp'] is None else tuple(np.round(e['ramp'], 9)))
                              for e in events], [expected], size)

    def test_no_spreading_across_a_camera_cut(self):
        # A cut at 4 s (the registration starts again, no motion row: a jump of unknown size) and an anchor 1 s after it
        # whose step arrives there (the ramp it asks for would reach before the cut, 1.7 s; 1 s glides at 0.34 m/s, not
        # squeezed): its ramp starts at the cut, nothing before the cut is corrected, and the cut keeps its jump freedom.
        t = np.arange(0, 8, 1/FPS)
        paths = team(t)
        k, k2 = int(np.argmax(t >= 4)), int(np.argmax(t >= 5))
        cut = np.where((t >= 4)[:, None], [2.5, 0.0], 0.0)
        step = np.where((t >= 5)[:, None], [.3, .15], 0.0)

        def run(offset):
            result = anchored(make_result(paths, t, seed=23, homographies=shifted(offset)), k2, 'paint')
            result['frames'][k]['cut'], result['frames'][k]['motion'] = True, None
            return trajectory.stabilize(result, W, HH)
        moved, only_cut = run(cut+step), run(cut)
        steps = moved['noise']['registrationSteps']
        self.assertEqual((steps['cuts'], steps['spreadEvents']), (1, 1))
        self.assertEqual((steps['events'][0]['reason'], steps['events'][0]['spread'], steps['events'][0]['squeezed']), ('arriving', 'before', None))
        self.assertAlmostEqual(steps['events'][0]['rampSeconds'], t[k2]-t[k], places=3)
        self.assertEqual(moved['summary']['splits'], 0)
        diff = shown(moved, paths)-shown(only_cut, paths)
        self.assertLess(np.max(np.abs(diff[:, t < 3.5])), .01)                       # nothing before the cut moves (in its last
        self.assertLess(np.max(np.abs(diff[:, t < 4])), .06)                         # 0.5 s the smoother's velocity reaches over)
        share = np.clip((t-t[k])/(t[k2]-t[k]), 0, 1)
        expected = np.where((t >= 5)[:, None], [.3, .15], share[:, None]*[.3, .15])
        self.assertLess(np.max(np.abs(diff[:, t >= 4]-expected[t >= 4])), .08)       # the ramp runs from the cut
        # On arrays: steps on both sides of a registration start are two events, the second's ramp starts there (0.1 m
        # steps: 0.3 s of room glides at 0.33 m/s, not squeezed).
        times = np.arange(20)/10.0
        level = np.zeros(20)
        level[[8, 13]] = .1
        maps = np.tile(np.eye(3), (20, 1, 1))
        maps[[8, 13], 0, 2] = .1
        starts = np.zeros(20, bool)
        starts[[0, 10]] = True
        anchors = np.zeros(20, bool)
        anchors[[8, 13]] = True
        events = trajectory.step_events(times, level, maps, starts, anchors)
        self.assertEqual([(e['start'], e['end'], e['spread'], e['ramp']) for e in events], [(8, 8, 'before', (0.0, .8)), (13, 13, 'before', (1.0, 1.3))])
        _, share = trajectory.spread_shares(times, events)
        self.assertEqual(share[8:11].tolist(), [[0.0, 0.0]]*3)                       # nothing from the first step to the start
        np.testing.assert_allclose(share[11:13, 1], [1/3, 2/3])

    def test_a_step_right_after_a_registration_start(self):
        # Two frames after a cut a blip has no room before it (the registration starts again at the cut): it is spread
        # after it, and nothing jumps. A user keyframe's step there can be spread neither way (no room before it, and the
        # keyframe is shown as registered): it keeps the jump freedom, every player takes it at once, nothing is rejected
        # or split (it was neither spread nor free: 10 observations rejected or down-weighted, the players 0.68 m off on
        # the two frames between the cut and the step).
        t = np.arange(0, 8, 1/FPS)
        paths = team(t)
        k = int(np.argmax(t >= 4))
        cut = np.where((t >= 4)[:, None], [2.5, 0.0], 0.0)
        step = np.where((np.arange(len(t)) >= k+2)[:, None], [.6, .3], 0.0)

        def run(offset, keyframe=False):
            result = make_result(paths, t, seed=31, homographies=shifted(offset))
            result['frames'][k]['cut'], result['frames'][k]['motion'] = True, None
            if keyframe:
                anchored(result, k+2, 'landmarks')
            return trajectory.stabilize(result, W, HH)
        moved, only_cut = run(cut+step), run(cut)
        steps = moved['noise']['registrationSteps']
        self.assertEqual((steps['spreadEvents'], steps['notSpread'], steps['jumpFrames']), (1, 0, 0))
        event = steps['events'][0]
        self.assertEqual((event['reason'], event['spread'], event['squeezed']), ('blip', 'after', 'turned'))
        diff = shown(moved, paths)-shown(only_cut, paths)
        self.assertLess(np.max(np.abs(diff[:, t < 3.5])), .01)
        self.assertLess(np.max(np.abs(diff-(step+event_ramp(t, event, (.6, .3))))[:, t >= 4]), .05)
        jump = np.hypot(*np.diff(shown(moved, paths), axis=1).transpose(2, 0, 1))
        calm = np.hypot(*np.diff(shown(only_cut, paths), axis=1).transpose(2, 0, 1))
        self.assertLess(np.max(jump[:, k+1]-calm[:, k+1]), .03)                      # no jump at the step (0.67 m raw)
        free = run(cut+step, keyframe=True)
        steps = free['noise']['registrationSteps']
        self.assertEqual((steps['spreadEvents'], steps['notSpread'], steps['jumpFrames'], steps['pinnedFrames']), (0, 1, 1, 1))
        self.assertEqual((steps['events'][0]['reason'], steps['events'][0]['spread'], steps['events'][0]['squeezed']), ('arriving', None, 'snapped'))
        self.assertIsNone(free['summary']['spreadMetres'])
        self.assertEqual(free['summary']['splits'], 0)
        self.assertFalse([f for f in free['flags'] if f['reason'].startswith(('rejected', 'down-weighted'))])
        near = (t >= 3.5) & (t <= 4.5)
        near[k:k+2] = False                    # (the two frames between the cut and the step have their own observations only)
        for pid, truth in paths.items():
            self.assertLess(np.max(np.hypot(*(xy(free['ids'][pid])-truth-cut-step)[near].T)), .3, pid)

    def test_without_steps_the_output_is_unchanged(self):
        # A static registration (identity motion rows, one homography): no step is measured, nothing is spread, and the
        # trajectories are those of the stabilizer before step spreading (the same as without any motion row, where no
        # step can be measured; and, with a copy of soccer/trajectory.py from before it in PITCHIQ_FROZEN_TRAJECTORY,
        # bit for bit that copy's). A gradual change of the registration, up to 0.03 m a frame, is no step either.
        t = np.arange(0, 6, 1/FPS)
        paths = dict(team(t, 5), ghost=np.c_[60+t, np.full_like(t, 20.0)])
        legs = lambda pid, i, b: b*[1, 1, 1, .7] if pid == 'A-02' and 60 <= i < 70 else b
        result = make_result(paths, t, seed=24, anonymous=('ghost',), edit=legs)
        out = trajectory.stabilize(result, W, HH)
        self.assertEqual(out['noise']['registrationSteps']['spreadEvents'], 0)
        self.assertIsNone(out['summary']['spreadMetres'])
        bare = json.loads(json.dumps(result))
        for f in bare['frames']:
            f['motion'] = None
        again = trajectory.stabilize(bare, W, HH)
        self.assertEqual((out['ids'], out['anonymous'], out['flags']), (again['ids'], again['anonymous'], again['flags']))
        drift = np.c_[.9*t/6, np.zeros_like(t)]                                      # 0.005 m a frame (0.15 m/s) ...
        drift[t > 3, 0] += .025*np.arange((t > 3).sum())                             # ... then 0.03 m a frame
        slow = make_result(paths, t, seed=25, anonymous=('ghost',), homographies=shifted(drift))
        self.assertEqual(trajectory.stabilize(slow, W, HH)['noise']['registrationSteps']['spreadEvents'], 0)
        frozen = os.environ.get('PITCHIQ_FROZEN_TRAJECTORY')
        if not frozen:
            return
        spec = importlib.util.spec_from_file_location('soccer._trajectory_frozen', frozen)
        old = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(old)

        def without(d, skip):
            return {k: v for k, v in d.items() if k not in skip}
        for res in (result, slow):
            a, b = trajectory.stabilize(res, W, HH), old.stabilize(res, W, HH)
            self.assertEqual(json.dumps([a['frameTimes'], a['ids'], a['anonymous'], a['flags']]),
                             json.dumps([b['frameTimes'], b['ids'], b['anonymous'], b['flags']]))
            self.assertEqual(without(a['summary'], ('runtimeSeconds', 'spreadMetres')), without(b['summary'], ('runtimeSeconds', 'spreadMetres')))
            self.assertEqual(without(a['noise'], ('registrationSteps',)), without(b['noise'], ('registrationSteps',)))
            old_steps = b['noise'].get('registrationSteps') or {}
            self.assertEqual({k: a['noise']['registrationSteps'][k] for k in old_steps}, old_steps)

    def test_a_moving_player_takes_the_step_without_lag_or_glide(self):
        # Runners at 5-6 m/s (one along the camera's line of sight), a sharp cut at 6 m/s and a standing player through a
        # step of 0.73 m at 4 s: each is shown where the clean registration shows it plus the ramp's offset, at its own
        # speed plus the ramp's, and from the step on at the new registration (no glide after it).
        t = np.arange(0, 8, 1/FPS)
        paths = {'A-01': np.c_[20+5*t, np.full_like(t, 30.0)], 'A-02': depth(t, -6*np.minimum(t, 6.0), 60.0),
                 'A-03': cut_path(t, 6.0, .3), 'A-04': still(t, 70.0, 20.0)}
        step = np.array([.7, -.2])
        offset = np.where((t >= 4)[:, None], step[None], 0.0)
        clean = trajectory.stabilize(make_result(paths, t, seed=26), W, HH)
        moved = trajectory.stabilize(make_result(paths, t, seed=26, homographies=shifted(offset)), W, HH)
        steps = moved['noise']['registrationSteps']
        self.assertEqual((steps['spreadEvents'], steps['events'][0]['spread']), (1, 'before'))
        a, b = shown(moved, paths), shown(clean, paths)
        expected = event_ramp(t, steps['events'][0], step)+offset
        self.assertLess(np.max(np.abs(a-b-expected)), .06)
        after = (t >= 4) & (t <= 4.5)
        self.assertLess(np.max(np.abs(a[:, after]-b[:, after]-step)), .03)           # no glide after the step
        va, vb = np.gradient(a, t, axis=1), np.gradient(b, t, axis=1)
        inner = (t > .5) & (t < 7.5)
        self.assertLess(np.max(np.abs(va-vb-np.gradient(expected, t, axis=0)[None])[:, inner]), .3)   # the speed is kept

    def test_anonymous_tracks_are_spread_alike(self):
        t = np.arange(0, 8, 1/FPS)
        paths = dict(team(t, 4), ghost=np.c_[60+2*t, np.full_like(t, 20.0)], other=still(t, 40.0, 50.0))
        tracks = {pid: np.full(len(t), 100+n) for n, pid in enumerate(paths)}
        step = np.array([-.6, .4])
        offset = np.where((t >= 4)[:, None], step[None], 0.0)
        clean = trajectory.stabilize(make_result(paths, t, seed=27, tracks=tracks, anonymous=('ghost', 'other')), W, HH)
        moved = trajectory.stabilize(make_result(paths, t, seed=27, tracks=tracks, anonymous=('ghost', 'other'), homographies=shifted(offset)), W, HH)
        self.assertEqual(set(moved['anonymous']), {str(tracks['ghost'][0]), str(tracks['other'][0])})
        expected = event_ramp(t, moved['noise']['registrationSteps']['events'][0], step)+offset
        for key in moved['anonymous']:
            diff = xy(moved['anonymous'][key])-xy(clean['anonymous'][key])
            self.assertLess(np.max(np.abs(diff-expected)), .05, key)
        named = [p for p in paths if p.startswith('A-')]
        self.assertLess(np.max(np.abs(shown(moved, named)-shown(clean, named)-expected)), .05)


class RobustnessTests(unittest.TestCase):
    def test_an_empty_result_and_frames_without_a_time(self):
        out = trajectory.stabilize({'frames': [], 'roster': {'slots': []}}, W, HH)
        self.assertEqual((out['frameTimes'], out['ids'], out['anonymous'], out['flags']), ([], {}, {}, []))
        t = np.arange(0, 4, 1/FPS)
        result = make_result({'A-01': np.c_[40+2*t, np.full_like(t, 30.0)]}, t, seed=14)
        result['frames'][30]['time'] = None
        out = trajectory.stabilize(result, W, HH)
        self.assertEqual(len(out['frameTimes']), len(t))
        self.assertAlmostEqual(out['frameTimes'][30], t[30], places=3)                # interpolated
        self.assertIn(out['ids']['A-01']['kind'][30], ('I', None))                    # no observation there
        json.dumps(out, allow_nan=False)

    def test_one_tiny_time_step_does_not_widen_the_windows(self):
        # One re-stamped frame 1 ms after its neighbour: the edge fits' windows were sized by the smallest step (32 times
        # wider here: 8.4 GB for 20 players x 60 s); they are sized by the typical step, the memory stays the same.
        def peak(times):
            paths = {f'A-{n:02d}': np.c_[20+3*n+1.5*times % 40, np.full_like(times, 10.0+2*n)] for n in range(1, 7)}
            result = make_result(paths, times, seed=15)
            tracemalloc.start()
            trajectory.stabilize(result, W, HH)
            used = tracemalloc.get_traced_memory()[1]
            tracemalloc.stop()
            return used
        regular = np.arange(0, 20, 1/FPS)
        tiny = regular.copy()
        tiny[100] = tiny[99]+.001
        self.assertLess(peak(tiny), 1.5*peak(regular))
        run = np.zeros(len(tiny), np.int64)
        self.assertAlmostEqual(trajectory._frame_step(run, tiny), 1/FPS, places=4)

    def test_a_given_noise_model_is_used(self):
        # A corrected run keeps its original's fitted noise (soccer/stabilized.py): the same result with its own noise
        # model handed in gives the same trajectories.
        t = np.arange(0, 4, 1/FPS)
        result = make_result({'A-01': np.c_[40+2*t, np.full_like(t, 30.0)], 'A-02': still(t, 60.0, 20.0)}, t, seed=16)
        first = trajectory.stabilize(result, W, HH)
        again = trajectory.stabilize(result, W, HH, noise=json.loads(json.dumps(first['noise'])))
        self.assertEqual(first['ids'], again['ids'])
        self.assertEqual(first['noise']['bottom'], again['noise']['bottom'])
        # With box events and doubtful boxes (a box cut at the hips for 1.5 s, the legs a box of their own) the
        # geometric height's fit has two passes, the second without the first's events and doubts: the model handed in
        # replays both (its 'firstPass'), so the trajectories are the same (with the final model in both passes, handing
        # the Veo run its own model moved 1072 samples by up to 0.25 m).
        t = np.arange(0, 8, 1/FPS)
        truth = np.c_[45+1.5*t, np.full_like(t, 28.0)]
        cut = (t >= 3.0) & (t < 4.5)
        P = camera()

        def legs(i):
            x, y, w, h = box_px(P, *truth[i])
            return [[x, y+.6*h-2, w, .4*h+2]] if cut[i] else []
        result = make_result({'A-01': truth, 'A-02': still(t, 60.0, 20.0)}, t, extra=legs, seed=3,
                             edit=lambda pid, i, b: b*[1, 1, 1, .6] if pid == 'A-01' and cut[i] else b)
        first = trajectory.stabilize(result, W, HH)
        self.assertEqual(set(first['noise']['geometric']['firstPass']), {'ratio', 'a', 'b'})
        again = trajectory.stabilize(result, W, HH, noise=json.loads(json.dumps(first['noise'])))
        self.assertEqual(first['ids'], again['ids'])
        # Where the two passes' fits differ (here the first forced much tighter), box_events with the model it returned
        # gives what the fit gave: the same ratios, doubts, references and repairs; and the first pass is given the first
        # pass's model, the second the final one (on the Veo run the two differ: ratio .391 / .396, 2.84 / 2.20 px).
        size = field.PitchSize()
        obs = trajectory.observations(result, W, HH, size)
        H, G = trajectory._homographies(result['frames'], W, HH)
        fits = trajectory.edge_fits(obs, H, G)
        real, calls = trajectory._fit_level, []

        def tighter_first(*args):
            a, b, n = real(*args)
            calls.append(1)
            return (a/20, b/20, n) if len(calls) == 1 else (a, b, n)
        with mock.patch.object(trajectory, '_fit_level', tighter_first):
            fitted = trajectory.box_events(obs, fits)
        model = json.loads(json.dumps(fitted['model']))
        self.assertLess(model['firstPass']['a'], model['a'])
        replay = trajectory.box_events(obs, fits, model)
        for key in ('event', 'repaired', 'href', 'doubt', 'ratio', 'sgeo'):
            np.testing.assert_array_equal(replay[key], fitted[key], err_msg=key)
        real_heights, ratios = trajectory.geometric_heights, []

        def recording(obs_, fits_, exclude, run_ratio=None):
            ratios.append(run_ratio)
            return real_heights(obs_, fits_, exclude, run_ratio)
        with mock.patch.object(trajectory, 'geometric_heights', recording):
            trajectory.box_events(obs, fits, {'ratio': .5, 'a': 2.0, 'b': .05, 'firstPass': {'ratio': .3, 'a': 1.0, 'b': .02}})
            trajectory.box_events(obs, fits, {'ratio': .5, 'a': 2.0, 'b': .05})     # an older model: the final one twice
        self.assertEqual(ratios, [.3, .5, .5, .5])
        real_noise, levels = trajectory._noise, []

        def noting(ab, h):
            levels.append(tuple(ab))
            return real_noise(ab, h)
        with mock.patch.object(trajectory, '_noise', noting):
            trajectory.box_events(obs, fits, {'ratio': .5, 'a': 2.0, 'b': .05, 'firstPass': {'ratio': .3, 'a': 1.0, 'b': .02}})
        self.assertEqual(levels, [(1.0, .02), (2.0, .05)])                    # each pass its own geometric noise

    def test_a_kept_model_keeps_the_footage_noise_level(self):
        # The box-edge fits scale their biweight by the footage's own noise level, a median over every observation:
        # measured again without an excluded player's observations, it moved other players (on the amateur run 36
        # samples of two players by up to 0.08 m when B-08 was excluded). The noise model carries it ('footage': the
        # level, the fits' span and the frame step), and fitting the other players' observations with it reproduces the
        # full fit's residuals for them exactly.
        t = np.arange(0, 6, 1/FPS)
        result = make_result(team(t, 5), t, seed=31, noise=(1.4, .03))                  # noisier than the prior
        size = field.PitchSize()
        obs = trajectory.observations(result, W, HH, size)
        H, G = trajectory._homographies(result['frames'], W, HH)
        full = trajectory.edge_fits(obs, H, G)
        self.assertEqual(set(full['footage']), {'level', 'span', 'step'})
        self.assertGreater(min(full['footage']['level']), 1/1.4826)                    # above the prior: it scales the fits
        for f in result['frames']:
            for person in f['people']:
                if person.get('id') == 'A-03':
                    person.update(id=None, display=None, team=None, state='rejected', role='UNKNOWN')
        rest = trajectory.observations(result, W, HH, size)
        keep = np.isin(obs['key'], [k for k, key in enumerate(obs['keys']) if key != ('id', 'A-03')])
        self.assertEqual(len(rest['frame']), int(keep.sum()))
        kept = trajectory.edge_fits(rest, H, G, json.loads(json.dumps(full['footage'])))
        for name in ('residual', 'g', 'hmed', 'long'):
            np.testing.assert_array_equal(kept[name], full[name][keep], err_msg=name)
        self.assertEqual(kept['footage'], full['footage'])
        refit = trajectory.edge_fits(rest, H, G)
        self.assertNotEqual(refit['footage']['level'], full['footage']['level'])     # measured again: another level


class ProjectionTests(unittest.TestCase):
    def test_an_unreliable_projection_keeps_its_uncertainty_and_is_not_forced_onto_the_pitch(self):
        # A camera 1.5 m high far behind the touchline: the ground is seen at about one degree and a vertical pixel is
        # worth metres. The player stands beyond the far touchline.
        P = camera(height=1.5, back=60.0, target=(52.5, 10.0), focal=1400.0)
        t = np.arange(0, 4, 1/FPS)
        truth = still(t, 50.0, -3.0)
        out = trajectory.stabilize(make_result({'A-01': truth}, t, P=P, seed=10), W, HH)
        good = trajectory.stabilize(make_result({'A-01': truth}, t, seed=10), W, HH)
        track = out['ids']['A-01']
        sy = series(track, 'sy')
        self.assertGreater(np.nanmedian(sy), .3)                                 # the uncertainty stays large: several
        self.assertGreater(np.nanmedian(sy), 4*np.nanmedian(series(good['ids']['A-01'], 'sy')))   # times the pole camera's
        raw = raw_xy(make_result({'A-01': truth}, t, P=P, seed=10), 'A-01')
        self.assertGreater(np.nanstd(raw[:, 1]), 2.0)                            # (a vertical pixel is worth metres here)
        est = xy(track)[np.array([k == 'O' for k in track['kind']])]
        self.assertGreater(len(est), .8*len(t))
        self.assertTrue(np.isfinite(est).all())
        self.assertLess(np.median(est[:, 1]), 0.0)                               # outside the pitch, where the player is
        self.assertLess(np.median(np.hypot(*(est-truth[:len(est)]).T)), 3.0)


class NoiseModelTests(unittest.TestCase):
    def test_the_pixel_noise_is_fitted_from_the_run(self):
        # Players near and far (boxes of about 25-85 px) with noise a = 0.8 px, b = 0.02: the fit moves from the priors
        # (0.45, 0.011) to the footage's noise; the shared fit of a and b leaves the box-height dependence a little short.
        t = np.arange(0, 10, 1/FPS)
        paths = {f'A-{n:02d}': still(t, 25+9*(n % 7), 4+2.6*n) for n in range(1, 25)}
        result = make_result(paths, t, noise=(.8, .02), seed=11)
        heights = [p['box'][3]*HH for p in result['frames'][0]['people']]
        self.assertLess(min(heights), 32)
        self.assertGreater(max(heights), 75)
        out = trajectory.stabilize(result, W, HH)
        bottom = out['noise']['bottom']
        self.assertGreater(bottom['fittedSamples'], 3000)
        for h in (30, 50, 80):
            fitted, true, prior = math.hypot(bottom['a'], bottom['b']*h), math.hypot(.8, .02*h), math.hypot(*trajectory.PRIOR_BOTTOM[:1], trajectory.PRIOR_BOTTOM[1]*h)
            self.assertLess(abs(fitted-true), .15*true, (h, bottom))
            self.assertLess(abs(fitted-true), .5*abs(prior-true), (h, bottom))


class ArrayTests(unittest.TestCase):
    def test_smooth_tracks_on_arrays(self):
        rng = np.random.default_rng(12)
        t = np.arange(0, 5, 1/25)
        T, K = len(t), 3
        truth = np.stack([np.c_[30+k+2*t, 20+5*k+0*t] for k in range(K)], axis=1)
        R = np.tile(np.diag([.1**2, .4**2]), (T, K, 1, 1))
        z = truth+rng.normal(size=truth.shape)*[.1, .4]
        z[40:60, 1] = np.nan                                                     # 0.8 s gap: bridged
        z[:, 2][t > 2.5] = np.nan                                                # column 2 ends
        out = trajectory.smooth_tracks(t, z, R)
        self.assertEqual(out['x'].shape, (T, K, 2))
        self.assertEqual(out['v'].shape, (T, K, 2))
        self.assertEqual(out['P'].shape, (T, K, 4, 4))
        self.assertEqual(out['w'].shape, (T, K))
        self.assertTrue((out['kind'][40:60, 1] == 'I').all())
        self.assertTrue((out['kind'][t > 2.5, 2] == '').all())
        self.assertTrue(np.isnan(out['x'][t > 2.5, 2]).all())
        self.assertTrue(np.isnan(out['w'][40:60, 1]).all())
        err = np.hypot(*(out['x'][:, 0]-truth[:, 0]).T)
        self.assertLess(np.sqrt(np.mean(err**2)), .12)
        # Starts: a column holding two tracks one after another is never smoothed across.
        z2 = np.full((T, 1, 2), np.nan)
        z2[:60, 0] = truth[:60, 0]
        z2[62:, 0] = truth[62:, 0]+[0, 4.0]
        starts = np.zeros((T, 1), bool)
        starts[62, 0] = True
        out2 = trajectory.smooth_tracks(t, z2, R[:, :1], options={'starts': starts})
        self.assertNotEqual(out2['segment'][59, 0], out2['segment'][62, 0])
        self.assertTrue((out2['kind'][60:62, 0] == '').all())


if __name__ == '__main__':
    unittest.main()
