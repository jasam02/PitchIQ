"""Full Match Mode's calibration discovery (fullmatch/calibration.py) on synthetic cameras, without a video.

camera_of recovers a pinhole camera; the consensus ignores a minority of wrong cameras; discovery refuses without a
majority camera and without an anchor that reaches MIN_OBS; the temporal rule drops an anchor that sits at the match
camera but that no neighbour agrees with (and one with no neighbour at all), and a wrong registration never enters a
keyframe's fusion; the keyframes lie inside their segment's
processed frames, at most PER_SEGMENT, MIN_GAP_S apart, carry the frame's own timestamp, and calibration.solve takes
their points; they reach every second of a segment within REACH_S where anchors allow, and a segment anchored only
late gets its keyframes there, the earliest reaching back as far as it can; a segment whose anchors cannot be fused
keeps them unfused; a user keyframe wins within MANUAL_REACH_S and the segment keeps its other keyframes; sampling and
chunking never cross a period and every pair is computed exactly once; the registration recovers a pan and ignores a
burned-in overlay.

Run:  python -m unittest test_fullmatch_calibration          (from this directory)
"""
import math
import os
import sys
import unittest

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fullmatch import calibration as cal, plan  # noqa: E402
from soccer import calibration as solver, field, registration  # noqa: E402

W, H = 1920, 1080
FPS = 25.0
CAMERA = (53.7, 113.0, 24.3)   # CAN-ISL's tactical camera: ground position (x, y) and height, metres
STEP = int(round(cal.STEP_S*FPS))


def homography(position, target, focal, yaw=0.0):
    """World metres -> pixels of a pinhole camera (square pixels, principal point at the centre) at position (x, y,
    height) looking at the ground point target; yaw turns it about the vertical (radians) after aiming."""
    C = np.array([position[0], position[1], -position[2]], np.float64)   # z points into the ground
    z = np.array([target[0], target[1], 0.0])-C
    z /= np.linalg.norm(z)
    if yaw:
        c, s = math.cos(yaw), math.sin(yaw)
        z = np.array([c*z[0]-s*z[1], s*z[0]+c*z[1], z[2]])
    down = np.array([0.0, 0.0, 1.0])
    y = down-(down @ z)*z
    y /= np.linalg.norm(y)
    x = np.cross(y, z)
    R = np.vstack([x, y, z])
    t = -R @ C
    K = np.array([[focal, 0, W/2], [0, focal, H/2], [0, 0, 1.0]])
    P = K @ np.c_[R[:, 0], R[:, 1], t]
    return field.orient(P/np.linalg.norm(P), W, H)


def view(k):
    """The match camera's view at sampled frame k: a slow pan along the pitch and a slow zoom."""
    return homography(CAMERA, (52.5+28*math.sin(k/23.0), 36+4*math.cos(k/17.0)), 1650+250*math.sin(k/31.0))


def anchor_row(index, Hm, observability=.8, route='pinhole'):
    row = {'index': index, 'time': round(index/FPS-.0116, 6), 'ok': True, 'H': Hm.tolist(), 'observability': observability, 'route': route}
    cam = cal.camera_of(Hm, W, H)
    row['camera'] = {'focal': cam[0], 'position': list(cam[1]), 'height': cam[2]}
    return row


def match_rows(n, wrong=(), drift=(), observability=.8, no_pairs=()):
    """n sampled frames of the match camera; frames in `wrong` are anchored by a partial-arc style camera (on the pitch,
    4.5 m high); frames in `drift` by the right camera turned 3 degrees (its camera passes the consensus, its pitch does
    not); pairs are the true image motion between frames up to max(GAPS) apart (none touching `no_pairs`)."""
    truth = [view(k) for k in range(n)]
    rows = []
    for k in range(n):
        if k in wrong:
            Hm = homography((52.0+3*math.sin(k), 54.0, 4.5), (52.5+20*math.sin(k/7.0), 30), 900)
        elif k in drift:
            Hm = homography(CAMERA, (52.5+28*math.sin(k/23.0), 36+4*math.cos(k/17.0)), 1650+250*math.sin(k/31.0), yaw=math.radians(3))
        else:
            Hm = truth[k]
        rows.append(anchor_row(k*STEP, Hm, observability))
    pairs = []
    for k in range(n):
        for g in cal.GAPS:
            if k+g < n and k not in no_pairs and k+g not in no_pairs:
                pairs.append({'a': k*STEP, 'b': (k+g)*STEP, 'ok': True, 'G': (truth[k+g] @ np.linalg.inv(truth[k])).tolist(),
                              'inliers': 500, 'rms': .5, 'overlap': .8})
    return rows, pairs, truth


def segments_for(n):
    return plan.plan(n*STEP, FPS, [(0.0, n*STEP/FPS)])


class CameraTests(unittest.TestCase):
    def test_camera_of_recovers_a_pinhole_camera(self):
        for position, target, focal in ((CAMERA, (40, 30), 1700), ((52.5, 110, 27.6), (70, 34), 2500), ((52.4, 70.3, 4.8), (52.5, 34), 1300)):
            f, (x, y), h = cal.camera_of(homography(position, target, focal), W, H)
            self.assertAlmostEqual(f, focal, delta=focal*1e-6)
            self.assertAlmostEqual(x, position[0], delta=1e-4)
            self.assertAlmostEqual(y, position[1], delta=1e-4)
            self.assertAlmostEqual(h, position[2], delta=1e-4)

    def test_the_consensus_ignores_a_minority_of_wrong_cameras(self):
        rng = np.random.default_rng(0)
        anchors = []
        for k in range(70):
            p = (CAMERA[0]+rng.normal(0, .5), CAMERA[1]+rng.normal(0, .5), CAMERA[2]+rng.normal(0, .3))
            anchors.append(anchor_row(k, homography(p, (50+k % 20, 34), 1700)))
        for k in range(30):
            p = (rng.uniform(20, 90), rng.uniform(40, 70), rng.uniform(2, 20))
            anchors.append(anchor_row(100+k, homography(p, (50, 34), 900)))
        (x, y), h, share, n = cal.consensus(anchors)
        self.assertLess(math.hypot(x-CAMERA[0], y-CAMERA[1]), .6)
        self.assertLess(abs(h-CAMERA[2]), .4)
        self.assertAlmostEqual(share, .7, delta=.03)
        self.assertEqual(n, 100)


class RefusalTests(unittest.TestCase):
    def test_no_majority_camera_is_refused(self):
        rows, pairs, _ = match_rows(60, wrong=set(range(0, 60, 2)))
        index = cal.discover(rows, pairs, segments_for(60), FPS, W, H)
        self.assertIsNotNone(index['refused'])
        self.assertIn('do not agree on one camera', index['refused'])
        self.assertEqual(index['keyframes'], [])
        self.assertEqual(index['uncovered'], [s['index'] for s in index['segments']])

    def test_nothing_reaching_min_obs_is_refused(self):
        rows, pairs, _ = match_rows(60, observability=cal.MIN_OBS-.05)
        index = cal.discover(rows, pairs, segments_for(60), FPS, W, H)
        self.assertIn('constrained well enough', index['refused'])
        self.assertEqual(index['keyframes'], [])
        self.assertEqual(cal.keyframes_for(index, 0), [])

    def test_anchors_no_neighbour_confirms_are_refused(self):
        rows, _, _ = match_rows(60)
        index = cal.discover(rows, [], segments_for(60), FPS, W, H)
        self.assertIn('confirmed by a neighbouring frame', index['refused'])
        self.assertEqual(index['keyframes'], [])

    def test_too_few_anchors_are_refused(self):
        rows, pairs, _ = match_rows(3)
        index = cal.discover(rows, pairs, segments_for(3), FPS, W, H)
        self.assertIn('Too few automatic anchors', index['refused'])


class RuleTests(unittest.TestCase):
    def test_a_neighbour_must_agree(self):
        n = 120
        rows, pairs, _ = match_rows(n, wrong={10, 11, 12}, drift={40}, no_pairs={70})
        index = cal.discover(rows, pairs, segments_for(n), FPS, W, H)
        self.assertIsNone(index['refused'])
        self.assertGreater(index['camera']['share'], .9)
        frames = {k['frame'] for k in index['keyframes']}
        # The three wrong anchors agree with each other (the same failure on consecutive frames) but not with the camera.
        self.assertFalse(frames & {10*STEP, 11*STEP, 12*STEP})
        self.assertEqual(index['kept'], n-5)   # wrong x3, the drifted one, the one without a neighbour
        by = {r['index']: r for r in rows}
        agree = cal.agreements([dict(r) for r in rows], cal._graph(pairs), W, H)
        self.assertTrue(all(d > cal.AGREE_M for _, d in agree[40*STEP]))
        self.assertTrue(any(d <= cal.AGREE_M for _, d in agree[41*STEP]))
        self.assertNotIn(70*STEP, agree)
        self.assertTrue(cal._camera_ok(by[40*STEP], tuple(index['camera']['position']), index['camera']['height']))


    def test_a_wrong_registration_never_enters_the_fusion(self):
        rows, pairs, truth = match_rows(60)
        T = np.array([[1.0, 0, 80], [0, 1, 0], [0, 0, 1]])   # frame 30's registrations to all but one neighbour put it 80 px right
        for p in pairs:
            a, b = p['a']//STEP, p['b']//STEP
            if 30 in (a, b) and 29 not in (a, b):
                j = b if a == 30 else a
                wrong = T @ truth[30] @ np.linalg.inv(truth[j])   # neighbour j's pixels -> frame 30's, shifted
                p['G'] = (wrong if b == 30 else np.linalg.inv(wrong)).tolist()
        G = cal._graph(pairs)
        anchors = {r['index']: dict(r, agreements=1) for r in rows}
        agree = cal.agreements(list(anchors.values()), G, W, H)
        agreeing = {i: {j for j, d in agree.get(i, ()) if d <= cal.AGREE_M} for i in anchors}
        self.assertEqual(agreeing[30*STEP], {29*STEP})
        Hf, n = cal.fused(anchors[30*STEP], anchors, G, W, H, agreeing)
        self.assertIsNone(Hf)   # the anchor and one agreeing neighbour: fewer than FUSE_MIN
        Hall, n_all = cal.fused(anchors[30*STEP], anchors, G, W, H)
        self.assertEqual(n_all, 7)   # every registered neighbour, the wrong ones included, would move it
        self.assertGreater(registration._disagreement(truth[30], Hall, W, H), cal.AGREE_M)


class KeyframeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.n = 300   # 600 s: four segments of 150 s
        cls.rows, cls.pairs, cls.truth = match_rows(cls.n)
        cls.segments = segments_for(cls.n)
        cls.index = cal.discover(cls.rows, cls.pairs, cls.segments, FPS, W, H)

    def test_keyframes_lie_in_their_segment_spaced_and_valid(self):
        self.assertIsNone(self.index['refused'])
        self.assertEqual(self.index['uncovered'], [])
        stamps = {r['index']: r['time'] for r in self.rows}
        for s in self.segments:
            kfs = cal.keyframes_for(self.index, s)
            self.assertTrue(1 <= len(kfs) <= cal.PER_SEGMENT, len(kfs))
            frames = sorted(k['frame'] for k in kfs)
            self.assertTrue(all(s.proc0 <= f < s.proc1 for f in frames))
            self.assertTrue(all(b-a >= cal.MIN_GAP_S*FPS for a, b in zip(frames, frames[1:])))
            for k in kfs:
                self.assertEqual(k['time'], stamps[k['frame']])   # the decoder's timestamp of that frame
                self.assertEqual(k['source'], 'discovery')
                self.assertGreaterEqual(k['fusedAnchors'], cal.FUSE_MIN)
                self.assertGreaterEqual(len(k['points']), cal.MIN_POINTS)
                self.assertTrue(all(0 <= p['x'] <= 1 and 0 <= p['y'] <= 1 for p in k['points']))
                Hk, rms = solver.solve(k['points'], W, H, max_error=4.0)
                self.assertLess(rms, .01)
                true = self.truth[k['frame']//STEP]
                marks = field.landmarks(field.PitchSize())
                px = field.project(true, [marks[p['name']] for p in k['points']])
                self.assertLess(float(np.abs(px-np.array([(p['x']*W, p['y']*H) for p in k['points']])).max()), .5)

    def test_keyframes_reach_every_second_of_their_segment(self):
        # Every sampled frame is a kept anchor here, so every second of a segment can be within REACH_S of a keyframe:
        # a run given keyframes turns its automatic calibration off, and a second they do not reach stays uncalibrated.
        for s in self.segments:
            times = [k['time'] for k in cal.keyframes_for(self.index, s)]
            grid = np.arange(s.proc0, s.proc1, FPS)/FPS
            self.assertTrue(all(min(abs(t-g) for t in times) <= cal.REACH_S for g in grid), (s.index, times))

    def test_a_segment_anchored_only_late_gets_its_keyframes_there(self):
        n = 75   # 150 s, one segment; its first 80 s solve to nothing (CAN-ISL's midfield spells)
        rows, pairs, _ = match_rows(n)
        rows = [dict(r, ok=False, H=None) if r['index'] < 40*STEP else r for r in rows]
        index = cal.discover(rows, pairs, segments_for(n), FPS, W, H)
        frames = sorted(k['frame'] for k in cal.keyframes_for(index, 0))
        self.assertTrue(frames)
        self.assertGreaterEqual(frames[0], 40*STEP)   # none invented where nothing was anchored
        self.assertLessEqual(frames[0], 40*STEP+cal.REACH_S*FPS)   # the earliest reaches back to the anchored part's start

    def test_every_segment_lists_its_numbers(self):
        for entry, s in zip(self.index['segments'], self.segments):
            self.assertEqual(entry['index'], s.index)
            self.assertEqual(entry['sampled'], sum(1 for r in self.rows if s.proc0 <= r['index'] < s.proc1))
            self.assertEqual(len(entry['keyframes']), len(entry['frames']))
            self.assertEqual(entry['manual'], [])

    def test_a_segment_without_a_fusable_anchor_keeps_its_anchors_unfused(self):
        rows, pairs, truth = match_rows(60)
        pairs = [p for p in pairs if p['b']-p['a'] == STEP and (p['a']//STEP) % 2 == 0]   # one agreeing neighbour each
        index = cal.discover(rows, pairs, segments_for(60), FPS, W, H)
        kfs = cal.keyframes_for(index, 0)
        self.assertTrue(kfs)
        self.assertTrue(all(k['fusedAnchors'] == 1 for k in kfs))
        for k in kfs:
            self.assertLess(registration._disagreement(truth[k['frame']//STEP], np.array(k['H']), W, H), 1e-6)

    def test_a_user_keyframe_wins_within_reach(self):
        s = self.segments[1]
        mid = (s.own0+s.own1)//2
        manual = [{'time': mid/FPS, 'points': [{'name': 'centre-spot', 'x': .5, 'y': .5}]}]
        index = cal.discover(self.rows, self.pairs, self.segments, FPS, W, H, manual=manual)
        entry = index['segments'][1]
        self.assertEqual(entry['manual'], [mid/FPS])
        times = [k['time'] for k in cal.keyframes_for(index, s)]
        self.assertTrue(times)   # the segment keeps its other keyframes
        self.assertTrue(all(abs(t-mid/FPS) > cal.MANUAL_REACH_S for t in times))
        # Segments the user keyframe does not reach are unchanged.
        self.assertEqual([k['frame'] for k in cal.keyframes_for(index, self.segments[3])],
                         [k['frame'] for k in cal.keyframes_for(self.index, self.segments[3])])


class SamplingTests(unittest.TestCase):
    PERIODS = [(250, 71350), (72125, 144790)]   # CAN-ISL's halves (10.0-2854.0 s and 2885.0-5791.6 s) in frames
    CAN_FPS = 25.000080239880965

    def test_samples_stay_in_their_period(self):
        indices = cal.sample(self.PERIODS, self.CAN_FPS)
        self.assertEqual(indices, sorted(set(indices)))
        self.assertTrue(all(any(a <= i < b for a, b in self.PERIODS) for i in indices))
        self.assertEqual(indices[0], 250)
        self.assertIn(72125, indices)
        step = cal.STEP_S*self.CAN_FPS
        for a, b in self.PERIODS:
            run = [i for i in indices if a <= i < b]
            self.assertEqual(len(run), math.ceil((b-a)/step))
            self.assertTrue(all(abs((j-i)-step) <= 1 for i, j in zip(run, run[1:])))

    def test_chunks_compute_every_pair_once_and_never_across_a_period(self):
        indices = cal.sample(self.PERIODS, self.CAN_FPS)
        for periods in (self.PERIODS, None):   # without periods the halftime break is found from the spacing
            chunks = cal.chunks(indices, periods=periods)
            owned, pairs = [], []
            for c in chunks:
                self.assertEqual(sum(1 for a, b in self.PERIODS if all(a <= i < b for i in c)), 1)   # one period per chunk
                self.assertTrue(len(c) <= cal.CHUNK+max(cal.GAPS))
                owned += c[:min(len(c), cal.CHUNK)]
                pairs += [(c[p], c[q]) for p, q in cal.chunk_pairs(c)]
            self.assertEqual(sorted(owned), indices)   # every sampled frame is solved exactly once
            expected = []
            for a, b in self.PERIODS:
                run = [i for i in indices if a <= i < b]
                expected += [(run[k], run[k+g]) for k in range(len(run)) for g in cal.GAPS if k+g < len(run)]
            self.assertEqual(len(pairs), len(set(pairs)))
            self.assertEqual(sorted(pairs), sorted(expected))
            self.assertFalse([p for p in pairs if p[0] < 72125 <= p[1]])


class RegistrationTests(unittest.TestCase):
    def test_a_pan_is_recovered_and_a_burned_in_overlay_ignored(self):
        rng = np.random.default_rng(1)
        base = cv2.GaussianBlur((rng.random((1300, 2300, 3))*255).astype(np.uint8), (0, 0), 2.0)
        a = base[100:100+H, 100:100+W].copy()
        b = base[100:100+H, 140:140+W].copy()   # the camera panned 40 px to the right: the scene moves 40 px left
        for img in (a, b):   # the same logo at the same place in both frames
            cv2.rectangle(img, (1650, 950), (1900, 1060), (255, 255, 255), -1)
            cv2.putText(img, 'LOGO 12:34', (1660, 1020), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 0), 3)
        m = cal.match(cal.features(a), cal.features(b), W, H)
        self.assertIsNotNone(m)
        moved = cv2.perspectiveTransform(np.float32([[[960, 700]]]), m['G']).ravel()
        self.assertAlmostEqual(float(moved[0]), 920.0, delta=1.0)
        self.assertAlmostEqual(float(moved[1]), 700.0, delta=1.0)
        self.assertGreater(m['overlap'], .9)


if __name__ == '__main__':
    unittest.main()
