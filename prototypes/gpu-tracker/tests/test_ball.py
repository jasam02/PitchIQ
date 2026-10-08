import math
import unittest

import cv2
import numpy as np

import broadcast as B
from soccer.ball import BALL_DIAMETER, BallTracker
from soccer.pitch import PitchDetector

FPS = 25
P = B.camera()
P_INV = np.linalg.inv(P)


def px_per_m(x, y):
    return B._px_per_m(P, x, y)


def scale(y_norm):
    """Expected player height (normalized) at image row y."""
    p = P_INV @ np.array([B.W/2, y_norm*B.H, 1.0])
    return 1.8*px_per_m(p[0]/p[2], p[1]/p[2])/B.H


def ball_row(x, y, conf):
    cx, cy = B._project(P, [(x, y)])[0]
    d = max(4.0, BALL_DIAMETER*px_per_m(x, y))
    return [cx-d/2, cy-d/2, cx+d/2, cy+d/2, conf, 0], (cx, cy, d)


def person_box(x, y):
    cx, cy = B._project(P, [(x, y)])[0]
    h = 1.8*px_per_m(x, y)
    return [(cx-h*.2)/B.W, (cy-h)/B.H, h*.4/B.W, h/B.H]


class Scene:
    """Ball rolling along the pitch at 8 m/s (hidden behind a player for 0.5 s), plus white things that
    are not the ball: the centre spot, a piece of debris, the halfway line and a player's white sock."""

    def __init__(self, ball_until=math.inf, spot_conf=.45):
        self.base = B.render(P, extra_white_dots=[(45, 20)])
        self.pitch = PitchDetector().analyze(self.base, 0.0)
        self.tracker = BallTracker(B.W, B.H, FPS)
        self.ball_until, self.spot_conf = ball_until, spot_conf
        self.log = []

    def ball_at(self, t):
        return 30+8*t, 40.0

    def frame(self, t):
        img = self.base.copy()
        rows, truth = [], None
        occluded = 1.2 <= t < 1.7
        bx, by = self.ball_at(t)
        hider = person_box(30+8*1.45, 40.6)
        sock_player = person_box(36, 47)
        people = [(1, hider), (2, sock_player)]
        if t < self.ball_until and not occluded:
            row, (cx, cy, d) = ball_row(bx, by, .5)
            cv2.circle(img, (int(cx), int(cy)), max(2, int(d/2)), (235, 235, 235), -1, cv2.LINE_AA)
            rows.append(row)
            truth = (cx, cy, d)
        rows.append(ball_row(52.5, 34, self.spot_conf)[0])                  # centre spot
        rows.append(ball_row(45, 20, self.spot_conf)[0])                    # debris
        rows.append(ball_row(52.5, 50, .3)[0])                              # on the halfway line
        sx, sy, sw, sh = sock_player
        cx, cy, d = (sx+sw*.4)*B.W, (sy+sh*.85)*B.H, 6.0
        cv2.rectangle(img, (int(cx-3), int(cy-4)), (int(cx+3), int(cy+4)), (240, 240, 240), -1)
        rows.append([cx-d/2, cy-d/2, cx+d/2, cy+d/2, .3, 0])               # white sock
        return img, np.asarray(rows, np.float32), people, truth

    def run(self, seconds):
        out = []
        for i in range(int(seconds*FPS)):
            t = i/FPS
            img, rows, people, truth = self.frame(t)
            ball, cands = self.tracker.step(img, t, rows, np.eye(3), True, False, self.pitch, people, scale,
                                            lambda x, y: (x, y), 0)
            out.append((t, ball, cands, truth))
        return out


class BallTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scene = Scene()
        cls.frames = cls.scene.run(3.0)

    def test_follows_the_moving_ball(self):
        tracked = [(t, b, truth) for t, b, _, truth in self.frames if t >= .5 and truth is not None]
        self.assertTrue(all(b['state'] == 'TRACKED' for _, b, _ in tracked), [(t, b['state']) for t, b, _ in tracked if b['state'] != 'TRACKED'][:5])
        for _, b, (cx, cy, d) in tracked:
            self.assertLess(math.hypot(b['center'][0]*B.W-cx, b['center'][1]*B.H-cy), 1.5*d)
            self.assertGreaterEqual(b['confidence'], .5)

    def test_white_spots_lines_and_socks_are_never_the_ball(self):
        objects = {'centre spot': ball_row(52.5, 34, 0)[1], 'debris': ball_row(45, 20, 0)[1], 'halfway line': ball_row(52.5, 50, 0)[1]}
        reasons = {name: set() for name in objects}
        reasons['sock'] = set()
        for t, _, cands, truth in self.frames:
            for c in cands:
                cx, cy = (c['box'][0]+c['box'][2]/2)*B.W, (c['box'][1]+c['box'][3]/2)*B.H
                if truth is not None and math.hypot(cx-truth[0], cy-truth[1]) < 3:
                    continue
                self.assertNotEqual(c['status'], 'ball', (t, c))
                name = next((n for n, (x, y, _) in objects.items() if math.hypot(cx-x, cy-y) < 3), 'sock')
                reasons[name].add(c.get('reason'))
        self.assertTrue({'STATIONARY', 'FIELD_LINE'} & reasons['centre spot'])  # the centre spot lies on the halfway line
        self.assertIn('STATIONARY', reasons['debris'])
        self.assertIn('FIELD_LINE', reasons['halfway line'])
        self.assertTrue({'PLAYER_PART', 'STATIONARY'} & reasons['sock'])

    def test_occlusion_keeps_the_same_ball(self):
        hidden = [b for t, b, _, _ in self.frames if 1.25 <= t < 1.65]
        self.assertTrue(all(b['state'] == 'MISSING' for b in hidden), [b['state'] for b in hidden])
        after = [b for t, b, _, _ in self.frames if t >= 1.95]
        self.assertTrue(all(b['state'] == 'TRACKED' for b in after))
        tracks = {b['track'] for t, b, _, _ in self.frames if b['state'] != 'UNKNOWN'}
        self.assertEqual(len(tracks), 1, 'one ball track, reconnected after the occlusion')
        self.assertFalse([e for e in self.scene.tracker.events if e['message'].startswith('BALL LOST')])

    def test_ball_becomes_unknown_after_it_is_gone(self):
        scene = Scene(ball_until=1.0)
        frames = scene.run(4.5)
        self.assertEqual(frames[-1][1]['state'], 'UNKNOWN')
        self.assertTrue(any(e['message'].startswith('BALL LOST') for e in scene.tracker.events))
        # Static white spots are never adopted while the ball is gone, even with confident detections.
        self.assertTrue(all(b['state'] == 'UNKNOWN' for t, b, _, _ in frames if t >= 3.6))

    def test_confident_static_spot_is_not_adopted(self):
        scene = Scene(ball_until=0.0, spot_conf=.75)
        frames = scene.run(3.0)
        self.assertTrue(all(b['state'] == 'UNKNOWN' for _, b, _, _ in frames))

    def test_acquisition_is_logged_with_scores(self):
        scene = Scene()
        scene.run(2.5)
        acquired = [e for e in scene.tracker.events if e['message'].startswith('BALL ACQUIRED')]
        self.assertEqual(len(acquired), 1)
        for line in ('Detector confidence', 'Shape score', 'Motion consistency', 'Final confidence'):
            self.assertIn(line, acquired[0]['message'])
        self.assertTrue(any(e['message'].startswith('BALL TRACK UPDATE') for e in scene.tracker.events))


if __name__ == '__main__':
    unittest.main()
