"""The offline ball trajectory: candidates plus a NOT VISIBLE node per frame, decoded over the whole
sequence. Hand-built frames (no rendering): what must the decoder keep, bridge, and leave out?"""
import math
import unittest

import numpy as np

import scene  # noqa: F401  (sys.path)
from soccer import ball_path

W, H, FPS = 1280, 720, 25
PXM = 40.0   # pixels per metre at the depth of these frames (people 72 px tall)


def person(track, x, y, h=72):
    """A player box (normalized) whose feet are at (x, y) pixels."""
    w = h*.38
    return {'track': track, 'display': f'P{track}', 'box': [(x-w/2)/W, (y-h)/H, w/W, h/H]}


def candidate(x, y, det=.6, d=8, status='candidate', reason=None, tracklet=None, **scores):
    s = {'size': .9, 'shape': .8, 'isolation': .9, 'field': 1.0, 'line': 0.0, 'attach': 0.0, 'static': 0.0, 'otherObject': False}
    s.update(scores)
    c = {'box': [(x-d/2)/W, (y-d/2)/H, d/W, d/H], 'det': det, 'score': det, 'status': status, 'scores': s, 'pxm': PXM, 'tracklet': tracklet}
    if reason:
        c['reason'] = reason
    return c


def frame(i, ball=None, extra=(), people=(), first=None):
    cands = ([candidate(*ball, status='ball', tracklet=1)] if ball else [])+list(extra)
    out = first or ({'state': 'TRACKED', 'phase': 'LOCKED', 'confidence': .9, 'track': 1, 'center': [ball[0]/W, ball[1]/H], 'box': cands[0]['box']} if ball
                    else {'state': 'UNKNOWN', 'phase': 'SEARCHING', 'confidence': 0.0})
    return {'time': i/FPS, 'ball': out, 'ballCandidates': cands, 'people': list(people)}


def run(frames):
    motions = [(np.eye(3), True, False)]*len(frames)
    report = ball_path.finalize(frames, motions, W, H, FPS)
    return [f['ball'] for f in frames], report


class PathTests(unittest.TestCase):
    def test_a_rolling_ball_with_a_detection_gap_is_one_bridged_trajectory(self):
        frames = []
        for i in range(40):
            x, y = 300+8*i, 400+2*i                       # 8 px/frame = 5 m/s
            frames.append(frame(i, None if 15 <= i < 21 else (x, y)))
        balls, report = run(frames)
        self.assertTrue(all(b['state'] == 'TRACKED' for k, b in enumerate(balls) if not 15 <= k < 21))
        bridged = balls[15:21]
        self.assertTrue(all(b['state'] == 'MISSING' and b['observed'] is False for b in bridged), [b['state'] for b in bridged])
        self.assertAlmostEqual(balls[18]['center'][0]*W, 300+8*18, delta=3)   # interpolated along the trajectory
        self.assertEqual(report['final']['runs'], 1)
        self.assertEqual(report['final']['teleports'], 0)
        self.assertTrue(all(b['track'] == 1 for b in balls))

    def test_a_boot_seen_while_the_ball_is_hidden_is_not_the_ball(self):
        # The ball rolls to player 1 and vanishes at their feet for 0.6 s; meanwhile player 2's white boot,
        # 6 m away, is a confident candidate for 10 frames (and the first pass took it). Then the ball
        # reappears at player 1's feet and rolls on.
        p1, p2 = (700, 500), (940, 500)
        frames = []
        for i in range(60):
            people = [person(1, *p1), person(2, p2[0]-i, p2[1])]
            ball = (520+12*i, 500) if i < 15 else ((700+5*(i-30), 500) if i >= 30 else None)
            extra = []
            first = None
            if 18 <= i < 28:
                boot = candidate(p2[0]-i-6, 496, det=.7, status='ball', tracklet=7)
                extra.append(boot)
                first = {'state': 'TRACKED', 'phase': 'LOCKED', 'confidence': .64, 'track': 1, 'center': [(p2[0]-i-6)/W, 496/H], 'box': boot['box']}
            frames.append(frame(i, ball, extra, people, first))
        balls, report = run(frames)
        stolen = [k for k in range(18, 28) if balls[k]['state'] == 'TRACKED' and abs(balls[k]['center'][0]*W-(p2[0]-k-6)) < 4]
        self.assertEqual(stolen, [])
        self.assertTrue(all(b['state'] in ('MISSING', 'UNKNOWN') for b in balls[15:30]), [b['state'] for b in balls[15:30]])
        self.assertTrue(all(b['state'] == 'TRACKED' for b in balls[30:]))
        self.assertEqual(report['final']['teleports'], 0)
        self.assertEqual(len(report['dropped']), 1)
        self.assertEqual(report['dropped'][0]['player'], 2)
        self.assertIn('m/s', report['dropped'][0]['why'])
        self.assertTrue(report['events'] and report['events'][0]['message'].startswith('BALL PATH'))

    def test_a_visible_pass_between_two_players_feet_is_kept(self):
        # Player 1 passes to player 2: the ball is seen leaving the feet, flying, arriving.
        p1, p2 = (400, 500), (900, 500)
        frames = []
        for i in range(50):
            people = [person(1, *p1), person(2, *p2)]
            if i < 10:
                ball = (p1[0]+8, 496)
            elif i < 35:
                ball = (p1[0]+8+(p2[0]-p1[0]-16)*(i-10)/25, 480)
            else:
                ball = (p2[0]-8, 496)
            frames.append(frame(i, ball, people=people))
        balls, report = run(frames)
        self.assertTrue(all(b['state'] == 'TRACKED' for b in balls))
        self.assertEqual(report['final']['footSwitches'], 0)
        self.assertEqual(report['dropped'], [])

    def test_a_teleport_from_one_foot_to_another_is_cut_not_linked(self):
        # The ball at player 1's feet (20 frames); then, with nothing seen travelling, a confident candidate
        # at player 2's feet 8 m away for 5 frames, then the ball again at player 1's feet.
        p1, p2 = (400, 500), (720, 500)
        frames = []
        for i in range(50):
            people = [person(1, *p1), person(2, *p2)]
            if 20 <= i < 25:
                frames.append(frame(i, (p2[0]-6, 496), people=people))
            elif 25 <= i < 28:
                frames.append(frame(i, None, people=people))
            else:
                frames.append(frame(i, (p1[0]+8+(i % 3), 496), people=people))
        balls, report = run(frames)
        self.assertTrue(all(b['state'] != 'TRACKED' for b in balls[20:25]), [b['state'] for b in balls[20:28]])
        self.assertEqual(report['final']['footSwitches'], 0)
        self.assertGreaterEqual(report['firstPass']['footSwitches'], 1)

    def test_a_resting_ball_is_kept_and_an_isolated_sighting_is_not(self):
        frames = [frame(i, (640, 420)) for i in range(50)]           # a ball at rest before a free kick
        frames[30]['ballCandidates'].append(candidate(200, 300, det=.9, tracklet=9))   # one bright blip elsewhere
        balls, report = run(frames)
        self.assertTrue(all(b['state'] == 'TRACKED' and abs(b['center'][0]*W-640) < 2 for b in balls))
        lone = [frame(i, None) for i in range(30)]
        lone[12]['ballCandidates'].append(candidate(500, 400, det=.95, tracklet=3))
        balls, report = run(lone)
        self.assertTrue(all(b['state'] == 'UNKNOWN' for b in balls))

    def test_camera_motion_is_compensated_before_judging_a_link(self):
        # A ball at rest while the camera pans 30 px per frame: in raw pixels it "moves" 19 m/s.
        frames = [frame(i, (640-30*i, 420)) for i in range(20)]
        motions = [(np.array([[1, 0, -30], [0, 1, 0], [0, 0, 1]], float), True, False)]*20
        report = ball_path.finalize(frames, motions, W, H, FPS)
        self.assertTrue(all(f['ball']['state'] == 'TRACKED' for f in frames))
        self.assertLess(max(f['ball']['speed'] or 0 for f in frames), 1.0)

    def test_pitch_coordinates_follow_the_final_path(self):
        frames = [frame(i, (300+8*i, 400)) for i in range(20)]
        motions = [(np.eye(3), True, False)]*20
        ball_path.finalize(frames, motions, W, H, FPS, to_pitch=lambda i, p: (p[0]/W, p[1]/H))
        self.assertAlmostEqual(frames[5]['ball']['pitch'][0], (300+40)/W, places=3)
        self.assertEqual(frames[5]['ball']['firstPass']['state'], 'TRACKED')


if __name__ == '__main__':
    unittest.main()
