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


class StationaryAndAnchorTests(unittest.TestCase):
    """What the decoder may invent: never a flight that nobody saw, never a trajectory on a white thing that
    never moved and that nothing led to; what it may assume: a ball seen again where it was stayed there."""

    def test_a_ball_hidden_at_a_player_is_not_sent_to_a_still_white_thing_elsewhere(self):
        # The broadcast clip: the ball is at player 1's feet until frame 50, hidden for 2.4 s, seen again there
        # (then kicked away). Meanwhile a ball boy's ball, 14 m down the touchline, is detected on every frame
        # of the gap (weak, never moving); the first pass rejected it (JUMP). The final path must not go there
        # and back: the gap is bridged where the ball was, as one stationary trajectory.
        frames = []
        ax, ay = 300, 400
        p1 = person(1, ax+4, ay+6)
        for i in range(160):
            people = [p1]
            if i <= 50:
                hidden = {'state': 'MISSING', 'phase': 'OCCLUDED', 'confidence': .7, 'track': 1, 'center': [ax/W, ay/H], 'nearTrack': 1}
                frames.append(frame(i, (ax, ay), people=people))
            elif i <= 110:
                extra = [candidate(ax+560, ay+120, det=.35, status='rejected', reason='JUMP', tracklet=100)] if 60 <= i <= 108 else []
                frames.append(frame(i, None, extra=extra, people=people, first={'state': 'MISSING', 'phase': 'OCCLUDED', 'confidence': .5, 'track': 1, 'center': [ax/W, ay/H], 'nearTrack': 1}))
            else:
                k = i-111
                frames.append(frame(i, (ax+3+10*k, ay-4-3*k), people=people))   # kicked: 10 px/frame
        balls, report = run(frames)
        far = [k for k, b in enumerate(balls) if b['state'] != 'UNKNOWN' and math.hypot(b['center'][0]*W-(ax+560), b['center'][1]*H-(ay+120)) < 30]
        self.assertEqual(far, [], 'the final path visited the ball boy\'s ball')
        self.assertEqual(report['final']['runs'], 1, report['final'])
        self.assertEqual(report['final']['staticRunsDropped'], 1, report['review'])
        self.assertEqual(report['final']['longUnseenLinks'], 0)
        gap = balls[51:111]
        self.assertTrue(all(b['state'] == 'MISSING' and b['origin'] == 'INTERPOLATION' and b['motion'] == 'OCCLUDED' for b in gap), [(k+51, b['state']) for k, b in enumerate(gap) if b['state'] != 'MISSING'][:5])
        self.assertTrue(all(math.hypot(b['center'][0]*W-ax, b['center'][1]*H-ay) < 12 for b in gap), 'the bridged ball stays where it was')
        self.assertTrue(all(b.get('nearTrack') == 1 for b in gap))
        review = [e for e in report['events'] if e['message'].startswith('BALL TRAJECTORY REVIEW')]
        self.assertEqual(len(review), 1, [e['message'][:60] for e in report['events']])
        self.assertIn('REJECTED_STATIC_UNANCHORED', review[0]['message'])
        self.assertIn('Camera motion compensated: YES', review[0]['message'].replace('camera motion compensated: YES', 'Camera motion compensated: YES'))

    def test_a_ball_seen_again_where_it_was_after_a_long_gap_is_one_stationary_trajectory(self):
        # Frame 100 at A, 101-170 undetected (2.8 s, longer than a flight may bridge), 171 at A again, then rolling.
        frames = []
        for i in range(220):
            if i <= 30:
                frames.append(frame(i, (500, 300)))
            elif i <= 100:
                frames.append(frame(i, None))
            elif i <= 140:
                frames.append(frame(i, (502, 301)))
            else:
                frames.append(frame(i, (502+6*(i-140), 301+(i-140))))
        balls, report = run(frames)
        self.assertEqual(report['final']['runs'], 1, report['final'])
        self.assertEqual(report['final']['stationaryLinks'], 1)
        gap = balls[31:101]
        self.assertTrue(all(b['state'] == 'MISSING' and b['origin'] == 'INTERPOLATION' for b in gap), [b['state'] for b in gap][:8])
        self.assertTrue(all(abs(b['center'][0]*W-501) < 4 for b in gap), 'the hidden ball is held where it was seen on both sides')
        self.assertTrue(all(b['motion'] == 'STATIONARY' for b in balls[101:141]), [b['motion'] for b in balls[101:141]][:6])
        self.assertTrue(all(b['motion'] in ('ROLLING', 'FAST') for b in balls[150:220]), [b['motion'] for b in balls[150:220]][:6])
        self.assertTrue(all(b['origin'] == 'OBSERVATION' for b in balls[101:141]))

    def test_a_long_unseen_flight_is_not_invented_between_two_different_things(self):
        # The ball at A until frame 30; nothing for a second; from frame 56 a white object 14 m away that moves
        # a little (a ball being carried along the touchline). Bridging them would be a 13 m/s flight that nobody
        # saw: the two stay apart with NOT VISIBLE between them.
        frames = []
        for i in range(120):
            if i <= 30:
                frames.append(frame(i, (300, 400)))
            elif i < 56:
                frames.append(frame(i, None))
            else:
                frames.append(frame(i, None, extra=[candidate(860, 520+1.2*(i-56), det=.7, tracklet=7)]))
        balls, report = run(frames)
        self.assertEqual(report['final']['runs'], 2, report['final'])
        self.assertTrue(all(b['state'] == 'UNKNOWN' for b in balls[31:56]), [b['state'] for b in balls[31:56]][:8])
        self.assertEqual(report['final']['longUnseenLinks'], 0)

    def test_a_fast_pass_missed_for_a_few_frames_is_still_bridged(self):
        # 15 m/s (24 px per frame here), undetected for 8 frames in flight: one trajectory, bridged on its line.
        frames = []
        for i in range(60):
            x, y = 100+24*i, 400-2*i
            frames.append(frame(i, None if 25 <= i < 33 else (x, y)))
        balls, report = run(frames)
        self.assertEqual(report['final']['runs'], 1, report['final'])
        self.assertTrue(all(b['state'] == 'MISSING' for b in balls[25:33]), [b['state'] for b in balls[25:33]])
        self.assertAlmostEqual(balls[29]['center'][0]*W, 100+24*29, delta=4)
        self.assertTrue(all(b['motion'] == 'FAST' for b in balls[5:25]), [b['motion'] for b in balls[5:25]][:6])

    def test_a_rolling_ball_that_stops_and_rolls_again_is_one_trajectory_with_its_motion_states(self):
        frames = []
        for i in range(120):
            if i < 40:
                frames.append(frame(i, (200+8*i, 400)))
            elif i < 80:
                frames.append(frame(i, (200+8*40, 400)))
            else:
                frames.append(frame(i, (200+8*40+5*(i-80), 400-2*(i-80))))
        balls, report = run(frames)
        self.assertEqual(report['final']['runs'], 1)
        self.assertEqual(report['final']['staticRunsDropped'], 0)
        self.assertTrue(all(b['state'] == 'TRACKED' for b in balls))
        self.assertTrue(all(b['motion'] == 'ROLLING' for b in balls[2:40]), [b['motion'] for b in balls[2:40]][:5])
        self.assertTrue(all(b['motion'] == 'STATIONARY' for b in balls[42:80]), [b['motion'] for b in balls[42:80]][:5])
        self.assertTrue(all(b['motion'] == 'ROLLING' for b in balls[82:120]), [b['motion'] for b in balls[82:120]][:5])

    def test_a_still_white_thing_nothing_led_to_is_dropped_however_confident_and_a_weak_sighting_is_labelled(self):
        # A confident white thing that never moves; from frame 10 the causal pass tracks the ball confidently elsewhere
        # and rules the thing out (OTHER_OBJECT: there is one ball); the ball is seen here from frame 30. Nothing led to
        # the thing, it never moved and the ball was known to be elsewhere: dropped, those frames NOT VISIBLE.
        frames = []
        for i in range(80):
            extra = [candidate(700, 300, det=.9, tracklet=5, **({'status': 'rejected', 'reason': 'OTHER_OBJECT'} if i >= 10 else {}))] if i < 50 else []
            ball = None if i < 30 else (100+6*(i-30), 500)
            frames.append(frame(i, ball, extra=extra))
        # One weak sighting inside the rolling ball's trajectory.
        frames[40]['ballCandidates'][0]['det'] = .12
        balls, report = run(frames)
        self.assertTrue(all(b['state'] == 'UNKNOWN' for b in balls[:30]), [b['state'] for b in balls[:30]][:5])
        self.assertEqual(report['final']['staticRunsDropped'], 1)
        self.assertEqual(balls[40]['origin'], 'WEAK_DETECTION')
        self.assertEqual(balls[41]['origin'], 'OBSERVATION')
        self.assertTrue(all(b['state'] == 'TRACKED' for b in balls[30:]))


class MarkingAndTeleportTests(unittest.TestCase):
    """The decoder knows the painted spots and things standing still for seconds, never jumps across the pitch to
    start a trajectory, and the review never chops a rolling ball."""

    SPOT_PX = (1100, 300)   # where the right penalty spot lands in these frames
    BALL_PX = (300, 420)

    def to_pitch(self, i, point):
        """A linear map that puts SPOT_PX on the right penalty spot and keeps BALL_PX far from every spot."""
        x, y = point
        return ((x-self.SPOT_PX[0])/PXM/105+94/105, (y-self.SPOT_PX[1])/PXM/68+.5)

    def with_registration(self, frames, confidence=.9):
        for f in frames:
            f['pitchConfidence'] = confidence
        motions = [(np.eye(3), True, False)]*len(frames)
        report = ball_path.finalize(frames, motions, W, H, FPS, to_pitch=self.to_pitch)
        return [f['ball'] for f in frames], report

    def test_a_rolling_ball_the_causal_pass_ruled_out_is_not_chopped_by_the_review(self):
        # The broadcast clip: the causal pass was locked on a boot and rejected the real ball as JUMP on every frame.
        # A ball rolling at 5 m/s passes through half a metre in a few frames; that is not a still stretch.
        frames = []
        for i in range(60):
            x, y = 200+8*i, 400
            frames.append(frame(i, None, extra=[candidate(x, y, det=.85, status='rejected', reason='JUMP', tracklet=3)]))
        balls, report = run(frames)
        self.assertTrue(all(b['state'] == 'TRACKED' for b in balls), [b['state'] for b in balls][:10])
        self.assertEqual((report['final']['runs'], report['final']['staticRunsDropped']), (1, 0), report['final'])

    def test_a_white_spot_on_the_penalty_mark_is_not_the_ball(self):
        # The ball is at A, hidden for a second, seen at A again. Meanwhile a confident white thing stands on the right
        # penalty spot (ruled out only as TRAJECTORY by the causal pass). The final path never goes there.
        frames = []
        ax, ay = self.BALL_PX
        for i in range(90):
            ball = (ax+1*(i % 3), ay) if i < 30 or i >= 60 else None
            extra = [candidate(self.SPOT_PX[0]+(i % 2), self.SPOT_PX[1], det=.85, status='rejected', reason='TRAJECTORY', tracklet=9)] if 20 <= i < 80 else []
            first = None if ball else {'state': 'MISSING', 'phase': 'OCCLUDED', 'confidence': .5, 'track': 1, 'center': [ax/W, ay/H]}
            frames.append(frame(i, ball, extra=extra, first=first))
        balls, report = self.with_registration(frames)
        visited = [k for k, b in enumerate(balls) if b['state'] != 'UNKNOWN' and math.hypot(b['center'][0]*W-self.SPOT_PX[0], b['center'][1]*H-self.SPOT_PX[1]) < 30]
        self.assertEqual(visited, [], 'the final path visited the penalty spot')
        self.assertGreater(report['final']['markingNodes'], 0)
        self.assertTrue(all(b['state'] == 'TRACKED' for b in balls[:30]+balls[60:]))
        self.assertTrue(all(b['state'] in ('MISSING', 'UNKNOWN') for b in balls[30:60]))

    def test_a_real_ball_placed_on_the_penalty_spot_is_kept(self):
        # The ball rolls to the spot, rests there for two seconds (the causal pass tracking it) and is kicked away.
        frames = []
        sx, sy = self.SPOT_PX
        for i in range(120):
            if i < 25:
                frames.append(frame(i, (sx-200+8*i, sy)))
            elif i < 75:
                frames.append(frame(i, (sx, sy)))
            else:
                frames.append(frame(i, (sx+10*(i-75), sy-2*(i-75))))
        balls, report = self.with_registration(frames)
        self.assertTrue(all(b['state'] == 'TRACKED' for b in balls), [b['state'] for b in balls][20:80:5])
        self.assertEqual(report['final']['runs'], 1)
        self.assertEqual(report['final']['markingNodes'], 0)

    def test_a_thing_standing_still_for_seconds_the_causal_pass_never_took_is_a_marking(self):
        # No registration: a tracklet that never moves for 2.5 s (debris, the centre spot) loses its ball-likeness; a
        # trajectory cannot start on it when the ball is elsewhere a moment before and after.
        frames = []
        for i in range(100):
            ball = (200+6*i, 500) if i < 30 or i >= 70 else None
            extra = [candidate(900, 250, det=.9, tracklet=5)] if 10 <= i < 90 else []
            frames.append(frame(i, ball, extra=extra))
        balls, report = run(frames)
        visited = [k for k, b in enumerate(balls) if b['state'] != 'UNKNOWN' and math.hypot(b['center'][0]*W-900, b['center'][1]*H-250) < 30]
        self.assertEqual(visited, [])
        self.assertGreater(report['final']['markingNodes'], 0)

    def test_a_trajectory_does_not_start_across_the_pitch_a_frame_after_the_last_sighting(self):
        # The ball at A until frame 40; from frame 42 a bright moving thing 30 m away for 15 frames; the ball at A
        # again from frame 60. Starting a trajectory 30 m away a frame later is the jump the links forbid: left unknown.
        frames = []
        for i in range(100):
            ball = (300+(i % 2), 420) if i <= 40 or i >= 60 else None
            extra = [candidate(300+1200*0+1180, 460+3*(i-42), det=.9, tracklet=7)] if 42 <= i < 57 else []
            frames.append(frame(i, ball, extra=extra))
        balls, report = run(frames)
        far = [k for k, b in enumerate(balls) if b['state'] != 'UNKNOWN' and b['center'][0]*W > 1000]
        self.assertEqual(far, [], 'the final path jumped across the pitch')
        self.assertTrue(all(b['state'] == 'TRACKED' for b in balls[:41]+balls[60:]))

