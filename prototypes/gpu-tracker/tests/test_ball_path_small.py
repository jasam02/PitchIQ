"""The offline decoder's small-scene rules (soccer/ball_path.py, runs whose frames mostly carry profile.smallScene): the
still-lock audit, proposals without the acceptance bonus, the drifting stillness of things nobody is near, the review of
unattended still stretches (and the ball placed for a set piece it must keep). Every case is decoded with the gate open
and closed; closed, nothing changes."""
import copy
import json
import unittest

import numpy as np

import ball_synth
import scene  # noqa: F401  (sys.path)
from soccer import ball_path, field
from soccer.ball import BallTracker, Candidate
from test_ball_path import FPS, H, W, candidate, person

SPECK = (1000.0, 300.0)


def frames_of(n, build, small):
    frames = []
    for i in range(n):
        cands, people, first = build(i)
        frames.append({'time': i/FPS, 'ballCandidates': cands, 'people': people, 'profile': {'smallScene': small},
                       'ball': first or {'state': 'UNKNOWN', 'phase': 'SEARCHING', 'confidence': 0.0}})
    return frames


def decode(frames, motions=None):
    motions = motions or [(np.eye(3), True, False)]*len(frames)
    report = ball_path.finalize(frames, motions, W, H, FPS)
    return [f['ball'] for f in frames], report


def at(ball, x, y, tol=4.0):
    return ball['state'] == 'TRACKED' and abs(ball['center'][0]*W-x) <= tol and abs(ball['center'][1]*H-y) <= tol


def locked(x, y, c):
    return {'state': 'TRACKED', 'phase': 'LOCKED', 'confidence': .9, 'track': 1, 'center': [x/W, y/H], 'box': c['box']}


def still_lock_scene(entered_rolling=False, competitor=True, competitor_on_feet=False):
    """The amateur 3 149-151 s case: the ball rolls (0-1 s), hides; the causal pass locks on a still speck (2.2-6 s,
    accepted, nobody near); from 2.6 s the real ball rolls free at 6 m/s, strongly detected, flagged "seen elsewhere".
    entered_rolling: the ball itself rolls to the speck's place and stops there (a ball at rest before a set piece)."""
    def build(i):
        t = i/FPS
        cands, people, first = [], [], None
        if t < 1.0:
            x, y = (SPECK[0]-240+240*t, SPECK[1]) if entered_rolling else (300+8*i, 500)
            c = candidate(x, y, det=.8, status='ball', tracklet=1)
            cands.append(c)
            first = locked(x, y, c)
        if 2.2 <= t < 6.0 or (entered_rolling and 1.0 <= t < 6.0):
            c = candidate(*SPECK, det=.5, d=6, status='ball', tracklet=2 if not entered_rolling else 1)
            cands.append(c)
            first = locked(*SPECK, c)
        if competitor and 2.6 <= t < 5.6:
            x, y = 200+9.6*(i-65), 600          # 6 m/s at 40 px per metre
            cands.append(candidate(x, y, det=.85, status='rejected', reason='OTHER_OBJECT', tracklet=3, otherObject=True))
            if competitor_on_feet:
                people.append(person(9, x-2, y+8))
        return cands, people, first
    return build


class StillLockAuditTests(unittest.TestCase):
    def test_the_decoder_stops_deferring_to_a_lock_on_a_still_thing(self):
        balls, report = decode(frames_of(150, still_lock_scene(), small=True))
        real = [k for k in range(66, 140) if balls[k]['state'] == 'TRACKED' and abs(balls[k]['center'][1]*H-600) < 4]
        self.assertGreaterEqual(len(real), 60, [balls[k]['state'] for k in range(66, 140)])
        self.assertFalse([k for k in range(55, 150) if at(balls[k], *SPECK)])
        self.assertEqual(len(report['audits']), 1)
        audit = report['audits'][0]
        self.assertEqual((audit['tracklet'], [r['tracklet'] for r in audit['competitors']]), (2, [3]))
        self.assertEqual(report['final']['lockAudits'], 1)
        self.assertTrue([e for e in report['events'] if e['message'].startswith('BALL LOCK AUDIT')])

    def test_with_the_gate_closed_the_lock_on_the_still_thing_is_followed_as_before(self):
        balls, report = decode(frames_of(150, still_lock_scene(), small=False))
        self.assertGreater(sum(1 for k in range(66, 140) if at(balls[k], *SPECK)), 60)
        self.assertEqual(report['audits'], [])

    def test_a_ball_that_rolled_to_its_resting_place_is_not_audited(self):
        balls, report = decode(frames_of(150, still_lock_scene(entered_rolling=True), small=True))
        self.assertEqual(report['audits'], [])
        self.assertGreater(sum(1 for k in range(30, 140) if at(balls[k], *SPECK)), 90)

    def test_without_a_free_moving_competitor_nothing_is_audited(self):
        for kwargs in ({'competitor': False}, {'competitor_on_feet': True}):
            _, report = decode(frames_of(150, still_lock_scene(**kwargs), small=True))
            self.assertEqual(report['audits'], [], kwargs)


class SmallSceneNodeTests(unittest.TestCase):
    def test_a_proposal_carries_no_acceptance_bonus(self):
        c = candidate(300, 300, det=.12, status='ball', tracklet=1)
        proposal = dict(c, source='proposal')
        self.assertAlmostEqual(ball_path.quality(c), min(1.0, ball_path.quality(proposal)*ball_path.CAUSAL_BONUS))
        self.assertAlmostEqual(ball_path.quality(dict(c, source='detector')), ball_path.quality(c))

    def test_a_still_thing_nobody_is_near_is_a_marking_although_the_camera_compensation_drifts(self):
        # A white thing seen for 4 s, drifting .25 m/s in the compensated image (amateur 3's pole camera); nobody near.
        def build(i):
            return [candidate(700+.25*40*i/FPS, 300, det=.5, tracklet=5)], [], None
        for small, marked in ((True, True), (False, False)):
            frames = frames_of(100, build, small)
            layers = [ball_path.nodes_of(i, f, W, H) for i, f in enumerate(frames)]
            times = [f['time'] for f in frames]
            T, _, _ = ball_path._chain([(np.eye(3), True, False)]*len(frames))
            ball_path._velocities(layers, times, T)
            ball_path._standing_markings(layers, times, T, frames if small else None)
            self.assertEqual(all(m.marking for l in layers for m in l), marked)

    def test_an_unattended_still_stretch_never_the_causal_ball_is_cut(self):
        # The ball rolls (0-1.2 s) and is lost; a speck the causal pass rejected only as off its trajectory stands
        # still for 3 s with nobody near, drifting .25 m/s in the compensated image and seen as two tracklets of 1.5 s
        # (too short to be a standing marking each; amateur 3 172-181 s): a trajectory starting there is not the ball.
        def build(i):
            t = i/FPS
            if t < 1.2:
                c = candidate(300+8*i, 500, det=.8, status='ball', tracklet=1)
                return [c], [], locked(300+8*i, 500, c)
            if 2.0 <= t < 5.0:
                x = SPECK[0]+.25*40*(t-2.0)
                return [candidate(x, SPECK[1], det=.5, status='rejected', reason='TRAJECTORY', tracklet=4 if t < 3.5 else 5)], [], None
            return [], [], None
        for small in (True, False):
            balls, _ = decode(frames_of(140, build, small))
            shown = sum(1 for b in balls[50:125] if b['state'] == 'TRACKED' and abs(b['center'][1]*H-SPECK[1]) < 4)
            self.assertEqual(shown, 0 if small else 75, small)

    def test_a_ball_placed_for_a_goal_kick_is_kept(self):
        # The ball rolls out (accepted, 0-1 s) and is fetched unseen (1-3 s); it is placed and stands still for 4 s, clearly
        # detected, the keeper stepped back (nobody within reach); the causal pass never takes it (it acquires a ball only
        # once it moves) until it is kicked at 7 s (taken from 7.3 s). Unattended and never accepted like the 172-181 s
        # speck, but kicked away: it stays, with the causal pass's reasons or without any.
        rest = (900.0, 450.0)

        def build(i, kick_taken=True, reason='UNCONFIRMED'):
            t = i/FPS
            if t < 1.0:
                c = candidate(300+8*i, 500, det=.8, status='ball', tracklet=1)
                return [c], [], locked(300+8*i, 500, c)
            if 3.0 <= t < 7.0:
                return [candidate(*rest, det=.8, status='rejected' if reason else 'candidate', reason=reason, tracklet=2)], [], None
            if 7.0 <= t < 8.0:
                x = rest[0]+40*(i-7*FPS)   # 25 m/s
                taken = kick_taken and t >= 7.3
                c = candidate(x, rest[1], det=.8, status='ball' if taken else 'rejected', reason=None if taken else 'UNCONFIRMED', tracklet=2)
                return [c], [], locked(x, rest[1], c) if taken else None
            return [], [], None
        resting = range(3*FPS, 7*FPS)
        for reason in ('UNCONFIRMED', None):
            for small in (True, False):
                balls, report = decode(frames_of(8*FPS, lambda i: build(i, reason=reason), small))
                self.assertEqual(sum(1 for k in resting if at(balls[k], *rest)), len(resting), (reason, small))
                self.assertEqual(report['review'], [], (reason, small))
        # The exit is a causal acceptance: a still thing the path leaves into sightings the causal pass never took was not
        # shown to be the ball (gate open), as before with the gate closed.
        for small, shown in ((True, 0), (False, len(resting))):
            balls, _ = decode(frames_of(8*FPS, lambda i: build(i, kick_taken=False), small))
            self.assertEqual(sum(1 for k in resting if at(balls[k], *rest)), shown, small)

    def test_the_rules_run_on_a_run_whose_frames_are_mostly_small(self):
        # The still-lock scene followed by 250 empty frames: small-scene on its 150 frames only (37.5% of the run), the run
        # decodes as any other; on 250 of 400 frames (62.5%) the audit runs.
        scene_of = still_lock_scene()
        for small_frames, audited in ((150, False), (250, True)):
            frames = frames_of(400, lambda i: scene_of(i) if i < 150 else ([], [], None), True)
            for i, f in enumerate(frames):
                f['profile'] = {'smallScene': i < small_frames}
            balls, report = decode(frames)
            self.assertEqual(bool(report['audits']), audited, small_frames)
            self.assertEqual(sum(1 for k in range(66, 140) if at(balls[k], *SPECK)) > 60, not audited, small_frames)

    def test_pitch_distances_use_the_field_size(self):
        node = lambda p: ball_path.Node(0, 0.0, 0.0, 8.0, .5, None, None, 0.0, 0.0, 0.0, 0.0, 40.0, None, {}, pitch=p)
        T = [np.eye(3)]
        self.assertEqual(ball_path._metres_apart(node((0.0, 0.0)), 0, node((1.0, 0.0)), 0, T), field.DEFAULT_LENGTH)
        self.assertEqual(ball_path._metres_apart(node((0.0, 0.0)), 0, node((0.0, 1.0)), 0, T), field.DEFAULT_WIDTH)
        spot = ball_path.PAINTED_SPOTS[0]
        self.assertTrue(ball_path._on_painted_spot((spot[0]+.99*ball_path.SPOT_RADIUS/field.DEFAULT_LENGTH, spot[1])))
        self.assertFalse(ball_path._on_painted_spot((spot[0]+1.01*ball_path.SPOT_RADIUS/field.DEFAULT_LENGTH, spot[1])))
        tracker = BallTracker(W, H, FPS)
        a = Candidate(0, [], .5, 0.0, 0.0, 8.0, 40.0, (0.0, 0.0), (.5, .2), None)
        b = Candidate(0, [], .5, 0.0, 0.0, 8.0, 40.0, (0.0, 0.0), (.5, .7), None)
        self.assertAlmostEqual(tracker._metres(a, b), .5*field.DEFAULT_WIDTH)

    def test_with_the_gate_closed_a_result_decodes_as_it_did_without_a_profile(self):
        frames, motions, to_pitch = ball_synth.synthetic(4, 300, 25.0, registered=True)
        outputs = []
        for profile in (None, {'smallScene': False}):
            copied = copy.deepcopy(frames)
            for f in copied:
                if profile is not None:
                    f['profile'] = profile
            report = ball_path.finalize(copied, motions, ball_synth.W, ball_synth.H, 25.0, to_pitch)
            outputs.append((json.dumps([f['ball'] for f in copied], sort_keys=True), json.dumps(report, sort_keys=True, default=str)))
        self.assertEqual(outputs[0], outputs[1])


if __name__ == '__main__':
    unittest.main()
