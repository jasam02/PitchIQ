"""The offline decoder's fixed places (soccer/ball_path.py _fixed_places, small-scene runs): a still thing seen at one
place for seconds before and after the causal pass took the ball somewhere else is a fixed object, the causal lock on it
included (amateur 3, 118.2-132.8 s: the tip of the penalty arc); a ball kicked from a mark, a ball rolling over one and
a mark someone stands at are left alone. Every case is decoded with the gate open and closed; closed, nothing changes."""
import copy
import json
import unittest

import numpy as np

import scene  # noqa: F401  (sys.path)
from soccer import ball_path
from test_ball_path import FPS, H, W, candidate, person
from test_ball_path_small import at, decode, frames_of, locked

MARK = (640.0, 300.0)


def arc_scene(watched=False):
    """The causal pass locks on a mark from the first frame (0-2.4 s) while the ball is out of sight; the ball rolls far
    away (2.5-3.5 s, rejected as off the lock's trajectory, then taken 3.6-4 s); the mark is seen again 3.8-7 s as two
    tracklets the causal pass rejects; the ball is taken again far away 7-8 s. watched: a player stands at the mark."""
    def build(i):
        t = i/FPS
        cands, people, first = [], [], None
        if t < 2.4:
            c = candidate(*MARK, det=.5, d=6, status='ball', tracklet=1)
            cands.append(c)
            first = locked(*MARK, c)
        if 2.5 <= t < 3.6:
            cands.append(candidate(100+240*(t-2.4), 600, det=.7, status='rejected', reason='TRAJECTORY', tracklet=2))
        if 3.6 <= t < 4.0 or 7.0 <= t < 8.0:
            x = 400+240*(t-3.6) if t < 4.0 else 300+240*(t-7.0)
            c = candidate(x, 600, det=.8, status='ball', tracklet=3 if t < 4.0 else 6)
            cands.append(c)
            first = locked(x, 600, c)
        if 3.8 <= t < 7.0:
            cands.append(candidate(*MARK, det=.5, d=6, status='rejected', reason='TRAJECTORY' if t < 5.4 else 'STATIC',
                                   tracklet=4 if t < 5.4 else 5))
        if watched:
            people.append(person(8, MARK[0]+10, MARK[1]+6))
        return cands, people, first
    return build


def kickoff_scene():
    """A ball at rest on a mark from the first frame (taken, 0-2.4 s), kicked at 10 m/s along the row (taken, 2.4-3.4 s);
    the mark alone is seen 4-7 s (rejected as static)."""
    def build(i):
        t = i/FPS
        cands, first = [], None
        if t < 3.4:
            x = MARK[0] if t < 2.4 else MARK[0]+16*(i-59)
            c = candidate(x, MARK[1], det=.8, status='ball', tracklet=1)
            cands.append(c)
            first = locked(x, MARK[1], c)
        if 4.0 <= t < 7.0:
            cands.append(candidate(*MARK, det=.5, d=6, status='rejected', reason='STATIC', tracklet=2))
        return cands, [], first
    return build


class FixedPlaceTests(unittest.TestCase):
    def test_a_mark_locked_from_the_start_and_seen_again_after_the_ball_was_elsewhere_is_not_the_ball(self):
        balls, report = decode(frames_of(200, arc_scene(), small=True))
        self.assertFalse([k for k in range(200) if at(balls[k], *MARK)])
        self.assertTrue(all(balls[k]['state'] == 'TRACKED' and abs(balls[k]['center'][1]*H-600) < 4 for k in list(range(91, 99))+list(range(176, 199))))
        self.assertEqual(len(report['fixed']), 1)
        place = report['fixed'][0]
        self.assertEqual((place['tracklets'], place['withdrawn'], place['keptStays']), ([1, 4, 5], 60, 0))
        self.assertEqual(report['final']['fixedPlaces'], 1)
        self.assertTrue([e for e in report['events'] if e['message'].startswith('BALL FIXED OBJECT')])

    def test_with_the_gate_closed_the_lock_on_the_mark_is_followed_as_before(self):
        balls, report = decode(frames_of(200, arc_scene(), small=False))
        self.assertGreater(sum(1 for k in range(60) if at(balls[k], *MARK)), 50)
        self.assertEqual((report['fixed'], report['final']['fixedPlaces']), ([], 0))

    def test_a_mark_a_player_stands_at_is_not_a_fixed_place(self):
        _, report = decode(frames_of(200, arc_scene(watched=True), small=True))
        self.assertEqual(report['fixed'], [])

    def test_a_ball_kicked_from_a_mark_is_kept_and_the_mark_seen_later_is_not(self):
        balls, report = decode(frames_of(200, kickoff_scene(), small=True))
        self.assertEqual(sum(1 for k in range(60) if at(balls[k], *MARK)), 60)
        self.assertFalse([k for k in range(100, 175) if at(balls[k], *MARK)])
        self.assertEqual([(p['keptStays'], p['withdrawn']) for p in report['fixed']], [(1, 0)])

    def test_a_ball_rolling_over_a_fixed_mark_is_kept(self):
        # The arc scene, and at 5.0-5.6 s the ball (taken) rolls across the mark at 6 m/s: the sightings on the mark stay.
        base = arc_scene()

        def build(i):
            cands, people, first = base(i)
            t = i/FPS
            if 5.0 <= t < 5.6:
                x = MARK[0]-36+240*(t-5.0)
                c = candidate(x, MARK[1], det=.8, status='ball', tracklet=7)
                cands.append(c)
                first = locked(x, MARK[1], c)
            return cands, people, first
        balls, report = decode(frames_of(200, build, small=True))
        over = [k for k in range(125, 140) if abs(MARK[0]-36+240*(k/FPS-5.0)-MARK[0]) <= 20]
        self.assertTrue(over)
        self.assertTrue(all(balls[k]['state'] == 'TRACKED' and abs(balls[k]['center'][0]*W-(MARK[0]-36+240*(k/FPS-5.0))) < 4 for k in over), over)
        self.assertEqual(report['fixed'][0]['withdrawn'], 60)   # the lock only: the rolling ball's acceptances keep their bonus

    def test_a_still_thing_cut_into_tracklets_is_one_place_and_a_rolling_ball_leaves_short_ones(self):
        frames = frames_of(200, arc_scene(), small=True)
        layers = [ball_path.nodes_of(i, f, W, H) for i, f in enumerate(frames)]
        times = [f['time'] for f in frames]
        T, _, _ = ball_path._chain([(np.eye(3), True, False)]*len(frames))
        places = ball_path._places(layers, times, T, [np.linalg.inv(t) for t in T])
        long = [p for p in places if len(p) >= 2*ball_path.MIN_RUN]
        self.assertEqual([sorted({m.tracklet for _, m in p}) for p in long], [[1, 4, 5]])
        self.assertEqual(len(long[0]), 60+80)
        # 6 m/s: .24 m a frame, .6 m around each first sighting (the ball passing one spot at 3.6-4 s and 7-8 s shares a
        # place): never the 2*MIN_RUN sightings a fixed place needs.
        self.assertTrue(all(len(p) < 2*ball_path.MIN_RUN for p in places if p is not long[0]), [len(p) for p in places])

    def test_with_the_gate_closed_a_result_decodes_as_it_did_without_a_profile(self):
        outputs = []
        for profile in (None, {'smallScene': False}):
            frames = frames_of(200, arc_scene(), small=False)
            for f in frames:
                if profile is None:
                    del f['profile']
            balls, report = decode(copy.deepcopy(frames))
            outputs.append((json.dumps(balls, sort_keys=True), json.dumps(report, sort_keys=True, default=str)))
        self.assertEqual(outputs[0], outputs[1])


if __name__ == '__main__':
    unittest.main()
