"""The ball tracker on the rendered broadcast sequence of ball_eval.py: does it keep following the
SAME ball past painted lines, static spots, a player's wrist tape, white boots, occlusions, a long
shot and a panning camera?"""
import math
import unittest

import ball_eval as E

W, H = E.W, E.H


def run(**kwargs):
    records, tracker = E.Sequence(**kwargs).run()
    return records, tracker, E.evaluate(records, tracker.events)


def near(out, truth, factor=1.5):
    return out['state'] != 'UNKNOWN' and math.hypot(out['center'][0]*W-truth['ball'][0], out['center'][1]*H-truth['ball'][1]) <= max(factor*truth['d'], 6)


def reasons_by_object(records):
    """Rejection reasons given to the candidates at each distractor (frames where the ball itself rolls
    over or past a distractor are left out)."""
    out = {}
    for r in records:
        truth = r['truth']
        for c in r['cands']:
            cx, cy = (c['box'][0]+c['box'][2]/2)*W, (c['box'][1]+c['box'][3]/2)*H
            for name, (ox, oy) in truth['objects'].items():
                if math.hypot(cx-ox, cy-oy) <= 3 and math.hypot(truth['ball'][0]-ox, truth['ball'][1]-oy) > 2*truth['d']:
                    out.setdefault(name, []).append((truth['time'], c['status'], c.get('reason')))
    return out


class MotionStateTests(unittest.TestCase):
    """The causal tracker on a plain synthetic frame: a ball observed at rest is STATIONARY with no velocity carried on,
    an unseen ball is OCCLUDED (never assumed still), and a ball that starts moving again is followed."""

    def run_ball(self, positions, fps=25):
        import numpy as np
        from soccer.ball import BallTracker
        frame = np.zeros((H, W, 3), np.uint8)
        frame[:] = (40, 140, 60)   # grass in BGR
        tracker = BallTracker(W, H, fps)
        outs = []
        for k, pos in enumerate(positions):
            img = frame.copy()
            rows = np.zeros((0, 6), np.float32)
            if pos is not None:
                x, y = pos
                import cv2
                cv2.circle(img, (int(x), int(y)), 4, (255, 255, 255), -1)
                rows = np.array([[x-4, y-4, x+4, y+4, .9, 0]], np.float32)
            out, _ = tracker.step(img, k/fps, rows, np.eye(3), True, False, None, [], None, lambda nx, ny: (nx, ny), 0)
            outs.append(out)
        return outs, tracker

    def test_observed_at_rest_is_stationary_without_a_carried_velocity(self):
        positions = [(300+6*k, 400) for k in range(15)]+[(300+6*14, 400)]*20+[(300+6*14+6*k, 400) for k in range(1, 15)]
        outs, tracker = self.run_ball(positions)
        self.assertEqual(outs[14]['state'], 'TRACKED')
        self.assertEqual(outs[14]['motion'], 'ROLLING')
        rest = outs[24:35]   # the sightings of the last 0.4 s must all show the ball at rest
        self.assertTrue(all(o['state'] == 'TRACKED' and o['motion'] == 'STATIONARY' and o['origin'] == 'OBSERVATION' for o in rest), [(o['state'], o.get('motion')) for o in rest])
        self.assertTrue(all(abs(o['center'][0]*W-(300+6*14)) < 3 for o in rest), 'no drift while at rest')
        self.assertLess(abs(outs[34]['speed']), .5)
        again = outs[40:49]
        self.assertTrue(all(o['state'] == 'TRACKED' and o['motion'] == 'ROLLING' for o in again), [(o['state'], o.get('motion')) for o in again])

    def test_an_unseen_ball_is_occluded_not_stationary(self):
        positions = [(300+6*k, 400) for k in range(15)]+[None]*6
        outs, tracker = self.run_ball(positions)
        hidden = outs[15:21]
        self.assertTrue(all(o['state'] != 'TRACKED' and o.get('motion') in ('OCCLUDED', 'UNKNOWN') and o.get('origin') in ('PREDICTION', 'UNKNOWN') for o in hidden), [(o['state'], o.get('motion')) for o in hidden])


class BallTrackingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.records, cls.tracker, cls.stats = run()

    def test_follows_the_same_ball_through_the_sequence(self):
        s, text = self.stats, E.report(self.stats)
        self.assertGreaterEqual(s['accuracy'], .85, text)
        self.assertEqual(s['falsePositives'], 0, text)
        self.assertEqual(s['trackSwitches'], 0, text)
        self.assertEqual(s['incorrectAcquisitions'], 0, text)
        self.assertEqual(s['stolenBy'], {}, text)
        self.assertEqual(s['trackIds'], [1], text)
        self.assertGreaterEqual(s['meanContinuitySeconds'], 3.0, text)
        # After the ball first moves (1.0 s) and the confirmation frames, every frame is the ball or its prediction.
        late = [r for r in self.records if r['truth']['time'] >= 1.3]
        bridged = lambda r: r['out']['state'] == 'MISSING' and math.hypot(r['out']['center'][0]*W-r['truth']['ball'][0], r['out']['center'][1]*H-r['truth']['ball'][1]) <= 2.0*r['truth']['px_m']
        self.assertTrue(all(near(r['out'], r['truth']) or bridged(r) for r in late),
                        [(r['truth']['time'], r['out']['state']) for r in late if not (near(r['out'], r['truth']) or bridged(r))][:5])

    def test_hidden_frames_are_bridged_by_prediction_not_rediscovery(self):
        s = self.stats
        self.assertEqual(s['bridged'], s['hiddenFrames'], E.report(s))
        self.assertFalse([e for e in self.tracker.events if e['message'].startswith('BALL LOST')])
        hidden = [r['out'] for r in self.records if not r['truth']['visible'] and r['truth']['inScene']]
        self.assertTrue(hidden)
        self.assertTrue(all(o['state'] == 'MISSING' and o['observed'] is False and o['phase'] in ('OCCLUDED', 'RECOVERING') for o in hidden))

    def test_crossing_lines_and_the_centre_circle_keeps_the_track(self):
        shot = [r for r in self.records if 4.25 <= r['truth']['time'] <= 5.4 and r['truth']['visible']]
        correct = [r for r in shot if r['out']['state'] == 'TRACKED' and near(r['out'], r['truth'])]
        self.assertGreaterEqual(len(correct)/len(shot), .9, [(r['truth']['time'], r['out']['state']) for r in shot if r not in correct])

    def test_markings_tape_boots_and_spots_are_rejected_for_the_right_reasons(self):
        by = reasons_by_object(self.records)
        self.assertTrue(all(status != 'ball' for name, items in by.items() for _, status, _ in items), 'a distractor was the ball')
        tape = [reason for t, _, reason in by['wrist tape'] if t >= 1.0]
        self.assertGreaterEqual(tape.count('PLAYER_ATTACHED')/len(tape), .6, set(tape))
        self.assertTrue(set(tape) <= {'PLAYER_ATTACHED', 'OTHER_OBJECT', 'TRAJECTORY', 'LOW_SCORE', 'OUTSIDE_PITCH'}, set(tape))
        boots = [reason for _, _, reason in by['boot 1']+by['boot 2']]
        self.assertTrue({'PLAYER_ATTACHED', 'OTHER_OBJECT'} & set(boots), set(boots))
        self.assertIn('FIELD_LINE', [reason for _, _, reason in by['halfway line']])
        self.assertIn('FIELD_LINE', [reason for _, _, reason in by['circle 1']+by['circle 2']])
        late = [reason for t, _, reason in by['debris'] if t >= 7.0]
        self.assertTrue(late and all(reason in ('STATIC', 'OTHER_OBJECT') for reason in late), set(late))
        self.assertTrue(all(reason in ('STATIC', 'OTHER_OBJECT', 'FIELD_LINE') for t, _, reason in by['centre spot'] if t >= 3.0))  # it lies on the halfway line

    def test_event_log_explains_every_decision(self):
        events = self.tracker.events
        acquired = [e for e in events if e['message'].startswith('BALL ACQUIRED')]
        self.assertEqual(len(acquired), 1)
        for line in ('Track BALL-1', 'BALL CANDIDATE #', 'Detector confidence', 'Independent motion', 'Field line overlap', 'Player attachment score', 'FINAL SCORE', 'ACCEPTED'):
            self.assertIn(line, acquired[0]['message'])
        rejected = [e for e in events if e['message'].startswith('BALL CANDIDATE REJECTED') and 'REJECTED: PLAYER_ATTACHED' in e['message']]
        self.assertTrue(rejected, [e['message'][:80] for e in events if 'REJECTED' in e['message']][:5])
        self.assertIn('same place on track 4', rejected[0]['message'])
        updates = [e for e in events if e['message'].startswith('BALL TRACK UPDATE')]
        self.assertGreaterEqual(len(updates), 8)
        self.assertIn('Track BALL-1: LOCKED', updates[3]['message'])
        self.assertIn('Trajectory consistency', updates[3]['message'])

    def test_output_describes_the_persistent_ball(self):
        locked = next(r['out'] for r in self.records if r['out']['state'] == 'TRACKED' and r['truth']['time'] > 2)
        for key in ('phase', 'observed', 'track', 'age', 'missingFrames', 'predicted', 'velocity', 'speed', 'direction', 'acceleration', 'lastSeen', 'lastConfident'):
            self.assertIn(key, locked)
        self.assertEqual((locked['phase'], locked['observed']), ('LOCKED', True))
        self.assertGreater(locked['speed'], 3)


class BallTrackingVariantTests(unittest.TestCase):
    def test_tape_with_higher_confidence_never_takes_over_while_the_ball_is_hidden(self):
        t0 = E.Match().t_arrive3
        records, tracker, s = run(tape_conf=.85, hidden=[(t0+.1, t0+1.3)], seconds=t0+2.0)
        self.assertEqual((s['falsePositives'], s['trackSwitches'], s['stolenBy'], s['trackIds']), (0, 0, {}, [1]), E.report(s))
        hidden = [r['out'] for r in records if t0+.2 <= r['truth']['time'] < t0+1.3]
        self.assertTrue(all(o['state'] == 'MISSING' and o.get('nearTrack') == 4 for o in hidden), [(o['state'], o.get('nearTrack')) for o in hidden][:6])
        back = [r for r in records if r['truth']['time'] >= t0+1.6]
        self.assertTrue(all(r['out']['state'] == 'TRACKED' and near(r['out'], r['truth']) for r in back), [(r['truth']['time'], r['out']['state']) for r in back][:6])
        self.assertTrue(any(e['message'].startswith('BALL REACQUIRED') for e in tracker.events))

    def test_undetected_spell_reacquires_the_same_ball(self):
        records, tracker, s = run(undetected=[(6.8, 8.8)], seconds=10.5)
        self.assertEqual((s['falsePositives'], s['stolenBy'], s['trackIds']), (0, {}, [1]), E.report(s))
        # Two seconds undetected: the track waits (RECOVERING) or is lost, but never takes anything else.
        self.assertTrue(all(r['out']['state'] != 'TRACKED' for r in records if 7.0 <= r['truth']['time'] < 8.8), [(r['truth']['time'], r['out']['state']) for r in records if 7.0 <= r['truth']['time'] < 8.8 and r['out']['state'] == 'TRACKED'][:5])
        self.assertTrue(any(e['message'].startswith('BALL REACQUIRED') and e['time'] >= 8.8 for e in tracker.events))
        back = [r for r in records if r['truth']['time'] >= 9.8 and r['truth']['visible']]  # hidden behind player 4 until 9.3 s, then 0.5 s to prove motion
        self.assertTrue(all(r['out']['state'] == 'TRACKED' and near(r['out'], r['truth']) for r in back), [(r['truth']['time'], r['out']['state']) for r in back if r['out']['state'] != 'TRACKED'][:6])

    def test_ball_gone_becomes_unknown_and_nothing_static_is_adopted(self):
        # The ball vanishes at a dribbling player's feet: it is carried (MISSING) for at most 3 s, then LOST.
        records, tracker, s = run(ball_until=4.0, seconds=9.0, spot_conf=.8)
        self.assertTrue(any(e['message'].startswith('BALL LOST') for e in tracker.events))
        after = [r['out'] for r in records if r['truth']['time'] >= 7.3]
        self.assertTrue(all(o['state'] == 'UNKNOWN' for o in after), [o['state'] for o in after][:8])
        self.assertEqual(s['falsePositives'], 0, E.report(s))

    def test_confident_static_spots_never_start_a_track(self):
        records, tracker, s = run(ball_until=0.0, seconds=4.0, spot_conf=.85)
        self.assertTrue(all(r['out']['state'] == 'UNKNOWN' for r in records))
        self.assertEqual(s['trackIds'], [])

    def test_calibrated_pitch_marks_painted_spots_as_static_at_once(self):
        records, tracker, s = run(calibrated=True, seconds=2.5)
        by = reasons_by_object(records)
        self.assertTrue(by['penalty spot left'] and all(reason in ('STATIC', 'OTHER_OBJECT') for _, _, reason in by['penalty spot left']), set(r for _, _, r in by['penalty spot left']))
        self.assertEqual(by['penalty spot left'][0][2], 'STATIC')
        self.assertTrue(all(reason in ('STATIC', 'FIELD_LINE') for _, _, reason in by['centre spot']))  # it lies on the halfway line
        self.assertEqual(s['falsePositives'], 0)

    def test_an_untrusted_calibration_does_not_declare_painted_spots(self):
        # The pitch mapping exists but its confidence is low (a carried, unconfirmed registration): the spot
        # under a candidate is not "known" from it; stillness over time still rejects, and nothing false starts.
        records, tracker, s = run(calibrated=True, calibration=.2, seconds=2.5)
        by = reasons_by_object(records)
        self.assertTrue(by['penalty spot left'])
        self.assertNotEqual(by['penalty spot left'][0][2], 'STATIC')
        self.assertEqual(s['falsePositives'], 0, E.report(s))
        self.assertEqual(s['trackIds'], [1], E.report(s))


class SwitchWarningTests(unittest.TestCase):
    """What the causal pass logs when something far from the hidden ball asks to be the ball."""

    def candidate(self, tracker, pitch, **scores):
        from soccer.ball import Candidate
        c = Candidate(index=0, box=[1780, 420, 1800, 440], det=.77, cx=1790.0, cy=430.0, d=20.0, px_m=30.0, stab=(.93, .40), pitch=pitch, person=None,
                      scores={'attach': 0.0, 'static': 0.0, 'settled': 0.0, **scores}, visual=.8)
        c.err, c.required, c.traj, c.entry, c.far = 1200.0, 119.0, 0.02, 'JUMP', True
        return c

    def hidden_tracker(self, calibration):
        from soccer.ball import BallTracker, Kalman
        tracker = BallTracker(W, H, 30.0)
        tracker.kf = Kalman(600.0, 480.0, 20.0, 30.0)
        tracker.phase, tracker.confidence, tracker.last_seen, tracker.calibration = 'OCCLUDED', .55, 12.0, calibration
        return tracker

    def test_a_candidate_on_the_penalty_spot_is_named_a_field_marking(self):
        tracker = self.hidden_tracker(calibration=.9)
        c = self.candidate(tracker, pitch=(94/105+.004, .5), static=1.0, settled=.66)
        tracker._warn_switch(c, 12.5, .5, 'REJECTED')
        message = tracker.events[-1]['message']
        lines = message.split('\n')
        self.assertEqual(lines[0], 'BALL SWITCH WARNING')
        for needle in ('Elapsed time: 0.500 s', 'Required velocity: 119.0 m/s', 'Near penalty spot: YES',
                       'Static field marking evidence: HIGH (static 1.00, settled 0.66)', 'Candidate detector confidence: 0.77',
                       'Decision: REJECTED (JUMP) / FIELD_MARKING'):
            self.assertIn(needle, message, message)

    def test_a_candidate_far_from_any_spot_is_not(self):
        tracker = self.hidden_tracker(calibration=.9)
        c = self.candidate(tracker, pitch=(.70, .30))
        tracker._warn_switch(c, 12.5, .5, 'REJECTED')
        message = tracker.events[-1]['message']
        self.assertIn('Near penalty spot: NO', message)
        self.assertIn('Static field marking evidence: LOW', message)
        self.assertNotIn('FIELD_MARKING', message)

    def test_an_untrusted_calibration_does_not_name_the_spot(self):
        tracker = self.hidden_tracker(calibration=.2)
        c = self.candidate(tracker, pitch=(94/105, .5))
        tracker._warn_switch(c, 12.5, .5, 'REJECTED')
        self.assertIn('Near penalty spot: NO', tracker.events[-1]['message'])


class SettledUnderDriftTests(unittest.TestCase):
    """A thing fixed to the field stays 'settled' while a carried calibration drifts: the camera-compensated image distance
    says still even when the pitch distance does not."""

    def test_image_distance_counts(self):
        from soccer.ball import BallTracker, Candidate
        tracker = BallTracker(W, H, 30.0)
        a = Candidate(index=0, box=[0, 0, 1, 1], det=.7, cx=1790.0, cy=430.0, d=20.0, px_m=30.0, stab=(.93, .40), pitch=(.893, .503), person=None)
        b = Candidate(index=0, box=[0, 0, 1, 1], det=.7, cx=1791.0, cy=430.0, d=20.0, px_m=30.0, stab=(.9305, .40), pitch=(.905, .510), person=None)
        self.assertGreater(tracker._metres(a, b), 1.0)         # the drifting registration says it moved more than a metre
        self.assertLess(tracker._metres_image(a, b), .1)       # the image says it did not move


if __name__ == '__main__':
    unittest.main()
