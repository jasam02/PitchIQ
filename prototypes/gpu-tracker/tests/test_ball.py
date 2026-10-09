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
        self.assertTrue(any(e['message'].startswith('BALL LOST') for e in tracker.events))
        self.assertTrue(any(e['message'].startswith('BALL REACQUIRED') and e['time'] > 8.8 for e in tracker.events))
        back = [r for r in records if r['truth']['time'] >= 9.5 and r['truth']['visible']]  # hidden behind player 4 until 9.3 s
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


if __name__ == '__main__':
    unittest.main()
