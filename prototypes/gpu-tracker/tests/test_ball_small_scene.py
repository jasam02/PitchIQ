"""The causal ball tracker's small-scene rules (soccer/ball.py, BallTracker.small_scene): each is checked against the same
scene with the gate closed, which must behave as before. A plain synthetic frame (turf, a cream ball drawn where the
test says), a stub detector the test controls, a static camera."""
import math
import unittest

import cv2
import numpy as np

import ball_eval as E
from soccer import ball as B
from soccer.ball import BallTracker, Candidate, Kalman, Tracklet

W, H, FPS = 1280, 720, 30
GRASS = (40, 120, 55)
CREAM = (215, 228, 232)
BALL_PX = 13          # drawn ball diameter: 1.5x the 0.22 m prior at the default scale below (real balls measure 1.4x)
PERSON = .098         # expected person height (normalized) everywhere: 39 px per metre


def turf():
    rng = np.random.default_rng(5)
    frame = np.empty((H, W, 3), np.uint8)
    frame[:] = GRASS
    return np.clip(frame.astype(int)+rng.integers(-10, 11, (H, W, 1)), 0, 255).astype(np.uint8)


TURF = turf()


def draw(frame, x, y, d=BALL_PX):
    cv2.circle(frame, (int(round(x)), int(round(y))), int(round(d/2)), CREAM, -1)


def row(x, y, det=.7, d=BALL_PX):
    return [x-d/2, y-d/2, x+d/2, y+d/2, det, 0.0]


def step(tracker, t, rows=(), drawn=(), people=(), scale=None, small=False, sources=None, M=None):
    """One frame: `drawn` [(x, y) or (x, y, d)] are painted on the turf, `rows` reported by the detector (with the pass
    that produced each as a seventh column when `sources` is given); M: the camera motion from the previous frame."""
    frame = TURF.copy()
    for p in drawn:
        draw(frame, *p)
    rows = np.asarray(list(rows), np.float64).reshape(-1, 6)
    if sources is not None:
        rows = np.hstack([rows, np.asarray(sources, np.float64).reshape(-1, 1)])
    tracker.small_scene = small
    return tracker.step(frame, t, rows, np.eye(3) if M is None else M, True, False, None, list(people), scale or (lambda y: PERSON),
                        lambda nx, ny: (nx, ny), 0)


def near(out, x, y, tol=8):
    return out['state'] != 'UNKNOWN' and math.hypot(out['center'][0]*W-x, out['center'][1]*H-y) <= tol


class ProposalTrackingTests(unittest.TestCase):
    """The detector goes blind on a ball it saw a second ago (the YOLO ball class at a dribbler's feet, amateur 3
    146-148.8 s): in a small scene the classical proposals carry the ball; with the gate closed nothing changes."""

    def run_blind(self, small):
        tracker = BallTracker(W, H, FPS)
        outs = []
        for k in range(75):
            t = k/FPS
            x, y = 300+4*k, 400+k   # 3.2 m/s
            rows = [row(x, y)] if k < 30 else []   # blind from 1.0 s on, while the ball stays in plain sight
            outs.append((t, x, y)+step(tracker, t, rows, [(x, y)], small=small))
        return outs, tracker

    def test_proposals_carry_the_ball_while_the_detector_is_blind(self):
        outs, tracker = self.run_blind(small=True)
        late = [(t, x, y, out, cands) for t, x, y, out, cands in outs if t >= 1.5]
        tracked = [o for o in late if o[3]['state'] == 'TRACKED' and near(o[3], o[1], o[2])]
        self.assertGreaterEqual(len(tracked), .8*len(late), [(round(o[0], 2), o[3]['state']) for o in late])
        chosen = [c for o in late for c in o[4] if c['status'] == 'ball']
        self.assertTrue(chosen and all(c['source'] == 'proposal' for c in chosen))
        self.assertTrue(all(c['det'] == .12 for c in chosen))    # weak evidence by construction
        self.assertTrue(any(o[3].get('phase') == 'LOCKED' for o in late))

    def test_with_the_gate_closed_the_blind_spell_is_unchanged(self):
        outs, tracker = self.run_blind(small=False)
        late = [o for o in outs if o[0] >= 1.5]
        self.assertFalse(any(o[3]['state'] == 'TRACKED' for o in late))
        self.assertFalse(any(c.get('source') == 'proposal' for o in outs for c in o[4]))
        self.assertIsNone(tracker.anchor)

    def test_a_lone_proposal_is_never_the_ball_and_the_detector_wins_when_it_is_back(self):
        # The ball vanishes (hidden) for .4 s; then a blob the detector does not report appears on its path: alone it is
        # not the ball, a chain of three is.
        tracker = BallTracker(W, H, FPS)
        statuses = []
        for k in range(50):
            x, y = 300+4*k, 400
            if k < 30:
                step(tracker, k/FPS, [row(x, y)], [(x, y)], small=True)
            elif k >= 42:
                _, cands = step(tracker, k/FPS, [], [(x, y)], small=True)
                statuses.append([c['status'] for c in cands if c.get('source') == 'proposal'])
            else:
                step(tracker, k/FPS, [], [], small=True)
        self.assertEqual(statuses[0], ['rejected'])
        self.assertIn('ball', [s for frame in statuses[2:] for s in frame])
        # A proposal continuing the ball's own chain of detections is part of that chain (the blind spell above).
        outs, _ = self.run_blind(small=True)
        first = next(o for o in outs if any(c.get('source') == 'proposal' for c in o[4]))
        self.assertGreaterEqual(first[0], 29/FPS+B.PROPOSE_AFTER-1e-6)   # unseen since the last detection (frame 29) for PROPOSE_AFTER
        # The detector sees the ball again: its candidate is the ball, and nothing is proposed inside its box.
        tracker = BallTracker(W, H, FPS)
        for k in range(75):
            x, y = 300+4*k, 400+k
            out, cands = step(tracker, k/FPS, [row(x, y)] if k < 30 or k >= 60 else [], [(x, y)], small=True)
            if k >= 61:
                self.assertEqual([c.get('source', 'detector') for c in cands if c['status'] == 'ball'], ['detector'])
                self.assertFalse(any(c.get('source') == 'proposal' for c in cands))


class ProposalLockTests(unittest.TestCase):
    """A classical proposal never beats the detector, and while proposals alone carry the lock the detector is blind on
    the ball: the pipeline must keep scanning the whole frame for it (scan_locked False)."""

    @staticmethod
    def ready(tracker, tid, y, n, visual, last_source, time):
        """A tracklet ready to become the ball at `time`: n sightings, 6 px (.15 m) apart per frame, the last from `last_source`."""
        scores = {'attach': 0.0, 'attachedFor': 0.0, 'motion': .5, 'speed': 4.5, 'line': 0.0, 'static': 0.0, 'size': .8, 'shape': .8,
                  'isolation': .8, 'field': 1.0, 'fit': 1.0}
        t = None
        for k in range(n):
            when, x = time-(n-1-k)/FPS, 300+6*k
            c = Candidate(0, [x-6, y-6, x+6, y+6], .7, x, y, 12.0, 40.0, (x/W, y/H), None, None, scores=dict(scores), visual=visual)
            c.source = last_source if k == n-1 else 0
            t = t or Tracklet(tid, Kalman(x, y, 12.0, 40.0), when, c, FPS)
            t.hits.append((when, c))
            t.last = when
        tracker.tracklets.append(t)
        return t

    def test_a_tracklet_the_detector_saw_is_promoted_before_one_a_proposal_carries(self):
        tracker = BallTracker(W, H, FPS)
        tracker.small_scene = True
        self.ready(tracker, 1, 300, 10, .8, B.PROPOSAL, 2.0)   # ranks first (visual x sightings 8.0) ...
        detected = self.ready(tracker, 2, 500, 5, .6, 0, 2.0)  # ... over this one (3.0), but its sighting now is a proposal
        chosen = tracker._promote(2.0)
        self.assertIs(chosen, detected.hits[-1][1])
        self.assertEqual((tracker.phase, tracker.lock_source, tracker.scan_locked), ('LOCKED', 0, True))
        # Alone, the proposal-led tracklet is taken (a chain that moved like a ball), but the frame scan goes on.
        tracker = BallTracker(W, H, FPS)
        tracker.small_scene = True
        proposed = self.ready(tracker, 1, 300, 10, .8, B.PROPOSAL, 2.0)
        self.assertIs(tracker._promote(2.0), proposed.hits[-1][1])
        self.assertEqual((tracker.phase, tracker.lock_source, tracker.scan_locked), ('LOCKED', B.PROPOSAL, False))

    def test_scan_locked_is_off_while_proposals_alone_carry_the_lock(self):
        tracker = BallTracker(W, H, FPS)
        states = []
        for k in range(75):
            x, y = 300+4*k, 400+k
            out, _ = step(tracker, k/FPS, [row(x, y)] if k < 30 else [], [(x, y)], small=True)
            states.append((k, tracker.phase, tracker.lock_source, tracker.scan_locked))
        self.assertTrue(all(scan for k, phase, _, scan in states if 10 <= k < 30))   # the detector's ball: a safeguard scan only
        carried = [s for s in states if s[0] >= 30 and s[1] == 'LOCKED']
        self.assertGreater(len(carried), 20)
        self.assertTrue(all(source == B.PROPOSAL and not scan for _, _, source, scan in carried), carried)
        # The gate closes (a far band that went away): no proposal carries the lock from here on.
        x, y = 300+4*75, 400+75
        step(tracker, 75/FPS, [], [(x, y)], small=False)
        self.assertEqual(tracker.scan_locked, tracker.phase == 'LOCKED')
        self.assertNotEqual(tracker.lock_source, B.PROPOSAL)


class ProvisionalLockTests(unittest.TestCase):
    def test_a_provisional_lock_proves_itself_and_then_rules_other_things_out(self):
        # The ball rolls (1 s), hides .8 s, reappears deflected 3 m beyond its mean prediction and rolls on past a player;
        # from 3.3 s a weak white thing is seen far away.
        player = (7, [500/W, 450/H, 40/W, 110/H])   # feet at (520, 560)
        for small in (True, False):
            tracker = BallTracker(W, H, FPS)
            log = []
            for k in range(140):
                t = k/FPS
                if k < 30:
                    x, y = 300+5*k, 400
                    step(tracker, t, [row(x, y)], [(x, y)], small=small)
                    continue
                if k < 54:
                    step(tracker, t, [], [], small=small)
                    continue
                x, y = 460+(k-54), 420+2*(k-54)
                rows, drawn = [row(x, y, det=.8)], [(x, y)]
                if k >= 100:
                    rows.append(row(1100, 200+3*(k-100), det=.4))
                    drawn.append((1100, 200+3*(k-100)))
                _, cands = step(tracker, t, rows, drawn, people=[player], small=small)
                far = [c for c in cands if c['box'][0]*W > 1000]
                log.append((k, tracker.anchor, tracker.scan_locked, tracker.phase, far[0].get('reason') if far else None))
            if not small:
                self.assertTrue(all(anchor is None for _, anchor, *_ in log))
                continue
            provisional = [k for k, anchor, *_ in log if anchor is not None]
            self.assertTrue(provisional)
            self.assertEqual(provisional[0], 54)                       # taken beyond the mean prediction after a long hide
            self.assertTrue(all(not scan for k, _, scan, *_ in log if k in provisional))
            proven = max(provisional)+1
            self.assertLess(proven, 100)                               # moved a metre and met the player: an ordinary lock
            self.assertTrue(all(scan for k, anchor, scan, phase, _ in log if k >= proven and phase == 'LOCKED'))
            self.assertIn('OTHER_OBJECT', [reason for k, *_, reason in log if k >= 100])   # it rules things out again

    @staticmethod
    def parked_on_a_spot(anchor):
        """A lock standing still for 2.7 s on a white spot first seen at 1.0 s; the track was born at 0 and, when `anchor`
        is set, provisionally re-taken at that time."""
        tracker = BallTracker(W, H, FPS)
        tracker.small_scene = True
        tracker.kf, tracker.phase, tracker.born, tracker.track, tracker.anchor = Kalman(700.0, 330.0, 8.0, 40.0), 'LOCKED', 0.0, 1, anchor
        stab = (700/W, 330/H)
        tracker.spots = [{'stab': stab, 'pitch': None, 'first': 1.0, 'last': 5.6, 'hits': 60, 'segment': 0, 'false_positive': False,
                          'flag_until': -math.inf, 'flag_track': None, 'flag_by': None, 'ball_track': 1}]
        for k in range(82):
            tracker.history.append((3.0+k/FPS, 700.0, 330.0, True, stab, None, 40.0))
        chosen = Candidate(0, [696, 326, 704, 334], .5, 700.0, 330.0, 8.0, 40.0, stab, None, None)
        tracker._check_parked(chosen, 5.7)
        return tracker

    def test_a_provisional_lock_is_judged_parked_against_its_own_sightings(self):
        # Re-taken at 3.0 s on a spot that was there before (1.0 s): a static false positive, lost. Judged from the
        # track's birth (0 s) the spot came later, and the ball merely rests.
        lost = self.parked_on_a_spot(3.0)
        self.assertIsNone(lost.kf)
        self.assertIn('static false positive', lost.events[-1]['message'])
        resting = self.parked_on_a_spot(None)
        self.assertEqual(resting.phase, 'LOCKED')
        self.assertIsNotNone(resting.parked_since)


class ProposalWindowTests(unittest.TestCase):
    def test_after_a_loss_proposals_look_around_the_last_sighting_for_a_while(self):
        # The ball rolls (1 s), hides until it is lost (2.5 s unseen), then rolls on near where it was last seen, in plain
        # sight but undetected: proposals find it for PROPOSE_LOST_S after the loss, never later, never with the gate closed.
        def run(reappear, small=True):
            """[(time, the ball was lost when the frame began, proposal)], the time of the loss, the tracker."""
            tracker = BallTracker(W, H, FPS)
            proposals, lost_at = [], None
            for k in range(reappear+30):
                t = k/FPS
                x, y = 300+5*min(k, 29)+2*max(0, k-reappear), 400
                if k < 30:
                    step(tracker, t, [row(x, y)], [(x, y)], small=small)
                    continue
                lost = tracker.kf is None
                _, cands = step(tracker, t, [], [(x, y)] if k >= reappear else [], small=small)
                if lost_at is None and tracker.lost_at is not None:
                    lost_at = tracker.lost_at['time']
                proposals += [(t, lost, c) for c in cands if c.get('source') == 'proposal']
            return proposals, lost_at, tracker
        proposals, lost_at, tracker = run(110)
        self.assertIsNotNone(lost_at)
        self.assertLess(lost_at, 110/FPS)
        searching = [t for t, lost, _ in proposals if lost]
        self.assertTrue(searching)
        self.assertLessEqual(max(searching)-lost_at, B.PROPOSE_LOST_S)
        self.assertTrue(all(abs((c['box'][1]+c['box'][3]/2)*H-400) < 3 for *_, c in proposals))   # on the ball
        self.assertEqual((tracker.phase, tracker.lock_source, tracker.scan_locked), ('LOCKED', B.PROPOSAL, False))   # taken again
        late, lost_at, tracker = run(int(round((lost_at+B.PROPOSE_LOST_S+.2)*FPS)))
        self.assertEqual((late, tracker.phase), ([], 'SEARCHING'))
        closed, _, _ = run(110, small=False)
        self.assertEqual(closed, [])

    def test_the_last_sighting_moves_with_the_camera(self):
        # The camera pans 6 px per frame; the ball rolls 2 px per frame the other way (4 px per frame in the image). The
        # detector goes blind at 1 s: the last sighting, moved with the camera, stays on the ball, and so do the proposals.
        M = np.array([[1.0, 0.0, -6.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        tracker = BallTracker(W, H, FPS)
        errors, found, last = [], [], None
        for k in range(50):
            x, y = 900-4*k, 400
            _, cands = step(tracker, k/FPS, [row(x, y)] if k < 30 else [], [(x, y)], small=True, M=M)
            taken = [c for c in cands if c['status'] == 'ball']
            # Where the ball was last seen, in this frame's image: the sighting taken now, else the previous one panned along.
            if taken:
                last = ((taken[0]['box'][0]+taken[0]['box'][2]/2)*W, (taken[0]['box'][1]+taken[0]['box'][3]/2)*H)
            elif last is not None:
                last = (last[0]-6, last[1])
            if k >= 30:
                errors.append(math.hypot(tracker.seen_at[0]-last[0], tracker.seen_at[1]-last[1]))
                found += [math.hypot((c['box'][0]+c['box'][2]/2)*W-x, (c['box'][1]+c['box'][3]/2)*H-y) for c in cands if c.get('source') == 'proposal']
        self.assertLess(max(errors), .05, errors)
        self.assertGreaterEqual(len(found), 10)
        self.assertLess(max(found), 2.0, found)


class GateAndLockTests(unittest.TestCase):
    """The wrong-lock chain of amateur 3 147-151 s: a lone far detection taken through the grown hidden-ball gate, a lock
    on a still thing that counted as mature, and the real ball ruled out as "seen while the ball was elsewhere"."""

    @staticmethod
    def depth(y):
        # Near rows: people 0.2 of the frame tall (80 px per metre); far rows: 0.06 (24 px per metre).
        return .2 if y > .5 else .06

    def test_a_lone_far_detection_after_a_long_hide_is_judged_at_its_own_depth(self):
        for small, taken in ((False, True), (True, False)):
            tracker = BallTracker(W, H, FPS)
            for k in range(30):   # the ball near the camera, rolling slowly
                x, y = 400+2*k, 500
                step(tracker, k/FPS, [row(x, y, d=22)], [(x, y, 22)], scale=self.depth, small=small)
            for k in range(30, 69):   # hidden 1.3 s
                step(tracker, k/FPS, [], [], scale=self.depth, small=small)
            # One detection at a far row, 250 px from the prediction: 3 m at its own depth, 10 m at the last sighting's.
            out, cands = step(tracker, 69/FPS, [row(520, 270, det=.6, d=6)], [(520, 270, 6)], scale=self.depth, small=small)
            far = [c for c in cands if abs((c['box'][0]+c['box'][2]/2)*W-520) < 3]
            self.assertEqual(len(far), 1)
            self.assertEqual(far[0]['status'] == 'ball', taken, (small, far[0]))
            if small:
                self.assertLess(far[0]['gate'][0], .5*tracker.gates['tight'])   # the gate at the candidate's depth

    def lock_on_a_still_thing(self, small):
        """A ball rolls for a second and hides; a still white thing far beyond its mean prediction (but inside the gate
        grown while hidden) is taken; then the real ball rolls free on the other side of the frame."""
        tracker = BallTracker(W, H, FPS)
        records = []
        for k in range(30):
            x, y = 300+5*k, 400
            records.append(step(tracker, k/FPS, [row(x, y)], [(x, y)], small=small))
        for k in range(30, 54):   # hidden .8 s
            records.append(step(tracker, k/FPS, [], [], small=small))
        speck = (720, 330)
        for k in range(54, 150):
            t = k/FPS
            rows, drawn = [row(*speck, det=.6)], [speck]
            if k >= 90:   # the real ball, free and rolling, strongly detected, far from the speck
                bx, by = 200+6*(k-90), 600
                rows.append(row(bx, by, det=.9))
                drawn.append((bx, by))
            records.append(step(tracker, t, rows, drawn, small=small))
            if k == 60:
                records[-1] = records[-1]+(tracker.scan_locked, tracker.phase, tracker._speed())
        return records, tracker

    def test_a_lock_taken_beyond_the_gate_is_provisional_and_rules_nothing_out(self):
        records, tracker = self.lock_on_a_still_thing(small=True)
        taken = [k for k, r in enumerate(records) if any(c['status'] == 'ball' and abs((c['box'][0]+c['box'][2]/2)*W-720) < 3 for c in r[1])]
        self.assertTrue(taken)   # the still thing was taken (the grown gate let it in) ...
        scan_locked, phase, speed = records[60][2:]
        self.assertEqual(phase, 'LOCKED')
        self.assertFalse(scan_locked)                      # ... provisionally: the pipeline keeps scanning the frame
        self.assertLess(speed, .5)                         # no velocity carried over from the ball it replaced
        late = [c for r in records[90:] for c in r[1] if abs((c['box'][1]+c['box'][3]/2)*H-600) < 3]
        self.assertTrue(late)
        self.assertFalse([c for c in late if c.get('reason') == 'OTHER_OBJECT'])
        events = [e['message'] for e in tracker.events if 'PROVISIONAL LOCK' in e['message']]
        self.assertTrue(events)

    def test_with_the_gate_closed_the_still_lock_rules_the_real_ball_out_as_before(self):
        records, tracker = self.lock_on_a_still_thing(small=False)
        scan_locked, phase, _ = records[60][2:]
        self.assertEqual(scan_locked, phase == 'LOCKED')
        late = [c for r in records[90:] for c in r[1] if abs((c['box'][1]+c['box'][3]/2)*H-600) < 3]
        flagged = [c for c in late if c.get('reason') == 'OTHER_OBJECT']
        self.assertTrue(flagged)
        self.assertTrue(all(isinstance(c.get('flaggedBy'), int) for c in flagged))   # who ruled it out (diagnostics)

    def test_a_weak_lock_never_rules_out_a_much_stronger_detection(self):
        for small in (False, True):
            tracker = BallTracker(W, H, FPS)
            reasons = []
            for k in range(90):
                t = k/FPS
                x, y = 300+5*k, 400
                rows, drawn = [row(x, y, det=.3)], [(x, y)]
                if k >= 40:
                    sx, sy, ox, oy = 900-4*(k-40), 620, 1100, 200+3*(k-40)
                    rows += [row(sx, sy, det=.9), row(ox, oy, det=.35)]
                    drawn += [(sx, sy), (ox, oy)]
                out, cands = step(tracker, t, rows, drawn, small=small)
                if k >= 60:
                    reasons.append({round((c['box'][0]+c['box'][2]/2)*W) > 1000: c.get('reason') for c in cands if (c['box'][1]+c['box'][3]/2)*H > 500 or (c['box'][0]+c['box'][2]/2)*W > 1000})
            strong = [r.get(False) for r in reasons]
            weak = [r.get(True) for r in reasons]
            self.assertEqual('OTHER_OBJECT' in strong, not small, strong)
            self.assertIn('OTHER_OBJECT', weak)   # an equally weak thing elsewhere is still ruled out

    def test_the_balls_own_tracklet_follows_a_kick_nobody_was_seen_making(self):
        results = {}
        for small in (False, True):
            tracker = BallTracker(W, H, FPS)
            tracked = 0
            x, y = 300.0, 400.0
            for k in range(60):
                vx = 2.5 if k < 30 else 30.0   # 2 m/s, then kicked to ~28 m/s by a player the detector did not see
                x += vx
                out, _ = step(tracker, k/FPS, [row(x, y)], [(x, y)], small=small)
                if 30 <= k < 42:
                    tracked += out['state'] == 'TRACKED' and near(out, x, y, 12)
            results[small] = tracked
        self.assertGreaterEqual(results[True], 8, results)
        self.assertLess(results[False], results[True], results)


class EntryAndSizeTests(unittest.TestCase):
    def hidden(self, small):
        tracker = BallTracker(W, H, FPS)
        tracker.kf = Kalman(600.0, 400.0, 13.0, 40.0)   # the prediction drifted to here ...
        tracker.phase, tracker.last_seen, tracker.small_scene = 'RECOVERING', 1.0, small
        tracker.seen_at = (300.0, 400.0, 40.0)           # ... from where the ball was last seen
        return tracker

    def chain(self, x, y, t0):
        mk = lambda: Candidate(0, [x-6, y-6, x+6, y+6], .7, x, y, 12.0, 40.0, (x/W, y/H), None, None, scores={'speed': 0.0}, visual=.8)
        return [(t0+k/FPS, mk()) for k in range(3)], mk()

    def test_the_entry_rule_also_measures_from_the_last_sighting(self):
        for small, verdict in ((False, 'JUMP'), (True, None)):
            tracker = self.hidden(small)
            previous, c = self.chain(310.0, 400.0, 1.5)
            self.assertEqual(tracker._entry(c, previous, 1.6), verdict)

    def test_the_size_prior_is_floored_and_centred_on_the_measured_ball(self):
        far = lambda y: .03   # people 22 px tall: the prior ball is 2.6 px
        for small in (False, True):
            tracker = BallTracker(W, H, FPS)
            _, cands = step(tracker, 0.0, [row(500, 300, det=.7, d=10)], [(500, 300, 10)], scale=far, small=small)
            self.assertEqual(cands[0].get('reason') == 'SIZE', not small, cands[0])
        # The measured ball (1.4x the prior) scores best in a small scene, not a speck of the prior's size.
        tracker = BallTracker(W, H, FPS)
        _, cands = step(tracker, 0.0, [row(300, 300, det=.5, d=12.1), row(800, 300, det=.5, d=8.6)], [(300, 300, 12), (800, 300, 9)], small=True)
        by_x = {round((c['box'][0]+c['box'][2]/2)*W): c['scores']['size'] for c in cands}
        self.assertGreater(by_x[300], by_x[800])

    def test_strong_candidates_past_the_twelve_written_are_kept_in_a_small_scene(self):
        rows = [row(100+60*k, 300, det=.95) for k in range(12)]+[row(500, 600, det=.5, d=40)]
        for small, written in ((False, 12), (True, 13)):
            tracker = BallTracker(W, H, FPS)
            _, cands = step(tracker, 0.0, rows, [], small=small)
            self.assertEqual(len(cands), written)


class DiagnosticsAndGateTests(unittest.TestCase):
    def test_a_source_column_changes_no_decision_and_is_written_unless_it_is_the_detector(self):
        outs, written = {}, []
        for columns in (6, 7):
            tracker = BallTracker(W, H, FPS)
            outs[columns] = []
            for k in range(40):
                x, y = 300+5*k, 400
                rows = ([row(x, y)] if k % 9 else [])+[row(900, 200, det=.4), row(600, 600, det=.5)]
                # The whole-frame detector, the ball tiles and the search crop, in turn on every row.
                sources = [(k+j) % 3 for j in range(len(rows))] if columns == 7 else None
                outs[columns].append(step(tracker, k/FPS, rows, [(x, y), (900, 200), (600, 600)], sources=sources))
                if sources is not None:
                    # Candidates are written in visual order, not row order: match them back by position.
                    by_row = {(round(r[0]+r[2]), round(r[1]+r[3])): s for r, s in zip(rows, sources)}
                    written += [(by_row[(round((c['box'][0]*2+c['box'][2])*W), round((c['box'][1]*2+c['box'][3])*H))], c.get('source'))
                                for c in outs[columns][-1][1]]
            debug = tracker.frame_debug
            self.assertTrue({'phase', 'tight', 'path', 'wide', 'pxm', 'lockTracklet'} <= set(debug), debug)
            self.assertFalse(tracker.small_scene or 'smallScene' in debug)
        strip = lambda cands: [{k: v for k, v in c.items() if k != 'source'} for c in cands]
        self.assertEqual([(o, strip(c)) for o, c in outs[6]], [(o, strip(c)) for o, c in outs[7]])
        self.assertEqual({s: {w for r, w in written if r == s} for s in (0, 1, 2)}, {0: {None}, 1: {'tiles'}, 2: {'crop'}})
        self.assertFalse(any('source' in c for _, cands in outs[6] for c in cands))   # a missing source: the detector

    def test_the_broadcast_like_sequence_is_followed_as_well_with_the_gate_open(self):
        stats = {}
        for small in (False, True):
            tracker = BallTracker(E.W, E.H, E.FPS)
            tracker.small_scene = small
            records, tracker = E.Sequence(seconds=8.0).run(tracker=tracker)
            s = E.evaluate(records, tracker.events)
            stats[small] = {k: s[k] for k in ('accuracy', 'falsePositives', 'trackSwitches', 'incorrectAcquisitions', 'trackIds')}
            if not small:
                self.assertTrue(all(r['out']['state'] != 'TRACKED' or r['out']['phase'] == 'LOCKED' for r in records))
        self.assertEqual(stats[False], stats[True])

    def test_scan_locked_is_the_phase_whenever_the_gate_is_closed(self):
        tracker = BallTracker(W, H, FPS)
        for k in range(60):
            x, y = 300+5*k, 400
            step(tracker, k/FPS, [row(x, y)] if k < 20 or k > 45 else [row(800, 300)], [(x, y), (800, 300)])
            self.assertEqual(tracker.scan_locked, tracker.phase == 'LOCKED')


if __name__ == '__main__':
    unittest.main()
