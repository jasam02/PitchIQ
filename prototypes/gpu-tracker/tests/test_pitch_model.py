"""The tracked pitch model: canonical geometry, confidence, carrying, refinement, offline propagation,
and the robust-footage behaviour of the person filter and the tracker (modes and confirmation requests)."""
import math
import unittest

import cv2
import numpy as np

import broadcast
from scene import H, W, draw_person, person_box, pitch_frame
from soccer import calibration, field
from soccer.camera import Keyframes
from soccer.pitch import FieldFilter, PitchDetector, PitchModel, assess_confidence, suggest_landmarks
from soccer.pitch_model import HIGH, MEDIUM, MIN_ANCHOR, PitchTracker, propagate
from soccer.relevance import assess, fit_size_model

SIZE = field.PitchSize()
MARKS = field.landmarks(SIZE)


def image(P, name):
    x, y = field.project(P, [MARKS[name]])[0]
    return {'name': name, 'x': float(x), 'y': float(y)}


def normalized(P, name):
    p = image(P, name)
    return {'name': name, 'x': p['x']/W, 'y': p['y']/H}


def youth_camera(length=90.0, width=55.0):
    src = np.float32([[0, 0], [length, 0], [length, width], [0, width]])
    dst = np.float32([[150, 200], [1130, 200], [1250, 680], [30, 680]])
    return cv2.getPerspectiveTransform(src, dst).astype(np.float64), field.PitchSize(length, width)


def segments(P, names):
    """Painted segments of the named marking lines as pitch.detect_lines would report them (normalized)."""
    catalogue = {name: (a, b) for name, a, b in field.lines(SIZE)}
    out = []
    for name in names:
        a, b = catalogue[name]
        qa, qb = field.project(P, [a, b])
        out.append({'kind': 'v' if name.endswith('line') and 'touch' not in name else 'h',
                    'a': (qa[0]/W, qa[1]/H), 'b': (qb[0]/W, qb[1]/H), 'support': .5})
    return out


class FieldTests(unittest.TestCase):
    def setUp(self):
        self.P = broadcast.camera()

    def test_homography_from_landmarks_is_exact(self):
        pts = [image(self.P, n) for n in ('halfway-far', 'halfway-near', 'centre-spot', 'left-penalty-box-front-far', 'right-penalty-box-front-near')]
        fit = field.fit({'points': pts}, W, H, SIZE)
        self.assertTrue(fit.ok, fit.reason)
        got = field.project(fit.H, [MARKS['right-penalty-spot']])[0]
        want = field.project(self.P, [MARKS['right-penalty-spot']])[0]
        self.assertLess(np.hypot(*(got-want)), .5)
        self.assertIsNone(field.sanity(fit.H, SIZE, W, H))
        self.assertGreater(len(field.outline(fit.H, SIZE, W, H)), 20)

    def test_painted_lines_pull_a_drifted_prior_back(self):
        drifted = np.array([[1, 0, 25], [0, 1, -8], [0, 0, 1]], float) @ self.P
        lines = []
        catalogue = {name: (a, b) for name, a, b in field.lines(SIZE)}
        for name in ('far touchline', 'near touchline', 'halfway line'):
            qa, qb = field.project(self.P, catalogue[name])
            lines.append({'name': name, 'a': tuple(qa), 'b': tuple(qb)})
        fit = field.fit({'lines': lines}, W, H, SIZE, prior=drifted)
        self.assertTrue(fit.ok, fit.reason)
        spot = MARKS['right-penalty-spot']
        before = np.hypot(*(field.project(drifted, [spot])[0]-field.project(self.P, [spot])[0]))
        after = np.hypot(*(field.project(fit.H, [spot])[0]-field.project(self.P, [spot])[0]))
        self.assertGreater(before, 20)
        self.assertLess(after, 2)

    def test_size_is_learned_only_when_the_landmarks_pin_it_down(self):
        Py, youth = youth_camera()
        marks = field.landmarks(youth)
        pt = lambda n: {'name': n, 'x': float(field.project(Py, [marks[n]])[0][0]), 'y': float(field.project(Py, [marks[n]])[0][1])}
        rich = [pt(n) for n in ('corner-far-left', 'halfway-far', 'halfway-near', 'centre-spot', 'left-penalty-box-front-far', 'left-goal-box-front-far', 'centre-circle-far')]
        learned = field.fit_keyframes([rich], W, H)
        self.assertIsNotNone(learned)
        self.assertAlmostEqual(learned[0].length, 90, delta=1.5)
        self.assertAlmostEqual(learned[0].width, 55, delta=1.5)
        # Corners of the far touchline and points on the halfway line fit any size exactly: nothing is learned.
        weak = [pt(n) for n in ('corner-far-left', 'corner-far-right', 'halfway-far', 'halfway-near', 'centre-spot')]
        self.assertIsNone(field.fit_keyframes([weak], W, H))
        # The calibration module keeps the standard size when the landmarks fit it about as well.
        standard = [normalized(self.P, n) for n in ('halfway-far', 'halfway-near', 'centre-spot', 'left-penalty-box-front-far', 'corner-far-right', 'right-penalty-spot')]
        self.assertIsNone(calibration.learn_size([{'time': 0, 'points': standard}], W, H))

    def test_paint_is_matched_only_to_the_marking_it_lies_on(self):
        segs = segments(self.P, ['far touchline', 'halfway line'])
        far = dict(segs[0], a=(segs[0]['a'][0], segs[0]['a'][1]+40/H), b=(segs[0]['b'][0], segs[0]['b'][1]+40/H))   # 40 px off any line
        matched = field.match_segments(self.P, SIZE, segs+[far], W, H)
        self.assertEqual(sorted(m['name'] for m in matched), ['far touchline', 'halfway line'])
        self.assertEqual(field.side_status(self.P, SIZE, W, H, matched), {'far': 'line', 'near': 'predicted', 'left': 'off-screen', 'right': 'off-screen'})


class FieldGeometryTests(unittest.TestCase):
    """The canonical field in one place: named markings, goalposts, arcs, the painted spots, and each
    dimension's source in the run metadata."""

    def test_semantic_names_are_the_established_strings(self):
        names = {name for name, _, _ in field.lines(field.PitchSize())}
        for name in (field.HALFWAY_LINE, field.TOUCHLINE_FAR, field.TOUCHLINE_NEAR, field.GOAL_LINE_LEFT, field.GOAL_LINE_RIGHT,
                     field.PENALTY_AREA_LEFT+' front', field.PENALTY_AREA_RIGHT+' far side', field.GOAL_AREA_LEFT+' near side'):
            self.assertIn(name, names)
        self.assertEqual(field.SIDES, {'far': 'far touchline', 'near': 'near touchline', 'left': 'left goal line', 'right': 'right goal line'})
        marks = field.landmarks(field.PitchSize())
        for name in (field.CENTRE_SPOT, field.PENALTY_SPOT_LEFT, field.PENALTY_SPOT_RIGHT):
            self.assertIn(name, marks)

    def test_goalposts_arcs_and_spots_follow_the_pitch_size(self):
        size = field.PitchSize(100.0, 64.0, 'landmarks')
        posts = field.goalposts(size)
        self.assertAlmostEqual(posts[field.GOALPOST_LEFT_NEAR][1]-posts[field.GOALPOST_LEFT_FAR][1], field.GOAL_WIDTH)
        self.assertAlmostEqual((posts[field.GOALPOST_RIGHT_NEAR][1]+posts[field.GOALPOST_RIGHT_FAR][1])/2, 32.0)
        self.assertEqual(posts[field.GOALPOST_RIGHT_FAR][0], 100.0)
        arcs = {name: (centre, radius, a0, a1) for name, centre, radius, a0, a1 in field.arcs(size)}
        (sx, sy), radius, a0, a1 = arcs[field.PENALTY_ARC_LEFT]
        for angle in (a0, a1):   # the arc ends on the penalty area's front
            self.assertAlmostEqual(sx+radius*math.cos(math.radians(angle)), field.PA, places=6)
        self.assertEqual(arcs[field.CENTER_CIRCLE][0], field.circle_centre(size))
        self.assertEqual(field.painted_spots(size)[0], (11/100, .5))
        from soccer.geometry import PAINTED_SPOTS
        self.assertEqual(PAINTED_SPOTS, ((11/105, .5), (94/105, .5), (.5, .5)))   # the standard pitch, as the ball code used it

    def test_the_run_metadata_says_which_dimensions_are_assumed(self):
        from soccer import registration
        self.assertEqual(field.PitchSize().dimensions(), {'length': {'value': 105.0, 'source': 'assumed'}, 'width': {'value': 68.0, 'source': 'assumed'}})
        self.assertEqual(field.PitchSize(100.0, 64.0, 'landmarks').dimensions()['width']['source'], 'landmarks')
        timeline = [{'H': None, 'confidence': 0.0, 'observability': 0.0, 'source': 'none'}]
        self.assertNotIn('fieldSize', registration.summarize(timeline, []))
        anchor = registration.Anchor(0, 0.0, np.eye(3), .3, 'auto-midfield', {'size': {'width': {'value': 68.0, 'source': 'assumed', 'measured': 70.2}}})
        size = registration.summarize(timeline, [anchor], 0, field.PitchSize())['fieldSize']
        self.assertEqual(size['length']['source'], 'assumed')
        self.assertEqual(size['widthMeasured'], {'median': 70.2, 'anchors': 1})


class ConfidenceTests(unittest.TestCase):
    def test_broadcast_frame_with_painted_touchlines_is_high(self):
        model = PitchDetector().analyze(broadcast.render(broadcast.camera()), 0.0)
        confidence, evidence = assess_confidence(model)
        self.assertGreaterEqual(confidence, HIGH)
        self.assertTrue(any(e.startswith('far touchline: FOUND') for e in evidence), evidence)

    def test_grass_without_painted_lines_is_low(self):
        model = PitchDetector().analyze(broadcast.render(broadcast.camera(), paint=False), 0.0)
        self.assertTrue(model.reliable)   # a grass region exists ...
        confidence, evidence = assess_confidence(model)
        self.assertLess(confidence, MEDIUM)  # ... but it is no evidence of where the touchlines are
        self.assertTrue(all('NOT FOUND' in e for e in evidence), evidence)


class PitchTrackerTests(unittest.TestCase):
    def setUp(self):
        self.P = broadcast.camera()
        self.tracker = PitchTracker(W, H)
        self.assertTrue(self.tracker.anchor(np.linalg.inv(self.P), 0.0, .1))

    def test_anchor_is_carried_by_the_camera_and_decays_without_confirmation(self):
        shift = np.array([[1, 0, -40], [0, 1, 0], [0, 0, 1]], float)
        for k in range(10):
            self.tracker.carry(shift, True, False, .2, .2*(k+1))
        self.assertEqual(self.tracker.level, 'HIGH')
        self.assertEqual(self.tracker.state, 'carried')
        centre = field.project(self.P, [MARKS['centre-spot']])[0]-[400, 0]
        x, y = self.tracker.to_pitch(centre)
        self.assertAlmostEqual(x, .5, places=3)
        self.assertAlmostEqual(y, .5, places=3)
        for k in range(30):   # six seconds of unreliable camera motion
            self.tracker.carry(None, False, False, .2, 2+.2*(k+1))
        self.assertIsNone(self.tracker.H)
        self.assertEqual(self.tracker.state, 'lost')
        self.assertIn('unreliable', self.tracker.lost)

    def test_painted_lines_refine_a_drifted_model(self):
        wrong = np.array([[1, 0, -14], [0, 1, 6], [0, 0, 1]], float)   # the motion estimate was off
        self.tracker.carry(wrong, True, False, 6.0, 6.0)             # ... and the model aged past the refinement gate
        self.assertLess(self.tracker.confidence, .9)
        self.tracker.refine(segments(self.P, ['far touchline', 'near touchline', 'halfway line']), 6.2)
        self.assertEqual(self.tracker.state, 'refined')
        self.assertGreaterEqual(self.tracker.confidence, .75)
        spot = MARKS['right-penalty-spot']
        error = np.hypot(*(field.project(self.tracker.H, [spot])[0]-field.project(self.P, [spot])[0]))
        self.assertLess(error, 3)
        self.assertTrue(any(e.startswith('far touchline: FOUND') for e in self.tracker.evidence), self.tracker.evidence)
        self.assertEqual(self.tracker.sides()['left'], 'off-screen')

    def test_people_standing_outside_the_prediction_lower_the_confidence(self):
        beyond = [(x, broadcast.image_point(self.P, 52.5, 0)[1]-.1) for x in np.linspace(.2, .8, 6)]   # all above the far touchline
        for k in range(3):
            self.tracker.refine([], .2*(k+1), beyond)
        self.assertLess(self.tracker.confidence, MEDIUM)
        self.assertTrue(any(kind == 'doubt' for _, kind, _ in self.tracker.events))

    def test_keyframes_learn_the_size_and_anchor_the_model(self):
        Py, youth = youth_camera()
        marks = field.landmarks(youth)
        pts = [{'name': n, 'x': float(field.project(Py, [marks[n]])[0][0])/W, 'y': float(field.project(Py, [marks[n]])[0][1])/H}
               for n in ('corner-far-left', 'halfway-far', 'halfway-near', 'centre-spot', 'left-penalty-box-front-far', 'left-goal-box-front-far', 'centre-circle-far')]
        keys = Keyframes([], [{'time': 0.0, 'points': pts}], W, H)
        self.assertAlmostEqual(keys.size.length, 90, delta=1.5)
        self.assertAlmostEqual(keys.size.width, 55, delta=1.5)
        keys.update(0.0, np.eye(3), True)
        self.assertTrue(keys.fresh)
        centre = field.project(Py, [marks['centre-spot']])[0]
        x, y = keys.to_pitch([centre[0]/W-.01, centre[1]/H-.1, .02, .1])
        self.assertAlmostEqual(x, .5, places=2)
        self.assertAlmostEqual(y, .5, places=2)


class PropagationTests(unittest.TestCase):
    def test_model_is_filled_backwards_until_a_cut(self):
        P = broadcast.camera()
        shift = np.array([[1, 0, 5], [0, 1, 0], [0, 0, 1]], float)
        log = [{'dt': .2, 'M': shift, 'reliable': True, 'cut': i == 3, 'H': None, 'confidence': 0.0} for i in range(10)]
        log[7]['H'], log[7]['confidence'] = P, 1.0
        filled = propagate(log, W, H, SIZE)
        self.assertTrue(all(h is None for h, _, _ in filled[:3]))          # before the cut nothing is known
        self.assertTrue(all(h is not None and f for h, _, f in filled[3:7]))
        H3, conf3, _ = filled[3]
        centre = field.project(P, [MARKS['centre-spot']])[0]-[20, 0]      # four frames earlier the view was 20 px further left
        x, y = field.to_pitch(H3, SIZE, centre)
        self.assertAlmostEqual(x, .5, places=3)
        self.assertLess(conf3, 1.0)
        self.assertGreaterEqual(conf3, MEDIUM)


class RobustRelevanceTests(unittest.TestCase):
    """Size rules behind scale conditions, and the provisional boundary of a LOW-confidence pitch."""

    def setUp(self):
        self.model = PitchDetector().analyze(pitch_frame(), 0.0)
        # Eight people whose heights grow linearly down the image: a reliable perspective model.
        self.players = [{'box': person_box(.15+.1*i, .45+.05*(i % 6), .03+.1*(.45+.05*(i % 6)-.45)), 'score': .9, 'cls': 'player'} for i in range(8)]

    def test_small_people_get_a_wider_tolerance_than_large_ones(self):
        fit = fit_size_model([d['box'] for d in self.players])
        self.assertTrue(fit.reliable)
        far = {'box': person_box(.5, .45, .33*fit.expected(.45)), 'score': .8, 'cls': 'player'}       # a third of the expected 22 px
        near = {'box': person_box(.5, .7, .33*fit.expected(.7)), 'score': .8, 'cls': 'player'}        # a third of the expected 180 px
        accepted, _, rejected, _ = assess(self.players+[far, near], self.model, FieldFilter(), height_px=H)
        self.assertIn(far['box'], [d['box'] for d in accepted])
        self.assertEqual([r['box'] for r in rejected], [near['box']])

    def test_small_detection_continuing_a_tracked_person_is_kept(self):
        fit = fit_size_model([d['box'] for d in self.players])
        tiny = {'box': person_box(.5, .45, .2*fit.expected(.45)), 'score': .8, 'cls': 'player'}
        accepted, _, rejected, _ = assess(self.players+[tiny], self.model, FieldFilter(), height_px=H)
        self.assertIn(tiny['box'], [r['box'] for r in rejected])
        accepted, _, rejected, _ = assess(self.players+[tiny], self.model, FieldFilter(), live=[tiny['box']], height_px=H)
        kept = next(d for d in accepted if d['box'] == tiny['box'])
        self.assertEqual(kept['kept'], 'continues a tracked person')

    def test_feet_just_beyond_a_low_confidence_boundary_are_held_not_rejected(self):
        outside = {'box': person_box(.4, .94), 'score': .9, 'cls': 'player'}    # beyond the near touchline tolerance
        _, _, rejected, _ = assess([outside], self.model, FieldFilter(), height_px=H)
        self.assertEqual([r['reason'] for r in rejected], ['outside-pitch'])
        self.model.mode, self.model.confidence = 'robust', .2
        accepted, _, rejected, _ = assess([outside], self.model, FieldFilter(), height_px=H)
        self.assertEqual(rejected, [])
        self.assertEqual(accepted[0]['zone'], 'boundary')


class TrackerModeTests(unittest.TestCase):
    """SoccerTracker without the detector: rendered frames and ground-truth detections."""

    @staticmethod
    def run_tracker(frames, rows, times):
        from soccer.local import BotSortTracker
        from soccer.tracker import SoccerTracker, config_from
        tracker = SoccerTracker(W, H, config_from({}), BotSortTracker(5), None)
        observations = [tracker.step(frame, t, r)[0] for frame, r, t in zip(frames, rows, times)]
        return tracker, observations

    def people_rows(self, P, k):
        rows = []
        for i in range(8):
            x, y = 20+10*i, 10+6*(i % 4)+.3*k
            fx, fy = broadcast.image_point(P, x, y)
            h = broadcast.player_px(P, x, y)/broadcast.H
            box = person_box(fx, fy, h)
            rows.append([box[0]*W, box[1]*H, (box[0]+box[2])*W, (box[1]+box[3])*H, .9, 2])
        return np.asarray(rows, np.float32)

    def test_broadcast_footage_stays_automatic_with_the_detectors_own_region(self):
        P = broadcast.camera()
        frames, rows = [], []
        for k in range(10):
            frame = broadcast.render(P)
            r = self.people_rows(P, k)
            for row_ in r:
                draw_person(frame, [row_[0]/W, row_[1]/H, (row_[2]-row_[0])/W, (row_[3]-row_[1])/H], 'A')
            frames.append(frame)
            rows.append(r)
        tracker, observations = self.run_tracker(frames, rows, [k/5 for k in range(10)])
        self.assertTrue(all(o['pitchMode'] == 'automatic' for o in observations))
        self.assertEqual(tracker.pitch_issues(), [])
        own = PitchDetector().analyze(frames[0], 0.0)
        first = next(o['pitch'] for o in observations if o['pitch'])
        self.assertEqual(first['source'], 'auto')
        self.assertEqual(first['level'], 'HIGH')
        self.assertEqual([tuple(p) for p in first['polygon']], [(round(x, 4), round(y, 4)) for x, y in own.polygon])

    def test_faded_lines_ask_for_confirmation_once_per_section(self):
        P = broadcast.camera()
        frames, rows = [], []
        for k in range(25):   # five seconds at five frames per second
            frame = broadcast.render(P, paint=False)
            r = self.people_rows(P, k)
            frames.append(frame)
            rows.append(r)
        tracker, observations = self.run_tracker(frames, rows, [k/5 for k in range(25)])
        modes = [o['pitchMode'] for o in observations]
        self.assertEqual(modes[0], 'robust')
        self.assertIn('assist', modes)
        self.assertTrue(all(o['pitchConfidence'] < MEDIUM for o in observations))
        issues = tracker.pitch_issues()
        self.assertEqual(len(issues), 1)
        self.assertLessEqual(issues[0]['start'], .2)
        self.assertGreaterEqual(issues[0]['end'], 4.5)
        self.assertTrue(issues[0]['start'] <= issues[0]['at'] <= issues[0]['end'])
        self.assertIn('centre-spot', [m['name'] for m in issues[0]['suggested']])

    def test_far_tile_detections_count_only_once_confirmed(self):
        P = broadcast.camera()
        far = broadcast.image_point(P, 52.5, 3)
        h = broadcast.player_px(P, 52.5, 3)/broadcast.H
        box = person_box(far[0], far[1], h)
        provisional = [box[0]*W, box[1]*H, (box[0]+box[2])*W, (box[1]+box[3])*H, .5, 2, 1]
        frames, rows = [], []
        for k in range(4):
            frames.append(broadcast.render(P))
            base = np.c_[self.people_rows(P, k), np.zeros(8)]
            rows.append(np.r_[base, [provisional]].astype(np.float32))
        tracker, observations = self.run_tracker(frames, rows, [k/5 for k in range(4)])
        seen = [any(abs(p['box'][0]-box[0]) < .002 and abs(p['box'][1]-box[1]) < .002 for p in o['people']) for o in observations]
        self.assertEqual(seen[:2], [False, False])
        self.assertTrue(seen[2] or seen[3])
        self.assertEqual(sum(1 for o in observations[:2] for r in o['rejected'] if r['reason'] == 'unconfirmed-far'), 2)


class SuggestionTests(unittest.TestCase):
    def test_halfway_line_ends_are_suggested_from_the_image(self):
        model = PitchDetector().analyze(pitch_frame(), 0.0)
        names = {m['name']: m for m in suggest_landmarks(model)}
        self.assertIn('halfway-far', names)
        self.assertTrue(names['halfway-far']['predicted'])
        self.assertAlmostEqual(names['halfway-far']['x'], .5, delta=.02)
        self.assertLess(names['halfway-far']['y'], names['halfway-near']['y'])
        self.assertFalse(names['centre-spot']['predicted'])


class RefinementAnchorTests(unittest.TestCase):
    """Phase 1: a painted-line refinement anchors the offline registration only on its own evidence; a weak
    match steadies the live model but never restarts a chain at the keyframe's certainty."""

    def setUp(self):
        self.P = broadcast.camera()
        self.tracker = PitchTracker(W, H)
        self.assertTrue(self.tracker.anchor(np.linalg.inv(self.P), 0.0, .1, observability=.8, index=0))
        self.assertEqual(self.tracker.anchored_index, 0)

    def drift(self, seconds):
        wrong = np.array([[1, 0, -6], [0, 1, 3], [0, 0, 1]], float)
        self.tracker.carry(wrong, True, False, seconds, seconds)

    def test_a_single_touchline_steadies_the_model_but_is_not_an_anchor(self):
        self.drift(6.0)
        self.tracker.refine(segments(self.P, ['far touchline']), 6.2)
        self.assertEqual(self.tracker.state, 'refined')
        self.assertGreaterEqual(self.tracker.temporal, .8)                   # the live model is steadied
        self.assertEqual(self.tracker.observability, .8)                      # the chain's certainty is untouched
        r = self.tracker.refinement
        self.assertLess(r['own'], MIN_ANCHOR)                                 # one touchline fixes neither the ends nor the scale
        self.assertEqual(r['anchor'], .8)                                     # (it would be worth the chain's certainty, but it does not anchor)
        self.assertGreater(r['anchor'], r['own'])

    def test_both_marking_families_make_an_anchor_of_their_own(self):
        self.drift(6.0)
        self.tracker.refine(segments(self.P, ['far touchline', 'near touchline', 'halfway line', 'left penalty area front']), 6.2)
        self.assertEqual(self.tracker.state, 'refined')
        r = self.tracker.refinement
        self.assertGreaterEqual(r['own'], MIN_ANCHOR)
        self.assertGreaterEqual(r['anchor'], r['own'])

    def test_an_anchoring_refinement_is_worth_the_chain_it_confirmed(self):
        self.drift(20.0)
        self.tracker.refine(segments(self.P, ['far touchline', 'near touchline', 'halfway line', 'left penalty area front']), 20.2)
        self.assertIsNotNone(self.tracker.refinement)
        self.assertGreaterEqual(self.tracker.refinement['own'], MIN_ANCHOR)
        self.assertEqual(self.tracker.refinement['anchor'], max(self.tracker.refinement['own'], .8))   # the chain's certainty, confirmed
        self.assertEqual(self.tracker.refinement['lines'], 4)


class RollTests(unittest.TestCase):
    """Phase 1: a calibration keyframe just before the segment (or just after it) reaches the segment through
    frames processed for camera motion and pitch evidence only; the offline registration carries it in."""

    def test_keyframes_outside_the_segment_are_carried_in(self):
        import pipeline
        from soccer.local import BotSortTracker
        from soccer.tracker import SoccerTracker, config_from
        P = broadcast.camera()
        names = ['corner-far-left', 'corner-far-right', 'halfway-far', 'halfway-near', 'centre-spot', 'left-penalty-box-front-far']
        points = [normalized(P, n) for n in names]
        config = config_from({'calibrations': [{'time': 0.0, 'points': points}, {'time': 2.0, 'points': points}]})
        tracker = SoccerTracker(W, H, config, BotSortTracker(5), None)
        frame = broadcast.render(P)
        rows = TrackerModeTests.people_rows(TrackerModeTests(), P, 0)
        for t in (0.0, .2, .4):                                   # pre-roll: the keyframe at 0 s, three frames before the start
            tracker.roll(frame, t)
        tracker.offset = len(tracker.frame_log)
        frames = []
        for t in (.6, .8, 1.0, 1.2, 1.4):                         # the segment
            observation, _, _ = tracker.step(frame, t, rows)
            frames.append({'time': t, 'frame': int(t*5), **observation})
        tracker.emitted = len(frames)
        for t in (1.6, 1.8, 2.0, 2.2):                            # post-roll up to the keyframe at 2 s
            tracker.roll(frame, t)
        self.assertEqual(len(tracker.frame_log), 12)
        self.assertEqual([a.index for a in tracker.anchors if a.source == 'landmarks'], [0, 10])
        pre = [e for e in tracker.events if e['kind'] == 'pitch' and e['time'] < .6]
        self.assertTrue(pre and all('outside the segment' in e['message'] for e in pre), pre)
        filled, timeline = pipeline.finalize_pitch(frames, tracker)
        self.assertEqual(len(timeline), 5)
        self.assertTrue(all(f['calibrated'] for f in frames))
        # Both keyframes, from outside the segment, reach the middle frame (a direct registration to the static view may anchor it too).
        self.assertIn(frames[2]['calibration']['source'], ('interpolated', 'reregistered'))
        self.assertTrue(all(f['calibration']['confidence'] >= .5 for f in frames))
        centre = field.project(P, [MARKS['centre-spot']])[0]
        x, y = field.to_pitch(timeline[2]['H'], SIZE, tuple(centre))
        self.assertAlmostEqual(x, .5, places=2)
        self.assertAlmostEqual(y, .5, places=2)
        summary = tracker.registration
        self.assertEqual(summary['calibratedFrames'], 5)
        self.assertEqual(sorted(a['frame'] for a in summary['anchors'] if a['source'] == 'landmarks'), [-3, 7])   # relative to the segment
        self.assertEqual(len(tracker.emitted_log()), 5)


class AdaptiveDetectionTests(unittest.TestCase):
    """Phase 3: confirmed far-tile detections start local tracks, crop detections count only next to the
    prediction they were made for, and feet just past an uncertain far edge continue a tracked person."""

    def test_a_confirmed_far_detection_with_a_low_score_starts_a_local_track(self):
        P = broadcast.camera()
        far = broadcast.image_point(P, 52.5, 3)
        h = broadcast.player_px(P, 52.5, 3)/broadcast.H
        box = person_box(far[0], far[1], h)
        weak = [box[0]*W, box[1]*H, (box[0]+box[2])*W, (box[1]+box[3])*H, .2, 2, 1]   # below BoT-SORT's new-track threshold
        frames, rows = [], []
        for k in range(6):
            frames.append(broadcast.render(P))
            base = np.c_[TrackerModeTests().people_rows(P, k), np.zeros(8)]
            rows.append(np.r_[base, [weak]].astype(np.float32))
        tracker, observations = TrackerModeTests.run_tracker(frames, rows, [k/5 for k in range(6)])
        seen = [[p for p in o['people'] if abs(p['box'][0]-box[0]) < .002 and abs(p['box'][1]-box[1]) < .002] for o in observations]
        self.assertTrue(any(seen[3:]), 'the confirmed far detection is tracked')
        self.assertTrue(all(p['score'] < .3 for s in seen for p in s), 'the person keeps the true detector score')
        self.assertGreaterEqual(tracker.detection['farStarts'], 1)
        self.assertGreaterEqual(tracker.detection['farDetections'], 1)

    def test_a_crop_detection_counts_only_next_to_its_prediction(self):
        P = broadcast.camera()
        frames, rows = [], []
        spot = broadcast.image_point(P, 60, 30)
        h = broadcast.player_px(P, 60, 30)/broadcast.H
        near = person_box(spot[0], spot[1], h)
        elsewhere = person_box(.2, .8, h)
        for k in range(3):
            frames.append(broadcast.render(P))
            base = np.c_[TrackerModeTests().people_rows(P, k), np.zeros(8)]
            crop_rows = [[near[0]*W, near[1]*H, (near[0]+near[2])*W, (near[1]+near[3])*H, .6, 2, 2],
                         [elsewhere[0]*W, elsewhere[1]*H, (elsewhere[0]+elsewhere[2])*W, (elsewhere[1]+elsewhere[3])*H, .6, 2, 2]]
            rows.append(np.r_[base, crop_rows].astype(np.float32))
        from soccer.local import BotSortTracker
        from soccer.tracker import SoccerTracker, config_from
        tracker = SoccerTracker(W, H, config_from({}), BotSortTracker(5), None)
        tracker.crop_predictions = [list(near)]            # the crop was made for a player predicted here
        observations = [tracker.step(frame, k/5, r)[0] for k, (frame, r) in enumerate(zip(frames, rows))]
        last = observations[-1]
        self.assertTrue(any(abs(p['box'][0]-near[0]) < .002 for p in last['people']), 'the detection next to the prediction counts')
        self.assertFalse(any(abs(p['box'][0]-elsewhere[0]) < .002 and abs(p['box'][1]-elsewhere[1]) < .002 for p in last['people']))
        self.assertTrue(any(r['reason'] == 'unconfirmed-crop' for o in observations for r in o['rejected']))
        self.assertEqual(tracker.detection['cropDetections'], 3)

    def test_a_crop_detection_next_to_another_person_is_not_accepted(self):
        # The lost player's prediction is covered by another detected person: the crop finds that person (or a
        # mixture) and must not count, or the lost identity would be carried onto them.
        P = broadcast.camera()
        spot = broadcast.image_point(P, 60, 30)
        h = broadcast.player_px(P, 60, 30)/broadcast.H
        near = person_box(spot[0], spot[1], h)
        other = [near[0]+.4*near[2], near[1], near[2], near[3]]      # a whole-frame person overlapping the prediction
        frames, rows = [], []
        for k in range(3):
            frames.append(broadcast.render(P))
            base = np.c_[TrackerModeTests().people_rows(P, k), np.zeros(8)]
            other_row = [[other[0]*W, other[1]*H, (other[0]+other[2])*W, (other[1]+other[3])*H, .8, 2, 0]]
            crop_row = [[near[0]*W, near[1]*H, (near[0]+near[2])*W, (near[1]+near[3])*H, .6, 2, 2]]
            rows.append(np.r_[base, other_row, crop_row].astype(np.float32))
        from soccer.local import BotSortTracker
        from soccer.tracker import SoccerTracker, config_from
        tracker = SoccerTracker(W, H, config_from({}), BotSortTracker(5), None)
        tracker.crop_predictions = [list(near)]
        observations = [tracker.step(frame, k/5, r)[0] for k, (frame, r) in enumerate(zip(frames, rows))]
        self.assertEqual(tracker.detection['cropDetections'], 0)
        self.assertEqual(tracker.detection['cropRejectedCrowded'], 3)
        self.assertTrue(any('hidden behind' in r.get('detail', '') for o in observations for r in o['rejected']))
        self.assertTrue(any(abs(p['box'][0]-other[0]) < .002 for p in observations[-1]['people']), 'the whole-frame person still counts')

    def test_no_crop_window_for_a_player_lost_next_to_another_tracked_person(self):
        from types import SimpleNamespace
        from soccer.local import BotSortTracker
        from soccer.tracker import SoccerTracker, config_from
        tracker = SoccerTracker(W, H, config_from({}), BotSortTracker(5), None)
        prediction = [600.0, 400.0, 630.0, 455.0]                                   # 30 x 55 px, under the crop size limit
        tracker.local.lost = lambda max_frames: [(1, prediction, 2)]
        tracker.ids.tracks = {1: SimpleNamespace(state='confirmed', last_box=[600/W, 400/H, 30/W, 55/H])}
        self.assertEqual(len(tracker.lost_windows()), 1)                            # alone: a crop window
        tracker.ids.tracks[2] = SimpleNamespace(state='confirmed', last_box=[640/W, 405/H, 30/W, 55/H])   # a neighbour within half a body
        self.assertEqual(tracker.lost_windows(), [])
        self.assertEqual(tracker.detection['cropSkippedCrowded'], 1)
        tracker.ids.tracks[2].last_box = [900/W, 405/H, 30/W, 55/H]                 # far away: the window is back
        self.assertEqual(len(tracker.lost_windows()), 1)

    def test_feet_just_past_an_uncertain_far_edge_continue_a_tracked_person(self):
        from soccer.pitch import PitchModel
        far_line = {'a': (0.0, .30), 'b': (1.0, .30), 'side': 'far', 'support': .6, 'age': 0}
        polygon = [(0.0, .30), (1.0, .30), (1.0, .95), (0.0, .95)]
        config = FieldFilter(True, .02, .005, .05)
        beyond = {'box': [.5, .16, .02, .095], 'score': .6, 'cls': 'player'}    # feet 4.5% of the height beyond the far edge: outside the tolerance
        live = [[.5, .16, .02, .095]]
        predicted = PitchModel(0.0, True, 16/9, polygon=polygon, lines=[], coverage=.5, source='calibration', sides={'far': 'predicted', 'near': 'predicted', 'left': 'off-screen', 'right': 'off-screen'})
        accepted, continuation, rejected, _ = assess([beyond], predicted, config, None, None, live, 720)
        self.assertEqual([d.get('continued') for d in accepted], ['far'])
        self.assertEqual(rejected, [])
        painted = PitchModel(0.0, True, 16/9, polygon=polygon, lines=[far_line], coverage=.5, source='auto')
        accepted, continuation, rejected, _ = assess([beyond], painted, config, None, None, live, 720)
        self.assertEqual(accepted, [])
        self.assertEqual([r['reason'] for r in rejected], ['audience'])     # a painted far touchline keeps its meaning
        accepted, continuation, rejected, _ = assess([beyond], predicted, config, None, None, [], 720)
        self.assertEqual(accepted, [])                                        # nobody tracked there: not continued


if __name__ == '__main__':
    unittest.main()
