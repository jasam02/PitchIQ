"""The small-scene wiring of SoccerTracker and the behaviours behind it (no detector, no GPU): the gate itself
(sustained far band or a low-resolution source, latched once open) and how it reaches the identity manager, the ball
tracker and every emitted frame; the pixel constants of a low-resolution source; the far-tile confirmation by foot
point for tiny rows and their duplicates; the buffered boxes BoT-SORT associates tiny people by, also across a change
of the gate; the debug list of detections the local tracker dropped; the ball rows' source column (7 columns decide
what 6 do); and the wiring other workstreams asked for. Every behaviour is also checked to be exactly the shipped one
while the gate is closed."""
import json
import math
import os
import unittest
from pathlib import Path
from types import SimpleNamespace

# Before anything imports ultralytics: its settings are the pipeline's own file (settings/Ultralytics, as pipeline.py
# sets it), never the user's global one, which two test processes could otherwise race on and rewrite.
(Path(__file__).resolve().parents[1]/'settings'/'Ultralytics').mkdir(parents=True, exist_ok=True)
os.environ['YOLO_CONFIG_DIR'] = str(Path(__file__).resolve().parents[1]/'settings')

import numpy as np

import broadcast
from scene import H, W, draw_person, person_box

try:
    import ultralytics  # noqa: F401
    AVAILABLE = True
except Exception:   # noqa: BLE001 - BoT-SORT needs ultralytics; the tests below are skipped without it
    AVAILABLE = False


def tracker(width=W, height=H, fps=10, debug=True):
    """A SoccerTracker whose identity manager and ball tracker hear the gate (recorded in heard['ids'] / heard['ball'])
    but run with it closed: their own small-scene behaviour belongs to test_identity*.py and test_ball*.py, and this
    module tests the wiring and the tracker's own behaviours only."""
    from soccer.local import BotSortTracker
    from soccer.tracker import SoccerTracker, config_from
    t = SoccerTracker(width, height, config_from({'debug': debug}), BotSortTracker(fps), None, None, fps)
    t.heard = {'ids': [], 'ball': []}
    update, step = t.ids.update, t.ball.step

    def ids_update(ctx):
        t.heard['ids'].append(ctx['small_scene'])
        return update(dict(ctx, small_scene=False))

    def ball_step(*args, **kwargs):
        t.heard['ball'].append(t.ball.small_scene)
        t.ball.small_scene = False
        return step(*args, **kwargs)
    t.ids.update, t.ball.step = ids_update, ball_step
    return t


def px_row(box, score=.8, cls=2, kind=0, width=W, height=H):
    x, y, w, h = box
    return [x*width, y*height, (x+w)*width, (y+h)*height, score, cls, kind]


def tiny_box(cx_px, foot_px, w_px=6, h_px=13):
    return [(cx_px-w_px/2)/W, (foot_px-h_px)/H, w_px/W, h_px/H]


def broadcast_rows(P, k):
    rows = []
    for i in range(8):
        x, y = 38+4*i, 10+6*(i % 4)+.3*k
        fx, fy = broadcast.image_point(P, x, y)
        rows.append(px_row(person_box(fx, fy, broadcast.player_px(P, x, y)/broadcast.H), .9))
    return rows


@unittest.skipUnless(AVAILABLE, 'ultralytics not installed')
class SmallSceneGateTests(unittest.TestCase):
    def scene_states(self, t, active, n, fps=10):
        """The gate frame by frame, as step() updates it, for a far band active on the frames `active` says."""
        t.far_band = lambda: (100, 200) if active(t.frame_index) else None
        states = []
        for k in range(n):
            t.frame_index = k
            t.small_scene = t._update_scene(k/fps)
            states.append(t.small_scene)
        return states

    def test_a_sustained_far_band_opens_the_gate_over_a_full_window_and_it_stays_open(self):
        t = tracker(fps=10)
        states = self.scene_states(t, lambda k: k < 30, 80)
        # 2 s at 10 fps is 20 frames; half of a full window is 10 active frames: the 10th frame opens the gate (the
        # first second of a run cannot) ...
        self.assertEqual(states.index(True), 9)
        # ... and it stays open after the band has gone (the footage is still what it was): no second change.
        self.assertTrue(all(states[9:]))

    def test_a_band_hovering_around_half_the_frames_changes_the_gate_once(self):
        # The band on for 1.5 s, off for 1.5 s: a 2 s window holds 5 to 15 active frames of 20, so a bare threshold at 10
        # changed the gate 13 times in these 20 s.
        states = self.scene_states(tracker(fps=10), lambda k: k % 30 < 15, 200)
        self.assertEqual(sum(1 for a, b in zip(states, states[1:]) if a != b), 1)
        self.assertTrue(states[-1])
        # Never half of a window: never open.
        self.assertFalse(any(self.scene_states(tracker(fps=10), lambda k: k % 30 < 9, 200)))

    def test_a_band_on_a_few_scattered_frames_never_opens_it(self):
        t = tracker(fps=30)
        on = {0, 200, 400, 401, 1000}   # broadcast 0-60 s: the far band fired on 1 of 1800 frames
        t.far_band = lambda: (10, 20) if t.frame_index in on else None
        opened = []
        for k in range(1200):
            t.frame_index = k
            opened.append(t._update_scene(k/30))
        self.assertFalse(any(opened))

    def test_time_going_backwards_starts_a_new_window(self):
        t = tracker(fps=10)
        t.far_band = lambda: (100, 200)
        for k in range(15):
            t.small_scene = t._update_scene(k/10)
        self.assertTrue(t._update_scene(1.5))
        self.assertFalse(t._update_scene(0.0))   # a new sequence: one active frame of a full window, and the latch released
        self.assertEqual(len(t.far_history), 1)

    def test_broadcast_frames_keep_it_closed_and_every_consumer_hears_so(self):
        P = broadcast.camera(span=40)   # far players of 43 px: no far band (a 70 m wide 720p view has 24 px ones, and the band)
        t = tracker(fps=5)
        frames, bands = [], []
        for k in range(12):
            bands.append(t.far_band())
            frames.append(t.step(broadcast.render(P), k/5, np.asarray(broadcast_rows(P, k), np.float32))[0])
        self.assertEqual(bands, [None]*12)
        self.assertEqual([f['profile'] for f in frames], [{'smallScene': False}]*12)
        self.assertEqual(t.heard['ids'], [False]*12)
        self.assertEqual(t.heard['ball'], [False]*12)
        self.assertEqual(t.detection['bufferedBoxes'], 0)
        self.assertNotIn('farDuplicates', t.detection)

    def test_the_gate_reaches_the_identity_context_the_ball_and_the_frame_when_open(self):
        t = tracker(fps=5)
        t._update_scene = lambda time: True
        frame = np.zeros((H, W, 3), np.uint8)
        observation = t.step(frame, 0.0, np.zeros((0, 6), np.float32))[0]
        self.assertEqual(observation['profile'], {'smallScene': True})
        self.assertEqual(t.heard, {'ids': [True], 'ball': [True]})

    def test_a_low_resolution_source_is_always_a_small_scene(self):
        from soccer.tracker import LOW_RES_HEIGHT, pixel_scale
        t = tracker(896, 506, fps=30)
        observation = t.step(np.zeros((506, 896, 3), np.uint8), 0.0, np.zeros((0, 6), np.float32))[0]
        self.assertTrue(observation['profile']['smallScene'])
        self.assertAlmostEqual(t.pixel_scale, 506/1080)
        self.assertEqual(pixel_scale(1080), 1.0)
        self.assertEqual(pixel_scale(LOW_RES_HEIGHT), 1.0)   # 720p and above: every pixel constant as tuned
        self.assertFalse(tracker(1280, 720).step(np.zeros((720, 1280, 3), np.uint8), 0.0, np.zeros((0, 6), np.float32))[0]['profile']['smallScene'])


class _Size:
    """A perspective size model stub: expected height (normalized) = a + b * foot y."""

    def __init__(self, a, b):
        self.a, self.b, self.reliable = a, b, True

    def expected(self, y):
        return self.a+self.b*y


@unittest.skipUnless(AVAILABLE, 'ultralytics not installed')
class LowResolutionTests(unittest.TestCase):
    """Below 720 rows the far band, the crops and the small-person tolerance count in 1080p pixels."""

    def far_band(self, width, height, far_px):
        t = tracker(width, height)
        t.pitch = SimpleNamespace(reliable=True, polygon=[(0, .3), (1, .3), (1, .95), (0, .95)])
        t.size = _Size(far_px/height-.3*.1, .1)   # far_px tall at the far edge (foot y .3), growing towards the camera
        return t.far_band()

    def test_the_far_band_trigger_scales_with_the_source(self):
        self.assertIsNotNone(self.far_band(1920, 1080, 20))   # 20 px far players at 1080p: the band
        self.assertIsNone(self.far_band(1920, 1080, 40))
        # At 506 rows a 20 px far player is a 43 px one at 1080p: no band (an absolute 30 px threshold would fire).
        self.assertIsNone(self.far_band(896, 506, 20))
        self.assertIsNotNone(self.far_band(896, 506, 10))

    def test_crops_on_a_low_resolution_source_are_scaled(self):
        def windows(width, height, h_px):
            t = tracker(width, height)
            box = [400/width, 200/height, .38*h_px/width, h_px/height]
            t.local.lost = lambda max_frames: [(1, [400, 200, 400+.38*h_px, 200+h_px], 2)]
            t.ids.tracks = {1: SimpleNamespace(state='confirmed', last_box=box)}
            return t.lost_windows()
        full = windows(1920, 1080, 40)
        low = windows(896, 506, 20)
        self.assertEqual(full[0][2]-full[0][0], 256)              # CROP_MIN_PX at 1080p
        self.assertAlmostEqual(low[0][2]-low[0][0], 256*506/1080, delta=2)   # the same share of the frame at 506 rows
        self.assertEqual(windows(896, 506, 40), [])              # 40 px at 506 rows is 85 px at 1080p: over CROP_MAX_PX
        self.assertEqual(len(windows(1920, 1080, 40)), 1)

    def test_the_size_filter_hears_the_1080p_equivalent_height(self):
        import soccer.tracker as module
        heard = []
        original = module.assess

        def spy(*args):
            heard.append(args[-1])
            return original(*args)
        module.assess = spy
        try:
            tracker(896, 506).step(np.zeros((506, 896, 3), np.uint8), 0.0, np.zeros((0, 6), np.float32))
            tracker(1920, 1080).step(np.zeros((1080, 1920, 3), np.uint8), 0.0, np.zeros((0, 6), np.float32))
        finally:
            module.assess = original
        self.assertAlmostEqual(heard[0], 1080.0)
        self.assertEqual(heard[1], 1080)


@unittest.skipUnless(AVAILABLE, 'ultralytics not installed')
class FarConfirmationTests(unittest.TestCase):
    """Tiny far-tile rows (under FAR_MIN_PX) repeat by foot point and height in a small scene."""

    JITTER = [(0, 0), (3, -2), (-1, 2), (3, 1), (0, -2), (-3, 0)]   # px: what the detector does to a 6x13 px far player

    def confirmations(self, small, height_px=13, steps=JITTER):
        t = tracker()
        t.small_scene = small
        out = []
        for k, (dx, dy) in enumerate(steps):
            out.append(t._confirm_far(tiny_box(640+dx, 260+dy, 6*height_px/13, height_px), [], k/10))
        return out

    def test_a_jittering_tiny_row_confirms_on_its_third_sighting_only_in_a_small_scene(self):
        self.assertEqual(self.confirmations(True), [False, False, True, True, True, True])
        self.assertFalse(any(self.confirmations(False)))     # IoU > .5 never holds: the shipped rule, unchanged

    def test_rows_of_the_far_tile_size_keep_the_iou_rule_in_a_small_scene(self):
        # A 40 px row stepping 12 px a frame: the same spot by the foot rule (within 20 px), never by IoU > .5.
        steps = [(12*k, 0) for k in range(6)]
        self.assertFalse(any(self.confirmations(True, height_px=40, steps=steps)))
        self.assertTrue(any(self.confirmations(True, height_px=29, steps=[(5*k, 0) for k in range(6)])))   # under FAR_MIN_PX: the foot rule

    def test_a_pending_row_counts_once_per_frame_and_the_nearest_takes_the_repeat(self):
        t = tracker()
        t.small_scene = True
        a, b = tiny_box(600, 260), tiny_box(608, 260)            # two neighbours 8 px apart, both within half a body of a
        self.assertFalse(t._confirm_far(a, [], 0.0))
        self.assertFalse(t._confirm_far(b, [], 0.0))             # same frame: a second pending, not a repeat of the first
        self.assertEqual([p[1] for p in t.far_pending], [1, 1])
        self.assertFalse(t._confirm_far(tiny_box(607, 261), [], .1))
        self.assertEqual([p[1] for p in t.far_pending], [1, 2])  # the nearer one (b) repeated

    def test_a_tiny_row_continues_a_tracked_person_at_the_same_spot(self):
        t = tracker()
        live = [tiny_box(700, 300)]
        moved = tiny_box(703, 302)
        t.small_scene = False
        self.assertFalse(t._confirm_far(moved, live, 0.0))       # IoU .3 is not reached by a 3 px step of a 6 px box
        t.small_scene = True
        self.assertTrue(t._confirm_far(moved, live, 0.0))
        self.assertFalse(t._confirm_far(tiny_box(720, 300), live, 0.1))   # 20 px away: somebody else

    def test_a_tile_row_the_whole_frame_passes_found_too_is_a_duplicate_in_a_small_scene(self):
        whole = tiny_box(640, 260)
        tile = tiny_box(643, 262)
        for small in (False, True):
            t = tracker(fps=5)
            t._update_scene = lambda time, small=small: small
            rows = np.asarray([px_row(whole, .5), px_row(tile, .4, kind=1)], np.float32)
            observation = t.step(np.zeros((H, W, 3), np.uint8), 0.0, rows)[0]
            reasons = [r['reason'] for r in observation['rejected']]
            if small:
                self.assertEqual(t.detection['farDuplicates'], 1)
                self.assertNotIn('unconfirmed-far', reasons)
            else:
                self.assertNotIn('farDuplicates', t.detection)
                self.assertIn('unconfirmed-far', reasons)

    def people_and_tracks(self, small, frames):
        """step() over `frames` (each a list of pixel rows) with the gate forced: people per frame, and the local tracks
        BoT-SORT matched."""
        t = tracker(fps=5)
        t._update_scene = lambda time: small
        tracks = set()
        update = t.local.update

        def spy(rows, *args):
            matches, gone = update(rows, *args)
            tracks.update(local_id for local_id, _ in matches)
            return matches, gone
        t.local.update = spy
        people = [len(t.step(np.zeros((H, W, 3), np.uint8), k/5, np.asarray(rows, np.float32))[0]['people']) for k, rows in enumerate(frames)]
        return t, people, tracks

    def test_a_weak_whole_frame_row_gives_way_to_the_confirmed_tile_row_of_the_same_person(self):
        # The whole-frame passes see a far player at .20 (no local track can start from that), the tiles at .50; the two
        # boxes overlap at IoU .27, so pipeline.merge_far keeps the tile row. Dropping the tile row as a duplicate lost
        # the player on every frame of a small scene (0 people-frames against 12 with the gate closed).
        frames = [[px_row(tiny_box(640+k, 260), .20), px_row(tiny_box(643+k, 262), .50, kind=1)] for k in range(15)]
        _, closed, _ = self.people_and_tracks(False, frames)
        t, people, tracks = self.people_and_tracks(True, frames)
        self.assertEqual(len(tracks), 1)
        self.assertGreaterEqual(sum(people), sum(closed))
        self.assertGreater(sum(closed), 0)
        self.assertNotIn('farDuplicates', t.detection)
        self.assertEqual(t.detection['farSuperseded'], 13)   # from the tile row's third sighting (its confirmation) on

    def test_a_tracked_person_seen_weakly_whole_and_by_the_tiles_keeps_one_local_track(self):
        # Tracked from strong whole-frame rows, then seen at .35 by the whole-frame passes (a score that continues a track
        # but starts none) and at .50 by the tiles: had both rows reached BoT-SORT, the one left unmatched could start
        # a second track (the tile row's .50); the confirmed tile row replaces the weak one instead.
        frames = [[px_row(tiny_box(640+k, 260), .8)] for k in range(6)]
        frames += [[px_row(tiny_box(640+k, 260), .35), px_row(tiny_box(643+k, 262), .50, kind=1)] for k in range(6, 20)]
        t, people, tracks = self.people_and_tracks(True, frames)
        self.assertEqual(len(tracks), 1)
        self.assertGreater(t.detection['farSuperseded'], 0)
        self.assertEqual(people[6:], [1]*14)


@unittest.skipUnless(AVAILABLE, 'ultralytics not installed')
class BufferedAssociationTests(unittest.TestCase):
    def test_tiny_boxes_are_inflated_about_their_centre_and_others_pass_unchanged(self):
        from soccer.tracker import BUFFER_MIN_H, BUFFER_MIN_W, _buffered
        x1, y1, x2, y2 = _buffered([100.0, 200.0, 106.0, 213.0])
        self.assertAlmostEqual(y2-y1, BUFFER_MIN_H)
        self.assertAlmostEqual(x2-x1, BUFFER_MIN_W)
        self.assertAlmostEqual((x1+x2)/2, 103.0)
        self.assertAlmostEqual((y1+y2)/2, 206.5)
        big = [100.0, 200.0, 120.0, 260.0]
        self.assertIs(_buffered(big), big)
        tall_thin = _buffered([100.0, 200.0, 104.0, 240.0])      # tall enough, too thin: only the width grows
        self.assertAlmostEqual(tall_thin[3]-tall_thin[1], 40.0)
        self.assertAlmostEqual(tall_thin[2]-tall_thin[0], BUFFER_MIN_W)

    def track_ids(self, buffered, seed):
        """Four 6x13 px people with one pixel of detector jitter for 60 frames: local tracks per person."""
        from soccer.local import BotSortTracker
        from soccer.tracker import _buffered
        rng = np.random.default_rng(seed)
        local = BotSortTracker(25)
        frame = np.zeros((1080, 1920, 3), np.uint8)
        ids = [set() for _ in range(4)]
        for k in range(60):
            rows = []
            for p in range(4):
                cx, foot = 300+300*p+k+rng.normal(0, 1.0), 300+rng.normal(0, 1.0)
                h, w = 13*np.exp(rng.normal(0, .12)), 6*np.exp(rng.normal(0, .12))
                box = [cx-w/2, foot-h, cx+w/2, foot]
                rows.append((_buffered(box) if buffered else box)+[.6, 2])
            matches, _ = local.update(np.asarray(rows, np.float32), frame)
            for local_id, index in matches:
                ids[index].add(local_id)
        return sum(len(s) for s in ids)

    def test_buffered_boxes_keep_one_track_per_tiny_person(self):
        # Measured over ten seeds: 13 tracks for the four people on average with the plain boxes, 4 with the buffered.
        for seed in range(3):
            self.assertEqual(self.track_ids(True, seed), 4)
            self.assertGreater(self.track_ids(False, seed), 6)

    def local_rows(self, small):
        t = tracker(fps=5)
        t._update_scene = lambda time: small
        seen = []
        update = t.local.update
        t.local.update = lambda rows, *a: seen.append(np.array(rows)) or update(rows, *a)
        tiny, big = tiny_box(640, 300), [.1, .5, .02, .1]
        rows = np.asarray([px_row(tiny, .8), px_row(big, .8)], np.float32)
        people = [t.step(np.zeros((H, W, 3), np.uint8), k/5, rows)[0]['people'] for k in range(3)]
        return t, seen, people, tiny, big

    def test_only_botsort_sees_the_buffered_box_and_only_in_a_small_scene(self):
        t, seen, people, tiny, big = self.local_rows(True)
        self.assertAlmostEqual(float(seen[0][0, 3]-seen[0][0, 1]), 24.0, places=3)
        self.assertAlmostEqual(float(seen[0][1, 3]-seen[0][1, 1]), big[3]*H, places=3)
        boxes = [p['box'] for p in people[-1]]
        self.assertTrue(any(abs(b[3]-tiny[3]) < 1e-4 and abs(b[1]-tiny[1]) < 1e-4 for b in boxes), boxes)   # the true box
        # The tracker knows which local track BoT-SORT holds by a buffered box: the tiny person's, not the big one's.
        self.assertEqual(t.buffered_ids, {p['track'] for p in people[-1] if abs(p['box'][3]-tiny[3]) < 1e-4})
        t, seen, _, tiny, _ = self.local_rows(False)
        self.assertAlmostEqual(float(seen[0][0, 3]-seen[0][0, 1]), tiny[3]*H, places=3)
        self.assertEqual(t.buffered_ids, set())

    def test_a_lost_buffered_track_s_crop_prediction_has_the_person_s_own_size(self):
        from soccer.geometry import iou
        from soccer.tracker import CROP_ACCEPT_IOU
        t = tracker()
        last = tiny_box(640, 300)
        t.ids.tracks = {1: SimpleNamespace(state='confirmed', last_box=last)}
        t.local.lost = lambda max_frames: [(1, [634.0, 276.0, 646.0, 300.0], 2)]   # the inflated 12x24 prediction
        t.buffered_ids = {1}
        # Also after the gate has closed (on a new sequence): the track was lost while BoT-SORT held it buffered, and the
        # crop's true-size detection would overlap the inflated prediction at IoU .27, under CROP_ACCEPT_IOU.
        for small in (True, False):
            t.small_scene = small
            t.lost_windows()
            x, y, w, h = t.crop_predictions[0]
            self.assertAlmostEqual(w*W, 6.0, places=3)
            self.assertAlmostEqual(h*H, 13.0, places=3)
            self.assertAlmostEqual((x+w/2)*W, 640.0, places=3)
            self.assertGreaterEqual(iou(last, t.crop_predictions[0]), CROP_ACCEPT_IOU)
        # A track BoT-SORT never held by a buffered box keeps its prediction, whatever the gate.
        t.buffered_ids = set()
        t.small_scene = True
        t.lost_windows()
        self.assertAlmostEqual(t.crop_predictions[0][3]*H, 24.0, places=3)

    def tracks_through_a_change(self, opened_at, closed_at=None, rebox=True, seed=0):
        """Four 6x13 px people with half a pixel of detector jitter for 90 frames at 25 fps through step(), the gate open
        from frame `opened_at` (until `closed_at`): local tracks per person, summed (4 is one each)."""
        t = tracker(fps=25)
        t._update_scene = lambda time: opened_at/25-1e-9 <= time < (math.inf if closed_at is None else closed_at/25-1e-9)
        if not rebox:
            t._scene_changed = lambda opened: None
        ids = [set() for _ in range(4)]
        update = t.local.update

        def spy(rows, *args):
            matches, gone = update(rows, *args)
            for local_id, index in matches:
                ids[int(round(((rows[index, 0]+rows[index, 2])/2-300)/250))].add(local_id)
            return matches, gone
        t.local.update = spy
        rng = np.random.default_rng(seed)
        frame = np.zeros((H, W, 3), np.uint8)
        for k in range(90):
            rows = []
            for p in range(4):
                cx, foot = 300+250*p+k+rng.normal(0, .5), 300+rng.normal(0, .5)
                rows.append([cx-3, foot-13, cx+3, foot, .6, 2])
            t.step(frame, k/25, np.asarray(rows, np.float32))
        return sum(len(s) for s in ids)

    def test_a_change_of_the_gate_keeps_every_tiny_person_s_local_track(self):
        # The live tracks are re-boxed into the new convention before the frame's update; without that, the 12x24 px
        # tracks and the 6x13 px detections (or the reverse) overlap at IoU .27 and every track restarted: 8 for 4 people.
        for seed in range(2):
            self.assertEqual(self.tracks_through_a_change(30, seed=seed), 4)
            self.assertEqual(self.tracks_through_a_change(0, 30, seed=seed), 4)   # closing (a new sequence) too
        self.assertEqual(self.tracks_through_a_change(30, rebox=False), 8)

    def test_local_tracks_take_a_new_box_about_their_kalman_state(self):
        from soccer.local import BotSortTracker
        from soccer.tracker import _buffered
        local = BotSortTracker(25)
        frame = np.zeros((720, 1280, 3), np.uint8)
        for k in range(3):
            matches, _ = local.update(np.asarray([[100+k, 200, 106+k, 213, .9, 2], [400, 300, 440, 400, .9, 2]], np.float32), frame)
        before = {local_id: [float(v) for v in t.xyxy] for t in local.tracker.tracked_stracks for local_id in [local.ids[(0, t.track_id)]]}

        def size(local_id, xyxy):
            box = _buffered(xyxy)
            return None if box is xyxy else box
        changed = local.rebox(size)
        tiny = matches[0][0]
        self.assertEqual(changed, [tiny])   # the big person's box was kept
        after = {local.ids[(0, t.track_id)]: [float(v) for v in t.xyxy] for t in local.tracker.tracked_stracks}
        x1, y1, x2, y2 = after[tiny]
        self.assertAlmostEqual(y2-y1, 24.0, places=4)
        self.assertAlmostEqual((x1+x2)/2, (before[tiny][0]+before[tiny][2])/2, places=4)
        self.assertEqual(after[matches[1][0]], before[matches[1][0]])


@unittest.skipUnless(AVAILABLE, 'ultralytics not installed')
class DroppedDetectionTests(unittest.TestCase):
    def test_the_local_tracker_reports_rows_it_neither_matched_nor_started(self):
        from soccer.local import BotSortTracker
        local = BotSortTracker(25)
        frame = np.zeros((720, 1280, 3), np.uint8)
        person = [100, 100, 120, 160, .9, 2]
        local.update(np.asarray([person], np.float32), frame)
        rows = np.asarray([[101, 100, 121, 160, .9, 2],     # continues the track
                           [600, 300, 620, 360, .35, 2],    # alone, scored .30-.40: cannot start a track
                           [900, 300, 920, 360, .55, 2],    # alone, scored .40+: starts a (not yet confirmed) track
                           [104, 104, 118, 150, .2, 2]],    # low score, its IoU with the matched track under .5
                          np.float32)
        matches, _ = local.update(rows, frame)
        self.assertEqual([i for _, i in matches], [0])
        self.assertEqual([i for i, _ in local.unmatched], [1, 3])
        overlap = dict(local.unmatched)
        self.assertEqual(overlap[1], 0.0)
        self.assertGreater(overlap[3], .3)

    def test_the_frame_lists_them_in_debug_only_and_counts_them(self):
        rows = np.asarray([px_row([.1, .5, .02, .1], .9), px_row([.5, .5, .02, .1], .35)], np.float32)
        t = tracker(fps=5)
        frames = [t.step(np.zeros((H, W, 3), np.uint8), k/5, rows)[0] for k in range(3)]
        dropped = frames[-1]['dropped']
        self.assertEqual(len(dropped), 1)
        self.assertEqual(dropped[0]['bin'], '.30-.40')
        self.assertAlmostEqual(dropped[0]['px'], .1*H, places=1)
        self.assertEqual(t.detection['trackerDroppedNoStart'], 3)
        quiet = tracker(fps=5, debug=False)
        self.assertNotIn('dropped', quiet.step(np.zeros((H, W, 3), np.uint8), 0.0, rows)[0])

    def test_the_list_is_capped_with_the_highest_scores_first(self):
        from soccer.tracker import DROPPED_PER_FRAME
        t = tracker(fps=5)
        rows = np.asarray([px_row([.03*i, .5, .01, .05], .30+.005*i) for i in range(20)], np.float32)
        dropped = t.step(np.zeros((H, W, 3), np.uint8), 0.0, rows)[0]['dropped']
        self.assertEqual(len(dropped), DROPPED_PER_FRAME)
        self.assertEqual([d['score'] for d in dropped], sorted((d['score'] for d in dropped), reverse=True))


def _without_candidate_sources(observation):
    """What a frame's ball rows can reach, as JSON, with the ball candidates' 'source' fields removed (the one thing the
    source column may change, debug): the ball, its candidates and diagnostics, and the people and detections of the
    frame. The pitch fields are left out: they are decided before the ball tracker runs, and the centre circle's inlier
    count is not reproducible between two runs of one process on this scene (the snapshot of the shipped code too)."""
    out = {k: observation.get(k) for k in ('people', 'ball', 'ballDebug', 'rejected', 'dropped', 'profile')}
    out['ballCandidates'] = [{k: v for k, v in c.items() if k != 'source'} for c in observation['ballCandidates']]
    return json.dumps(out, sort_keys=True, default=lambda v: v.item() if isinstance(v, np.generic) else v.tolist())


@unittest.skipUnless(AVAILABLE, 'ultralytics not installed')
class BallRowColumnsTests(unittest.TestCase):
    """pipeline.ball_candidates hands the tracker (M, 7) ball rows on every frame, the source in the last column: the
    one data change on every broadcast frame. The source is diagnostic: through SoccerTracker and BallTracker the
    frames must be what (M, 6) rows give, apart from the candidates' 'source' fields."""

    def run_scene(self, columns, small=False):
        """40 frames of a broadcast-like scene: ten players and three ball candidates a frame, one from each source. The
        real identity manager and ball tracker hear the gate (forced open when small)."""
        from soccer.local import BotSortTracker
        from soccer.tracker import SoccerTracker, config_from
        P = broadcast.camera(span=40)   # far players of 43 px: no far band, as on broadcast footage
        t = SoccerTracker(W, H, config_from({}), BotSortTracker(5), None, None, 5)
        if small:
            t._update_scene = lambda time: True
        rng = np.random.default_rng(4)
        out = []
        for k in range(40):
            frame = broadcast.render(P)
            rows = []
            for i in range(10):
                x, y = 36+3.5*i+.2*k, 8+5*(i % 5)+.2*k
                fx, fy = broadcast.image_point(P, x, y)
                b = person_box(fx, fy, broadcast.player_px(P, x, y)/broadcast.H)
                draw_person(frame, b, 'A' if i % 2 else 'B')
                rows.append([b[0]*W, b[1]*H, (b[0]+b[2])*W, (b[1]+b[3])*H, .9, 2])
            bx = 300+15*k
            # The ball from the whole-frame passes (source 0), a flickering speck from the tiles (1), a still one from the crop (2).
            balls = [[bx, 500, bx+10, 510, .8, 0, 0], [900+rng.normal(), 300, 908, 308, .3, 0, 1], [200, 600, 207, 607, .25, 0, 2]]
            if k % 9 == 4:
                balls = balls[1:]   # the ball unseen on some frames
            observation = t.step(frame, k/5, np.asarray(rows, np.float32), np.asarray(balls, np.float32)[:, :columns])[0]
            out.append(observation)
        return out

    def test_seven_column_ball_rows_make_the_decisions_of_six_column_ones(self):
        six, seven = self.run_scene(6), self.run_scene(7)
        self.assertFalse(any(f['profile']['smallScene'] for f in six+seven))   # the broadcast case: the gate closed
        self.assertTrue(any(f['ball']['state'] == 'TRACKED' for f in six))
        # The tiles' and the crop's candidates say so (the ball tracker names no source for the whole-frame detector's).
        self.assertLessEqual({'crop', 'tiles'}, {c['source'] for f in seven for c in f['ballCandidates'] if 'source' in c})
        for k, (a, b) in enumerate(zip(six, seven)):
            self.assertEqual(a['ball'], b['ball'], k)   # the frame's ball dict, which the review compares, has no source
            self.assertEqual(_without_candidate_sources(a), _without_candidate_sources(b), k)

    def test_also_in_a_small_scene(self):
        # The plan's interface: the source is diagnostic for the ball tracker's small-scene rules too (only its own
        # proposals carry a source that matters, and those it marks itself).
        six, seven = self.run_scene(6, small=True), self.run_scene(7, small=True)
        self.assertTrue(all(f['profile']['smallScene'] for f in six+seven))
        self.assertTrue(any(f['ball']['state'] == 'TRACKED' for f in six))
        for k, (a, b) in enumerate(zip(six, seven)):
            self.assertEqual(_without_candidate_sources(a), _without_candidate_sources(b), k)


@unittest.skipUnless(AVAILABLE, 'ultralytics not installed')
class RequestedWiringTests(unittest.TestCase):
    """Wiring other workstreams asked for in soccer/tracker.py, each inert while its gate is closed."""

    def test_a_run_without_keyframes_tries_the_midfield_anchor_on_every_tick_without_a_model(self):
        frame = np.zeros((H, W, 3), np.uint8)   # no paint: the shipped search finds no centre circle, so it never tried
        for auto in (False, True):
            t = tracker(fps=5)
            t.auto_landmarks = auto
            calls = []
            t._midfield_anchor = lambda model, halfway, time, boxes=(): calls.append(time)
            for k in range(3):
                t.step(frame, k/5, np.zeros((0, 6), np.float32))
            self.assertEqual(calls, [0.0, .2, .4] if auto else [])

    def test_the_tick_s_pitch_carries_the_image_area_on_small_scenes_only(self):
        frame = np.zeros((H, W, 3), np.uint8)
        frame[:, int(.12*W):] = (60, 140, 55)   # grass, and a dewarp's black strip along the left edge (12% of the frame)
        for small in (False, True):
            t = tracker(fps=5)
            t._update_scene = lambda time, small=small: small
            pitch = t.step(frame, 0.0, np.zeros((0, 6), np.float32))[0]['pitch']
            self.assertEqual('imageArea' in pitch, small)


if __name__ == '__main__':
    unittest.main()
