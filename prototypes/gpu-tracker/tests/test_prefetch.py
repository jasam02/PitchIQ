"""The prefetch worker's batched whole-frame detection: the frames of a segment go through the detector
several at a time, and what comes out per frame (rows, order, roll frames, tile scans) is what the serial
loop computed. A stub detector stands in for the CUDA model: it answers a list of images with one result
per image, derived from that image alone, so a batch can only be right when every frame's rows come from
its own image in its own position."""
import os
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import cv2
    import torch
    import pipeline
    AVAILABLE = True
except Exception:   # noqa: BLE001 - torch / ultralytics missing: the tests below are skipped
    AVAILABLE = False

FPS = 25
SIZE = (64, 48)   # width, height


class StubBoxes:
    def __init__(self, rows):
        self.data = torch.tensor(np.asarray(rows, np.float32).reshape(-1, 6))


class StubResult:
    def __init__(self, rows):
        self.boxes = StubBoxes(rows)


class StubDetector:
    """Rows from the image content: the grey level of the top-left pixel places two people (one of them seen by
    both size passes, so the joint suppression has work to do) and a ball; the size pass adds its own row."""
    names = {0: 'ball', 1: 'goalkeeper', 2: 'player', 3: 'referee'}

    def __init__(self):
        self.calls = []   # (number of images, imgsz) per predict call

    def predict(self, source, imgsz=640, **kwargs):
        images = source if isinstance(source, (list, tuple)) else [source]
        self.calls.append((len(images), imgsz))
        out = []
        for image in images:
            v = float(image[0, 0, 0])
            rows = [[v, v, v+10, v+10, .9 if imgsz > 960 else .7, 2],       # the same person on both passes: one survives
                    [v+20, 2, v+30, 14, .6, 2 if imgsz > 960 else 1],       # a role disagreement across scales: suppressed jointly
                    [1, 1, 4, 4, .5+imgsz/10000, 0]]                        # the ball, a different score per pass
            if imgsz > 960:
                rows.append([30, 30, 40, 45, .55, 3])
            out.append(StubResult(rows))
        return out


def write_video(path, frames):
    for suffix, codec in (('.mp4', 'mp4v'), ('.avi', 'MJPG')):
        target = path+suffix
        writer = cv2.VideoWriter(target, cv2.VideoWriter_fourcc(*codec), FPS, SIZE)
        if writer.isOpened():
            for frame in frames:
                writer.write(frame)
            writer.release()
            return target
        writer.release()
    return None


def read_frames(path):
    capture = cv2.VideoCapture(path)
    out = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        out.append(frame)
    capture.release()
    return out


@unittest.skipUnless(AVAILABLE, 'torch / ultralytics not installed')
class BatchedDetectionTests(unittest.TestCase):
    def frames(self, n=11):
        rng = np.random.default_rng(3)
        out = []
        for i in range(n):
            frame = np.full((SIZE[1], SIZE[0], 3), 40+12*i, np.uint8)
            frame[8:, 8:] = rng.integers(0, 255, (SIZE[1]-8, SIZE[0]-8, 3), dtype=np.uint8)
            out.append(frame)
        return out

    def test_a_batch_gives_every_frame_the_rows_of_its_own_serial_detection(self):
        frames = self.frames()
        single = StubDetector()
        expected = [pipeline.detect(single, f, 1536) for f in frames]
        batched = StubDetector()
        rows = pipeline.detect_batch(batched, frames, 1536)
        self.assertEqual(len(rows), len(frames))
        for a, b in zip(expected, rows):
            self.assertTrue(np.array_equal(a, b), (a, b))
        self.assertEqual(batched.calls, [(len(frames), 960), (len(frames), 1536)])   # one call per size pass for all frames
        self.assertEqual(single.calls, [(1, 960), (1, 1536)]*len(frames))
        # The suppression worked per frame: one of each scale-duplicate pair survives (the role disagreement
        # too, whichever class the suppression keeps), the better-scored ball among them.
        for r in rows:
            self.assertEqual(len(r), 4, r)
            self.assertEqual(int((r[:, 5] == 0).sum()), 1)
            self.assertIn(sorted(r[:, 5].tolist()), ([0.0, 1.0, 2.0, 3.0], [0.0, 2.0, 2.0, 3.0]))
            self.assertAlmostEqual(float(r[r[:, 5] == 0][0, 4]), .5+1536/10000, places=5)
        self.assertEqual(pipeline.detect_batch(StubDetector(), [], 1536), [])

    def test_the_prefetch_worker_hands_the_frames_out_in_order_with_the_serial_results(self):
        frames = self.frames(11)
        with tempfile.TemporaryDirectory() as folder:
            path = write_video(os.path.join(folder, 'clip'), frames)
            if path is None:
                self.skipTest('no video codec available')
            decoded = read_frames(path)
            self.assertEqual(len(decoded), len(frames))
            reference = StubDetector()
            expected = {i: pipeline.detect(reference, f, 1536) for i, f in enumerate(decoded)}
            tiles = {i: pipeline.ball_tiles(reference, f) for i, f in enumerate(decoded)}
            original = pipeline.YOLO
            pipeline.YOLO = lambda path: StubDetector()
            try:
                for batch in (4, 1, 16):
                    # Frames 0-2 are a pre-roll (camera motion only), 3-10 the segment: 8 frames, two groups of four.
                    ahead = pipeline.Prefetch(path, 'unused.pt', 0, FPS, 3/FPS, 1.0, None, 1536, True, lambda: False, batch=batch)
                    items = list(ahead)
                    ahead.close()
                    self.assertEqual([(kind, index) for kind, index, *_ in items], [('roll', i) for i in range(3)]+[('frame', i) for i in range(3, 11)])
                    for kind, index, stamp, frame, rows, tile_rows, spent in items:
                        self.assertAlmostEqual(stamp, index/FPS, places=3)
                        self.assertTrue(np.array_equal(frame, decoded[index]))
                        if kind == 'roll':
                            self.assertIsNone(rows)
                            continue
                        self.assertTrue(np.array_equal(rows, expected[index]), index)
                        if index % 2 == 0:
                            self.assertEqual(tile_rows, tiles[index])
                            self.assertGreater(spent['ballTiles'], 0.0)
                        else:
                            self.assertIsNone(tile_rows)
                        self.assertGreater(spent['detect'], 0.0)
            finally:
                pipeline.YOLO = original

    def test_the_tile_scans_follow_the_shared_lock_flag_with_the_hand_over_s_lag(self):
        frames = self.frames(12)
        with tempfile.TemporaryDirectory() as folder:
            path = write_video(os.path.join(folder, 'clip'), frames)
            if path is None:
                self.skipTest('no video codec available')
            original = pipeline.YOLO
            pipeline.YOLO = lambda path: StubDetector()
            try:
                # One frame handed over at a time: the worker decides frame i+1's scan after the consumer finished
                # frame i. Not locked: every other frame. Locked (from frame 4 on, as the consumer says): every 4th.
                ahead = pipeline.Prefetch(path, 'unused.pt', 0, FPS, 0.0, 1.0, None, 1536, True, lambda: False, depth=1, batch=1, scan_every=4/FPS)
                scanned = []
                for kind, index, stamp, frame, rows, tile_rows, spent in ahead:
                    if tile_rows is not None:
                        scanned.append(index)
                    ahead.ball_locked = index >= 4
                ahead.close()
                self.assertEqual(scanned, [0, 2, 4, 8])
                self.assertEqual(ahead.scans, 4)
                # Scans off: nothing is scanned whatever the flag says.
                ahead = pipeline.Prefetch(path, 'unused.pt', 0, FPS, 0.0, 1.0, None, 1536, False, lambda: False, depth=1, batch=1)
                self.assertTrue(all(tile_rows is None for _, _, _, _, _, tile_rows, _ in ahead))
                ahead.close()
                self.assertEqual(ahead.scans, 0)
                # A consumer that stops early (a cancelled run) closes the hand-over; the worker ends without waiting for a slot.
                ahead = pipeline.Prefetch(path, 'unused.pt', 0, FPS, 0.0, 1.0, None, 1536, True, lambda: False, depth=1, batch=4)
                for n, _ in enumerate(ahead):
                    if n == 1:
                        break
                ahead.close()
                self.assertFalse(ahead.thread.is_alive())
                # A detector instance handed in (the process's kept worker detector) is used; none is loaded.
                def refuse(path):
                    raise AssertionError('the worker loaded a detector of its own')
                pipeline.YOLO = refuse
                own = StubDetector()
                ahead = pipeline.Prefetch(path, 'unused.pt', 0, FPS, 0.0, 1.0, None, 1536, False, lambda: False, batch=4, model=own)
                items = list(ahead)
                ahead.close()
                self.assertEqual(len(items), 12)
                self.assertEqual(own.calls, [(4, 960), (4, 1536)]*3)
            finally:
                pipeline.YOLO = original


class ClassAwareDetector(StubDetector):
    """The stub detector, honouring `classes`, recording per call the input size, the classes and which frames (by the
    grey level of their top-left pixel) it saw. A NEAR_PASS call adds a person of its own (a near player the other
    passes miss) and a duplicate of the first person (suppressed jointly)."""

    def __init__(self):
        super().__init__()
        self.seen = []

    def predict(self, source, imgsz=640, classes=None, **kwargs):
        images = source if isinstance(source, (list, tuple)) else [source]
        self.seen.append((imgsz, classes, [int(image[0, 0, 0]) for image in images]))
        out = []
        for result, image in zip(super().predict(source, imgsz=imgsz, **kwargs), images):
            rows = result.boxes.data.numpy()
            if imgsz == pipeline.NEAR_PASS:
                v = float(image[0, 0, 0])
                rows = np.r_[rows, [[v+1, v, v+11, v+10, .8, 2], [40, 2, 50, 30, .5, 2]]]
            if classes is not None:
                rows = rows[np.isin(rows[:, 5], classes)]
            out.append(StubResult(rows))
        return out


@unittest.skipUnless(AVAILABLE, 'torch / ultralytics not installed')
class SmallSceneDetectionTests(unittest.TestCase):
    """The near pass on small scenes, the scaled ball and far tiles of a low-resolution source and the ball
    candidates' source column; each identical to the shipped behaviour while its gate is closed."""

    def frames(self, n=12):
        return BatchedDetectionTests.frames(self, n)

    def test_the_near_pass_only_fills_in_tall_people_the_other_passes_missed(self):
        frames = self.frames(3)
        plain = pipeline.detect_batch(ClassAwareDetector(), frames, 1536)
        reference = pipeline.detect_batch(StubDetector(), frames, 1536)
        for a, b in zip(plain, reference):
            self.assertTrue(np.array_equal(a, b))                       # near=False: the shipped detections
        detector = ClassAwareDetector()
        near = pipeline.detect_batch(detector, frames, 1536, near=True, near_min=20)
        self.assertEqual([(size, classes) for size, classes, _ in detector.seen], [(960, None), (1536, None), (640, [1, 2, 3])])
        for a, b in zip(plain, near):
            # Every row of the other passes is kept as it was (the near pass's higher-scored duplicate of the first person
            # replaces nothing), and only the 28 px near player is added.
            self.assertTrue(np.array_equal(a, b[:len(a)]))
            self.assertEqual(len(b), len(a)+1)
            self.assertTrue(np.allclose(b[-1][:4], [40, 2, 50, 30]))
        for a, b in zip(plain, pipeline.detect_batch(ClassAwareDetector(), frames, 1536, near=True, near_min=30)):
            self.assertTrue(np.array_equal(a, b))                       # under near_min: nothing added
        detector = ClassAwareDetector()
        pipeline.detect_batch(detector, frames, 640, near=True)          # a 640 detail pass already looks at that size
        self.assertEqual([size for size, _, _ in detector.seen], [640, 960])

    def test_a_near_row_partly_over_a_detected_person_is_not_added(self):
        rows = np.array([[100, 100, 120, 160, .8, 2], [5, 5, 8, 8, .5, 0]], np.float32)
        extra = np.array([[110, 100, 140, 170, .9, 2],     # IoU .25 with the person (the joint suppression kept both): covers .5 of it
                          [300, 100, 330, 170, .4, 2],     # nobody there: added
                          [4, 4, 9, 70, .4, 2]], np.float32)   # over the ball only: a person row is what counts
        out = pipeline._fill_in(rows, extra, 60)
        self.assertTrue(np.array_equal(out[:2], rows))
        self.assertEqual([r[:4].tolist() for r in out[2:]], [[300, 100, 330, 170], [4, 4, 9, 70]])
        self.assertTrue(np.array_equal(pipeline._fill_in(rows, np.zeros((0, 6), np.float32), 60), rows))
        self.assertEqual(len(pipeline._fill_in(np.zeros((0, 6), np.float32), extra, 60)), 3)

    def near_frames(self, delay):
        """Run the prefetch over a 12-frame clip with a consumer that reports a small scene from frame 4 on; the
        frames whose detection included the near pass, by index."""
        import time as clock
        frames = self.frames(12)
        with tempfile.TemporaryDirectory() as folder:
            path = write_video(os.path.join(folder, 'clip'), frames)
            if path is None:
                self.skipTest('no video codec available')
            decoded = read_frames(path)
            grey = [int(f[0, 0, 0]) for f in decoded]
            detector = ClassAwareDetector()
            ahead = pipeline.Prefetch(path, 'unused.pt', 0, FPS, 0.0, 1.0, None, 1536, False, lambda: False, depth=1, batch=1,
                                      model=detector, near=True)
            for kind, index, *_ in ahead:
                clock.sleep(delay)
                ahead.note_scene(index, index >= 4)
            ahead.close()
        near = [grey.index(v[0]) for size, _, v in detector.seen if size == pipeline.NEAR_PASS]
        return near, ahead.near_frames

    def test_the_near_pass_follows_the_small_scene_word_of_a_fixed_earlier_frame(self):
        # depth 1: frame i reads the word of frame i-2, so the small scene from frame 4 on brings the pass from frame 6;
        # a slow consumer and a fast one give the same frames (the word read never depends on the timing).
        fast, count = self.near_frames(0.0)
        slow, _ = self.near_frames(.02)
        self.assertEqual(fast, list(range(6, 12)))
        self.assertEqual(slow, fast)
        self.assertEqual(count, 6)

    def test_no_near_pass_when_it_is_switched_off(self):
        frames = self.frames(6)
        with tempfile.TemporaryDirectory() as folder:
            path = write_video(os.path.join(folder, 'clip'), frames)
            if path is None:
                self.skipTest('no video codec available')
            detector = ClassAwareDetector()
            ahead = pipeline.Prefetch(path, 'unused.pt', 0, FPS, 0.0, 1.0, None, 1536, False, lambda: False, depth=1, batch=1, model=detector)
            for kind, index, *_ in ahead:
                ahead.note_scene(index, True)
            ahead.close()
        self.assertNotIn(pipeline.NEAR_PASS, [size for size, _, _ in detector.seen])
        self.assertEqual(ahead.near_frames, 0)

    def test_near_pass_frames_count_only_frames_the_extra_pass_ran_on(self):
        # A detail size of NEAR_PASS already looks at that size: no extra pass, and none counted in nearPassFrames.
        frames = self.frames(8)
        with tempfile.TemporaryDirectory() as folder:
            path = write_video(os.path.join(folder, 'clip'), frames)
            if path is None:
                self.skipTest('no video codec available')
            counted = {}
            for detail in (pipeline.NEAR_PASS, 1536):
                detector = ClassAwareDetector()
                ahead = pipeline.Prefetch(path, 'unused.pt', 0, FPS, 0.0, 1.0, None, detail, False, lambda: False, depth=1, batch=1,
                                          model=detector, near=True)
                for kind, index, *_ in ahead:
                    ahead.note_scene(index, True)
                ahead.close()
                people_only = [v for size, classes, v in detector.seen if classes == [1, 2, 3]]
                counted[detail] = (ahead.near_frames, len(people_only))
        self.assertEqual(counted[pipeline.NEAR_PASS], (0, 0))
        self.assertEqual(counted[1536], (6, 6))   # frames 2-7: each reads the word of the frame two before it

    def test_tiles_and_crops_of_a_low_resolution_source_are_scaled(self):
        frame = np.zeros((506, 896, 3), np.uint8)
        detector = ClassAwareDetector()
        pipeline.ball_tiles(detector, frame)
        pipeline.ball_tiles(detector, frame, 506/1080)
        self.assertEqual([len(v) for _, _, v in detector.seen], [2, 6])     # 506 px tiles at 1.26x; 375 px ones at 1.71x
        sizes = []

        class CropDetector:   # ball_crop reads the boxes one by one: none here, only the input size matters
            def predict(self, crop, imgsz, **kwargs):
                sizes.append(imgsz)
                return [SimpleNamespace(boxes=[])]
        pipeline.ball_crop(CropDetector(), frame, (100, 100, 300, 300))
        pipeline.ball_crop(CropDetector(), frame, (100, 100, 300, 300), 506/1080)
        self.assertEqual(sizes, [320, 416])                             # 1x (at least 320); 2.13x, as at 1080p
        full = np.zeros((1080, 1920, 3), np.uint8)
        detector = ClassAwareDetector()
        pipeline.ball_tiles(detector, full)
        pipeline.ball_tiles(detector, full, 1.0)
        self.assertEqual([len(v) for _, _, v in detector.seen], [6, 6])

    def test_ball_candidates_carry_their_source_through_the_shipped_suppression(self):
        rng = np.random.default_rng(7)
        for trial in range(20):
            def rows(n):
                xy = rng.uniform(0, 600, (n, 2))
                size = rng.uniform(4, 40, (n, 1))
                return np.c_[xy, xy+size, rng.uniform(.05, .95, (n, 1)), np.zeros((n, 1))].astype(np.float32)
            people = np.r_[rows(5), np.c_[rows(3)[:, :5], np.full(3, 2.0)]].astype(np.float32)
            tiles = rows(int(rng.integers(0, 6))).tolist() if trial % 3 else None
            crop = [[float(v) for v in r] for r in rows(int(rng.integers(0, 4)))]
            # The shipped inline code of pipeline.run.
            old = people[people[:, 5] == 0].tolist()+(tiles or [])+crop
            if old:
                raw = torch.tensor(old, dtype=torch.float32)
                old = raw[pipeline.nms(raw[:, :4], raw[:, 4], .3)].numpy()
                old = old[np.maximum(old[:, 2]-old[:, 0], old[:, 3]-old[:, 1]) <= 640*.035]
            old = np.asarray(old, np.float32).reshape(-1, 6)
            new, counts = pipeline.ball_candidates(people, tiles, crop, 640, 480)
            self.assertEqual(new.shape[1], 7)
            self.assertTrue(np.array_equal(new[:, :6], old))
            self.assertEqual(counts['raw'], [5, len(tiles or []), len(crop)])
            self.assertEqual(counts['kept'], [int((new[:, 6] == s).sum()) for s in (0, 1, 2)])
            sources = {tuple(r[:5]): s for r, s in zip(people[people[:, 5] == 0].tolist()+(tiles or [])+crop,
                                                        [0]*5+[1]*len(tiles or [])+[2]*len(crop))}
            for r in new:
                self.assertEqual(sources[tuple(np.float32(r[:5]).tolist())], r[6])
        empty, counts = pipeline.ball_candidates(np.zeros((0, 6), np.float32), None, [], 640, 480)
        self.assertEqual(empty.shape, (0, 7))
        self.assertEqual(counts, {'raw': [0, 0, 0], 'kept': [0, 0, 0]})

    def test_automatic_landmarks_only_where_no_keyframe_reaches_the_segment(self):
        self.assertTrue(pipeline.auto_landmarks([0.0, 22.3], 118.2, 182.0))     # amateur 3, 118-182 s
        self.assertFalse(pipeline.auto_landmarks([0.0, 22.3], 0.0, 60.0))       # its 0-60 s regression window
        self.assertFalse(pipeline.auto_landmarks([10.0], 25.0, 40.0))           # ROLL_SECONDS before the start: carried in
        self.assertTrue(pipeline.auto_landmarks([9.9], 25.0, 40.0))
        self.assertFalse(pipeline.auto_landmarks([55.0], 25.0, 40.0))           # ... and after the end
        self.assertTrue(pipeline.auto_landmarks([], 0.0, 60.0))

    def test_the_run_profile_counts_small_scene_frames(self):
        frames = [{'time': i/10, 'profile': {'smallScene': i % 3 != 0}} for i in range(9)]+[{'time': .9}]
        profile = pipeline.run_profile(frames, SimpleNamespace(auto_landmarks=True, pixel_scale=1.0), True)
        self.assertEqual(profile['smallSceneFrames'], 6)
        self.assertTrue(profile['smallSceneRun'])
        self.assertEqual(profile['smallSceneShare'], .6)
        self.assertEqual(profile['smallSceneChanges'], 6)   # F T T F T T F T T F: every change of the gate between frames
        self.assertEqual(profile['smallSceneFrom'], .1)
        empty = pipeline.run_profile([], SimpleNamespace(auto_landmarks=False, pixel_scale=1.0), False)
        self.assertFalse(empty['smallSceneRun'])
        self.assertEqual((empty['smallSceneChanges'], empty['smallSceneFrom']), (0, None))


if __name__ == '__main__':
    unittest.main()
