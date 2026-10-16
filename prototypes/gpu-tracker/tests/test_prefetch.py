"""The prefetch worker's batched whole-frame detection: the frames of a segment go through the detector
several at a time, and what comes out per frame (rows, order, roll frames, tile scans) is what the serial
loop computed. A stub detector stands in for the CUDA model: it answers a list of images with one result
per image, derived from that image alone, so a batch can only be right when every frame's rows come from
its own image in its own position."""
import os
import sys
import tempfile
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


if __name__ == '__main__':
    unittest.main()
