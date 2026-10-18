"""A CPU smoke test of pipeline.run, the wiring nothing else exercises before a GPU run: a short written clip, a stub
detector (connected components of kit and ball colours, answering every pass the pipeline makes: the whole-frame
sizes, the near pass, the ball tiles, the far tiles, the crops) and a stub appearance embedder, handed in through
pipeline.Resources; CUDA is never touched. Checks what every run must carry: the frames' profile, ballPasses and
dropped keys, the summary's profile and detection counters, the offline passes' wiring, and a result.json that parses.
One clip at 720p (a closed gate: nothing small) and the same clip at 360 rows (a low-resolution source: a small scene on
every frame, so the near pass runs)."""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import cv2
    import torch
    import pipeline   # sets YOLO_CONFIG_DIR to the pipeline's own settings before ultralytics is imported
    from scene import pitch_frame
    AVAILABLE = True
except Exception:   # noqa: BLE001 - torch / ultralytics missing: the tests below are skipped
    AVAILABLE = False

FPS = 25
FRAMES = 30
RED, BLUE = (40, 40, 210), (200, 90, 20)   # BGR jerseys of the two teams


class StubBoxes:
    """What the pipeline reads of a result's boxes: `.data` (N, 6) [x1, y1, x2, y2, score, class], and per box
    `.xyxy` and `.conf` (ball_crop)."""

    def __init__(self, rows):
        self.data = torch.tensor(np.asarray(rows, np.float32).reshape(-1, 6))

    def __iter__(self):
        for row in self.data:
            yield type('Box', (), {'xyxy': row[None, :4], 'conf': row[4:5]})()


class StubResult:
    def __init__(self, rows):
        self.boxes = StubBoxes(rows)


class ColourDetector:
    """People are the red and blue blobs (players, class 2), the ball a small white square (class 0), in whatever image
    it is given (a frame, a tile, a crop), in that image's pixels; `classes` is honoured like the real predictor's."""
    names = {0: 'ball', 1: 'goalkeeper', 2: 'player', 3: 'referee'}

    def __init__(self):
        self.calls = []   # (images, imgsz, classes)

    @staticmethod
    def blobs(mask, keep):
        count, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
        return [(x, y, x+w, y+h) for x, y, w, h, area in stats[1:count] if keep(w, h, area)]

    def detect(self, image):
        b, g, r = (image[..., i].astype(np.int16) for i in range(3))
        rows = []
        for mask in ((r > 150) & (g < 110) & (b < 110), (b > 150) & (g < 140) & (r < 90)):
            rows += [[x1, y1, x2, y2, .85, 2] for x1, y1, x2, y2 in self.blobs(mask, lambda w, h, area: h >= 6 and h > w and area >= 12)]
        white = (r > 200) & (g > 200) & (b > 200)
        rows += [[x1, y1, x2, y2, .7, 0] for x1, y1, x2, y2 in self.blobs(white, lambda w, h, area: 3 <= w <= 14 and 3 <= h <= 14 and area >= .5*w*h)]
        return rows

    def predict(self, source, imgsz=640, classes=None, **kwargs):
        images = source if isinstance(source, (list, tuple)) else [source]
        self.calls.append((len(images), imgsz, classes))
        out = []
        for image in images:
            rows = self.detect(image)
            if classes is not None:
                rows = [r for r in rows if r[5] in classes]
            out.append(StubResult(rows))
        return out


class StubAppearance:
    """A 512-d unit descriptor of the box's mean colour (a fixed projection): one per box, like vision.Appearance."""
    projection = np.random.default_rng(0).normal(size=(3, 512)).astype(np.float32)

    def embed(self, frame, boxes):
        out = []
        for x1, y1, x2, y2 in boxes:
            crop = frame[max(0, int(y1)):max(int(y1)+1, int(y2)), max(0, int(x1)):max(int(x1)+1, int(x2))].reshape(-1, 3)
            v = (crop.mean(axis=0)/255.0 if len(crop) else np.zeros(3)).astype(np.float32) @ self.projection
            out.append(v/max(float(np.linalg.norm(v)), 1e-9))
        return out


def clip(width, height):
    """FRAMES frames of a pitch (no stands), six players running across it and a ball rolling between them."""
    frames = []
    for k in range(FRAMES):
        frame = pitch_frame(stands=False)
        for i in range(6):
            x, y = 200+170*i+4*k, 330+40*(i % 3)
            cv2.rectangle(frame, (int(x), y), (int(x)+14, y+38), RED if i % 2 else BLUE, -1)
        bx = 300+12*k
        cv2.rectangle(frame, (bx, 560), (bx+9, 569), (245, 245, 245), -1)
        frames.append(cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA) if (width, height) != frame.shape[1::-1] else frame)
    return frames


def write_video(path, frames):
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), FPS, (width, height))
    if not writer.isOpened():
        writer.release()
        return False
    for frame in frames:
        writer.write(frame)
    writer.release()
    return True


@unittest.skipUnless(AVAILABLE, 'torch / ultralytics not installed')
class PipelineRunSmokeTests(unittest.TestCase):
    def run_clip(self, width, height):
        with tempfile.TemporaryDirectory() as folder:
            video = os.path.join(folder, 'clip.mp4')
            if not write_video(video, clip(width, height)):
                self.skipTest('no mp4v writer')
            shared = pipeline.Resources()
            shared.detector, shared.worker_detector, shared.appearance = ColourDetector(), ColourDetector(), StubAppearance()
            shared.model_path = Path('stub.pt')
            config = {'start': 0.0, 'seconds': FRAMES/FPS, 'imageSize': 1280, 'ballTiles': True, 'debug': True}
            output = Path(folder)/'run'
            gpu = {'name': 'stub', 'torch': torch.__version__, 'cuda': None, 'memoryGB': 0.0}
            with mock.patch.object(pipeline, 'gpu_info', lambda: gpu), mock.patch.object(pipeline, 'get_model', lambda: 'stub.pt'), \
                    mock.patch.object(pipeline.torch.cuda, 'reset_peak_memory_stats', lambda *a, **k: None), \
                    mock.patch.object(pipeline.torch.cuda, 'is_available', lambda: False):
                report = pipeline.run(Path(video), output, config, lambda notice: None, lambda: False, resources=shared)
            files = {name: json.loads((output/name).read_text(encoding='utf-8')) for name in ('result.json', 'registration.json', 'identities.json')}
            calls = shared.detector.calls+shared.worker_detector.calls
        return report, files, calls

    def check_run(self, report, files):
        result = files['result.json']
        self.assertEqual(result['status'], 'complete')
        frames = result['frames']
        self.assertGreaterEqual(len(frames), FRAMES-1)
        for f in frames:
            self.assertIsInstance(f['profile']['smallScene'], bool)
            self.assertEqual(set(f['ballPasses']), {'raw', 'kept', 'tiles', 'crop'})
            self.assertEqual(len(f['ballPasses']['raw']), 3)
            self.assertIsInstance(f['dropped'], list)
        # The ball search crop: none before the ball is followed (the `window = None` branch), then around it.
        crops = [f['ballPasses']['crop'] for f in frames]
        self.assertIsNone(crops[0])
        self.assertTrue(any(c is not None for c in crops), crops)
        self.assertTrue(any(f['ballPasses']['tiles'] for f in frames))
        self.assertTrue(any(f['ball']['state'] == 'TRACKED' for f in frames))
        summary = result['summary']
        profile = summary['profile']
        self.assertTrue(profile['autoLandmarks'])   # no calibration keyframe at all
        self.assertEqual(set(profile), {'smallSceneFrames', 'smallSceneShare', 'smallSceneRun', 'smallSceneChanges', 'smallSceneFrom',
                                        'autoLandmarks', 'pixelScale', 'nearPass'})
        self.assertIn('nearPassFrames', summary['detection'])
        self.assertIn('fieldSize', summary['pitch']['registration'])   # registration.summarize got the run's pitch size
        self.assertIn('lockAudits', summary['ball'])
        self.assertGreater(summary['meanVisiblePeople'], 0)
        self.assertEqual(report['summary']['profile'], profile)
        self.assertEqual(files['registration.json']['emitted'], len(frames))
        self.assertIn('registry', files['identities.json'])
        return result

    def test_a_720p_run_keeps_the_gate_closed(self):
        report, files, calls = self.run_clip(1280, 720)
        result = self.check_run(report, files)
        profile = result['summary']['profile']
        self.assertEqual((profile['smallSceneFrames'], profile['smallSceneChanges'], profile['pixelScale']), (0, 0, 1.0))
        self.assertEqual(result['summary']['detection']['nearPassFrames'], 0)
        self.assertNotIn((1, pipeline.NEAR_PASS, [1, 2, 3]), calls)

    def test_a_low_resolution_run_is_a_small_scene_with_the_near_pass(self):
        report, files, calls = self.run_clip(640, 360)
        result = self.check_run(report, files)
        profile = result['summary']['profile']
        self.assertTrue(profile['smallSceneRun'])
        self.assertEqual(profile['smallSceneFrames'], len(result['frames']))
        self.assertEqual(profile['smallSceneChanges'], 0)
        self.assertAlmostEqual(profile['pixelScale'], 360/1080, places=4)
        near = result['summary']['detection']['nearPassFrames']
        self.assertGreater(near, 0)
        self.assertEqual(near, sum(1 for n, size, classes in calls if size == pipeline.NEAR_PASS and classes == [1, 2, 3]))


if __name__ == '__main__':
    unittest.main()
