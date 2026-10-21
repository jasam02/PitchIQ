"""Exact source frames for the coaching screenshots (soccer/coaching/frames.py): a grabbed frame is the very frame a
sequential decode gives at that index (seek path and grab-ahead path), the decoder's own time is checked, frames
are read once and in order, the JPEG cache is reused, and nothing is ever enlarged.

Run:  python -m unittest test_coaching_frames          (from this directory)
"""
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from soccer.coaching import frames  # noqa: E402

VEO_UPLOAD = ROOT/'uploads'/'f597d73d53f7b1fd.mp4'
W, H, N, FPS = 320, 240, 120, 30.0


def code(image):
    """The frame index the synthetic clip paints into each frame (7 black/white blocks)."""
    return sum(1 << b for b in range(7) if image[205, 10+b*42+18].mean() > 128)


class ClipTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.clip = Path(cls.tmp.name)/'clip.mp4'
        writer = cv2.VideoWriter(str(cls.clip), cv2.VideoWriter_fourcc(*'mp4v'), FPS, (W, H))
        if not writer.isOpened():
            raise unittest.SkipTest('no mp4v encoder in this OpenCV build')
        noise = np.random.default_rng(0).integers(0, 60, (H, W, 3), dtype=np.uint8)
        for n in range(N):
            img = noise.copy()
            img[:, :, 1] = (np.arange(W)[None, :]*255//W+n*3) % 256
            for b in range(7):
                img[180:230, 10+b*42:46+b*42] = 255 if (n >> b) & 1 else 0
            writer.write(img)
        writer.release()
        capture = cv2.VideoCapture(str(cls.clip))
        cls.sequential = []
        while True:
            ok, img = capture.read()
            if not ok:
                break
            cls.sequential.append(img)
        capture.release()
        if len(cls.sequential) != N or any(code(img) != n for n, img in enumerate(cls.sequential)):
            raise unittest.SkipTest('the synthetic clip did not decode as written')

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_grabbed_frames_equal_the_sequential_decode(self):
        # 3 -> 5 decodes forward (grab), 5 -> 70 seeks (more than GRAB_AHEAD ahead), 70 -> 71 the next frame.
        wanted = [100, 3, 5, 5, 70, 71, 119, 2]
        got = list(frames.iter_frames(self.clip, wanted))
        self.assertEqual([f for f, _, _ in got], sorted(set(wanted)))      # once each, in order
        for f, img, err in got:
            self.assertIsNone(err)
            self.assertEqual(code(img), f)
            self.assertEqual(int(np.abs(img.astype(int)-self.sequential[f]).max()), 0, f)

    def test_the_decoder_time_is_checked(self):
        late = {5: 5/FPS+.4/FPS, 6: 6/FPS+.6/FPS}                         # within half a frame / beyond it
        got = {f: (img, err) for f, img, err in frames.iter_frames(self.clip, [5, 6], times=late)}
        self.assertIsNone(got[5][1])
        self.assertIsNone(got[6][0])
        self.assertIn('not the expected', got[6][1])

    def test_frames_outside_the_video_and_a_missing_video(self):
        got = {f: err for f, _, err in frames.iter_frames(self.clip, [-1, N+5, 4])}
        self.assertIsNone(got[4])
        self.assertTrue(got[-1] and got[N+5])
        missing = list(frames.iter_frames(Path(self.tmp.name)/'none.mp4', [1, 2]))
        self.assertEqual([(f, img) for f, img, _ in missing], [(1, None), (2, None)])
        self.assertTrue(all(err for _, _, err in missing))

    def test_the_cache_is_written_once_and_reused(self):
        cache = Path(self.tmp.name)/'cache'
        first = frames.extract(self.clip, [10, 12, 90], cache, widths=(640, 1280))
        self.assertEqual(first['decoded'], 3)
        self.assertEqual(first['errors'], {})
        self.assertTrue(frames.frame_path(cache, 12, 640).is_file())
        self.assertEqual(sorted(p.name for p in cache.iterdir()),
                         sorted(f'f{f}-w{w}.jpg' for f in (10, 12, 90) for w in (640, 1280)))   # no temp files
        again = frames.extract(self.clip, [10, 12, 90], cache, widths=(640, 1280))
        self.assertEqual((again['decoded'], again['cached']), (0, 6))
        self.assertEqual(again['jpegs'], first['jpegs'])
        img = cv2.imdecode(np.frombuffer(first['jpegs'][(12, 640)], np.uint8), cv2.IMREAD_COLOR)
        self.assertEqual(img.shape[:2], (H, W))                              # a 320 px source is not enlarged
        self.assertEqual(code(img), 12)
        with self.assertRaises(ValueError):
            frames.extract(self.clip, [1], cache, widths=(800,))

    def test_the_provider(self):
        provider = frames.FrameProvider(self.clip)
        provider.prefetch([7, 8], [640])
        self.assertEqual(provider.decoded, 2)
        data = provider(8, 640)
        self.assertEqual(code(cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)), 8)
        self.assertEqual(provider.decoded, 2)                                # served from memory
        self.assertIsNone(provider(N+50, 640))
        self.assertIsNotNone(provider(9))


class JpegTests(unittest.TestCase):
    def test_resize_and_baseline_jpeg(self):
        img = np.zeros((1080, 1920, 3), np.uint8)
        img[:, 960:] = 200
        for width, height in ((1280, 720), (640, 360)):
            data = frames.encode_jpeg(img, width)
            dec = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            self.assertEqual(dec.shape[:2], (height, width))
            self.assertIn(b'\xff\xc0', data)                                 # baseline (SOF0) ...
            self.assertNotIn(b'\xff\xc2', data)                              # ... never progressive
        small = img[:100, :500]
        self.assertIs(frames.resize(small, 640), small)                      # never enlarged


@unittest.skipUnless(VEO_UPLOAD.is_file(), 'the Veo upload is not on this machine')
class UploadTests(unittest.TestCase):
    def test_h264_frames_equal_the_sequential_decode(self):
        # The Veo run starts at frame 30 (1.0 s). Seek path, grab path and neighbours, against a decode from 0.
        targets = [30, 31, 77, 120, 121, 165, 199]
        capture = cv2.VideoCapture(str(VEO_UPLOAD))
        sequential = {}
        for n in range(max(targets)+1):
            ok, img = capture.read()
            self.assertTrue(ok)
            if n in targets:
                sequential[n] = img
        capture.release()
        times = {f: f/30.0 for f in targets}
        got = list(frames.iter_frames(VEO_UPLOAD, targets, times=times, fps=30.0))
        self.assertEqual([f for f, _, _ in got], targets)
        for f, img, err in got:
            self.assertIsNone(err, f)
            self.assertEqual(float(np.abs(img.astype(np.int16)-sequential[f].astype(np.int16)).mean()), 0.0, f)


if __name__ == '__main__':
    unittest.main()
