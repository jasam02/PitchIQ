"""The detectors and the embedder kept between runs (pipeline.Resources): loaded once per process, the same
instances handed to every run, the prefetch worker given its own instance instead of loading one; a different
checkpoint reloads; the wrong classes are refused. Stubs stand in for the CUDA model and the ONNX session."""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import pipeline
    AVAILABLE = True
except Exception:   # noqa: BLE001 - torch / ultralytics missing
    AVAILABLE = False


class StubYOLO:
    made = 0

    def __init__(self, path, names=None):
        StubYOLO.made += 1
        self.path = path
        self.names = names or {0: 'ball', 1: 'goalkeeper', 2: 'player', 3: 'referee'}


class StubAppearance:
    made = 0

    def __init__(self):
        StubAppearance.made += 1


@unittest.skipUnless(AVAILABLE, 'torch / ultralytics not installed')
class ResourcesTests(unittest.TestCase):
    def setUp(self):
        self.original = pipeline.YOLO, pipeline.Appearance
        pipeline.YOLO, pipeline.Appearance = StubYOLO, StubAppearance
        StubYOLO.made = StubAppearance.made = 0

    def tearDown(self):
        pipeline.YOLO, pipeline.Appearance = self.original

    def test_loaded_once_and_kept(self):
        shared = pipeline.Resources()
        self.assertFalse(shared.loaded)
        notices = []
        first = shared.load('models/a.pt', notices.append)
        self.assertIs(first, shared)
        self.assertTrue(shared.loaded)
        self.assertEqual((StubYOLO.made, StubAppearance.made), (2, 1))   # the main thread's and the worker's detector, one embedder
        self.assertIsNot(shared.detector, shared.worker_detector)
        self.assertEqual(len(notices), 1)
        detector, worker, appearance = shared.detector, shared.worker_detector, shared.appearance
        shared.load('models/a.pt', notices.append)
        self.assertEqual((StubYOLO.made, StubAppearance.made), (2, 1))   # nothing loaded again
        self.assertEqual(len(notices), 1)
        self.assertIs(shared.detector, detector)
        self.assertIs(shared.worker_detector, worker)
        self.assertIs(shared.appearance, appearance)
        shared.load('models/b.pt')   # another checkpoint: loaded anew
        self.assertEqual((StubYOLO.made, StubAppearance.made), (4, 2))
        self.assertEqual(shared.detector.path, 'models/b.pt')
        shared.release()
        self.assertFalse(shared.loaded)
        self.assertIsNone(shared.detector)

    def test_the_wrong_classes_are_refused(self):
        pipeline.YOLO = lambda path: StubYOLO(path, {0: 'person'})
        with self.assertRaises(RuntimeError):
            pipeline.Resources().load('models/other.pt')

    def test_the_process_pair_exists_and_starts_empty_or_loaded_once(self):
        self.assertIsInstance(pipeline.RESOURCES, pipeline.Resources)


if __name__ == '__main__':
    unittest.main()
