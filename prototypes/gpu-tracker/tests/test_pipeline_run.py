"""A CPU smoke test of pipeline.run, the wiring nothing else exercises before a GPU run: a short written clip, a stub
detector (connected components of kit and ball colours, answering every pass the pipeline makes: the whole-frame
sizes, the near pass, the ball tiles, the far tiles, the crops) and a stub appearance embedder, handed in through
pipeline.Resources; CUDA is never touched. Checks what every run must carry: the frames' profile, ballPasses and
dropped keys, the summary's profile and detection counters, the offline passes' wiring, and a result.json that parses.
One clip at 720p (a closed gate: nothing small) and the same clip at 360 rows (a low-resolution source: a small scene on
every frame, so the near pass runs)."""
import copy
import importlib.util
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


# ---------- the 2D replay's stabilized block (soccer/stabilized.py, attached by pipeline.run) ----------
# The helpers are test_corrections' (no torch there); these tests need none either, except the whole runs below.
from test_corrections import SHIFT, failing, stabilizer_module, stabilizer_result, stub_stabilize, without_noise  # noqa: E402


def broken(change):
    """The stub stabilizer with its output changed by change(out, result, size)."""
    stub = stub_stabilize()

    def stabilize(result, width, height, size=None):
        out = stub(result, width, height, size)
        change(out, result, size)
        return out
    return stabilize


def every_track(out, key, value):
    for cols in list(out['ids'].values())+list(out['anonymous'].values()):
        cols[key] = [value(v) for v in cols[key]]


class StabilizedBlockTests(unittest.TestCase):
    def test_the_block_follows_the_contract_and_leaves_the_raw_data_alone(self):
        from soccer import stabilized
        result = stabilizer_result()
        before = copy.deepcopy({k: result[k] for k in ('frames', 'timeline', 'roster')})
        calls = []
        summary = stabilized.attach(result, 1920, 1080, stabilize=stub_stabilize(calls))
        self.assertEqual(summary['status'], 'ok', summary)
        self.assertEqual(calls[0][:2], (1920, 1080))
        self.assertEqual((calls[0][2].length, calls[0][2].width), (105.0, 68.0))   # the result's own pitch
        self.assertIsNone(calls[0][3])                                             # a new run fits its own noise model
        block, timeline = result['stabilized'], result['timeline']
        self.assertEqual(set(block), {'version', 'method', 'frameTimes', 'ids', 'anonymous', 'flags', 'summary', 'noise', 'constants'})
        self.assertEqual((block['version'], block['method']), (1, 'stub'))
        self.assertEqual(block['frameTimes'], timeline['frameTimes'])
        self.assertEqual(set(block['ids']), set(timeline['ids']))
        T = len(result['frames'])
        for pid, cols in block['ids'].items():
            self.assertEqual(set(cols), {'team', 'x', 'y', 'kind', 'sigma', 'w'}, pid)
            self.assertEqual(cols['team'], timeline['ids'][pid]['team'])
            for key in ('x', 'y', 'kind', 'sigma', 'w'):
                self.assertEqual(len(cols[key]), T, (pid, key))
            self.assertTrue(set(cols['kind']) <= {'O', 'I', 'P', 'U'})
        a01, raw = block['ids']['A-01'], timeline['ids']['A-01']
        # An observation: the stabilizer's position (normalized, 4 decimals), sigma (1) and weight (2; a whole one an int).
        x5 = result['frames'][5]['people'][0]['pitch'][0]
        self.assertEqual((a01['kind'][5], a01['x'][5], a01['sigma'][5], a01['w'][5]), ('O', round((x5*105.0+SHIFT)/105.0, 4), [.4, .2], 1))
        self.assertIs(type(a01['w'][5]), int)
        self.assertNotEqual(a01['x'][5], raw['x'][5])
        self.assertEqual((a01['kind'][10], a01['w'][10]), ('I', 0))           # the stabilizer's bridge over a rejected one
        self.assertEqual((a01['kind'][7], a01['x'][7], a01['w'][7]), ('U', None, None))   # no registration: nobody placed
        # A-03 vanished at frame 15: imputation's estimate on the stabilized observations starts where the stabilized
        # player was (three players a team: no voted centre, so the last position held), not where the raw one was,
        # and from the raw observation's sigma: its kinds and sigmas are the timeline's.
        a03, raw03 = block['ids']['A-03'], timeline['ids']['A-03']
        self.assertEqual(a03['kind'][14], 'O')
        self.assertEqual(a03['kind'][15], 'P')
        self.assertEqual(a03['x'][15], a03['x'][14])
        self.assertNotEqual(a03['x'][15], raw03['x'][15])
        self.assertEqual(a03['w'][15], None)
        self.assertEqual((a03['kind'][15:], a03['sigma'][15:]), (raw03['kind'][15:], raw03['sigma'][15:]))
        self.assertGreater(a03['sigma'][20][0], a03['sigma'][16][0])
        # The anonymous '?' track: its team from its observations, its own kinds and sigmas.
        self.assertEqual(set(block['anonymous']), {'90'})
        anon = block['anonymous']['90']
        self.assertEqual(set(anon), {'team', 'x', 'y', 'kind', 'sigma'})
        self.assertEqual(anon['team'], 'B')
        self.assertEqual((anon['kind'][4], anon['kind'][5], anon['kind'][7], anon['x'][5]), ('U', 'O', 'U', round((.85*105.0+SHIFT)/105.0, 4)))
        # Flags and the stabilizer's own blocks as plain JSON; its runtime is the run summary's, not the block's.
        self.assertEqual(block['flags'], [{'id': 'A-01', 'track': 1, 'frame': 10, 'time': 12.0, 'reason': 'extent',
                                           'raw': [round(result['frames'][10]['people'][0]['pitch'][0]*105.0+4.123, 2), round(result['frames'][10]['people'][0]['pitch'][1]*68.0, 2)],
                                           'used': None}])
        self.assertEqual(block['noise'], {'sigmaPx': .4, 'unfitted': None})
        self.assertEqual(block['summary'], {'observations': 5, 'rejected': [1]})
        text = json.dumps(block, separators=(',', ':'), allow_nan=False)        # no numpy, no NaN: the server writes it again
        self.assertEqual(summary['bytes'], len(text))
        # The run summary's entry: kinds and slots over the roster slots, allTracks the stabilizer's own and the flags.
        self.assertEqual(set(summary), {'status', 'noise', 'ids', 'anonymous', 'rawSlots', 'rawFrames', 'rawShare', 'kinds', 'slots', 'allTracks',
                                        'seconds', 'bytes'})
        self.assertEqual((summary['noise'], summary['ids'], summary['anonymous']), ('fitted', 6, 1))
        self.assertEqual((summary['rawSlots'], summary['rawFrames'], summary['rawShare']), ([], 0, 0.0))
        self.assertAlmostEqual(sum(summary['kinds'].values()), 1.0, places=2)
        slots = summary['slots']
        self.assertEqual(slots['observations'], sum(1 for cols in block['ids'].values() for w in cols['w'] if w is not None))
        self.assertGreater(slots['rejectedShare'], 0)
        self.assertEqual(slots['downweightedShare'], 0.0)
        self.assertEqual(set(slots['correctionMetres']), {'samples', 'median', 'p90', 'max'})
        self.assertAlmostEqual(slots['correctionMetres']['median'], SHIFT, delta=.02)
        self.assertEqual(summary['allTracks'], {'observations': 5, 'rejectedShare': None, 'downweightedShare': None, 'repairedShare': None,
                                                'splits': None, 'flags': 1, 'flagReasons': {'extent': 1}})
        # The raw positions, the timeline and the roster are exactly what they were; the same result, the same block.
        self.assertEqual({k: result[k] for k in ('frames', 'timeline', 'roster')}, before)
        again = stabilizer_result()
        stabilized.attach(again, 1920, 1080, stabilize=stub_stabilize())
        self.assertEqual(json.dumps(again['stabilized'], sort_keys=True), json.dumps(block, sort_keys=True))

    def test_the_timeline_observations_the_stabilizer_leaves_unplaced_are_drawn_raw_within_a_limit(self):
        from soccer import stabilized

        def few(out, result, size):
            out['ids']['B-01']['x'][20] = float('nan')                          # not a position
            out['ids']['B-01']['sx'][21] = float('inf')                         # a position without a sigma
            out['ids']['B-02']['x'][22] = 2.0*size.length                       # beyond the drawable pitch
            out['anonymous']['90']['kind'][6] = None                            # the '?' track's too
        result = stabilizer_result()
        with mock.patch.object(stabilized, 'MAX_RAW_SHARE', .05):                 # 3 of 228 observations: over the 1 % a real run allows
            summary = stabilized.attach(result, 1920, 1080, stabilize=broken(few))
        self.assertEqual(summary['status'], 'ok', summary)
        self.assertEqual(summary['rawFrames'], 3)
        block, timeline = result['stabilized'], result['timeline']
        b01, raw01 = block['ids']['B-01'], timeline['ids']['B-01']
        self.assertEqual((b01['kind'][20], b01['x'][20], b01['y'][20], b01['sigma'][20], b01['w'][20]),
                         ('O', raw01['x'][20], raw01['y'][20], raw01['sigma'][20], None))   # the raw observation, as the timeline has it
        self.assertEqual((b01['kind'][21], b01['sigma'][21]), ('O', None))
        self.assertEqual((block['ids']['B-02']['kind'][22], block['ids']['B-02']['x'][22]), ('O', timeline['ids']['B-02']['x'][22]))
        anon = block['anonymous']['90']
        self.assertEqual((anon['kind'][6], anon['x'][6], anon['sigma'][6]), ('O', .85, [1.0, 1.0]))
        json.dumps(block, allow_nan=False)
        # A slot the stabilizer did not return at all is its raw observations: within the limit drawn so, named in rawSlots.
        result = stabilizer_result()
        without_b02 = broken(lambda out, result, size: out['ids'].pop('B-02'))
        with mock.patch.object(stabilized, 'MAX_RAW_SHARE', .5):
            summary = stabilized.attach(result, 1920, 1080, stabilize=without_b02)
        self.assertEqual((summary['status'], summary['rawSlots']), ('ok', ['B-02']))
        self.assertEqual(summary['rawFrames'], sum(1 for k in result['timeline']['ids']['B-02']['kind'] if k == 'O'))
        b02, raw = result['stabilized']['ids']['B-02'], result['timeline']['ids']['B-02']
        self.assertEqual((b02['kind'], b02['x'], b02['sigma']), (raw['kind'], raw['x'], raw['sigma']))
        self.assertTrue(all(w is None for w in b02['w']))
        # ... and beyond it (a sixth of the observations) the block is left out.
        result = stabilizer_result()
        with mock.patch('traceback.print_exc'), mock.patch('builtins.print'):
            summary = stabilized.attach(result, 1920, 1080, stabilize=without_b02)
        self.assertEqual(summary['status'], 'failed')
        self.assertIn('B-02', summary['error'])
        self.assertNotIn('stabilized', result)

    def test_a_failing_stabilizer_leaves_the_block_out_and_never_raises(self):
        from soccer import stabilized

        def truncated(out, result, size):
            for cols in out['ids'].values():
                for key in cols:
                    cols[key] = cols[key][:10]
        cases = (('raises', failing), ('wrong frame count', lambda r, *a, **k: {'frameTimes': [0.0], 'ids': {}}),
                 ('not a dict', lambda r, *a, **k: None),
                 ('all NaN', broken(lambda out, result, size: every_track(out, 'x', lambda v: None if v is None else float('nan')))),
                 ('no kinds', broken(lambda out, result, size: every_track(out, 'kind', lambda v: None))),
                 ('no slots', broken(lambda out, result, size: out.update(ids=None))),
                 ('truncated columns', broken(truncated)),
                 ('far off the pitch', broken(lambda out, result, size: every_track(out, 'x', lambda v: None if v is None else 1e9))),
                 ('everybody moved 5 m', broken(lambda out, result, size: every_track(out, 'y', lambda v: None if v is None else v+5.0))),
                 ('unserializable', broken(lambda out, result, size: out['noise'].update(bad={1, 2}))))
        for name, stabilize in cases:
            result = stabilizer_result()
            result['stabilized'] = {'version': 1, 'ids': {'A-09': {}}}            # an older block never survives
            timeline = copy.deepcopy(result['timeline'])
            with mock.patch('traceback.print_exc'), mock.patch('builtins.print'):
                summary = stabilized.attach(result, 1920, 1080, stabilize=stabilize)
            self.assertEqual(summary['status'], 'failed', name)
            self.assertEqual(set(summary), {'status', 'error', 'seconds'}, name)
            self.assertNotIn('stabilized', result, name)
            self.assertEqual(result['timeline'], timeline)
        # A broken module (no stabilize) is a failure of the block too, not of the caller.
        result = stabilizer_result()
        with stabilizer_module(None), mock.patch('traceback.print_exc'), mock.patch('builtins.print'):
            summary = stabilized.attach(result, 1920, 1080)
        self.assertEqual(summary['status'], 'failed')
        self.assertIn('AttributeError', summary['error'])
        # A result without a trusted registration on any frame has nobody to place: no block, not a failure.
        result = stabilizer_result()
        for f in result['frames']:
            f['calibrated'] = False
        calls = []
        summary = stabilized.attach(result, 1920, 1080, stabilize=stub_stabilize(calls))
        self.assertEqual((summary['status'], set(summary), calls), ('skipped', {'status', 'reason', 'seconds'}, []))
        self.assertNotIn('stabilized', result)

    def test_a_noise_model_is_handed_on_when_the_stabilizer_takes_one(self):
        from soccer import stabilized
        noise = {'scale': .1, 'inflation': {'lowScore': 1.5}}
        calls = []
        result = stabilizer_result()
        summary = stabilized.attach(result, 1920, 1080, stabilize=stub_stabilize(calls), noise=noise)
        self.assertEqual((summary['status'], summary['noise']), ('ok', 'kept'))
        self.assertEqual(calls[0][3], noise)
        self.assertIsNot(calls[0][3], noise)                                       # a copy: the stabilizer may write into it
        self.assertAlmostEqual(result['stabilized']['ids']['A-01']['x'][5]*105.0-result['timeline']['ids']['A-01']['x'][5]*105.0, SHIFT+.1, delta=.02)
        self.assertEqual(stabilized.noise_of(result), {'scale': .1, 'inflation': {'lowScore': 1.5}})
        self.assertEqual(stabilized.attach(stabilizer_result(), 1920, 1080, stabilize=without_noise(stub_stabilize()), noise=noise)['noise'], 'refitted')
        self.assertIsNone(stabilized.noise_of(stabilizer_result()))

    @unittest.skipUnless(importlib.util.find_spec('soccer.trajectory') is not None, 'soccer/trajectory.py not written yet')
    def test_the_real_stabilizer_keeps_the_contract(self):
        from soccer import stabilized
        result = stabilizer_result()
        summary = stabilized.attach(result, 1920, 1080)
        self.assertEqual(summary['status'], 'ok', summary)
        block = result['stabilized']
        self.assertEqual(block['frameTimes'], result['timeline']['frameTimes'])
        self.assertEqual(set(block['ids']), set(result['timeline']['ids']))
        self.assertEqual((summary['rawSlots'], summary['rawFrames']), ([], 0))
        self.assertNotIn('runtimeSeconds', block['summary'])
        a01 = block['ids']['A-01']
        self.assertEqual(a01['kind'][5], 'O')
        self.assertEqual(a01['kind'][7], 'U')
        json.dumps(block, allow_nan=False)


@unittest.skipUnless(AVAILABLE, 'torch / ultralytics not installed')
class StabilizedRunTests(unittest.TestCase):
    def test_a_run_carries_the_block_and_survives_a_broken_stabilizer(self):
        from soccer import stabilized
        self.assertIs(pipeline.attach_stabilized, stabilized.attach)               # the older names, for the tools
        self.assertIs(pipeline.stabilized_block, stabilized.block)
        smoke = PipelineRunSmokeTests('test_a_720p_run_keeps_the_gate_closed')
        # The smoke clip has no registration: nobody to place, so no block and no stabilizer call.
        calls = []
        with stabilizer_module(stub_stabilize(calls)):
            _, files, _ = smoke.run_clip(1280, 720)
        result = files['result.json']
        self.assertEqual(result['summary']['stabilizer']['status'], 'skipped', result['summary']['stabilizer'])
        self.assertNotIn('stabilized', result)
        self.assertEqual(calls, [])
        # Its frames taken as registered on their way into the block, the stabilizer runs inside the run: its block
        # reaches result.json, and a stabilizer that raises leaves the run complete without one.
        real = stabilized.attach
        seen = []

        def registered(result, *args, **kwargs):
            # What the stabilizer is handed must be the finished result: the summary already lists the registration
            # (its anchors' sources and roots decide how the registration's snaps are spread), as result.json has it.
            seen.append('registration' in ((result.get('summary') or {}).get('pitch') or {}))
            for f in result['frames']:
                f['calibrated'] = True
            return real(result, *args, **kwargs)
        with stabilizer_module(stub_stabilize(calls)), mock.patch.object(stabilized, 'attach', registered):
            _, files, _ = smoke.run_clip(1280, 720)
        result = files['result.json']
        self.assertEqual(seen, [True])
        self.assertEqual(len(calls), 1)
        self.assertEqual(result['summary']['stabilizer']['status'], 'ok', result['summary']['stabilizer'])
        self.assertEqual(result['stabilized']['frameTimes'], result['timeline']['frameTimes'])
        self.assertEqual(set(result['stabilized']['ids']), set(result['timeline']['ids']))
        with stabilizer_module(failing), mock.patch.object(stabilized, 'attach', registered), mock.patch('traceback.print_exc'):
            _, files, _ = smoke.run_clip(1280, 720)
        result = files['result.json']
        self.assertEqual(result['status'], 'complete')
        self.assertNotIn('stabilized', result)
        self.assertEqual(result['summary']['stabilizer']['status'], 'failed')
        self.assertTrue(result['timeline']['frameTimes'])

    @unittest.skipUnless(importlib.util.find_spec('soccer.trajectory') is not None, 'soccer/trajectory.py not written yet')
    def test_a_run_with_the_real_stabilizer_says_why_it_has_no_block(self):
        _, files, _ = PipelineRunSmokeTests('test_a_720p_run_keeps_the_gate_closed').run_clip(1280, 720)
        result = files['result.json']
        self.assertEqual(result['summary']['stabilizer']['status'], 'skipped', result['summary']['stabilizer'])
        self.assertNotIn('stabilized', result)

if __name__ == '__main__':
    unittest.main()
