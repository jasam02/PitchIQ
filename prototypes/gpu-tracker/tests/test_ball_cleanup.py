"""ball_cleanup.py, the offline re-decode of a stored result: the camera motion it decodes with (the run's
registration.json first, the result's rounded per-frame motion otherwise) and the --small-scene override, which must
change the decode only, never the profile the written result keeps for the other offline passes."""
import contextlib
import io
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

import scene  # noqa: F401  (sys.path)
import ball_cleanup
from test_ball_path import FPS, H, W
from test_ball_path_small import frames_of, still_lock_scene

NO_FILES = SimpleNamespace(registration=None, motion=None, video=None)


def shift(dx):
    return [1.0, 0.0, float(dx), 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]


class CleanupTests(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.dir, True)

    def write(self, frames, registration=None):
        path = self.dir/'result.json'
        path.write_text(json.dumps({'video': {'width': W, 'height': H, 'fps': FPS}, 'frames': frames}), encoding='utf-8')
        if registration is not None:
            (self.dir/'registration.json').write_text(json.dumps(registration), encoding='utf-8')
        return path

    def test_motion_comes_from_the_registration_record_at_its_offset(self):
        frames = [{'time': i/FPS, 'motion': [1, 0, 2.5, 0, 1, 0], 'cameraReliable': True, 'cut': False} for i in range(6)]
        # The record starts two frames before the result (its offset): entry i+2 is frame i.
        log = [{'M': shift(10*k) if k != 4 else None, 'reliable': k != 4, 'cut': k == 5} for k in range(8)]
        path = self.write(frames, {'offset': 2, 'frames': log})
        motions = ball_cleanup.motions_of(json.loads(path.read_text(encoding='utf-8')), NO_FILES, path)
        self.assertEqual([m[0][0, 2] if m[0] is not None else None for m in motions], [20.0, 30.0, None, 50.0, 60.0, 70.0])
        self.assertEqual([(m[1], m[2]) for m in motions], [(True, False), (True, False), (False, False), (True, True), (True, False), (True, False)])
        self.assertTrue(all(m[0] is None or m[0].shape == (3, 3) for m in motions))
        # Without a record covering every frame, the result's own (rounded) motion.
        path = self.write(frames, {'offset': 4, 'frames': log})
        motions = ball_cleanup.motions_of(json.loads(path.read_text(encoding='utf-8')), NO_FILES, path)
        self.assertEqual([m[0][0, 2] for m in motions], [2.5]*6)
        (self.dir/'registration.json').unlink()
        motions = ball_cleanup.motions_of(json.loads(path.read_text(encoding='utf-8')), NO_FILES, path)
        self.assertEqual([m[0][0, 2] for m in motions], [2.5]*6)

    def run_main(self, *argv):
        with mock.patch.object(sys, 'argv', ['ball_cleanup.py', *map(str, argv)]), contextlib.redirect_stdout(io.StringIO()) as out:
            ball_cleanup.main()
        return out.getvalue()

    def test_a_forced_small_scene_gate_is_not_written_into_the_result(self):
        # The still-lock scene of test_ball_path_small, stored without a profile (an older result) on some frames and
        # with the causal pass's own on the others.
        frames = frames_of(150, still_lock_scene(), small=False)
        for i, f in enumerate(frames):
            if i % 2:
                del f['profile']
            else:
                f['profile'] = {'smallScene': False, 'note': i}
        before = [f.get('profile', 'absent') for f in frames]
        path = self.write(frames, {'offset': 0, 'frames': [{'M': shift(0), 'reliable': True, 'cut': False}]*150})
        printed = self.run_main(path, '--small-scene', 'on')
        self.assertIn('lock audit', printed)   # the forced gate was used for the decode ...
        cleaned = json.loads((self.dir/'result.cleaned.json').read_text(encoding='utf-8'))
        self.assertTrue([e for e in cleaned['events'] if e['message'].startswith('BALL LOCK AUDIT')])
        self.assertEqual([f.get('profile', 'absent') for f in cleaned['frames']], before)   # ... and only for it
        # auto: the frames' own gate (closed here); --dry writes nothing.
        (self.dir/'result.cleaned.json').unlink()
        printed = self.run_main(path, '--dry')
        self.assertNotIn('lock audit', printed)
        self.assertFalse((self.dir/'result.cleaned.json').exists())

    def test_the_override_helper_restores_the_profiles_even_when_the_decode_fails(self):
        frames = [{'time': 0.0, 'profile': {'smallScene': False}}, {'time': .04}]
        with mock.patch.object(ball_cleanup.ball_path, 'finalize', side_effect=RuntimeError('decode failed')):
            with self.assertRaises(RuntimeError):
                ball_cleanup.finalize(frames, [(np.eye(3), True, False)]*2, W, H, FPS, 'on')
        self.assertEqual(frames, [{'time': 0.0, 'profile': {'smallScene': False}}, {'time': .04}])


if __name__ == '__main__':
    unittest.main()
