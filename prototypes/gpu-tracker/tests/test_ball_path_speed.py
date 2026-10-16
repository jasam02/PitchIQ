"""The vectorized offline decoder (speed pass 3) against a frozen copy of the one-source-frame-at-a-time
decode it replaced: the same path, node for node, with the same link costs and speeds bit for bit, on
synthetic layers shaped like a real result (tests/ball_synth.py) and on small random cases with camera
cuts, unreliable frames, empty frames and ties; and finalize() unchanged end to end."""
import copy
import json
import math
import os
import random
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ball_synth  # noqa: E402
from ball_path_reference import decode_reference  # noqa: E402
from soccer import ball_path  # noqa: E402

W, H = ball_synth.W, ball_synth.H


def layers_of(frames, motions, fps, to_pitch=None):
    layers = [ball_path.nodes_of(i, f, W, H, to_pitch) for i, f in enumerate(frames)]
    ball_path._attachment(layers, fps)
    times = [f['time'] for f in frames]
    T, _, _ = ball_path._chain(motions)
    ball_path._velocities(layers, times, T)
    ball_path._standing_markings(layers, times, T)
    return layers, times


class SideBySideTests(unittest.TestCase):
    def check(self, layers, times, motions, fps):
        expected = decode_reference(layers, times, motions, fps)
        actual = ball_path.decode(layers, times, motions, fps)
        same, why = ball_synth.same_path(expected, actual)
        self.assertTrue(same, why)
        return actual

    def test_realistic_layers_decode_to_the_same_path(self):
        # 400 frames of broadcast-like candidates (2 to 10 a frame, boots, a spot, blips, a cut, unreliable frames),
        # with and without a trusted registration (painted-spot markings).
        for seed, registered in ((0, False), (1, True)):
            frames, motions, to_pitch = ball_synth.synthetic(seed, 400, 25.0, registered=registered)
            layers, times = layers_of(frames, motions, 25.0, to_pitch)
            self.assertGreaterEqual(max(len(l) for l in layers), 6)
            path = self.check(layers, times, motions, 25.0)
            self.assertGreater(sum(1 for p in path if p is not None), 100)

    def test_small_random_cases_with_cuts_ties_and_empty_frames(self):
        rng = random.Random(7)
        for case in range(60):
            fps = rng.choice([15.0, 25.0, 30.0, 50.0])
            n = rng.randint(1, 70)
            frames, motions = [], []
            tracklets = list(range(1, 6))
            for i in range(n):
                cands = []
                for _ in range(rng.choice([0, 0, 1, 1, 2, 3, 4])):
                    x, y = rng.uniform(0, W), rng.uniform(300, H)
                    d = rng.choice([6.0, 9.0, 12.0])
                    c = {'box': [(x-d/2)/W, (y-d/2)/H, d/W, d/H], 'det': rng.choice([.1, .3, .6, .9]), 'status': rng.choice(['ball', 'candidate', 'rejected']),
                         'scores': {'size': .8, 'shape': .7, 'isolation': .9, 'field': 1.0, 'line': rng.choice([0.0, .5]), 'attach': rng.choice([0.0, .4]),
                                    'static': rng.choice([0.0, 1.0]), 'otherObject': rng.random() < .2},
                         'tracklet': rng.choice(tracklets+[None]), 'pxm': rng.choice([30.0, 40.0, 55.0])}
                    if rng.random() < .3:
                        c['reason'] = rng.choice(['JUMP', 'STATIC', 'TRAJECTORY', 'SIZE'])
                    cands.append(c)
                # A couple of players whose lower bodies may hold candidates; ties come from identical positions.
                people = [{'track': k, 'box': [rng.uniform(0, .9), rng.uniform(.3, .8), .03, .1]} for k in (1, 2, 3)]
                if cands and rng.random() < .3:
                    cands.append(dict(cands[0], tracklet=cands[0]['tracklet'] if rng.random() < .5 else None))   # the same spot twice: a tie
                frames.append({'time': i/fps, 'people': people, 'ballCandidates': cands, 'ball': {'state': 'UNKNOWN', 'phase': 'SEARCHING', 'confidence': 0.0}})
                reliable = rng.random() > .2
                M = np.array([[1, 0, rng.uniform(-30, 30)], [0, 1, rng.uniform(-5, 5)], [0, 0, 1]], float)
                motions.append((M if reliable else None, reliable, rng.random() < .05))
            layers, times = layers_of(frames, motions, fps)
            self.check(layers, times, motions, fps)
        self.assertEqual(ball_path.decode([], [], [], 25.0), [])

    def test_finalize_is_unchanged_end_to_end(self):
        frames, motions, to_pitch = ball_synth.synthetic(2, 300, 25.0, registered=True)
        outputs = {}
        original = ball_path.decode
        try:
            for name, fn in (('reference', decode_reference), ('new', original)):
                ball_path.decode = fn
                copied = copy.deepcopy(frames)
                report = ball_path.finalize(copied, motions, W, H, 25.0, to_pitch)
                outputs[name] = (json.dumps([f['ball'] for f in copied], sort_keys=True), json.dumps(report, sort_keys=True, default=str))
        finally:
            ball_path.decode = original
        self.assertEqual(outputs['reference'][0], outputs['new'][0])
        self.assertEqual(outputs['reference'][1], outputs['new'][1])

    def test_an_undefined_link_offers_nothing_from_its_frame_as_before(self):
        # A degenerate camera chain puts a node at infinity: its link costs are NaN. The one-at-a-time decode let
        # such a frame contribute nothing to that candidate; so does the stacked one.
        frames = [{'time': i/25, 'people': [], 'ballCandidates': [{'box': [.3, .5, .005, .008], 'det': .8, 'status': 'ball', 'scores': {}, 'tracklet': 1, 'pxm': 40.0},
                                                                   {'box': [.6, .5, .005, .008], 'det': .7, 'status': 'candidate', 'scores': {}, 'tracklet': 2, 'pxm': 40.0}],
                   'ball': {'state': 'UNKNOWN', 'phase': 'SEARCHING', 'confidence': 0.0}} for i in range(6)]
        motions = [(np.eye(3), True, False)]*6
        layers, times = layers_of(frames, motions, 25.0)
        for node in layers[2]:
            node.x, node.y = math.inf, math.inf
        with np.errstate(invalid='ignore'):
            self.check(layers, times, motions, 25.0)


if __name__ == '__main__':
    unittest.main()
