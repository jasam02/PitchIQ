"""The evaluation harness (tools/): the hold-out test rebuilds a keyframe from the other anchors only,
and the comparison of a run with itself is perfect agreement."""
import importlib.util
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import broadcast  # noqa: E402
from soccer import field  # noqa: E402

TOOLS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'tools')


def load_tool(name):
    spec = importlib.util.spec_from_file_location(f'tools_{name}', os.path.join(TOOLS, f'{name}.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


holdout = load_tool('holdout')
compare = load_tool('compare')
W, H = broadcast.W, broadcast.H
SIZE = field.PitchSize()
NAMES = ['centre-spot', 'centre-circle-far', 'centre-circle-near', 'halfway-far', 'halfway-near', 'right-penalty-box-front-far', 'corner-far-right']


def keyframe(P, time, index):
    marks = field.landmarks(SIZE)
    pts = field.project(P, [marks[n] for n in NAMES])
    return {'time': time, 'index': index, 'rms': 0.0, 'observability': .8,
            'points': [{'name': n, 'x': float(x/W), 'y': float(y/H)} for n, (x, y) in zip(NAMES, pts)]}


def record(n=30, bad_paint=False):
    P = field.orient(broadcast.camera(), W, H)
    frames = [{'t': i/25, 'dt': 0.0 if i == 0 else .04, 'M': np.eye(3).ravel().tolist(), 'reliable': True, 'cut': False, 'quality': 'flow'} for i in range(n)]
    anchors = [{'index': 0, 'time': 0.0, 'H': P.ravel().tolist(), 'observability': .8, 'source': 'landmarks', 'root': 0, 'details': {}},
               {'index': 20, 'time': .8, 'H': P.ravel().tolist(), 'observability': .8, 'source': 'landmarks', 'root': 20, 'details': {}}]
    if bad_paint:
        wrong = P.copy()
        wrong[0, 2] += 120   # a paint refinement rooted at keyframe 0 that shifted the view by 120 px
        anchors.append({'index': 10, 'time': .4, 'H': wrong.ravel().tolist(), 'observability': .8, 'source': 'paint', 'root': 0, 'details': {'score': .3}})
    return {'version': 1, 'width': W, 'height': H, 'fps': 25, 'offset': 0, 'size': SIZE.to_json(), 'frames': frames, 'anchors': anchors,
            'keyframes': [keyframe(P, 0.0, 0), keyframe(P, .8, 20)]}


class HoldoutTests(unittest.TestCase):
    def test_each_keyframe_is_rebuilt_from_the_other_one_alone(self):
        report = holdout.evaluate(record())
        self.assertEqual(len(report['keyframes']), 2)
        for k in report['keyframes']:
            self.assertEqual(k['status'], 'rebuilt')
            self.assertLess(k['pxMax'], .5)
            self.assertLess(k['metresMax'], .05)
            self.assertEqual(k['anchorsLeftOut'], 1)
            self.assertEqual(k['rebuiltFrom']['source'], 'carried')
        self.assertEqual(report['panStability']['maxPx'], 0.0)

    def test_a_derived_anchor_is_left_out_with_its_root(self):
        report = holdout.evaluate(record(bad_paint=True))
        first = report['keyframes'][0]
        self.assertEqual(first['anchorsLeftOut'], 2)       # the keyframe and the paint anchor rooted at it
        self.assertLess(first['pxMax'], .5)                 # rebuilt from keyframe 20 only: exact
        second = report['keyframes'][1]
        self.assertEqual(second['anchorsLeftOut'], 1)
        self.assertGreater(second['pxMax'], 50)             # the wrong paint anchor is part of the evidence for keyframe 20


class CompareTests(unittest.TestCase):
    def result(self):
        people = [{'track': 1, 'box': [.1, .2, .02, .06], 'state': 'confirmed', 'id': 'A-01', 'team': 'A', 'pitch': [.3, .4]},
                  {'track': 2, 'box': [.5, .5, .02, .06], 'state': 'uncertain', 'id': None, 'team': 'B', 'pitch': [.6, .7]},
                  {'track': 3, 'box': [.7, .5, .02, .06], 'state': 'candidate', 'id': None, 'team': None}]
        frames = [{'time': i/25, 'people': people, 'calibration': {'confidence': .8, 'disagreement': .5}} for i in range(5)]
        return {'video': {'height': 1080}, 'config': {'start': 0, 'seconds': .2}, 'frames': frames,
                'match': {'players': [{'id': 'A-01', 'team': 'A'}], 'goalkeepers': []},
                'summary': {'identifiedShare': .5, 'identity': {'idSwitches': 0}, 'pitch': {'meanConfidence': .8, 'registration': {'anchors': []}}, 'ball': {}},
                'tactical': {'pitchSize': {'length': 105, 'width': 68}}}

    def test_metrics_and_self_agreement(self):
        r = self.result()
        m = compare.metrics(r, 'a')
        self.assertEqual(m['perFrame'], {'tracked': 2.0, 'candidates': 1.0, 'drawnIdentified': 1.0, 'anonymousWithPosition': 1.0, 'trackedUndrawn': 1.0})
        self.assertEqual(m['identifiedByHeight']['60-90px'], .5)
        agree = compare.agreement(r, r)
        self.assertEqual(agree['boxAgreement'], 1.0)
        self.assertEqual(agree['labelAgreementExact'], 1.0)
        self.assertEqual(agree['labelAgreementMapped'], 1.0)
        self.assertEqual(agree['positionMeanMetres'], 0.0)
        self.assertIn('| perFrame.tracked | 2 |', compare.table([m]))


if __name__ == '__main__':
    unittest.main()
