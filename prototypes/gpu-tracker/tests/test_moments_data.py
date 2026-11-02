"""The key-moments data layer (soccer/moments/data.py and kits.py): owned frames turned into 5 Hz arrays (positions
smoothed within unbroken runs, velocities, unseen and uncertain positions left out), the possession record per sample
(settled spells, dead balls), the coach's roster corrections applied on load, the cache round trip, the camera's
footprint, and the kit colour names of the CAN-ISL prototypes.

Run: python -m unittest test_moments_data        (from this directory)
"""
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

import moments_synth as ms  # noqa: F401 - puts the gpu-tracker folder on the path
from soccer.moments import data as md
from soccer.moments import kits

L, W = 105.0, 68.0


def frames(n=100, speed=5.0, gap=None, unseen_at=None, low_conf_at=None):
    """Owned frames of one player running along x at `speed` m/s (25 fps), with an optional gap and bad frames."""
    out = []
    for k in range(n):
        x = 20.0+speed*k/25.0
        p = {'id': 'A-01', 'team': 'A', 'role': 'PLAYER', 'evidence': 'observed', 'pitch': [x/L, 30.0/W],
             'pitchConfidence': .9, 'pitchSigma': [.4, .5], 'identityConfidence': .85, 'box': [.1, .1, .01, .03]}
        if gap and gap[0] <= k < gap[1]:
            p = None
        elif unseen_at is not None and k == unseen_at:
            p['evidence'] = 'predicted'
        elif low_conf_at is not None and k == low_conf_at:
            p['pitchConfidence'] = .1
        out.append({'frame': 1000+k, 'time': 40.0+k/25.0, 'calibrated': True, 'pitchConfidence': .9,
                    'pitch': {'level': 'HIGH'}, 'view': [[0, 0], [.5, 0], [1, 0], [1, 1], [.5, 1], [0, 1]],
                    'ball': {'state': 'TRACKED', 'pitch': [.5, .5], 'confidence': .9, 'pitchConfidence': .9},
                    'calibration': {'H': [1, 0, 0, 0, 1, 0, 0, 0, 1]}, 'people': [p] if p else []})
    return out


def collect(fr, possession=None):
    col = md._Collector(L, W, {'A-01': {'display': 'A-01', 'team': 'A', 'role': 'PLAYER'}})
    for f in fr:
        col.add_frame(f, 0)
    arrays, runs, changes = col.finish(25.0, [{'index': 0, 'start': 0, 'end': 999, 'attacksRight': 'A'}], possession or {})
    return arrays, runs, col


class CollectTests(unittest.TestCase):
    def test_grid_positions_and_velocity(self):
        arrays, _, _ = collect(frames())
        self.assertEqual(len(arrays['t']), 20)                         # every 5th of 100 frames
        mid = slice(int(arrays['obsPtr'][10]), int(arrays['obsPtr'][11]))
        self.assertAlmostEqual(float(arrays['vx'][mid][0]), 5.0, places=3)
        self.assertAlmostEqual(float(arrays['vy'][mid][0]), 0.0, places=3)
        self.assertAlmostEqual(float(arrays['x'][mid][0]), 20.0+5.0*50/25.0, places=3)
        self.assertTrue(np.all(arrays['calibrated']))
        self.assertEqual(arrays['H'].shape, (20, 9))

    def test_a_gap_starts_a_new_run_and_unseen_or_uncertain_positions_are_left_out(self):
        arrays, _, col = collect(frames(gap=(40, 60), unseen_at=10, low_conf_at=15))
        per = np.diff(arrays['obsPtr'])
        self.assertEqual(int(per[2]), 0)                              # frame 10 (sample 2) was only predicted
        self.assertEqual(int(per[3]), 0)                              # frame 15: pitch confidence 0.1
        self.assertTrue(np.all(per[8:12] == 0))                        # the gap
        first_after = int(arrays['obsPtr'][12])
        self.assertTrue(np.isfinite(arrays['vx'][first_after]))        # one-sided at the run's start
        self.assertEqual(col.evidence['predicted'], 1)

    def test_possession_spells_and_settled_samples(self):
        timeline = [{'t0': 40.0, 't1': 41.0, 'team': 'A', 'state': 'control', 'carrier': 'A-01', 'confidence': 'medium'},
                    {'t0': 41.0, 't1': 42.6, 'team': 'A', 'state': 'flight', 'carrier': None, 'confidence': 'medium'},
                    {'t0': 42.6, 't1': 44.0, 'team': None, 'state': 'dead', 'carrier': None, 'confidence': 'medium'}]
        arrays, runs, _ = collect(frames(), {'timeline': timeline})
        self.assertEqual(runs, [{'team': 'A', 't0': 40.0, 't1': 42.6}])  # control + flight of one team: one spell
        t = arrays['t']
        settled = arrays['possSettled']
        self.assertTrue(settled[np.argmin(abs(t-41.4))])
        self.assertFalse(settled[np.argmin(abs(t-40.2))])               # within a second of the spell's start
        self.assertEqual(int(arrays['possState'][np.argmin(abs(t-43.0))]), md.POSS_CODES['dead'])
        self.assertEqual(int(arrays['possCarrier'][np.argmin(abs(t-40.4))]), 0)


class CorrectionsAndCacheTests(unittest.TestCase):
    def scene(self):
        s = ms.Scene(30, 'A')
        p1 = s.add('A', 30, 0)
        p2 = s.add('A', 40, 10)
        p3 = s.add('B', 50, 0)
        s.hide(p1, 25, 40)
        s.hide(p2, 10, 25)                              # never seen together with p1: a merge candidate
        return s, p1, p2, p3

    def test_merge_exclude_and_team_override(self):
        s, p1, p2, p3 = self.scene()
        data = s.build()
        rows_before = len(data.pid)
        md.apply_corrections(data, aliases={'A-02': 'A-01'}, excluded={'B-01'}, teams={'A-01': 'B'})
        self.assertEqual([p['id'] for p in data.people], ['A-01', 'B-01'])
        self.assertEqual(data.people[0]['team'], 'B')
        self.assertNotIn(1, set(data.pid.tolist()))     # B-01's rows are gone (it is index 1 after the merge)
        self.assertLess(len(data.pid), rows_before)
        self.assertEqual(int(data.obsPtr[-1]), len(data.pid))

    def test_save_and_load_round_trip(self):
        s, *_ = self.scene()
        data = s.build()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'data.npz'
            md.save(data, path)
            back = md.load(path)
        for k in md.SAMPLE_FIELDS+md.OBS_FIELDS:
            np.testing.assert_array_equal(np.asarray(getattr(back, k)), np.asarray(getattr(data, k)), err_msg=k)
        self.assertEqual(back.people, data.people)


class FootprintAndKitsTests(unittest.TestCase):
    def test_footprint_contains_and_share(self):
        half = np.array([[0, 0], [26, 0], [52.5, 0], [52.5, 68], [26, 68], [0, 68]], float)
        self.assertTrue(md.footprint_contains(half, np.array([10.0]), np.array([30.0]))[0])
        self.assertFalse(md.footprint_contains(half, np.array([80.0]), np.array([30.0]))[0])
        self.assertAlmostEqual(md.view_share(half, L, W), .5, delta=.05)
        self.assertEqual(md.view_share(np.full((6, 2), np.nan), L, W), 0.0)

    def test_the_can_isl_prototypes_are_white_and_red(self):
        a = [0.0144, 0.0562, 0.1796, 0.7066, 0.0005, 0.0016, 0.0016, 0.0046, 0.0076, 0.0213, 0.0003, 0.0009, 0.0, 0.0,
             0.0, 0.0, 0.0001, 0.0003, 0.0, 0.0001, 0.0001, 0.0006, 0.0005, 0.0031]
        b = [0.0025, 0.0086, 0.0033, 0.0075, 0.0838, 0.2222, 0.0161, 0.0247, 0.0023, 0.0037, 0.0, 0.0, 0.0, 0.0, 0.0,
             0.0, 0.0, 0.0, 0.0, 0.0, 0.0006, 0.0066, 0.13, 0.4881]
        self.assertEqual(kits.team_kits({'teams': {'prototypes': {'A': a, 'B': b}}}), {'A': 'white', 'B': 'red'})
        self.assertEqual(kits.team_kits({'teams': {'prototypes': {'A': a, 'B': a}}}), {'A': None, 'B': None})


if __name__ == '__main__':
    unittest.main()
