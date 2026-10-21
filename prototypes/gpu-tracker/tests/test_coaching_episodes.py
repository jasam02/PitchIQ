"""Honest counting of tactical situations (soccer/coaching/episodes.py): one continuous situation is one
occurrence, a value hovering at a threshold is not many, an unmeasured stretch is never bridged, and an id stays the
same when the same situation is found again.

Run:  python -m unittest test_coaching_episodes          (from this directory)
"""
import json
import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from soccer.coaching import episodes as ep, reference  # noqa: E402

RATE = 5.0


def grid(seconds, start=0.0):
    return start+np.arange(int(round(seconds*RATE)))/RATE


class EpisodeTests(unittest.TestCase):
    def test_a_continuous_ten_second_gap_is_one_event(self):
        t = grid(60)
        v = np.full(len(t), 10.0)
        v[(t >= 20) & (t < 30)] = 22.0                      # a 10 s spell above the threshold
        found = ep.episodes(t, v, np.ones(len(t), bool), enter=20.0, exit=18.0, min_seconds=5, merge_seconds=3)
        self.assertEqual(len(found), 1)
        self.assertAlmostEqual(found[0]['t0'], 20.0)
        self.assertAlmostEqual(found[0]['seconds'], 10.0)
        self.assertFalse(found[0]['censoredStart'] or found[0]['censoredEnd'])

    def test_a_situation_that_ends_and_reoccurs_is_two_events(self):
        t = grid(90)
        v = np.full(len(t), 10.0)
        v[(t >= 10) & (t < 20)] = 22.0
        v[(t >= 40) & (t < 50)] = 23.0                      # 20 s later, after a real reset
        found = ep.episodes(t, v, np.ones(len(t), bool), 20.0, 18.0, 5, 3)
        self.assertEqual([round(e['t0']) for e in found], [10, 40])
        self.assertEqual(found[1]['peak'], 23.0)

    def test_hovering_at_the_threshold_is_one_event_not_many(self):
        # Alternating 19 / 21 every sample around an enter of 20: sample by sample it would be 25 crossings.
        t = grid(10)
        v = np.where(np.arange(len(t)) % 2 == 0, 21.0, 19.0)
        found = ep.episodes(t, v, np.ones(len(t), bool), 20.0, 18.0, 5, 3)
        self.assertEqual(len(found), 1)

    def test_a_brief_interruption_is_merged_and_a_long_one_is_not(self):
        t = grid(60)
        v = np.full(len(t), 10.0)
        v[(t >= 10) & (t < 16)] = 22.0
        v[(t >= 18) & (t < 24)] = 22.0                      # 2 s below: the same situation
        v[(t >= 40) & (t < 46)] = 22.0                      # 16 s later: another one
        found = ep.episodes(t, v, np.ones(len(t), bool), 20.0, 18.0, 5, 3)
        self.assertEqual([round(e['t0']) for e in found], [10, 40])
        self.assertAlmostEqual(found[0]['t1'], 23.8)

    def test_a_passing_moment_is_not_counted(self):
        t = grid(30)
        v = np.full(len(t), 10.0)
        v[(t >= 10) & (t < 12)] = 30.0                      # 2 s
        self.assertEqual(ep.episodes(t, v, np.ones(len(t), bool), 20.0, 18.0, 5, 3), [])

    def test_an_unmeasured_stretch_ends_an_episode_and_is_never_bridged(self):
        t = grid(60)
        v = np.full(len(t), 10.0)
        v[(t >= 10) & (t < 40)] = 22.0
        ok = np.ones(len(t), bool)
        ok[(t >= 20) & (t < 30)] = False                    # 10 s off camera in the middle
        found = ep.episodes(t, v, ok, 20.0, 18.0, 5, 3)
        self.assertEqual(len(found), 2)
        self.assertTrue(found[0]['censoredEnd'])
        self.assertTrue(found[1]['censoredStart'])
        self.assertAlmostEqual(found[1]['t0'], 30.0)
        short = ok.copy()
        short[:] = True
        short[(t >= 20) & (t < 21)] = False                 # 1 s: bridged
        self.assertEqual(len(ep.episodes(t, v, short, 20.0, 18.0, 5, 3)), 1)

    def test_an_episode_running_at_the_data_edges_is_censored(self):
        t = grid(20)
        v = np.full(len(t), 22.0)
        found = ep.episodes(t, v, np.ones(len(t), bool), 20.0, 18.0, 5, 3)
        self.assertEqual(len(found), 1)
        self.assertTrue(found[0]['censoredStart'] and found[0]['censoredEnd'])

    def test_below_mirrors_above(self):
        t = grid(40)
        v = np.full(len(t), 10.0)
        v[(t >= 10) & (t < 20)] = 2.0
        found = ep.episodes(t, v, np.ones(len(t), bool), enter=4.0, exit=6.0, direction='below')
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0]['peak'], 2.0)
        with self.assertRaises(ValueError):
            ep.episodes(t, v, np.ones(len(t), bool), enter=4.0, exit=3.0, direction='below')

    def test_nan_values_are_unmeasured(self):
        t = grid(30)
        v = np.full(len(t), 22.0)
        v[(t >= 10) & (t < 20)] = np.nan
        found = ep.episodes(t, v, np.ones(len(t), bool), 20.0, 18.0, 5, 3)
        self.assertEqual(len(found), 2)

    def test_measured_time_stretches_and_rates(self):
        t = grid(60)
        ok = np.ones(len(t), bool)
        ok[(t >= 20) & (t < 30)] = False
        self.assertAlmostEqual(ep.measured_seconds(t, ok), 50.0)
        self.assertEqual([(round(a, 1), round(b, 1)) for a, b in ep.stretches(t, ok)], [(0.0, 19.8), (30.0, 59.8)])
        self.assertAlmostEqual(ep.longest_stretch(t, ok), 30.0)
        self.assertAlmostEqual(ep.rate_per_minute(3, 120.0), 1.5)
        self.assertIsNone(ep.rate_per_minute(3, 0))
        self.assertAlmostEqual(ep.share_beyond(np.r_[1, 2, 3, 4.0], np.ones(4, bool), 3.0), .5)

    def test_ids_are_stable_and_distinct(self):
        a = ep.stable_id('occ', 'unit_gap_dm', 'A', 20.0, 30.0)
        self.assertEqual(a, ep.stable_id('occ', 'unit_gap_dm', 'A', 20.04, 29.96))   # an edge moved by a fraction
        self.assertNotEqual(a, ep.stable_id('occ', 'unit_gap_dm', 'B', 20.0, 30.0))
        self.assertTrue(a.startswith('occ-') and len(a) == 16)


class ReferenceTests(unittest.TestCase):
    def setUp(self):
        q = {'n': 5000, 'p05': 5, 'p10': 7, 'p25': 10, 'p50': 13, 'p75': 16, 'p90': 20, 'p95': 22, 'p975': 24}
        small = dict(q, n=100, p90=99)
        self.ref = {'version': 1, 'source': 'test', 'rules': {'default': {'enter': 'p95', 'exit': 'p90', 'minSeconds': 5, 'mergeSeconds': 3}},
                    'metrics': {'gapDM': {'unit': 'm', 'contexts': {'all': q, 'third:final': dict(q, p90=27), 'third:own': small},
                                          'episodes': {'all': {'default': {'perMinute': {'p10': .05, 'p50': .1, 'p90': .2}}}}}}}

    def test_quantiles_fall_back_to_all_without_enough_samples(self):
        self.assertEqual(reference.quantile(self.ref, 'gapDM', 'p90', 'third:final'), (27.0, 'third:final'))
        self.assertEqual(reference.quantile(self.ref, 'gapDM', 'p90', 'third:own'), (20.0, 'all'))   # 100 samples
        self.assertEqual(reference.quantile(self.ref, 'gapDM', 'p90', 'third:middle'), (20.0, 'all'))
        self.assertEqual(reference.quantile(self.ref, 'width', 'p90'), (None, None))

    def test_episode_baselines_and_rules(self):
        b = reference.episode_baseline(self.ref, 'gapDM', 'default', 'third:final')
        self.assertEqual((b['context'], b['perMinute']['p90']), ('all', .2))
        self.assertIsNone(reference.episode_baseline(self.ref, 'width'))
        self.assertEqual(reference.rule(self.ref)['enter'], 'p95')
        with self.assertRaises(KeyError):
            reference.rule(self.ref, 'other')

    def test_the_file_is_loaded_once_and_identified(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, 'reference.json')
            with open(p, 'w', encoding='utf-8') as f:
                json.dump(self.ref, f)
            a = reference.load_reference(p)
            self.assertIs(a, reference.load_reference(p))
            self.assertTrue(reference.reference_id(a).startswith('test@'))
            with self.assertRaises(FileNotFoundError):
                reference.load_reference(os.path.join(d, 'missing.json'))


if __name__ == '__main__':
    unittest.main()
