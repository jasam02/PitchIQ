"""The halftime of a full match (fullmatch/halftime.py): defaults from the plan's break and the inferred direction, the
coach's settings validated, the direction periods by time (a halftime inside a processing segment too), team_shape's
direction, the per-sample periods of the match data with the break left out, persistence, and what it never changes
(positions, identities).

Run:  python -m unittest test_intel_halftime          (from this directory)
"""
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
from fullmatch import halftime as ht  # noqa: E402
import intel_synth as S  # noqa: E402


def index_two_halves(first='B'):
    second = 'A' if first == 'B' else 'B'
    return {'periods': [{'index': 0, 'start': 10.0, 'end': 2854.0, 'name': 'first half', 'attacksRight': first,
                         'defendsLeft': first, 'directionAgreement': .97},
                        {'index': 1, 'start': 2885.0, 'end': 5791.7, 'name': 'second half', 'attacksRight': second,
                         'defendsLeft': second, 'directionAgreement': .85}]}


class Defaults(unittest.TestCase):
    def test_plan_gap_and_inferred_direction(self):
        d = ht.defaults(index_two_halves('B'))                    # a 31 s gap the recording left out: one marker
        self.assertEqual((d['mode'], d['firstHalfEnd'], d['secondHalfStart']), ('marker', 2854.0, 2854.0))
        self.assertEqual(d['teamAFirstHalf'], 'left')            # B attacks right in the first half: A attacks left
        self.assertTrue(d['inferred']['consistent'])
        self.assertFalse(d['confirmed'])

    def test_a_recorded_break_gives_two_times(self):
        index = index_two_halves('B')
        index['periods'][1]['start'] = 3800.0                      # the 16-minute break was recorded
        d = ht.defaults(index)
        self.assertEqual((d['mode'], d['firstHalfEnd'], d['secondHalfStart']), ('break', 2854.0, 3800.0))
        self.assertEqual(ht.interval(d), (2854.0, 3800.0))

    def test_one_period_has_no_halftime_until_set(self):
        d = ht.defaults({'periods': [{'index': 0, 'start': 0.0, 'end': 600.0, 'attacksRight': 'A'}]})
        self.assertIsNone(d['firstHalfEnd'])
        self.assertIsNone(ht.interval(d))
        self.assertEqual(d['teamAFirstHalf'], 'right')


class Clean(unittest.TestCase):
    def setUp(self):
        self.index = index_two_halves()

    def test_marker_mode_is_one_time(self):
        s = ht.clean(self.index, {'mode': 'marker', 'firstHalfEnd': 2900, 'secondHalfStart': 3000, 'teamAFirstHalf': 'left'})
        self.assertEqual((s['firstHalfEnd'], s['secondHalfStart']), (2900.0, 2900.0))
        self.assertTrue(s['confirmed'])

    def test_break_mode_keeps_the_interval(self):
        s = ht.clean(self.index, {'mode': 'break', 'firstHalfEnd': 2850, 'secondHalfStart': 3790, 'teamAFirstHalf': 'right'})
        self.assertEqual(ht.interval(s), (2850.0, 3790.0))

    def test_refusals(self):
        bad = [{'mode': 'break', 'firstHalfEnd': 3000, 'secondHalfStart': 2900, 'teamAFirstHalf': 'left'},   # backwards
               {'mode': 'marker', 'firstHalfEnd': 9999, 'teamAFirstHalf': 'left'},                       # outside
               {'mode': 'marker', 'firstHalfEnd': 5, 'teamAFirstHalf': 'left'},
               {'mode': 'marker', 'firstHalfEnd': 2900, 'teamAFirstHalf': 'up'},                         # direction
               {'mode': 'sideways', 'firstHalfEnd': 2900, 'teamAFirstHalf': 'left'},
               {'mode': 'marker', 'firstHalfEnd': 'soon', 'teamAFirstHalf': 'left'}]
        for body in bad:
            with self.assertRaises(ValueError, msg=str(body)):
                ht.clean(self.index, body)


class Periods(unittest.TestCase):
    def test_direction_flips_at_the_marker(self):
        index = index_two_halves()
        s = {'mode': 'marker', 'firstHalfEnd': 2700.0, 'secondHalfStart': 2700.0, 'teamAFirstHalf': 'right'}
        p = ht.periods(index, s)
        self.assertEqual([(q['start'], q['end'], q['attacksRight']) for q in p], [(10.0, 2700.0, 'A'), (2700.0, 5791.7, 'B')])
        self.assertEqual(p[0]['defendsLeft'], 'A')

    def test_break_interval_is_between_the_halves(self):
        index = index_two_halves()
        s = {'mode': 'break', 'firstHalfEnd': 2800.0, 'secondHalfStart': 2950.0, 'teamAFirstHalf': 'left'}
        p = ht.periods(index, s)
        self.assertEqual((p[0]['end'], p[1]['start']), (2800.0, 2950.0))
        self.assertEqual((p[0]['attacksRight'], p[1]['attacksRight']), ('B', 'A'))

    def test_team_shape_direction(self):
        index = index_two_halves()
        s = {'mode': 'marker', 'firstHalfEnd': 2700.0, 'secondHalfStart': 2700.0, 'teamAFirstHalf': 'right'}
        self.assertEqual(ht.direction_settings(index, s), {'A': 'left', 'switchAt': 2700.0})   # A defends the left goal
        s['teamAFirstHalf'] = 'left'
        self.assertEqual(ht.direction_settings(index, s)['A'], 'right')

    def test_fingerprint_follows_the_settings(self):
        a = {'mode': 'marker', 'firstHalfEnd': 2700.0, 'secondHalfStart': 2700.0, 'teamAFirstHalf': 'right'}
        b = dict(a, firstHalfEnd=2701.0, secondHalfStart=2701.0)
        self.assertNotEqual(ht.fingerprint(a), ht.fingerprint(b))
        self.assertEqual(ht.fingerprint(a), ht.fingerprint(dict(a, confirmed=True, at='x')))


class MatchData(unittest.TestCase):
    def setUp(self):
        m = S.Match(seconds=120.0, halftime=(60.0, 70.0))
        m.team('A')
        m.team('B')
        self.data, self.index = m.build(segment_s=20.0)

    def test_periods_by_time_inside_a_segment(self):
        x0, y0, pid0 = self.data.x.copy(), self.data.y.copy(), self.data.pid.copy()
        s = {'mode': 'marker', 'firstHalfEnd': 45.0, 'secondHalfStart': 45.0, 'teamAFirstHalf': 'left'}
        ht.apply_to_data(self.data, self.index, s)
        t = self.data.t
        self.assertTrue(np.all(self.data.period[t < 45.0] == 0))
        self.assertTrue(np.all(self.data.period[t > 45.0] == 1))
        self.assertEqual(int(self.data.segment[np.searchsorted(t, 45.0)]), 1)   # the marker inside the second segment
        self.assertEqual(self.data.periods[0]['attacksRight'], 'B')
        self.assertEqual(self.data.periods[1]['attacksRight'], 'A')
        # never a position, never an identity
        np.testing.assert_array_equal(self.data.x, x0)
        np.testing.assert_array_equal(self.data.y, y0)
        np.testing.assert_array_equal(self.data.pid, pid0)

    def test_the_break_has_no_period(self):
        s = {'mode': 'break', 'firstHalfEnd': 60.0, 'secondHalfStart': 70.0, 'teamAFirstHalf': 'right'}
        ht.apply_to_data(self.data, self.index, s)
        t = self.data.t
        inside = (t > 60.0) & (t < 70.0)
        self.assertTrue(inside.any())
        self.assertTrue(np.all(self.data.period[inside] == -1))
        # every sample keeps its place in the timeline: none duplicated, none dropped
        self.assertEqual(len(self.data.period), len(t))
        self.assertTrue(np.all(np.diff(t) > 0))


class Persistence(unittest.TestCase):
    def test_saved_settings_come_back_and_reset(self):
        index = index_two_halves()
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder/'assembly').mkdir()
            s = ht.clean(index, {'mode': 'break', 'firstHalfEnd': 2840, 'secondHalfStart': 2890, 'teamAFirstHalf': 'right'})
            ht.save(folder, s)
            back = ht.load(folder, index)                   # the match opened again
            self.assertEqual((back['firstHalfEnd'], back['secondHalfStart'], back['teamAFirstHalf'], back['confirmed']),
                             (2840.0, 2890.0, 'right', True))
            ht.reset(folder)
            again = ht.load(folder, index)
            self.assertEqual((again['firstHalfEnd'], again['teamAFirstHalf'], again['confirmed']), (2854.0, 'left', False))


if __name__ == '__main__':
    unittest.main()
