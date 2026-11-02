"""Full Match Mode's roster corrections and merge suggestions (fullmatch/corrections.py) and their effect on the match
analytics (fullmatch/analytics.py).

What is checked: a merge of two fragments of one player is stored and followed through chains (A-12 into A-03, then A-03
into A-01: both are A-01); a merge of two people seen in the same frame, of different teams, or of a goalkeeper with a
player is refused with the reason; an exclusion and a team change; the served index merges the roster entries (seconds
summed, first/last seen, members) and carries the alias map; the match samples of merged identities are one track and an
excluded one is gone; possession carriers follow the merge; suggestions propose a fragment that starts soon after and
near where another ended, never one seen together with it nor one out of reach, never across teams.

Run:  python -m unittest test_fullmatch_corrections          (from this directory)
"""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fullmatch import analytics as ma, corrections as mc  # noqa: E402


def person(mid, team='A', role='PLAYER', first=0.0, last=100.0, seconds=50.0, history=None):
    return {'matchId': mid, 'display': mid, 'team': team, 'role': role, 'firstSeen': first, 'lastSeen': last, 'seconds': seconds,
            'members': [[0, mid]], 'segments': [0], 'how': 'one segment', 'confidence': 'medium',
            'history': history or [[int(first*25), .3, .3], [int(last*25), .3, .3]]}


INDEX = {'roster': [person('A-01', first=0, last=100), person('A-03', first=130, last=300), person('A-12', first=320, last=500),
                    person('A-05', first=0, last=500), person('B-01', team='B', first=110, last=500),
                    person('GK-1', role='GOALKEEPER', first=0, last=500)],
         'periods': [{'start': 0.0, 'end': 600.0}]}
TOGETHER = {mc.together_key('A-01', 'A-05'): 40, mc.together_key('A-03', 'A-05'): 10}


class CorrectionTests(unittest.TestCase):
    def setUp(self):
        self.folder = Path(tempfile.mkdtemp())
        (self.folder/'assembly').mkdir()

    def test_merges_chain_and_refusals_say_why(self):
        mc.add(self.folder, INDEX, TOGETHER, {'kind': 'merge', 'target': 'A-12', 'into': 'A-03'})
        mc.add(self.folder, INDEX, TOGETHER, {'kind': 'merge', 'target': 'A-03', 'into': 'A-01'})
        aliases, excluded, teams = mc.resolve_aliases(mc.load(self.folder))
        self.assertEqual(aliases, {'A-12': 'A-01', 'A-03': 'A-01'})
        with self.assertRaisesRegex(ValueError, 'same time'):
            mc.add(self.folder, INDEX, TOGETHER, {'kind': 'merge', 'target': 'A-05', 'into': 'A-12'})   # A-05 was with A-01
        with self.assertRaisesRegex(ValueError, 'different teams'):
            mc.add(self.folder, INDEX, TOGETHER, {'kind': 'merge', 'target': 'B-01', 'into': 'A-01'})
        with self.assertRaisesRegex(ValueError, 'goalkeeper'):
            mc.add(self.folder, INDEX, TOGETHER, {'kind': 'merge', 'target': 'GK-1', 'into': 'A-01'})
        with self.assertRaisesRegex(ValueError, 'already'):
            mc.add(self.folder, INDEX, TOGETHER, {'kind': 'merge', 'target': 'A-12', 'into': 'A-01'})

    def test_exclusion_team_change_and_the_served_index(self):
        mc.add(self.folder, INDEX, TOGETHER, {'kind': 'merge', 'target': 'A-12', 'into': 'A-03'})
        mc.add(self.folder, INDEX, TOGETHER, {'kind': 'exclude', 'target': 'A-05'})
        mc.add(self.folder, INDEX, TOGETHER, {'kind': 'team', 'target': 'B-01', 'team': 'A'})
        out = mc.apply_index(INDEX, mc.load(self.folder))
        roster = {p['matchId']: p for p in out['roster']}
        self.assertNotIn('A-12', roster)
        self.assertEqual(roster['A-03']['seconds'], 100.0)
        self.assertEqual((roster['A-03']['firstSeen'], roster['A-03']['lastSeen']), (130, 500))
        self.assertEqual(roster['A-03']['mergedFrom'], ['A-12'])
        self.assertEqual(roster['A-03']['how'], 'joined by you')
        self.assertTrue(roster['A-05']['excluded'])
        self.assertEqual(roster['B-01']['team'], 'A')
        self.assertEqual(out['aliases'], {'A-12': 'A-03'})
        items = mc.load(self.folder)
        mc.remove(self.folder, items[0]['id'])
        self.assertEqual(mc.resolve_aliases(mc.load(self.folder))[0], {})
        with self.assertRaises(ValueError):
            mc.remove(self.folder, 'nope')

    def test_suggestions_are_near_soon_after_never_together_nor_across_teams(self):
        roster = [person('A-01', first=0, last=100, history=[[0, .3, .3], [2500, .30, .30]]),
                  person('A-03', first=110, last=300, history=[[2750, .31, .31], [7500, .3, .3]]),     # 10 s later, 1.4 m away
                  person('A-04', first=105, last=300, history=[[2625, .9, .9], [7500, .9, .9]]),       # 5 s later but 70 m away
                  person('A-05', first=0, last=500, history=[[0, .3, .3], [12500, .3, .3]]),
                  person('A-06', first=120, last=300, history=[[3000, .3, .32], [7500, .3, .3]]),      # seen with A-01
                  person('B-02', team='B', first=101, last=300, history=[[2525, .3, .3], [7500, .3, .3]])]
        together = {mc.together_key('A-01', 'A-06'): 5}
        out = mc.suggestions(roster, together, [{'start': 0.0, 'end': 600.0}], 25.0)
        intos = [s['into'] for s in out.get('A-01', [])]
        self.assertEqual(intos[0], 'A-03')
        self.assertNotIn('A-04', intos)
        self.assertNotIn('A-06', intos)
        self.assertNotIn('B-02', intos)


class MatchAnalyticsWithCorrections(unittest.TestCase):
    def test_merged_identities_are_one_track_and_excluded_ones_are_gone(self):
        folder = Path(tempfile.mkdtemp())
        people = {'A-01': {'id': 'A-01', 'display': 'A-01', 'team': 'A', 'role': 'PLAYER'},
                  'A-12': {'id': 'A-12', 'display': 'A-12', 'team': 'A', 'role': 'PLAYER'},
                  'A-05': {'id': 'A-05', 'display': 'A-05', 'team': 'A', 'role': 'PLAYER'}}
        arrays = {}
        for pid, t0 in (('A-01', 0.0), ('A-12', 20.0), ('A-05', 0.0)):
            t = t0+np.arange(250)/25.0
            arrays.update({f'{pid}|t': t, f'{pid}|x': np.full(250, .3), f'{pid}|y': np.full(250, .4),
                           f'{pid}|identity': np.full(250, .9, np.float32), f'{pid}|quality': np.full(250, .8, np.float32),
                           f'{pid}|sigma': np.full(250, .5, np.float32)})
        meta = {'length': 105.0, 'width': 68.0, 'fps': 25.0, 'start': 0.0, 'end': 30.0, 'people': people, 'excluded': {}}
        path = folder/'samples.npz'
        np.savez_compressed(path, __meta__=np.frombuffer(json.dumps(meta).encode('utf-8'), np.uint8), **arrays)
        plain = ma.MatchSamples(path)
        self.assertEqual(set(plain.tracks), {'A-01', 'A-12', 'A-05'})
        merged = ma.MatchSamples(path, aliases={'A-12': 'A-01'}, excluded_ids={'A-05'})
        self.assertEqual(set(merged.tracks), {'A-01'})
        self.assertEqual(set(merged.people), {'A-01'})
        self.assertEqual(len(merged.tracks['A-01']['t']), 500)
        self.assertTrue(np.all(np.diff(merged.tracks['A-01']['t']) >= 0))
        self.assertIn('unseen', set(merged.tracks['A-01']['why']))     # the 10 s between the fragments is a break, not a run

    def test_possession_carriers_follow_the_merge(self):
        record = {'timeline': [{'t0': 0, 't1': 1, 'team': 'A', 'state': 'control', 'carrier': 'A-12'},
                               {'t0': 1, 't1': 2, 'team': 'A', 'state': 'control', 'carrier': 'A-05'}],
                  'changes': [{'frame': 25, 'lostBy': 'A-12', 'wonBy': 'B-01'}]}
        out = ma.possession_with(record, {'A-12': 'A-01'}, {'A-05'})
        self.assertEqual([r['carrier'] for r in out['timeline']], ['A-01', None])
        self.assertEqual(out['changes'][0]['lostBy'], 'A-01')
        self.assertEqual(record['timeline'][0]['carrier'], 'A-12')   # the stored record is not touched


if __name__ == '__main__':
    unittest.main()
