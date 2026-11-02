"""The match's people and places (fullmatch/persons.py) on synthetic matches (tests/intel_synth.py): the fifteen
identity cases of the Full Match Intelligence spec, plus what is never a player, the slot labels, the coach's
corrections and the index the page reads.

The places are ten time-ordered paths per team (a min-cost flow); a fragment that continues a place shows its slot with
'?' (an estimate), the one that began it the plain slot, a fragment no place takes 'A-?'. Nothing joins two people
automatically (measured on CAN-ISL: even confident links were right only 54-74% of the time); the coach's merges and
confirmations do.

Run:  python -m unittest test_intel_persons          (from this directory)
"""
import os
import re
import sys
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
from fullmatch import corrections, persons  # noqa: E402
import intel_synth as S  # noqa: E402

LABEL = re.compile(r'^(?:[AB]-(?:0[1-9]|10)\??|[AB]-\?|GK-[AB]\??|\?)$')


def place_of(res, fragment):
    for team, plist in res['places'].items():
        for pl in plist:
            for j, node in enumerate(pl['nodes']):
                if fragment in node['fragments']:
                    return team, pl['slot'], j
    return None


def display(res, fragment):
    return res['people'][res['person'][fragment]]['display']


def check_structure(test, data, res):
    """No two fragments seen at one sample share a place; no display beyond slot 10."""
    for team, plist in res['places'].items():
        test.assertLessEqual(len(plist), 10)
        for pl in plist:
            frs = [f for node in pl['nodes'] for f in node['fragments']]
            samples = []
            for f in frs:
                i = data.index_of[f]
                samples.append(set(data.obs_sample()[data.pid == i].tolist()))
            for a in range(len(samples)):
                for b in range(a+1, len(samples)):
                    test.assertFalse(samples[a] & samples[b], f'{frs[a]} and {frs[b]} were seen together in one place')
    for p in res['people'].values():
        test.assertRegex(p['display'], LABEL)


class Gaps(unittest.TestCase):
    def test_1_2_a_player_lost_2_s_and_20_s_keeps_his_place(self):
        m = S.Match(seconds=200)
        a = m.team('A')
        m.team('B')
        lb_ids = m.scene.people[a['LB']]['id']
        lb2 = m.cut(a['LB'], 80.0, gap=2.0)
        rcm2 = m.cut(a['RCM'], 120.0, gap=20.0)
        data, index, auto, res = S.people_of(m)
        check_structure(self, data, res)
        lb, rcm = lb_ids, m.scene.people[a['RCM']]['id']
        for first, later in ((lb, m.scene.people[lb2]['id']), (rcm, m.scene.people[rcm2]['id'])):
            p0, p1 = place_of(res, first), place_of(res, later)
            self.assertEqual(p0[:2], p1[:2], f'{later} should continue {first}\'s place')
            self.assertEqual(display(res, first), f'A-{p0[1]:02d}')
            self.assertEqual(display(res, later), f'A-{p0[1]:02d}?')       # an estimate, said so
            self.assertNotEqual(res['person'][first], res['person'][later])  # never joined automatically

    def test_3_a_camera_cut_keeps_every_place(self):
        m = S.Match(seconds=200)
        a = m.team('A')
        b = m.team('B')
        later = {}
        for ids in (a, b):
            for role, pid in ids.items():
                if role != 'GK':
                    later[m.scene.people[pid]['id']] = m.scene.people[m.cut(pid, 100.0, gap=4.0)]['id']
        data, index, auto, res = S.people_of(m)
        check_structure(self, data, res)
        for first, nxt in later.items():
            self.assertEqual(place_of(res, first)[:2], place_of(res, nxt)[:2], f'{nxt} should continue {first}')

    def test_4_5_teammates_crossing_never_share_a_place(self):
        m = S.Match(seconds=200)
        a = m.team('A')
        m.team('B')
        s = m.scene
        # LCM and RCM cross paths (swap sides) between 90 and 110 s and are both lost at 110 s for 3 s
        lcm, rcm = a['LCM'], a['RCM']
        k0, k1 = s.k(90.0), s.k(110.0)
        ya, yb = s.pos[lcm][k0, 1], s.pos[rcm][k0, 1]
        w = np.linspace(0, 1, k1-k0)
        s.pos[lcm][k0:k1, 1] = ya*(1-w)+yb*w
        s.pos[rcm][k0:k1, 1] = yb*(1-w)+ya*w
        s.pos[lcm][k1:, 1] = s.pos[lcm][k1:, 1]-ya+yb
        s.pos[rcm][k1:, 1] = s.pos[rcm][k1:, 1]-yb+ya
        l2, r2 = m.cut(lcm, 110.0, gap=3.0), m.cut(rcm, 110.0, gap=3.0)
        data, index, auto, res = S.people_of(m)
        check_structure(self, data, res)
        for f in (s.people[l2]['id'], s.people[r2]['id']):
            self.assertTrue(display(res, f).endswith('?'))                  # a continuation is never claimed sure
        # two people seen together are never one: the coach's merge is refused too
        idx = dict(index)
        together = {'|'.join(sorted((s.people[lcm]['id'], s.people[rcm]['id']))): 50}
        self.assertIn('same time', corrections.check_merge(idx, together, [], s.people[lcm]['id'], s.people[rcm]['id']))


class NotPlayers(unittest.TestCase):
    def test_6_7_an_official_and_a_warming_up_substitute_are_left_out(self):
        m = S.Match(seconds=200)
        m.team('A')
        m.team('B')
        ref = m.person('A', 52.0, 30.0, 20.0, 150.0, kit=S.OTHER_KIT)         # the runs took him for a team-A player
        warm = m.person('A', 50.0, 70.5, 20.0, 150.0)                        # beyond the touchline
        overlay = m.overlay('A', 30.0, 120.0)                                # a broadcast graphic
        data, index, auto, res = S.people_of(m)
        flags = {f: v['flag'] for f, v in auto['fragments'].items() if v['flag']}
        self.assertEqual(flags.get(m.scene.people[ref]['id']), 'kit')
        self.assertEqual(flags.get(m.scene.people[warm]['id']), 'offpitch')
        self.assertEqual(flags.get(m.scene.people[overlay]['id']), 'overlay')
        for pid in (ref, warm, overlay):
            f = m.scene.people[pid]['id']
            self.assertIn(f, res['excluded'])
            self.assertIsNone(place_of(res, f))
        # the coach can take one back
        res2 = persons.resolve(auto, [{'kind': 'include', 'target': m.scene.people[warm]['id']}])
        self.assertNotIn(m.scene.people[warm]['id'], res2['excluded'])


class Substitutions(unittest.TestCase):
    def test_8_a_substitute_is_a_new_person_in_the_freed_slot(self):
        m = S.Match(seconds=240)
        a = m.team('A')
        m.team('B')
        s = m.scene
        rm = a['RM']
        s.hide(rm, 150.0, 1e9)                                              # the right midfielder goes off
        sub = m.cut(rm, 240.0, gap=0.0)                                     # (an empty fragment, not used)
        new = m.person('A', 0.0, 0.0, 0.0, 0.0)                              # the substitute, placed below
        s.pos[new][s.k(158.0):] = s.pos[a['RCM']][s.k(158.0):]+np.array([2.0, 15.0])    # where a right midfielder plays
        data, index, auto, res = S.people_of(m)
        rm_id, new_id = s.people[rm]['id'], s.people[new]['id']
        self.assertNotEqual(res['person'][rm_id], res['person'][new_id])      # two people
        self.assertEqual(place_of(res, rm_id)[:2], place_of(res, new_id)[:2])  # the incoming one takes the slot
        self.assertTrue(display(res, new_id).endswith('?'))
        confirmed = persons.resolve(auto, [{'kind': 'substitution', 'out': rm_id, 'in': new_id, 'status': 'confirmed'}])
        self.assertNotEqual(confirmed['person'][rm_id], confirmed['person'][new_id])
        self.assertEqual(confirmed['substitutions'][0]['status'], 'confirmed')
        # the slot passes to the substitute, shown sure, and the two stay two people
        self.assertEqual(place_of(confirmed, rm_id)[:2], place_of(confirmed, new_id)[:2])
        slot = place_of(confirmed, rm_id)[1]
        self.assertEqual(display(confirmed, new_id), f'A-{slot:02d}')
        self.assertEqual(display(confirmed, rm_id), f'A-{slot:02d}')
        del sub

    def test_two_substitutes_are_never_merged(self):
        m = S.Match(seconds=240)
        a = m.team('A')
        m.team('B')
        s = m.scene
        incoming = []
        for role in ('LM', 'RS'):
            pid = a[role]
            s.hide(pid, 140.0, 1e9)
            new = m.person('A', 0.0, 0.0, 0.0, 0.0)
            s.pos[new][s.k(146.0):] = s.pos[a['LCM' if role == 'LM' else 'LS']][s.k(146.0):]+np.array([2.0, -15.0 if role == 'LM' else 14.0])
            incoming.append(s.people[new]['id'])
        data, index, auto, res = S.people_of(m)
        self.assertNotEqual(res['person'][incoming[0]], res['person'][incoming[1]])
        check_structure(self, data, res)


class Periods(unittest.TestCase):
    def test_9_a_player_keeps_his_place_after_halftime(self):
        m = S.Match(seconds=300, halftime=(140.0, 160.0))
        a = m.team('A')
        m.team('B')
        later = {}
        for role, pid in a.items():
            if role == 'GK':
                continue
            later[m.scene.people[pid]['id']] = m.scene.people[m.cut(pid, 150.0, gap=0.0)]['id']
        data, index, auto, res = S.people_of(m)
        check_structure(self, data, res)
        kept = sum(place_of(res, f)[:2] == place_of(res, nxt)[:2] for f, nxt in later.items())
        self.assertGreaterEqual(kept, 8, 'most players should continue their place across halftime')

    def test_10_segment_boundaries_make_no_new_labels(self):
        m = S.Match(seconds=300)
        a = m.team('A')
        b = m.team('B')
        for ids in (a, b):
            for role, pid in ids.items():
                cur = pid
                for t in (60.0, 120.0, 180.0, 240.0):
                    cur = m.cut(cur, t, gap=0.0)
        data, index, auto, res = S.people_of(m)
        check_structure(self, data, res)
        numbers = {int(p['display'][2:4]) for p in res['people'].values() if re.match(r'^[AB]-\d\d', p['display'])}
        self.assertTrue(numbers <= set(range(1, 11)))


class Keepers(unittest.TestCase):
    def test_11_a_keeper_lost_and_found_is_one_person(self):
        m = S.Match(seconds=200)
        a = m.team('A')
        m.team('B')
        gk2 = m.cut(a['GK'], 90.0, gap=30.0)
        data, index, auto, res = S.people_of(m)
        g1, g2 = m.scene.people[a['GK']]['id'], m.scene.people[gk2]['id']
        self.assertEqual(res['person'][g1], res['person'][g2])
        self.assertEqual(display(res, g1), 'GK-A')

    def test_12_a_replaced_keeper_is_two_people(self):
        m = S.Match(seconds=200)
        a = m.team('A')
        m.team('B')
        s = m.scene
        s.hide(a['GK'], 120.0, 1e9)
        x, y = s.pos[a['GK']][s.k(100.0)]
        new = m.person('A', x, y, 125.0, 200.0, keeper=True)
        data, index, auto, res = S.people_of(m)
        g1, g2 = s.people[a['GK']]['id'], s.people[new]['id']
        sub = persons.resolve(auto, [{'kind': 'substitution', 'out': g1, 'in': g2, 'status': 'confirmed'}])
        self.assertNotEqual(sub['person'][g1], sub['person'][g2])
        self.assertEqual({display(sub, g1), display(sub, g2)}, {'GK-A'})


class Counts(unittest.TestCase):
    def test_13_eight_visible_outfield_players_make_no_phantoms(self):
        m = S.Match(seconds=200)
        a = m.team('A')
        m.team('B')
        m.scene.hide(a['LB'], 60.0, 120.0)
        m.scene.hide(a['RB'], 60.0, 120.0)
        data, index, auto, res = S.people_of(m)
        check_structure(self, data, res)
        self.assertEqual(len(res['places']['A']), 10)
        fragments_a = [f for f, v in auto['fragments'].items() if v['team'] == 'A']
        self.assertEqual(len(fragments_a), 11)                               # no fragment was made up
        self.assertEqual(sum(len(pl['nodes']) for pl in res['places']['A']), 10)

    def test_14_an_unresolved_fragment_shows_question_mark(self):
        m = S.Match(seconds=200)
        m.team('A')
        b = m.team('B')
        extra = m.person('B', 70.0, 40.0, 30.0, 170.0)                       # an 11th outfield 'B' the whole time
        data, index, auto, res = S.people_of(m)
        shown = sorted(display(res, m.scene.people[pid]['id']) for pid in list(b.values())+[extra])
        self.assertIn('B-?', shown)
        self.assertEqual(sum(1 for d in shown if d == 'B-?'), 1)

    def test_15_labels_never_inflate(self):
        m = S.Match(seconds=300, seed=3)
        a = m.team('A')
        b = m.team('B')
        rng = np.random.default_rng(7)
        for ids in (a, b):
            for role, pid in ids.items():
                if role == 'GK':
                    continue
                cur = pid
                for t in np.arange(30.0, 290.0, 30.0):
                    cur = m.cut(cur, float(t), gap=float(rng.uniform(1.0, 8.0)))
        data, index, auto, res = S.people_of(m)
        check_structure(self, data, res)
        self.assertGreater(len(auto['fragments']), 150)
        numbers = [int(p['display'][2:4]) for p in res['people'].values() if re.match(r'^[AB]-\d\d', p['display'])]
        self.assertTrue(numbers and max(numbers) <= 10)


class Mixed(unittest.TestCase):
    """A stitched chain joining two people (a keeper's run identity stitched to a player's at a segment start, or two
    unlike kits) is a person with its label left open, outside the places; the coach's include places it after all."""

    def build(self, first_local, first_kit):
        m = S.Match(seconds=200)
        m.team('A')
        b = m.team('B')
        extra = m.scene.people[m.person('B', 70.0, 40.0, 30.0, 170.0)]['id']      # an 11th 'B' the whole time
        data, index = m.build(segment_s=100.0)
        for r in index['roster']:
            if r['matchId'] == extra:
                r['members'] = [[0, first_local], [1, extra]]
        gals = m.galleries(data)
        gals[('kit', 0, first_local)] = [(np.asarray(first_kit[0], float), np.asarray(first_kit[1], float))]*3
        auto = persons.build_from(index, data, gals, {})
        real = [m.scene.people[pid]['id'] for role, pid in b.items() if role != 'GK']
        return data, index, auto, extra, real

    def test_a_keeper_stitched_to_a_player_stays_out_of_the_places(self):
        data, index, auto, extra, real = self.build('GK-1', S.OTHER_KIT)
        self.assertEqual(auto['fragments'][extra]['mixed']['why'], 'keeper')
        res = persons.resolve(auto, [])
        check_structure(self, data, res)
        self.assertIsNone(place_of(res, extra))
        self.assertEqual(display(res, extra), 'B-?')
        self.assertIn('keeper', res['people'][res['person'][extra]]['note'])
        self.assertNotIn(extra, res['excluded'])                              # still a person the analytics see
        self.assertTrue(all(place_of(res, f) for f in real))                  # the ten real players keep the places
        out = persons.corrected_index(index, res, [])
        self.assertEqual(next(f for f in out['fragments'] if f['matchId'] == extra)['mixed']['why'], 'keeper')

    def test_unlike_kits_across_a_boundary_make_it_mixed(self):
        data, index, auto, extra, real = self.build('B-77', S.OTHER_KIT)
        self.assertEqual(auto['fragments'][extra]['mixed']['why'], 'kit')
        self.assertIsNone(auto['fragments'][extra]['flag'])                    # half in the kit: not left out
        res = persons.resolve(auto, [])
        self.assertEqual(display(res, extra), 'B-?')

    def test_one_kit_on_both_sides_is_not_mixed(self):
        data, index, auto, extra, real = self.build('B-77', S.TEAM_KIT['B'])
        self.assertIsNone(auto['fragments'][extra]['mixed'])

    def test_saying_whose_slot_it_is_lifts_the_hold(self):
        data, index, auto, extra, real = self.build('GK-1', S.OTHER_KIT)
        res = persons.resolve(auto, [{'kind': 'confirm', 'target': extra, 'slot': 3}])
        check_structure(self, data, res)
        self.assertNotIn(extra, res['mixed'])
        held = persons.resolve(auto, [])
        entry = next(r for r in persons.corrected_index(index, held, [])['roster'] if extra in r.get('fragments', []))
        self.assertEqual((entry['display'], entry['mixed']['why']), ('B-?', 'keeper'))       # what the page shows

    def test_include_places_it_after_all(self):
        data, index, auto, extra, real = self.build('GK-1', S.OTHER_KIT)
        self.assertIn(extra, auto['teams']['B']['nodes'])                     # scored, only held out
        res = persons.resolve(auto, [{'kind': 'include', 'target': extra}])
        check_structure(self, data, res)
        self.assertNotIn(extra, res['mixed'])
        self.assertNotIn('note', res['people'][res['person'][extra]])


class Flow(unittest.TestCase):
    def test_min_cost_tracks_takes_the_cheap_links(self):
        # a1 -> b1 cheap, a2 -> b2 cheap, crosses dear; two paths
        order = ['a1', 'a2', 'b1', 'b2']
        edges = {('a1', 'b1'): -2.0, ('a2', 'b2'): -2.0, ('a1', 'b2'): 3.0, ('a2', 'b1'): 3.0}
        succ, starts = persons.min_cost_tracks(order, edges, 2, {v: 5.0 for v in order})
        self.assertEqual(succ, {'a1': 'b1', 'a2': 'b2'})
        self.assertEqual(sorted(starts), ['a1', 'a2'])

    def test_one_path_cannot_hold_two_at_once(self):
        nodes = {'a': {'t0': 0.0, 't1': 10.0, 'seconds': 10}, 'b': {'t0': 5.0, 't1': 20.0, 'seconds': 15}}
        tracks, conf = persons.solve_places(nodes, {}, K=1)
        self.assertEqual(len(tracks), 1)
        self.assertEqual(len(tracks[0]), 1)


class Corrections(unittest.TestCase):
    def setUp(self):
        m = S.Match(seconds=200)
        a = m.team('A')
        m.team('B')
        s = m.scene
        self.first = s.people[a['LB']]['id']
        self.later = s.people[m.cut(a['LB'], 80.0, gap=5.0)]['id']
        self.data, self.index, self.auto, self.res = S.people_of(m)

    def test_confirm_makes_it_sure_and_one_person(self):
        slot = place_of(self.res, self.first)[1]
        res = persons.resolve(self.auto, [{'kind': 'confirm', 'target': self.later, 'slot': slot}])
        self.assertEqual(display(res, self.later), f'A-{slot:02d}')
        self.assertEqual(res['person'][self.first], res['person'][self.later])

    def test_merge_joins_people(self):
        res = persons.resolve(self.auto, [{'kind': 'merge', 'target': self.later, 'into': self.first}])
        self.assertEqual(res['person'][self.first], res['person'][self.later])
        self.assertFalse(display(res, self.later).endswith('?'))

    def test_split_takes_it_out_of_the_place(self):
        res = persons.resolve(self.auto, [{'kind': 'split', 'target': self.later, 'from': [self.first]}])
        p0, p1 = place_of(res, self.first), place_of(res, self.later)
        # no longer one place; with ten places for eleven pieces one of the two may now be unresolved
        self.assertTrue(p0 is None or p1 is None or p0[:2] != p1[:2])
        self.assertNotEqual(res['person'][self.first], res['person'][self.later])

    def test_person_ids_survive_a_correction(self):
        others = {f: pid for f, pid in self.res['person'].items() if f not in (self.first, self.later)}
        res = persons.resolve(self.auto, [{'kind': 'merge', 'target': self.later, 'into': self.first}])
        self.assertEqual({f: res['person'][f] for f in others}, others)         # nobody else renumbered
        self.assertEqual(res['person'][self.later], self.res['person'][self.first])

    def test_a_person_target_reaches_every_fragment(self):
        res = persons.resolve(self.auto, [{'kind': 'merge', 'target': self.later, 'into': self.first}])
        pid = res['person'][self.first]
        index = dict(self.index)
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp)/'assembly').mkdir()
            corrections.save(tmp, [{'kind': 'merge', 'target': self.later, 'into': self.first, 'id': 'x', 'at': 'now'}])
            entry = corrections.add(tmp, index, {}, {'kind': 'exclude', 'target': pid}, people=res)
        self.assertEqual(sorted([entry['target']]+entry.get('also', [])), sorted([self.first, self.later]))
        gone = persons.resolve(self.auto, [{'kind': 'merge', 'target': self.later, 'into': self.first}, entry])
        self.assertIn(self.first, gone['excluded'])
        self.assertIn(self.later, gone['excluded'])

    def test_exclude_and_team(self):
        res = persons.resolve(self.auto, [{'kind': 'exclude', 'target': self.later}])
        self.assertIn(self.later, res['excluded'])
        res = persons.resolve(self.auto, [{'kind': 'team', 'target': self.later, 'team': 'B'}])
        self.assertEqual(res['teams'].get(self.later), 'B')
        self.assertTrue(display(res, self.later).startswith('B-'))

    def test_the_index_the_page_reads(self):
        out = persons.corrected_index(self.index, self.res, [])
        ids = {r['matchId'] for r in out['roster']}
        self.assertTrue(all(pid in ids for pid in out['aliases'].values()))
        self.assertEqual({f['matchId'] for f in out['fragments']}, {r['matchId'] for r in self.index['roster']})
        entry = next(r for r in out['roster'] if self.later in r.get('fragments', []))
        self.assertEqual((entry['provisional'], entry['confidence']), (True, 'estimate'))
        self.assertEqual(len(out['places']['A']), 10)


if __name__ == '__main__':
    unittest.main()
