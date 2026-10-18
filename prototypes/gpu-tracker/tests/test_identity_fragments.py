"""Offline association of small fragments in a small-scene run (far camera, amateur footage). OSNet on crops of about
thirty pixels cannot tell teammates apart, so the appearance floor left fragments anonymous whose position was
near-certain, and identities created from far crops have empty galleries. There, a small fragment is placed on
position: appearance only vetoes, the fragment's own kit votes must not name the other team, and every identity the
position does not rule out competes for the lead. Broadcast runs (no small-scene frames) are untouched."""
import math
import unittest
from collections import Counter
from types import SimpleNamespace

import numpy as np

import scene  # noqa: F401  (sys.path)
from soccer import tracklets
from soccer.appearance import Descriptor
from soccer.identity import Stab, TrackState
from test_identity import Driver, Person, box_at, roster, unit
from test_identity_amateur import Sized

H_SMALL = .03     # box height: 22 px of a 720 px frame


def blurred(person, c, quality=.3):
    """A small crop of `person`: its embedding only c-alike the person's (OSNet on an upscaled 22 px crop)."""
    d = person.descriptor()
    noise = unit(np.random.default_rng(int(1000*c)+7).normal(size=512))
    e = d.embedding
    noise = unit(noise-np.dot(noise, e)*e)
    return Descriptor(d.jersey, d.shorts, d.socks, d.layout, quality, unit(c*e+math.sqrt(1-c*c)*noise))


class SmallFragmentTests(unittest.TestCase):
    def scene(self, gone=(13,), small=True, likeness=.7, votes=('B',)*6, state='uncertain', people=None, at=None, extra_gone=(), delay=0,
              later=None):
        """Teams A (tracks 10-12) and B (13-15) on small boxes for 4 s; the players on the `gone` tracks then vanish and,
        0.4 s (plus `delay` ticks) later, an anonymous fragment (track 60, 2 s, small crops `likeness`-alike the first of
        them) appears where the first one vanished, while the others stay in view. later: {track: ticks}, players of
        `gone` that vanish that many ticks later than the rest. Returns (manager, frames, fragment rows, pids)."""
        d = Driver()
        team = {10+i: t for i, t in enumerate(roster('A', 3, x0=.1)+roster('B', 3, x0=.6))}
        for tid, person in (people or {}).items():
            team[tid] = (person, team[tid][1], team[tid][2])
        boxes = lambda v: {tid: box_at(x, y, h=H_SMALL) for tid, (_, x, y) in v.items()}
        d.run(4, team, boxes=boxes(team))
        pids = {tid: d.ids.tracks[tid].player_id for tid in team}
        later = later or {}
        rest = {k: v for k, v in team.items() if k not in gone+extra_gone or k in later}
        d.step(rest, ended=[k for k in gone+extra_gone if k not in later], boxes=boxes(rest))
        first = len(d.people)
        for k in range(round(2.6/.2)+delay):
            ending = [tid for tid, ticks in later.items() if ticks == k+1]
            rest = {tid: v for tid, v in rest.items() if tid not in ending}
            d.step(rest, ended=ending, boxes=boxes(rest))
        first += delay
        frames = [{'time': round(k*.2, 3), 'people': list(f)} for k, f in enumerate(d.people)]
        person, x, y = team[gone[0]]
        x, y = at or (x+.004, y)
        tr = TrackState(60, frames[first+1]['time'], 0, Stab(x, y, H_SMALL), None, 'inside', '', box_at(x, y, H_SMALL), Stab(x, y, H_SMALL), 0)
        rows = []
        for k, f in enumerate(frames[first+1:first+12]):
            p = {'time': f['time'], 'track': 60, 'box': [round(v, 5) for v in box_at(x+.0005*k, y, H_SMALL)], 'score': .6, 'cls': 'player',
                 'zone': 'inside', 'evidence': 'observed', 'id': None, 'display': 'B-?' if state == 'uncertain' else 'CAND',
                 'label': 'IDENTITY_UNCERTAIN' if state == 'uncertain' else 'CANDIDATE', 'team': 'B' if state == 'uncertain' else None,
                 'role': 'PLAYER' if state == 'uncertain' else 'UNKNOWN', 'state': state, 'identityConfidence': 0.0, 'roleConfidence': .5}
            f['people'].append(p)
            rows.append(p)
            if k % 3 == 0 and likeness is not None:
                tr.clean_history.append((f['time'], blurred(person, likeness)))
        tr.vote_history = [(rows[k]['time'], v) for k, v in enumerate(votes)]
        d.ids.ended[60] = tr
        for f in frames:
            f['profile'] = {'smallScene': small}
        return d.ids, frames, rows, pids

    def refine(self, manager, frames):
        motions = [(np.eye(3), True, False)]*len(frames)
        return tracklets.refine(manager, frames, motions, 1280, 720)

    def test_a_small_fragment_where_the_player_vanished_takes_the_identity_in_a_small_scene_run(self):
        manager, frames, rows, pids = self.scene()
        report = self.refine(manager, frames)
        self.assertEqual(report['smallConnected'], 1, [e['message'] for e in report['events']])
        self.assertTrue(all(p['id'] == pids[13] and p['evidence'] == 'connected offline' for p in rows), rows[0])
        msg = next(e['message'] for e in report['events'] if e['message'].startswith('TRACKLET CONNECT'))
        self.assertIn('Small crops (median 22 px, crop quality 0.30): appearance not required', msg)
        self.assertIn('Kit votes in the fragment: team B 6', msg)
        m = tracklets.metrics(manager, frames, report['events'], report)
        self.assertEqual(m['smallFragments'], {'connected': 1})

    def test_the_same_fragment_stays_anonymous_outside_small_scene_runs(self):
        manager, frames, rows, pids = self.scene(small=False)
        report = self.refine(manager, frames)
        self.assertNotIn('smallConnected', report)
        self.assertTrue(all(p['id'] is None for p in rows))
        unresolved = [e['message'] for e in report['events'] if e['message'].startswith('TRACKLET UNRESOLVED')]
        self.assertTrue(unresolved and 'under the floor' in unresolved[0], unresolved)
        self.assertNotIn('smallFragments', tracklets.metrics(manager, frames, report['events'], report))

    def test_position_alone_places_a_fragment_only_within_a_jog_of_the_sighting(self):
        # 1.8 s after the player vanished: 5 m away is within reach (SMALL_REACH + SMALL_SPEED per second: 7.4 m), 10 m
        # is not, though the spatial score (.81 there: its sigma is 15.6 m after 1.8 s) would have allowed it.
        dx = lambda metres: metres/1.8*H_SMALL*720/1280   # metres -> image x units at this box height
        for metres, placed in ((5.0, True), (10.0, False)):
            with self.subTest(metres=metres):
                manager, frames, rows, pids = self.scene(delay=6, at=(.6+dx(metres), .5))
                report = self.refine(manager, frames)
                self.assertEqual(report['smallConnected'], int(placed), [e['message'] for e in report['events']])
                self.assertEqual({p['id'] for p in rows}, {pids[13] if placed else None})
                if not placed:
                    unresolved = next(e['message'] for e in report['events'] if e['message'].startswith('TRACKLET UNRESOLVED'))
                    self.assertIn(f'best {pids[13]} 10.0 m from its sighting 1.8 s away (needs at most 2 m + 3 m/s within 2 s)', unresolved)

    def test_crops_that_contradict_the_identity_veto_it(self):
        manager, frames, rows, pids = self.scene(likeness=.3)
        report = self.refine(manager, frames)
        self.assertEqual(report['smallConnected'], 0)
        self.assertTrue(all(p['id'] is None for p in rows))

    def test_kit_votes_for_the_other_team_veto_it(self):
        manager, frames, rows, pids = self.scene(votes=('A',)*4+('B',)*2, state='candidate')
        report = self.refine(manager, frames)
        self.assertEqual(report['smallConnected'], 0, [e['message'] for e in report['events']])
        self.assertTrue(all(p['id'] is None for p in rows))

    def test_two_teammates_lost_at_the_same_spot_leave_it_anonymous(self):
        # Two look-alike team B players vanish next to each other: the position cannot tell which one the fragment is.
        first = Person('B')
        mate = Person('B', unique=unit(first.unique+.1*unit(np.random.default_rng(3).normal(size=512))))
        manager, frames, rows, pids = self.scene(gone=(13, 14), people={13: first, 14: mate}, at=(.63, .515))
        report = self.refine(manager, frames)
        self.assertEqual(report['smallConnected'], 0, [e['message'] for e in report['events']])
        self.assertTrue(all(p['id'] is None for p in rows))
        unresolved = [e['message'] for e in report['events'] if e['message'].startswith('TRACKLET UNRESOLVED')]
        self.assertTrue(unresolved and 'small crops (median 22 px' in unresolved[0], unresolved)

    def test_an_identity_seen_elsewhere_during_the_fragment_is_never_its_identity(self):
        # The fragment appears next to a team B player who stays in view: only the one who vanished may be it, and it is
        # too far away; nobody seen during the fragment is ever considered.
        manager, frames, rows, pids = self.scene(at=(.665, .53))
        report = self.refine(manager, frames)
        self.assertNotIn(pids[14], {p['id'] for p in rows})
        frame_ids = [[p['id'] for p in f['people'] if p.get('id')] for f in frames]
        self.assertTrue(all(len(ids) == len(set(ids)) for ids in frame_ids), 'one identity on two people')

    def test_assignment_places_a_small_candidate_run_on_an_identity_without_a_gallery(self):
        # The identity was created from far crops (no gallery sample reaches the gallery quality), and the fragment is a
        # candidate run (no team label yet): the usual assignment never compares them; the small-scene phase does.
        far = Sized('B', quality=.3)
        manager, frames, rows, pids = self.scene(people={13: far}, state='candidate')
        self.assertFalse(manager.registry[pids[13]].gallery)
        motions = [(np.eye(3), True, False)]*len(frames)
        report = tracklets.assign_fragments(manager, frames, motions, 1280, 720)
        self.assertEqual(report['smallAccepted'], 1, report)
        self.assertTrue(all(p['id'] == pids[13] and p['evidence'] == 'assigned offline' for p in rows), rows[0])
        msg = report['events'][0]['message']
        self.assertIn('no gallery to compare', msg)

    def test_a_better_placed_identity_out_of_reach_in_time_blocks_the_assignment(self):
        # Two far team B players without galleries vanish: B-13's player 2.2 s before the fragment begins, 2.7 m from it
        # (beyond SMALL_GAP), B-14's 0.6 s before, 3.5 m from it (within reach). The position favours the first, who may
        # not be placed on position alone; the second is not more likely for that, so the fragment stays anonymous. With
        # the first player still in view (ruled out), the second takes it.
        far = {13: Sized('B', quality=.3), 14: Sized('B', quality=.3)}
        motions = lambda frames: [(np.eye(3), True, False)]*len(frames)
        spot = (.6+.435*.06, .5+.435*.03)   # between where the two vanished
        manager, frames, rows, pids = self.scene(gone=(13, 14), later={14: 8}, delay=8, at=spot, people=far, state='candidate', likeness=None)
        report = tracklets.assign_fragments(manager, frames, motions(frames), 1280, 720)
        self.assertEqual(report['smallAccepted'], 0, report)
        self.assertGreaterEqual(report['rejectedByScore'], 1)
        self.assertTrue(all(p['id'] is None for p in rows))
        manager, frames, rows, pids = self.scene(gone=(14,), later={14: 8}, delay=8, at=spot, people=far, state='candidate', likeness=None)
        report = tracklets.assign_fragments(manager, frames, motions(frames), 1280, 720)
        self.assertEqual(report['smallAccepted'], 1, report)
        self.assertTrue(all(p['id'] == pids[14] for p in rows))

    def test_assignment_outside_small_scene_runs_is_unchanged(self):
        far = Sized('B', quality=.3)
        manager, frames, rows, pids = self.scene(people={13: far}, state='candidate', small=False)
        motions = [(np.eye(3), True, False)]*len(frames)
        report = tracklets.assign_fragments(manager, frames, motions, 1280, 720)
        self.assertNotIn('smallAccepted', report)
        self.assertEqual(report['accepted'], 0)


class ImpossibleSideTests(unittest.TestCase):
    """The spatial score takes the better of the identity's two sightings around a fragment, so a fragment the identity
    could not have reached on one side was still connected on the other (amateur 3, 13.8 s: 10 m from B-10's sighting
    0.13 s before it, "before 0.09, after 0.92", final 0.94). A side farther than a player runs vetoes the identity."""

    def scorer(self, before_y, after_y=.45):
        frames = [{'time': round(.1*k, 3), 'people': []} for k in range(25)]
        sighting = lambda t, y: {'id': 'B-10', 'time': t, 'box': [.5, y-.02, .01, .02], 'pitch': [.5, y], 'pitchConfidence': .5}
        g = SimpleNamespace(pid='B-10', team='B', gallery=[], emitted_all=[sighting(1.0, before_y), sighting(1.7, after_y)])
        manager = SimpleNamespace(registry={'B-10': g})
        sc = tracklets._Scorer(manager, frames, [(np.eye(3), True, False)]*len(frames), 1280, 720,
                               SimpleNamespace(length=105.0, width=68.0))
        rows = [(k, {'time': frames[k]['time'], 'box': [.5, .43, .01, .02], 'pitch': [.5, .45], 'pitchConfidence': .5}) for k in range(11, 16)]
        return sc, g, rows

    def test_a_side_beyond_running_reach_vetoes_the_identity_whatever_the_other_side_says(self):
        # Before: 10.2 m in 0.1 s (counted as 0.5 s: 6 m at most). After: the same spot.
        sc, g, rows = self.scorer(before_y=.3)
        self.assertIsNone(sc.score(g, rows, []))
        self.assertIsNone(sc.score_small(g, rows, [], Counter()))

    def test_both_sides_within_reach_are_scored_as_before(self):
        # Before: 2 m in 0.1 s. The better side still decides the spatial score.
        sc, g, rows = self.scorer(before_y=.42)
        s = sc.score(g, rows, [])
        self.assertIsNotNone(s)
        self.assertEqual(s['spatial'], max(s['forward'], s['backward']))
        self.assertIsNotNone(sc.score_small(g, rows, [], Counter()))

    def test_the_reach_grows_with_the_gap(self):
        self.assertFalse(tracklets._Scorer.impossible(5.9, .1))
        self.assertTrue(tracklets._Scorer.impossible(6.1, .1))
        self.assertFalse(tracklets._Scorer.impossible(23.0, 2.0))
        self.assertTrue(tracklets._Scorer.impossible(25.0, 2.0))
        self.assertFalse(tracklets._Scorer.impossible(None, .1))


if __name__ == '__main__':
    unittest.main()
