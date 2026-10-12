"""A person's global identity, team and role must survive close encounters with officials, local-track
changes and contaminated crops; and the offline tracklet refinement reconnects what the causal pass
left in pieces. Driven on the synthetic people of test_identity.py with box overlaps measured."""
import math
import unittest

import numpy as np

import scene  # noqa: F401  (sys.path)
from soccer import tracklets
from soccer.appearance import Descriptor, add_to_gallery
from soccer.geometry import iou
from soccer.identity import IdentityManager, Options, Stab, TrackState
from soccer.teams import kit_vote
from test_identity import JERSEY, MODEL, SHORTS, Driver, Person, box_at, onehot, roster, unit


class Mixed(Person):
    """A crop that contains two people: the embedding and jersey are a blend, the quality poor."""

    def __init__(self, a, b):
        super().__init__(a.kit, a.unique, a.cls)
        self.a, self.b = a, b

    def descriptor(self, quality=.45):
        shorts = onehot(SHORTS[self.a.shorts])
        jersey = .5*JERSEY[self.a.kit]+.5*JERSEY[self.b.kit]
        embedding = unit(.5*self.a.embedding()+.5*self.b.embedding())
        return Descriptor(jersey, shorts, shorts[:8].copy(), np.full(24, .5), quality, embedding)


class EncounterDriver(Driver):
    """Like Driver, with the box overlap of the visible people measured (what tracker.py supplies)."""

    def step(self, visible, ended=(), zone='inside', segment=0, dt=.2, cues=None, ball=None, cls=None):
        samples = []
        boxes = {tid: box_at(x, y) for tid, (_, x, y) in visible.items()}
        for tid, (p, x, y) in visible.items():
            d = p.descriptor()
            b = boxes[tid]
            overlap, other = 0.0, None
            for k, o in boxes.items():
                if k != tid and iou(b, o) > overlap:
                    overlap, other = iou(b, o), k
            z = zone(tid) if callable(zone) else zone
            c = cues(tid) if callable(cues) else cues
            detector = cls(tid) if callable(cls) else p.detector_class()
            samples.append({'id': tid, 'box': b, 'score': .9, 'cls': detector, 'zone': z, 'occluded': overlap > .02,
                            'overlap': overlap, 'overlapWith': other,
                            'stab': Stab(x, y, .12), 'descriptor': d, 'vote': kit_vote(self.model, d.jersey, d.shorts), 'cues': c or {}})
        out = self.ids.update({'time': self.t, 'tick': True, 'samples': samples, 'ended': [{'id': i} for i in ended],
                               'model': self.model, 'pitch_reliable': True, 'filter_enabled': True, 'aspect': 16/9,
                               'segment': segment, 'drift': 0, 'ball': {'stab': ball, 'pitch': None} if ball else None})
        self.events += out['events']
        self.people.append(out['people'])
        self.t += dt
        return {p['track']: p for p in out['people']}


def team_scene():
    return {10+i: t for i, t in enumerate(roster('A', 3, x0=.1)+roster('B', 3, x0=.6))}


class CloseEncounterTests(unittest.TestCase):
    def test_player_running_beside_the_referee_keeps_identity_and_role(self):
        # Regression 1. A confirmed B player and the referee overlap for 1.4 s; their crops are mixtures and
        # the detector calls the player a referee on those frames. The player's identity, team and role must
        # come out of the encounter unchanged, and nothing from the overlap may enter their memory.
        d = EncounterDriver()
        player, ref = Person('B'), Person('REF')
        team = team_scene()
        out = d.run(4, {1: (player, .40, .50), 2: (ref, .55, .50), **team})
        pid = out[1]['id']
        self.assertTrue(pid and pid.startswith('B-'), out[1])
        self.assertEqual(out[2]['id'], 'REF-1')
        gallery_before = [s.time for s in d.ids.registry[pid].gallery]
        t_start = d.t
        mixed_player = Mixed(player, ref)
        for k in range(7):   # 1.4 s of overlap: the player's box sits on the referee's
            x = .47+.005*k
            d.step({1: (mixed_player, x, .50), 2: (ref, x+.004, .50), **team}, cls=lambda tid: 'referee' if tid == 1 else None)
        t_end = d.t
        during = [p for frame in d.people[-7:] for p in frame if p['track'] == 1]
        self.assertTrue(all(p['id'] == pid and p['role'] == 'PLAYER' and p['team'] == 'B' for p in during), [(p['id'], p['role']) for p in during])
        self.assertTrue(all(str(p.get('memory', '')).startswith('frozen') for p in during), [p.get('memory') for p in during])
        self.assertTrue(all(p.get('occlusion', 0) >= .15 for p in during))
        self.assertTrue(any(e['kind'] == 'occlusion' and 'FREEZE APPEARANCE' in e['message'] for e in d.events))
        out = d.run(5, {1: (player, .60, .50), 2: (ref, .35, .50), **team})
        self.assertEqual((out[1]['id'], out[1]['role'], out[1]['team']), (pid, 'PLAYER', 'B'))
        self.assertEqual((out[2]['id'], out[2]['role']), ('REF-1', 'REFEREE'))
        self.assertFalse([e for e in d.kinds('role') if 'New role: REFEREE' in e['message']], 'nobody changed role')
        g = d.ids.registry[pid]
        self.assertFalse([s.time for s in g.gallery if t_start-1e-6 <= s.time <= t_end+1e-6], 'no overlap crop in the gallery')
        self.assertEqual(g.status, 'active')
        self.assertEqual(len(d.ids.summary()['referees']), 1)

    def test_player_lost_and_back_on_a_new_track_is_restored_before_any_role_change(self):
        # Regression 2. B-02 is confirmed; a second box (a split detection) appears on them for a few frames,
        # the old track ends, and the new track's first second is contaminated by a passing referee with the
        # detector calling them a referee. The new track must come back as B-02, never as B-? -> REF-?.
        d = EncounterDriver()
        ref = Person('REF')
        team = team_scene()
        out = d.run(4, {**team, 1: (ref, .3, .35)})
        self.assertEqual(out[1]['id'], 'REF-1')
        target = team[13][0]            # B-01's person (the first B of the roster)
        pid = out[13]['id']
        self.assertTrue(pid.startswith('B-'))
        # A duplicate box on B-xx for 0.4 s, then the old track ends.
        for k in range(2):
            d.step({**team, 1: (ref, .3, .35), 94: (target, team[13][1]+.004, team[13][2]-.01)})
        rest = {k: v for k, v in team.items() if k != 13}
        d.step({**rest, 1: (ref, .3, .35), 94: (target, team[13][1]+.004, team[13][2]-.01)}, ended=[13])
        mixed = Mixed(target, ref)
        for k in range(5):   # one second contaminated by the referee passing by
            x = team[13][1]+.004
            d.step({**rest, 1: (ref, x+.003, team[13][2]), 94: (mixed, x, team[13][2])}, cls=lambda tid: 'referee' if tid == 94 else None)
        out = d.run(4, {**rest, 1: (ref, .3, .35), 94: (target, team[13][1]+.01, team[13][2])})
        self.assertEqual((out[94]['id'], out[94]['role'], out[94]['team']), (pid, 'PLAYER', 'B'), out[94])
        labels = [p['display'] for frame in d.people for p in frame if p['track'] == 94]
        self.assertNotIn('REF-?', labels, labels)
        self.assertFalse([e for e in d.kinds('new-identity') if e['track'] == 94])
        self.assertEqual(len(d.ids.summary()['referees']), 1)
        self.assertEqual(len(d.ids.summary()['players']), 6)
        reid = [e for e in d.kinds('reid') if e['track'] == 94]
        self.assertTrue(reid and f'Matched Global Player: TeamB_Player_{pid[2:]}' in reid[0]['message'], [e['message'][:80] for e in d.events if e['track'] == 94])

    def test_gallery_resemblance_never_makes_a_referee_of_a_player_whose_evidence_says_player(self):
        # The referee's gallery was contaminated by this B player's crops (an earlier hidden overlap). The
        # player's own temporal evidence says team B player; a gallery resemblance alone must not turn them
        # into an official, and no referee identity may be created from player evidence.
        d = EncounterDriver()
        ref = Person('REF')
        team = team_scene()
        out = d.run(3, {**team, 1: (ref, .3, .35)})
        self.assertEqual(out[1]['id'], 'REF-1')
        player = Person('B', cls='player')
        contaminated = d.ids.registry['REF-1']
        for k in range(4):
            contaminated.gallery = add_to_gallery(contaminated.gallery, player.descriptor(), 100+k, 6, trusted=True)
        d.ids._looks = lambda tr, t: {'referee': 0.0, 'A': 0.1, 'B': .6, 'probes': 3}   # the batch measure, as on the real clip
        out = d.run(6, {**team, 1: (ref, .3, .35), 40: (player, .5, .45)})
        self.assertEqual((out[40]['role'], out[40]['team']), ('PLAYER', 'B'), out[40])
        self.assertTrue(out[40]['id'] and out[40]['id'].startswith('B-'), out[40])
        self.assertEqual(len(d.ids.summary()['referees']), 1, [g['id'] for g in d.ids.summary()['referees']])
        self.assertFalse([e for e in d.kinds('role') if 'REFEREE IDENTIFIED' in e['message'] and 'REF-2' in e['message']])

    def test_a_missing_player_leaves_room_in_a_full_team(self):
        # Team B is at its cap. B-03 leaves; a track in B's kit the detector sometimes calls a referee appears
        # where they left: with a teammate missing the team is not "full", so the referee threshold stays
        # where it is, and the track is re-identified as the missing player.
        d = EncounterDriver(Options(max_per_team=3))
        team = team_scene()
        out = d.run(3, team)
        missing = team[15][0]
        pid = out[15]['id']
        rest = {k: v for k, v in team.items() if k != 15}
        d.step(rest, ended=[15])
        d.run(1, rest)
        missing.cycle = ['referee', 'referee', 'player', 'referee', 'player']
        out = d.run(5, {**rest, 50: (missing, team[15][1]+.01, team[15][2])})
        self.assertEqual((out[50]['id'], out[50]['role'], out[50]['team']), (pid, 'PLAYER', 'B'), out[50])
        self.assertEqual(d.ids.summary()['referees'], [])

    def test_a_confirmed_player_needs_sustained_clean_referee_evidence_to_change_role(self):
        d = EncounterDriver()
        player = Person('B', cls='player')
        team = team_scene()
        out = d.run(4, {**team, 1: (player, .5, .45)})
        pid = out[1]['id']
        # The detector calls them a referee for nine seconds (kit and look stay the player's): the raw role
        # reaches REFEREE after ~8 s, but one or two checks never change a confirmed player; then the
        # detector returns to normal and the role was never touched.
        player.cls = 'referee'
        d.run(9.5, {**team, 1: (player, .5, .45)})
        self.assertEqual(d.ids.tracks[1].raw_role[0], 'referee')
        player.cls = 'player'
        out = d.run(5, {**team, 1: (player, .5, .45)})
        self.assertEqual((out[1]['id'], out[1]['role']), (pid, 'PLAYER'))
        ignored = [e for e in d.kinds('role') if e['message'].startswith('ROLE UPDATE IGNORED')]
        self.assertTrue(ignored, [e['message'][:60] for e in d.kinds('role')])
        self.assertIn('Raw role: REFEREE', ignored[0]['message'])
        self.assertIn('Stable role: PLAYER (team B)', ignored[0]['message'])
        self.assertFalse([e for e in d.kinds('role') if 'New role: REFEREE' in e['message']])

    def test_track_that_jumps_onto_the_referee_is_reborn_and_the_referee_reconnected(self):
        # The local tracker hands the player's track to the referee when they cross (the referee's own track
        # ends). The track must be re-born: the player's identity goes missing (neither converted nor
        # retired), the track is re-identified as the known referee (not REF-? or a new REF-2), the player's
        # body on its new track gets the identity back, and no frame since the swap shows the player on the
        # referee's body.
        d = EncounterDriver()
        player, ref = Person('B'), Person('REF')
        team = team_scene()
        out = d.run(4, {1: (player, .40, .50), 2: (ref, .45, .50), **team})
        pid = out[1]['id']
        self.assertTrue(pid and pid.startswith('B-'), out[1])
        self.assertEqual(out[2]['id'], 'REF-1')
        swap = d.t
        d.step({1: (Mixed(player, ref), .425, .50), 2: (Mixed(ref, player), .428, .50), **team})
        d.step({1: (ref, .43, .50), 3: (player, .415, .50), **team}, ended=[2])
        for k in range(1, 30):
            out = d.step({1: (ref, .43+.004*k, .50), 3: (player, .415-.004*k, .50), **team})
        switch = [e for e in d.events if e['kind'] == 'switch' and e['track'] == 1]
        self.assertTrue(switch, [e['message'][:70] for e in d.events if e['track'] == 1])
        self.assertTrue(switch[0]['message'].startswith('TRACK SWITCH'))
        self.assertIn(f'Global {pid}', switch[0]['message'])
        self.assertEqual((out[1]['id'], out[1]['role']), ('REF-1', 'REFEREE'), out[1])
        self.assertEqual((out[3]['id'], out[3]['role'], out[3]['team']), (pid, 'PLAYER', 'B'), out[3])
        self.assertEqual(len(d.ids.summary()['referees']), 1)
        self.assertEqual(len(d.ids.summary()['players']), 7)
        self.assertNotEqual(d.ids.registry[pid].status, 'retired')
        self.assertFalse([e for e in d.kinds('role') if 'New role: REFEREE' in e['message']], 'no role change: the identity was never the referee')
        after = [p for frame in d.people for p in frame if p['track'] == 1 and p['time'] >= swap+.2-1e-6]
        self.assertFalse([p['time'] for p in after if p['id'] == pid], 'no frame after the swap shows the player on the referee')
        self.assertEqual(len(d.ids.tracks[1].switches), 1)
        self.assertFalse([e for e in d.kinds('new-identity') if e['track'] == 1 and e['time'] >= swap], 'no new referee identity')


class OfflineTrackletTests(unittest.TestCase):
    def test_fragment_labelled_ref_is_reconnected_to_the_dormant_player(self):
        # The causal pass left B-xx on track 13 until 4.0 s, then an uncertain fragment on track 60 (first
        # B-?, then REF-?) in the same place: the refinement connects it, and the 2D roster keeps the player.
        d = EncounterDriver()
        team = team_scene()
        out = d.run(4, team)
        pid, person = out[13]['id'], team[13][0]
        rest = {k: v for k, v in team.items() if k != 13}
        d.step(rest, ended=[13])
        manager = d.ids
        frames = [{'time': k*.2, 'people': list(frame)} for k, frame in enumerate(d.people)]
        t0 = frames[-1]['time']+.2
        x, y = team[13][1]+.006, team[13][2]-.004
        fragment = []
        for k in range(12):
            t = t0+.2*k
            label = 'B-?' if k < 5 else 'REF-?'
            p = {'time': round(t, 3), 'track': 60, 'box': [round(v, 5) for v in box_at(x+.002*k, y)], 'score': .9, 'cls': 'player', 'zone': 'inside',
                 'evidence': 'observed', 'id': None, 'display': label, 'label': 'IDENTITY_UNCERTAIN', 'team': 'B' if k < 5 else None,
                 'role': 'PLAYER' if k < 5 else 'REFEREE', 'state': 'uncertain', 'identityConfidence': 0.0, 'roleConfidence': .5}
            frames.append({'time': round(t, 3), 'people': [p]})
            fragment.append(p)
        tr = TrackState(60, t0, 0, Stab(x, y, .12), None, 'inside', '', box_at(x, y), Stab(x, y, .12), 0)
        for k in range(4):
            tr.descs.append((person.descriptor(), t0+.2*k, False))
        manager.ended[60] = tr
        motions = [(np.eye(3), True, False)]*len(frames)
        report = tracklets.refine(manager, frames, motions, 1280, 720)
        self.assertEqual(report['connected'], 1, report)
        self.assertTrue(all(p['id'] == pid and p['role'] == 'PLAYER' and p['team'] == 'B' and p['evidence'] == 'connected offline' for p in fragment), fragment[-1])
        self.assertEqual(fragment[-1]['firstPassLabel'], 'REF-?')
        self.assertTrue(report['events'] and report['events'][0]['message'].startswith('TRACKLET CONNECT'))
        self.assertIn(f'-> TeamB_Player_{pid[2:]}', report['events'][0]['message'])
        m = tracklets.metrics(manager, frames, report['events'], report)
        self.assertEqual((m['trackletsConnected'], m['framesRestoredToPlayers']), (1, 12))
        self.assertEqual(m['restoredFromLabels'], {'B-?': 5, 'REF-?': 7})

    def test_a_fragment_seen_while_the_player_was_seen_elsewhere_is_never_connected(self):
        d = EncounterDriver()
        team = team_scene()
        out = d.run(4, team)
        pid, person = out[13]['id'], team[13][0]
        manager = d.ids
        frames = [{'time': k*.2, 'people': list(frame)} for k, frame in enumerate(d.people)]
        # A second, unidentified track in the same frames as the player (they are both in view): not them.
        for k, f in enumerate(frames[5:], start=5):
            f['people'].append({'time': f['time'], 'track': 60, 'box': [round(v, 5) for v in box_at(.9, .8)], 'score': .9, 'cls': 'player', 'zone': 'inside',
                                'evidence': 'observed', 'id': None, 'display': 'B-?', 'label': 'IDENTITY_UNCERTAIN', 'team': 'B', 'role': 'PLAYER',
                                'state': 'uncertain', 'identityConfidence': 0.0, 'roleConfidence': .5})
        tr = TrackState(60, 1.0, 0, Stab(.9, .8, .12), None, 'inside', '', box_at(.9, .8), Stab(.9, .8, .12), 0)
        for k in range(4):
            tr.descs.append((person.descriptor(), 1.0+.2*k, False))
        manager.ended[60] = tr
        report = tracklets.refine(manager, frames, [(np.eye(3), True, False)]*len(frames), 1280, 720)
        self.assertEqual(report['connected'], 0)

    def test_offline_cut_where_the_crops_change_person(self):
        # One local track (60) that the tracker handed from a B player to the referee, labelled B-? throughout
        # by the causal pass. The refinement cuts it where its clean crops change, connects the first part to
        # the dormant player and the second to the dormant referee, and the label the second part wore counts
        # for nothing.
        d = EncounterDriver()
        ref = Person('REF')
        team = team_scene()
        out = d.run(4, {**team, 1: (ref, team[13][1]+.03, team[13][2])})
        pid, person = out[13]['id'], team[13][0]
        self.assertEqual(out[1]['id'], 'REF-1')
        rest = {k: v for k, v in team.items() if k != 13}
        d.step(rest, ended=[13, 1])
        manager = d.ids
        frames = [{'time': k*.2, 'people': list(frame)} for k, frame in enumerate(d.people)]
        t0 = frames[-1]['time']+.2
        x, y = team[13][1]+.006, team[13][2]
        tr = TrackState(60, t0, 0, Stab(x, y, .12), None, 'inside', '', box_at(x, y), Stab(x, y, .12), 0)
        fragment = []
        for k in range(16):
            t = round(t0+.2*k, 3)
            p = {'time': t, 'track': 60, 'box': [round(v, 5) for v in box_at(x+.002*k, y)], 'score': .9, 'cls': 'player', 'zone': 'inside',
                 'evidence': 'observed', 'id': None, 'display': 'B-?', 'label': 'IDENTITY_UNCERTAIN', 'team': 'B', 'role': 'PLAYER',
                 'state': 'uncertain', 'identityConfidence': 0.0, 'roleConfidence': .5}
            frames.append({'time': t, 'people': [p]})
            fragment.append(p)
            tr.clean_history.append((t, (person if k < 8 else ref).descriptor()))
        manager.ended[60] = tr
        report = tracklets.refine(manager, frames, [(np.eye(3), True, False)]*len(frames), 1280, 720)
        self.assertEqual(report['splits'], 1, report)
        self.assertEqual(report['connected'], 2, report)
        self.assertTrue(all(p['id'] == pid and p['role'] == 'PLAYER' for p in fragment[:8]), fragment[7])
        self.assertTrue(all(p['id'] == 'REF-1' and p['role'] == 'REFEREE' and p['team'] is None for p in fragment[8:]), fragment[8])
        self.assertTrue([e for e in report['events'] if e['message'].startswith('TRACKLET SPLIT')])
        m = tracklets.metrics(manager, frames, report['events'], report)
        self.assertEqual((m['trackletsSplit'], m['officialsConnected'], m['trackletsConnected']), (1, 1, 2))

    def test_a_label_that_continues_across_a_cut_is_dropped_for_the_person_the_crops_show(self):
        # The causal pass kept B-xx on track 13 although its crops became the referee's half-way (the swap
        # went unnoticed online, the referee's own track had ended). Offline the track is cut where the crops
        # change; the label after the cut is stale, the fragment goes to the dormant referee, and the
        # player's observations after the cut are no longer theirs.
        d = EncounterDriver()
        ref = Person('REF')
        team = team_scene()
        out = d.run(4, {**team, 1: (ref, team[13][1]+.03, team[13][2])})
        pid, person = out[13]['id'], team[13][0]
        self.assertEqual(out[1]['id'], 'REF-1')
        rest = {k: v for k, v in team.items() if k != 13}
        d.step({**rest, 13: (person, team[13][1], team[13][2])}, ended=[1])
        manager = d.ids
        g = manager.registry[pid]
        frames = [{'time': k*.2, 'people': list(frame)} for k, frame in enumerate(d.people)]
        t0 = frames[-1]['time']+.2
        x, y = team[13][1]+.006, team[13][2]
        tr = manager.tracks[13]
        tr.clean_history = [(when, person.descriptor()) for when in (t0-1.0, t0-.8, t0-.6, t0-.4, t0-.2)]
        before = g.observations
        rows = []
        for k in range(16):
            t = round(t0+.2*k, 3)
            p = {'time': t, 'track': 13, 'box': [round(v, 5) for v in box_at(x+.002*k, y)], 'score': .9, 'cls': 'player', 'zone': 'inside',
                 'evidence': 'observed', 'id': pid, 'display': pid, 'label': 'PLAYER', 'team': 'B', 'role': 'PLAYER', 'state': 'confirmed',
                 'identityConfidence': .9, 'roleConfidence': .9}
            frames.append({'time': t, 'people': [p]})
            rows.append(p)
            g.emitted_all.append(p)
            g.observations += 1
            tr.clean_history.append((t, (person if k < 6 else ref).descriptor()))
        report = tracklets.refine(manager, frames, [(np.eye(3), True, False)]*len(frames), 1280, 720)
        self.assertEqual(report['splits'], 1, report['events'])
        self.assertTrue(all(p['id'] == pid for p in rows[:6]), [p['id'] for p in rows[:6]])
        self.assertTrue(all(p['id'] == 'REF-1' and p['role'] == 'REFEREE' and p['evidence'] == 'connected offline' for p in rows[6:]), rows[6])
        self.assertEqual(g.observations, before+6)
        self.assertFalse([p for p in rows[6:] if p in g.emitted_all])
        split = [e for e in report['events'] if e['message'].startswith('TRACKLET SPLIT')]
        self.assertTrue(split and '-> REF-1' in split[0]['message'], split)

    def test_a_scene_wide_appearance_change_cuts_nothing(self):
        # A zoom or a change of light makes every track's crops change at once: that is not a swap.
        d = EncounterDriver()
        team = team_scene()
        d.run(4, team)
        manager = d.ids
        frames = [{'time': k*.2, 'people': list(frame)} for k, frame in enumerate(d.people)]
        other = {tid: Person('REF') for tid in team}   # everyone's crops become another kit at once
        for tid, (person, x, y) in team.items():
            tr = manager.tracks[tid]
            tr.clean_history = [(k*.2, (person if k < 10 else other[tid]).descriptor()) for k in range(20)]
        report = tracklets.refine(manager, frames, [(np.eye(3), True, False)]*len(frames), 1280, 720)
        self.assertEqual(report['splits'], 0, report['events'])
        self.assertTrue([e for e in report['events'] if e['message'].startswith('TRACKLET SPLIT SKIPPED')])


if __name__ == '__main__':
    unittest.main()
