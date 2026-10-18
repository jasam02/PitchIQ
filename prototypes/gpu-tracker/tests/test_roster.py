"""The active roster (soccer/roster.py): one slot per identity with its end-of-segment status, teams over the
cap reported rather than hidden, anonymous observations counted, and a 'left the match' correction closing a slot."""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from soccer import roster  # noqa: E402


def person(pid, track, team='A', state='confirmed', pitch=(.5, .5)):
    return {'track': track, 'id': pid, 'team': team, 'role': 'PLAYER', 'state': state, 'box': [.1, .1, .02, .1], 'pitch': list(pitch)}


def entry(pid, team, role='PLAYER', exit_edge=''):
    return {'id': pid, 'display': pid, 'team': team, 'role': role, 'status': 'ACTIVE', 'exitEdge': exit_edge, 'lastPitch': [.4, .4]}


def result():
    frames = []
    for i in range(50):
        t = round(i*.2, 3)
        people = [person('A-01', 1)]
        if i < 20:
            people.append(person('A-02', 2))            # leaves at 4 s through the right edge
        if i < 10:
            people.append(person('A-03', 3))            # vanishes at 2 s, no edge seen
        if 25 <= i < 40:
            people.append(person(None, 9, team='B', state='uncertain'))   # an anonymous team-B person
        people.append(person('GK-1', 5, team='B'))
        frames.append({'time': t, 'people': people})
    match = {'players': [entry('A-01', 'A'), entry('A-02', 'A', exit_edge='right'), entry('A-03', 'A'), entry('A-04', 'A')],
             'goalkeepers': [entry('GK-1', 'B', 'GOALKEEPER')], 'retired': [{'id': 'A-09', 'display': 'A-09', 'team': 'A', 'role': 'PLAYER', 'retiredInto': 'A-01'}],
             'excluded': [], 'withdrawn': [], 'referees': []}
    return {'frames': frames, 'match': match, 'config': {'maxPerTeam': 3}, 'summary': {}}


class RosterTests(unittest.TestCase):
    def test_slots_statuses_cap_and_anonymous(self):
        r = roster.build(result())
        by = {s['id']: s for s in r['slots']}
        self.assertEqual(by['A-01']['status'], 'ACTIVE')
        self.assertEqual(by['A-02']['status'], 'OFF_SCREEN')
        self.assertEqual(by['A-03']['status'], 'UNKNOWN')
        self.assertEqual(by['A-04']['status'], 'UNKNOWN')           # never observed in this segment
        self.assertEqual(by['A-04']['observedFrames'], 0)
        self.assertEqual(by['GK-1']['role'], 'GOALKEEPER')
        self.assertEqual(by['A-09']['status'], 'RETIRED')
        self.assertEqual(by['A-02']['lastSeen'], 3.8)
        self.assertEqual(by['A-02']['tracks'], [2])
        self.assertEqual(r['counts'], {'A': 4, 'B': 0, 'goalkeepers': 1})
        self.assertEqual(r['overCap'], {'A': 1})                     # four identities for a cap of three: reported, not hidden
        self.assertEqual(r['anonymous'], {'fragments': 1, 'frames': 15})
        self.assertEqual(r['statuses']['UNKNOWN'], 2)
        s = roster.summary(r)
        self.assertEqual(s['anonymousFrames'], 15)

    def test_a_player_marked_as_left_closes_the_slot(self):
        res = result()
        res['match']['players'][1]['left'] = {'time': 4.0, 'reason': 'substituted or sent off (marked by you)'}
        r = roster.build(res)
        by = {s['id']: s for s in r['slots']}
        self.assertEqual(by['A-02']['status'], 'LEFT')
        self.assertEqual(by['A-02']['left']['time'], 4.0)
        self.assertEqual(r['counts']['A'], 3)
        self.assertEqual(r['overCap'], {})


class ProvisionalPersonTests(unittest.TestCase):
    """Small-scene runs: anonymous local tracks that continue each other (a far player's track breaking every second or
    two) are one provisional person P-n, reported in roster.anonymous and never made an identity."""

    def scene(self, small=True, rival=False, team_b=None, cut_at=None, pitch=None, lanes=1):
        # One far team-B player walks right at 1 m/s; the local tracker loses them every 1.2 s and starts a new track
        # 0.4 s later where they are: tracks 60, 61, 62, 63 (4.4 s in all). Boxes 22 px of 720. cut_at: a camera cut on
        # that frame; pitch(track, frame) -> (x, y, confidence): registered pitch positions (105 x 68 m); lanes: that
        # many such players, 0.1 of the frame height apart (tracks 60-63, 160-163, ...).
        frames = []
        spans = {60: (0, 6), 61: (8, 14), 62: (16, 22), 63: (24, 30)}
        for i in range(32):
            t = round(i*.2, 3)
            people = [person('A-01', 1)]
            for lane in range(lanes):
                for tid, (a, b) in spans.items():
                    if a <= i < b:
                        x = .3+.004*i
                        q = {'track': tid+100*lane, 'id': None, 'team': (team_b if tid == 62 and team_b else 'B'), 'role': 'PLAYER',
                             'state': 'uncertain' if tid != 61 else 'candidate', 'box': [x, .4+.1*lane, .006, .03]}
                        if pitch is not None:
                            px, py, conf = pitch(tid, i)
                            q['pitch'], q['pitchConfidence'] = [px, py], conf
                        people.append(q)
            if rival and 8 <= i < 14:   # a second anonymous track starting at the same moment right next to track 61
                people.append({'track': 70, 'id': None, 'team': 'B', 'role': 'PLAYER', 'state': 'uncertain', 'box': [.3+.004*i+.002, .4, .006, .03]})
            frames.append({'time': t, 'people': people, 'motion': [1, 0, 0, 0, 1, 0], 'cameraReliable': True, 'cut': i == cut_at,
                           'profile': {'smallScene': small}})
        res = result()
        res['frames'] = frames
        res['video'] = {'width': 1280, 'height': 720, 'fps': 5.0}
        if pitch is not None:
            res['tactical'] = {'pitchSize': {'length': 105.0, 'width': 68.0}}
        return res

    def test_anonymous_tracks_that_continue_each_other_are_one_provisional_person(self):
        r = roster.build(self.scene())
        chains = r['anonymous']['provisional']
        self.assertEqual(len(chains), 1, chains)
        self.assertEqual((chains[0]['id'], chains[0]['team'], chains[0]['tracks'], chains[0]['fragments']), ('P-1', 'B', [60, 61, 62, 63], 4))
        self.assertEqual((chains[0]['start'], chains[0]['end'], chains[0]['frames']), (0.0, 5.8, 24))
        self.assertEqual(roster.summary(r)['provisionalPersons'], {'count': 1, 'frames': 24})
        self.assertEqual([s['id'] for s in r['slots'] if s['id'].startswith('P-')], [], 'never an identity or a slot')

    def test_two_tracks_that_could_continue_the_same_one_break_the_chain(self):
        chains = roster.build(self.scene(rival=True))['anonymous']['provisional']
        self.assertFalse([c for c in chains if 61 in c['tracks'] or 70 in c['tracks']], chains)

    def test_a_track_labelled_with_the_other_team_is_not_linked(self):
        chains = roster.build(self.scene(team_b='A'))['anonymous']['provisional']
        self.assertFalse([c for c in chains if 62 in c['tracks']], chains)

    def test_no_link_across_a_camera_cut(self):
        chains = roster.build(self.scene(cut_at=7))['anonymous']['provisional']   # a cut between tracks 60 and 61
        self.assertEqual([c['tracks'] for c in chains], [[61, 62, 63]])

    def test_registered_pitch_positions_decide_the_reach(self):
        # The player walks at 1 m/s on the pitch too (.2 m a frame); track 62's registered positions are 20 m away while
        # its boxes continue the others. Where the registration is confident, the pitch decides (no link to or from 62);
        # under PITCH_MIN the camera-compensated image decides (all linked).
        walk = lambda tid, i, jump, conf: (.3+.2*i/105+(20/105 if tid == 62 and jump else 0.0), .5, conf)
        linked = lambda **kw: [c['tracks'] for c in roster.build(self.scene(**kw))['anonymous']['provisional']]
        self.assertEqual(linked(pitch=lambda tid, i: walk(tid, i, False, .5)), [[60, 61, 62, 63]])
        self.assertEqual(linked(pitch=lambda tid, i: walk(tid, i, True, .5)), [])
        self.assertEqual(linked(pitch=lambda tid, i: walk(tid, i, True, .1)), [[60, 61, 62, 63]])

    def test_links_are_assigned_per_group_of_competing_tracks(self):
        # Twelve far players, each breaking into four tracks: every possible link competes with no other, so each
        # assignment is one end and one start, however many players and tracks the segment holds (a matrix of every
        # end by every start grew with the square of the segment's length).
        from unittest import mock
        from soccer import tracklets
        with mock.patch.object(tracklets, 'linear_sum_assignment', wraps=tracklets.linear_sum_assignment) as solve:
            chains = roster.build(self.scene(lanes=12))['anonymous']['provisional']
        self.assertEqual(sorted(c['tracks'] for c in chains), [[t+100*lane for t in (60, 61, 62, 63)] for lane in range(12)])
        self.assertEqual(solve.call_count, 36)
        self.assertEqual({call.args[0].shape for call in solve.call_args_list}, {(1, 1)})

    def test_broadcast_runs_report_no_provisional_persons(self):
        r = roster.build(self.scene(small=False))
        self.assertNotIn('provisional', r['anonymous'])
        self.assertNotIn('provisionalPersons', roster.summary(r))


if __name__ == '__main__':
    unittest.main()
