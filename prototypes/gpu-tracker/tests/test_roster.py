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


if __name__ == '__main__':
    unittest.main()
