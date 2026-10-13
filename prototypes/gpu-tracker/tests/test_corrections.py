"""Corrections made in the review page (corrections.py): relabelling a result, the saved identities a
continued run starts from, what later runs remember, undo, and the files on disk."""
import copy
import json
import shutil
import tempfile
import unittest
from pathlib import Path

import scene  # noqa: F401  (adds the prototype directory to sys.path)
import corrections as C

LABELS = {'PLAYER': 'PLAYER_TEAM_{}', 'GOALKEEPER': 'GOALKEEPER_TEAM_{}', 'REFEREE': 'REFEREE'}


def person(pid, track, t, role='PLAYER', team='B', state='confirmed', display=None):
    label = LABELS[role].format(team) if role != 'REFEREE' else 'REFEREE'
    return {'time': t, 'track': track, 'box': [.1, .1, .02, .1], 'cls': 'player', 'zone': 'inside', 'evidence': 'observed', 'id': pid,
            'display': display or pid or '?', 'team': team if role != 'REFEREE' else None, 'role': role, 'label': label, 'state': state,
            'identityConfidence': .9 if pid else .3, 'roleConfidence': .8}


def entry(pid, role, team, first, last, n):
    return {'id': pid, 'longId': C.long_id(pid), 'display': pid, 'team': team, 'role': role, 'label': (LABELS[role].format(team) if role != 'REFEREE' else 'REFEREE'),
            'status': 'ACTIVE', 'firstSeen': first, 'lastSeen': last, 'observations': n, 'identityConfidence': .9, 'teamConfidence': .8,
            'roleConfidence': .8, 'roleLocked': role != 'PLAYER', 'refereeConfidence': 0, 'goalkeeperConfidence': 0, 'roleHistory': [],
            'gallerySize': 6, 'lastBox': [.1, .1, .02, .1], 'lastPitch': None, 'velocity': [0, 0], 'exitEdge': '', 'restored': False,
            'history': [], 'official': None, 'roleEvidence': {}, 'revoked': None, 'keeperGoal': 0.0, 'manual': False}


def build():
    """10 frames: A-01 (track 1) and B-08 (track 7, really a referee) throughout; REF-1 (track 3) in
    frames 0-3 and REF-2 (track 9) in frames 5-9 (a result of an earlier version, which still had referee
    identities); B-03 (track 4) in frames 0-3; an unidentified 'B-?' (track 12) in frames 6-9."""
    frames = []
    for i in range(10):
        t = round(10+i*.2, 3)
        people = [person('A-01', 1, t, team='A'), person('B-08', 7, t)]
        if i <= 3:
            people.append(person('REF-1', 3, t, role='REFEREE'))
            people.append(person('B-03', 4, t))
        if i >= 5:
            people.append(person('REF-2', 9, t, role='REFEREE'))
        if i >= 6:
            people.append(person(None, 12, t, state='uncertain', display='B-?'))
        frames.append({'time': t, 'frame': 300+i, 'people': people, 'ball': {'state': 'UNKNOWN'}, 'rejected': []})
    match = {'players': [entry('A-01', 'PLAYER', 'A', 10, 11.8, 10), entry('B-03', 'PLAYER', 'B', 10, 10.6, 4), entry('B-08', 'PLAYER', 'B', 10, 11.8, 10)], 'goalkeepers': [],
             'referees': [entry('REF-1', 'REFEREE', None, 10, 10.6, 4), entry('REF-2', 'REFEREE', None, 11, 11.8, 5)], 'retired': []}
    events = [{'time': 10.0, 'kind': 'new-identity', 'track': 7, 'playerId': 'B-08', 'message': 'Track 7 becomes new identity TeamB_Player_08'},
              {'time': 11.0, 'kind': 'new-identity', 'track': 9, 'playerId': 'REF-2', 'message': 'REF-2'}]
    summary = {'globalIdentities': {'teamA': 1, 'teamB': 2, 'goalkeepers': 0, 'referees': 2, 'merged': 0}, 'roles': {'mergedIdentities': 0},
               'goalkeeperSlots': {'A': None, 'B': None}}
    result = {'version': 3, 'frames': frames, 'match': match, 'events': events, 'summary': summary, 'teams': {'learned': True},
              'labels': {e['id']: C.long_id(e['id']) for g in match.values() for e in g}}
    gallery = lambda tag: [{'t': 10+k*.4, 'd': {'tag': f'{tag}{k}'}} for k in range(6)]
    snapshot = {'registry': {'version': 1, 'counters': {'A': 1, 'B': 8, 'REF': 2},
                             'identities': [{'id': pid, 'role': role, 'team': team, 'status': 'active', 'gallery': gallery(pid), 'roleLocked': role != 'player',
                                             'lastPitch': [.5, .5], 'exitEdge': ''}
                                            for pid, role, team in (('A-01', 'player', 'A'), ('B-03', 'player', 'B'), ('B-08', 'player', 'B'),
                                                                    ('REF-1', 'referee', None), ('REF-2', 'referee', None))]},
                'teams': {'a': [1, 0], 'b': [0, 1]}, 'run': 'run-1', 'start': 10, 'end': 11.8}
    return result, snapshot


def people_of(result, pid=None, track=None):
    return [p for f in result['frames'] for p in f['people'] if (p.get('id') == pid if pid else p.get('track') == track)]


class CorrectionTests(unittest.TestCase):
    def correct(self, result, snapshot, request, items=()):
        c = C.make(result, snapshot, request, list(items), 'run-1')
        C.apply_result(result, list(items)+[c])
        return c

    def test_marking_someone_a_referee_means_not_a_player(self):
        # Referees are not tracked: "B-08 is really a referee" (as an older page sends it) leaves B-08 out like
        # "not part of the match": every frame relabelled OUT, no REF-n identity, remembered to be left out.
        result, snapshot = build()
        c = C.make(result, snapshot, {'target': {'id': 'B-08', 'time': 10.4}, 'role': 'referee'}, [], 'run-1')
        self.assertEqual((c['after']['role'], c['after']['id'], c['before']['role'], c['before']['team']), ('excluded', None, 'player', 'B'))
        C.apply_result(result, [c])
        mine = people_of(result, track=7)
        self.assertEqual(len(mine), 10)
        self.assertEqual({(p['id'], p['display'], p['label'], p['excludedId'], p['manual']) for p in mine}, {(None, 'OUT', 'EXCLUDED_BY_YOU', 'B-08', True)})
        m = result['match']
        self.assertEqual([e['id'] for e in m['players']], ['A-01', 'B-03'])
        self.assertEqual([e['id'] for e in m['referees']], ['REF-1', 'REF-2'])   # the earlier version's, untouched
        self.assertEqual([e['id'] for e in m['excluded']], ['B-08'])
        g = result['summary']['globalIdentities']
        self.assertEqual((g['teamA'], g['teamB'], g['excluded']), (1, 1, 1))
        event = next(e for e in result['events'] if e['kind'] == 'manual')
        for line in ('CORRECTED BY YOU', 'Global ID: B-08 -> not part of the match', 'Was: team B player', 'Now: not part of the match',
                     'Remembered for later runs of this video: yes'):
            self.assertIn(line, event['message'])
        ident = c['identity']
        self.assertEqual((ident['id'], ident['excluded'], ident['manual'], ident['status']), ('B-08', True, True, 'unknown'))
        self.assertIsNone(c['teams'])
        snap = C.apply_snapshot(snapshot, [c])
        b08 = next(e for e in snap['registry']['identities'] if e['id'] == 'B-08')
        self.assertEqual((b08['excluded'], b08['manual'], b08['status']), (True, True, 'unknown'))
        self.assertFalse([e for e in snap['registry']['identities'] if e['id'] not in ('A-01', 'B-03', 'B-08', 'REF-1', 'REF-2')])

    def test_no_change_and_two_people_at_once_are_refused(self):
        result, snapshot = build()
        with self.assertRaisesRegex(ValueError, 'already is a team A player'):
            C.make(result, snapshot, {'target': {'id': 'A-01'}, 'role': 'player', 'team': 'A'}, [], 'run-1')
        with self.assertRaisesRegex(ValueError, 'both on the video at 00:10.00'):
            C.make(result, snapshot, {'target': {'id': 'B-08'}, 'role': 'player', 'team': 'B', 'into': 'B-03'}, [], 'run-1')
        with self.assertRaisesRegex(ValueError, 'Choose team A or team B'):
            C.make(result, snapshot, {'target': {'id': 'B-08'}, 'role': 'goalkeeper'}, [], 'run-1')
        with self.assertRaisesRegex(ValueError, r'REF-1 is a referee \(not a player\), not a team B player'):
            C.make(result, snapshot, {'target': {'id': 'B-08'}, 'role': 'player', 'team': 'B', 'into': 'REF-1'}, [], 'run-1')
        with self.assertRaisesRegex(ValueError, 'not in this result'):
            C.make(result, snapshot, {'target': {'id': 'B-99'}, 'role': 'referee'}, [], 'run-1')

    def test_unidentified_track_joins_an_identity_it_never_overlaps(self):
        result, snapshot = build()
        c = self.correct(result, snapshot, {'target': {'track': 12}, 'role': 'player', 'team': 'B', 'into': 'B-03'})
        self.assertEqual(c['after']['id'], 'B-03')
        self.assertIsNone(c['identity'])   # a local track only: nothing saved to recognise them by
        self.assertEqual({p['id'] for p in people_of(result, track=12)}, {'B-03'})
        b03 = next(e for e in result['match']['players'] if e['id'] == 'B-03')
        self.assertEqual((b03['observations'], b03['firstSeen'], b03['lastSeen']), (8, 10, 11.8))
        self.assertEqual(C.apply_snapshot(snapshot, [c]), snapshot)   # nothing to change in the saved identities

    def test_an_ignored_person_can_be_made_a_player(self):
        # Someone the tracker ignored as referee-like (NON_TARGET, no identity) who is really a team A player.
        result, snapshot = build()
        for f in result['frames'][6:]:
            for p in f['people']:
                if p['track'] == 12:
                    p.update(id=None, display='IGNORE', label='NON_TARGET', team=None, role='NON_TARGET', state='nontarget', hint='REFEREE_LIKE')
        c = self.correct(result, snapshot, {'target': {'track': 12}, 'role': 'player', 'team': 'A'})
        self.assertEqual((c['before']['role'], c['after']['id']), ('nontarget', 'A-02'))
        self.assertEqual({(p['id'], p['team'], p['state']) for p in people_of(result, track=12)}, {('A-02', 'A', 'confirmed')})
        self.assertIn('Was: ignored (not a player)', next(e for e in result['events'] if e['kind'] == 'manual')['message'])

    def test_unidentified_track_becomes_a_new_identity_in_this_result_only(self):
        result, snapshot = build()
        c = self.correct(result, snapshot, {'target': {'track': 12}, 'role': 'player', 'team': 'A'})
        self.assertEqual(c['after']['id'], 'A-02')
        self.assertEqual({(p['id'], p['team'], p['identityConfidence']) for p in people_of(result, track=12)}, {('A-02', 'A', 1.0)})
        new = next(e for e in result['match']['players'] if e['id'] == 'A-02')
        self.assertEqual((new['observations'], new['trackOnly'], new['manual']), (4, 12, True))
        self.assertIn('Remembered for later runs of this video: no', next(e for e in result['events'] if e['kind'] == 'manual')['message'])
        self.assertEqual(C.seeds([c])['identities'], [])

    def test_not_part_of_the_match_leaves_the_person_out(self):
        result, snapshot = build()
        c = self.correct(result, snapshot, {'target': {'id': 'B-08'}, 'role': 'excluded'})
        self.assertIsNone(c['after']['id'])
        mine = people_of(result, track=7)
        self.assertEqual({(p['id'], p['display'], p['state'], p['label'], p['excludedId']) for p in mine}, {(None, 'OUT', 'rejected', 'EXCLUDED_BY_YOU', 'B-08')})
        m = result['match']
        self.assertEqual([e['id'] for e in m['players']], ['A-01', 'B-03'])
        self.assertEqual([e['id'] for e in m['excluded']], ['B-08'])
        self.assertEqual(m['retired'], [])
        self.assertEqual(result['summary']['globalIdentities']['excluded'], 1)
        snap = C.apply_snapshot(snapshot, [c])
        b08 = next(e for e in snap['registry']['identities'] if e['id'] == 'B-08')
        self.assertEqual((b08['excluded'], b08['manual'], b08['status']), (True, True, 'unknown'))
        seeds = C.seeds([c])['identities']
        self.assertEqual([(e['id'], e['excluded'], e['status']) for e in seeds], [('B-08', True, 'unknown')])

    def test_a_later_correction_of_the_same_person_wins_and_undo_cascades(self):
        result, snapshot = build()
        c1 = C.make(result, snapshot, {'target': {'id': 'B-08'}, 'role': 'goalkeeper', 'team': 'B'}, [], 'run-1')
        self.assertEqual(c1['after']['id'], 'GK-1')
        C.apply_result(result, [c1])
        snap = C.apply_snapshot(snapshot, [c1])
        c2 = C.make(result, snap, {'target': {'id': 'GK-1'}, 'role': 'player', 'team': 'A'}, [c1], 'run-1')
        self.assertEqual(c2['after']['id'], 'A-02')
        fresh, _ = build()
        C.apply_result(fresh, [c1, c2])
        self.assertEqual({p['id'] for p in people_of(fresh, track=7)}, {'A-02'})
        memory = C.seeds([c1, c2])
        self.assertEqual([(e['id'], e['role'], e['team']) for e in memory['identities']], [('A-02', 'player', 'A')])
        self.assertEqual(memory['teams'], snapshot['teams'])   # team letters need the kit model they were given under
        kept, gone = C.remove([c1, c2], c1['id'])
        self.assertEqual((kept, gone), ([], {c1['id'], c2['id']}))

    def test_seed_state_adds_remembered_people_to_a_new_run(self):
        result, snapshot = build()
        c = C.make(result, snapshot, {'target': {'id': 'B-08'}, 'role': 'goalkeeper', 'team': 'B'}, [], 'run-1')
        self.assertEqual(c['after']['id'], 'GK-1')
        fresh = C.seed_state(None, [c])
        self.assertIsNone(fresh.get('run'))   # a fresh run is not 'continued'
        self.assertEqual(fresh['manualSeeds'], ['GK-1'])
        gk = fresh['registry']['identities'][0]
        self.assertEqual((gk['id'], gk['role'], gk['team'], gk['manual'], gk['status'], gk['lastPitch'], gk['keeperGoal']), ('GK-1', 'goalkeeper', 'B', True, 'missing', None, 1.0))
        self.assertEqual(fresh['teams'], snapshot['teams'])
        continued = C.seed_state(C.apply_snapshot(snapshot, [c]), [c])
        self.assertEqual(continued['manualSeeds'], [])   # already in the continued identities
        self.assertEqual(continued['run'], 'run-1')
        self.assertIsNone(C.seed_state(None, []))

    def test_files_are_rebuilt_from_the_originals_and_undo_is_exact(self):
        result, snapshot = build()
        root = Path(tempfile.mkdtemp())
        try:
            run = root/'run-1'
            run.mkdir()
            (run/'result.json').write_text(json.dumps(result, separators=(',', ':')), encoding='utf-8')
            (run/'identities.json').write_text(json.dumps(snapshot), encoding='utf-8')
            original = {name: (run/name).read_bytes() for name in ('result.json', 'identities.json')}
            listed = C.add(root, run, {'target': {'id': 'B-08', 'time': 10.4}, 'role': 'goalkeeper', 'team': 'B'})
            self.assertEqual((listed['from'], listed['to'], listed['what'], listed['remembered'], listed['thisRun']), ('B-08', 'GK-1', 'goalkeeper (team B)', True, True))
            shown = json.loads((run/'result.json').read_text(encoding='utf-8'))
            self.assertEqual({p['id'] for p in people_of(shown, track=7)}, {'GK-1'})
            self.assertTrue(any(e['id'] == 'GK-1' for e in json.loads((run/'identities.json').read_text(encoding='utf-8'))['registry']['identities']))
            self.assertEqual((run/'result.raw.json').read_bytes(), original['result.json'])
            # A second correction is applied on top of the first, from the originals.
            C.add(root, run, {'target': {'id': 'GK-1'}, 'role': 'excluded'})
            shown = json.loads((run/'result.json').read_text(encoding='utf-8'))
            self.assertEqual({p['label'] for p in people_of(shown, track=7)}, {'EXCLUDED_BY_YOU'})
            self.assertEqual(len(C.load(root)), 2)
            first = C.load(root)[0]['id']
            gone = C.drop(root, run, first)   # undoing the first undoes the one that built on it
            self.assertEqual(len(gone), 2)
            self.assertEqual(C.load(root), [])
            self.assertFalse((root/'corrections.json').exists())
            for name, data in original.items():
                self.assertEqual((run/name).read_bytes(), data, name)
        finally:
            shutil.rmtree(root, ignore_errors=True)


class LeftTheMatchTests(unittest.TestCase):
    """Phase 2: a player marked as substituted or sent off closes the roster slot from that time; the
    observations before it are kept, and the roster and visibility are re-derived with the correction."""

    def test_left_closes_the_slot_and_is_remembered_as_substituted(self):
        result, _ = build()
        result['video'] = {'width': 1280, 'height': 720}
        result['tactical'] = {'pitchSize': {'length': 105, 'width': 68}}
        snapshot = {'registry': {'version': 1, 'identities': [{'id': 'A-01', 'role': 'player', 'team': 'A', 'status': 'missing', 'gallery': ['x']}], 'counters': {}}, 'teams': None}
        c = C.make(result, snapshot, {'target': {'id': 'A-01', 'time': 10.8}, 'role': 'left', 'team': None}, [], 'run-1')
        self.assertEqual(c['after'], {'role': 'left', 'team': 'A', 'id': 'A-01', 'into': False})
        self.assertIsNone(c['identity'])
        C.apply_result(result, [c])
        entry = C.find_entry(result['match'], 'A-01')
        self.assertEqual(entry[0], 'players')
        self.assertEqual(entry[1]['status'], 'LEFT')
        self.assertEqual(entry[1]['left']['time'], 10.8)
        self.assertTrue(all(p['id'] == 'A-01' for p in people_of(result, 'A-01')))        # observations untouched
        self.assertEqual(result['roster']['statuses'].get('LEFT'), 1)
        self.assertIn('visibility', result['summary'])
        self.assertTrue(any(e['kind'] == 'manual' and 'left the match' in e['message'] for e in result['events']))
        with self.assertRaises(ValueError):
            C.make(result, snapshot, {'target': {'id': 'A-01', 'time': 11.0}, 'role': 'left', 'team': None}, [c], 'run-1')
        with self.assertRaises(ValueError):                                                 # only an identified player can leave
            C.make(result, snapshot, {'target': {'track': 12, 'time': 11.0}, 'role': 'left', 'team': None}, [c], 'run-1')
        saved = C.apply_snapshot(snapshot, [c])
        self.assertEqual(saved['registry']['identities'][0]['status'], 'substituted')
        self.assertEqual(C.describe('left'), 'left the match (substituted or sent off)')


if __name__ == '__main__':
    unittest.main()
