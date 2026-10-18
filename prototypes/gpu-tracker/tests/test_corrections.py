"""Corrections made in the review page (corrections.py): relabelling a result, the saved identities a
continued run starts from, what later runs remember, undo, the files on disk, and the 2D replay's stabilized block of a
corrected result (soccer/stabilized.py; no torch needed). The block's helpers below are shared with test_pipeline_run."""
import contextlib
import copy
import importlib
import inspect
import json
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

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


# ---------- the 2D replay's stabilized block (soccer/stabilized.py) ----------
SHIFT = .5   # m: how far the stub stabilizer moves every observation along the pitch, so stabilized and raw can be told apart


def stabilizer_result(frames_n=40, step=.2):
    """A finished result as pipeline.run has it before the stabilized block: A-01..A-03 and B-01..B-03 running
    across the pitch, A-03 hidden from frame 15 on, an unidentified '?' of team B (track 90) on frames 5-24, frame 7
    without a registration (its people keep a pitch position, which nothing may draw); roster, timeline and visibility
    derived as the pipeline derives them."""
    from soccer import imputation, roster, visibility
    from test_labels import entry as roster_entry
    spec = [(f'{team}-0{k}', 10*(team == 'B')+k, team) for team in 'AB' for k in (1, 2, 3)]
    frames = []
    for i in range(frames_n):
        t = round(10+i*step, 3)
        people = []
        for pid, track, team in spec:
            if pid == 'A-03' and i >= 15:
                continue
            people.append({'time': t, 'track': track, 'box': [.1+.02*track, .4, .02, .1], 'cls': 'player', 'zone': 'inside', 'evidence': 'observed',
                           'id': pid, 'display': pid, 'team': team, 'role': 'PLAYER', 'label': f'PLAYER_TEAM_{team}', 'state': 'confirmed',
                           'identityConfidence': .9, 'roleConfidence': .8, 'score': .8, 'pitch': [round(.2+.04*track+.002*i, 6), round(.3+.02*track, 6)],
                           'pitchConfidence': .8, 'pitchSigma': [1.0, .5]})
        if 5 <= i < 25:
            people.append({'time': t, 'track': 90, 'box': [.8, .4, .02, .1], 'cls': 'player', 'zone': 'inside', 'evidence': 'observed', 'id': None,
                           'display': 'B-?', 'team': 'B', 'role': 'PLAYER', 'label': 'PLAYER_TEAM_B', 'state': 'uncertain', 'identityConfidence': .3,
                           'roleConfidence': .5, 'score': .7, 'pitch': [.85, .4], 'pitchConfidence': .7, 'pitchSigma': [1.0, .5]})
        frames.append({'time': t, 'frame': 300+i, 'people': people, 'ball': {'state': 'UNKNOWN'}, 'rejected': [], 'pitch': {'reliable': True},
                       'calibrated': i != 7})
    match = {'players': [roster_entry(pid, team, first=10.0, last=frames[-1]['time']) for pid, _, team in spec], 'goalkeepers': [], 'referees': [],
             'retired': [], 'excluded': [], 'withdrawn': []}
    summary = {'globalIdentities': {'teamA': 3, 'teamB': 3, 'goalkeepers': 0, 'referees': 0, 'merged': 0}, 'roles': {'mergedIdentities': 0},
               'goalkeeperSlots': {'A': None, 'B': None}}
    result = {'version': 3, 'video': {'width': 1920, 'height': 1080, 'fps': 30.0}, 'config': {'maxPerTeam': 11}, 'frames': frames, 'match': match,
              'events': [], 'summary': summary, 'teams': {'learned': True}, 'tactical': {'pitchSize': {'length': 105.0, 'width': 68.0, 'source': 'standard'}},
              'labels': {e['id']: e['longId'] for e in match['players']}}
    result['roster'] = roster.build(result)
    result['timeline'] = imputation.estimate(result, 105.0, 68.0)
    summary['visibility'] = visibility.annotate(result)
    return result


def stub_stabilize(calls=None, fitted=False):
    """A stabilizer that keeps the contract of soccer/trajectory.py without smoothing anything: every observation moved
    SHIFT metres along the pitch, A-01's frame 10 rejected and bridged (kind I, weight 0, a flag), numpy numbers and a
    NaN in its own blocks (the block must still serialize with allow_nan=False), a runtime in its summary. fitted: like
    the real one, a noise model fitted per run on the observations it is handed (unless it is given one with noise=),
    which moves every observation by its scale: a correction that changes the observations refits it."""
    def stabilize(result, width, height, size=None, noise=None):
        if calls is not None:
            calls.append((width, height, size, noise))
        frames = result['frames']
        T = len(frames)
        L, W = size.length, size.width
        empty = lambda: {k: [None]*T for k in ('x', 'y', 'vx', 'vy', 'sx', 'sy', 'kind', 'w')}
        ids, anonymous, flags = {}, {}, []
        seen = [(t, p) for t, f in enumerate(frames) for p in f['people'] if isinstance(p.get('pitch'), list) and f.get('calibrated')
                and (p.get('id') or (p.get('state') == 'uncertain' and p.get('team') in ('A', 'B')))]
        if noise is None:
            noise = {'sigmaPx': np.float64(.4), 'unfitted': float('nan')}
            if fitted:
                noise['scale'] = .0005*len(seen)   # about .1 m here
        shift = SHIFT+(noise.get('scale') or 0.0)
        for t, p in seen:
            cols = ids.setdefault(p['id'], empty()) if p.get('id') else anonymous.setdefault(str(p['track']), empty())
            cols['x'][t], cols['y'][t] = np.float64(p['pitch'][0]*L+shift), p['pitch'][1]*W
            cols['vx'][t], cols['vy'][t], cols['sx'][t], cols['sy'][t] = 1.004, 0.0, .44, .21
            cols['kind'][t], cols['w'][t] = 'O', 1.0
        if 'A-01' in ids and T > 10 and ids['A-01']['kind'][10] == 'O':
            ids['A-01']['kind'][10], ids['A-01']['w'][10] = 'I', 0.0
            p = next(p for p in frames[10]['people'] if p['id'] == 'A-01')
            flags.append({'id': 'A-01', 'track': p['track'], 'frame': 10, 'time': np.float64(frames[10]['time']+1e-5), 'reason': 'extent',
                          'raw': np.array([p['pitch'][0]*L+4.123, p['pitch'][1]*W]), 'used': None})
        return {'frameTimes': [f['time'] for f in frames], 'ids': ids, 'anonymous': anonymous, 'flags': flags, 'noise': noise,
                'summary': {'observations': np.int64(5), 'rejected': [np.int64(1)], 'runtimeSeconds': .123}, 'constants': {'BRIDGE_GAP': .5},
                'method': 'stub'}
    return stabilize


def without_noise(stabilize):
    """The same stabilizer without a noise parameter: it always fits its own model."""
    return lambda result, width, height, size=None: stabilize(result, width, height, size)


@contextlib.contextmanager
def stabilizer_module(stabilize):
    """soccer.trajectory replaced by a module holding this stabilize (None: a module without one), for the callers that
    import it themselves (soccer/stabilized.py, through pipeline.run and corrections.apply_result)."""
    import soccer
    module = types.ModuleType('soccer.trajectory')
    if stabilize is not None:
        module.stabilize = stabilize
    with mock.patch.dict(sys.modules, {'soccer.trajectory': module}), mock.patch.object(soccer, 'trajectory', module, create=True):
        yield module


def failing(*args, **kwargs):
    raise RuntimeError('stabilizer exploded')


COLUMNS = ('x', 'y', 'kind', 'sigma', 'w')


class CorrectedBlockTests(unittest.TestCase):
    def test_corrections_regenerate_the_block(self):
        from soccer import stabilized
        from test_labels import exclude
        result = stabilizer_result()
        calls = []
        with stabilizer_module(stub_stabilize(calls)):
            stabilized.attach(result, 1920, 1080)
            self.assertIn('B-03', result['stabilized']['ids'])
            C.apply_result(result, [exclude(result, 'B-03')])
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[1][:2], (1920, 1080))
        self.assertNotIn('B-03', result['timeline']['ids'])
        self.assertEqual(set(result['stabilized']['ids']), set(result['timeline']['ids']))
        self.assertEqual(result['summary']['stabilizer']['status'], 'ok')
        # The files a correction writes (corrections.rebuild: allow_nan=False, no numpy default) carry the new block.
        with tempfile.TemporaryDirectory() as folder, stabilizer_module(stub_stabilize(calls)):
            run = Path(folder)/'run-1'
            run.mkdir()
            original = stabilizer_result()
            stabilized.attach(original, 1920, 1080)
            (run/'result.json').write_text(json.dumps(original, separators=(',', ':'), allow_nan=False), encoding='utf-8')
            C.rebuild(run, [exclude(original, 'B-01')])
            written = json.loads((run/'result.json').read_text(encoding='utf-8'))
            self.assertEqual(set(written['stabilized']['ids']), set(written['timeline']['ids']))
            self.assertNotIn('B-01', written['stabilized']['ids'])
            self.assertIn('B-01', json.loads((run/'result.raw.json').read_text(encoding='utf-8'))['stabilized']['ids'])
            # Undoing the last correction gives the original back, byte for byte.
            C.rebuild(run, [])
            self.assertEqual((run/'result.json').read_bytes(), (run/'result.raw.json').read_bytes())
            # A result only looked at (corrections.add makes the new correction against one) gets no block.
            n = len(calls)
            looked = C.apply_result(copy.deepcopy(original), [exclude(original, 'B-01')], block=False)
            self.assertEqual(len(calls), n)
            self.assertNotIn('stabilized', looked)
            self.assertNotIn('stabilizer', looked['summary'])
        # A correction with a failing stabilizer still applies; the block made before it is gone.
        with stabilizer_module(failing), mock.patch('traceback.print_exc'), mock.patch('builtins.print'):
            C.apply_result(result, [exclude(result, 'B-02')])
        self.assertNotIn('B-02', result['timeline']['ids'])
        self.assertNotIn('stabilized', result)
        self.assertEqual(set(result['summary']['stabilizer']), {'status', 'error', 'seconds'})
        self.assertEqual(result['summary']['stabilizer']['status'], 'failed')

    def test_a_correction_moves_only_what_it_corrects(self):
        # The stabilizer fits its noise model per run on the observations it is handed: excluding B-03 changes them, and
        # a refit would move everyone else. The corrected result's block is built with the original's model instead.
        from soccer import stabilized
        from test_labels import exclude
        original = stabilizer_result()
        calls = []
        with stabilizer_module(stub_stabilize(calls, fitted=True)):
            stabilized.attach(original, 1920, 1080)
            fixed = C.apply_result(copy.deepcopy(original), [exclude(original, 'B-03')])
        self.assertIsNone(calls[0][3])
        self.assertEqual(calls[1][3], original['stabilized']['noise'])      # the original's model, handed on
        self.assertEqual(fixed['summary']['stabilizer']['noise'], 'kept')
        self.assertEqual(fixed['stabilized']['noise'], original['stabilized']['noise'])
        self.assertEqual(set(fixed['stabilized']['ids']), set(original['stabilized']['ids'])-{'B-03'})
        for pid, cols in fixed['stabilized']['ids'].items():
            for key in COLUMNS:
                self.assertEqual(cols[key], original['stabilized']['ids'][pid][key], (pid, key))
        self.assertEqual(fixed['stabilized']['anonymous'], original['stabilized']['anonymous'])
        # A stabilizer that takes no model fits it again: the summary says so, and the others move.
        with stabilizer_module(without_noise(stub_stabilize(fitted=True))):
            refit = C.apply_result(copy.deepcopy(original), [exclude(original, 'B-03')])
        self.assertEqual(refit['summary']['stabilizer']['noise'], 'refitted')
        self.assertNotEqual(refit['stabilized']['ids']['A-01']['x'], original['stabilized']['ids']['A-01']['x'])

    def test_a_run_without_a_block_is_corrected_without_one(self):
        # A run analysed before the stabilizer (or whose stabilizer failed): a correction does not add a block, so the 2D
        # view keeps drawing it the old way instead of switching every player to stabilized positions.
        from test_labels import exclude
        calls = []
        result = stabilizer_result()
        with stabilizer_module(stub_stabilize(calls)):
            C.apply_result(result, [exclude(result, 'B-03')])
        self.assertEqual(calls, [])
        self.assertNotIn('stabilized', result)
        self.assertNotIn('stabilizer', result['summary'])
        failed = stabilizer_result()
        failed['summary']['stabilizer'] = {'status': 'failed', 'error': 'RuntimeError: x', 'seconds': {'total': 0.0}}
        with stabilizer_module(stub_stabilize(calls)):
            C.apply_result(failed, [exclude(failed, 'B-03')])
        self.assertEqual(calls, [])
        self.assertNotIn('stabilized', failed)
        self.assertEqual(failed['summary']['stabilizer']['status'], 'failed')   # still says why the run has none

    def test_corrections_and_the_block_never_import_the_detector_stack(self):
        code = ('import sys; sys.path.insert(0, sys.argv[1]); import corrections; from soccer import stabilized; '
                'print(sorted(m for m in ("torch", "torchvision", "ultralytics", "pipeline", "soccer.trajectory") if m in sys.modules))')
        out = subprocess.run([sys.executable, '-c', code, str(Path(C.__file__).parent)], capture_output=True, text=True, check=True).stdout
        self.assertEqual(out.strip(), '[]')

    def test_with_the_real_stabilizer_a_correction_moves_only_what_it_corrects(self):
        trajectory = importlib.import_module('soccer.trajectory')
        if 'noise' not in inspect.signature(trajectory.stabilize).parameters:
            self.skipTest('soccer/trajectory.py stabilize() takes no noise model yet')
        from soccer import stabilized
        from test_labels import exclude
        original = stabilizer_result()
        self.assertEqual(stabilized.attach(original, 1920, 1080)['status'], 'ok')
        fixed = C.apply_result(copy.deepcopy(original), [exclude(original, 'B-03')])
        self.assertEqual(fixed['summary']['stabilizer']['noise'], 'kept')
        for pid, cols in fixed['stabilized']['ids'].items():
            for key in COLUMNS:
                self.assertEqual(cols[key], original['stabilized']['ids'][pid][key], (pid, key))


if __name__ == '__main__':
    unittest.main()
