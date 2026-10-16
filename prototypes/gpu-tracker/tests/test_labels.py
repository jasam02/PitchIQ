"""Display labels regenerated from the confirmed roster (soccer/labels.py): the first outfield slots numbered
without gaps, extra identities unconfirmed, deleting a false player renumbers the rest, and the internal
identities never change."""
import copy
import unittest

import scene  # noqa: F401  (adds the prototype directory to sys.path)
import corrections as C
from soccer import labels, roster


def person(pid, track, t, team, role='PLAYER'):
    display = f'GK-{team}' if role == 'GOALKEEPER' else pid
    return {'time': t, 'track': track, 'box': [.1+.01*track, .2, .02, .1], 'cls': 'player', 'zone': 'inside', 'evidence': 'observed', 'id': pid,
            'display': display, 'team': team, 'role': role, 'label': f'{role}_TEAM_{team}', 'state': 'confirmed', 'identityConfidence': .9,
            'roleConfidence': .8, 'pitch': [.3+.01*track, .5], 'pitchConfidence': .9}


def entry(pid, team, role='PLAYER', first=10.0, last=11.8):
    display = f'GK-{team}' if role == 'GOALKEEPER' else pid
    return {'id': pid, 'longId': C.long_id(pid), 'display': display, 'team': team, 'role': role, 'label': f'{role}_TEAM_{team}', 'status': 'ACTIVE',
            'firstSeen': first, 'lastSeen': last, 'observations': 10, 'identityConfidence': .9, 'teamConfidence': .8, 'roleConfidence': .8,
            'roleLocked': role != 'PLAYER', 'refereeConfidence': 0, 'goalkeeperConfidence': 0, 'roleHistory': [], 'gallerySize': 6,
            'lastBox': [.1, .1, .02, .1], 'lastPitch': [.4, .4], 'velocity': [0, 0], 'exitEdge': '', 'restored': False, 'history': [],
            'official': None, 'roleEvidence': {}, 'revoked': None, 'keeperGoal': 0.0, 'manual': False}


def build(b_players=12, a_players=10):
    """The user's scenario: team B has B-01..B-12 confirmed (B-07 and B-08 are really referees), team A has
    ten, each team a goalkeeper; everyone on every frame."""
    people_spec = [(f'A-{k:02d}', k, 'A', 'PLAYER') for k in range(1, a_players+1)] + \
                  [(f'B-{k:02d}', 20+k, 'B', 'PLAYER') for k in range(1, b_players+1)] + [('GK-1', 50, 'A', 'GOALKEEPER'), ('GK-2', 51, 'B', 'GOALKEEPER')]
    frames = [{'time': round(10+i*.2, 3), 'frame': 300+i, 'people': [person(pid, track, round(10+i*.2, 3), team, role) for pid, track, team, role in people_spec],
               'ball': {'state': 'UNKNOWN'}, 'rejected': [], 'pitch': {'reliable': True}, 'calibrated': True}
              for i in range(10)]
    match = {'players': [entry(pid, team) for pid, _, team, role in people_spec if role == 'PLAYER'],
             'goalkeepers': [entry(pid, team, 'GOALKEEPER') for pid, _, team, role in people_spec if role == 'GOALKEEPER'],
             'referees': [], 'retired': [], 'excluded': [], 'withdrawn': []}
    summary = {'globalIdentities': {'teamA': a_players, 'teamB': b_players, 'goalkeepers': 2, 'referees': 0, 'merged': 0}, 'roles': {'mergedIdentities': 0},
               'goalkeeperSlots': {'A': 'GK-1', 'B': 'GK-2'}}
    result = {'version': 3, 'video': {'width': 1920, 'height': 1080, 'fps': 30.0}, 'config': {'maxPerTeam': 11}, 'frames': frames, 'match': match,
              'events': [], 'summary': summary, 'teams': {'learned': True}, 'tactical': {'pitchSize': {'length': 105.0, 'width': 68.0}},
              'labels': {e['id']: C.long_id(e['id']) for g in match.values() for e in g}}
    result['roster'] = roster.build(result)
    return result


def exclude(result, pid, when=10.4):
    """A correction marking an identity as not part of the match (a referee), as corrections.make builds it."""
    before = {'id': pid, 'track': None, 'role': 'player', 'team': 'B', 'display': pid}
    return {'id': f'c-{pid}', 'run': 'run-1', 'made': '2026-10-15T00:00:00', 'time': when, 'target': {'id': pid, 'track': None, 'trackAt': None},
            'before': before, 'after': {'role': 'excluded', 'team': None, 'id': None, 'into': False}, 'identity': None, 'teams': None}


def displays(result, team):
    return {e['id']: e['display'] for e in result['match']['players'] if e['team'] == team}


def core(counts):
    """The four counts of a team (the rest names the unconfirmed identities and their participation)."""
    return {k: counts[k] for k in ('numbered', 'active', 'left', 'unconfirmed')}


class LabelTests(unittest.TestCase):
    def test_only_the_first_ten_outfield_players_are_numbered_and_the_rest_wait(self):
        result = build()
        out = labels.assign(result)
        b = displays(result, 'B')
        self.assertEqual([b[f'B-{k:02d}'] for k in range(1, 11)], [f'B-{k:02d}' for k in range(1, 11)])
        self.assertEqual((b['B-11'], b['B-12']), ('B-?', 'B-?'))
        self.assertEqual(core(out['counts']['B']), {'numbered': 10, 'active': 10, 'left': 0, 'unconfirmed': 2})
        self.assertEqual(core(out['counts']['A']), {'numbered': 10, 'active': 10, 'left': 0, 'unconfirmed': 0})
        self.assertEqual(out['counts']['B']['unconfirmedIds'], ['B-11', 'B-12'])
        # The frames and the labels map follow; the internal id stays on every frame.
        for f in result['frames']:
            by = {p['id']: p for p in f['people']}
            self.assertEqual(by['B-11']['display'], 'B-?')
            self.assertEqual(by['B-03']['display'], 'B-03')
            self.assertEqual(by['GK-2']['display'], 'GK-B')
        self.assertEqual(result['labels']['B-05'], 'TeamB_Player_05')
        self.assertEqual(result['summary']['labelling']['outfieldPerTeam'], 10)
        self.assertEqual(next(e for e in result['match']['players'] if e['id'] == 'B-12')['numbered'], False)

    def test_deleting_two_false_players_renumbers_the_rest_without_touching_internal_identities(self):
        # BEFORE: B-01..B-12 (two of them referees). DELETE B-07 and B-08. AFTER: ten numbered B-01..B-10, the
        # previous B-09..B-12 shown as B-07..B-10, the same internal identities, goalkeepers untouched.
        result = build()
        labels.assign(result)
        raw = copy.deepcopy(result)
        C.apply_result(result, [exclude(result, 'B-07'), exclude(result, 'B-08', when=10.6)])
        b = displays(result, 'B')
        self.assertEqual(set(b), {f'B-{k:02d}' for k in range(1, 13)} - {'B-07', 'B-08'}, 'internal identities of the remaining players unchanged')
        self.assertEqual(sorted(b.values()), [f'B-{k:02d}' for k in range(1, 11)], 'exactly ten, numbered without gaps')
        self.assertEqual((b['B-09'], b['B-10'], b['B-11'], b['B-12']), ('B-07', 'B-08', 'B-09', 'B-10'))
        self.assertEqual([b[f'B-{k:02d}'] for k in range(1, 7)], [f'B-{k:02d}' for k in range(1, 7)])
        self.assertEqual(len(set(b.values())), len(b), 'no duplicate labels')
        self.assertEqual(displays(result, 'A'), {f'A-{k:02d}': f'A-{k:02d}' for k in range(1, 11)}, 'team A untouched')
        # Every frame: the deleted people are OUT, the others keep their internal id and show the new label.
        for f in result['frames']:
            by = {p['id']: p for p in f['people'] if p.get('id')}
            self.assertEqual(by['B-11']['display'], 'B-09')
            self.assertEqual(by['B-12']['display'], 'B-10')
            self.assertEqual(by['B-09']['display'], 'B-07')
            self.assertNotIn('B-07', by)
            self.assertNotIn('B-08', by)
            self.assertEqual(by['GK-2']['display'], 'GK-B')
            self.assertEqual(by['GK-1']['display'], 'GK-A')
            self.assertEqual(sum(1 for p in f['people'] if p.get('label') == 'EXCLUDED_BY_YOU'), 2)
        # The roster, the 2D view's source of labels, and the labels map follow; the timeline keys stay internal.
        slots = {s['id']: s for s in result['roster']['slots']}
        self.assertEqual((slots['B-11']['display'], slots['B-12']['display']), ('B-09', 'B-10'))
        self.assertEqual(result['roster']['counts']['B'], 10)
        self.assertEqual(result['roster']['overCap'], {})
        self.assertEqual(result['labels']['B-11'], 'TeamB_Player_09')
        self.assertIn('B-11', result['timeline']['ids'])
        self.assertEqual(core(result['summary']['labelling']['teams']['B']), {'numbered': 10, 'active': 10, 'left': 0, 'unconfirmed': 0})
        event = [e for e in result['events'] if e['kind'] == 'labels']
        self.assertTrue(event and event[-1]['message'].startswith('PLAYER LABELS RENUMBERED'), [e['message'][:40] for e in result['events']])
        self.assertIn('B-11: B-? -> B-09', event[-1]['message'])
        self.assertIn('B-09: B-09 -> B-07', event[-1]['message'])
        # Undo (rebuilding from the untouched originals) gives the labels before the deletion back.
        again = copy.deepcopy(raw)
        C.apply_result(again, [])
        self.assertEqual(displays(again, 'B'), displays(raw, 'B'))

    def test_a_substitute_takes_the_next_number_and_a_player_who_left_keeps_theirs(self):
        result = build(b_players=11)
        labels.assign(result)
        self.assertEqual(displays(result, 'B')['B-11'], 'B-?')
        left = next(e for e in result['match']['players'] if e['id'] == 'B-04')
        left['left'] = {'time': 10.6, 'reason': 'substituted or sent off (marked by you)'}
        out = labels.assign(result)
        b = displays(result, 'B')
        self.assertEqual(b['B-04'], 'B-04', 'a player who left keeps the number')
        self.assertEqual(b['B-11'], 'B-11', 'the substitute takes the next number: ten active plus one who left')
        self.assertEqual(core(out['counts']['B']), {'numbered': 11, 'active': 10, 'left': 1, 'unconfirmed': 0})

    def test_fewer_than_ten_confirmed_players_are_not_padded(self):
        result = build(b_players=7)
        out = labels.assign(result)
        self.assertEqual(sorted(displays(result, 'B').values()), [f'B-{k:02d}' for k in range(1, 8)])
        self.assertEqual(out['counts']['B']['numbered'], 7)

    def test_labels_are_stable_when_nothing_about_the_roster_changed(self):
        result = build()
        labels.assign(result)
        before = displays(result, 'B')
        out = labels.assign(result)
        self.assertEqual(displays(result, 'B'), before)
        self.assertEqual(out['changed'], {})


class ParticipationTests(unittest.TestCase):
    """Over the cap, the slots go to the identities with the most participation evidence, not to the first created:
    the user's far-touchline player (B-14, tracked for the whole minute, identified late) is numbered; two people seen
    for four seconds at the bottom edge are not. Internal ids never change; the roster and the timeline follow."""

    def scene(self):
        # Team B: nine of B-01..B-10 on every frame (B-05 is a sideline person: at the bottom edge for the first 20 of
        # 300 frames), B-13 another one (inside but only 3 s), B-14 the far-touchline player identified from
        # frame 40 on, B-12 excluded earlier: 12 identities for 10 slots.
        result = build(b_players=10)
        frames = result['frames']
        fps = result['video']['fps']
        n = 300
        frames[:] = [{'time': round(10+i/fps, 3), 'frame': 300+i, 'people': [], 'ball': {'state': 'UNKNOWN'}, 'rejected': [], 'pitch': {'reliable': True}, 'calibrated': True}
                     for i in range(n)]
        spec = [(f'B-{k:02d}', 20+k) for k in range(1, 11) if k != 5] + [('B-14', 34)]
        for i, f in enumerate(frames):
            t = f['time']
            for pid, track in spec:
                if pid == 'B-14' and i < 40:
                    continue        # identified only from frame 40 on (small and far at first)
                f['people'].append(person(pid, track, t, 'B'))
            if i < 20:
                q = person('B-05', 25, t, 'B')
                q['box'] = [.9, .93, .02, .08]     # cut by the bottom edge of the frame
                f['people'].append(q)
            if 3 <= i < 93:
                f['people'].append(person('B-13', 33, t, 'B'))
        match = result['match']
        match['players'] = [e for e in match['players'] if e['team'] != 'B'] + [entry(pid, 'B') for pid in [f'B-{k:02d}' for k in range(1, 11)] + ['B-13', 'B-14']]
        for e in match['players']:
            e['display'] = e['id']
        result['roster'] = roster.build(result)
        return result

    def test_the_full_time_player_identified_late_takes_the_slot_of_the_sideline_people(self):
        result = self.scene()
        out = labels.assign(result)
        b = displays(result, 'B')
        self.assertEqual(b['B-14'], 'B-10', b)
        self.assertEqual((b['B-05'], b['B-13']), ('B-?', 'B-?'), b)
        self.assertEqual([b[f'B-{k:02d}'] for k in (1, 2, 3, 4, 6, 7, 8, 9, 10)], ['B-01', 'B-02', 'B-03', 'B-04', 'B-05', 'B-06', 'B-07', 'B-08', 'B-09'])
        self.assertEqual(core(out['counts']['B']), {'numbered': 10, 'active': 10, 'left': 0, 'unconfirmed': 2})
        self.assertEqual(out['counts']['B']['unconfirmedIds'], ['B-05', 'B-13'])
        e14 = next(e for e in result['match']['players'] if e['id'] == 'B-14')
        self.assertTrue(e14['confirmed'] and e14['participation']['presence'] > 8.0, e14['participation'])
        e05 = next(e for e in result['match']['players'] if e['id'] == 'B-05')
        self.assertFalse(e05['confirmed'])
        self.assertGreater(e05['participation']['edgeShare'], .9)
        self.assertTrue(all(p['id'] == 'B-14' and p['display'] == 'B-10' for f in result['frames'] for p in f['people'] if p['id'] == 'B-14'))
        text = '\n'.join(labels.describe(out))
        self.assertIn('B-05 unconfirmed: seen 0.7 s, 0.0 s clear of the frame edges', text)
        # Through a correction pass the roster and the timeline follow: no estimate for an unconfirmed slot.
        C.apply_result(result, [])
        slots = {s['id']: s for s in result['roster']['slots']}
        self.assertTrue(slots['B-14']['confirmed'] and not slots['B-05']['confirmed'])
        self.assertEqual(slots['B-14']['display'], 'B-10')
        kinds05 = result['timeline']['ids']['B-05']['kind']
        self.assertEqual(set(kinds05[20:]), {'U'}, 'no estimated positions for an unconfirmed identity')
        self.assertEqual(set(kinds05[:20]), {'O'}, 'its observations are kept')
        self.assertNotIn('U', result['timeline']['ids']['B-14']['kind'][40:])

    def test_equal_evidence_keeps_the_creation_order(self):
        result = build(b_players=12)
        labels.assign(result)
        b = displays(result, 'B')
        self.assertEqual((b['B-11'], b['B-12']), ('B-?', 'B-?'))
        self.assertEqual([b[f'B-{k:02d}'] for k in range(1, 11)], [f'B-{k:02d}' for k in range(1, 11)])

    def test_a_player_who_left_is_never_demoted_and_frees_the_slot(self):
        result = self.scene()
        gone = next(e for e in result['match']['players'] if e['id'] == 'B-01')
        gone['left'] = {'time': 10.6, 'reason': 'substituted or sent off (marked by you)'}
        labels.assign(result)
        b = displays(result, 'B')
        self.assertEqual(b['B-01'], 'B-01', 'a player who left keeps the number')
        self.assertEqual(b['B-13'], 'B-10', 'the freed slot goes to the next best evidence; numbers follow the creation order')
        self.assertEqual(b['B-14'], 'B-11')
        self.assertEqual(b['B-05'], 'B-?')

