"""Possession (soccer/possession.py) on synthetic scenes: a pass between teammates keeps possession, an interception
changes it once (who lost it, who won it, the cause), a half-second duel flicker and a deflection do not, a ball out
of play is dead until the restart and the restart belongs to the taker's team, a ball out of view is unknown, change
ids are stable, a correction of a player's team changes the carrier's team, and the record is the contract's JSON. The
PitchIQ-specific rules of stage B: a nearTrack from a first pass that followed another ball is no evidence, a person
without a team at the ball is evidence for nobody, a goalkeeper without a team plays for the team defending his goal, a
camera cut restarts possession (a low-confidence change), and the carrier is the player near on both metrics. The rules
of possession-3 (PossessionV3): a change rests on an observed control (an interpolated or image-only spell starts none),
an excursion beyond a line that the ball path jumped to, or with another ball in play, is no ball out of play, a
stoppage with the ball in view (everyone slows down, nobody plays the ball) is dead, wonBy is the player who then holds
the ball, a back-and-forth pair is low confidence, and the players' image speeds follow the camera motion. The rules of
possession-4 (PossessionV4): a result of 1 to 31 frames gives a record, a ball held in play is no stoppage while a placed
ball the taker walks up to stays dead, a keeper's catch is his team's from the catch, a ball knocked past a defender is
not his, a result without ball.origin still has changes, the first-pass check doubles the image rows, a ball placed
after an unseen stretch is dead, a replay between two cuts names nobody, an identity on another team's player plays for
the team its frames name, and a change's confidence is high or uncertain.

The scene: a camera whose image maps the pitch linearly at 33 px per metre across and 16.5 px per metre down (so the
image metric of ball_path.within_reach, with its rows foreshortened 2x and metres from 1.8 m tall boxes, measures the
same metres as the pitch), people as 1.8 m boxes standing on their pitch positions, the ball as ball_path's final
path records it (TRACKED with motion and speed, nearTrack at a player's feet, a pitch position under a HIGH
registration), 30 frames per second.

Run:  python -m unittest test_possession          (from this directory)
"""
import copy
import json
import math
import unittest

import numpy as np

import scene  # noqa: F401  (adds the prototype directory to sys.path)
import corrections
from soccer import possession

FPS = 30.0
W, H = 4000, 1400           # px: 105 m x 33 px across, 68 m x 16.5 px down, with margins
PX, PY = 33.0, 16.5         # px per metre across and down
U0, V0 = 50.0, 120.0


def image(x, y):
    return U0+PX*x, V0+PY*y


def person(pid, team, x, y, track, role='PLAYER'):
    u, v = image(x, y)
    h, w = 1.8*PX, .7*PX
    return {'track': track, 'box': [(u-w/2)/W, (v-h)/H, w/W, h/H], 'id': pid, 'display': pid, 'team': team, 'role': role,
            'label': f'{role}_TEAM_{team}', 'state': 'confirmed', 'identityConfidence': .9, 'pitch': [x/105.0, y/68.0],
            'pitchConfidence': .9, 'pitchSigma': [.3, .3]}


def ball(x, y, speed, near=None, state='TRACKED'):
    if state == 'UNKNOWN':
        return {'state': 'UNKNOWN', 'phase': 'PATH', 'origin': 'UNKNOWN', 'motion': 'UNKNOWN'}
    u, v = image(x, y)
    d = 7.0
    out = {'state': state, 'phase': 'PATH', 'observed': state == 'TRACKED', 'origin': 'OBSERVATION' if state == 'TRACKED' else 'INTERPOLATION',
           'motion': ('STATIONARY' if speed < 1 else 'ROLLING' if speed < 8 else 'FAST') if state == 'TRACKED' else 'OCCLUDED',
           'speed': round(speed, 2), 'box': [(u-d/2)/W, (v-d)/H, d/W, d/H], 'center': [u/W, (v-d/2)/H],
           'pitch': [x/105.0, y/68.0], 'pitchConfidence': .9, 'pitchSigma': [.3, .3]}
    if near is not None:
        out['nearTrack'] = near
    return out


class Scene:
    """Players moving on straight lines between keyframes and a ball given per time; build() makes the result."""

    def __init__(self, seconds):
        self.n = int(round(seconds*FPS))
        self.players = {}       # id: (team, track, [(t, x, y)])
        self.ball = None        # function t -> (x, y) or None (unseen)

    def add(self, pid, team, track, keys):
        self.players[pid] = (team, track, keys)

    def at(self, pid, t):
        keys = self.players[pid][2]
        if t <= keys[0][0]:
            return keys[0][1:]
        for (t0, x0, y0), (t1, x1, y1) in zip(keys, keys[1:]):
            if t0 <= t <= t1:
                s = (t-t0)/max(t1-t0, 1e-9)
                return x0+s*(x1-x0), y0+s*(y1-y0)
        return keys[-1][1:]

    def build(self, first=300):
        frames, prev = [], None
        for i in range(self.n):
            t = (first+i)/FPS
            people = []
            for pid, (team, track, _) in self.players.items():
                x, y = self.at(pid, t)
                people.append(person(pid, team, x, y, track, 'GOALKEEPER' if pid.startswith('GK') else 'PLAYER'))
            b = self.ball(t)
            if b is None:
                frames.append({'time': round(t, 6), 'frame': first+i, 'people': people, 'ball': ball(0, 0, 0, state='UNKNOWN'),
                               'profile': {'smallScene': False}})
                prev = None
                continue
            x, y = b
            speed = 0.0 if prev is None else math.hypot(x-prev[0], y-prev[1])*FPS
            near = next((p['track'] for p in people if math.hypot(p['pitch'][0]*105-x, p['pitch'][1]*68-y) < .4), None)
            frames.append({'time': round(t, 6), 'frame': first+i, 'people': people, 'ball': ball(x, y, speed, near),
                           'profile': {'smallScene': False}})
            prev = (x, y)
        players = [{'id': pid, 'display': pid, 'team': team, 'role': 'PLAYER', 'confirmed': True}
                   for pid, (team, _, _) in self.players.items() if not pid.startswith('GK')]
        keepers = [{'id': pid, 'display': pid, 'team': team, 'role': 'GOALKEEPER', 'confirmed': True}
                   for pid, (team, _, _) in self.players.items() if pid.startswith('GK')]
        return {'version': 3, 'video': {'name': 'synthetic.mp4', 'width': W, 'height': H, 'fps': FPS, 'frames': self.n},
                'frames': frames, 'match': {'players': players, 'goalkeepers': keepers, 'referees': [], 'retired': [], 'excluded': [], 'withdrawn': []},
                'tactical': {'pitchSize': {'length': 105.0, 'width': 68.0, 'source': 'standard'}, 'goalEnds': {'left': 'A', 'right': 'B'}}}


def feet(sc, pid, t, ahead=.3):
    """The ball at a player's feet, a little ahead of him."""
    x, y = sc.at(pid, t)
    return x+ahead, y


def flight(a, b, t0, t1, t):
    s = min(1.0, max(0.0, (t-t0)/(t1-t0)))
    return a[0]+s*(b[0]-a[0]), a[1]+s*(b[1]-a[1])


def run_of(rec, t):
    for r in rec['timeline']:
        if r['t0']-1e-6 <= t <= r['t1']+1e-6:
            return r
    return None


def team_at(rec, t):
    r = run_of(rec, t)
    return None if r is None else r['team']


def passing_scene(receiver_team='A', deflect=False):
    """A-01 dribbles from 10.0 s to 12.0 s, then passes 15 m to A-02 (or B-05: an interception), who keeps the ball
    from 13.5 s to 16 s. With deflect, B-07 stands in the pass's path at 12.75 s and knocks it sideways (a bend of 13
    m/s: a touch), and A-03 picks it up a second later."""
    sc = Scene(6.5)
    sc.add('A-01', 'A', 1, [(10.0, 40, 30), (12.0, 46, 30), (16.5, 47, 32)])
    receiver = 'A-02' if receiver_team == 'A' else 'B-05'
    sc.add(receiver, receiver_team, 2, [(10.0, 61.5, 30), (13.5, 61.5, 30), (16.5, 70, 33)])
    sc.add('B-07' if deflect else 'B-08', 'B', 3, [(10.0, 54.0, 29.7), (16.5, 54.0, 29.7)] if deflect else [(10.0, 40, 50), (16.5, 45, 52)])
    sc.add('A-03', 'A', 4, [(10.0, 54.2, 38.3), (16.5, 58, 40)] if deflect else [(10.0, 30, 10), (16.5, 35, 12)])
    sc.add('B-09', 'B', 5, [(10.0, 80, 10), (16.5, 80, 15)])
    mid = (53.75, 30.0)

    def b(t):
        if t < 12.0:
            return feet(sc, 'A-01', t)
        start, end = feet(sc, 'A-01', 12.0), feet(sc, receiver, 13.5, ahead=-.3)
        if deflect and t >= 12.75:
            pick = feet(sc, 'A-03', 13.75, ahead=-.3)
            return flight(mid, pick, 12.75, 13.75, t) if t < 13.75 else feet(sc, 'A-03', t, ahead=-.3)
        if t < 13.5:
            return flight(start, mid if deflect else end, 12.0, 12.75 if deflect else 13.5, t)
        return feet(sc, receiver, t, ahead=-.3)
    sc.ball = b
    return sc.build()


class Possession(unittest.TestCase):
    def test_pass_between_teammates_keeps_possession(self):
        rec = possession.compute(passing_scene('A'))
        self.assertTrue(rec['available'], rec['reason'])
        self.assertEqual(rec['changes'], [])
        self.assertEqual(team_at(rec, 11.0), 'A')
        self.assertEqual(team_at(rec, 12.8), 'A')            # the pass in flight is still A's
        self.assertEqual(run_of(rec, 12.8)['state'], 'flight')
        self.assertEqual(team_at(rec, 15.0), 'A')
        self.assertEqual(run_of(rec, 11.0)['carrier'], 'A-01')
        self.assertEqual(run_of(rec, 15.0)['carrier'], 'A-02')
        self.assertGreater(rec['shares']['A'], 5.0)
        self.assertEqual(rec['shares']['B'], 0.0)

    def test_interception_changes_possession_once(self):
        rec = possession.compute(passing_scene('B'))
        self.assertEqual(len(rec['changes']), 1, rec['changes'])
        c = rec['changes'][0]
        self.assertEqual((c['fromTeam'], c['toTeam'], c['lostBy'], c['wonBy']), ('A', 'B', 'A-01', 'B-05'))
        self.assertAlmostEqual(c['wonAt'], 13.5, delta=.3)
        self.assertAlmostEqual(c['lostAt'], 12.0, delta=.3)
        self.assertEqual(c['cause'], 'interception')
        self.assertEqual(team_at(rec, 12.8), 'A')            # the convention: the loser keeps it until the winner's control
        self.assertEqual(run_of(rec, 12.8)['state'], 'loose')
        self.assertEqual(team_at(rec, 15.0), 'B')
        self.assertEqual(c['frame'], round(c['wonAt']*FPS))

    def test_deflection_is_not_a_change(self):
        result = passing_scene('A', deflect=True)
        F = possession.features(result)
        D = possession.decide(F)
        touched = (D['e'][1] > 0) & (abs(F['t']-12.75) < .15)
        self.assertTrue(touched.any())                         # B-07's touch is seen ...
        rec = possession.compute(result)
        self.assertEqual(rec['changes'], [])                   # ... and is not a change: A-03 picks the ball up
        self.assertEqual(team_at(rec, 13.2), 'A')
        self.assertEqual(run_of(rec, 13.2)['state'], 'loose')
        self.assertEqual(run_of(rec, 15.0)['carrier'], 'A-03')

    def test_half_second_duel_flicker_is_not_a_change(self):
        sc = Scene(6.0)
        sc.add('A-01', 'A', 1, [(10.0, 40, 30), (16.0, 52, 30)])
        sc.add('B-04', 'B', 2, [(10.0, 47, 33), (12.5, 45.3, 31.3), (13.0, 46.3, 31.3), (16.0, 46.3, 36)])
        sc.add('A-02', 'A', 3, [(10.0, 70, 20), (16.0, 72, 20)])

        def b(t):
            if 12.5 <= t < 13.0:     # half a second at B-04's foot, A-01 a metre from it (a challenge)
                x, y = sc.at('B-04', t)
                return x, y-.3
            return feet(sc, 'A-01', t)
        sc.ball = b
        result = sc.build()
        F = possession.features(result)
        D = possession.decide(F)
        self.assertGreater(D['e'][1][(F['t'] >= 12.5) & (F['t'] < 13.0)].sum(), .05)   # B-04 had the ball for a moment
        rec = possession.compute(result)
        self.assertEqual(rec['changes'], [])
        self.assertIn(team_at(rec, 12.8), ('A', None))        # never B; unsure during the challenge is allowed
        self.assertEqual(team_at(rec, 14.0), 'A')

    def test_ball_out_dead_then_restart_by_the_other_team(self):
        sc = Scene(14.0)
        sc.add('A-01', 'A', 1, [(10.0, 50, 60), (12.0, 55, 64), (24.0, 56, 64)])
        sc.add('B-03', 'B', 2, [(10.0, 70, 50), (17.0, 60, 69.2), (24.0, 60, 69.2)])
        sc.add('B-04', 'B', 3, [(10.0, 70, 40), (24.0, 66, 55)])

        def b(t):
            if t < 12.0:
                return feet(sc, 'A-01', t)
            if t < 13.5:                                        # kicked over the near touchline, rolling on beyond it
                return flight(feet(sc, 'A-01', 12.0), (58.0, 72.0), 12.0, 13.5, t)
            if t < 18.0:
                return None                                     # out of view
            if t < 19.5:
                return 60.0, 69.0                               # held for the throw-in by B-03
            return flight((60.0, 69.0), (66.0, 55.3), 19.5, 21.0, t) if t < 21.0 else feet(sc, 'B-04', t, ahead=-.3)
        sc.ball = b
        rec = possession.compute(sc.build())
        self.assertEqual(len(rec['deadSpells']), 1, rec['deadSpells'])
        spell = rec['deadSpells'][0]
        self.assertEqual((spell['reason'], spell['restartTeam']), ('out_of_play', 'B'))
        self.assertLess(spell['t0'], 13.0)
        self.assertAlmostEqual(spell['t1'], 19.5, delta=.3)
        self.assertEqual(run_of(rec, 16.0)['state'], 'dead')
        self.assertIsNone(team_at(rec, 16.0))
        self.assertEqual(len(rec['changes']), 1)
        c = rec['changes'][0]
        self.assertEqual((c['fromTeam'], c['toTeam'], c['cause'], c['lostBy']), ('A', 'B', 'out_of_play', 'A-01'))
        self.assertAlmostEqual(c['wonAt'], 19.5, delta=.3)
        self.assertEqual(team_at(rec, 20.0), 'B')
        self.assertGreater(rec['shares']['dead'], 5.0)

    def test_missing_ball_is_unknown(self):
        sc = Scene(8.0)
        sc.add('A-01', 'A', 1, [(10.0, 40, 30), (18.0, 50, 30)])
        sc.add('B-02', 'B', 2, [(10.0, 60, 40), (18.0, 52, 34)])

        def b(t):
            if t < 12.0:
                return feet(sc, 'A-01', t)
            if t < 15.0:
                return None
            return feet(sc, 'B-02', t)
        sc.ball = b
        rec = possession.compute(sc.build())
        r = run_of(rec, 13.5)
        self.assertEqual((r['team'], r['state']), (None, 'unknown'))
        self.assertGreater(rec['shares']['unknown'], 2.5)
        self.assertEqual(team_at(rec, 11.0), 'A')
        self.assertEqual(team_at(rec, 16.0), 'B')
        self.assertTrue(any('out of view' in note for note in rec['quality']['notes']))

    def test_change_ids_are_stable(self):
        result = passing_scene('B')
        a = possession.compute(result)
        b = possession.compute(copy.deepcopy(result))
        self.assertEqual([c['id'] for c in a['changes']], [c['id'] for c in b['changes']])
        c = a['changes'][0]
        self.assertEqual(c['id'], f"pc-{c['frame']}-B")
        longer = copy.deepcopy(result)                         # more frames before it: the same change, the same id
        extra = passing_scene('B')['frames'][:30]
        for k, f in enumerate(extra):
            f['frame'], f['time'] = 270+k, round((270+k)/FPS, 6)
        longer['frames'] = extra+longer['frames']
        self.assertIn(c['id'], [x['id'] for x in possession.compute(longer)['changes']])

    def test_correcting_a_players_team_changes_the_carriers_team(self):
        result = passing_scene('A')
        self.assertEqual(possession.compute(result)['changes'], [])
        fix = {'id': 'c1', 'run': 'r', 'time': 13.6, 'target': {'id': 'A-02', 'track': 2, 'trackAt': 2},
               'before': {'role': 'player', 'team': 'A'}, 'after': {'role': 'player', 'team': 'B', 'id': 'B-11', 'into': None}}
        corrected = corrections.apply_result(copy.deepcopy(result), [fix], block=False)
        rec = possession.compute(corrected)
        self.assertEqual(len(rec['changes']), 1, rec['changes'])
        c = rec['changes'][0]
        self.assertEqual((c['fromTeam'], c['toTeam'], c['wonBy']), ('A', 'B', 'B-11'))
        self.assertEqual(run_of(rec, 15.0)['carrier'], 'B-11')
        self.assertEqual(team_at(rec, 15.0), 'B')

    def test_record_is_the_contract_and_plain_json(self):
        rec = possession.compute(passing_scene('B'))
        json.dumps(rec, allow_nan=False)
        self.assertEqual(set(rec), {'version', 'fps', 't0', 't1', 'available', 'reason', 'quality', 'shares', 'timeline', 'changes', 'deadSpells'})
        self.assertEqual(set(rec['quality']) >= {'level', 'coverage', 'notes'}, True)
        self.assertEqual(set(rec['shares']), {'A', 'B', 'dead', 'unknown'})
        self.assertAlmostEqual(sum(rec['shares'].values()), len(passing_scene('B')['frames'])/FPS, delta=.05)
        for r in rec['timeline']:
            self.assertEqual(set(r), {'t0', 't1', 'f0', 'f1', 'team', 'state', 'carrier', 'estimated', 'confidence'})
            self.assertIn(r['state'], possession.STATE_NAMES)
            self.assertIn(r['confidence'], possession.LEVELS)
        for a, b in zip(rec['timeline'], rec['timeline'][1:]):
            self.assertEqual(b['f0'], a['f1']+1)
        for c in rec['changes']:
            self.assertEqual(set(c), {'id', 'lostAt', 'wonAt', 'frame', 'fromTeam', 'toTeam', 'lostBy', 'wonBy', 'cause', 'confidence'})
            self.assertIn(c['cause'], ('tackle_or_duel', 'interception', 'out_of_play', 'unclear'))
            self.assertIn(c['confidence'], possession.CHANGE_LEVELS)

    def test_near_track_of_a_first_pass_that_followed_another_ball_is_not_evidence(self):
        # A-01 passes to A-02; the pass is bridged (not detected in flight) and the first pass, which followed something
        # else near B-05 meanwhile, names B-05 as the player hiding the ball: as on the broadcast, where the nearTrack of a
        # bridged frame is more than 2 m from the final ball on half of the frames
        result = passing_scene('A')
        far = None
        for f in result['frames']:
            if 12.3 <= f['time'] < 13.2:
                b = f['ball']
                b.update(state='MISSING', observed=False, origin='INTERPOLATION', motion='OCCLUDED', bridged=True, nearTrack=5)
                b5 = next(p for p in f['people'] if p['id'] == 'B-09')
                far = [b5['box'][0]+b5['box'][2]/2, b5['box'][1]+b5['box'][3]]
                b['firstPass'] = {'state': 'MISSING', 'phase': 'OCCLUDED', 'track': 1, 'center': far, 'motion': 'OCCLUDED'}
        window = None
        for settings, expect in (({}, False), ({'nt_agree': 1e9}, True)):
            F = possession.features(result)
            D = possession.decide(F, settings)
            window = (F['t'] >= 12.3) & (F['t'] < 13.2)
            self.assertEqual(bool((D['e'][1][window] > 0).any()), expect, settings)
        for f in result['frames']:   # the first pass on this ball (its centre where the final path put it): trusted
            if 12.3 <= f['time'] < 13.2:
                f['ball']['firstPass']['center'] = list(f['ball']['center'])
        D = possession.decide(possession.features(result))
        self.assertTrue((D['e'][1][window] > 0).any())
        self.assertEqual(possession.compute(passing_scene('A'))['changes'], [])

    def test_a_person_without_a_team_at_the_ball_is_evidence_for_nobody(self):
        # A-01 plays the ball to a player whose team is not decided yet (an unresolved track, '?'), who keeps it for 1.5 s
        # with B-04 a metre away and then passes to A-02: B-04 never had the ball
        sc = Scene(7.0)
        sc.add('A-01', 'A', 1, [(10.0, 40, 30), (17.0, 42, 30)])
        sc.add('U', 'A', 9, [(10.0, 46, 30), (17.0, 46, 30)])
        sc.add('B-04', 'B', 2, [(10.0, 46.0, 31.0), (17.0, 46.0, 31.0)])
        sc.add('A-02', 'A', 3, [(10.0, 56, 30), (17.0, 60, 30)])

        def b(t):
            if t < 11.5:
                return feet(sc, 'A-01', t)
            if t < 12.0:
                return flight(feet(sc, 'A-01', 11.5), feet(sc, 'U', 12.0, ahead=-.3), 11.5, 12.0, t)
            if t < 13.5:
                return feet(sc, 'U', t, ahead=-.3)
            if t < 14.3:
                return flight(feet(sc, 'U', 13.5, ahead=-.3), feet(sc, 'A-02', 14.3, ahead=-.3), 13.5, 14.3, t)
            return feet(sc, 'A-02', t, ahead=-.3)
        sc.ball = b
        result = sc.build()
        for f in result['frames']:
            for p in f['people']:
                if p['id'] == 'U':
                    p.update(id=None, display='?', team=None, role='UNKNOWN', label='IDENTITY_UNCERTAIN', state='uncertain')
        result['match']['players'] = [e for e in result['match']['players'] if e['id'] != 'U']
        rec = possession.compute(result)
        self.assertEqual(rec['changes'], [])
        self.assertEqual(team_at(rec, 15.0), 'A')
        self.assertNotEqual(team_at(rec, 13.0), 'B')
        off = possession.compute(result, {'unlabelled': False})   # without the rule B-04 "wins" the ball and loses it again
        self.assertEqual([c['toTeam'] for c in off['changes']], ['B', 'A'])

    def test_a_goalkeeper_without_a_team_plays_for_the_team_defending_his_goal(self):
        # B-09 runs at the left goal (team A defends it: goalEnds) and loses the ball to a goalkeeper the tracking has not
        # given a team
        sc = Scene(6.0)
        sc.add('B-09', 'B', 1, [(10.0, 25, 34), (12.0, 14, 34), (16.0, 14, 36)])
        sc.add('GK-3', 'A', 2, [(10.0, 4, 34), (16.0, 4, 34)])
        sc.add('A-05', 'A', 3, [(10.0, 30, 20), (16.0, 30, 20)])

        def b(t):
            if t < 12.0:
                return feet(sc, 'B-09', t)
            if t < 13.0:
                return flight(feet(sc, 'B-09', 12.0), feet(sc, 'GK-3', 13.0, ahead=.3), 12.0, 13.0, t)
            return feet(sc, 'GK-3', t, ahead=.3)
        sc.ball = b
        result = sc.build()
        for f in result['frames']:
            for p in f['people']:
                if p['id'] == 'GK-3':
                    p.update(team=None, label='GOALKEEPER')
        result['match']['goalkeepers'] = []
        rec = possession.compute(result)
        self.assertEqual([(c['fromTeam'], c['toTeam'], c['wonBy']) for c in rec['changes']], [('B', 'A', 'GK-3')])
        self.assertEqual(team_at(rec, 15.0), 'A')
        result['tactical']['goalEnds'] = {'left': None, 'right': None}   # no goal ends: he plays for nobody
        self.assertEqual(possession.compute(result)['changes'], [])

    def test_a_camera_cut_restarts_possession(self):
        # A-01 has the ball, the camera cuts (a new segment) and the next view shows B-02 with it: nobody saw the ball
        # change hands, so the change is uncertain and the cut frame is unknown; with the same team after the cut nothing
        # changes, and (possession-4, cut_fresh) a team is named again only from the first observed control after the cut
        def scene(team):
            sc = Scene(6.0)
            sc.add('A-01', 'A', 1, [(10.0, 40, 30), (16.0, 46, 30)])
            sc.add('B-02' if team == 'B' else 'A-02', team, 2, [(10.0, 70, 40), (16.0, 76, 40)])
            other = 'B-02' if team == 'B' else 'A-02'
            sc.ball = lambda t: feet(sc, 'A-01', t) if t < 13.0 else feet(sc, other, t)
            result = sc.build()
            for f in result['frames']:
                f['segment'] = 0 if f['time'] < 13.0 else 1
            return result
        rec = possession.compute(scene('B'))
        self.assertEqual([(c['fromTeam'], c['toTeam'], c['confidence'], c['cause']) for c in rec['changes']], [('A', 'B', 'uncertain', 'unclear')])
        self.assertEqual(run_of(rec, 13.0)['state'], 'unknown')
        self.assertEqual(team_at(rec, 14.0), 'B')
        same = possession.compute(scene('A'))
        self.assertEqual(same['changes'], [])
        self.assertIsNone(team_at(same, 13.0))                   # the cut frame: no team until one is seen in control
        self.assertEqual(team_at(same, 13.5), 'A')
        self.assertEqual(team_at(possession.compute(scene('A'), {'cut_fresh': False}), 13.0), 'A')   # possession-3 carried it

    def test_the_carrier_is_near_on_both_metrics(self):
        # A-03 stands a metre from the ball on the pitch but his box is elsewhere in the image (a registration error on
        # his position); A-01's feet are at the ball in the image and 1.2 m from it on the pitch: A-01 has the ball
        result = passing_scene('A')
        for f in result['frames']:
            if 10.5 <= f['time'] < 11.5:
                bx, by = f['ball']['pitch']
                for p in f['people']:
                    if p['id'] == 'A-03':
                        p['pitch'] = [bx+1.0/105, by]
                    if p['id'] == 'A-01':
                        p['pitch'] = [bx-1.2/105, by]
        rec = possession.compute(result)
        self.assertEqual(run_of(rec, 11.0)['carrier'], 'A-01')
        self.assertEqual(rec['changes'], [])

    def test_small_scene_run_abstains(self):
        result = passing_scene('B')
        for f in result['frames']:
            f['profile'] = {'smallScene': True}
        rec = possession.compute(result)
        self.assertFalse(rec['available'])
        self.assertIn('small', rec['reason'])
        self.assertEqual((rec['changes'], rec['quality']['level']), ([], 'unavailable'))
        self.assertEqual([r['state'] for r in rec['timeline']], ['unknown'])
        json.dumps(rec, allow_nan=False)

    def test_empty_result(self):
        rec = possession.compute({'frames': []})
        self.assertFalse(rec['available'])
        json.dumps(rec, allow_nan=False)


def hide(result, t0, t1, track=None):
    """The ball interpolated (MISSING, INTERPOLATION) over [t0, t1), at the feet of the player with this track."""
    for f in result['frames']:
        if t0 <= f['time'] < t1:
            f['ball'].update(state='MISSING', observed=False, origin='INTERPOLATION', motion='OCCLUDED', bridged=True)
            if track is not None:
                f['ball']['nearTrack'] = track
    return result


def challenge_scene():
    """A-01 dribbles to 12.0 s and passes to B-05 (an interception), who holds the ball 12.6-13.4 s and passes it on to
    A-02, who keeps it from 14.1 s: a short possession of team B."""
    sc = Scene(7.0)
    sc.add('A-01', 'A', 1, [(10.0, 40, 30), (12.0, 46, 30), (17.0, 47, 32)])
    sc.add('B-05', 'B', 2, [(10.0, 52, 30), (17.0, 52, 30)])
    sc.add('A-02', 'A', 3, [(10.0, 60, 36), (17.0, 66, 36)])
    sc.add('B-09', 'B', 5, [(10.0, 80, 10), (17.0, 80, 15)])

    def b(t):
        if t < 12.0:
            return feet(sc, 'A-01', t)
        if t < 12.6:
            return flight(feet(sc, 'A-01', 12.0), feet(sc, 'B-05', 12.6, ahead=-.3), 12.0, 12.6, t)
        if t < 13.4:
            return feet(sc, 'B-05', t, ahead=-.3)
        if t < 14.1:
            return flight(feet(sc, 'B-05', 13.4, ahead=-.3), feet(sc, 'A-02', 14.1, ahead=-.3), 13.4, 14.1, t)
        return feet(sc, 'A-02', t, ahead=-.3)
    sc.ball = b
    return sc.build()


class PossessionV3(unittest.TestCase):
    """The v3 rules (possession-3): a change rests on an observed control, an excursion beyond a line is out of play only
    if consistent, a stoppage with the ball in view, who won the ball, the confidence of a back-and-forth pair."""

    def test_a_change_rests_on_an_observed_control(self):
        # B-05's whole spell at the ball is interpolated (hidden at his feet, never seen there, 1.15 s): nothing starts
        # a change, A keeps the ball; one observed frame of his spell confirms it from its first frame
        hidden = hide(challenge_scene(), 12.35, 13.5, track=2)
        rec = possession.compute(hidden)
        self.assertEqual(rec['changes'], [])
        v2 = possession.compute(hidden, {'start_on': 'any'})       # the v2 rule: the interpolated spell is a change and back
        self.assertEqual([c['toTeam'] for c in v2['changes']], ['B', 'A'])
        seen = hide(hide(challenge_scene(), 12.35, 13.0, track=2), 13.1, 13.5, track=2)
        rec = possession.compute(seen)
        self.assertEqual([(c['fromTeam'], c['toTeam']) for c in rec['changes']], [('A', 'B'), ('B', 'A')])
        self.assertAlmostEqual(rec['changes'][0]['wonAt'], 12.5, delta=.15)   # from the start of B-05's spell
        self.assertEqual(rec['changes'][0]['wonBy'], 'B-05')

    def test_an_image_only_spell_starts_no_change(self):
        # a registration hole over B-05's spell (no pitch position for the ball or anyone): image-only evidence may keep a
        # possession but never start one
        result = challenge_scene()
        for f in result['frames']:
            if 12.5 <= f['time'] < 13.6:
                f['ball']['pitchConfidence'] = .05
                for p in f['people']:
                    p['pitchConfidence'] = 0.0
        self.assertEqual(possession.compute(result)['changes'], [])
        self.assertEqual([c['toTeam'] for c in possession.compute(result, {'start_on': 'any'})['changes']], ['B', 'A'])

    def excursion(self, jump):
        """A circulates the ball; at 12.0 s the ball path goes to an object 1.5 m beyond the near touchline and stays on it
        3 s, then back at A-02's feet. With jump, it gets there from the middle of the pitch in one frame (the path on a
        ball boy's ball); without, it is kicked there across the line at 27 m/s."""
        sc = Scene(8.0)
        sc.add('A-01', 'A', 1, [(10.0, 40, 30), (18.0, 48, 30)])
        sc.add('A-02', 'A', 2, [(10.0, 50, 36), (18.0, 56, 36)])
        sc.add('B-05', 'B', 3, [(10.0, 60, 20), (18.0, 60, 25)])
        far = (52.0, 69.5)

        def b(t):
            if t < 10.5:
                return feet(sc, 'A-01', t)
            if t < 12.0:
                start = feet(sc, 'A-01', 10.5)
                return flight(start, far, 10.5, 12.0, t) if not jump else flight(start, feet(sc, 'A-02', 12.0, ahead=-.3), 10.5, 12.0, t)
            if t < 15.0:
                return far
            return feet(sc, 'A-02', t, ahead=-.3)
        sc.ball = b
        return sc.build()

    def test_an_excursion_reached_by_a_jump_is_no_ball_out_of_play(self):
        rec = possession.compute(self.excursion(jump=True))
        self.assertEqual(rec['deadSpells'], [])
        self.assertEqual(rec['changes'], [])
        self.assertEqual(team_at(rec, 13.5), 'A')       # carried over the stretch with no ball: A on both sides
        self.assertTrue(run_of(rec, 13.5)['estimated'])
        self.assertEqual(run_of(rec, 13.5)['confidence'], 'low')
        v2 = possession.compute(self.excursion(jump=True), {'out_consistent': False})
        self.assertEqual([s['reason'] for s in v2['deadSpells']], ['out_of_play'])
        rolled = possession.compute(self.excursion(jump=False))   # a ball that rolled out across the line is out of play
        self.assertEqual([s['reason'] for s in rolled['deadSpells']], ['out_of_play'])

    def test_an_excursion_with_another_ball_in_play_is_no_ball_out_of_play(self):
        result = self.excursion(jump=False)
        for f in result['frames']:
            if 12.0 <= f['time'] < 15.0:      # the ball tracker saw another ball moving on the pitch at A-01's feet
                u, v = image(*feet_at(result, f, 'A-01'))
                f['ballCandidates'] = [{'status': 'rejected', 'box': [(u-3.5)/W, (v-7)/H, 7/W, 7/H],
                                        'scores': {'field': 1.0, 'static': 0.0, 'speed': 2.5}}]
        rec = possession.compute(result)
        self.assertEqual(rec['deadSpells'], [])

    def stoppage(self, walk):
        """Eight players run at 4 m/s; at 12.0 s A-01 is fouled: the ball rolls on untouched at 1 m/s far from everyone
        while the players walk at walk m/s, until B-01 restarts at 17.0 s."""
        sc = Scene(9.0)
        lanes = [('A-01', 'A', 30), ('A-02', 'A', 20), ('A-03', 'A', 40), ('A-04', 'A', 50),
                 ('B-01', 'B', 34), ('B-02', 'B', 24), ('B-03', 'B', 44), ('B-04', 'B', 54)]
        for k, (pid, team, y) in enumerate(lanes):
            x0 = 30.0+(k % 4)
            x1 = x0+4.0*2.0
            sc.add(pid, team, k+1, [(10.0, x0, y), (12.0, x1, y), (17.0, x1+walk*5.0, y), (19.0, x1+walk*5.0+8.0, y)])
        start = (42.0, 30.0)

        def b(t):
            if t < 12.0:
                return feet(sc, 'A-01', t)
            if t < 17.0:
                return start[0]+(t-12.0)*1.0, start[1]+6.0+(t-12.0)*.2
            return feet(sc, 'B-01', t)
        sc.ball = b
        result = sc.build()
        for f in result['frames']:    # B-01 walks to the ball and takes the free kick at 17.0 s
            if 16.5 <= f['time'] < 17.0:
                bx, by = f['ball']['pitch'][0]*105, f['ball']['pitch'][1]*68
                p = next(q for q in f['people'] if q['id'] == 'B-01')
                p['pitch'] = [(bx-.3)/105, by/68]
                u, v = image(bx-.3, by)
                p['box'] = [(u-.35*PX)/W, (v-1.8*PX)/H, .7*PX/W, 1.8*PX/H]
        return result

    def test_a_stoppage_with_the_ball_in_view(self):
        rec = possession.compute(self.stoppage(walk=.5))
        spells = [s for s in rec['deadSpells'] if s['reason'] == 'stoppage']
        self.assertEqual(len(spells), 1, rec['deadSpells'])
        self.assertLess(spells[0]['t0'], 13.5)
        self.assertGreater(spells[0]['t1'], 15.5)
        self.assertEqual(run_of(rec, 15.0)['state'], 'dead')
        live = possession.compute(self.stoppage(walk=4.0))     # the same ball while everyone keeps running: play goes on
        self.assertEqual([s for s in live['deadSpells'] if s['reason'] == 'stoppage'], [])
        self.assertNotEqual(run_of(live, 15.0)['state'], 'dead')

    def test_won_by_is_the_player_who_then_holds_the_ball(self):
        # B-02 pokes the ball off A-01 (at his foot for two frames) and B-03, 1.6 m away, collects it and keeps it
        sc = Scene(6.0)
        sc.add('A-01', 'A', 1, [(10.0, 40, 30), (12.0, 46, 30), (16.0, 44, 30)])
        sc.add('B-02', 'B', 2, [(10.0, 47, 32), (16.0, 47, 32)])
        sc.add('B-03', 'B', 3, [(10.0, 48.3, 33.6), (12.5, 48.3, 33.6), (16.0, 56, 36)])

        def b(t):
            if t < 12.0:
                return feet(sc, 'A-01', t)
            if t < 12.07:
                return 47.0, 32.3
            if t < 12.25:
                return flight((47.0, 32.3), feet(sc, 'B-03', 12.25, ahead=-.3), 12.07, 12.25, t)
            return feet(sc, 'B-03', t, ahead=-.3)
        sc.ball = b
        result = sc.build()
        rec = possession.compute(result)
        self.assertEqual([(c['toTeam'], c['wonBy']) for c in rec['changes']], [('B', 'B-03')])
        first = possession.compute(result, {'won_by': 'first'})
        self.assertEqual([c['wonBy'] for c in first['changes']], ['B-02'])

    def test_a_back_and_forth_pair_is_low_confidence(self):
        rec = possession.compute(challenge_scene(), {'conf_rule': 'v3'})   # B held it 0.9 s and lost it again: a pair
        self.assertEqual([(c['toTeam'], c['confidence']) for c in rec['changes']], [('B', 'low'), ('A', 'low')])
        rec = possession.compute(challenge_scene())      # possession-4: medium and low are one level, 'uncertain'
        self.assertEqual([(c['toTeam'], c['confidence']) for c in rec['changes']], [('B', 'uncertain'), ('A', 'uncertain')])
        rec = possession.compute(passing_scene('B'))     # B keeps it: an interception with evidence and a hold
        self.assertEqual([(c['toTeam'], c['confidence']) for c in rec['changes']], [('B', 'high')])
        v2 = possession.compute(challenge_scene(), {'conf_rule': 'v2'})
        self.assertNotEqual([c['confidence'] for c in v2['changes']], ['low', 'low'])

    def test_player_speeds_follow_the_camera(self):
        # a player standing still while the camera pans 12 px a frame, no pitch position: the image fallback carries his
        # feet through the camera motion and finds him still; without the motion rows there is no image speed
        n, fps = 60, 30.0
        rows = [(i, 0, 0, float('nan'), float('nan'), 500.0+12.0*i, 600.0, 90.0) for i in range(n)]
        cam = np.tile(np.eye(3), (n, 1, 1))
        for i in range(1, n):
            M = np.eye(3)
            M[0, 2] = 12.0
            cam[i] = M@cam[i-1]
        spd = possession.player_speeds(n, fps, rows, 1, cam, np.zeros(n, np.int64), 1.0)
        self.assertLess(float(np.nanmax(spd)), .05)
        self.assertGreater(int(np.isfinite(spd).sum()), 20)
        none = possession.player_speeds(n, fps, rows, 1, np.tile(np.eye(3), (n, 1, 1)), np.arange(n), 1.0)
        self.assertFalse(np.isfinite(none).any())


def feet_at(result, f, pid):
    p = next(q for q in f['people'] if q['id'] == pid)
    return p['pitch'][0]*105+.3, p['pitch'][1]*68



def walkers(sc, speed, first_track=20, start=10.0, end=30.0, y0=8.0):
    """Six players of each team walking (or jogging) along a touchline at speed m/s from start to end, far from the
    ball: the players' activity of the slowdown rule (at least six team players)."""
    for k in range(12):
        team = 'A' if k < 6 else 'B'
        x0 = 20.0+6.0*(k % 6)
        sc.add(f'{team}-{30+k}', team, first_track+k, [(start, x0, y0+(k // 6)*3.0), (end, x0+speed*(end-start), y0+(k // 6)*3.0)])


class PossessionV4(unittest.TestCase):
    """The rules of possession-4: short results, a ball held in play, a keeper's catch, a knock-past dribble, results
    without ball.origin, the first pass in foreshortened metres, a placed ball after an unseen stretch, camera cuts and
    replays, an identity on another team's player, the merged 'uncertain' change confidence."""

    def hold(self, seconds, crowd):
        """A-01 receives a pass at 12.0 s and keeps the ball still at his feet for `seconds` (shielding it, B-04 a metre
        and a half away), then passes to A-02; twelve others walk at `crowd` m/s. Live play throughout."""
        sc = Scene(seconds+6.0)
        t1, end = 12.0+seconds, sc.n/FPS+10.0
        sc.add('A-01', 'A', 1, [(10.0, 50, 34), (end, 50, 34)])
        sc.add('A-02', 'A', 2, [(10.0, 62, 40), (end, 64, 40)])
        sc.add('A-03', 'A', 3, [(10.0, 40, 30), (end, 41, 30)])
        sc.add('B-04', 'B', 4, [(10.0, 51.8, 34.5), (end, 51.8, 34.5)])
        walkers(sc, crowd, end=end)

        def b(t):
            if t < 11.5:
                return feet(sc, 'A-03', t)
            if t < 12.0:
                return flight(feet(sc, 'A-03', 11.5), feet(sc, 'A-01', 12.0), 11.5, 12.0, t)
            if t < t1:
                return feet(sc, 'A-01', t)
            if t < t1+1.0:
                return flight(feet(sc, 'A-01', t1), feet(sc, 'A-02', t1+1.0, ahead=-.3), t1, t1+1.0, t)
            return feet(sc, 'A-02', t, ahead=-.3)
        sc.ball = b
        return sc.build()

    def test_short_results_do_not_crash(self):
        # possession-3 raised on any result shorter than the 1 s smoothing window (30 frames at 30 fps)
        full = self.hold(6.0, 1.0)
        for n in (1, 15, 30, 31):
            r = copy.deepcopy(full)
            r['frames'] = r['frames'][:n]
            rec = possession.compute(r)
            json.dumps(rec, allow_nan=False)
            self.assertTrue(rec['available'], n)
            self.assertEqual(rec['timeline'][0]['f0'], r['frames'][0]['frame'])
            self.assertEqual(rec['timeline'][-1]['f1'], r['frames'][-1]['frame'])
        rng = np.random.default_rng(1)
        for n, w in ((45, 31), (31, 31), (30, 30), (100, 30)):   # the centred sum is np.convolve 'same', bit for bit
            x = rng.random(n)
            self.assertTrue(np.array_equal(possession._centred_sum(x, w), np.convolve(x, np.ones(w), 'same')))
        self.assertEqual(possession._centred_sum(np.ones(15), 31).tolist(), [15.0]*15)

    def test_a_ball_held_in_play_is_no_stoppage(self):
        # a player keeps the ball still for 6 s while everyone walks: possession-3 called it a stoppage (the still-ball
        # rule); a player in control of a ball he received in play is live play
        result = self.hold(6.0, .5)
        rec = possession.compute(result)
        self.assertEqual(rec['deadSpells'], [])
        self.assertEqual({team_at(rec, t) for t in (13.0, 15.0, 17.5)}, {'A'})
        self.assertEqual(rec['changes'], [])
        v3 = possession.compute(result, {'live_hold': False})
        self.assertEqual([s['reason'] for s in v3['deadSpells']], ['stoppage'])

    def test_a_placed_ball_the_taker_walks_up_to_stays_dead(self):
        # the restart pattern: after a foul the ball lies still with nobody at it while everyone walks; B-01 walks up,
        # stands at it for 2 s and kicks: dead to the kick, the restart B's
        sc = Scene(10.0)
        sc.add('A-01', 'A', 1, [(10.0, 30, 34), (12.0, 38, 34), (20.0, 40, 34)])
        sc.add('B-01', 'B', 2, [(10.0, 48, 40), (14.5, 48, 40), (15.0, 44.8, 34.0), (17.0, 44.8, 34.0), (20.0, 47, 37)])
        sc.add('B-02', 'B', 3, [(10.0, 60, 44), (20.0, 64, 44)])
        walkers(sc, .5)

        def b(t):
            if t < 12.0:
                return feet(sc, 'A-01', t)
            if t < 12.5:
                return flight(feet(sc, 'A-01', 12.0), (45.0, 34.0), 12.0, 12.5, t)
            if t < 17.0:
                return 45.0, 34.0
            if t < 18.0:
                return flight((45.0, 34.0), feet(sc, 'B-02', 18.0, ahead=-.3), 17.0, 18.0, t)
            return feet(sc, 'B-02', t, ahead=-.3)
        sc.ball = b
        rec = possession.compute(sc.build())
        dead = [s for s in rec['deadSpells'] if s['reason'] == 'stoppage']
        self.assertEqual(len(dead), 1, rec['deadSpells'])
        self.assertLess(dead[0]['t0'], 14.0)
        self.assertGreater(dead[0]['t1'], 16.8)
        self.assertEqual(dead[0]['restartTeam'], 'B')
        self.assertEqual(run_of(rec, 16.0)['state'], 'dead')

    def keeper(self, hands=True):
        """B-09 shoots at 12.0 s; GK-A catches the ball at chest height at 12.6 s and walks out with it (2 m/s) while
        everyone walks, then rolls it to A-05 at 17.0 s. With hands False the ball is held at his feet."""
        sc = Scene(9.0)
        sc.add('B-09', 'B', 1, [(10.0, 30, 34), (12.0, 21, 34), (19.0, 24, 36)])
        sc.add('GK-A', 'A', 2, [(10.0, 3, 34), (12.6, 3, 34), (17.0, 11.8, 34), (19.0, 11.8, 34)])
        sc.add('A-05', 'A', 3, [(10.0, 30, 44), (19.0, 30, 44)])
        walkers(sc, .8, y0=50.0)
        start = (20.6, 34.0)

        def b(t):
            if t < 12.0:
                return feet(sc, 'B-09', t, ahead=-.4)
            if t < 12.6:
                return flight(start, feet(sc, 'GK-A', 12.6, ahead=.2), 12.0, 12.6, t)
            if t < 17.0:
                return feet(sc, 'GK-A', t, ahead=.2)
            if t < 18.0:
                return flight(feet(sc, 'GK-A', 17.0, ahead=.2), feet(sc, 'A-05', 18.0, ahead=-.3), 17.0, 18.0, t)
            return feet(sc, 'A-05', t, ahead=-.3)
        sc.ball = b
        result = sc.build()
        if hands:
            for f in result['frames']:
                h = 1.1*min(1.0, max(0.0, (f['time']-12.0)/.6)) if 12.0 <= f['time'] < 17.0 else 0.0
                if h > 0:   # drawn h metres up in the image, its pitch position (the box bottom projected) 2h farther
                    bl = f['ball']
                    bl['box'][1] -= PX*h/H
                    bl['center'][1] -= PX*h/H
                    bl['pitch'][1] -= 2.0*h/68.0
                    bl.pop('nearTrack', None)
        return result

    def test_a_keeper_holding_the_ball_in_his_hands_has_it_from_the_catch(self):
        rec = possession.compute(self.keeper())
        self.assertEqual([(c['fromTeam'], c['toTeam'], c['wonBy']) for c in rec['changes']], [('B', 'A', 'GK-A')])
        self.assertAlmostEqual(rec['changes'][0]['wonAt'], 12.6, delta=.2)      # dated at the catch, not the release
        self.assertEqual(rec['deadSpells'], [])
        self.assertEqual({team_at(rec, t) for t in (13.5, 15.0, 16.5)}, {'A'})
        v3 = possession.compute(self.keeper(), {'keeper_hands': False})   # possession-3: nobody at a ball in the air
        self.assertTrue(all(c['wonAt'] > 16.5 for c in v3['changes'] if c['toTeam'] == 'A'), v3['changes'])
        self.assertNotEqual({team_at(v3, t) for t in (13.5, 15.0, 16.5)}, {'A'})

    def test_a_ball_knocked_past_a_defender_is_not_his(self):
        # A-01 runs at 5 m/s knocking the ball up to 2 m ahead (it eases off between his touches, every .8 s) past B-05,
        # who stands half a metre beside its line: possession-3 gave B-05 a change and a change back
        sc = Scene(6.0)
        sc.add('A-01', 'A', 1, [(10.0, 30, 34), (16.0, 60, 34)])
        sc.add('B-05', 'B', 2, [(10.0, 42, 34.5), (16.0, 42, 34.5)])
        sc.add('A-02', 'A', 3, [(10.0, 70, 20), (16.0, 72, 20)])

        def b(t):
            ph = ((t-10.0) % .8)/.8
            x, y = sc.at('A-01', t)
            return x+.3+1.7*math.sin(math.pi*ph), y
        sc.ball = b
        result = sc.build()
        self.assertEqual(possession.compute(result)['changes'], [])
        self.assertEqual([c['toTeam'] for c in possession.compute(result, {'knock_past': False})['changes']], ['B', 'A'])

    def test_a_result_without_ball_origin_still_has_changes(self):
        # every run made before 2026-10-15 lacks ball.origin: possession-3 started no change there, silently
        result = passing_scene('B')
        for f in result['frames']:
            f['ball'].pop('origin', None)
        rec = possession.compute(result)
        self.assertEqual([(c['fromTeam'], c['toTeam']) for c in rec['changes']], [('A', 'B')])
        self.assertTrue(any('ball.origin' in n for n in rec['quality']['notes']))
        self.assertEqual(possession.compute(result, {'origin_fallback': False})['changes'], [])

    def test_the_first_pass_check_doubles_the_image_rows(self):
        # the path kicked onto an object 1.5 m beyond the near touchline while the first pass followed the real ball 3 m
        # deeper (3 m in depth is 1.5 m of image rows: under nt_agree undoubled, as possession-3 measured it)
        result = PossessionV3.excursion(self, jump=False)
        for f in result['frames']:
            f['people'].append(person('A-04', 'A', 30.0, 60.0, 9))   # near the touchline's image rows: their scale
            if 12.0 <= f['time'] < 15.0:
                c = f['ball']['center']
                f['ball']['firstPass'] = {'state': 'TRACKED', 'center': [c[0], c[1]-3.0*PY/H]}
        self.assertEqual(possession.compute(result)['deadSpells'], [])
        self.assertEqual([s['reason'] for s in possession.compute(result, {'fp_rows': 1.0})['deadSpells']], ['out_of_play'])

    def test_a_ball_placed_after_an_unseen_stretch_is_dead(self):
        # the clip starts with the ball out of view for 3 s, then placed still for 3 s with the keeper beside it while
        # everyone walks, then kicked long: possession-3 compared the placed ball with the unseen frame a second before
        # and found no stoppage at all
        sc = Scene(10.0)
        sc.add('GK-A', 'A', 1, [(10.0, 7, 30), (13.0, 5.5, 31), (15.7, 5.5, 31), (16.0, 6.2, 31), (20.0, 4, 34)])
        sc.add('A-04', 'A', 2, [(10.0, 40, 20), (18.0, 42, 22), (20.0, 50, 26)])
        sc.add('B-09', 'B', 3, [(10.0, 25, 34), (20.0, 30, 30)])
        walkers(sc, 1.0, y0=50.0)

        def b(t):
            if t < 13.0:
                return None
            if t < 16.0:
                return 6.0, 31.5
            if t < 18.0:
                return flight((6.0, 31.5), feet(sc, 'A-04', 18.0, ahead=-.3), 16.0, 18.0, t)
            return feet(sc, 'A-04', t, ahead=-.3)
        sc.ball = b
        result = sc.build()
        dead = possession.compute(result)['deadSpells']
        self.assertEqual(len(dead), 1, dead)
        self.assertLess(dead[0]['t0'], 13.5)
        self.assertGreater(dead[0]['t1'], 15.0)
        self.assertEqual(possession.compute(result, {'still_since': False})['deadSpells'], [])

    def test_a_replay_between_two_cuts_is_unknown(self):
        # A-01 carries the ball; at 14.0 s the broadcast cuts to a replay of B-05 elsewhere and at 18.0 s back to live
        # play, A-02 on the ball: no change, and the replay names nobody
        sc = Scene(10.0)
        sc.add('A-01', 'A', 1, [(10.0, 40, 30), (20.0, 50, 30)])
        sc.add('A-02', 'A', 2, [(10.0, 60, 36), (20.0, 64, 36)])
        sc.add('B-05', 'B', 3, [(10.0, 20, 50), (20.0, 30, 50)])
        sc.ball = lambda t: feet(sc, 'A-01', t) if t < 14.0 else feet(sc, 'B-05', t) if t < 18.0 else feet(sc, 'A-02', t)
        result = sc.build()
        for f in result['frames']:
            f['segment'] = 0 if f['time'] < 14.0 else 1 if f['time'] < 18.0 else 2
        rec = possession.compute(result)
        self.assertEqual(rec['changes'], [])
        self.assertIsNone(team_at(rec, 16.0))
        self.assertEqual((team_at(rec, 13.0), team_at(rec, 19.0)), ('A', 'A'))
        self.assertTrue(any('replay' in n for n in rec['quality']['notes']))
        self.assertEqual(team_at(possession.compute(result, {'replays': False, 'cut_fresh': False}), 16.0), 'B')

    def test_an_identity_on_another_teams_player_plays_for_its_kit(self):
        # A-07 is carried onto a B player's track at 13.0 s (the frames say team B, the roster keeps A-07 in team A):
        # that player wins the ball from A-01 and keeps it; a person the user corrected or identified keeps his team
        sc = Scene(7.0)
        sc.add('A-01', 'A', 1, [(10.0, 40, 30), (13.0, 46, 30), (17.0, 47, 32)])
        sc.add('A-07', 'A', 7, [(10.0, 50, 33), (13.0, 46.6, 30.6), (17.0, 54, 34)])
        sc.add('B-02', 'B', 2, [(10.0, 70, 20), (17.0, 72, 20)])
        sc.ball = lambda t: feet(sc, 'A-01', t) if t < 13.0 else feet(sc, 'A-07', t, ahead=.4)
        result = sc.build()
        for f in result['frames']:
            for p in f['people']:
                if p['id'] == 'A-07' and f['time'] >= 13.0:
                    p.update(team='B', label='PLAYER_TEAM_B')
        rec = possession.compute(result)
        self.assertEqual([(c['fromTeam'], c['toTeam'], c['wonBy']) for c in rec['changes']], [('A', 'B', 'A-07')])
        self.assertTrue(any("frame's team" in n for n in rec['quality']['notes']))
        self.assertEqual(possession.compute(result, {'kit_team': False})['changes'], [])
        for f in result['frames']:
            for p in f['people']:
                if p['id'] == 'A-07':
                    p['manual'] = True
        self.assertEqual(possession.compute(result)['changes'], [])

    def test_change_confidence_is_high_or_uncertain(self):
        for rec in (possession.compute(passing_scene('B')), possession.compute(challenge_scene())):
            self.assertTrue(all(c['confidence'] in possession.CHANGE_LEVELS for c in rec['changes']))
        self.assertEqual([c['confidence'] for c in possession.compute(passing_scene('B'))['changes']], ['high'])


if __name__ == '__main__':
    unittest.main()
