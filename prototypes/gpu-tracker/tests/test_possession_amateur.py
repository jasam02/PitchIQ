"""The image-based possession estimate (soccer/possession_amateur.py) on synthetic small-scene results, and the dispatch
that keeps possession-4's record byte for byte wherever possession-4 judges the footage.

The scene: a camera whose image maps the pitch linearly (PX px per metre across, PY down: a low amateur camera when PY
is a small fraction of PX), its calibration.H saying so, people as 1.8 m tall boxes on their feet, kit readings in
people[].why as the identity tracker writes them ('looks ref R A a B b'), every frame a small scene, 30 frames per
second. What is checked: the dispatch (a broadcast result gets possession-4's own record, byte-identical; a small scene
or a failed agreement gets the estimate; the 'engine' setting), the record's shape (possession-4's keys, version
possession-amateur-1, never 'high', JSON-safe), the depth-aware reach (a player 3 m behind the ball in depth is not at
it, though possession-4's image metric with its rows doubled would put him 0.75 m away), the trusted ball (a bridge
longer than 0.5 s, the first pass on another team's player), the carrier (a confirmed identity whose kit readings agree;
an identity on the other kit, an unconfirmed track: nobody's), the teamless goalkeeper from tactical.goalEnds, a
contested ball, the no-carry rule (a placed ball nobody is at for 1.5 s breaks possession; an unseen ball between two
spells of one team is carried), a change reported as the boundary of two named spans with low confidence, and a dead
ball only with a line near the ball in view.

Run:  python -m unittest test_possession_amateur          (from this directory)
"""
import copy
import json
import math
import unittest

import numpy as np

import scene  # noqa: F401  (adds the prototype directory to sys.path)
from soccer import possession, possession_amateur as amateur

FPS = 30.0
W, H = 4000, 1400
PX = 33.0
U0, V0 = 50.0, 120.0
FULL_VIEW = [[-.05, -.05], [1.05, -.05], [1.05, 1.05], [-.05, 1.05]]
MIDDLE_VIEW = [[.3, .25], [.7, .25], [.7, .75], [.3, .75]]


def kit_why(kit, team):
    a, b = (.6, .1) if kit == 'A' else (.1, .6)
    return f'looks ref 0.05 A {a:.2f} B {b:.2f} | shape 1.00 | PLAYER TEAM {team} 1.00'


class Scene:
    """Players on straight lines between keyframes, a ball per time, a camera with PY px per metre down the image."""

    def __init__(self, seconds, py=16.5, small=True, view=FULL_VIEW, ends=None, calibrated=True):
        self.n = int(round(seconds*FPS))
        self.py = py
        self.small = small
        self.view = view
        self.ends = {'left': 'A', 'right': 'B'} if ends is None else ends
        self.calibrated = calibrated
        self.players = {}
        self.ball = None          # t -> None (unseen) | (x, y) | (x, y, {'state': 'MISSING', 'firstPass': (x, y), ...})

    def add(self, pid, team, track, keys, role='PLAYER', state='confirmed', kit=None, roster=True):
        self.players[pid] = {'team': team, 'track': track, 'keys': keys, 'role': role, 'state': state,
                             'kit': team if kit is None else kit, 'roster': roster}

    def at(self, pid, t):
        keys = self.players[pid]['keys']
        if t <= keys[0][0]:
            return keys[0][1:]
        for (t0, x0, y0), (t1, x1, y1) in zip(keys, keys[1:]):
            if t0 <= t <= t1:
                s = (t-t0)/max(t1-t0, 1e-9)
                return x0+s*(x1-x0), y0+s*(y1-y0)
        return keys[-1][1:]

    def image(self, x, y):
        return U0+PX*x, V0+self.py*y

    def person(self, pid, t):
        sp = self.players[pid]
        x, y = self.at(pid, t)
        u, v = self.image(x, y)
        h, w = 1.8*PX, .7*PX
        team = sp['team']
        return {'track': sp['track'], 'box': [(u-w/2)/W, (v-h)/H, w/W, h/H], 'id': pid, 'display': pid, 'team': team,
                'role': sp['role'], 'label': f"{sp['role']}_TEAM_{team}" if team else sp['role'], 'state': sp['state'],
                'identityConfidence': .9, 'pitch': [x/105.0, y/68.0], 'pitchConfidence': .9, 'pitchSigma': [.3, .3],
                'why': kit_why(sp['kit'], team) if sp['kit'] in ('A', 'B') else None}

    def ball_record(self, x, y, speed, extra):
        u, v = self.image(x, y)
        d = 7.0
        state = extra.get('state', 'TRACKED')
        out = {'state': state, 'phase': 'PATH', 'observed': state == 'TRACKED',
               'origin': 'OBSERVATION' if state == 'TRACKED' else 'INTERPOLATION',
               'motion': ('STATIONARY' if speed < 1 else 'ROLLING' if speed < 8 else 'FAST') if state == 'TRACKED' else 'OCCLUDED',
               'speed': round(speed, 2), 'box': [(u-d/2)/W, (v-d)/H, d/W, d/H], 'center': [u/W, (v-d/2)/H],
               'pitch': [x/105.0, y/68.0], 'pitchConfidence': .9, 'pitchSigma': [.3, .3]}
        if 'firstPass' in extra:
            fu, fv = self.image(*extra['firstPass'])
            out['firstPass'] = {'state': 'TRACKED', 'center': [fu/W, (fv-d/2)/H]}
        return out

    def build(self, first=300):
        frames, prev = [], None
        Hm = [PX, 0.0, U0, 0.0, self.py, V0, 0.0, 0.0, 1.0]
        for i in range(self.n):
            t = (first+i)/FPS
            people = [self.person(pid, t) for pid in self.players]
            b = self.ball(t)
            f = {'time': round(t, 6), 'frame': first+i, 'people': people, 'profile': {'smallScene': self.small},
                 'view': self.view}
            if self.calibrated:
                f['calibration'] = {'H': Hm}
            if b is None:
                f['ball'] = {'state': 'UNKNOWN', 'phase': 'PATH', 'origin': 'UNKNOWN', 'motion': 'UNKNOWN'}
                prev = None
            else:
                x, y = b[0], b[1]
                extra = b[2] if len(b) > 2 else {}
                speed = 0.0 if prev is None else math.hypot(x-prev[0], y-prev[1])*FPS
                f['ball'] = self.ball_record(x, y, speed, extra)
                prev = (x, y)
            frames.append(f)
        rostered = [(pid, sp) for pid, sp in self.players.items() if sp['roster'] and sp['team'] in ('A', 'B')]
        players = [{'id': pid, 'display': pid, 'team': sp['team'], 'role': 'PLAYER', 'confirmed': True}
                   for pid, sp in rostered if sp['role'] == 'PLAYER']
        keepers = [{'id': pid, 'display': pid, 'team': sp['team'], 'role': 'GOALKEEPER', 'confirmed': True}
                   for pid, sp in rostered if sp['role'] == 'GOALKEEPER']
        return {'version': 3, 'video': {'name': 'synthetic.mp4', 'width': W, 'height': H, 'fps': FPS, 'frames': self.n},
                'frames': frames, 'match': {'players': players, 'goalkeepers': keepers, 'referees': [], 'retired': [],
                                            'excluded': [], 'withdrawn': []},
                'tactical': {'pitchSize': {'length': 105.0, 'width': 68.0, 'source': 'standard'}, 'goalEnds': dict(self.ends)}}


def feet(sc, pid, t, ahead=.3):
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


def why_at(rec, t):
    r = run_of(rec, t)
    return None if r is None else r.get('why')


def passing(receiver_team='A', **kw):
    """A-01 dribbles 10-12 s, passes 15 m to A-02 (or B-05) in 1.5 s, who keeps it 13.5-16.5 s; B-08 and A-03 far
    away. (A pass this long, with nobody at the ball until the receiver's control, breaks possession: no carry.)"""
    sc = Scene(6.5, **kw)
    sc.add('A-01', 'A', 1, [(10.0, 40, 30), (12.0, 46, 30), (16.5, 47, 32)])
    receiver = 'A-02' if receiver_team == 'A' else 'B-05'
    sc.add(receiver, receiver_team, 2, [(10.0, 61.5, 30), (13.5, 61.5, 30), (16.5, 70, 33)])
    sc.add('B-08', 'B', 3, [(10.0, 40, 50), (16.5, 45, 52)])
    sc.add('A-03', 'A', 4, [(10.0, 30, 10), (16.5, 35, 12)])

    def b(t):
        if t < 12.0:
            return feet(sc, 'A-01', t)
        if t < 13.5:
            return flight(feet(sc, 'A-01', 12.0), feet(sc, receiver, 13.5, ahead=-.3), 12.0, 13.5, t)
        return feet(sc, receiver, t, ahead=-.3)
    sc.ball = b
    return sc


class Dispatch(unittest.TestCase):
    def test_broadcast_result_gets_possession4_byte_for_byte(self):
        r = passing('B', small=False, py=16.5).build()
        ref = possession.compute(r)
        self.assertTrue(ref['available'])
        self.assertEqual(json.dumps(amateur.compute(r)), json.dumps(ref))
        self.assertEqual(json.dumps(amateur.compute(r, {'min_posterior': .9})), json.dumps(possession.compute(r, {'min_posterior': .9})),
                         'shared settings reach possession-4 as they would directly')
        self.assertEqual(json.dumps(amateur.compute(r, {'engine': 'possession-4', 'reach_r1': 3.0})), json.dumps(ref),
                         'settings of the estimate alone never reach possession-4')
        self.assertEqual(amateur.compute(r, {'engine': 'amateur'})['version'], amateur.VERSION, 'the estimate when asked for')

    def test_small_scene_gets_the_estimate(self):
        r = passing('A').build()
        self.assertFalse(possession.compute(r)['available'])
        self.assertEqual(amateur.dispatch_reason(possession.features(r)), 'small_scene')
        rec = amateur.compute(r)
        self.assertEqual(rec['version'], 'possession-amateur-1')
        self.assertTrue(rec['available'], rec.get('reason'))
        self.assertEqual(rec['basis'], 'image')
        self.assertEqual(rec['fallbackFrom']['version'], 'possession-4')
        self.assertEqual(rec['fallbackFrom']['gate'], 'small_scene')
        self.assertEqual(rec['quality']['level'], 'limited')
        self.assertTrue(any('image-based' in n for n in rec['quality']['notes']))
        self.assertEqual(json.dumps(amateur.compute(r, {'engine': 'possession-4'})), json.dumps(possession.compute(r)))

    def test_unjudged_footage_and_other_abstentions(self):
        sc = passing('A')
        sc.ball = lambda t: None
        r = sc.build()
        rec = amateur.compute(r)
        self.assertEqual((rec['version'], rec['available']), (amateur.VERSION, False), 'a small scene with no ball at all')
        self.assertIn(possession.compute(r)['reason'], rec['reason'], "possession-4's reason is quoted")
        self.assertIn('could not name a team', rec['reason'])
        json.dumps(rec, allow_nan=False)
        r2 = passing('A', small=False).build()
        for f in r2['frames']:
            f['ball'] = {'state': 'UNKNOWN'} if f['frame'] % 10 else f['ball']
        self.assertEqual(amateur.dispatch_reason(possession.features(r2)), 'ball')
        self.assertEqual(json.dumps(amateur.compute(r2)), json.dumps(possession.compute(r2)))


class Record(unittest.TestCase):
    def test_pass_between_teammates_keeps_possession(self):
        rec = amateur.compute(passing('A').build())
        self.assertEqual(rec['changes'], [])
        self.assertEqual(team_at(rec, 11.0), 'A')
        self.assertEqual(run_of(rec, 11.0)['carrier'], 'A-01')
        self.assertIsNone(team_at(rec, 12.7), 'a 1.5 s pass with nobody at the ball: no carry')
        self.assertEqual(why_at(rec, 12.7), 'nobody_at_ball')
        self.assertEqual(team_at(rec, 15.0), 'A')
        self.assertEqual(run_of(rec, 15.0)['carrier'], 'A-02')
        self.assertEqual(rec['shares']['B'], 0.0)

    def test_interception_is_a_low_confidence_span_boundary(self):
        rec = amateur.compute(passing('B').build())
        self.assertEqual(len(rec['changes']), 1, rec['changes'])
        c = rec['changes'][0]
        self.assertEqual((c['fromTeam'], c['toTeam'], c['lostBy'], c['wonBy'], c['confidence']), ('A', 'B', 'A-01', 'B-05', 'low'))
        self.assertEqual(c['id'], f"pc-{c['frame']}-B")
        named = [r for r in rec['timeline'] if r['team'] is not None]
        self.assertIn(c['lostAt'], [r['t1'] for r in named if r['team'] == 'A'], 'lostAt: the last named frame of the old team')
        self.assertIn(c['wonAt'], [r['t0'] for r in named if r['team'] == 'B'], 'wonAt: the first named frame of the new team')
        self.assertAlmostEqual(c['wonAt'], 13.5, delta=.5)
        self.assertEqual(team_at(rec, 15.0), 'B')

    def test_record_shape_and_never_high(self):
        recs = [amateur.compute(passing(k).build()) for k in ('A', 'B')]
        ref = possession.compute(passing('B', small=False).build())
        for rec in recs:
            json.dumps(rec, allow_nan=False)
            self.assertTrue(set(ref) <= set(rec), set(ref)-set(rec))
            self.assertEqual(set(rec['shares']), set(ref['shares']))
            for run in rec['timeline']:
                self.assertTrue(set(ref['timeline'][0]) <= set(run))
                self.assertIn(run['state'], possession.STATE_NAMES)
                self.assertNotEqual(run['confidence'], 'high')
                self.assertEqual('why' in run, run['state'] == 'unknown', run)
                if 'why' in run:
                    self.assertIn(run['why'], amateur.REASONS)
            for c in rec['changes']:
                self.assertEqual(set(c), set(ref['changes'][0]))
                self.assertEqual(c['confidence'], 'low')
            self.assertEqual(set(rec['quality']['abstained']) - set(amateur.REASONS), set())


class Reach(unittest.TestCase):
    def test_shape_of_the_calibration(self):
        A, k = amateur._shape([PX, 0, U0, 0, PX/8, V0, 0, 0, 1], U0+PX*50, V0+PX/8*30, 105.0, 68.0, 25.0)
        self.assertTrue(np.allclose(A, [[1, 0], [0, 8]]), A)
        self.assertAlmostEqual(k, 8.0)
        self.assertIsNone(amateur._shape(None, 0, 0, 105.0, 68.0, 25.0))
        self.assertAlmostEqual(amateur._shape([PX, 0, U0, 0, PX/40, V0, 0, 0, 1], U0, V0+10, 105.0, 68.0, 25.0)[1], 25.0,
                               msg='the depth factor is capped')

    def scene(self, depth_m, side_m=0.0):
        """A ball lying still; A-01 stands depth_m behind it (in depth) and side_m beside it; B-09 far away. The camera
        foreshortens depth 8 times (PY = PX / 8)."""
        sc = Scene(4.0, py=PX/8)
        sc.add('A-01', 'A', 1, [(10.0, 50+side_m, 30-depth_m), (14.0, 50+side_m, 30-depth_m)])
        sc.add('B-09', 'B', 2, [(10.0, 80, 50), (14.0, 80, 50)])
        sc.ball = lambda t: (50.0, 30.0)
        return sc.build()

    def test_depth_counts_as_metres(self):
        far = self.scene(3.0)
        D = amateur.decide(amateur.features(far))
        self.assertFalse((D['e'][0] > 0).any(), 'a player 3 m away in depth is not at the ball')
        # possession-4's image metric (rows doubled) puts him 0.75 m away: at the ball
        F = possession.features(far)
        self.assertTrue((F['dI'][0] < 1.0).all())
        near = self.scene(0.0, side_m=-.5)
        rec = amateur.compute(near)
        self.assertEqual(team_at(rec, 12.0), 'A')
        self.assertEqual(run_of(rec, 12.0)['carrier'], 'A-01')

    def test_without_calibration_the_row_weight(self):
        r = self.scene(1.0)
        G = amateur.features(r)
        self.assertTrue(np.allclose(G['depth'][np.isfinite(G['depth'])], 8.0))
        for f in r['frames'][::2]:
            f.pop('calibration')
        G = amateur.features(r)
        self.assertEqual(G['rowWeight'], 8.0, "the run's own median where 30 frames or more have a calibration")
        self.assertEqual(int(G['calibrated'].sum()), len(r['frames'])//2)
        for f in r['frames']:
            f.pop('calibration', None)
        self.assertEqual(amateur.features(r)['rowWeight'], amateur.DEFAULTS['row_weight'], 'a small scene: the amateur median')
        for f in r['frames']:
            f['profile'] = {'smallScene': False}
        self.assertEqual(amateur.features(r)['rowWeight'], amateur.DEFAULTS['row_weight_wide'], 'any other run: the broadcast one')


class Ball(unittest.TestCase):
    def test_long_bridge_is_untrusted(self):
        """The ball interpolated for 1 s (14.0-15.0 s) runs past B-04's feet: no evidence for B while the bridge is
        longer than max_bridge; trusted, it would be his."""
        sc = passing('A')
        sc.add('B-04', 'B', 9, [(10.0, 66.0, 31.5), (16.5, 66.0, 31.5)])
        base = sc.ball
        sc.ball = lambda t: ((*flight((63.0, 31.0), (66.3, 31.5), 14.0, 14.5, t), {'state': 'MISSING'}) if 14.0 <= t < 15.0
                             else base(t))
        r = sc.build()
        k = np.arange(len(r['frames']))
        bridge = (k >= int(4.4*FPS)) & (k < int(5.0*FPS))
        D = amateur.decide(amateur.features(r))
        self.assertFalse((D['e'][1][bridge] > 0).any())
        self.assertFalse(D['trusted'][bridge].any())
        self.assertEqual(amateur.compute(r)['changes'], [], 'no change rests on a long bridge')
        D = amateur.decide(amateur.features(r, {'max_bridge': 2.0}), {'max_bridge': 2.0})
        self.assertTrue((D['e'][1][bridge] > 0).any(), 'trusted, the bridged ball would be at B-04')

    def test_first_pass_on_another_team(self):
        sc = passing('A')
        sc.add('B-04', 'B', 9, [(10.0, 70, 50), (16.5, 70, 50)])
        base = sc.ball
        sc.ball = lambda t: (*base(t), {'firstPass': (70.3, 50.0)}) if t >= 14.0 else base(t)
        D = amateur.decide(amateur.features(sc.build()))
        late = np.arange(D['e'].shape[1]) >= int(4.0*FPS)
        self.assertFalse(D['trusted'][late].any(), 'the first pass on a B player 20 m away: the ball is not trusted')
        sc.add('A-06', 'A', 10, [(10.0, 70, 50), (16.5, 70, 50)])
        sc.players.pop('B-04')
        D = amateur.decide(amateur.features(sc.build()))
        self.assertTrue(D['trusted'][late].all(), 'both passes point to team A: trusted')


class Carrier(unittest.TestCase):
    def holder(self, **kw):
        """B-02 dribbles 10-11 s and passes to B-05 (the player varied), who keeps the ball 11.5-14 s; A-01 15 m away."""
        sc = Scene(4.0)
        sc.add('B-02', 'B', 2, [(10.0, 44, 30), (11.0, 46, 30), (14.0, 46, 30)])
        sc.add('B-05', 'B', 5, [(10.0, 50, 30), (14.0, 54, 30)], **kw)
        sc.add('A-01', 'A', 1, [(10.0, 30, 40), (14.0, 32, 40)])

        def b(t):
            if t < 11.0:
                return feet(sc, 'B-02', t)
            if t < 11.5:
                return flight(feet(sc, 'B-02', 11.0), feet(sc, 'B-05', 11.5, ahead=-.3), 11.0, 11.5, t)
            return feet(sc, 'B-05', t, ahead=-.3)
        sc.ball = b
        return sc.build()

    def test_kit_reading_must_agree(self):
        self.assertEqual(team_at(amateur.compute(self.holder()), 13.0), 'B')
        rec = amateur.compute(self.holder(kit='A'))
        self.assertEqual(team_at(rec, 10.5), 'B')
        self.assertIsNone(team_at(rec, 13.0), 'an identity B-05 on a player wearing A\'s kit carries nothing')
        self.assertEqual(why_at(rec, 13.0), 'carrier_unidentified')
        r = self.holder(kit='')       # the track has no kit reading at all
        self.assertEqual(team_at(amateur.compute(r), 13.0), 'B', 'no kit reading: accepted by default')
        self.assertIsNone(team_at(amateur.compute(r, {'kit_none': 'reject'}), 13.0), 'or refused when asked')

    def test_unconfirmed_track_is_nobody(self):
        rec = amateur.compute(self.holder(state='uncertain'))
        self.assertEqual(team_at(rec, 10.5), 'B')
        self.assertIsNone(team_at(rec, 13.0))
        self.assertEqual(why_at(rec, 13.0), 'carrier_unidentified')

    def test_teamless_keeper_from_goal_ends(self):
        def keeper(ends, settings=None):
            sc = Scene(4.0, ends=ends)
            sc.add('GK-1', None, 7, [(10.0, 6, 34), (14.0, 9, 34)], role='GOALKEEPER', state='uncertain', kit=None, roster=False)
            sc.add('A-01', 'A', 1, [(10.0, 40, 40), (14.0, 40, 40)])
            sc.add('B-01', 'B', 2, [(10.0, 60, 40), (14.0, 60, 40)])
            sc.ball = lambda t: feet(sc, 'GK-1', t)
            return amateur.compute(sc.build(), settings)
        self.assertEqual(team_at(keeper({'left': 'A', 'right': 'B'}), 12.0), 'A')
        self.assertEqual(team_at(keeper({'left': 'B', 'right': 'A'}), 12.0), 'B', 'the corrected ends decide')
        self.assertEqual(team_at(keeper({'left': None, 'right': 'A'}), 12.0), 'B', 'one end named gives the other')
        self.assertIsNone(team_at(keeper({'left': 'A', 'right': 'B'}, {'keeper_ends': 'ignore'}), 12.0), 'ends ignored: nobody')

    def test_contested_ball(self):
        """A-01 dribbles 10-12 s and stops; B-01 is at the ball with him 12-15 s."""
        sc = Scene(5.0)
        sc.add('A-01', 'A', 1, [(10.0, 45, 30), (12.0, 49.3, 30), (15.0, 49.3, 30)])
        sc.add('B-01', 'B', 2, [(10.0, 56, 30), (12.0, 50.7, 30), (15.0, 50.7, 30)])
        sc.ball = lambda t: feet(sc, 'A-01', t) if t < 12.0 else (50.0, 30.0)
        rec = amateur.compute(sc.build())
        self.assertEqual(team_at(rec, 11.0), 'A')
        self.assertIsNone(team_at(rec, 13.5))
        self.assertEqual(why_at(rec, 13.5), 'contested')


class NoCarry(unittest.TestCase):
    def scene(self, alone_s, unseen=False):
        """A-01 dribbles 10-12 s and leaves the ball; for alone_s nobody is within reach of it (or it is unseen); then
        A-02 picks it up and keeps it to the end."""
        end = 12.0+alone_s
        sc = Scene(4.0+alone_s)
        sc.add('A-01', 'A', 1, [(10.0, 40, 30), (12.0, 46, 30), (end, 40, 30)])
        sc.add('A-02', 'A', 2, [(10.0, 55, 40), (end, 50.3, 30.0), (14.0+alone_s, 54, 30)])
        sc.add('B-09', 'B', 3, [(10.0, 80, 50), (14.0+alone_s, 80, 50)])

        def b(t):
            if t < 12.0:
                return feet(sc, 'A-01', t)
            if t < end:
                return None if unseen else (50.0, 30.0)
            return feet(sc, 'A-02', t)
        sc.ball = b
        return amateur.compute(sc.build())

    def test_nobody_at_a_placed_ball_breaks_possession(self):
        rec = self.scene(2.5)
        self.assertEqual(team_at(rec, 11.0), 'A')
        self.assertIsNone(team_at(rec, 13.2), 'no carry across 2.5 s of a placed ball nobody was at')
        self.assertEqual(why_at(rec, 13.2), 'nobody_at_ball')
        self.assertEqual(team_at(rec, 15.5), 'A')
        self.assertEqual(rec['changes'], [])

    def test_a_short_stretch_or_an_unseen_ball_is_carried(self):
        rec = self.scene(1.0)
        self.assertEqual(team_at(rec, 12.5), 'A', 'one second alone: carried by the decoder')
        rec = self.scene(2.5, unseen=True)
        self.assertEqual(team_at(rec, 13.2), 'A', 'an unseen ball between two spells of A is carried')
        self.assertTrue(run_of(rec, 13.2)['estimated'])
        self.assertEqual(run_of(rec, 13.2)['confidence'], 'low')


class DeadBall(unittest.TestCase):
    def scene(self, view):
        """A-01 dribbles to the near touchline and leaves the ball there at 12 s; everyone walks away from it; A-02
        takes the throw-in at 18 s."""
        sc = Scene(10.0, view=view)
        sc.add('A-01', 'A', 1, [(10.0, 45, 60), (12.0, 50, 66), (18.0, 47, 60), (20.0, 45, 58)])
        sc.add('A-02', 'A', 2, [(10.0, 60, 50), (17.5, 50.3, 67.5), (20.0, 55, 60)])
        sc.add('B-01', 'B', 3, [(10.0, 40, 50), (20.0, 41, 50)])
        sc.add('B-02', 'B', 4, [(10.0, 70, 40), (20.0, 70.5, 40)])

        def b(t):
            if t < 12.0:
                return feet(sc, 'A-01', t)
            if t < 18.0:
                return (50.3, 67.6)
            return feet(sc, 'A-02', t)
        sc.ball = b
        return amateur.compute(sc.build())

    def test_dead_ball_with_the_touchline_in_view(self):
        rec = self.scene(FULL_VIEW)
        self.assertTrue(rec['deadSpells'], rec['timeline'])
        self.assertEqual(run_of(rec, 15.0)['state'], 'dead')
        self.assertNotEqual(run_of(rec, 15.0)['confidence'], 'high')

    def test_no_dead_ball_without_a_line_in_view(self):
        rec = self.scene(MIDDLE_VIEW)
        self.assertEqual(rec['deadSpells'], [])
        self.assertIsNone(team_at(rec, 15.0))
        self.assertEqual(why_at(rec, 15.0), 'registration')


if __name__ == '__main__':
    unittest.main()
