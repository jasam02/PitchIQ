"""Tests of the suggested-zone engine (soccer/coaching/alternatives.py; DESIGN v2 sections 5.1-5.3 and the 13.1
property tests).

Why these tests. A suggested zone is a claim that ONE team-mate could have stood there in time and that it would have
fixed what the detector found, without opening something worse (spec Part 8). So the tests check that claim with code
that does not share the engine's vectorized geometry (tests/coaching_zone_props.py re-measures each point with plain
tactics_geom loops):
  - random scenes per detector (D1 cover, D1 pressure, D2, D3, D4, D5), random side, field and level: every emitted
    zone is inside the reach disc, never within 3 m of a team-mate, restores the relation on at least 80% of its grid
    points, passes the no-new-gap checks on all of them; the records validate; zones are found in most random cases
    (the property is not vacuous);
  - every NoAlternative reason of DESIGN 5.1 on a hand-built scene (guardrail_failed, scale_unknown, mover_unseen,
    context_missing, no_reachable_fix, would_open_new_gap, no_zone_for_detector) and None for attacking shape;
  - the level's parameters: the age speed cap shrinks the reach, a point only at level 4+ with sigma <= 1 m on a
    'world' frame, a direction only for a zone above a quarter of the half;
  - the wide-defender advance (critic B-8): no zone vertex more than LEVEL_M upfield of the deepest opponent in the
    vacated channel (a sliver or an empty channel leaves the principle only, no_reachable_fix; a measured channel
    opponent missing from the scene gives context_missing), and level 2's "someone stays back" is a spot behind the
    ball restoring the behind-the-ball count, not the channel;
  - EVAL's synthetic scenes through the real tiers (DESIGN 13.1 rows): the planted unsupported advances (their channel
    opponent deeper than every reachable spot) keep the principle at every level from the same moment, an identity
    crossing or a single instance draws nothing, covered advances are never criticised, the defending scene's cover
    zone moves a team-mate and holds the properties on a pinned frame too, the poorer role-tier scene still gives a
    valid record, outputs are deterministic and inputs untouched;
  - module style: ASCII, CRLF, principle codes that TEXT words, reasons the schema knows.
"""
import copy
import gc
import math
import time
import unittest
from pathlib import Path

import numpy as np

import coaching_detect_fixtures as F
import coaching_zone_fixtures as Z
import coaching_zone_props as P
from soccer.coaching import alternatives as A, contextual as CX, explain as E, profiles as PR, schema as S
from soccer.coaching import tactics_geom as G

ROOT = Path(__file__).resolve().parents[1]
T0, FRAME, FPS = 700.0, 17500, 25.0
AGE = {1: 'U8', 2: 'U10', 3: 'U12', 4: 'U14', 5: 'U16', 6: 'Adult', 7: 'Adult'}


# ---------------------------------------------------------------------------------------------------------------
# A hand-built case: one opportunity with one sample, its moment (scene now and REACH_H earlier) and a finding.
class Case:
    def __init__(self, detector, side='left', L=105.0, W=68.0, players=11):
        self.detector, self.side, self.L, self.W, self.players = detector, side, L, W, players
        self.own, self.opp, self.before, self.sigma, self.p90 = {}, {}, {}, {}, {}
        self.keeper = (2.0, W/2.0)
        self.ball = None
        self.rel, self.actors, self.variant, self.records, self.medians = {}, {}, None, [], {}
        self.tier, self.horizon, self.mode, self.verdict = 'world', 2.0, None, 'work_on'
        self.guardrails = {g: 'pass' for g in S.GUARDRAILS}
        self.observation = detector
        self.declared = True

    def norm(self, X, Y):
        return (X/self.L, Y/self.W) if self.side == 'left' else (1-X/self.L, 1-Y/self.W)

    def entry(self, team, pid, X, Y, keeper=False):
        x, y = self.norm(X, Y)
        return {'team': team, 'x': x, 'y': y, 'kind': 'O', 'sigma': self.sigma.get(pid, .4), 'goalkeeper': keeper,
                'box': None}

    def scene(self, t, pos):
        players = {pid: self.entry('A', pid, *p) for pid, p in pos.items()}
        players['GK-1'] = self.entry('A', 'GK-1', *self.keeper, keeper=True)
        for pid, p in self.opp.items():
            players[pid] = self.entry('B', pid, *p)
        ball = list(self.norm(*self.ball)) if self.ball is not None else None
        return {'t': t, 'frame': FRAME-int(round((T0-t)*FPS)), 'players': players, 'ball': ball,
                'ballReliable': ball is not None}

    def files(self):
        L, W = self.L, self.W
        key = S.opportunity_key(self.detector, 'A', T0, T0, self.variant)
        oid = S.record_id('opp', key)
        phase = 'own' if self.detector in ('wide_defender_advance', 'behind_ball') else 'opp'
        tier = 'role' if self.detector == 'wide_defender_advance' else 'measurement'
        opp = {'id': oid, 'key': key, 'detector': self.detector, 'team': 'A', 'tier': tier, 't0': T0, 't1': T0,
               'frame0': FRAME, 'frame1': FRAME,
               'phase': {'state': phase, 'source': 'possession-4', 'estimated': True, 'spanSeconds': 8.0,
                         'nearChange': False},
               'ball': {'zone': None, 'channel': None, 'reliable': self.ball is not None},
               'actors': {'primary': [], 'supporting': [], 'opponents': [], 'carrier': None, **self.actors},
               'roles': {}, 'relations': {'t': [T0], **{k: [v] for k, v in self.rel.items()}},
               'quality': {'identity': 'high', 'position': 'high', 'visibility': 'complete',
                           'ball': 'reliable' if self.ball is not None else 'not_needed', 'samples': 1},
               'intentSignals': {'recovering': False, 'interchange': None, 'formationChangeNear': False},
               'abstain': None}
        if self.variant:
            opp['variant'] = self.variant
        before = self.scene(T0-self.horizon, self.before) if self.before is not None else None
        m = {'id': S.record_id('ev', oid, 0), 'opportunity': oid, 'detector': self.detector, 'team': 'A', 'sample': 0,
             't': T0, 'frame': FRAME, 'seek': S.seek_of(FRAME, FPS), 'fps': FPS, 'tier': self.tier,
             'calibration': None, 'players': [], 'flagged': [], 'scene': self.scene(T0, self.own), 'before': before,
             'image': None}
        players = {pid: {'team': 'A', 'goalkeeper': False, 'observedSeconds': 60.0,
                         'p90SpeedMS': self.p90.get(pid, 6.0)} for pid in self.own}
        players.update({pid: {'team': 'B', 'goalkeeper': False, 'observedSeconds': 60.0, 'p90SpeedMS': 6.0}
                        for pid in self.opp})
        scale = {'lengthM': L, 'widthM': W, 'source': 'declared' if self.declared else 'assumed',
                 'metresUsable': self.declared}
        bx = by = None
        if self.ball is not None:
            bx, by = self.ball
        obs = {'team': 'A', 'timeline': {'scale': scale, 'players': players, 'format': f'{self.players}v{self.players}',
                                         'step': .2},
               'opportunities': [] if tier == 'role' else [opp], 'moments': [] if tier == 'role' else [m],
               'roleInputs': {'detect': {'version': 1, 'step': .2, 'medianY': self.medians,
                                         'samples': {'t': [T0], 'side': [self.side], 'period': [-1], 'ballX': [bx],
                                                     'ballY': [by]}}}}
        roles = {'opportunities': [opp] if tier == 'role' else [], 'moments': [m] if tier == 'role' else [],
                 'rolebook': {'records': self.records}}
        fkey = S.finding_key(self.observation, 'A', self.variant or 'team', T0, T0)
        finding = {'id': S.record_id('fnd', fkey), 'key': fkey, 'observation': self.observation, 'team': 'A',
                   'verdict': self.verdict, 'guardrails': dict(self.guardrails),
                   'measurements': {'failureMode': self.mode} if self.mode else {},
                   'timestamps': [{'evidence': m['id'], 'opportunity': oid, 't': T0, 'frame': FRAME,
                                   'seek': m['seek']}]}
        return finding, obs, roles, m, opp


def profile(level, players=11, field=None, **params):
    block = {'playingLevel': level, 'ageGroup': AGE[level], 'format': {'players': players, 'goalkeeper': True}}
    if field:
        block['field'] = {'length': field[0], 'width': field[1]}
    p = PR.resolve(block, team='A')
    for det, kv in params.items():
        p['detectors'][det]['params'].update(kv)
    return p


def failing(case, prof):
    """Does the case's stored sample fail under the profile (the engine's precondition)?"""
    finding, obs, roles, m, opp = case.files()
    det = case.observation
    as_det = det if det != case.detector else None
    outs, modes = CX.sample_outcomes(opp, prof['detectors'][det]['params'], obs['timeline']['scale'],
                                     as_detector=as_det)
    return outs[0] == -1, modes[0]


def displace(rng, pos, reach):
    """Positions REACH_H earlier: each moved back by a random vector of at most `reach` metres."""
    out = {}
    for pid, (x, y) in pos.items():
        a, d = rng.uniform(0, 2*math.pi), rng.uniform(0, reach)
        out[pid] = (x+d*math.cos(a), y+d*math.sin(a))
    return out


# ---------- random scene generators per detector (attack frame, team A) ----------
def random_d1_cover(rng, level):
    small = bool(rng.random() < .4)         # a 55 x 36 m field (7v7 size) on some scenes: scaled radii, a 0.52 m grid
    c = Case('pressure_cover', side=rng.choice(['left', 'right']), L=55.0 if small else 105.0, W=36.0 if small else 68.0)
    L, W = c.L, c.W
    kx, ky = L/105.0, W/68.0
    radius = G.cover_radius({'lengthM': L, 'widthM': W})
    for _ in range(200):
        cx, cy = rng.uniform(12, 45)*kx, rng.uniform(8, 60)*ky
        a = rng.uniform(-1.2, 1.2)
        d1 = rng.uniform(1.5, 3.0)      # pressed without closing: within 3 m (the three-state call, critic S-2)
        g = math.atan2(W/2-cy, -cx)
        fd = (cx+d1*math.cos(g+a), cy+d1*math.sin(g+a))
        own = {'A-01': fd}
        k = 2
        while k <= 10:
            p = (rng.uniform(4, 70)*kx, rng.uniform(2, 66)*ky)
            if math.hypot(p[0]-cx, p[1]-cy) <= d1+1.0:
                continue
            if G.covers(fd, p, W, radius, 60.0):
                continue
            own[f'A-{k:02d}'] = p
            k += 1
        c.own = own
        c.opp = {'B-01': (cx, cy)}
        for j in range(2, 11):
            c.opp[f'B-{j:02d}'] = (rng.uniform(15, 100)*kx, rng.uniform(2, 66)*ky)
        X = np.array([own[p][0] for p in sorted(own)])
        Y = np.array([own[p][1] for p in sorted(own)])
        f = sorted(own).index('A-01')
        (cj, cd, ca), (sj, sd, _) = G.cover_candidate(fd[0], fd[1], X, Y, W, exclude=[f])
        c.rel = {'d1': d1, 'fd': 'A-01', 'fdGoalSide': fd[0] < cx, 'closingMS': 0.0,
                 'coverId': None if cj is None else sorted(own)[cj], 'coverDist': None if cj is None else cd,
                 'coverAngle': None if cj is None else ca, 'coverVisible': True,
                 'secondId': None if sj is None else sorted(own)[sj], 'secondDist': None if sj is None else sd,
                 'goalSideShare': .5, 'primaryNamed': True, 'carrierX': cx, 'carrierY': cy,
                 'carrierZone': G.zone_of(cx, L), 'complete': True}
        c.actors = {'primary': ['A-01'], 'opponents': ['B-01'], 'carrier': 'B-01'}
        c.ball = (cx, cy)
        c.p90 = {pid: rng.uniform(3.5, 7.5) for pid in own}
        c.before = displace(rng, own, 6.0*kx)
        c.mode = 'no_cover'
        prof = profile(level, field=(L, W) if small else None)
        ok, mode = failing(c, prof)
        if ok and mode == 'no_cover':
            return c, prof
    raise AssertionError('no failing D1 cover scene drawn')


def random_d1_pressure(rng, level=1):
    c = Case('pressure_cover', side=rng.choice(['left', 'right']))
    W = c.W
    for _ in range(200):
        cx, cy = rng.uniform(15, 45), rng.uniform(8, 60)
        own = {}
        for k in range(1, 11):
            if k <= 3:          # a few players near the carrier (6-11 m), the rest anywhere
                a, d = rng.uniform(0, 2*math.pi), rng.uniform(6.0, 11.0)
                p = (cx+d*math.cos(a), min(W-1, max(1, cy+d*math.sin(a))))
            else:
                for _ in range(100):
                    p = (rng.uniform(4, 70), rng.uniform(2, W-2))
                    if math.hypot(p[0]-cx, p[1]-cy) > 6.0:
                        break
            own[f'A-{k:02d}'] = p
        X = np.array([own[p][0] for p in sorted(own)])
        Y = np.array([own[p][1] for p in sorted(own)])
        j, d1 = G.first_defender(cx, cy, X, Y)
        fd = sorted(own)[j]
        c.own = own
        c.opp = {'B-01': (cx, cy)}
        for i in range(2, 11):
            c.opp[f'B-{i:02d}'] = (rng.uniform(15, 100), rng.uniform(2, W-2))
        c.rel = {'d1': d1, 'fd': fd, 'fdGoalSide': True, 'closingMS': 0.0, 'coverId': None, 'coverDist': None,
                 'coverAngle': None, 'coverVisible': True, 'secondId': None, 'secondDist': None,
                 'goalSideShare': .5, 'primaryNamed': True, 'carrierX': cx, 'carrierY': cy,
                 'carrierZone': G.zone_of(cx, c.L), 'complete': True}
        c.actors = {'primary': [fd], 'opponents': ['B-01'], 'carrier': 'B-01'}
        c.ball = (cx, cy)
        c.p90 = {pid: rng.uniform(4.0, 7.5) for pid in own}
        c.before = displace(rng, own, 5.0)
        c.mode = 'no_pressure'
        prof = profile(level)
        ok, mode = failing(c, prof)
        if ok and mode == 'no_pressure':
            return c, prof
    raise AssertionError('no failing D1 pressure scene drawn')


def random_d2(rng, level):
    c = Case('weak_side_balance', side=rng.choice(['left', 'right']))
    W = c.W
    bx, by = rng.uniform(25, 60), rng.uniform(3, W/3-4)
    own = {}
    for k in range(1, 11):
        if k <= 4:      # a back line on the ball side, goal-side of the ball
            p = (rng.uniform(8, 20), rng.uniform(4, W/2-3))
        elif k <= 6:    # far-side players ahead of the ball
            p = (rng.uniform(bx+3, bx+25), rng.uniform(W/2+2, W-3))
        else:
            p = (rng.uniform(bx-15, bx+20), rng.uniform(3, W/2-2))
        own[f'A-{k:02d}'] = p
    c.own = own
    c.opp = {f'B-{j:02d}': (rng.uniform(bx-10, 100), rng.uniform(2, W-2)) for j in range(1, 11)}
    c.opp['B-01'] = (bx+.5, by)
    c.ball = (bx, by)
    X = np.array([own[p][0] for p in sorted(own)])
    Y = np.array([own[p][1] for p in sorted(own)])
    w = G.weak_side(bx, by, X, Y, W, {'lengthM': c.L, 'widthM': W}, side='left')
    c.rel = {'ballSide': 'left', 'weakCount': w['weakCount'], 'openWidthM': w['openWidthM'],
             'centralCount': w['centralCount'], 'nearestWeak': None, 'primaryNamed': True, 'ballX': bx, 'ballY': by}
    c.actors = {'primary': ['A-01'], 'opponents': ['B-01'], 'carrier': 'B-01'}
    c.p90 = {pid: rng.uniform(4.0, 7.5) for pid in own}
    c.before = displace(rng, own, 5.0)
    return c, profile(level)


def random_d3(rng, level):
    c = Case('wide_defender_advance', side=rng.choice(['left', 'right']))
    L, W = c.L, c.W
    for _ in range(200):
        cbx = rng.uniform(22, 38)
        own = {'A-02': (cbx+rng.uniform(-1.5, 1.5), rng.uniform(24, 30)),
               'A-03': (cbx+rng.uniform(-1.5, 1.5), rng.uniform(38, 44)),
               'A-04': (cbx+rng.uniform(-3, 6), rng.uniform(56, 64))}
        cbm = (own['A-02'][0]+own['A-03'][0])/2
        wd = (cbm+rng.uniform(17, 32), rng.uniform(3, 18))
        own['A-01'] = wd
        for k in range(5, 11):
            for _ in range(100):
                p = (rng.uniform(cbm, 90), rng.uniform(4, W-4))
                if not (G.in_outer_third(p[1], 'left', W) and p[0] <= wd[0]-10.0):
                    break
            own[f'A-{k:02d}'] = p
        ball = own['A-07']
        c.own, c.ball = own, ball
        c.opp = {f'B-{j:02d}': (rng.uniform(30, 100), rng.uniform(2, W-2)) for j in range(1, 11)}
        c.opp['B-01'] = (rng.uniform(cbm+3, min(wd[0]-2, L/2-1)), rng.uniform(3, 18))
        c.records = [{'id': pid, 't0': T0-30, 't1': T0+30, 'role': role, 'group': 'D', 'side': None,
                      'source': 'coach', 'confidence': 'high', 'support': None, 'period': None, 'ambiguousWith': [],
                      'conflict': None}
                     for pid, role in (('A-01', 'LB'), ('A-02', 'LCB'), ('A-03', 'RCB'), ('A-04', 'RB'))]
        c.medians = {'A-02': {'-1': own['A-02'][1]}, 'A-03': {'-1': own['A-03'][1]}}
        X = np.array([own[p][0] for p in sorted(own)])
        Y = np.array([own[p][1] for p in sorted(own)])
        occ, by_ = G.channel_occupied(X, Y, sorted(own).index('A-01'), 'left', W, 10.0)
        behind = G.behind_ball(X, ball[0], 10.0)[0]
        och = len(G.opponents_in_channel(np.array([o[0] for o in c.opp.values()]),
                                         np.array([o[1] for o in c.opp.values()]), wd[0], 'left', W, L))
        c.rel = {'advanceM': wd[0]-cbm, 'beyondHalfM': wd[0]-L/2, 'channelOccupied': bool(occ), 'channelBy': None,
                 'dmDropped': None, 'cbAdjusted': False, 'nearCb': 'A-02', 'nearCbShiftM': 0.0, 'behindN': behind,
                 'interchange': False, 'opponentsInChannel': och, 'threatN': None, 'restMargin': None,
                 'dmCbSpacingM': None, 'wdVelX': .5, 'complete': True, 'relaxed': False, 'primaryNamed': True,
                 'wdSource': 'coach', 'wdConfidence': 'high'}
        c.variant = 'left'
        c.actors = {'primary': ['A-01'], 'supporting': ['A-02'], 'opponents': ['B-01']}
        c.p90 = {pid: rng.uniform(4.0, 7.5) for pid in own}
        c.before = displace(rng, own, 5.0)
        prof = profile(level)
        if failing(c, prof)[0]:
            return c, prof
    raise AssertionError('no failing D3 scene drawn')


def random_d4(rng, level):
    c = Case('behind_ball', side=rng.choice(['left', 'right']))
    W = c.W
    for _ in range(200):
        bx, by = rng.uniform(72, 95), rng.uniform(8, 60)
        own = {'A-01': (bx-rng.uniform(11, 25), rng.uniform(20, 48))}
        for k in range(2, 11):
            own[f'A-{k:02d}'] = (rng.uniform(bx-9, min(103, bx+10)), rng.uniform(3, W-3))
        own['A-09'] = (bx-.5, by)
        c.own, c.ball = own, (bx, by)
        c.opp = {f'B-{j:02d}': (rng.uniform(60, 103), rng.uniform(2, W-2)) for j in range(1, 11)}
        X = np.array([own[p][0] for p in sorted(own)])
        n = G.behind_ball(X, bx, 10.0)[0]
        c.rel = {'ballX': bx, 'behindN': n, 'threatN': 1, 'restMargin': n-1, 'behindSpreadM': None, 'deepest': 'A-01',
                 'behindIds': 'A-01', 'oppComplete': True}
        c.actors = {'primary': ['A-01']}
        c.p90 = {pid: rng.uniform(4.0, 7.5) for pid in own}
        c.before = displace(rng, own, 5.0)
        prof = profile(level)
        if failing(c, prof)[0]:
            return c, prof
    raise AssertionError('no failing D4 scene drawn')


def random_d5(rng, level):
    c = Case('defensive_spacing', side=rng.choice(['left', 'right']))
    L, W = c.L, c.W
    for _ in range(200):
        lx = rng.uniform(14, 30)
        ys = sorted([rng.uniform(5, 14), rng.uniform(18, 26), rng.uniform(28, 33), rng.uniform(55, 64)])
        own = {f'A-0{k+1}': (lx+rng.uniform(-1.5, 1.5), y) for k, y in enumerate(ys)}
        for k in range(5, 11):
            own[f'A-{k:02d}'] = (rng.uniform(lx+12, 70), rng.uniform(4, W-4))
        c.own = own
        c.ball = (rng.uniform(lx+15, 70), rng.uniform(40, 62))
        c.opp = {f'B-{j:02d}': (rng.uniform(lx+8, 100), rng.uniform(2, W-2)) for j in range(1, 11)}
        c.opp['B-02'] = (lx+rng.uniform(1, 10), rng.uniform(36, 50))
        mem = ['A-01', 'A-02', 'A-03', 'A-04']
        X = np.array([own[p][0] for p in mem])
        Y = np.array([own[p][1] for p in mem])
        gap, a, b, gy = G.lateral_gaps(Y)
        inn = G.opponent_in_gap(np.array([o[0] for o in c.opp.values()]), np.array([o[1] for o in c.opp.values()]),
                                (X[a], Y[a]), (X[b], Y[b]), {'lengthM': L, 'widthM': W})
        prim = mem[b]
        c.rel = {'lineStdM': G.line_std(X), 'lineSize': 4, 'lateralGapM': gap, 'gapLeft': mem[a], 'gapRight': mem[b],
                 'gapY': gy, 'gapBallSide': gy > W/2, 'opponentInGap': bool(inn), 'gapOpponents': len(inn),
                 'ballSide': 'right', 'dmGapM': 11.0, 'dmGapState': '', 'lineHeightM': float(np.mean(X)),
                 'blockHeight': G.block_height(float(np.mean(X)), L), 'primaryNamed': True}
        c.actors = {'primary': [prim], 'supporting': [mem[a]]}
        c.p90 = {pid: rng.uniform(4.0, 7.5) for pid in own}
        c.before = displace(rng, own, 5.0)
        c.mode = 'lateral_gap'
        prof = profile(level)
        ok, mode = failing(c, prof)
        if ok and mode == 'lateral_gap':
            return c, prof
    raise AssertionError('no failing D5 scene drawn')


GENERATORS = {'D1 cover': (random_d1_cover, (2, 3, 4, 5, 6, 7)), 'D1 pressure': (random_d1_pressure, (1,)),
              'D2': (random_d2, (3, 4, 5, 6, 7)), 'D3': (random_d3, (3, 4, 5, 6, 7)), 'D4': (random_d4, (4, 5, 6, 7)),
              'D5': (random_d5, (5, 6, 7))}
RANDOM_N = 40           # cases per generator


class RandomSceneProperties(unittest.TestCase):
    """DESIGN 13.1: every zone on random scenes is reachable, unstacked, restoring and gap-free; records validate."""

    @classmethod
    def setUpClass(cls):
        cls.results = {}
        for name, (gen, levels) in GENERATORS.items():
            rng = np.random.default_rng(sum(map(ord, name)))
            rows = []
            for i in range(RANDOM_N):
                level = levels[i % len(levels)]
                case, prof = gen(rng, level)
                case.sigma = {pid: float(rng.uniform(.3, 1.5)) for pid in list(case.own)+list(case.opp)}
                finding, obs, roles, m, opp = case.files()
                # The cost of propose itself (EVAL, fix round 2): in the full suite one timed call held a full garbage
                # collection of the suite's 4.6 M tracked objects (615 ms; the call itself takes 3 ms, every other
                # one 22 ms or less), so the collector is held back during the call and runs after it.
                held = gc.isenabled()
                gc.disable()
                try:
                    t = time.perf_counter()
                    alt = A.propose(finding, obs, roles, prof)
                    took = time.perf_counter()-t
                finally:
                    if held:
                        gc.enable()
                rows.append((case, prof, finding, obs, roles, alt, took))
            cls.results[name] = rows

    def test_every_zone_has_the_properties(self):
        for name, rows in self.results.items():
            for case, prof, finding, obs, roles, alt, _ in rows:
                if alt is None or 'zone' not in alt:
                    continue
                with self.subTest(generator=name, level=prof['level'], side=case.side):
                    self.assertEqual(P.zone_problems(alt, finding, obs, roles, prof, case.side), [])

    def test_zones_are_found_on_most_random_scenes(self):
        # The property must not hold vacuously: each generator yields zones on most of its scenes.
        for name, rows in self.results.items():
            zones = sum(1 for r in rows if r[5] and 'zone' in r[5])
            with self.subTest(generator=name):
                self.assertGreaterEqual(zones, RANDOM_N//2, f'{name}: {zones} zones of {RANDOM_N}')

    def test_every_record_validates(self):
        for name, rows in self.results.items():
            for *_, alt, _ in rows:
                kind = 'alternative' if 'zone' in alt else 'no_alternative'
                with self.subTest(generator=name):
                    self.assertEqual(S.validate(kind, alt), [])
                    self.assertIn(alt['reason'] if kind == 'no_alternative' else 'zone', S.ALT_REASONS+('zone',))

    def test_movers_follow_the_principle(self):
        for name, rows in self.results.items():
            for case, prof, finding, obs, roles, alt, _ in rows:
                if 'zone' not in alt:
                    continue
                poly = [(case.L*(x if case.side == 'left' else 1-x), case.W*(y if case.side == 'left' else 1-y))
                        for x, y in alt['zone']['polygon']]
                with self.subTest(generator=name):
                    if name == 'D1 cover':      # the first defender is never moved away from the ball; cover is behind him
                        self.assertNotEqual(alt['mover'], case.rel['fd'])
                        fx = case.own[case.rel['fd']][0]
                        self.assertLess(A.polygon_centroid(poly)[0], fx)
                        self.assertLessEqual(max(p[0] for p in poly), fx+2.0)        # + the sigma dilation
                    if name == 'D1 pressure':   # near the carrier, goal-side, never on top of him
                        cx, cy = case.rel['carrierX'], case.rel['carrierY']
                        self.assertFalse(P.inside(poly, (cx, cy)))
                        self.assertGreaterEqual(P.edge_distance(poly, (cx, cy)), A.PRESS_MIN_M-.02)
                        self.assertLessEqual(A.polygon_centroid(poly)[0], cx)
                    if name in ('D3', 'D4'):    # our player on the ball is never moved
                        self.assertGreater(math.dist(case.own[alt['mover']], case.ball), A.ON_BALL_M)
                    if name == 'D5':
                        self.assertIn(alt['mover'], (case.rel['gapLeft'], case.rel['gapRight']))
                    self.assertNotIn(alt['mover'], [r['mover'] for r in alt['rejected']])
                    self.assertEqual(alt['assumption'], 'single_mover')

    def test_the_arrow_starts_at_the_mover_and_the_reach_matches_his_speed(self):
        for name, rows in self.results.items():
            for case, prof, finding, obs, roles, alt, _ in rows:
                if 'zone' not in alt:
                    continue
                with self.subTest(generator=name):
                    x, y = case.norm(*case.own[alt['mover']])
                    self.assertAlmostEqual(alt['from'][0], x, places=3)
                    self.assertAlmostEqual(alt['from'][1], y, places=3)
                    cap = prof['detectors'][finding['observation']]['params']['reachSpeedCapMS']
                    self.assertAlmostEqual(alt['reach']['speedMS'], round(min(case.p90[alt['mover']], cap), 2),
                                           places=6)

    def test_cost(self):
        worst = max(r[6] for rows in self.results.values() for r in rows)
        self.assertLess(worst, .25, f'slowest propose {worst*1000:.0f} ms')


# ---------------------------------------------------------------------------------------------------------------
def d1_fixed(**kw):
    """A fixed D1 'pressed without cover' scene, team A defending the left goal (attack frame = pitch)."""
    c = Case('pressure_cover', side=kw.pop('side', 'left'))
    c.own = {'A-01': (30.0, 34.0), 'A-02': (14.0, 20.0), 'A-03': (14.0, 48.0), 'A-04': (40.0, 10.0),
             'A-05': (40.0, 58.0), 'A-06': (50.0, 34.0), 'A-07': (60.0, 20.0), 'A-08': (60.0, 48.0),
             'A-09': (70.0, 30.0), 'A-10': (70.0, 40.0)}
    c.opp = {'B-01': (33.0, 34.0), 'B-02': (50.0, 20.0), 'B-03': (50.0, 50.0), 'B-04': (70.0, 10.0),
             'B-05': (70.0, 58.0), 'B-06': (80.0, 34.0), 'B-07': (90.0, 20.0), 'B-08': (90.0, 48.0),
             'B-09': (95.0, 30.0), 'B-10': (95.0, 40.0)}
    c.ball = (33.0, 34.0)
    c.before = dict(c.own)
    c.p90 = {pid: 6.0 for pid in c.own}
    c.rel = {'d1': 3.0, 'fd': 'A-01', 'fdGoalSide': True, 'closingMS': 0.0, 'coverId': None, 'coverDist': None,
             'coverAngle': None, 'coverVisible': True, 'secondId': 'A-02', 'secondDist': 21.3, 'goalSideShare': .2,
             'primaryNamed': True, 'carrierX': 33.0, 'carrierY': 34.0, 'carrierZone': 'own_third', 'complete': True}
    c.actors = {'primary': ['A-01'], 'opponents': ['B-01'], 'carrier': 'B-01'}
    c.mode = 'no_cover'
    for k, v in kw.items():
        setattr(c, k, v)
    return c


class Reasons(unittest.TestCase):
    """Every NoAlternative reason of DESIGN 5.1 (and None for attacking shape) on a hand-built scene."""

    def setUp(self):
        self.prof = profile(4)

    def propose(self, case, prof=None):
        finding, obs, roles, m, opp = case.files()
        return A.propose(finding, obs, roles, prof or self.prof), finding, m

    def test_the_fixed_scene_gets_a_zone_from_the_nearest_covering_team_mate(self):
        finding, obs, roles, m, opp = d1_fixed().files()
        alt = A.propose(finding, obs, roles, self.prof)
        self.assertIn('zone', alt)
        self.assertIn(alt['mover'], ('A-02', 'A-03'))
        self.assertEqual(alt['improves']['relation'], 'cover')
        self.assertEqual(alt['principle'], 'cover_first_defender')
        self.assertEqual(alt['evidence'], m['id'])
        self.assertEqual(alt['finding'], finding['id'])
        self.assertEqual(P.zone_problems(alt, finding, obs, roles, self.prof, 'left'), [])

    def test_attacking_shape_has_no_zones(self):
        c = d1_fixed()
        c.observation = 'attacking_shape'
        self.assertIsNone(self.propose(c)[0])

    def test_a_failed_guardrail_draws_nothing_precise(self):
        for g in ('identity', 'position', 'role', 'context', 'phase', 'intent', 'evidence'):
            c = d1_fixed()
            c.guardrails[g] = 'fail'
            alt, _, m = self.propose(c)
            with self.subTest(guardrail=g):
                self.assertEqual(alt['reason'], 'guardrail_failed')
                self.assertEqual(alt['evidence'], m['id'])
                self.assertEqual(S.validate('no_alternative', alt), [])

    def test_unusable_metres_draw_nothing(self):
        c = d1_fixed()
        c.declared = False
        self.assertEqual(self.propose(c)[0]['reason'], 'scale_unknown')

    def test_a_mover_not_seen_two_seconds_earlier_is_never_teleported(self):
        alt = self.propose(d1_fixed(before=None))[0]
        self.assertEqual(alt['reason'], 'mover_unseen')
        self.assertEqual({r['reason'] for r in alt['rejected']}, {'mover_unseen'})
        c = d1_fixed()
        c.horizon = 3.0             # the 'before' scene is not at the reach horizon
        self.assertEqual(self.propose(c)[0]['reason'], 'mover_unseen')

    def test_a_scene_that_does_not_show_the_failure_draws_nothing(self):
        c = d1_fixed()
        c.own = dict(c.own, **{'A-02': (24.0, 33.0)})      # the scene has a covering team-mate after all
        self.assertEqual(self.propose(c)[0]['reason'], 'context_missing')
        c = d1_fixed()
        c.own = {pid: p for pid, p in c.own.items() if pid != 'A-01'}   # the first defender is not in the scene
        self.assertEqual(self.propose(c)[0]['reason'], 'context_missing')

    def test_a_sample_that_holds_under_the_level_is_not_a_problem(self):
        c = d1_fixed()
        c.rel = dict(c.rel, coverDist=8.0, coverAngle=20.0, coverId='A-02')
        self.assertEqual(self.propose(c)[0]['reason'], 'context_missing')

    def test_slow_team_mates_cannot_reach_the_cover(self):
        c = d1_fixed()
        c.p90 = {pid: .4 for pid in c.own}
        alt = self.propose(c)[0]
        self.assertEqual(alt['reason'], 'no_reachable_fix')
        self.assertEqual({r['reason'] for r in alt['rejected']}, {'no_reachable_fix'})

    def test_every_reachable_cover_spot_would_free_an_opponent(self):
        # A-02 and A-03 each mark an opponent in our half; the cover sector is far from both opponents.
        c = d1_fixed()
        c.opp = dict(c.opp, **{'B-04': (12.0, 12.0), 'B-05': (12.0, 56.0)})
        alt = self.propose(c)[0]
        self.assertEqual(alt['reason'], 'would_open_new_gap')
        self.assertEqual({r['reason'] for r in alt['rejected']}, {'would_open_new_gap'})

    def test_a_midfield_to_defence_gap_is_not_a_single_player_fix(self):
        c = d1_fixed()
        c.observation = 'defensive_spacing'
        c.mode = 'dm_gap'
        self.assertEqual(self.propose(c, profile(5))[0]['reason'], 'no_zone_for_detector')

    def test_reasons_are_the_schemas(self):
        self.assertTrue(set(A.REASON_ORDER) <= set(S.ALT_REASONS))


class RulesThatShapeTheAnswer(unittest.TestCase):
    """The rules a property test cannot see: which mover, which reason, how much area, which gaps count."""

    def test_only_a_sliver_within_reach_is_not_an_area(self):
        # A-02 can reach a single grid point of the cover sector (MIN_ZONE_POINTS); A-03 cannot move.
        c = d1_fixed()
        c.p90 = dict({pid: 6.0 for pid in c.own}, **{'A-02': .5, 'A-03': .3})
        c.before = dict(c.own, **{'A-02': (17.6, 34.0)})
        alt = A.propose(*c.files()[:3], profile(4))
        self.assertEqual(alt['reason'], 'no_reachable_fix')
        self.assertEqual({r['mover']: r['reason'] for r in alt['rejected']},
                         {'A-02': 'no_reachable_fix', 'A-03': 'no_reachable_fix'})

    def test_checks_that_leave_a_sliver_are_named(self):
        # A-02 marks B-04: of his 16 restoring points one keeps B-04 marked: the checks, not the reach, deny the area.
        c = d1_fixed()
        c.p90 = dict({pid: 6.0 for pid in c.own}, **{'A-03': .3})
        c.opp = dict(c.opp, **{'B-04': (10.0, 27.0)})
        alt = A.propose(*c.files()[:3], profile(4))
        self.assertEqual(alt['reason'], 'would_open_new_gap')
        self.assertEqual({r['mover']: r['reason'] for r in alt['rejected']},
                         {'A-02': 'would_open_new_gap', 'A-03': 'no_reachable_fix'})

    def test_the_most_informative_reason_is_reported(self):
        c = d1_fixed()
        c.p90 = dict({pid: 6.0 for pid in c.own}, **{'A-02': .3})
        c.before = {pid: p for pid, p in c.own.items() if pid != 'A-03'}
        alt = A.propose(*c.files()[:3], profile(4))
        self.assertEqual(alt['reason'], 'no_reachable_fix')
        self.assertEqual({r['mover']: r['reason'] for r in alt['rejected']},
                         {'A-02': 'no_reachable_fix', 'A-03': 'mover_unseen'})

    def test_the_mover_needing_the_least_of_his_reach_is_chosen(self):
        # A-02 and A-03 stand equally near the cover sector (A-02 first by id); A-03 stood inside it 2 s earlier.
        c = d1_fixed()
        c.before = dict(c.own, **{'A-03': (21.0, 36.0)})
        alt = A.propose(*c.files()[:3], profile(4))
        self.assertEqual(alt['mover'], 'A-03')
        self.assertEqual(alt['rejected'], [])

    def test_a_team_mate_beside_the_area_does_not_shrink_it(self):
        # A-04 stands just outside the cover sector; his 3 m disc cuts into it. The zone avoids the disc but keeps most
        # of the remaining restoring points (the stacking points are refused before the shape is drawn).
        c = d1_fixed()
        c.own = dict(c.own, **{'A-04': (21.0, 45.5)})
        c.before = dict(c.own, **{'A-02': (20.0, 30.0)})
        finding, obs, roles, m, opp = c.files()
        prof = profile(4)
        alt = A.propose(finding, obs, roles, prof)
        self.assertEqual(alt['mover'], 'A-02')
        case = P.Case(alt, finding, obs, roles, prof, 'left')
        bx, by = case.before
        r = alt['reach']['radiusM']
        good = 0
        for i in range(int(bx-r)-1, int(bx+r)+2):
            for j in range(int(by-r)-1, int(by+r)+2):
                q = (float(i), float(j))
                if math.dist(q, (bx, by)) > r-.02 or any(math.dist(q, mq) < A.STACK_M+.02 for mq in case.mates()):
                    continue
                d = math.dist(q, (30.0, 34.0))
                ang = math.degrees(math.acos(max(-1.0, min(1.0, (30.0-q[0])/d)))) if d else 0.0
                if 4.0 <= d <= 12.0 and ang <= 60.0 and q[0] < 30.0 and case.restored(q) and not case.checks_ok(q):
                    good += 1
        self.assertGreater(good, 20)
        self.assertGreaterEqual(alt['zone']['areaM2'], .6*good)
        self.assertEqual(P.zone_problems(alt, finding, obs, roles, prof, 'left'), [])


def d3_fixed():
    """An advanced left back (A-01) without cover; centre-backs A-02 / A-03 wide of the central band (nobody protects
    the middle goal-side of the ball); A-07 has the ball in midfield; everyone else far ahead."""
    c = Case('wide_defender_advance')
    c.own = {'A-01': (55.0, 8.0), 'A-02': (30.0, 23.0), 'A-03': (30.0, 45.0), 'A-04': (34.0, 62.0),
             'A-05': (75.0, 20.0), 'A-06': (75.0, 50.0), 'A-07': (60.0, 34.0), 'A-08': (85.0, 30.0),
             'A-09': (90.0, 45.0), 'A-10': (95.0, 20.0)}
    c.opp = {'B-01': (42.0, 10.0), 'B-02': (70.0, 34.0), 'B-03': (80.0, 10.0), 'B-04': (80.0, 58.0),
             'B-05': (88.0, 34.0), 'B-06': (60.0, 50.0), 'B-07': (64.0, 20.0), 'B-08': (95.0, 25.0),
             'B-09': (95.0, 45.0), 'B-10': (100.0, 34.0)}
    c.ball = (59.6, 34.0)
    c.before = dict(c.own)
    c.p90 = {pid: .3 for pid in c.own}
    c.records = [{'id': pid, 't0': T0-30, 't1': T0+30, 'role': role, 'group': 'D', 'side': None, 'source': 'coach',
                  'confidence': 'high', 'support': None, 'period': None, 'ambiguousWith': [], 'conflict': None}
                 for pid, role in (('A-01', 'LB'), ('A-02', 'LCB'), ('A-03', 'RCB'), ('A-04', 'RB'))]
    c.medians = {'A-02': {'-1': 23.0}, 'A-03': {'-1': 45.0}}
    c.rel = {'advanceM': 25.0, 'beyondHalfM': 2.5, 'channelOccupied': False, 'channelBy': None, 'dmDropped': None,
             'cbAdjusted': False, 'nearCb': 'A-02', 'nearCbShiftM': 0.0, 'behindN': 4, 'interchange': False,
             'opponentsInChannel': 1, 'threatN': None, 'restMargin': None, 'dmCbSpacingM': None, 'wdVelX': .5,
             'complete': True, 'relaxed': False, 'primaryNamed': True, 'wdSource': 'coach', 'wdConfidence': 'high'}
    c.variant = 'left'
    c.actors = {'primary': ['A-01'], 'supporting': ['A-02'], 'opponents': ['B-01']}
    return c


def d3_pushed_up():
    """Team A pushed up behind an attack (ball 75 m, carried by A-07): every outfielder stands 10 m or less behind the
    ball or ahead of it, the left back A-01 is 21 m beyond the centre-backs (67 m), nobody covers his channel, and the
    centre-backs stand central (34 and 44 m): in 2 s neither reaches the left outer third, and without a median
    lateral position (fewer than 25 samples) the centre-back shift test is unknown, so a zone can only restore the
    behind-the-ball count, never the cover."""
    c = Case('wide_defender_advance')
    c.own = {'A-01': (88.0, 8.0), 'A-02': (67.0, 34.0), 'A-03': (67.0, 44.0), 'A-04': (70.0, 60.0),
             'A-05': (80.0, 30.0), 'A-06': (80.0, 50.0), 'A-07': (75.5, 34.0), 'A-08': (90.0, 30.0),
             'A-09': (92.0, 45.0), 'A-10': (95.0, 20.0)}
    c.opp = {'B-01': (60.0, 60.0), 'B-02': (85.0, 34.0), 'B-03': (90.0, 10.0), 'B-04': (90.0, 58.0),
             'B-05': (95.0, 34.0), 'B-06': (80.0, 40.0), 'B-07': (84.0, 20.0), 'B-08': (98.0, 25.0),
             'B-09': (98.0, 45.0), 'B-10': (100.0, 34.0)}
    c.ball = (75.0, 34.0)
    c.before = dict(c.own)
    c.p90 = {pid: 6.0 for pid in c.own}
    c.records = d3_fixed().records
    c.medians = {}
    c.rel = dict(d3_fixed().rel, advanceM=21.0, cbAdjusted=None, beyondHalfM=35.5, behindN=0, opponentsInChannel=0, ballX=75.0)
    c.variant = 'left'
    c.actors = {'primary': ['A-01'], 'supporting': ['A-02'], 'opponents': []}
    return c


def replayed(name, index, sigma_hi):
    """The index-th scene of a random generator replayed from its own seed (1000 + the name's code points), every player
    with a random sigma up to sigma_hi: a fixed scene found by emulating a rule being undone."""
    gen, levels = GENERATORS[name]
    rng = np.random.default_rng(1000+sum(map(ord, name)))
    for i in range(index+1):
        case, prof = gen(rng, levels[i % len(levels)])
        case.sigma = {pid: float(rng.uniform(.3, sigma_hi)) for pid in list(case.own)+list(case.opp)}
    return case, prof


class Replayed(unittest.TestCase):
    """Scenes where undoing one rule changes the answer (found by emulating the mutant, scratch zones_w4; re-found
    Oct 23, scratch zones_fix1/search_survivors.py, after the D1 cover generator moved to pressed carriers)."""

    def check(self, name, index):
        case, prof = replayed(name, index, 3.0)
        finding, obs, roles, m, opp = case.files()
        alt = A.propose(finding, obs, roles, prof)
        self.assertIn('zone', alt)
        self.assertEqual(P.zone_problems(alt, finding, obs, roles, prof, case.side), [])
        return alt

    def test_stacking_points_are_refused_before_the_shape(self):
        # Accepting points within 3 m of a team-mate made the hull and the star fail: no zone where one exists.
        for index in (7, 45, 55):
            with self.subTest(index=index):
                self.check('D1 cover', index)

    def test_a_wide_uncertainty_does_not_stretch_the_zone_out_of_its_region(self):
        # Without the 80% rule a 3 m sigma dilation spreads the zone outside the cover sector / the gap band.
        for name, index in (('D1 cover', 73), ('D1 cover', 147), ('D5', 54)):
            with self.subTest(name=name, index=index):
                self.check(name, index)

    def test_a_pressure_zone_keeps_off_the_carrier(self):
        for index in (10, 14, 15):
            with self.subTest(index=index):
                alt = self.check('D1 pressure', index)
                case, prof = replayed('D1 pressure', index, 3.0)
                poly = [(case.L*(x if case.side == 'left' else 1-x), case.W*(y if case.side == 'left' else 1-y))
                        for x, y in alt['zone']['polygon']]
                self.assertGreaterEqual(P.edge_distance(poly, (case.rel['carrierX'], case.rel['carrierY'])),
                                        A.PRESS_MIN_M-.02)


class WideDefender(unittest.TestCase):

    def test_the_wide_defender_holding_deeper_is_a_fix(self):
        # Only the advanced left back can move: his zone is a deeper spot in his own channel ("delay the run"), here
        # goal-side of the opponent in the channel (B-01 at 46 m).
        c = d3_fixed()
        c.p90 = dict(c.p90, **{'A-01': 7.0})
        c.opp = dict(c.opp, **{'B-01': (46.0, 10.0)})
        finding, obs, roles, m, opp = c.files()
        prof = profile(4)
        alt = A.propose(finding, obs, roles, prof)
        self.assertEqual(alt['mover'], 'A-01')
        self.assertTrue(all(p[1]*68.0 <= 68.0/3+.01 for p in alt['zone']['polygon']))
        self.assertTrue(all(p[0]*105.0 < 30.0+15.0 for p in alt['zone']['polygon']))   # no longer 15 m beyond the CBs
        self.assertEqual(P.zone_problems(alt, finding, obs, roles, prof, 'left'), [])

    def test_no_cover_spot_upfield_of_the_opponent_in_the_channel(self):
        # Critic B-8: B-01 stands in the vacated channel at 42 m. The left back's 12 m reach (age cap 6 m/s) ends at
        # 43 m, so his only goal-side points (x <= 42 + LEVEL_M) are one column at 44 m: a sliver, no zone, and the
        # reason is the reach (no check refused a point), not a new gap. The near centre-back's zone stays level
        # with or goal-side of B-01, also when a wide sigma would dilate it upfield.
        c = d3_fixed()
        c.p90 = dict(c.p90, **{'A-01': 7.0})
        alt = A.propose(*c.files()[:3], profile(4))
        self.assertEqual(alt['reason'], 'no_reachable_fix')
        self.assertEqual({r['mover']: r['reason'] for r in alt['rejected']},
                         {'A-01': 'no_reachable_fix', 'A-02': 'no_reachable_fix', 'A-03': 'no_reachable_fix'})
        for sigma in (.4, 1.5, 3.0):
            c = d3_fixed()
            c.p90 = dict(c.p90, **{'A-02': 7.0})
            c.opp = dict(c.opp, **{'B-01': (36.0, 12.0)})
            c.sigma = {pid: sigma for pid in list(c.own)+list(c.opp)}
            finding, obs, roles, m, opp = c.files()
            prof = profile(4)
            alt = A.propose(finding, obs, roles, prof)
            with self.subTest(sigma=sigma):
                self.assertEqual(alt['mover'], 'A-02')
                self.assertLessEqual(max(p[0]*105.0 for p in alt['zone']['polygon']), 36.0+A.LEVEL_M+.02)
                self.assertEqual(P.zone_problems(alt, finding, obs, roles, prof, 'left'), [])

    def test_a_spot_level_with_the_channel_opponent_still_counts(self):
        # B-01 at 43 m: the left back's reachable channel points (43-45 m) are level with him (within LEVEL_M), a
        # "delay the run" spot; strictly goal-side of him only a sliver would remain.
        c = d3_fixed()
        c.p90 = dict(c.p90, **{'A-01': 7.0})
        c.opp = dict(c.opp, **{'B-01': (43.0, 10.0)})
        finding, obs, roles, m, opp = c.files()
        prof = profile(4)
        alt = A.propose(finding, obs, roles, prof)
        self.assertEqual(alt['mover'], 'A-01')
        self.assertGreater(max(p[0]*105.0 for p in alt['zone']['polygon']), 43.0+.5)
        self.assertEqual(P.zone_problems(alt, finding, obs, roles, prof, 'left'), [])

    def test_an_opponent_deeper_than_the_whole_channel_leaves_the_principle_only(self):
        # B-01 at 20 m is deeper than the vacated channel starts (centre-back line 30 m - 5 m): no spot of the channel
        # is goal-side of him, whoever moves; the finding keeps the principle (no_reachable_fix).
        c = d3_fixed()
        c.p90 = {pid: 7.0 for pid in c.own}
        c.opp = dict(c.opp, **{'B-01': (20.0, 10.0)})
        alt = A.propose(*c.files()[:3], profile(4))
        self.assertEqual((alt['reason'], alt['principle']), ('no_reachable_fix', 'cover_behind_advancing_defender'))
        self.assertTrue(alt['rejected'])
        self.assertEqual({r['reason'] for r in alt['rejected']}, {'no_reachable_fix'})

    def test_a_channel_opponent_the_scene_does_not_show_cannot_be_placed_against(self):
        c = d3_fixed()
        c.p90 = dict(c.p90, **{'A-02': 7.0})
        c.opp = {pid: p for pid, p in c.opp.items() if pid != 'B-01'}     # measured (opponentsInChannel 1), not seen
        self.assertEqual(A.propose(*c.files()[:3], profile(4))['reason'], 'context_missing')
        c.rel = dict(c.rel, opponentsInChannel=0)                           # nobody in the channel: no limit
        self.assertIn('zone', A.propose(*c.files()[:3], profile(4)))

    def test_level_two_someone_back_is_a_spot_behind_the_ball(self):
        # Level 2 (teamLevelOnly, requireNobodyBehind): the whole team stands 10 m or less behind the ball (75 m)
        # while the left back is far upfield. The principle is "someone stays back", so the zone lies 10 m goal-side
        # of the ball in the central two-thirds and puts one player there (FACTS M6: the channel zone lay ahead of the
        # ball and left the behind count at 0); the restored relation is behind_ball.
        prof = profile(2)
        c = d3_pushed_up()
        self.assertEqual(failing(c, prof)[0], True)
        finding, obs, roles, m, opp = c.files()
        alt = A.propose(finding, obs, roles, prof)
        self.assertEqual((alt['principle'], alt['improves']['relation']), ('someone_back', 'behind_ball'))
        self.assertIn(alt['mover'], ('A-02', 'A-03'))
        self.assertTrue(all(p[0]*105.0 <= 75.0-10.0+.02 for p in alt['zone']['polygon']))
        self.assertEqual(P.zone_problems(alt, finding, obs, roles, prof, 'left'), [])
        # an opponent waiting in the vacated channel at 50 m: the spot must also be level with or goal-side of him,
        # which nobody reaches in 2 s
        c.opp = dict(c.opp, **{'B-01': (50.0, 12.0)})
        c.rel = dict(c.rel, opponentsInChannel=1)
        alt = A.propose(*c.files()[:3], prof)
        self.assertEqual((alt['reason'], alt['principle']), ('no_reachable_fix', 'someone_back'))

    def test_level_one_reframing_judges_the_scene_with_the_detectors_ball_depth(self):
        # Level 1 judges the advance with the behind_ball rule (as assess reframes it). When the detector's relations
        # carry the ball depth, so do the re-measured ones: the same final-third gate then decides on the scene.
        prof = profile(1)
        c = d3_pushed_up()
        c.observation = 'behind_ball'
        self.assertEqual(failing(c, prof)[0], True)
        finding, obs, roles, m, opp = c.files()
        alt = A.propose(finding, obs, roles, prof)
        self.assertEqual((alt['principle'], alt['improves']['relation']), ('someone_back', 'behind_ball'))
        self.assertEqual(P.zone_problems(alt, finding, obs, roles, prof, 'left'), [])
        c.ball = (60.0, 34.0)                       # the scene's ball outside the final third the stored sample was in
        c.own = dict(c.own, **{'A-07': (60.5, 34.0)})
        self.assertEqual(A.propose(*c.files()[:3], prof)['reason'], 'context_missing')

    def test_central_protection_that_never_existed_is_not_required(self):
        # Nobody protected the middle goal-side of the ball before the move (centre-backs at 23 and 45 m), so the near
        # centre-back moving into the channel opens no NEW central gap.
        c = d3_fixed()
        c.p90 = dict(c.p90, **{'A-02': 7.0})
        finding, obs, roles, m, opp = c.files()
        prof = profile(4)
        alt = A.propose(finding, obs, roles, prof)
        self.assertEqual(alt['mover'], 'A-02')
        self.assertEqual(alt['improves']['relation'], 'channel_cover')
        self.assertEqual(P.zone_problems(alt, finding, obs, roles, prof, 'left'), [])


class OtherDetectors(unittest.TestCase):

    def test_the_only_presser_is_not_sent_to_the_far_side(self):
        # D2: A-01 is the only player within 5 m of the carrier; A-03 still marks the carrier from 9.6 m. Moving A-01
        # to the far side would restore the far-side defender but take the pressure off the ball (check i).
        c = Case('weak_side_balance')
        c.own = {'A-01': (30.0, 14.0), 'A-03': (38.0, 16.0)}
        c.own.update({f'A-{k:02d}': (60.0+3*k, 6.0+6*k) for k in (2, 4, 5, 6, 7, 8, 9, 10)})
        c.opp = {'B-01': (30.5, 10.0)}
        c.opp.update({f'B-{k:02d}': (70.0+2*k, 5.0+6*k) for k in range(2, 11)})
        c.ball = (30.5, 10.0)
        c.before = dict(c.own, **{'A-01': (30.0, 24.0)})
        c.p90 = {pid: 7.0 for pid in c.own}
        c.rel = {'ballSide': 'left', 'weakCount': 0, 'openWidthM': 34.0, 'centralCount': 0, 'nearestWeak': 'A-03',
                 'primaryNamed': True, 'ballX': 30.5, 'ballY': 10.0}
        c.actors = {'primary': ['A-03'], 'opponents': ['B-01'], 'carrier': 'B-01'}
        alt = A.propose(*c.files()[:3], profile(4))
        self.assertEqual(alt['reason'], 'would_open_new_gap')
        self.assertEqual({r['mover']: r['reason'] for r in alt['rejected']},
                         {'A-01': 'would_open_new_gap', 'A-03': 'no_reachable_fix'})

    def test_closing_a_gap_does_not_need_an_even_line(self):
        # D5: an uneven back line (depth std about 4.5 m) with a 28 m ball-side gap and an opponent in it: moving the
        # ball-side defender closes the gap; the line stays too uneven for the strength band, which is not asked.
        c = Case('defensive_spacing')
        c.own = {'A-01': (12.0, 8.0), 'A-02': (22.0, 20.0), 'A-03': (15.0, 30.0), 'A-04': (24.0, 58.0)}
        c.own.update({f'A-{k:02d}': (50.0+4*k, 6.0+6*k) for k in range(5, 11)})
        c.opp = {'B-02': (21.0, 45.0)}
        c.opp.update({f'B-{k:02d}': (70.0+2*k, 5.0+6*k) for k in (1, 3, 4, 5, 6, 7, 8, 9, 10)})
        c.ball = (50.0, 55.0)
        c.before = dict(c.own)
        c.p90 = {pid: 7.0 for pid in c.own}
        X = [c.own[p][0] for p in ('A-01', 'A-02', 'A-03', 'A-04')]
        c.rel = {'lineStdM': float(np.std(X)), 'lineSize': 4, 'lateralGapM': 28.0, 'gapLeft': 'A-03',
                 'gapRight': 'A-04', 'gapY': 44.0, 'gapBallSide': True, 'opponentInGap': True, 'gapOpponents': 1,
                 'ballSide': 'right', 'dmGapM': 11.0, 'dmGapState': '', 'lineHeightM': float(np.mean(X)),
                 'blockHeight': 'low', 'primaryNamed': True}
        c.actors = {'primary': ['A-04'], 'supporting': ['A-03']}
        c.mode = 'lateral_gap'
        finding, obs, roles, m, opp = c.files()
        prof = profile(5)
        alt = A.propose(finding, obs, roles, prof)
        self.assertIn(alt['mover'], ('A-03', 'A-04'))
        self.assertEqual(alt['improves']['relation'], 'lateral_gap')
        self.assertEqual(P.zone_problems(alt, finding, obs, roles, prof, 'left'), [])


class BehindTheBall(unittest.TestCase):

    def test_the_player_on_the_ball_is_not_sent_back(self):
        # D4: A-02 carries the ball and is the deepest of those not behind it; the others are 15 m or more from the
        # area behind the ball: only A-02 could have reached it, and moving him would move the ball.
        c = Case('behind_ball')
        c.own = {'A-01': (20.0, 34.0), 'A-02': (79.6, 34.0)}
        c.own.update({f'A-{k:02d}': (85.0+k, 4.0+6*k) for k in range(3, 11)})
        c.opp = {f'B-{k:02d}': (100.0, 3.0+6*k) for k in range(1, 11)}
        c.ball = (80.0, 34.0)
        c.before = dict(c.own)
        c.p90 = {pid: 7.0 for pid in c.own}
        c.rel = {'ballX': 80.0, 'behindN': 1, 'threatN': 0, 'restMargin': 1, 'behindSpreadM': None, 'deepest': 'A-01',
                 'behindIds': 'A-01', 'oppComplete': True}
        c.actors = {'primary': ['A-01']}
        alt = A.propose(*c.files()[:3], profile(4))
        self.assertEqual(alt['reason'], 'no_reachable_fix')
        self.assertNotIn('A-02', [r['mover'] for r in alt['rejected']])

    def test_a_very_uncertain_position_never_stretches_the_zone_ahead_of_the_ball(self):
        # A 12 m position sigma would dilate the area (10 m behind the ball) past the ball; the zone falls back to a
        # smaller dilation and stays behind it.
        c = Case('behind_ball')
        c.own = {'A-01': (20.0, 34.0), 'A-02': (76.0, 34.0)}
        c.own.update({f'A-{k:02d}': (85.0+k, 4.0+6*k) for k in range(3, 11)})
        c.opp = {f'B-{k:02d}': (100.0, 3.0+6*k) for k in range(1, 11)}
        c.ball = (80.0, 20.0)
        c.before = dict(c.own)
        c.p90 = {pid: 7.0 for pid in c.own}
        c.sigma = {pid: 12.0 for pid in list(c.own)+list(c.opp)}
        c.rel = {'ballX': 80.0, 'behindN': 1, 'threatN': 0, 'restMargin': 1, 'behindSpreadM': None, 'deepest': 'A-01',
                 'behindIds': 'A-01', 'oppComplete': True}
        c.actors = {'primary': ['A-01']}
        finding, obs, roles, m, opp = c.files()
        prof = profile(4)
        alt = A.propose(finding, obs, roles, prof)
        self.assertEqual(alt['mover'], 'A-02')
        self.assertLessEqual(max(p[0]*105.0 for p in alt['zone']['polygon']), 80.0+.02)
        self.assertEqual(P.zone_problems(alt, finding, obs, roles, prof, 'left'), [])


class LevelsAndDisplay(unittest.TestCase):
    """The level's parameters change the reach and the display, never the measured problem."""

    def test_the_age_cap_shrinks_the_reach(self):
        r = {}
        for level in (2, 4, 6):
            c = d1_fixed()
            c.p90 = {pid: 8.0 for pid in c.own}
            c.before = dict(c.own, **{'A-02': (17.0, 24.0), 'A-03': (17.0, 44.0)})
            finding, obs, roles, m, opp = c.files()
            alt = A.propose(finding, obs, roles, profile(level))
            r[level] = alt['reach']
            self.assertEqual(alt['evidence'], m['id'])        # the same moment at every level
        self.assertEqual([r[lv]['speedMS'] for lv in (2, 4, 6)], [5.0, 6.0, 7.0])
        self.assertEqual([r[lv]['radiusM'] for lv in (2, 4, 6)], [10.0, 12.0, 14.0])

    def tiny(self, level, tier='world', sigma=.3):
        """A slow mover whose whole 1.6 m reach lies inside the cover sector: a zone of about 2 x 2 m."""
        c = d1_fixed(tier=tier)
        c.sigma = {pid: sigma for pid in list(c.own)+list(c.opp)}
        c.p90 = dict({pid: 6.0 for pid in c.own}, **{'A-02': .8})
        c.before = dict(c.own, **{'A-02': (22.0, 34.0)})
        finding, obs, roles, m, opp = c.files()
        return A.propose(finding, obs, roles, profile(level)), c, finding, obs, roles

    def test_a_small_precise_zone_is_a_point_only_from_level_four_on_a_world_frame(self):
        alt, c, finding, obs, roles = self.tiny(4)
        self.assertEqual(alt['mover'], 'A-02')
        self.assertEqual((alt['display'], alt['zone']['kind']), ('point', 'point'))
        self.assertGreaterEqual(alt['zone']['radiusM'], A.POINT_MIN_RADIUS_M)
        self.assertEqual(P.zone_problems(alt, finding, obs, roles, profile(4), 'left'), [])
        self.assertEqual(self.tiny(3)[0]['display'], 'zone')                # level 3: no points
        self.assertEqual(self.tiny(4, tier='pinned')[0]['display'], 'zone')  # not a world frame
        self.assertEqual(self.tiny(4, sigma=1.5)[0]['display'], 'zone')      # positions too uncertain
        self.assertEqual(self.tiny(4, sigma=None)[0]['display'], 'zone')     # precision unknown

    def test_a_zone_larger_than_a_quarter_of_the_half_shows_a_direction(self):
        c = d1_fixed()
        c.observation = c.detector = 'behind_ball'
        c.own = {'A-01': (20.0, 34.0), 'A-02': (84.0, 34.0)}
        c.own.update({f'A-{k:02d}': (85.0+k, 4.0+6*k) for k in range(3, 11)})
        c.opp = {f'B-{k:02d}': (100.0, 3.0+6*k) for k in range(1, 11)}
        c.ball = (92.0, 30.0)
        c.horizon = 5.0
        c.p90 = {pid: 7.0 for pid in c.own}
        c.before = dict(c.own)
        c.rel = {'ballX': 92.0, 'behindN': 1, 'threatN': 1, 'restMargin': 0, 'behindSpreadM': None, 'deepest': 'A-01',
                 'behindIds': 'A-01', 'oppComplete': True}
        c.actors = {'primary': ['A-01']}
        c.mode = None
        finding, obs, roles, m, opp = c.files()
        prof = profile(6, behind_ball={'reachHorizonS': 5.0})
        alt = A.propose(finding, obs, roles, prof)
        self.assertEqual(alt['display'], 'direction')
        self.assertGreater(alt['zone']['areaM2'], A.ZONE_MAX_HALF_SHARE*105*68/2)
        self.assertEqual(alt['reach']['horizonS'], 5.0)
        c.horizon = 2.0
        finding, obs, roles, m, opp = c.files()
        alt = A.propose(finding, obs, roles, profile(6))
        self.assertEqual(alt['display'], 'zone')        # a 14 m reach: an area to draw
        self.assertEqual(P.zone_problems(alt, finding, obs, roles, prof, 'left'), [])

    def test_the_side_defended_turns_the_zone_with_the_scene(self):
        left = A.propose(*d1_fixed().files()[:3], profile(4))
        finding, obs, roles, m, opp = d1_fixed(side='right').files()
        right = A.propose(finding, obs, roles, profile(4))
        self.assertEqual(left['mover'], right['mover'])
        self.assertEqual(len(left['zone']['polygon']), len(right['zone']['polygon']))
        for a, b in zip(left['zone']['polygon'], right['zone']['polygon']):
            self.assertAlmostEqual(a[0], 1-b[0], places=3)
            self.assertAlmostEqual(a[1], 1-b[1], places=3)


class AttackSide(unittest.TestCase):

    def test_the_moment_then_the_stored_sample_decides(self):
        finding, obs, roles, m, opp = d1_fixed(side='right').files()
        self.assertEqual(A.attack_side(obs, dict(m, calibration={'side': 'left'})), 'left')   # EVIDENCE's side first
        self.assertEqual(A.attack_side(obs, m), 'right')
        obs['roleInputs']['detect']['samples']['t'] = [T0+5.0]
        self.assertIsNone(A.attack_side(obs, m))            # no stored sample, no track

    def test_the_context_track_decides_otherwise(self):
        for side in ('left', 'right'):
            c = d1_fixed(side=side)
            finding, obs, roles, m, opp = c.files()
            ids = sorted(c.own)
            obs['roleInputs'] = {'context': {'ids': ids, 'scale': obs['timeline']['scale'],
                                             'track': {'t': [T0-1, T0, T0+1], 'step': 1.0,
                                                       'x': [[c.own[p][0]+.5]*3 for p in ids],
                                                       'y': [[c.own[p][1]-.5]*3 for p in ids]}}}
            with self.subTest(side=side):
                self.assertEqual(A.attack_side(obs, m), side)
        crowd = {'ids': ['A-01', 'A-02'], 'scale': obs['timeline']['scale'],
                 'track': {'t': [T0], 'step': 1.0, 'x': [[52.5], [52.5]], 'y': [[34.0], [34.0]]}}
        m2 = copy.deepcopy(m)
        m2['scene']['players'] = {'A-01': dict(m['scene']['players']['A-01'], x=.5, y=.5),
                                  'A-02': dict(m['scene']['players']['A-02'], x=.5, y=.5)}
        self.assertIsNone(A.attack_side({'team': 'A', 'roleInputs': {'context': crowd}}, m2))   # centre spot: no tell


class PolygonHelpers(unittest.TestCase):

    def test_hull_area_centroid_inside(self):
        hull = A.convex_hull([[0, 0], [2, 0], [2, 2], [0, 2], [1, 1], [1, 0]])
        self.assertEqual(len(hull), 4)
        self.assertAlmostEqual(A.polygon_area(hull), 4.0)
        self.assertEqual(A.polygon_centroid(hull), [1.0, 1.0])
        self.assertAlmostEqual(A.polygon_diameter(hull), math.sqrt(8))
        inside = A.points_in_polygon(hull, [1, 2, 3, 0], [1, 1, 1, 2])
        self.assertEqual(inside.tolist(), [True, True, False, True])     # boundary points count as inside

    def test_simplifying_a_convex_polygon_keeps_it_inside(self):
        circle = [[10*math.cos(a/40*2*math.pi), 10*math.sin(a/40*2*math.pi)] for a in range(40)]
        small = A._simplify(circle)
        self.assertLessEqual(len(small), A.MAX_VERTICES)
        self.assertTrue(all(A.points_in_polygon(circle, [p[0]], [p[1]])[0] for p in small))
        self.assertGreater(A.polygon_area(small), .85*A.polygon_area(circle))


# ---------------------------------------------------------------------------------------------------------------
class SyntheticScenes(unittest.TestCase):
    """DESIGN 13.1 rows through the real measurement and role tiers (EVAL's scenes; team A defends the left goal)."""

    def alts(self, scene, variant=None, levels=range(1, 8), tier='world', detector=None):
        obs, roles, tl = Z.setup(scene, variant, tier=tier)
        out = []
        for level in levels:
            prof = F.profile(scene, variant, level=level)
            for f in Z.findings(obs, roles, prof):
                if detector and f['observation'] != detector:
                    continue
                out.append((level, f, A.propose(f, obs, roles, prof), obs, roles, prof))
        return out

    def test_unsupported_advances_draw_no_zone_upfield_of_the_opponent_in_the_channel(self):
        # Critic B-8 (COACH B2): the planted opponent B-10 stands in the vacated channel at about 31 m, deeper than
        # every team-A outfielder; the old channel zones (36-45 m) lay 5-11 m upfield of him. No spot level with or
        # goal-side of him is reachable in 2 s, so every level keeps the principle only (no_reachable_fix), from the
        # same moment, and level 2's "someone stays back" is judged behind the ball, not in the channel.
        for variant in ('none_behind', 'one_behind'):
            rows = self.alts('unsupported_advance', variant, detector='wide_defender_advance')
            self.assertTrue(rows)
            evidence = set()
            for level, f, alt, obs, roles, prof in rows:
                with self.subTest(variant=variant, level=level):
                    self.assertNotIn('zone', alt)
                    self.assertEqual(alt['reason'], 'no_reachable_fix')
                    self.assertEqual(alt['principle'], 'someone_back' if level <= 2 else
                                     'cover_behind_advancing_defender')
                    self.assertEqual({r['reason'] for r in alt['rejected']}, {'no_reachable_fix'})
                    self.assertIn(alt['evidence'], [ts['evidence'] for ts in f['timestamps']])
                    evidence.add(alt['evidence'])
            self.assertEqual(len(evidence), 1)          # the same moment at every level

    def test_the_zone_never_sits_on_a_team_mate(self):
        rows = self.alts('stacking', levels=(4, 5, 6, 7), detector='wide_defender_advance')
        self.assertTrue(rows)
        for level, f, alt, obs, roles, prof in rows:
            if 'zone' not in alt:
                continue
            c = P.Case(alt, f, obs, roles, prof, 'left')
            with self.subTest(level=level):
                for q in c.mates():
                    self.assertFalse(P.inside(c.poly, q))
                    self.assertGreaterEqual(P.edge_distance(c.poly, q), A.STACK_M-.02)

    def test_identity_crossings_and_single_instances_draw_nothing(self):
        # Fix round 3 (critic B-1): level 2 judges an advance left exposed (an opponent in the vacated channel) wherever
        # the ball is, so the crossing advances (36-42 m) are judged at level 2 too, and there, as at levels 3-7, the
        # identity guardrail fails at the moment: a review, never a work item, and no area drawn for anyone. One
        # advance is below the evidence floor (critic S-1: fewer than reviewFloorShare x minOpportunities), so the
        # check is not assessed and nothing is proposed.
        rows = self.alts('identity_crossing', levels=(2, 3, 4, 5, 6, 7), detector='wide_defender_advance')
        self.assertEqual(sorted({r[0] for r in rows}), [2, 3, 4, 5, 6, 7])
        for level, f, alt, *_ in rows:
            with self.subTest(scene='identity_crossing', level=level):
                self.assertEqual(f['verdict'], 'review')
                self.assertEqual(f['guardrails']['identity'], 'fail')
                self.assertEqual(alt['reason'], 'guardrail_failed')
                self.assertNotIn('zone', alt)
        self.assertFalse(self.alts('single_advance', levels=(2, 3, 4, 5, 6, 7), detector='wide_defender_advance'))

    def test_covered_advances_are_never_criticised_so_nothing_is_proposed(self):
        for variant in ('cb_shift', 'channel'):
            self.assertEqual(self.alts('covered_advance', variant, detector='wide_defender_advance'), [])

    def test_a_pinned_frame_still_gets_the_diagram_zone_but_never_a_point(self):
        # The defending scenes' zones (D1 cover, D2 far side) on pinned frames: a diagram zone, never a point. Two
        # scenes, so the check survives a rule change that stops one of them being criticised.
        zones = 0
        for variant, det in (('press_no_cover', 'pressure_cover'), ('weak_empty', 'weak_side_balance')):
            for level, f, alt, *_ in self.alts('defending', variant, levels=(4, 5), tier='pinned', detector=det):
                if 'zone' in alt:
                    zones += 1
                    self.assertNotEqual(alt['display'], 'point')
        self.assertGreater(zones, 0)

    def test_pressure_without_cover_moves_a_team_mate_not_the_first_defender(self):
        rows = self.alts('defending', 'press_no_cover', levels=(2, 4, 6), detector='pressure_cover')
        self.assertTrue(rows)
        for level, f, alt, obs, roles, prof in rows:
            with self.subTest(level=level):
                self.assertIn('zone', alt)
                c = P.Case(alt, f, obs, roles, prof, 'left')
                self.assertNotEqual(alt['mover'], c.rel['fd'])
                self.assertEqual(alt['improves']['relation'], 'cover')
                self.assertEqual(P.zone_problems(alt, f, obs, roles, prof, 'left'), [])

    def test_nobody_going_to_the_ball_at_level_one_gets_a_pressure_area(self):
        rows = self.alts('defending', 'no_press', levels=(1,), detector='pressure_cover')
        self.assertTrue(rows)
        level, f, alt, obs, roles, prof = rows[0]
        self.assertEqual(f['verdict'], 'try_this')
        self.assertEqual((alt['principle'], alt['improves']['relation']), ('pressure_on_ball', 'pressure'))
        self.assertEqual(P.zone_problems(alt, f, obs, roles, prof, 'left'), [])

    def test_attacking_shape_findings_get_no_zone(self):
        # Every attacking-shape finding of the scenes, whatever its verdict (D6 is described at most levels now).
        rows = []
        for scene in ('kids_swarm', 'professional'):
            obs, roles, tl = Z.setup(scene, None)
            for level in range(1, 8):
                prof = F.profile(scene, None, level=level)
                for f in Z.findings(obs, roles, prof, verdicts=S.FINDING_VERDICTS):
                    if f['observation'] == 'attacking_shape':
                        rows.append(A.propose(f, obs, roles, prof))
        self.assertTrue(rows)
        self.assertTrue(all(alt is None for alt in rows))

    def test_a_role_tier_scene_from_the_obs_file_alone_still_validates(self):
        obs, roles, tl = Z.setup('unsupported_advance', 'none_behind')
        roles = copy.deepcopy(roles)
        roles['moments'] = [Z.role_tier_scene(obs, m) for m in roles['moments']]
        prof = F.profile('unsupported_advance', 'none_behind', level=4)
        f = next(f for f in Z.findings(obs, roles, prof) if f['observation'] == 'wide_defender_advance')
        alt = A.propose(f, obs, roles, prof)
        if 'zone' in alt:
            self.assertEqual(P.zone_problems(alt, f, obs, roles, prof, 'left'), [])
        else:
            self.assertEqual(S.validate('no_alternative', alt), [])

    def test_the_orchestrator_takes_the_records(self):
        # coaching.interpret with stand-ins for the later stages: the zone reaches the report and the finding's
        # suggested action; a NoAlternative turns it into a principle; attacking shape keeps assess's principle.
        from soccer import coaching
        drawn = 0
        for scene, variant in (('defending', 'weak_empty'), ('unsupported_advance', 'none_behind')):
            obs, roles, tl = Z.setup(scene, variant)
            captured = {}

            def report(**kw):
                captured.update(kw)
                return {}
            stages = {'annotate': lambda findings, obs, roles, profile, alternatives: [],
                      'explain': lambda findings, profile, **kw: {'findings': findings, 'withheld': []},
                      'drills': lambda findings, profile, priorities, **kw: [], 'report': report}
            st = copy.deepcopy(F.prepared(scene, variant)[1])
            st['teams']['A']['playingLevel'], st['teams']['A']['ageGroup'] = 4, 'U14'
            coaching.interpret(obs, roles, st, 'A', stages=stages, validate=False)
            got = [f for f in captured['findings'] if f['verdict'] in ('work_on', 'try_this', 'review') and
                   f['observation'] in A.ZONE_DETECTORS]
            alt_ids = {a['id'] for a in captured['alternatives']}
            none = {n['finding']: n for n in captured['no_alternatives']}
            with self.subTest(scene=scene, variant=variant):
                self.assertTrue(got)
                for f in got:
                    self.assertEqual(f['guardrails']['alternative'], 'pass')
                    sa = f['suggestedAction']
                    if sa['mode'] == 'principle':
                        self.assertEqual(sa['alternatives'], [])
                        self.assertIn(f['id'], none)
                    else:
                        drawn += 1
                        self.assertIn(sa['mode'], ('zone', 'point', 'direction'))
                        self.assertTrue(sa['alternatives'] and set(sa['alternatives']) <= alt_ids)
                    if f['observation'] == 'wide_defender_advance':     # B-8: the channel opponent is goal-side of
                        self.assertEqual(sa['mode'], 'principle')        # every reachable spot
                        self.assertEqual(none[f['id']]['reason'], 'no_reachable_fix')
                for a in captured['alternatives']:
                    self.assertEqual(S.validate('alternative', a), [])
                for n in captured['no_alternatives']:
                    self.assertEqual(S.validate('no_alternative', n), [])
        self.assertGreater(drawn, 0)        # a zone reached the report and a suggested action

    def test_deterministic_and_inputs_untouched(self):
        obs, roles, tl = Z.setup('unsupported_advance', 'none_behind')
        prof = F.profile('unsupported_advance', 'none_behind', level=4)
        f = next(f for f in Z.findings(obs, roles, prof) if f['observation'] == 'wide_defender_advance')
        before = copy.deepcopy((f, obs, roles, prof))
        a = A.propose(f, obs, roles, prof)
        self.assertEqual((f, obs, roles, prof), before)
        roles2 = copy.deepcopy(roles)
        for m in roles2['moments']:
            m['scene']['players'] = dict(reversed(list(m['scene']['players'].items())))
        self.assertEqual(A.propose(f, obs, roles2, prof), a)


class EvidenceMoments(unittest.TestCase):
    """The same through EVIDENCE's real stages (evidence.moments / moments_from_obs): the moments it builds carry what
    propose reads (both teams, keepers, ball, our players 2 s earlier, calibration['side'])."""

    @classmethod
    def setUpClass(cls):
        from soccer import coaching
        cls.files = {}
        for scene, variant in (('unsupported_advance', 'none_behind'), ('stacking', None),
                               ('defending', 'press_no_cover'), ('defending', 'weak_empty')):
            b, st, cd, poss = F.prepared(scene, variant)
            obs = coaching.build_observations(b.result, 'A', coaching=st, series=cd, possession=poss,
                                              result_sha='0'*64)
            cls.files[(scene, variant)] = (obs, coaching.build_roles(obs, st, 'A'), variant)

    def test_zones_on_real_moments_hold_the_properties(self):
        # The two advance scenes plant the channel opponent deeper than any reachable spot (B-8): their moments give
        # the principle only (no_reachable_fix); the defending scenes draw their cover / far-side zones (at least one
        # of the two, so a rule change that stops one being criticised does not hide the check).
        drawn = 0
        for scene, variant, det, want in (('unsupported_advance', 'none_behind', 'wide_defender_advance',
                                           'no_reachable_fix'),
                                          ('stacking', None, 'wide_defender_advance', 'no_reachable_fix'),
                                          ('defending', 'press_no_cover', 'pressure_cover', 'zone'),
                                          ('defending', 'weak_empty', 'weak_side_balance', 'zone')):
            obs, roles, variant = self.files[(scene, variant)]
            n = 0
            for level in (2, 4, 6):
                prof = F.profile(scene, variant, level=level)
                for f in Z.findings(obs, roles, prof):
                    if f['observation'] != det:
                        continue
                    alt = A.propose(f, obs, roles, prof)
                    m = next(m for m in obs['moments']+roles['moments'] if m['id'] == alt['evidence'])
                    with self.subTest(scene=scene, level=level):
                        self.assertEqual(A.attack_side(obs, m), 'left')     # scene truth: A defends left
                        if want == 'zone':
                            if 'zone' in alt:
                                self.assertEqual(P.zone_problems(alt, f, obs, roles, prof, 'left'), [])
                                drawn += 1
                        else:
                            self.assertEqual(alt['reason'], want)
                        n += 1
            if want != 'zone':
                self.assertGreater(n, 0, scene)
        self.assertGreater(drawn, 0)


class Style(unittest.TestCase):

    def test_ascii_crlf(self):
        for path in (ROOT/'soccer/coaching/alternatives.py', Path(__file__), Path(__file__).with_name(
                'coaching_zone_fixtures.py'), Path(__file__).with_name('coaching_zone_props.py')):
            data = path.read_bytes()
            with self.subTest(path=path.name):
                data.decode('ascii')
                self.assertEqual(data.count(b'\n'), data.count(b'\r\n'))

    def test_principles_are_worded_by_text_and_thresholds_are_proposals(self):
        self.assertTrue(set(A.PRINCIPLE.values()) | {'pressure_on_ball'} <= set(E.PRINCIPLES))
        self.assertIn('PITCHIQ-PROPOSAL', A.__doc__)
        self.assertNotIn('attacking_shape', A.ZONE_DETECTORS)


if __name__ == '__main__':
    unittest.main()
