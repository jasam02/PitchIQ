"""Synthetic matches for the key-moments tests (soccer/moments): a MatchData built directly from arrays, no views.

    s = Scene(seconds=80, attacks_right='A')
    a = s.add_team('A')                 # 4-4-2 at its defensive positions: {'GK', 'LB', 'LCB', ..., 'RS'} -> person index
    b = s.add_team('B')
    s.move(a['LB'], 50, 56, 55, 26)     # team coordinates of the person's own team: depth from own goal, metres to the left
    s.possession('B', 10, 50, carrier=b['RCM'])
    s.ball_with(b['RCM'], 10, 50)
    data = s.build()

Positions are static unless moved (a linear move between two times, then held); velocities are their 5 Hz gradient.
The camera sees the whole pitch and every frame is calibrated unless set otherwise (s.view, s.calibrated).
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from soccer.moments import data as md  # noqa: E402

L, W = 105.0, 68.0
FPS = 25.0
DT = .2
T0 = 10.0
F442 = {'GK': (4.0, 0.0), 'LB': (32.0, 24.0), 'LCB': (30.0, 8.0), 'RCB': (30.0, -8.0), 'RB': (32.0, -24.0),
        'LM': (46.0, 22.0), 'LCM': (44.0, 7.0), 'RCM': (44.0, -7.0), 'RM': (46.0, -22.0),
        'LS': (60.0, 7.0), 'RS': (60.0, -7.0)}
FULL_VIEW = [[-5.0, -5.0], [52.5, -5.0], [110.0, -5.0], [110.0, 73.0], [52.5, 73.0], [-5.0, 73.0]]


class Scene:
    def __init__(self, seconds=80.0, attacks_right='A'):
        self.n = int(round(seconds/DT))
        self.t = T0+np.arange(self.n)*DT
        self.attacks_right = attacks_right
        self.people, self.pos = [], []
        self.spans, self.changes = [], []
        self.ball = np.full((self.n, 2), np.nan)
        self.ball_state = np.zeros(self.n, np.int8)
        self.calibrated = np.ones(self.n, bool)
        self.view = np.tile(np.array(FULL_VIEW, np.float32), (self.n, 1, 1))
        self.counts = {'A': 0, 'B': 0}

    # ------------------------------------------------------------ coordinates
    def sign(self, team):
        return 1 if self.attacks_right == team else -1

    def pitch(self, team, depth, lateral):
        d = self.sign(team)
        x = depth if d > 0 else L-depth
        return x, W/2-lateral/d

    def k(self, t):
        return int(np.clip(np.searchsorted(self.t, t-1e-9), 0, self.n))

    # ------------------------------------------------------------ people
    def add(self, team, depth, lateral, keeper=False, name=None):
        self.counts[team] += 1
        pid = len(self.people)
        label = name or (f'GK-{team}{self.counts[team]}' if keeper else f'{team}-{self.counts[team]:02d}')
        self.people.append({'id': label, 'display': label, 'team': team, 'role': 'GOALKEEPER' if keeper else 'PLAYER'})
        x, y = self.pitch(team, depth, lateral)
        self.pos.append(np.tile([x, y], (self.n, 1)).astype(float))
        return pid

    def add_team(self, team, formation=F442):
        return {role: self.add(team, d, l, keeper=(role == 'GK')) for role, (d, l) in formation.items()}

    def team_of(self, pid):
        return self.people[pid]['team']

    def move(self, pid, t0, t1, depth, lateral):
        """Move person pid linearly from where they are at t0 to (depth, lateral) of their own team at t1, then hold."""
        a, b = self.k(t0), self.k(t1)
        x1, y1 = self.pitch(self.team_of(pid), depth, lateral)
        x0, y0 = self.pos[pid][a]
        w = np.linspace(0, 1, max(1, b-a)+1)[:, None]
        seg = np.array([x0, y0])*(1-w)+np.array([x1, y1])*w
        self.pos[pid][a:b+1] = seg[:len(self.pos[pid][a:b+1])]
        self.pos[pid][b:] = [x1, y1]

    def place(self, pid, t0, depth, lateral):
        """Put person pid at (depth, lateral) from t0 on (a jump)."""
        x, y = self.pitch(self.team_of(pid), depth, lateral)
        self.pos[pid][self.k(t0):] = [x, y]

    def hide(self, pid, t0, t1):
        self.pos[pid][self.k(t0):self.k(t1)] = np.nan

    # ------------------------------------------------------------ ball, possession, camera
    def ball_with(self, pid, t0, t1, offset=(.6, 0.0)):
        a, b = self.k(t0), self.k(t1)
        self.ball[a:b] = self.pos[pid][a:b]+np.array(offset)
        self.ball_state[a:b] = 1

    def possession(self, team, t0, t1, carrier=None, state='control'):
        self.spans.append({'t0': float(t0), 't1': float(t1), 'team': team, 'state': state,
                           'carrier': self.people[carrier]['id'] if carrier is not None else None,
                           'confidence': 'medium', 'estimated': False})

    def dead(self, t0, t1):
        self.spans.append({'t0': float(t0), 't1': float(t1), 'team': None, 'state': 'dead', 'carrier': None,
                           'confidence': 'medium', 'estimated': False})

    def change(self, lost_at, won_at, from_team, to_team, lost_by=None, won_by=None, confidence='medium'):
        self.changes.append({'id': f'pc-{int(lost_at*10)}', 'lostAt': lost_at, 'wonAt': won_at, 'frame': int(lost_at*FPS),
                             'fromTeam': from_team, 'toTeam': to_team,
                             'lostBy': self.people[lost_by]['id'] if lost_by is not None else None,
                             'wonBy': self.people[won_by]['id'] if won_by is not None else None,
                             'cause': 'tackle', 'confidence': confidence})

    def see_only(self, t0, t1, x0, y0, x1, y1):
        """The camera shows only the rectangle [x0, x1] x [y0, y1] between t0 and t1."""
        rect = np.array([[x0, y0], [(x0+x1)/2, y0], [x1, y0], [x1, y1], [(x0+x1)/2, y1], [x0, y1]], np.float32)
        self.view[self.k(t0):self.k(t1)] = rect

    # ------------------------------------------------------------ build
    def build(self):
        n = self.n
        rows = []
        for pid, pos in enumerate(self.pos):
            seen = np.all(np.isfinite(pos), axis=1)
            v = np.full_like(pos, np.nan)
            if seen.sum() >= 3:
                g = np.gradient(np.where(seen[:, None], pos, np.nan), DT, axis=0)
                v = np.where(np.isfinite(g), g, np.nan)
            for s in np.flatnonzero(seen):
                inside = self.view[s]
                rows.append((s, pid, pos[s, 0], pos[s, 1], v[s, 0], v[s, 1]))
        rows.sort(key=lambda r: (r[0], r[1]))
        r = np.array(rows, float) if rows else np.zeros((0, 6))
        sample = r[:, 0].astype(int)
        counts = np.bincount(sample, minlength=n)
        arrays = {'t': self.t.copy(), 'frame': np.round(self.t*FPS).astype(np.int64), 'segment': np.zeros(n, np.int16),
                  'period': np.zeros(n, np.int8), 'calibrated': self.calibrated.copy(),
                  'calibConf': np.full(n, .95, np.float32), 'pitchLevel': np.full(n, 3, np.int8), 'view': self.view.copy(),
                  'H': np.full((n, 9), np.nan), 'ball': self.ball.astype(np.float32), 'ballState': self.ball_state.copy(),
                  'ballConf': np.where(self.ball_state == 1, .9, 0.0).astype(np.float32),
                  'ballPConf': np.where(self.ball_state == 1, .9, 0.0).astype(np.float32),
                  'obsPtr': np.concatenate(([0], np.cumsum(counts))).astype(np.int64), 'pid': r[:, 1].astype(np.int32),
                  'x': r[:, 2].astype(np.float32), 'y': r[:, 3].astype(np.float32), 'vx': r[:, 4].astype(np.float32),
                  'vy': r[:, 5].astype(np.float32), 'sigma': np.full(len(r), .5, np.float32),
                  'idConf': np.full(len(r), .85, np.float32), 'box': np.full((len(r), 4), np.nan, np.float32)}
        ids = {p['id']: i for i, p in enumerate(self.people)}
        tables, runs = md._possession_tables({'timeline': sorted(self.spans, key=lambda s: s['t0'])}, ids)
        arrays.update(md._possession_samples(self.t, tables, runs))
        meta = {'version': md.VERSION, 'kind': 'match', 'matchId': 'synthetic', 'length': L, 'width': W, 'fps': FPS,
                'periods': [{'index': 0, 'start': T0, 'end': float(self.t[-1]), 'name': 'match',
                             'attacksRight': self.attacks_right}],
                'possRuns': runs, 'changes': self.changes, 'videoSize': [1920, 1080], 'fingerprint': 'synthetic',
                'possession': {'shares': None}}
        return md.MatchData(arrays, [dict(p) for p in self.people], meta)
