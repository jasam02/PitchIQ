"""Synthetic matches for the Full Match Intelligence tests (fullmatch/persons.py, gamestate.py, halftime.py, intel.py):
a MatchData built directly from arrays (tests/moments_synth.py's Scene), two halves when asked, and the stitched roster
an assembled match would have, so that a player can be cut into FRAGMENTS (the way segment boundaries and camera
gaps cut real ones) and the people / places / state engines run on it in memory.

    m = Match(seconds=300, halftime=(150.0, 160.0))      # two halves; team A attacks right first
    a, b = m.team('A'), m.team('B')                        # 4-4-2 per team, moving as a block with some play
    later = m.cut(a['LB'], 100.0, gap=2.0)                 # LB unseen 100-102 s, then a new fragment from 102 s
    data, index = m.build()

Fragments are the Scene's people ('A-01', ...); index['roster'] lists them as the match's stitched chains.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if HERE not in sys.path:
    sys.path.insert(0, HERE)
from moments_synth import F442, L, W, Scene  # noqa: E402

TEAM_KIT = {'A': (np.r_[np.zeros(20), [.1, .2, .3, .4]], np.r_[np.zeros(9), [.2, .3, .5]]),     # white-ish bins
            'B': (np.r_[[.5, .3, .2], np.zeros(21)], np.r_[[.6, .4], np.zeros(10)])}              # red-ish bins
OTHER_KIT = (np.r_[np.zeros(8), [.5, .5], np.zeros(14)], np.r_[np.zeros(5), [1.0], np.zeros(6)])  # an official


class Match:
    def __init__(self, seconds=300.0, halftime=None, attacks_right='A', seed=0):
        self.scene = Scene(seconds=seconds, attacks_right=attacks_right)
        self.halftime = halftime
        self.rng = np.random.default_rng(seed)
        self.first_right = attacks_right
        self.kits = {}
        self.boxes = {}
        self.segment_s = None

    # ------------------------------------------------------------ people
    def team(self, team, formation=F442, play=True):
        """A team at its defensive positions, then (play) moving as a block up and down and across, each player with a
        little wobble of his own; in the second half the same people at the mirrored places (the teams changed ends)."""
        s = self.scene
        ids = s.add_team(team, formation)
        for role, pid in ids.items():
            self.kits[pid] = TEAM_KIT[team]
            if not play:
                continue
            base = s.pos[pid].copy()
            reach = .2 if role == 'GK' else 1.0          # a keeper stays near his goal
            shift_x = reach*8.0*np.sin(s.t/23.0+(0 if team == 'A' else 1.3))
            shift_y = reach*6.0*np.sin(s.t/31.0+(.4 if team == 'A' else 2.1))
            wob = self.rng.normal(0, .15, (s.n, 2)).cumsum(axis=0)
            wob -= np.linspace(0, 1, s.n)[:, None]*wob[-1]
            s.pos[pid] = base+np.c_[shift_x, shift_y]+np.clip(wob, -2.5, 2.5)
        if self.halftime is not None:
            h1 = s.k(self.halftime[1])
            for pid in ids.values():
                s.pos[pid][h1:] = np.c_[L-s.pos[pid][h1:, 0], W-s.pos[pid][h1:, 1]]
            s.hide(ids['GK'], self.halftime[0], self.halftime[1])
            for pid in ids.values():
                s.hide(pid, self.halftime[0], self.halftime[1])
        return ids

    def cut(self, pid, t, gap=0.0, name=None):
        """The player pid's track ends at t and a NEW fragment carries him from t+gap: returns the new fragment."""
        s = self.scene
        team = s.people[pid]['team']
        keeper = s.people[pid]['role'] == 'GOALKEEPER'
        new = s.add(team, 0.0, 0.0, keeper=keeper, name=name)
        s.pos[new][:] = np.nan
        a, b = s.k(t), s.k(t+gap)
        s.pos[new][b:] = s.pos[pid][b:]
        s.pos[pid][a:] = np.nan
        self.kits[new] = self.kits.get(pid, TEAM_KIT[team])
        return new

    def person(self, team, x, y, t0, t1, keeper=False, name=None, kit=None):
        """Someone at a fixed pitch place between t0 and t1 only (a substitute in, a coach, an official)."""
        s = self.scene
        pid = s.add(team, 0.0, 0.0, keeper=keeper, name=name)
        s.pos[pid][:] = np.nan
        s.pos[pid][s.k(t0):s.k(t1)] = [x, y]
        self.kits[pid] = kit if kit is not None else TEAM_KIT[team]
        return pid

    def overlay(self, team, t0, t1, box=(.95, .93, .012, .03)):
        """A box that stays put in the picture (a broadcast graphic) while the camera pans: its pitch place drifts."""
        s = self.scene
        pid = s.add(team, 0.0, 0.0)
        s.pos[pid][:] = np.nan
        k0, k1 = s.k(t0), s.k(t1)
        s.pos[pid][k0:k1, 0] = np.linspace(80.0, 95.0, k1-k0)
        s.pos[pid][k0:k1, 1] = 64.0
        self.boxes[pid] = box
        self.kits[pid] = TEAM_KIT[team]
        return pid

    # ------------------------------------------------------------ build
    def build(self, segment_s=None):
        """(MatchData, index): the halves' directions and periods, boxes (moving with each person unless an overlay),
        and the roster of fragments (one member each, in segment segment_s-long pieces' numbering)."""
        s = self.scene
        data = s.build()
        n = data.samples
        t = data.t
        if self.halftime is not None:
            a, b = self.halftime
            other = 'B' if self.first_right == 'A' else 'A'
            data.period = np.where(t < b, 0, 1).astype(np.int8)
            data.periods = [{'index': 0, 'start': float(t[0]), 'end': float(a), 'name': 'first half', 'attacksRight': self.first_right},
                            {'index': 1, 'start': float(b), 'end': float(t[-1]), 'name': 'second half', 'attacksRight': other}]
            data.meta = dict(data.meta, periods=data.periods)
        obs_s = data.obs_sample()
        box = np.zeros((len(data.pid), 4), np.float32)
        for r in range(len(data.pid)):
            pid = int(data.pid[r])
            if pid in self.boxes:
                box[r] = self.boxes[pid]
            else:
                x, y = float(data.x[r]), float(data.y[r])
                box[r] = (x/L*.9+.02, .3+y/W*.6, .01, .03)
        data.box = box
        if segment_s:
            data.segment = (((t-t[0])//segment_s).astype(np.int16))
        roster = []
        for i, p in enumerate(data.people):
            rows = np.flatnonzero(data.pid == i)
            if not len(rows):
                continue
            ts = t[obs_s[rows]]
            seg = int(data.segment[obs_s[rows[0]]])
            roster.append({'matchId': p['id'], 'display': p['id'], 'team': p['team'], 'role': p['role'],
                           'members': [[seg, p['id']]], 'segments': [seg], 'seconds': round(len(rows)/5.0, 2),
                           'firstSeen': round(float(ts[0]), 2), 'lastSeen': round(float(ts[-1]), 2),
                           'firstFrame': int(round(ts[0]*25)), 'lastFrame': int(round(ts[-1]*25)),
                           'mean': [round(float(np.mean(data.x[rows]))/L, 4), round(float(np.mean(data.y[rows]))/W, 4)],
                           'history': [], 'how': 'one segment', 'links': [], 'confidence': 'medium'})
        periods = data.meta['periods']
        index = {'matchId': 'synthetic', 'roster': roster, 'segments': [{'index': 0, 'key': 'seg0', 'start': float(t[0]),
                                                                          'end': float(t[-1]), 'period': 0}],
                 'periods': [{'index': p['index'], 'start': p['start'], 'end': p['end'], 'name': p.get('name'),
                              'attacksRight': p.get('attacksRight'), 'defendsLeft': p.get('attacksRight'),
                              'directionAgreement': 1.0} for p in periods],
                 'pitchSize': {'length': L, 'width': W}, 'video': {'fps': 25.0}}
        return data, index

    def galleries(self, data):
        """The runs' galleries the persons engine reads: kit colours per fragment (no appearance embeddings)."""
        out = {}
        segments = range(int(np.max(data.segment))+1 if len(data.segment) else 1)
        for i, p in enumerate(data.people):
            j, sh = self.kits.get(i, TEAM_KIT.get(p['team'], OTHER_KIT))
            for seg in segments:
                out[(seg, p['id'])] = []
                out[('kit', seg, p['id'])] = [(np.asarray(j, float), np.asarray(sh, float))]*3
        return out


def people_of(match, items=(), segment_s=None):
    """(data, index, auto, resolved) of a synthetic match through fullmatch/persons.py."""
    from fullmatch import persons
    data, index = match.build(segment_s=segment_s)
    auto = persons.build_from(index, data, match.galleries(data), {})
    return data, index, auto, persons.resolve(auto, list(items))


# ---------------------------------------------------------------- play, stoppages, restarts (the state engine's tests)
def live(match, t0, t1, speed=3.6, period=8.0):
    """Outfield players running (median speed about `speed` m/s) between t0 and t1: each oscillates around where he is."""
    s = match.scene
    a, b = s.k(t0), s.k(t1)
    amp = speed*period/(2*np.pi)
    for pid, p in enumerate(s.people):
        if p['role'] == 'GOALKEEPER' or pid in match.boxes:
            continue
        ph = 1.7*pid
        tt = s.t[a:b]
        s.pos[pid][a:b, 0] += amp*np.sin(2*np.pi*tt/period+ph)
        s.pos[pid][a:b, 1] += .5*amp*np.cos(2*np.pi*tt/period+ph)


def kickoff_layout(match, t0, t1, taker_team, taker=None):
    """Both teams in their own halves between t0 and t1 (their formation pulled back), someone of taker_team on the
    centre spot (taker, else the first forward)."""
    s = match.scene
    a, b = s.k(t0), s.k(t1)
    for pid, p in enumerate(s.people):
        if p['role'] == 'GOALKEEPER' or pid in match.boxes or not np.isfinite(s.pos[pid][a:b]).all():
            continue
        x = s.pos[pid][a:b, 0]
        right = (s.attacks_right == p['team'])
        if match.halftime is not None and s.t[a] >= match.halftime[1]:
            right = not right
        s.pos[pid][a:b, 0] = np.minimum(x, L/2-3.0) if right else np.maximum(x, L/2+3.0)
    pid = taker
    if pid is None:
        pid = next(i for i, p in enumerate(s.people) if p['team'] == taker_team and p['role'] != 'GOALKEEPER'
                   and np.isfinite(s.pos[i][a:b]).all())
    s.pos[pid][a:b] = [L/2, W/2]


def ball_at(match, t0, t1, x, y):
    s = match.scene
    s.ball[s.k(t0):s.k(t1)] = [x, y]
    s.ball_state[s.k(t0):s.k(t1)] = 1


def ball_path(match, t0, t1, p0, p1):
    s = match.scene
    a, b = s.k(t0), s.k(t1)
    w = np.linspace(0, 1, max(1, b-a))[:, None]
    s.ball[a:b] = np.array(p0)*(1-w)+np.array(p1)*w
    s.ball_state[a:b] = 1


def dead(match, t0, t1, reason='stoppage', team=None):
    match.scene.dead(t0, t1)
    match.spells = getattr(match, 'spells', [])+[{'t0': float(t0), 't1': float(t1), 'reason': reason, 'restartTeam': team}]


def state_of(match, settings=None, people=None, places=None):
    """(result, data, index) of fullmatch/gamestate.analyse on the synthetic match."""
    from fullmatch import gamestate, halftime as ht
    data, index = match.build()
    settings = settings or ht.defaults(index)
    ht.apply_to_data(data, index, settings)
    possession = {'deadSpells': sorted(getattr(match, 'spells', []), key=lambda s: s['t0'])}
    return gamestate.analyse(data, possession, settings, index, people=people, places=places), data, index
