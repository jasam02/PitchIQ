"""The STATE of a full match over time and its EVENTS (Full Match Intelligence), from what the analysis saved: the 5 Hz
match data (soccer/moments/data.py: positions, speeds, the ball where it was placed, the camera's registration), the
possession record's dead-ball spells (soccer/possession.py, measured on Metrica: dead recall .87 / precision .98 through
a simulated camera), the people (fullmatch/persons.py) and the halftime (fullmatch/halftime.py). No video is read and
nothing is re-run.

States (intervals, source-video seconds, each with a confidence and the evidence that made it):
    IN_PLAY              the ball in play
    POSSIBLE_STOPPAGE    a stop the evidence cannot confirm (a dead spell with no restart seen, players walking without
                         the ball being dead)
    OUT_OF_PLAY          the ball out of play or play stopped, confirmed by the restart that follows
    RESTART_PREPARATION  the last part of a stoppage before a recognised set piece (players taking positions)
    HALFTIME             the halftime the coach set (or the plan's break)
    NON_MATCH_FOOTAGE    the picture is not the match: no pitch, no players for NON_MATCH_MIN_S or more inside a stoppage
                         (close-ups, replays, the crowd)
    UNKNOWN              no positions during play (a registration hole) or no evidence either way
Events (instants with an interval, a team when it is known, a confidence and the evidence):
    KICK_OFF, GOAL, CORNER_KICK, GOAL_KICK, THROW_IN, FREE_KICK, PENALTY_KICK, UNKNOWN_RESTART, SUBSTITUTION,
    FOUL_OR_WHISTLE_STOPPAGE, OTHER_STOPPAGE
A corner kick is an event; the time players gather for it is RESTART_PREPARATION; play is IN_PLAY again when it is
taken.

How (the rules, each checked on CAN-ISL video frames, scratch matchintel):
- A stop: the possession record's dead spell (its own evidence: the ball out over a line, a placed ball, a long unseen
  ball with the players walking - never an unseen ball alone), extended back to the goal before a kick-off.
- Confirmed (OUT_OF_PLAY) when play visibly restarts at the spell's end: the players' median speed over RESTART_S after
  it is RESTART_RISE above the RESTART_S before, or a set piece's layout is recognised there; else POSSIBLE_STOPPAGE.
  Players walking (median speed under WALK_MS for WALK_MIN_S) outside a dead spell is a POSSIBLE_STOPPAGE too: a slow
  spell is not a stoppage by itself.
- The restart's kind from the layout in the RESTART_S before it (team coordinates from the halftime's directions) and
  the ball where it was placed:
      KICK_OFF      someone within SPOT_M of the centre spot, each team's visible outfield players in their own half
                    (OWN_HALF_SHARE, KICK_SIDE_MIN of them or more), the players still (median speed under KICK_STILL_MS)
                    for KICK_HOLD_S, then a burst
      PENALTY_KICK  the ball within 1.5 m of a penalty mark, still
      CORNER_KICK   the ball within CORNER_M of a corner, or a player there with BOX_CROWD players in that box
      GOAL_KICK     the ball in a goal area (6 yards), or the defending keeper there with no opponent in the box
      THROW_IN      the ball or a lone player on / beyond a touchline (TOUCH_M)
      FREE_KICK     the ball still (under STILL_MS for STILL_S) in the field away from the lines
      UNKNOWN_RESTART  a confirmed restart whose kind the evidence does not name (no invented subtype)
- A KICK_OFF that is not a period's first is preceded by a GOAL (law 10: play restarts with a kick-off after a goal):
  a GOAL candidate with the conceding team = the kick-off team (the one at the spot) and its time = the last moment the
  ball was placed in a goal mouth (GOAL_MOUTH_M) within GOAL_LOOKBACK_S before, else the start of the stoppage before
  the kick-off, flagged 'time uncertain'. Never from the ball reaching x = 0 or 105 alone; scorers are never named.
- SUBSTITUTION candidates (law 3: a substitute enters at the halfway line during a stoppage): a team's player last seen
  by the halfway line on a touchline and one of the same team first seen there, in one stoppage (each player once);
  the incoming player is a different person whatever the coach decides.
Analytics masks (masks()): ACTIVE_OPEN_PLAY, SET_PIECE_PREPARATION, SET_PIECE_EXECUTION (the EXEC_S after a corner, free
kick or penalty is taken), OTHER_STOPPAGE, HALFTIME, NON_MATCH_FOOTAGE, UNKNOWN; eligible() says which ones each kind of
analysis uses. Nothing is deleted: the masks only decide what an analysis counts.

The coach's corrections (assembly/intel/events.json) are applied on top (apply_corrections): confirm, retype, move,
remove an event, add a missing one (a goal, a substitution), set the state of an interval.
"""
import hashlib
import json
import math
import uuid
from collections import Counter
from pathlib import Path

import numpy as np

from soccer.moments import goals as goal_rules

VERSION = 5
STATES = ('IN_PLAY', 'POSSIBLE_STOPPAGE', 'OUT_OF_PLAY', 'RESTART_PREPARATION', 'HALFTIME', 'NON_MATCH_FOOTAGE', 'UNKNOWN')
EVENTS = ('GOAL', 'CORNER_KICK', 'GOAL_KICK', 'THROW_IN', 'FREE_KICK', 'PENALTY_KICK', 'KICK_OFF', 'SUBSTITUTION',
          'FOUL_OR_WHISTLE_STOPPAGE', 'OTHER_STOPPAGE', 'UNKNOWN_RESTART')
MASKS = ('ACTIVE_OPEN_PLAY', 'SET_PIECE_PREPARATION', 'SET_PIECE_EXECUTION', 'OTHER_STOPPAGE', 'HALFTIME',
         'NON_MATCH_FOOTAGE', 'UNKNOWN')
SET_PIECES = ('CORNER_KICK', 'FREE_KICK', 'PENALTY_KICK')
RESTARTS = ('KICK_OFF', 'CORNER_KICK', 'GOAL_KICK', 'THROW_IN', 'FREE_KICK', 'PENALTY_KICK', 'UNKNOWN_RESTART')

MERGE_S = 2.0              # dead spells this close are one stoppage
RESTART_S = 3.0
RESTART_RISE = .6          # m/s of median player speed after the restart over before it
WALK_MS = 1.0
WALK_MIN_S = 10.0
MIN_PLAYERS = 6            # visible outfield players for a speed or a layout to count
NON_MATCH_MIN_S = 3.0
SPOT_M = 1.5
OWN_HALF_SHARE = .8
KICK_SIDE_MIN = 3
KICK_STILL_MS = 1.6
KICK_HOLD_S = 2.0
KICK_BURST_S = 12.0        # the burst after a kick-off layout within this
KICK_MERGE_S = 30.0
LINEUP_S = 8.0
LINEUP_M = 10.0
PERIOD_START_S = 120.0     # a kick-off this close to a period's start is that period's
PERIOD_PLAY_S = 20.0       # ... unless this much play came before it
CORNER_M = 4.0
BOX_CROWD = 6
TOUCH_M = 1.5
PLACE_SLACK_M = 2.5          # where a placed ball may really be (its error and the calibration's)
PLACED_S = .6                # a ball still this long before the restart is a placed ball
GOAL_KICK_BOX = 4            # players in the penalty area at a goal kick (the keeper and defenders: law 16)
STILL_MS = .6
STILL_S = 1.0
GOAL_MOUTH_M = 4.0            # m of the goal line (the ball's own error and the calibration's)
GOAL_WIDE_M = 1.5             # m beyond a post
MOUTH_GAP_S = .3              # samples of the ball in a goal mouth this close together are one spell (it sits in the net)
LINE_NEAR_M = 1.5             # the spell's first sample this close to the goal line, or past it: when the ball went in
STRIKE_MS = 12.0              # the ball this fast or faster towards that goal line, from STRIKE_WITHIN_M of it or closer,
STRIKE_WITHIN_M = 35.0        # in the STRIKE_LOOK_S before its last sight in the goal mouth: the strike that went in (the
STRIKE_LOOK_S = 6.0           # ball often lost at the shot, then seen in the net seconds later)
RESTART_GAP_S = 5.0           # a restart of play this long or more before a goal's stop: the goal came after it
PEN_AFTER_S = 4.0             # the ball in the goal mouth this soon after a penalty's kick is that penalty's goal
PEN_BEFORE_S = 15.0           # a penalty's own kick can be the restart that sets the floor: its layout is looked for from
                              # this long before the floor, its kick no more than a second before it
PLAY_MS = 2.0               # median player speed of play (CAN-ISL: 2.2-2.4 in play, 1.2 in dead spells)
PLAY_ON_S = 12.0
GOAL_LOOKBACK_S = 240.0
EXEC_S = 8.0
PREP_MAX_S = 10.0
SUB_HALFWAY_M = 15.0
SUB_TOUCH_M = 8.0
SUB_AFTER_S = 20.0
FILE = 'events.json'


# ---------------------------------------------------------------- features
class Features:
    """Per sample of a MatchData: visible outfield players per team, their median speed, own-half shares, the nearest
    player to the centre spot and who it is, the ball (placed, its speed)."""

    def __init__(self, data, team_of, keeper_of, excluded):
        n = data.samples
        L, W = data.length, data.width
        self.data, self.L, self.W = data, L, W
        self.t = data.t
        self.nA, self.nB = np.zeros(n, int), np.zeros(n, int)
        self.act = np.full(n, np.nan)
        self.ownA, self.ownB = np.full(n, np.nan), np.full(n, np.nan)
        self.spot = np.full(n, np.nan)
        self.spot_team = np.full(n, '', dtype=object)
        self.halfA, self.halfB = np.full(n, np.nan), np.full(n, np.nan)   # each team's nearest player to the halfway line
        self.leftA, self.leftB = np.full(n, np.nan), np.full(n, np.nan)   # share of each team's players in the left half
        self.rightA, self.rightB = np.full(n, np.nan), np.full(n, np.nan)
        dirs = {t: _directions(data, t) for t in ('A', 'B')}
        pid_team = np.array([team_of.get(p['id']) or '' for p in data.people], dtype=object)
        pid_keeper = np.array([bool(keeper_of.get(p['id'])) for p in data.people])
        pid_out = np.array([p['id'] in excluded for p in data.people])
        speed = np.hypot(data.vx, data.vy)
        for s in range(n):
            r = data.rows(s)
            if r.stop == r.start:
                continue
            p = data.pid[r]
            ok = ~pid_keeper[p] & ~pid_out[p]
            tm = pid_team[p]
            x, y = data.x[r], data.y[r]
            a, b = ok & (tm == 'A'), ok & (tm == 'B')
            self.nA[s], self.nB[s] = int(a.sum()), int(b.sum())
            sp = speed[r][(a | b) & np.isfinite(speed[r])]
            if len(sp) >= MIN_PLAYERS:
                self.act[s] = float(np.median(sp))
            for code, sel, own, half, left, right in (('A', a, self.ownA, self.halfA, self.leftA, self.rightA),
                                                      ('B', b, self.ownB, self.halfB, self.leftB, self.rightB)):
                if not sel.any():
                    continue
                xs = x[sel]
                half[s] = float(np.min(np.abs(xs-L/2)))
                left[s] = float(np.mean(xs <= L/2+1.5))
                right[s] = float(np.mean(xs >= L/2-1.5))
                d = dirs[code][s]
                if d != 0:
                    own[s] = left[s] if d > 0 else right[s]
            both = a | b
            if both.any():
                dist = np.hypot(x[both]-L/2, y[both]-W/2)
                k = int(np.argmin(dist))
                self.spot[s] = float(dist[k])
                self.spot_team[s] = tm[both][k]
        self.ball = data.ball.astype(float)
        self.ball_ok = np.all(np.isfinite(self.ball), axis=1) & (data.ballState == 1)
        bs = np.full(n, np.nan)
        if n > 2:
            dt = np.maximum(data.t[2:]-data.t[:-2], 1e-3)
            bs[1:-1] = np.hypot(*(self.ball[2:]-self.ball[:-2]).T)/dt
            bs[1:-1][~(self.ball_ok[2:] & self.ball_ok[:-2])] = np.nan
        self.ball_speed = bs
        self.blind = (self.nA+self.nB) == 0

    def window(self, t0, t1):
        return np.flatnonzero((self.t >= t0) & (self.t <= t1))


def _directions(data, team):
    out = np.zeros(data.samples, np.int8)
    for p in data.periods:
        if p.get('attacksRight') not in ('A', 'B'):
            continue
        out[data.period == int(p['index'])] = 1 if p['attacksRight'] == team else -1
    return out


def _runs(mask, t, min_s):
    """[(t0, t1)] of the runs of True in `mask` (samples at times t) lasting min_s or more."""
    out = []
    idx = np.flatnonzero(mask)
    if not len(idx):
        return out
    splits = np.flatnonzero(np.diff(idx) > 1)+1
    for run in np.split(idx, splits):
        a, b = float(t[run[0]]), float(t[run[-1]])
        if b-a >= min_s:
            out.append((a, b))
    return out


def _median(x):
    x = x[np.isfinite(x)]
    return float(np.median(x)) if len(x) else None


# ---------------------------------------------------------------- kick-offs
def _side_ok(n, share):
    """A team's side holds when KICK_SIDE_MIN or more of its players are seen and OWN_HALF_SHARE of them are in that
    half, or when fewer are seen and all of them are; a team not seen does not refuse it."""
    return np.where(n >= KICK_SIDE_MIN, share >= OWN_HALF_SHARE, np.where(n >= 1, share >= .999, True))


def kick_offs(F):
    """[(t layout start, t kick, team at the spot or None, how, side)]: the two teams in opposite halves (each team's
    visible outfield players, OWN_HALF_SHARE of them; either way round, so a wrong direction setting cannot hide a
    kick-off), the players still, for KICK_HOLD_S with someone within SPOT_M of the centre spot at some point, or for
    LINEUP_S with both teams lined up within LINEUP_M of the halfway line (the taker not placed); then a burst of play
    within KICK_BURST_S, or play already on when the picture is placed again after a gap that began right after the
    layout (the kick-off taken off screen). side: 'A left' or 'A right', the half team A held."""
    a_left = _side_ok(F.nA, np.nan_to_num(F.leftA)) & _side_ok(F.nB, np.nan_to_num(F.rightB))
    a_right = _side_ok(F.nA, np.nan_to_num(F.rightA)) & _side_ok(F.nB, np.nan_to_num(F.leftB))
    some = (F.nA >= KICK_SIDE_MIN) | (F.nB >= KICK_SIDE_MIN)
    still = np.where(np.isfinite(F.act), F.act <= KICK_STILL_MS, False)
    base = (a_left | a_right) & some & (F.nA+F.nB >= MIN_PLAYERS) & still
    lined = (F.nA >= 1) & (F.nB >= 1) & np.where(F.nA >= 1, F.halfA <= LINEUP_M, True) & \
        np.where(F.nB >= 1, F.halfB <= LINEUP_M, True)
    out = []
    for a, b in _runs(base, F.t, KICK_HOLD_S):
        w = F.window(a, b)
        on_spot = [i for i in F.window(a-3.0, b) if F.spot[i] <= SPOT_M]
        held = sum(1 for i in w if F.spot[i] <= SPOT_M) >= 5          # the taker on the spot a second or more
        if on_spot:
            how = 'someone on the centre spot'
        elif b-a >= LINEUP_S and np.mean(lined[w]) >= .8:
            how = 'both teams lined up either side of the halfway line'
        else:
            continue
        after = F.window(b, b+KICK_BURST_S)
        rise = [i for i in after if np.isfinite(F.act[i]) and F.act[i] >= KICK_STILL_MS+1.0]
        kick = float(F.t[rise[0]]) if rise else None
        if kick is None:
            nxt = F.window(b, b+3.0)
            if len(nxt) and F.blind[nxt].any():
                j = int(nxt[np.argmax(F.blind[nxt])])
                while j < len(F.t) and F.blind[j]:
                    j += 1
                seen = F.window(F.t[min(j, len(F.t)-1)], F.t[min(j, len(F.t)-1)]+3.0)
                if len(seen) and _median(F.act[seen]) is not None and _median(F.act[seen]) >= KICK_STILL_MS+1.0:
                    kick = float(b)+1.0
                    how += '; the kick taken off screen'
        if kick is None:
            continue
        side = 'A left' if np.sum(a_left[w]) >= np.sum(a_right[w]) else 'A right'
        team = Counter(F.spot_team[on_spot].tolist()).most_common(1)[0][0] if on_spot else None
        if team and not held:
            how += '; who took it is not certain (nobody held the spot)'
        if out and kick-out[-1][1] < KICK_MERGE_S:
            out[-1] = (out[-1][0], kick, out[-1][2] or team or None, out[-1][3], out[-1][4])
            continue
        out.append((a, kick, team or None, how, side))
    return out


# ---------------------------------------------------------------- restarts
def classify_restart(F, t, team, defending=None):
    """(kind, confidence, evidence) of the restart at time t: from where the ball was placed just before it (with
    PLACE_SLACK_M for the ball's own error and the calibration's), else from a corner layout; UNKNOWN_RESTART when
    neither names it (a lone player near a touchline is not a throw-in: assistants, ball boys and slow play stand
    there too)."""
    L, W = F.L, F.W
    ev = []
    bwin = [i for i in F.window(t-2.0, t+1.0) if F.ball_ok[i]]
    ball = F.ball[bwin[len(bwin)//2]] if bwin else None
    still = [i for i in F.window(t-RESTART_S, t) if F.ball_ok[i] and np.isfinite(F.ball_speed[i]) and F.ball_speed[i] < STILL_MS]
    still_s = len(still)/5.0
    slack = PLACE_SLACK_M
    if ball is not None and still_s >= PLACED_S:
        # a restart is a placed ball kicked (or thrown): a ball moving at the restart says nothing about its kind
        bx, by = float(ball[0]), float(ball[1])
        ev.append(f'the ball placed at ({bx:.0f}, {by:.0f}) m, still {still_s:.1f} s')
        for px in (11.0, L-11.0):
            if math.hypot(bx-px, by-W/2) <= 1.5 and still_s >= STILL_S:
                return 'PENALTY_KICK', 'medium', ev+['on a penalty mark']
        for cx in (0.0, L):
            for cy in (0.0, W):
                if abs(bx-cx) <= CORNER_M and abs(by-cy) <= CORNER_M:
                    return 'CORNER_KICK', 'medium', ev+['at a corner arc']
        if (bx <= 5.5+slack or bx >= L-5.5-slack) and abs(by-W/2) <= 9.16+slack:
            gx = 0.0 if bx < L/2 else L
            inbox = _in_box(F, t, gx)
            if inbox <= GOAL_KICK_BOX:
                return 'GOAL_KICK', 'medium', ev+[f'in a goal area, {inbox} players in that penalty area']
            ev.append(f'in a goal area but {inbox} players in that penalty area (not a goal kick)')
        elif by <= slack or by >= W-slack:
            return 'THROW_IN', 'medium', ev+['on a touchline']
        elif still_s >= STILL_S and 5.5+slack < bx < L-5.5-slack:
            return 'FREE_KICK', 'medium', ev+['in the field']
    elif ball is not None:
        ev.append(f'the ball moving at ({float(ball[0]):.0f}, {float(ball[1]):.0f}) m: not a placed ball')
    data = F.data
    for i in F.window(t-RESTART_S, t+.5):
        r = data.rows(int(i))
        xs, ys = data.x[r], data.y[r]
        if not len(xs):
            continue
        for cx in (0.0, L):
            for cy in (0.0, W):
                if np.any(np.hypot(xs-cx, ys-cy) <= 2.5):
                    inbox = int(np.sum((np.abs(xs-cx) <= 16.5) & (np.abs(ys-W/2) <= 20.16)))
                    if inbox >= BOX_CROWD:
                        return 'CORNER_KICK', 'low', ev+[f'a player at a corner with {inbox} players in that penalty area']
    return 'UNKNOWN_RESTART', 'low', ev+['the restart is seen but not its kind']


def _in_box(F, t, goal_x):
    """The most players seen inside the penalty area of the goal at goal_x in the RESTART_S before t."""
    data, L, W = F.data, F.L, F.W
    most = 0
    for i in F.window(t-RESTART_S, t):
        r = data.rows(int(i))
        xs, ys = data.x[r], data.y[r]
        most = max(most, int(np.sum((np.abs(xs-goal_x) <= 16.5) & (np.abs(ys-W/2) <= 20.16))))
    return most


# ---------------------------------------------------------------- the engine
def analyse(data, possession, halftime_settings, index, people=None, places=None, video=None, footage_cache=None):
    """The automatic state timeline, events and masks (see the module doc). data: MatchData with the halftime applied
    (halftime.apply_to_data); people: persons.resolve()'s result (teams, keepers, exclusions, places) or None."""
    from . import halftime as ht
    team_of, keeper_of, excluded = {}, {}, set()
    for p in data.people:
        team_of[p['id']] = p.get('team')
        keeper_of[p['id']] = p.get('role') == 'GOALKEEPER'
    if people:
        excluded = set(people.get('excluded') or {})
        for f, t in (people.get('teams') or {}).items():
            team_of[f] = t
        for pid, p in (people.get('people') or {}).items():
            for f in p['fragments']:
                if p.get('keeper'):
                    keeper_of[f] = True
    F = Features(data, team_of, keeper_of, excluded)
    t_all = F.t
    rng = ht.match_range(index)
    half = ht.interval(halftime_settings)
    periods = ht.periods(index, halftime_settings)
    starts = [float(p['start']) for p in periods]
    events = []

    def add_event(kind, t, t0=None, t1=None, team=None, confidence='low', evidence=(), **extra):
        e = {'id': f'{kind.lower()}-{int(round(t*10))}', 'type': kind, 't': round(float(t), 2),
             't0': round(float(t if t0 is None else t0), 2), 't1': round(float(t if t1 is None else t1), 2),
             'team': team, 'confidence': confidence, 'evidence': list(evidence), 'source': 'automatic'}
        e.update(extra)
        events.append(e)
        return e
    # 1. stoppages: the dead spells, merged
    spells = sorted(((float(s['t0']), float(s['t1']), s.get('reason'), s.get('restartTeam'))
                     for s in (possession or {}).get('deadSpells') or [] if float(s['t1']) > float(s['t0'])))
    merged = []
    for a, b, why, team in spells:
        if merged and a-merged[-1]['t1'] <= MERGE_S:
            merged[-1]['t1'] = max(merged[-1]['t1'], b)
            merged[-1]['reasons'].append(why)
            merged[-1]['team'] = team or merged[-1]['team']
        else:
            merged.append({'t0': a, 't1': b, 'reasons': [why], 'team': team})
    # 2. kick-offs (any time: a goal's kick-off may fall outside a dead spell)
    kicks = kick_offs(F)
    stoppages = []
    for st in merged:
        a, b = st['t0'], st['t1']
        before = _median(F.act[F.window(b-RESTART_S, b)])
        after = _median(F.act[F.window(b, b+RESTART_S)])
        rise = before is not None and after is not None and after-before >= RESTART_RISE
        st.update(actBefore=before, actAfter=after, restartSeen=bool(rise))
        stoppages.append(st)
    # every kick-off belongs to a stoppage: the one that ends at it, else a new one from its layout
    for lay0, kick, team, how, side in kicks:
        own = [st for st in stoppages if st['t0']-5.0 <= kick <= st['t1']+KICK_BURST_S]
        if own:
            st = min(own, key=lambda s: abs(s['t1']-kick))
            st['t1'] = max(st['t1'], kick) if kick-st['t1'] < KICK_BURST_S else st['t1']
            st['kickoff'] = (lay0, kick, team, how, side)
        else:
            stoppages.append({'t0': lay0, 't1': kick, 'reasons': ['kick-off layout'], 'team': team, 'kickoff': (lay0, kick, team, how, side),
                              'actBefore': None, 'actAfter': None, 'restartSeen': True})
    stoppages.sort(key=lambda s: s['t0'])
    # the first kick-off near each period's start is that period's (law 8); any later one follows a goal (law 10)
    firsts = set()
    for s0 in starts:
        near = sorted((k for k in kicks if s0-5.0 <= k[1] <= s0+PERIOD_START_S), key=lambda k: k[1])
        if not near:
            continue
        lay0, kick = near[0][0], near[0][1]
        w = F.window(s0, lay0)
        played = sum(b-a for a, b in _runs(np.where(np.isfinite(F.act[w]), F.act[w] >= PLAY_MS, False), F.t[w], 2.0))
        if played < PERIOD_PLAY_S:          # play before it: the period's own kick-off was not seen, this one is a goal's
            firsts.add(kick)
    # 3. restarts, goals, the stoppage's own event
    for st in stoppages:
        b = st['t1']
        if st.get('kickoff'):
            lay0, kick, kteam, how, side = st['kickoff']
            first = kick in firsts
            e = add_event('KICK_OFF', kick, lay0, kick, team=kteam, confidence='high' if 'spot' in how else 'medium',
                          evidence=[f'the teams in opposite halves ({side.replace("A ", "team A in the ")} one), the players '
                                    'still', how], periodStart=first, side=side)
            st['restart'] = e
            if not first:
                # the last restart of play before this stop (a stoppage that ended with the players speeding up): the
                # goal came after it
                floor = max((s['t1'] for s in stoppages if s is not st and s.get('restartSeen') and s['t1'] <= st['t0']-RESTART_GAP_S),
                            default=None)
                goal = _goal_before(F, st, kick, kteam, periods, lay0, weak='not certain' in how, side=side, floor=floor)
                if goal.get('kickTeam'):
                    e['team'] = goal['kickTeam']
                    e['evidence'].append(f"team {goal['kickTeam']}: the goal before it was team {goal['team']}'s")
                g = add_event('GOAL', goal['t'], goal['t'], kick, team=goal['team'], confidence=goal['confidence'],
                              evidence=goal['evidence'], timeUncertain=goal['timeUncertain'], concededBy=kteam,
                              scorer=None, assist=None)
                st['goal'] = g
                st['t0'] = min(st['t0'], goal['t'])
            continue
        kind, conf, ev = classify_restart(F, b, st.get('team'))
        if st['restartSeen'] or kind != 'UNKNOWN_RESTART':
            e = add_event(kind, b, b, b, team=st.get('team'), confidence=conf if st['restartSeen'] else 'low',
                          evidence=ev+(['play restarts: the players speed up'] if st['restartSeen'] else []))
            st['restart'] = e
            if kind == 'FREE_KICK' and 'out_of_play' not in st['reasons']:
                add_event('FOUL_OR_WHISTLE_STOPPAGE', st['t0'], st['t0'], b, confidence='low',
                          evidence=['play stopped in the field and restarted with a free kick (the foul is not seen)'])
            elif kind == 'UNKNOWN_RESTART' and b-st['t0'] >= 60.0:
                add_event('OTHER_STOPPAGE', st['t0'], st['t0'], b, confidence='low',
                          evidence=[f'a stoppage of {b-st["t0"]:.0f} s with no recognised restart'])
    # a period starts with a kick-off (law 8): where none was recognised there, at the first play after the start
    kos = sorted((e for e in events if e['type'] == 'KICK_OFF'), key=lambda e: e['t'])
    first_team = next((e['team'] for e in kos if e.get('periodStart') and e['team']), None)
    for k, s0 in enumerate(starts):
        if any(e.get('periodStart') and abs(e['t']-s0) <= PERIOD_START_S for e in kos):
            continue
        play = [i for i in F.window(s0, s0+PERIOD_START_S) if np.isfinite(F.act[i]) and F.act[i] >= 2.0]
        t = float(F.t[play[0]]) if play else s0
        team = None if first_team is None or k == 0 else (first_team if k % 2 == 0 else ('B' if first_team == 'A' else 'A'))
        add_event('KICK_OFF', t, s0, t, team=team, confidence='medium', periodStart=True, timeUncertain=True,
                  evidence=['a period starts with a kick-off (law 8); the kick itself is not seen']+
                           ([f'team {team}: the other team kicked off the first half'] if team else []))
    # 4. substitutions (law 3: a substitute enters at the halfway line during a stoppage): a team's outfield player
    # last seen by the halfway line on a touchline and one of the same team first seen there, in one stoppage
    person_of = (people or {}).get('person') or {}
    slot_of = {}
    for team, plist in (places or {}).items():
        for place in plist:
            for node in place['nodes']:
                for f in node['fragments']:
                    slot_of[f] = place['slot']
    ends, begins = [], []
    for i, p in enumerate(data.people):
        f = p['id']
        if f in excluded or keeper_of.get(f) or team_of.get(f) not in ('A', 'B'):
            continue
        rows = np.flatnonzero(data.pid == i)
        if not len(rows):
            continue
        for at, r, out in (('last', rows[-1], ends), ('first', rows[0], begins)):
            x, y = float(data.x[r]), float(data.y[r])
            if abs(x-data.length/2) <= SUB_HALFWAY_M and (y <= SUB_TOUCH_M or y >= data.width-SUB_TOUCH_M):
                sample = int(np.searchsorted(data.obsPtr, r, side='right')-1)
                out.append((f, team_of[f], float(data.t[sample]), x, y))
    used = set()                     # each player leaves or comes on once: the nearest in time pair first
    for st in stoppages:
        a0, b0 = st['t0']-2.0, st['t1']+SUB_AFTER_S
        pairs = sorted(((abs(ti-to), fo, team, to, fi, ti) for fo, team, to, xo, yo in ends if a0 <= to <= b0
                        for fi, team_i, ti, xi, yi in begins
                        if team_i == team and fi != fo and a0 <= ti <= b0 and ti >= to-30.0), key=lambda q: q[0])
        for _, fo, team, to, fi, ti in pairs:
            if fo in used or fi in used:
                continue
            used |= {fo, fi}
            add_event('SUBSTITUTION', ti, to, ti, team=team, confidence='low',
                      evidence=[f'a team {team} player last seen by the halfway line on a touchline at {to:.0f} s',
                                f'a team {team} player first seen there at {ti:.0f} s', 'during a stoppage'],
                      slot=slot_of.get(fo), outSlot=slot_of.get(fo), inSlot=slot_of.get(fi),
                      outPerson=person_of.get(fo), inPerson=person_of.get(fi), outFragment=fo, inFragment=fi)
    # the halftime's directions against the kick-offs' sides (each team stands in the half it defends)
    check = {'kickOffs': 0, 'agree': 0, 'disagree': 0, 'details': []}
    for e in events:
        if e['type'] != 'KICK_OFF' or e.get('side') not in ('A left', 'A right'):
            continue
        p = next((q for q in periods if float(q['start'])-PERIOD_START_S <= e['t'] <= float(q['end'])+1.0), None)
        if not p or p.get('attacksRight') not in ('A', 'B'):
            continue
        says = 'A' if e['side'] == 'A left' else 'B'       # the team in the left half attacks to the right
        ok = says == p['attacksRight']
        check['kickOffs'] += 1
        check['agree' if ok else 'disagree'] += 1
        check['details'].append({'t': e['t'], 'period': p.get('name'), 'side': e['side'], 'agrees': ok})
    if check['disagree']:
        check['note'] = (f"{check['disagree']} of {check['kickOffs']} kick-offs had the teams on the other sides than the "
                         'halftime setting says: check which way team A attacks in the first half')
    # 5. the state timeline
    footage = pitch_check(video, _runs(F.blind, F.t, NON_MATCH_MIN_S), cache=footage_cache) if video else None
    timeline = _timeline(F, stoppages, rng, half, footage)
    events.sort(key=lambda e: e['t'])
    return {'version': VERSION, 'timeline': timeline, 'events': events, 'directionCheck': check,
            'stoppages': [{'t0': round(s['t0'], 2), 't1': round(s['t1'], 2), 'reasons': s['reasons'],
                           'restartSeen': s['restartSeen'], 'restart': (s.get('restart') or {}).get('type')}
                          for s in stoppages],
            'stats': _stats(timeline, events)}


def pitch_check(video, runs, frames=6, green_min=.25, cache=None):
    """{(t0, t1): 'no pitch' | 'pitch'} for stretches without positions: the median share of grass-green pixels (HSV)
    over a few frames of each; under green_min there is no pitch in the picture. cache: a JSON file remembering the
    answers per video and stretch (they depend on nothing else)."""
    import cv2
    out = {}
    if not runs:
        return out
    known = {}
    stamp = None
    if cache is not None:
        try:
            st = Path(video).stat()
            stamp = f'{Path(video).name}|{st.st_size}'
            saved = json.loads(Path(cache).read_text(encoding='utf-8'))
            if saved.get('video') == stamp:
                known = saved.get('runs') or {}
        except (OSError, ValueError):
            known = {}
    todo = []
    for a, b in runs:
        key = f'{a:.1f}-{b:.1f}'
        if key in known:
            out[(round(a, 1), round(b, 1))] = known[key]
        else:
            todo.append((a, b))
    if not todo:
        return out
    runs = todo
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        return out
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    for a, b in runs:
        shares = []
        for tt in np.linspace(a, b, frames):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(tt*fps))
            ok, fr = cap.read()
            if not ok:
                continue
            hsv = cv2.cvtColor(cv2.resize(fr, (320, 180), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2HSV)
            h, sat, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
            shares.append(float(np.mean((h >= 35) & (h <= 85) & (sat >= 60) & (v >= 40))))
        if shares:
            out[(round(a, 1), round(b, 1))] = 'no pitch' if float(np.median(shares)) < green_min else 'pitch'
            known[f'{a:.1f}-{b:.1f}'] = out[(round(a, 1), round(b, 1))]
    cap.release()
    if cache is not None and stamp is not None:
        try:
            Path(cache).parent.mkdir(parents=True, exist_ok=True)
            Path(cache).write_text(json.dumps({'video': stamp, 'runs': known}), encoding='utf-8')
        except OSError:
            pass
    return out


def _position(data, frag_id, at):
    i = data.index_of.get(frag_id)
    if i is None:
        return None
    rows = np.flatnonzero(data.pid == i)
    if not len(rows):
        return None
    r = rows[0] if at == 'first' else rows[-1]
    return float(data.x[r]), float(data.y[r])


def _goal_time(F, mouth, k, gx):
    """When the goal went in: the earliest of the last sight of the ball in the goal mouth (mouth[k]: often still in the
    net seconds later), the start of that spell in the mouth (its first sample within LINE_NEAR_M of the line or past
    it) and the line crossing worked out from the ball's last fast move towards that goal (STRIKE_MS)."""
    t_last = float(F.t[mouth[k]])
    j = k
    while j > 0 and F.t[mouth[j]]-F.t[mouth[j-1]] <= MOUTH_GAP_S:
        j -= 1
    sign = 1.0 if gx > 0 else -1.0                 # x grows towards the right goal
    spell = mouth[j:k+1]
    near = [i for i in spell if sign*(gx-float(F.ball[i][0])) <= LINE_NEAR_M]
    t_spell = float(F.t[near[0] if near else spell[0]])
    t_cross = t_last
    idx = [i for i in F.window(t_last-STRIKE_LOOK_S, t_last+.01) if F.ball_ok[i]]
    for a, b in zip(idx, idx[1:]):
        dt = float(F.t[b]-F.t[a])
        if not 0 < dt <= .45:
            continue
        xa, xb = float(F.ball[a][0]), float(F.ball[b][0])
        v = sign*(xb-xa)/dt
        if v >= STRIKE_MS and sign*(gx-xa) <= STRIKE_WITHIN_M:
            t_cross = float(F.t[b])+max(0.0, sign*(gx-xb))/v
    return min(t_last, t_spell, t_cross)


def _goal_before(F, st, kick, kteam, periods, lay0, weak=False, side=None, floor=None):
    """The goal a kick-off restarts after: when (the last time before the kick-off layout the ball was placed in a goal
    mouth, GOAL_MOUTH_M of the line and GOAL_WIDE_M of the posts, with no spell of play (median speed PLAY_MS for
    PLAY_ON_S) between it and the layout and after `floor` (the last restart of play before the goal's stop: anything
    earlier was not the goal), taken back to when it went in (_goal_time); a penalty taken in that span (the players'
    layout of law 14, soccer/moments/goals.penalty_layout) when the ball was not seen in a goal mouth after it, or only
    within PEN_AFTER_S; else the start of the stop before the kick-off: back from the layout over the samples without
    play), and who scored (the team that did not kick off; where the goal's end says otherwise the team is left open)."""
    L, W = F.L, F.W
    scorer = None if kteam not in ('A', 'B') else ('B' if kteam == 'A' else 'A')
    ev = [f'a kick-off at {kick:.0f} s'+(f' by team {kteam}' if kteam else '')]
    lo = lay0-GOAL_LOOKBACK_S if floor is None else max(lay0-GOAL_LOOKBACK_S, float(floor))
    idx = F.window(lo, lay0)
    mouth = [i for i in idx if F.ball_ok[i] and (F.ball[i][0] <= GOAL_MOUTH_M or F.ball[i][0] >= L-GOAL_MOUTH_M)
             and abs(F.ball[i][1]-W/2) <= 3.66+GOAL_WIDE_M]
    found = None                      # (when it went in, the last sight of it in the mouth, which end)
    for k in range(len(mouth)-1, -1, -1):
        i = mouth[k]
        later = F.window(F.t[i], lay0)
        busy = _runs(np.where(np.isfinite(F.act[later]), F.act[later] >= PLAY_MS, False), F.t[later], PLAY_ON_S)
        if busy:
            break                     # play went on after it: neither this nor an earlier one is the goal
        end = 'left' if float(F.ball[i][0]) < L/2 else 'right'
        found = (_goal_time(F, mouth, k, 0.0 if end == 'left' else float(L)), float(F.t[i]), end)
        break
    pen = None
    for gx, which in ((0.0, 'left'), (float(L), 'right')):
        spell = goal_rules.penalty_layout(F.data, lo-PEN_BEFORE_S, lay0, gx)
        if spell and spell[1] >= lo-1.0 and (pen is None or spell[1] > pen[1]):
            pen = (spell[0], spell[1], which)
    if pen and (found is None or found[0] <= pen[1]+PEN_AFTER_S):
        t, end = pen[1], pen[2]
        ev.append(f'a penalty at the {end} goal at about {t:.0f} s: the players set out for it {pen[1]-pen[0]:.1f} s (law 14)')
        if found and found[0] < pen[0]:
            ev.append(f'the ball in the {found[2]} goal mouth at {found[1]:.0f} s came before it: not the goal')
    elif found:
        t, t_last, end = found
        ev.append(f'the ball placed in the {end} goal mouth at {t_last:.0f} s, no play after it')
        if t < t_last-.5:
            ev.append(f'it went in at about {t:.0f} s (when the ball reached the goal line)')
    if pen and (found is None or found[0] <= pen[1]+PEN_AFTER_S) or found:
        attacker = None
        if side in ('A left', 'A right'):
            # at the kick-off each team stands in the half it defends: the goal in team A's half was scored by B
            a_end = 'left' if side == 'A left' else 'right'
            attacker = 'B' if end == a_end else 'A'
        else:
            p = next((q for q in periods if float(q['start']) <= t <= float(q['end'])), None)
            if p and p.get('attacksRight') in ('A', 'B'):
                attacker = p['attacksRight'] if end == 'right' else ('B' if p['attacksRight'] == 'A' else 'A')
        if attacker and scorer and attacker != scorer:
            if weak:
                ev.append(f"that goal is team {attacker}'s target; the kick-off taker was not certain: team {attacker}")
                return {'t': t, 'team': attacker, 'confidence': 'medium', 'evidence': ev, 'timeUncertain': False,
                        'kickTeam': 'B' if attacker == 'A' else 'A'}
            ev.append(f"that goal is team {attacker}'s target but team {scorer} did not kick off: the team is left open")
            return {'t': t, 'team': None, 'confidence': 'low', 'evidence': ev, 'timeUncertain': False}
        return {'t': t, 'team': scorer or attacker, 'confidence': 'high' if attacker and scorer else 'medium',
                'evidence': ev, 'timeUncertain': False}
    # no goal mouth: the stop before the kick-off, back from the layout over the samples without play
    idx = F.window(lay0-GOAL_LOOKBACK_S, lay0)[::-1]
    t_goal, live_run = lay0, 0.0
    prev_t = lay0
    hidden = False
    for i in idx:
        if F.blind[i]:
            hidden = True             # the picture is not placed before here: the goal may be earlier
            break
        live = int(F.data.possState[i]) != 4 and np.isfinite(F.act[i]) and F.act[i] >= 2.0
        live_run = live_run+(prev_t-float(F.t[i])) if live else 0.0
        prev_t = float(F.t[i])
        if live_run >= 4.0:
            break
        if not live:
            t_goal = float(F.t[i])
    ev.append('the ball was not placed in a goal mouth before it: the time is the start of the stop before the kick-off')
    if hidden:
        ev.append(f'the picture was not placed on the pitch just before {t_goal:.0f} s: the goal may be earlier')
    return {'t': t_goal, 'team': scorer, 'confidence': 'medium' if scorer else 'low', 'evidence': ev, 'timeUncertain': True}


def _timeline(F, stoppages, rng, half, footage=None):
    """Intervals of states over the match range."""
    t = F.t
    n = len(t)
    state = np.array(['IN_PLAY']*n, dtype=object)
    conf = np.array(['medium']*n, dtype=object)
    why = np.array(['']*n, dtype=object)
    # players walking with no dead spell: possible stoppages
    walking = np.where(np.isfinite(F.act), F.act < WALK_MS, False)
    for a, b in _runs(walking, t, WALK_MIN_S):
        sel = (t >= a) & (t <= b)
        state[sel] = 'POSSIBLE_STOPPAGE'
        conf[sel] = 'low'
        why[sel] = 'players walking, the ball not dead'
    for st in stoppages:
        a, b = st['t0'], st['t1']
        sel = (t >= a) & (t < b)
        confirmed = st['restartSeen'] or st.get('restart') is not None
        state[sel] = 'OUT_OF_PLAY' if confirmed else 'POSSIBLE_STOPPAGE'
        conf[sel] = 'high' if (st['restartSeen'] and st.get('restart') is not None) else 'medium' if confirmed else 'low'
        why[sel] = (('goal, then ' if st.get('goal') else '')+(st['restart']['type'].lower().replace('_', ' ')
                    if st.get('restart') else 'a dead ball, no restart seen'))
        r = st.get('restart')
        if r and r['type'] in RESTARTS and r['type'] != 'UNKNOWN_RESTART':
            prep0 = max(a, b-min(PREP_MAX_S, .5*(b-a)))
            if r['type'] == 'KICK_OFF':
                prep0 = max(a, r['t0'])
            psel = (t >= prep0) & (t < b)
            state[psel] = 'RESTART_PREPARATION'
            why[psel] = r['type'].lower().replace('_', ' ')
    # no positions for NON_MATCH_MIN_S: no pitch in the picture (the video says so) is non-match footage; inside a
    # stoppage the match is still stopped; in play it is unknown
    for a, b in _runs(F.blind, t, NON_MATCH_MIN_S):
        sel = (t >= a) & (t <= b)
        seen = (footage or {}).get((round(a, 1), round(b, 1)))
        if seen == 'no pitch':
            state[sel] = 'NON_MATCH_FOOTAGE'
            conf[sel] = 'medium'
            why[sel] = 'no pitch in the picture (crowd, bench, a close-up, graphics)'
        elif np.isin(state[sel], ('OUT_OF_PLAY', 'RESTART_PREPARATION')).mean() >= .5:
            why[sel] = why[sel]+'; the picture is not placed on the pitch here'
        else:
            state[sel] = 'UNKNOWN'
            conf[sel] = 'low'
            why[sel] = 'no positions: the picture could not be placed on the pitch (a close-up, a replay or a gap)'
    if half is not None and half[1] > half[0]:
        sel = (t > half[0]) & (t < half[1])
        state[sel] = 'HALFTIME'
        conf[sel] = 'high'
        why[sel] = 'the halftime you set'
    out = []
    for i in range(n):
        if out and out[-1]['state'] == state[i] and out[-1]['why'] == why[i] and t[i]-out[-1]['t1'] <= 1.0:
            out[-1]['t1'] = float(t[i])
            continue
        out.append({'t0': float(t[i]), 't1': float(t[i]), 'state': state[i], 'confidence': conf[i], 'why': why[i]})
    # close the gaps between samples (owned frames are continuous within a period) and the halftime break
    for a, b in zip(out, out[1:]):
        if b['t0']-a['t1'] <= 1.0:
            a['t1'] = b['t0']
    if half is not None and half[1] > half[0]:
        out = _override(out, half[0], half[1], 'HALFTIME')
        for r in out:
            if r['state'] == 'HALFTIME':
                r['confidence'], r['why'] = 'high', 'the halftime'
    for r in out:
        r['t0'], r['t1'] = round(r['t0'], 2), round(r['t1'], 2)
        r['review'] = r['confidence'] == 'low' and r['state'] != 'IN_PLAY'
    return out


def _stats(timeline, events):
    total = sum(r['t1']-r['t0'] for r in timeline)
    by = Counter()
    for r in timeline:
        by[r['state']] += r['t1']-r['t0']
    stop = sum(by[s] for s in ('OUT_OF_PLAY', 'RESTART_PREPARATION'))
    return {'seconds': round(total, 1), 'byState': {k: round(v, 1) for k, v in by.items()},
            'share': {'IN_PLAY': round(by['IN_PLAY']/max(total, 1e-9), 3), 'STOPPAGE': round(stop/max(total, 1e-9), 3),
                      'POSSIBLE_STOPPAGE': round(by['POSSIBLE_STOPPAGE']/max(total, 1e-9), 3),
                      'UNKNOWN': round((by['UNKNOWN']+by['NON_MATCH_FOOTAGE'])/max(total, 1e-9), 3),
                      'HALFTIME': round(by['HALFTIME']/max(total, 1e-9), 3)},
            'events': dict(Counter(e['type'] for e in events))}


# ---------------------------------------------------------------- masks
STATE_MASK = {'IN_PLAY': 'ACTIVE_OPEN_PLAY', 'RESTART_PREPARATION': 'SET_PIECE_PREPARATION', 'OUT_OF_PLAY': 'OTHER_STOPPAGE',
              'POSSIBLE_STOPPAGE': 'UNKNOWN', 'HALFTIME': 'HALFTIME', 'NON_MATCH_FOOTAGE': 'NON_MATCH_FOOTAGE',
              'UNKNOWN': 'UNKNOWN'}
ELIGIBLE = {
    # what each kind of analysis counts (the rest is kept, not counted)
    'open_play': ('ACTIVE_OPEN_PLAY',),                                    # team shape, formations, line height, coaching
    'movement': ('ACTIVE_OPEN_PLAY', 'SET_PIECE_EXECUTION', 'UNKNOWN'),    # heatmaps, average positions, trails, metrics
    'set_pieces': ('SET_PIECE_PREPARATION', 'SET_PIECE_EXECUTION'),        # a future set-piece analysis
}


def masks(result):
    """[{'t0', 't1', 'mask'}] covering the timeline: the states' masks, SET_PIECE_EXECUTION for EXEC_S after a corner,
    free kick or penalty is taken."""
    out = [{'t0': r['t0'], 't1': r['t1'], 'mask': STATE_MASK.get(r['state'], 'UNKNOWN')} for r in result['timeline']]
    for e in result['events']:
        if e['type'] in SET_PIECES and e.get('status') != 'removed':
            a, b = e['t'], e['t']+EXEC_S
            nxt = []
            for m in out:
                if m['t1'] <= a or m['t0'] >= b or m['mask'] != 'ACTIVE_OPEN_PLAY':
                    nxt.append(m)
                    continue
                if m['t0'] < a:
                    nxt.append(dict(m, t1=a))
                nxt.append(dict(m, t0=max(a, m['t0']), t1=min(b, m['t1']), mask='SET_PIECE_EXECUTION'))
                if m['t1'] > b:
                    nxt.append(dict(m, t0=b))
            out = nxt
    return out


def mask_at(mask_runs, times):
    """The mask name at each time (an array), UNKNOWN outside the runs."""
    times = np.asarray(times, float)
    out = np.array(['UNKNOWN']*len(times), dtype=object)
    if not mask_runs:
        return out
    t0 = np.array([m['t0'] for m in mask_runs])
    t1 = np.array([m['t1'] for m in mask_runs])
    k = np.searchsorted(t0, times, side='right')-1
    ok = (k >= 0) & (times <= t1[np.clip(k, 0, None)]+1e-6)
    names = np.array([m['mask'] for m in mask_runs], dtype=object)
    out[ok] = names[k[ok]]
    return out


def eligible(mask_names, kind):
    """Boolean array: which of these masks the given kind of analysis counts (ELIGIBLE)."""
    allowed = ELIGIBLE[kind]
    return np.isin(np.asarray(mask_names, dtype=object), allowed)


# ---------------------------------------------------------------- corrections (the event review)
def load_corrections(folder):
    try:
        return json.loads((Path(folder)/'assembly'/'intel'/FILE).read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return []


def save_corrections(folder, items):
    from . import manifest as mf
    path = Path(folder)/'assembly'/'intel'/FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    mf.write_json(path, items)


def add_correction(folder, result, body):
    """{kind: confirm|retype|move|remove|add|state, ...} validated against the automatic result; returns the entry."""
    from . import manifest as mf
    kind = body.get('kind')
    ids = {e['id'] for e in result['events']}
    items = load_corrections(folder)
    ids |= {c['event']['id'] for c in items if c.get('kind') == 'add'}
    entry = {'kind': kind}
    if kind in ('confirm', 'remove'):
        if body.get('id') not in ids:
            raise ValueError('No such event.')
        entry['id'] = body['id']
    elif kind == 'retype':
        if body.get('id') not in ids or body.get('type') not in EVENTS:
            raise ValueError('retype needs a known event and a type among '+', '.join(EVENTS)+'.')
        entry |= {'id': body['id'], 'type': body['type']}
    elif kind == 'move':
        if body.get('id') not in ids:
            raise ValueError('No such event.')
        entry |= {'id': body['id'], 't': float(body['t'])}
    elif kind == 'add':
        typ = body.get('type')
        if typ not in EVENTS:
            raise ValueError('add needs a type among '+', '.join(EVENTS)+'.')
        t = float(body['t'])
        team = body.get('team') if body.get('team') in ('A', 'B') else None
        entry['event'] = {'id': f'user-{uuid.uuid4().hex[:8]}', 'type': typ, 't': round(t, 2), 't0': round(t, 2),
                          't1': round(t, 2), 'team': team, 'confidence': 'confirmed', 'evidence': ['added by you'],
                          'source': 'you'}
        for k in ('outPerson', 'inPerson', 'outFragment', 'inFragment', 'slot'):
            if body.get(k) is not None:
                entry['event'][k] = body[k]
    elif kind == 'state':
        if body.get('state') not in STATES:
            raise ValueError('state needs one of '+', '.join(STATES)+'.')
        a, b = float(body['t0']), float(body['t1'])
        if not b > a:
            raise ValueError('The interval must end after it starts.')
        entry |= {'t0': round(a, 2), 't1': round(b, 2), 'state': body['state']}
    else:
        raise ValueError('kind must be confirm, retype, move, remove, add or state.')
    entry |= {'cid': uuid.uuid4().hex[:10], 'at': mf.now()}
    items.append(entry)
    save_corrections(folder, items)
    return entry


def remove_correction(folder, cid):
    items = load_corrections(folder)
    kept = [c for c in items if c.get('cid') != cid]
    if len(kept) == len(items):
        raise ValueError('No such correction.')
    save_corrections(folder, kept)
    return kept


def apply_corrections(result, items):
    """The result with the coach's corrections: events confirmed, retyped, moved, removed (kept with status 'removed'),
    added; intervals set to a state (overriding the automatic ones there). The stats are recomputed."""
    events = {e['id']: dict(e, status='detected' if e['confidence'] != 'low' else 'candidate') for e in result['events']}
    timeline = [dict(r) for r in result['timeline']]
    for c in items or []:
        k = c.get('kind')
        if k == 'add':
            events[c['event']['id']] = dict(c['event'], status='confirmed')
        elif c.get('id') in events:
            e = events[c['id']]
            if k == 'confirm':
                e['status'] = 'confirmed'
            elif k == 'remove':
                e['status'] = 'removed'
            elif k == 'retype':
                e['type'] = c['type']
                e['status'] = 'confirmed'
            elif k == 'move':
                d = c['t']-e['t']
                e['t'], e['t0'], e['t1'] = round(c['t'], 2), round(e['t0']+d, 2), round(e['t1']+d, 2)
                e['status'] = 'confirmed'
        elif k == 'state':
            timeline = _override(timeline, c['t0'], c['t1'], c['state'])
    out = dict(result, events=sorted(events.values(), key=lambda e: e['t']), timeline=timeline, corrections=items or [])
    live = [e for e in out['events'] if e['status'] != 'removed']
    out['stats'] = _stats(timeline, live)
    return out


def _override(timeline, a, b, state):
    out = []
    for r in timeline:
        if r['t1'] <= a or r['t0'] >= b:
            out.append(r)
            continue
        if r['t0'] < a:
            out.append(dict(r, t1=a))
        if r['t1'] > b:
            out.append(dict(r, t0=b))
    out.append({'t0': a, 't1': b, 'state': state, 'confidence': 'confirmed', 'why': 'set by you', 'review': False})
    return sorted(out, key=lambda r: r['t0'])


# ---------------------------------------------------------------- build and cache
def fingerprint(folder, data, halftime_settings, people_fp):
    from . import halftime as ht
    asm = Path(folder)/'assembly'
    h = hashlib.sha256(f'gamestate-{VERSION}'.encode())
    h.update((data.meta.get('fingerprint') or '').encode())
    h.update(ht.fingerprint(halftime_settings).encode())
    h.update((people_fp or '').encode())
    p = asm/'possession.json'
    if p.is_file():
        h.update(hashlib.sha256(p.read_bytes()).digest())
    return h.hexdigest()


def cached(folder, data, possession, halftime_settings, index, people, people_fp, places=None, video=None):
    """analyse() through assembly/intel/state-<fp16>.json."""
    folder = Path(folder)
    fp = fingerprint(folder, data, halftime_settings, people_fp)
    path = folder/'assembly'/'intel'/f'state-{fp[:16]}.json'
    if path.is_file():
        try:
            out = json.loads(path.read_text(encoding='utf-8'))
            if out.get('fingerprint') == fp and out.get('version') == VERSION:
                return out
        except (OSError, ValueError):
            pass
    from . import halftime as ht
    out = analyse(data, possession, halftime_settings, index, people=people, places=places, video=video,
                  footage_cache=folder/'assembly'/'intel'/'footage.json')
    out['fingerprint'] = fp
    out['keys'] = [ht.fingerprint(halftime_settings), people_fp]
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f'.{path.name}.tmp')
    tmp.write_text(json.dumps(out, separators=(',', ':')), encoding='utf-8')
    tmp.replace(path)
    return out
