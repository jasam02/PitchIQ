"""Context of every sample for one team: attacking direction, team coordinates, the phase (who had the ball, as the match's
possession record estimates it) and what the camera saw.

Team coordinates (metres): depth = distance from the team's own goal line along the direction it attacks (0 at its own
goal line, `length` at the opponents'), lateral = metres to the LEFT of the pitch's long axis as the team faces the
opponents' goal (y = 0 is the far touchline, y = width the camera's side, so facing x+ the left is y < width / 2).

Phase per sample and team: 'own' / 'opp' while the possession record names a team (control, flight or contested), 'dead'
while it calls the ball dead, else 'unknown'. 'Settled' (data.possSettled) means well inside one team's spell: the few
seconds around a change are the least certain part of any possession estimate and never decide a judgement here. The
record is an ESTIMATE from the tracking (possession-4 or possession-amateur-1 per segment); the report says so.
"""
import numpy as np

from . import data as md

UNKNOWN, OWN, OPP, DEAD = 0, 1, 2, 3
PHASE_NAMES = ('unknown', 'own', 'opp', 'dead')


class TeamContext:
    """Per-sample context for one team ('A' or 'B') of a MatchData."""

    def __init__(self, data, team):
        self.data, self.team = data, team
        self.code = md.TEAM_CODES[team]
        self.other = 'B' if team == 'A' else 'A'
        n = data.samples
        self.dir = np.array([data.direction(team, s) for s in range(n)], dtype=np.int8) if n < 64 else self._directions()
        pt = data.possTeam
        phase = np.full(n, UNKNOWN, np.int8)
        phase[pt == self.code] = OWN
        phase[(pt != 0) & (pt != self.code)] = OPP
        phase[data.possState == md.POSS_CODES['dead']] = DEAD
        open_play = getattr(data, 'open_play', None)     # the match's state (fullmatch/gamestate.py), when it is known
        if open_play is not None and len(open_play) == n:
            phase[~np.asarray(open_play, bool)] = DEAD
        self.phase = phase
        self.settled = data.possSettled & ((phase == OWN) | (phase == OPP))

    def _directions(self):
        out = np.zeros(self.data.samples, np.int8)
        for p in self.data.periods:
            if p.get('attacksRight') not in ('A', 'B'):
                continue
            sign = 1 if p['attacksRight'] == self.team else -1
            out[self.data.period == int(p['index'])] = sign
        return out

    # ---------------- coordinates
    def depth(self, x, s):
        """Metres from this team's own goal line (along its attack) of pitch x at sample(s) s."""
        d = self.dir[s]
        return np.where(d >= 0, x, self.data.length-np.asarray(x))

    def lateral(self, y, s):
        """Metres to the left of the long axis as this team faces the opponents' goal."""
        return self.dir[s]*(self.data.width/2-np.asarray(y))

    def to_pitch(self, depth, lateral, s):
        """Pitch (x, y) of team coordinates at sample s."""
        d = int(self.dir[s]) or 1
        x = depth if d > 0 else self.data.length-depth
        y = self.data.width/2-lateral/d
        return x, y

    def own_goal(self, s):
        return (0.0 if self.dir[s] >= 0 else self.data.length, self.data.width/2)

    # ---------------- people at a sample
    def players(self, s, outfield=True, team=None):
        """(rows, pids) of this team's (or `team`'s) people at sample s; outfield drops goalkeepers."""
        code = md.TEAM_CODES[team] if team else self.code
        r = self.data.rows(s)
        pid = self.data.pid[r]
        mask = self.data.team_of[pid] == code
        if outfield:
            mask &= ~self.data.is_keeper[pid]
        idx = np.arange(r.start, r.stop)[mask]
        return idx, pid[mask]

    def opponents(self, s, outfield=True):
        return self.players(s, outfield=outfield, team=self.other)

    def keeper(self, s, team=None):
        """Row of the team's goalkeeper at sample s (a keeper fragment without a team is placed by the goal it stands
        nearest to), or -1."""
        code = md.TEAM_CODES[team] if team else self.code
        r = self.data.rows(s)
        pid = self.data.pid[r]
        keep = self.data.is_keeper[pid]
        for i in np.arange(r.start, r.stop)[keep]:
            t = self.data.team_of[self.data.pid[i]]
            if t == code:
                return int(i)
            if t == 0 and self.dir[s] != 0:
                own_x = 0.0 if (self.dir[s] > 0) == (code == self.code) else self.data.length
                if abs(self.data.x[i]-own_x) < self.data.length/2:
                    return int(i)
        return -1

    def phase_at(self, s):
        return PHASE_NAMES[int(self.phase[s])]
