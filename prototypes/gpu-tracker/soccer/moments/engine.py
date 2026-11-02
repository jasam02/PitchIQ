"""The key-moments engine: one MatchData, the context and roles of both teams, the detectors, and the episodes they find.

A detector looks at every live sample (never a dead ball, never an uncalibrated frame) for the situations it knows and
records, per candidate (a team and the player or pair it is about), a STATUS per sample with the measurements behind it.
`runs()` turns those into episodes: consecutive samples of one candidate (gaps up to MAX_GAP_S bridged), at least
MIN_RUN_S long. The detector then classifies each episode (positive / improvement / neutral / insufficient), picks its
key time, and describes the players, measurements, zones and what happened next.
"""
import time
from collections import defaultdict

import numpy as np

from . import context as cx
from . import roles as rl

MAX_GAP_S = 1.0
MIN_RUN_S = 1.0


class Engine:
    def __init__(self, data):
        started = time.perf_counter()
        self.data = data
        self.ctx = {t: cx.TeamContext(data, t) for t in 'AB'}
        self.roles = {t: rl.Roles(data, self.ctx[t]) for t in 'AB'}
        self._role_cache = {}
        self.timings = {'context': round(time.perf_counter()-started, 2)}

    def role(self, team, pid, s):
        """Role record of `pid` (team `team`) at sample s, cached per (pid, 5 s bucket)."""
        key = (int(pid), int(s)//25)
        if key not in self._role_cache:
            self._role_cache[key] = self.roles[team].role_at(pid, float(self.data.t[s]))
        return self._role_cache[key]

    def live(self, team):
        """Samples a detector may look at: calibrated, direction known, not a dead ball."""
        ctx = self.ctx[team]
        return np.flatnonzero(self.data.calibrated & (ctx.dir != 0) & (ctx.phase != cx.DEAD))


def runs(items, max_gap_s=MAX_GAP_S, min_len_s=MIN_RUN_S):
    """Episodes from per-sample items [(key, s, t, record)]: per key, consecutive items with gaps <= max_gap_s, kept when
    they last at least min_len_s. Returns [(key, [records])], ordered by start time."""
    by_key = defaultdict(list)
    for key, s, t, rec in items:
        by_key[key].append((t, s, rec))
    out = []
    for key, seq in by_key.items():
        seq.sort(key=lambda it: it[0])
        cur = [seq[0]]
        for it in seq[1:]:
            if it[0]-cur[-1][0] <= max_gap_s:
                cur.append(it)
            else:
                if cur[-1][0]-cur[0][0] >= min_len_s:
                    out.append((key, [r for _, _, r in cur]))
                cur = [it]
        if cur[-1][0]-cur[0][0] >= min_len_s:
            out.append((key, [r for _, _, r in cur]))
    out.sort(key=lambda kv: kv[1][0]['t'])
    return out


def level_of(value, high, medium):
    return 'high' if value >= high else 'medium' if value >= medium else 'low'


def lower(*levels):
    order = {'low': 0, 'medium': 1, 'high': 2}
    return min(levels, key=lambda v: order[v])
