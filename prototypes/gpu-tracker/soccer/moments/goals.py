"""Goal analysis: how each goal the match state recognised came about, for the team that scored (what it did well) and
for the team that conceded (what went wrong), measured from what the tracking saved. Nothing is re-run.

Inputs: the key-moments MatchData (5 Hz positions, the ball where it was placed), its Engine (team coordinates), every
key-moments episode, the match state's events and timeline (fullmatch/gamestate.py: GOAL events come from the kick-off
that follows a goal, law 10) and the possession v2 record (soccer/possession_v2.py: who had the ball, its changes and
the player on it).

Per goal (analyse()):
    THE END of the attack. The state's goal time is when the ball was last seen in the goal mouth, often seconds after
    it crossed the line (in the net, picked out), so the ball track gives the last STRIKE towards goal before it: the
    ball moving FINAL_MS or faster towards the goal line, within FINAL_WITHIN_M of it, in the FINAL_LOOK_S before the
    state's time; the goal time is estimated from the strike's last fast step. Nothing after the strike is analysed.
    Without a strike the state's time stands.
    PENALTIES: the layout of law 14 held PEN_LAYOUT_S or more at the goal the team attacks (one person on the goal line
    inside the goal, one within TAKER_M of the penalty mark, every other placed person, OTHERS_MIN or more, outside the
    penalty area and 9.15 m or more from the mark, PEN_SLACK_M allowed for position error), or the ball still on that
    mark before a restart. The play that won it: the scoring team's last loss of the ball to the other team (possession
    v2) in the FOUL_LOOK_S before, where the whistle ended its possession.
    THE START of any other attack, the latest of: the restart that began it (a stop of the match state ending within
    LOOKBACK_S, its type from the restart event there; a set piece is the scoring team's own corner, free kick or
    throw-in within SET_PIECE_S; a goal within DIRECT_S of a restart that ended a stop of DIRECT_STOP_S or more came
    straight from it, a penalty or a direct free kick when neither the layout nor the ball tells which), the scoring
    team's last gain of possession (possession v2's change to it, from when the other team lost the ball unless it was
    loose longer than TURNOVER_GAP_S, or the conceding team's possession followed within TURNOVER_GAP_S by the scoring
    team's) and the end of the conceding team's last possession after that gain (how the
    ball was won is then not decoded). None within LOOKBACK_S: a 'sustained' attack from LOOKBACK_S before the goal or,
    with too little placed to tell, a goal whose making was not seen (kind 'unknown': the scoring team's last
    possession before it is named).
    QUALITY: the share of the build-up's samples with both teams placed (MIN_OUTFIELD outfield players or more each) and
    the share with the ball placed; under MIN_QUALITY nothing is measured and the goal says why.
    FACTS: the time from the start to the goal; where the ball was won (thirds and channels in the team's direction);
    how far it moved towards goal; the players possession v2 put on the ball, and the last one; at the strike, the
    players in and around the box (attackers, defenders, the keeper), the nearest outfield defender to the ball and the
    keeper's distance from the goal line; the conceding team's outfield players goal-side of the ball when the attack
    started and RECOVER_S later; the distance of the nearest defender to the ball over the build-up (pressure) and to
    the last player on it (marking); the key-moments episodes inside the build-up (the scoring team's positive ones, the
    conceding team's improvement ones).
    FINDINGS: sentences from those facts, each with the time to watch, its source (tracking, possession estimate, match
    state, key moment) and how sure it is; a fact the data cannot support is left out, never guessed. There is no ball
    height and no pass event: nothing here names a scorer or an assist, and the strike is the ball's last fast move
    towards goal, which a deflection or a shot the tracking lost can hide.
"""
import math

import numpy as np

LOOKBACK_S = 30.0         # the build-up looked at goes back at most this far
SHOT_BEFORE_S = 1.0       # without a strike, the box is counted this long before the goal time
RECOVER_S = 5.0           # goal-side defenders are counted again this long after the attack started
MIN_QUALITY = .5          # a build-up with both teams placed on fewer of its samples is not analysed
MIN_OUTFIELD = 3          # outfield players of a team that must be placed for a sample to count as placed (the camera
                          # shows the play around the ball near a goal, not every outfield player)
PEN_DEPTH, PEN_HALF = 16.5, 20.16
GOAL_HALF = 3.66
BOX_MARGIN = 2.0          # metres around the penalty area that still count as in the box (position noise)
PENALTY_MARK_M = 11.0     # the penalty mark's distance from the goal line ...
PENALTY_NEAR_M = 2.5      # ... a ball this close to it, still, before the restart: a penalty
SET_PIECE_S = 15.0        # a restart this long or less before the goal is the goal's set piece
DIRECT_S = 4.0            # a goal this soon after a restart that followed a long stop (DIRECT_STOP_S) came straight from it:
DIRECT_STOP_S = 20.0      # a penalty or a direct free kick, when neither the layout nor the ball tells which
FINAL_MS = 12.0           # the ball moving this fast or faster towards the goal line ...
FINAL_WITHIN_M = 35.0     # ... from this close to it or closer: a strike towards goal
FINAL_LOOK_S = 6.0        # the strike is looked for this long before the state's goal time
FINAL_AFTER_S = 2.0       # the estimated goal time is at most this long after the strike's last fast step
PEN_LAYOUT_S = 1.0        # the penalty layout held this long or longer (CAN-ISL: its two penalties 5.2 and 1.2 s, any
                          # other moment of the match 0.8 s or less)
TAKER_M = 5.0             # the taker within this of the penalty mark
LINE_M = 2.5              # the keeper within this of the goal line
OTHERS_MIN = 4            # placed people besides the keeper and the taker for a layout to be told
PEN_SLACK_M = 1.5         # position error allowed at the penalty area's edge and at the 9.15 m
PEN_DISTANCE_M = 9.15
FOUL_LOOK_S = 300.0       # the play that won a penalty is looked for this far back
TURNOVER_GAP_S = 4.0      # the conceding team's possession followed this soon by the scoring team's: the ball changed hands
UNMARKED_M = 3.0          # the last player on the ball with no defender this close was free
NO_PRESSURE_M = 5.0       # the nearest defender to the ball this far on average: no pressure
STRIKE_FREE_M = 4.0       # no outfield defender this close to the ball when it was struck: nobody closed it down
KEEPER_OFF_M = 6.0        # the keeper this far or farther off the goal line at the strike is said
FEW_GOAL_SIDE = 5         # this many outfield players or fewer between the ball and goal: the defence was short ...
GOAL_SIDE_FROM_M = 30.0   # ... counted only with the ball this far or more from that goal (near it nobody can be goal-side)
GOAL_SIDE_SEEN = 8        # ... and this many of the team's outfield players in view (the camera does not show everyone)
LAST_SEEN_S = 120.0       # a goal whose making was not seen: the scoring team's last possession this far back is named
RESTART_WORDS = {'KICK_OFF': 'kick-off', 'CORNER_KICK': 'corner', 'GOAL_KICK': 'goal kick', 'THROW_IN': 'throw-in',
                 'FREE_KICK': 'free kick', 'PENALTY_KICK': 'penalty', 'UNKNOWN_RESTART': 'restart'}
STOPS = ('OUT_OF_PLAY', 'RESTART_PREPARATION', 'HALFTIME', 'NON_MATCH_FOOTAGE')
# what a key-moments episode says, in a few words (the report's own moment text says more)
EPISODE_WORDS = {
    'run_in_behind': 'ran in behind the defence', 'created_separation': 'moved away from a marker into space',
    'offered_support': 'offered a supporting option to the ball carrier', 'free_between_lines': 'was free between the lines',
    'press_and_cover': 'pressed the ball with cover behind', 'stepped_up_together': 'the back line stepped up together',
    'stepped_out_covered': 'a defender stepped out with cover', 'rotated_to_cover': 'a team-mate rotated to cover the space',
    'covered_advance': 'a wide player advanced with cover behind', 'recovery_run': 'made a recovery run',
    'gap_attacker': 'a gap in the back line an attacker could use', 'stepped_out_open': 'a defender stepped out and left space behind',
    'dropped_off': 'the back line dropped off', 'carrier_isolated': 'the ball carrier had no support',
    'exposed_channel': 'the wide channel was left exposed', 'unseen_channel': 'the wide channel may have been exposed (partly out of view)',
    'press_no_cover': 'pressed the ball with nobody covering behind', 'double_press_free': 'two players pressed the same opponent and left one free'}


def _num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _a(word):
    """'a corner', 'an unknown restart'."""
    return ('an ' if word[:1] in 'aeiou' else 'a ')+word


def _n(k, word):
    """'1 defender', '3 defenders'."""
    return f'{k} {word}' + ('' if k == 1 else 's')


def _zone(depth, lateral, length, width):
    """(third, channel) of team coordinates: 'own third' / 'middle third' / 'attacking third', 'left' / 'centre' / 'right'."""
    third = 'own third' if depth < length/3 else ('middle third' if depth < 2*length/3 else 'attacking third')
    channel = 'centre' if abs(lateral) < width/6 else ('left' if lateral > 0 else 'right')
    return third, channel


def _restart_before(timeline, events, t_goal, lookback):
    """(end of the last stop before the goal within lookback, its restart event or None)."""
    best = None
    for r in timeline or []:
        if r.get('state') in STOPS and t_goal-lookback <= float(r['t1']) <= t_goal-.3:
            if best is None or float(r['t1']) > best:
                best = float(r['t1'])
    if best is None:
        return None, None
    ev = [e for e in events or [] if e.get('type') in RESTART_WORDS and e.get('status') != 'removed' and abs(float(e['t'])-best) <= 3.0]
    ev.sort(key=lambda e: abs(float(e['t'])-best))
    return best, (ev[0] if ev else None)


def penalty_layout(data, lo_t, hi_t, goal_x):
    """(setup start, kick) of the last spell between lo_t and hi_t in which the people placed stand as law 14 has them
    for a penalty at the goal at x = goal_x (0 or the pitch length), held PEN_LAYOUT_S or more, or None: one person on
    the goal line inside the goal, one within TAKER_M of the penalty mark, every other placed person (OTHERS_MIN or more)
    outside the penalty area and 9.15 m or more from the mark (PEN_SLACK_M allowed for position error). The kick is when
    the layout breaks. fullmatch/gamestate.py tells the goal's penalty the same way."""
    L, W = float(data.length), float(data.width)
    sign = 1.0 if goal_x > L/2 else -1.0                # towards that goal
    mark_x = goal_x-sign*PENALTY_MARK_M
    spell, best = None, None
    for s in range(max(0, data.sample_at(lo_t)), min(len(data.t)-1, data.sample_at(hi_t))+1):
        ok = False
        r = data.rows(s)
        if data.calibrated[s] and r.stop-r.start >= OTHERS_MIN+2:
            x, y = data.x[r].astype(float), data.y[r].astype(float)
            front = sign*(goal_x-x)                      # metres in front of the goal line (negative behind it)
            lat = y-W/2
            line = (front <= LINE_M) & (np.abs(lat) <= GOAL_HALF+1.0)
            mark = np.hypot(x-mark_x, lat)
            taker = (mark <= TAKER_M) & ~line
            others = ~line & ~taker
            outside = (((front >= PEN_DEPTH-PEN_SLACK_M) | (np.abs(lat) >= PEN_HALF-PEN_SLACK_M))
                       & (mark >= PEN_DISTANCE_M-PEN_SLACK_M))
            ok = line.sum() == 1 and taker.sum() == 1 and others.sum() >= OTHERS_MIN and bool(np.all(outside[others]))
        if ok:
            spell = (spell[0] if spell else float(data.t[s]), float(data.t[s]))
            continue
        if spell and spell[1]-spell[0] >= PEN_LAYOUT_S:
            best = spell
        spell = None
    if spell and spell[1]-spell[0] >= PEN_LAYOUT_S:
        best = spell
    return best


def _possession(rec2):
    """Columns of the possession v2 timeline and its people, or None."""
    if not rec2 or not isinstance(rec2.get('timeline'), dict):
        return None
    tl = rec2['timeline']
    return {k: np.asarray(tl.get(k) or [], float) for k in ('t0', 't1', 'state', 'designation', 'possessor')}, tl.get('people') or []


def _run_person(cols, people, i):
    k = int(cols['possessor'][i])
    p = people[k] if 0 <= k < len(people) else None
    return {'who': p.get('id') if p else None, 'label': p.get('label') if p else None}


class Goal:
    """One goal's analysis (see the module doc). Findings keep person indices of the MatchData; text() words them."""

    def __init__(self, data, engine, eps_all, rec2, event, timeline, events):
        self.data, self.engine = data, engine
        self.event = event
        self.t = float(event['t'])                # the match state's time (the timeline's marker)
        self.team = event.get('team') if event.get('team') in ('A', 'B') else None
        self.conceding = None if self.team is None else ('B' if self.team == 'A' else 'A')
        self.facts, self.notes = {}, []
        self.kind = 'unknown'
        self.start, self.start_kind, self.restart, self.gain, self.foul = None, None, None, None, None
        self.t_strike, self.t_in = None, self.t     # the strike towards goal (ball track) and the estimated goal time
        self.quality = None
        if self.team is None:
            self.notes.append('Which team scored is not known, so the goal is not analysed.')
            return
        self._strike()
        self._locate(rec2, timeline, events)
        self._measure(eps_all, rec2)

    @property
    def t_end(self):
        """The end of the attack: the strike, else the (estimated) goal time."""
        return self.t_strike if self.t_strike is not None else self.t_in

    # ---------------- the end of the attack
    def _strike(self):
        """The ball's last fast move towards the goal the team attacks before the state's goal time, and the goal time
        estimated from it (see the module doc)."""
        d, ctx = self.data, self.engine.ctx[self.team]
        L = float(d.length)
        lo, hi = d.sample_at(self.t-FINAL_LOOK_S), d.sample_at(self.t+.5)
        ss = [s for s in range(lo, hi+1) if np.isfinite(d.ball[s, 0]) and d.calibrated[s] and ctx.dir[s] != 0]
        first = last = run = None
        for a, b in zip(ss, ss[1:]):
            dt = float(d.t[b]-d.t[a])
            fast = False
            if 0 < dt <= .45:
                da, db = (float(ctx.depth(float(d.ball[s, 0]), s)) for s in (a, b))
                v = (db-da)/dt
                fast = v >= FINAL_MS and L-da <= FINAL_WITHIN_M
            if fast:
                run = a if run is None else run
                first, last = run, (b, db, v)
            else:
                run = None
        if first is None:
            return
        b, db, v = last
        tb = float(d.t[b])
        self.t_strike = float(d.t[first])
        self.t_in = min(tb+max(0.0, L-db)/v, tb+FINAL_AFTER_S)

    # ---------------- where the decisive attack began
    def _locate(self, rec2, timeline, events):
        team = self.team
        layout = self._penalty_layout(self.t-LOOKBACK_S, self.t+3.0)
        if layout is not None:
            setup, kick = layout
            self.kind, self.start, self.start_kind = 'penalty', setup, 'restart'
            self.restart = {'t': round(kick, 2), 'type': 'PENALTY_KICK', 'team': team, 'how': 'layout'}
            self.t_strike, self.t_in = None, kick
            self.foul = self._foul(rec2, setup)
            return
        end = self.t_end
        stop_end, rev = _restart_before(timeline, events, self.t_in, LOOKBACK_S)
        gain = None
        for c in (rec2 or {}).get('changes') or []:
            if (c.get('to') == team and c.get('supported', True) and _num(c.get('wonAt'))
                    and end-LOOKBACK_S <= float(c['wonAt']) <= end and (gain is None or float(c['wonAt']) > float(gain['wonAt']))):
                gain = c
        cands = []
        if stop_end is not None:
            cands.append((stop_end, 'restart', None))
        if gain is not None:
            # the attack began where the conceding team lost the ball, unless the ball was loose long before it was won
            lost = float(gain['lostAt']) if _num(gain.get('lostAt')) else float(gain['wonAt'])
            won = float(gain['wonAt'])
            cands.append((lost if 0 <= won-lost <= TURNOVER_GAP_S else won, 'turnover', gain))
        run = self._conceding_run(rec2, end-LOOKBACK_S, end+.01)
        if run is not None and (gain is None or run['t1'] > float(gain['wonAt'])):
            nxt = run['next']
            if nxt is not None:
                # the conceding team's possession, then the scoring team's: the ball changed hands there
                cands.append((run['t1'], 'turnover', {'from': self.conceding, 'to': team, 'lostAt': run['t1'], 'wonAt': nxt['t0'],
                                                      'lostBy': run['who'], 'lostByLabel': run['label'], 'wonBy': nxt['who'],
                                                      'wonByLabel': nxt['label'], 'confidence': 'estimated', 'implied': True}))
            else:
                # possession v2 named the conceding team on the ball after the scoring team's last gain: the attack
                # began when it lost control (how and to whom is not decoded)
                cands.append((run['t1'], 'loss', None))
        if cands:
            self.start, self.start_kind, payload = max(cands, key=lambda c: c[0])
        else:
            self.start, self.start_kind, payload = max(end-LOOKBACK_S, float(self.data.t[0])), 'sustained', None
        self.gain = payload if self.start_kind == 'turnover' else None
        self.kind = 'open_play'
        if self.start_kind == 'restart':
            rtype = (rev or {}).get('type') or 'UNKNOWN_RESTART'
            rteam = (rev or {}).get('team')
            if self._penalty_spot(stop_end) and self.t_in-stop_end <= 6.0:
                rtype = 'PENALTY_KICK'
            self.restart = {'t': round(stop_end, 2), 'type': rtype, 'team': rteam}
            if rtype == 'PENALTY_KICK':
                self.kind = 'penalty'
                self.restart['how'] = 'ball'
                self.foul = self._foul(rec2, stop_end)
            elif rtype in ('CORNER_KICK', 'FREE_KICK', 'THROW_IN') and self.t_in-stop_end <= SET_PIECE_S and rteam in (team, None):
                self.kind = 'set_piece'        # the opponents' restart followed by a goal is a counter, not a set piece
            elif rtype == 'KICK_OFF':
                self.kind = 'kick_off'
        if self.kind in ('open_play', 'set_piece') and self.restart and self.restart['type'] in ('UNKNOWN_RESTART', 'FREE_KICK'):
            stop = self._stop_length(timeline, self.restart['t'])
            if self.t_in-self.restart['t'] <= DIRECT_S and stop >= DIRECT_STOP_S:
                self.kind = 'direct_set_piece'
                self.restart['stopSeconds'] = round(stop, 1)
                self.foul = self._foul(rec2, self.restart['t'])

    @staticmethod
    def _stop_length(timeline, t_restart):
        """Seconds of the contiguous spell before t_restart in which play was not seen running (stops, possible stops,
        no positions, footage that is not the match)."""
        rows = sorted((r for r in timeline or [] if float(r['t1']) <= t_restart+.5), key=lambda r: float(r['t0']))
        start = t_restart
        for r in reversed(rows):
            if r.get('state') == 'IN_PLAY':
                break
            start = min(start, float(r['t0']))
        return t_restart-start

    def _conceding_run(self, rec2, lo, hi):
        """The conceding team's last named possession run ending in (lo, hi): {'t1', 'who', 'label', 'next'}, 'next' the
        scoring team's run that began within TURNOVER_GAP_S after it ({'t0', 'who', 'label'}) or None; None without one."""
        p = _possession(rec2)
        if p is None:
            return None
        cols, people = p
        own, other = (0, 1) if self.conceding == 'A' else (1, 0)
        idx = np.flatnonzero((cols['state'] == own) & (cols['t1'] > lo) & (cols['t1'] < hi))
        if not len(idx):
            return None
        i = int(idx[-1])
        out = dict(_run_person(cols, people, i), t1=float(cols['t1'][i]), next=None)
        j = np.flatnonzero((cols['state'] == other) & (cols['t0'] >= out['t1']-.05) & (cols['t0'] <= out['t1']+TURNOVER_GAP_S)
                           & (cols['t0'] < hi))
        if len(j):
            k = int(j[0])
            out['next'] = dict(_run_person(cols, people, k), t0=float(cols['t0'][k]))
        return out

    def _penalty_spot(self, t):
        """Was the ball placed, still, within PENALTY_NEAR_M of the penalty mark of the goal the scoring team attacks in
        the 3 s before t?"""
        d = self.data
        L = float(d.length)
        lo, hi = d.sample_at(t-3.0), d.sample_at(t)
        ss = np.arange(lo, hi+1)
        ss = ss[np.isfinite(d.ball[ss, 0])]
        if len(ss) < 3:
            return False
        ctx = self.engine.ctx[self.team]
        if not np.all(ctx.dir[ss] != 0):
            return False
        dep = ctx.depth(d.ball[ss, 0].astype(float), ss)
        lat = ctx.lateral(d.ball[ss, 1].astype(float), ss)
        near = np.hypot(dep-(L-PENALTY_MARK_M), lat) <= PENALTY_NEAR_M
        return bool(near.mean() >= .6)

    def _penalty_layout(self, lo_t, hi_t):
        """(setup start, kick) of the last penalty layout at the goal the scoring team attacks between lo_t and hi_t
        (penalty_layout), or None."""
        d, ctx = self.data, self.engine.ctx[self.team]
        direction = int(ctx.dir[min(len(d.t)-1, max(0, d.sample_at(self.t)))])
        if direction == 0:
            return None
        return penalty_layout(d, lo_t, hi_t, float(d.length) if direction > 0 else 0.0)

    def _foul(self, rec2, before):
        """The play that won the penalty (or the free kick): the scoring team's last loss of the ball to the conceding
        team (possession v2) in the FOUL_LOOK_S before `before`, and where the ball was then; None without one."""
        best = None
        for c in (rec2 or {}).get('changes') or []:
            if (c.get('from') == self.team and c.get('to') == self.conceding and c.get('supported', True) and _num(c.get('lostAt'))
                    and before-FOUL_LOOK_S <= float(c['lostAt']) <= before and (best is None or float(c['lostAt']) > float(best['lostAt']))):
                best = c
        if best is None:
            return None
        t = float(best['lostAt'])
        out = {'t': round(t, 2), 'who': best.get('lostBy'), 'label': best.get('lostByLabel')}
        s = self._ball_near(t, 1.0)
        if s is not None:
            out['ball'] = self._ball_place(s)
        return out

    # ---------------- measurements
    def _rows(self, s, team, outfield=True):
        return self.engine.ctx[self.team].players(s, outfield=outfield, team=team)

    def _measure(self, eps_all, rec2):
        d, team, opp = self.data, self.team, self.conceding
        ctx_a = self.engine.ctx[team]
        L, W = float(d.length), float(d.width)
        end = self.t_end
        s0, s1 = d.sample_at(self.start), d.sample_at(end)
        s1 = max(s0, s1)
        ss = np.arange(s0, s1+1)
        placed = []
        for s in ss:
            na = len(self._rows(s, team)[0])
            nd = len(self._rows(s, opp)[0])
            placed.append(bool(d.calibrated[s]) and na >= MIN_OUTFIELD and nd >= MIN_OUTFIELD)
        placed = np.asarray(placed, bool)
        ball_ok = np.isfinite(d.ball[ss, 0])
        self.quality = {'samples': int(len(ss)), 'placed': round(float(placed.mean()), 2) if len(ss) else 0.0,
                        'ball': round(float(ball_ok.mean()), 2) if len(ss) else 0.0}
        self.facts['seconds'] = round(self.t_in-self.start, 1)
        self.facts['goalTime'] = round(self.t_in, 2)
        if self.foul:
            self.facts['foul'] = self.foul
        if self.t_strike is not None:
            self.facts['strike'] = {'t': round(self.t_strike, 2)}
        if self.kind in ('penalty', 'direct_set_piece'):
            return                         # the attack is the kick: the play that won it is the foul above
        if self.start_kind == 'sustained' and self.quality['placed'] < MIN_QUALITY:
            self.kind = 'unknown'
            last = self._last_seen(rec2)
            if last:
                self.facts['lastSeen'] = last
        if not len(ss) or self.quality['placed'] < MIN_QUALITY:
            self.notes.append(f"The tracking placed both teams on {round(100*self.quality['placed'])}% of the "
                              f"{self.facts['seconds']:.0f} s before this goal: too little to analyse the build-up.")
            return
        # where the attack began, and how far the ball travelled towards goal
        sb = self._ball_sample(s0, s0+int(2*5))
        if sb is not None:
            bx, by = (float(v) for v in d.ball[sb])
            depth, lat = float(ctx_a.depth(bx, sb)), float(ctx_a.lateral(by, sb))
            third, channel = _zone(depth, lat, L, W)
            self.facts['startZone'] = {'third': third, 'channel': channel, 'depthM': round(depth, 1), 'lateralM': round(lat, 1),
                                       'fromGoalM': round(L-depth, 1)}
            se = self._ball_sample(max(s0, s1-int(3*5)), s1, last=True)
            if se is not None:
                depth_end = float(ctx_a.depth(float(d.ball[se, 0]), se))
                self.facts['progressionM'] = round(depth_end-depth, 1)
        # the players possession v2 put on the ball
        self._possessors(rec2)
        # the box and the ball when it was struck (else just before the goal)
        sshot = d.sample_at(self.t_strike if self.t_strike is not None else max(self.start, self.t_in-SHOT_BEFORE_S))
        box = self._box(sshot)
        if box:
            self.facts['box'] = box
        if self.t_strike is not None:
            self.facts['strike'].update(self._at_strike(sshot))
        # goal-side defenders when the attack started and RECOVER_S later
        g0 = self._goal_side(s0)
        g5 = self._goal_side(min(s1, d.sample_at(self.start+RECOVER_S)))
        if g0:
            self.facts['goalSideStart'] = g0
        if g5 and end-self.start > RECOVER_S+1:
            self.facts['goalSideLater'] = g5
        # pressure on the ball over the build-up
        dists = []
        for s in ss:
            if not np.isfinite(d.ball[s, 0]):
                continue
            rows, _ = self._rows(s, opp)
            if len(rows):
                dists.append(float(np.min(np.hypot(d.x[rows]-d.ball[s, 0], d.y[rows]-d.ball[s, 1]))))
        if len(dists) >= 5:
            self.facts['pressureM'] = {'median': round(float(np.median(dists)), 1), 'samples': len(dists)}
        # marking of the last player on the ball
        last = self.facts.get('lastOnBall')
        if last and last.get('pid') is not None:
            s = d.sample_at(last['t'])
            me = self._row_of(s, last['pid'])
            if me is not None:
                rows, _ = self._rows(s, opp, outfield=False)
                if len(rows):
                    m = float(np.min(np.hypot(d.x[rows]-d.x[me], d.y[rows]-d.y[me])))
                    last['nearestDefenderM'] = round(m, 1)
        # the key-moments episodes inside the build-up
        lo, hi = self.start-2.0, end+1.0
        for e in eps_all or []:
            if not (lo <= float(e.get('tKey', -1)) <= hi):
                continue
            if e['team'] == team and e.get('category') == 'positive':
                self.facts.setdefault('goodEpisodes', []).append(self._ep(e))
            elif e['team'] == opp and e.get('category') == 'improvement':
                self.facts.setdefault('wrongEpisodes', []).append(self._ep(e))

    def _ep(self, e):
        focus = (e.get('players') or {}).get('focus') or []
        pid = self.data.index_of.get(focus[0]) if focus else None
        return {'id': e.get('id'), 'detector': e.get('detector'), 'kind': e.get('kind'), 't': float(e.get('tKey')),
                'pid': pid, 'reliability': e.get('reliability'), 'interpretation': e.get('interpretation')}

    def _ball_sample(self, lo, hi, last=False):
        rng = range(hi, lo-1, -1) if last else range(lo, hi+1)
        for s in rng:
            if 0 <= s < len(self.data.t) and np.isfinite(self.data.ball[s, 0]) and self.data.calibrated[s]:
                return s
        return None

    def _ball_near(self, t, within):
        """The calibrated sample with the ball placed nearest to t, within `within` seconds, or None."""
        d = self.data
        best = None
        for s in range(max(0, d.sample_at(t-within)), min(len(d.t)-1, d.sample_at(t+within))+1):
            if np.isfinite(d.ball[s, 0]) and d.calibrated[s] and (best is None or abs(d.t[s]-t) < abs(d.t[best]-t)):
                best = s
        return best

    def _ball_place(self, s):
        """Where the ball was at sample s, from the goal the scoring team attacks."""
        d, ctx = self.data, self.engine.ctx[self.team]
        L = float(d.length)
        dep = float(ctx.depth(float(d.ball[s, 0]), s))
        lat = float(ctx.lateral(float(d.ball[s, 1]), s))
        return {'fromGoalM': round(L-dep, 1), 'lateralM': round(lat, 1),
                'inBox': bool(L-dep <= PEN_DEPTH and abs(lat) <= PEN_HALF)}

    def _row_of(self, s, pid):
        r = self.data.rows(s)
        hit = np.flatnonzero(self.data.pid[r] == pid)
        return int(r.start+hit[0]) if len(hit) else None

    def _possessors(self, rec2):
        p = _possession(rec2)
        if p is None:
            return
        cols, people = p
        code = 0 if self.team == 'A' else 1
        sel = (cols['t1'] > self.start) & (cols['t0'] < self.t_end) & (cols['state'] == code) & (cols['possessor'] >= 0)
        order, seen = [], set()
        last = None
        for i in np.flatnonzero(sel):
            person = people[int(cols['possessor'][i])]
            pid = self.data.index_of.get(person['id'])
            if person['id'] not in seen:
                seen.add(person['id'])
                order.append({'id': person['id'], 'label': person.get('label'), 'pid': pid})
            last = {'id': person['id'], 'label': person.get('label'), 'pid': pid, 't': float(min(cols['t1'][i], self.t_end))}
        if order:
            self.facts['onBall'] = order
        if last:
            self.facts['lastOnBall'] = last
        if self.gain:
            g = self.gain
            self.facts['gain'] = {'lostAt': g.get('lostAt'), 'wonAt': g.get('wonAt'), 'lostBy': g.get('lostByLabel'),
                                  'lostById': g.get('lostBy'), 'wonBy': g.get('wonByLabel'), 'wonById': g.get('wonBy'),
                                  'confidence': g.get('confidence'), 'implied': bool(g.get('implied'))}

    def _box(self, s):
        """Attackers and defenders in and around the penalty area the goal was scored in, at sample s."""
        d = self.data
        if not d.calibrated[s]:
            return None
        ctx_a = self.engine.ctx[self.team]
        L = float(d.length)
        rows_a, _ = self._rows(s, self.team)
        rows_d, _ = self._rows(s, self.conceding, outfield=False)
        if len(rows_a) < 3 or len(rows_d) < 3:
            return None

        def inside(rows):
            dep = ctx_a.depth(d.x[rows], s)
            lat = ctx_a.lateral(d.y[rows], s)
            return int(np.sum((dep >= L-PEN_DEPTH-BOX_MARGIN) & (np.abs(lat) <= PEN_HALF+BOX_MARGIN)))
        return {'attackers': inside(rows_a), 'defenders': inside(rows_d), 't': round(float(d.t[s]), 1)}

    def _at_strike(self, s):
        """The ball's place, the nearest outfield defender to it and the keeper's distance from the goal line at the
        strike (sample s)."""
        d = self.data
        out = {}
        if not (d.calibrated[s] and np.isfinite(d.ball[s, 0])):
            return out
        ctx_a = self.engine.ctx[self.team]
        L = float(d.length)
        out.update(self._ball_place(s))
        rows, _ = self._rows(s, self.conceding)
        if len(rows) >= MIN_OUTFIELD:
            out['nearestDefenderM'] = round(float(np.min(np.hypot(d.x[rows]-d.ball[s, 0], d.y[rows]-d.ball[s, 1]))), 1)
        k = self.engine.ctx[self.conceding].keeper(s)
        if k >= 0:
            out['keeperOffLineM'] = round(float(L-ctx_a.depth(float(d.x[k]), s)), 1)
        return out

    def _goal_side(self, s):
        """{goalSide, outfield}: the conceding team's outfield players between the ball and their own goal at sample s."""
        d = self.data
        if not d.calibrated[s] or not np.isfinite(d.ball[s, 0]):
            return None
        ctx_d = self.engine.ctx[self.conceding]
        rows, _ = self._rows(s, self.conceding)
        if len(rows) < MIN_OUTFIELD:
            return None
        ball_depth = float(ctx_d.depth(float(d.ball[s, 0]), s))
        dep = ctx_d.depth(d.x[rows], s)
        return {'goalSide': int(np.sum(dep < ball_depth)), 'outfield': int(len(rows)), 't': round(float(d.t[s]), 1),
                'ballFromGoalM': round(ball_depth, 1)}

    def _last_seen(self, rec2):
        """The scoring team's last named possession in the LAST_SEEN_S before the goal: {'t', 'who', 'label', 'gapS',
        'ball'}, or None."""
        p = _possession(rec2)
        if p is None:
            return None
        cols, people = p
        code = 0 if self.team == 'A' else 1
        idx = np.flatnonzero((cols['state'] == code) & (cols['t1'] > self.t-LAST_SEEN_S) & (cols['t1'] <= self.t))
        if not len(idx):
            return None
        i = int(idx[-1])
        t = float(cols['t1'][i])
        out = dict(_run_person(cols, people, i), t=round(t, 2), gapS=round(self.t_in-t, 1))
        s = self._ball_near(t, 1.0)
        if s is not None:
            out['ball'] = self._ball_place(s)
        return out

    # ---------------- words
    def _who(self, names, person_id, label):
        """A person in the report's words (its own players by label, the opponents by kit), else the record's label."""
        pid = self.data.index_of.get(person_id) if person_id is not None else None
        if pid is not None:
            return names.person(pid, with_role=False)
        return label or 'a player'

    @staticmethod
    def _where(place):
        """' in the penalty area, about 5 m from goal' / ' about 25 m from goal' / ''."""
        if not place:
            return ''
        if place.get('inBox'):
            return f" in the penalty area, about {place['fromGoalM']:.0f} m from goal"
        return f" about {place['fromGoalM']:.0f} m from goal"

    def text(self, names, clock):
        """(scored findings, conceded findings): [{'text', 't', 'source', 'sure'}] in the report's words."""
        f, scored, conceded = self.facts, [], []
        tw, ow = names.team_word(self.team), names.team_word(self.conceding)
        secs = f.get('seconds')
        start_t = self.start
        foul = f.get('foul')
        if self.kind in ('penalty', 'direct_set_piece'):
            if self.kind == 'penalty':
                how = ('the players set out for a penalty' if self.restart.get('how') == 'layout'
                       else 'the ball still on the penalty mark')
                scored.append((f'Scored from the penalty spot ({how}).', self.restart['t'], 'tracking', 'fairly sure'))
                what = 'a penalty'
            else:
                scored.append((f'Scored {secs:.0f} s after a restart that ended a long stop: straight from a penalty or a direct '
                               f'free kick (the tracking could not place the ball or the players to tell which).',
                               self.restart['t'], 'match state', 'fairly sure'))
                what = 'a penalty or a direct free kick'
            if foul:
                who, where = self._who(names, foul.get('who'), foul.get('label')), self._where(foul.get('ball'))
                scored.append((f"Won {what}: the attack that earned it stopped at {clock(foul['t'])} with {who} on the ball{where}.",
                               foul['t'], 'possession estimate', 'estimated'))
                conceded.append((f"Gave away {what} at {clock(foul['t'])}: {who} had the ball{where} when the attack stopped. Watch "
                                 f"what happened there.", foul['t'], 'possession estimate', 'estimated'))
            else:
                conceded.append((f'Gave away {what}. The tracking did not place the play that won it: watch the play before the '
                                 f'stop.', self.restart['t'], 'match state', 'fairly sure'))
            return self._words(scored), self._words(conceded)
        if self.kind == 'unknown':
            last = f.get('lastSeen')
            if last:
                who, where = self._who(names, last.get('who'), last.get('label')), self._where(last.get('ball'))
                line = (f"The play before this goal was not seen (the tracking placed too little of it). {tw} last had the ball "
                        f"at {clock(last['t'])} ({who}{where}), {last['gapS']:.0f} s before the goal.")
                scored.append((line, last['t'], 'possession estimate', 'estimated'))
                conceded.append((line, last['t'], 'possession estimate', 'estimated'))
            return self._words(scored), self._words(conceded)
        if self.kind == 'set_piece':
            w = RESTART_WORDS.get(self.restart['type'], 'restart')
            scored.append((f'Scored {secs:.0f} s after {_a(w)}.', self.restart['t'], 'match state', 'fairly sure'))
            conceded.append((f'Conceded {secs:.0f} s after {_a(w)} against it.', self.restart['t'], 'match state', 'fairly sure'))
        elif self.start_kind == 'turnover' and 'gain' in f:
            z = f.get('startZone')
            where = f" in the {z['third']} ({z['channel']}), about {z['fromGoalM']:.0f} m from goal" if z else ''
            g = f['gain']
            scored.append((f'Won the ball{where} and scored {secs:.0f} s later.', start_t, 'possession estimate', 'estimated'))
            lost_by = f" ({self._who(names, g.get('lostById'), g.get('lostBy'))} last had it)" if g.get('lostBy') or g.get('lostById') else ''
            lost_where = ''
            if z:
                # the same place in the conceding team's own direction: its third and its left or right
                third, channel = _zone(float(self.data.length)-z['depthM'], -z.get('lateralM', 0.0), float(self.data.length),
                                       float(self.data.width))
                third = {'own third': 'its own third', 'middle third': 'the middle third', 'attacking third': 'its attacking third'}[third]
                lost_where = f" in {third} ({channel}), about {z['fromGoalM']:.0f} m from its own goal"
            conceded.append((f'Lost the ball{lost_where}{lost_by} {secs:.0f} s before the goal.',
                             g['lostAt'] if _num(g.get('lostAt')) else g['wonAt'], 'possession estimate', 'estimated'))
        elif self.start_kind == 'loss':
            scored.append((f'{tw} scored {secs:.0f} s after {ow} lost control of the ball (how it was won is not decoded).', start_t,
                           'possession estimate', 'estimated'))
            conceded.append((f'Lost control of the ball {secs:.0f} s before the goal (PitchIQ last saw {ow} in possession then).', start_t,
                             'possession estimate', 'estimated'))
        elif self.start_kind == 'restart' and self.restart:
            w = RESTART_WORDS.get(self.restart['type'], 'restart')
            by = self.restart.get('team')
            if by == self.team:
                scored.append((f'The goal came {secs:.0f} s after its own {w}.', self.restart['t'], 'match state', 'fairly sure'))
                conceded.append((f'Conceded {secs:.0f} s after an opponent {w}.', self.restart['t'], 'match state', 'fairly sure'))
            elif by == self.conceding:
                scored.append((f'The goal came {secs:.0f} s after an opponent {w}: the ball was won after it.', self.restart['t'],
                               'match state', 'fairly sure'))
                conceded.append((f'Conceded {secs:.0f} s after its own {w}: the ball was lost after the restart.', self.restart['t'],
                                 'match state', 'fairly sure'))
            else:
                scored.append((f'The goal came {secs:.0f} s after {_a(w)}.', self.restart['t'], 'match state', 'fairly sure'))
        if not self.quality or self.quality.get('placed', 0) < MIN_QUALITY:
            return self._words(scored), self._words(conceded)
        if _num(f.get('progressionM')) and f['progressionM'] >= 15:
            scored.append((f"Moved the ball about {f['progressionM']:.0f} m towards goal in {secs:.0f} s.", start_t, 'tracking', 'measured'))
        if f.get('onBall'):
            labels = [self._who(names, p.get('id'), p.get('label')) for p in f['onBall'] if p.get('label') or p.get('pid') is not None][:6]
            if labels:
                scored.append((f"On the ball in the build-up (estimated): {', '.join(labels)}.", start_t, 'possession estimate', 'estimated'))
        box = f.get('box')
        if box and box['attackers'] >= 1 and box['defenders'] >= 1:
            a, dd = box['attackers'], box['defenders']
            if a >= dd:
                scored.append((f"{_n(a, tw + ' player')} in and around the box when the ball was played in, against "
                               f"{_n(dd, 'defender')} (keeper included): numbers where the ball arrived.", box['t'], 'tracking', 'measured'))
                conceded.append((f"Outnumbered or matched in the box: {_n(dd, 'defender')} (keeper included) against "
                                 f"{_n(a, 'attacker')} when the ball was played in.", box['t'], 'tracking', 'measured'))
            else:
                conceded.append((f"{_n(dd, 'defender')} (keeper included) against {_n(a, 'attacker')} in and around the box when the "
                                 f"ball was played in: the numbers were there, so the goal came from what happened at the ball.",
                                 box['t'], 'tracking', 'measured'))
        st = f.get('strike') or {}
        if _num(st.get('nearestDefenderM')) and st['nearestDefenderM'] >= STRIKE_FREE_M:
            scored.append((f"Struck towards goal with no {ow} outfield player within {st['nearestDefenderM']:.0f} m of the ball.",
                           st['t'], 'tracking', 'estimated'))
            conceded.append((f"Nobody closed the ball down: no outfield player within {st['nearestDefenderM']:.0f} m of it when it "
                             f"was struck towards goal.", st['t'], 'tracking', 'estimated'))
        if _num(st.get('keeperOffLineM')) and st['keeperOffLineM'] >= KEEPER_OFF_M:
            conceded.append((f"The keeper was about {st['keeperOffLineM']:.0f} m off the goal line when the ball was struck towards "
                             f"goal.", st['t'], 'tracking', 'estimated'))
        last = f.get('lastOnBall')
        if last and _num(last.get('nearestDefenderM')) and last['nearestDefenderM'] >= UNMARKED_M:
            who = self._who(names, last.get('id'), last.get('label'))
            scored.append((f"{who} had {last['nearestDefenderM']:.0f} m to the nearest defender when last seen on the ball "
                           f"before the goal.", last['t'], 'tracking', 'estimated'))
            conceded.append((f"Nobody was within {last['nearestDefenderM']:.0f} m of {who}, the last {tw} player PitchIQ saw on "
                             f"the ball before the goal.", last['t'], 'tracking', 'estimated'))
        g0, g5 = f.get('goalSideStart'), f.get('goalSideLater')
        if (g0 and g0['goalSide'] <= FEW_GOAL_SIDE and g0['outfield'] >= GOAL_SIDE_SEEN and g0.get('ballFromGoalM', 0) >= GOAL_SIDE_FROM_M
                and self.kind != 'set_piece'):
            tail = (f"; {g5['goalSide']} of {g5['outfield']} {RECOVER_S:.0f} s later"
                    if g5 and g5['outfield'] >= GOAL_SIDE_SEEN else '')
            conceded.append((f"Only {g0['goalSide']} of the {g0['outfield']} outfield players in view were between the ball and goal "
                             f"when the attack began, {g0['ballFromGoalM']:.0f} m out{tail}.", g0['t'], 'tracking', 'measured'))
        p = f.get('pressureM')
        if p and p['median'] >= NO_PRESSURE_M:
            conceded.append((f"Little pressure on the ball: the nearest {ow} player was {p['median']:.0f} m from it on a typical moment "
                             f"of the build-up.", start_t, 'tracking', 'measured'))
        for e in f.get('goodEpisodes') or []:
            words = EPISODE_WORDS.get(e['kind'])
            if words:
                who = names.person(e['pid']) if e['pid'] is not None else tw
                scored.append((f"{who}: {words} (key moment at {clock(e['t'])}).", e['t'], 'key moment', e.get('interpretation') or 'plausible'))
        for e in f.get('wrongEpisodes') or []:
            words = EPISODE_WORDS.get(e['kind'])
            if words:
                who = names.person(e['pid']) if e['pid'] is not None else ow
                conceded.append((f"{who}: {words} (key moment at {clock(e['t'])}).", e['t'], 'key moment', e.get('interpretation') or 'plausible'))
        return self._words(scored), self._words(conceded)

    @staticmethod
    def _words(items):
        return [{'text': t, 't': round(float(at), 2) if _num(at) else None, 'source': src, 'sure': sure} for t, at, src, sure in items]

    def record(self, names, clock):
        scored, conceded = self.text(names, clock)
        lead = max(float(self.data.t[0]), (self.start if self.start is not None else self.t)-3.0)
        return {'id': f"goal-{self.t:.1f}", 't': round(self.t, 2), 'tEstimated': round(self.t_in, 2), 'team': self.team,
                'conceding': self.conceding, 'confidence': self.event.get('confidence'),
                'timeUncertain': bool(self.event.get('timeUncertain')), 'status': self.event.get('status') or 'detected',
                'kind': self.kind, 'start': None if self.start is None else round(self.start, 2), 'startKind': self.start_kind,
                'restart': self.restart, 'watch': {'t0': round(lead, 2), 't1': round(max(self.t, self.t_in)+2.0, 2)},
                'quality': self.quality, 'facts': _plain(self.facts), 'scored': scored, 'conceded': conceded, 'notes': self.notes}


def _plain(x):
    if isinstance(x, dict):
        return {k: _plain(v) for k, v in x.items() if k != 'pid'}
    if isinstance(x, list):
        return [_plain(v) for v in x]
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, np.bool_):
        return bool(x)
    return x


def goal_events(state):
    """The match state's GOAL events the coach has not removed, in time order."""
    return sorted((e for e in (state or {}).get('events') or [] if e.get('type') == 'GOAL' and e.get('status') != 'removed'),
                  key=lambda e: float(e['t']))


def signature(state):
    """What the goal analysis depends on of the match state (the report's cache key)."""
    return [(round(float(e['t']), 2), e.get('team'), e.get('status'), bool(e.get('timeUncertain'))) for e in goal_events(state)]


def analyse(data, engine, eps_all, state, rec2):
    """Every goal of the match state analysed: [Goal]."""
    timeline, events = (state or {}).get('timeline') or [], (state or {}).get('events') or []
    return [Goal(data, engine, eps_all, rec2, e, timeline, events) for e in goal_events(state)]


def for_team(goals, team, names, clock):
    """The report's goals section for one team: the goals it scored (what it did well) and conceded (what went wrong)."""
    scored, conceded = [], []
    for g in goals:
        r = g.record(names, clock)
        if g.team == team:
            scored.append({k: v for k, v in r.items() if k != 'conceded'} | {'findings': r['scored']})
        elif g.conceding == team:
            conceded.append({k: v for k, v in r.items() if k != 'scored'} | {'findings': r['conceded']})
    return {'scored': scored, 'conceded': conceded,
            'note': ("Goals come from the match state (a kick-off after a goal, law 10). The build-up is measured from the "
                     "tracking at five samples a second; who had the ball is PitchIQ\u2019s possession estimate. There is no ball "
                     "height and no pass event, so no scorer or assist is named.")}
