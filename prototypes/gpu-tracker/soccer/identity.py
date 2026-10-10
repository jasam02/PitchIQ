"""GLOBAL IDENTITY MANAGER. Sits above BoT-SORT.

Local track IDs are temporary (an occlusion, an exit or a camera cut ends them). Global identities
(A-07, B-03, GK-1, REF-1) are the actual people and persist for the whole run, and into the next
run when identities are continued.

Rules:
- A detection becomes a CANDIDATE track first; it is promoted only after several observations,
  mostly inside the pitch, with a stable role/team decision.
- A promoted track is first compared with every missing identity of its team and role. A new
  identity is created only when no missing identity can be this person.
- Ambiguous matches are deferred (IDENTITY_UNCERTAIN), never guessed. Reconnection needs several
  scored observations, a minimum score and a clear lead over the second-best identity and over an
  unseen teammate.
- One identity has at most one live track. Identities are never deleted; they become MISSING or
  OFF_SCREEN and keep their appearance gallery.
- Crossing players are never swapped on proximity alone; appearance must favour a swap.

Identity, team and role are separate: an identity (A-01) has a team (A) and a role (PLAYER,
GOALKEEPER or REFEREE). Roles are re-evaluated from accumulated evidence; a confirmed referee or
goalkeeper role is locked and survives leaving the goal area or a few odd frames. A player identity
that turns out to be a referee is retired into a referee identity (REF-n) and its observations are
relabelled, so a referee never stays a team player.
"""
import math
from collections import Counter, deque
from dataclasses import dataclass, field

import numpy as np

from .appearance import Sample, add_to_gallery, batch_similarity, compare, decode_descriptor, encode_descriptor
from .geometry import clamp, edge_of, iou
from .roles import RoleDecision, RoleEvidence, decide_role
from .teams import role_label

# Final re-ID score weights, re-normalized over the components known for a pair.
REID_WEIGHTS = {'appearance': .35, 'uniform': .1, 'spatial': .3, 'movement': .15, 'temporal': .1}
PLAYER_HEIGHT = 1.8      # metres; converts image distances when there is no pitch calibration
PITCH_LENGTH, PITCH_WIDTH = 105.0, 68.0
SPATIAL_HORIZON = 35.0   # metres: beyond this reach the last position says nothing about who it is
LOST = .35               # seconds without detection before a bound identity may move to a new track
BUFFER_SECONDS = 12.0    # unidentified observations kept for retroactive labelling
WINDOW, DESCS, HIST, VOTES = 10, 6, 10, 15
ZONE_WINDOW = 25        # evidence ticks (5 s) used for the inside-the-pitch ratio
SWAP_CAP, CLEAN, KIT_MISMATCH, MISSES = 2.0, .4, .6, 3
RELEASE_HOLD = 3.0
KIT_WINDOW = 4.0
MAX_EVENTS_PER_STEP = 60
ROLE_LOCK = .7           # role confidence that converts and locks a referee / goalkeeper role
GK_DOUBT_CHECKS = 3      # consecutive role checks (~3 s) of contradictory evidence that revoke a confirmed goalkeeper
ROLE_OUT = {'player': 'PLAYER', 'goalkeeper': 'GOALKEEPER', 'referee': 'REFEREE'}
OFFICIAL_OUT = {'assistant': 'ASSISTANT_REFEREE', 'centre': 'CENTER_REFEREE', '': 'UNKNOWN_OFFICIAL'}
LOOKS_PROBES = 3         # recent clean crops compared with the referee and team appearance galleries


def r3(v):
    return round(float(v), 3)


def pct(v):
    return f'{round(clamp(v)*100)}%'


def long_id(pid):
    if pid.startswith('A-') or pid.startswith('B-'):
        return f'Team{pid[0]}_Player_{pid[2:]}'
    if pid.startswith('GK-'):
        return f'Goalkeeper_{int(pid[3:]):02d}'
    if pid.startswith('REF-'):
        return f'Referee_{int(pid[4:]):02d}'
    return pid


def combine(c):
    """Weighted mean over known components; a team mismatch is a hard gate (0)."""
    if not c.get('team', True):
        return 0.0
    s = w = 0.0
    for key, weight in REID_WEIGHTS.items():
        v = c.get(key)
        if v is not None and math.isfinite(v):
            s += weight*v
            w += weight
    return s/w if w else 0.0


@dataclass
class Stab:
    """Camera-compensated foot point and box height, in units of the camera segment's first frame."""
    x: float
    y: float
    h: float


@dataclass
class Options:
    max_per_team: int = 11
    max_referees: int = 3
    max_goalkeepers: int = 2
    min_hits: int = 5            # evidence ticks before promotion (5 x 0.2 s = 1 s)
    boundary_hits: int = 10      # stronger requirement for tracks first seen on the touchline
    reid_min: float = .72
    reid_margin: float = .15
    gallery_size: int = 6
    tick: float = .2             # seconds between evidence ticks
    over_cap_seconds: float = 5.0


@dataclass
class GlobalPlayer:
    pid: str
    role: str
    team: str | None
    status: str = 'missing'      # active | missing | offscreen | unknown | substituted | retired
    gallery: list = field(default_factory=list)
    first_seen: float = 0.0
    last_seen: float = 0.0
    last_box: list = field(default_factory=lambda: [.5, .5, .01, .02])
    last_pitch: tuple | None = None
    pitch_velocity: tuple | None = None
    exit_edge: str = ''
    velocity: tuple = (0.0, 0.0)
    identity_confidence: float = 0.0
    team_confidence: float = 0.0
    role_confidence: float = 0.0
    track: int | None = None
    segment: int | None = None
    stab: Stab | None = None
    drift: int = 0
    last_observed: float = -math.inf
    cap_until: float = 0.0
    crossed_at: float = -math.inf
    bound_at: float = -math.inf
    created_on: int | None = None
    here: bool = False
    restored: bool = False
    observations: int = 0
    gk_votes: Counter = field(default_factory=Counter)
    history: deque = field(default_factory=lambda: deque(maxlen=40))
    role_locked: bool = False
    referee_confidence: float = 0.0
    goalkeeper_confidence: float = 0.0
    role_history: list = field(default_factory=list)      # [(time, role, reason)]
    emitted_all: list = field(default_factory=list)       # every person dict labelled with this identity
    retired_into: str | None = None
    official: str = ''                                   # referees: assistant | centre | '' (kind unknown)
    role_scores: dict = field(default_factory=dict)      # the latest role decision's evidence scores
    doubt: int = 0                                       # goalkeepers: consecutive checks with contradictory evidence
    revoked: str = ''                                    # goalkeepers: why the goalkeeper status was taken away
    keeper_goal: float = 0.0                             # goalkeepers: the strongest goal-area residence ever seen (what confirmed them)

    @property
    def role_label(self):
        return role_label(self.role, self.team)

    @property
    def display(self):
        """Shown label: A-04, GK-A / GK-B (goalkeepers by team), REF-1."""
        if self.role == 'goalkeeper' and self.team:
            return f'GK-{self.team}'
        return self.pid


@dataclass
class TrackState:
    id: int
    born: float
    born_segment: int
    first_stab: Stab
    first_pitch: tuple | None
    first_zone: str
    entry_edge: str
    box: list
    stab: Stab
    segment: int
    score: float = 0.0
    cls: str = 'player'
    seen_at: float = 0.0
    zone: str = 'unknown'
    pitch: tuple | None = None
    hits: int = 0
    inside: int = 0
    boundary: int = 0
    outside: int = 0
    outside_run: int = 0
    zones: deque = field(default_factory=lambda: deque(maxlen=ZONE_WINDOW))
    evidence: RoleEvidence = field(default_factory=RoleEvidence)
    decision: object = None
    team_votes: deque = field(default_factory=lambda: deque(maxlen=VOTES))
    descs: deque = field(default_factory=lambda: deque(maxlen=DESCS))
    current: object = None
    occluded: bool = False
    velocity: tuple = (0.0, 0.0)
    pitch_velocity: tuple | None = None
    hist: deque = field(default_factory=lambda: deque(maxlen=HIST))
    last_box: list = None
    last_time: float = 0.0
    last_tick: float = -math.inf
    last_pitch: tuple | None = None
    state: str = 'candidate'     # candidate | confirmed | uncertain | unknown | rejected
    player_id: str | None = None
    team: str | None = None
    role: str = 'unknown'
    team_confidence: float = 0.0
    role_confidence: float = 0.0
    retired: bool = False
    reacquired: bool = False
    crossed_at: float = -math.inf
    kit_miss: int = 0
    look_miss: int = 0
    pending: deque = field(default_factory=deque)   # unidentified person dicts (retroactive labelling)
    emitted: deque = field(default_factory=deque)   # (time, person dict) recently emitted while bound
    role_checked: float = -math.inf
    former: set = field(default_factory=set)        # identities this track was bound to before
    first: list = field(default_factory=list)       # first clean descriptors (did the track switch people?)
    reid: dict = field(default_factory=dict)
    best: float = 0.0
    last_deferred: float = -math.inf
    deferrals: int = 0
    reason: str = ''
    blind: list = field(default_factory=list)
    released: tuple | None = None
    looks: tuple | None = None                      # (pool version, time, {'referee', 'A', 'B', 'probes'})
    role_streak: int = 0                            # consecutive role checks asking for the same new role
    role_wanted: str | None = None
    keeper_logged: float = -math.inf                # when this track's goalkeeper-candidate evidence was last logged
    keeper_unknown: int = 0                         # consecutive checks on which a goalkeeper candidate's decision was CANDIDATE


class IdentityManager:
    def __init__(self, options=None, saved=None):
        self.o = options or Options()
        self.registry = {}
        self.tracks = {}
        self.pairs = {}
        self.together = set()
        self.limits = {}
        self.kits = {}
        self.counters = Counter()
        self.side_votes = {'left': Counter(), 'right': Counter()}
        self.time = 0.0
        self.started = None
        self.last_tick = -math.inf
        self.events_log = []
        self.pools, self.pools_at = None, -math.inf
        if saved:
            self.restore(saved)

    # ---------- persistence ----------
    def snapshot(self):
        out = []
        for g in self.registry.values():
            out.append({'id': g.pid, 'role': g.role, 'team': g.team, 'status': g.status,
                        'gallery': [{'t': r3(s.time), 'd': encode_descriptor(s.d)} for s in g.gallery],
                        'firstSeen': r3(g.first_seen), 'lastSeen': r3(g.last_seen), 'lastBox': [round(float(v), 5) for v in g.last_box],
                        'lastPitch': None if g.last_pitch is None else [r3(g.last_pitch[0]), r3(g.last_pitch[1])],
                        'exitEdge': g.exit_edge, 'identityConfidence': r3(g.identity_confidence),
                        'teamConfidence': r3(g.team_confidence), 'roleConfidence': r3(g.role_confidence),
                        'observations': g.observations, 'gkVotes': dict(g.gk_votes), 'roleLocked': g.role_locked,
                        'refereeConfidence': r3(g.referee_confidence), 'goalkeeperConfidence': r3(g.goalkeeper_confidence), 'revoked': g.revoked,
                        'keeperGoal': r3(g.keeper_goal),
                        'retiredInto': g.retired_into, 'official': g.official})
        return {'version': 1, 'identities': out, 'counters': dict(self.counters)}

    def restore(self, saved):
        for v in saved.get('identities', []) if isinstance(saved, dict) else []:
            try:
                pid, role = str(v['id']), v['role']
                if pid in self.registry or role not in ('player', 'goalkeeper', 'referee'):
                    continue
                gallery = []
                for s in v.get('gallery', [])[-self.o.gallery_size:]:
                    d = decode_descriptor(s.get('d', {}))
                    if d is not None:
                        gallery.append(Sample(d, float(s.get('t', 0))))
                box = [float(x) for x in v.get('lastBox', [.5, .5, .01, .02])][:4]
                edge = str(v.get('exitEdge', ''))[:8]
                retired = v.get('status') == 'retired'
                g = GlobalPlayer(pid, role, v.get('team') if v.get('team') in ('A', 'B') else None,
                                 'retired' if retired else 'unknown' if v.get('revoked') else 'offscreen' if edge else 'missing', gallery, float(v.get('firstSeen', 0)), float(v.get('lastSeen', 0)), box,
                                 tuple(v['lastPitch']) if v.get('lastPitch') else None, None, edge,
                                 identity_confidence=clamp(float(v.get('identityConfidence', 0))),
                                 team_confidence=clamp(float(v.get('teamConfidence', 0))), role_confidence=clamp(float(v.get('roleConfidence', 0))),
                                 restored=True, gk_votes=Counter(v.get('gkVotes', {})), role_locked=bool(v.get('roleLocked')),
                                 referee_confidence=clamp(float(v.get('refereeConfidence', 0))), goalkeeper_confidence=clamp(float(v.get('goalkeeperConfidence', 0))),
                                 revoked=str(v.get('revoked', '') or ''),
                                 keeper_goal=clamp(float(v.get('keeperGoal', 1.0 if v.get('roleLocked') and role == 'goalkeeper' else 0.0))),
                                 retired_into=v.get('retiredInto'), official=v.get('official') if v.get('official') in ('assistant', 'centre') else '')
                self.registry[pid] = g
            except (KeyError, TypeError, ValueError):
                continue
        for key, value in (saved.get('counters', {}) if isinstance(saved, dict) else {}).items():
            self.counters[key] = max(self.counters[key], int(value))

    # ---------- per frame ----------
    def update(self, ctx):
        """ctx: dict with time, samples, ended, model, pitch_reliable, filter_enabled, aspect, segment,
        drift, stabilize. Returns {'people', 'events', 'issues', 'drop'}."""
        out = {'people': [], 'events': [], 'issues': [], 'drop': []}
        t = ctx['time']
        if self.started is None:
            self.started = t
        if t < self.time-1e-6:
            for g in self.registry.values():
                g.last_observed = -math.inf
        self.time = t
        tick = ctx.get('tick', True)
        for e in ctx['ended']:
            self._end_track(e, out)
        seen = set()
        for s in ctx['samples']:
            self._observe(s, ctx, tick)
            seen.add(s['id'])
        live = sorted((self.tracks[i] for i in seen if i in self.tracks), key=lambda x: x.id)
        if tick:
            self._behaviour_cues(live, ctx)
            for tr in live:
                self._check_team(tr, ctx, out)
            self._check_roles(live, ctx, out)
            waiting = [tr for tr in live if tr.state not in ('confirmed', 'uncertain')]
            for tr in waiting:
                tr.decision = self._decide(tr, ctx)
            for tr in waiting:
                self._promote(tr, ctx, out)
            self._resolve([tr for tr in live if tr.state == 'uncertain'], ctx, out)
            self._guard_swaps([tr for tr in live if tr.state != 'rejected' and not tr.retired], ctx, out)
            self._check_reacquired(live, ctx, out)
            self._goalkeeper_teams(live, ctx, out)
        self._refresh(live, ctx, out, tick)
        if tick and self._allow('sanity-scan', t, 5):
            for e in self.sanity(t):
                if self._allow('sanity:'+e['message'][:60], t, 30):
                    out['events'].append(e)
        if len(out['events']) > MAX_EVENTS_PER_STEP:
            out['events'] = out['events'][-MAX_EVENTS_PER_STEP:]
        return out

    def _end_track(self, end, out):
        tid = end['id']
        tr = self.tracks.pop(tid, None)
        if tr is None:
            return
        self._drop_pairs(tid, out)
        g = self.registry.get(tr.player_id) if tr.player_id else None
        if g is not None and g.track == tid:
            edge = edge_of(tr.last_box)
            g.track = None
            g.exit_edge = edge
            if g.status != 'substituted':
                g.status = 'offscreen' if edge else 'missing'
            return
        if tr.state == 'uncertain' and not tr.retired:
            out['events'].append(self._event('deferred', tr.id, None,
                                             f'Track {tr.id} ({self._describe(tr.role, tr.team)}) ended before its identity could be decided; it stays unidentified.'))

    def _observe(self, s, ctx, tick):
        tr = self.tracks.get(s['id'])
        t = ctx['time']
        if tr is None:
            tr = TrackState(s['id'], t, ctx['segment'], s['stab'], s.get('pitch'), s['zone'], edge_of(s['box']), s['box'], s['stab'], ctx['segment'])
            tr.last_box, tr.last_time = s['box'], t
            self.tracks[s['id']] = tr
        tr.reacquired = t-tr.last_time > LOST and tr.hits > 0
        tr.box, tr.score, tr.cls, tr.seen_at, tr.stab, tr.pitch = s['box'], s['score'], s['cls'], t, s['stab'], s.get('pitch')
        tr.zone, tr.occluded = s['zone'], s['occluded']
        if tr.segment != ctx['segment']:
            tr.segment, tr.velocity, tr.pitch_velocity = ctx['segment'], (0.0, 0.0), None
            tr.hist.clear()
        tr.current = None
        if tick:
            tr.last_tick = t
            tr.hits += 1
            tr.inside += tr.zone == 'inside'
            tr.boundary += tr.zone == 'boundary'
            tr.outside += tr.zone == 'outside'
            tr.outside_run = tr.outside_run+1 if tr.zone == 'outside' else 0
            tr.zones.append(tr.zone)
            d = s.get('descriptor')
            if d is not None:
                tr.current = d
                if d.quality >= .2:
                    tr.descs.append((d, t, s['occluded']))
                    if not s['occluded'] and len(tr.first) < 2:
                        tr.first.append(d)
            if tr.hist:
                prev = tr.hist[-1]
                dt = t-prev[0]
                if 1e-3 < dt <= 1.5:
                    vx, vy = (tr.stab.x-prev[1])/dt, (tr.stab.y-prev[2])/dt
                    tr.velocity = (.6*tr.velocity[0]+.4*vx, .6*tr.velocity[1]+.4*vy) if len(tr.hist) > 1 else (vx, vy)
            tr.hist.append((t, tr.stab.x, tr.stab.y))
            vote = s.get('vote')
            h = max(tr.stab.h, .005)
            motion = (tr.velocity[0]*ctx['aspect']/h*PLAYER_HEIGHT, tr.velocity[1]/h*PLAYER_HEIGHT)  # metres per second
            tr.evidence.add(s['cls'], s['score'], vote, dict(s.get('cues', {}), zone=tr.zone, v=motion))
            if vote is not None and vote.valid:
                tr.team_votes.append(vote.team or ('x' if vote.outlier else '-'))
            if tr.pitch is not None and tr.last_pitch is not None and 1e-3 < t-tr.last_time <= 1.5:
                dt = t-tr.last_time
                vx, vy = (tr.pitch[0]-tr.last_pitch[0])/dt, (tr.pitch[1]-tr.last_pitch[1])/dt
                tr.pitch_velocity = (.6*tr.pitch_velocity[0]+.4*vx, .6*tr.pitch_velocity[1]+.4*vy) if tr.pitch_velocity else (vx, vy)
            elif tr.pitch is None:
                tr.pitch_velocity = None
            tr.last_pitch = tr.pitch
        tr.last_box, tr.last_time = s['box'], t

    # ---------- team / role ----------
    def _check_team(self, tr, ctx, out):
        """A confirmed player whose kit votes switched (>= 80% of the last 15) is released and re-identified."""
        if tr.state not in ('confirmed', 'uncertain') or tr.team is None or tr.role != 'player' or len(tr.team_votes) < VOTES:
            return
        other = 'B' if tr.team == 'A' else 'A'
        n = sum(1 for v in tr.team_votes if v == other)
        if n < .8*VOTES:
            return
        g = self.registry.get(tr.player_id) if tr.player_id else None
        if g is not None and g.track == tr.id:
            self._unbind(g)
            tr.former.add(g.pid)
            out['issues'].append({'id': g.pid, 'time': ctx['time'], 'reason': 'Kit changed to the other team; identity released.'})
        tr.player_id, tr.team, tr.state, tr.best = None, other, 'uncertain', 0.0
        tr.reid.clear()
        tr.team_votes.clear()
        tr.reason = f'kit changed to team {other}'
        out['events'].append(self._event('team-change', tr.id, g.pid if g else None,
                                         f"Track {tr.id}{f' ({g.pid})' if g else ''} matched team {other}'s kit in {n} of the last {VOTES} observations; re-identifying as team {other}."))

    def _promote(self, tr, ctx, out):
        """CANDIDATE -> role. Never from one frame: min_hits observations, mostly inside the pitch,
        and a temporal role decision. Tracks born on the touchline need stronger evidence."""
        if tr.retired:
            tr.state = 'unknown'
            return
        o, d = self.o, tr.decision
        if d.label == 'CANDIDATE' and d.scores.get('neitherTeam', 0) >= .6 and tr.hits >= o.min_hits and len(tr.descs) >= 3:
            # A kit matching neither team is not a role by itself, but the person may be a known referee or
            # goalkeeper: their own appearance decides.
            match = self._special_match(tr, ('referee', 'goalkeeper'))
            if match is not None and match[1] >= .8:
                g = match[0]
                d = tr.decision = RoleDecision(g.role_label, g.role, g.team, .8 if g.team else 0.0, round(match[1], 3),
                                               f'looks like {g.display} (appearance {match[1]:.2f})', d.referee_confidence, d.goalkeeper_confidence, d.scores)
        recent = [z for z in tr.zones if z != 'unknown']
        zr = sum(1 if z == 'inside' else .5 if z == 'boundary' else 0 for z in recent)/len(recent) if recent else 0.0
        born = tr.first_zone == 'boundary'
        need = o.boundary_hits if born else o.min_hits
        pitch_ok = ctx['pitch_reliable'] or not ctx['filter_enabled']
        ok = pitch_ok and d.label != 'CANDIDATE' and d.role != 'unknown' and tr.hits >= o.min_hits
        # Assistant referees run the touchline: allowed for a confident referee decision.
        runner = d.role == 'referee' and d.role_confidence >= .6
        sideline = ''
        if ok and d.role == 'goalkeeper':
            # The detector may call a sideline official a goalkeeper; feet on or outside the touchline, or
            # in its corridor, keep them a candidate rather than a goalkeeper candidate.
            k = d.scores.get('keeper') or {}
            if k.get('vetoed'):
                ok, sideline = False, k['vetoed']
        if ok:
            ok = tr.hits >= need if runner else zr >= .8 and (not born or (tr.hits >= o.boundary_hits and tr.inside >= 3))
        if ok:
            tr.state, tr.team, tr.role = 'uncertain', d.team, d.role
            tr.team_confidence, tr.role_confidence, tr.reason = d.team_confidence, d.role_confidence, ''
            out['events'].append(self._event('promotion', tr.id, None,
                                             f'Track {tr.id} promoted to {self._describe(d.role, d.team)} after {tr.hits} observations (inside {pct(zr)}, team {pct(d.team_confidence)}, role {pct(d.role_confidence)}).'))
            return
        if tr.hits >= 3*o.min_hits:
            tr.state = 'rejected' if zr < .8 and not runner else 'unknown'
        else:
            tr.state = 'candidate'
        if not pitch_ok:
            tr.reason = 'pitch not reliable on this frame: no promotion'
        elif sideline:
            tr.reason = f'looks like a goalkeeper to the detector, but {sideline}: not a goalkeeper candidate'
        elif tr.state == 'rejected':
            tr.reason = f'mostly outside the playable area (inside {pct(zr)})'
        elif tr.state == 'unknown':
            tr.reason = f'no consistent role or kit after {tr.hits} observations ({d.reason})'
        else:
            tr.reason = f"{tr.hits}/{need} observations, inside {pct(zr)}{' (born on the touchline)' if born else ''}; {d.reason}"

    # ---------- roles of confirmed identities ----------
    def _check_roles(self, live, ctx, out):
        """Confirmed identities keep collecting role evidence (about once a second). A player identity
        whose evidence becomes clearly referee or goalkeeper is converted and the role is locked; a
        locked referee or goalkeeper is never turned back into a player by a few odd frames."""
        t = ctx['time']
        for tr in live:
            if tr.state == 'uncertain' and t-tr.role_checked >= 1.0 and len(tr.evidence.records) >= 2*self.o.min_hits and not tr.retired:
                tr.role_checked = t
                d = tr.decision = self._decide(tr, ctx)
                confidence = d.referee_confidence if d.role == 'referee' else d.goalkeeper_confidence if d.role == 'goalkeeper' else 0
                if d.role in ('referee', 'goalkeeper') and d.role != tr.role and confidence >= ROLE_LOCK:
                    old = self._describe(tr.role, tr.team)
                    tr.role, tr.team, tr.role_confidence = d.role, (d.team if d.role == 'goalkeeper' else None), confidence
                    tr.reid.clear()
                    out['events'].append(self._event('role', tr.id, None, '\n'.join(['ROLE UPDATE', f'Local Track: {tr.id}', f'Old role: {old}',
                                                                                      f'New role: {ROLE_OUT[d.role]}'] + self._role_lines(d) + [f'Role confidence: {confidence:.2f}'])))
                elif tr.role == 'goalkeeper':
                    # A goalkeeper candidate (no identity yet) whose feet say sideline, whose evidence turned to
                    # another role, or whose goalkeeper evidence faded, goes back to being a candidate for
                    # anything: OUT beats a wrong keeper.
                    k = d.scores.get('keeper', {})
                    tr.keeper_unknown = tr.keeper_unknown+1 if d.role == 'unknown' else 0
                    if k.get('vetoed') or d.role in ('referee', 'player') or tr.keeper_unknown >= 3:
                        why = k.get('reason') or d.reason or 'no goalkeeper evidence'
                        tr.role, tr.team, tr.state, tr.role_confidence = 'unknown', None, 'candidate', 0.0
                        tr.reid.clear()
                        tr.reason = f'goalkeeper candidate dropped: {why}'
                        out['events'].append(self._event('role', tr.id, None, '\n'.join(
                            ['GOALKEEPER CANDIDATE DROPPED', f'Local Track: {tr.id}', f'Reason: {why}'] + self._keeper_lines(d) + ['State: ' + ('OUT' if k.get('offPitch', 0) >= .5 else ROLE_OUT.get(d.role, 'UNKNOWN'))])))
                continue
            if tr.state != 'confirmed' or tr.player_id is None or t-tr.role_checked < 1.0 or len(tr.evidence.records) < 2*self.o.min_hits:
                continue
            g = self.registry.get(tr.player_id)
            if g is None or g.track != tr.id:
                continue
            tr.role_checked = t
            d = tr.decision = self._decide(tr, ctx)
            g.referee_confidence, g.goalkeeper_confidence = d.referee_confidence, d.goalkeeper_confidence
            g.role_scores = d.scores
            if d.role == g.role:
                current = d.referee_confidence if g.role == 'referee' else d.goalkeeper_confidence if g.role == 'goalkeeper' else d.role_confidence
                g.role_confidence = r3(max(current, g.role_confidence) if g.role_locked else current)
            if g.role in ('referee', 'goalkeeper'):
                if g.role == 'referee' and d.role == 'referee' and d.official:
                    g.official = d.official
                confidence = d.referee_confidence if g.role == 'referee' else d.goalkeeper_confidence
                if g.role == 'goalkeeper':
                    g.keeper_goal = max(g.keeper_goal, d.scores['nearGoal'])
                    # Hysteresis: a confirmed goalkeeper survives odd frames, a walk upfield (the detector may well
                    # call them a player there) and a ball fetched behind the goal line; only sustained sideline,
                    # off-pitch-away-from-the-goal or referee evidence over a full window, after the track has
                    # been theirs for a while, revokes the status and frees the team's slot.
                    settled = len(tr.evidence.records) >= WINDOW and t-g.bound_at >= 5.0
                    contradiction = self._keeper_contradiction(d) if settled else ''
                    g.doubt = g.doubt+1 if contradiction else 0
                    if g.doubt >= GK_DOUBT_CHECKS:
                        self._revoke_goalkeeper(tr, g, d, contradiction, ctx, out)
                        continue
                if not g.role_locked and confidence >= ROLE_LOCK and (g.role != 'goalkeeper' or d.scores.get('keeper', {}).get('confirmed')):
                    g.role_locked, g.role_confidence = True, max(g.role_confidence, confidence)
                    g.role_history.append((r3(t), g.role, f'locked at {confidence:.2f}'))
                elif not g.role_locked and g.role == 'referee' and d.role == 'goalkeeper' and d.goalkeeper_confidence >= ROLE_LOCK:
                    self._to_goalkeeper(tr, g, d, ctx, out)  # an unconfirmed referee that is the goalkeeper
                elif not g.role_locked and g.role == 'goalkeeper' and d.role == 'referee' and d.referee_confidence >= ROLE_LOCK:
                    self._to_referee(tr, g, d, ctx, out)
                elif g.role_locked and d.role == 'player' and d.role_confidence >= .9 and self._allow('role-doubt-'+g.pid, t, 30):
                    out['issues'].append({'id': g.pid, 'time': t, 'reason': f'{g.display} now looks like a team {d.team} player; role kept (locked). Check.'})
                continue
            if g.role_locked:
                continue
            # A confirmed player changes role only after two checks in a row (about 2 s) ask for the same
            # new role: the evidence accumulates, and the label never flips back and forth.
            want = 'referee' if d.role == 'referee' and d.referee_confidence >= ROLE_LOCK else \
                'goalkeeper' if d.role == 'goalkeeper' and d.goalkeeper_confidence >= ROLE_LOCK else None
            tr.role_streak = tr.role_streak+1 if want is not None and want == tr.role_wanted else (1 if want else 0)
            tr.role_wanted = want
            if want is None or tr.role_streak < 2:
                continue
            if want == 'referee':
                self._to_referee(tr, g, d, ctx, out)
            else:
                self._to_goalkeeper(tr, g, d, ctx, out)

    def _role_lines(self, d):
        """The evidence behind a role decision, one line each (kit colour, whole appearance against the known
        referees and each team, detector votes, touchline and central-referee behaviour, team count)."""
        s = d.scores
        lines = []
        if s.get('teamA') is not None:
            lines += [f"Team A similarity: {s['teamA']:.2f}", f"Team B similarity: {s['teamB']:.2f}"]
        if s.get('refereeAppearance') is not None or s.get('teamAAppearance') is not None or s.get('teamBAppearance') is not None:
            f = lambda key: 'n/a' if s.get(key) is None else f"{s[key]:.2f}"
            lines.append(f"Appearance vs known referees / team A / team B: {f('refereeAppearance')} / {f('teamAAppearance')} / {f('teamBAppearance')}")
        lines += [f"Kit matches neither team: {s['neitherTeam']:.0%} of observations", f"Referee kit: {s['refereeKit']:.0%}",
                  f"Detector referee / goalkeeper votes: {s['detectorReferee']:.0%} / {s['detectorGoalkeeper']:.0%}"]
        if s.get('touchlineTime'):
            lines.append(f"Time on or outside the touchline: {s['touchlineTime']:.0%}" +
                         (f", movement along it: {s['parallelMovement']:.2f}" if s.get('parallelMovement') is not None else ''))
        if s.get('formationConsistency') is not None:
            lines.append(f"Team formation consistency: {s['formationConsistency']:.2f}")
        if s.get('ballFollowing') is not None:
            lines.append(f"Ball-following behaviour: {s['ballFollowing']:.2f}")
        if s.get('refereeBehaviour') is not None:
            lines.append(f"Referee movement behaviour: {s['refereeBehaviour']:.2f}")
        if s.get('teamCrowded'):
            lines.append('Team already has its full count of identities')
        if s.get('final'):
            lines.append(f"Final: {s['final']}")
        return lines

    @staticmethod
    def _keeper_lines(d):
        """The goalkeeper evidence behind a decision, one line each, with where the person stands and
        whether that confirms a goalkeeper identity or leaves a candidate."""
        s = d.scores
        k = s.get('keeper') or {}
        lines = [f"Role detector: GK {s['detectorGoalkeeper']:.0%} / REF {s['detectorReferee']:.0%}",
                 f"Pitch state: INSIDE {k.get('inside', 0):.0%} / ON OR OUTSIDE THE TOUCHLINE {k.get('offPitch', 0):.0%}"
                 + (f" (touchline corridor {k['touchline']:.0%})" if k.get('touchline') else '') + (f", goal side {k['goalSide']}" if k.get('goalSide') else ''),
                 f"Goal proximity score: {s['nearGoal']:.2f}", f"Penalty-area residence: {s['nearGoal']:.0%} of observations",
                 f"Uniform difference score: {max(s['neitherTeam'], s['keeperKit']):.2f}", f"Deepest / isolated: {s['deepest']:.0%} / {s['isolated']:.0%}",
                 f"Detector goalkeeper votes: {s['detectorGoalkeeper']:.0%}", f'Temporal confidence: {d.goalkeeper_confidence:.2f}']
        if k:
            lines.append(f"Goalkeeper confirmation: {k['state']} ({k['reason']})")
        return lines

    @staticmethod
    def _keeper_confirmed(d):
        return d is not None and d.role == 'goalkeeper' and bool((d.scores.get('keeper') or {}).get('confirmed'))

    @staticmethod
    def _keeper_contradiction(d):
        """Evidence over the window that a confirmed goalkeeper is not one: '' when none."""
        s, k = d.scores, d.scores.get('keeper') or {}
        if d.role == 'referee' and d.referee_confidence >= ROLE_LOCK:
            return f'referee evidence {d.referee_confidence:.2f}'
        if k.get('outside', 0) >= .5 and s['nearGoal'] < .3:
            return f"outside the pitch {k['outside']:.0%} of the time, away from any goal"
        if k.get('touchline', 0) >= .5 and s['nearGoal'] < .1:
            return f"patrols the touchline ({k['touchline']:.0%}) away from any goal"
        # Detector 'player' votes while the keeper stands upfield are not a contradiction (invariant: a
        # confirmed keeper walking out of the box keeps the role); that case is only noted as an issue.
        return ''

    def goalkeeper_of(self, team, exclude=None):
        """The identity currently holding team `team`'s goalkeeper slot, or None. Revoked, retired and
        substituted identities hold nothing: the slot is a live relationship, not a historical fact."""
        holders = [g for g in self.registry.values()
                   if g is not exclude and g.role == 'goalkeeper' and g.team == team and g.status not in ('retired', 'substituted', 'unknown')]
        if not holders:
            return None
        return max(holders, key=lambda g: (g.track is not None and self._lost_for(g) < LOST, g.last_seen))

    def keeper_slots(self):
        """{'A': pid or None, 'B': pid or None}: who holds each team's goalkeeper slot right now."""
        return {team: (g.pid if g is not None else None) for team in ('A', 'B') for g in [self.goalkeeper_of(team)]}

    def _keeper_outranks(self, tr, d, owner):
        """A confirmed goalkeeper candidate on track tr takes a team's slot from its current owner when
        the owner is not clearly the keeper now: not visible (or visible without goal-area evidence) and
        with weaker goalkeeper evidence than the candidate's. Candidates compete; the first seen does not
        win. Returns 'revoke' (the owner's goalkeeper status was weak: taken away), 'deteam' (the owner
        was itself confirmed at a goal, so it is a goalkeeper of the other team: only the team is taken)
        or '' (the owner keeps the slot)."""
        if not self._keeper_confirmed(d):
            return ''
        s = d.scores
        at_goal = s['nearGoal'] >= .5
        strong_owner = owner.role_locked and owner.keeper_goal >= .5   # confirmed at a goal, whatever its last window looks like
        visible = owner.track is not None and self._lost_for(owner) < LOST
        if visible:
            # A visible owner only ever loses its status through its own contradictory evidence; a weakly
            # confirmed one (never seen at a goal) at most loses the team to a candidate standing at the goal.
            return 'deteam' if (not strong_owner and at_goal) else ''
        missing = max(0.0, self.time-owner.last_seen)
        if strong_owner:
            # Two keepers confirmed at the same goal: the one there now has the team, the absent one is a
            # goalkeeper of the other team until the side votes say more. Needs the candidate at the goal and
            # the owner gone for a while, so a brief occlusion never costs a keeper its team.
            return 'deteam' if at_goal and missing >= 10.0 else ''
        strength = lambda conf, goal: conf*(1.0 if goal >= .5 else .6)
        if strength(d.goalkeeper_confidence, s['nearGoal']) >= strength(owner.goalkeeper_confidence, owner.keeper_goal)+.15:
            return 'revoke'
        return 'deteam' if at_goal and missing >= 10.0 else ''

    def _deteam_goalkeeper(self, owner, successor, why, ctx, out):
        """A confirmed goalkeeper loses a team it cannot have (another keeper defends that goal): the
        identity keeps its goalkeeper role and is re-teamed by the side votes later."""
        was, team = owner.display, owner.team
        owner.team, owner.team_confidence = None, 0.0
        owner.gk_votes.clear()
        owner.role_history.append((r3(ctx['time']), 'goalkeeper', f'team {team} released: {why}'))
        self._restyle(owner)
        tr = self.tracks.get(owner.track) if owner.track is not None else None
        if tr is not None and tr.player_id == owner.pid:
            tr.team = None
        out['events'].append(self._event('role', tr.id if tr is not None else None, owner.pid,
                                         f'GOALKEEPER TEAM RELEASED\nGlobal ID: {owner.pid} (was {was}, now {owner.display})\nReason: {why}\nTeam {team} goalkeeper slot: released to {successor}'))

    def _take_slot(self, tr, d, owner, successor, why, ctx, out):
        """Give owner's team slot to successor according to _keeper_outranks; True when taken."""
        mode = self._keeper_outranks(tr, d, owner)
        if mode == 'revoke':
            self._revoke_goalkeeper(self.tracks.get(owner.track) if owner.track is not None else None, owner, None, why, ctx, out, successor=successor)
        elif mode == 'deteam':
            self._deteam_goalkeeper(owner, successor, why, ctx, out)
        return bool(mode)

    def _revoke_goalkeeper(self, tr, g, d, why, ctx, out, successor=None):
        """Take the goalkeeper status away from an identity: its team's slot is free again at once, it no
        longer counts, seeds a keeper kit, blocks a team or matches returning keepers, and its track is
        re-decided (OUT, referee or player, whatever the evidence says). The identity is kept, marked."""
        t = ctx['time']
        was, slot = g.display, g.team
        if tr is not None and tr.player_id == g.pid:
            self._unbind(g)
            self._release(tr, f'goalkeeper status revoked: {why}')
            tr.role = d.role if d is not None and d.role in ('referee', 'player') else 'unknown'
            tr.team = d.team if d is not None and d.role == 'player' else None
            tr.role_confidence = (d.referee_confidence if d.role == 'referee' else d.role_confidence) if d is not None and tr.role != 'unknown' else 0.0
            tr.state = 'uncertain' if tr.role != 'unknown' else 'candidate'
            tr.reid.clear()
        g.status, g.role_locked, g.doubt, g.revoked = 'unknown', False, 0, why
        g.team = None
        g.role_history.append((r3(t), 'revoked', why))
        for person in g.emitted_all:
            if person.get('id') == g.pid:
                person.update(display=g.pid, team=None, role='UNKNOWN', label='GOALKEEPER_REVOKED')
        lines = ['GOALKEEPER REVOKED', f'Global ID: {g.pid} (was {was})', f'Local Track: {tr.id if tr is not None else "-"}', f'Reason: {why}']
        if d is not None:
            lines += self._keeper_lines(d)
        lines.append(f'Team {slot} goalkeeper slot: ' + (f'released to {successor}' if successor else 'released') if slot else 'Goalkeeper slot: none held')
        out['events'].append(self._event('role', tr.id if tr is not None else None, g.pid, '\n'.join(lines), d.scores if d is not None else None))
        out['issues'].append({'id': g.pid, 'time': t, 'reason': f'Goalkeeper status revoked: {why}.'})

    def _hold_keeper_candidate(self, tr, d, ctx, out, what):
        """A goalkeeper candidate without confirmed evidence keeps collecting it; the team's slot stays free."""
        k = (d.scores.get('keeper') or {}) if d is not None else {}
        tr.reason = f"goalkeeper candidate: {k.get('reason', 'waiting for evidence')}"
        t = ctx['time']
        if t-tr.keeper_logged >= 5.0 or t < tr.keeper_logged:
            tr.keeper_logged = t
            out['events'].append(self._event('role', tr.id, None, '\n'.join(
                ['GOALKEEPER CANDIDATE', f'Local Track: {tr.id}', f'Team: {tr.team or "unknown"}', what] + (self._keeper_lines(d) if d is not None else [])
                + [f"State: GK_CANDIDATE{'_'+tr.team if tr.team else ''}"]), d.scores if d is not None else None))

    def _special_match(self, tr, roles):
        """Best missing identity with one of `roles` whose gallery this track resembles: (identity, appearance)."""
        probes = self._probes(tr)
        best = None
        for g in self.registry.values():
            if g.role not in roles or g.status in ('substituted', 'unknown', 'retired') or not g.gallery:
                continue
            if g.track is not None and self._lost_for(g) < LOST:
                continue
            c = compare(g.gallery, probes) if probes else None
            a = None if c is None else c['appearance'] if c['appearance'] is not None else clamp(1-c['total']/.6)
            if a is not None and (best is None or a > best[1]):
                best = (g, a)
        return best

    def _best_appearance(self, tr, accept):
        """Highest appearance similarity of this track to any identity accepted by `accept` (visible or not)."""
        probes = self._probes(tr)
        best = 0.0
        for g in self.registry.values():
            if not accept(g) or g.status == 'retired' or not g.gallery or g.track == tr.id or not probes:
                continue
            c = compare(g.gallery, probes)
            if c is not None:
                best = max(best, c['appearance'] if c['appearance'] is not None else clamp(1-c['total']/.6))
        return best

    def _seed_gallery(self, target, tr, g=None):
        for desc, when, occluded in tr.descs:
            if not occluded:
                target.gallery = add_to_gallery(target.gallery, desc, when, self.o.gallery_size)
        for sample in (g.gallery if g is not None else []):
            target.gallery = add_to_gallery(target.gallery, sample.d, sample.time, self.o.gallery_size)

    def _resembles(self, samples, tr):
        """The person on track tr now looks like these appearance samples: the same person."""
        probes = self._probes(tr)
        c = compare(samples, probes) if samples and probes else None
        if c is None:
            return False
        return (c['appearance'] if c['appearance'] is not None else clamp(1-c['total']/.6)) >= .6

    def _always_this_person(self, g, tr):
        """Identity g's earliest samples show the person on track tr now: g was this person all along,
        not someone the local track switched away from."""
        return self._resembles(sorted(g.gallery, key=lambda x: x.time)[:2], tr)

    def _hand_over(self, tr, g, target, t, out):
        """Track tr (bound to g) continues as identity target. If g was this person all along, g is
        retired into target and every observation moves. Otherwise g is someone else (the local track
        switched people, or a wrong re-identification): g becomes missing, and this track's observations
        move only if the track followed one person from its start."""
        g.track = None
        if self._always_this_person(g, tr):
            g.status, g.retired_into = 'retired', target.pid
            g.role_history.append((r3(t), 'retired', f'was {target.pid}'))
            self._move_observations(g, target)
            return True
        g.status = 'missing'
        if self._resembles([Sample(d, 0.0) for d in tr.first], tr):
            self._move_observations(g, target, only_track=tr.id)
        else:
            out['issues'].append({'id': g.pid, 'time': t, 'reason': f'Track {tr.id} switched from {g.pid} to {target.pid} at an unknown time; its earlier observations keep {g.pid}. Check.'})
        return False

    def _old_role(self, g):
        return f'PLAYER (team {g.team})' if g.role == 'player' else ROLE_OUT[g.role]

    def _to_referee(self, tr, g, d, ctx, out):
        t = ctx['time']
        target = None
        match = self._special_match(tr, ('referee',))
        if match is not None and match[1] >= .7:
            target = match[0]
        if target is None:
            if self._count('referee') >= self.o.max_referees:
                if self._allow('ref-cap', t, 30):
                    out['events'].append(self._event('sanity', tr.id, g.pid, f'{g.display} looks like a referee, but {self._count("referee")} referee identities already exist; kept as is. Check.'))
                return
            target = GlobalPlayer(self._next_id_for('referee', None), 'referee', None, first_seen=g.first_seen, last_seen=t, last_box=list(tr.box), created_on=tr.id)
            self.registry[target.pid] = target
            self._seed_gallery(target, tr, g if self._always_this_person(g, tr) else None)
        old = self._old_role(g)
        born_here = self._hand_over(tr, g, target, t, out)
        tr.role, tr.team = 'referee', None
        tr.team_votes.clear()
        self._bind(tr, target, t, max(.6, g.identity_confidence))
        target.role_locked, target.role_confidence, target.referee_confidence = True, d.referee_confidence, d.referee_confidence
        target.role_history.append((r3(t), 'referee', f'from {g.pid}'))
        target.role_scores = d.scores
        if d.official:
            target.official = d.official
        lines = ['ROLE UPDATE', f'Global ID: {g.pid} -> {target.pid}' + (f' ({g.pid} retired: it was this referee)' if born_here else ''),
                 f'Local Track: {tr.id}', f'Old role: {old}', 'New role: REFEREE'] + self._role_lines(d) + [f'Referee confidence: {d.referee_confidence:.2f}']
        out['events'].append(self._event('role', tr.id, target.pid, '\n'.join(lines), {'referee': d.referee_confidence, **d.scores}))

    def _to_goalkeeper(self, tr, g, d, ctx, out):
        t = ctx['time']
        if not self._keeper_confirmed(d):
            # Behaves like a goalkeeper on some frames, but the evidence is not sustained and spatial: the
            # identity keeps its role and the team's goalkeeper slot stays free.
            self._hold_keeper_candidate(tr, d, ctx, out, f'Identity: {g.pid} keeps role {self._old_role(g)}')
            return
        rival = self.goalkeeper_of(d.team or g.team, exclude=g) if (d.team or g.team) is not None else None
        s = d.scores
        lines = self._keeper_lines(d)
        if rival is not None and g.role != 'referee':
            # The team's keeper returning under a player identity merges into their own goalkeeper identity
            # (the merge branch below); only a track that does not look like the absent rival competes for the slot.
            visible = rival.track is not None and self._lost_for(rival) < LOST
            match = self._special_match(tr, ('goalkeeper',)) if not visible else None
            resembles = match is not None and match[0] is rival and match[1] >= .6
            if not resembles and self._take_slot(tr, d, rival, g.pid, f'a stronger goalkeeper candidate for team {rival.team} ({g.pid}: {s["keeper"]["route"]})', ctx, out):
                rival = None
        if g.role == 'referee':
            # An unconfirmed referee identity that is really a goalkeeper: a goalkeeper identity takes over.
            match = self._special_match(tr, ('goalkeeper',))
            target = match[0] if match is not None and match[1] >= .7 else None
            if target is None:
                if self._count('goalkeeper') >= self.o.max_goalkeepers:
                    if self._allow('gk-cap', t, 30):
                        out['events'].append(self._event('sanity', tr.id, g.pid, f'{g.display} looks like a goalkeeper, but {self._count("goalkeeper")} goalkeeper identities already exist; kept as is. Check.'))
                    return
                team = d.team if d.team and (rival is None or self._take_slot(tr, d, rival, None, f'a stronger goalkeeper candidate for team {d.team} ({s["keeper"]["route"]})', ctx, out)) else None
                target = GlobalPlayer(self._next_id_for('goalkeeper', None), 'goalkeeper', team, first_seen=g.first_seen, last_seen=t, last_box=list(tr.box), created_on=tr.id)
                self.registry[target.pid] = target
                self._seed_gallery(target, tr, g if self._always_this_person(g, tr) else None)
            born_here = self._hand_over(tr, g, target, t, out)
            tr.role, tr.team = 'goalkeeper', target.team
            tr.team_votes.clear()
            self._bind(tr, target, t, max(.6, g.identity_confidence))
            target.role_locked, target.role_confidence, target.goalkeeper_confidence = True, d.goalkeeper_confidence, d.goalkeeper_confidence
            target.keeper_goal = max(target.keeper_goal, s['nearGoal'])
            target.role_history.append((r3(t), 'goalkeeper', f'from {g.pid}'))
            out['events'].append(self._event('role', tr.id, target.pid, '\n'.join(
                ['GOALKEEPER IDENTIFIED', f'Global ID: {g.pid} -> {target.pid}' + (f' ({g.pid} retired: it was this goalkeeper)' if born_here else ''),
                 f'Team: {target.team or "unknown yet"}', 'Old role: REFEREE', 'New role: GOALKEEPER'] + lines), {'goalkeeper': d.goalkeeper_confidence, **s}))
            return
        if rival is not None:
            visible = rival.track is not None and self._lost_for(rival) < LOST
            match = self._special_match(tr, ('goalkeeper',)) if not visible else None
            if visible or match is None or match[0] is not rival or match[1] < .6:
                if self._allow('gk-conflict-'+g.pid, t, 30):
                    out['events'].append(self._event('role', tr.id, g.pid, f'{g.pid} behaves like a goalkeeper, but team {g.team} already has goalkeeper {rival.pid} ({"visible at the same time" if visible else "who looks different"}); role kept. Check.'))
                return
            # The team's goalkeeper had been given a player identity on this track: merge into the goalkeeper.
            self._hand_over(tr, g, rival, t, out)
            tr.role = 'goalkeeper'
            self._bind(tr, rival, t, max(.6, g.identity_confidence))
            out['events'].append(self._event('role', tr.id, rival.pid, '\n'.join(['GOALKEEPER IDENTIFIED', f'Global ID: {g.pid} -> {rival.pid} ({rival.display})',
                                                                                    f'Team: {rival.team}'] + lines)))
            return
        g.role, g.role_locked = 'goalkeeper', True
        g.role_confidence = g.goalkeeper_confidence = d.goalkeeper_confidence
        g.keeper_goal = max(g.keeper_goal, s['nearGoal'])
        g.role_history.append((r3(t), 'goalkeeper', 'position and kit evidence'))
        tr.role = 'goalkeeper'
        self._restyle(g)
        out['events'].append(self._event('role', tr.id, g.pid, '\n'.join(['GOALKEEPER IDENTIFIED', f'Global ID: {g.pid} (shown as {g.display})', f'Team: {g.team}',
                                                                            'Old role: PLAYER', 'New role: GOALKEEPER'] + lines), {'goalkeeper': d.goalkeeper_confidence, **s}))

    # ---------- evidence for the role decision ----------
    def _decide(self, tr, ctx):
        """The temporal role decision with this person's whole-appearance similarity to the known referees
        and to each team's players, and whether the kit's team already has its full count of identities."""
        return decide_role(tr.evidence, ctx['model'], self.o.min_hits, self._looks(tr, ctx['time']), self._crowded(tr))

    def _crowded(self, tr):
        """{team: True} where that team already has max_per_team identities other than this track's own."""
        own = self.registry.get(tr.player_id) if tr.player_id else None
        out = {}
        for team in ('A', 'B'):
            n = self._count(team)-(1 if own is not None and own.role == 'player' and own.team == team else 0)
            out[team] = n >= self.o.max_per_team
        return out

    def _pools(self, t):
        """Reference appearance galleries, refreshed once a second: every locked referee with all its crops
        (the referee appearance gallery) and the confident players of each team."""
        if self.pools is not None and 0 <= t-self.pools_at < 1.0:
            return self.pools
        pools = {'referee': [], 'A': [], 'B': []}
        for g in self.registry.values():
            if g.status in ('retired', 'substituted', 'unknown') or not g.gallery:
                continue
            if g.role == 'referee' and g.role_locked:
                pools['referee'].append((g.pid, [x.d for x in g.gallery]))
            elif g.role == 'player' and g.team in ('A', 'B') and g.observations >= 5 and g.team_confidence >= .5:
                pools[g.team].append((g.pid, [x.d for x in g.gallery[-3:]]))
        self.pools, self.pools_at = pools, t
        return pools

    def _looks(self, tr, t):
        """Whole-appearance similarity of this person to the known referees (the most similar one) and to
        each team's players (the median player, so a few mislabelled identities cannot define a team's
        look): learned appearance plus the whole uniform, from the last few clean crops. A track's own
        identity is never its own evidence. Cached for a second."""
        pools = self._pools(t)
        if tr.looks is not None and tr.looks[0] == self.pools_at and 0 <= t-tr.looks[1] < 1.0:
            return tr.looks[2]
        probes = [d for d, _, occluded in tr.descs if not occluded and d.quality >= .3][-LOOKS_PROBES:] or [d for d, _, _ in tr.descs][-LOOKS_PROBES:]
        out = {'referee': None, 'A': None, 'B': None, 'probes': len(probes)}
        for group, members in pools.items():
            members = [(pid, descs) for pid, descs in members if pid != tr.player_id]
            if not probes or not members:
                continue
            sim = batch_similarity([d for _, descs in members for d in descs], probes)
            values, i = [], 0
            for _, descs in members:
                block = np.sort(sim[i:i+len(descs)].ravel())
                values.append(float(block[-2:].mean()))
                i += len(descs)
            out[group] = round(max(values) if group == 'referee' else float(np.median(values)), 3)
        tr.looks = (self.pools_at, t, out)
        return out

    def _behaviour_cues(self, live, ctx):
        """Per evidence tick, for everyone seen now: metres to the ball, where
        they sit in each team's shape (distance from that team's centroid in units of its spread, from the
        confirmed players of the team other than themselves) and whether they move with that team. Written
        onto the tick's evidence record; the role decision reads them over its window."""
        t, aspect = ctx['time'], ctx['aspect']
        seen = [tr for tr in live if tr.seen_at == t and tr.state != 'rejected' and tr.evidence.records]
        if not seen:
            return
        point = lambda tr: (tr.stab.x*aspect, tr.stab.y, max(tr.stab.h, .005))
        ball = ctx.get('ball')
        distances = {}
        if ball is not None:
            bx, by = ball['stab'][0]*aspect, ball['stab'][1]
            for tr in seen:
                if ball.get('pitch') is not None and tr.pitch is not None:
                    distances[tr.id] = math.hypot((tr.pitch[0]-ball['pitch'][0])*PITCH_LENGTH, (tr.pitch[1]-ball['pitch'][1])*PITCH_WIDTH)
                else:
                    x, y, h = point(tr)
                    distances[tr.id] = math.hypot(x-bx, y-by)/h*PLAYER_HEIGHT
        teams = {}
        for tr in seen:
            g = self.registry.get(tr.player_id) if tr.state == 'confirmed' and tr.player_id else None
            if g is not None and g.role == 'player' and g.team in ('A', 'B'):
                teams.setdefault(g.team, []).append(tr)
        for tr in seen:
            rec = tr.evidence.records[-1]
            if tr.id in distances:
                rec['ball_m'] = round(distances[tr.id], 1)
            x, y, h = point(tr)
            own = (tr.velocity[0]*aspect/h*PLAYER_HEIGHT, tr.velocity[1]/h*PLAYER_HEIGHT)
            formation, move = {}, {}
            for team, members in teams.items():
                members = [m for m in members if m is not tr]
                if len(members) < 3:
                    continue
                pts = [point(m) for m in members]
                cx, cy, ch = (sum(p[0] for p in pts)/len(pts), sum(p[1] for p in pts)/len(pts), sum(p[2] for p in pts)/len(pts))
                spread = sum(math.hypot(p[0]-cx, p[1]-cy) for p in pts)/len(pts)/ch*PLAYER_HEIGHT
                formation[team] = round(math.hypot(x-cx, y-cy)/((h+ch)/2)*PLAYER_HEIGHT/max(spread, 5.0), 2)
                vx = sum(m.velocity[0] for m in members)/len(members)*aspect/ch*PLAYER_HEIGHT
                vy = sum(m.velocity[1] for m in members)/len(members)/ch*PLAYER_HEIGHT
                if math.hypot(vx, vy) >= 1.0 and math.hypot(*own) >= 1.0:
                    move[team] = round((own[0]*vx+own[1]*vy)/(math.hypot(*own)*math.hypot(vx, vy)), 2)
            rec['formation'], rec['move'] = formation or None, move or None

    # ---------- re-identification ----------
    def _lost_for(self, g):
        tr = self.tracks.get(g.track) if g.track is not None else None
        return math.inf if tr is None else max(0.0, self.time-tr.last_time)

    def _same_group(self, tr, g):
        if tr.role == 'referee':
            return g.role == 'referee'
        if tr.role == 'goalkeeper':
            return g.role == 'goalkeeper'
        return g.role == 'player' and (tr.team is None or g.team is None or g.team == tr.team)

    def _brief_lost(self, tr):
        for g in self.registry.values():
            if g.track is not None and g.track != tr.id and self._same_group(tr, g) and 0 < self._lost_for(g) < LOST:
                return g
        return None

    def _candidates(self, tr):
        out = []
        for g in self.registry.values():
            if g.status in ('substituted', 'unknown', 'retired') or g.track == tr.id:
                continue
            if g.track is not None:
                if self._lost_for(g) < LOST:
                    continue
            elif g.status not in ('missing', 'offscreen'):
                continue
            if tr.role == 'referee':
                if g.role == 'referee':
                    out.append(g)
            elif tr.role == 'goalkeeper':
                if (tr.decision is not None and (tr.decision.scores.get('keeper') or {}).get('vetoed')):
                    continue  # feet on the sideline: not this (or any) goalkeeper until that changes
                if g.role == 'goalkeeper' and (tr.team is None or g.team is None or g.team == tr.team):
                    out.append(g)
            elif g.role == 'player' and g.team == tr.team:
                out.append(g)
        return out

    def _resolve(self, uncertain, ctx, out):
        o = self.o
        for tr in uncertain:
            if tr.retired or tr.role != 'player' or len(tr.descs) < 3:
                continue
            # A returning referee or goalkeeper whose kit voted for a team: their own appearance decides.
            match = self._special_match(tr, ('referee', 'goalkeeper'))
            if match is None or match[1] < .8:
                continue
            rival = self._best_appearance(tr, lambda g: g.role == 'player' and g.team == tr.team)
            if match[1] >= rival+.2:
                g = match[0]
                old = self._describe(tr.role, tr.team)
                closest = f'the closest team {tr.team} player' if tr.team else 'the closest player'
                tr.role, tr.team = g.role, (None if g.role == 'referee' else g.team or tr.team)
                tr.reid.clear()
                name = g.pid if g.display == g.pid else f'{g.display} ({g.pid})'
                out['events'].append(self._event('role', tr.id, g.pid, '\n'.join([
                    'ROLE UPDATE', f'Local Track: {tr.id}', f'Old role: {old} (kit vote)', f'New role: {ROLE_OUT[g.role]}',
                    f'Looks like {name}: appearance {match[1]:.2f} vs {rival:.2f} for {closest}'])))
        for tr in uncertain:
            if tr.retired:
                continue
            cands = self._candidates(tr)
            tr.blind = []
            if not cands:
                tr.reid.clear()
                tr.best = 0.0
                self._create_or_hold(tr, ctx, out)
                continue
            ids = {g.pid for g in cands}
            for k in list(tr.reid):
                if k not in ids:
                    del tr.reid[k]
            for g in cands:
                s = self._score(tr, g, ctx)
                if s['appearance'] is None and s['spatial'] is None:
                    tr.reid.pop(g.pid, None)
                    tr.blind.append((g.pid, s['temporal']))
                    continue
                tr.reid.setdefault(g.pid, deque(maxlen=WINDOW)).append(s)
        plans = []
        for tr in uncertain:
            if tr.state != 'uncertain' or tr.retired:
                continue
            names = ', '.join(pid for pid, _ in tr.blind)
            if not tr.reid:
                if not tr.blind:
                    continue
                tr.best = 0.0
                tr.reason = f'deferred: {names} cannot be compared yet (no appearance sample or position)'
                if self._defer_allowed(tr, ctx['time']):
                    out['events'].append(self._event('deferred', tr.id, tr.blind[0][0], f'Track {tr.id}: identity deferred; {names} cannot be compared yet.'))
                continue
            summaries = sorted((self._summarize(pid, steps) for pid, steps in tr.reid.items()), key=lambda s: (-s['final'], s['id']))
            best = summaries[0]
            stranger = blind = None
            free = 0 if tr.role in ('referee', 'goalkeeper') else self._free_count(tr.team)
            if free:
                prior = free/(free+len(tr.reid)+len(tr.blind))
                stranger = self._neutral(best, prior, self._teammate_baseline(tr, summaries))
            if tr.blind:
                blind = max(self._neutral(best, temporal, best['appearance']) for _, temporal in tr.blind)
            tr.best = best['final']
            plans.append({'tr': tr, 'all': summaries, 'need': 3 if best['spatialShare'] >= .5 else 5, 'stranger': stranger, 'blind': blind})

        def lead(p):
            return p['all'][0]['final']-max(p['all'][1]['final'] if len(p['all']) > 1 else 0.0, p['stranger'] or 0.0, p['blind'] or 0.0)

        def decisive(p):
            return p['all'][0]['n'] >= p['need'] and p['all'][0]['final'] >= o.reid_min and lead(p) >= o.reid_margin

        open_plans, taken, why = list(plans), set(), {}
        for _ in range(3):
            progress = False
            for p in sorted([p for p in open_plans if decisive(p)], key=lambda p: (-lead(p), p['tr'].id)):
                best = p['all'][0]
                lost = self._brief_lost(p['tr'])
                if lost is not None:
                    why[id(p)] = f'{lost.pid} just lost detection'
                    continue
                if best['id'] in taken:
                    why[id(p)] = f"{best['id']} was just assigned to another track"
                    continue
                rival = next((q for q in open_plans if q is not p and any(s['id'] == best['id'] and s['n'] >= 2 and s['final'] >= best['final']-o.reid_margin for s in q['all'])), None)
                if rival is not None:
                    why[id(p)] = f"track {rival['tr'].id} matches {best['id']} almost as well"
                    continue
                taken.add(best['id'])
                open_plans.remove(p)
                why.pop(id(p), None)
                self._reconnect(p['tr'], p['all'], ctx, out)
                progress = True
            if not progress:
                break
        for p in open_plans:
            tr, best = p['tr'], p['all'][0]

            def implausible(s):
                # Goalkeepers and referees do not change jerseys mid-match: a clearly different jersey rules
                # a missing one out, however well the place and the timing fit.
                other_kit = tr.role in ('goalkeeper', 'referee') and s['jersey'] <= .2 and s['appearance'] is not None and s['appearance'] < .6
                return s['final'] < .3 or (s['spatial'] is not None and s['spatialShare'] >= .5 and s['spatial'] < .15) or \
                    (p['stranger'] is not None and p['stranger']-s['final'] >= o.reid_margin) or other_kit
            hold = p['blind'] is not None or (tr.released is not None and ctx['time']-tr.released[1] < RELEASE_HOLD)
            if tr.role == 'goalkeeper' and self._keeper_confirmed(tr.decision) and ctx['time']-tr.born >= 10.0 and best['n'] >= p['need'] and not hold:
                # A keeper confirmed at a goal for 10 s is decided: the absent keeper it resembles enough, or
                # a new identity (which then competes for the team slot).
                if best['final'] >= .55 and self._brief_lost(tr) is None:
                    self._reconnect(tr, p['all'], ctx, out)
                    continue
                if self._brief_lost(tr) is None and self._create_or_hold(tr, ctx, out, rejected=p['all'][:3], stranger=p['stranger']):
                    continue
            if all(s['n'] >= p['need'] and implausible(s) for s in p['all']) and self._brief_lost(tr) is None and not hold:
                created = self._create_or_hold(tr, ctx, out, rejected=p['all'][:3], stranger=p['stranger'])
                if created:
                    continue
            if best['n'] < p['need'] and id(p) not in why:
                tr.reason = f"collecting re-ID evidence ({best['n']}/{p['need']})"
                continue
            second = p['all'][1]['final'] if len(p['all']) > 1 else 0.0
            if id(p) in why:
                reason = why[id(p)]
            elif best['final'] < o.reid_min:
                reason = f"best score {pct(best['final'])} below {pct(o.reid_min)}"
            elif p['blind'] is not None and p['blind'] >= second and p['blind'] >= (p['stranger'] or 0):
                reason = f"{', '.join(pid for pid, _ in tr.blind)} cannot be compared (no appearance or position)"
            elif p['stranger'] is not None and p['stranger'] >= second:
                reason = f"an unseen teammate would score {pct(p['stranger'])}"
            else:
                reason = f'lead {pct(lead(p))} below {pct(o.reid_margin)}'
            self._defer(tr, p['all'], ctx, out, reason)

    def _neutral(self, best, temporal, appearance):
        return combine({'appearance': appearance, 'uniform': best['uniform'],
                        'spatial': .5 if best['spatial'] is not None else None,
                        'movement': .5 if best['movement'] is not None else None, 'temporal': temporal, 'team': True})

    def _teammate_baseline(self, tr, summaries):
        """How alike this person looks to the most similar known teammate other than the best
        candidate: the other candidates and teammates visible right now. At most one identity is this
        person, so if the best candidate is right, these are all someone else; an unseen teammate is
        assumed to look about as alike. Without any such reference: as alike as the best candidate."""
        best = summaries[0]
        if best['appearance'] is None:
            return None
        values = [s['appearance'] for s in summaries[1:] if s['appearance'] is not None]
        probes = self._probes(tr)
        for g in self.registry.values():
            other = self.tracks.get(g.track) if g.track is not None else None
            if not probes or other is None or other.id == tr.id or other.seen_at != self.time or g.role != 'player' or g.team != tr.team or not g.gallery:
                continue
            c = compare(g.gallery, probes)
            if c and c['appearance'] is not None:
                values.append(c['appearance'])
        return max(values) if values else best['appearance']

    def _probes(self, tr):
        clean = [d for d, _, occluded in tr.descs if not occluded]
        return clean or [d for d, _, _ in tr.descs]

    def _score(self, tr, g, ctx):
        """Components in [0,1]; None = unknown (its weight is removed)."""
        gap = abs(tr.born-g.last_seen) if g.restored and g.track is None and g.segment is None else max(0.0, tr.born-g.last_seen)
        k = min(gap, 2.0)
        team = tr.role in ('referee', 'goalkeeper') or tr.team is None or g.team is None or tr.team == g.team
        c = compare(g.gallery, self._probes(tr)) if g.gallery and tr.descs else None
        appearance = c['appearance'] if c else None
        if c and appearance is None:
            appearance = clamp(1-c['total']/.6)  # colour-only fallback when no embeddings exist
        uniform = (clamp(1-c['shorts']/.6)+clamp(1-c['socks']/.6))/2 if c else None
        jersey = clamp(1-c['jersey']/.6) if c else .5
        same = g.segment is not None and g.segment == ctx['segment'] and tr.born_segment == ctx['segment'] and g.stab is not None
        released = tr.released is not None and tr.released[0] == g.pid and g.last_seen <= tr.released[1]+1e-6
        if released and same:
            dt = max(0.0, ctx['time']-g.last_seen)
            h = max(.005, (g.stab.h+tr.stab.h)/2)
            k2 = min(dt, 2.0)
            m = math.hypot((tr.stab.x-g.stab.x-g.velocity[0]*k2)*ctx['aspect'], tr.stab.y-g.stab.y-g.velocity[1]*k2)/h*PLAYER_HEIGHT
            spatial = math.exp(-.5*(m/(3+7*dt))**2)
            final = combine({'appearance': appearance, 'uniform': uniform, 'spatial': spatial, 'temporal': 1.0, 'team': team})
            return {'appearance': appearance, 'jersey': jersey, 'uniform': uniform, 'team': team, 'spatial': spatial,
                    'movement': None, 'temporal': 1.0, 'final': final, 'missing': dt, 'cosine': c.get('cosine') if c else None}
        # Detected while the identity was still detected on another track: two different people.
        together = same and tr.born < g.last_seen-1e-3 and not released
        spatial = movement = None
        sigma = 3+7*gap
        if tr.first_pitch is not None and g.last_pitch is not None and sigma <= SPATIAL_HORIZON:
            v = g.pitch_velocity or (0.0, 0.0)
            m = math.hypot((tr.first_pitch[0]-g.last_pitch[0]-v[0]*k)*PITCH_LENGTH, (tr.first_pitch[1]-g.last_pitch[1]-v[1]*k)*PITCH_WIDTH)
            spatial = math.exp(-.5*(m/sigma)**2)
        if same:
            extra = max(0, ctx['drift']-g.drift)
            h = max(.005, (g.stab.h+tr.first_stab.h)/2)
            metres = lambda dx, dy: math.hypot(dx*ctx['aspect'], dy)/h*PLAYER_HEIGHT
            loose = sigma+5*extra
            if spatial is None and extra <= 6 and loose <= SPATIAL_HORIZON:
                m = metres(tr.first_stab.x-g.stab.x-g.velocity[0]*k, tr.first_stab.y-g.stab.y-g.velocity[1]*k)
                spatial = math.exp(-.5*(m/loose)**2)
            # Exit edge vs entry edge, and (after a short gap) whether the new track appeared where the old
            # one was heading. After a longer absence a returning player usually comes back the other way.
            exit_edge, entry = g.exit_edge, tr.entry_edge
            e = (1.0 if entry == exit_edge else .25 if entry else .35) if exit_edge else (.4 if entry else .8)
            speed = metres(*g.velocity)
            dx, dy = tr.first_stab.x-g.stab.x, tr.first_stab.y-g.stab.y
            if gap < 3 and speed > 1 and metres(dx, dy) > 1:
                a = ctx['aspect']
                cos = (dx*g.velocity[0]*a*a+dy*g.velocity[1])/(math.hypot(dx*a, dy)*math.hypot(g.velocity[0]*a, g.velocity[1]))
                e = .7*e+.3*(.5+.5*cos)
            movement = clamp(e)
        if together:
            spatial = 0.0
        temporal = clamp(math.exp(-gap/30), .2, 1)
        final = 0.0 if together else combine({'appearance': appearance, 'uniform': uniform, 'spatial': spatial,
                                               'movement': movement, 'temporal': temporal, 'team': team})
        return {'appearance': appearance, 'jersey': jersey, 'uniform': uniform, 'team': team, 'spatial': spatial,
                'movement': movement, 'temporal': temporal, 'final': final, 'missing': gap, 'cosine': c.get('cosine') if c else None}

    @staticmethod
    def _summarize(pid, steps):
        steps = list(steps)
        mean = lambda key: (sum(s[key] for s in steps if s[key] is not None)/max(1, sum(1 for s in steps if s[key] is not None))) \
            if any(s[key] is not None for s in steps) else None
        return {'id': pid, 'n': len(steps), 'final': sum(s['final'] for s in steps)/len(steps), 'appearance': mean('appearance'),
                'jersey': sum(s['jersey'] for s in steps)/len(steps), 'uniform': mean('uniform'), 'team': all(s['team'] for s in steps),
                'spatial': mean('spatial'), 'spatialShare': sum(1 for s in steps if s['spatial'] is not None)/len(steps),
                'movement': mean('movement'), 'temporal': sum(s['temporal'] for s in steps)/len(steps), 'missing': steps[-1]['missing'],
                'cosine': mean('cosine')}

    @staticmethod
    def _scores(s, second=None):
        v = lambda x: None if x is None else r3(x)
        out = {'appearance': v(s['appearance']), 'osnetCosine': v(s['cosine']), 'jersey': r3(s['jersey']), 'uniform': v(s['uniform']),
               'team': s['team'], 'spatial': v(s['spatial']), 'movement': v(s['movement']), 'temporal': r3(s['temporal']),
               'final': r3(s['final']), 'missingSeconds': r3(s['missing'])}
        if second is not None:
            out['secondBest'] = r3(second['final'])
        return out

    def _reconnect(self, tr, summaries, ctx, out):
        best = summaries[0]
        second = summaries[1] if len(summaries) > 1 else None
        g = self.registry.get(best['id'])
        if g is None:
            return
        if g.track is not None and g.track != tr.id:
            ghost = g.track
            self.tracks.pop(ghost, None)
            out['drop'].append(ghost)
            self._drop_pairs(ghost, out)
        crossed = self._bind(tr, g, ctx['time'], best['final'])
        lines = ['RE-ID EVENT', f'Local Track: {tr.id}', f'Matched Global Player: {long_id(g.pid)}']
        if best['appearance'] is not None:
            lines.append(f"Appearance similarity: {best['appearance']:.2f}" + (f" (OSNet cosine {best['cosine']:.2f})" if best['cosine'] is not None else ''))
        lines += [f"Team match: {'yes' if best['team'] else 'no'}", f"Jersey similarity: {best['jersey']:.2f}"]
        lines.append(f"Spatial plausibility: {best['spatial']:.2f}" if best['spatial'] is not None else 'Spatial plausibility: not comparable (camera cut or long gap)')
        lines += [f"Time missing: {best['missing']:.1f} seconds", f"Final identity confidence: {best['final']:.2f}"]
        lines.append(f"Second best: {second['id']} {second['final']:.2f}" if second else
                     f"Second best: none ({', '.join(pid for pid, _ in tr.blind)} not comparable)" if tr.blind else 'Second best: none (only missing candidate)')
        if crossed:
            lines.append('Follows a crossing: confidence capped, check this identity.')
            out['issues'].append({'id': g.pid, 'time': ctx['time'], 'reason': 'Re-identified shortly after a crossing: check identity.'})
        out['events'].append(self._event('reid', tr.id, g.pid, '\n'.join(lines), self._scores(best, second), accepted=True))
        for s in summaries[1:4]:
            out['events'].append(self._event('reid-rejected', tr.id, s['id'],
                                             f"RE-ID REJECTED\nLocal Track: {tr.id}\nCandidate: {long_id(s['id'])}\nScore {s['final']:.2f}, {best['final']-s['final']:.2f} behind {best['id']}.", self._scores(s)))
        self._flush(tr, g, ctx['time'])

    def _defer(self, tr, summaries, ctx, out, reason):
        best = summaries[0]
        second = summaries[1] if len(summaries) > 1 else None
        versus = f" vs {second['id']} {pct(second['final'])}" if second else ''
        tr.reason = f"deferred: {best['id']} {pct(best['final'])}{versus} ({reason})"
        if not self._defer_allowed(tr, ctx['time']):
            return
        lines = ['RE-ID DEFERRED', f'Local Track: {tr.id}', f"Best: {long_id(best['id'])} {best['final']:.2f}"]
        for s in summaries[1:3]:
            lines.append(f"Also possible: {long_id(s['id'])} {s['final']:.2f}")
        lines.append(f'Reason: {reason}')
        out['events'].append(self._event('deferred', tr.id, best['id'], '\n'.join(lines), self._scores(best, second)))

    def _create_or_hold(self, tr, ctx, out, rejected=(), stranger=None):
        lost = self._brief_lost(tr)
        if lost is not None:
            tr.reason = f'waiting: {lost.pid} lost detection {self._lost_for(lost):.1f} s ago'
            return False
        if tr.role == 'goalkeeper' and not self._keeper_confirmed(tr.decision):
            # A provisional goalkeeper never creates (and so never reserves) a goalkeeper identity.
            self._hold_keeper_candidate(tr, tr.decision, ctx, out, 'No goalkeeper identity created yet')
            return False
        group = 'referee' if tr.role == 'referee' else 'goalkeeper' if tr.role == 'goalkeeper' else tr.team
        n, cap = self._count(group), self._cap(group)
        over = n >= cap
        if over:
            observed = ctx['time']-tr.born
            hard = n >= cap+5 or tr.role in ('referee', 'goalkeeper')
            if hard or observed < self.o.over_cap_seconds or not rejected:
                tr.reason = f'{self._group_name(group)} already has {n} identities: kept unidentified'
                if self._allow('cap-'+str(group), ctx['time'], 5):
                    out['events'].append(self._event('sanity', tr.id, None,
                                                     f'{self._group_name(group)} already has {n} identities (expected at most {cap}); track {tr.id} stays unidentified. Likely a missed re-identification.'))
                return False
        for s in rejected:
            why = f"score {pct(s['final'])}" if s['final'] < .3 else \
                f"spatially implausible ({pct(s['spatial'])})" if s['spatial'] is not None and s['spatial'] < .15 else \
                f"an unseen teammate fits better ({pct(stranger or 0)} vs {pct(s['final'])})"
            out['events'].append(self._event('reid-rejected', tr.id, s['id'], f"RE-ID REJECTED\nLocal Track: {tr.id}\nCandidate: {long_id(s['id'])}\nReason: {why}", self._scores(s)))
        pid = self._next_id(tr)
        role = tr.role if tr.role in ('referee', 'goalkeeper') else 'player'
        team = tr.team if role != 'referee' else None
        if role == 'goalkeeper' and team is not None:
            # The team from the keeper-kit vote is evidence, not ownership: the slot decides.
            owner = self.goalkeeper_of(team)
            if owner is not None and not self._take_slot(tr, tr.decision, owner, pid, f'a stronger goalkeeper candidate for team {team} ({pid}: {(tr.decision.scores.get("keeper") or {}).get("route", "")})', ctx, out):
                team, tr.team = None, None
        g = GlobalPlayer(pid, role, team, first_seen=ctx['time'], last_seen=ctx['time'], last_box=list(tr.box), created_on=tr.id)
        if role == 'goalkeeper' and tr.decision is not None:
            g.keeper_goal = tr.decision.scores['nearGoal']
        if role != 'player':
            g.role_locked = tr.role_confidence >= ROLE_LOCK
            g.role_history.append((r3(ctx['time']), role, f'created with role confidence {tr.role_confidence:.2f}'))
            if tr.decision is not None:
                g.referee_confidence, g.goalkeeper_confidence = tr.decision.referee_confidence, tr.decision.goalkeeper_confidence
                g.role_scores, g.official = tr.decision.scores, (tr.decision.official if role == 'referee' else '')
        self.registry[pid] = g
        for d, t, occluded in tr.descs:
            if not occluded:
                g.gallery = add_to_gallery(g.gallery, d, t, self.o.gallery_size)
        conf = tr.role_confidence if tr.role in ('referee', 'goalkeeper') else tr.team_confidence
        crossed = self._bind(tr, g, ctx['time'], clamp(.55+.4*conf, .5, .95))
        why = 'every missing identity of this group is implausible here' if rejected else f'no missing {self._group_name(group)} identity'
        if over:
            why += f'; {self._group_name(group)} now has {n+1} identities (expected at most {cap}), check for duplicates'
            out['issues'].append({'id': pid, 'time': ctx['time'], 'reason': f'{self._group_name(group)} exceeds the expected {cap} identities.'})
        out['events'].append(self._event('new-identity', tr.id, pid, f'Track {tr.id} becomes new identity {long_id(pid)} ({self._describe(g.role, g.team)}; {why}{"; follows a crossing, check" if crossed else ""}).'))
        d = tr.decision
        if role != 'player' and d is not None:
            if role == 'goalkeeper':
                lines = ['GOALKEEPER IDENTIFIED', f'Global ID: {pid}', f'Team: {g.team or "unknown yet"}'] + self._keeper_lines(d)
            else:
                lines = ['REFEREE IDENTIFIED', f'Global ID: {pid}', f'Local Track: {tr.id}'] + self._role_lines(d) + [f'Referee confidence: {d.referee_confidence:.2f}']
            out['events'].append(self._event('role', tr.id, pid, '\n'.join(lines + [f"Role locked: {'yes' if g.role_locked else 'not yet'}"]),
                                             {role: d.goalkeeper_confidence if role == 'goalkeeper' else d.referee_confidence, **d.scores}))
        self._flush(tr, g, ctx['time'])
        return True

    def _next_id(self, tr):
        return self._next_id_for(tr.role, tr.team)

    def _next_id_for(self, role, team):
        if role == 'referee':
            key, fmt = 'REF', 'REF-{}'
        elif role == 'goalkeeper':
            key, fmt = 'GK', 'GK-{}'
        else:
            key, fmt = team, team+'-{:02d}'
        while True:
            self.counters[key] += 1
            pid = fmt.format(self.counters[key])
            if pid not in self.registry:
                return pid

    def _group_name(self, group):
        return {'referee': 'Officials', 'goalkeeper': 'Goalkeepers'}.get(group, f'Team {group}')

    def _cap(self, group):
        return self.o.max_referees if group == 'referee' else self.o.max_goalkeepers if group == 'goalkeeper' else self.o.max_per_team

    def _count(self, group):
        n = 0
        for g in self.registry.values():
            if g.status in ('substituted', 'unknown', 'retired'):
                continue
            if group == 'referee':
                n += g.role == 'referee'
            elif group == 'goalkeeper':
                n += g.role == 'goalkeeper'
            else:
                n += g.role == 'player' and g.team == group
        return n

    def _free_count(self, team):
        return max(0, self.o.max_per_team-self._count(team))

    def _bind(self, tr, g, time, conf):
        tr.player_id, tr.state, tr.blind, tr.reason = g.pid, 'confirmed', [], ''
        tr.reid.clear()
        g.doubt = 0
        if tr.team is None and g.team is not None:
            tr.team = g.team
        g.track, g.status, g.here, g.bound_at, g.restored = tr.id, 'active', True, time, False
        g.identity_confidence = r3(clamp(conf))
        g.team_confidence = r3(tr.team_confidence) if g.role == 'player' else g.team_confidence
        g.role_confidence = r3(max(g.role_confidence, tr.role_confidence) if g.role_locked else tr.role_confidence)
        g.cap_until = 0.0
        crossed = time-g.crossed_at < 10 or time-tr.crossed_at < 10
        if crossed:
            g.cap_until = time+SWAP_CAP
        if g.role in ('referee', 'goalkeeper'):
            # A player identity created on this track that shows this same person (not someone the track
            # switched away from) was this referee / goalkeeper: retire it and move its observations.
            for pid in list(tr.former):
                old = self.registry.get(pid)
                if old is not None and old is not g and old.role == 'player' and old.created_on == tr.id and old.track is None and \
                        old.status != 'retired' and self._always_this_person(old, tr):
                    old.status, old.retired_into = 'retired', g.pid
                    old.role_history.append((r3(time), 'retired', f'was {g.pid}'))
                    self._move_observations(old, g)
            tr.former.clear()
        return crossed

    def _unbind(self, g):
        g.track, g.status, g.exit_edge = None, 'missing', ''

    def _release(self, tr, why):
        if tr.player_id:
            tr.former.add(tr.player_id)
        tr.player_id = None
        tr.state = 'uncertain' if tr.decision is not None and tr.role != 'unknown' else 'candidate'
        tr.reid.clear()
        tr.reason = why

    def _conf(self, g, time):
        return r3(min(.5, g.identity_confidence) if time < g.cap_until else g.identity_confidence)

    def _style(self, person, g, conf):
        """Write identity, team and role (separately) onto an output person."""
        person.update(id=g.pid, display=g.display, team=g.team, role=ROLE_OUT.get(g.role, 'UNKNOWN'), label=g.role_label,
                      state='confirmed', identityConfidence=conf, roleConfidence=r3(g.role_confidence))
        person.pop('reason', None)

    def _restyle(self, g):
        """After a role or team change of an identity, every observation already labelled with it follows."""
        for person in g.emitted_all:
            if person.get('id') == g.pid:
                person.update(display=g.display, team=g.team, role=ROLE_OUT.get(g.role, 'UNKNOWN'), label=g.role_label)

    def _move_observations(self, old, new, only_track=None):
        """Relabel observations of `old` (all, or one local track's) as identity `new`."""
        keep = []
        for person in old.emitted_all:
            if person.get('id') == old.pid and (only_track is None or person.get('track') == only_track):
                self._style(person, new, min(person.get('identityConfidence', 1), new.identity_confidence or 1))
                person['evidence'] = 'relabelled'
                new.emitted_all.append(person)
                new.observations += 1
                old.observations -= 1
            else:
                keep.append(person)
        old.emitted_all = keep

    def _flush(self, tr, g, time):
        """Observations of a track that was not yet identified are labelled retroactively
        ('reidentified'), only for times when the identity had no observation of its own."""
        conf = self._conf(g, time)
        while tr.pending:
            person = tr.pending.popleft()
            if g.last_observed+1e-6 < person['time'] < time-1e-6:
                self._style(person, g, conf)
                person['evidence'] = 'reidentified'
                g.emitted_all.append(person)
                g.observations += 1

    # ---------- crossings ----------
    def _guard_swaps(self, live, ctx, out):
        """Confirmed tracks that overlap are paired; once they separate the identities are compared
        KEEP vs SWAP. Appearance must favour a swap; trajectories alone never swap identities."""
        t = ctx['time']
        for key, p in list(self.pairs.items()):
            a, b = self.tracks.get(p['a']), self.tracks.get(p['b'])
            valid = a is not None and b is not None and a.state == 'confirmed' and b.state == 'confirmed' and \
                a.player_id == p['ids'][0] and b.player_id == p['ids'][1] and t-p['start'] <= 6
            if not valid:
                del self.pairs[key]
                if a is not None and b is not None and t-p['start'] > 6:
                    self._dissolve(p, out)
                continue
            if a.seen_at != t or b.seen_at != t:
                p['hard'], p['sep'] = True, 0
                continue
            overlap = iou(a.box, b.box)
            if overlap > .4:
                p['hard'] = True
            p['sep'] = p['sep']+1 if overlap < .05 and not self._centre_close(a.box, b.box, ctx['aspect']) else 0
            clean = lambda x: x.current is not None and x.current.quality >= CLEAN and not x.occluded
            if p['sep'] >= 2 and ((clean(a) and clean(b)) or p['sep'] >= 6):
                del self.pairs[key]
                self._judge_crossing(a, b, p, ctx, out, clean(a) and clean(b))
        for i in range(len(live)):
            for j in range(i+1, len(live)):
                a, b = live[i], live[j]
                if a.state != 'confirmed' or b.state != 'confirmed':
                    continue
                key = (min(a.id, b.id), max(a.id, b.id))
                if key in self.pairs:
                    continue
                overlap = iou(a.box, b.box)
                if overlap <= .15 and not self._centre_close(a.box, b.box, ctx['aspect']):
                    continue
                snap = lambda x: next(((h[0], h[1], h[2]) for h in reversed(x.hist) if h[0] < t-1e-6), (t, x.stab.x, x.stab.y))
                self.pairs[key] = {'a': a.id, 'b': b.id, 'ids': (a.player_id, b.player_id), 'start': t, 'sep': 0, 'hard': overlap > .4,
                                   'snap': {a.id: snap(a)+(a.velocity,), b.id: snap(b)+(b.velocity,)}}

    @staticmethod
    def _centre_close(a, b, aspect):
        return math.hypot((a[0]+a[2]/2-b[0]-b[2]/2)*aspect, a[1]+a[3]/2-b[1]-b[3]/2) < .6*(a[2]+b[2])/2*aspect

    def _appearance_distance(self, g, tr):
        if tr.current is None or not g.gallery:
            return None
        c = compare(g.gallery, [tr.current])
        if c is None:
            return None
        return 1-c['appearance'] if c['appearance'] is not None else clamp(c['total']/.6)

    def _judge_crossing(self, a, b, p, ctx, out, clean):
        ga, gb = self.registry.get(p['ids'][0]), self.registry.get(p['ids'][1])
        if ga is None or gb is None:
            return
        t = ctx['time']
        h = max(.005, (a.stab.h+b.stab.h)/2)

        def predicted(s):
            k = min(2.0, t-s[0])
            return s[1]+s[3][0]*k, s[2]+s[3][1]*k
        qa, qb = predicted(p['snap'][a.id]), predicted(p['snap'][b.id])
        m = lambda st, q: math.hypot((st.x-q[0])*ctx['aspect'], st.y-q[1])/h*PLAYER_HEIGHT
        trajectory = (m(a.stab, qb)+m(b.stab, qa))-(m(a.stab, qa)+m(b.stab, qb))  # metres; > 0 favours KEEP
        d = lambda g, tr: self._appearance_distance(g, tr) if clean else None
        aa, ab, bb, ba = d(ga, a), d(ga, b), d(gb, b), d(gb, a)
        look = (ab+ba)-(aa+bb) if None not in (aa, ab, bb, ba) else None  # > 0 favours KEEP
        names = f'{ga.pid} and {gb.pid}'
        if look is not None and (look <= -.5 or (look <= -.2 and trajectory < 4)):
            why = f'appearance favours the swap by {-look:.2f}, trajectory {trajectory:.1f} m'
            a.player_id, b.player_id = gb.pid, ga.pid
            ga.track, gb.track = b.id, a.id
            ga.bound_at = gb.bound_at = t
            a.team, b.team, a.role, b.role = gb.team, ga.team, gb.role, ga.role
            a.team_votes.clear()
            b.team_votes.clear()
            # Observations emitted since the crossing began follow the corrected identities.
            for tr, g in ((a, gb), (b, ga)):
                for when, person in tr.emitted:
                    if when >= p['start']-1e-6:
                        moved_from = self.registry.get(person.get('id'))
                        if moved_from is not None and person in moved_from.emitted_all:
                            moved_from.emitted_all.remove(person)
                        self._style(person, g, min(person.get('identityConfidence', 1), .5))
                        g.emitted_all.append(person)
                out['events'].append(self._event('swap-corrected', tr.id, g.pid, f'Identities of {names} swapped back after crossing: {g.pid} is track {tr.id} ({why}).'))
            return
        similar = look is None or abs(look) < .2
        if (similar and (p['hard'] or trajectory < 2)) or (look is not None and look <= -.2):
            why = (f"appearance too similar{', one was hidden' if p['hard'] else ''}, paths {'favour keeping' if trajectory >= 0 else 'favour swapping'} by {abs(trajectory):.1f} m"
                   if similar else f'appearance suggests a swap but the paths disagree ({trajectory:.1f} m)')
            self._flag_crossing([ga, gb], [a, b], a.id, f'{names} crossed (tracks {a.id}/{b.id}); {why}. Identities kept; confidence lowered. Check identities.', out)

    def _flag_crossing(self, ids, tracks, track, message, out):
        for g in ids:
            g.cap_until = max(g.cap_until, self.time+SWAP_CAP)
            g.crossed_at = self.time
        for tr in tracks:
            tr.crossed_at = self.time
        if not self._allow('cross-'+'|'.join(sorted(g.pid for g in ids)), self.time, 2):
            return
        out['events'].append(self._event('swap-uncertain', track, ids[0].pid, message))
        for g in ids:
            out['issues'].append({'id': g.pid, 'time': self.time, 'reason': 'Crossed a similar player: check identities.'})

    def _dissolve(self, p, out):
        if not p['hard']:
            return
        ids = [g for g in (self.registry.get(p['ids'][0]), self.registry.get(p['ids'][1])) if g is not None]
        tracks = [tr for tr in (self.tracks.get(p['a']), self.tracks.get(p['b'])) if tr is not None]
        if ids:
            self._flag_crossing(ids, tracks, p['a'], f"{' and '.join(g.pid for g in ids)} were hidden together and did not separate cleanly. Identities kept; confidence lowered. Check identities.", out)

    def _drop_pairs(self, tid, out):
        for key, p in list(self.pairs.items()):
            if tid in (p['a'], p['b']):
                del self.pairs[key]
                self._dissolve(p, out)

    def _check_reacquired(self, live, ctx, out):
        """A confirmed track picked up again after being hidden right next to a similar person may now
        follow that person: confidence is lowered and the identity flagged (never swapped)."""
        for tr in live:
            if not tr.reacquired or tr.state != 'confirmed' or tr.player_id is None:
                continue
            if any(tr.id in (p['a'], p['b']) for p in self.pairs.values()):
                continue
            g = self.registry.get(tr.player_id)
            w = tr.box[2]*ctx['aspect']
            cx, cy = tr.box[0]+tr.box[2]/2, tr.box[1]+tr.box[3]/2
            similar = next((o for o in live if o is not tr and o.state not in ('rejected',) and not o.retired and
                            (o.team == tr.team and o.role == tr.role if o.state == 'confirmed' else True) and
                            math.hypot((o.box[0]+o.box[2]/2-cx)*ctx['aspect'], o.box[1]+o.box[3]/2-cy) < 3*w), None)
            if similar is None or g is None:
                continue
            g.cap_until = max(g.cap_until, ctx['time']+SWAP_CAP)
            if self._allow('reacq-'+g.pid, ctx['time'], 2):
                out['events'].append(self._event('swap-uncertain', tr.id, g.pid, f'{g.pid} (track {tr.id}) re-acquired after being hidden next to a similar player (track {similar.id}). Identity kept; confidence lowered.'))

    # ---------- goalkeeper team ----------
    def _goalkeeper_teams(self, live, ctx, out):
        """Which team defends which goal, and so which team a goalkeeper belongs to. Whenever both teams
        have outfield players in view, the two outfield players deepest towards each side are usually
        that side's defenders (offside line): their team votes for that side. A goalkeeper near a goal
        then belongs to the team defending it; without a known goal side, the deepest players next to
        the goalkeeper vote directly. Decisions need >= 15 votes and a 75% majority."""
        t = ctx['time']
        players = [(tr, self.registry[tr.player_id]) for tr in live if tr.state == 'confirmed' and tr.player_id in self.registry]
        outfield = [(tr, g) for tr, g in players if g.role == 'player' and g.team and tr.seen_at == t]
        both = sum(g.team == 'A' for _, g in outfield) >= 2 and sum(g.team == 'B' for _, g in outfield) >= 2
        if both:
            for side, sign in (('left', -1), ('right', 1)):
                deepest = sorted(outfield, key=lambda item: -sign*item[0].stab.x)[:2]
                if deepest[0][1].team == deepest[1][1].team:
                    self.side_votes[side][deepest[0][1].team] += 1
        xs = sorted(tr.stab.x for tr, _ in outfield)
        for tr, g in players:
            if g.role != 'goalkeeper' or tr.seen_at != t:
                continue
            records = [r for r in tr.evidence.records if r['goal_side']]
            side = Counter(r['goal_side'] for r in records).most_common(1)[0][0] if len(records) >= 5 else None
            if g.team is not None:
                if side is not None:
                    self.side_votes[side][g.team] += 2  # a known goalkeeper at a goal: strong evidence for the side
                continue
            team, why = None, ''
            votes = self.side_votes.get(side) if side else None
            if votes and sum(votes.values()) >= 15:
                best, n = votes.most_common(1)[0]
                if n/sum(votes.values()) >= .75:
                    team, why = best, f'defends the {side} goal (team {best} defended that side in {n} of {sum(votes.values())} observations)'
            if team is None and both and xs:
                middle = xs[len(xs)//2]
                sign = 1 if tr.stab.x > middle else -1
                deepest = sorted(outfield, key=lambda item: -sign*item[0].stab.x)[:2]
                if all(sign*(o.stab.x-middle) > 0 for o, _ in deepest) and deepest[0][1].team == deepest[1][1].team:
                    g.gk_votes[deepest[0][1].team] += 1
                total = sum(g.gk_votes.values())
                if total >= 15:
                    best, n = g.gk_votes.most_common(1)[0]
                    if n/total >= .75:
                        team, why = best, f'the deepest outfield players next to this goalkeeper were team {best} in {n} of {total} observations'
            if team is None:
                continue
            owner = self.goalkeeper_of(team, exclude=g)
            if owner is not None:
                d = tr.decision
                if d is not None and self._take_slot(tr, d, owner, g.pid, f'a stronger goalkeeper for team {team} ({g.pid}: {d.scores["keeper"]["route"]})', ctx, out):
                    pass
                else:
                    if self._allow('gk-conflict-'+g.pid, t, 30):
                        out['events'].append(self._event('role', tr.id, g.pid, f'{g.pid} defends like team {team}, but team {team} already has goalkeeper {owner.pid}'
                                                         f' with {"stronger" if owner.goalkeeper_confidence >= g.goalkeeper_confidence else "comparable"} evidence; team left unknown.'))
                    continue
            g.team, tr.team = team, team
            g.team_confidence = tr.team_confidence = .8
            g.role_history.append((r3(t), 'goalkeeper', f'team {team}'))
            self._restyle(g)
            out['events'].append(self._event('role', tr.id, g.pid, f'GOALKEEPER TEAM\nGlobal ID: {g.pid} (now shown as {g.display})\nTeam: {team}\nReason: {why}'))

    # ---------- per-frame output ----------
    def _refresh(self, live, ctx, out, tick):
        t = ctx['time']
        visible = []
        for tr in live:
            g = self.registry.get(tr.player_id) if tr.state == 'confirmed' and tr.player_id else None
            if tr.state == 'confirmed' and (g is None or g.track != tr.id):
                tr.player_id, tr.state = None, 'uncertain'
                g = None
            person = {'time': t, 'track': tr.id, 'box': [round(float(v), 5) for v in tr.box], 'score': round(float(tr.score), 3),
                      'cls': tr.cls, 'zone': tr.zone, 'evidence': 'observed'}
            if tr.pitch is not None:
                person['pitch'] = [round(float(tr.pitch[0]), 4), round(float(tr.pitch[1]), 4)]
            if tick and tr.decision is not None and tr.decision.why:
                person['why'] = tr.decision.why
            if g is not None and tick:
                self._check_mismatch(tr, g, ctx, out)
                if tr.player_id != g.pid:
                    g = None
            if g is None:
                display, label = self._unbound_label(tr)
                role = ROLE_OUT.get(tr.role, 'UNKNOWN') if tr.state == 'uncertain' else 'UNKNOWN'
                person.update(id=None, display=display, label=label, team=tr.team if tr.role != 'referee' else None, role=role, state=tr.state,
                              identityConfidence=r3(tr.best if tr.state == 'uncertain' else 0), roleConfidence=r3(tr.role_confidence))
                if tr.reason:
                    person['reason'] = tr.reason[:160]
                if not tr.retired and tr.state in ('candidate', 'uncertain'):
                    tr.pending.append(person)
                    while tr.pending and t-tr.pending[0]['time'] > BUFFER_SECONDS:
                        tr.pending.popleft()
                out['people'].append(person)
                continue
            g.last_seen, g.last_box, g.stab, g.segment, g.drift = t, list(tr.box), tr.stab, ctx['segment'], ctx['drift']
            g.velocity, g.exit_edge, g.status = tr.velocity, '', 'active'
            if tr.pitch is not None:
                g.last_pitch, g.pitch_velocity = tr.pitch, tr.pitch_velocity
            else:
                g.last_pitch = g.pitch_velocity = None
            if tick:
                if tr.current is not None and not tr.occluded and tr.zone != 'outside' and self._conf(g, t) >= .6:
                    g.gallery = add_to_gallery(g.gallery, tr.current, t, self.o.gallery_size)
                g.history.append({'t': r3(t), 'box': person['box'], **({'pitch': person['pitch']} if 'pitch' in person else {})})
            visible.append(g)
            if tr.zone == 'outside' and tr.outside_run > 10:
                person.update(id=None, display=g.display, label=g.role_label, team=g.team, role=ROLE_OUT.get(g.role, 'UNKNOWN'), state='confirmed',
                              identityConfidence=0.0, roleConfidence=r3(g.role_confidence), reason='long outside the pitch: not saved as a match observation')
                out['people'].append(person)
                continue
            self._style(person, g, self._conf(g, t))
            g.emitted_all.append(person)
            g.observations += 1
            g.last_observed = t
            tr.emitted.append((t, person))
            while tr.emitted and t-tr.emitted[0][0] > 8:
                tr.emitted.popleft()
            out['people'].append(person)
        for i in range(len(visible)):
            for j in range(i+1, len(visible)):
                if visible[i].team and visible[i].team == visible[j].team:
                    self.together.add(tuple(sorted((visible[i].pid, visible[j].pid))))

    def _check_mismatch(self, tr, g, ctx, out):
        """A crop whose jersey (or learned appearance) clearly contradicts the identity is not this
        player. Repeated contradictions mean the local track moved to someone else: release."""
        cur = tr.current
        if cur is None or tr.occluded or cur.quality < .3 or not g.gallery:
            return
        c = compare(g.gallery, [cur])
        if c is None:
            return
        kit_bad = cur.has_jersey and c['jersey'] > KIT_MISMATCH
        look_bad = c['appearance'] is not None and c.get('cosine', 1) < .5 and len(g.gallery) >= 3
        tr.kit_miss = tr.kit_miss+1 if kit_bad else 0
        tr.look_miss = tr.look_miss+1 if look_bad else 0
        if tr.kit_miss >= MISSES or tr.look_miss >= MISSES:
            what = f"jersey distance {c['jersey']:.2f}" if tr.kit_miss >= MISSES else f"OSNet cosine {c.get('cosine', 0):.2f}"
            self._unbind(g)
            self._release(tr, f'no longer matches {g.pid}')
            tr.kit_miss = tr.look_miss = 0
            tr.released = (g.pid, ctx['time'])
            out['events'].append(self._event('reid-rejected', tr.id, g.pid, f"Track {tr.id} no longer matches {long_id(g.pid)} ({what}); identity released."))
            out['issues'].append({'id': g.pid, 'time': ctx['time'], 'reason': 'Tracked person stopped matching this identity; released.'})
        elif kit_bad or look_bad:
            tr.reason = 'appearance mismatch: watching'

    def _unbound_label(self, tr):
        """(display, combined label) for a person without a global identity."""
        if tr.state == 'uncertain':
            display = 'REF-?' if tr.role == 'referee' else (f'GK-{tr.team}?' if tr.team else 'GK-?') if tr.role == 'goalkeeper' else f'{tr.team}-?' if tr.team else '?'
            return display, ('GOALKEEPER_CANDIDATE' if tr.role == 'goalkeeper' else 'IDENTITY_UNCERTAIN')
        if tr.state == 'rejected':
            return 'OUT', 'REJECTED_OUTSIDE_FIELD'
        if tr.state == 'unknown':
            return f'UNK-{tr.id}', 'UNKNOWN'
        return 'CAND', 'CANDIDATE'

    # ---------- sanity ----------
    def sanity(self, time):
        out = []
        for team in ('A', 'B'):
            n = self._count(team)
            if n > self.o.max_per_team:
                out.append(self._event('sanity', None, None, f'Team {team} has {n} player identities (expected at most {self.o.max_per_team}). Players are probably being recreated instead of re-identified, or substitutions happened.'))
        refs = self._count('referee')
        if refs > self.o.max_referees:
            out.append(self._event('sanity', None, None, f'{refs} referee identities (expected at most {self.o.max_referees}).'))
        for team in ('A', 'B'):
            keepers = [g.pid for g in self.registry.values() if g.role == 'goalkeeper' and g.team == team and g.status not in ('retired', 'substituted', 'unknown')]
            if len(keepers) > 1:
                out.append(self._event('sanity', None, keepers[0], f'Team {team} has {len(keepers)} goalkeeper identities ({", ".join(keepers)}): possible duplicate.'))
        alike = []
        for team in ('A', 'B'):
            ids = [g for g in self.registry.values() if g.here and g.team == team and g.role == 'player' and g.status not in ('substituted', 'unknown', 'retired') and len(g.gallery) >= 2]
            for i in range(len(ids)):
                for j in range(i+1, len(ids)):
                    a, b = ids[i], ids[j]
                    if tuple(sorted((a.pid, b.pid))) in self.together:
                        continue
                    c = compare(a.gallery, [s.d for s in b.gallery])
                    if c is None:
                        continue
                    same = c['cosine'] >= .93 if c.get('cosine') is not None else c['total'] < .08
                    if same:
                        alike.append((c.get('cosine', 1-c['total']), a.pid, b.pid))
        for score, a, b in sorted(alike, reverse=True)[:3]:
            out.append(self._event('sanity', None, a, f'{long_id(a)} and {long_id(b)} were never visible together and look alike ({score:.2f}): possible duplicate identity.'))
        return out

    # ---------- helpers ----------
    def _describe(self, role, team):
        if role == 'goalkeeper' and not team:
            return 'GOALKEEPER (team unknown)'
        return role_label(role, team)

    def _event(self, kind, track, pid, message, scores=None, accepted=False):
        e = {'time': r3(self.time), 'kind': kind, 'track': track, 'playerId': pid, 'message': message}
        if scores:
            e['scores'] = scores
        if accepted:
            e['accepted'] = True
        return e

    def _defer_allowed(self, tr, time):
        wait = min(30.0, 2*2**min(4, max(0, tr.deferrals-1)))
        if time-tr.last_deferred < wait and time >= tr.last_deferred:
            return False
        tr.last_deferred = time
        tr.deferrals += 1
        return True

    def _allow(self, key, time, every):
        last = self.limits.get(key)
        if last is not None and time-last < every and time >= last:
            return False
        self.limits[key] = time
        if len(self.limits) > 400:
            self.limits.pop(next(iter(self.limits)))
        return True

    # ---------- team model input ----------
    def team_samples(self, current, time):
        """Jersey samples for the kit model: live tracks the detector calls players (plus tracks seen in
        the last few seconds), inside or on the edge of the pitch, with a clean enough crop.
        current: {track id: (descriptor, detector class, zone)} for this frame."""
        out = []
        for tid, (d, cls, zone) in current.items():
            tr = self.tracks.get(tid)
            if d is None or d.quality < .25 or not d.has_jersey or zone == 'outside' or (tr is not None and (tr.retired or tr.state == 'rejected')):
                self.kits.pop(tid, None)
                continue
            player = tr.evidence.detector_shares()['player'] >= .6 if tr is not None and tr.evidence.detector else cls == 'player'
            if not player:
                self.kits.pop(tid, None)
                continue
            sample = {'jersey': d.jersey, 'shorts': d.shorts, 'weight': d.quality, 'order': tr.born if tr is not None else time}
            out.append(sample)
            if tr is not None and tr.hits >= self.o.min_hits:
                self.kits[tid] = (sample, time)
        for tid, (sample, when) in list(self.kits.items()):
            if tid in current:
                continue
            if time-when > KIT_WINDOW or when > time+1e-6:
                del self.kits[tid]
            else:
                out.append(sample)
        return out

    def referee_samples(self, model=None):
        """(jersey, shorts) of people the detector consistently calls referees and of referee identities.
        A track the detector calls referee most of the time whose kit only loosely matches a team (near
        the edge of that team's colour spread) counts too: that is a referee in a kit close to a team's."""
        out = []
        for tr in self.tracks.values():
            if len(tr.evidence.detector) < 5:
                continue
            share = tr.evidence.detector_shares()['referee']
            kits = tr.evidence.kits
            if share < .8 and (share < .6 or model is None or not self._loose_kit(kits, model)):
                continue
            clean = [d for d, _, occluded in tr.descs if not occluded and d.has_jersey]
            out += [(d.jersey, d.shorts) for d in clean[-3:]]
        for g in self.registry.values():
            if g.role == 'referee' and g.status != 'retired':
                out += [(x.d.jersey, x.d.shorts) for x in g.gallery[-2:] if x.d.has_jersey]
        return out

    @staticmethod
    def _loose_kit(kits, model):
        """The kit votes of a track match a team only loosely, or match neither team."""
        if not kits:
            return False
        votes = [v for v in kits if v.team is not None]
        if len(votes) < .5*len(kits):
            return True
        team = Counter(v.team for v in votes).most_common(1)[0][0]
        limit = min(.5, 2.5*model.team_spread(team)+.1)
        mine = [v.dist_a if team == 'A' else v.dist_b for v in votes if v.team == team]
        return sum(mine)/len(mine) >= .8*limit

    def keeper_samples(self):
        """[(team, jerseys)] of goalkeeper identities, for goalkeeper kit prototypes."""
        return [(g.team, [x.d.jersey for x in g.gallery if x.d.has_jersey]) for g in self.registry.values()
                if g.role == 'goalkeeper' and g.status not in ('retired', 'substituted', 'unknown') and g.gallery]

    def summary(self):
        groups = {'players': [], 'goalkeepers': [], 'referees': [], 'retired': []}
        for g in sorted(self.registry.values(), key=lambda g: g.pid):
            status = 'missing' if g.status == 'active' and self._lost_for(g) > LOST else g.status
            entry = {'id': g.pid, 'longId': long_id(g.pid), 'display': g.display, 'team': g.team, 'role': ROLE_OUT.get(g.role, 'UNKNOWN'),
                     'label': g.role_label, 'status': status.upper().replace('OFFSCREEN', 'OFF_SCREEN'),
                     'firstSeen': r3(g.first_seen), 'lastSeen': r3(g.last_seen), 'observations': g.observations,
                     'identityConfidence': r3(g.identity_confidence), 'teamConfidence': r3(g.team_confidence),
                     'roleConfidence': r3(g.role_confidence), 'roleLocked': g.role_locked,
                     'refereeConfidence': r3(g.referee_confidence), 'goalkeeperConfidence': r3(g.goalkeeper_confidence),
                     'roleHistory': [list(h) for h in g.role_history[-6:]], 'gallerySize': len(g.gallery),
                     'lastBox': [round(float(v), 4) for v in g.last_box],
                     'lastPitch': None if g.last_pitch is None else [r3(g.last_pitch[0]), r3(g.last_pitch[1])],
                     'velocity': [round(float(g.velocity[0]), 4), round(float(g.velocity[1]), 4)], 'exitEdge': g.exit_edge,
                     'restored': g.restored, 'history': list(g.history)[-10:],
                     'official': OFFICIAL_OUT.get(g.official, '') if g.role == 'referee' else None, 'roleEvidence': g.role_scores,
                     'revoked': g.revoked or None, 'keeperGoal': r3(g.keeper_goal)}
            if g.status == 'retired':
                entry['retiredInto'] = g.retired_into
                groups['retired'].append(entry)
            else:
                groups['referees' if g.role == 'referee' else 'goalkeepers' if g.role == 'goalkeeper' else 'players'].append(entry)
        return groups

    def goal_sides(self):
        """Which team defends which side of the image, as accumulated votes."""
        return {side: dict(votes) for side, votes in self.side_votes.items()}
