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

Identity, team and role are separate: an identity (A-01) has a team (A) and a role (PLAYER or
GOALKEEPER). Only match entities get identities: team players and goalkeepers. Referees, assistant
referees, coaches and staff are NON-TARGET people: they keep their local track (an overlap with them
still freezes a player's memory, and the ball still sees their boots) but never get a global
identity, a team, an appearance gallery or a 2D position. The role classifier is a filter: a
referee-like reading makes an unidentified person non-target, after the missing players they may be
were compared (identity first). A confirmed player never becomes non-target from a role reading: the
reading is ignored, and a local track that moved to someone else is split (TRACK SWITCH). The one
exception is an identity that never became an established player (its clean kit votes never clearly
wore its team's kit) and has shown the same referee-like person since it was created: a non-target
person promoted by mistake, withdrawn. A confirmed goalkeeper role is locked and survives leaving the
goal area or a few odd frames.

Close encounters. A crop whose box overlaps another person's is contaminated: it updates no
appearance gallery, no kit vote and no detector vote (the position cues still count), and a crop
must look like the identity it would be stored under. Identity comes before role: a track that
may be a missing player is compared with that player before any role reading may make it
non-target; a track born as a duplicate box on an identity's own track is the same person, not
someone "seen together" with them; a confirmed player becomes a goalkeeper only after several clean
checks in a row, and a referee-like reading of a confirmed player changes nothing. A local track
whose clean crops stop looking like its own first crops (and its identity's gallery) several ticks
in a row was handed to another person by the local tracker: it is re-born there (evidence
restarted, compared with the missing players from scratch) and the identity it carried goes
missing. tracklets.py cuts the tracks at those points and where the crops change, and connects the
fragments offline.
"""
import math
from collections import Counter, deque
from dataclasses import dataclass, field

import numpy as np

from .appearance import Sample, add_to_gallery, batch_similarity, compare, decode_descriptor, embedding_cosine, encode_descriptor
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
ROLE_LOCK = .7           # role confidence that locks a goalkeeper role, and that a referee-like reading needs to make
                         # an unidentified player track non-target
GK_DOUBT_CHECKS = 3      # consecutive role checks (~3 s) of contradictory evidence that revoke a confirmed goalkeeper
# Identities are PLAYER or GOALKEEPER. The classifier's referee reading is a raw role only: NON_TARGET.
ROLE_OUT = {'player': 'PLAYER', 'goalkeeper': 'GOALKEEPER', 'referee': 'NON_TARGET'}
NON_TARGET_HINTS = {'assistant': 'SIDELINE_OFFICIAL', 'centre': 'REFEREE_LIKE', '': 'REFEREE_LIKE'}
LOOKS_PROBES = 3         # recent clean crops compared with the officials' look and the team appearance galleries
# People you identified in the review page ("manual" identities, see corrections.py) are recognised by
# appearance before anything else; their role and team are never changed by automatic evidence.
MANUAL_MATCH = .7        # appearance similarity at which a track is recognised as someone you identified (the same
                         # person across a match measured .6-.85 on broadcast footage, other people <= .56)
MANUAL_LEAD = .15        # ... and how much more like them it must look than like any automatic identity
MANUAL_EVERY = 1.0       # seconds between those appearance checks, per track
MANUAL_PROBES = 3        # only a track's latest crops are compared: after two people cross, older crops show the other one
MANUAL_LOST = .45        # a track carrying someone you identified that looks less like them than this, three checks
                         # in a row, has moved to someone else and lets the identity go
OVERLAP_FREEZE = .15     # IoU with another person's box above which a crop is contaminated: it updates no memory
ROLE_FLIP_CHECKS = 4     # consecutive clean role checks (~4 s) before a confirmed player identity may become a goalkeeper,
                         # or before one that never was an established player may be withdrawn as a non-target person
ESTABLISHED_VOTES = 25   # clean kit votes (5 s of evidence ticks) clearly in the identity's team kit, making up ...
ESTABLISHED_SHARE = .6   # ... this share of its clean votes: an established player, never withdrawn by any reading
NON_TARGET_EXIT = 10     # evidence ticks (2 s) of clear player or goalkeeper evidence before an ignored person is reconsidered
OFFICIAL_LOOK_SECONDS = 60.0   # ignored referee-like people seen this recently teach the classifier the officials' look
GALLERY_MIN_COSINE = .45 # a crop stored under an identity must look like that identity's gallery
DUPLICATE_BIRTH = .2     # IoU at birth with an identity's own track that makes a new track the same person
SWITCH_COSINE = .55      # a clean crop this unlike the track's own first crops (and its identity's gallery) shows someone else
                         # (OSNet on small crops: another kit scores .36-.5, the same person .75-.95, one odd crop may dip to .5)
SWITCH_CHECKS = 3        # ... that many ticks in a row, the crops alike each other (SWITCH_ALIKE): the local track moved to another
SWITCH_ALIKE = .7        # person (an id switch inside one track); unlike crops are noise (a shadow, a half-hidden body)
COVERED_FREEZE = .5      # share of a box inside another person's box above which its crop is contaminated as well: a small player behind a
                         # near one keeps a small IoU with them while most of the crop shows the other person
PARTIAL_HEIGHT = .8      # a box next to someone (occluded) shorter than this share of the track's recent clean height is a partial box: the
                         # body is cut off, so its foot point, position and speed are wrong; the kinematic memory keeps the last clean box
MAX_SPEED = 9.0          # m/s: a remembered velocity above a sprint is a measurement artifact and is never extrapolated faster than this
SHORT_LOSS = 3.0         # s: an identity lost this recently that a new track strongly resembles (STRONG_LOOK) is never ruled out on the
STRONG_LOOK = .65        # predicted position alone (that position came from where the occlusion lost them): the track waits for clean
SHORT_HOLD = 4.0         # frames, IDENTITY_UNCERTAIN, for up to this many seconds of its own observation before the usual rules decide


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
        return f'Referee_{int(pid[4:]):02d}'   # results of earlier versions only
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
    kit_clear: int = 0                                   # players: clean kit votes clearly (not loosely) in their team's kit ...
    kit_total: int = 0                                   # ... of all clean, valid kit votes while identified (established?)
    withdrawn: str = ''                                  # why the identity was withdrawn (a non-target person promoted by mistake)
    role_scores: dict = field(default_factory=dict)      # the latest role decision's evidence scores
    doubt: int = 0                                       # goalkeepers: consecutive checks with contradictory evidence
    revoked: str = ''                                    # goalkeepers: why the goalkeeper status was taken away
    keeper_goal: float = 0.0                             # goalkeepers: the strongest goal-area residence ever seen (what confirmed them)
    manual: bool = False                                 # identified by you: role and team are yours, never changed automatically
    anchor: list = field(default_factory=list)           # identified by you: their appearance as you corrected it (never updated in a run)
    excluded: bool = False                               # identified by you as not part of the match (recognised, then left out)
    last_track: int | None = None                        # the local track this identity was last carried by
    tracks: list = field(default_factory=list)           # every local track that carried this identity (fragmentation)
    clean_seen: float = -math.inf                        # when the kinematic memory (stab, velocity, last_pitch, pitch_velocity) was last
                                                         # taken from a clean box (a partial box at an occlusion holds the memory)

    @property
    def role_label(self):
        return role_label(self.role, self.team)

    @property
    def display(self):
        """Shown label: A-04, GK-A / GK-B (goalkeepers by team)."""
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
    manual_checked: float = -math.inf               # last appearance check against the people you identified
    manual_hits: int = 0                            # consecutive checks that found the same one of them
    manual_pid: str | None = None
    manual_score: tuple = (0.0, 0.0)                # (similarity to them, best similarity to an automatic identity)
    manual_doubt: int = 0                           # consecutive checks on which a track no longer looked like the person you identified
    excluded: str | None = None                     # the identity you marked as not part of the match that this track is
    overlap: float = 0.0                            # IoU of this frame's box with the nearest other person's box
    overlap_with: int | None = None                 # that person's local track
    covered: float = 0.0                            # share of this frame's box inside another person's box
    partial: bool = False                           # this frame's box is contaminated or cut off by someone: no kinematic memory from it
    heights: deque = field(default_factory=lambda: deque(maxlen=HIST))   # camera-compensated box heights of recent clean evidence ticks
    last_pitch_at: float = -math.inf                # the evidence tick last_pitch was taken on
    born_over: dict = field(default_factory=dict)   # {other track: max IoU} during the first 0.6 s (a duplicate box of them?)
    contaminated_at: float = -math.inf              # last frame whose crop overlapped another person
    memory: str = ''                                # what happened to the identity's appearance memory on this tick
    raw_role: tuple = ('unknown', 0.0)              # the latest frame-window role reading (role, confidence), never the stable role
    clean_history: list = field(default_factory=list)   # (time, descriptor) of clean crops over the whole track, thinned, for the offline pass
    switch_miss: int = 0                            # consecutive clean crops unlike the track's own first crops
    switch_since: float = -math.inf                 # when those started
    last_match: float = -math.inf                   # the last clean crop that did look like the track's own first crops
    switch_crops: deque = field(default_factory=lambda: deque(maxlen=SWITCH_CHECKS))   # (time, descriptor) of those unlike crops
    switches: list = field(default_factory=list)    # times at which this local track was found to show another person (re-born)
    hint: str = ''                                  # non-target people: REFEREE_LIKE | SIDELINE_OFFICIAL (why they are ignored)
    nontarget_since: float = -math.inf
    exit_streak: int = 0                            # non-target people: consecutive ticks of clear player / goalkeeper evidence
    official_look: bool = False                     # confidently referee-like with independent support: the clean crops teach the
                                                    # classifier what the officials look like (anonymous: no identity is kept)

    @property
    def contaminated(self):
        """This frame's crop is a mixture of two people (a box overlapping another, or mostly inside one): it updates no memory."""
        return self.overlap >= OVERLAP_FREEZE or self.covered >= COVERED_FREEZE


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
        self.ended = {}          # every finished local track, kept for the offline tracklet refinement
        self.encounters = {}     # (track a, track b) -> {'start', 'ids', 'max'}: close player / official overlaps
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
                        'keeperGoal': r3(g.keeper_goal), 'manual': g.manual, 'excluded': g.excluded,
                        **({'anchor': [{'t': r3(s.time), 'd': encode_descriptor(s.d)} for s in g.anchor]} if g.anchor else {}),
                        'retiredInto': g.retired_into, 'kitClear': g.kit_clear, 'kitTotal': g.kit_total, 'withdrawn': g.withdrawn})
        return {'version': 1, 'identities': out, 'counters': dict(self.counters)}

    def restore(self, saved):
        for v in saved.get('identities', []) if isinstance(saved, dict) else []:
            try:
                pid, role = str(v['id']), v['role']
                if pid in self.registry or role not in ('player', 'goalkeeper', 'referee'):
                    continue
                if role == 'referee':
                    # Referees are not match entities. One you corrected is still recognised by appearance and
                    # left out (like someone you marked as not part of the match); an automatic referee identity
                    # saved by an earlier version is not loaded at all.
                    if not v.get('manual'):
                        continue
                    role, v = 'player', dict(v, team=None, excluded=True)
                gallery = []
                for s in v.get('gallery', [])[-self.o.gallery_size:]:
                    d = decode_descriptor(s.get('d', {}))
                    if d is not None:
                        gallery.append(Sample(d, float(s.get('t', 0))))
                box = [float(x) for x in v.get('lastBox', [.5, .5, .01, .02])][:4]
                edge = str(v.get('exitEdge', ''))[:8]
                retired = v.get('status') == 'retired'
                g = GlobalPlayer(pid, role, v.get('team') if v.get('team') in ('A', 'B') else None,
                                 'retired' if retired else 'unknown' if v.get('revoked') or v.get('excluded') else 'offscreen' if edge else 'missing', gallery, float(v.get('firstSeen', 0)), float(v.get('lastSeen', 0)), box,
                                 tuple(v['lastPitch']) if v.get('lastPitch') else None, None, edge,
                                 identity_confidence=clamp(float(v.get('identityConfidence', 0))),
                                 team_confidence=clamp(float(v.get('teamConfidence', 0))), role_confidence=clamp(float(v.get('roleConfidence', 0))),
                                 restored=True, gk_votes=Counter(v.get('gkVotes', {})), role_locked=bool(v.get('roleLocked')),
                                 referee_confidence=clamp(float(v.get('refereeConfidence', 0))), goalkeeper_confidence=clamp(float(v.get('goalkeeperConfidence', 0))),
                                 revoked=str(v.get('revoked', '') or ''), manual=bool(v.get('manual')), excluded=bool(v.get('excluded')),
                                 keeper_goal=clamp(float(v.get('keeperGoal', 1.0 if v.get('roleLocked') and role == 'goalkeeper' else 0.0))),
                                 retired_into=v.get('retiredInto'), withdrawn=str(v.get('withdrawn', '') or ''))
                g.clean_seen = g.last_seen
                if 'kitTotal' in v:
                    g.kit_clear, g.kit_total = int(v.get('kitClear', 0)), int(v.get('kitTotal', 0))
                elif role == 'player':
                    g.kit_clear = g.kit_total = ESTABLISHED_VOTES   # saved before kit histories existed: an earlier run's player is established
                if g.manual:
                    anchor = [Sample(d, float(s.get('t', 0))) for s in v.get('anchor') or [] for d in [decode_descriptor(s.get('d', {}))] if d is not None]
                    g.anchor = anchor or list(gallery)
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
            self._announce_manual(out)
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
        for tr in live:
            # A track born on top of another one is a second box on the same person (a split detection),
            # not someone seen together with them: remembered for the re-identification.
            if t-tr.born <= .6:
                for other in live:
                    if other is not tr:
                        o = iou(tr.box, other.box)
                        if o > tr.born_over.get(other.id, 0.0):
                            tr.born_over[other.id] = o
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
        self.ended[tid] = tr
        self._drop_pairs(tid, out)
        g = self.registry.get(tr.player_id) if tr.player_id else None
        if g is not None and g.track == tid:
            edge = edge_of(tr.last_box)
            g.track, g.last_track = None, tid
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
        tr.overlap, tr.overlap_with, tr.covered = float(s.get('overlap', 0.0)), s.get('overlapWith'), float(s.get('covered', 0.0))
        contaminated = tr.contaminated
        if contaminated:
            tr.contaminated_at = t
        if tr.segment != ctx['segment']:
            tr.segment, tr.velocity, tr.pitch_velocity = ctx['segment'], (0.0, 0.0), None
            tr.hist.clear()
            tr.heights.clear()
        # A box cut off by the person in front (the detector saw a sliver of this player) has its foot point too high:
        # a partial box. It says roughly where the person is, nothing about their speed; the kinematic memory keeps
        # the last clean box.
        typical = sorted(tr.heights)[len(tr.heights)//2] if len(tr.heights) >= 3 else None
        tr.partial = contaminated or (tr.occluded and typical is not None and tr.stab.h < PARTIAL_HEIGHT*typical)
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
                    tr.descs.append((d, t, s['occluded'] or contaminated))
                    if not s['occluded'] and not contaminated and len(tr.first) < 2:
                        tr.first.append(d)
                    if not s['occluded'] and not contaminated and d.quality >= .3 and len(tr.clean_history) < 600 and \
                            (len(tr.clean_history) < 12 or t-tr.clean_history[-1][0] >= 1.0):
                        tr.clean_history.append((t, d))
            if not tr.partial:
                if tr.hist:
                    prev = tr.hist[-1]
                    dt = t-prev[0]
                    if 1e-3 < dt <= 1.5:
                        vx, vy = (tr.stab.x-prev[1])/dt, (tr.stab.y-prev[2])/dt
                        tr.velocity = (.6*tr.velocity[0]+.4*vx, .6*tr.velocity[1]+.4*vy) if len(tr.hist) > 1 else (vx, vy)
                tr.hist.append((t, tr.stab.x, tr.stab.y))
                tr.heights.append(tr.stab.h)
            vote = s.get('vote')
            h = max(tr.stab.h, .005)
            motion = (tr.velocity[0]*ctx['aspect']/h*PLAYER_HEIGHT, tr.velocity[1]/h*PLAYER_HEIGHT)  # metres per second
            # A contaminated crop (two people in one box) says nothing about the kit or the detector's class;
            # only where the person is still counts.
            tr.evidence.add(None if contaminated else s['cls'], s['score'], None if contaminated else vote,
                            dict(s.get('cues', {}), zone=tr.zone, v=motion, clean=not contaminated))
            if vote is not None and vote.valid and not contaminated:
                tr.team_votes.append(vote.team or ('x' if vote.outlier else '-'))
                g = self.registry.get(tr.player_id) if tr.state == 'confirmed' and tr.player_id else None
                if g is not None and g.role == 'player' and g.team in ('A', 'B') and g.track == tr.id:
                    # The identity's team history: how clearly (not loosely) its clean crops wear its team's kit.
                    g.kit_total += 1
                    g.kit_clear += self._clear_vote(vote, g.team, ctx['model'])
            # The pitch speed over the evidence tick (last_pitch is taken on ticks, so the interval is the tick's, not
            # one frame's), from clean boxes only.
            if not tr.partial:
                if tr.pitch is not None and tr.last_pitch is not None and 1e-3 < t-tr.last_pitch_at <= 1.5:
                    dt = t-tr.last_pitch_at
                    vx, vy = (tr.pitch[0]-tr.last_pitch[0])/dt, (tr.pitch[1]-tr.last_pitch[1])/dt
                    tr.pitch_velocity = (.6*tr.pitch_velocity[0]+.4*vx, .6*tr.pitch_velocity[1]+.4*vy) if tr.pitch_velocity else (vx, vy)
                elif tr.pitch is None:
                    tr.pitch_velocity = None
                tr.last_pitch, tr.last_pitch_at = tr.pitch, t
        tr.last_box, tr.last_time = s['box'], t

    # ---------- team / role ----------
    def _check_team(self, tr, ctx, out):
        """A confirmed player whose kit votes switched (>= 80% of the last 15) is released and re-identified."""
        if tr.state not in ('confirmed', 'uncertain') or tr.team is None or tr.role != 'player' or len(tr.team_votes) < VOTES:
            return
        bound = self.registry.get(tr.player_id) if tr.player_id else None
        if bound is not None and bound.manual:
            return   # you gave this person their team
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
        if tr.excluded:
            tr.state = 'rejected'   # someone you marked as not part of the match
            return
        mine = self._manual_check(tr, ctx['time'])
        if mine is not None:
            self._adopt_manual(tr, mine, ctx, out)
            return
        if tr.manual_hits:
            tr.state = 'candidate'
            tr.reason = f'checking: looks like {tr.manual_pid}, identified by you'
            return
        o, d = self.o, tr.decision
        if tr.state == 'nontarget' and not self._leave_nontarget(tr, d, ctx, out):
            return
        if d.label == 'CANDIDATE' and d.scores.get('neitherTeam', 0) >= .6 and tr.hits >= o.min_hits and len(tr.descs) >= 3:
            # A kit matching neither team is not a role by itself, but the person may be a known goalkeeper:
            # their own appearance decides.
            match = self._special_match(tr, ('goalkeeper',))
            if match is not None and match[1] >= .8:
                g = match[0]
                d = tr.decision = RoleDecision(g.role_label, g.role, g.team, .8 if g.team else 0.0, round(match[1], 3),
                                               f'looks like {g.display} (appearance {match[1]:.2f})', d.referee_confidence, d.goalkeeper_confidence, d.scores)
        recent = [z for z in tr.zones if z != 'unknown']
        zr = sum(1 if z == 'inside' else .5 if z == 'boundary' else 0 for z in recent)/len(recent) if recent else 0.0
        born = tr.first_zone == 'boundary'
        need = o.boundary_hits if born else o.min_hits
        if ctx.get('pitch_provisional'):
            need *= 2   # the playable area itself is uncertain: new people must persist longer before they become identities
        pitch_ok = ctx['pitch_reliable'] or not ctx['filter_enabled']
        # No usable pitch on this frame (the registration failed, not the tracking): the image evidence alone
        # decides, more slowly, and nobody is rejected for a zone that was never measured.
        blind = not pitch_ok
        if blind:
            need = max(need, o.boundary_hits)
        ok = d.label != 'CANDIDATE' and d.role != 'unknown' and tr.hits >= o.min_hits
        # Assistant referees run the touchline: a confident referee-like decision there is decided (as non-target).
        runner = d.role == 'referee' and d.role_confidence >= .6
        sideline = ''
        if ok and d.role == 'goalkeeper':
            # The detector may call a sideline official a goalkeeper; feet on or outside the touchline, or
            # in its corridor, keep them a candidate rather than a goalkeeper candidate.
            k = d.scores.get('keeper') or {}
            if k.get('vetoed'):
                ok, sideline = False, k['vetoed']
        if ok and blind:
            ok = tr.hits >= need and (not recent or zr >= .8 or runner)
        elif ok:
            ok = tr.hits >= need if runner else zr >= .8 and (not born or (tr.hits >= o.boundary_hits and tr.inside >= 3))
        if d is not None:
            tr.raw_role = (d.role, r3(d.referee_confidence if d.role == 'referee' else d.goalkeeper_confidence if d.role == 'goalkeeper' else d.role_confidence))
        if ok and d.role == 'referee':
            # The classifier is a filter, not an identity: a referee-like person is ignored, after the missing
            # players they may be were compared (identity first).
            hold = self._player_hold(tr, ctx)
            if hold is None:
                self._mark_nontarget(tr, d, ctx, out, 'referee-like from the start')
                return
            g, score = hold
            tr.state, tr.team, tr.role = 'uncertain', g.team, 'player'
            tr.team_confidence, tr.role_confidence = .5, .5
            tr.reason = f'identity first: may be {g.pid} ({score:.2f}); the referee-like reading waits'
            if self._allow('role-wait-'+str(tr.id), ctx['time'], 5):
                out['events'].append(self._event('role', tr.id, g.pid, '\n'.join(['ROLE UPDATE DEFERRED', f'Local Track: {tr.id}',
                                                                                 f'Raw role: NON_TARGET {d.referee_confidence:.2f} (referee-like)',
                                                                                 f"Identity first: missing {g.pid} scores {score:.2f} on this person's crops and place",
                                                                                 'Decision: compared with the missing player before anything else'])))
            return
        if ok:
            tr.state, tr.team, tr.role = 'uncertain', d.team, d.role
            tr.team_confidence, tr.role_confidence, tr.reason = d.team_confidence, d.role_confidence, ''
            out['events'].append(self._event('promotion', tr.id, None,
                                             f'Track {tr.id} promoted to {self._describe(d.role, d.team)} after {tr.hits} observations (inside {pct(zr)}, team {pct(d.team_confidence)}, role {pct(d.role_confidence)}).'))
            return
        if tr.hits >= 3*o.min_hits:
            tr.state = 'rejected' if recent and zr < .8 and not runner else 'unknown'
        else:
            tr.state = 'candidate'
        if blind and tr.state == 'candidate':
            tr.reason = f'pitch position unresolved on this frame (no calibration): image evidence decides after {need} observations ({tr.hits} so far; {d.reason})'
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
        """Tracks keep collecting role evidence (about once a second). An unidentified player track read as
        referee-like becomes non-target once no missing player may be them; a confirmed player is never
        changed by a referee-like reading (see _referee_reading) and becomes a goalkeeper only on sustained
        clean evidence; a locked goalkeeper is never turned back into a player by a few odd frames."""
        t = ctx['time']
        for tr in live:
            if tr.state == 'uncertain' and t-tr.role_checked >= 1.0 and len(tr.evidence.records) >= 2*self.o.min_hits and not tr.retired:
                tr.role_checked = t
                d = tr.decision = self._decide(tr, ctx)
                confidence = d.referee_confidence if d.role == 'referee' else d.goalkeeper_confidence if d.role == 'goalkeeper' else 0
                tr.raw_role = (d.role, r3(confidence if d.role in ('referee', 'goalkeeper') else d.role_confidence))
                flip = d.role in ('referee', 'goalkeeper') and d.role != tr.role and confidence >= ROLE_LOCK
                if flip and tr.role == 'player':
                    why = self._identity_first(tr, d, t)
                    if why:
                        flip = False
                        if self._allow('role-wait-'+str(tr.id), t, 5):
                            out['events'].append(self._event('role', tr.id, None, '\n'.join(['ROLE UPDATE DEFERRED', f'Local Track: {tr.id}',
                                                                                              f'Raw role: {ROLE_OUT[d.role]} {confidence:.2f}', f'Identity first: {why}', 'Decision: role kept, re-identification continues'])))
                if flip and d.role == 'referee':
                    self._mark_nontarget(tr, d, ctx, out, f'was {self._describe(tr.role, tr.team)}; no missing player may be them')
                elif flip:
                    old = self._describe(tr.role, tr.team)
                    tr.role, tr.team, tr.role_confidence = d.role, d.team, confidence
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
            if not g.manual:
                mine = self._manual_check(tr, t)
                if mine is not None:
                    self._adopt_manual(tr, mine, ctx, out)   # the person is someone you identified
                    continue
            d = tr.decision = self._decide(tr, ctx)
            g.referee_confidence, g.goalkeeper_confidence = d.referee_confidence, d.goalkeeper_confidence
            g.role_scores = d.scores
            if g.manual:
                # Your call: the role and team you gave this person are not changed by automatic evidence.
                if g.role == 'goalkeeper':
                    g.keeper_goal = max(g.keeper_goal, d.scores['nearGoal'])
                self._verify_manual(tr, g, ctx, out)
                continue
            if d.role == g.role:
                current = d.goalkeeper_confidence if g.role == 'goalkeeper' else d.role_confidence
                g.role_confidence = r3(max(current, g.role_confidence) if g.role_locked else current)
            if g.role == 'goalkeeper':
                confidence = d.goalkeeper_confidence
                g.keeper_goal = max(g.keeper_goal, d.scores['nearGoal'])
                # Hysteresis: a confirmed goalkeeper survives odd frames, a walk upfield (the detector may well
                # call them a player there) and a ball fetched behind the goal line; only sustained sideline,
                # off-pitch-away-from-the-goal or referee-like-away-from-any-goal evidence over a full window,
                # after the track has been theirs for a while, revokes the status and frees the team's slot.
                settled = len(tr.evidence.records) >= WINDOW and t-g.bound_at >= 5.0
                contradiction = self._keeper_contradiction(d) if settled else ''
                g.doubt = g.doubt+1 if contradiction else 0
                if g.doubt >= GK_DOUBT_CHECKS:
                    self._revoke_goalkeeper(tr, g, d, contradiction, ctx, out)
                    continue
                if not g.role_locked and confidence >= ROLE_LOCK and d.scores.get('keeper', {}).get('confirmed'):
                    g.role_locked, g.role_confidence = True, max(g.role_confidence, confidence)
                    g.role_history.append((r3(t), g.role, f'locked at {confidence:.2f}'))
                elif g.role_locked and d.role == 'player' and d.role_confidence >= .9 and self._allow('role-doubt-'+g.pid, t, 30):
                    out['issues'].append({'id': g.pid, 'time': t, 'reason': f'{g.display} now looks like a team {d.team} player; role kept (locked). Check.'})
                continue
            tr.raw_role = (d.role, r3(d.referee_confidence if d.role == 'referee' else d.goalkeeper_confidence if d.role == 'goalkeeper' else d.role_confidence))
            if g.role_locked:
                continue
            # A confirmed player is never turned into a non-target person by a role reading (_referee_reading),
            # and becomes a goalkeeper only after several clean checks in a row ask for it: the evidence
            # accumulates and crops contaminated by an overlap never count.
            # A referee-like decision counts at the classifier's own threshold; a goalkeeper reading at ROLE_LOCK.
            want = 'referee' if d.role == 'referee' else \
                'goalkeeper' if d.role == 'goalkeeper' and d.goalkeeper_confidence >= ROLE_LOCK else None
            tr.role_streak = tr.role_streak+1 if want is not None and want == tr.role_wanted else (1 if want else 0)
            tr.role_wanted = want
            if want is None:
                continue
            confidence = d.referee_confidence if want == 'referee' else d.goalkeeper_confidence
            clean = self._clean_share(tr)
            contact = t-tr.contaminated_at < 3.0 or t-g.crossed_at < 5.0
            if want == 'referee':
                self._referee_reading(tr, g, d, confidence, clean, contact, ctx, out)
                continue
            if tr.role_streak < ROLE_FLIP_CHECKS or clean < .6 or contact:
                if self._allow('role-ignored-'+g.pid, t, 10):
                    reason = (f'overlapped another person {t-tr.contaminated_at:.1f} s ago; contaminated crops change no role' if contact else
                              f'only {clean:.0%} of the recent evidence is clean' if clean < .6 else
                              f'{tr.role_streak} check(s) in a row, {ROLE_FLIP_CHECKS} needed')
                    out['events'].append(self._event('role', tr.id, g.pid, '\n'.join(['ROLE UPDATE IGNORED', f'Global ID: {g.pid}', f'Local Track: {tr.id}',
                                                                                        f'Stable role: {self._old_role(g)}', f'Raw role: {ROLE_OUT[want]} {confidence:.2f}',
                                                                                        f'Reason: {reason}', 'Decision: role kept'])))
                continue
            self._to_goalkeeper(tr, g, d, ctx, out)

    def _referee_reading(self, tr, g, d, confidence, clean, contact, ctx, out):
        """A confirmed player read as referee-like. A person does not turn into an official during play, so
        the identity, team and role never change: the reading is a misclassification, a contaminated crop
        or a local track that moved to someone else (which the switch check splits, see _check_switch).
        The one exception: an identity that never became an established player (its clean kit votes never
        clearly wore its team's kit) and has shown this same person since it was created, read as
        referee-like over several clean checks in a row with independent support (see _supported), was a
        non-target person promoted by mistake and is withdrawn."""
        t = ctx['time']
        s = d.scores
        established = self._established(g)
        sustained = tr.role_streak >= ROLE_FLIP_CHECKS and clean >= .6 and not contact
        supported = self._supported(d)
        same = not established and sustained and supported and self._always_this_person(g, tr)
        if same:
            self._withdraw(tr, g, d, ctx, out)
            return
        if not self._allow('role-ignored-'+g.pid, t, 10):
            return
        reason = (f'an established team player ({g.kit_clear} of {g.kit_total} clean kit votes clearly team {g.team}): a referee-like reading never changes who they are'
                  if established else
                  f'overlapped another person {t-tr.contaminated_at:.1f} s ago; contaminated crops change nothing' if contact else
                  f'only {clean:.0%} of the recent evidence is clean' if clean < .6 else
                  f'{tr.role_streak} check(s) in a row, {ROLE_FLIP_CHECKS} needed' if tr.role_streak < ROLE_FLIP_CHECKS else
                  "no independent support (detector, referee kit, touchline, referee movement or a kit matching neither team)" if not supported else
                  'the crops no longer show the person this identity started with: a switch of the local track, not of the person')
        out['events'].append(self._event('role', tr.id, g.pid, '\n'.join(['ROLE UPDATE IGNORED', f'Global ID: {g.pid}', f'Local Track: {tr.id}',
                                                                            f'Stable role: {self._old_role(g)}', f'Raw role: NON_TARGET {confidence:.2f} (referee-like)',
                                                                            f'Reason: {reason}', 'Decision: identity, team and role kept'])))

    @staticmethod
    def _established(g):
        """An established team player: identified by you, or enough clean kit votes, mostly clearly in their
        team's kit. An established player is never withdrawn, whatever a role reading says."""
        return g.manual or (g.kit_total >= ESTABLISHED_VOTES and g.kit_clear >= ESTABLISHED_SHARE*g.kit_total)

    @staticmethod
    def _clear_vote(vote, team, model):
        """A kit vote clearly (not loosely: well inside that team's colour spread) for this team."""
        if vote.team != team:
            return False
        limit = min(.5, 2.5*model.team_spread(team)+.1)
        return (vote.dist_a if team == 'A' else vote.dist_b) < .8*limit

    def _withdraw(self, tr, g, d, ctx, out):
        """g was never a player: a referee-like (non-target) person promoted by mistake. The identity is
        withdrawn (kept, marked, never a candidate again), every observation it labelled becomes non-target
        (out of the roster and the 2D view), and the track is ignored from now on."""
        t = ctx['time']
        hint = NON_TARGET_HINTS.get(d.official, 'REFEREE_LIKE')
        self._unbind(g)
        g.status = 'retired'
        g.withdrawn = f'{hint} person promoted by mistake (never an established team {g.team} player)'
        g.role_history.append((r3(t), 'withdrawn', g.withdrawn))
        relabelled = 0
        for person in g.emitted_all:
            if person.get('id') == g.pid:
                self._as_nontarget(person, hint, f'{g.pid} withdrawn: never a player')
                person['withdrawnId'] = g.pid
                relabelled += 1
        g.emitted_all, g.observations = [], 0
        streak = tr.role_streak
        self._mark_nontarget(tr, d, ctx, out, None)
        out['events'].append(self._event('nontarget', tr.id, g.pid, '\n'.join([
            'PLAYER IDENTITY WITHDRAWN', f'Global ID: {g.pid} (was team {g.team})', f'Local Track: {tr.id}',
            f"Team history: {g.kit_clear} of {g.kit_total} clean kit votes clearly in team {g.team}'s kit "
            f'(an established player has at least {ESTABLISHED_VOTES}, {ESTABLISHED_SHARE:.0%} of them clear)',
            'Same person since the identity was created: yes (its first crops look like the crops now)',
            f'Referee-like evidence: {streak} clean checks in a row ({d.why})',
            f'Observations relabelled NON_TARGET: {relabelled}', 'Decision: identity withdrawn; this person is ignored from now on'])))
        out['issues'].append({'id': g.pid, 'time': r3(t), 'reason': f'{g.pid} withdrawn: a referee-like person promoted by mistake. Check.'})

    @staticmethod
    def _as_nontarget(person, hint, reason=None):
        """Write an ignored (non-target) person onto an output person: no identity, team or role."""
        person.update(id=None, display='IGNORE', label='NON_TARGET', team=None, role='NON_TARGET', state='nontarget', identityConfidence=0.0, hint=hint)
        for key in ('stableRole', 'memory', 'manual'):
            person.pop(key, None)
        if reason:
            person['reason'] = reason

    @staticmethod
    def _official_evidence(d):
        """Referee-like with independent support (the detector, the referee kit or the touchline), not the
        officials' look alone: only such people teach the classifier what the officials look like."""
        s = d.scores
        return d.role == 'referee' and d.referee_confidence >= ROLE_LOCK and \
            (s.get('detectorReferee', 0.0) >= .5 or s.get('refereeKit', 0.0) >= .3 or (s.get('touchlineBehaviour') or 0.0) >= .5)

    def _mark_nontarget(self, tr, d, ctx, out, why):
        """This person is not a target (referee-like, an official, staff): no global identity, team, gallery
        or 2D position, only a local track that still tells everyone else "someone is here" (an overlap with
        it freezes their memory). Re-evaluated every tick: two seconds of clear player or goalkeeper
        evidence make them a candidate again. why=None: no event (the caller logs)."""
        t = ctx['time']
        was = self._unbound_label(tr)[0] if tr.state in ('candidate', 'uncertain') else tr.state.upper()
        hint = NON_TARGET_HINTS.get(d.official, 'REFEREE_LIKE') if d is not None else 'REFEREE_LIKE'
        tr.state, tr.role, tr.team, tr.player_id = 'nontarget', 'unknown', None, None
        tr.team_confidence, tr.role_confidence, tr.best = 0.0, (d.referee_confidence if d is not None else 0.0), 0.0
        tr.reid.clear()
        tr.blind, tr.exit_streak, tr.role_streak, tr.role_wanted = [], 0, 0, None
        tr.hint, tr.nontarget_since = hint, t
        tr.official_look = tr.official_look or (d is not None and self._official_evidence(d))
        tr.reason = f"ignored ({hint.lower().replace('_', ' ')})" + (f': {why}' if why else '')
        if why is None or not self._allow('nontarget-'+str(tr.id), t, 10):
            return
        lines = ['NON-TARGET PERSON', f'Local Track: {tr.id}', f'Was: {was}', f'Hint: {hint}', f'Why: {why}',
                 'Global identity: none (ignored; the local box still counts for overlaps and the ball)']
        if d is not None:
            lines += self._role_lines(d)+[f'Non-target confidence: {d.referee_confidence:.2f}']
        out['events'].append(self._event('nontarget', tr.id, None, '\n'.join(lines)))

    def _leave_nontarget(self, tr, d, ctx, out):
        """An ignored person stays ignored until two seconds of clear player or goalkeeper evidence (their
        own, over the role window): then they are a candidate again and go through promotion and
        re-identification like anyone else. Returns True when they leave."""
        clear = d is not None and ((d.role == 'player' and d.role_confidence >= .6 and d.team_confidence >= .6) or
                                   (d.role == 'goalkeeper' and d.goalkeeper_confidence >= .6))
        tr.exit_streak = tr.exit_streak+1 if clear else 0
        if d is not None and d.role == 'referee':
            tr.hint = NON_TARGET_HINTS.get(d.official, 'REFEREE_LIKE')
            tr.official_look = tr.official_look or self._official_evidence(d)
        if tr.exit_streak < NON_TARGET_EXIT:
            tr.reason = f"ignored ({tr.hint.lower().replace('_', ' ')})" + (f'; player evidence {tr.exit_streak}/{NON_TARGET_EXIT}' if clear else '')
            return False
        tr.state, tr.exit_streak, tr.hint = 'candidate', 0, ''
        out['events'].append(self._event('nontarget', tr.id, None, '\n'.join(['NON-TARGET PERSON RECONSIDERED', f'Local Track: {tr.id}', f'Evidence: {d.why}',
                                                                               'Decision: a candidate again (compared with the missing players first)'])))
        return True

    def _player_hold(self, tr, ctx):
        """Identity first: the missing player this person may be, before a referee-like reading makes them
        non-target. (identity, score) when one scores near the reconnection threshold on this person's own
        crops and place (and their crops look like that player's), else None."""
        if len(tr.descs) < 3:
            return None
        best = None
        for g in self.registry.values():
            if g.role != 'player' or g.team not in ('A', 'B') or g.excluded or g.status in ('substituted', 'unknown', 'retired'):
                continue
            if g.track is not None and (g.track == tr.id or self._lost_for(g) < LOST):
                continue
            if g.track is None and g.status not in ('missing', 'offscreen'):
                continue
            s = self._score(tr, g, ctx)
            if s['appearance'] is None or s['appearance'] < .5:
                continue
            if best is None or s['final'] > best[1]:
                best = (g, s['final'])
        return best if best is not None and best[1] >= self.o.reid_min-.1 else None

    def _role_lines(self, d):
        """The evidence behind a role decision, one line each (kit colour, whole appearance against the
        officials' look and each team, detector votes, touchline and central-referee behaviour, team count)."""
        s = d.scores
        lines = []
        if s.get('teamA') is not None:
            lines += [f"Team A similarity: {s['teamA']:.2f}", f"Team B similarity: {s['teamB']:.2f}"]
        if s.get('refereeAppearance') is not None or s.get('teamAAppearance') is not None or s.get('teamBAppearance') is not None:
            f = lambda key: 'n/a' if s.get(key) is None else f"{s[key]:.2f}"
            lines.append(f"Appearance vs the officials' look / team A / team B: {f('refereeAppearance')} / {f('teamAAppearance')} / {f('teamBAppearance')}")
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
        if d.role == 'referee' and d.referee_confidence >= ROLE_LOCK and s['nearGoal'] < .3:
            # Away from any goal only: a confirmed keeper at their goal is never made non-target by a reading.
            return f'referee-like evidence {d.referee_confidence:.2f} away from any goal'
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
        if not self._keeper_confirmed(d) or owner.manual:
            return ''   # a goalkeeper you identified keeps the slot
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
        re-decided (OUT, non-target or player, whatever the evidence says). The identity is kept, marked."""
        t = ctx['time']
        was, slot = g.display, g.team
        if tr is not None and tr.player_id == g.pid:
            self._unbind(g)
            self._release(tr, f'goalkeeper status revoked: {why}')
            if d is not None and d.role == 'referee':
                self._mark_nontarget(tr, d, ctx, out, f'was {was}, goalkeeper status revoked')
            else:
                tr.role = 'player' if d is not None and d.role == 'player' else 'unknown'
                tr.team = d.team if tr.role == 'player' else None
                tr.role_confidence = d.role_confidence if tr.role == 'player' else 0.0
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

    # ---------- people you identified (manual identities) ----------
    @staticmethod
    def _plain(g):
        if g.excluded:
            return 'not part of the match'
        if g.role == 'goalkeeper':
            return f'goalkeeper (team {g.team})' if g.team else 'goalkeeper'
        return f'team {g.team} player' if g.team else 'player'

    def _announce_manual(self, out):
        mine = sorted((g for g in self.registry.values() if g.manual and g.status != 'retired'), key=lambda g: g.pid)
        if mine:
            out['events'].append(self._event('manual', None, None, '\n'.join(
                ['REMEMBERED FROM YOUR CORRECTIONS'] + [f'{g.pid}: {self._plain(g)}' for g in mine] +
                ['Recognised by appearance; their roles and teams are not changed automatically.'])))

    def _manual_check(self, tr, t):
        """Is the person on this track someone you identified (including someone you marked as not part
        of the match)? Appearance decides, about once a second: clearly similar to them (MANUAL_MATCH)
        and clearly more similar than to any automatically identified person (MANUAL_LEAD), on two
        checks in a row. Returns that identity once decided; tr.manual_hits > 0 while a first match
        waits for its confirmation. Without any such identity this costs nothing and changes nothing."""
        if tr.retired or len(tr.descs) < 3 or 0 <= t-tr.manual_checked < MANUAL_EVERY:
            return None
        mine = [g for g in self.registry.values() if g.manual and g.gallery and g.status not in ('retired', 'substituted') and g.track != tr.id
                and (g.excluded or g.track is None or self._lost_for(g) >= LOST)]
        if not mine:
            tr.manual_hits, tr.manual_pid = 0, None
            return None
        tr.manual_checked = t
        probes = self._probes(tr)[-MANUAL_PROBES:]
        best = None
        for g in mine:
            c = compare(g.anchor or g.gallery, probes)
            if c is None:
                continue
            a = c['appearance'] if c['appearance'] is not None else clamp(1-c['total']/.6)
            if best is None or a > best[1]:
                best = (g, a)
        rival = 0.0
        if best is not None and best[1] >= MANUAL_MATCH:
            rival = self._best_appearance(tr, lambda o: not o.manual and o.status != 'unknown' and o.pid not in tr.former)
        if best is None or best[1] < MANUAL_MATCH or best[1] < rival+MANUAL_LEAD:
            tr.manual_hits, tr.manual_pid = 0, None
            return None
        tr.manual_hits = tr.manual_hits+1 if tr.manual_pid == best[0].pid else 1
        tr.manual_pid, tr.manual_score = best[0].pid, (best[1], rival)
        # Taking a track from an identity it already has needs one more confirmation than giving one to a
        # newcomer: a track that just crossed someone may still carry that person's crops for a moment.
        need = 3 if tr.player_id else 2
        return best[0] if tr.manual_hits >= need else None

    def _verify_manual(self, tr, g, ctx, out):
        """The track carrying someone you identified must keep looking like them (their appearance as you
        corrected it, not crops gathered in this run, which a wrong match would pollute). Three checks in a
        row clearly unlike them mean the track moved to someone else: the identity is let go."""
        c = compare(g.anchor, self._probes(tr)[-MANUAL_PROBES:]) if g.anchor and tr.descs else None
        if c is None:
            return
        a = c['appearance'] if c['appearance'] is not None else clamp(1-c['total']/.6)
        tr.manual_doubt = tr.manual_doubt+1 if a < MANUAL_LOST else 0
        if tr.manual_doubt < 3:
            return
        tr.manual_doubt = 0
        self._unbind(g)
        self._release(tr, f'no longer looks like {g.pid}, identified by you')
        tr.released = (g.pid, ctx['time'])
        tr.role, tr.team, tr.state = 'unknown', None, 'candidate'
        tr.reid.clear()
        out['events'].append(self._event('reid-rejected', tr.id, g.pid, f'Track {tr.id} no longer looks like {g.pid}, whom you identified (appearance {a:.2f}); identity released.'))
        out['issues'].append({'id': g.pid, 'time': ctx['time'], 'reason': 'The track carrying this identity moved to someone else; released.'})

    def _adopt_manual(self, tr, g, ctx, out):
        """Track tr is the person you identified as g: bind it (taking over from an automatic identity
        that was this person), or, for someone you marked as not part of the match, leave it out."""
        t = ctx['time']
        similarity, rival = tr.manual_score
        tr.manual_hits, tr.manual_pid = 0, None
        old = self.registry.get(tr.player_id) if tr.player_id else None
        if old is not None and old.track != tr.id:
            old = None
        tr.manual_doubt = 0
        lines = [f'Local Track: {tr.id}', f'Appearance similarity: {similarity:.2f} (best automatic identity {rival:.2f})']
        if g.excluded:
            if old is not None:
                self._unbind(old)
                tr.former.add(old.pid)
                lines.append(f'{old.pid} released from this track')
            tr.player_id, tr.state, tr.excluded, tr.role, tr.team = None, 'rejected', g.pid, 'unknown', None
            tr.reid.clear()
            tr.reason = f'not part of the match: looks like {g.pid}, whom you marked so'
            out['events'].append(self._event('manual', tr.id, g.pid, '\n'.join(
                ['RECOGNISED: NOT PART OF THE MATCH', f'Looks like {g.pid}, whom you marked as not part of the match'] + lines)))
            return
        if g.track is not None and g.track != tr.id:
            ghost = g.track
            self.tracks.pop(ghost, None)
            out['drop'].append(ghost)
            self._drop_pairs(ghost, out)
        if old is not None:
            retired = self._hand_over(tr, old, g, t, out)
            lines.append(f'Took over from {old.pid}' + (' (retired: it was this person)' if retired else ''))
        tr.role, tr.team, tr.excluded = g.role, g.team, None
        tr.team_confidence, tr.role_confidence = (.9 if g.team else 0.0), 1.0
        tr.team_votes.clear()
        self._bind(tr, g, t, max(.8, similarity))
        out['events'].append(self._event('reid', tr.id, g.pid, '\n'.join(
            ['RECOGNISED: IDENTIFIED BY YOU', f'Global ID: {g.pid} ({self._plain(g)}, set by you)'] + lines), accepted=True))
        self._flush(tr, g, t)

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
        if rival is not None:
            # The team's keeper returning under a player identity merges into their own goalkeeper identity
            # (the merge branch below); only a track that does not look like the absent rival competes for the slot.
            visible = rival.track is not None and self._lost_for(rival) < LOST
            match = self._special_match(tr, ('goalkeeper',)) if not visible else None
            resembles = match is not None and match[0] is rival and match[1] >= .6
            if not resembles and self._take_slot(tr, d, rival, g.pid, f'a stronger goalkeeper candidate for team {rival.team} ({g.pid}: {s["keeper"]["route"]})', ctx, out):
                rival = None
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
    @staticmethod
    def _clean_share(tr):
        """Share of the recent evidence ticks whose crop overlapped nobody."""
        records = list(tr.evidence.records)
        return sum(1 for r in records if r.get('clean', True))/len(records) if records else 1.0

    def _identity_first(self, tr, d, t):
        """Why an uncertain player track must not be made non-target yet ('' when it may): a missing
        player it resembles is still being compared, its evidence is contaminated, or the referee reading
        rests on one gallery resemblance that the batch appearance and the detector do not share."""
        # A missing player still in the running: a score near the reconnection threshold, or too few
        # comparisons so far. A candidate the comparisons have already ruled out does not hold the role.
        pending = max((s['final'] for steps in tr.reid.values() for s in steps), default=0.0)
        if pending >= self.o.reid_min-.1:
            best = max(tr.reid, key=lambda pid: max(s['final'] for s in tr.reid[pid]))
            return f'{best} is still a possible match ({pending:.2f}); identity is decided before the role'
        if tr.reid and max(len(steps) for steps in tr.reid.values()) < 3:
            return 're-identification has too few comparisons yet'
        if tr.blind:
            return f"{', '.join(pid for pid, _ in tr.blind)} cannot be compared yet"
        clean = self._clean_share(tr)
        if clean < .6:
            return f'only {clean:.0%} of the recent evidence is clean (overlaps)'
        if d.role == 'referee':
            looks = self._looks(tr, t)
            if (looks.get('referee') or 0.0) < .55 and d.scores.get('detectorReferee', 0.0) < .5 and d.scores.get('refereeKit', 0.0) < .5:
                return 'neither the batch appearance, the detector nor the kit says referee'
        return ''

    def _decide(self, tr, ctx):
        """The temporal role decision with this person's whole-appearance similarity to the officials' look
        and to each team's players, and whether the kit's team already has its full count of identities.
        The officials' look alone never makes anyone referee-like: a referee reading that nothing else
        supports (no detector votes, referee kit, touchline, referee movement or kit matching neither team)
        is decided again without it, on the person's own kit, detector and position evidence."""
        looks = self._looks(tr, ctx['time'])
        d = decide_role(tr.evidence, ctx['model'], self.o.min_hits, looks, self._crowded(tr))
        if d.role == 'referee' and looks.get('referee') is not None and not self._supported(d):
            d = decide_role(tr.evidence, ctx['model'], self.o.min_hits, dict(looks, referee=None), self._crowded(tr))
        return d

    @staticmethod
    def _supported(d):
        """A referee-like reading backed by something other than the officials' look: the detector, the
        referee kit, the touchline, referee movement (following play without playing it, outside the team's
        shape), or a kit matching neither team."""
        s = d.scores
        return s.get('detectorReferee', 0.0) >= .3 or s.get('refereeKit', 0.0) >= .3 or (s.get('touchlineBehaviour') or 0.0) >= .5 or \
            (s.get('refereeBehaviour') or 0.0) >= .5 or s.get('neitherTeam', 0.0) >= .5

    def _crowded(self, tr):
        """{team: True} where that team already has max_per_team identities *on the pitch now* other than
        this track's own. A missing player leaves room: the new track may well be them."""
        own = self.registry.get(tr.player_id) if tr.player_id else None
        out = {}
        for team in ('A', 'B'):
            n = sum(1 for g in self.registry.values() if g.role == 'player' and g.team == team and g is not own and
                    g.status not in ('retired', 'substituted', 'unknown') and g.track is not None and self._lost_for(g) < 5.0)
            out[team] = n >= self.o.max_per_team
        return out

    def _pools(self, t):
        """Reference appearance galleries, refreshed once a second: the confident players of each team, and
        the officials' look: the clean crops of people ignored as referee-like (with independent support)
        in the last minute. That look is anonymous, one cue of the classifier; nobody is identified by it."""
        if self.pools is not None and 0 <= t-self.pools_at < 1.0:
            return self.pools
        pools = {'referee': [], 'A': [], 'B': []}
        for g in self.registry.values():
            if g.status in ('retired', 'substituted', 'unknown') or not g.gallery:
                continue
            if g.role == 'player' and g.team in ('A', 'B') and g.observations >= 5 and g.team_confidence >= .5:
                pools[g.team].append((g.pid, [x.d for x in g.gallery[-3:]]))
        for tr in list(self.tracks.values())+list(self.ended.values()):
            if tr.state != 'nontarget' or not tr.official_look or not 0 <= t-tr.last_time <= OFFICIAL_LOOK_SECONDS:
                continue
            crops = self._official_crops(tr, 6)
            if crops:
                pools['referee'].append((('track', tr.id), crops))
        self.pools, self.pools_at = pools, t
        return pools

    @staticmethod
    def _official_crops(tr, n):
        """Up to n clean crops of an ignored official spread over their track (newest first), like a gallery."""
        history = [d for _, d in tr.clean_history]
        if not history:
            return [d for d, _, occluded in tr.descs if not occluded and d.quality >= .3][-n:]
        step = max(1, len(history)//n)
        return history[::-1][::step][:n]

    def _looks(self, tr, t):
        """Whole-appearance similarity of this person to the officials' look (the most similar ignored
        official) and to each team's players (the median player, so a few mislabelled identities cannot
        define a team's look): learned appearance plus the whole uniform, from the last few clean crops. A
        track's own identity, or its own crops, are never its own evidence. Cached for a second."""
        pools = self._pools(t)
        if tr.looks is not None and tr.looks[0] == self.pools_at and 0 <= t-tr.looks[1] < 1.0:
            return tr.looks[2]
        probes = [d for d, _, occluded in tr.descs if not occluded and d.quality >= .3][-LOOKS_PROBES:] or [d for d, _, _ in tr.descs][-LOOKS_PROBES:]
        out = {'referee': None, 'A': None, 'B': None, 'probes': len(probes)}
        for group, members in pools.items():
            members = [(pid, descs) for pid, descs in members if pid != tr.player_id and pid != ('track', tr.id)]
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
            if tr.role == 'goalkeeper':
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
            if tr.retired or tr.state != 'uncertain':
                continue
            mine = self._manual_check(tr, ctx['time'])
            if mine is not None:
                self._adopt_manual(tr, mine, ctx, out)
        uncertain = [tr for tr in uncertain if tr.state == 'uncertain']
        for tr in uncertain:
            if tr.retired or tr.role != 'player' or len(tr.descs) < 3:
                continue
            # A returning goalkeeper whose kit voted for a team: their own appearance decides, unless this
            # person's own temporal evidence plainly says team player (a gallery can be contaminated).
            d = tr.decision
            if d is not None and d.role == 'player' and d.role_confidence >= .6 and d.referee_confidence < .4 and d.goalkeeper_confidence < .4 \
                    and (self._looks(tr, ctx['time']).get('referee') or 0.0) < .55:
                continue
            match = self._special_match(tr, ('goalkeeper',))
            if match is None or match[1] < .8:
                continue
            rival = self._best_appearance(tr, lambda g: g.role == 'player' and g.team == tr.team)
            if match[1] >= rival+.2:
                g = match[0]
                old = self._describe(tr.role, tr.team)
                closest = f'the closest team {tr.team} player' if tr.team else 'the closest player'
                tr.role, tr.team = g.role, g.team or tr.team
                tr.reid.clear()
                name = g.pid if g.display == g.pid else f'{g.display} ({g.pid})'
                out['events'].append(self._event('role', tr.id, g.pid, '\n'.join([
                    'ROLE UPDATE', f'Local Track: {tr.id}', f'Old role: {old} (kit vote)', f'New role: {ROLE_OUT[g.role]}',
                    f'Looks like {name}: appearance {match[1]:.2f} vs {rival:.2f} for {closest}'])))
        for tr in uncertain:
            if tr.retired:
                continue
            if tr.manual_hits:
                tr.reason = f'checking: looks like {tr.manual_pid}, identified by you'
                tr.reid.clear()
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
            free = 0 if tr.role == 'goalkeeper' else self._free_count(tr.team)
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

            def waits(s):
                # An identity lost moments ago that this person strongly resembles is not ruled out on the predicted
                # position alone (it came from where the occlusion lost them): the track waits for clean frames
                # instead of becoming a new identity, for a while.
                return s['missing'] <= SHORT_LOSS and s['appearance'] is not None and s['appearance'] >= STRONG_LOOK and \
                    ctx['time']-tr.born <= SHORT_HOLD

            def implausible(s):
                # Goalkeepers do not change jerseys mid-match: a clearly different jersey rules a missing one
                # out, however well the place and the timing fit.
                other_kit = tr.role == 'goalkeeper' and s['jersey'] <= .2 and s['appearance'] is not None and s['appearance'] < .6
                far = s['spatial'] is not None and s['spatialShare'] >= .5 and s['spatial'] < .15 and not waits(s)
                return s['final'] < .3 or far or (p['stranger'] is not None and p['stranger']-s['final'] >= o.reid_margin) or other_kit
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
            elif best['final'] < o.reid_min and best['spatial'] is not None and best['spatial'] < .15 and waits(best):
                reason = f"resembles {best['id']} (appearance {pct(best['appearance'])}), lost {best['missing']:.1f} s ago, " \
                    f"but only {pct(best['spatial'])} plausible where this track appeared: waiting for clean frames"
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
        # The kinematic memory may be older than the last sighting (partial boxes at an occlusion hold it): the reach
        # of the remembered position grows from there.
        since = max(gap, tr.born-g.clean_seen) if math.isfinite(g.clean_seen) else gap
        k = min(since, 2.0)
        team = tr.role == 'goalkeeper' or tr.team is None or g.team is None or tr.team == g.team
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
            m = self._reach(tr.stab.x-g.stab.x, tr.stab.y-g.stab.y, g.velocity, k2, lambda dx, dy: math.hypot(dx*ctx['aspect'], dy)/h*PLAYER_HEIGHT)
            spatial = math.exp(-.5*(m/(3+7*dt))**2)
            final = combine({'appearance': appearance, 'uniform': uniform, 'spatial': spatial, 'temporal': 1.0, 'team': team})
            return {'appearance': appearance, 'jersey': jersey, 'uniform': uniform, 'team': team, 'spatial': spatial,
                    'movement': None, 'temporal': 1.0, 'final': final, 'missing': dt, 'cosine': c.get('cosine') if c else None}
        # Detected while the identity was still detected on another track: two different people, unless this
        # track was born as a second box on that very track (a split detection of the same person).
        duplicate = g.last_track is not None and tr.born_over.get(g.last_track, 0.0) >= DUPLICATE_BIRTH
        together = same and tr.born < g.last_seen-1e-3 and not released and not duplicate
        spatial = movement = None
        sigma = 3+7*since
        if tr.first_pitch is not None and g.last_pitch is not None and sigma <= SPATIAL_HORIZON:
            m = self._reach(tr.first_pitch[0]-g.last_pitch[0], tr.first_pitch[1]-g.last_pitch[1], g.pitch_velocity, k,
                            lambda dx, dy: math.hypot(dx*PITCH_LENGTH, dy*PITCH_WIDTH))
            spatial = math.exp(-.5*(m/sigma)**2)
        if same:
            extra = max(0, ctx['drift']-g.drift)
            h = max(.005, (g.stab.h+tr.first_stab.h)/2)
            metres = lambda dx, dy: math.hypot(dx*ctx['aspect'], dy)/h*PLAYER_HEIGHT
            loose = sigma+5*extra
            if spatial is None and extra <= 6 and loose <= SPATIAL_HORIZON:
                m = self._reach(tr.first_stab.x-g.stab.x, tr.first_stab.y-g.stab.y, g.velocity, k, metres)
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
    def _reach(dx, dy, v, k, metres):
        """Metres between where a new track appeared (dx, dy from the remembered position, in image or pitch units)
        and where the identity is expected: it kept moving at its remembered velocity for k seconds, or it stopped
        (a player hidden behind someone often has), whichever is closer. The velocity is capped at a sprint: anything
        faster is a measurement artifact, not a run."""
        vx, vy = v or (0.0, 0.0)
        speed = metres(vx, vy)
        if speed > MAX_SPEED:
            vx, vy = vx*MAX_SPEED/speed, vy*MAX_SPEED/speed
        return min(metres(dx-vx*k, dy-vy*k), metres(dx, dy))

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
        if tr.decision is not None and tr.decision.role not in ('unknown', g.role):
            lines.append(f'Raw current role prediction: {ROLE_OUT.get(tr.decision.role, "UNKNOWN")} {tr.raw_role[1]:.2f} -> IGNORED (identity history: {self._old_role(g)})')
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
        d = tr.decision
        if tr.role != 'goalkeeper' and d is not None and d.role == 'referee':
            # New player confirmation: a person whose own latest evidence reads referee-like is never made a new
            # player. The missing players were compared first; nobody fits, so the person is ignored.
            self._mark_nontarget(tr, d, ctx, out, 'no missing player fits, and a new player needs player evidence')
            return False
        group = 'goalkeeper' if tr.role == 'goalkeeper' else tr.team
        n, cap = self._count(group), self._cap(group)
        over = n >= cap
        if over:
            observed = ctx['time']-tr.born
            hard = n >= cap+5 or tr.role == 'goalkeeper'
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
        role = 'goalkeeper' if tr.role == 'goalkeeper' else 'player'
        team = tr.team
        if role == 'goalkeeper' and team is not None:
            # The team from the keeper-kit vote is evidence, not ownership: the slot decides.
            owner = self.goalkeeper_of(team)
            if owner is not None and not self._take_slot(tr, tr.decision, owner, pid, f'a stronger goalkeeper candidate for team {team} ({pid}: {(tr.decision.scores.get("keeper") or {}).get("route", "")})', ctx, out):
                team, tr.team = None, None
        g = GlobalPlayer(pid, role, team, first_seen=ctx['time'], last_seen=ctx['time'], last_box=list(tr.box), created_on=tr.id, clean_seen=ctx['time'])
        if role == 'goalkeeper' and tr.decision is not None:
            g.keeper_goal = tr.decision.scores['nearGoal']
        if role != 'player':
            g.role_locked = tr.role_confidence >= ROLE_LOCK
            g.role_history.append((r3(ctx['time']), role, f'created with role confidence {tr.role_confidence:.2f}'))
            if tr.decision is not None:
                g.referee_confidence, g.goalkeeper_confidence = tr.decision.referee_confidence, tr.decision.goalkeeper_confidence
                g.role_scores = tr.decision.scores
        self.registry[pid] = g
        for d, t, occluded in tr.descs:
            if not occluded:
                g.gallery = add_to_gallery(g.gallery, d, t, self.o.gallery_size)
        conf = tr.role_confidence if tr.role == 'goalkeeper' else tr.team_confidence
        crossed = self._bind(tr, g, ctx['time'], clamp(.55+.4*conf, .5, .95))
        why = 'every missing identity of this group is implausible here' if rejected else f'no missing {self._group_name(group)} identity'
        if over:
            why += f'; {self._group_name(group)} now has {n+1} identities (expected at most {cap}), check for duplicates'
            out['issues'].append({'id': pid, 'time': ctx['time'], 'reason': f'{self._group_name(group)} exceeds the expected {cap} identities.'})
        out['events'].append(self._event('new-identity', tr.id, pid, f'Track {tr.id} becomes new identity {long_id(pid)} ({self._describe(g.role, g.team)}; {why}{"; follows a crossing, check" if crossed else ""}).'))
        d = tr.decision
        if role == 'goalkeeper' and d is not None:
            lines = ['GOALKEEPER IDENTIFIED', f'Global ID: {pid}', f'Team: {g.team or "unknown yet"}'] + self._keeper_lines(d)
            out['events'].append(self._event('role', tr.id, pid, '\n'.join(lines + [f"Role locked: {'yes' if g.role_locked else 'not yet'}"]),
                                             {role: d.goalkeeper_confidence, **d.scores}))
        self._flush(tr, g, ctx['time'])
        return True

    def _next_id(self, tr):
        return self._next_id_for(tr.role, tr.team)

    def _next_id_for(self, role, team):
        if role == 'goalkeeper':
            key, fmt = 'GK', 'GK-{}'
        else:
            key, fmt = team, team+'-{:02d}'
        while True:
            self.counters[key] += 1
            pid = fmt.format(self.counters[key])
            if pid not in self.registry:
                return pid

    def _group_name(self, group):
        return {'goalkeeper': 'Goalkeepers'}.get(group, f'Team {group}')

    def _cap(self, group):
        return self.o.max_goalkeepers if group == 'goalkeeper' else self.o.max_per_team

    def _count(self, group):
        n = 0
        for g in self.registry.values():
            if g.status in ('substituted', 'unknown', 'retired'):
                continue
            if group == 'goalkeeper':
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
        g.last_track = tr.id
        if not g.tracks or g.tracks[-1] != tr.id:
            g.tracks.append(tr.id)
        g.identity_confidence = r3(clamp(conf))
        g.team_confidence = r3(tr.team_confidence) if g.role == 'player' else g.team_confidence
        g.role_confidence = r3(max(g.role_confidence, tr.role_confidence) if g.role_locked else tr.role_confidence)
        g.cap_until = 0.0
        crossed = time-g.crossed_at < 10 or time-tr.crossed_at < 10
        if crossed:
            g.cap_until = time+SWAP_CAP
        if g.role == 'goalkeeper':
            # A player identity created on this track that shows this same person (not someone the track
            # switched away from) was this goalkeeper: retire it and move its observations.
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
        if g.manual:
            person['manual'] = True

    def _restyle(self, g):
        """After a role or team change of an identity, every observation already labelled with it follows."""
        for person in g.emitted_all:
            if person.get('id') == g.pid:
                person.update(display=g.display, team=g.team, role=ROLE_OUT.get(g.role, 'UNKNOWN'), label=g.role_label)

    def _move_observations(self, old, new, only_track=None, since=-math.inf):
        """Relabel observations of `old` (all, or one local track's, from `since` on) as identity `new`."""
        keep = []
        for person in old.emitted_all:
            if person.get('id') == old.pid and (only_track is None or person.get('track') == only_track) and person.get('time', 0) >= since:
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
            # A similar person: a teammate in the same role, or someone not yet decided. An ignored (referee-like)
            # person is not similar (another kind of person, as a referee identity was before); a real swap with
            # them is caught by the switch check, and an overlap with them freezes the memory anyway.
            similar = next((o for o in live if o is not tr and o.state not in ('rejected', 'nontarget') and not o.retired and
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
            if tick and self._check_switch(tr, g, ctx, out):
                g = None
            person = {'time': t, 'track': tr.id, 'box': [round(float(v), 5) for v in tr.box], 'score': round(float(tr.score), 3),
                      'cls': tr.cls, 'zone': tr.zone, 'evidence': 'observed'}
            if tr.pitch is not None:
                person['pitch'] = [round(float(tr.pitch[0]), 4), round(float(tr.pitch[1]), 4)]
            if tick and tr.decision is not None and tr.decision.why:
                person['why'] = tr.decision.why
            if tick:
                person['rawRole'], person['rawRoleConfidence'] = ROLE_OUT.get(tr.raw_role[0], 'UNKNOWN'), tr.raw_role[1]
                person['occlusion'] = round(tr.overlap, 2)
                if tr.overlap_with is not None:
                    other = self.tracks.get(tr.overlap_with)
                    og = self.registry.get(other.player_id) if other is not None and other.player_id else None
                    person['occludedWith'] = og.display if og is not None else f'IGNORE (track {tr.overlap_with})' if other is not None and \
                        other.state == 'nontarget' else f'track {tr.overlap_with}'
            if g is not None and tick:
                self._check_mismatch(tr, g, ctx, out)
                if tr.player_id != g.pid:
                    g = None
            if g is None:
                display, label = self._unbound_label(tr)
                role = ROLE_OUT.get(tr.role, 'UNKNOWN') if tr.state == 'uncertain' else 'NON_TARGET' if tr.state == 'nontarget' else 'UNKNOWN'
                person.update(id=None, display=display, label=label, team=None if tr.state == 'nontarget' else tr.team, role=role, state=tr.state,
                              identityConfidence=r3(tr.best if tr.state == 'uncertain' else 0), roleConfidence=r3(tr.role_confidence))
                if tr.state == 'nontarget':
                    person['hint'] = tr.hint
                if tr.reason:
                    person['reason'] = tr.reason[:160]
                if tr.excluded:
                    person['manual'], person['excludedId'] = True, tr.excluded
                if not tr.retired and tr.state in ('candidate', 'uncertain', 'unknown', 'nontarget'):
                    # Kept: labelled retroactively once the person is identified. A spell with the kit undecided
                    # ('unknown': a small, distant player) is the same local track and the same person.
                    tr.pending.append(person)
                    while tr.pending and t-tr.pending[0]['time'] > BUFFER_SECONDS:
                        tr.pending.popleft()
                out['people'].append(person)
                continue
            # The kinematic memory (where they are and how fast they move, in the image and on the pitch) comes from
            # clean boxes: a partial box at an occlusion holds the last clean one, unless that is stale or from another
            # camera segment.
            fresh = not tr.partial or g.stab is None or g.segment != ctx['segment'] or t-g.clean_seen > 1.5
            g.last_seen, g.last_box, g.segment, g.drift = t, list(tr.box), ctx['segment'], ctx['drift']
            g.exit_edge, g.status = '', 'active'
            if fresh:
                g.stab, g.velocity, g.clean_seen = tr.stab, tr.velocity, t
                if tr.pitch is not None:
                    g.last_pitch, g.pitch_velocity = tr.pitch, tr.pitch_velocity
                else:
                    g.last_pitch = g.pitch_velocity = None
            if tick:
                self._remember(tr, g, t, out)
                person['stableRole'], person['memory'] = ROLE_OUT.get(g.role, 'UNKNOWN'), tr.memory
                if tr.partial:
                    person['partial'] = True
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

    def _remember(self, tr, g, t, out):
        """Selective memory: this tick's crop joins the identity's appearance gallery only when it is clean
        (overlapping nobody, good quality, on the pitch), the binding has settled (not just made, not
        after a crossing) and it looks like the gallery already there. Otherwise the memory is frozen
        for this tick, and a close encounter between a player (or goalkeeper) and an ignored person is
        logged."""
        cur = tr.current
        if cur is None:
            tr.memory = 'frozen: no crop'
            return
        if tr.contaminated:
            tr.memory = (f'frozen: overlap {tr.overlap:.2f}' if tr.overlap >= OVERLAP_FREEZE else f'frozen: {tr.covered:.0%} of the box inside another') + \
                (f' with track {tr.overlap_with}' if tr.overlap_with is not None else '')
            other = self.tracks.get(tr.overlap_with) if tr.overlap_with is not None else None
            if other is not None and other.state == 'nontarget' and tr.overlap >= .25:
                key = (min(tr.id, other.id), max(tr.id, other.id))
                e = self.encounters.setdefault(key, {'start': t, 'ids': (g.pid,), 'tracks': (tr.id, other.id), 'with': other.hint, 'max': 0.0})
                e['max'] = max(e['max'], tr.overlap)
                if self._allow(f'occlusion-{key[0]}-{key[1]}', t, 5):
                    out['events'].append(self._event('occlusion', tr.id, g.pid, '\n'.join(['OCCLUSION EVENT', f'Track {tr.id}: Global {g.pid} ({ROLE_OUT[g.role]})',
                                                                                             f"Track {other.id}: ignored person ({other.hint.lower().replace('_', ' ')}; no identity)",
                                                                                             f'IoU: {tr.overlap:.2f}',
                                                                                             f'Action: FREEZE APPEARANCE + TEAM + ROLE MEMORY of {g.pid}; the ignored person cannot take the identity'])))
            return
        if tr.occluded:
            tr.memory = 'frozen: occluded'
            return
        if cur.quality < .3 or tr.zone == 'outside':
            tr.memory = 'frozen: low quality' if cur.quality < .3 else 'frozen: outside the pitch'
            return
        if t-g.bound_at < .6 or t < g.cap_until or t-tr.crossed_at <= 2.0:
            tr.memory = 'frozen: identity just bound or after a crossing'
            return
        if self._conf(g, t) < .6:
            tr.memory = 'frozen: identity confidence low'
            return
        if len(g.gallery) >= 2:
            c = compare(g.gallery, [cur])
            if c is not None and c.get('cosine') is not None and c['cosine'] < GALLERY_MIN_COSINE:
                tr.memory = f"frozen: crop unlike the identity (cosine {c['cosine']:.2f})"
                return
        g.gallery = add_to_gallery(g.gallery, cur, t, self.o.gallery_size)
        tr.memory = 'updated'

    def _check_switch(self, tr, g, ctx, out):
        """An id switch inside one local track: the tracker handed the track from one person to another
        when they crossed. A clean crop that looks like neither the track's own first clean crops nor its
        identity's gallery, SWITCH_CHECKS ticks in a row, means the track now shows someone else. The track
        is re-born at the first such crop: evidence, kit votes, first crops and birth time start afresh, so
        the "seen together" rule no longer counts its past and the person it shows now is decided from
        scratch (compared with the missing players first; non-target if referee-like). A bound identity
        is released, missing, with its last sighting moved back to before the switch; the frames since
        the switch lose that label (the offline pass reconnects them). Returns True when the track was
        re-born."""
        cur = tr.current
        if cur is None or tr.retired or tr.excluded or tr.occluded or tr.contaminated or cur.quality < .3 or \
                cur.embedding is None or len(tr.first) < 2:
            return False
        refs = [f.embedding for f in tr.first]+([x.d.embedding for x in g.gallery] if g is not None else [])
        cosines = [c for c in (embedding_cosine(r, cur.embedding) for r in refs) if c is not None]
        if not cosines:
            return False
        best = max(cosines)
        t = ctx['time']
        if best >= SWITCH_COSINE:
            tr.switch_miss, tr.switch_since, tr.last_match = 0, -math.inf, t
            tr.switch_crops.clear()
            return False
        if tr.switch_miss == 0:
            tr.switch_since = t
        tr.switch_miss += 1
        tr.switch_crops.append((t, cur))
        if tr.switch_miss < SWITCH_CHECKS:
            return False
        # One other person, not noise: the unlike crops must look like each other.
        recent = list(tr.switch_crops)
        alike = [c for c in (embedding_cosine(a.embedding, b.embedding) for k, (_, a) in enumerate(recent) for _, b in recent[k+1:]) if c is not None]
        if not alike or min(alike) < SWITCH_ALIKE:
            return False
        # The switch happened somewhere after the last crop that still matched (the overlap in between is
        # ambiguous): the labels since then go, at most 3 s before the first of the unlike crops; the track's
        # new life starts at that crop (the person it shows may have been on their own track until just
        # before: "seen together" counts from here).
        born = recent[0][0]
        since = max(tr.last_match+1e-3, born-3.0) if tr.last_match > -math.inf else born
        was = self._describe(tr.role, tr.team) if tr.state in ('uncertain', 'confirmed') else 'CANDIDATE'
        lines = ['TRACK SWITCH', f'Local Track: {tr.id}',
                 f'Before: {was}' + (f', Global {g.pid}' if g is not None else ' (no global identity yet)'),
                 f'Now: a different person (OSNet cosine {best:.2f} vs the track\'s own first crops' + (' and the identity gallery' if g is not None else '') +
                 f', {SWITCH_CHECKS} clean crops since {born:.2f} s that agree with each other at {min(alike):.2f})']
        stale = []
        if g is not None:
            self._unbind(g)
            g.last_track = tr.id
            g.last_seen = min(g.last_seen, since-1e-3)
            for when, person in tr.emitted:
                if when >= since-1e-6 and person.get('id') == g.pid:
                    if person in g.emitted_all:
                        g.emitted_all.remove(person)
                        g.observations -= 1
                    stale.append(person)
            lines.append(f'Action: {g.pid} released (MISSING, last seen before {since:.2f} s); identity, team and role unchanged')
        else:
            stale = [person for person in tr.pending if person.get('time', 0) >= since-1e-6]
        for person in stale:
            person.update(id=None, display='?', label='IDENTITY_UNCERTAIN', team=None, role='UNKNOWN', state='uncertain',
                          identityConfidence=0.0, reason=f'track switched person at {since:.2f} s')
            for key in ('stableRole', 'memory'):
                person.pop(key, None)
        lines.append('Action: track re-born: evidence restarted, decided from scratch (missing players first)')
        self._rebirth(tr, since, born, ctx)
        tr.pending.extend(stale)
        out['events'].append(self._event('switch', tr.id, g.pid if g is not None else None, '\n'.join(lines)))
        if g is not None:
            out['issues'].append({'id': g.pid, 'time': r3(t), 'reason': f'Track {tr.id} switched from {g.pid} to someone else at {since:.2f} s; {g.pid} is missing.'})
        return True

    def _rebirth(self, tr, since, born, ctx):
        """The track starts afresh: cut at `since`, born at `born`, with only what it showed from then on."""
        tr.switches.append(r3(since))
        tr.born, tr.born_segment = born, ctx['segment']
        tr.first_stab, tr.first_pitch, tr.first_zone, tr.entry_edge = tr.stab, tr.pitch, tr.zone, ''
        tr.hits = tr.inside = tr.boundary = tr.outside = tr.outside_run = 0
        tr.zones.clear()
        tr.evidence, tr.decision = RoleEvidence(), None
        tr.team_votes.clear()
        tr.hist.clear()
        tr.heights.clear()
        tr.velocity, tr.pitch_velocity = (0.0, 0.0), None
        recent = [(d, when, occluded) for d, when, occluded in tr.descs if when >= since-1e-6]
        tr.descs.clear()
        tr.descs.extend(recent)
        tr.first = [d for d, _, occluded in recent if not occluded][:2]
        tr.player_id, tr.state, tr.team, tr.role = None, 'candidate', None, 'unknown'
        tr.team_confidence = tr.role_confidence = tr.best = 0.0
        tr.reid.clear()
        tr.blind, tr.released, tr.looks = [], None, None
        tr.former.clear()
        tr.born_over = {}
        tr.role_streak, tr.role_wanted, tr.role_checked = 0, None, -math.inf
        tr.kit_miss = tr.look_miss = tr.switch_miss = 0
        tr.switch_since = tr.last_match = -math.inf
        tr.switch_crops.clear()
        tr.last_deferred, tr.deferrals = -math.inf, 0
        tr.manual_checked, tr.manual_hits, tr.manual_pid, tr.manual_doubt = -math.inf, 0, None, 0
        tr.raw_role, tr.reacquired = ('unknown', 0.0), False
        tr.hint, tr.nontarget_since, tr.exit_streak, tr.official_look = '', -math.inf, 0, False
        tr.pending.clear()
        tr.emitted.clear()
        tr.reason = f'track switched person at {since:.2f} s: re-identifying'

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
        if tr.excluded:
            return 'OUT', 'EXCLUDED_BY_YOU'
        if tr.state == 'nontarget':
            return 'IGNORE', 'NON_TARGET'
        if tr.state == 'uncertain':
            display = (f'GK-{tr.team}?' if tr.team else 'GK-?') if tr.role == 'goalkeeper' else f'{tr.team}-?' if tr.team else '?'
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
            if d is None or d.quality < .25 or not d.has_jersey or zone == 'outside' or (tr is not None and (tr.retired or tr.state in ('rejected', 'nontarget'))):
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
        """(jersey, shorts) of people the detector consistently calls referees and of people ignored as
        referee-like with independent support. A track the detector calls referee most of the time whose
        kit only loosely matches a team (near the edge of that team's colour spread) counts too: that is a
        referee in a kit close to a team's. For the referee kit prototype of the team model only."""
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
        for tr in list(self.tracks.values())+list(self.ended.values()):
            if tr.state != 'nontarget' or not tr.official_look:
                continue
            if tr.id in self.tracks and tr.evidence.detector_shares()['referee'] >= .8:
                continue   # already counted above
            out += [(d.jersey, d.shorts) for d in self._official_crops(tr, 2) if d.has_jersey]
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
        # 'referees' stays (always empty) for pages and tools that read results of earlier versions.
        groups = {'players': [], 'goalkeepers': [], 'referees': [], 'retired': [], 'excluded': [], 'withdrawn': []}
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
                     'roleEvidence': g.role_scores, 'teamHistory': {'clearKitVotes': g.kit_clear, 'kitVotes': g.kit_total, 'established': self._established(g)},
                     'revoked': g.revoked or None, 'keeperGoal': r3(g.keeper_goal), 'manual': g.manual}
            if g.withdrawn:
                entry.update(status='WITHDRAWN', withdrawn=g.withdrawn)
                groups['withdrawn'].append(entry)
            elif g.status == 'retired':
                entry['retiredInto'] = g.retired_into
                groups['retired'].append(entry)
            elif g.excluded:
                entry.update(excluded=True, status='EXCLUDED')
                groups['excluded'].append(entry)
            else:
                groups['goalkeepers' if g.role == 'goalkeeper' else 'players'].append(entry)
        return groups

    def goal_sides(self):
        """Which team defends which side of the image, as accumulated votes."""
        return {side: dict(votes) for side, votes in self.side_votes.items()}
