"""Team kits from torso colour, per-observation kit votes, and temporal role decisions.

The soccer detector already separates player / goalkeeper / referee classes; those class votes
are accumulated per track and combined with kit evidence. Two team kits are learned from the
jersey (upper torso) histograms of people the detector calls players, so referees and keepers do
not pull a team prototype. Nothing is decided from one frame: a track stays CANDIDATE until enough
consistent votes exist, and an established team only changes after a strong majority of recent
votes points the other way.
"""
import math
from collections import Counter, deque
from dataclasses import dataclass, field

import numpy as np

KIT_MARGIN = .06       # required lead of the nearer kit prototype
MIN_SAMPLES = 6
MIN_KIT_SEPARATION = .35
MIN_KIT_GAP = .3
MAX_VOTES = 40


@dataclass
class TeamModel:
    a: np.ndarray | None = None
    b: np.ndarray | None = None
    referee: np.ndarray | None = None
    spread: float = .2
    samples: int = 0

    @property
    def ready(self):
        return self.a is not None and self.b is not None

    def to_json(self):
        r = lambda v: None if v is None else [round(float(x), 4) for x in v]
        return {'learned': self.ready, 'a': r(self.a), 'b': r(self.b), 'referee': r(self.referee),
                'spread': round(self.spread, 3), 'samples': self.samples,
                'meaning': 'Kit groups learned from torso colour; A/B naming is arbitrary within the run.'}

    @classmethod
    def from_json(cls, v):
        if not isinstance(v, dict):
            return cls()
        arr = lambda x: np.asarray(x, np.float64) if isinstance(x, list) and len(x) == 24 else None
        return cls(arr(v.get('a')), arr(v.get('b')), arr(v.get('referee')), float(v.get('spread', .2)), int(v.get('samples', 0)))


def _normalize(x):
    x = np.maximum(np.asarray(x, np.float64), 0)
    s = x.sum()
    return x/s if s > 0 else x


def hellinger(a, b):
    if a is None or b is None:
        return 1.0
    a, b = _normalize(a), _normalize(b)
    if a.sum() <= 0 or b.sum() <= 0:
        return 1.0
    return math.sqrt(max(0.0, 1-float(np.sqrt(a*b).sum())))


def pairwise(xs):
    roots = np.sqrt(np.array([_normalize(x) for x in xs]))
    return np.sqrt(np.clip(1-roots @ roots.T, 0, 1))


def hue_only(v):
    """Jersey histogram without its two brightness levels per hue: shade and floodlights move mass
    between the levels of one hue, a different kit changes the hue (or the greys)."""
    v = np.asarray(v, np.float64)
    if len(v) != 24:
        return v
    return np.concatenate([v[:4], v[4::2]+v[5::2]])


def _trim_limit(d):
    return min(.65, max(.3, 2.5*float(np.median(d))+.1)) if len(d) else .65


def weighted_mean(xs, ws):
    xs = [_normalize(x) for x in xs]
    total = sum(ws)
    if not xs or total <= 0:
        return None
    return sum(w*x for x, w in zip(xs, ws))/total


def _refine(xs, ws, centres, fixed=(False, False)):
    labels = inlier = dist = None
    for _ in range(20):
        da = np.array([hellinger(x, centres[0]) for x in xs])
        db = np.array([hellinger(x, centres[1]) for x in xs])
        nl = (db < da).astype(int)
        nd = np.where(nl == 0, da, db)
        limits = [_trim_limit(nd[nl == k]) for k in (0, 1)]
        ni = nd <= np.array(limits)[nl]
        stable = labels is not None and (nl == labels).all() and (ni == inlier).all()
        labels, inlier, dist = nl, ni, nd
        if stable:
            break
        for k in (0, 1):
            if fixed[k]:
                continue
            idx = np.flatnonzero((labels == k) & inlier)
            if len(idx):
                centres[k] = weighted_mean([xs[i] for i in idx], [ws[i] for i in idx])
    members = [np.flatnonzero((labels == k) & inlier) for k in (0, 1)]
    return centres, members, dist, inlier


def _modes(D, maximum=5):
    """Density modes: samples with two close neighbours, densest first, well separated. A lone
    referee or keeper never seeds a kit cluster."""
    if len(D) < 3:
        return []
    knn = np.sort(D+np.eye(len(D))*9, axis=1)[:, 1]
    out = []
    for i in np.argsort(knn, kind='stable'):
        if knn[i] > MIN_KIT_GAP or len(out) >= maximum:
            break
        if all(D[i, j] >= MIN_KIT_SEPARATION for j in out):
            out.append(int(i))
    return out


def _min_group(n):
    return max(3 if n >= 10 else 2, math.ceil(n*.25))


def fit_team_model(samples, previous=None):
    """samples: [{'jersey': hist(24), 'weight': float, 'order': float}] from people the detector calls
    players. Returns a TeamModel whose A/B mapping stays stable relative to `previous`."""
    previous = previous or TeamModel()
    usable = [s for s in samples if s['jersey'] is not None and np.isfinite(s['jersey']).all() and np.sum(s['jersey']) > 0]
    if len(usable) < MIN_SAMPLES:
        return TeamModel(previous.a, previous.b, previous.referee, previous.spread, len(usable))
    if len(usable) > 200:
        step = math.ceil(len(usable)/200)
        usable = usable[::step]
    xs = [s['jersey'] for s in usable]
    ws = [max(1e-3, float(s.get('weight', 1))) for s in usable]
    order = [s.get('order', i) for i, s in enumerate(usable)]
    D = pairwise(xs)
    seeds, need = _modes(D), _min_group(len(xs))
    best = None
    for i in range(len(seeds)):
        for j in range(i+1, len(seeds)):
            centres, members, dist, inlier = _refine(xs, ws, [_normalize(xs[seeds[i]]), _normalize(xs[seeds[j]])])
            spread = min(.35, max(.08, float(np.median(dist[inlier])))) if inlier.sum() >= 4 else .2
            if any(len(m) < need for m in members):
                continue
            if hellinger(centres[0], centres[1]) < max(MIN_KIT_SEPARATION, 2*spread+.1):
                continue
            gaps = [min(D[a, b] for b in members[1-k]) for k in (0, 1) for a in members[k]]
            if float(np.median(gaps)) < max(MIN_KIT_GAP, 1.5*spread):
                continue  # one kit under changing light is a continuum, not two teams
            score = sum(ws[i] for m in members for i in m)
            if best is None or score > best[0]+1e-9:
                best = (score, centres, members, spread)
    if best is None:
        return TeamModel(previous.a, previous.b, previous.referee, previous.spread, len(xs))
    _, centres, members, spread = best
    first = [min(order[i] for i in m) for m in members]
    a, b = (centres[0], centres[1]) if first[0] <= first[1] else (centres[1], centres[0])
    if previous.ready:
        if hellinger(a, previous.b)+hellinger(b, previous.a) < hellinger(a, previous.a)+hellinger(b, previous.b):
            a, b = b, a
        # Both clusters are shades of one known kit (the other team left the view): a lighting split.
        hue = lambda x, y: hellinger(hue_only(x), hue_only(y))
        near = lambda x: 'a' if hue(x, previous.a) <= hue(x, previous.b) else 'b'
        if near(a) == near(b):
            return TeamModel(previous.a, previous.b, previous.referee, previous.spread, len(xs))
        # Smooth so a single refit cannot jump the prototypes.
        a = _normalize(.7*_normalize(previous.a)+.3*_normalize(a))
        b = _normalize(.7*_normalize(previous.b)+.3*_normalize(b))
    return TeamModel(a, b, previous.referee, spread, len(xs))


def with_referee(model, jerseys):
    """Referee kit prototype from people the detector consistently calls referees."""
    if len(jerseys) < 2:
        return model
    D = pairwise(jerseys)
    medoid = int(np.argmin(D.sum(axis=1)))
    keep = [x for x, d in zip(jerseys, D[medoid]) if d <= _trim_limit(D[medoid])]
    return TeamModel(model.a, model.b, weighted_mean(keep, [1]*len(keep)), model.spread, model.samples)


@dataclass
class KitVote:
    team: str | None
    dist_a: float
    dist_b: float
    dist_ref: float
    outlier: bool
    margin: float
    ref_like: bool = False
    valid: bool = False


def kit_vote(model, jersey):
    """team: nearer prototype when it leads by KIT_MARGIN and is within 2.5*spread+.1. outlier: far
    from both kits or clearly closer to the referee kit. Ties stay undecided."""
    has = jersey is not None and float(np.sum(jersey)) > 0
    da = hellinger(jersey, model.a) if has and model.a is not None else 1.0
    db = hellinger(jersey, model.b) if has and model.b is not None else 1.0
    dr = hellinger(jersey, model.referee) if has and model.referee is not None else 1.0
    vote = KitVote(None, round(da, 3), round(db, 3), round(dr, 3), False, round(abs(da-db), 3))
    if not has or not model.ready:
        vote.margin = 0.0
        return vote
    vote.valid = True
    limit = 2.5*model.spread+.1
    near = min(da, db)
    if dr < limit and dr+KIT_MARGIN <= near:
        vote.outlier = vote.ref_like = True
    elif near >= limit:
        # A darker or brighter crop of one team's hue (stadium shadow) is still that team.
        ha, hb = hellinger(hue_only(jersey), hue_only(model.a)), hellinger(hue_only(jersey), hue_only(model.b))
        hr = hellinger(hue_only(jersey), hue_only(model.referee)) if model.referee is not None else 1.0
        if min(ha, hb) < limit and abs(ha-hb) >= KIT_MARGIN and hr >= min(ha, hb)+KIT_MARGIN:
            vote.team, vote.margin = ('A' if ha < hb else 'B'), round(abs(ha-hb), 3)
        else:
            vote.outlier = True
    elif abs(da-db) >= KIT_MARGIN:
        vote.team = 'A' if da < db else 'B'
    return vote


ROLE_NAMES = {'player': 'PLAYER', 'goalkeeper': 'GOALKEEPER', 'referee': 'REFEREE'}


def role_label(role, team):
    if role == 'referee':
        return 'REFEREE'
    if role == 'goalkeeper':
        return f'GOALKEEPER_TEAM_{team}' if team else 'GOALKEEPER'
    if role == 'player' and team:
        return f'PLAYER_TEAM_{team}'
    return 'UNKNOWN'


@dataclass
class RoleEvidence:
    hits: int = 0
    detector: deque = field(default_factory=lambda: deque(maxlen=MAX_VOTES))  # (class, score)
    kits: deque = field(default_factory=lambda: deque(maxlen=MAX_VOTES))      # KitVote
    near_goal: int = 0

    def add(self, cls, score, vote, near_goal):
        self.hits += 1
        if cls in ('player', 'goalkeeper', 'referee'):
            self.detector.append((cls, max(.05, float(score))))
        if vote is not None and vote.valid:
            self.kits.append(vote)
        self.near_goal += 1 if near_goal else 0

    def detector_shares(self):
        total = sum(s for _, s in self.detector)
        shares = Counter()
        for cls, s in self.detector:
            shares[cls] += s
        return {k: (shares[k]/total if total else 0.0) for k in ('player', 'goalkeeper', 'referee')}


@dataclass
class RoleDecision:
    label: str            # PLAYER_TEAM_A ... | REFEREE | GOALKEEPER[_TEAM_x] | CANDIDATE
    role: str             # player | goalkeeper | referee | unknown
    team: str | None
    team_confidence: float
    role_confidence: float
    reason: str = ''


def decide_role(e, model, min_votes=5):
    """Temporal decision from detector class votes and kit votes; CANDIDATE until strong enough.
    PLAYER needs >= 70% kit votes for one team (mean margin >= .05) and a player majority from the
    detector. REFEREE / GOALKEEPER need a >= 60% detector majority; a referee whose kit clearly
    matches a team is held back (a detector confusion), as is anyone in neither kit."""
    n_det = len(e.detector)
    minimum = max(3, min_votes)
    support = min(1.0, .75+.25*(e.hits-minimum)/minimum) if e.hits >= minimum else .75*e.hits/minimum
    shares = e.detector_shares()
    kits = list(e.kits)
    nk = len(kits)
    count = lambda f: sum(1 for v in kits if f(v))
    na, nb, no = count(lambda v: v.team == 'A'), count(lambda v: v.team == 'B'), count(lambda v: v.outlier)
    team, team_share = None, 0.0
    for t, n in (('A', na), ('B', nb)):
        mine = [v.margin for v in kits if v.team == t]
        if nk and n/nk >= .7 and np.mean(mine) >= .05:
            team, team_share = t, n/nk
    r3 = lambda v: round(float(v), 3)
    candidate = lambda why: RoleDecision('CANDIDATE', 'unknown', None, r3(max(na, nb)/nk*support if nk else 0),
                                         r3(max(shares.values())*support if n_det else 0), why)
    if e.hits < minimum or n_det < minimum:
        return candidate(f'{e.hits}/{minimum} observations')
    if shares['referee'] >= .6:
        if team and shares['referee'] < .85:
            return candidate(f"detector says referee ({shares['referee']:.0%}) but kit matches team {team}")
        out = no/nk if nk else .6
        return RoleDecision('REFEREE', 'referee', None, 0.0, r3(shares['referee']*support*(.6+.4*out)))
    if shares['goalkeeper'] >= .6:
        if team and shares['goalkeeper'] < .85:
            return candidate(f"detector says goalkeeper ({shares['goalkeeper']:.0%}) but kit matches team {team}")
        return RoleDecision('GOALKEEPER', 'goalkeeper', None, 0.0, r3(shares['goalkeeper']*support))
    if not model.ready:
        return candidate('waiting for two team kits')
    if team and shares['player'] >= .5:
        return RoleDecision(role_label('player', team), 'player', team, r3(team_share*support), r3(shares['player']*support))
    if nk and no/nk >= .6:
        return candidate('kit matches neither team')
    return candidate(f'kit votes A {na} / B {nb} / other {no}')
