"""Team kits from torso colour and per-observation kit votes.

Two team kits are learned from the jersey (upper torso) histograms of people the detector calls
players, with a spread per team and a shorts prototype per team. Kits worn by few people (referees,
goalkeepers) are kept as separate prototypes, so a person in another kit is reported as matching
neither team instead of being pushed into whichever team is closest. Role decisions over time live
in roles.py.
"""
import math
from dataclasses import dataclass, field

import numpy as np

KIT_MARGIN = .06       # required lead of the nearer kit prototype
MIN_SAMPLES = 6
MIN_KIT_SEPARATION = .35
MIN_KIT_GAP = .3


@dataclass
class TeamModel:
    a: np.ndarray | None = None
    b: np.ndarray | None = None
    referee: np.ndarray | None = None
    spread: float = .2
    samples: int = 0
    spread_a: float | None = None
    spread_b: float | None = None
    shorts_a: np.ndarray | None = None
    shorts_b: np.ndarray | None = None
    referee_shorts: np.ndarray | None = None
    keepers: list = field(default_factory=list)    # [{'team': 'A'|'B'|None, 'jersey': hist}]

    @property
    def ready(self):
        return self.a is not None and self.b is not None

    def team_spread(self, team):
        value = self.spread_a if team == 'A' else self.spread_b
        return self.spread if value is None else value

    def to_json(self):
        r = lambda v: None if v is None else [round(float(x), 4) for x in v]
        return {'learned': self.ready, 'a': r(self.a), 'b': r(self.b), 'referee': r(self.referee),
                'spread': round(self.spread, 3), 'spreadA': self.spread_a, 'spreadB': self.spread_b,
                'shortsA': r(self.shorts_a), 'shortsB': r(self.shorts_b), 'refereeShorts': r(self.referee_shorts),
                'keepers': [{'team': k['team'], 'jersey': r(k['jersey'])} for k in self.keepers], 'samples': self.samples,
                'meaning': 'Kit groups learned from torso colour; A/B naming is arbitrary within the run.'}

    @classmethod
    def from_json(cls, v):
        if not isinstance(v, dict):
            return cls()
        arr = lambda x, n=24: np.asarray(x, np.float64) if isinstance(x, list) and len(x) == n else None
        keepers = [{'team': k.get('team'), 'jersey': arr(k.get('jersey'))} for k in v.get('keepers', []) if arr(k.get('jersey')) is not None]
        return cls(arr(v.get('a')), arr(v.get('b')), arr(v.get('referee')), float(v.get('spread', .2)), int(v.get('samples', 0)),
                   v.get('spreadA'), v.get('spreadB'), arr(v.get('shortsA'), 12), arr(v.get('shortsB'), 12), arr(v.get('refereeShorts'), 12), keepers)

    def replace(self, **changes):
        values = {name: getattr(self, name) for name in self.__dataclass_fields__}
        values.update(changes)
        return TeamModel(**values)


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
        return previous.replace(samples=len(usable))
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
        return previous.replace(samples=len(xs))
    _, centres, members, spread = best
    first = [min(order[i] for i in m) for m in members]
    swap = first[0] > first[1]
    if previous.ready:
        a0, b0 = (centres[1], centres[0]) if swap else (centres[0], centres[1])
        if hellinger(a0, previous.b)+hellinger(b0, previous.a) < hellinger(a0, previous.a)+hellinger(b0, previous.b):
            swap = not swap
    order_ab = (1, 0) if swap else (0, 1)
    a, b = centres[order_ab[0]], centres[order_ab[1]]
    team_members = [members[order_ab[0]], members[order_ab[1]]]
    spreads = [min(.35, max(.06, float(np.median([hellinger(xs[i], c) for i in m])))) if len(m) >= 3 else spread
               for m, c in zip(team_members, (a, b))]
    shorts = []
    for m in team_members:
        sh = [usable[i]['shorts'] for i in m if usable[i].get('shorts') is not None and np.sum(usable[i]['shorts']) > 0]
        shorts.append(weighted_mean(sh, [1]*len(sh)) if len(sh) >= 3 else None)
    if previous.ready:
        # Both clusters are shades of one known kit (the other team left the view): a lighting split.
        hue = lambda x, y: hellinger(hue_only(x), hue_only(y))
        near = lambda x: 'a' if hue(x, previous.a) <= hue(x, previous.b) else 'b'
        if near(a) == near(b):
            return previous.replace(samples=len(xs))
        # Smooth so a single refit cannot jump the prototypes.
        a = _normalize(.7*_normalize(previous.a)+.3*_normalize(a))
        b = _normalize(.7*_normalize(previous.b)+.3*_normalize(b))
        blend = lambda old, new: new if old is None else old if new is None else _normalize(.7*_normalize(old)+.3*_normalize(new))
        shorts = [blend(previous.shorts_a, shorts[0]), blend(previous.shorts_b, shorts[1])]
        mix = lambda old, new: new if old is None else .7*old+.3*new
        spreads = [mix(previous.spread_a, spreads[0]), mix(previous.spread_b, spreads[1])]
    return previous.replace(a=a, b=b, spread=spread, samples=len(xs), spread_a=spreads[0], spread_b=spreads[1],
                            shorts_a=shorts[0], shorts_b=shorts[1])


def with_referee(model, samples):
    """Referee kit (jersey and shorts) from people the detector consistently calls referees and from
    confirmed referee identities. samples: [(jersey, shorts or None)] or plain jersey histograms."""
    samples = [s if isinstance(s, tuple) else (s, None) for s in samples]
    jerseys = [j for j, _ in samples if j is not None and np.sum(j) > 0]
    if len(jerseys) < 2:
        return model
    D = pairwise(jerseys)
    medoid = int(np.argmin(D.sum(axis=1)))
    keep = [i for i, d in enumerate(D[medoid]) if d <= _trim_limit(D[medoid])]
    shorts = [samples[i][1] for i in keep if samples[i][1] is not None and np.sum(samples[i][1]) > 0]
    return model.replace(referee=weighted_mean([jerseys[i] for i in keep], [1]*len(keep)),
                         referee_shorts=weighted_mean(shorts, [1]*len(shorts)) if len(shorts) >= 2 else model.referee_shorts)


def with_keepers(model, keepers):
    """Goalkeeper kit prototypes from confirmed goalkeeper identities: [(team or None, [jerseys])]."""
    out = []
    for team, jerseys in keepers:
        jerseys = [j for j in jerseys if j is not None and np.sum(j) > 0]
        if jerseys:
            out.append({'team': team, 'jersey': weighted_mean(jerseys, [1]*len(jerseys))})
    return model.replace(keepers=out[:4]) if out else model


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
    keeper_like: bool = False
    keeper_team: str | None = None
    dist_keeper: float = 1.0
    shorts_mismatch: bool = False


def kit_vote(model, jersey, shorts=None):
    """team: the nearer team kit, when it leads by KIT_MARGIN and lies within that team's own spread.
    A kit far from both teams, clearly closer to the referee or a goalkeeper kit, or wearing the
    wrong shorts for the nearer team is an outlier: it matches neither team. Ties stay undecided."""
    has = jersey is not None and float(np.sum(jersey)) > 0
    da = hellinger(jersey, model.a) if has and model.a is not None else 1.0
    db = hellinger(jersey, model.b) if has and model.b is not None else 1.0
    dr = hellinger(jersey, model.referee) if has and model.referee is not None else 1.0
    dk, keeper_team = 1.0, None
    for k in model.keepers:
        d = hellinger(jersey, k['jersey']) if has else 1.0
        if d < dk:
            dk, keeper_team = d, k['team']
    vote = KitVote(None, round(da, 3), round(db, 3), round(dr, 3), False, round(abs(da-db), 3), dist_keeper=round(dk, 3))
    if not has or not model.ready:
        vote.margin = 0.0
        return vote
    vote.valid = True
    near_team = 'A' if da <= db else 'B'
    near = min(da, db)
    limit = min(.5, 2.5*model.team_spread(near_team)+.1)
    special = min(dr, dk)
    if special < .45 and special+KIT_MARGIN <= near:
        vote.outlier = True
        if dr <= dk:
            vote.ref_like = True
        else:
            vote.keeper_like, vote.keeper_team = True, keeper_team
    elif near >= limit:
        # A darker or brighter crop of one team's hue (stadium shadow) is still that team.
        ha, hb = hellinger(hue_only(jersey), hue_only(model.a)), hellinger(hue_only(jersey), hue_only(model.b))
        hr = hellinger(hue_only(jersey), hue_only(model.referee)) if model.referee is not None else 1.0
        if min(ha, hb) < limit and abs(ha-hb) >= KIT_MARGIN and hr >= min(ha, hb)+KIT_MARGIN:
            vote.team, vote.margin = ('A' if ha < hb else 'B'), round(abs(ha-hb), 3)
        else:
            vote.outlier = True
    elif abs(da-db) >= KIT_MARGIN:
        vote.team = near_team
    if vote.team is not None and shorts is not None and np.sum(shorts) > 0:
        team_shorts = model.shorts_a if vote.team == 'A' else model.shorts_b
        if team_shorts is not None:
            ds = hellinger(shorts, team_shorts)
            dref = hellinger(shorts, model.referee_shorts) if model.referee_shorts is not None else 1.0
            if ds > .7 or (ds > .5 and dref+.15 < ds):
                vote.team, vote.outlier, vote.shorts_mismatch = None, True, True
                vote.ref_like = dref+.15 < ds and dr < .55
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
