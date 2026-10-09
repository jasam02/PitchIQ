"""Temporal role classification: PLAYER (team A/B), GOALKEEPER, REFEREE or still undecided.

Every evidence tick (5 per second) a track adds one record: the detector's class (player /
goalkeeper / referee) and score, the kit vote against both teams and against the referee and
goalkeeper kits, and where the person is (near a goal, the deepest person on the pitch, isolated,
on the touchline). Decisions use the last ~10 seconds, never one crop:

- REFEREE: detector referee votes; or a kit like the referee's (or matching neither team) while
  following play away from the goals; or an outlier kit running the touchline (assistant). A kit
  that clearly matches a team holds a referee decision back unless the detector's votes are a clear
  majority. Being consistently the deepest person on the pitch is a goalkeeper's position, not a
  referee's.
- GOALKEEPER: detector goalkeeper votes; or time spent near a goal in a kit unlike the outfield
  kits (or like a known goalkeeper kit) while being the deepest or most isolated person. Needs at
  least one strong cue (detector, goal proximity, a known keeper kit, or a kit matching nobody while
  consistently being the deepest person).
- PLAYER: >= 70% of kit votes for one team.
- Otherwise CANDIDATE (and UNKNOWN after a long time), never forced into Team A or Team B.
"""
from collections import Counter, deque
from dataclasses import dataclass, field

import numpy as np

from .geometry import clamp
from .teams import role_label

WINDOW = 50          # evidence ticks kept (10 s at 5 Hz)
ROLE_THRESHOLD = .55


@dataclass
class RoleEvidence:
    hits: int = 0
    records: deque = field(default_factory=lambda: deque(maxlen=WINDOW))

    def add(self, cls, score, vote, cues=None):
        cues = cues or {}
        self.hits += 1
        self.records.append({'det': cls if cls in ('player', 'goalkeeper', 'referee') else None, 'score': max(.05, float(score)),
                             'vote': vote if vote is not None and vote.valid else None,
                             'near_goal': bool(cues.get('near_goal')), 'goal_side': cues.get('goal_side'),
                             'extreme': bool(cues.get('extreme')), 'isolated': bool(cues.get('isolated')),
                             'zone': cues.get('zone', 'inside')})

    @property
    def detector(self):
        return [r for r in self.records if r['det']]

    @property
    def kits(self):
        return [r['vote'] for r in self.records if r['vote'] is not None]

    def detector_shares(self):
        total = sum(r['score'] for r in self.detector)
        shares = Counter()
        for r in self.detector:
            shares[r['det']] += r['score']
        return {k: (shares[k]/total if total else 0.0) for k in ('player', 'goalkeeper', 'referee')}

    def team_votes(self):
        return [v.team or ('x' if v.outlier else '-') for v in self.kits]


@dataclass
class RoleDecision:
    label: str                  # PLAYER_TEAM_A ... | REFEREE | GOALKEEPER[_TEAM_x] | CANDIDATE
    role: str                   # player | goalkeeper | referee | unknown
    team: str | None
    team_confidence: float
    role_confidence: float
    reason: str = ''
    referee_confidence: float = 0.0
    goalkeeper_confidence: float = 0.0
    scores: dict = field(default_factory=dict)


def _similarity(distance):
    return clamp(1-distance/.6)


def decide_role(e, model, min_votes=5):
    """Temporal decision; CANDIDATE until the evidence is strong enough."""
    records = list(e.records)
    n = len(records)
    minimum = max(3, min_votes)
    support = min(1.0, .75+.25*(e.hits-minimum)/minimum) if e.hits >= minimum else .75*e.hits/minimum
    shares = e.detector_shares()
    kits = e.kits
    nk = len(kits)
    share = lambda f: (sum(1 for v in kits if f(v))/nk) if nk else 0.0
    pos = lambda key: (sum(1 for r in records if r[key])/n) if n else 0.0
    team, team_share = None, 0.0
    for t in ('A', 'B'):
        mine = [v.margin for v in kits if v.team == t]
        if nk and len(mine)/nk >= .7 and np.mean(mine) >= .05:
            team, team_share = t, len(mine)/nk
    outlier, ref_like, keeper_like = share(lambda v: v.outlier), share(lambda v: v.ref_like), share(lambda v: v.keeper_like)
    near_goal, extreme, isolated = pos('near_goal'), pos('extreme'), pos('isolated')
    boundary = (sum(1 for r in records if r['zone'] == 'boundary')/n) if n else 0.0
    central = (sum(1 for r in records if r['zone'] == 'inside' and not r['near_goal'])/n) if n else 0.0
    plain = outlier*(1-ref_like)  # in neither team kit, and not the referee kit
    deep = clamp((extreme-.3)/.4)  # consistently the deepest person (now and then is normal for anyone)
    # Two routes each: the soccer detector's class votes, or position plus appearance.
    consistent = clamp((outlier-.5)/.4)  # how consistently the kit matches neither team (needs > 50%)
    # A kit that matches neither team is not a referee by itself (a team kit in shadow does that too):
    # the detector must agree at least sometimes, unless the kit matches the learned referee kit.
    backed = max(ref_like, clamp(shares['referee']/.3))
    ref = max(.85*shares['referee']+.15*max(ref_like, outlier),
              .55*ref_like+(.35*(1-near_goal)+.25*central)*consistent*backed*(1-.6*deep))
    if boundary >= .5 and outlier >= .6 and shares['goalkeeper'] < .5:
        ref = max(ref, .55+.3*ref_like)  # an outlier kit running the touchline: an assistant referee
    gk = max(.85*shares['goalkeeper']+.15*max(near_goal, keeper_like, plain),
             .4*near_goal+.3*max(plain, keeper_like)+.3*extreme+.1*isolated)
    if team is not None:
        # The kit matches a team: a referee or keeper decision needs a clear detector majority. A loose
        # match (near the edge of that team's colour spread, e.g. a pink referee next to a garnet kit)
        # holds the detector back less.
        mine = [v.dist_a if team == 'A' else v.dist_b for v in kits if v.team == team]
        limit = min(.5, 2.5*model.team_spread(team)+.1)
        loose = clamp((float(np.mean(mine))-.7*limit)/(.3*limit)) if mine else 0.0
        ref *= max(.35+.65*clamp((shares['referee']-.5)/.35), loose*clamp(shares['referee']/.5))
        gk *= .5+.5*clamp((shares['goalkeeper']-.4)/.3)
    strong_gk = shares['goalkeeper'] >= .5 or near_goal >= .6 or keeper_like >= .5 or (deep >= .75 and plain >= .6)
    keeper_teams = Counter(v.keeper_team for v in kits if v.keeper_like and v.keeper_team)
    r3 = lambda v: round(float(v), 3)
    scores = {'teamA': r3(np.mean([_similarity(v.dist_a) for v in kits])) if nk else None,
              'teamB': r3(np.mean([_similarity(v.dist_b) for v in kits])) if nk else None,
              'refereeKit': r3(ref_like), 'neitherTeam': r3(outlier), 'keeperKit': r3(keeper_like),
              'detectorReferee': r3(shares['referee']), 'detectorGoalkeeper': r3(shares['goalkeeper']),
              'nearGoal': r3(near_goal), 'deepest': r3(extreme), 'isolated': r3(isolated), 'touchline': r3(boundary),
              'teamFit': None if team is None else r3(1-loose)}
    common = {'referee_confidence': r3(clamp(ref)*support), 'goalkeeper_confidence': r3(clamp(gk)*support), 'scores': scores}

    def candidate(why):
        return RoleDecision('CANDIDATE', 'unknown', None, r3(team_share*support), r3(max(shares.values())*support if e.detector else 0), why, **common)
    if e.hits < minimum or len(e.detector) < minimum:
        return candidate(f'{e.hits}/{minimum} observations')
    if ref >= ROLE_THRESHOLD and ref >= gk:
        return RoleDecision('REFEREE', 'referee', None, 0.0, r3(clamp(ref)*support), **common)
    if gk >= ROLE_THRESHOLD and strong_gk:
        keeper_team = keeper_teams.most_common(1)[0][0] if keeper_teams and keeper_teams.most_common(1)[0][1] >= .6*sum(keeper_teams.values()) else None
        return RoleDecision(role_label('goalkeeper', keeper_team), 'goalkeeper', keeper_team, r3(.8 if keeper_team else 0), r3(clamp(gk)*support), **common)
    if not model.ready:
        return candidate('waiting for two team kits')
    if team and shares['player'] >= .4:
        return RoleDecision(role_label('player', team), 'player', team, r3(team_share*support), r3(shares['player']*support), **common)
    if outlier >= .6:
        return candidate(f'kit matches neither team (referee {clamp(ref):.2f}, goalkeeper {clamp(gk):.2f})')
    na, nb = share(lambda v: v.team == 'A'), share(lambda v: v.team == 'B')
    return candidate(f'kit votes A {na:.0%} / B {nb:.0%} / neither {outlier:.0%}')
