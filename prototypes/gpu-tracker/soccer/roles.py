"""Temporal role classification: PLAYER (team A/B), GOALKEEPER, REFEREE or still undecided.

Every evidence tick (5 per second) a track adds one record: the detector's class (player /
goalkeeper / referee) and score, the kit vote against both teams and against the referee and
goalkeeper kits, where the person is (near a goal, the deepest person on the pitch, isolated, how far
from the nearest pitch edge and which way that edge runs), how fast they move, how far the ball is and
where they sit in each team's shape. Decisions use the last ~10 seconds, never one crop. On top of the
kit vote, the person's whole appearance (learned OSNet embedding plus jersey, shorts, socks and body
layout colours, from several recent crops) is compared with the known referees and with each team's
players, so "looks like the referees" can win over "the shirt colour is near team B's".

Referee routes (the strongest wins; a referee needs >= 0.55):
- detector: the soccer detector's referee votes;
- appearance: the person looks more like the known referees than like either team's players, scaled
  by how referee-like they behave (outside the supposed team's shape, following the ball without
  playing it, moving with neither team);
- touchline (assistant referee): time on or just outside one pitch edge, moving along it, together
  with referee appearance, the referee kit or detector votes. A kit matching neither team is not an
  official by itself on the touchline either (coaches and staff stand there too);
- position + kit: a kit like the referee's, or matching neither team and backed by detector votes or
  referee appearance, while following play away from the goals.
A kit that matches a team holds the detector and position routes back unless the detector's votes
are a clear majority or the match is loose (near the edge of that team's colour spread); it never
holds back the appearance and touchline routes, which already weigh the person against that team.
When that team already has its full count of identities, another "team player" is doubted and the
referee threshold drops. Being consistently the deepest person on the pitch is a goalkeeper's
position, not a referee's.
- GOALKEEPER: detector goalkeeper votes; or time spent near a goal in a kit unlike the outfield
  kits (or like a known goalkeeper kit) while being the deepest or most isolated person. Needs at
  least one strong cue (detector, goal proximity, a known keeper kit, or a kit matching nobody while
  consistently being the deepest person). This is the track-level role: a GOALKEEPER CANDIDATE. A
  goalkeeper *identity* (GK-A / GK-B) is confirmed only by keeper_confirmation(): sustained evidence
  with a spatial part (goal-area residence, or consistently the deepest person, or sustained
  detector votes in a distinct kit together with some position evidence) and nothing that says
  touchline official or spectator (feet on or outside the pitch, the touchline corridor, referee
  evidence). The detector calling someone a goalkeeper is never enough on its own.
- PLAYER: >= 70% of kit votes for one team, and the person does not look more like the referees.
- Otherwise CANDIDATE (and UNKNOWN after a long time), never forced into Team A or Team B.
"""
import math
from collections import Counter, deque
from dataclasses import dataclass, field

import numpy as np

from .geometry import clamp
from .teams import role_label

WINDOW = 50              # evidence ticks kept (10 s at 5 Hz)
ROLE_THRESHOLD = .55
GK_CONFIRM_RECORDS = 15  # evidence ticks (3 s) of goalkeeper evidence before a goalkeeper identity can be confirmed
GK_CONFIRM = .7          # temporal goalkeeper confidence needed to confirm
GK_OFF_PITCH = .2        # share of observations on or outside the touchline that rules a candidate out
GK_TOUCHLINE = .3        # share of observations in the touchline corridor that rules a candidate out
CROWDED_THRESHOLD = .45  # referee threshold when the kit's team already has its full count of identities
CORRIDOR = (-1.0, 4.0)   # metres from the nearest pitch edge where a touchline official stands: on the line to 4 m out
PLAY_RADIUS = 25.0       # metres from the ball: following play
TOUCH_RADIUS = 1.5       # metres from the ball: playing it
OFFICIAL_NAMES = {'assistant': 'ASSISTANT REFEREE', 'centre': 'CENTRE REFEREE', '': 'REFEREE'}


@dataclass
class RoleEvidence:
    hits: int = 0
    records: deque = field(default_factory=lambda: deque(maxlen=WINDOW))

    def add(self, cls, score, vote, cues=None):
        """One evidence tick. cues: near_goal, goal_side, extreme, isolated, zone, edge ({'side', 'metres',
        'dir'}: the nearest pitch edge) and v (velocity in metres per second). The ball distance and the
        team-shape cues are written onto the returned record afterwards."""
        cues = cues or {}
        self.hits += 1
        self.records.append({'det': cls if cls in ('player', 'goalkeeper', 'referee') else None, 'score': max(.05, float(score)),
                             'vote': vote if vote is not None and vote.valid else None,
                             'near_goal': bool(cues.get('near_goal')), 'goal_side': cues.get('goal_side'),
                             'extreme': bool(cues.get('extreme')), 'isolated': bool(cues.get('isolated')),
                             'zone': cues.get('zone', 'inside'), 'edge': cues.get('edge'), 'v': cues.get('v'),
                             'ball_m': None, 'formation': None, 'move': None})
        return self.records[-1]

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
    official: str = ''          # referees: 'assistant' (touchline), 'centre' or '' (unknown kind)
    why: str = ''               # one line: the evidence behind the decision


def _similarity(distance):
    return clamp(1-distance/.6)


def _touchline(records, boundary):
    """An assistant referee's pattern: time on or just outside one pitch edge and movement along it.
    Without edge measurements (no pitch model on those frames) the touchline zone share stands in."""
    n = len(records)
    edges = [(r['edge'], r['v']) for r in records if r.get('edge')]
    if not edges:
        return {'score': .7*boundary, 'time': boundary if n else None, 'parallel': None, 'side': None}
    corridor = [(e, v) for e, v in edges if CORRIDOR[0] <= e['metres'] <= CORRIDOR[1]]
    time = len(corridor)/n
    sides = Counter(e['side'] for e, _ in corridor)
    side, same = (sides.most_common(1)[0][0], sides.most_common(1)[0][1]/len(corridor)) if corridor else (None, 0.0)
    along = across = 0.0
    for e, v in corridor:
        if not v or not e.get('dir'):
            continue
        speed = math.hypot(v[0], v[1])
        if speed < .3:
            continue
        a = abs(v[0]*e['dir'][0]+v[1]*e['dir'][1])
        along += a
        across += math.sqrt(max(0.0, speed*speed-a*a))
    parallel = along/(along+across) if along+across > 0 else None
    return {'score': time*same*(.4+.6*(.5 if parallel is None else parallel)), 'time': time, 'parallel': parallel, 'side': side}


def _central(records, team):
    """A central referee's pattern over the window: following play (within PLAY_RADIUS of the ball) without
    playing it, sitting outside the supposed team's shape, and not moving with that team. Each part needs
    enough records; the score is None when nothing is known."""
    balls = [r['ball_m'] for r in records if r.get('ball_m') is not None]
    ball = None
    if len(balls) >= 10:
        near = sum(1 for m in balls if m <= PLAY_RADIUS)/len(balls)
        touches = sum(1 for m in balls if m <= TOUCH_RADIUS)/len(balls)
        ball = near*(1-clamp(touches/.1))
    formation = moves = None
    if team:
        zs = [r['formation'][team] for r in records if r.get('formation') and r['formation'].get(team) is not None]
        if len(zs) >= 10:
            formation = sum(1 for z in zs if z <= 1.3)/len(zs)
        cs = [r['move'][team] for r in records if r.get('move') and r['move'].get(team) is not None]
        if len(cs) >= 5:
            moves = clamp(sum(cs)/len(cs))
    parts = [p for p in ((None if formation is None else 1-formation, .5), (ball, .3), (None if moves is None else 1-moves, .2)) if p[0] is not None]
    score = sum(v*w for v, w in parts)/sum(w for _, w in parts) if parts else None
    return {'score': score, 'ball': ball, 'formation': formation, 'moves': moves}


def keeper_confirmation(records, scores, goalkeeper_confidence, referee_confidence):
    """Is the goalkeeper evidence sustained, spatial and free of touchline-official signs, so that a
    goalkeeper identity may be confirmed? A goalkeeper is a role with a place: their own goal area. So
    the detector's votes alone never confirm; feet on or outside the pitch, the touchline corridor or
    referee evidence veto. Returns a dict with 'confirmed', 'state' ('CONFIRMED' | 'CANDIDATE'), the
    'route' that confirms, a one-line 'reason', and the shares behind it (for the debug output)."""
    n = len(records)
    zones = [r.get('zone', 'inside') for r in records]
    inside = sum(1 for z in zones if z == 'inside')/n if n else 0.0
    outside = sum(1 for z in zones if z == 'outside')/n if n else 0.0
    off = sum(1 for z in zones if z in ('boundary', 'outside'))/n if n else 0.0
    touch = scores.get('touchlineTime') or 0.0
    near_goal, deep, isolated = scores['nearGoal'], scores['deepest'], scores['isolated']
    det, kit = scores['detectorGoalkeeper'], max(scores['keeperKit'], scores['neitherTeam'])
    sides = Counter(r['goal_side'] for r in records if r.get('goal_side'))
    side = sides.most_common(1)[0][0] if sides else None
    out = {'confirmed': False, 'state': 'CANDIDATE', 'route': '', 'reason': '', 'inside': round(inside, 3), 'offPitch': round(off, 3),
           'outside': round(outside, 3), 'touchline': round(touch, 3), 'goalSide': side, 'records': n, 'vetoed': ''}
    # Vetoes: a person on the sideline is an official, a substitute or staff, whatever the detector says.
    # While vetoed, a track is not even a goalkeeper candidate: it is neither promoted as one nor
    # matched to a missing goalkeeper identity.
    if off > GK_OFF_PITCH:
        out['reason'] = out['vetoed'] = f'feet on or outside the touchline in {off:.0%} of observations'
        return out
    if touch >= GK_TOUCHLINE:
        out['reason'] = out['vetoed'] = f'patrols the touchline corridor ({touch:.0%} of observations)'
        return out
    if referee_confidence >= .5:
        out['reason'] = out['vetoed'] = f'referee evidence {referee_confidence:.2f}'
        return out
    # Positive routes, each with a spatial part.
    if near_goal >= .5:
        route = f'goal-area residence {near_goal:.0%}'
    elif deep >= .6 and (isolated >= .5 or det >= .5 or kit >= .6):
        route = f'the deepest person {deep:.0%} of the time' + (f', alone {isolated:.0%}' if isolated >= .5 else '')
    elif det >= .75 and kit >= .6 and (deep >= .3 or near_goal >= .2):
        route = f'detector {det:.0%} in a distinct kit, deepest {deep:.0%}, goal area {near_goal:.0%}'
    else:
        out['reason'] = f'no goal-area or deepest-person evidence yet (goal area {near_goal:.0%}, deepest {deep:.0%}, detector {det:.0%})'
        return out
    if n < GK_CONFIRM_RECORDS:
        out['reason'] = f'{route}; {n}/{GK_CONFIRM_RECORDS} observations'
        return out
    if goalkeeper_confidence < GK_CONFIRM:
        out['reason'] = f'{route}; temporal confidence {goalkeeper_confidence:.2f} < {GK_CONFIRM}'
        return out
    out.update(confirmed=True, state='CONFIRMED', route=route, reason=route)
    return out


def _why(s):
    """One line of the evidence behind a decision, for labels and logs."""
    parts = []
    if s.get('refereeAppearance') is not None or s.get('teamAAppearance') is not None:
        f = lambda key: '-' if s.get(key) is None else f'{s[key]:.2f}'
        parts.append(f"looks ref {f('refereeAppearance')} A {f('teamAAppearance')} B {f('teamBAppearance')}")
    if s['detectorReferee'] or s['detectorGoalkeeper']:
        parts.append(f"det ref {s['detectorReferee']:.0%} gk {s['detectorGoalkeeper']:.0%}")
    if s['neitherTeam']:
        parts.append(f"neither kit {s['neitherTeam']:.0%}")
    if s.get('touchlineTime'):
        parts.append(f"touchline {s['touchlineTime']:.0%}" + (f" along {s['parallelMovement']:.2f}" if s.get('parallelMovement') is not None else ''))
    if s.get('formationConsistency') is not None:
        parts.append(f"shape {s['formationConsistency']:.2f}")
    if s.get('ballFollowing') is not None:
        parts.append(f"ball {s['ballFollowing']:.2f}")
    if s.get('teamCrowded'):
        parts.append('team full')
    k = s.get('keeper')
    if k is not None and s['final'].startswith('GOALKEEPER'):
        parts.append(f"gk {'confirmed' if k['confirmed'] else 'candidate'}: inside {k['inside']:.0%} goal {s['nearGoal']:.2f} deepest {s['deepest']:.0%}")
    parts.append(s['final'])
    return ' | '.join(parts)


def decide_role(e, model, min_votes=5, looks=None, crowded=None):
    """Temporal decision; CANDIDATE until the evidence is strong enough. looks: whole-appearance
    similarity to the known referees and to each team's players ({'referee', 'A', 'B', 'probes'}; None
    where unknown). crowded: {team: True} where that team already has its full count of identities."""
    records = list(e.records)
    n = len(records)
    minimum = max(3, min_votes)
    support = min(1.0, .75+.25*(e.hits-minimum)/minimum) if e.hits >= minimum else .75*e.hits/minimum
    shares = e.detector_shares()
    det_ref = shares['referee']
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
    boundary = (sum(1 for r in records if r['zone'] in ('boundary', 'outside'))/n) if n else 0.0
    central_zone = (sum(1 for r in records if r['zone'] == 'inside' and not r['near_goal'])/n) if n else 0.0
    plain = outlier*(1-ref_like)  # in neither team kit, and not the referee kit
    deep = clamp((extreme-.3)/.4)  # consistently the deepest person (now and then is normal for anyone)
    consistent = clamp((outlier-.5)/.4)  # how consistently the kit matches neither team (needs > 50%)
    # Whole appearance: more like the known referees than like either team's players? Needs a clear
    # similarity to the referees, a lead over the better team, and several crops.
    looks = looks or {}
    s_ref, s_a, s_b = looks.get('referee'), looks.get('A'), looks.get('B')
    team_look = max(v for v in (s_a, s_b) if v is not None) if s_a is not None or s_b is not None else None
    looks_referee = 0.0
    if s_ref is not None:
        looks_referee = clamp((s_ref-.45)/.3)*clamp((s_ref-(.5 if team_look is None else team_look)+.05)/.2)*clamp(looks.get('probes', 3)/3)
    touch = _touchline(records, boundary)
    central = _central(records, team)
    behaviour = .5 if central['score'] is None else central['score']
    loose, hold, ref_threshold = 0.0, 1.0, ROLE_THRESHOLD
    crowded_team = bool(team and crowded and crowded.get(team))
    gk = max(.85*shares['goalkeeper']+.15*max(near_goal, keeper_like, plain),
             .4*near_goal+.3*max(plain, keeper_like)+.3*extreme+.1*isolated)
    if team is not None:
        # The kit matches a team: the detector and position routes need a clear detector majority. A loose
        # match (near the edge of that team's colour spread, e.g. a pink referee next to a garnet kit)
        # holds them back less; a team that already has its full count of identities does not hold them
        # back at all and lowers the bar, because another "team player" is then doubtful.
        mine = [v.dist_a if team == 'A' else v.dist_b for v in kits if v.team == team]
        limit = min(.5, 2.5*model.team_spread(team)+.1)
        loose = clamp((float(np.mean(mine))-.7*limit)/(.3*limit)) if mine else 0.0
        if crowded_team:
            ref_threshold = CROWDED_THRESHOLD
        else:
            hold = max(.35+.65*clamp((det_ref-.5)/.35), loose*clamp(det_ref/.5))
        gk *= .5+.5*clamp((shares['goalkeeper']-.4)/.3)
    # Four referee routes. A kit matching neither team is never enough alone: the detector, the learned
    # referee kit or the known referees' appearance must agree. The appearance and touchline routes are
    # not held back by a team kit: they already weigh the person against that team's players.
    backed = max(ref_like, clamp(det_ref/.3), looks_referee)
    routes = {'detector': hold*(.85*det_ref+.15*max(ref_like, outlier)),
              'appearance': looks_referee*(.8+.2*behaviour),
              'touchline': touch['score']*max(looks_referee, ref_like, hold*clamp(det_ref/.5), .5*consistent),
              'position': hold*(.55*ref_like+(.35*(1-near_goal)+.25*central_zone)*consistent*backed*(1-.6*deep))}
    ref = max(routes.values())
    # Which kind of official: an assistant patrols the touchline, a centre referee works inside the pitch.
    official = 'assistant' if touch['score'] >= .5 else 'centre' if central_zone >= .5 else ''
    strong_gk = shares['goalkeeper'] >= .5 or near_goal >= .6 or keeper_like >= .5 or (deep >= .75 and plain >= .6)
    keeper_teams = Counter(v.keeper_team for v in kits if v.keeper_like and v.keeper_team)
    r3 = lambda v: round(float(v), 3)
    opt = lambda v: None if v is None else r3(v)
    scores = {'teamA': r3(np.mean([_similarity(v.dist_a) for v in kits])) if nk else None,
              'teamB': r3(np.mean([_similarity(v.dist_b) for v in kits])) if nk else None,
              'refereeKit': r3(ref_like), 'neitherTeam': r3(outlier), 'keeperKit': r3(keeper_like),
              'detectorReferee': r3(det_ref), 'detectorGoalkeeper': r3(shares['goalkeeper']),
              'nearGoal': r3(near_goal), 'deepest': r3(extreme), 'isolated': r3(isolated), 'touchline': r3(boundary),
              'teamFit': None if team is None else r3(1-loose),
              'refereeAppearance': opt(s_ref), 'teamAAppearance': opt(s_a), 'teamBAppearance': opt(s_b), 'looksReferee': r3(looks_referee),
              'touchlineBehaviour': r3(touch['score']), 'touchlineTime': opt(touch['time']), 'parallelMovement': opt(touch['parallel']),
              'formationConsistency': opt(central['formation']), 'ballFollowing': opt(central['ball']), 'movesWithTeam': opt(central['moves']),
              'refereeBehaviour': opt(central['score']), 'teamCrowded': crowded_team, 'routes': {k: r3(v) for k, v in routes.items()}}
    common = {'referee_confidence': r3(clamp(ref)*support), 'goalkeeper_confidence': r3(clamp(gk)*support), 'scores': scores}
    scores['keeper'] = keeper_confirmation(records, scores, common['goalkeeper_confidence'], common['referee_confidence'])

    def done(d):
        d.official = official if d.role == 'referee' else ''
        if d.role == 'referee':
            d.scores['final'] = f'{OFFICIAL_NAMES[d.official]} {d.role_confidence:.2f}'
        elif d.label == 'CANDIDATE':
            d.scores['final'] = 'CANDIDATE'
        else:
            d.scores['final'] = f'{d.label.replace("_", " ")} {d.role_confidence:.2f}'
        d.why = _why(d.scores)
        return d

    def candidate(why):
        return done(RoleDecision('CANDIDATE', 'unknown', None, r3(team_share*support), r3(max(shares.values())*support if e.detector else 0), why, **common))
    if e.hits < minimum or len(e.detector) < minimum:
        return candidate(f'{e.hits}/{minimum} observations')
    if ref >= ref_threshold and ref >= gk:
        return done(RoleDecision('REFEREE', 'referee', None, 0.0, r3(clamp(ref)*support), **common))
    if gk >= ROLE_THRESHOLD and strong_gk:
        keeper_team = keeper_teams.most_common(1)[0][0] if keeper_teams and keeper_teams.most_common(1)[0][1] >= .6*sum(keeper_teams.values()) else None
        return done(RoleDecision(role_label('goalkeeper', keeper_team), 'goalkeeper', keeper_team, r3(.8 if keeper_team else 0), r3(clamp(gk)*support), **common))
    if not model.ready:
        return candidate('waiting for two team kits')
    if team and shares['player'] >= .4:
        if looks_referee >= .5:
            return candidate(f'kit votes team {team}, but looks like the known referees ({s_ref:.2f}) more than team {team} ({(team_look or 0):.2f})')
        return done(RoleDecision(role_label('player', team), 'player', team, r3(team_share*support), r3(shares['player']*support), **common))
    if outlier >= .6:
        return candidate(f'kit matches neither team (referee {clamp(ref):.2f}, goalkeeper {clamp(gk):.2f})')
    na, nb = share(lambda v: v.team == 'A'), share(lambda v: v.team == 'B')
    return candidate(f'kit votes A {na:.0%} / B {nb:.0%} / neither {outlier:.0%}')
