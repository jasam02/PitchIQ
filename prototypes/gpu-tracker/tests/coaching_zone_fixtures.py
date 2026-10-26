"""Shared fixtures of the ZONES tests (test_coaching_alternatives): EVAL's synthetic scenes through the real
measurement and role tiers (coaching_detect_fixtures), with evidence moments that carry what the suggested-zone engine
reads, built the way DESIGN 5.1 and schema 'moment' describe them, until EVIDENCE (wave 4) lands evidence.moments /
moments_from_obs:
  scene   every player of both teams observed at the moment's sample (kind 'O', posOk), the keepers (goalkeeper True),
          the ball when reliable;
  before  the same REACH_H (2.0 s) earlier, None when that sample does not exist.
The moment of an opportunity is its contextual.key_sample, with the stand-in's id (record_id('ev', opp id, sample)), so
the obs / roles files built by coaching_detect_fixtures keep validating after the swap. role_tier_scene() builds the
scene a role-tier moment can have (the obs file only: own outfield players, the channel opponents DETECT stores on
11v11, the ball when stored), to test the engine on the poorer context too.
"""
import copy

import numpy as np

import coaching_detect_fixtures as F
from soccer.coaching import assess as AS, contextual as CX, schema as S

REACH_H = 2.0


def _entry(team, x, y, sigma, keeper=False):
    return {'team': team, 'x': float(np.clip(x, -.1, 1.1)), 'y': float(np.clip(y, -.1, 1.1)), 'kind': 'O',
            'sigma': None if sigma is None or not np.isfinite(sigma) else round(float(sigma), 3),
            'goalkeeper': keeper, 'box': None}


def scene_at(tl, s, keepers):
    """The schema Scene of a SituationTimeline at sample s (keepers: {team: keeper id})."""
    players = {}
    for ids, x, y, ok, sig, team in ((tl.ids, tl.x, tl.y, tl.posOk, tl.sigma, tl.team),
                                     (tl.oppIds, tl.oppx, tl.oppy, tl.oppPosOk, tl.oppSigma, tl.other)):
        for j, pid in enumerate(ids):
            if ok[s, j] and np.isfinite(x[s, j]) and np.isfinite(y[s, j]):
                players[pid] = _entry(team, x[s, j], y[s, j], sig[s, j])
    for gk, team in ((tl.gk, tl.team), (tl.oppGk, tl.other)):
        pid = keepers.get(team)
        if pid and np.all(np.isfinite(gk[s])):
            players[pid] = _entry(team, gk[s][0], gk[s][1], None, keeper=True)
    ball = None
    if tl.ballReliable[s] and np.isfinite(tl.ballX[s]) and np.isfinite(tl.ballY[s]):
        ball = [round(float(tl.ballX[s]), 5), round(float(tl.ballY[s]), 5)]
    return {'t': float(tl.t[s]), 'frame': int(tl.frame[s]), 'players': players, 'ball': ball,
            'ballReliable': ball is not None}


def keepers_of(obs):
    return {v['team']: pid for pid, v in sorted((obs['timeline'].get('players') or {}).items()) if v['goalkeeper']}


def moment_of(tl, opp, keepers, k=None, tier='world', sigma=None):
    """The moment of an opportunity at sample k (default its key sample) with full scenes now and REACH_H earlier."""
    k = CX.key_sample(opp, tl.scale) if k is None else k
    t = opp['relations']['t'][k]
    s = int(np.argmin(np.abs(np.asarray(tl.t)-t)))
    fps = float(tl.fps or 25.0)
    frame = int(tl.frame[s])
    scene = scene_at(tl, s, keepers)
    b = s-int(round(REACH_H/float(tl.step)))
    before = scene_at(tl, b, keepers) if b >= 0 else None
    if sigma is not None:
        for sc in (scene, before):
            for e in (sc or {}).get('players', {}).values():
                e['sigma'] = sigma
    ids = [p for p in opp['actors']['primary']+opp['actors']['supporting'] if p in scene['players']]
    return {'id': S.record_id('ev', opp['id'], k), 'opportunity': opp['id'], 'detector': opp['detector'],
            'team': opp['team'], 'sample': k, 't': float(tl.t[s]), 'frame': frame, 'seek': S.seek_of(frame, fps),
            'fps': fps, 'tier': tier, 'calibration': None, 'players': ids, 'flagged': [], 'scene': scene,
            'before': before, 'image': None}


def with_rich_moments(obs, roles, tl, tier='world', sigma=None):
    """Copies of (obs, roles) whose moments are rebuilt with full scenes (same ids as the stand-ins)."""
    obs, roles = copy.deepcopy(obs), copy.deepcopy(roles)
    keepers = keepers_of(obs)
    obs['moments'] = [moment_of(tl, o, keepers, tier=tier, sigma=sigma) for o in obs['opportunities']]
    roles['moments'] = [moment_of(tl, o, keepers, tier=tier, sigma=sigma) for o in roles['opportunities']]
    S.check('obs', obs)
    S.check('roles', roles)
    return obs, roles


def _stored_scene(geo, i, side):
    s = geo['samples']
    L, W = geo['scale']['lengthM'], geo['scale']['widthM']

    def norm(X, Y):
        return (X/L, Y/W) if side == 'left' else (1.0-X/L, 1.0-Y/W)
    players = {}
    for j, pid in enumerate(geo['ids']):
        X, Y = s['X'][j][i], s['Y'][j][i]
        if X is not None and Y is not None and s['ok'][i][j] == '1':
            x, y = norm(X, Y)
            players[pid] = _entry(geo['team'], x, y, None)
    other = 'B' if geo['team'] == 'A' else 'A'
    for key in ('oppL', 'oppR'):
        for pid, X, Y in s[key][i]:
            x, y = norm(X, Y)
            players[pid] = _entry(other, x, y, None)
    ball = None
    if s['ballX'][i] is not None and s['ballY'][i] is not None:
        bx, by = norm(s['ballX'][i], s['ballY'][i])
        ball = [round(bx, 5), round(by, 5)]
    return {'t': float(s['t'][i]), 'frame': int(s['frame'][i]), 'players': players, 'ball': ball,
            'ballReliable': ball is not None}


def role_tier_scene(obs, moment):
    """A copy of a role-tier moment whose scene / before come from the obs file only (DETECT's stored geometry)."""
    geo = obs['roleInputs']['detect']
    ts = np.asarray(geo['samples']['t'], float)
    m = copy.deepcopy(moment)
    i = int(np.argmin(np.abs(ts-m['t'])))
    side = geo['samples']['side'][i]
    m['scene'] = _stored_scene(geo, i, side)
    m['scene']['t'], m['scene']['frame'] = m['t'], m['frame']
    b = int(np.argmin(np.abs(ts-(m['t']-REACH_H))))
    m['before'] = _stored_scene(geo, b, side) if abs(ts[b]-(m['t']-REACH_H)) < .11 else None
    m['players'] = [p for p in m['players'] if p in m['scene']['players']]
    return m


def setup(scene_name, variant=None, team='A', tier='world', sigma=None):
    """(obs, roles, timeline) of a scene with rich moments (cached tiers from coaching_detect_fixtures)."""
    obs, roles = F.tiers(scene_name, variant, team=team)
    tl = F.timeline(scene_name, variant, team=team)
    obs, roles = with_rich_moments(obs, roles, tl, tier=tier, sigma=sigma)
    return obs, roles, tl


def findings(obs, roles, profile, verdicts=('work_on', 'try_this', 'review')):
    out = AS.assess(obs, roles, profile)
    return [f for f in out['findings'] if f['verdict'] in verdicts]
