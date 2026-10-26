"""Shared fixtures of the DETECT tests (test_coaching_geom, test_coaching_contextual, test_coaching_assess): EVAL's
synthetic scenes (tools/coaching_synth.py) through the real measurement pipeline (analytics, team_shape, possession-4,
the coaching series, situations.build_timeline, contextual.detect) and the role tier (roles.build_rolebook,
contextual.detect_roles), with stand-in evidence moments until EVIDENCE (wave 4) lands evidence.moments.

The stand-in moment of an opportunity is its contextual.key_sample (the middle of its longest run of samples that fail
under DESIGN Appendix A's reference parameters, else of demonstrated ones): a valid schema 'moment' whose scene holds
the observed players at that sample, so assess.py can cite it and the findings validate. The scenes are built once per
process (about 2 s each) and shared by the three test modules.
"""
import copy
import sys
from pathlib import Path

import numpy as np

import scene  # noqa: F401  (adds the prototype directory to sys.path)

TOOLS = Path(__file__).resolve().parents[1]/'tools'
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))
import coaching_synth as cs  # noqa: E402
from soccer import analytics, possession, team_shape as ts  # noqa: E402
from soccer import coaching  # noqa: E402
from soccer.coaching import contextual as CX, profiles as PR, schema as S, series, situations as SI  # noqa: E402

_PREPARED = {}
_TIERS = {}


def prepared(scene_name, variant=None):
    """(built, settings at level 4, coaching series, possession-4 record) of a scene, cached for the process."""
    key = (scene_name, variant)
    if key not in _PREPARED:
        b = cs.build(scene_name, variant)
        st = cs.settings_for(b, level=4)
        samples = analytics.Samples(b.result)
        shape = ts.TeamShape(b.result, samples=samples)
        cd = series.build_series(b.result, samples=samples, shape=shape)
        _PREPARED[key] = (b, st, cd, possession.compute(b.result))
    return _PREPARED[key]


def timeline(scene_name, variant=None, team='A', coach=True):
    """The situation timeline of a scene (the scene's coach spans when it has them and coach is true)."""
    b, st, cd, poss = prepared(scene_name, variant)
    meas = S.tier_settings(st['teams'][team])['measurement']
    return SI.build_timeline(cd, poss, team, meas, coach_spans=b.coach_spans if coach else None)


def _scene_players(tl, s):
    out = {}
    for ids, x, y, ok, team in ((tl.ids, tl.x, tl.y, tl.posOk, tl.team),
                                (tl.oppIds, tl.oppx, tl.oppy, tl.oppPosOk, tl.other)):
        for j, pid in enumerate(ids):
            if ok[s, j] and np.isfinite(x[s, j]) and np.isfinite(y[s, j]):
                out[pid] = {'team': team, 'x': float(np.clip(x[s, j], -.1, 1.1)),
                            'y': float(np.clip(y[s, j], -.1, 1.1)), 'kind': 'O', 'sigma': None, 'goalkeeper': False,
                            'box': None}
    return out


def stand_in_moments(cd, tl, opportunities):
    """The 'moments' stage stand-in: one moment per opportunity at its key sample."""
    fps = float(tl.fps or 25.0)
    out = []
    for o in opportunities:
        k = CX.key_sample(o, tl.scale)
        t = o['relations']['t'][k]
        s = int(np.argmin(np.abs(tl.t-t)))
        frame = int(tl.frame[s])
        players = _scene_players(tl, s)
        ids = [p for p in o['actors']['primary']+o['actors']['supporting'] if p in players]
        out.append({'id': S.record_id('ev', o['id'], k), 'opportunity': o['id'], 'detector': o['detector'],
                    'team': o['team'], 'sample': k, 't': float(tl.t[s]), 'frame': frame, 'seek': S.seek_of(frame, fps),
                    'fps': fps, 'tier': 'world', 'calibration': None, 'players': ids, 'flagged': [],
                    'scene': {'t': float(tl.t[s]), 'frame': frame, 'players': players, 'ball': None,
                              'ballReliable': False},
                    'before': None, 'image': None})
    return {'moments': out, 'frameTiers': []}


def stand_in_role_moments(obs, opportunities):
    """The 'role_moments' stage stand-in (the role tier reads only the obs file: an empty scene)."""
    fps = float(obs['segment']['fps'] or 25.0)
    out = []
    for o in opportunities:
        k = CX.key_sample(o, obs['timeline']['scale'])
        t = o['relations']['t'][k]
        frame = int(o['frame0']+round((t-o['t0'])*fps))
        out.append({'id': S.record_id('ev', o['id'], k), 'opportunity': o['id'], 'detector': o['detector'],
                    'team': o['team'], 'sample': k, 't': float(t), 'frame': frame, 'seek': S.seek_of(frame, fps),
                    'fps': fps, 'tier': 'world', 'calibration': None, 'players': [], 'flagged': [],
                    'scene': {'t': float(t), 'frame': frame, 'players': {}, 'ball': None, 'ballReliable': False},
                    'before': None, 'image': None})
    return {'moments': out}


def tiers(scene_name, variant=None, team='A', coach=True, settings=None):
    """(obs, roles) of a scene through coaching.build_observations / build_roles (schema-validated by them) with the
    stand-in moments and the scene's coach spans; cached per (scene, variant, team, coach) when settings is None."""
    key = (scene_name, variant, team, coach)
    if settings is None and key in _TIERS:
        return copy.deepcopy(_TIERS[key])
    b, st, cd, poss = prepared(scene_name, variant)
    st = settings or st
    stages = {'moments': stand_in_moments, 'role_moments': stand_in_role_moments}
    if coach and b.coach_spans is not None:
        spans = b.coach_spans
        stages['situations'] = lambda cd_, p_, tm, meas: SI.build_timeline(cd_, p_, tm, meas, coach_spans=spans)
    obs = coaching.build_observations(b.result, team, coaching=st, series=cd, possession=poss, result_sha='0'*64,
                                      stages=stages)
    roles = coaching.build_roles(obs, st, team, stages=stages)
    if settings is None:
        _TIERS[key] = copy.deepcopy((obs, roles))
    return obs, roles


def profile(scene_name, variant=None, level=4, team='A', style=None, age=None):
    """The resolved profile of a scene's settings at a level (age: the level's typical age unless given)."""
    b, st, _, _ = prepared(scene_name, variant)
    block = copy.deepcopy(st['teams'][team])
    block['playingLevel'] = level
    block['ageGroup'] = age or PR.default_age(level)
    if style:
        block['style'] = style
    return PR.resolve(block, team=team)


def findings_of(out, detector, verdicts=None):
    return [f for f in out['findings']
            if f['observation'] == detector and (verdicts is None or f['verdict'] in verdicts)]


def status_of(out, detector):
    return next(d for d in out['detectors'] if d['detector'] == detector)
