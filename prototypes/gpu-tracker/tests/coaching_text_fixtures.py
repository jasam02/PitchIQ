"""Shared fixtures of the TEXT tests (test_coaching_explain, test_coaching_report, test_coaching_language): EVAL's
synthetic scenes (tools/coaching_synth.py) through the real measurement and role tiers WITH the real evidence moments
(evidence.moments / moments_from_obs, the pixel anchors merged into obs.roleInputs as EVIDENCE asks the orchestrator
to), then coaching.interpret with every real interpretation stage (profiles.resolve, assess, alternatives.propose,
evidence.annotate, explain, drills.select, report.build_report_v2). Built once per process and shared; callers that
change a record copy it first.

Why real stages and not hand-made findings: the wave-1 explanations were developed on findings shaped after DESIGN
6.1, and the first run on real findings found false and refused sentences the hand-made ones never produced (the
attacking-shape relation that failed, "0 of our players", "on average" for a value at one moment, a describe saying
"it stays a review", the level-2 team-level wide-defender check naming "our left defender").
"""
import copy

import coaching_detect_fixtures as FX
from soccer import coaching
from soccer.coaching import evidence as EV, profiles as PR, schema as S, situations as SI

_TIERS = {}
_REPORTS = {}


def tiers(scene_name, variant=None, team='A'):
    """(built, settings, obs, roles) of a scene with the real moments stage and the scene's coach spans."""
    key = (scene_name, variant, team)
    if key not in _TIERS:
        b, st, cd, poss = FX.prepared(scene_name, variant)
        held = {}

        def moments(cd_, tl, opps):
            out = EV.moments(cd_, tl, opps)
            held['ri'] = out.get('roleInputs') or {}
            return out
        stages = {'moments': moments}
        if b.coach_spans is not None:
            spans = b.coach_spans
            stages['situations'] = lambda cd_, p_, tm, meas: SI.build_timeline(cd_, p_, tm, meas, coach_spans=spans)
        obs = coaching.build_observations(b.result, team, coaching=st, series=cd, possession=poss,
                                          result_sha='0'*64, stages=stages)
        obs['roleInputs'].update(held['ri'])      # EVIDENCE's anchors (its request to LEAD, INTERFACES.md)
        roles = coaching.build_roles(obs, st, team)
        _TIERS[key] = (b, st, obs, roles)
    return _TIERS[key]


def settings_at(settings, team, level, age=None, style=None):
    """The scene's coaching settings with the team's level (age: the level's typical age unless given)."""
    st = copy.deepcopy(settings)
    block = st['teams'][team]
    block['playingLevel'] = level
    block['ageGroup'] = age or PR.default_age(level)
    if style:
        block['style'] = style
    return st


def report(scene_name, variant=None, level=4, team='A', age=None, style=None, t0=None, t1=None):
    """coaching.interpret's report of a scene at a level (schema-validated and cross-checked by interpret)."""
    key = (scene_name, variant, level, team, age, style, t0, t1)
    if key not in _REPORTS:
        b, st, obs, roles = tiers(scene_name, variant, team)
        _REPORTS[key] = coaching.interpret(obs, roles, settings_at(st, team, level, age, style), team, t0=t0, t1=t1,
                                           meta={'run': 'synthetic', 'teamName': None})
    return copy.deepcopy(_REPORTS[key])


def findings_of(rep):
    return rep['strengths']+rep['needsWork']+rep['reviews']+rep['described']


def texts_of(rep):
    """Every coach-facing text of a report with a path (finding texts, overview, notices, captions, objective,
    drill fields, diagram captions)."""
    out = []
    for f in findings_of(rep):
        for k, v in (f['text'] or {}).items():
            if v:
                out.append((f'{f["id"]}.{k}', v))
    ov = rep['overview']
    if ov['text']:
        out.append(('overview.text', ov['text']))
    for n in ov['notices']:
        out.append((f'notice.{n["code"]}', n['text']))
    for k in ('caveat', 'reason'):
        if ov['possession'].get(k):
            out.append((f'possession.{k}', ov['possession'][k]))
    for e in rep['keyMoments']:
        out.append((f'{e["id"]}.caption', e['caption']))
        if e['annotation']:
            out.append((f'{e["id"]}.annotation', e['annotation']['caption']))
    for d in rep['diagrams']:
        out.append((f'{d["id"]}.caption', d['caption']))
    out.append(('nextObjective', rep['nextObjective']['text']))
    for p in rep['drills']:
        for k in ('trains', 'why', 'easier', 'harder', 'progress'):
            out.append((f'drill.{p["drill"]}.{k}', p['fields'][k]))
    return out


__all__ = ['tiers', 'settings_at', 'report', 'findings_of', 'texts_of', 'S']
