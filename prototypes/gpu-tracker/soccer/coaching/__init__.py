"""PitchIQ's post-match coaching layer: deterministic tactical observations, evidence, diagrams and a coaching report
built on top of the finalized tracking result and the existing analytics (soccer/analytics.py, soccer/team_shape.py).
It never changes the tracking, the result or the analytics; see the README section "Coaching report".

The pipeline of one report:
    result.json -> series.build_series (per-sample measurements and their eligibility, both teams)
      -> detectors.detect (Observation records against the professional reference, reference.py / reference.json)
      -> evidence.attach (the moments to watch: frame, times, diagram, video overlay)
      -> report.build_report (words, drills, sections; language.py, drills.py)
build_observations runs the first three, build_report the last, build_report_for all of them for the server
(store.build_report_for). The submodules are imported when first used, so importing the package (language.py,
drills.py and store.py do) stays light.
"""
import json
import math
from pathlib import Path

from .episodes import ENGINE_VERSION

FOOTAGE_KINDS = ('broadcast', 'amateur', 'wide')


def _num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def run_meta(cd, reference=None, t0=None, t1=None):
    """What the report needs to know about the run and the analysed range (report.build_report's meta, without the
    server's fingerprint, video and run names): the range (absolute seconds, clamped to the segment), the segment start,
    the pitch and whether metres are estimates, the footage kind (broadcast; amateur: small players; wide: a few
    players per team in view, as on the Veo clip), the registration shares of the range's frames, the corrections
    applied, the coverage and the run-level limitations (reason codes of language.REASONS)."""
    from . import detectors
    result = cd.result or {}
    frames = result.get('frames') or []
    first = float(frames[0]['time']) if frames else float(cd.start)
    last = float(frames[-1]['time']) if frames else float(cd.end)
    lo = first if t0 is None else min(max(float(t0), first), last)
    hi = last if t1 is None else max(min(float(t1), last), first)
    lo, hi = min(lo, hi), max(lo, hi)
    run = detectors.run_facts(cd)
    footage = 'broadcast' if not run['smallScene'] else ('wide' if run['fewPlayers'] else 'amateur')
    high = medium = low = 0
    for f in frames:
        t = float(f.get('time', 0.0))
        if t < lo-1e-6 or t > hi+1e-6:
            continue
        cal = f.get('calibration') if isinstance(f.get('calibration'), dict) else None
        c = cal.get('confidence') if cal else None
        if cal is None:
            continue
        if f.get('calibrated') is not False and _num(c) and c >= .7:
            high += 1
        elif f.get('calibrated') is not False and _num(c) and c >= .35:
            medium += 1
        else:
            low += 1
    n = high+medium+low
    registration = {'high': round(high/n, 3), 'medium': round(medium/n, 3), 'low': round(low/n, 3)} if n else None
    applied = [c for c in result.get('corrections') or () if isinstance(c, dict) and c.get('applied', True)]
    limitations = []
    if reference is None:
        limitations.append('no_reference')
    if footage == 'wide':
        limitations.append('few_players')
    if run['smallScene']:
        limitations.append('small_scene')
    if run['keyless']:
        limitations.append('keyless')
    coverage = {'samples': int(sum(1 for t in cd.t if lo-1e-6 <= t <= hi+1e-6)),
                # one text row (the page and the PDF lower-case the keys of a nested group: "a 9.99 - b 9.96")
                'observedOutfieldPlayers': ', '.join(f'Team {team} {v:.1f}' for team, v in sorted(run['observed'].items()))
                                           + ' (mean per sample)'}
    return {'engineVersion': ENGINE_VERSION, 'range': {'t0': round(lo, 3), 't1': round(hi, 3),
                                                       'seconds': round(hi-lo, 2)},
            'segmentStart': round(first, 3), 'segmentEnd': round(last, 3), 'fps': cd.fps,
            'metresEstimated': bool(cd.metres_estimated), 'pitch': dict(cd.pitch), 'footage': footage,
            'smallScene': run['smallScene'], 'keyless': run['keyless'], 'lensBend': run['lens'],
            'registration': registration, 'corrections': len(applied), 'coverage': coverage, 'unavailable': {},
            # the same means as numbers (report.py states them in the data-quality note of a footage with a few players
            # per team in view: "only 2.0 players of Team A and 2.6 of Team B"), not only as the coverage text row
            'observedPlayers': {team: round(float(v), 1) for team, v in sorted(run['observed'].items())},
            'limitations': limitations,
            'reference': detectors.refs.reference_id(reference) if reference else None,
            'video': {'name': (result.get('video') or {}).get('name'), 'key': None}}


def build_observations(result, team=None, t0=None, t1=None, reference=None, samples=None, shape=None, settings=None,
                       series=None, library=None, keep_series=False):
    """The coaching observations and their evidence for one team ('A' or 'B') or both (team None) of a result, over
    [t0, t1] (absolute seconds; the whole segment by default). reference: a reference dict, a path, None
    (soccer/coaching/reference.json; when it does not exist every comparative detector reports "no professional
    reference available") or False (none). samples / shape: the analytics.Samples and observed team_shape.TeamShape of
    this result when the caller has them (the server's caches); settings: the video's team-shape settings for a
    TeamShape built here; series: a series.CoachingData already built. Returns {'engineVersion', 'observations',
    'evidence', 'meta'} (JSON-safe), plus 'series' (the CoachingData) with keep_series."""
    from . import detectors, evidence, series as cs
    cd = series if series is not None else cs.build_series(result, samples=samples, shape=shape, settings=settings)
    ref = detectors.resolve_reference(reference)
    teams = ('A', 'B') if team is None else (team,)
    obs = []
    for tm in teams:
        obs += detectors.detect(cd, tm, ref if ref is not None else False, t0, t1, library=library)
    ev = evidence.attach(obs, cd)
    out = {'engineVersion': ENGINE_VERSION, 'observations': [detectors.public(o) for o in obs], 'evidence': ev,
           'meta': run_meta(cd, ref, t0, t1)}
    if keep_series:
        out['series'] = cd
    return out


def build_report(observations, evidence, meta, *args, **kwargs):
    """report.build_report (soccer/coaching/report.py, imported when used)."""
    from . import report
    return report.build_report(observations, evidence, meta, *args, **kwargs)


def build_report_for(result_path, team, options=None):
    """The coaching report of one team of a saved result, as the server builds it (soccer/coaching/store.py):
    options {'t0', 't1', 'context', 'settings' (team-shape settings), 'run', 'fingerprint', 'video' {key, name, fps},
    'teamName', 'reference', 'cache' {'samples', 'shape'} (the server's Samples and observed TeamShape of these same
    bytes)}."""
    from . import report
    options = dict(options or {})
    result = json.loads(Path(result_path).read_bytes().decode('utf-8'))
    cache = options.get('cache') or {}
    analysis = build_observations(result, team=team, t0=options.get('t0'), t1=options.get('t1'),
                                  reference=options.get('reference'), samples=cache.get('samples'),
                                  shape=cache.get('shape'), settings=options.get('settings'))
    meta = analysis['meta']
    video = dict(meta.get('video') or {})
    video.update({k: v for k, v in (options.get('video') or {}).items() if k in ('name', 'key') and v})
    meta.update(fingerprint=options.get('fingerprint'), run=options.get('run'), video=video,
                teamName=options.get('teamName'))
    context = options.get('context') or report.language.DEFAULT_LEVEL
    return report.build_report(analysis['observations'], analysis['evidence'], meta, context=context, team=team,
                               team_name=options.get('teamName'))


__all__ = ['ENGINE_VERSION', 'FOOTAGE_KINDS', 'build_observations', 'build_report', 'build_report_for', 'run_meta']
