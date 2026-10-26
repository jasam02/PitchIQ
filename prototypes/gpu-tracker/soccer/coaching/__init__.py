"""PitchIQ's post-match coaching layer: deterministic tactical observations, evidence, diagrams and a coaching report
built on top of the finalized tracking result and the existing analytics (soccer/analytics.py, soccer/team_shape.py,
soccer/possession.py). It never changes the tracking, the result or the analytics; see the README section "Coaching
report".

Two engines live here while the v2 work lands (DESIGN v2 section 12.2; the v1 callers move in wave 7).

v1 (the committed report; kept working until its callers are replaced):
    result.json -> series.build_series (per-sample measurements and their eligibility, both teams)
      -> detectors.detect (Observation records against the professional reference, reference.py / reference.json)
      -> evidence.attach (the moments to watch: frame, times, diagram, video overlay)
      -> report.build_report (words, drills, sections; language.py, drills.py)
build_observations(result, ...) runs the first three, build_report the last, build_report_for all of them for the
server (store.build_report_for).

v2 (adaptive coaching; DESIGN 1.2) in three cache tiers, each a pure function of its inputs:
    build_observations(result, team, coaching=settings)  MEASUREMENT tier -> obs-<team>-<mfp16>.json
        series + possession-4 -> situations.build_timeline -> contextual.detect -> evidence.moments
        inputs: the result's bytes, the team-shape settings, the team's format and field size; nothing else
    build_roles(obs, settings, team)                     ROLE tier -> roles-<team>-<mfp16>-<rfp16>.json
        roles.build_rolebook -> contextual.detect_roles -> evidence.moments_from_obs
        inputs: the obs file, the coach's confirmed roles and intended formation
    interpret(obs, roles, settings, team, t0, t1)        INTERPRETATION tier -> report-<...>-<ifp16>.json
        profiles.resolve -> assess.assess -> alternatives.propose -> evidence.annotate -> explain.explain
        -> drills.select -> report.build_report_v2
        inputs: the two files, age group, playing level, style, federation, the report range, profiles/drills data
Why three tiers, measured on the scratch run copies (CPU, this machine, Oct 21): one measurement rebuild of the 180 s
broadcast run (110.9 MB result) is json.loads 0.88 s + build_series 1.49 s + possession 0.72 s, about 3.1 s (BC60:
0.28 + 0.40 + 0.24 s). DESIGN's targets for a role change (< 1 s) and a level change (< 300 ms on a 3-min clip) are
out of reach if either re-reads the result, so the role and interpretation tiers read only the obs file: anything they
need (role geometry, frame tiers, the scene at each moment) is stored there by the measurement tier. The split is also
what makes invariant 2 hold by construction: the measurement stages receive only schema.tier_settings(...)
['measurement'] (format and field), so no age, level or style can reach the observations.

The stages are looked up by name in STAGES ('module:function') when first used, so a role can land its module in any
order and a missing one fails with StageUnavailable naming the module and its owner, never at import. Tests pass
stages={name: fn} to run the orchestration with stand-ins. Every output is checked against soccer/coaching/schema.py
unless validate=False: a stage that breaks the contract fails where it ran.

The fingerprints (mfp, rfp, ifp: sha256 of canonical JSON inputs) hash the code each tier runs: the tier's root modules
and every soccer/ module they import (found by reading their import statements, so a helper module added later is
covered without editing a list), plus __init__.py and schema.py; the interpretation tier also hashes its data files.
v1's store.CODE_EXTRA misses possession.py; the measurement tier here includes it (DESIGN 1.2). The code is hashed once
per process, as store.code_sha does: the code this process runs.
Two corrections of fix round 1 (critic S-12): the store's tier plumbing (measuring.py) is a root of the measurement
and role tiers, and the text-only modules (TEXT_ONLY: words, drills, the v1 reference) are not followed below those
two tiers, which never run them (traced; see NOT_FOLLOWED).

The submodules are imported when first used; importing the package costs what episodes.py imports (numpy) plus
schema.py (standard library only).
"""
import ast
import copy
import hashlib
import importlib
import json
import math
from pathlib import Path

from . import schema
from .episodes import ENGINE_VERSION

FOOTAGE_KINDS = ('broadcast', 'amateur', 'wide')
PACKAGE = Path(__file__).resolve().parent
SOCCER = PACKAGE.parent

# ---------- v2 stages: 'module:function', resolved when first used ----------
STAGES = {
    # measurement tier
    'series': 'soccer.coaching.series:build_series',
    'possession': 'soccer.possession:compute',
    'situations': 'soccer.coaching.situations:build_timeline',
    'detect': 'soccer.coaching.contextual:detect',
    'moments': 'soccer.coaching.evidence:moments',
    # role tier
    'rolebook': 'soccer.coaching.roles:build_rolebook',
    'detect_roles': 'soccer.coaching.contextual:detect_roles',
    'role_moments': 'soccer.coaching.evidence:moments_from_obs',
    # interpretation tier
    'profile': 'soccer.coaching.profiles:resolve',
    'assess': 'soccer.coaching.assess:assess',
    'alternatives': 'soccer.coaching.alternatives:propose',
    'annotate': 'soccer.coaching.evidence:annotate',
    'explain': 'soccer.coaching.explain:explain',
    'drills': 'soccer.coaching.drills:select',
    'report': 'soccer.coaching.report:build_report_v2',
}
STAGE_OWNERS = {'series': 'CONTEXT', 'possession': '(frozen engine)', 'situations': 'CONTEXT, wave 2',
                'detect': 'DETECT, wave 3', 'moments': 'EVIDENCE, wave 4', 'rolebook': 'CONTEXT, wave 2',
                'detect_roles': 'DETECT, wave 3', 'role_moments': 'EVIDENCE, wave 4', 'profile': 'PROFILE, wave 1',
                'assess': 'DETECT, wave 3', 'alternatives': 'ZONES, wave 4', 'annotate': 'EVIDENCE, wave 4',
                'explain': 'TEXT, wave 5', 'drills': 'DRILLS, wave 5', 'report': 'TEXT, wave 5'}
ALT_VERDICTS = ('work_on', 'try_this', 'review')   # findings that get a suggested zone or a principle (DESIGN 5.1)

# ---------- v2 fingerprints: the code and data each tier is computed from ----------
TIER_ROOTS = {
    'measurement': ('coaching/series.py', 'coaching/situations.py', 'coaching/contextual.py',
                    'coaching/tactics_geom.py', 'coaching/evidence.py', 'possession.py', 'team_shape.py',
                    'analytics.py', 'coaching/measuring.py'),
    'role': ('coaching/roles.py', 'coaching/contextual.py', 'coaching/tactics_geom.py', 'coaching/evidence.py',
             'coaching/measuring.py'),
    'interpretation': ('coaching/profiles.py', 'coaching/assess.py', 'coaching/alternatives.py',
                       'coaching/tactics_geom.py', 'coaching/evidence.py', 'coaching/explain.py',
                       'coaching/language.py', 'coaching/drills.py', 'coaching/report.py', 'coaching/progress.py'),
}
TIER_ALWAYS = ('coaching/__init__.py', 'coaching/schema.py')   # hashed in every tier
NO_FOLLOW = {'__init__.py', 'coaching/__init__.py'}            # package inits: hashed, their imports not followed
# Modules that make words, drill picks or the v1 professional comparison, never an obs or a roles file: the measurement
# and role roots reach them only through the v1 detectors module (evidence.py imports detectors.py for run_facts and
# its unit table; detectors.py imports reference.py, and drills.py inside its v1 functions, which imports explain.py
# and language.py). Followed, a wording or drill edit made every run's measurement stale ("engine updated", a
# Regenerate of every clip: CODE M5 / critic S-12). Tracing the two tiers (sys.monitoring PY_START over store.measure
# and store.build_role_tier on EVAL's short_segment scene and the BC60, AM and VEO run copies, Oct 23) ran none of their
# functions; test_coaching_store.TierCode re-checks that on every suite run, and that every soccer/ file the two tiers
# run is hashed. A tier that runs one of them lists it as a root (the interpretation tier does).
TEXT_ONLY = frozenset(('coaching/drills.py', 'coaching/explain.py', 'coaching/language.py', 'coaching/reference.py'))
NOT_FOLLOWED = {'measurement': TEXT_ONLY, 'role': TEXT_ONLY, 'interpretation': frozenset()}
TIER_DATA = {'measurement': (), 'role': (),
             'interpretation': ('coaching/profiles.json', 'coaching/sources.json', 'coaching/drills.json',
                                'coaching/progress_rules.json')}
_code = {}
_V1 = object()        # build_observations without coaching=...: the v1 engine


class StageUnavailable(NotImplementedError):
    """A v2 stage's module or function does not exist yet (its role has not landed it)."""


def _stage(name, stages=None):
    if stages and name in stages:
        return stages[name]
    target = STAGES[name]
    module, fn = target.split(':')
    try:
        mod = importlib.import_module(module)
    except ModuleNotFoundError as error:
        if error.name == module:
            raise StageUnavailable(f'coaching v2 stage {name!r} needs {module}.{fn} ({STAGE_OWNERS[name]}), and '
                                   f'{module} does not exist yet') from None
        raise
    f = getattr(mod, fn, None)
    if not callable(f):
        raise StageUnavailable(f'coaching v2 stage {name!r} needs {module}.{fn} ({STAGE_OWNERS[name]}), which is not '
                               'implemented yet')
    return f


def _imports(rel):
    """The soccer/ files (relative paths) one module imports, by reading its import statements (function-level
    imports included)."""
    path = SOCCER/rel
    try:
        tree = ast.parse(path.read_bytes())
    except (OSError, SyntaxError, ValueError):
        return set()
    pkg = rel.split('/')[:-1]
    out = set()

    def resolve(parts):
        for cand in ('/'.join(parts)+'.py' if parts else None, '/'.join(parts+['__init__.py'])):
            if cand and (SOCCER/cand).is_file():
                out.add(cand)
                return

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.level:
                if node.level-1 > len(pkg):
                    continue
                base = pkg[:len(pkg)-(node.level-1)]
                parts = base+(node.module.split('.') if node.module else [])
            elif node.module == 'soccer' or (node.module or '').startswith('soccer.'):
                parts = node.module.split('.')[1:]
            else:
                continue
            resolve(parts)
            for alias in node.names:
                resolve(parts+[alias.name])
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith('soccer.'):
                    resolve(alias.name.split('.')[1:])
    return out


def code_files(tier):
    """The soccer/ files (relative paths, sorted) a tier's fingerprint hashes: its roots, every soccer/ module they
    import transitively except the tier's NOT_FOLLOWED modules (and what only they import), and TIER_ALWAYS. Roots that
    do not exist yet are listed too (hashed as missing)."""
    roots = TIER_ROOTS[tier]
    skip = NOT_FOLLOWED.get(tier, frozenset())-set(roots)
    todo = list(roots)
    seen = set(TIER_ALWAYS)
    while todo:
        rel = todo.pop()
        if rel in seen:
            continue
        seen.add(rel)
        if rel not in NO_FOLLOW:
            todo.extend(_imports(rel)-seen-skip)
    return sorted(seen)


def _file_digest(digest, rel):
    p = SOCCER/rel
    try:
        data = p.read_bytes().replace(b'\r\n', b'\n')
    except OSError:
        data = None
    digest.update(rel.encode('utf-8')+b'\0'+(data if data is not None else b'<missing>')+b'\0')


def code_sha(tier):
    """sha256 of a tier's code (and, for the interpretation tier, its data files), line endings normalized; computed
    once per process."""
    if tier not in _code:
        digest = hashlib.sha256()
        for rel in code_files(tier):
            _file_digest(digest, rel)
        for rel in TIER_DATA[tier]:
            _file_digest(digest, rel)
        _code[tier] = digest.hexdigest()
    return _code[tier]


def fingerprint(inputs):
    """sha256 (64 hex) of the canonical JSON of a tier's inputs."""
    return hashlib.sha256(json.dumps(inputs, sort_keys=True, separators=(',', ':'), allow_nan=False)
                          .encode('utf-8')).hexdigest()


def shape_settings_sha(settings):
    """The team-shape settings as the analysis reads them (team_shape.clean_settings), hashed canonically: the same
    value as store.settings_sha."""
    from .. import team_shape
    clean = team_shape.clean_settings(settings or {})
    return hashlib.sha256(json.dumps(clean, sort_keys=True, separators=(',', ':')).encode('utf-8')).hexdigest()


def result_sha_of(result):
    """sha256 of a result dict's canonical JSON, for callers without the file's bytes (tests, tools). The server passes
    the file's own sha (store.file_sha) instead: re-serializing a 110.9 MB result costs seconds."""
    return hashlib.sha256(json.dumps(result, sort_keys=True, separators=(',', ':')).encode('utf-8')).hexdigest()


def _base(tier):
    return {'tier': tier, 'schema': schema.SCHEMA_VERSION, 'engine': schema.ENGINE_VERSION, 'code': code_sha(tier)}


def measurement_inputs(result_sha, shape_sha, team_settings, team):
    """What an obs file is computed from: the result's bytes (sha), the team-shape settings (sha), the team's format
    and field size. shape_sha: shape_settings_sha(settings)."""
    return dict(_base('measurement'), result=result_sha, shapeSettings=shape_sha, team=team,
                measurement=schema.tier_settings(team_settings)['measurement'])


def role_inputs(mfp, team_settings, team):
    """What a roles file is computed from: the obs file (mfp), the confirmed roles and the intended formation."""
    return dict(_base('role'), measurement=mfp, team=team, role=schema.tier_settings(team_settings)['role'])


def _range_key(t0, t1):
    return [None if t0 is None else round(float(t0), 2), None if t1 is None else round(float(t1), 2)]


def interpretation_inputs(mfp, rfp, team_settings, team, t0=None, t1=None):
    """What a report is computed from: the two files, the interpretation settings (age group, level, style,
    federation), the range (None, None for the whole segment) and the interpretation code and data."""
    return dict(_base('interpretation'), measurement=mfp, roles=rfp, team=team, range=_range_key(t0, t1),
                interpretation=schema.tier_settings(team_settings)['interpretation'])


def cache_name(kind, team, mfp, rfp=None, ifp=None):
    """The cache file name of a tier output (DESIGN 1.2): obs-<team>-<mfp16>.json, roles-<team>-<mfp16>-<rfp16>.json,
    report-<team>-<mfp16>-<rfp16>-<ifp16>.json."""
    if kind == 'obs':
        return f'obs-{team}-{mfp[:16]}.json'
    if kind == 'roles':
        return f'roles-{team}-{mfp[:16]}-{rfp[:16]}.json'
    if kind == 'report':
        return f'report-{team}-{mfp[:16]}-{rfp[:16]}-{ifp[:16]}.json'
    raise ValueError(f'unknown cache kind {kind!r}')


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
                       series=None, library=None, keep_series=False, *, coaching=_V1, possession=None, result_sha=None,
                       stages=None, validate=True):
    """Without coaching= (v1): the coaching observations and their evidence for one team ('A' or 'B') or both (team
    None) of a result, over [t0, t1] (absolute seconds; the whole segment by default). reference: a reference dict, a
    path, None (soccer/coaching/reference.json; when it does not exist every comparative detector reports "no
    professional reference available") or False (none). samples / shape: the analytics.Samples and observed
    team_shape.TeamShape of this result when the caller has them (the server's caches); settings: the video's
    team-shape settings for a TeamShape built here; series: a series.CoachingData already built. Returns
    {'engineVersion', 'observations', 'evidence', 'meta'} (JSON-safe), plus 'series' (the CoachingData) with
    keep_series.

    With coaching= (v2 MEASUREMENT tier): the obs file (schema 'obs') of team ('A' or 'B'), or {'A': obs, 'B': obs}
    for team None, over the whole segment. coaching: the video's coaching settings (schema 'settings', {} or None for
    none); only each team's format and field size are read. settings / samples / shape / series as above;
    possession: the possession-4 record of this result (computed when None); result_sha: the sha256 of the result
    file's bytes (computed from the dict when None: slow on long runs); stages: stand-ins for STAGES (tests). The v1
    range and reference options are refused here: a v2 range is an interpretation option (interpret)."""
    if coaching is _V1:
        if possession is not None or result_sha is not None or stages is not None:
            raise TypeError('possession, result_sha and stages are v2 options: pass coaching= for the v2 tier')
        return _observations_v1(result, team, t0, t1, reference, samples, shape, settings, series, library,
                                keep_series)
    if t0 is not None or t1 is not None or reference is not None or library is not None or keep_series:
        raise TypeError('t0, t1, reference, library and keep_series are v1 options; the v2 observation tier covers '
                        'the whole segment (pass a range to interpret)')
    return _observations_v2(result, team, coaching, settings, samples, shape, series, possession, result_sha, stages,
                            validate)


def _observations_v1(result, team, t0, t1, reference, samples, shape, settings, series, library, keep_series):
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


def _teams(team):
    if team is None:
        return schema.TEAMS
    if team not in schema.TEAMS:
        raise ValueError('team must be A or B')
    return (team,)


def _segment(result):
    frames = result.get('frames') or []
    video = result.get('video') or {}
    fps = float(video.get('fps') or 0.0)
    times = [float(f['time']) for f in frames if _num(f.get('time'))]
    return {'t0': round(min(times), 3) if times else 0.0, 't1': round(max(times), 3) if times else 0.0,
            'fps': fps if math.isfinite(fps) and fps > 0 else 0.0, 'frames': len(frames)}


def _video(result):
    name = (result.get('video') or {}).get('name')
    return {'name': name if isinstance(name, str) and name else None, 'key': None}


def _summary(timeline):
    summary = getattr(timeline, 'summary', None)
    if not callable(summary):
        raise TypeError('situations.build_timeline must return an object with a summary() method (schema timeline)')
    return summary()


def _observations_v2(result, team, coaching, settings, samples, shape, series, possession, result_sha, stages,
                     validate):
    teams = _teams(team)
    rsha = result_sha or result_sha_of(result)
    cd = series if series is not None else _stage('series', stages)(result, samples=samples, shape=shape,
                                                                     settings=settings)
    poss = possession if possession is not None else _stage('possession', stages)(result)
    shape_sha = shape_settings_sha(settings)
    segment, video = _segment(result), _video(result)
    out = {}
    for tm in teams:
        block = schema.team_block(coaching, tm)
        measurement = schema.tier_settings(block)['measurement']
        mfp = fingerprint(measurement_inputs(rsha, shape_sha, block, tm))
        timeline = _stage('situations', stages)(cd, poss, tm, copy.deepcopy(measurement))
        found = _stage('detect', stages)(timeline)
        made = _stage('moments', stages)(cd, timeline, found['opportunities'])
        obs = {'kind': 'coaching-observations', 'version': schema.SCHEMA_VERSION, 'team': tm,
               'engine': {'version': schema.ENGINE_VERSION, 'measurement': mfp[:16]}, 'fingerprint': mfp,
               'inputs': {'result': rsha, 'shapeSettings': shape_sha, 'measurement': measurement},
               'segment': segment, 'video': video, 'timeline': _summary(timeline), 'detectors': found['detectors'],
               'opportunities': found['opportunities'], 'moments': made['moments'], 'frameTiers': made['frameTiers'],
               'roleInputs': found.get('roleInputs') or {}}
        if validate:
            schema.check('obs', obs)
        out[tm] = obs
    return out[team] if team is not None else out


def build_roles(observations, coaching=None, team=None, *, stages=None, validate=True):
    """The v2 ROLE tier: the roles file (schema 'roles') of an obs file and the coach's confirmed roles and intended
    formation (coaching: the video's coaching settings; only the team's 'roles' and 'formation' are read). It reads
    nothing but the obs file, so a role confirmation never re-reads the result."""
    obs = observations
    tm = obs['team'] if team is None else team
    if tm != obs['team']:
        raise ValueError(f'the obs file is team {obs["team"]}, not {tm}')
    block = schema.team_block(coaching, tm)
    role = schema.tier_settings(block)['role']
    rfp = fingerprint(role_inputs(obs['fingerprint'], block, tm))
    book = _stage('rolebook', stages)(obs, copy.deepcopy(role))
    found = _stage('detect_roles', stages)(obs, book)
    made = _stage('role_moments', stages)(obs, found['opportunities'])
    roles = {'kind': 'coaching-roles', 'version': schema.SCHEMA_VERSION, 'team': tm,
             'engine': {'version': schema.ENGINE_VERSION, 'measurement': obs['fingerprint'][:16], 'roles': rfp[:16]},
             'measurement': obs['fingerprint'], 'fingerprint': rfp, 'inputs': role, 'rolebook': book,
             'detectors': found['detectors'], 'opportunities': found['opportunities'], 'moments': made['moments']}
    if validate:
        schema.check('roles', roles)
    return roles


def _report_range(obs, t0, t1):
    """(lo, hi, whole): the range clamped to the segment; whole when neither end is given."""
    seg = obs['segment']
    if t0 is None and t1 is None:
        return float(seg['t0']), float(seg['t1']), True
    lo = float(seg['t0']) if t0 is None else min(max(float(t0), seg['t0']), seg['t1'])
    hi = float(seg['t1']) if t1 is None else min(max(float(t1), seg['t0']), seg['t1'])
    return min(lo, hi), max(lo, hi), False


def interpret(observations, roles, coaching=None, team=None, *, t0=None, t1=None, progress=None, meta=None,
              generated=None, stages=None, validate=True):
    """The v2 INTERPRETATION tier: the report (schema 'report') of one team from its obs and roles files and the
    coaching settings (only ageGroup, playingLevel, style and federation are read here; the profile also shows the
    format and formation it was given). t0/t1: the report range (absolute seconds, clamped to the segment; the whole
    segment when both are None): opportunities outside it are left out, the observations are not rebuilt.
    progress: the squad's progression section, merged at serve time (None in a cached report); meta: run facts the
    report shows (run name, team name, video key); generated: an ISO time or None (None keeps a cached report a pure
    function of its inputs). Raises SchemaError when a stage breaks the contract (validate=True) or the files do not
    belong together."""
    obs = observations
    tm = obs['team'] if team is None else team
    if tm != obs['team'] or roles['team'] != tm:
        raise ValueError('the obs file, the roles file and the team must agree')
    if roles['measurement'] != obs['fingerprint']:
        raise ValueError('the roles file was built from another obs file (stale role tier)')
    block = schema.team_block(coaching, tm)
    lo, hi, whole = _report_range(obs, t0, t1)
    ifp = fingerprint(interpretation_inputs(obs['fingerprint'], roles['fingerprint'], block, tm,
                                            None if whole else lo, None if whole else hi))
    profile = _stage('profile', stages)(copy.deepcopy(block), team=tm)
    if validate:
        schema.check('profile', profile)
    assessment = _stage('assess', stages)(obs, roles, profile, t0=lo, t1=hi)
    findings = assessment['findings']
    alternatives, none = [], []
    for f in findings:
        if f['verdict'] not in ALT_VERDICTS:
            f['guardrails']['alternative'] = 'not_applicable'
            continue
        alt = _stage('alternatives', stages)(f, obs, roles, profile)
        if alt is None:                          # a detector without zones (D6): assess's principle, if any, stays
            f['guardrails']['alternative'] = 'pass'
            continue
        if 'zone' in alt:
            alternatives.append(alt)
            f['suggestedAction'] = {'principle': alt['principle'], 'mode': alt['display'], 'alternatives': [alt['id']]}
        else:
            none.append(alt)
            f['suggestedAction'] = {'principle': alt['principle'], 'mode': 'principle', 'alternatives': []}
        f['guardrails']['alternative'] = 'pass'  # a validated zone, or the finding is phrased as a principle (4.8)
    evidence = _stage('annotate', stages)(findings, obs, roles, profile, alternatives=alternatives)
    explained = _stage('explain', stages)(findings, profile, obs=obs, roles=roles, evidence=evidence,
                                          alternatives=alternatives)
    picks = _stage('drills', stages)(explained['findings'], profile, priorities=assessment['priorities'],
                                     next_objective=assessment['nextObjective'])
    report = _stage('report', stages)(obs=obs, roles=roles, profile=profile, assessment=assessment,
                                      findings=explained['findings'], evidence=evidence, alternatives=alternatives,
                                      no_alternatives=none, drills=picks, withheld=explained['withheld'],
                                      progress=progress, meta=dict(meta or {}),
                                      range={'t0': round(lo, 3), 't1': round(hi, 3), 'seconds': round(hi-lo, 3),
                                             'whole': whole})
    report['engine'] = {'version': schema.ENGINE_VERSION, 'measurement': obs['fingerprint'][:16],
                        'roles': roles['fingerprint'][:16], 'interpretation': ifp[:16]}
    report['fingerprints'] = {'measurement': obs['fingerprint'], 'roles': roles['fingerprint'], 'interpretation': ifp}
    report['generated'] = generated
    if validate:
        problems = schema.validate('report', report)+schema.cross_check(report, obs, roles)
        if problems:
            raise schema.SchemaError('report', problems)
    return report


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


__all__ = ['ENGINE_VERSION', 'FOOTAGE_KINDS', 'build_observations', 'build_report', 'build_report_for', 'run_meta',
           'build_roles', 'interpret', 'StageUnavailable', 'STAGES', 'code_files', 'code_sha', 'fingerprint',
           'measurement_inputs', 'role_inputs', 'interpretation_inputs', 'cache_name', 'shape_settings_sha',
           'result_sha_of', 'schema']
