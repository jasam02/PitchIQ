"""The measurement and role tiers as the server and the tools build them: the coach spans folded into the result key,
the two stage additions the store makes until coaching.build_observations makes them itself, the call that builds the
obs files and the one that builds a roles file.

Why a module of its own (critic S-12 / CODE M5, fix round 1, Oct 23). These functions lived in store.py, which no tier
fingerprint hashes (store.py is the cache, not the computation), yet measurement_stages decides what an obs file holds
(the coach spans handed to situations.build_timeline, evidence.moments' pixel anchors merged into obs['roleInputs']):
an edit there left every stored obs file served as fresh. Tracing which soccer/ files run while the measurement and the
role tiers are built (sys.monitoring PY_START over store.measure and store.build_role_tier on EVAL's short_segment scene
and the BC60, AM and VEO run copies; scratch fix1_server/trace_tiers.py) found store.py as the only file that ran and
was not hashed, in both tiers. Here the code is a root of the measurement and the role tiers (coaching.TIER_ROOTS; an
edit here changes the measurement key, which every roles and report key follows anyway), and store.py's
other code (feedback, settings, PDF names, the cache itself) stays out of every fingerprint, so editing it never makes
a coach regenerate. store.py re-exports every name below; callers keep using store.measure and friends.

Nothing here reads settings beyond what coaching.build_observations reads (the team's format and field), so DESIGN 1.3
invariant 2 (no age, level or style reaches an obs file) still holds by construction.
"""
import hashlib
import json


def _sha256(data):
    return hashlib.sha256(data).hexdigest()


def spans_sha(record):
    """sha256 of a coach spans record's canonical JSON; None when there are no spans (a run without spans keeps the
    measurement key it had before spans existed)."""
    if not record or not record.get('spans'):
        return None
    return _sha256(json.dumps(record, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8'))


def measured_result_sha(result_sha, spans):
    """The result sha the measurement tier is keyed on: the result file's own sha, combined with the coach spans' sha
    when the run has spans. coaching.build_observations fingerprints the result, the team-shape settings, the format and
    the field only (LEAD's measurement_inputs), so spans handed to the situations stage would otherwise share an obs
    file with the engine-only measurement; folded into the result key they reach every tier's fingerprint (obs, roles
    and report caches follow them together, as DESIGN 1.2 wants for a measurement input). A run without spans keeps
    the plain sha: its keys are the ones it had before spans existed."""
    sha = spans_sha(spans)
    return result_sha if sha is None else _sha256(f'{result_sha}+coach-spans:{sha}'.encode('ascii'))


def stage(name):
    """A v2 stage function by name (coaching.STAGES), resolved as the package resolves it (StageUnavailable when its
    module or function does not exist)."""
    from soccer import coaching
    return coaching._stage(name)


def measurement_stages(spans=None):
    """The measurement stages the store runs: the package's own, with two additions until coaching.build_observations
    makes them itself (requests to LEAD in INTERFACES.md): the run's coach spans reach situations.build_timeline
    (coach_spans=; CONTEXT's request), and evidence.moments' pixel anchors (its 'roleInputs') are merged into
    obs['roleInputs'] (EVIDENCE's request), without which the role tier's D3 / small-format D5 moments have a diagram
    but no boxes or homography for a video overlay."""
    held = {}
    detect, moments = stage('detect'), stage('moments')

    def detect_kept(timeline):
        held['found'] = found = detect(timeline)
        return found

    def moments_merged(cd, timeline, opportunities):
        made = moments(cd, timeline, opportunities)
        extra = made.get('roleInputs') if isinstance(made, dict) else None
        found = held.get('found')
        if extra and isinstance(found, dict):
            merged = dict(found.get('roleInputs') or {})
            merged.update(extra)
            found['roleInputs'] = merged
        return made
    out = {'detect': detect_kept, 'moments': moments_merged}
    if spans and spans.get('spans'):
        situations = stage('situations')
        out['situations'] = lambda cd, possession, team, measurement: situations(cd, possession, team, measurement,
                                                                                coach_spans=spans)
    return out


def measure(result, *, result_sha, shape_settings, settings, spans=None, team=None, samples=None, shape=None,
            series=None, possession=None):
    """The measurement tier (obs files) of one team, or of both with team None ({'A', 'B'}: the series and possession
    are built once for the two): coaching.build_observations with measurement_stages, keyed on measured_result_sha, so
    obs['fingerprint'] equals store.tier_keys(...)['mfp']. samples / shape / series / possession: the caller's own,
    built from the same bytes (the server's caches), or None to build them here."""
    from soccer import coaching
    return coaching.build_observations(result, team=team, coaching=settings or {}, settings=shape_settings,
                                       samples=samples, shape=shape, series=series, possession=possession,
                                       result_sha=measured_result_sha(result_sha, spans),
                                       stages=measurement_stages(spans))


def build_role_tier(obs, settings, team):
    """The role tier (roles file) of an obs file: coaching.build_roles (reads only the obs file and the team's roles
    and intended formation)."""
    from soccer import coaching
    return coaching.build_roles(obs, coaching=settings or {}, team=team)
