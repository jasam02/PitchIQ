"""Full Match Mode's coaching report: ONE coaching v2 report per team for the whole match, built from the assembled
segment views with the unchanged engine of soccer/coaching (its three cache tiers: measurement -> role ->
interpretation), the measurement made per segment over the frames each segment OWNS and merged into one obs file.

    per segment (one view in memory at a time; measure_segment / measure_segments)
        the view (assembly.load_view: the segment's result with the match's names)
        -> the coach's roster corrections applied (assembly/corrections.json: merged ids, excluded people, team changes;
           apply_fixes), the match's pitch size set (index pitchSize: one scale for the whole match)
        -> possession decoded over the WHOLE view (soccer.possession_amateur.compute, as assembly's possession panel
           decodes it: the context helps the decoder at the edges), its shares and coverage re-counted over the owned
           frames only (owned_possession)
        -> the view trimmed to its owned frames [cut0, cut1) (owned_result: the frames and every per-frame column of the
           timeline and stabilized blocks), so the series, the team shape, the situations, the detectors and the
           evidence see exactly the frames the segment owns and nothing of its context
        -> coaching.build_observations with the server's measurement stages (measuring.measurement_stages: the coach
           spans of the segment, evidence's pixel anchors merged into roleInputs) plus one observer stage that keeps
           the few per-sample statistics a merge cannot rebuild from the obs file (the part's 'extra': speeds for the
           p90, lateral positions for the medians, lineup counts, sums of the per-sample shares)
    merge_obs(parts)   -> ONE obs file (schema 'obs') for the match, keyed like the store's tiers
    coaching.build_roles(obs) and coaching.interpret(obs, roles)   -> the report, plus the match's coverage notices
    cached under assembly/coaching/v2/ in the store's own layout (store.tier_path with the assembly folder as the run
    folder: obs-<team>-<mfp16>.json, roles-..., report-...), so store.find_report_v2, staleness_v2, served_report,
    drills_of and measured_roles work on a match unchanged.

Ownership: nothing from the context frames is counted. A segment's view holds a few seconds beyond its cuts (the
overlap the stitching needed); only the possession decoder reads them (the phase, the carrier and the contest distance
of an OWNED sample may come from a change, a dead ball or a loose ball the decoder saw a second before the cut, exactly
as a long run would), and the identity taint reads the result's issue and event times (a crossing a second before the
cut taints the first owned second: suppression only). Every sample, opportunity, moment, phase second, frame tier and
eligible second of the merged file belongs to exactly one segment's owned frames (tested).

Why the long-range rules then hold. A segment of 180 s is a short range (situations.SHORT_RANGE_S, explain.py): its
own report never says "repeated". The merged file is one long range: assess counts distinct failing spells over the
merged opportunities (situations.distinct: starts 10 s apart, across the whole match) and the 3-minute rule is decided
on the match's range; the owned ranges partition the match, so no opportunity is duplicated by the overlap.

Merge rules, field by field (merge_obs; "exact" = what one run over the same frames, measured stage by stage on the same
samples, would write; the 'extra' of each part is what makes the non-additive ones exact):
    segment            t0 / t1 of the first / last owned frame, fps (one), frames summed.
    inputs             result = the match's measurement sha (every view's bytes, the cuts, the corrections, the coach
                       spans per segment, the pitch size, the possession policy, this module's code), shapeSettings,
                       measurement (format, field); fingerprint = coaching.measurement_inputs of those, as
                       store.tier_keys computes it, so the role and interpretation keys follow from the store's code.
    opportunities, moments   concatenated in time order (ids are content hashes of disjoint times; a repeat refuses).
    frameTiers         concatenated; a run ending on a segment's last grid sample and the next segment's run starting on
                       its first, of one tier, become one run (its confidence the lower), as one run would hold them.
    detectors          contextual.coverage re-run over the merged opportunities with the summed relevant-phase seconds
                       and the summed seconds lost per abstention reason: eligible seconds, spans, counts and the
                       dominant reason recomputed from sums, never averaged.
    timeline           samples summed; t0 / t1 / frame0 / frame1 of the first / last sample; phaseSeconds summed per
                       state (sample counts x step); phaseSpans concatenated, joined across a boundary when one state
                       and source runs on, plus a 'dead' span WITHOUT a source over each break between periods (no
                       play is there; assess derives the unknown-phase seconds from the timeline's range and the
                       restart after a dead ball is unsettled: with it neither counts the halftime break);
                       completeShare, observedMean,
                       ballReliableShare, carrierShare from summed counts; dataQuality's means from summed counts, its
                       lineup the mode of the summed per-sample counts, its dominant cause recomputed
                       (situations._dominant_cause); footage: the small-scene share from summed frame counts, the kind
                       by the run rule on the summed frames and observed players, keyless only when no segment had a
                       keyframe anchor (one run's anchor list is the union), the lens when any segment measured one;
                       formationPeriods concatenated and re-indexed (every period index of the file shifts with them);
                       players: observed seconds summed, the p90 speed from the pooled speed samples (exact).
    phaseQuality       available when possession-4 judged any segment; the image-based estimate block only where
                       possession-4 judged none (its checked / offered seconds summed, agreement from the summed
                       agreeing seconds, used when any segment used it); level 'good' only when every segment was
                       good; coverage = named live seconds / live seconds over every segment (a segment without an
                       estimate counts all its time as live and unnamed); the rough shares from the summed owned shares
                       (or the merged phase seconds where the coach marked spans), rounded to 5% as situations rounds
                       them; the coach block summed (agreement from summed agreeing seconds); the notes regenerated
                       from those sums. Both possession-4 and a USED image estimate in one match cannot be stated by
                       one phaseQuality (the schema refuses it): MergeError (it needs coach spans, which a match does
                       not have today).
    roleInputs         context: ids united, the 1 Hz track concatenated (None where a player was not in a segment's
                       ids), windows and periods concatenated with the period indices shifted; detect: the stored
                       samples concatenated (per-player columns re-ordered to the united ids, the identity bit strings
                       re-mapped, period indices shifted), phase seconds and losses summed, the medians of lateral
                       position recomputed from the pooled positions per (shifted) period (exact); evidence: the pixel
                       anchors united (keyed by the absolute frame); match: this module's per-segment summary (what the
                       coverage notices state; an open roleInputs key, which no tier reads).
What cannot be merged faithfully, and the conservative behaviour chosen:
    - A spell, a phase span or an opportunity that runs across a cut is measured as two (each segment measured alone):
      a 10 s opportunity piece can start at the cut instead of 10 s into the spell, a shape span shorter than 3 s on
      one side drops that side's samples, team_shape's 15 s windows and formation periods restart at the cut, the
      velocities (+-0.8 s) and the analytics smoothing lose their far neighbour within a second of it; the possession
      decoder sees a view, not the match, so a change's window near a cut can move a little. Per boundary and detector
      the opportunity count and the distinct spells differ from one run over the same frames by at most 2 (tested on
      EVAL's unsupported_advance scene cut in two and three: 0-1 measured, D5 27 / 28 and D3 15 / 16 the largest), the
      phase seconds by at most 6 s per state (1.8 s measured), and the verdicts were the same. The role book's
      template slices of the first / last window of a segment cover the window's centre +-2.5 s only (one run would
      have windows there): role-based opportunities lose those seconds (fewer, never more).
    - A formation change on a segment boundary is invisible to both segments' change rule (situations._formation): the
      label runs of the merged periods are re-run within each match period with the same rule
      (_boundary_formation_changes) and every opportunity and stored sample within 15 s of such a change gets
      formationChangeNear (a suppressor only: never a criticism more).
    - A coach-marked change exactly on a cut (one team's span ending there, the other's starting) is not seen by the
      +-2 s coach-change window of either segment (the samples cannot be removed afterwards). Matches have no coach
      spans today; documented, not fixed.
    - The interpretation's phase seconds derive from the timeline's range, so the break between periods would count as
      unknown play: the 'dead' break spans above keep it out; the report's range (t0..t1, interpret's) still spans
      the break, so the overview's "covers N of play" is rewritten to the owned seconds and a notice says so.
    - The video's team-shape settings (runs/<video>/team_shape.json) are made for a Segment run's team letters and are
      not applied: the match's attacking directions per period (the index) are, as the match's team-shape panel does
      (fullmatch/routes.match_shape). The coach's confirmed roles name a Segment run's ids and are dropped
      (match_settings): match ids are another numbering.
    - Without the part's 'extra' (an obs made elsewhere) the p90 speed falls back to the lowest segment value (a smaller
      reach: fewer suggested zones), the lateral medians and the lineup to the segment values weighted by samples.

The coaching settings of a match (match_settings): a match-level file assembly/coaching/coaching_settings.json when one
was saved (store.save_team_settings(settings_dir(folder), team, block) writes it, validated as any video's), else the
video's runs/<video>/coaching_settings.json (store.read_settings: age group, level, format, field, style, federation
and formation per team LETTER) with the confirmed roles dropped. Team letters in a match are the MATCH's letters: the
first segment's run named the kits, every later segment was mapped to them (identities.team_mapping), and a Segment run
of the same video named its kits on its own: the video's block 'A' is applied to the match's team A without any link
between the two (both teams usually share age, level and format; when they do not, check which kit is A in the match).

Routes (fullmatch/routes.py sets context['match_coaching']): report_state() answers what server.v2_answer answers for a
run (state missing / building / ready / stale / error, the tiers, the served report, fingerprints, options, drills,
inferred roles) and builds the role and interpretation tiers at once when the merged obs is stored; start_build()
measures every segment of the match on a background thread (both teams in one pass), one match build at a time.
Measured on dev12 (CAN-ISL 7:00-19:00, 4 x 180 s, CPU, while a full match ran on the GPU): 18 s to measure both teams
(per segment: view 0.9 s, possession 1.8-2.9 s, both teams' measurement 0.8-1.1 s), merge 0.13 s, roles 0.02 s,
report 0.16 s; the other team's report from its stored obs 0.04 s; obs files 0.3-0.55 MB. A 96-minute match of 33
segments is about 2.5 minutes of CPU.
"""
import bisect
import copy
import hashlib
import json
import math
import re
import threading
import time
import traceback
from collections import Counter
from pathlib import Path

import numpy as np

from . import assemble as assembly
from . import corrections as match_corrections

VERSION = 1                       # the merged obs layout and the merge rules; folded into the measurement key
FOLDER = 'coaching'               # assembly/coaching/: match-level settings; the tiers in assembly/coaching/v2/
POSSESSION_POLICY = 'view-decoded/owned-shares-1'   # how a segment's possession record is made (owned_possession)
RUN_PREFIX = 'match-'             # report['run'] of a match report: 'match-<matchId>' (schema RunName)
MAX_SPANS_NAMED = 6               # stretches named in a coverage notice before "and N more"
NOTICE_LIMIT = 400                # schema Notice.text
NOTE_LIMIT = 300                  # schema PhaseQuality.notes
MAX_ERRORS = 32
_lock = threading.RLock()
_sha_memo = {}                    # (path, mtime_ns, size) -> sha256 of a view file
_segment_memo = {}                # (path, mtime_ns, size) -> a stored obs file's segment
_measure_locks = {}               # assembly folder -> Lock: one measurement of a match at a time
_builds = {}                      # assembly folder -> the running background build's status
_errors = {}                      # fingerprint -> the last build error
_code = []


class MergeError(ValueError):
    """Segment measurements that one obs file cannot state faithfully (module docstring)."""


# ---------- folders, names ----------
def assembly_dir(folder):
    """The assembly folder of a match: folder is the match folder (runs/<video>/matches/<id>) or its assembly/."""
    folder = Path(folder)
    if (folder/'assembly'/'match.json').exists() or not (folder/'match.json').exists():
        return folder/'assembly'
    return folder


def match_dir(folder):
    return assembly_dir(folder).parent


def load_index(folder):
    """The match index (assembly/match.json) as assembly wrote it (the corrections are read apart: fixes_of)."""
    return json.loads((assembly_dir(folder)/'match.json').read_text(encoding='utf-8'))


def settings_dir(folder):
    """Where a match's own coaching settings live (assembly/coaching/coaching_settings.json, store.SETTINGS_FILE)."""
    return assembly_dir(folder)/FOLDER


def run_name(index):
    """report['run'] of a match report: 'match-<matchId>', in the schema's RunName alphabet."""
    return re.sub(r'[^A-Za-z0-9_.\-]', '-', f"{RUN_PREFIX}{index.get('matchId') or 'match'}")[:120]


def _segments(index):
    return sorted(index.get('segments') or [], key=lambda s: int(s['cut0']))


def _fps(index):
    fps = float((index.get('video') or {}).get('fps') or 25.0)
    return fps if math.isfinite(fps) and fps > 0 else 25.0


# ---------- the settings a match uses ----------
def match_settings(video_dir=None, folder=None):
    """(settings, info): the coaching settings (schema 'settings') of a match's reports and where they came from (module
    docstring): the match's own file when saved, else the video's coaching_settings.json without the confirmed roles
    (they name a Segment run's ids), else none (PitchIQ's defaults; profiles.resolve says so in a notice). info:
    {'source' 'match' | 'video' | 'default', 'path', 'droppedRoles' {letter: [ids]}, 'letters' (the rule above)}."""
    from soccer.coaching import profiles, store
    letters = ("Team letters are the match's: the first segment named the kits and every later segment was mapped to "
               "them; a Segment run of the same video named its kits on its own.")
    if folder is not None and store.settings_path(settings_dir(folder)).exists():
        return store.read_settings(settings_dir(folder)), {'source': 'match', 'droppedRoles': {}, 'letters': letters,
                                                           'path': str(store.settings_path(settings_dir(folder)))}
    if video_dir is not None and store.settings_path(video_dir).exists():
        settings = store.read_settings(video_dir)
        dropped = {}
        for letter, block in (settings.get('teams') or {}).items():
            if isinstance(block, dict) and block.get('roles'):
                dropped[letter] = sorted(block['roles'])
                block['roles'] = {}
        return settings, {'source': 'video', 'droppedRoles': dropped, 'letters': letters,
                          'path': str(store.settings_path(video_dir))}
    return profiles.clean_settings({}), {'source': 'default', 'droppedRoles': {}, 'letters': letters, 'path': None}


def with_profile(settings, team, *, age=None, level=None, style=None, federation=None):
    """A copy of settings with team's age group, playing level, style or federation replaced (the block cleaned by
    profiles.clean_team; ValueError when a value does not clean). The other team and every other field stay; a team
    without a block gets a new one, completed with the age's default format (which is a measurement change: the format
    is a measurement input)."""
    from soccer.coaching import profiles, schema as S
    if team not in S.TEAMS:
        raise ValueError('team must be A or B')
    out = copy.deepcopy(settings) if isinstance(settings, dict) else profiles.clean_settings({})
    out.setdefault('teams', {'A': None, 'B': None})
    block = dict(out['teams'].get(team) or {})
    for key, value in (('ageGroup', age), ('playingLevel', level), ('style', style), ('federation', federation)):
        if value is not None:
            block[key] = value
    problems = []
    cleaned = profiles.clean_team(block, problems=problems, path=f'teams.{team}')
    if problems or cleaned is None:
        raise ValueError('; '.join(problems) or 'the team settings do not clean')
    out['teams'][team] = cleaned
    return out


# ---------- keys ----------
def _file_sha(path):
    """sha256 of a file's bytes, remembered per (path, mtime, size): a status poll hashes nothing twice."""
    p = Path(path)
    st = p.stat()
    key = (str(p), st.st_mtime_ns, st.st_size)
    with _lock:
        hit = _sha_memo.get(key)
    if hit is not None:
        return hit
    digest = hashlib.sha256()
    with p.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            digest.update(chunk)
    sha = digest.hexdigest()
    with _lock:
        if len(_sha_memo) > 1024:
            _sha_memo.clear()
        _sha_memo[key] = sha
    return sha


def code_sha():
    """sha256 of this module (line endings normalized): the merge rules and the match notices are part of what the
    merged obs and its reports are computed from (soccer/coaching's code is in the tiers' own fingerprints)."""
    if not _code:
        _code.append(hashlib.sha256(Path(__file__).read_bytes().replace(b'\r\n', b'\n')).hexdigest())
    return _code[0]


def shape_settings_of(index):
    """The team-shape settings a match is measured with: the attacking direction from the match's periods (team A's
    defending side in the first period, switched from the second period's start when the teams changed ends), exactly
    as fullmatch/routes.match_shape gives the match's team-shape panel; no lineup or formation labels (the video's are
    made for a Segment run's letters and times)."""
    from soccer import team_shape
    periods = index.get('periods') or []
    first = periods[0].get('defendsLeft') if periods else None
    direction = {'A': None, 'switchAt': None}
    if first in ('A', 'B'):
        direction['A'] = 'left' if first == 'A' else 'right'
        if len(periods) > 1 and periods[1].get('defendsLeft') in ('A', 'B') and periods[1]['defendsLeft'] != first:
            direction['switchAt'] = float(periods[1]['start'])
    return team_shape.clean_settings({'direction': direction})


def fixes_of(folder):
    """The coach's roster corrections of the match (assembly/corrections.json) resolved, or the match's people when they
    are built (fullmatch/intel.py: the corrections are applied in them): {'aliases' {id: final id}, 'excluded' [ids],
    'teams' {id: letter}}, sorted (a key input)."""
    from . import intel
    try:
        snap = intel.snapshot(match_dir(folder))
    except Exception:   # noqa: BLE001 - without the match's people the coach's corrections apply
        snap = None
    if snap is not None:
        aliases, excluded, teams = intel.aliases_of(snap['people'])
    else:
        aliases, excluded, teams = match_corrections.resolve_aliases(match_corrections.load(match_dir(folder)))
    return {'aliases': dict(sorted(aliases.items())), 'excluded': sorted(excluded),
            'teams': dict(sorted(teams.items()))}


def segment_spans(spans, start, end):
    """The coach-marked spans (schema 'coach_spans', absolute seconds over the match) that reach one segment's owned
    range [start, end), clipped to it; None when none do. A span across a cut is cut there (module docstring)."""
    raw = spans.get('spans') if isinstance(spans, dict) else spans
    out = []
    for s in raw or ():
        if not isinstance(s, dict) or s.get('state') not in ('A', 'B', 'dead'):
            continue
        try:
            a, b = max(float(s['t0']), float(start)), min(float(s['t1']), float(end))
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(a) and math.isfinite(b) and b > a:
            out.append({'t0': round(a, 3), 't1': round(b, 3), 'state': s['state']})
    return {'version': 1, 'kind': 'coach-possession-spans', 'source': 'coach', 'spans': out} if out else None


def measurement_key(folder, index, spans=None, fixes=None):
    """(sha, inputs): what the match's measurement is computed from besides the team-shape settings and the team's
    format and field (store.tier_keys adds those): every view's bytes and cuts, the coach spans of each segment, the
    corrections, the pitch size, the possession policy and this module's code (module docstring)."""
    from soccer import coaching
    from soccer.coaching import measuring
    a = assembly_dir(folder)
    fixes = fixes_of(folder) if fixes is None else fixes
    rows = [{'index': int(s['index']), 'cut0': int(s['cut0']), 'cut1': int(s['cut1']), 'view': _file_sha(a/s['view']),
             'spans': measuring.spans_sha(segment_spans(spans, s['start'], s['end']))} for s in _segments(index)]
    inputs = {'kind': 'fullmatch-coaching', 'version': VERSION, 'code': code_sha(), 'possession': POSSESSION_POLICY,
              'matchId': index.get('matchId'), 'pitchSize': index.get('pitchSize'), 'fixes': fixes, 'segments': rows}
    return coaching.fingerprint(inputs), inputs


def _index_segment(index):
    """The match's first and last owned frame times as the merged obs file's segment will state them (the last owned
    frame is one frame before the last segment's end)."""
    segs = _segments(index)
    return {'t0': round(float(segs[0]['start']), 3), 't1': round(float(segs[-1]['end'])-1.0/_fps(index), 3)}


def _obs_segment(path):
    """A stored obs file's segment (remembered per file version), or None."""
    p = Path(path)
    try:
        st = p.stat()
    except OSError:
        return None
    key = (str(p), st.st_mtime_ns, st.st_size)
    with _lock:
        if key in _segment_memo:
            return _segment_memo[key]
    from soccer.coaching import store
    obs = store.load_tier(p)
    seg = {'t0': obs['segment']['t0'], 't1': obs['segment']['t1']} if obs and obs.get('segment') else None
    with _lock:
        if len(_segment_memo) > 64:
            _segment_memo.clear()
        _segment_memo[key] = seg
    return seg


def match_keys(folder, index, team, settings, *, shape_settings=None, spans=None, fixes=None, t0=None, t1=None,
               segment=None, measured=None):
    """The three tier keys of one team's match report (store.tier_keys' dict: team, mfp, rfp, ifp, range, inputs),
    with the match's measurement sha as the result sha; measured: (sha, inputs) of measurement_key when known."""
    from soccer.coaching import store
    shape = shape_settings_of(index) if shape_settings is None else shape_settings
    sha, inputs = measured or measurement_key(folder, index, spans, fixes)
    seg = segment
    if seg is None and (t0 is not None or t1 is not None):
        probe = store.tier_keys(sha, shape, settings, team, None, _index_segment(index))
        seg = _obs_segment(store.tier_path(assembly_dir(folder), 'obs', probe))
    keys = store.tier_keys(sha, shape, settings, team, None, seg or _index_segment(index), t0, t1)
    keys['match'] = {'sha': sha, 'segments': len(inputs['segments'])}
    return keys


# ---------- one segment ----------
def _slice_block(block, a, b, n):
    """A timeline / stabilized block with every per-frame column cut to the frames [a, b) of the n it was made for."""
    out = dict(block)
    out['frameTimes'] = list(block['frameTimes'][a:b])
    for group in ('ids', 'anonymous'):
        people = block.get(group)
        if isinstance(people, dict):
            out[group] = {k: ({f: (x[a:b] if isinstance(x, list) and len(x) == n else x) for f, x in v.items()}
                              if isinstance(v, dict) else v) for k, v in people.items()}
    return out


def owned_result(view, cut0, cut1):
    """The view as a result of exactly its owned frames [cut0, cut1) (module docstring): the frames and every per-frame
    column of the timeline and stabilized blocks cut together (team_shape reads the timeline only while its frameTimes
    match the frames); everything else (roster, match, tactical, summary, issues, events) is the view's."""
    frames = view.get('frames') or []
    nums = [int(f['frame']) for f in frames]
    if any(b <= a for a, b in zip(nums, nums[1:])):
        raise ValueError('the view\'s frames are not in frame order')
    a, b = bisect.bisect_left(nums, int(cut0)), bisect.bisect_left(nums, int(cut1))
    out = dict(view)
    out['frames'] = frames[a:b]
    for name in ('timeline', 'stabilized'):
        block = view.get(name)
        if isinstance(block, dict) and len(block.get('frameTimes') or ()) == len(frames):
            out[name] = _slice_block(block, a, b, len(frames))
    return out


def _left_time(left):
    """A roster slot's 'left' as a time (a number or {'time': t}), None when it has not left."""
    left = left.get('time') if isinstance(left, dict) else left
    return float(left) if isinstance(left, (int, float)) and not isinstance(left, bool) else None


def _merge_columns(keep, other):
    """One person's per-frame columns of a timeline / stabilized block with another's (a merged identity: never seen in
    the same frame): where keep has no position, other's values."""
    out = dict(keep)
    xs, ox = keep.get('x'), other.get('x')
    if not (isinstance(xs, list) and isinstance(ox, list) and len(xs) == len(ox)):
        return out
    fill = [k for k, v in enumerate(xs) if v is None and ox[k] is not None]
    if not fill:
        return out
    for f, col in keep.items():
        oc = other.get(f)
        if isinstance(col, list) and isinstance(oc, list) and len(col) == len(xs) == len(oc):
            col = list(col)
            for k in fill:
                col[k] = oc[k]
            out[f] = col
    return out


def apply_fixes(view, fixes):
    """In place: the coach's roster corrections on a view (fullmatch/corrections.py), as the match's analytics apply
    them: a merged identity's observations are the identity it now is (people, match entries, roster slots with the
    earliest first sighting, timeline / stabilized columns joined, visibility, labels, issues, events), an excluded one
    is nobody the analysis counts (people marked not a target without an id or team, left out of the match players,
    goalkeepers and roster slots), a re-teamed one plays for its new team. Returns the view."""
    aliases, teams = fixes.get('aliases') or {}, fixes.get('teams') or {}
    excluded = set(fixes.get('excluded') or ())
    if not (aliases or excluded or teams):
        return view

    def final(pid):
        return aliases.get(pid, pid) if pid is not None else None
    for f in view.get('frames') or ():
        for p in f.get('people') or ():
            pid = final(p.get('id'))
            if pid is None:
                continue
            if pid in excluded:
                p.update(id=None, team=None, state='nontarget', excludedBy='coach')
                continue
            p['id'] = pid
            if pid in teams:
                p['team'] = teams[pid]
        vis = f.get('visibility')
        if isinstance(vis, dict):
            joined = {}
            for k, v in vis.items():
                fk = final(k)
                if fk not in excluded and fk not in joined:
                    joined[fk] = v
            f['visibility'] = joined
    match = view.get('match') or {}
    for group, entries in match.items():
        if not isinstance(entries, list):
            continue
        kept, other = {}, []
        for e in entries:
            if not isinstance(e, dict) or e.get('id') is None:
                other.append(e)
                continue
            pid = final(e['id'])
            if pid in excluded and group in ('players', 'goalkeepers'):
                continue
            entry = dict(e, id=pid, team=teams.get(pid, e.get('team')))
            if pid not in kept or e['id'] == pid:       # the identity's own entry wins over a merged one's
                kept[pid] = entry
        match[group] = list(kept.values())+other
    roster = view.get('roster') or {}
    if isinstance(roster.get('slots'), list):
        members = {}
        for s in roster['slots']:
            pid = final(s.get('id'))
            if pid is not None and pid not in excluded:
                members.setdefault(pid, []).append(s)
        slots = []
        for pid, group in members.items():
            # the identity's own slot (else the first merged one), seen from the earliest first sighting of any member
            # (team_shape masks a slot before it: a merged fragment seen earlier would be dropped), and left only
            # when every member had left (the latest leaving)
            own = next((s for s in group if s.get('id') == pid), group[0])
            firsts = [s['firstSeen'] for s in group if isinstance(s.get('firstSeen'), (int, float))]
            lefts = [_left_time(s.get('left')) for s in group]
            left = None
            if all(x is not None for x in lefts):
                left = max(zip(lefts, (s.get('left') for s in group)), key=lambda x: x[0])[1]
            slots.append(dict(own, id=pid, team=teams.get(pid, own.get('team')),
                              firstSeen=min(firsts) if firsts else own.get('firstSeen'), left=left))
        roster['slots'] = slots
    for name in ('timeline', 'stabilized'):
        block = view.get(name)
        if isinstance(block, dict) and isinstance(block.get('ids'), dict):
            joined = {}
            for k, v in block['ids'].items():
                fk = final(k)
                if fk in excluded:
                    continue
                v = dict(v, team=teams[fk]) if isinstance(v, dict) and fk in teams and 'team' in v else v
                if fk in joined and isinstance(v, dict) and isinstance(joined[fk], dict):
                    joined[fk] = _merge_columns(joined[fk], v) if k != fk else _merge_columns(v, joined[fk])
                else:
                    joined[fk] = v
            block['ids'] = joined
    if isinstance(view.get('labels'), dict):
        view['labels'] = {final(k): v for k, v in view['labels'].items() if final(k) not in excluded}
    for e in view.get('events') or ():
        if isinstance(e, dict) and e.get('playerId') is not None:
            e['playerId'] = final(e['playerId'])
    for i in view.get('issues') or ():
        if isinstance(i, dict) and i.get('id') is not None:
            i['id'] = final(i['id'])
    return view


def owned_possession(record, owned_frames, cut0, cut1, fps):
    """(the record the segment's measurement uses, facts for the merge) from the possession record of the whole view:
    the runs, changes and dead spells of the whole view are kept (situations maps them onto the owned samples only, by
    frame and time: a change a second before the cut still excludes the first owned seconds, as in one long run), the
    shares and the coverage are re-counted over the owned frames (possession-4's definitions: seconds per team, dead and
    unknown; coverage = named / live), so nothing of the context is counted. facts: version, available, image-based or
    not, the owned shares and coverage, the owned changes (frame in [cut0, cut1)), the quality level and the reasons."""
    from soccer.coaching import situations as SI
    record = record if isinstance(record, dict) else {}
    rec = dict(record)
    facts = {'version': record.get('version'), 'available': record.get('available') is True,
             'image': SI.is_image_estimate(record), 'shares': None, 'coverage': None,
             'level': (record.get('quality') or {}).get('level'), 'reason': record.get('reason'),
             'fallbackReason': (record.get('fallbackFrom') or {}).get('reason'),
             'changes': sum(1 for c in record.get('changes') or () if isinstance(c, dict) and
                            isinstance(c.get('frame'), (int, float)) and int(cut0) <= c['frame'] < int(cut1))}
    if facts['available'] and isinstance(record.get('timeline'), list):
        nums = np.asarray(sorted(int(f['frame']) for f in owned_frames), np.int64)
        dt = 1.0/float(fps)
        sec = {'A': 0.0, 'B': 0.0, 'dead': 0.0, 'unknown': 0.0}
        for r in record['timeline']:
            if not isinstance(r, dict) or not isinstance(r.get('f0'), (int, float)) or \
                    not isinstance(r.get('f1'), (int, float)):
                continue
            n = int(np.searchsorted(nums, r['f1'], 'right')-np.searchsorted(nums, r['f0'], 'left'))
            if n <= 0:
                continue
            key = r['team'] if r.get('team') in ('A', 'B') else ('dead' if r.get('state') == 'dead' else 'unknown')
            sec[key] += n*dt
        live = sec['A']+sec['B']+sec['unknown']
        facts['shares'] = {k: round(v, 3) for k, v in sec.items()}
        facts['coverage'] = round((sec['A']+sec['B'])/live, 3) if live > 0 else 0.0
        rec['shares'] = {k: round(v, 2) for k, v in sec.items()}
        rec['quality'] = dict(record.get('quality') or {}, coverage=facts['coverage'])
    return rec, facts


def _extras(tl, cd):
    """What merge_obs needs beyond the obs file to be exact (module docstring), from one team's SituationTimeline and
    the coaching series it was built from: counts and sums per sample, the pooled speed and lateral-position samples,
    the run facts of the footage."""
    from soccer.coaching import detectors as DT
    n = len(tl.t)
    period = np.asarray(tl.formationPeriod, int) if tl.format['players'] == 11 else np.full(n, -1, int)
    Y = np.where(tl.posOk, tl.Y, np.nan)
    lateral = {}
    for j, pid in enumerate(tl.ids):
        ok = np.isfinite(Y[:, j])
        if ok.any():
            lateral[pid] = (Y[ok, j].astype(np.float64), period[ok].astype(np.int64))
    speeds, seen = {}, {}
    for ids, speed, vok, pok in ((tl.ids, tl.speed, tl.velOk, tl.posOk),
                                 (tl.oppIds, tl.oppSpeed, tl.oppVelOk, tl.oppPosOk)):
        for j, pid in enumerate(ids):
            speeds[pid] = np.asarray(speed[:, j][vok[:, j]], np.float64)
            seen[pid] = int(np.asarray(pok[:, j]).sum())
    keepers = {}
    data = tl.cd.shape.observed if tl.cd is not None and tl.cd.shape is not None else None
    if data is not None:
        for team, key in ((tl.team, 'gk'), (tl.other, 'oppGk')):
            gks = data.teams[team].get('gks') or []
            if len(gks) == 1:
                keepers[gks[0]] = int(np.isfinite(getattr(tl, key)[:, 0]).sum())
    phase = np.asarray(tl.phase)
    run = DT.run_facts(cd)
    reg = (((cd.result or {}).get('summary') or {}).get('pitch') or {}).get('registration') or {}
    anchors = reg.get('anchors') if isinstance(reg, dict) else None
    frames = (cd.result or {}).get('frames') or []
    return {'samples': n, 'outfield': np.bincount(np.asarray(tl.outfield, int)).tolist() if n else [],
            'inKit': int(np.sum(tl.inKit)), 'confirmed': int(np.sum(tl.confirmedInKit)),
            'positioned': int(np.sum(tl.observedCount)), 'complete': int(np.sum(tl.complete)),
            'ball': int(np.sum(tl.ballReliable)), 'opp': int((phase == 'opp').sum()),
            'oppCarrier': int(((phase == 'opp') & (np.asarray(tl.carrier) != '')).sum()),
            'speeds': speeds, 'seen': seen, 'keepers': keepers, 'lateral': lateral,
            'observed': {team: (float(np.sum(s.observed)), int(len(s.observed))) for team, s in cd.teams.items()},
            'smallFrames': sum(1 for f in frames if (f.get('profile') or {}).get('smallScene')), 'frames': len(frames),
            'calibrated': bool((cd.available or {}).get('calibration')), 'anchorList': isinstance(anchors, list),
            'landmarks': bool(isinstance(anchors, list) and any(isinstance(x, dict) and x.get('source') == 'landmarks'
                                                                for x in anchors)),
            'lens': bool(run['lens'])}


def _possession_engine(view):
    from soccer import possession_amateur
    return possession_amateur.compute(view)


def measure_segment(view, segment, settings, *, shape_settings, spans=None, fixes=None, pitch_size=None, team=None,
                    possession=None, result_sha=None):
    """One segment's measurement over its owned frames (module docstring): {team: part} for team None (both teams, one
    series and one possession pass), else that team's part. part: {'segment' (index, period, cuts, owned start / end,
    the index's quality facts), 'obs' (schema 'obs' of the owned frames), 'extra' (_extras + the possession facts),
    'timings'}. The view is changed in place (corrections, pitch size); possession(view) stands in for the possession
    dispatch (tests)."""
    from soccer import coaching
    from soccer.coaching import measuring
    started = time.time()
    cut0, cut1 = int(segment['cut0']), int(segment['cut1'])
    if pitch_size:
        view['tactical'] = dict(view.get('tactical') or {}, pitchSize=dict(pitch_size))
    if fixes:
        apply_fixes(view, fixes)
    fps = float((view.get('video') or {}).get('fps') or 25.0)
    t = time.time()
    record = (possession or _possession_engine)(view)
    t_poss = time.time()-t
    owned = owned_result(view, cut0, cut1)
    if not owned['frames']:
        raise ValueError(f'segment {segment.get("index")} owns no frame of its view')
    rec, pfacts = owned_possession(record, owned['frames'], cut0, cut1, fps)
    part_spans = segment_spans(spans, segment['start'], segment['end'])
    sha = result_sha or coaching.fingerprint({'segment': int(segment.get('index', 0)), 'cut0': cut0, 'cut1': cut1})
    stages = measuring.measurement_stages(part_spans)
    base = stages.get('situations') or measuring.stage('situations')
    captured = {}

    def situations(cd, poss, tm, measurement):
        tl = base(cd, poss, tm, measurement)
        captured[tm] = _extras(tl, cd)
        return tl
    stages['situations'] = situations
    t = time.time()
    made = coaching.build_observations(owned, team=team, coaching=settings or {}, settings=shape_settings,
                                       possession=rec, result_sha=measuring.measured_result_sha(sha, part_spans),
                                       stages=stages)
    t_measure = time.time()-t
    made = made if team is None else {team: made}
    quality = segment.get('quality') or {}
    info = {'index': int(segment.get('index', 0)), 'period': int(segment.get('period') or 0), 'cut0': cut0,
            'cut1': cut1, 'start': float(segment['start']), 'end': float(segment['end']),
            'key': segment.get('key'), 'spans': part_spans is not None,
            'quality': {k: quality.get(k) for k in ('keyframeAnchors', 'automaticAnchors', 'calibratedShare',
                                                     'underConstrainedShare', 'identifiedShare', 'smallSceneShare')}}
    timings = {'possession': round(t_poss, 2), 'measure': round(t_measure, 2), 'total': round(time.time()-started, 2)}
    return {tm: {'segment': dict(info), 'obs': obs, 'extra': dict(captured[tm], possession=pfacts), 'timings': timings}
            for tm, obs in made.items()}


def measure_segments(folder, index=None, team=None, settings=None, *, shape_settings=None, spans=None, fixes=None,
                     progress=None, possession=None):
    """Every segment of an assembled match measured over its owned frames, one view in memory at a time: the parts of
    one team in match order ([part]), or {'A': [...], 'B': [...]} for team None (one pass for both teams).
    shape_settings: shape_settings_of(index) by default; fixes: the match's corrections by default (fixes_of); spans: a
    match-level coach_spans record or None; progress(done, total, stage) is called per segment."""
    from soccer import coaching
    a = assembly_dir(folder)
    index = index or load_index(folder)
    shape = shape_settings_of(index) if shape_settings is None else shape_settings
    fixes = fixes_of(folder) if fixes is None else fixes
    segs = _segments(index)
    out = {tm: [] for tm in (('A', 'B') if team is None else (team,))}
    for k, s in enumerate(segs):
        if progress:
            progress(k, len(segs), f'Measuring segment {k+1} of {len(segs)}')
        t = time.time()
        view = assembly.load_view(a, s['view'])
        loaded = time.time()-t
        # the part's own result key (64 hex, schema ObsInputs): its view's bytes, cuts, corrections and pitch size
        sha = coaching.fingerprint({'view': _file_sha(a/s['view']), 'cut0': int(s['cut0']), 'cut1': int(s['cut1']),
                                    'fixes': fixes, 'pitchSize': index.get('pitchSize'), 'version': VERSION})
        made = measure_segment(view, s, settings, shape_settings=shape, spans=spans, fixes=fixes,
                               pitch_size=index.get('pitchSize'), team=team, possession=possession, result_sha=sha)
        del view
        for tm, part in made.items():
            part['timings'] = dict(part['timings'], load=round(loaded, 2))
            out[tm].append(part)
    if progress:
        progress(len(segs), len(segs), 'Merging')
    return out if team is None else out[team]


# ---------- merging ----------
def _r(x, d=3):
    return None if x is None else round(float(x), d)


def _union(lists):
    out, seen = [], set()
    for ids in lists:
        for pid in ids:
            if pid not in seen:
                seen.add(pid)
                out.append(pid)
    return out


def _same(values, what):
    first = values[0]
    for v in values[1:]:
        if json.dumps(v, sort_keys=True) != json.dumps(first, sort_keys=True):
            raise MergeError(f'the segments disagree on {what} ({first!r} and {v!r})')
    return copy.deepcopy(first)


def _join_runs(runs, boundaries, key, step):
    """Concatenated runs (phase spans or frame tiers) with a run that ends on a segment's last grid sample and the next
    segment's run that starts on its first joined when key(run) agrees (one run would hold them as one); boundaries:
    (last grid frame, next first grid frame) of the segments whose owned ranges touch (each segment's 5 Hz grid starts
    on its own first frame, so the two samples are one to two steps apart)."""
    out = []
    for r in runs:
        r = dict(r)
        if out:
            prev = out[-1]
            joined = any(prev['frame1'] == last and r['frame0'] == first for last, first in boundaries)
            if joined and key(prev) == key(r) and float(r['t0'])-float(prev['t1']) < 2.0*step+1e-6:
                prev['t1'], prev['frame1'] = r['t1'], r['frame1']
                if 'confidence' in prev:
                    c = [x for x in (prev.get('confidence'), r.get('confidence')) if x is not None]
                    prev['confidence'] = min(c) if c else None
                continue
        out.append(r)
    return out


def _phase_quality(parts, sec, footage_kind):
    """The merged phaseQuality (module docstring) and its notes."""
    from soccer.coaching import situations as SI
    pqs = [p['obs']['timeline']['phaseQuality'] for p in parts]
    facts = [(p.get('extra') or {}).get('possession') or {} for p in parts]
    images = [pq.get('imageEstimate') for pq in pqs]
    used = [bool(im and im.get('used')) for im in images]
    available = any(pq.get('available') for pq in pqs)
    if available and any(used):
        raise MergeError('possession-4 named the phase in some segments and a used image-based estimate in others: '
                         'one phaseQuality cannot state both')
    n_engine = sum(1 for pq in pqs if pq.get('available'))
    image = None
    if not available and any(im is not None for im in images):
        ims = [im for im in images if im is not None]
        checked = sum(float(im.get('checkedSeconds') or 0.0) for im in ims)
        agree = sum(float(im['agreement'])*float(im.get('checkedSeconds') or 0.0) for im in ims
                    if im.get('agreement') is not None)
        any_used = any(used)
        reasons = Counter(im.get('reason') for im in ims if im.get('reason'))
        image = {'version': ims[0].get('version') or 'possession-amateur', 'used': any_used,
                 'reason': None if any_used else (reasons.most_common(1)[0][0] if reasons else 'no_marks'),
                 'agreement': round(agree/checked, 3) if checked > 0 else None, 'checkedSeconds': round(checked, 2),
                 'offeredSeconds': round(sum(float(im.get('offeredSeconds') or 0.0) for im in ims), 2)}
    engine = available or bool(image and image['used'])
    coaches = [pq.get('coach') for pq in pqs if pq.get('coach')]
    coach = None
    if coaches:
        named = sum(float(c.get('engineNamedSeconds') or 0.0) for c in coaches)
        agree = sum(float(c['agreement'])*float(c.get('engineNamedSeconds') or 0.0) for c in coaches
                    if c.get('agreement') is not None)
        coach = {'spans': sum(int(c['spans']) for c in coaches), 'dropped': sum(int(c['dropped']) for c in coaches),
                 **{k: round(sum(float(c[k]) for c in coaches), 2) for k in ('seconds', 'own', 'opp', 'dead')},
                 'engineNamedSeconds': round(named, 2), 'agreement': round(agree/named, 3) if named > 0 else None}
    # coverage: named live seconds / live seconds, every segment counted (one without an estimate: all live, unnamed)
    named_s = live_s = 0.0
    for p, pq, f in zip(parts, pqs, facts):
        part_s = float(p['obs']['timeline']['samples'])*float(p['obs']['timeline']['step'])
        shares = f.get('shares') if (pq.get('available') or ((pq.get('imageEstimate') or {}).get('used'))) else None
        if shares:
            named_s += shares['A']+shares['B']
            live_s += shares['A']+shares['B']+shares['unknown']
        elif pq.get('available') or ((pq.get('imageEstimate') or {}).get('used')):
            cov = pq.get('coverage')
            if isinstance(cov, (int, float)):            # no facts (an obs made elsewhere): its own coverage
                named_s += float(cov)*part_s
            live_s += part_s
        else:
            live_s += part_s
    coverage = round(named_s/live_s, 3) if engine and live_s > 0 else None
    rough = None
    step = SI.ROUGH_SHARE_STEP
    team = parts[0]['obs']['team']
    other = 'B' if team == 'A' else 'A'
    if engine and coach is not None and coach['spans'] and sec['own']+sec['opp'] > 0:
        x = round(sec['own']/(sec['own']+sec['opp'])/step)*step
        rough = {team: round(x, 2), other: round(1.0-x, 2)}
        rough = {'A': rough['A'], 'B': rough['B']}
    elif available:
        # possession-4's own shares only (an image-based estimate that was not used names nothing)
        a = sum(f['shares']['A'] for f, pq in zip(facts, pqs) if f.get('shares') and pq.get('available'))
        b = sum(f['shares']['B'] for f, pq in zip(facts, pqs) if f.get('shares') and pq.get('available'))
        if a+b > 0:
            x = round(a/(a+b)/step)*step
            rough = {'A': round(x, 2), 'B': round(1.0-x, 2)}
    if available:
        level = 'good' if all(pq.get('available') and pq.get('level') == 'good' for pq in pqs) else 'limited'
        reason = None
    else:
        level = 'limited' if image and image['used'] else 'unavailable'
        reasons = Counter(pq.get('reason') for pq in pqs if pq.get('reason'))
        reason = reasons.most_common(1)[0][0] if reasons else None
    changes = sum(int(f.get('changes') or 0) for f in facts)
    notes = []
    n = len(parts)
    if available:
        if changes:
            notes.append(f'Who had the ball is estimated by the possession engine; the {SI.CHANGE_EXCLUDE_S:.0f} s '
                         f'either side of each of its {changes} changes of possession are left out '
                         f'({sec["nearChange"]:.0f} s).')
        else:
            notes.append('Who had the ball is estimated by the possession engine (it found no change of possession).')
        if n_engine < n:
            notes.append(f'The possession engine judged {n_engine} of the {n} segments of the match; in the others who '
                         'had the ball is not named.')
        if level == 'limited' and coverage is not None:
            notes.append(f'Possession quality is limited on this match (coverage {coverage*100:.0f}%).')
        if footage_kind == 'broadcast':
            notes.append(SI.STOPPAGE_NOTE)
    elif image is not None and image['used']:
        notes.append(f'Who had the ball is estimated from the image (an image-based estimate) outside the spans you '
                     f'marked: inside them it named your team on {(image["agreement"] or 0.0)*100:.0f}% of '
                     f'{image["checkedSeconds"]:.0f} s; the {SI.CHANGE_EXCLUDE_S:.0f} s either side of each of its '
                     f'changes are left out.')
    elif image is not None:
        why = {'no_marks': 'it is used only once spans you mark confirm it',
               'too_few_seconds': f'it named a team on only {image["checkedSeconds"]:.0f} s inside your marked spans '
                                  f'({SI.IMAGE_MIN_S:.0f} s are needed to check it)',
               'disagrees': f'inside your marked spans it named your team on only '
                            f'{(image["agreement"] or 0.0)*100:.0f}% of the time '
                            f'({SI.IMAGE_AGREE_MIN*100:.0f}% needed)'}
        notes.append('The possession engine does not judge this footage; an image-based estimate exists but is not '
                     f'used: {why[image["reason"]]}. Checks that need the phase are assessed only where the coach '
                     'marks who had the ball.')
    else:
        notes.append('The possession engine did not name who had the ball on this footage, so checks that need the '
                     'phase are not assessed unless the coach marks who had the ball.')
    if coach is not None:
        if coach['spans']:
            notes.append(f'{coach["spans"]} coach-marked spans decide the phase for {coach["seconds"]:.0f} s'
                         + ('; elsewhere the estimate is used.' if engine else '.'))
        if coach['dropped']:
            notes.append(f'{coach["dropped"]} coach-marked spans were not used (outside the analysed video, '
                         'overlapping or incomplete).')
        side = SI.one_sided(coach, engine, team)
        if side is not None:
            missing = 'B' if side == 'A' else 'A'
            notes.append(f'You marked only Team {side}\'s possession; mark Team {missing}\'s spells (or dead balls) '
                         f'too to run the checks that need Team {missing} on the ball.')
    out = {'available': bool(available), 'reason': SI._ascii(reason, NOTE_LIMIT) if reason else None, 'level': level,
           'coverage': coverage, 'roughShares': rough, 'notes': [x[:NOTE_LIMIT] for x in notes], 'coach': coach}
    if image is not None:
        out['imageEstimate'] = image
    return out, {'engineSegments': n_engine, 'changes': changes}


def _p90(v):
    from soccer.coaching import situations as SI
    return round(min(float(np.percentile(v, 90)), 15.0), 2) if len(v) >= SI.P90_MIN_SAMPLES else None


def _players(parts, step, exact):
    """The merged timeline 'players': observed seconds summed (exactly, from the counts), the p90 speed of the pooled
    speed samples; without the parts' extra the lowest segment p90 (module docstring)."""
    out = {}
    pooled = {}
    for p in parts:
        ex = p.get('extra') or {}
        for pid, rec in p['obs']['timeline']['players'].items():
            cur = out.get(pid)
            if cur is None:
                cur = out[pid] = {'team': rec['team'], 'goalkeeper': bool(rec['goalkeeper']), '_n': 0,
                                  '_s': 0.0, '_p90': []}
            cur['goalkeeper'] = cur['goalkeeper'] or bool(rec['goalkeeper'])
            if exact:
                counts = (ex.get('keepers') if rec['goalkeeper'] else ex.get('seen')) or {}
                cur['_n'] += int(counts.get(pid, 0))
                if not rec['goalkeeper']:
                    pooled.setdefault(pid, []).append((ex.get('speeds') or {}).get(pid, np.zeros(0)))
            else:
                cur['_s'] += float(rec['observedSeconds'])
                if rec.get('p90SpeedMS') is not None:
                    cur['_p90'].append(float(rec['p90SpeedMS']))
    for pid, cur in out.items():
        if exact:
            seconds = round(cur['_n']*step, 2)
            v = np.concatenate(pooled[pid]) if pooled.get(pid) else np.zeros(0)
            p90 = None if cur['goalkeeper'] else _p90(v)
        else:
            seconds = round(cur['_s'], 2)
            p90 = min(cur['_p90']) if cur['_p90'] and not cur['goalkeeper'] else None
        out[pid] = {'team': cur['team'], 'goalkeeper': cur['goalkeeper'], 'observedSeconds': seconds,
                    'p90SpeedMS': p90}
    return out


def _footage(parts, exact):
    """The merged footage (module docstring): the run rules on the summed frames, observed players and anchors."""
    from soccer import labels
    from soccer.coaching import situations as SI
    fos = [p['obs']['timeline']['footage'] for p in parts]
    if not exact:
        frames = [int(p['obs']['segment']['frames']) for p in parts]
        share = sum(f['smallSceneShare']*n for f, n in zip(fos, frames))/max(1, sum(frames))
        kinds = Counter()
        for f, n in zip(fos, frames):
            kinds[f['kind']] += n
        return {'kind': kinds.most_common(1)[0][0], 'smallSceneShare': round(share, 3),
                'keyless': all(f['keyless'] for f in fos), 'lensBent': any(f['lensBent'] for f in fos)}
    ex = [p['extra'] for p in parts]
    frames, small = sum(e['frames'] for e in ex), sum(e['smallFrames'] for e in ex)
    small_run = bool(frames) and small >= labels.SMALL_SCENE_SHARE*frames
    observed = []
    for team in ('A', 'B'):
        total = sum(e['observed'].get(team, (0.0, 0))[0] for e in ex)
        count = sum(e['observed'].get(team, (0.0, 0))[1] for e in ex)
        if count:
            observed.append(total/count)
    kind = 'broadcast' if not small_run else ('veo' if observed and max(observed) < SI.FEW_PLAYERS else 'amateur')
    keyless = any(e['calibrated'] for e in ex) and any(e['anchorList'] for e in ex) and not any(e['landmarks']
                                                                                               for e in ex)
    return {'kind': kind, 'smallSceneShare': round(small/frames, 3) if frames else 0.0, 'keyless': bool(keyless),
            'lensBent': any(e['lens'] for e in ex)}


def _medians(parts, offsets, exact):
    """roleInputs.detect.medianY over the match: per player the median lateral position of his usable samples overall
    ('-1') and per formation period (shifted indices), from the pooled positions (contextual.role_geometry's rule:
    at least MEDIAN_MIN_SAMPLES samples)."""
    from soccer.coaching import contextual as CX
    if not exact:
        out = {}
        for p, off in zip(parts, offsets):
            for pid, m in ((p['obs']['roleInputs'].get('detect') or {}).get('medianY') or {}).items():
                for k, v in m.items():
                    key = k if k == '-1' else str(int(k)+off)
                    out.setdefault(pid, {}).setdefault(key, []).append((v, p['obs']['timeline']['samples']))
        return {pid: {k: round(sum(v*w for v, w in vs)/max(1, sum(w for _, w in vs)), 2) for k, vs in m.items()}
                for pid, m in out.items()}
    pooled = {}
    for p, off in zip(parts, offsets):
        for pid, (ys, per) in (p['extra'].get('lateral') or {}).items():
            per = np.where(per >= 0, per+off, -1)
            pooled.setdefault(pid, []).append((ys, per))
    out = {}
    for pid, chunks in pooled.items():
        ys = np.concatenate([c[0] for c in chunks])
        per = np.concatenate([c[1] for c in chunks])
        m = {}
        if len(ys) >= CX.MEDIAN_MIN_SAMPLES:
            m['-1'] = round(float(np.median(ys)), 2)
        for k in np.unique(per[per >= 0]):
            sel = per == k
            if sel.sum() >= CX.MEDIAN_MIN_SAMPLES:
                m[str(int(k))] = round(float(np.median(ys[sel])), 2)
        if m:
            out[pid] = m
    return out


def _merge_context(parts, offsets):
    ctxs = [p['obs']['roleInputs']['context'] for p in parts]
    ids = _union(c['ids'] for c in ctxs)
    track = {'t': [], 'x': [[] for _ in ids], 'y': [[] for _ in ids], 'phase': [], 'complete': [],
             'step': _same([c['track'].get('step') for c in ctxs], 'the role track step')}
    windows, periods = [], []
    for c, off in zip(ctxs, offsets):
        tr = c['track']
        k = len(tr.get('t') or [])
        col = {pid: j for j, pid in enumerate(c['ids'])}
        track['t'] += list(tr.get('t') or [])
        track['phase'] += list(tr.get('phase') or [])
        track['complete'] += list(tr.get('complete') or [True]*k)
        for j, pid in enumerate(ids):
            src = col.get(pid)
            track['x'][j] += list(tr['x'][src]) if src is not None and src < len(tr.get('x') or []) else [None]*k
            track['y'][j] += list(tr['y'][src]) if src is not None and src < len(tr.get('y') or []) else [None]*k
        for w in c.get('windows') or []:
            windows.append(dict(w, period=None if w.get('period') is None else int(w['period'])+off))
        for q in c.get('periods') or []:
            periods.append(dict(q, index=int(q['index'])+off))
    return {'version': _same([c['version'] for c in ctxs], 'the role inputs version'), 'team': ctxs[0]['team'],
            'format': _same([c['format'] for c in ctxs], 'the format'),
            'outfield': _same([c['outfield'] for c in ctxs], 'the outfield count'),
            'scale': copy.deepcopy(ctxs[0]['scale']), 'ids': ids,
            'segment': {'t0': ctxs[0]['segment']['t0'], 't1': ctxs[-1]['segment']['t1']}, 'track': track,
            'windows': windows, 'periods': periods}


PLAYER_COLUMNS = ('X', 'Y', 'vx')
OPP_COLUMNS = ('oX', 'oY')
BIT_COLUMNS = ('ok', 'taint')


def _merge_detect(parts, offsets, exact):
    geos = [p['obs']['roleInputs']['detect'] for p in parts]
    ids = _union(g['ids'] for g in geos)
    opp_ids = _union(g['oppIds'] for g in geos)
    full = any(g['samples'].get('oX') for g in geos)
    samples = {}
    for g, off in zip(geos, offsets):
        s = g['samples']
        n = len(s.get('t') or [])
        col = {pid: j for j, pid in enumerate(g['ids'])}
        ocol = {pid: j for j, pid in enumerate(g['oppIds'])}
        for name, values in s.items():
            if name in PLAYER_COLUMNS or name in OPP_COLUMNS:
                continue
            if name in BIT_COLUMNS:
                values = [''.join(bits[col[pid]] if pid in col and col[pid] < len(bits) else '0' for pid in ids)
                          for bits in values]
            elif name == 'period':
                values = [int(v)+off if int(v) >= 0 else int(v) for v in values]
            samples.setdefault(name, []).extend(values)
        for name in PLAYER_COLUMNS:
            cols = samples.setdefault(name, [[] for _ in ids])
            for j, pid in enumerate(ids):
                src = col.get(pid)
                cols[j] += list(s[name][src]) if src is not None and src < len(s.get(name) or []) else [None]*n
        for name in OPP_COLUMNS:
            cols = samples.setdefault(name, [[] for _ in opp_ids] if full else [])
            if not full:
                continue
            for j, pid in enumerate(opp_ids):
                src = ocol.get(pid)
                cols[j] += list(s[name][src]) if src is not None and src < len(s.get(name) or []) else [None]*n
    lost = {}
    for g in geos:
        for state, reasons in (g.get('lost') or {}).items():
            for reason, v in reasons.items():
                lost.setdefault(state, {})[reason] = lost.get(state, {}).get(reason, 0.0)+float(v)
    first = geos[0]
    return {'version': _same([g['version'] for g in geos], 'the detect inputs version'), 'team': first['team'],
            'format': first['format'], 'players': first['players'], 'outfield': first['outfield'],
            'scale': copy.deepcopy(first['scale']), 'step': first['step'], 'fps': first['fps'], 'ids': ids,
            'oppIds': opp_ids,
            'phaseSeconds': {k: round(sum(float(g['phaseSeconds'].get(k) or 0.0) for g in geos), 2)
                             for k in ('own', 'opp')},
            'lost': {state: {k: round(v, 2) for k, v in sorted(r.items())} for state, r in lost.items()},
            'medianY': _medians(parts, offsets, exact), 'stationM': first['stationM'], 'samples': samples,
            'lengthM': first['lengthM']}


def _merge_evidence(parts):
    evs = [p['obs']['roleInputs'].get('evidence') for p in parts if isinstance(p['obs']['roleInputs'].get('evidence'),
                                                                               dict)]
    if not evs:
        return None
    frames = {}
    for e in evs:
        frames.update(e.get('frames') or {})
    return dict(copy.deepcopy({k: v for k, v in evs[0].items() if k != 'frames'}), frames=frames)


def _boundary_formation_changes(merged, parts, offsets):
    """The formation-change suppression one long run would apply (module docstring): the label runs of the merged
    periods, joined across the cuts within each match period, under situations._formation's rule (both runs held
    FORMATION_PERSIST_S and read clear / likely or were set by the coach, no churn above FORMATION_MAX_RATE per 10
    minutes of the period's owned time). A run joined across a cut can hold long enough where neither segment's part
    of it did, and a change on a cut is seen by neither segment, so every change is judged again; every opportunity and
    stored role sample within FORMATION_CHANGE_S of a suppressing change gets formationChangeNear, on top of what the
    segments flagged (a suppressor only: or-ed, never cleared). Returns the windows."""
    from soccer import team_shape as ts
    from soccer.coaching import situations as SI
    periods = merged['roleInputs']['context'].get('periods') or []
    if not periods:
        return []
    owner = {}
    for k, (p, off) in enumerate(zip(parts, offsets)):
        for q in p['obs']['roleInputs']['context'].get('periods') or []:
            owner[int(q['index'])+off] = k
    by_match_period = {}
    for q in periods:
        k = owner.get(int(q['index']))
        if k is not None:
            by_match_period.setdefault(parts[k]['segment']['period'], []).append(q)
    windows = []
    for mp, qs in by_match_period.items():
        seconds = sum(float(p['obs']['timeline']['samples'])*float(p['obs']['timeline']['step'])
                      for p in parts if p['segment']['period'] == mp)
        runs = []
        for q in sorted(qs, key=lambda q: float(q['t0'])):
            lab = q.get('label')
            if lab not in ts.TEMPLATES:
                continue
            sure = q.get('confidence') in SI.FORMATION_CONFIDENT or bool(q.get('corrected'))
            if runs and runs[-1][0] == lab:
                runs[-1][2] = float(q['t1'])
                runs[-1][3] = runs[-1][3] or sure
            else:
                runs.append([lab, float(q['t0']), float(q['t1']), sure])
        churn = len(runs) > 1 and seconds > 0 and (len(runs)-1)*600.0/seconds > SI.FORMATION_MAX_RATE+1e-9
        for a, b in zip(runs, runs[1:]):
            held = a[2]-a[1] >= SI.FORMATION_PERSIST_S-1e-6 and b[2]-b[1] >= SI.FORMATION_PERSIST_S-1e-6
            if churn or not held or not (a[3] and b[3]):
                continue
            lo, hi = min(a[2], b[1]), max(a[2], b[1])
            windows.append((lo-SI.FORMATION_CHANGE_S, hi+SI.FORMATION_CHANGE_S))
    if not windows:
        return []

    def near(t):
        return any(w0-1e-6 <= t <= w1+1e-6 for w0, w1 in windows)
    for o in merged['opportunities']:
        if not o['intentSignals']['formationChangeNear'] and any(near(t) for t in o['relations']['t']):
            o['intentSignals']['formationChangeNear'] = True
    smp = merged['roleInputs']['detect']['samples']
    if smp.get('fcn'):
        smp['fcn'] = [bool(v) or near(t) for v, t in zip(smp['fcn'], smp['t'])]
    return [[round(a, 2), round(b, 2)] for a, b in windows]


def _coverage_summary(parts, merged_detectors):
    """roleInputs.match: per segment what the coverage notices state (owned range, the phase named, the possession
    source, the footage, how many opportunities each check found there, the calibration facts of the index)."""
    rows = []
    for p in parts:
        tl = p['obs']['timeline']
        pq = tl['phaseQuality']
        ps = tl['phaseSeconds']
        counts = Counter(o['detector'] for o in p['obs']['opportunities'])
        rows.append({'index': p['segment']['index'], 'period': p['segment']['period'],
                     't0': round(p['segment']['start'], 3), 't1': round(p['segment']['end'], 3),
                     'frames': int(p['obs']['segment']['frames']), 'samples': int(tl['samples']),
                     'named': round(float(ps['own'])+float(ps['opp']), 2), 'dead': float(ps['dead']),
                     'unknown': float(ps['unknown']),
                     'possession': 'possession-4' if pq.get('available') else
                     ('possession-amateur' if (pq.get('imageEstimate') or {}).get('used') else None),
                     'footage': tl['footage']['kind'], 'complete': tl['completeShare'],
                     'opportunities': {d: int(counts.get(d, 0)) for d in merged_detectors},
                     'keyframe': bool((p.get('extra') or {}).get('landmarks')) if p.get('extra') else
                     bool((p['segment'].get('quality') or {}).get('keyframeAnchors')),
                     'spans': bool(p['segment'].get('spans'))})
    return {'version': VERSION, 'segments': rows}


def merge_obs(parts, *, result_sha=None, shape_settings=None, team_settings=None, validate=True):
    """ONE obs file (schema 'obs') of the match from the segments' parts (measure_segments; any order: sorted by their
    owned start), merged by the module docstring's rules. result_sha: the match's measurement sha (measurement_key;
    by default the sha of the parts' own result shas); shape_settings: the team-shape settings the parts were measured
    with; team_settings: the team's coaching block (its format and field: the measurement settings). The fingerprint is
    coaching.measurement_inputs of those, as store.tier_keys computes it. Raises MergeError where the parts cannot be
    one file (other teams, formats, steps or frame rates; overlapping owned ranges; possession-4 beside a used
    image-based estimate)."""
    from soccer import coaching
    from soccer.coaching import contextual as CX, schema as S, situations as SI
    if not parts:
        raise MergeError('no segment to merge')
    parts = sorted(parts, key=lambda p: (p['segment']['start'], p['segment']['cut0']))
    for a, b in zip(parts, parts[1:]):
        if b['segment']['cut0'] < a['segment']['cut1']:
            raise MergeError(f'segments {a["segment"]["index"]} and {b["segment"]["index"]} own overlapping frames')
    obs_list = [p['obs'] for p in parts]
    team = _same([o['team'] for o in obs_list], 'the team')
    exact = all(isinstance(p.get('extra'), dict) and 'speeds' in p['extra'] for p in parts)
    tls = [o['timeline'] for o in obs_list]
    step = float(_same([tl['step'] for tl in tls], 'the sample step'))
    fps = _same([o['segment']['fps'] for o in obs_list], 'the frame rate')
    fmt = _same([tl['format'] for tl in tls], 'the format')
    measurement = _same([o['inputs']['measurement'] for o in obs_list], 'the measurement settings')
    shape_sha = _same([o['inputs']['shapeSettings'] for o in obs_list], 'the team-shape settings')
    if shape_settings is not None and coaching.shape_settings_sha(shape_settings) != shape_sha:
        raise MergeError('the parts were measured with other team-shape settings')
    if team_settings is not None and S.tier_settings(team_settings)['measurement'] != measurement:
        raise MergeError('the parts were measured with another format or field')
    # period indices shift with the concatenated formation periods
    offsets, total = [], 0
    for tl in tls:
        offsets.append(total)
        total += len(tl['formationPeriods'])
    touching = [pa['segment']['cut1'] == pb['segment']['cut0'] for pa, pb in zip(parts, parts[1:])]
    boundaries = [(a['frame1'], b['frame0']) for a, b, touch in zip(tls, tls[1:], touching) if touch]
    # opportunities, moments
    opps = [o for ob in obs_list for o in ob['opportunities']]
    opps.sort(key=lambda o: (o['t0'], S.DETECTORS.index(o['detector'])))
    if len({o['id'] for o in opps}) != len(opps):
        raise MergeError('two segments made the same opportunity id')
    moments = [m for ob in obs_list for m in ob['moments']]
    # timeline
    sec = {k: round(sum(float(tl['phaseSeconds'][k]) for tl in tls), 2)
           for k in ('own', 'opp', 'dead', 'unknown', 'nearChange')}
    spans = _join_runs([s for tl in tls for s in tl['phaseSpans']], boundaries,
                       lambda s: (s['state'], s.get('source')), step)
    breaks = []          # the time no segment owns (between periods): a 'dead' span without a source, one step inside
    for a, b, touch in zip(tls, tls[1:], touching):
        g0, g1 = round(float(a['t1'])+step, 3), round(float(b['t0'])-step, 3)
        if not touch and g1 >= g0-1e-9:
            breaks.append({'state': 'dead', 't0': g0, 't1': g1, 'frame0': int(a['frame1'])+1,
                           'frame1': max(int(a['frame1'])+1, int(b['frame0'])-1)})
    spans = sorted(spans+breaks, key=lambda s: (s['t0'], s['t1']))
    samples = sum(int(tl['samples']) for tl in tls)
    footage = _footage(parts, exact)
    pq, pfacts = _phase_quality(parts, sec, footage['kind'])
    if exact:
        ex = [p['extra'] for p in parts]
        n = max(1, samples)
        lineups = Counter()
        for e in ex:
            for k, c in enumerate(e['outfield']):
                lineups[k] += c
        lineup = max(sorted(lineups), key=lambda k: lineups[k]) if lineups else None
        in_kit, conf, posd = (sum(e[k] for e in ex)/n for k in ('inKit', 'confirmed', 'positioned'))
        dq = {'inKit': _r(in_kit, 1), 'confirmed': _r(conf, 1), 'positioned': _r(posd, 1),
              'lineup': max(1, min(11, lineup)) if lineup is not None else None,
              'dominantCause': SI._dominant_cause(lineup, in_kit, conf, posd) if samples else None}
        complete_share = round(sum(e['complete'] for e in ex)/n, 3) if samples else 0.0
        observed_mean = round(sum(e['positioned'] for e in ex)/n, 2) if samples else 0.0
        ball_share = round(sum(e['ball'] for e in ex)/n, 3) if samples else 0.0
        opp_n = sum(e['opp'] for e in ex)
        carrier = round(sum(e['oppCarrier'] for e in ex)/opp_n, 3) if opp_n else None
    else:
        # no per-sample counts (an obs made elsewhere): the segments' values weighted by their samples (approximate:
        # rounded inputs; the lineup the sample-weighted mode of the segments' modes)
        weights = [int(tl['samples']) for tl in tls]

        def wmean(values, d):
            pairs = [(v, k) for v, k in zip(values, weights) if v is not None]
            total = sum(k for _, k in pairs)
            return round(sum(v*k for v, k in pairs)/total, d) if pairs and total else None
        lineups = Counter()
        for tl, k in zip(tls, weights):
            if tl['dataQuality']['lineup'] is not None:
                lineups[tl['dataQuality']['lineup']] += k
        lineup = lineups.most_common(1)[0][0] if lineups else None
        in_kit, conf, posd = (wmean([tl['dataQuality'][k] for tl in tls], 1) for k in ('inKit', 'confirmed',
                                                                                       'positioned'))
        known = None not in (lineup, in_kit, conf, posd)
        dq = {'inKit': in_kit, 'confirmed': conf, 'positioned': posd, 'lineup': lineup,
              'dominantCause': SI._dominant_cause(lineup, in_kit, conf, posd) if known else None}
        complete_share = wmean([tl['completeShare'] for tl in tls], 3) or 0.0
        observed_mean = wmean([tl['observedMean'] for tl in tls], 2) or 0.0
        ball_share = wmean([tl['ballReliableShare'] for tl in tls], 3) or 0.0
        opp = [(tl['carrierShare'], float(tl['phaseSeconds']['opp'])) for tl in tls if tl['carrierShare'] is not None]
        carrier = round(sum(c*s for c, s in opp)/sum(s for _, s in opp), 3) if opp and sum(s for _, s in opp) else None
    scales = [tl['scale'] for tl in tls]
    scale = copy.deepcopy(scales[0])
    if any(json.dumps(s, sort_keys=True) != json.dumps(scale, sort_keys=True) for s in scales[1:]):
        kinds = Counter()
        for s, tl in zip(scales, tls):
            kinds[json.dumps({k: s[k] for k in ('lengthM', 'widthM')}, sort_keys=True)] += int(tl['samples'])
        scale = dict(json.loads(kinds.most_common(1)[0][0]),
                     source='assumed' if any(s['source'] == 'assumed' for s in scales) else
                     ('measured' if any(s['source'] == 'measured' for s in scales) else 'declared'),
                     metresUsable=all(s['metresUsable'] for s in scales))
    periods = []
    for tl, off in zip(tls, offsets):
        periods += [dict(q, index=int(q['index'])+off) for q in tl['formationPeriods']]
    timeline = {'team': team, 'samples': samples, 'step': step, 't0': tls[0]['t0'], 't1': tls[-1]['t1'],
                'frame0': tls[0]['frame0'], 'frame1': tls[-1]['frame1'], 'fps': float(tls[0]['fps']), 'format': fmt,
                'phaseSeconds': sec, 'phaseSpans': spans, 'phaseQuality': pq, 'footage': footage, 'dataQuality': dq,
                'scale': scale, 'completeShare': complete_share, 'observedMean': observed_mean,
                'ballReliableShare': ball_share, 'carrierShare': carrier, 'formationPeriods': periods,
                'players': _players(parts, step, exact)}
    # detectors: coverage recomputed from the merged opportunities and the summed seconds
    names = [d for d in S.DETECTORS if any(d in ob['detectors'] for ob in obs_list)]
    detectors = {}
    for d in names:
        covs = [ob['detectors'][d] for ob in obs_list if d in ob['detectors']]
        phase = [c['phaseSeconds'] for c in covs if c['phaseSeconds'] is not None]
        abstain = Counter()
        for c in covs:
            abstain.update({k: float(v) for k, v in c['abstain'].items()})
        detectors[d] = CX.coverage(d, _same([c['tier'] for c in covs], f'{d} tier'),
                                   [o for o in opps if o['detector'] == d], step, sum(phase) if phase else None,
                                   dict(abstain))
    tiers = _join_runs([r for ob in obs_list for r in ob['frameTiers']], boundaries, lambda r: r['tier'], step)
    # the measurement key
    if result_sha is None:
        result_sha = coaching.fingerprint({'kind': 'fullmatch-coaching-parts', 'version': VERSION,
                                           'parts': [ob['inputs']['result'] for ob in obs_list]})
    block = team_settings if team_settings is not None else {'format': measurement['format'],
                                                             'field': measurement['field']}
    mfp = coaching.fingerprint(coaching.measurement_inputs(result_sha, shape_sha, block, team))
    segment = {'t0': min(float(o['segment']['t0']) for o in obs_list),
               't1': max(float(o['segment']['t1']) for o in obs_list), 'fps': fps,
               'frames': sum(int(o['segment']['frames']) for o in obs_list)}
    merged = {'kind': 'coaching-observations', 'version': S.SCHEMA_VERSION, 'team': team,
              'engine': {'version': S.ENGINE_VERSION, 'measurement': mfp[:16]}, 'fingerprint': mfp,
              'inputs': {'result': result_sha, 'shapeSettings': shape_sha, 'measurement': measurement},
              'segment': segment, 'video': copy.deepcopy(obs_list[0]['video']), 'timeline': timeline,
              'detectors': detectors, 'opportunities': opps, 'moments': moments, 'frameTiers': tiers,
              'roleInputs': {'context': _merge_context(parts, offsets), 'detect': _merge_detect(parts, offsets, exact)}}
    evidence = _merge_evidence(parts)
    if evidence is not None:
        merged['roleInputs']['evidence'] = evidence
    summary = _coverage_summary(parts, names)
    summary.update(exact=exact, breaks=[[s['t0'], s['t1']] for s in breaks], changes=pfacts['changes'],
                   engineSegments=pfacts['engineSegments'],
                   formationChanges=_boundary_formation_changes(merged, parts, offsets))
    merged['roleInputs']['match'] = summary
    if validate:
        S.check('obs', merged)
    return merged


# ---------- the report ----------
def _join_stretches(rows, want):
    """[(t0, t1)] of consecutive segments (touching owned ranges) where want(row) holds."""
    out = []
    for r in rows:
        if not want(r):
            continue
        if out and abs(out[-1][1]-r['t0']) < 1e-3:
            out[-1][1] = r['t1']
        else:
            out.append([r['t0'], r['t1']])
    return out


def match_notices(obs, profile, rng=None):
    """The coverage notices of a match report (schema Notice, validated as report.py validates its own notices:
    explain.validate_free with the facts they state): the segments and minutes the report joins and the breaks it
    leaves out, where who had the ball was named (the stretches) or that it was named nowhere, and how many segments
    had no pitch keyframe. rng: the report's range (its length is the duration stated when no break is left out, as
    the overview states it). A text the level's gate refuses is left out."""
    from soccer.coaching import explain as EX, language as L, report as RP
    summary = (obs.get('roleInputs') or {}).get('match') or {}
    rows = summary.get('segments') or []
    if not rows:
        return []
    out = []

    def add(code, text, facts):
        if text and not EX.validate_free(text, profile, facts, numbers=False, limit=NOTICE_LIMIT):
            out.append({'code': code, 'text': text})
    n = len(rows)
    breaks = summary.get('breaks') or []
    played = _played_seconds(obs)
    if not breaks and isinstance((rng or {}).get('seconds'), (int, float)):
        played = float(rng['seconds'])
    first, last = rows[0]['t0'], rows[-1]['t1']
    periods = len({r['period'] for r in rows})
    text = (f'This report joins the {n} analysed segments of the match: {RP.duration_words(played)} of play from '
            f'{L.fmt_time(first)} to {L.fmt_time(last)}')
    facts = [{'seconds': played}, n, periods, {'t': first}, {'t': last}]
    gap = sum(max(0.0, float(b['t0'])-float(a['t1'])) for a, b in zip(rows, rows[1:]))
    if breaks and gap > 0:
        text += f', in {periods} periods; the {L.fmt_duration(gap)} between them is not analysed.'
        facts.append({'seconds': gap})
    else:
        text += '.'
    add('match_segments', text, RP._facts(*facts))
    named = [r for r in rows if r['named'] > 0]
    if not named:
        add('match_phase', f'Who had the ball was not named in any of the {n} segments, so no check that depends on it '
                           'was made anywhere in the match.', RP._facts(n))
    elif len(named) < n:
        stretches = _join_stretches(rows, lambda r: r['named'] > 0)
        shown = ', '.join(f'{L.fmt_time(a)} to {L.fmt_time(b)}' for a, b in stretches[:MAX_SPANS_NAMED])
        more = len(stretches)-MAX_SPANS_NAMED
        seconds = sum(r['named'] for r in named)
        text = (f'Who had the ball was named in {len(named)} of the {n} segments ({L.fmt_duration(seconds)} in all): '
                f'the checks that depend on it come only from {shown}' + (f' and {more} more stretches' if more > 0
                                                                          else '') + '.')
        facts = RP._facts(len(named), n, {'seconds': seconds}, *({'t': x} for s in stretches[:MAX_SPANS_NAMED]
                                                                 for x in s), *((more,) if more > 0 else ()))
        add('match_phase', text if len(text) <= NOTICE_LIMIT else
            f'Who had the ball was named in {len(named)} of the {n} segments ({L.fmt_duration(seconds)} in all): the '
            'checks that depend on it come only from those.', facts)
    keyless = [r for r in rows if not r['keyframe']]
    if keyless:
        add('match_calibration', (f'{len(keyless)} of the {n} segments had no pitch keyframe of their own: the player '
                                  'positions there rest on the automatic pitch registration, which is less sure.'),
            RP._facts(len(keyless), n))
    if summary.get('exact') is False:
        add('match_estimates', 'Some match totals are combined from segment summaries (a part without its sample '
                               'counts), so they are approximate.', None)
    return out


def _played_seconds(obs, lo=None, hi=None):
    """The owned seconds of the match (inside [lo, hi] when given): the segments' owned ranges, the breaks left out."""
    rows = ((obs.get('roleInputs') or {}).get('match') or {}).get('segments') or []
    if lo is None and hi is None:
        return sum(float(r['samples']) for r in rows)*float(obs['timeline']['step'])
    lo = -math.inf if lo is None else float(lo)
    hi = math.inf if hi is None else float(hi)
    return sum(max(0.0, min(float(r['t1']), hi)-max(float(r['t0']), lo)) for r in rows)


def _overview_played(report, obs, profile):
    """The overview's "This report covers N of play." with the owned seconds when the range spans a break between
    periods (interpret states the range's length); unchanged otherwise or when the level's gate refuses the text."""
    from soccer.coaching import explain as EX, report as RP
    ov = report.get('overview') or {}
    text, rng = ov.get('text'), report.get('range') or {}
    if not isinstance(text, str) or not isinstance(rng.get('seconds'), (int, float)):
        return
    played = _played_seconds(obs) if rng.get('whole') else _played_seconds(obs, rng.get('t0'), rng.get('t1'))
    if played >= float(rng['seconds'])-1.0:
        return
    old = f'This report covers {RP.duration_words(rng["seconds"])} of play.'
    new = f'This report covers {RP.duration_words(played)} of play (the breaks between periods are left out).'
    if text.startswith(old) and not EX.validate_free(new, profile, RP._facts({'seconds': played}), limit=200,
                                                     short=played < EX.SHORT_RANGE_S):
        ov['text'] = new+text[len(old):]


def finish_report(report, obs, roles):
    """In place: the match's coverage notices added to the report's overview and its "covers N of play" made the
    owned seconds; the report re-validated (schema 'report' and cross_check), the additions dropped if it does not
    validate with them. Returns the report."""
    from soccer.coaching import schema as S
    before = copy.deepcopy(report.get('overview'))
    profile = report['profile']
    _overview_played(report, obs, profile)
    have = {n['code'] for n in report['overview']['notices']}
    rng = report.get('range') or {}
    report['overview']['notices'] += [n for n in match_notices(obs, profile, rng if rng.get('whole') else None)
                                      if n['code'] not in have]
    if S.validate('report', report)+S.cross_check(report, obs, roles):
        report['overview'] = before
    return report


def _meta(index):
    key = str((index.get('source') or {}).get('key') or '')[:16] or None
    return {'run': run_name(index), 'video': {'key': key, 'name': (index.get('video') or {}).get('name')}}


def _measure_lock(a):
    with _lock:
        return _measure_locks.setdefault(str(a), threading.Lock())


def _record(a, keys, options, report, obs):
    """Store a match report as a store record (store.save_report_v2's layout, so load_record_v2, find_report_v2,
    staleness_v2 and served_report read it) with the match's coverage summary beside the report ('match')."""
    from datetime import datetime
    from soccer.coaching import store
    rows = (obs.get('roleInputs') or {}).get('match') or {}
    record = {'version': store.RECORD_VERSION, 'kind': 'coaching-report-record',
              'keys': {'measurement': keys['mfp'], 'roles': keys['rfp'], 'interpretation': keys['ifp']},
              'options': dict(options), 'inputs': keys['inputs'], 'corrections': None,
              'saved': datetime.now().isoformat(timespec='seconds'), 'report': report,
              'match': {'segments': rows.get('segments') or [], 'breaks': rows.get('breaks') or [],
                        'exact': rows.get('exact'), 'playedSeconds': round(_played_seconds(obs), 2),
                        'formationChanges': rows.get('formationChanges') or []}}
    store.save_tier(a, 'report', keys, record)
    return record


def ensure_report(folder, keys, options, settings, index, *, obs=None, timings=None):
    """The fresh report record of keys from the stored (or given) merged obs: the roles file built when missing, the
    report interpreted, finished (finish_report) and stored. None when the obs file is not stored."""
    from soccer import coaching
    from soccer.coaching import store
    a = assembly_dir(folder)
    timings = timings if timings is not None else {}
    with store.tier_lock(a, keys['team']):
        record = store.load_record_v2(store.tier_path(a, 'report', keys))
        if record is not None:
            return record
        obs = obs if obs is not None else store.load_tier(store.tier_path(a, 'obs', keys))
        if obs is None:
            return None
        if obs.get('fingerprint') != keys['mfp']:
            raise store.StoreError('The stored match observations do not match their key; build the report again.')
        t = time.time()
        roles = store.load_tier(store.tier_path(a, 'roles', keys))
        if roles is None or roles.get('fingerprint') != keys['rfp'] or roles.get('measurement') != keys['mfp']:
            roles = coaching.build_roles(obs, coaching=settings or {}, team=keys['team'])
            store.save_tier(a, 'roles', keys, roles)
        timings['roles'] = round(time.time()-t, 3)
        t = time.time()
        rng = keys['range']
        report = coaching.interpret(obs, roles, coaching=settings or {}, team=keys['team'], t0=rng['t0'], t1=rng['t1'],
                                    meta=_meta(index), generated=None)
        got = report['fingerprints']
        if (got['measurement'], got['roles'], got['interpretation']) != (keys['mfp'], keys['rfp'], keys['ifp']):
            raise store.StoreError('The match report was computed from other inputs than its key names; it was not '
                                   'stored.')
        finish_report(report, obs, roles)
        timings['interpret'] = round(time.time()-t, 3)
        return _record(a, keys, options, report, obs)


def build_match_report(folder, index=None, team='A', settings=None, *, video_dir=None, shape_settings=None, spans=None,
                       t0=None, t1=None, age=None, level=None, style=None, federation=None, force=False,
                       progress=None, possession=None):
    """One team's match report: {'record' (the stored record; record['report'] the report), 'keys', 'cached'
    (the report was already stored), 'measured' (the segments were measured now), 'timings' (seconds per step)}.
    settings: the coaching settings (match_settings(video_dir, folder) by default); age / level / style / federation:
    replace the team's (with_profile); t0 / t1: a report range (absolute seconds; None for the whole match). The merged
    obs of BOTH teams is measured and stored when this team's is missing (one pass), the roles file and the report
    from it. force: measure again and rebuild every tier of these keys."""
    from soccer.coaching import schema as S, store
    started = time.time()
    timings = {}
    a = assembly_dir(folder)
    index = index or load_index(folder)
    if settings is None:
        settings, _ = match_settings(video_dir, folder)
    if any(v is not None for v in (age, level, style, federation)):
        settings = with_profile(settings, team, age=age, level=level, style=style, federation=federation)
    shape = shape_settings_of(index) if shape_settings is None else shape_settings
    fixes = fixes_of(folder)
    measured = measurement_key(folder, index, spans, fixes)
    keys = match_keys(folder, index, team, settings, shape_settings=shape, measured=measured, t0=t0, t1=t1)
    options = {'team': team, 't0': keys['range']['t0'], 't1': keys['range']['t1']}
    if force:
        for kind in ('report', 'roles', 'obs'):
            store.tier_path(a, kind, keys).unlink(missing_ok=True)
    record = store.load_record_v2(store.tier_path(a, 'report', keys))
    if record is not None:
        return {'record': record, 'keys': keys, 'cached': True, 'measured': False,
                'timings': {'total': round(time.time()-started, 3)}}
    did_measure = False
    obs = None
    with _measure_lock(a):
        if force or not store.tier_path(a, 'obs', keys).exists():
            t = time.time()
            parts = measure_segments(folder, index, None, settings, shape_settings=shape, spans=spans, fixes=fixes,
                                     progress=progress, possession=possession)
            timings['measure'] = round(time.time()-t, 2)
            timings['segments'] = [p['timings'] for p in parts['A']]
            t = time.time()
            for tm in ('A', 'B'):
                k_tm = match_keys(folder, index, tm, settings, shape_settings=shape, measured=measured)
                merged = merge_obs(parts[tm], result_sha=measured[0], shape_settings=shape,
                                   team_settings=S.team_block(settings, tm) or {})
                if merged['fingerprint'] != k_tm['mfp']:
                    raise store.StoreError('The merged observations do not match their key; nothing was stored.')
                store.save_tier(a, 'obs', k_tm, merged)
                if tm == team:
                    obs = merged
            timings['merge'] = round(time.time()-t, 3)
            del parts
            did_measure = True
    if obs is None:
        obs = store.load_tier(store.tier_path(a, 'obs', keys))
    if t0 is not None or t1 is not None:      # the range clamped to the merged file's own segment, as interpret does
        keys = match_keys(folder, index, team, settings, shape_settings=shape, measured=measured, t0=t0, t1=t1,
                          segment={'t0': obs['segment']['t0'], 't1': obs['segment']['t1']})
        options = {'team': team, 't0': keys['range']['t0'], 't1': keys['range']['t1']}
    record = ensure_report(folder, keys, options, settings, index, obs=obs, timings=timings)
    timings['total'] = round(time.time()-started, 2)
    return {'record': record, 'keys': keys, 'cached': False, 'measured': did_measure, 'timings': timings}


# ---------- what the routes call ----------
def _remember(key, error):
    with _lock:
        if len(_errors) >= MAX_ERRORS:
            _errors.pop(next(iter(_errors)))
        _errors[key] = error


def build_status(folder=None):
    """The running background build ({'folder', 'team', 'mfps', 'started', 'done', 'total', 'stage'}) of this match
    (or of any match with folder None), else None."""
    with _lock:
        if folder is None:
            return copy.deepcopy(next(iter(_builds.values()), None))
        return copy.deepcopy(_builds.get(str(assembly_dir(folder))))


def report_state(folder, index=None, team='A', settings=None, *, video_dir=None, t0=None, t1=None,
                 shape_settings=None, spans=None, make=True):
    """The coaching panel's answer for a match, shaped like server.v2_answer's for a run: {engine 2, state missing |
    building | ready | stale | error, tiers {measurement, role, interpretation}, report (store.served_report: generated
    = when stored, progression None), inferred (the team's measured roles once its roles file is stored), feedback [],
    reasons, reasonCodes, busy (another match's build holds the slot), error, options {team, t0, t1}, saved, run
    ('match-<id>'), drills, fingerprints, coachSpans, match {matchId, segments, progress (while building), coverage
    (the record's per-segment summary)}}. With the merged obs stored, a missing roles file and report are made here at
    once (make=True; tens of milliseconds, as a level or role change on a run); a missing obs file needs start_build."""
    from soccer.coaching import store
    a = assembly_dir(folder)
    index = index or load_index(folder)
    if settings is None:
        settings, _ = match_settings(video_dir, folder)
    keys = match_keys(folder, index, team, settings, shape_settings=shape_settings, spans=spans, t0=t0, t1=t1)
    options = {'team': team, 't0': keys['range']['t0'], 't1': keys['range']['t1']}
    record, fresh = store.find_report_v2(a, keys, options)
    tiers = store.tier_states(a, keys)
    status = build_status()
    mine = status is not None and status['folder'] == str(a)
    building = bool(mine and (status.get('mfps') or {}).get(team) == keys['mfp'])
    with _lock:
        error = _errors.get(keys['ifp']) or _errors.get(keys['mfp'])
    if building:
        tiers['measurement'] = 'building'
    if make and not fresh and not building and not error and tiers['measurement'] == 'ready':
        try:
            made = ensure_report(folder, keys, options, settings, index)
        except Exception as failure:   # noqa: BLE001 - reported to the page, remembered until the next build
            traceback.print_exc()
            error = f'The match coaching report could not be built: {failure}'
            _remember(keys['ifp'], error)
        else:
            if made is not None:
                record, fresh = made, True
                tiers = store.tier_states(a, keys)
    state = 'building' if building else 'ready' if fresh else 'error' if error else 'stale' if record else 'missing'
    reasons = store.staleness_v2(record['inputs'], keys['inputs']) if record and not fresh else []
    report = store.served_report(record) if record else None
    inferred = None
    if tiers['role'] == 'ready':
        inferred = store.measured_roles(store.load_tier(store.tier_path(a, 'roles', keys)),
                                        obs=lambda: store.load_tier(store.tier_path(a, 'obs', keys)))
    return {'engine': 2, 'state': state, 'tiers': tiers, 'report': report, 'inferred': inferred, 'feedback': [],
            'reasons': [r['text'] for r in reasons], 'reasonCodes': [r['code'] for r in reasons],
            'busy': bool(status is not None and not building), 'error': error if state == 'error' else None,
            'options': options, 'saved': record['saved'] if record else None, 'run': run_name(index),
            'drills': store.drills_of(report) if report else {},
            'fingerprints': {'measurement': keys['mfp'], 'roles': keys['rfp'], 'interpretation': keys['ifp']},
            'coachSpans': spans is not None,
            'match': {'matchId': index.get('matchId'), 'segments': len(index.get('segments') or []),
                      'progress': {k: status.get(k) for k in ('done', 'total', 'stage', 'started')} if mine else None,
                      'coverage': (record or {}).get('match')}}


def start_build(folder, index=None, team='A', settings=None, *, video_dir=None, t0=None, t1=None, shape_settings=None,
                spans=None, force=False, on_done=None):
    """Start building one team's match report on a background thread (both teams' merged obs when missing, then this
    team's roles file and report), one match build at a time: {'ok', 'state' 'building' | 'ready' | 'busy',
    'fingerprints', 'error'}. on_done(result or None, error or None) runs on the thread when it ends."""
    a = assembly_dir(folder)
    index = index or load_index(folder)
    if settings is None:
        settings, _ = match_settings(video_dir, folder)
    keys = match_keys(folder, index, team, settings, shape_settings=shape_settings, spans=spans, t0=t0, t1=t1)
    prints = {'measurement': keys['mfp'], 'roles': keys['rfp'], 'interpretation': keys['ifp']}
    from soccer.coaching import store
    if not force and store.load_record_v2(store.tier_path(a, 'report', keys)) is not None:
        return {'ok': True, 'state': 'ready', 'fingerprints': prints, 'error': None}
    measured = measurement_key(folder, index, spans)
    mfps = {tm: match_keys(folder, index, tm, settings, shape_settings=shape_settings, measured=measured)['mfp']
            for tm in ('A', 'B')}
    with _lock:
        if _builds:
            return {'ok': False, 'state': 'busy', 'fingerprints': prints,
                    'error': 'A match coaching report is being built: wait until it has finished.'}
        status = {'folder': str(a), 'team': team, 'mfps': mfps, 'started': time.time(), 'done': 0,
                  'total': len(index.get('segments') or []), 'stage': 'Starting'}
        _builds[str(a)] = status
        for k in (keys['ifp'], keys['mfp']):
            _errors.pop(k, None)

    def progress(done, total, stage):
        with _lock:
            status.update(done=done, total=total, stage=stage)

    def work():
        result, error = None, None
        try:
            result = build_match_report(folder, index, team, settings, shape_settings=shape_settings, spans=spans,
                                        t0=t0, t1=t1, force=force, progress=progress)
        except Exception as failure:   # noqa: BLE001 - reported to the page
            traceback.print_exc()
            error = f'The match coaching report could not be built: {failure}'
            _remember(keys['mfp'], error)
        finally:
            with _lock:
                _builds.pop(str(a), None)
        if on_done is not None:
            on_done(result, error)
    threading.Thread(target=work, name='match-coaching', daemon=True).start()
    return {'ok': True, 'state': 'building', 'fingerprints': prints, 'error': None}


def wait(timeout=None):
    """Block until no match build runs (tools and tests); True when none runs."""
    end = None if timeout is None else time.time()+timeout
    while True:
        with _lock:
            if not _builds:
                return True
        if end is not None and time.time() > end:
            return False
        time.sleep(.05)


__all__ = ['match_settings', 'with_profile', 'settings_dir', 'shape_settings_of', 'fixes_of', 'segment_spans',
           'measurement_key', 'match_keys', 'owned_result', 'apply_fixes', 'owned_possession', 'measure_segment',
           'measure_segments', 'merge_obs', 'MergeError', 'match_notices', 'finish_report', 'ensure_report',
           'build_match_report', 'report_state', 'start_build', 'build_status', 'wait', 'run_name', 'VERSION']
