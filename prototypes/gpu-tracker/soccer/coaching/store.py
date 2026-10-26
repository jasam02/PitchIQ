"""Where the coaching report lives between requests: a fingerprint of everything a report is computed from, the
cached report files, the coach's feedback on the findings, and the screenshots of the evidence frames.

Why a content fingerprint and not the file's modification time. A correction rewrites result.json in place (the run
folder keeps its name) and undoing it restores the original bytes exactly: on a scratch copy of the Veo run the sha was
a565edcd7c33 before a correction, 6511a4936410 with it and a565edcd7c33 again after the undo, while the modification
time changed both times. Keyed on the bytes, an undo finds the earlier report again, and a copy of the run folder (the
cloud plan) keeps its cache. Hashing is cheap next to parsing: 0.07 s for the 110.9 MB broadcast 42-222 s result
against 1.07 s for json.loads; the server remembers (path, mtime, size) -> sha, so a status poll hashes nothing.

The fingerprint covers the engine (ENGINE_VERSION and the bytes of the coaching code and of every soccer/ module it
reads, as loaded by this process: team_shape and analytics, and labels, pitch_model, registration, field and geometry,
which the series and the evidence tiers are computed with), the professional reference and the drill library (their
bytes), the result (its bytes), the video's team-shape settings (attacking direction, lineups and the coach's formation
labels change what the detectors see) and the report options (team, range, context level). The coach's feedback is not
in it: a verdict on a finding never makes the report recompute; it is merged in when the report is served or exported.

Feedback belongs to one run. Observation ids leave the run out and team letters are arbitrary per run (the verifier's
r1 verdict "Incorrect" printed on run r2's finding by id alone), and every observation of a detector shares the
analysed range's window (an "incorrect" on team width reattached to team area by detector + team + window). So a verdict
stores the finding's pattern, comparison context and players with its run, and reattaches only to a finding of the same
run: by its id, else by the same detector, team, pattern and players over a window sharing at least half of each (a
regeneration that moves the period slightly). Verdicts from other runs are listed apart, as from an earlier run, never
on a finding.

Nothing non-finite reaches the page: Flask's jsonify writes NaN, which the page's JSON.parse rejects (a NaN t0
hand-edited into the feedback file broke /api/coaching), so a NaN or Infinity token reads as null in the feedback file and
makes a report file malformed, and json_text() is the routes' last guard.

A report whose fingerprint no longer matches is still served, marked stale, with the reasons found by comparing the
stored inputs with the current ones ("2 corrections made since this report", "the team-shape settings changed", "the
report engine was updated"), and the page offers to rebuild it. Staleness is judged from the result's bytes, never
from corrections.json: removing a correction made on another run does not rebuild that run, and corrections reach
later runs only by seeding, never through their result file.

v2 (adaptive coaching, DESIGN v2 1.2): three cache tiers instead of one fingerprint. Why: the v1 key held the level,
so a level change rebuilt everything from the result, and re-reading a result is the expensive part. Measured on the
five run copies (CPU, Oct 22): measuring both teams (result parsed, series, possession-4, situations, detectors,
moments) takes 0.13 s (Veo 30 s), 0.79 s (amateur 64 s), 1.40 s (broadcast 60 s), 2.25 s (broadcast 81 s) and 2.94 s
plus 0.95 s of json.loads (broadcast 180 s, 106 MB); from the stored obs file the role tier and the report take
14-101 ms, and a level change (interpretation only) 11-33 ms. So the measurement tier (obs-<team>-<mfp16>.json: result
bytes, team-shape settings, the team's format and field, this run's coach spans) is built on the worker thread, once
for both teams (one series and possession pass); the role tier (roles-..-<rfp16>.json: confirmed roles, intended
formation) and the interpretation (report-..-<ifp16>.json: age, level, style, federation, range, rules and drill data)
are made when asked, from the obs file alone. The keys are coaching.measurement_inputs / role_inputs /
interpretation_inputs (LEAD), with the coach spans folded into the result sha (measured_result_sha) because LEAD's
measurement key has no field for them yet: a span edit then makes every tier of both teams stale, as a measurement
input must, and a run without spans keeps the key it had. obs files are 9-397 KB on the run copies, roles 1-49 KB,
reports 18-54 KB; KEEP_TIERS keeps 4 / 8 / 24 per team (critic S14: v1's 4 reports were too few once every level,
range and role change is its own report). Invariant 2 holds by construction: nothing but the measurement inputs reaches
an obs file, and a profile change never rewrites one (tested).

v2 settings, spans and feedback are kept per video apart from the tracking, like corrections.json: the coach's settings
(coaching_settings.json, DESIGN 2.1, cleaned by profiles.clean_team and refused rather than silently trimmed: a coach who
saved a role must not find it dropped), the coach-marked possession spans (coach_spans.json, per run because team
letters belong to a run; cleaned against the segment exactly as situations.build_timeline cleans them, so what is
stored is what is measured), the match date for progress, and the feedback file, now version 2: v2 verdicts are keyed
by finding key, run, team and range (critic C1/C2/S2/S3: a v1 verdict reattached across ranges and to findings not
shown) and the coach's adjusted suggested areas sit in its 'adjustments' list (DESIGN 10); every writer reads the file
strictly and writes both lists back, so neither engine drops the other's entries. None of these is in a fingerprint
except the spans and the settings fields each tier reads.

Files (runs/ is gitignored):
  runs/<video>/<run>/coaching/report-<team>-<fp16>.json   v1: one per fingerprint, the newest KEEP_REPORTS per team kept
  runs/<video>/<run>/coaching/v2/obs-<team>-<mfp16>.json, roles-<team>-<mfp16>-<rfp16>.json,
                     report-<team>-<mfp16>-<rfp16>-<ifp16>.json   v2 tiers (v1's pruning globs coaching/ only)
  runs/<video>/coaching_feedback.json                    per video, like corrections.json and team_shape.json
  runs/<video>/coaching_settings.json, coach_spans.json, coaching_match.json   v2, per video
  runs/_coaching/squads.json, progress.json              squads and progress (soccer/coaching/progress.py)
  runs/<video>/coaching/frames/f<frame>-w<width>.jpg      the raw pixels of an evidence frame, shared by every run of
                                                         the video (the folder name pins the video's content). No
                                                         overlay is baked in: display labels renumber after
                                                         corrections, so overlays are drawn when shown.
Writes are atomic (a uuid temporary file and os.replace, retried while a Windows reader holds the old file, as
corrections._write does). The feedback, settings and spans files are read strictly before every write: an unreadable or
unparsable file is an error, never an empty list, so a transient read failure cannot wipe a coach's earlier entries
(corrections.load returns [] on OSError; that is the pattern avoided here).
"""
import copy
import hashlib
import json
import math
import os
import re
import threading
import time
import uuid
from datetime import date, datetime
from pathlib import Path

from . import episodes
# The measurement plumbing lives in measuring.py, which the measurement tier's fingerprint hashes (critic S-12):
# re-exported here, so store.measure, store.build_role_tier and the rest keep working for every caller.
from .measuring import (build_role_tier, measure, measured_result_sha, measurement_stages, spans_sha,  # noqa: F401
                        stage)

STORE_VERSION = 1
KEEP_REPORTS = 4            # report files kept per run and team: the current one, the one from before a correction (an
                            # undo finds it again) and a couple of other ranges or context levels
CONTEXTS = ('development', 'competitive', 'advanced')
DEFAULT_CONTEXT = 'competitive'
VERDICTS = ('correct', 'partial', 'incorrect', 'insufficient')
MAX_NOTE = 2000             # characters of a feedback note (the request itself is limited by server.MAX_JSON, 512 KB)
MIN_RANGE_SECONDS = 5.0     # a custom range shorter than the episode rule's minimum duration (5 s) measures nothing
WHOLE_TOLERANCE = .05       # s: a custom range within this of both segment ends is the whole segment (one report)
FRAME_WIDTHS = (640, 1280)  # screenshot widths: the evidence column and the PDF (frames.WIDTHS)
REPLACE_TRIES, REPLACE_WAIT = 20, .1   # os.replace retries against a Windows reader (corrections._write)
READ_TRIES, READ_WAIT = 8, .05         # read retries while the file is being replaced (server.shape_settings)
RECORD_CACHE = 16           # parsed report files kept in memory (keyed by path, mtime and size)

PACKAGE = Path(__file__).resolve().parent
REFERENCE_FILE = PACKAGE/'reference.json'
# The v1 report's drill library: drills.json became the v2 library in wave 1 (DRILLS), the v1 one moved to
# drills_v1.json; a v1 report is computed from the v1 file, so its fingerprint follows that file.
DRILLS_FILE = PACKAGE/'drills_v1.json'
# The code a report is computed from: the coaching package (except this storage and the PDF output, which change no
# report) and every soccer/ module it reads: series.py imports analytics, labels, pitch_model, registration and
# team_shape, evidence.py field, pitch_model and registration (field decides the overlay tiers), and those reach
# geometry. An edit to any of them makes the stored reports stale (the code reviewer's CR-13: an edit to field.py did not).
CODE_EXCLUDED = {'store.py', 'frames.py', 'render.py', 'pdf.py'}
CODE_EXTRA = tuple(PACKAGE.parent/name for name in ('team_shape.py', 'analytics.py', 'labels.py', 'pitch_model.py',
                                                    'registration.py', 'field.py', 'geometry.py'))
# Ids from the page, matched whole (fullmatch: '$' also accepts a trailing newline).
OBSERVATION_ID = re.compile(r'obs-[0-9a-f]{6,40}')
FEEDBACK_ID = re.compile(r'f\d{8}-\d{6}-[0-9a-f]{6}')
FEEDBACK_VERSION = 2        # coaching_feedback.json: 1 = v1 verdicts only; 2 adds v2 verdicts and 'adjustments' (DESIGN 10)
UNPLACED =('excluded', 'withdrawn')   # match groups of identities that are no longer players (corrections.ALL_GROUPS)

_lock = threading.Lock()          # one feedback writer at a time (read, change, write)
_memo_lock = threading.Lock()
_file_shas = {}
_records = {}
_code = {}


class StoreError(Exception):
    """A stored file could not be read or written; nothing was changed."""


class FeedbackUnreadable(StoreError):
    pass


class EngineUnavailable(RuntimeError):
    """soccer.coaching does not provide build_report_for yet."""


class PdfUnavailable(RuntimeError):
    """soccer.coaching.pdf does not provide the PDF writer yet."""


class FrameUnavailable(RuntimeError):
    """The evidence frame could not be decoded from the source video."""


# ---------- hashes and the fingerprint ----------
def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def file_sha(path):
    """The sha256 of a file's bytes, remembered per (path, mtime, size); None when the file does not exist."""
    p = Path(path)
    try:
        st = p.stat()
    except OSError:
        return None
    key = (str(p), st.st_mtime_ns, st.st_size)
    with _memo_lock:
        if key in _file_shas:
            return _file_shas[key]
    try:
        sha = sha256_bytes(p.read_bytes())
    except OSError:
        return None
    with _memo_lock:
        if len(_file_shas) > 64:
            _file_shas.clear()
        _file_shas[key] = sha
    return sha


def settings_sha(settings):
    """The team-shape settings as the analysis reads them (team_shape.clean_settings), hashed in a canonical order."""
    return sha256_bytes(json.dumps(settings or {}, sort_keys=True, separators=(',', ':')).encode('utf-8'))


def code_sha():
    """The coaching code and the analytics it builds on, hashed once per process: the code this process runs (an
    edit on disk changes nothing until the server restarts, and then the reports it cached go stale)."""
    with _memo_lock:
        if 'sha' in _code:
            return _code['sha']
    digest = hashlib.sha256()
    files = sorted(p for p in PACKAGE.glob('*.py') if p.name not in CODE_EXCLUDED) + [p for p in CODE_EXTRA if p.exists()]
    for p in files:
        try:
            data = p.read_bytes()
        except OSError:
            continue
        digest.update(p.name.encode('utf-8') + b'\0' + data.replace(b'\r\n', b'\n') + b'\0')
    sha = digest.hexdigest()
    with _memo_lock:
        _code['sha'] = sha
    return sha


def engine():
    """{'version': ENGINE_VERSION, 'code': sha16 of the code}."""
    return {'version': episodes.ENGINE_VERSION, 'code': code_sha()[:16]}


def is_observation_id(value):
    return isinstance(value, str) and OBSERVATION_ID.fullmatch(value) is not None


def is_feedback_id(value):
    return isinstance(value, str) and FEEDBACK_ID.fullmatch(value) is not None


# ---------- JSON without non-finite numbers ----------
def _refuse_constant(token):
    raise ValueError(f'{token} is not a finite number')


def _finite_or_refuse(text):
    x = float(text)
    if not math.isfinite(x):          # 1e999 parses as inf without passing parse_constant
        raise ValueError(f'{text} is not a finite number')
    return x


def _finite_or_none(text):
    x = float(text)
    return x if math.isfinite(x) else None


def loads_strict(text):
    """JSON that holds only finite numbers (a stored report: allow_nan=False wrote it, so NaN means a damaged file)."""
    return json.loads(text, parse_constant=_refuse_constant, parse_float=_finite_or_refuse)


def loads_lenient(text):
    """JSON whose NaN, Infinity and out-of-range numbers read as null (a hand-edited feedback file keeps its entries)."""
    return json.loads(text, parse_constant=lambda token: None, parse_float=_finite_or_none)


def finite(data):
    """data with every non-finite float replaced by None, recursively."""
    if isinstance(data, float):
        return data if math.isfinite(data) else None
    if isinstance(data, dict):
        return {k: finite(v) for k, v in data.items()}
    if isinstance(data, (list, tuple)):
        return [finite(v) for v in data]
    return data


def json_text(data):
    """The JSON text of a coaching response, never with a NaN or Infinity token (Flask's jsonify writes them and the
    page's JSON.parse rejects them): strict first (one C-encoder pass, 97 kB for the largest integration report), and
    only when that refuses, again with the non-finite numbers as null."""
    try:
        return json.dumps(data, allow_nan=False, separators=(',', ':'))
    except ValueError:
        return json.dumps(finite(data), allow_nan=False, separators=(',', ':'))


def _seconds(value, name):
    if value is None or value == '':
        return None
    try:
        x = float(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f'{name} must be seconds') from None
    if isinstance(value, bool) or not math.isfinite(x):
        raise ValueError(f'{name} must be seconds')
    return x


def clock(t):
    """mm:ss.ss, as the page shows source-video time (coaching.js clock)."""
    t = max(0.0, float(t))
    m = int(t//60)
    return f'{m:02d}:{t-60*m:05.2f}'


def clean_options(team, t0=None, t1=None, context=None, bounds=None):
    """The report options as the fingerprint keys them: {'team', 't0', 't1', 'context'}. t0/t1 are absolute seconds
    (the video's time), both None for the whole segment; a custom range is clamped to bounds (the segment's first and
    last frame times) and rounded to 0.01 s, and a range that covers the segment is the whole segment, so the two
    requests share one report. The page rounds a typed time to 0.01 s once and sends that value in both its status and
    its build request (an exact tie like 00:10.125 was asked as 10.13 by toFixed and built as 10.12 by round), so the
    rounding here only ever meets values that are already hundredths and leaves them unchanged. Raises ValueError for
    an unknown team or context or a range under MIN_RANGE_SECONDS (the message names the segment's bounds)."""
    if not isinstance(team, str) or team not in ('A', 'B'):
        raise ValueError('team must be A or B')
    context = DEFAULT_CONTEXT if context in (None, '') else context
    if not isinstance(context, str) or context not in CONTEXTS:
        raise ValueError('context must be development, competitive or advanced')
    a, b = _seconds(t0, 't0'), _seconds(t1, 't1')
    if a is None and b is None:
        return {'team': team, 't0': None, 't1': None, 'context': context}
    lo, hi = bounds if bounds else (-math.inf, math.inf)
    a = lo if a is None else a
    b = hi if b is None else b
    if not (math.isfinite(a) and math.isfinite(b)):
        raise ValueError('a custom range needs t0 and t1')
    a, b = min(a, b), max(a, b)
    a, b = max(a, lo), min(b, hi)
    if bounds and a <= lo+WHOLE_TOLERANCE and b >= hi-WHOLE_TOLERANCE:
        return {'team': team, 't0': None, 't1': None, 'context': context}
    if b-a < MIN_RANGE_SECONDS:
        inside = f' ({clock(lo)}-{clock(hi)}, source-video time)' if bounds else ''
        raise ValueError(f'Choose a range of at least {MIN_RANGE_SECONDS:g} seconds inside the analysed segment{inside}.')
    return {'team': team, 't0': round(a, 2), 't1': round(b, 2), 'context': context}


def fingerprint_inputs(result_sha, settings, options, engine_info=None, reference_sha=None, drills_sha=None):
    """What a report is computed from (the shas cut to 16 hex). The reference and drill shas are read from their files
    unless given; 'missing' when a file does not exist."""
    ref = reference_sha if reference_sha is not None else file_sha(REFERENCE_FILE)
    drills = drills_sha if drills_sha is not None else file_sha(DRILLS_FILE)
    return {'store': STORE_VERSION, 'engine': dict(engine_info or engine()),
            'reference': (ref or 'missing')[:16], 'drills': (drills or 'missing')[:16],
            'result': result_sha[:16], 'settings': settings_sha(settings)[:16], 'options': dict(options)}


def fingerprint(inputs):
    """The sha256 (64 hex) of the canonical JSON of the inputs."""
    return sha256_bytes(json.dumps(inputs, sort_keys=True, separators=(',', ':')).encode('utf-8'))


def applied_corrections(corrections):
    """The ids of the corrections a result lists as applied (result['corrections'])."""
    return sorted(c['id'] for c in corrections or [] if isinstance(c, dict) and isinstance(c.get('id'), str) and c.get('applied', True))


def _plural(n, word):
    return f'{n} {word}{"" if n == 1 else "s"}'


def staleness(stored, current, stored_corrections=None, current_corrections=None):
    """Why a stored report no longer matches: [{'code', 'text'}] comparing the stored fingerprint inputs with the current
    ones (the corrections lists, when both are known, say how many were made or undone)."""
    out = []
    if stored.get('result') != current.get('result'):
        text = 'Corrections changed the tracking result since this report was made'
        if stored_corrections is not None and current_corrections is not None:
            made = len(set(current_corrections)-set(stored_corrections))
            undone = len(set(stored_corrections)-set(current_corrections))
            parts = ([f'{_plural(made, "correction")} made'] if made else []) + ([f'{_plural(undone, "correction")} undone'] if undone else [])
            if parts:
                text = f'{" and ".join(parts)} since this report: the tracking result changed'
            else:
                text = 'The tracking result changed since this report was made'
        out.append({'code': 'result', 'text': text+'.'})
    if stored.get('settings') != current.get('settings'):
        out.append({'code': 'settings', 'text': 'The team-shape settings changed (attacking direction, lineup or formation labels).'})
    if stored.get('engine') != current.get('engine') or stored.get('store') != current.get('store'):
        out.append({'code': 'engine', 'text': 'The report engine was updated.'})
    if stored.get('reference') != current.get('reference'):
        out.append({'code': 'reference', 'text': 'The professional reference was updated.'})
    if stored.get('drills') != current.get('drills'):
        out.append({'code': 'drills', 'text': 'The drill library was updated.'})
    if not out and stored.get('options') != current.get('options'):
        out.append({'code': 'options', 'text': 'The report options differ.'})
    return out


# ---------- atomic files ----------
def _replace(tmp, path):
    for attempt in range(REPLACE_TRIES):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            # Windows refuses to replace a file another request is reading at this moment: try again shortly.
            if attempt == REPLACE_TRIES-1:
                Path(tmp).unlink(missing_ok=True)
                raise
            time.sleep(REPLACE_WAIT)


def write_json(path, data, indent=None):
    """JSON written atomically (no NaN: the page's JSON.parse rejects it)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(data, indent=indent, separators=None if indent else (',', ':'), allow_nan=False)
    tmp = path.with_name(f'.{path.name}.{uuid.uuid4().hex}.tmp')
    tmp.write_text(text, encoding='utf-8')
    _replace(tmp, path)


def write_bytes(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f'.{path.name}.{uuid.uuid4().hex}.tmp')
    tmp.write_bytes(data)
    _replace(tmp, path)


# ---------- cached reports ----------
def report_dir(run_dir):
    return Path(run_dir)/'coaching'


def report_path(run_dir, team, fp):
    return report_dir(run_dir)/f'report-{team}-{fp[:16]}.json'


def save_report(run_dir, team, inputs, report, corrections=None, now=None):
    """Store a built report under its fingerprint and keep the newest KEEP_REPORTS files of the team. The record is
    {'version', 'fingerprint', 'inputs', 'saved', 'corrections' (applied ids, to word the staleness), 'report'}.
    Raises ValueError when the report holds a non-finite number (it is not stored)."""
    fp = fingerprint(inputs)
    record = {'version': STORE_VERSION, 'fingerprint': fp, 'inputs': inputs,
              'saved': (now or datetime.now()).isoformat(timespec='seconds'),
              'corrections': list(corrections) if corrections is not None else None,
              'report': dict(report, fingerprint=fp)}
    path = report_path(run_dir, team, fp)
    write_json(path, record)
    files = sorted(report_dir(run_dir).glob(f'report-{team}-*.json'), key=lambda p: p.stat().st_mtime_ns, reverse=True)
    for old in files[KEEP_REPORTS:]:
        if old != path:
            old.unlink(missing_ok=True)
    return path


def load_record(path):
    """A stored report record (cached per path, mtime and size), or None when it is missing or malformed (a NaN or
    Infinity in it counts as malformed: save_report never writes one)."""
    p = Path(path)
    try:
        st = p.stat()
        key = (str(p), st.st_mtime_ns, st.st_size)
        with _memo_lock:
            if key in _records:
                return _records[key]
        record = loads_strict(p.read_text(encoding='utf-8'))
    except (OSError, ValueError, RecursionError):
        return None
    if not (isinstance(record, dict) and isinstance(record.get('report'), dict) and isinstance(record.get('inputs'), dict)
            and isinstance(record.get('fingerprint'), str)):
        return None
    with _memo_lock:
        if len(_records) >= RECORD_CACHE:
            _records.pop(next(iter(_records)))
        _records[key] = record
    return record


def records(run_dir, team=None):
    """The stored records of a run (of one team), newest first."""
    out = []
    for p in report_dir(run_dir).glob(f'report-{team or "*"}-*.json'):
        r = load_record(p)
        if r is not None:
            out.append(r)
    out.sort(key=lambda r: str(r.get('saved') or ''), reverse=True)
    return out


def find_report(run_dir, options, current=None):
    """(record, fresh): the stored report with the current fingerprint (fresh), else the newest one with the same
    options (stale), else (None, False)."""
    same = [r for r in records(run_dir, options['team']) if (r['inputs'].get('options') or {}) == options]
    for r in same:
        if r['fingerprint'] == current:
            return r, True
    return (same[0], False) if same else (None, False)


def find_observation(record, observation_id):
    """The observation with this id in a record's report, or None."""
    for o in (record.get('report') or {}).get('observations') or []:
        if isinstance(o, dict) and o.get('id') == observation_id:
            return o
    return None


def evidence_frames(record):
    """The absolute source frame indices of a record's evidence."""
    return set(evidence_times(record))


# ---------- feedback ----------
def feedback_path(video_dir):
    return Path(video_dir)/'coaching_feedback.json'


def read_feedback_doc(video_dir, strict=False):
    """The feedback file as {'version', 'feedback': [dict], 'adjustments': [dict]} (version 1 files have no
    adjustments). Missing file: both lists empty. While the file is being replaced a read is retried. In strict mode
    (before a write) an unreadable or unparsable file raises FeedbackUnreadable, so a transient read failure can never
    wipe a coach's earlier verdicts; otherwise it counts as no feedback. Every stored dict is returned as it is (a
    write keeps entries it does not understand). A NaN or Infinity in the file (hand-edited) reads as null, so the
    entry keeps its other fields and no non-finite number reaches a response."""
    path = feedback_path(video_dir)
    empty = {'version': FEEDBACK_VERSION, 'feedback': [], 'adjustments': []}
    data = None
    for attempt in range(READ_TRIES):
        try:
            if not path.exists():
                return empty
            data = loads_lenient(path.read_text(encoding='utf-8'))
            break
        except (ValueError, TypeError, RecursionError):
            if strict:
                raise FeedbackUnreadable('The coaching feedback file could not be read; nothing was changed.') from None
            return empty
        except OSError:
            if attempt == READ_TRIES-1:
                if strict:
                    raise FeedbackUnreadable('The coaching feedback file could not be read; nothing was changed.') from None
                return empty
            time.sleep(READ_WAIT)
    entries = data.get('feedback') if isinstance(data, dict) else None
    adjustments = data.get('adjustments', []) if isinstance(data, dict) else None
    if not isinstance(entries, list) or not isinstance(adjustments, list):
        if strict:
            raise FeedbackUnreadable('The coaching feedback file is not in the expected format; nothing was changed.')
        return empty
    return {'version': data.get('version'), 'feedback': [e for e in entries if isinstance(e, dict)],
            'adjustments': [a for a in adjustments if isinstance(a, dict)]}


def read_feedback(video_dir, strict=False):
    """The stored feedback entries (read_feedback_doc's 'feedback'). In strict mode every stored dict, else only the
    well-formed v1 entries (verdicts by observation id): the v1 report's view of the file."""
    entries = read_feedback_doc(video_dir, strict)['feedback']
    return entries if strict else [e for e in entries if _well_formed(e)]


def _write_feedback(video_dir, feedback, adjustments):
    """The feedback file, version 2 (DESIGN 10: verdicts of both engines in 'feedback', the coach's adjusted areas in
    'adjustments'); a writer always passes both lists as read strictly, so neither engine drops the other's entries."""
    write_json(feedback_path(video_dir), {'version': FEEDBACK_VERSION, 'feedback': feedback, 'adjustments': adjustments},
               indent=1)


def _well_formed(e):
    o = e.get('observation') if isinstance(e, dict) else None
    text_or_none = (str, type(None))
    return (isinstance(o, dict) and isinstance(e.get('id'), str) and isinstance(e.get('run'), str) and e.get('verdict') in VERDICTS
            and isinstance(o.get('key'), str) and isinstance(e.get('text'), text_or_none) and isinstance(e.get('made'), text_or_none))


def _headline(report, observation_id):
    for key in ('improvements', 'strengths', 'reviews'):
        for f in report.get(key) or []:
            if isinstance(f, dict) and f.get('observation') == observation_id and isinstance(f.get('headline'), str):
                return f['headline']
    return None


def _number_or_none(x):
    return float(x) if isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) else None


def _text_or_none(x):
    return x if isinstance(x, str) else None


def _players(o):
    """The player ids of an observation or a stored feedback observation, as a sorted list (compared as a set)."""
    return sorted({p for p in (o.get('players') if isinstance(o.get('players'), list) else []) if isinstance(p, str)})


def add_feedback(video_dir, run, record, observation_id, verdict, text=None, now=None):
    """Store the coach's verdict on one finding of a stored report: {id, run, made, report (fp16), level (the report's
    context level), observation: {key, detector, team, pattern, context (the comparison context, e.g. third:middle),
    t0, t1, players, headline (in id form: labels renumber), classification}, verdict, text}. A verdict on the same
    finding of the same run replaces the earlier one. Raises KeyError when the observation is not in the record,
    ValueError for an unknown verdict, FeedbackUnreadable when the stored file cannot be read (nothing is written)."""
    if verdict not in VERDICTS:
        raise ValueError('verdict must be correct, partial, incorrect or insufficient')
    note = '' if text is None else str(text).strip()
    if len(note) > MAX_NOTE:
        raise ValueError(f'A note can be at most {MAX_NOTE} characters.')
    obs = find_observation(record, observation_id)
    if obs is None:
        raise KeyError(observation_id)
    report = record.get('report') or {}
    window = obs.get('window') if isinstance(obs.get('window'), dict) else {}
    baseline = obs.get('baseline') if isinstance(obs.get('baseline'), dict) else {}
    now = now or datetime.now()
    entry = {'id': f'f{now:%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}', 'run': str(run), 'made': now.isoformat(timespec='seconds'),
             'report': record['fingerprint'][:16], 'level': _text_or_none(report.get('context')),
             'observation': {'key': observation_id, 'detector': _text_or_none(obs.get('detector')), 'team': _text_or_none(obs.get('team')),
                             'pattern': _text_or_none(obs.get('pattern')), 'context': _text_or_none(baseline.get('context')),
                             't0': _number_or_none(window.get('t0')), 't1': _number_or_none(window.get('t1')),
                             'players': _players(obs),
                             'headline': _headline(report, observation_id) or _text_or_none(obs.get('pattern')),
                             'classification': _text_or_none(obs.get('classification'))},
             'verdict': verdict, 'text': note}
    with _lock:
        doc = read_feedback_doc(video_dir, strict=True)
        kept = [e for e in doc['feedback'] if not (e.get('run') == entry['run'] and (e.get('observation') or {}).get('key') == observation_id)]
        kept.append(entry)
        _write_feedback(video_dir, kept, doc['adjustments'])
    return entry


def remove_feedback(video_dir, feedback_id):
    """Remove one feedback entry (a verdict of either engine, f<date>-<time>-<hex>) or one adjusted area (adj-<hex>)
    by id. Returns False when there is no such entry (nothing is written)."""
    with _lock:
        doc = read_feedback_doc(video_dir, strict=True)
        entries, adjustments = doc['feedback'], doc['adjustments']
        kept = [e for e in entries if e.get('id') != feedback_id]
        kept_adj = [a for a in adjustments if (a.get('adjustment') or {}).get('id') != feedback_id]
        if len(kept) == len(entries) and len(kept_adj) == len(adjustments):
            return False
        _write_feedback(video_dir, kept, kept_adj)
    return True


def _overlap(a0, a1, b0, b1):
    """At least half of each period shared (the rule soccer/team_shape._overlaps uses for the coach's labels)."""
    if None in (a0, a1, b0, b1):
        return False
    lo, hi = max(a0, b0), min(a1, b1)
    return hi > lo and (hi-lo) >= .5*max(a1-a0, 1e-6) and (hi-lo) >= .5*max(b1-b0, 1e-6)


def _same_finding(o, obs):
    """A stored verdict's observation o and a report observation are the same finding: the same detector, team, pattern
    and players over periods sharing at least half of each. An entry stored before patterns were kept never matches
    this way (only by its id)."""
    w = obs.get('window') if isinstance(obs.get('window'), dict) else {}
    return (isinstance(o.get('pattern'), str) and o.get('pattern') == obs.get('pattern') and o.get('detector') == obs.get('detector')
            and o.get('team') == obs.get('team') and _players(o) == _players(obs)
            and _overlap(_number_or_none(o.get('t0')), _number_or_none(o.get('t1')), _number_or_none(w.get('t0')), _number_or_none(w.get('t1'))))


def attach_feedback(report, entries, run):
    """The feedback entries, each with 'thisRun' and 'attached': the id of the observation of this report it belongs to,
    or None. Only a verdict given on this run is ever attached: by its observation id, else to the one observation that
    is the same finding (_same_finding; of several, the one with the same comparison context, and none when that
    still leaves more than one). A verdict from another run is never attached (ids leave the run out and team letters
    are arbitrary per run); the page lists it apart, as from an earlier run."""
    observations = [o for o in (report or {}).get('observations') or [] if isinstance(o, dict) and isinstance(o.get('id'), str)]
    ids = {o['id'] for o in observations}
    out = []
    for e in entries:
        o = e.get('observation') if isinstance(e.get('observation'), dict) else {}
        this_run = e.get('run') == run
        attached = None
        if this_run:
            if o.get('key') in ids:
                attached = o['key']
            else:
                same = [obs for obs in observations if _same_finding(o, obs)]
                if len(same) > 1:
                    same = [obs for obs in same if isinstance(obs.get('baseline'), dict) and obs['baseline'].get('context') == o.get('context')]
                attached = same[0]['id'] if len(same) == 1 else None
        out.append(dict(e, thisRun=this_run, attached=attached))
    return out


# ---------- display labels ----------
def unplaced_label(group, display):
    """What an identity that is no longer a player shows as: 'excluded (was A-03)'. It never shows as a bare label, which
    after the renumbering belongs to someone else: the fact check excluded A-03 on a BC60 copy, and a stale report's
    "(A-02, A-03, A-07)" then read "(A-02, A-03, A-06)" with the excluded person's raw id beside A-04, now shown as A-03."""
    return f'{group} (was {display})'


def labels_of(result):
    """id -> display label of the result as it is now (players, goalkeepers, roster slots; an identity excluded by a
    correction or withdrawn by the tracker shows as unplaced_label(); a merged identity shows the label of the identity it
    was merged into). Reports store ids; labels renumber after corrections."""
    out = {}
    match = (result or {}).get('match') or {}
    for key in ('players', 'goalkeepers', 'referees'):
        for e in match.get(key) or []:
            if isinstance(e, dict) and isinstance(e.get('id'), str):
                out[e['id']] = e.get('display') or e['id']
    for key in UNPLACED:
        for e in match.get(key) or []:
            if isinstance(e, dict) and isinstance(e.get('id'), str) and e['id'] not in out:
                shown = e.get('display') if isinstance(e.get('display'), str) and e.get('display') not in ('', 'OUT') else e['id']
                out[e['id']] = unplaced_label(key, shown)
    for s in ((result or {}).get('roster') or {}).get('slots') or []:
        if isinstance(s, dict) and isinstance(s.get('id'), str) and s['id'] not in out:
            out[s['id']] = s.get('display') or s['id']
    for e in match.get('retired') or []:
        if isinstance(e, dict) and isinstance(e.get('id'), str) and e['id'] not in out:
            out[e['id']] = out.get(e.get('retiredInto')) or e.get('retiredInto') or e['id']
    return out


# ---------- screenshots ----------
def frame_dir(video_dir):
    return Path(video_dir)/'coaching'/'frames'


def frame_cache_path(video_dir, frame, width):
    return frame_dir(video_dir)/f'f{int(frame)}-w{int(width)}.jpg'


def frame_jpeg(source, video_dir, frame, width, at=None, fps=None):
    """The JPEG of one source frame at a width (raw pixels, no overlay), cached per video, read by soccer/coaching/
    frames.py: the capture is positioned by the absolute frame index (exact on 180 of 180 targets over the three test
    videos, where a time seek of a frame's centre lands on the next frame 59 times in 60) and the decoder's own time
    must agree with the frame's time at (the evidence record's; frame/fps without it) within half a frame. Raises
    FrameUnavailable."""
    if width not in FRAME_WIDTHS:
        raise ValueError(f'width must be one of {FRAME_WIDTHS}')
    from . import frames
    frame = int(frame)
    got = frames.extract(source, [frame], frame_dir(video_dir), (int(width),), times={frame: at} if at is not None else None, fps=fps)
    data = got['jpegs'].get((frame, int(width)))
    if data is None:
        raise FrameUnavailable(f'Frame {frame} could not be read from the source video: {got["errors"].get(frame, "not decoded")}.')
    return data


def evidence_times(record):
    """{absolute source frame: time} of a record's evidence (a v1 report's 'evidence', a v2 report's 'keyMoments'):
    the only frames the screenshot route serves, with the time the decoder must agree with."""
    out = {}
    report = record.get('report') or {}
    for e in list(report.get('evidence') or [])+list(report.get('keyMoments') or []):
        f = e.get('frame') if isinstance(e, dict) else None
        if isinstance(f, int) and not isinstance(f, bool) and f >= 0:
            out[f] = _number_or_none(e.get('time'))
    return out


class EvidenceFrames:
    """The PDF writer's frame source for one report: provider(frame, width) -> JPEG bytes or None, and
    prefetch(frames, widths) (one sorted decoding pass, frames.FrameProvider), limited to the report's evidence frames
    and checked against their times; the JPEGs go to the video's frame cache the page's screenshots use."""

    def __init__(self, source, video_dir, report, fps=None):
        from . import frames
        times = evidence_times({'report': report})
        self.allowed = set(times)
        self.provider = frames.FrameProvider(source, frame_dir(video_dir), times={f: t for f, t in times.items() if t is not None}, fps=fps)

    def prefetch(self, frames, widths=(FRAME_WIDTHS[0],)):
        wanted = [int(f) for f in frames if int(f) in self.allowed]
        return self.provider.prefetch(wanted, [int(w) for w in widths if int(w) in FRAME_WIDTHS])

    def __call__(self, frame, width=FRAME_WIDTHS[0]):
        if int(frame) not in self.allowed or int(width) not in FRAME_WIDTHS:
            return None
        return self.provider(int(frame), int(width))


# ---------- the report and its PDF (provided by the rest of the package) ----------
def build_report_for(result_path, team, options):
    """The coaching report of one team of a saved result: soccer.coaching.build_report_for(result_path, team, options),
    the engine and the report assembly. Until the package provides it this raises EngineUnavailable: no stand-in
    report is ever served."""
    from soccer import coaching
    build = getattr(coaching, 'build_report_for', None)
    if build is None or build is build_report_for:
        raise EngineUnavailable('The coaching report engine is not available in this version yet.')
    return build(result_path, team, options)


PDF_WRITER = 'report_pdf'     # soccer/coaching/pdf.py: report_pdf(report, *, labels, frame, stale) -> bytes


def pdf_filename(report, video_name, run, options, key=None):
    """The PDF's download name, pitchiq-coaching-<video>-<run>-<team>-<level>.pdf, by soccer/coaching/pdf.py
    pdf_filename (the CLI and the server name files alike): the context level is in it, since the three levels of one
    report are three different texts that downloaded under one name before. The same pattern is built here when the PDF
    module offers no helper."""
    level = options.get('context') if options.get('context') in CONTEXTS else (report or {}).get('context')
    try:
        from . import pdf
        name = pdf.pdf_filename(video_name, run, options.get('team'), level, key)
    except Exception:                                    # noqa: BLE001 - the name never fails an export
        name = None
    if isinstance(name, str) and name.endswith('.pdf'):
        return name
    def part(v, n):
        return re.sub(r'[^A-Za-z0-9._-]+', '-', str(v or '')).strip('-.')[:n].strip('-.')
    stem = part(Path(str(video_name)).stem if video_name else '', 48) or part(key, 16) or 'video'
    return '-'.join(p for p in ('pitchiq-coaching', stem, part(run, 64), part(options.get('team'), 4),
                                level if level in CONTEXTS else None) if p) + '.pdf'


def render_pdf(report, labels, frame, stale=None, **options):
    """The PDF of a report: soccer.coaching.pdf.report_pdf(report, labels={id: display}, frame=provider(frame, width)
    -> JPEG bytes or None (an EvidenceFrames), stale=[reason texts] or None, **options). A v2 report also passes
    feedback=[the answer's entries the report carries] (the v2 report record is closed, so feedback travels beside
    it, not inside) and colours={'A': '#rrggbb', 'B': '#rrggbb'} (the viewer's team colours, critic N6), and since fix
    round 3 coach_spans (the run has coach-marked spans: the answer's coachSpans) and marked (marked_teams of those
    spans), so the possession row says where the shares come from as the page does (CODE M-2). Raises PdfUnavailable
    when the package does not provide the writer."""
    try:
        from . import pdf
    except ImportError as error:
        raise PdfUnavailable('PDF export is not available in this version yet.') from error
    write = getattr(pdf, PDF_WRITER, None)
    if write is None:
        raise PdfUnavailable('PDF export is not available in this version yet.')
    return write(report, labels=labels, frame=frame, stale=stale, **options)


# =====================================================================================================================
# v2: adaptive coaching (DESIGN v2: 1.2 the three cache tiers, 2.1 the settings, 8 squads and progress, 10 feedback and
# adjusted areas, 12.1 the server API). The module docstring's v2 part says why. The v1 functions above stay for the v1
# report, which the page's legacy mode, tools/coaching_report.py --context and the v1 tests still ask for with a
# context level (DESIGN 11.3: deprecated code is kept while something uses it).
# =====================================================================================================================
V2_DIR = 'v2'               # runs/<video>/<run>/coaching/v2/: the tier files. v1's report-<team>-*.json pruning globs
                            # coaching/ only, so it can never delete a v2 file (the names would otherwise collide)
KEEP_TIERS = {'obs': 4, 'roles': 8, 'report': 24}   # files kept per run and team (DESIGN 1.2; critic S14: v1's 4 were
                                                     # too few once a level, range or role change is one report each)
RECORD_VERSION = 2          # a stored v2 report record
SETTINGS_FILE = 'coaching_settings.json'    # per video (DESIGN 2.1)
SPANS_FILE = 'coach_spans.json'             # per video, one record per run (team letters belong to a run)
MATCH_FILE = 'coaching_match.json'          # per video: the coach's match date for progress entries
SPANS_VERSION = 1
ADJUSTMENT_ID = re.compile(r'adj-[0-9a-f]{12}')
SQUAD_ID = re.compile(r'sq-[0-9a-f]{12}')
FINDING_KEY =re.compile(r'D[1-6]\|[AB]\|[A-Za-z0-9_.\-]{1,40}\|\d+\.\d\|\d+\.\d')   # schema.FindingKey
RUN_NAME = re.compile(r'[A-Za-z0-9_.\-]{1,120}')                                     # schema.RunName
HEX_COLOUR = re.compile(r'#[0-9a-fA-F]{6}')
DAY = re.compile(r'\d{4}-\d{2}-\d{2}')
RANGE_MATCH = .005          # s: two report ranges are one range (both are rounded to 0.01 s before they are keyed)
SOURCE_ORDER = ('coach', 'stable', 'carried', 'window', 'relative', 'none')   # roles.SOURCE_ORDER, most trusted first
FINDING_SECTIONS = ('strengths', 'needsWork', 'reviews', 'described')        # the findings a v2 report shows
POLYGON_MARGIN = .1         # normalized pitch: an adjusted area may reach this far outside the pitch (schema NORM_MARGIN)
MIN_POLYGON_AREA = 1e-4     # normalized pitch: about 0.7 m2 on 105 x 68 m; the smallest suggested area is far larger
SPAN_MIN_S = .2             # s: a shorter coach span is dropped, as the page drops it (coaching.js SPAN_MIN_S)
MAX_COACH_SPANS = 3000      # spans per run: one change every 2.4 s over 120 minutes (CODE L5: 10,000 were stored)

_settings_lock = threading.Lock()     # coaching_settings.json, coach_spans.json, coaching_match.json: read, change, write
_tier_locks = {}                      # (run folder, team) -> Lock: one role / interpretation build of a team at a time
_v2_records = {}
_catalog = {}


class SettingsUnreadable(StoreError):
    """A per-video coaching file (settings, coach spans, match date) exists but cannot be read; nothing was changed."""


class SettingsRefused(ValueError):
    """A coaching settings, spans or feedback request that does not validate; .problems lists why."""

    def __init__(self, problems):
        self.problems = [str(p) for p in problems] or ['the request does not validate']
        super().__init__('; '.join(self.problems[:6]))


def _iso(now=None):
    return (now or datetime.now()).isoformat(timespec='seconds')


def _safe_name(value):
    """A run or video folder name that cannot leave the runs folder."""
    return isinstance(value, str) and RUN_NAME.fullmatch(value) is not None and value not in ('.', '..')


def _read_doc(path, strict, what):
    """The JSON object of a small per-video file, None when it does not exist. An unreadable or malformed file is None,
    or in strict mode (before a write) SettingsUnreadable: a failed read never becomes an empty file on the next save."""
    path = Path(path)
    data = None
    for attempt in range(READ_TRIES):
        try:
            if not path.exists():
                return None
            data = loads_strict(path.read_text(encoding='utf-8'))
            break
        except (ValueError, TypeError, RecursionError):
            break
        except OSError:
            if attempt == READ_TRIES-1:
                break
            time.sleep(READ_WAIT)
    if isinstance(data, dict):
        return data
    if strict:
        raise SettingsUnreadable(f'The {what} file could not be read; nothing was changed.')
    return None


# ---------- coaching settings per video (DESIGN 2.1) ----------
def settings_path(video_dir):
    return Path(video_dir)/SETTINGS_FILE


def read_settings(video_dir, strict=False):
    """The video's coaching settings, cleaned (profiles.clean_settings: schema 'settings'). No file means no team set:
    the report then uses PitchIQ's defaults and says so in a notice (profiles.resolve)."""
    from . import profiles
    return profiles.clean_settings(_read_doc(settings_path(video_dir), strict, 'coaching settings') or {})


def _other(team):
    return 'B' if team == 'A' else 'A'


def _merged_links(old, new):
    """The posted block's run links, plus the stored links of the video's other runs it does not mention while the squad
    stays the same (the page posts the link of the run it shows; a squad's other runs keep theirs)."""
    posted = list(new.get('runLinks') or [])
    if not (old and old.get('squad') and new.get('squad') and old['squad']['id'] == new['squad']['id']):
        return posted
    named = {link['run'] for link in posted}
    return posted+[link for link in old.get('runLinks') or [] if link['run'] not in named]


def link_problems(block, other, squad_exists=None):
    """Why a team block's squad and run links cannot be saved beside the other team's block ([] when they can): one
    link per run, links need a squad, the squad must exist (squad_exists(id) -> bool), one squad is not both teams, and
    a run's team letter is confirmed as at most one squad."""
    out = []
    links = block.get('runLinks') or []
    runs = [link['run'] for link in links]
    if len(runs) != len(set(runs)):
        out.append('runLinks: one link per run (a squad is one team of a run)')
    sq = block.get('squad')
    if links and not sq:
        out.append('runLinks: a run link needs a squad')
    if sq and squad_exists is not None and not squad_exists(sq['id']):
        out.append(f'squad: {sq["id"]} is not a known squad (create it first)')
    osq = (other or {}).get('squad')
    if sq and osq and osq['id'] == sq['id']:
        out.append(f'squad: {sq["name"]} is already the other team\'s squad')
    if sq and osq:
        mine = {(link['run'], link['team']) for link in links if link['confirmed']}
        theirs = {(link['run'], link['team']) for link in other.get('runLinks') or [] if link['confirmed']}
        for run, team in sorted(mine & theirs):
            out.append(f'runLinks: team {team} of run {run} is already confirmed as {osq["name"]}')
    return out


def save_team_settings(video_dir, team, raw, *, now=None, squad_exists=None):
    """Save one team's coaching settings block (POST /api/coaching/settings). The block is cleaned by
    profiles.clean_team with the save time stamped on new role confirmations, and refused (SettingsRefused) when the
    cleaning had to drop or replace anything, or the squad links conflict; it is written atomically under the lock
    after a strict read (an unreadable file raises SettingsUnreadable and is left as it is). Returns {'settings' (the
    whole cleaned file), 'old', 'new' (the team's blocks), 'stale' (schema.stale_tiers: the cache tiers the change
    makes stale, cascading)}."""
    from . import profiles, schema as S
    if team not in S.TEAMS:
        raise SettingsRefused(['team must be A or B'])
    stamp = _iso(now)
    problems = []
    block = profiles.clean_team(raw, now=stamp, problems=problems, path=f'teams.{team}')
    if block is None:
        raise SettingsRefused(problems or ['settings: expected a team settings object'])
    if problems:
        raise SettingsRefused(problems)
    with _settings_lock:
        current = read_settings(video_dir, strict=True)
        old = current['teams'].get(team)
        block['runLinks'] = _merged_links(old, block)
        problems = link_problems(block, current['teams'].get(_other(team)), squad_exists)
        if problems:
            raise SettingsRefused(problems)
        current['teams'][team] = block
        current['updated'] = stamp
        problems = S.validate('settings', current)
        if problems:
            raise SettingsRefused(problems)
        write_json(settings_path(video_dir), current, indent=1)
    return {'settings': current, 'old': old, 'new': block, 'stale': S.stale_tiers(old, block)}


# ---------- coach-marked possession spans per (video, run) (user decision 1, 2026-10-21) ----------
def spans_path(video_dir):
    return Path(video_dir)/SPANS_FILE


def empty_spans():
    return {'version': 1, 'kind': 'coach-possession-spans', 'source': 'coach', 'spans': []}


def _spans_doc(video_dir, strict):
    doc = _read_doc(spans_path(video_dir), strict, 'coach-marked possession')
    if doc is not None and not isinstance(doc.get('runs'), dict):
        if strict:
            raise SettingsUnreadable('The coach-marked possession file is not in the expected format; nothing was '
                                     'changed.')
        return None
    return doc


def read_spans(video_dir, run, strict=False):
    """(the run's coach spans, schema 'coach_spans', or None when it has none; the ISO time they were saved or None).
    Team letters belong to a run, so spans are kept per run of the video. Not strict: a malformed file or record
    counts as none."""
    from . import schema as S
    entry = ((_spans_doc(video_dir, strict) or {}).get('runs') or {}).get(run)
    if not isinstance(entry, dict) or S.validate('coach_spans', entry.get('spans')) or not entry['spans']['spans']:
        return None, None
    return entry['spans'], entry.get('saved') if isinstance(entry.get('saved'), str) else None


def save_spans(video_dir, run, raw, segment, *, now=None):
    """Replace a run's coach spans (POST /api/coaching/spans). The record must validate (schema 'coach_spans'); its spans
    are then cleaned against the analysed segment exactly as situations.build_timeline cleans them
    (situations.clean_coach_spans: inside the segment with COACH_EDGE_S of tolerance, then clipped; overlaps dropped in
    start order), so the stored spans are the ones the measurement tier uses; a span shorter than SPAN_MIN_S is dropped
    as the page drops it, and more than MAX_COACH_SPANS are refused. An empty list removes the run's spans.
    Returns {'spans' (the stored record, the empty one when none), 'dropped', 'changed', 'saved' (ISO or None)}."""
    from . import schema as S, situations
    if not _safe_name(run):
        raise SettingsRefused(['run must be a run name'])
    problems = S.validate('coach_spans', raw)
    if problems:
        raise SettingsRefused(problems)
    if len(raw['spans']) > MAX_COACH_SPANS:
        raise SettingsRefused([f'spans: at most {MAX_COACH_SPANS} per run (got {len(raw["spans"])})'])
    kept, dropped = situations.clean_coach_spans(raw, float(segment['t0']), float(segment['t1']))
    # The page keeps no phase shorter than SPAN_MIN_S (two clicks in a row); a request that sends one stores it no
    # more (CODE L5: a 1 ms span was stored). Dropped after the cleaning, so what is stored is still what is measured.
    long_enough = [s for s in kept if float(s['t1'])-float(s['t0']) >= SPAN_MIN_S-1e-9]
    kept, dropped = long_enough, dropped+len(kept)-len(long_enough)
    record = dict(empty_spans(), spans=[{'t0': float(s['t0']), 't1': float(s['t1']), 'state': s['state']} for s in kept])
    problems = S.validate('coach_spans', record)
    if problems:                                       # clean_coach_spans keeps a valid record; never stored otherwise
        raise SettingsRefused(problems)
    stamp = _iso(now)
    with _settings_lock:
        doc = _spans_doc(video_dir, True) or {'version': SPANS_VERSION, 'runs': {}}
        runs = doc['runs']
        old = runs.get(run) if isinstance(runs.get(run), dict) else {}
        old_spans = ((old.get('spans') or {}).get('spans') if isinstance(old.get('spans'), dict) else None) or []
        changed = old_spans != record['spans']
        saved = stamp if changed else (old.get('saved') if record['spans'] else None)
        if changed:
            if record['spans']:
                runs[run] = {'spans': record, 'saved': stamp}
            else:
                runs.pop(run, None)
            write_json(spans_path(video_dir), {'version': SPANS_VERSION, 'runs': runs}, indent=1)
    return {'spans': record, 'dropped': dropped, 'changed': changed, 'saved': saved if record['spans'] else None}


# ---------- the coach's match date per video (progress entries; PROGRESS's request) ----------
def match_path(video_dir):
    return Path(video_dir)/MATCH_FILE


def read_match_date(video_dir):
    """The match date (YYYY-MM-DD) the coach gave for this video, or None (progress then uses the video file's date)."""
    day = (_read_doc(match_path(video_dir), False, 'match date') or {}).get('date')
    return day if isinstance(day, str) and DAY.fullmatch(day) else None


def save_match_date(video_dir, day, now=None):
    """Store (or with None remove) the coach's match date of this video. Raises SettingsRefused for a date that is not
    YYYY-MM-DD."""
    if day is not None:
        try:
            if not (isinstance(day, str) and DAY.fullmatch(day)):
                raise ValueError
            date.fromisoformat(day)
        except ValueError:
            raise SettingsRefused(['date must be YYYY-MM-DD']) from None
    with _settings_lock:
        if day is None:
            match_path(video_dir).unlink(missing_ok=True)
        else:
            write_json(match_path(video_dir), {'version': 1, 'date': day, 'saved': _iso(now)}, indent=1)
    return day


# ---------- the three cache tiers (DESIGN 1.2) ----------
def segment_of(frames):
    """{'t0', 't1'} of a result's frames as coaching.build_observations writes obs['segment'] (rounded to 1 ms)."""
    times = [float(f['time']) for f in frames or () if isinstance(f, dict) and _number_or_none(f.get('time')) is not None]
    return {'t0': round(min(times), 3), 't1': round(max(times), 3)} if times else {'t0': 0.0, 't1': 0.0}


def report_range(segment, t0=None, t1=None):
    """(lo, hi, whole): a report range clamped to the segment, exactly as coaching.interpret clamps it (whole when
    neither end is given)."""
    a, b = float(segment['t0']), float(segment['t1'])
    if t0 is None and t1 is None:
        return a, b, True
    lo = a if t0 is None else min(max(float(t0), a), b)
    hi = b if t1 is None else min(max(float(t1), a), b)
    return min(lo, hi), max(lo, hi), False


def clean_range(team, t0=None, t1=None, bounds=None):
    """The v2 report options {'team', 't0', 't1'}: clean_options without the v1 context level (a v2 report's level comes
    from the team's coaching settings). Raises ValueError as clean_options does."""
    out = clean_options(team, t0, t1, None, bounds)
    out.pop('context')
    return out


def tier_keys(result_sha, shape_settings, settings, team, spans, segment, t0=None, t1=None):
    """The fingerprints of the three tiers for one team and report range: {'team', 'mfp', 'rfp', 'ifp' (64 hex),
    'range' {t0, t1 (None for the whole segment), whole} (clamped as interpret clamps), 'inputs' {'result' (the file's
    own sha), 'coachSpans' (sha or None), 'measurement', 'role', 'interpretation' (each tier's fingerprint inputs: what
    staleness compares)}}. The server and tools/coaching_report.py key their files with this one function."""
    from soccer import coaching
    from . import schema as S
    block = S.team_block(settings or {}, team)
    m_in = coaching.measurement_inputs(measured_result_sha(result_sha, spans), coaching.shape_settings_sha(shape_settings),
                                       block, team)
    mfp = coaching.fingerprint(m_in)
    r_in = coaching.role_inputs(mfp, block, team)
    rfp = coaching.fingerprint(r_in)
    lo, hi, whole = report_range(segment, t0, t1)
    i_in = coaching.interpretation_inputs(mfp, rfp, block, team, None if whole else lo, None if whole else hi)
    return {'team': team, 'mfp': mfp, 'rfp': rfp, 'ifp': coaching.fingerprint(i_in),
            'range': {'t0': None if whole else lo, 't1': None if whole else hi, 'whole': whole},
            'inputs': {'result': result_sha, 'coachSpans': spans_sha(spans), 'measurement': m_in, 'role': r_in,
                       'interpretation': i_in}}


def tier_dir(run_dir):
    return report_dir(run_dir)/V2_DIR


def tier_path(run_dir, kind, keys):
    """runs/<video>/<run>/coaching/v2/ + coaching.cache_name: obs-<team>-<mfp16>.json, roles-<team>-<mfp16>-<rfp16>.json,
    report-<team>-<mfp16>-<rfp16>-<ifp16>.json."""
    from soccer import coaching
    return tier_dir(run_dir)/coaching.cache_name(kind, keys['team'], keys['mfp'], keys.get('rfp'), keys.get('ifp'))


def tier_states(run_dir, keys):
    """{'measurement', 'role', 'interpretation'}: 'ready' when the tier's file for these keys is stored, else
    'missing'."""
    return {tier: 'ready' if tier_path(run_dir, kind, keys).exists() else 'missing'
            for tier, kind in (('measurement', 'obs'), ('role', 'roles'), ('interpretation', 'report'))}


def load_tier(path):
    """A parsed obs or roles file (a fresh object every time: the stages may annotate what they are given), or None
    when it is missing or malformed."""
    try:
        data = loads_strict(Path(path).read_text(encoding='utf-8'))
    except (OSError, ValueError, RecursionError):
        return None
    return data if isinstance(data, dict) else None


def _prune(folder, kind, team, keep):
    files = []
    for p in Path(folder).glob(f'{kind}-{team}-*.json'):
        try:
            files.append((p.stat().st_mtime_ns, p))
        except OSError:
            continue
    files.sort(reverse=True)
    for _, old in files[KEEP_TIERS[kind]:]:
        if old != keep:
            old.unlink(missing_ok=True)


def save_tier(run_dir, kind, keys, data):
    """Write a tier file atomically (no NaN) and keep the newest KEEP_TIERS[kind] files of the team."""
    path = tier_path(run_dir, kind, keys)
    write_json(path, data)
    _prune(path.parent, kind, keys['team'], path)
    return path


def build_interpretation(obs, roles, settings, keys, meta=None):
    """The report of keys' team and range: coaching.interpret (schema-validated and cross-checked there). Raises
    StoreError when its fingerprints are not the keys' (a cached report must sit under the key that finds it)."""
    from soccer import coaching
    rng = keys['range']
    report = coaching.interpret(obs, roles, coaching=settings or {}, team=keys['team'], t0=rng['t0'], t1=rng['t1'],
                                meta=dict(meta or {}), generated=None)
    got = report.get('fingerprints') or {}
    if (got.get('measurement'), got.get('roles'), got.get('interpretation')) != (keys['mfp'], keys['rfp'], keys['ifp']):
        raise StoreError('The report was computed from other inputs than its cache key names; it was not stored.')
    return report


# ---------- stored v2 report records ----------
def save_report_v2(run_dir, keys, options, report, corrections=None, now=None):
    """Store a v2 report under its three fingerprints: {'version', 'kind', 'keys' {measurement, roles, interpretation},
    'options' {team, t0, t1} (the request's: a stale report is found by them), 'inputs' (tier_keys' inputs: what
    staleness compares), 'corrections' (applied ids, to word staleness), 'saved', 'report'}; the newest
    KEEP_TIERS['report'] of the team are kept. Returns the record. Raises ValueError for a non-finite number."""
    record = {'version': RECORD_VERSION, 'kind': 'coaching-report-record',
              'keys': {'measurement': keys['mfp'], 'roles': keys['rfp'], 'interpretation': keys['ifp']},
              'options': dict(options), 'inputs': keys['inputs'],
              'corrections': list(corrections) if corrections is not None else None, 'saved': _iso(now),
              'report': report}
    save_tier(run_dir, 'report', keys, record)
    return record


def load_record_v2(path):
    """A stored v2 report record (cached per path, mtime and size; never change it: served_report copies), or None
    when it is missing or malformed."""
    p = Path(path)
    try:
        st = p.stat()
        key = (str(p), st.st_mtime_ns, st.st_size)
        with _memo_lock:
            if key in _v2_records:
                return _v2_records[key]
        record = loads_strict(p.read_text(encoding='utf-8'))
    except (OSError, ValueError, RecursionError):
        return None
    ok = (isinstance(record, dict) and record.get('kind') == 'coaching-report-record'
          and isinstance(record.get('report'), dict) and isinstance(record.get('keys'), dict)
          and isinstance(record.get('options'), dict) and isinstance(record.get('inputs'), dict))
    if not ok:
        return None
    with _memo_lock:
        if len(_v2_records) >= RECORD_CACHE:
            _v2_records.pop(next(iter(_v2_records)))
        _v2_records[key] = record
    return record


def records_v2(run_dir, team=None):
    """The stored v2 report records of a run (of one team), newest first."""
    out = []
    for p in tier_dir(run_dir).glob(f'report-{team or "*"}-*.json'):
        r = load_record_v2(p)
        try:
            stamp = p.stat().st_mtime_ns
        except OSError:
            continue
        if r is not None:
            out.append((stamp, r))
    out.sort(key=lambda x: x[0], reverse=True)       # the file's write time: 'saved' has whole seconds only
    return [r for _, r in out]


def _closeness(stored, current):
    """How closely a stored record's tier inputs match the current ones, compared most visible first: the
    interpretation settings (age group, level, style, federation), the role settings (confirmed roles, intended
    formation), the measurement settings (format, field), then the result, the coach spans and the team-shape settings,
    then the code of each tier. A tuple of booleans: larger is closer."""
    sm, cm = stored.get('measurement') or {}, current.get('measurement') or {}
    sr, cr = stored.get('role') or {}, current.get('role') or {}
    si, ci = stored.get('interpretation') or {}, current.get('interpretation') or {}
    return (si.get('interpretation') == ci.get('interpretation'), sr.get('role') == cr.get('role'),
            sm.get('measurement') == cm.get('measurement'),
            (stored.get('result'), stored.get('coachSpans'), sm.get('shapeSettings'))
            == (current.get('result'), current.get('coachSpans'), cm.get('shapeSettings')),
            all(_code_part(a) == _code_part(b) for a, b in ((sm, cm), (sr, cr), (si, ci))))


def find_report_v2(run_dir, keys, options):
    """(record, fresh): the record stored under keys (fresh), else, of the records asked with the same options, the one
    whose tier inputs are closest to the current ones (_closeness; the newest among equals), served stale with the
    reasons; else (None, False).

    Why not simply the newest: a report read from the cache is not rewritten, so its file time is the time it was
    first made. Measured on the BC60 copy through the page (CODE M2): level 4, then 2, then 4 again (a cache hit),
    then only the format changed to 9v9: the newest file was the level-2 report, served under the coach's level-4
    setup with "The age group, playing level, style or federation changed" among the reasons. Ranked by closeness the
    level-4 report is served and the reasons name the format (and the formation the 9v9 format no longer offers)."""
    record = load_record_v2(tier_path(run_dir, 'report', keys))
    if record is not None:
        return record, True
    same = [r for r in records_v2(run_dir, options['team']) if r['options'] == options]
    if not same:
        return None, False
    current = keys.get('inputs') or {}
    best = max(range(len(same)), key=lambda k: (_closeness(same[k].get('inputs') or {}, current), -k))
    return same[best], False


def _code_part(inputs):
    return {k: inputs.get(k) for k in ('tier', 'schema', 'engine', 'code')}


def staleness_v2(stored, current, stored_corrections=None, current_corrections=None):
    """Why a stored v2 report no longer matches: [{'code', 'text'}] comparing its tier inputs with the current ones,
    measurement first (the corrections lists, when both are known, count the corrections made or undone)."""
    out = []
    sm, cm = stored.get('measurement') or {}, current.get('measurement') or {}
    sr, cr = stored.get('role') or {}, current.get('role') or {}
    si, ci = stored.get('interpretation') or {}, current.get('interpretation') or {}
    if stored.get('result') != current.get('result'):
        out += staleness({'result': stored.get('result')}, {'result': current.get('result')}, stored_corrections,
                         current_corrections)[:1]
    if stored.get('coachSpans') != current.get('coachSpans'):
        out.append({'code': 'coach_spans', 'text': 'The possession you marked changed.'})
    if sm.get('shapeSettings') != cm.get('shapeSettings'):
        out.append({'code': 'settings', 'text': 'The team-shape settings changed (attacking direction, lineup or '
                                                'formation labels).'})
    if sm.get('measurement') != cm.get('measurement'):
        out.append({'code': 'format', 'text': 'The match format or field size changed.'})
    if sr.get('role') != cr.get('role'):
        out.append({'code': 'roles', 'text': 'The confirmed player roles or the intended formation changed.'})
    if si.get('interpretation') != ci.get('interpretation'):
        out.append({'code': 'profile', 'text': 'The age group, playing level, style or federation changed.'})
    if any(_code_part(a) != _code_part(b) for a, b in ((sm, cm), (sr, cr), (si, ci))):
        out.append({'code': 'engine', 'text': 'The coaching engine, its rules or the drill library were updated.'})
    if not out:
        out.append({'code': 'options', 'text': 'The report options differ.'})
    return out


def tier_lock(run_dir, team):
    with _memo_lock:
        return _tier_locks.setdefault((str(run_dir), team), threading.Lock())


def ensure_report(run_dir, keys, options, settings, *, meta=None, corrections=None, after=None):
    """The fresh report record of keys from the stored obs file, made when it is missing: the roles file is built when
    it is missing too, then the report, both stored (DESIGN 1.2: a level, age, style or role change never re-reads the
    result). Returns None when the obs file of keys is not stored (the measurement tier must be built first).
    after(record, obs, roles) runs once a new report is stored (the server records progress there). One build per run
    and team at a time; a second caller waits and reads the stored record."""
    with tier_lock(run_dir, keys['team']):
        record = load_record_v2(tier_path(run_dir, 'report', keys))
        if record is not None:
            return record
        obs = load_tier(tier_path(run_dir, 'obs', keys))
        if obs is None:
            return None
        if obs.get('fingerprint') != keys['mfp']:
            raise StoreError('The stored observations do not match their cache key; generate the report again.')
        roles = load_tier(tier_path(run_dir, 'roles', keys))
        if roles is None or roles.get('fingerprint') != keys['rfp'] or roles.get('measurement') != keys['mfp']:
            roles = build_role_tier(obs, settings, keys['team'])
            save_tier(run_dir, 'roles', keys, roles)
        report = build_interpretation(obs, roles, settings, keys, meta)
        record = save_report_v2(run_dir, keys, options, report, corrections)
        if after is not None:
            after(record, obs, roles)
        return record


def served_report(record, *, progression=None, team_name=None):
    """A copy of a stored report as it is served and exported: 'generated' = when it was stored, 'progression' = the
    squad's DEVELOPMENT PROGRESSION (serve-time, in no fingerprint), 'teamName' = the confirmed squad's name."""
    report = copy.deepcopy(record['report'])
    report['generated'] = record.get('saved') if isinstance(record.get('saved'), str) else None
    report['progression'] = progression
    if team_name:
        report['teamName'] = str(team_name)[:80]
    return report


def drills_of(report):
    """{drill_id: drill record of drills.json} for the drills a v2 report names (the page shows their cards)."""
    from . import drills
    library = drills.load_library()
    out = {}
    for pick in report.get('drills') or []:
        d = drills.get(library, pick.get('drill')) if isinstance(pick, dict) else None
        if d is not None:
            out[pick['drill']] = d
    return out


def inferred_roles(roles_file):
    """[{id, role, source, confidence}] per player of a roles file's role book: the role held longest over the
    segment, its most trusted source and that record's confidence (the roles editor shows "Measured: LCB (stable,
    medium)"); a player with records but no role reads role None, source 'none'."""
    per = {}
    for r in ((roles_file or {}).get('rolebook') or {}).get('records') or []:
        if isinstance(r, dict) and isinstance(r.get('id'), str):
            per.setdefault(r['id'], []).append(r)
    out = []
    for pid in sorted(per):
        recs = [r for r in per[pid] if r.get('role') and _number_or_none(r.get('t0')) is not None
                and _number_or_none(r.get('t1')) is not None]
        if not recs:
            out.append({'id': pid, 'role': None, 'source': 'none', 'confidence': None})
            continue
        total = {}
        for r in recs:
            total[r['role']] = total.get(r['role'], 0.0)+max(0.0, float(r['t1'])-float(r['t0']))
        role = max(sorted(total), key=lambda k: total[k])
        best = min((r for r in recs if r['role'] == role),
                   key=lambda r: (SOURCE_ORDER.index(r.get('source')) if r.get('source') in SOURCE_ORDER else 99,
                                  -(float(r['t1'])-float(r['t0']))))
        out.append({'id': pid, 'role': role, 'source': best.get('source') or 'none', 'confidence': best.get('confidence')})
    return out


def measured_roles(roles_file, obs=None):
    """inferred_roles of a roles file with the coach's confirmations left out: what the roles editor shows as
    "Measured" (GET /api/coaching/settings for the newest roles file of each team, and GET /api/coaching for the
    answer's team once its role tier is stored).

    A confirmed player's role book holds only the coach's record over the confirmed span (roles.build_rolebook cuts the
    measured records a coach span covers), so dropping the coach records dropped the player: after A-07 was confirmed
    as RWB on the BC60 copy the editor read "Measured: no stable role" where PitchIQ had measured RB (stable, high)
    (CODE L-2, round 2). When the book has a coach record, the measured reading is the role book rebuilt from the obs
    file without the confirmations (the same intended formation): obs is that obs file, or a callable returning it,
    called only then. Players nobody confirmed read the same either way (build_rolebook's measured records do not
    depend on the confirmations). Without a matching obs file the old reading is kept (the confirmed players left
    out). The rebuilt reading is remembered per (roles, obs) fingerprint."""
    rf = roles_file or {}
    book = rf.get('rolebook') or {}
    records = [r for r in book.get('records') or [] if isinstance(r, dict)]
    measured = [r for r in records if r.get('source') != 'coach']
    if len(measured) == len(records):
        return inferred_roles(dict(rf, rolebook=dict(book, records=measured)))
    rebuilt = _measured_book(rf, obs)
    if rebuilt is not None:
        return inferred_roles(dict(rf, rolebook=rebuilt))
    return inferred_roles(dict(rf, rolebook=dict(book, records=measured)))


MEASURED_MEMO = 16          # rebuilt measured role books kept in memory (one per roles file with confirmations)
_measured_books = {}


def _measured_book(roles_file, obs):
    """The role book of roles_file's obs file without the coach's role confirmations (same formation), or None when
    the obs file is missing or belongs to another measurement or team, or the rolebook stage fails."""
    key = (roles_file.get('fingerprint'), roles_file.get('measurement'))
    with _memo_lock:
        hit = _measured_books.get(key) if key[0] else None
    if hit is not None:
        return copy.deepcopy(hit)
    try:
        data = obs() if callable(obs) else obs
    except Exception:   # noqa: BLE001 - the editor's "Measured" line is information; the old reading is kept
        data = None
    if (not isinstance(data, dict) or data.get('fingerprint') != roles_file.get('measurement')
            or data.get('team') != roles_file.get('team')):
        return None
    formation = copy.deepcopy((roles_file.get('inputs') or {}).get('formation'))
    try:
        book = stage('rolebook')(data, {'roles': {}, 'formation': formation})
    except Exception:   # noqa: BLE001 - as above
        return None
    if key[0]:
        with _memo_lock:
            if len(_measured_books) >= MEASURED_MEMO:
                _measured_books.pop(next(iter(_measured_books)))
            _measured_books[key] = copy.deepcopy(book)
    return book


TABLE_FAILURE = re.compile(r'(profiles\.json|sources\.json|progress_rules\.json|drill library) invalid: (.*)\Z', re.S)


def table_failure(error):
    """{'table', 'problems'} when error says that one of the coaching tables does not validate: profiles.ProfileError
    ('profiles.json invalid: ...', 'sources.json invalid: ...'), progress.ProgressError ('progress_rules.json
    invalid: ...') or the drill library's ValueError ('drill library invalid: ...', drills.json); else None. The v2
    routes answer it with a JSON 503 naming the problems (server.coaching_v2). CODE M-1 (round 2): with
    parameters.coverAngleDeg set to 70.0 (beyond its 60.0 measurement bound) GET /api/coaching/profiles answered
    Flask's HTML 500, the page could not parse it, took it for a server without the v2 routes and built the v1
    professional-reference report without a word."""
    if not isinstance(error, ValueError):
        return None
    m = TABLE_FAILURE.match(str(error))
    if m is None:
        return None
    table = 'drills.json' if m.group(1) == 'drill library' else m.group(1)
    problems = [p.strip() for p in m.group(2).split('; ') if p.strip()]
    return {'table': table, 'problems': problems[:20]}


def marked_teams(spans):
    """Whose possession a run's coach spans (schema 'coach_spans', or None) name: None (no team), 'A' or 'B' (one team
    only) or 'both'; dead-ball spans name no team. The PDF says it as the page's markedSide does (CODE M-2)."""
    teams = {s.get('state') for s in (spans or {}).get('spans') or [] if isinstance(s, dict)} & {'A', 'B'}
    return 'both' if len(teams) == 2 else next(iter(teams)) if teams else None


def profiles_catalog_text():
    """GET /api/coaching/profiles: profiles.catalog() as JSON text, made once per version of profiles.json and
    sources.json (about 110 KB). The version is read from the files profiles.load reads (profiles.PROFILES_PATH and
    SOURCES_PATH), so a table swapped in under another path is never answered from the old text. A table that does not
    validate raises profiles.ProfileError (table_failure)."""
    from . import profiles
    stamp = []
    for name, path in (('profiles.json', profiles.PROFILES_PATH), ('sources.json', profiles.SOURCES_PATH)):
        try:
            st = Path(path).stat()
            stamp.append((str(path), st.st_mtime_ns, st.st_size))
        except OSError:
            stamp.append((name, None, None))
    stamp = tuple(stamp)
    with _memo_lock:
        if _catalog.get('stamp') == stamp:
            return _catalog['text']
    text = json_text(profiles.catalog())
    with _memo_lock:
        _catalog.update(stamp=stamp, text=text)
    return text


def pdf_filename_v2(report, video_name, run, key=None):
    """pitchiq-coaching-<video>-<run>-<team>-level<N>[-<t0>-<t1>s].pdf: the playing level is in the name (each level is
    another text of the same observations), and a custom range its whole seconds (critic R4: two ranges of one run
    downloaded under one name); the spelling of pdf.pdf_filename."""
    def part(v, n):
        return re.sub(r'[^A-Za-z0-9._-]+', '-', str(v or '')).strip('-.')[:n].strip('-.')
    report = report or {}
    level = (report.get('profile') or {}).get('level')
    rng = report.get('range') if isinstance(report.get('range'), dict) else {}
    t0, t1 = _number_or_none(rng.get('t0')), _number_or_none(rng.get('t1'))
    span = (f'{int(math.floor(t0))}-{int(math.ceil(t1))}s' if rng.get('whole') is False and t0 is not None
            and t1 is not None and t1 > t0 >= 0 else None)
    stem = part(Path(str(video_name)).stem if video_name else '', 48) or part(key, 16) or 'video'
    return '-'.join(p for p in ('pitchiq-coaching', stem, part(run, 64), part(report.get('team'), 4),
                                f'level{int(level)}' if isinstance(level, int) else None, span) if p)+'.pdf'


# ---------- feedback on v2 findings and adjusted areas (DESIGN 10) ----------
def _range_of(report):
    r = (report or {}).get('range') or {}
    return {'t0': _number_or_none(r.get('t0')), 't1': _number_or_none(r.get('t1'))}


def same_range(a, b):
    """Two report ranges {t0, t1} are one range (both rounded to 0.01 s: RANGE_MATCH)."""
    a, b = a if isinstance(a, dict) else {}, b if isinstance(b, dict) else {}
    for k in ('t0', 't1'):
        x, y = _number_or_none(a.get(k)), _number_or_none(b.get(k))
        if x is None or y is None or abs(x-y) > RANGE_MATCH:
            return False
    return True


def shown_findings(report):
    """{finding key: finding} of the findings a v2 report shows (strengths, work items, reviews, described): feedback
    attaches only to these (critic C1/S2), never to a withheld one."""
    return {f['key']: f for section in FINDING_SECTIONS for f in (report or {}).get(section) or []
            if isinstance(f, dict) and isinstance(f.get('key'), str)}


def find_feedback_record(run_dir, team, finding_key, rng):
    """The newest stored v2 record of the team whose report covers rng and shows the finding, or None."""
    for r in records_v2(run_dir, team):
        report = r['report']
        if same_range(_range_of(report), rng) and finding_key in shown_findings(report):
            return r
    return None


def _finding_summary(f):
    text = f.get('text') if isinstance(f.get('text'), dict) else {}
    return {'headline': text.get('headline') if isinstance(text.get('headline'), str) else None,
            'detector': f.get('observation') if isinstance(f.get('observation'), str) else None,
            'verdict': f.get('verdict') if isinstance(f.get('verdict'), str) else None}


def _note(text, limit=MAX_NOTE):
    note = '' if text is None else str(text).strip()
    if len(note) > limit:
        raise ValueError(f'A note can be at most {limit} characters.')
    return note


def _same_target(e, run, team, key, rng):
    return e.get('run') == run and e.get('team') == team and e.get('findingKey') == key and same_range(e.get('range'), rng)


def add_verdict_v2(video_dir, run, record, finding_key, verdict, text=None, now=None):
    """Store the coach's verdict on one finding of a stored v2 report: {id, engine 2, run, team, findingKey, range (the
    report's), made, report (ifp16), level, finding {headline, detector, verdict}, verdict, text}. A verdict on the same
    finding, run, team and range replaces the earlier one (critic C2/S3: a verdict belongs to its range). Raises KeyError
    when the report does not show the finding, ValueError for an unknown verdict or a long note, FeedbackUnreadable
    when the stored file cannot be read (nothing is written)."""
    if verdict not in VERDICTS:
        raise ValueError('verdict must be correct, partial, incorrect or insufficient')
    note = _note(text)
    report = record['report']
    finding = shown_findings(report).get(finding_key)
    if finding is None:
        raise KeyError(finding_key)
    now = now or datetime.now()
    rng = _range_of(report)
    entry = {'id': f'f{now:%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}', 'engine': 2, 'run': str(run), 'team': report['team'],
             'findingKey': finding_key, 'range': rng, 'made': now.isoformat(timespec='seconds'),
             'report': str(record['keys'].get('interpretation') or '')[:16],
             'level': (report.get('profile') or {}).get('level'), 'finding': _finding_summary(finding),
             'verdict': verdict, 'text': note}
    with _lock:
        doc = read_feedback_doc(video_dir, strict=True)
        kept = [e for e in doc['feedback'] if not _same_target(e, entry['run'], entry['team'], finding_key, rng)]
        kept.append(entry)
        _write_feedback(video_dir, kept, doc['adjustments'])
    return entry


def _polygon(raw):
    if not isinstance(raw, list) or not 3 <= len(raw) <= 64:
        raise ValueError('adjustment.polygon must have 3 to 64 points')
    out = []
    for p in raw:
        if not (isinstance(p, (list, tuple)) and len(p) == 2) or any(isinstance(v, bool) for v in p):
            raise ValueError('adjustment.polygon points are [x, y] numbers')
        x, y = _number_or_none(p[0]), _number_or_none(p[1])
        if x is None or y is None or not (-POLYGON_MARGIN <= x <= 1+POLYGON_MARGIN and
                                          -POLYGON_MARGIN <= y <= 1+POLYGON_MARGIN):
            raise ValueError('adjustment.polygon points are on the normalized pitch (-0.1 to 1.1)')
        out.append([round(x, 4), round(y, 4)])
    # A degenerate area cannot be compared with the suggested one (coaching_eval --feedback's IoU is undefined for it):
    # collinear points (no area) and edges that cross are refused (CODE L4: both were stored with 200).
    n = len(out)
    area = .5*abs(sum(out[k][0]*out[(k+1) % n][1]-out[(k+1) % n][0]*out[k][1] for k in range(n)))
    if area < MIN_POLYGON_AREA:
        raise ValueError(f'adjustment.polygon must enclose an area (at least {MIN_POLYGON_AREA} of the normalized '
                         'pitch)')
    for i in range(n):
        for j in range(i+1, n):
            if j == i+1 or (i == 0 and j == n-1):
                continue                                   # neighbouring edges share a corner
            if _crossing(out[i], out[(i+1) % n], out[j], out[(j+1) % n]):
                raise ValueError('adjustment.polygon edges must not cross')
    return out


def _crossing(a, b, c, d):
    """Whether segments ab and cd cross or touch (collinear overlaps included)."""
    def orient(p, q, r):
        v = (q[0]-p[0])*(r[1]-p[1])-(q[1]-p[1])*(r[0]-p[0])
        return 0 if abs(v) < 1e-12 else (1 if v > 0 else -1)

    def on(p, q, r):
        return (min(p[0], q[0])-1e-12 <= r[0] <= max(p[0], q[0])+1e-12
                and min(p[1], q[1])-1e-12 <= r[1] <= max(p[1], q[1])+1e-12)
    o1, o2, o3, o4 = orient(a, b, c), orient(a, b, d), orient(c, d, a), orient(c, d, b)
    if o1 != o2 and o3 != o4:
        return True
    return ((o1 == 0 and on(a, b, c)) or (o2 == 0 and on(a, b, d)) or (o3 == 0 and on(c, d, a))
            or (o4 == 0 and on(c, d, b)))


def has_suggested_area(report, evidence):
    """Whether the report's key moment `evidence` carries a suggested area: a suggestedZone element on its diagram
    (what the page's "Adjust the area" starts from, coaching.js zoneOf) or an alternative the report lists."""
    moment = next((e for e in (report or {}).get('keyMoments') or [] if isinstance(e, dict) and e.get('id') == evidence),
                  None)
    if moment is None:
        return False
    diagram = moment.get('diagram') if isinstance(moment.get('diagram'), dict) else {}
    if any(isinstance(e, dict) and e.get('type') == 'suggestedZone' for e in diagram.get('elements') or []):
        return True
    alts = {a.get('id') for a in (report or {}).get('alternatives') or [] if isinstance(a, dict)}
    return isinstance(moment.get('alternative'), str) and moment['alternative'] in alts


def add_adjustment(video_dir, run, record, finding_key, raw, now=None):
    """Store the coach's adjusted suggested area for one key moment of a finding of a stored v2 report (DESIGN 10):
    {run, team, range, report (ifp16), level, finding {...}, adjustment: schema 'adjustment' {id adj-, findingKey,
    evidence, polygon (normalized pitch), mover, note, created}}. One area per finding, moment, run, team and range (a
    new one replaces it). It never changes the result, the tracking or a measurement, and it is in no fingerprint.
    Raises KeyError (finding not shown), ValueError (a malformed area: the message says what), FeedbackUnreadable.
    Only a key moment that carries a suggested area can have an adjusted one (has_suggested_area): the probe of CODE
    L-6 (round 2) stored an area on a 'moments worth a look' review, which by design has none."""
    from . import schema as S
    report = record['report']
    finding = shown_findings(report).get(finding_key)
    if finding is None:
        raise KeyError(finding_key)
    if not isinstance(raw, dict):
        raise ValueError('adjustment must be an object {evidence, polygon, mover, note}')
    evidence = raw.get('evidence')
    cited = {t.get('evidence') for t in finding.get('timestamps') or [] if isinstance(t, dict)}
    if not isinstance(evidence, str) or evidence not in cited:
        raise ValueError('adjustment.evidence must be a key moment of this finding')
    if not has_suggested_area(report, evidence):
        raise ValueError('adjustment.evidence has no suggested area to adjust: only a key moment with a suggested area '
                         'can have an adjusted one')
    mover = raw.get('mover')
    if mover is not None and not (isinstance(mover, str) and re.fullmatch(S.PLAYER_ID, mover)):
        raise ValueError('adjustment.mover must be a player id or null')
    note = raw.get('note')
    if note is not None and not isinstance(note, str):
        raise ValueError('adjustment.note must be text or null')
    note = re.sub(r'\s+', ' ', _note(note)) or None
    if note is not None and not re.fullmatch(r'[ -~]*', note):
        raise ValueError('A note on an adjusted area is printed in the PDF: use plain letters, digits and punctuation.')
    rng = _range_of(report)
    adj = {'id': S.record_id('adj', run, report['team'], finding_key, evidence, rng['t0'], rng['t1']),
           'findingKey': finding_key, 'evidence': evidence, 'polygon': _polygon(raw.get('polygon')), 'mover': mover,
           'note': note, 'created': _iso(now)}
    problems = S.validate('adjustment', adj)
    if problems:
        raise ValueError('; '.join(problems[:4]))
    wrapper = {'run': str(run), 'team': report['team'], 'range': rng,
               'report': str(record['keys'].get('interpretation') or '')[:16],
               'level': (report.get('profile') or {}).get('level'), 'finding': _finding_summary(finding),
               'adjustment': adj}
    with _lock:
        doc = read_feedback_doc(video_dir, strict=True)
        kept = [a for a in doc['adjustments'] if (a.get('adjustment') or {}).get('id') != adj['id']]
        kept.append(wrapper)
        _write_feedback(video_dir, doc['feedback'], kept)
    return wrapper


def _well_formed_v2(e):
    text_or_none = (str, type(None))
    return (isinstance(e, dict) and isinstance(e.get('id'), str) and FEEDBACK_ID.fullmatch(e['id']) is not None
            and isinstance(e.get('run'), str) and e.get('team') in ('A', 'B') and isinstance(e.get('findingKey'), str)
            and FINDING_KEY.fullmatch(e['findingKey']) is not None and isinstance(e.get('range'), dict)
            and _number_or_none(e['range'].get('t0')) is not None and _number_or_none(e['range'].get('t1')) is not None
            and e.get('verdict') in VERDICTS and isinstance(e.get('text'), text_or_none)
            and isinstance(e.get('made'), text_or_none))


def _well_formed_adjustment(a):
    from . import schema as S
    return (isinstance(a, dict) and isinstance(a.get('run'), str) and a.get('team') in ('A', 'B')
            and isinstance(a.get('range'), dict) and _number_or_none(a['range'].get('t0')) is not None
            and _number_or_none(a['range'].get('t1')) is not None and not S.validate('adjustment', a.get('adjustment')))


def feedback_answer(report, doc, run):
    """The page's feedback list for a v2 report (FRONTEND's route table): every entry of the video, each with
    'findingKey' (None for a v1 verdict, listed by the page under "Feedback from the previous report engine"),
    'thisRun' and 'attached' (the finding key it sits on: only for this run, this report's team and range, and a finding
    the report shows). Adjusted areas come as entries with verdict None and their 'adjustment'. Entries that are not
    understood are left out (and kept in the file)."""
    report = report or {}
    shown = set(shown_findings(report))
    team, rng = report.get('team'), _range_of(report)

    def attached(e_run, e_team, key, e_range):
        ok = report and e_run == run and e_team == team and key in shown and same_range(e_range, rng)
        return key if ok else None
    out = []
    for e in doc.get('feedback') or []:
        if _well_formed(e):
            out.append(dict(e, findingKey=None, attached=None, thisRun=e['run'] == run,
                            finding={'headline': e['observation'].get('headline')}))
        elif _well_formed_v2(e):
            out.append(dict(e, attached=attached(e['run'], e['team'], e['findingKey'], e['range']),
                            thisRun=e['run'] == run))
    for a in doc.get('adjustments') or []:
        if _well_formed_adjustment(a):
            adj = a['adjustment']
            out.append({'id': adj['id'], 'engine': 2, 'run': a['run'], 'team': a['team'], 'findingKey': adj['findingKey'],
                        'range': a['range'], 'made': adj['created'], 'report': a.get('report'), 'level': a.get('level'),
                        'finding': a.get('finding') if isinstance(a.get('finding'), dict) else {'headline': None},
                        'verdict': None, 'text': adj['note'] or '', 'adjustment': adj,
                        'attached': attached(a['run'], a['team'], adj['findingKey'], a['range']),
                        'thisRun': a['run'] == run})
    return out


def is_removable_id(value):
    """A verdict id of either engine or an adjusted area's id (POST /api/coaching/feedback/remove)."""
    return is_feedback_id(value) or (isinstance(value, str) and ADJUSTMENT_ID.fullmatch(value) is not None)


# ---------- squads and progress at serve time (DESIGN 8; progress.py does the reading and writing) ----------
def progress_link(settings, run, team):
    """(link, block): the squad confirmed as this run's team (progress.links), else the team's own block's squad while
    its link to this run is not confirmed (the section then asks to confirm it), else (None, None)."""
    from . import progress as PG, schema as S
    links = PG.links(settings, run)
    for link in links:
        if link['confirmed'] and link['team'] == team:
            return link, S.team_block(settings, link['block'])
    for link in links:
        if link['block'] == team and not link['confirmed']:
            return link, S.team_block(settings, link['block'])
    return None, None


def progression_for(runs_root, settings, run, team):
    """(the report's DEVELOPMENT PROGRESSION section, schema Progression, or None; the confirmed squad's name or None),
    merged at serve time (the progress store is in no fingerprint)."""
    from . import progress as PG
    link, block = progress_link(settings, run, team)
    if link is None:
        return None, None
    entries = PG.read_entries(runs_root, link['squad']['id'])
    section = PG.section(entries, link['squad'], block, linked=link['confirmed'])
    return section, link['squad']['name'] if link['confirmed'] else None


def record_progress(runs_root, settings, run, video_key, team, obs, roles, rng, *, video_path=None, coach_date=None):
    """Write the progress entry of the squad confirmed as this run's team for the report's range (progress.record);
    [] when no squad is confirmed for it (an unconfirmed link never writes progress, DESIGN 8.1)."""
    from . import progress as PG
    if not any(link['confirmed'] and link['team'] == team for link in PG.links(settings, run)):
        return []
    return PG.record(runs_root, settings, run=run, video=video_key, tiers={team: (obs, roles)}, t0=rng.get('t0'),
                     t1=rng.get('t1'), coach_date=coach_date, video_path=video_path)


def record_progress_if_missing(runs_root, settings, run, video_key, run_dir, keys, report, *, video_path=None,
                              coach_date=None):
    """record_progress for a report that was built before its squad link was confirmed: when a squad is confirmed as
    this team of this run and has no entry for the report's range, the entry is made from the stored tier files (a
    link confirmed after the report never loses the match). Returns the rows written ([] when nothing was needed)."""
    from . import progress as PG
    link = next((x for x in PG.links(settings, run) if x['confirmed'] and x['team'] == keys['team']), None)
    if link is None:
        return []
    rng = _range_of(report)
    if any(e.get('video') == video_key and e.get('run') == run and e.get('team') == keys['team']
           and same_range(e.get('range'), rng) for e in PG.read_entries(runs_root, link['squad']['id'])):
        return []
    obs, roles = load_tier(tier_path(run_dir, 'obs', keys)), load_tier(tier_path(run_dir, 'roles', keys))
    if obs is None or obs.get('fingerprint') != keys['mfp']:
        return []
    return record_progress(runs_root, settings, run, video_key, keys['team'], obs, roles, keys['range'],
                           video_path=video_path, coach_date=coach_date)


def mark_progress_stale(runs_root, settings, run, video_key, team, mfp):
    """progress.mark_stale for this run and team with the current measurement (16 hex), only when the video has a
    squad (no progress file is touched otherwise). Returns how many entries changed."""
    from . import progress as PG
    if not PG.links(settings, run):
        return 0
    return PG.mark_stale(runs_root, video=video_key, run=run, team=team, measurement=mfp[:16])


def rederive_progress(runs_root, squad_id):
    """Re-derive the squad's entries that the current rules or engine cannot read (progress.needs_rederive) from their
    own cached obs (and roles) files when those are still stored; a stale entry (another measurement of its run) is
    left for the next report of that run to replace. Returns how many were re-derived."""
    from . import progress as PG
    done = 0
    for e in list(PG.read_entries(runs_root, squad_id)):
        if e.get('stale') or PG.needs_rederive(e) is None or not (_safe_name(e.get('video')) and _safe_name(e.get('run'))):
            continue
        folder = Path(runs_root)/e['video']/e['run']/'coaching'/V2_DIR
        mfp16 = (e.get('engine') or {}).get('measurement')
        if not (isinstance(mfp16, str) and re.fullmatch(r'[0-9a-f]{16}', mfp16)):
            continue
        obs = load_tier(folder/f'obs-{e["team"]}-{mfp16}.json')
        if obs is None:
            continue
        roles = None
        for p in sorted(folder.glob(f'roles-{e["team"]}-{mfp16}-*.json'), key=lambda q: q.stat().st_mtime_ns,
                        reverse=True):
            cand = load_tier(p)
            if cand is not None and cand.get('measurement') == obs.get('fingerprint'):
                roles = cand
                break
        try:
            PG.upsert(runs_root, PG.rederive(e, obs, roles))
        except (PG.ProgressError, ValueError, KeyError, TypeError):
            continue
        done += 1
    return done
