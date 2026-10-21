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

Files (runs/ is gitignored):
  runs/<video>/<run>/coaching/report-<team>-<fp16>.json   one per fingerprint, the newest KEEP_REPORTS per team kept
  runs/<video>/coaching_feedback.json                    per video, like corrections.json and team_shape.json
  runs/<video>/coaching/frames/f<frame>-w<width>.jpg      the raw pixels of an evidence frame, shared by every run of
                                                         the video (the folder name pins the video's content). No
                                                         overlay is baked in: display labels renumber after
                                                         corrections, so overlays are drawn when shown.
Writes are atomic (a uuid temporary file and os.replace, retried while a Windows reader holds the old file, as
corrections._write does). The feedback file is read strictly before every write: an unreadable or unparsable file is an
error, never an empty list, so a transient read failure cannot wipe a coach's earlier verdicts (corrections.load
returns [] on OSError; that is the pattern avoided here).
"""
import hashlib
import json
import math
import os
import re
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path

from . import episodes

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
DRILLS_FILE = PACKAGE/'drills.json'
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
UNPLACED = ('excluded', 'withdrawn')   # match groups of identities that are no longer players (corrections.ALL_GROUPS)

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


def read_feedback(video_dir, strict=False):
    """The stored feedback entries. Missing file: []. While the file is being replaced a read is retried. In strict
    mode (before a write) an unreadable or unparsable file raises FeedbackUnreadable and every stored dict is returned
    as it is, so a write keeps entries it does not understand; otherwise those count as no feedback and only
    well-formed entries are returned. A NaN or Infinity in the file (hand-edited) reads as null, so the entry keeps its
    other fields and no non-finite number reaches a response."""
    path = feedback_path(video_dir)
    data = None
    for attempt in range(READ_TRIES):
        try:
            if not path.exists():
                return []
            data = loads_lenient(path.read_text(encoding='utf-8'))
            break
        except (ValueError, TypeError, RecursionError):
            if strict:
                raise FeedbackUnreadable('The coaching feedback file could not be read; nothing was changed.') from None
            return []
        except OSError:
            if attempt == READ_TRIES-1:
                if strict:
                    raise FeedbackUnreadable('The coaching feedback file could not be read; nothing was changed.') from None
                return []
            time.sleep(READ_WAIT)
    entries = data.get('feedback') if isinstance(data, dict) else None
    if not isinstance(entries, list):
        if strict:
            raise FeedbackUnreadable('The coaching feedback file is not in the expected format; nothing was changed.')
        return []
    if strict:
        return [e for e in entries if isinstance(e, dict)]
    return [e for e in entries if _well_formed(e)]


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
        entries = read_feedback(video_dir, strict=True)
        kept = [e for e in entries if not (e.get('run') == entry['run'] and (e.get('observation') or {}).get('key') == observation_id)]
        kept.append(entry)
        write_json(feedback_path(video_dir), {'version': 1, 'feedback': kept}, indent=1)
    return entry


def remove_feedback(video_dir, feedback_id):
    """Remove one feedback entry by id. Returns False when there is no such entry (nothing is written)."""
    with _lock:
        entries = read_feedback(video_dir, strict=True)
        kept = [e for e in entries if e.get('id') != feedback_id]
        if len(kept) == len(entries):
            return False
        write_json(feedback_path(video_dir), {'version': 1, 'feedback': kept}, indent=1)
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
    """{absolute source frame: time} of a record's evidence: the only frames the screenshot route serves, with the
    time the decoder must agree with."""
    out = {}
    for e in (record.get('report') or {}).get('evidence') or []:
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


def render_pdf(report, labels, frame, stale=None):
    """The PDF of a report: soccer.coaching.pdf.report_pdf(report, labels={id: display}, frame=provider(frame, width)
    -> JPEG bytes or None (an EvidenceFrames), stale=[reason texts] or None). Raises PdfUnavailable when the package
    does not provide the writer."""
    try:
        from . import pdf
    except ImportError as error:
        raise PdfUnavailable('PDF export is not available in this version yet.') from error
    write = getattr(pdf, PDF_WRITER, None)
    if write is None:
        raise PdfUnavailable('PDF export is not available in this version yet.')
    return write(report, labels=labels, frame=frame, stale=stale)
