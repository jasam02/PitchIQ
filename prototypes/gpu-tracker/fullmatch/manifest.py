"""The match manifest: everything about one full-match analysis that must survive a crash, in one JSON file,
runs/<video>/matches/<match id>/manifest.json.

It holds the source (upload key = sha256 of the file, frame count, fps, size), the periods and the segment plan
(plan.py), the processing settings and their fingerprint, the calibration keyframes shared by all segments, the
worker configuration, and per segment its state (pending, running, done, failed), attempts, last error, timings and
the artifact folder. The match itself is planned, running, paused, stopped, assembling, complete or failed.

Writes are atomic (a temporary file in the same folder, flushed, then os.replace, retried while Windows holds the old
file for a moment), so a crash leaves the previous version or the new one, never half a file. A segment is only done
once its artifact is verified: the worker writes the artifact into a temporary folder, keeps the result compressed
(compress_result: result.json.gz, read back against the original before the original goes), writes a small
segment.json with the facts of the result last (frames, first and last frame, status, the stored file, its size and
sha256), and renames the folder; verify_artifact() then checks the owned frames are covered and the files are those
segment.json describes. A result that does not match its sha256 (a write cut short) is not done. An artifact from
before the compression (a plain result.json, a segment.json that names no file) still verifies and reads (read_result).

Resuming checks compatibility: the same source, the same frames and fps, the same plan and the same processing
settings fingerprint; a calibration change invalidates the segments within reach of the keyframes that changed
(calibration_changes). Anything else incompatible means a new analysis, never a silent mix.
"""
import gzip
import hashlib
import json
import os
import shutil
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path

from . import plan as planning

VERSION = 1
MATCH_STATES = ('planned', 'running', 'paused', 'stopped', 'assembling', 'complete', 'failed')
SEGMENT_STATES = ('pending', 'running', 'done', 'failed')
SIDECAR = 'segment.json'
RESULT = 'result.json'          # what a run writes
RESULT_GZ = 'result.json.gz'    # how a match keeps it (compress_result)
RESULT_GZIP_LEVEL = 6           # 8 CAN-ISL and broadcast segment results (70-93 MiB): 10.4-13.4x smaller, 1.2-2.0 KB a
                                # frame, in .6-1.1 s (level 5: 9.8-12.4x in .5-.7 s; level 9: 11.2-14.4x in 1.1-2.7 s)
REQUIRED = ('identities.json', 'registration.json')   # and the result, plain or compressed (segment.json names it)
# Settings that change what a segment run produces (the page's processing settings); start/seconds are the plan's.
RESULT_SETTINGS = ('imageSize', 'ballTiles', 'boundaries', 'fieldFilter', 'touchlineMargin', 'maxPerTeam', 'debug',
                   'detectBatch', 'ballScanEvery', 'nearPass')
ROLL_SECONDS = 15.0   # pipeline.ROLL_SECONDS: a keyframe this close to a segment's processed frames changes its run
REPLACE_TRIES = 20
TAIL_TOLERANCE = 30   # frames a period's last segment may end short of its owned end (see verify_artifact)


def now():
    return datetime.now().isoformat(timespec='seconds')


def new_id():
    return datetime.now().strftime('%Y%m%d-%H%M%S')+'-'+uuid.uuid4().hex[:6]


def canonical(data):
    return json.dumps(data, sort_keys=True, separators=(',', ':'), allow_nan=False)


def fingerprint(data):
    return hashlib.sha256(canonical(data).encode('utf-8')).hexdigest()


def settings_fingerprint(settings):
    """sha256 of the settings that change a segment's result (RESULT_SETTINGS; the calibrations are tracked per keyframe)."""
    return fingerprint({k: settings.get(k) for k in RESULT_SETTINGS})


def _replace(source, target):
    for attempt in range(REPLACE_TRIES):
        try:
            os.replace(source, target)
            return
        except PermissionError:
            # Windows: another process (an indexer, a virus scanner, a reader) holds the target for a moment.
            if attempt == REPLACE_TRIES-1:
                raise
            time.sleep(.05*(attempt+1))


def write_json(path, data, compact=False):
    """Atomic: a temporary file next to the target, flushed to disk, then renamed over it."""
    path = Path(path)
    temporary = path.with_name(f'.{path.name}.{uuid.uuid4().hex[:8]}.tmp')
    text = json.dumps(data, separators=(',', ':')) if compact else json.dumps(data, indent=1)
    with temporary.open('w', encoding='utf-8') as out:
        out.write(text)
        out.flush()
        os.fsync(out.fileno())
    try:
        _replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def file_sha256(path, chunk=8*1024*1024):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def sidecar(result, result_path):
    """The facts of a finished segment result that verify_artifact() checks later (the worker writes them last)."""
    frames = result.get('frames') or []
    indices = [f['frame'] for f in frames]
    return {'status': result.get('status'), 'frames': len(frames), 'first': indices[0] if indices else None,
            'last': indices[-1] if indices else None,
            'contiguous': bool(indices) and all(b == a+1 for a, b in zip(indices, indices[1:])),
            'firstTime': frames[0]['time'] if frames else None, 'lastTime': frames[-1]['time'] if frames else None,
            'resultFile': Path(result_path).name, 'resultBytes': Path(result_path).stat().st_size,
            'resultSha256': file_sha256(result_path)}


def _unlink(path):
    for attempt in range(REPLACE_TRIES):
        try:
            Path(path).unlink()
            return
        except PermissionError:
            # Windows: a virus scanner or an indexer looks at a file just written for a moment.
            if attempt == REPLACE_TRIES-1:
                raise
            time.sleep(.05*(attempt+1))


def compress_result(folder, level=RESULT_GZIP_LEVEL, chunk=8*1024*1024):
    """Keep a finished run's result.json gzip-compressed, as result.json.gz: the plain results were 2.56 of the 2.87 GB
    the 96.5-minute CAN-ISL match took on disk. Written under a temporary name, read back and compared with the original
    (the length and sha256 of what it decompresses to), renamed into place, and only then is the original deleted. The
    same bytes in, the same bytes out (no name or time in the gzip header). Returns the stored path and the original's
    size."""
    folder = Path(folder)
    source, target = folder/RESULT, folder/RESULT_GZ
    temporary = folder/f'.{RESULT_GZ}.{uuid.uuid4().hex[:8]}.tmp'
    digest, size = hashlib.sha256(), 0
    try:
        with source.open('rb') as handle, temporary.open('wb') as raw:
            with gzip.GzipFile(filename='', mode='wb', compresslevel=level, fileobj=raw, mtime=0) as out:
                while block := handle.read(chunk):
                    digest.update(block)
                    size += len(block)
                    out.write(block)
            raw.flush()
            os.fsync(raw.fileno())
        check, length = hashlib.sha256(), 0
        with gzip.open(temporary, 'rb') as back:
            while block := back.read(chunk):
                check.update(block)
                length += len(block)
        if length != size or check.digest() != digest.digest():
            raise OSError(f'{RESULT_GZ} did not read back as the {RESULT} it was made from')
        _replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    _unlink(source)
    return target, size


def read_result(folder):
    """A segment's result: result.json.gz (compress_result), or the plain result.json of an artifact made before."""
    folder = Path(folder)
    if (folder/RESULT_GZ).is_file():
        return json.loads(gzip.decompress((folder/RESULT_GZ).read_bytes()).decode('utf-8'))
    return json.loads((folder/RESULT).read_text(encoding='utf-8'))


def verify_artifact(folder, segment, deep=True):
    """None when the artifact folder holds a complete, untruncated result that covers the segment's owned frames, else
    the reason it is not done. deep: also recompute the sha256 of the stored result (about 0.3 s for 100 MB)."""
    folder = Path(folder)
    if not folder.is_dir():
        return 'no artifact folder'
    try:
        facts = json.loads((folder/SIDECAR).read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return 'no segment.json (the run did not finish writing)'
    stored = facts.get('resultFile') or RESULT   # a segment.json from before the compression names none: the plain file
    if stored not in (RESULT, RESULT_GZ):
        return f'segment.json names an unknown result file {stored!r}'
    for name in (stored,)+REQUIRED:
        if not (folder/name).is_file():
            return f'{name} is missing'
    if facts.get('status') != 'complete':
        return f"the run ended as {facts.get('status')!r}"
    if not facts.get('contiguous'):
        return 'the result skips frames'
    first, last = facts.get('first'), facts.get('last')
    # At the end of a period with no context after it (the end of the video, typically), a decoder whose timestamps run
    # ahead of frame/fps ends the run up to a second early (the pipeline stops at the duration frames/fps): the frames
    # it never decoded are not owned by anyone; the assembly cuts at the last one there is.
    short = TAIL_TOLERANCE if segment.proc1 == segment.own1 else 0
    if first is None or first > segment.own0 or last < segment.own1-1-short:
        return f'the result covers frames {first}-{last}, not the owned {segment.own0}-{segment.own1-1}'
    result = folder/stored
    if result.stat().st_size != facts.get('resultBytes'):
        return f'{stored} has the wrong size (a write was cut short)'
    if deep and file_sha256(result) != facts.get('resultSha256'):
        return f'{stored} does not match its checksum'
    return None


def calibration_changes(old, new, segments, fps):
    """Indices of the segments a change of the calibration keyframes affects: a keyframe added, removed or moved within
    ROLL_SECONDS of the segment's processed frames. (Every keyframe also sets the learned pitch size; the scheduler
    compares the sizes separately.)"""
    def key(c):
        return (round(float(c['time']), 3), canonical(c.get('points')))
    a, b = {key(c) for c in old or []}, {key(c) for c in new or []}
    changed = [t for t, _ in a ^ b]
    hit = []
    for s in segments:
        t0, t1 = s.proc0/fps-ROLL_SECONDS, s.proc1/fps+ROLL_SECONDS
        if any(t0 <= t <= t1 for t in changed):
            hit.append(s.index)
    return hit


class Manifest:
    """One match's manifest in memory, saved atomically after every change. Thread-safe: the scheduler thread and the
    server's request threads share it."""

    def __init__(self, path, data):
        self.path = Path(path)
        self.data = data
        self.lock = threading.RLock()

    @property
    def folder(self):
        return self.path.parent

    @classmethod
    def create(cls, folder, source, video, periods, segments, settings, workers, pad_s, segment_s, extra=None):
        """A new manifest: source = {'key', 'path', 'bytes'}; video = pipeline.metadata() of the source."""
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        fps = float(video['fps'])
        data = {'version': VERSION, 'matchId': folder.name, 'created': now(), 'updated': now(), 'state': 'planned',
                'source': dict(source), 'video': {k: video[k] for k in ('name', 'width', 'height', 'fps', 'frames', 'duration')},
                'periods': [{'index': i, 'start': a, 'end': b, 'name': period_name(i, len(periods))} for i, (a, b) in enumerate(periods)],
                'plan': {'segmentSeconds': segment_s, 'padSeconds': pad_s, 'segments': len(segments),
                         'processedFrames': planning.processed_frames(segments)},
                'settings': settings, 'settingsFingerprint': settings_fingerprint(settings),
                'calibrations': settings.get('calibrations', []),
                'workers': dict(workers),
                'segments': [s.to_json(fps) | {'state': 'pending', 'attempts': 0, 'error': None, 'artifact': None}
                             for s in segments],
                'assembly': {'state': 'pending', 'rosterVersion': 0},
                'history': [{'at': now(), 'event': 'created'}]}
        if extra:
            data.update(extra)
        manifest = cls(folder/'manifest.json', data)
        manifest.save()
        return manifest

    @classmethod
    def load(cls, path):
        path = Path(path)
        data = json.loads(path.read_text(encoding='utf-8'))
        if data.get('version') != VERSION:
            raise ValueError(f"Unknown manifest version {data.get('version')!r}.")
        manifest = cls(path, data)
        planning.check(manifest.segments())
        return manifest

    def save(self):
        with self.lock:
            self.data['updated'] = now()
            write_json(self.path, self.data)

    def segments(self):
        return [planning.Segment.from_json(s) for s in self.data['segments']]

    def entry(self, index):
        return self.data['segments'][index]

    def update_segment(self, index, **fields):
        with self.lock:
            self.data['segments'][index].update(fields)
            self.save()

    def set_state(self, state, **fields):
        if state not in MATCH_STATES:
            raise ValueError(f'Unknown match state {state!r}.')
        with self.lock:
            self.data['state'] = state
            self.data.update(fields)
            self.data['history'].append({'at': now(), 'event': state})
            self.data['history'] = self.data['history'][-200:]
            self.save()

    def note(self, event, **fields):
        with self.lock:
            self.data['history'].append({'at': now(), 'event': event} | fields)
            self.data['history'] = self.data['history'][-200:]
            self.save()

    def artifact_folder(self, index):
        entry = self.entry(index)
        return self.folder/'segments'/entry['key']

    def compatible(self, source_key, frames, fps, settings):
        """None if this manifest's segments can be reused for a run with this source and settings, else why not."""
        source, video = self.data['source'], self.data['video']
        if source.get('key') != source_key:
            return 'the source video is a different file'
        if int(video['frames']) != int(frames) or abs(float(video['fps'])-float(fps)) > 1e-9:
            return 'the video reports different frames or frame rate'
        if self.data['settingsFingerprint'] != settings_fingerprint(settings):
            return 'the processing settings changed'
        return None

    def reconcile(self, deep=False):
        """On load or resume: a segment marked done whose artifact no longer verifies goes back to pending; a segment left
        'running' by a crash goes back to pending; a verified artifact of a segment marked pending (the manifest write
        was lost) counts as done. Returns the indices whose state changed."""
        changed = []
        with self.lock:
            for s in self.segments():
                entry = self.entry(s.index)
                folder = self.artifact_folder(s.index)
                problem = verify_artifact(folder, s, deep=deep) if folder.exists() else 'no artifact folder'
                if entry['state'] == 'done' and problem:
                    entry.update(state='pending', error=f'artifact no longer valid: {problem}', artifact=None)
                    changed.append(s.index)
                elif entry['state'] == 'running':
                    entry.update(state='pending', error='interrupted (the app or the worker stopped)')
                    changed.append(s.index)
                elif entry['state'] in ('pending', 'failed') and problem is None:
                    entry.update(state='done', error=None, artifact=str(folder.relative_to(self.folder)))
                    changed.append(s.index)
            if changed:
                self.save()
        return changed

    def counts(self):
        out = {k: 0 for k in SEGMENT_STATES}
        for entry in self.data['segments']:
            out[entry['state']] = out.get(entry['state'], 0)+1
        return out

    def remove_partials(self):
        """Delete temporary artifact folders left by a crash (never a verified one)."""
        gone = []
        root = self.folder/'segments'
        for path in root.glob('*.partial-*') if root.exists() else []:
            shutil.rmtree(path, ignore_errors=True)
            gone.append(path.name)
        return gone


def period_name(index, count):
    if count == 1:
        return 'match'
    return ('first half', 'second half', 'extra time, first half', 'extra time, second half')[index] if index < 4 else f'period {index+1}'


def find(directory):
    """Manifests of one video's matches (runs/<video>/matches/*/manifest.json), newest first by creation (a match folder
    may have any name, not only new_id's timestamp: a folder named dev12 sorted before the newer timestamped ones and
    was opened as the latest); unreadable ones skipped."""
    root = Path(directory)/'matches'
    out = []
    for path in sorted(root.glob('*/manifest.json'), reverse=True) if root.exists() else []:
        try:
            out.append(Manifest.load(path))
        except (OSError, ValueError, KeyError, TypeError):
            continue
    out.sort(key=lambda m: str(m.data.get('created') or ''), reverse=True)   # stable: the same second keeps the name order
    return out
