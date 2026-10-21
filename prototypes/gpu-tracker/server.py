"""Run: start.cmd (then upload a video in the page) or start.cmd "C:\\path\\match.mp4". Local-only, port 8765."""
import argparse
import hashlib
import importlib
import json
import functools
import math
import threading
import time
import traceback
import uuid
from datetime import datetime
from pathlib import Path
from urllib.parse import unquote

from flask import Flask, abort, jsonify, request, send_file
from waitress import serve
import corrections
from download_model import get_model
from pipeline import RESOURCES, ROOT, gpu_info, metadata, run
from soccer import analytics, calibration, team_shape


def coaching_store():
    """soccer/coaching/store.py, imported by the first coaching request instead of at start: a broken coaching package
    then fails only the coaching routes (503 with the error), never the server's start or the other views. Once loaded
    this is a sys.modules lookup."""
    from soccer.coaching import store
    return store


def possession_engine():
    """soccer/possession.py, imported by the first possession request instead of at start (as coaching_store): a missing
    or broken possession module then answers 503 on /api/possession only, never stops the server's start or the other
    views. importlib looks in sys.modules first, so once loaded this is a lookup."""
    return importlib.import_module('soccer.possession')


class Refused(Exception):
    """A request the coaching routes turn down, with the HTTP status to answer it with."""

    def __init__(self, message, status=400):
        super().__init__(message)
        self.status = status


def _number(payload, key, default, low, high, message):
    value = float(payload.get(key, default))
    if not math.isfinite(value) or not low <= value <= high:
        raise ValueError(message)
    return value


def validate(payload, meta):
    duration = meta['duration']
    if not isinstance(payload, dict):
        raise ValueError('Expected a JSON settings object.')
    start, seconds = float(payload.get('start', 0)), float(payload.get('seconds', 60))
    if not math.isfinite(start) or not 0 <= start < duration:
        raise ValueError('Start time must be inside the video.')
    if not math.isfinite(seconds) or not .5 <= seconds <= 180:
        raise ValueError('Choose between 0.5 and 180 seconds for this prototype.')
    image_size = payload.get('imageSize', 1280)
    if image_size not in (960, 1280, 1536):
        raise ValueError('Choose one of the supported detection resolutions.')
    boundaries = payload.get('boundaries', [])
    if not isinstance(boundaries, list) or len(boundaries) > 100:
        raise ValueError('At most 100 field keyframes are supported.')
    clean = []
    for anchor in boundaries:
        timestamp = float(anchor['time'])
        if not math.isfinite(timestamp) or not 0 <= timestamp < duration:
            raise ValueError('Invalid field-keyframe time.')
        points = anchor['points']
        if not isinstance(points, list) or not 3 <= len(points) <= 32:
            raise ValueError('A field boundary needs 3 to 32 points.')
        if any(not isinstance(p, list) or len(p) != 2 or
               any(not isinstance(v, (int, float)) or not math.isfinite(v) or not 0 <= v <= 1 for v in p) for p in points):
            raise ValueError('Field points must lie inside the image.')
        clean.append({'time': timestamp, 'points': points})
    if len(set(a['time'] for a in clean)) != len(clean):
        raise ValueError('Duplicate boundary times; replace the existing keyframe.')
    calibrations = payload.get('calibrations', [])
    if not isinstance(calibrations, list) or len(calibrations) > 100:
        raise ValueError('At most 100 calibration keyframes are supported.')
    marks = []
    for anchor in calibrations:
        timestamp = float(anchor['time'])
        if not math.isfinite(timestamp) or not 0 <= timestamp < duration:
            raise ValueError('Invalid calibration-keyframe time.')
        points = anchor['points']
        if not isinstance(points, list) or not 4 <= len(points) <= 30:
            raise ValueError('A calibration keyframe needs 4 to 30 landmarks.')
        clean_points = []
        for p in points:
            name, x, y = p['name'], p['x'], p['y']
            if name not in calibration.LANDMARKS or any(not isinstance(v, (int, float)) or not math.isfinite(v) or not 0 <= v <= 1 for v in (x, y)):
                raise ValueError('Calibration landmarks need a known name and a point inside the image.')
            clean_points.append({'name': name, 'x': float(x), 'y': float(y)})
        try:
            # A non-standard pitch (youth, small-sided) disagrees with the standard size by a few metres;
            # the run learns the size from all keyframes together, so only gross errors are refused here.
            calibration.solve(clean_points, meta['width'], meta['height'], max_error=4.0)
        except ValueError as error:
            raise ValueError(f'Calibration at {timestamp:.2f}s: {error}')
        marks.append({'time': timestamp, 'points': clean_points})
    if len(set(a['time'] for a in marks)) != len(marks):
        raise ValueError('Duplicate calibration times; replace the existing keyframe.')
    margin = _number(payload, 'touchlineMargin', .35, 0, 1, 'Touchline tolerance must be between 0% and 100% of player height.')
    per_team = int(_number(payload, 'maxPerTeam', 11, 5, 30, 'Expected identities per team must be between 5 and 30.'))
    settings = {'start': start, 'seconds': seconds, 'imageSize': image_size,
                'ballTiles': bool(payload.get('ballTiles', False)), 'boundaries': sorted(clean, key=lambda a: a['time']),
                'calibrations': sorted(marks, key=lambda a: a['time']), 'fieldFilter': bool(payload.get('fieldFilter', True)),
                'touchlineMargin': margin, 'maxPerTeam': per_team, 'continueIdentities': bool(payload.get('continueIdentities', False)),
                'debug': bool(payload.get('debug', True))}
    if 'detectBatch' in payload:
        # Frames per whole-frame detector call on the prefetch worker; the page does not set it (pipeline.DETECT_BATCH applies).
        settings['detectBatch'] = int(_number(payload, 'detectBatch', 1, 1, 16, 'Detector batch must be between 1 and 16 frames.'))
    if 'ballScanEvery' in payload:
        # Seconds between the safeguard ball tile scans while the ball is locked; the page does not set it (pipeline.BALL_SCAN_S applies).
        settings['ballScanEvery'] = _number(payload, 'ballScanEvery', 1.0, .04, 30, 'The ball scan interval must be between 0.04 and 30 seconds.')
    return settings


UPLOADS = ROOT/'uploads'
VIDEO_TYPES = {'.mp4', '.m4v', '.mov', '.mkv', '.avi', '.webm'}
MAX_JSON = 512*1024
MAX_UPLOAD = 64*1024**3


class Video:
    """The video currently open for review: its metadata, run folder and saved settings."""

    def __init__(self, source, name=None):
        self.source = Path(source).resolve(strict=True)
        self.meta = metadata(self.source)
        if name:
            self.meta['name'] = name
        with self.source.open('rb') as handle:
            self.key = hashlib.file_digest(handle, 'sha256').hexdigest()
        self.directory = ROOT/'runs'/self.key[:16]
        self.directory.mkdir(parents=True, exist_ok=True)
        self.last = self.directory/'latest.json'
        self.saved = self.directory/'settings.json'
        self.result_dir = None
        if self.last.exists():
            try:
                candidate = self.directory/json.loads(self.last.read_text())['run']
                if candidate.parent == self.directory and (candidate/'result.json').exists():
                    self.result_dir = candidate
            except (ValueError, KeyError, TypeError, OSError):
                pass
        self.defaults = validate({'seconds': max(.5, min(60, self.meta['duration']))}, self.meta)
        if self.saved.exists():
            try:
                self.defaults = validate(json.loads(self.saved.read_text()), self.meta)
            except (ValueError, KeyError, TypeError):
                pass

    def previous_identities(self):
        """Saved global identities of the latest completed run of this video, if any."""
        if not self.last.exists():
            return None
        try:
            run_dir = self.directory/json.loads(self.last.read_text())['run']
            if run_dir.parent != self.directory or not (run_dir/'identities.json').exists():
                return None
            return json.loads((run_dir/'identities.json').read_text())
        except (ValueError, KeyError, TypeError, OSError):
            return None


def uploaded_videos():
    """Videos uploaded through the page earlier (newest first)."""
    index = UPLOADS/'index.json'
    try:
        names = json.loads(index.read_text()) if index.exists() else {}
    except (ValueError, OSError):
        names = {}
    out = []
    for path in sorted(UPLOADS.glob('*'), key=lambda p: p.stat().st_mtime, reverse=True) if UPLOADS.exists() else []:
        if path.suffix.lower() in VIDEO_TYPES and path.is_file():
            out.append({'file': path.name, 'name': names.get(path.name, path.name), 'sizeMB': round(path.stat().st_size/1024**2, 1)})
    return out


def preload_models():
    """Load the detectors and the appearance embedder (pipeline.RESOURCES) in the background while the page comes
    up: they stay loaded in this process for every run, so no run pays the model loading again. A run that
    starts before they are ready waits for them (Resources.load holds a lock); a failure here is only printed
    and the first run loads them itself."""
    def load():
        try:
            RESOURCES.load(get_model())
            print('Detectors and appearance model loaded; they stay loaded between runs.', flush=True)
        except Exception as error:   # noqa: BLE001 - the first run reports the real error
            print(f'Models not preloaded ({error}); the first run will load them.', flush=True)
    threading.Thread(target=load, name='preload', daemon=True).start()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', help='Optional: open this video at start. Videos can also be uploaded in the page.')
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--run', action='store_true', help='Process the first 60 seconds immediately (needs --video).')
    args = parser.parse_args()
    gpu = gpu_info()
    preload_models()
    active = {'video': None}
    if args.video:
        source = Path(args.video)
        if not source.is_file():
            parser.error('Supply a video file.')
        active['video'] = Video(source)
    app = Flask(__name__, static_folder=None)
    gate = threading.Lock()
    cancel = threading.Event()
    state = {'state': 'idle', 'stage': 'Ready', 'progress': 0, 'gpu': gpu}
    if active['video'] and active['video'].result_dir:
        state.update(state='complete', stage='Previous result loaded', progress=1)

    def need_video():
        if active['video'] is None:
            abort(404)
        return active['video']

    def switch(video):
        with gate:
            if state['state'] == 'running':
                raise ValueError('Stop the current run before opening another video.')
            active['video'] = video
            state.clear()
            state.update(state='complete' if video.result_dir else 'idle', stage='Previous result loaded' if video.result_dir else 'Ready',
                         progress=1 if video.result_dir else 0, gpu=gpu)
        print(f'Video: {video.source}\nResults: {video.directory}', flush=True)

    @app.before_request
    def local_only():
        if request.host not in {f'127.0.0.1:{args.port}', f'localhost:{args.port}'}:
            abort(403)
        origin = request.headers.get('Origin')
        if origin and origin not in {f'http://127.0.0.1:{args.port}', f'http://localhost:{args.port}'}:
            abort(403)
        if request.method == 'POST':
            if request.path == '/api/upload':
                # A custom header cannot be sent cross-site without a CORS preflight this server never allows.
                if not request.headers.get('X-Filename'):
                    abort(400)
            elif not request.is_json:
                abort(415)
            elif (request.content_length or 0) > MAX_JSON:
                abort(413)

    @app.after_request
    def headers(response):
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['Cache-Control'] = 'no-store'
        return response

    @app.get('/')
    def index():
        return send_file(ROOT/'index.html')

    @app.get('/tactical.js')
    def tactical():
        return send_file(ROOT/'tactical.js', mimetype='application/javascript')

    @app.get('/analytics.js')
    def analytics_script():
        return send_file(ROOT/'analytics.js', mimetype='application/javascript')

    @app.get('/teamshape.js')
    def team_shape_script():
        return send_file(ROOT/'teamshape.js', mimetype='application/javascript')

    @app.get('/coaching.js')
    def coaching_script():
        return send_file(ROOT/'coaching.js', mimetype='application/javascript')

    @app.get('/possession.js')
    def possession_script():
        return send_file(ROOT/'possession.js', mimetype='application/javascript')

    # The observed samples of the shown result, extracted once per result file (a correction rewrites the file, so its
    # modification time changes and the samples are rebuilt with the corrected roster).
    samples_cache = {}
    samples_lock = threading.Lock()

    def samples_entry(video):
        """(stamp, samples, slim result) of the shown result as one consistent entry: the samples and the slim result
        always come from the same version of the file."""
        path = video.result_dir/'result.json' if video.result_dir else None
        if path is None or not path.exists():
            abort(404)
        stamp = (str(path), path.stat().st_mtime_ns, path.stat().st_size)
        with samples_lock:
            cached = samples_cache.get('current')
            if cached is not None and cached[0] == stamp:
                return cached
        result = json.loads(path.read_text(encoding='utf-8'))
        built = analytics.Samples(result)
        # What the team shape analysis needs besides the samples, without the frames' people (they are in the samples).
        slim = {'frames': [{'time': f['time'], 'view': f.get('view')} for f in result.get('frames') or []],
                'roster': result.get('roster'), 'tactical': result.get('tactical'), 'match': result.get('match'), 'timeline': result.get('timeline'),
                'corrections': result.get('corrections')}
        entry = (stamp, built, slim)
        with samples_lock:
            samples_cache['current'] = entry
        return entry

    def samples_of(video):
        return samples_entry(video)[1]

    shape_cache = {}
    # One writer at a time for the team-shape settings (read, change, write), so two saves never lose each other.
    settings_lock = threading.Lock()

    def shape_settings_path(video):
        return video.directory/'team_shape.json'

    def shape_settings(video, strict=False):
        """The stored settings, cleaned. A file that cannot be parsed counts as no settings. A file that cannot be read
        (on Windows briefly, while it is being replaced) is retried, and in strict mode (before a write) an unreadable
        file raises instead of counting as empty, so a save never wipes the stored settings."""
        path = shape_settings_path(video)
        for attempt in range(8):
            try:
                if not path.exists():
                    return team_shape.clean_settings({})
                return team_shape.clean_settings(json.loads(path.read_text(encoding='utf-8')))
            except (ValueError, TypeError, OverflowError, RecursionError):
                return team_shape.clean_settings({})
            except OSError:
                if attempt == 7:
                    if strict:
                        raise
                    return team_shape.clean_settings({})
                time.sleep(.05)

    def save_shape_settings(video, settings):
        clean = team_shape.clean_settings(settings)
        path = shape_settings_path(video)
        tmp = path.with_name(f'.team_shape.{uuid.uuid4().hex}.tmp')
        tmp.write_text(json.dumps(clean, indent=1), encoding='utf-8')
        for attempt in range(10):
            try:
                tmp.replace(path)
                break
            except PermissionError:
                # Windows refuses to replace a file that another request is reading at this moment: try again shortly.
                if attempt == 9:
                    tmp.unlink(missing_ok=True)
                    raise
                time.sleep(.05)
        return clean

    def update_shape_settings(video, change):
        """Read, change and write the settings under the lock. change(current) edits them in place."""
        with settings_lock:
            try:
                current = shape_settings(video, strict=True)
            except OSError:
                return jsonify(error='The team-shape settings file could not be read; nothing was changed.'), 503
            change(current)
            try:
                return jsonify(save_shape_settings(video, current))
            except OSError:
                return jsonify(error='The team-shape settings could not be saved; try again.'), 503

    def same_period(c, team, t0, t1):
        """A stored label of `team` for the period [t0, t1]: the same bounds, or one the analysis would attach to that
        period (soccer/team_shape._overlaps), so saving replaces it and removing removes it, even after the periods
        moved."""
        if c['team'] != team:
            return False
        return (abs(c['t0']-t0) < .01 and abs(c['t1']-t1) < .01) or team_shape._overlaps(c, {'t0': t0, 't1': t1})

    def shape_of(video, mode):
        """The team shape of the shown result for a position mode and the video's settings, cached until the result
        (an identity correction rewrites it) or the settings change."""
        stamp, samples, slim = samples_entry(video)
        settings = shape_settings(video)
        key = (stamp, mode, json.dumps(settings, sort_keys=True))
        with samples_lock:
            cached = shape_cache.get(mode)
            if cached is not None and cached[0] == key:
                return cached[1]
        built = team_shape.TeamShape(slim, samples=samples, mode=mode, settings=settings)
        with samples_lock:
            shape_cache[mode] = (key, built)
        return built

    def shape_args():
        team = request.args.get('team', 'A')
        mode = request.args.get('mode', 'observed')
        if team not in ('A', 'B'):
            raise ValueError('team must be A or B')
        if mode not in ('observed', 'estimated'):
            raise ValueError('mode must be observed or estimated')
        out = {'team': team, 'mode': mode}
        for key in ('t0', 't1', 't'):
            if key in request.args:
                value = float(request.args[key])
                if not math.isfinite(value):
                    raise ValueError(f'{key} must be seconds')
                out[key] = value
        return out

    @app.get('/api/team-shape')
    def team_shape_report():
        """Team width and depth, defensive line and formations of one team over a time range, from the saved result."""
        v = need_video()
        try:
            a = shape_args()
        except (TypeError, ValueError) as error:
            return jsonify(error=str(error)), 400
        return jsonify(shape_of(v, a['mode']).report(a['team'], a.get('t0'), a.get('t1')))

    @app.get('/api/team-shape/snapshot')
    def team_shape_snapshot():
        """The measured players, width and depth, defensive line and formation roles of one team at one time."""
        v = need_video()
        try:
            a = shape_args()
            if 't' not in a:
                raise ValueError('t is required')
        except (TypeError, ValueError) as error:
            return jsonify(error=str(error)), 400
        return jsonify(shape_of(v, a['mode']).snapshot(a['team'], a['t']))

    @app.post('/api/team-shape/settings')
    def team_shape_settings():
        """Set which goal team A defends (and when the sides swap) and the outfield lineup per team. Stored apart from the
        tracking data, in the video's run folder."""
        v = need_video()
        body = request.get_json(silent=True)
        if not isinstance(body, dict):
            return jsonify(error='Expected a JSON object.'), 400

        def seconds(x):
            return team_shape._number(x) and x >= 0

        def outfield(x):
            return isinstance(x, int) and not isinstance(x, bool) and 7 <= x <= 10
        changes = {}
        if 'direction' in body:
            d = body['direction']
            if not isinstance(d, dict) or 'A' not in d:
                return jsonify(error='direction must be an object {A: left, right or null, switchAt}'), 400
            side = d.get('A')
            if side is not None and (not isinstance(side, str) or side not in ('left', 'right')):
                return jsonify(error='direction.A must be left, right or null'), 400
            switch = d.get('switchAt')
            if switch is not None and not seconds(switch):
                return jsonify(error='direction.switchAt must be seconds or null'), 400
            changes['direction'] = {'A': side, 'switchAt': switch if side else None}
        if 'lineup' in body:
            lu = body['lineup']
            if not isinstance(lu, dict):
                return jsonify(error='lineup must be an object {A, B}'), 400
            for team in ('A', 'B'):
                if team not in lu:
                    continue
                value = lu[team]
                if isinstance(value, dict):
                    start = value.get('from')
                    if not outfield(value.get('outfield')) or (start is not None and not seconds(start)):
                        return jsonify(error='a lineup change is {outfield: 7 to 10, from: seconds or null}'), 400
                    value = {'outfield': value['outfield'], 'from': start} if start is not None else value['outfield']
                elif value is not None and not outfield(value):
                    return jsonify(error='lineup values must be 7 to 10 outfield players, {outfield, from} or null'), 400
                changes.setdefault('lineup', {})[team] = value

        def change(current):
            if 'direction' in changes:
                current['direction'] = changes['direction']
            for team, value in changes.get('lineup', {}).items():
                current['lineup'][team] = value
        return update_shape_settings(v, change)

    @app.post('/api/team-shape/formation-label')
    def team_shape_formation_label():
        """A coach's formation label for a period of one team, kept apart from the tracking data and from the prediction
        (which stays as it was); a label for the same team and period replaces the earlier one."""
        v = need_video()
        body = request.get_json(silent=True)
        if not isinstance(body, dict):
            return jsonify(error='Expected a JSON object.'), 400
        try:
            team, t0, t1, formation = body['team'], float(body['t0']), float(body['t1']), body['formation']
        except (KeyError, TypeError, ValueError, OverflowError):
            return jsonify(error='Expected team, t0, t1 and formation.'), 400
        known = isinstance(team, str) and team in ('A', 'B') and isinstance(formation, str) and formation in team_shape.FORMATION_LABELS
        if not known or not (math.isfinite(t0) and math.isfinite(t1)) or t1 <= t0:
            return jsonify(error='Unknown team or formation, or an empty period.'), 400
        predicted = body.get('predicted') if isinstance(body.get('predicted'), str) else None

        def change(current):
            # A new label replaces every label of the team that would belong to the same period.
            kept = [c for c in current['formations'] if not same_period(c, team, t0, t1)]
            kept.append({'team': team, 't0': t0, 't1': t1, 'formation': formation, 'predicted': predicted,
                         'created': datetime.now().isoformat(timespec='seconds')})
            current['formations'] = kept
        return update_shape_settings(v, change)

    @app.post('/api/team-shape/formation-label/remove')
    def team_shape_formation_label_remove():
        v = need_video()
        body = request.get_json(silent=True)
        if not isinstance(body, dict):
            return jsonify(error='Expected a JSON object.'), 400
        try:
            team, t0, t1 = body['team'], float(body['t0']), float(body['t1'])
        except (KeyError, TypeError, ValueError, OverflowError):
            return jsonify(error='Expected team, t0 and t1.'), 400
        if not isinstance(team, str) or team not in ('A', 'B') or not (math.isfinite(t0) and math.isfinite(t1)):
            return jsonify(error='Unknown team or period.'), 400

        def change(current):
            current['formations'] = [c for c in current['formations'] if not same_period(c, team, t0, t1)]
        return update_shape_settings(v, change)

    @app.get('/api/analytics')
    def analytics_report():
        """Heatmap, average positions and movement trail of one player over a time range, from the saved result."""
        v = need_video()
        player = request.args.get('player', '')
        scope = request.args.get('scope', 'selected')
        if scope not in ('selected', 'A', 'B', 'both'):
            return jsonify(error='scope must be selected, A, B or both'), 400
        try:
            t0 = float(request.args['t0']) if 't0' in request.args else None
            t1 = float(request.args['t1']) if 't1' in request.args else None
            if (t0 is not None and not math.isfinite(t0)) or (t1 is not None and not math.isfinite(t1)):
                raise ValueError
        except (TypeError, ValueError):
            return jsonify(error='t0 and t1 must be seconds'), 400
        samples = samples_of(v)
        a, b = analytics.clamp_range(samples, t0, t1)
        return jsonify(analytics.report(samples, player, a, b, scope))

    # ---------- possession (soccer/possession.py, imported by the first request: see possession_engine) ----------
    # The record is computed from the whole saved result (the frames with their people and ball, the timeline) once per
    # version of the file and kept per run as JSON text, the version being the file's (path, mtime, size) as for the
    # analytics samples: a correction rewrites result.json, so the next request computes again with the corrected
    # roster. One computation at a time: a second request for the same version waits and reads the cache.
    possession_cache = {}
    possession_lock = threading.Lock()
    POSSESSION_RUNS = 4   # records kept (one per run folder); the page asks only for the run it shows

    def possession_video(run):
        """The open video when its shown result is the run the page asks about (the coaching_video rule)."""
        if run is not None and not isinstance(run, str):
            raise Refused('run must be the name of the run the page shows.', 400)
        with gate:
            if state['state'] == 'running':
                raise Refused('Processing is running: possession is available again when it has finished.', 409)
        v = active['video']
        if v is None or v.result_dir is None or not (v.result_dir/'result.json').exists():
            raise Refused('There is no result to judge possession on yet.', 404)
        if run and run != v.result_dir.name:
            raise Refused('The page shows an older result than the latest run: reload the page.', 409)
        return v

    def possession_text(path):
        """The possession record of a result file as JSON text, computed once per version of the file. Refused (503)
        when the possession module cannot be imported, its computation fails, or its record is not plain JSON (a NaN,
        an Infinity or a type JSON has not): the page then says so in one line, and the other views never notice."""
        def stamp_of():
            st = path.stat()
            return (str(path), st.st_mtime_ns, st.st_size)
        key = str(path.parent)
        with possession_lock:
            try:
                stamp = stamp_of()
            except OSError:
                raise Refused('There is no result to judge possession on yet.', 404) from None
            cached = possession_cache.get(key)
            if cached is not None and cached[0] == stamp:
                return cached[1]
            try:
                engine = possession_engine()
            except Exception as error:   # noqa: BLE001 - whatever broke in the possession module stays in its route
                traceback.print_exc()
                raise Refused(f'Possession tracking is not available in this version: {error}', 503) from None
            try:
                data = path.read_bytes()
                record = engine.compute(json.loads(data))
            except Exception as error:   # noqa: BLE001 - a failing computation fails this route only
                traceback.print_exc()
                raise Refused(f'The possession record could not be computed: {error}', 503) from None
            try:
                text = json.dumps(record, separators=(',', ':'), allow_nan=False)
            except (TypeError, ValueError) as error:
                raise Refused(f'The possession record could not be served: {error}', 503) from None
            try:
                same = stamp_of() == stamp and len(data) == stamp[2]
            except OSError:
                same = False
            if same:   # remembered only when the file did not change while it was read
                possession_cache.pop(key, None)
                possession_cache[key] = (stamp, text)
                while len(possession_cache) > POSSESSION_RUNS:
                    possession_cache.pop(next(iter(possession_cache)))
            return text

    @app.get('/api/possession')
    def possession_report():
        """Who had the ball when, and where it changed hands, for the shown run: soccer/possession.py's record of the
        saved result as it is now (keyed by internal ids; the page shows display labels), cached per run until the
        result changes."""
        try:
            v = possession_video(request.args.get('run'))
            text = possession_text(v.result_dir/'result.json')
        except Refused as error:
            return jsonify(error=str(error)), error.status
        return app.response_class(text, mimetype='application/json')

    # ---------- the coaching report (soccer/coaching; storage, fingerprint and feedback in soccer/coaching/store.py) ----------
    # One build at a time, on a worker thread like a tracking run; the report is cached under the fingerprint of what it
    # was computed from, so the page's status requests only hash (once per result file version) and compare.
    coaching = {'build': None, 'errors': {}}
    coaching_lock = threading.Lock()
    result_shas = {}

    def coaching_route(handler):
        """A coaching route: soccer/coaching/store.py is imported first (lazily, see coaching_store), and a coaching
        package that fails to import answers 503 here and nowhere else."""
        @functools.wraps(handler)
        def wrapped(*args, **kwargs):
            try:
                coaching_store()
            except Exception as error:   # noqa: BLE001 - whatever broke in the coaching package stays in its routes
                traceback.print_exc()
                return jsonify(error=f'The coaching report is not available in this version: {error}'), 503
            return handler(*args, **kwargs)
        return wrapped

    def coaching_json(data, status=200):
        """A coaching answer as JSON without a NaN or Infinity token (store.json_text): Flask's jsonify writes them, the
        page's JSON.parse rejects them, and the answers carry values read from the stored report and feedback files."""
        return app.response_class(coaching_store().json_text(data), status=status, mimetype='application/json')

    def result_sha(path):
        """The sha256 of the result file's bytes, remembered per (path, mtime, size): a status request hashes nothing."""
        st = path.stat()
        stamp = (str(path), st.st_mtime_ns, st.st_size)
        with coaching_lock:
            if result_shas.get('stamp') == stamp:
                return result_shas['sha']
        sha = coaching_store().sha256_bytes(path.read_bytes())
        st = path.stat()
        if stamp == (str(path), st.st_mtime_ns, st.st_size):   # remembered only when the file did not change meanwhile
            with coaching_lock:
                result_shas.update(stamp=stamp, sha=sha)
        return sha

    def coaching_video(run):
        """The open video when its shown result is the run the page asks about (the correctable() rule): refused while a
        tracking run is processing (the run folder then has no result yet) and when the page shows an older run."""
        if run is not None and not isinstance(run, str):
            raise Refused('run must be the name of the run the page shows.', 400)
        with gate:
            if state['state'] == 'running':
                raise Refused('Processing is running: coaching reports are available again when it has finished.', 409)
        v = active['video']
        if v is None or v.result_dir is None or not (v.result_dir/'result.json').exists():
            raise Refused('There is no result to report on yet.', 404)
        if run and run != v.result_dir.name:
            raise Refused('The page shows an older result than the latest run: reload the page.', 409)
        return v

    def coaching_options(v, source):
        """The report options of a request (team, t0, t1, context), the range clamped to the analysed segment."""
        frames = samples_entry(v)[2]['frames']
        bounds = (float(frames[0]['time']), float(frames[-1]['time'])) if frames else None
        try:
            return coaching_store().clean_options(source.get('team', 'A'), source.get('t0'), source.get('t1'), source.get('context'), bounds)
        except ValueError as error:
            raise Refused(str(error), 400) from None

    def coaching_now(v, options):
        """(current fingerprint inputs, fingerprint, applied corrections, slim result) of the shown result."""
        cs, slim = coaching_store(), samples_entry(v)[2]
        inputs = cs.fingerprint_inputs(result_sha(v.result_dir/'result.json'), shape_settings(v), options)
        return inputs, cs.fingerprint(inputs), cs.applied_corrections(slim.get('corrections')), slim

    def coaching_answer(v, options):
        cs = coaching_store()
        inputs, current, applied, slim = coaching_now(v, options)
        record, fresh = cs.find_report(v.result_dir, options, current)
        with coaching_lock:
            build = coaching['build']
            error = coaching['errors'].get(current)
        building = build is not None and build['runDir'] == str(v.result_dir) and build['options'] == options
        state_name = 'building' if building else 'ready' if fresh else 'error' if error else 'stale' if record else 'missing'
        reasons = cs.staleness(record['inputs'], inputs, record.get('corrections'), applied) if record and not fresh else []
        # Every verdict on this video, each marked with the finding of this report it belongs to (only verdicts of this
        # run are ever attached). The report carries what its PDF prints: the verdicts on its findings, and apart from
        # them (thisRun false) those of earlier runs; not this run's verdicts on findings it does not have (another team's
        # or another range's report).
        feedback = cs.attach_feedback(record['report'] if record else None, cs.read_feedback(v.directory), v.result_dir.name)
        report = None
        if record:
            report = dict(record['report'], feedback=[f for f in feedback if f['attached'] or not f['thisRun']])
        return {'state': state_name, 'fingerprint': record['fingerprint'] if record else None, 'current': current,
                'reasons': [r['text'] for r in reasons], 'reasonCodes': [r['code'] for r in reasons], 'report': report,
                'feedback': feedback, 'saved': record['saved'] if record else None, 'options': options, 'run': v.result_dir.name,
                'error': error if state_name == 'error' else None, 'busy': build is not None and not building}

    def coaching_fail(error):
        return jsonify(error=str(error)), error.status

    @app.get('/api/coaching')
    @coaching_route
    def coaching_report():
        """The coaching report of one team for the shown run (state missing, building, ready, stale or error), with the
        reasons a stored report is stale and the coach's feedback attached to its findings."""
        try:
            v = coaching_video(request.args.get('run'))
            options = coaching_options(v, request.args)
        except Refused as error:
            return coaching_fail(error)
        return coaching_json(coaching_answer(v, options))

    @app.post('/api/coaching/build')
    @coaching_route
    def coaching_build():
        """Start building the report of one team (one build at a time, on a worker thread)."""
        cs = coaching_store()
        body = request.get_json(silent=True)
        if not isinstance(body, dict):
            return jsonify(error='Expected a JSON object.'), 400
        try:
            v = coaching_video(body.get('run'))
            options = coaching_options(v, body)
        except Refused as error:
            return coaching_fail(error)
        path, run_dir = v.result_dir/'result.json', v.result_dir
        with coaching_lock:
            if coaching['build'] is not None:
                return jsonify(error='A coaching report is being built: wait until it has finished.'), 409
            coaching['build'] = {'runDir': str(run_dir), 'options': options, 'started': time.time()}
        try:
            # The bytes are read once: the fingerprint is their hash, and the finished report is kept only if the file
            # still has them (a correction landing during the build makes it stale, never mislabelled).
            stamp, samples, slim = samples_entry(v)
            data = path.read_bytes()
            sha = cs.sha256_bytes(data)
            settings = shape_settings(v)
            inputs = cs.fingerprint_inputs(sha, settings, options)
            fp = cs.fingerprint(inputs)
            st = path.stat()
            same = stamp == (str(path), st.st_mtime_ns, st.st_size)
            applied = cs.applied_corrections(slim.get('corrections')) if same else None
        except Exception as error:   # noqa: BLE001 - the slot must be released whatever went wrong
            with coaching_lock:
                coaching['build'] = None
            return jsonify(error=f'The coaching report could not be started: {error}'), 500
        del data
        record, fresh = cs.find_report(run_dir, options, fp)
        if fresh and not body.get('force'):
            with coaching_lock:
                coaching['build'] = None
            return jsonify(ok=True, state='ready', fingerprint=fp)
        with coaching_lock:
            coaching['build'].update(fingerprint=fp)
            coaching['errors'].pop(fp, None)
        run_name = run_dir.name

        def work():
            error = None
            try:
                # Options of the report, then what the engine may reuse: the team-shape settings the Analytics tab uses,
                # and the server's parsed samples and observed team shape when they come from these same bytes.
                job = dict(options, settings=settings, run=run_name, fingerprint=fp, source=str(v.source),
                           video={'key': v.key[:16], 'name': v.meta.get('name'), 'fps': v.meta.get('fps')},
                           framesDir=str(cs.frame_dir(v.directory)),
                           cache={'stamp': stamp, 'samples': samples, 'shape': shape_of(v, 'observed')} if same else None)
                report = cs.build_report_for(path, options['team'], job)
                if not isinstance(report, dict):
                    raise ValueError('the engine returned no report')
                if cs.sha256_bytes(path.read_bytes()) != sha or shape_settings(v) != settings:
                    error = 'The result or the team-shape settings changed while the report was built: generate it again.'
                else:
                    try:
                        cs.save_report(run_dir, options['team'], inputs, report, applied)
                    except ValueError:
                        error = 'The report held a number that is not finite, so it was not stored.'
            except cs.EngineUnavailable as failure:
                error = str(failure)
            except Exception as failure:   # noqa: BLE001 - reported to the page
                traceback.print_exc()
                error = f'The coaching report could not be built: {failure}'
            with coaching_lock:
                coaching['build'] = None
                if error:
                    if len(coaching['errors']) >= 32:          # the failures of the latest builds are enough to report
                        coaching['errors'].pop(next(iter(coaching['errors'])))
                    coaching['errors'][fp] = error
        threading.Thread(target=work, name='coaching', daemon=True).start()
        return jsonify(ok=True, state='building', fingerprint=fp)

    @app.get('/api/coaching/frame')
    @coaching_route
    def coaching_frame():
        """The raw pixels of one evidence frame (JPEG, no overlay) of a report stored for the shown run: only frames a
        stored report cites are decoded, which bounds the work a request can cause."""
        cs = coaching_store()
        try:
            v = coaching_video(request.args.get('run'))
            frame, width = int(request.args.get('frame', '')), int(request.args.get('w', 640))
        except Refused as error:
            return coaching_fail(error)
        except (TypeError, ValueError):
            return jsonify(error='frame and w must be whole numbers'), 400
        if frame < 0:
            return jsonify(error='frame must be a source frame index (0 or more)'), 400
        if width not in cs.FRAME_WIDTHS:
            return jsonify(error=f'w must be one of {", ".join(map(str, cs.FRAME_WIDTHS))}'), 400
        times = {}
        for r in cs.records(v.result_dir):
            times.update(cs.evidence_times(r))
        if frame not in times:
            return jsonify(error='That frame is not evidence in a coaching report of this run.'), 404
        try:
            data = cs.frame_jpeg(v.source, v.directory, frame, width, times[frame], v.meta.get('fps'))
        except cs.FrameUnavailable as error:
            return jsonify(error=str(error)), 404
        from io import BytesIO
        return send_file(BytesIO(data), mimetype='image/jpeg')

    @app.get('/api/coaching/pdf')
    @coaching_route
    def coaching_pdf():
        """The stored report of one team as a PDF (with the coach's feedback, display labels of the result as it is now,
        and the staleness reasons when the result changed since the report)."""
        cs = coaching_store()
        try:
            v = coaching_video(request.args.get('run'))
            options = coaching_options(v, request.args)
        except Refused as error:
            return coaching_fail(error)
        answer = coaching_answer(v, options)
        if answer['report'] is None:
            return jsonify(error='Generate the coaching report first.'), 404
        labels = cs.labels_of(samples_entry(v)[2])
        frame = cs.EvidenceFrames(v.source, v.directory, answer['report'], v.meta.get('fps'))
        try:
            data = cs.render_pdf(answer['report'], labels, frame, answer['reasons'] if answer['state'] != 'ready' else None)
        except cs.PdfUnavailable as error:
            return jsonify(error=str(error)), 503
        except Exception as error:   # noqa: BLE001 - reported to the page
            traceback.print_exc()
            return jsonify(error=f'the PDF writer failed ({error})'), 500
        from io import BytesIO
        return send_file(BytesIO(data), mimetype='application/pdf', as_attachment=True,
                         download_name=cs.pdf_filename(answer['report'], v.meta.get('name'), v.result_dir.name, options, v.key[:16]))

    @app.post('/api/coaching/feedback')
    @coaching_route
    def coaching_feedback():
        """The coach's verdict on one finding (correct, partial, incorrect, insufficient) and an optional note, kept per
        video apart from the tracking data; a new verdict on the same finding of the same run replaces the earlier one."""
        cs = coaching_store()
        body = request.get_json(silent=True)
        if not isinstance(body, dict):
            return jsonify(error='Expected a JSON object.'), 400
        try:
            v = coaching_video(body.get('run'))
        except Refused as error:
            return coaching_fail(error)
        observation, verdict, text = body.get('observation'), body.get('verdict'), body.get('text')
        if not cs.is_observation_id(observation):
            return jsonify(error='observation must be the id of a finding (obs- and its hex digits).'), 400
        if verdict not in cs.VERDICTS:
            return jsonify(error='verdict must be correct, partial, incorrect or insufficient.'), 400
        if text is not None and not isinstance(text, str):
            return jsonify(error='text must be the note as a string.'), 400
        stored = cs.records(v.result_dir, body.get('team') if body.get('team') in ('A', 'B') else None)
        stored.sort(key=lambda r: r['fingerprint'] != body.get('fingerprint'))     # the report the page shows first
        record = next((r for r in stored if cs.find_observation(r, observation)), None)
        if record is None:
            return jsonify(error='That finding is not in a stored coaching report of this run.'), 404
        try:
            entry = cs.add_feedback(v.directory, v.result_dir.name, record, observation, verdict, text)
        except ValueError as error:
            return jsonify(error=str(error)), 400
        except cs.FeedbackUnreadable as error:
            return jsonify(error=str(error)), 503
        except OSError:
            return jsonify(error='The feedback could not be saved; try again.'), 503
        return coaching_json({'ok': True, 'feedback': entry})

    @app.post('/api/coaching/feedback/remove')
    @coaching_route
    def coaching_feedback_remove():
        cs = coaching_store()
        body = request.get_json(silent=True)
        if not isinstance(body, dict):
            return jsonify(error='Expected a JSON object.'), 400
        if not cs.is_feedback_id(body.get('id')):
            return jsonify(error='id must be the id of a stored review (f, the date, the time and six hex digits).'), 400
        v = active['video']
        if v is None:
            return jsonify(error='No video is open.'), 404
        try:
            removed = cs.remove_feedback(v.directory, body['id'])
        except cs.FeedbackUnreadable as error:
            return jsonify(error=str(error)), 503
        except OSError:
            return jsonify(error='The feedback could not be saved; try again.'), 503
        if not removed:
            return jsonify(error='That feedback no longer exists.'), 404
        return jsonify(ok=True, removed=body['id'])

    @app.get('/video')
    def video():
        return send_file(need_video().source, conditional=True)

    @app.get('/api/info')
    def info():
        v = active['video']
        base = {'gpu': gpu, 'landmarks': sorted(calibration.LANDMARKS), 'uploads': uploaded_videos()}
        if v is None:
            return jsonify(base | {'video': None})
        saved_ids = v.previous_identities()
        summary = None
        if saved_ids:
            identities = saved_ids.get('registry', {}).get('identities', [])
            summary = {'run': saved_ids.get('run'), 'start': saved_ids.get('start'), 'end': saved_ids.get('end'), 'identities': len(identities)}
        run_name = v.result_dir.name if v.result_dir else None
        items = corrections.load(v.directory)
        return jsonify(base | {'video': v.meta, 'videoKey': v.key, 'settings': v.defaults, 'previousIdentities': summary,
                               'corrections': corrections.listing(v.directory, run_name),
                               'remembered': [{'id': e['id'], 'what': 'not part of the match' if e.get('excluded') else corrections.describe(e['role'], e.get('team'))}
                                              for e in corrections.seeds(items)['identities']]})

    @app.post('/api/upload')
    def upload():
        """Raw request body = the video file. Streamed to uploads/, hashed while writing."""
        name = Path(unquote(request.headers['X-Filename'])).name[:200] or 'video.mp4'
        suffix = Path(name).suffix.lower()
        if suffix not in VIDEO_TYPES:
            return jsonify(error=f"Choose a video file ({', '.join(sorted(VIDEO_TYPES))})."), 400
        if (request.content_length or 0) > MAX_UPLOAD:
            return jsonify(error='This video is too large.'), 413
        with gate:
            if state['state'] == 'running':
                return jsonify(error='Stop the current run before uploading another video.'), 409
        UPLOADS.mkdir(exist_ok=True)
        partial = UPLOADS/f'.partial-{uuid.uuid4().hex}'
        digest = hashlib.sha256()
        try:
            with partial.open('wb') as out:
                while chunk := request.stream.read(4*1024*1024):
                    digest.update(chunk)
                    out.write(chunk)
            target = UPLOADS/f'{digest.hexdigest()[:16]}{suffix}'
            if target.exists():
                partial.unlink()
            else:
                partial.replace(target)
            try:
                video = Video(target, name)
            except ValueError as error:
                target.unlink(missing_ok=True)
                return jsonify(error=str(error)), 400
            index = UPLOADS/'index.json'
            names = json.loads(index.read_text()) if index.exists() else {}
            names[target.name] = name
            index.write_text(json.dumps(names, indent=2), encoding='utf-8')
            switch(video)
            return jsonify(ok=True, video=video.meta)
        except ValueError as error:
            return jsonify(error=str(error)), 409
        finally:
            partial.unlink(missing_ok=True)

    @app.post('/api/open')
    def open_upload():
        """Open a video uploaded earlier."""
        try:
            file = str(request.get_json().get('file', ''))
            entry = next((u for u in uploaded_videos() if u['file'] == file), None)
            if entry is None:
                return jsonify(error='That uploaded video no longer exists.'), 404
            switch(Video(UPLOADS/entry['file'], entry['name']))
            return jsonify(ok=True)
        except (ValueError, AttributeError) as error:
            return jsonify(error=str(error)), 400

    @app.get('/api/status')
    def status():
        with gate:
            v = active['video']
            has = bool(v and v.result_dir and (v.result_dir/'result.json').exists())
            return jsonify(dict(state) | {'hasResult': has, 'videoKey': v.key if v else None,
                                          'resultRun': v.result_dir.name if has else None})

    @app.get('/api/result')
    def result():
        path = need_video().result_dir
        if path is None or not (path/'result.json').exists():
            abort(404)
        return send_file(path/'result.json', as_attachment=request.args.get('download') == '1', download_name='pitchiq-gpu-result.json')

    @app.get('/api/preview')
    def live_preview():
        path = need_video().result_dir
        if path is None or not (path/'preview.jpg').exists():
            abort(404)
        # Read into memory so Windows does not hold a file lock during replacement.
        from io import BytesIO
        return send_file(BytesIO((path/'preview.jpg').read_bytes()), mimetype='image/jpeg')

    def begin(config):
        v = active['video']
        if v is None:
            raise ValueError('Upload or open a video first.')
        previous = None
        if config['continueIdentities']:
            previous = v.previous_identities()
            if previous is None:
                raise ValueError('There is no earlier completed run of this video to continue identities from.')
        # People corrected in the review page are remembered by appearance in every later run.
        previous = corrections.seed_state(previous, corrections.load(v.directory))
        with gate:
            if state['state'] == 'running':
                raise ValueError('A run is already processing.')
            cancel.clear()
            run_name = datetime.now().strftime('%Y%m%d-%H%M%S')+'-'+uuid.uuid4().hex[:6]
            destination = v.directory/run_name
            v.result_dir = destination
            state.clear()
            state.update(state='running', stage='Starting', progress=0, gpu=gpu, run=run_name)
            v.defaults.update(config)
            v.saved.write_text(json.dumps(config, indent=2), encoding='utf-8')

        def progress(values):
            with gate:
                state.update(values)

        def worker():
            try:
                report = run(v.source, destination, config, progress, cancel.is_set, previous)
                report['sourceSHA256'] = v.key
                (destination/'result.json').write_text(json.dumps(report, separators=(',', ':'), allow_nan=False), encoding='utf-8')
                v.last.write_text(json.dumps({'run': run_name}), encoding='utf-8')
                progress({'state': report['status']})
            except Exception as error:
                traceback.print_exc()
                progress({'state': 'error', 'stage': 'Processing failed', 'error': str(error)})
        threading.Thread(target=worker, daemon=True).start()

    @app.post('/api/run')
    def start():
        try:
            v = active['video']
            if v is None:
                raise ValueError('Upload or open a video first.')
            begin(validate(request.get_json(), v.meta))
            return jsonify(ok=True)
        except (ValueError, KeyError, TypeError) as error:
            return jsonify(error=str(error)), 400

    edit = threading.Lock()

    def correctable(payload):
        with gate:
            if state['state'] == 'running':
                raise ValueError('Wait until processing has finished, then correct people in the result.')
        v = need_video()
        if v.result_dir is None or not (v.result_dir/'result.json').exists():
            raise ValueError('There is no result to correct yet.')
        if payload.get('run') and payload['run'] != v.result_dir.name:
            raise ValueError('The page shows an older result than the latest run: reload the page.')
        return v

    @app.post('/api/correction')
    def correct():
        """Change who a person in the shown result is; remembered for later runs of this video."""
        try:
            payload = request.get_json()
            if not isinstance(payload, dict):
                raise ValueError('Expected a correction.')
            v = correctable(payload)
            with edit:
                entry = corrections.add(v.directory, v.result_dir, payload)
            print(f"Correction: {entry['from']} -> {entry['to']} ({entry['what']})", flush=True)
            return jsonify(ok=True, correction=entry)
        except (ValueError, KeyError, TypeError, OSError) as error:
            return jsonify(error=str(error)), 400

    @app.post('/api/correction/remove')
    def uncorrect():
        try:
            payload = request.get_json()
            if not isinstance(payload, dict) or not isinstance(payload.get('id'), str):
                raise ValueError('Which correction?')
            v = correctable({})
            with edit:
                gone = corrections.drop(v.directory, v.result_dir, payload['id'])
            return jsonify(ok=True, removed=sorted(gone))
        except (ValueError, KeyError, TypeError, OSError) as error:
            return jsonify(error=str(error)), 400

    @app.post('/api/stop')
    def stop():
        cancel.set()
        return jsonify(ok=True)

    print(f"PitchIQ GPU prototype | {gpu['name']} | CUDA {gpu['cuda']}", flush=True)
    print(f"Review: http://127.0.0.1:{args.port}", flush=True)
    if active['video']:
        print(f"Video: {active['video'].source}\nResults: {active['video'].directory}", flush=True)
    else:
        print('No video given: upload one in the page.', flush=True)
    print('Ctrl+C stops the server.', flush=True)
    if args.run:
        try:
            begin(active['video'].defaults if active['video'] else {})
        except (ValueError, KeyError) as error:
            print(f'Could not start: {error}', flush=True)
    serve(app, host='127.0.0.1', port=args.port, threads=6, max_request_body_size=MAX_UPLOAD)


if __name__ == '__main__':
    main()
