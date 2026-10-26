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
        package that fails to import answers 503 here and nowhere else. A request another site's page makes is refused
        (403) when the browser says so (Sec-Fetch-Site cross-site or same-site): GET /api/coaching builds and stores
        roles, reports and progress entries from the stored measurements, and a plain cross-site <img> GET sends no
        Origin, so the Host check alone let one through (CODE L7). The page's own requests are same-origin, and tools
        such as curl send no such header."""
        @functools.wraps(handler)
        def wrapped(*args, **kwargs):
            if request.headers.get('Sec-Fetch-Site') in ('cross-site', 'same-site'):
                return jsonify(error='Coaching requests are answered for this page only.'), 403
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

    def coaching_body():
        """The JSON object of a coaching POST, or None. A body nested deeper than the parser's recursion limit raised
        RecursionError through get_json(silent=True) and answered 500 (CODE L6, a checkout behaviour every POST route
        shares); the coaching routes answer it as any other unreadable body, 400."""
        try:
            return request.get_json(silent=True)
        except RecursionError:
            return None

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

    # ---------- coaching v2 (adaptive coaching, DESIGN v2 1.2 and 12.1): three cache tiers, settings, coach-marked
    # possession, squads and progress. A request without a v1 'context' level is a v2 request; the v1 report stays
    # reachable with one (the page's legacy mode and the v1 tools). ----------
    runs_root = ROOT/'runs'           # squads and progress live in runs/_coaching/ (soccer/coaching/progress.py)

    def coaching_v2(handler):
        """A v2 coaching route: a v2 module that is missing or does not import (profiles, progress, a stage) answers 503
        on the v2 routes only; the v1 report and every other view keep working. A coaching table that does not
        validate (profiles.json, sources.json, drills.json, progress_rules.json: store.table_failure) answers a JSON 503
        with its problems: it answered Flask's HTML 500, which the page took for a server without the v2 routes, and
        the coach got the v1 professional-reference report without a word (CODE M-1, round 2)."""
        @functools.wraps(handler)
        def wrapped(*args, **kwargs):
            try:
                return handler(*args, **kwargs)
            except Refused as error:
                return coaching_fail(error)
            except (ImportError, SyntaxError, NotImplementedError) as error:   # NotImplementedError: StageUnavailable
                traceback.print_exc()
                return jsonify(error=f'The adaptive coaching report is not available in this version: {error}'), 503
            except ValueError as error:
                broken = coaching_store().table_failure(error)
                if broken is None:
                    raise
                traceback.print_exc()
                shown = '; '.join(broken['problems'][:3]) or 'no detail'
                return coaching_json({'error': f'The adaptive coaching report is not available: {broken["table"]} does '
                                               f'not validate ({shown}). Fix or restore the file, then reload the page.',
                                      'table': broken['table'], 'problems': broken['problems']}, 503)
        return wrapped

    def coaching_progress():
        """soccer/coaching/progress.py, imported when a v2 route first needs it (see coaching_v2)."""
        return importlib.import_module('soccer.coaching.progress')

    def v2_now(v):
        """What a v2 answer is keyed on, read once per request: the analysed segment and its bounds, the result file's
        sha (hashed once per file version), the team-shape settings, the video's coaching settings, this run's coach
        spans and the corrections the result lists."""
        cs = coaching_store()
        slim = samples_entry(v)[2]
        frames = slim['frames']
        return {'bounds': (float(frames[0]['time']), float(frames[-1]['time'])) if frames else None,
                'segment': cs.segment_of(frames), 'sha': result_sha(v.result_dir/'result.json'),
                'shape': shape_settings(v), 'settings': cs.read_settings(v.directory),
                'spans': cs.read_spans(v.directory, v.result_dir.name)[0],
                'corrections': cs.applied_corrections(slim.get('corrections'))}

    def v2_options(now, source):
        """The v2 report options {team, t0, t1} of a request (the range clamped to the analysed segment)."""
        try:
            return coaching_store().clean_range(source.get('team', 'A'), source.get('t0'), source.get('t1'), now['bounds'])
        except ValueError as error:
            raise Refused(str(error), 400) from None

    def v2_keys(now, options):
        return coaching_store().tier_keys(now['sha'], now['shape'], now['settings'], options['team'], now['spans'],
                                          now['segment'], options['t0'], options['t1'])

    def v2_meta(v):
        return {'run': v.result_dir.name, 'video': {'key': v.key[:16], 'name': v.meta.get('name')}}

    def v2_after(v, now, keys):
        """What runs once a new v2 report is stored: progress (progress.record for the squad confirmed as this team in
        this run, the entries of an older measurement of the run marked stale). Progress never fails a report."""
        cs = coaching_store()

        def after(record, obs, roles):
            try:
                cs.mark_progress_stale(runs_root, now['settings'], v.result_dir.name, v.key[:16], keys['team'], keys['mfp'])
                cs.record_progress(runs_root, now['settings'], v.result_dir.name, v.key[:16], keys['team'], obs, roles,
                                   keys['range'], video_path=v.source, coach_date=cs.read_match_date(v.directory))
            except Exception:   # noqa: BLE001 - logged; the report is stored either way
                traceback.print_exc()
        return after

    def remember_error(key, error):
        with coaching_lock:
            if len(coaching['errors']) >= 32:          # the failures of the latest builds are enough to report
                coaching['errors'].pop(next(iter(coaching['errors'])))
            coaching['errors'][key] = error

    def v2_answer(v, source):
        """The v2 answer of GET /api/coaching (FRONTEND's route table): {engine 2, state missing|building|ready|stale|
        error, tiers {measurement, role, interpretation}, report, feedback, reasons, reasonCodes, busy, error, options,
        saved, run, drills, fingerprints, coachSpans, inferred (the team's measured roles when its roles file is stored,
        else None)}. When the obs file of the current settings is stored, a missing
        roles file and report are made here at once (a level, age, style or role change: tens of milliseconds to a
        second, DESIGN 1.2); a missing obs file needs POST /api/coaching/build (a format, field size, coach-span,
        team-shape or tracking change)."""
        cs = coaching_store()
        now = v2_now(v)
        options = v2_options(now, source)
        keys = v2_keys(now, options)
        run_dir, run = v.result_dir, v.result_dir.name
        record, fresh = cs.find_report_v2(run_dir, keys, options)
        tiers = cs.tier_states(run_dir, keys)
        with coaching_lock:
            build = coaching['build']
            error = coaching['errors'].get(keys['ifp']) or coaching['errors'].get(keys['mfp'])
        building = bool(build and build.get('engine') == 2 and build['runDir'] == str(run_dir)
                        and (build.get('mfps') or {}).get(keys['team']) == keys['mfp'])
        if building:
            tiers['measurement'] = 'building'
        if not fresh and not building and not error and tiers['measurement'] == 'ready':
            try:
                made = cs.ensure_report(run_dir, keys, options, now['settings'], meta=v2_meta(v),
                                        corrections=now['corrections'], after=v2_after(v, now, keys))
            except (ImportError, SyntaxError, NotImplementedError):
                raise
            except Exception as failure:   # noqa: BLE001 - reported to the page, remembered until the next build
                traceback.print_exc()
                error = f'The coaching report could not be built: {failure}'
                remember_error(keys['ifp'], error)
            else:
                if made is not None:
                    record, fresh = made, True
                    tiers = cs.tier_states(run_dir, keys)
        state_name = 'building' if building else 'ready' if fresh else 'error' if error else 'stale' if record else 'missing'
        reasons = cs.staleness_v2(record['inputs'], keys['inputs'], record.get('corrections'), now['corrections']) \
            if record and not fresh else []
        progression = team_name = None
        if record:
            try:
                if fresh:
                    cs.mark_progress_stale(runs_root, now['settings'], run, v.key[:16], keys['team'], keys['mfp'])
                    cs.record_progress_if_missing(runs_root, now['settings'], run, v.key[:16], run_dir, keys,
                                                  record['report'], video_path=v.source,
                                                  coach_date=cs.read_match_date(v.directory))
                progression, team_name = cs.progression_for(runs_root, now['settings'], run, keys['team'])
            except (ImportError, SyntaxError):
                raise
            except Exception:   # noqa: BLE001 - a progress store that cannot be read leaves the section out
                traceback.print_exc()
                progression = team_name = None
        report = cs.served_report(record, progression=progression, team_name=team_name) if record else None
        # The measured roles of this team once its roles file is stored: the roles editor asked GET
        # /api/coaching/settings once per run, before anything was measured, and showed "no stable role" for every player after the first
        # Generate (CODE M3); the answer now carries them (one roles file read, 1-49 KB on the run copies).
        # A player the coach confirmed keeps his measured reading (read from the obs file only then; CODE L-2).
        inferred = None
        if tiers['role'] == 'ready':
            inferred = cs.measured_roles(cs.load_tier(cs.tier_path(run_dir, 'roles', keys)),
                                         obs=lambda: cs.load_tier(cs.tier_path(run_dir, 'obs', keys)))
        return {'engine': 2, 'state': state_name, 'tiers': tiers, 'report': report, 'inferred': inferred,
                'feedback': cs.feedback_answer(report, cs.read_feedback_doc(v.directory), run),
                'reasons': [r['text'] for r in reasons], 'reasonCodes': [r['code'] for r in reasons],
                'busy': build is not None and not building, 'error': error if state_name == 'error' else None,
                'options': options, 'saved': record['saved'] if record else None, 'run': run,
                'drills': cs.drills_of(report) if report else {},
                'fingerprints': {'measurement': keys['mfp'], 'roles': keys['rfp'], 'interpretation': keys['ifp']},
                'coachSpans': now['spans'] is not None}

    @app.get('/api/coaching')
    @coaching_route
    def coaching_report():
        """The coaching report of one team for the shown run (state missing, building, ready, stale or error), with the
        reasons a stored report is stale and the coach's feedback attached to its findings: v2 (adaptive coaching), or
        the v1 report when the request names a v1 context level."""
        if 'context' not in request.args:
            return coaching_report_v2()
        try:
            v = coaching_video(request.args.get('run'))
            options = coaching_options(v, request.args)
        except Refused as error:
            return coaching_fail(error)
        return coaching_json(coaching_answer(v, options))

    @coaching_v2
    def coaching_report_v2():
        v = coaching_video(request.args.get('run'))
        return coaching_json(v2_answer(v, request.args))

    @app.get('/api/coaching/profiles')
    @coaching_route
    @coaching_v2
    def coaching_profiles():
        """The coaching profile catalogue (levels, ages, formats and their formations and roles, format defaults, styles,
        unavailable categories, sources): profiles.catalog(), static per version of profiles.json and sources.json."""
        return app.response_class(coaching_store().profiles_catalog_text(), mimetype='application/json')

    @app.get('/api/coaching/settings')
    @coaching_route
    @coaching_v2
    def coaching_settings_get():
        """The video's coaching settings (cleaned), the measured role of each player of the current measurement (from the
        newest roles file of each team, coach confirmations left out: what the roles editor shows as "Measured"), the
        squads, this run's squad links and the coach's match date."""
        cs, PG = coaching_store(), coaching_progress()
        v = coaching_video(request.args.get('run'))
        now = v2_now(v)
        inferred = {}
        for team in ('A', 'B'):
            keys = v2_keys(now, {'team': team, 't0': None, 't1': None})
            newest = None
            for p in sorted(cs.tier_dir(v.result_dir).glob(f'roles-{team}-{keys["mfp"][:16]}-*.json'),
                            key=lambda q: q.stat().st_mtime_ns, reverse=True):
                newest = cs.load_tier(p)
                if newest is not None:
                    break
            obs_path = cs.tier_path(v.result_dir, 'obs', keys)
            inferred[team] = cs.measured_roles(newest, obs=lambda path=obs_path: cs.load_tier(path))
        return coaching_json({'run': v.result_dir.name, 'settings': now['settings'], 'inferred': inferred,
                              'squads': PG.read_squads(runs_root), 'links': PG.links(now['settings'], v.result_dir.name),
                              'matchDate': cs.read_match_date(v.directory)})

    @app.post('/api/coaching/settings')
    @coaching_route
    @coaching_v2
    def coaching_settings_post():
        """Save one team's coaching settings block {run, team, settings} (validated: anything the cleaning would drop is
        refused with the problems listed); the answer says which cache tiers became stale. A squad confirmed as this
        team of this run is kept in step with the block's age group, level and format (its progress is read with them)."""
        cs, PG = coaching_store(), coaching_progress()
        body = coaching_body()
        if not isinstance(body, dict):
            return jsonify(error='Expected a JSON object.'), 400
        v = coaching_video(body.get('run'))
        team, raw = body.get('team'), body.get('settings')
        if team not in ('A', 'B'):
            return jsonify(error='team must be A or B'), 400
        if not isinstance(raw, dict):
            return jsonify(error='settings must be the team settings object'), 400
        try:
            saved = cs.save_team_settings(v.directory, team, raw,
                                          squad_exists=lambda sq: PG.squad(runs_root, sq) is not None)
        except cs.SettingsRefused as error:
            return coaching_json({'error': f'The settings were not saved: {error}', 'problems': error.problems}, 400)
        except cs.SettingsUnreadable as error:
            return jsonify(error=str(error)), 503
        except OSError:
            return jsonify(error='The coaching settings could not be saved; try again.'), 503
        block, squad, note = saved['new'], None, None
        run = v.result_dir.name
        if block.get('squad') and any(link['run'] == run and link['team'] == team and link['confirmed']
                                      for link in block['runLinks']):
            try:
                squad = PG.update_squad(runs_root, block['squad']['id'], age_group=block['ageGroup'],
                                        level=block['playingLevel'], fmt=block['format'])
            except PG.ProgressError as error:
                note = f'The squad was not updated: {error}'
        return coaching_json({'ok': True, 'settings': saved['settings'], 'stale': saved['stale'], 'squad': squad,
                              'note': note})

    def spans_answer(v, now, stored=None, saved=None):
        cs = coaching_store()
        return {'run': v.result_dir.name, 'spans': stored if stored is not None else cs.empty_spans(),
                'segment': now['segment'], 'saved': saved}

    @app.get('/api/coaching/spans')
    @coaching_route
    @coaching_v2
    def coaching_spans_get():
        """This run's coach-marked possession spans (who had the ball, or a dead ball, from t0 to t1; absolute source
        seconds), stored per video apart from the tracking, and the analysed segment they must lie in."""
        cs = coaching_store()
        v = coaching_video(request.args.get('run'))
        now = v2_now(v)
        stored, saved = cs.read_spans(v.directory, v.result_dir.name)
        return coaching_json(spans_answer(v, now, stored, saved))

    def spans_save(v, raw):
        cs = coaching_store()
        now = v2_now(v)
        try:
            out = cs.save_spans(v.directory, v.result_dir.name, raw, now['segment'])
        except cs.SettingsRefused as error:
            return coaching_json({'error': f'The marked possession was not saved: {error}', 'problems': error.problems}, 400)
        except cs.SettingsUnreadable as error:
            return jsonify(error=str(error)), 503
        except OSError:
            return jsonify(error='The marked possession could not be saved; try again.'), 503
        return coaching_json(dict(spans_answer(v, now, out['spans'], out['saved']), ok=True, dropped=out['dropped'],
                                  stale=['measurement', 'role', 'interpretation'] if out['changed'] else []))

    @app.post('/api/coaching/spans')
    @coaching_route
    @coaching_v2
    def coaching_spans_post():
        """Replace this run's coach-marked spans {run, spans: schema 'coach_spans' record}. They override possession-4
        for the phase wherever they lie, so every cache tier of both teams goes stale (the answer says so)."""
        body = coaching_body()
        if not isinstance(body, dict):
            return jsonify(error='Expected a JSON object.'), 400
        return spans_save(coaching_video(body.get('run')), body.get('spans'))

    @app.post('/api/coaching/spans/remove')
    @coaching_route
    @coaching_v2
    def coaching_spans_remove():
        """Remove all of this run's coach-marked spans {run} (the engine's possession estimate decides the phase again)."""
        body = coaching_body()
        if not isinstance(body, dict):
            return jsonify(error='Expected a JSON object.'), 400
        return spans_save(coaching_video(body.get('run')), coaching_store().empty_spans())

    @app.get('/api/coaching/squads')
    @coaching_route
    @coaching_v2
    def coaching_squads_get():
        """The squads (teams followed across matches; runs/_coaching/squads.json)."""
        return coaching_json({'squads': coaching_progress().read_squads(runs_root)})

    @app.post('/api/coaching/squads')
    @coaching_route
    @coaching_v2
    def coaching_squads_post():
        """Create a squad {name, ageGroup, level, format}; a squad with the same name (case-insensitive) is returned
        instead (created false)."""
        PG = coaching_progress()
        body = coaching_body()
        if not isinstance(body, dict):
            return jsonify(error='Expected a JSON object.'), 400
        try:
            made, created = PG.create_squad(runs_root, body.get('name'), body.get('ageGroup'), body.get('level'),
                                            body.get('format'))
        except PG.ProgressUnreadable as error:
            return jsonify(error=f'The squads file could not be read; nothing was changed ({error}).'), 503
        except PG.ProgressError as error:
            return jsonify(error=f'The squad was not created: {error}'), 400
        except OSError:
            return jsonify(error='The squad could not be saved; try again.'), 503
        return coaching_json({'squad': made, 'created': created})

    @app.get('/api/coaching/progress')
    @coaching_route
    @coaching_v2
    def coaching_progress_get():
        """A squad's development progression (progress.mastery: per skill its state, judged opportunities, matches,
        trend, rates and history), read with the viewed run's team settings when run and team name the squad's team
        (else the squad's own age group, level and format). Entries the current rules cannot read are first re-derived
        from their cached observation files when those are still stored."""
        cs, PG = coaching_store(), coaching_progress()
        squad_id = request.args.get('squad', '')
        if not cs.SQUAD_ID.fullmatch(squad_id):
            return jsonify(error='squad must be a squad id (sq- and 12 hex digits).'), 400
        try:
            rec = PG.squad(runs_root, squad_id)
            if rec is None:
                return jsonify(error='There is no such squad.'), 404
            block = None
            run, team = request.args.get('run'), request.args.get('team')
            v = active['video']
            if run and team in ('A', 'B') and v is not None and v.result_dir is not None and v.result_dir.name == run:
                link, b = cs.progress_link(cs.read_settings(v.directory), run, team)
                if link is not None and link['squad']['id'] == squad_id:
                    block = b
            rederived = cs.rederive_progress(runs_root, squad_id)
            entries = PG.read_entries(runs_root, squad_id)
            detail = PG.mastery(entries, PG.read_block(rec, block), squad_id=squad_id)
        except PG.ProgressUnreadable as error:
            return jsonify(error=f'The progress store could not be read ({error}).'), 503
        return coaching_json({'squad': rec, 'progress': detail, 'entries': len(entries), 'rederived': rederived})

    @app.post('/api/coaching/match-date')
    @coaching_route
    @coaching_v2
    def coaching_match_date():
        """The coach's match date of this video {run, date: YYYY-MM-DD or null}: progress entries use it instead of the
        video file's date (the entries already stored for the video's squads are updated)."""
        cs, PG = coaching_store(), coaching_progress()
        body = coaching_body()
        if not isinstance(body, dict):
            return jsonify(error='Expected a JSON object.'), 400
        v = coaching_video(body.get('run'))
        try:
            day = cs.save_match_date(v.directory, body.get('date'))
        except cs.SettingsRefused as error:
            return jsonify(error=str(error)), 400
        updated = 0
        if day is not None:
            try:
                for link in PG.links(cs.read_settings(v.directory), v.result_dir.name):
                    updated += PG.set_match_date(runs_root, squad_id=link['squad']['id'], video=v.key[:16], day=day)
            except PG.ProgressError as error:
                return jsonify(error=f'The date was saved; the stored progress could not be updated ({error}).'), 503
        return coaching_json({'ok': True, 'date': day, 'updated': updated})

    @coaching_v2
    def coaching_build_v2(body):
        """Start building the v2 report of one team (POST /api/coaching/build without a context level): the
        measurement tier of both teams when an obs file is missing (one series and possession for the two; the
        server's parsed samples, observed team shape and possession record are reused when they come from the same
        bytes), then the roles file and the report, on the worker thread (one build at a time, the v1 slot)."""
        cs = coaching_store()
        v = coaching_video(body.get('run'))
        now = v2_now(v)
        options = v2_options(now, body)
        path, run_dir = v.result_dir/'result.json', v.result_dir
        force = bool(body.get('force'))
        with coaching_lock:
            if coaching['build'] is not None:
                return jsonify(error='A coaching report is being built: wait until it has finished.'), 409
            coaching['build'] = {'engine': 2, 'runDir': str(run_dir), 'options': options, 'started': time.time(), 'mfps': {}}
        try:
            # The bytes are read once: every key is computed from their hash, and the tiers are stored only if the file
            # still has them when the build ends (a correction landing meanwhile is a new measurement, never mislabelled).
            stamp, samples, slim = samples_entry(v)
            data = path.read_bytes()
            now['sha'] = cs.sha256_bytes(data)
            keys = v2_keys(now, options)
            both = {team: v2_keys(now, {'team': team, 't0': None, 't1': None}) for team in ('A', 'B')}
            st = path.stat()
            same = stamp == (str(path), st.st_mtime_ns, st.st_size)
        except Exception as error:   # noqa: BLE001 - the slot must be released whatever went wrong
            with coaching_lock:
                coaching['build'] = None
            return jsonify(error=f'The coaching report could not be started: {error}'), 500
        record, fresh = cs.find_report_v2(run_dir, keys, options)
        answer = {'fingerprints': {'measurement': keys['mfp'], 'roles': keys['rfp'], 'interpretation': keys['ifp']}}
        if fresh and not force:
            with coaching_lock:
                coaching['build'] = None
            return coaching_json(dict(answer, ok=True, state='ready'))
        # The teams whose obs file is missing are measured (both in one pass when both are); only they are 'building'.
        missing = [team for team in ('A', 'B') if force or not cs.tier_path(run_dir, 'obs', both[team]).exists()]
        with coaching_lock:
            coaching['build']['mfps'] = {team: both[team]['mfp'] for team in missing}
            for k in (keys['ifp'], keys['mfp']):
                coaching['errors'].pop(k, None)

        def work():
            error = None
            try:
                if missing:
                    possession = None
                    if same:
                        try:
                            possession = json.loads(possession_text(path))
                        except Refused:
                            possession = None
                    made = cs.measure(json.loads(data), result_sha=now['sha'], shape_settings=now['shape'],
                                      settings=now['settings'], spans=now['spans'],
                                      team=None if len(missing) == 2 else missing[0],
                                      samples=samples if same else None, shape=shape_of(v, 'observed') if same else None,
                                      possession=possession)
                    made = made if len(missing) == 2 else {missing[0]: made}
                    if cs.sha256_bytes(path.read_bytes()) != now['sha'] or shape_settings(v) != now['shape']:
                        error = 'The result or the team-shape settings changed while the report was built: generate it again.'
                    else:
                        for team in missing:
                            cs.save_tier(run_dir, 'obs', both[team], made[team])
                if not error:
                    if force:
                        for kind in ('report', 'roles'):
                            cs.tier_path(run_dir, kind, keys).unlink(missing_ok=True)
                    record = cs.ensure_report(run_dir, keys, options, now['settings'], meta=v2_meta(v),
                                              corrections=now['corrections'] if same else None,
                                              after=v2_after(v, now, keys))
                    if record is None:
                        error = 'The measured situations could not be stored; generate the report again.'
            except Exception as failure:   # noqa: BLE001 - reported to the page
                traceback.print_exc()
                error = f'The coaching report could not be built: {failure}'
            with coaching_lock:
                coaching['build'] = None
            if error:
                remember_error(keys['mfp'], error)
        threading.Thread(target=work, name='coaching', daemon=True).start()
        return coaching_json(dict(answer, ok=True, state='building'))

    @app.post('/api/coaching/build')
    @coaching_route
    def coaching_build():
        """Start building the report of one team (one build at a time, on a worker thread): v2, or the v1 report when
        the request names a v1 context level."""
        cs = coaching_store()
        body = coaching_body()
        if not isinstance(body, dict):
            return jsonify(error='Expected a JSON object.'), 400
        if 'context' not in body:
            return coaching_build_v2(body)
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
        for r in cs.records(v.result_dir)+cs.records_v2(v.result_dir):     # v1 evidence and v2 key moments
            times.update(cs.evidence_times(r))
        if frame not in times:
            return jsonify(error='That frame is not evidence in a coaching report of this run.'), 404
        try:
            data = cs.frame_jpeg(v.source, v.directory, frame, width, times[frame], v.meta.get('fps'))
        except cs.FrameUnavailable as error:
            return jsonify(error=str(error)), 404
        from io import BytesIO
        return send_file(BytesIO(data), mimetype='image/jpeg')

    @coaching_v2
    def coaching_pdf_v2():
        """The v2 report of one team as a PDF: the report as GET /api/coaching serves it (made from the stored tiers
        when needed), the coach's feedback it carries, the display labels of the result as it is now, the staleness
        reasons, the screenshots of its key moments and the viewer's team colours (colourA, colourB: #rrggbb)."""
        cs = coaching_store()
        v = coaching_video(request.args.get('run'))
        answer = v2_answer(v, request.args)
        if answer['report'] is None:
            return jsonify(error='Generate the coaching report first.'), 404
        colours = {team: request.args.get(f'colour{team}') for team in ('A', 'B')}
        colours = {team: c for team, c in colours.items() if isinstance(c, str) and cs.HEX_COLOUR.fullmatch(c)}
        labels = cs.labels_of(samples_entry(v)[2])
        frame = cs.EvidenceFrames(v.source, v.directory, answer['report'], v.meta.get('fps'))
        feedback = [f for f in answer['feedback'] if f['attached'] or not f['thisRun']]
        progression, name = answer['report'].get('progression'), answer['report'].get('teamName')
        squad = {'id': progression['squad'], 'name': name} if progression and progression.get('squad') and name else None
        # Whose possession this run's coach spans name: the PDF words the possession row as the page does (CODE M-2).
        marked = cs.marked_teams(cs.read_spans(v.directory, v.result_dir.name)[0])
        try:
            data = cs.render_pdf(answer['report'], labels, frame, answer['reasons'] if answer['state'] != 'ready' else None,
                                 feedback=feedback, colours=colours or None, squad=squad,
                                 coach_spans=answer['coachSpans'], marked=marked)
        except cs.PdfUnavailable as error:
            return jsonify(error=str(error)), 503
        except Exception as error:   # noqa: BLE001 - reported to the page
            traceback.print_exc()
            return jsonify(error=f'the PDF writer failed ({error})'), 500
        from io import BytesIO
        return send_file(BytesIO(data), mimetype='application/pdf', as_attachment=True,
                         download_name=cs.pdf_filename_v2(answer['report'], v.meta.get('name'), v.result_dir.name,
                                                          v.key[:16]))

    @app.get('/api/coaching/pdf')
    @coaching_route
    def coaching_pdf():
        """The stored report of one team as a PDF (with the coach's feedback, display labels of the result as it is now,
        and the staleness reasons when the result changed since the report): v2, or the v1 report when the request
        names a v1 context level."""
        cs = coaching_store()
        if 'context' not in request.args:
            return coaching_pdf_v2()
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

    @coaching_v2
    def coaching_feedback_v2(body):
        """A v2 verdict {run, team, findingKey, verdict, text, range {t0, t1}} or an adjusted suggested area {run, team,
        findingKey, range, adjustment {evidence, polygon, mover, note}} on a finding a stored v2 report of this run shows
        for that range (DESIGN 10: by finding key and range, never by id alone). Kept per video apart from the tracking
        and the measurements; in no fingerprint."""
        cs = coaching_store()
        v = coaching_video(body.get('run'))
        team, key, rng = body.get('team'), body.get('findingKey'), body.get('range')
        verdict, text, adjustment = body.get('verdict'), body.get('text'), body.get('adjustment')
        if team not in ('A', 'B'):
            return jsonify(error='team must be A or B.'), 400
        if not (isinstance(key, str) and cs.FINDING_KEY.fullmatch(key)):
            return jsonify(error='findingKey must be the key of a finding (D1|A|...).'), 400
        if not (isinstance(rng, dict) and all(isinstance(rng.get(k), (int, float)) and not isinstance(rng.get(k), bool)
                                              and math.isfinite(rng[k]) for k in ('t0', 't1'))):
            return jsonify(error='range must be the report range {t0, t1} in seconds.'), 400
        if (adjustment is None) == (verdict is None):
            return jsonify(error='Send either a verdict or an adjusted area.'), 400
        if verdict is not None and verdict not in cs.VERDICTS:
            return jsonify(error='verdict must be correct, partial, incorrect or insufficient.'), 400
        if text is not None and not isinstance(text, str):
            return jsonify(error='text must be the note as a string.'), 400
        record = cs.find_feedback_record(v.result_dir, team, key, rng)
        if record is None:
            return jsonify(error='That finding is not in a stored coaching report of this run and range.'), 404
        try:
            if adjustment is not None:
                stored = cs.add_adjustment(v.directory, v.result_dir.name, record, key, adjustment)
                doc = {'feedback': [], 'adjustments': [stored]}
            else:
                stored = cs.add_verdict_v2(v.directory, v.result_dir.name, record, key, verdict, text)
                doc = {'feedback': [stored], 'adjustments': []}
        except ValueError as error:
            return jsonify(error=str(error)), 400
        except cs.FeedbackUnreadable as error:
            return jsonify(error=str(error)), 503
        except OSError:
            return jsonify(error='The feedback could not be saved; try again.'), 503
        entry = cs.feedback_answer(record['report'], doc, v.result_dir.name)[0]
        return coaching_json({'ok': True, 'feedback': entry})

    @app.post('/api/coaching/feedback')
    @coaching_route
    def coaching_feedback():
        """The coach's verdict on one finding (correct, partial, incorrect, insufficient) and an optional note, kept per
        video apart from the tracking data; a new verdict on the same finding of the same run replaces the earlier one.
        A body with a findingKey is v2 (verdict or adjusted area); one with an observation id is v1."""
        cs = coaching_store()
        body = coaching_body()
        if not isinstance(body, dict):
            return jsonify(error='Expected a JSON object.'), 400
        if 'findingKey' in body:
            return coaching_feedback_v2(body)
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
        body = coaching_body()
        if not isinstance(body, dict):
            return jsonify(error='Expected a JSON object.'), 400
        if not cs.is_removable_id(body.get('id')):
            return jsonify(error='id must be the id of a stored review (f, the date, the time and six hex digits) or of '
                                 'an adjusted area (adj- and 12 hex digits).'), 400
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
