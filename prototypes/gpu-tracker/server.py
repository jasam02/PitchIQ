"""Run: start.cmd (then upload a video in the page) or start.cmd "C:\\path\\match.mp4". Local-only, port 8765."""
import argparse
import hashlib
import json
import math
import threading
import traceback
import uuid
from datetime import datetime
from pathlib import Path
from urllib.parse import unquote

from flask import Flask, abort, jsonify, request, send_file
from waitress import serve
from pipeline import ROOT, gpu_info, metadata, run
from soccer import calibration


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
            calibration.solve(clean_points, meta['width'], meta['height'])
        except ValueError as error:
            raise ValueError(f'Calibration at {timestamp:.2f}s: {error}')
        marks.append({'time': timestamp, 'points': clean_points})
    if len(set(a['time'] for a in marks)) != len(marks):
        raise ValueError('Duplicate calibration times; replace the existing keyframe.')
    margin = _number(payload, 'touchlineMargin', .35, 0, 1, 'Touchline tolerance must be between 0% and 100% of player height.')
    per_team = int(_number(payload, 'maxPerTeam', 11, 5, 30, 'Expected identities per team must be between 5 and 30.'))
    return {'start': start, 'seconds': seconds, 'imageSize': image_size,
            'ballTiles': bool(payload.get('ballTiles', False)), 'boundaries': sorted(clean, key=lambda a: a['time']),
            'calibrations': sorted(marks, key=lambda a: a['time']), 'fieldFilter': bool(payload.get('fieldFilter', True)),
            'touchlineMargin': margin, 'maxPerTeam': per_team, 'continueIdentities': bool(payload.get('continueIdentities', False)),
            'debug': bool(payload.get('debug', True))}


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', help='Optional: open this video at start. Videos can also be uploaded in the page.')
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--run', action='store_true', help='Process the first 60 seconds immediately (needs --video).')
    args = parser.parse_args()
    gpu = gpu_info()
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
        return jsonify(base | {'video': v.meta, 'videoKey': v.key, 'settings': v.defaults, 'previousIdentities': summary})

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
            return jsonify(dict(state) | {'hasResult': has, 'videoKey': v.key if v else None})

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
