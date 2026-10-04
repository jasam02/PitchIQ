"""Run: start.cmd "C:\\path\\match.mp4". Local-only review server on port 8765."""
import argparse
import hashlib
import json
import math
import threading
import traceback
import uuid
from datetime import datetime
from pathlib import Path

from flask import Flask, abort, jsonify, request, send_file
from waitress import serve
from pipeline import ROOT, gpu_info, metadata, run


def validate(payload, duration):
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
    return {'start': start, 'seconds': seconds, 'imageSize': image_size,
            'ballTiles': bool(payload.get('ballTiles', False)), 'boundaries': sorted(clean, key=lambda a: a['time'])}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', required=True)
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--run', action='store_true', help='Process the first 60 seconds immediately.')
    args = parser.parse_args()
    source = Path(args.video).resolve(strict=True)
    if not source.is_file():
        parser.error('Supply a video file.')
    meta, gpu = metadata(source), gpu_info()
    with source.open('rb') as handle:
        video_key = hashlib.file_digest(handle, 'sha256').hexdigest()
    directory = ROOT/'runs'/video_key[:16]
    directory.mkdir(parents=True, exist_ok=True)
    app = Flask(__name__, static_folder=None)
    app.config['MAX_CONTENT_LENGTH'] = 128*1024
    gate = threading.Lock()
    cancel = threading.Event()
    state = {'state': 'idle', 'stage': 'Ready', 'progress': 0, 'gpu': gpu}
    current = {'directory': None}
    last = directory/'latest.json'
    if last.exists():
        previous = json.loads(last.read_text())
        candidate = directory/previous['run']
        if candidate.parent == directory and (candidate/'result.json').exists():
            current['directory'] = candidate
            state.update(state='complete', stage='Previous result loaded', progress=1)
    defaults = {'start': 0, 'seconds': min(60, meta['duration']), 'imageSize': 1280, 'ballTiles': False, 'boundaries': []}
    saved = directory/'settings.json'
    if saved.exists():
        try:
            defaults = validate(json.loads(saved.read_text()), meta['duration'])
        except (ValueError, KeyError, TypeError):
            pass

    @app.before_request
    def local_only():
        if request.host not in {f'127.0.0.1:{args.port}', f'localhost:{args.port}'}:
            abort(403)
        origin = request.headers.get('Origin')
        if origin and origin not in {f'http://127.0.0.1:{args.port}', f'http://localhost:{args.port}'}:
            abort(403)
        if request.method == 'POST' and not request.is_json:
            abort(415)

    @app.after_request
    def headers(response):
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['Cache-Control'] = 'no-store'
        return response

    @app.get('/')
    def index():
        return send_file(ROOT/'index.html')

    @app.get('/video')
    def video():
        return send_file(source, conditional=True)

    @app.get('/api/info')
    def info():
        return jsonify(video=meta, gpu=gpu, videoKey=video_key, settings=defaults)

    @app.get('/api/status')
    def status():
        with gate:
            return jsonify(dict(state) | {'hasResult': bool(current['directory'] and (current['directory']/'result.json').exists())})

    @app.get('/api/result')
    def result():
        path = current['directory']
        if path is None or not (path/'result.json').exists():
            abort(404)
        return send_file(path/'result.json', as_attachment=request.args.get('download') == '1', download_name='pitchiq-gpu-result.json')

    @app.get('/api/preview')
    def live_preview():
        path = current['directory']
        if path is None or not (path/'preview.jpg').exists():
            abort(404)
        # Read into memory so Windows does not hold a file lock during replacement.
        from io import BytesIO
        return send_file(BytesIO((path/'preview.jpg').read_bytes()), mimetype='image/jpeg')

    def begin(config):
        with gate:
            if state['state'] == 'running':
                raise ValueError('A run is already processing.')
            cancel.clear()
            run_name = datetime.now().strftime('%Y%m%d-%H%M%S')+'-'+uuid.uuid4().hex[:6]
            destination = directory/run_name
            current['directory'] = destination
            state.clear()
            state.update(state='running', stage='Starting', progress=0, gpu=gpu, run=run_name)
            defaults.update(config)
            saved.write_text(json.dumps(config, indent=2), encoding='utf-8')

        def progress(values):
            with gate:
                state.update(values)

        def worker():
            try:
                report = run(source, destination, config, progress, cancel.is_set)
                report['sourceSHA256'] = video_key
                (destination/'result.json').write_text(json.dumps(report, separators=(',', ':'), allow_nan=False), encoding='utf-8')
                last.write_text(json.dumps({'run': run_name}), encoding='utf-8')
                progress({'state': report['status']})
            except Exception as error:
                traceback.print_exc()
                progress({'state': 'error', 'stage': 'Processing failed', 'error': str(error)})
        threading.Thread(target=worker, daemon=True).start()

    @app.post('/api/run')
    def start():
        try:
            begin(validate(request.get_json(), meta['duration']))
            return jsonify(ok=True)
        except (ValueError, KeyError, TypeError) as error:
            return jsonify(error=str(error)), 400

    @app.post('/api/stop')
    def stop():
        cancel.set()
        return jsonify(ok=True)

    print(f"PitchIQ GPU prototype | {gpu['name']} | CUDA {gpu['cuda']}", flush=True)
    print(f"Review: http://127.0.0.1:{args.port}", flush=True)
    print(f"Video: {source}\nResults: {directory}\nCtrl+C stops the server.", flush=True)
    if args.run:
        begin(defaults)
    serve(app, host='127.0.0.1', port=args.port, threads=6)


if __name__ == '__main__':
    main()
