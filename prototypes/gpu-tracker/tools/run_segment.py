"""Run the tracking pipeline on a segment of an uploaded video into a scratch directory, for evaluation.

    python tools/run_segment.py --video 6dfdc88ec6f2d100 --start 0 --seconds 60 --out C:/scratch/amateur3
    python tools/run_segment.py --batch jobs.json          # several segments in one process (same code for all)

The video is an upload key (uploads/<key>.mp4) or a path. Settings (image size, ball tiles, keyframes,
boundaries) come from runs/<key>/settings.json unless --settings names another file; --drop-keyframe T
leaves the calibration keyframe at T seconds out (hold-out experiments); --no-calibrations leaves all
of them out. Nothing under runs/ is read for the result or written: the run lands in --out with
result.json, registration.json, identities.json, events.log, preview.jpg and meta.json (the commit and
the configuration), exactly what the review server would have produced.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def commit_hash():
    try:
        return subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], cwd=ROOT, capture_output=True, text=True, timeout=10).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def resolve_video(video):
    path = Path(video)
    if path.exists():
        return path, path.stem
    candidate = ROOT/'uploads'/f'{video}.mp4'
    if candidate.exists():
        return candidate, video
    raise SystemExit(f'No video at {video} and no upload {candidate}')


def load_settings(key, explicit=None):
    path = Path(explicit) if explicit else ROOT/'runs'/key/'settings.json'
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding='utf-8'))


def build_config(settings, start, seconds, image_size=None, ball_tiles=None, drop_keyframes=(), no_calibrations=False, detect_batch=None,
                 ball_scan_every=None):
    calibrations = [] if no_calibrations else [c for c in settings.get('calibrations', [])
                                                if not any(abs(float(c['time'])-float(t)) < .05 for t in drop_keyframes)]
    config = {'start': float(start), 'seconds': float(seconds),
              'imageSize': int(image_size if image_size is not None else settings.get('imageSize', 1536)),
              'ballTiles': bool(settings.get('ballTiles', True) if ball_tiles is None else ball_tiles),
              'boundaries': settings.get('boundaries', []), 'calibrations': calibrations,
              'fieldFilter': bool(settings.get('fieldFilter', True)), 'touchlineMargin': float(settings.get('touchlineMargin', .35)),
              'maxPerTeam': int(settings.get('maxPerTeam', 11)), 'continueIdentities': False, 'debug': True}
    if detect_batch is not None:
        config['detectBatch'] = int(detect_batch)   # frames per whole-frame detector call (pipeline.DETECT_BATCH otherwise)
    if ball_scan_every is not None:
        config['ballScanEvery'] = float(ball_scan_every)   # seconds between safeguard tile scans while the ball is locked (pipeline.BALL_SCAN_S otherwise)
    return config


def run_job(job, quiet=False, resources=None):
    """resources: pipeline.Resources to run with (the process's shared pair by default: the detectors and the
    embedder stay loaded from one job of a --batch to the next, as in the server; a fresh Resources() loads
    them anew, as a separate process would)."""
    import pipeline
    video, key = resolve_video(job['video'])
    settings = load_settings(key, job.get('settings'))
    config = build_config(settings, job['start'], job['seconds'], job.get('image_size'), job.get('ball_tiles'),
                          job.get('drop_keyframes', ()), job.get('no_calibrations', False), job.get('detect_batch'), job.get('ball_scan_every'))
    out = Path(job['out'])
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    last = {'stage': None, 'tenth': -1}

    def progress(info):
        stage = info.get('stage')
        tenth = int(10*info.get('progress', 0)) if 'progress' in info else last['tenth']
        if not quiet and (stage != last['stage'] or tenth != last['tenth']):
            print(f"[{time.perf_counter()-started:6.0f}s] {stage} {info.get('progress', 0):.0%}"
                  + (f" fps {info['processingFPS']}" if 'processingFPS' in info else ''), flush=True)
        last['stage'], last['tenth'] = stage, tenth
    (out/'meta.json').write_text(json.dumps({'commit': commit_hash(), 'video': str(video), 'key': key, 'config': config,
                                             'started': time.strftime('%Y-%m-%d %H:%M:%S')}, indent=1), encoding='utf-8')
    report = pipeline.run(video, out, config, progress, lambda: False, None, resources)
    summary = report['summary']
    print(f"{out}: {summary['processedFrames']} frames in {summary['elapsedSeconds']:.0f}s ({summary['processingFPS']} fps), "
          f"identified {summary['identifiedShare']}, pitch {summary['pitch']['meanConfidence']}, ball tracked {summary['ball']['trackedShare']}", flush=True)
    return report


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--video')
    ap.add_argument('--start', type=float, default=0.0)
    ap.add_argument('--seconds', type=float, default=60.0)
    ap.add_argument('--out')
    ap.add_argument('--settings')
    ap.add_argument('--image-size', type=int)
    ap.add_argument('--ball-tiles', type=int, choices=(0, 1))
    ap.add_argument('--drop-keyframe', type=float, action='append', default=[])
    ap.add_argument('--no-calibrations', action='store_true')
    ap.add_argument('--detect-batch', type=int, help='frames per whole-frame detector call on the prefetch worker (default: pipeline.DETECT_BATCH)')
    ap.add_argument('--ball-scan-every', type=float, help='seconds between safeguard ball tile scans while the ball is locked (default: pipeline.BALL_SCAN_S)')
    ap.add_argument('--batch', help='JSON file: a list of jobs {video, start, seconds, out, settings?, drop_keyframes?, no_calibrations?, detect_batch?, ball_scan_every?}')
    ap.add_argument('--quiet', action='store_true')
    ap.add_argument('--fresh-models', action='store_true', help='load the detectors and the embedder anew for every job of a --batch (as separate processes would) '
                                                              'instead of keeping them loaded from one job to the next')
    a = ap.parse_args(argv)
    if a.batch:
        jobs = json.loads(Path(a.batch).read_text(encoding='utf-8'))
    else:
        if not a.video or not a.out:
            ap.error('--video and --out are required without --batch')
        jobs = [{'video': a.video, 'start': a.start, 'seconds': a.seconds, 'out': a.out, 'settings': a.settings, 'image_size': a.image_size,
                 'ball_tiles': None if a.ball_tiles is None else bool(a.ball_tiles), 'drop_keyframes': a.drop_keyframe, 'no_calibrations': a.no_calibrations,
                 'detect_batch': a.detect_batch, 'ball_scan_every': a.ball_scan_every}]
    for job in jobs:
        resources = None
        if a.fresh_models:
            import pipeline
            resources = pipeline.Resources()
        run_job(job, a.quiet, resources)


if __name__ == '__main__':
    main()
