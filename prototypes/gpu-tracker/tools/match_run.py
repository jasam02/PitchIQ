"""Run Full Match Mode without the page, into a folder of your choosing (benchmarks, end-to-end tests).

    python tools/match_run.py --video C:/path/match.mp4 --out C:/scratch/match1 --workers 3
    python tools/match_run.py --video 3a043f106ecf56c2 --start 60 --end 5700 --halftime 2820 3700 --out C:/scratch/m2
    python tools/match_run.py --resume C:/scratch/match1                      # continue an interrupted one
    python tools/match_run.py --video 3a043f106ecf56c2 --discovery off --out C:/scratch/m3   # no calibration discovery

The same planner, manifest, scheduler, workers and assembly the page uses (fullmatch/), with the processing settings
of runs/<key>/settings.json unless --settings names another file (the keyframes in it are shared by every segment), and
the calibration discovery first where no keyframe is within reach of a segment, as the page does (--discovery off: none).
Nothing under runs/ is written: the match lands in --out (manifest.json, segments/, logs/, assembly/). Prints the
progress every --every seconds and, at the end, the runtime report (wall clock, aggregate fps, per-worker fps, peaks).
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fullmatch import manifest as mf, plan, routes, scheduler as sch, system  # noqa: E402


def resolve_video(video):
    path = Path(video)
    if path.exists():
        return path
    candidate = ROOT/'uploads'/f'{video}.mp4'
    if candidate.exists():
        return candidate
    raise SystemExit(f'No video at {video} and no upload {candidate}')


def source_key(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def load_settings(key, explicit=None):
    path = Path(explicit) if explicit else ROOT/'runs'/key[:16]/'settings.json'
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}


def processing_settings(raw):
    out = {'imageSize': int(raw.get('imageSize', 1280)), 'ballTiles': bool(raw.get('ballTiles', False)),
           'boundaries': raw.get('boundaries', []), 'calibrations': raw.get('calibrations', []),
           'fieldFilter': bool(raw.get('fieldFilter', True)), 'touchlineMargin': float(raw.get('touchlineMargin', .35)),
           'maxPerTeam': int(raw.get('maxPerTeam', 11)), 'debug': bool(raw.get('debug', True))}
    for key in ('detectBatch', 'ballScanEvery', 'nearPass'):
        if key in raw:
            out[key] = raw[key]
    return out


def folder_bytes(path):
    return sum(p.stat().st_size for p in Path(path).rglob('*') if p.is_file())


def report(s, manifest):
    st = s.status()
    runtime = manifest.data.get('runtime') or {}
    entries = manifest.data['segments']
    fps = [e['summary']['processingFPS'] for e in entries if e.get('summary') and e['summary'].get('processingFPS')]
    return {'state': manifest.data['state'], 'reason': manifest.data.get('reason'), 'segments': st['segments'],
            'videoSeconds': st['video'], 'frames': st['frames'], 'discoverySeconds': runtime.get('discoverySeconds'),
            'discovery': (manifest.data.get('discovery') or {}).get('summary'), 'trackingSeconds': runtime.get('trackingSeconds'),
            'assemblySeconds': runtime.get('assemblySeconds'), 'totalSeconds': runtime.get('totalSeconds'),
            'aggregateFPS': runtime.get('aggregateFPS'), 'workers': st['workers']['target'], 'threads': st['workers']['threads'],
            'perSegmentFPS': {'min': min(fps), 'mean': round(sum(fps)/len(fps), 2), 'max': max(fps)} if fps else None,
            'peaks': runtime.get('peaks') or st['peaks'], 'restarts': st['restarts'], 'errors': st['errors'],
            'diskBytes': {part: folder_bytes(manifest.folder/part) for part in ('segments', 'assembly')} |
                         {'total': folder_bytes(manifest.folder)}}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--video', help='a video path or an upload key')
    ap.add_argument('--out', help='the match folder to create')
    ap.add_argument('--resume', help='an existing match folder to continue')
    ap.add_argument('--start', type=float)
    ap.add_argument('--end', type=float)
    ap.add_argument('--halftime', type=float, nargs=2, metavar=('BREAK_START', 'BREAK_END'))
    ap.add_argument('--segment-seconds', type=float, default=plan.SEGMENT_S)
    ap.add_argument('--pad', type=float, default=plan.PAD_S, help='seconds of context on each side of a segment')
    ap.add_argument('--workers', default='auto', help="'auto' or 1-4")
    ap.add_argument('--threads', type=int, default=0, help='CPU threads per worker for OpenCV/torch/BLAS (0: library defaults)')
    ap.add_argument('--settings', help='processing settings JSON (default runs/<key>/settings.json)')
    ap.add_argument('--no-assemble', action='store_true')
    ap.add_argument('--discovery', choices=('auto', 'off'), default='auto',
                    help="the calibration discovery before tracking ('auto': where no keyframe is within reach)")
    ap.add_argument('--every', type=float, default=30.0)
    a = ap.parse_args(argv)
    snap = system.snapshot()
    if a.resume:
        manifest = mf.Manifest.load(Path(a.resume)/'manifest.json')
    else:
        if not (a.video and a.out):
            ap.error('--video and --out (or --resume)')
        import pipeline
        source = resolve_video(a.video)
        meta = pipeline.metadata(source)
        key = source_key(source)
        settings = processing_settings(load_settings(key, a.settings))
        periods = plan.periods(meta['duration'], a.start, a.end, a.halftime)
        segments = plan.plan(meta['frames'], meta['fps'], periods, a.segment_seconds, a.pad)
        status = routes.calibration_status(settings, segments, float(meta['fps']))
        runs = a.discovery == 'auto' and status['automatic'] > 0
        manifest = mf.Manifest.create(Path(a.out), {'key': key, 'path': str(source.resolve()), 'bytes': source.stat().st_size},
                                      meta, periods, segments, settings, {'mode': a.workers}, a.pad, a.segment_seconds,
                                      extra={'calibrationStatus': status,
                                             'discovery': {'mode': a.discovery, 'state': 'pending' if runs else 'off'}})
    if a.workers == 'auto':
        workers, why = system.auto_workers(snap)
    else:
        workers, note = system.allowed(int(a.workers), snap)
        why = note
    print(f"match {manifest.data['matchId']}: {len(manifest.data['segments'])} segments, {workers} workers ({why}), "
          f"{manifest.data['plan']['processedFrames']} frames", flush=True)
    assemble = None
    if not a.no_assemble:
        from fullmatch import assemble as assembly
        assemble = assembly.assemble
    s = sch.MatchScheduler(manifest, workers, threads=a.threads, assemble=assemble)
    started = time.time()
    s.start(resume=bool(a.resume))
    try:
        while not s.wait(a.every):
            st = s.status()
            eta = st['eta']
            print(f"[{time.time()-started:7.0f}s] {st['stage']:10s} {st['segments']['done']}/{st['segments']['total']} segments, "
                  f"{st['video']['doneSeconds']/60:.1f}/{st['video']['totalSeconds']/60:.1f} min, {st['progress']*100:.1f}%, "
                  f"{st['workers']['active']} active, rate {st['rate']} fps, ETA {eta['likely']/60:.0f} min "
                  f"({eta['low']/60:.0f}-{eta['high']/60:.0f}{'' if eta['measured'] else ', benchmark'})"
                  + (f", HELD: {st['held']}" if st['held'] else ''), flush=True)
    except KeyboardInterrupt:
        print('stopping...', flush=True)
        s.stop()
        s.wait()
    out = report(s, manifest)
    mf.write_json(manifest.folder/'runtime_report.json', out)
    print(json.dumps(out, indent=1), flush=True)
    return 0 if manifest.data['state'] == 'complete' else 1


if __name__ == '__main__':
    sys.exit(main())
