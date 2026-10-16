"""The verification protocol of speed pass 3, run end to end on the GPU machine (tools/verify_speed_pass_3.cmd
calls this). Everything lands in a scratch folder outside runs/; nothing under runs/ is written.

    .venv\\Scripts\\python.exe tools\\verify_speed_pass_3.py --scratch C:/scratch/speed-pass-3
    .venv\\Scripts\\python.exe tools\\verify_speed_pass_3.py --dry-run          # print the commands only
    .venv\\Scripts\\python.exe tools\\verify_speed_pass_3.py --only tests,windows,diffs

Steps (--only picks a subset, in this order):
  gpu       nvidia-smi: the GPU must be idle (a game skewed earlier measurements)
  tests     the unit tests of this checkout
  base      a git worktree of the committed code (--base, default gpu-tracking-new) in <scratch>/base, with
            models/ copied in, so the same detector checkpoint is used
  windows   the two reference windows with the committed code (<scratch>/before/A, B) and this code (<scratch>/after/A, B):
              broadcast 2bac432528d0e568.mp4 10-50 s and Veo f597d73d53f7b1fd.mp4 0-20 s, 1536, ball tiles on,
              the owner's runs/<key>/settings.json
  diffs     tools/exact_diff.py and tools/compare.py before/after for A and B
  timing    the 0-60 s broadcast window, committed code then this code, back to back (<scratch>/timing/before, after):
            elapsed seconds, fps and the per-stage ms per frame from summary.timing
  reuse     two consecutive runs of window B in one process (the kept detectors and embedder) against two fresh
            runs (--fresh-models), compared with exact_diff: all four must be identical
  decode    tools/check_ball_decode.py on <scratch>/after/A and the timing run: the vectorized decode against the
            frozen copy of the old one on real frames
  clips     tools/ball_clips.py on the 60 s broadcast run: the 12.0-15.0, 22.1-24.3 and 40.8-42.8 s clips
Pass criteria (from the task): pure speedups identical in everything; the behavioural changes (batched detection,
scan policy) may move detection scores in the last digits: people, pitch positions, roster, tracks, labels and
events identical or explained, identified share and id switches unchanged, ball tracked share within 0.01, and the
clips still right."""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                     # prototypes/gpu-tracker
REPO = ROOT.parents[1]
PY = sys.executable
VIDEOS = {'A': ('2bac432528d0e568', 10, 40), 'B': ('f597d73d53f7b1fd', 0, 20), 'T': ('2bac432528d0e568', 0, 60)}
STEPS = ('gpu', 'tests', 'base', 'windows', 'diffs', 'timing', 'reuse', 'decode', 'clips')


class Runner:
    def __init__(self, dry):
        self.dry = dry
        self.log = []

    def run(self, command, cwd=None, check=True):
        shown = ' '.join(str(c) for c in command)
        print(f'\n$ {shown}', flush=True)
        self.log.append(shown)
        if self.dry:
            return ''
        t0 = time.perf_counter()
        completed = subprocess.run([str(c) for c in command], cwd=cwd, text=True, capture_output=True)
        print(completed.stdout[-6000:], flush=True)
        if completed.returncode and check:
            print(completed.stderr[-4000:], flush=True)
            raise SystemExit(f'failed ({completed.returncode}) after {time.perf_counter()-t0:.0f} s: {shown}')
        if completed.stderr.strip():
            print(completed.stderr[-2000:], flush=True)
        print(f'  ({time.perf_counter()-t0:.0f} s)', flush=True)
        return completed.stdout


def segment(checkout, key, start, seconds, out, extra=()):
    video = ROOT/'uploads'/f'{key}.mp4'
    settings = ROOT/'runs'/key/'settings.json'
    return [PY, checkout/'prototypes'/'gpu-tracker'/'tools'/'run_segment.py', '--video', video, '--start', start, '--seconds', seconds,
            '--out', out, '--settings', settings, '--image-size', 1536, '--ball-tiles', 1, *extra]


def timing_of(out):
    path = Path(out)/'result.json'
    if not path.exists():
        return None
    summary = json.loads(path.read_text(encoding='utf-8'))['summary']
    t = summary.get('timing', {})
    return {'elapsedSeconds': summary.get('elapsedSeconds'), 'processingFPS': summary.get('processingFPS'), 'frames': summary.get('processedFrames'),
            **{k: t.get(k) for k in ('detectMsPerFrame', 'ballTilesMsPerFrame', 'ballCropMsPerFrame', 'farTilesMsPerFrame', 'localCropsMsPerFrame',
                                     'trackingMsPerFrame', 'onlineMsPerFrame', 'offlineSeconds')},
            'ballTileScans': (summary.get('detection') or {}).get('ballTileScans'), 'identifiedShare': summary.get('identifiedShare'),
            'ballTrackedShare': (summary.get('ball') or {}).get('trackedShare'), 'pitchMeanConfidence': (summary.get('pitch') or {}).get('meanConfidence')}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--scratch', default='C:/scratch/speed-pass-3' if os.name == 'nt' else str(Path.home()/'scratch'/'speed-pass-3'))
    ap.add_argument('--base', default='gpu-tracking-new', help='the committed code to compare against (a git ref)')
    ap.add_argument('--only', help='comma-separated subset of: '+', '.join(STEPS))
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args(argv)
    steps = [s.strip() for s in a.only.split(',')] if a.only else list(STEPS)
    unknown = [s for s in steps if s not in STEPS]
    if unknown:
        ap.error(f'unknown steps {unknown}')
    scratch = Path(a.scratch)
    base = scratch/'base'
    r = Runner(a.dry_run)
    if not a.dry_run:
        scratch.mkdir(parents=True, exist_ok=True)
    if 'gpu' in steps:
        r.run(['nvidia-smi', '--query-gpu=name,utilization.gpu,memory.used', '--format=csv'], check=False)
        print('The GPU must be idle for the timing steps: no game, no other run.')
    if 'tests' in steps:
        r.run([PY, '-m', 'unittest', 'discover', '-s', ROOT/'tests', '-t', ROOT/'tests'], cwd=ROOT)
    if 'base' in steps:
        if not (base/'.git').exists():
            r.run(['git', 'worktree', 'add', base, a.base], cwd=REPO)
        else:
            print(f'\nbase worktree already at {base}')
        models = base/'prototypes'/'gpu-tracker'/'models'
        if not a.dry_run and not (models/'football-yolov8x.pt').exists() and (ROOT/'models'/'football-yolov8x.pt').exists():
            models.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT/'models'/'football-yolov8x.pt', models/'football-yolov8x.pt')
            print(f'copied the detector checkpoint into {models}')
    if 'windows' in steps:
        for name in ('A', 'B'):
            key, start, seconds = VIDEOS[name]
            r.run(segment(base, key, start, seconds, scratch/'before'/name))
            r.run(segment(REPO, key, start, seconds, scratch/'after'/name))
    if 'diffs' in steps:
        for name in ('A', 'B'):
            before, after = scratch/'before'/name/'result.json', scratch/'after'/name/'result.json'
            print(f'\n=== window {name}: exact_diff before/after ===')
            r.run([PY, HERE/'exact_diff.py', before, after], check=False)
            print(f'\n=== window {name}: compare before/after ===')
            r.run([PY, HERE/'compare.py', before, after], check=False)
    if 'timing' in steps:
        key, start, seconds = VIDEOS['T']
        r.run(segment(base, key, start, seconds, scratch/'timing'/'before'))
        r.run(segment(REPO, key, start, seconds, scratch/'timing'/'after'))
        if not a.dry_run:
            print('\n=== 60 s broadcast window, committed code vs this code (back to back) ===')
            for label in ('before', 'after'):
                print(label, json.dumps(timing_of(scratch/'timing'/label), indent=1))
            print('\n=== timing run: exact_diff before/after ===')
            r.run([PY, HERE/'exact_diff.py', scratch/'timing'/'before'/'result.json', scratch/'timing'/'after'/'result.json'], check=False)
    if 'reuse' in steps:
        key, start, seconds = VIDEOS['B']
        jobs = scratch/'reuse'
        if not a.dry_run:
            jobs.mkdir(parents=True, exist_ok=True)
        job = {'video': str(ROOT/'uploads'/f'{key}.mp4'), 'start': start, 'seconds': seconds, 'settings': str(ROOT/'runs'/key/'settings.json'),
               'image_size': 1536, 'ball_tiles': True}
        consecutive = [dict(job, out=str(jobs/'consecutive-1')), dict(job, out=str(jobs/'consecutive-2'))]
        fresh = [dict(job, out=str(jobs/'fresh-1')), dict(job, out=str(jobs/'fresh-2'))]
        if not a.dry_run:
            (jobs/'consecutive.json').write_text(json.dumps(consecutive, indent=1), encoding='utf-8')
            (jobs/'fresh.json').write_text(json.dumps(fresh, indent=1), encoding='utf-8')
        r.run([PY, HERE/'run_segment.py', '--batch', jobs/'consecutive.json'])
        r.run([PY, HERE/'run_segment.py', '--batch', jobs/'fresh.json', '--fresh-models'])
        for x, y in (('consecutive-1', 'consecutive-2'), ('consecutive-1', 'fresh-1'), ('fresh-1', 'fresh-2')):
            print(f'\n=== reuse: exact_diff {x} vs {y} (must be identical in everything) ===')
            r.run([PY, HERE/'exact_diff.py', jobs/x/'result.json', jobs/y/'result.json'], check=False)
    if 'decode' in steps:
        for folder in (scratch/'after'/'A', scratch/'timing'/'after'):
            if a.dry_run or (folder/'result.json').exists():
                r.run([PY, HERE/'check_ball_decode.py', folder], check=False)
    if 'clips' in steps:
        result = scratch/'timing'/'after'/'result.json'
        if a.dry_run or result.exists():
            r.run([PY, HERE/'ball_clips.py', result], check=False)
        before = scratch/'timing'/'before'/'result.json'
        if a.dry_run or before.exists():
            print('\n=== the same clips in the committed code\'s run, for comparison ===')
            r.run([PY, HERE/'ball_clips.py', before], check=False)
    if not a.dry_run:
        (scratch/'commands.txt').write_text('\n'.join(r.log)+'\n', encoding='utf-8')
        print(f'\nDone. Outputs under {scratch}; the commands run are in {scratch/"commands.txt"}.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
