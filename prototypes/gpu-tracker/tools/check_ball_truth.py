"""Score a run's ball against hand-marked ball positions (tools/annotations/ball_positions.json).

    python tools/check_ball_truth.py RUN_DIR [RUN_DIR ...]

For every marked time inside the run's segment it takes the nearest processed frame and reports the final ball (state,
origin, centre in source pixels, distance to the mark) and the first (causal) pass. A frame counts as:
    hit       the final ball is shown (TRACKED or MISSING) within max(2 diameters, 15 px) of the mark;
    wrong     the final ball is shown farther away: the path is on something else (the failure the user saw);
    absent    no ball shown although the mark is visible (a miss, but no false position);
    hidden    the mark is not visible (a leg in front) and no ball is shown: correct either way.
Nothing is written; the annotation file says how the marks were made."""
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def video_key(run):
    """The upload key: from a scratch run's meta.json, else from the runs/<key>/<run> layout of the server's runs."""
    meta = run/'meta.json'
    if meta.exists():
        info = json.loads(meta.read_text(encoding='utf-8'))
        return info.get('key') or Path(info.get('video', '')).stem
    return run.resolve().parent.name


def score(run):
    run = Path(run)
    marks = json.loads((ROOT/'tools'/'annotations'/'ball_positions.json').read_text(encoding='utf-8'))
    key = video_key(run)
    points = marks.get(key) or []
    result = json.loads((run/'result.json').read_text(encoding='utf-8'))
    W, H = result['video']['width'], result['video']['height']
    frames = result['frames']
    times = [f['time'] for f in frames]
    counts = {'hit': 0, 'wrong': 0, 'absent': 0, 'hidden': 0}
    rows = []
    for m in points:
        if not (times[0]-.05 <= m['time'] <= times[-1]+.05):
            continue
        i = min(range(len(times)), key=lambda k: abs(times[k]-m['time']))
        f = frames[i]
        b = f.get('ball') or {}
        shown = b.get('state') in ('TRACKED', 'MISSING') and b.get('center')
        first = b.get('firstPass') or {}
        if shown:
            cx, cy = b['center'][0]*W, b['center'][1]*H
            d = math.hypot(cx-m['x'], cy-m['y'])
            verdict = 'hit' if d <= max(2*m['diameter'], 15) else 'wrong'
            where = f'({cx:.0f},{cy:.0f}) {d:.0f} px off'
        else:
            verdict = 'absent' if m['visible'] else 'hidden'
            where = '-'
        counts[verdict] += 1
        rows.append(f"  {m['time']:6.1f} s mark ({m['x']},{m['y']}){'' if m['visible'] else ' hidden'}: {verdict:6} final {b.get('state')}/{b.get('origin')} {where}"
                    f" | first pass {first.get('state')}/{first.get('phase')}")
    return key, counts, rows


def main(argv=None):
    runs = (argv if argv is not None else sys.argv[1:])
    if not runs:
        raise SystemExit(__doc__)
    for run in runs:
        key, counts, rows = score(run)
        print(f'{run} ({key}): ' + ', '.join(f'{k} {v}' for k, v in counts.items()))
        for r in rows:
            print(r)


if __name__ == '__main__':
    main()
