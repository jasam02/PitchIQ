"""Score a run's final pitch registration against hand-marked landmarks (tools/annotations/pitch_landmarks.json).

    python tools/check_pitch_truth.py RUN_DIR [RUN_DIR ...]

For every marked frame inside the run's segment it takes the nearest processed frame's final homography
(frames[].calibration.H, re-oriented) and reports, per named landmark, the distance between the marked pixel and the
projected landmark (pixels) and between the marked pixel's pitch position and the landmark (metres); per marked line,
how far each marked point lies from the registered line (pixels, and metres on the pitch). Uses the run's own pitch
size (tactical.pitchSize), which is an assumption unless its source says otherwise. Read-only."""
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from soccer import field  # noqa: E402


def video_key(run):
    meta = run/'meta.json'
    if meta.exists():
        info = json.loads(meta.read_text(encoding='utf-8'))
        return info.get('key') or Path(info.get('video', '')).stem
    return run.resolve().parent.name


def line_world(name, size):
    for n, a, b in field.lines(size):
        if n == name:
            return np.array(a, float), np.array(b, float)
    return None


def px_to_line(H, a, b, point):
    """Pixel distance from an image point to the image of the world line through a and b."""
    pa, pb = field.project(H, [a, b])
    d = pb-pa
    n = math.hypot(*d)
    if n < 1e-9:
        return None
    return abs((point[0]-pa[0])*d[1]-(point[1]-pa[1])*d[0])/n


def metres_to_line(H, a, b, point):
    if field.image_depth(H, [point])[0] <= 0:
        return None
    w = field.unproject(H, [point])[0]
    d = b-a
    return abs((w[0]-a[0])*d[1]-(w[1]-a[1])*d[0])/math.hypot(*d)


def score(run):
    run = Path(run)
    marks = json.loads((ROOT/'tools'/'annotations'/'pitch_landmarks.json').read_text(encoding='utf-8')).get(video_key(run)) or []
    result = json.loads((run/'result.json').read_text(encoding='utf-8'))
    W, Hh = result['video']['width'], result['video']['height']
    ps = (result.get('tactical') or {}).get('pitchSize') or {}
    size = field.PitchSize(float(ps.get('length', field.DEFAULT_LENGTH)), float(ps.get('width', field.DEFAULT_WIDTH)), ps.get('source', 'standard'))
    frames = result['frames']
    times = [f['time'] for f in frames]
    out, errors = [], {'px': [], 'm': []}
    for m in marks:
        if not (times[0]-.05 <= m['time'] <= times[-1]+.05):
            continue
        f = frames[min(range(len(times)), key=lambda k: abs(times[k]-m['time']))]
        cal = f.get('calibration') or {}
        H = field.orient(np.array(cal['H'], float).reshape(3, 3), W, Hh) if cal.get('H') else None
        head = f"  {m['time']:7.2f} s (frame {f['time']:.2f}): source {cal.get('source')} confidence {cal.get('confidence')}"
        if H is None:
            out.append(head+': no registration')
            continue
        out.append(head)
        names = field.landmarks(size)
        for name, (x, y) in m.get('points', {}).items():
            if name not in names:
                continue
            p = field.project(H, [names[name]])[0]
            dpx = math.hypot(p[0]-x, p[1]-y)
            dm = None
            if field.image_depth(H, [(x, y)])[0] > 0:
                w = field.unproject(H, [(x, y)])[0]
                dm = math.hypot(w[0]-names[name][0], w[1]-names[name][1])
            errors['px'].append(dpx)
            if dm is not None:
                errors['m'].append(dm)
            out.append(f"    {name:22} marked ({x},{y}) registered ({p[0]:.0f},{p[1]:.0f}): {dpx:6.1f} px, {dm if dm is None else round(dm, 2)} m")
        for name, pts in m.get('lines', {}).items():
            ab = line_world(name, size)
            if ab is None:
                continue
            for point in pts:
                dpx, dm = px_to_line(H, *ab, point), metres_to_line(H, *ab, point)
                if dpx is not None:
                    errors['px'].append(dpx)
                if dm is not None:
                    errors['m'].append(dm)
                out.append(f"    {name:22} point {tuple(point)}: {dpx if dpx is None else round(dpx, 1)} px, {dm if dm is None else round(dm, 2)} m from the registered line")
    return out, errors, size


def main(argv=None):
    runs = argv if argv is not None else sys.argv[1:]
    if not runs:
        raise SystemExit(__doc__)
    for run in runs:
        rows, errors, size = score(run)
        px, m = np.array(errors['px']), np.array(errors['m'])
        summary = (f"median {np.median(px):.1f} px / {np.median(m):.2f} m, max {px.max():.1f} px / {m.max():.2f} m" if len(px) and len(m)
                   else 'no registered marked frame')
        print(f'{run}: pitch {size.length} x {size.width} m ({size.source}); {summary}')
        for r in rows:
            print(r)


if __name__ == '__main__':
    main()
