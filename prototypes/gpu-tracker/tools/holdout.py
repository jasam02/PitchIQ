"""Calibration hold-out test of a run, from its saved registration.json (no detector needed).

    python tools/holdout.py RUN_DIR [--json out.json]

For every calibration keyframe of the run, the offline registration is rebuilt WITHOUT that keyframe and
without every anchor whose chain started from it (paint refinements rooted at it), and the rebuilt
homography at the keyframe's frame is scored against the landmarks you clicked there: the pixel error of
the projected landmarks, the error in metres of the unprojected clicks, and how the rebuilt timeline
reached that frame (source, nearest anchor, its age, confidence). This is the error of carrying the other
keyframes to this one through the camera motion, which is what every frame between keyframes relies on.

Also reported, from the full registration: the pan stability (how far the projected pitch corners move
between consecutive frames, in pixels, excluding cuts) and, when result.json is present, the share of
frames whose forward/backward disagreement exceeds 3 m.
"""
import argparse
import json
import math
import statistics
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from soccer import field, registration  # noqa: E402


def load_record(run_dir):
    run_dir = Path(run_dir)
    record = json.loads((run_dir/'registration.json').read_text(encoding='utf-8'))
    result_path = run_dir/'result.json'
    result = json.loads(result_path.read_text(encoding='utf-8')) if result_path.exists() else None
    return record, result


def anchors_of(record):
    out = []
    for a in record['anchors']:
        out.append(registration.Anchor(int(a['index']), float(a['time']), np.asarray(a['H'], np.float64).reshape(3, 3), float(a['observability']),
                                       a['source'], dict(a.get('details') or {}), int(a.get('root', a['index']))))
    return out


def log_of(record):
    return [{'t': e['t'], 'dt': e['dt'], 'M': None if e['M'] is None else np.asarray(e['M'], np.float64).reshape(3, 3),
             'reliable': e['reliable'], 'cut': e['cut'], 'quality': e.get('quality') or 'none'} for e in record['frames']]


def landmark_errors(H, keyframe, width, height, size):
    """(pixel errors, metre errors) of a keyframe's landmarks under H (world -> pixels)."""
    marks = field.landmarks(size)
    points = [p for p in keyframe['points'] if p['name'] in marks]
    if not points or H is None:
        return [], []
    clicks = np.array([(p['x']*width, p['y']*height) for p in points], np.float64)
    world = np.array([marks[p['name']] for p in points], np.float64)
    projected = field.project(H, world)
    px = np.hypot(*(projected-clicks).T)
    front = field.image_depth(H, clicks) > 1e-9
    metres = np.full(len(points), np.nan)
    if front.any():
        metres[front] = np.hypot(*(field.unproject(H, clicks[front])-world[front]).T)
    return [float(v) for v in px], [float(v) for v in metres]


def evaluate(record, result=None, disagreement_limit=3.0):
    width, height = record['width'], record['height']
    size = field.PitchSize.from_json(record.get('size'))
    log = log_of(record)
    anchors = anchors_of(record)
    out = {'keyframes': [], 'panStability': None, 'disagreementShareOver3m': None, 'frames': len(log)}
    for k in record.get('keyframes', []):
        index = k.get('index')
        if index is None:
            out['keyframes'].append({'time': k['time'], 'index': None, 'status': 'not applied in this run (outside the segment?)'})
            continue
        kept = [a for a in anchors if a.root != index and a.index != index]
        timeline = registration.build_timeline(log, kept, width, height, size)
        entry = timeline[index] if 0 <= index < len(timeline) else None
        H = entry['H'] if entry else None
        px, metres = landmark_errors(H, k, width, height, size)
        finite = [m for m in metres if math.isfinite(m)]
        out['keyframes'].append({
            'time': k['time'], 'index': index, 'landmarks': len(k['points']), 'ownRmsMetres': k.get('rms'), 'ownObservability': k.get('observability'),
            'status': 'no coverage without this keyframe' if H is None else 'rebuilt',
            'pxMean': round(statistics.mean(px), 1) if px else None, 'pxMax': round(max(px), 1) if px else None,
            'metresMean': round(statistics.mean(finite), 2) if finite else None, 'metresMax': round(max(finite), 2) if finite else None,
            'behindHorizon': len(metres)-len(finite) if metres else None,
            'rebuiltFrom': None if entry is None else {'source': entry['source'], 'direction': entry['direction'], 'anchor': entry['anchor'],
                                                       'ageSeconds': entry['age'], 'confidence': entry['confidence'], 'observability': entry['observability']},
            'anchorsLeftOut': sum(1 for a in anchors if a.root == index or a.index == index)})
    # Pan stability over the full registration: how far the model's visible ground jumps between consecutive
    # frames beyond what the measured camera motion predicts (a smooth carry gives ~0; a correction at an
    # anchor, or a wrong motion estimate, gives a jump). Measured at image grid points in the lower part of
    # the frame that see the ground under both homographies.
    full = registration.build_timeline(log, anchors, width, height, size)
    xs, ys = np.linspace(.1, .9, 5)*width, np.linspace(.4, .95, 4)*height
    grid = np.array([(x, y) for y in ys for x in xs], np.float64)
    jumps, at_anchors = [], []
    anchor_frames = {a.index for a in anchors}
    for i in range(1, len(full)):
        previous, current, step = full[i-1]['H'], full[i]['H'], log[i]
        if previous is None or current is None or step['cut'] or step['M'] is None or not step['reliable']:
            continue
        predicted = np.asarray(step['M'], np.float64) @ previous
        front = (field.image_depth(predicted, grid) > 1e-9) & (field.image_depth(current, grid) > 1e-9)
        if front.sum() < 4:
            continue
        try:
            moved = field.project(current, field.unproject(predicted, grid[front]))
        except np.linalg.LinAlgError:
            continue
        jump = float(np.hypot(*(moved-grid[front]).T).mean())
        (at_anchors if i in anchor_frames else jumps).append(jump)
    if jumps:
        ordered = sorted(jumps)
        out['panStability'] = {'frames': len(jumps), 'medianPx': round(ordered[len(jumps)//2], 2), 'p95Px': round(ordered[int(.95*len(jumps))-1], 2),
                               'maxPx': round(ordered[-1], 2), 'jumpsOver20Px': sum(1 for j in jumps if j > 20),
                               'anchorCorrectionsPx': [round(j, 1) for j in at_anchors]}
    if result is not None:
        values = [(f.get('calibration') or {}).get('disagreement') for f in result['frames']]
        values = [v for v in values if v is not None]
        if values:
            out['disagreementShareOver3m'] = round(sum(1 for v in values if v > disagreement_limit)/len(values), 3)
            out['disagreementMedianMetres'] = round(statistics.median(values), 2)
    out['calibratedShare'] = round(sum(1 for e in full if e['H'] is not None and e['confidence'] >= .2)/max(1, len(full)), 3)
    return out


def table(report):
    lines = ['| keyframe | landmarks | own rms m | px mean / max | metres mean / max | rebuilt from |', '|---|---|---|---|---|---|']
    for k in report['keyframes']:
        src = k.get('rebuiltFrom')
        how = k['status'] if not src or k['status'] != 'rebuilt' else f"{src['source']} {src['direction'] or ''} anchor {src['anchor']} ({src['ageSeconds']}s away, confidence {src['confidence']})"
        rms = k.get('ownRmsMetres')
        lines.append(f"| {k['time']:.1f}s | {k.get('landmarks', '-')} | {f'{rms:.2f}' if rms is not None else '-'} | "
                     f"{k.get('pxMean', '-')} / {k.get('pxMax', '-')} | {k.get('metresMean', '-')} / {k.get('metresMax', '-')} | {how if k['index'] is not None else k['status']} |")
    ps = report.get('panStability')
    lines.append('')
    lines.append(f"pan stability (ground jump beyond the measured camera motion, px per frame): {ps}" if ps else 'pan stability: no calibrated frames')
    lines.append(f"forward/backward disagreement > 3 m: {report.get('disagreementShareOver3m')} of frames (median {report.get('disagreementMedianMetres')} m); "
                 f"calibrated share {report.get('calibratedShare')}")
    return '\n'.join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('run_dir')
    ap.add_argument('--json')
    a = ap.parse_args(argv)
    record, result = load_record(a.run_dir)
    report = evaluate(record, result)
    print(table(report))
    if a.json:
        Path(a.json).write_text(json.dumps(report, indent=1), encoding='utf-8')


if __name__ == '__main__':
    sys.exit(main())
