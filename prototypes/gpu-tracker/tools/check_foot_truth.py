"""Score a run's player positions against hand-marked ground contacts (tools/annotations/foot_positions.json).

    python tools/check_foot_truth.py RUN_DIR [RUN_DIR ...] [--stabilized] [--annotations FILE]

For every mark at a processed frame of the run (matched by the source frame index) it projects the marked foot through
that frame's final homography (frames[].calibration.H, re-oriented; field.unproject): the reference position in metres,
which shares the run's calibration but not its detector. Against it, per mark:
    box px      the raw box's bottom-centre (geometry.foot_point, the pixel the pipeline maps) minus the marked pixel:
                dx, dy (dy > 0: the box ends below the feet) and the distance. The detector's error, independent of the
                calibration
    raw m       the person's stored pitch position minus the reference: the distance, and its component along the
                camera's line of sight at the mark ('away' > 0: the player is shown farther from the camera)
    stabilized  the stabilized position at that frame, its kind (O/I/P/U) and its error the same way: from
                result['stabilized'] when the run has it ('stored'), and with --stabilized from
                soccer.trajectory.stabilize() computed on the result ('computed')
The person is found by the box: the marks keep the raw detector box they were made on, so the person on that frame whose
box overlaps it by MATCH_IOU or more. When two do (a second box on the same player), the one with the mark's id; on
another run of the same clip, where the ids were handed out anew, the best-overlapping box whatever its id ('by box').
A person who carries the mark's id with another box is another player there and is never taken. The overlap (iou) is
reported per mark.
A mark is an event when the box's bottom-centre misses it vertically by more than max(EVENT_PX, EVENT_SHARE h_ref), h_ref
the median height of the person's local track over +-HEIGHT_HALF (the stabilizer's rule; the box's own height when the
track has fewer than HEIGHT_MIN boxes there): the box lost the legs, ended at another player's head or ran past the feet.
It is normal when the miss is under that threshold, and ambiguous when it lies within the mark's own sigma_px of it
(either side). sigma_m is the mark's own uncertainty in metres (sigma_px times the metres per vertical pixel there): an
error below it is within the reference's precision. The summary gives the median, 90th percentile and maximum per
class. Uses the run's pitch size (tactical.pitchSize). Read-only."""
import argparse
import bisect
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from soccer import field  # noqa: E402
from soccer.geometry import foot_point  # noqa: E402

EVENT_PX = 3.0       # px: a box bottom farther than this from the marked feet ...
EVENT_SHARE = .15    # ... and than this share of the reference height is a box-extent event (soccer/trajectory.py's rule)
HEIGHT_HALF = 1.0    # s: the reference height h_ref is the median box height of the track within this of the mark ...
HEIGHT_MIN = 5       # ... over at least this many boxes (else the box's own height)
MATCH_IOU = .5       # box overlap with the marked box that identifies the person
CLASSES = ('normal', 'event', 'ambiguous')


def annotations_path():
    return ROOT/'tools'/'annotations'/'foot_positions.json'


def video_key(run):
    """The upload key: from a scratch run's meta.json, else from the runs/<key>/<run> layout of the server's runs."""
    meta = run/'meta.json'
    if meta.exists():
        info = json.loads(meta.read_text(encoding='utf-8'))
        return info.get('key') or Path(info.get('video', '')).stem
    return run.resolve().parent.name


def iou(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[0]+a[2], b[0]+b[2]), min(a[1]+a[3], b[1]+b[3])
    inter = max(0.0, x1-x0)*max(0.0, y1-y0)
    union = a[2]*a[3]+b[2]*b[3]-inter
    return inter/union if union > 0 else 0.0


def person_for(frame, mark):
    """(person, how, overlap) for a mark: the person with the mark's id when its box overlaps the marked box by MATCH_IOU
    or more ('id'), else the best-overlapping box if by MATCH_IOU or more ('box'); (None, 'none', best overlap) when no
    box does. overlap: the IoU of the chosen (or best) box with the marked one."""
    people = [p for p in frame.get('people') or [] if p.get('box')]
    if not people:
        return None, 'none', 0.0
    overlaps = [iou(p['box'], mark['box']) for p in people]
    same = [k for k, p in enumerate(people) if mark.get('id') and p.get('id') == mark['id'] and overlaps[k] >= MATCH_IOU]
    if same:
        k = max(same, key=lambda k: overlaps[k])
        return people[k], 'id', overlaps[k]
    k = int(np.argmax(overlaps))
    if overlaps[k] >= MATCH_IOU:
        return people[k], 'box', overlaps[k]
    return None, 'none', overlaps[k]


def reference_height(frames, times, index, person, height):
    """h_ref in pixels: the median box height of the person's local track within +-HEIGHT_HALF of frame index (the
    person's own box when the track has fewer than HEIGHT_MIN boxes there). times: the frames' times."""
    track, t0 = person.get('track'), times[index]
    heights = []
    if track is not None:
        for f in frames[bisect.bisect_left(times, t0-HEIGHT_HALF-1e-9):]:
            if f['time'] > t0+HEIGHT_HALF+1e-9:
                break
            heights += [p['box'][3]*height for p in f.get('people') or [] if p.get('track') == track and p.get('box')]
    return float(np.median(heights)) if len(heights) >= HEIGHT_MIN else person['box'][3]*height


def lookup(block, frames, scale):
    """position(person, frame index) -> ((x, y) metres, kind) or None from a block of per-identity tracks
    ({'frameTimes', 'ids', 'anonymous'}; positions times scale are metres)."""
    times = block.get('frameTimes') or []

    def at(person, i):
        tracks, key = (block.get('ids'), person.get('id')) if person.get('id') else (block.get('anonymous'), str(person['track']))
        track = (tracks or {}).get(key)
        k = nearest(times, frames[i]['time'])
        if track is None or k is None or track['x'][k] is None:
            return None
        return (track['x'][k]*scale[0], track['y'][k]*scale[1]), track['kind'][k]
    return at


def sources_of(result, width, height, size, computed):
    """({name: lookup}, notes): 'stored' from result['stabilized'] (normalized positions) when the run has it,
    'computed' from soccer.trajectory.stabilize() (metres) when asked for and importable."""
    sources, notes = {}, []
    if result.get('stabilized'):
        sources['stored'] = lookup(result['stabilized'], result['frames'], (size.length, size.width))
    if computed:
        try:
            from soccer import trajectory
        except ImportError as error:
            notes.append(f'soccer.trajectory not importable ({error})')
        else:
            sources['computed'] = lookup(trajectory.stabilize(result, width, height, size), result['frames'], (1.0, 1.0))
    return sources, notes


def nearest(times, t, tolerance=.01):
    k = bisect.bisect_left(times, t)
    best = min((j for j in (k-1, k) if 0 <= j < len(times)), key=lambda j: abs(times[j]-t), default=None)
    return best if best is not None and abs(times[best]-t) <= tolerance else None


def score(run, computed=False, annotations=None):
    """(records, notes, source names): one record per mark at a frame of the run, what could not be computed, and the
    stabilized sources scored. annotations: the marks file (tools/annotations/foot_positions.json when None)."""
    run = Path(run)
    path = Path(annotations) if annotations else annotations_path()
    marks = json.loads(path.read_text(encoding='utf-8')).get(video_key(run)) or []
    result = json.loads((run/'result.json').read_text(encoding='utf-8'))
    W, Hh = result['video']['width'], result['video']['height']
    size = field.PitchSize.from_json((result.get('tactical') or {}).get('pitchSize'))
    by_frame = {f['frame']: i for i, f in enumerate(result['frames'])}
    times = [f['time'] for f in result['frames']]
    sources, notes = sources_of(result, W, Hh, size, computed)
    records = []
    for m in marks:
        i = by_frame.get(m['frame'])
        if i is None:
            continue
        frame = result['frames'][i]
        rec = {'mark': m, 'index': i, 'state': 'ok'}
        records.append(rec)
        cal = frame.get('calibration') or {}
        H = field.orient(np.array(cal['H'], float).reshape(3, 3), W, Hh) if cal.get('H') else None
        x, y = m['foot']
        if H is None or (field.image_depth(H, [(x, y-1), (x, y+1)]) <= 1e-9).any():
            rec['state'] = 'no registration' if H is None else 'beyond the horizon'
            continue
        ref, up, down = field.unproject(H, [(x, y), (x, y-1), (x, y+1)])
        away = up-down
        rec['ref'] = ref
        rec['away'] = away/max(1e-12, float(np.hypot(*away)))
        rec['sigma_m'] = m['sigma_px']*float(np.hypot(*away))/2
        person, rec['match'], rec['iou'] = person_for(frame, m)
        if person is None:
            rec['state'] = 'no person'
            continue
        fx, fy = foot_point(person['box'])
        rec['dx'], rec['dy'] = fx*W-x, fy*Hh-y
        rec['h_ref'] = reference_height(result['frames'], times, i, person, Hh)
        rec['threshold'] = max(EVENT_PX, EVENT_SHARE*rec['h_ref'])
        rec['event'] = abs(rec['dy']) > rec['threshold']
        rec['class'] = 'ambiguous' if abs(abs(rec['dy'])-rec['threshold']) < m['sigma_px'] else 'event' if rec['event'] else 'normal'
        if person.get('pitch'):
            rec['raw'] = np.array([person['pitch'][0]*size.length, person['pitch'][1]*size.width])
        for name, at in sources.items():
            got = at(person, i)
            if got is not None:
                rec[name] = (np.array(got[0], float), got[1])
    return records, notes, list(sources)


def error(rec, position):
    d = position-rec['ref']
    return float(np.hypot(*d)), float(d @ rec['away'])


def row(rec, sources):
    m = rec['mark']
    head = f"  {m['time']:8.3f} s {m['id']:5} ({m['display']}) +-{m['sigma_px']:g} px"
    if rec['state'] != 'ok':
        return f"{head}: {rec['state']}" + (f" (best box overlap {rec['iou']:.2f})" if rec['state'] == 'no person' else '')
    out = (f"{head} = {rec['sigma_m']:.2f} m {rec['class']:9}{'' if rec['match'] == 'id' else ' (by box)'} iou {rec['iou']:.2f}"
           f" | box dx {rec['dx']:+6.1f} dy {rec['dy']:+6.1f} px (threshold {rec['threshold']:.1f})")
    if 'raw' in rec:
        dist, along = error(rec, rec['raw'])
        out += f" | raw {dist:5.2f} m (away {along:+5.2f})"
    else:
        out += ' | raw -'
    for name in sources:
        if name in rec:
            dist, along = error(rec, rec[name][0])
            out += f" | {name} {rec[name][1]} {dist:5.2f} m (away {along:+5.2f})"
        else:
            out += f' | {name} -'
    return out


def stats(values):
    if not values:
        return '-'
    v = np.asarray(values, float)
    return f'{np.median(v):.2f} / {np.percentile(v, 90):.2f} / {v.max():.2f}'


def summary(records, sources):
    """Lines of the per-class summary: median / p90 / max of the box pixel errors and of the metre errors."""
    scored = [r for r in records if r['state'] == 'ok']
    lines = []
    for label, group in (('all', scored),)+tuple((c, [r for r in scored if r['class'] == c]) for c in CLASSES):
        raw = [error(r, r['raw']) for r in group if 'raw' in r]
        line = (f"  {label:9} n {len(group):3}: box |dy| px {stats([abs(r['dy']) for r in group])} (dy median {median([r['dy'] for r in group]):+.1f}),"
                f" box px {stats([math.hypot(r['dx'], r['dy']) for r in group])} | raw m {stats([e[0] for e in raw])} (away median {median([e[1] for e in raw]):+.2f})")
        for name in sources:
            got = [error(r, r[name][0]) for r in group if name in r]
            line += f" | {name} m {stats([e[0] for e in got])} (n {len(got)}, away median {median([e[1] for e in got]):+.2f})"
        lines.append(line)
    uncertain = sum(r['mark']['sigma_px'] > 3 for r in scored)
    by_box = sum(r['match'] == 'box' for r in scored)
    lines.append(f"  marks scored {len(scored)} of {len(records)} at the run's frames ({by_box} matched by box alone; box overlap median "
                 f"{median([r['iou'] for r in scored]):.2f}, min {min([r['iou'] for r in scored], default=float('nan')):.2f}); {uncertain} uncertain "
                 f"(sigma > 3 px); mark sigma median {median([r['sigma_m'] for r in scored]):.2f} m")
    return lines


def median(values):
    return float(np.median(values)) if len(values) else float('nan')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('runs', nargs='+')
    parser.add_argument('--stabilized', action='store_true', help='also compute soccer.trajectory.stabilize() on each result')
    parser.add_argument('--annotations', help='the marks file (default tools/annotations/foot_positions.json)')
    args = parser.parse_args(argv)
    for run in args.runs:
        records, notes, sources = score(run, args.stabilized, args.annotations)
        print(f'{run} ({video_key(Path(run))}): {len(records)} marks at its frames (median / p90 / max)')
        for note in notes:
            print(f'  {note}')
        for line in summary(records, sources):
            print(line)
        for rec in records:
            print(row(rec, sources))


if __name__ == '__main__':
    main()
