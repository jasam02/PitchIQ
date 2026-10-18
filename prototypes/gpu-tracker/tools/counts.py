"""Count-recall of a run against the hand-counted frames in tools/annotations/counts.json.

    python tools/counts.py RUN_DIR [--key <upload key>] [--json out]

At every annotated video time inside the run's segment, the people the run tracked on the nearest frame
(every state but rejected: players, candidates, unresolved people and ignored officials, since the counts
include the referee) are compared with the count of people on the playing field. Reported per frame and
overall: tracked, counted, the recall proxy tracked/counted (capped at 1), the over-count beyond the
count plus its uncertainty, and the same for the small people (box height under 40 px) as a share of the
frame's people. This is the only detection ground truth available without a box annotation tool; it says
how many people were tracked, not which. To split the misses by cause, each row also gives the people whose kit is
undecided ('unknown' state, included in tracked), the detections rejected on that frame by reason, and how many of
those the final homography puts inside the pitch (tools/diagnose.py; '-' where the frame's registration is not
confident enough to check): a person missing although a detection inside the final pitch was rejected was filtered,
otherwise never detected (or dropped by the local tracker: the frame's 'dropped' list when the result has one).
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import diagnose  # noqa: E402

ANNOTATIONS = ROOT/'tools'/'annotations'/'counts.json'
TRACKED = ('confirmed', 'uncertain', 'candidate', 'nontarget', 'unknown')


def evaluate(result, annotations, key):
    frames = result['frames']
    height = result['video']['height']
    rows = []
    for stamp, truth in sorted(((float(t), v) for t, v in annotations.get(key, {}).items() if not str(t).startswith('_')), key=lambda x: x[0]):
        if not frames or stamp < frames[0]['time']-.05 or stamp > frames[-1]['time']+.05:
            continue
        frame = min(frames, key=lambda f: abs(f['time']-stamp))
        people = [p for p in frame['people'] if p.get('state') in TRACKED]
        small = sum(1 for p in people if p['box'][3]*height < 40)
        count, unc = int(truth['count']), int(truth.get('uncertainty', 0))
        checked = diagnose.counts_rows({'video': result['video'], 'tactical': result.get('tactical'), 'frames': [frame]})[0][0]
        rows.append({'time': stamp, 'frame': frame['time'], 'tracked': len(people), 'counted': count, 'uncertainty': unc,
                     'recall': round(min(1.0, len(people)/max(1, count)), 3), 'overCount': max(0, len(people)-count-unc),
                     'missing': max(0, count-unc-len(people)), 'smallTracked': small,
                     'identified': sum(1 for p in people if p.get('id')), 'medianHeightPx': round(sorted(p['box'][3]*height for p in people)[len(people)//2], 1) if people else None,
                     'kitUndecided': checked['unknown'], 'rejected': dict(sorted((k[9:], v) for k, v in checked.items() if k.startswith('rejected:'))),
                     'rejectedInside': checked['rejectedInside'], 'rejectedInsideReasons': checked['rejectedInsideReasons'],
                     'dropped': checked['dropped'] if checked['dropped'] != '' else None})
    if not rows:
        return {'frames': 0}
    return {'frames': len(rows), 'recall': round(sum(r['recall'] for r in rows)/len(rows), 3),
            'overCountFrames': sum(1 for r in rows if r['overCount']), 'missingTotal': sum(r['missing'] for r in rows),
            'trackedTotal': sum(r['tracked'] for r in rows), 'countedTotal': sum(r['counted'] for r in rows), 'rows': rows}


def table(report):
    if not report.get('frames'):
        return 'no annotated frame inside this run'
    lines = ['| time | tracked | counted (±) | recall | missing | over | small (<40 px) | identified | median px | kit undecided | rejected | inside final pitch | dropped |',
             '|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    for r in report['rows']:
        rejected = ', '.join(f'{k} {v}' for k, v in r['rejected'].items()) or '-'
        lines.append(f"| {r['time']:.0f} s | {r['tracked']} | {r['counted']} (±{r['uncertainty']}) | {r['recall']} | {r['missing']} | {r['overCount']} | {r['smallTracked']} | {r['identified']} | {r['medianHeightPx']} "
                     f"| {r['kitUndecided']} | {rejected} | {'-' if r['rejectedInside'] is None else r['rejectedInside']}"
                     f"{' ('+r['rejectedInsideReasons']+')' if r['rejectedInside'] else ''} | {'-' if r['dropped'] is None else r['dropped']} |")
    lines.append(f"\noverall: recall {report['recall']} over {report['frames']} frames, {report['missingTotal']} people missing beyond the uncertainty, "
                 f"{report['overCountFrames']} frames over-counted ({report['trackedTotal']} tracked vs {report['countedTotal']} counted)")
    return '\n'.join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('run_dir')
    ap.add_argument('--key', help='upload key (default: from meta.json or the video name)')
    ap.add_argument('--json')
    a = ap.parse_args(argv)
    run = Path(a.run_dir)
    result = json.loads((run/'result.json').read_text(encoding='utf-8'))
    key = a.key
    if not key and (run/'meta.json').exists():
        key = json.loads((run/'meta.json').read_text(encoding='utf-8')).get('key')
    if not key:
        key = Path(result['video']['name']).stem
    annotations = json.loads(ANNOTATIONS.read_text(encoding='utf-8'))
    report = evaluate(result, annotations, key)
    print(table(report))
    if a.json:
        Path(a.json).write_text(json.dumps(report, indent=1), encoding='utf-8')


if __name__ == '__main__':
    sys.exit(main())
