"""Metrics of one tracking result, or of two results side by side with their agreement.

    python tools/compare.py A/result.json                 # metrics table of one run
    python tools/compare.py A/result.json B/result.json   # both runs and how much B agrees with A
    python tools/compare.py A B --json out.json           # the numbers as JSON too

Per frame: people tracked (confirmed or uncertain), candidates, people drawn in the 2D view (an identity
of the final roster with a valid pitch position), tracked people with a position but no identity (what the
2D view cannot show yet), and the identified share by box height. The identity, pitch, ball and timing
summaries come from the result. The agreement between two runs matches people frame by frame by box
overlap (IoU >= .5): box agreement (how many people both runs tracked), label agreement (the same global
id, exactly and under the best one-to-one mapping of ids, since ids are numbered in creation order), and
the mean distance between the two runs' pitch positions of the same people in metres.
"""
import argparse
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

HEIGHT_BUCKETS = ((0, 30), (30, 40), (40, 60), (60, 90), (90, 10000))
TRACKED = ('confirmed', 'uncertain')


def _valid(p):
    return isinstance(p, list) and len(p) == 2 and all(isinstance(v, (int, float)) and math.isfinite(v) for v in p) and -.1 <= p[0] <= 1.1 and -.1 <= p[1] <= 1.1


def roster_ids(result):
    match = result.get('match', {})
    return {p['id'] for p in match.get('players', []) if p.get('team') in ('A', 'B')} | {p['id'] for p in match.get('goalkeepers', [])}


def metrics(result, name=''):
    frames = result['frames']
    n = max(1, len(frames))
    height = result['video']['height']
    roster = roster_ids(result)
    tracked = candidates = drawn = anonymous = undrawn = 0
    by_height = {b: [0, 0] for b in HEIGHT_BUCKETS}
    disagreements, low_positions = [], 0
    for f in frames:
        for p in f['people']:
            state = p.get('state')
            if state == 'candidate':
                candidates += 1
            if state not in TRACKED:
                continue
            tracked += 1
            h = p['box'][3]*height
            for b in HEIGHT_BUCKETS:
                if b[0] <= h < b[1]:
                    by_height[b][0] += 1
                    by_height[b][1] += bool(p.get('id'))
            placed = _valid(p.get('pitch'))
            if p.get('id') in roster and placed:
                drawn += 1
            elif placed and p.get('team') in ('A', 'B'):
                anonymous += 1
            if not (p.get('id') in roster and placed):
                undrawn += 1
            if placed and (f.get('calibration') or {}).get('confidence', 1) < .2:
                low_positions += 1
        d = (f.get('calibration') or {}).get('disagreement')
        if d is not None:
            disagreements.append(d)
    # Tactical consistency: an identity drawn twice on a frame, jumps between consecutive timeline samples of one
    # identity faster than 12 m/s (by kind transition), the share of frames with a camera footprint.
    duplicates = 0
    for f in frames:
        seen = Counter(p['id'] for p in f['people'] if p.get('id') and _valid(p.get('pitch')))
        duplicates += sum(1 for v in seen.values() if v > 1)
    size = (result.get('tactical') or {}).get('pitchSize') or {}
    length, width = float(size.get('length', 105.0)), float(size.get('width', 68.0))
    block = result.get('timeline') or {}
    # Jumps: at a change of kind (an estimate meeting an observation, or an observation turning into an estimate) the
    # speed between the two consecutive samples; within a run of observations the speed over one second (consecutive
    # observations jitter by their own noise, which says nothing about consistency).
    jumps = Counter()
    if block.get('ids') and block.get('frameTimes'):
        times = block['frameTimes']
        for pid, cols in block['ids'].items():
            run_start = None   # index where the current run of one kind began
            for i in range(1, len(times)):
                k0, k1 = cols['kind'][i-1], cols['kind'][i]
                if k1 != k0:
                    run_start = i
                if k1 == 'U' or k0 == 'U' or cols['x'][i] is None or cols['x'][i-1] is None:
                    continue
                if k1 != k0:
                    dt = times[i]-times[i-1]
                    if dt > 0 and math.hypot((cols['x'][i]-cols['x'][i-1])*length, (cols['y'][i]-cols['y'][i-1])*width)/dt > 12.0:
                        jumps[f'{k0}->{k1}'] += 1
                elif k1 == 'O':
                    j = i-1
                    while j > (run_start or 0) and times[i]-times[j] < 1.0:
                        j -= 1
                    if times[i]-times[j] >= 1.0 and cols['x'][j] is not None:
                        if math.hypot((cols['x'][i]-cols['x'][j])*length, (cols['y'][i]-cols['y'][j])*width)/(times[i]-times[j]) > 12.0:
                            jumps['O->O (1 s)'] += 1
    footprint = sum(1 for f in frames if f.get('view'))
    s = result.get('summary', {})
    ident = s.get('identity') or {}
    reg = (s.get('pitch') or {}).get('registration') or {}
    anchors = reg.get('anchors', [])
    weak = sum(1 for a in anchors if a.get('source') == 'paint' and a.get('score') is not None and a['score'] < .15)
    out = {'run': name, 'frames': len(frames), 'start': result.get('config', {}).get('start'), 'seconds': result.get('config', {}).get('seconds'),
           'perFrame': {'tracked': round(tracked/n, 2), 'candidates': round(candidates/n, 2), 'drawnIdentified': round(drawn/n, 2),
                        'anonymousWithPosition': round(anonymous/n, 2), 'trackedUndrawn': round(undrawn/n, 2)},
           'identifiedShare': s.get('identifiedShare'),
           'identifiedByHeight': {f'{lo}-{hi if hi < 10000 else ""}px': (round(v[1]/v[0], 3) if v[0] else None) for (lo, hi), v in by_height.items()},
           'identity': {k: ident.get(k) for k in ('idSwitches', 'trackSwitchesFoundOnline', 'globalIdFragmentation', 'newPlayerIdentities', 'teamsOverCap',
                                                  'trackletsConnected', 'localTracks', 'fragments', 'closeEncounters', 'playersLostAfterEncounter')},
           'deferredDecisions': s.get('deferredDecisions'), 'globalIdentities': s.get('globalIdentities'),
           'pitch': {'meanConfidence': (s.get('pitch') or {}).get('meanConfidence'), 'levels': (s.get('pitch') or {}).get('levels'),
                     'calibratedShare': reg.get('calibratedShare'), 'confidentShare': reg.get('confidentShare'),
                     'meanObservability': reg.get('meanObservability'), 'mappingLossFrames': reg.get('mappingLossFrames'),
                     'anchors': dict(Counter(a.get('source') for a in anchors)), 'weakPaintAnchors': weak,
                     'positionsBelowConfidence.2': low_positions,
                     'disagreementMedianMetres': round(statistics.median(disagreements), 2) if disagreements else None,
                     'disagreementP90Metres': round(sorted(disagreements)[int(.9*len(disagreements))-1], 2) if len(disagreements) >= 10 else None,
                     'lensCurvature': reg.get('lensCurvature'), 'cameraCuts': s.get('cameraCuts')},
           'ball': {k: (s.get('ball') or {}).get(k) for k in ('trackedShare', 'missingShare', 'unknownShare', 'meanConfidence')},
           'roster': {k: (s.get('roster') or {}).get(k) for k in ('counts', 'overCap', 'statuses', 'anonymousFragments', 'anonymousFrames', 'mergeProposals')},
           'assignment': ident.get('assignment'), 'assignedOffline': ident.get('assignedOffline'), 'framesAssignedOffline': ident.get('framesAssignedOffline'),
           'visibility': (s.get('visibility') or {}).get('shares'), 'timelineKinds': (s.get('timeline') or {}).get('kinds'),
           'detection': s.get('detection'),
           'tactical': {'duplicates': duplicates, 'jumpsOver12mps': dict(jumps), 'footprintShare': round(footprint/n, 3)},
           'timing': {'elapsedSeconds': s.get('elapsedSeconds'), 'processingFPS': s.get('processingFPS'), 'peakGPUMemoryGB': s.get('peakGPUMemoryGB'),
                      **{k: v for k, v in (s.get('timing') or {}).items() if k != 'elapsedSeconds'}}}
    return out


def _iou(a, b):
    ax0, ay0, aw, ah = a
    bx0, by0, bw, bh = b
    ix = max(0.0, min(ax0+aw, bx0+bw)-max(ax0, bx0))
    iy = max(0.0, min(ay0+ah, by0+bh)-max(ay0, by0))
    inter = ix*iy
    union = aw*ah+bw*bh-inter
    return inter/union if union > 0 else 0.0


def agreement(a, b, iou_min=.5):
    """How much result b agrees with result a on the frames they share (matched by time)."""
    size = (a.get('tactical') or {}).get('pitchSize') or {}
    length, width = float(size.get('length', 105.0)), float(size.get('width', 68.0))
    frames_b = {round(f['time'], 2): f for f in b['frames']}
    pairs = Counter()
    matched = only_a = only_b = both_pos = 0
    distances = []
    shared = 0
    for fa in a['frames']:
        fb = frames_b.get(round(fa['time'], 2))
        if fb is None:
            continue
        shared += 1
        pa = [p for p in fa['people'] if p.get('state') != 'rejected']
        pb = [p for p in fb['people'] if p.get('state') != 'rejected']
        candidates = sorted(((_iou(x['box'], y['box']), i, j) for i, x in enumerate(pa) for j, y in enumerate(pb)), reverse=True)
        used_a, used_b = set(), set()
        for score, i, j in candidates:
            if score < iou_min:
                break
            if i in used_a or j in used_b:
                continue
            used_a.add(i)
            used_b.add(j)
            matched += 1
            x, y = pa[i], pb[j]
            pairs[(x.get('id'), y.get('id'))] += 1
            if _valid(x.get('pitch')) and _valid(y.get('pitch')):
                both_pos += 1
                distances.append(math.hypot((x['pitch'][0]-y['pitch'][0])*length, (x['pitch'][1]-y['pitch'][1])*width))
        only_a += len(pa)-len(used_a)
        only_b += len(pb)-len(used_b)
    # The best one-to-one mapping of a's ids onto b's ids, greedily by co-occurrence.
    mapping, taken = {}, set()
    for (ida, idb), count in pairs.most_common():
        if ida is None or idb is None or ida in mapping or idb in taken:
            continue
        mapping[ida] = idb
        taken.add(idb)
    exact = sum(c for (ida, idb), c in pairs.items() if ida == idb)
    mapped = sum(c for (ida, idb), c in pairs.items() if (ida is None and idb is None) or (ida is not None and mapping.get(ida) == idb))
    labelled = sum(c for (ida, idb), c in pairs.items() if ida is not None or idb is not None)
    # Among the people labelled in both runs: the same (mapped) id or not; labels only one run has are
    # counted apart, since a label added offline is not a disagreement about who someone is.
    both = sum(c for (ida, idb), c in pairs.items() if ida is not None and idb is not None)
    both_same = sum(c for (ida, idb), c in pairs.items() if ida is not None and idb is not None and mapping.get(ida) == idb)
    labels_only_a = sum(c for (ida, idb), c in pairs.items() if ida is not None and idb is None)
    labels_only_b = sum(c for (ida, idb), c in pairs.items() if ida is None and idb is not None)
    return {'sharedFrames': shared, 'matchedPeople': matched, 'onlyInA': only_a, 'onlyInB': only_b,
            'boxAgreement': round(2*matched/max(1, 2*matched+only_a+only_b), 4),
            'labelAgreementExact': round(exact/max(1, matched), 4), 'labelAgreementMapped': round(mapped/max(1, matched), 4),
            'labelledPairs': labelled, 'labelledInBoth': both, 'labelAgreementBothLabelled': round(both_same/max(1, both), 4),
            'labelsOnlyInA': labels_only_a, 'labelsOnlyInB': labels_only_b, 'idMappingSize': len(mapping),
            'positionBothShare': round(both_pos/max(1, matched), 4),
            'positionMeanMetres': round(statistics.mean(distances), 2) if distances else None,
            'positionP90Metres': round(sorted(distances)[int(.9*len(distances))-1], 2) if len(distances) >= 10 else None}


def _flatten(d, prefix=''):
    out = {}
    for k, v in d.items():
        key = f'{prefix}{k}'
        if isinstance(v, dict):
            out.update(_flatten(v, key+'.'))
        else:
            out[key] = v
    return out


def table(runs, agree=None):
    """Markdown table of the metrics of one or two runs."""
    flat = [_flatten(m) for m in runs]
    keys = []
    for f in flat:
        for k in f:
            if k not in keys and k != 'run':
                keys.append(k)
    lines = ['| metric | '+' | '.join(m.get('run') or f'run {i+1}' for i, m in enumerate(runs))+' |', '|---|'+'---|'*len(runs)]
    for k in keys:
        lines.append(f'| {k} | '+' | '.join(_fmt(f.get(k)) for f in flat)+' |')
    if agree:
        lines += ['', '| agreement of run 2 with run 1 | |', '|---|---|'] + [f'| {k} | {_fmt(v)} |' for k, v in agree.items()]
    return '\n'.join(lines)


def _fmt(v):
    if v is None:
        return '-'
    if isinstance(v, float):
        return f'{v:.3f}'.rstrip('0').rstrip('.') if abs(v) < 1000 else f'{v:.0f}'
    return str(v)


def load(path):
    path = Path(path)
    if path.is_dir():
        path = path/'result.json'
    return json.loads(path.read_text(encoding='utf-8')), path.parent.name


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('results', nargs='+', help='result.json files or run directories (one or two)')
    ap.add_argument('--json')
    a = ap.parse_args(argv)
    loaded = [load(p) for p in a.results[:2]]
    runs = [metrics(r, name) for r, name in loaded]
    agree = agreement(loaded[0][0], loaded[1][0]) if len(loaded) == 2 else None
    print(table(runs, agree))
    if a.json:
        Path(a.json).write_text(json.dumps({'runs': runs, 'agreement': agree}, indent=1), encoding='utf-8')


if __name__ == '__main__':
    sys.exit(main())
