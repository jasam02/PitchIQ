"""Exact comparison of two results of the same segment: which frames differ in people (track, identity,
label, team, role, state, box), in pitch positions, in the ball, in the per-frame pitch mapping, and how
far the numbers drift. python exact_diff.py A/result.json B/result.json"""
import json
import math
import sys


def load(path):
    with open(path, encoding='utf-8') as handle:
        return json.load(handle)


def flat(value, prefix=''):
    """All numbers of a nested structure as {path: number}."""
    out = {}
    if isinstance(value, dict):
        for k, v in value.items():
            out.update(flat(v, f'{prefix}.{k}'))
    elif isinstance(value, (list, tuple)):
        for i, v in enumerate(value):
            out.update(flat(v, f'{prefix}[{i}]'))
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        out[prefix] = float(value)
    else:
        out[prefix] = value
    return out


def person_key(p):
    return (p.get('track'), p.get('id'), p.get('display'), p.get('label'), p.get('team'), p.get('role'), p.get('state'), tuple(p.get('box', ())))


a, b = load(sys.argv[1]), load(sys.argv[2])
fa, fb = a['frames'], b['frames']
print('frames', len(fa), len(fb))
n = min(len(fa), len(fb))
people_diff = pitch_diff = ball_diff = mapping_diff = 0
max_pitch = max_map = max_box = 0.0
examples = []
for i in range(n):
    A, B = fa[i], fb[i]
    ka, kb = {person_key(p) for p in A['people']}, {person_key(p) for p in B['people']}
    if ka != kb:
        people_diff += 1
        if len(examples) < 5:
            examples.append((A['time'], sorted(ka-kb)[:2], sorted(kb-ka)[:2]))
    # pitch positions of the same track
    pa = {p['track']: p.get('pitch') for p in A['people']}
    pb = {p['track']: p.get('pitch') for p in B['people']}
    for t in pa.keys() & pb.keys():
        x, y = pa[t], pb[t]
        if (x is None) != (y is None):
            pitch_diff += 1
        elif x is not None:
            d = math.hypot(x[0]-y[0], x[1]-y[1])
            max_pitch = max(max_pitch, d)
            if d > 1e-6:
                pitch_diff += 1
    if json.dumps(A.get('ball'), sort_keys=True) != json.dumps(B.get('ball'), sort_keys=True):
        ball_diff += 1
    ma, mb = flat(A.get('mapping')), flat(B.get('mapping'))
    ca, cb = flat(A.get('calibration')), flat(B.get('calibration'))
    for x, y in ((ma, mb), (ca, cb)):
        if x.keys() != y.keys():
            mapping_diff += 1
            continue
        for k in x:
            if isinstance(x[k], float) and isinstance(y[k], float):
                max_map = max(max_map, abs(x[k]-y[k]))
            elif x[k] != y[k]:
                mapping_diff += 1
                break
print('frames with different people (track/id/label/team/role/state/box):', people_diff)
print('pitch positions differing by > 1e-6 (or presence):', pitch_diff, ' max distance m:', max_pitch)
print('frames with a different ball:', ball_diff)
print('frames with different mapping/calibration structure:', mapping_diff, ' max numeric drift in them:', max_map)
for e in examples:
    print('  e.g. at', e[0], 'only in A:', e[1], 'only in B:', e[2])
for key in ('match', 'roster', 'tracks', 'teams', 'labels'):
    same = json.dumps(a.get(key), sort_keys=True) == json.dumps(b.get(key), sort_keys=True)
    print(f'{key}: {"identical" if same else "DIFFERENT"}')
ea, eb = [(e['time'], e['kind']) for e in a.get('events', [])], [(e['time'], e['kind']) for e in b.get('events', [])]
print('events (time, kind):', 'identical' if ea == eb else f'DIFFERENT ({len(ea)} vs {len(eb)})')
sa, sb = a['summary'], b['summary']
for k in ('identifiedShare', 'newIdentities', 'reidentifications', 'swapCorrections', 'localTrackSegments'):
    print(f'summary.{k}: {sa.get(k)} vs {sb.get(k)}')
print('ball tracked share:', sa.get('ball', {}).get('trackedShare'), 'vs', sb.get('ball', {}).get('trackedShare'))
print('pitch mean confidence:', sa.get('pitch', {}).get('meanConfidence'), 'vs', sb.get('pitch', {}).get('meanConfidence'))
ta, tb = sa.get('timing', {}), sb.get('timing', {})
print('timing online ms/frame:', ta.get('onlineMsPerFrame'), 'vs', tb.get('onlineMsPerFrame'), '| tracking:', ta.get('trackingMsPerFrame'), 'vs', tb.get('trackingMsPerFrame'),
      '| offline:', ta.get('offlineSeconds'), 'vs', tb.get('offlineSeconds'), '| elapsed:', ta.get('elapsedSeconds'), 'vs', tb.get('elapsedSeconds'))
