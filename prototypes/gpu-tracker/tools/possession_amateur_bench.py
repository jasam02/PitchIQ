"""Scores the possession dispatch (soccer/possession_amateur.py: possession-4's record where possession-4 judges the
footage, else its image-based estimate) on hand-labelled PitchIQ runs, with the hand labels' metrics of
tools/possession_bench.py (points every 0.5 s: accuracy and coverage of the team points, dead points called dead and
engine-dead points labelled dead; changes matched on wonAt within 1 s and 2 s with the same new team), plus the
estimate's abstentions: the unknown team points by the reason of their span, and quality.abstained. Every number of
soccer/possession_amateur.py's docstring on a labelled run comes from this script (the AM and VEO labels are DEV data).

    python -I -B tools/possession_amateur_bench.py --labels tools/annotations/possession --run AM=RUN/result.json
        [--ends AM=A,B] [--ablations] [--force] [--json out.json]

--ends NAME=LEFT,RIGHT sets the run's tactical.goalEnds before scoring (the labels' ends: amateur 3 as stored has them
reversed); --force scores the estimate even where possession-4 judges the run (settings engine 'amateur', as on the
broadcast runs); --ablations adds the estimate with each rule switched off or varied (ABLATIONS).
"""
import argparse
import collections
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'tools'))
from soccer import possession, possession_amateur as amateur  # noqa: E402
import possession_bench as pb  # noqa: E402

ABLATIONS = {
    'kit readings required': {'kit_none': 'reject'},
    'first-pass rule off': {'fp_agree': 1e9},
    'no-carry off (carried like an unseen ball)': {'nocarry': 1e9},
    'bridges up to 1.2 s trusted': {'max_bridge': 1.2},
    'reach 1.5-3.0 m': {'reach_r0': 1.5, 'reach_r1': 3.0},
    'reach 2.0-4.0 m': {'reach_r0': 2.0, 'reach_r1': 4.0},
    'depth factor capped at 4': {'max_depth': 4.0},
    'depth factor capped at 2 (possession-4 rows)': {'max_depth': 2.0},
}


def unknown_by_reason(rec, F, lab):
    """The labelled team points the record leaves unknown, by the reason ('why') of their timeline run."""
    out = collections.Counter()
    runs = rec.get('timeline') or []
    for p in lab['points']:
        if p['team'] not in ('A', 'B'):
            continue
        run = next((r for r in runs if r['t0']-1e-6 <= p['t'] <= r['t1']+.5/F['fps']), None)
        if run is not None and run['team'] is None and run['state'] != 'dead':
            out[run.get('why') or 'unknown'] += 1
    return dict(out)


def score(rec, F, lab):
    return {'version': rec['version'], 'available': rec['available'], 'reason': rec.get('reason'),
            'coverageLive': rec['quality'].get('coverage'), 'points': pb.label_points(rec, F, lab),
            'changes1s': pb.label_changes(rec, lab, 1.0), 'changes2s': pb.label_changes(rec, lab, 2.0),
            'deadSpells': rec.get('deadSpells'), 'abstained': rec['quality'].get('abstained'),
            'unknownByReason': unknown_by_reason(rec, F, lab)}


def line(name, s):
    p, c1, c2 = s['points'], s['changes1s'], s['changes2s']
    return (f"{name}: {s['version']} available {s['available']} | accuracy {p['accuracy']} coverage {p['coverage']} "
            f"({p['right']}/{p['named']} of {p['teamPoints']}) | dead {p['deadPoints']}, {p['engineDeadPoints']} | changes "
            f"{c2['engine']} reported, {c2['matched']}/{c2['truth']} within 2 s, {c1['matched']} within 1 s | wrong "
            f"{[w[0] for w in p['wrong']]} | unknown by reason {json.dumps(s['unknownByReason'])}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--labels', required=True, help='folder of the PitchIQ hand labels (tools/annotations/possession)')
    ap.add_argument('--run', action='append', default=[], help='NAME=path/to/result.json, scored against --labels/NAME.json')
    ap.add_argument('--ends', action='append', default=[], help='NAME=LEFT,RIGHT: the tactical.goalEnds to score the run with')
    ap.add_argument('--force', action='store_true', help='the estimate even where possession-4 judges the run')
    ap.add_argument('--ablations', action='store_true', help='also the estimate with each rule off or varied')
    ap.add_argument('--json', help='write every number here')
    a = ap.parse_args(argv)
    ends = {}
    for spec in a.ends:
        name, pair = spec.split('=', 1)
        left, right = pair.split(',')
        ends[name] = {'left': left or None, 'right': right or None}
    report = {'version': amateur.VERSION, 'settings': amateur.DEFAULTS, 'runs': {}}
    for spec in a.run:
        name, path = spec.split('=', 1)
        lab = json.loads((Path(a.labels)/f'{name}.json').read_text(encoding='ascii'))
        result = json.loads(Path(path).read_text())
        if name in ends:
            result['tactical'] = dict(result.get('tactical') or {}, goalEnds=ends[name])
        t0 = time.time()
        F = possession.features(result)
        rec = amateur.compute(result, {'engine': 'amateur'} if a.force else None)
        out = {'seconds': round(time.time()-t0, 2), 'goalEnds': (result.get('tactical') or {}).get('goalEnds'),
               'gate': amateur.dispatch_reason(F), 'dispatch': score(rec, F, lab)}
        print(line(f'{name} dispatch', out['dispatch']), f"| {out['seconds']} s", flush=True)
        if a.ablations:
            out['ablations'] = {}
            for label, settings in ABLATIONS.items():
                s = score(amateur.estimate(result, settings, F4=F), F, lab)
                out['ablations'][label] = s
                print(line(f'{name} {label}', s), flush=True)
        report['runs'][name] = out
    if a.json:
        Path(a.json).write_text(json.dumps(report, indent=1, default=float), encoding='ascii')


if __name__ == '__main__':
    main()
