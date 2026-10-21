"""The coaching report of a saved result on the command line: the observations and their evidence
(soccer/coaching/detectors.py, evidence.py) as JSON, the full report when soccer.coaching.report is available, and
optionally its PDF (soccer.coaching.pdf) with the evidence frames read from the source video.

Nothing here changes the result: it is read, analysed and the output goes where --out / --pdf say (stdout by default),
never next to the result. The range is cleaned and the report fingerprinted exactly as the server does
(soccer/coaching/store.py), so a report made here and one the page made from the same result, settings and options
carry the same fingerprint. The printed table lists every observation per team: classification, severity, measured
seconds, occurrences and their rate, the professional rate, the confidence and the evidence moments, which is how the
engine's smoke runs on the test runs were reported.

    python tools/coaching_report.py RUN/result.json [--team A|B|both] [--t0 S] [--t1 S]
        [--context development|competitive|advanced] [--reference reference.json | --no-reference]
        [--out OUT.json] [--observations-only] [--pdf OUT.pdf --video SOURCE.mp4] [--quiet]
Exit status 0; 2 for bad arguments.
"""
import argparse
import importlib.util
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import soccer.coaching as coaching  # noqa: E402
from soccer.coaching import detectors as det  # noqa: E402


def report_available():
    """soccer.coaching.report exists in this version (checked at run time: the report assembly is a separate part)."""
    return importlib.util.find_spec('soccer.coaching.report') is not None


def table(analysis):
    """The observations as text lines, per team."""
    evs = {e['id']: e for e in analysis['evidence']}
    lines = []
    for o in analysis['observations']:
        base = o.get('baseline') or {}
        pro = (base.get('professionalRatePerMinute') or {}).get('p90')
        ev = ', '.join(f"{evs[e]['time']:.2f}s/{evs[e]['tier']}" for e in o.get('evidence') or () if e in evs)
        lines.append(f"{o['team']} {o['detector']:23s} {o['pattern']:36s} {str(base.get('context')):13s} "
                     f"{o['classification']:12s} {str(o.get('severity') or '-'):23s} measured {o['measuredSeconds']:7.1f}s "
                     f"occ {o['occurrenceCount']:2d} rate {o.get('ratePerMeasuredMinute')} pro p90 {pro} "
                     f"conf {o['confidence']['overall']} {','.join(o.get('insufficientReasons') or [])} [{ev}]")
    return lines


def bounds_of(result):
    """(first, last) frame time of the segment, or None."""
    frames = result.get('frames') or []
    return (float(frames[0]['time']), float(frames[-1]['time'])) if frames else None


def video_of(result, path):
    """{'name', 'key'} of the source video: the result's video name and the key the server files the run under
    (runs/<key16>/<run>/result.json), else the start of the result's sourceSHA256, else None."""
    key = Path(path).resolve().parent.parent.name
    if not re.fullmatch(r'[0-9a-f]{16}', key):
        key = str(result.get('sourceSHA256') or '')[:16] or None
    return {'name': (result.get('video') or {}).get('name'), 'key': key}


def fingerprint_of(data, result, team, args, settings, reference):
    """The report's fingerprint as the server computes it (soccer/coaching/store.py: the bytes of the result, the
    cleaned team-shape settings, the options, the coaching code, the reference and drill files), so a report made
    here and one the page made from the same inputs carry the same id. Another reference file is hashed by its bytes;
    no reference reads 'none'."""
    from soccer import team_shape
    from soccer.coaching import store
    options = store.clean_options(team, args.t0, args.t1, args.context, bounds_of(result))
    ref_sha = None if reference is None else ('none' if reference is False else store.file_sha(reference) or 'missing')
    clean = settings if settings is not None else team_shape.clean_settings({})
    return store.fingerprint(store.fingerprint_inputs(store.sha256_bytes(data), clean, options, reference_sha=ref_sha))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('result', help='result.json of a run')
    ap.add_argument('--team', default='both', choices=('A', 'B', 'both'))
    ap.add_argument('--t0', type=float, help='range start, absolute video seconds (default: the segment start)')
    ap.add_argument('--t1', type=float, help='range end, absolute video seconds (default: the segment end)')
    ap.add_argument('--context', default='competitive', choices=('development', 'competitive', 'advanced'))
    ap.add_argument('--reference', help='a reference.json (default: soccer/coaching/reference.json)')
    ap.add_argument('--no-reference', action='store_true', help='compare with no professional reference')
    ap.add_argument('--settings', help="the video's team_shape.json (direction, lineup, formation labels)")
    ap.add_argument('--out', help='write the JSON here (default: stdout)')
    ap.add_argument('--observations-only', action='store_true', help='skip the report assembly')
    ap.add_argument('--pdf', help='write the PDF of the report here (needs --video and a single --team)')
    ap.add_argument('--video', help='the source video, for the PDF screenshots')
    ap.add_argument('--quiet', action='store_true', help='no table on stderr')
    a = ap.parse_args(argv)
    if a.pdf and (a.team == 'both' or not a.video):
        ap.error('--pdf needs one --team and --video')
    t = time.perf_counter()
    data = Path(a.result).read_bytes()
    result = json.loads(data.decode('utf-8'))
    settings = json.loads(Path(a.settings).read_text(encoding='utf-8')) if a.settings else None
    if settings is not None:
        from soccer import team_shape
        settings = team_shape.clean_settings(settings)
    reference = False if a.no_reference else (a.reference or None)
    team = None if a.team == 'both' else a.team
    # The range as the server takes it (store.clean_options): clamped to the segment, rounded to 0.01 s, the whole
    # segment when it covers it, refused under 5 s; so the same request gives the same analysis and fingerprint.
    from soccer.coaching import store
    try:
        rng = store.clean_options('A', a.t0, a.t1, a.context, bounds_of(result))
    except ValueError as error:
        ap.error(str(error))
    a.t0, a.t1 = rng['t0'], rng['t1']
    analysis = coaching.build_observations(result, team=team, t0=a.t0, t1=a.t1, reference=reference,
                                           settings=settings)
    out = {'engineVersion': analysis['engineVersion'], 'result': str(a.result), 'meta': analysis['meta'],
           'observations': analysis['observations'], 'evidence': analysis['evidence'],
           'reports': {}, 'seconds': None}
    reports_ok = report_available() and not a.observations_only
    if reports_ok:
        from soccer.coaching import report
        for tm in (('A', 'B') if team is None else (team,)):
            meta = dict(analysis['meta'], run=Path(a.result).parent.name, video=video_of(result, a.result),
                        fingerprint=fingerprint_of(data, result, tm, a, settings, reference))
            out['reports'][tm] = report.build_report(analysis['observations'], analysis['evidence'], meta,
                                                     context=a.context, team=tm)
    elif not a.observations_only:
        out['reports'] = None
        out['note'] = 'soccer.coaching.report is not available in this version: observations and evidence only'
    out['seconds'] = round(time.perf_counter()-t, 2)
    text = json.dumps(out, allow_nan=False, ensure_ascii=True, indent=1)
    if a.out:
        Path(a.out).write_text(text, encoding='utf-8')
    else:
        sys.stdout.write(text+'\n')
    if a.pdf:
        if not reports_ok or team not in out['reports']:
            ap.error('--pdf needs soccer.coaching.report')
        from soccer.coaching import frames, pdf
        provider = frames.FrameProvider(a.video, times={e['frame']: e['time'] for e in analysis['evidence']},
                                        fps=analysis['meta'].get('fps'))
        pdf.write_report_pdf(out['reports'][team], a.pdf, provider, {'labels': store.labels_of(result)})
    if not a.quiet:
        sys.stderr.write('\n'.join(table(analysis))+'\n')
        sys.stderr.write(f"{len(analysis['observations'])} observations, {len(analysis['evidence'])} evidence moments, "
                         f"{out['seconds']} s; reference {analysis['meta'].get('reference') or 'none'}; "
                         f"rules: {det.MIN_MEASURED_SECONDS:g} s measured, strengths from {det.STRENGTH_MIN_SECONDS:g} s\n")
    return 0


if __name__ == '__main__':
    sys.exit(main())
