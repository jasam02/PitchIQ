"""The coaching report of a saved result on the command line: the v2 adaptive report (DESIGN v2) of one team or both,
built through the same three cache tiers and keyed by the same fingerprints as the server (soccer/coaching/store.py:
tier_keys, measure, ensure-style role and interpretation tiers), and optionally its PDF (soccer.coaching.pdf) with the
key-moment frames read from the source video. With --context the deprecated v1 report is made instead, as before.

Why one code path with the server. A report from the command line and one from the page must be the same report for
the same inputs, or a coach comparing them is misled: the result bytes, the team-shape settings, the video's coaching
settings and this run's coach-marked spans are hashed by store.tier_keys here exactly as there (measured on the
synthetic short_segment scene: the three fingerprints equal the server's and the reports are identical apart from the
serve-time fields 'generated', 'progression', 'teamName' and the video name the upload carries; tested).

Nothing here changes the result or writes next to it: the output goes where --out / --pdf say (stdout by default); no
cache file, progress entry or setting is written (the tiers are built in memory). The printed table lists, per team,
the profile, every check's status with its reason, and every finding with its verdict, key, counts and key moments.

    python tools/coaching_report.py RUN/result.json [--team A|B|both] [--t0 S] [--t1 S]
        [--coaching coaching_settings.json] [--age U12] [--level 1-7] [--format 4v4|5v5|7v7|9v9|11v11]
        [--keeper | --no-keeper] [--field 55x36] [--style possession|counter|high_press|compact|unknown]
        [--federation auto|US|England|Canada] [--spans coach_spans.json [--run NAME]] [--settings team_shape.json]
        [--out OUT.json] [--pdf OUT.pdf --video SOURCE.mp4] [--quiet]
    v1 (deprecated): add --context development|competitive|advanced [--reference reference.json | --no-reference]
        [--observations-only] (any of the last three alone also selects v1, at the competitive level)
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

FORMATS = {'4v4': 4, '5v5': 5, '7v7': 7, '9v9': 9, '11v11': 11}


def report_available():
    """soccer.coaching.report exists in this version (checked at run time: the report assembly is a separate part)."""
    return importlib.util.find_spec('soccer.coaching.report') is not None


def table(analysis):
    """The v1 observations as text lines, per team."""
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


def table_v2(report):
    """A v2 report as text lines: the profile, each check's status, each finding."""
    p = report['profile']
    out = [f"{report['team']} profile: {p['ageGroup']} level {p['level']} ({p['levelName']}), {p['format']['name']}, "
           f"style {p['style']}; range {report['range']['t0']:.2f}-{report['range']['t1']:.2f} s"]
    for d in report['overview']['detectors']:
        out.append(f"{report['team']}   {d['detector']:22s} {d['status']:13s} {str(d.get('reason') or '-'):22s} "
                   f"eligible {d['eligibleSeconds']:7.1f}s opportunities {d['opportunities']}")
    for section in ('strengths', 'needsWork', 'reviews', 'described'):
        for f in report[section]:
            a = f['aggregate']
            times = ', '.join(f"{t['t']:.2f}s" for t in f['timestamps'])
            out.append(f"{report['team']}   {f['verdict']:9s} {f['key']:34s} opp {a['opportunities']:3d} "
                       f"demo {a['demonstrated']:3d} not {a['notDemonstrated']:3d} unclear {a['unclear']:3d} [{times}]")
    return out


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
    """The v1 report's fingerprint as the server computes it (store.fingerprint_inputs: the bytes of the result, the
    cleaned team-shape settings, the options, the coaching code, the reference and drill files)."""
    from soccer import team_shape
    from soccer.coaching import store
    options = store.clean_options(team, args.t0, args.t1, args.context, bounds_of(result))
    ref_sha = None if reference is None else ('none' if reference is False else store.file_sha(reference) or 'missing')
    clean = settings if settings is not None else team_shape.clean_settings({})
    return store.fingerprint(store.fingerprint_inputs(store.sha256_bytes(data), clean, options, reference_sha=ref_sha))


def coaching_settings(a, teams, error):
    """The coaching settings the report is made with: --coaching (a coaching_settings.json, cleaned as the server cleans
    it), then the profile flags applied to the chosen team(s) (refused, like the server, when the cleaning would drop
    anything)."""
    from soccer.coaching import profiles, store
    raw = {}
    if a.coaching:
        try:
            raw = json.loads(Path(a.coaching).read_text(encoding='utf-8'))
        except (OSError, ValueError) as e:
            error(f'--coaching: {e}')
    problems = []
    settings = profiles.clean_settings(raw, problems=problems)
    if problems:
        error('--coaching: ' + '; '.join(problems[:4]))
    change = {}
    if a.age:
        change['ageGroup'] = a.age
    if a.level is not None:
        change['playingLevel'] = a.level
    if a.format:
        players = FORMATS[a.format]
        keeper = a.keeper if a.keeper is not None else profiles.load()['formats'][str(players)]['goalkeeperDefault']
        change['format'] = {'players': players, 'goalkeeper': bool(keeper)}
    if a.field:
        m = re.fullmatch(r'(\d+(?:\.\d+)?)x(\d+(?:\.\d+)?)', a.field)
        if not m:
            error('--field is LENGTHxWIDTH in metres, e.g. 55x36')
        change['field'] = {'length': float(m.group(1)), 'width': float(m.group(2))}
    if a.style:
        change['style'] = a.style
    if a.federation:
        change['federation'] = a.federation
    if not change:
        return settings
    for team in teams:
        block = dict(settings['teams'].get(team) or {}, **change)
        if 'format' in change and (settings['teams'].get(team) or {}).get('format') != change['format']:
            block['formation'] = {'intended': None, 'periods': []}     # a formation belongs to its format
            block['roles'] = {}
        problems = []
        clean = profiles.clean_team(block, now=store._iso(), problems=problems, path=f'teams.{team}')
        if problems:
            error('; '.join(problems[:4]))
        settings['teams'][team] = clean
    return settings


def coach_spans(a, result_path, error):
    """The run's coach spans: --spans is a coach-spans record (schema 'coach_spans') or the server's per-video
    coach_spans.json ({version, runs: {run: {spans}}}; --run names the run, default the result's folder)."""
    from soccer.coaching import schema as S
    if not a.spans:
        return None
    try:
        doc = json.loads(Path(a.spans).read_text(encoding='utf-8'))
    except (OSError, ValueError) as e:
        error(f'--spans: {e}')
    if isinstance(doc, dict) and isinstance(doc.get('runs'), dict):
        run = a.run or Path(result_path).resolve().parent.name
        doc = (doc['runs'].get(run) or {}).get('spans')
        if doc is None:
            return None
    problems = S.validate('coach_spans', doc)
    if problems:
        error('--spans: ' + '; '.join(problems[:4]))
    return doc if doc['spans'] else None


def main_v2(a, ap):
    from soccer import team_shape
    from soccer.coaching import store
    t = time.perf_counter()
    teams = ('A', 'B') if a.team == 'both' else (a.team,)
    data = Path(a.result).read_bytes()
    result = json.loads(data.decode('utf-8'))
    shape = team_shape.clean_settings(json.loads(Path(a.settings).read_text(encoding='utf-8')) if a.settings else {})
    settings = coaching_settings(a, teams, ap.error)
    frames = result.get('frames') or []
    segment = store.segment_of(frames)
    spans = coach_spans(a, a.result, ap.error)
    if spans is not None:
        from soccer.coaching import situations
        kept, _ = situations.clean_coach_spans(spans, segment['t0'], segment['t1'])
        spans = dict(store.empty_spans(), spans=kept) if kept else None     # as the server stores them (save_spans)
    try:
        options = {team: store.clean_range(team, a.t0, a.t1, bounds_of(result)) for team in teams}
    except ValueError as error:
        ap.error(str(error))
    sha = store.sha256_bytes(data)
    keys = {team: store.tier_keys(sha, shape, settings, team, spans, segment, options[team]['t0'], options[team]['t1'])
            for team in teams}
    obs = store.measure(result, result_sha=sha, shape_settings=shape, settings=settings, spans=spans,
                        team=None if len(teams) == 2 else teams[0])
    obs = obs if len(teams) == 2 else {teams[0]: obs}
    run = a.run or Path(a.result).resolve().parent.name
    meta = {'run': run if store.RUN_NAME.fullmatch(run) else None, 'video': video_of(result, a.result)}
    out = {'engine': 2, 'result': str(a.result), 'settings': settings, 'coachSpans': spans, 'reports': {},
           'fingerprints': {}, 'seconds': None}
    for team in teams:
        roles = store.build_role_tier(obs[team], settings, team)
        out['reports'][team] = store.build_interpretation(obs[team], roles, settings, keys[team], meta)
        out['fingerprints'][team] = {'measurement': keys[team]['mfp'], 'roles': keys[team]['rfp'],
                                     'interpretation': keys[team]['ifp']}
    out['seconds'] = round(time.perf_counter()-t, 2)
    text = json.dumps(out, allow_nan=False, ensure_ascii=True, indent=1)
    if a.out:
        Path(a.out).write_text(text, encoding='utf-8')
    else:
        sys.stdout.write(text+'\n')
    if a.pdf:
        from soccer.coaching import frames as fr, pdf
        report = out['reports'][teams[0]]
        provider = fr.FrameProvider(a.video, times={e['frame']: e['time'] for e in report['keyMoments']},
                                    fps=(result.get('video') or {}).get('fps'))
        pdf.write_report_pdf(report, a.pdf, provider, {'labels': store.labels_of(result)})
    if not a.quiet:
        for team in teams:
            sys.stderr.write('\n'.join(table_v2(out['reports'][team]))+'\n')
        sys.stderr.write(f"{out['seconds']} s; coach spans {'yes' if spans else 'no'}\n")
    return 0


def main_v1(a, ap):
    from soccer.coaching import detectors as det, store
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


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('result', help='result.json of a run')
    ap.add_argument('--team', default='both', choices=('A', 'B', 'both'))
    ap.add_argument('--t0', type=float, help='range start, absolute video seconds (default: the segment start)')
    ap.add_argument('--t1', type=float, help='range end, absolute video seconds (default: the segment end)')
    ap.add_argument('--settings', help="the video's team_shape.json (direction, lineup, formation labels)")
    ap.add_argument('--out', help='write the JSON here (default: stdout)')
    ap.add_argument('--pdf', help='write the PDF of the report here (needs --video and a single --team)')
    ap.add_argument('--video', help='the source video, for the PDF screenshots')
    ap.add_argument('--quiet', action='store_true', help='no table on stderr')
    v2 = ap.add_argument_group('v2 (adaptive coaching; the default)')
    v2.add_argument('--coaching', help="the video's coaching_settings.json (default: none saved)")
    v2.add_argument('--age', help='age group of the chosen team(s): U6..U19, U21, U23, Adult')
    v2.add_argument('--level', type=int, choices=range(1, 8), help='playing level 1-7')
    v2.add_argument('--format', choices=tuple(FORMATS), help='match format')
    v2.add_argument('--keeper', dest='keeper', action='store_true', default=None, help='with a goalkeeper (4v4/5v5)')
    v2.add_argument('--no-keeper', dest='keeper', action='store_false', help='without a goalkeeper (4v4/5v5)')
    v2.add_argument('--field', help='declared field size LENGTHxWIDTH in metres, e.g. 55x36')
    v2.add_argument('--style', choices=('possession', 'counter', 'high_press', 'compact', 'unknown'))
    v2.add_argument('--federation', choices=('auto', 'US', 'England', 'Canada'))
    v2.add_argument('--spans', help="coach-marked possession: a coach spans record or the video's coach_spans.json")
    v2.add_argument('--run', help='the run name (the coach spans of this run; default: the result\'s folder)')
    v1 = ap.add_argument_group('v1 (deprecated report against one professional match)')
    v1.add_argument('--context', choices=('development', 'competitive', 'advanced'), help='make the v1 report')
    v1.add_argument('--reference', help='a reference.json (default: soccer/coaching/reference.json)')
    v1.add_argument('--no-reference', action='store_true', help='compare with no professional reference')
    v1.add_argument('--observations-only', action='store_true', help='skip the report assembly')
    a = ap.parse_args(argv)
    if a.pdf and (a.team == 'both' or not a.video):
        ap.error('--pdf needs one --team and --video')
    if a.context is None and (a.reference or a.no_reference or a.observations_only):
        a.context = 'competitive'       # a v1-only option asks for the v1 report (its default level, as before)
    if a.context is None:
        return main_v2(a, ap)
    if any(x is not None for x in (a.coaching, a.age, a.level, a.format, a.keeper, a.field, a.style, a.federation,
                                   a.spans)):
        ap.error('the coaching profile options belong to the v2 report: leave out --context')
    return main_v1(a, ap)


if __name__ == '__main__':
    sys.exit(main())
