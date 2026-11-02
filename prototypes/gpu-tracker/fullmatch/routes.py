"""Full Match Mode's routes in the review server (registered by server.main; nothing here runs on a Segment run).

    GET  /api/match                  the open video's matches, the one being processed, the resumable one, the computer
    POST /api/match/plan             what a full match would be: periods, segments, workers, estimated time, disk, calibration
    POST /api/match/start            plan and start it (the processing settings and keyframes are the dock's, validated like a run)
    POST /api/match/pause            no new segments; the running ones finish and are kept
    POST /api/match/stop             cancel the running segments (they go back to pending); finished ones are kept
    POST /api/match/resume           continue a paused, stopped, failed or interrupted match: only what is missing
    GET  /api/match/status           progress, ETA, workers, resources of the match being processed (or the last one)
    GET  /api/match/intel            the match's state timeline, events, masks and halftime (fullmatch/intel.py)
    POST /api/match/intel/build      make them (the match data once, about a minute; then seconds), on a thread
    POST /api/match/halftime         {match, mode, firstHalfEnd, secondHalfStart, teamAFirstHalf}: the coach's halftime
    POST /api/match/halftime/reset   back to the plan's break and the inferred direction
    POST /api/match/event            {match, kind: confirm|retype|move|remove|add|state, ...}: an event correction
    POST /api/match/event/remove     {match, cid}: undo one
    POST /api/possession2/build      {match}: the match's possession v2 record (fullmatch/possession_match.py), on a
                                     thread: the match state first when it is missing, then the frame table (about a
                                     minute once), then the decode (seconds); GET /api/possession2?match= is server.py's

One match is processed at a time, on the whole computer: while it runs the server refuses Segment runs (they would share
the GPU) and frees its own detectors and embedder (the workers load theirs; the server reloads its pair afterwards).
A match the server did not finish (closed, crashed) is found at start and shown as resumable; its segments that verify
on disk are kept.
"""
import json
import math
import os
import threading
import time
from pathlib import Path

from flask import jsonify, request

from . import manifest as mf
from . import plan as planning
from . import scheduler as sch
from . import system

THROUGHPUT_FILE = '_match_throughput.json'   # runs/_match_throughput.json: this computer's measured aggregate fps per worker count
COACHING_RUN = 'match-'                      # the run name of a match's coaching report (fullmatch.coaching.run_name)
RESTATE_KINDS = ('merge', 'split', 'exclude', 'include', 'team', 'keeper', 'confirm', 'substitution')
ROLL_SECONDS = mf.ROLL_SECONDS


def measured_throughput(runs):
    try:
        return json.loads((Path(runs)/THROUGHPUT_FILE).read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return {}


def remember_throughput(runs, workers, fps):
    data = measured_throughput(runs)
    data[str(int(workers))] = round(float(fps), 2)
    try:
        mf.write_json(Path(runs)/THROUGHPUT_FILE, data)
    except OSError:
        pass


def calibration_status(settings, segments, fps, discovery=None):
    """Which segments have a calibration keyframe within reach (inside their processed frames or ROLL_SECONDS of them):
    those are anchored by it; the others rely on the automatic calibration, which a high, wide camera can get wrong.
    discovery: {'runs', 'seconds', ...} when the calibration discovery will look for keyframes first (calibration.py)."""
    times = sorted(float(c['time']) for c in settings.get('calibrations') or [])
    anchored = []
    for s in segments:
        t0, t1 = s.proc0/fps-ROLL_SECONDS, s.proc1/fps+ROLL_SECONDS
        if any(t0 <= t <= t1 for t in times):
            anchored.append(s.index)
    n = len(segments)
    out = {'keyframes': len(times), 'segments': n, 'anchored': len(anchored), 'anchoredSegments': anchored,
           'automatic': n-len(anchored), 'discovery': discovery,
           'note': ('Every segment has a calibration keyframe within reach.' if len(anchored) == n else
                    f'{n-len(anchored)} of {n} segments have no calibration keyframe within 15 s and rely on the automatic '
                    'calibration. On a high, wide camera that can place players metres off; add keyframes in the stretches '
                    'that matter, or check the calibration report after the run.')}
    if discovery and discovery.get('runs'):
        out['note'] = (f'{n-len(anchored)} of {n} segments have no calibration keyframe within 15 s: PitchIQ first looks for '
                       f"their calibration automatically (about {max(1, round(discovery['seconds']/60))} min: frames across the "
                       'match, kept only where they agree with one camera position). Segments it cannot anchor safely keep the '
                       'per-segment calibration and are listed when it ends.')
    return out


def parse_periods(body, meta):
    duration = float(meta['duration'])
    ranges = body.get('ranges')
    if ranges:
        return planning.periods(duration, ranges=[(float(a), float(b)) for a, b in ranges])
    halftime = body.get('halftime')
    return planning.periods(duration, body.get('start'), body.get('end'),
                            (float(halftime[0]), float(halftime[1])) if halftime else None)


def register(app, context):
    """context: 'video' () -> the open server.Video or None; 'single_busy' () -> a Segment run is processing;
    'validate' (payload, meta) -> clean run settings (server.validate); 'release_models', 'preload_models' (); 'runs'
    the runs folder; 'seeds' (video) -> the corrections' remembered people (or None). Sets context['match_busy']."""
    lock = threading.RLock()
    active = {'scheduler': None, 'manifest': None}
    runs = Path(context['runs'])

    def busy():
        s = active['scheduler']
        return s is not None and s.mode in ('running', 'pausing', 'stopping', 'assembling')
    context['match_busy'] = busy

    # A match the server left running (closed or crashed mid-run) is interrupted: resumable, not running.
    for path in runs.glob('*/matches/*/manifest.json'):
        try:
            m = mf.Manifest.load(path)
            if m.data['state'] in ('running', 'assembling'):
                m.set_state('stopped', reason='interrupted: the app was closed while it ran')
        except (OSError, ValueError, KeyError, TypeError):
            continue

    def clean_settings(payload, meta):
        """The dock's processing settings and keyframes, checked exactly as a Segment run's (start/length are the plan's)."""
        body = dict(payload.get('settings') or {})
        body.update(start=0, seconds=min(60.0, float(meta['duration'])))
        settings = context['validate'](body, meta)
        for key in ('start', 'seconds', 'continueIdentities'):
            settings.pop(key, None)
        return settings

    def workers_for(choice, snap):
        if choice in (None, 'auto', 'AUTO'):
            count, why = system.auto_workers(snap)
            return {'mode': 'auto', 'count': count, 'limits': why['limits'], 'binding': why['binding'], 'note': None}
        count, note = system.allowed(int(choice), snap)
        return {'mode': str(int(choice)), 'count': count, 'note': note,
                'experimental': int(choice) > system.AUTO_MAX}

    def describe(body):
        v = context['video']()
        if v is None:
            raise ValueError('Upload or open a video first.')
        meta = v.meta
        periods = parse_periods(body, meta)
        segment_s = float(body.get('segmentSeconds', planning.SEGMENT_S))
        pad_s = float(body.get('padSeconds', planning.PAD_S))
        if not 30 <= segment_s <= planning.SEGMENT_S or not 0 <= pad_s <= 15:
            raise ValueError('Segments must be 30-180 s and the context 0-15 s.')
        segments = planning.plan(meta['frames'], meta['fps'], periods, segment_s, pad_s)
        settings = clean_settings(body, meta)
        snap = system.snapshot()
        workers = workers_for(body.get('workers'), snap)
        frames = planning.processed_frames(segments)
        measured = measured_throughput(runs)
        anchored = calibration_status(settings, segments, float(meta['fps']))
        mode = 'off' if str(body.get('discovery', 'auto')).lower() == 'off' else 'auto'
        streams = min(sch.DISCOVERY_STREAMS, max(1, (os.cpu_count() or 2)//2))
        runs_discovery = mode == 'auto' and anchored['automatic'] > 0
        discovery = {'mode': mode, 'runs': runs_discovery, 'streams': streams,
                     'seconds': round(system.discovery_seconds(frames, float(meta['fps']), streams)) if runs_discovery else 0}
        low, likely, high = system.estimate_seconds(frames, workers['count'], measured,
                                                    sch.ASSEMBLY_GUESS_S[0]+sch.ASSEMBLY_GUESS_S[1]*len(segments)+discovery['seconds'])
        needed, free = system.disk_needed(frames, len(segments)), system.disk_free(v.directory)
        owned = sum(s.own1-s.own0 for s in segments)
        plan_out = {'periods': [{'start': a, 'end': b, 'name': mf.period_name(i, len(periods))} for i, (a, b) in enumerate(periods)],
                    'segments': len(segments), 'segmentSeconds': segment_s, 'padSeconds': pad_s,
                    'matchSeconds': round(owned/meta['fps'], 1), 'processedSeconds': round(frames/meta['fps'], 1),
                    'workers': workers, 'system': snap,
                    'estimate': {'low': round(low), 'likely': round(likely), 'high': round(high),
                                 'basis': 'this computer' if str(workers['count']) in measured else 'benchmark',
                                 'fps': round(system.throughput(workers['count'], measured), 2)},
                    'disk': {'needed': needed, 'free': free, 'ok': free > 1.2*needed,
                             'neededText': system.human_bytes(needed), 'freeText': system.human_bytes(free)},
                    'calibration': calibration_status(settings, segments, float(meta['fps']), discovery),
                    'video': {k: meta[k] for k in ('name', 'duration', 'fps', 'frames', 'width', 'height')}}
        return v, periods, segments, settings, plan_out

    def matches_of(v):
        out = []
        for m in mf.find(v.directory):
            counts = m.counts()
            out.append({'matchId': m.data['matchId'], 'state': m.data['state'], 'created': m.data['created'],
                        'updated': m.data['updated'], 'reason': m.data.get('reason'), 'segments': counts,
                        'total': len(m.data['segments']), 'periods': m.data['periods'],
                        'compatible': m.data['source'].get('key') == v.key, 'runtime': m.data.get('runtime')})
        return out

    def current_status():
        with lock:
            s = active['scheduler']
            if s is not None:
                return s.status()
        return None

    def after_match(manifest):
        runtime = manifest.data.get('runtime') or {}
        if runtime.get('aggregateFPS') and manifest.data['state'] in ('complete', 'assembling'):
            remember_throughput(runs, runtime.get('workers', 1), runtime['aggregateFPS'])

    def launch(manifest, workers, threads, resume):
        v = context['video']()
        seeds = context['seeds'](v) if v is not None and context.get('seeds') else None

        def finished():
            after_match(manifest)
            context['preload_models']()
        assemble = None
        try:
            from . import assemble as assembly
            assemble = assembly.assemble
        except ImportError:
            assemble = None
        s = sch.MatchScheduler(manifest, workers, threads=threads, assemble=assemble, before_start=context['release_models'],
                               after_stop=finished, seeds=seeds, measured=measured_throughput(runs))
        active['scheduler'], active['manifest'] = s, manifest
        s.start(resume=resume)
        return s

    def coaching_match_id():
        """The match a coaching request is for: ?match= (the page adds it in match playback), else a run named
        'match-<id>' (the match report's run: its evidence frames and its PDF are asked for by run)."""
        if request.args.get('match'):
            return request.args['match']
        run = request.args.get('run') or ''
        if not run and request.method == 'POST':
            try:
                body = request.get_json(silent=True)
            except RecursionError:
                body = None
            run = body.get('run') if isinstance(body, dict) and isinstance(body.get('run'), str) else ''
        return run[len(COACHING_RUN):] if run.startswith(COACHING_RUN) else None

    def coaching_range(values):
        team = values.get('team', 'A')
        if team not in ('A', 'B'):
            raise ValueError('team must be A or B')
        out = []
        for key in ('t0', 't1'):
            x = values.get(key)
            if x in (None, ''):
                out.append(None)
                continue
            x = float(x)
            if not math.isfinite(x):
                raise ValueError(f'{key} must be a number of seconds')
            out.append(x)
        return team, out[0], out[1]

    def coaching_text(store, data, status=200):
        return app.response_class(store.json_text(data), status=status, mimetype='application/json')

    def coaching_inputs(folder):
        """The match coaching report's inputs from the match's state (fullmatch/intel.py): the stoppages, set pieces and
        halftime as 'dead' coach spans (left out of the tactical judgements), the attacking direction from the halftime;
        (None, None) when the state is not built (then the report reads the possession record and the plan's periods)."""
        from . import halftime as ht, intel
        from soccer import team_shape
        snap = intel.snapshot(folder)
        if snap is None or snap.get('state') is None:
            return None, None
        index = snap['index']
        shape = team_shape.clean_settings({'direction': ht.direction_settings(index, snap['halftime'])})
        return intel.dead_spans(snap), shape

    def coaching_report(v, folder, mc, store):
        """GET /api/coaching?match=: one team's match report, as the Segment answer (fullmatch.coaching.report_state)."""
        team, t0, t1 = coaching_range(request.args)
        spans, shape = coaching_inputs(folder)
        return coaching_text(store, mc.report_state(folder, None, team, None, video_dir=v.directory, t0=t0, t1=t1,
                                                    spans=spans, shape_settings=shape))

    def coaching_build(v, folder, mc, store):
        """POST /api/coaching/build?match=: measure the segments (both teams, when not stored) and build the report on a
        background thread, one match build at a time (409 while another runs)."""
        body = request.get_json(silent=True) or {}
        team, t0, t1 = coaching_range(body)
        spans, shape = coaching_inputs(folder)
        out = mc.start_build(folder, None, team, None, video_dir=v.directory, t0=t0, t1=t1, force=bool(body.get('force')),
                             spans=spans, shape_settings=shape)
        return coaching_text(store, out, 409 if out.get('state') == 'busy' else 200)

    def coaching_settings_get(v, folder, mc, store):
        """GET /api/coaching/settings?match=: the match's coaching settings (its own file, else the video's without the
        confirmed roles, which name a Segment run's ids), and each team's measured roles once its roles file is stored."""
        settings, info = mc.match_settings(v.directory, folder)
        inferred = {}
        for team in ('A', 'B'):
            try:
                spans, shape = coaching_inputs(folder)
                inferred[team] = mc.report_state(folder, None, team, settings, video_dir=v.directory, make=False,
                                                 spans=spans, shape_settings=shape).get('inferred')
            except Exception:   # noqa: BLE001 - the roles editor then shows no measured role
                inferred[team] = None
        return coaching_text(store, {'run': mc.run_name({'matchId': folder.name}), 'settings': settings, 'inferred': inferred,
                                     'squads': [], 'links': [], 'matchDate': None, 'source': info})

    def coaching_settings_post(v, folder, mc, store):
        """POST /api/coaching/settings?match= {team, settings}: one team's block saved in the match's own settings file
        (assembly/coaching/coaching_settings.json); squads are not linked to a match."""
        body = request.get_json(silent=True) or {}
        team, raw = body.get('team'), body.get('settings')
        if team not in ('A', 'B') or not isinstance(raw, dict):
            raise ValueError('Send {team: A or B, settings: the team settings object}.')
        if raw.get('squad'):
            return jsonify(error='Squads and progress are not linked to a full match yet; save the team without a squad.'), 400
        try:
            saved = store.save_team_settings(mc.settings_dir(folder), team, raw, squad_exists=lambda squad: False)
        except store.SettingsRefused as error:
            return coaching_text(store, {'error': f'The settings were not saved: {error}', 'problems': error.problems}, 400)
        except store.SettingsUnreadable as error:
            return jsonify(error=str(error)), 503
        return coaching_text(store, {'ok': True, 'settings': saved['settings'], 'stale': saved['stale'], 'squad': None, 'note': None})

    def coaching_spans_get(v, folder, mc, store):
        """GET /api/coaching/spans for a match: none (coach-marked possession is a Segment run's)."""
        return coaching_text(store, {'run': mc.run_name({'matchId': folder.name}), 'spans': store.empty_spans(), 'segment': None,
                                     'saved': None, 'unavailable': 'Coach-marked possession is not available for a full match yet.'})

    def coaching_frame(v, folder, mc, store):
        """GET /api/coaching/frame?run=match-<id>: one evidence frame of a stored match report (only those)."""
        try:
            frame, width = int(request.args.get('frame', '')), int(request.args.get('w', 640))
        except (TypeError, ValueError):
            raise ValueError('frame and w must be whole numbers')
        if frame < 0 or width not in store.FRAME_WIDTHS:
            raise ValueError(f'frame must be 0 or more and w one of {", ".join(map(str, store.FRAME_WIDTHS))}')
        times = {}
        for record in store.records_v2(mc.assembly_dir(folder)):
            times.update(store.evidence_times(record))
        if frame not in times:
            return jsonify(error='That frame is not evidence in a coaching report of this match.'), 404
        try:
            data = store.frame_jpeg(v.source, v.directory, frame, width, times[frame], v.meta.get('fps'))
        except store.FrameUnavailable as error:
            return jsonify(error=str(error)), 404
        from io import BytesIO
        from flask import send_file
        return send_file(BytesIO(data), mimetype='image/jpeg')

    def coaching_pdf(v, folder, mc, store):
        """GET /api/coaching/pdf?run=match-<id>: the match report of one team as a PDF, the people named as the corrected
        match roster names them."""
        team, t0, t1 = coaching_range(request.args)
        spans, shape = coaching_inputs(folder)
        answer = mc.report_state(folder, None, team, None, video_dir=v.directory, t0=t0, t1=t1, spans=spans,
                                 shape_settings=shape)
        if answer['report'] is None:
            return jsonify(error='Generate the coaching report first.'), 404
        labels = {p['matchId']: p.get('display') or p['matchId'] for p in corrected_index(folder).get('roster') or []}
        colours = {t: request.args.get(f'colour{t}') for t in ('A', 'B')}
        colours = {t: c for t, c in colours.items() if isinstance(c, str) and store.HEX_COLOUR.fullmatch(c)}
        frames = store.EvidenceFrames(v.source, v.directory, answer['report'], v.meta.get('fps'))
        try:
            data = store.render_pdf(answer['report'], labels, frames, answer['reasons'] if answer['state'] != 'ready' else None,
                                    feedback=[], colours=colours or None, squad=None, coach_spans=False, marked=None)
        except store.PdfUnavailable as error:
            return jsonify(error=str(error)), 503
        from io import BytesIO
        from flask import send_file
        return send_file(BytesIO(data), mimetype='application/pdf', as_attachment=True,
                         download_name=store.pdf_filename_v2(answer['report'], v.meta.get('name'), answer['run'], v.key[:16]))

    moments_builds = {}               # (match id, team) -> {'state', 'progress', 'error', 'started'}
    moments_lock = threading.Lock()

    moments_kits = {}

    def moments_teams(folder):
        """{letter: {kit, label}} for every answer (the page names the team buttons before a report exists), from the
        kit colours the tracker learned (soccer/moments/kits.py), cached per assembled index."""
        from soccer.moments import kits
        path = Path(folder)/'assembly'/'match.json'
        try:
            stamp = (str(path), path.stat().st_mtime_ns)
        except OSError:
            return {t: {'kit': None, 'label': f'Team {t}'} for t in ('A', 'B')}
        if stamp not in moments_kits:
            named = kits.team_kits(json.loads(path.read_text(encoding='utf-8')))
            moments_kits.clear()
            moments_kits[stamp] = {t: {'kit': named.get(t), 'label': f'Team {t} ({named[t]})' if named.get(t) else f'Team {t}'}
                                   for t in ('A', 'B')}
        return moments_kits[stamp]

    def moments_profile(v, folder, mc, team):
        from soccer.moments import report as moments
        settings, _ = mc.match_settings(v.directory, folder)
        return moments.profile_from_settings(settings, team)

    def moments_report(v, folder, mc, store):
        """GET /api/coaching/moments?match=&team=: the team's key-moments report (soccer/moments/report.py) when it is
        stored for the match as it is now (roster corrections and the coaching level included), else the build's state."""
        from soccer.moments import report as moments
        team = request.args.get('team', 'A')
        if team not in ('A', 'B'):
            raise ValueError('team must be A or B')
        profile = moments_profile(v, folder, mc, team)
        report = moments.cached(folder, team, profile)
        with moments_lock:
            build = dict(moments_builds.get((folder.name, team)) or {})
        teams = moments_teams(folder)
        if report is not None:
            return coaching_text(store, {'state': 'ready', 'report': report, 'profile': profile, 'teams': teams})
        if build.get('state') == 'building':
            return coaching_text(store, {'state': 'building', 'progress': build.get('progress'), 'profile': profile,
                                         'teams': teams})
        return coaching_text(store, {'state': 'missing', 'error': build.get('error'), 'profile': profile, 'teams': teams})

    def moments_build(v, folder, mc, store):
        """POST /api/coaching/moments/build?match= {team}: build the key-moments report on a background thread (the
        match's data is read once from the views, about a minute on a 96-minute match, then cached); one at a time."""
        from soccer.moments import report as moments
        body = request.get_json(silent=True) or {}
        team = body.get('team', 'A')
        if team not in ('A', 'B'):
            raise ValueError('team must be A or B')
        profile = moments_profile(v, folder, mc, team)
        report = moments.cached(folder, team, profile)
        teams = moments_teams(folder)
        if report is not None:
            return coaching_text(store, {'state': 'ready', 'report': report, 'profile': profile, 'teams': teams})
        key = (folder.name, team)
        with moments_lock:
            if any(b.get('state') == 'building' for b in moments_builds.values()):
                return coaching_text(store, {'state': 'building', 'progress': (moments_builds.get(key) or {}).get('progress'),
                                             'busy': True, 'profile': profile, 'teams': teams})
            moments_builds[key] = {'state': 'building', 'progress': {'stage': 'starting'}, 'started': time.time()}

        def progress(i, n):
            with moments_lock:
                moments_builds[key]['progress'] = {'stage': 'reading the match', 'done': i, 'of': n}

        def work():
            try:
                moments.build(folder, team, profile, progress=progress)
                with moments_lock:
                    moments_builds[key] = {'state': 'ready'}
            except Exception as error:   # noqa: BLE001 - reported to the page
                with moments_lock:
                    moments_builds[key] = {'state': 'error', 'error': f'The key-moments report could not be built: {error}'}
        threading.Thread(target=work, name='moments-build', daemon=True).start()
        return coaching_text(store, {'state': 'building', 'progress': {'stage': 'starting'}, 'profile': profile, 'teams': teams})

    coaching_handlers = {('GET', '/api/coaching'): coaching_report, ('POST', '/api/coaching/build'): coaching_build,
                         ('GET', '/api/coaching/moments'): moments_report, ('POST', '/api/coaching/moments/build'): moments_build,
                         ('GET', '/api/coaching/settings'): coaching_settings_get,
                         ('POST', '/api/coaching/settings'): coaching_settings_post,
                         ('GET', '/api/coaching/spans'): coaching_spans_get, ('GET', '/api/coaching/frame'): coaching_frame,
                         ('GET', '/api/coaching/pdf'): coaching_pdf}

    @app.before_request
    def coaching_in_match_mode():
        """The coaching routes of a full match (fullmatch/coaching.py: ONE report per team over the whole match, from the
        assembled views): a request with match= (the page adds it in match playback) or for a 'match-<id>' run is
        answered here, never by the Segment run the video last showed. The catalogue of profiles is the same for both;
        what a match has no data for yet (coach-marked possession edits, squads and progress, feedback, the match date)
        is refused plainly."""
        if not request.path.startswith('/api/coaching'):
            return None
        match_id = coaching_match_id()
        if not match_id or request.path == '/api/coaching/profiles':
            return None
        if request.headers.get('Sec-Fetch-Site') in ('cross-site', 'same-site'):
            return jsonify(error='Coaching requests are answered for this page only.'), 403
        if 'context' in request.args:
            return jsonify(error='The v1 coaching report is not made for a full match.'), 400
        try:
            v, folder = assembled(match_id)
        except LookupError as error:
            return jsonify(error=str(error), state='unavailable'), 404
        handler = coaching_handlers.get((request.method, request.path))
        if handler is None:
            return jsonify(error='Not available for a full match yet.', state='unavailable'), 404
        try:
            from soccer.coaching import store
            from . import coaching as match_coaching
        except Exception as error:   # noqa: BLE001 - a broken coaching package fails the coaching routes only
            return jsonify(error=f'The coaching engine could not be loaded: {error}'), 503
        try:
            return handler(v, folder, match_coaching, store)
        except (ValueError, TypeError) as error:
            return jsonify(error=str(error)), 400
    context['match_coaching'] = True

    @app.get('/match.js')
    def match_script():
        """The page's Full Match Mode (the dock's match panel and the match player)."""
        from flask import send_file
        return send_file(Path(context['root'])/'match.js', mimetype='text/javascript')

    @app.get('/api/match')
    def match_overview():
        v = context['video']()
        snap = system.snapshot(gpu=False)
        auto, why = system.auto_workers(snap)
        out = {'running': current_status() if busy() else None, 'last': current_status() if not busy() else None,
               'video': None, 'matches': [], 'resumable': None, 'busy': busy(),
               'defaults': {'segmentSeconds': planning.SEGMENT_S, 'padSeconds': planning.PAD_S, 'workers': 'auto',
                            'autoWorkers': auto, 'maxWorkers': system.MAX_WORKERS, 'autoMax': system.AUTO_MAX}}
        if v is not None:
            out['video'] = {'key': v.key, 'name': v.meta.get('name'), 'duration': v.meta['duration'], 'fps': v.meta['fps']}
            out['matches'] = matches_of(v)
            out['resumable'] = resumable_of(out['matches'])
        return jsonify(out)

    @app.post('/api/match/plan')
    def match_plan():
        try:
            _, _, _, _, out = describe(request.get_json() or {})
            return jsonify(out)
        except (ValueError, KeyError, TypeError) as error:
            return jsonify(error=str(error)), 400

    @app.post('/api/match/start')
    def match_start():
        try:
            body = request.get_json() or {}
            with lock:
                if busy():
                    raise ValueError('A full match analysis is already running.')
                if context['single_busy']():
                    raise ValueError('A segment run is processing; wait for it or stop it first.')
                v, periods, segments, settings, out = describe(body)
                if not out['disk']['ok']:
                    raise ValueError(f"Not enough free disk space: about {out['disk']['neededText']} needed, {out['disk']['freeText']} free.")
                folder = v.directory/'matches'/mf.new_id()
                source = {'key': v.key, 'path': str(v.source), 'bytes': v.source.stat().st_size, 'name': v.meta.get('name')}
                manifest = mf.Manifest.create(folder, source, v.meta, periods, segments, settings,
                                              {'mode': out['workers']['mode'], 'count': out['workers']['count']},
                                              out['padSeconds'], out['segmentSeconds'],
                                              extra={'estimate': out['estimate'], 'calibrationStatus': out['calibration'],
                                                     'discovery': {'mode': out['calibration']['discovery']['mode'],
                                                                   'state': 'pending' if out['calibration']['discovery']['runs'] else 'off'}})
                # The dock's settings become the video's saved settings, as a Segment run's do.
                saved = dict(settings) | {'start': float(v.defaults.get('start', 0)), 'seconds': float(v.defaults.get('seconds', 60))}
                v.saved.write_text(json.dumps(saved, indent=2), encoding='utf-8')
                v.defaults.update(settings)
                launch(manifest, out['workers']['count'], int(body.get('threads', 0)), resume=False)
            return jsonify(ok=True, matchId=manifest.data['matchId'], plan=out)
        except (ValueError, KeyError, TypeError, RuntimeError, OSError) as error:
            return jsonify(error=str(error)), 400

    @app.post('/api/match/resume')
    def match_resume():
        try:
            body = request.get_json() or {}
            with lock:
                if busy():
                    raise ValueError('A full match analysis is already running.')
                if context['single_busy']():
                    raise ValueError('A segment run is processing; wait for it or stop it first.')
                v = context['video']()
                if v is None:
                    raise ValueError('Open the match video first.')
                match_id = str(body.get('matchId') or '')
                path = v.directory/'matches'/match_id/'manifest.json'
                if not match_id or path.parent.parent != v.directory/'matches' or not path.exists():
                    raise ValueError('That match does not exist for this video.')
                manifest = mf.Manifest.load(path)
                problem = manifest.compatible(v.key, v.meta['frames'], v.meta['fps'], manifest.data['settings'])
                if problem:
                    raise ValueError(f'This match cannot be resumed: {problem}.')
                if manifest.data['state'] == 'complete':
                    raise ValueError('This match is already complete.')
                snap = system.snapshot()
                chosen = body.get('workers', manifest.data['workers'].get('mode', 'auto'))
                workers = workers_for(chosen, snap)
                launch(manifest, workers['count'], int(body.get('threads', 0)), resume=True)
            return jsonify(ok=True, matchId=match_id)
        except (ValueError, KeyError, TypeError, RuntimeError, OSError) as error:
            return jsonify(error=str(error)), 400

    @app.post('/api/match/pause')
    def match_pause():
        with lock:
            if not busy():
                return jsonify(error='No full match analysis is running.'), 400
            active['scheduler'].pause()
        return jsonify(ok=True)

    @app.post('/api/match/stop')
    def match_stop():
        with lock:
            if not busy():
                return jsonify(error='No full match analysis is running.'), 400
            active['scheduler'].stop()
        return jsonify(ok=True)

    @app.get('/api/match/status')
    def match_status():
        status = current_status()
        if status is None:
            return jsonify(state='none')
        return jsonify(status)

    def assembled(match_id):
        """(video, manifest folder) of an assembled match of the open video, or raise LookupError."""
        v = context['video']()
        if v is None:
            raise LookupError('Open the match video first.')
        root = v.directory/'matches'
        folder = root/str(match_id or '')
        if not match_id or folder.parent != root or not (folder/'assembly'/'match.json').exists():
            raise LookupError('That match has not been assembled for this video.')
        return v, folder

    samples_cache = {}

    def match_samples(match_id, orient=None, kind='movement'):
        """The match's player samples (assembly/samples.npz as fullmatch.analytics.MatchSamples), one cached per match,
        orientation and kind: the server's /api/analytics answers a match from them exactly as it answers a segment.
        With the match's people built (fullmatch/intel.py): one person per real-person estimate, the attacking direction
        from the halftime, only the samples `kind` of analysis counts (gamestate.ELIGIBLE: 'movement' for heatmaps,
        positions and trails; 'open_play' for team shape), trails broken at the halftime."""
        _, folder = assembled(match_id)
        path = folder/'assembly'/'samples.npz'
        if not path.exists():
            raise LookupError('This match has no player samples yet: assemble it again.')
        orient = 'attack' if orient == 'attack' else None
        from . import corrections as match_corrections, intel
        snap = intel.snapshot(folder)
        fixes = folder/'assembly'/match_corrections.FILE
        key = (str(path), path.stat().st_mtime_ns, orient, kind, fixes.stat().st_mtime_ns if fixes.exists() else 0,
               id(snap) if snap is not None else None)
        with lock:
            if key in samples_cache:
                return samples_cache[key]
        index = json.loads((folder/'assembly'/'match.json').read_text(encoding='utf-8'))
        from . import analytics as match_analytics, halftime as ht
        if snap is not None:
            aliases, excluded, teams = intel.aliases_of(snap['people'])
            names = {pid: {'display': p['display'], 'team': p['team'], 'role': 'GOALKEEPER' if p['keeper'] else 'PLAYER'}
                     for pid, p in snap['people']['people'].items()}
            periods = [dict(q, defendsLeft=q.get('attacksRight')) for q in snap['directionPeriods']]
            half = ht.interval(snap['halftime'])
            built = match_analytics.MatchSamples(path, periods=periods, orient=orient, aliases=aliases, excluded_ids=excluded,
                                                 teams=teams, keep=intel.keep_function(snap, kind),
                                                 breaks_at=sorted(set(half)) if half else (), people=names)
        else:
            aliases, excluded, teams = match_corrections.resolve_aliases(match_corrections.load(folder))
            built = match_analytics.MatchSamples(path, periods=index['periods'], orient=orient, aliases=aliases,
                                                 excluded_ids=excluded, teams=teams)
        with lock:
            if len(samples_cache) >= 3:
                samples_cache.pop(next(iter(samples_cache)))
            samples_cache[key] = built
        return built

    def match_possession(match_id):
        """The match's possession record (assembly/possession.json) as JSON text."""
        _, folder = assembled(match_id)
        path = folder/'assembly'/'possession.json'
        if not path.exists():
            raise LookupError('This match has no possession record yet: assemble it again.')
        from . import analytics as match_analytics, corrections as match_corrections, intel
        snap = intel.snapshot(folder)
        items = match_corrections.load(folder)
        if snap is not None:
            aliases, excluded, _ = intel.aliases_of(snap['people'])
        elif not items:
            return path.read_text(encoding='utf-8')
        else:
            aliases, excluded, _ = match_corrections.resolve_aliases(items)
        record = json.loads(path.read_text(encoding='utf-8'))
        return json.dumps(match_analytics.possession_with(record, aliases, excluded), separators=(',', ':'))
    shape_cache = {}

    def match_shape(match_id, mode='observed'):
        """The match's team shape (soccer/team_shape.TeamShape) over every owned frame: the match samples, the owned frames'
        times and camera views (assembly/frames.json.gz), the corrected match roster, and the attacking direction from the
        periods (team A's defending side in the first period, swapped from the second period's start when the teams
        changed ends). Observed positions only: the estimated mode needs each segment's estimated timeline merged."""
        if mode != 'observed':
            raise LookupError('A full match has observed team shapes only (the estimated positions are per segment).')
        _, folder = assembled(match_id)
        path = folder/'assembly'/'frames.json.gz'
        if not path.exists():
            raise LookupError('This match has no team-shape inputs yet: assemble it again.')
        samples = match_samples(match_id, kind='open_play')
        from . import assemble as assembly, halftime as ht, intel
        from soccer import team_shape
        index = corrected_index(folder)
        key = (match_id, id(samples), path.stat().st_mtime_ns)
        with lock:
            if key in shape_cache:
                return shape_cache[key]
        frames = assembly.load_view(folder/'assembly', 'frames.json.gz')
        snap = intel.snapshot(folder)
        if snap is not None:
            direction = ht.direction_settings(index, snap['halftime'])
            first = None if direction['A'] is None else ('A' if direction['A'] == 'left' else 'B')
        else:
            periods = index['periods']
            first = periods[0].get('defendsLeft') if periods else None
            direction = {'A': None, 'switchAt': None}
            if first in ('A', 'B'):
                direction['A'] = 'left' if first == 'A' else 'right'
                if len(periods) > 1 and periods[1].get('defendsLeft') in ('A', 'B') and periods[1]['defendsLeft'] != first:
                    direction['switchAt'] = float(periods[1]['start'])
        people = [p for p in index['roster'] if not p.get('excluded') and p.get('team') in ('A', 'B')
                  and p.get('confidence') != 'unresolved']
        entries = [{'id': p['matchId'], 'display': p['display'], 'team': p['team'], 'role': p['role'], 'status': 'ACTIVE',
                    'firstSeen': p['firstSeen'], 'lastSeen': p['lastSeen']} for p in people]
        slim = {'frames': [{'time': t, 'view': v} for t, v in zip(frames['time'], frames['view'])],
                'roster': {'expected': {'A': 11, 'B': 11, 'goalkeepers': 2},
                           'slots': [e | {'left': None} for e in entries]},
                'tactical': {'goalEnds': ({'left': first, 'right': 'B' if first == 'A' else 'A'} if first in ('A', 'B') else {}),
                             'pitchSize': index.get('pitchSize') or {'length': 105.0, 'width': 68.0}},
                'match': {'players': [e for e in entries if e['role'] != 'GOALKEEPER'],
                          'goalkeepers': [e for e in entries if e['role'] == 'GOALKEEPER']},
                'timeline': None, 'corrections': None}
        built = team_shape.TeamShape(slim, samples=samples, mode='observed',
                                     settings=team_shape.clean_settings({'direction': direction}))
        with lock:
            shape_cache.clear()
            shape_cache[key] = built
        return built

    context['match_samples'] = match_samples
    context['match_possession'] = match_possession
    context['match_shape'] = match_shape

    def corrected_index(folder):
        """The assembled index with the roster the page shows: the match's people and slots (fullmatch/persons.py)
        once they are built, else the coach's merges over the stitched roster as before."""
        from . import corrections as match_corrections, intel, persons
        snap = intel.snapshot(folder)
        index = json.loads((folder/'assembly'/'match.json').read_text(encoding='utf-8'))
        kits = moments_teams(folder)
        if snap is None:
            out = match_corrections.apply_index(index, match_corrections.load(folder))
            out['intel'] = {'state': 'missing'}
            out['teamKits'] = kits
            return out
        out = persons.corrected_index(index, snap['people'], snap['items'])
        out['halftime'] = snap['halftime']
        out['directionPeriods'] = snap['directionPeriods']
        out['intel'] = {'state': 'ready' if snap.get('state') is not None else 'people'}
        out['teamKits'] = kits
        return out

    @app.get('/api/match/index')
    def match_index():
        """The assembled match (assembly/match.json) with the coach's roster corrections applied: periods, segments with
        their cuts and views, boundaries, the roster, the merge suggestions, and the alias map the page renames views by."""
        try:
            _, folder = assembled(request.args.get('id'))
        except LookupError as error:
            return jsonify(error=str(error)), 404
        return jsonify(corrected_index(folder))

    @app.post('/api/match/correction')
    def match_correction():
        """{match, kind: merge|exclude|team, target, into?, team?}: a roster correction of an assembled match."""
        from . import corrections as match_corrections
        body = request.get_json() or {}
        try:
            _, folder = assembled(body.get('match'))
            index = json.loads((folder/'assembly'/'match.json').read_text(encoding='utf-8'))
            try:
                together = json.loads((folder/'assembly'/'together.json').read_text(encoding='utf-8'))
            except (OSError, ValueError):
                together = {}
            from . import intel
            snap = intel.snapshot(folder)
            with lock:
                entry = match_corrections.add(folder, index, together, body, people=snap['people'] if snap else None)
            if snap is not None and entry.get('kind') in RESTATE_KINDS:
                start_intel(context['video'](), folder)       # the people's teams or places changed: remake the state
            return jsonify(ok=True, correction=entry, index=corrected_index(folder))
        except LookupError as error:
            return jsonify(error=str(error)), 404
        except (ValueError, KeyError, TypeError) as error:
            return jsonify(error=str(error)), 400

    @app.post('/api/match/correction/remove')
    def match_correction_remove():
        from . import corrections as match_corrections
        body = request.get_json() or {}
        try:
            _, folder = assembled(body.get('match'))
            with lock:
                match_corrections.remove(folder, str(body.get('id') or ''))
            from . import intel
            if intel.snapshot(folder) is not None:
                start_intel(context['video'](), folder)
            return jsonify(ok=True, index=corrected_index(folder))
        except LookupError as error:
            return jsonify(error=str(error)), 404
        except ValueError as error:
            return jsonify(error=str(error)), 400

    intel_builds = {}            # folder -> {'state', 'progress', 'stage', 'error'}

    def intel_answer(folder):
        """The match's state for the page: ready (timeline, events, masks, halftime, stats), building, or missing."""
        from . import gamestate, halftime as ht, intel
        snap = intel.snapshot(folder)
        with lock:
            build = dict(intel_builds.get(str(folder)) or {})
        index = snap['index'] if snap else json.loads((folder/'assembly'/'match.json').read_text(encoding='utf-8'))
        settings = snap['halftime'] if snap else ht.load(folder, index)
        base = {'halftime': settings, 'directionPeriods': ht.periods(index, settings), 'build': build,
                'eligible': {k: list(v) for k, v in gamestate.ELIGIBLE.items()}, 'states': list(gamestate.STATES),
                'events': [], 'eventTypes': list(gamestate.EVENTS), 'masks': None}
        if snap is None or snap.get('state') is None:
            return base | {'state': build.get('state') if build.get('state') in ('building', 'error') else 'missing',
                           'error': build.get('error')}
        st = snap['state']
        return base | {'state': 'ready', 'timeline': st['timeline'], 'events': st['events'], 'stats': st['stats'],
                       'masks': snap['masks'], 'corrections': st.get('corrections') or [],
                       'stoppages': st.get('stoppages') or [], 'directionCheck': st.get('directionCheck')}

    def start_intel(v, folder):
        """Make the match's state on a thread (one per match at a time)."""
        from . import intel
        key = str(folder)
        with lock:
            if (intel_builds.get(key) or {}).get('state') == 'building':
                return
            intel_builds[key] = {'state': 'building', 'progress': 0.0, 'stage': 'starting'}

        def progress(f, text):
            with lock:
                intel_builds[key] = {'state': 'building', 'progress': round(float(f), 3), 'stage': text}

        def work():
            try:
                intel.build(folder, video=getattr(v, 'source', None), progress=progress)
                with lock:
                    intel_builds[key] = {'state': 'ready'}
            except Exception as error:   # noqa: BLE001 - reported to the page
                import traceback
                traceback.print_exc()
                with lock:
                    intel_builds[key] = {'state': 'error', 'error': f'The match state could not be made: {error}'}
        threading.Thread(target=work, name='match-intel', daemon=True).start()

    @app.get('/api/match/intel')
    def match_intel():
        """?id=<match>: the match's state timeline, events (with the coach's corrections), masks and halftime."""
        try:
            _, folder = assembled(request.args.get('id'))
        except LookupError as error:
            return jsonify(error=str(error)), 404
        return jsonify(intel_answer(folder))

    @app.post('/api/match/intel/build')
    def match_intel_build():
        body = request.get_json(silent=True) or {}
        try:
            v, folder = assembled(body.get('match'))
        except LookupError as error:
            return jsonify(error=str(error)), 404
        start_intel(v, folder)
        return jsonify(intel_answer(folder))

    # ---------- possession v2 (fullmatch/possession_match.py): controlled in-play possession over the whole match ----------
    possession2_builds = {}      # folder -> {'state', 'progress', 'stage', 'error'}

    def possession2_answer(folder, t0=None, t1=None):
        """The match's possession v2 for the page: ready (the record, the range's stats when t0 / t1 are given, the team
        labels), building (progress), needs-state (the match state is not built: the build makes it first), missing or
        error."""
        from soccer import possession_v2 as pv
        from . import intel, possession_match as pm
        key = str(folder)
        with lock:
            build = dict(possession2_builds.get(key) or {})
        base = {'teams': moments_teams(folder), 'definition': pv.DEFINITION}
        if build.get('state') == 'building':
            return base | {'state': 'building', 'progress': build.get('progress'), 'stage': build.get('stage')}
        snap = intel.snapshot(folder)
        if snap is None or snap.get('state') is None:
            return base | {'state': 'error' if build.get('state') == 'error' else 'needs-state', 'error': build.get('error')}
        rec = pm.latest(folder, snap)
        if rec is None:
            return base | {'state': 'error' if build.get('state') == 'error' else 'missing', 'error': build.get('error')}
        out = base | {'state': 'ready', 'record': rec}
        if t0 is not None or t1 is not None:
            out['range'] = pv.range_stats(rec, t0, t1) | {'t0': t0, 't1': t1}
        return out

    def start_possession2(v, folder):
        """Make the match's possession v2 record on a thread (one per match at a time)."""
        from . import intel, possession_match as pm
        key = str(folder)
        with lock:
            if (possession2_builds.get(key) or {}).get('state') == 'building':
                return
            possession2_builds[key] = {'state': 'building', 'progress': 0.0, 'stage': 'starting'}

        def progress(f, text):
            with lock:
                possession2_builds[key] = {'state': 'building', 'progress': round(float(f), 3), 'stage': text}

        def work():
            try:
                snap = intel.snapshot(folder)
                if snap is None or snap.get('state') is None:
                    snap = intel.build(folder, video=getattr(v, 'source', None),
                                       progress=lambda f, text: progress(.4*f, f'Match state: {text}'))
                pm.build(folder, snap, progress=lambda f, text: progress(.4+.6*f, text))
                with lock:
                    possession2_builds[key] = {'state': 'ready'}
            except Exception as error:   # noqa: BLE001 - reported to the page
                import traceback
                traceback.print_exc()
                with lock:
                    possession2_builds[key] = {'state': 'error', 'error': f'Possession could not be worked out: {error}'}
        threading.Thread(target=work, name='match-possession2', daemon=True).start()

    def match_possession2(match_id, t0=None, t1=None):
        _, folder = assembled(match_id)
        return possession2_answer(folder, t0, t1)

    context['match_possession2'] = match_possession2

    @app.post('/api/possession2/build')
    def possession2_build():
        body = request.get_json(silent=True) or {}
        try:
            v, folder = assembled(body.get('match'))
        except LookupError as error:
            return jsonify(error=str(error)), 404
        start_possession2(v, folder)
        return jsonify(possession2_answer(folder))

    @app.post('/api/match/halftime')
    def match_halftime():
        """{match, mode: marker|break, firstHalfEnd, secondHalfStart, teamAFirstHalf: left|right}: saved, and the state
        made again for it (on a thread: directions, masks and the reports that read them follow; no detection runs)."""
        from . import halftime as ht
        body = request.get_json(silent=True) or {}
        try:
            v, folder = assembled(body.get('match'))
            index = json.loads((folder/'assembly'/'match.json').read_text(encoding='utf-8'))
            settings = ht.clean(index, body)
        except LookupError as error:
            return jsonify(error=str(error)), 404
        except ValueError as error:
            return jsonify(error=str(error)), 400
        with lock:
            ht.save(folder, settings)
        start_intel(v, folder)
        return jsonify(ok=True, intel=intel_answer(folder), index=corrected_index(folder))

    @app.post('/api/match/halftime/reset')
    def match_halftime_reset():
        from . import halftime as ht
        body = request.get_json(silent=True) or {}
        try:
            v, folder = assembled(body.get('match'))
        except LookupError as error:
            return jsonify(error=str(error)), 404
        with lock:
            ht.reset(folder)
        start_intel(v, folder)
        return jsonify(ok=True, intel=intel_answer(folder), index=corrected_index(folder))

    @app.post('/api/match/event')
    def match_event():
        """{match, kind, ...}: an event correction (fullmatch/gamestate.add_correction)."""
        from . import gamestate, intel
        body = request.get_json(silent=True) or {}
        try:
            _, folder = assembled(body.get('match'))
            snap = intel.snapshot(folder)
            if snap is None or snap.get('state') is None:
                return jsonify(error='Make the match state first.'), 409
            with lock:
                entry = gamestate.add_correction(folder, snap['state'], body)
        except LookupError as error:
            return jsonify(error=str(error)), 404
        except (ValueError, KeyError, TypeError) as error:
            return jsonify(error=str(error)), 400
        return jsonify(ok=True, correction=entry, intel=intel_answer(folder))

    @app.post('/api/match/event/remove')
    def match_event_remove():
        from . import gamestate
        body = request.get_json(silent=True) or {}
        try:
            _, folder = assembled(body.get('match'))
            with lock:
                gamestate.remove_correction(folder, str(body.get('cid') or ''))
        except LookupError as error:
            return jsonify(error=str(error)), 404
        except ValueError as error:
            return jsonify(error=str(error)), 400
        return jsonify(ok=True, intel=intel_answer(folder))

    @app.get('/api/match/view')
    def match_view():
        """One segment's view (its result with the match's names and the frames it owns), sent compressed as stored: the
        browser inflates it (Content-Encoding: gzip). ?id=<match>&segment=<index>."""
        try:
            _, folder = assembled(request.args.get('id'))
            index = json.loads((folder/'assembly'/'match.json').read_text(encoding='utf-8'))
            k = int(request.args.get('segment', -1))
            entry = next(s for s in index['segments'] if s['index'] == k)
        except (LookupError, StopIteration, ValueError) as error:
            return jsonify(error=str(error) or 'No such segment.'), 404
        from flask import Response
        path = folder/'assembly'/entry['view']
        data = path.read_bytes()
        response = Response(data, mimetype='application/json')
        response.headers['Content-Encoding'] = 'gzip'
        response.headers['Content-Length'] = str(len(data))
        return response

    @app.get('/api/match/latest')
    def match_latest():
        """The newest assembled match of the open video (what 'Open the full match' opens)."""
        v = context['video']()
        if v is None:
            return jsonify(match=None)
        for m in mf.find(v.directory):
            if (m.folder/'assembly'/'match.json').exists() and m.data['state'] == 'complete':
                return jsonify(match={'matchId': m.data['matchId'], 'created': m.data['created'],
                                      'assembledAt': (m.data.get('assembly') or {}).get('at'),
                                      'segments': len(m.data['segments']), 'periods': m.data['periods'],
                                      'runtime': m.data.get('runtime')})
        return jsonify(match=None)

    def close():
        s = active['scheduler']
        if s is not None and busy():
            s.stop()
            s.wait(20)
    context['match_close'] = close
    return context


def resumable_of(matches):
    """The incomplete match RESUME ANALYSIS offers (matches newest first): the newest incomplete one of this file, unless
    a newer analysis of it completed (a stopped match the complete one replaced was offered above it)."""
    for m in matches:
        if m['state'] == 'complete' and m['compatible']:
            return None
        if m['state'] in ('paused', 'stopped', 'failed', 'planned') and m['compatible']:
            return m
    return None


def ceil_minutes(seconds):
    return int(math.ceil(float(seconds)/60))
