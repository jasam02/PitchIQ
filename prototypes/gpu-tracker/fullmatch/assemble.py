"""The assembled match: the finished segment artifacts turned into one match the page plays from start to end, without
re-running anything on the GPU.

    1. every segment done and its artifact verified (manifest.verify_artifact, with the checksum)
    2. the plan's owned frames cover each period without a gap (plan.check) and each artifact covers its owned frames
    3. the people stitched into match identities (identities.resolve: team letters, overlaps, gap links, the roster)
    4. every boundary looked at over the frames both neighbours processed: where the two runs' balls agree, how far
       their pitch calibrations disagree (metres), how many people matched; and the CUT chosen: the frame within
       CUT_WINDOW_S of the planned boundary where the two balls agree best (no visible jump when the page swaps from one
       segment to the next); without any agreeing frame the planned boundary stays and the boundary is flagged
    5. every segment's VIEW written: its own result, untouched in every number, with the match's names (people, the
       roster, the 2D timeline and stabilized columns, the labels, the tracks, the events renamed; the team letters
       swapped where the run named the kits the other way round) and the range of frames it owns in the match
       ('match': {cut0, cut1}), gzip-compressed
    6. the match index (assembly/match.json): periods with their attacking directions, segments with their cuts, views,
       quality and calibration coverage, the boundaries, the roster, how it was stitched, timings
    7. the match's people, places, state and events (fullmatch/intel.py: about a minute and a half on a full match;
       its failure is noted in the manifest, never fatal)

Nothing here changes a position, a box, a ball or a calibration: a view is its segment's result with names changed and a
note of what it owns; the page swaps views at the cuts and the match analytics count every frame once (the frames
between the cuts).
"""
import gzip
import json
import math
import re
import time
from pathlib import Path

import numpy as np

from . import identities
from . import manifest as mf
from . import plan as planning

VERSION = 1
CUT_WINDOW_S = 1.5        # the cut may move this far from the planned boundary (inside the overlap)
BALL_AGREE_PX = 24.0      # two balls this close (1080p pixels) are the same ball
GZIP_LEVEL = 5
UNDERCONSTRAINED_CONF = .35


def _H(value):
    if not value:
        return None
    try:
        H = np.asarray(value, float).reshape(3, 3)
        return H if np.all(np.isfinite(H)) else None
    except (ValueError, TypeError):
        return None


def _letters(text, swap):
    if not swap or not isinstance(text, str):
        return text
    return re.sub(r'(TEAM_|Team)([AB])', lambda m: m.group(1)+('B' if m.group(2) == 'A' else 'A'), text)


def relabel(result, ids, swap, displays, roles, segment, cut0, cut1, match_id):
    """In place: the segment's result with match names (ids: local id -> match id; displays: match id -> shown label),
    team letters swapped when `swap`, and what it owns in the match. Numbers are not touched."""
    sw = lambda t: identities._swap(t, swap)
    rename = lambda pid: ids.get(pid, pid) if pid is not None else None
    for f in result.get('frames') or []:
        for p in f.get('people') or []:
            if p.get('id') is not None:
                p['segmentId'] = p['id']
                p['id'] = rename(p['id'])
                if p['id'] in displays:
                    p['display'] = displays[p['id']]
            p['team'] = sw(p.get('team'))
            p['label'] = _letters(p.get('label'), swap)
            if p.get('display') and swap and re.match(r'^[AB]-\?$', p['display']):
                p['display'] = sw(p['display'][0])+p['display'][1:]
        vis = f.get('visibility')
        if isinstance(vis, dict):
            f['visibility'] = {rename(k): v for k, v in vis.items()}
    match = result.get('match') or {}
    for group, entries in match.items():
        if not isinstance(entries, list):
            continue
        for e in entries:
            if e.get('id') is not None:
                e['segmentId'] = e['id']
                e['id'] = rename(e['id'])
            e['team'] = sw(e.get('team'))
            e['label'] = _letters(e.get('label'), swap)
            if e.get('id') in displays:
                e['display'] = displays[e['id']]
                from soccer.labels import long_label
                e['longId'] = long_label(e['display'])
            if e.get('id') in roles and group in ('players', 'goalkeepers'):
                e['matchRole'] = roles[e['id']]
    roster = result.get('roster') or {}
    for s in roster.get('slots') or []:
        s['segmentId'] = s.get('id')
        s['id'] = rename(s.get('id'))
        s['team'] = sw(s.get('team'))
        if s['id'] in displays:
            s['display'] = displays[s['id']]
    if swap:
        for key in ('expected', 'counts', 'overCap'):
            block = roster.get(key)
            if isinstance(block, dict) and ('A' in block or 'B' in block):
                roster[key] = {sw(k): v for k, v in block.items()}
    for block_name in ('timeline', 'stabilized'):
        block = result.get(block_name)
        if isinstance(block, dict) and isinstance(block.get('ids'), dict):
            renamed = {}
            for k, v in block['ids'].items():
                if isinstance(v, dict) and 'team' in v:
                    v['team'] = sw(v['team'])
                renamed[rename(k)] = v
            block['ids'] = renamed
    labels_map = result.get('labels')
    if isinstance(labels_map, dict):
        from soccer.labels import long_label
        result['labels'] = {rename(k): (long_label(displays[rename(k)]) if rename(k) in displays else _letters(v, swap))
                            for k, v in labels_map.items()}
    for t in (result.get('tracks') or {}).values():
        if isinstance(t.get('ids'), dict):
            t['ids'] = {rename(k): v for k, v in t['ids'].items()}
    for e in result.get('events') or []:
        if e.get('playerId') is not None:
            e['playerId'] = rename(e['playerId'])
    if swap:
        teams = result.get('teams') or {}
        for a, b in (('a', 'b'), ('shortsA', 'shortsB'), ('spreadA', 'spreadB')):
            if a in teams and b in teams:
                teams[a], teams[b] = teams[b], teams[a]
        for k in teams.get('keepers') or []:
            if isinstance(k, dict):
                k['team'] = sw(k.get('team'))
        ends = (result.get('tactical') or {}).get('goalEnds')
        if isinstance(ends, dict):
            result['tactical']['goalEnds'] = {k: sw(v) for k, v in ends.items()}
        summary = result.get('summary') or {}
        if isinstance(summary.get('goalSides'), dict):
            summary['goalSides'] = {side: {sw(k): v for k, v in votes.items()} for side, votes in summary['goalSides'].items()}
        if isinstance(summary.get('goalkeeperSlots'), dict):
            summary['goalkeeperSlots'] = {sw(k): v for k, v in summary['goalkeeperSlots'].items()}
        gi = summary.get('globalIdentities')
        if isinstance(gi, dict) and 'teamA' in gi and 'teamB' in gi:
            gi['teamA'], gi['teamB'] = gi['teamB'], gi['teamA']
    result['matchView'] = {'matchId': match_id, 'segment': segment.index, 'period': segment.period,
                           'own0': segment.own0, 'own1': segment.own1, 'cut0': cut0, 'cut1': cut1,
                           'teamsSwapped': bool(swap), 'version': VERSION}
    return result


def _ball_px(f, width, height):
    b = f.get('ball') or {}
    c = b.get('center')
    if not c or b.get('state') not in ('TRACKED', 'MISSING'):
        return None
    return float(c[0])*width, float(c[1])*height


def boundary_overlap(result_a, result_b, a, b, fps, width, height, size):
    """The frames a and b both processed: ball agreement per frame, calibration disagreement, and the cut frame."""
    from soccer import field, registration
    fa = {f['frame']: f for f in result_a['frames'] if f['frame'] >= b.proc0}
    fb = {f['frame']: f for f in result_b['frames'] if f['frame'] < a.proc1}
    shared = sorted(set(fa) & set(fb))
    rows, metres = [], []
    pitch_size = field.PitchSize(float(size.get('length', 105.0)), float(size.get('width', 68.0))) if size else None
    scale = 1080.0/max(1.0, float(height))
    for k in shared:
        pa, pb = _ball_px(fa[k], width, height), _ball_px(fb[k], width, height)
        if pa is not None and pb is not None:
            d = math.hypot(pa[0]-pb[0], pa[1]-pb[1])*scale
            rows.append((k, 'both', d))
        elif pa is None and pb is None:
            rows.append((k, 'neither', None))
        else:
            rows.append((k, 'one', None))
        Ha, Hb = _H((fa[k].get('calibration') or {}).get('H')), _H((fb[k].get('calibration') or {}).get('H'))
        if Ha is not None and Hb is not None and k % 5 == 0:
            try:
                metres.append(registration._disagreement(Ha, Hb, width, height, pitch_size))
            except (np.linalg.LinAlgError, ValueError):
                pass
    planned = b.own0
    window = int(round(CUT_WINDOW_S*fps))
    candidates = [r for r in rows if abs(r[0]-planned) <= window]
    agreeing = [r for r in candidates if r[1] == 'neither' or (r[1] == 'both' and r[2] <= BALL_AGREE_PX)]
    at_plan = next((r for r in rows if r[0] == planned), None)
    if at_plan is not None and (at_plan[1] == 'neither' or (at_plan[1] == 'both' and at_plan[2] <= BALL_AGREE_PX)):
        cut, how = planned, 'planned boundary: both runs agree there'
    elif agreeing:
        best = min(agreeing, key=lambda r: (r[1] != 'both', abs(r[0]-planned)))
        cut, how = best[0], 'moved to where both runs agree on the ball'
    else:
        cut, how = planned, 'the runs disagree on the ball around the boundary (flagged)'
    both = [r[2] for r in rows if r[1] == 'both']
    finite = [m for m in metres if math.isfinite(m)]
    times = {k: fa[k]['time'] for k in shared}
    return {'between': [a.index, b.index], 'plannedFrame': planned, 'cutFrame': cut, 'cutTime': times.get(cut, cut/fps), 'how': how,
            'sharedFrames': len(shared),
            'ball': {'bothSeen': len(both), 'agreeing': sum(1 for d in both if d <= BALL_AGREE_PX),
                     'medianPx': round(float(np.median(both)), 1) if both else None, 'oneSeen': sum(1 for r in rows if r[1] == 'one'),
                     'flagged': how.startswith('the runs disagree')},
            'calibration': {'medianMetres': round(float(np.median(finite)), 2) if finite else None,
                            'maxMetres': round(float(np.max(finite)), 2) if finite else None, 'samples': len(finite),
                            'flagged': bool(finite) and float(np.median(finite)) > 2.0}}


def segment_quality(result, cut0, cut1):
    """Coverage of the frames a segment owns in the match: calibrated, under-constrained, the ball."""
    owned = [f for f in result['frames'] if cut0 <= f['frame'] < cut1]
    n = max(1, len(owned))
    conf = [((f.get('calibration') or {}).get('confidence') or 0.0) for f in owned]
    under = sum(1 for f, c in zip(owned, conf) if c < UNDERCONSTRAINED_CONF or not f.get('calibrated'))
    anchors = ((result.get('summary') or {}).get('pitch') or {}).get('registration', {}).get('anchors') or []
    auto = [a for a in anchors if a.get('source') == 'auto-midfield']
    summary = result.get('summary') or {}
    keyframes = (result.get('config') or {}).get('calibrations') or []
    return {'frames': len(owned), 'calibratedShare': round(sum(1 for f in owned if f.get('calibrated'))/n, 3),
            'discoveredKeyframes': sum(1 for c in keyframes if c.get('source') == 'discovery'),
            'underConstrainedShare': round(under/n, 3), 'meanCalibrationConfidence': round(float(np.mean(conf)) if conf else 0.0, 3),
            'keyframeAnchors': sum(1 for a in anchors if a.get('source') == 'landmarks'), 'automaticAnchors': len(auto),
            'autoLandmarks': (summary.get('profile') or {}).get('autoLandmarks'),
            'identifiedShare': summary.get('identifiedShare'), 'ballTracked': (summary.get('ball') or {}).get('trackedShare'),
            'smallSceneShare': (summary.get('profile') or {}).get('smallSceneShare')}


def write_view(path, data):
    path = Path(path)
    text = json.dumps(data, separators=(',', ':'), allow_nan=False, default=_default).encode('utf-8')
    temporary = path.with_name(path.name+'.tmp')
    with gzip.open(temporary, 'wb', compresslevel=GZIP_LEVEL) as out:
        out.write(text)
    mf._replace(temporary, path)
    return path.stat().st_size, len(text)


def _default(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f'{type(value).__name__} is not JSON serializable')


def load_result(folder):
    return mf.read_result(folder)


def assemble(manifest, progress=None, deep=True, intel_stage=True):
    """Assemble a match whose segments are all done; writes assembly/ and returns the match index. intel_stage: make
    the match's people and state too (step 7)."""
    started = time.time()
    say = progress or (lambda fraction, stage=None: None)
    segments = manifest.segments()
    planning.check(segments)
    folders = {}
    for s in segments:
        entry = manifest.entry(s.index)
        if entry['state'] != 'done':
            raise RuntimeError(f'Segment {s.index} is not done ({entry["state"]}).')
        folder = manifest.artifact_folder(s.index)
        problem = mf.verify_artifact(folder, s, deep=deep)
        if problem:
            raise RuntimeError(f'Segment {s.index}: {problem}.')
        folders[s.index] = folder
    say(.02, 'Stitching')
    video = manifest.data['video']
    fps, width, height = float(video['fps']), int(video['width']), int(video['height'])
    pad = int(round(float(manifest.data['plan']['padSeconds'])*fps))
    stitch = identities.resolve(segments, folders, pad, fps, progress=lambda p, stage=None: say(.02+.38*p, 'Stitching'))
    displays = {p['matchId']: p['display'] for p in stitch['roster']}
    roles = {p['matchId']: p['role'] for p in stitch['roster']}
    out_dir = manifest.folder/'assembly'
    (out_dir/'views').mkdir(parents=True, exist_ok=True)
    from . import analytics as match_analytics
    store = match_analytics.SampleStore()
    possession_parts = []
    slim_frames = {'time': [], 'view': []}   # the owned frames' time and camera view (the team shape's per-frame inputs)
    pitch_size = {}
    boundaries = {}
    index_segments = []
    timings = {'views': 0.0, 'samples': 0.0, 'possession': 0.0}

    def finish(s, result, cut1):
        """Write segment s's view (its next cut is known now), add its owned frames to the match samples and possession."""
        cut0 = boundaries[s.index]['cutFrame'] if s.index in boundaries else s.own0
        quality = segment_quality(result, cut0, cut1)
        times = {f['frame']: f['time'] for f in result['frames']}
        swap = stitch['mapping'][s.index]['swap']
        relabel(result, stitch['ids'].get(s.index, {}), swap, displays, roles, s, cut0, cut1, manifest.data['matchId'])
        t0 = time.time()
        store.add(result, cut0, cut1)
        for f in result['frames']:
            if cut0 <= f['frame'] < cut1:
                slim_frames['time'].append(f['time'])
                slim_frames['view'].append(f.get('view'))
        if not pitch_size:
            pitch_size.update((result.get('tactical') or {}).get('pitchSize') or {})
        size = (result.get('tactical') or {}).get('pitchSize') or {}
        sizes.add((size.get('length'), size.get('width')))
        t1 = time.time()
        try:
            possession_parts.append(match_analytics.possession_segment(result, cut0, cut1))
        except Exception as error:   # noqa: BLE001 - a segment without possession is a gap, not a failed match
            possession_parts.append({'version': None, 'available': False, 'reason': f'possession failed: {error}', 'fps': fps,
                                     't0': times.get(cut0, cut0/fps), 't1': times.get(cut1-1, cut1/fps), 'timeline': [],
                                     'changes': [], 'deadSpells': [], 'quality': None})
        t2 = time.time()
        name = f'{s.key}.json.gz'
        stored, raw = write_view(out_dir/'views'/name, result)
        timings['samples'] += t1-t0
        timings['possession'] += t2-t1
        timings['views'] += time.time()-t2
        index_segments.append({'index': s.index, 'key': s.key, 'period': s.period, 'own0': s.own0, 'own1': s.own1,
                               'proc0': s.proc0, 'proc1': s.proc1, 'cut0': cut0, 'cut1': cut1,
                               'start': times.get(cut0, cut0/fps), 'end': times.get(cut1, times.get(cut1-1, (cut1-1)/fps)+1/fps),
                               'firstTime': result['frames'][0]['time'], 'lastTime': result['frames'][-1]['time'],
                               'view': f'views/{name}', 'viewBytes': stored, 'jsonBytes': raw,
                               'teams': stitch['mapping'][s.index], 'quality': quality})

    sizes = set()   # every segment's pitch size: one physical pitch, so one size (the keyframes all set it)
    # One pass over the results, a segment behind: segment k is written once k+1 is loaded and the cut between them known.
    previous = None
    for k, s in enumerate(segments):
        result = load_result(folders[s.index])
        joined = previous is not None and previous[0].period == s.period and previous[0].own1 == s.own0
        if joined:
            size = (result.get('tactical') or {}).get('pitchSize') or {}
            boundaries[s.index] = boundary_overlap(previous[1], result, previous[0], s, fps, width, height, size)
        if previous is not None:
            finish(previous[0], previous[1], boundaries[s.index]['cutFrame'] if joined else previous[0].own1)
        previous = (s, result)
        say(.4+.58*(k+1)/len(segments), 'Writing the match')
    if previous is not None:
        finish(previous[0], previous[1], previous[0].own1)
    del previous
    people = match_analytics.roster_people(stitch['roster'])
    store.save(out_dir/'samples.npz', people)
    mf.write_json(out_dir/'together.json', stitch['together'], compact=True)
    write_view(out_dir/'frames.json.gz', slim_frames)
    from . import corrections as match_corrections
    suggested = match_corrections.suggestions(stitch['roster'], stitch['together'], manifest.data['periods'], fps)
    mf.write_json(out_dir/'possession.json', match_analytics.merge_possession(possession_parts, manifest.data['periods']), compact=True)
    periods = []
    for i, p in enumerate(manifest.data['periods']):
        direction = stitch['periods'].get(i) or {}
        periods.append(dict(p) | {'defendsLeft': direction.get('defendsLeft'), 'directionVotes': direction.get('votes'),
                                  'directionAgreement': direction.get('agreement'),
                                  'attacksRight': direction.get('defendsLeft')})
    seconds = sum((s['cut1']-s['cut0'])/fps for s in index_segments)
    under = sum(s['quality']['underConstrainedShare']*(s['cut1']-s['cut0']) for s in index_segments)/max(1, sum(s['cut1']-s['cut0'] for s in index_segments))
    index = {'version': VERSION, 'matchId': manifest.data['matchId'], 'source': {k: manifest.data['source'].get(k) for k in ('key', 'name')},
             'video': video, 'periods': periods, 'segments': index_segments,
             'boundaries': [boundaries[k] for k in sorted(boundaries)],
             'roster': stitch['roster'], 'suggestions': suggested, 'teams': {'prototypes': stitch['prototypes']},
             'stitching': {'boundaries': [{k: b[k] for k in ('previous', 'next', 'frames', 'seenA', 'seenB', 'accepted')} |
                                          {'pairs': len(b['pairs'])} for b in stitch['boundaries']],
                           'gapLinks': stitch['links'], 'people': len(stitch['roster'])},
             'coverage': {'seconds': round(seconds, 1), 'underConstrainedShare': round(under, 3)},
             'summaries': {'samples': 'samples.npz', 'possession': 'possession.json', 'frames': 'frames.json.gz'},
             'pitchSize': pitch_size or {'length': 105.0, 'width': 68.0},
             'pitchSizes': {'consistent': len(sizes) <= 1, 'values': sorted([list(v) for v in sizes], key=str)},
             'calibration': {'status': manifest.data.get('calibrationStatus'),
                             'discovery': {k: (manifest.data.get('discovery') or {}).get(k) for k in ('state', 'summary', 'seconds', 'sampled')}},
             'assembledAt': mf.now(), 'assemblySeconds': round(time.time()-started, 1),
             'assemblyTimings': {k: round(v, 1) for k, v in timings.items()}, 'rosterVersion': 1}
    mf.write_json(out_dir/'match.json', index, compact=True)
    manifest.data['assembly'] = {'state': 'complete', 'index': 'assembly/match.json', 'seconds': index['assemblySeconds'],
                                 'rosterVersion': 1, 'at': index['assembledAt']}
    manifest.save()
    if intel_stage:
        # 7. the match's people, places and state (fullmatch/intel.py): made now so the page opens with them; a failure
        # never fails the assembly (the page makes them on its first open and says why when it cannot)
        try:
            from . import intel
            intel.build(manifest.folder, video=(manifest.data.get('source') or {}).get('path'),
                        progress=lambda f, text: say(.97+.03*f, text))
            manifest.data['assembly']['intel'] = 'ready'
        except Exception as error:   # noqa: BLE001 - reported in the manifest
            manifest.data['assembly']['intel'] = f'not made: {error}'
        manifest.save()
    say(1.0, 'Ready')
    return index


def load_view(folder, view):
    with gzip.open(Path(folder)/view, 'rb') as handle:
        return json.loads(handle.read().decode('utf-8'))
