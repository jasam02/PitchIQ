"""Read-only diagnostics of one run: four exports and a short summary, to find out why a clip goes wrong
(no calibration, people rejected against a wrong pitch, the ball path on the wrong object, identities that
fall apart) without re-running anything.

    python tools/diagnose.py RUN_DIR [--window S E] [--window S E ...] [--out DIR]

RUN_DIR holds result.json (required) and registration.json (optional: its anchor list names the source and
the time of the anchor a carried or blended frame came from; without it the anchor is looked up by its time in
the result summary's anchor list). Nothing is written into the run directory; the files go to --out (default:
<system temp>/pitchiq-diagnose/<upload>-<run>):

  calibration.csv         per frame: the final registration (confidence, temporal, observability, source,
                          direction, the anchor's segment frame, time and source, its age, the forward/backward
                          disagreement), the frame's pitch mode and confidence, the live pitch model (level,
                          source, state, the midfield attempt, circle coverage), the last pitch tick (its age,
                          source, level, the four sides' status, curvature, vanishing point), the pitch size and
                          its source, the mapping counts, the stored homography's H[2,2] and whether it is
                          mis-oriented (the bottom centre of the image sees the ground behind the camera), and on
                          tick frames the final homography checked against the detected paint (matched markings,
                          median error in pixels, the sides verified by paint).
  counts.csv              per frame: people by state; mapped (all, live, kit-undecided), unresolved and above the
                          horizon; rejected detections by reason, and those whose foot the final homography puts
                          inside the pitch by at least a metre on a confident frame (confidence >= .5); people
                          suppressed as 'long outside the pitch' and those of them the final homography puts
                          inside (both blank where the frame was not checked: insideChecked 0); positioned live
                          people whose pitchConfidence is under .2 by box height, and how many of them the frame's
                          registration would still support (imputation's small-scene rule: confidence >= .35,
                          sigma <= 6 m); the frame's smallScene flag and debug lists when the result has them.
  ball_window.txt         per frame of each --window (the whole segment without one): the final ball, the first
                          pass and every ball candidate with its scores, reason and, when the result has them, its
                          source and the per-frame ball debug; the candidate the final ball sits on is marked.
  identity_fragments.csv  per identity, its local-track segments (track, first, last, frames, evidence kinds,
                          first-pass labels) with the gap before each and the identity events inside that gap
                          (re-identification, new identity, tracklet connect/assign/split, team change, team
                          change held, switch).
  summary.json            the numbers of the stdout summary.

The thresholds that define 'inside', 'confident' and 'supportable' are the constants below, so a before/after
pair of runs is always measured the same way."""
import argparse
import csv
import json
import math
import re
import statistics
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from soccer import field, imputation  # noqa: E402
from soccer.geometry import foot_point  # noqa: E402

INSIDE_METRES = 1.0        # a foot counts as inside the pitch when it is at least this far inside every line
CONFIDENT = .5             # registration confidence at which the final homography is trusted for 'inside' checks
LOW_POSITION = .2          # pitchConfidence below which the timeline drops an observation (imputation.estimate's min_confidence)
SUPPORTABLE_SIGMA = imputation.OBS_MAX_SIGMA   # m: a low-confidence position this precise is still supportable by its frame's registration
SUPPORTABLE_FRAME = imputation.OBS_MIN_FRAME   # when that registration is at least this confident
HEIGHT_BINS = ((0, 20, 'lt20'), (20, 40, '20_40'), (40, 1e9, 'ge40'))   # box height in source pixels
SEGMENT_GAP = .5           # s: an identity seen again on the same local track after more than this starts a new segment
GAP_SLACK = .1             # s: events this close to a gap's ends still belong to it
DECISION_LAG = 2.5         # s: events of the identity on another track, or of the track for another identity, this long after
                           # a segment starts still explain it (a decision relabels the frames back to the segment's start:
                           # broadcast 0-60 s, A-05 back on track 107 from 42.9 s, RE-ID EVENT at 44.2 s); events of the
                           # identity on the segment's own track count up to its last frame on that track (amateur 3:
                           # 156.7 s, RE-ID at 160.4 s; B-11 on track 327 from 171.6 s, RE-ID at 181.8 s)
MAX_GAP_EVENTS = 8         # events listed per segment (the rest are counted)
FINAL_MATCH = 1.5          # ball diameters: a candidate this close to the final ball can be the one it sits on (the nearest is)
ANCHOR_STEP = 1.5          # frames: an anchor of the summary's list is the frame's anchor when its time is this close to the expected one
LIVE = ('confirmed', 'uncertain')
FIELD_FILTER = ('audience', 'outside-pitch')   # the rejections the playable-area polygon decided (the others are size and confirmation)
STATES = ('confirmed', 'uncertain', 'unknown', 'candidate', 'nontarget', 'rejected')
IDENTITY_EVENTS = ('reid', 'new-identity', 'tracklet', 'merge', 'team-change', 'team-hold', 'switch', 'swap-corrected', 'deferred', 'reid-rejected')
KNOWN_CANDIDATE_KEYS = {'index', 'box', 'det', 'score', 'visual', 'status', 'scores', 'pxm', 'tracklet', 'player', 'entry',
                        'required', 'reason', 'text'}


# ---------- loading ----------
def load_run(run_dir):
    """(result, registration record or None, run name) of a run directory or a result.json path."""
    path = Path(run_dir)
    result_path = path/'result.json' if path.is_dir() else path
    result = json.loads(result_path.read_text(encoding='utf-8'))
    reg_path = result_path.parent/'registration.json'
    record = json.loads(reg_path.read_text(encoding='utf-8')) if reg_path.exists() else None
    return result, record, f'{result_path.parent.parent.name}-{result_path.parent.name}'


def pitch_size(result):
    return field.PitchSize.from_json((result.get('tactical') or {}).get('pitchSize') or {})


def anchor_table(record):
    """{log frame index: (source, time)} of the validated anchors, and the log offset of the segment."""
    if not record:
        return {}, 0
    return {int(a['index']): (a.get('source'), float(a['time'])) for a in record.get('anchors', [])}, int(record.get('offset') or 0)


def summary_anchors(result):
    """[(segment frame, time, source)] of the validated anchors the result's summary lists (registration.summary)."""
    reg = ((result.get('summary') or {}).get('pitch') or {}).get('registration') or {}
    return [(a.get('frame'), float(a['time']), a.get('source')) for a in reg.get('anchors', []) if a.get('time') is not None]


def expected_anchor_times(time, cal):
    """Where the frame's anchor lies in time, from its age and the carrying direction (both ways when blended)."""
    age = cal.get('anchorAge') or 0.0
    direction = cal.get('direction')
    return {'forward': [time-age], 'backward': [time+age], 'both': [time-age, time+age]}.get(direction, [time])


# ---------- homography helpers ----------
def stored_matrix(frame):
    H = (frame.get('calibration') or {}).get('H')
    if not isinstance(H, list) or len(H) != 9:
        return None
    M = np.asarray(H, np.float64).reshape(3, 3)
    return M if np.isfinite(M).all() else None


def misoriented(H, width, height):
    """True when the stored homography sees the ground in front of the camera at the bottom centre of the image
    only with the opposite sign (registration.homography_json lost the orientation), None when singular."""
    try:
        return bool(field.image_depth(H, [(width/2, .95*height)])[0] <= 0)
    except np.linalg.LinAlgError:
        return None


def final_matrix(frame, width, height):
    """The frame's final homography, re-oriented (field.orient), or None."""
    H = stored_matrix(frame)
    return None if H is None else field.orient(H, width, height)


def to_world(H, points_px):
    """World metres of image points (pixels) in front of the camera; NaN rows for points above the horizon."""
    pts = np.asarray(points_px, np.float64).reshape(-1, 2)
    out = np.full((len(pts), 2), np.nan)
    if not len(pts):
        return out
    try:
        front = field.image_depth(H, pts) > 1e-9
        if front.any():
            out[front] = field.unproject(H, pts[front])
    except np.linalg.LinAlgError:
        pass
    return out


def inside(world, size, margin=INSIDE_METRES):
    """Boolean per world point: inside the pitch by at least margin metres."""
    w = np.asarray(world, np.float64).reshape(-1, 2)
    with np.errstate(invalid='ignore'):
        return (w[:, 0] >= margin) & (w[:, 0] <= size.length-margin) & (w[:, 1] >= margin) & (w[:, 1] <= size.width-margin)


def inside_normalized(p, size, margin=INSIDE_METRES):
    return bool(inside([(p[0]*size.length, p[1]*size.width)], size, margin)[0])


# ---------- calibration.csv ----------
CALIBRATION_COLUMNS = ['time', 'calibrated', 'confidence', 'temporal', 'observability', 'source', 'direction', 'anchorFrame', 'anchorTime',
                       'anchorSource', 'anchorAge', 'disagreement', 'pitchCoordinates', 'pitchMode', 'pitchConfidence', 'liveLevel', 'liveSource',
                       'liveState', 'liveConfidence', 'midfield', 'circleCoverage', 'tickAge', 'tickSource', 'tickLevel', 'tickConfidence',
                       'edgeFar', 'edgeNear', 'edgeLeft', 'edgeRight', 'curvature', 'vanishing', 'sizeLength', 'sizeWidth', 'sizeSource',
                       'imageTracks', 'mapped', 'storedH22', 'misoriented', 'paintSegments', 'paintMatched', 'paintNames', 'paintErrorPx', 'paintSides']


def paint_check(H, size, markings, width, height):
    """The final homography against the paint the tick detected: (matched markings [{'name', 'error'}], the four
    sides' status with that match: 'line' where paint verifies the side)."""
    segs = [{'a': (m[0], m[1]), 'b': (m[2], m[3])} for m in markings or [] if isinstance(m, (list, tuple)) and len(m) == 4]
    matched = field.match_segments(H, size, segs, width, height) if segs else []
    return matched, field.side_status(H, size, width, height, matched)


def calibration_rows(result, record=None):
    W, Hh = result['video']['width'], result['video']['height']
    size = pitch_size(result)
    anchors, offset = anchor_table(record)
    listed, step = summary_anchors(result), ANCHOR_STEP/(result['video'].get('fps') or 25.0)
    rows, tick = [], None
    for f in result['frames']:
        if f.get('pitch'):
            tick = (f['time'], f['pitch'])
        cal = f.get('calibration') or {}
        live = f.get('pitchModel') or {}
        mid = live.get('midfield') or {}
        psize = live.get('size') or size.to_json()
        row = {'time': round(f['time'], 3), 'calibrated': int(bool(f.get('calibrated'))), 'confidence': cal.get('confidence'),
               'temporal': cal.get('temporal'), 'observability': cal.get('observability'), 'source': cal.get('source'),
               'direction': cal.get('direction'), 'anchorAge': cal.get('anchorAge'), 'disagreement': cal.get('disagreement'),
               'pitchCoordinates': f.get('pitchCoordinates'), 'pitchMode': f.get('pitchMode'), 'pitchConfidence': f.get('pitchConfidence'),
               'liveLevel': live.get('level'), 'liveSource': live.get('source'), 'liveState': live.get('state'), 'liveConfidence': live.get('confidence'),
               'midfield': ('taken' if mid.get('taken') else f"no: {mid.get('reason', '')}"[:80]) if mid else '',
               'circleCoverage': (live.get('circle') or {}).get('coverage'),
               'sizeLength': psize.get('length'), 'sizeWidth': psize.get('width'), 'sizeSource': psize.get('source'),
               'imageTracks': (f.get('mapping') or {}).get('imageTracks'), 'mapped': (f.get('mapping') or {}).get('mapped')}
        index = cal.get('anchor')
        if index is not None:
            if record is not None:
                row['anchorFrame'] = int(index)-offset
                if int(index) in anchors:
                    row['anchorSource'], at = anchors[int(index)]
                    row['anchorTime'] = round(at, 3)
            else:
                # Without registration.json the stored index counts the pre-roll (a log index, not a segment frame): the
                # anchor is the summary's at the time its age and the carrying direction give, or left blank.
                found = next(((frame, at, source) for t in expected_anchor_times(f['time'], cal) for frame, at, source in listed
                              if abs(at-t) <= step), None)
                if found is not None:
                    row['anchorFrame'], row['anchorTime'], row['anchorSource'] = found[0], round(found[1], 3), found[2]
            if row.get('anchorTime') is None and cal.get('anchorAge') is not None and cal.get('direction') in ('forward', 'backward', None):
                # In neither list: the anchor time still follows from the age and the carrying direction.
                row['anchorTime'] = round(expected_anchor_times(f['time'], cal)[0], 3)
        if tick is not None:
            t, p = tick
            edges = p.get('edges') or {}
            row.update({'tickAge': round(f['time']-t, 3), 'tickSource': p.get('source'), 'tickLevel': p.get('level'), 'tickConfidence': p.get('confidence'),
                        'edgeFar': edges.get('far'), 'edgeNear': edges.get('near'), 'edgeLeft': edges.get('left'), 'edgeRight': edges.get('right'),
                        'curvature': p.get('curvature'), 'vanishing': int(bool(p.get('vanishing')))})
        stored = stored_matrix(f)
        if stored is not None:
            row['storedH22'] = round(float(stored[2, 2]), 6)
            flag = misoriented(stored, W, Hh)
            row['misoriented'] = '' if flag is None else int(flag)
        Hf = final_matrix(f, W, Hh)
        if f.get('pitch') and Hf is not None:
            matched, sides = paint_check(Hf, size, f['pitch'].get('markings'), W, Hh)
            row['paintSegments'] = len(f['pitch'].get('markings') or [])
            row['paintMatched'] = len(matched)
            row['paintNames'] = ';'.join(sorted({m['name'] for m in matched}))
            row['paintErrorPx'] = round(statistics.median(m['error'] for m in matched), 2) if matched else None
            row['paintSides'] = ';'.join(k for k in ('far', 'near', 'left', 'right') if sides.get(k) == 'line')
        rows.append(row)
    return rows


# ---------- counts.csv ----------
def counts_rows(result):
    """Per frame counts (dicts) and the rejection reasons seen in the run (for the column order)."""
    W, Hh = result['video']['width'], result['video']['height']
    size = pitch_size(result)
    rows, reasons = [], Counter()
    for f in result['frames']:
        people = f.get('people', [])
        cal = f.get('calibration') or {}
        confidence = cal.get('confidence') or 0.0
        Hf = final_matrix(f, W, Hh) if confidence >= CONFIDENT else None
        row = {'time': round(f['time'], 3)}
        states = Counter(p.get('state') for p in people)
        for s in STATES:
            row[s] = states.get(s, 0)
        placed = [p for p in people if isinstance(p.get('pitch'), list)]
        row['mapped'] = len(placed)
        row['mappedLive'] = sum(1 for p in placed if p.get('state') in LIVE)
        row['unknownMapped'] = sum(1 for p in placed if p.get('state') == 'unknown')
        row['unresolved'] = sum(1 for p in people if p.get('pitchStatus') == 'unresolved')
        row['horizon'] = sum(1 for p in people if p.get('pitchStatus') == 'horizon')
        row['imageTracks'] = (f.get('mapping') or {}).get('imageTracks')
        rejected = f.get('rejected') or []
        row['rejected'] = len(rejected)
        by_reason = Counter(r.get('reason') or 'unknown' for r in rejected)
        reasons.update(by_reason)
        for k, v in by_reason.items():
            row[f'rejected:{k}'] = v
        # The 'inside the final pitch' checks need a trusted final homography: on other frames they are None (blank), never
        # 0, so that a run that gains a calibration does not seem to gain rejections inside the pitch.
        checked = Hf is not None
        inside_reasons = Counter()
        if checked and rejected:
            feet = [(foot_point(r['box'])[0]*W, foot_point(r['box'])[1]*Hh) for r in rejected]
            for r, ok in zip(rejected, inside(to_world(Hf, feet), size)):
                if ok:
                    inside_reasons[r.get('reason') or 'unknown'] += 1
        row['insideChecked'] = int(checked)
        row['rejectedInside'] = sum(inside_reasons.values()) if checked else None
        row['rejectedInsideFieldFilter'] = sum(inside_reasons.get(k, 0) for k in FIELD_FILTER) if checked else None
        row['rejectedInsideReasons'] = ';'.join(f'{k}:{v}' for k, v in sorted(inside_reasons.items()))
        row['_insideReasons'] = inside_reasons   # for the summary (not a column)
        outside = [p for p in people if 'long outside the pitch' in (p.get('reason') or '')]
        row['longOutside'] = len(outside)
        row['longOutsideInside'] = sum(1 for p in outside if isinstance(p.get('pitch'), list) and inside_normalized(p['pitch'], size)) if checked else None
        low = Counter()
        positioned = Counter()
        supportable = 0
        for p in placed:
            if p.get('state') not in LIVE:
                continue
            h = p['box'][3]*Hh
            name = next(n for lo, hi, n in HEIGHT_BINS if lo <= h < hi)
            positioned[name] += 1
            if (p.get('pitchConfidence') or 0.0) < LOW_POSITION:
                low[name] += 1
                sigma = p.get('pitchSigma')
                if confidence >= SUPPORTABLE_FRAME and isinstance(sigma, list) and sigma and max(sigma) <= SUPPORTABLE_SIGMA:
                    supportable += 1
        for _, _, name in HEIGHT_BINS:
            row[f'positioned_{name}'] = positioned.get(name, 0)
            row[f'lowConfidence_{name}'] = low.get(name, 0)
        row['lowConfidenceSupportable'] = supportable
        profile = f.get('profile') or {}
        row['smallScene'] = '' if 'smallScene' not in profile else int(bool(profile['smallScene']))
        row['dropped'] = len(f['dropped']) if isinstance(f.get('dropped'), list) else ''
        rows.append(row)
    return rows, reasons


def counts_columns(reasons):
    return (['time', *STATES, 'mapped', 'mappedLive', 'unknownMapped', 'unresolved', 'horizon', 'imageTracks', 'rejected']
            + [f'rejected:{k}' for k in sorted(reasons)]
            + ['insideChecked', 'rejectedInside', 'rejectedInsideFieldFilter', 'rejectedInsideReasons', 'longOutside', 'longOutsideInside']
            + [f'{kind}_{name}' for kind in ('positioned', 'lowConfidence') for _, _, name in HEIGHT_BINS]
            + ['lowConfidenceSupportable', 'smallScene', 'dropped'])


# ---------- ball_window.txt ----------
def _px(point, W, Hh):
    return f'({point[0]*W:5.0f},{point[1]*Hh:5.0f})' if point else '(    -,    -)'


def _num(v, digits=2):
    return '-' if v is None else f'{v:.{digits}f}'


def _centre(box):
    return (box[0]+box[2]/2, box[1]+box[3]/2)


def final_candidate(ball, candidates, W, Hh):
    """The position in `candidates` of the one the final ball sits on, or None: of those within FINAL_MATCH ball diameters
    of the final ball, the nearest of status 'ball' when there is one, else the nearest (candidates of one ball often
    overlap: a tracklet's detection and a tile's)."""
    c, box = ball.get('center'), ball.get('box')
    if not c or not box:
        return None
    reach = FINAL_MATCH*max(box[2]*W, box[3]*Hh, 1.0)
    near = []
    for k, cand in enumerate(candidates):
        x, y = _centre(cand.get('box') or [0, 0, 0, 0])
        d = math.hypot((x-c[0])*W, (y-c[1])*Hh)
        if d <= reach:
            near.append((cand.get('status') != 'ball', d, k))
    return min(near)[2] if near else None


def ball_window_lines(result, windows=None):
    """Text lines: per frame of each window (start, end) the final ball, the first pass and every candidate."""
    W, Hh = result['video']['width'], result['video']['height']
    frames = result['frames']
    if not frames:
        return ['(no frames)']
    windows = windows or [(frames[0]['time'], frames[-1]['time'])]
    out = []
    for start, end in windows:
        rows = [f for f in frames if start-1e-6 <= f['time'] <= end+1e-6]
        out.append(f'== {start:.2f}-{end:.2f} s ({len(rows)} frames) ==')
        for f in rows:
            shown = {q['track']: q.get('display') or q.get('id') for q in f.get('people', [])}
            b = f.get('ball') or {}
            c = b.get('center')
            near = b.get('nearTrack')
            p = b.get('pitch')
            out.append(f"{f['time']:8.3f}  final {b.get('state', '-'):<8}{(b.get('phase') or ''):<11}{(b.get('origin') or ''):<15}{(b.get('motion') or ''):<12}"
                       f"px {_px(c, W, Hh)}  visual {_num(b.get('visual'))} trajectory {_num(b.get('trajectory'))} identity {_num(b.get('identity'))}"
                       + (f"  near T{near} ({shown.get(near, '?')})" if near is not None else '')
                       + (f'  pitch ({p[0]:.3f}, {p[1]:.3f})' if p else ''))
            fp = b.get('firstPass')
            if fp:
                out.append(f"          first pass {fp.get('state', '-'):<8}{(fp.get('phase') or ''):<11}confidence {_num(fp.get('confidence'))}"
                           f"  track {fp.get('track')}  px {_px(fp.get('center'), W, Hh)}")
            debug = f.get('ballDebug')
            if debug:
                out.append('          ball debug ' + json.dumps(debug, sort_keys=True)[:400])
            candidates = f.get('ballCandidates') or []
            on = final_candidate(b, candidates, W, Hh)
            for k, cand in enumerate(candidates):
                box = cand.get('box') or [0, 0, 0, 0]
                centre = _centre(box)
                s = cand.get('scores') or {}
                player = cand.get('player') or {}
                on_final = k == on
                extra = {k: v for k, v in cand.items() if k not in KNOWN_CANDIDATE_KEYS}
                out.append(f"          #{cand.get('index', '?'):<3} px {_px(centre, W, Hh)} {box[2]*W:3.0f}x{box[3]*Hh:<3.0f} det {_num(cand.get('det'))} "
                           f"visual {_num(cand.get('visual', cand.get('score')))} {cand.get('status', '-'):<9} {(cand.get('reason') or '-'):<15}"
                           f"attach {_num(s.get('attach'))} other {'yes' if s.get('otherObject') else 'no '} line {_num(s.get('line'))} "
                           f"static {_num(s.get('static'))} motion {_num(s.get('motion'))}"
                           + (f" player T{player.get('track')} ({shown.get(player.get('track'), '?')}){' lower' if player.get('lower') else ''}" if player else '')
                           + (f" tracklet {cand.get('tracklet')}" if cand.get('tracklet') is not None else '')
                           + (f" entry {cand.get('entry')}" if cand.get('entry') else '')
                           + (f" needs {cand.get('required')} m/s" if cand.get('required') is not None else '')
                           + (' ' + ' '.join(f'{k}={json.dumps(v, sort_keys=True)}' for k, v in sorted(extra.items())) if extra else '')
                           + ('  <= FINAL' if on_final else ''))
    return out


# ---------- identity_fragments.csv ----------
FRAGMENT_COLUMNS = ['id', 'display', 'team', 'segment', 'track', 'first', 'last', 'frames', 'gapBefore', 'bridge', 'firstEvidence', 'evidence',
                    'firstPassLabels', 'gapEvents']
OFFLINE_EVIDENCE = ('connected offline', 'assigned offline', 'merged offline')


def _event_name(e):
    if e['kind'] == 'tracklet':
        head = e.get('message', '').split('\n', 1)[0].strip()
        return 'tracklet-' + head.replace('TRACKLET', '').strip().lower().replace(' ', '-') if head.startswith('TRACKLET') else 'tracklet'
    return e['kind']


def _bridge(pid, segment, previous, gap_events):
    """What joined a later segment to its identity: the deciding event of the identity, else the offline evidence on
    the segment, else the same local track continuing, else 'unexplained'."""
    mine = [_event_name(e) for e in gap_events if e.get('playerId') == pid]
    for name in ('reid', 'swap-corrected', 'tracklet-connect', 'tracklet-assigned', 'merge', 'team-change', 'team-hold', 'new-identity'):
        if name in mine:
            return name
    if segment['firstEvidence'] in OFFLINE_EVIDENCE:
        return segment['firstEvidence']
    if previous is not None and previous['track'] == segment['track']:
        return 'same track'
    return 'unexplained'


def identity_fragments(result):
    """Rows of identity_fragments.csv: every identity's local-track segments in time order."""
    open_, segments = {}, defaultdict(list)
    displays, teams = defaultdict(Counter), {}
    for f in result['frames']:
        t = f['time']
        for p in f.get('people', []):
            pid = p.get('id')
            if not pid:
                continue
            displays[pid][p.get('display') or pid] += 1
            if p.get('team'):
                teams[pid] = p['team']
            key = (pid, p['track'])
            s = open_.get(key)
            if s is None or t-s['last'] > SEGMENT_GAP:
                s = {'track': p['track'], 'first': t, 'last': t, 'frames': 0, 'evidence': Counter(), 'labels': Counter(),
                     'firstEvidence': p.get('evidence') or ''}
                open_[key] = s
                segments[pid].append(s)
            s['last'] = t
            s['frames'] += 1
            s['evidence'][p.get('evidence') or ''] += 1
            if p.get('firstPassLabel'):
                s['labels'][p['firstPassLabel']] += 1
    events = sorted((e for e in result.get('events', []) if e.get('kind') in IDENTITY_EVENTS), key=lambda e: e['time'])   # listed in time order
    rows = []
    for pid in sorted(segments):
        prev_end, previous = None, None
        for n, s in enumerate(sorted(segments[pid], key=lambda s: s['first']), 1):
            # The gap before the segment, plus the time a decision may lag the frames it relabels.
            lo = (prev_end if prev_end is not None else s['first']-SEGMENT_GAP)-GAP_SLACK
            lag = min(s['last'], s['first']+DECISION_LAG)+GAP_SLACK
            own = max(x['last'] for x in segments[pid] if x['track'] == s['track'])+GAP_SLACK   # the identity's last frame on this track
            in_gap = [e for e in events if lo <= e['time'] and ((e.get('playerId') == pid and e.get('track') == s['track'] and e['time'] <= own)
                                                                 or ((e.get('playerId') == pid or e.get('track') == s['track']) and e['time'] <= lag))]
            listed = [f"{_event_name(e)}@{e['time']:.2f}(T{e.get('track')})" for e in in_gap]
            rows.append({'id': pid, 'display': displays[pid].most_common(1)[0][0], 'team': teams.get(pid, ''), 'segment': n, 'track': s['track'],
                         'first': round(s['first'], 3), 'last': round(s['last'], 3), 'frames': s['frames'],
                         'gapBefore': '' if prev_end is None else round(s['first']-prev_end, 3),
                         'bridge': '' if previous is None else _bridge(pid, s, previous, in_gap), 'firstEvidence': s['firstEvidence'],
                         'evidence': ';'.join(f'{k}:{v}' for k, v in s['evidence'].most_common()),
                         'firstPassLabels': ';'.join(f'{k}:{v}' for k, v in s['labels'].most_common()),
                         'gapEvents': ';'.join(listed[:MAX_GAP_EVENTS]) + (f';+{len(listed)-MAX_GAP_EVENTS} more' if len(listed) > MAX_GAP_EVENTS else '')})
            prev_end = s['last'] if prev_end is None else max(prev_end, s['last'])
            previous = s
    return rows


# ---------- summary ----------
def _share(n, d):
    return round(n/d, 4) if d else None


def summarize(result, cal, counts, fragments):
    frames = result['frames']
    n = max(1, len(frames))
    s = {'frames': len(frames), 'video': result['video'].get('name'), 'size': f"{result['video']['width']}x{result['video']['height']}",
         'start': frames[0]['time'] if frames else None, 'end': frames[-1]['time'] if frames else None}
    calibrated = sum(r['calibrated'] for r in cal)
    ticks = [r for r in cal if r.get('paintSegments') is not None]
    errors = [r['paintErrorPx'] for r in ticks if r.get('paintErrorPx') is not None]
    reg = ((result.get('summary') or {}).get('pitch') or {}).get('registration') or {}
    s['calibration'] = {'calibratedShare': _share(calibrated, n), 'confidentShare': _share(sum(1 for r in cal if (r['confidence'] or 0) >= CONFIDENT), n),
                        'sources': dict(Counter(r['source'] or 'none' for r in cal)), 'anchors': dict(Counter(a.get('source') for a in reg.get('anchors', []))),
                        'misorientedFrames': sum(1 for r in cal if r.get('misoriented') == 1),
                        'framesWithStoredH': sum(1 for r in cal if r.get('storedH22') is not None),
                        'paintTicks': len(ticks), 'paintVerifiedShare': _share(sum(1 for r in ticks if r['paintMatched']), len(ticks)),
                        'paintMedianErrorPx': round(statistics.median(errors), 2) if errors else None,
                        'midfieldTaken': sum(1 for r in cal if r.get('midfield') == 'taken'),
                        'midfieldRefusals': dict(Counter(re.sub(r'[0-9.]+', '#', r['midfield'][4:])[:60] for r in cal
                                                         if str(r.get('midfield', '')).startswith('no:')).most_common(4)),
                        'sizeSources': dict(Counter(r.get('sizeSource') for r in cal))}
    tot = Counter()
    for r in counts:
        for k, v in r.items():
            if isinstance(v, int) and not isinstance(v, bool) and k != 'time':
                tot[k] += v
    s['peoplePerFrame'] = {k: round(tot[k]/n, 2) for k in (*STATES, 'mapped', 'mappedLive', 'unknownMapped', 'unresolved', 'horizon')}
    # The inside checks count over the frames that were checked (unchecked frames are None in the rows, skipped by tot).
    checked = tot['insideChecked']
    s['rejected'] = {'perFrame': round(tot['rejected']/n, 2), 'byReason': {k[9:]: v for k, v in sorted(tot.items()) if k.startswith('rejected:')},
                     'insideCheckedFrames': checked, 'insideFinalPitch': tot['rejectedInside'] if checked else None,
                     'insideFinalPitchFieldFilter': tot['rejectedInsideFieldFilter'] if checked else None,
                     'insideByReason': dict(sum((r['_insideReasons'] for r in counts), Counter()))}
    s['longOutside'] = {'personFrames': tot['longOutside'], 'insideCheckedFrames': checked, 'insideFinalPitch': tot['longOutsideInside'] if checked else None}
    s['lowConfidencePositions'] = {name: {'low': tot[f'lowConfidence_{name}'], 'positioned': tot[f'positioned_{name}'],
                                          'share': _share(tot[f'lowConfidence_{name}'], tot[f'positioned_{name}'])} for _, _, name in HEIGHT_BINS}
    positioned = sum(tot[f'positioned_{name}'] for _, _, name in HEIGHT_BINS)
    s['lowConfidencePositions']['supportable'] = tot['lowConfidenceSupportable']
    s['lowConfidencePositions']['supportableShare'] = _share(tot['lowConfidenceSupportable'], positioned)
    small = [r['smallScene'] for r in counts if r['smallScene'] != '']
    s['smallSceneShare'] = _share(sum(small), len(small)) if small else None
    states, statuses, reasons, sources, per_frame = Counter(), Counter(), Counter(), Counter(), []
    for f in frames:
        states[(f.get('ball') or {}).get('state')] += 1
        cands = f.get('ballCandidates') or []
        per_frame.append(len(cands))
        for c in cands:
            statuses[c.get('status')] += 1
            if c.get('reason'):
                reasons[c['reason']] += 1
            sources[str(c.get('source', 'detector'))] += 1   # no 'source' key: the whole-frame detector (soccer/ball.py)
    s['ball'] = {'states': {k: round(v/n, 3) for k, v in states.items()}, 'candidatesPerFrame': round(sum(per_frame)/n, 2),
                 'maxCandidates': max(per_frame) if per_frame else 0, 'byStatus': dict(statuses), 'rejections': dict(reasons.most_common()),
                 'bySource': dict(sources) or None}
    ids = {r['id'] for r in fragments}
    kinds = Counter(e.get('kind') for e in result.get('events', []))
    s['identities'] = {'identities': len(ids), 'segments': len(fragments), 'segmentsPerIdentity': round(len(fragments)/len(ids), 2) if ids else None,
                       'bridges': dict(Counter(r['bridge'] for r in fragments if r['segment'] > 1).most_common()),
                       'events': {k: kinds[k] for k in IDENTITY_EVENTS if kinds[k]},
                       'overlapping': sum(1 for r in fragments if r['gapBefore'] != '' and r['gapBefore'] < 0)}
    return s


def print_summary(name, s, out_dir):
    c, p, rj = s['calibration'], s['peoplePerFrame'], s['rejected']
    low = s['lowConfidencePositions']
    lines = [f"run {name}: {s['video']} {s['size']}, {s['frames']} frames, {s['start']}-{s['end']} s",
             f"calibration: calibrated {c['calibratedShare']}, confident (>= {CONFIDENT}) {c['confidentShare']}, sources {c['sources']}, anchors {c['anchors']}",
             f"  stored H mis-oriented on {c['misorientedFrames']} of {c['framesWithStoredH']} frames; paint check on {c['paintTicks']} ticks: "
             f"verified {c['paintVerifiedShare']}, median error {c['paintMedianErrorPx']} px",
             f"  midfield anchor taken on {c['midfieldTaken']} frames; refusals {c['midfieldRefusals']}; size sources {c['sizeSources']}",
             'people per frame: ' + ', '.join(f'{k} {v}' for k, v in p.items()),
             f"rejected per frame {rj['perFrame']} {rj['byReason']}",
             f"  inside the final pitch (checked on the {rj['insideCheckedFrames']} frames with confidence >= {CONFIDENT}; >= {INSIDE_METRES} m): "
             f"{rj['insideFinalPitch']} {rj['insideByReason']}, of them by the field filter {rj['insideFinalPitchFieldFilter']}",
             f"long outside the pitch: {s['longOutside']['personFrames']} person-frames, {s['longOutside']['insideFinalPitch']} inside the final pitch",
             'live positions with pitchConfidence < .2: ' + ', '.join(f"{name} px {low[name]['low']}/{low[name]['positioned']}" for _, _, name in HEIGHT_BINS)
             + f"; supportable (frame >= {SUPPORTABLE_FRAME}, sigma <= {SUPPORTABLE_SIGMA} m) {low['supportable']} ({low['supportableShare']} of positioned)",
             f"small-scene frames: {s['smallSceneShare']}",
             f"ball: states {s['ball']['states']}, candidates per frame {s['ball']['candidatesPerFrame']} (max {s['ball']['maxCandidates']}), "
             f"by status {s['ball']['byStatus']}" + (f", by source {s['ball']['bySource']}" if s['ball']['bySource'] else ''),
             f"  rejections {s['ball']['rejections']}",
             f"identities: {s['identities']['identities']} with {s['identities']['segments']} segments ({s['identities']['segmentsPerIdentity']} each); "
             f"later segments bridged by {s['identities']['bridges']}; overlapping {s['identities']['overlapping']}",
             f"  identity events {s['identities']['events']}",
             f'files: {out_dir}']
    print('\n'.join(lines))


# ---------- output ----------
def write_csv(path, rows, columns):
    with open(path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        for r in rows:
            writer.writerow({k: ('' if r.get(k) is None else r.get(k)) for k in columns})


def default_out(name):
    return Path(tempfile.gettempdir())/'pitchiq-diagnose'/name


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('run_dir', help='run directory (or its result.json)')
    ap.add_argument('--window', nargs=2, type=float, action='append', metavar=('START', 'END'), help='ball window in video seconds (repeatable)')
    ap.add_argument('--out', help='output directory (never inside the run directory)')
    a = ap.parse_args(argv)
    result, record, name = load_run(a.run_dir)
    run_dir = Path(a.run_dir) if Path(a.run_dir).is_dir() else Path(a.run_dir).parent
    out = Path(a.out) if a.out else default_out(name)
    if out.resolve() == run_dir.resolve() or run_dir.resolve() in out.resolve().parents:
        ap.error('the run directory is read-only evidence: choose an --out outside it')
    out.mkdir(parents=True, exist_ok=True)
    cal = calibration_rows(result, record)
    counts, reasons = counts_rows(result)
    fragments = identity_fragments(result)
    write_csv(out/'calibration.csv', cal, CALIBRATION_COLUMNS)
    write_csv(out/'counts.csv', counts, counts_columns(reasons))
    write_csv(out/'identity_fragments.csv', fragments, FRAGMENT_COLUMNS)
    (out/'ball_window.txt').write_text('\n'.join(ball_window_lines(result, [tuple(w) for w in a.window] if a.window else None))+'\n', encoding='utf-8')
    summary = summarize(result, cal, counts, fragments)
    (out/'summary.json').write_text(json.dumps(summary, indent=1), encoding='utf-8')
    print_summary(name, summary, out)
    return 0


if __name__ == '__main__':
    sys.exit(main())
