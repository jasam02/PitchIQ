"""Calibration discovery: pitch anchors for every segment of a match, found automatically before tracking and kept only
when the one camera the match is filmed from vouches for them.

Why: a segment run without a keyframe within 15 s anchors its pitch on the first tick the automatic midfield route
passes, for good. On CAN-ISL's high tactical camera that tick came from a partial arc of the centre circle (436.6 s:
focal 330 px, camera 3.4 m high, the centre spot 21 m off) and the wrong pitch held at MEDIUM all segment; taking the
first passing anchor of each of the match's 33 segments puts at least 4 of them 14-21 m off. Hand keyframes fix it (3
gave HIGH within ~1 m), but a 90-minute match would need 35-60 of them.

The stage (solve_chunk, run by the scheduler's workers on the CPU): sampled frames, STEP_S apart and never across a
period, are each solved with the tracker's keyless route (pitch.PitchDetector.analyze -> pitch.find_circle on the grass
-> midfield.touchline_candidates -> midfield.solve(allow_pinhole=True)), without people (no detector), and every anchor's
homography is decomposed into a camera (camera_of: square pixels, the principal point at the image centre); sampled
frames GAPS apart are registered to each other by SIFT (nothing masked: matches that stay put while the scene moves are
burned-in overlays and are dropped). discover() keeps an anchor only when all three hold:

  1. Its camera is the match's. A stadium camera pans and zooms about one place, so every correct anchor decomposes to
     the same ground position and height; the consensus is the trimmed median of the pinhole anchors' cameras, and an
     anchor passes within POS_TOL_M (ground) and HEIGHT_TOL_M of it. On CAN-ISL (96.5 min at 2 s, every anchor measured
     against 9 hand keyframes carried to its frame by SIFT registration) the anchors more than 3 m off (8.7%) all had
     their camera 20.4 m or more from the consensus (5th percentile 37 m), the correct ones at most 4.8 m (95th
     percentile), and every anchor closer than 20 m was within 2.35 m.
  2. A neighbour agrees: a sampled frame GAPS away whose own anchor, carried into this frame by their registration,
     lies within AGREE_M (median over the visible pitch, registration._disagreement). Not safe alone: wrong solutions
     repeat on consecutive frames (the same partial arc) and agree with each other.
  3. Observability MIN_OBS or more (midfield's score). The amateur pole camera and the Veo give nothing above .58.

Measured with this module (CAN-ISL's two halves, 10-2854 s and 2885-5791.6 s: 2876 sampled frames, 1722 anchors; the
broadcast's 5 min: 152 frames, 119 anchors), against that reference (it agrees with a second truth view to .21 m
median): of all anchors 144 of 1679 measured (8.6%; the broadcast 9 of 97) are more than 3 m off; the camera rule
alone leaves none (largest error 1.20 m), the neighbour rule alone 18 (1.2%). With all three CAN-ISL keeps 1259 anchors,
2-49 per segment: none of the 1243 measured more than 1.5 m off (median .26 m, p99 .58, max .92); the broadcast keeps 93
(76 measured: median .26, max .97). Further from the circle the error grows: the 90th percentile over a frame's
visible pitch has a median of .49 m but reached 2.35 m on wide views (a penalty box 45 m from the circle).

The keyframes are placed for reach. A segment run given a keyframe turns its automatic calibration off
(pipeline.auto_landmarks), and the keyframe's calibration carries only so far without other evidence: its confidence
fell under registration.MIN_POSITION 26.7 s before a lone keyframe in each of CAN-ISL's segments 20, 23 and 30 (the
06:21 match), whose first halves stayed uncalibrated although they were plain wide views. So each pick is the kept
anchor that brings the most of the segment's still unreached seconds within REACH_S of a keyframe (a user keyframe
reaches too; the better score on a tie), MIN_GAP_S from the others, up to PER_SEGMENT, none within MANUAL_REACH_S of a
user keyframe (the user's wins). Each is fused with the kept neighbours that agree with it (FUSE_MIN homographies at
least: the anchor and two others carried into its frame, the per-point median of their ground positions over an image
grid, refitted as one homography; only agreeing ones, so that a wrong registration never takes part), which cut the
frames whose 90th percentile exceeded 1.5 m from 8 to 1 in the research; a kept anchor that cannot be fused is taken
as it is only for the seconds no fused one reaches. The first rule (one keyframe per third of a segment, 40 s apart,
the best observability first: 86 keyframes on CAN-ISL, 82 measured: median .24 m, p90 .35, max .51) left 9.1% of the
match's owned frames uncalibrated, 309 s of it in spans with kept anchors inside. This one gives CAN-ISL 141
keyframes (136 measured: median .25 m, max .77, the 90th percentile over the frame 1.27 m at worst); three of its
segments run again with them were calibrated over 100%, 100% and 72% of their owned frames instead of 50%, 86% and
49%, with 3.4% of the calibrated reference samples more than 1.5 m off instead of 4.9% (none over 3 m). A keyframe's
points are the landmarks the homography puts inside the image, in the server's calibration-keyframe format; its time
is the decoder's own timestamp of the frame (the pipeline applies a keyframe within .12 s of a frame's stamp).

Refused, with no keyframe at all, when the pinhole anchors do not agree on one camera (CONSENSUS_SHARE of them within
the tolerances: CAN-ISL .86, the broadcast .87; a moving or a multi-camera production splits them) or no kept anchor
reaches MIN_OBS (the amateur and Veo footage): the plain reason tells the user to add keyframes.

Cost (one process, OpenCV single-threaded as in a GPU worker, the machine shared with other runs): .37 s per sampled
frame with its registrations (.27 s with OpenCV's own threads, as in a light worker): CAN-ISL's 2876 frames are 18 min
of CPU, 6.4 min of wall time on 4 workers.

Not claimed: the far end of wide views (above), cameras that move (a Veo is fixed but its lens is not rectilinear), and
any accuracy finer than the ~.2-.3 m of the hand keyframes the numbers were measured against.
"""
import math

import cv2
import numpy as np

from soccer import calibration as keyframe_solver, field, midfield, registration
from soccer.geometry import clip_unit, polygon_area
from soccer.pitch import PitchDetector, find_circle

VERSION = 'discovery-1'
STEP_S = 2.0           # seconds between sampled frames: 1 per 5 s found 40% of the anchors (18 kept per segment, 1 segment
                       # under 3 with the observability floor) at the same accuracy
GAPS = (1, 2, 3)       # sampled frames apart that are registered (2-6 s at STEP_S): the neighbours the rules and the fusion use
CHUNK = 48             # sampled frames a discovery job owns (~96 s of video, ~18 s of CPU)
POS_TOL_M = 6.0        # an anchor's camera ground position this close to the match's (good <= 4.8 m p95, false >= 20.4 m)
HEIGHT_TOL_M = 2.5     # ... and its height (good <= 1.45 m p95)
AGREE_M = 1.0          # a neighbour's carried anchor within this (median metres over the frame) agrees (good: .05 m median)
MIN_OBS = .6           # midfield's observability an anchor needs (false ones .21-.40; amateur and Veo <= .58)
CONSENSUS_SHARE = .6   # pinhole anchors within the tolerances of the consensus, at least (CAN-ISL .86, broadcast .87)
REACH_S = 25.0         # seconds a keyframe's calibration is counted to carry each way (measured 26.7 s back, above)
PER_SEGMENT = 6        # synthetic keyframes per segment at most (a 186 s segment needs 4 at REACH_S)
MIN_GAP_S = 20.0       # ... at least this far apart
MANUAL_REACH_S = 15.0  # a synthetic keyframe this close to a user keyframe is dropped (pipeline.ROLL_SECONDS)
MIN_PINHOLE = 5        # pinhole anchors needed to look for a consensus at all
TRIM_POS_M, TRIM_HEIGHT_M = 8.0, 4.0   # the consensus median is re-taken over the anchors this close to it (3 times)
FUSE_MIN = 3           # homographies a keyframe is fused from (the anchor and 2 agreeing kept neighbours)
PAIR_OVERLAP = .3      # share of a frame its registered neighbour must cover to be compared with it
MARGIN_PX = 8.0        # a keyframe's landmarks lie at least this far inside the image
MIN_POINTS = 4
WORK_W = 960           # registration width (camera.WORK_W)
FEATURES = 2500        # SIFT features per frame (camera.REGISTER_FEATURES)
RATIO = .75            # Lowe ratio
MIN_INLIERS = 80
RANSAC_PX = 2.0        # at WORK_W
MAX_RMS_PX = 2.5       # at WORK_W
STILL_PX, MOVING_PX = .75, 4.0   # WORK_W px: matches that stay put while the median one moves are burned-in overlays (score,
                       # clock, logos); nothing is masked: a broadcast frame's texture is mostly the stands at the top, and
                       # masking the top quarter (camera.CameraMotion's convention) left 7 of 114 broadcast pairs registered;
                       # unmasked: 309 of 309, and CAN-ISL 4135 of 4138
GRAB_LIMIT = 250       # frames: a sample this close after the last one is reached by decoding forward, not by seeking


def _json(v):
    if isinstance(v, dict):
        return {str(k): _json(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_json(x) for x in v]
    if isinstance(v, np.ndarray):
        return _json(v.tolist())
    if isinstance(v, (np.floating, float)):
        v = float(v)
        return v if math.isfinite(v) else None
    if isinstance(v, (np.integer, np.bool_)):
        return v.item()
    return v


# ---------- sampling ----------
def sample(periods, fps, step_s=STEP_S):
    """Sorted unique frame indices STEP_S apart inside each period ((frame0, frame1) half-open), from its first frame."""
    out = set()
    step = float(step_s)*float(fps)
    for f0, f1 in periods:
        f0, f1 = int(f0), int(f1)
        k = 0
        while True:
            index = f0+int(round(k*step))
            if index >= f1:
                break
            out.add(index)
            k += 1
    return sorted(out)


def _runs(indices, periods):
    """The indices split into their periods (each run sorted); without periods, split where the spacing jumps (more than
    1.5 times the median spacing: a break cut out of the samples)."""
    indices = sorted(int(i) for i in indices)
    if periods:
        runs = [[i for i in indices if int(f0) <= i < int(f1)] for f0, f1 in periods]
        return [r for r in runs if r]
    if len(indices) < 3:
        return [indices] if indices else []
    gaps = np.diff(indices)
    limit = 1.5*float(np.median(gaps))
    runs, current = [], [indices[0]]
    for i, g in zip(indices[1:], gaps):
        if g > limit:
            runs.append(current)
            current = []
        current.append(i)
    return runs+[current]


def chunks(indices, size=CHUNK, gaps=GAPS, periods=None):
    """Discovery jobs: contiguous runs of `size` sampled frames, each followed by the next max(gaps) frames of its period
    as look-ahead (registered to, never solved). A chunk owns its first `size` frames (all of them in the last chunk of a
    period, which is never longer than `size`), so every pair (i, i+g) within a period is computed exactly once, by the
    chunk owning i, and no pair crosses a period. Pass the periods ((frame0, frame1)) whenever there is more than one."""
    look = max(gaps) if gaps else 0
    out = []
    for run in _runs(indices, periods):
        for s in range(0, len(run), size):
            out.append(run[s:s+size+look])
    return out


def chunk_pairs(indices, size=CHUNK, gaps=GAPS):
    """The (first, second) position pairs a chunk computes: first among its owned frames, second up to max(gaps) later."""
    own = min(len(indices), size)
    return [(p, p+g) for p in range(own) for g in gaps if p+g < len(indices)]


# ---------- one frame ----------
def camera_of(H, width, height):
    """The pinhole camera of a world -> pixel homography with square pixels and the principal point at the image centre:
    (focal px, (x, y) ground position in pitch metres, height m), or None when no such camera explains it. The focal
    length is the least-squares solution of r1 . r2 = 0 and |r1| = |r2|."""
    H = field.orient(np.asarray(H, np.float64), width, height)
    if H is None:
        return None
    K0 = np.array([[1.0, 0, width/2], [0, 1.0, height/2], [0, 0, 1.0]])
    A = np.linalg.inv(K0) @ H
    h1, h2 = A[:, 0], A[:, 1]
    a1, b1 = h1[0]*h2[0]+h1[1]*h2[1], h1[2]*h2[2]
    a2, b2 = h1[0]**2+h1[1]**2-h2[0]**2-h2[1]**2, h1[2]**2-h2[2]**2
    w = -(a1*b1+a2*b2)/max(a1*a1+a2*a2, 1e-30)   # 1/f^2
    if not w > 0 or not math.isfinite(w):
        return None
    f = 1/math.sqrt(w)
    M = np.diag([1/f, 1/f, 1.0]) @ A
    M = M/((np.linalg.norm(M[:, 0])+np.linalg.norm(M[:, 1]))/2)
    r1, r2 = M[:, 0]/np.linalg.norm(M[:, 0]), M[:, 1]/np.linalg.norm(M[:, 1])
    R = np.c_[r1, r2, np.cross(r1, r2)]
    C = -R.T @ M[:, 2]
    if not np.isfinite(C).all():
        return None
    return float(f), (float(C[0]), float(C[1])), float(abs(C[2]))


def solve_frame(frame, stamp, size=None):
    """The keyless midfield anchor of one frame (no people: the stage runs no detector), as a JSON-able row."""
    size = size or field.PitchSize()
    Hh, W = frame.shape[:2]
    det = PitchDetector()
    model = det.analyze(frame, stamp)
    row = {'time': round(float(stamp), 6), 'ok': False}
    circle, halfway = find_circle(det.masks, model, on_grass=True) if model.reliable else (None, None)
    if circle is None:
        row['reason'] = 'no centre circle on the grass'
        return row
    ellipse = midfield.ellipse_from_points([(x*W, y*Hh) for x, y in circle['points']])
    if ellipse is None:
        row['reason'] = 'the circle paint does not fit an ellipse'
        return row
    try:
        result = midfield.solve(circle, halfway, midfield.touchline_candidates(model, ellipse, W, Hh), W, Hh, size, det.masks,
                                allow_pinhole=True, people=[], markings=model.markings)
    except (ValueError, ArithmeticError, np.linalg.LinAlgError, cv2.error) as error:
        result = {'reason': f'{type(error).__name__}: {error}'}
    row['coverage'] = circle['coverage']
    if 'H' not in result:
        row['reason'] = result.get('reason', 'refused')
        return row
    d = result['details']
    row.update(ok=True, H=np.asarray(result['H']).tolist(), observability=round(float(result['observability']), 4),
               cap=float(result.get('cap') or 0.0), route=d.get('route'), confirmed=list(d.get('confirmed') or []),
               focal=d.get('focalPx'), ellipseRatio=d.get('ellipseRatio'), sigmaFar=d.get('sigmaFarMetres'))
    cam = camera_of(result['H'], W, Hh)
    if cam is not None:
        row['camera'] = {'focal': round(cam[0], 1), 'position': [round(cam[1][0], 2), round(cam[1][1], 2)], 'height': round(cam[2], 2)}
    return _json(row)


# ---------- registration ----------
def _sift():
    return cv2.SIFT_create(nfeatures=FEATURES)


def features(frame, sift=None):
    """SIFT keypoints (WORK_W pixels) and descriptors of a frame, or (None, None)."""
    scale = WORK_W/frame.shape[1]
    gray = cv2.cvtColor(cv2.resize(frame, (WORK_W, int(round(frame.shape[0]*scale))), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    kp, des = (sift or _sift()).detectAndCompute(gray, None)
    if des is None or len(kp) < MIN_INLIERS:
        return None, None
    return np.float32([k.pt for k in kp]), des


def match(fa, fb, width, height):
    """The homography taking frame a's pixels to frame b's (full resolution), with inliers, rms (WORK_W px) and overlap
    (share of frame b that frame a covers), or None."""
    (pa, da), (pb, db) = fa, fb
    if da is None or db is None:
        return None
    knn = cv2.BFMatcher(cv2.NORM_L2).knnMatch(da, db, k=2)
    good = [p[0] for p in knn if len(p) == 2 and p[0].distance < RATIO*p[1].distance]
    if len(good) < MIN_INLIERS:
        return None
    src, dst = pa[[g.queryIdx for g in good]], pb[[g.trainIdx for g in good]]
    moved = np.hypot(*(dst-src).T)
    if np.median(moved) > MOVING_PX:
        keep = moved > STILL_PX
        src, dst = src[keep], dst[keep]
        if len(src) < MIN_INLIERS:
            return None
    G, inl = cv2.findHomography(src, dst, cv2.RANSAC, RANSAC_PX, maxIters=4000, confidence=.999)
    if G is None or inl is None or not np.isfinite(G).all():
        return None
    inl = inl.ravel().astype(bool)
    if inl.sum() < MIN_INLIERS:
        return None
    refined, _ = cv2.findHomography(src[inl], dst[inl], 0)
    if refined is not None and np.isfinite(refined).all():
        G = refined
    proj = cv2.perspectiveTransform(src[inl].reshape(-1, 1, 2), G).reshape(-1, 2)
    rms = float(np.sqrt(np.mean(np.sum((proj-dst[inl])**2, axis=1))))
    if rms > MAX_RMS_PX:
        return None
    w, h = WORK_W, WORK_W*height/width
    corners = cv2.perspectiveTransform(np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2), G).reshape(-1, 2)
    clipped = clip_unit([(float(x/w), float(y/h)) for x, y in corners])
    S = np.diag([WORK_W/width, WORK_W/width, 1.0])
    return {'G': np.linalg.inv(S) @ G @ S, 'inliers': int(inl.sum()), 'rms': rms, 'overlap': polygon_area(clipped) if len(clipped) >= 3 else 0.0}


def _read(capture, index, position, fps):
    """Frame `index` (frame-exact: a seek, then decoding forward) and the decoder's timestamp of it, as the pipeline reads
    them. Returns (frame or None, stamp, the index the next read returns)."""
    if position is None or not 0 <= index-position <= GRAB_LIMIT:
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        position = index
    while position < index:
        if not capture.grab():
            return None, None, None
        position += 1
    ok, frame = capture.read()
    if not ok:
        return None, None, None
    stamp = float(capture.get(cv2.CAP_PROP_POS_MSEC))/1000
    if stamp <= 0 and index:
        stamp = index/fps
    return frame, stamp, index+1


def solve_chunk(source, indices, width, height, size=None, cancelled=None, chunk=CHUNK, gaps=GAPS):
    """One discovery job: the owned frames of a chunk (chunks()) solved one by one (rows), then the registrations of its
    pairs (chunk_pairs) that join two anchors (a look-ahead frame's anchor is not known here: its pair is made when the
    owned frame is one). `chunk` and `gaps` must be what chunks() was given (with other values the results stay right:
    frames and pairs computed twice are merged by discover). Runs inside a worker process: no thread setting and no
    process pool here. cancelled() is polled between frames. Returns {'rows', 'pairs', 'cancelled'?}."""
    indices = [int(i) for i in indices]
    own = min(len(indices), int(chunk))
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise OSError(f'cannot open {source}')
    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 25.0
    rows, feats, position = [], {}, None
    sift = _sift()
    try:
        for n, index in enumerate(indices):
            if cancelled is not None and cancelled():
                return {'rows': rows, 'pairs': [], 'cancelled': True}
            frame, stamp, position = _read(capture, index, position, fps)
            if frame is None:
                if n < own:
                    rows.append({'index': index, 'ok': False, 'reason': 'unreadable frame'})
                continue
            if frame.shape[1] != width or frame.shape[0] != height:
                frame = cv2.resize(frame, (int(width), int(height)), interpolation=cv2.INTER_AREA)
            if n < own:
                row = solve_frame(frame, stamp, size)
                row['index'] = index
                rows.append(row)
                if not row['ok']:
                    continue
            feats[index] = features(frame, sift)
    finally:
        capture.release()
    anchor = {r['index'] for r in rows if r.get('ok')}
    pairs = []
    for p, q in chunk_pairs(indices, chunk, gaps):
        a, b = indices[p], indices[q]
        if a not in anchor or (q < own and b not in anchor) or a not in feats or b not in feats:
            continue
        if cancelled is not None and cancelled():
            return {'rows': rows, 'pairs': pairs, 'cancelled': True}
        m = match(feats[a], feats[b], width, height)
        if m is None:
            pairs.append({'a': a, 'b': b, 'ok': False})
        else:
            pairs.append({'a': a, 'b': b, 'ok': True, 'G': np.asarray(m['G']).tolist(), 'inliers': m['inliers'],
                          'rms': round(m['rms'], 3), 'overlap': round(m['overlap'], 3)})
    return {'rows': rows, 'pairs': pairs}


# ---------- the match ----------
def consensus(anchors):
    """The match camera: (ground position, height, share of the pinhole anchors within POS_TOL_M / HEIGHT_TOL_M, their
    count). The median of the pinhole anchors' cameras, re-taken three times over those within TRIM_POS_M / TRIM_HEIGHT_M
    of it, so that a minority of wrong cameras does not move it. None without MIN_PINHOLE pinhole anchors."""
    pin = [a for a in anchors if a.get('route') == 'pinhole' and a.get('camera')]
    if len(pin) < MIN_PINHOLE:
        return None
    P = np.array([a['camera']['position'] for a in pin], np.float64)
    h = np.array([a['camera']['height'] for a in pin], np.float64)
    centre, height = np.median(P, axis=0), float(np.median(h))
    for _ in range(3):
        near = (np.hypot(*(P-centre).T) < TRIM_POS_M) & (np.abs(h-height) < TRIM_HEIGHT_M)
        if near.sum() >= MIN_PINHOLE:
            centre, height = np.median(P[near], axis=0), float(np.median(h[near]))
    inside = (np.hypot(*(P-centre).T) <= POS_TOL_M) & (np.abs(h-height) <= HEIGHT_TOL_M)
    return (float(centre[0]), float(centre[1])), height, float(inside.mean()), len(pin)


def _camera_ok(a, centre, elevation):
    c = a.get('camera')
    return c is not None and math.hypot(c['position'][0]-centre[0], c['position'][1]-centre[1]) <= POS_TOL_M and abs(c['height']-elevation) <= HEIGHT_TOL_M


def _graph(pairs):
    """{(a, b): G} of the usable registrations (frame a's pixels -> frame b's)."""
    return {(int(p['a']), int(p['b'])): np.asarray(p['G'], np.float64) for p in pairs if p.get('ok') and p.get('overlap', 0) >= PAIR_OVERLAP}


def _carried(i, j, Hj, G, width, height):
    """Anchor j's homography carried into frame i, or None without a registration between them."""
    if (j, i) in G:
        return field.orient(G[(j, i)] @ Hj, width, height)
    if (i, j) in G:
        return field.orient(np.linalg.inv(G[(i, j)]) @ Hj, width, height)
    return None


def agreements(anchors, G, width, height, size=None):
    """{index: [(neighbour, metres)]}: every pair of anchors joined by a registration, compared in the later frame."""
    by = {a['index']: a for a in anchors}
    out = {}
    for (a, b) in G:
        if a in by and b in by:
            carried = _carried(b, a, np.asarray(by[a]['H'], np.float64), G, width, height)
            if carried is None:
                continue
            d = registration._disagreement(np.asarray(by[b]['H'], np.float64), carried, width, height, size)
            out.setdefault(a, []).append((b, d))
            out.setdefault(b, []).append((a, d))
    return out


def _neighbours(G):
    out = {}
    for a, b in G:
        out.setdefault(a, set()).add(b)
        out.setdefault(b, set()).add(a)
    return out


def fused(anchor, kept, G, width, height, neighbours=None):
    """The anchor fused with its kept neighbours carried into its frame (neighbours: {index: indices}, the agreeing ones
    in discover, so that a wrong registration never takes part): the per-point median of their ground positions over an
    image grid, refitted as one homography; (H, homographies used), H None with fewer than FUSE_MIN."""
    i = anchor['index']
    Hs = [np.asarray(anchor['H'], np.float64)]
    for j in sorted((neighbours if neighbours is not None else _neighbours(G)).get(i, ())):
        if j in kept:
            c = _carried(i, j, np.asarray(kept[j]['H'], np.float64), G, width, height)
            if c is not None:
                Hs.append(c)
    if len(Hs) < FUSE_MIN:
        return None, len(Hs)
    xs, ys = np.linspace(.04, .96, 12)*width, np.linspace(.3, .97, 9)*height
    grid = np.array([(x, y) for y in ys for x in xs])
    front = np.all([field.image_depth(H, grid) > 1e-9 for H in Hs], axis=0)
    if front.sum() < 8:
        return None, len(Hs)
    pts = grid[front]
    ground = np.median(np.stack([field.unproject(H, pts) for H in Hs]), axis=0)
    H, _ = cv2.findHomography(ground, pts, 0)
    if H is None or not np.isfinite(H).all():
        return None, len(Hs)
    return field.orient(H, width, height), len(Hs)


def points_of(H, width, height, size=None):
    """The landmarks a homography puts inside the image (MARGIN_PX in), normalized: a calibration keyframe's points."""
    marks = field.landmarks(size or field.PitchSize())
    names = list(marks)
    P = np.array([marks[n] for n in names], np.float64)
    front = field.depth(H, P) > 1e-9
    q = field.project(H, P)
    return [{'name': n, 'x': round(float(x/width), 6), 'y': round(float(y/height), 6)} for n, (x, y), f in zip(names, q, front)
            if f and MARGIN_PX <= x <= width-MARGIN_PX and MARGIN_PX <= y <= height-MARGIN_PX]


def _valid(points, width, height, size):
    """Would the server and the pipeline take these points (4-30 landmarks the solver accepts, standard and match size)?"""
    if not MIN_POINTS <= len(points) <= 30:
        return False
    try:
        keyframe_solver.solve(points, width, height, max_error=4.0)
        keyframe_solver.solve(points, width, height, max_error=4.0, size=size)
    except ValueError:
        return False
    return True


def discover(rows, pairs, segments, fps, width, height, size=None, manual=()):
    """The match calibration index from the discovery jobs' rows and pairs (all chunks together): the camera consensus,
    the kept anchors, up to PER_SEGMENT synthetic keyframes per segment (fullmatch.plan.Segment: inside its processed
    frames [proc0, proc1)), the user's keyframes (manual: {time, points}) winning within MANUAL_REACH_S. JSON-able."""
    size = size or field.PitchSize()
    fps = float(fps)
    by = {}
    for r in rows:
        by.setdefault(int(r['index']), r)
    rows = [by[i] for i in sorted(by)]
    anchors = [dict(r, index=int(r['index'])) for r in rows if r.get('ok') and r.get('H') is not None]
    for a in anchors:
        if not a.get('camera'):
            cam = camera_of(a['H'], width, height)
            if cam is not None:
                a['camera'] = {'focal': cam[0], 'position': list(cam[1]), 'height': cam[2]}
    manual_times = sorted(float(m['time']) for m in manual or ())
    out = {'version': VERSION, 'step': STEP_S, 'sampled': len(rows), 'anchors': len(anchors), 'kept': 0, 'camera': None, 'refused': None,
           'rules': {'posTolM': POS_TOL_M, 'heightTolM': HEIGHT_TOL_M, 'agreeM': AGREE_M, 'minObs': MIN_OBS, 'consensusShare': CONSENSUS_SHARE,
                     'manualReachS': MANUAL_REACH_S, 'perSegment': PER_SEGMENT, 'minGapS': MIN_GAP_S, 'reachS': REACH_S},
           'keyframes': [], 'segments': [], 'uncovered': [], 'keptAnchors': []}
    found = consensus(anchors)
    kept, G, agree = {}, {}, {}
    if found is None:
        out['refused'] = (f'Too few automatic anchors to find the camera ({len(anchors)} in {len(rows)} sampled frames): '
                          'add calibration keyframes.')
    else:
        centre, elevation, share, n_pin = found
        out['camera'] = {'position': [round(centre[0], 2), round(centre[1], 2)], 'height': round(elevation, 2), 'share': round(share, 3), 'pinhole': n_pin}
        G = _graph(pairs)
        agree = agreements(anchors, G, width, height, size)
        for a in anchors:
            a['agreements'] = sum(1 for _, d in agree.get(a['index'], ()) if d <= AGREE_M)
            if _camera_ok(a, centre, elevation) and a['agreements'] >= 1 and a['observability'] >= MIN_OBS:
                kept[a['index']] = a
        best = max((a['observability'] for a in anchors if _camera_ok(a, centre, elevation)), default=0.0)
        if share < CONSENSUS_SHARE:
            out['refused'] = (f'The automatic anchors do not agree on one camera (only {share:.0%} of them within {POS_TOL_M:.0f} m and '
                              f'{HEIGHT_TOL_M} m of the most common position): not one fixed camera, or the automatic calibration '
                              'fails on this footage. Add calibration keyframes.')
        elif not kept and best < MIN_OBS:
            out['refused'] = (f'No automatic anchor is constrained well enough (best observability {best:.2f}, {MIN_OBS:.2f} needed): '
                              'add calibration keyframes.')
        elif not kept:
            out['refused'] = ('No automatic anchor is confirmed by a neighbouring frame (the sampled frames could not be registered '
                              'to each other): add calibration keyframes.')
    out['kept'] = len(kept) if out['refused'] is None else 0
    if out['refused'] is None:   # the record: [frame, observability, agreeing neighbours] of every kept anchor
        out['keptAnchors'] = [[i, round(float(kept[i]['observability']), 3), int(kept[i]['agreements'])] for i in sorted(kept)]
    neighbours = {i: {j for j, d in agree.get(i, ()) if d <= AGREE_M} for i in kept}   # fused: the agreeing kept neighbours
    chosen = {}
    for s in segments:
        lo, hi = s.proc0/fps, s.proc1/fps
        entry = {'index': s.index, 'sampled': sum(1 for r in rows if s.proc0 <= r['index'] < s.proc1),
                 'anchors': sum(1 for a in anchors if s.proc0 <= a['index'] < s.proc1),
                 'kept': sum(1 for i in kept if s.proc0 <= i < s.proc1), 'keyframes': [], 'frames': [],
                 'manual': [t for t in manual_times if lo-MANUAL_REACH_S <= t <= hi+MANUAL_REACH_S]}
        if out['refused'] is None:
            for k in _choose(s, kept, G, neighbours, width, height, size, fps, manual_times):
                chosen.setdefault(k['frame'], k)
                entry['keyframes'].append(k['time'])
                entry['frames'].append(k['frame'])
        if not entry['keyframes'] and not entry['manual']:
            out['uncovered'].append(s.index)
        out['segments'].append(entry)
    out['keyframes'] = [chosen[f] for f in sorted(chosen)]
    return _json(out)


def _choose(segment, kept, G, neighbours, width, height, size, fps, manual_times):
    """Up to PER_SEGMENT keyframes inside the segment's processed frames, placed for reach: each the kept anchor that
    brings the most still unreached seconds of the segment within REACH_S of a keyframe (user keyframes reach too), the
    better score on a tie, MIN_GAP_S from the others, none within MANUAL_REACH_S of a user keyframe. Fused ones first;
    then kept anchors that cannot be fused (too few agreeing neighbours: CAN-ISL 4079-4256 s, set pieces at one goal, 2
    kept anchors 12 s apart) as they are, for what the fused ones leave unreached (fusedAnchors 1; kept anchors measured
    .92 m at worst)."""
    cands = [a for i, a in kept.items() if segment.proc0 <= i < segment.proc1
             and all(abs(a['time']-t) > MANUAL_REACH_S for t in manual_times)]
    score = lambda a: a['observability']*min(1.0, .5+.25*a['agreements'])
    grid = np.arange(segment.proc0, segment.proc1, max(1.0, float(fps)))/float(fps)   # the segment's seconds
    reached = np.zeros(len(grid), bool)
    for t in manual_times:   # the user keyframes this run applies (within MANUAL_REACH_S = pipeline.ROLL_SECONDS)
        if segment.proc0/fps-MANUAL_REACH_S <= t <= segment.proc1/fps+MANUAL_REACH_S:
            reached |= np.abs(grid-t) <= REACH_S
    out = []

    def make(a, unfused):
        H, n = (field.orient(np.asarray(a['H'], np.float64), width, height), 1) if unfused else fused(a, kept, G, width, height, neighbours)
        if H is None:
            return None
        points = points_of(H, width, height, size)
        if not _valid(points, width, height, size):
            return None
        return {'time': round(float(a['time']), 6), 'points': points, 'source': 'discovery', 'frame': a['index'],
                'observability': round(float(a['observability']), 4), 'agreements': int(a['agreements']), 'fusedAnchors': n,
                'H': np.asarray(H).tolist()}
    for unfused in (False, True):
        failed = set()
        while len(out) < PER_SEGMENT:
            best, key = None, None
            for a in cands:
                if a['index'] in failed or any(abs(a['index']-o['frame'])/fps < MIN_GAP_S for o in out):
                    continue
                gain = int(np.count_nonzero(~reached & (np.abs(grid-a['time']) <= REACH_S)))
                if gain and (key is None or (gain, score(a), -a['index']) > key):
                    best, key = a, (gain, score(a), -a['index'])
            if best is None:
                break
            k = make(best, unfused)
            if k is None:
                failed.add(best['index'])
                continue
            out.append(k)
            reached |= np.abs(grid-best['time']) <= REACH_S
    return sorted(out, key=lambda k: k['frame'])


def keyframes_for(index, segment):
    """A segment's synthetic keyframes ({time, points: [{name, x, y}]} normalized, the server's calibration format, with
    their record keys) from the match calibration index; segment is a plan.Segment or its index."""
    k = segment if isinstance(segment, int) else segment.index
    entry = next((s for s in index.get('segments') or () if s['index'] == k), None)
    if entry is None or index.get('refused'):
        return []
    frames = set(entry.get('frames') or ())
    return [dict(kf) for kf in index.get('keyframes') or () if kf['frame'] in frames]
