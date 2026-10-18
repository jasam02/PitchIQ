"""Field registration over the whole clip (passes 2 and 3), separate from image-space tracking.

Image-space tracking (detection, local tracks, global identities, the ball) never depends on where the
pitch is believed to be in a frame: a wrong or missing homography cannot end a track. Pitch positions
are produced afterwards, offline, from the saved image observations:

    pass 1  tracking: per frame the camera motion (M, how it was measured) and the calibration anchors
            (user landmarks, painted markings that confirmed the model, automatic midfield anchors),
            each with its geometric observability (field.observability)
    pass 2  build_timeline: anchors are carried forward AND backward through the camera-motion chain;
            between anchors the two directions are blended by confidence, and their disagreement lowers
            it. A weak anchor starts low and fades sooner; an anchor below the observability floor is
            never created, so it cannot be carried smoothly into a wrong result
    pass 3  remap_frames: every person's foot point and the ball are mapped through the frame's final
            homography; each position carries a confidence (the calibration's times how much a few
            pixels of error are worth in metres there) and a sigma in metres along and across the pitch
            (a few pixels of foot error through the local Jacobian of the mapping, more when the box is
            cut by the frame's bottom edge, plus the forward/backward disagreement), and positions that
            do not land on the pitch, or whose frame is below the confidence floor, are left unresolved
            rather than invented. bridge_ball fills short gaps in the ball's image track from both sides
            when the motion is plausible.

Confidence is two things kept apart: observability (is the geometry constrained by enough independent
evidence?) and temporal distance (how far an anchor was carried, and how well the motion was measured).
A smoothly carried wrong anchor is never confident: its observability was low to begin with.
"""
import math
from dataclasses import dataclass, field as dc_field

import numpy as np

from . import field
from .geometry import clamp, foot_point
from .pitch_model import DECAY, DECAY_COARSE, DECAY_LOOSE, combine

STILL_LIMIT = 3          # frames the chain survives without any motion estimate (the camera is assumed still)
MIN_CONFIDENCE = .05
DISAGREE_METRES = 8.0    # forward/backward estimates this far apart divide the confidence by e
MAX_FOOT_ERROR = 5.0     # metres of foot-point error at which a position's confidence reaches 0
FOOT_PIXELS = 3.0        # pixels of foot-point uncertainty assumed for that
CLIPPED_FOOT = .35       # ... as a share of the box height when the box is cut by the bottom of the frame (the feet are below it)
MIN_POSITION = .2        # registration confidence below which no pitch position is written (the frame counts as uncalibrated)
SAMPLE_POINTS = ((.5, .9), (.2, .75), (.8, .75), (.5, .6))   # fallback points where two homographies are compared (normalized)
GRID_X, GRID_Y = (.08, .92, 6), (.12, .97, 5)                 # the grid over the frame where they are compared on the visible pitch
PITCH_MARGIN = 5.0       # metres beyond the lines that still count as the visible pitch for that comparison
BALL_GAP = .5            # seconds: the longest ball gap bridged from both sides
BALL_SPEED = .6          # frame widths per second: the fastest plausible ball motion across a gap


@dataclass
class Anchor:
    index: int                 # frame index in the run
    time: float
    H: np.ndarray              # world metres -> image pixels (field convention), oriented
    observability: float       # geometric certainty, 0..1
    source: str                # 'landmarks' (user), 'paint' (markings confirmed the model), 'auto-midfield', 'reregistered' (direct match to a keyframe view)
    details: dict = dc_field(default_factory=dict)
    root: int = -1             # frame index of the anchor that started this chain (its own index for a fresh anchor):
                               # a derived (paint) anchor is left out with its root in a hold-out test


def _normalize(H):
    norm = float(np.linalg.norm(H))
    return H/norm if norm > 0 else H


def ground_points(Ha, Hb, width, height, size=None):
    """Image points on a grid over the frame that see the ground in front of the camera under both
    homographies and land on (or within PITCH_MARGIN of) the pitch under the first: where two estimates
    of the same frame are compared. Far points count as much as near ones, where a pixel is worth far more
    metres. Returns the points (pixels) or None when fewer than four qualify."""
    size = size or field.PitchSize()
    xs, ys = np.linspace(*GRID_X)*width, np.linspace(*GRID_Y)*height
    pts = np.array([(x, y) for y in ys for x in xs], np.float64)
    try:
        front = (field.image_depth(Ha, pts) > 1e-9) & (field.image_depth(Hb, pts) > 1e-9)
        pts = pts[front]
        if len(pts) < 4:
            return None
        world = field.unproject(Ha, pts)
    except np.linalg.LinAlgError:
        return None
    on_pitch = ((world[:, 0] >= -PITCH_MARGIN) & (world[:, 0] <= size.length+PITCH_MARGIN) &
                (world[:, 1] >= -PITCH_MARGIN) & (world[:, 1] <= size.width+PITCH_MARGIN))
    if on_pitch.sum() >= 4:
        pts = pts[on_pitch]
    return pts


def _disagreement(Ha, Hb, width, height, size=None):
    """Median distance (metres) between two homographies' mappings of the visible pitch (the grid of
    ground_points; the fixed near points when the grid has too few). Points behind either horizon are
    left out, never turned into infinite distances."""
    pts = ground_points(Ha, Hb, width, height, size)
    if pts is None:
        pts = np.array([(nx*width, ny*height) for nx, ny in SAMPLE_POINTS], np.float64)
        try:
            front = (field.image_depth(Ha, pts) > 1e-9) & (field.image_depth(Hb, pts) > 1e-9)
        except np.linalg.LinAlgError:
            return math.inf
        pts = pts[front]
        if len(pts) < 2:
            return math.inf
    try:
        a, b = field.unproject(Ha, pts), field.unproject(Hb, pts)
    except np.linalg.LinAlgError:
        return math.inf
    d = np.hypot(*(a-b).T)
    d = d[np.isfinite(d)]
    return float(np.median(d)) if len(d) else math.inf


def _carry(state, motion, forward, width, height):
    """One step of the chain. state: (H, temporal, still_frames) or None."""
    if state is None or motion['cut']:
        return None
    H, temporal, still = state
    quality = motion.get('quality') or ('flow' if motion.get('reliable') else 'none')
    M = motion.get('M')
    dt = max(0.0, motion.get('dt', 0.0))
    if M is None or quality == 'none':
        if still+1 > STILL_LIMIT:
            return None
        return (H, temporal-DECAY_LOOSE*dt, still+1)
    M = np.asarray(M, np.float64)
    try:
        moved = M @ H if forward else np.linalg.inv(M) @ H
    except np.linalg.LinAlgError:
        return None
    moved = field.orient(moved, width, height)
    if moved is None:
        return None
    return (moved, temporal-(DECAY_COARSE if quality == 'coarse' else DECAY)*dt, 0)


def build_timeline(log, anchors, width, height, size=None):
    """log: per frame {'t', 'dt', 'M' (3x3 pixel transform from the previous frame to this one, or None),
    'reliable', 'cut', 'quality': 'flow' | 'coarse' | 'none'}. Returns per frame: H (world -> pixels, or
    None), confidence, temporal, observability, source ('landmarks', 'paint', 'auto-midfield', 'carried',
    'interpolated', 'none'), direction, anchor (frame index), age (seconds), disagreement (metres)."""
    size = size or field.PitchSize()
    n = len(log)
    best = {}
    for a in anchors:
        if 0 <= a.index < n and (a.index not in best or a.observability > best[a.index].observability):
            best[a.index] = a

    def sweep(forward):
        out = [None]*n
        state, anchor = None, None
        order = range(n) if forward else range(n-1, -1, -1)
        for i in order:
            if state is not None:
                step = log[i] if forward else log[i+1]
                state = _carry(state, step, forward, width, height)
            if i in best and (state is None or combine(1.0, best[i].observability) >= combine(state[1], anchor.observability)):
                anchor, state = best[i], (best[i].H, 1.0, 0)
            if state is not None and combine(state[1], anchor.observability) >= MIN_CONFIDENCE:
                out[i] = (state, anchor)
            else:
                state = None if state is None or combine(state[1], anchor.observability) < MIN_CONFIDENCE else state
        return out
    forward, backward = sweep(True), sweep(False)
    out = []
    for i in range(n):
        t = log[i]['t']
        f, b = forward[i], backward[i]
        entry = {'H': None, 'confidence': 0.0, 'temporal': 0.0, 'observability': 0.0, 'source': 'none', 'direction': None,
                 'anchor': None, 'age': None, 'disagreement': None}
        if f is not None and b is not None and f[1] is b[1]:
            b = None   # the same anchor on both sides (its own frame): one estimate
        if f is None and b is None:
            out.append(entry)
            continue
        if f is not None and b is not None:
            (Hf, tf, _), af = f
            (Hb, tb, _), ab = b
            cf, cb = combine(tf, af.observability), combine(tb, ab.observability)
            d = _disagreement(Hf, Hb, width, height, size)
            w = cf/(cf+cb)
            Nf, Nb = _normalize(Hf), _normalize(Hb)
            if float(np.sum(Nf*Nb)) < 0:
                Nb = -Nb
            H = field.orient(w*Nf+(1-w)*Nb, width, height)
            if H is None:
                H = Hf if cf >= cb else Hb
            agreement = math.exp(-d/DISAGREE_METRES) if math.isfinite(d) else 0.0
            nearer, nearer_t = (af, tf) if w >= .5 else (ab, tb)
            entry.update(H=H, confidence=max(cf, cb)*agreement, temporal=max(tf, tb)*agreement, observability=nearer.observability,
                         source='interpolated', direction='both', anchor=nearer.index, age=round(abs(t-nearer.time), 3),
                         disagreement=round(d, 2) if math.isfinite(d) else None)
        else:
            (H, temporal, _), a = f if f is not None else b
            entry.update(H=H, confidence=combine(temporal, a.observability), temporal=temporal, observability=a.observability,
                         source=a.source if a.index == i else 'carried', direction=None if a.index == i else ('forward' if f is not None else 'backward'),
                         anchor=a.index, age=round(abs(t-a.time), 3))
        entry['confidence'] = round(float(clamp(entry['confidence'])), 3)
        entry['temporal'] = round(float(clamp(entry['temporal'])), 3)
        out.append(entry)
    return out


def homography_json(H):
    """The frame's final homography (world metres -> pixels) as nine floats, scaled so that |H[2, 2]| = 1 when
    that is numerically safe, or None. Stored per frame so that the camera footprint and the hold-out
    tests can be computed from a saved result. The scale is positive: H is oriented (field.orient: the
    ground in front of the camera at positive depth), and H[2, 2] is the depth of the far-left corner,
    negative when that corner lies behind the camera; dividing by it flipped the orientation of those
    frames (Veo 1-31 s: 272 of 900; broadcast 0-60 s: none)."""
    if H is None:
        return None
    H = np.asarray(H, np.float64)
    if abs(H[2, 2]) > 1e-9:
        H = H/abs(H[2, 2])
    return [float(v) for v in H.ravel()]


def position(H, size, point_px, confidence, sigma_px=None, disagreement=None):
    """(normalized pitch position, position confidence, sigma) for an image point, or (None, 0, None)
    when it does not land on the pitch. The confidence is the calibration's times the local projection
    certainty: a few pixels of foot-point error are worth many metres far from the camera. The sigma is
    that error in metres, along and across the pitch: the pixel uncertainty (sigma_px: one value or
    (x, y); FOOT_PIXELS by default) through the local Jacobian of the mapping, plus the registration's
    forward/backward disagreement (metres) in quadrature."""
    p = field.to_pitch(H, size, point_px)
    if p is None:
        return None, 0.0, None
    scale = field.px_per_metre(H, (p[0]*size.length, p[1]*size.width))
    local = clamp(1-FOOT_PIXELS/max(scale, 1e-6)/MAX_FOOT_ERROR)
    sx, sy = (sigma_px, sigma_px) if sigma_px is None or np.isscalar(sigma_px) else sigma_px
    sx = FOOT_PIXELS if sx is None else float(sx)
    sy = FOOT_PIXELS if sy is None else float(sy)
    x, y = point_px
    probes = np.array([(x-1, y), (x+1, y), (x, y-1), (x, y+1)], np.float64)
    sigma = None
    try:
        if (field.image_depth(H, probes) > 1e-9).all():
            w = field.unproject(H, probes)
            jac = np.array([[(w[1, 0]-w[0, 0])/2, (w[3, 0]-w[2, 0])/2], [(w[1, 1]-w[0, 1])/2, (w[3, 1]-w[2, 1])/2]])
            cov = jac @ np.diag([sx*sx, sy*sy]) @ jac.T
            if disagreement is not None and math.isfinite(disagreement):
                cov += np.eye(2)*disagreement*disagreement
            sigma = (min(999.0, math.sqrt(max(0.0, cov[0, 0]))), min(999.0, math.sqrt(max(0.0, cov[1, 1]))))
    except np.linalg.LinAlgError:
        sigma = None
    return p, round(float(confidence*local), 3), sigma


def clipped_below(box, tolerance=.004):
    """Is this [x, y, w, h] box cut by the bottom edge of the frame (the feet may be below it)?"""
    return box[1]+box[3] >= 1-tolerance


def remap_frames(frames, timeline, width, height, size=None):
    """Pass 3: pitch positions for every person and the ball from the final homographies. Writes
    `calibration` and `calibrated` on each frame, `pitch`, `pitchConfidence` and `pitchSource` on each
    person (or `pitchStatus: 'unresolved'`), and the same on the ball. Returns mapping diagnostics:
    frames with many live image tracks of which fewer than half got a position, and the median share."""
    size = size or field.PitchSize()
    loss_frames, ratios = 0, []
    for i, frame in enumerate(frames):
        entry = timeline[i] if i < len(timeline) else None
        H = entry['H'] if entry else None
        info = {'confidence': entry['confidence'] if entry else 0.0, 'temporal': entry['temporal'] if entry else 0.0,
                'observability': entry['observability'] if entry else 0.0, 'source': entry['source'] if entry else 'none',
                'direction': entry['direction'] if entry else None, 'anchor': entry['anchor'] if entry else None,
                'anchorAge': entry['age'] if entry else None, 'disagreement': entry['disagreement'] if entry else None,
                'H': homography_json(H)}
        frame['calibration'] = info
        usable = H is not None and info['confidence'] >= MIN_POSITION
        frame['calibrated'] = usable
        frame['pitchCoordinates'] = None if not usable else ('calibration' if info['source'] in ('landmarks', 'paint', 'auto-midfield', 'reregistered') else 'predicted')
        tracks = mapped = 0
        for person in frame['people']:
            for key in ('pitch', 'pitchConfidence', 'pitchSource', 'pitchStatus', 'pitchSigma', 'pitchClipped'):
                person.pop(key, None)
            live = person.get('state') in ('confirmed', 'uncertain')
            tracks += live
            if not usable:
                person['pitchStatus'] = 'unresolved'
                continue
            fx, fy = foot_point(person['box'])
            foot = (fx*width, fy*height)
            if field.image_depth(H, [foot])[0] <= 1e-9:
                person['pitchStatus'] = 'horizon'
                continue
            clipped = clipped_below(person['box'])
            sigma_px = (FOOT_PIXELS, max(FOOT_PIXELS, CLIPPED_FOOT*person['box'][3]*height)) if clipped else None
            p, c, sigma = position(H, size, foot, info['confidence'], sigma_px, info['disagreement'])
            if p is None or c <= 0:
                person['pitchStatus'] = 'unresolved'
                continue
            person['pitch'] = [round(float(p[0]), 4), round(float(p[1]), 4)]
            person['pitchConfidence'], person['pitchSource'] = c, info['source']
            if sigma is not None:
                person['pitchSigma'] = [round(sigma[0], 1), round(sigma[1], 1)]
            if clipped:
                person['pitchClipped'] = True
            mapped += live
        ball = frame.get('ball') or {}
        for key in ('pitch', 'pitchConfidence', 'pitchSigma'):
            ball.pop(key, None)
        if usable and ball.get('center') and ball.get('box'):
            cx, cy = ball['center']
            p, c, sigma = position(H, size, (cx*width, min(1.0, cy+ball['box'][3]/2)*height), info['confidence'], None, info['disagreement'])
            if p is not None and c > 0:
                ball['pitch'], ball['pitchConfidence'] = [round(float(p[0]), 4), round(float(p[1]), 4)], c
                if sigma is not None:
                    ball['pitchSigma'] = [round(sigma[0], 1), round(sigma[1], 1)]
        frame['mapping'] = {'imageTracks': tracks, 'mapped': mapped}
        if tracks >= 6:
            ratios.append(mapped/tracks)
            if mapped < .5*tracks:
                loss_frames += 1
    return {'mappingLossFrames': loss_frames, 'medianMappedShare': round(float(np.median(ratios)), 3) if ratios else None,
            'framesWithTracks': len(ratios)}


def bridge_ball(frames, width):
    """Offline ball cleanup: label every ball by its source (DETECTED, PREDICTED) and fill gaps of at most
    BALL_GAP seconds between two known positions of the same ball track by linear interpolation when the
    motion across the gap is plausible (INTERPOLATED). Longer or implausible gaps stay unknown. Returns
    the number of frames filled."""
    known = []
    for i, f in enumerate(frames):
        b = f.get('ball') or {}
        if b.get('state') == 'TRACKED':
            b['source'] = 'DETECTED'
        elif b.get('state') == 'MISSING':
            b['source'] = 'PREDICTED'
        if b.get('state') in ('TRACKED', 'MISSING') and b.get('center') and b.get('box') and b.get('track') is not None:
            known.append(i)
    filled = 0
    for a, b in zip(known, known[1:]):
        if b-a <= 1:
            continue
        fa, fb = frames[a], frames[b]
        ba, bb = fa['ball'], fb['ball']
        dt = fb['time']-fa['time']
        if ba.get('track') != bb.get('track') or dt <= 0 or dt > BALL_GAP:
            continue
        (x0, y0), (x1, y1) = ba['center'], bb['center']
        if math.hypot((x1-x0), (y1-y0))/dt > BALL_SPEED:
            continue
        for k in range(a+1, b):
            f = frames[k]
            u = (f['time']-fa['time'])/dt
            d = ba['box'][2]+(bb['box'][2]-ba['box'][2])*u
            dh = ba['box'][3]+(bb['box'][3]-ba['box'][3])*u
            cx, cy = x0+(x1-x0)*u, y0+(y1-y0)*u
            f['ball'] = {'state': 'MISSING', 'phase': 'OCCLUDED', 'observed': False, 'source': 'INTERPOLATED',
                         'confidence': round(.8*min(ba.get('confidence', 0), bb.get('confidence', 0)), 3), 'track': ba['track'],
                         'box': [round(cx-d/2, 5), round(cy-dh/2, 5), round(d, 5), round(dh, 5)], 'center': [round(cx, 5), round(cy, 5)],
                         'missingFor': round(f['time']-fa['time'], 2), 'bridged': True}
            filled += 1
    return filled


def warnings(frames, min_tracks=6):
    """Events for spans where many live image tracks had no pitch position at once: a calibration
    failure to look at, never '14 players left the match'."""
    events, start, worst = [], None, None
    for f in frames+[None]:
        m = f.get('mapping') if f else None
        lost = bool(m and m['imageTracks'] >= min_tracks and m['mapped'] < .5*m['imageTracks'])
        if lost and start is None:
            start, worst = f, m
        elif lost:
            if m['mapped']/m['imageTracks'] < worst['mapped']/worst['imageTracks']:
                worst = m
        elif start is not None:
            end = f['time'] if f else frames[-1]['time']
            if end-start['time'] >= .5:
                c = start.get('calibration', {})
                events.append({'time': round(start['time'], 3), 'kind': 'calibration', 'track': None, 'playerId': None,
                               'message': f"MAPPING LOSS\n{start['time']:.2f}-{end:.2f} s: at worst {worst['mapped']} of {worst['imageTracks']} tracked people "
                                          f"had a pitch position (calibration confidence {c.get('confidence', 0):.2f}, source {c.get('source', 'none')}). "
                                          'The image tracks are intact; the pitch registration is what failed here. Mark landmarks on a clear frame in this span.'})
            start = worst = None
    return events


def summarize(timeline, anchors, offset=0, size=None):
    """timeline: the segment's entries; offset: log frames before the segment (anchors are listed by their
    frame index within the segment, negative for a pre-roll anchor). size: the run's pitch size, exported
    with each dimension's source (assumed or learned) and the widths automatic anchors measured, for a
    field-size selector."""
    n = len(timeline)
    known = sum(1 for e in timeline if e['H'] is not None)
    confident = sum(1 for e in timeline if e['H'] is not None and e['confidence'] >= .5)
    sources = {}
    for e in timeline:
        sources[e['source']] = sources.get(e['source'], 0)+1
    keys = ('landmarks', 'lines', 'circlePoints', 'count', 'spread', 'axes', 'conditioning', 'residual', 'rmsMetres', 'rmsPixels', 'circleCoverage', 'coverage', 'score',
            'route', 'focalPx', 'cameraHeightMetres', 'sigmaMetres', 'ellipseRatio', 'cap')
    extra = {}
    if size is not None:
        widths = [a.details['size']['width']['measured'] for a in anchors if a.source == 'auto-midfield'
                  and isinstance(a.details.get('size'), dict) and 'measured' in (a.details['size'].get('width') or {})]
        extra['fieldSize'] = size.dimensions() | ({'widthMeasured': {'median': round(float(np.median(widths)), 1), 'anchors': len(widths)}} if widths else {})
    return extra | {'frames': n, 'calibratedFrames': known, 'calibratedShare': round(known/n, 3) if n else 0.0,
            'confidentShare': round(confident/n, 3) if n else 0.0,
            'meanConfidence': round(float(np.mean([e['confidence'] for e in timeline])), 3) if n else 0.0,
            'meanObservability': round(float(np.mean([e['observability'] for e in timeline if e['H'] is not None])), 3) if known else 0.0,
            'anchors': [{'frame': a.index-offset, 'time': round(a.time, 3), 'observability': round(a.observability, 3), 'source': a.source, 'root': a.root-offset,
                         **{k: v for k, v in a.details.items() if k in keys}} for a in sorted(anchors, key=lambda a: a.index)],
            'sources': sources}
