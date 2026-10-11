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
            pixels of error are worth in metres there), and positions that do not land on the pitch are
            left unresolved rather than invented. bridge_ball fills short gaps in the ball's image track
            from both sides when the motion is plausible.

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
SAMPLE_POINTS = ((.5, .9), (.2, .75), (.8, .75), (.5, .6))   # where two homographies are compared (normalized)
BALL_GAP = .5            # seconds: the longest ball gap bridged from both sides
BALL_SPEED = .6          # frame widths per second: the fastest plausible ball motion across a gap


@dataclass
class Anchor:
    index: int                 # frame index in the run
    time: float
    H: np.ndarray              # world metres -> image pixels (field convention), oriented
    observability: float       # geometric certainty, 0..1
    source: str                # 'landmarks' (user), 'paint' (markings confirmed the model), 'auto-midfield'
    details: dict = dc_field(default_factory=dict)


def _normalize(H):
    norm = float(np.linalg.norm(H))
    return H/norm if norm > 0 else H


def _disagreement(Ha, Hb, width, height):
    """Mean distance (metres) between two homographies' mappings of a few image points."""
    try:
        a = field.unproject(Ha, [(nx*width, ny*height) for nx, ny in SAMPLE_POINTS])
        b = field.unproject(Hb, [(nx*width, ny*height) for nx, ny in SAMPLE_POINTS])
    except np.linalg.LinAlgError:
        return math.inf
    d = np.hypot(*(a-b).T)
    return float(np.mean(d)) if np.isfinite(d).all() else math.inf


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
            d = _disagreement(Hf, Hb, width, height)
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


def position(H, size, point_px, confidence):
    """(normalized pitch position, position confidence) for an image point, or (None, 0) when it does not
    land on the pitch. The confidence is the calibration's times the local projection certainty: a few
    pixels of foot-point error are worth many metres far from the camera."""
    p = field.to_pitch(H, size, point_px)
    if p is None:
        return None, 0.0
    scale = field.px_per_metre(H, (p[0]*size.length, p[1]*size.width))
    local = clamp(1-FOOT_PIXELS/max(scale, 1e-6)/MAX_FOOT_ERROR)
    return p, round(float(confidence*local), 3)


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
                'anchorAge': entry['age'] if entry else None, 'disagreement': entry['disagreement'] if entry else None}
        frame['calibration'] = info
        frame['calibrated'] = H is not None and info['confidence'] >= .2
        frame['pitchCoordinates'] = None if H is None else ('calibration' if info['source'] in ('landmarks', 'paint', 'auto-midfield') else 'predicted')
        tracks = mapped = 0
        for person in frame['people']:
            for key in ('pitch', 'pitchConfidence', 'pitchSource', 'pitchStatus'):
                person.pop(key, None)
            live = person.get('state') in ('confirmed', 'uncertain')
            tracks += live
            if H is None:
                person['pitchStatus'] = 'unresolved'
                continue
            fx, fy = foot_point(person['box'])
            p, c = position(H, size, (fx*width, fy*height), info['confidence'])
            if p is None or c <= 0:
                person['pitchStatus'] = 'unresolved'
                continue
            person['pitch'] = [round(float(p[0]), 4), round(float(p[1]), 4)]
            person['pitchConfidence'], person['pitchSource'] = c, info['source']
            mapped += live
        ball = frame.get('ball') or {}
        for key in ('pitch', 'pitchConfidence'):
            ball.pop(key, None)
        if H is not None and ball.get('center') and ball.get('box'):
            cx, cy = ball['center']
            p, c = position(H, size, (cx*width, min(1.0, cy+ball['box'][3]/2)*height), info['confidence'])
            if p is not None and c > 0:
                ball['pitch'], ball['pitchConfidence'] = [round(float(p[0]), 4), round(float(p[1]), 4)], c
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


def summarize(timeline, anchors):
    n = len(timeline)
    known = sum(1 for e in timeline if e['H'] is not None)
    confident = sum(1 for e in timeline if e['H'] is not None and e['confidence'] >= .5)
    sources = {}
    for e in timeline:
        sources[e['source']] = sources.get(e['source'], 0)+1
    keys = ('landmarks', 'lines', 'circlePoints', 'count', 'spread', 'axes', 'conditioning', 'residual', 'rmsMetres', 'rmsPixels', 'circleCoverage', 'coverage')
    return {'frames': n, 'calibratedFrames': known, 'calibratedShare': round(known/n, 3) if n else 0.0,
            'confidentShare': round(confident/n, 3) if n else 0.0,
            'meanConfidence': round(float(np.mean([e['confidence'] for e in timeline])), 3) if n else 0.0,
            'meanObservability': round(float(np.mean([e['observability'] for e in timeline if e['H'] is not None])), 3) if known else 0.0,
            'anchors': [{'frame': a.index, 'time': round(a.time, 3), 'observability': round(a.observability, 3), 'source': a.source,
                         **{k: v for k, v in a.details.items() if k in keys}} for a in sorted(anchors, key=lambda a: a.index)],
            'sources': sources}
