"""Synthetic ball-candidate layers of realistic size for the offline decoder's side-by-side tests: a minute
of broadcast-like frames (22 people, a ball that rolls, flies, rests at feet and hides, boots and socks
glued to players, a static spot, blips), with camera motion, unreliable frames and a cut. The numbers are
random but shaped like the real result files (normalized boxes, detector scores, the causal pass's scores,
statuses and reasons, tracklet ids, pixels per metre)."""
import math

import numpy as np

W, H = 1920, 1080
STATUSES = ('candidate', 'rejected')
REASONS = ('TRAJECTORY', 'LOW_SCORE', 'JUMP', 'NO_ENTRY', 'OTHER_OBJECT', 'STATIC', 'FIELD_LINE', 'PLAYER_ATTACHED', 'UNCONFIRMED')


def _box(cx, cy, d):
    return [(cx-d/2)/W, (cy-d/2)/H, d/W, d/H]


def _candidate(rng, cx, cy, d, det, tracklet, status='candidate', reason=None, pxm=None, **scores):
    s = {'size': float(rng.uniform(.3, 1.0)), 'shape': float(rng.uniform(.2, 1.0)), 'isolation': float(rng.uniform(.3, 1.0)),
         'field': float(rng.choice([1.0, .9, .7])), 'line': float(rng.choice([0.0, 0.0, 0.0, .3, .6, 1.0])),
         'attach': float(rng.choice([0.0, 0.0, .2, .5, .7])), 'static': float(rng.choice([0.0, 0.0, 0.0, 1.0])), 'otherObject': bool(rng.random() < .15)}
    s.update(scores)
    c = {'box': _box(cx, cy, d), 'det': float(det), 'score': float(det), 'status': status, 'scores': s, 'tracklet': tracklet,
         'pxm': float(pxm if pxm is not None else rng.uniform(20, 60))}
    if rng.random() < .5:
        c['visual'] = float(rng.uniform(.1, .9))
    if reason:
        c['reason'] = reason
    return c


def synthetic(seed=0, n=1500, fps=25.0, people=22, registered=False):
    """frames, motions, to_pitch (None unless registered): n frames with 2 to 10 ball candidates each."""
    rng = np.random.default_rng(seed)
    dt = 1/fps
    # People: boxes drifting slowly, one of them the ball's carrier.
    px = rng.uniform(.05, .95, people)
    py = rng.uniform(.35, .95, people)
    vx = rng.normal(0, .002, people)
    vy = rng.normal(0, .001, people)
    heights = rng.uniform(.07, .14, people)
    # The ball: a position plan (pixels) over the segment.
    ball = np.zeros((n, 2))
    x, y = 600.0, 700.0
    state, until = 'roll', 0
    hidden = np.zeros(n, bool)
    carrier = np.full(n, -1)
    for i in range(n):
        if i >= until:
            state = rng.choice(['roll', 'fly', 'rest', 'feet', 'hidden'], p=[.3, .15, .15, .25, .15])
            until = i+int(rng.integers(5, 60))
            vxb, vyb = rng.normal(0, 6), rng.normal(0, 3)
            if state == 'fly':
                vxb, vyb = rng.normal(0, 20), rng.normal(0, 8)
            who = int(rng.integers(0, people))
        if state in ('roll', 'fly'):
            x, y = x+vxb, y+vyb
        elif state == 'feet':
            x, y = (px[who]+heights[who]*.19)*W+rng.normal(0, 2), (py[who]+heights[who]*.98)*H+rng.normal(0, 2)
            carrier[i] = who
        elif state == 'hidden':
            hidden[i] = True
        x, y = min(max(x, 20.0), W-20.0), min(max(y, 380.0), H-20.0)
        ball[i] = (x, y)
        px += vx
        py += vy
        px, py = np.clip(px, .02, .95), np.clip(py, .3, .95)
    frames, motions = [], []
    tracklet = 1
    ball_tracklet = tracklet
    boots = {k: tracklet+1+k for k in range(people)}
    tracklet += people+1
    spot = (1500.0, 500.0)
    spot_tracklet = tracklet
    tracklet += 1
    reliable_run = True
    for i in range(n):
        time = i*dt
        # Camera: a slow pan with occasional unreliable frames and one cut in the middle.
        cut = i == n//2
        if rng.random() < .03:
            reliable_run = not reliable_run
        reliable = reliable_run or cut
        M = np.array([[1.0, 0.0, rng.normal(0, 4)], [0.0, 1.0, rng.normal(0, 1)], [0.0, 0.0, 1.0]])
        if i == 0:
            M = np.eye(3)
        motions.append((None if not reliable else M, reliable, cut))
        people_rows = []
        for k in range(people):
            hx = px[k]+rng.normal(0, .0005)
            people_rows.append({'track': k+1, 'box': [float(hx), float(py[k]-heights[k]), float(heights[k]*.38*H/W), float(heights[k])],
                                'state': 'confirmed' if k < people-2 else 'rejected'})
        cands = []
        pxm_here = float(np.median(heights)*H/1.8)
        if not hidden[i] and rng.random() < .85:
            seen = rng.random() < .7
            det = rng.uniform(.25, .9) if seen else rng.uniform(.09, .3)
            d = rng.uniform(8, 16)
            cands.append(_candidate(rng, ball[i, 0]+rng.normal(0, 1), ball[i, 1]+rng.normal(0, 1), d, det, ball_tracklet, 'ball' if seen else 'candidate',
                                    None, pxm_here, attach=0.0, static=0.0, otherObject=False))
        # Boots and socks on a few players' lower bodies (the same place, the same tracklet, frame after frame).
        for k in rng.choice(people, size=int(rng.integers(0, 4)), replace=False):
            bx = (px[k]+heights[k]*.19*.5)*W
            by = (py[k]-heights[k]*.1)*H
            cands.append(_candidate(rng, bx, by, rng.uniform(6, 12), rng.uniform(.1, .7), boots[k], 'rejected' if rng.random() < .6 else 'candidate',
                                    'PLAYER_ATTACHED' if rng.random() < .5 else None, pxm_here, attach=float(rng.uniform(.3, .9))))
        # A static spot seen on and off.
        if rng.random() < .5:
            cands.append(_candidate(rng, spot[0]+rng.normal(0, .5), spot[1]+rng.normal(0, .5), 9, rng.uniform(.1, .6), spot_tracklet,
                                    'rejected' if rng.random() < .5 else 'candidate', 'STATIC' if rng.random() < .5 else None, pxm_here, static=1.0))
        # Blips: new tracklets or short-lived ones.
        for _ in range(int(rng.integers(0, 4))):
            if rng.random() < .3:
                tracklet += 1
            cands.append(_candidate(rng, rng.uniform(20, W-20), rng.uniform(380, H-20), rng.uniform(5, 20), rng.uniform(.09, .8),
                                    tracklet if rng.random() < .8 else None, STATUSES[int(rng.random() < .4)], REASONS[int(rng.integers(0, len(REASONS)))] if rng.random() < .5 else None,
                                    pxm_here))
        if rng.random() < .05:
            cands.append(_candidate(rng, rng.uniform(20, W-20), rng.uniform(380, H-20), 60, .5, None, 'rejected', 'SIZE', pxm_here))
        rng.shuffle(cands)
        if hidden[i]:
            first = {'state': 'MISSING', 'phase': 'OCCLUDED', 'confidence': .5, 'track': 1, 'center': [ball[i, 0]/W, ball[i, 1]/H],
                     'box': _box(ball[i, 0], ball[i, 1], 10), 'nearTrack': int(carrier[i-1])+1 if i and carrier[i-1] >= 0 else None}
        elif cands and cands[0]['status'] == 'ball' or any(c['status'] == 'ball' for c in cands):
            c = next(c for c in cands if c['status'] == 'ball')
            first = {'state': 'TRACKED', 'phase': 'LOCKED', 'confidence': .8, 'track': 1, 'center': [c['box'][0]+c['box'][2]/2, c['box'][1]+c['box'][3]/2], 'box': c['box']}
        else:
            first = {'state': 'UNKNOWN', 'phase': 'SEARCHING', 'confidence': 0.0}
        frame = {'time': round(time, 6), 'frame': i, 'people': people_rows, 'ballCandidates': cands[:12], 'ball': first}
        if registered:
            frame['pitchConfidence'] = float(rng.choice([.9, .7, .4]))
        frames.append(frame)
    to_pitch = None
    if registered:
        def to_pitch(i, point):
            x, y = point
            return (x/W, .3+.7*y/H) if i % 7 else None
    return frames, motions, to_pitch


def same_path(a, b):
    """Two decode outputs are the same path: the same nodes (by identity) and the same links, bit for bit."""
    if len(a) != len(b):
        return False, f'lengths {len(a)} vs {len(b)}'
    for i, (x, y) in enumerate(zip(a, b)):
        if (x is None) != (y is None):
            return False, f'frame {i}: {x is None} vs {y is None}'
        if x is None:
            continue
        if x[0] is not y[0]:
            return False, f'frame {i}: different node'
        la, lb = x[1], y[1]
        if (la is None) != (lb is None):
            return False, f'frame {i}: link {la} vs {lb}'
        if la is None:
            continue
        if la['from'] != lb['from'] or la['node'] is not lb['node'] or la['skipped'] != lb['skipped']:
            return False, f'frame {i}: link {la["from"]}/{la["skipped"]} vs {lb["from"]}/{lb["skipped"]}'
        for key in ('cost', 'speed'):
            u, v = la[key], lb[key]
            if not (u == v or (math.isnan(u) and math.isnan(v))) or math.copysign(1, u) != math.copysign(1, v):
                return False, f'frame {i}: link {key} {u!r} vs {v!r}'
    return True, ''
