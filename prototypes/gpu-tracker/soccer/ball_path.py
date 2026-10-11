"""The final ball trajectory, decoded offline over the whole segment.

The causal tracker (ball.py) decides frame by frame. Here the video is known end to end, so the
question becomes: which sequence of this frame's candidates, that frame's candidates and "the ball is
not visible" forms ONE physically plausible ball path? Every frame is a layer of nodes (the recorded
ball candidates, scored for ball-likeness, plus an explicit NOT VISIBLE node); a path pays for the
nodes it uses and for the links between them (camera-compensated displacement, the speed that
implies, a jump straight from one player's lower body to another's), for frames it bridges inside a
link (the ball occluded on its way), and for starting or ending a trajectory. Links may skip a few
frames, so a ball seen at frame 100 and 104 need not be "found" in 101-103. Dynamic programming
(Viterbi) picks the cheapest path. A boot that the first pass took for the ball while the real ball
was hidden is a short, isolated run that connects to nothing: it never pays for its start, and the
decoded path leaves those frames NOT VISIBLE instead.

The result's per-frame `ball`, the 2D view and the ball metrics use this trajectory; the first pass
stays available as `firstPass` for comparison.
"""
import math
from dataclasses import dataclass

import numpy as np

from .geometry import clamp

NULL_COST = 1.0      # per frame: the ball is not visible, nothing known (about -log .37)
SKIP_COST = .8       # per frame bridged inside a link: occluded on a known trajectory (dearer than a plausible
                     # sighting, -log .45, so the path keeps the sightings it has; cheaper than NOT VISIBLE)
START_COST = 3.2     # a trajectory starting from nothing: an isolated sighting never pays for it
FOOT_START = 4.0     # ... more when it starts on a player's lower body: nothing arrived there (the entry rule)
END_COST = .6        # a trajectory ending
MAX_SKIP_S = 1.2     # longest gap a link may bridge; longer gaps go through NOT VISIBLE and a new start
V0 = 25.0            # m/s at which a link's motion cost reaches 1 (a pass at 12 m/s costs .23, a 40 m/s shot 2.6)
FOOT_SWITCH = 5.0    # a link from one player's lower body straight to another's with nothing seen travelling,
FOOT_FADE = 2.0      # fading over this many seconds: a real pass is seen in flight, an unseen one stays NOT VISIBLE
FOOT_ENTER = 1.5     # a link arriving at a player's lower body that the causal pass did not see arrive (a new object there)
ATTACHED = .55       # ball-likeness factor of a candidate glued to the same spot on one player's lower body
ATTACHED_LONG = .3   # ... for a second or more (a boot, a sock)
ELSEWHERE = .5       # ball-likeness factor of a candidate the causal pass saw while it confidently tracked the ball elsewhere
MIN_RUN = 4          # sightings a trajectory needs to be kept on the final path
UNRELIABLE = .3      # extra cost per frame of unreliable camera motion inside a link
BALL_DIAMETER = .22


@dataclass
class Node:
    frame: int
    x: float           # pixels
    y: float
    d: float           # diameter, px
    q: float           # ball-likeness (0..1) from the recorded evidence
    pid: object        # the player whose lower body it sits in, else None
    near: object       # the player within reach of it (a touch, a dribble), else None
    u: float
    v: float
    bw: float          # that player's box, px
    bh: float
    pxm: float         # pixels per metre at this depth
    tracklet: object
    source: dict


def lower_body(people, x, y, W, H):
    """(player track, u, v, box w px, box h px) of the player whose lower box holds the point, or None."""
    best = None
    for p in people:
        bx, by, bw, bh = p['box']
        if bw <= 0 or bh <= 0:
            continue
        u, v = (x/W-bx)/bw, (y/H-by)/bh
        if -.1 <= u <= 1.1 and .6 <= v <= 1.08 and (best is None or abs(u-.5) < best[0]):
            best = (abs(u-.5), p.get('track'), u, v, bw*W, bh*H)
    return best[1:] if best else None


def within_reach(people, x, y, W, H, pxm, reach=1.5):
    """The player whose feet are within `reach` metres of the point (image rows foreshortened 2x), or None."""
    best = None
    for p in people:
        bx, by, bw, bh = p['box']
        if bw <= 0 or bh <= 0:
            continue
        fx, fy = (bx+bw/2)*W, (by+bh)*H
        metres = math.hypot(x-fx, 2.0*(y-fy))/max(pxm, 1e-3)
        if metres <= reach and (best is None or metres < best[0]):
            best = (metres, p.get('track'))
    return best[1] if best else None


def _pixels_per_metre(people, y, H):
    """From the people near this image row (1.8 m tall), else a broadcast default."""
    heights = sorted(p['box'][3]*H for p in people if p['box'][3] > 0 and abs(p['box'][1]+p['box'][3]-y/H) < .25)
    if heights:
        return heights[len(heights)//2]/1.8
    return .09*H/1.8


def nodes_of(i, frame, W, H):
    people = frame.get('people', [])
    out = []
    for c in frame.get('ballCandidates', []):
        if c.get('reason') in ('SIZE', 'OUTSIDE_PITCH'):
            continue
        x, y, w, h = c['box']
        cx, cy = (x+w/2)*W, (y+h/2)*H
        d = math.sqrt(max(w*W*h*H, 1.0))
        s = c.get('scores', {})
        visual = c.get('visual')
        if visual is None:
            det = clamp((c.get('det', 0)-.08)/.45)
            visual = .4*det+.2*s.get('size', .5)+.15*s.get('shape', .5)+.15*s.get('isolation', .5)+.1*s.get('field', .7)
        q = visual*(1-.5*s.get('line', 0.0))*(1-.7*s.get('attach', 0.0))*(1-.6*s.get('static', 0.0))*(ELSEWHERE if s.get('otherObject') else 1.0)
        if c.get('status') == 'ball':
            q = min(1.0, q*1.15)   # the causal pass, with its full-resolution look, took it
        lower = lower_body(people, cx, cy, W, H)
        pid, u, v, bw, bh = lower if lower is not None else (None, 0.0, 0.0, 0.0, 0.0)
        pxm = c.get('pxm') or _pixels_per_metre(people, cy, H)
        near = pid if pid is not None else within_reach(people, cx, cy, W, H, pxm)
        out.append(Node(i, cx, cy, d, max(q, 1e-3), pid, near, u, v, bw, bh, pxm, c.get('tracklet'), c))
    return out


def _attachment(layers, fps):
    """A candidate that stays at the same place on one player's lower body over the previous frames
    moves with that player: boot-like, the more so the longer it lasts. Its ball-likeness is reduced
    (never zeroed: the ball is at players' feet most of the time, and the links decide)."""
    short, long = max(2, int(fps*.4)), max(3, int(fps*1.2))
    for i, nodes in enumerate(layers):
        for n in nodes:
            if n.pid is None:
                continue
            still = lambda m: math.hypot((m.u-n.u)*n.bw, (m.v-n.v)*n.bh)/max(n.pxm, 1e-3) < .25
            recent = [m for j in range(max(0, i-short), i) for m in layers[j] if m.pid == n.pid]
            if len(recent) < 3 or not all(still(m) for m in recent):
                continue
            older = [m for j in range(max(0, i-long), max(0, i-short)) for m in layers[j] if m.pid == n.pid and still(m)]
            n.q = max(1e-3, n.q*(ATTACHED_LONG if len(older) >= fps*.5 else ATTACHED))


def _chain(motions):
    """Cumulative camera transforms (frame 0 -> frame i) and where links may not cross."""
    n = len(motions)
    T = [np.eye(3)]
    breaks = [False]*n
    unreliable = [False]*n
    for i in range(1, n):
        M, reliable, cut = motions[i]
        breaks[i] = bool(cut)
        unreliable[i] = not reliable or M is None
        T.append((np.asarray(M, np.float64) @ T[-1]) if (M is not None and reliable) else T[-1].copy())
    return T, breaks, unreliable


def decode(layers, times, motions, fps):
    """Viterbi over the layers. Returns per frame either None (not visible) or (node, link) where link
    is {'from': frame index, 'node': Node, 'cost': motion cost, 'speed': m/s} or None for a start."""
    n = len(layers)
    T, breaks, unreliable = _chain(motions)
    K = max(1, int(round(MAX_SKIP_S*fps)))
    best_null = [math.inf]*n
    back_null = [None]*n          # ('null', i-1) or ('node', i-1, j)
    best = [np.full(len(l), math.inf) for l in layers]
    back = [[None]*len(l) for l in layers]
    link = [[None]*len(l) for l in layers]
    for i in range(n):
        # Not visible at i.
        if i == 0:
            best_null[i], back_null[i] = NULL_COST, None
        else:
            options = [(best_null[i-1], ('null', i-1))]
            if len(layers[i-1]):
                j = int(np.argmin(best[i-1]))
                options.append((best[i-1][j]+END_COST, ('node', i-1, j)))
            best_null[i], back_null[i] = min(options, key=lambda o: o[0])
            best_null[i] += NULL_COST
        if not layers[i]:
            continue
        xy = np.array([(m.x, m.y) for m in layers[i]])
        pxm = np.array([m.pxm for m in layers[i]])
        start = START_COST if i == 0 else best_null[i-1]+START_COST
        for j, m in enumerate(layers[i]):
            best[i][j], back[i][j] = start+(FOOT_START if m.pid is not None else 0.0), ('null', i-1)
        for k in range(1, K+1):
            h = i-k
            if h < 0 or any(breaks[h+1:i+1]):
                break
            if not layers[h] or not np.isfinite(best[h]).any():
                continue
            A = T[i] @ np.linalg.inv(T[h])
            prev = np.array([(m.x, m.y, 1.0) for m in layers[h]]) @ A.T
            prev = prev[:, :2]/prev[:, 2:3]
            dt = max(1e-3, times[i]-times[h])
            dist = np.hypot(prev[:, None, 0]-xy[None, :, 0], prev[:, None, 1]-xy[None, :, 1])
            scale = (np.array([m.pxm for m in layers[h]])[:, None]+pxm[None, :])/2
            speed = dist/np.maximum(scale, 1e-3)/dt
            # The displacement an unseen gap must explain grows dearer with the gap: a long flight that
            # nobody saw needs more than a plausible average speed.
            motion = (speed/V0)**2*(1+dt)+(k-1)*SKIP_COST+UNRELIABLE*sum(unreliable[h+1:i+1])
            for a, ma in enumerate(layers[h]):
                for b, mb in enumerate(layers[i]):
                    cost = motion[a, b]
                    if k == 1 and ma.tracklet is not None and ma.tracklet == mb.tracklet:
                        cost = min(cost, .05+UNRELIABLE*unreliable[i])
                    # Arriving at a player's lower body: free when the causal pass saw it arrive or the ball was
                    # already with that player; dear when it comes from another player's feet or reach with
                    # nothing seen travelling (an unseen pass stays NOT VISIBLE; a boot never becomes the ball).
                    if mb.pid is not None and ma.near != mb.pid and not (k == 1 and ma.tracklet is not None and ma.tracklet == mb.tracklet):
                        cost += FOOT_ENTER
                        if ma.near is not None:
                            cost += FOOT_SWITCH*max(0.0, 1-dt/FOOT_FADE)
                    total = best[h][a]+cost
                    if total < best[i][b]:
                        best[i][b], back[i][b] = total, ('node', h, a)
                        link[i][b] = {'from': h, 'node': ma, 'cost': float(motion[a, b] if k == 1 else cost), 'speed': float(speed[a, b]), 'skipped': k-1}
        for j, m in enumerate(layers[i]):
            best[i][j] += -math.log(m.q)
    # Backtrack.
    path = [None]*n
    if n == 0:
        return path
    state = ('null', n-1)
    if len(layers[n-1]):
        j = int(np.argmin(best[n-1]))
        if best[n-1][j] < best_null[n-1]:
            state = ('node', n-1, j)
    while state is not None:
        if state[0] == 'null':
            i = state[1]
            if i < 0:
                break
            state = back_null[i]
        else:
            _, i, j = state
            path[i] = (layers[i][j], link[i][j])
            state = back[i][j]
    return path


def _runs(path):
    """Consecutive stretches of the path, with bridged frames: [(first, last, [frames with sightings])]."""
    runs = []
    current = None
    for i, entry in enumerate(path):
        if entry is None:
            continue
        node, lk = entry
        if current is not None and lk is not None:   # a link always joins the previous sighting on the path
            current[1] = i
            current[2].append(i)
        else:
            if current is not None:
                runs.append(tuple(current))
            current = [i, i, [i]]
    if current is not None:
        runs.append(tuple(current))
    return runs


def shown_jumps(frames, W, H):
    """Teleports (the shown ball moving faster than any shot between consecutive frames) and foot-to-foot
    switches (the shown ball in one player's lower body, then in another's within 0.3 s) of a result."""
    teleports = switches = 0
    prev = None
    for f in frames:
        b = f['ball']
        if b['state'] == 'UNKNOWN':
            prev = None
            continue
        x, y = b['center'][0]*W, b['center'][1]*H
        lower = lower_body(f.get('people', []), x, y, W, H)
        pid = lower[0] if lower else None
        pxm = _pixels_per_metre(f.get('people', []), y, H)
        if prev is not None:
            dt = f['time']-prev['t']
            dist_m = math.hypot(x-prev['x'], y-prev['y'])/pxm
            if 0 < dt <= .15 and dist_m/dt > 40.0:
                teleports += 1
            if pid is not None and prev['pid'] is not None and pid != prev['pid'] and dt <= .3 and dist_m > 1.0:
                switches += 1
        prev = {'t': f['time'], 'x': x, 'y': y, 'pid': pid}
    return {'teleports': teleports, 'footSwitches': switches}


def finalize(frames, motions, W, H, fps, to_pitch=None):
    """Decode the final trajectory and rewrite frames[i]['ball'] (the first pass is kept as 'firstPass').
    motions: per frame (M pixels previous->this or None, reliable, cut). to_pitch(i, (x, y) px) ->
    normalized pitch position or None. Returns a report: shares, jumps before and after, dropped
    first-pass runs and their explanations."""
    n = len(frames)
    before = shown_jumps(frames, W, H)
    first = [dict(f['ball']) for f in frames]
    layers = [nodes_of(i, f, W, H) for i, f in enumerate(frames)]
    _attachment(layers, fps)
    times = [f['time'] for f in frames]
    path = decode(layers, times, motions, fps)
    runs = [r for r in _runs(path) if len(r[2]) >= MIN_RUN]
    kept = set()
    for first_i, last_i, sightings in runs:
        kept.update(range(first_i, last_i+1))
    T, _, _ = _chain(motions)
    final = [None]*n
    for first_i, last_i, sightings in runs:
        qs = [path[i][0].q for i in sightings]
        quality = sum(qs)/len(qs)
        support = clamp(len(sightings)/(fps*.6))
        identity = round(min(.99, quality*(.5+.5*support)), 3)
        for k, i in enumerate(sightings):
            node, lk = path[i]
            traj = 1.0 if lk is None else round(math.exp(-lk['cost']), 3)
            speed = None if lk is None else round(lk['speed'], 2)
            final[i] = {'state': 'TRACKED', 'phase': 'PATH', 'observed': True, 'confidence': identity, 'identity': identity,
                        'visual': round(node.q, 3), 'trajectory': traj, 'track': 1, 'source': 'PATH',
                        'box': node.source['box'], 'center': [round(node.x/W, 5), round(node.y/H, 5)], 'speed': speed}
            if node.pid is not None:
                final[i]['nearTrack'] = node.pid
            # Frames bridged inside the link that led here: the ball occluded on its trajectory.
            if lk is not None and lk['skipped']:
                h = lk['from']
                a, b = lk['node'], node
                for m in range(h+1, i):
                    s = (times[m]-times[h])/max(1e-6, times[i]-times[h])
                    Af = T[m] @ np.linalg.inv(T[h])
                    Ab = T[m] @ np.linalg.inv(T[i])
                    pa = Af @ np.array([a.x, a.y, 1.0])
                    pb = Ab @ np.array([b.x, b.y, 1.0])
                    pa, pb = pa[:2]/pa[2], pb[:2]/pb[2]
                    x, y = (1-s)*pa[0]+s*pb[0], (1-s)*pa[1]+s*pb[1]
                    d = (1-s)*a.d+s*b.d
                    final[m] = {'state': 'MISSING', 'phase': 'PATH', 'observed': False, 'confidence': round(identity*.8, 3), 'identity': identity,
                                'visual': None, 'trajectory': traj, 'track': 1, 'source': 'PATH', 'bridged': True,
                                'box': [round((x-d/2)/W, 5), round((y-d/2)/H, 5), round(d/W, 5), round(d/H, 5)],
                                'center': [round(x/W, 5), round(y/H, 5)], 'missingFor': round(times[m]-times[h], 2), 'speed': speed}
                    if first[m].get('nearTrack') is not None:
                        final[m]['nearTrack'] = first[m]['nearTrack']
    dropped = []
    events = []
    # First-pass sightings the final path does not use: explained.
    i = 0
    while i < n:
        if first[i]['state'] != 'TRACKED':
            i += 1
            continue
        j = i
        while j+1 < n and first[j+1]['state'] == 'TRACKED' and first[j+1].get('track') == first[i].get('track'):
            j += 1
        frames_run = list(range(i, j+1))
        agree = 0
        for m in frames_run:
            if final[m] is not None and final[m]['state'] == 'TRACKED':
                dx = (final[m]['center'][0]-first[m]['center'][0])*W
                dy = (final[m]['center'][1]-first[m]['center'][1])*H
                if math.hypot(dx, dy) <= 2*max(first[m]['box'][2]*W, 4.0):
                    agree += 1
        if agree < .6*len(frames_run):
            x, y = first[i]['center'][0]*W, first[i]['center'][1]*H
            lower = lower_body(frames[i].get('people', []), x, y, W, H)
            where = f", on track {lower[0]}'s lower body" if lower and lower[0] is not None else ''
            prev = next((m for m in range(i-1, -1, -1) if final[m] is not None and final[m]['state'] == 'TRACKED'), None)
            if prev is not None:
                A = T[i] @ np.linalg.inv(T[prev])
                p = A @ np.array([final[prev]['center'][0]*W, final[prev]['center'][1]*H, 1.0])
                dist = math.hypot(p[0]/p[2]-x, p[1]/p[2]-y)/_pixels_per_metre(frames[i].get('people', []), y, H)
                dt = max(1e-3, times[i]-times[prev])
                why = f'reaching it from the previous sighting at {times[prev]:.2f} s would have needed {dist/dt:.1f} m/s over {dt:.2f} s unseen'
            else:
                why = 'nothing leads into it'
            dropped.append({'start': round(times[i], 3), 'end': round(times[j], 3), 'frames': len(frames_run), 'player': lower[0] if lower else None, 'why': why})
            events.append({'time': round(times[i], 3), 'kind': 'ball', 'track': None, 'playerId': None,
                           'message': f'BALL PATH\nThe first pass showed the ball from {times[i]:.2f} s to {times[j]:.2f} s ({len(frames_run)} frames{where}); '
                                      f'the final trajectory leaves those frames NOT VISIBLE: {why}.'})
        i = j+1
    for i, f in enumerate(frames):
        out = final[i] if final[i] is not None else {'state': 'UNKNOWN', 'phase': 'PATH', 'confidence': 0.0, 'identity': 0.0, 'source': 'PATH'}
        if out['state'] != 'UNKNOWN' and to_pitch is not None:
            p = to_pitch(i, (out['center'][0]*W, (out['center'][1]+out['box'][3]/2)*H))
            if p is not None:
                out['pitch'] = [round(float(p[0]), 4), round(float(p[1]), 4)]
        out['firstPass'] = {'state': first[i]['state'], 'phase': first[i].get('phase'), 'confidence': first[i].get('confidence', 0.0), 'track': first[i].get('track'),
                            **({'center': first[i]['center']} if first[i].get('center') else {})}
        f['ball'] = out
    after = shown_jumps(frames, W, H)
    states = lambda rows: {k: round(sum(1 for b in rows if b['state'] == k)/max(1, n), 3) for k in ('TRACKED', 'MISSING', 'UNKNOWN')}
    longest_unknown = 0
    run = 0
    for f in frames:
        run = run+1 if f['ball']['state'] == 'UNKNOWN' else 0
        longest_unknown = max(longest_unknown, run)
    return {'firstPass': {'states': states(first), **before}, 'final': {'states': states([f['ball'] for f in frames]), **after, 'runs': len(runs),
                                                                        'longestUnknownSeconds': round(longest_unknown/fps, 2)},
            'dropped': dropped, 'events': events}
