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
from collections import Counter
from dataclasses import dataclass

import numpy as np

from .geometry import PAINTED_SPOTS, clamp

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
ELSEWHERE = .2       # ball-likeness factor of a candidate the causal pass saw while it confidently tracked the ball elsewhere: there is
                     # one ball, so such a thing never pays for a frame by itself (at .5 it was cheaper to sit on than to bridge the
                     # real ball hidden at a player)
MIN_RUN = 4          # sightings a trajectory needs to be kept on the final path
UNRELIABLE = .3      # extra cost per frame of unreliable camera motion inside a link
BALL_DIAMETER = .22
TRAVEL_SPEED = 8.0   # m/s: from here a link is a flight (and the motion state FAST); below, a roll
JUMP_SPEED = 40.0    # m/s (the causal pass's MAX_SPEED): faster than a hard shot between two sightings of different objects is not
JUMP_COST = 12.0     # the same ball; such a link costs this much more (a sighting of the same tracklet is the linker's business)
STILL_SPEED = 1.0    # m/s: a gap whose implied speed is below this is a ball that stayed where it was (hidden, or missed) ...
MAX_STILL_S = 3.0    # ... and may be bridged this long (MAX_SKIP_S otherwise): the stationary hypothesis is physical; it costs the
                     # same per frame as any bridge (cheaper, sitting on a static white thing beat following the real ball)
STILL_REGION = .5    # metres within which a stretch of sightings on the path stood still (camera-compensated). A still stretch whose
                     # sightings the causal pass ruled out (REJECTED_REASONS: the hidden ball could not have got there, it was seen
                     # while the ball was confidently elsewhere, or it is a known spot, paint or a boot) is a white thing standing
                     # where the ball was known not to be (a ball boy's ball, a spot, a bag) when nothing was seen rolling to it from
                     # the last sighting the causal pass accepted: a flight, or ENTRY_GAP_S or more unseen (rejected waypoints in
                     # between do not count as sightings), or nothing at all. It is cut out and the path decoded again. A ball at
                     # rest the causal pass tracked, found nothing against, or saw rolling there, is kept. UNCONFIRMED, LOW_SCORE and
                     # TRAJECTORY are not negatives: the real ball gets them while the causal pass re-acquires it
REJECTED_REASONS = ('JUMP', 'NO_ENTRY', 'OTHER_OBJECT', 'STATIC', 'FIELD_LINE', 'PLAYER_ATTACHED')
REJECTED_SHARE = .4  # share of a stretch's sightings with such a reason that counts as "ruled out"
ENTRY_GAP_S = .5     # seconds unseen before a still, rejected stretch that make its entry an arrival nobody saw
REVIEW_ROUNDS = 8    # decode, cut such stretches, decode again: at most this many times
DEAD_STOP = 12.0     # cost of a link that flies in (TRAVEL_SPEED or more) to a sighting that stands still with no player within reach,
                     # or leaves one: a ball does not stop dead on the grass by itself, nor start from rest without a touch. More than
                     # any bridge can earn over NOT VISIBLE (MAX_SKIP_S of SKIP_COST plus a start and an end), so such a flight is
                     # never made for its own sake; the sightings beyond it are the review's business
NEIGHBOUR_S = .3     # seconds around a sighting within which its tracklet's other sightings give its own speed
MARKING_FACTOR = .4  # ball-likeness factor of a candidate on a painted spot (SPOT_RADIUS of a penalty or centre spot under a trusted
SPOT_RADIUS = 1.0    # registration, TRUSTED) or of a tracklet that stood still for STATIC_SECONDS within STATIC_RADIUS: a field marking,
TRUSTED = .5         # never worth a frame by itself, unless the causal pass tracked the ball there (a penalty about to be taken)
STATIC_SECONDS = 2.0
STATIC_RADIUS = .6
ISLAND_S = 1.0       # a run shorter than this that no ball could have reached from the run before it nor left for the run after it
                     # (both faster than JUMP_SPEED) is an island: a white thing seen while the ball was known to be elsewhere, cut
STRONG_DET = .3      # detector confidence and ...
STRONG_Q = .5        # ... ball-likeness from which a sighting is a confirmed OBSERVATION; below, a WEAK_DETECTION (labelled, kept)


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
    own_speed: object = None   # m/s, from the tracklet's neighbouring sightings (None: unknown)
    pitch: object = None       # normalized pitch position from the offline registration, when trusted
    marking: bool = False      # on a painted spot, or standing still for seconds: a field marking unless the causal pass tracked the ball there


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


def _on_painted_spot(p):
    return p is not None and any(math.hypot((p[0]-x)*105, (p[1]-y)*68) <= SPOT_RADIUS for x, y in PAINTED_SPOTS)


def nodes_of(i, frame, W, H, to_pitch=None):
    # People rejected as outside the pitch (ball boys, staff) are not players who can take, stop or hide the ball.
    people = [p for p in frame.get('people', []) if p.get('state') != 'rejected']
    trusted = (frame.get('pitchConfidence') or (frame.get('calibration') or {}).get('confidence') or 0.0) >= TRUSTED
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
        pitch = to_pitch(i, (cx, cy+d/2)) if to_pitch is not None and trusted else None
        # A white thing on a penalty or centre spot of the registered pitch is the spot, unless the causal pass
        # tracked the ball there (a penalty about to be taken): a known field marking, never worth a frame by itself.
        marking = _on_painted_spot(pitch) and c.get('status') != 'ball'
        if marking:
            q *= MARKING_FACTOR
        n = Node(i, cx, cy, d, max(q, 1e-3), pid, near, u, v, bw, bh, pxm, c.get('tracklet'), c)
        n.pitch, n.marking = pitch, marking
        out.append(n)
    return out


def _standing_markings(layers, times, T):
    """A tracklet whose sightings span STATIC_SECONDS and all lie within STATIC_RADIUS of the first (on the pitch when
    registered, else camera-compensated in the image) is a thing standing still for seconds: a field marking, debris,
    a bag. Its sightings lose ball-likeness like a painted spot, unless the causal pass tracked the ball there (a
    ball at rest before a free kick). Returns the number of sightings discounted."""
    by_tracklet = {}
    for i, nodes in enumerate(layers):
        for m in nodes:
            if m.tracklet is not None:
                by_tracklet.setdefault(m.tracklet, []).append((i, m))
    discounted = 0
    for entries in by_tracklet.values():
        if times[entries[-1][0]]-times[entries[0][0]] < STATIC_SECONDS or len(entries) < 3:
            continue
        i0, first = entries[0]
        still = True
        for i, m in entries[1:]:
            if first.pitch is not None and m.pitch is not None:
                metres = math.hypot((m.pitch[0]-first.pitch[0])*105, (m.pitch[1]-first.pitch[1])*68)
            else:
                p = T[i0] @ np.linalg.inv(T[i]) @ np.array([m.x, m.y, 1.0])
                metres = math.hypot(p[0]/p[2]-first.x, p[1]/p[2]-first.y)/max((first.pxm+m.pxm)/2, 1e-3)
            if metres > STATIC_RADIUS:
                still = False
                break
        if not still:
            continue
        for _, m in entries:
            if m.source.get('status') != 'ball' and not m.marking:
                m.q, m.marking = max(1e-3, m.q*MARKING_FACTOR), True
                discounted += 1
    return discounted


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


def _velocities(layers, times, T):
    """Each node's own speed (m/s, camera-compensated) from the nearest earlier and later sightings of its
    tracklet within NEIGHBOUR_S; None without a neighbour."""
    by_tracklet = {}
    for i, nodes in enumerate(layers):
        for m in nodes:
            if m.tracklet is not None:
                by_tracklet.setdefault(m.tracklet, []).append((i, m))
    for entries in by_tracklet.values():
        for k, (i, m) in enumerate(entries):
            earlier = next(((j, q) for j, q in reversed(entries[:k]) if times[i]-times[j] <= NEIGHBOUR_S), None)
            later = next(((j, q) for j, q in entries[k+1:] if times[j]-times[i] <= NEIGHBOUR_S), None)
            (ja, a), (jb, b) = earlier or (i, m), later or (i, m)
            if ja == jb:
                continue
            pa = T[i] @ np.linalg.inv(T[ja]) @ np.array([a.x, a.y, 1.0])
            pb = T[i] @ np.linalg.inv(T[jb]) @ np.array([b.x, b.y, 1.0])
            m.own_speed = math.hypot(pb[0]/pb[2]-pa[0]/pa[2], pb[1]/pb[2]-pa[1]/pa[2])/max(m.pxm, 1e-3)/max(1e-3, times[jb]-times[ja])


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
    is {'from': frame index, 'node': Node, 'cost': motion cost, 'speed': m/s, 'skipped': bridged frames}
    or None for a start. A link may bridge up to MAX_SKIP_S of unseen frames, or up to MAX_STILL_S when
    the speed it implies says the ball stayed where it was (STILL_SPEED). A flight into or out of a ball
    standing still with nobody near it (DEAD_STOP), or faster than a shot between two different things
    (JUMP_COST), is paid for: the ball does neither by itself."""
    n = len(layers)
    T, breaks, unreliable = _chain(motions)
    cuts = np.cumsum(np.asarray(breaks, int))
    shaky = np.cumsum(np.asarray(unreliable, int))
    K = max(1, int(round(MAX_SKIP_S*fps)))
    K_still = max(K, int(round(MAX_STILL_S*fps)))
    best_null = [math.inf]*n
    back_null = [None]*n          # ('null', i-1) or ('node', i-1, j)
    best = [np.full(len(l), math.inf) for l in layers]
    back = [[None]*len(l) for l in layers]
    link = [[None]*len(l) for l in layers]
    # Node positions in the coordinates of frame 0: a link from h to i maps them through T[i].
    origin = [np.array([(m.x, m.y, 1.0) for m in l]) @ np.linalg.inv(T[i]).T if l else None for i, l in enumerate(layers)]
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
        cols = np.arange(len(layers[i]))
        for k in range(1, K_still+1):
            h = i-k
            if h < 0 or cuts[i]-cuts[h] > 0:
                break
            if not layers[h] or not np.isfinite(best[h]).any():
                continue
            prev = origin[h] @ T[i].T
            prev = prev[:, :2]/prev[:, 2:3]
            dt = max(1e-3, times[i]-times[h])
            dist = np.hypot(prev[:, None, 0]-xy[None, :, 0], prev[:, None, 1]-xy[None, :, 1])
            scale = (np.array([m.pxm for m in layers[h]])[:, None]+pxm[None, :])/2
            speed = dist/np.maximum(scale, 1e-3)/dt
            if k > K:
                still = speed <= STILL_SPEED
                if not still.any():
                    continue
            # The displacement an unseen gap must explain grows dearer with the gap: a long flight that
            # nobody saw needs more than a plausible average speed.
            motion = (speed/V0)**2*(1+dt)+(k-1)*SKIP_COST+UNRELIABLE*(shaky[i]-shaky[h])
            cost = motion.copy()
            same = np.zeros(cost.shape, bool)
            if k == 1:
                same = np.array([[ma.tracklet is not None and ma.tracklet == mb.tracklet for mb in layers[i]] for ma in layers[h]], bool).reshape(cost.shape)
                cost = np.where(same, np.minimum(cost, .05+UNRELIABLE*unreliable[i]), cost)
            # Arriving at a player's lower body: free when the causal pass saw it arrive or the ball was
            # already with that player; dear when it comes from another player's feet or reach with
            # nothing seen travelling (an unseen pass stays NOT VISIBLE; a boot never becomes the ball).
            near_a = [ma.near for ma in layers[h]]
            enter = np.array([[mb.pid is not None and na != mb.pid for mb in layers[i]] for na in near_a], bool).reshape(cost.shape) & ~same
            from_feet = np.array([na is not None for na in near_a], bool)[:, None]
            cost = cost+enter*FOOT_ENTER+(enter & from_feet)*(FOOT_SWITCH*max(0.0, 1-dt/FOOT_FADE))
            # A flight cannot leave a ball standing still, nor end on one, with no player within reach of it: the
            # ball does not start from rest or stop dead on the grass by itself.
            if k > 1 or not same.all():
                still_a = np.array([ma.own_speed is not None and ma.own_speed < STILL_SPEED and ma.near is None for ma in layers[h]], bool)[:, None]
                still_b = np.array([mb.own_speed is not None and mb.own_speed < STILL_SPEED and mb.near is None for mb in layers[i]], bool)[None, :]
                flight = speed >= TRAVEL_SPEED
                cost = cost+np.where(flight & (still_a | still_b) & ~same, DEAD_STOP, 0.0)
                # Faster than a hard shot between sightings of two different things: not the same ball.
                cost = cost+np.where((speed > JUMP_SPEED) & ~same, JUMP_COST, 0.0)
            if k > K:
                cost = np.where(still, cost, np.inf)
            total = best[h][:, None]+cost
            a_best = np.argmin(total, axis=0)
            t_best = total[a_best, cols]
            for b in np.flatnonzero(t_best < best[i]):
                a = int(a_best[b])
                best[i][b], back[i][b] = float(t_best[b]), ('node', h, a)
                link[i][b] = {'from': h, 'node': layers[h][a], 'cost': float(motion[a, b] if k == 1 else cost[a, b]), 'speed': float(speed[a, b]), 'skipped': k-1}
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


def _metres_between(path, i, j, T):
    """Metres between the sightings at frames i and j of the path, camera-compensated to frame i."""
    a, m = path[i][0], path[j][0]
    p = T[i] @ np.linalg.inv(T[j]) @ np.array([m.x, m.y, 1.0])
    return math.hypot(p[0]/p[2]-a.x, p[1]/p[2]-a.y)/max((a.pxm+m.pxm)/2, 1e-3)


def _strong(node):
    return node.source.get('det', 0.0) >= STRONG_DET and node.q >= STRONG_Q


def _review(path, runs, times, T, layers, fps):
    """Still stretches on the path (every sighting within STILL_REGION of the stretch's first, at least MIN_RUN
    of them) whose sightings the causal pass rejected (it followed the ball somewhere else, or knew the thing
    for a spot, paint or a boot), and that nothing was seen rolling to: the stretch starts a trajectory, or the
    link into it is a flight (TRAVEL_SPEED or more) or bridged ENTRY_GAP_S unseen. Such a stretch is a white thing standing where the ball was known
    not to be, not the ball. The ball at rest stays on the path when it rolled there, or when the causal pass
    had nothing against it (a ball waiting for a kick-off). layers and fps are kept for the callers' sake.
    Returns the stretches to cut out with what was seen."""
    out = []
    for first, last, sightings in runs:
        k = 0
        while k < len(sightings):
            stretch = [sightings[k]]
            while k+len(stretch) < len(sightings) and _metres_between(path, stretch[0], sightings[k+len(stretch)], T) < STILL_REGION:
                stretch.append(sightings[k+len(stretch)])
            k += len(stretch)
            nodes = [path[i][0] for i in stretch]
            # Fewer sightings than a run are not judged, except when every one of them is a known marking: a
            # painted spot or a thing standing for seconds is never a stepping stone across a gap.
            if len(stretch) < MIN_RUN and not all(n.marking for n in nodes):
                continue
            duration = times[stretch[-1]]-times[stretch[0]]
            moved = max((_metres_between(path, stretch[0], i, T) for i in stretch[1:]), default=0.0)
            if duration > 0 and moved/duration >= STILL_SPEED:
                continue   # a rolling ball passes through half a metre in a few frames: not a still stretch
            elsewhere = sum(1 for n in nodes if n.source.get('reason') in REJECTED_REASONS or n.marking)
            if elsewhere < REJECTED_SHARE*len(nodes):
                continue   # the causal pass had nothing against these sightings: a ball at rest is kept
            tracklets = {n.tracklet for n in nodes if n.tracklet is not None}
            # The entry, judged from the last sighting before the stretch that the causal pass accepted: rejected
            # waypoints in between (a spot, a boot the path stepped on) are not sightings of the ball.
            previous, lk = stretch[0], path[stretch[0]][1]
            while lk is not None and path[lk['from']][0].source.get('reason') and times[stretch[0]]-times[lk['from']] <= MAX_STILL_S:
                previous, lk = lk['from'], path[lk['from']][1]
            previous = lk['from'] if lk is not None else None
            gap = None
            if previous is not None:
                a = path[previous][0]
                seconds = times[stretch[0]]-times[previous]
                metres = _metres_between(path, stretch[0], previous, T)
                speed = metres/max(seconds, 1e-3)
                if speed < TRAVEL_SPEED and seconds < ENTRY_GAP_S:
                    continue   # the ball was seen rolling here: kept
                gap = {'time': times[previous], 'x': a.x, 'y': a.y, 'metres': metres, 'seconds': seconds, 'speed': speed}
            out.append({'first': stretch[0], 'last': stretch[-1], 'sightings': stretch, 'moved': moved, 'elsewhere': elsewhere,
                        'reasons': dict(Counter(('MARKING' if n.marking else n.source.get('reason')) or 'none' for n in nodes)),
                        'det': sum(n.source.get('det', 0.0) for n in nodes)/len(nodes), 'q': sum(n.q for n in nodes)/len(nodes),
                        'strong': sum(1 for n in nodes if _strong(n)), 'tracklets': tracklets, 'marking': all(n.marking for n in nodes),
                        'x': nodes[0].x, 'y': nodes[0].y, 'previous': gap, 'kind': 'still',
                        'entered': 'from nothing' if gap is None else f"{gap['seconds']:.2f} s and {gap['metres']:.1f} m after the last accepted sighting ({gap['speed']:.1f} m/s)"})
    # Islands: a short run that no ball could have reached from the run before it, nor left for the run after it.
    # Whatever it was, the ball was somewhere else just before and just after; the frames stay unknown.
    cut = {i for s in out for i in range(s['first'], s['last']+1)}
    for k in range(1, len(runs)-1):
        first, last, sightings = runs[k]
        if times[last]-times[first] >= ISLAND_S or any(i in cut for i in sightings):
            continue
        before, after = runs[k-1][1], runs[k+1][0]
        entry_m, exit_m = _metres_between(path, before, first, T), _metres_between(path, last, after, T)
        entry_s, exit_s = times[first]-times[before], times[after]-times[last]
        entry, exit_ = entry_m/max(entry_s, 1e-3), exit_m/max(exit_s, 1e-3)
        if entry <= JUMP_SPEED or exit_ <= JUMP_SPEED:
            continue
        nodes = [path[i][0] for i in sightings]
        a, b = path[before][0], path[after][0]
        out.append({'first': first, 'last': last, 'sightings': list(sightings), 'moved': max((_metres_between(path, first, i, T) for i in sightings[1:]), default=0.0),
                    'elsewhere': sum(1 for n in nodes if n.source.get('reason') in REJECTED_REASONS or n.marking),
                    'reasons': dict(Counter(('MARKING' if n.marking else n.source.get('reason')) or 'none' for n in nodes)),
                    'det': sum(n.source.get('det', 0.0) for n in nodes)/len(nodes), 'q': sum(n.q for n in nodes)/len(nodes),
                    'strong': sum(1 for n in nodes if _strong(n)), 'tracklets': {n.tracklet for n in nodes if n.tracklet is not None},
                    'marking': all(n.marking for n in nodes), 'x': nodes[0].x, 'y': nodes[0].y, 'kind': 'island',
                    'previous': {'time': times[before], 'x': a.x, 'y': a.y, 'metres': entry_m, 'seconds': entry_s, 'speed': entry},
                    'next': {'time': times[after], 'x': b.x, 'y': b.y, 'metres': exit_m, 'seconds': exit_s, 'speed': exit_},
                    'entered': f"{entry_s:.2f} s and {entry_m:.1f} m after the run before it ({entry:.0f} m/s), "
                               f"left {exit_s:.2f} s and {exit_m:.1f} m before the run after it ({exit_:.0f} m/s)"})
    return out


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


def _motion(speed):
    if speed is None:
        return 'UNKNOWN'
    return 'STATIONARY' if speed < STILL_SPEED else 'ROLLING' if speed < TRAVEL_SPEED else 'FAST'


def finalize(frames, motions, W, H, fps, to_pitch=None):
    """Decode the final trajectory and rewrite frames[i]['ball'] (the first pass is kept as 'firstPass').
    motions: per frame (M pixels previous->this or None, reliable, cut). to_pitch(i, (x, y) px) ->
    normalized pitch position or None. Every frame says where its position came from (`origin`:
    OBSERVATION, WEAK_DETECTION, INTERPOLATION) and how the ball moved (`motion`: STATIONARY, ROLLING,
    FAST, OCCLUDED). Trajectories that started from nothing and never moved are reviewed and dropped
    (BALL TRAJECTORY REVIEW), and the path is decoded again without them. Returns a report: shares,
    jumps before and after, dropped first-pass runs and their explanations, the review."""
    n = len(frames)
    before = shown_jumps(frames, W, H)
    first = [dict(f['ball']) for f in frames]
    layers = [nodes_of(i, f, W, H, to_pitch) for i, f in enumerate(frames)]
    _attachment(layers, fps)
    times = [f['time'] for f in frames]
    T, _, _ = _chain(motions)
    _velocities(layers, times, T)
    marking_nodes = sum(1 for l in layers for m in l if m.marking)+_standing_markings(layers, times, T)
    reviews, events = [], []
    for _ in range(REVIEW_ROUNDS):
        path = decode(layers, times, motions, fps)
        runs = [r for r in _runs(path) if len(r[2]) >= MIN_RUN]
        static = _review(path, runs, times, T, layers, fps)
        if not static:
            break
        for s in static:
            t0, t1 = times[s['first']], times[s['last']]
            prev = s['previous']
            island = s.get('kind') == 'island'
            lines = ['BALL TRAJECTORY REVIEW', f"Frames: {t0:.2f}-{t1:.2f} s ({len(s['sightings'])} sightings, {s['strong']} confirmed)",
                     (f"Current mode: a run of the path at x = {s['x']:.0f}, y = {s['y']:.0f} px that began {s['entered']}" if island else
                      f"Current mode: a still stretch of the path at x = {s['x']:.0f}, y = {s['y']:.0f} px, entered {s['entered']}"),
                     ('Last confirmed observation before it: ' + (f"{prev['time']:.2f} s at x = {prev['x']:.0f}, y = {prev['y']:.0f} px, {prev['metres']:.1f} m away, {prev['seconds']:.2f} s earlier" if prev else 'none')),
                     f"Visual evidence: detector {s['det']:.2f}, ball-likeness {s['q']:.2f}", f"Motion over its sightings: {s['moved']:.2f} m (camera motion compensated: YES)",
                     f"Causal pass: ruled out {s['elsewhere']} of {len(s['sightings'])} sightings ({', '.join(f'{k} {v}' for k, v in s['reasons'].items())})",
                     ('Decision: REJECTED_IMPOSSIBLE_JUMP (no ball could have come from the run before it nor reached the run after it: a thing seen while the ball was elsewhere; left NOT VISIBLE)' if island else
                      'Decision: REJECTED_STATIC_UNANCHORED (a white thing that never moved, that nothing rolled to and that was seen while the ball was known to be elsewhere; left NOT VISIBLE)')]
            events.append({'time': round(t0, 3), 'kind': 'ball', 'track': None, 'playerId': None, 'message': '\n'.join(lines)})
            reviews.append({'start': round(t0, 3), 'end': round(t1, 3), 'kind': s.get('kind', 'still'), 'sightings': len(s['sightings']), 'strong': s['strong'], 'moved': round(s['moved'], 2),
                            'det': round(s['det'], 3), 'q': round(s['q'], 3), 'reasons': s['reasons'], 'entered': s['entered'],
                            'previousMetres': None if prev is None else round(prev['metres'], 1)})
            used = {id(path[i][0]) for i in s['sightings']}
            for i in range(s['first'], s['last']+1):
                layers[i] = [m for m in layers[i] if id(m) not in used and not (m.tracklet is not None and m.tracklet in s['tracklets'])]
            if s['marking']:
                # A marking cut once is a marking on every frame: its other sightings would only serve as the next
                # stepping stone across the same gap.
                for i in range(n):
                    layers[i] = [m for m in layers[i] if not (m.marking and m.tracklet is not None and m.tracklet in s['tracklets'])]
    kept = set()
    for first_i, last_i, sightings in runs:
        kept.update(range(first_i, last_i+1))
    final = [None]*n
    long_unseen = still_links = 0
    for first_i, last_i, sightings in runs:
        qs = [path[i][0].q for i in sightings]
        quality = sum(qs)/len(qs)
        support = clamp(len(sightings)/(fps*.6))
        identity = round(min(.99, quality*(.5+.5*support)), 3)
        speeds = [path[i][1]['speed'] if path[i][1] is not None else None for i in sightings]
        for k, i in enumerate(sightings):
            node, lk = path[i]
            traj = 1.0 if lk is None else round(math.exp(-lk['cost']), 3)
            speed = speeds[k] if speeds[k] is not None else (speeds[k+1] if k+1 < len(speeds) else None)
            final[i] = {'state': 'TRACKED', 'phase': 'PATH', 'observed': True, 'confidence': identity, 'identity': identity,
                        'visual': round(node.q, 3), 'trajectory': traj, 'track': 1, 'source': 'PATH',
                        'origin': 'OBSERVATION' if _strong(node) else 'WEAK_DETECTION', 'motion': _motion(speed),
                        'box': node.source['box'], 'center': [round(node.x/W, 5), round(node.y/H, 5)], 'speed': None if speed is None else round(speed, 2)}
            if node.pid is not None:
                final[i]['nearTrack'] = node.pid
            # Frames bridged inside the link that led here: the ball occluded on its trajectory.
            if lk is not None and lk['skipped']:
                h = lk['from']
                a, b = lk['node'], node
                long_unseen += lk['speed'] >= TRAVEL_SPEED
                still_links += lk['skipped'] > int(round(MAX_SKIP_S*fps))
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
                                'visual': None, 'trajectory': traj, 'track': 1, 'source': 'PATH', 'bridged': True, 'origin': 'INTERPOLATION', 'motion': 'OCCLUDED',
                                'box': [round((x-d/2)/W, 5), round((y-d/2)/H, 5), round(d/W, 5), round(d/H, 5)],
                                'center': [round(x/W, 5), round(y/H, 5)], 'missingFor': round(times[m]-times[h], 2), 'speed': round(lk['speed'], 2)}
                    if first[m].get('nearTrack') is not None:
                        final[m]['nearTrack'] = first[m]['nearTrack']
    dropped = []
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
        out = final[i] if final[i] is not None else {'state': 'UNKNOWN', 'phase': 'PATH', 'confidence': 0.0, 'identity': 0.0, 'source': 'PATH', 'origin': 'UNKNOWN', 'motion': 'UNKNOWN'}
        if out['state'] != 'UNKNOWN' and to_pitch is not None:
            p = to_pitch(i, (out['center'][0]*W, (out['center'][1]+out['box'][3]/2)*H))
            if p is not None:
                out['pitch'] = [round(float(p[0]), 4), round(float(p[1]), 4)]
        out['firstPass'] = {'state': first[i]['state'], 'phase': first[i].get('phase'), 'confidence': first[i].get('confidence', 0.0), 'track': first[i].get('track'),
                            **({'center': first[i]['center']} if first[i].get('center') else {}),
                            **({'motion': first[i]['motion']} if first[i].get('motion') else {})}
        f['ball'] = out
    after = shown_jumps(frames, W, H)
    states = lambda rows: {k: round(sum(1 for b in rows if b['state'] == k)/max(1, n), 3) for k in ('TRACKED', 'MISSING', 'UNKNOWN')}
    longest_unknown = 0
    run = 0
    for f in frames:
        run = run+1 if f['ball']['state'] == 'UNKNOWN' else 0
        longest_unknown = max(longest_unknown, run)
    events.sort(key=lambda e: e['time'])
    return {'firstPass': {'states': states(first), **before}, 'final': {'states': states([f['ball'] for f in frames]), **after, 'runs': len(runs),
                                                                        'longestUnknownSeconds': round(longest_unknown/fps, 2),
                                                                        'staticRunsDropped': len(reviews), 'longUnseenLinks': long_unseen, 'stationaryLinks': still_links,
                                                                        'markingNodes': marking_nodes},
            'dropped': dropped, 'review': reviews, 'events': events}
