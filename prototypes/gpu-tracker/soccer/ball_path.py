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

Small-scene runs (amateur footage; never broadcast) add four rules. A run is small-scene when at least SMALL_RUN of its
frames carry `profile.smallScene` (plan interface 6, the rule the other offline passes use too); each rule then applies
only to the frames that carry it. A classical near-feet proposal of the causal pass (source 'proposal') never earns the
bonus of a causal acceptance; a causal lock on a still thing is audited (_audit_locks): when the causal pass stayed
locked on something that stood still with nobody near it and that nothing was seen rolling to, while a strong, free,
moving competitor was seen at the same time, the decoder stops deferring to that lock; things nobody is near count as
standing still within the drift of the camera compensation (SMALL_DRIFT); and the review rules out a stretch that stood
still for seconds, unattended, that the causal pass never took for the ball and that nothing the causal pass took for
the ball left (a ball placed for a set piece is kicked away: it stays). A fifth rule, the fixed places (_fixed_places), runs
on every frame of a small-scene run, whether the frame carries the flag or not: a place is a property of the scene, seen
across frames (amateur 3's penalty-arc tip stood 118.2-132.5 s, before the run's first small-scene frame at 143.7 s).
"""
import bisect
import itertools
import math
from collections import Counter
from dataclasses import dataclass

import numpy as np

from .field import DEFAULT_LENGTH, DEFAULT_WIDTH
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
CAUSAL_BONUS = 1.15  # ball-likeness factor of a sighting the causal pass took (with its full-resolution look), unless it was a proposal
# Small-scene frames only (frame['profile']['smallScene']): the still-lock audit.
SMALL_RUN = .5       # share of a run's frames that must be small-scene for any of these rules to run (plan interface 6: a partly small
                     # run, e.g. a broadcast-like segment with a spell of far play, decodes like any other)
SMALL_DRIFT = .3     # m/s: the offline camera compensation of such footage drifts on still things (amateur 3: a speck 2.2 m in 7.5 s,
                     # another .27 m in 3.5 s); a thing nobody was near is still for _standing_markings within this much more per second
AUDIT_FREE = .25     # at most this share of a lock's (or a competitor's) sightings may have a player within reach (on a lower body)
AUDIT_DET = .5       # a competitor's mean detector score ...
AUDIT_SPEED = 2.0    # ... and the speed (m/s, camera-compensated, first to last sighting) from which it is a moving ball; amateur 3
                     # 149.2-151.2 s: the real ball 6.7 m/s at detector .74, the locked speck .3 m/s (2.2 m in 7.5 s)
DEPART_S = 1.0       # an unattended still stretch the path leaves, within this many seconds, into a sighting the causal pass took for
                     # the ball was a ball kicked from rest (a goal kick, a free kick with the taker stepped back): the causal pass
                     # acquires a moving ball after a few consistent frames, a second covers a frame or two behind the kicker's leg
# Fixed places (small-scene runs): a place the scene kept while the ball was somewhere else.
FIXED_AWAY = 3.0     # metres (twice a player's reach): a causal acceptance this far from a place is the ball somewhere else
PLACE_GAP_S = 6.0    # seconds a place may go unseen (the camera panned away, a player stood on it) and still be the same place
# Spot locks (every run): a causal lock that is a painted spot.
SPOT_SHARE = .9      # share of a tracklet's causal acceptances on one painted spot (registered pitch) that makes it a spot lock
SPOT_LOCK_S = 2.0    # a stay on the spot this long or longer is judged
DEPART_MIN_M = 2.0   # a departure ends at least this far from the spot (the spot itself, re-detected, never left) ...
SPOT_STAY_M = 2.5    # during a stay judged to be the spot, any sighting this close to it is the same paint
DEPART_MAX_M = 15.0  # ... and no farther than this: a ball kicked off a spot (a penalty's goal is 11 m away, a kickoff pass
                     # is short) is not first seen 25 m away within a second (CAN-ISL: the real ball elsewhere, 29 m/s)


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
    return p is not None and any(math.hypot((p[0]-x)*DEFAULT_LENGTH, (p[1]-y)*DEFAULT_WIDTH) <= SPOT_RADIUS for x, y in PAINTED_SPOTS)


def quality(c, elsewhere=True, bonus=True):
    """A recorded candidate's ball-likeness from its evidence: appearance, paint, attachment, stillness, having been seen
    while the causal pass confidently tracked the ball elsewhere (`elsewhere`), and the causal pass having taken it
    (`bonus`; never for a classical proposal, which is weak evidence by construction)."""
    s = c.get('scores', {})
    visual = c.get('visual')
    if visual is None:
        det = clamp((c.get('det', 0)-.08)/.45)
        visual = .4*det+.2*s.get('size', .5)+.15*s.get('shape', .5)+.15*s.get('isolation', .5)+.1*s.get('field', .7)
    q = visual*(1-.5*s.get('line', 0.0))*(1-.7*s.get('attach', 0.0))*(1-.6*s.get('static', 0.0))*(ELSEWHERE if elsewhere and s.get('otherObject') else 1.0)
    if bonus and c.get('status') == 'ball' and c.get('source') != 'proposal':
        q = min(1.0, q*CAUSAL_BONUS)   # the causal pass, with its full-resolution look, took it
    return q


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
        q = quality(c)
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


def _standing_markings(layers, times, T, frames=None):
    """A tracklet whose sightings span STATIC_SECONDS and all lie within STATIC_RADIUS of the first (on the pitch when
    registered, else camera-compensated in the image) is a thing standing still for seconds: a field marking, debris,
    a bag. Its sightings lose ball-likeness like a painted spot, unless the causal pass tracked the ball there (a
    ball at rest before a free kick). On small-scene frames (`frames` given), for a tracklet nobody was within reach
    of, the radius grows by SMALL_DRIFT per second: the camera compensation of such footage drifts on still things.
    Returns the number of sightings discounted."""
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
        drifts = frames is not None and all(small_scene(frames[i]) for i, _ in entries) and \
            sum(1 for _, m in entries if m.near is not None) <= AUDIT_FREE*len(entries)
        still = True
        for i, m in entries[1:]:
            if first.pitch is not None and m.pitch is not None:
                metres = math.hypot((m.pitch[0]-first.pitch[0])*DEFAULT_LENGTH, (m.pitch[1]-first.pitch[1])*DEFAULT_WIDTH)
            else:
                p = T[i0] @ np.linalg.inv(T[i]) @ np.array([m.x, m.y, 1.0])
                metres = math.hypot(p[0]/p[2]-first.x, p[1]/p[2]-first.y)/max((first.pxm+m.pxm)/2, 1e-3)
            if metres > (STATIC_RADIUS+SMALL_DRIFT*(times[i]-times[i0]) if drifts else STATIC_RADIUS):
                still = False
                break
        if not still:
            continue
        for _, m in entries:
            if m.source.get('status') != 'ball' and not m.marking:
                m.q, m.marking = max(1e-3, m.q*MARKING_FACTOR), True
                discounted += 1
    return discounted


def small_scene(frame):
    """The frame's small-scene gate as the causal pass recorded it (absent on broadcast and older results: closed)."""
    return bool((frame.get('profile') or {}).get('smallScene', False))


def _metres_apart(a, i, b, j, T):
    """Metres between node a (frame i) and node b (frame j): on the registered pitch when both have a position, else
    camera-compensated in the image at their mean depth."""
    if a.pitch is not None and b.pitch is not None:
        return math.hypot((a.pitch[0]-b.pitch[0])*DEFAULT_LENGTH, (a.pitch[1]-b.pitch[1])*DEFAULT_WIDTH)
    p = T[i] @ np.linalg.inv(T[j]) @ np.array([b.x, b.y, 1.0])
    return math.hypot(p[0]/p[2]-a.x, p[1]/p[2]-a.y)/max((a.pxm+b.pxm)/2, 1e-3)


def _audit_locks(layers, frames, times, T, fps):
    """Small-scene frames: the decoder must not defer to a causal lock on a still thing (amateur 3, 149.2-151.2 s: the
    causal pass stayed on a 3-4 px speck while the real ball rolled free at detector .5-.95; the lock's acceptance bonus,
    its exemption from the marking discount and the "seen elsewhere" flags it put on the real ball made the decoder
    follow the speck). The causal acceptances of one tracklet are audited when all of these hold:
      - every one of its frames is a small-scene frame, it has MIN_RUN sightings or more;
      - it stood still: camera-compensated, nothing further than STILL_SPEED x its duration from its first sighting;
      - nobody within reach of it (at most AUDIT_FREE of its sightings);
      - nothing was seen rolling into it: no earlier causal acceptance, or the last one (another tracklet) ENTRY_GAP_S or
        more before it, or that far away that it was a flight (TRAVEL_SPEED);
      - a competitor was seen during it: another tracklet with MIN_RUN sightings in its span, mean detector AUDIT_DET or
        more, moving AUDIT_SPEED or faster, and free (at most AUDIT_FREE of its sightings on a player's lower body).
    Then the lock's sightings lose the acceptance bonus and count as a field marking (MARKING_FACTOR, and the review may
    cut them as one), and the competitors' sightings lose the "seen elsewhere" factor. A ball at rest before a set
    piece is not touched: it was seen rolling or carried there, or nothing moving competed with it. Changes the nodes in
    place; returns the audits (for the events and the report)."""
    by_tracklet = {}
    for i, nodes in enumerate(layers):
        for m in nodes:
            if m.tracklet is not None:
                by_tracklet.setdefault(m.tracklet, []).append((i, m))
    accepted = sorted(((i, m) for entries in by_tracklet.values() for i, m in entries if m.source.get('status') == 'ball'), key=lambda e: e[0])
    audits = []
    for k, entries in by_tracklet.items():
        # A tracklet is one object (the causal linker lets it go after half a second unseen): its causal acceptances are one lock.
        lock = [(i, m) for i, m in entries if m.source.get('status') == 'ball']
        if len(lock) < MIN_RUN or not all(small_scene(frames[i]) for i, _ in lock):
            continue
        (i0, first), (i1, _) = lock[0], lock[-1]
        duration = times[i1]-times[i0]
        moved = max(_metres_apart(first, i0, m, i, T) for i, m in lock[1:])
        if duration <= 0 or moved >= STILL_SPEED*duration:
            continue
        if sum(1 for _, m in lock if m.near is not None) > AUDIT_FREE*len(lock):
            continue
        previous = next(((j, m) for j, m in reversed(accepted) if j < i0 and m.tracklet != k), None)
        if previous is not None:
            j, m = previous
            seconds = times[i0]-times[j]
            metres = _metres_apart(first, i0, m, j, T)
            if seconds < ENTRY_GAP_S and metres/max(seconds, 1e-3) < TRAVEL_SPEED:
                continue   # seen rolling (or carried) there
            entered = f'{seconds:.2f} s and {metres:.1f} m after the previous causal sighting (tracklet {m.tracklet})'
        else:
            entered = 'from nothing'
        rivals = []
        for r, others in by_tracklet.items():
            inside = [(i, m) for i, m in others if i0 <= i <= i1 and small_scene(frames[i])]
            if r == k or len(inside) < MIN_RUN:
                continue
            det = sum(m.source.get('det', 0.0) for _, m in inside)/len(inside)
            (ja, a), (jb, b) = inside[0], inside[-1]
            speed = _metres_apart(a, ja, b, jb, T)/max(times[jb]-times[ja], 1e-3)
            lower = sum(1 for _, m in inside if m.pid is not None)
            if det >= AUDIT_DET and speed >= AUDIT_SPEED and lower <= AUDIT_FREE*len(inside):
                rivals.append({'tracklet': r, 'sightings': len(inside), 'det': round(det, 3), 'speed': round(speed, 2)})
        if not rivals:
            continue
        for _, m in lock:
            m.q, m.marking = max(1e-3, quality(m.source, bonus=False)*MARKING_FACTOR), True
        free = {r['tracklet'] for r in rivals}
        for r in free:
            for _, m in by_tracklet[r]:
                if m.source.get('scores', {}).get('otherObject'):
                    m.q = max(1e-3, quality(m.source, elsewhere=False)*(MARKING_FACTOR if m.marking else 1.0))
        audits.append({'start': round(times[i0], 3), 'end': round(times[i1], 3), 'tracklet': k, 'sightings': len(lock), 'moved': round(moved, 2),
                       'x': first.x, 'y': first.y, 'entered': entered, 'competitors': rivals})
    return audits


def _spot_of(p):
    """The index of the painted spot (PAINTED_SPOTS) within SPOT_RADIUS of a pitch position, or None."""
    if p is None:
        return None
    for k, (x, y) in enumerate(PAINTED_SPOTS):
        if math.hypot((p[0]-x)*DEFAULT_LENGTH, (p[1]-y)*DEFAULT_WIDTH) <= SPOT_RADIUS:
            return k
    return None


def _spot_locks(layers, times, T, Tinv):
    """Causal locks that are a painted spot. The causal pass may lock onto a penalty or centre spot (the detector scores
    a spot like a ball) and the painted-spot discount spares what the causal pass tracked (a ball placed for a penalty),
    so such a lock wins the path: CAN-ISL tactical wide 562-578 s, one tracklet on the right penalty spot for 15 s at
    detector .5-.9 while the real ball was hidden in the midfield crowd; it ended with the causal pass jumping 20 m to the
    real ball within a tenth of a second (the possession then read a dead ball for half a minute of play).
    A tracklet whose causal acceptances lie on one painted spot (SPOT_SHARE of them, on the registered pitch) is a stay on
    that spot; stays on the same spot less than PLACE_GAP_S apart are one stay (the spot re-detected as a new tracklet
    after a pan). A stay of SPOT_LOCK_S or longer that nothing was seen arriving at (no causal acceptance of another
    tracklet within ENTRY_GAP_S before it, at least DEPART_MIN_M from the spot, from where a ball could have come at
    JUMP_SPEED: the spot re-detected just before is no arrival) and nothing was seen leaving (no
    causal acceptance of another tracklet within DEPART_S after it, at least DEPART_MIN_M from the spot, reachable at
    JUMP_SPEED) is the spot: its sightings count as a field marking without the acceptance exemption. A real ball on the
    spot is kicked off it (a departure; or the same tracklet follows it off the spot, and then it is no spot lock), so it
    keeps the exemption. Changes the nodes in place; returns the stays judged to be the spot."""
    by_tracklet = {}
    for i, nodes in enumerate(layers):
        for m in nodes:
            if m.tracklet is not None and m.source.get('status') == 'ball':
                by_tracklet.setdefault(m.tracklet, []).append((i, m))
    spot_tracklets = []
    for k, lock in by_tracklet.items():
        spots = Counter(_spot_of(m.pitch) for _, m in lock)
        spot, n = spots.most_common(1)[0]
        if spot is None or n < SPOT_SHARE*len(lock):
            continue
        # Short tracklets join too: the spot re-detected for a frame or three (CAN-ISL: tracklets 359, 370, 396 around the
        # 15 s tracklet 360) belongs to the stay, and its last sighting is where the stay ends.
        spot_tracklets.append({'spot': spot, 'tracklets': [k], 'lock': lock})
    stays = []
    for s in sorted(spot_tracklets, key=lambda s: s['lock'][0][0]):
        last = stays[-1] if stays else None
        if last is not None and last['spot'] == s['spot'] and times[s['lock'][0][0]]-times[last['end']] <= PLACE_GAP_S:
            last['tracklets'] = last['tracklets']+s['tracklets']
            last['lock'] = last['lock']+s['lock']
            last['end'] = max(last['end'], s['lock'][-1][0])
        else:
            stays.append(dict(s) | {'end': s['lock'][-1][0]})
    if not stays:
        return []
    accepted = sorted(((i, m) for entries in by_tracklet.values() for i, m in entries), key=lambda e: e[0])
    out = []
    for stay in stays:
        lock = sorted(stay['lock'], key=lambda e: e[0])
        (i0, first), (i1, final) = lock[0], lock[-1]
        if times[i1]-times[i0] < SPOT_LOCK_S or len(lock) < MIN_RUN:
            continue
        sx, sy = PAINTED_SPOTS[stay['spot']]
        off = lambda m: m.pitch is not None and math.hypot((m.pitch[0]-sx)*DEFAULT_LENGTH, (m.pitch[1]-sy)*DEFAULT_WIDTH) >= DEPART_MIN_M
        if any(off(m) for _, m in lock):
            continue   # the stay's own tracklet left the spot: a ball rolled or kicked away (a kickoff tap), not the paint
        members = set(stay['tracklets'])
        # An arrival comes from somewhere else (DEPART_MIN_M or more away, at a ball's speed): a sighting of the spot
        # itself just before the stay (CAN-ISL: a one-frame tracklet on it) is the spot re-detected, not the ball arriving.
        # Only the nearest sighting off the spot on each side decides: reachable at a ball's speed, the ball came or went;
        # out of reach, the causal pass was elsewhere (CAN-ISL: the real ball 25 m away the moment the stay ended, and still
        # 28 m away a second later, which a later sighting alone would have made look like a kick).
        before, entered = None, False
        for j, m in reversed(accepted):
            if j > i0 or m.tracklet in members:
                continue
            seconds = times[i0]-times[j]
            if seconds >= ENTRY_GAP_S:
                break
            metres = _apart(first, i0, m, j, T, Tinv)
            if metres < DEPART_MIN_M:
                continue      # the spot again
            before, entered = (j, m), metres <= DEPART_MAX_M and metres/max(seconds, 1e-3) < JUMP_SPEED
            break
        departed = False
        for j, m in accepted:
            if j < i1 or m.tracklet in members:
                continue
            seconds = times[j]-times[i1]
            if seconds >= DEPART_S:
                break
            metres = _apart(final, i1, m, j, T, Tinv)
            if metres < DEPART_MIN_M:
                continue      # the spot again
            departed = metres <= DEPART_MAX_M and metres/max(seconds, 1e-3) < JUMP_SPEED
            break
        if entered or departed:
            continue
        # The spot, not the ball: no candidate at all (q 0: finalize drops it, as if the detector had not reported it).
        # Discounted (MARKING_FACTOR, even the 1e-3 floor) the paint still anchored the path: the decoder chained its
        # stationary bridges (MAX_STILL_S each) through a paint sighting every few seconds rather than call the ball
        # unknown, so the path stayed on the spot as a hidden ball (CAN-ISL 562-578 s). Every sighting of that paint goes:
        # the stay's own and the further candidates the causal pass's search crop and the whole-frame pass report of it
        # (their slightly different boxes place it up to ~2 m off the spot on a distant view: CAN-ISL, 3 px of box bottom
        # = 0.9 m), from PLACE_GAP_S before the stay to PLACE_GAP_S after it (a whole-frame sighting of the paint just
        # before the causal pass took it anchored the first bridge).
        gap = int(round(PLACE_GAP_S*len(times)/max(times[-1]-times[0], 1e-6))) if len(times) > 1 else 0
        for i in range(max(0, i0-gap), min(len(layers), i1+gap+1)):
            if abs(times[i]-times[i0]) > PLACE_GAP_S and i < i0 or abs(times[i]-times[i1]) > PLACE_GAP_S and i > i1:
                continue
            for m in layers[i]:
                if m.pitch is not None and math.hypot((m.pitch[0]-sx)*DEFAULT_LENGTH, (m.pitch[1]-sy)*DEFAULT_WIDTH) <= SPOT_STAY_M:
                    m.q, m.marking = 0.0, True
        out.append({'start': round(times[i0], 3), 'end': round(times[i1], 3), 'spot': stay['spot'], 'tracklets': sorted(members),
                    'sightings': len(lock), 'x': first.x, 'y': first.y,
                    'before': None if before is None else round(times[i0]-times[before[0]], 2)})
    return out


def _apart(a, i, b, j, T, Tinv):
    """_metres_apart with the inverse camera transforms computed once."""
    if a.pitch is not None and b.pitch is not None:
        return math.hypot((a.pitch[0]-b.pitch[0])*DEFAULT_LENGTH, (a.pitch[1]-b.pitch[1])*DEFAULT_WIDTH)
    p = T[i] @ (Tinv[j] @ np.array([b.x, b.y, 1.0]))
    return math.hypot(p[0]/p[2]-a.x, p[1]/p[2]-a.y)/max((a.pxm+b.pxm)/2, 1e-3)


def _places(layers, times, T, Tinv):
    """Every sighting, whatever its tracklet or status, assigned to a place: the scene point of the first sighting of the
    place within STATIC_RADIUS of it (on the pitch when both are registered, else camera-compensated in the image), the
    place unseen for no longer than PLACE_GAP_S. A thing that moves leaves a place of its own at every sighting; a thing
    that stands still keeps one place across the tracklets the causal linker cut it into. Returns the places with two
    sightings or more, each [(frame, node), ...] in time order."""
    places, anchors, open_ = [], [], []   # anchors: per place, (point in frame 0's coordinates, pxm, pitch or None)
    for i, nodes in enumerate(layers):
        if not nodes:
            continue
        open_ = [k for k in open_ if times[i]-times[places[k][-1][0]] <= PLACE_GAP_S]
        if open_:
            here = np.array([anchors[k][0] for k in open_]) @ T[i].T
            here = here[:, :2]/here[:, 2:3]
            pxm = np.array([anchors[k][1] for k in open_])
            pitch = np.array([anchors[k][2] if anchors[k][2] is not None else (np.nan, np.nan) for k in open_], np.float64)
        added = []
        for m in nodes:
            if open_:
                d = np.hypot(here[:, 0]-m.x, here[:, 1]-m.y)/np.maximum((pxm+m.pxm)/2, 1e-3)
                if m.pitch is not None:
                    on = np.hypot((pitch[:, 0]-m.pitch[0])*DEFAULT_LENGTH, (pitch[:, 1]-m.pitch[1])*DEFAULT_WIDTH)
                    d = np.where(np.isnan(on), d, on)
                r = int(np.argmin(d))
                if d[r] <= STATIC_RADIUS:
                    places[open_[r]].append((i, m))
                    continue
            places.append([(i, m)])
            anchors.append((Tinv[i] @ np.array([m.x, m.y, 1.0]), m.pxm, m.pitch))
            added.append(len(places)-1)
        open_ += added
    return [p for p in places if len(p) >= 2]


def _fixed_places(layers, times, T):
    """There is one ball, and it does not come back to the exact spot it left by itself. A place where a still thing
    stood for STATIC_SECONDS (MIN_RUN sightings or more), then the causal pass took the ball somewhere else (MIN_RUN
    causal acceptances at least FIXED_AWAY from it), and then a still thing stood again for STATIC_SECONDS, holds a fixed
    object: a painted mark, debris, a spare ball. Nobody may stand at it (at most AUDIT_FREE of its sightings within a
    player's reach): a ball at a still player's feet is passed away and played back. The registration need not be
    trusted and the mark need not be a known spot (amateur 3, 118.2-132.8 s: the tip of the penalty arc, locked by the
    causal pass from the first frame of the run while the ball was being thrown in, then seen for seven more seconds
    after the causal pass followed the ball at the far goal). Small-scene runs only: on the broadcast 42-222 s window the
    rule also took the path off a spare ball and a painted dash beside the touchline (93.7-96.2 s, 100.9-103.7 s) but,
    with those stepping stones gone, bridged two other spells through different wrong places.
    The place's sightings are split into stays at those acceptances. A stay the ball was seen coming into or going out
    of was the ball on that spot (a kick-off, a corner, a ball stopping on a mark) and is left alone: the nearest causal
    acceptance elsewhere before (after) it within DEPART_S, at a speed a ball can have (JUMP_SPEED), its tracklet seen
    in between only nearer to the place and not turning back beyond it. Every sighting of the other stays counts as a
    field marking (MARKING_FACTOR), the causal acceptances among them included, which lose the acceptance bonus, except
    the ball rolling over the place: a sighting whose own tracklet is seen across it (2 STATIC_RADIUS from where the
    place was first seen) within NEIGHBOUR_S. Changes the nodes in place; returns the fixed places (for the events and
    the report)."""
    Tinv = [np.linalg.inv(t) for t in T]
    taken = [(i, m) for i, nodes in enumerate(layers) for m in nodes if _taken(m)]
    taken_times = np.array([times[i] for i, _ in taken])
    by_tracklet = {}
    for i, nodes in enumerate(layers):
        for m in nodes:
            if m.tracklet is not None:
                by_tracklet.setdefault(m.tracklet, []).append((i, m))
    found = []
    for sightings in _places(layers, times, T, Tinv):
        if len(sightings) < 2*MIN_RUN or sum(1 for _, m in sightings if m.near is not None) > AUDIT_FREE*len(sightings):
            continue
        (i0, a), (i1, _) = sightings[0], sightings[-1]
        # Stood still for STATIC_SECONDS before the acceptances elsewhere and STATIC_SECONDS after them.
        stood = [times[i] for i, _ in sightings]
        k_lo = max(MIN_RUN-1, bisect.bisect_left(stood, stood[0]+STATIC_SECONDS))
        k_hi = min(len(stood)-MIN_RUN, bisect.bisect_right(stood, stood[-1]-STATIC_SECONDS)-1)
        if k_lo >= len(stood) or k_hi < 0 or stood[k_lo] >= stood[k_hi]:
            continue
        members = {id(m) for _, m in sightings}
        lo, hi = int(np.searchsorted(taken_times, stood[k_lo], 'right')), int(np.searchsorted(taken_times, stood[k_hi], 'left'))
        away = [(j, e) for j, e in taken[lo:hi] if id(e) not in members and _apart(a, i0, e, j, T, Tinv) >= FIXED_AWAY]
        if len(away) < MIN_RUN:
            continue
        away_times = [times[j] for j, _ in away]
        stays = [[sightings[0]]]
        for (h, _), (i, m) in zip(sightings, sightings[1:]):
            if bisect.bisect_right(away_times, times[h]) < bisect.bisect_left(away_times, times[i]):
                stays.append([])   # the causal pass took the ball elsewhere in between
            stays[-1].append((i, m))

        def seen_moving(stay, step):
            """The ball seen coming into the stay (step -1) or going out of it (+1): the nearest causal acceptance
            elsewhere on that side within DEPART_S, reachable at JUMP_SPEED, its tracklet's sightings between it and the
            stay no further from the place, and its next one beyond it (within NEIGHBOUR_S) no nearer (by more than a
            ball's diameter: a ball pausing at a foot)."""
            i, m = stay[0] if step < 0 else stay[-1]
            k = int(np.searchsorted(taken_times, times[i], 'left'))-1 if step < 0 else int(np.searchsorted(taken_times, times[i], 'right'))
            while 0 <= k < len(taken) and id(taken[k][1]) in members:
                k += step
            if not 0 <= k < len(taken):
                return False
            j, e = taken[k]
            seconds = abs(times[j]-times[i])
            metres = _apart(m, i, e, j, T, Tinv)
            if seconds > DEPART_S or metres/max(seconds, 1e-3) > JUMP_SPEED:
                return False
            own = [(f, n) for f, n in by_tracklet.get(e.tracklet, []) if id(n) not in members]
            between = [(f, n) for f, n in own if min(i, j) < f < max(i, j)]
            beyond = [(f, n) for f, n in own if (f-j)*step > 0 and abs(times[f]-times[j]) <= NEIGHBOUR_S]
            beyond = beyond[:1] if step > 0 else beyond[-1:]
            return all(_apart(m, i, n, f, T, Tinv) <= metres+BALL_DIAMETER for f, n in between) and \
                all(_apart(m, i, n, f, T, Tinv) >= metres-BALL_DIAMETER for f, n in beyond)

        def passing(i, m):
            """The ball rolling over the place: its own tracklet seen across it, 2 STATIC_RADIUS or more from where the
            place was first seen, within NEIGHBOUR_S of this sighting (a still thing's sightings only jitter about it)."""
            entries = by_tracklet.get(m.tracklet, [])
            k = bisect.bisect_left(entries, i, key=lambda e: e[0])
            near = itertools.chain(itertools.takewhile(lambda e: times[i]-times[e[0]] <= NEIGHBOUR_S, reversed(entries[:k])),
                                   itertools.takewhile(lambda e: times[e[0]]-times[i] <= NEIGHBOUR_S, entries[k:]))
            return any(id(n) not in members and _apart(a, i0, n, f, T, Tinv) >= 2*STATIC_RADIUS for f, n in near)
        demoted = withdrawn = kept = 0
        for stay in stays:
            if seen_moving(stay, -1) or seen_moving(stay, 1):
                kept += 1
                continue
            for i, m in stay:
                if m.marking or passing(i, m):
                    continue
                if m.source.get('status') == 'ball':
                    m.q = max(1e-3, quality(m.source, bonus=False)*MARKING_FACTOR)
                    withdrawn += 1
                else:
                    m.q = max(1e-3, m.q*MARKING_FACTOR)
                m.marking = True
                demoted += 1
        if demoted:
            j, e = away[0]
            found.append({'start': round(times[i0], 3), 'end': round(times[i1], 3), 'sightings': len(sightings), 'x': a.x, 'y': a.y,
                          'tracklets': sorted({m.tracklet for _, m in sightings if m.tracklet is not None}, key=str),
                          'stays': len(stays), 'keptStays': kept, 'demoted': demoted, 'withdrawn': withdrawn, 'away': len(away),
                          'awayFirst': {'time': round(times[j], 3), 'x': e.x, 'y': e.y, 'metres': round(_apart(a, i0, e, j, T, Tinv), 1)}})
    return found


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


def _layer_arrays(layers):
    """What decode compares between nodes, as one array per layer: positions, pixels per metre, the player
    ids (`pid`, `near`) and tracklet ids as integer codes (-1 for None; two objects get one code exactly when
    they are equal, so `code_a == code_b` is the original `a == b`), and 'standing still with nobody near'."""
    codes = {}

    def code(value):
        if value is None:
            return -1
        return codes.setdefault(value, len(codes))
    out = []
    for nodes in layers:
        out.append({'xy': np.array([(m.x, m.y) for m in nodes]).reshape(-1, 2), 'pxm': np.array([m.pxm for m in nodes]),
                    'pid': np.array([code(m.pid) for m in nodes], int), 'near': np.array([code(m.near) for m in nodes], int),
                    'tracklet': np.array([code(m.tracklet) for m in nodes], int),
                    'still': np.array([m.own_speed is not None and m.own_speed < STILL_SPEED and m.near is None for m in nodes], bool)})
    return out


def decode(layers, times, motions, fps):
    """Viterbi over the layers. Returns per frame either None (not visible) or (node, link) where link
    is {'from': frame index, 'node': Node, 'cost': motion cost, 'speed': m/s, 'skipped': bridged frames}
    or None for a start. A link may bridge up to MAX_SKIP_S of unseen frames, or up to MAX_STILL_S when
    the speed it implies says the ball stayed where it was (STILL_SPEED). A flight into or out of a ball
    standing still with nobody near it (DEAD_STOP), or faster than a shot between two different things
    (JUMP_COST), is paid for: the ball does neither by itself.

    The links into a frame from every earlier frame it may be reached from are costed in one go: the
    source frames' nodes are stacked into rows (nearest frame first, each frame's nodes in order), every
    term is the same elementwise arithmetic as costing one source frame at a time (the per-frame
    quantities become per-row columns; the camera transform of each source frame is still applied to that
    frame's nodes alone, so the BLAS call sees the same shapes), and the first row reaching the minimum of
    a column is the first frame and node the one-at-a-time loop would have accepted."""
    n = len(layers)
    T, breaks, unreliable = _chain(motions)
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
    arrays = _layer_arrays(layers)
    sizes = [len(l) for l in layers]
    reached = [False]*n           # a frame has a node a path reached (a finite cost) once its costs are final
    # The last cut at or before each frame: links never reach back across it.
    fence = np.maximum.accumulate(np.where(np.asarray(breaks, bool), np.arange(n), 0)) if n else np.zeros(0, int)
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
        here = arrays[i]
        xy, pxm = here['xy'], here['pxm']
        start = START_COST if i == 0 else best_null[i-1]+START_COST
        for j, m in enumerate(layers[i]):
            best[i][j], back[i][j] = start+(FOOT_START if m.pid is not None else 0.0), ('null', i-1)
        # The frames a link may come from: back to K_still frames, never across a cut, with nodes that were reached.
        sources = [(i-h, h) for h in range(i-1, max(0, i-K_still, int(fence[i]))-1, -1) if reached[h]]
        if sources:
            Ti = T[i].T
            counts = np.array([sizes[h] for _, h in sources])
            rows = np.repeat(np.arange(len(sources)), counts)                      # the source frame of every row
            ks = np.repeat(np.array([k for k, _ in sources]), counts)
            dts = [max(1e-3, times[i]-times[h]) for _, h in sources]
            dt_col = np.repeat(np.array(dts), counts)[:, None]
            prev = np.concatenate([origin[h] @ Ti for _, h in sources])
            prev = prev[:, :2]/prev[:, 2:3]
            src = [arrays[h] for _, h in sources]
            dist = np.hypot(prev[:, None, 0]-xy[None, :, 0], prev[:, None, 1]-xy[None, :, 1])
            scale = (np.concatenate([s['pxm'] for s in src])[:, None]+pxm[None, :])/2
            speed = dist/np.maximum(scale, 1e-3)/dt_col
            # The displacement an unseen gap must explain grows dearer with the gap: a long flight that
            # nobody saw needs more than a plausible average speed.
            growth = np.repeat(np.array([1+dt for dt in dts]), counts)[:, None]
            skips = np.repeat(np.array([(k-1)*SKIP_COST for k, _ in sources]), counts)[:, None]
            shake = np.repeat(np.array([UNRELIABLE*(shaky[i]-shaky[h]) for _, h in sources]), counts)[:, None]
            motion = (speed/V0)**2*growth+skips+shake
            cost = motion.copy()
            tracklet = np.concatenate([s['tracklet'] for s in src])
            same = ((ks == 1) & (tracklet >= 0))[:, None] & (tracklet[:, None] == here['tracklet'][None, :])
            cost = np.where(same, np.minimum(cost, .05+UNRELIABLE*unreliable[i]), cost)
            # Arriving at a player's lower body: free when the causal pass saw it arrive or the ball was
            # already with that player; dear when it comes from another player's feet or reach with
            # nothing seen travelling (an unseen pass stays NOT VISIBLE; a boot never becomes the ball).
            near = np.concatenate([s['near'] for s in src])
            enter = ((here['pid'] >= 0)[None, :] & (near[:, None] != here['pid'][None, :])) & ~same
            from_feet = (near >= 0)[:, None]
            switch = np.repeat(np.array([FOOT_SWITCH*max(0.0, 1-dt/FOOT_FADE) for dt in dts]), counts)[:, None]
            cost = cost+enter*FOOT_ENTER+(enter & from_feet)*switch
            # A flight cannot leave a ball standing still, nor end on one, with no player within reach of it: the
            # ball does not start from rest or stop dead on the grass by itself. (Not judged between consecutive
            # frames whose candidates all continue each other's tracklets.)
            judged = ks > 1
            if sources[0][0] == 1 and not same[rows == 0].all():
                judged = judged | (rows == 0)
            still_a = np.concatenate([s['still'] for s in src])[:, None]
            still_b = here['still'][None, :]
            flight = speed >= TRAVEL_SPEED
            cost = cost+np.where(judged[:, None] & flight & (still_a | still_b) & ~same, DEAD_STOP, 0.0)
            # Faster than a hard shot between sightings of two different things: not the same ball.
            cost = cost+np.where(judged[:, None] & (speed > JUMP_SPEED) & ~same, JUMP_COST, 0.0)
            # Beyond MAX_SKIP_S only a ball that stayed where it was may be bridged.
            cost = np.where((ks > K)[:, None] & ~(speed <= STILL_SPEED), np.inf, cost)
            total = np.concatenate([best[h] for _, h in sources])[:, None]+cost
            bad = np.isnan(total)
            if bad.any():
                # A frame with an undefined link to a candidate offers it nothing, whatever its other nodes say.
                undefined = np.zeros((len(sources), len(layers[i])), bool)
                np.logical_or.at(undefined, rows, bad)
                total = np.where(undefined[rows], np.inf, total)
            a_best = np.argmin(total, axis=0)
            t_best = total[a_best, np.arange(len(layers[i]))]
            offsets = np.cumsum(counts)-counts
            for b in np.flatnonzero(t_best < best[i]):
                r = int(a_best[b])
                k, h = sources[rows[r]]
                a = r-int(offsets[rows[r]])
                best[i][b], back[i][b] = float(t_best[b]), ('node', h, a)
                link[i][b] = {'from': h, 'node': layers[h][a], 'cost': float(motion[r, b] if k == 1 else cost[r, b]), 'speed': float(speed[r, b]), 'skipped': k-1}
        for j, m in enumerate(layers[i]):
            best[i][j] += -math.log(m.q)
        reached[i] = bool(np.isfinite(best[i]).any())
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


def _taken(node):
    """The causal pass took this sighting for the ball on a detector's evidence (a classical proposal is weak evidence)."""
    return node.source.get('status') == 'ball' and node.source.get('source') != 'proposal'


def _review(path, runs, times, T, layers, fps, small=None):
    """Still stretches on the path (every sighting within STILL_REGION of the stretch's first, at least MIN_RUN
    of them) whose sightings the causal pass rejected (it followed the ball somewhere else, or knew the thing
    for a spot, paint or a boot), and that nothing was seen rolling to: the stretch starts a trajectory, or the
    link into it is a flight (TRAVEL_SPEED or more) or bridged ENTRY_GAP_S unseen. Such a stretch is a white thing standing where the ball was known
    not to be, not the ball. The ball at rest stays on the path when it rolled there, or when the causal pass
    had nothing against it (a ball waiting for a kick-off). layers and fps are kept for the callers' sake.
    small: per frame, the small-scene gate (None: closed everywhere). On small-scene frames the stillness radius grows
    by SMALL_DRIFT per second, and a stretch that stood still for STATIC_SECONDS with nobody within reach and that the
    causal pass never took for the ball is ruled out as a whole (amateur 3, 172-181 s: a speck the causal pass
    rejected only as off its trajectory, chained by stationary links for nine seconds), unless the path leaves it
    within DEPART_S into a sighting the causal pass took for the ball: the exit counterpart of the entry rule. The
    causal pass never takes a ball at rest (it acquires one once it moves), so a ball placed for a set piece, the taker
    stepped back, is unattended and never taken until it is kicked away; a speck is never kicked.
    Returns the stretches to cut out with what was seen."""
    out = []
    for first, last, sightings in runs:
        k = 0
        while k < len(sightings):
            stretch = [sightings[k]]
            while k+len(stretch) < len(sightings):
                i = sightings[k+len(stretch)]
                drifts = small is not None and small[stretch[0]] and small[i]
                region = STILL_REGION+SMALL_DRIFT*(times[i]-times[stretch[0]]) if drifts else STILL_REGION
                if _metres_between(path, stretch[0], i, T) >= region:
                    break
                stretch.append(i)
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
            unattended = small is not None and all(small[i] for i in stretch) and duration >= STATIC_SECONDS and \
                not any(n.source.get('status') == 'ball' for n in nodes) and sum(1 for n in nodes if n.near is not None) <= AUDIT_FREE*len(nodes)
            if unattended:
                # Kicked away: the path goes on from the stretch (sightings[k:], the rest of this run) into a causal acceptance.
                after = itertools.takewhile(lambda i: times[i]-times[stretch[-1]] <= DEPART_S, sightings[k:])
                unattended = not any(_taken(path[i][0]) for i in after)
            if elsewhere < REJECTED_SHARE*len(nodes) and not unattended:
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
                        'x': nodes[0].x, 'y': nodes[0].y, 'previous': gap, 'kind': 'still', 'unattended': unattended,
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
    times = [f['time'] for f in frames]
    T, _, _ = _chain(motions)
    flags = [small_scene(f) for f in frames]
    small = n > 0 and sum(flags) >= SMALL_RUN*n   # a small-scene run (frame['profile']['smallScene']); the rules are off on every other run
    audits = _audit_locks(layers, frames, times, T, fps) if small else []
    fixed = _fixed_places(layers, times, T) if small else []
    spots = _spot_locks(layers, times, T, [np.linalg.inv(t) for t in T]) if any(m.pitch is not None for l in layers for m in l) else []
    if spots:
        layers = [[m for m in l if m.q > 0.0] for l in layers]   # the painted spots of the spot locks are no candidates
    _attachment(layers, fps)
    _velocities(layers, times, T)
    marking_nodes = sum(1 for l in layers for m in l if m.marking)+_standing_markings(layers, times, T, frames if small else None)
    reviews, events = [], []
    for p in fixed:
        if not p['withdrawn']:
            continue   # only a decision against the causal pass is logged; the other places are counted in the report
        away = p['awayFirst']
        events.append({'time': p['start'], 'kind': 'ball', 'track': None, 'playerId': None, 'message': '\n'.join([
            'BALL FIXED OBJECT', f"Frames: {p['start']:.2f}-{p['end']:.2f} s ({p['sightings']} sightings of a still thing at x = {p['x']:.0f}, y = {p['y']:.0f} px, "
                                 f"tracklets {', '.join(str(t) for t in p['tracklets'])})",
            f"Seen there again after the causal pass took the ball elsewhere: {p['away']} causal sightings at least {FIXED_AWAY:.0f} m away in between "
            f"(the first at {away['time']:.2f} s, x = {away['x']:.0f}, y = {away['y']:.0f} px, {away['metres']:.1f} m away; camera motion compensated: YES)",
            f"Stays: {p['stays']}, {p['keptStays']} seen arriving or leaving (the ball on that spot: kept)",
            f"Decision: {p['demoted']} sightings count as a field marking; {p['withdrawn']} causal acceptances among them lose the acceptance bonus"])})
    for s in spots:
        events.append({'time': s['start'], 'kind': 'ball', 'track': None, 'playerId': None, 'message': '\n'.join([
            'BALL SPOT LOCK', f"Frames: {s['start']:.2f}-{s['end']:.2f} s ({s['sightings']} causal sightings of tracklets "
                              f"{', '.join(str(t) for t in s['tracklets'])} on a painted spot of the registered pitch, at x = {s['x']:.0f}, y = {s['y']:.0f} px)",
            'Nothing was seen arriving at it (rolling or carried) and nothing was seen leaving it (kicked away): the spot itself',
            'Decision: these sightings are the painted spot, never the ball'])})
    for a in audits:
        rivals = '; '.join(f"tracklet {r['tracklet']} ({r['sightings']} sightings, detector {r['det']:.2f}, {r['speed']:.1f} m/s)" for r in a['competitors'])
        events.append({'time': a['start'], 'kind': 'ball', 'track': None, 'playerId': None, 'message': '\n'.join([
            'BALL LOCK AUDIT', f"Frames: {a['start']:.2f}-{a['end']:.2f} s ({a['sightings']} causal sightings of tracklet {a['tracklet']} at x = {a['x']:.0f}, y = {a['y']:.0f} px)",
            f"Motion over them: {a['moved']:.2f} m (camera motion compensated: YES), nobody within reach; entered {a['entered']}",
            f'Seen at the same time, free and moving: {rivals}',
            'Decision: the lock counts as a field marking (no acceptance bonus); the competitors lose the "seen while the ball was elsewhere" discount'])})
    for _ in range(REVIEW_ROUNDS):
        path = decode(layers, times, motions, fps)
        runs = [r for r in _runs(path) if len(r[2]) >= MIN_RUN]
        static = _review(path, runs, times, T, layers, fps, flags if small else None)
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
                     f"Causal pass: ruled out {s['elsewhere']} of {len(s['sightings'])} sightings ({', '.join(f'{k} {v}' for k, v in s['reasons'].items())})"
                     + (', never took it for the ball, nobody within reach, never kicked away (small scene)' if s.get('unattended') else ''),
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
                                                                        'markingNodes': marking_nodes, 'lockAudits': len(audits), 'fixedPlaces': len(fixed),
                                                                        'spotLocks': len(spots)},
            'dropped': dropped, 'review': reviews, 'audits': audits, 'fixed': fixed, 'events': events}
