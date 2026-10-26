"""The suggested-zone engine of coaching v2 (DESIGN v2 sections 5.1-5.3; spec Part 8): for one criticised moment, an
area ONE team-mate could physically have reached that would have restored the relation the detector found broken,
without opening another gap; or, when no such area exists, the reason none is drawn.

Why an engine instead of "the player's formation spot". The v1 layer circled a player and had a dormant 'suggested'
circle it never filled (0 of 63 evidence records); pulling a player towards a template anchor mixes both phases and
ignores the ball, the first defender and the opponents (spec Part 7-8). Here a zone is the set of grid points where the
mover, moved there with everyone else held still (a single-mover counterfactual), turns the detector's own relation
from failed to held: re-measured with the tactics_geom geometry the detectors use and judged by the same
contextual.sample_outcomes rule with the level's parameters, so a zone is checked with exactly what was measured.

Steps (DESIGN 5.1, per finding; its cited key moments are tried in time order and the first that yields a zone wins):
  1 problem   the moment's sample must fail under the level's parameters in the stored relations AND when re-measured
              on the moment's scene (else the scene does not show it: context_missing). D1 at that sample decides
              between the cover fix and the pressure fix (failure mode no_cover / no_pressure).
  2 movers    at most 3 (5.2): D1 cover: the 2 team-mates nearest the cover sector (DESIGN says "nearest the first
              defender": the same players unless one stands deeper behind him, the natural cover), never the first
              defender; D1 pressure: the 2 players nearest the carrier; D2: the 2 nearest the far-side region; D3:
              the wide defender ("delay the run"), the near centre-back, the holding midfielder (levels with
              useDmRotation), then the team-mate nearest the channel; D4 (and D3 judged as "someone stays back":
              level 1's reframing, level 2's teamLevelOnly rule): the 2 deepest players not yet behind the ball;
              D5: the two defenders beside the gap (the ball-side one first). In our own phase (D3, D4) the
              player on the ball (within ON_BALL_M) is never a mover: moving him would move the ball.
  3 context   every observed player of both teams in the moment's scene, the keepers, the ball (or the carrier), in
              the team's attack frame (attack_side(): the goal defended at that time, EVIDENCE's calibration['side']).
  4 region    the principle region (5.2), never a formation anchor: D1 cover the annular sector 4 m to R behind the
              first defender within the cover angle and goal-side of him; D1 pressure (an extension: DESIGN 5.2 has
              no pressure row, but levels 1-2 judge "goes to the ball") within the pressure distance of the
              carrier and at least 1 m goal-side of him (the zone never covers him); D2 the far lateral half
              goal-side of the ball within 8 m x L/105 of the back line's median depth; D3 the vacated channel; D4
              goal-side of the ball by 10 m x L/105 in the central two-thirds of the width (also D3 judged as
              "someone stays back": at level 2 the channel zone lay 2-8 m ahead of the ball and left the behind
              count at 0, FACTS M6); D5 the gap between its two defenders at the line's depth +-3 m. Every D3
              region also ends LEVEL_M (2 m) upfield of the deepest opponent in the vacated channel (his outer
              third, goal-side of the wide defender, our half; critic B-8: a "cover" upfield of him does not stand
              between him and our goal): nothing reachable there leaves the principle only (no_reachable_fix); a
              channel opponent the detector measured but the scene does not show gives context_missing.
  5 reach     a disc around the mover's position in the moment's 'before' scene (REACH_H, 2 s, earlier), radius =
              horizon x min(his own p90 speed in this run (none: his team's median), the age cap 5 / 6 / 7 m/s); a
              mover not observed then gets no zone (no teleport from an unknown position).
  6 improve   a grid of GRID_M x L/105 (>= 0.5 m) over the reach disc; each region point re-runs the relation with the
              mover there; restored = the rule now demonstrates it (D3 also: the wide defender himself no longer
              stands advanced; D3 at level 2: the rule stops failing with someone now behind the ball; D5: the
              ball-side gap no longer fails: one mover is not asked for the strength band). The re-measured D3
              relations carry the ball depth whenever the detector's do, so a ball-zone gate judges both alike.
  7 no new gap  a point is rejected when, with the mover there, (i) pressure on the carrier disappears, (ii) the
              central protection (our players goal-side of the ball within 10 m x W/68 of the centre line) drops to
              none, (iii) a ball-side back-line gap above 20 m x W/68 opens (the role-free line: players within 8 m x
              L/105 of the deepest, at least 3; D5: its own line), (iv) an opponent in our half who had a team-mate
              within 10 m x L/105 has none, or (v) it is within 3 m of another team-mate (keepers included). Only NEW
              gaps count: what was already open is not the mover's doing.
  8 shape     per mover, the largest connected set (8 neighbours) of accepted points -> its convex hull dilated by
              the mover's position sigma (>= 0.5 m, then half of it, then none) and simplified to <= 12 vertices;
              when no hull verifies, a 12-ray star polygon around the set's most interior point, shrunk until it
              verifies. Verification, on the polygon exactly as stored (normalized coordinates rounded to 1e-4):
              every vertex inside the reach disc and the pitch and never upfield of the D3 channel opponent +
              LEVEL_M or, for a behind-the-ball zone, of the ball (a sigma dilation must not cross either), no
              team-mate inside or within 3 m of it, at least
              MIN_ZONE_POINTS (4) grid points inside, all of them passing the checks of 7, and at least
              ZONE_KEEP_SHARE (.8) of them restoring the relation (DESIGN 13.1's property, checked independently by
              tests/coaching_zone_props.py).
Choice between movers: the least effort (the share of his reach the nearest restoring point needs; 0 = he stood there
REACH_H earlier); ties in the 5.2 order. Movers without a zone are listed in 'rejected' with their reason.
Display (5.3): 'point' (a dashed circle of radius max(sigma, 1 m), stored as the 12-gon around it) only when the level
allows it (zonePoint, level 4+), the restoring set spans at most 3 m, every involved position sigma is known and at
most 1 m, the moment's tier is 'world' and the circle verifies; 'direction' (only an arrow) when the zone covers more
than a quarter of the team's half (L x W / 8); else 'zone'. 'from' is the mover's position at the key frame.

No zone (a NoAlternative; DESIGN 5.1 reasons): guardrail_failed (any failed guardrail, 'alternative' aside: spec Part
15 "no precise suggestion" for a review), scale_unknown (metres not usable), no_zone_for_detector (D5 midfield-to-
defence distance: the gap between two units is not a single-player fix), context_missing (no cited moment shows the
failure on its scene, or the side, the carrier, the ball or the line is not in it), mover_unseen (no candidate seen
REACH_H earlier), no_reachable_fix (no reachable area restores it, or only a sliver that makes no area),
would_open_new_gap (the restoring points would make an area but the checks of 7 remove some of them; since fix round
1 never when no check refused a point: that came back for single rows of points with every check passing, and the
text says "every reachable spot would have opened another space"). The reason of the moment that got furthest is
reported (would_open_new_gap
> no_reachable_fix > mover_unseen > context_missing). D6 (attacking shape) has no zones: propose() returns None and the
finding keeps assess's principle.

Measured (CPU, Oct 22; scratch coaching2/build/zones_w4; INTERFACES.md ZONES wave 4 has the tables):
- Metrica game 1 (90 min, both teams; full tracking / simulated camera), every key moment that fails at level 4 cited
  alone: a zone for D1 56 of 133 / 29 of 74, D2 15 of 39 / 5 of 15, D3 49 of 74 / 10 of 16, D4 13 of 16 / 5 of 8, D5
  7 of 53 / 8 of 25 (41 / 10 are midfield-to-defence gaps: no zone); the others mostly no_reachable_fix (nobody could
  get there in 2 s), would_open_new_gap 0-10 per detector. Zone area p50 14-38 m2 (p90 35-89); 0 of the 505 zones of
  levels 2 / 4 / 7 broke a property (reach, stacking, restoring share, checks, principle region) under the
  independent check; propose 1.8 ms p50, 12.4 ms p99.
- EVAL's synthetic scenes (levels 1-7, every failing key moment and every finding): 1700 zones on moments built from
  the timeline and 1407 on EVIDENCE's own moments, 0 property problems, restoring share on a finer 0.25 m grid >= .81;
  the planted unsupported advances get a channel zone at levels 2-7, an identity crossing or a single instance none
  (guardrail_failed), the stacking scene's zones keep 3 m off every team-mate.
- Random scenes (tests; 6 generators x 40, sigma up to 1.5 m, 55 x 36 m fields too): every zone holds the
  properties; zones on at least half of the scenes of each generator. 36 single-rule mutants of this module: all
  caught by tests/test_coaching_alternatives.py (scratch zones_w4/mutate.py).
- Fix round 1 (Oct 23, critic B-8; scratch coaching2/build/zones_fix1; before = this module as of Oct 22, same
  inputs and the same code of every other module, the new properties checked by tests/coaching_zone_props.py):
  COACH's 330 scene reports: 50 D3 zones before, 50 of 50 failing (48 at levels 3-7 lay 5-11 m upfield of the
  channel opponent; the 2 level-2 ones also ahead of the ball, restoring nothing); after 0 D3 zones (no spot level
  with or goal-side of the planted opponent is reachable in 2 s: 56 principle-only, no_reachable_fix), the 22
  D1 / D2 zones unchanged, 0 property problems. Random D3 scenes: 17 of 38 zones (90 of 192 on another seed)
  upfield of the channel opponent before, after 38 (180) zones and 0 problems. Metrica game 1 (levels 1, 2, 4, 7,
  full / camera, 1297 failing key moments cited alone): 569 -> 562 zones, property problems 12 -> 0; D3 level 4
  49 -> 48 (the one zone with a channel opponent, upfield of him, gone), level 7 1 -> 0, level 2 9 -> 5 (all 9
  before lay in the channel and restored nothing; the 5 after are behind the ball), level 1 16 -> 16; D1 / D2 / D4 /
  D5 zones identical; 37 reasons would_open_new_gap -> no_reachable_fix (no check had refused a point). Cost
  unchanged: p50 2.9 / 3.1 ms, p99 20.0 / 19.8 ms after / before on the same 720 calls under the same load.
  13 single-rule mutants of the new rules: 11 caught; 2 survive as equivalents (level 2's "someone now behind"
  test is implied by its region; the ball limit of a D4 zone is implied by its 10 m margin and the 80% share).

Every constant is a PITCHIQ-PROPOSAL (origin PITCHIQ-PROPOSAL, unvalidated; no federation gives positional numbers);
the ZONE coach labels of DESIGN 13.3 (a coach draws an acceptable area blind) must validate them. Standard library,
numpy and the coaching modules contextual / schema / tactics_geom only; nothing here reads a result.
"""
import math
from collections import deque

import numpy as np

from . import contextual as CX
from . import schema as S
from . import tactics_geom as G

# ---------- parameters (all PITCHIQ-PROPOSAL, unvalidated; DESIGN Appendix A where named) ----------
REACH_H_S = 2.0                 # s: REACH_H_S, the horizon a mover starts from (profile reachHorizonS overrides)
REACH_TOL_S = .3                # s: the moment's 'before' scene must lie this close to the horizon (1-2 samples)
REACH_V_CAP_MS = 7.0            # m/s: the age cap when the profile gives none (DESIGN: U6-U10 5, U11-U14 6, U15+ 7)
SPEED_MIN_MS = .1               # m/s: schema Reach.speedMS floor
GRID_M = 1.0                    # m x L/105: the improvement grid (DESIGN 5.1 step 6) ...
GRID_MIN_M = .5                 # ... never finer than this
STACK_M = 3.0                   # m (body): no zone point this close to another team-mate (check v, STACK_M)
ZONE_KEEP_SHARE = .8            # share of a zone's grid points that must restore the relation (ZONE_KEEP_SHARE)
ZONE_MAX_HALF_SHARE = .25       # a zone above this share of the team's half is shown as a direction only
POINT_MAX_DIAMETER_M = 3.0      # m: a zone this small may be shown as a point (POINT_MAX_DIAMETER_M) ...
POINT_MAX_SIGMA_M = 1.0         # m: ... when every involved position sigma is at most this ...
POINT_MIN_RADIUS_M = 1.0        # m: ... drawn as a dashed circle of radius max(sigma, this)
SIGMA_MIN_M = .5                # m: the zone is dilated by the mover's position sigma, at least this
MIN_ZONE_POINTS = 4             # a zone holds at least this many improvement-grid points (about 4 m2 on 105 m): a
                                # smaller shape is a spot, not an area, and sat on the edge of the restoring set (a
                                # 0.06 m2 star around one boundary point restored nothing on a 0.25 m grid)
MAX_VERTICES = 12               # a zone polygon has at most this many vertices
MAX_MOVERS = 3                  # candidate movers per moment
ON_BALL_M = 1.5                 # m: our player this close to the ball is on it and never a mover (possession-4's r1:
                                # no evidence of being at the ball beyond 1.5 m); moving him would move the ball too
COVER_MIN_M = 4.0               # m (body): the D1 cover sector starts this far behind the first defender
PRESS_MIN_M = 1.0               # m (body): a pressure area lies at least this far goal-side of the carrier and never
                                # covers him (a hull of the half-disc around him passed through him)
WEAK_DEPTH_M = 8.0              # m x L/105: the D2 region's depth band around the back line's median depth
GAP_DEPTH_M = 3.0               # m (body): the D5 region's depth band around the line's depth
LINE_BAND_M = 8.0               # m x L/105: role-free back line (check iii): outfield players this near the deepest
LINE_MIN = 3                    # a back line needs this many players before a gap is judged (D5 needs 3 too)
MARK_M = 10.0                   # m x L/105: an opponent in our half with a team-mate this close is marked (check iv)
LEVEL_M = 2.0                   # m (body): a D3 zone reaches at most this far upfield of the deepest opponent in the
                                # vacated channel ("level with or goal-side of him"; spec Part 8: cover stays related to
                                # the goal and the attacking threats; critic B-8). About 2 position sigmas of broadcast.
LATERAL_GAP_M = 20.0            # m x W/68: check (iii) threshold when the profile's D5 gives none (LATERAL_GAP_M)
PRESSURE_M = 5.0                # m (body): pressure distance for check (i) when the profile's D1 gives none
SIDE_MAX_RESID_M = 8.0          # m: attack_side() accepts a side whose median track residual is at most this ...
SIDE_MARGIN_M = 5.0             # m: ... and at least this much below the other side's
EDGE_TOL_M = .02                # m: exact checks keep this margin (coordinates are rounded to 1e-4 of the pitch)
STAR_RAYS = 12                  # rays of the fallback star polygon (<= MAX_VERTICES)
SHRINK = .75                    # the fallback polygon shrinks by this factor per failed verification ...
SHRINK_STEPS = 12               # ... at most this many times
DILATE_STEPS = (1.0, .5, 0.0)   # tried dilations, as shares of the mover's sigma
NORM_DIGITS = 4                 # decimals of normalized coordinates in the record (1 cm on 105 m)

ZONE_DETECTORS = ('pressure_cover', 'weak_side_balance', 'wide_defender_advance', 'behind_ball', 'defensive_spacing')
GUARDRAILS_CHECKED = ('identity', 'position', 'role', 'context', 'phase', 'intent', 'evidence')
# Principle codes are explain.PRINCIPLES keys (TEXT words them); the relation codes a zone restores (improves.relation)
# are drills.RELATIONS / explain's: cover, pressure, weak_side, channel_cover, behind_ball, rest_margin, lateral_gap.
PRINCIPLE = {'pressure_cover': 'cover_first_defender', 'weak_side_balance': 'far_side_balance',
             'wide_defender_advance': 'cover_behind_advancing_defender', 'behind_ball': 'someone_back',
             'defensive_spacing': 'unit_spacing'}
REASON_ORDER = ('would_open_new_gap', 'no_reachable_fix', 'mover_unseen', 'context_missing')   # most informative first
CHECK_KEYS = ('pressureKept', 'centralKept', 'gapOk', 'noNewFreeOpponent', 'noStacking')
NOT_JUDGED = 2                  # sample_outcomes' None (outside what the rule judges) in the outcome arrays


# ---------- small helpers ----------
def _num(v):
    return v is not None and not isinstance(v, bool) and isinstance(v, (int, float)) and math.isfinite(float(v))


def _attack_xy(x, y, side, L, W):
    """Normalized pitch -> the team's attack frame in metres (situations: defending left keeps the axes)."""
    if side == 'left':
        return float(x)*L, float(y)*W
    return (1.0-float(x))*L, (1.0-float(y))*W


def _norm_xy(X, Y, side, L, W):
    if side == 'left':
        return X/L, Y/W
    return 1.0-X/L, 1.0-Y/W


def _norm_point(X, Y, side, L, W):
    x, y = _norm_xy(float(X), float(Y), side, L, W)
    return [round(x, NORM_DIGITS), round(y, NORM_DIGITS)]


def _rel(opp, key, k):
    v = (opp.get('relations') or {}).get(key)
    if v is None or not 0 <= k < len(v):
        return None
    return v[k]


def _detector_params(profile, detector):
    return dict(((profile.get('detectors') or {}).get(detector) or {}).get('params') or {})


# ---------- geometry of polygons (attack-frame metres) ----------
def _cross(o, a, b):
    return (a[0]-o[0])*(b[1]-o[1])-(a[1]-o[1])*(b[0]-o[0])


def convex_hull(points):
    """Monotone-chain convex hull, counter-clockwise, no repeated end point (fewer than 3 distinct points: as given)."""
    pts = sorted({(round(float(p[0]), 9), round(float(p[1]), 9)) for p in points})
    if len(pts) <= 2:
        return [list(p) for p in pts]
    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and _cross(lower[-2], lower[-1], p) <= 1e-12:
            lower.pop()
        lower.append(p)
    for p in reversed(pts):
        while len(upper) >= 2 and _cross(upper[-2], upper[-1], p) <= 1e-12:
            upper.pop()
        upper.append(p)
    return [list(p) for p in lower[:-1]+upper[:-1]]


def polygon_area(poly):
    """Unsigned shoelace area."""
    n = len(poly)
    if n < 3:
        return 0.0
    s = 0.0
    for i in range(n):
        x0, y0 = poly[i]
        x1, y1 = poly[(i+1) % n]
        s += x0*y1-x1*y0
    return abs(s)/2.0


def polygon_centroid(poly):
    n = len(poly)
    a = 0.0
    cx = cy = 0.0
    for i in range(n):
        x0, y0 = poly[i]
        x1, y1 = poly[(i+1) % n]
        c = x0*y1-x1*y0
        a += c
        cx += (x0+x1)*c
        cy += (y0+y1)*c
    if abs(a) < 1e-12:
        return [sum(p[0] for p in poly)/n, sum(p[1] for p in poly)/n]
    return [cx/(3.0*a), cy/(3.0*a)]


def _segment_distance(poly, qx, qy):
    """Distances (vector over the query points) from (qx, qy) to the polygon's boundary."""
    qx, qy = np.asarray(qx, float), np.asarray(qy, float)
    best = np.full(qx.shape, np.inf)
    n = len(poly)
    for i in range(n):
        ax, ay = poly[i]
        bx, by = poly[(i+1) % n]
        dx, dy = bx-ax, by-ay
        ll = dx*dx+dy*dy
        t = np.zeros(qx.shape) if ll <= 0 else np.clip(((qx-ax)*dx+(qy-ay)*dy)/ll, 0.0, 1.0)
        best = np.minimum(best, np.hypot(qx-(ax+t*dx), qy-(ay+t*dy)))
    return best


def points_in_polygon(poly, qx, qy, tol=1e-6):
    """Vectorized even-odd point-in-polygon; points within tol of the boundary count as inside."""
    qx, qy = np.asarray(qx, float), np.asarray(qy, float)
    inside = np.zeros(qx.shape, bool)
    n = len(poly)
    if n < 3:
        return inside
    for i in range(n):
        x0, y0 = poly[i]
        x1, y1 = poly[(i+1) % n]
        cond = (y0 > qy) != (y1 > qy)
        with np.errstate(divide='ignore', invalid='ignore'):
            xi = (x1-x0)*(qy-y0)/(y1-y0)+x0
        inside ^= cond & (qx < xi)
    return inside | (_segment_distance(poly, qx, qy) <= tol)


def polygon_diameter(poly):
    best = 0.0
    for i in range(len(poly)):
        for j in range(i+1, len(poly)):
            best = max(best, math.hypot(poly[i][0]-poly[j][0], poly[i][1]-poly[j][1]))
    return best


def _simplify(poly, n=MAX_VERTICES):
    """Visvalingam-Whyatt down to n vertices (removing the vertex of the smallest triangle; on a convex polygon each
    removal cuts a corner, so the result lies inside the original)."""
    pts = [list(p) for p in poly]
    while len(pts) > n:
        m = len(pts)
        areas = [abs(_cross(pts[i-1], pts[i], pts[(i+1) % m]))/2.0 for i in range(m)]
        del pts[int(np.argmin(areas))]
    return pts


def _dilate(poly, r, k=16):
    """A convex polygon grown by r (the hull of its vertices moved by r in k directions)."""
    if r <= 0:
        return [list(p) for p in poly]
    pts = []
    for x, y in poly:
        for a in range(k):
            ang = 2.0*math.pi*a/k
            pts.append((x+r*math.cos(ang), y+r*math.sin(ang)))
    return convex_hull(pts)


def _circle(c, r, k=MAX_VERTICES):
    """A k-gon around the circle of radius r (circumscribed: the whole circle is inside it)."""
    R = r/math.cos(math.pi/k)
    return [[c[0]+R*math.cos(2.0*math.pi*a/k), c[1]+R*math.sin(2.0*math.pi*a/k)] for a in range(k)]


# ---------- the side defended at a moment ----------
def attack_side(obs, moment):
    """'left' / 'right': the goal the team defends at the moment (its attack frame), None when it cannot be told.
    EVIDENCE writes it into the moment (calibration['side'], from the timeline at that frame); without it, DETECT's
    stored own-phase samples (roleInputs['detect'], exact) when one lies within half a step of the moment; else the
    moment's own observed players compared with CONTEXT's 1 Hz attack-frame track (roleInputs['context']): the side
    whose conversion of the scene matches the track (median residual at most SIDE_MAX_RESID_M and SIDE_MARGIN_M below
    the other side's)."""
    side = (moment.get('calibration') or {}).get('side')
    if side in ('left', 'right'):
        return side
    ri = (obs or {}).get('roleInputs') or {}
    t = float(moment['t'])
    geo = ri.get('detect')
    if isinstance(geo, dict) and isinstance(geo.get('samples'), dict):
        ts = geo['samples'].get('t') or []
        sides = geo['samples'].get('side') or []
        if ts and len(sides) == len(ts):
            k = int(np.argmin(np.abs(np.asarray(ts, float)-t)))
            if abs(ts[k]-t) <= float(geo.get('step') or .2)/2.0+1e-6 and sides[k] in ('left', 'right'):
                return sides[k]
    ctx = ri.get('context')
    if not isinstance(ctx, dict) or not isinstance(ctx.get('track'), dict):
        return None
    tr = ctx['track']
    tt = tr.get('t') or []
    if not tt:
        return None
    i = int(np.argmin(np.abs(np.asarray(tt, float)-t)))
    if abs(tt[i]-t) > float(tr.get('step') or 1.0)/2.0+.1:
        return None
    scale = ctx.get('scale') or (obs.get('timeline') or {}).get('scale') or {}
    L, W = float(scale.get('lengthM') or 105.0), float(scale.get('widthM') or 68.0)
    team = obs.get('team')
    res = {'left': [], 'right': []}
    players = (moment.get('scene') or {}).get('players') or {}
    for j, pid in enumerate(ctx.get('ids') or ()):
        e = players.get(pid)
        if not e or e.get('team') != team or e.get('kind') != 'O':
            continue
        xs, ys = tr['x'][j], tr['y'][j]
        if i >= len(xs) or xs[i] is None or ys[i] is None:
            continue
        for side in res:
            X, Y = _attack_xy(e['x'], e['y'], side, L, W)
            res[side].append(math.hypot(X-xs[i], Y-ys[i]))
    if min(len(res['left']), len(res['right'])) < 2:
        return None
    ml, mr = float(np.median(res['left'])), float(np.median(res['right']))
    best, other = ('left', mr) if ml <= mr else ('right', ml)
    if min(ml, mr) <= SIDE_MAX_RESID_M and other-min(ml, mr) >= SIDE_MARGIN_M:
        return best
    return None


def _stored_sample(obs, t):
    """DETECT's stored role-geometry sample at t (index, samples) or (None, None)."""
    geo = ((obs or {}).get('roleInputs') or {}).get('detect')
    if not isinstance(geo, dict) or not isinstance(geo.get('samples'), dict):
        return None, None
    ts = geo['samples'].get('t') or []
    if not ts:
        return None, None
    k = int(np.argmin(np.abs(np.asarray(ts, float)-float(t))))
    if abs(ts[k]-float(t)) > float(geo.get('step') or .2)/2.0+1e-6:
        return None, None
    return k, geo['samples']


# ---------- the scene in the attack frame ----------
class _Scene:
    """A moment's scene (schema Scene) in the team's attack frame: own outfield players (columns, id order), own
    keeper(s), opponents' outfield players, the ball when reliable."""

    def __init__(self, scene, team, side, L, W):
        own, gks, opp = [], [], []
        self.sigma = {}
        for pid in sorted((scene or {}).get('players') or {}):
            e = scene['players'][pid]
            if e.get('kind') != 'O' or not (_num(e.get('x')) and _num(e.get('y'))):
                continue
            X, Y = _attack_xy(e['x'], e['y'], side, L, W)
            self.sigma[pid] = float(e['sigma']) if _num(e.get('sigma')) else None
            keeper = bool(e.get('goalkeeper')) or pid.startswith('GK')
            if e.get('team') == team:
                (gks if keeper else own).append((pid, X, Y))
            elif not keeper:
                opp.append((pid, X, Y))
        self.ids = [p for p, _, _ in own]
        self.X = np.array([x for _, x, _ in own], float)
        self.Y = np.array([y for _, _, y in own], float)
        self.col = {p: j for j, p in enumerate(self.ids)}
        self.gk = np.array([[x, y] for _, x, y in gks], float).reshape(-1, 2)
        self.opp_ids = [p for p, _, _ in opp]
        self.OX = np.array([x for _, x, _ in opp], float)
        self.OY = np.array([y for _, _, y in opp], float)
        self.ocol = {p: j for j, p in enumerate(self.opp_ids)}
        b = (scene or {}).get('ball')
        self.ball = _attack_xy(b[0], b[1], side, L, W) if (scene or {}).get('ballReliable') and b and \
            _num(b[0]) and _num(b[1]) else None

    def pos(self, pid):
        j = self.col.get(pid)
        if j is not None:
            return float(self.X[j]), float(self.Y[j])
        j = self.ocol.get(pid)
        if j is not None:
            return float(self.OX[j]), float(self.OY[j])
        return None


def _moved(sc, j, P):
    """(K x N) own positions with column j replaced by the K candidate points P (j None: nobody moved)."""
    K = len(P)
    X = np.tile(sc.X, (K, 1))
    Y = np.tile(sc.Y, (K, 1))
    if j is not None:
        X[:, j] = P[:, 0]
        Y[:, j] = P[:, 1]
    return X, Y


def _as_list(a):
    """Plain Python values, NaN -> None. Never rounded: a re-measured relation is compared with the level's thresholds
    exactly as the detector's own geometry would be (rounding flipped points at the cover radius / angle)."""
    out = []
    for v in np.asarray(a).tolist():
        if v is None or (isinstance(v, float) and not math.isfinite(v)):
            out.append(None)
        else:
            out.append(v)
    return out


# ---------- the problems: region, movers and the re-measured relation per detector ----------
class _Problem:
    """One failed relation at one moment: what can move, where the principle says to look, and how the relation is
    re-measured with one player moved (relations(j, X, Y) -> the relations sample_outcomes judges)."""

    def __init__(self, **kw):
        self.carrier = None         # (X, Y) of the ball carrier (check i), None when not known
        self.ball = None            # (X, Y) of the ball, or the carrier, for checks (ii) / (iii); None = not known
        self.ball_side = None       # 'left' / 'right': the ball's half (check iii)
        self.line = None            # fixed back-line member columns (D5), None = the role-free band
        self.anchors = []           # ids whose sigma decide a point display
        self.keep = []              # (x, y, radius): points a zone must keep this far from (the carrier for pressure)
        self.x_max = math.inf       # no zone vertex deeper upfield than this (attack frame X): D3 the channel opponent
                                    # + LEVEL_M, every behind-the-ball region also the ball
        self.as_detector = None
        self.__dict__.update(kw)


def _ball_of(sc, opp, k, stored=None):
    """The ball at the moment (attack frame): the scene's reliable ball, else the opportunity's ballX/ballY, else
    DETECT's stored sample; None when unknown. Y may be None (depth only)."""
    if sc.ball is not None:
        return sc.ball
    bx, by = _rel(opp, 'ballX', k), _rel(opp, 'ballY', k)
    if _num(bx):
        return float(bx), (float(by) if _num(by) else None)
    if stored is not None:
        idx, s = stored
        if idx is not None and _num(s['ballX'][idx]):
            return float(s['ballX'][idx]), (float(s['ballY'][idx]) if _num(s['ballY'][idx]) else None)
    return None


def _nearest(sc, cols, x, y, limit):
    cols = [c for c in cols if np.isfinite(sc.X[c])]
    cols.sort(key=lambda c: (math.hypot(sc.X[c]-x, sc.Y[c]-y), sc.ids[c]))
    return [sc.ids[c] for c in cols[:limit]]


def _on_ball(sc, ball):
    """Our players on the ball (within ON_BALL_M of it): never movers in our own phase."""
    if ball is None or ball[1] is None:
        return set()
    d = np.hypot(sc.X-ball[0], sc.Y-ball[1])
    with np.errstate(invalid='ignore'):
        return {sc.ids[c] for c in np.flatnonzero(d <= ON_BALL_M)}


def _region_distance(sc, cols, region, bbox, h):
    """Per column: the distance from the player to the nearest grid point of the region inside bbox (inf: none)."""
    x0, x1, y0, y1 = bbox
    xs = np.arange(math.floor(x0/h), math.ceil(x1/h)+1)*h
    ys = np.arange(math.floor(y0/h), math.ceil(y1/h)+1)*h
    gx, gy = np.meshgrid(xs, ys, indexing='ij')
    P = np.stack([gx.ravel(), gy.ravel()], axis=1)
    P = P[region(P)] if len(P) else P
    out = {}
    for c in cols:
        out[c] = float(np.min(np.hypot(P[:, 0]-sc.X[c], P[:, 1]-sc.Y[c]))) if len(P) else math.inf
    return out


def _rect_distance(x, y, x0, x1, y0, y1):
    dx = max(x0-x, 0.0, x-x1)
    dy = max(y0-y, 0.0, y-y1)
    return math.hypot(dx, dy)


def _held(outs, j, X, Y):
    """Restored = the rule now demonstrates the behaviour (D1, D2, D4)."""
    return outs == 1


def _not_failing(outs, j, X, Y):
    """Restored = the ball-side gap no longer fails (D5: one mover is not asked for the strength band too)."""
    return (outs == 0) | (outs == 1)


def _problem_d1(sc, opp, k, mode, params, scale, L, W):
    carrier = (opp.get('actors') or {}).get('carrier')
    pos = sc.pos(carrier) if carrier else None
    if pos is None:
        cx, cy = _rel(opp, 'carrierX', k), _rel(opp, 'carrierY', k)
        if not (_num(cx) and _num(cy)):
            return 'context_missing'
        pos = (float(cx), float(cy))
    cx, cy = pos
    fd = _rel(opp, 'fd', k)
    if fd not in sc.col:
        return 'context_missing'
    f = sc.col[fd]
    fx, fy = float(sc.X[f]), float(sc.Y[f])
    radius = G.cover_radius(scale, params.get('coverRadiusM', G.COVER_RADIUS_M),
                            params.get('coverRadiusMinM', G.COVER_RADIUS_MIN_M))
    angle = min(float(params.get('coverAngleDeg') or G.COVER_ANGLE_DEG), G.COVER_ANGLE_DEG)
    pm = float(params.get('pressureM') or PRESSURE_M)
    closing = _rel(opp, 'closingMS', k)
    visible = _rel(opp, 'coverVisible', k)
    gx, gy = 0.0, W/2.0

    def relations(j, X, Y):
        K = len(X)
        d = np.hypot(X-cx, Y-cy)
        d[~np.isfinite(d)] = np.inf
        fi = np.argmin(d, axis=1)
        r = np.arange(K)
        d1 = d[r, fi]
        ax, ay = X[r, fi], Y[r, fi]
        dx, dy = X-ax[:, None], Y-ay[:, None]
        dist = np.hypot(dx, dy)
        vx, vy = gx-ax, gy-ay
        with np.errstate(invalid='ignore', divide='ignore'):
            cos = (dx*vx[:, None]+dy*vy[:, None])/(dist*np.hypot(vx, vy)[:, None])
        ang = np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))
        dist[r, fi] = np.inf
        dist[~np.isfinite(dist)] = np.inf
        with np.errstate(invalid='ignore'):
            within = np.isfinite(ang) & (ang <= G.COVER_ANGLE_DEG)
        dd = np.where(within, dist, np.inf)
        ci = np.argmin(dd, axis=1)
        cd = dd[r, ci]
        ca = np.where(np.isfinite(cd), ang[r, ci], np.nan)
        same = fi == f
        return {'t': [0.0]*K, 'd1': _as_list(np.where(np.isfinite(d1), d1, np.nan)),
                'closingMS': [closing if s else None for s in same.tolist()],
                'coverDist': _as_list(np.where(np.isfinite(cd), cd, np.nan)), 'coverAngle': _as_list(ca),
                'coverVisible': [visible]*K, 'carrierX': [cx]*K}

    if mode == 'no_pressure':
        movers = _nearest(sc, list(range(len(sc.ids))), cx, cy, 2)

        def region(P):
            d = np.hypot(P[:, 0]-cx, P[:, 1]-cy)
            with np.errstate(invalid='ignore'):
                return (d <= pm) & (P[:, 0] <= cx-PRESS_MIN_M)
        keep = [(cx, cy, PRESS_MIN_M)]
        relation = 'pressure'
    else:
        def region(P):
            dist = np.hypot(P[:, 0]-fx, P[:, 1]-fy)
            ang = G.angles_at(fx, fy, P[:, 0], P[:, 1], gx, gy)
            with np.errstate(invalid='ignore'):
                return (dist >= COVER_MIN_M) & (dist <= radius) & (ang <= angle) & (P[:, 0] < fx)
        # The first defender's team-mates nearest the cover sector (DESIGN 5.2 says "nearest the first defender": the
        # same players unless one stands deeper behind him, who is the natural cover); never the first defender.
        cols = [c for c in range(len(sc.ids)) if c != f and np.isfinite(sc.X[c])]
        dist = _region_distance(sc, cols, region, (fx-radius, fx, fy-radius, fy+radius), .5)
        cols.sort(key=lambda c: (round(dist[c], 6), sc.ids[c]))
        movers = [sc.ids[c] for c in cols[:2]]
        keep = []
        relation = 'cover'
    return _Problem(movers=movers, region=region, relations=relations, restored=_held, relation=relation,
                    carrier=(cx, cy), ball=(cx, cy), keep=keep,
                    ball_side='left' if cy < W/2.0 else 'right', anchors=[fd]+([carrier] if carrier else []),
                    pressure_m=pm)


def _problem_d2(sc, opp, k, params, scale, L, W):
    side = _rel(opp, 'ballSide', k)
    ball = _ball_of(sc, opp, k)
    if side not in ('left', 'right') or ball is None or ball[1] is None:
        return 'context_missing'
    bx, by = ball
    band_x = G.scaled(LINE_BAND_M, 'length', scale)
    seen = np.isfinite(sc.X)
    if seen.sum() < 1:
        return 'context_missing'
    deep = float(np.min(sc.X[seen]))
    line = seen & (sc.X <= deep+band_x)
    med = float(np.median(sc.X[line]))
    depth = G.scaled(WEAK_DEPTH_M, 'length', scale)
    x0, x1 = max(0.0, med-depth), min(bx, med+depth)
    y0, y1 = (W/2.0, W) if side == 'left' else (0.0, W/2.0)
    if x1 <= x0:
        return 'no_region'
    carrier = (opp.get('actors') or {}).get('carrier')
    cpos = sc.pos(carrier) if carrier else None

    def far(Y):
        return (Y > W/2.0) if side == 'left' else (Y < W/2.0)

    def region(P):
        with np.errstate(invalid='ignore'):
            return (P[:, 0] >= x0) & (P[:, 0] <= x1) & far(P[:, 1])

    def relations(j, X, Y):
        with np.errstate(invalid='ignore'):
            weak = (np.isfinite(X) & (X <= bx) & far(Y)).sum(axis=1)
        return {'t': [0.0]*len(X), 'weakCount': [int(v) for v in weak]}
    cols = [c for c in range(len(sc.ids)) if np.isfinite(sc.X[c])]
    cols.sort(key=lambda c: (_rect_distance(sc.X[c], sc.Y[c], x0, x1, y0, y1), sc.ids[c]))
    return _Problem(movers=[sc.ids[c] for c in cols[:2]], region=region, relations=relations, restored=_held,
                    relation='weak_side', carrier=cpos, ball=(bx, by), ball_side=side, pressure_m=PRESSURE_M)


def _roles_now(roles, ids, t):
    """(role, source, confidence) per id at t from the role book ('' when none)."""
    book = (roles or {}).get('rolebook') or {}
    R, Sr, C = CX.roles_at(book, ids, [float(t)])
    return [str(v) for v in R[0]], [str(v) for v in Sr[0]], [str(v) for v in C[0]]


def _slot(role):
    if not role or '-' not in role or '/' not in role:
        return None
    block, rest = role.split('-', 1)
    a, b = rest.split('/')
    return block, int(a), int(b)


def _channel_threat(sc, xw, side, L, W):
    """(id, depth) of the deepest opponent in the vacated channel at the moment, nobody moved: an opponent outfield
    player in the wide defender's outer third, goal-side of him and in our half (tactics_geom.opponents_in_channel,
    the detector's test f); None when there is none."""
    if not sc.opp_ids:
        return None
    idx = G.opponents_in_channel(sc.OX, sc.OY, xw, side, W, L)
    if not idx:
        return None
    j = min(idx, key=lambda i: (float(sc.OX[i]), sc.opp_ids[i]))
    return sc.opp_ids[j], float(sc.OX[j])


def _problem_d3(sc, opp, k, params, scale, L, W, obs, roles, t, players, as_detector):
    side = opp.get('variant')
    wd = ((opp.get('actors') or {}).get('primary') or [None])[0]
    if side not in ('left', 'right') or wd not in sc.col:
        return 'context_missing'
    w = sc.col[wd]
    R, Sr, C = _roles_now(roles, sc.ids, t)
    small = int(players) != 11
    if small:
        back = []
        for j, (r_, s_, c_) in enumerate(zip(R, Sr, C)):
            sl = _slot(r_)
            if sl and sl[0] == 'back' and (s_ == 'coach' or (s_ == 'relative' and c_ in ('medium', 'high'))):
                back.append((j, sl[1], sl[2]))
        n = max((b[2] for b in back), default=0)
        cbs = [j for j, a, _ in back if 1 < a < n]
        dms = []
    else:
        cbs = [j for j, r_ in enumerate(R) if r_ in CX.CB_ROLES]
        dms = [j for j, (r_, s_) in enumerate(zip(R, Sr)) if r_ in CX.DM_ROLES and s_ in ('coach', 'stable')]
    cbs = [c for c in cbs if c != w]
    if not cbs or not all(np.isfinite(sc.X[c]) for c in cbs):
        return 'context_missing'
    # The opponent the advance left space for (critic B-8): every D3 zone stays level with or goal-side of the deepest
    # one, else a lost ball sends him in behind the "cover". A threat the detector measured but the scene does not show
    # cannot be placed against.
    chan_opp = _channel_threat(sc, float(sc.X[w]), side, L, W)
    och0 = _rel(opp, 'opponentsInChannel', k)
    if chan_opp is None and _num(och0) and och0 >= 1:
        return 'context_missing'
    x_lim = math.inf if chan_opp is None else chan_opp[1]+LEVEL_M
    kl = float(scale['lengthM'])/G.REF_LENGTH
    chan_m = G.CHANNEL_BEHIND_M*kl
    behind_m = G.BEHIND_BALL_M*kl
    cbx0 = float(np.mean(sc.X[cbs]))
    x0, x1, y0, y1 = G.vacated_channel(cbx0, float(sc.X[w]), side, W, chan_m)
    stored = _stored_sample(obs, t)
    ball = _ball_of(sc, opp, k, stored)
    idx, samp = stored
    period = str(int(samp['period'][idx])) if idx is not None and samp.get('period') else '-1'
    medians = (((obs.get('roleInputs') or {}).get('detect') or {}).get('medianY') or {})
    threat = _rel(opp, 'threatN', k)
    behind0 = _rel(opp, 'behindN', k)
    vel = _rel(opp, 'wdVelX', k)
    inter = _rel(opp, 'interchange', k)
    dm_seen = [d for d in dms if np.isfinite(sc.X[d])]
    # The re-measured relations carry the ball depth when the detector's do, so a ball-zone gate in sample_outcomes
    # judges the scene as it judged the stored sample.
    with_ball_x = 'ballX' in (opp.get('relations') or {})

    def relations(j, X, Y):
        K = len(X)
        cbx = X[:, cbs].mean(axis=1)
        xw, yw = X[:, w], Y[:, w]
        adv = xw-cbx
        half = xw-L/2.0
        with np.errstate(invalid='ignore'):
            in_third = G.in_outer_third(Y, side, W)
            occ_m = np.isfinite(X) & in_third & (X <= (xw-chan_m)[:, None])
        occ_m[:, w] = False
        occ = occ_m.any(axis=1)
        cby = Y[:, cbs]
        near_local = np.argmin(cby, axis=1) if side == 'left' else np.argmax(cby, axis=1)
        near_cols = np.asarray(cbs)[near_local]
        adj = []
        for kk in range(K):
            pid = sc.ids[int(near_cols[kk])]
            med = (medians.get(pid) or {})
            m = med.get(period, med.get('-1'))
            sh = G.cb_shift(float(Y[kk, near_cols[kk]]), m if m is not None else math.nan, side)
            adj.append(None if not math.isfinite(sh) else bool(sh >= G.CB_SHIFT_M))
        if dm_seen:
            dmd = (X[:, dm_seen] <= (cbx+G.DM_DROP_M)[:, None]).any(axis=1).tolist()
        else:
            dmd = [None]*K
        if sc.opp_ids:
            with np.errstate(invalid='ignore'):
                och = (G.in_outer_third(sc.OY, side, W)[None, :] & (sc.OX[None, :] < xw[:, None]) &
                       (sc.OX[None, :] < L/2.0)).sum(axis=1)
        else:
            och = np.zeros(K, int)
        if ball is not None and behind0 is not None:
            with np.errstate(invalid='ignore'):
                bn = (np.isfinite(X) & (X <= ball[0]-behind_m)).sum(axis=1)
            behind = [int(v) for v in bn]
        else:
            behind = [behind0]*K
        out = {'t': [0.0]*K, 'advanceM': _as_list(adv), 'beyondHalfM': _as_list(half),
               'channelOccupied': [bool(v) for v in occ], 'cbAdjusted': adj, 'interchange': [inter]*K,
               'dmDropped': dmd, 'opponentsInChannel': [int(v) for v in och],
               'wdVelX': [None if j == w else vel]*K, 'behindN': behind,
               'threatN': [threat]*K,
               'restMargin': [None if (b is None or threat is None) else b-threat for b in behind]}
        if with_ball_x:
            out['ballX'] = [None if ball is None else float(ball[0])]*K
        return out

    adv_m = CX._scaled_param(dict(CX.REFERENCE_PARAMS['wide_defender_advance'], **params), 'advancedM', 15.0, scale)
    halfway = params.get('advancedRule') == 'halfway'
    share = float(params.get('advancedHalfwayShare') or G.ADVANCED_HALFWAY_SHARE)
    # Level 2 (teamLevelOnly, "someone stays back"): the principle is a player behind the ball, so the zone is the
    # behind-the-ball region, not the vacated channel (critic B-8 / FACTS M6: the channel zone lay 2-8 m ahead of the
    # ball and left the behind count at 0).
    team_level = as_detector is None and bool(params.get('teamLevelOnly'))
    bx = None if ball is None else float(ball[0])

    def restored(outs, j, X, Y):
        # Restored: the advance is covered (held), or the wide defender himself no longer stands advanced ("delay the
        # run": the rule then has nothing to judge). The level-1 reframing judges only the behind-the-ball count; level
        # 2 (requireNobodyBehind) wants the rule to stop failing with someone now behind the ball.
        if as_detector:
            return outs == 1
        if team_level:
            with np.errstate(invalid='ignore'):
                back = (np.isfinite(X) & (X <= bx-behind_m)).sum(axis=1) >= 1
            return (outs != -1) & back
        if halfway:
            still = (X[:, w]-L/2.0) >= share*L
        else:
            still = (X[:, w]-X[:, cbs].mean(axis=1)) >= adv_m
        return (outs == 1) | ~still
    if as_detector == 'behind_ball' or team_level:
        if ball is None:
            return 'context_missing'

        def region(P):
            with np.errstate(invalid='ignore'):
                return (P[:, 0] <= bx-behind_m) & (P[:, 0] <= x_lim) & (P[:, 1] >= W/6.0) & (P[:, 1] <= 5.0*W/6.0)
        busy = _on_ball(sc, ball)
        cand = [c for c in range(len(sc.ids)) if np.isfinite(sc.X[c]) and sc.X[c] > bx-behind_m and
                sc.ids[c] not in busy]
        cand.sort(key=lambda c: (sc.X[c], sc.ids[c]))
        movers = [sc.ids[c] for c in cand[:2]]
        relation = 'behind_ball'
        x_max = min(x_lim, bx)
    else:
        if x1 <= x0:
            return 'no_region'

        def region(P):
            with np.errstate(invalid='ignore'):
                return (P[:, 0] >= x0) & (P[:, 0] <= x1) & (P[:, 0] <= x_lim) & G.in_outer_third(P[:, 1], side, W)
        x_max = x_lim
        near = min(cbs, key=lambda c: sc.Y[c]) if side == 'left' else max(cbs, key=lambda c: sc.Y[c])
        movers = [wd, sc.ids[near]]
        if params.get('useDmRotation') and dm_seen:
            movers.append(sc.ids[min(dm_seen, key=lambda d: (math.hypot(sc.X[d]-sc.X[near], sc.Y[d]-sc.Y[near]),
                                                             sc.ids[d]))])
        busy = _on_ball(sc, ball)
        movers = [pid for pid in movers if pid not in busy]
        rest = [c for c in range(len(sc.ids)) if sc.ids[c] not in movers and sc.ids[c] not in busy
                and np.isfinite(sc.X[c])]
        rest.sort(key=lambda c: (_rect_distance(sc.X[c], sc.Y[c], x0, x1, y0, y1), sc.ids[c]))
        movers += [sc.ids[c] for c in rest[:MAX_MOVERS-len(movers)]]
        movers = movers[:MAX_MOVERS]
        relation = 'channel_cover'
    has_y = ball is not None and ball[1] is not None
    return _Problem(movers=movers, region=region, relations=relations, restored=restored, relation=relation,
                    ball=ball if has_y else None, ball_side=('left' if ball[1] < W/2.0 else 'right') if has_y else None,
                    anchors=[wd], pressure_m=PRESSURE_M, as_detector=as_detector, x_max=x_max)


def _problem_d4(sc, opp, k, params, scale, L, W, obs, t):
    ball = _ball_of(sc, opp, k, _stored_sample(obs, t))
    if ball is None:
        return 'context_missing'
    bx = ball[0]
    margin = G.BEHIND_BALL_M*float(scale['lengthM'])/G.REF_LENGTH
    threat = _rel(opp, 'threatN', k)

    def region(P):
        with np.errstate(invalid='ignore'):
            return (P[:, 0] <= bx-margin) & (P[:, 1] >= W/6.0) & (P[:, 1] <= 5.0*W/6.0)

    def relations(j, X, Y):
        with np.errstate(invalid='ignore'):
            n = (np.isfinite(X) & (X <= bx-margin)).sum(axis=1)
        return {'t': [0.0]*len(X), 'ballX': [bx]*len(X), 'behindN': [int(v) for v in n],
                'restMargin': [None if threat is None else int(v)-int(threat) for v in n]}
    busy = _on_ball(sc, ball)
    cand = [c for c in range(len(sc.ids)) if np.isfinite(sc.X[c]) and sc.X[c] > bx-margin and sc.ids[c] not in busy]
    cand.sort(key=lambda c: (sc.X[c], sc.ids[c]))
    relation = 'rest_margin' if params.get('behindRule') == 'margin' else 'behind_ball'
    has_y = ball[1] is not None
    return _Problem(movers=[sc.ids[c] for c in cand[:2]], region=region, relations=relations, restored=_held,
                    relation=relation, ball=ball if has_y else None,
                    ball_side=('left' if ball[1] < W/2.0 else 'right') if has_y else None, pressure_m=PRESSURE_M,
                    x_max=float(bx))


def _problem_d5(sc, opp, k, params, scale, L, W):
    a, b = _rel(opp, 'gapLeft', k), _rel(opp, 'gapRight', k)
    size, height = _rel(opp, 'lineSize', k), _rel(opp, 'lineHeightM', k)
    gap0 = _rel(opp, 'lateralGapM', k)
    if a not in sc.col or b not in sc.col or not _num(size) or not _num(height) or not _num(gap0):
        return 'context_missing'
    ca, cb = sc.col[a], sc.col[b]
    others = [c for c in range(len(sc.ids)) if c not in (ca, cb) and np.isfinite(sc.X[c])]
    others.sort(key=lambda c: (abs(sc.X[c]-float(height)), sc.ids[c]))
    members = [ca, cb]+others[:max(0, int(size)-2)]
    if len(members) < LINE_MIN:
        return 'context_missing'
    gap, li, ri, _ = G.lateral_gaps(sc.Y[members])
    if not math.isfinite(gap) or abs(gap-float(gap0)) > 1.5 or {members[li], members[ri]} != {ca, cb}:
        return 'context_missing'           # the scene does not reproduce the measured line
    ball = _ball_of(sc, opp, k)
    side = _rel(opp, 'ballSide', k)
    side = side if side in ('left', 'right') else (('left' if ball[1] < W/2.0 else 'right')
                                                    if ball and ball[1] is not None else None)
    lx = float(np.mean(sc.X[members]))
    lo_y, hi_y = sorted((float(sc.Y[ca]), float(sc.Y[cb])))
    dm0 = _rel(opp, 'dmGapM', k)
    kl = float(scale['lengthM'])/G.REF_LENGTH
    mem = np.asarray(members)

    def region(P):
        with np.errstate(invalid='ignore'):
            return (P[:, 1] > lo_y) & (P[:, 1] < hi_y) & (np.abs(P[:, 0]-lx) <= GAP_DEPTH_M)

    def relations(j, X, Y):
        K = len(X)
        Xm, Ym = X[:, mem], Y[:, mem]
        order = np.argsort(Ym, axis=1, kind='stable')
        rr = np.arange(K)[:, None]
        ys = Ym[rr, order]
        d = np.diff(ys, axis=1)
        kk = np.argmax(d, axis=1)
        r1 = np.arange(K)
        gap = d[r1, kk]
        lft, rgt = order[r1, kk], order[r1, kk+1]
        gy = (ys[r1, kk]+ys[r1, kk+1])/2.0
        xl, xr = Xm[r1, lft], Xm[r1, rgt]
        yl, yr = Ym[r1, lft], Ym[r1, rgt]
        if len(sc.opp_ids):
            lo_x = np.minimum(xl, xr)-G.GAP_BEHIND_M*kl
            hi_x = np.maximum(xl, xr)+G.GAP_AHEAD_M*kl
            ly, hy = np.minimum(yl, yr), np.maximum(yl, yr)
            with np.errstate(invalid='ignore'):
                inside = ((sc.OY[None, :] > ly[:, None]) & (sc.OY[None, :] < hy[:, None]) &
                          (sc.OX[None, :] >= lo_x[:, None]) & (sc.OX[None, :] <= hi_x[:, None]))
            in_gap = inside.any(axis=1)
        else:
            in_gap = np.zeros(K, bool)
        ball_side = [None]*K if side is None else [bool((g < W/2.0) == (side == 'left')) for g in gy]
        std = Xm.std(axis=1)
        mean0 = float(np.mean(sc.X[mem]))
        dm = [None if dm0 is None else float(dm0)+mean0-float(v) for v in Xm.mean(axis=1)]
        return {'t': [0.0]*K, 'lateralGapM': _as_list(gap), 'gapBallSide': ball_side,
                'opponentInGap': [bool(v) for v in in_gap], 'dmGapM': dm, 'lineStdM': _as_list(std)}
    prim = ((opp.get('actors') or {}).get('primary') or [None])[0]
    movers = [prim] if prim in (a, b) else []
    movers += [p for p in (a, b) if p not in movers]
    carrier = (opp.get('actors') or {}).get('carrier')
    return _Problem(movers=movers, region=region, relations=relations, restored=_not_failing, relation='lateral_gap',
                    carrier=sc.pos(carrier) if carrier else None,
                    ball=ball if (ball and ball[1] is not None) else None, ball_side=side, line=members,
                    anchors=[a, b], pressure_m=PRESSURE_M)


# ---------- the checks of step 7 (vectorized over candidate points) ----------
def _line_bad(X, Y, members, band, thr, side, W):
    """Per row: is the largest back-line gap above thr and on the ball's side? members: fixed columns (D5) or None
    (the role-free line: players within `band` of the deepest)."""
    K, N = X.shape
    if members is not None:
        mask = np.zeros((K, N), bool)
        mask[:, members] = True
        mask &= np.isfinite(X)
    else:
        with np.errstate(invalid='ignore'):
            deep = np.nanmin(np.where(np.isfinite(X), X, np.inf), axis=1)
            mask = np.isfinite(X) & (X <= deep[:, None]+band)
    n = mask.sum(axis=1)
    Yv = np.where(mask, Y, np.nan)
    ys = np.sort(Yv, axis=1)
    d = np.diff(ys, axis=1) if N > 1 else np.full((K, 0), np.nan)
    if d.shape[1] == 0:
        return np.zeros(K, bool)
    d = np.where(np.isfinite(d), d, -np.inf)
    kk = np.argmax(d, axis=1)
    r = np.arange(K)
    gap = d[r, kk]
    centre = (ys[r, kk]+ys[r, np.minimum(kk+1, N-1)])/2.0
    if side == 'left':
        on = centre < W/2.0
    elif side == 'right':
        on = centre > W/2.0
    else:
        on = np.ones(K, bool)
    return (n >= LINE_MIN) & np.isfinite(gap) & (gap > thr) & on


def _checks(sc, j, P, pb, scale, L, W, gap_thr):
    """The five no-new-gap checks of DESIGN 5.1 step 7 for mover column j at the K points P: {key: bool array}."""
    K = len(P)
    X, Y = _moved(sc, j, P)
    X0, Y0 = sc.X[None, :], sc.Y[None, :]
    out = {}
    # (i) pressure on the carrier kept
    if pb.carrier is not None:
        cx, cy = pb.carrier
        before = bool(np.any(np.hypot(X0-cx, Y0-cy) <= pb.pressure_m))
        out['pressureKept'] = (np.hypot(X-cx, Y-cy) <= pb.pressure_m).any(axis=1) if before else np.ones(K, bool)
    else:
        out['pressureKept'] = np.ones(K, bool)
    # (ii) central protection (goal-side of the ball within the central band) does not drop to nothing
    band = G.scaled(G.CENTRAL_BAND_M, 'width', scale)
    if pb.ball is not None:
        bx = pb.ball[0]
        with np.errstate(invalid='ignore'):
            before = int((np.isfinite(X0) & (X0 <= bx) & (np.abs(Y0-W/2.0) <= band)).sum())
            after = (np.isfinite(X) & (X <= bx) & (np.abs(Y-W/2.0) <= band)).sum(axis=1)
        out['centralKept'] = after >= 1 if before >= 1 else np.ones(K, bool)
    else:
        out['centralKept'] = np.ones(K, bool)
    # (iii) no new ball-side back-line gap above the threshold
    lband = G.scaled(LINE_BAND_M, 'length', scale)
    was = bool(_line_bad(X0, Y0, pb.line, lband, gap_thr, pb.ball_side, W)[0])
    out['gapOk'] = np.ones(K, bool) if was else ~_line_bad(X, Y, pb.line, lband, gap_thr, pb.ball_side, W)
    # (iv) no opponent in our half newly left without a team-mate within MARK_M
    mark = G.scaled(MARK_M, 'length', scale)
    if len(sc.opp_ids):
        with np.errstate(invalid='ignore'):
            ours = np.isfinite(sc.OX) & (sc.OX < L/2.0)
        if ours.any():
            OX, OY = sc.OX[ours], sc.OY[ours]
            dist = np.hypot(sc.X[:, None]-OX[None, :], sc.Y[:, None]-OY[None, :])
            dist[~np.isfinite(dist)] = np.inf
            marked = (dist <= mark).any(axis=0)
            others = np.delete(dist, j, axis=0) if j is not None else dist
            still = (others <= mark).any(axis=0) if len(others) else np.zeros(len(OX), bool)
            lost = marked & ~still
            if lost.any():
                new = np.hypot(P[:, 0][:, None]-OX[None, lost], P[:, 1][:, None]-OY[None, lost]) <= mark
                out['noNewFreeOpponent'] = new.all(axis=1)
            else:
                out['noNewFreeOpponent'] = np.ones(K, bool)
        else:
            out['noNewFreeOpponent'] = np.ones(K, bool)
    else:
        out['noNewFreeOpponent'] = np.ones(K, bool)
    # (v) no stacking on another team-mate (keepers included)
    mates = _mates(sc, j)
    if len(mates):
        d = np.hypot(P[:, 0][:, None]-mates[None, :, 0], P[:, 1][:, None]-mates[None, :, 1])
        out['noStacking'] = (d >= STACK_M+EDGE_TOL_M).all(axis=1)
    else:
        out['noStacking'] = np.ones(K, bool)
    return out


def _mates(sc, j):
    pts = [[sc.X[c], sc.Y[c]] for c in range(len(sc.ids)) if c != j and np.isfinite(sc.X[c])]
    pts += [list(g) for g in sc.gk.tolist() if all(math.isfinite(v) for v in g)]
    return np.asarray(pts, float).reshape(-1, 2)


# ---------- one mover: the grid, the accepted points, the zone ----------
class _Grid:
    """The rectangular grid around the reach disc (one cell margin): points P (n x 2), the valid ones (inside the
    disc and the pitch), and per point the region / restore / checks results."""

    def __init__(self, centre, r, h, L, W):
        self.h = h
        i0 = int(math.floor((centre[0]-r)/h))-1
        i1 = int(math.ceil((centre[0]+r)/h))+1
        j0 = int(math.floor((centre[1]-r)/h))-1
        j1 = int(math.ceil((centre[1]+r)/h))+1
        self.i0, self.j0 = i0, j0
        self.nx, self.ny = i1-i0+1, j1-j0+1
        ii, jj = np.meshgrid(np.arange(i0, i1+1), np.arange(j0, j1+1), indexing='ij')
        self.P = np.stack([ii.ravel()*h, jj.ravel()*h], axis=1).astype(float)
        d = np.hypot(self.P[:, 0]-centre[0], self.P[:, 1]-centre[1])
        self.valid = (d <= r-EDGE_TOL_M) & (self.P[:, 0] >= 0.0) & (self.P[:, 0] <= L) & (self.P[:, 1] >= 0.0) & \
            (self.P[:, 1] <= W)

    def index(self, q):
        i = int(round(q[0]/self.h))-self.i0
        j = int(round(q[1]/self.h))-self.j0
        if 0 <= i < self.nx and 0 <= j < self.ny:
            return i*self.ny+j
        return None


def _components(grid, mask):
    """Connected sets (8 neighbours) of the masked grid points, largest first (ties: the first index)."""
    seen = np.zeros(len(mask), bool)
    comps = []
    for start in np.flatnonzero(mask):
        if seen[start]:
            continue
        comp, q = [], deque([int(start)])
        seen[start] = True
        while q:
            p = q.popleft()
            comp.append(p)
            i, j = divmod(p, grid.ny)
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    if not di and not dj:
                        continue
                    a, b = i+di, j+dj
                    if 0 <= a < grid.nx and 0 <= b < grid.ny:
                        n = a*grid.ny+b
                        if mask[n] and not seen[n]:
                            seen[n] = True
                            q.append(n)
        comps.append(sorted(comp))
    comps.sort(key=lambda c: (-len(c), c[0]))
    return comps


def _stored(poly, side, L, W):
    """The polygon exactly as the record will hold it: rounded to NORM_DIGITS in normalized coordinates and turned
    back into the attack frame (a grid point on the boundary must not move out of the zone by the rounding)."""
    out = []
    for X, Y in poly:
        x, y = _norm_point(X, Y, side, L, W)
        out.append(list(_attack_xy(x, y, side, L, W)))
    return out


def _verify(poly, grid, accepted, hard, centre, r, mates, L, W, keep=(), x_max=math.inf):
    """The polygon's share of restoring grid points when it verifies (DESIGN 13.1 property: inside the reach disc,
    no team-mate within STACK_M, every inside grid point passing the checks, >= ZONE_KEEP_SHARE restoring), else
    None. poly: as stored (_stored); keep: (x, y, radius) points it must stay clear of; x_max: no vertex deeper upfield
    (the D3 channel opponent + LEVEL_M, the ball for a behind-the-ball zone: the sigma dilation must not cross it)."""
    if len(poly) < 3 or polygon_area(poly) <= 1e-6:
        return None
    V = np.asarray(poly, float)
    if np.any(np.hypot(V[:, 0]-centre[0], V[:, 1]-centre[1]) > r-EDGE_TOL_M):
        return None
    if np.any(V[:, 0] > x_max+EDGE_TOL_M):
        return None
    if np.any(V[:, 0] < -1e-9) or np.any(V[:, 0] > L+1e-9) or np.any(V[:, 1] < -1e-9) or np.any(V[:, 1] > W+1e-9):
        return None
    if len(mates):
        if points_in_polygon(poly, mates[:, 0], mates[:, 1]).any():
            return None
        if np.any(_segment_distance(poly, mates[:, 0], mates[:, 1]) < STACK_M+EDGE_TOL_M):
            return None
    for kx, ky, kr in keep:
        if points_in_polygon(poly, [kx], [ky])[0] or _segment_distance(poly, [kx], [ky])[0] < kr-1e-9:
            return None
    inside = points_in_polygon(poly, grid.P[:, 0], grid.P[:, 1])
    if inside.sum() < MIN_ZONE_POINTS or not np.all(grid.valid[inside]) or not np.all(hard[inside]):
        return None
    share = float(accepted[inside].mean())
    return share if share >= ZONE_KEEP_SHARE-1e-9 else None


def _star(grid, comp_set, seed, h):
    """A <= STAR_RAYS-vertex star polygon around seed, each ray stopped at the last point still in the component."""
    poly = []
    for a in range(STAR_RAYS):
        ang = 2.0*math.pi*a/STAR_RAYS
        ux, uy = math.cos(ang), math.sin(ang)
        d, step = 0.0, h/4.0
        while d < 500.0:
            n = grid.index((seed[0]+(d+step)*ux, seed[1]+(d+step)*uy))
            if n is None or n not in comp_set:
                break
            d += step
        d = max(d, h/4.0)
        poly.append([seed[0]+d*ux, seed[1]+d*uy])
    return poly


def _shape(grid, accepted, hard, comp, centre, r, mates, sigma, side, L, W, keep=(), x_max=math.inf):
    """(polygon as stored, share) of the zone from one connected set of accepted points (step 8), or (None, None)."""
    pts = grid.P[comp]
    hull = convex_hull(pts.tolist())
    if len(hull) >= 3:
        for k in DILATE_STEPS:
            poly = _stored(_simplify(_dilate(hull, sigma*k)) if k > 0 else _simplify(hull), side, L, W)
            share = _verify(poly, grid, accepted, hard, centre, r, mates, L, W, keep, x_max)
            if share is not None:
                return poly, share
    # Fallback: a star polygon around the most interior accepted point, shrunk until it verifies.
    comp_set = set(int(c) for c in comp)
    out = np.ones(len(grid.P), bool)
    out[list(comp_set)] = False
    far = grid.P[out]
    if len(far):
        d = np.min(np.hypot(pts[:, 0][:, None]-far[None, :, 0], pts[:, 1][:, None]-far[None, :, 1]), axis=1)
    else:
        d = np.zeros(len(pts))
    cen = pts.mean(axis=0)
    order = sorted(range(len(pts)), key=lambda i: (-round(float(d[i]), 6),
                                                  math.hypot(pts[i][0]-cen[0], pts[i][1]-cen[1]), i))
    seed = pts[order[0]].tolist()
    star = _star(grid, comp_set, seed, grid.h)
    for step in range(SHRINK_STEPS+1):
        f = SHRINK**step
        poly = _stored([[seed[0]+(p[0]-seed[0])*f, seed[1]+(p[1]-seed[1])*f] for p in star], side, L, W)
        share = _verify(poly, grid, accepted, hard, centre, r, mates, L, W, keep, x_max)
        if share is not None:
            return poly, share
    return None, None


def _speed(obs, pid, cap):
    """min(the player's own p90 speed in this run, the age cap); a player without one gets his team's median p90
    (same run), else the cap."""
    players = ((obs.get('timeline') or {}).get('players') or {})
    own = (players.get(pid) or {}).get('p90SpeedMS')
    if not _num(own):
        team = (players.get(pid) or {}).get('team') or obs.get('team')
        vals = [float(v['p90SpeedMS']) for k, v in players.items()
                if v.get('team') == team and not v.get('goalkeeper') and _num(v.get('p90SpeedMS'))]
        own = float(np.median(vals)) if vals else cap
    return max(SPEED_MIN_MS, min(float(own), float(cap)))


def _try_mover(pb, sc, before, pid, ctx):
    """One mover: ('ok', result dict) or (reason, None)."""
    j = sc.col.get(pid)
    if j is None or not np.isfinite(sc.X[j]) or pid not in before:
        return 'mover_unseen', None
    L, W, scale = ctx['L'], ctx['W'], ctx['scale']
    v = round(_speed(ctx['obs'], pid, ctx['cap']), 2)
    H = round(ctx['horizon'], 2)
    r = round(H*v, 2)
    centre = before[pid]
    grid = _Grid(centre, r, ctx['h'], L, W)
    valid = grid.valid
    n = len(grid.P)
    region = np.zeros(n, bool)
    if valid.any():
        region[valid] = pb.region(grid.P[valid])
    if not region.any():
        return 'no_reachable_fix', None
    restored = np.zeros(n, bool)
    cand = np.flatnonzero(region)
    Pc = grid.P[cand]
    X, Y = _moved(sc, j, Pc)
    rel = pb.relations(j, X, Y)
    outs, _ = CX.sample_outcomes({'detector': ctx['opp_detector'], 'relations': rel}, ctx['params'], scale,
                                 as_detector=pb.as_detector)
    o = np.array([NOT_JUDGED if x is None else x for x in outs], int)
    restored[cand] = np.asarray(pb.restored(o, j, X, Y), bool)
    if not restored.any():
        return 'no_reachable_fix', None
    hard = np.zeros(n, bool)
    if valid.any():
        ch = _checks(sc, j, grid.P[valid], pb, scale, L, W, ctx['gap_thr'])
        hard[valid] = np.logical_and.reduce([ch[key] for key in CHECK_KEYS])
    accepted = region & restored & hard
    if not accepted.any():
        return 'would_open_new_gap', None
    comps = _components(grid, accepted)
    mates = _mates(sc, j)
    sigma = max(SIGMA_MIN_M, sc.sigma.get(pid) or 0.0)
    for comp in comps:
        poly, share = _shape(grid, accepted, hard, comp, centre, r, mates, sigma, ctx['side'], L, W, pb.keep,
                             pb.x_max)
        if poly is not None:
            pts = grid.P[comp]
            # effort: the share of his reach the nearest restoring point needs (0 = he stood there REACH_H earlier)
            effort = float(np.min(np.hypot(pts[:, 0]-centre[0], pts[:, 1]-centre[1])))/max(r, 1e-9)
            return 'ok', {'mover': pid, 'poly': poly, 'share': share, 'effort': effort, 'speed': v, 'horizon': H,
                          'extent': polygon_diameter(convex_hull(pts.tolist())),
                          'core': [float(pts[:, 0].mean()), float(pts[:, 1].mean())],
                          'radius': r, 'centre': centre, 'now': (float(sc.X[j]), float(sc.Y[j])), 'grid': grid,
                          'accepted': accepted, 'hard': hard, 'mates': mates, 'sigma': sigma, 'keep': pb.keep,
                          'x_max': pb.x_max}
    # No verified area. The checks are to blame when they removed restoring points and the restoring points alone
    # would have made one; otherwise only a sliver was reachable (a single row of points makes no area: with the D3
    # threat limit the 9 points of one column of the channel came back as would_open_new_gap with no check failing).
    removed = bool((region & restored & ~hard).any())
    area = removed and any(len(c) >= MIN_ZONE_POINTS for c in _components(grid, region & restored))
    return ('would_open_new_gap' if area else 'no_reachable_fix'), None


# ---------- the stage ----------
def _no_alternative(finding, moment, principle, reason, rejected):
    return {'finding': finding['id'], 'evidence': moment['id'] if moment else None,
            't': float(moment['t']) if moment else None, 'principle': principle, 'reason': reason,
            'rejected': [dict(x) for x in rejected]}


def _principle_of(finding, profile):
    det = finding['observation']
    mode = (finding.get('measurements') or {}).get('failureMode')
    if det == 'pressure_cover' and mode == 'no_pressure':
        return 'pressure_on_ball'
    if det == 'wide_defender_advance' and _detector_params(profile, det).get('teamLevelOnly'):
        return 'someone_back'
    return PRINCIPLE.get(det, 'height_width_depth')


def _index(obs, roles, key):
    out = {}
    for rec in list((obs or {}).get(key) or [])+list((roles or {}).get(key) or []):
        out[rec['id']] = rec
    return out


def _at_moment(finding, moment, opp, obs, roles, profile, ctx):
    """(Alternative, None) or (None, (reason, rejected)) for one cited moment."""
    det = finding['observation']
    params = ctx['params']
    scale = ctx['scale']
    L, W = ctx['L'], ctx['W']
    if moment is None or opp is None:
        return None, ('context_missing', [])
    k = int(moment.get('sample', -1))
    if not 0 <= k < len(opp['relations']['t']):
        return None, ('context_missing', [])
    as_detector = det if opp['detector'] != det else None
    outs, modes = CX.sample_outcomes(opp, params, scale, as_detector=as_detector)
    if outs[k] != -1:
        return None, ('context_missing', [])
    mode = modes[k]
    want = (finding.get('measurements') or {}).get('failureMode')
    if want and mode and want != mode:
        return None, ('context_missing', [])
    side = attack_side(obs, moment)
    if side is None:
        return None, ('context_missing', [])
    sc = _Scene(moment.get('scene'), finding['team'], side, L, W)
    t = float(moment['t'])
    if opp['detector'] == 'pressure_cover':
        pb = _problem_d1(sc, opp, k, mode, params, scale, L, W)
    elif opp['detector'] == 'weak_side_balance':
        pb = _problem_d2(sc, opp, k, params, scale, L, W)
    elif opp['detector'] == 'wide_defender_advance':
        players = int(str((obs.get('timeline') or {}).get('format') or '11v11').split('v')[0])
        pb = _problem_d3(sc, opp, k, params, scale, L, W, obs, roles, t, players, as_detector)
    elif opp['detector'] == 'behind_ball':
        pb = _problem_d4(sc, opp, k, params, scale, L, W, obs, t)
    elif opp['detector'] == 'defensive_spacing':
        if mode == 'dm_gap':
            return None, ('no_zone_for_detector', [])
        pb = _problem_d5(sc, opp, k, params, scale, L, W)
    else:
        return None, ('no_zone_for_detector', [])
    if pb == 'no_region':
        return None, ('no_reachable_fix', [])
    if isinstance(pb, str):
        return None, (pb, [])
    if not pb.movers:
        return None, ('context_missing', [])
    # The scene must show the failure: re-measured with nobody moved, the relation fails too.
    X0, Y0 = _moved(sc, None, np.zeros((1, 2)))
    o0, _ = CX.sample_outcomes({'detector': opp['detector'], 'relations': pb.relations(None, X0, Y0)}, params, scale,
                               as_detector=pb.as_detector)
    if o0[0] != -1:
        return None, ('context_missing', [])
    # Movers REACH_H earlier.
    bsc = moment.get('before')
    before = {}
    horizon = float(params.get('reachHorizonS') or REACH_H_S)
    if isinstance(bsc, dict) and _num(bsc.get('t')):
        h_eff = t-float(bsc['t'])
        if abs(h_eff-horizon) <= REACH_TOL_S and h_eff > 0:
            bs = _Scene(bsc, finding['team'], side, L, W)
            before = {pid: bs.pos(pid) for pid in bs.ids}
            horizon = h_eff
    mctx = dict(ctx, horizon=horizon, opp_detector=opp['detector'], side=side)
    results, rejected = [], []
    for pid in pb.movers[:MAX_MOVERS]:
        status, res = _try_mover(pb, sc, before, pid, mctx)
        if status == 'ok':
            results.append(res)
        else:
            rejected.append({'mover': pid, 'reason': status})
    if not results:
        reasons = [r_['reason'] for r_ in rejected]
        reason = next(x for x in REASON_ORDER if x in reasons)
        return None, (reason, rejected)
    order = {pid: i for i, pid in enumerate(pb.movers)}
    best = min(results, key=lambda r_: (round(r_['effort'], 6), order[r_['mover']]))
    return _alternative(finding, moment, pb, sc, best, rejected, side, ctx), None


def _alternative(finding, moment, pb, sc, best, rejected, side, ctx):
    L, W = ctx['L'], ctx['W']
    poly = best['poly']
    area = polygon_area(poly)
    kind, radius, display = 'zone', None, 'zone'
    involved = [best['mover']]+[a for a in pb.anchors if a]
    sig = [sc.sigma.get(pid) for pid in involved]
    # Point or zone (DESIGN 5.3) is decided on the restoring set itself (its grid points' extent), not on the zone
    # as drawn, which the sigma dilation widens.
    if (ctx['params'].get('zonePoint') and best['extent'] <= POINT_MAX_DIAMETER_M+1e-9 and
            moment.get('tier') == 'world' and all(_num(s) and s <= POINT_MAX_SIGMA_M for s in sig)):
        rad = max(float(sc.sigma.get(best['mover']) or 0.0), POINT_MIN_RADIUS_M)
        circle = _stored(_circle(best['core'], rad), side, L, W)
        share = _verify(circle, best['grid'], best['accepted'], best['hard'], best['centre'], best['radius'],
                        best['mates'], L, W, best['keep'], best['x_max'])
        if share is not None:
            poly, kind, radius, display = circle, 'point', round(rad, 2), 'point'
            best = dict(best, share=share)
            area = polygon_area(poly)
    if kind == 'zone' and area > ZONE_MAX_HALF_SHARE*L*W/2.0:
        display = 'direction'
    norm = [_norm_point(p[0], p[1], side, L, W) for p in poly]
    now = best['now']
    return {'id': S.record_id('alt', finding['key'], moment['id'], best['mover']), 'finding': finding['id'],
            'evidence': moment['id'], 't': float(moment['t']), 'mover': best['mover'], 'principle': ctx['principle'],
            'zone': {'polygon': norm, 'areaM2': round(area, 2), 'kind': kind, 'radiusM': radius},
            'display': display, 'from': _norm_point(now[0], now[1], side, L, W),
            'reach': {'horizonS': best['horizon'], 'speedMS': best['speed'], 'radiusM': best['radius']},
            'improves': {'relation': pb.relation, 'before': False, 'after': True,
                         'shareOfZone': round(min(1.0, best['share']), 3)},
            'checks': {k: True for k in CHECK_KEYS}, 'assumption': 'single_mover',
            'rejected': [dict(x) for x in rejected if x['mover'] != best['mover']]}


def propose(finding, obs, roles, profile):
    """The 'alternatives' stage (INTERFACES.md LEAD 4) for one work_on / try_this / review finding: an Alternative
    (schema 'alternative': the zone a single team-mate could have reached to restore the failed relation), a
    NoAlternative (schema 'no_alternative': why none is drawn) or None for a detector without zones (attacking
    shape: the finding keeps its principle). Pure: the same inputs give the same record."""
    det = finding['observation']
    if det not in ZONE_DETECTORS:
        return None
    principle = _principle_of(finding, profile)
    stamps = finding.get('timestamps') or []
    moments = _index(obs, roles, 'moments')
    first = moments.get(stamps[0]['evidence']) if stamps else None
    g = finding.get('guardrails') or {}
    if any(g.get(k) == 'fail' for k in GUARDRAILS_CHECKED):
        return _no_alternative(finding, first, principle, 'guardrail_failed', [])
    if (finding.get('measurements') or {}).get('failureMode') == 'dm_gap':
        return _no_alternative(finding, first, principle, 'no_zone_for_detector', [])
    scale = (obs.get('timeline') or {}).get('scale') or {}
    if not scale.get('metresUsable') or not (_num(scale.get('lengthM')) and _num(scale.get('widthM'))):
        return _no_alternative(finding, first, principle, 'scale_unknown', [])
    params = _detector_params(profile, det)
    L, W = float(scale['lengthM']), float(scale['widthM'])
    d5 = _detector_params(profile, 'defensive_spacing')
    gap_m = d5.get('lateralGapM') if (profile.get('detectors') or {}).get('defensive_spacing', {}).get('enabled') \
        else None
    gap_thr = G.scaled(gap_m if _num(gap_m) else LATERAL_GAP_M, 'width', scale)
    ctx = {'obs': obs, 'scale': scale, 'L': L, 'W': W, 'params': params,
           'cap': float(params.get('reachSpeedCapMS') or REACH_V_CAP_MS),
           'h': max(GRID_MIN_M, GRID_M*L/G.REF_LENGTH), 'gap_thr': gap_thr, 'principle': principle}
    opps = _index(obs, roles, 'opportunities')
    tried = []
    for ts in stamps:
        m = moments.get(ts['evidence'])
        alt, why = _at_moment(finding, m, opps.get(ts['opportunity']), obs, roles, profile, ctx)
        if alt is not None:
            return alt
        tried.append((m, why))
    if not tried:
        return _no_alternative(finding, None, principle, 'context_missing', [])
    if any(w[0] == 'no_zone_for_detector' for _, w in tried):
        m, w = next((m, w) for m, w in tried if w[0] == 'no_zone_for_detector')
        return _no_alternative(finding, m, principle, 'no_zone_for_detector', [])
    rank = {r_: i for i, r_ in enumerate(REASON_ORDER)}
    m, (reason, rejected) = min(tried, key=lambda mw: rank.get(mw[1][0], 99))
    return _no_alternative(finding, m, principle, reason, rejected)


__all__ = ['propose', 'attack_side', 'convex_hull', 'polygon_area', 'polygon_centroid', 'polygon_diameter',
           'points_in_polygon', 'ZONE_DETECTORS', 'PRINCIPLE', 'CHECK_KEYS', 'STACK_M',
           'ZONE_KEEP_SHARE', 'REACH_H_S', 'GRID_M', 'GRID_MIN_M']
