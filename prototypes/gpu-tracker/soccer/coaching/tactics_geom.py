"""Tactical geometry of the coaching v2 detectors (DESIGN v2 section 4.0): pure functions on positions in a team's
ATTACK FRAME, shared by the detectors (soccer/coaching/contextual.py) and the suggested-zone engine
(soccer/coaching/alternatives.py), so a suggested zone is tested with exactly the relation the detector measured.

Frame. Every function takes metres in the evaluated team's attack frame on the coaching scale
(situations.SituationTimeline X / Y): X from the team's own goal line (0) towards the goal it attacks (L), Y from its
own left touchline (0, looking towards the attacked goal) to its right (W). Its own goal centre is (0, W/2). "Goal-side
of" a point means a smaller X. Unseen players are NaN and never count.

Why these definitions (ported and hardened from the feasibility prototypes, coaching2/feasibility/fe_geom.py; the
numbers are from FEASIBILITY.md, Metrica game 1 full tracking unless said, team A / B):
- first defender: the defending team's outfield player nearest the ball carrier (keepers excluded); camera agreement
  on "pressed at most 5 m" .93-.94 against full tracking; the labellers' "pressed" notes on broadcast measured d1 p50
  2.6 m (83% within 5 m), their "unpressed" notes 9.2 m (8%).
- cover: another outfield defender within R of the first defender and at most A degrees off the line from him to his
  own goal centre (R 12 m x min(1, L/105), A 60 deg: professionals have it on .89 / .76 of pressed moments in their own
  third, .48 / .61 in a high press); "no cover while pressed" agrees with full tracking .91-.95 on the camera.
- weak side: with the ball in an outer third of the width, the defenders goal-side of the ball in the far lateral half
  (professionals have none there on 2.6% / 8.7% of wide-ball samples; camera agreement .995).
- behind the ball / rest defence: own outfield players goal-side of the ball by 10 m x L/105, minus the opponents
  that far on the team's side of the ball (final-third margin p10 / p50 +1 / +2; camera agreement .98-.99).
- wide-defender cover tests: (a) a team-mate in the defender's outer third at least 10 m x L/105 behind him (role-free;
  camera agreement .81-.90), (c) the near centre-back at least 4 m nearer that touchline than his own median, the
  vacated channel (the outer third from 5 m behind the centre-back line up to 10 m behind the defender).
- back-line gaps: the largest lateral gap between adjacent back-line players (professionals out of possession p50
  13.3 m, above 20 m 3.4-5.2% of the time; camera error p50 0.55 m).
- team extent: width = lateral spread, depth = length spread of the observed outfield players (professional width in
  possession p50 47.6 / 50.4 m, out of it 36.0 / 39.0 m).
Every constant is a PITCHIQ-PROPOSAL (origin PITCHIQ-PROPOSAL, unvalidated); no federation gives positional numbers.
Scaled distances are written for a 105 x 68 m pitch and multiplied by L/105 or W/68 of the coaching scale (DESIGN 2.5);
body-scale distances (pressure, cover angle, a centre-back's shift) are absolute metres.

Standard library and numpy only; nothing here reads a result, a timeline or a profile.
"""
import math

import numpy as np

REF_LENGTH = 105.0              # m: the pitch every scaled distance is written for
REF_WIDTH = 68.0
COVER_ANGLE_DEG = 60.0          # deg: cover is at most this far off the first defender's line to his own goal [P]
COVER_RADIUS_M = 12.0           # m (x min(1, L/105), never below COVER_RADIUS_MIN_M): the cover radius [P]
COVER_RADIUS_MIN_M = 6.0
COVER_VIEW_M = 15.0             # m (x min(1, L/105)): the disc around the first defender that must be in view for a
                                # cover reading (the widest radius a profile may ask for; fe_geom used 15 m)
WIDE_SHARE = 1.0/3.0            # the outer third of the width is a wide channel
CENTRAL_BAND_M = 10.0           # m x W/68: "central protection" = goal-side defenders this close to the centre line
ADVANCED_M = 15.0               # m x L/105 beyond the centre-back line: a wide defender is advanced [P]
ADVANCED_HALFWAY_SHARE = .1     # small formats (level 2): beyond halfway by this share of L [P]
CHANNEL_BEHIND_M = 10.0         # m x L/105: a team-mate covering the channel stands at least this far behind [P]
CHANNEL_CB_MARGIN_M = 5.0       # m: the vacated channel starts this far behind the centre-back line (fe_geom)
CB_SHIFT_M = 4.0                # m: the near centre-back's shift towards the touchline that counts as cover [P]
DM_DROP_M = 5.0                 # m: a holding midfielder this close to the centre-back line (or deeper) [P]
BEHIND_BALL_M = 10.0            # m x L/105: goal-side of the ball by at least this [P]
RECOVERING_MS = 3.0             # m/s towards the own goal: a player sprinting back (camera precision .86-.91) [P]
GAP_BEHIND_M = 5.0              # m x L/105: an opponent "inside" a back-line gap stands at most this far behind ...
GAP_AHEAD_M = 15.0              # m x L/105: ... and at most this far ahead of the gap's defenders [P]
LOW_BLOCK_SHARE = .25           # back-line height below this share of L: a low block; below MID_BLOCK_SHARE: mid [P]
MID_BLOCK_SHARE = .40
LEVEL_TOL_MIN_M = 1.0           # m: a defender within max(this, ball sigma + his sigma) of the ball's depth (or of its
                                # distance to the own goal centre) is LEVEL with the ball, neither goal-side nor not
                                # (fix round 1, critic B-5: BC180 Barcelona's back four stood 0.01-2.3 m upfield of a
                                # wide ball at the box edge, ball sigma 0.2-0.5 m, and the depth rule called it "no
                                # defender goal-side") [P]


def _f(x):
    return float(x) if x is not None and math.isfinite(float(x)) else math.nan


def _arr(v):
    return np.asarray(v, float).reshape(-1)


# ---------- scale ----------
def factors(scale):
    """(L / 105, W / 68) of a coaching scale {lengthM, widthM}."""
    return float(scale['lengthM'])/REF_LENGTH, float(scale['widthM'])/REF_WIDTH


def scaled(metres, kind, scale):
    """A distance written for 105 x 68 on this scale: kind 'length' (x L/105), 'width' (x W/68) or None (absolute)."""
    if kind is None:
        return float(metres)
    kl, kw = factors(scale)
    return float(metres)*(kl if kind == 'length' else kw)


def cover_radius(scale, radius=COVER_RADIUS_M, minimum=COVER_RADIUS_MIN_M):
    """DESIGN 4.1: R = radius x min(1, L/105), never below `minimum` (a small field shortens it, a big one never
    stretches it)."""
    kl = float(scale['lengthM'])/REF_LENGTH
    return max(float(minimum), float(radius)*min(1.0, kl))


def zone_of(x, length):
    """'own_third' / 'middle_third' / 'final_third' of a depth X (no hysteresis), None when unknown."""
    x = _f(x)
    if not math.isfinite(x):
        return None
    if x < length/3.0:
        return 'own_third'
    return 'middle_third' if x < 2.0*length/3.0 else 'final_third'


def channel_of(y, width):
    """'left' / 'centre' / 'right' third of a lateral Y (attack frame), None when unknown."""
    y = _f(y)
    if not math.isfinite(y):
        return None
    if y < width*WIDE_SHARE:
        return 'left'
    return 'centre' if y <= width*(1.0-WIDE_SHARE) else 'right'


def in_outer_third(y, side, width):
    """Is a lateral position in the outer third of `side` ('left' / 'right')? (vectorized)"""
    y = np.asarray(y, float)
    with np.errstate(invalid='ignore'):
        return (y <= width*WIDE_SHARE) if side == 'left' else (y >= width*(1.0-WIDE_SHARE))


# ---------- pressure and cover (D1) ----------
def first_defender(cx, cy, X, Y):
    """(index, distance) of the defender nearest the carrier at (cx, cy); (None, nan) when nobody is seen."""
    X, Y = _arr(X), _arr(Y)
    d = np.hypot(X-float(cx), Y-float(cy))
    d[~np.isfinite(d)] = np.inf
    if not len(d) or not np.isfinite(d.min()):
        return None, math.nan
    j = int(np.argmin(d))
    return j, float(d[j])


def angle_at(ax, ay, bx, by, gx, gy):
    """The angle (deg) at A between A->B and A->G; nan when either vector is null."""
    v = np.array([bx-ax, by-ay], float)
    g = np.array([gx-ax, gy-ay], float)
    n = float(np.hypot(*v)*np.hypot(*g))
    if not n or not math.isfinite(n):
        return math.nan
    return math.degrees(math.acos(max(-1.0, min(1.0, float(v @ g)/n))))


def angles_at(ax, ay, X, Y, gx, gy):
    """Vectorized angle_at over the points (X, Y)."""
    X, Y = _arr(X), _arr(Y)
    vx, vy = X-ax, Y-ay
    g = np.array([gx-ax, gy-ay], float)
    with np.errstate(invalid='ignore', divide='ignore'):
        cos = (vx*g[0]+vy*g[1])/(np.hypot(vx, vy)*np.hypot(*g))
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def cover_candidate(fx, fy, X, Y, width, exclude=None, max_angle=COVER_ANGLE_DEG):
    """The cover candidates of a first defender at (fx, fy): (index, distance, angle) of the nearest team-mate at most
    `max_angle` off his line to his own goal centre (0, W/2), and the same for the nearest team-mate at any angle.
    Each is (None, nan, nan) when nobody qualifies; `exclude` = indices never considered (the first defender)."""
    X, Y = _arr(X), _arr(Y)
    dist = np.hypot(X-fx, Y-fy)
    ang = angles_at(fx, fy, X, Y, 0.0, width/2.0)
    ok = np.isfinite(dist)
    for j in (exclude if exclude is not None else ()):
        if j is not None and 0 <= j < len(ok):
            ok[j] = False

    def pick(mask):
        if not mask.any():
            return None, math.nan, math.nan
        k = int(np.argmin(np.where(mask, dist, np.inf)))
        return k, float(dist[k]), float(ang[k])
    with np.errstate(invalid='ignore'):
        within = ok & (ang <= max_angle)
    return pick(within), pick(ok)


def covers(fd, mate, width, radius, max_angle=COVER_ANGLE_DEG):
    """Does a team-mate at `mate` cover a first defender at `fd` (both (X, Y))? Within `radius` and at most
    `max_angle` off his line to his own goal centre: the relation D1 measures and a suggested cover zone must
    restore."""
    d = math.hypot(mate[0]-fd[0], mate[1]-fd[1])
    a = angle_at(fd[0], fd[1], mate[0], mate[1], 0.0, width/2.0)
    return bool(d <= radius and math.isfinite(a) and a <= max_angle)


def closing_speed(px, py, vx, vy, cx, cy):
    """The component of a player's velocity (vx, vy) towards the point (cx, cy), m/s (+ closing); nan when unknown."""
    d = math.hypot(cx-px, cy-py)
    if not d or not all(math.isfinite(v) for v in (vx, vy, d)):
        return math.nan
    return (vx*(cx-px)+vy*(cy-py))/d


def goal_side_share(X, ref_x, exclude=None):
    """Share of the seen players goal-side of depth ref_x (X < ref_x), the excluded indices left out; nan when none."""
    X = _arr(X).copy()
    for j in (exclude if exclude is not None else ()):
        if j is not None and 0 <= j < len(X):
            X[j] = np.nan
    seen = np.isfinite(X)
    if not seen.any():
        return math.nan
    return float((X[seen] < ref_x).mean())


# ---------- weak side (D2) ----------
def weak_side(bx, by, X, Y, width, scale=None, central_m=CENTRAL_BAND_M, side=None):
    """With the ball at (bx, by) in an outer third: {'ballSide', 'weakCount' (defenders goal-side of the ball, X <= bx,
    in the far lateral half), 'openWidthM' (the far touchline's distance from the widest of them; W/2 when none),
    'centralCount' (goal-side within central_m x W/68 of the centre line), 'nearestWeak' (index of the goal-side
    defender nearest the far half, None)}; None when the ball is not wide or not known. side: the ball's channel when
    the caller decided it (the timeline's channel has a 3 m hysteresis), else the plain third of by."""
    side = channel_of(by, width) if side is None else side
    if side not in ('left', 'right') or not math.isfinite(_f(bx)):
        return None
    X, Y = _arr(X), _arr(Y)
    seen = np.isfinite(X) & np.isfinite(Y)
    with np.errstate(invalid='ignore'):
        gs = seen & (X <= bx)
        far = (Y > width/2.0) if side == 'left' else (Y < width/2.0)
        band = central_m*(float(scale['widthM'])/REF_WIDTH if scale else 1.0)
        central = gs & (np.abs(Y-width/2.0) <= band)
    weak = gs & far
    if weak.any():
        open_w = float(width-Y[weak].max()) if side == 'left' else float(Y[weak].min())
    else:
        open_w = width/2.0
    nearest = None
    if gs.any():
        score = np.where(gs, Y if side == 'left' else -Y, -np.inf)
        nearest = int(np.argmax(score))
    return {'ballSide': side, 'weakCount': int(weak.sum()), 'openWidthM': open_w, 'centralCount': int(central.sum()),
            'nearestWeak': nearest}


def goal_side_masks(bx, by, X, Y, width, tol):
    """(goal, level) boolean masks of the seen players against the ball at (bx, by): goal-side = deeper than the ball by
    at least `tol` OR nearer the own goal centre (0, W/2) than the ball by at least `tol` (along the ball-to-goal
    line: a centre-back level with a wide ball at the box edge is between it and the goal); level = not goal-side but
    within `tol` of the ball's depth or of its goal distance (unclear: the measurement cannot tell). tol: metres, one
    value or one per player (max(LEVEL_TOL_MIN_M, ball sigma + the player's sigma))."""
    X, Y = _arr(X), _arr(Y)
    tol = np.broadcast_to(np.asarray(tol, float), X.shape)
    seen = np.isfinite(X) & np.isfinite(Y)
    db = math.hypot(float(bx), float(by)-width/2.0)
    with np.errstate(invalid='ignore'):
        dp = np.hypot(X, Y-width/2.0)
        goal = seen & ((X <= float(bx)-tol) | (dp <= db-tol))
        level = seen & ~goal & ((X <= float(bx)+tol) | (dp <= db+tol))
    return goal, level


def far_side_protection(bx, by, X, Y, width, scale=None, side=None, tol=LEVEL_TOL_MIN_M, vx=None,
                        central_m=CENTRAL_BAND_M, recovering_ms=RECOVERING_MS):
    """D2 with the level tolerance (fix round 1, critic B-5): with the ball at (bx, by) in an outer third, {'weakGoal'
    (defenders goal-side of the ball, goal_side_masks, in the far lateral half), 'weakLevel' (far-half defenders level
    with it), 'centralGoal' (goal-side within central_m x W/68 of the centre line), 'weakRecovering' (far-half
    defenders not yet goal-side running towards the own goal at recovering_ms or more: vx <= -recovering_ms)}; None when
    the ball is not wide or not known. weak_side's counts stay the plain depth rule (X <= bx) they always were."""
    side = channel_of(by, width) if side is None else side
    if side not in ('left', 'right') or not math.isfinite(_f(bx)) or not math.isfinite(_f(by)):
        return None
    X, Y = _arr(X), _arr(Y)
    goal, level = goal_side_masks(bx, by, X, Y, width, tol)
    with np.errstate(invalid='ignore'):
        far = (Y > width/2.0) if side == 'left' else (Y < width/2.0)
        band = central_m*(float(scale['widthM'])/REF_WIDTH if scale else 1.0)
        central = goal & (np.abs(Y-width/2.0) <= band)
    rec = 0
    if vx is not None:
        v = _arr(vx)
        with np.errstate(invalid='ignore'):
            rec = int((far & ~goal & np.isfinite(X) & np.isfinite(v) & (v <= -float(recovering_ms))).sum())
    return {'weakGoal': int((goal & far).sum()), 'weakLevel': int((level & far).sum()),
            'centralGoal': int(central.sum()), 'weakRecovering': rec}


# ---------- behind the ball / rest defence (D4, D3 test d) ----------
def behind_ball(X, bx, margin):
    """(count, mask) of the seen players goal-side of the ball depth bx by at least `margin` metres."""
    X = _arr(X)
    with np.errstate(invalid='ignore'):
        m = np.isfinite(X) & (X <= float(bx)-margin)
    return int(m.sum()), m


def rest_margin(X, OX, bx, margin):
    """(behind, threat, margin): own players goal-side of the ball by `margin`, opponents that far on the team's side
    of the ball (between the ball and the team's goal), and their difference (DESIGN 4.4)."""
    behind, _ = behind_ball(X, bx, margin)
    threat, _ = behind_ball(OX, bx, margin)
    return behind, threat, behind-threat


# ---------- wide defenders (D3) ----------
def vacated_channel(cb_x, wd_x, side, width, behind_m, margin=CHANNEL_CB_MARGIN_M):
    """The channel an advanced wide defender left: (x0, x1, y0, y1) = depth from `margin` behind the centre-back line
    up to `behind_m` behind the defender (never shorter than empty), across his outer third."""
    x0 = max(0.0, float(cb_x)-margin)
    x1 = max(x0, float(wd_x)-behind_m)
    y0, y1 = (0.0, width*WIDE_SHARE) if side == 'left' else (width*(1.0-WIDE_SHARE), width)
    return x0, x1, y0, y1


def channel_occupied(X, Y, wd, side, width, behind_m):
    """Test (a): (occupied, index) of a team-mate (not the defender `wd`) in the defender's outer third at least
    `behind_m` behind him; the index is the one standing highest (nearest behind the defender), None when none."""
    X, Y = _arr(X), _arr(Y)
    if wd is None or not math.isfinite(X[wd]):
        return False, None
    with np.errstate(invalid='ignore'):
        m = np.isfinite(X) & np.isfinite(Y) & in_outer_third(Y, side, width) & (X <= X[wd]-behind_m)
    m[wd] = False
    if not m.any():
        return False, None
    return True, int(np.argmax(np.where(m, X, -np.inf)))


def covers_channel(mate, wd, side, width, behind_m):
    """Does a team-mate at `mate` occupy the channel behind a wide defender at `wd` (test (a), one point)?"""
    return bool(in_outer_third(mate[1], side, width) and mate[0] <= wd[0]-behind_m)


def cb_shift(y, median_y, side):
    """Test (c): metres the near centre-back stands nearer `side`'s touchline than his own median (+ = towards it)."""
    y, m = _f(y), _f(median_y)
    if not (math.isfinite(y) and math.isfinite(m)):
        return math.nan
    return (m-y) if side == 'left' else (y-m)


def opponents_in_channel(OX, OY, wd_x, side, width, length):
    """Test (f): indices of opponents in the defender's outer third, goal-side of him and in the team's own half."""
    OX, OY = _arr(OX), _arr(OY)
    with np.errstate(invalid='ignore'):
        m = np.isfinite(OX) & np.isfinite(OY) & in_outer_third(OY, side, width) & (OX < wd_x) & (OX < length/2.0)
    return [int(j) for j in np.flatnonzero(m)]


# ---------- back line (D5) ----------
def lateral_gaps(Y):
    """The largest lateral gap between adjacent players of a line: (gap, left index, right index, centre Y), or
    (nan, None, None, nan) with fewer than two seen."""
    Y = _arr(Y)
    seen = np.flatnonzero(np.isfinite(Y))
    if len(seen) < 2:
        return math.nan, None, None, math.nan
    order = seen[np.argsort(Y[seen], kind='stable')]
    ys = Y[order]
    d = np.diff(ys)
    k = int(np.argmax(d))
    return float(d[k]), int(order[k]), int(order[k+1]), float((ys[k]+ys[k+1])/2.0)


def opponent_in_gap(OX, OY, left, right, scale):
    """Indices of opponents laterally strictly between the two defenders `left` and `right` ((X, Y) each) and in depth
    from GAP_BEHIND_M behind the deeper of them to GAP_AHEAD_M ahead of the higher (x L/105)."""
    OX, OY = _arr(OX), _arr(OY)
    kl = float(scale['lengthM'])/REF_LENGTH
    lo_y, hi_y = sorted((left[1], right[1]))
    lo_x = min(left[0], right[0])-GAP_BEHIND_M*kl
    hi_x = max(left[0], right[0])+GAP_AHEAD_M*kl
    with np.errstate(invalid='ignore'):
        m = np.isfinite(OX) & np.isfinite(OY) & (OY > lo_y) & (OY < hi_y) & (OX >= lo_x) & (OX <= hi_x)
    return [int(j) for j in np.flatnonzero(m)]


def line_std(X):
    """The population std of the seen depths (nan with fewer than two)."""
    X = _arr(X)
    X = X[np.isfinite(X)]
    return float(X.std()) if len(X) >= 2 else math.nan


def block_height(height, length):
    """'low' / 'mid' / 'high' of a back-line height (m from the own goal line) [P]."""
    h = _f(height)
    if not math.isfinite(h):
        return None
    if h < LOW_BLOCK_SHARE*length:
        return 'low'
    return 'mid' if h < MID_BLOCK_SHARE*length else 'high'


# ---------- team extent (D6) ----------
def team_extent(X, Y):
    """(width, depth) of the seen players: lateral and length spread (nan with fewer than two)."""
    X, Y = _arr(X), _arr(Y)
    seen = np.isfinite(X) & np.isfinite(Y)
    if seen.sum() < 2:
        return math.nan, math.nan
    return float(Y[seen].max()-Y[seen].min()), float(X[seen].max()-X[seen].min())


def channels_occupied(Y, width):
    """How many lateral thirds (left, centre, right) hold at least one seen player."""
    Y = _arr(Y)
    Y = Y[np.isfinite(Y)]
    return len({channel_of(y, width) for y in Y})


def bands_occupied(X, length):
    """How many length thirds (own, middle, final) hold at least one seen player."""
    X = _arr(X)
    X = X[np.isfinite(X)]
    return len({zone_of(x, length) for x in X})


__all__ = ['factors', 'scaled', 'cover_radius', 'zone_of', 'channel_of', 'in_outer_third', 'first_defender',
           'angle_at', 'angles_at', 'cover_candidate', 'covers', 'closing_speed', 'goal_side_share', 'weak_side',
           'goal_side_masks', 'far_side_protection',
           'behind_ball', 'rest_margin', 'vacated_channel', 'channel_occupied', 'covers_channel', 'cb_shift',
           'opponents_in_channel', 'lateral_gaps', 'opponent_in_gap', 'line_std', 'block_height', 'team_extent',
           'channels_occupied', 'bands_occupied']
