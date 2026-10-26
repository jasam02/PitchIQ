"""The coaching series: per team and per sample, the measurements a coaching observation may rest on, and for each
sample whether it may be believed (`eligible`) or why not (`reasons`).

Every number a detector (soccer/coaching/detectors.py) counts comes from here, on TeamShape's grid of real frames (5 a
second, `shape.observed.t`, absolute source seconds; `frame` is the absolute source frame index of the sample's frame).
The positions are the Analytics view's validated OBSERVED positions (soccer/analytics.Samples -> team_shape.TeamShapeData:
registration glitches, sigma above 3 m and duplicate identities already removed), never an estimate and never the 2D
view's stabilized stream; that stream only vetoes evidence (below). A sample that cannot be trusted for a metric is not
a small or a large value: it is ineligible, with the first gate it failed as its reason, so a report can say how long a
metric could not be measured and why. The measurements were checked on the four test runs (broadcast 0-60 s and
42-222 s, the keyless amateur clip, the Veo clip) and on the Metrica Sports professional sample game 1 (audit of
2026-10-18); the numbers below are from there.

Registration. The final registration of the sample's frame (frames[].calibration), never the live
frames[].pitchConfidence (it equals the final one on only 0.8-42% of the frames). A frame that is not calibrated
(confidence below 0.2: no positions are written) or below MEDIUM (0.35, the analytics' own floor) measures nothing, and
neither does one whose forward and backward registration chains disagree by more than 3 m: on broadcast 42-222 s they
disagree by 3-5.2 m at 42.4-51.6 s while the confidence is .37-.47, so all 5561 positions there pass the 0.35 floor
although the projected paint is 80-117 px off (interpolated frames disagreeing by more than 4 m are 101 px off at the
median). A field the result does not have never vetoes: a Metrica-shaped result (tools/team_shape_bench.build_result)
has no calibration, view, ball, profile or stabilized block, and nothing speaks against its samples.

Width and depth: only TeamShape's complete and reliable samples (the whole lineup observed, or 8 observed with every
missing player's estimate inside the extremes). Under a simulated camera on Metrica those are within p50 1.2-1.5 m,
p90 3-4 m of the truth; a partial value is a lower bound, 3-5 m short. Hull area and spread (RMS distance to the
centroid): complete samples only (every outfield player of the lineup observed, nobody extra), because with part of the
team off camera the hull is 368/351 m2 (about 40%) too small.

Units. The formation window containing the sample (the nearest window centre, and the sample inside that 15 s window)
decides the formation (its stabilized label, or your label of the period); the defensive unit D is the one the
defensive line uses (team_shape._defenders_at: the back line by majority over +-30 s of that period); M1 is the
template's line index 1 (the first midfield line), M every middle line, F the last line. A player the window put in the
back line whom the majority did not takes the line of the player who moved into the unit (a swap keeps both lines
whole). A unit counts only with its full membership (the template's line size) and every member observed: on Metrica
game 2 under a simulated 60 m camera the D-M gap of fully seen units is off by p50 0.8 m, p90 2.1-2.6 m, and the
per-sample formation agrees with the full-coverage one .835/.912 of the time, while with one member missing the gap
reads 2.1-2.6 m short with p90 6 m. The D-M gap uses M1, not M: in 4-2-3-1 and 4-1-4-1 the middle group spans two
lines. Gaps are centre to centre (mean depth of one unit minus the other's): the edge-to-edge gap (deepest midfielder
minus most advanced defender) is negative 30-35% of professional time, because full-backs overlap.

Interleaved units (critic B2). A gap is measured only while every member of the deeper unit stands behind the next
unit's mean depth (D against M1, M against F); otherwise the sample is `units_interleaved`. The template splits the
players into lines by their 15 s average, so in a rotating shape its "midfielders" can stand ahead of its
"forwards", and the gap between such units is the template's split, not the team's spacing: on broadcast 0-60 s team
B's midfield-to-front gap read 4.3 m at the median, and on 71% of its samples (203 of 287) a midfielder stood ahead of
the forwards' mean depth (critic: 4.1 m by the template, 8.8 m with the players ranked by depth); the others read
7.6 m. On Metrica game 1 (full tracking) the rule removes 5.0 / 7.3% (team A / B) of the D-M gap's samples out of
possession, where coaching uses that gap, and 18.2 / 5.5% in possession (full-backs push on); of the M-F gap's, 12.5 /
6.6% out of possession and 46.7 / 21.8% in it. The out-of-possession D-M medians move by 0.02 / 0.37 m; the
in-possession M-F medians rise from 9.5 to 12.2 m (A) and from 11.8 to 13.4 m (B). The unit centres (unitD ... unitF)
are unchanged. soccer/coaching/reference.json was built before this rule (its gap distributions keep the interleaved
samples); coaching v2 does not judge against it.

Defensive line: defensive_line()'s complete samples (every defender measured) whose unit was recognised from the
formation (not the role memory or the depth-rank fallback, low confidence by design), neither displaced nor
disrupted. Height error p50 0.5-0.6 m under the simulated camera. The std of the defenders' depth depends on how many
form the line (Metrica team A's back four p50/p90 2.82/7.77 m, team B's back three 2.22/4.94 m), so lineSize travels
with it.

Each defender's offset (`lineOffset`, ported from the stopped fix round 2): on a complete formation back line of three
or more, disrupted ones included (a defender far out of the line is what it measures), never a displaced one, the
distance of each defender's depth from the median depth of the other defenders (`lineOffsetSigned`: + further from
the own goal; `lineRef`: the others' median, metres from the own goal line; `lineRole`: 'wide' for the two outermost
across the pitch, 'central' otherwise). Every other player is `not_in_line`. On Metrica game 1 (full tracking) it is
p50 1.3 / 1.9 m, p90 6.4 / 7.0 m out of possession and p50 5.8 / 3.7 m, p90 14.9 / 10.0 m in possession (team A /
B); under the simulated camera it is off by p50 0.25 m, p90 0.7 / 1.7 m (p99 6-8 m), and "more than 5 m from the
others" agrees with full tracking at precision .96 / .86, recall .96 / .91, on the 44 / 54% of full tracking's
defender-samples the camera measures.

Team centre and third. On a complete sample the outfield centroid; otherwise the window's VOTED centre (each visible
window player's position minus his mean offset from the centre over the window, as team_shape._window_shape computes
it and throws away, with at least 60% of the window's players visible), never the centroid of whoever is visible: on
partial samples (Metrica game 2, 60 m camera) the naive centroid is off by p90 4.15 m across and 11.8 m along the
pitch, the voted centre by 2.17 and 3.76 m; on the amateur clip the visible set changes on 37-44% of the 0.2 s steps and
the naive centroid then jumps p90 1.8-2.7 m against 0.57 m when it does not. The third is the centre's distance from
the team's own goal line: own below L/3 (35 m), final above 2L/3 (70 m), middle between.

Nearest teammate (support, isolation), per observed outfield player: the distance to the nearest other outfield
player of his team, observed teammates and every other person the tracking put in the team's kit with a position
(unconfirmed X-? and anonymous team-labelled people are on the pitch too: broadcast 42-222's A-07 is unconfirmed with
41 s of presence; a person within 0.5 m of an observed teammate is that teammate seen twice). It counts only when it is
CERTAIN: the disc of that radius around the player lies inside the camera's view of the pitch (frames[].view; the part
of the disc beyond the pitch lines does not matter), so no unseen teammate can be nearer. Under a simulated camera on
Metrica that held for 77-79% of the player-samples with an error of p50 -0.2 m, p90 +2.0 m, while the uncertain ones
erred by p90 +4.0 m, p99 +12 m. The rule assumes everyone in view was detected: true on the broadcast (9.99 of 10
observed), not on small-scene runs (the amateur and Veo clips miss players inside their view; on Veo the nearest
observed teammate is beyond 15 m 27-40% of the time, mostly because the teammate is not seen), so it is vetoed on
small-scene runs (labels.small_scene_run), and on a sample where someone in the team's kit is in the picture without a
usable position (he could be the nearest). Where the whole lineup is observed (`complete`) no unseen teammate can be
nearer, so the view does not veto there (critic S8): under the simulated camera on Metrica game 1 that adds 15848 /
14754 player-samples (+10 / +9%), off by p50 0.30 m, p90 0.85 m, p99 1.5-1.7 m against full tracking, no worse than
the disc-certain ones (p50 0.33-0.35, p90 1.05-1.10, p99 4.8-5.3 m); on broadcast 0-60 s team B is measured 593
player-seconds instead of 499, all ten players for at least 10 s (nine before).

Ball context: the final ball path where it is TRACKED or MISSING (a bridged interpolation), its pitch confidence MEDIUM
or better, its larger sigma at most 3 m (p90 13.6 m on broadcast 42-222, through the registration loss) and inside the
pitch with the analytics' 2% margin; vetoed on small-scene runs (18-20% of the amateur clip's shown ball is the end of
a painted arc; Veo places a rolling ball 1-6 m beyond the goal line). Its third (from the team's own goal line) and
channel (thirds of the width, looking towards the goal the team attacks) change only once the ball is 3 m inside the
new zone: 13 of 49 channel changes on broadcast 42-222 reverted within 1 s, and 4-16% of the frames lie within 3 m of a
third line. After an unmeasured second the zone is decided afresh.

Stabilized block: never a measurement. Its flags (a box that lost the legs or reaches into a nearer player, a
down-weighted or rejected foot) mark the players within 0.5 s of a sample whose box foot is wrong (2-3 px off
normally, 15-32 px on those frames), so evidence avoids them (`flagged`).

Velocity (`velX`, `velY` in the attack frame, `speed`; m/s), per observed outfield player: the difference of his own
positions VEL_HALF_WINDOW (0.8 s) either side of the sample over their time apart, central where his usable
positions allow and one-sided at the ends of a stretch (at least 0.4 s apart), never from a frame the registration
gates veto, never across a gap longer than 0.6 s, a change of side, or a jump no player runs (consecutive positions
more than 10 m/s x their time apart + 2 m apart); nothing is interpolated. The window is twice DESIGN 3.1's +-0.4 s
because the camera's position noise dominates the error: on Metrica game 1 under the simulated camera, against
Metrica's own 25 Hz tracking (central difference over +-0.2 s), the speed is off by p50 0.30, p90 0.88 m/s at
+-0.8 s, 0.49 / 1.49 at +-0.4 s and 0.77 / 2.4 at +-0.2 s, while the wider difference's own lag costs p90 0.18 m/s on
full tracking (0.06 at +-0.4 s); "towards the own goal at 3 m/s or more" agrees with Metrica at precision .86 / .91,
recall .86 / .89 (team A / B; at +-0.4 s .73 / .82 and .84 / .87). The noise still inflates speeds, by 0.12 m/s on
average (0.37 at +-0.4 s). The jump rule breaks 3-8 of the camera's 218000-229000 steps per team and 3-7 of full
tracking's 288000 (Metrica's own glitches, up to 27 m/s between samples), none on the broadcast clips, 4 and 2 on
the amateur and Veo clips (team B), where a lone B-07 sighting 11.3 m from the next one 0.6 s later read 16.6 m/s.
Speeds then: Metrica full tracking p99.9 7.7-7.8 m/s, max 9.0; the broadcast clips p99 4.8-6.0, max 6.0-7.9; the
amateur and Veo clips max 4.3-7.1. Identities are taken as given: an identity swap moves a velocity as it moves a
position (the consumers' identity gates apply, DESIGN 3.5).

Data quality (critic B4: the report must say why players were not measured, not only that they were not): per team
and sample, the people in the team's kit in the picture (outfield role, not rejected or a non-target, placed on the
pitch or without a usable position: someone placed beyond the lines is not on the pitch) and those of them with a
confirmed identity; the positioned count is `observed` (the analytics' validated positions). The chain lineup ->
in kit -> confirmed -> positioned names the dominant cause of a short count (soccer/coaching/situations.py). Means
per sample (A / B): broadcast 0-60 s 10.1 / 10.0 in kit, 10.1 / 10.0 confirmed, 10.0 / 10.0 positioned; broadcast
42-222 s 10.1 / 10.0, 9.8 / 10.0, 7.8 / 8.1 (the registration hole); the Veo clip 5.7 / 6.8, 2.9 / 3.5, 2.0 / 2.6.
These counts, the camera's view polygons and the unplaced counts are kept on CoachingData (`in_kit`,
`confirmed_in_kit`, `view`, `unplaced`) for the situation timeline; nothing else changes with them.

The same code runs on a Metrica-shaped result, which is how the professional reference is measured
(tools/coaching_bench.py): a PitchIQ value and its baseline are the same computation.
"""
import bisect
import math
import time

import numpy as np

from .. import analytics, labels, pitch_model, registration, team_shape as ts
from . import episodes as ep

SAMPLE_STEP = 1.0/ts.SAMPLE_RATE        # s: the grid's spacing (each sample stands for one step in the measured time)
REASON_DTYPE = '<U20'

# Registration gates (frames[].calibration, the final registration).
UNCALIBRATED = registration.MIN_POSITION    # .2: below it no pitch position is written (the frame counts as uncalibrated)
MIN_CALIBRATION = pitch_model.MEDIUM        # .35: the analytics' floor (analytics.MIN_PITCH_CONFIDENCE)
MAX_DISAGREEMENT = 3.0                      # m: forward and backward chains further apart than this: no distance claim
                                            # (broadcast 42-222: 3-5.2 m at 42.4-51.6 s, 80-117 px off, all at MEDIUM)

# Units and the team centre.
UNIT_HORIZON = ts.WINDOW_SECONDS/2          # s: the deciding formation window must contain the sample (its centre this close)
CENTRE_MIN_SHARE = .6                       # of the window's players visible for a voted centre (team_shape._window_shape)
THIRDS = 3                                  # the pitch length in thirds from the team's own goal line: own / middle / final
                                            # (L/3 and 2L/3: 35 and 70 m on 105 m)

# Nearest teammate.
SAME_PERSON_METRES = 1.25*ts.DUPLICATE_METRES   # .5 m: another kit-labelled person this close to an observed teammate is him

# Ball context.
BALL_STATES = ('TRACKED', 'MISSING')        # MISSING: the path bridged a short gap (bridged: true); UNKNOWN is no ball
BALL_MIN_CONFIDENCE = pitch_model.MEDIUM    # the ball's own pitch confidence
BALL_MAX_SIGMA = 3.0                        # m: the larger axis of ball.pitchSigma
BALL_OUTSIDE = analytics.OUTSIDE            # .02: a ball this far beyond a line is still on the pitch (as a player's foot)
BALL_HYSTERESIS = 3.0                       # m: the ball must be this far inside a new third or channel to be in it
BALL_RESET_SECONDS = 1.0                    # s: after an unmeasured stretch longer than this the zone is decided afresh
CHANNELS = 3                                # the width in thirds from the team's left touchline: left / centre / right

# Evidence veto from the stabilized block.
FLAG_SECONDS = .5                           # s: a stabilizer flag this close to a sample marks the player there

# Player velocity (module docstring "Velocity"): observed positions differenced across the sample.
VEL_HALF_WINDOW = .8                        # s: the positions this far either side of the sample are differenced (camera
                                            # speed error p90 0.88 m/s; 1.49 at 0.4 s, the window DESIGN 3.1 proposed)
VEL_MAX_GAP = .6                            # s: a velocity never spans a gap in the player's usable positions longer than this
VEL_MIN_BASE = .4                           # s: the shortest time between the two differenced positions (one-sided at a
                                            # stretch's ends; 2 grid steps)
VEL_TOLERANCE = .25*SAMPLE_STEP             # s: grid times are real frame times (0.2002 s apart at 29.97 fps, 0.2083 at 24)
VEL_JUMP_SPEED = 10.0                       # m/s: two consecutive positions further apart than this speed allows (plus
VEL_JUMP_SLACK = 2.0                        # m: this much position noise) are not one player moving: the stretch breaks

# Reason codes, in the order the gates are checked (a sample's reason is the first gate it failed).
NOT_CALIBRATED = 'not_calibrated'           # the frame has no registration (or below UNCALIBRATED)
LOW_CALIBRATION = 'low_calibration'         # the registration is below MEDIUM
DISAGREEMENT = 'disagreement'               # the registration chains disagree by more than MAX_DISAGREEMENT
SMALL_SCENE = 'small_scene_veto'            # nearest teammate and ball context: not on small-scene (amateur, Veo) runs
NO_DIRECTION = 'no_direction'               # the side the team defends is unknown or contradicted there
NO_FORMATION = 'no_formation'               # no classified formation window contains the sample (units unknown)
INCOMPLETE_UNITS = 'incomplete_units'       # a unit misses a member, or a member is not observed
UNITS_INTERLEAVED = 'units_interleaved'     # a member of the deeper unit stands ahead of the next unit's mean depth
INCOMPLETE_TEAM = 'incomplete_team'         # fewer outfield players observed than the lineup (or not reliable)
EXCESS = 'excess_players'                   # more observed outfield players than the lineup (a duplicate, a departed player)
LINE_DISPLACED = 'line_displaced'           # the defensive unit is ahead of most of its team (set piece, all-out attack)
LINE_DISRUPTED = 'line_disrupted'           # the defenders are spread beyond team_shape.LINE_DISRUPTED along the pitch
NOT_OBSERVED = 'not_observed'               # the player has no observed position at the sample
NOT_IN_LINE = 'not_in_line'                 # the player is not one of the back line's measured defenders at the sample
SHORT_TRACK = 'short_track'                 # the player's usable positions around the sample span less than VEL_MIN_BASE
NO_TEAMMATE = 'no_teammate'                 # nobody else of the team has a position
UNPLACED_TEAMMATE = 'unplaced_teammate'     # someone in the team's kit is in the picture without a usable position
OUT_OF_VIEW = 'out_of_view'                 # the disc of the nearest-teammate distance is not all in the camera's view
NO_BALL = 'no_ball'                         # no ball on the path (UNKNOWN, or no ball record)
BALL_UNCERTAIN = 'ball_uncertain'           # the ball's pitch position is below MEDIUM or less certain than BALL_MAX_SIGMA
BALL_OFF_PITCH = 'ball_off_pitch'           # the ball's pitch position is beyond the pitch lines (with the margin)
REASONS = (NOT_CALIBRATED, LOW_CALIBRATION, DISAGREEMENT, SMALL_SCENE, NO_DIRECTION, NO_FORMATION, INCOMPLETE_UNITS,
           UNITS_INTERLEAVED, INCOMPLETE_TEAM, EXCESS, LINE_DISPLACED, LINE_DISRUPTED, NOT_OBSERVED, NOT_IN_LINE, SHORT_TRACK,
           NO_TEAMMATE, UNPLACED_TEAMMATE, OUT_OF_VIEW, NO_BALL, BALL_UNCERTAIN, BALL_OFF_PITCH)

# The metrics with an eligibility mask: per sample (S) and per player (S x N: `nearestTeammate`, `lineOffset`,
# `velocity`).
TEAM_METRICS = ('width', 'depth', 'hull', 'spread', 'centre', 'third', 'unitD', 'unitM1', 'unitM', 'unitF', 'gapDM',
                'gapMF', 'lineStd', 'lineHeight', 'lineSize', 'ball', 'ballThird', 'ballChannel')
PLAYER_METRICS = ('nearestTeammate', 'lineOffset', 'velocity')
# The value arrays and the metric whose eligibility they carry: a value exists only where it may be believed (NaN, or
# '' for a zone, everywhere else), so no detector can count an unmeasured sample by mistake.
VALUE_METRIC = {'width': 'width', 'depth': 'depth', 'hull': 'hull', 'spread': 'spread', 'centreX': 'centre',
                'centreY': 'centre', 'third': 'third', 'DX': 'unitD', 'DY': 'unitD', 'M1X': 'unitM1', 'M1Y': 'unitM1',
                'MX': 'unitM', 'MY': 'unitM', 'FX': 'unitF', 'FY': 'unitF', 'gapDM': 'gapDM', 'gapMF': 'gapMF',
                'lineStd': 'lineStd', 'lineHeight': 'lineHeight', 'lineSize': 'lineSize', 'ballX': 'ball', 'ballY': 'ball',
                'ballThird': 'ballThird', 'ballChannel': 'ballChannel', 'nearestTeammate': 'nearestTeammate',
                'lineOffset': 'lineOffset', 'lineOffsetSigned': 'lineOffset', 'lineRef': 'lineOffset',
                'lineRole': 'lineOffset', 'velX': 'velocity', 'velY': 'velocity', 'speed': 'velocity'}
GOOD_STATUS = ('complete', 'reliable')


def _number(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _gates(shape, gates):
    """Reason codes of a mask of `shape`: '' where every gate passes, else the code of the first gate that failed.
    gates: [(fail mask broadcastable to shape, code), ...] in the order they are checked."""
    reasons = np.full(shape, '', dtype=REASON_DTYPE)
    for fail, code in reversed(gates):
        reasons[np.broadcast_to(np.asarray(fail, bool), shape)] = code
    return reasons


def _hull_area(x, y):
    """Area of the convex hull of the points (monotone chain), 0 for fewer than 3 or collinear points."""
    pts = sorted(set(zip(x.tolist(), y.tolist())))
    if len(pts) < 3:
        return 0.0

    def cross(o, a, b):
        return (a[0]-o[0])*(b[1]-o[1])-(a[1]-o[1])*(b[0]-o[0])
    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    h = lower[:-1]+upper[:-1]
    return abs(sum(h[i][0]*h[(i+1) % len(h)][1]-h[(i+1) % len(h)][0]*h[i][1] for i in range(len(h))))/2.0


def _disc_inside(poly, x, y, r, length, width):
    """Per point: is the disc of radius r around (x, y) (metres) inside the polygon `poly` (K x 2 metres)? Only the
    pitch matters: an edge lying wholly beyond one pitch line (the view polygon clipped a few metres beyond the pitch)
    bounds nothing a teammate could stand on, so a disc may cross it. The point itself must be inside."""
    a, b = poly, np.roll(poly, -1, axis=0)
    beyond = (((a[:, 0] <= 0) & (b[:, 0] <= 0)) | ((a[:, 0] >= length) & (b[:, 0] >= length))
              | ((a[:, 1] <= 0) & (b[:, 1] <= 0)) | ((a[:, 1] >= width) & (b[:, 1] >= width)))
    px, py = np.asarray(x, float)[:, None], np.asarray(y, float)[:, None]
    # Point in polygon (even-odd ray casting over every edge).
    ax, ay, bx, by = a[None, :, 0], a[None, :, 1], b[None, :, 0], b[None, :, 1]
    straddle = (ay > py) != (by > py)
    with np.errstate(divide='ignore', invalid='ignore'):
        xcross = ax+(py-ay)*(bx-ax)/(by-ay)
    inside = (straddle & (px < xcross)).sum(axis=1) % 2 == 1
    keep = ~beyond
    if not keep.any():
        return inside
    ex, ey = (bx-ax)[:, keep], (by-ay)[:, keep]
    L2 = ex**2+ey**2
    with np.errstate(divide='ignore', invalid='ignore'):
        u = np.clip(np.where(L2 > 0, ((px-ax[:, keep])*ex+(py-ay[:, keep])*ey)/L2, 0.0), 0.0, 1.0)
    dist = np.hypot(px-(ax[:, keep]+u*ex), py-(ay[:, keep]+u*ey)).min(axis=1)
    return inside & (dist >= np.asarray(r, float))


def _zones(t, v, ok, edges, names, hysteresis=BALL_HYSTERESIS, reset=BALL_RESET_SECONDS):
    """Zone of a 1-D position per sample with hysteresis: the zone changes only once the value is `hysteresis` metres
    inside the new one (past its boundary with the current zone); after an unmeasured stretch longer than `reset`
    seconds the zone is decided afresh. edges: the two interior boundaries (metres), names: the three zones."""
    out = np.full(len(t), '', dtype='<U6')
    lo, hi = edges

    def plain(x):
        return 0 if x < lo else (2 if x > hi else 1)

    def depth_inside(x, z):
        return lo-x if z == 0 else (x-hi if z == 2 else min(x-lo, hi-x))
    state, last = None, None
    for s in np.flatnonzero(ok):
        x = float(v[s])
        if state is None or t[s]-last > reset+1e-9:
            state = plain(x)
        else:
            z = plain(x)
            if z != state and depth_inside(x, z) >= hysteresis:
                state = z
        out[s] = names[state]
        last = t[s]
    return out


def velocities(t, X, Y, usable, side, half=VEL_HALF_WINDOW, max_gap=VEL_MAX_GAP, min_base=VEL_MIN_BASE,
               tolerance=VEL_TOLERANCE, jump_speed=VEL_JUMP_SPEED, jump_slack=VEL_JUMP_SLACK):
    """Per sample and player (S x N): the velocity (m/s) from the player's own positions (module docstring
    "Velocity"). X, Y: S x N metres; usable: S x N bool, the positions that may be differenced (NaN elsewhere is
    fine); side: S, the frame's orientation (a change splits a stretch: in the attack frame the side flips the axes).
    A player's usable positions form stretches broken by any gap longer than max_gap seconds, by a side change and by
    a jump (two consecutive positions further apart than jump_speed * their time apart + jump_slack metres). At a
    usable sample the velocity is (P(b) - P(a)) / (t(b) - t(a)), a and b the furthest usable positions of the same
    stretch within half seconds before and after it (central where the stretch allows, one-sided at its ends), when
    t(b) - t(a) is at least min_base; NaN elsewhere. Nothing is interpolated or smoothed beyond that difference.
    Returns (vx, vy)."""
    t = np.asarray(t, float)
    X, Y = np.asarray(X, float), np.asarray(Y, float)
    usable = np.asarray(usable, bool)
    side = np.asarray(side)
    S = len(t)
    N = usable.shape[1] if usable.ndim == 2 else 0
    vx, vy = np.full((S, N), np.nan), np.full((S, N), np.nan)
    for j in range(N):
        u = np.flatnonzero(usable[:, j] & np.isfinite(X[:, j]) & np.isfinite(Y[:, j]))
        if len(u) < 2:
            continue
        tu = t[u]
        brk = np.ones(len(u), bool)
        dt = np.diff(tu)
        jump = np.hypot(np.diff(X[u, j]), np.diff(Y[u, j])) > jump_speed*dt+jump_slack
        brk[1:] = (dt > max_gap+tolerance) | (side[u][1:] != side[u][:-1]) | jump
        seg = np.cumsum(brk)-1
        first = np.flatnonzero(brk)
        last = np.r_[first[1:]-1, len(u)-1]
        a = np.maximum(np.searchsorted(tu, tu-half-tolerance, side='left'), first[seg])
        b = np.minimum(np.searchsorted(tu, tu+half+tolerance, side='right')-1, last[seg])
        base = tu[b]-tu[a]
        ok = base >= min_base-tolerance
        if not ok.any():
            continue
        rows, ia, ib = u[ok], u[a[ok]], u[b[ok]]
        vx[rows, j] = (X[ib, j]-X[ia, j])/base[ok]
        vy[rows, j] = (Y[ib, j]-Y[ia, j])/base[ok]
    return vx, vy


class TeamSeries:
    """One team's coaching series on the common grid. Arrays (length S = samples, or S x N for the N outfield
    identities in `ids` order): `values[name]` (also attributes), `eligible[metric]` (bool masks), `reasons[metric]`
    (codes, '' where eligible). Per sample: `units[s]` {'D', 'M1', 'M', 'F': [ids], 'formation', 'period' (index in
    `periods`), 'confidence' (the period's: clear / likely / ambiguous), 'full' {unit: membership complete}, 'lines'
    [[ids] per template line], 'source'} or None (shared between samples: read only); `positions(s)`,
    `goalkeeper(s)`, `flagged_ids(s)` for diagrams and evidence.

    Attributes t, frame, segment_t (S) and ids, col. Values (S): side, directionStatus, observed, expected, complete;
    width, depth (+Status, From, To: the extreme players), geometric, identityMin; hull, spread; centreX, centreY
    (attack frame metres: X from the own goal line, Y from the team's left touchline), centreSource ('complete',
    'voted'), centreWindow, third; formation, formationPeriod, formationConfidence, unitDefined; DX, DY, M1X, M1Y, MX,
    MY, FX, FY (unit centres, attack frame), gapDM, gapMF; lineStd, lineHeight, lineSize, lineStatus, lineSource,
    lineQuality, lineDefenders; ballX, ballY (attack frame metres), ballThird, ballChannel, ballObserved. Values
    (S x N): x, y (normalized pitch), kind ('O' or ''), group ('D', 'M', 'F'), line (template line index, -1),
    nearestTeammate (metres), nearestExtras (S: kit-labelled people counted beside the observed teammates), flagged;
    lineOffset, lineOffsetSigned, lineRef (metres), lineRole ('wide', 'central'); velX, velY (attack frame, m/s:
    +X towards the goal the team attacks, +Y towards its right touchline), speed (m/s); gk (S x 2). Every value in
    VALUE_METRIC is NaN ('' for a zone or a role) wherever its metric is not eligible."""

    def __init__(self, team, t, frame, step, ids, segment_start):
        self.team = team
        self.t = t
        self.frame = frame
        self.step = step
        self.ids = list(ids)
        self.col = {pid: j for j, pid in enumerate(self.ids)}
        self.segment_t = t-segment_start
        self.values = {}
        self.eligible = {}
        self.reasons = {}
        self.units = []
        self.windows, self.labels, self.periods, self.window_roles = [], [], [], []
        self.timings = {}
        self._data = None

    def _put(self, **kw):
        for k, v in kw.items():
            self.values[k] = v
            setattr(self, k, v)

    def _metric(self, name, reasons):
        self.reasons[name] = reasons
        self.eligible[name] = reasons == ''

    def mask(self, metric, player=None):
        """The eligible samples of a metric (bool, length S). For a per-player metric, of that player, or without a
        player the samples where at least one player is eligible."""
        e = self.eligible[metric]
        if e.ndim == 2:
            if player is None:
                return e.any(axis=1) if e.shape[1] else np.zeros(e.shape[0], bool)
            j = self.col.get(player)
            return e[:, j] if j is not None else np.zeros(e.shape[0], bool)
        return e

    def coverage(self, metric, player=None):
        """{measuredSeconds, eligibleShare, longestStretchSeconds} of a metric (soccer/coaching/episodes: each eligible
        sample stands for one sample step)."""
        m = self.mask(metric, player)
        n = len(m)
        return {'measuredSeconds': round(ep.measured_seconds(self.t, m, self.step), 2),
                'eligibleShare': round(float(np.count_nonzero(m))/n, 3) if n else 0.0,
                'longestStretchSeconds': round(ep.longest_stretch(self.t, m, self.step), 2)}

    def reason_seconds(self, metric, player=None):
        """Seconds per reason code where the metric was not measured ({code: seconds}; per-player metrics without a
        player: player-seconds over all players)."""
        r = self.reasons[metric]
        if r.ndim == 2:
            if player is not None:
                j = self.col.get(player)
                r = r[:, j] if j is not None else np.zeros(0, dtype=REASON_DTYPE)
            else:
                r = r.ravel()
        codes, counts = np.unique(r[r != ''], return_counts=True)
        return {str(c): round(float(k)*self.step, 2) for c, k in zip(codes, counts)}

    def positions(self, s):
        """{id: (x, y, 'O')}: the team's observed outfield positions at sample s (normalized pitch coordinates), as the
        measurements used them. Estimates never appear here."""
        x, y, kind = self.values['x'][s], self.values['y'][s], self.values['kind'][s]
        return {pid: (float(x[j]), float(y[j]), 'O') for j, pid in enumerate(self.ids) if kind[j] == 'O' and np.isfinite(x[j])}

    def goalkeeper(self, s):
        g = self.values['gk'][s]
        return (float(g[0]), float(g[1])) if np.isfinite(g[0]) else None

    def flagged_ids(self, s):
        """Players the stabilizer flagged within FLAG_SECONDS of sample s (keep them out of evidence)."""
        f = self.values['flagged'][s]
        return {pid for j, pid in enumerate(self.ids) if f[j]}

    def registration(self, mask=None):
        """How certain the registration behind the team's observed positions is over the masked samples (the HIGH /
        MEDIUM shares of team_shape.registration_quality)."""
        rows = np.flatnonzero(mask) if mask is not None else np.arange(len(self.t))
        return ts.registration_quality(self._data, self.team, rows)

    def summary(self):
        """JSON-safe coverage of every metric: {metric: {measuredSeconds, eligibleShare, longestStretchSeconds,
        notMeasured: {reason: seconds}}}."""
        return {m: dict(self.coverage(m), notMeasured=self.reason_seconds(m)) for m in TEAM_METRICS+PLAYER_METRICS}


class CoachingData:
    """The coaching series of one result: `teams` {'A': TeamSeries, 'B': TeamSeries} on the common grid (`t`,
    `frame`), the run-level facts (`pitch`, `metresEstimated`, `smallScene`, `available` fields), the per-sample frame
    context shared by both teams (`calibration`, `ball`, `view`; per team `unplaced`, `in_kit`, `confirmed_in_kit`),
    the TeamShape and Samples it was built from, and timings."""

    def __init__(self):
        self.teams = {}
        self.t = np.zeros(0)
        self.frame = np.zeros(0, np.int64)
        self.step = SAMPLE_STEP
        self.fps = 0.0
        self.start = self.end = 0.0
        self.pitch = {}
        self.metres_estimated = True
        self.small_scene = False
        self.available = {}
        self.calibration = {}
        self.ball = {}
        self.view = []              # per sample: the camera's view polygon (metres, K x 2), None (no such field), False
        self.unplaced = {}          # team -> (S,) people in the team's kit in the picture without a usable position
        self.in_kit = {}            # team -> (S,) people in the team's kit in the picture (data quality)
        self.confirmed_in_kit = {}  # team -> (S,) ... of them with a confirmed identity (data quality)
        self.shape = None
        self.samples = None
        self.result = None
        self.timings = {}

    def sample_at(self, t):
        """The sample nearest to time t (None without samples)."""
        if not len(self.t) or not _number(t):
            return None
        return int(np.argmin(np.abs(self.t-float(t))))

    def seek(self, s):
        """The browser seek time that shows sample s's frame exactly: (frame + 0.5) / fps."""
        return (float(self.frame[s])+.5)/self.fps if self.fps else float(self.t[s])

    def summary(self):
        """JSON-safe: the run facts and every team's coverage per metric."""
        return {'samples': int(len(self.t)), 'step': self.step, 'range': {'t0': ep.finite_or_none(self.start, 3), 't1': ep.finite_or_none(self.end, 3)},
                'pitch': self.pitch, 'metresEstimated': bool(self.metres_estimated), 'smallScene': bool(self.small_scene),
                'available': dict(self.available), 'teams': {team: s.summary() for team, s in self.teams.items()},
                'timings': {k: round(float(v), 3) for k, v in self.timings.items()}}


# ---------- the frame context shared by both teams ----------
def _frame_context(result, data):
    frames = result.get('frames') or []
    F = [frames[i] for i in data.frame_index]
    S = len(F)
    fps = float((result.get('video') or {}).get('fps') or 0.0) or float(data.fps)
    ctx = {'frames': F, 'fps': fps}
    ctx['frame'] = np.array([int(f['frame']) if _number(f.get('frame')) else int(round(float(f['time'])*fps)) for f in F], np.int64)
    has_cal = np.zeros(S, bool)
    conf, dis = np.full(S, np.nan), np.full(S, np.nan)
    calibrated = np.ones(S, bool)
    for s, f in enumerate(F):
        if f.get('calibrated') is False:
            calibrated[s] = False
        c = f.get('calibration')
        if isinstance(c, dict):
            has_cal[s] = True
            if _number(c.get('confidence')):
                conf[s] = float(c['confidence'])
            if _number(c.get('disagreement')):
                dis[s] = float(c['disagreement'])
    with np.errstate(invalid='ignore'):
        not_cal = ~calibrated | (has_cal & ~(conf >= UNCALIBRATED))
        low = has_cal & ~not_cal & (conf < MIN_CALIBRATION)
        disagree = np.isfinite(dis) & (dis > MAX_DISAGREEMENT)
    ctx['calibration'] = {'present': has_cal, 'confidence': conf, 'disagreement': dis, 'calibrated': ~not_cal,
                          'gates': [(not_cal, NOT_CALIBRATED), (low, LOW_CALIBRATION), (disagree, DISAGREEMENT)]}
    # The camera's view of the pitch: None where the frame has no such field (no veto), False where it has no usable
    # polygon (uncalibrated), else the polygon in metres.
    L, W = data.length, data.width
    views = []
    for f in F:
        if 'view' not in f:
            views.append(None)
            continue
        v = f.get('view')
        try:
            P = np.array(v, np.float64)
        except (TypeError, ValueError):
            P = None
        if P is None or P.ndim != 2 or P.shape[0] < 3 or P.shape[1] != 2 or not np.isfinite(P).all():
            views.append(False)
        else:
            views.append(P*np.array([L, W]))
    ctx['view'] = views
    # Ball (team independent): normalized position and the first ball gate failed.
    bx, by = np.full(S, np.nan), np.full(S, np.nan)
    state = np.full(S, '', dtype='<U8')
    observed = np.zeros(S, bool)
    breason = np.full(S, '', dtype=REASON_DTYPE)
    for s, f in enumerate(F):
        b = f.get('ball')
        if not isinstance(b, dict) or b.get('state') not in BALL_STATES:
            breason[s] = NO_BALL
            state[s] = str(b.get('state') or '')[:8] if isinstance(b, dict) else ''
            continue
        state[s] = b['state']
        observed[s] = b['state'] == 'TRACKED' and b.get('origin', 'OBSERVATION') == 'OBSERVATION'
        p = b.get('pitch')
        if not (isinstance(p, (list, tuple)) and len(p) == 2 and all(_number(v) for v in p)):
            breason[s] = BALL_UNCERTAIN
            continue
        bx[s], by[s] = float(p[0]), float(p[1])
        conf_b = b.get('pitchConfidence')
        sig = b.get('pitchSigma')
        sig = max(sig) if isinstance(sig, (list, tuple)) and sig and all(_number(v) for v in sig) else (sig if _number(sig) else None)
        if (_number(conf_b) and conf_b < BALL_MIN_CONFIDENCE) or (sig is not None and sig > BALL_MAX_SIGMA):
            breason[s] = BALL_UNCERTAIN
        elif not (-BALL_OUTSIDE <= bx[s] <= 1+BALL_OUTSIDE and -BALL_OUTSIDE <= by[s] <= 1+BALL_OUTSIDE):
            breason[s] = BALL_OFF_PITCH
    ctx['ball'] = {'x': bx, 'y': by, 'state': state, 'observed': observed, 'reason': breason,
                   'present': any(isinstance(f.get('ball'), dict) for f in F)}
    # Everyone in a team's kit on the frame (outfield role): placed (metres) or not placed. Goalkeepers and people
    # rejected or not a target are left out; someone placed beyond the pitch (with the margin) is not on it. The same
    # pass counts the data-quality block (module docstring "Data quality"): the people in kit in the picture (placed
    # on the pitch, or without a usable position) and those of them with a confirmed identity.
    people = {'A': [[] for _ in range(S)], 'B': [[] for _ in range(S)]}
    unplaced = {'A': np.zeros(S, int), 'B': np.zeros(S, int)}
    in_kit = {'A': np.zeros(S, int), 'B': np.zeros(S, int)}
    confirmed = {'A': np.zeros(S, int), 'B': np.zeros(S, int)}
    for s, f in enumerate(F):
        for p in f.get('people') or ():
            team = p.get('team')
            if team not in people or p.get('role') != 'PLAYER' or p.get('state') in ('rejected', 'nontarget'):
                continue
            pos = p.get('pitch')
            pc = p.get('pitchConfidence')
            placed = isinstance(pos, (list, tuple)) and len(pos) == 2 and all(_number(v) for v in pos)
            if not placed or analytics._valid(pos):
                in_kit[team][s] += 1
                if p.get('state') == 'confirmed' and p.get('id'):
                    confirmed[team][s] += 1
            if not placed or (_number(pc) and pc < analytics.MIN_PITCH_CONFIDENCE):
                unplaced[team][s] += 1
            elif analytics._valid(pos):
                people[team][s].append((p.get('id'), float(pos[0])*L, float(pos[1])*W))
    ctx['people'], ctx['unplaced'] = people, unplaced
    ctx['quality'] = {'inKit': in_kit, 'confirmed': confirmed}
    # Stabilizer flags: per identity the flagged times.
    st = result.get('stabilized') if isinstance(result.get('stabilized'), dict) else None
    flags = {}
    for fl in (st or {}).get('flags') or ():
        if isinstance(fl, dict) and isinstance(fl.get('id'), str) and _number(fl.get('time')):
            flags.setdefault(fl['id'], []).append(float(fl['time']))
    ctx['flags'] = {pid: np.sort(np.array(v)) for pid, v in flags.items()}
    ctx['stabilized'] = st is not None
    return ctx


# ---------- units ----------
def _unit_groups(t, window_roles, centres, j0, line_of):
    """The units at time t from the deciding window j0 (its fitted template lines, line_of {id: line}) and the
    defensive unit by majority (team_shape._defenders_at); see the module docstring for the swap rule."""
    unit, size, source = ts._defenders_at(t, window_roles, centres, memory=ts.ROLE_HORIZON)
    _, decides, _, _, period = window_roles[j0]
    tpl = ts.TEMPLATES[decides]
    n = len(tpl['lines'])
    final = {pid: 0 for pid in unit}
    moved = sorted(line_of[pid] for pid in unit if line_of.get(pid, 0) != 0)       # into the unit from another line
    displaced = sorted(pid for pid, line in line_of.items() if line == 0 and pid not in unit)
    for pid, line in line_of.items():
        if pid not in unit and line != 0:
            final[pid] = line
    for pid, line in zip(displaced, moved):
        final[pid] = line
    by = [sorted(pid for pid, line in final.items() if line == k) for k in range(n)]
    full = [len(by[k]) == tpl['lines'][k] for k in range(n)]
    full[0] = full[0] and size == tpl['lines'][0]
    middle = sorted(pid for k in range(1, n-1) for pid in by[k])
    return {'D': by[0], 'M1': by[1], 'M': middle, 'F': by[n-1], 'formation': decides, 'period': int(period),
            'full': {'D': bool(full[0]), 'M1': bool(full[1]), 'M': bool(all(full[1:n-1])), 'F': bool(full[n-1])},
            'lines': by, 'source': source}


# ---------- one team ----------
def _team_series(team, shape, ctx, small_scene, segment_start):
    clock = time.perf_counter
    a0 = clock()
    data = shape.observed
    d = data.teams[team]
    L, W = data.length, data.width
    t = data.t
    S = len(t)
    ids = list(d['ids'])
    N = len(ids)
    out = TeamSeries(team, t, ctx['frame'], SAMPLE_STEP, ids, segment_start)
    out._data = data
    F_ = shape.shapes[team]
    out.windows, out.labels, out.periods, out.window_roles = F_['windows'], F_['labels'], F_['periods'], F_['windowRoles']
    side = np.asarray(shape.directions[team]['side']).astype('<U5') if S else np.zeros(0, '<U5')
    known = (side == 'left') | (side == 'right')
    x, y = d['x'].reshape(S, N), d['y'].reshape(S, N)
    obs = (d['kind'].reshape(S, N) == 'O') & np.isfinite(x) & np.isfinite(y)
    nobs = obs.sum(axis=1)
    expected = np.asarray(d['expected_t'], int).reshape(S)
    px, py = np.where(obs, x*L, np.nan), np.where(obs, y*W, np.nan)
    X, Y = ts.attack_frame(x, y, side[:, None])
    Xm = np.where(obs & known[:, None], X*L, np.nan)
    Ym = np.where(obs & known[:, None], Y*W, np.nan)
    cal = ctx['calibration']['gates']
    no_dir = ~known
    complete = (nobs == expected) & (nobs >= 3)
    excess = nobs > expected
    out._put(x=np.where(obs, x, np.nan), y=np.where(obs, y, np.nan), kind=np.where(obs, 'O', '').astype('<U1'), gk=d['gk'].copy(), side=side,
             directionStatus=np.asarray(shape.directions[team]['status']).astype('<U12') if S else np.zeros(0, '<U12'),
             observed=nobs, expected=expected, complete=complete)

    # Width and depth (team_shape.width_depth_series: complete or reliable only).
    a = clock()
    wd = ts.width_depth_series(data, team, data.start, data.end) if S else []
    if len(wd) != S:
        raise RuntimeError('width_depth_series does not cover the grid')
    vals = {}
    for key in ('width', 'depth'):
        v = np.full(S, np.nan)
        status = np.array([r.get(key+'Status') or 'unavailable' for r in wd], dtype='<U12') if S else np.zeros(0, '<U12')
        frm = np.full(S, '', dtype=object)
        to = np.full(S, '', dtype=object)
        for s, r in enumerate(wd):
            m = r.get(key)
            if m:
                frm[s], to[s] = m['from'], m['to']
                if status[s] in GOOD_STATUS:
                    v[s] = m['metres']
        vals[key], vals[key+'Status'], vals[key+'From'], vals[key+'To'] = v, status, frm, to
        out._metric(key, _gates(S, cal+[(status == 'excess', EXCESS), (~np.isin(status, GOOD_STATUS), INCOMPLETE_TEAM)]))
    geometric = np.array([r['geometric'] if _number(r.get('geometric')) else np.nan for r in wd], float) if S else np.zeros(0)
    identity = np.array([r['identity'] if _number(r.get('identity')) else np.nan for r in wd], float) if S else np.zeros(0)
    out._put(geometric=geometric, identityMin=identity, **vals)

    # Hull area and spread (complete samples).
    hull, spread = np.full(S, np.nan), np.full(S, np.nan)
    for s in np.flatnonzero(complete):
        hull[s] = _hull_area(px[s, obs[s]], py[s, obs[s]])
    if N:
        n = np.maximum(nobs, 1)
        cx, cy = np.where(obs, px, 0.0).sum(axis=1)/n, np.where(obs, py, 0.0).sum(axis=1)/n
        sp = np.sqrt(np.where(obs, (px-cx[:, None])**2+(py-cy[:, None])**2, 0.0).sum(axis=1)/n)
        spread = np.where(complete, sp, np.nan)
    team_gates = cal+[(excess, EXCESS), (~complete, INCOMPLETE_TEAM)]
    out._metric('hull', _gates(S, team_gates))
    out._metric('spread', _gates(S, team_gates))
    out._put(hull=hull, spread=spread)
    out.timings['shape'] = clock()-a

    # Team centre (attack frame metres): the centroid on complete samples, else the window's voted centre.
    a = clock()
    centre_x, centre_y = np.full(S, np.nan), np.full(S, np.nan)
    centre_src = np.full(S, '', dtype='<U8')
    if N:
        full_team = complete & known
        centre_x[full_team] = np.where(obs, Xm, 0.0)[full_team].sum(axis=1)/nobs[full_team]
        centre_y[full_team] = np.where(obs, Ym, 0.0)[full_team].sum(axis=1)/nobs[full_team]
        centre_src[full_team] = 'complete'
    windows = F_['windows']
    best, win_of = np.full(S, np.inf), np.full(S, -1)
    for i, w in enumerate(windows):
        rows = w.get('_rows')
        if w.get('status') != 'ok' or '_P' not in w or rows is None or not len(rows):
            continue
        dist = np.abs(t[rows]-float(w['centre']))
        upd = dist < best[rows]
        best[rows[upd]] = dist[upd]
        win_of[rows[upd]] = i
    col = out.col
    for i in np.unique(win_of[win_of >= 0]):
        w = windows[i]
        rows = np.flatnonzero((win_of == i) & ~complete & known)
        if not len(rows):
            continue
        wc = np.array([col[pid] for pid in w['ids']], int)
        P = np.asarray(w['_P'], float)
        vis = obs[np.ix_(rows, wc)]
        cnt = vis.sum(axis=1)
        enough = cnt >= max(2, int(math.ceil(CENTRE_MIN_SHARE*len(wc))))
        with np.errstate(all='ignore'):
            vx = np.where(vis, Xm[np.ix_(rows, wc)]-P[None, :, 0], 0.0).sum(axis=1)/np.maximum(cnt, 1)
            vy = np.where(vis, Ym[np.ix_(rows, wc)]-P[None, :, 1], 0.0).sum(axis=1)/np.maximum(cnt, 1)
        r = rows[enough]
        centre_x[r], centre_y[r] = vx[enough], vy[enough]
        centre_src[r] = 'voted'
    third = np.full(S, '', dtype='<U6')
    fin = np.isfinite(centre_x)
    third[fin] = 'middle'
    third[fin & (centre_x < L/THIRDS)] = 'own'
    third[fin & (centre_x > 2*L/THIRDS)] = 'final'
    centre_reason = _gates(S, cal+[(no_dir, NO_DIRECTION), (~fin, INCOMPLETE_TEAM)])
    out._metric('centre', centre_reason)
    out._metric('third', centre_reason.copy())
    out._put(centreX=centre_x, centreY=centre_y, centreSource=centre_src, third=third, centreWindow=win_of)
    out.timings['centre'] = clock()-a

    # Units per sample.
    a = clock()
    wr = F_['windowRoles'] or []
    centres = [float(e[0]) for e in wr]
    by_centre = {w['centre']: i for i, w in enumerate(windows) if w.get('status') == 'ok'}
    line_of, spans = [], []
    for c, decides, _, _, _ in wr:
        w = windows[by_centre[c]]
        tpl = ts.TEMPLATES[decides]
        line_of.append({pid: int(tpl['line'][r]) for pid, r in zip(w['ids'], w['_fits'][decides]['roles'])})
        rows = w['_rows']
        spans.append((float(t[rows[0]]), float(t[rows[-1]])) if len(rows) else (np.inf, -np.inf))
    group = np.full((S, N), '', dtype='<U1')
    line = np.full((S, N), -1, dtype=np.int8)
    mem = {u: np.zeros((S, N), bool) for u in ('D', 'M1', 'M', 'F')}
    full = {u: np.zeros(S, bool) for u in ('D', 'M1', 'M', 'F')}
    defined = np.zeros(S, bool)
    formation = np.full(S, '', dtype='<U8')
    period = np.full(S, -1, dtype=np.int32)
    fconf = np.full(S, '', dtype='<U10')
    units = [None]*S
    cache = {}
    nwr = len(wr)
    for s in range(S):
        if not nwr:
            break
        tt = float(t[s])
        k = bisect.bisect_left(centres, tt)
        j0 = min((j for j in (k-1, k) if 0 <= j < nwr), key=lambda j: abs(centres[j]-tt))    # as _defenders_at
        lo_t, hi_t = spans[j0]
        if not (lo_t-1e-6 <= tt <= hi_t+1e-6) or abs(centres[j0]-tt) > UNIT_HORIZON+1e-6:
            continue
        key = (j0, bisect.bisect_left(centres, tt-ts.ROLE_HORIZON), bisect.bisect_right(centres, tt+ts.ROLE_HORIZON))
        hit = cache.get(key)
        if hit is None:
            u = _unit_groups(tt, wr, centres, j0, line_of[j0])
            unit_cols = {name: np.array([col[p] for p in u[name] if p in col], int) for name in ('D', 'M1', 'M', 'F')}
            line_cols = [np.array([col[p] for p in members if p in col], int) for members in u['lines']]
            p = F_['periods'][u['period']] if 0 <= u['period'] < len(F_['periods']) else {}
            u['confidence'] = p.get('confidence')
            hit = cache[key] = (u, unit_cols, line_cols)
        u, unit_cols, line_cols = hit
        units[s] = u
        defined[s] = True
        for name in ('D', 'M1', 'M', 'F'):
            mem[name][s, unit_cols[name]] = True
            full[name][s] = u['full'][name]
        for kk, cols_k in enumerate(line_cols):
            line[s, cols_k] = kk
            group[s, cols_k] = 'D' if kk == 0 else ('F' if kk == len(line_cols)-1 else 'M')
        formation[s] = u['formation']
        period[s] = u['period']
        fconf[s] = u['confidence'] or ''
    centre_of = {}
    unit_ok = {}
    for name in ('D', 'M1', 'M', 'F'):
        m = mem[name]
        cnt = m.sum(axis=1)
        all_seen = ~(m & ~obs).any(axis=1) if N else np.ones(S, bool)
        ok = defined & full[name] & (cnt > 0) & all_seen & known
        with np.errstate(all='ignore'):
            ux = np.where(ok, np.where(m, Xm, 0.0).sum(axis=1)/np.maximum(cnt, 1), np.nan)
            uy = np.where(ok, np.where(m, Ym, 0.0).sum(axis=1)/np.maximum(cnt, 1), np.nan)
        centre_of[name] = (ux, uy)
        unit_ok[name] = ok
        out._metric('unit'+name, _gates(S, cal+[(no_dir, NO_DIRECTION), (~defined, NO_FORMATION), (~ok, INCOMPLETE_UNITS)]))
    gap_dm = np.where(unit_ok['D'] & unit_ok['M1'], centre_of['M1'][0]-centre_of['D'][0], np.nan)
    gap_mf = np.where(unit_ok['M'] & unit_ok['F'], centre_of['F'][0]-centre_of['M'][0], np.nan)
    # Interleaved units (module docstring "Interleaved units"): a member of the deeper unit ahead of the next unit's
    # mean depth.
    with np.errstate(invalid='ignore'):
        front_d = np.where(mem['D'] & obs, Xm, -np.inf).max(axis=1) if N else np.full(S, -np.inf)
        front_m = np.where(mem['M'] & obs, Xm, -np.inf).max(axis=1) if N else np.full(S, -np.inf)
        mixed_dm = unit_ok['D'] & unit_ok['M1'] & (front_d > centre_of['M1'][0])
        mixed_mf = unit_ok['M'] & unit_ok['F'] & (front_m > centre_of['F'][0])
    out._metric('gapDM', _gates(S, cal+[(no_dir, NO_DIRECTION), (~defined, NO_FORMATION), (~(unit_ok['D'] & unit_ok['M1']), INCOMPLETE_UNITS),
                                        (mixed_dm, UNITS_INTERLEAVED)]))
    out._metric('gapMF', _gates(S, cal+[(no_dir, NO_DIRECTION), (~defined, NO_FORMATION), (~(unit_ok['M'] & unit_ok['F']), INCOMPLETE_UNITS),
                                        (mixed_mf, UNITS_INTERLEAVED)]))
    out.units = units
    out._put(group=group, line=line, formation=formation, formationPeriod=period, formationConfidence=fconf,
             unitDefined=defined, gapDM=gap_dm, gapMF=gap_mf,
             DX=centre_of['D'][0], DY=centre_of['D'][1], M1X=centre_of['M1'][0], M1Y=centre_of['M1'][1],
             MX=centre_of['M'][0], MY=centre_of['M'][1], FX=centre_of['F'][0], FY=centre_of['F'][1])
    out.timings['units'] = clock()-a

    # Defensive line (team_shape.defensive_line).
    a = clock()
    rows = ts.defensive_line(data, team, shape.directions, F_, data.start, data.end) if S else []
    if len(rows) != S:
        raise RuntimeError('defensive_line does not cover the grid')
    lstd, lheight, lsize = np.full(S, np.nan), np.full(S, np.nan), np.full(S, np.nan)
    lstatus = np.array([r.get('status') or '' for r in rows], dtype='<U12') if S else np.zeros(0, '<U12')
    lsource = np.array([r.get('source') or '' for r in rows], dtype='<U12') if S else np.zeros(0, '<U12')
    lquality = np.array([r.get('quality') or '' for r in rows], dtype='<U10') if S else np.zeros(0, '<U10')
    defenders = [[v['id'] for v in r.get('defenders') or ()] for r in rows]
    line_no_dir, no_unit = no_dir.copy(), np.zeros(S, bool)
    for s, r in enumerate(rows):
        why = str(r.get('reason') or '')
        if r.get('status') == 'unavailable':
            line_no_dir[s] |= why.startswith('direction')
            no_unit[s] = why == 'no defensive unit'
        if r.get('status') == 'complete':
            lheight[s] = r['height']
            dd = np.array([v['distance'] for v in r['defenders']], float)
            lstd[s] = float(dd.std()) if len(dd) else np.nan
            lsize[s] = float(r['unit'])
    lgood = (lstatus == 'complete') & (lsource == 'formation') & (lquality != 'disrupted')
    # A unit from the role memory or the depth-rank fallback is not the formation's back line.
    not_formation = no_unit | ((lsource != '') & (lsource != 'formation'))
    line_reason = _gates(S, cal+[(line_no_dir, NO_DIRECTION), (not_formation, NO_FORMATION),
                                 (np.isin(lstatus, ('unavailable', 'partial', 'estimated')), INCOMPLETE_UNITS),
                                 (lstatus == 'displaced', LINE_DISPLACED), (lquality == 'disrupted', LINE_DISRUPTED),
                                 (~lgood, INCOMPLETE_UNITS)])
    for name in ('lineStd', 'lineHeight', 'lineSize'):
        out._metric(name, line_reason.copy())
    out._put(lineStd=np.where(lgood, lstd, np.nan), lineHeight=np.where(lgood, lheight, np.nan), lineSize=np.where(lgood, lsize, np.nan),
             lineStatus=lstatus, lineSource=lsource, lineQuality=lquality, lineDefenders=defenders)
    # Each defender's distance from the median depth of the other defenders (module docstring "Defensive line"): on
    # the complete formation back lines of three or more, disrupted ones included (a defender far out of the line is
    # what is measured), never on a displaced unit.
    off, off_signed, ref = np.full((S, N), np.nan), np.full((S, N), np.nan), np.full((S, N), np.nan)
    in_line = np.zeros((S, N), bool)
    line_role = np.full((S, N), '', dtype='<U7')
    lfull = (lstatus == 'complete') & (lsource == 'formation')
    for s in np.flatnonzero(lfull):
        dd = rows[s].get('defenders') or ()
        if len(dd) < 3:
            continue
        dist = np.array([v['distance'] for v in dd], float)
        across = np.array([v['y'] for v in dd], float)
        edge = {int(np.argmin(across)), int(np.argmax(across))}        # the two outermost defenders across the pitch
        for i, v in enumerate(dd):
            j = col.get(v['id'])
            if j is None or not obs[s, j]:
                continue
            mid = float(np.median(np.delete(dist, i)))
            ref[s, j] = mid
            off_signed[s, j] = dist[i]-mid
            off[s, j] = abs(dist[i]-mid)
            in_line[s, j] = True
            line_role[s, j] = 'wide' if i in edge else 'central'
    offset_reason = _gates((S, N), [(g[:, None], c) for g, c in cal]+[
        (line_no_dir[:, None], NO_DIRECTION), (not_formation[:, None], NO_FORMATION),
        (np.isin(lstatus, ('unavailable', 'partial', 'estimated'))[:, None], INCOMPLETE_UNITS),
        ((lstatus == 'displaced')[:, None], LINE_DISPLACED), (~lfull[:, None], INCOMPLETE_UNITS),
        (~in_line, NOT_IN_LINE)])
    out._metric('lineOffset', offset_reason)
    out._put(lineOffset=off, lineOffsetSigned=off_signed, lineRef=ref, lineRole=line_role)
    out.timings['line'] = clock()-a

    # Nearest teammate per observed outfield player, with its certainty.
    a = clock()
    near = np.full((S, N), np.nan)
    extra_used = np.zeros(S, int)
    if N:
        with np.errstate(invalid='ignore'):
            D = np.hypot(px[:, :, None]-px[:, None, :], py[:, :, None]-py[:, None, :])
        D = np.where(np.isfinite(D), D, np.inf)
        idx = np.arange(N)
        D[:, idx, idx] = np.inf
        near = D.min(axis=2)
        near[~obs | ~np.isfinite(near)] = np.nan
        del D
        for s, people in enumerate(ctx['people'][team]):
            if not people:
                continue
            pts = [(xm, ym) for pid, xm, ym in people if not (pid in col and obs[s, col[pid]])]
            if not pts:
                continue
            E = np.array(pts, float)
            oj = np.flatnonzero(obs[s])
            if not len(oj):
                continue
            de = np.hypot(px[s, oj][:, None]-E[None, :, 0], py[s, oj][:, None]-E[None, :, 1])     # (observed, extra)
            keep = de.min(axis=0) >= SAME_PERSON_METRES
            if not keep.any():
                continue
            extra_used[s] = int(keep.sum())
            near[s, oj] = np.fmin(near[s, oj], de[:, keep].min(axis=1))
    certain = np.ones((S, N), bool)
    for s, v in enumerate(ctx['view']):
        if v is None or not N or complete[s]:
            continue            # the whole lineup observed: no unseen teammate can be nearer, the view does not matter
        if v is False:
            certain[s] = False
            continue
        oj = np.flatnonzero(obs[s] & np.isfinite(near[s]))
        if len(oj):
            certain[s, oj] = _disc_inside(v, px[s, oj], py[s, oj], near[s, oj], L, W)
    run_small = np.full(S, bool(small_scene))
    near_reason = _gates((S, N), [(g[:, None], c) for g, c in cal]+[
        (run_small[:, None], SMALL_SCENE), (~obs, NOT_OBSERVED), (~np.isfinite(near), NO_TEAMMATE),
        ((ctx['unplaced'][team] > 0)[:, None], UNPLACED_TEAMMATE), (~certain, OUT_OF_VIEW)])
    out._metric('nearestTeammate', near_reason)
    out._put(nearestTeammate=near, nearestExtras=extra_used)
    out.timings['nearest'] = clock()-a

    # Ball context, from this team's point of view.
    a = clock()
    ball = ctx['ball']
    bX, bY = ts.attack_frame(ball['x'], ball['y'], side)
    bX, bY = bX*L, bY*W
    ball_reason = _gates(S, cal+[(run_small, SMALL_SCENE)])
    free = ball_reason == ''
    ball_reason[free] = ball['reason'][free]           # the ball's own gate (no ball, uncertain, off the pitch)
    out._metric('ball', ball_reason)
    zr = ball_reason.copy()
    zr[(zr == '') & no_dir] = NO_DIRECTION
    out._metric('ballThird', zr)
    out._metric('ballChannel', zr.copy())
    ok = zr == ''
    out._put(ballX=np.where(known, bX, np.nan), ballY=np.where(known, bY, np.nan),
             ballThird=_zones(t, bX, ok, (L/THIRDS, 2*L/THIRDS), ('own', 'middle', 'final')),
             ballChannel=_zones(t, bY, ok, (W/CHANNELS, 2*W/CHANNELS), ('left', 'centre', 'right')),
             ballObserved=ball['observed'] & (ball_reason == ''))
    out.timings['ball'] = clock()-a

    # Velocity per observed outfield player (attack frame, m/s), from positions on frames the registration gates pass.
    a = clock()
    frame_ok = _gates(S, cal) == ''
    usable = obs & known[:, None] & frame_ok[:, None]
    vel_x, vel_y = velocities(t, Xm, Ym, usable, side)
    vel_reason = _gates((S, N), [(g[:, None], c) for g, c in cal]+[
        (no_dir[:, None], NO_DIRECTION), (~obs, NOT_OBSERVED), (~np.isfinite(vel_x), SHORT_TRACK)])
    out._metric('velocity', vel_reason)
    out._put(velX=vel_x, velY=vel_y, speed=np.hypot(vel_x, vel_y))
    out.timings['velocity'] = clock()-a

    # Stabilizer flags near each sample (evidence veto only).
    flagged = np.zeros((S, N), bool)
    for pid, times in ctx['flags'].items():
        j = col.get(pid)
        if j is None or not len(times) or not S:
            continue
        k = np.searchsorted(times, t-FLAG_SECONDS, side='left')
        hit = k < len(times)
        flagged[hit, j] = times[k[hit]] <= t[hit]+FLAG_SECONDS
    out._put(flagged=flagged)
    # A value exists only where it may be believed (in place: the attributes are the same arrays).
    for name, metric in VALUE_METRIC.items():
        v = out.values[name]
        v[~out.eligible[metric]] = '' if v.dtype.kind == 'U' else np.nan
    out.timings['total'] = clock()-a0
    return out


def build_series(result, samples=None, shape=None, settings=None, mode='observed'):
    """The coaching series of a result for both teams (CoachingData). samples: an analytics.Samples of the result, and
    shape: its team_shape.TeamShape (both built here when not given; the server passes its cached ones); settings: the
    video's team-shape settings (team_shape.json: direction, lineup, formation labels) for a TeamShape built here; mode:
    the TeamShape mode built here. Whatever the mode, every coaching measurement uses the OBSERVED data
    (shape.observed): an estimate is never a coaching fact."""
    if mode not in ('observed', 'estimated'):
        raise ValueError('mode must be observed or estimated')
    clock = time.perf_counter
    a0 = clock()
    out = CoachingData()
    if shape is None:
        a = clock()
        samples = samples if samples is not None else analytics.Samples(result)
        out.timings['samples'] = clock()-a
        a = clock()
        shape = ts.TeamShape(result, samples=samples, mode=mode, settings=settings)
        out.timings['teamShape'] = clock()-a
    data = shape.observed
    out.shape, out.samples, out.result = shape, samples if samples is not None else data.samples, result
    a = clock()
    ctx = _frame_context(result, data)
    out.timings['frames'] = clock()-a
    frames = result.get('frames') or []
    out.t = data.t
    out.frame = ctx['frame']
    out.fps = ctx['fps']
    out.start, out.end = data.start, data.end
    out.pitch = {'length': data.length, 'width': data.width, 'source': data.size_source}
    out.metres_estimated = bool(data.metres_estimated)
    out.small_scene = labels.small_scene_run(frames)
    out.available = {'calibration': bool(ctx['calibration']['present'].any()), 'view': any(v is not None for v in ctx['view']),
                     'ball': bool(ctx['ball']['present']), 'stabilized': bool(ctx['stabilized'])}
    out.calibration = {k: v for k, v in ctx['calibration'].items() if k != 'gates'}
    out.ball = {k: v for k, v in ctx['ball'].items() if k != 'present'}
    out.view = ctx['view']
    out.unplaced = ctx['unplaced']
    out.in_kit, out.confirmed_in_kit = ctx['quality']['inKit'], ctx['quality']['confirmed']
    for team in ('A', 'B'):
        a = clock()
        out.teams[team] = _team_series(team, shape, ctx, out.small_scene, data.start)
        out.timings['team'+team] = clock()-a
    out.timings['total'] = clock()-a0
    return out
