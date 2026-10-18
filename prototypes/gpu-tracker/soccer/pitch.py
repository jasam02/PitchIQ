"""Playable-field detection, recomputed for every analysed frame.

The legal field is defined by its white boundary lines, not by green grass. White paint is found as
a thin bright ridge on the grass (morphological top-hat at 640 px width), so even a thin, blurred far
touchline is seen. Long straight lines are fitted, and for each side of the pitch the outermost line
that is painted on grass (grass on both sides, a run-off strip beyond it) becomes the boundary:
advertising-board edges have no grass beyond them and are never boundaries. A steep line with short
lines leaving it towards the image edge is a penalty/goal-area front, not a goal line. Boundary lines
are tracked over time (smoothed, carried through camera motion, changed only when a new position is
confirmed), and the grass region is only the fallback for sides where no boundary line is visible.

Image-plane only. Metric pitch coordinates come from calibration.py and the tracked pitch model
(pitch_model.py), which also supplies the playable region when its confidence is high enough; the
confidence of this image-plane detection itself is scored by assess_confidence().
"""
import math
from dataclasses import dataclass, field

import cv2
import numpy as np

from .geometry import (centroid, clip_polygon, clip_unit, convex_hull, foot_point, line_distance,
                       point_in_polygon, polygon_area, segment_distance, signed_distance,
                       transform_points, x_at, y_at)
from .pitch_model import MEDIUM, levels

MIN_GRASS = .12          # grass share of the frame needed for a reliable pitch
LINE_MAX_AGE = 3.0       # seconds a boundary line is carried with a reliable camera estimate
LINE_MAX_AGE_LOOSE = .8  # ... when the camera estimate is unreliable
FINE_W = 320             # grass sampling grid width (cells)
LINE_W = 640             # paint / line analysis width (pixels)
CIRCLE_EDGE = .02        # frame heights: paint this close to the grass region's top or bottom edge is left out of the on-grass circle search
CIRCLE_MIN_DENSITY = .6  # inlier pixels per pixel of arc seen (mask resolution) an on-grass circle needs: a long thin scatter is not paint
LOW_RES_HEIGHT = 720     # source rows below which a frame whose turf the fixed grass rule rejects learns the turf colour instead (ground_colour)
GROUND_FROM = .45        # frame heights: the turf colour is learned below this row
GROUND_HUE = (15, 55)    # OpenCV hue (0-180) of plausible turf: yellow-olive to blue-green (A_wrong 21-39, natural grass 35-50)
GROUND_MIN_SAT = 40      # of 255: grey concrete, sand and snow are not turf
GROUND_VALUE = (40, 230)
GROUND_VALUE_RATIO = 2.2 # brightness range around the turf's (shadows, mowing stripes)
GROUND_HUE_TOL, GROUND_SAT_TOL = 6, 30   # at least this tolerance around the learned hue and saturation
GROUND_SHARE = .5        # of the lower frame that the learned colour must cover
CIRCLE_CENTRE_TOLERANCE = .25 # an on-grass circle's centre lies this close to the halfway line in the ellipse's normalized coordinates (true circles: <= .07)
CIRCLE_MAX_SPAN = .6     # frame widths: a straight marking longer than this is a line, never an arc of the circle (the near arc of amateur 3's circle at 22.3 s spans .47)


@dataclass
class FieldFilter:
    enabled: bool = True
    boundary_margin: float = .35  # tolerance beyond the boundary, fraction of the person's box height
    min_margin: float = .006      # floor of that tolerance, frame-height units
    max_margin: float = .03       # ceiling of that tolerance, frame-height units

    @classmethod
    def with_margin(cls, margin):
        """Touchline tolerance from the UI; floor and ceiling scale with it so 0 means none."""
        base = cls()
        margin = max(0.0, float(margin)) if math.isfinite(float(margin)) else base.boundary_margin
        k = margin/base.boundary_margin
        return cls(True, margin, base.min_margin*k, base.max_margin*k)


@dataclass
class PitchModel:
    time: float
    reliable: bool
    aspect: float
    polygon: list = field(default_factory=list)        # playable region, normalized points
    grass_polygon: list = field(default_factory=list)  # grass hull before line clipping
    lines: list = field(default_factory=list)          # boundary lines [{'a','b','side','support','age'}]
    coverage: float = 0.0
    source: str = 'none'                               # auto | calibration | model | none
    goals: list = field(default_factory=list)          # goal-end hints [{'side','kind': goal-line|box-front,'a','b'}]
    markings: list = field(default_factory=list)       # all fitted paint lines (debug)
    confidence: float = 0.0                            # how far this playable region can be trusted (0..1)
    mode: str = 'automatic'                            # automatic | robust | assist (see tracker.py)
    status: str = ''                                   # one line for the overlay: where the region comes from
    evidence: list = field(default_factory=list)       # human-readable lines: what was found, predicted, missing
    sides: dict = None                                 # per side: line | carried line | grass edge | predicted | off-screen
    vanishing: str | None = None                       # 'left' / 'right': the touchlines converge there with no goal-line evidence
    curvature: float | None = None                     # sagitta of the longest touchline-family line, px per 1000 px: lens check

    @property
    def level(self):
        return levels(self.confidence)

    def sources(self):
        """Where each side of the polygon comes from: a detected line, a carried line, or the grass edge."""
        if self.sides:
            return dict(self.sides)
        out = {side: 'grass edge' for side in ('far', 'near', 'left', 'right')}
        if self.source == 'calibration':
            return {side: 'calibration' for side in out}
        for l in self.lines:
            out[l['side']] = 'line' if l['age'] <= 0 else 'carried line'
        if self.vanishing:
            out[self.vanishing] = 'unknown (the touchlines converge there: a vanishing point, not the pitch end)'
        return out

    def to_json(self):
        r = lambda p: [round(float(p[0]), 4), round(float(p[1]), 4)]
        return {'reliable': self.reliable, 'source': self.source, 'coverage': round(self.coverage, 3),
                'confidence': round(float(self.confidence), 3), 'level': self.level, 'mode': self.mode, 'status': self.status,
                'evidence': list(self.evidence),
                'polygon': [r(p) for p in self.polygon], 'grass': [r(p) for p in self.grass_polygon],
                'lines': [{'a': r(l['a']), 'b': r(l['b']), 'side': l['side'], 'age': round(l['age'], 2)} for l in self.lines],
                'goals': [{'side': g['side'], 'kind': g['kind'], 'a': r(g['a']), 'b': r(g['b'])} for g in self.goals],
                'markings': [[*r(m['a']), *r(m['b'])] for m in self.markings[:16]], 'edges': self.sources(),
                'vanishing': self.vanishing, 'curvature': self.curvature}


# ---------- sampling ----------
def _grass(r, g, b):
    # Accepts worn / yellow-green turf but not sand, concrete, sky or paint.
    return (g > 30) & (g > r*.95) & (g > b*1.12) & (g-b > 12)


def ground_colour(frame):
    """The turf's colour learned from the frame, for footage whose turf the fixed grass rule rejects (a screen
    recording's colour cast puts red above green: A_wrong's turf is BGR 72/104/115): the median hue,
    saturation and value of the lower part of the frame (a match view shows turf there) with their spread,
    or None when that colour is not a plausible turf (hue, saturation, brightness) or covers too little of
    the lower frame."""
    h = frame.shape[0]
    hsv = cv2.cvtColor(frame[int(GROUND_FROM*h):], cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(np.int16)
    if not len(hsv):
        return None
    centre = np.median(hsv, axis=0)
    spread = np.median(np.abs(hsv-centre), axis=0)
    hue, sat, val = centre
    if not (GROUND_HUE[0] <= hue <= GROUND_HUE[1] and sat >= GROUND_MIN_SAT and GROUND_VALUE[0] <= val <= GROUND_VALUE[1]):
        return None
    colour = (float(hue), float(sat), float(val), max(GROUND_HUE_TOL, 3*float(spread[0])), max(GROUND_SAT_TOL, 3*float(spread[1])))
    if _ground(hsv.reshape(1, -1, 3), colour).mean() < GROUND_SHARE:
        return None
    return colour


def _ground(hsv, colour):
    """Pixels (an HSV image, int16) of a learned ground colour."""
    hue, sat, val, hue_tol, sat_tol = colour
    return ((np.abs(hsv[..., 0]-hue) <= hue_tol) & (np.abs(hsv[..., 1]-sat) <= sat_tol) &
            (hsv[..., 2] >= val/GROUND_VALUE_RATIO) & (hsv[..., 2] <= val*GROUND_VALUE_RATIO))


def line_masks(frame, ground=None):
    """Grass and white-paint masks at <= 640 px width. Paint is a thin bright ridge: the top-hat of the
    darkest colour channel (white is bright in all three, grass is dark in red and blue), so a 1-2 px
    far touchline blended with grass still stands out, while wide white advertising boards do not.
    ground: a learned turf colour (ground_colour) that replaces the fixed grass rule."""
    h, w = frame.shape[:2]
    fw = min(FINE_W, w)
    fh = max(2, min(h, round(fw*h/w)))
    small = cv2.resize(frame, (fw*2, fh*2), interpolation=cv2.INTER_AREA)
    b8, g8_, r8 = cv2.split(small)
    b, g, r = b8.astype(np.int16), g8_.astype(np.int16), r8.astype(np.int16)
    grass = _grass(r, g, b) if ground is None else _ground(cv2.cvtColor(small, cv2.COLOR_BGR2HSV).astype(np.int16), ground)
    low = cv2.min(cv2.min(b8, g8_), r8)
    high = cv2.max(cv2.max(b8, g8_), r8)
    spread = high.astype(np.int16)-low
    tophat = cv2.morphologyEx(low, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9)))
    ridge = (tophat >= 28) & (spread <= 110) & (low >= 80)
    g8 = grass.astype(np.uint8)
    # Grass at >= 2 of the 4 pixels 3..6 px away on one side (box filters along the offset direction).
    above = cv2.filter2D(g8, -1, np.array([[1], [1], [1], [1], [0], [0], [0], [0], [0], [0], [0], [0], [0]], np.float32), anchor=(0, 6), borderType=cv2.BORDER_CONSTANT) >= 2
    below = cv2.filter2D(g8, -1, np.array([[0], [0], [0], [0], [0], [0], [0], [0], [0], [1], [1], [1], [1]], np.float32), anchor=(0, 6), borderType=cv2.BORDER_CONSTANT) >= 2
    left = cv2.filter2D(g8, -1, np.array([[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]], np.float32), anchor=(6, 0), borderType=cv2.BORDER_CONSTANT) >= 2
    right = cv2.filter2D(g8, -1, np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]], np.float32), anchor=(6, 0), borderType=cv2.BORDER_CONSTANT) >= 2
    return {'w': fw*2, 'h': fh*2, 'grass': grass, 'ridge': ridge, 'low': low, 'high': high, 'sum': b+g+r, 'tophat': tophat,
            'h_line': ridge & above & below,   # paint with grass above and below
            'v_line': ridge & left & right}    # paint with grass left and right


def sample_frame(frame, masks=None):
    """Per-cell grass fraction, white paint and dark (letterbox) flags on a <=320 wide grid."""
    masks = masks or line_masks(frame)
    fw, fh = masks['w']//2, masks['h']//2
    grass, low, high = masks['grass'], masks['low'], masks['high']
    white = (low >= 110) & (high.astype(np.int16)-low <= 70) & (masks['sum'] >= 380) & ~grass
    pool = lambda mask: cv2.resize(mask.astype(np.float32), (fw, fh), interpolation=cv2.INTER_AREA)
    return {'w': fw, 'h': fh, 'frac': pool(grass), 'white': pool(white | masks['ridge']) > 1e-3, 'dark': pool(high < 28) > 1-1e-3}


def _shift(m, dy, dx):
    """m shifted so out[y, x] = m[y+dy, x+dx] (False outside)."""
    out = np.zeros_like(m)
    h, w = m.shape
    ys, yd = (slice(dy, h), slice(0, h-dy)) if dy >= 0 else (slice(0, h+dy), slice(-dy, h))
    xs, xd = (slice(dx, w), slice(0, w-dx)) if dx >= 0 else (slice(0, w+dx), slice(-dx, w))
    out[yd, xd] = m[ys, xs]
    return out


def _near(m, dy, dx, reach=3):
    hit = np.zeros_like(m)
    for k in range(1, reach+1):
        hit |= _shift(m, dy*k, dx*k)
    return hit


def close_gaps(m, white, gap=3):
    """Fill <= gap non-grass cells between grass cells when every filled cell is paint or a narrow
    object with grass on both sides across the run (a player). A band of advertising boards is
    neither, so green seats behind the boards never join the pitch."""
    src = m.copy()
    ok_rows = white | (_near(src, -1, 0) & _near(src, 1, 0))  # horizontal runs: grass above and below
    ok_cols = white | (_near(src, 0, -1) & _near(src, 0, 1))  # vertical runs: grass left and right
    for axis, ok in ((1, ok_rows), (0, ok_cols)):
        for length in range(1, gap+1):
            start = np.ones_like(src)
            dy, dx = (0, 1) if axis == 1 else (1, 0)
            start &= _shift(src, -dy, -dx)                      # grass just before the run
            start &= _shift(src, dy*length, dx*length)          # grass just after the run
            for k in range(length):
                start &= ~_shift(src, dy*k, dx*k) & _shift(ok, dy*k, dx*k)
            for k in range(length):
                m |= _shift(start, -dy*k, -dx*k)
    return m


def grass_region(f):
    k = 2 if f['w'] >= 200 else 1
    h, w = f['h']//k, f['w']//k
    frac = f['frac'][:h*k, :w*k].reshape(h, k, w, k).mean(axis=(1, 3))
    white = f['white'][:h*k, :w*k].reshape(h, k, w, k).any(axis=(1, 3))
    m = close_gaps(frac >= .5, white)
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)), borderType=cv2.BORDER_REPLICATE).astype(bool)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=4)
    if count <= 1:
        mask = np.zeros_like(m)
    else:
        best = 1+int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        mask = labels == best
    size = int(mask.sum())
    rows, cols = mask.any(axis=1), mask.any(axis=0)
    left = np.where(rows, mask.argmax(axis=1), -1)
    right = np.where(rows, w-1-mask[:, ::-1].argmax(axis=1), -1)
    top = np.where(cols, mask.argmax(axis=0), -1)
    bottom = np.where(cols, h-1-mask[::-1, :].argmax(axis=0), -1)
    ys, xs = np.nonzero(mask)
    cw, ch = k/f['w'], k/f['h']
    # Uniformly dark rows/columns at the frame edges (4:3 footage in a 16:9 frame) are not part of the view.
    dark = f['dark']
    def run(flags, limit):
        n = 0
        while n < limit and flags[n]:
            n += 1
        return n
    bl, br = run(dark.all(axis=0), f['w']//4), run(dark.all(axis=0)[::-1], f['w']//4)
    bt, bb = run(dark.all(axis=1), f['h']//4), run(dark.all(axis=1)[::-1], f['h']//4)
    return {'w': w, 'h': h, 'cw': cw, 'ch': ch, 'mask': mask, 'count': size,
            'cx': (xs.mean()+.5)*cw if size else .5, 'cy': (ys.mean()+.5)*ch if size else .5,
            'left': left, 'right': right, 'top': top, 'bottom': bottom,
            'x0': math.ceil(bl/k), 'x1': w-1-math.ceil(br/k), 'y0': math.ceil(bt/k), 'y1': h-1-math.ceil(bb/k)}


# ---------- boundary lines ----------
def hough(U, V, u_len, v_len, max_slope, step, bin_w, max_lines, min_support, weights=None):
    """Lines through points (u along, v across) with |dv/du| <= max_slope. OpenCV's Hough transform
    proposes peaks; each is refined by least squares on the remaining points, and accepted only when
    enough distinct u positions (grid columns or rows) support it, so thick lines are not
    over-counted. Inliers are removed before the next peak."""
    U, V = np.asarray(U, np.int64), np.asarray(V, np.float64)
    Wt = np.ones(len(U)) if weights is None else np.asarray(weights, np.float64)
    out = []
    if len(U) < min_support:
        return out
    mask = np.zeros((int(v_len)+1, int(u_len)+1), np.uint8)
    mask[np.clip(V.astype(np.int64), 0, int(v_len)), np.clip(U, 0, int(u_len))] = 255
    low = math.atan2(1, max_slope)
    peaks = cv2.HoughLines(mask, 1, np.pi/180, max(2, int(min_support*.6)), min_theta=low, max_theta=math.pi-low)
    if peaks is None:
        return out
    uc = u_len/2
    alive = np.ones(len(U), bool)
    for rho, theta in peaks[:24, 0]:
        if len(out) >= max_lines or alive.sum() < min_support:
            break
        sin, cos = math.sin(theta), math.cos(theta)
        if abs(sin) < 1e-6:
            continue
        s, c = -cos/sin, (rho-uc*cos)/sin
        u, v, wt = U[alive], V[alive], Wt[alive]
        du = u-uc
        for tol in (bin_w+1, 1.5):
            sel = np.abs(v-(c+s*du)) <= tol
            if sel.sum() < 2:
                break
            # Weighted by paint strength: a half-blended edge pixel pulls the fit less than the line's core.
            ws, dus, vs = wt[sel], du[sel], v[sel]
            n, su, sv, suu, suv = ws.sum(), (ws*dus).sum(), (ws*vs).sum(), (ws*dus*dus).sum(), (ws*dus*vs).sum()
            den = n*suu-su*su
            if den <= 1e-9:
                break
            s = (n*suv-su*sv)/den
            c = (sv-s*su)/n
        residual = np.abs(V-(c+s*(U-uc)))
        near = alive & (residual <= 1.5)
        columns = np.unique(U[near])
        if len(columns) < min_support or abs(s) > max_slope*1.15:
            continue
        # Keep the longest run of columns without a big gap: a fit that strings together unrelated paint
        # (a tangent to the centre circle and a box edge) falls apart into short pieces.
        breaks = np.flatnonzero(np.diff(columns) > max(4, .05*u_len))
        starts, ends = np.r_[0, breaks+1], np.r_[breaks, len(columns)-1]
        best = int(np.argmax(ends-starts))
        run = columns[starts[best]:ends[best]+1]
        if len(run) < min_support:
            continue
        u0, u1 = float(run[0]), float(run[-1])
        alive &= ~((residual <= 3) & (U >= u0) & (U <= u1))
        out.append({'u0': u0, 'u1': u1, 'v0': c+s*(u0-uc), 'v1': c+s*(u1-uc), 'support': len(run)})
    return out


def detect_lines(masks):
    """Straight paint lines with grass on both sides: near-horizontal (touchline family, including short
    penalty-box side edges) and steep (goal lines, halfway line, box fronts). Normalized endpoints."""
    w, h = masks['w'], masks['h']
    hy, hx = np.nonzero(masks['h_line'])
    vy, vx = np.nonzero(masks['v_line'])
    strength = masks['tophat']
    segs = []
    for l in hough(hx, hy, w, h, .36, .02, 2, 8, .04*w, strength[hy, hx]):
        segs.append({'kind': 'h', 'a': ((l['u0']+.5)/w, (l['v0']+.5)/h), 'b': ((l['u1']+.5)/w, (l['v1']+.5)/h), 'support': l['support']/w})
    for l in hough(vy, vx, h, w, 2.9, .05, 3, 8, .1*h, strength[vy, vx]):
        segs.append({'kind': 'v', 'a': ((l['v0']+.5)/w, (l['u0']+.5)/h), 'b': ((l['v1']+.5)/w, (l['u1']+.5)/h), 'support': l['support']/h})
    return segs


def line_quality(line, side, masks):
    """Is this a painted boundary on the grass? Samples perpendicular profiles along the line: paint on
    the line, grass just inside it, and grass just beyond it (the run-off strip). An advertising-board
    edge or a stand has no grass beyond it."""
    w, h = masks['w'], masks['h']
    (ax, ay), (bx, by) = line['a'], line['b']
    dx, dy = (bx-ax)*w, (by-ay)*h
    length = math.hypot(dx, dy)
    if length < 4:
        return {'paint': 0.0, 'inside': 0.0, 'beyond': 0.0}
    nx, ny = -dy/length, dx/length
    out = {'near': (0, 1), 'far': (0, -1), 'left': (-1, 0), 'right': (1, 0)}[side]
    if nx*out[0]+ny*out[1] < 0:
        nx, ny = -nx, -ny
    t = np.linspace(.03, .97, 40)
    x, y = ax*w+dx*t, ay*h+dy*t
    keep = (x >= 0) & (x < w) & (y >= 0) & (y < h)
    x, y = x[keep], y[keep]
    if not len(x):
        return {'paint': 0.0, 'inside': 0.0, 'beyond': 0.0}

    def hits(mask, offsets):
        xs = np.rint(x[:, None]+nx*offsets[None, :]).astype(int)
        ys = np.rint(y[:, None]+ny*offsets[None, :]).astype(int)
        ok = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
        values = np.zeros(xs.shape, bool)
        values[ok] = mask[ys[ok], xs[ok]]
        return values.sum(axis=1)

    return {'paint': float((hits(masks['ridge'], np.array([-1, 0, 1])) >= 1).mean()),
            'inside': float((hits(masks['grass'], -np.arange(3, 9)) >= 3).mean()),
            'beyond': float((hits(masks['grass'], np.arange(3, 9)) >= 3).mean())}


def box_front(line, side, segs, aspect):
    """A steep line with short touchline-family lines leaving it towards its side of the image is the
    front edge of a penalty or goal area (their side edges run to the goal line), not a goal line."""
    out = -1 if side == 'left' else 1
    seg = (line['a'], line['b'])
    lo, hi = min(line['a'][1], line['b'][1])-.03, max(line['a'][1], line['b'][1])+.03
    for other in segs:
        if other['kind'] != 'h' or abs(other['b'][0]-other['a'][0]) > .5:
            continue
        for p, q in ((other['a'], other['b']), (other['b'], other['a'])):
            if not lo <= p[1] <= hi or line_distance(p, seg[0], seg[1], aspect) > .03:
                continue
            if out*(q[0]-x_at(seg, q[1])) > .02:  # the other end lies beyond this line, towards the goal
                return True
    return False


def _horizontal(side):
    return side in ('near', 'far')


def side_of(line, horizontal, c):
    seg = (line['a'], line['b'])
    if horizontal:
        return 'far' if y_at(seg, c[0]) < c[1] else 'near'
    return 'left' if x_at(seg, c[1]) < c[0] else 'right'


def band_stats(line, side, r):
    """Grass beyond vs inside a line, whether the grass beyond ends inside the frame (a real
    boundary has boards/stands behind it), the median depth of that band and line/edge agreement."""
    seg = (line['a'], line['b'])
    horizontal, out = _horizontal(side), 1 if side in ('near', 'right') else -1
    ys, xs = np.nonzero(r['mask'])
    px, py = (xs+.5)*r['cw'], (ys+.5)*r['ch']
    (ax, ay), (bx, by) = seg
    if horizontal:
        ly = ay+(by-ay)*(px-ax)/(bx-ax) if abs(bx-ax) > 1e-9 else np.full_like(px, ay)
        o = out*(py-ly)
        tol = r['ch']
    else:
        lx = ax+(bx-ax)*(py-ay)/(by-ay) if abs(by-ay) > 1e-9 else np.full_like(py, ax)
        o = out*(px-lx)
        tol = r['cw']
    inner, beyond = int((o < 0).sum()), int((o > tol).sum())
    count = term = on = 0
    depths = []
    if horizontal:
        x0 = max(0, int(min(ax, bx)/r['cw']))
        x1 = min(r['w']-1, int(max(ax, bx)/r['cw']))
        for x in range(x0, x1+1):
            if r['top'][x] < 0:
                continue
            count += 1
            ly = y_at(seg, (x+.5)*r['cw'])
            if side == 'far':
                term += r['top'][x] >= r['y0']+1
                depths.append(max(0.0, ly-r['top'][x]*r['ch']))
                on += ly >= r['top'][x]*r['ch']-r['ch']
            else:
                term += r['bottom'][x] <= r['y1']-1
                depths.append(max(0.0, (r['bottom'][x]+1)*r['ch']-ly))
                on += ly <= (r['bottom'][x]+2)*r['ch']
    else:
        y0 = max(0, int(min(ay, by)/r['ch']))
        y1 = min(r['h']-1, int(max(ay, by)/r['ch']))
        for y in range(y0, y1+1):
            if r['left'][y] < 0:
                continue
            count += 1
            lx = x_at(seg, (y+.5)*r['ch'])
            if side == 'left':
                term += r['left'][y] >= r['x0']+1
                depths.append(max(0.0, lx-r['left'][y]*r['cw']))
                on += lx >= r['left'][y]*r['cw']-r['cw']
            else:
                term += r['right'][y] <= r['x1']-1
                depths.append(max(0.0, (r['right'][y]+1)*r['cw']-lx))
                on += lx <= (r['right'][y]+2)*r['cw']
    depths.sort()
    return {'inner': inner, 'beyond': beyond, 'term': term/count if count else 0, 'count': count,
            'depth': depths[len(depths)//2] if depths else math.inf, 'on': on/count if count else 0}


def choose_boundaries(segs, r, aspect, masks):
    """For each side the outermost painted line with a thin grass run-off beyond it is the boundary.
    Lines that are not paint on grass are skipped; a line with too much grass beyond it is interior,
    and so is every line inside it. Paint at the very edge of the grass with nothing beyond it (an
    advertising-board edge, or a touchline with no visible run-off) is used only when no painted line
    with grass beyond it lies just inside it. Returns (lines, goal hints)."""
    c = (r['cx'], r['cy'])
    by_side = {}
    for seg in segs:
        horizontal = seg['kind'] == 'h'
        span = abs(seg['b'][0]-seg['a'][0]) if horizontal else abs(seg['b'][1]-seg['a'][1])
        if (seg['support'] < .25 or span < .4) if horizontal else (seg['support'] < .2 or span < .25):
            continue
        side = side_of(seg, horizontal, c)
        by_side.setdefault(side, []).append((line_distance(c, seg['a'], seg['b'], aspect), seg))
    found, goals = {}, []
    for side, items in by_side.items():
        chosen, edge_only, interior = None, None, False
        for distance, seg in sorted(items, key=lambda item: -item[0]):
            q = line_quality(seg, side, masks)
            if q['paint'] < .35 or q['inside'] < .5:
                continue  # not paint lying on the grass: look further in
            s = band_stats(seg, side, r)
            if not s['count'] or s['on'] < .5:
                continue
            if side in ('left', 'right') and box_front(seg, side, segs, aspect):
                # The front of a penalty or goal area marks the goal end, also when the goal line beyond
                # it is out of view or cannot be confirmed (it runs off the image edge).
                goals.append({'side': side, 'kind': 'box-front', 'a': seg['a'], 'b': seg['b']})
                break
            if interior:
                continue  # inside an interior line nothing is a boundary; only a box front is still looked for
            if s['beyond'] >= .25*s['inner']:
                interior = True  # deep grass beyond: an interior line (and so is everything inside it)
                continue
            if (s['term'] < .5 and s['depth'] > .12) if side == 'near' else s['term'] < .5:
                interior = True  # the grass beyond does not end inside the view: not a boundary
                continue
            if q['beyond'] < .4:
                if edge_only is None:
                    edge_only = (distance, seg)
                continue
            if edge_only is None or edge_only[0]-distance <= .12:
                chosen = seg
            break
        if chosen is None and edge_only is not None:
            chosen = edge_only[1]
        if chosen is not None:
            found[side] = {'a': chosen['a'], 'b': chosen['b'], 'side': side, 'support': chosen['support'], 'age': 0.0}
            if side in ('left', 'right'):
                goals.append({'side': side, 'kind': 'goal-line', 'a': chosen['a'], 'b': chosen['b']})
    return found, goals


def move_line(line, matrix):
    if matrix is None:
        return dict(line)
    a, b = transform_points(matrix, [line['a'], line['b']])
    return dict(line, a=(float(a[0]), float(a[1])), b=(float(b[0]), float(b[1])))


def _params(line):
    """(slope, intercept, lo, hi): y = m*x + c for the touchline family, x = m*y + c for steep lines."""
    (ax, ay), (bx, by) = line['a'], line['b']
    if _horizontal(line['side']):
        m = (by-ay)/(bx-ax) if abs(bx-ax) > 1e-9 else 0.0
        return m, ay-m*ax, min(ax, bx), max(ax, bx)
    m = (bx-ax)/(by-ay) if abs(by-ay) > 1e-9 else 0.0
    return m, ax-m*ay, min(ay, by), max(ay, by)


def _from_params(side, m, c, lo, hi, **extra):
    if _horizontal(side):
        a, b = (lo, m*lo+c), (hi, m*hi+c)
    else:
        a, b = (m*lo+c, lo), (m*hi+c, hi)
    return dict(extra, a=(float(a[0]), float(a[1])), b=(float(b[0]), float(b[1])), side=side)


def lines_close(l1, l2, tolerance=.025):
    """Two lines of the same side agree within `tolerance` (normalized) across their common extent."""
    m1, c1, lo1, hi1 = _params(l1)
    m2, c2, lo2, hi2 = _params(l2)
    lo, hi = max(lo1, lo2), min(hi1, hi2)
    if hi < lo:
        lo, hi = min(lo1, lo2), max(hi1, hi2)
    return all(abs((m1*t+c1)-(m2*t+c2)) <= tolerance for t in (lo, (lo+hi)/2, hi))


def blend_lines(old, new, weight=.5):
    m1, c1, _, _ = _params(old)
    m2, c2, lo, hi = _params(new)
    return _from_params(new['side'], m1+(m2-m1)*weight, c1+(c2-c1)*weight, lo, hi, support=new['support'], age=0.0)


class PitchDetector:
    """Re-detects the playable region on every call and tracks each side's boundary line over time:
    a new detection close to the tracked line refines it (smoothing); a detection far from it must be
    seen twice before the boundary moves (hysteresis); without a detection the line is carried with
    the camera motion for a few seconds while it stays consistent with the grass."""

    def __init__(self):
        self.previous = None
        self.tracked = {}   # side -> line
        self.pending = {}   # side -> unconfirmed new position
        self.masks = None   # the paint and grass masks of the last analysed frame (for the centre circle, paint checks)

    def reset(self):
        self.previous = None
        self.tracked, self.pending = {}, {}

    def analyze(self, frame, time, motion=None, reliable=False, cut=False):
        """motion: normalized 3x3 transform from the previous analysed frame to this one."""
        h, w = frame.shape[:2]
        aspect = w/max(1, h)
        empty = PitchModel(time, False, aspect)
        if cut:
            self.tracked, self.pending = {}, {}
        if w < 16 or h < 16:
            self.previous = empty
            return empty
        masks = line_masks(frame)
        f = sample_frame(frame, masks)
        r = grass_region(f)
        if r['count'] < r['w']*r['h']*MIN_GRASS and h < LOW_RES_HEIGHT:
            # A low-resolution frame (a screen recording) whose turf the fixed rule rejects: the turf colour
            # learned from the frame stands in. Never on footage 720 px tall or more.
            colour = ground_colour(frame)
            if colour is not None:
                masks = line_masks(frame, colour)
                f = sample_frame(frame, masks)
                r = grass_region(f)
        self.masks = masks
        if r['count'] < r['w']*r['h']*MIN_GRASS:
            self.previous = empty
            return empty
        corners = []
        for y in range(r['h']):
            if r['left'][y] < 0:
                continue
            x0, x1 = r['left'][y]*r['cw'], (r['right'][y]+1)*r['cw']
            y0, y1 = y*r['ch'], (y+1)*r['ch']
            corners += [(x0, y0), (x0, y1), (x1, y0), (x1, y1)]
        grass_polygon = convex_hull(corners)
        segs = detect_lines(masks)
        found, goals = choose_boundaries(segs, r, aspect, masks)
        dt = max(0.0, time-self.previous.time) if self.previous is not None else 0.0
        max_age = LINE_MAX_AGE if reliable else LINE_MAX_AGE_LOOSE
        c = (r['cx'], r['cy'])
        tracked = {}
        for side in ('far', 'near', 'left', 'right'):
            old = self.tracked.get(side)
            if old is not None:
                old = move_line(old, motion if reliable else None)
                if side_of(old, _horizontal(side), c) != side:
                    old = None
            new = found.get(side)
            pending = self.pending.pop(side, None)
            if pending is not None:
                pending = move_line(pending, motion if reliable else None)
            if new is not None and old is not None and lines_close(old, new):
                tracked[side] = blend_lines(old, new)
            elif new is not None and (old is None or (pending is not None and lines_close(pending, new))):
                tracked[side] = new
            else:
                if new is not None:
                    self.pending[side] = new  # a jump: wait for confirmation
                if old is not None and old['age']+dt <= max_age:
                    stats = band_stats(old, side, r)
                    if stats['count'] and stats['beyond'] < .35*stats['inner']:
                        tracked[side] = dict(old, age=old['age']+dt)
        self.tracked = tracked
        lines = list(tracked.values())
        polygon = grass_polygon
        for line in lines:
            clipped = clip_polygon(polygon, line['a'], line['b'], c)
            if len(clipped) >= 3:
                polygon = clipped
        goal_hints = [g for g in goals if g['kind'] == 'box-front']
        goal_hints += [{'side': l['side'], 'kind': 'goal-line', 'a': l['a'], 'b': l['b']} for l in lines if l['side'] in ('left', 'right')]
        model = PitchModel(time, len(polygon) >= 3, aspect, polygon, grass_polygon, lines,
                           polygon_area(polygon) if len(polygon) >= 3 else 0.0, 'auto', goal_hints, segs,
                           vanishing=converging_end(tracked, goal_hints), curvature=line_curvature(segs, masks))
        self.previous = model
        return model


def warp_model(model, motion, time, reliable, cut):
    """Carry the last analysed model to the next frame with the camera motion (between analyses)."""
    extra = (model.confidence, model.mode, model.status, list(model.evidence), model.sides, model.vanishing, model.curvature)
    if cut or not reliable or motion is None:
        return PitchModel(time, False, model.aspect) if cut else PitchModel(time, model.reliable, model.aspect, model.polygon, model.grass_polygon, model.lines, model.coverage, model.source, model.goals, model.markings, *extra)
    move = lambda poly: clip_unit([(float(x), float(y)) for x, y in transform_points(motion, poly)]) if len(poly) >= 3 else []
    polygon = move(model.polygon)
    dt = max(0.0, time-model.time)
    lines = [dict(move_line(l, motion), age=l['age']+dt) for l in model.lines if l['age']+dt <= LINE_MAX_AGE]
    goals = [move_line(g, motion) for g in model.goals]
    markings = [move_line(m, motion) for m in model.markings]
    return PitchModel(time, model.reliable and len(polygon) >= 3, model.aspect, polygon, move(model.grass_polygon), lines,
                      polygon_area(polygon) if len(polygon) >= 3 else 0.0, model.source, goals, markings, *extra)


def calibrated_model(model, polygon, source='calibration', confidence=1.0, evidence=(), sides=None, status='', mode='automatic'):
    """Replace the playable polygon with the projected pitch rectangle of a calibration or the tracked
    pitch model (its off-screen edges are predicted, not read from the grass)."""
    polygon = clip_unit([tuple(p) for p in polygon]) if polygon is not None and len(polygon) >= 3 else []
    if len(polygon) < 3:
        return model
    return PitchModel(model.time, True, model.aspect, polygon, model.grass_polygon, model.lines, polygon_area(polygon), source, model.goals,
                      model.markings, confidence, mode, status, list(evidence), dict(sides) if sides else None, None, model.curvature)


def converging_end(tracked, goal_hints):
    """The side where the far and near touchlines meet inside or just beyond the frame while no goal line
    or box front was seen there. Two parallel touchlines converge towards a vanishing point in the image;
    that point is never where the pitch ends. The polygon may still taper there, but the end is marked
    unknown: no edge cue is measured against it, nobody is rejected beyond it, and the confidence of a
    region bounded by it cannot be HIGH."""
    far, near = tracked.get('far'), tracked.get('near')
    if far is None or near is None:
        return None
    (ax, ay), (bx, by) = far['a'], far['b']
    (cx, cy), (dx, dy) = near['a'], near['b']
    den = (ax-bx)*(cy-dy)-(ay-by)*(cx-dx)
    if abs(den) < 1e-12:
        return None
    x = ((ax*by-ay*bx)*(cx-dx)-(ax-bx)*(cx*dy-cy*dx))/den
    if not -.25 <= x <= 1.25:
        return None
    side = 'right' if x >= .5 else 'left'
    return None if any(g['side'] == side for g in goal_hints) else side


def line_curvature(segs, masks):
    """Lens-distortion check: the sagitta of the paint along the longest touchline-family line, in pixels
    per 1000 px of line length (a straight line gives ~0; several px means a wide-angle or stitched lens)."""
    best = max((s for s in segs if s['kind'] == 'h'), key=lambda s: abs(s['b'][0]-s['a'][0]), default=None)
    if best is None:
        return None
    w, h = masks['w'], masks['h']
    (ax, ay), (bx, by) = best['a'], best['b']
    length = abs(bx-ax)*w
    if length < .4*w:
        return None
    ys, xs = np.nonzero(masks['h_line'])
    if not len(xs):
        return None
    t = (xs-ax*w)/max(1e-9, (bx-ax)*w)
    expected = (ay+(by-ay)*t)*h
    near = (t >= 0) & (t <= 1) & (np.abs(ys-expected) <= 6)
    if near.sum() < 40:
        return None
    a, _, _ = np.polyfit(t[near], (ys[near]-expected[near]).astype(np.float64), 2)
    return round(float(abs(a)/4)/length*1000, 2)


def line_samples(segs, masks, max_lines=4, tol=6):
    """Paint pixels along the longest detected marking segments (up to max_lines of each kind), as lists of
    normalized (x, y) points: the raw material of the lens-distortion estimate (soccer/lens.py), which asks
    how bent these nearly straight lines are."""
    w, h = masks['w'], masks['h']
    out = []
    for kind in ('h', 'v'):
        mask = masks['h_line'] if kind == 'h' else masks['v_line']
        ys, xs = np.nonzero(mask)
        if not len(xs):
            continue
        ordered = sorted((s for s in segs if s['kind'] == kind), key=lambda s: -math.hypot((s['b'][0]-s['a'][0])*w, (s['b'][1]-s['a'][1])*h))
        for s in ordered[:max_lines]:
            ax, ay, bx, by = s['a'][0]*w, s['a'][1]*h, s['b'][0]*w, s['b'][1]*h
            dx, dy = bx-ax, by-ay
            length = math.hypot(dx, dy)
            if length < .25*max(w, h):
                continue
            t = ((xs-ax)*dx+(ys-ay)*dy)/(length*length)
            dist = np.abs((xs-ax)*dy-(ys-ay)*dx)/length
            near = (t >= 0) & (t <= 1) & (dist <= tol)
            if near.sum() < 30:
                continue
            out.append([(float(x/w), float(y/h)) for x, y in zip(xs[near], ys[near])])
    return out


def _off_segments(px, py, segs, w, h, halfway=None, tol=2.5):
    """Mask of the pixels (normalized coordinates) farther than tol pixels from every straight paint
    segment (within the segment's extent); the halfway line is excluded along its whole length."""
    keep = np.ones(len(px), bool)
    X, Y = px*w, py*h
    for seg in segs:
        ax, ay, bx, by = seg['a'][0]*w, seg['a'][1]*h, seg['b'][0]*w, seg['b'][1]*h
        dx, dy = bx-ax, by-ay
        length = math.hypot(dx, dy)
        if length < 1e-6:
            continue
        t = ((X-ax)*dx+(Y-ay)*dy)/length**2
        d = np.abs((X-ax)*dy-(Y-ay)*dx)/length
        extent = np.ones(len(px), bool) if seg is halfway else (t >= -.05) & (t <= 1.05)
        keep &= ~((d <= tol) & extent)
    return keep


def circle_points(masks, halfway, segs=(), max_points=1500, seed=0, iterations=300, min_density=None, centre_tolerance=None):
    """Centre-circle paint around the halfway line: ridge pixels that one ellipse explains. Candidate
    ellipses come from the connected arcs of the paint that is not on a straight segment (single arcs
    and unions of two or three, since players and the halfway line break the circle) and from random
    five-point samples; the one with most inliers is refit on them. Returns {'points': normalized (x, y)
    inliers, 'ellipse': ((cx, cy), (major, minor), angle) normalized, 'coverage': share of the ellipse's
    arc seen, 'inliers'} or None. A thin ellipse that only explains a straight line is rejected; with
    min_density, so is one whose inliers (all the window's, not the max_points subsample's) are spread
    thinner than that many per pixel of the arc seen, and
    with centre_tolerance one whose centre lies farther than that from the halfway line in the ellipse's
    own normalized coordinates (the ellipse a unit circle): the circle's centre is on the line, and its
    image is off the ellipse's centre only by the perspective shift (renders of broadcast, pole and tripod
    cameras: at most .07)."""
    if halfway is None:
        return None
    w, h = masks['w'], masks['h']
    ys, xs = np.nonzero(masks['ridge'])
    if len(xs) < 60:
        return None
    (ax, ay), (bx, by) = halfway['a'], halfway['b']
    lo, hi = min(ay, by), max(ay, by)
    mid_y = (lo+hi)/2
    x_mid = x_at(((ax, ay), (bx, by)), mid_y)
    px, py = xs/w, ys/h
    window = (np.abs(px-x_mid) <= .3) & (py >= lo-.05) & (py <= hi+.05)
    if window.sum() < 60:
        return None
    px, py = px[window], py[window]
    keep = _off_segments(px, py, [halfway]+[s for s in segs if s is not halfway], w, h, halfway, tol=3.5)
    px, py = px[keep], py[keep]
    if len(px) < 40:
        return None
    rng = np.random.default_rng(seed)
    window_pts = None
    if len(px) > max_points:
        if min_density is not None:
            window_pts = np.c_[px*w, py*h].astype(np.float32)   # the density is counted on every pixel, not on the subsample
        pick = rng.choice(len(px), max_points, replace=False)
        px, py = px[pick], py[pick]
    pts = np.c_[px*w, py*h].astype(np.float32)
    aspect = w/h

    def inliers_of(ellipse, tol=2.5, points=None):
        (cx, cy), (ma, mi), angle = ellipse
        if not np.isfinite([cx, cy, ma, mi, angle]).all() or min(ma, mi) < 8 or max(ma, mi) > .8*w or min(ma, mi)/max(ma, mi) < .07:
            return None   # a sliver along a painted line is not a circle, however many pixels it touches
        points = pts if points is None else points
        theta = math.radians(angle)
        dx, dy = points[:, 0]-cx, points[:, 1]-cy
        u = (dx*math.cos(theta)+dy*math.sin(theta))/(ma/2)
        v = (-dx*math.sin(theta)+dy*math.cos(theta))/(mi/2)
        r = np.hypot(u, v)
        return np.abs(r-1)*min(ma, mi)/2 <= tol

    candidates = []

    def try_fit(sample):
        if len(sample) < 5:
            return
        try:
            ellipse = cv2.fitEllipse(np.asarray(sample, np.float32))
        except cv2.error:
            return
        inl = inliers_of(ellipse)
        if inl is not None and inl.sum() >= 40:
            candidates.append((int(inl.sum()), ellipse))
    # Connected arcs of the remaining paint (the ridge mask at its own resolution).
    local = np.zeros((h, w), np.uint8)
    rows_, cols_ = np.clip(np.rint(pts[:, 1]).astype(int), 0, h-1), np.clip(np.rint(pts[:, 0]).astype(int), 0, w-1)
    local[rows_, cols_] = 1
    n, labels, stats, _ = cv2.connectedComponentsWithStats(local, connectivity=8)
    parts = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= 12]
    parts.sort(key=lambda i: -stats[i, cv2.CC_STAT_AREA])
    parts = parts[:10]
    members = {i: pts[labels[rows_, cols_] == i] for i in parts}
    for i in parts:
        try_fit(members[i])
    for a in range(len(parts)):
        for b in range(a+1, min(len(parts), 8)):
            try_fit(np.vstack([members[parts[a]], members[parts[b]]]))
            for c in range(b+1, min(len(parts), 6)):
                try_fit(np.vstack([members[parts[a]], members[parts[b]], members[parts[c]]]))
    for _ in range(iterations):
        try_fit(pts[rng.choice(len(pts), 5, replace=False)])
    candidates.sort(key=lambda c: -c[0])
    hx, hy = (halfway['b'][0]-halfway['a'][0])*w, (halfway['b'][1]-halfway['a'][1])*h

    def validate(ellipse):
        """Refit on the inliers and check that this is a circle's image around the halfway line."""
        inl = inliers_of(ellipse)
        if inl is None or inl.sum() < 40:
            return None
        try:
            refined = cv2.fitEllipse(pts[inl])
        except cv2.error:
            return None
        inl = inliers_of(refined)
        if inl is None or inl.sum() < 40:
            return None
        (cx, cy), (ma, mi), angle = refined
        # A very flat ellipse hugging a painted line explains that line, not a circle: the inliers must
        # spread away from their own best straight line, and the halfway line must cross the ellipse at
        # an angle rather than run along its long axis.
        sel = pts[inl]
        mean = sel.mean(axis=0)
        _, _, vt = np.linalg.svd(sel-mean, full_matrices=False)
        off = np.abs((sel-mean) @ vt[1])
        if float(np.sqrt(np.mean(off**2))) < 3.0:
            return None
        major = math.radians(angle) if ma >= mi else math.radians(angle)+math.pi/2
        between = abs(math.atan2(hy, hx)-major) % math.pi
        if min(between, math.pi-between) < math.radians(12):
            return None
        # The circle's centre lies on the halfway line (its image is close to the line); the arc must be long.
        if line_distance((cx/w, cy/h), halfway['a'], halfway['b'], aspect) > .12*max(ma, mi)/w*aspect+.01:
            return None
        if centre_tolerance is not None:
            t = math.radians(angle)
            ends = []
            for px_, py_ in ((halfway['a'][0]*w, halfway['a'][1]*h), (halfway['b'][0]*w, halfway['b'][1]*h)):
                dx, dy = px_-cx, py_-cy
                ends.append(((dx*math.cos(t)+dy*math.sin(t))/(ma/2), (-dx*math.sin(t)+dy*math.cos(t))/(mi/2)))
            (ux, uy), (vx, vy) = ends
            if abs(ux*vy-uy*vx)/max(math.hypot(vx-ux, vy-uy), 1e-9) > centre_tolerance:
                return None
        theta = math.radians(angle)
        dx, dy = pts[inl, 0]-cx, pts[inl, 1]-cy
        u = (dx*math.cos(theta)+dy*math.sin(theta))/(ma/2)
        v = (-dx*math.sin(theta)+dy*math.cos(theta))/(mi/2)
        bins = np.unique(np.floor((np.degrees(np.arctan2(v, u))+180)/15).astype(int))
        coverage = len(bins)/24
        if coverage < .33:
            return None
        if min_density is not None:
            a, b = ma/2, mi/2
            perimeter = math.pi*(3*(a+b)-math.sqrt((3*a+b)*(a+3*b)))   # Ramanujan
            # On the window's every pixel: on the subsample the count would shrink with the unrelated paint
            # around the circle (amateur 3: 10-28 % fewer), and clutter would decide the density.
            count = int(inl.sum()) if window_pts is None else int(inliers_of(refined, points=window_pts).sum())
            if count < min_density*coverage*perimeter:
                return None
        return {'points': [(float(x/w), float(y/h)) for x, y in pts[inl]], 'ellipse': ((cx/w, cy/h), (ma/w, mi/h), float(angle)),
                'coverage': round(coverage, 3), 'inliers': int(inl.sum())}
    for _, ellipse in candidates[:8]:
        found = validate(ellipse)
        if found is not None:
            return found
    return None


def circle_paint(masks, model, boxes=()):
    """The masks with the ridge restricted to paint a centre circle can be: on the grass (grass on both
    sides along one axis), inside the grass region and not at its top or bottom edge (boards, a running
    track, the next field), and outside people's boxes (white shirts). The straight paint stays: the near
    and far arcs of a distant circle are straight enough for the line detector, and removing them (as the
    shipped search does) left only the background to fit (amateur 3, 118-182 s)."""
    w, h = masks['w'], masks['h']
    region = np.zeros((h, w), np.uint8)
    if len(model.grass_polygon) >= 3:
        cv2.fillPoly(region, [np.int32(np.round(np.asarray(model.grass_polygon, np.float64)*[w, h]))], 1)
        k = max(1, int(round(CIRCLE_EDGE*h)))
        region = cv2.erode(region, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 2*k+1)))
    keep = masks['ridge'] & (masks['h_line'] | masks['v_line']) & region.astype(bool)
    for x, y, bw, bh in boxes or ():
        x0, y0 = int(math.floor(x*w))-1, int(math.floor(y*h))-1
        x1, y1 = int(math.ceil((x+bw)*w))+1, int(math.ceil((y+bh)*h))+1
        keep[max(0, y0):max(0, y1), max(0, x0):max(0, x1)] = False
    return dict(masks, ridge=keep)


def find_circle(masks, model, tries=3, on_grass=False, boxes=()):
    """The centre circle and the halfway line it is centred on: circle_points is tried around the likeliest
    halfway-line candidates (the longest well-supported paint segments not running flat across the frame),
    the first ellipse that passes its checks wins. Returns (circle, halfway segment) or (None, None).

    on_grass (the keyframe-less midfield anchor only): the search runs on circle_paint (people's boxes
    masked), keeps the straight paint and needs CIRCLE_MIN_DENSITY and CIRCLE_CENTRE_TOLERANCE."""
    if not masks:
        return None, None
    if on_grass:
        masks = circle_paint(masks, model, boxes)
    w, h = masks['w'], masks['h']
    candidates = []
    for seg in model.markings:
        if seg['support'] < .08:
            continue
        dx, dy = abs(seg['b'][0]-seg['a'][0])*w, abs(seg['b'][1]-seg['a'][1])*h
        if dy < .25*dx:
            continue   # runs flat across the frame: a touchline or box side from the side camera
        mid = (seg['a'][0]+seg['b'][0])/2
        if not .08 <= mid <= .92:
            continue
        candidates.append(seg)
    candidates.sort(key=lambda s: -s['support'])
    first = halfway_candidate(model)
    if first is not None and first in candidates:
        candidates.remove(first)
        candidates.insert(0, first)
    # On the grass, only paint that cannot be the circle is removed: the boundary lines and markings running
    # across more of the frame than a circle's arc does.
    lines = [{'a': l['a'], 'b': l['b']} for l in model.lines]+[m for m in model.markings if abs(m['b'][0]-m['a'][0]) > CIRCLE_MAX_SPAN] if on_grass else None
    for seg in candidates[:tries]:
        circle = (circle_points(masks, seg, lines, min_density=CIRCLE_MIN_DENSITY, centre_tolerance=CIRCLE_CENTRE_TOLERANCE) if on_grass
                  else circle_points(masks, seg, model.markings))
        if circle is not None:
            return circle, seg
    return None, None


def assess_confidence(model):
    """How far the image-plane detection can be trusted, with the evidence behind the score: both
    touchlines painted and found is a confident playable region; a grass contour alone (no painted
    boundary line) is not, since grass runs on beyond the touchline into run-offs, car parks and
    neighbouring pitches. Returns (confidence, evidence lines)."""
    if not model.reliable or len(model.polygon) < 3:
        return 0.0, ['no grass region large enough for a pitch']
    lines = {l['side']: l for l in model.lines}
    score, evidence = .2, []
    for side, what in (('far', 'touchline'), ('near', 'touchline'), ('left', 'goal line'), ('right', 'goal line')):
        line = lines.get(side)
        if line is None:
            evidence.append(f'{side} {what}: NOT FOUND (grass edge used)')
            continue
        gain = .3 if side in ('far', 'near') else .1
        if line['age'] > 0:
            gain *= max(.4, 1-line['age']/LINE_MAX_AGE)
            evidence.append(f'{side} {what}: FOUND (painted line carried {line["age"]:.1f}s)')
        else:
            evidence.append(f'{side} {what}: FOUND (painted line, support {line["support"]:.2f})')
        score += gain
    if model.coverage < .1:
        score = min(score, .3)
        evidence.append('the region is small')
    if model.vanishing:
        score = min(score, MEDIUM-.01)
        evidence.append(f'{model.vanishing} end: the touchlines converge in view with no goal line or box front there (a vanishing point, not the pitch end): not HIGH')
    return min(.95, score), evidence


def halfway_candidate(model):
    """The painted steep line most likely to be the halfway line (the longest one well inside the
    frame), as a marking segment, or None."""
    best = None
    for seg in model.markings:
        if seg['kind'] != 'v' or seg['support'] < .12:
            continue
        mid = (seg['a'][0]+seg['b'][0])/2
        if not .12 <= mid <= .88:
            continue
        if best is None or seg['support'] > best['support']:
            best = seg
    return best


def suggest_landmarks(model):
    """Landmarks worth marking on this frame to calibrate it, with predicted positions where the
    image gives them: the halfway line's ends where it meets the grass edges. The centre circle and
    spot are listed without positions."""
    out = []
    line = halfway_candidate(model)
    if line is not None and len(model.grass_polygon) >= 3:
        seg = (line['a'], line['b'])
        inside = [y for y in np.linspace(0, 1, 201) if 0 <= x_at(seg, y) <= 1 and point_in_polygon((x_at(seg, y), y), model.grass_polygon)]
        # An end is predicted only where the grass ends inside the frame: a touchline out of view
        # (the near one on a low camera) must not be placed at the image border.
        if inside and inside[0] > .03:
            out.append({'name': 'halfway-far', 'x': round(float(x_at(seg, inside[0])), 4), 'y': round(float(inside[0]), 4), 'predicted': True})
        if inside and inside[-1] < .97:
            out.append({'name': 'halfway-near', 'x': round(float(x_at(seg, inside[-1])), 4), 'y': round(float(inside[-1]), 4), 'predicted': True})
    out += [{'name': name, 'predicted': False} for name in ('centre-spot', 'centre-circle-left', 'centre-circle-right', 'centre-circle-far', 'centre-circle-near')]
    return out


def _apex(model):
    """The polygon vertex at the converging end, or None."""
    if not model.vanishing or len(model.polygon) < 3:
        return None
    return max(model.polygon, key=lambda q: q[0]) if model.vanishing == 'right' else min(model.polygon, key=lambda q: q[0])


def _at_apex(model, p, a, b):
    """Whether the closest point of segment (a, b) to p is its end at the converging apex, or p lies
    beyond the apex along the pitch."""
    apex = _apex(model)
    if apex is None:
        return False
    aspect = model.aspect
    ax, ay, bx, by, px, py = a[0]*aspect, a[1], b[0]*aspect, b[1], p[0]*aspect, p[1]
    length2 = (bx-ax)**2+(by-ay)**2
    t = ((px-ax)*(bx-ax)+(py-ay)*(by-ay))/length2 if length2 > 1e-12 else 0.0
    end = a if t <= 0 else b if t >= 1 else None
    if end is not None and abs(end[0]-apex[0]) < 1e-9 and abs(end[1]-apex[1]) < 1e-9:
        return True
    return (p[0] >= apex[0]-.02) if model.vanishing == 'right' else (p[0] <= apex[0]+.02)


def nearest_boundary(model, p):
    poly, aspect = model.polygon, model.aspect
    if len(poly) < 3:
        return None
    best, index = math.inf, 0
    for i in range(len(poly)):
        d = segment_distance(p, poly[i], poly[(i+1) % len(poly)], aspect)
        if d < best:
            best, index = d, i
    a, b = poly[index], poly[(index+1) % len(poly)]
    c = centroid(poly)
    nx, ny = b[1]-a[1], -(b[0]-a[0])*aspect
    length = math.hypot(nx, ny) or 1
    nx, ny = nx/length, ny/length
    if nx*(c[0]-a[0])*aspect+ny*(c[1]-a[1]) > 0:
        nx, ny = -nx, -ny
    line = next((l for l in model.lines if line_distance(l['a'], a, b, aspect) < .006 and line_distance(l['b'], a, b, aspect) < .006), None)
    side = line['side'] if line else 'far' if ny < -.6 else 'near' if ny > .6 else 'left' if nx < 0 else 'right'
    if model.vanishing and _at_apex(model, p, a, b):
        # The nearest point of the region is where the touchlines converge: a vanishing point, not an edge.
        return {'distance': -best if point_in_polygon(p, poly) else best, 'side': model.vanishing, 'label': 'the converging end (a vanishing point, not a pitch edge)',
                'line': None, 'a': a, 'b': b}
    if line:
        label = f'{side} touchline' if _horizontal(side) else f'{side} goal line'
    elif model.source == 'calibration':
        label = f'{side} pitch edge'
    else:
        label = f'{side} grass edge'
    return {'distance': -best if point_in_polygon(p, poly) else best, 'side': side, 'label': label, 'line': line, 'a': a, 'b': b}


def zone_of(model, box, config):
    """'inside' (foot on the pitch), 'boundary' (within the touchline tolerance), 'outside', or
    'unknown' (no reliable pitch on this frame). Returns (zone, signed distance)."""
    if not config.enabled:
        return 'inside', 0.0
    if not model.reliable or len(model.polygon) < 3:
        return 'unknown', 0.0
    d = signed_distance(foot_point(box), model.polygon, model.aspect)
    if d <= 0:
        return 'inside', d
    margin = min(config.max_margin, max(config.min_margin, config.boundary_margin*box[3]))
    if d > 0 and model.vanishing:
        # Towards the converging end the pitch is not known to end: that is a vanishing point, not a line.
        edge = nearest_boundary(model, foot_point(box))
        if edge is not None and edge['side'] == model.vanishing:
            return 'unknown', d
    return ('boundary' if d <= margin else 'outside'), d


def zone_in_polygon(polygon, aspect, box, config):
    """Zone against a user-drawn boundary polygon (pixel-free, normalized)."""
    d = signed_distance(foot_point(box), polygon, aspect)
    if d <= 0:
        return 'inside', d
    margin = min(config.max_margin, max(config.min_margin, config.boundary_margin*box[3]))
    return ('boundary' if d <= margin else 'outside'), d
