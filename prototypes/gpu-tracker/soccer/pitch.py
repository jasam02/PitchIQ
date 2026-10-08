"""Playable-field detection, recomputed for every analysed frame.

The legal field is defined by its white boundary lines, not by green grass. White paint is found as
a thin bright ridge on the grass (morphological top-hat at 640 px width), so even a thin, blurred far
touchline is seen. Long straight lines are fitted, and for each side of the pitch the outermost line
that is painted on grass (grass on both sides, a run-off strip beyond it) becomes the boundary:
advertising-board edges have no grass beyond them and are never boundaries. A steep line with short
lines leaving it towards the image edge is a penalty/goal-area front, not a goal line. Boundary lines
are tracked over time (smoothed, carried through camera motion, changed only when a new position is
confirmed), and the grass region is only the fallback for sides where no boundary line is visible.

Image-plane only. Metric pitch coordinates come from calibration.py.
"""
import math
from dataclasses import dataclass, field

import cv2
import numpy as np

from .geometry import (centroid, clip_polygon, clip_unit, convex_hull, foot_point, line_distance,
                       point_in_polygon, polygon_area, segment_distance, signed_distance,
                       transform_points, x_at, y_at)

MIN_GRASS = .12          # grass share of the frame needed for a reliable pitch
LINE_MAX_AGE = 3.0       # seconds a boundary line is carried with a reliable camera estimate
LINE_MAX_AGE_LOOSE = .8  # ... when the camera estimate is unreliable
FINE_W = 320             # grass sampling grid width (cells)
LINE_W = 640             # paint / line analysis width (pixels)


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
    source: str = 'none'                               # auto | calibration | none
    goals: list = field(default_factory=list)          # goal-end hints [{'side','kind': goal-line|box-front,'a','b'}]
    markings: list = field(default_factory=list)       # all fitted paint lines (debug)

    def sources(self):
        """Where each side of the polygon comes from: a detected line, a carried line, or the grass edge."""
        out = {side: 'grass edge' for side in ('far', 'near', 'left', 'right')}
        if self.source == 'calibration':
            return {side: 'calibration' for side in out}
        for l in self.lines:
            out[l['side']] = 'line' if l['age'] <= 0 else 'carried line'
        return out

    def to_json(self):
        r = lambda p: [round(float(p[0]), 4), round(float(p[1]), 4)]
        return {'reliable': self.reliable, 'source': self.source, 'coverage': round(self.coverage, 3),
                'polygon': [r(p) for p in self.polygon], 'grass': [r(p) for p in self.grass_polygon],
                'lines': [{'a': r(l['a']), 'b': r(l['b']), 'side': l['side'], 'age': round(l['age'], 2)} for l in self.lines],
                'goals': [{'side': g['side'], 'kind': g['kind'], 'a': r(g['a']), 'b': r(g['b'])} for g in self.goals],
                'markings': [[*r(m['a']), *r(m['b'])] for m in self.markings[:16]], 'edges': self.sources()}


# ---------- sampling ----------
def _grass(r, g, b):
    # Accepts worn / yellow-green turf but not sand, concrete, sky or paint.
    return (g > 30) & (g > r*.95) & (g > b*1.12) & (g-b > 12)


def line_masks(frame):
    """Grass and white-paint masks at <= 640 px width. Paint is a thin bright ridge: the top-hat of the
    darkest colour channel (white is bright in all three, grass is dark in red and blue), so a 1-2 px
    far touchline blended with grass still stands out, while wide white advertising boards do not."""
    h, w = frame.shape[:2]
    fw = min(FINE_W, w)
    fh = max(2, min(h, round(fw*h/w)))
    small = cv2.resize(frame, (fw*2, fh*2), interpolation=cv2.INTER_AREA)
    b8, g8_, r8 = cv2.split(small)
    b, g, r = b8.astype(np.int16), g8_.astype(np.int16), r8.astype(np.int16)
    grass = _grass(r, g, b)
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
                           polygon_area(polygon) if len(polygon) >= 3 else 0.0, 'auto', goal_hints, segs)
        self.previous = model
        return model


def warp_model(model, motion, time, reliable, cut):
    """Carry the last analysed model to the next frame with the camera motion (between analyses)."""
    if cut or not reliable or motion is None:
        return PitchModel(time, False, model.aspect) if cut else PitchModel(time, model.reliable, model.aspect, model.polygon, model.grass_polygon, model.lines, model.coverage, model.source, model.goals, model.markings)
    move = lambda poly: clip_unit([(float(x), float(y)) for x, y in transform_points(motion, poly)]) if len(poly) >= 3 else []
    polygon = move(model.polygon)
    dt = max(0.0, time-model.time)
    lines = [dict(move_line(l, motion), age=l['age']+dt) for l in model.lines if l['age']+dt <= LINE_MAX_AGE]
    goals = [move_line(g, motion) for g in model.goals]
    markings = [move_line(m, motion) for m in model.markings]
    return PitchModel(time, model.reliable and len(polygon) >= 3, model.aspect, polygon, move(model.grass_polygon), lines,
                      polygon_area(polygon) if len(polygon) >= 3 else 0.0, model.source, goals, markings)


def calibrated_model(model, polygon):
    """Replace the playable polygon with the projected calibrated pitch outline."""
    polygon = clip_unit([tuple(p) for p in polygon]) if polygon is not None and len(polygon) >= 3 else []
    if len(polygon) < 3:
        return model
    return PitchModel(model.time, True, model.aspect, polygon, model.grass_polygon, model.lines, polygon_area(polygon), 'calibration', model.goals, model.markings)


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
    if line:
        label = f'{side} touchline' if _horizontal(side) else f'{side} goal line'
    elif model.source == 'calibration':
        label = f'{side} pitch edge'
    else:
        label = f'{side} grass edge'
    return {'distance': -best if point_in_polygon(p, poly) else best, 'side': side, 'label': label, 'line': line}


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
    return ('boundary' if d <= margin else 'outside'), d


def zone_in_polygon(polygon, aspect, box, config):
    """Zone against a user-drawn boundary polygon (pixel-free, normalized)."""
    d = signed_distance(foot_point(box), polygon, aspect)
    if d <= 0:
        return 'inside', d
    margin = min(config.max_margin, max(config.min_margin, config.boundary_margin*box[3]))
    return ('boundary' if d <= margin else 'outside'), d
