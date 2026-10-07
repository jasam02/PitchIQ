"""Playable-field detection, recomputed for every analysed frame.

Grass segmentation (largest connected grass region, painted lines bridged,
advertising boards not bridged) gives a grass hull. Long white boundary lines
(touchlines, goal lines) with only a thin grass run-off band beyond them clip
that hull, so a coach standing on the run-off grass is outside the pitch.
A boundary line that briefly disappears is carried with the camera motion for
up to two seconds. Nothing here is a permanent rectangle: the region follows
pans and zooms because it is re-detected from the image itself.

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
LINE_MAX_AGE = 2.0       # seconds a boundary line is carried with a reliable camera estimate
LINE_MAX_AGE_LOOSE = .6  # ... when the camera estimate is unreliable
FINE_W = 320             # sampling grid width (cells)


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
    lines: list = field(default_factory=list)          # [{'a','b','side','support','age'}]
    coverage: float = 0.0
    source: str = 'none'                               # auto | calibration | none

    def to_json(self):
        r = lambda p: [round(float(p[0]), 4), round(float(p[1]), 4)]
        return {'reliable': self.reliable, 'source': self.source, 'coverage': round(self.coverage, 3),
                'polygon': [r(p) for p in self.polygon], 'grass': [r(p) for p in self.grass_polygon],
                'lines': [{'a': r(l['a']), 'b': r(l['b']), 'side': l['side'], 'age': round(l['age'], 2)} for l in self.lines]}


# ---------- sampling ----------
def _grass(r, g, b):
    # Accepts worn / yellow-green turf but not sand, concrete, sky or paint.
    return (g > 30) & (g > r*.95) & (g > b*1.12) & (g-b > 12)


def sample_frame(frame):
    """Per-cell grass fraction, white paint and dark (letterbox) flags on a <=320 wide grid."""
    h, w = frame.shape[:2]
    fw = min(FINE_W, w)
    fh = max(2, min(h, round(fw*h/w)))
    small = cv2.resize(frame, (fw*2, fh*2), interpolation=cv2.INTER_AREA).astype(np.int16)
    b, g, r = small[..., 0], small[..., 1], small[..., 2]
    grass = _grass(r, g, b)
    lo, hi = np.minimum(np.minimum(r, g), b), np.maximum(np.maximum(r, g), b)
    white = (lo >= 110) & (hi-lo <= 70) & (r+g+b >= 380) & ~grass
    pool = lambda mask: cv2.resize(mask.astype(np.float32), (fw, fh), interpolation=cv2.INTER_AREA)
    return {'w': fw, 'h': fh, 'frac': pool(grass), 'white': pool(white) > 1e-3, 'dark': pool(hi < 28) > 1-1e-3}


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
def hough(U, V, u_len, v_len, max_slope, step, bin_w, max_lines, min_support):
    """Lines through points (u along, v across) with |dv/du| <= max_slope. OpenCV's Hough transform
    proposes peaks; each is refined by least squares on the remaining points, and accepted only when
    enough distinct u positions (grid columns or rows) support it, so thick lines are not
    over-counted. Inliers are removed before the next peak."""
    U, V = np.asarray(U, np.int64), np.asarray(V, np.float64)
    out = []
    if len(U) < min_support:
        return out
    mask = np.zeros((int(v_len)+1, int(u_len)+1), np.uint8)
    mask[np.clip(V.astype(np.int64), 0, int(v_len)), np.clip(U, 0, int(u_len))] = 255
    low = math.atan2(1, max_slope)
    peaks = cv2.HoughLines(mask, 1, np.pi/360, max(2, int(min_support*.6)), min_theta=low, max_theta=math.pi-low)
    if peaks is None:
        return out
    uc = u_len/2
    alive = np.ones(len(U), bool)
    for rho, theta in peaks[:40, 0]:
        if len(out) >= max_lines or alive.sum() < min_support:
            break
        sin, cos = math.sin(theta), math.cos(theta)
        if abs(sin) < 1e-6:
            continue
        s, c = -cos/sin, (rho-uc*cos)/sin
        u, v = U[alive], V[alive]
        du = u-uc
        for tol in (bin_w+1, 1.5):
            sel = np.abs(v-(c+s*du)) <= tol
            n = int(sel.sum())
            su, sv, suu, suv = du[sel].sum(), v[sel].sum(), (du[sel]**2).sum(), (du[sel]*v[sel]).sum()
            den = n*suu-su*su
            if n < 2 or den <= 1e-9:
                break
            s = (n*suv-su*sv)/den
            c = (sv-s*su)/n
        residual = np.abs(V-(c+s*(U-uc)))
        inliers = np.sort(U[alive & (residual <= 1.5)])
        support = len(np.unique(inliers))
        if support < min_support or abs(s) > max_slope*1.15:
            continue
        alive &= residual > 3
        u0 = inliers[int(len(inliers)*.02)]
        u1 = inliers[max(0, math.ceil(len(inliers)*.98)-1)]
        out.append({'u0': float(u0), 'u1': float(u1), 'v0': c+s*(u0-uc), 'v1': c+s*(u1-uc), 'support': support})
    return out


def detect_lines(f):
    """Long straight white lines with grass on both sides: near-horizontal (touchline family) and
    steep (goal-line family). Interior lines are found too and rejected by choose_boundaries."""
    w, h = f['w'], f['h']
    grass, white = f['frac'] >= .5, f['white']
    d = max(2, round(h/60))
    def near(dy, dx):
        hit = np.zeros_like(grass)
        for k in range(d, 2*d+1):
            hit |= _shift(grass, dy*k, dx*k)
        return hit
    hy, hx = np.nonzero(white & near(-1, 0) & near(1, 0))
    vy, vx = np.nonzero(white & near(0, -1) & near(0, 1))
    segs = []
    for l in hough(hx, hy, w, h, .36, .02, 2, 4, .2*w):
        segs.append({'kind': 'h', 'a': ((l['u0']+.5)/w, (l['v0']+.5)/h), 'b': ((l['u1']+.5)/w, (l['v1']+.5)/h), 'support': l['support']/w})
    for l in hough(vy, vx, h, w, 2.9, .05, 3, 4, .15*h):
        segs.append({'kind': 'v', 'a': ((l['v0']+.5)/w, (l['u0']+.5)/h), 'b': ((l['v1']+.5)/w, (l['u1']+.5)/h), 'support': l['support']/h})
    return segs


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


def edge_line(line, side, r, f):
    """Share of the line's extent where thin paint runs along the outer grass edge (with no white
    further out, so not a white board): the real boundary is there and this line is interior."""
    k = f['w']//r['w']
    horizontal, out = _horizontal(side), 1 if side in ('near', 'right') else -1
    white = f['white']
    def is_white(along, across):
        x, y = (along, across) if horizontal else (across, along)
        return 0 <= x < f['w'] and 0 <= y < f['h'] and bool(white[y, x])
    a, b = line['a'], line['b']
    unit = r['cw'] if horizontal else r['ch']
    lo = max(0, int(min(a[0] if horizontal else a[1], b[0] if horizontal else b[1])/unit))
    hi = min((r['w'] if horizontal else r['h'])-1, int(max(a[0] if horizontal else a[1], b[0] if horizontal else b[1])/unit))
    n = hit = 0
    for c in range(lo, hi+1):
        e = (r['top'][c] if side == 'far' else r['bottom'][c]) if horizontal else (r['left'][c] if side == 'left' else r['right'][c])
        if e < 0:
            continue
        n += 1
        edge = (e+1)*k-1 if out > 0 else e*k
        at = lambda dist: any(is_white(c*k+i, edge+out*dist) for i in range(k))
        if any(at(dist) for dist in (-1, 0, 1, 2, 3)) and not any(at(dist) for dist in (5, 6, 7)):
            hit += 1
    return hit/n if n else 0.0


def choose_boundaries(segs, r, aspect, f):
    """A line is a boundary only if it is outermost on its side and the grass beyond it is a thin
    run-off band. Short lines (technical-area markings) are skipped; a failing line fails for every
    line inside it too."""
    c = (r['cx'], r['cy'])
    by_side = {}
    for seg in segs:
        horizontal = seg['kind'] == 'h'
        span = abs(seg['b'][0]-seg['a'][0]) if horizontal else abs(seg['b'][1]-seg['a'][1])
        if (seg['support'] < .3 or span < .5) if horizontal else (seg['support'] < .2 or span < .3):
            continue
        side = side_of(seg, horizontal, c)
        by_side.setdefault(side, []).append((line_distance(c, seg['a'], seg['b'], aspect), seg))
    found = {}
    for side, items in by_side.items():
        for _, seg in sorted(items, key=lambda item: -item[0]):
            s = band_stats(seg, side, r)
            if not s['count'] or s['on'] < .5:
                continue
            if s['beyond'] >= .25*s['inner']:
                break
            interior = s['depth'] > .06 and edge_line(seg, side, r, f) >= .5
            if side == 'near':
                fail = s['term'] < .5 and s['depth'] > .12
            else:
                fail = s['term'] < .5 or interior
            if fail:
                break
            found[side] = {'a': seg['a'], 'b': seg['b'], 'side': side, 'support': seg['support'], 'age': 0.0}
            break
    return found


def move_line(line, matrix):
    if matrix is None:
        return dict(line)
    a, b = transform_points(matrix, [line['a'], line['b']])
    return dict(line, a=(float(a[0]), float(a[1])), b=(float(b[0]), float(b[1])))


class PitchDetector:
    """Re-detects the playable region on every call; carries boundary lines across short gaps."""

    def __init__(self):
        self.previous = None

    def reset(self):
        self.previous = None

    def analyze(self, frame, time, motion=None, reliable=False, cut=False):
        """motion: normalized 3x3 transform from the previous analysed frame to this one."""
        h, w = frame.shape[:2]
        aspect = w/max(1, h)
        empty = PitchModel(time, False, aspect)
        if w < 16 or h < 16:
            self.previous = empty
            return empty
        f = sample_frame(frame)
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
        found = choose_boundaries(detect_lines(f), r, aspect, f)
        previous = self.previous
        if previous is not None and previous.lines and not cut:
            dt = max(0.0, time-previous.time)
            max_age = LINE_MAX_AGE if reliable else LINE_MAX_AGE_LOOSE
            for old in previous.lines:
                if old['side'] in found or old['age']+dt > max_age:
                    continue
                moved = move_line(old, motion if reliable else None)
                if side_of(moved, _horizontal(old['side']), (r['cx'], r['cy'])) != old['side']:
                    continue
                s = band_stats(moved, old['side'], r)
                if s['beyond'] >= .35*s['inner']:
                    continue  # no longer consistent with the grass
                found[old['side']] = dict(moved, age=old['age']+dt)
        lines = list(found.values())
        polygon = grass_polygon
        c = (r['cx'], r['cy'])
        for line in lines:
            clipped = clip_polygon(polygon, line['a'], line['b'], c)
            if len(clipped) >= 3:
                polygon = clipped
        model = PitchModel(time, len(polygon) >= 3, aspect, polygon, grass_polygon, lines, polygon_area(polygon) if len(polygon) >= 3 else 0.0, 'auto')
        self.previous = model
        return model


def warp_model(model, motion, time, reliable, cut):
    """Carry the last analysed model to the next frame with the camera motion (between analyses)."""
    if cut or not reliable or motion is None:
        return PitchModel(time, False, model.aspect) if cut else PitchModel(time, model.reliable, model.aspect, model.polygon, model.grass_polygon, model.lines, model.coverage, model.source)
    move = lambda poly: clip_unit([(float(x), float(y)) for x, y in transform_points(motion, poly)]) if len(poly) >= 3 else []
    polygon = move(model.polygon)
    dt = max(0.0, time-model.time)
    lines = [dict(move_line(l, motion), age=l['age']+dt) for l in model.lines if l['age']+dt <= LINE_MAX_AGE]
    return PitchModel(time, model.reliable and len(polygon) >= 3, model.aspect, polygon, move(model.grass_polygon), lines,
                      polygon_area(polygon) if len(polygon) >= 3 else 0.0, model.source)


def calibrated_model(model, polygon):
    """Replace the playable polygon with the projected calibrated pitch outline."""
    polygon = clip_unit([tuple(p) for p in polygon]) if polygon is not None and len(polygon) >= 3 else []
    if len(polygon) < 3:
        return model
    return PitchModel(model.time, True, model.aspect, polygon, model.grass_polygon, model.lines, polygon_area(polygon), 'calibration')


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
