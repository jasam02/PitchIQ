"""Geometry for the key-moments detectors: time to reach a point, zones, what the camera saw, lines and gaps.

Time to reach (REACT_S, VMAX): a player keeps moving with their current velocity for REACT_S, then runs straight at VMAX
(the simple model of Spearman's pitch control, 2017, with its reaction time). It decides WHO GETS THERE FIRST between
players of both teams, always with the same model, so its error matters less than its consistency. VMAX = 7.0 m/s:
on CAN-ISL the smoothed speeds were p99 6.6 m/s and max 8.5 m/s over 414,524 observations (soccer/moments/data.py), so a
player at full sprint is credited with slightly less than their best; nobody gets there faster than VMAX.
"""
import math

import numpy as np

from . import data as md

REACT_S = .5
VMAX = 7.0


def time_to_reach(px, py, vx, vy, tx, ty, react=REACT_S, vmax=VMAX):
    """Seconds for players at (px, py) moving (vx, vy) to reach (tx, ty); NaN velocities count as standing."""
    vx = np.nan_to_num(np.asarray(vx, float))
    vy = np.nan_to_num(np.asarray(vy, float))
    ex = np.asarray(px, float)+vx*react
    ey = np.asarray(py, float)+vy*react
    return react+np.hypot(tx-ex, ty-ey)/vmax


def disc_points(cx, cy, radius, step=3.0):
    """Points of a disc (a grid every `step` metres) for visibility checks."""
    r = np.arange(-radius, radius+1e-9, step)
    gx, gy = np.meshgrid(r, r)
    keep = gx**2+gy**2 <= radius**2
    return cx+gx[keep], cy+gy[keep]


def visible_share(view, px, py, length, width):
    """The share of the points (clipped to the pitch) inside the camera footprint; points off the pitch are ignored."""
    px, py = np.asarray(px, float), np.asarray(py, float)
    on = (px >= 0) & (px <= length) & (py >= 0) & (py <= width)
    if not on.any():
        return 0.0
    inside = md.footprint_contains(view, px[on], py[on])
    return float(inside.mean())


def rect_in_team(ctx, s, depth0, depth1, lat0, lat1):
    """A rectangle given in team coordinates (depth from own goal, lateral to the left) as a pitch polygon [(x, y)]."""
    pts = []
    for d, l in ((depth0, lat0), (depth1, lat0), (depth1, lat1), (depth0, lat1)):
        x, y = ctx.to_pitch(d, l, s)
        pts.append((float(np.clip(x, 0, ctx.data.length)), float(np.clip(y, 0, ctx.data.width))))
    return pts


def polygon_points(poly, step=2.0):
    """Grid points inside a convex polygon (pitch metres)."""
    p = np.asarray(poly, float)
    x0, y0 = p.min(0)
    x1, y1 = p.max(0)
    gx, gy = np.meshgrid(np.arange(x0+step/2, x1, step), np.arange(y0+step/2, y1, step))
    if not gx.size:
        return np.zeros(0), np.zeros(0)
    from matplotlib.path import Path as MPath
    inside = MPath(p).contains_points(np.column_stack([gx.ravel(), gy.ravel()]))
    return gx.ravel()[inside], gy.ravel()[inside]


def point_segment_distance(px, py, ax, ay, bx, by):
    """Distance of point(s) P from segment AB."""
    abx, aby = bx-ax, by-ay
    L2 = abx*abx+aby*aby
    t = np.clip(((px-ax)*abx+(py-ay)*aby)/L2, 0, 1) if L2 > 0 else 0.0
    return np.hypot(px-(ax+t*abx), py-(ay+t*aby))


def lane_blocked(ox, oy, ax, ay, bx, by, width=2.5, min_along=1.0):
    """Whether any of the points (ox, oy) blocks the straight pass from A to B: in front along it (between min_along
    metres from A and B) and within `width` of it. A defender beside the passer (a presser standing next to the ball)
    does not block a pass the other way; one in the line does (the plain distance-to-segment test counted every presser
    as blocking every lane, found by a synthetic test)."""
    ox, oy = np.asarray(ox, float), np.asarray(oy, float)
    if not ox.size:
        return False
    dx, dy = bx-ax, by-ay
    length = math.hypot(dx, dy)
    if length < 1e-6:
        return False
    ux, uy = dx/length, dy/length
    along = (ox-ax)*ux+(oy-ay)*uy
    perp = np.abs(-(ox-ax)*uy+(oy-ay)*ux)
    return bool(np.any((along >= min_along) & (along <= length) & (perp <= width)))


def angle_between(ux, uy, vx, vy):
    """Unsigned angle (degrees) between vectors u and v."""
    nu, nv = math.hypot(ux, uy), math.hypot(vx, vy)
    if nu == 0 or nv == 0:
        return 180.0
    c = max(-1.0, min(1.0, (ux*vx+uy*vy)/(nu*nv)))
    return math.degrees(math.acos(c))


def norm_polygon(poly, length, width):
    """A pitch polygon in normalized coordinates (what the page's diagrams take), rounded."""
    return [[round(float(x)/length, 4), round(float(y)/width, 4)] for x, y in poly]
