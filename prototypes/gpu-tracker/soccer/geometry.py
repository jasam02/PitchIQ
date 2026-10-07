"""Small geometry helpers. Points are normalized image coordinates [0,1]."""
import math

import numpy as np


def clamp(value, low=0.0, high=1.0):
    return low if value < low else high if value > high else value


def iou(a, b):
    """IoU of two [x, y, w, h] boxes."""
    ix = max(0.0, min(a[0]+a[2], b[0]+b[2])-max(a[0], b[0]))
    iy = max(0.0, min(a[1]+a[3], b[1]+b[3])-max(a[1], b[1]))
    inter = ix*iy
    union = a[2]*a[3]+b[2]*b[3]-inter
    return inter/union if union > 0 else 0.0


def foot_point(box):
    """Bottom-centre of [x, y, w, h]: where the person touches the ground."""
    return (box[0]+box[2]/2, box[1]+box[3])


def edge_of(box, tolerance=0.004):
    """Which image edge a box touches ('' when none). Used for exit/entry."""
    x, y, w, h = box
    if x <= tolerance:
        return 'left'
    if x+w >= 1-tolerance:
        return 'right'
    if y <= tolerance:
        return 'top'
    if y+h >= 1-tolerance:
        return 'bottom'
    return ''


def cross(o, a, b):
    return (a[0]-o[0])*(b[1]-o[1])-(a[1]-o[1])*(b[0]-o[0])


def convex_hull(points):
    pts = sorted(set((float(x), float(y)) for x, y in points))
    if len(pts) < 3:
        return pts
    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 1e-12:
            lower.pop()
        lower.append(p)
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 1e-12:
            upper.pop()
        upper.append(p)
    return lower[:-1]+upper[:-1]


def polygon_area(poly):
    s = 0.0
    for i in range(len(poly)):
        a, b = poly[i], poly[(i+1) % len(poly)]
        s += a[0]*b[1]-b[0]*a[1]
    return abs(s)/2


def centroid(poly):
    return (sum(p[0] for p in poly)/len(poly), sum(p[1] for p in poly)/len(poly))


def clip_polygon(poly, a, b, keep):
    """Part of a convex polygon on the same side of line a-b as point `keep`."""
    sign = 1 if cross(a, b, keep) >= 0 else -1
    out = []
    for i in range(len(poly)):
        p, q = poly[i], poly[(i+1) % len(poly)]
        fp, fq = sign*cross(a, b, p), sign*cross(a, b, q)
        if fp >= 0:
            out.append(p)
        if (fp >= 0) != (fq >= 0):
            t = fp/(fp-fq)
            out.append((p[0]+(q[0]-p[0])*t, p[1]+(q[1]-p[1])*t))
    return out


def clip_unit(poly):
    centre = (.5, .5)
    for a, b in (((0, 0), (1, 0)), ((1, 0), (1, 1)), ((1, 1), (0, 1)), ((0, 1), (0, 0))):
        if len(poly) < 3:
            return []
        poly = clip_polygon(poly, a, b, centre)
    return poly if len(poly) >= 3 else []


def point_in_polygon(p, poly):
    inside = False
    j = len(poly)-1
    for i in range(len(poly)):
        a, b = poly[i], poly[j]
        if (a[1] > p[1]) != (b[1] > p[1]) and p[0] < (b[0]-a[0])*(p[1]-a[1])/(b[1]-a[1])+a[0]:
            inside = not inside
        j = i
    return inside


def segment_distance(p, a, b, aspect):
    """Distance from p to segment a-b in frame-height units (x scaled by aspect)."""
    ax, bx, px = a[0]*aspect, b[0]*aspect, p[0]*aspect
    dx, dy = bx-ax, b[1]-a[1]
    length = dx*dx+dy*dy
    t = clamp(((px-ax)*dx+(p[1]-a[1])*dy)/length) if length > 0 else 0.0
    return math.hypot(px-ax-t*dx, p[1]-a[1]-t*dy)


def line_distance(p, a, b, aspect):
    dx, dy = (b[0]-a[0])*aspect, b[1]-a[1]
    length = math.hypot(dx, dy)
    if length <= 0:
        return math.inf
    return abs(dx*(p[1]-a[1])-dy*(p[0]-a[0])*aspect)/length


def signed_distance(p, poly, aspect):
    """Negative inside the polygon, positive outside (frame-height units)."""
    if len(poly) < 3:
        return math.inf
    best = min(segment_distance(p, poly[i], poly[(i+1) % len(poly)], aspect) for i in range(len(poly)))
    return -best if point_in_polygon(p, poly) else best


def y_at(line, x):
    (ax, ay), (bx, by) = line
    return ay if abs(bx-ax) < 1e-9 else ay+(by-ay)*(x-ax)/(bx-ax)


def x_at(line, y):
    (ax, ay), (bx, by) = line
    return ax if abs(by-ay) < 1e-9 else ax+(bx-ax)*(y-ay)/(by-ay)


def transform_points(matrix, points):
    """Apply a 3x3 projective transform to Nx2 points."""
    pts = np.asarray(points, np.float64).reshape(-1, 2)
    if not len(pts):
        return pts
    homogeneous = np.c_[pts, np.ones(len(pts))] @ np.asarray(matrix, np.float64).T
    w = homogeneous[:, 2:3]
    w[np.abs(w) < 1e-12] = 1e-12
    return homogeneous[:, :2]/w


def scale_matrix(sx, sy):
    return np.array([[sx, 0, 0], [0, sy, 0], [0, 0, 1]], np.float64)


def to_normalized(matrix_px, width, height):
    """Express a pixel->pixel transform in normalized image coordinates."""
    return scale_matrix(1/width, 1/height) @ matrix_px @ scale_matrix(width, height)
