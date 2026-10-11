"""The canonical rectangular soccer pitch (the MATCH PITCH MODEL) and its projection into the camera image.

The physical field is a rectangle in world metres whose markings are standardised; only its length and
width vary between pitches, so they are learned once per match (from calibration landmarks) and then
kept. The CURRENT CAMERA HOMOGRAPHY H maps world metres (origin at the far-left corner, x along the
touchlines towards the right goal, y across the pitch towards the near touchline) to image pixels; it
is the part that changes with the camera. This module is pure geometry: the landmark and marking
catalogues for a pitch of any size, projection with the in-front-of-the-camera test, the fit of H (and
optionally the size) to point, line and circle observations by non-linear least squares around a
prior, matching of detected paint segments to the markings H predicts, and sanity checks.
"""
import math
from dataclasses import dataclass

import cv2
import numpy as np
from scipy.optimize import least_squares

from .geometry import transform_points

DEFAULT_LENGTH, DEFAULT_WIDTH = 105.0, 68.0
LENGTH_RANGE, WIDTH_RANGE = (60.0, 120.0), (40.0, 90.0)
PA, GA, SPOT, RADIUS = 16.5, 5.5, 11.0, 9.15     # penalty area depth, goal area depth, spot, circle radius
PA_HALF, GA_HALF = 20.16, 9.16
SIDES = {'far': 'far touchline', 'near': 'near touchline', 'left': 'left goal line', 'right': 'right goal line'}


@dataclass
class PitchSize:
    """The physical pitch. Standard until calibration landmarks show a different size."""
    length: float = DEFAULT_LENGTH
    width: float = DEFAULT_WIDTH
    source: str = 'standard'     # standard | landmarks

    def to_json(self):
        return {'length': round(self.length, 1), 'width': round(self.width, 1), 'source': self.source}

    @classmethod
    def from_json(cls, data):
        if not isinstance(data, dict):
            return cls()
        try:
            length, width = float(data['length']), float(data['width'])
        except (KeyError, TypeError, ValueError):
            return cls()
        if not (LENGTH_RANGE[0] <= length <= LENGTH_RANGE[1] and WIDTH_RANGE[0] <= width <= WIDTH_RANGE[1]):
            return cls()
        return cls(length, width, str(data.get('source', 'landmarks')))


def landmarks(size):
    """Named pitch landmarks (metres) of a pitch of this size; the names the calibration UI uses."""
    L, W = size.length, size.width
    cy = W/2
    out = {'corner-far-left': (0, 0), 'corner-far-right': (L, 0), 'corner-near-left': (0, W), 'corner-near-right': (L, W),
           'halfway-far': (L/2, 0), 'halfway-near': (L/2, W), 'centre-spot': (L/2, cy),
           'centre-circle-far': (L/2, cy-RADIUS), 'centre-circle-near': (L/2, cy+RADIUS),
           # The circle's left and right ends (level with the centre spot): the landmarks a midfield
           # view offers off the halfway line, which a calibration needs to span an area.
           'centre-circle-left': (L/2-RADIUS, cy), 'centre-circle-right': (L/2+RADIUS, cy)}
    for end, x0, sign in (('left', 0.0, 1), ('right', L, -1)):
        for box, depth, half in (('penalty-box', PA, PA_HALF), ('goal-box', GA, GA_HALF)):
            out[f'{end}-{box}-goal-line-far'] = (x0, cy-half)
            out[f'{end}-{box}-goal-line-near'] = (x0, cy+half)
            out[f'{end}-{box}-front-far'] = (x0+sign*depth, cy-half)
            out[f'{end}-{box}-front-near'] = (x0+sign*depth, cy+half)
        out[f'{end}-penalty-spot'] = (x0+sign*SPOT, cy)
    return out


def lines(size):
    """Every straight painted line of the pitch as (name, a, b) in metres."""
    L, W = size.length, size.width
    cy = W/2
    out = [('far touchline', (0, 0), (L, 0)), ('near touchline', (0, W), (L, W)), ('left goal line', (0, 0), (0, W)),
           ('right goal line', (L, 0), (L, W)), ('halfway line', (L/2, 0), (L/2, W))]
    for end, x0, sign in (('left', 0.0, 1), ('right', L, -1)):
        for box, depth, half in (('penalty area', PA, PA_HALF), ('goal area', GA, GA_HALF)):
            front = x0+sign*depth
            out += [(f'{end} {box} front', (front, cy-half), (front, cy+half)),
                    (f'{end} {box} far side', (x0, cy-half), (front, cy-half)),
                    (f'{end} {box} near side', (x0, cy+half), (front, cy+half))]
    return out


def circle_centre(size):
    return (size.length/2, size.width/2)


# ---------- projection ----------
def orient(H, width, height):
    """H with the sign that puts the ground just in front of the camera (the bottom-centre of the image)
    at positive depth, or None when H is unusable."""
    H = np.asarray(H, np.float64)
    if H.shape != (3, 3) or not np.isfinite(H).all() or abs(np.linalg.det(H)) < 1e-15:
        return None
    try:
        inverse = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        return None
    w = inverse[2] @ [width/2, .95*height, 1.0]
    if abs(w) < 1e-12:
        return None
    return H if w > 0 else -H


def project(H, points):
    return transform_points(H, points)


def unproject(H, points):
    return transform_points(np.linalg.inv(H), points)


def depth(H, points):
    """Positive for world points in front of the camera (H oriented)."""
    pts = np.asarray(points, np.float64).reshape(-1, 2)
    return np.c_[pts, np.ones(len(pts))] @ H[2]


def image_depth(H, points):
    """Positive for image points that see the ground in front of the camera (below the horizon)."""
    pts = np.asarray(points, np.float64).reshape(-1, 2)
    return np.c_[pts, np.ones(len(pts))] @ np.linalg.inv(H)[2]


def image_segment(H, a, b, samples=64):
    """The image of the world segment a-b restricted to the part in front of the camera: (q0, q1) in
    pixels, or None when (almost) none of it is in front."""
    t = np.linspace(0, 1, samples)
    pts = np.c_[a[0]+(b[0]-a[0])*t, a[1]+(b[1]-a[1])*t]
    front = depth(H, pts) > 1e-9
    if front.sum() < 2:
        return None
    pts = pts[front]
    q = project(H, [pts[0], pts[-1]])
    return (float(q[0][0]), float(q[0][1])), (float(q[1][0]), float(q[1][1]))


def visible(segment, width, height, margin=0.0):
    """Does an image segment (pixels) cross the frame?"""
    if segment is None:
        return False
    a, b = segment
    big = 1e7
    if not all(abs(v) < big for p in (a, b) for v in p):
        return False
    ok, _, _ = cv2.clipLine((int(-margin), int(-margin), int(width+2*margin), int(height+2*margin)),
                            (int(round(a[0])), int(round(a[1]))), (int(round(b[0])), int(round(b[1]))))
    return bool(ok)


def outline(H, size, width, height, samples=24):
    """The pitch rectangle projected into the image as normalized points (the part in front of the
    camera, clipped to a generous margin), or None."""
    L, W = size.length, size.width
    t = np.linspace(0, 1, samples, endpoint=False)
    edges = np.concatenate([np.c_[t*L, np.zeros_like(t)], np.c_[np.full_like(t, L), t*W],
                            np.c_[(1-t)*L, np.full_like(t, W)], np.c_[np.zeros_like(t), (1-t)*W]])
    keep = depth(H, edges) > 1e-9
    if keep.sum() < 3:
        return None
    pts = project(H, edges[keep])/[width, height]
    pts = pts[np.isfinite(pts).all(axis=1)]
    if len(pts) < 3:
        return None
    pts = np.clip(pts, -2, 3)
    return [(float(x), float(y)) for x, y in pts]


def image_lines(H, size, width, height):
    """Where every marking line lies in this image: {name: {'a', 'b' (pixels), 'visible'}}."""
    out = {}
    for name, a, b in lines(size):
        seg = image_segment(H, a, b)
        out[name] = {'segment': seg, 'visible': visible(seg, width, height)}
    return out


def to_pitch(H, size, point_px):
    """Image pixels -> normalized pitch coordinates (x 0..1 left to right goal line, y 0..1 far to near
    touchline), or None beyond the horizon or far off the pitch."""
    x, y = point_px
    inverse = np.linalg.inv(H)
    w = inverse[2, 0]*x+inverse[2, 1]*y+inverse[2, 2]
    if w <= 1e-12:
        return None
    px, py = (inverse[0, 0]*x+inverse[0, 1]*y+inverse[0, 2])/w, (inverse[1, 0]*x+inverse[1, 1]*y+inverse[1, 2])/w
    if not (math.isfinite(px) and math.isfinite(py)):
        return None
    nx, ny = px/size.length, py/size.width
    if -.15 <= nx <= 1.15 and -.15 <= ny <= 1.15:
        return nx, ny
    return None


def sanity(H, size, width, height):
    """None when H is a plausible view of the pitch, else the reason it is not."""
    H = orient(H, width, height)
    if H is None:
        return 'no homography'
    if np.linalg.det(H) <= 0:
        return 'mirrored (left/right or near/far swapped)'
    poly = outline(H, size, width, height)
    if poly is None:
        return 'pitch behind the camera'
    # The far touchline must appear above the near one and the left goal line left of the right one,
    # where both are in front of the camera.
    L, W = size.length, size.width
    mid = [(L/2, 0), (L/2, W), (0, W/2), (L, W/2)]
    if (depth(H, mid) > 1e-9).all():
        far, near, left, right = project(H, mid)
        if far[1] >= near[1] or left[0] >= right[0]:
            return 'pitch upside down'
    scale = px_per_metre(H, (L/2, W/2))
    if not .3 <= scale <= width:
        return 'implausible scale'
    return None


def px_per_metre(H, world_point):
    """Local image scale at a world point: pixels per metre (geometric mean over the two axes)."""
    x, y = world_point
    p = project(H, [(x, y), (x+1, y), (x, y+1)])
    return float(math.sqrt(max(1e-12, np.hypot(*(p[1]-p[0]))*np.hypot(*(p[2]-p[0])))))


# ---------- fitting ----------
@dataclass
class Fit:
    H: np.ndarray           # world metres -> image pixels, oriented
    size: PitchSize
    rms_px: float           # root-mean-square residual of the observations (pixels)
    rms_m: float            # ... of the point landmarks in metres (0 when there are none)
    points: int
    lines: int
    circle_points: int
    reason: str = ''        # why the fit was rejected ('' when accepted)
    jac: np.ndarray | None = None   # the observations' Jacobian at the solution (conditioning)

    @property
    def ok(self):
        return not self.reason


def _initial(points, size):
    """A direct homography from four or more named points, or None."""
    marks = landmarks(size)
    src = [marks[p['name']] for p in points if p['name'] in marks]
    dst = [(p['x'], p['y']) for p in points if p['name'] in marks]
    if len(src) < 4:
        return None
    H, _ = cv2.findHomography(np.asarray(src, np.float64), np.asarray(dst, np.float64), 0)
    if H is None or not np.isfinite(H).all():
        return None
    return H


def _residuals(params, obs, width, height, size, prior, learn, prior_grid, prior_weight, circle_scale):
    H = params[:9].reshape(3, 3)
    sz = PitchSize(params[9], params[10], size.source) if learn else size
    marks = landmarks(sz)
    out = []
    metric = obs.get('metric', False)
    for p in obs.get('points', ()):
        X = marks[p['name']]
        q = H @ [X[0], X[1], 1.0]
        w = q[2] if abs(q[2]) > 1e-12 else 1e-12
        rx, ry = q[0]/w-p['x'], q[1]/w-p['y']
        if metric:
            # Residual in metres: pixel error over the local image scale, so far landmarks (few pixels
            # per metre) weigh as much as near ones.
            qx, qy = H @ [X[0]+1.0, X[1], 1.0], H @ [X[0], X[1]+1.0, 1.0]
            wx, wy = (qx[2] if abs(qx[2]) > 1e-12 else 1e-12), (qy[2] if abs(qy[2]) > 1e-12 else 1e-12)
            scale = math.sqrt(max(1e-6, math.hypot(qx[0]/wx-q[0]/w, qx[1]/wx-q[1]/w)*math.hypot(qy[0]/wy-q[0]/w, qy[1]/wy-q[1]/w)))
            rx, ry = rx/scale, ry/scale
        out += [rx, ry]
    catalogue = {name: (a, b) for name, a, b in lines(sz)}
    for l in obs.get('lines', ()):
        a, b = catalogue[l['name']]
        qa, qb = H @ [a[0], a[1], 1.0], H @ [b[0], b[1], 1.0]
        line = np.cross(qa, qb)
        norm = math.hypot(line[0], line[1])
        if norm < 1e-12:
            out += [0.0, 0.0]
            continue
        wt = l.get('weight', 1.0)
        for x, y in (l['a'], l['b']):
            out.append(wt*(line[0]*x+line[1]*y+line[2])/norm)
    circle = obs.get('circle')
    if circle is not None and len(circle):
        try:
            inverse = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            inverse = None
        cx, cy = circle_centre(sz)
        for x, y in circle:
            if inverse is None:
                out.append(0.0)
                continue
            u = inverse @ [x, y, 1.0]
            w = u[2] if abs(u[2]) > 1e-12 else 1e-12
            r = math.hypot(u[0]/w-cx, u[1]/w-cy)
            out.append((r-RADIUS)*circle_scale)
    if prior is not None and prior_grid is not None and len(prior_grid):
        q = np.c_[prior_grid, np.ones(len(prior_grid))] @ H.T
        w = q[:, 2:3].copy()
        w[np.abs(w) < 1e-12] = 1e-12
        q = q[:, :2]/w
        out += (prior_weight*(q-obs['prior_px']).ravel()).tolist()
    out.append(1e3*(np.linalg.norm(params[:9])-1.0))
    if learn:
        # Weak pull towards the standard size: the size only moves when the landmarks demand it.
        out += [.02*(params[9]-DEFAULT_LENGTH), .02*(params[10]-DEFAULT_WIDTH)]
    return np.asarray(out, np.float64)


def fit(obs, width, height, size=None, prior=None, learn_size=False, prior_weight=.08, max_rms_px=None):
    """Fit the camera homography to observations in image pixels:

        obs = {'points': [{'name', 'x', 'y'}], 'lines': [{'name', 'a': (x, y), 'b': (x, y), 'weight'?}],
               'circle': [(x, y), ...], 'metric': False}

    With 'metric' the point residuals are measured in metres on the pitch instead of pixels (as the
    calibration solver does), so far landmarks count as much as near ones.

    `prior` (world -> pixels) seeds the fit and, with `prior_weight`, holds the parts of the view the
    observations do not constrain (a grid of ground points visible under the prior must stay close to
    where the prior puts them). `learn_size` also fits the pitch length and width. Returns a Fit; its
    `reason` says why it is unusable."""
    size = size or PitchSize()
    points, segs, circle = list(obs.get('points', ())), list(obs.get('lines', ())), list(obs.get('circle', ()))
    marks = landmarks(size)
    points = [p for p in points if p['name'] in marks]
    names = {name for name, _, _ in lines(size)}
    segs = [l for l in segs if l['name'] in names]
    bad = Fit(np.eye(3), size, math.inf, math.inf, len(points), len(segs), len(circle))
    H0 = orient(prior, width, height) if prior is not None else None
    if H0 is None:
        H0 = _initial(points, size)
        if H0 is None:
            return _rejected(bad, 'not enough landmarks to start from')
        H0 = orient(H0, width, height)
        if H0 is None:
            return _rejected(bad, 'the landmarks do not define a pitch plane')
    constraints = 2*len(points)+2*len(segs)+len(circle)
    if constraints < (8 if prior is None else 2):
        return _rejected(bad, 'too few observations')
    obs = dict(obs, points=points, lines=segs, circle=circle)
    prior_grid = None
    if prior is not None:
        # Ground points spread over the image under the prior, as world coordinates.
        xs, ys = np.linspace(.1, .9, 4)*width, np.linspace(.15, .95, 4)*height
        px = np.array([(x, y) for y in ys for x in xs], np.float64)
        front = image_depth(H0, px) > 1e-9
        px = px[front]
        if len(px) >= 4:
            prior_grid = unproject(H0, px)
            obs['prior_px'] = px
    circle_scale = px_per_metre(H0, circle_centre(size)) if circle else 1.0
    h0 = H0.ravel()/np.linalg.norm(H0)
    params = np.r_[h0, size.length, size.width] if learn_size else h0
    bounds = (-np.inf, np.inf)
    if learn_size:
        lo = np.r_[np.full(9, -np.inf), LENGTH_RANGE[0], WIDTH_RANGE[0]]
        hi = np.r_[np.full(9, np.inf), LENGTH_RANGE[1], WIDTH_RANGE[1]]
        bounds = (lo, hi)
    try:
        result = least_squares(_residuals, params, bounds=bounds, x_scale='jac', max_nfev=200,
                               args=(obs, width, height, size, prior, learn_size, prior_grid, prior_weight, circle_scale))
    except (ValueError, np.linalg.LinAlgError):
        return _rejected(bad, 'the fit diverged')
    H = orient(result.x[:9].reshape(3, 3), width, height)
    if H is None:
        return _rejected(bad, 'the fit diverged')
    jac = result.jac[:, :9] if getattr(result, 'jac', None) is not None else None
    learned = PitchSize(float(result.x[9]), float(result.x[10]), 'landmarks') if learn_size else size
    # Residuals of the observations alone (prior and gauge terms excluded).
    res = _residuals(np.r_[H.ravel()/np.linalg.norm(H), [learned.length, learned.width]] if learn_size else H.ravel()/np.linalg.norm(H),
                     obs, width, height, size, None, learn_size, None, 0.0, circle_scale)
    n = 2*len(points)+2*len(segs)+len(circle)
    rms_px = float(math.sqrt((res[:n]**2).mean())) if n else 0.0
    marks = landmarks(learned)
    if points:
        world = unproject(H, [(p['x'], p['y']) for p in points])
        rms_m = float(math.sqrt(((world-np.asarray([marks[p['name']] for p in points]))**2).sum(axis=1).mean()))
    else:
        rms_m = 0.0
    out = Fit(H, learned, rms_px, rms_m, len(points), len(segs), len(circle), jac=jac)
    problem = sanity(H, learned, width, height)
    if problem:
        return _rejected(out, problem)
    if points:
        w = depth(H, [marks[p['name']] for p in points])
        if (w <= 0).any():
            return _rejected(out, 'the landmarks fold the pitch; check their names')
    if max_rms_px is not None and rms_px > max_rms_px:
        return _rejected(out, f'the observations disagree by {rms_px:.1f} px')
    return out


def _rejected(fit_, reason):
    fit_.reason = reason
    return fit_


def fit_keyframes(sets, width, height, size=None):
    """Learn the pitch size from several calibration keyframes at once (each has its own camera, all
    share the physical pitch). sets: [[{'name', 'x', 'y'} ...] ...] in pixels. Returns (PitchSize, rms
    metres) or None when the landmarks cannot tell the size: a dimension counts as observed only when
    a pitch 6 m longer or wider fits the same landmarks clearly worse."""
    size = size or PitchSize()
    usable = [s for s in sets if len(s) >= 4 and _initial(s, size) is not None]
    if not usable or sum(len(s) for s in usable) < 6:
        return None

    def rms_at(length, width_m):
        sz = PitchSize(length, width_m, 'landmarks')
        errors = []
        for s in usable:
            f = fit({'points': s, 'metric': True}, width, height, sz)
            if not f.ok:
                return None
            errors.append(f.rms_m**2*len(s))
        return math.sqrt(sum(errors)/sum(len(s) for s in usable))

    def solve(free_length, free_width):
        def residual(params):
            length = params[0] if free_length else size.length
            width_m = params[-1] if free_width else size.width
            r = rms_at(length, width_m)
            return [r if r is not None else 50.0]
        x0 = [v for v, free in ((size.length, free_length), (size.width, free_width)) if free]
        lo = [v for v, free in ((LENGTH_RANGE[0], free_length), (WIDTH_RANGE[0], free_width)) if free]
        hi = [v for v, free in ((LENGTH_RANGE[1], free_length), (WIDTH_RANGE[1], free_width)) if free]
        try:
            result = least_squares(residual, x0, bounds=(lo, hi), diff_step=.02, max_nfev=60)
        except (ValueError, np.linalg.LinAlgError):
            return None
        length = float(result.x[0]) if free_length else size.length
        width_m = float(result.x[-1]) if free_width else size.width
        return length, width_m

    solved = solve(True, True)
    if solved is None:
        return None
    length, width_m = solved
    best = rms_at(length, width_m)
    if best is None:
        return None
    # Both dimensions must be pinned down: with one of them free, the other can drift to compensate.
    margin = max(.1, .25*best)
    for other in ((length+6.0, width_m), (length-6.0, width_m), (length, width_m+6.0), (length, width_m-6.0)):
        worse = rms_at(*other)
        if worse is None or worse < best+margin:
            return None
    return PitchSize(length, width_m, 'landmarks'), best


# ---------- observability: how well an evidence set constrains the pitch plane ----------
LONGITUDINAL = ('far touchline', 'near touchline', 'penalty area far side', 'penalty area near side', 'goal area far side', 'goal area near side')


def _family(name):
    return 'longitudinal' if any(name.endswith(n) for n in LONGITUDINAL) else 'transverse'


def observability(obs, width, height, size=None, rms_m=None, rms_px=None, jac=None):
    """Geometric certainty of a calibration from its evidence alone, 0 (useless) .. 1 (excellent), with
    the parts. It is independent of how smoothly an estimate was carried through time:

      points        how many named landmarks, how much of the frame they span, whether they pin down
                    both pitch axes (landmarks along one line, or two converging touchlines, do not)
      lines         how many marking lines of each family: along the pitch (touchlines, box sides) and
                    across it (goal lines, halfway line, box fronts); one family alone fixes neither the
                    length direction nor where the pitch ends
      circle        how much of the centre circle is seen (arc coverage), which fixes scale and the
                    centre in a midfield view where straight-line intersections are few
      conditioning  of the fit's Jacobian when given (a flat direction means a parameter the evidence
                    cannot see)
      residual      how well the evidence agrees with one pitch plane (metres for landmarks, pixels
                    for paint)
    The parts combine as: coverage = evidence of both axes and enough of it; score = coverage x
    conditioning x residual."""
    size = size or PitchSize()
    marks = landmarks(size)
    points = [p for p in obs.get('points', ()) if p['name'] in marks]
    segs = list(obs.get('lines', ()))
    circle = list(obs.get('circle', ()))
    details = {'landmarks': len(points), 'lines': len(segs), 'circlePoints': len(circle)}
    # Points: count, spread over the frame, both axes.
    pts_score = 0.0
    if len(points) >= 4:
        src = np.array([(p['x'], p['y']) for p in points], np.float64)
        dst = np.array([marks[p['name']] for p in points], np.float64)
        count = min(1.0, (len(points)-2)/4)   # four landmarks determine the plane with no check; six check it
        spread = min(1.0, float(cv2.contourArea(cv2.convexHull(src.astype(np.float32))))/(width*height)/.12)
        axes = min(1.0, float(np.ptp(dst[:, 0]))/30)*min(1.0, float(np.ptp(dst[:, 1]))/20)
        pts_score = math.sqrt(count*spread)*math.sqrt(axes)
        details.update(count=round(count, 3), spread=round(spread, 3), axes=round(axes, 3))
    # Lines: both families needed; the length direction and the pitch end come only from transverse lines.
    families = {'longitudinal': set(), 'transverse': set()}
    for l in segs:
        families[_family(l['name'])].add(l['name'])
    nl, nt = len(families['longitudinal']), len(families['transverse'])
    lines_score = min(1.0, nl/2)*min(1.0, nt/2)*min(1.0, (nl+nt)/4)
    if nl and not nt:
        lines_score = .1*min(1.0, nl/2)   # touchlines alone: a vanishing direction, no end and no scale
    elif nt and not nl:
        lines_score = .15*min(1.0, nt/2)
    details.update(longitudinal=sorted(families['longitudinal']), transverse=sorted(families['transverse']))
    # Circle: arc coverage around the centre spot under the fit, when a homography is available.
    circle_score = 0.0
    if len(circle) >= 12 and obs.get('H') is not None:
        H = np.asarray(obs['H'], np.float64)
        try:
            world = unproject(H, circle)
            cx, cy = circle_centre(size)
            angles = np.degrees(np.arctan2(world[:, 1]-cy, world[:, 0]-cx))
            bins = np.unique(np.floor((angles+180)/15).astype(int))
            coverage = len(bins)/24
            radii = np.hypot(world[:, 0]-cx, world[:, 1]-cy)
            agreement = float(np.mean(np.abs(radii-RADIUS) <= .6))
            circle_score = min(1.0, coverage/.5)*agreement
            details.update(circleCoverage=round(coverage, 3), circleAgreement=round(agreement, 3))
        except np.linalg.LinAlgError:
            pass
    # The circle with the halfway line fixes the centre and the scale; with a touchline it fixes everything.
    combined = max(pts_score, lines_score, .5*circle_score)
    if circle_score and nt:
        combined = max(combined, .6*circle_score+.15*nl)
    coverage = min(1.0, combined)
    conditioning = 1.0
    if jac is not None:
        singular = np.linalg.svd(np.asarray(jac, np.float64), compute_uv=False)
        cond = float(singular[0]/max(singular[min(len(singular)-1, 7)], 1e-12))   # the 8 non-gauge parameters
        conditioning = min(1.0, max(0.0, 1-(math.log10(max(cond, 1.0))-3)/4))      # 1e3 -> 1, 1e7 -> 0
        details['conditioning'] = round(conditioning, 3)
    residual = 1.0
    if rms_m is not None:
        residual = min(residual, min(1.0, max(0.0, 1-rms_m/2.5)))
        details['rmsMetres'] = round(rms_m, 2)
    if rms_px is not None:
        residual = min(residual, min(1.0, max(0.0, 1-rms_px/(.02*width))))
        details['rmsPixels'] = round(rms_px, 1)
    score = coverage*conditioning*residual
    details.update(score=round(score, 3), coverage=round(coverage, 3), residual=round(residual, 3),
                   pointsScore=round(pts_score, 3), linesScore=round(lines_score, 3), circleScore=round(circle_score, 3))
    return details


def paint_agreement(H, size, masks, width, height, names=None):
    """Bidirectional check: project the canonical markings into the image and measure how much of each
    visible one lies on paint (the ridge mask of pitch.line_masks). Returns {name: share} for the markings
    in view (None where the marking is in view but the mask has no row there)."""
    ridge = masks['ridge']
    mh, mw = ridge.shape
    out = {}
    catalogue = lines(size)
    for name, a, b in catalogue:
        if names is not None and name not in names:
            continue
        seg = image_segment(H, a, b, samples=96)
        if seg is None or not visible(seg, width, height):
            continue
        (x0, y0), (x1, y1) = seg
        t = np.linspace(0, 1, 60)
        xs, ys = (x0+(x1-x0)*t)*mw/width, (y0+(y1-y0)*t)*mh/height
        inside = (xs >= 1) & (xs < mw-1) & (ys >= 1) & (ys < mh-1)
        if inside.sum() < 10:
            continue
        xi, yi = np.rint(xs[inside]).astype(int), np.rint(ys[inside]).astype(int)
        hit = np.zeros(len(xi), bool)
        for dx, dy in ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1), (2, 0), (-2, 0), (0, 2), (0, -2)):
            hit |= ridge[np.clip(yi+dy, 0, mh-1), np.clip(xi+dx, 0, mw-1)]
        out[name] = round(float(hit.mean()), 3)
    return out


# ---------- matching detected paint to the predicted markings ----------
def match_segments(H, size, segs, width, height, tol_px=None, angle_tol=8.0):
    """Assign detected paint segments (normalized endpoints, from pitch.detect_lines) to the marking
    lines that H predicts near them: both endpoints within tol_px of the predicted line, directions
    within angle_tol degrees, the segment overlapping the predicted line's in-view extent, and no
    second marking almost as close (an ambiguous segment is left out). Returns [{'name', 'a', 'b'
    (pixels), 'error' (px), 'support'}]."""
    tol = tol_px if tol_px is not None else max(6.0, .015*width)
    predicted = image_lines(H, size, width, height)
    out = []
    for seg in segs:
        a = (seg['a'][0]*width, seg['a'][1]*height)
        b = (seg['b'][0]*width, seg['b'][1]*height)
        length = math.hypot(b[0]-a[0], b[1]-a[1])
        if length < 1e-6:
            continue
        direction = ((b[0]-a[0])/length, (b[1]-a[1])/length)
        candidates = []
        for name, info in predicted.items():
            if not info['visible']:
                continue
            (qa, qb) = info['segment']
            dx, dy = qb[0]-qa[0], qb[1]-qa[1]
            plen = math.hypot(dx, dy)
            if plen < 1e-6:
                continue
            ux, uy = dx/plen, dy/plen
            if abs(ux*direction[0]+uy*direction[1]) < math.cos(math.radians(angle_tol)):
                continue
            dist = lambda p: abs(ux*(p[1]-qa[1])-uy*(p[0]-qa[0]))
            error = max(dist(a), dist(b))
            if error > tol:
                continue
            # Overlap along the predicted line.
            ta, tb = sorted(((p[0]-qa[0])*ux+(p[1]-qa[1])*uy for p in (a, b)))
            overlap = min(tb, plen)-max(ta, 0.0)
            if overlap < .5*length:
                continue
            candidates.append((error, name))
        if not candidates:
            continue
        candidates.sort()
        if len(candidates) > 1 and candidates[1][0] < 2*candidates[0][0]+2:
            continue
        error, name = candidates[0]
        out.append({'name': name, 'a': a, 'b': b, 'error': float(error), 'support': float(seg.get('support', 0.0))})
    return out


def at_horizon(H, size, width, height, side, min_px=None):
    """Is this goal line's projection at the touchlines' vanishing point: the two touchlines less than
    min_px apart there? Such an end is a perspective limit, not a measured pitch end."""
    L, W = size.length, size.width
    x = 0.0 if side == 'left' else L
    ends = [(x, 0.0), (x, W)]
    if (depth(H, ends) <= 1e-9).any():
        return True
    far, near = project(H, ends)
    return float(np.hypot(*(near-far))) < (min_px if min_px is not None else .02*height)


def side_status(H, size, width, height, matched=()):
    """For each logical side of the pitch: 'line' (paint matched on it this tick), 'predicted' (in view
    but no paint matched), 'horizon' (a goal line whose projection collapses onto the touchlines'
    vanishing point: no pitch end is known there) or 'off-screen'."""
    predicted = image_lines(H, size, width, height)
    found = {m['name'] for m in matched}
    out = {}
    for side, name in SIDES.items():
        info = predicted[name]
        if name in found:
            out[side] = 'line'
        elif side in ('left', 'right') and info['visible'] and at_horizon(H, size, width, height, side):
            out[side] = 'horizon'
        else:
            out[side] = 'predicted' if info['visible'] else 'off-screen'
    return out
