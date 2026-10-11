"""Automatic midfield anchor: the centre circle, the halfway line and a touchline register the pitch
where a low sideline camera shows no corner and no goal line.

Pole-polar geometry is projective, so it holds in the image as it does on the pitch:

  * the halfway line is a diameter of the centre circle, so its pole with respect to the circle's image
    (an ellipse) is the vanishing point of the touchline direction;
  * the halfway line meets the circle at the far and near circle points (9.15 m from the centre) and
    the touchlines at the halfway points (half the pitch width from the centre): points of known
    position along one world line, whose 1-D projectivity gives the centre spot;
  * the line through the centre spot and that vanishing point is the diameter along the pitch; it
    meets the ellipse at the left and right circle points.

Those derived points only start a full fit (field.fit) on the paint itself: the halfway line, the
touchlines and the circle's ridge pixels. The result is validated in both directions (the canonical
markings projected back into the image must lie on paint: field.paint_agreement), scored for
observability (field.observability, including the fit's conditioning) and refused when it is not
constrained. The intersection of the two touchlines is never used as a pitch corner: it is the
vanishing point the pole-polar construction expects.
"""
import math

import cv2
import numpy as np

from . import field

MIN_OBSERVABILITY = .2     # an automatic anchor needs this much geometric certainty (user landmarks: pitch_model.MIN_ANCHOR)
MIN_PAINT = .45            # share of the evidence markings' reprojection that must lie on paint
MAX_RMS_FRACTION = .006    # of the frame width: the fit's residual on the paint
MIN_CIRCLE_AGREEMENT = .6  # share of circle pixels within .6 m of the canonical circle under the fit
TOUCHLINE_MIN_SUPPORT = .2


def _line(a, b):
    l = np.cross([a[0], a[1], 1.0], [b[0], b[1], 1.0])
    n = math.hypot(l[0], l[1])
    return l/n if n > 1e-12 else None


def _meet(l, m):
    p = np.cross(l, m)
    if abs(p[2]) < 1e-9*max(1.0, abs(p[0]), abs(p[1])):
        return None
    return np.array([p[0]/p[2], p[1]/p[2]])


def conic(ellipse):
    """3x3 symmetric matrix of an ellipse ((cx, cy), (major, minor), angle in degrees), pixels."""
    (cx, cy), (ma, mi), angle = ellipse
    a, b = ma/2, mi/2
    t = math.radians(angle)
    R = np.array([[math.cos(t), math.sin(t)], [-math.sin(t), math.cos(t)]])
    M = R.T @ np.diag([1/a**2, 1/b**2]) @ R
    c = np.array([cx, cy])
    C = np.zeros((3, 3))
    C[:2, :2] = M
    C[:2, 2] = C[2, :2] = -M @ c
    C[2, 2] = c @ M @ c-1
    return C


def pole(C, l):
    p = np.linalg.solve(C, l)
    return p


def line_conic(l, C, origin, direction):
    """The two points where line l (through `origin` along `direction`) meets the conic, or None."""
    p0 = np.array([origin[0], origin[1], 1.0])
    d = np.array([direction[0], direction[1], 0.0])
    A, B, Cc = d @ C @ d, 2*(p0 @ C @ d), p0 @ C @ p0
    disc = B*B-4*A*Cc
    if abs(A) < 1e-12 or disc <= 0:
        return None
    r = math.sqrt(disc)
    t1, t2 = (-B-r)/(2*A), (-B+r)/(2*A)
    return [origin+t1*np.asarray(direction), origin+t2*np.asarray(direction)]


def _projectivity(samples):
    """1-D projectivity t = (a u + b)/(c u + 1) from (u, t) pairs (least squares), or None."""
    if len(samples) < 3:
        return None
    A = np.array([[u, 1.0, -u*t] for u, t in samples])
    rhs = np.array([t for _, t in samples])
    sol, *_ = np.linalg.lstsq(A, rhs, rcond=None)
    a, b, c = sol
    return lambda u: (a*u+b)/(c*u+1) if abs(c*u+1) > 1e-9 else None


def ellipse_from_points(points_px):
    pts = np.asarray(points_px, np.float32)
    if len(pts) < 8:
        return None
    try:
        (cx, cy), (ma, mi), angle = cv2.fitEllipse(pts)
    except cv2.error:
        return None
    if not np.isfinite([cx, cy, ma, mi, angle]).all() or min(ma, mi) <= 1:
        return None
    return ((float(cx), float(cy)), (float(ma), float(mi)), float(angle))


def touchline_candidates(model, ellipse_px, width, height):
    """(side, segment) touchlines for the solver: the detector's boundary lines first, else the long
    flat-ish paint segments that do not cross the ellipse, above it (far) or below it (near)."""
    out = []
    for l in model.lines:
        if l['side'] in ('far', 'near') and l['age'] <= 0:
            out.append((l['side'], l))
    if out:
        return out
    (cx, cy), (ma, mi), _ = ellipse_px
    for seg in sorted(model.markings, key=lambda s: -s['support']):
        if seg['support'] < TOUCHLINE_MIN_SUPPORT:
            continue
        a, b = (seg['a'][0]*width, seg['a'][1]*height), (seg['b'][0]*width, seg['b'][1]*height)
        l = _line(a, b)
        if l is None:
            continue
        d = l[0]*cx+l[1]*cy+l[2]
        if abs(d) < .6*max(ma, mi):
            continue   # runs through the circle: not a touchline
        side = 'far' if d*(1 if l[1] > 0 else -1) < 0 else 'near'
        mid_y = (a[1]+b[1])/2
        side = 'far' if mid_y < cy else 'near'
        if all(s != side for s, _ in out):
            out.append((side, seg))
        if len(out) == 2:
            break
    return out


def solve(circle, halfway, touchlines, width, height, size=None, masks=None):
    """Register the pitch from the centre-circle paint (pitch.circle_points: normalized inlier points),
    the halfway line segment (normalized) and touchlines [(side, segment)] (normalized). Returns
    {'H' (world metres -> pixels), 'observability', 'details', 'rms_px', 'points'} or {'reason'}."""
    size = size or field.PitchSize()
    W, Hh = width, height
    if circle is None or halfway is None:
        return {'reason': 'no centre circle or halfway line'}
    if not touchlines:
        return {'reason': 'the circle and the halfway line alone leave the pitch width unresolved: no touchline'}
    circle_px = [(x*W, y*Hh) for x, y in circle['points']]
    ellipse = ellipse_from_points(circle_px)
    if ellipse is None:
        return {'reason': 'the circle paint does not fit an ellipse'}
    C = conic(ellipse)
    ha, hb = (halfway['a'][0]*W, halfway['a'][1]*Hh), (halfway['b'][0]*W, halfway['b'][1]*Hh)
    h = _line(ha, hb)
    if h is None:
        return {'reason': 'degenerate halfway line'}
    d = np.array([hb[0]-ha[0], hb[1]-ha[1]])
    d /= np.linalg.norm(d)
    if d[1] < 0:
        d = -d   # along the halfway line from the far side (top of the image) to the near side
    hits = line_conic(h, C, np.array(ha), d)
    if hits is None:
        return {'reason': 'the halfway line does not cross the circle'}
    hits.sort(key=lambda p: p @ d)
    circle_far, circle_near = hits
    R = field.RADIUS
    along = lambda p: float((p-circle_far) @ d)
    samples = [(size.width/2-R, along(circle_far)), (size.width/2+R, along(circle_near))]
    named = {'centre-circle-far': circle_far, 'centre-circle-near': circle_near}
    used_lines = [{'name': 'halfway line', 'a': ha, 'b': hb}]
    for side, seg in touchlines:
        a, b = (seg['a'][0]*W, seg['a'][1]*Hh), (seg['b'][0]*W, seg['b'][1]*Hh)
        l = _line(a, b)
        q = _meet(h, l) if l is not None else None
        if q is None:
            continue
        t = along(q)
        if side == 'far' and t >= samples[0][1]-R*0 and t > along(circle_far):
            continue   # a far touchline meeting the halfway line beyond the far circle point is not one
        if side == 'near' and t < along(circle_near):
            continue
        samples.append((0.0 if side == 'far' else size.width, t))
        named['halfway-far' if side == 'far' else 'halfway-near'] = q
        used_lines.append({'name': f'{side} touchline', 'a': a, 'b': b})
    if len(samples) < 3:
        return {'reason': 'no touchline meets the halfway line on the right side of the circle'}
    proj = _projectivity(samples)
    tc = proj(size.width/2) if proj else None
    if tc is None or not math.isfinite(tc):
        return {'reason': 'the positions along the halfway line are inconsistent'}
    centre = circle_far+tc*d
    named['centre-spot'] = centre
    if not (along(circle_far) < tc < along(circle_near)):
        return {'reason': 'the centre spot would fall outside the circle'}
    # The diameter along the pitch: through the centre spot towards the pole of the halfway line.
    try:
        v = pole(C, h)
    except np.linalg.LinAlgError:
        return {'reason': 'degenerate circle'}
    if abs(v[2]) > 1e-9*max(1.0, abs(v[0]), abs(v[1])):
        direction = np.array([v[0]/v[2]-centre[0], v[1]/v[2]-centre[1]])
    else:
        direction = np.array([v[0], v[1]])
    n = np.linalg.norm(direction)
    if n < 1e-9:
        return {'reason': 'degenerate circle'}
    direction /= n
    sides = line_conic(_line(centre, centre+direction*100), C, centre, direction)
    if sides is None:
        return {'reason': 'the diameter along the pitch misses the circle'}
    sides.sort(key=lambda p: p[0])
    named['centre-circle-left'], named['centre-circle-right'] = sides
    points = [{'name': k, 'x': float(p[0]), 'y': float(p[1])} for k, p in named.items()]
    # The fit on the paint itself, started from the derived points and then freed from them.
    first = field.fit({'points': points, 'lines': used_lines, 'circle': circle_px}, W, Hh, size)
    if not first.ok:
        return {'reason': f'the derived points do not fit a pitch plane ({first.reason})'}
    fit = field.fit({'lines': used_lines, 'circle': circle_px}, W, Hh, size, prior=first.H, prior_weight=.02,
                    max_rms_px=MAX_RMS_FRACTION*W)
    if not fit.ok:
        fit = first
    H = field.orient(fit.H, W, Hh)
    if H is None:
        return {'reason': 'the solution is not a pitch plane in front of the camera'}
    why = field.sanity(H, size, W, Hh)
    if why:
        return {'reason': why}
    if fit.rms_px > MAX_RMS_FRACTION*W:
        return {'reason': f'the paint disagrees with the fit ({fit.rms_px:.1f} px)'}
    obs = field.observability({'lines': used_lines, 'circle': circle_px, 'H': H}, W, Hh, size, rms_px=fit.rms_px, jac=fit.jac)
    details = dict(obs, derived=sorted(named), evidence=[l['name'] for l in used_lines]+['centre circle'])
    if obs.get('circleAgreement', 0.0) < MIN_CIRCLE_AGREEMENT:
        return {'reason': f'the circle paint does not lie on the canonical circle under the fit (agreement {obs.get("circleAgreement", 0):.2f})', 'details': details}
    if masks is not None:
        agreement = field.paint_agreement(H, size, masks, W, Hh)
        details['paint'] = agreement
        evidence = [agreement[n] for n in [l['name'] for l in used_lines] if n in agreement and agreement[n] is not None]
        if evidence and min(evidence) < MIN_PAINT:
            return {'reason': f'reprojected markings leave the paint (agreement {min(evidence):.2f})', 'details': details}
    if obs['score'] < MIN_OBSERVABILITY:
        return {'reason': f'observability {obs["score"]:.2f}: the evidence does not constrain the pitch (conditioning {obs.get("conditioning", 1):.2f})', 'details': details}
    return {'H': H, 'observability': obs['score'], 'details': details, 'rms_px': fit.rms_px, 'points': points}
