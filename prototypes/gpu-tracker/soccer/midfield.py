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

Without a touchline that construction has one unknown left: a homography has eight degrees of freedom,
the circle and its diameter give seven. A real camera has fewer. With square pixels, no skew and the
principal point at the image centre (broadcast lenses, and the rectilinear views a panoramic camera
renders), H = K [r1 r2 t] with one intrinsic, the focal length: seven unknowns. The pinhole route
(solve(..., allow_pinhole=True), used only for runs without user keyframes) solves that camera; see
pinhole_solve for how far that goes and how its confidence is kept honest.
"""
import math

import cv2
import numpy as np
from scipy.optimize import least_squares

from . import field, registration

MIN_OBSERVABILITY = .2     # an automatic anchor needs this much geometric certainty (user landmarks: pitch_model.MIN_ANCHOR)
MIN_PAINT = .45            # share of the evidence markings' reprojection that must lie on paint
MAX_RMS_FRACTION = .006    # of the frame width: the fit's residual on the paint
MIN_CIRCLE_AGREEMENT = .6  # share of circle pixels within .6 m of the canonical circle under the fit
TOUCHLINE_MIN_SUPPORT = .2

# The pinhole route (automatic landmarks only).
FOV_RANGE = (6.0, 150.0)   # degrees of horizontal view a match camera plausibly has (the users' keyframes decompose to 42-73 broadcast, 60-68 pole)
HEIGHT_RANGE = (2.0, 40.0) # metres above the pitch: the users' keyframes decompose to 5.1 m (amateur 3 pole, both keyframes) and 15-30 m (broadcast)
SCAN_STEPS = 120           # focal lengths tried across FOV_RANGE (log-spaced) to seed the fit
MAX_SEEDS = 6              # fits started from those seeds, best first
CIRCLE_EQUIVALENT = 16.0   # the circle's hundreds of points weigh together as this many observations, so the arc cannot bend the lines off
LINE_EQUIVALENT = 4.0      # ... and each painted line as this many
LINE_SAMPLES = 5           # points along a painted segment where the predicted line is compared
CIRCLE_SAMPLES = 400       # circle points kept for the fit (evenly over the inliers): they weigh as CIRCLE_EQUIVALENT together
FIT_TOL = 1e-6             # relative cost and step tolerance of the fits (scipy's 1e-8 doubled the iterations for nothing measurable)
PAINT_SIGMA = 1.0          # paint position noise in pixels of the analysis mask (pitch.LINE_W wide): its quantisation
ROBUST_SCALE = 3.0         # residuals beyond this many noise sigmas count linearly (soft L1): stray circle pixels, a crouching player
PERSON_SIGMA = .12         # log spread of standing people's box heights (pose, age, the detector's box)
PERSON_HEIGHT, PERSON_PRIOR = 1.7, .1   # metres, and the log spread of that prior (youth to adult players)
PERSON_RANGE = (1.2, 2.1)  # metres: the bounds of the fitted people height
MIN_PEOPLE = 6             # standing people needed before their heights (the horizon) enter the fit
PERSON_MIN_ROWS = 8        # rows of a 1080-row frame a person's box must span (a pixel of a smaller box is over 12 % of its height)
PERSON_ASPECT = 1.2        # a standing person's box is at least this much taller than wide
PERSON_ROBUST = 2.0        # noise sigmas beyond which a box's height stops pulling the fit (Cauchy): the detections include non-people
SIGMA_ZERO = 5.0           # metres of position uncertainty (90th percentile over the visible pitch) at which a pinhole anchor is worth nothing
PINHOLE_CAP = .55          # observability ceiling without an independent painted line: combine(1, .55) = .66, below HIGH
FLAT_RATIO = .25           # ellipses flatter than this (minor/major) lower that ceiling in proportion: a grazing circle barely fixes depth
CONFIRM_PAINT = .6         # paint agreement at which a marking the fit did not use confirms it
CONFIRM_LENGTH = .15       # frame widths of a marking's predicted image that must be in view for its paint to confirm
CONFIRM_METRES = 1.0       # a confirming line must be pinned this well on the ground: its pixel error times the metres per pixel across it
                           # (amateur 3's far touchline is .76 m/px: 'confirmed' by paint within 6 px it let the far landmarks sit 2-7 m off)
CONFIRM_SUPPORT = .05      # detected segments weaker than this (pitch.detect_lines support: a few scattered pixels) confirm nothing
PAINT_REACH = 2.0          # mask pixels: how far field.paint_agreement looks for paint around a predicted line
PEOPLE_MARGIN = 6.0        # metres beyond the lines where a person still counts as on the pitch (benches, assistant referees)
PEOPLE_ON_PITCH = .5       # share of the detected people that must stand on the solved pitch
CIRCLE_TOLERANCE = 2.0     # noise sigmas (mask pixels) within which a circle pixel lies on the solved circle's image
AMBIGUOUS_CHI2 = 9.0       # a second solution within this chi-square of the best (3 sigma) and DISAGREE_M away makes the frame ambiguous
DISAGREE_M = 2.0
JOIN_TOLERANCE = 3.0       # mask pixels: a matched segment this close to the chosen camera's prediction joins its fit
JOIN_RMS = 2.5             # noise sigmas: a joined line the joint fit leaves farther than this is dropped again
MIN_SPREAD = .02           # largest |n . K^-1 v| over the focal scan below which the circle and the halfway line cannot tell the focal length
WEAK_SPREAD = .2           # ... below which they tell it poorly: seeds across the focal range join the scan's roots (amateur 3: .07-.16)
TOUCHLINE_GROUND_DEG = 4.0 # degrees: a segment the solved camera puts this close to parallel with the touchlines on the ground is touchline-family
TOUCHLINE_MIN_HALF = 23.0  # metres from the centre: nearer, a touchline-parallel line is a box side (20.16 m); 11-a-side pitches are 45 m wide or more
TOUCHLINE_REACH = 4.0      # metres: a touchline lies this close to the one the camera predicts at the assumed (or learned) half width, so a
                           # touchline measures widths of 60-76 m around the assumed 68 m (the standard pitches; the users' measured 68.9 and
                           # 69.4 m). Amateur 3's detector finds a parallel line 6-11 m beyond its far touchline (127-172 s) and not the
                           # touchline: joined, it measured 79-89 m wide pitches and pulled the far landmarks 3-7 m off


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


def solve(circle, halfway, touchlines, width, height, size=None, masks=None, allow_pinhole=False, people=(), markings=()):
    """Register the pitch from the centre-circle paint (pitch.circle_points: normalized inlier points),
    the halfway line segment (normalized) and touchlines [(side, segment)] (normalized). Returns
    {'H' (world metres -> pixels), 'observability', 'details', 'rms_px', 'points'} or {'reason'}.

    allow_pinhole (runs without user keyframes only): the pinhole route first (pinhole_solve, with the
    people's boxes and the frame's other markings as evidence), this projective construction second."""
    if allow_pinhole:
        return _automatic(circle, halfway, touchlines, width, height, size, masks, people, markings)
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


# ---------- the pinhole route (runs without user keyframes) ----------
def _automatic(circle, halfway, touchlines, width, height, size, masks, people, markings):
    """The keyframe-less midfield anchor: the pinhole camera first. When it refuses for want of evidence and a
    touchline was found, the projective construction, whose touchline sits at the assumed (or learned) half
    width; never when the evidence contradicted the camera (an ambiguity, paint or people off the solved pitch:
    'contradiction'), which the assumption would only paper over. The construction's observability is capped
    like an unconfirmed pinhole anchor's, lower for a flat ellipse (the shipped route credits the assumed
    width as evidence). Both routes return 'fitted': the painted lines the anchor was fitted to, which the
    chain must not count as independent confirmation (pitch_model.PitchTracker._lifts_cap)."""
    size = size or field.PitchSize()
    result = pinhole_solve(circle, halfway, touchlines, width, height, size, masks, people, markings)
    if 'H' in result or not touchlines or result.get('contradiction'):
        return result
    fallback = solve(circle, halfway, touchlines, width, height, size, masks)
    if 'H' not in fallback:
        return {'reason': f"{result['reason']}; with the assumed width: {fallback['reason']}", 'details': result.get('details')}
    ellipse = ellipse_from_points([(x*width, y*height) for x, y in circle['points']])
    flat = min(ellipse[1])/max(ellipse[1])
    cap = PINHOLE_CAP*min(1.0, flat/FLAT_RATIO)
    score = min(fallback['observability'], cap)
    dimensions = size.dimensions()
    details = dict(fallback['details'], route='projective', pinholeRefused=result['reason'], widthAssumed=dimensions['width']['source'] == 'assumed',
                   size=dimensions, ellipseRatio=round(flat, 3), cap=round(cap, 3), observabilityBeforeCap=round(fallback['observability'], 3), score=round(score, 3))
    if score < MIN_OBSERVABILITY:
        return {'reason': f"{result['reason']}; with the assumed width: observability {score:.2f} (ellipse ratio {flat:.2f}, unconfirmed)", 'details': details}
    return dict(fallback, observability=score, cap=cap, details=details, fitted=[n for n in details['evidence'] if n != field.CENTER_CIRCLE])


def _normalizer(width, height):
    """Pixels -> coordinates centred on the image centre, in units of the longer image side: the principal
    point is the origin, and the conic algebra is well conditioned."""
    s = float(max(width, height))
    return np.array([[1/s, 0, -width/2/s], [0, 1/s, -height/2/s], [0, 0, 1.0]]), s


def circle_normals(Q):
    """The two planes that cut the cone Q (camera coordinates, x^T Q x = 0) in a circle: [(n, m)] with n the
    unit plane normal pointing away from the camera and m the unit direction of the circle's centre. The
    cone's eigenvalues l1 >= l2 > 0 > l3 give n = +-sqrt((l1-l2)/(l1-l3)) e1 + sqrt((l2-l3)/(l1-l3)) e3,
    and the centre is the pole of the plane: Q^-1 n."""
    scale = float(np.max(np.abs(Q)))
    if not math.isfinite(scale) or scale <= 0:
        return []
    Q = Q/scale
    w, V = np.linalg.eigh(Q)
    if (w > 0).sum() == 1:
        Q = -Q
        w, V = np.linalg.eigh(Q)
    order = np.argsort(-w)
    w, V = w[order], V[:, order]
    l1, l2, l3 = w
    if not l1 >= l2 > 0 > l3:
        return []
    p, q = math.sqrt(max(0.0, (l1-l2)/(l1-l3))), math.sqrt(max(0.0, (l2-l3)/(l1-l3)))
    out = []
    for sign in (1, -1):
        n = sign*p*V[:, 0]+q*V[:, 2]
        n /= np.linalg.norm(n)
        try:
            m = np.linalg.solve(Q, n)
        except np.linalg.LinAlgError:
            continue
        if m[2] < 0:
            m = -m
        if n @ m < 0:
            n = -n
        out.append((n, m/np.linalg.norm(m)))
    return out


def _phi_range(width, s):
    """The focal length (in units of s) at the wide and the narrow end of FOV_RANGE."""
    return tuple((width/2/s)/math.tan(math.radians(fov/2)) for fov in (FOV_RANGE[1], FOV_RANGE[0]))


def focal_seeds(Cn, v, width, s):
    """Focal lengths (units of s) where the halfway line is a diameter of the circle's plane: for every
    focal length the circle's cone gives two candidate ground planes, and the touchline direction (the
    pole v of the halfway line) must lie in the true one. Sign changes of n . K^-1 v along each branch
    are the roots; when there is none (the camera stands in or near the vertical plane of the halfway line,
    where every focal length fits the circle and its diameter), the best near-misses and a spread of focal
    lengths seed the fit and the other evidence decides. Returns (seeds best first, the largest |n . K^-1 v|
    seen: near zero means the circle and the halfway line alone cannot tell the focal length)."""
    lo, hi = _phi_range(width, s)
    phis = np.geomspace(lo, hi, SCAN_STEPS)
    g = np.full((SCAN_STEPS, 2), np.nan)
    previous = None
    for i, phi in enumerate(phis):
        K = np.diag([phi, phi, 1.0])
        pair = circle_normals(K @ Cn @ K)
        if len(pair) != 2:
            previous = None
            continue
        if previous is not None and abs(pair[0][0] @ previous) < abs(pair[1][0] @ previous):
            pair = pair[::-1]
        previous = pair[0][0]
        d = np.array([v[0]/phi, v[1]/phi, v[2]])
        d /= np.linalg.norm(d)
        g[i] = [pair[0][0] @ d, pair[1][0] @ d]
    seeds = []
    for b in (0, 1):
        for i in range(SCAN_STEPS-1):
            a, c = g[i, b], g[i+1, b]
            if np.isfinite(a) and np.isfinite(c) and a*c < 0:
                t = a/(a-c)
                seeds.append((0.0, float(phis[i]*(phis[i+1]/phis[i])**t)))
    best = np.fmin(np.abs(g[:, 0]), np.abs(g[:, 1]))
    spread = float(np.nanmax(np.abs(g))) if np.isfinite(g).any() else 0.0
    if not seeds:
        for i in range(1, SCAN_STEPS-1):
            if np.isfinite(best[i]) and best[i] < best[i-1] and best[i] < best[i+1]:
                seeds.append((float(best[i]), float(phis[i])))
        seeds.sort()
        seeds = seeds[:MAX_SEEDS-3]
    if len(seeds) < MAX_SEEDS-2 and spread < WEAK_SPREAD:
        # Nearly symmetric: a root of a flat scan is noise, so focal lengths across the range seed the fit too.
        seeds += [(1.0, float(phi)) for phi in np.geomspace(lo, hi, 5)[1:-1]]
    unique = []
    for quality, phi in sorted(seeds):
        if all(abs(math.log(phi/u)) > .02 for u in unique):
            unique.append(phi)
    return unique[:MAX_SEEDS], spread


def seed_poses(Cn, v, phi):
    """The cameras at focal length phi (units of s) from the circle alone, one per candidate ground plane,
    the halfway line choosing the rotation about the ground normal: [(|n . d| (how well the touchline
    direction d lies in that plane), R with columns x along the touchlines, y towards the near side, the
    ground normal; t the circle centre in camera coordinates, metres)]. Both planes are kept: when the
    camera stands near the vertical plane of the halfway line both fit the circle and its diameter at the
    true focal length (one of them is a mirror image), and only the fit on all the evidence can tell."""
    K = np.diag([phi, phi, 1.0])
    d = np.array([v[0]/phi, v[1]/phi, v[2]])
    d /= np.linalg.norm(d)
    out = []
    for n, m in circle_normals(K @ Cn @ K):
        x = d-(d @ n)*n
        if np.linalg.norm(x) < 1e-9:
            continue
        x /= np.linalg.norm(x)
        y = np.cross(n, x)
        a, b = K @ m, K @ x
        A, B = a @ Cn @ a, b @ Cn @ b
        if A*B >= 0:
            continue
        t = field.RADIUS*math.sqrt(-B/A)*m
        near, far = K @ (t+field.RADIUS*y), K @ (t-field.RADIUS*y)
        if near[2] <= 0 or far[2] <= 0:
            continue
        if near[1]/near[2] < far[1]/far[2]:
            x, y = -x, -y   # y runs towards the near touchline, lower in the image
        out.append((float(abs(d @ n)), np.c_[x, y, n], t))
    return sorted(out, key=lambda c: c[0])


def _cauchy(z):
    """Cauchy residual in units of the noise: z when small, its influence fading beyond PERSON_ROBUST (a box
    that is not a standing person, part of one, two people in one box, must not pull the horizon)."""
    c = PERSON_ROBUST
    return np.sign(z)*c*np.sqrt(np.log1p((z/c)**2))


def _soft(z):
    """Soft-L1 residual in units of the noise: z when small, growing like sqrt(2|z|c) beyond ROBUST_SCALE."""
    c = ROBUST_SCALE
    return np.sign(z)*c*np.sqrt(2*(np.sqrt(1+(z/c)**2)-1))


def _rotation_vector(R):
    """The rotation vector of a rotation matrix (the inverse of _rotations)."""
    rvec, _ = cv2.Rodrigues(np.asarray(R, np.float64))
    return rvec.ravel()


def _rotations(rvecs):
    """Rotation matrices (B, 3, 3) of rotation vectors (B, 3) (Rodrigues' formula, as cv2.Rodrigues)."""
    rvecs = np.asarray(rvecs, np.float64).reshape(-1, 3)
    angle = np.linalg.norm(rvecs, axis=1)
    k = rvecs/np.where(angle > 1e-12, angle, 1.0)[:, None]
    K = np.zeros((len(rvecs), 3, 3))
    K[:, 0, 1], K[:, 0, 2], K[:, 1, 2] = -k[:, 2], k[:, 1], -k[:, 0]
    K[:, 1, 0], K[:, 2, 0], K[:, 2, 1] = k[:, 2], -k[:, 1], k[:, 0]
    return np.eye(3)+np.sin(angle)[:, None, None]*K+(1-np.cos(angle))[:, None, None]*(K @ K)


def _local_cameras(Theta, layout):
    """(phi (B,), R (B, 3, 3), t (B, 3), H local metres -> normalized image (B, 3, 3), half width (B,),
    people height (B,)) of parameter vectors (B, n)."""
    Theta = np.atleast_2d(np.asarray(Theta, np.float64))
    phi = np.exp(Theta[:, 0])
    R = _rotations(Theta[:, 1:4])
    t = Theta[:, 4:7]
    Hl = np.stack([R[:, :, 0], R[:, :, 1], t], axis=2)*np.stack([phi, phi, np.ones_like(phi)], axis=1)[:, :, None]
    half = Theta[:, layout['half']] if 'half' in layout else np.full(len(Theta), layout['half_fixed'])
    person = np.exp(Theta[:, layout['person']]) if 'person' in layout else np.full(len(Theta), PERSON_HEIGHT)
    return phi, R, t, Hl, half, person


def _local_camera(theta, layout):
    """_local_cameras of one parameter vector: (phi, R, t, H local metres -> normalized image, half width, people height)."""
    phi, R, t, Hl, half, person = _local_cameras(theta, layout)
    return float(phi[0]), R[0], t[0], Hl[0], float(half[0]), float(person[0])


def _residual_rows(Theta, data):
    """Residuals in units of their noise, one row per parameter vector (B, n) -> (B, m): the circle points'
    Sampson distance to the circle's image, the painted lines' samples' distance to the predicted lines (soft
    L1), the people's log height ratio (Cauchy). With data['robust'] False the distances are returned as they
    are (the fit's true residual, which the robust losses understate)."""
    phi, R, t, Hl, half, person = _local_cameras(Theta, data['layout'])
    B = len(phi)
    try:
        Hi = np.linalg.inv(Hl)
    except np.linalg.LinAlgError:
        return np.full((B, data['size']), 1e3)
    robust = data.get('robust', True)
    soft = _soft if robust else (lambda z: z)
    sigma = data['sigma']
    HiT = np.transpose(Hi, (0, 2, 1))
    parts = []
    C = HiT @ np.diag([1.0, 1.0, -field.RADIUS**2]) @ Hi
    P = data['circle']
    CP = np.einsum('nj,bjk->bnk', P, C)
    distance = np.einsum('nj,bnj->bn', P, CP)/np.maximum(2*np.hypot(CP[..., 0], CP[..., 1]), 1e-12)
    parts.append(soft(distance/sigma)*data['circle_weight'])
    for kind, samples, weight in data['lines']:
        if isinstance(kind, str):
            world = np.zeros((B, 3))
            if kind == 'halfway':
                world[:, 0] = 1.0
            else:
                world[:, 1], world[:, 2] = 1.0, half if kind == 'far' else -half
        else:
            world = np.broadcast_to(kind, (B, 3))
        l = np.einsum('bjk,bk->bj', HiT, world)
        norm = np.maximum(np.hypot(l[:, 0], l[:, 1]), 1e-12)
        parts.append(soft((l @ samples.T)/norm[:, None]/sigma)*weight)
    if data['feet'] is not None:
        F = data['feet']
        G = np.einsum('nj,bkj->bnk', F, Hi)
        w = G[..., 2]
        safe = np.where(np.abs(w) > 1e-12, w, 1e-12)
        gx, gy = G[..., 0]/safe, G[..., 1]/safe
        X = gx[..., None]*R[:, None, :, 0]+gy[..., None]*R[:, None, :, 1]-person[:, None, None]*R[:, None, :, 2]+t[:, None, :]
        depth = X[..., 2]
        head = phi[:, None]*X[..., 1]/np.where(np.abs(depth) > 1e-12, depth, 1e-12)
        predicted = F[None, :, 1]-head
        ok = (w > 0) & (depth > 0) & (predicted > 1e-6)   # w = 1/depth of the foot: positive in front of the camera
        ratio = np.where(ok, np.log(data['heights'][None, :]/np.where(ok, predicted, 1.0))/PERSON_SIGMA, 10.0)
        parts.append(_cauchy(ratio) if robust else ratio)
        parts.append(((np.log(person)-math.log(PERSON_HEIGHT))/PERSON_PRIOR)[:, None])
    return np.concatenate(parts, axis=1)


def _pinhole_residuals(theta, data):
    """_residual_rows of one parameter vector."""
    return _residual_rows(np.asarray(theta, np.float64)[None], data)[0]


def _pinhole_jacobian(theta, data, bounds):
    """Forward-difference Jacobian with every perturbed vector evaluated in one batched call (scipy's
    default steps: sqrt(eps) x max(1, |x|), turned back inside the bounds)."""
    x0 = np.asarray(theta, np.float64)
    lo, hi = bounds
    h = np.sqrt(np.finfo(np.float64).eps)*np.where(x0 >= 0, 1.0, -1.0)*np.maximum(1.0, np.abs(x0))
    h = np.where((x0+h > hi) | (x0+h < lo), -h, h)
    X = np.repeat(x0[None], len(x0)+1, axis=0)
    X[np.arange(1, len(x0)+1), np.arange(len(x0))] += h
    F = _residual_rows(X, data)
    return ((F[1:]-F[0][None])/((x0+h)-x0)[:, None]).T


def _samples(a, b, T):
    t = np.linspace(0, 1, LINE_SAMPLES)
    pts = np.c_[a[0]+(b[0]-a[0])*t, a[1]+(b[1]-a[1])*t, np.ones(LINE_SAMPLES)]
    return pts @ T.T


def _usable_people(boxes, width, height):
    """Standing people for the height evidence: normalized [x, y, w, h] boxes clear of the frame edges, of
    a standing person's shape (taller than wide) and not tiny. Returns [(foot x, foot y, height)] normalized."""
    out = []
    for x, y, w, h in boxes or ():
        if h <= 0 or w <= 0 or y < .01 or y+h > .99 or x < .005 or x+w > .995:
            continue
        if h*height < PERSON_MIN_ROWS*height/1080 or h*height < PERSON_ASPECT*w*width:
            continue
        out.append((x+w/2, y+h, h))
    return out


def pinhole_solve(circle, halfway, touchlines, width, height, size=None, masks=None, people=(), markings=()):
    """Register the pitch as a pinhole camera (square pixels, no skew, the principal point at the image
    centre; one unknown intrinsic, the focal length) from the centre circle and the halfway line, with any
    touchlines (which then also measure the pitch width) and the heights of the people standing on the
    pitch (which fix the horizon) as further evidence.

    Seeds: focal_seeds and seed_poses. Fit: least squares over the focal length, the rotation, the position
    (and the half width with a touchline, the people's height with people), the circle weighing as
    CIRCLE_EQUIVALENT observations. Of the solutions, the one most independent paint agrees with wins
    (_confirmations); the detected segments matched to it then join a second fit. Honest confidence: the
    fit's covariance is carried to metres on the visible pitch (sigmaMetres the median, sigmaFarMetres the
    90th percentile, which scales the observability); without an independent painted line seen finely
    enough the observability is capped at PINHOLE_CAP, and lower still for a flat ellipse. Refused when the
    solution is implausible (camera height, field of view, the pitch upside down or behind the camera, the
    camera over the far half), ambiguous, or when the paint or the people disagree with it. Returns the
    solve() dictionary, with 'cap' (the observability ceiling, 1 when confirmed) and 'fitted' (the painted
    lines the fit used); a refusal because the evidence contradicts the camera carries 'contradiction'.

    Measured limits (synthetic renders and the users' footage): a camera in or near the vertical plane of
    the halfway line, the usual midfield camera (both users' cameras stand within 4 m of it), leaves the
    focal length unobservable when it looks along the halfway line and gives two mirror-image solutions
    otherwise; a far circle seen at a grazing angle fixes the depth poorly. Then only the other paint, the
    touchlines and the people decide, and without them the route refuses."""
    size = size or field.PitchSize()
    W, Hh = width, height
    if circle is None or halfway is None:
        return {'reason': 'no centre circle or halfway line'}
    circle_px = [(x*W, y*Hh) for x, y in circle['points']]
    ellipse = ellipse_from_points(circle_px)
    if ellipse is None:
        return {'reason': 'the circle paint does not fit an ellipse'}
    T, s = _normalizer(W, Hh)
    Ti = np.linalg.inv(T)
    Cn = Ti.T @ conic(ellipse) @ Ti
    Cn /= np.linalg.norm(Cn)
    ha, hb = (halfway['a'][0]*W, halfway['a'][1]*Hh), (halfway['b'][0]*W, halfway['b'][1]*Hh)
    h = _line(ha, hb)
    if h is None:
        return {'reason': 'degenerate halfway line'}
    hn = Ti.T @ h
    hn /= np.linalg.norm(hn)
    try:
        v = np.linalg.solve(Cn, hn)
    except np.linalg.LinAlgError:
        return {'reason': 'degenerate circle'}
    v /= np.linalg.norm(v)
    flat = min(ellipse[1])/max(ellipse[1])
    step = max(1, len(circle_px)//CIRCLE_SAMPLES)
    P = np.c_[np.asarray(circle_px[::step], np.float64), np.ones(len(circle_px[::step]))] @ T.T
    halfway_only = [{'name': 'halfway line', 'a': ha, 'b': hb}]
    used = list(halfway_only)
    for side, seg in touchlines or ():
        if side in ('far', 'near') and all(l['name'] != f'{side} touchline' for l in used):
            used.append({'name': f'{side} touchline', 'a': (seg['a'][0]*W, seg['a'][1]*Hh), 'b': (seg['b'][0]*W, seg['b'][1]*Hh)})
    standing = _usable_people(people, W, Hh)
    seeds, spread = focal_seeds(Cn, v, W, s)
    if spread < MIN_SPREAD and len(used) == 1 and len(standing) < MIN_PEOPLE:
        return {'reason': 'pinhole camera: the view is symmetric about the halfway line, so the circle and the halfway line alone cannot tell '
                          'the focal length (no touchline and too few people to add the horizon)'}

    def fits(evidence):
        data = _evidence(P, evidence, standing, W, Hh, T, size)
        found = []
        for phi in seeds:
            for _, R0, t0 in seed_poses(Cn, v, phi):
                rvec = _rotation_vector(R0)
                c = _fit(np.r_[math.log(phi), rvec, t0], data, W, Hh, T, size)
                c['used'] = evidence
                found.append(c)
        return [c for c in found if 'H' in c], [c for c in found if 'H' not in c]
    solutions, refusals = fits(used)
    if len(used) > 1:
        # The detector's touchline candidates are slope-classified and sometimes a box line or the boards' edge
        # (a far boundary a few pixels off the paint is metres off on the ground). The camera is also solved
        # without them when something else fixes it (the people's horizon, an asymmetric view); a solution must
        # hold its touchlines (_touchlines_hold), and the paint decides between the two below.
        solutions = [c for c in solutions if _touchlines_hold(c, used, T)]
        if len(standing) >= MIN_PEOPLE or spread >= MIN_SPREAD:
            alone, more = fits(halfway_only)
            solutions, refusals = solutions+alone, refusals+more
    if not solutions:
        reason = refusals[0]['reason'] if refusals else 'no camera explains the circle and the halfway line'
        return {'reason': f'pinhole camera: {reason}'}
    # Near the vertical plane of the halfway line two cameras (one a mirror image) explain the circle and its
    # diameter alike; the paint each predicts elsewhere (touchlines, box lines) tells them apart: the paint seen
    # finely first (a far line a pixel off is metres off), then all of it, then the solution on more evidence.
    for c in solutions:
        c['support'], c['fine'], c['confirmed'], c['agreement'], c['matched'] = _confirmations(c, c['used'], masks, markings, W, Hh, size)
    rank = lambda c: (-len(c['fine']), -len(c['support']), -len(c['used']))
    solutions.sort(key=lambda c: rank(c)+(c['cost'],))
    best = solutions[0]
    used = best['used']
    for other in solutions[1:]:
        if rank(other) == rank(best) and other['used'] is used and other['cost']-best['cost'] <= AMBIGUOUS_CHI2:
            d = registration._disagreement(best['H'], other['H'], W, Hh, size)
            if d > DISAGREE_M:
                return {'reason': f"pinhole camera: ambiguous (focal lengths {best['focal']:.0f} and {other['focal']:.0f} px explain the paint "
                                  f"alike, {d:.1f} m apart)", 'contradiction': True}
    # The detected segments the chosen camera predicts onto markings within JOIN_TOLERANCE, and the touchlines
    # it finds on the ground, join the fit (a touchline then measures the width); one the joint fit cannot
    # explain is dropped again, touchlines the joint fit does not hold (_touchlines_hold: a width at its
    # bound, a touchline off its prediction) are dropped together, and a joint fit that fails validation
    # leaves the first one standing. The confirmation that chose the camera is kept for its score.
    tolerance = JOIN_TOLERANCE*W/640
    names = {l['name'] for l in used}
    extra = [{'name': m['name'], 'a': m['a'], 'b': m['b']} for m in best['matched'] if m['error'] <= tolerance and 'touchline' not in m['name']]
    extra += [l for l in _ground_touchlines(best, markings, W, Hh, T) if l['name'] not in names]
    while extra:
        joined = used+extra
        refit = _fit(best['theta'][:7], _evidence(P, joined, standing, W, Hh, T, size), W, Hh, T, size)
        if 'H' not in refit:
            break
        if not _touchlines_hold(refit, joined, T):
            if not any('touchline' in l['name'] for l in extra):
                break   # the first fit's own touchline no longer holds: the joint fit is not better
            extra = [l for l in extra if 'touchline' not in l['name']]
            continue
        refit['support'], refit['fine'], refit['confirmed'] = best['support'], best['fine'], best['confirmed']
        refit['agreement'] = _confirmations(refit, joined, masks, (), W, Hh, size)[3]
        # The line the joint fit explains worst (off its prediction, or its prediction off the paint) goes first.
        misfit = [(float(np.sqrt(np.mean(_line_residuals(refit, l, T)**2)))/JOIN_RMS, l['name']) for l in extra]
        if refit['agreement'] is not None:
            misfit += [(MIN_PAINT/max(refit['agreement'][l['name']], 1e-3), l['name']) for l in joined if refit['agreement'].get(l['name']) is not None]
        worst, name = max(misfit)
        if worst > 1:
            if all(l['name'] != name for l in extra):
                break   # the paint now leaves a line of the first fit: the joint fit is not better
            extra = [l for l in extra if l['name'] != name]
            continue
        result = _pinhole_validated(refit, joined, circle_px, ellipse, flat, spread, len(standing), people, masks, W, Hh, size)
        if 'H' in result:
            return result
        break
    return _pinhole_validated(best, used, circle_px, ellipse, flat, spread, len(standing), people, masks, W, Hh, size)


def _touchlines_hold(c, used, T):
    """Does a solution explain its touchlines: a width inside the plausible range (not at its bound) and within
    2 TOUCHLINE_REACH of the assumed (or learned) one, and each touchline within JOIN_RMS of the prediction?"""
    if 'half' in c['layout']:
        width = 2*c['half']
        if not field.WIDTH_RANGE[0]+.5 < width < field.WIDTH_RANGE[1]-.5 or abs(width-c['data']['size_info'].width) > 2*TOUCHLINE_REACH:
            return False
    return all(float(np.sqrt(np.mean(_line_residuals(c, l, T)**2))) <= JOIN_RMS for l in used if 'touchline' in l['name'])


def _ground_touchlines(c, markings, W, Hh, T):
    """Touchlines found through the solved camera: detected segments it puts on the ground parallel to the
    touchlines (TOUCHLINE_GROUND_DEG), beyond the circle and at a plausible half width; on each side the one
    nearest the touchline the camera predicts (its half width, assumed or measured), within TOUCHLINE_REACH
    of it, not the outermost: the straight things beyond a touchline (board feet, a running track's edge, a
    fence base, the next pitch's line) lie in the same band and would stretch the measured width, and when
    the touchline itself is not detected one of them is all there is. The slope in the image misleads (an
    elevated camera's near touchline is steep and the detector calls it a goal line), and so does the pole
    of the halfway line, which a grazing ellipse moves by hundreds of pixels (amateur 3: 22 degrees off the
    near touchline at 157.5 s). The width they measure is the fit's (the half width is a parameter once a
    touchline joins)."""
    _, _, _, Hl, _, _ = _local_camera(c['theta'], c['layout'])
    Hi = np.linalg.inv(Hl)
    best = {}
    for m in markings or ():
        a, b = (m['a'][0]*W, m['a'][1]*Hh), (m['b'][0]*W, m['b'][1]*Hh)
        G = np.array([[a[0], a[1], 1.0], [b[0], b[1], 1.0]]) @ T.T @ Hi.T
        if (G[:, 2] <= 0).any():
            continue   # an end beyond the horizon
        g = G[:, :2]/G[:, 2:3]
        dx, dy = abs(g[1, 0]-g[0, 0]), abs(g[1, 1]-g[0, 1])
        half = abs(float(g[:, 1].mean()))
        if dx < 1e-6 or math.degrees(math.atan2(dy, dx)) > TOUCHLINE_GROUND_DEG or not TOUCHLINE_MIN_HALF <= half <= field.WIDTH_RANGE[1]/2:
            continue
        side = 'far' if g[:, 1].mean() < 0 else 'near'
        gap = abs(half-c['half'])
        if gap <= TOUCHLINE_REACH and (side not in best or gap < best[side][0]):
            best[side] = (gap, {'name': f'{side} touchline', 'a': a, 'b': b})
    return [best[side][1] for side in ('far', 'near') if side in best]


def _line_residuals(c, line, T):
    """A painted line's sample distances to the candidate's prediction of it, in noise sigmas."""
    _, _, _, Hl, half, _ = _local_camera(c['theta'], c['layout'])
    kind = _world_line(line['name'], c['data']['size_info'])
    world = kind if not isinstance(kind, str) else np.array([1.0, 0.0, 0.0]) if kind == 'halfway' else np.array([0.0, 1.0, half if kind == 'far' else -half])
    l = np.linalg.inv(Hl).T @ world
    return (_samples(line['a'], line['b'], T) @ l)/max(math.hypot(l[0], l[1]), 1e-12)/c['data']['sigma']


def _world_line(name, size):
    """A marking line in the circle-centred frame (metres) as homogeneous coefficients, or the kind of a
    touchline (its distance from the centre is the fitted half width)."""
    if name == 'halfway line':
        return 'halfway'
    if name in ('far touchline', 'near touchline'):
        return name.split()[0]
    for n, a, b in field.lines(size):
        if n == name:
            l = np.cross([a[0]-size.length/2, a[1]-size.width/2, 1.0], [b[0]-size.length/2, b[1]-size.width/2, 1.0])
            return l/math.hypot(l[0], l[1])
    return None


def _evidence(P, used, standing, W, Hh, T, size):
    """The residual data of a fit: the circle points and the painted lines (pixels, named) in normalized
    coordinates, the standing people (when MIN_PEOPLE), and the parameter layout they imply."""
    s = float(max(W, Hh))
    weight = math.sqrt(LINE_EQUIVALENT/LINE_SAMPLES)
    lines = []
    for l in used:
        kind = _world_line(l['name'], size)
        if kind is not None:
            lines.append((kind, _samples(l['a'], l['b'], T), weight))
    feet = heights = None
    if len(standing) >= MIN_PEOPLE:
        q = np.array([(fx*W, fy*Hh) for fx, fy, _ in standing], np.float64)
        feet = np.c_[q, np.ones(len(q))] @ T.T
        heights = np.array([bh*Hh/s for _, _, bh in standing], np.float64)
    layout = {'half_fixed': size.width/2}
    n = 7
    if any(isinstance(kind, str) and kind in ('far', 'near') for kind, _, _ in lines):
        layout['half'], n = n, n+1
    if feet is not None:
        layout['person'], n = n, n+1
    layout['n'] = n
    data = {'layout': layout, 'circle': P, 'circle_weight': math.sqrt(CIRCLE_EQUIVALENT/len(P)), 'lines': lines, 'size_info': size,
            'sigma': PAINT_SIGMA*(W/640)/s,   # pitch.LINE_W: the paint was found on a 640-px wide mask
            'feet': feet, 'heights': heights}
    data['size'] = len(P)+LINE_SAMPLES*len(lines)+(len(feet)+1 if feet is not None else 0)
    return data


def _fit(base, data, W, Hh, T, size):
    """One least-squares fit from a start (log focal, rotation vector, position), the half width and the
    people's height starting from the standard values. Returns _pinhole_candidate's dictionary."""
    layout = data['layout']
    s = float(max(W, Hh))
    phi_lo, phi_hi = _phi_range(W, s)
    lo, hi = np.full(layout['n'], -np.inf), np.full(layout['n'], np.inf)
    lo[0], hi[0] = math.log(phi_lo), math.log(phi_hi)
    theta = np.r_[np.asarray(base, np.float64)[:7], np.zeros(layout['n']-7)]
    if 'half' in layout:
        lo[layout['half']], hi[layout['half']] = field.WIDTH_RANGE[0]/2, field.WIDTH_RANGE[1]/2
        theta[layout['half']] = size.width/2
    if 'person' in layout:
        lo[layout['person']], hi[layout['person']] = math.log(PERSON_RANGE[0]), math.log(PERSON_RANGE[1])
        theta[layout['person']] = math.log(PERSON_HEIGHT)
    theta = np.clip(theta, lo+1e-9, hi-1e-9)
    try:
        fit = least_squares(_pinhole_residuals, theta, jac=lambda x, d: _pinhole_jacobian(x, d, (lo, hi)), args=(data,), bounds=(lo, hi),
                            x_scale='jac', ftol=FIT_TOL, xtol=FIT_TOL, max_nfev=200)
    except (ValueError, np.linalg.LinAlgError):
        return {'reason': 'the fit diverged'}
    return _pinhole_candidate(fit, data, W, Hh, T, size)


def _measured(c, size):
    """The candidate's homography and pitch size with the width it measured (the assumed one without a touchline)."""
    if 'half' in c['layout']:
        return c['H_measured'], field.PitchSize(size.length, 2*c['half'], 'measured')
    return c['H'], size


def _in_view(segment, W, Hh):
    """Length (pixels) of an image segment's part inside the frame."""
    if segment is None or not all(abs(v) < 1e7 for point in segment for v in point):
        return 0.0
    ok, p, q = cv2.clipLine((0, 0, int(W), int(Hh)), tuple(int(round(v)) for v in segment[0]), tuple(int(round(v)) for v in segment[1]))
    return math.hypot(q[0]-p[0], q[1]-p[1]) if ok else 0.0


def _resolution(H, a, b, W, Hh):
    """Metres on the ground per image pixel across a line (pixels a-b) at the middle of its part in view: how
    finely paint lying on its predicted image pins the line down. None when it is not in view."""
    if a is None or not all(abs(v) < 1e7 for point in (a, b) for v in point):
        return None
    ok, p, q = cv2.clipLine((0, 0, int(W), int(Hh)), tuple(int(round(v)) for v in a), tuple(int(round(v)) for v in b))
    length = math.hypot(q[0]-p[0], q[1]-p[1]) if ok else 0.0
    if length < 1:
        return None
    mid = np.array([(p[0]+q[0])/2, (p[1]+q[1])/2])
    normal = np.array([-(q[1]-p[1]), q[0]-p[0]])/length
    probes = [mid, mid+normal]
    if (field.image_depth(H, probes) <= 1e-9).any():
        return None
    g = field.unproject(H, probes)
    return float(np.hypot(*(g[1]-g[0])))


def _confirmations(c, used, masks, markings, W, Hh, size):
    """Paint that agrees with a candidate camera besides the circle and the halfway line, judged alike for
    the markings the fit used and the others: a marking whose predicted image lies on paint along at least
    CONFIRM_LENGTH of the frame, or one a detected segment is matched to; a touchline of the fit only when it
    measured a width inside the plausible range (not pressed against its bound). Returns (support: their
    names, which tell mirror-image cameras apart; fine: those pinned to CONFIRM_METRES on the ground (the
    match's pixel error, or the paint test's reach, times the metres per pixel across the line), used or not,
    which rank the candidates; confirmed: the fine ones the fit did not use, independent of it, which lift
    the observability ceiling; the paint agreement of every marking in view, or None without masks; the
    matched detected segments of markings the fit did not use, which may join it)."""
    H, sz = _measured(c, size)
    names = {l['name'] for l in used}
    in_range = field.WIDTH_RANGE[0]+.5 < 2*c['half'] < field.WIDTH_RANGE[1]-.5
    support, fine, agreement = [], [], None

    def agree(name, a, b, error_px):
        if name in support or name == 'halfway line' or (name in names and 'touchline' in name and not in_range):
            return
        support.append(name)
        resolution = _resolution(H, a, b, W, Hh)
        if resolution is not None and max(error_px, 1.0)*resolution <= CONFIRM_METRES:
            fine.append(name)
    if masks is not None:
        agreement = field.paint_agreement(H, sz, masks, W, Hh)
        for name, a, b in field.lines(sz):
            if (agreement.get(name) or 0.0) < CONFIRM_PAINT:
                continue
            seg = field.image_segment(H, a, b, samples=96)
            if _in_view(seg, W, Hh) >= CONFIRM_LENGTH*W:
                agree(name, *seg, PAINT_REACH*W/640)
    matched = []
    for m in field.match_segments(H, sz, markings or (), W, Hh):
        if m['support'] < CONFIRM_SUPPORT:
            continue
        agree(m['name'], m['a'], m['b'], m['error'])
        if m['name'] not in names and all(x['name'] != m['name'] for x in matched):
            matched.append(m)
    return support, fine, [n for n in fine if n not in names], agreement, matched


def _pinhole_candidate(fit, data, W, Hh, T, size):
    """A finished fit as a camera: plausibility, the homographies (assumed and measured width), the
    covariance carried to metres on the visible pitch."""
    theta = fit.x
    layout = data['layout']
    phi, R, t, Hl, half, person = _local_camera(theta, layout)
    s = float(max(W, Hh))
    focal = phi*s
    fov = 2*math.degrees(math.atan(W/2/focal))
    camera = -R.T @ t
    elevation = float(R[:, 2] @ t)   # the camera's height above the ground (the normal points into it)
    if not HEIGHT_RANGE[0] <= elevation <= HEIGHT_RANGE[1]:
        return {'reason': f'a camera {elevation:.1f} m above the pitch is implausible'}
    if camera[1] < -size.width/4:
        return {'reason': 'the camera would stand over the far half of the pitch (the far touchline is by definition the one away from it)'}
    lo, hi = _phi_range(W, s)
    if phi <= lo*1.01 or phi >= hi*.99:
        return {'reason': f'the focal length ran to the edge of the plausible range ({fov:.0f} degrees of view)'}
    Ti = np.linalg.inv(T)
    centre = lambda half_width: np.array([[1, 0, -size.length/2], [0, 1, -half_width], [0, 0, 1.0]])
    H = field.orient(Ti @ Hl @ centre(size.width/2), W, Hh)
    if H is None:
        return {'reason': 'the solution is not a pitch plane in front of the camera'}
    why = field.sanity(H, size, W, Hh)
    if why:
        return {'reason': why}
    jac = fit.jac
    m, p = jac.shape
    try:
        information = jac.T @ jac
        if np.linalg.cond(information) > 1e14:
            raise np.linalg.LinAlgError
        cov = np.linalg.inv(information)*max(1.0, 2*fit.cost/max(1, m-p))
    except np.linalg.LinAlgError:
        cov = None
    sigma, far_sigma = _metric_sigma(theta, layout, cov, W, Hh, T, size, half)
    measured = field.orient(Ti @ Hl @ centre(half), W, Hh) if 'half' in layout else H
    return {'H': H, 'H_measured': measured, 'half': float(half), 'person': float(person), 'cost': float(2*fit.cost), 'focal': float(focal), 'fov': fov,
            'height': elevation, 'camera': (float(camera[0]+size.length/2), float(camera[1]+size.width/2)), 'sigma': sigma, 'farSigma': far_sigma,
            'theta': theta, 'layout': layout, 'data': data}


def _metric_sigma(theta, layout, cov, W, Hh, T, size, half):
    """Median and 90th percentile (metres) of the position uncertainty over the visible pitch: image points
    on a grid that see the ground within PEOPLE_MARGIN of the pitch, their ground positions differentiated
    by the fit parameters and propagated through the fit's covariance."""
    if cov is None or not np.isfinite(cov).all():
        return math.inf, math.inf
    xs, ys = np.linspace(.05, .95, 10)*W, np.linspace(.05, .95, 8)*Hh
    grid = np.array([(x, y, 1.0) for y in ys for x in xs]) @ T.T

    steps = 1e-6*np.maximum(1.0, np.abs(theta))
    Theta = np.repeat(theta[None], len(theta)+1, axis=0)
    Theta[np.arange(1, len(theta)+1), np.arange(len(theta))] += steps
    try:
        Hl = _local_cameras(Theta, layout)[3]
        G = np.einsum('nj,bkj->bnk', grid, np.linalg.inv(Hl))
    except np.linalg.LinAlgError:
        return math.inf, math.inf
    X = G[..., :2]/np.where(np.abs(G[..., 2:3]) > 1e-12, G[..., 2:3], 1e-12)
    keep = (G[0, :, 2] > 0) & (np.abs(X[0, :, 0]) <= size.length/2+PEOPLE_MARGIN) & (np.abs(X[0, :, 1]) <= half+PEOPLE_MARGIN)
    if keep.sum() < 4:
        return math.inf, math.inf
    J = np.transpose((X[1:, keep]-X[0, keep][None])/steps[:, None, None], (1, 2, 0))   # (points, 2, parameters)
    var = np.einsum('kij,jl,kil->k', J, cov, J)
    spread = np.sqrt(np.maximum(var, 0.0))
    return float(np.median(spread)), float(np.percentile(spread, 90))


def _pinhole_validated(best, used, circle_px, ellipse, flat, spread, n_people, people, masks, W, Hh, size):
    """Validate the chosen camera against the paint and the people, and score it."""
    H = best['H']
    measured, size_measured = _measured(best, size)
    half = best['half']
    width_measured = size_measured is not size
    data = best['data']
    # The geometric residual (pixels) of the paint alone, without the robust loss that let the fit discount
    # stray paint: what the observability's residual factor and the reported rmsPixels must measure.
    raw = _pinhole_residuals(best['theta'], dict(data, feet=None, robust=False))
    paint = raw[:len(data['circle'])]/data['circle_weight']
    line_part = raw[len(data['circle']):]
    rms_px = float(math.sqrt(np.mean(np.r_[paint, line_part/math.sqrt(LINE_EQUIVALENT/LINE_SAMPLES)]**2)))*data['sigma']*float(max(W, Hh))
    on_circle = float(np.mean(np.abs(paint) <= CIRCLE_TOLERANCE))
    obs = field.observability({'lines': used+[{'name': n} for n in best['confirmed'] if all(n != l['name'] for l in used)], 'circle': circle_px, 'H': H}, W, Hh, size, rms_px=rms_px,
                              circle_agreement=on_circle)
    constraints = ['centre circle', 'halfway line']+[l['name'] for l in used[1:]]+(['people heights'] if data['feet'] is not None else [])
    dimensions = size.dimensions()   # 'assumed' unless the keyframes of the upload learned the size
    if width_measured:
        dimensions['width']['measured'] = round(2*half, 1)
    details = dict(obs, route='pinhole', evidence=[l['name'] for l in used]+['centre circle'], constraints=constraints,
                   intrinsics='square pixels, no skew, principal point at the image centre',
                   focalPx=round(best['focal'], 1), fovDegrees=round(best['fov'], 1), cameraHeightMetres=round(best['height'], 2),
                   cameraPosition=[round(best['camera'][0], 2), round(best['camera'][1], 2)], ellipseRatio=round(flat, 3),
                   focalObservability=round(spread, 3), people=n_people, sigmaMetres=round(best['sigma'], 2) if math.isfinite(best['sigma']) else None,
                   sigmaFarMetres=round(best['farSigma'], 2) if math.isfinite(best['farSigma']) else None, size=dimensions)
    if data['feet'] is not None:
        details['peopleHeightMetres'] = round(best['person'], 2)
    if on_circle < MIN_CIRCLE_AGREEMENT:
        return {'reason': f'pinhole camera: the circle paint does not lie on the image of the canonical circle (agreement {on_circle:.2f})', 'details': details}
    confirmed, agreement = best['confirmed'], best['agreement']
    if agreement is not None:
        details['paint'] = agreement
        evidence = [agreement.get(l['name']) for l in used]
        evidence = [a for a in evidence if a is not None]
        if evidence and min(evidence) < MIN_PAINT:
            return {'reason': f'pinhole camera: reprojected markings leave the paint (agreement {min(evidence):.2f})', 'details': details, 'contradiction': True}
    details['confirmed'] = confirmed
    details['support'] = best['support']
    # The people must stand on the solved pitch.
    feet = [(x+w/2, y+h) for x, y, w, h in people or () if h > 0]
    if len(feet) >= MIN_PEOPLE:
        on = 0
        for fx, fy in feet:
            if field.image_depth(measured, [(fx*W, fy*Hh)])[0] <= 1e-9:
                continue
            gx, gy = field.unproject(measured, [(fx*W, fy*Hh)])[0]
            on += bool(-PEOPLE_MARGIN <= gx <= size_measured.length+PEOPLE_MARGIN and -PEOPLE_MARGIN <= gy <= size_measured.width+PEOPLE_MARGIN)
        details['peopleOnPitch'] = round(float(on)/len(feet), 3)
        if on/len(feet) < PEOPLE_ON_PITCH:
            return {'reason': f'pinhole camera: most detected people ({len(feet)-on} of {len(feet)}) would stand off the solved pitch', 'details': details,
                    'contradiction': True}
    certainty = max(0.0, 1-best['farSigma']/SIGMA_ZERO) if math.isfinite(best['farSigma']) else 0.0
    score = obs['score']*certainty
    cap = 1.0
    if not confirmed:
        cap = PINHOLE_CAP*min(1.0, flat/FLAT_RATIO)
        score = min(score, cap)
    details.update(certainty=round(certainty, 3), cap=round(cap, 3), score=round(score, 3), observabilityBeforeCap=round(obs['score']*certainty, 3))
    if score < MIN_OBSERVABILITY:
        sigma_text = f"{best['farSigma']:.1f} m" if math.isfinite(best['farSigma']) else 'unbounded'
        return {'reason': f'pinhole camera: observability {score:.2f} (position uncertainty up to {sigma_text} on the visible pitch, '
                          f'ellipse ratio {flat:.2f}{", unconfirmed" if not confirmed else ""})', 'details': details}
    marks = field.landmarks(size)
    points = [{'name': n, 'x': float(q[0]), 'y': float(q[1])} for n, q in
              zip(('centre-spot', 'centre-circle-far', 'centre-circle-near', 'halfway-far', 'halfway-near'),
                  field.project(H, [marks[n] for n in ('centre-spot', 'centre-circle-far', 'centre-circle-near', 'halfway-far', 'halfway-near')]))]
    return {'H': H, 'observability': score, 'details': details, 'rms_px': rms_px, 'points': points, 'cap': cap, 'fitted': [l['name'] for l in used]}
