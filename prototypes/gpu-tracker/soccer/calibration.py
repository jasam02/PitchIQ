"""Image -> pitch homography from user-marked pitch landmarks.

There is no automatic metric calibration (no keypoint model ships with this prototype). A user
marks four or more named landmarks on a paused frame; the homography anchors the tracked pitch
model (pitch_model.py), which carries it through pans and zooms by the camera-motion estimate
(camera.py), confirms it against painted lines and lets it go when its confidence runs out. Pitch
coordinates: x 0..1 from the left goal line to the right goal line, y 0..1 from the far touchline to
the near touchline. The pitch is 105 x 68 m unless the landmarks of all keyframes together show a
different size (field.fit_keyframes), which is then kept for the whole run.
"""
import math

import cv2
import numpy as np

from . import field
from .geometry import transform_points

LENGTH, WIDTH = field.DEFAULT_LENGTH, field.DEFAULT_WIDTH
LANDMARKS = field.landmarks(field.PitchSize())   # the standard pitch; the names the UI offers


def _general_position(points):
    """Some 4 points with no 3 (nearly) collinear must exist, otherwise H is not determined."""
    pts = np.asarray(points, np.float64)
    spread = np.ptp(pts, axis=0).max() or 1
    q = (pts-pts.mean(axis=0))/spread
    area = lambda a, b, c: abs((b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0]))
    n = len(q)
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                if area(q[i], q[j], q[k]) <= 2e-2:
                    continue
                for m in range(k+1, n):
                    if area(q[i], q[j], q[m]) > 2e-2 and area(q[i], q[k], q[m]) > 2e-2 and area(q[j], q[k], q[m]) > 2e-2:
                        return True
    return False


OUTLIER_MIN = 6          # landmarks from which one clearly wrong click is found and left out
OUTLIER_METRES = 3.0     # a landmark this far from the plane the others define, and far worse than them, is a wrong click


def solve(points, width, height, max_error=1.5, size=None):
    """points: [{'name', 'x', 'y'}] with x/y normalized image coordinates. Returns (H, rms metres):
    H maps image pixels to pitch metres of a pitch of `size` (standard when None). Raises ValueError
    for unusable landmark sets."""
    details = solve_details(points, width, height, max_error, size)
    return details['H'], details['rms']


def solve_details(points, width, height, max_error=1.5, size=None):
    """solve() with the details: {'H', 'rms', 'inliers' (names used), 'outliers' (a clearly wrong click left
    out when six or more landmarks were given: more than OUTLIER_METRES off the plane the others define and
    at least three times the others' median error), 'points' (the inlier points in pixels)}. Measured on
    real keyframes, a Jacobian condition number adds nothing for point sets that pass the general-position
    check, so the observability of a keyframe rests on its spread, axes and residual."""
    size = size or field.PitchSize()
    marks = field.landmarks(size)
    src, dst, names = [], [], []
    for p in points:
        if p['name'] not in marks:
            raise ValueError(f"Unknown pitch landmark: {p['name']}")
        src.append((p['x']*width, p['y']*height))
        dst.append(marks[p['name']])
        names.append(p['name'])
    if len(src) < 4 or len(set(names)) != len(points):
        raise ValueError('Mark at least four different pitch landmarks.')
    if not _general_position(src) or not _general_position(dst):
        raise ValueError('The landmarks are (nearly) on one line; choose landmarks that span an area.')
    src_a, dst_a = np.asarray(src, np.float64), np.asarray(dst, np.float64)
    keep = np.ones(len(src), bool)

    def fit(mask):
        Hm, _ = cv2.findHomography(src_a[mask], dst_a[mask], 0)
        if Hm is None or not np.isfinite(Hm).all() or abs(np.linalg.det(Hm)) < 1e-12:
            raise ValueError('These landmarks do not define a pitch plane.')
        errors = np.hypot(*(transform_points(Hm, src_a[mask])-dst_a[mask]).T)
        return Hm, errors
    H, errors = fit(keep)
    if len(src) >= OUTLIER_MIN:
        worst = int(np.argmax(errors))
        if errors[worst] > OUTLIER_METRES and errors[worst] >= 3*max(.05, float(np.median(errors))):
            proposed = keep.copy()
            proposed[worst] = False
            if _general_position(src_a[proposed]) and _general_position(dst_a[proposed]):
                try:
                    H2, errors2 = fit(proposed)
                except ValueError:
                    H2 = None
                if H2 is not None and float(np.sqrt((errors2**2).mean())) < float(np.sqrt((errors**2).mean()))*.5:
                    keep, H, errors = proposed, H2, errors2
    w = np.c_[src_a[keep], np.ones(int(keep.sum()))] @ H[2]
    if (w <= 0).any() and (w >= 0).any():
        raise ValueError('The landmarks fold the pitch; check their names.')
    # Image and pitch keep the same orientation (the Jacobian determinant det(H)/w^3 is positive);
    # a mirrored solution means left/right or near/far labels were swapped.
    if np.linalg.det(H)*np.sign(w.mean()) <= 0:
        raise ValueError('The landmark sides look mirrored (left/right or near/far swapped).')
    rms = float(np.sqrt((errors**2).mean()))
    if rms > max_error and int(keep.sum()) > 4:
        raise ValueError(f'The landmarks disagree by {rms:.1f} m; check their positions.')
    inliers = [n for n, k in zip(names, keep) if k]
    px_points = [{'name': n, 'x': float(x), 'y': float(y)} for n, (x, y), k in zip(names, src, keep) if k]
    return {'H': H, 'rms': rms, 'inliers': inliers, 'outliers': [n for n, k in zip(names, keep) if not k], 'points': px_points}


def learn_size(keyframes, width, height):
    """The pitch size all calibration keyframes agree on, or None when their landmarks cannot tell it
    (the standard size is then used). keyframes: [{'time', 'points': [{'name', 'x', 'y'}]}]."""
    sets = [[{'name': p['name'], 'x': p['x']*width, 'y': p['y']*height} for p in k['points'] if p['name'] in LANDMARKS] for k in keyframes]
    try:
        learned = field.fit_keyframes(sets, width, height)
    except (ValueError, np.linalg.LinAlgError):
        return None
    if learned is None:
        return None
    size, rms = learned
    # A learned size must explain the landmarks clearly better than the standard one does.
    errors = []
    for s in sets:
        if len(s) < 4:
            continue
        try:
            _, e = solve([{'name': p['name'], 'x': p['x']/width, 'y': p['y']/height} for p in s], width, height, max_error=math.inf)
        except ValueError:
            return None
        errors.append(e**2*len(s))
    standard_rms = math.sqrt(sum(errors)/sum(len(s) for s in sets if len(s) >= 4)) if errors else math.inf
    if abs(size.length-LENGTH) < .5 and abs(size.width-WIDTH) < .5:
        return None
    return size if rms < standard_rms*.8 and standard_rms-rms >= .05 else None


def to_pitch(H, foot_px, size=None):
    """Foot point (pixels) -> normalized pitch coordinates, or None beyond the horizon / far off the pitch."""
    size = size or field.PitchSize()
    x, y = foot_px
    w = H[2, 0]*x+H[2, 1]*y+H[2, 2]
    if abs(w) < 1e-12:
        return None
    px, py = (H[0, 0]*x+H[0, 1]*y+H[0, 2])/w, (H[1, 0]*x+H[1, 1]*y+H[1, 2])/w
    if not (math.isfinite(px) and math.isfinite(py)):
        return None
    nx, ny = px/size.length, py/size.width
    if -.15 <= nx <= 1.15 and -.15 <= ny <= 1.15:
        return nx, ny
    return None


def pitch_outline(H, width, height, samples=24, size=None):
    """The calibrated pitch rectangle projected into the image (normalized points), clipped to what
    lies in front of the camera."""
    try:
        inverse = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        return None
    oriented = field.orient(inverse, width, height)
    if oriented is None:
        return None
    return field.outline(oriented, size or field.PitchSize(), width, height, samples)
