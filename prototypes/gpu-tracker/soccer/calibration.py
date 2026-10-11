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


def solve(points, width, height, max_error=1.5, size=None):
    """points: [{'name', 'x', 'y'}] with x/y normalized image coordinates. Returns (H, rms metres):
    H maps image pixels to pitch metres of a pitch of `size` (standard when None). Raises ValueError
    for unusable landmark sets."""
    marks = field.landmarks(size or field.PitchSize())
    src, dst = [], []
    for p in points:
        if p['name'] not in marks:
            raise ValueError(f"Unknown pitch landmark: {p['name']}")
        src.append((p['x']*width, p['y']*height))
        dst.append(marks[p['name']])
    if len(src) < 4 or len({p['name'] for p in points}) != len(points):
        raise ValueError('Mark at least four different pitch landmarks.')
    if not _general_position(src) or not _general_position(dst):
        raise ValueError('The landmarks are (nearly) on one line; choose landmarks that span an area.')
    H, _ = cv2.findHomography(np.asarray(src, np.float64), np.asarray(dst, np.float64), 0)
    if H is None or not np.isfinite(H).all() or abs(np.linalg.det(H)) < 1e-12:
        raise ValueError('These landmarks do not define a pitch plane.')
    projected = transform_points(H, src)
    w = np.c_[np.asarray(src), np.ones(len(src))] @ H[2]
    if (w <= 0).any() and (w >= 0).any():
        raise ValueError('The landmarks fold the pitch; check their names.')
    # Image and pitch keep the same orientation (the Jacobian determinant det(H)/w^3 is positive);
    # a mirrored solution means left/right or near/far labels were swapped.
    if np.linalg.det(H)*np.sign(w.mean()) <= 0:
        raise ValueError('The landmark sides look mirrored (left/right or near/far swapped).')
    rms = float(np.sqrt(((projected-np.asarray(dst))**2).sum(axis=1).mean()))
    if rms > max_error and len(src) > 4:
        raise ValueError(f'The landmarks disagree by {rms:.1f} m; check their positions.')
    return H, rms


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
