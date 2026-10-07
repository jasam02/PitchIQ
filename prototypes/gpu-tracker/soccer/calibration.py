"""Image -> pitch homography from user-marked pitch landmarks.

There is no automatic metric calibration (no keypoint model ships with this prototype). A user
marks four or more named landmarks on a paused frame; the homography is then carried through pans
and zooms by the camera-motion estimate (camera.py) and dropped when the motion is unreliable, until
the next calibration keyframe. Pitch coordinates: x 0..1 from the left goal line to the right goal
line, y 0..1 from the far touchline to the near touchline. Dimensions assume 105 x 68 m.
"""
import math

import cv2
import numpy as np

from .geometry import transform_points

LENGTH, WIDTH = 105.0, 68.0
_PA, _GA, _SPOT, _CIRCLE = 16.5, 5.5, 11.0, 9.15
_PA_HALF, _GA_HALF = 20.16, 9.16

LANDMARKS = {
    'corner-far-left': (0, 0), 'corner-far-right': (LENGTH, 0),
    'corner-near-left': (0, WIDTH), 'corner-near-right': (LENGTH, WIDTH),
    'halfway-far': (LENGTH/2, 0), 'halfway-near': (LENGTH/2, WIDTH), 'centre-spot': (LENGTH/2, WIDTH/2),
    'centre-circle-far': (LENGTH/2, WIDTH/2-_CIRCLE), 'centre-circle-near': (LENGTH/2, WIDTH/2+_CIRCLE),
    'left-penalty-box-goal-line-far': (0, WIDTH/2-_PA_HALF), 'left-penalty-box-goal-line-near': (0, WIDTH/2+_PA_HALF),
    'left-penalty-box-front-far': (_PA, WIDTH/2-_PA_HALF), 'left-penalty-box-front-near': (_PA, WIDTH/2+_PA_HALF),
    'left-goal-box-goal-line-far': (0, WIDTH/2-_GA_HALF), 'left-goal-box-goal-line-near': (0, WIDTH/2+_GA_HALF),
    'left-goal-box-front-far': (_GA, WIDTH/2-_GA_HALF), 'left-goal-box-front-near': (_GA, WIDTH/2+_GA_HALF),
    'left-penalty-spot': (_SPOT, WIDTH/2),
    'right-penalty-box-goal-line-far': (LENGTH, WIDTH/2-_PA_HALF), 'right-penalty-box-goal-line-near': (LENGTH, WIDTH/2+_PA_HALF),
    'right-penalty-box-front-far': (LENGTH-_PA, WIDTH/2-_PA_HALF), 'right-penalty-box-front-near': (LENGTH-_PA, WIDTH/2+_PA_HALF),
    'right-goal-box-goal-line-far': (LENGTH, WIDTH/2-_GA_HALF), 'right-goal-box-goal-line-near': (LENGTH, WIDTH/2+_GA_HALF),
    'right-goal-box-front-far': (LENGTH-_GA, WIDTH/2-_GA_HALF), 'right-goal-box-front-near': (LENGTH-_GA, WIDTH/2+_GA_HALF),
    'right-penalty-spot': (LENGTH-_SPOT, WIDTH/2),
}


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


def solve(points, width, height, max_error=1.5):
    """points: [{'name', 'x', 'y'}] with x/y normalized image coordinates. Returns (H, rms metres):
    H maps image pixels to pitch metres. Raises ValueError for unusable landmark sets."""
    src, dst = [], []
    for p in points:
        if p['name'] not in LANDMARKS:
            raise ValueError(f"Unknown pitch landmark: {p['name']}")
        src.append((p['x']*width, p['y']*height))
        dst.append(LANDMARKS[p['name']])
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


def to_pitch(H, foot_px):
    """Foot point (pixels) -> normalized pitch coordinates, or None beyond the horizon / far off the pitch."""
    x, y = foot_px
    w = H[2, 0]*x+H[2, 1]*y+H[2, 2]
    if abs(w) < 1e-12:
        return None
    px, py = (H[0, 0]*x+H[0, 1]*y+H[0, 2])/w, (H[1, 0]*x+H[1, 1]*y+H[1, 2])/w
    if not (math.isfinite(px) and math.isfinite(py)):
        return None
    nx, ny = px/LENGTH, py/WIDTH
    if -.15 <= nx <= 1.15 and -.15 <= ny <= 1.15:
        return nx, ny
    return None


def pitch_outline(H, width, height, samples=24):
    """The calibrated pitch rectangle projected into the image (normalized points), clipped to what
    lies in front of the camera."""
    try:
        inverse = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        return None
    t = np.linspace(0, 1, samples, endpoint=False)
    edges = np.concatenate([np.c_[t*LENGTH, np.zeros_like(t)], np.c_[np.full_like(t, LENGTH), t*WIDTH],
                            np.c_[(1-t)*LENGTH, np.full_like(t, WIDTH)], np.c_[np.zeros_like(t), (1-t)*WIDTH]])
    w = np.c_[edges, np.ones(len(edges))] @ inverse[2]
    pts = transform_points(inverse, edges)
    keep = w > 1e-9
    if keep.sum() < 3:
        return None
    pts = pts[keep]/[width, height]
    pts = pts[np.isfinite(pts).all(axis=1)]
    pts = np.clip(pts, -2, 3)
    return [(float(x), float(y)) for x, y in pts]
