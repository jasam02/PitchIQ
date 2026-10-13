"""Lens distortion as a measurement: how bent the straight pitch markings are, and the one-parameter division
model that straightens them best.

A pinhole camera maps straight lines to straight lines; wide-angle action cameras (Veo and the like) and
stitched panoramas bend them. The division model p_u = c + (p_d - c) / (1 + k r^2), with r the distance of the
observed point from the image centre in units of the half-diagonal, undoes most of that with one number.
This module estimates k from the paint along the longest marking lines of a frame (the k that makes their
sample points straightest) and reports how much straighter they got, so that a run can state the lens
curvature of its video with a number. It only measures: applying the undistortion to the geometry is a
separate, evidence-gated step.
"""
import math

import numpy as np

K_RANGE = (-.6, .6)       # division-model coefficients considered (half-diagonal units)
MIN_POINTS = 30           # sample points a line needs to count
MIN_GAIN = .2             # share of the straightness residual the best k must remove for the estimate to count


def _centre_and_radius(width, height):
    return np.array([width/2, height/2], np.float64), math.hypot(width, height)/2


def undistort_points(points, k, width, height):
    """Observed pixels -> pixels of the pinhole image (division model)."""
    pts = np.asarray(points, np.float64).reshape(-1, 2)
    c, radius = _centre_and_radius(width, height)
    d = pts-c
    r2 = (d*d).sum(axis=1)/(radius*radius)
    return c+d/(1+k*r2)[:, None]


def distort_points(points, k, width, height, iterations=8):
    """Pinhole pixels -> observed pixels (the inverse of undistort_points, by fixed-point iteration)."""
    pts = np.asarray(points, np.float64).reshape(-1, 2)
    c, radius = _centre_and_radius(width, height)
    d_u = pts-c
    d = d_u.copy()
    for _ in range(iterations):
        r2 = (d*d).sum(axis=1)/(radius*radius)
        d = d_u*(1+k*r2)[:, None]
    return c+d


def straightness(points):
    """Root-mean-square distance (pixels) of points from their best-fitting straight line."""
    pts = np.asarray(points, np.float64).reshape(-1, 2)
    if len(pts) < 3:
        return 0.0
    centred = pts-pts.mean(axis=0)
    _, s, vt = np.linalg.svd(centred, full_matrices=False)
    normal = vt[-1]
    return float(np.sqrt(((centred @ normal)**2).mean()))


def estimate_k(lines, width, height):
    """lines: sample points (pixels) along several straight markings of one frame. Returns {'k', 'gain',
    'residualPx', 'lines'} for the k that makes them straightest, or None when there is too little paint
    or straightening them changes nothing (gain below MIN_GAIN: a pinhole lens, or lines too short to tell)."""
    usable = [np.asarray(l, np.float64).reshape(-1, 2) for l in lines if len(l) >= MIN_POINTS]
    if not usable:
        return None
    weights = np.array([len(l) for l in usable], np.float64)

    def residual(k):
        return float(np.sum([straightness(undistort_points(l, k, width, height))**2*w for l, w in zip(usable, weights)])/weights.sum())
    lo, hi = K_RANGE
    golden = (math.sqrt(5)-1)/2
    a, b = hi-golden*(hi-lo), lo+golden*(hi-lo)
    fa, fb = residual(a), residual(b)
    for _ in range(40):
        if fa < fb:
            hi, b, fb = b, a, fa
            a = hi-golden*(hi-lo)
            fa = residual(a)
        else:
            lo, a, fa = a, b, fb
            b = lo+golden*(hi-lo)
            fb = residual(b)
    k = (lo+hi)/2
    before, after = residual(0.0), residual(k)
    if before <= 1e-9:
        return None
    gain = 1-after/before
    if gain < MIN_GAIN:
        return None
    return {'k': round(k, 4), 'gain': round(gain, 3), 'residualPx': round(math.sqrt(after), 2), 'straightPx': round(math.sqrt(before), 2), 'lines': len(usable)}


def summarize(samples):
    """The lens of a video from the per-frame estimates: the median k of the frames whose markings were
    clearly straightened, the number of such frames and the median gain; None when nothing was measured."""
    good = [s for s in samples if s is not None]
    if not good:
        return None
    ks = sorted(s['k'] for s in good)
    gains = sorted(s['gain'] for s in good)
    return {'k': round(ks[len(ks)//2], 4), 'frames': len(good), 'gain': round(gains[len(gains)//2], 3),
            'agreement': round(float(np.mean(np.sign(ks) == np.sign(ks[len(ks)//2]))), 3)}
