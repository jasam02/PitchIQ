"""Perspective broadcast frames for tests: every pitch marking (touchlines, goal lines, halfway line,
centre circle, penalty and goal areas, spots) is projected from metres through a camera homography and
drawn thin and anti-aliased, with a grass run-off strip, advertising boards (some white) and stands."""
import math

import cv2
import numpy as np

import scene  # noqa: F401  (sys.path setup)
from soccer.geometry import transform_points

W, H = 1280, 720
GRASS, GRASS_DARK = (58, 138, 52), (50, 122, 45)
LENGTH, WIDTH = 105.0, 68.0


def camera(centre_x=52.5, span=70.0, far_y=.30, near_y=.97, tilt=.35):
    """Pitch metres -> image pixels for a side camera looking at centre_x, `span` metres of touchline wide
    at the near side. tilt: how much narrower the far touchline appears (perspective)."""
    near_half, far_half = span/2, span/2*(1+tilt)
    src = np.float32([[centre_x-far_half, 0], [centre_x+far_half, 0], [centre_x+near_half, WIDTH], [centre_x-near_half, WIDTH]])
    dst = np.float32([[0, far_y*H], [W, far_y*H], [W, near_y*H], [0, near_y*H]])
    return cv2.getPerspectiveTransform(src, dst).astype(np.float64)


def _project(P, pts):
    return transform_points(P, pts)


def _poly(P, pts):
    return np.int32(np.round(_project(P, pts)))


def _px_per_m(P, x, y):
    a, b = _project(P, [(x, y), (x+1, y)])
    return float(np.hypot(*(b-a)))


def render(P, seed=0, run_off=4.0, goal_run_off=6.0, board_white=True, line_m=.12, stands=True, spots=True, extra_white_dots=()):
    rng = np.random.default_rng(seed)
    img = rng.integers(40, 200, size=(H, W, 3), dtype=np.uint8) if stands else np.full((H, W, 3), 90, np.uint8)
    # Grass including the run-off around the field, mowing stripes across the pitch.
    xs = np.arange(-goal_run_off, LENGTH+goal_run_off, 5.25)
    for i, x0 in enumerate(xs):
        x1 = min(x0+5.25, LENGTH+goal_run_off)
        quad = [(x0, -run_off), (x1, -run_off), (x1, WIDTH+run_off), (x0, WIDTH+run_off)]
        cv2.fillPoly(img, [_poly(P, quad)], GRASS if i % 2 == 0 else GRASS_DARK)
    # Boards behind both goal lines (standing at the end of the run-off).
    for gx in (-goal_run_off, LENGTH+goal_run_off):
        edge = _project(P, [(gx, y) for y in np.linspace(-run_off, WIDTH+run_off, 30)])
        height = max(8, int(.9*_px_per_m(P, gx, WIDTH/2)))
        for i in range(len(edge)-1):
            (x0, y0), (x1, y1) = edge[i], edge[i+1]
            colour = (245, 245, 245) if board_white and i % 3 != 2 else (30, 30, 200)
            cv2.fillPoly(img, [np.int32([[x0, y0], [x1, y1], [x1, y1-height], [x0, y0-height]])], colour)
    # Advertising boards stand at the far edge of the run-off: a band in the image above it.
    far_edge = _project(P, [(x, -run_off) for x in np.linspace(-40, LENGTH+40, 60)])
    height = max(8, int(.9*_px_per_m(P, 52.5, -run_off)))
    for i in range(len(far_edge)-1):
        (x0, y0), (x1, y1) = far_edge[i], far_edge[i+1]
        colour = (245, 245, 245) if board_white and i % 3 != 2 else ((30, 30, 200) if i % 2 else (160, 60, 10))
        cv2.fillPoly(img, [np.int32([[x0, y0], [x1, y1], [x1, y1-height], [x0, y0-height]])], colour)
        if board_white and i % 3 != 2:
            cv2.line(img, (int(x0+4), int(y0-height/2)), (int(x1-4), int(y1-height/2)), (40, 40, 40), 2)
    # Near side: a dark track below the near run-off.
    near_edge = _project(P, [(x, WIDTH+run_off) for x in np.linspace(-40, LENGTH+40, 60)])
    pts = np.int32(np.vstack([near_edge, [[W*3, H*3], [-W*2, H*3]]]))
    cv2.fillPoly(img, [pts], (70, 70, 80))

    def line(points, metres=line_m):
        pts = _project(P, points)
        mid = points[len(points)//2]
        thick = max(1, int(round(metres*_px_per_m(P, *mid))))
        cv2.polylines(img, [np.int32(np.round(pts*4))], False, (235, 235, 235), thick, cv2.LINE_AA, shift=2)

    seg = lambda a, b, n=40: [(a[0]+(b[0]-a[0])*t, a[1]+(b[1]-a[1])*t) for t in np.linspace(0, 1, n)]
    line(seg((0, 0), (LENGTH, 0), 120))            # far touchline
    line(seg((0, WIDTH), (LENGTH, WIDTH), 120))    # near touchline
    line(seg((0, 0), (0, WIDTH)))                  # left goal line
    line(seg((LENGTH, 0), (LENGTH, WIDTH)))        # right goal line
    line(seg((LENGTH/2, 0), (LENGTH/2, WIDTH)))    # halfway line
    line([(LENGTH/2+9.15*math.cos(a), WIDTH/2+9.15*math.sin(a)) for a in np.linspace(0, 2*math.pi, 80)])
    for x0, sign in ((0, 1), (LENGTH, -1)):
        for depth, half in ((16.5, 20.16), (5.5, 9.16)):
            line(seg((x0, WIDTH/2-half), (x0+sign*depth, WIDTH/2-half)))
            line(seg((x0+sign*depth, WIDTH/2-half), (x0+sign*depth, WIDTH/2+half)))
            line(seg((x0+sign*depth, WIDTH/2+half), (x0, WIDTH/2+half)))
    dots = [(11, WIDTH/2), (LENGTH-11, WIDTH/2), (LENGTH/2, WIDTH/2)] if spots else []
    for x, y in list(dots)+list(extra_white_dots):
        c = _project(P, [(x, y)])[0]
        r = max(2, int(round(.11*_px_per_m(P, x, y))))
        cv2.circle(img, (int(c[0]), int(c[1])), r, (240, 240, 240), -1, cv2.LINE_AA)
    return cv2.GaussianBlur(img, (3, 3), 0)


def image_point(P, x, y):
    p = _project(P, [(x, y)])[0]
    return float(p[0]/W), float(p[1]/H)


def player_px(P, x, y):
    """Approximate player height in pixels standing at pitch (x, y): 1.8 m at the local scale."""
    return 1.8*_px_per_m(P, x, y)
