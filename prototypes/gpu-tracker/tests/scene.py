"""Synthetic broadcast-like frames for tests: stands, advertising boards, run-off grass, white
touchlines, a striped pitch and simple people (jersey / shorts / socks / skin)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

W, H = 1280, 720
GRASS = (60, 140, 55)        # BGR
GRASS_DARK = (50, 120, 45)
WHITE = (235, 235, 235)
KITS = {
    'A': {'jersey': (40, 40, 210), 'shorts': (240, 240, 240), 'socks': (40, 40, 210)},    # red / white / red
    'B': {'jersey': (200, 90, 20), 'shorts': (30, 30, 30), 'socks': (200, 90, 20)},       # blue / black / blue
    'REF': {'jersey': (20, 230, 240), 'shorts': (20, 20, 20), 'socks': (20, 20, 20)},     # yellow / black
    'FAN': {'jersey': (120, 70, 160), 'shorts': (80, 60, 50), 'socks': (80, 60, 50)},
}
FAR, NEAR, LEFT, RIGHT = .34, .9, .06, .94  # touchlines and goal lines (normalized)


def pitch_frame(dx=0.0, seed=0, far=FAR, near=NEAR, left=LEFT, right=RIGHT, stands=True):
    """dx: horizontal camera pan in normalized units (the scene moves left by dx)."""
    rng = np.random.default_rng(seed)
    img = np.zeros((H, W, 3), np.uint8)
    ys = np.arange(H)/H
    xs = np.arange(W)/W-dx
    # Stands: colourful noise in the upper part.
    if stands:
        crowd = rng.integers(40, 200, size=(H, W, 3), dtype=np.uint8)
        img[:] = crowd
    # Boards band just above the far run-off.
    board_top, board_bottom = far-.07, far-.04
    img[(ys >= board_top) & (ys < board_bottom)] = (150, 60, 20)
    # Grass from the boards down to the bottom (run-off included), mowing stripes.
    grass_rows = ys >= board_bottom
    stripes = ((xs*12).astype(int) % 2 == 0)
    for y in np.flatnonzero(grass_rows):
        img[y] = np.where(stripes[:, None], GRASS, GRASS_DARK)
    # Near side: a dark track below the near run-off.
    img[ys >= near+.06] = (70, 70, 80)
    thick = 3
    def hline(y):
        r = int(y*H)
        cols = (xs >= left) & (xs <= right)
        img[r-thick//2:r+thick//2+1, cols] = WHITE
    def vline(x, y0=far, y1=near):
        c = int((x+dx)*W)
        if 0 <= c < W:
            img[int(y0*H):int(y1*H), max(0, c-thick//2):c+thick//2+1] = WHITE
    hline(far)
    hline(near)
    vline(left)
    vline(right)
    vline(.5)
    return img


def draw_person(img, box, kit, skin=(140, 170, 210)):
    """box: normalized [x, y, w, h]."""
    x, y, w, h = box
    x1, y1, x2, y2 = int(x*W), int(y*H), int((x+w)*W), int((y+h)*H)
    bh = y2-y1
    colours = KITS[kit] if isinstance(kit, str) else kit
    def band(f0, f1, colour, inset=0.0):
        a, b = y1+int(bh*f0), y1+int(bh*f1)
        ix = int((x2-x1)*inset)
        left, right = max(0, x1+ix), min(img.shape[1], x2-ix)
        if right > left:
            img[max(0, a):max(0, b), left:right] = colour
    band(0, .15, skin, .3)
    band(.15, .48, colours['jersey'], .1)
    band(.48, .7, colours['shorts'], .15)
    band(.7, .88, skin, .25)
    band(.75, .95, colours['socks'], .25)
    band(.95, 1, (20, 20, 20), .25)
    return img


def person_box(foot_x, foot_y, height=.12, ratio=.38):
    w = height*ratio*H/W
    return [foot_x-w/2, foot_y-height, w, height]


def row(box, score=.9, cls=2):
    x, y, w, h = box
    return [x*W, y*H, (x+w)*W, (y+h)*H, score, cls]
