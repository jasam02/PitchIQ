"""Classical ball proposals where the detector is blind: small scenes only (BallTracker.small_scene).

The soccer YOLO ball class misses a clearly visible ball at a dribbler's feet on amateur footage: on amateur 3
(uploads/6dfdc88ec6f2d100.mp4) it produced nothing for a ~22 px ball at 146.0-148.8 s at any input size (640 to 1536,
crops at 0.5x-1x, confidence .03) and fired again only once the ball rolled free (149.2 s). This module looks for the
ball the way a person would, in the few places the hidden ball can be (around its last sighting and its prediction):
a compact, round, bright, unsaturated blob that stands out from the grass around it, as large as a ball at that image
row, round and solid (not elongated paint, not a letter of a burned-in watermark), not part of a larger white thing
(a line, a kit) and not on a person's shirt or shins.

A proposal is weak evidence by construction: it carries a low detector score (PROPOSAL_DET), so the ball tracker
accepts one only as part of a chain of consistent sightings and never in preference to a real detector candidate on
the same frame, and the offline decoder treats it as a weak detection.
"""
import math

import cv2
import numpy as np

BRIGHT_DELTA = 60      # the blob's darkest colour channel is this much above the grass's (amateur 3 ball at 146-149 s: 116-136 above;
                       # the turf's own texture stays within ~30)
SATURATION_MAX = 70    # max-min channel spread of white / cream leather (the ball: 4-31; turf 28; blue kit, orange tents > 100)
RATIO_MIN = .8         # blob diameter / expected ball diameter at its row: real ball boxes measure 1.43x the 0.22 m prior on this
RATIO_MAX = 2.6        # footage (p90 2.05, the ball at 146-149 s 1.07-1.95); smaller blobs at the far rows are bits of kit
MIN_PX = 8.0           # px: a ball smaller than this is the detector's business (blobs of 5-7 px are bits of kit at the mid-far rows: 0.93
                       # false proposals per frame over the whole of amateur 3 118-182 s at 6 px, 0.44 at 8 px; the ball at the feet: 13-21 px)
ASPECT_MIN = .6        # minor / major side of the blob's box (the ball: .69-.95; a sock or a line is long)
FILL_MIN = .6          # share of the blob's box it fills (a disk: .79; the ball at 146-149 s: .65-.81 once its dark panels are closed;
                       # .5 let through twice the false proposals on amateur 3 and Veo)
SOLIDITY_MIN = .9      # blob area / its convex hull: the ball .96-1.1 (small blobs measure above 1); letters of a burned-in watermark
                       # (Veo's 'e' and 'o', 47-51 px) and broken pieces of thick paint .75-.8
HOLE_MAX = .12         # share of the blob's outline left open inside it: the ball 0-.07 (its panels are closed first), an 'o' .17
RING_BRIGHT_MAX = .25  # share of bright pixels in the ring around the blob: a ball is not part of a larger white structure (a line
                       # through a ball-sized blob covers about half of it; a ball next to a line .12-.15)
RING_GRASS_MIN = .3    # share of grass in that ring (the ball: .36-.99; legs and shadows next to it cost some)
UPPER_BODY = .72       # box rows above this (from the top, in box heights) are a person's upper body: shirts, arms, faces
SHIN_V = .9            # ... and from there to this row, in the middle SHIN_U of the box, the shins: white socks of a white kit were
SHIN_U = (.25, .75)    # most of the remaining false proposals on amateur 3 (118-182 s); the ball at the feet is lower or beside them
MAX_PROPOSALS = 4      # per frame, the most ball-like first
PROPOSAL_DET = .12     # the detector score a proposal carries: below the tracker's LOW_SCORE (.3) and STRONG_DET (.3) lines
GRASS_MIN = 200        # pixels of grass a window needs before anything in it is judged against the grass
GRASS_STEP = 4         # the grass level is measured on every 4th pixel each way (a full-resolution mask cost 6 ms per megapixel)


def _grass(b, g, r):
    """The fixed turf rule of pitch.py / ball.analyse_patch, on int16 channels."""
    return (g > 30) & (g > r*.95) & (g > b*1.12) & (g-b > 12)


def _on_a_body(cx, cy, people, W, H):
    """Inside a person's upper body (shirt, arms, face), or on the shins in the middle of their box (white socks):
    never a ball proposal. The ball at a player's feet lies at the bottom of the box or beside the legs."""
    for _, (bx, by, bw, bh) in people:
        if bw <= 0 or bh <= 0:
            continue
        u, v = (cx/W-bx)/bw, (cy/H-by)/bh
        if 0 <= u <= 1 and 0 <= v <= UPPER_BODY:
            return True
        if SHIN_U[0] <= u <= SHIN_U[1] and UPPER_BODY < v <= SHIN_V:
            return True
    return False


def find(frame, windows, expected, people=(), taken=()):
    """Ball proposals inside `windows` [(x0, y0, x1, y1) pixels]: a list of (x1, y1, x2, y2, quality) in frame pixels,
    the most ball-like first, at most MAX_PROPOSALS. expected(y) -> expected ball diameter (px) of a ball whose bottom is
    at image row y. people: [(track id or None, normalized box)]. taken: boxes (x1, y1, x2, y2 pixels) the detector
    already reported this frame; nothing is proposed inside them."""
    H, W = frame.shape[:2]
    found = []
    for x0, y0, x1, y1 in windows:
        x0, y0, x1, y1 = max(0, int(x0)), max(0, int(y0)), min(W, int(x1)), min(H, int(y1))
        if x1-x0 < 16 or y1-y0 < 16:
            continue
        patch = frame[y0:y1, x0:x1]
        b, g, r = cv2.split(patch)
        low = cv2.min(cv2.min(b, g), r)
        sat = cv2.subtract(cv2.max(cv2.max(b, g), r), low)
        # The grass level from every GRASS_STEP-th pixel (the median does not need them all).
        sub = patch[::GRASS_STEP, ::GRASS_STEP].astype(np.int16)
        grass = _grass(sub[..., 0], sub[..., 1], sub[..., 2])
        if grass.sum()*GRASS_STEP**2 < GRASS_MIN:
            continue
        grass_low = float(np.median(low[::GRASS_STEP, ::GRASS_STEP][grass]))
        bright = cv2.bitwise_and(cv2.compare(low, min(255.0, grass_low+BRIGHT_DELTA), cv2.CMP_GE), cv2.compare(sat, SATURATION_MAX, cv2.CMP_LE))
        # The ball's dark panels split its bright surface: close them so the blob is one round thing.
        bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        count, labels, stats, _ = cv2.connectedComponentsWithStats(bright, connectivity=8)
        for k in range(1, count):
            bx, by, bw, bh, area = (int(v) for v in stats[k])
            if bx == 0 or by == 0 or bx+bw >= x1-x0 or by+bh >= y1-y0:
                continue   # cut by the window: not judged
            d = math.sqrt(bw*bh)
            if d < MIN_PX:
                continue
            cx, cy = x0+bx+bw/2, y0+by+bh/2
            de = expected(cy+d/2)
            if not de or de <= 0 or not RATIO_MIN <= d/de <= RATIO_MAX:
                continue
            aspect = min(bw, bh)/max(bw, bh)
            if aspect < ASPECT_MIN or area/(bw*bh) < FILL_MIN:
                continue
            mask = (labels[by:by+bh, bx:bx+bw] == k).astype(np.uint8)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            outline = max(contours, key=cv2.contourArea)
            filled = cv2.drawContours(np.zeros_like(mask), [outline], -1, 1, -1)
            if area < SOLIDITY_MIN*cv2.contourArea(cv2.convexHull(outline)) or area < (1-HOLE_MAX)*int(filled.sum()):
                continue   # a notch or a hole: a letter, a broken line, not a round ball
            if _on_a_body(cx, cy, people, W, H) or any(tx1 <= cx <= tx2 and ty1 <= cy <= ty2 for tx1, ty1, tx2, ty2 in taken):
                continue
            # The ring around it: grass (or the ball's shadow), not more of a white structure.
            R = int(math.ceil(2.2*d/2+2))
            lx, ly = bx+bw/2, by+bh/2
            rx0, ry0 = max(0, int(lx)-R), max(0, int(ly)-R)
            rx1, ry1 = min(x1-x0, int(lx)+R+1), min(y1-y0, int(ly)+R+1)
            yy, xx = np.ogrid[ry0:ry1, rx0:rx1]
            dist = np.sqrt((xx-lx)**2+(yy-ly)**2)
            ring = (dist >= 1.3*d/2+1) & (dist <= 2.2*d/2+2)
            if not ring.any():
                continue
            local = patch[ry0:ry1, rx0:rx1].astype(np.int16)
            ring_bright = float(bright[ry0:ry1, rx0:rx1][ring].mean())/255
            ring_grass = float(_grass(local[..., 0], local[..., 1], local[..., 2])[ring].mean())
            if ring_bright > RING_BRIGHT_MAX or ring_grass < RING_GRASS_MIN:
                continue
            quality = ring_grass*aspect
            found.append((cx-bw/2, cy-bh/2, cx+bw/2, cy+bh/2, quality))
    found.sort(key=lambda p: -p[4])
    out = []
    for p in found:   # overlapping windows see the same blob twice
        if all(math.hypot((p[0]+p[2])/2-(q[0]+q[2])/2, (p[1]+p[3])/2-(q[1]+q[3])/2) > max(p[2]-p[0], 4) for q in out):
            out.append(p)
        if len(out) >= MAX_PROPOSALS:
            break
    return out
