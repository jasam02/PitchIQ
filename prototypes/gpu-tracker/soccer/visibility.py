"""Which part of the physical pitch the camera sees on a frame, and what that says about every roster slot
that is not observed on it.

The frame's final homography (calibration.H, world metres -> pixels, re-oriented first: a stored matrix can have lost
its sign) maps the image rectangle, cut at the horizon, onto the ground: `view`, a polygon in normalized pitch
coordinates clipped to the pitch plus a margin. Where a tick recorded the part of the frame that carries image
(`imageArea`, from image_area: a dewarped camera leaves curved black regions at the corners or along the top), that
outline takes the black away, so it is not counted as seen. A slot that is not observed on the frame is then one of:

    C  IN_FRAME_BUT_OCCLUDED-or-missed: its estimated position lies inside the view by more than its
       uncertainty, so the camera should see it (hidden behind someone, or not detected)
    S  OFF_SCREEN: its estimated position lies outside the view by more than its uncertainty, or it was last
       seen leaving through a frame edge (the roster slot's exit edge) and has not been observed since
    U  UNKNOWN_VISIBILITY: no estimate, no view, or too close to the edge to say

('O', observed in frame, is implied by the slot's presence in the frame's people.) The estimates come from
the result's `timeline` (soccer/imputation.py) when it has one; without it, the estimate is the slot's last
observed position with an uncertainty that grows with the time since (2 m + 1.5 m/s), nothing older than 10 s. The
visibility is evidence, never a rule: it is stored per frame and summarised, and nothing is removed
or invented because of it.
"""
import math

import cv2
import numpy as np

from . import field
from .geometry import clip_polygon, point_in_polygon

MARGIN_METRES = 5.0      # the view is clipped to the pitch dilated by this much
HORIZON_SHARE = .02      # points closer to the horizon than this share of the frame's depth range are left out
ESTIMATE_MAX_AGE = 10.0  # seconds a last observation is still used as an estimate
SIGMA_BASE, SIGMA_RATE = 2.0, 1.5   # metres, metres per second: the estimate's uncertainty from its age
MIN_SIGMA = 1.0
NO_IMAGE_LEVEL = 8         # every channel below this: no image (where a dewarp leaves the lens image), never dark grass or shadow
NO_IMAGE_WIDTH = 320       # px: the resolution the no-image region is measured at
NO_IMAGE_MIN_SHARE = .02   # the near-black region touching the frame border must cover this share of the frame to count.
                           # Measured on sampled frames: broadcast 0 (4 of 4 frames), Veo at most .0011, amateur 3 (dewarped
                           # follow camera) .119 at 124 s, .037 at 133 s, .044 at 160 s, .009 at 141 s, 0 at 148 s
NO_IMAGE_MAX_SHARE = .5    # more than this without image is a fade or a black frame, not a dewarp's edge: no outline (the whole
                           # frame counts, as without one). Measured every 2 s of the uploads: amateur 3 at most .211, the Veo
                           # upload 1.0 on one (black) frame, broadcast 0
NO_IMAGE_SIMPLIFY = 1.0    # px at NO_IMAGE_WIDTH: the outline's polygon tolerance


def _channels_max(image):
    return image.max(axis=-1) if image.ndim == 3 else image


def _dark_border(frame, small_height):
    """The cheap structural pre-check of image_area: is any pixel of the frame's border near-black at the measuring
    resolution (each border strip averaged one measuring pixel deep)? A region without image that touches the border
    has such pixels; the frame corners alone would not do, since a virtual camera tilted down meets the edge of the lens
    image as a bulge in the middle of the top border (amateur 3 at 150-158 s: 25 of the 58 sampled frames with at
    least NO_IMAGE_MIN_SHARE black have no black corner). Every broadcast frame stops here (measured: 0 of 152)."""
    h, w = frame.shape[:2]
    kx, ky = max(1, int(math.ceil(w/NO_IMAGE_WIDTH))), max(1, int(math.ceil(h/small_height)))
    rows = [cv2.resize(strip, (NO_IMAGE_WIDTH, 1), interpolation=cv2.INTER_AREA) for strip in (frame[:ky], frame[h-ky:])]
    cols = [cv2.resize(strip, (1, small_height), interpolation=cv2.INTER_AREA) for strip in (frame[:, :kx], frame[:, w-kx:])]
    return any(bool((_channels_max(s) < NO_IMAGE_LEVEL).any()) for s in rows+cols)


def image_area(frame):
    """The part of a frame (BGR or grey pixels) that carries image, as a normalized polygon: the outline of the largest
    region left when the near-black pixels connected to the frame border are taken away; None when those cover less than
    NO_IMAGE_MIN_SHARE of the frame (the whole frame is image: every broadcast frame) or more than NO_IMAGE_MAX_SHARE (a
    fade or a black frame), or leave nothing. For the tick output, so that view_polygon does not count where a dewarp
    left the lens image as seen. Cheap where there is nothing to find: a frame without a near-black border pixel returns after
    looking at its border strips alone (measured at 1080p: 0.05 ms, against 2.4 ms for the whole measurement)."""
    h, w = frame.shape[:2]
    small_height = max(1, int(round(NO_IMAGE_WIDTH*h/w)))
    if not _dark_border(frame, small_height):
        return None
    small = cv2.resize(frame, (NO_IMAGE_WIDTH, small_height), interpolation=cv2.INTER_AREA)
    dark = (_channels_max(small) < NO_IMAGE_LEVEL).astype(np.uint8)
    _, labels = cv2.connectedComponents(dark, connectivity=8)
    border = np.unique(np.r_[labels[0], labels[-1], labels[:, 0], labels[:, -1]])
    no_image = np.isin(labels, border[border > 0])
    if not NO_IMAGE_MIN_SHARE <= no_image.mean() <= NO_IMAGE_MAX_SHARE:
        return None
    contours, _ = cv2.findContours((~no_image).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    outline = cv2.approxPolyDP(max(contours, key=cv2.contourArea), NO_IMAGE_SIMPLIFY, True).reshape(-1, 2)
    if len(outline) < 3:
        return None
    sh, sw = no_image.shape
    return [[round(float(x)/(sw-1), 4), round(float(y)/(sh-1), 4)] for x, y in outline]


def view_polygon(H, width, height, size=None, area=None):
    """The visible pitch as a normalized polygon (x along the length, y across), or None when the frame
    sees no ground in front of the camera. H: world metres -> pixels (3x3, list or array), re-oriented here
    (field.orient: results stored before registration.homography_json kept the sign can hold the matrix with
    its sign lost, the Veo regression window on 272 of 900 frames; a no-op on every matrix stored since).
    area: the normalized outline of the part of the frame that carries image (image_area), or None for the
    whole frame. The outline only takes area away: whether and where the view is cut at the horizon is
    decided by the frame rectangle in both cases (an outline that lost its deepest corners would otherwise
    move the cut towards the horizon and admit the band HORIZON_SHARE leaves out), so the outlined view is
    always part of the plain one. An outline bitten at the corners is not convex: clipping it by a half-plane
    still keeps its region (pieces may be joined along the cut line)."""
    if H is None:
        return None
    H = field.orient(np.asarray(H, np.float64).reshape(3, 3), width, height)
    if H is None:
        return None
    size = size or field.PitchSize()
    try:
        inverse = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        return None
    a, b, c = inverse[2]          # image_depth(p) = a x + b y + c: positive in front of the camera
    corners = [(0.0, 0.0), (float(width), 0.0), (float(width), float(height)), (0.0, float(height))]
    depths = [a*x+b*y+c for x, y in corners]
    if max(depths) <= 0:
        return None
    polygon = [(float(x)*width, float(y)*height) for x, y in area] if area and len(area) >= 3 else corners
    if min(depths) <= 0:
        # Cut the image outline at a line a little in front of the horizon (depth = share of the frame's deepest corner).
        level = HORIZON_SHARE*max(depths)
        if abs(a) < 1e-12 and abs(b) < 1e-12:
            return None
        if polygon is not corners and max(a*x+b*y+c for x, y in polygon) <= level:
            return None       # the image left by the outline lies entirely beyond the cut
        # Two points on the line a x + b y + c = level.
        if abs(b) >= abs(a):
            p0, p1 = (0.0, (level-c)/b), (float(width), (level-c-a*width)/b)
        else:
            p0, p1 = ((level-c)/a, 0.0), ((level-c-b*height)/a, float(height))
        keep = max(corners, key=lambda p: a*p[0]+b*p[1]+c)
        polygon = clip_polygon(polygon, p0, p1, keep)
        if len(polygon) < 3:
            return None
    world = field.unproject(H, polygon)
    if not np.isfinite(world).all():
        return None
    L, W = size.length, size.width
    poly = [(float(x), float(y)) for x, y in world]
    centre = (L/2, W/2)
    for a_, b_ in (((-MARGIN_METRES, -MARGIN_METRES), (L+MARGIN_METRES, -MARGIN_METRES)), ((L+MARGIN_METRES, -MARGIN_METRES), (L+MARGIN_METRES, W+MARGIN_METRES)),
                   ((L+MARGIN_METRES, W+MARGIN_METRES), (-MARGIN_METRES, W+MARGIN_METRES)), ((-MARGIN_METRES, W+MARGIN_METRES), (-MARGIN_METRES, -MARGIN_METRES))):
        if len(poly) < 3:
            return None
        poly = clip_polygon(poly, a_, b_, centre)
    if len(poly) < 3:
        return None
    return [(round(x/L, 4), round(y/W, 4)) for x, y in poly]


def _distance_metres(point, polygon, size):
    """Signed distance (metres) from a normalized point to the polygon's boundary: negative inside."""
    L, W = size.length, size.width
    px, py = point[0]*L, point[1]*W
    poly = [(x*L, y*W) for x, y in polygon]
    best = math.inf
    for i in range(len(poly)):
        (ax, ay), (bx, by) = poly[i], poly[(i+1) % len(poly)]
        dx, dy = bx-ax, by-ay
        length = dx*dx+dy*dy
        t = max(0.0, min(1.0, ((px-ax)*dx+(py-ay)*dy)/length)) if length > 0 else 0.0
        best = min(best, math.hypot(px-ax-t*dx, py-ay-t*dy))
    return -best if point_in_polygon(point, polygon) else best


def classify(estimate, sigma, view, size=None):
    """'C', 'S' or 'U' for an unobserved slot from its estimated normalized position and uncertainty (metres)."""
    if estimate is None or view is None or len(view) < 3:
        return 'U'
    size = size or field.PitchSize()
    d = _distance_metres(estimate, view, size)
    margin = max(MIN_SIGMA, float(sigma))
    if d <= -margin:
        return 'C'
    if d >= margin:
        return 'S'
    return 'U'


def _estimates(result, slots):
    """Per slot per frame index: (position, sigma) from the timeline block when present, else from the last
    observation (aged); None where nothing is known."""
    frames = result.get('frames', [])
    ids = {s['id'] for s in slots}
    timeline = result.get('timeline')
    if timeline and timeline.get('ids'):
        out = {}
        for pid, cols in timeline['ids'].items():
            if pid not in ids:
                continue
            xs, ys, sig = cols.get('x', []), cols.get('y', []), cols.get('sigma', [])
            out[pid] = [None if xs[i] is None or ys[i] is None else ((xs[i], ys[i]), max(sig[i]) if i < len(sig) and sig[i] else SIGMA_BASE)
                        for i in range(min(len(xs), len(ys)))]
        return out, 'timeline'
    last = {}
    out = {pid: [None]*len(frames) for pid in ids}
    for i, f in enumerate(frames):
        for p in f.get('people', []):
            pid = p.get('id')
            if pid in ids and isinstance(p.get('pitch'), list):
                last[pid] = (f['time'], p['pitch'])
        for pid in ids:
            seen = last.get(pid)
            if seen is None:
                continue
            age = f['time']-seen[0]
            if age <= ESTIMATE_MAX_AGE:
                out[pid][i] = (seen[1], SIGMA_BASE+SIGMA_RATE*age)
    return out, 'last-observation'


def annotate(result):
    """Write `view` (normalized polygon) and `visibility` ({slot id: 'C'|'S'|'U'} for the slots not observed on
    the frame) on every frame of the result, and return the summary (shares of slot-frames per class)."""
    frames = result.get('frames', [])
    roster = result.get('roster') or {}
    slots = [s for s in roster.get('slots', []) if s.get('status') not in ('RETIRED', 'LEFT')]
    size = field.PitchSize.from_json(((result.get('tactical') or {}).get('pitchSize')) or {})
    width, height = result['video']['width'], result['video']['height']
    estimates, source = _estimates(result, slots)
    counts = {'O': 0, 'C': 0, 'S': 0, 'U': 0}
    views = cut = 0
    area = None
    for i, f in enumerate(frames):
        if f.get('pitch'):
            area = f['pitch'].get('imageArea')   # the latest tick's outline of the image (absent: the whole frame is image)
        H = (f.get('calibration') or {}).get('H') if f.get('calibrated') else None
        view = view_polygon(H, width, height, size, area) if H else None
        cut += view is not None and area is not None
        f['view'] = view
        views += view is not None
        observed = {p.get('id') for p in f.get('people', []) if p.get('id')}
        codes = {}
        for s in slots:
            pid = s['id']
            if pid in observed:
                counts['O'] += 1
                continue
            est = estimates.get(pid)
            e = est[i] if est and i < len(est) else None
            code = classify(e[0], e[1], view, size) if e else 'U'
            if code == 'U' and view is not None and s.get('exitEdge') and s.get('lastSeen') is not None and f['time'] > s['lastSeen']:
                code = 'S'   # last seen leaving through a frame edge and not observed since: off-screen, however uncertain the estimate
            codes[pid] = code
            counts[code] += 1
        f['visibility'] = codes
    total = max(1, sum(counts.values()))
    return {'estimates': source, 'framesWithView': views, 'slotFrames': total,
            'shares': {k: round(v/total, 3) for k, v in counts.items()}, **({'viewsWithoutBlackCorners': cut} if cut else {})}
