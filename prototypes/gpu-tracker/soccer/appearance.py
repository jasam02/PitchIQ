"""Appearance evidence for re-identification.

Two complementary parts per crop:
- OSNet embedding (learned, 512-d, from vision.Appearance): the main identity cue.
- Part colour descriptor: jersey (upper torso), shorts, socks soft histograms and a coarse body
  layout. Grass pixels are skipped, people standing in front are masked, and a quality score
  (size, occlusion, truncation, blur, visible body pixels) decides what enters a gallery.

Each identity keeps a gallery of several diverse, high-quality samples; a returning player is
compared against all of them, never against one old frame.
"""
import base64
import math
from dataclasses import dataclass

import cv2
import numpy as np

from .geometry import clamp

GREYS_JERSEY, HUES_JERSEY, LEVELS_JERSEY = (.1, .38, .65, .92), 10, 2   # 4 + 20 = 24 bins
GREYS_SHORTS, HUES_SHORTS = (.1, .45, .85), 9                         # 3 + 9 = 12 bins
GREYS_SOCKS, HUES_SOCKS = (.1, .45, .85), 5                           # 3 + 5 = 8 bins
LAYOUT_COLS, LAYOUT_ROWS = 2, 4
PART_WEIGHTS = {'jersey': .45, 'shorts': .2, 'socks': .1, 'layout': .25}
MIN_GALLERY_QUALITY = .45
GALLERY_SPACING = .8  # seconds between gallery samples
CROP_W, CROP_H = 32, 64
# OSNet cosine -> similarity in [0,1]. Same person typically > .88, same-kit teammates .7-.85.
COS_LOW, COS_HIGH = .6, .95


def cosine_similarity(cos):
    return clamp((cos-COS_LOW)/(COS_HIGH-COS_LOW))


@dataclass
class Descriptor:
    jersey: np.ndarray
    shorts: np.ndarray
    socks: np.ndarray
    layout: np.ndarray
    quality: float
    embedding: np.ndarray | None = None

    @property
    def has_jersey(self):
        return float(self.jersey.sum()) > 0


def empty_descriptor(embedding=None):
    return Descriptor(np.zeros(24), np.zeros(12), np.zeros(8), np.full(LAYOUT_COLS*LAYOUT_ROWS*3, -1.0), 0.0, embedding)


def _soft_hist(rgb, greys, hues, levels):
    """rgb: (N,3) in [0,1]. Soft bins so a small hue/brightness change moves mass between
    neighbouring bins instead of flipping them."""
    n_grey = len(greys)
    hist = np.zeros(n_grey+hues*levels)
    if not len(rgb):
        return hist
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    mx, mn = rgb.max(axis=1), rgb.min(axis=1)
    c = mx-mn
    chroma = np.clip((c-.06)/.12, 0, 1)*np.clip((mx-.1)/.12, 0, 1)
    grey = 1-chroma
    v = (mx+mn)/2
    gs = np.asarray(greys)
    k = (v[:, None] > gs[None, 1:]).sum(axis=1)
    edge = (v <= gs[0]) | (k == n_grey-1)
    kk = np.minimum(k, n_grey-2)
    t = np.clip((v-gs[kk])/(gs[kk+1]-gs[kk]), 0, 1)
    np.add.at(hist, k[edge], grey[edge])
    mid = ~edge
    np.add.at(hist, kk[mid], grey[mid]*(1-t[mid]))
    np.add.at(hist, kk[mid]+1, grey[mid]*t[mid])
    sel = chroma > 0
    if sel.any():
        r, g, b, mx, c, w = r[sel], g[sel], b[sel], mx[sel], np.maximum(c[sel], 1e-9), chroma[sel]
        hue = np.where(mx == r, (g-b)/c, np.where(mx == g, 2+(b-r)/c, 4+(r-g)/c))
        hue = (hue+6) % 6
        p = hue/6*hues-.5
        i0 = np.floor(p).astype(int)
        t = p-i0
        a, nxt = (i0+hues) % hues, (i0+1) % hues
        if levels == 1:
            np.add.at(hist, n_grey+a, w*(1-t))
            np.add.at(hist, n_grey+nxt, w*t)
        else:
            lv = np.clip((mx-.2)/.6, 0, 1)
            np.add.at(hist, n_grey+a*2, w*(1-t)*(1-lv))
            np.add.at(hist, n_grey+a*2+1, w*(1-t)*lv)
            np.add.at(hist, n_grey+nxt*2, w*t*(1-lv))
            np.add.at(hist, n_grey+nxt*2+1, w*t*lv)
    return hist


def grass_reference(frame):
    """Median Lab chroma (a, b) of grass-coloured pixels, or None. Lets a green jersey that differs
    from this particular turf keep its pixels."""
    small = cv2.resize(frame, (160, max(2, round(160*frame.shape[0]/frame.shape[1]))), interpolation=cv2.INTER_AREA)
    s = small.astype(np.int16)
    b, g, r = s[..., 0], s[..., 1], s[..., 2]
    mask = (g > r*1.08) & (g > b*1.15) & (g > 35)
    if mask.sum() < 50:
        return None
    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB).astype(np.float32)
    return float(np.median(lab[..., 1][mask])), float(np.median(lab[..., 2][mask]))


def _grass_mask(crop_bgr, reference):
    s = crop_bgr.astype(np.int16)
    b, g, r = s[..., 0], s[..., 1], s[..., 2]
    mask = (g > r*1.08) & (g > b*1.15) & (g > 35)
    if reference is not None and mask.any():
        lab = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        mask &= (np.abs(lab[..., 1]-reference[0]) < 9) & (np.abs(lab[..., 2]-reference[1]) < 12)
    return mask


def describe(frame, box_px, others_px=(), embedding=None, grass_ref=None):
    """Part descriptor of one person. box_px: [x1, y1, x2, y2] in frame pixels. others_px: other
    person boxes; those whose feet are lower stand in front and are masked out."""
    H, W = frame.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in box_px]
    bw, bh = x2-x1, y2-y1
    if bw <= 1 or bh <= 1 or x2 <= 0 or y2 <= 0 or x1 >= W or y1 >= H:
        return empty_descriptor(embedding)
    cx1, cy1, cx2, cy2 = max(0, int(x1)), max(0, int(y1)), min(W, int(math.ceil(x2))), min(H, int(math.ceil(y2)))
    if cx2-cx1 < 2 or cy2-cy1 < 4:
        return empty_descriptor(embedding)
    raw = frame[cy1:cy2, cx1:cx2]
    # Resample the full (unclipped) box onto a fixed grid; pixels outside the frame are invalid.
    gx = x1+(np.arange(CROP_W)+.5)/CROP_W*bw
    gy = y1+(np.arange(CROP_H)+.5)/CROP_H*bh
    inside = ((gy >= 0) & (gy < H))[:, None] & ((gx >= 0) & (gx < W))[None, :]
    map_x = np.repeat(np.clip(gx-cx1, 0, cx2-cx1-1)[None, :], CROP_H, axis=0).astype(np.float32)
    map_y = np.repeat(np.clip(gy-cy1, 0, cy2-cy1-1)[:, None], CROP_W, axis=1).astype(np.float32)
    interpolation = cv2.INTER_AREA if bw > CROP_W*1.5 else cv2.INTER_LINEAR
    crop = cv2.resize(raw, (CROP_W, CROP_H), interpolation=interpolation) if inside.all() else cv2.remap(raw, map_x, map_y, cv2.INTER_LINEAR)
    visible = inside.copy()
    near = []
    for o in others_px:
        ox1, oy1, ox2, oy2 = [float(v) for v in o[:4]]
        if (ox1, oy1, ox2, oy2) == (x1, y1, x2, y2):
            continue
        overlap = max(0.0, min(x2, ox2)-max(x1, ox1))*max(0.0, min(y2, oy2)-max(y1, oy1))
        if overlap <= 0:
            continue
        near.append((overlap, oy2 > y2+1e-6))
        if oy2 > y2+1e-6:
            hidden = ((gy >= oy1) & (gy < oy2))[:, None] & ((gx >= ox1) & (gx < ox2))[None, :]
            visible &= ~hidden
    rgb = crop[..., ::-1].astype(np.float32)/255
    body = visible & ~_grass_mask(crop, grass_ref)

    def part(u0, u1, v0, v1, greys, hues, levels):
        ys = slice(int(v0*CROP_H), max(int(v0*CROP_H)+1, int(math.ceil(v1*CROP_H))))
        xs = slice(int(u0*CROP_W), max(int(u0*CROP_W)+1, int(math.ceil(u1*CROP_W))))
        seen, used = visible[ys, xs], body[ys, xs]
        n_seen, n_used = int(seen.sum()), int(used.sum())
        hist = _soft_hist(rgb[ys, xs][used], greys, hues, levels)
        if n_used >= max(3, n_seen*.12) and hist.sum() > 0:
            return np.round(hist/hist.sum(), 4), n_used/max(1, n_seen)
        return np.zeros_like(hist), n_used/max(1, n_seen)

    jersey, jersey_frac = part(.25, .75, .18, .45, GREYS_JERSEY, HUES_JERSEY, LEVELS_JERSEY)
    shorts, _ = part(.2, .8, .5, .68, GREYS_SHORTS, HUES_SHORTS, 1)
    socks, _ = part(.12, .88, .75, .92, GREYS_SOCKS, HUES_SOCKS, 1)
    layout = []
    for row in range(LAYOUT_ROWS):
        ys = slice(int((.02+.96*row/LAYOUT_ROWS)*CROP_H), int((.02+.96*(row+1)/LAYOUT_ROWS)*CROP_H))
        for col in range(LAYOUT_COLS):
            xs = slice(int((.1+.8*col/LAYOUT_COLS)*CROP_W), int((.1+.8*(col+1)/LAYOUT_COLS)*CROP_W))
            seen, used = visible[ys, xs], body[ys, xs]
            n = int(used.sum())
            if n >= max(1, math.ceil(seen.sum()*.25)):
                px = rgb[ys, xs][used]
                r, g, b = px[:, 0], px[:, 1], px[:, 2]
                layout += [round(float(((r+g+b)/3).mean()), 3), round(float(((r-g)/2+.5).mean()), 3), round(float(((r+g-2*b)/4+.5).mean()), 3)]
            else:
                layout += [-1, -1, -1]
    body_frac = body.sum()/max(1, visible.sum())
    # Quality: size, occlusion, truncation, visible body pixels, aspect ratio, sharpness.
    vis_h, vis_w = cy2-cy1, cx2-cx1
    size_f = clamp((vis_h-16)/56)**.6
    area = bw*bh
    occ = sum(o/area*(1 if is_front else .5) for o, is_front in near)
    occ_f = clamp(1-1.4*min(1.0, occ))
    e = 1.5
    trunc_f = (.55 if x1 <= e else 1)*(.55 if x2 >= W-e else 1)*(.75 if y1 <= e else 1)*(.5 if y2 >= H-e else 1)
    pix_f = clamp((jersey_frac-.2)/.3)*clamp(body_frac/.25)
    ratio = vis_h/max(1, vis_w)
    aspect_f = clamp((ratio-.6)/.7) if ratio < 1.3 else clamp((9-ratio)/3.5) if ratio > 5.5 else 1.0
    quality = round(clamp(size_f*occ_f*trunc_f*pix_f*aspect_f*_sharpness(raw)), 3)
    return Descriptor(jersey, shorts, socks, np.asarray(layout, np.float64), quality, embedding)


def _sharpness(crop):
    """~1 for crisp edges, ~.75 for blurred ramps (largest 1-px step vs the 3-px step)."""
    h, w = crop.shape[:2]
    if h < 8 or w < 8:
        return .9
    lum = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32)[int(h*.1):int(h*.9), int(w*.2):int(w*.8)]
    if lum.shape[0] < 5 or lum.shape[1] < 5:
        return .9
    fine = coarse = 0.0
    for img in (lum, lum.T):
        d1 = np.abs(np.diff(img, axis=1))
        fine += np.maximum(np.maximum(d1[:, :-2], d1[:, 1:-1]), d1[:, 2:]).sum()
        coarse += np.abs(img[:, 3:]-img[:, :-3]).sum()
    points = lum[:, 3:].size+lum.T[:, 3:].size
    if coarse < points*2:
        return .9
    return .75+.25*clamp((fine/coarse-.5)/.45)


# ---------- distances ----------
def bhattacharyya(a, b):
    """Hellinger distance of two histograms: 0 identical .. 1 disjoint; None when either is empty."""
    a, b = np.maximum(np.asarray(a, np.float64), 0), np.maximum(np.asarray(b, np.float64), 0)
    sa, sb = a.sum(), b.sum()
    if sa <= 1e-9 or sb <= 1e-9:
        return None
    bc = float(np.sqrt(a*b).sum()/math.sqrt(sa*sb))
    return math.sqrt(clamp(1-bc))


def layout_distance(a, b):
    a, b = np.asarray(a).reshape(-1, 3), np.asarray(b).reshape(-1, 3)
    ok = (a[:, 0] >= 0) & (b[:, 0] >= 0)
    if not ok.any():
        return None
    return float(np.minimum(1, np.linalg.norm(a[ok]-b[ok], axis=1)/.5).mean())


def descriptor_distance(a, b):
    parts = {'jersey': bhattacharyya(a.jersey, b.jersey), 'shorts': bhattacharyya(a.shorts, b.shorts),
             'socks': bhattacharyya(a.socks, b.socks), 'layout': layout_distance(a.layout, b.layout)}
    total = weight = 0.0
    for key, w in PART_WEIGHTS.items():
        if parts[key] is not None:
            total += w*parts[key]
            weight += w
    out = {k: (.5 if v is None else round(v, 3)) for k, v in parts.items()}
    out['total'] = round(total/weight, 3) if weight else .5
    return out


def embedding_cosine(a, b):
    if a is None or b is None:
        return None
    return float(np.dot(a, b))


@dataclass
class Sample:
    d: Descriptor
    time: float


def _diversity(a, b):
    cos = embedding_cosine(a.d.embedding, b.d.embedding)
    if cos is not None:
        return clamp(1-cos)
    return descriptor_distance(a.d, b.d)['total']


def add_to_gallery(gallery, d, time, max_size=6, trusted=False):
    """Quality- and diversity-aware update. Low-quality samples are ignored; samples stay >= .8 s
    apart (a much better one may replace a close one); when full, the sample with the lowest
    quality + diversity value is dropped, except the oldest one: it records who the identity was
    when it was created, which tells a track that switched people from the person it always was."""
    g = list(gallery)
    quality = max(d.quality, MIN_GALLERY_QUALITY) if trusted and d.quality >= .2 else d.quality
    if quality >= MIN_GALLERY_QUALITY and math.isfinite(time):
        d = Descriptor(d.jersey, d.shorts, d.socks, d.layout, quality, d.embedding)
        close = [s for s in g if abs(s.time-time) < GALLERY_SPACING]
        if len(close) == 1 and d.quality >= close[0].d.quality+.1:
            g = [s for s in g if s is not close[0]]+[Sample(d, time)]
        elif not close:
            g.append(Sample(d, time))
    while len(g) > max(0, max_size):
        values = []
        for i, s in enumerate(g):
            m = min((_diversity(s, o) for j, o in enumerate(g) if j != i), default=1.0)
            values.append(s.d.quality+1.5*min(m, .3))
        oldest = min(range(len(g)), key=lambda i: g[i].time)
        worst = min((i for i in range(len(g)) if i != oldest), key=lambda i: (values[i], g[i].time))
        g.pop(worst)
    return sorted(g, key=lambda s: s.time)


def compare(gallery, probes):
    """Robust comparison of several probe descriptors with several gallery samples: the mean of the
    best two pair scores. Returns appearance (OSNet similarity, None without embeddings) and colour
    part distances; None when nothing can be compared."""
    good = [p for p in probes if p.quality >= .25]
    use = good or list(probes)
    if not gallery or not use:
        return None
    pairs = [descriptor_distance(s.d, p) for s in gallery for p in use]
    best = lambda key: round(float(np.mean(sorted(x[key] for x in pairs)[:2])), 3)
    out = {key: best(key) for key in ('total', 'jersey', 'shorts', 'socks', 'layout')}
    cosines = [c for c in (embedding_cosine(s.d.embedding, p.embedding) for s in gallery for p in use) if c is not None]
    if cosines:
        top = sorted(cosines, reverse=True)[:2]
        out['cosine'] = round(float(np.mean(top)), 4)
        out['appearance'] = round(cosine_similarity(out['cosine']), 3)
    else:
        out['appearance'] = None
    return out


# ---------- batch comparison: galleries of several people against several probe crops ----------
def _stack(descs, key, floor=None):
    rows = [np.asarray(getattr(d, key), np.float64) for d in descs]
    return np.stack([r if floor is None else np.maximum(r, floor) for r in rows])


def _batch_hellinger(A, B):
    """(N, K) x (M, K) histograms -> (N, M) Hellinger distances; NaN where either histogram is empty."""
    sa, sb = A.sum(axis=1), B.sum(axis=1)
    bc = (np.sqrt(A) @ np.sqrt(B).T)/np.sqrt(np.maximum(np.outer(sa, sb), 1e-18))
    d = np.sqrt(np.clip(1-bc, 0, 1))
    d[(sa <= 1e-9)[:, None] | (sb <= 1e-9)[None, :]] = np.nan
    return d


def _batch_layout(A, B):
    """(N, 24) x (M, 24) body layouts -> (N, M) layout distances over the cells both know; NaN with none."""
    a, b = A.reshape(len(A), -1, 3), B.reshape(len(B), -1, 3)
    ok = (a[:, None, :, 0] >= 0) & (b[None, :, :, 0] >= 0)
    diff = np.minimum(1, np.linalg.norm(a[:, None]-b[None, :], axis=3)/.5)
    count = ok.sum(axis=2)
    return np.where(count > 0, (diff*ok).sum(axis=2)/np.maximum(count, 1), np.nan)


def batch_similarity(samples, probes):
    """(N, M) whole-appearance similarity of N gallery descriptors to M probe descriptors: the learned
    appearance (OSNet) blended with the whole-uniform colour distance (jersey, shorts, socks and body
    layout, weighted as in descriptor_distance); colour alone where an embedding is missing. Used to ask
    whether a person looks more like the known referees or like one team's players."""
    if not samples or not probes:
        return np.zeros((len(samples), len(probes)))
    parts, weights = [], []
    for key, w in PART_WEIGHTS.items():
        if key == 'layout':
            parts.append(_batch_layout(_stack(samples, key), _stack(probes, key)))
        else:
            parts.append(_batch_hellinger(_stack(samples, key, 0), _stack(probes, key, 0)))
        weights.append(w)
    D = np.stack(parts)
    W = np.asarray(weights)[:, None, None]*~np.isnan(D)
    known = W.sum(axis=0)
    total = np.where(known > 0, (np.nan_to_num(D)*W).sum(axis=0)/np.maximum(known, 1e-9), .5)
    colour = np.clip(1-total/.6, 0, 1)
    ea, eb = [d.embedding for d in samples], [d.embedding for d in probes]
    size = next((len(e) for e in ea+eb if e is not None), 0)
    usable = lambda e: e is not None and len(e) == size
    if not size or not any(usable(e) for e in ea) or not any(usable(e) for e in eb):
        return colour
    E = lambda es: np.stack([np.asarray(e, np.float64) if usable(e) else np.zeros(size) for e in es])
    both = np.outer([usable(e) for e in ea], [usable(e) for e in eb])
    appearance = np.clip((E(ea) @ E(eb).T-COS_LOW)/(COS_HIGH-COS_LOW), 0, 1)
    return np.where(both, .7*appearance+.3*colour, colour)


# ---------- persistence ----------
def encode_descriptor(d):
    out = {'q': round(float(d.quality), 3), 'j': [round(float(v), 4) for v in d.jersey],
           's': [round(float(v), 4) for v in d.shorts], 'k': [round(float(v), 4) for v in d.socks],
           'l': [round(float(v), 3) for v in d.layout]}
    if d.embedding is not None:
        out['e'] = base64.b64encode(np.asarray(d.embedding, np.float16).tobytes()).decode('ascii')
    return out


def decode_descriptor(v):
    try:
        embedding = None
        if v.get('e'):
            embedding = np.frombuffer(base64.b64decode(v['e']), np.float16).astype(np.float32)
            norm = np.linalg.norm(embedding)
            embedding = embedding/norm if norm > 0 else None
        d = Descriptor(np.asarray(v['j'], np.float64), np.asarray(v['s'], np.float64), np.asarray(v['k'], np.float64),
                       np.asarray(v['l'], np.float64), clamp(float(v['q'])), embedding)
        if len(d.jersey) != 24 or len(d.shorts) != 12 or len(d.socks) != 8 or len(d.layout) != LAYOUT_COLS*LAYOUT_ROWS*3:
            return None
        return d
    except (KeyError, TypeError, ValueError):
        return None
