"""Is a person detection part of the match?

The test uses the foot point (bottom-centre of the box), never box overlap with
grass: a spectator's box can overlap the pitch while their feet are in the
stands. Zones: inside / boundary (small touchline tolerance) / outside /
unknown (no reliable pitch). Outside detections are rejected with a reason;
a perspective size model rejects implausibly large or small people.
Precision over recall: briefly missing a player on the touchline is better
than tracking the crowd. One exception: a lone person just outside the near
touchline whom the detector confidently calls a referee (an assistant referee)
is kept in the boundary zone; a track that stays there can only ever become an
official, never a player.

Three things are relaxed, each behind a scale condition that broadcast footage
never meets: the expected size of far-away people comes from the heights
actually seen at that image row when the linear perspective model is clearly
off there (wide lenses compress the far field), people expected smaller than
SMALL_PX get a wider size tolerance, and a small detection that continues a
tracked person is kept even when its box is cut short. Feet just beyond a
LOW-confidence boundary are held at the edge rather than rejected (the
boundary itself is uncertain then).
"""
import math

import numpy as np

from .geometry import foot_point, iou, point_in_polygon, signed_distance
from .pitch import nearest_boundary, zone_in_polygon, zone_of

MIN_PERSON_SCORE = .10
SIZE_FIT_SCORE = .35
SIZE_MIN_SAMPLES, SIZE_MAX_RATIO, SIZE_MIN_RATIO, LYING_MIN_RATIO = 6, 2.2, .4, .6
SMALL_PX = 36            # people expected smaller than this (pixels) get the wide size tolerance
SMALL_MIN_RATIO = .28    # ... a cut box of a 30 px person can be a third of the expected height
TRACKED_IOU = .4         # a small size-rejected box this close to a tracked person continues that person
PROVISIONAL_MARGIN = .05 # frame heights beyond a LOW-confidence boundary within which feet still count as 'boundary'
OFFICIAL_SCORE = .5      # detector 'referee' score for an assistant referee outside the touchline
OFFICIAL_REACH = .8      # how far beyond the near touchline, in body heights
ZONE_RANK = {'inside': 0, 'unknown': 1, 'boundary': 2, 'outside': 3}

REJECT_TEXT = {'outside-pitch': 'REJECTED: OUTSIDE PITCH', 'audience': 'REJECTED: AUDIENCE',
               'implausible-size': 'REJECTED: SIZE', 'low-confidence': 'REJECTED: LOW PLAYER CONFIDENCE',
               'unconfirmed-far': 'REJECTED: FAR DETECTION NOT CONFIRMED', 'unconfirmed-crop': 'REJECTED: CROP DETECTION AWAY FROM THE PREDICTION'}
FAR_CONTINUE = .05       # frame heights beyond an uncertain (not painted) far edge within which feet continuing a tracked person count as 'boundary'


class SizeModel:
    """Expected box height = a + b * footY (people lower in the image are closer to the camera).
    `bands` are the median heights actually seen per image row band; they replace the line where the
    line is clearly off (a wide-angle lens compresses the far field)."""

    def __init__(self, a=0.0, b=0.0, reliable=False, bands=()):
        self.a, self.b, self.reliable = a, b, reliable
        self.bands = tuple(bands)

    def linear(self, foot_y):
        return self.a+self.b*foot_y

    def expected(self, foot_y, local=True):
        line = self.linear(foot_y)
        if not local or len(self.bands) < 2:
            return line
        ys = [y for y, _ in self.bands]
        if not ys[0]-.05 <= foot_y <= ys[-1]+.05:
            return line
        measured = float(np.interp(foot_y, ys, [h for _, h in self.bands]))
        return measured if line <= 0 or abs(measured-line)/line > .25 else line

    def moved(self, scale, dy):
        """Same model after a zoom (scale) and vertical pan (dy) in normalized units."""
        return SizeModel(self.a*scale-self.b*dy, self.b, self.reliable, [(y+dy, h*scale) for y, h in self.bands])


def fit_size_model(boxes):
    pts = [(b[1]+b[3], b[3]) for b in boxes if b[3] > 0]
    if not pts:
        return SizeModel()
    slopes = [(pts[j][1]-pts[i][1])/(pts[j][0]-pts[i][0]) for i in range(len(pts)) for j in range(i+1, len(pts))
              if abs(pts[j][0]-pts[i][0]) >= .03]
    b = max(0.0, float(np.median(slopes))) if slopes else 0.0
    a = float(np.median([h-b*y for y, h in pts]))
    ys = [y for y, _ in pts]
    reliable = len(pts) >= SIZE_MIN_SAMPLES and max(ys)-min(ys) >= .06 and a+b*min(ys) > 0
    bands = []
    if reliable:
        for k in range(10):
            rows = [h for y, h in pts if k/10 <= y < (k+1)/10]
            if len(rows) >= 3:
                bands.append(((k+.5)/10, float(np.median(rows))))
    return SizeModel(a, b, reliable, bands)


def combine_zone(auto, user):
    """The stricter of the automatic pitch zone and a user-drawn boundary zone."""
    if user is None:
        return auto
    if auto[0] == 'unknown':
        return user
    return auto if ZONE_RANK[auto[0]] >= ZONE_RANK[user[0]] else user


def assess(detections, pitch, config, size=None, user_polygon=None, live=(), height_px=1080):
    """Split person detections into match-relevant and rejected ones.

    detections: [{'box': [x,y,w,h] normalized, 'score', 'cls', ...}]; live: boxes of people tracked
    on the previous frames (robust mode keeps small detections that continue them); height_px: frame
    height, for the pixel size of people (SMALL_PX); the tracker passes its 1080p equivalent for a
    low-resolution source (tracker.pixel_scale), whose people the detector sees magnified.
    Returns (accepted, continuation, rejected, size_model). Each accepted/continuation item is the
    detection dict extended with 'foot', 'zone', 'outsideBy', 'sizeRatio'. Continuation items are
    just beyond a touchline (not audience): they may extend an existing track but never start one.
    """
    robust = getattr(pitch, 'mode', 'automatic') != 'automatic'
    provisional = robust and getattr(pitch, 'level', 'HIGH') == 'LOW'
    scored = []
    for d in detections:
        zone = zone_of(pitch, d['box'], config)
        if user_polygon is not None and config.enabled:
            zone = combine_zone(zone, zone_in_polygon(user_polygon, pitch.aspect, d['box'], config))
        if provisional and zone[0] == 'outside' and zone[1] <= PROVISIONAL_MARGIN:
            # The boundary itself is uncertain: feet just beyond it are not rejected, only held at the edge.
            zone = ('boundary', zone[1])
        continued = None
        if zone[0] == 'outside' and zone[1] <= FAR_CONTINUE and live and pitch.reliable:
            # Feet a little beyond a far edge that is not a painted line (a predicted or grass edge), continuing a
            # tracked person: the person, not the audience. A painted far touchline keeps its full meaning.
            edge = nearest_boundary(pitch, foot_point(d['box']))
            far_sources = pitch.sources() if hasattr(pitch, 'sources') else {}
            if edge is not None and edge['side'] == 'far' and far_sources.get('far') != 'line' and any(iou(d['box'], b) > .3 for b in live):
                zone, continued = ('boundary', zone[1]), 'far'
        scored.append(dict(d, foot=foot_point(d['box']), zone=zone[0], outsideBy=zone[1], sizeRatio=1.0, **({'continued': continued} if continued else {})))
    filtering = config.enabled and (pitch.reliable or user_polygon is not None)
    model = size or SizeModel()
    if filtering:
        fit = fit_size_model([d['box'] for d in scored if d['zone'] == 'inside' and d['score'] >= SIZE_FIT_SCORE])
        if fit.reliable:
            model = fit
    use_size = filtering and model.reliable
    if use_size:
        for d in scored:
            e = model.expected(d['foot'][1])
            if e > .004:
                ratio = d['box'][3]/e
                long_axis = max(d['box'][3], d['box'][2]*pitch.aspect)
                # A player lying after a tackle or a diving keeper: measure the long body axis.
                d['sizeRatio'] = long_axis/e if ratio < SIZE_MIN_RATIO and long_axis/e >= LYING_MIN_RATIO else ratio
                d['expected'] = e
    outside = [d for d in scored if d['zone'] == 'outside']
    accepted, continuation, rejected = [], [], []
    for d in scored:
        if d['zone'] == 'outside':
            audience, detail = _audience(d, pitch, outside)
            if not audience and _assistant_referee(d, pitch, outside):
                d['zone'], d['official'] = 'boundary', True
            else:
                reason = 'audience' if audience else 'outside-pitch'
                rejected.append({'box': d['box'], 'score': d['score'], 'reason': reason, 'detail': detail})
                if not audience:
                    continuation.append(d)
                continue
        if use_size:
            e = d.get('expected', model.expected(d['foot'][1]))
            small = e*height_px < SMALL_PX
            low = SMALL_MIN_RATIO if small else SIZE_MIN_RATIO
            if not low <= d['sizeRatio'] <= SIZE_MAX_RATIO:
                tracked = d['box'][3]*height_px < SMALL_PX and any(iou(d['box'], b) > TRACKED_IOU for b in live)
                if not tracked:
                    rejected.append({'box': d['box'], 'score': d['score'], 'reason': 'implausible-size',
                                     'detail': f"height {d['box'][3]:.3f} is {d['sizeRatio']:.2f}x the expected {e:.3f} at foot y {d['foot'][1]:.2f}"})
                    continue
                d['kept'] = 'continues a tracked person'
        if d['score'] < MIN_PERSON_SCORE:
            rejected.append({'box': d['box'], 'score': d['score'], 'reason': 'low-confidence',
                             'detail': f"score {d['score']:.2f} below {MIN_PERSON_SCORE}"})
            continue
        accepted.append(d)
    return accepted, continuation, rejected, model


def _assistant_referee(d, pitch, outside):
    """Alone, just beyond the near touchline, and a confident detector 'referee'."""
    if d.get('cls') != 'referee' or d['score'] < OFFICIAL_SCORE or d['outsideBy'] > OFFICIAL_REACH*d['box'][3]:
        return False
    edge = nearest_boundary(pitch, d['foot']) if pitch.reliable else None
    if edge is None or edge['side'] != 'near':
        return False
    radius = 2*d['box'][3]
    return not any(o is not d and math.hypot((o['foot'][0]-d['foot'][0])*pitch.aspect, o['foot'][1]-d['foot'][1]) < radius for o in outside)


def _audience(d, pitch, outside):
    """Spectators: beyond the far boundary, off the grass in the upper half (stands), or in a dense
    group of outside detections. Everyone else outside is 'outside-pitch' (bench, technical area,
    behind boards, beyond a goal line)."""
    edge = nearest_boundary(pitch, d['foot']) if pitch.reliable else None
    base = f"foot {d['outsideBy']:.3f} beyond {edge['label'] if edge else 'the field boundary'}"
    if edge and edge['side'] == 'far':
        return True, f'{base} (far side)'
    grass = pitch.grass_polygon
    if len(grass) >= 3 and not point_in_polygon(d['foot'], grass) and signed_distance(d['foot'], grass, pitch.aspect) > .01:
        if pitch.polygon and d['foot'][1] < (min(p[1] for p in pitch.polygon)+max(p[1] for p in pitch.polygon))/2:
            return True, f'{base} (stands)'
    radius = max(.08, 2*d['box'][3])
    near = sum(1 for o in outside if o is not d and math.hypot((o['foot'][0]-d['foot'][0])*pitch.aspect, o['foot'][1]-d['foot'][1]) < radius)
    if near >= 3 and not (edge and edge['side'] == 'near' and d['outsideBy'] < .15):
        return True, f'{base} (crowd of {near+1})'
    return False, base
