"""Coaching diagrams and evidence overlays as vector drawing operations, for the PDF and for PNG previews.

Why one list of drawing operations. A coaching figure appears in the report tab (SVG, coaching.js), in the PDF
(vector operators, soccer/coaching/pdf.py) and in the PNG previews the tests and the self-check look at. evidence.py
describes each figure once: a DiagramSpec in normalized pitch coordinates (x 0..1 from the left goal line, y 0..1
from the far touchline, the convention of tactical.js and soccer/analytics.py) or a VideoOverlaySpec in source-image
pixels. This module turns a spec into a short list of primitive operations (rect, circle, path, text, clip; points,
origin top left, y down) that pdf.py writes as PDF operators and ops_to_png() rasterizes with cv2. No PDF renderer
exists in this environment (the server audit's headless viewers returned blank pages for every option), so the PNG
of the same list is how the drawing is checked.

What is drawn as a fact and what is not:
- Players are the spec's own elements at their tracked positions. An element whose kind is not O (observed) is drawn
  hollow and dashed, the 2D view's way of showing an estimate: an estimate is never drawn as an observation.
- A v1 suggested position (render-only for old stored reports; the v2 engine never emits one) is dashed and always
  labelled COACHING ILLUSTRATION, with a dashed link from the player's real position: it is advice, not data.
- A movement arrow never bridges more than ARROW_GAP seconds between its samples (contract section 6): the arrow is
  cut there instead.
- Elements with coordinates outside the drawable area (beyond OUTSIDE of the pitch, tactical.js RULES.outside) or of
  an unknown type are skipped and listed in 'skipped', never clamped onto the pitch.
- Metre values appear only as the spec's own text; the legend says they are estimated when the pitch size is assumed
  (every run so far: 105 x 68 m, source 'standard').
- A team width or length is a bracket beside the team, not a line through it: evidence.py's measure with bracket
  True (placed by it BRACKET_OFFSET outside the team, with the two end ids) is drawn where it stands, its ticks and
  dashed leaders pointing back at the two players that set the extent and its value outside it; a 'bracket' element
  without a placement keeps BRACKET_GAP clear of the analysed team's markers and labels and prefers the side where it
  covers fewer opponents. On the BC180 and BC60 width_depth evidence (15 diagrams, both teams) every bracket stands
  outside the team with its value inside the drawing, where the first measure line crossed the team and put its
  value among the players' labels.

Video overlays are limited by the frame's tier (audit/calibration, measured against hand-marked truth):
  world  (T3: registration confidence >= .7 and the painted lines checked within +-0.5 s): broadcast median error
         4.1-5.5 px, worst landmark p90 9.5-11.6 px, 0% over 20 px. Anything in pitch geometry may be drawn.
  pinned (T2: confidence >= .35): shapes whose corners are the current frame's foot pixels (box bottom centre) carry
         no registration error at the feet; extensions within 10 m are 3-4 px off on broadcast, 1.5 px on amateur,
         4.7 px on Veo. Lines between players, unit polylines, hulls and measures; no suggested positions.
  image  (T1: below .35, or not calibrated): boxes, labels and arrows between box feet only (box foot 2-3 px, 15-32
         px on flagged box problems). No metre values on the video.
The element list of each tier is enforced here, not only in evidence.py: an element a tier does not allow is skipped
and reported, so a spec mistake cannot put pitch geometry on a frame whose registration does not support it (Veo
frames rated MEDIUM are 50-60 px off; pitch geometry over them would show a wrong picture). An overlay's caption
says it is an analysis overlay; nothing is ever drawn into the frame's pixels (pdf.py places it over the JPEG).

Coaching v2 annotations (DESIGN 5.4-5.5; schema.py is the contract, coaching.js draws the same records on the page).
- The new elements: focus (the reviewed player: a ring in the review colour), support (the people the annotation
  relies on: a dashed ring), relation (a line, dashed when the relation was missing, its value placed where it hides
  the fewest markers), channel (the area the check is about: the review colour at 12%), suggestedZone (a dotted outline
  of round dots over a halo, a 45-degree hatch at 45% and a 10% fill; a 'point' is a small dashed circle, never a dot),
  moveArrow (dashed, in the zone colour) and caption (in a strip above the pitch, never over it, as on the page).
  At most schema.ELEMENT_BUDGET of each is drawn; the rest are skipped and reported. In a v2 diagram the people the
  annotation is not about are faint and unlabelled, and nobody of the annotation is labelled where evidence.py set
  focus.label False (levels that name no player, an unconfirmed identity).
- Overlays follow schema.TIER_ELEMENTS: a suggested area, a channel or a movement arrow only on a world-tier frame
  (pinned and image frames show them on the diagram), no relation line at image tier, where the focus is the box
  outlined. A projected area may reach past the frame edge (BC180 B's far-side channel did): the clip cuts it.
- Colours are chosen per pair of team colours (annotation_palette(), the page's rule, test vectors shared with
  coaching.js). The light print pitch needs darker twins of the zone hues: the page's mint, lime and aqua have WCAG
  contrast 1.00-1.14 against the print stripes, the twins 3.31-3.59 (see PRINT_ZONE); screenshots keep the page's
  colours, since they are the video.
- A dash shorter than its line is a round dot in the PDF (round caps); ops_to_png draws it as one, so the preview
  shows the dotted area as the viewer does.
"""
import math
import re

import cv2
import numpy as np

from . import pdf as _pdf   # base-14 Helvetica widths, shared with the PDF writer so both backends agree
from . import schema as S   # the v2 annotation contract: element tiers and budget (stdlib only)

# Pitch markings in metres (tactical.js PITCH); on a measured non-standard pitch they stay in metres from the goal
# lines and the centre, as the laws fix them.
PITCH = {'length': 105.0, 'width': 68.0, 'penaltyDepth': 16.5, 'penaltyHalf': 20.16, 'goalDepth': 5.5,
         'goalHalf': 9.16, 'spot': 11.0, 'circle': 9.15, 'goalWidth': 7.32, 'goalNet': 2.2, 'cornerArc': 1.0}
MARGIN = 4.0                # m of surround drawn around the pitch (the goal nets are 2.2 m deep)
OUTSIDE = 0.1               # normalized: a position this far beyond a line is still drawn (tactical.js RULES.outside)
ARROW_GAP = 1.0             # s: consecutive arrow samples further apart are not joined (contract section 6)
DIAGRAM_WIDTH = 243.0       # pt: one of two columns of the report's A4 content width
PLAYER_RADIUS = 1.1         # m: marker radius on the diagram, kept between MIN_RADIUS and MAX_RADIUS points
MIN_RADIUS, MAX_RADIUS = 2.4, 6.0
LEGEND_HEIGHT = 10.0        # pt: one legend line under the pitch (the legend wraps onto more lines)
ILLUSTRATION = 'COACHING ILLUSTRATION'
OVERLAY_CAPTION = 'analysis overlay'

# Print-friendly colours (the report's light theme; the Studio's dark tokens are for the screen).
INK = '#18211d'
INK_2 = '#4a5a51'
EDGE = '#1f2a24'
BRAND = '#19b96b'
BRAND_DARK = '#117b49'
WHITE = '#ffffff'
SURROUND = '#f1f7f2'
PITCH_FILL = '#e2f0e5'
PITCH_STRIPE = '#d9eadc'
PITCH_LINE = '#86b293'
NEUTRAL = '#8d99a6'                                 # tactical.js NEUTRAL: a person without a known team
COLOURS = {'A': '#b98cff', 'B': '#f6b564'}          # tactical.js DEFAULT_COLOURS (the viewer may choose others)
ZONE_STYLES = {'gap': ('#d9a961', .30), 'space': ('#19b96b', .22)}    # fill colour, alpha
UNIT_NAMES = {'D': 'defence', 'M1': 'first midfield line', 'M': 'midfield', 'F': 'forwards'}

DIAGRAM_ELEMENTS = ('player', 'unit', 'measure', 'bracket', 'zone', 'band', 'arrow', 'suggested', 'label', 'focus',
                    'support', 'relation', 'channel', 'suggestedZone', 'moveArrow', 'caption')
# Drawing order: areas under lines, lines under markers, text last; the v2 types slot in as on the page (coaching.js
# ORDER), so a suggested area never hides the players and relation lines run under the markers they join.
LAYERS = ('band', 'zone', 'channel', 'szone', 'unit', 'arrow', 'measure', 'relation', 'marrow', 'link', 'player',
          'support', 'ring', 'focus', 'suggested', 'label')
_IMAGE = frozenset(('player', 'box', 'label', 'arrow'))
TIER_ELEMENTS = {'image': _IMAGE,
                 'pinned': _IMAGE | {'unit', 'measure', 'bracket', 'zone'},
                 'world': _IMAGE | {'unit', 'measure', 'bracket', 'zone', 'suggested'}}
# A team length/width bracket ({'type': 'bracket', 'a', 'b', 'side', 'text'}): a and b are the two ends of the
# measured extent (the outermost players' positions, or the bracket's own ends); it is drawn parallel to a-b, just
# outside the analysed team on the given side, with end ticks pointing back at the team, dashed leaders to the two
# ends and its value beside it. The first diagrams drew team width as a measure line through the middle of the team,
# its label among the players (BC180 A width_depth diagrams), which is hard to read on a 243 pt column.
BRACKET_GAP = 3.0           # m between the outermost player of the team and the bracket
BRACKET_TICK = 1.6          # m: the end ticks
BRACKET_SIDES = {'across': ('left', 'right'), 'along': ('top', 'bottom')}   # width runs across, length along
TIER_NOTES = {'world': 'Overlay drawn in pitch coordinates; the registration was checked against the pitch markings.',
              'pinned': "Overlay lines join the detected players' feet; pitch geometry is approximate.",
              'image': 'Overlay marks detected players only; no pitch geometry is drawn on the video.'}
WORLD_LINES_NOTE = ("Overlay lines join the detected players' feet; the frame's registration was checked against the "
                    'pitch markings.')
PLAYERS_NOTE = 'Overlay marks the detected players only.'

# ---------------------------------------------------------------------------------------------------------------
# Coaching v2 annotations (DESIGN 5.4-5.5; schema.py is the contract, coaching.js draws the same records on the page)

V2_TYPES = ('focus', 'support', 'relation', 'channel', 'suggestedZone', 'moveArrow', 'caption')
V2_TIERS = S.TIER_ELEMENTS              # where each new element may be drawn ('diagram' = the top-down drawing)
V2_BUDGET = S.ELEMENT_BUDGET            # at most this many of each new element per drawing (DESIGN 5.5)
# The tier sentences of the page (coaching.js TIER_UI, ASCII): what a v2 overlay of each tier may show.
V2_TIER_NOTES = {
    'world': ("Pitch-accurate: drawn through this frame's pitch registration (paint-checked); areas and lines are "
              'placed on the pitch.'),
    'pinned': ("From the players' feet: drawn from the players' feet on this frame; pitch areas are shown on the 2D "
               'diagram only.'),
    'image': ('Boxes only: the registration is not reliable on this frame, so only the players are marked on the '
              'video; areas are on the 2D diagram.')}
COACH_AREA = "coach's area"   # a coach-adjusted suggested area (DESIGN 10): beside PitchIQ's, thin and solid

# Annotation colours are chosen per pair of team colours, never fixed (DESIGN 5.5): the spec asks for a red/orange
# outline and a green dotted area that stay distinguishable from the team colours, and Team B's default orange
# #f6b564 is "orange" while green is the pitch. PALETTE and annotation_palette() are coaching.js's PALETTE and
# annotationPalette() (FRONTEND's rule, INTERFACES.md): the review outline is the candidate with the largest smallest
# CIEDE2000 distance (Sharma, Wu and Dalal 2005) to both team colours and the grass, white with a dashed dark halo
# when none reaches minDE; the zone colour is the candidate with the best WCAG contrast against both grass stripes
# among those >= minContrast, >= minDE from both teams and >= ballMinDE from the video's ball marker (lime is the
# ball's own colour, 1.2 apart), else the one farthest from the teams over a heavier halo. Measured with the default
# teams (#b98cff, #f6b564): review #ff4d6d (32.1 from team A), zone #7dffb0 (contrast 3.07 on #30944d); the screen's
# aqua fails there (2.85). The PDF draws screenshots with exactly these colours (the board greens stand in for grass,
# as on the page's video).
PALETTE = {'review': ('#ff4d6d', '#ff2fd1', '#ffffff'),      # rose red, magenta, white
           'zone': ('#7dffb0', '#c6ff4d', '#5cf2ff'),        # mint, lime, aqua
           'pitch': ('#2b8a46', '#30944d'),                  # the board's two grass stripes (tactical.js)
           'ball': '#c6ff3d',                                # index.html COLORS.BALL
           'minDE': 20.0, 'ballMinDE': 10.0, 'minContrast': 3.0,   # [P] PALETTE_MIN_DE; ball; WCAG 2 non-text
           'relation': '#ffffff', 'relationOpacity': .45,    # relation lines: white at 45%, dashed when missing
           'halo': '#07131b', 'haloOpacity': .85}            # coaching.js '#07131bd9'
# The PDF's top-down diagrams stand on a light print pitch (#e2f0e5 / #d9eadc), where the page's colours fail the
# same rule: measured, every zone candidate has contrast 1.00-1.14 against the print stripes (the rule needs 3) and
# white is 9.4-11.5 CIEDE2000 from them (invisible on paper). print_palette() therefore keeps the page's choice and
# prints it legibly: the same review colour (white becomes the report ink #18211d), and the darkened twin of the
# page's zone hue, each scaled in sRGB until its contrast reaches 3.3 against both print stripes (mint #448a5f
# 3.53 / 3.32, lime #678528 3.59 / 3.38, aqua #34888f 3.52 / 3.31; 32-70 from the default teams). The same rule
# (contrast >= 3 on the print stripes, >= minDE from the teams) is then checked on the twin; when it fails, the other
# passing twins are tried, best contrast first, else the farthest from the teams over a heavier halo.
PRINT_ZONE = {'#7dffb0': '#448a5f', '#c6ff4d': '#678528', '#5cf2ff': '#34888f'}
PRINT_REVIEW_FOR_WHITE = INK
PRINT_STYLE = {'halo': WHITE, 'haloOpacity': 1.0, 'relation': INK, 'relationOpacity': .6, 'support': INK_2,
               'supportOpacity': .85}
VIDEO_STYLE = {'halo': PALETTE['halo'], 'haloOpacity': PALETTE['haloOpacity'], 'relation': PALETTE['relation'],
               'relationOpacity': PALETTE['relationOpacity'], 'support': WHITE, 'supportOpacity': .8}


def _rgb255(colour):
    m = re.fullmatch(r'#?([0-9a-fA-F]{3}|[0-9a-fA-F]{6})', str(colour or '').strip())
    if not m:
        return None
    h = m.group(1) if len(m.group(1)) == 6 else ''.join(c+c for c in m.group(1))
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))


def _linear(v):
    v /= 255.0
    return v/12.92 if v <= .04045 else ((v+.055)/1.055)**2.4


def lab_of(colour):
    """sRGB (D65) '#rrggbb' -> CIE L*a*b* (coaching.js labOf); None for a malformed colour."""
    rgb = _rgb255(colour)
    if rgb is None:
        return None
    r, g, b = (_linear(v) for v in rgb)
    x = (.4124564*r+.3575761*g+.1804375*b)/.95047
    y = .2126729*r+.7151522*g+.072175*b
    z = (.0193339*r+.119192*g+.9503041*b)/1.08883

    def f(t):
        return t**(1.0/3.0) if t > 216.0/24389.0 else (24389.0/27.0*t+16.0)/116.0
    return [116.0*f(y)-16.0, 500.0*(f(x)-f(y)), 200.0*(f(y)-f(z))]


def ciede2000(lab1, lab2):
    """CIEDE2000 colour difference of two L*a*b* colours (kL = kC = kH = 1), as Sharma, Wu and Dalal (2005) give it
    (their 14 test pairs are reproduced to 4 decimals in test_coaching_render.py)."""
    L1, a1, b1 = lab1
    L2, a2, b2 = lab2
    rad, p7 = math.radians, 25.0**7
    C1, C2 = math.hypot(a1, b1), math.hypot(a2, b2)
    cb7 = ((C1+C2)/2)**7
    G = .5*(1-math.sqrt(cb7/(cb7+p7)))
    a1p, a2p = (1+G)*a1, (1+G)*a2
    C1p, C2p = math.hypot(a1p, b1), math.hypot(a2p, b2)

    def hue(b, a):
        if b == 0 and a == 0:
            return 0.0
        h = math.degrees(math.atan2(b, a))
        return h+360 if h < 0 else h
    h1, h2 = hue(b1, a1p), hue(b2, a2p)
    zero = C1p*C2p == 0
    dh = 0.0 if zero else h2-h1
    if dh > 180:
        dh -= 360
    elif dh < -180:
        dh += 360
    dL, dC = L2-L1, C2p-C1p
    dH = 2*math.sqrt(C1p*C2p)*math.sin(rad(dh/2))
    Lb, Cb = (L1+L2)/2, (C1p+C2p)/2
    if zero:
        hb = h1+h2
    elif abs(h1-h2) > 180:
        hb = (h1+h2+360)/2 if h1+h2 < 360 else (h1+h2-360)/2
    else:
        hb = (h1+h2)/2
    T = 1-.17*math.cos(rad(hb-30))+.24*math.cos(rad(2*hb))+.32*math.cos(rad(3*hb+6))-.2*math.cos(rad(4*hb-63))
    cbp7 = Cb**7
    Rt = -math.sin(rad(2*30*math.exp(-((hb-275)/25)**2)))*2*math.sqrt(cbp7/(cbp7+p7))
    Sl = 1+.015*(Lb-50)**2/math.sqrt(20+(Lb-50)**2)
    Sc, Sh = 1+.045*Cb, 1+.015*Cb*T
    return math.sqrt((dL/Sl)**2+(dC/Sc)**2+(dH/Sh)**2+Rt*(dC/Sc)*(dH/Sh))


def delta_e(p, q):
    """CIEDE2000 between two '#rrggbb' colours (inf when one is malformed)."""
    a, b = lab_of(p), lab_of(q)
    return ciede2000(a, b) if a is not None and b is not None else float('inf')


def luminance(colour):
    r, g, b = (_linear(v) for v in (_rgb255(colour) or (0, 0, 0)))
    return .2126*r+.7152*g+.0722*b


def contrast_ratio(p, q):
    """WCAG 2 contrast ratio of two colours (1 to 21)."""
    a, b = luminance(p), luminance(q)
    return (max(a, b)+.05)/(min(a, b)+.05)


def _round1(v):
    return math.floor(v*10+.5)/10 if math.isfinite(v) else v      # JavaScript's Math.round(v * 10) / 10


def annotation_palette(team_a, team_b, pitch=None):
    """The annotation colours for these team colours on this grass (coaching.js annotationPalette, the same rule):
    {review, reviewFallback, zone, zoneFallback, relation, relationOpacity, halo, haloOpacity, scores}."""
    greens = [c for c in (pitch or PALETTE['pitch']) if _rgb255(c)] or list(PALETTE['pitch'])
    teams = [c for c in (team_a, team_b) if _rgb255(c)]

    def min_to(c, others):
        return min((delta_e(c, o) for o in others), default=float('inf'))
    reviews = sorted(((c, min_to(c, teams+greens)) for c in PALETTE['review']),
                     key=lambda p: (-p[1], PALETTE['review'].index(p[0])))
    review_ok = reviews[0][1] >= PALETTE['minDE']
    zones = [{'c': c, 'contrast': min(contrast_ratio(c, g) for g in greens),
              'team': min_to(c, teams), 'ball': delta_e(c, PALETTE['ball'])} for c in PALETTE['zone']]
    passing = sorted((z for z in zones if z['contrast'] >= PALETTE['minContrast'] and z['team'] >= PALETTE['minDE']
                      and z['ball'] >= PALETTE['ballMinDE']), key=lambda z: -z['contrast'])
    zone = passing[0] if passing else sorted((z for z in zones if z['ball'] >= PALETTE['ballMinDE']),
                                             key=lambda z: (-z['team'], -z['contrast']))[0]
    return {'review': reviews[0][0] if review_ok else WHITE, 'reviewFallback': not review_ok, 'zone': zone['c'],
            'zoneFallback': not passing, 'relation': PALETTE['relation'], 'relationOpacity': PALETTE['relationOpacity'],
            'halo': PALETTE['halo'], 'haloOpacity': PALETTE['haloOpacity'],
            'scores': {'review': _round1(reviews[0][1]), 'zoneContrast': math.floor(zone['contrast']*100+.5)/100,
                       'zoneTeam': _round1(zone['team'])}}


def video_palette(colours=None):
    """The colours of an overlay drawn over a video frame: annotation_palette() of the viewer's team colours on the
    board greens (the page's own colours) with the video's line styles."""
    c = team_colours(colours)
    return dict(annotation_palette(c['A'], c['B']), **VIDEO_STYLE)


def print_palette(colours=None):
    """The colours of a v2 top-down diagram on the light print pitch (see PRINT_ZONE): the page's review colour
    (white printed as the report ink), the print twin of the page's zone hue under the same rule, and the print line
    styles. Also {'screen': the page's palette} so a test can compare them."""
    c = team_colours(colours)
    teams = [c['A'], c['B']]
    screen = annotation_palette(c['A'], c['B'])
    review = PRINT_REVIEW_FOR_WHITE if screen['review'] == WHITE else screen['review']
    review_fallback = screen['reviewFallback'] or min(delta_e(review, t) for t in teams) < PALETTE['minDE']
    greens = (PITCH_FILL, PITCH_STRIPE)

    def ok(z):
        return min(contrast_ratio(z, g) for g in greens) >= PALETTE['minContrast'] \
            and min(delta_e(z, t) for t in teams) >= PALETTE['minDE']
    twin = PRINT_ZONE[screen['zone']]
    if ok(twin):
        zone, fallback = twin, screen['zoneFallback']
    else:
        passing = sorted((z for z in PRINT_ZONE.values() if ok(z)),
                         key=lambda z: -min(contrast_ratio(z, g) for g in greens))
        zone = passing[0] if passing else max(PRINT_ZONE.values(), key=lambda z: min(delta_e(z, t) for t in teams))
        fallback = not passing
    return dict(PRINT_STYLE, review=review, reviewFallback=review_fallback, zone=zone, zoneFallback=fallback,
                screen=screen)


def _centroid(q):
    return (sum(p[0] for p in q)/len(q), sum(p[1] for p in q)/len(q)) if q else None


def hatch(q, spacing):
    """45-degree hatch lines inside a polygon as segments (coaching.js hatch): each line x - y = c is cut at the
    polygon's edges and the inside pieces are kept (even-odd), so no clipping path is needed."""
    if not spacing or spacing <= 0 or len(q) < 3:
        return []
    d = [x-y for x, y in q]
    lo, hi = min(d), max(d)
    out = []
    c = lo+spacing/2
    while c < hi and len(out) < 400:
        xs = []
        for i in range(len(q)):
            (x0, y0), (x1, y1) = q[i], q[(i+1) % len(q)]
            f0, f1 = x0-y0-c, x1-y1-c
            if (f0 < 0) != (f1 < 0):
                t = f0/(f0-f1)
                xs.append((x0+(x1-x0)*t, y0+(y1-y0)*t))
        xs.sort()
        out += [[xs[k], xs[k+1]] for k in range(0, len(xs)-1, 2)]
        c += spacing
    return out


def ellipse(cx, cy, rx, ry, steps=36):
    """The points of an ellipse (a closed path; the PDF has circles only)."""
    return [(cx+rx*math.cos(2*math.pi*i/steps), cy+ry*math.sin(2*math.pi*i/steps)) for i in range(steps)]


def _check_glyph(x, y, size, colour):
    """A check mark (WinAnsi has no check character): a strength caption starts with it, as on the page."""
    return path([(x, y-size*.38), (x+size*.28, y-size*.1), (x+size*.75, y-size*.72)], stroke=colour,
                width=max(.6, size*.14))


def _free_spot(cands, label, size, points, placed, bounds):
    """The candidate (x, baseline, align) whose label stays farthest from the drawn markers and earlier labels (a
    candidate whose label would leave the drawing comes last): coaching.js freeSpot. placed collects the chosen
    label boxes."""
    width = _pdf.text_width(label, size)
    best, far = cands[0], -float('inf')
    for c in cands:
        x, y, align = c
        x0 = x if align == 'left' else x-width if align == 'right' else x-width/2
        box = (x0, y-size*.8, x0+width, y+size*.22)
        pts = list(points)+[((b[0]+b[2])/2, (b[1]+b[3])/2) for b in placed]
        d = min((math.hypot(px-min(box[2], max(box[0], px)), py-min(box[3], max(box[1], py))) for px, py in pts),
                default=1e6)
        if box[0] < bounds[0] or box[2] > bounds[2] or box[1] < bounds[1] or box[3] > bounds[3]:
            d -= 1e7
        if d > far:
            best, far = c, d
    x, y, align = best
    x0 = x if align == 'left' else x-width if align == 'right' else x-width/2
    placed.append((x0, y-size*.8, x0+width, y+size*.22))
    return best


# ---------------------------------------------------------------------------------------------------------------
# Primitive operations. Coordinates are points (pt) from the top left, rounded to 0.01 so the bytes are stable.

def _r(v):
    return round(float(v), 2)


def rect(x, y, w, h, fill=None, stroke=None, width=.8, alpha=1.0, dash=None):
    return {'op': 'rect', 'x': _r(x), 'y': _r(y), 'w': _r(w), 'h': _r(h), 'fill': fill, 'stroke': stroke,
            'width': _r(width), 'alpha': _r(alpha), 'dash': dash}


def circle(cx, cy, r, fill=None, stroke=None, width=.8, alpha=1.0, dash=None):
    return {'op': 'circle', 'cx': _r(cx), 'cy': _r(cy), 'r': _r(r), 'fill': fill, 'stroke': stroke,
            'width': _r(width), 'alpha': _r(alpha), 'dash': dash}


def path(points, fill=None, stroke=None, width=.8, alpha=1.0, dash=None, closed=False):
    return {'op': 'path', 'points': [[_r(x), _r(y)] for x, y in points], 'closed': bool(closed), 'fill': fill,
            'stroke': stroke, 'width': _r(width), 'alpha': _r(alpha), 'dash': dash}


def text(x, y, s, size=7.0, colour=INK, font='regular', align='left', halo=None, alpha=1.0):
    """Text with its baseline at y; align left/center/right about x; halo: a background colour behind it."""
    return {'op': 'text', 'x': _r(x), 'y': _r(y), 'text': str(s), 'size': _r(size), 'colour': colour, 'font': font,
            'align': align, 'halo': halo, 'alpha': _r(alpha)}


def clip(x, y, w, h):
    return {'op': 'clip', 'x': _r(x), 'y': _r(y), 'w': _r(w), 'h': _r(h)}


def unclip():
    return {'op': 'unclip'}


def text_box(op):
    """(x0, y0, x1, y1) of a text op: the width from the Helvetica tables, ascender .8 and descender .22 em."""
    w = _pdf.text_width(op['text'], op['size'], op.get('font', 'regular'))
    x = op['x']-(w/2 if op.get('align') == 'center' else w if op.get('align') == 'right' else 0)
    return x, op['y']-op['size']*.8, x+w, op['y']+op['size']*.22


# ---------------------------------------------------------------------------------------------------------------
# Helpers

def team_colours(given=None):
    """The team colours to draw with: the defaults, overridden by the viewer's valid '#rrggbb' choices only."""
    out = dict(COLOURS)
    for team, colour in (given.items() if isinstance(given, dict) else ()):
        if team in COLOURS and isinstance(colour, str) and re.fullmatch('#[0-9a-fA-F]{6}', colour):
            out[team] = colour.lower()
    return out


def _finite(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _team_of(el, ident=None):
    team = el.get('team') if isinstance(el, dict) else None
    if team in ('A', 'B'):
        return team
    ident = ident if ident is not None else (el.get('id') if isinstance(el, dict) else None)
    if isinstance(ident, str):
        if ident[:2] in ('A-', 'B-'):
            return ident[0]
        if ident in ('GK-A', 'GK-B'):
            return ident[-1]
    return None


def _shade(colour, k):
    """colour mixed toward black by k (0..1)."""
    c = colour.lstrip('#')
    r, g, b = (int(c[i:i+2], 16) for i in (0, 2, 4))
    return '#%02x%02x%02x' % tuple(int(round(v*(1-k))) for v in (r, g, b))


def _arrow_head(ax, ay, bx, by, length, half):
    """The triangle of an arrow head pointing from a to b, tip at b; None when a and b coincide."""
    dx, dy = bx-ax, by-ay
    n = math.hypot(dx, dy)
    if n < 1e-6:
        return None
    ux, uy = dx/n, dy/n
    return [(bx, by), (bx-ux*length-uy*half, by-uy*length+ux*half), (bx-ux*length+uy*half, by-uy*length-ux*half)]


def _arc(cx, cy, r, a0, a1, steps=None):
    steps = steps or max(8, int(abs(a1-a0)/math.radians(6)))
    return [(cx+r*math.cos(a0+(a1-a0)*i/steps), cy+r*math.sin(a0+(a1-a0)*i/steps)) for i in range(steps+1)]


def pitch_size(spec):
    """(length, width) in metres: the spec's pitch when it gives a sane one, else 105 x 68."""
    p = (spec or {}).get('pitch') or {}
    L, W = p.get('length'), p.get('width')
    if _finite(L) and _finite(W) and 50 <= L <= 130 and 30 <= W <= 100:
        return float(L), float(W)
    return PITCH['length'], PITCH['width']


def _pitch_ops(L, W, scale, ox, oy, width, height):
    """The pitch at scale pt/m with its top-left corner at (ox, oy), on a light surround."""
    m = lambda x, y: (ox+x*scale, oy+y*scale)                      # noqa: E731
    lw = max(.5, .12*scale)
    ops = [rect(0, 0, width, height, fill=SURROUND), rect(ox, oy, L*scale, W*scale, fill=PITCH_FILL)]
    for i in range(1, 10, 2):
        ops.append(rect(ox+L*scale*i/10, oy, L*scale/10, W*scale, fill=PITCH_STRIPE))
    line = lambda pts: ops.append(path([m(*p) for p in pts], stroke=PITCH_LINE, width=lw))   # noqa: E731
    line([(0, 0), (L, 0), (L, W), (0, W)])
    ops[-1]['closed'] = True
    line([(L/2, 0), (L/2, W)])
    ops.append(path([m(*p) for p in _arc(L/2, W/2, PITCH['circle'], 0, 2*math.pi, 60)], stroke=PITCH_LINE,
                    width=lw, closed=True))
    dot = max(.6, .3*scale)
    ops.append(circle(*m(L/2, W/2), dot, fill=PITCH_LINE))
    for x0, sign in ((0.0, 1), (L, -1)):
        pd, ph, gd, gh = PITCH['penaltyDepth'], PITCH['penaltyHalf'], PITCH['goalDepth'], PITCH['goalHalf']
        line([(x0, W/2-ph), (x0+sign*pd, W/2-ph), (x0+sign*pd, W/2+ph), (x0, W/2+ph)])
        line([(x0, W/2-gh), (x0+sign*gd, W/2-gh), (x0+sign*gd, W/2+gh), (x0, W/2+gh)])
        sx = x0+sign*PITCH['spot']
        ops.append(circle(*m(sx, W/2), dot, fill=PITCH_LINE))
        spread = math.acos((pd-PITCH['spot'])/PITCH['circle'])     # the arc outside the penalty area
        a0, a1 = (-spread, spread) if sign > 0 else (math.pi-spread, math.pi+spread)
        ops.append(path([m(*p) for p in _arc(sx, W/2, PITCH['circle'], a0, a1)], stroke=PITCH_LINE, width=lw))
        net = PITCH['goalNet']
        gx = -net if sign > 0 else L
        ops.append(rect(*m(gx, W/2-PITCH['goalWidth']/2), net*scale, PITCH['goalWidth']*scale, fill=WHITE,
                        stroke=PITCH_LINE, width=lw))
    for x, y, a in ((0, 0, 0), (L, 0, math.pi/2), (L, W, math.pi), (0, W, -math.pi/2)):
        ops.append(path([m(*p) for p in _arc(x, y, PITCH['cornerArc'], a, a+math.pi/2, 6)], stroke=PITCH_LINE,
                        width=lw))
    return ops


# ---------------------------------------------------------------------------------------------------------------
# DiagramSpec -> operations

def _bracket(el, point, P, marks, others, L, W, scale, width, height, size, placed=False):
    """The operations of a team length/width bracket (see BRACKET_GAP): (line ops, text ops), or the reason it is
    skipped. marks: {id: the drawn extent of one of the team's observed players, (x0, y0, x1, y1, cx, cy) pt with its
    ring and label}; others: the other players' (x, y) pt. placed: the ends a and b already stand where the bracket
    goes (evidence.py's measure with bracket True, BRACKET_OFFSET outside the team): it is drawn there, its ticks and
    leaders toward the team. Otherwise the bracket sits BRACKET_GAP beyond the outermost mark on
    el['side'] (left/right for a width, top/bottom for a length); when the side is missing or does not suit the
    bracket's direction, on the side where it and its value cover fewer other players (then: more room). A side
    that leaves no room for it and its value inside the drawing gives way to the other. A dashed leader runs from
    each end back to the player standing at that end (else to the end point), and the value is printed outside the
    bracket when it fits there, else inside it on a white halo."""
    a, b = point(el.get('a')), point(el.get('b'))
    if a is None or b is None:
        return 'bracket end missing or outside the pitch'
    axis = el.get('axis')
    across = axis == 'y' if axis in ('x', 'y') else abs(b[1]-a[1])*W >= abs(b[0]-a[0])*L   # width runs across
    if (abs(b[1]-a[1]) if across else abs(b[0]-a[0])) < 1e-6:
        return 'bracket ends coincide'
    pa, pb = P(*a), P(*b)
    perp, par = (0, 1) if across else (1, 0)          # the pt coordinate the bracket sits at, and the one it spans
    lo_i, hi_i = (0, 2) if across else (1, 3)         # a mark's low and high edge in that coordinate
    gap, tick, snap = BRACKET_GAP*scale, max(2.5, BRACKET_TICK*scale), max(1.0, .75*scale)
    label = el.get('text') if isinstance(el.get('text'), str) and el['text'].strip() else None
    tw = _pdf.text_width(label, size, 'bold') if label else 0.0
    need = (tw+3.0 if across else size+2.0) if label else 0.0
    lo_bound, hi_bound = 1.5, (width if across else height)-1.5
    team = list(marks.values())
    neg = min([m[lo_i] for m in team]+[pa[perp], pb[perp]])-gap
    pos = max([m[hi_i] for m in team]+[pa[perp], pb[perp]])+gap
    sides = BRACKET_SIDES['across' if across else 'along']
    room = {sides[0]: neg-need-lo_bound, sides[1]: hi_bound-need-pos}      # with the value outside it
    s0, s1 = sorted((pa[par], pb[par]))

    def covered(sd):
        """Other players under the bracket and its value on side sd (an opponent drawn over it hides it)."""
        o = -1.0 if sd == sides[0] else 1.0
        c_ = neg if o < 0 else pos
        lo, hi = sorted((c_-o*tick, c_+o*(need+2.2)))
        return sum(1 for q in others if lo-snap <= q[perp] <= hi+snap and s0-snap <= q[par] <= s1+snap)

    if placed:
        c = min(hi_bound, max(lo_bound, (pa[perp]+pb[perp])/2))
        centre = (min(m[lo_i] for m in team)+max(m[hi_i] for m in team))/2 if team else (lo_bound+hi_bound)/2
        outward = 1.0 if c >= centre else -1.0
    else:
        side = el.get('side') if el.get('side') in sides else min(sides, key=lambda s: (covered(s), -room[s]))
        other = sides[1-sides.index(side)]
        if room[side] < 0 <= room[other]:
            side = other                               # no room on the asked side: the other one
        outward = -1.0 if side == sides[0] else 1.0
        c = min(hi_bound, max(lo_bound, neg if outward < 0 else pos))
    named = [marks[i] for i in (el.get('ids') or ()) if isinstance(i, str) and i in marks]

    def at(p_val, s_val):
        return (p_val, s_val) if across else (s_val, p_val)

    lines = [path([at(c, s0), at(c, s1)], stroke=WHITE, width=2.6),
             path([at(c, s0), at(c, s1)], stroke=INK, width=.9)]
    for s_val, end in ((pa[par], pa), (pb[par], pb)):
        lines.append(path([at(c, s_val), at(c-outward*tick, s_val)], stroke=INK, width=.9))
        near = named or [m for m in team if abs(m[4+par]-s_val) <= snap]   # the end players: named, else nearest
        target = end[perp]
        if near:                                       # the edge of the player's mark that faces the bracket
            m = min(near, key=lambda m: abs(m[4+par]-s_val))
            target = (m[hi_i] if outward > 0 else m[lo_i])+outward*.8
        start = c-outward*tick
        if (start-target)*outward > 1.0:
            lines.append(path([at(start, s_val), at(target, s_val)], stroke=INK_2, width=.45, alpha=.8,
                              dash=[1.2, 1.0]))
    texts = []
    if label:
        mid = (s0+s1)/2
        if across:
            outside = c+outward*2.2
            fits = outside-tw >= 1.0 if outward < 0 else outside+tw <= width-1.0
            x = outside if fits else c-outward*(tick+1.2)
            align = ('right' if outward < 0 else 'left') if fits else ('left' if outward < 0 else 'right')
            texts.append(text(x, mid+size*.35, label, size, INK, 'bold', align, halo=WHITE))
        else:
            above, below = c-2.2, c+size*.8+2.2
            y = (above if above-size*.8 >= 1.0 else below) if outward < 0 else \
                (below if below+size*.22 <= height-1.0 else above)
            texts.append(text(mid, y, label, size, INK, 'bold', 'center', halo=WHITE))
    return lines, texts


CAPTION_LINES = 3           # lines of a diagram caption strip (a caption is at most 90 characters: schema CAPTION_MAX)


def wrap_caption(s, size, width):
    """The caption's lines in the strip above the pitch (bold), at most CAPTION_LINES."""
    lines = _pdf.wrap(str(s), size, width, 'bold')
    if len(lines) > CAPTION_LINES:
        lines = lines[:CAPTION_LINES]
        lines[-1] = lines[-1].rstrip('.,;: ')+'...'
    return lines


def _area_ops(q, kind, colour, pal, lw, radius, heavy=1.0):
    """A suggested area (DESIGN 5.3, the page's drawing): a dotted outline (round dots) over a halo, a 45-degree
    hatch at 45% and a 10% fill; a 'point' is a small dashed circle, never a dot."""
    cx, cy = _centroid(q)
    if kind == 'point':
        r = max(radius*.8, max(math.hypot(p[0]-cx, p[1]-cy) for p in q))
        return [circle(cx, cy, r, stroke=pal['halo'], width=2.2*lw*heavy, alpha=pal['haloOpacity']),
                circle(cx, cy, r, stroke=colour, width=1.1*lw, dash=[1.8*lw, 1.4*lw])]
    ops = [path(seg, stroke=colour, width=.5*lw, alpha=.45) for seg in hatch(q, 2.6*lw)]
    ops.append(path(q, fill=colour, alpha=.10, closed=True))
    ops.append(path(q, stroke=pal['halo'], width=2.4*lw*heavy, alpha=pal['haloOpacity'], closed=True))
    ops.append(path(q, stroke=colour, width=1.15*lw, dash=[DOT, 2.0*lw], closed=True))
    return ops


DOT = .01                   # pt: a dash this short with round caps is a dot of the line's width (the dotted area)


def _area_label_spots(q, size):
    """Candidate label spots around an area: above, below, left, right of its box."""
    xs, ys = [p[0] for p in q], [p[1] for p in q]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    cx, cy = (x0+x1)/2, (y0+y1)/2
    return [(cx, y0-size*.6, 'center'), (cx, y1+size*1.25, 'center'), (x0-size*.5, cy+size*.35, 'right'),
            (x1+size*.5, cy+size*.35, 'left')]


def _arrow_ops(q, colour, pal, lw, dashed=False):
    """A movement arrow over its halo with a filled head at the last point."""
    ops = [path(q, stroke=pal['halo'], width=2.4*lw, alpha=pal['haloOpacity']),
           path(q, stroke=colour, width=1.1*lw, dash=[2.0*lw, 1.5*lw] if dashed else None)]
    head = _arrow_head(*q[-2], *q[-1], 3.8*lw, 1.9*lw)
    if head:
        ops.append(path(head, fill=colour, stroke=pal['halo'], width=.4*lw, closed=True))
    return ops


def diagram_ops(spec, width=DIAGRAM_WIDTH, colours=None, labels=None, team_names=None, legend=True, palette=None,
                caption=True, adjustments=None):
    """A DiagramSpec (contract section 6) as drawing operations in a box of the given width (pt). Returns
    {'width', 'height', 'ops', 'skipped': [{'type', 'id', 'reason'}], 'drawn': {type: count}, 'legend': [...]}.
    colours: {'A': '#rrggbb', 'B': ...} (the viewer's team colours); labels: {id: display label at build time}
    (ids are shown when no label is given); team_names: {'A': 'Team A', ...}.
    The v2 annotation elements (focus, support, relation, channel, suggestedZone, moveArrow, caption; DESIGN 5.5) are
    drawn in palette (print_palette(colours) by default), at most V2_BUDGET of each; in a v2 drawing the people the
    annotation is not about are faint and unlabelled. caption: the caption element stands in a strip above the
    pitch (never over it, as on the page) when True and is left out when False (the PDF prints it beside the figure).
    adjustments: [{'polygon', 'label'}] the coach's own areas (DESIGN 10), thin and solid beside PitchIQ's."""
    spec = spec if isinstance(spec, dict) else {}
    colours = team_colours(colours)
    labels = labels or {}
    names = {'A': 'Team A', 'B': 'Team B', **(team_names or {})}
    pal = palette or print_palette(colours)
    L, W = pitch_size(spec)
    scale = float(width)/(L+2*MARGIN)
    els_in = spec.get('elements') if isinstance(spec.get('elements'), list) else []
    v2 = any(isinstance(el, dict) and el.get('type') in ('focus', 'support') for el in els_in)
    cap = next((el for el in els_in if isinstance(el, dict) and el.get('type') == 'caption'
                and isinstance(el.get('text'), str) and el['text'].strip()), None)
    strong = any(isinstance(el, dict) and el.get('type') == 'focus' and el.get('tone') == 'strength' for el in els_in)
    csize = max(4.8, min(7.0, width/44.0))
    glyph = csize*1.1 if strong else 0.0
    cap_lines = wrap_caption(cap['text'], csize, width-8-glyph) if cap is not None and caption else []
    strip = len(cap_lines)*csize*1.3+4 if cap_lines else 0.0
    ox, oy = MARGIN*scale, strip+MARGIN*scale
    pitch_h = strip+(W+2*MARGIN)*scale
    ops = _pitch_ops(L, W, scale, ox, oy, width, pitch_h)
    if cap_lines:
        ops.append(rect(0, 0, width, strip, fill='#f7faf8'))
        ops.append(path([(0, strip), (width, strip)], stroke=PITCH_LINE, width=.4))
        for k, line in enumerate(cap_lines):
            y = 2+csize*(.95+1.3*k)
            if k == 0 and strong:
                ops.append(_check_glyph(4, y, csize, BRAND_DARK))
            ops.append(text(4+glyph, y, line, csize, INK, 'bold'))
    P = lambda nx, ny: (ox+nx*L*scale, oy+ny*W*scale)               # noqa: E731
    radius = min(MAX_RADIUS, max(MIN_RADIUS, PLAYER_RADIUS*scale))
    small = max(4.6, radius*1.5)
    lw = width/DIAGRAM_WIDTH                                         # v2 stroke unit: 1 pt on a 243 pt column
    skipped, drawn = [], {}
    analysed = spec.get('team') if spec.get('team') in ('A', 'B') else None
    if cap is not None:
        drawn['caption'] = 1 if cap_lines else 0

    def skip(el, reason):
        skipped.append({'type': el.get('type') if isinstance(el, dict) else None,
                        'id': el.get('id') if isinstance(el, dict) else None, 'reason': reason})

    def point(v):
        if isinstance(v, (list, tuple)) and len(v) >= 2 and _finite(v[0]) and _finite(v[1]) \
                and -OUTSIDE <= v[0] <= 1+OUTSIDE and -OUTSIDE <= v[1] <= 1+OUTSIDE:
            return float(v[0]), float(v[1])
        return None

    layers = {k: [] for k in LAYERS}
    players = {}
    elements = els_in
    for el in elements:                                  # players first: units, arrows and suggestions refer to them
        if isinstance(el, dict) and el.get('type') == 'player':
            p = point((el.get('x'), el.get('y')))
            if p is None:
                skip(el, 'position missing or outside the pitch')
            elif el.get('id') in players:
                skip(el, 'duplicate player id')
            else:
                players[el.get('id')] = (el, P(*p))
    # The people of a v2 annotation (focus, support) carry their own positions; a marker is drawn for each of them
    # that no 'player' element draws. Naming follows the record: evidence.py sets focus.label False where the level
    # names nobody (or the identity was not confirmed), and then no person of the annotation is labelled.
    v2pos = {}
    for el in elements:
        if isinstance(el, dict) and el.get('type') in ('focus', 'support') and isinstance(el.get('id'), str):
            p = point((el.get('x'), el.get('y')))
            if p is not None and el['id'] not in v2pos:
                v2pos[el['id']] = P(*p) if el['id'] not in players else players[el['id']][1]
    named = not any(isinstance(el, dict) and el.get('type') == 'focus' and el.get('label') is False for el in elements)
    used, markered = {}, set()
    texts_later = []           # relation values and area labels: placed once every marker is known (_free_spot)

    def at(ident, given=None):
        """A v2 endpoint: given pixels / coordinates, a person, the suggested area's centre or the own goal."""
        if given is not None:
            q = point(given)
            return P(*q) if q is not None else None
        if ident == 'zone':
            z = next((e for e in elements if isinstance(e, dict) and e.get('type') == 'suggestedZone'), None)
            q = [point(v) for v in (z or {}).get('polygon') or []] if z else []
            q = [P(*v) for v in q if v is not None]
            return _centroid(q) if len(q) >= 3 else None
        if ident == 'goal':                                # the team's own goal (it attacks the other one)
            attack = spec.get('attack')
            return P(0.0 if attack == 'right' else 1.0, .5) if attack in ('left', 'right') else None
        if ident in v2pos:
            return v2pos[ident]
        return players[ident][1] if ident in players else None
    marks = {}                    # the drawn extent of the analysed team's observed players (brackets keep clear)
    for pid, (el, (x, y)) in players.items():
        if el.get('kind', 'O') != 'O' or (analysed is not None and _team_of(el) != analysed):
            continue
        r = radius*(1.75 if el.get('emphasis') else 1.0)
        x0, y0, x1, y1 = x-r, y-r, x+r, y+r
        if el.get('emphasis') or (analysed is not None and _team_of(el) == analysed):
            half = _pdf.text_width(str(labels.get(pid, pid)), small, 'bold')/2+1.2
            x0, x1, y0 = min(x0, x-half), max(x1, x+half), min(y0, y-radius-1.4-small*.8-.4)
        marks[pid] = (x0, y0, x1, y1, x, y)
    others = [q for pid, (el, q) in players.items()
              if not (el.get('kind', 'O') == 'O' and (analysed is None or _team_of(el) == analysed))]
    present = set()
    for el in elements:
        kind = el.get('type') if isinstance(el, dict) else None
        if kind not in DIAGRAM_ELEMENTS:
            skip(el if isinstance(el, dict) else {}, 'unknown element type')
            continue
        if kind in V2_BUDGET:
            used[kind] = used.get(kind, 0)+1
            if used[kind] > V2_BUDGET[kind]:
                skip(el, f'more than {V2_BUDGET[kind]} {kind} element{"s" if V2_BUDGET[kind] > 1 else ""} '
                         '(element budget, DESIGN 5.5)')
                continue
        if kind == 'player':
            if el.get('id') not in players or players[el.get('id')][0] is not el:
                continue
            (x, y) = players[el['id']][1]
            team = _team_of(el)
            colour = colours.get(team, NEUTRAL)
            observed = el.get('kind', 'O') == 'O'
            faint = v2 and not el.get('emphasis') and el.get('id') not in v2pos   # not part of the annotation
            if observed:
                layers['player'].append(circle(x, y, radius, fill=colour, stroke=EDGE, width=.5,
                                               alpha=.5 if faint else 1.0))
                present.add(f'team:{team}')
            else:
                layers['player'].append(circle(x, y, radius, fill=colour, stroke=_shade(colour, .35), width=.6,
                                               alpha=.55*(.6 if faint else 1.0), dash=[1.2, .9]))
                present.add('estimate')
            if el.get('emphasis') and not v2:
                layers['ring'].append(circle(x, y, radius*1.75, stroke=BRAND_DARK, width=1.1))
                present.add('emphasis')
            if (el.get('emphasis') or (team is not None and team == analysed and not v2)) \
                    and (named or el.get('id') not in v2pos):
                label = str(labels.get(el['id'], el['id']))
                layers['label'].append(text(x, y-radius-1.4, label, small, INK, 'bold', 'center', halo=WHITE))
            if spec.get('preset') == 'formation' and isinstance(el.get('role'), str) and el['role'].strip():
                layers['label'].append(text(x, y+radius+small*.95, el['role'].strip()[:8], small*.85, INK_2,
                                            'regular', 'center', halo=WHITE))
        elif kind in ('focus', 'support'):
            ident = el.get('id')
            if not isinstance(ident, str) or ident not in v2pos:
                skip(el, 'position missing or outside the pitch')
                continue
            x, y = v2pos[ident]
            team = el.get('team') if el.get('team') in ('A', 'B') else _team_of(el, ident) or \
                (analysed if kind == 'focus' else None)
            if ident not in players and ident not in markered:
                markered.add(ident)
                layers['player'].append(circle(x, y, radius, fill=colours.get(team, NEUTRAL), stroke=EDGE, width=.5))
                present.add(f'team:{team}')
                if named:
                    layers['label'].append(text(x, y-radius*1.9-1.0, str(labels.get(ident, ident)), small, INK,
                                                'bold', 'center', halo=WHITE))
            if kind == 'focus':
                k = .7 if el.get('secondary') else 1.0
                r = radius*1.85
                layers['focus'].append(circle(x, y, r, stroke=pal['halo'], width=2.6*lw*k, alpha=pal['haloOpacity'],
                                              dash=[1.6*lw, 1.2*lw] if pal['reviewFallback'] else None))
                layers['focus'].append(circle(x, y, r, stroke=pal['review'], width=1.3*lw*k))
                present.add('focus:strength' if el.get('tone') == 'strength' else 'focus:review')
            else:
                layers['support'].append(circle(x, y, radius*1.5, stroke=pal['support'], width=.7*lw,
                                                alpha=pal['supportOpacity'], dash=[1.3*lw, 1.0*lw]))
                present.add('support')
        elif kind == 'relation':
            a = at(el.get('a'), el.get('pa'))
            b = at(el.get('b'), el.get('pb'))
            if a is None or b is None:
                skip(el, 'relation end not placed on the pitch')
                continue
            ok = el.get('ok') is not False
            layers['relation'].append(path([a, b], stroke=pal['halo'], width=1.9*lw, alpha=.5*pal['haloOpacity']))
            layers['relation'].append(path([a, b], stroke=pal['relation'], width=.8*lw, alpha=pal['relationOpacity'],
                                           dash=None if ok else [2.0*lw, 1.6*lw]))
            present.add('relation:ok' if ok else 'relation:missing')
            if isinstance(el.get('text'), str) and el['text'].strip():
                n = math.hypot(b[0]-a[0], b[1]-a[1]) or 1.0
                nx, ny = -(b[1]-a[1])/n*small*.9, (b[0]-a[0])/n*small*.9
                mx, my = (a[0]+b[0])/2, (a[1]+b[1])/2
                texts_later.append(('relation', el['text'].strip(), small*.85,
                                    [(mx+nx, my+ny+small*.3, 'center'), (mx-nx, my-ny+small*.3, 'center')], INK))
        elif kind == 'channel':
            pts = [point(v) for v in el.get('polygon') or []] if isinstance(el.get('polygon'), list) else []
            if len(pts) < 3 or any(p is None for p in pts):
                skip(el, 'area polygon missing or outside the pitch')
                continue
            q = [P(*p) for p in pts]
            layers['channel'].append(path(q, fill=pal['review'], alpha=.12, closed=True))
            layers['channel'].append(path(q, stroke=pal['review'], width=.7*lw, alpha=.75, dash=[2.2*lw, 1.8*lw],
                                          closed=True))
            present.add('channel')
        elif kind == 'suggestedZone':
            pts = [point(v) for v in el.get('polygon') or []] if isinstance(el.get('polygon'), list) else []
            if len(pts) < 3 or any(p is None for p in pts):
                skip(el, 'area polygon missing or outside the pitch')
                continue
            q = [P(*p) for p in pts]
            heavy = 1.6 if pal['zoneFallback'] else 1.0
            layers['szone'].extend(_area_ops(q, el.get('kind'), pal['zone'], pal, lw, radius, heavy))
            present.add('suggestedZone:point' if el.get('kind') == 'point' else 'suggestedZone')
            label_ = el.get('label') if isinstance(el.get('label'), str) and el['label'].strip() else None
            if label_:
                texts_later.append(('zone', label_.strip(), small*.8, _area_label_spots(q, small*.8), pal['zone']))
        elif kind == 'moveArrow':
            a, b = at(None, el.get('from')), at(None, el.get('to'))
            if a is None or b is None:
                skip(el, 'arrow end outside the pitch')
                continue
            layers['marrow'].extend(_arrow_ops([a, b], pal['zone'], pal, lw, dashed=True))
            present.add('moveArrow')
        elif kind == 'caption':
            continue                                     # drawn in the strip above the pitch (or beside the figure)
        elif kind == 'unit':
            ids = el.get('ids') if isinstance(el.get('ids'), list) else []
            pts = [players[i][1] for i in ids if i in players and players[i][0].get('kind', 'O') == 'O']
            if len(pts) < 2:
                skip(el, 'fewer than two observed players of the unit in the diagram')
                continue
            if el.get('order') != 'given':
                pts.sort(key=lambda q: (q[1], q[0]))          # across the pitch, touchline to touchline
            team = _team_of(players[ids[0]][0]) if ids and ids[0] in players else None
            colour = colours.get(team, NEUTRAL)
            layers['unit'].append(path(pts, stroke=EDGE, width=2.4, alpha=.45))
            layers['unit'].append(path(pts, stroke=colour, width=1.4))
        elif kind == 'measure' and el.get('bracket'):     # evidence.py's width / length bracket, placed by it
            got = _bracket(el, point, P, marks, others, L, W, scale, width, pitch_h, small, placed=True)
            if isinstance(got, str):
                skip(el, got.replace('bracket', 'measure'))
                continue
            layers['measure'].extend(got[0])
            layers['label'].extend(got[1])
        elif kind == 'measure':
            a, b = point(el.get('a')), point(el.get('b'))
            if a is None or b is None:
                skip(el, 'measure end outside the pitch')
                continue
            (ax, ay), (bx, by) = P(*a), P(*b)
            layers['measure'].append(path([(ax, ay), (bx, by)], stroke=INK, width=.9))
            n = math.hypot(bx-ax, by-ay) or 1.0
            px, py = -(by-ay)/n*2.4, (bx-ax)/n*2.4
            for qx, qy in ((ax, ay), (bx, by)):
                layers['measure'].append(path([(qx-px, qy-py), (qx+px, qy+py)], stroke=INK, width=.9))
            if el.get('text'):
                layers['label'].append(text((ax+bx)/2, (ay+by)/2+2.2, el['text'], small, INK, 'bold', 'center',
                                            halo=WHITE))
        elif kind == 'bracket':
            got = _bracket(el, point, P, marks, others, L, W, scale, width, pitch_h, small)
            if isinstance(got, str):
                skip(el, got)
                continue
            lines, labels_ = got
            layers['measure'].extend(lines)
            layers['label'].extend(labels_)
        elif kind == 'zone':
            pts = [point(v) for v in (el.get('polygon') or [])] if isinstance(el.get('polygon'), list) else []
            if len(pts) < 3 or any(p is None for p in pts):
                skip(el, 'zone polygon missing or outside the pitch')
                continue
            fill, alpha = ZONE_STYLES.get(el.get('style'), ZONE_STYLES['gap'])
            q = [P(*p) for p in pts]
            layers['zone'].append(path(q, fill=fill, alpha=alpha, closed=True))
            layers['zone'].append(path(q, stroke=_shade(fill, .25), width=.6, dash=[2.0, 1.5], closed=True))
            present.add(f"zone:{el.get('style') if el.get('style') in ZONE_STYLES else 'gap'}")
        elif kind == 'band':
            lo, hi = el.get('from'), el.get('to')
            if el.get('axis') not in ('x', 'y') or not (_finite(lo) and _finite(hi)) \
                    or not (-OUTSIDE <= min(lo, hi) and max(lo, hi) <= 1+OUTSIDE):
                skip(el, 'band axis or range invalid')
                continue
            lo, hi = sorted((float(lo), float(hi)))
            fill, alpha = ZONE_STYLES.get(el.get('style'), ZONE_STYLES['gap'])
            if el['axis'] == 'x':
                (x0, y0), (x1, y1) = P(lo, 0), P(hi, 1)
                edges = [[(x0, y0), (x0, y1)], [(x1, y0), (x1, y1)]]
            else:
                (x0, y0), (x1, y1) = P(0, lo), P(1, hi)
                edges = [[(x0, y0), (x1, y0)], [(x0, y1), (x1, y1)]]
            layers['band'].append(rect(x0, y0, x1-x0, y1-y0, fill=fill, alpha=alpha))
            for e in edges:
                layers['band'].append(path(e, stroke=_shade(fill, .25), width=.6, dash=[2.0, 1.5]))
            present.add(f"zone:{el.get('style') if el.get('style') in ZONE_STYLES else 'gap'}")
        elif kind == 'arrow':
            raw = el.get('points') if isinstance(el.get('points'), list) else []
            pts, times = [], []
            for v in raw:
                p = point(v)
                if p is None:
                    pts = None
                    break
                pts.append(P(*p))
                times.append(float(v[2]) if len(v) > 2 and _finite(v[2]) else None)
            if not pts or len(pts) < 2:
                skip(el, 'arrow needs two or more points on the pitch')
                continue
            pieces, cur = [], [pts[0]]
            for i in range(1, len(pts)):
                if times[i] is not None and times[i-1] is not None and times[i]-times[i-1] > ARROW_GAP+1e-9:
                    pieces.append(cur)
                    cur = []
                cur.append(pts[i])
            pieces.append(cur)
            if len(pieces) > 1:
                skip(el, f'arrow cut where its samples are more than {ARROW_GAP:g} s apart')
            team = _team_of(el, el.get('id'))
            colour = colours.get(team, NEUTRAL)
            for piece in pieces:
                if len(piece) < 2:
                    continue
                layers['arrow'].append(path(piece, stroke=EDGE, width=2.2, alpha=.45))
                layers['arrow'].append(path(piece, stroke=colour, width=1.2))
            last = next((q for q in reversed(pieces) if len(q) >= 2), None)
            if last is not None:
                head = _arrow_head(*last[-2], *last[-1], 4.2, 2.1)
                if head:
                    layers['arrow'].append(path(head, fill=_shade(colour, .3), closed=True))
        elif kind == 'suggested':
            p = point((el.get('x'), el.get('y')))
            if p is None:
                skip(el, 'suggested position outside the pitch')
                continue
            x, y = P(*p)
            if el.get('id') in players:
                rx, ry = players[el['id']][1]
                layers['link'].append(path([(rx, ry), (x, y)], stroke=BRAND_DARK, width=.6, dash=[1.4, 1.2]))
            layers['suggested'].append(circle(x, y, radius*1.3, stroke=BRAND_DARK, width=.9, dash=[1.6, 1.2]))
            layers['suggested'].append(text(x, y+radius*1.3+small, ILLUSTRATION, max(4.2, small*.85), BRAND_DARK,
                                            'bold', 'center', halo=WHITE))
            present.add('suggested')
        elif kind == 'label':
            p = point(el.get('at'))
            if p is None or not el.get('text'):
                skip(el, 'label without text or outside the pitch')
                continue
            layers['label'].append(text(*P(*p), el['text'], small, INK, 'regular', 'center', halo=WHITE))
        drawn[kind] = drawn.get(kind, 0)+1
    for adj in adjustments or ():                        # the coach's own areas (DESIGN 10), beside PitchIQ's
        pts = [point(v) for v in (adj.get('polygon') or [])] if isinstance(adj, dict) else []
        if len(pts) < 3 or any(p is None for p in pts):
            skip({'type': 'adjustment'}, 'area polygon missing or outside the pitch')
            continue
        q = [P(*p) for p in pts]
        layers['szone'].append(path(q, stroke=pal['halo'], width=1.6*lw, alpha=pal['haloOpacity'], closed=True))
        layers['szone'].append(path(q, stroke=INK_2, width=.7*lw, closed=True))
        label_ = adj.get('label') if isinstance(adj.get('label'), str) and adj['label'].strip() else COACH_AREA
        texts_later.append(('zone', label_.strip()[:40], small*.8, _area_label_spots(q, small*.8), INK_2))
        present.add('adjustment')
        drawn['adjustment'] = drawn.get('adjustment', 0)+1
    if texts_later:                                      # values and area labels where they hide the fewest players
        pts = [q for _, q in players.values()]+list(v2pos.values())
        placed = []
        bounds = (0.5, strip+0.5, width-0.5, pitch_h-0.5)
        for _, s, size, cands, colour in texts_later:
            x, y, align = _free_spot(cands, s, size, pts, placed, bounds)
            layers['label'].append(text(x, y, s, size, colour, 'bold', align, halo=WHITE))
    for key in LAYERS:
        ops.extend(layers[key])
    attack = spec.get('attack')
    if attack in ('left', 'right') and analysed:
        y = oy+W*scale+MARGIN*scale*.62
        cx = ox+L*scale/2
        span = min(18.0, L*scale*.12)
        sign = 1 if attack == 'right' else -1
        tail, tip = (cx-sign*span/2, y-1.6), (cx+sign*span/2, y-1.6)
        ops.append(path([tail, tip], stroke=INK_2, width=.8))
        ops.append(path(_arrow_head(*tail, *tip, 3.2, 1.6), fill=INK_2, closed=True))
        label = f'{names.get(analysed, "Team " + analysed)} attacks'
        lx = cx-sign*(span/2+3)
        ops.append(text(lx, y, label, max(4.4, small*.9), INK_2, 'regular', 'right' if sign > 0 else 'left'))
    items = []
    if legend:
        for team in ('A', 'B'):
            if f'team:{team}' in present:
                items.append(('dot', colours[team], names.get(team, 'Team ' + team)))
        if 'emphasis' in present:
            items.append(('ring', BRAND_DARK, 'highlighted'))
        if 'estimate' in present:
            items.append(('estimate', NEUTRAL, 'estimated position'))
        for style, word in (('gap', 'gap'), ('space', 'space')):
            if f'zone:{style}' in present:
                items.append(('zone', ZONE_STYLES[style][0], word))
        if 'suggested' in present:
            items.append(('suggested', BRAND_DARK, 'coaching illustration'))
        if 'focus:review' in present or 'focus:strength' in present:
            items.append(('focus', pal['review'], 'player the moment is about'))
        if 'support' in present:
            items.append(('support', pal['support'], 'player involved'))
        if 'relation:ok' in present:
            items.append(('line', pal['relation'], 'link in place'))
        if 'relation:missing' in present:
            items.append(('dashline', pal['relation'], 'link not in place'))
        if 'channel' in present:
            items.append(('area', pal['review'], 'area the check is about'))
        if 'suggestedZone' in present:
            items.append(('dotted', pal['zone'], 'possible area (a suggestion, not tracked data)'))
        if 'suggestedZone:point' in present:
            items.append(('dotted', pal['zone'], 'possible spot (a suggestion, not tracked data)'))
        if 'moveArrow' in present:
            items.append(('dasharrow', pal['zone'], 'possible movement'))
        if 'adjustment' in present:
            items.append(('solidarea', INK_2, COACH_AREA))
        if spec.get('metresEstimated', True) is not False:          # absent: assumed, as on every run so far
            items.append(('note', None, f'metres estimated (assumed {L:g} x {W:g} m pitch)'))
    legend_ops, rows = _legend(items, pitch_h, width) if items else ([], 0)
    if rows:
        ops.append(rect(0, pitch_h, width, rows*LEGEND_HEIGHT, fill=WHITE))
        ops.extend(legend_ops)
    return {'width': _r(width), 'height': _r(pitch_h+rows*LEGEND_HEIGHT), 'ops': ops, 'skipped': skipped,
            'drawn': drawn, 'legend': [i[2] for i in items]}


def _legend(items, top, width, size=5.2):
    """Legend lines of (kind, colour, text) items under the pitch (top: its upper edge), wrapped to the width.
    Returns (ops, number of lines)."""
    ops = []
    x0 = x = 2.0
    row = 0
    for kind, colour, label in items:
        need = (0 if kind == 'note' else 7.5)+_pdf.text_width(label, size)
        if x > x0 and x+need > width-2:
            row += 1
            x = x0
        y = top+row*LEGEND_HEIGHT+LEGEND_HEIGHT*.72
        cy = y-size*.32
        if kind == 'dot':
            ops.append(circle(x+2.5, cy, 2.2, fill=colour, stroke=EDGE, width=.4))
        elif kind == 'ring':
            ops.append(circle(x+2.5, cy, 2.4, stroke=colour, width=.8))
        elif kind == 'estimate':
            ops.append(circle(x+2.5, cy, 2.2, fill=colour, alpha=.55, stroke=_shade(colour, .35), width=.5,
                              dash=[1.0, .8]))
        elif kind == 'zone':
            ops.append(rect(x, cy-2.2, 5, 4.4, fill=colour, alpha=.45))
        elif kind == 'suggested':
            ops.append(circle(x+2.5, cy, 2.4, stroke=colour, width=.7, dash=[1.2, .9]))
        elif kind == 'focus':
            ops.append(circle(x+2.5, cy, 2.3, stroke=colour, width=1.1))
        elif kind == 'support':
            ops.append(circle(x+2.5, cy, 2.3, stroke=colour, width=.6, dash=[1.1, .8]))
        elif kind in ('line', 'dashline'):
            ops.append(path([(x, cy), (x+5.5, cy)], stroke=colour, width=.8, alpha=.8,
                            dash=[1.4, 1.0] if kind == 'dashline' else None))
        elif kind == 'area':
            ops.append(rect(x, cy-2.2, 5.5, 4.4, fill=colour, alpha=.15))
            ops.append(rect(x, cy-2.2, 5.5, 4.4, stroke=colour, width=.5, dash=[1.2, 1.0]))
        elif kind == 'dotted':
            ops.append(rect(x, cy-2.2, 5.5, 4.4, fill=colour, alpha=.12))
            ops.append(rect(x, cy-2.2, 5.5, 4.4, stroke=colour, width=.8, dash=[.3, 1.1]))
        elif kind == 'dasharrow':
            ops.append(path([(x, cy), (x+4.2, cy)], stroke=colour, width=.8, dash=[1.2, .9]))
            ops.append(path(_arrow_head(x+3.0, cy, x+5.8, cy, 2.2, 1.3), fill=colour, closed=True))
        elif kind == 'solidarea':
            ops.append(rect(x, cy-2.2, 5.5, 4.4, stroke=colour, width=.6))
        if kind != 'note':
            x += 7.5
        ops.append(text(x, y, label, size, INK_2))
        x += _pdf.text_width(label, size)+6
    return ops, row+1


# ---------------------------------------------------------------------------------------------------------------
# VideoOverlaySpec -> operations over a placed frame

AREA_REACH = 10.0          # frame widths: how far beyond the picture a projected area's corner may stand
OVERLAY_LAYERS = ('zone', 'channel', 'szone', 'unit', 'arrow', 'measure', 'relation', 'marrow', 'box', 'player',
                  'support', 'focus', 'suggested', 'label')

def overlay_ops(spec, x, y, w, h, colours=None, labels=None, caption=True, palette=None, unit=None):
    """A VideoOverlaySpec (source-image pixels: {'tier', 'width', 'height', 'elements'}) as operations over a frame
    placed at (x, y, w, h) pt, scaled to that placement and clipped to it. Elements the tier does not allow are
    skipped and reported. Returns {'ops', 'skipped', 'tier', 'note'}.
    v2 elements follow schema.TIER_ELEMENTS (DESIGN 5.4: a suggested area, a channel or a movement arrow only on a
    world-tier frame, relation lines not at image tier) and the element budget, in the page's video colours
    (video_palette: the review colour around the feet, white involved players, white relation lines, the zone colour
    dotted over a dark halo). A v2 caption element is not drawn over the frame: the PDF prints it beside the figure
    (the page shows it as a chip). unit: the v2 stroke unit in pt (default w / 400: 0.61 pt on a 243 pt column); a
    caller that magnifies the frame (the PDF's zoom on the players involved) passes the unit of the visible column,
    so strokes keep their printed weight."""
    spec = spec if isinstance(spec, dict) else {}
    colours = team_colours(colours)
    labels = labels or {}
    pal = palette or video_palette(colours)
    tier = spec.get('tier') if spec.get('tier') in TIER_ELEMENTS else 'image'      # unknown: the most restrictive
    allowed = set(TIER_ELEMENTS[tier]) | {t for t in V2_TYPES if tier in V2_TIERS[t]}
    W, H = spec.get('width'), spec.get('height')
    skipped = []
    if not (_finite(W) and _finite(H) and W > 0 and H > 0):
        return {'ops': [], 'skipped': [{'type': None, 'id': None, 'reason': 'overlay without an image size'}],
                'tier': tier, 'note': TIER_NOTES[tier]}
    sx, sy = w/float(W), h/float(H)
    S = lambda px, py: (x+px*sx, y+py*sy)                            # noqa: E731
    ops = [clip(x, y, w, h)]
    size = max(4.6, min(7.0, w/55.0))
    ring = max(2.6, min(5.0, w/70.0))

    def ok(v):
        return isinstance(v, (list, tuple)) and len(v) >= 2 and _finite(v[0]) and _finite(v[1]) \
            and -.25*W <= v[0] <= 1.25*W and -.25*H <= v[1] <= 1.25*H

    def corner(v):
        # A projected area's corner may stand well outside the picture (evidence.py keeps an area whose corners
        # are in front of the camera and whose centre is in the picture; BC180 B's far-side channel reached past
        # the frame edge); the clip cuts it. Only an absurd value is refused.
        return isinstance(v, (list, tuple)) and len(v) >= 2 and _finite(v[0]) and _finite(v[1]) \
            and -AREA_REACH*W <= v[0] <= (1+AREA_REACH)*W and -AREA_REACH*H <= v[1] <= (1+AREA_REACH)*H

    def skip(el, reason):
        skipped.append({'type': el.get('type'), 'id': el.get('id'), 'reason': reason})

    feet = {}
    elements = spec.get('elements') if isinstance(spec.get('elements'), list) else []
    for el in elements:
        if isinstance(el, dict) and el.get('type') == 'player' and ok((el.get('x'), el.get('y'))) \
                and el.get('id') not in feet:
            feet[el.get('id')] = (el, S(el['x'], el['y']))
    people = {}                  # the v2 annotation's people (focus, support): their feet and boxes in the frame
    for el in elements:
        if isinstance(el, dict) and el.get('type') in ('focus', 'support') and isinstance(el.get('id'), str) \
                and ok((el.get('x'), el.get('y'))) and el['id'] not in people:
            people[el['id']] = (el, S(el['x'], el['y']))
    named = not any(isinstance(el, dict) and el.get('type') == 'focus' and el.get('label') is False for el in elements)
    lw = float(unit) if _finite(unit) and unit > 0 else max(.4, w/400.0)     # v2 stroke unit
    layers = {k: [] for k in OVERLAY_LAYERS}
    shown, used = set(), {}
    for el in elements:
        if not isinstance(el, dict):
            continue
        kind = el.get('type')
        if kind == 'caption':
            continue                     # printed beside the figure (the page shows it as a chip over the video)
        if kind not in allowed:
            if kind == 'band':
                reason = 'a band is pitch geometry: an overlay draws it as a zone polygon in image pixels'
            elif kind in V2_TIERS:
                reason = (f'not drawn on {"an" if tier == "image" else "a"} {tier}-tier frame (shown on the 2D '
                          'diagram)')
            elif kind in TIER_ELEMENTS['world']:
                reason = f'not drawn on {"an" if tier == "image" else "a"} {tier}-tier frame'
            else:
                reason = 'unknown element type'
            skip(el, reason)
            continue
        if kind in V2_BUDGET:
            used[kind] = used.get(kind, 0)+1
            if used[kind] > V2_BUDGET[kind]:
                skip(el, f'more than {V2_BUDGET[kind]} {kind} element{"s" if V2_BUDGET[kind] > 1 else ""} '
                         '(element budget, DESIGN 5.5)')
                continue
        team = _team_of(el, el.get('id'))
        colour = colours.get(team, NEUTRAL)
        if kind in ('player', 'box'):
            box = el.get('box')
            if isinstance(box, (list, tuple)) and len(box) == 4 and all(_finite(v) for v in box) and box[2] > 0 \
                    and box[3] > 0:
                bx, by = S(box[0], box[1])
                layers['box'].append(rect(bx, by, box[2]*sx, box[3]*sy, stroke=EDGE, width=1.6, alpha=.6))
                layers['box'].append(rect(bx, by, box[2]*sx, box[3]*sy,
                                          stroke=BRAND if el.get('emphasis') else colour,
                                          width=1.0 if el.get('emphasis') else .7))
                top = by
            elif kind == 'box':
                skip(el, 'box missing')
                continue
            else:
                top = None
            if kind == 'player':
                if el.get('id') not in feet or feet[el['id']][0] is not el:
                    skip(el, 'foot point missing or outside the image' if el.get('id') not in feet
                         else 'duplicate player id')
                    continue
                fx, fy = feet[el['id']][1]
                layers['player'].append(circle(fx, fy, ring, stroke=EDGE, width=1.8, alpha=.6))
                layers['player'].append(circle(fx, fy, ring, stroke=BRAND if el.get('emphasis') else colour,
                                               width=1.1))
                top = top if top is not None else fy-ring*4
            if el.get('id') is not None and top is not None:
                cx = (S(box[0]+box[2]/2, 0)[0] if isinstance(box, (list, tuple)) and len(box) == 4
                      else feet[el['id']][1][0])
                layers['label'].append(text(cx, top-1.5, str(labels.get(el['id'], el['id'])), size, WHITE, 'bold',
                                            'center', halo=EDGE))
        elif kind == 'unit':
            ids = el.get('ids') if isinstance(el.get('ids'), list) else []
            members = [feet[i] for i in ids if i in feet]
            if len(members) < 2:
                skip(el, 'fewer than two of its players located on this frame')
                continue
            if all(isinstance(m[0].get('pitch'), (list, tuple)) and len(m[0]['pitch']) >= 2 for m in members):
                members.sort(key=lambda m: (m[0]['pitch'][1], m[0]['pitch'][0]))   # along the line, by pitch y
            pts = [m[1] for m in members]
            colour = colours.get(_team_of(members[0][0]), NEUTRAL)
            layers['unit'].append(path(pts, stroke=EDGE, width=2.6, alpha=.55))
            layers['unit'].append(path(pts, stroke=colour, width=1.5))
        elif kind in ('measure', 'bracket'):
            if not (ok(el.get('a')) and ok(el.get('b'))):
                skip(el, f'{kind} end outside the image')
                continue
            (ax, ay), (bx, by) = S(*el['a'][:2]), S(*el['b'][:2])
            layers['measure'].append(path([(ax, ay), (bx, by)], stroke=EDGE, width=2.2, alpha=.55))
            layers['measure'].append(path([(ax, ay), (bx, by)], stroke=WHITE, width=1.0))
            n = math.hypot(bx-ax, by-ay)
            if kind == 'bracket' and n > 1e-6:          # on the video a bracket sits where its ends say: end ticks
                px, py = -(by-ay)/n*ring, (bx-ax)/n*ring
                for qx, qy in ((ax, ay), (bx, by)):
                    layers['measure'].append(path([(qx-px, qy-py), (qx+px, qy+py)], stroke=EDGE, width=2.2,
                                                  alpha=.55))
                    layers['measure'].append(path([(qx-px, qy-py), (qx+px, qy+py)], stroke=WHITE, width=1.0))
            if el.get('text'):
                layers['label'].append(text((ax+bx)/2, (ay+by)/2-2, el['text'], size, WHITE, 'bold', 'center',
                                            halo=EDGE))
        elif kind == 'zone':
            poly = el.get('polygon') if isinstance(el.get('polygon'), list) else []
            if len(poly) < 3 or not all(ok(v) for v in poly):
                skip(el, 'zone polygon missing or outside the image')
                continue
            fill, alpha = ZONE_STYLES.get(el.get('style'), ZONE_STYLES['gap'])
            q = [S(*v[:2]) for v in poly]
            layers['zone'].append(path(q, fill=fill, alpha=alpha+.05, closed=True))
            layers['zone'].append(path(q, stroke=fill, width=.8, closed=True))
        elif kind == 'arrow':
            pts = el.get('points') if isinstance(el.get('points'), list) else []
            if len(pts) < 2 or not all(ok(v) for v in pts):
                skip(el, 'arrow needs two or more points in the image')
                continue
            q = [S(*v[:2]) for v in pts]
            layers['arrow'].append(path(q, stroke=EDGE, width=2.4, alpha=.55))
            layers['arrow'].append(path(q, stroke=colour, width=1.3))
            head = _arrow_head(*q[-2], *q[-1], 4.5, 2.3)
            if head:
                layers['arrow'].append(path(head, fill=colour, stroke=EDGE, width=.4, closed=True))
        elif kind == 'suggested':
            if not ok((el.get('x'), el.get('y'))):
                skip(el, 'suggested position outside the image')
                continue
            sxp, syp = S(el['x'], el['y'])
            if el.get('id') in feet:
                layers['suggested'].append(path([feet[el['id']][1], (sxp, syp)], stroke=WHITE, width=.7,
                                                dash=[1.6, 1.2]))
            layers['suggested'].append(circle(sxp, syp, ring*1.3, stroke=WHITE, width=1.0, dash=[1.8, 1.3]))
            layers['label'].append(text(sxp, syp+ring*1.3+size, ILLUSTRATION, size*.85, WHITE, 'bold', 'center',
                                        halo=BRAND_DARK))
        elif kind == 'label':
            if not ok(el.get('at')) or not el.get('text'):
                skip(el, 'label without text or outside the image')
                continue
            layers['label'].append(text(*S(*el['at'][:2]), el['text'], size, WHITE, 'regular', 'center', halo=EDGE))
        elif kind in ('focus', 'support'):
            if el.get('id') not in people or people[el['id']][0] is not el:
                skip(el, 'foot point missing or outside the image' if el.get('id') not in people
                     else 'duplicate person')
                continue
            fx, fy = people[el['id']][1]
            box = el.get('box')
            box = [float(v) for v in box] if isinstance(box, (list, tuple)) and len(box) == 4 \
                and all(_finite(v) for v in box) and box[2] > 0 and box[3] > 0 else None
            if tier == 'image' and box is None:
                skip(el, 'an image-tier frame outlines boxes only, and this person has none')
                continue
            focus = kind == 'focus'
            k = .7 if el.get('secondary') else 1.0
            dash_halo = [1.6*lw, 1.2*lw] if focus and pal['reviewFallback'] else None
            if box is not None:
                bx, by = S(box[0], box[1])
                bw, bh = box[2]*sx, box[3]*sy
            if tier == 'image':                      # the box itself, outlined (no pitch geometry on this frame)
                pad = 1.2*lw
                r_ = (bx-pad, by-pad, bw+2*pad, bh+2*pad)
                if focus:
                    layers['focus'].append(rect(*r_, stroke=pal['halo'], width=3.0*lw*k, alpha=pal['haloOpacity'],
                                                dash=dash_halo))
                    layers['focus'].append(rect(*r_, stroke=pal['review'], width=1.5*lw*k))
                else:
                    layers['support'].append(rect(*r_, stroke=pal['support'], width=.8*lw, alpha=.7))
                top, cx = r_[1], bx+bw/2
            else:                                    # an ellipse around the feet
                least = (6.5 if focus else 5.5)*lw                 # a ring around the feet, never a blob
                rx = max(bw*(.85 if focus else .75), least) if box is not None else least*1.1
                q = ellipse(fx, fy, rx, rx*(.42 if focus else .38))  # rounder than the page's: print is small
                if focus:
                    layers['focus'].append(path(q, stroke=pal['halo'], width=3.2*lw*k, alpha=pal['haloOpacity'],
                                                dash=dash_halo, closed=True))
                    layers['focus'].append(path(q, stroke=pal['review'], width=1.7*lw*k, closed=True))
                else:
                    layers['support'].append(path(q, stroke=pal['halo'], width=1.9*lw, alpha=pal['haloOpacity'],
                                                  closed=True))
                    layers['support'].append(path(q, stroke=pal['support'], width=.85*lw, alpha=pal['supportOpacity'],
                                                  dash=[1.9*lw, 1.3*lw], closed=True))
                top, cx = (by if box is not None else fy-9*lw), fx
            if focus and named:
                layers['label'].append(text(cx, top-1.6, str(labels.get(el['id'], el['id'])), size,
                                            pal['review'], 'bold', 'center', halo=EDGE))
        elif kind == 'relation':
            def end(ident, given):
                if ok(given):
                    return S(*given[:2])
                return people[ident][1] if ident in people else feet[ident][1] if ident in feet else None
            a, b = end(el.get('a'), el.get('pa')), end(el.get('b'), el.get('pb'))
            if a is None or b is None:
                skip(el, 'relation end not located on this frame')
                continue
            okr = el.get('ok') is not False
            layers['relation'].append(path([a, b], stroke=pal['halo'], width=1.8*lw, alpha=.5))
            layers['relation'].append(path([a, b], stroke=pal['relation'], width=.9*lw, alpha=pal['relationOpacity'],
                                           dash=None if okr else [2.2*lw, 1.8*lw]))
            if isinstance(el.get('text'), str) and el['text'].strip():
                n = math.hypot(b[0]-a[0], b[1]-a[1]) or 1.0
                ox_, oy_ = -(b[1]-a[1])/n*size*.9, (b[0]-a[0])/n*size*.9
                tx, ty = (a[0]+b[0])/2+ox_, (a[1]+b[1])/2+oy_+size*.3
                layers['label'].append(text(tx, ty, el['text'].strip(), size*.85, WHITE, 'bold', 'center', halo=EDGE))
        elif kind == 'channel':
            poly = el.get('polygon') if isinstance(el.get('polygon'), list) else []
            if len(poly) < 3 or not all(corner(v) for v in poly):
                skip(el, 'area polygon missing or outside the image')
                continue
            q = [S(*v[:2]) for v in poly]
            layers['channel'].append(path(q, fill=pal['review'], alpha=.12, closed=True))
            layers['channel'].append(path(q, stroke=pal['review'], width=.8*lw, alpha=.75, dash=[2.4*lw, 2.0*lw],
                                          closed=True))
        elif kind == 'suggestedZone':
            poly = el.get('polygon') if isinstance(el.get('polygon'), list) else []
            if len(poly) < 3 or not all(corner(v) for v in poly):
                skip(el, 'area polygon missing or outside the image')
                continue
            q = [S(*v[:2]) for v in poly]
            layers['szone'].extend(_area_ops(q, el.get('kind'), pal['zone'], pal, lw, ring,
                                             1.6 if pal['zoneFallback'] else 1.0))
            if isinstance(el.get('label'), str) and el['label'].strip():
                xs, ys = [p_[0] for p_ in q], [p_[1] for p_ in q]
                layers['label'].append(text((min(xs)+max(xs))/2, min(ys)-size*.5, el['label'].strip(), size*.8,
                                            pal['zone'], 'bold', 'center', halo=EDGE))
        elif kind == 'moveArrow':
            a, b = el.get('from'), el.get('to')
            if not (ok(a) and ok(b)):
                skip(el, 'arrow end outside the image')
                continue
            layers['marrow'].extend(_arrow_ops([S(*a[:2]), S(*b[:2])], pal['zone'], pal, lw, dashed=True))
        shown.add(kind)
    for key in OVERLAY_LAYERS:
        ops.extend(layers[key])
    if caption:
        cs = max(4.4, min(6.0, w/60.0))
        cw = _pdf.text_width(OVERLAY_CAPTION, cs)+5
        ops.append(rect(x+3, y+h-cs-6, cw, cs+3.4, fill=EDGE, alpha=.7))
        ops.append(text(x+5.5, y+h-5.2, OVERLAY_CAPTION, cs, WHITE))
    ops.append(unclip())
    return {'ops': ops, 'skipped': skipped, 'tier': tier, 'note': overlay_note(tier, shown)}


def overlay_note(tier, shown):
    """The caption sentence of an overlay: what its tier allows, narrowed to what it actually drew (the fact check
    found the world-tier sentence 'drawn in pitch coordinates' under overlays that only joined foot pixels). A v2
    overlay gets the page's tier sentence (coaching.js TIER_UI), narrowed the same way at world tier."""
    shown = set(shown or ())
    if shown & set(V2_TYPES):
        if tier == 'world' and not shown & {'channel', 'suggestedZone', 'moveArrow', 'measure'}:
            return WORLD_LINES_NOTE if 'relation' in shown else PLAYERS_NOTE
        return V2_TIER_NOTES[tier]
    if tier == 'world':
        if shown & {'zone', 'measure', 'bracket', 'suggested'}:
            return TIER_NOTES['world']
        return WORLD_LINES_NOTE if 'unit' in shown else PLAYERS_NOTE
    if tier == 'pinned':
        return TIER_NOTES['pinned'] if shown & {'unit', 'measure', 'bracket', 'zone'} else PLAYERS_NOTE
    return TIER_NOTES['image']


# ---------------------------------------------------------------------------------------------------------------
# Operations -> PNG (cv2): the previews and the self-check of what the PDF draws

_FONT = cv2.FONT_HERSHEY_SIMPLEX


def _bgr(colour):
    c = (colour or '#000000').lstrip('#')
    r, g, b = (int(c[i:i+2], 16) for i in (0, 2, 4))
    return (b, g, r)


def _dashes(points, dash, closed):
    """Split a polyline into its dash segments ([on, off] lengths in the same units as the points)."""
    pts = list(points)+([points[0]] if closed else [])
    on, off = max(0.0, float(dash[0])), max(0.0, float(dash[1] if len(dash) > 1 else dash[0]))
    if on <= 1e-6:
        return []
    if off <= 1e-6:
        return [pts]
    out, cur = [], None
    drawing, left = True, on                  # the dash state and the length left in it (no modulo: no float traps)
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        seg = math.hypot(x1-x0, y1-y0)
        pos = 0.0
        while seg-pos > 1e-9:
            step = min(left, seg-pos)
            a = (x0+(x1-x0)*pos/seg, y0+(y1-y0)*pos/seg)
            b = (x0+(x1-x0)*(pos+step)/seg, y0+(y1-y0)*(pos+step)/seg)
            if drawing:
                if cur is None:
                    cur = [a]
                cur.append(b)
            pos += step
            left -= step
            if left <= 1e-9:
                if drawing and cur is not None:
                    out.append(cur)
                    cur = None
                drawing = not drawing
                left = on if drawing else off
    if cur is not None:
        out.append(cur)
    return out


def _op_box(op, scale):
    """The op's pixel bounding box (x0, y0, x1, y1), generous by the line width."""
    kind = op['op']
    pad = (op.get('width') or 1)*scale+3
    if kind == 'rect' or kind == 'image':
        xs, ys = (op['x'], op['x']+op['w']), (op['y'], op['y']+op['h'])
    elif kind == 'circle':
        xs, ys = (op['cx']-op['r'], op['cx']+op['r']), (op['cy']-op['r'], op['cy']+op['r'])
    elif kind == 'path':
        xs, ys = [p[0] for p in op['points']], [p[1] for p in op['points']]
    elif kind == 'text':
        x0, y0, x1, y1 = text_box(op)
        xs, ys = (x0-3, x1+3), (y0-3, y1+3)
    else:
        return None
    return (int(math.floor(min(xs)*scale-pad)), int(math.floor(min(ys)*scale-pad)),
            int(math.ceil(max(xs)*scale+pad)), int(math.ceil(max(ys)*scale+pad)))


def _draw(img, op, scale, ox, oy):
    """Draw one op onto img whose top-left pixel is (ox, oy) of the page raster."""
    fx = lambda v: int(round((v*scale-ox)*16))                      # noqa: E731
    fy = lambda v: int(round((v*scale-oy)*16))                      # noqa: E731
    kind = op['op']
    thick = max(1, int(round((op.get('width') or 0)*scale)))
    if kind == 'image':
        data = np.frombuffer(op['jpeg'], np.uint8)
        pic = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if pic is None:
            return
        w, h = max(1, int(round(op['w']*scale))), max(1, int(round(op['h']*scale)))
        pic = cv2.resize(pic, (w, h), interpolation=cv2.INTER_AREA)
        x0, y0 = int(round(op['x']*scale-ox)), int(round(op['y']*scale-oy))
        X0, Y0 = max(0, x0), max(0, y0)
        X1, Y1 = min(img.shape[1], x0+w), min(img.shape[0], y0+h)
        if X1 > X0 and Y1 > Y0:
            img[Y0:Y1, X0:X1] = pic[Y0-y0:Y1-y0, X0-x0:X1-x0]
        return
    if kind == 'text':
        x0, y0, x1, y1 = text_box(op)
        if op.get('halo'):
            cv2.rectangle(img, (int(round((x0-1.2)*scale-ox)), int(round((y0-.4)*scale-oy))),
                          (int(round((x1+1.2)*scale-ox)), int(round((y1+.4)*scale-oy))), _bgr(op['halo']), -1)
        s = ''.join(c if 32 <= ord(c) < 127 else '?' for c in op['text'])
        if not s.strip():
            return
        want = (x1-x0)*scale
        base = cv2.getTextSize(s, _FONT, 1.0, 1)[0][0] or 1
        nominal = op['size']*scale*.72/22.0
        fs = min(nominal*1.35, max(nominal*.7, want/base))
        t = 2 if op.get('font') == 'bold' and fs > .45 else 1
        cv2.putText(img, s, (int(round(x0*scale-ox)), int(round(op['y']*scale-oy))), _FONT, fs, _bgr(op['colour']),
                    t, cv2.LINE_AA)
        return
    if kind == 'rect':
        pts = [(op['x'], op['y']), (op['x']+op['w'], op['y']), (op['x']+op['w'], op['y']+op['h']),
               (op['x'], op['y']+op['h'])]
        closed = True
    elif kind == 'circle':
        pts = _arc(op['cx'], op['cy'], op['r'], 0, 2*math.pi, 48)[:-1]
        closed = True
    elif kind == 'path':
        pts = [tuple(p) for p in op['points']]
        closed = op.get('closed', False)
    else:
        return
    arr = np.array([[fx(px), fy(py)] for px, py in pts], np.int32)
    if op.get('fill') and len(pts) >= 3:
        cv2.fillPoly(img, [arr], _bgr(op['fill']), cv2.LINE_AA, shift=4)
    if op.get('stroke'):
        if op.get('dash'):
            for seg in _dashes(pts, op['dash'], closed):
                length = sum(math.hypot(q[0]-p[0], q[1]-p[1]) for p, q in zip(seg, seg[1:]))
                if length*scale < thick:         # a dash shorter than the line is a round dot (the PDF's round caps)
                    mx, my = seg[len(seg)//2]
                    cv2.circle(img, (fx(mx), fy(my)), max(8, int(round(thick*8))), _bgr(op['stroke']), -1,
                               cv2.LINE_AA, shift=4)
                    continue
                a = np.array([[fx(px), fy(py)] for px, py in seg], np.int32)
                cv2.polylines(img, [a], False, _bgr(op['stroke']), thick, cv2.LINE_AA, shift=4)
        else:
            cv2.polylines(img, [arr], closed, _bgr(op['stroke']), thick, cv2.LINE_AA, shift=4)


def ops_to_png(ops, width, height, scale=2.0, background=WHITE, show_links=False):
    """Rasterize operations (a diagram, an overlay or a whole PDF page) to a BGR image of width x height points at
    scale pixels per point. Translucency, clipping and dashes follow the PDF; text uses cv2's Hershey font sized to
    the Helvetica width tables, so line breaks and alignment match the PDF approximately. show_links outlines link
    areas (the self-check of where the PDF's clickable rectangles sit)."""
    W, H = int(math.ceil(width*scale)), int(math.ceil(height*scale))
    img = np.empty((H, W, 3), np.uint8)
    img[:] = _bgr(background)
    clips = [(0, 0, W, H)]
    for op in ops:
        kind = op.get('op')
        if kind == 'clip':
            c = clips[-1]
            x0, y0 = int(math.floor(op['x']*scale)), int(math.floor(op['y']*scale))
            x1, y1 = int(math.ceil((op['x']+op['w'])*scale)), int(math.ceil((op['y']+op['h'])*scale))
            clips.append((max(c[0], x0), max(c[1], y0), min(c[2], x1), min(c[3], y1)))
            continue
        if kind == 'unclip':
            if len(clips) > 1:
                clips.pop()
            continue
        if kind == 'link':
            if show_links:
                cv2.rectangle(img, (int(op['x']*scale), int(op['y']*scale)),
                              (int((op['x']+op['w'])*scale), int((op['y']+op['h'])*scale)), (230, 160, 60), 1)
            continue
        box = _op_box(op, scale)
        if box is None:
            continue
        c = clips[-1]
        x0, y0, x1, y1 = max(box[0], c[0]), max(box[1], c[1]), min(box[2], c[2]), min(box[3], c[3])
        if x1 <= x0 or y1 <= y0:
            continue
        region = img[y0:y1, x0:x1]
        layer = region.copy()
        _draw(layer, op, scale, x0, y0)
        alpha = op.get('alpha', 1.0)
        if alpha is None or alpha >= 1:
            region[:] = layer
        else:
            region[:] = cv2.addWeighted(layer, float(alpha), region, 1-float(alpha), 0)
    return img


def diagram_png(spec, width_px=720, colours=None, labels=None, team_names=None):
    """A DiagramSpec as a BGR image width_px wide (a preview; the PDF draws the same operations as vectors)."""
    d = diagram_ops(spec, DIAGRAM_WIDTH, colours, labels, team_names)
    return ops_to_png(d['ops'], d['width'], d['height'], width_px/d['width'])
