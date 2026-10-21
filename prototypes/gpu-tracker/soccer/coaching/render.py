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
- A suggested position is dashed and always labelled COACHING ILLUSTRATION, with a dashed link from the player's real
  position: it is advice, not data.
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
"""
import math
import re

import cv2
import numpy as np

from . import pdf as _pdf   # base-14 Helvetica widths, shared with the PDF writer so both backends agree

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

DIAGRAM_ELEMENTS = ('player', 'unit', 'measure', 'bracket', 'zone', 'band', 'arrow', 'suggested', 'label')
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


def diagram_ops(spec, width=DIAGRAM_WIDTH, colours=None, labels=None, team_names=None, legend=True):
    """A DiagramSpec (contract section 6) as drawing operations in a box of the given width (pt). Returns
    {'width', 'height', 'ops', 'skipped': [{'type', 'id', 'reason'}], 'drawn': {type: count}, 'legend': [...]}.
    colours: {'A': '#rrggbb', 'B': ...} (the viewer's team colours); labels: {id: display label at build time}
    (ids are shown when no label is given); team_names: {'A': 'Team A', ...}."""
    spec = spec if isinstance(spec, dict) else {}
    colours = team_colours(colours)
    labels = labels or {}
    names = {'A': 'Team A', 'B': 'Team B', **(team_names or {})}
    L, W = pitch_size(spec)
    scale = float(width)/(L+2*MARGIN)
    ox = oy = MARGIN*scale
    pitch_h = (W+2*MARGIN)*scale
    ops = _pitch_ops(L, W, scale, ox, oy, width, pitch_h)
    P = lambda nx, ny: (ox+nx*L*scale, oy+ny*W*scale)               # noqa: E731
    radius = min(MAX_RADIUS, max(MIN_RADIUS, PLAYER_RADIUS*scale))
    small = max(4.6, radius*1.5)
    skipped, drawn = [], {}
    analysed = spec.get('team') if spec.get('team') in ('A', 'B') else None

    def skip(el, reason):
        skipped.append({'type': el.get('type') if isinstance(el, dict) else None,
                        'id': el.get('id') if isinstance(el, dict) else None, 'reason': reason})

    def point(v):
        if isinstance(v, (list, tuple)) and len(v) >= 2 and _finite(v[0]) and _finite(v[1]) \
                and -OUTSIDE <= v[0] <= 1+OUTSIDE and -OUTSIDE <= v[1] <= 1+OUTSIDE:
            return float(v[0]), float(v[1])
        return None

    layers = {k: [] for k in ('band', 'zone', 'unit', 'arrow', 'measure', 'link', 'player', 'ring', 'suggested',
                              'label')}
    players = {}
    elements = spec.get('elements') if isinstance(spec.get('elements'), list) else []
    for el in elements:                                  # players first: units, arrows and suggestions refer to them
        if isinstance(el, dict) and el.get('type') == 'player':
            p = point((el.get('x'), el.get('y')))
            if p is None:
                skip(el, 'position missing or outside the pitch')
            elif el.get('id') in players:
                skip(el, 'duplicate player id')
            else:
                players[el.get('id')] = (el, P(*p))
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
        if kind == 'player':
            if el.get('id') not in players or players[el.get('id')][0] is not el:
                continue
            (x, y) = players[el['id']][1]
            team = _team_of(el)
            colour = colours.get(team, NEUTRAL)
            observed = el.get('kind', 'O') == 'O'
            if observed:
                layers['player'].append(circle(x, y, radius, fill=colour, stroke=EDGE, width=.5))
                present.add(f'team:{team}')
            else:
                layers['player'].append(circle(x, y, radius, fill=colour, stroke=_shade(colour, .35), width=.6,
                                               alpha=.55, dash=[1.2, .9]))
                present.add('estimate')
            if el.get('emphasis'):
                layers['ring'].append(circle(x, y, radius*1.75, stroke=BRAND_DARK, width=1.1))
                present.add('emphasis')
            if el.get('emphasis') or (team is not None and team == analysed):
                label = str(labels.get(el['id'], el['id']))
                layers['label'].append(text(x, y-radius-1.4, label, small, INK, 'bold', 'center', halo=WHITE))
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
    for key in ('band', 'zone', 'unit', 'arrow', 'measure', 'link', 'player', 'ring', 'suggested', 'label'):
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
        if kind != 'note':
            x += 7.5
        ops.append(text(x, y, label, size, INK_2))
        x += _pdf.text_width(label, size)+6
    return ops, row+1


# ---------------------------------------------------------------------------------------------------------------
# VideoOverlaySpec -> operations over a placed frame

def overlay_ops(spec, x, y, w, h, colours=None, labels=None, caption=True):
    """A VideoOverlaySpec (source-image pixels: {'tier', 'width', 'height', 'elements'}) as operations over a frame
    placed at (x, y, w, h) pt, scaled to that placement and clipped to it. Elements the tier does not allow are
    skipped and reported. Returns {'ops', 'skipped', 'tier', 'note'}."""
    spec = spec if isinstance(spec, dict) else {}
    colours = team_colours(colours)
    labels = labels or {}
    tier = spec.get('tier') if spec.get('tier') in TIER_ELEMENTS else 'image'      # unknown: the most restrictive
    allowed = TIER_ELEMENTS[tier]
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

    def skip(el, reason):
        skipped.append({'type': el.get('type'), 'id': el.get('id'), 'reason': reason})

    feet = {}
    elements = spec.get('elements') if isinstance(spec.get('elements'), list) else []
    for el in elements:
        if isinstance(el, dict) and el.get('type') == 'player' and ok((el.get('x'), el.get('y'))) \
                and el.get('id') not in feet:
            feet[el.get('id')] = (el, S(el['x'], el['y']))
    layers = {k: [] for k in ('zone', 'unit', 'arrow', 'measure', 'box', 'player', 'suggested', 'label')}
    shown = set()
    for el in elements:
        if not isinstance(el, dict):
            continue
        kind = el.get('type')
        if kind not in allowed:
            if kind == 'band':
                reason = 'a band is pitch geometry: an overlay draws it as a zone polygon in image pixels'
            elif kind in TIER_ELEMENTS['world']:
                reason = f'not drawn on {"an" if tier == "image" else "a"} {tier}-tier frame'
            else:
                reason = 'unknown element type'
            skip(el, reason)
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
        shown.add(kind)
    for key in ('zone', 'unit', 'arrow', 'measure', 'box', 'player', 'suggested', 'label'):
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
    found the world-tier sentence 'drawn in pitch coordinates' under overlays that only joined foot pixels)."""
    shown = set(shown or ())
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
