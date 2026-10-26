"""The coaching report as a PDF: a dependency-free PDF 1.4 writer and the report's page layout.

Why a writer of our own. The server audit built the same two-page mock-up three ways (a title, a paragraph, a URI
link, an internal link, three 1280 px frames and a vector pitch):
- this kind of small writer: 415 kB, 2 ms, the JPEGs passed through untouched (/DCTDecode, full resolution), URI and
  internal links, and deterministic bytes the unittest suite can compare;
- matplotlib's PdfPages: 944 kB with the frames resampled to 685-711 px and re-encoded, no internal links, and
  matplotlib is not pinned (it arrives only through ultralytics);
- the browser's print-to-PDF: needs a Chromium binary on the server (the cloud plan) or a print dialog, and can be
  tested only through a browser.
The price is text: the base-14 Helvetica fonts in WinAnsi (cp1252) encoding, which every viewer has, so no font is
embedded. Templates and drills are ASCII/WinAnsi by contract; any other character (a typed note, a team name) is
written as '?' and counted in the result, or refused with strict=True. The widths below are the Adobe Core14 AFM
metrics as shipped with matplotlib (phvr8a/phvb8a.afm, in WinAnsi order; the oblique face has the regular widths),
so wrapping and alignment are exact. The writer was checked by parsing its own output (tests/test_coaching_pdf.py:
every xref offset, stream length and reference) and, on 2026-10-18, by a scratch reader that interprets the content
streams themselves (q Q cm re m l c h W n S f B rg RG w d gs BT Tf Tm Tj ET Do) and paints them with cv2: on layout
tests built from real frames of the broadcast, amateur and Veo uploads (5 pages, 3 screenshots with overlays,
165-185 kB, built in 13-29 ms after 0.4-0.9 s of layout and frame decoding) and on a report assembled by report.py
(12 pages), every page matched the cv2 drawing of the same operations (mean difference 0.0-0.14 grey levels: arc
sampling and image rounding). No PDF viewer exists in this environment, so the output still needs one look in Edge
or Acrobat.

The layout (write_report_pdf) prints what the report holds and nothing more: the report's own sentences, numbers
and evidence (soccer/coaching/report.py, evidence.py), with fixed headings and explanations from this module. On
light, print-friendly A4 pages with the PitchIQ green, in the order of the Studio's Coaching tab (section_order):
the title, analysed range and team; summary; observed strengths, potential improvements and review opportunities
with their evidence (a top-down diagram and the exact source frame with its analysis overlay, side by side); the
tactical breakdown; key statistics; priorities; training recommendations; coverage, confidence and limitations; every
evidence timestamp; expert feedback. Choices made for honesty rather than looks:
- a professional comparison is claimed only where the observation's baseline was available (compared()): the coach
  review of 2026-10-18 found "Compared with the professional reference" under all four formation findings, whose
  baselines say available False;
- each limitation line is printed once, under Limitations, in words: the first PDFs printed every line three times
  (page 1, breakdown, coverage; 318 printings of 141 lines over the 24 integration reports) and a fourth time as raw
  reason codes; now 141 printings, no code, and 137 pages instead of 149;
- a coach's note is printed exactly as typed (never through the id -> display substitution), and a verdict of another
  run is listed as that run's, never on this report's finding (team letters and ids are assigned per run);
- a table row taller than a page continues on the next page instead of running off it;
- every timestamp is printed as text, absolute source time first (mm:ss.ss, the Studio's transport format) and the
  offset into the analysed segment second; a timestamp is a link only when the caller passes a valid http(s)
  playback URL (the local server: it works only on that machine with that video open, so the text is what counts);
- no video is embedded; a screenshot is the exact frame (frames.py) and the overlay is drawn over it as vectors, at
  the frame's tier (render.py), captioned as an analysis overlay;
- the report stores identities (ids), the Studio shows display labels that renumber after corrections: with
  options['labels'] (id -> display, from the current result) diagrams and text show the display labels;
- an empty section says so in one sentence instead of disappearing, and observations without enough evidence are
  listed as not assessed, never as strengths or weaknesses.

Coaching v2 reports (schema 'report' version 2: is_v2(), _ReportV2). The PDF is the page's seven sections (DESIGN
9.2) in the page's words (coaching.js renderV2, its word tables copied in ASCII, and a test compares the codes the two
cover): MATCH OVERVIEW (the coaching profile, footage, scale, who had the ball and its caveat, notes, what the level
aims for and does not expect with the source ids, the shape described, what was checked and why not, what PitchIQ
does not measure, the described findings), WHAT WENT WELL, WHAT NEEDS WORK (then the possible issues to review), KEY
MOMENTS, TACTICAL DIAGRAMS, TRAINING DRILLS, DEVELOPMENT PROGRESSION and the next objective, then the sources credited
with their links, the coach's feedback and the report's identity. The v1 layout above stays for stored v1 reports.
- Every number a card adds follows the level's register (profile.register.numbers): no counts at level 1, counts at
  level 2, metres from level 3 (rounded at 3), shares from 5, rates from 6, as the page. The fixed words of this
  module pass schema.event_problems and every level's banned words (tested on the scene reports at levels 1, 2, 4, 7).
- Key moments are in time order (E1, E2, ...), each with its absolute and segment times, frame, replay window, the
  top-down diagram (print colours) and the exact source frame with the annotation at what its tier allows (video
  colours). The screenshot is zoomed on the people of the annotation by clipping the magnified frame to the column
  (ZOOM_*): at the half-page column a broadcast player is about 2.5 pt wide in the whole frame. Measured on the five
  run copies (both teams, levels 2/4/6; scratch pdf_w7/measure.json): 32 of 32 screenshots zoomed, 223 of 225 moments
  over all 50 reports, zoom 1.31-2.22 (median 2.22, the cap); 4-9 pages, 20-757 kB, 0.02-0.49 s per PDF with the frames
  decoded from the uploads; 0 elements skipped, 0 links dropped, 0 characters replaced. Chrome 155's PDF viewer
  (PDFium, headless) showed BC180 B's level-6 pages with the zoomed frames clipped to their columns and each ring at
  the player's feet.
- The dotted-area sentence of "How to read this report" is printed only when the report holds a suggested area, and
  'COACHING ILLUSTRATION' never appears in a v2 PDF (critic B4 / M2: the v1 cover promised illustrations no producer
  emitted; the v1 cover now names them only when one is drawn). Possession the coach marked is labelled wherever it is
  used. A moment that may name nobody (evidence.py's naming rule, read per finding) gets no player label, no
  'Players:' line and no mover id.
- Fixed with this version for both layouts: the out-of-date sentence no longer prints '..' (critic N1), the footer is
  cut before it reaches the page number (N2), a heading keeps the first block that follows it (N5), clock() rounds
  before it splits minutes (N8: 59.999 s was '00:60.00'), and a custom range is in the file name (R4).
"""
import hashlib
import math
import os
import re
import struct
import time
import uuid
import zlib
from pathlib import Path

A4 = (595.28, 841.89)                 # pt
MARGINS = (48.0, 52.0, 48.0, 54.0)    # pt: left, top, right, bottom (the footer sits in the bottom margin)
COLUMN_GAP = 13.0                     # pt between the evidence diagram and the screenshot
FRAME_WIDTH = 640                     # px of the screenshots: a 243 pt column at about 190 dpi (frames.py WIDTHS)
MAX_EVIDENCE_PER_FINDING = 2          # evidence rows drawn under a finding (all of them are in the timestamp list)
MAX_SCREENSHOTS = 16                  # screenshots in one PDF: about 40-60 kB each at 640 px
PRODUCER = 'PitchIQ coaching report (soccer/coaching/pdf.py)'

# Report colours for print (light pages; the Studio's dark tokens are for the screen).
INK = '#18211d'
INK_2 = '#4a5a51'
MUTED = '#6f7f76'
RULE = '#d5ded8'
BRAND = '#19b96b'
BRAND_DARK = '#117b49'
TINT = '#eef7f1'
ZEBRA = '#f5f8f6'
HEAD = '#e6efe9'
LABEL_STYLES = {                      # finding label -> (text colour, background)
    'Observed strength': ('#117b49', '#e1f4e9'),
    'Review opportunity': ('#3f6690', '#e6eef7'),
    'Potential improvement': ('#8a5f12', '#faf0dc'),
    'High-confidence concern': ('#9c3a3a', '#f7e3e3'),
}

# Helvetica and Helvetica-Bold advance widths (1/1000 em) for WinAnsi codes 32..255 (0: not a WinAnsi character).
_WIDTHS_REGULAR = (
    278, 278, 355, 556, 556, 889, 667, 191, 333, 333, 389, 584, 278, 333, 278, 278,
    556, 556, 556, 556, 556, 556, 556, 556, 556, 556, 278, 278, 584, 584, 584, 556,
    1015, 667, 667, 722, 722, 667, 611, 778, 722, 278, 500, 667, 556, 833, 722, 778,
    667, 778, 722, 667, 611, 722, 667, 944, 667, 667, 611, 278, 278, 278, 469, 556,
    333, 556, 556, 500, 556, 556, 278, 556, 556, 222, 222, 500, 222, 833, 556, 556,
    556, 556, 333, 500, 278, 556, 500, 722, 500, 500, 500, 334, 260, 334, 584, 0,
    556, 0, 222, 556, 333, 1000, 556, 556, 333, 1000, 667, 333, 1000, 0, 611, 0,
    0, 222, 222, 333, 333, 350, 556, 1000, 333, 1000, 500, 333, 944, 0, 500, 667,
    278, 333, 556, 556, 556, 556, 260, 556, 333, 737, 370, 556, 584, 333, 737, 333,
    400, 584, 333, 333, 333, 556, 537, 278, 333, 333, 365, 556, 834, 834, 834, 611,
    667, 667, 667, 667, 667, 667, 1000, 722, 667, 667, 667, 667, 278, 278, 278, 278,
    722, 722, 778, 778, 778, 778, 778, 584, 778, 722, 722, 722, 722, 667, 667, 611,
    556, 556, 556, 556, 556, 556, 889, 500, 556, 556, 556, 556, 278, 278, 278, 278,
    556, 556, 556, 556, 556, 556, 556, 584, 611, 556, 556, 556, 556, 500, 556, 500)
_WIDTHS_BOLD = (
    278, 333, 474, 556, 556, 889, 722, 238, 333, 333, 389, 584, 278, 333, 278, 278,
    556, 556, 556, 556, 556, 556, 556, 556, 556, 556, 333, 333, 584, 584, 584, 611,
    975, 722, 722, 722, 722, 667, 611, 778, 722, 278, 556, 722, 611, 833, 722, 778,
    667, 778, 722, 667, 611, 722, 667, 944, 667, 667, 611, 333, 278, 333, 584, 556,
    333, 556, 611, 556, 611, 556, 333, 611, 611, 278, 278, 556, 278, 889, 611, 611,
    611, 611, 389, 556, 333, 611, 556, 778, 556, 556, 500, 389, 280, 389, 584, 0,
    556, 0, 278, 556, 500, 1000, 556, 556, 333, 1000, 667, 333, 1000, 0, 611, 0,
    0, 278, 278, 500, 500, 350, 556, 1000, 333, 1000, 556, 333, 944, 0, 500, 667,
    278, 333, 556, 556, 556, 556, 280, 556, 333, 737, 370, 556, 584, 333, 737, 333,
    400, 584, 333, 333, 333, 611, 556, 278, 333, 333, 365, 556, 834, 834, 834, 611,
    722, 722, 722, 722, 722, 722, 1000, 722, 667, 667, 667, 667, 278, 278, 278, 278,
    722, 722, 778, 778, 778, 778, 778, 584, 778, 722, 722, 722, 722, 667, 667, 611,
    556, 556, 556, 556, 556, 556, 889, 556, 556, 556, 556, 556, 278, 278, 278, 278,
    611, 611, 611, 611, 611, 611, 611, 584, 611, 611, 611, 611, 611, 556, 611, 556)
FONTS = {'regular': ('F1', 'Helvetica', _WIDTHS_REGULAR), 'bold': ('F2', 'Helvetica-Bold', _WIDTHS_BOLD),
         'oblique': ('F3', 'Helvetica-Oblique', _WIDTHS_REGULAR)}


class WinAnsiError(ValueError):
    """Text the base-14 fonts cannot show (outside WinAnsi / cp1252)."""


# ---------------------------------------------------------------------------------------------------------------
# Text

def _char(ch):
    """The WinAnsi byte of one character, or None."""
    if ch in '\t\n\r':
        return 32
    o = ord(ch)
    if o < 32 or o == 127:
        return None
    try:
        b = ch.encode('cp1252')
    except UnicodeEncodeError:
        return None
    return b[0] if _WIDTHS_REGULAR[b[0]-32] else None


def winansi_problems(text):
    """The characters of text that the PDF cannot write (sorted, unique); [] when all of it is WinAnsi."""
    return sorted({ch for ch in str(text) if _char(ch) is None})


def encode_text(text, strict=False):
    """(WinAnsi bytes, number of replaced characters). Tabs and line breaks become spaces; any other character
    outside WinAnsi raises WinAnsiError with strict, else becomes '?'."""
    out = bytearray()
    bad = 0
    for ch in str(text):
        b = _char(ch)
        if b is None:
            if strict:
                raise WinAnsiError(f'character {ch!r} (U+{ord(ch):04X}) is not in WinAnsi')
            b = 63
            bad += 1
        out.append(b)
    return bytes(out), bad


def text_width(text, size, font='regular'):
    """Width in points of text set in the given face ('regular', 'bold', 'oblique') at size."""
    table = FONTS.get(font, FONTS['regular'])[2]
    data, _ = encode_text(text)
    return sum(table[b-32] for b in data)*size/1000.0


def _advance(ch, table):
    """The advance width of one character in 1/1000 em, as encode_text writes it (an unknown character is '?')."""
    b = _char(ch)
    return table[(63 if b is None else b)-32]


def wrap(text, size, width, font='regular'):
    """Lines of at most width points; a word wider than a line is broken by characters. One pass over the text:
    widths are additive (base-14 fonts without kerning), so the line width is kept as a running sum of integer font
    units (exact, no float drift) instead of re-measuring the line for every word and every character. The previous
    version re-measured word[:k] for each k: a 2000-character word (a pasted URL in a coach note, MAX_NOTE 2000)
    took 21.3 s ('a') and 34.9 s ('W') in one call on 2026-10-18, and a PDF with such a note 34 s."""
    table = FONTS.get(font, FONTS['regular'])[2]
    limit = float(width)*1000.0/float(size) if size else float('inf')     # the line width in font units
    space = table[0]
    lines, line, used = [], [], 0
    for word in str(text).split():
        units = [_advance(ch, table) for ch in word]
        w = sum(units)
        if line and used+space+w <= limit:
            line.append(word)
            used += space+w
            continue
        if not line and w <= limit:
            line, used = [word], w
            continue
        if line:
            lines.append(' '.join(line))
            line, used = [], 0
        start, run = 0, 0
        for k, u in enumerate(units):                  # a word wider than the line: break it by characters
            if run+u > limit and k > start:
                lines.append(word[start:k])
                start, run = k, 0
            run += u
        line, used = [word[start:]], run
    if line:
        lines.append(' '.join(line))
    return lines


def _string(data):
    return b'(' + data.replace(b'\\', b'\\\\').replace(b'(', b'\\(').replace(b')', b'\\)') + b')'


def _num(v):
    """A PDF number with at most 3 decimals, never an exponent or '-0': the same value always gives the same bytes."""
    s = f'{float(v):.3f}'.rstrip('0').rstrip('.')
    return '0' if s in ('', '-0') else s


def _rgb(colour):
    c = colour.lstrip('#')
    return ' '.join(_num(int(c[i:i+2], 16)/255.0) for i in (0, 2, 4))


_URI = re.compile(r'^https?://[A-Za-z0-9.\-\[\]:]+(/[!-~]*)?$')


def safe_uri(uri):
    """uri when it is a plain http(s) URL of printable ASCII without spaces, else None (no link is written)."""
    if not isinstance(uri, str) or len(uri) > 2000 or not _URI.match(uri):
        return None
    return uri


def jpeg_info(data):
    """(width, height, components) of a baseline or progressive JPEG; ValueError when it is not one."""
    if not isinstance(data, (bytes, bytearray)) or data[:2] != b'\xff\xd8':
        raise ValueError('not a JPEG')
    i = 2
    while i+4 <= len(data):
        if data[i] != 0xFF:
            raise ValueError('JPEG marker expected')
        marker = data[i+1]
        if marker == 0xFF:                       # fill byte
            i += 1
            continue
        if marker in (0x01, *range(0xD0, 0xD8)):     # markers without a length
            i += 2
            continue
        length = struct.unpack('>H', data[i+2:i+4])[0]
        if marker in (0xC0, 0xC1, 0xC2):
            h, w = struct.unpack('>HH', data[i+5:i+9])
            comps = data[i+9]
            if comps not in (1, 3) or not w or not h:
                raise ValueError('unsupported JPEG (only greyscale or YCbCr colour)')
            return w, h, comps
        if marker == 0xDA:
            break
        i += 2+length
    raise ValueError('JPEG without a frame header')


# ---------------------------------------------------------------------------------------------------------------
# The writer: pages hold the drawing operations of soccer/coaching/render.py (plus 'image' and 'link')

class Document:
    """A PDF under construction. Pages are lists of drawing operations (points, origin top left); build() writes
    PDF 1.4 bytes. Images are deduplicated by content; translucency uses one ExtGState per alpha value."""

    def __init__(self, page_size=A4, title=None, subject=None, author=None, created=None, strict=False):
        self.size = (float(page_size[0]), float(page_size[1]))
        self.pages = []
        self.anchors = {}
        self.bookmarks = []
        self.info = {'Title': title, 'Subject': subject, 'Author': author, 'Creator': 'PitchIQ'}
        self.created = created
        self.strict = strict
        self.stats = {}

    def add_page(self):
        self.pages.append([])
        return len(self.pages)-1

    def add(self, page, ops):
        self.pages[page].extend(ops)

    def anchor(self, name, page, y):
        """A named destination (for internal links) at height y of a page; the first one of a name counts."""
        self.anchors.setdefault(name, (page, float(y)))

    def bookmark(self, title, name):
        """An outline entry (the viewer's sidebar) pointing at an anchor."""
        self.bookmarks.append((title, name))

    # -- content streams ------------------------------------------------------------------------------------
    def _text(self, op, res):
        font = op.get('font', 'regular')
        name = FONTS.get(font, FONTS['regular'])[0]
        size = float(op.get('size', 9))
        data, bad = encode_text(op.get('text', ''), self.strict)
        res['replaced'] += bad
        w = text_width(op.get('text', ''), size, font)
        x = float(op['x'])-(w/2 if op.get('align') == 'center' else w if op.get('align') == 'right' else 0.0)
        y = self.size[1]-float(op['y'])
        parts = ['q']
        alpha = op.get('alpha', 1.0)
        if alpha is not None and alpha < 1:
            parts.append(f'/{res["alpha"](alpha)} gs')
        if op.get('halo'):
            parts.append(f'{_rgb(op["halo"])} rg {_num(x-1.2)} {_num(y-size*.22-.4)} {_num(w+2.4)} '
                         f'{_num(size*1.02+.8)} re f')
        parts.append(f'BT /{name} {_num(size)} Tf {_rgb(op.get("colour") or INK)} rg 1 0 0 1 {_num(x)} {_num(y)} Tm')
        return ' '.join(parts).encode('ascii')+b' '+_string(data)+b' Tj ET Q'

    def _shape(self, op, res):
        H = self.size[1]
        parts = ['q']
        alpha = op.get('alpha', 1.0)
        if alpha is not None and alpha < 1:
            parts.append(f'/{res["alpha"](alpha)} gs')
        fill, stroke = op.get('fill'), op.get('stroke')
        if stroke:
            parts.append(f'{_rgb(stroke)} RG {_num(op.get("width") or .8)} w 1 J 1 j')
            if op.get('dash'):
                parts.append('[' + ' '.join(_num(v) for v in op['dash']) + '] 0 d')
        if fill:
            parts.append(f'{_rgb(fill)} rg')
        kind = op['op']
        closed = True
        if kind == 'rect':
            parts.append(f'{_num(op["x"])} {_num(H-op["y"]-op["h"])} {_num(op["w"])} {_num(op["h"])} re')
        elif kind == 'circle':
            cx, cy, r = float(op['cx']), H-float(op['cy']), float(op['r'])
            k = .5523*r
            parts.append(f'{_num(cx+r)} {_num(cy)} m '
                         f'{_num(cx+r)} {_num(cy+k)} {_num(cx+k)} {_num(cy+r)} {_num(cx)} {_num(cy+r)} c '
                         f'{_num(cx-k)} {_num(cy+r)} {_num(cx-r)} {_num(cy+k)} {_num(cx-r)} {_num(cy)} c '
                         f'{_num(cx-r)} {_num(cy-k)} {_num(cx-k)} {_num(cy-r)} {_num(cx)} {_num(cy-r)} c '
                         f'{_num(cx+k)} {_num(cy-r)} {_num(cx+r)} {_num(cy-k)} {_num(cx+r)} {_num(cy)} c h')
        else:
            pts = op.get('points') or []
            if len(pts) < 2:
                return b''
            seg = [f'{_num(pts[0][0])} {_num(H-pts[0][1])} m']
            seg += [f'{_num(x)} {_num(H-y)} l' for x, y in pts[1:]]
            closed = bool(op.get('closed'))
            if closed:
                seg.append('h')
            parts.append(' '.join(seg))
        if fill and stroke:
            parts.append('B')
        elif fill:
            parts.append('f')
        elif stroke:
            parts.append('S')
        else:
            parts.append('n')
        parts.append('Q')
        return ' '.join(parts).encode('ascii')

    def _image(self, op, res):
        name = res['image'](op['jpeg'])
        H = self.size[1]
        return (f'q {_num(op["w"])} 0 0 {_num(op["h"])} {_num(op["x"])} {_num(H-op["y"]-op["h"])} cm /{name} Do Q'
                .encode('ascii'))

    def build(self):
        """The PDF file's bytes."""
        W, H = self.size
        objects = {}                 # id -> (dictionary bytes, stream bytes or None)
        order = []

        def new(body=None, stream=None):
            i = len(order)+1
            order.append(i)
            objects[i] = (body, stream)
            return i

        catalog, pages_id = new(), new()
        font_ids = {}
        for key in ('regular', 'bold', 'oblique'):
            name, base, _ = FONTS[key]
            font_ids[name] = new(f'<< /Type /Font /Subtype /Type1 /BaseFont /{base} /Encoding /WinAnsiEncoding >>'
                                 .encode('ascii'))
        page_ids = [new() for _ in self.pages]
        images, alphas = {}, {}
        stats = {'images': 0, 'imageBytes': 0, 'links': 0, 'internalLinks': 0, 'droppedLinks': 0, 'replaced': 0}

        def image(data):
            key = hashlib.sha1(data).hexdigest()
            if key not in images:
                w, h, comps = jpeg_info(data)
                cs = '/DeviceRGB' if comps == 3 else '/DeviceGray'
                ref = new(f'<< /Type /XObject /Subtype /Image /Width {w} /Height {h} /ColorSpace {cs} '
                          f'/BitsPerComponent 8 /Filter /DCTDecode /Length {len(data)} >>'.encode('ascii'), bytes(data))
                images[key] = (f'Im{len(images)+1}', ref)
                stats['images'] += 1
                stats['imageBytes'] += len(data)
            return images[key][0]

        def alpha(a):
            key = _num(min(1.0, max(0.0, float(a))))
            if key not in alphas:
                ref = new(f'<< /Type /ExtGState /CA {key} /ca {key} >>'.encode('ascii'))
                alphas[key] = (f'GA{key.replace(".", "_")}', ref)
            return alphas[key][0]

        def dest(name):
            if name not in self.anchors:
                return None
            page, y = self.anchors[name]
            return f'[{page_ids[page]} 0 R /XYZ 0 {_num(H-y+6)} null]'

        for index, ops in enumerate(self.pages):
            used_images, used_alphas = set(), set()
            res = {'replaced': 0,
                   'image': lambda d: (used_images.add(image(d)) or image(d)),
                   'alpha': lambda a: (used_alphas.add(alpha(a)) or alpha(a))}
            chunks, links, depth = [], [], 0
            for op in ops:
                kind = op.get('op')
                if kind == 'clip':
                    chunks.append(f'q {_num(op["x"])} {_num(H-op["y"]-op["h"])} {_num(op["w"])} {_num(op["h"])} re W n'
                                  .encode('ascii'))
                    depth += 1
                elif kind == 'unclip':
                    if depth:
                        chunks.append(b'Q')
                        depth -= 1
                elif kind == 'link':
                    links.append(op)
                elif kind == 'text':
                    chunks.append(self._text(op, res))
                elif kind in ('rect', 'circle', 'path'):
                    chunks.append(self._shape(op, res))
                elif kind == 'image':
                    chunks.append(self._image(op, res))
                else:
                    raise ValueError(f'unknown drawing operation {kind!r}')
            chunks.extend([b'Q']*depth)
            stats['replaced'] += res['replaced']
            content = zlib.compress(b'\n'.join(c for c in chunks if c), 6)
            content_id = new(f'<< /Length {len(content)} /Filter /FlateDecode >>'.encode('ascii'), content)
            annots = []
            for link in links:
                rect = (f'[{_num(link["x"])} {_num(H-link["y"]-link["h"])} {_num(link["x"]+link["w"])} '
                        f'{_num(H-link["y"])}]')
                if link.get('uri'):
                    uri = safe_uri(link['uri'])
                    if uri is None:
                        stats['droppedLinks'] += 1
                        continue
                    action = b'/A << /S /URI /URI ' + _string(uri.encode('ascii')) + b' >>'
                    stats['links'] += 1
                else:
                    d = dest(link.get('dest'))
                    if d is None:
                        stats['droppedLinks'] += 1
                        continue
                    action = f'/Dest {d}'.encode('ascii')
                    stats['internalLinks'] += 1
                annots.append(new(f'<< /Type /Annot /Subtype /Link /Rect {rect} /Border [0 0 0] '.encode('ascii')
                                  + action + b' >>'))
            fonts = ' '.join(f'/{n} {r} 0 R' for n, r in font_ids.items())
            xobj = ' '.join(f'/{n} {r} 0 R' for n, r in sorted(v for v in images.values() if v[0] in used_images))
            gs = ' '.join(f'/{n} {r} 0 R' for n, r in sorted(v for v in alphas.values() if v[0] in used_alphas))
            body = (f'<< /Type /Page /Parent {pages_id} 0 R /MediaBox [0 0 {_num(W)} {_num(H)}] '
                    f'/Contents {content_id} 0 R /Resources << /ProcSet [/PDF /Text /ImageC /ImageB] '
                    f'/Font << {fonts} >>' + (f' /XObject << {xobj} >>' if xobj else '')
                    + (f' /ExtGState << {gs} >>' if gs else '') + ' >>'
                    + (' /Annots [' + ' '.join(f'{a} 0 R' for a in annots) + ']' if annots else '') + ' >>')
            objects[page_ids[index]] = (body.encode('ascii'), None)
        objects[pages_id] = (f'<< /Type /Pages /Kids [{" ".join(f"{p} 0 R" for p in page_ids)}] '
                             f'/Count {len(page_ids)} >>'.encode('ascii'), None)
        outline = ''
        marks = [(t, dest(n)) for t, n in self.bookmarks if dest(n)]
        if marks:
            root = new()
            ids = [new() for _ in marks]
            for k, ((title, d), oid) in enumerate(zip(marks, ids)):
                data, bad = encode_text(title, self.strict)
                stats['replaced'] += bad
                body = b'<< /Title ' + _string(data) + f' /Parent {root} 0 R /Dest {d}'.encode('ascii')
                if k:
                    body += f' /Prev {ids[k-1]} 0 R'.encode('ascii')
                if k+1 < len(ids):
                    body += f' /Next {ids[k+1]} 0 R'.encode('ascii')
                objects[oid] = (body+b' >>', None)
            objects[root] = (f'<< /Type /Outlines /First {ids[0]} 0 R /Last {ids[-1]} 0 R /Count {len(ids)} >>'
                             .encode('ascii'), None)
            outline = f' /Outlines {root} 0 R'
        objects[catalog] = (f'<< /Type /Catalog /Pages {pages_id} 0 R{outline} >>'.encode('ascii'), None)
        info = b'<< /Producer ' + _string(PRODUCER.encode('ascii'))
        for key, value in self.info.items():
            if value:
                data, bad = encode_text(value, self.strict)
                stats['replaced'] += bad
                info += f' /{key} '.encode('ascii')+_string(data)
        stamp = _pdf_date(self.created)
        if stamp:
            info += b' /CreationDate ' + _string(stamp.encode('ascii'))
        info_id = new(info+b' >>')
        out = bytearray(b'%PDF-1.4\n%\xe2\xe3\xcf\xd3\n')
        offsets = {}
        for i in order:
            body, stream = objects[i]
            offsets[i] = len(out)
            out += f'{i} 0 obj\n'.encode('ascii')+body
            if stream is not None:
                out += b'\nstream\n'+stream+b'\nendstream'
            out += b'\nendobj\n'
        xref = len(out)
        out += f'xref\n0 {len(order)+1}\n0000000000 65535 f \n'.encode('ascii')
        for i in order:
            out += f'{offsets[i]:010d} 00000 n \n'.encode('ascii')
        digest = hashlib.md5(bytes(out)).hexdigest()
        out += (f'trailer\n<< /Size {len(order)+1} /Root {catalog} 0 R /Info {info_id} 0 R '
                f'/ID [<{digest}> <{digest}>] >>\nstartxref\n{xref}\n%%EOF\n').encode('ascii')
        stats['pages'] = len(self.pages)
        stats['bytes'] = len(out)
        self.stats = stats
        return bytes(out)


def _pdf_date(created):
    """'D:YYYYMMDDHHmmSS' from an ISO date-time string ('2026-10-18T19:03:12'), else None."""
    if not isinstance(created, str):
        return None
    m = re.match(r'^(\d{4})-(\d{2})-(\d{2})(?:[T ](\d{2}):(\d{2})(?::(\d{2}))?)?', created)
    if not m:
        return None
    return 'D:' + ''.join(g or '00' for g in m.groups())


# ---------------------------------------------------------------------------------------------------------------
# Report layout

SECTIONS = (('summary', 'Summary'), ('strengths', 'Observed strengths'), ('improvements', 'Potential improvements'),
            ('reviews', 'Review opportunities'), ('breakdown', 'Tactical breakdown'), ('statistics', 'Key statistics'),
            ('priorities', 'Priorities'), ('training', 'Training recommendations'),
            ('confidence', 'Coverage and confidence'), ('evidence', 'Evidence timestamps'),
            ('feedback', 'Expert feedback'))
# The report's sections in the order the Studio's Coaching tab shows them (contract sections 7 and 9; coaching.js
# renderMain): the PDF follows the same order, so a coach moving between the page and the printout finds every part in
# the same place (the first PDFs put the priorities before the breakdown, the page after it). report.py may publish
# this order (section_order() reads it); this tuple is the fallback and equals the page's order on 2026-10-18.
PAGE_ORDER = ('overview', 'summary', 'strengths', 'improvements', 'reviews', 'breakdown', 'priorities', 'training',
              'confidence')
# Sections only the PDF has (the page shows them in its side column and on each finding): each follows the shared
# section named here, so the shared ones keep the page's order.
PDF_ONLY = (('statistics', 'breakdown'), ('evidence', 'confidence'), ('feedback', 'evidence'))
SECTION_ALIASES = {'coverage': 'confidence', 'limitations': 'confidence', 'timestamps': 'evidence',
                   'cover': 'overview'}
_ORDER_NAMES = ('SECTIONS', 'REPORT_SECTIONS', 'PAGE_SECTIONS', 'DOCUMENT_SECTIONS', 'SECTION_ORDER', 'ORDER')


def section_order(order=None):
    """The report's sections in print order: the shared order (given, else report.py's published order, else
    PAGE_ORDER) with the PDF-only sections placed after their anchors. An order is accepted only when it names the
    main sections (summary and strengths): report.SECTIONS currently lists the breakdown's subsections instead."""
    def keys(seq):
        out = []
        for item in seq or ():
            k = item[0] if isinstance(item, (list, tuple)) and item else item
            k = SECTION_ALIASES.get(k, k) if isinstance(k, str) else None
            if k and k not in out:
                out.append(k)
        return out

    shared = keys(order) if order is not None else []
    if not {'summary', 'strengths'} <= set(shared):
        shared = []
        try:
            from . import report as _report
        except Exception:                                          # the PDF never depends on the assembler loading
            _report = None
        for name in _ORDER_NAMES:
            cand = keys(getattr(_report, name, None) if _report else None)
            if {'summary', 'strengths'} <= set(cand):
                shared = cand
                break
    shared = [k for k in (shared or list(PAGE_ORDER)) if k in PAGE_ORDER or k in dict(PDF_ONLY)]
    shared += [k for k in PAGE_ORDER if k not in shared]
    for key, after in PDF_ONLY:
        if key not in shared:
            shared.insert(shared.index(after)+1 if after in shared else len(shared), key)
    return tuple(shared)


def pdf_filename(video, run, team, context=None, key=None, level=None, t0=None, t1=None):
    """The download name of a report PDF: pitchiq-coaching-<video>-<run>-<team>-<level>[-<t0>-<t1>s].pdf. The level is
    part of the name because the levels of one report are different documents (the first PDFs of a development and an
    advanced report downloaded under the same name and replaced each other): a v1 context name, or 'level1' to
    'level7' for a coaching v2 playing level (store.pdf_filename_v2's spelling). A custom range adds its whole
    seconds (critic R4: two ranges of one run downloaded under one name). video: the upload's name (its stem is
    used), key: the fallback when there is none; every part is
    reduced to [A-Za-z0-9._-]. The server's PDF route uses it through store.pdf_filename."""
    def part(v, n=48):
        return re.sub(r'[^A-Za-z0-9._-]+', '-', str(v or '')).strip('-.')[:n].strip('-.')
    name = part(Path(str(video)).stem if video else '') or part(key, 16) or 'video'
    tag = context if context in CONTEXT_NAMES else None
    if tag is None and isinstance(level, int) and not isinstance(level, bool) and 1 <= level <= 7:
        tag = f'level{level}'
    span = (f'{int(math.floor(t0))}-{int(math.ceil(t1))}s' if _is_num(t0) and _is_num(t1) and t1 > t0 >= 0
            else None)
    return '-'.join(p for p in ('pitchiq-coaching', name, part(run, 64), part(team, 4), tag, span) if p) + '.pdf'


def report_filename(report):
    """pdf_filename() of a report's own fields (video name or key, run, team, v1 context level or v2 playing level,
    and the range when it is not the whole analysed segment)."""
    r = report if isinstance(report, dict) else {}
    video = r.get('video') if isinstance(r.get('video'), dict) else {}
    if is_v2(r):
        p = r.get('profile') if isinstance(r.get('profile'), dict) else {}
        rng = r.get('range') if isinstance(r.get('range'), dict) else {}
        whole = rng.get('whole') is not False
        return pdf_filename(video.get('name'), r.get('run'), r.get('team'), None, video.get('key'), p.get('level'),
                            None if whole else rng.get('t0'), None if whole else rng.get('t1'))
    return pdf_filename(video.get('name'), r.get('run'), r.get('team'), r.get('context'), video.get('key'))
BREAKDOWN = (('defensiveOrganization', 'Defensive organisation'), ('spacing', 'Spacing'), ('formation', 'Formation'),
             ('attackingStructure', 'Attacking structure'), ('movementSupport', 'Movement and support'))
CLASS_LABELS = {'strength': 'Observed strength', 'review': 'Review opportunity', 'neutral': 'Descriptive',
                'insufficient': 'Not assessed'}
SEVERITY_LABELS = {'observed_strength': 'Observed strength', 'review_opportunity': 'Review opportunity',
                   'potential_improvement': 'Potential improvement',
                   'high_confidence_concern': 'High-confidence concern'}
VERDICTS = {'correct': 'Correct', 'partial': 'Partially correct', 'incorrect': 'Incorrect',
            'insufficient': 'Not enough evidence'}
CONTEXT_NAMES = {'development': 'Development', 'competitive': 'Competitive', 'advanced': 'Advanced'}
ASPECT_WORDS = {'tracking': 'player tracking', 'calibration': 'pitch registration',      # language.ASPECTS
                'identity': 'player identification', 'ball': 'ball position'}
METRICS = {                           # the coaching series' metric keys (contract section 2): name and unit
    'width': ('team width', 'm'), 'depth': ('team depth', 'm'), 'hull': ('team area', 'm2'),
    'spread': ('team spread', 'm'), 'gapDM': ('defence-midfield gap', 'm'), 'gapMF': ('midfield-forward gap', 'm'),
    'lineStd': ('back-line unevenness (std)', 'm'), 'lineHeight': ('back-line height', 'm'),
    'lineSize': ('players in the back line', None), 'nearestTeammate': ('nearest teammate', 'm'),
    'lagDM': ('defence-midfield lateral lag', 's'), 'corrDM': ('defence-midfield lateral correlation', 'ratio')}
EMPTY = {
    'strengths': 'No observed strength met the evidence rules in this range.',
    'improvements': 'No potential improvement met the evidence rules in this range.',
    'reviews': 'No review opportunity was found in this range.',
    'priorities': 'No priority is set: no finding was strong enough to lead the training plan.',
    'training': 'No training recommendation: no finding in this report calls for one.',
    'evidence': 'This report holds no evidence moments.',
}
BUILD_ON = re.compile(r'To build on it:\s*')   # how language.py opens a strength's whatToImprove (its title here)
DRILL_FIELDS = (                      # drill record fields shown, in order (soccer/coaching/drills.json, drill_ref)
    ('purpose', 'Purpose', None), ('objective', 'Objective', None), ('tacticalIssue', 'When to use it', None),
    ('description', 'Description', None), ('players', 'Players', 'players'),
    ('durationMinutes', 'Duration', 'minutes'), ('duration', 'Duration', None), ('setup', 'Set-up', None),
    ('organisation', 'Organisation', None), ('area', 'Area', None), ('equipment', 'Equipment', None),
    ('instructions', 'How it runs', None), ('steps', 'How it runs', None), ('coachingPoints', 'Coaching points', None),
    ('progressions', 'Progressions', None), ('progression', 'Progression', None), ('regressions', 'Regressions', None),
    ('variation', 'At this level', None),
    ('successCriteria', 'Success looks like', None), ('safety', 'Safety', None), ('source', 'Format', None))


# report.py's line per measurement on a footage with a few players per team in view (Veo)
BARE_LINE = re.compile(r'^(.+): not assessed on this footage\.$')


def join_bare(lines):
    """The bare "X: not assessed on this footage." lines of a footage with a few players per team in view (report.py
    writes one per measurement under its data-quality block: ten on the Veo clip) as one line, where the first of them
    stood; every other line unchanged."""
    bare = [m.group(1) for m in (BARE_LINE.match(x) for x in lines) if m]
    if len(bare) < 2:
        return list(lines)
    out, joined = [], False
    for x in lines:
        if not BARE_LINE.match(x):
            out.append(x)
        elif not joined:
            out.append('Not assessed on this footage: ' + ', '.join(
                b if i == 0 else b[:1].lower()+b[1:] for i, b in enumerate(bare)) + '.')
            joined = True
    return out


def clock(t):
    """Absolute source time as mm:ss.ss (the Studio's transport and corrections.timestamp format). Rounded to the
    hundredth before the minutes are split (critic N8: 59.999 s printed as '00:60.00'; now '01:00.00', as the
    page's coaching clock and language.fmt_time)."""
    if not isinstance(t, (int, float)) or isinstance(t, bool) or not math.isfinite(t):
        return 'n/a'
    sign = '-' if t < 0 else ''
    c = int(math.floor(abs(float(t))*100+.5))           # hundredths, half up (JavaScript's Math.round)
    m, rest = divmod(c, 6000)
    return f'{sign}{m:02d}:{rest//100:02d}.{rest % 100:02d}'


def offset(t):
    """Segment-relative time as +mm:ss.ss."""
    s = clock(t)
    return s if s == 'n/a' or s.startswith('-') else '+'+s


def _is_num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def value_text(v, unit=None, estimated=False):
    """A measured value with its unit (the units of report.py's metric rows): metres to 0.1 (marked est. when the
    pitch size is assumed), sq m to 1, seconds to 0.1, a share as a percentage, a ratio (a correlation) to 0.01, a
    count as a whole number, a rate per minute to 0.01, a label as it is."""
    if v is None:
        return 'n/a'
    if isinstance(v, bool):
        return 'yes' if v else 'no'
    if not _is_num(v):
        return str(v)
    u = (unit or '').strip()
    est = ' (est.)' if estimated else ''
    if u == 'share':
        return f'{v*100:.1f}%'
    if u == '%':
        return f'{v:.1f}%'
    if u == 'm':
        return f'{v:.1f} m{est}'
    if u == 'm2':
        return f'{v:.0f} sq m{est}'
    if u == 'ratio':
        return f'{v:.2f}'
    if u == 's':
        return f'{v:.1f} s'
    if u == 'count':
        return f'{int(round(v))}'
    if u in ('per_min', 'per min', '/min', 'perMinute'):
        return f'{v:.2f} per min'
    if float(v).is_integer():
        return f'{int(v)}' + (f' {u}' if u else '')
    return f'{v:.2f}' + (f' {u}' if u else '')


def _context(code):
    """A reference context code in words: 'third:middle' -> 'with the team's centre in the middle third'; '' for a
    code without words (a code is never printed raw: the fact check found '(window:20)' in the first PDFs)."""
    code = str(code)
    kind, _, value = code.partition(':')
    if kind == 'third' and value in ('own', 'middle', 'final'):
        return "with the team's centre in " + ('its own' if value == 'own' else 'the ' + value) + ' third'
    if kind == 'backline' and value.isdigit():
        return f'with a back line of {value}'
    if kind == 'window' and value.isdigit():
        return f'over {value} s windows'
    if code == 'all':
        return 'over all measured play'
    return ''


def compared(observation):
    """True when the observation was actually compared with the professional reference: its baseline says available
    (detectors.py). Formation readings, ball location and lateral shift without a usable stretch carry a baseline
    record with available False (BC60 A/B, BC180 A and AM B formation findings on 2026-10-18), and the first PDFs still
    printed 'Compared with the professional reference' under them: a claim the report has no data for."""
    base = (observation or {}).get('baseline') if isinstance(observation, dict) else None
    return isinstance(base, dict) and base.get('available') is True and bool(base.get('reference'))


def _humanize(key):
    s = re.sub(r'(?<=[a-z0-9])([A-Z])', r' \1', str(key)).replace('_', ' ').strip()
    return s[:1].upper()+s[1:].lower() if s else ''


def _seconds(s):
    if not _is_num(s):
        return 'n/a'
    return f'{s:.1f} s' if s < 120 else f'{s/60:.1f} min'


def _translate(ops, dx, dy):
    out = []
    for op in ops:
        op = dict(op)
        if 'points' in op:
            op['points'] = [[p[0]+dx, p[1]+dy] for p in op['points']]
        if 'cx' in op:
            op['cx'] += dx
            op['cy'] += dy
        if 'x' in op and op['op'] != 'circle':
            op['x'] += dx
            op['y'] += dy
        out.append(op)
    return out


class _Layout:
    """A flowing column on A4 pages: headings, wrapped paragraphs, bullets, tables and figures, with page breaks."""

    def __init__(self, doc, header):
        self.doc = doc
        self.W, self.H = doc.size
        self.left, self.top = MARGINS[0], MARGINS[1]
        self.right, self.bottom = self.W-MARGINS[2], self.H-MARGINS[3]
        self.width = self.right-self.left
        self.header = header
        self.page = -1
        self.y = 0.0
        self.new_page()

    # -- primitives -------------------------------------------------------------------------------------------
    def new_page(self):
        from . import render
        self.page = self.doc.add_page()
        self.y = self.top
        self.add([render.rect(0, 0, self.W, 5, fill=BRAND)])
        if self.page > 0 and self.header:
            self.add([render.text(self.left, 30, 'PITCHIQ COACHING REPORT', 6.5, BRAND_DARK, 'bold'),
                      render.text(self.right, 30, self.header, 6.5, MUTED, 'regular', 'right')])

    def add(self, ops):
        self.doc.add(self.page, ops)

    def ensure(self, h):
        if self.y+h > self.bottom and self.y > self.top+1:
            self.new_page()

    def space(self, h):
        self.y += h

    def anchor(self, name):
        if name:
            self.doc.anchor(name, self.page, self.y)

    def text_line(self, x, s, size=9.5, colour=INK, font='regular', align='left', leading=None):
        """One line whose box starts at the cursor; returns the box height (the leading)."""
        from . import render
        leading = leading or size*1.38
        base = self.y+(leading+size*.7)/2
        self.add([render.text(x, base, s, size, colour, font, align)])
        return leading

    def para(self, s, size=9.5, colour=INK, font='regular', indent=0.0, after=4.0, width=None, link=None):
        """A wrapped paragraph; returns nothing. link: {'uri': ...} or {'dest': ...} over each line."""
        if s is None or str(s).strip() == '':
            return
        width = (width or self.width)-indent
        leading = size*1.38
        for line in wrap(s, size, width, font):
            self.ensure(leading)
            self.text_line(self.left+indent, line, size, colour, font, leading=leading)
            if link:
                w = text_width(line, size, font)
                self.add([dict({'op': 'link', 'x': self.left+indent, 'y': self.y, 'w': w, 'h': leading}, **link)])
            self.y += leading
        self.y += after

    def bullets(self, items, size=9.0, colour=INK, indent=0.0, after=4.0, marker='-'):
        leading = size*1.38
        for item in items:
            if isinstance(item, dict):
                s, link = item.get('text'), item.get('link')
            else:
                s, link = item, None
            if s is None or str(s).strip() == '':
                continue
            lines = wrap(s, size, self.width-indent-10)
            for k, line in enumerate(lines):
                self.ensure(leading)
                if k == 0:
                    self.text_line(self.left+indent, marker, size, BRAND_DARK, 'bold', leading=leading)
                self.text_line(self.left+indent+10, line, size, BRAND_DARK if link else colour, font='regular',
                               leading=leading)
                if link:
                    self.add([dict({'op': 'link', 'x': self.left+indent+10, 'y': self.y,
                                    'w': text_width(line, size), 'h': leading}, **link)])
                self.y += leading
            self.y += 1.5
        self.y += after

    def heading(self, s, level=1, anchor=None, bookmark=False, keep=0.0):
        """keep: the height of what must follow the heading on the same page (critic N5: 'Observed strengths' stood
        alone at the foot of BC60 A page 1 because the first finding needed 80 pt and the heading kept only 42)."""
        from . import render
        size = {1: 15.0, 2: 11.5, 3: 9.5}[level]
        self.ensure(size*1.6+max(42.0, float(keep)))   # keep a heading with the first lines that follow it
        if level == 1:
            self.y += 6
        self.anchor(anchor)
        if bookmark and anchor:
            self.doc.bookmark(s, anchor)
        lines = wrap(s, size, self.width, 'bold')
        for line in lines:
            self.text_line(self.left, line, size, INK if level < 3 else INK_2, 'bold', leading=size*1.3)
            self.y += size*1.3
        if level == 1:
            self.add([render.rect(self.left, self.y+1, 34, 2.2, fill=BRAND)])
            self.y += 9
        else:
            self.y += 3

    def kicker(self, s, colour=BRAND_DARK):
        self.ensure(12)
        self.text_line(self.left, s.upper(), 6.8, colour, 'bold', leading=10)
        self.y += 10

    def box(self, lines_spec, fill=TINT, pad=8.0, after=8.0, accent=BRAND):
        """A tinted box around paragraphs: [(text, size, font, colour), ...]; kept on one page."""
        from . import render
        wrapped = [(wrap(t, s, self.width-2*pad, f), s, f, c) for t, s, f, c in lines_spec if t]
        h = 2*pad+sum(len(ls)*s*1.38+2 for ls, s, f, c in wrapped)
        if h > self.bottom-self.top:                  # taller than a page: plain paragraphs, which break pages
            for t, s, f, c in lines_spec:
                self.para(t, s, c, f, indent=pad, after=2)
            self.y += after
            return
        self.ensure(h)
        self.add([render.rect(self.left, self.y, self.width, h, fill=fill),
                  render.rect(self.left, self.y, 2.4, h, fill=accent)])
        y = self.y+pad
        for ls, s, f, c in wrapped:
            for line in ls:
                self.add([render.text(self.left+pad, y+(s*1.38+s*.7)/2, line, s, c, f)])
                y += s*1.38
            y += 2
        self.y += h+after

    def table(self, columns, rows, size=7.8, after=8.0, header=True):
        """columns: [(title, share of the width, align)]; rows: lists of cells, a cell is text or
        {'text', 'uri'|'dest', 'font', 'colour'}. A row is kept on one page when it fits on one; a row taller than a
        page (a 2000-character coach note of wide letters is about 60 lines in its column) continues on the next
        page under the repeated header, instead of being drawn past the bottom edge (to y = 1357 pt on an 842 pt
        page in the first version, its end lost)."""
        from . import render
        widths = [share*self.width for _, share, _ in columns]
        pad, leading = 3.5, size*1.32

        def cell_lines(cell, w):
            t = cell.get('text') if isinstance(cell, dict) else cell
            return wrap('' if t is None else str(t), size, w-2*pad, (cell.get('font') if isinstance(cell, dict)
                                                                     else None) or 'regular') or ['']

        head_lines = [cell_lines({'text': c[0], 'font': 'bold'}, w) for c, w in zip(columns, widths)]
        head_h = max(len(x) for x in head_lines)*leading+2*pad if header else 0.0

        def draw_header():
            self.add([render.rect(self.left, self.y, self.width, head_h, fill=HEAD)])
            x = self.left
            for lines, w, (title, _, align) in zip(head_lines, widths, columns):
                for k, line in enumerate(lines):
                    tx = x+pad if align != 'right' else x+w-pad
                    self.add([render.text(tx, self.y+pad+k*leading+size*.95, line, size, INK, 'bold',
                                          'right' if align == 'right' else 'left')])
                x += w
            self.y += head_h

        def next_page():
            self.new_page()
            if header:
                draw_header()

        def draw_piece(row, cl, a, b, zebra):
            """Lines a..b of every cell of a row, as one band from the cursor down."""
            h = (b-a)*leading+2*pad
            if zebra:
                self.add([render.rect(self.left, self.y, self.width, h, fill=ZEBRA)])
            x = self.left
            for cell, lines, w, (_, _, align) in zip(row, cl, widths, columns):
                d = cell if isinstance(cell, dict) else {}
                link = {k: d[k] for k in ('uri', 'dest') if d.get(k)}
                colour = d.get('colour') or (BRAND_DARK if link else INK)
                font = d.get('font') or 'regular'
                for k, line in enumerate(lines[a:b]):
                    tx = x+pad if align != 'right' else x+w-pad
                    ty = self.y+pad+k*leading+size*.95
                    self.add([render.text(tx, ty, line, size, colour, font, 'right' if align == 'right' else 'left')])
                    if link and line:
                        lw = text_width(line, size, font)
                        lx = tx if align != 'right' else tx-lw
                        self.add([dict({'op': 'link', 'x': lx, 'y': ty-size*.85, 'w': lw, 'h': size*1.15}, **link)])
                x += w
            self.add([render.path([(self.left, self.y+h), (self.right, self.y+h)], stroke=RULE, width=.4)])
            self.y += h

        if header:
            self.ensure(max(3*leading+4*pad, head_h+leading+2*pad))     # never a header alone at a page end
            draw_header()
        fresh = self.bottom-self.top-head_h                     # the room for rows on a new page
        for r, row in enumerate(rows):
            cl = [cell_lines(c, w) for c, w in zip(row, widths)]
            n = max(len(x) for x in cl)
            h = n*leading+2*pad
            room = self.bottom-self.y
            if h > room and (h <= fresh or room < leading+2*pad):  # fits on a new page, or not one line fits here
                next_page()
            start = 0
            while start < n:
                fit = max(1, int((self.bottom-self.y-2*pad)//leading))
                end = min(n, start+fit)
                draw_piece(row, cl, start, end, r % 2)
                start = end
                if start < n:
                    next_page()
        self.y += after


def _items(v, wrap_text=False):
    """A list field as a list: a lone string becomes one item when wrap_text, any other wrong type nothing."""
    if isinstance(v, str):
        return [v] if wrap_text else []
    return list(v) if isinstance(v, (list, tuple)) else []


def _map(v):
    return dict(v) if isinstance(v, dict) else {}


def sanitize(report):
    """A shallow-copied report whose list and dictionary fields have the contract's types (section 7), so a
    malformed field leaves its part of the PDF empty instead of failing the export. Values are not changed."""
    r = _map(report)
    for key in ('kind', 'context', 'team', 'teamName', 'run', 'generated', 'engineVersion', 'fingerprint'):
        v = r.get(key)
        r[key] = v if isinstance(v, str) else str(v) if _is_num(v) else None
    for key in ('summary', 'strengths', 'improvements', 'reviews', 'priorities', 'training', 'evidence',
                'observations', 'feedback'):
        r[key] = [x for x in _items(r.get(key), key == 'summary') if key == 'summary' or isinstance(x, dict)]
    for key in ('overview', 'range', 'video', 'breakdown', 'confidence'):
        r[key] = _map(r.get(key))
    r['overview']['coverage'] = _map(r['overview'].get('coverage'))
    r['overview']['limitations'] = _items(r['overview'].get('limitations'), True)
    r['confidence']['notes'] = _items(r['confidence'].get('notes'), True)
    for key in ('strengths', 'improvements', 'reviews'):
        r[key] = [dict(f, evidence=[e for e in _items(f.get('evidence')) if isinstance(e, str)],
                       metrics=[m for m in _items(f.get('metrics')) if isinstance(m, dict)]) for f in r[key]]
    obs = []
    for o in r['observations']:
        o = dict(o, confidence=_map(o.get('confidence')), baseline=_map(o.get('baseline')))
        o['confidence']['reasons'] = _items(o['confidence'].get('reasons'), True)
        o['baseline']['professionalRatePerMinute'] = _map(o['baseline'].get('professionalRatePerMinute'))
        obs.append(o)
    r['observations'] = obs
    r['evidence'] = [dict(e, metrics=_map(e.get('metrics')),
                          diagram=e.get('diagram') if isinstance(e.get('diagram'), dict) else None,
                          videoOverlay=e.get('videoOverlay') if isinstance(e.get('videoOverlay'), dict) else None)
                     for e in r['evidence']]
    r['breakdown'] = {k: dict(_map(v), findings=_items(_map(v).get('findings')),
                              notes=_items(_map(v).get('notes'), True)) for k, v in r['breakdown'].items()}
    r['priorities'] = r['priorities'][:3]
    r['training'] = [dict(t, findings=_items(t.get('findings'))) for t in r['training']]
    r['feedback'] = [dict(f, observation=_map(f.get('observation'))) for f in r['feedback']]
    return r


class _Report:
    """Lays out one coaching report on a Document."""

    def __init__(self, report, frame_provider, options):
        self.r = sanitize(report)
        self.frames = frame_provider
        self.o = dict(options or {})
        self.labels = {str(k): str(v) for k, v in (self.o.get('labels') or {}).items()}
        self.colours = self.o.get('colours') or None
        team = self.r.get('team') if self.r.get('team') in ('A', 'B') else None
        self.team = team
        name = self.r.get('teamName')
        self.team_short = name or (f'Team {team}' if team else 'Both teams')
        self.team_long = name or (f'Team {team} (as labelled in this run)' if team else 'Both teams (as labelled)')
        other = {'A': 'B', 'B': 'A'}.get(team)
        self.team_names = {team: self.team_short} if team else {}
        if other:
            self.team_names[other] = f'Team {other}'
        self.observations = {o.get('id'): o for o in (self.r.get('observations') or []) if isinstance(o, dict)}
        self.evidence = {e.get('id'): e for e in (self.r.get('evidence') or []) if isinstance(e, dict)}
        self.findings = {}
        # Every observation a finding stands for (report.py groups the strengths of one compact block into one finding:
        # 'observation' is its first member, 'observations' all of them): each member's evidence, statistics row and
        # training link point at the finding, whose block carries an anchor for every member.
        self.members = {}
        for key in ('strengths', 'improvements', 'reviews'):
            for f in self.r.get(key) or []:
                if isinstance(f, dict) and f.get('observation'):
                    self.findings.setdefault(f['observation'], f)
                    for m in [f['observation']]+[x for x in f.get('observations') or [] if isinstance(x, str)]:
                        self.members.setdefault(m, f)
        self.drills = {}
        for t in self.r.get('training') or []:
            d = t.get('drill') if isinstance(t, dict) else None
            did = d.get('id') if isinstance(d, dict) else d
            if did:
                self.drills[did] = d if isinstance(d, dict) else {'id': did}
        self.url = self.o.get('playback_url')
        stale = self.o.get('stale')
        self.stale = ([stale] if isinstance(stale, str) else list(stale) if isinstance(stale, (list, tuple))
                      else [] if stale else None)
        diagrams = [e.get('diagram') for e in self.evidence.values() if isinstance(e.get('diagram'), dict)]
        cov = (self.r.get('overview') or {}).get('coverage') or {}
        # Metres are estimated unless the report says the pitch was measured (every run so far: assumed 105 x 68).
        self.estimated = not (cov.get('metresEstimated') is False
                              or (diagrams and all(d.get('metresEstimated') is False for d in diagrams)))
        self.max_rows = int(self.o.get('max_evidence_per_finding', MAX_EVIDENCE_PER_FINDING))
        self.shots_left = int(self.o.get('max_screenshots', MAX_SCREENSHOTS))
        self.frame_width = int(self.o.get('image_width', FRAME_WIDTH))
        self.numbers = {}                 # evidence id -> E number (in the order the findings show them)
        self.drawn_evidence = set()
        self.missing = []
        self.skipped = []
        self.screenshots = 0
        self.overlays = 0
        self.resolve = None
        if self.labels and self.o.get('substitute_text', True):
            from . import language                   # the package's one id -> display rule (one pass, no chains)
            self.resolve = language.resolve_players
        self.level = self.r.get('context') if self.r.get('context') in CONTEXT_NAMES else 'competitive'
        order = [k for k in section_order(self.o.get('section_order')) if k != 'overview']
        self.order = ['overview'] + order                 # the title page always comes first
        # Each limitation line is printed once, under Coverage and confidence: the first PDFs printed every line
        # three times (page 1, the breakdown and the coverage list; AM B: 11 lines, 24 printings) and a fourth time
        # as raw reason codes in the "Not assessed" list.
        self.limits = []
        ov = self.r.get('overview') or {}
        lines = [x for x in ov.get('limitations') or [] if isinstance(x, str) and x.strip()]
        # The run's data-quality note (report.py: the first overview.dataQualityLines limitations, stated once with what
        # would help: small players, no pitch keyframes, a few players per team in view) is printed on page 1, where it
        # tells how far every finding can be read; the other lines, one per measurement, under Limitations.
        n = ov.get('dataQualityLines')
        n = min(n, len(lines)) if isinstance(n, int) and not isinstance(n, bool) and n > 0 else 0
        self.quality = []
        for x in lines[:n]:
            if x.strip() not in self.quality:
                self.quality.append(x.strip())
        for x in lines[n:]:
            if x.strip() not in self.limits and x.strip() not in self.quality:
                self.limits.append(x.strip())
        # The observations the report's own limitation lines already cover (overview.notAssessed): the Limitations list
        # adds a line of its own only for an insufficient or suppressed observation that has none.
        self.covered = {x.get('observation') for x in ov.get('notAssessed') or []
                        if isinstance(x, dict) and isinstance(x.get('observation'), str)}
        self.compared_ids = {i for i, o in self.observations.items() if compared(o)}

    # -- helpers ----------------------------------------------------------------------------------------------
    def say(self, s):
        """Report text with identities shown as the current display labels (when the caller gave them)."""
        if s is None:
            return None
        s = str(s)
        return self.resolve(s, self.labels) if self.resolve else s

    def playback(self, ev):
        u = self.url
        if not u or not isinstance(ev, dict):
            return None
        try:
            if callable(u):
                link = u(ev)
            else:
                link = str(u).format(frame=int(ev.get('frame')), seek=f'{float(ev.get("seek")):.4f}',
                                     time=f'{float(ev.get("time")):.3f}', id=ev.get('id', ''),
                                     video=(self.r.get('video') or {}).get('key', ''), run=self.r.get('run', ''))
        except (TypeError, ValueError, KeyError, IndexError, AttributeError):
            return None
        return safe_uri(link)

    def label_of(self, f=None, obs=None):
        if f and f.get('label'):
            return f['label']
        obs = obs or {}
        if obs.get('severity') in SEVERITY_LABELS:
            return SEVERITY_LABELS[obs['severity']]
        if obs.get('classification') == 'improvement':
            return 'Potential improvement'
        return CLASS_LABELS.get(obs.get('classification'), 'Observation')

    def headline(self, obs_id):
        f = self.findings.get(obs_id)
        if f and f.get('headline'):
            return self.say(f['headline'])
        return self.topic(self.observations.get(obs_id) or {})

    def topic(self, obs):
        """What an observation measures, in the report level's words (language.PATTERNS topics: 'Sideways shift of
        back line and midfield', not the pattern code 'lateral_lag' the first PDFs printed as 'Lateral lag')."""
        try:
            from . import language
        except Exception:
            language = None
        if language is not None and isinstance(obs, dict):
            pats, dets = getattr(language, 'PATTERNS', {}), getattr(language, 'DETECTORS', {})
            keys = getattr(language, 'KEY_TOPIC', {})
            names = [obs.get('pattern'), (dets.get(obs.get('detector')) or (None, None, None))[2],
                     (dets.get(obs.get('pattern')) or (None, None, None))[2]]
            for name in names:
                t = ((pats.get(name) or {}).get('topic') or {}) if isinstance(name, str) else {}
                if t.get(self.level):
                    return t[self.level]
            t = keys.get(obs.get('recommendationKey')) or {}
            if t.get(self.level):
                return t[self.level]
        return 'A further measurement'

    def reason_words(self, obs):
        """The plain-words reasons behind an observation's confidence or suppression (language.REASONS); an unknown
        code is left out, never printed raw."""
        try:
            from . import language
            words = language.reason_phrases(obs)
        except Exception:
            words = []
        why = obs.get('suppressionReason') if isinstance(obs, dict) else None
        if isinstance(why, str) and why.strip():
            try:
                from . import language
                text = language.REASONS.get(why.strip())
            except Exception:
                text = None
            if text is None and ' ' in why.strip() and not re.search(r'[a-z]_[a-z]', why):
                text = self.say(why.strip().rstrip('.'))
            if text and text not in words:
                words = [text] + list(words)
        return [w for w in words if isinstance(w, str) and w]

    def evidence_time(self, ev):
        t = ev.get('time')
        seg = ev.get('segmentTime')
        if not _is_num(seg) and _is_num(t) and _is_num((self.r.get('range') or {}).get('t0')):
            seg = t-self.r['range']['t0']
        return clock(t), offset(seg) if _is_num(seg) else 'n/a'

    def number(self, ev_id):
        if ev_id not in self.numbers:
            self.numbers[ev_id] = len(self.numbers)+1
        return self.numbers[ev_id]

    def frame_jpeg(self, ev):
        frame = ev.get('frame')
        if isinstance(frame, float) and frame.is_integer():
            frame = int(frame)
        if self.frames is None:
            return None, 'no source video was available when this PDF was made'
        if not isinstance(frame, int) or isinstance(frame, bool) or frame < 0:
            return None, 'the evidence has no frame index'
        if self.shots_left <= 0:
            return None, 'screenshot left out to keep the file small (see the timestamp)'
        try:
            data = self.frames(frame, self.frame_width)
        except Exception as exc:                                   # a broken video must not break the report
            data, why = None, f'the frame could not be read ({type(exc).__name__})'
        else:
            why = 'the frame could not be read from the source video'
        if not data:
            self.missing.append(frame)
            return None, why
        try:
            jpeg_info(data)
        except ValueError:
            self.missing.append(frame)
            return None, 'the frame provider returned no JPEG'
        self.shots_left -= 1
        return data, None

    # -- sections ---------------------------------------------------------------------------------------------
    def layout(self, doc):
        rng = self.r.get('range') or {}
        header = f'{self.team_short}  |  {clock(rng.get("t0"))} - {clock(rng.get("t1"))}'
        if self.stale is not None:
            header = 'OUT OF DATE  |  ' + header
        self.L = _Layout(doc, header)
        self.prefetch()
        draw = {'overview': self.cover, 'summary': self.summary,
                'strengths': lambda: self.findings_section('strengths', self.r.get('strengths')),
                'improvements': lambda: self.findings_section('improvements', self.r.get('improvements')),
                'reviews': lambda: self.findings_section('reviews', self.r.get('reviews')),
                'breakdown': self.breakdown, 'statistics': self.statistics, 'priorities': self.priorities,
                'training': self.training, 'confidence': self.coverage, 'evidence': self.evidence_index,
                'feedback': self.feedback}
        for key in self.order:
            draw[key]()
        self.footers(doc)

    def prefetch(self):
        """Ask the provider for every screenshot frame at once (frames.py reads them sorted, in one pass)."""
        if self.frames is None or not hasattr(self.frames, 'prefetch'):
            return
        wanted = []
        for key in ('strengths', 'improvements', 'reviews'):
            for f in self.r.get(key) or []:
                for ev_id in (f.get('evidence') or [])[:self.max_rows] if isinstance(f, dict) else []:
                    ev = self.evidence.get(ev_id)
                    if ev and isinstance(ev.get('frame'), int):
                        wanted.append(ev['frame'])
        try:
            self.frames.prefetch(sorted(set(wanted))[:max(0, self.shots_left)*4], [self.frame_width])
        except Exception:                                          # the per-frame calls report what is missing
            pass

    def cover(self):
        from . import render
        L, r = self.L, self.r
        ov = r.get('overview') or {}
        kind = r.get('kind')
        title = self.say(ov.get('title')) or ('MATCH COACHING REPORT' if kind == 'match' else 'SEGMENT COACHING REPORT')
        L.y += 4
        L.kicker('PitchIQ')
        for line in wrap(title, 21, L.width, 'bold'):
            L.text_line(L.left, line, 21, INK, 'bold', leading=26)
            L.y += 26
        L.text_line(L.left, self.team_long, 12.5, INK_2, 'regular', leading=18)
        L.y += 24
        if self.stale:
            L.box([('OUT OF DATE', 8.5, 'bold', '#8a5f12'),
                   (stale_sentence(self.stale, self.say), 8.6, 'regular', INK)],
                  fill='#faf0dc', accent='#d9a961')
        rng = r.get('range') or {}
        video = r.get('video') or {}
        span = rng.get('seconds')
        if not _is_num(span) and _is_num(rng.get('t0')) and _is_num(rng.get('t1')):
            span = rng['t1']-rng['t0']
        rows = [('Video', video.get('name') or video.get('key') or 'n/a'), ('Run', r.get('run') or 'n/a'),
                ('Analysed range', f'{clock(rng.get("t0"))} - {clock(rng.get("t1"))} of the source video'
                 + (f' ({_seconds(span)})' if _is_num(span) else '')
                 + (f'; segment {offset(rng.get("segmentT0"))} to {offset(rng.get("segmentT1"))}'
                    if _is_num(rng.get('segmentT0')) and _is_num(rng.get('segmentT1')) else ''))]
        if kind == 'match':
            rows.append(('Report kind', 'Match report'))
        else:
            rows.append(('Report kind', 'Segment report: it describes this observed range only, not a whole match'))
        ctx = r.get('context')
        if ctx:
            rows.append(('Context level', f'{CONTEXT_NAMES.get(ctx, _humanize(ctx))} (the level changes the wording '
                                          'only; every measurement is the same at every level)'))
        if r.get('generated'):
            rows.append(('Generated', str(r['generated']).replace('T', ' ')[:19]))
        ref = self.reference_name()
        if ref:
            name, _, sha = ref.partition('@')
            rows.append(('Professional reference', name + (f' (version {sha})' if sha else '')
                         + ': one professional match, a sanity range and not a standard'))
        ident = ', '.join(x for x in (f'engine {r["engineVersion"]}' if r.get('engineVersion') else None,
                                      f'fingerprint {r["fingerprint"]}' if r.get('fingerprint') else None) if x)
        if ident:
            rows.append(('Report id', ident))
        self.kv(rows)
        # At a glance: counts straight from the report's lists, and its own overall confidence.
        counts = [(len(r.get(key) or []), one if len(r.get(key) or []) == 1 else many, key)
                  for key, one, many in (('strengths', 'observed strength', 'observed strengths'),
                                         ('improvements', 'potential improvement', 'potential improvements'),
                                         ('reviews', 'review opportunity', 'review opportunities'))]
        conf = (r.get('confidence') or {}).get('overall')
        L.ensure(46)
        cw = (L.width-3*8)/4
        x = L.left
        for n, label, anchor in counts:
            L.add([render.rect(x, L.y, cw, 38, fill=TINT),
                   render.text(x+8, L.y+19, str(n), 15, BRAND_DARK, 'bold'),
                   render.text(x+8, L.y+31, label, 7, INK_2),
                   {'op': 'link', 'x': x, 'y': L.y, 'w': cw, 'h': 38, 'dest': anchor}])
            x += cw+8
        L.add([render.rect(x, L.y, cw, 38, fill=TINT),
               render.text(x+8, L.y+19, _humanize(conf) if conf else 'n/a', 13, BRAND_DARK, 'bold'),
               render.text(x+8, L.y+31, 'overall confidence', 7, INK_2),
               {'op': 'link', 'x': x, 'y': L.y, 'w': cw, 'h': 38, 'dest': 'confidence'}])
        L.y += 50
        if self.quality or self.limits:
            L.kicker('Read with care', '#8a5f12')
        if self.quality:                 # the footage's data-quality note, here only
            L.bullets([self.say(x) for x in self.quality], size=8.8)
        if self.limits:                  # a pointer only: each limitation is printed once, under Limitations
            n = len(join_bare(self.limits))
            more = ' further' if self.quality else ''
            L.para(f'This report has {n}{more} limitation{"s" if n != 1 else ""} (measurements that could not be '
                   'assessed, or limits of the footage); each is listed once under Limitations, in Coverage and '
                   'confidence.', 8.8, INK, after=6, link={'dest': 'limitations'})
        L.heading('Contents', 2)
        titles = dict(SECTIONS)
        entries = [(titles[k], k) for k in self.order if k in titles and (k != 'feedback' or r.get('feedback'))]
        half = (len(entries)+1)//2
        leading = 12.5
        L.ensure(half*leading+6)
        top = L.y
        for k, (t, anchor) in enumerate(entries):
            x = L.left+(0 if k < half else L.width/2)
            y = top+(k if k < half else k-half)*leading
            L.add([render.text(x, y+9, f'{k+1}.  {t}', 9, BRAND_DARK),
                   {'op': 'link', 'x': x, 'y': y, 'w': text_width(f'{k+1}.  {t}', 9), 'h': leading, 'dest': anchor}])
        L.y = top+half*leading+8
        L.box([('How to read this report', 8.5, 'bold', INK),
               ('Every statement comes from the tracking of this video and from the existing PitchIQ analytics. '
                + self.comparison_sentence() + ' Times are source-video times (mm:ss.ss) followed by the offset into '
                'the analysed segment. '
                + ('Metre values are estimated from an assumed pitch size. ' if self.estimated else '')
                + 'Diagrams show tracked positions'
                + ('; dashed shapes marked COACHING ILLUSTRATION are suggestions, not tracked data'
                   if self.illustrated() else '')
                + '. This PDF holds no video: use the timestamps to find each moment.',
                8.3, 'regular', INK_2)])

    def illustrated(self):
        """True when an evidence figure holds a 'suggested' shape: the cover describes COACHING ILLUSTRATION shapes
        only then (critic B4/M2: the sentence printed on page 1 of every PDF while no producer emitted one)."""
        return any(isinstance(el, dict) and el.get('type') == 'suggested'
                   for e in self.evidence.values() for spec in (e.get('diagram'), e.get('videoOverlay'))
                   if isinstance(spec, dict) for el in (spec.get('elements') or []))

    def shown_ids(self):
        return [f.get('observation') for key in ('strengths', 'improvements', 'reviews')
                for f in self.r.get(key) or [] if isinstance(f, dict) and f.get('observation')]

    def comparison_sentence(self):
        """Which findings were compared with the professional reference: only those whose baseline was available
        (the first PDFs said every statement was, under formation readings too)."""
        shown = self.shown_ids()
        with_ref = [o for o in shown if o in self.compared_ids]
        if not shown:
            return ('Measurements with a professional baseline are compared with professional tracking data under '
                    'the same rule; formation readings and ball location are descriptions without one.')
        if len(with_ref) == len(shown):
            return 'Each finding is compared with professional tracking data under the same rule and context.'
        if not with_ref:
            return 'No finding of this report has a professional comparison: they describe the tracked positions only.'
        without = []
        for o in shown:
            t = self.topic(self.observations.get(o) or {})
            t = t[:1].lower()+t[1:]                                   # mid-sentence
            if o not in self.compared_ids and t not in without:
                without.append(t)
        names = ', '.join(without[:-1]) + (' and ' if len(without) > 1 else '') + without[-1]
        return (f'Findings marked "compared with the professional reference" are set against professional tracking '
                f'data under the same rule and context; {names} {"have" if len(without) > 1 else "has"} no '
                f'professional comparison and {"are" if len(without) > 1 else "is"} described from the tracked '
                f'positions only.')

    def reference_name(self):
        """The professional reference the report's comparisons used; None when no assessed observation was compared
        (one that was not assessed compared nothing: AM B had only those and a formation review, and the first PDFs
        still named the reference on its title page)."""
        for o in self.observations.values():
            ref = (o.get('baseline') or {}).get('reference')
            if ref and compared(o) and o.get('classification') != 'insufficient' and o.get('status') != 'suppressed':
                return str(ref)
        return None

    def kv(self, rows, key_width=118.0, size=8.8):
        from . import render
        L = self.L
        leading = size*1.4
        for key, value in rows:
            lines = wrap(self.say(value), size, L.width-key_width) or ['']
            L.ensure(min(len(lines), 6)*leading)          # a long value continues on the next page, line by line
            L.add([render.text(L.left, L.y+(leading+size*.7)/2, key, size, MUTED, 'bold')])
            for k, line in enumerate(lines):
                if k:
                    L.ensure(leading)
                L.text_line(L.left+key_width, line, size, INK, leading=leading)
                L.y += leading
            L.add([render.path([(L.left, L.y+1), (L.right, L.y+1)], stroke=RULE, width=.35)])
            L.y += 3
        L.y += 8

    def summary(self):
        L = self.L
        L.heading('Summary', 1, 'summary', True)
        items = [self.say(s) for s in (self.r.get('summary') or []) if s]
        if items:
            L.bullets(items, size=9.6)
        else:
            L.para('The report has no summary sentences for this range.', colour=INK_2, font='oblique')
        context = [self.say(s) for s in self.r['overview'].get('context') or [] if isinstance(s, str) and s]
        if context:
            L.heading('Context of the observed play', 3)
            L.bullets(context, size=8.8)

    def findings_section(self, key, findings):
        L = self.L
        findings = [f for f in (findings or []) if isinstance(f, dict)]
        L.heading(dict(SECTIONS)[key], 1, key, True, keep=90 if findings else 0)
        if not findings:
            L.para(EMPTY[key], colour=INK_2, font='oblique')
            return
        for f in findings:
            self.finding(f)

    def finding(self, f):
        from . import render
        L = self.L
        obs = self.observations.get(f.get('observation')) or {}
        label = self.label_of(f, obs)
        L.ensure(80)
        for m in dict.fromkeys([f.get('observation')]+[x for x in f.get('observations') or [] if isinstance(x, str)]):
            L.anchor(f'obs-{m}')
        fg, bg = LABEL_STYLES.get(label, (INK_2, HEAD))
        bw = text_width(label.upper(), 6.6, 'bold')+10
        L.add([render.rect(L.left, L.y, bw, 11, fill=bg), render.text(L.left+5, L.y+8, label.upper(), 6.6, fg, 'bold')])
        L.y += 15
        L.para(self.say(f.get('headline')) or self.headline(f.get('observation')), 11.2, INK, 'bold', after=3)
        strength = obs.get('classification') == 'strength' or label == 'Observed strength'
        for title, field in (('What happened', 'whatHappened'), ('Why it matters', 'whyItMatters'),
                             ('To build on it' if strength else 'What to improve', 'whatToImprove'),
                             ('How to practise', 'howToPractice')):
            if f.get(field):
                body = f[field]
                if field == 'whatToImprove' and strength and BUILD_ON.match(body):
                    body = BUILD_ON.sub('', body, 1)        # language.py starts it "To build on it:": the title says it
                    body = body[:1].upper()+body[1:]
                L.ensure(26)
                L.text_line(L.left, title.upper(), 6.6, BRAND_DARK, 'bold', leading=10)
                L.y += 10
                L.para(self.say(body), 9.3, INK, after=4)
        metrics = [m for m in (f.get('metrics') or []) if isinstance(m, dict) and m.get('name')]
        if metrics:
            L.table([('Measure', .62, 'left'), ('Value', .38, 'right')],
                     [[self.say(m['name']), value_text(m.get('value'), m.get('unit'),
                                                       m.get('unit') in ('m', 'm2')
                                                       and m.get('estimated', self.estimated) is not False)]
                      for m in metrics], size=7.8, after=5)
        if f.get('video'):
            L.ensure(26)
            L.text_line(L.left, 'ON THE VIDEO', 6.6, BRAND_DARK, 'bold', leading=10)
            L.y += 10
            L.para(self.say(f['video']), 9.0, INK, after=4)
        facts = self.observation_facts(obs, brief=bool(metrics))
        if facts:
            L.para(facts, 7.6, MUTED, after=3)
        reasons = [self.say(x) for x in ((obs.get('confidence') or {}).get('reasons') or []) if x]
        if f.get('confidenceNote'):
            L.para(self.say(f['confidenceNote']), 7.6, MUTED, after=3)
        elif reasons:
            L.para('Confidence notes: ' + '; '.join(reasons) + '.', 7.6, MUTED, after=3)
        drill = f.get('drill')
        if drill:
            d = self.drills.get(drill) or {}
            name = d.get('title') or d.get('name') or _humanize(drill)
            L.para(f'Suggested practice: {self.say(name)} (see Training recommendations)', 8.4, BRAND_DARK,
                   after=4, link={'dest': f'drill-{drill}'} if drill in self.drills else None)
        evs = [e for e in (f.get('evidence') or []) if e in self.evidence]
        for k, ev_id in enumerate(evs[:self.max_rows]):
            self.evidence_row(self.evidence[ev_id], k+1, len(evs))
        if len(evs) > self.max_rows:
            more = ', '.join(f'E{self.number(e)}' for e in evs[self.max_rows:])
            L.para(f'Further evidence moments for this finding: {more} (see Evidence timestamps).', 7.8, INK_2,
                   after=4, link={'dest': 'evidence'})
        L.add([render.path([(L.left, L.y+2), (L.right, L.y+2)], stroke=RULE, width=.6)])
        L.y += 12

    def metric_text(self, key, v):
        """'defence-midfield gap 21.4 m (est.)' for a metric of the coaching series (a value or {peak, mean})."""
        name, unit = METRICS.get(key, (_humanize(key).lower(), None))
        est = self.estimated and unit in ('m', 'm2')
        if isinstance(v, dict):
            parts = [f'{k} {value_text(x, unit, est)}' for k, x in v.items() if _is_num(x)]
            return f'{name} ' + ', '.join(parts) if parts else None
        return f'{name} {value_text(v, unit, est)}' if _is_num(v) else None

    def observation_facts(self, obs, brief=False):
        """One sentence of the observation's own numbers (measured time, occurrences, rate, baseline, confidence);
        brief: only the comparison context and the confidence (the finding's metric rows show the numbers)."""
        if not obs:
            return None
        bits = []
        base = obs.get('baseline') or {}
        ref = compared(obs)                    # only a baseline that was available is a comparison
        if brief:
            if ref:
                ctx = _context(base.get('context') or 'all')
                bits.append('compared with the professional reference' + (f' {ctx}' if ctx else ''))
            else:
                bits.append('no professional comparison: described from the tracked positions')
        elif _is_num(obs.get('measuredSeconds')):
            share = obs.get('eligibleShare')
            bits.append(f'measured {_seconds(obs["measuredSeconds"])}'
                        + (f' ({share*100:.0f}% of the range)' if _is_num(share) else ''))
        if not brief and _is_num(obs.get('occurrenceCount')):
            n = int(obs['occurrenceCount'])
            bits.append(f'{n} occurrence' + ('' if n == 1 else 's'))
        if not brief and _is_num(obs.get('ratePerMeasuredMinute')):
            bits.append(f'{obs["ratePerMeasuredMinute"]:.2f} per measured minute')
        pro = base.get('professionalRatePerMinute') or {}
        if not brief and ref and _is_num(pro.get('p50')) and _is_num(pro.get('p90')):
            ctx = _context(base.get('context') or 'all')
            bits.append(f'professional rate p50 {pro["p50"]:.2f} / p90 {pro["p90"]:.2f} per minute'
                        + (f' {ctx}' if ctx else ''))
        elif not brief and not ref:
            bits.append('no professional comparison: described from the tracked positions')
        conf = obs.get('confidence') or {}
        parts = [f'{ASPECT_WORDS[k]} {conf[k]}' for k in ('tracking', 'calibration', 'identity', 'ball')
                 if conf.get(k) and conf.get(k) != 'n/a']
        if conf.get('overall'):
            bits.append(f'confidence {conf["overall"]}' + (f' ({", ".join(parts)})' if parts else ''))
        if not bits:
            return None
        s = '; '.join(bits)
        return s[:1].upper()+s[1:]+'.'

    def evidence_row(self, ev, k, total):
        from . import render
        L = self.L
        col = (L.width-COLUMN_GAP)/2
        number = self.number(ev.get('id'))
        abs_t, seg_t = self.evidence_time(ev)
        diagram = None
        if isinstance(ev.get('diagram'), dict):
            diagram = render.diagram_ops(ev['diagram'], col, self.colours, self.labels, self.team_names)
            self.skipped += [dict(s, evidence=ev.get('id'), figure='diagram') for s in diagram['skipped']]
        jpeg, why = self.frame_jpeg(ev)
        img_h = 0.0
        if jpeg:
            iw, ih, _ = jpeg_info(jpeg)
            img_h = col*ih/iw
        fig_h = max(diagram['height'] if diagram else 0.0, img_h, 34.0)
        title = f'Evidence E{number} ({k} of {total})  |  {abs_t}  (segment {seg_t})'
        if isinstance(ev.get('frame'), int):
            title += f'  |  frame {ev["frame"]}'
        cap = []
        if diagram and (ev['diagram'].get('title')):
            cap.append(f'Diagram: {self.say(ev["diagram"]["title"])}.')
        tier_note, ov = None, None
        if jpeg:
            cap.append('Screenshot: the exact source frame.')
            overlay = ev.get('videoOverlay')
            if isinstance(overlay, dict):
                ow, oh = overlay.get('width'), overlay.get('height')
                if _is_num(ow) and _is_num(oh) and oh and abs(ow/oh-iw/ih)/(iw/ih) <= .02:
                    ov = render.overlay_ops(overlay, 0.0, 0.0, col, img_h, self.colours, self.labels)
                    tier_note = ov['note']                 # what this overlay drew, at its tier
                else:
                    cap.append('The overlay was left out: its image size does not match the frame.')
            if tier_note:
                cap.append(tier_note)
        if ev.get('caption'):
            cap.append(self.say(ev['caption']))
        metrics = ev.get('metrics') or {}
        if isinstance(metrics, dict) and metrics:
            shown = [self.metric_text(k, v) for k, v in metrics.items()]
            shown = [x for x in shown if x]
            if shown:
                cap.append('Measured: ' + ', '.join(shown) + '.')
        if ev.get('confidence'):
            cap.append(f'Evidence confidence: {ev["confidence"]}.')
        cap_lines = wrap(' '.join(cap), 7.6, L.width)
        url = self.playback(ev)
        h = 13+fig_h+4+len(cap_lines)*7.6*1.36+(11 if url else 0)+6
        L.ensure(h)
        L.anchor(f'ev-{ev.get("id")}')
        self.drawn_evidence.add(ev.get('id'))
        L.text_line(L.left, title, 8.2, INK, 'bold', leading=11)
        L.y += 13
        top = L.y
        if diagram:
            L.add(_translate(diagram['ops'], L.left, top))
        else:
            L.add([render.rect(L.left, top, col, 34, fill=ZEBRA),
                   render.text(L.left+6, top+20, 'No diagram for this moment.', 7.6, MUTED, 'oblique')])
        x2 = L.left+col+COLUMN_GAP
        if jpeg:
            L.add([{'op': 'image', 'x': x2, 'y': top, 'w': col, 'h': img_h, 'jpeg': jpeg, 'frame': ev.get('frame')}])
            self.screenshots += 1
            if ov is not None:
                self.skipped += [dict(s, evidence=ev.get('id'), figure='overlay') for s in ov['skipped']]
                L.add(_translate(ov['ops'], x2, top))
                self.overlays += 1
            L.add([render.rect(x2, top, col, img_h, stroke=RULE, width=.5)])
        else:
            lines = wrap(f'Screenshot unavailable: {why}.', 7.6, col-12)
            bh = max(34.0, 12+len(lines)*10.5)
            L.add([render.rect(x2, top, col, bh, fill=ZEBRA)])
            for i, line in enumerate(lines):
                L.add([render.text(x2+6, top+14+i*10.5, line, 7.6, MUTED, 'oblique')])
        L.y = top+fig_h+4
        for line in cap_lines:
            L.text_line(L.left, line, 7.6, INK_2, leading=7.6*1.36)
            L.y += 7.6*1.36
        if url:
            s = f'Open {abs_t} in PitchIQ (needs the local PitchIQ server with this video open)'
            L.text_line(L.left, s, 7.6, BRAND_DARK, leading=11)
            L.add([render.path([(L.left, L.y+9.6), (L.left+text_width(s, 7.6), L.y+9.6)], stroke=BRAND_DARK,
                               width=.4),
                   {'op': 'link', 'x': L.left, 'y': L.y, 'w': text_width(s, 7.6), 'h': 11, 'uri': url}])
            L.y += 11
        L.y += 6

    def priorities(self):
        L = self.L
        L.heading('Priorities', 1, 'priorities', True)
        items = [p for p in (self.r.get('priorities') or []) if isinstance(p, dict)][:3]
        if not items:
            L.para(EMPTY['priorities'], colour=INK_2, font='oblique')
            return
        for n, p in enumerate(items, 1):
            L.para(f'{n}. {self.say(p.get("title")) or self.headline(p.get("finding"))}', 10.2, INK, 'bold', after=2)
            if p.get('finding'):
                L.para(f'Finding: {self.headline(p["finding"])}', 8.6, BRAND_DARK, indent=14, after=1,
                       link={'dest': f'obs-{p["finding"]}'} if p['finding'] in self.members else None)
            if p.get('drill'):
                d = self.drills.get(p['drill']) or {}
                L.para(f'Practice: {self.say(d.get("title") or d.get("name") or _humanize(p["drill"]))}', 8.6,
                       BRAND_DARK, indent=14, after=1,
                       link={'dest': f'drill-{p["drill"]}'} if p['drill'] in self.drills else None)
            L.y += 5

    def breakdown(self):
        L = self.L
        L.heading('Tactical breakdown', 1, 'breakdown', True)
        b = self.r.get('breakdown') or {}
        for key, title in BREAKDOWN:
            sec = b.get(key) if isinstance(b.get(key), dict) else {}
            L.heading(title, 2)
            if not sec.get('available'):
                reason, n = self.without_limits(sec.get('reason'))
                reason = self.say(reason) if reason else None
                if not reason:
                    reason = ('not enough reliable measurement in this range' if n else
                              'not enough measured evidence in this range')
                L.para(f'Not assessed: {reason}', 8.8, INK_2, 'oblique', after=2 if n else 4)
                if n:
                    L.para(self.limits_pointer(n), 8.4, BRAND_DARK, after=4, link={'dest': 'limitations'})
                continue
            items = []
            for obs_id in sec.get('findings') or []:
                obs = self.observations.get(obs_id) or {}
                items.append({'text': f'{self.label_of(self.findings.get(obs_id), obs)}: {self.headline(obs_id)}',
                              'link': {'dest': f'obs-{obs_id}'} if obs_id in self.members else None})
            hidden = 0
            for note in sec.get('notes') or []:
                rest, n = self.without_limits(note)
                hidden += n
                if rest:
                    items.append(self.say(rest))
            if hidden:
                items.append({'text': self.limits_pointer(hidden), 'link': {'dest': 'limitations'}})
            if items:
                L.bullets(items, size=8.8)
            else:
                L.para('No pattern in this area met the evidence rules in this range.', 8.8, INK_2, 'oblique')

    def without_limits(self, text):
        """(text without the report's limitation lines, how many it held): a breakdown note or reason that repeats a
        limitation (report.py adds the section's limitation lines to its notes, and an unavailable section's reason
        is 'Not enough reliable measurement. ' followed by them) keeps only its own words; the lines themselves are
        printed once, under Limitations."""
        if not isinstance(text, str) or not text.strip():
            return None, 0
        n = 0
        for line in sorted(self.quality, key=len, reverse=True):      # page 1's note: left out, not counted
            text = text.replace(line, ' ')
        for line in sorted(self.limits, key=len, reverse=True):
            if line in text:
                text = text.replace(line, ' ')
                n += 1
        text = re.sub(r'\s+', ' ', text).strip()
        return (text if re.search(r'[A-Za-z]', text) else None), n

    @staticmethod
    def limits_pointer(n):
        return (f'{n} measurement{"s" if n != 1 else ""} of this area could not be assessed or '
                f'{"were" if n != 1 else "was"} limited: see Limitations (Coverage and confidence).')

    def statistics(self):
        L = self.L
        L.heading('Key statistics', 1, 'statistics', True)
        rows = []
        for o in self.observations.values():
            if o.get('status', 'reported') != 'reported' or o.get('classification') == 'insufficient':
                continue
            base = o.get('baseline') or {}
            pro = base.get('professionalRatePerMinute') or {}
            oid = o.get('id')
            counted = o.get('classification') in ('improvement', 'review')     # a count of situations means something
            rows.append([{'text': self.headline(oid), 'dest': f'obs-{oid}' if oid in self.members else None},
                         self.label_of(self.findings.get(oid), o),
                         _seconds(o.get('measuredSeconds')),
                         value_text(o.get('occurrenceCount'), 'count') if counted else '-',
                         value_text(o.get('ratePerMeasuredMinute'), 'ratio') if counted else '-',
                         f'{pro["p50"]:.2f} - {pro["p90"]:.2f}' if counted and compared(o)
                         and _is_num(pro.get('p50')) and _is_num(pro.get('p90')) else '-',
                         (o.get('confidence') or {}).get('overall') or 'n/a'])
        if not rows:
            L.para('No observation in this report has measured statistics to show.', colour=INK_2, font='oblique')
            return
        L.para('Per observation, from the report: the measured time it rests on, how often the situation occurred, '
               'the rate per measured minute and the professional rate for the same rule and context (p50 - p90 '
               'per minute). Counts are shown for improvements and reviews only.', 8.4, INK_2)
        L.table([('Observation', .30, 'left'), ('Assessment', .17, 'left'), ('Measured', .10, 'right'),
                 ('Occurrences', .11, 'right'), ('Per min', .09, 'right'), ('Professional', .12, 'right'),
                 ('Confidence', .11, 'left')], rows)

    def training(self):
        L = self.L
        L.heading('Training recommendations', 1, 'training', True)
        items = [t for t in (self.r.get('training') or []) if isinstance(t, dict) and t.get('drill')]
        if not items:
            L.para(EMPTY['training'], colour=INK_2, font='oblique')
            return
        for t in items:
            d = t['drill'] if isinstance(t['drill'], dict) else {'id': t['drill']}
            did = d.get('id')
            L.heading(self.say(d.get('title') or d.get('name') or _humanize(did or 'drill')), 2,
                      f'drill-{did}' if did else None)
            seen = set()
            for field, title, unit in DRILL_FIELDS:
                v = d.get(field)
                if isinstance(v, dict):                     # a range: {'min', 'max', 'note'}
                    lo, hi = v.get('min'), v.get('max')
                    span = (value_text(lo) if lo == hi or not _is_num(hi) else f'{value_text(lo)}-{value_text(hi)}'
                            if _is_num(lo) else value_text(hi) if _is_num(hi) else '')
                    v = ' '.join(x for x in ((f'{span} {unit}.' if span and unit else span),
                                             v.get('note') if isinstance(v.get('note'), str) else None) if x)
                if v in (None, '', []) or title in seen:
                    continue
                seen.add(title)
                L.ensure(26)
                L.text_line(L.left, title.upper(), 6.6, BRAND_DARK, 'bold', leading=10)
                L.y += 10
                if isinstance(v, (list, tuple)):
                    L.bullets([self.say(x) for x in v if isinstance(x, (str, int, float))], size=8.8, after=2)
                else:
                    L.para(self.say(value_text(v) if _is_num(v) else v), 8.9, after=3)
            linked = [{'text': self.headline(o), 'link': {'dest': f'obs-{o}'} if o in self.members else None}
                      for o in t.get('findings') or []]
            if linked:
                L.text_line(L.left, 'ADDRESSES', 6.6, BRAND_DARK, 'bold', leading=10)
                L.y += 10
                L.bullets(linked, size=8.6)

    def evidence_index(self):
        L = self.L
        L.heading('Evidence timestamps', 1, 'evidence', True)
        evs = [e for e in (self.r.get('evidence') or []) if isinstance(e, dict)]
        if not evs:
            L.para(EMPTY['evidence'], colour=INK_2, font='oblique')
            return
        for e in evs:
            self.number(e.get('id'))
        L.para('Every evidence moment of the report. Video time is the time in the source video (mm:ss.ss); segment '
               'time is the offset from the start of the analysed range. Seek to the frame to see the exact moment.',
               8.4, INK_2)
        rows = []
        any_url = False
        for e in sorted(evs, key=lambda e: self.numbers[e.get('id')]):
            abs_t, seg_t = self.evidence_time(e)
            url = self.playback(e)
            any_url = any_url or bool(url)
            ref = {'text': f'E{self.numbers[e.get("id")]}',
                   'dest': f'ev-{e.get("id")}' if e.get('id') in self.drawn_evidence else None}
            rows.append([ref, {'text': self.headline(e.get('observation')),
                               'dest': f'obs-{e.get("observation")}' if e.get('observation') in self.members
                               else None},
                         abs_t, seg_t, str(e.get('frame')) if isinstance(e.get('frame'), int) else 'n/a',
                         e.get('confidence') or 'n/a', {'text': 'open', 'uri': url} if url else '-'])
        L.table([('Ref', .07, 'left'), ('Finding', .39, 'left'), ('Video time', .13, 'right'),
                 ('Segment time', .13, 'right'), ('Frame', .09, 'right'), ('Confidence', .11, 'left'),
                 ('Link', .08, 'left')], rows)
        if any_url:
            L.para('Links open the moment in PitchIQ on the computer that made this report, with the server running '
                   'and this video open.', 7.8, MUTED)

    def coverage(self):
        L = self.L
        r = self.r
        L.heading(dict(SECTIONS)['confidence'], 1, 'confidence', True)
        conf = r.get('confidence') or {}
        if conf.get('overall'):
            L.para(f'Overall confidence: {conf["overall"]}.', 10, INK, 'bold', after=3)
        notes = [self.say(n) for n in conf.get('notes') or [] if n]
        if notes:
            L.bullets(notes, size=8.8)
        cov = (r.get('overview') or {}).get('coverage') or {}
        if isinstance(cov, dict) and cov:
            L.heading('Coverage', 2)
            rows = []
            for k, v in cov.items():
                if isinstance(v, dict):
                    nums = [b for b in v.values() if _is_num(b)]
                    shares = nums and len(nums) == len(v) and all(0 <= b <= 1 for b in nums) and sum(nums) <= 1.0001
                    unit_key = k if re.search('second|share', str(k), re.I) else None    # measuredSeconds: {...}
                    v = ', '.join(f'{_humanize(a)} {value_text(b, "share") if shares else self._cov(unit_key or a, b)}'
                                  for a, b in v.items())
                elif isinstance(v, (list, tuple)):
                    v = ', '.join(str(self._cov(k, x)) for x in v)
                else:
                    v = self._cov(k, v)
                rows.append((_humanize(k), v))
            self.kv(rows, key_width=150, size=8.4)
        # Limitations, once: the report's own lines, then any observation that was not assessed (or was suppressed)
        # and has no line of its own, named by its topic and with its reasons in words (never codes).
        items = join_bare([self.say(x) for x in self.limits])
        topics = {x.partition(':')[0].strip().lower() for x in self.limits}
        for o in self.observations.values():
            if o.get('team') not in (None, self.team) and self.team:
                continue
            if o.get('classification') != 'insufficient' and o.get('status') != 'suppressed':
                continue
            if o.get('id') in self.covered and (self.limits or self.quality):
                continue                     # the report's own line (or its data-quality note) says it
            topic = self.topic(o)
            if topic.lower() in topics:
                continue
            topics.add(topic.lower())
            words = self.reason_words(o)
            items.append(f'{topic}: not assessed; measured {_seconds(o.get("measuredSeconds"))}'
                         + ('; ' + '; '.join(words) if words else '') + '.')
        if items or self.quality:
            L.heading('Limitations', 2, 'limitations')
            L.para('What could not be assessed in this range, or limits what the report can say; none of it is '
                   'reported as a strength or a weakness.', 8.4, INK_2)
            if self.quality:
                L.para('The data-quality note of this footage is on page 1 (Read with care).', 8.4, INK_2)
            if items:
                L.bullets(items, size=8.8)
        L.heading('How the numbers were made', 2)
        ref = self.reference_name()
        method = ['Every statement comes from this run\'s tracking result and the existing PitchIQ analytics; '
                  'nothing is inferred about events that were not tracked.',
                  'Counts are of continuous situations: a situation that lasts is one occurrence, and a stretch the '
                  'camera or the registration did not cover is never bridged.',
                  'Metre values are estimated from the assumed pitch size and carry the registration error.',
                  'Screenshots are the exact frames of the source video; drawings over them are analysis overlays '
                  'added for this report.',
                  'No video is embedded in this PDF.']
        if ref:
            method.insert(1, f'Professional reference: {ref.partition("@")[0]}, one professional match, used as '
                             'a sanity range and not as a standard.')
        L.bullets(method, size=8.6)

    @staticmethod
    def _cov(key, v):
        k = str(key).lower()
        if _is_num(v):
            if 'share' in k:
                return value_text(v, 'share')
            if 'second' in k:
                return value_text(v, 's')
        return value_text(v) if not isinstance(v, str) else v

    def feedback(self):
        L = self.L
        fb = [f for f in (self.r.get('feedback') or []) if isinstance(f, dict)]
        if not fb:
            return
        L.heading('Expert feedback', 1, 'feedback', True)
        L.para('Notes recorded by a coach or analyst on this video. They do not change the measurements above. Notes '
               'are printed exactly as they were typed.', 8.4, INK_2)
        here, earlier = [], []
        for f in fb:
            o = f.get('observation') if isinstance(f.get('observation'), dict) else {}
            verdict = VERDICTS.get(f.get('verdict'), _humanize(f.get('verdict') or 'n/a'))
            # A coach's note is free text typed against the labels the page showed: it is never put through the
            # id -> display substitution (with labels A-05 -> A-07 the first PDFs printed "A-05 was the one out of
            # line" as "A-07 ..."). Non-WinAnsi characters still become '?'.
            note = f.get('text') if isinstance(f.get('text'), str) and f.get('text').strip() else '-'
            made = str(f.get('made') or '')[:10] or 'n/a'
            if self.this_run(f):
                obs_id = f.get('attached') if isinstance(f.get('attached'), str) else o.get('key')
                same = obs_id == o.get('key') or not o.get('key')
                if obs_id in self.findings:
                    finding = {'text': self.headline(obs_id) + ('' if same else ' (verdict given on an earlier '
                                                                 'version of this finding)'), 'dest': f'obs-{obs_id}'}
                else:
                    finding = self.say(o.get('headline')) or self.topic(self.observations.get(obs_id) or o)
                here.append([finding, verdict, note, made])
            else:
                # A verdict from another run is that run's: team letters and player ids are assigned per run, so it
                # is never shown as given on this report's finding, and its headline keeps the ids of that run.
                run = f'run {f["run"]}' if isinstance(f.get('run'), str) and f.get('run') else 'run not recorded'
                headline = o.get('headline') if isinstance(o.get('headline'), str) and o.get('headline') else \
                    self.topic(o)
                earlier.append([headline, verdict, note, f'{made}, {run}'])
        if here:
            L.table([('Finding', .36, 'left'), ('Verdict', .15, 'left'), ('Note', .33, 'left'),
                     ('Date', .16, 'left')], here)
        if earlier:
            L.heading('Feedback recorded on earlier runs of this video', 3)
            L.para('These verdicts were given on another run\'s report. Team letters and player numbers are assigned '
                   'per run, so they are listed with that run, not on this report\'s findings, and their findings '
                   'keep the player ids of that run.', 8.4, INK_2)
            L.table([('Finding', .32, 'left'), ('Verdict', .14, 'left'), ('Note', .30, 'left'),
                     ('Date and run', .24, 'left')], earlier)

    def this_run(self, entry):
        """True only when a feedback entry is known to belong to this report's run (store.attach_feedback's
        thisRun, else the same run name); an entry of unknown origin is treated as another run's."""
        if isinstance(entry.get('thisRun'), bool):
            return entry['thisRun']
        return bool(entry.get('run')) and entry.get('run') == self.r.get('run')

    def footers(self, doc):
        video = (self.r.get('video') or {}).get('name') or (self.r.get('video') or {}).get('key') or ''
        add_footers(doc, self.L, ' | '.join(x for x in ('PitchIQ coaching report', video, self.team_short) if x))


def stale_sentence(reasons, say=None):
    """The out-of-date banner's sentence. The reasons (store.py's texts) already end with a full stop: it is stripped
    before they are joined (critic N1: '...the tracking result changed.. Regenerate')."""
    say = say or (lambda s: s)
    shown = [say(x).strip().rstrip('.').strip() for x in (reasons or []) if isinstance(x, str) and x.strip()]
    shown = [x for x in shown if x]
    return ('The tracking result or its settings changed after this report was made' + (': ' + '; '.join(shown) if
            shown else '') + '. Regenerate the report in PitchIQ before relying on these findings.')


def fit_text(s, size, width, font='regular'):
    """s cut with '...' to at most width points (one line)."""
    s = str(s)
    if text_width(s, size, font) <= width:
        return s
    while s and text_width(s+'...', size, font) > width:
        s = s[:-1]
    return s.rstrip()+'...'


def add_footers(doc, L, left):
    """The footer of every page: the left text cut to the room the page number leaves (critic N2: an upload with a
    long name ran into 'Page 1 of 12')."""
    from . import render
    n = len(doc.pages)
    y = L.H-MARGINS[3]+26
    for i in range(n):
        number = f'Page {i+1} of {n}'
        room = L.width-text_width(number, 7)-14
        doc.add(i, [render.path([(L.left, y-10), (L.right, y-10)], stroke=RULE, width=.5),
                    render.text(L.left, y, fit_text(left, 7, room), 7, MUTED),
                    render.text(L.right, y, number, 7, MUTED, 'regular', 'right')])


# ---------------------------------------------------------------------------------------------------------------
# The coaching v2 report (schema 'report' version 2; DESIGN 9.2): the page's seven sections, for print

def is_v2(report):
    """True for a coaching v2 report (schema 'report': kind 'coaching-report', version 2)."""
    return isinstance(report, dict) and report.get('kind') == 'coaching-report' and report.get('version') == 2


V2_SECTIONS = (('overview', 'Match overview'), ('strengths', 'What went well'), ('needsWork', 'What needs work'),
               ('keyMoments', 'Key moments'), ('diagrams', 'Tactical diagrams'), ('drills', 'Training drills'),
               ('progression', 'Development progression'))
V2_MORE = (('sources', 'Sources'), ('feedback', 'Coach feedback'), ('about', 'About this report'))
# The page's words (coaching.js VERDICT_UI, TIER_UI, REASON_UI, GUARDRAIL_UI, DETECTOR_UI, MEASURE_UI, MASTERY_UI,
# TREND_UI, MASTERY_WHY, ALT_REASON_UI, DRILL_SOURCE_UI as of 2026-10-22, in ASCII): a coach moving between the page
# and the printout reads the same labels (test_coaching_pdf compares the codes the two cover).
V2_VERDICTS = {'strength': ('Strength', '#117b49', '#e1f4e9'), 'try_this': ('Try this', '#1f6f78', '#e2f2f3'),
               'work_on': ('Work on', '#8a5f12', '#faf0dc'),
               'review': ('Possible issue to review', '#3f6690', '#e6eef7'), 'describe': ('Context', INK_2, HEAD)}
# Fix round 2 (LEAD decision (b); report.REVIEW_PARTS / review_part, TEXT's request): a 'moments worth a look' review
# (assess limitation MOMENTS_CODE) is printed in its own part of WHAT NEEDS WORK after the possible issues, with this
# badge, WHAT TO LOOK FOR and its clear counts; it fails the evidence guardrail by construction (that note is not
# printed) and has no area, no drill and no objective (the page's coaching.js MOMENTS_UI).
MOMENTS_CODE = 'moments_worth_a_look'
V2_MOMENTS = ('Moments worth a look', '#3f6690', '#e6eef7')


def is_moments(f):
    """A 'moments worth a look' review (report.review_part(f) == 'moments')."""
    return isinstance(f, dict) and MOMENTS_CODE in (f.get('limitations') or ())


# Polish round (Oct 23; assess.LEVEL_FOCUS_LABEL, a test keeps the copies equal): with no objective from the match the
# NEXT DEVELOPMENT OBJECTIVE is the level's suggested focus from its guidance, "<label>: <check>. <guidance> Source:
# <ids>."; the printout shows the label on its own line, the focus, and each source's title with its id, as the page.
LEVEL_FOCUS_LABEL = "Suggested focus for this level - from the level's guidance, not from this match"
FOCUS_SOURCE_RE = re.compile(r'\s*Source: ([A-Z0-9][A-Za-z0-9-]*(?:, [A-Z0-9][A-Za-z0-9-]*)*)\.$')


def split_focus(text):
    """(label, focus, source ids) of a suggested-focus objective text, or None for any other objective text."""
    if not isinstance(text, str) or not text.startswith(LEVEL_FOCUS_LABEL+': '):
        return None
    body = text[len(LEVEL_FOCUS_LABEL)+2:]
    m = FOCUS_SOURCE_RE.search(body)
    ids = m.group(1).split(', ') if m else []
    return LEVEL_FOCUS_LABEL, (body[:m.start()] if m else body).strip(), ids


TIER_UI = {'world': 'Pitch-accurate', 'pinned': "From the players' feet", 'image': 'Boxes only'}
REASON_UI = {
    'phase_unknown': 'the ball and who had it could not be followed reliably on this footage',
    'possession_unavailable': 'who had the ball could not be estimated on this footage',
    'ball_unreliable': 'the ball was not tracked reliably enough', 'no_carrier': 'the player on the ball was not known',
    'out_of_view': 'the players involved were often out of the camera view',
    'team_incomplete': 'too few teammates were in view',
    'identity_uncertain': 'player identities were uncertain at those moments',
    'role_unconfirmed': "the players' roles are not confirmed (confirm them under Player roles in the coaching setup)",
    'scale_unknown': 'the field size is not known (enter it in the coaching setup)',
    'format_not_supported': 'this check does not apply to this match format',
    'registration_low': 'the camera-to-pitch registration was not reliable enough',
    'short_range': 'the range is too short',
    'recovering': 'players were running back towards their goal',
    'units_interleaved': "the team's lines overlapped, so its units could not be told apart",
    'near_change': 'it was close to a change of who had the ball', 'off': "not part of this level's checks",
    # fix round 1 (DETECT's new reasons, schema ABSTAIN_REASONS; critic S-7, N6, S-3), the page's words in ASCII
    'phase_absent': 'the phase this check needs did not happen in this range',
    'coach_marks_one_side': "you marked who had the ball for one team only: mark the other team's spells too to run "
                            "this check",
    'formation_changing': "the team's formation was changing at those moments",
    # fix round 3 (ASSESS; critic S-2, COACH F-14): enough play, too few of the check's situations; with a count the
    # PDF says how many (few_situations), as the page does
    'too_few_situations': 'too few of these situations happened in this range to judge'}
GUARDRAIL_UI = {'identity': 'who the player was', 'position': "the player's pitch position",
                'role': "the player's tactical role", 'context': 'the teammates around the moment',
                'phase': 'who had the ball', 'intent': 'whether the movement was intended',
                'alternative': 'a reachable better position', 'evidence': 'enough repeated evidence'}
DETECTOR_UI = {'pressure_cover': 'Going to the ball and helping a teammate',
               'weak_side_balance': 'Protecting the far side',
               'wide_defender_advance': 'Support behind an advancing wide defender',
               'behind_ball': 'Players ready behind the ball when attacking', 'defensive_spacing': 'Back-line spacing',
               'attacking_shape': 'Width and depth when attacking'}
SKILL_OF = {'pressure_cover': 'pressure_and_cover', 'behind_ball': 'someone_back_when_attacking',
            'wide_defender_advance': 'wide_defender_support', 'weak_side_balance': 'far_side_protection',
            'defensive_spacing': 'back_line_coordination', 'attacking_shape': 'spread_when_attacking'}
MEASURE_UI = {   # measurement -> (label, kind, first level shown)
    'd1M': ('Nearest player to the ball', 'm', 3), 'd1Median': ('Nearest player to the ball', 'm', 3),
    'coverDistM': ('Cover distance', 'm', 4), 'coverAngleDeg': ('Cover angle', 'deg', 6),
    'pressurePresent': ('Pressure on the ball', 'bool', 3), 'coverPresent': ('Cover behind', 'bool', 3),
    'coverPresentShare': ('Cover present', 'share', 5), 'goalSideShare': ('Goal-side', 'share', 5),
    'weakCount': ('Defenders on the far side', 'count', 3), 'centralCount': ('Defenders in the middle', 'count', 4),
    'openWidthM': ('Open width on the far side', 'm', 5), 'advanceM': ('Advance', 'm', 3),
    'behindN': ('Behind the ball', 'count', 3), 'minBehind': ('Needed behind the ball', 'count', 3),
    'channelOccupied': ('Channel behind covered', 'bool', 3), 'interchange': ('Positions swapped', 'bool', 3),
    'opponentsInChannel': ('Opponents in the channel', 'count', 4),
    'cbAdjusted': ('Centre-back moved across', 'bool', 5), 'nearCbShiftM': ('Near centre-back shift', 'm', 5),
    'dmDropped': ('Holding midfielder dropped', 'bool', 5),
    'threatN': ('Their players on our side of the ball', 'count', 5),
    'restMargin': ('Margin behind the ball', 'signed', 6),
    'dmCbSpacingM': ('Midfielder to centre-backs', 'm', 7), 'lateralGapM': ('Widest gap in the back line', 'm', 3),
    'gapBallSide': ('Gap on the ball side', 'bool', 4), 'opponentInGap': ('Opponent in the gap', 'bool', 4),
    'dmGapM': ('Defence to midfield gap', 'm', 5), 'lineStdM': ('Back-line unevenness', 'm', 5),
    # (pre-freeze, critic N-4 / FACTS F-11: contextual counts the opponents between the lines on the ball's half only)
    'dmGapOpponents': ("Opponents between defence and midfield on the ball's side", 'count', 4),
    'widthM': ('Width', 'm', 3), 'depthM': ('Depth', 'm', 3), 'channelsOccupied': ('Lanes occupied', 'count', 3),
    'widthShare': ('Width of the pitch used', 'share', 5), 'depthShare': ('Length of the pitch used', 'share', 5)}
# Back-line spacing cited for the gap between defence and midfield (failureMode 'dm_gap'): its own chips from the level
# its text states them (the distance from level 3, the opponents in it from level 4, as opponentInGap), and none of
# the lateral gap's yes/no chips. A level-4 'moments worth a look' review cited for a 12.2 m gap with an opponent in
# it showed only "Gap on the ball side no; Opponent in the gap no" (CODE L-4, round 2). dmGapOpponents is shown only
# then (the lateral mode never states it). The page has the same rule (coaching.js chipLevels).
DM_GAP_CHIPS = {'dmGapM': 3, 'dmGapOpponents': 4}
LATERAL_CHIPS = ('gapBallSide', 'opponentInGap')


def chip_levels(f):
    """{measurement: first level shown} for a finding's chips: MEASURE_UI's, changed by the cited failure mode
    (DM_GAP_CHIPS / LATERAL_CHIPS); None for a measurement not shown at all."""
    m = f.get('measurements') or {}
    out = {k: ui[2] for k, ui in MEASURE_UI.items()}
    dm = (f.get('observation') == 'defensive_spacing' and f.get('verdict') not in ('strength', 'describe')
          and (m.get('failureMode') == 'dm_gap'
               or (m.get('failureMode') is None and _is_num(m.get('dmGapM')) and not _is_num(m.get('lateralGapM')))))
    if dm:
        out.update(DM_GAP_CHIPS)
        out.update({k: None for k in LATERAL_CHIPS})
    else:
        out['dmGapOpponents'] = None
    return out
MASTERY_UI = {'NOT ENOUGH EVIDENCE': 'Not enough evidence yet', 'LEARNING': 'Learning', 'DEVELOPING': 'Developing',
              'CONSISTENT': 'Consistent', 'ADAPTABLE': 'Adaptable'}
TREND_UI = {'up': 'improving', 'down': 'fewer successes', 'flat': 'steady', 'unknown': '-'}
ALT_REASON_UI = {'no_reachable_fix': 'no teammate could have reached a better area in time',
                 'would_open_new_gap': 'every reachable area would have opened another gap',
                 'mover_unseen': 'the teammates who could help were not in view just before',
                 'context_missing': 'the players around were not all in view',
                 'scale_unknown': 'the field size is not known',
                 'guardrail_failed': 'the moment could not be confirmed',
                 'no_zone_for_detector': "this check describes the team's shape, not one position"}
DRILL_SOURCE_UI = {'adapted_from': 'Adapted from', 'inspired_by': 'Inspired by',
                   'pitchiq_original': 'PitchIQ original, informed by'}
PHASE_UI = {'own': 'when we had the ball', 'opp': 'when the other team had the ball',
            'dead': 'while the ball was out of play'}
FOOTAGE_UI = {'broadcast': 'Broadcast camera', 'amateur': 'Amateur camera', 'veo': 'Automatic camera (Veo type)'}
SCALE_UI = {'measured': 'measured by calibration', 'declared': 'entered in the coaching setup', 'assumed': 'assumed'}
CAUSE_UI = {'registration': 'mostly limited by the pitch registration',
            'identity': 'mostly limited by player identities',
            'out_of_view': 'mostly limited by players out of the camera view'}
FEDERATION_UI = {'auto': 'Automatic (U.S. Soccer formats by age; the stricter of U.S. Soccer and England on heading)',
                 'US': 'U.S. Soccer', 'England': 'England (The FA)', 'Canada': 'Canada Soccer'}
DRILL_FIELDS_V2 = (('trains', 'What it trains'), ('why', 'Why it was chosen'), ('easier', 'Make it easier'),
                   ('harder', 'Make it harder'), ('progress', 'What progress looks like next match'))
# Screenshots of key moments are zoomed on the people of the annotation (the PDF clips the whole frame, magnified,
# to the column: the JPEG bytes stay the exact frame, nothing is re-encoded). At the half-page column (243 pt) a
# broadcast player is about 2.5 pt wide in the full frame, under a millimetre on paper. The crop keeps at least
# ZOOM_MIN_SHARE of the frame's width (so the zoom is at most 1 / 0.45 = 2.2x, and a 1280 px frame still gives about
# 170 dpi), a margin of ZOOM_MARGIN of the width around the people (more above them, for the labels over their
# heads), and is used only when it magnifies by at least ZOOM_FROM.
ZOOM_MIN_SHARE = .45
ZOOM_MARGIN = .06
ZOOM_FROM = 1.3
ZOOM_WIDTH = 1280             # px of a zoomed screenshot (frames.WIDTHS); an unzoomed one uses FRAME_WIDTH
V2_EMPTY = {
    'strengths': ("No strength met this level's evidence bar in this range. A strength needs the behaviour actually "
                  'shown in separate moments, never only the absence of problems.'),
    'needsWork': "Nothing met this level's evidence bar for a work item in this range.",
    'keyMoments': 'No key moment in this range.',
    'diagrams': 'No diagram in this range.',
    'drills': 'No drill: there is no work item in this range and no next objective.'}   # no_drill_text says why


BUILD_ON_V2 = re.compile(r'To build on it[,:.]?\s+')   # how explain.py opens all 20 v2 build-on texts ("To build on it, ")


def without_lead(action):
    """A strength's action without its own opening "To build on it, ": the card's heading TO BUILD ON IT (the page's
    "To build on it.") already says it, and BC60 L5 read "To build on it. To build on it, hold the same far-side
    position deeper in our half." (CODE L-3, round 2). The rest starts with a capital."""
    m = BUILD_ON_V2.match(action or '')
    if m is None:
        return action
    rest = action[m.end():]
    return rest[:1].upper()+rest[1:]


def no_drill_text(report):
    """Why the drills section is empty, in the page's words (coaching.js noDrillText). Since the polish round a report
    without a work item offers one optional drill for the next objective, so "a drill is suggested only for a work item
    found in this range, never at random" misstated the rule wherever no drill fitted the objective (CODE L-5,
    round 2: SFS L4, objective "Back-line coordination")."""
    work = [f for f in (report or {}).get('needsWork') or [] if isinstance(f, dict)]
    nxt = (report or {}).get('nextObjective') if isinstance((report or {}).get('nextObjective'), dict) else None
    if work:
        return "No drill: PitchIQ has none that fits the work items in this range for this team's age, level and format."
    if nxt is None or not (isinstance(nxt.get('text'), str) and nxt['text'].strip()):
        return 'No drill: there is no work item in this range and no next objective.'
    if not nxt.get('skill'):
        return ('No drill: there is no work item in this range, and the next objective is a general one with no drill '
                'tied to it.')
    return ("No drill: there is no work item in this range, and PitchIQ has none that fits the next objective for this "
            "team's age, level, format and range.")


ORIGIN_NAMED = re.compile(r"from this match|from the level|this match's|level's guidance", re.I)


# Pre-freeze (Oct 24, critic S-3: DRILLS / PAGE; the page's OPTIONAL_LABELS / optionalLabel, a test keeps them equal):
# an optional drill read "not from this match's evidence" also when the next objective was chosen from this match's
# checks (a skill, not from a work item, not the level's suggested focus: report.S8_LABEL), where it was false.
OPTIONAL_LABELS = {'match': "Optional: for the next objective, chosen from this match's checks (not a work item).",
                   'level': "Optional: for the next objective, not from this match's evidence."}


def optional_label(report):
    """The line an optional drill prints in place of its finding (OPTIONAL_LABELS)."""
    n = report.get('nextObjective') if isinstance(report.get('nextObjective'), dict) else {}
    if n and n.get('skill') is not None and not n.get('fromMatch') and split_focus(n.get('text')) is None:
        return OPTIONAL_LABELS['match']
    return OPTIONAL_LABELS['level']


def origin_line(n, finding):
    """The line the next objective's box adds under its text (the page's originLine): the link's lead for an objective
    from a shown work item, the plain "From the level's objectives, not from this match." only for a skill not from
    this match whose text names no origin itself (ORIGIN_NAMED), else none. Before fix round 3 an objective from this
    match with its finding not shown, or a text worded "chosen from this match's checks", got "not from this match"."""
    if n.get('fromMatch'):
        return 'From this match: see the finding linked below.' if finding else None
    if n.get('skill') is None or ORIGIN_NAMED.search(str(n.get('text') or '')):
        return None
    return "From the level's objectives, not from this match."


def _pct(v):
    return f'{int(math.floor(float(v)*100+.5))}%'


def _secs(v):
    if not _is_num(v):
        return '-'
    return f'{v/60:.1f} min' if v >= 90 else f'{v:.1f} s'


def _plural(n, word):
    return word if n == 1 else word+'s'


def _title(s):
    return ' '.join(w[:1].upper()+w[1:].lower() for w in str(s or '').split())


def _dicts(v):
    return [x for x in v if isinstance(x, dict)] if isinstance(v, (list, tuple)) else []


def sanitize_v2(report):
    """A shallow copy of a v2 report whose lists and maps have the schema's types, so a malformed field leaves its
    part of the PDF empty instead of failing the export (the report is validated when it is made; the PDF still
    never trusts it). Values are not changed."""
    r = _map(report)
    for key in ('team', 'teamName', 'run', 'generated'):
        v = r.get(key)
        r[key] = v if isinstance(v, str) else None
    for key in ('strengths', 'needsWork', 'reviews', 'described', 'keyMoments', 'diagrams', 'alternatives',
                'noAlternatives', 'drills', 'sources', 'withheld'):
        r[key] = _dicts(r.get(key))
    for key in ('video', 'range', 'segment', 'profile', 'overview', 'engine', 'fingerprints', 'nextObjective'):
        r[key] = _map(r.get(key))
    r['progression'] = r.get('progression') if isinstance(r.get('progression'), dict) else None
    ov = r['overview']
    for key in ('notices', 'formations', 'detectors', 'unavailable'):
        ov[key] = _dicts(ov.get(key))
    for key in ('footage', 'dataQuality', 'scale', 'possession'):
        ov[key] = _map(ov.get(key))
    p = r['profile']
    for key in ('notices', 'objectives', 'principles', 'notExpected'):
        p[key] = _dicts(p.get(key))
    for key in ('format', 'register', 'skills', 'safety', 'evidence'):
        p[key] = _map(p.get(key))
    for key in ('strengths', 'needsWork', 'reviews', 'described'):
        r[key] = [dict(f, text=_map(f.get('text')), timestamps=_dicts(f.get('timestamps')),
                       aggregate=_map(f.get('aggregate')), measurements=_map(f.get('measurements')),
                       guardrails=_map(f.get('guardrails')), context=_map(f.get('context')),
                       limitations=[c for c in _items(f.get('limitations')) if isinstance(c, str)],
                       drills=[d for d in _items(f.get('drills')) if isinstance(d, str)])
                  for f in r[key] if isinstance(f.get('id'), str)]
    r['keyMoments'] = [dict(e, diagram=e.get('diagram') if isinstance(e.get('diagram'), dict) else None,
                            videoOverlay=e.get('videoOverlay') if isinstance(e.get('videoOverlay'), dict) else None,
                            annotation=e.get('annotation') if isinstance(e.get('annotation'), dict) else None,
                            players=[x for x in _items(e.get('players')) if isinstance(x, str)])
                       for e in r['keyMoments'] if isinstance(e.get('id'), str)]
    r['diagrams'] = [d for d in r['diagrams'] if isinstance(d.get('diagram'), dict)]
    r['drills'] = [dict(d, fields=_map(d.get('fields'))) for d in r['drills'] if isinstance(d.get('drill'), str)]
    return r


def focus_box(overlay):
    """The pixel box of what a v2 overlay is about (its people with their boxes, relation ends, the suggested area,
    the movement arrow, a measure), or None. The channel is left out: it is context, often half the pitch."""
    W, H = overlay.get('width'), overlay.get('height')
    if not (_is_num(W) and _is_num(H) and W > 0 and H > 0):
        return None
    pts = []
    for el in overlay.get('elements') or []:
        if not isinstance(el, dict):
            continue
        t = el.get('type')
        if t in ('focus', 'support') or (t == 'player' and el.get('emphasis')):
            pts.append((el.get('x'), el.get('y')))
            b = el.get('box')
            if isinstance(b, (list, tuple)) and len(b) == 4 and all(_is_num(v) for v in b):
                pts += [(b[0], b[1]), (b[0]+b[2], b[1]+b[3])]
        elif t == 'relation':
            pts += [el.get('pa'), el.get('pb')]
        elif t == 'suggestedZone':
            pts += list(el.get('polygon') or [])
        elif t == 'moveArrow':
            pts += [el.get('from'), el.get('to')]
        elif t == 'measure':
            pts += [el.get('a'), el.get('b')]
    pts = [(float(p[0]), float(p[1])) for p in pts if isinstance(p, (list, tuple)) and len(p) >= 2
           and _is_num(p[0]) and _is_num(p[1]) and -.05*W <= p[0] <= 1.05*W and -.05*H <= p[1] <= 1.05*H]
    if not pts:
        return None
    return (min(p[0] for p in pts), min(p[1] for p in pts), max(p[0] for p in pts), max(p[1] for p in pts))


def zoom_crop(box, W, H):
    """(x0, y0, w, h, zoom) of the frame region to show for a focus box (source pixels), or None when the zoom would
    be under ZOOM_FROM (see ZOOM_MIN_SHARE)."""
    if box is None or not (_is_num(W) and _is_num(H) and W > 0 and H > 0):
        return None
    m = ZOOM_MARGIN*W
    x0, y0, x1, y1 = box[0]-m, box[1]-1.8*m, box[2]+m, box[3]+.6*m
    cw = max(ZOOM_MIN_SHARE*W, x1-x0, (y1-y0)*W/H)
    if cw*ZOOM_FROM > W:
        return None
    ch = cw*H/W
    cx0 = min(max((x0+x1)/2-cw/2, 0.0), W-cw)
    cy0 = min(max((y0+y1)/2-ch/2, 0.0), H-ch)
    return (cx0, cy0, cw, ch, W/cw)


class _ReportV2:
    """Lays out one coaching v2 report on a Document: the page's seven sections (coaching.js renderV2) in its words,
    for print. Everything printed is the report's own text and numbers (validated when the report was made) or a
    fixed label of this module; the key moments add the exact source frame with the annotation drawn over it."""

    def __init__(self, report, frame_provider, options):
        from . import render
        self.render = render
        self.r = r = sanitize_v2(report)
        self.frames = frame_provider
        self.o = dict(options or {})
        self.labels = {str(k): str(v) for k, v in (self.o.get('labels') or {}).items()}
        self.colours = render.team_colours(self.o.get('colours'))
        self.team = r['team'] if r['team'] in ('A', 'B') else 'A'
        self.other = 'B' if self.team == 'A' else 'A'
        name = r.get('teamName')
        self.team_short = name or f'Team {self.team}'
        self.team_long = name or f'Team {self.team} (as labelled in this run)'
        self.team_names = {self.team: self.team_short, self.other: f'Team {self.other}'}
        self.p = r['profile']
        lv = self.p.get('level')
        self.level = lv if isinstance(lv, int) and not isinstance(lv, bool) else None
        reg = self.p.get('register') or {}
        self.numbers = reg.get('numbers') if isinstance(reg.get('numbers'), str) else 'metres'
        self.findings = {f['id']: f for key in ('strengths', 'needsWork', 'reviews', 'described') for f in r[key]}
        self.moments = sorted(r['keyMoments'], key=lambda e: (e.get('time') if _is_num(e.get('time')) else 0.0))
        self.ev = {e['id']: e for e in self.moments}
        self.number = {e['id']: k+1 for k, e in enumerate(self.moments)}
        self.alts = {a.get('id'): a for a in r['alternatives']}
        fb = self.o.get('feedback')
        fb = fb if isinstance(fb, (list, tuple)) else report.get('feedback') if isinstance(report, dict) else None
        self.feedback = _dicts(fb)
        self.adjust = {}
        for f in self.feedback:
            adj = f.get('adjustment')
            if f.get('thisRun') and isinstance(adj, dict) and isinstance(adj.get('evidence'), str):
                self.adjust.setdefault(adj['evidence'], []).append({'polygon': adj.get('polygon'),
                                                                    'label': render.COACH_AREA})
        self.url = self.o.get('playback_url')
        stale = self.o.get('stale')
        self.stale = ([stale] if isinstance(stale, str) else list(stale) if isinstance(stale, (list, tuple))
                      else [] if stale else None)
        self.shots_left = int(self.o.get('max_screenshots', MAX_SCREENSHOTS))
        self.frame_width = int(self.o.get('image_width', FRAME_WIDTH))
        self.zoom = bool(self.o.get('zoom', True))
        self.missing, self.skipped, self.drawn_evidence = [], [], set()
        self.screenshots = self.overlays = self.zoomed = 0
        self.resolve = None
        if self.labels and self.o.get('substitute_text', True):
            from . import language                   # the package's one id -> display rule (one pass, no chains)
            self.resolve = language.resolve_players
        self.pal_print = render.print_palette(self.colours)
        self.pal_video = render.video_palette(self.colours)
        self.sources = {s.get('id'): s for s in r['sources']}
        self.library = None
        self.plan = {}
        self.naming_ok = (self.p.get('evidence') or {}).get('naming') in ('evidence_only', 'yes')
        self.unnamed = self._unnamed()

    # -- helpers ----------------------------------------------------------------------------------------------
    def say(self, s):
        if s is None:
            return None
        s = str(s)
        return self.resolve(s, self.labels) if self.resolve else s

    def skill_name(self, detector):
        name = (self.p.get('skills') or {}).get(SKILL_OF.get(detector))
        return name if isinstance(name, str) and name else DETECTOR_UI.get(detector) or _humanize(detector or 'check')

    def phase_words(self, f):
        ph = (f.get('context') or {}).get('phase') if isinstance(f, dict) else None
        if not isinstance(ph, dict) or ph.get('state') not in PHASE_UI:
            return None
        source = (f.get('measurements') or {}).get('phaseSource')
        how = ('marked by the coach' if ph.get('estimated') is False or source == 'coach' else
               'estimated, partly marked by the coach' if source == 'mixed' else 'estimated')
        return f'{PHASE_UI[ph["state"]]} ({how})'

    def coach_marked(self):
        """True when any finding (or the possession estimate) rests on possession marked by the coach."""
        if any((f.get('measurements') or {}).get('phaseSource') in ('coach', 'mixed') for f in self.findings.values()):
            return True
        cav = (self.r['overview'].get('possession') or {}).get('caveat')
        return isinstance(cav, str) and 'coach-marked' in cav

    def named(self, ev):
        """False when a moment may name nobody: then the PDF prints no player label, no 'Players:' line and no
        mover id for it. evidence.py's rule (focus.label is True only where the profile names players as
        evidence_only / yes, the finding's identity was confirmed, the check is not team-level only and the finding
        is not a description), read per finding, so a finding's moments without a drawn focus follow the one
        with it."""
        for spec in (ev.get('diagram'), ev.get('videoOverlay')):
            for el in (spec or {}).get('elements') or []:
                if isinstance(el, dict) and el.get('type') == 'focus' and el.get('label') is False:
                    return False
        f = self.findings.get(ev.get('finding'))
        return self.naming_ok if f is None else f['id'] not in self.unnamed

    def _unnamed(self):
        """The findings whose moments name nobody (see named())."""
        out = set()
        dets = self.p.get('detectors') if isinstance(self.p.get('detectors'), dict) else {}
        for f in self.findings.values():
            params = (dets.get(f.get('observation')) or {}).get('params') if isinstance(dets.get(f.get('observation')),
                                                                                       dict) else None
            if not self.naming_ok or f.get('verdict') == 'describe' \
                    or (f.get('guardrails') or {}).get('identity') == 'fail' \
                    or (isinstance(params, dict) and params.get('teamLevelOnly')):
                out.add(f['id'])
        for ev in self.moments:
            for spec in (ev.get('diagram'), ev.get('videoOverlay')):
                for el in (spec or {}).get('elements') or []:
                    if isinstance(el, dict) and el.get('type') == 'focus' and el.get('label') is False:
                        out.add(ev.get('finding'))
        return out

    def moment_ref(self, ev_id):
        ev = self.ev.get(ev_id)
        return (f'E{self.number[ev_id]} {clock(ev.get("time"))}', f'ev-{ev_id}') if ev else None

    def drill_record(self, did):
        if self.library is None:
            try:
                from . import drills as D
                self.library = {d.get('drill_id'): d for d in D.load_library().get('drills') or []
                                if isinstance(d, dict)}
            except Exception:                                   # the PDF never fails on the library
                self.library = {}
        return self.library.get(did)

    def style_name(self, style):
        if style in (None, 'unknown'):
            return 'Unknown: no intended style is assumed'
        try:
            from . import profiles
            for s in profiles.catalog().get('styles') or []:
                if s.get('id') == style and isinstance(s.get('name'), str):
                    return s['name']
        except Exception:
            pass
        return _humanize(style)

    def playback(self, ev):
        u = self.url
        if not u or not isinstance(ev, dict):
            return None
        try:
            link = u(ev) if callable(u) else str(u).format(
                frame=int(ev.get('frame')), seek=f'{float(ev.get("seek")):.4f}', time=f'{float(ev.get("time")):.3f}',
                id=ev.get('id', ''), video=(self.r.get('video') or {}).get('key') or '', run=self.r.get('run') or '')
        except (TypeError, ValueError, KeyError, IndexError, AttributeError):
            return None
        return safe_uri(link)

    def kv(self, rows, key_width=118.0, size=8.8):
        L, render = self.L, self.render
        leading = size*1.4
        for key, value in rows:
            lines = wrap(self.say(value), size, L.width-key_width) or ['']
            L.ensure(min(len(lines), 6)*leading)
            L.add([render.text(L.left, L.y+(leading+size*.7)/2, key, size, MUTED, 'bold')])
            for k, line in enumerate(lines):
                if k:
                    L.ensure(leading)
                L.text_line(L.left+key_width, line, size, INK, leading=leading)
                L.y += leading
            L.add([render.path([(L.left, L.y+1), (L.right, L.y+1)], stroke=RULE, width=.35)])
            L.y += 3
        L.y += 8

    def chips(self, lead, items, size=7.8):
        """A line of linked items ('E1 10:10.00', a drill's name) after a lead word, wrapped; items: [(text, dest)]."""
        L, render = self.L, self.render
        items = [(t, d) for t, d in items if t]
        if not items:
            return
        leading = size*1.5
        L.ensure(leading)
        x = L.left
        if lead:
            L.text_line(x, lead, size, INK_2, 'bold', leading=leading)
            x += text_width(lead, size, 'bold')+5
        for t, dest in items:
            w = text_width(t, size)+8
            if x+w > L.right and x > L.left+1:
                L.y += leading
                L.ensure(leading)
                x = L.left+12
            L.add([render.rect(x, L.y+1.2, w, leading-2.2, fill=TINT),
                   render.text(x+4, L.y+(leading+size*.7)/2, t, size, BRAND_DARK if dest else INK_2)])
            if dest:
                L.add([dict({'op': 'link', 'x': x, 'y': L.y, 'w': w, 'h': leading}, **dest)])
            x += w+4
        L.y += leading+3

    def badge(self, verdict, x=None, moments=False):
        """The verdict chip (the page's badge words) at the cursor; returns its width."""
        L, render = self.L, self.render
        label, fg, bg = V2_MOMENTS if moments else V2_VERDICTS.get(verdict, ('Finding', INK_2, HEAD))
        bw = text_width(label.upper(), 6.6, 'bold')+10
        x = L.left if x is None else x
        L.add([render.rect(x, L.y, bw, 11, fill=bg), render.text(x+5, L.y+8, label.upper(), 6.6, fg, 'bold')])
        return bw

    def section_heading(self, key, number, title, count=None, keep=0.0):
        t = f'{number}. {title}' if number else title
        self.L.heading(t+(f' ({count})' if count is not None else ''), 1, key, True, keep=keep)

    # -- the parts ------------------------------------------------------------------------------------------
    def layout(self, doc):
        rng = self.r['range']
        header = f'{self.team_short}  |  {clock(rng.get("t0"))} - {clock(rng.get("t1"))}'
        if self.level is not None:
            header += f'  |  level {self.level}'
        if self.stale is not None:
            header = 'OUT OF DATE  |  '+header
        self.L = _Layout(doc, header)
        self.plan_shots()
        for part in (self.cover, self.overview, self.went_well, self.needs_work, self.key_moments, self.diagrams,
                     self.drills, self.progression, self.sources_section, self.feedback_section, self.about):
            part()
        video = self.r['video'].get('name') or self.r['video'].get('key') or ''
        add_footers(doc, self.L, ' | '.join(x for x in ('PitchIQ coaching report', video, self.team_short) if x))

    def plan_shots(self):
        """Which key moments get a screenshot (the first max_screenshots, in time order), at which width and zoom;
        then one sorted decoding pass per width (frames.py prefetch)."""
        wanted = {}
        for ev in self.moments:
            frame = ev.get('frame')
            if not isinstance(frame, int) or isinstance(frame, bool) or frame < 0:
                continue
            if len(self.plan) >= self.shots_left:
                break
            ov = ev.get('videoOverlay')
            crop = zoom_crop(focus_box(ov), ov.get('width'), ov.get('height')) if self.zoom and ov else None
            width = ZOOM_WIDTH if crop else self.frame_width
            self.plan[ev['id']] = (width, crop)
            wanted.setdefault(width, set()).add(frame)
        if self.frames is None or not hasattr(self.frames, 'prefetch'):
            return
        for width, frames in sorted(wanted.items()):
            try:
                self.frames.prefetch(sorted(frames), [width])
            except Exception:                                  # the per-frame calls report what is missing
                pass

    def frame_jpeg(self, ev):
        frame = ev.get('frame')
        if self.frames is None:
            return None, None, 'no source video was available when this PDF was made'
        if ev['id'] not in self.plan:
            if not isinstance(frame, int) or isinstance(frame, bool) or frame < 0:
                return None, None, 'the moment has no frame index'
            return None, None, 'screenshot left out to keep the file small (see the timestamp)'
        width, crop = self.plan[ev['id']]
        data = None
        for w in ((width, self.frame_width) if width != self.frame_width else (width,)):
            try:
                data = self.frames(frame, w)
            except Exception:                                   # a broken video must not break the report
                data = None
            if data:
                if w != width:
                    crop = None                                 # the small frame is not magnified
                break
        if not data:
            self.missing.append(frame)
            return None, None, 'the frame could not be read from the source video'
        try:
            jpeg_info(data)
        except ValueError:
            self.missing.append(frame)
            return None, None, 'the frame provider returned no JPEG'
        return data, crop, None

    def cover(self):
        L, r, render = self.L, self.r, self.render
        L.y += 4
        L.kicker('PitchIQ')
        L.text_line(L.left, 'COACHING REPORT', 21, INK, 'bold', leading=26)
        L.y += 26
        L.text_line(L.left, self.say(self.team_long), 12.5, INK_2, 'regular', leading=18)
        L.y += 24
        if self.stale is not None:
            L.box([('OUT OF DATE', 8.5, 'bold', '#8a5f12'),
                   (stale_sentence(self.stale, self.say), 8.6, 'regular', INK)],
                  fill='#faf0dc', accent='#d9a961')
        self.kv(self.profile_rows())
        n_mw = sum(1 for f in r['reviews'] if is_moments(f))
        tiles = (('strengths', len(r['strengths']), ('strength', 'strengths')),
                 ('needsWork', len(r['needsWork']), ('thing to try', 'things to try') if self.level in (1, 2)
                  else ('work item', 'work items')),
                 ('needsWork', len(r['reviews'])-n_mw, ('issue to review', 'issues to review')))
        if n_mw:
            tiles += (('needsWork', n_mw, ('moment worth a look', 'moments worth a look')),)
        tiles += (('keyMoments', len(self.moments), ('key moment', 'key moments')),)
        L.ensure(46)
        cw = (L.width-(len(tiles)-1)*8)/len(tiles)
        x = L.left
        for anchor, n, (one, many) in tiles:
            L.add([render.rect(x, L.y, cw, 38, fill=TINT), render.text(x+8, L.y+19, str(n), 15, BRAND_DARK, 'bold'),
                   render.text(x+8, L.y+31, one if n == 1 else many, 7, INK_2),
                   {'op': 'link', 'x': x, 'y': L.y, 'w': cw, 'h': 38, 'dest': anchor}])
            x += cw+8
        L.y += 50
        text = r['overview'].get('text')
        if isinstance(text, str) and text.strip():
            L.box([('In short', 8.5, 'bold', INK), (self.say(text), 9.2, 'regular', INK)])
        L.heading('Contents', 2)
        entries = [(f'{k}. {t}', key) for k, (key, t) in enumerate(V2_SECTIONS, 1)]
        entries += [(t, key) for key, t in V2_MORE if key != 'feedback' or self.feedback]
        half = (len(entries)+1)//2
        leading = 12.5
        L.ensure(half*leading+6)
        top = L.y
        for k, (t, anchor) in enumerate(entries):
            x = L.left+(0 if k < half else L.width/2)
            y = top+(k if k < half else k-half)*leading
            L.add([render.text(x, y+9, t, 9, BRAND_DARK),
                   {'op': 'link', 'x': x, 'y': y, 'w': text_width(t, 9), 'h': leading, 'dest': anchor}])
        L.y = top+half*leading+8
        L.box([('How to read this report', 8.5, 'bold', INK), (self.how_to_read(), 8.2, 'regular', INK_2)])

    def how_to_read(self):
        r = self.r
        zones = any(isinstance(el, dict) and el.get('type') == 'suggestedZone'
                    for spec in [e.get('diagram') for e in self.moments]+[d['diagram'] for d in r['diagrams']]
                    if isinstance(spec, dict) for el in spec.get('elements') or [])
        out = ['Every statement comes from the tracking of this video, read for the age group and level above; '
               "PitchIQ's checks and thresholds are its own proposals: no coach has validated them and no "
               'federation has reviewed them.',
               'Times are source-video times (mm:ss.ss), with the offset into the analysed segment.']
        if self.moments:
            out.append('Each key moment shows the exact source frame with the annotation drawn over it, as far as '
                       "the frame's pitch registration allows, and the same moment as a top-down diagram.")
        if zones:     # critic B4 / M2: the dotted area is described only when the report holds one
            out.append('A dotted area is a possible better position, with a dashed arrow for the movement towards '
                       'it: an illustration with one player moved and everyone else kept where they were, never an '
                       'exact spot.')
        if self.coach_marked():
            out.append('Where you marked who had the ball, your marks are used and labelled as marked by the coach.')
        scale = r['overview'].get('scale') or {}
        if scale.get('source', 'assumed') == 'assumed':
            out.append('Distances are estimates from an assumed pitch size.')
        out.append('This PDF holds no video: use the timestamps to find each moment.')
        return ' '.join(out)

    def profile_rows(self):
        r, p = self.r, self.p
        rows = []
        video = r['video']
        rows.append(('Video', video.get('name') or video.get('key') or 'n/a'))
        if r.get('run'):
            rows.append(('Run', r['run']))
        rng, seg = r['range'], r['segment']
        if _is_num(rng.get('t0')) and _is_num(rng.get('t1')):
            span = rng.get('seconds') if _is_num(rng.get('seconds')) else rng['t1']-rng['t0']
            tail = (', the whole analysed segment' if rng.get('whole') else
                    f'; the analysed segment runs {clock(seg.get("t0"))} - {clock(seg.get("t1"))}'
                    if _is_num(seg.get('t0')) and _is_num(seg.get('t1')) else '')
            rows.append(('Analysed range', f'{clock(rng["t0"])} - {clock(rng["t1"])} of the source video '
                                           f'({_secs(span)}{tail})'))
        lv = f'level {self.level} {_title(p.get("levelName") or "")}'.strip() if self.level is not None else None
        prof = ' - '.join(x for x in (p.get('ageGroup'), lv) if x)
        if p.get('band'):
            prof += f' ({p["band"]})'
        rows.append(('Coaching profile', prof or 'n/a'))
        fm = p.get('format') or {}
        if fm.get('name'):
            rows.append(('Match format', f'{fm["name"]}{" without goalkeeper" if fm.get("goalkeeper") is False else ""}'
                                         + (f' - {fm["outfield"]} outfield players' if _is_num(fm.get('outfield'))
                                            else '')))
        rows.append(('Intended formation', f'{p["formation"]} (a starting model, not where every player must stand)'
                     if p.get('formation') else 'Other / not sure'))
        rows.append(('Style of play', self.style_name(p.get('style'))))
        if p.get('federation') in FEDERATION_UI:
            rows.append(('Federation guidance', FEDERATION_UI[p['federation']]))
        fs = p.get('field') if isinstance(p.get('field'), dict) else None
        if fs and _is_num(fs.get('length')) and _is_num(fs.get('width')):
            rows.append(('Field size', f'{fs["length"]:g} x {fs["width"]:g} m (entered in the coaching setup)'))
        if r.get('generated'):
            rows.append(('Generated', str(r['generated']).replace('T', ' ')[:19]))
        return rows

    def overview(self):
        L, r = self.L, self.r
        ov, p = r['overview'], self.p
        self.section_heading('overview', 1, 'Match overview', keep=60)
        rows = [('Footage', self.footage_text(ov.get('footage') or {})),
                ('Pitch scale', self.scale_text(ov.get('scale') or {})),
                ('Who had the ball', self.possession_text(ov.get('possession') or {}))]
        dq = self.quality_text(ov.get('dataQuality') or {})
        if dq:
            rows.append(('Data quality', dq))
        self.kv(rows, key_width=110)
        po = ov.get('possession') or {}
        if po.get('available') and isinstance(po.get('caveat'), str) and po['caveat'].strip():
            L.para(self.say(po['caveat']), 8.3, INK_2, after=6)
        seen, notices = set(), []
        for n in ov['notices']+p['notices']:
            if isinstance(n.get('text'), str) and n['text'].strip() and n.get('code') not in seen:
                seen.add(n.get('code'))
                notices.append(self.say(n['text']))
        if notices:
            L.heading('Notes on this profile', 3)
            L.bullets(notices, size=8.6)
        periods = [x for x in ov['formations'] if _is_num(x.get('t0')) and _is_num(x.get('t1'))]
        if periods:
            L.heading('Shape measured (described, not judged)', 3)
            L.bullets([f'{clock(x["t0"])} - {clock(x["t1"])}: {x.get("label") or "no clear shape"}'
                       + (f' ({x["confidence"]})' if isinstance(x.get('confidence'), str) else '') for x in periods],
                      size=8.6)
        dets = [d for d in ov['detectors'] if isinstance(d.get('detector'), str)]
        if dets:
            L.heading('What was checked', 3)
            L.bullets([f'{self.skill_name(d["detector"])}: {self.checked_text(d)}' for d in dets], size=8.6)
        un = [x for x in ov['unavailable'] if isinstance(x.get('category'), str) and isinstance(x.get('reason'), str)]
        if un:
            L.heading('Not measured by PitchIQ', 3)
            L.bullets([f'{self.category_name(x["category"])}: {self.say(x["reason"])}'
                       + (f' What would make it possible: {self.say(x["unlock"])}' if isinstance(x.get('unlock'), str)
                          and x['unlock'].strip() else '') for x in un], size=8.4)
        if r['described']:
            L.heading('Context from this match', 3, keep=60)
            for f in r['described']:
                self.card(f)
        held = len(r['withheld'])
        if held:
            L.para(f'{held} {"finding was" if held == 1 else "findings were"} held back: '
                   f'{"its" if held == 1 else "their"} wording could not be checked against the measurements.',
                   8.3, INK_2, 'oblique')

    @staticmethod
    def category_name(code):
        """The name of a category PitchIQ does not measure (profiles.json unavailable.categories), else the code
        in words."""
        try:
            from . import profiles
            name = ((profiles.load().get('unavailable') or {}).get('categories') or {}).get(code, {}).get('category')
        except Exception:
            name = None
        return name if isinstance(name, str) and name.strip() else _humanize(code)

    def guidance(self, g):
        tag = ' (PitchIQ proposal)' if g.get('tag') == 'PITCHIQ-PROPOSAL' else ''
        ids = [s for s in _items(g.get('sources')) if isinstance(s, str)]
        return self.say(g['text'].strip())+tag+(f' [{", ".join(ids)}]' if ids else '')

    def checked_text(self, d):
        if d.get('status') == 'assessed':
            n = d.get('opportunities')
            if self.numbers == 'none' or not isinstance(n, int):
                return 'checked'
            return (f'checked in {n} {_plural(n, "moment")}'
                    + (f' ({_secs(d["eligibleSeconds"])} usable)' if _is_num(d.get('eligibleSeconds')) else ''))
        if d.get('status') == 'off':
            return REASON_UI['off']
        why = d.get('reason')
        if why == 'coach_marks_one_side':     # the page's words when it knows whose spells were marked (CODE M-2)
            ov = self.r.get('overview') or {}
            side = self.marked_side(ov.get('possession') or {}, ov.get('detectors') or [])
            if side in ('A', 'B'):
                return (f"not assessed: you marked only Team {side}'s possession: mark Team "
                        f"{'B' if side == 'A' else 'A'}'s spells too to run this check")
        n = d.get('opportunities')
        if (why == 'too_few_situations' and self.numbers != 'none' and isinstance(n, int) and not isinstance(n, bool)
                and n > 0):
            return f'not assessed: only {n} {_plural(n, "situation")} of this kind happened in this range, too few to judge'
        return 'not assessed: '+(REASON_UI.get(why) or (_humanize(why).lower() if why else 'not enough data'))

    @staticmethod
    def footage_text(fo):
        kind = FOOTAGE_UI.get(fo.get('kind'), 'Camera type not known')
        more = []
        if _is_num(fo.get('smallSceneShare')) and fo['smallSceneShare'] >= .2:
            more.append(f'players appear small in {_pct(fo["smallSceneShare"])} of it')
        if fo.get('keyless'):
            more.append('no pitch keyframes were set')
        if fo.get('lensBent'):
            more.append('a curved lens')
        return kind+(': '+', '.join(more) if more else '')

    @staticmethod
    def scale_text(sc):
        if not (_is_num(sc.get('lengthM')) and _is_num(sc.get('widthM'))):
            return 'not known'
        how = SCALE_UI.get(sc.get('source'), 'assumed')
        tail = ('; distances between players are not judged (enter the field size in the setup)'
                if sc.get('metresUsable') is False else '; distances are estimates' if sc.get('source') == 'assumed'
                else '')
        return f'{sc["lengthM"]:.0f} x {sc["widthM"]:.0f} m, {how}{tail}'

    def possession_source(self, po):
        """Where the shares come from (the page's possessionSource): the report's own word (PossessionOverview.source),
        else the run's coach spans (option coach_spans, the answer's coachSpans) and the caveat report.py writes when
        only the coach's marks name the phase."""
        if po.get('source') in ('engine', 'coach', 'mixed'):
            return po['source']
        if not self.o.get('coach_spans'):
            return 'engine'
        return 'coach' if re.search('could not follow', str(po.get('caveat') or ''), re.I) else 'mixed'

    def marked_side(self, po, dets=()):
        """Whose possession the coach marked (the page's markedSide): option marked ('A', 'B', 'both', from the run's
        stored spans, store.marked_teams), else from the report itself: shares of 100% to 0% from the coach's marks
        alone, or a check left out for coach_marks_one_side; None when nothing says."""
        if self.o.get('marked') in ('A', 'B', 'both'):
            return self.o['marked']
        rs = po.get('roughShares') if isinstance(po.get('roughShares'), dict) else None
        if not self.o.get('coach_spans') or not rs or not (_is_num(rs.get('A')) and _is_num(rs.get('B'))):
            return None
        full = 'A' if rs['A'] >= .999 else 'B' if rs['B'] >= .999 else None
        told = any(isinstance(d, dict) and d.get('reason') == 'coach_marks_one_side' for d in dets)
        return full if full and (told or self.possession_source(po) == 'coach') else None

    def possession_text(self, po):
        """Who had the ball, said with where it comes from, in the page's words (coaching.js possessionText; CODE M-2,
        round 2: the PDF said "estimated" of the coach's own marks: "(estimated from 42% of the time)" with marks of
        both teams on the AM copy, "Estimated from 31% of the time" where the page gave no share for Team A's marks
        alone, and "(estimated from 100% of the time)" with marks over PitchIQ's estimate on BC60)."""
        if not po.get('available'):
            why = po.get('reason') if isinstance(po.get('reason'), str) and po['reason'].strip() else None
            return 'Not available on this footage'+(f': {why[:1].lower()+why[1:]}' if why else '')
        dets = (self.r.get('overview') or {}).get('detectors') or []
        source, side = self.possession_source(po), self.marked_side(po, dets)
        one = side in ('A', 'B')
        cover = f'{_pct(po["coverage"])} of the time' if _is_num(po.get('coverage')) else None
        if source == 'coach':
            how = 'from your marks'+(f', {cover}' if cover else '')
        elif source == 'mixed':
            how = "your marks where you made them, PitchIQ's estimate elsewhere"+(f'; from {cover}' if cover else '')
        else:
            how = f'estimated from {cover}' if cover else 'estimated'
        rs = po.get('roughShares') if isinstance(po.get('roughShares'), dict) else None
        if source == 'coach' and one:
            other = 'B' if side == 'A' else 'A'
            text = (f"No share given: you marked only Team {side}'s possession"+(f' ({cover})' if cover else '')
                    + f"; mark Team {other}'s spells too for a share")
        elif rs and _is_num(rs.get('A')) and _is_num(rs.get('B')):
            names = {'A': self.team_names.get('A', 'Team A'), 'B': self.team_names.get('B', 'Team B')}
            text = f'{names["A"]} about {_pct(rs["A"])}, {names["B"]} about {_pct(rs["B"])} ({how})'
        else:
            text = how[:1].upper()+how[1:]
        return text+('; limited quality' if po.get('quality') == 'limited' else '')

    @staticmethod
    def quality_text(dq):
        parts = []
        if _is_num(dq.get('positioned')):
            parts.append(f'about {dq["positioned"]:.1f} players per team placed on the pitch')
        if isinstance(dq.get('lineup'), int) and not isinstance(dq.get('lineup'), bool):
            parts.append(f'lineup of {dq["lineup"]}')
        if dq.get('dominantCause') in CAUSE_UI:
            parts.append(CAUSE_UI[dq['dominantCause']])
        return '; '.join(parts)

    def went_well(self):
        items = self.r['strengths']
        self.section_heading('strengths', 2, 'What went well', len(items), keep=110 if items else 0)
        if not items:
            self.L.para(V2_EMPTY['strengths'], colour=INK_2, font='oblique')
        for f in items:
            self.card(f)

    def needs_work(self):
        L, r = self.L, self.r
        from . import report as R
        work, parts = r['needsWork'], R.reviews_by_part(r)
        issues = sum(len(fs) for part, fs in parts if part['part'] != 'moments')
        self.section_heading('needsWork', 3, 'What needs work', len(work)+issues, keep=110)
        mx = (self.p.get('evidence') or {}).get('maxWorkItems')
        if isinstance(mx, int) and not isinstance(mx, bool):
            L.para(f'At most {mx} {"item is" if mx == 1 else "items are"} shown at this level, the most important '
                   'first; nothing is added to fill the section.', 8.4, INK_2)
        if not work:
            L.para(V2_EMPTY['needsWork'], colour=INK_2, font='oblique')
        for f in work:
            self.card(f)
        for part, fs in parts:
            mw = part['part'] == 'moments'
            L.heading(part['title']+(f' ({len(fs)})' if mw else ''), 2, 'moments' if mw else 'reviews', keep=110)
            L.para(part['intro'], 8.4, INK_2)
            for f in fs:
                self.card(f)

    def card(self, f):
        L, render = self.L, self.render
        t = f.get('text') or {}
        verdict = f.get('verdict')
        mw = is_moments(f)
        L.ensure(78)
        L.anchor(f'fnd-{f["id"]}')
        bw = self.badge(verdict, moments=mw)
        x = L.left+bw+7
        L.add([render.text(x, L.y+8.2, self.skill_name(f.get('observation')), 7.4, MUTED, 'bold')])
        if isinstance(f.get('priority'), int) and verdict in ('work_on', 'try_this'):
            L.add([render.text(L.right, L.y+8.2, f'Priority {f["priority"]}', 7.4, BRAND_DARK, 'bold', 'right')])
        L.y += 15
        L.para(self.say(t.get('headline')) or self.skill_name(f.get('observation')), 11, INK, 'bold', after=3)
        if t.get('explanation'):
            L.para(self.say(t['explanation']), 9.2, INK, after=4)
        if t.get('action'):
            lead = 'TO BUILD ON IT' if verdict == 'strength' else None if verdict == 'try_this' else \
                'WHAT TO LOOK FOR' if mw else 'WHAT COULD HELP'
            if lead:
                L.ensure(24)
                L.text_line(L.left, lead, 6.6, BRAND_DARK, 'bold', leading=10)
                L.y += 10
                L.para(self.say(without_lead(t['action']) if verdict == 'strength' else t['action']), 9.0, INK, after=4)
            else:
                L.box([(self.say(t['action']), 9.0, 'regular', INK)], fill='#e2f2f3', accent='#1f6f78', after=5)
        for line in (self.fact_line(f), self.measure_line(f), *self.guard_notes(f), None if mw else self.action_note(f)):
            if line:
                L.para(self.say(line), 7.8, INK_2, after=2.5)
        refs = [self.moment_ref(ts.get('evidence')) for ts in f.get('timestamps') or []]
        self.chips('Key moments:', [(ref[0], {'dest': ref[1]}) for ref in refs if ref])
        if verdict in ('work_on', 'try_this'):
            names = [(self.drill_name(d), {'dest': f'drill-{d}'}) for d in dict.fromkeys(f.get('drills') or [])]
            if names:
                self.chips('Drill:', names)
            else:
                why = self.no_drill(f)
                if why:
                    L.para(why, 7.8, INK_2, 'oblique', after=2.5)
        for fb in self.feedback:
            if fb.get('thisRun') and fb.get('verdict') and fb.get('attached', fb.get('findingKey')) == f.get('key'):
                note = fb.get('text') if isinstance(fb.get('text'), str) and fb['text'].strip() else None
                L.para(f'Your review: {VERDICTS.get(fb["verdict"], _humanize(fb["verdict"]))}'
                       + (f' - "{note}"' if note else '')+(f' ({str(fb.get("made"))[:10]})' if fb.get('made') else ''),
                       7.8, BRAND_DARK, after=2.5)
        L.add([render.path([(L.left, L.y+2), (L.right, L.y+2)], stroke=RULE, width=.6)])
        L.y += 11

    def fact_line(self, f):
        """The counts behind a finding in the level's register (the page's factLine)."""
        a = f.get('aggregate') or {}
        n, ph = self.numbers, self.phase_words(f)
        opp = a.get('opportunities')
        m = f.get('measurements') or {}
        mw = is_moments(f) and all(isinstance(m.get(k), int) and not isinstance(m.get(k), bool)
                                   for k in ('clearMoments', 'clearSpells', 'decidedMoments'))
        if not isinstance(opp, int) or isinstance(opp, bool) or opp == 0:
            return None
        if n == 'none':
            return f'Looked at {ph}.' if ph else None
        verdict = f.get('verdict')
        if verdict == 'strength':
            k = a.get('distinctDemonstratedSpells')
            text = f'Shown in {a.get("demonstrated")} of {opp} moments ({k} separate {_plural(k, "spell")})'
        elif verdict == 'describe':
            text = f'{opp} {_plural(opp, "moment")} looked at'
        elif mw:
            c, d, k = m['clearMoments'], m['decidedMoments'], m['clearSpells']
            of = f'all {d}' if c >= d else f'{c} of the {d}'
            text = f'Clear in {of} moments PitchIQ could judge ({k} separate {_plural(k, "spell")})'
            failing = m.get('failingMoments')       # the counts add up (critic B-3, FACTS F-4 / COACH F-10, round 2)
            if isinstance(failing, int) and not isinstance(failing, bool) and failing > c:
                text += f', {failing-c} more fell short less clearly'
        else:
            k = a.get('distinctFailingSpells')
            text = f'Missing in {a.get("notDemonstrated")} of {opp} moments ({k} separate {_plural(k, "spell")})'
        if a.get('unclear'):
            text += f', {a["unclear"]} unclear'
        if ph:
            text += f', {ph}'
        if n != 'counts' and _is_num(a.get('phaseSeconds')) and a['phaseSeconds'] > 0:
            text += f', over {_secs(a["phaseSeconds"])} of that time'
        if n == 'rates' and _is_num(a.get('ratePer10')) and not mw:
            text += f' ({a["ratePer10"]:.1f} per 10 minutes of it)'
        return text+'.'

    def measure_line(self, f):
        """The finding's measurements at its first key moment (the page's measure chips): from level 3, metres
        rounded at level 3, shares from level 5; metres only where the scale is usable."""
        n = self.numbers
        level = f.get('level') if isinstance(f.get('level'), int) else self.level
        m = f.get('measurements') or {}
        if n in ('none', 'counts'):
            return None
        est = m.get('metres') == 'estimated'
        parts = []
        first = chip_levels(f)
        for k, v in m.items():
            ui = MEASURE_UI.get(k)
            if not ui or v is None or first.get(k) is None or (isinstance(level, int) and level < first[k]):
                continue
            text = None
            if ui[1] == 'bool':
                text = ('yes' if v else 'no') if isinstance(v, bool) else None
            elif not _is_num(v):
                text = None
            elif ui[1] == 'share':
                text = _pct(v) if n in ('shares', 'rates') else None
            elif ui[1] == 'm':
                if m.get('metres') != 'unusable':
                    text = (f'{int(math.floor(v+.5))}' if n == 'rounded_metres' else f'{v:.1f}')+' m'+(' (est.)' if est
                                                                                                         else '')
            elif ui[1] == 'deg':
                text = f'{int(math.floor(v+.5))} degrees'
            elif ui[1] == 'signed':
                r_ = int(math.floor(v+.5))
                text = f'+{r_}' if r_ > 0 else str(r_)
            else:
                text = str(int(math.floor(v+.5)))
            if text is not None:
                parts.append(f'{ui[0]} {text}')
        source = m.get('phaseSource')
        if source in ('coach', 'mixed'):
            parts.append('who had the ball: '+('marked by the coach' if source == 'coach' else
                                               'partly marked by the coach'))
        return ('At the first key moment: '+'; '.join(parts)+'.') if parts else None

    def guard_notes(self, f):
        out = []
        g = f.get('guardrails') or {}
        failed = [k for k in GUARDRAIL_UI if g.get(k) == 'fail' and not (k == 'evidence' and is_moments(f))]
        if failed:
            out.append('Could not be confirmed: '+', '.join(GUARDRAIL_UI[k] for k in failed)+'.')
        lim = [REASON_UI[c] for c in f.get('limitations') or [] if c in REASON_UI]
        if lim:
            out.append('Limits: '+'; '.join(lim)+'.')
        ctx = f.get('context') or {}
        if ctx.get('styleEffect'):
            name = self.style_name(ctx.get('style'))
            known = ctx.get('style') not in (None, 'unknown')
            out.append('Read with the style you chose'+(f' ({name.lower()})' if known else '')+'.')
        return out

    @staticmethod
    def action_note(f):
        sa = f.get('suggestedAction')
        if not isinstance(sa, dict) or f.get('verdict') in ('strength', 'describe'):
            return None
        if sa.get('mode') in ('zone', 'point'):
            return ('A possible better area is drawn on the key moment: one player moved, everyone else kept where '
                    'they were.')
        if sa.get('mode') == 'direction':
            return ('The direction of a possible movement is drawn on the key moment; the area itself was too large '
                    'to show as one place.')
        return 'No reachable better position could be confirmed, so only the principle is given.'

    def drill_name(self, did):
        d = self.drill_record(did)
        return d['name'] if isinstance(d, dict) and isinstance(d.get('name'), str) else _humanize(did)

    def no_drill(self, f):
        """Why a work item shows no drill (drills.no_drill_reason, worded as report.NO_DRILL_WORDS)."""
        try:
            from . import drills as D, report as R
            lib = D.load_library()
            why = D.no_drill_reason(lib, self.p, f.get('observation'))
            short = (f.get('text') or {}).get('short') or self.skill_name(f.get('observation'))
            return R.NO_DRILL_WORDS[why].format(short=short[:1].lower()+short[1:])
        except Exception:
            return None

    def key_moments(self):
        L = self.L
        self.section_heading('keyMoments', 4, 'Key moments', len(self.moments), keep=230 if self.moments else 0)
        if not self.moments:
            L.para(V2_EMPTY['keyMoments'], colour=INK_2, font='oblique')
            return
        L.para('In time order. Each is the exact frame of the source video at its timestamp, with the annotation '
               'PitchIQ drew for it, beside the same moment as a top-down diagram. In PitchIQ, Replay plays it from '
               'three seconds before to two seconds after.', 8.4, INK_2)
        for ev in self.moments:
            self.moment(ev)

    def moment(self, ev):
        L, render = self.L, self.render
        col = (L.width-COLUMN_GAP)/2
        n = self.number[ev['id']]
        f = self.findings.get(ev.get('finding'))
        named = self.named(ev)
        diagram = None
        if isinstance(ev.get('diagram'), dict):
            diagram = render.diagram_ops(ev['diagram'], col, self.colours, self.labels, self.team_names,
                                         palette=self.pal_print, caption=False, adjustments=self.adjust.get(ev['id']))
            self.skipped += [dict(s, evidence=ev['id'], figure='diagram') for s in diagram['skipped']]
        jpeg, crop, why = self.frame_jpeg(ev)
        img_h, ov_ops, ov_note = 0.0, None, None
        x2 = L.left+col+COLUMN_GAP
        ov = ev.get('videoOverlay')
        overlay_left_out = False
        if jpeg:
            iw, ih, _ = jpeg_info(jpeg)
            img_h = col*ih/iw
            if isinstance(ov, dict):
                ow, oh = ov.get('width'), ov.get('height')
                if not (_is_num(ow) and _is_num(oh) and oh and abs(ow/oh-iw/ih)/(iw/ih) <= .02):
                    ov, crop, overlay_left_out = None, None, True
            else:
                ov, crop = None, None
        fig_h = max(diagram['height'] if diagram else 0.0, img_h if jpeg else 34.0, 34.0)
        head = (f.get('text') or {}).get('short') or (f.get('text') or {}).get('headline') if f else None
        head = head or self.skill_name(ev.get('detector'))
        lines = []
        cap = ev.get('caption') if isinstance(ev.get('caption'), str) else ''
        if cap and cap.strip() != str(head).strip():
            lines.append(cap)
        tier = ev.get('tier')
        if jpeg and ov is not None:
            lines.append('__overlay__')                   # replaced by the overlay's own tier sentence below
        elif jpeg:
            lines.append('Screenshot: the exact source frame.'+(' The overlay was left out: its image size does not '
                                                                 'match the frame.' if overlay_left_out else
                                                                 ' Nothing could be drawn over it at this frame.'))
        if crop:
            lines.append(f'Zoomed {crop[4]:.1f}x on the players involved; the frame itself is not changed.')
        if named and ev.get('players'):
            lines.append('Players: '+', '.join(ev['players'])+'.')      # ids: say() shows them as labels once
        win = (ev.get('annotation') or {}).get('window') if isinstance(ev.get('annotation'), dict) else None
        if isinstance(win, dict) and _is_num(win.get('t0')) and _is_num(win.get('t1')):
            lines.append(f'Replay window {clock(win["t0"])} - {clock(win["t1"])}.')
        alt = self.alts.get(ev.get('alternative')) if ev.get('alternative') else None
        if alt:
            lines.append(self.alternative_text(alt, named))
        else:
            none = next((x for x in self.r['noAlternatives'] if x.get('evidence') == ev['id']), None)
            if none and none.get('reason') in ALT_REASON_UI and f and f.get('verdict') in ('work_on', 'try_this',
                                                                                             'review') \
                    and not is_moments(f):
                lines.append(f'No area is drawn: {ALT_REASON_UI[none["reason"]]}.')
        if jpeg and ov is not None:              # drawn at top 0 here, moved to the figure's top when placed
            k = col/(crop[2] if crop else ov['width'])
            ox_, oy_ = (x2-crop[0]*k, -crop[1]*k) if crop else (x2, 0.0)
            ov_ops = render.overlay_ops(ov, ox_, oy_, ov['width']*k, ov['height']*k, self.colours, self.labels,
                                        caption=False, palette=self.pal_video, unit=col/400.0)
            ov_note = ov_ops['note']
            lines[lines.index('__overlay__')] = 'Screenshot: the exact source frame. '+ov_note
        elif not jpeg and tier in TIER_UI:
            lines.append(f'Video tier: {TIER_UI[tier]}.')
        text = ' '.join(self.say(x) for x in lines if x)
        cap_lines = wrap(text, 7.6, L.width)
        url = self.playback(ev)
        title_h = 26
        h = title_h+fig_h+4+len(cap_lines)*7.6*1.36+(11 if url else 0)+(11 if f else 0)+8
        L.ensure(h)
        L.anchor(f'ev-{ev["id"]}')
        self.drawn_evidence.add(ev['id'])
        seg = ev.get('segmentTime')
        title = f'E{n}  |  {clock(ev.get("time"))}  (segment {offset(seg) if _is_num(seg) else "n/a"})'
        if isinstance(ev.get('frame'), int):
            title += f'  |  frame {ev["frame"]}'
        L.text_line(L.left, title, 8.4, INK, 'bold', leading=11)
        if f:
            label = V2_MOMENTS[0] if is_moments(f) else V2_VERDICTS.get(f.get('verdict'), ('', ''))[0]
            bw = text_width(label.upper(), 6.6, 'bold')+10
            self.badge(f.get('verdict'), x=L.right-bw, moments=is_moments(f))
        L.y += 12
        L.text_line(L.left, fit_text(self.say(head), 8.2, L.width), 8.2, INK_2, 'regular', leading=11)
        L.y += 14
        top = L.y
        if diagram:
            L.add(_translate(diagram['ops'], L.left, top))
        else:
            L.add([render.rect(L.left, top, col, 34, fill=ZEBRA),
                   render.text(L.left+6, top+20, 'No diagram for this moment.', 7.6, MUTED, 'oblique')])
        if jpeg:
            k = col/(crop[2] if crop else (ov['width'] if ov else iw))
            fw = (ov['width'] if ov else iw)*k
            fh = (ov['height'] if ov else ih)*k
            ix, iy = (x2-crop[0]*k, top-crop[1]*k) if crop else (x2, top)
            L.add([render.clip(x2, top, col, img_h),
                   {'op': 'image', 'x': ix, 'y': iy, 'w': fw, 'h': fh, 'jpeg': jpeg, 'frame': ev.get('frame')}])
            self.screenshots += 1
            self.zoomed += 1 if crop else 0
            if ov_ops is not None:
                self.skipped += [dict(s, evidence=ev['id'], figure='overlay') for s in ov_ops['skipped']]
                L.add(_translate([op for op in ov_ops['ops'] if op['op'] not in ('clip', 'unclip')], 0.0, top))
                self.overlays += 1
            cs = 5.0
            cw = text_width(render.OVERLAY_CAPTION if ov_ops is not None else 'source frame', cs)+5
            L.add([render.rect(x2+3, top+img_h-cs-6, cw, cs+3.4, fill=render.EDGE, alpha=.7),
                   render.text(x2+5.5, top+img_h-5.2, render.OVERLAY_CAPTION if ov_ops is not None else 'source frame',
                               cs, render.WHITE),
                   render.unclip(), render.rect(x2, top, col, img_h, stroke=RULE, width=.5)])
        else:
            msg = wrap(f'Screenshot unavailable: {why}.', 7.6, col-12)
            bh = max(34.0, 12+len(msg)*10.5)
            L.add([render.rect(x2, top, col, bh, fill=ZEBRA)])
            for i, line in enumerate(msg):
                L.add([render.text(x2+6, top+14+i*10.5, line, 7.6, MUTED, 'oblique')])
        L.y = top+fig_h+4
        for line in cap_lines:
            L.text_line(L.left, line, 7.6, INK_2, leading=7.6*1.36)
            L.y += 7.6*1.36
        if f:
            head = (f.get('text') or {}).get('headline') or self.skill_name(f.get('observation'))
            s = 'From: '+fit_text(self.say(head), 7.6, L.width-30)
            L.text_line(L.left, s, 7.6, BRAND_DARK, leading=11)
            L.add([{'op': 'link', 'x': L.left, 'y': L.y, 'w': text_width(s, 7.6), 'h': 11, 'dest': f'fnd-{f["id"]}'}])
            L.y += 11
        if url:
            s = f'Open {clock(ev.get("time"))} in PitchIQ (needs the local PitchIQ server with this video open)'
            L.text_line(L.left, s, 7.6, BRAND_DARK, leading=11)
            L.add([render.path([(L.left, L.y+9.6), (L.left+text_width(s, 7.6), L.y+9.6)], stroke=BRAND_DARK,
                               width=.4),
                   {'op': 'link', 'x': L.left, 'y': L.y, 'w': text_width(s, 7.6), 'h': 11, 'uri': url}])
            L.y += 11
        L.add([render.path([(L.left, L.y+3), (L.right, L.y+3)], stroke=RULE, width=.4)])
        L.y += 10

    def alternative_text(self, alt, named):
        reach, imp = alt.get('reach') or {}, alt.get('improves') or {}
        who = alt.get('mover') if named and isinstance(alt.get('mover'), str) else None   # an id: said once
        base = (f'The dotted area is one possibility for {who}' if who else
                'The dotted area is one possibility for the teammate who could move')
        base += ', with everyone else kept where they were: an illustration, not an instruction.'
        if self.numbers in ('none', 'counts') or not (_is_num(reach.get('horizonS')) and _is_num(reach.get('speedMS'))):
            return base
        share = imp.get('shareOfZone')
        return (f'{base} It is within reach of where {who or "they"} stood {reach["horizonS"]:.1f} s earlier at up '
                f'to {reach["speedMS"]:.1f} m/s'
                + (f', and the {_humanize(imp.get("relation") or "relation").lower()} holds in {_pct(share)} of it'
                   if _is_num(share) else '')+'.')

    def diagrams(self):
        L, render, r = self.L, self.render, self.r
        refs = r['diagrams']
        self.section_heading('diagrams', 5, 'Tactical diagrams', len(refs), keep=60)
        if not refs:
            L.para(V2_EMPTY['diagrams'], colour=INK_2, font='oblique')
            return
        fdg = [d for d in refs if d.get('kind') == 'finding' and d.get('evidence') in self.ev]
        if fdg:
            L.para('The diagram of each key moment, with any suggested area, is drawn beside its screenshot under '
                   'Key moments:', 8.4, INK_2, after=2)
            items = []
            for d in fdg:
                ref = self.moment_ref(d['evidence'])
                zone = any(isinstance(el, dict) and el.get('type') == 'suggestedZone'
                           for el in d['diagram'].get('elements') or [])
                items.append((ref[0]+(' (possible area)' if zone else ''), {'dest': ref[1]}))
            self.chips('', items)
        others = [d for d in refs if d.get('kind') in ('movement', 'formation')]
        col = (L.width-COLUMN_GAP)/2
        for kind, title, intro in (
                ('movement', 'How our players moved', 'Arrows join where each of our players stood two seconds before '
                 'a key moment to where they stood at it (observed positions only).'),
                ('formation', 'Average shape', 'Average positions of our outfield players per measured shape and '
                 'phase: described, not judged.')):
            group = [d for d in others if d['kind'] == kind]
            if not group:
                continue
            rows = []
            for i in range(0, len(group), 2):
                figs = []
                for d in group[i:i+2]:
                    dd = render.diagram_ops(d['diagram'], col, self.colours, self.labels, self.team_names,
                                            palette=self.pal_print, caption=False)
                    self.skipped += [dict(s, diagram=d.get('id'), figure='diagram') for s in dd['skipped']]
                    cap = self.say(d.get('caption') or d['diagram'].get('title') or '')
                    figs.append((d, dd, wrap(cap, 7.4, col)))
                rows.append((figs, max(dd['height']+4+len(c)*7.4*1.36+(11 if d.get('evidence') in self.ev else 0)
                                       for d, dd, c in figs)))
            intro_h = len(wrap(intro, 8.4, L.width))*8.4*1.38+4
            L.heading(title, 2, keep=intro_h+rows[0][1]+8)    # the heading stays with its first row of diagrams
            L.para(intro, 8.4, INK_2)
            for figs, h in rows:
                L.ensure(h+8)
                top = L.y
                for k, (d, dd, cap_lines) in enumerate(figs):
                    x = L.left+k*(col+COLUMN_GAP)
                    L.add(_translate(dd['ops'], x, top))
                    y = top+dd['height']+4
                    for line in cap_lines:
                        L.add([render.text(x, y+7.4, line, 7.4, INK_2)])
                        y += 7.4*1.36
                    ref = self.moment_ref(d.get('evidence')) if d.get('evidence') in self.ev else None
                    if ref:
                        s = f'Show {ref[0]}'
                        L.add([render.text(x, y+8, s, 7.4, BRAND_DARK),
                               {'op': 'link', 'x': x, 'y': y, 'w': text_width(s, 7.4), 'h': 10, 'dest': ref[1]}])
                L.y = top+h+10
        if any(d['diagram'].get('metresEstimated') is not False for d in refs):
            L.para('Positions are the observed ones at each frame; distances are estimates.', 7.8, MUTED)

    def drills(self):
        L, render, r = self.L, self.render, self.r
        picks = r['drills']
        self.section_heading('drills', 6, 'Training drills', len({p['drill'] for p in picks}), keep=80)
        sf = self.p.get('safety') or {}
        notes = [self.say(x) for x in _items(sf.get('notes')) if isinstance(x, str) and x.strip()]
        if notes:
            L.box([(f'Safety at this age ({self.p.get("ageGroup") or "n/a"})', 8.4, 'bold', INK)]
                  + [(x, 8.2, 'regular', INK_2) for x in notes], fill='#faf0dc', accent='#d9a961')
        if not picks:
            L.para(no_drill_text(r), colour=INK_2, font='oblique')
        groups = {}
        for pk in picks:
            groups.setdefault(pk['drill'], []).append(pk)
        for did, group in groups.items():
            self.drill(did, group)
        for f in r['needsWork']:
            if not f.get('drills'):
                why = self.no_drill(f)
                if why:
                    L.para(why, 8.4, INK_2, 'oblique')

    def drill(self, did, picks):
        L, render = self.L, self.render
        d = self.drill_record(did)
        L.heading(self.drill_name(did), 2, f'drill-{did}', keep=90)
        for pk in picks:
            f = self.findings.get(pk.get('finding')) if pk.get('finding') else None
            if f and not pk.get('optional'):
                head = self.say((f.get('text') or {}).get('headline') or self.skill_name(f.get('observation')))
                prio = f' (priority {f["priority"]})' if isinstance(f.get('priority'), int) else ''
                s = fit_text(f'For: {head}{prio}', 8.4, L.width)
                L.ensure(12)
                L.text_line(L.left, s, 8.4, BRAND_DARK, 'bold', leading=12)
                L.add([{'op': 'link', 'x': L.left, 'y': L.y, 'w': text_width(s, 8.4, 'bold'), 'h': 12,
                        'dest': f'fnd-{f["id"]}'}])
                L.y += 13
            else:
                L.para(optional_label(self.r), 8.4, BRAND_DARK, 'bold', after=2)
            fields = pk.get('fields') or {}
            for key, title in DRILL_FIELDS_V2:
                if key == 'easier':
                    refs = [self.moment_ref(e) for e in _items(fields.get('moments'))]
                    self.chips('Moments it addresses:', [(x[0], {'dest': x[1]}) for x in refs if x])
                v = fields.get(key)
                if isinstance(v, str) and v.strip():
                    L.ensure(24)
                    L.text_line(L.left, title.upper(), 6.6, BRAND_DARK, 'bold', leading=10)
                    L.y += 10
                    L.para(self.say(v), 8.8, INK, after=3)
        if not isinstance(d, dict):
            L.para('PitchIQ proposal, not validated by a coach.', 7.8, MUTED)
            return
        dim = (lambda v: v if _is_num(v) else v.get('value') if isinstance(v, dict) and _is_num(v.get('value'))
               else None)
        rows = []
        pc = d.get('player_count') if isinstance(d.get('player_count'), dict) else {}
        if _is_num(pc.get('min')) and _is_num(pc.get('max')):
            span = f'{pc["min"]}' if pc['min'] == pc['max'] else f'{pc["min"]}-{pc["max"]}'
            rows.append(('Players', f'{span} players'+(f'. {pc["note"]}' if isinstance(pc.get('note'), str) else '')))
        du = d.get('duration') if isinstance(d.get('duration'), dict) else {}
        if _is_num(du.get('minutes')):
            rows.append(('Duration', f'{du["minutes"]} min'+(f' ({du["blocks"]})' if isinstance(du.get('blocks'), str)
                                                             else '')))
        fs = d.get('field_setup') if isinstance(d.get('field_setup'), dict) else {}
        ln, wd = dim(fs.get('length')), dim(fs.get('width'))
        area = '. '.join(x for x in (f'About {ln:g} x {wd:g} m' if ln is not None and wd is not None else '',
                                     fs.get('notes') if isinstance(fs.get('notes'), str) else '') if x)
        if area:
            rows.append(('Area', area))
        eq = [x for x in _items(d.get('equipment')) if isinstance(x, str)]
        if eq:
            rows.append(('Equipment', ', '.join(eq)))
        sf = d.get('safety') if isinstance(d.get('safety'), dict) else {}
        if sf:
            rows.append(('Safety', ', '.join(x for x in (
                'no heading' if sf.get('heading') == 'none' else 'heading only within the age limits',
                f'{sf.get("physicalLoad") or "low"} physical load',
                'no tackling' if sf.get('contact') == 'no_tackle' else None) if x)+'.'))
        if rows:
            self.kv(rows, key_width=80, size=8.4)
        for key, title in (('instructions', 'How it runs'), ('coaching_points', 'Coaching points')):
            items = [self.say(x) for x in _items(d.get(key)) if isinstance(x, str) and x.strip()]
            if items:
                L.ensure(24)
                L.text_line(L.left, title.upper(), 6.6, BRAND_DARK, 'bold', leading=10)
                L.y += 10
                L.bullets(items, size=8.6, after=2)
        refs = [x for x in _items(d.get('source_reference')) if isinstance(x, dict) and x.get('source')]
        if refs:
            parts = []
            for x in refs:
                src = self.sources.get(x['source']) or {}
                title = src.get('title') or x['source']
                parts.append(f'{DRILL_SOURCE_UI.get(x.get("relation"), "From")} {title}'
                             + (f' ({x["note"]})' if isinstance(x.get('note'), str) and x['note'].strip() else ''))
            L.para('Source: '+'; '.join(parts)+'.', 7.8, INK_2, after=3)
        L.para('PitchIQ proposal, not validated by a coach: field sizes, durations and loads are suggestions.', 7.8,
               MUTED, after=8)

    def progression(self):
        L, render, r = self.L, self.render, self.r
        pr = r.get('progression')
        skills = [k for k in _dicts((pr or {}).get('skills')) if isinstance(k.get('skill'), str)]
        self.section_heading('progression', 7, 'Development progression', keep=80)
        if skills:
            name = self.o.get('squad', {}).get('name') if isinstance(self.o.get('squad'), dict) else None
            if isinstance(name, str) and name.strip():
                L.para(f'Squad: {name.strip()}', 9, INK, 'bold', after=3)
            L.table([('Skill', .40, 'left'), ('State', .22, 'left'), ('Moments', .12, 'right'),
                     ('Matches', .11, 'right'), ('Trend', .15, 'left')],
                    [[k.get('name') or _humanize(k['skill']), MASTERY_UI.get(k.get('state'), str(k.get('state'))),
                      str(k.get('opportunities')), str(k.get('matches')), TREND_UI.get(k.get('trend'), '-')]
                     for k in skills])
            if isinstance(pr.get('note'), str) and pr['note'].strip():
                L.para(self.say(pr['note']), 8.3, INK_2)
            L.para("States follow PitchIQ's own rules (proposals, not validated: review them with a coach). They "
                   'count the moments where each skill was needed, so a short clip and a full match are compared '
                   'fairly.', 7.8, MUTED)
        else:
            L.para('Link this team to a squad in the coaching setup to follow its skills across matches.', 8.8, INK_2,
                   'oblique')
        n = r['nextObjective']
        if isinstance(n.get('text'), str) and n['text'].strip():
            f = self.findings.get(n.get('finding')) if n.get('finding') else None
            # (fix round 2: a next objective with no skill, where assess found none, is its text alone; the origin is
            # not said twice when the text says it, as on the page. Fix round 3: the plain origin line only for a
            # skill not from this match whose text names no origin at all: an objective chosen from this match's
            # checks (critic S-3, COACH F-4 / FACTS F-9) or one from a work item whose finding is not shown must never
            # read "not from this match"; origin_line)
            more = origin_line(n, f)
            focus = split_focus(n['text']) if not n.get('fromMatch') else None
            if focus:
                # the level's suggested focus (polish round): its label, the focus, its sources by title and id
                label, body, ids = focus
                src = '; '.join(f'{(self.sources.get(i) or {}).get("title") or i} ({i})' for i in ids)
                L.box([('Next development objective', 8.6, 'bold', INK), (label+'.', 7.8, 'regular', INK_2),
                       (self.say(body), 9.2, 'regular', INK)]
                      + ([('Source: '+src+'.', 7.8, 'regular', INK_2)] if src else []))
            else:
                L.box([('Next development objective', 8.6, 'bold', INK), (self.say(n['text']), 9.2, 'regular', INK)]
                      + ([(more, 7.8, 'regular', INK_2)] if more else []))
            if n.get('fromMatch') and f:
                s = fit_text('Finding: '+self.say((f.get('text') or {}).get('headline') or
                                                  self.skill_name(f.get('observation'))), 8.2, L.width)
                L.text_line(L.left, s, 8.2, BRAND_DARK, leading=12)
                L.add([{'op': 'link', 'x': L.left, 'y': L.y, 'w': text_width(s, 8.2), 'h': 12,
                        'dest': f'fnd-{f["id"]}'}])
                L.y += 16

    def sources_section(self):
        """The sources credited, after what they say for this level. The level's guidance is the profile's
        paraphrase of federation material for the coach (the page's 'Sources for this level' panel): it uses the
        game's own words ('win the ball', 'pressing', 'transition'), which the report's text keeps away from at the
        levels that ban them and never uses as a claim about the match, so it stands here, labelled, and not among
        the findings (measured: 22 objectives, principles and not-expected items of levels 1-7, at each level's
        typical age, trip schema.event_problems or the level's banned words; none is a statement about the match)."""
        L, r, p = self.L, self.r, self.p
        self.section_heading('sources', None, 'Sources', len(r['sources']), keep=60)
        L.para("The level's guidance, notes and drills paraphrase these sources in PitchIQ's own words. No "
               "federation reviewed or endorses PitchIQ; every check and threshold is PitchIQ's own proposal.", 8.4,
               INK_2)
        for key, title in (('objectives', 'What the sources aim for at this level'),
                           ('notExpected', 'What they do not expect at this level')):
            items = [g for g in p[key] if isinstance(g.get('text'), str) and g['text'].strip()]
            if items:
                L.heading(title, 3, keep=30)
                L.bullets([self.guidance(g) for g in items], size=8.2)
        if p['objectives'] or p['notExpected']:
            L.para('Guidance for the coach, paraphrased with the source ids in brackets; it describes the level, not '
                   'this match.', 7.8, MUTED)
        if not r['sources']:
            L.para('This report cites no source.', colour=INK_2, font='oblique')
            return
        rows = []
        for s in r['sources']:
            title = ' '.join(x for x in (s.get('title') or s.get('id') or '',
                                         f'({s["publisher"]})' if isinstance(s.get('publisher'), str) else '') if x)
            url = safe_uri(s.get('url'))
            rows.append([f'[{s.get("id")}] {title}', {'text': s.get('url') or '-', 'uri': url} if url else
                         (s.get('url') or '-')])
        L.table([('Source', .55, 'left'), ('Link', .45, 'left')], rows, size=7.4)

    def feedback_section(self):
        L = self.L
        fb = [f for f in self.feedback if f.get('verdict') or f.get('adjustment')]
        if not fb:
            return
        self.section_heading('feedback', None, 'Coach feedback', keep=60)
        L.para('Reviews kept with this video. They never change the measurements. Notes are printed exactly as they '
               'were typed.', 8.4, INK_2)
        keys = {f.get('key'): f for f in self.findings.values()}
        groups = (('On this report', lambda f: f.get('thisRun') and f.get('attached') in keys),
                  ('Your reviews of this run that match no finding here',
                   lambda f: f.get('thisRun') and f.get('findingKey') and f.get('attached') not in keys),
                  ('From an earlier run', lambda f: f.get('findingKey') and not f.get('thisRun')),
                  ('Feedback from the previous report engine', lambda f: not f.get('findingKey')))
        done = set()
        for title, test in groups:
            rows = []
            for f in fb:
                if id(f) in done or not test(f):
                    continue
                done.add(id(f))
                g = keys.get(f.get('attached')) if f.get('thisRun') else None
                if g is not None:
                    finding = {'text': self.say((g.get('text') or {}).get('headline') or ''),
                               'dest': f'fnd-{g["id"]}'}
                else:          # another run's (or a vanished) finding keeps that run's words and ids
                    finding = ((f.get('finding') or {}).get('headline') if isinstance(f.get('finding'), dict)
                               else None) or 'A finding'
                verdict = VERDICTS.get(f.get('verdict'), _humanize(f.get('verdict') or '-')) if f.get('verdict') \
                    else 'Adjusted area'
                note = f.get('text') if isinstance(f.get('text'), str) and f['text'].strip() else '-'
                adj = f.get('adjustment') if isinstance(f.get('adjustment'), dict) else None
                if adj and adj.get('evidence') in self.ev:
                    note += f' (adjusted area on E{self.number[adj["evidence"]]})'
                made = str(f.get('made') or '')[:10] or 'n/a'
                if not f.get('thisRun') and isinstance(f.get('run'), str):
                    made += f', run {f["run"]}'
                rows.append([finding, verdict, note, made])
            if rows:
                L.heading(title, 3, keep=40)
                L.table([('Finding', .36, 'left'), ('Verdict', .15, 'left'), ('Note', .31, 'left'),
                         ('Date', .18, 'left')], rows)

    def about(self):
        L, r = self.L, self.r
        self.section_heading('about', None, 'About this report', keep=60)
        eng, fp = r['engine'], r['fingerprints']
        rows = []
        if eng.get('version'):
            rows.append(('Engine', f'coaching v{eng["version"]}'))
        ids = ', '.join(f'{k} {eng[k]}' for k in ('measurement', 'roles', 'interpretation')
                        if isinstance(eng.get(k), str))
        if ids:
            rows.append(('Report id', ids))
        if self.p.get('version') is not None or self.p.get('sha'):
            parts = (f'version {self.p.get("version")}' if self.p.get('version') is not None else '',
                     f'({self.p["sha"]})' if self.p.get('sha') else '')
            rows.append(('Profile table', ' '.join(x for x in parts if x)))
        if rows:
            self.kv(rows, key_width=100, size=8.4)
        L.bullets(["Every statement comes from this run's tracking result; nothing is said about events PitchIQ "
                   'does not measure.',
                   'The same measurements are read for every age group and level: changing them changes the reading, '
                   'never the measurements.',
                   'Each finding rests on separate moments where the behaviour was needed; a short clip says less '
                   'than a full match.',
                   "Every threshold is PitchIQ's own proposal, not validated by a coach.",
                   'Screenshots are the exact frames of the source video; the drawings over them are analysis overlays '
                   'added for this report.'], size=8.4)


def layout_report(report, frame_provider=None, options=None):
    """(Document, info): the report laid out on pages of drawing operations (render.ops_to_png can draw them).
    frame_provider(frame, width) -> JPEG bytes or None (frames.FrameProvider; may have prefetch(frames, widths)).
    options: labels {id: display} (diagrams, overlays and, unless substitute_text is False, the text),
    colours {'A','B'}, playback_url (a str.format template with {frame} {seek} {time} {id} {video} {run}, or a
    callable(evidence) -> url), max_evidence_per_finding, max_screenshots, image_width, strict (refuse non-WinAnsi),
    stale (the reasons the report is out of date: a banner on the first page and a mark in every page header),
    section_order (the shared sections' order; default: report.py's published order, else PAGE_ORDER).
    A coaching v2 report (is_v2) is laid out by _ReportV2: the page's seven sections; options there also take
    feedback (the coaching answer's feedback entries: verdicts, notes and adjusted areas), squad {'id', 'name'} (the
    linked squad's name for the progression section), zoom (False: whole frames, never magnified), coach_spans (the
    run has coach-marked possession spans) and marked ('A', 'B', 'both' or None: whose possession they name), with
    which the possession row says where the shares come from as the page does (fix round 3, CODE M-2)."""
    options = dict(options or {})
    if is_v2(report):
        return _layout_v2(report, frame_provider, options)
    r = sanitize(report)
    ov = r['overview']
    title = f'{ov.get("title") or "Coaching report"} - {r.get("teamName") or ("Team " + str(r.get("team", "")))}'
    doc = Document(title=title, subject=f'PitchIQ coaching report, {r["video"].get("name") or ""}'.strip(' ,'),
                   created=r.get('generated'), strict=bool(options.get('strict')))
    lay = _Report(report, frame_provider, options)
    lay.layout(doc)
    info = {'pages': len(doc.pages), 'screenshots': lay.screenshots, 'overlays': lay.overlays,
            'missingFrames': sorted(set(lay.missing)), 'skippedElements': lay.skipped,
            'evidenceShown': len(lay.drawn_evidence)}
    return doc, info


def _layout_v2(report, frame_provider, options):
    r = sanitize_v2(report)
    p = r['profile']
    level = f', level {p["level"]}' if isinstance(p.get('level'), int) else ''
    title = f'Coaching report - {r.get("teamName") or ("Team " + str(r.get("team") or ""))}{level}'
    doc = Document(title=title, subject=f'PitchIQ coaching report, {r["video"].get("name") or ""}'.strip(' ,'),
                   created=r.get('generated'), strict=bool(options.get('strict')))
    lay = _ReportV2(report, frame_provider, options)
    lay.layout(doc)
    info = {'pages': len(doc.pages), 'screenshots': lay.screenshots, 'overlays': lay.overlays, 'zoomed': lay.zoomed,
            'missingFrames': sorted(set(lay.missing)), 'skippedElements': lay.skipped,
            'evidenceShown': len(lay.drawn_evidence), 'version': 2}
    return doc, info


def build_report_pdf(report, frame_provider=None, options=None):
    """(PDF bytes, info) of a coaching report. info: pages, bytes, images, links, internalLinks, droppedLinks,
    replaced (characters written as '?'), screenshots, overlays, missingFrames, skippedElements, evidenceShown."""
    doc, info = layout_report(report, frame_provider, options)
    data = doc.build()
    info.update(doc.stats)
    return data, info


def report_pdf(report, *, labels=None, frame=None, stale=None, **options):
    """The PDF bytes of a report, as soccer/coaching/store.render_pdf asks for them: labels {id: display} of the
    result as it is now, frame(frame, width) -> JPEG bytes or None, stale: the reasons the report is out of date (a
    list of texts; [] or True for out of date without a reason; None when current), further options as in
    layout_report."""
    data, _ = build_report_pdf(report, frame, dict(options, labels=labels, stale=stale))
    return data


def write_report_pdf(report, path, frame_provider=None, options=None):
    """Write the report's PDF to path atomically (uuid temp file + os.replace, retried against Windows reader
    locks as corrections._write does); returns the info of build_report_pdf plus 'path'."""
    data, info = build_report_pdf(report, frame_provider, options)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f'.{path.name}.{uuid.uuid4().hex}.tmp')
    temp.write_bytes(data)
    for attempt in range(20):
        try:
            os.replace(temp, path)
            break
        except PermissionError:
            if attempt == 19:
                temp.unlink(missing_ok=True)
                raise
            time.sleep(.1)
    info['path'] = str(path)
    return info
