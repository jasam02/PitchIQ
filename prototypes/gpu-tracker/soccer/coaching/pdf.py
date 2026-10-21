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


def pdf_filename(video, run, team, context=None, key=None):
    """The download name of a report PDF: pitchiq-coaching-<video>-<run>-<team>-<level>.pdf. The level is part of the
    name because the three context levels of one report are different documents (the first PDFs of a development and
    an advanced report downloaded under the same name and replaced each other). video: the upload's name (its stem is
    used), key: the fallback when there is none; every part is reduced to [A-Za-z0-9._-]. The server's PDF route
    uses it through store.pdf_filename."""
    def part(v, n=48):
        return re.sub(r'[^A-Za-z0-9._-]+', '-', str(v or '')).strip('-.')[:n].strip('-.')
    name = part(Path(str(video)).stem if video else '') or part(key, 16) or 'video'
    level = context if context in CONTEXT_NAMES else None
    return '-'.join(p for p in ('pitchiq-coaching', name, part(run, 64), part(team, 4), level) if p) + '.pdf'


def report_filename(report):
    """pdf_filename() of a report's own fields (video name or key, run, team, context level)."""
    r = report if isinstance(report, dict) else {}
    video = r.get('video') if isinstance(r.get('video'), dict) else {}
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
    """Absolute source time as mm:ss.ss (the Studio's transport and corrections.timestamp format)."""
    if not isinstance(t, (int, float)) or isinstance(t, bool) or not math.isfinite(t):
        return 'n/a'
    sign = '-' if t < 0 else ''
    t = abs(float(t))
    return f'{sign}{int(t//60):02d}:{t % 60:05.2f}'


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

    def heading(self, s, level=1, anchor=None, bookmark=False):
        from . import render
        size = {1: 15.0, 2: 11.5, 3: 9.5}[level]
        self.ensure(size*1.6+42)                    # keep a heading with the first lines that follow it
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
            reasons = [self.say(x) for x in self.stale if isinstance(x, str) and x.strip()]
            L.box([('OUT OF DATE', 8.5, 'bold', '#8a5f12'),
                   ('The tracking result or its settings changed after this report was made'
                    + (': ' + '; '.join(reasons) if reasons else '') + '. Regenerate the report in PitchIQ before '
                    'relying on these findings.', 8.6, 'regular', INK)], fill='#faf0dc', accent='#d9a961')
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
                + 'Diagrams show tracked positions; dashed shapes marked COACHING ILLUSTRATION are suggestions, not '
                'tracked data. This PDF holds no video: use the timestamps to find each moment.',
                8.3, 'regular', INK_2)])

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
        L.heading(dict(SECTIONS)[key], 1, key, True)
        findings = [f for f in (findings or []) if isinstance(f, dict)]
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
        from . import render
        n = len(doc.pages)
        video = (self.r.get('video') or {}).get('name') or (self.r.get('video') or {}).get('key') or ''
        left = ' | '.join(x for x in ('PitchIQ coaching report', video, self.team_short) if x)
        y = self.L.H-MARGINS[3]+26
        for i in range(n):
            doc.add(i, [render.path([(self.L.left, y-10), (self.L.right, y-10)], stroke=RULE, width=.5),
                        render.text(self.L.left, y, left, 7, MUTED),
                        render.text(self.L.right, y, f'Page {i+1} of {n}', 7, MUTED, 'regular', 'right')])


def layout_report(report, frame_provider=None, options=None):
    """(Document, info): the report laid out on pages of drawing operations (render.ops_to_png can draw them).
    frame_provider(frame, width) -> JPEG bytes or None (frames.FrameProvider; may have prefetch(frames, widths)).
    options: labels {id: display} (diagrams, overlays and, unless substitute_text is False, the text),
    colours {'A','B'}, playback_url (a str.format template with {frame} {seek} {time} {id} {video} {run}, or a
    callable(evidence) -> url), max_evidence_per_finding, max_screenshots, image_width, strict (refuse non-WinAnsi),
    stale (the reasons the report is out of date: a banner on the first page and a mark in every page header),
    section_order (the shared sections' order; default: report.py's published order, else PAGE_ORDER)."""
    options = dict(options or {})
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
