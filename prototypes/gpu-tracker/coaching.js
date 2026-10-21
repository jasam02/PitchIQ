/* PitchIQ · Coaching report: the report view of the fourth workspace tab, the evidence of a finding (its frame with
   the overlay, the top-down diagram) and the drawing of a finding on the 2D board and over the paused video.

   Data. Nothing is measured here. The server builds the report from the saved result (soccer/coaching: observations
   measured over the segment and compared with a professional reference, worded by templates filled only from those
   measurements) and caches it under a fingerprint of everything it was computed from (soccer/coaching/store.py). This
   view asks for the report of one selection (team, range, context level), shows it, offers to build it when there is
   none and to rebuild it when the result, the team-shape settings or the engine changed since (the banner lists the
   reasons; the earlier report stays readable meanwhile), and posts the coach's verdict on each finding. The report keys
   players by their internal id; every id in its text and drawings is shown as the player's display label in the result
   the page shows now (labels renumber after corrections), with the id in the tooltip; an identity a correction excluded
   shows as "excluded (was A-03)", never as a bare label that now belongs to someone else.

   Requests. Each selection (team, range, level) is one status request; a newer one, also one answered from the cache,
   drops the late answer of an older one, so the report shown and the one Export PDF saves are always the selection's.
   A build in progress (this selection's, or another one's: busy) is asked again every second and never cached. Feedback
   belongs to one run: a verdict sits on its finding only in the run it was given in; verdicts from earlier runs of the
   video are listed apart (team letters are arbitrary per run).

   Time. Every report time is absolute source seconds (the video's currentTime), shown as mm:ss.ss with the offset into
   the analysed segment. An evidence record carries the absolute source frame and seek = (frame + 0.5)/fps: the browser
   then shows exactly that frame (20 of 20 targets per video in Chromium; seeking to the frame's own time showed the frame
   before on 1 of 20). The screenshot is that frame's raw pixels (/api/coaching/frame); its overlay is drawn here in the
   source video's pixels, so labels follow corrections.

   Drawings never claim more than the data: the evidence's tier says what may be drawn over the video (world: pitch
   lines through a paint-checked HIGH registration; pinned: shapes from the players' feet on that frame; image: boxes and
   labels only), diagrams draw the observed positions the record lists (an estimate would be dashed and hollow), and a
   suggested position is dashed and labelled as a coaching illustration. */
(function (global) {
'use strict';

const SVG_NS = 'http://www.w3.org/2000/svg';
const BOARD = {length: 105, width: 68, margin: 4.5};   // the board is drawn 105 x 68 m (tactical.js PITCH); spec coordinates are normalized
const KINDS = {'Observed strength': 'strength', 'Review opportunity': 'review', 'Potential improvement': 'improvement', 'High-confidence concern': 'concern'};
// The strength and improvement rules as soccer/coaching/detectors.py applies them: a strength's typical value sits on the
// favourable side of the professional median and rarely beyond the professional upper quarter over enough measured time; an
// improvement has more spells beyond the professional level than 99% of professional stretches of the same measured length.
const GROUPS = [
  ['strengths', 'Strengths', 'What the team kept on the favourable side of the professional median, and rarely beyond the professional upper quarter, over enough measured time.',
   'No strength met the evidence bar in this range (a typical value on the favourable side of the professional median, rarely beyond its upper quarter, over enough measured time).'],
  ['improvements', 'Areas to improve', 'What went beyond the professional level more often than professional teams do over the same measured time, in the same context.',
   'Nothing went beyond the professional level more often than professional teams do over the same measured time.'],
  ['reviews', 'Worth reviewing', 'Context-dependent moments: worth a look on the video with the players, not a verdict.',
   'Nothing context-dependent to review in this range.']];
const SECTIONS = [['defensiveOrganization', 'Defensive organisation'], ['spacing', 'Spacing and compactness'], ['formation', 'Formation'],
                  ['attackingStructure', 'Attacking structure'], ['movementSupport', 'Movement and support']];
const VERDICTS = [['correct', 'Correct'], ['partial', 'Partially correct'], ['incorrect', 'Incorrect'], ['insufficient', 'Not enough evidence']];
const CONTEXTS = {development: 'Development', competitive: 'Competitive', advanced: 'Advanced'};
const TIERS = {world: 'Pitch lines drawn through this frame’s registration (high confidence, checked against the painted lines).',
               pinned: 'Shapes drawn from the players’ feet on this frame; pitch geometry is approximate.',
               image: 'Boxes and labels only: the registration is not reliable enough on this frame for pitch lines.'};
const METRICS = {width: ['Team width', 'm'], depth: ['Team depth', 'm'], hull: ['Team area', 'm2'], spread: ['Spread', 'm'],
                 gapDM: ['Defence to midfield gap', 'm'], gapMF: ['Midfield to attack gap', 'm'], lineStd: ['Back-line unevenness', 'm'],
                 lineHeight: ['Back-line height', 'm'], nearestTeammate: ['Nearest teammate', 'm'], lagDM: ['Midfield lag behind the defence', 's'],
                 corrDM: ['Defence and midfield moving together', 'ratio']};
// The page's tokens (--text, --text-2, --amber, --slate, --danger) and the board's neutral and label halo.
const TOKENS = {text: '#F1F5F2', muted: '#9AAA9F', halo: '#07131bd9', amber: '#D9A961', slate: '#8FB3D9', danger: '#D96B6B', neutral: '#8d99a6'};
const ID_PATTERN = /(?<![\w-])(?:[AB]-(?:\d{1,3}|\?)|GK-[A-Z0-9]{1,3})(?!\w)/g;   // ids in report text: language.ID_TOKEN, one pass
const POLL_MS = 1000;
const NARROW = '(max-width: 1180px)';                    // the grid's one-column breakpoint: evidence goes under its finding

const clock = t => { t = Math.max(0, Number(t) || 0); const m = Math.floor(t / 60); return `${String(m).padStart(2, '0')}:${(t - m * 60).toFixed(2).padStart(5, '0')}`; };
const parseClock = text => { const m = String(text).trim().match(/^(?:(\d+):)?(\d+(?:\.\d+)?)$/); return m ? (m[1] ? Number(m[1]) * 60 : 0) + Number(m[2]) : null; };
const seconds = v => !Number.isFinite(v) ? '–' : v >= 90 ? `${(v / 60).toFixed(1)} min` : `${v.toFixed(1)} s`;
const words = key => String(key).replace(/([a-z0-9])([A-Z])/g, '$1 $2').replace(/_/g, ' ').toLowerCase().replace(/^./, c => c.toUpperCase());
const isNum = v => typeof v === 'number' && Number.isFinite(v);
const r2 = v => Math.round(v * 100) / 100;
// The one rounding of a typed range time: the status request sends toFixed(2) of this value and the build request the
// value itself, so both name the same hundredth (rounding twice, toFixed and Python's round, asked for 00:10.125 as 10.13
// and built it as 10.12, and the report was never found).
const hundredths = r2;
const UNPLACED = ['excluded', 'withdrawn'];             // match groups of identities that are no longer players (store.UNPLACED)

function el(tag, cls, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined && text !== null) e.textContent = String(text);
  return e;
}
async function saveFile(url) {
  // The PDF is fetched first, so a refusal (no report, no PDF writer) is shown as a message instead of downloaded.
  const response = await fetch(url);
  if (!response.ok) {
    let message = response.statusText || `HTTP ${response.status}`;
    try { message = (await response.json()).error || message; } catch (e) { /* not JSON */ }
    throw new Error(message);
  }
  const blob = await response.blob(), name = (/filename="?([^";]+)"?/.exec(response.headers.get('Content-Disposition') || '') || [])[1] || 'pitchiq-coaching.pdf';
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob); a.download = name; document.body.append(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}
const drillName = d => (d && (d.name || d.title)) || (d && d.id ? words(d.id) : 'Drill');
const verdictName = v => (VERDICTS.find(x => x[0] === v) || [0, String(v)])[1];
function button(text, cls, onclick, title) {
  const b = el('button', cls, text);
  b.type = 'button';
  if (title) b.title = title;
  if (onclick) b.onclick = onclick;
  return b;
}

// ---------- display labels ----------
function labelMap(result) {
  // id -> display of the result as it is now (store.labels_of): an identity a correction excluded or the tracker
  // withdrew shows as 'excluded (was A-03)', since after the renumbering its bare label belongs to another player (on a
  // BC60 copy with A-03 excluded, A-04 shows as A-03); a merged identity shows the label of the one it was merged into.
  const out = new Map();
  if (!result) return out;
  const m = result.match || {};
  for (const key of ['players', 'goalkeepers', 'referees']) for (const e of m[key] || []) if (e && e.id) out.set(e.id, e.display || e.id);
  for (const key of UNPLACED) {
    for (const e of m[key] || []) {
      if (!e || typeof e.id !== 'string' || out.has(e.id)) continue;
      out.set(e.id, `${key} (was ${typeof e.display === 'string' && e.display && e.display !== 'OUT' ? e.display : e.id})`);
    }
  }
  for (const s of ((result.roster || {}).slots) || []) if (s && s.id && !out.has(s.id)) out.set(s.id, s.display || s.id);
  for (const e of m.retired || []) if (e && e.id && !out.has(e.id)) out.set(e.id, out.get(e.retiredInto) || e.retiredInto || e.id);
  return out;
}
function plainLabels(text, labels) {
  return String(text === undefined || text === null ? '' : text).replace(ID_PATTERN, id => labels.has(id) ? labels.get(id) : id);
}
function labelNodes(text, labels) {
  // The text as nodes, every known id shown as its display label (the id in the tooltip).
  const s = String(text === undefined || text === null ? '' : text), out = [];
  let last = 0;
  for (const m of s.matchAll(ID_PATTERN)) {
    if (!labels.has(m[0])) continue;
    if (m.index > last) out.push(document.createTextNode(s.slice(last, m.index)));
    const shown = labels.get(m[0]), span = el('span', 'cr-id', shown);
    span.title = shown === m[0] ? m[0] : `${shown} (internal id ${m[0]})`;
    out.push(span);
    last = m.index + m[0].length;
  }
  if (last < s.length || !out.length) out.push(document.createTextNode(s.slice(last)));
  return out;
}

// ---------- drawing: one painter, two pens (SVG for the report, canvas for the board and the video) ----------
function svgPen(svg) {
  const add = (tag, attrs) => {
    const e = document.createElementNS(SVG_NS, tag);
    for (const [k, v] of Object.entries(attrs)) if (v !== undefined && v !== null) e.setAttribute(k, String(v));
    svg.append(e);
    return e;
  };
  const paint = s => ({fill: s.fill || 'none', 'fill-opacity': s.fillOpacity, stroke: s.stroke || 'none', 'stroke-width': s.width ? r2(s.width) : undefined,
                       'stroke-dasharray': s.dash ? s.dash.map(r2).join(' ') : undefined, 'stroke-linecap': 'round', 'stroke-linejoin': 'round'});
  return {
    circle: (x, y, r, s) => add('circle', {cx: r2(x), cy: r2(y), r: r2(r), ...paint(s)}),
    ellipse: (x, y, rx, ry, s) => add('ellipse', {cx: r2(x), cy: r2(y), rx: r2(rx), ry: r2(ry), ...paint(s)}),
    rect: (x, y, w, h, s) => add('rect', {x: r2(x), y: r2(y), width: r2(w), height: r2(h), ...paint(s)}),
    poly: (pts, s) => pts.length < 2 ? null : add(s.close ? 'polygon' : 'polyline', {points: pts.map(p => `${r2(p[0])},${r2(p[1])}`).join(' '), ...paint(s)}),
    text: (x, y, str, s) => {
      const e = add('text', {x: r2(x), y: r2(y), 'font-size': r2(s.size), 'text-anchor': s.anchor || 'middle', fill: s.fill || TOKENS.text, 'font-weight': s.weight || 600,
                             'font-family': 'Inter, Segoe UI, system-ui, sans-serif', stroke: s.halo, 'stroke-width': s.halo ? r2(s.size * .28) : undefined,
                             'paint-order': s.halo ? 'stroke' : undefined, 'stroke-linejoin': s.halo ? 'round' : undefined});
      e.textContent = str;
      return e;
    },
  };
}
function canvasPen(ctx) {
  const run = (s, path) => {
    ctx.save();
    ctx.lineCap = 'round'; ctx.lineJoin = 'round';
    ctx.beginPath(); path();
    if (s.fill && s.fill !== 'none') { ctx.globalAlpha = isNum(s.fillOpacity) ? s.fillOpacity : 1; ctx.fillStyle = s.fill; ctx.fill(); ctx.globalAlpha = 1; }
    if (s.stroke && s.stroke !== 'none') { ctx.setLineDash(s.dash || []); ctx.lineWidth = s.width || 1; ctx.strokeStyle = s.stroke; ctx.stroke(); }
    ctx.restore();
  };
  return {
    circle: (x, y, r, s) => run(s, () => ctx.arc(x, y, r, 0, Math.PI * 2)),
    ellipse: (x, y, rx, ry, s) => run(s, () => ctx.ellipse(x, y, rx, ry, 0, 0, Math.PI * 2)),
    rect: (x, y, w, h, s) => run(s, () => ctx.rect(x, y, w, h)),
    poly: (pts, s) => { if (pts.length < 2) return; run(s, () => { pts.forEach(([x, y], i) => i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)); if (s.close) ctx.closePath(); }); },
    text: (x, y, str, s) => {
      ctx.save();
      ctx.font = `${s.weight || 600} ${Math.round(s.size)}px Inter, Segoe UI, system-ui, sans-serif`;
      ctx.textAlign = s.anchor === 'start' ? 'left' : s.anchor === 'end' ? 'right' : 'center'; ctx.textBaseline = 'alphabetic';
      if (s.halo) { ctx.lineWidth = s.size * .28; ctx.strokeStyle = s.halo; ctx.lineJoin = 'round'; ctx.strokeText(str, x, y); }
      ctx.fillStyle = s.fill || TOKENS.text; ctx.fillText(str, x, y);
      ctx.restore();
    },
  };
}
const arc = (cx, cy, r, a0, a1, n = 24) => Array.from({length: n + 1}, (_, k) => { const a = a0 + (a1 - a0) * k / n; return [cx + r * Math.cos(a), cy + r * Math.sin(a)]; });

function alongAxis(points) {
  // Points in the order of their main direction: a unit's line runs across its players, not in id order.
  if (points.length < 3) return points.slice();
  const n = points.length, mx = points.reduce((a, p) => a + p[0], 0) / n, my = points.reduce((a, p) => a + p[1], 0) / n;
  let sxx = 0, syy = 0, sxy = 0;
  for (const [x, y] of points) { sxx += (x - mx) ** 2; syy += (y - my) ** 2; sxy += (x - mx) * (y - my); }
  const a = .5 * Math.atan2(2 * sxy, sxx - syy), ux = Math.cos(a), uy = Math.sin(a);
  return points.slice().sort((p, q) => ((p[0] - mx) * ux + (p[1] - my) * uy) - ((q[0] - mx) * ux + (q[1] - my) * uy));
}

const ORDER = ['band', 'zone', 'unit', 'measure', 'bracket', 'arrow', 'suggested', 'player', 'label'];

function paint(pen, spec, o) {
  // spec: a DiagramSpec (o.mode 'pitch', normalized pitch coordinates) or a VideoOverlaySpec (o.mode 'video', source
  // pixels). o: {toPx, line, font, radius (px), colours, labels (Map id -> display), positions (Map id -> spec
  // coordinates: the board's drawn markers), players (draw the markers; the board draws its own)}.
  const els = spec && Array.isArray(spec.elements) ? spec.elements.filter(e => e && typeof e.type === 'string') : [];
  const video = o.mode === 'video', P = (x, y) => o.toPx(x, y), labels = o.labels || new Map(), label = id => labels.get(id) || id;
  const own = new Map(), teams = new Map();
  for (const e of els) if (e.type === 'player' && isNum(e.x) && isNum(e.y)) { own.set(e.id, [e.x, e.y]); teams.set(e.id, e.team); }
  const at = id => (o.positions && o.positions.get(id)) || own.get(id) || null;
  const teamOf = id => teams.get(id) || (/^[AB]-/.test(String(id)) ? String(id)[0] : spec.team);
  const colour = team => (o.colours && o.colours[team]) || TOKENS.neutral;
  const tone = style => style === 'space' ? TOKENS.slate : TOKENS.amber;
  const pt = p => Array.isArray(p) && isNum(p[0]) && isNum(p[1]) ? P(p[0], p[1]) : null;
  const halo = {stroke: TOKENS.halo};
  const bracket = e => {
    // A team width or length as a bracket beside the team (soccer/coaching/render.py _bracket, the PDF's drawing): end
    // ticks pointing back at the team, dashed leaders to the two players that set the extent, the value outside. A
    // measure with bracket true already stands where it goes (evidence.py places it BRACKET_OFFSET outside the team); a
    // 'bracket' element's ends are the extent's own and it is moved beyond the team on its side (or the roomier one).
    let a = pt(e.a), b = pt(e.b);
    if (!a || !b) return;
    const len = Math.hypot(b[0] - a[0], b[1] - a[1]);
    if (len < 1e-6) return;
    const ux = (b[0] - a[0]) / len, uy = (b[1] - a[1]) / len, team = [];
    for (const id of own.keys()) if (teamOf(id) === spec.team) { const q = pt(at(id)); if (q) team.push(q); }
    let nx = -uy, ny = ux;
    const side = (q, sx, sy) => (q[0] - a[0]) * sx + (q[1] - a[1]) * sy;
    if (e.type === 'bracket') {
      const want = {left: [-1, 0], right: [1, 0], top: [0, -1], bottom: [0, 1]}[e.side];
      const ext = s => Math.max(0, ...team.map(q => side(q, nx * s, ny * s)));
      let sign = want ? Math.sign(want[0] * nx + want[1] * ny) : 0;
      if (!sign) sign = ext(1) <= ext(-1) ? 1 : -1;      // the side the team reaches least far beyond the extent's ends
      nx *= sign; ny *= sign;
      const d = ext(1) + o.radius * 2.4;
      a = [a[0] + nx * d, a[1] + ny * d]; b = [b[0] + nx * d, b[1] + ny * d];
    } else if (team.length) {
      const cx = team.reduce((s, q) => s + q[0], 0) / team.length, cy = team.reduce((s, q) => s + q[1], 0) / team.length;
      if (side([cx, cy], nx, ny) > 0) { nx = -nx; ny = -ny; }   // outward: away from the team's centre
    }
    const tick = o.font * .45;
    for (const s of [{...halo, width: o.line * 2.4}, {stroke: TOKENS.text, width: o.line}]) {
      pen.poly([a, b], s);
      for (const p of [a, b]) pen.poly([p, [p[0] - nx * tick, p[1] - ny * tick]], s);
    }
    const ids = (Array.isArray(e.ids) ? e.ids : []).filter(id => typeof id === 'string' && at(id));
    for (const p of [a, b]) {
      // the end's player: the named one standing nearest along the bracket
      const near = ids.map(id => pt(at(id))).filter(Boolean).sort((q, r) => Math.abs(side(q, ux, uy) - side(p, ux, uy)) - Math.abs(side(r, ux, uy) - side(p, ux, uy)))[0];
      if (!near) continue;
      const from = [p[0] - nx * tick, p[1] - ny * tick], reach = (from[0] - near[0]) * nx + (from[1] - near[1]) * ny - o.radius * 1.4;
      if (reach > o.line) pen.poly([from, [from[0] - nx * reach, from[1] - ny * reach]], {stroke: TOKENS.muted, width: o.line * .6, dash: [o.line * 1.6, o.line * 1.3]});
    }
    if (e.text) {
      const mx = (a[0] + b[0]) / 2, my = (a[1] + b[1]) / 2, across = Math.abs(nx) > .7, txt = plainLabels(e.text, labels), size = o.font * .85;
      if (across) pen.text(mx + nx * o.font * .6, my + size * .35, txt, {size, halo: TOKENS.halo, anchor: nx > 0 ? 'start' : 'end'});
      else pen.text(mx, my + (ny > 0 ? size * 1.15 : -size * .45), txt, {size, halo: TOKENS.halo});
    }
  };
  for (const type of ORDER) for (const e of els) {
    if (e.type !== type) continue;
    if (type === 'band' && !video && isNum(e.from) && isNum(e.to)) {
      const c = tone(e.style), q = e.axis === 'y' ? [[0, e.from], [1, e.from], [1, e.to], [0, e.to]] : [[e.from, 0], [e.to, 0], [e.to, 1], [e.from, 1]];
      pen.poly(q.map(p => P(p[0], p[1])), {close: true, fill: c, fillOpacity: .16, stroke: c, width: o.line * .7, dash: [o.line * 3, o.line * 2.5]});
    } else if (type === 'zone' && Array.isArray(e.polygon)) {
      const q = e.polygon.map(pt).filter(Boolean), c = tone(e.style);
      if (q.length >= 3) pen.poly(q, {close: true, fill: c, fillOpacity: .2, stroke: c, width: o.line * .8, dash: [o.line * 2.5, o.line * 2]});
    } else if (type === 'unit' && Array.isArray(e.ids)) {
      const q = alongAxis(e.ids.map(at).filter(Boolean)).map(p => P(p[0], p[1]));
      if (q.length >= 2) { pen.poly(q, {...halo, width: o.line * 2.6}); pen.poly(q, {stroke: colour(teamOf(e.ids[0])), width: o.line * 1.4}); }
    } else if ((type === 'measure' && e.bracket) || type === 'bracket') {
      if (!video) bracket(e);
      else {      // on the video a bracket sits where its ends say (render.py's video overlay): the line and its value
        const a = pt(e.a), b = pt(e.b);
        if (!a || !b) continue;
        for (const s of [{...halo, width: o.line * 2.4}, {stroke: TOKENS.text, width: o.line}]) pen.poly([a, b], s);
        if (e.text) pen.text((a[0] + b[0]) / 2, (a[1] + b[1]) / 2 - o.font * .3, plainLabels(e.text, labels), {size: o.font * .85, halo: TOKENS.halo});
      }
    } else if (type === 'measure') {
      const a = pt(e.a), b = pt(e.b);
      if (!a || !b) continue;
      const len = Math.hypot(b[0] - a[0], b[1] - a[1]) || 1, nx = -(b[1] - a[1]) / len, ny = (b[0] - a[0]) / len, tick = o.font * .45;
      for (const s of [{...halo, width: o.line * 2.4}, {stroke: TOKENS.text, width: o.line}]) {
        pen.poly([a, b], s);
        for (const p of [a, b]) pen.poly([[p[0] - nx * tick, p[1] - ny * tick], [p[0] + nx * tick, p[1] + ny * tick]], s);
      }
      if (e.text) pen.text((a[0] + b[0]) / 2 + nx * o.font * .9, (a[1] + b[1]) / 2 + ny * o.font * .9 + o.font * .35, plainLabels(e.text, labels), {size: o.font * .85, halo: TOKENS.halo});
    } else if (type === 'arrow' && Array.isArray(e.points)) {
      const q = e.points.map(pt).filter(Boolean);
      if (q.length < 2) continue;
      const c = colour(teamOf(e.id)), [x1, y1] = q[q.length - 1], [x0, y0] = q[q.length - 2], ang = Math.atan2(y1 - y0, x1 - x0), h = o.font * .75;
      pen.poly(q, {...halo, width: o.line * 2.6}); pen.poly(q, {stroke: c, width: o.line * 1.3});
      pen.poly([[x1, y1], [x1 - h * Math.cos(ang - .45), y1 - h * Math.sin(ang - .45)], [x1 - h * Math.cos(ang + .45), y1 - h * Math.sin(ang + .45)]], {close: true, fill: c, stroke: TOKENS.halo, width: o.line * .5});
    } else if (type === 'suggested' && isNum(e.x) && isNum(e.y)) {
      const [x, y] = P(e.x, e.y), r = video ? o.radius * 1.2 : o.radius * 1.35;
      pen.circle(x, y, r, {stroke: TOKENS.slate, width: o.line, dash: [o.line * 2, o.line * 1.6]});
      pen.text(x, y - r - o.font * .35, e.label || 'COACHING ILLUSTRATION', {size: o.font * .62, fill: TOKENS.slate, halo: TOKENS.halo});
    } else if (type === 'player' && isNum(e.x) && isNum(e.y)) {
      const p = at(e.id) || [e.x, e.y], [x, y] = P(p[0], p[1]), c = colour(e.team || teamOf(e.id)), observed = !e.kind || e.kind === 'O';
      if (video) {
        const box = Array.isArray(e.box) && e.box.length === 4 && e.box.every(isNum) ? e.box : null, rx = box ? Math.max(box[2] * .75, o.radius * .6) : o.radius, ry = rx * .32;
        pen.ellipse(x, y, rx, ry, {...halo, width: o.line * 2.2});
        pen.ellipse(x, y, rx, ry, {stroke: c, width: o.line * (e.emphasis ? 1.4 : .9), dash: observed ? null : [o.line * 2, o.line * 1.6]});
        if (e.emphasis) pen.text(x, (box ? box[1] : y - o.radius * 3.4) - o.font * .4, label(e.id), {size: o.font, fill: TOKENS.text, halo: TOKENS.halo});
      } else if (o.players) {
        pen.circle(x, y, o.radius, observed ? {fill: c, stroke: TOKENS.halo, width: o.line * .5} : {stroke: c, width: o.line * .6, dash: [o.line * 1.5, o.line * 1.2]});
        if (e.emphasis) pen.circle(x, y, o.radius * 1.65, {stroke: TOKENS.text, width: o.line * .7});
        pen.text(x, y - o.radius * (e.emphasis ? 1.75 : 1.25) - o.font * .2, label(e.id), {size: o.font * .75, halo: TOKENS.halo});
      } else if (e.emphasis) {
        pen.circle(x, y, o.radius * 1.9, {stroke: TOKENS.text, width: o.line * .8});
      }
    } else if (type === 'label' && pt(e.at) && e.text) {
      const [x, y] = pt(e.at);
      pen.text(x, y, plainLabels(e.text, labels), {size: o.font * .8, halo: TOKENS.halo, fill: TOKENS.text});
    }
  }
}

function pitchLines(pen, L, W, M) {
  // The board's pitch (tactical.js drawPitch), in metres.
  const P = (global.Tactical && global.Tactical.PITCH) || {penaltyDepth: 16.5, penaltyHalf: 20.16, goalDepth: 5.5, goalHalf: 9.16, spot: 11, circle: 9.15, goalWidth: 7.32, goalNet: 2.2, cornerArc: 1};
  pen.rect(0, 0, L + 2 * M, W + 2 * M, {fill: '#1e6b36'});
  for (let i = 0; i < 10; i++) pen.rect(M + L * i / 10, M, L / 10 + .05, W, {fill: i % 2 ? '#2b8a46' : '#30944d'});
  const s = {stroke: '#f4f7f4', width: .22};
  pen.rect(M, M, L, W, s);
  pen.poly([[M + L / 2, M], [M + L / 2, M + W]], s);
  pen.poly(arc(M + L / 2, M + W / 2, P.circle, 0, Math.PI * 2, 48), s);
  pen.circle(M + L / 2, M + W / 2, .35, {fill: '#f4f7f4'});
  for (const [x0, sign] of [[0, 1], [L, -1]]) {
    for (const [depth, half] of [[P.penaltyDepth, P.penaltyHalf], [P.goalDepth, P.goalHalf]]) {
      const fx = x0 + sign * depth;
      pen.poly([[M + x0, M + W / 2 - half], [M + fx, M + W / 2 - half], [M + fx, M + W / 2 + half], [M + x0, M + W / 2 + half]], s);
    }
    pen.circle(M + x0 + sign * P.spot, M + W / 2, .3, {fill: '#f4f7f4'});
    const spread = Math.acos((P.penaltyDepth - P.spot) / P.circle), cx = M + x0 + sign * P.spot;
    pen.poly(sign > 0 ? arc(cx, M + W / 2, P.circle, -spread, spread) : arc(cx, M + W / 2, P.circle, Math.PI - spread, Math.PI + spread), s);
    pen.rect(sign > 0 ? M - P.goalNet : M + L, M + W / 2 - P.goalWidth / 2, P.goalNet, P.goalWidth, {...s, fill: '#ffffff', fillOpacity: .13});
  }
}

function svgRoot(viewBox, cls, title) {
  const svg = document.createElementNS(SVG_NS, 'svg');
  svg.setAttribute('viewBox', viewBox);
  svg.setAttribute('class', cls);
  svg.setAttribute('role', 'img');
  if (title) svg.setAttribute('aria-label', title);
  return svg;
}

function svgDiagram(spec, colours, labels) {
  // The top-down diagram of a DiagramSpec as SVG (in metres; the board's 105 x 68 m with its margin).
  const L = BOARD.length, W = BOARD.width, M = BOARD.margin, ls = labels || new Map();
  const svg = svgRoot(`0 0 ${L + 2 * M} ${W + 2 * M + 4}`, 'cr-diagram', plainLabels(spec.title || 'Diagram', ls)), pen = svgPen(svg);
  pen.rect(0, 0, L + 2 * M, W + 2 * M + 4, {fill: '#14211b'});
  pitchLines(pen, L, W, M);
  paint(pen, spec, {mode: 'pitch', toPx: (x, y) => [M + x * L, M + y * W], line: .35, font: 3.1, radius: 1.6, colours, labels: ls, players: true});
  if (spec.attack === 'left' || spec.attack === 'right') {
    const y = W + 2 * M + 1.8, right = spec.attack === 'right';
    pen.text(L / 2 + M, y, `Team ${spec.team || ''} attacks ${right ? '→' : '←'}`.replace('  ', ' '), {size: 2.6, fill: TOKENS.muted, weight: 500});
  }
  return svg;
}

function svgOverlay(spec, colours, labels) {
  // A VideoOverlaySpec over the screenshot: the SVG has the source video's pixel size, stretched over the image.
  const W = spec.width, H = spec.height, ls = labels || new Map(), u = W / 640;
  const svg = svgRoot(`0 0 ${W} ${H}`, 'cr-overlay', 'Overlay'), pen = svgPen(svg);
  svg.setAttribute('preserveAspectRatio', 'none');
  paint(pen, spec, {mode: 'video', toPx: (x, y) => [x, y], line: 1.6 * u, font: 13 * u, radius: 11 * u, colours, labels: ls, players: true});
  return svg;
}

function validOverlay(spec) {
  return !!spec && isNum(spec.width) && isNum(spec.height) && spec.width > 0 && spec.height > 0 && Array.isArray(spec.elements);
}

// For the page: a finding's diagram on the 2D board (a tactical.overlays layer, under the board's own markers: the
// units run through the drawn markers, the players involved get a ring) and its overlay on the video canvas.
function drawBoard(ctx, g, spec, colours, state, labels) {
  if (!spec || !Array.isArray(spec.elements)) return;
  const positions = new Map();
  for (const p of (state && state.players) || []) if (p.id && isNum(p.x) && isNum(p.y)) positions.set(p.id, [p.x, p.y]);
  paint(canvasPen(ctx), spec, {mode: 'pitch', toPx: g.toPx, line: Math.max(1.5, g.scale * .3), font: Math.max(11, g.playerRadius * 1.15), radius: g.playerRadius,
                               colours, labels: labels || new Map(), positions, players: false});
}
function drawVideo(ctx, spec, width, height, colours, labels) {
  if (!validOverlay(spec)) return;
  const s = width / spec.width, u = spec.width / 640 * s;
  paint(canvasPen(ctx), spec, {mode: 'video', toPx: (x, y) => [x * s, y * (height / spec.height)], line: 1.6 * u, font: 13 * u, radius: 11 * u,
                               colours, labels: labels || new Map(), players: true});
}

function metricText(name, value, unit, estimated) {
  const known = METRICS[name], u = unit || (known ? known[1] : '');
  let v;
  if (!isNum(value)) v = value === undefined || value === null ? '–' : String(value);
  else if (u === 'm') v = `${value.toFixed(1)} m${estimated ? ' (est.)' : ''}`;
  else if (u === 'm2') v = `${Math.round(value)} m²${estimated ? ' (est.)' : ''}`;
  else if (u === 's') v = value >= 90 ? `${(value / 60).toFixed(1)} min` : `${value.toFixed(1)} s`;
  else if (u === 'share' || u === '%') v = `${(value * 100).toFixed(value * 100 >= 10 || Number.isInteger(value * 100) ? 0 : 1)}%`;
  else if (u === 'per_min') v = `${value.toFixed(2)} per min`;
  else if (u === 'count') v = String(Math.round(value));
  else if (u === 'ratio' || u === 'label' || !u) v = Number.isInteger(value) ? String(value) : value.toFixed(2);
  else v = `${Number.isInteger(value) ? value : value.toFixed(1)} ${u}`;
  return [known ? known[0] : words(name), v];
}

// ---------- the view ----------
class CoachingView {
  constructor(options) {
    this.main = options.main; this.side = options.side; this.banner = options.banner; this.c = options.controls;
    this.api = options.api; this.post = options.post; this.video = options.video; this.board = options.board;
    this.runName = options.run || (() => null);
    this.jumpTo = options.jump || (() => false);
    this.download = options.download || saveFile;
    this.media = options.media !== undefined ? options.media : (typeof global.matchMedia === 'function' ? global.matchMedia(NARROW) : null);
    this.home = this.side.parentElement || null;     // the grid column the evidence lives in on wide screens
    this.result = null; this.info = null; this.labels = new Map(); this.loadedRun = undefined;
    this.cache = new Map(); this.answer = null; this.seq = 0; this.gen = 0; this.dataGen = -1; this.active = false; this.loading = false;
    // The selection the shown answer belongs to (Export PDF saves that one), its cache key and JSON, and the gen the
    // report was last drawn for: an answer identical to the one drawn redraws only the bar and the banner.
    this.answerOptions = null; this.answerKey = null; this.answerSig = null; this.renderedGen = -1;
    this.selected = null; this.selectedEvidence = null; this.cards = new Map(); this.boxes = new Map(); this.drafts = new Map(); this.notes = new Map();
    this.timer = null; this.message = '';
    this.c.team.addEventListener('change', () => this.request());
    this.c.range.addEventListener('change', () => { this.syncRange(true); this.request(); });
    for (const input of [this.c.t0, this.c.t1]) input.addEventListener('change', () => this.request());
    this.c.context.addEventListener('change', () => this.request());
    this.c.generate.addEventListener('click', () => this.generate());
    this.c.pdf.addEventListener('click', () => this.exportPdf());
    for (const input of options.colourInputs || []) input.addEventListener('input', () => this.renderSide());
    if (this.media && typeof this.media.addEventListener === 'function') this.media.addEventListener('change', () => this.placeSide());
    this.syncRange(false);
    this.render();
  }

  label(id) { return this.labels.get(id) || id; }
  colours() { return this.board && this.board.colours ? this.board.colours : {A: '#b98cff', B: '#f6b564'}; }
  fps() { return (this.info && this.info.video && this.info.video.fps) || 25; }
  bounds() { const f = this.result && this.result.frames; return f && f.length ? [f[0].time, f[f.length - 1].time] : [0, 0]; }

  // ---------- selection and data ----------
  load(result, info) {
    // Every showResult (a new run, or the same run after a correction): the answers cached for the earlier result are
    // dropped; unsaved reviews survive a correction of the same run, not a new run.
    const run = this.runName();
    if (run !== this.loadedRun) { this.drafts.clear(); this.notes.clear(); this.selected = null; this.selectedEvidence = null; this.loadedRun = run; }
    this.result = result; this.info = info; this.labels = labelMap(result);
    this.cache.clear(); this.answer = null; this.answerOptions = null; this.answerKey = null; this.answerSig = null;
    this.gen++; this.seq++; this.loading = false; this.stopPoll();     // an answer still in flight belongs to the earlier result
    this.syncRange(false);
    if (this.active && result) this.request(); else this.render();
  }

  setActive(on) {
    // Called when the coaching view is shown or hidden, and a no-op otherwise: a window resize re-marks the views, and
    // re-asking then rebuilt the whole report (5 resizes replaced the note a coach was typing in and every screenshot).
    // Shown again, the report is asked again, since a team-shape save on the Analytics tab (which reloads no result)
    // can have made it stale; the status request is cheap (the server hashes each result version once).
    on = !!on;
    if (on === this.active) return;
    this.active = on;
    if (!on) { this.stopPoll(); return; }
    this.cache.clear();
    if (this.result) this.request(); else this.render();
  }

  syncRange(fill) {
    const custom = this.c.range.value === 'custom';
    this.c.custom.hidden = !custom;
    if (this.result) {
      const [a, b] = this.bounds();
      this.c.custom.title = `Source-video time (the video’s own clock, as on the timestamps). The analysed segment runs ${clock(a)}–${clock(b)}.`;
      if (custom && fill) { this.c.t0.value = clock(a); this.c.t1.value = clock(b); }
    }
  }

  options() {
    const out = {team: this.c.team.value === 'B' ? 'B' : 'A', context: CONTEXTS[this.c.context.value] ? this.c.context.value : 'competitive', t0: null, t1: null};
    if (this.c.range.value === 'custom') {
      const a = parseClock(this.c.t0.value), b = parseClock(this.c.t1.value);
      if (a !== null && b !== null) { out.t0 = hundredths(Math.min(a, b)); out.t1 = hundredths(Math.max(a, b)); } else out.invalid = true;
    }
    return out;
  }
  key(o) { return `${o.team}|${o.context}|${o.t0}|${o.t1}`; }
  params(o, extra) {
    const p = new URLSearchParams({run: this.runName() || '', team: o.team, context: o.context});
    if (o.t0 !== null) { p.set('t0', o.t0.toFixed(2)); p.set('t1', o.t1.toFixed(2)); }
    for (const [k, v] of Object.entries(extra || {})) p.set(k, v);
    return p;
  }

  async request() {
    // Every request supersedes the ones still in flight, also one answered at once (from the cache, or an unreadable
    // range): their late answers fail the seq check. Before, a cache hit left an older request current: with team B
    // asked and A then shown from the cache, B's late answer replaced A's report while the selector said A, and the
    // stuck loading flag kept Generate disabled.
    this.stopPoll(); this.message = '';
    const seq = ++this.seq, gen = this.gen;
    if (!this.result) { this.loading = false; this.show(null, null, null, gen); return; }
    const o = this.options(), key = this.key(o);
    if (o.invalid) {
      this.loading = false;
      this.show({state: 'error', error: 'Enter the custom range as two times (mm:ss.ss, source-video time) inside the analysed segment.', report: null, feedback: [], reasons: []}, o, null, gen);
      return;
    }
    if (this.cache.has(key)) { this.loading = false; this.show(this.cache.get(key), o, key, gen); return; }
    this.loading = true; this.renderBar();
    let data, failed = false;
    try {
      data = await this.api('/api/coaching?' + this.params(o).toString());
    } catch (error) {
      data = {state: 'error', error: error.message, report: null, feedback: [], reasons: []}; failed = true;
    }
    if (seq !== this.seq || gen !== this.gen) return;     // a newer selection, result or save superseded this one
    // A build in progress, this selection's or another one's (busy), is asked again until it ends and never cached: a
    // cached 'busy' kept Generate disabled after the other build had finished, and nothing asked again.
    const waiting = !failed && (data.state === 'building' || !!data.busy);
    if (!failed && !waiting) this.cache.set(key, data);
    this.loading = false;
    this.show(data, o, key, gen);
    if (waiting && this.active) this.timer = setTimeout(() => { this.timer = null; this.request(); }, POLL_MS);
  }
  show(answer, options, key, gen) {
    // Shows an answer; one identical to the answer drawn for the same selection and result (a poll during a build, the
    // tab shown again) redraws only the bar and the banner, so a note being typed and the loaded screenshots stay.
    const sig = answer ? JSON.stringify(answer) : null;
    const same = key !== null && key === this.answerKey && sig === this.answerSig && this.renderedGen === gen;
    this.answer = answer; this.answerOptions = options; this.answerKey = key; this.answerSig = sig; this.dataGen = gen;
    if (same) { this.renderBar(); this.renderBanner(); } else this.render();
  }
  stopPoll() { if (this.timer) { clearTimeout(this.timer); this.timer = null; } }

  async generate() {
    const a = this.answer;
    if (!this.result || this.loading || (a && (a.state === 'building' || a.busy))) return;
    const o = this.options(), body = {run: this.runName(), team: o.team, context: o.context};
    if (o.invalid) return;
    if (o.t0 !== null) { body.t0 = o.t0; body.t1 = o.t1; }     // the hundredths the status request asked for
    this.message = ''; this.c.generate.disabled = true;
    try {
      await this.post('/api/coaching/build', body);
    } catch (error) {
      this.message = error.message; this.render(); return;
    }
    this.cache.clear(); this.gen++;
    await this.request();
  }

  async exportPdf() {
    // The PDF of the report on screen: the selection its answer was asked for, never one still loading.
    if (!this.answer || !this.answer.report || this.loading) return;
    const o = this.answerOptions && !this.answerOptions.invalid ? this.answerOptions : this.options();
    this.message = ''; this.c.pdf.disabled = true;
    try {
      await this.download('/api/coaching/pdf?' + this.params(o).toString());
    } catch (error) {
      this.message = `The PDF could not be made: ${error.message}`;
    }
    this.renderBar(); this.renderBanner();
  }

  async sendFeedback(observation) {
    const a = this.answer, draft = this.drafts.get(observation);
    if (!a || !a.report || !draft || !draft.verdict) return;
    this.notes.set(observation, 'Saving…'); this.refreshFeedback(observation);
    try {
      await this.post('/api/coaching/feedback', {run: this.runName(), observation, verdict: draft.verdict, text: draft.text || '', fingerprint: a.fingerprint, team: a.report.team});
    } catch (error) {
      this.notes.set(observation, error.message); this.refreshFeedback(observation); return;
    }
    this.drafts.delete(observation); this.notes.set(observation, 'Saved. Your review is kept with this video and never changes the measurements.');
    this.cache.clear(); this.gen++;
    await this.request();
  }

  async removeFeedback(entry) {
    try {
      await this.post('/api/coaching/feedback/remove', {id: entry.id});
    } catch (error) {
      this.notes.set(entry.attached, error.message); this.refreshFeedback(entry.attached); return;
    }
    this.drafts.delete(entry.attached); this.notes.set(entry.attached, 'Review removed.');
    this.cache.clear(); this.gen++;
    await this.request();
  }

  // ---------- lookups ----------
  report() { return this.answer && this.answer.report ? this.answer.report : null; }
  evidenceOf(id) { const r = this.report(); return r && Array.isArray(r.evidence) ? r.evidence.find(e => e && e.id === id) || null : null; }
  observationOf(id) { const r = this.report(); return r && Array.isArray(r.observations) ? r.observations.find(o => o && o.id === id) || null : null; }
  findings() { const r = this.report(); return r ? GROUPS.flatMap(([key]) => (Array.isArray(r[key]) ? r[key] : []).filter(f => f && f.observation)) : []; }
  findingOf(id) { return this.findings().find(f => f.observation === id) || null; }
  estimated() { const r = this.report(); return !r || (r.metresEstimated !== false && ((r.overview || {}).coverage || {}).metresEstimated !== false); }   // metres from a homography on an assumed pitch: estimates unless the report says otherwise
  feedbackFor(id) { return ((this.answer && this.answer.feedback) || []).filter(f => f.attached === id); }

  // ---------- rendering ----------
  render() {
    this.renderBar(); this.renderBanner(); this.renderMain(); this.renderSide();
  }

  renderBar() {
    const a = this.answer, st = a ? a.state : null, b = this.c.generate;
    b.disabled = !this.result || this.loading || st === 'building' || st === 'ready' || !!(a && a.busy);
    b.textContent = st === 'building' ? 'Building…' : st === 'ready' ? 'Report up to date' : st === 'stale' ? 'Regenerate' : st === 'error' && a && a.report ? 'Regenerate' : 'Generate report';
    b.classList.toggle('primary', st !== 'ready');
    b.title = st === 'stale' ? 'Build the report again from the result as it is now' : st === 'ready' ? 'The report matches the result, the settings and the engine' : 'Build the report from this run’s tracking';
    this.c.pdf.disabled = !(a && a.report) || st === 'building' || this.loading;
  }

  renderBanner() {
    const a = this.answer, box = this.banner;
    box.replaceChildren(); box.className = 'cr-banner';
    let tone = null;
    if (this.message) { tone = 'error'; box.append(el('span', null, this.message)); }
    else if (!a || !this.result) tone = null;
    else if (a.state === 'stale') {
      tone = 'stale';
      box.append(el('b', null, 'This report is out of date.'), el('span', null, ' The findings below describe the result as it was when the report was made. Regenerate to update them:'));
      const list = el('ul');
      for (const r of a.reasons || []) list.append(el('li', null, r));
      box.append(list);
    } else if (a.state === 'building') {
      tone = 'info';
      box.append(el('span', null, a.report ? 'Building the report again from this run’s tracking; the earlier version stays shown until it is ready.' : 'Building the report from this run’s tracking…'));
    } else if (a.state === 'error') {
      tone = 'error';
      box.append(el('span', null, a.error || 'The report could not be built.'));
    } else if (a.busy) {
      tone = 'info';
      box.append(el('span', null, 'Another coaching report is being built; this one can be generated when it has finished (checked every second).'));
    }
    const note = this.rangeNote();
    if (note) { tone = tone || 'info'; box.append(el('span', 'cr-range-note', note)); }
    box.hidden = !tone;
    if (tone) box.classList.add(tone);
  }

  rangeNote() {
    // A custom range is typed in source-video time and the server clamps it to the analysed segment: say so when that
    // changed it (on BC180, 00:30-01:30 silently became 00:42.43-01:30.00).
    const a = this.answer, o = this.answerOptions, given = a && a.options;
    if (!this.result || !o || o.invalid || o.t0 === null || !given || typeof given !== 'object') return null;
    const [lo, hi] = this.bounds(), typed = `${clock(o.t0)}–${clock(o.t1)}`, segment = `${clock(lo)}–${clock(hi)}, source-video time`;
    if (given.t0 === null || given.t0 === undefined) return `The range ${typed} covers the whole analysed segment (${segment}), so this is the whole-segment report.`;
    if (isNum(given.t0) && isNum(given.t1) && (Math.abs(given.t0 - o.t0) > .005 || Math.abs(given.t1 - o.t1) > .005))
      return `The range ${typed} reaches outside the analysed segment (${segment}), so the report covers ${clock(given.t0)}–${clock(given.t1)}.`;
    return null;
  }

  section(key, title, intro) {
    const s = el('section', 'cr-sec');
    s.dataset.section = key;
    const h = el('h3', null, title);
    s.append(h);
    if (intro) s.append(el('p', 'cr-intro', intro));
    return s;
  }
  para(lead, text, cls) {
    const p = el('p', cls || null);
    if (lead) p.append(el('b', 'lead', lead + ' '));
    p.append(...labelNodes(text, this.labels));
    return p;
  }

  renderMain() {
    const main = this.main;
    main.replaceChildren(); this.cards = new Map(); this.boxes = new Map(); this.renderedGen = this.gen;
    const a = this.answer, r = this.report();
    if (!this.result) { main.append(this.empty('Coaching reports appear here after a run.')); return; }
    if (!r) {
      if (!a || this.loading) main.append(this.empty('Loading the coaching report…'));
      else if (a.state === 'building') main.append(this.empty('Building the report: the observations are measured from this run’s tracking and compared with the professional reference. This takes a few seconds.'));
      else if (a.state === 'error') main.append(this.empty(a.error || 'The report could not be built.'));
      else {
        // The range as the server took it (clamped to the analysed segment), else as typed.
        const o = this.options(), given = a.options && typeof a.options === 'object' && 't0' in a.options ? a.options : o;
        const box = this.empty(`No coaching report yet for Team ${o.team} (${given.t0 === null ? 'whole analysed segment' : `${clock(given.t0)}–${clock(given.t1)}`}, ${CONTEXTS[o.context].toLowerCase()} level).`);
        box.append(el('p', 'cr-intro', 'Generate builds it from this run’s tracking: team shape, spacing and line patterns measured over the segment and compared with a professional reference. Only what the tracking measured is reported; anything measured too briefly is listed as not assessed.'));
        main.append(box);
      }
      return;
    }
    // The PDF prints the same order (soccer/coaching/pdf.py section_order): overview, summary, the three groups of
    // findings, breakdown, priorities, training, coverage and confidence, then the feedback that sits on no finding.
    main.append(this.overview(r), this.summary(r));
    for (const [key, title, intro, none] of GROUPS) main.append(this.group(r, key, title, intro, none));
    main.append(this.breakdown(r), this.priorities(r), this.training(r), this.confidence(r));
    const apart = this.feedbackSection(r);
    if (apart) main.append(apart);
    if (!this.selected || !this.findingOf(this.selected)) {
      const first = ['improvements', 'reviews', 'strengths'].map(k => (Array.isArray(r[k]) ? r[k] : []).find(f => f && (f.evidence || []).length)).find(Boolean);
      this.selected = first ? first.observation : null; this.selectedEvidence = null;
    }
    for (const [id, card] of this.cards) card.classList.toggle('selected', id === this.selected);
  }

  empty(text) { const box = el('div', 'cr-sec cr-empty'); box.dataset.section = 'empty'; box.append(el('p', null, text)); return box; }

  overview(r) {
    const s = this.section('overview', 'Overview'), ov = r.overview || {}, range = r.range || {};
    const head = el('div', 'cr-ov-head');
    head.append(el('div', 'kicker', ov.title || (r.kind === 'match' ? 'MATCH COACHING REPORT' : 'SEGMENT COACHING REPORT')),
                el('div', 'cr-team', r.teamName || `Team ${r.team} (as labelled in this run)`));
    s.append(head);
    const rows = [];
    if (isNum(range.t0) && isNum(range.t1)) rows.push(['Range', `${clock(range.t0)}–${clock(range.t1)}` + (isNum(range.segmentT0) ? ` (segment ${clock(range.segmentT0)}–${clock(range.segmentT1)})` : '')]);
    if (isNum(range.seconds)) rows.push(['Length', seconds(range.seconds)]);
    rows.push(['Report', r.kind === 'match' ? 'Match report' : 'Segment report (under 20 minutes measured: no recurring-weakness claims)']);
    rows.push(['Context level', `${CONTEXTS[r.context] || words(r.context || 'competitive')} (changes the wording, never the measurements)`]);
    const value = (k, v) => isNum(v) ? (/seconds$/i.test(k) ? seconds(v) : /share$/i.test(k) ? `${Math.round(v * 100)}%` : String(r2(v)))
      : typeof v === 'boolean' ? (v ? 'yes' : 'no') : String(v);
    for (const [k, v] of Object.entries(ov.coverage || {})) {
      if (v === null || v === undefined || Array.isArray(v) || k === 'rangeSeconds') continue;   // the range's length is its own row
      if (typeof v === 'object') {
        // A group of counts or times (observations, measured seconds per category): one row.
        const parts = Object.entries(v).filter(([, x]) => isNum(x) || typeof x === 'string').map(([kk, x]) => `${words(kk).toLowerCase()} ${value(/seconds$/i.test(k) ? k : kk, x)}`);
        if (parts.length) rows.push([words(k), parts.join(' \u00b7 ')]);
        continue;
      }
      rows.push([words(k), value(k, v)]);
    }
    if (this.answer && this.answer.saved) rows.push(['Generated', String(this.answer.saved).replace('T', ' ')]);
    for (const [k, v] of rows) s.append(this.kv(k, v));
    // The run's data-quality note (soccer/coaching/report.py: the first overview.dataQualityLines limitations, stated once
    // with what would help) belongs with the overview: it says how far every finding below can be read. The rest of the
    // limitations, one line per measurement, stay under Confidence and limitations.
    const quality = this.qualityLines(r);
    if (quality.length) { s.append(el('h4', 'cr-sub-h', 'Data quality')); s.append(this.list(quality, 'cr-list cr-quality')); }
    // The context paragraph: block height of both teams, then where the ball was (descriptive, never a finding).
    const context = Array.isArray(ov.context) ? ov.context : [];
    if (context.length) { s.append(el('h4', 'cr-sub-h', 'Context')); s.append(this.list(context)); }
    return s;
  }

  qualityLines(r) {
    const ov = (r && r.overview) || {}, lim = Array.isArray(ov.limitations) ? ov.limitations : [];
    const n = Number.isInteger(ov.dataQualityLines) && ov.dataQualityLines > 0 ? Math.min(ov.dataQualityLines, lim.length) : 0;
    return lim.slice(0, n).filter(t => typeof t === 'string' && t);
  }

  limitationLines(r) {
    // The limitations after the data-quality note, with the bare "X: not assessed on this footage." lines of a footage that
    // shows a few players per team (the Veo clip: ten of them under one data-quality block) joined into one line.
    const ov = (r && r.overview) || {}, lim = Array.isArray(ov.limitations) ? ov.limitations : [];
    // (pdf.py join_bare does the same, so page and PDF list the same lines)
    const rest = lim.slice(this.qualityLines(r).length).filter(t => typeof t === 'string' && t), BARE = /^(.+): not assessed on this footage\.$/;
    const bare = rest.map(t => BARE.exec(t)).filter(Boolean).map(m => m[1]);
    if (bare.length < 2) return rest;
    const joined = `Not assessed on this footage: ${bare.map((x, i) => i ? x.charAt(0).toLowerCase() + x.slice(1) : x).join(', ')}.`;
    return rest.filter((t, i) => !BARE.test(t) || rest.findIndex(x => BARE.test(x)) === i).map(t => BARE.test(t) ? joined : t);
  }

  kv(label, value, low) {
    const row = el('div', 'kv'), v = el('b', low ? 'low' : null);
    v.append(...labelNodes(value, this.labels));
    row.append(el('span', null, label), v);
    return row;
  }
  list(items, cls) {
    const ul = el('ul', cls || 'cr-list');
    for (const t of items) { const li = el('li'); li.append(...labelNodes(t, this.labels)); ul.append(li); }
    return ul;
  }

  summary(r) {
    const s = this.section('summary', 'Summary'), items = Array.isArray(r.summary) ? r.summary : [];
    s.append(items.length ? this.list(items) : el('p', 'cr-intro', 'Nothing in this range met the evidence bar for a summary.'));
    return s;
  }

  group(r, key, title, intro, none) {
    const items = (Array.isArray(r[key]) ? r[key] : []).filter(f => f && f.observation);
    const s = this.section(key, `${title} · ${items.length}`, intro);
    if (!items.length) s.append(el('p', 'cr-none', none));
    for (const f of items) s.append(this.card(f));
    return s;
  }

  card(f) {
    const kind = KINDS[f.label] || 'review', obs = this.observationOf(f.observation) || {}, card = el('article', `cr-finding k-${kind}`);
    card.dataset.observation = f.observation;
    const head = el('div', 'cr-f-head'), conf = f.confidence || (obs.confidence && obs.confidence.overall);
    head.append(el('span', 'cr-badge', f.label || 'Finding'));
    if (conf) head.append(el('span', 'cr-conf', `Confidence ${conf}`));
    card.append(head);
    const h = el('h4'); h.append(...labelNodes(f.headline || words(obs.pattern || 'finding'), this.labels)); card.append(h);
    if (f.whatHappened) card.append(this.para('What happened.', f.whatHappened));
    if (f.whyItMatters) card.append(this.para('Why it matters.', f.whyItMatters));
    if (f.whatToImprove) {
      // A strength's whatToImprove is how to build on it (language.py starts it "To build on it:"): its own lead.
      const build = kind === 'strength' && /^To build on it:\s*/.exec(f.whatToImprove), rest = build ? f.whatToImprove.slice(build[0].length) : f.whatToImprove;
      card.append(this.para(kind === 'strength' ? 'To build on it.' : 'What to improve.', build ? rest.charAt(0).toUpperCase() + rest.slice(1) : rest));
    }
    if (f.howToPractice) card.append(this.para('How to practise.', f.howToPractice));
    const drill = this.drillOf(f.drill);
    if (drill) card.append(this.para('Drill.', drillName(drill), 'cr-drill-ref'));
    if (f.video) card.append(this.para('On the video.', f.video));
    if (f.confidenceNote) card.append(this.para('Confidence.', f.confidenceNote, 'cr-intro'));
    const metrics = Array.isArray(f.metrics) ? f.metrics : [];
    if (metrics.length) {
      const box = el('div', 'cr-metrics');
      for (const m of metrics) { const [name, value] = metricText(m.name, m.value, m.unit, m.estimated === true || (m.estimated === undefined && this.estimated())), chip = el('span', 'cr-metric'); chip.append(document.createTextNode(name + ' '), el('b', null, value)); box.append(chip); }
      card.append(box);
    }
    const evidence = (f.evidence || []).map(id => this.evidenceOf(id)).filter(Boolean), actions = el('div', 'cr-actions');
    actions.append(button(evidence.length ? `Evidence (${evidence.length})` : 'Details', 'quiet cr-show', () => this.select(f.observation), 'Show the evidence of this finding'));
    for (const ev of evidence) {
      const t = button(clock(ev.time), 'cr-time', () => { this.select(f.observation, ev.id); this.jump(ev, 'tracking'); }, `Show ${clock(ev.time)} on the video (segment ${clock(ev.segmentTime)})`);
      t.dataset.evidence = ev.id;
      actions.append(t);
    }
    card.append(actions, this.feedbackBlock(f));
    this.cards.set(f.observation, card);
    return card;
  }

  drillOf(id) {
    const r = this.report();
    if (!id || !r) return null;
    for (const t of r.training || []) { const d = t && t.drill; if (d && (d.id === id || d === id)) return typeof d === 'object' ? d : {id: d}; }
    return {id};
  }

  feedbackBlock(f) {
    const box = el('div', 'cr-fb');
    box.dataset.feedback = f.observation;
    this.boxes.set(f.observation, box);
    this.fillFeedback(box, f);
    return box;
  }
  refreshFeedback(id) {
    const box = this.boxes.get(id), f = this.findingOf(id);
    if (box && f) this.fillFeedback(box, f);
  }
  fillFeedback(box, f) {
    // The verdict given on this finding in this run (the server attaches no other run's verdict to a finding).
    const id = f.observation, mine = this.feedbackFor(id).filter(e => e.thisRun).sort((a, b) => String(b.made).localeCompare(String(a.made)))[0] || null;
    const draft = this.drafts.get(id) || {verdict: mine ? mine.verdict : null, text: mine ? mine.text || '' : ''};
    box.replaceChildren(el('span', 'label', 'Your review'));
    const buttons = VERDICTS.map(([v, text]) => {
      const b = button(text, 'cr-verdict', () => {
        const d = this.drafts.get(id) || {...draft};
        d.verdict = v; this.drafts.set(id, d);
        for (const x of buttons) x.setAttribute('aria-pressed', String(x.dataset.verdict === v));
        save.disabled = false;
      });
      b.dataset.verdict = v; b.setAttribute('aria-pressed', String(draft.verdict === v));
      return b;
    });
    box.append(...buttons);
    const note = el('textarea', 'cr-note-input');
    note.value = draft.text || ''; note.maxLength = 2000; note.rows = 2;
    note.placeholder = 'Optional note: what the tracking got right or missed';
    note.setAttribute('aria-label', 'Review note');
    note.addEventListener('input', () => { const d = this.drafts.get(id) || {...draft}; d.text = note.value; this.drafts.set(id, d); save.disabled = !d.verdict; });
    const save = button(mine ? 'Update review' : 'Save review', 'primary cr-save', () => this.sendFeedback(id));
    save.disabled = !this.drafts.has(id) || !(this.drafts.get(id) || {}).verdict;
    box.append(note, save);
    if (mine) box.append(button('Remove', 'quiet cr-remove', () => this.removeFeedback(mine), 'Remove your review of this finding'));
    const lines = [];
    if (mine) lines.push(`Saved ${String(mine.made || '').replace('T', ' ')}: ${verdictName(mine.verdict)}${mine.text ? ` · “${mine.text}”` : ''}`);
    if (this.notes.has(id)) lines.push(this.notes.get(id));
    if (lines.length) box.append(el('p', 'msg', lines.join('\n')));
  }

  link(id) {
    const f = this.findingOf(id), obs = this.observationOf(id);
    const b = button('', 'cr-link', () => { this.select(id); const card = this.cards.get(id); if (card && card.scrollIntoView) card.scrollIntoView({behavior: 'smooth', block: 'center'}); });
    b.append(...labelNodes(f ? f.headline : obs ? words(obs.pattern || obs.detector || id) : id, this.labels));
    return b;
  }

  breakdown(r) {
    const s = this.section('breakdown', 'Breakdown'), b = r.breakdown || {};
    for (const [key, title] of SECTIONS) {
      const sec = b[key] || {available: false, reason: 'Not part of this report.'}, row = el('div', 'cr-part');
      row.dataset.part = key;
      row.append(el('h4', null, title));
      if (!sec.available) row.append(this.para(null, sec.reason || 'Not available for this footage.', 'cr-none'));
      else {
        const ids = Array.isArray(sec.findings) ? sec.findings : [];
        if (ids.length) { const list = el('div', 'cr-links'); for (const id of ids) list.append(this.link(id)); row.append(list); }
        else row.append(el('p', 'cr-none', 'Measured; nothing here met the evidence bar.'));
      }
      for (const n of Array.isArray(sec.notes) ? sec.notes : []) row.append(this.para(null, n, 'cr-intro'));
      s.append(row);
    }
    return s;
  }

  priorities(r) {
    const items = (Array.isArray(r.priorities) ? r.priorities : []).slice(0, 3), s = this.section('priorities', 'Priorities');
    if (!items.length) { s.append(el('p', 'cr-none', 'No priority: nothing in this range calls for a training focus.')); return s; }
    const ol = el('ol', 'cr-list');
    for (const p of items) {
      const li = el('li');
      li.append(...labelNodes(p.title || '', this.labels));
      if (p.finding) { li.append(document.createTextNode(' · ')); li.append(this.link(p.finding)); }
      const d = this.drillOf(p.drill);
      if (p.drill && d) li.append(document.createTextNode(` \u00b7 drill: ${drillName(d)}`));
      ol.append(li);
    }
    s.append(ol);
    return s;
  }

  training(r) {
    // Drills as soccer/coaching/drills.drill_ref carries them: the library record and the variation for the level.
    const items = Array.isArray(r.training) ? r.training : [], s = this.section('training', 'Training plan');
    if (!items.length) { s.append(el('p', 'cr-none', 'No drill is suggested for this range.')); return s; }
    const span = (x, unit) => x && typeof x === 'object' ? [isNum(x.min) && isNum(x.max) ? (x.min === x.max ? `${x.min}${unit}` : `${x.min}\u2013${x.max}${unit}`) : '', x.note || ''].filter(Boolean).join('. ') : typeof x === 'string' || isNum(x) ? String(x) : '';
    for (const t of items) {
      const d = t && typeof t.drill === 'object' && t.drill ? t.drill : {id: t && t.drill}, box = el('div', 'cr-drill');
      box.dataset.drill = d.id || '';
      box.append(el('h4', null, drillName(d)));
      for (const [key, lead] of [['purpose', 'Purpose.'], ['objective', 'Objective.'], ['tacticalIssue', 'Addresses.'], ['summary', null], ['description', null]])
        if (typeof d[key] === 'string' && d[key]) box.append(this.para(lead, d[key]));
      const players = span(d.players, ' players'), minutes = span(d.durationMinutes || d.duration, ' min');
      if (players) box.append(this.para('Players.', players));
      if (minutes) box.append(this.para('Duration.', minutes));
      if (typeof d.area === 'string' && d.area) box.append(this.para('Area.', d.area));
      if (typeof d.setup === 'string' && d.setup) box.append(this.para('Set-up.', d.setup));
      // drills.json's fields, in the PDF's order (soccer/coaching/pdf.py DRILL_FIELDS): equipment, how it runs, coaching
      // points, progressions, regressions, success criteria (one in the exercise, one naming the report measurement it
      // should move), safety.
      for (const [key, title] of [['equipment', 'Equipment'], ['instructions', 'How it runs'], ['coachingPoints', 'Coaching points'], ['progressions', 'Progressions'],
                                  ['regressions', 'Regressions'], ['successCriteria', 'Success looks like'], ['safety', 'Safety']]) {
        const v = d[key];
        if (Array.isArray(v) && v.length) { box.append(el('div', 'cr-sub-h', title)); box.append(this.list(v.map(String))); }
        else if (typeof v === 'string' && v) box.append(this.para(title + '.', v));
      }
      if (typeof d.variation === 'string' && d.variation) box.append(this.para(`At the ${CONTEXTS[d.level] ? CONTEXTS[d.level].toLowerCase() : 'chosen'} level.`, d.variation));
      if (typeof d.source === 'string' && d.source) box.append(this.para(null, d.source, 'cr-intro'));
      const ids = Array.isArray(t.findings) ? t.findings : [];
      if (ids.length) { const list = el('div', 'cr-links'); list.append(el('span', 'cr-intro', 'For: ')); for (const id of ids) list.append(this.link(id)); box.append(list); }
      s.append(box);
    }
    return s;
  }

  confidence(r) {
    const s = this.section('confidence', 'Confidence and limitations'), c = r.confidence || {};
    if (c.overall) s.append(this.kv('Overall confidence', String(c.overall), c.overall === 'low'));
    if (Array.isArray(c.notes) && c.notes.length) s.append(this.list(c.notes));
    // What could not be assessed and why, as the report words it (an observation measured too briefly is a
    // limitation, never a strength or a weakness); the data-quality note is in the overview, shown once.
    const lim = this.limitationLines(r);
    if (lim.length) { s.append(el('h4', 'cr-sub-h', 'Limitations')); s.append(this.list(lim)); }
    if (this.qualityLines(r).length) s.append(el('p', 'cr-intro', 'The data-quality note of this footage is in the overview.'));
    s.append(el('p', 'cr-intro', 'Metres assume a standard 105 \u00d7 68 m pitch and carry the registration\u2019s error; no possession, passes or pressing are recorded, so nothing here is tied to a phase of play.'));
    return s;
  }

  feedbackSection(r) {
    // Reviews kept with this video that sit on no finding of this report, where the PDF has its Expert feedback: this
    // run's verdicts on findings this report does not have (another range, or a finding that changed since), and
    // verdicts from earlier runs, which never sit on a finding (observation ids leave the run out and team letters are
    // arbitrary per run: the verifier saw a run-r1 "Incorrect" printed on run r2's finding). Those name players by that
    // run's ids, so their text is shown as stored, not as this run's labels.
    const all = ((this.answer && this.answer.feedback) || []).filter(f => f && typeof f === 'object');
    const teamOf = f => (f.observation && f.observation.team) || null;
    const loose = all.filter(f => f.thisRun && !f.attached && (!teamOf(f) || teamOf(f) === r.team));
    const earlier = all.filter(f => !f.thisRun);
    if (!loose.length && !earlier.length) return null;
    const s = this.section('feedback', 'Expert feedback', 'Reviews kept with this video that are not on a finding of this report. They never change the measurements.');
    const line = f => `${(f.observation && f.observation.headline) || 'A finding'} \u00b7 ${verdictName(f.verdict)}${f.text ? ` \u00b7 \u201c${f.text}\u201d` : ''}`;
    if (loose.length) { s.append(el('h4', 'cr-sub-h', 'Your reviews of this run that match no finding here')); s.append(this.list(loose.map(line))); }
    if (earlier.length) {
      s.append(el('h4', 'cr-sub-h', 'From an earlier run'));
      const ul = el('ul', 'cr-list');
      for (const f of earlier) ul.append(el('li', null, `${line(f)} (run ${f.run}${teamOf(f) ? `, Team ${teamOf(f)} as labelled in that run` : ''}; players named by that run\u2019s ids)`));
      s.append(ul);
    }
    return s;
  }

  select(observation, evidence) {
    this.selected = observation; this.selectedEvidence = evidence || null;
    for (const [id, card] of this.cards) card.classList.toggle('selected', id === observation);
    this.renderSide();
  }

  jump(ev, view) {
    const fps = isNum(ev.fps) && ev.fps > 0 ? ev.fps : this.fps();
    const seek = isNum(ev.seek) ? ev.seek : Number.isInteger(ev.frame) ? (ev.frame + .5) / fps : ev.time;
    return this.jumpTo({frame: ev.frame, seek, time: ev.time, view: view === 'board' ? 'board' : 'tracking', focus: (ev.players || []).filter(p => typeof p === 'string'),
                        diagram: ev.diagram || null, overlay: validOverlay(ev.videoOverlay) ? ev.videoOverlay : null, evidence: ev.id, observation: ev.observation});
  }

  renderSide() {
    const side = this.side, r = this.report();
    side.replaceChildren(el('h3', null, 'Evidence'));
    side.hidden = !r;
    const f = this.selected ? this.findingOf(this.selected) : null;
    if (!r || !f) { side.append(el('p', 'an-empty', r ? 'Select a finding to see its evidence: the frame, the measured shape and a diagram.' : 'The evidence of a finding appears here: its frame, the measured shape and a diagram.')); this.placeSide(); return; }
    const head = el('div', `cr-side-head k-${KINDS[f.label] || 'review'}`), h = el('div', 'cr-side-title');
    h.append(...labelNodes(f.headline || '', this.labels));
    head.append(el('span', 'cr-badge', f.label || 'Finding'), h);
    side.append(head);
    const list = (f.evidence || []).map(id => this.evidenceOf(id)).filter(Boolean);
    if (!list.length) side.append(el('p', 'an-empty', 'No evidence frame for this finding: it rests on the measured totals listed with it.'));
    for (const ev of list) side.append(this.figure(ev));
    this.placeSide();
  }

  figure(ev) {
    const fig = el('figure', 'cr-ev'), colours = this.colours(), video = (this.info && this.info.video) || {};
    fig.dataset.evidence = ev.id;
    if (ev.id === this.selectedEvidence) fig.classList.add('selected');
    const times = el('div', 'cr-times');
    const at = button(clock(ev.time), 'cr-time', () => this.jump(ev, 'tracking'), `Show this frame on the video (segment ${clock(ev.segmentTime)})`);
    at.dataset.evidence = ev.id;
    const board = button('Show on 2D', 'quiet cr-board', () => this.jump(ev, 'board'), 'Show this moment on the 2D match view');
    board.dataset.evidence = ev.id;
    times.append(at, el('span', 'cr-seg', `segment ${clock(ev.segmentTime)}`), board);
    fig.append(times);
    if (Number.isInteger(ev.frame)) {
      const shot = el('div', 'cr-shot'), img = el('img');
      if (video.width && video.height) shot.style.aspectRatio = `${video.width} / ${video.height}`;
      img.alt = `Frame ${ev.frame} at ${clock(ev.time)}`; img.loading = 'lazy'; img.decoding = 'async';
      img.onerror = () => shot.replaceChildren(el('p', 'cr-noshot', 'The screenshot could not be loaded.'));
      img.src = '/api/coaching/frame?' + new URLSearchParams({run: this.runName() || '', frame: String(ev.frame), w: '640'}).toString();
      shot.append(img);
      if (validOverlay(ev.videoOverlay)) shot.append(svgOverlay(ev.videoOverlay, colours, this.labels));
      fig.append(shot);
      const tier = (ev.videoOverlay && ev.videoOverlay.tier) || ev.tier;
      if (TIERS[tier]) fig.append(el('p', 'cr-note', TIERS[tier]));
    }
    if (ev.caption) fig.append(this.para(null, ev.caption, 'cr-caption'));
    if (ev.diagram && Array.isArray(ev.diagram.elements)) {
      if (ev.diagram.title) fig.append(this.para(null, ev.diagram.title, 'cr-dtitle'));
      fig.append(svgDiagram(ev.diagram, colours, this.labels));
      if (ev.diagram.metresEstimated !== false) fig.append(el('p', 'cr-note', 'Top-down from the observed positions at this frame; distances estimated (assumed 105 × 68 m pitch).'));
    }
    for (const [k, v] of Object.entries(ev.metrics || {})) { const [name, value] = metricText(k, v, null, this.estimated()); fig.append(this.kv(name, value)); }
    if (ev.confidence) fig.append(this.kv('Confidence', String(ev.confidence), ev.confidence === 'low'));
    const cal = ev.calibration || {};
    if (isNum(cal.confidence)) fig.append(this.kv('Registration confidence', cal.confidence.toFixed(2), cal.confidence < .7));
    if (Array.isArray(ev.players) && ev.players.length) fig.append(this.kv('Players', ev.players.join(', ')));   // ids: kv shows their labels
    return fig;
  }

  placeSide() {
    // Wide screens: the evidence column beside the report, moved down to the selected finding. Narrow screens (one
    // column): the evidence goes right under the selected finding.
    const narrow = !!(this.media && this.media.matches), card = this.selected ? this.cards.get(this.selected) : null;
    if (narrow && card && typeof card.after === 'function') {
      card.after(this.side);
      this.side.classList.add('inline'); this.side.style.marginTop = '';
      return;
    }
    if (this.home && this.side.parentElement !== this.home) this.home.append(this.side);
    this.side.classList.remove('inline');
    const top = card && isNum(card.offsetTop) && isNum(this.main.offsetTop) ? Math.max(0, card.offsetTop - this.main.offsetTop) : 0;
    this.side.style.marginTop = top ? `${Math.round(top)}px` : '';
  }
}

global.Coaching = {create: (options) => new CoachingView(options), drawBoard, drawVideo, svgDiagram, svgOverlay, labelMap, plainLabels, clock};
})(window);
