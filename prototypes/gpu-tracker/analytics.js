/* PitchIQ · Player analytics: heatmap, average positions and movement trail of the finalized tracking, drawn on
   the same pitch as the 2D match view (tactical.js draws the board; this file draws one layer over it).

   Data. Nothing is computed here: the page asks the server (/api/analytics, soccer/analytics.py) for one selection
   (player, time range, scope) and caches the answer per result; the server computes from the saved result with the
   validity rules documented there (observed samples only, trusted registration, confirmed identities of the final
   roster, time-weighted). The same normalized pitch coordinates as the board: x 0..1 left goal line to right goal
   line, y 0..1 far touchline to near touchline.

   The <video> element is the clock for the "current 5 minutes" range and for the marker that follows the trail. */
(function (global) {
'use strict';

const TABS = ['heatmap', 'average', 'trail'];
const CURRENT_WINDOW = 300;   // s, the server's five-minute block
// Observed-time ramp, readable over the green: transparent -> deep purple -> magenta -> orange -> pale yellow.
const RAMP = [[0, 0, 0, 0, 0], [.12, 59, 15, 112, .45], [.4, 182, 54, 121, .7], [.7, 251, 136, 97, .85], [1, 252, 253, 191, .95]];

function ramp(v) {
  v = Math.max(0, Math.min(1, v));
  for (let i = 1; i < RAMP.length; i++) {
    if (v <= RAMP[i][0]) {
      const [a, b] = [RAMP[i - 1], RAMP[i]], u = (v - a[0]) / (b[0] - a[0] || 1);
      return [a[1] + (b[1] - a[1]) * u, a[2] + (b[2] - a[2]) * u, a[3] + (b[3] - a[3]) * u, a[4] + (b[4] - a[4]) * u];
    }
  }
  return RAMP[RAMP.length - 1].slice(1);
}

const clock = t => `${Math.floor(t / 60).toString().padStart(2, '0')}:${Math.floor(t % 60).toString().padStart(2, '0')}`;
const seconds = v => v >= 90 ? `${(v / 60).toFixed(1)} min` : `${v.toFixed(1)} s`;

class AnalyticsView {
  constructor(options) {
    this.canvas = options.canvas;
    this.ctx = this.canvas.getContext('2d');
    this.notice = options.notice;
    this.status = options.status;
    this.legend = options.legend;
    this.insights = options.insights || null;   // the insight panel beside the pitch (rows of label / value)
    this.c = options.controls;
    this.api = options.api;              // async (url) => json
    this.video = options.video;
    this.board = options.board;          // the page's TacticalView: team colours
    this.colourInputs = options.colourInputs || [];
    this.result = null; this.info = null; this.view = null;
    this.cache = new Map(); this.data = null; this.loading = null; this.seq = 0;
    this.tab = 'heatmap'; this.layer = null; this.layerKey = null; this.markerKey = null; this.block = null;
    this.c.tabs.forEach(b => b.addEventListener('click', () => this.setTab(b.dataset.tab)));
    this.c.team.addEventListener('change', () => { this.fillPlayers(); this.request(); });
    this.c.player.addEventListener('change', () => this.request());
    this.c.range.addEventListener('change', () => { this.c.custom.hidden = this.c.range.value !== 'custom'; this.request(); });
    for (const input of [this.c.t0, this.c.t1]) input.addEventListener('change', () => this.request());
    this.c.scope.addEventListener('change', () => this.request());
    for (const input of this.colourInputs) input.addEventListener('input', () => this.redraw());
    global.addEventListener('resize', () => this.redraw());
    // The canvas is sized from its CSS width; when the layout gives it another width (the view shown, a grid column,
    // a drawer) the board module resizes and clears it, so the layer is drawn again.
    if (typeof ResizeObserver === 'function') new ResizeObserver(() => { if (this.result) setTimeout(() => this.redraw(), 0); }).observe(this.canvas);
    this.setTab('heatmap');
  }

  // ---------- selection ----------
  load(result, info) {
    this.result = result; this.info = info; this.cache.clear(); this.data = null; this.layerKey = null;
    this.view = result ? Tactical.create({canvas: this.canvas, controls: {}}) : null;
    if (this.view) this.view.load(result, info);
    this.fillPlayers();
    if (!result) { this.setNotice('Player analytics appear here after a run.'); this.setStatus(''); this.clear(); return; }
    this.request();
  }

  players() {
    const r = this.result;
    if (!r || !r.match) return [];
    const out = [];
    for (const [key, role] of [['players', 'PLAYER'], ['goalkeepers', 'GOALKEEPER']])
      for (const e of r.match[key] || []) if ((e.team === 'A' || e.team === 'B') && e.confirmed !== false) out.push({id: e.id, display: e.display || e.id, team: e.team, role});
    out.sort((a, b) => a.team.localeCompare(b.team) || (a.role !== b.role ? (a.role === 'PLAYER' ? -1 : 1) : a.display.localeCompare(b.display)));
    return out;
  }

  fillPlayers() {
    const keep = this.c.player.value, team = this.c.team.value;
    const list = this.players().filter(p => p.team === team);
    this.c.player.replaceChildren(...list.map(p => { const o = document.createElement('option'); o.value = p.id; o.textContent = p.display === p.id ? p.id : `${p.display} · ${p.id}`; return o; }));
    if (list.some(p => p.id === keep)) this.c.player.value = keep;
    this.c.player.disabled = !list.length;
  }

  select(id) {
    // The page selected an identity (roster click): follow it when it is analysable.
    const p = this.players().find(q => q.id === id);
    if (!p) return;
    if (this.c.team.value !== p.team) { this.c.team.value = p.team; this.fillPlayers(); }
    this.c.player.value = id;
    this.request();
  }

  setTab(tab) {
    const changed = this.tab !== tab;
    this.tab = TABS.includes(tab) ? tab : 'heatmap';
    this.c.tabs.forEach(b => b.classList.toggle('active', b.dataset.tab === this.tab));
    this.c.scopeWrap.hidden = this.tab !== 'average';
    this.layerKey = null;
    if (changed) { this.canvas.style.animation = 'none'; void this.canvas.offsetWidth; this.canvas.style.animation = ''; }
    this.redraw();
  }

  range() {
    const v = this.c.range.value;
    if (v === 'current') {
      const k = Math.floor(Math.max(0, this.video.currentTime) / CURRENT_WINDOW);
      return {range: 'current', t0: k * CURRENT_WINDOW, t1: (k + 1) * CURRENT_WINDOW};
    }
    if (v === 'custom') {
      const a = Number(this.c.t0.value), b = Number(this.c.t1.value);
      if (Number.isFinite(a) && Number.isFinite(b)) return {range: 'custom', t0: Math.min(a, b), t1: Math.max(a, b)};
    }
    return {range: 'full', t0: null, t1: null};
  }

  key() {
    const r = this.range();
    return `${this.c.player.value}|${r.range}|${r.t0}|${r.t1}|${this.c.scope.value}`;
  }

  async request() {
    if (!this.result) return;
    const player = this.c.player.value;
    if (!player) { this.data = null; this.setNotice('No confirmed player of this team has pitch positions.'); this.setStatus(''); this.clear(); return; }
    const key = this.key();
    if (this.cache.has(key)) { this.data = this.cache.get(key); this.layerKey = null; this.redraw(); return; }
    const r = this.range(), seq = ++this.seq;
    const params = new URLSearchParams({player, scope: this.c.scope.value});
    if (r.t0 !== null) { params.set('t0', r.t0); params.set('t1', r.t1); }
    this.setStatus('Computing…');
    try {
      const data = await this.api('/api/analytics?' + params.toString());
      if (seq !== this.seq) return;   // a newer selection superseded this one
      this.cache.set(key, data);
      this.data = data; this.layerKey = null;
      this.redraw();
    } catch (error) {
      if (seq !== this.seq) return;
      this.data = null; this.setNotice(error.message); this.setStatus(''); this.clear();
    }
  }

  // ---------- drawing ----------
  setNotice(text) { if (this.notice.textContent !== text) { this.notice.textContent = text; this.notice.hidden = !text; } }
  setStatus(text) { if (this.status.textContent !== text) this.status.textContent = text; }
  clear() { if (this.view) { this.view.resize(); const g = this.view.geometry(); this.ctx.clearRect(0, 0, g.width, g.height); this.ctx.drawImage(this.view.drawPitch(g), 0, 0); } else this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height); this.legend.replaceChildren(); }
  colours() { return this.board ? this.board.colours : {A: '#b98cff', B: '#f6b564'}; }

  redraw() {
    if (!this.view) return;
    this.view.resize();
    const g = this.view.geometry(), colours = this.colours(), d = this.data;
    const key = `${this.tab}|${this.canvas.width}|${colours.A}|${colours.B}|${d ? this.key() : 'none'}`;
    if (key !== this.layerKey) {
      this.layerKey = key;
      this.layer = document.createElement('canvas'); this.layer.width = g.width; this.layer.height = g.height;
      const c = this.layer.getContext('2d');
      if (!this.view.pitchLayer || this.view.pitchLayer.width !== g.width) this.view.pitchLayer = this.view.drawPitch(g);
      c.drawImage(this.view.pitchLayer, 0, 0);
      this.legend.replaceChildren();
      if (d && d.player) {
        if (this.tab === 'heatmap') this.drawHeatmap(c, g, d);
        else if (this.tab === 'average') this.drawAverages(c, g, d);
        else this.drawTrail(c, g, d);
      }
      this.describe(d);
      this.markerKey = null;
    }
    this.ctx.clearRect(0, 0, g.width, g.height);
    this.ctx.drawImage(this.layer, 0, 0);
    if (this.tab === 'trail') this.drawMarker(g);
  }

  describe(d) {
    const panel = this.insights;
    const empty = text => { if (panel) { panel.replaceChildren(); const p = document.createElement('p'); p.className = 'an-empty'; p.textContent = text; panel.append(p); } };
    if (!d) { this.setNotice(this.loading ? '' : 'Choose a player.'); empty('Choose a player to see their observed coverage, duration and samples.'); this.setStatus(''); return; }
    if (!d.player) { this.setNotice('This identity is not analysable (not a confirmed team player or goalkeeper).'); empty('This identity is not a confirmed team player or goalkeeper.'); this.setStatus(''); return; }
    const cov = d.coverage, r = d.range, ex = d.excluded || {};
    const dropped = (ex.lowPitchConfidence || 0) + (ex.offPitch || 0) + (ex.clipped || 0);
    if (panel) {
      panel.replaceChildren();
      const who = document.createElement('div'); who.className = 'who'; who.textContent = d.player.display;
      if (d.player.display !== d.player.id) { const small = document.createElement('small'); small.textContent = d.player.id; who.append(small); }
      const team = document.createElement('div'); team.className = 'team';
      const sw = document.createElement('i'); sw.style.background = this.colours()[d.player.team] || '#8d99a6';
      team.append(sw, document.createTextNode(`Team ${d.player.team} · ${d.player.role === 'GOALKEEPER' ? 'goalkeeper' : 'outfield player'}`));
      panel.append(who, team);
      const rows = [['Selected interval', `${clock(r.t0)} – ${clock(r.t1)}`], ['Observed duration', seconds(cov.observedSeconds)],
                    ['Observed coverage', `${Math.round(cov.coverage * 100)}%`, cov.low], ['Position samples', cov.samples.toLocaleString()]];
      if (this.tab === 'heatmap' && d.heatmap) rows.push(['Peak occupancy', `${d.heatmap.max.toFixed(1)} s per ${d.heatmap.cell} m cell`], ['Smoothing', `${d.heatmap.sigma} m`]);
      if (this.tab === 'average' && d.average && d.average.position) rows.push(['Average position', `X ${d.average.position[0].toFixed(2)} · Y ${d.average.position[1].toFixed(2)}`], ['Players shown', String((d.averages || []).length)]);
      if (this.tab === 'trail' && d.trail) rows.push(['Observed segments', String(d.trail.segments.length)], ['Distance observed', `${d.trail.metres} m`], ['Gaps (not drawn)', `${d.trail.breaks.length} · ${seconds(d.trail.unseenSeconds)} unseen`]);
      if (dropped) rows.push(['Samples excluded', dropped.toLocaleString()]);
      for (const [label, value, low] of rows) {
        const kv = document.createElement('div'); kv.className = 'kv';
        const l = document.createElement('span'); l.textContent = label;
        const v = document.createElement('b'); v.textContent = value; if (low) v.className = 'low';
        kv.append(l, v); panel.append(kv);
      }
    }
    const notes = [];
    if (!cov.samples) notes.push('Not enough observed positions for this player in the selected range.');
    else if (cov.low) notes.push(`Low coverage: this ${this.tab === 'heatmap' ? 'heatmap' : this.tab === 'average' ? 'average' : 'trail'} describes only the observed ${Math.round(cov.coverage * 100)}% of the interval, not where the player was in between.`);
    if (this.tab === 'average' && d.average && d.average.position) notes.push('An average is a summary of the observed positions, not a place the player stood.');
    if (dropped) notes.push('Excluded samples had an untrusted registration, lay off the pitch or had the feet unseen.');
    this.setStatus(notes.join(' '));
    this.setNotice(!cov.samples ? 'Not enough observed positions for this player.' : '');
  }

  drawHeatmap(c, g, d) {
    const hm = d.heatmap;
    if (!hm || !hm.max) return;
    const img = document.createElement('canvas'); img.width = hm.cols; img.height = hm.rows;
    const ic = img.getContext('2d'), px = ic.createImageData(hm.cols, hm.rows);
    for (let j = 0; j < hm.rows; j++) for (let i = 0; i < hm.cols; i++) {
      const [r, gg, b, a] = ramp(Math.sqrt(hm.grid[j][i] / hm.max));   // square root: the quieter areas stay visible
      const k = 4 * (j * hm.cols + i);
      px.data[k] = r; px.data[k + 1] = gg; px.data[k + 2] = b; px.data[k + 3] = Math.round(a * 255);
    }
    ic.putImageData(px, 0, 0);
    const [x0, y0] = g.metres(0, 0), L = d.pitch.length * g.scale, W = d.pitch.width * g.scale;
    c.save(); c.imageSmoothingEnabled = true; c.imageSmoothingQuality = 'high';
    c.drawImage(img, 0, 0, hm.cols, hm.rows, x0, y0, L, W);
    c.restore();
    // Legend: lower to higher observed time per cell.
    const bar = document.createElement('div'); bar.className = 'heatbar';
    const stops = RAMP.map(s => `rgba(${Math.round(s[1])},${Math.round(s[2])},${Math.round(s[3])},${Math.max(.15, s[4])}) ${Math.round(s[0] * 100)}%`).join(',');
    bar.style.background = `linear-gradient(90deg, ${stops})`;
    const lo = document.createElement('span'); lo.textContent = 'less observed time';
    const hi = document.createElement('span'); hi.textContent = `more (${hm.max.toFixed(1)} s per ${hm.cell} m cell, smoothed ${hm.sigma} m)`;
    this.legend.append(lo, bar, hi);
  }

  circle(c, g, a, colour, options = {}) {
    const [x, y] = g.toPx(a.position[0], a.position[1]), r = g.playerRadius * (options.selected ? 1.15 : 1);
    c.save();
    c.lineWidth = Math.max(1, r * .16); c.fillStyle = a.low ? colour + '99' : colour; c.strokeStyle = '#0d1a24b8';
    if (a.low) c.setLineDash([r * .5, r * .4]);
    c.beginPath(); c.arc(x, y, r, 0, Math.PI * 2); c.fill(); c.stroke(); c.setLineDash([]);
    if (options.selected) { c.strokeStyle = '#ffffff'; c.beginPath(); c.arc(x, y, r * 1.5, 0, Math.PI * 2); c.stroke(); }
    if (a.role === 'GOALKEEPER') { c.font = `bold ${Math.round(r * .95)}px system-ui, sans-serif`; c.textAlign = 'center'; c.textBaseline = 'middle'; c.fillStyle = '#ffffff'; c.fillText('GK', x, y + r * .05); }
    c.font = `${Math.max(9, Math.round(r * .95))}px system-ui, sans-serif`; c.textAlign = 'center'; c.textBaseline = 'alphabetic';
    const label = a.display + (a.low ? ` · ${Math.round(a.coverage * 100)}%` : '');
    const ly = options.below ? y + r * 2.2 : y - r - 3;
    c.lineWidth = 3; c.strokeStyle = '#07131bcc'; c.strokeText(label, x, ly); c.fillStyle = '#f4f7f4'; c.fillText(label, x, ly);
    c.restore();
  }

  drawAverages(c, g, d) {
    const colours = this.colours(), list = (d.averages || []).filter(a => a.position);
    if (!list.length) return;
    // Labels alternate above/below when two markers would overlap.
    const placed = [];
    for (const a of list) {
      const [x, y] = g.toPx(a.position[0], a.position[1]);
      const near = placed.some(([px, py]) => Math.abs(px - x) < g.playerRadius * 4 && Math.abs(py - y) < g.playerRadius * 3);
      placed.push([x, y]);
      this.circle(c, g, a, colours[a.team] || '#8d99a6', {selected: a.id === d.player.id && list.length > 1, below: near});
    }
    if (list.length === 1 && list[0].id === d.player.id) {
      const lines = [`Average pitch position of ${d.player.display}: X = ${list[0].position[0].toFixed(2)}, Y = ${list[0].position[1].toFixed(2)}`,
                     'A mathematical summary of the observed positions, not a place the player stood.'];
      c.font = `${Math.max(10, g.scale * 1.6)}px system-ui, sans-serif`; c.textAlign = 'left'; c.fillStyle = '#e8edf0';
      lines.forEach((l, i) => c.fillText(l, g.ox, g.oy * .55 + i * g.scale * 1.9));
    }
  }

  drawTrail(c, g, d) {
    const tr = d.trail;
    if (!tr || !tr.segments.length) return;
    const colour = this.colours()[d.player.team] || '#8d99a6', r = g.playerRadius, width = Math.max(2, r * .45);
    c.save(); c.lineJoin = 'round'; c.lineCap = 'round';
    for (const s of tr.segments) {
      if (s.x.length < 2) { const [x, y] = g.toPx(s.x[0], s.y[0]); c.fillStyle = colour; c.beginPath(); c.arc(x, y, width, 0, Math.PI * 2); c.fill(); continue; }
      c.beginPath();
      s.x.forEach((sx, i) => { const [x, y] = g.toPx(sx, s.y[i]); if (i) c.lineTo(x, y); else c.moveTo(x, y); });
      c.lineWidth = width + 3; c.strokeStyle = '#0b1520aa'; c.stroke();
      c.lineWidth = width; c.strokeStyle = colour; c.stroke();
      // Direction ticks every ~8 m along the segment.
      let since = 0;
      for (let i = 1; i < s.x.length; i++) {
        const [ax, ay] = g.toPx(s.x[i - 1], s.y[i - 1]), [bx, by] = g.toPx(s.x[i], s.y[i]);
        since += Math.hypot(bx - ax, by - ay);
        if (since < 8 * g.scale) continue;
        since = 0;
        const angle = Math.atan2(by - ay, bx - ax), h = width * 1.6;
        c.fillStyle = '#ffffffdd'; c.beginPath();
        c.moveTo(bx, by); c.lineTo(bx - h * Math.cos(angle - .5), by - h * Math.sin(angle - .5)); c.lineTo(bx - h * Math.cos(angle + .5), by - h * Math.sin(angle + .5)); c.closePath(); c.fill();
      }
      // The ends of a segment next to a gap: a small hollow dot where the player was last and next seen.
      for (const k of [0, s.x.length - 1]) {
        const [x, y] = g.toPx(s.x[k], s.y[k]);
        c.lineWidth = Math.max(1, width * .5); c.strokeStyle = '#ffffffbb'; c.fillStyle = '#0b1520';
        c.beginPath(); c.arc(x, y, width * .9, 0, Math.PI * 2); c.fill(); c.stroke();
      }
    }
    const mark = (p, text, fill) => {
      const [x, y] = g.toPx(p.x, p.y);
      c.lineWidth = Math.max(1.5, r * .18); c.fillStyle = fill; c.strokeStyle = '#ffffff';
      c.beginPath(); c.arc(x, y, r * .95, 0, Math.PI * 2); c.fill(); c.stroke();
      c.font = `bold ${Math.round(r * .9)}px system-ui, sans-serif`; c.textAlign = 'center'; c.textBaseline = 'middle'; c.fillStyle = '#ffffff'; c.fillText(text, x, y + r * .05);
      c.font = `${Math.max(9, Math.round(r * .85))}px system-ui, sans-serif`; c.textBaseline = 'alphabetic';
      const label = `${text === 'S' ? 'START' : 'END'} ${clock(p.t)}`;
      c.lineWidth = 3; c.strokeStyle = '#07131bcc'; c.strokeText(label, x, y - r - 4); c.fillStyle = '#f4f7f4'; c.fillText(label, x, y - r - 4);
    };
    mark(tr.start, 'S', '#2f9e5b'); mark(tr.end, 'E', '#c04a4a');
    c.restore();
    const note = document.createElement('span'); note.className = 'muted';
    note.textContent = `${d.player.display}'s observed path: solid where the player was seen on consecutive frames; nothing is drawn across the ${tr.breaks.length} gap${tr.breaks.length === 1 ? '' : 's'} (no route is invented). Arrowheads show the direction; the white dot follows the video.`;
    this.legend.append(note);
  }

  // The player's observed position at the video time, on the trail (nothing when there is no observation then).
  markerAt(t) {
    const tr = this.data && this.data.trail;
    if (!tr) return null;
    const tol = (this.info && this.info.video && this.info.video.fps ? 1 / this.info.video.fps : .04) * 1.6;
    for (const s of tr.segments) {
      const ts = s.t;
      if (!ts.length || t < ts[0] - tol || t > ts[ts.length - 1] + tol) continue;
      let lo = 0, hi = ts.length - 1;
      while (lo < hi) { const mid = (lo + hi + 1) >> 1; if (ts[mid] <= t) lo = mid; else hi = mid - 1; }
      const k = Math.abs(ts[lo] - t) <= tol ? lo : lo + 1 < ts.length && Math.abs(ts[lo + 1] - t) <= tol ? lo + 1 : -1;
      if (k >= 0) return {x: s.x[k], y: s.y[k]};
    }
    return null;
  }

  drawMarker(g) {
    const m = this.markerAt(this.video.currentTime);
    if (!m) return;
    const [x, y] = g.toPx(m.x, m.y), r = g.playerRadius * .75;
    const c = this.ctx;
    c.save(); c.lineWidth = Math.max(1.5, r * .3); c.fillStyle = '#ffffff'; c.strokeStyle = '#0b1520';
    c.beginPath(); c.arc(x, y, r, 0, Math.PI * 2); c.fill(); c.stroke(); c.restore();
  }

  // Every animation frame from the page: the current-time marker and the "current 5 minutes" block.
  tick(t) {
    if (!this.result || !this.data) return;
    if (this.c.range.value === 'current') {
      const k = Math.floor(Math.max(0, t) / CURRENT_WINDOW);
      if (k !== this.block) { this.block = k; this.request(); return; }
    }
    if (this.tab !== 'trail' || !this.layer) return;
    const m = this.markerAt(t), key = m ? `${m.x}|${m.y}` : 'none';
    if (key === this.markerKey) return;
    this.markerKey = key;
    const g = this.view.geometry();
    this.ctx.clearRect(0, 0, g.width, g.height);
    this.ctx.drawImage(this.layer, 0, 0);
    this.drawMarker(g);
  }
}

global.Analytics = {create: (options) => new AnalyticsView(options), ramp};
})(window);
