/* PitchIQ · Team shape: team width and depth, defensive line height and formation recognition, drawn on the same
   Analytics pitch as the player analytics (tactical.js draws the board; this file draws one layer over it).

   Data. Nothing is computed here. The page asks the server for one team, range and position mode
   (/api/team-shape, soccer/team_shape.py) and for the measured positions at one sample (/api/team-shape/snapshot),
   and caches both per result. The report says how much of the team's observed time rests on a MEDIUM (not HIGH)
   registration; the status line says so when that is most of it. Coordinates are the result's normalized pitch
   coordinates (x along the goal-to-goal axis, y across), so a measurement drawn "along y" is the team's width
   whatever way the board is shown: everything is placed with the board's own toPx(x, y), never with screen
   assumptions.

   The snapshot slider picks a sample of the analysed data; it is not a second player and never moves the video.
   The canvas, notice, legend, insight panel and status line are shared with the player analytics: this view writes
   to them only while it is the active category. */
(function (global) {
'use strict';

const TABS = ['width', 'line', 'formation'];
const FORMATION_COLOURS = {'4-4-2': '#5aa9e6', '4-3-3': '#e6a35a', '4-2-3-1': '#b48ce6', '4-1-4-1': '#e6d35a', '3-5-2': '#5ae6b4',
                           '3-4-3': '#e65a7d', '5-3-2': '#5ad1e6', '5-4-1': '#9ad15a', '4-4-1': '#5aa9e6', '4-3-2': '#e6a35a', '5-3-1': '#5ad1e6'};
const PERIOD_TEXT = {unclear: 'No clear shape', transition: 'Transition', insufficient: 'Not enough players observed',
                     direction: 'Direction uncertain', lineup: 'Lineup change / not supported'};
const QUALITY_TEXT = {coherent: 'Coherent', uneven: 'Uneven (one defender stepped up)', stretched: 'Stretched', disrupted: 'Disrupted'};
const STATUS_TEXT = {complete: 'Complete', reliable: 'Reliable', estimated: 'Mostly estimated', partial: 'Partial (observed spread)',
                     excess: 'More players than the lineup', unavailable: 'Unavailable', displaced: 'Unit displaced (set piece or attack)'};
const SOURCE_TEXT = {formation: 'from the formation roles', 'role memory': 'from the last known roles', 'depth rank': 'deepest players (no formation)'};
const EMPTY = 'Team shape appears here after a run.';

const clock = t => `${Math.floor(t / 60).toString().padStart(2, '0')}:${Math.floor(t % 60).toString().padStart(2, '0')}`;
const clockTenth = t => `${clock(t)}.${Math.floor((t * 10) % 10)}`;
const pct = v => `${Math.round(v * 100)}%`;
const num = v => (v === null || v === undefined ? '–' : v.toFixed(1));
const el = (tag, cls, text) => { const e = document.createElement(tag); if (cls) e.className = cls; if (text !== undefined) e.textContent = text; return e; };

class TeamShapeView {
  constructor(options) {
    this.canvas = options.canvas;
    this.ctx = this.canvas.getContext('2d');
    this.notice = options.notice;
    this.status = options.status;
    this.legend = options.legend;
    this.insights = options.insights;
    this.c = options.controls;
    this.api = options.api;
    this.post = options.post;
    this.video = options.video;
    this.board = options.board;
    this.under = options.under;
    this.chart = options.chart;
    this.timeline = options.timeline;
    this.snap = options.snap;
    this.snapTime = options.snapTime;
    this.colourInputs = options.colourInputs || [];
    this.active = false; this.result = null; this.info = null; this.view = null;
    this.cache = new Map(); this.snaps = new Map(); this.data = null; this.other = null; this.shot = null;
    this.tab = 'width'; this.seq = 0; this.snapSeq = 0; this.snapT = null; this.period = null; this.formationView = 'average';
    this.layer = null; this.layerKey = null; this.snapTimer = null;
    this.gen = 0;              // bumped by every load and save: responses of an older generation are not cached
    this.drafts = new Map();   // unsaved choices in the panel's forms, kept while the panel is rebuilt
    this.message = EMPTY;      // what the empty state says (no result, a failed request, an invalid range)
    this.c.tabs.forEach(b => { b.setAttribute('role', 'tab'); b.addEventListener('click', () => this.setTab(b.dataset.tsTab)); });
    for (const input of [this.c.team, this.c.mode]) input.addEventListener('change', () => { this.period = null; this.request(); });
    this.c.range.addEventListener('change', () => { this.c.custom.hidden = this.c.range.value !== 'custom'; this.period = null; this.request(); });
    for (const input of [this.c.t0, this.c.t1]) input.addEventListener('change', () => { this.period = null; this.request(); });
    this.snap.addEventListener('input', () => this.setSnap(Number(this.snap.value)));
    this.c.formationView.addEventListener('change', () => { this.formationView = this.c.formationView.value; this.layerKey = null; this.redraw(); });
    for (const input of this.colourInputs) input.addEventListener('input', () => { this.layerKey = null; this.redraw(); });
    global.addEventListener('resize', () => { this.layerKey = null; this.redraw(); });
    if (typeof ResizeObserver === 'function') {
      new ResizeObserver(() => { if (this.active && this.result) setTimeout(() => { this.layerKey = null; this.redraw(); }, 0); }).observe(this.canvas);
      new ResizeObserver(() => { if (this.active) this.drawChart(); }).observe(this.chart);
    }
    this.chart.addEventListener('click', e => this.chartClick(e));
    this.setTab('width', true);
  }

  // ---------- state ----------
  load(result, info) {
    this.result = result; this.info = info; this.cache.clear(); this.snaps.clear(); this.data = null; this.other = null; this.shot = null;
    this.layerKey = null; this.period = null; this.snapT = null; this.gen++; this.drafts.clear(); this.message = EMPTY;
    ++this.seq; ++this.snapSeq; clearTimeout(this.snapTimer);   // responses for the previous result are dropped
    this.view = result ? Tactical.create({canvas: this.canvas, controls: {}}) : null;
    if (this.view) this.view.load(result, info);
    if (!this.active) return;
    if (result) this.request(); else this.redraw();
  }

  setActive(on) {
    this.active = !!on;
    this.under.hidden = !this.active;
    if (!this.active) return;
    this.layerKey = null;
    // A report from before the last save (or a new result) is not shown again: ask for it afresh.
    if (this.result && (!this.data || this.dataGen !== this.gen)) this.request(); else this.redraw();
  }

  setTab(tab, quiet) {
    this.tab = TABS.includes(tab) ? tab : 'width';
    this.c.tabs.forEach(b => { b.classList.toggle('active', b.dataset.tsTab === this.tab); b.setAttribute('aria-selected', String(b.dataset.tsTab === this.tab)); });
    this.c.modeWrap.hidden = this.tab === 'formation';   // formations are recognised from observed positions only
    this.c.formationViewWrap.hidden = this.tab !== 'formation';
    this.layerKey = null;
    if (quiet) return;
    if (this.tab === 'formation' && this.c.mode.value !== 'observed') { this.c.mode.value = 'observed'; this.request(); return; }
    if (this.tab === 'formation' && this.data) {
      const k = this.snapT === null ? null : this.periodAt(this.snapT);
      if (k !== null) this.period = k; else this.periodShown();
    }
    this.redraw();
  }

  range() {
    if (this.c.range.value !== 'custom') return {t0: null, t1: null, custom: false};
    const a = this.c.t0.value.trim(), b = this.c.t1.value.trim();
    const t0 = Number(a), t1 = Number(b);
    if (a === '' || b === '' || !Number.isFinite(t0) || !Number.isFinite(t1)) return {invalid: true, custom: true};
    return {t0: Math.min(t0, t1), t1: Math.max(t0, t1), custom: true, reversed: t1 < t0};
  }

  key(team) { const r = this.range(); return `${team}|${this.c.mode.value}|${r.t0}|${r.t1}`; }
  colours() { return this.board ? this.board.colours : {A: '#b98cff', B: '#f6b564'}; }
  colour() { return this.colours()[this.data ? this.data.team : this.c.team.value] || '#8d99a6'; }

  async fetchReport(team) {
    const key = this.key(team), gen = this.gen;
    if (this.cache.has(key)) return this.cache.get(key);
    const r = this.range(), params = new URLSearchParams({team, mode: this.c.mode.value});
    if (r.t0 !== null) { params.set('t0', r.t0); params.set('t1', r.t1); }
    const data = await this.api('/api/team-shape?' + params.toString());
    if (gen === this.gen) this.cache.set(key, data);   // not when a save or a new result made it stale meanwhile
    return data;
  }

  empty(text) {
    // Nothing to show (no result, an invalid range, a failed request): the pitch alone and the reason, everywhere.
    this.message = text || EMPTY; this.data = null; this.other = null; this.shot = null; this.layerKey = null;
    this.setStatus('');
    this.redraw();
  }

  async request() {
    if (!this.result || !this.active) return;
    const r = this.range();
    if (r.invalid) { ++this.seq; this.empty('Enter both times of the custom range, in seconds.'); return; }
    const seq = ++this.seq, team = this.c.team.value;
    this.setStatus('Computing team shape…');
    try {
      const data = await this.fetchReport(team);
      if (seq !== this.seq) return;
      if (data.range.note === 'outside') {
        const [s0, s1] = data.range.segment;
        this.empty(`The range ${clock(r.t0)}–${clock(r.t1)} lies outside the analysed segment (${clock(s0)}–${clock(s1)}).`);
        return;
      }
      this.data = data; this.message = ''; this.dataGen = this.gen;
      if (this.shot && (this.shot.team !== data.team || this.shot.mode !== data.mode)) this.shot = null;   // never another team's snapshot
      this.other = null;
      this.fetchReport(team === 'A' ? 'B' : 'A').then(o => { if (seq === this.seq && this.data === data) { this.other = o; this.describe(); } }).catch(() => {});
      const lo = data.range.t0, hi = data.range.t1;
      this.snap.min = lo; this.snap.max = hi; this.snap.step = 1 / (data.rules.sampleRate || 5); this.snap.disabled = false;
      let t = this.snapT;
      if (t === null || t < lo || t > hi) {
        const v = this.video ? this.video.currentTime : lo;
        t = v >= lo && v <= hi ? v : (lo + hi) / 2;
      }
      if (this.tab === 'formation' && this.period === null) this.period = this.bestPeriodIndex(t);
      this.layerKey = null;
      await this.setSnap(t, true);
    } catch (error) {
      if (seq !== this.seq) return;
      this.empty(error.message);
      this.setStatus('');
    }
  }

  async setSnap(t, immediate) {
    if (!this.data) return;
    const d = this.data;
    this.snapT = Math.min(d.range.t1, Math.max(d.range.t0, t));
    this.snap.value = this.snapT;
    this.snapTime.textContent = clockTenth(this.snapT);
    if (this.tab === 'formation') {
      // The selected period follows the snapshot: its panel and correction form are about the time shown.
      const periods = d.formation.periods, cur = this.period !== null ? periods[this.period] : null;
      const [a, b] = cur ? (cur.clipped || [cur.t0, cur.t1]) : [Infinity, -Infinity];
      if (!(a <= this.snapT && this.snapT <= b)) { const k = this.periodAt(this.snapT); if (k !== null) { this.period = k; this.layerKey = null; } }
    }
    this.drawChart();
    const run = async () => {
      // The team and mode of the report shown, not of the controls (a newer request may be pending).
      const team = d.team, mode = d.mode, q = Math.round(this.snapT * 5) / 5, gen = this.gen;
      const key = `${team}|${mode}|${q}`, seq = ++this.snapSeq;
      try {
        let shot = this.snaps.get(key);
        if (!shot) { shot = await this.api(`/api/team-shape/snapshot?${new URLSearchParams({team, mode, t: q})}`); if (gen === this.gen) this.snaps.set(key, shot); }
        if (seq !== this.snapSeq || this.data !== d) return;
        this.shot = shot; this.layerKey = null; this.redraw();
      } catch (error) { if (seq === this.snapSeq) this.setStatus(error.message); }
    };
    clearTimeout(this.snapTimer);
    if (immediate) await run(); else this.snapTimer = setTimeout(run, 70);
  }

  periodAt(t) {
    const periods = (this.data && this.data.formation.periods) || [];
    const k = periods.findIndex(p => { const [a, b] = p.clipped || [p.t0, p.t1]; return a <= t && t <= b; });
    return k < 0 ? null : k;
  }

  bestPeriodIndex(t) {
    const periods = (this.data && this.data.formation.periods) || [];
    let k = periods.findIndex(p => p.t0 <= t && t <= p.t1 && FORMATION_COLOURS[p.label]);
    if (k < 0) k = periods.findIndex(p => FORMATION_COLOURS[p.label]);
    return k < 0 ? (periods.length ? 0 : null) : k;
  }

  // ---------- drawing ----------
  setNotice(text) { if (!this.active) return; if (this.notice.textContent !== text) { this.notice.textContent = text; this.notice.hidden = !text; } }
  setStatus(text) { if (!this.active) return; if (this.status.textContent !== text) this.status.textContent = text; }

  redraw() {
    if (!this.active) return;
    if (!this.view) {
      // No result: an empty board, the reason, and no chart or history.
      this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
      this.legend.replaceChildren();
      this.layerKey = null;
      this.snap.disabled = true; this.snapTime.textContent = '–';
      this.describe(); this.setNotice(this.message || EMPTY); this.setStatus('');
      this.drawChart(); this.drawTimeline();
      return;
    }
    this.view.resize();
    const g = this.view.geometry(), colours = this.colours();
    const key = `${this.tab}|${this.canvas.width}|${colours.A}|${colours.B}|${this.data ? `${this.data.team}|${this.data.mode}|${this.data.range.t0}|${this.data.range.t1}` : '-'}|${this.shot ? `${this.shot.team}|${this.shot.t}` : '-'}|${this.period}|${this.formationView}|${this.message}`;
    if (key !== this.layerKey) {
      this.layerKey = key;
      const layer = document.createElement('canvas'); layer.width = g.width; layer.height = g.height;
      const c = layer.getContext('2d');
      if (!this.view.pitchLayer || this.view.pitchLayer.width !== g.width) this.view.pitchLayer = this.view.drawPitch(g);
      c.drawImage(this.view.pitchLayer, 0, 0);
      this.legend.replaceChildren();
      if (this.data && this.shot) {
        if (this.tab === 'width') this.drawWidthDepth(c, g);
        else if (this.tab === 'line') this.drawLine(c, g);
        else this.drawFormation(c, g);
      } else if (!this.data) this.setNotice(this.message);
      this.layer = layer;
      this.describe();
    }
    this.snap.disabled = !this.data;
    if (!this.data) this.snapTime.textContent = '–';
    this.ctx.clearRect(0, 0, g.width, g.height);
    this.ctx.drawImage(this.layer, 0, 0);
    this.drawChart();
    this.drawTimeline();
  }

  metresText(v, estimated) { return v === null || v === undefined ? '–' : `${v.toFixed(1)} m${estimated ? ' (est.)' : ''}`; }
  unit(estimated) { return estimated ? ' m (est.)' : ' m'; }

  dot(c, g, x, y, colour, options = {}) {
    const [px, py] = g.toPx(x, y), r = g.playerRadius * (options.scale || 1);
    c.save();
    c.globalAlpha = options.alpha === undefined ? 1 : options.alpha;
    c.lineWidth = Math.max(1, r * .16);
    c.fillStyle = options.estimated ? colour + '77' : colour; c.strokeStyle = options.estimated ? colour : '#0d1a24b8';
    if (options.estimated) c.setLineDash([r * .5, r * .4]);
    c.beginPath(); c.arc(px, py, r, 0, Math.PI * 2); c.fill(); c.stroke(); c.setLineDash([]);
    if (options.ring) { c.strokeStyle = '#ffffff'; c.lineWidth = Math.max(1.5, r * .2); c.beginPath(); c.arc(px, py, r * 1.45, 0, Math.PI * 2); c.stroke(); }
    if (options.gk) { c.font = `bold ${Math.round(r * .95)}px system-ui, sans-serif`; c.textAlign = 'center'; c.textBaseline = 'middle'; c.fillStyle = '#fff'; c.fillText('GK', px, py + r * .05); }
    if (options.label) {
      c.font = `${Math.max(9, Math.round(r * .9))}px system-ui, sans-serif`; c.textAlign = 'center'; c.textBaseline = 'alphabetic';
      c.lineWidth = 3; c.strokeStyle = '#07131bcc'; c.strokeText(options.label, px, py - r - 3); c.fillStyle = '#f4f7f4'; c.fillText(options.label, px, py - r - 3);
    }
    c.restore();
  }

  drawPlayers(c, g, options = {}) {
    const s = this.shot, colour = this.colour();
    for (const o of s.opponents || []) this.dot(c, g, o.x, o.y, '#9aa7b2', {scale: .62, alpha: .45});
    if (s.goalkeeper) this.dot(c, g, s.goalkeeper.x, s.goalkeeper.y, colour, {gk: true, alpha: .85});
    for (const p of s.players) {
      const ring = options.ring ? options.ring.has(p.id) : false;
      const label = options.labels ? options.labels(p) : p.display;
      this.dot(c, g, p.x, p.y, colour, {ring, estimated: p.kind === 'E', label, alpha: options.faded && !ring ? .55 : 1});
    }
  }

  measure(c, g, a, b, text, options = {}) {
    // A crisp measurement indicator between two pitch points, with end ticks and a label at its middle.
    const [ax, ay] = g.toPx(a[0], a[1]), [bx, by] = g.toPx(b[0], b[1]);
    const colour = options.colour || '#ffffff', w = Math.max(1.5, g.scale * .22);
    c.save();
    c.strokeStyle = colour; c.lineWidth = w; if (options.dashed) c.setLineDash([g.scale * .9, g.scale * .6]);
    c.beginPath(); c.moveTo(ax, ay); c.lineTo(bx, by); c.stroke(); c.setLineDash([]);
    const ang = Math.atan2(by - ay, bx - ax), tick = g.scale * 1.1;
    for (const [px, py] of [[ax, ay], [bx, by]]) {
      c.beginPath(); c.moveTo(px - Math.sin(ang) * tick, py + Math.cos(ang) * tick); c.lineTo(px + Math.sin(ang) * tick, py - Math.cos(ang) * tick); c.stroke();
    }
    if (text) {
      const mx = (ax + bx) / 2, my = (ay + by) / 2, size = Math.max(10, Math.round(g.scale * 1.55));
      c.font = `600 ${size}px system-ui, sans-serif`;
      const tw = c.measureText(text).width + size * .9, th = size * 1.6;
      const vertical = Math.abs(by - ay) > Math.abs(bx - ax);
      let lx = vertical ? mx + (options.labelSide || 1) * (tw / 2 + size * .6) : mx, ly = vertical ? my : my + (options.labelSide || 1) * th * .85;
      lx = Math.min(g.width - tw / 2 - 4, Math.max(tw / 2 + 4, lx)); ly = Math.min(g.height - th / 2 - 4, Math.max(th / 2 + 4, ly));   // inside the board
      c.fillStyle = '#0d1714e6'; c.strokeStyle = colour; c.lineWidth = 1;
      c.beginPath(); c.roundRect ? c.roundRect(lx - tw / 2, ly - th / 2, tw, th, 4) : c.rect(lx - tw / 2, ly - th / 2, tw, th); c.fill(); c.stroke();
      c.fillStyle = '#f1f5f2'; c.textAlign = 'center'; c.textBaseline = 'middle'; c.fillText(text, lx, ly + 1);
    }
    c.restore();
  }

  drawWidthDepth(c, g) {
    const s = this.shot, wd = s.widthDepth, colour = this.colour(), est = this.data.pitch.estimated;
    const players = s.players;
    const ext = wd && wd.width ? [wd.width.from, wd.width.to, wd.depth.from, wd.depth.to] : [];
    if (wd && wd.width) {
      const xs = players.map(p => p.x), ys = players.map(p => p.y);
      const x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(...ys), y1 = Math.max(...ys);
      // The bounding rectangle of the measured extremes: an outline only (it does not claim the area inside).
      const [ra, rb] = g.toPx(x0, y0), [rc, rd] = g.toPx(x1, y1);
      c.save(); c.strokeStyle = colour + 'aa'; c.lineWidth = 1; c.setLineDash([g.scale * .6, g.scale * .6]);
      c.strokeRect(Math.min(ra, rc), Math.min(rb, rd), Math.abs(rc - ra), Math.abs(rd - rb)); c.restore();
      // Guides from the extreme players to the indicators, then the indicators just outside the shape.
      const L = this.data.pitch.length, W = this.data.pitch.width, offX = 3.5 / L, offY = 3.5 / W;
      const wx = Math.min(1 + 2 / L, x1 + offX), dy = Math.min(1 + 2 / W, y1 + offY);
      const partialW = wd.widthStatus === 'partial', partialD = wd.depthStatus === 'partial';
      c.save(); c.strokeStyle = '#ffffff55'; c.lineWidth = 1; c.setLineDash([3, 3]);
      const pW0 = players.find(p => p.id === wd.width.from), pW1 = players.find(p => p.id === wd.width.to);
      const pD0 = players.find(p => p.id === wd.depth.from), pD1 = players.find(p => p.id === wd.depth.to);
      for (const [p, end] of [[pW0, [wx, pW0.y]], [pW1, [wx, pW1.y]], [pD0, [pD0.x, dy]], [pD1, [pD1.x, dy]]]) {
        const [a1, b1] = g.toPx(p.x, p.y), [a2, b2] = g.toPx(end[0], end[1]); c.beginPath(); c.moveTo(a1, b1); c.lineTo(a2, b2); c.stroke();
      }
      c.restore();
      this.measure(c, g, [wx, pW0.y], [wx, pW1.y], `${partialW ? '≥ ' : ''}Width ${this.metresText(wd.width.metres, est)}`, {colour, dashed: partialW, labelSide: wx > .85 ? -1 : 1});
      this.measure(c, g, [pD0.x, dy], [pD1.x, dy], `${partialD ? '≥ ' : ''}Depth ${this.metresText(wd.depth.metres, est)}`, {colour, dashed: partialD, labelSide: dy > .9 ? -1 : 1});
    }
    this.drawPlayers(c, g, {ring: new Set(ext)});
    if (!wd || !wd.width) this.setNotice(`Not enough outfield players observed at ${clockTenth(s.t)} (${wd ? wd.players : 0} of ${s.expected}) for a width or depth.`);
    else this.setNotice('');
    const legend = el('span', 'muted', 'Rings: the players defining the measured width and depth. Dashed indicators: partial (an extreme player may be unobserved). Grey dots: opponents, for context.');
    this.legend.append(legend);
  }

  drawLine(c, g) {
    const s = this.shot, line = s.line, colour = this.colour(), est = this.data.pitch.estimated;
    const unit = new Set(line && line.defenders ? line.defenders.map(d => d.id) : []);
    if (line && line.height !== undefined && line.status !== 'displaced' && line.lateral) {
      const L = this.data.pitch.length, side = line.side;
      const x = side === 'left' ? line.height / L : 1 - line.height / L;
      const xGoal = side === 'left' ? 0 : 1;
      const lat = [Math.max(-.02, line.lateral[0] - 2 / this.data.pitch.width), Math.min(1.02, line.lateral[1] + 2 / this.data.pitch.width)];
      const shaky = line.status !== 'complete' || line.quality !== 'coherent';
      // The line across the defensive unit's lateral extent (not an official full-width offside line).
      const [ax, ay] = g.toPx(x, lat[0]), [bx, by] = g.toPx(x, lat[1]);
      c.save(); c.strokeStyle = colour; c.lineWidth = Math.max(2, g.scale * .3); if (shaky) c.setLineDash([g.scale * 1.1, g.scale * .7]);
      c.beginPath(); c.moveTo(ax, ay); c.lineTo(bx, by); c.stroke(); c.restore();
      // Deepest defender: a short tick at his distance.
      const xd = side === 'left' ? line.deepest / L : 1 - line.deepest / L, [dx1, dy1] = g.toPx(xd, lat[0]), [dx2, dy2] = g.toPx(xd, lat[0] + .05);
      c.save(); c.strokeStyle = '#ffffffaa'; c.lineWidth = 1.5; c.beginPath(); c.moveTo(dx1, dy1); c.lineTo(dx2, dy2); c.stroke(); c.restore();
      // Measurement from the own goal line to the line, below the unit.
      const ym = Math.min(1.03, lat[1] + 2.5 / this.data.pitch.width);
      this.measure(c, g, [xGoal, ym], [x, ym], `DEFENSIVE LINE — ${this.metresText(line.height, est)} FROM OWN GOAL`, {colour, labelSide: ym > .92 ? -1 : 1});
    }
    this.drawPlayers(c, g, {ring: unit, faded: unit.size > 0});
    const why = !line ? '' : line.status === 'unavailable' ? `No defensive line at ${clockTenth(s.t)}: ${line.reason || 'not enough defenders measured'}.` :
      line.status === 'displaced' ? `At ${clockTenth(s.t)} the defensive unit is ahead of most of its team (a set piece or an attack): no defensive line is drawn.` : '';
    this.setNotice(why);
    this.legend.append(el('span', 'muted', 'Rings: the defensive unit (from the formation roles). Solid line: complete and coherent; dashed: partial, uneven or stretched. The white tick marks the deepest defender.'));
  }

  periodShown() {
    const periods = (this.data && this.data.formation.periods) || [];
    if ((this.period === null || !periods[this.period]) && periods.length) this.period = this.bestPeriodIndex(this.snapT === null ? periods[0].t0 : this.snapT);
    return this.period !== null && periods[this.period] ? periods[this.period] : null;
  }

  drawFormation(c, g) {
    const p = this.periodShown(), colour = this.colour();
    const snapshot = this.formationView === 'snapshot';
    let points = [];
    if (snapshot) {
      const roles = this.shot.roles || {};
      points = this.shot.players.filter(q => q.kind === 'O').map(q => ({...q, role: roles[q.id] ? roles[q.id].role : null, line: roles[q.id] ? roles[q.id].line : null}));
    } else if (p && p.shape) {
      const roles = (p.correction && p.correctedRoles) || p.roles || {};
      points = p.shape.map(q => ({...q, role: roles[q.id] ? roles[q.id].role : null, line: roles[q.id] ? roles[q.id].line : null}));
    }
    // Idealized role anchors of the fitted template: faint, and never where the players are drawn.
    if (!snapshot && p && p.anchors && !p.correction) {
      c.save();
      for (const a of p.anchors) {
        const [ax, ay] = g.toPx(a.x, a.y), r = g.playerRadius * .75;
        c.strokeStyle = '#ffffff4d'; c.lineWidth = 1; c.setLineDash([3, 3]); c.beginPath(); c.arc(ax, ay, r, 0, Math.PI * 2); c.stroke(); c.setLineDash([]);
      }
      c.restore();
    }
    // Units: the players of each line joined from the team's left to its right (its own attacking point of view), so
    // the lines never cross whichever formation the roles come from (the prediction, or your label).
    const side = snapshot ? this.shot.side : (p && p.side);
    const lateral = q => (side === 'right' ? 1 - q.y : q.y);
    const lines = new Map();
    for (const q of points) if (q.line !== null && q.line !== undefined) { if (!lines.has(q.line)) lines.set(q.line, []); lines.get(q.line).push(q); }
    c.save(); c.strokeStyle = colour + 'aa'; c.lineWidth = Math.max(1.5, g.scale * .18);
    for (const members of lines.values()) {
      members.sort((a, b) => lateral(a) - lateral(b));
      if (members.length < 2) continue;
      c.beginPath(); members.forEach((q, i) => { const [x, y] = g.toPx(q.x, q.y); if (i) c.lineTo(x, y); else c.moveTo(x, y); }); c.stroke();
    }
    c.restore();
    if (snapshot) { if (this.shot.goalkeeper) this.dot(c, g, this.shot.goalkeeper.x, this.shot.goalkeeper.y, colour, {gk: true, alpha: .85}); for (const o of this.shot.opponents || []) this.dot(c, g, o.x, o.y, '#9aa7b2', {scale: .62, alpha: .4}); }
    else if (p && p.goalkeeper && points.length) this.dot(c, g, p.goalkeeper.x, p.goalkeeper.y, colour, {gk: true, alpha: .85});
    for (const q of points) this.dot(c, g, q.x, q.y, colour, {label: q.role ? `${q.display || q.id} · ${q.role}` : (q.display || q.id)});
    if (!points.length) this.setNotice(p ? `${PERIOD_TEXT[p.label] || p.label} in ${clock(p.t0)}–${clock(p.t1)}: no shape to show.` : 'No formation period in this range.');
    else this.setNotice('');
    const f = this.shot.formation;
    this.legend.append(el('span', 'muted', snapshot ? `Measured positions at ${clockTenth(this.shot.t)} with the roles of the formation window around it${f && f.coach ? ' (under your label)' : ''}.` :
      `Average shape of the period from observed positions, with the goalkeeper's average position${p && p.correction ? '; roles under your label' : '; faint circles are the fitted template\'s idealized role anchors (players are never moved onto them)'}.`));
  }

  // ---------- chart and timeline ----------
  chartGeometry() {
    const dpr = Math.min(3, global.devicePixelRatio || 1), w = Math.max(200, Math.round((this.chart.clientWidth || 600) * dpr)), h = Math.round(124 * dpr);
    if (this.chart.width !== w || this.chart.height !== h) { this.chart.width = w; this.chart.height = h; }
    const m = {l: 40 * dpr, r: 10 * dpr, t: 24 * dpr, b: 20 * dpr};
    const r = this.data.range, x = t => m.l + (t - r.t0) / Math.max(1e-6, r.t1 - r.t0) * (w - m.l - m.r);
    return {w, h, m, dpr, x, t: px => r.t0 + (px - m.l) / Math.max(1, w - m.l - m.r) * (r.t1 - r.t0)};
  }

  drawChart() {
    if (!this.active || !this.data || this.tab === 'formation') { this.chart.hidden = true; return; }
    this.chart.hidden = false;
    const cg = this.chartGeometry(), c = this.chart.getContext('2d'), d = this.data, colour = this.colour(), est = d.pitch.estimated;
    c.clearRect(0, 0, cg.w, cg.h);
    const series = this.tab === 'width' ? [
      {key: 'width', status: 'widthStatus', label: 'Width', colour, dash: []},
      {key: 'depth', status: 'depthStatus', label: 'Depth', colour: '#f1f5f2', dash: [5 * cg.dpr, 4 * cg.dpr]}] : [
      {key: 'height', status: 'status', label: 'Line height', colour, dash: []},
      {key: 'deepest', status: 'status', label: 'Deepest defender', colour: '#f1f5f2', dash: [5 * cg.dpr, 4 * cg.dpr]}];
    const rows = this.tab === 'width' ? d.widthDepth.series : d.line.series;
    // The server thins long series for the chart; samples are joined across at most 1.5 of its steps.
    const step = (this.tab === 'width' ? d.widthDepth.step : d.line.step) || d.chartStep || .2, gap = Math.max(1.2, 1.5 * step + .05);
    let top = 10;
    for (const s of series) for (const r of rows) if (Number.isFinite(r[s.key])) top = Math.max(top, r[s.key]);
    top = Math.ceil((top + 4) / 10) * 10;
    const y = v => cg.h - cg.m.b - v / top * (cg.h - cg.m.t - cg.m.b);
    c.font = `${11 * cg.dpr}px system-ui, sans-serif`; c.fillStyle = '#718278'; c.strokeStyle = '#334038'; c.lineWidth = 1;
    for (let v = 0; v <= top; v += top > 60 ? 20 : 10) {
      c.beginPath(); c.moveTo(cg.m.l, y(v)); c.lineTo(cg.w - cg.m.r, y(v)); c.stroke();
      c.textAlign = 'right'; c.textBaseline = 'middle'; c.fillText(`${v}`, cg.m.l - 6 * cg.dpr, y(v));
    }
    const r = d.range, span = r.t1 - r.t0, stepT = span > 3600 ? 600 : span > 1200 ? 300 : span > 600 ? 120 : span > 240 ? 60 : span > 90 ? 20 : 10;
    c.textAlign = 'center'; c.textBaseline = 'top';
    for (let t = Math.ceil(r.t0 / stepT) * stepT; t <= r.t1; t += stepT) c.fillText(clock(t), cg.x(t), cg.h - cg.m.b + 4 * cg.dpr);
    const good = this.tab === 'width' ? (st => st === 'complete' || st === 'reliable') : (st => st === 'complete' || st === 'partial');
    // Samples are joined only to neighbours of the same class: reliable values form the solid line, partial,
    // mostly estimated or low-quality ones a faded line of their own (a lone one is a faded dot), so a lower bound
    // never draws a dip.
    for (const s of series) {
      c.save(); c.strokeStyle = s.colour; c.fillStyle = s.colour; c.lineWidth = 2 * cg.dpr; c.setLineDash(s.dash);
      let prev = null;
      const points = [];
      for (const row of rows) {
        const v = row[s.key], ok = Number.isFinite(v) && (row[s.status] !== 'unavailable' && row[s.status] !== 'displaced');
        if (!ok) { if (prev && !prev.linked) points.push(prev); prev = null; continue; }
        const strong = good(row[s.status]) && row.quality !== 'disrupted';
        const joined = prev && prev.strong === strong && row.t - prev.t <= gap;
        if (joined) {
          c.globalAlpha = strong ? 1 : .35;
          c.beginPath(); c.moveTo(cg.x(prev.t), y(prev.v)); c.lineTo(cg.x(row.t), y(v)); c.stroke();
          prev.linked = true;
        } else if (prev && !prev.linked) points.push(prev);
        prev = {t: row.t, v, strong, linked: joined};
      }
      if (prev && !prev.linked) points.push(prev);
      for (const q of points) { c.globalAlpha = q.strong ? 1 : .35; c.beginPath(); c.arc(cg.x(q.t), y(q.v), 1.8 * cg.dpr, 0, Math.PI * 2); c.fill(); }
      c.restore();
    }
    if (this.snapT !== null) { c.strokeStyle = '#ffffffcc'; c.lineWidth = 1.5 * cg.dpr; c.beginPath(); c.moveTo(cg.x(this.snapT), cg.m.t); c.lineTo(cg.x(this.snapT), cg.h - cg.m.b); c.stroke(); }
    // Legend on its own row above the plot: the unit, the series, how to read the faded parts and the gaps.
    const ly = 12 * cg.dpr, cssWidth = cg.w / cg.dpr;
    c.textAlign = 'left'; c.textBaseline = 'middle'; c.fillStyle = '#718278';
    const unit = cssWidth < 420 ? (est ? 'm (est.)' : '') : `${this.tab === 'width' ? 'metres' : 'metres from own goal'}${est ? ' (est.)' : ''}`;
    if (unit) c.fillText(unit, 8 * cg.dpr, ly);
    let lx = 8 * cg.dpr + (unit ? c.measureText(unit).width + 18 * cg.dpr : 0);
    for (const s of series) {
      c.strokeStyle = s.colour; c.lineWidth = 2 * cg.dpr; c.setLineDash(s.dash); c.beginPath(); c.moveTo(lx, ly); c.lineTo(lx + 18 * cg.dpr, ly); c.stroke(); c.setLineDash([]);
      c.fillStyle = '#9aaa9f'; c.fillText(s.label, lx + 22 * cg.dpr, ly); lx += c.measureText(s.label).width + 40 * cg.dpr;
    }
    c.fillStyle = '#718278'; c.textAlign = 'right';
    const hint = cssWidth >= 900 ? 'faded: partial or low quality · gaps: no reliable value · click to choose the snapshot' : cssWidth >= 600 ? 'faded: partial · gaps: no value' : '';
    if (hint && c.measureText(hint).width < cg.w - cg.m.r - lx - 12 * cg.dpr) c.fillText(hint, cg.w - cg.m.r, ly);
  }

  chartClick(e) {
    if (!this.data || this.tab === 'formation') return;
    const rect = this.chart.getBoundingClientRect(), cg = this.chartGeometry();
    const px = (e.clientX - rect.left) * (cg.w / Math.max(1, rect.width));
    this.setSnap(cg.t(px));
  }

  drawTimeline() {
    const tl = this.timeline;
    if (!this.active || !this.data || this.tab !== 'formation') { tl.hidden = true; tl.dataset.key = ''; return; }
    tl.hidden = false;
    const r = this.data.range, span = Math.max(1e-6, r.t1 - r.t0), periods = this.data.formation.periods;
    const key = `${this.data.team}|${this.data.mode}|${r.t0}|${r.t1}|${this.period}|${tl.clientWidth}|${periods.map(p => `${p.label}:${p.t0}:${p.t1}:${p.confidence}:${p.correction ? p.correction.formation : ''}`).join(',')}`;
    if (tl.dataset.key === key) return;
    tl.dataset.key = key;
    tl.replaceChildren();
    const bar = el('div', 'ts-bar');
    bar.setAttribute('role', 'group'); bar.setAttribute('aria-label', 'Formation periods');
    periods.forEach((p, i) => {
      const [a, b] = p.clipped || [p.t0, p.t1];
      const block = el('button', 'ts-period' + (FORMATION_COLOURS[p.label] ? '' : ' none') + (i === this.period ? ' selected' : ''));
      block.type = 'button';
      block.setAttribute('aria-pressed', String(i === this.period));
      block.style.left = `${(a - r.t0) / span * 100}%`; block.style.width = `${Math.max(.6, (b - a) / span * 100)}%`;
      if (FORMATION_COLOURS[p.label]) block.style.setProperty('--fc', FORMATION_COLOURS[p.label]);
      const label = p.correction ? `${p.correction.formation} (set)` : (FORMATION_COLOURS[p.label] ? p.label : (PERIOD_TEXT[p.label] || p.label));
      block.title = `${clock(a)}–${clock(b)} · ${label}${p.confidence ? ' · ' + p.confidence : ''}`;
      block.setAttribute('aria-label', block.title);
      block.append(el('span', '', label));
      block.addEventListener('click', () => { this.period = i; this.layerKey = null; this.setSnap((a + b) / 2); });
      bar.append(block);
    });
    const axis = el('div', 'ts-axis');
    axis.append(el('span', '', clock(r.t0)), el('span', '', clock(r.t1)));
    tl.append(bar, axis);
  }

  // ---------- insight panel ----------
  describe() {
    const panel = this.insights, d = this.data, s = this.shot;
    if (!panel || !this.active) return;
    panel.replaceChildren();
    if (!d) { panel.append(el('p', 'an-empty', this.message || EMPTY)); return; }
    const team = d.team, est = d.pitch.estimated, m = this.unit(est);
    const head = el('div', 'who', `Team ${team}`);
    const sub = el('div', 'team'); const sw = el('i'); sw.style.background = this.colour();
    sub.append(sw, document.createTextNode(`${{width: 'Width & depth', line: 'Defensive line', formation: 'Formation'}[this.tab]} · ${clock(d.range.t0)}–${clock(d.range.t1)}`));
    panel.append(head, sub);
    const row = (label, value, low) => { const kv = el('div', 'kv'); const b = el('b', low ? 'low' : '', value); kv.append(el('span', '', label), b); panel.append(kv); return kv; };
    const notes = [];
    if (d.range.note === 'reversed' || this.range().reversed) notes.push('The custom range was given end first; it is shown from the earlier time.');
    if (d.range.note === 'clipped') notes.push(`The custom range reaches beyond the analysed segment; it is shown within ${clock(d.range.segment[0])}–${clock(d.range.segment[1])}.`);
    const change = d.coverage.lineupChange;
    if (change && change.from !== null && change.from !== undefined) notes.push(`From ${clock(change.from)} the team plays with ${change.outfield} outfield players (set by you).`);
    if (this.tab === 'width') {
      const w = d.widthDepth.width, dp = d.widthDepth.depth, wd = s && s.widthDepth;
      row('Average width', this.metresText(w.mean, est));
      row('Average depth', this.metresText(dp.mean, est));
      row('Median width · depth', `${num(w.median)} · ${num(dp.median)}${m}`);
      row('Width range', w.min === null ? '–' : `${w.min.toFixed(1)}–${w.max.toFixed(1)}${m}`);
      row('Depth range', dp.min === null ? '–' : `${dp.min.toFixed(1)}–${dp.max.toFixed(1)}${m}`);
      if (wd) {
        const weak = wd.widthStatus === 'partial' || wd.widthStatus === 'estimated';
        row(`Snapshot ${clockTenth(s.t)} width`, wd.width ? `${wd.widthStatus === 'partial' ? '≥ ' : ''}${this.metresText(wd.width.metres, est)}` : '–', weak);
        row(`Snapshot depth`, wd.depth ? `${wd.depthStatus === 'partial' ? '≥ ' : ''}${this.metresText(wd.depth.metres, est)}` : '–', wd.depthStatus === 'partial' || wd.depthStatus === 'estimated');
        row('Snapshot coverage', `${wd.observed}/${wd.expected} observed${wd.estimated ? ` + ${wd.estimated} estimated` : ''} · ${STATUS_TEXT[wd.widthStatus] || wd.widthStatus}`, wd.widthStatus !== 'complete' && wd.widthStatus !== 'reliable');
        if (wd.geometric !== null && wd.geometric !== undefined) row('Snapshot confidence', `registration ${pct(wd.geometric)} · identity ≥ ${pct(wd.identity)}`, wd.identity < .5);
        if (wd.robustWidth !== undefined && wd.robustWidth !== null) row('Central spread (10–90%)', `${wd.robustWidth.toFixed(1)} × ${wd.robustDepth.toFixed(1)}${m}`);
      }
      row('Reliable time', pct(w.reliableShare || 0), (w.reliableShare || 0) < .5);
      if (this.other && this.other.widthDepth && this.other.team !== team) {
        const o = this.other.widthDepth;
        panel.append(el('h3', '', `Team ${this.other.team}, same range`));
        row('Average width · depth', `${num(o.width.mean)} · ${num(o.depth.mean)}${m}`);
      }
      if ((w.partialShare || 0) > 0) notes.push(`${pct(w.partialShare)} of the time only a partial spread was observed (an extreme player may have been off camera); those values are lower bounds and are not in the averages.`);
      if ((w.estimatedShare || 0) > 0) notes.push(`${pct(w.estimatedShare)} of the time the shape was mostly made of estimated positions; those values are shown faded and are not in the averages.`);
      if ((w.excessShare || 0) > 0) notes.push(`${pct(w.excessShare)} of the time more players than the lineup were observed (a duplicate identity, or a player who left still tracked: mark him as left); those values are not in the averages.`);
    } else if (this.tab === 'line') {
      const ls = d.line.summary, line = s && s.line, dir = d.direction;
      const how = {set: 'set by you', tracking: 'from tracking', inferred: 'inferred', evidence: 'from positions'}[dir.source] || 'from positions';
      // A range across the side swap names both goals; one with both sides and no swap says which side dominated.
      const goal = side => (side === 'left' ? 'left goal' : 'right goal'), cap = text => text[0].toUpperCase() + text.slice(1);
      const both = Object.keys(dir.sides || {}).length > 1, swapped = both && dir.switchAt !== null && dir.switchAt !== undefined;
      let dirText = 'Uncertain';
      if (dir.side && swapped) {
        const before = dir.base || dir.side, after = before === 'left' ? 'right' : 'left';
        dirText = `${cap(goal(before))} until ${clock(dir.switchAt)}, then ${goal(after)} · ${how}`;
      } else if (dir.side) dirText = `${both ? `Mostly the ${goal(dir.side)}` : cap(goal(dir.side))} · ${how}`;
      row('Defends', dirText, !dir.side);
      row('Line height (median)', ls.height ? this.metresText(ls.height.median, est) : '–');
      row('Deepest defender (median)', ls.deepest ? this.metresText(ls.deepest.median, est) : '–');
      row('Trend', ls.trend ? `${ls.trend.label} · ${ls.trend.change > 0 ? '+' : ''}${ls.trend.change.toFixed(1)} m (${ls.trend.metresPerMinute > 0 ? '+' : ''}${ls.trend.metresPerMinute.toFixed(1)} m/min)${est ? ' (est.)' : ''}` : 'not enough data');
      if (line) {
        row(`Snapshot ${clockTenth(s.t)} line`, line.height !== undefined && line.status !== 'displaced' ? this.metresText(line.height, est) : (STATUS_TEXT[line.status] || line.status), line.status !== 'complete');
        if (line.deepest !== undefined) row('Snapshot deepest', `${line.status === 'partial' ? '≤ ' : ''}${this.metresText(line.deepest, est)}`, line.status === 'partial');
        if (line.quality) row('Line quality', `${QUALITY_TEXT[line.quality]} · spread ${line.spread.toFixed(1)}${m}`, line.quality !== 'coherent');
        if (line.defenders) row('Defensive unit', `${line.measured}/${line.unit} measured · ${SOURCE_TEXT[line.source] || line.source}`, line.status !== 'complete' || line.source !== 'formation');
        if (line.geometric !== null && line.geometric !== undefined) row('Snapshot confidence', `registration ${pct(line.geometric)} · identity ≥ ${pct(line.identity)}`, line.identity < .5);
      }
      row('Measured time', pct(ls.measuredShare || 0), (ls.measuredShare || 0) < .5);
      row('Context', 'Positional (possession not available)');
      if (ls.measuredShare > 0 && !ls.deepest) notes.push('No sample of this range had every defender measured, so the deepest defender is not summarised.');
      if ((ls.estimatedShare || 0) > 0) notes.push(`${pct(ls.estimatedShare)} of the time the line stood mostly on estimated defenders; it is shown faded and is not summarised.`);
      if (dir.contradicted && dir.contradicted.length) notes.push(`The players' positions contradict the defending direction in ${dir.contradicted.map(c => `${clock(c.t0)}–${clock(c.t1)}`).join(', ')} (a side change?). No line is reported there until you set it.`);
      panel.append(this.directionForm(d));
    } else {
      const p = this.periodShown();
      if (p) {
        const named = !!FORMATION_COLOURS[p.label];
        const predicted = named ? (p.confidence === 'ambiguous' ? `${p.label} or ${p.alternative}` : `${p.confidence === 'clear' ? '' : 'LIKELY '}${p.label}`) : (PERIOD_TEXT[p.label] || p.label);
        // The coach's label leads when there is one; the prediction stays visible beside it, unchanged.
        const big = el('div', 'ts-formation', p.correction ? p.correction.formation : predicted);
        if (p.correction) big.append(el('small', 'ts-tag', 'set by you'));
        panel.append(big);
        if (p.correction) row('PitchIQ prediction', predicted);
        row('Time range', `${clock(p.clipped ? p.clipped[0] : p.t0)}–${clock(p.clipped ? p.clipped[1] : p.t1)}`);
        const expected = p.expected || d.formation.expected;
        if (named) {
          row('Confidence', `${p.confidence}${p.related ? ' · related shapes' : ''}`, p.confidence !== 'clear');
          if (p.alternative) {
            const alt = (p.ranking || []).find(x => x.formation === p.alternative), mine = (p.ranking || []).find(x => x.formation === p.label);
            row('Alternative', `${p.alternative}${alt && mine ? ` · mismatch ${mine.rms.toFixed(1)} vs ${alt.rms.toFixed(1)}${m}` : ''}`);
          }
          row('Supporting windows', `${pct(p.support || 0)} of ${p.evidenceWindows}`);
          if (p.roleStability !== undefined) row('Stable roles', pct(p.roleStability), p.roleStability < .7);
        }
        if (p.players !== undefined) row('Coverage', `${p.minPlayers === p.players ? p.players : `${p.minPlayers}–${Math.round(p.players * 10) / 10}`}/${expected} outfield players`, (p.minPlayers || 0) < expected);
        if (p.identity !== undefined) row('Identity confidence', pct(p.identity), p.identity < .6);
        row('Phase', 'Observed positional structure');
        if (p.excess) notes.push('More players than the lineup were observed in some windows (a substitution?); the least observed ones were left out of the shape.');
        const form = this.labelForm(d, p);
        if (form) panel.append(form);
      } else panel.append(el('p', 'an-empty', 'No formation period in this range.'));
      const orphans = (d.formation.corrections || []).filter(c => c.applied === false);
      if (orphans.length) notes.push(`${orphans.length === 1 ? 'One of your formation labels no longer matches' : `${orphans.length} of your formation labels no longer match`} a period (the periods moved after a re-run or a setting change): kept, but not applied.`);
      notes.push('Formations are recognised from observed positions only, over 15 s windows, and a change must be supported for a sustained stretch. Possession is not known, so attacking and defending shapes are not separated.');
      panel.append(this.lineupForm(d));
    }
    // The registration behind the team's observed positions in the range (soccer/team_shape.registration_quality): said
    // when a MEDIUM registration holds most of that time, with the measured share and nothing more.
    const q = d.quality;
    if (q && q.mostlyMedium) notes.push(`${pct(q.medium)} of the observed positions' time comes from positions the pitch registration places with only medium confidence (not high), so these measurements are less certain.`);
    if (est) notes.push(`The pitch size was not measured: metres assume a ${d.pitch.length} × ${d.pitch.width} m pitch.`);
    this.setStatus(notes.join(' '));
  }

  draft(key, value) { if (value === undefined) return this.drafts.get(key); this.drafts.set(key, value); return value; }

  directionForm(d) {
    const box = el('div', 'ts-form'), key = 'direction', draft = this.draft(key) || {};
    box.append(el('h3', '', 'Defending direction'));
    const sel = el('select'); sel.setAttribute('aria-label', 'Team A defends');
    for (const [v, t] of [['', 'From tracking'], ['left', 'Team A defends the left goal'], ['right', 'Team A defends the right goal']]) { const o = el('option', '', t); o.value = v; sel.append(o); }
    sel.value = draft.side !== undefined ? draft.side : (d.settings.direction.A || '');
    const sw = el('input'); sw.type = 'number'; sw.min = 0; sw.step = 1; sw.placeholder = 'sides swap at (s)'; sw.setAttribute('aria-label', 'Sides swap at (seconds)');
    const stored = d.settings.direction.switchAt;
    sw.value = draft.swap !== undefined ? draft.swap : (stored !== null && stored !== undefined ? stored : '');
    sw.disabled = !sel.value;
    const keep = () => this.draft(key, {side: sel.value, swap: sw.value});
    sel.addEventListener('change', () => { sw.disabled = !sel.value; keep(); });
    sw.addEventListener('input', keep);
    const apply = el('button', '', 'Apply'); apply.type = 'button';
    apply.addEventListener('click', async () => {
      const text = sw.value.trim(), value = Number(text);
      if (sel.value && text !== '' && !(Number.isFinite(value) && value >= 0)) { this.setStatus('The side-swap time must be seconds (0 or more).'); return; }
      await this.save('/api/team-shape/settings', {direction: {A: sel.value || null, switchAt: sel.value && text !== '' ? value : null}}, key);
    });
    box.append(sel, sw, apply);
    return box;
  }

  lineupForm(d) {
    const team = d.team, box = el('div', 'ts-form'), key = `lineup|${team}`, draft = this.draft(key) || {};
    const stored = d.settings.lineup[team], def = d.coverage.default || 10;
    const storedN = stored === null || stored === undefined ? def : (typeof stored === 'object' ? stored.outfield : stored);
    const storedFrom = stored && typeof stored === 'object' && stored.from !== null && stored.from !== undefined ? String(stored.from) : '';
    box.append(el('h3', '', `Outfield players on the pitch (team ${team})`));
    const sel = el('select'); sel.setAttribute('aria-label', 'Outfield players on the pitch');
    for (const n of [...new Set([10, 9, 8, 7, def])].sort((a, b) => b - a)) {
      const o = el('option', '', n === def ? `${n} (full team)` : n < def ? `${n} (after dismissals)` : `${n}`); o.value = n; sel.append(o);
    }
    sel.value = String(draft.n !== undefined ? draft.n : storedN);
    const from = el('input'); from.type = 'number'; from.min = 0; from.step = 1; from.placeholder = 'from (s), blank = whole video'; from.setAttribute('aria-label', 'From (seconds)');
    from.value = draft.from !== undefined ? draft.from : storedFrom;
    const keep = () => this.draft(key, {n: sel.value, from: from.value});
    sel.addEventListener('change', keep); from.addEventListener('input', keep);
    const apply = el('button', '', 'Apply'); apply.type = 'button';
    apply.addEventListener('click', () => {
      const n = Number(sel.value), text = from.value.trim(), t = Number(text);
      if (text !== '' && !(Number.isFinite(t) && t >= 0)) { this.setStatus('"From" must be seconds (0 or more), or blank for the whole video.'); return; }
      const value = text === '' ? (n === def ? null : n) : {outfield: n, from: t};
      this.save('/api/team-shape/settings', {lineup: {[team]: value}}, key);
    });
    box.append(sel, from, apply);
    return box;
  }

  labelForm(d, p) {
    const candidates = p.candidates && p.candidates.length ? p.candidates : d.formation.candidates;
    if (!candidates || !candidates.length) return null;   // a lineup without formations (fewer than nine outfield players)
    const box = el('div', 'ts-form'), key = `label|${d.team}|${p.t0}|${p.t1}`;
    box.append(el('h3', '', 'Correct this period'));
    const sel = el('select'); sel.setAttribute('aria-label', 'Formation for this period');
    for (const f of candidates) { const o = el('option', '', f); o.value = f; sel.append(o); }
    const fallback = p.correction ? p.correction.formation : (candidates.includes(p.label) ? p.label : candidates[0]);
    const draft = this.draft(key);
    sel.value = draft && candidates.includes(draft) ? draft : fallback;
    sel.addEventListener('change', () => this.draft(key, sel.value));
    const save = el('button', '', 'Save label'); save.type = 'button';
    save.addEventListener('click', () => this.save('/api/team-shape/formation-label', {team: d.team, t0: p.t0, t1: p.t1, formation: sel.value, predicted: p.label || null}, key));
    box.append(sel, save);
    if (p.correction) {
      const rm = el('button', 'quiet', 'Remove'); rm.type = 'button';
      rm.addEventListener('click', () => this.save('/api/team-shape/formation-label/remove', {team: d.team, t0: p.correction.t0, t1: p.correction.t1}, key));
      box.append(rm);
    }
    box.append(el('p', 'muted', p.evidenceWindows ? 'Your label is stored apart from the tracking; no position changes. It decides the defensive unit of this period.' :
      'Your label is stored apart from the tracking. This period has no formation windows (too few players observed, or no direction), so it cannot decide a defensive unit here.'));
    return box;
  }

  async save(path, body, draftKey) {
    try {
      await this.post(path, body);
      // A new generation: responses still in flight for the old settings are not cached. The saved form shows the
      // stored value again; unsaved choices in the other forms stay.
      this.gen++; this.cache.clear(); this.snaps.clear(); if (draftKey) this.drafts.delete(draftKey); this.layerKey = null;
      await this.request();
    } catch (error) { this.setStatus(error.message); }
  }
}

global.TeamShape = {create: (options) => new TeamShapeView(options)};
})(window);
