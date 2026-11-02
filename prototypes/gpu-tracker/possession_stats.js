/* PitchIQ · Ball possession: the Analytics tab's third category. A dashboard in the place of the pitch and the insight
   panel (index.html #psDash): each team's share of controlled in-play possession, how much of the in-play time those
   shares rest on, the halves side by side, the possession timeline of the chosen range and a breakdown per team.

   Data. Nothing is judged here. GET /api/possession2 (soccer/possession_v2.py; a full match's record is made by
   fullmatch/possession_match.py) answers {state, record, teams, definition} for the shown run (?run=) or, in Full Match
   Mode, the open match (index.html's api() adds match=<id>: match.js SCOPED). The record carries the stats of the whole
   range and of each half (record.halves names them); a custom range is asked for with t0 / t1 and its stats come back
   as `range`. A match's record is made on request: 'missing' or 'needs-state' offers "Work out possession" (POST
   /api/possession2/build {match}), 'building' shows the progress and is asked again every POLL_MS while this category
   is shown, 'error' (or a failed request) says why in one line. The percentages are the server's: where it gives none
   (fewer than settings.minResolved resolved seconds) none is shown and the bar stays empty, never 50/50. Contested, free
   and unresolved time are shown beside the shares, never handed to a team.

   Loading is lazy: load() (a new run, a correction, a full match opened or corrected) only drops what was shown; the
   server is asked when the category is active (setActive), and an answer of an older load is dropped.

   Time. Every time is source seconds, the <video> element's clock. The timeline comes as columns, one entry per
   interval of constant state; the intervals tile the analysed frames in time order (a stretch nobody analysed, such as
   a half-time break that was not processed, is a gap). intervalAt() finds the interval holding a time (binary search).

   Drawing. The strip is one canvas at device resolution: every pixel column takes the kind that fills most of its time
   (pixels(); a full match is about 8000 intervals and never a DOM node each). Team colours are the board's (the colour
   pickers repaint it); an inferred stretch is its team's colour at half strength, contested and free ball two neutrals,
   unresolved a subtle grey, hatched; a confirmed stoppage is dimmed, a possible stoppage dimmed under a dotted top edge,
   half-time shaded apart. The changes of possession are ticks over the strip in the winning team's colour, filled when
   'high', hollow when 'uncertain'. Hover names the interval (or the change) under the pointer; a click selects it, says
   it under the strip and seeks the shared playback to its start, paused (possession.js's seek); with the strip focused,
   Left / Right step through the intervals of the range and Home / End go to its ends. */
(function (global) {
'use strict';

const POLL_MS = 1500;          // a build in progress is asked again this often
const DEFAULT_COLOURS = {A: '#b98cff', B: '#f6b564'};   // tactical.js DEFAULT_COLOURS, until the board says otherwise
const CODES = ['A', 'B', 'contested', 'free', 'unknown', 'possible_stoppage', 'stoppage', 'halftime', 'non_match'];   // record.codes when absent
// What a pixel of the strip shows (pixels()): one kind per interval, the team ones split by observed / inferred.
const KIND = {A_OBS: 0, A_INF: 1, B_OBS: 2, B_INF: 3, CONTESTED: 4, FREE: 5, UNKNOWN: 6, POSSIBLE: 7, STOPPAGE: 8, HALFTIME: 9, NON_MATCH: 10};
const NK = 11;
const BY_CODE = {contested: KIND.CONTESTED, free: KIND.FREE, unknown: KIND.UNKNOWN, possible_stoppage: KIND.POSSIBLE, stoppage: KIND.STOPPAGE,
                 halftime: KIND.HALFTIME, non_match: KIND.NON_MATCH};
// The neutrals were checked against the default team colours with the dataviz palette validator (every pair apart under
// protan / deutan simulation, OKLab dE >= 12; unsimulated >= 15); the inferred shade is the team colour at INFERRED_ALPHA.
const SHADE = {contested: '#d5dde6', free: '#97a3ad', unknown: '#2e3833', hatch: 'rgba(160,176,168,.34)', stoppage: '#161e1a', possible: '#1c2621',
               dots: '#9aaa9f', halftime: 'rgba(143,179,217,.22)', halftimeEdge: 'rgba(143,179,217,.6)', nonMatch: '#111815', select: '#f1f5f2',
               divider: 'rgba(241,245,242,.42)'};
const INFERRED_ALPHA = .5;
const LANE = 12, STRIP_H = 42;  // CSS px: the change ticks' lane over the bar, and the whole canvas (index.html .ps-strip)
const SEEK_INTO = .7;           // frames: an interval starts half a frame before its first frame; a seek lands .2 frame into it
const TICK_HIT = 5;             // CSS px either side of a change tick that hovers or clicks it
const TIP_EDGE = 150;           // CSS px from the strip's ends where the tooltip is anchored to its side, not centred
const COVERAGE = [['observed', 'Observed control', 'c-obs'], ['inferred', 'Inferred control', 'c-inf'], ['contested', 'Contested', 'c-con'],
                  ['free', 'Free ball', 'c-free'], ['unknown', 'Unresolved', 'c-unk']];
const STATE_TITLE = {contested: 'Contested ball', free: 'Free ball', unknown: 'Unresolved', possible_stoppage: 'Possible stoppage',
                     stoppage: 'Confirmed stoppage', halftime: 'Half-time', non_match: 'Not match footage'};
const LEFT_OUT = ['possible_stoppage', 'stoppage', 'halftime', 'non_match'];
const ESTIMATE_NOTE = 'More of the controlled time is inferred than directly observed: the shares are an estimate.';
const HINT = 'Hover the timeline to read an interval; click one (or focus the strip and use ← →) to select it and move playback to its start.';
const EMPTY = 'Ball possession appears here after a run.';

const finite = (v) => typeof v === 'number' && Number.isFinite(v);
const pad2 = (n) => String(n).padStart(2, '0');
const capital = (s) => { const t = String(s === undefined || s === null ? '' : s); return t.charAt(0).toUpperCase() + t.slice(1); };

// mm:ss of source time (the minutes run past 59, as on the transport: a 96-minute match ends at 96:00)
function clock(t) { t = Math.max(0, Math.round(finite(t) ? t : 0)); return `${pad2(Math.floor(t / 60))}:${pad2(t % 60)}`; }
// mm:ss.ss of source time, the page's timestamp()
function clockFine(t) {
  t = Math.round(Math.max(0, finite(t) ? t : 0) * 100) / 100;
  const m = Math.floor(t / 60);
  return `${pad2(m)}:${(t - m * 60).toFixed(2).padStart(5, '0')}`;
}
// a duration as M:SS (the minutes unbounded)
function span(s) { s = Math.max(0, Math.round(finite(s) ? s : 0)); return `${Math.floor(s / 60)}:${pad2(s % 60)}`; }
// a short duration: tenths of a second under a minute, else M:SS
function brief(s) { return !finite(s) ? '–' : s < 59.95 ? `${Math.max(0, s).toFixed(1)} s` : span(s); }
// a percentage (0-100) as a whole number; above 0 and under 1 '<1%', under 100 and above 99 '>99%'. Rounded to the tenth
// first, so that a share of .315 (31.499999 per cent in floating point) shows 32%.
function percent(p) {
  if (!finite(p)) return '–';
  const q = Math.round(p * 10) / 10, r = Math.round(q);
  if (q > 0 && r === 0) return '<1%';
  if (q < 100 && r === 100) return '>99%';
  return `${r}%`;
}
const share = (v) => percent(finite(v) ? v * 100 : NaN);
// Team A's percentage as shown and Team B's as its complement, so that the two shown always add up to 100.
function pair(a) {
  const q = Math.round(a * 10) / 10, r = Math.round(q);
  if (q > 0 && r === 0) return ['<1%', '>99%'];
  if (q < 100 && r === 100) return ['>99%', '<1%'];
  return [`${r}%`, `${100 - r}%`];
}
const sentence = (text) => { const s = String(text || '').trim(); return !s ? '' : /[.!?]$/.test(s) ? s : s + '.'; };
const reason = (error) => sentence(error && error.message || error) || 'The server did not answer.';

// "mm:ss", "m:ss", "h:mm:ss" or seconds (a decimal part on the last field only), as typed in the custom range; null when
// it is none of them. Minutes may pass 59 in "mm:ss"; a field after a colon must be under 60.
function parseTime(text) {
  const s = String(text === undefined || text === null ? '' : text).trim();
  if (!s) return null;
  const parts = s.split(':');
  if (parts.length > 3) return null;
  let v = 0;
  for (let k = 0; k < parts.length; k++) {
    const p = parts[k], last = k === parts.length - 1;
    if (!(last ? /^\d+(\.\d+)?$/ : /^\d+$/).test(p)) return null;
    if (k > 0 && Number(p) >= 60) return null;
    v = v * 60 + Number(p);
  }
  return v;
}

// ---------- the timeline's columns ----------
// The record's timeline as typed arrays (null when its columns are not all there and of one length).
function columns(record) {
  const c = record && record.timeline;
  if (!c || typeof c !== 'object' || !Array.isArray(c.t0)) return null;
  const n = c.t0.length;
  for (const k of ['t1', 'state', 'designation', 'confidence', 'possessor', 'ballSeen', 'decoded']) if (!Array.isArray(c[k]) || c[k].length !== n) return null;
  return {n, t0: Float64Array.from(c.t0), t1: Float64Array.from(c.t1), state: Int16Array.from(c.state), designation: Int8Array.from(c.designation),
          confidence: Int16Array.from(c.confidence), possessor: Int32Array.from(c.possessor), ballSeen: Int16Array.from(c.ballSeen),
          decoded: Int16Array.from(c.decoded), people: Array.isArray(c.people) ? c.people : []};
}
// The kind of every interval (KIND), by the names of the record's codes.
function kindsOf(cols, codes) {
  const names = Array.isArray(codes) && codes.length ? codes : CODES, out = new Int8Array(cols ? cols.n : 0);
  for (let i = 0; i < out.length; i++) {
    const name = names[cols.state[i]], seen = cols.designation[i] === 1;
    out[i] = name === 'A' ? (seen ? KIND.A_OBS : KIND.A_INF) : name === 'B' ? (seen ? KIND.B_OBS : KIND.B_INF)
      : Object.prototype.hasOwnProperty.call(BY_CODE, name) ? BY_CODE[name] : KIND.UNKNOWN;
  }
  return out;
}
// The index of the interval holding time t (t0 <= t < t1; the last one holds its own end too), or -1: before the first,
// after the last, or in a gap between two intervals that do not touch.
function intervalAt(cols, t) {
  if (!cols || !cols.n || !finite(t)) return -1;
  const {t0, t1, n} = cols;
  if (t < t0[0] || t > t1[n - 1]) return -1;
  let lo = 0, hi = n - 1;
  while (lo < hi) { const mid = (lo + hi + 1) >> 1; if (t0[mid] <= t) lo = mid; else hi = mid - 1; }
  return t < t1[lo] || (lo === n - 1 && t <= t1[lo]) ? lo : -1;
}
// The first interval that ends after t (the one holding t, or the next one); n when none does.
function firstAfter(cols, t) {
  let lo = 0, hi = cols.n;
  while (lo < hi) { const mid = (lo + hi) >> 1; if (cols.t1[mid] <= t) lo = mid + 1; else hi = mid; }
  return lo;
}
// The last interval that starts before t; -1 when none does.
function lastBefore(cols, t) {
  let lo = -1, hi = cols.n - 1;
  while (lo < hi) { const mid = (lo + hi + 1) >> 1; if (cols.t0[mid] < t) lo = mid; else hi = mid - 1; }
  return lo;
}
/* The strip at `width` pixel columns over [lo, hi): per column the kind that covers most of its time, or -1 where less
   than half of it is covered at all (time nobody analysed). Every interval adds its overlap with each column it
   touches, so the cost is the intervals plus the columns, whatever the zoom. */
function pixels(cols, kinds, lo, hi, width) {
  width = Math.max(0, Math.floor(width));
  const out = new Int8Array(width).fill(-1);
  if (!cols || !cols.n || !(hi > lo) || !width) return out;
  const acc = new Float64Array(width * NK), scale = width / (hi - lo);
  for (let i = firstAfter(cols, lo); i < cols.n && cols.t0[i] < hi; i++) {
    const a = Math.max(cols.t0[i], lo), b = Math.min(cols.t1[i], hi);
    if (!(b > a)) continue;
    const xa = (a - lo) * scale, xb = (b - lo) * scale, k = kinds[i], last = Math.min(width - 1, Math.ceil(xb) - 1);
    for (let c = Math.max(0, Math.floor(xa)); c <= last; c++) {
      const w = Math.min(xb, c + 1) - Math.max(xa, c);
      if (w > 0) acc[c * NK + k] += w;
    }
  }
  for (let c = 0; c < width; c++) {
    let best = -1, top = 0, sum = 0;
    for (let k = 0; k < NK; k++) { const w = acc[c * NK + k]; sum += w; if (w > top) { top = w; best = k; } }
    out[c] = sum >= .5 ? best : -1;
  }
  return out;
}

// ---------- small DOM helpers (text only: labels and reasons are never parsed as HTML) ----------
function el(tag, cls, text) {
  const e = global.document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined && text !== null) e.textContent = text;
  return e;
}
const textNode = (t) => global.document.createTextNode(t);
function swatch(cls, background, opacity) {
  const s = el('i', 'ps-sw' + (cls ? ' ' + cls : ''));
  if (background) s.style.background = background;
  if (opacity !== undefined) s.style.opacity = String(opacity);
  return s;
}

class PossessionStatsView {
  constructor(options) {
    this.root = options.root; this.c = options.controls || {};
    this.api = options.api; this.post = options.post || options.api;
    this.runName = options.run || (() => null); this.matchOf = options.match || (() => null);
    this.board = options.board || null; this.seekTo = options.seek || (() => {});
    this.dpr = () => Math.min(3, global.devicePixelRatio || 1);
    this.active = false; this.result = null; this.info = null;
    this.gen = 0; this.seq = 0; this.timer = null; this.loading = false; this.posting = false; this.stale = true;
    this.state = 'off'; this.stateInfo = {}; this.teams = null; this.definition = null;
    this.record = null; this.cols = null; this.kinds = null; this.changes = []; this.ranges = new Map();
    this.filter = 'full'; this.choice = 'full'; this.notice = ''; this.rangeError = ''; this.shown = null; this.drawn = false;
    this.selected = -1; this.movedTo = null; this.hatch = null; this.howKey = null; this.stateKey = null;
    this.build();
    const c = this.c;
    if (c.range) c.range.addEventListener('change', () => this.setFilter(c.range.value));
    for (const input of [c.t0, c.t1]) {
      if (!input) continue;
      input.addEventListener('change', () => this.customChanged());
      input.addEventListener('keydown', (e) => { if (e.key === 'Enter') { e.preventDefault(); this.customChanged(); } });
    }
    for (const input of options.colourInputs || []) input.addEventListener('input', () => { if (this.drawn) this.render(); });
    if (typeof global.ResizeObserver === 'function' && this.el.track) new global.ResizeObserver(() => { if (this.active) this.drawStrip(); }).observe(this.el.track);
    this.fillRange();
    this.render();
  }

  // ---------- the skeleton, made once ----------
  build() {
    const E = this.el = {};
    E.state = el('div', 'ps-state');
    E.state.setAttribute('role', 'status');
    E.body = el('div', 'ps-body');
    // the share of possession
    const main = el('section', 'ps-card ps-main'), head = el('div', 'ps-head');
    main.setAttribute('aria-label', 'Possession share');
    E.range = el('span', 'ps-range');
    head.append(el('h3', null, 'Controlled in-play possession'), E.range);
    const side = (team) => {
      const box = el('div', `ps-side ${team.toLowerCase()}`), name = el('span', 'ps-name'), sw = el('i'), label = el('span'), pct = el('b', 'ps-pct', '–');
      box.dataset.team = team;
      name.append(sw, label);
      box.append(name, pct);
      return {box, sw, label, pct};
    };
    E.a = side('A'); E.b = side('B');
    E.bar = el('div', 'ps-bar'); E.bar.setAttribute('aria-hidden', 'true');
    E.segA = el('i'); E.segB = el('i');
    E.segA.dataset.team = 'A'; E.segB.dataset.team = 'B';
    E.bar.append(E.segA, E.segB);
    const hero = el('div', 'ps-hero');
    hero.append(E.a.box, E.bar, E.b.box);
    E.none = el('p', 'ps-none'); E.sub = el('p', 'ps-sub');
    main.append(head, hero, E.none, E.sub);
    // coverage of the in-play time, and the halves
    const cov = el('section', 'ps-card ps-cov'), covHead = el('div', 'ps-head');
    E.live = el('span', 'ps-range');
    covHead.append(el('h3', null, 'Coverage of in-play time'), E.live);
    E.stack = el('div', 'ps-stack'); E.stack.setAttribute('role', 'img');
    E.covSegs = COVERAGE.map(([key, , cls]) => { const s = el('i', cls); s.dataset.key = key; E.stack.append(s); return s; });
    E.covLegend = el('div', 'ps-legend');
    E.left = el('p', 'ps-left'); E.est = el('p', 'ps-est');
    cov.append(covHead, E.stack, E.covLegend, E.left, E.est);
    E.halves = el('section', 'ps-card ps-halves-card');
    E.halfList = el('div', 'ps-halves');
    E.halves.append(el('h3', null, 'Half comparison'), E.halfList);
    E.row = el('div', 'ps-row');
    E.row.append(cov, E.halves);
    // the timeline
    const tl = el('section', 'ps-card ps-tl'), tlHead = el('div', 'ps-head');
    E.tlRange = el('span', 'ps-range');
    tlHead.append(el('h3', null, 'Possession timeline'), E.tlRange);
    E.track = el('div', 'ps-track');
    E.strip = el('canvas', 'ps-strip');
    E.strip.tabIndex = 0;
    E.strip.setAttribute('role', 'slider');
    E.strip.setAttribute('aria-label', 'Possession timeline of the range: Left and Right step through its intervals, a click selects one; either moves playback to its start');
    E.tip = el('div', 'ps-tip'); E.tip.hidden = true; E.tip.setAttribute('aria-hidden', 'true');
    E.track.append(E.strip, E.tip);
    E.axis = el('div', 'ps-axis'); E.axis0 = el('span'); E.axis1 = el('span');
    E.axis.append(E.axis0, E.axis1);
    E.key = el('div', 'ps-legend ps-key');
    E.detail = el('p', 'ps-detail'); E.detail.setAttribute('aria-live', 'polite');
    tl.append(tlHead, E.track, E.axis, E.key, E.detail);
    E.strip.addEventListener('pointermove', (e) => this.hover(e));
    E.strip.addEventListener('pointerleave', () => this.hideTip());
    E.strip.addEventListener('blur', () => this.hideTip());
    E.strip.addEventListener('click', (e) => this.click(e));
    E.strip.addEventListener('keydown', (e) => this.key(e));
    // the breakdown, and how it is counted
    const br = el('section', 'ps-card ps-break');
    E.tables = el('div', 'ps-tables');
    br.append(el('h3', null, 'Breakdown'), E.tables);
    E.how = el('details', 'ps-how');
    E.howBody = el('div', 'ps-how-body');
    E.how.append(el('summary', null, 'How this is counted'), E.howBody);
    E.body.append(main, E.row, tl, br, E.how);
    if (this.root) this.root.replaceChildren(E.state, E.body);
  }

  // ---------- state ----------
  load(result, info) {
    // Every showResult (a new run, the same run after a correction) and every full match opened or corrected: what was
    // shown is dropped; the server is asked when the category is active, now or later.
    this.gen++; this.stopPoll();
    this.result = result || null; this.info = info || null;
    this.record = null; this.cols = null; this.kinds = null; this.changes = []; this.ranges.clear();
    this.teams = null; this.definition = null; this.loading = false; this.posting = false;
    this.state = this.result ? 'idle' : 'off'; this.stateInfo = {}; this.notice = ''; this.rangeError = '';
    this.selected = -1; this.movedTo = null; this.shown = null; this.drawn = false; this.howKey = null; this.stale = true;
    this.hideTip();
    this.fillRange();
    if (this.active && this.result) this.request(); else this.render();
  }

  setActive(on) {
    this.active = !!on;
    if (!this.active) { this.stopPoll(); this.hideTip(); return; }
    if (this.result && (this.stale || this.state === 'building' || this.state === 'idle')) this.request();
    else this.render();
  }

  matchId() {
    const r = this.result, m = r && r.matchLevel && r.matchLevel.matchId;
    if (typeof m === 'string' && m) return m;
    try { const id = this.matchOf(); return typeof id === 'string' && id ? id : null; } catch (e) { return null; }
  }
  isMatch() { return !!this.matchId(); }
  colours() { return this.board && this.board.colours ? this.board.colours : DEFAULT_COLOURS; }
  teamLabels() {
    const t = this.teams || {}, label = (k) => (t[k] && typeof t[k].label === 'string' && t[k].label) || `Team ${k}`;
    return {A: label('A'), B: label('B')};
  }
  fps() {
    const r = this.record, v = this.info && this.info.video;
    return r && finite(r.fps) && r.fps > 0 ? r.fps : v && finite(v.fps) && v.fps > 0 ? v.fps : 25;
  }
  codeName(k) { const names = this.record && Array.isArray(this.record.codes) && this.record.codes.length ? this.record.codes : CODES; return names[k]; }

  url(range) {
    // a Segment run by its name; in a full match index.html's api() adds match=<id> (match.js SCOPED)
    const p = new URLSearchParams();
    if (!this.isMatch()) { const run = this.runName(); if (run) p.set('run', run); }
    if (range) { p.set('t0', String(range.t0)); p.set('t1', String(range.t1)); }
    const q = p.toString();
    return '/api/possession2' + (q ? '?' + q : '');
  }

  async request() {
    this.stopPoll();
    if (!this.active || !this.result) { this.render(); return; }
    const gen = this.gen, seq = ++this.seq;
    const c = this.filter === 'custom' ? this.customRange() : null, range = c && !c.invalid ? c : null;
    this.notice = c && c.invalid ? c.invalid : '';
    this.rangeError = '';
    this.loading = true;
    this.render();
    let answer = null, failed = null;
    try { answer = await this.api(this.url(range)); } catch (error) { failed = error; }
    if (gen !== this.gen || seq !== this.seq) return;   // a newer load or request superseded this one
    this.loading = false; this.stale = false;
    this.take(answer, failed, range);
    this.render();
    this.schedule();
  }

  take(a, failed, range) {
    if (failed) {
      if (range && this.record) { this.rangeError = `The range could not be worked out. ${reason(failed)}`; return; }   // the record stays
      this.state = 'error'; this.stateInfo = {error: `Possession is not shown. ${reason(failed)}`};
      return;
    }
    if (!a || typeof a !== 'object') { this.state = 'error'; this.stateInfo = {error: 'Possession is not shown. The server sent no answer.'}; return; }
    if (a.teams && typeof a.teams === 'object') this.teams = a.teams;
    if (typeof a.definition === 'string' && a.definition) this.definition = a.definition;
    switch (a.state) {
      case 'ready':
        if (!this.adopt(a.record)) { this.state = 'error'; this.stateInfo = {error: 'Possession is not shown. The server sent a record this page cannot read.'}; return; }
        this.state = 'ready'; this.stateInfo = {};
        if (range) {
          if (a.range && typeof a.range === 'object') this.ranges.set(range.key, a.range);
          else this.rangeError = 'The server did not send the stats of this range.';
        }
        return;
      case 'building': this.state = 'building'; this.stateInfo = {progress: a.progress, stage: a.stage}; return;
      case 'missing': case 'needs-state': this.state = a.state; this.stateInfo = {error: a.error || null}; return;
      case 'error': this.state = 'error'; this.stateInfo = {error: sentence(a.error) || 'Possession could not be worked out.'}; return;
      default: this.state = 'error'; this.stateInfo = {error: `Possession is not shown. The server answered "${a.state}".`};
    }
  }

  adopt(record) {
    // The record of a ready answer. A range's answer carries the record again: the same one keeps the selection.
    if (!record || typeof record !== 'object' || !record.stats || typeof record.stats !== 'object' || !record.stats.match) return false;
    if (this.record && this.same(record)) return true;
    const cols = columns(record);
    if (!cols) return false;
    this.record = record; this.cols = cols; this.kinds = kindsOf(cols, record.codes);
    this.changes = (Array.isArray(record.changes) ? record.changes : []).filter((c) => c && finite(c.wonAt)).slice().sort((x, y) => x.wonAt - y.wonAt);
    this.ranges.clear(); this.selected = -1; this.movedTo = null; this.howKey = null;
    this.fillRange();
    return true;
  }
  same(r) {
    const o = this.record, n = (x) => (x.timeline && Array.isArray(x.timeline.t0) ? x.timeline.t0.length : -1);
    if (o.fingerprint !== undefined || r.fingerprint !== undefined) return o.fingerprint === r.fingerprint;
    return o.version === r.version && o.t0 === r.t0 && o.t1 === r.t1 && n(o) === n(r) && (o.changes || []).length === (r.changes || []).length;
  }

  schedule() {
    if (this.active && this.state === 'building' && !this.timer) this.timer = setTimeout(() => { this.timer = null; this.request(); }, POLL_MS);
  }
  stopPoll() { if (this.timer) { clearTimeout(this.timer); this.timer = null; } }

  async startBuild() {
    // "Work out possession": POST the build with the match id, then ask until it is ready.
    const id = this.matchId();
    if (!id || this.posting) return;
    this.stopPoll();
    const gen = this.gen, seq = ++this.seq;
    this.posting = true; this.render();
    let answer = null, failed = null;
    try { answer = await this.post('/api/possession2/build', {match: id}); } catch (error) { failed = error; }
    this.posting = false;
    if (gen !== this.gen || seq !== this.seq) return;
    if (failed) { this.state = 'error'; this.stateInfo = {error: `Possession could not be started. ${reason(failed)}`}; this.render(); return; }
    this.take(answer, null, null);
    if (this.state === 'ready' && this.filter === 'custom') { this.request(); return; }   // the build's answer has no range
    this.render();
    this.schedule();
  }

  // ---------- the range filter ----------
  halves() {
    const r = this.record;
    if (!r || !Array.isArray(r.halves)) return [];
    return r.halves.filter((h) => h && finite(h.t0) && finite(h.t1) && h.t1 > h.t0 && r.stats[h.name] && typeof r.stats[h.name] === 'object')
      .map((h) => ({name: String(h.name), t0: h.t0, t1: h.t1, stats: r.stats[h.name]}));
  }
  full() {
    // the whole analysed range: the record's first to last frame, widened to its intervals
    const r = this.record, c = this.cols;
    let lo = finite(r && r.t0) ? r.t0 : Infinity, hi = finite(r && r.t1) ? r.t1 : -Infinity;
    if (c && c.n) { lo = Math.min(lo, c.t0[0]); hi = Math.max(hi, c.t1[c.n - 1]); }
    if (!finite(lo) || !finite(hi)) return {lo: 0, hi: 1};
    return hi > lo ? {lo, hi} : {lo, hi: lo + 1};
  }
  fillRange() {
    // Full match (in a Segment run: the entire analysed segment), one option per half the record names, Custom range.
    // The range the user chose comes back when the record that arrives still has it (a half after a correction).
    const sel = this.c.range;
    if (!sel || typeof sel.replaceChildren !== 'function') return;
    const keep = this.choice, opts = [['full', this.isMatch() ? 'Full match' : 'Entire analysed segment']];
    this.halves().forEach((h, k) => opts.push([`half:${k}`, capital(h.name)]));
    opts.push(['custom', 'Custom range']);
    sel.replaceChildren(...opts.map(([value, label]) => { const o = el('option', null, label); o.value = value; return o; }));
    this.filter = opts.some(([value]) => value === keep) ? keep : 'full';
    sel.value = this.filter;
    if (this.c.custom) this.c.custom.hidden = this.filter !== 'custom';
  }
  setFilter(value) {
    const known = value === 'full' || value === 'custom' || (/^half:\d+$/.test(String(value)) && Number(String(value).slice(5)) < this.halves().length);
    this.filter = this.choice = known ? value : 'full';
    if (this.c.range && this.c.range.value !== this.filter) this.c.range.value = this.filter;
    if (this.c.custom) this.c.custom.hidden = this.filter !== 'custom';
    if (this.filter === 'custom' && this.c.t0 && this.c.t1 && !String(this.c.t0.value || '').trim() && !String(this.c.t1.value || '').trim()) {
      // a first custom range starts as the range shown, to be edited
      const v = this.shown || (this.record ? this.full() : null);
      if (v) { this.c.t0.value = clock(v.lo); this.c.t1.value = clock(v.hi); }
    }
    this.update();
  }
  customChanged() { if (this.filter === 'custom') this.update(); }
  customRange() {
    const a = parseTime(this.c.t0 && this.c.t0.value), b = parseTime(this.c.t1 && this.c.t1.value);
    if (a === null || b === null) return {invalid: 'Enter both times of the custom range, as mm:ss or seconds.'};
    const t0 = Math.min(a, b), t1 = Math.max(a, b);
    if (!(t1 > t0)) return {invalid: 'The custom range is empty: its end must come after its start.'};
    if (this.record) {
      const f = this.full();
      if (t1 <= f.lo || t0 >= f.hi) return {invalid: `The range ${clock(t0)}–${clock(t1)} lies outside the analysed ${this.isMatch() ? 'match' : 'segment'} (${clock(f.lo)}–${clock(f.hi)}).`};
    }
    return {t0, t1, key: `${t0}|${t1}`};
  }
  update() {
    // after the filter or a custom time changed
    this.rangeError = '';
    if (!this.record || this.state !== 'ready') { this.render(); return; }
    if (this.filter === 'custom') {
      const c = this.customRange();
      this.notice = c.invalid || '';
      if (!c.invalid && !this.ranges.has(c.key)) { this.request(); return; }
    } else this.notice = '';
    this.render();
  }
  current() {
    // the range shown: {kind, label, lo, hi, stats, closed} (closed: the whole record, every change counts)
    const f = this.full();
    if (this.filter.startsWith('half:')) {
      const h = this.halves()[Number(this.filter.slice(5))];
      if (h) return {kind: 'half', label: capital(h.name), lo: h.t0, hi: h.t1, stats: h.stats, closed: false};
    }
    if (this.filter === 'custom') {
      const c = this.customRange();
      if (c.invalid) return null;   // the last range shown stays, with the reason above it
      const a = this.ranges.get(c.key);
      return {kind: 'custom', label: 'Custom range', lo: c.t0, hi: c.t1, stats: a || null, closed: false, pending: !a && this.loading, failed: !a && !this.loading};
    }
    return {kind: 'full', label: this.isMatch() ? 'Full match' : 'Entire analysed segment', lo: f.lo, hi: f.hi, stats: this.record.stats.match, closed: true};
  }

  // ---------- rendering ----------
  render() {
    if (!this.root) return;
    this.hideTip();   // placed for the strip as it was drawn: a new render may move or redraw it
    this.renderState();
    const ready = this.state === 'ready' && !!this.record;
    this.el.body.hidden = !ready;
    if (!ready) { this.el.body.classList.remove('ps-wait'); return; }
    let v = this.current();
    if (!v || v.failed) v = this.shown || {kind: 'full', label: this.isMatch() ? 'Full match' : 'Entire analysed segment', ...this.full(), stats: this.record.stats.match, closed: true};
    const wait = !!v.pending;
    this.el.body.classList.toggle('ps-wait', wait);
    if (wait && this.drawn) return;   // the previous render stays, faded, until the range's stats arrive
    if (this.selected >= 0 && (this.cols.t1[this.selected] <= v.lo || this.cols.t0[this.selected] >= v.hi)) { this.selected = -1; this.movedTo = null; }
    this.shown = v;
    this.renderMain(v); this.renderCoverage(v); this.renderHalves(); this.renderTimeline(v); this.renderBreakdown(v); this.renderHow();
    this.drawn = true;
  }

  renderState() {
    // A live region: written again only when what it says changes.
    const box = this.el.state, st = this.state, info = this.stateInfo || {}, nodes = [];
    const key = JSON.stringify([!!this.result, st, info, this.loading && !this.record, this.posting, this.isMatch(), st === 'ready' ? [this.notice, this.rangeError] : null]);
    if (key === this.stateKey) return;
    this.stateKey = key;
    const line = (text, cls) => { const p = el('p', 'ps-msg' + (cls ? ' ' + cls : '')); p.append(el('i'), el('span', null, text)); return p; };
    if (!this.result) nodes.push(line(EMPTY, 'off'));
    else if (st === 'idle' || (this.loading && !this.record && st !== 'building')) nodes.push(line('Working out possession…', 'busy'));
    else if (st === 'building') nodes.push(this.buildCard());
    else if (st === 'missing' || st === 'needs-state') nodes.push(this.startCard(st));
    else if (st === 'error') {
      const p = line(info.error || 'Possession could not be worked out.', 'error');
      p.dataset.ps = 'error';
      const retry = el('button', null, this.posting ? 'Starting…' : 'Try again');
      retry.type = 'button'; retry.dataset.ps = 'retry'; retry.disabled = this.posting || this.loading;
      retry.addEventListener('click', () => (this.isMatch() ? this.startBuild() : this.request()));
      p.append(retry);
      nodes.push(p);
    }
    if (st === 'ready' && (this.notice || this.rangeError)) {
      const p = line(this.notice || this.rangeError, this.rangeError && !this.notice ? 'error' : 'warn');
      p.dataset.ps = 'notice';
      nodes.push(p);
    }
    box.replaceChildren(...nodes);
    box.hidden = !nodes.length;
  }
  buildCard() {
    const info = this.stateInfo || {}, p = finite(info.progress) ? Math.max(0, Math.min(1, info.progress)) : 0;
    const card = el('div', 'ps-build'), text = el('div', 'ps-build-text'), bar = el('progress');
    card.dataset.ps = 'building';
    text.append(el('b', null, 'Working out possession'), el('span', null, info.stage ? capital(sentence(info.stage)) : 'Starting.'));
    bar.max = 1; bar.value = p;
    bar.setAttribute('aria-label', 'Possession build progress');
    card.append(text, bar, el('b', 'ps-build-pct', `${Math.round(p * 100)}%`));
    return card;
  }
  startCard(st) {
    const card = el('div', 'ps-build'), text = el('div', 'ps-build-text'), id = this.matchId(), info = this.stateInfo || {};
    card.dataset.ps = st;
    if (st === 'needs-state') {
      text.append(el('b', null, 'Possession needs the match state first.'),
                  el('span', null, 'Working it out builds the match state (stoppages, half-time, restarts) first, about a minute, then possession over the whole match.'));
    } else {
      text.append(el('b', null, 'Possession has not been worked out for this match yet.'),
                  el('span', null, 'PitchIQ reads the whole match once (about a minute the first time), then decides who controls the ball, also where it is hidden.'));
    }
    if (info.error) text.append(el('span', 'ps-build-error', sentence(info.error)));
    card.append(text);
    if (id) {
      const go = el('button', 'primary', this.posting ? 'Starting…' : 'Work out possession');
      go.type = 'button'; go.dataset.ps = 'build'; go.disabled = this.posting;
      go.addEventListener('click', () => this.startBuild());
      card.append(go);
    }
    return card;
  }

  renderMain(v) {
    const E = this.el, S = v.stats, teams = this.teamLabels(), col = this.colours();
    E.range.textContent = `${v.label} · ${clock(v.lo)}–${clock(v.hi)}`;
    E.a.label.textContent = teams.A; E.b.label.textContent = teams.B;
    E.a.sw.style.background = col.A; E.b.sw.style.background = col.B;
    const ok = !!S && !!S.percent && S.insufficient !== true && finite(S.percent.A) && finite(S.percent.B);
    E.bar.classList.toggle('empty', !ok);
    E.none.hidden = ok;
    if (ok) {
      const [a, b] = pair(S.percent.A);
      E.a.pct.textContent = a; E.b.pct.textContent = b;
      for (const [seg, value, colour] of [[E.segA, S.percent.A, col.A], [E.segB, S.percent.B, col.B]]) {
        seg.hidden = !(value > 0); seg.style.flex = `${value} 1 0px`; seg.style.background = colour;
      }
      const sub = [textNode('of '), el('b', null, span(S.resolved)), textNode(' resolved controlled in-play time')];
      if (finite(S.live) && S.live > 0) sub.push(textNode(` (${percent(100 * S.resolved / S.live)} of in-play time)`));
      E.sub.replaceChildren(...sub);
      E.sub.title = `${teams.A} ${S.percent.A}% · ${teams.B} ${S.percent.B}% of the resolved controlled time (the server's shares, to the tenth)`;
    } else {
      E.a.pct.textContent = '–'; E.b.pct.textContent = '–';
      E.segA.hidden = true; E.segB.hidden = true;
      E.none.textContent = v.pending ? 'Working out this range…' : 'Not enough resolved possession to give percentages';
      const min = finite(S && S.minResolved) ? S.minResolved : this.record.settings && this.record.settings.minResolved;
      E.sub.replaceChildren(S ? textNode(`${span(S.resolved)} of resolved controlled in-play time in this range` + (finite(min) ? `; percentages need at least ${span(min)}.` : '.')) : textNode(''));
      E.sub.title = '';
    }
  }

  renderCoverage(v) {
    const E = this.el, S = v.stats, sh = S && S.shareOfLive && typeof S.shareOfLive === 'object' ? S.shareOfLive : null, sec = (S && S.seconds) || {};
    E.live.textContent = S && finite(S.live) ? `${span(S.live)} in play` + (finite(S.coverage) ? ` · coverage ${share(S.coverage)}` : '') : '';
    E.stack.hidden = !sh;
    const items = [];
    if (sh) {
      COVERAGE.forEach(([key, name, cls], k) => {
        const value = finite(sh[key]) ? Math.max(0, sh[key]) : 0, seg = E.covSegs[k];
        seg.hidden = !(value > 0); seg.style.flex = `${value} 1 0px`; seg.title = `${name} ${(value * 100).toFixed(1)}%`;
        const item = el('span');
        item.dataset.key = key;
        item.append(swatch(cls), textNode(name + ' '), el('b', null, share(value)));
        items.push(item);
      });
      E.stack.setAttribute('aria-label', 'In-play time: ' + COVERAGE.map(([key, name]) => `${name.toLowerCase()} ${share(sh[key])}`).join(', '));
    } else items.push(el('span', null, 'No in-play time in this range.'));
    E.covLegend.replaceChildren(...items);
    const left = `Left out: confirmed stoppages ${span(sec.stoppage)}, possible stoppages ${span(sec.possible_stoppage)}, halftime ${span(sec.halftime)}`;
    E.left.textContent = left + (finite(sec.non_match) && sec.non_match >= .5 ? `, not match footage ${span(sec.non_match)}` : '');
    const estimate = !!sh && finite(sh.inferred) && finite(sh.observed) && sh.inferred > sh.observed;
    const note = (Array.isArray(this.record.notes) ? this.record.notes : []).find((n) => typeof n === 'string' && /inferred/i.test(n));
    E.est.hidden = !estimate;
    E.est.textContent = estimate ? sentence(note || ESTIMATE_NOTE) : '';
  }

  renderHalves() {
    const E = this.el, halves = this.halves(), col = this.colours();
    E.halves.hidden = !halves.length;
    E.row.classList.toggle('one', !halves.length);
    E.halfList.replaceChildren(...halves.map((h, k) => {
      const S = h.stats, ok = !!S.percent && S.insufficient !== true && finite(S.percent.A) && finite(S.percent.B);
      const b = el('button', 'ps-half'), mini = el('span', 'ps-mini' + (ok ? '' : ' empty'));
      b.type = 'button'; b.dataset.half = String(k);
      b.setAttribute('aria-pressed', String(this.filter === `half:${k}`));
      b.title = `Show the ${h.name} only`;
      if (ok) for (const [team, value] of [['A', S.percent.A], ['B', S.percent.B]]) { if (value > 0) { const s = el('i'); s.style.flex = `${value} 1 0px`; s.style.background = col[team]; mini.append(s); } }
      const pct = ok ? (() => { const [a, c] = pair(S.percent.A); return `A ${a} / B ${c}`; })() : 'Not enough resolved possession';
      b.append(el('span', 'ps-half-name', capital(h.name)), el('span', 'ps-half-when', `${clock(h.t0)}–${clock(h.t1)}`), mini,
               el('span', 'ps-half-pct', pct), el('span', 'ps-half-cov', finite(S.coverage) ? `coverage ${share(S.coverage)} of ${span(S.live)} in play` : 'no in-play time'));
      b.addEventListener('click', () => this.setFilter(`half:${k}`));
      return b;
    }));
  }

  renderTimeline(v) {
    const E = this.el, teams = this.teamLabels(), col = this.colours();
    E.tlRange.textContent = `${v.label} · ${clock(v.lo)}–${clock(v.hi)}`;
    E.axis0.textContent = clock(v.lo); E.axis1.textContent = clock(v.hi);
    E.strip.setAttribute('aria-valuemin', String(Math.round(v.lo)));
    E.strip.setAttribute('aria-valuemax', String(Math.round(v.hi)));
    const item = (parts, text) => { const s = el('span'); s.append(...parts, textNode(text)); return s; };
    const keys = [
      item([swatch(null, col.A), swatch(null, col.A, INFERRED_ALPHA)], teams.A), item([swatch(null, col.B), swatch(null, col.B, INFERRED_ALPHA)], teams.B),
      el('span', 'ps-key-note', 'solid observed, light inferred'),
      item([swatch('k-con')], 'Contested'), item([swatch('k-free')], 'Free ball'), item([swatch('k-unk')], 'Unresolved'),
      item([swatch('k-stop')], 'Stoppage'), item([swatch('k-poss')], 'Possible stoppage')];
    if (this.halves().length || this.kinds.includes(KIND.HALFTIME)) keys.push(item([swatch('k-half')], 'Half-time'));
    keys.push(item([el('i', 'ps-tri')], 'change (high confidence)'), item([el('i', 'ps-tri hollow')], 'uncertain change'));
    E.key.replaceChildren(...keys);
    this.renderDetail();
    this.drawStrip();
  }

  renderDetail() {
    const E = this.el, k = this.selected;
    if (k < 0 || !this.cols || k >= this.cols.n) {
      E.detail.replaceChildren(textNode(HINT));
      E.strip.removeAttribute('aria-valuenow'); E.strip.removeAttribute('aria-valuetext');
      return;
    }
    const d = this.describe(k), nodes = [el('b', null, d.title), textNode(` · ${d.parts.join(' · ')}.`)];
    if (finite(this.movedTo)) nodes.push(textNode(` Playback was moved to ${clockFine(this.movedTo)}: open Tracking or 2D match to watch it.`));
    E.detail.replaceChildren(...nodes);
    E.strip.setAttribute('aria-valuenow', String(Math.round(this.cols.t0[k])));
    E.strip.setAttribute('aria-valuetext', `${d.title}, ${d.parts.join(', ')}`);
  }

  renderBreakdown(v) {
    const E = this.el, S = v.stats, teams = this.teamLabels(), col = this.colours();
    if (!S) { E.tables.replaceChildren(el('p', 'ps-left', 'No stats for this range.')); return; }
    const sec = S.seconds || {}, obs = S.observed || {}, inf = S.inferred || {}, seqs = S.sequences || {}, won = this.wonIn(v);
    const table = (caption, rows, colour, team) => {
      const t = el('table', 'ps-table'), cap = el('caption'), body = el('tbody');
      if (team) t.dataset.team = team;
      if (colour) cap.append(swatch(null, colour));
      cap.append(textNode(caption));
      for (const [name, value] of rows) {
        const tr = el('tr'), th = el('th', null, name);
        th.setAttribute('scope', 'row');
        tr.append(th, el('td', null, value));
        body.append(tr);
      }
      t.append(cap, body);
      return t;
    };
    const teamRows = (k) => {
      const s = seqs[k] || {}, count = finite(s.count) ? s.count : 0;
      const rows = [['Controlled time', span(sec[k])], ['Observed / inferred', `${span(obs[k])} / ${span(inf[k])}`], ['Possession sequences', String(count)]];
      if (count > 0 && finite(s.meanSeconds)) rows.push(['Mean sequence', brief(s.meanSeconds)]);
      rows.push(['Changes won', String(won[k])]);
      return rows;
    };
    E.tables.replaceChildren(table(teams.A, teamRows('A'), col.A, 'A'), table(teams.B, teamRows('B'), col.B, 'B'),
                             table('Neither team', [['Contested', span(sec.contested)], ['Free ball', span(sec.free)], ['Unresolved', span(sec.unknown)]], null, 'none'));
  }

  renderHow() {
    const r = this.record, key = `${r.fingerprint || r.version}|${r.t0}|${r.t1}|${this.definition || ''}`;
    if (key === this.howKey) return;
    this.howKey = key;
    const nodes = [el('p', null, this.definition || r.definition || '')];
    const m = r.model || {}, prov = m.provenance && typeof m.provenance === 'object' ? m.provenance : {};
    let model = `Model ${r.version || 'not named'}`;
    if (prov.fitted || prov.data) model += `, fitted${prov.fitted ? ` ${prov.fitted}` : ''}${prov.data ? ` on ${prov.data}` : ''}`;
    if (prov.tool) model += ` (${prov.tool})`;
    if (m.id) model += `; model id ${m.id}`;
    nodes.push(el('p', null, model + '.'));
    const s = r.settings || {}, rules = [];
    if (finite(s.minPosterior)) rules.push(`a team, a duel or a free ball is named where the model gives it at least ${percent(100 * s.minPosterior)}, else the time is unresolved`);
    if (finite(s.minResolved)) rules.push(`percentages need at least ${span(s.minResolved)} of resolved controlled time`);
    if (rules.length) nodes.push(el('p', null, capital(rules.join('; ')) + '. Every share is an estimate from the tracking.'));
    const bs = r.ballStates && typeof r.ballStates === 'object' ? r.ballStates : null;
    if (bs && Object.keys(bs).length) nodes.push(el('p', null, 'The ball over in-play time: ' + Object.entries(bs).map(([k, val]) => `${k.toLowerCase()} ${share(val)}`).join(', ') + '.'));
    const notes = (Array.isArray(r.notes) ? r.notes : []).filter((n) => typeof n === 'string' && n);
    if (notes.length) { const ul = el('ul'); for (const n of notes) ul.append(el('li', null, sentence(n))); nodes.push(ul); }
    const src = r.source && typeof r.source === 'object' ? r.source : null;
    if (src) {
      const parts = [src.kind === 'match' ? `the whole match${finite(src.segments) ? ` (${src.segments} segments)` : ''}` : src.kind === 'segment' ? 'this segment run' : String(src.kind || 'not named')];
      if (src.state) parts.push(`stoppages from ${src.state}`);
      if (src.people) parts.push(`players from ${src.people}`);
      nodes.push(el('p', null, `Source: ${parts.join('; ')}.`));
    }
    this.el.howBody.replaceChildren(...nodes);
  }

  // ---------- counting ----------
  wonIn(v) {
    // the changes of possession won by each team in the range ([lo, hi); the whole record counts every change)
    const out = {A: 0, B: 0};
    for (const c of this.changes) if ((c.to === 'A' || c.to === 'B') && (v.closed || (c.wonAt >= v.lo && c.wonAt < v.hi))) out[c.to]++;
    return out;
  }

  describe(k) {
    // The words of one interval, for its tooltip and for the selection under the strip.
    const c = this.cols, name = this.codeName(c.state[k]), teams = this.teamLabels();
    let title = name === 'A' || name === 'B' ? `${teams[name]} in possession · ${c.designation[k] === 1 ? 'observed' : 'inferred'}` : STATE_TITLE[name] || 'Unresolved';
    if (name === 'possible_stoppage' && c.decoded[k] >= 0) {
      const d = this.codeName(c.decoded[k]);
      title += ` (play read as ${d === 'A' || d === 'B' ? teams[d] : (STATE_TITLE[d] || d).toLowerCase()})`;
    }
    const parts = [`${clockFine(c.t0[k])} – ${clockFine(c.t1[k])}`, brief(c.t1[k] - c.t0[k])];
    if (c.confidence[k] >= 0) parts.push(`confidence ${c.confidence[k]}%`);
    const p = c.possessor[k], who = p >= 0 && c.people[p] ? (c.people[p].label || c.people[p].id) : null;
    if (who) parts.push(`on the ball: ${who}`);
    if (c.ballSeen[k] >= 0) parts.push(`ball seen on ${c.ballSeen[k]}% of its frames`);
    if (LEFT_OUT.includes(name)) parts.push('left out of the shares');
    return {title, parts, name};
  }
  describeChange(ch) {
    const teams = this.teamLabels(), to = ch.to === 'A' || ch.to === 'B' ? teams[ch.to] : 'The other team';
    const parts = [], who = (id, label) => (typeof label === 'string' && label) || (typeof id === 'string' && id) || null;
    const lost = who(ch.lostBy, ch.lostByLabel), won = who(ch.wonBy, ch.wonByLabel);
    if (lost) parts.push(`${lost} lost it`);
    if (won) parts.push(`${won} won it`);
    parts.push(finite(ch.lostAt) && ch.wonAt - ch.lostAt > .05 ? `lost ${clockFine(ch.lostAt)}, won ${clockFine(ch.wonAt)}` : clockFine(ch.wonAt));
    parts.push(ch.confidence === 'high' ? 'high confidence' : 'uncertain');
    if (ch.supported === false) parts.push('no control by the winner seen after it');
    return {title: `${to} won the ball`, parts};
  }

  // ---------- the strip ----------
  drawStrip() {
    const c = this.el.strip, v = this.shown;
    if (!c || !v || !this.cols || typeof c.getContext !== 'function') return;
    const css = c.clientWidth || 0;
    if (!(css > 0)) return;   // hidden: drawn when shown (the ResizeObserver)
    const dpr = this.dpr(), W = Math.max(1, Math.round(css * dpr)), H = Math.max(1, Math.round(STRIP_H * dpr));
    if (c.width !== W) c.width = W;
    if (c.height !== H) c.height = H;
    const ctx = c.getContext('2d');
    if (!ctx) return;
    const lane = Math.round(LANE * dpr), top = lane, bh = H - lane, lo = v.lo, hi = v.hi, X = (t) => (t - lo) / (hi - lo) * W, col = this.colours();
    ctx.globalAlpha = 1;
    if (typeof ctx.setLineDash === 'function') ctx.setLineDash([]);
    ctx.clearRect(0, 0, W, H);
    // the break between two halves nobody analysed: shaded as half-time (an interval over it would be drawn on top)
    const halves = this.halves();
    for (let k = 1; k < halves.length; k++) {
      const a = Math.max(lo, halves[k - 1].t1), b = Math.min(hi, halves[k].t0);
      if (b > a) this.fillKind(ctx, KIND.HALFTIME, X(a), Math.max(1, X(b) - X(a)), top, bh, dpr, col);
    }
    // every pixel column its kind (runs of one kind are one rectangle)
    const px = pixels(this.cols, this.kinds, lo, hi, W);
    for (let start = 0, x = 1; x <= W; x++) {
      if (x < W && px[x] === px[start]) continue;
      if (px[start] >= 0) this.fillKind(ctx, px[start], start, x - start, top, bh, dpr, col);
      start = x;
    }
    // the start of each later half
    ctx.globalAlpha = 1; ctx.fillStyle = SHADE.divider;
    for (let k = 1; k < halves.length; k++) { const x = X(halves[k].t0); if (x > 0 && x < W) ctx.fillRect(Math.round(x), top, Math.max(1, Math.round(dpr)), bh); }
    // the selected interval
    const s = this.selected;
    if (s >= 0 && s < this.cols.n && this.cols.t1[s] > lo && this.cols.t0[s] < hi) {
      const a = Math.max(0, X(this.cols.t0[s])), b = Math.min(W, X(this.cols.t1[s])), w = Math.max(3 * dpr, b - a), x = Math.min(Math.max(0, a - (w - (b - a)) / 2), W - w);
      ctx.strokeStyle = SHADE.select; ctx.lineWidth = 2 * dpr;
      ctx.strokeRect(x + dpr, top + dpr, Math.max(1, w - 2 * dpr), bh - 2 * dpr);
    }
    // the changes of possession: a tick in the winner's colour, filled when 'high', hollow when 'uncertain'
    const half = 4 * dpr;
    for (const ch of this.changes) {
      if (ch.wonAt < lo || ch.wonAt > hi) continue;
      const x = X(ch.wonAt), colour = ch.to === 'A' || ch.to === 'B' ? col[ch.to] : SHADE.free;
      ctx.beginPath(); ctx.moveTo(x - half, dpr); ctx.lineTo(x + half, dpr); ctx.lineTo(x, lane - 2 * dpr); ctx.closePath();
      if (ch.confidence === 'high') { ctx.fillStyle = colour; ctx.fill(); }
      else { ctx.strokeStyle = colour; ctx.lineWidth = 1.2 * dpr; ctx.stroke(); }
    }
    ctx.globalAlpha = 1;
  }
  fillKind(ctx, k, x, w, top, h, dpr, col) {
    const fill = (colour, alpha) => { ctx.globalAlpha = alpha; ctx.fillStyle = colour; ctx.fillRect(x, top, w, h); ctx.globalAlpha = 1; };
    switch (k) {
      case KIND.A_OBS: fill(col.A, 1); break;
      case KIND.A_INF: fill(col.A, INFERRED_ALPHA); break;
      case KIND.B_OBS: fill(col.B, 1); break;
      case KIND.B_INF: fill(col.B, INFERRED_ALPHA); break;
      case KIND.CONTESTED: fill(SHADE.contested, 1); break;
      case KIND.FREE: fill(SHADE.free, 1); break;
      case KIND.STOPPAGE: fill(SHADE.stoppage, 1); break;
      case KIND.NON_MATCH: fill(SHADE.nonMatch, 1); break;
      case KIND.POSSIBLE:
        fill(SHADE.possible, 1);
        ctx.strokeStyle = SHADE.dots; ctx.lineWidth = dpr;
        if (typeof ctx.setLineDash === 'function') ctx.setLineDash([2 * dpr, 2 * dpr]);
        ctx.beginPath(); ctx.moveTo(x, top + .5 * dpr); ctx.lineTo(x + w, top + .5 * dpr); ctx.stroke();
        if (typeof ctx.setLineDash === 'function') ctx.setLineDash([]);
        break;
      case KIND.HALFTIME:
        fill(SHADE.halftime, 1);
        ctx.fillStyle = SHADE.halftimeEdge; ctx.fillRect(x, top, Math.max(1, dpr), h); ctx.fillRect(x + w - Math.max(1, dpr), top, Math.max(1, dpr), h);
        break;
      default: {   // unresolved: subtle grey, hatched
        fill(SHADE.unknown, 1);
        const hatch = this.pattern(ctx, dpr);
        if (hatch) { ctx.fillStyle = hatch; ctx.fillRect(x, top, w, h); }
      }
    }
  }
  pattern(ctx, dpr) {
    // thin diagonal lines (135 degrees), made once per pixel ratio
    if (this.hatch && this.hatch.dpr === dpr) return this.hatch.p;
    let p = null;
    if (typeof ctx.createPattern === 'function') {
      const tile = el('canvas'), n = Math.max(4, Math.round(5 * dpr));
      tile.width = n; tile.height = n;
      const t = typeof tile.getContext === 'function' ? tile.getContext('2d') : null;
      if (t) {
        t.strokeStyle = SHADE.hatch; t.lineWidth = Math.max(1, dpr);
        t.beginPath(); t.moveTo(0, n); t.lineTo(n, 0); t.moveTo(-n / 2, n / 2); t.lineTo(n / 2, -n / 2); t.moveTo(n / 2, n * 1.5); t.lineTo(n * 1.5, n / 2); t.stroke();
        p = ctx.createPattern(tile, 'repeat') || null;
      }
    }
    this.hatch = {dpr, p};
    return p;
  }

  // ---------- the strip's pointer and keys ----------
  rangeIndexes(v) { return this.cols ? [firstAfter(this.cols, v.lo), lastBefore(this.cols, v.hi)] : [0, -1]; }
  pick(t, half) {
    // the interval under the pointer: of those within half a pixel of t, the one covering most of that pixel (what the
    // pixel shows), the one holding t on a tie
    const c = this.cols;
    if (!c || !c.n) return -1;
    const k = intervalAt(c, t);
    if (!(half > 0)) return k;
    const a = t - half, b = t + half;
    let i = k >= 0 ? k : Math.min(c.n - 1, firstAfter(c, t));
    while (i > 0 && c.t1[i - 1] > a) i--;
    let best = -1, most = 0;
    for (let n = 0; i < c.n && c.t0[i] < b && n < 2000; i++, n++) {
      const w = Math.min(b, c.t1[i]) - Math.max(a, c.t0[i]);
      if (w > most + 1e-9 || (w > 0 && Math.abs(w - most) <= 1e-9 && i === k)) { most = w; best = i; }
    }
    return best;
  }
  changeNear(x, width, v) {
    // the change whose tick is nearest x (CSS px), within TICK_HIT
    const list = this.changes;
    if (!list.length) return null;
    const t = v.lo + x / width * (v.hi - v.lo), tol = TICK_HIT / width * (v.hi - v.lo);
    let lo = 0, hi = list.length;
    while (lo < hi) { const mid = (lo + hi) >> 1; if (list[mid].wonAt < t) lo = mid + 1; else hi = mid; }
    let best = null, gap = Infinity;
    for (const j of [lo - 1, lo]) {
      const ch = list[j];
      if (!ch || ch.wonAt < v.lo || ch.wonAt > v.hi) continue;
      const d = Math.abs(ch.wonAt - t);
      if (d <= tol && d < gap) { gap = d; best = ch; }
    }
    return best;
  }
  hitAt(e) {
    const v = this.shown, c = this.el.strip;
    if (!v || !this.cols || !e || !finite(e.clientX)) return null;
    const r = c.getBoundingClientRect();
    if (!r || !(r.width > 0)) return null;
    const x = Math.max(0, Math.min(r.width, e.clientX - r.left)), t = Math.min(v.hi, v.lo + x / r.width * (v.hi - v.lo));
    const y = finite(e.clientY) && finite(r.top) ? e.clientY - r.top : null;
    if (y !== null && y >= 0 && y < LANE) { const ch = this.changeNear(x, r.width, v); if (ch) return {kind: 'change', change: ch, x, width: r.width}; }
    const k = this.pick(t, (v.hi - v.lo) / r.width / 2);
    return k >= 0 ? {kind: 'interval', index: k, x, width: r.width} : null;
  }
  hover(e) {
    if (!this.active) return;
    const hit = this.hitAt(e);
    if (!hit) { this.hideTip(); return; }
    this.showTip(hit.kind === 'change' ? this.describeChange(hit.change) : this.describe(hit.index), hit.x, hit.width);
  }
  click(e) {
    const hit = this.hitAt(e);
    if (!hit) return;
    if (hit.kind === 'change') {
      // a change: the interval it was won in is selected, and playback goes to the won moment (.2 frame into it)
      const k = intervalAt(this.cols, hit.change.wonAt), at = hit.change.wonAt + .2 / this.fps();
      if (k >= 0) { this.select(k, at); return; }
      this.movedTo = at; this.seekTo(at, true); this.renderDetail();
    } else this.select(hit.index, this.seekTime(hit.index));
  }
  key(e) {
    const step = {ArrowLeft: -1, ArrowRight: 1, Home: 'home', End: 'end'}[e.key];
    if (step === undefined || !this.cols || !this.shown) return;
    e.preventDefault();
    if (typeof e.stopPropagation === 'function') e.stopPropagation();   // the page's arrow keys never also step the video
    const [first, last] = this.rangeIndexes(this.shown);
    if (first > last) return;
    let k = this.selected;
    if (step === 'home') k = first;
    else if (step === 'end') k = last;
    else if (k < first || k > last) k = step > 0 ? first : last;
    else k = Math.max(first, Math.min(last, k + step));
    this.select(k, this.seekTime(k));
    this.tipOn(k);   // focus shows what hover shows
  }
  seekTime(k) {
    const c = this.cols, step = 1 / this.fps();
    return Math.max(c.t0[k], Math.min(c.t0[k] + SEEK_INTO * step, c.t1[k] - .1 * step));
  }
  select(k, at) {
    if (!this.cols || k < 0 || k >= this.cols.n) return;
    this.selected = k;
    if (finite(at)) { this.movedTo = at; this.seekTo(at, true); }
    this.renderDetail();
    this.drawStrip();
  }
  tipOn(k) {
    const v = this.shown, r = this.el.strip.getBoundingClientRect(), w = r && r.width > 0 ? r.width : this.el.strip.clientWidth || 0;
    if (!v || !(w > 0)) return;
    const a = Math.max(v.lo, this.cols.t0[k]), b = Math.min(v.hi, this.cols.t1[k]);
    this.showTip(this.describe(k), ((a + b) / 2 - v.lo) / (v.hi - v.lo) * w, w);
  }
  showTip(d, x, width) {
    const tip = this.el.tip;
    tip.replaceChildren(el('b', null, d.title), ...d.parts.map((p) => el('span', null, p)));
    tip.style.left = `${Math.round(x)}px`;
    tip.classList.toggle('at-left', x < TIP_EDGE);
    tip.classList.toggle('at-right', x > width - TIP_EDGE && x >= TIP_EDGE);
    tip.hidden = false;
  }
  hideTip() { if (this.el && this.el.tip) this.el.tip.hidden = true; }
}

global.PossessionStats = {create: (options) => new PossessionStatsView(options), parseTime, columns, kindsOf, intervalAt, pixels, firstAfter, lastBefore,
                          clock, clockFine, span, percent, pair, KIND};
})(window);
