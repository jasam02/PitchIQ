/* PitchIQ · Possession on the 2D match view: which team has the ball, who carries it and where it changed hands. A strip
   under the board (one colour per team, as chosen on the board), a marker for every change of possession, a live line
   for the moment shown, the share of the analysed segment per team with its coverage, and a ring around the carrier on
   the board (and, subtly, an ellipse at the carrier's feet on the tracking video).

   Data. Nothing is judged here. The server computes the possession record from the saved result once per version of
   the file and keeps it until the result changes (/api/possession; soccer/possession.py judges from the 2D positions of
   the players and the ball). A correction rewrites the result, the page reloads it, and this view asks again. The
   record keys players by internal id (A-07); every id is shown as the display label of the result the page shows now,
   since labels renumber after corrections (soccer/labels.py), so a tooltip is written when it is drawn, from that
   result. Team possession persists through the team's passes, flights and loose balls until the other team controls
   the ball or the ball goes out of play (the record's convention); a run the engine could not judge is 'unknown' and
   drawn grey, never filled in here. A record that is not available says why in one plain line.

   Time. Every time is absolute source seconds, the <video> element's currentTime. The record's timeline is a list of
   runs of constant (team, state, carrier) in time order; the run at the video time is found by a binary search (runAt),
   with the last run found tried first, so a frame of playback costs a comparison or two and writes nothing to the page
   unless the live line or the playhead moved. A change marker seeks the shared playback to the frame the ball was won,
   0.2 frame into it, where the board draws the markers as observed (the coaching jumps' rule), and pauses there; on
   that frame the live line says which team won the ball and who, and the ring marks the winner (dashed), because the
   record seldom names a team on the won frame itself (its posterior is below the operating point right at a change:
   with possession-4, 5 of the 6 won frames of the broadcast run 42-222 s are in an 'unknown' run). A click on the strip
   seeks to that time and leaves playback as it was.

   Drawing. The strip is one canvas: control in the team's colour, the team's flights and loose balls fainter, an
   estimated run hatched, unknown grey, dead ball near black. The board's colour pickers repaint it. The ring is a
   layer of the board's overlays (under the markers and labels, so a label stays readable over it), solid while the
   carrier controls the ball, dashed for an estimate or another state; it is drawn only where the board draws the
   carrier's marker.

   Change confidence. possession-4 rates a change 'high' or 'uncertain'. 'uncertain' merges possession-3's 'medium' and
   'low', which were not ordered: on Metrica game 1 (where the engine was tuned), pooled over full tracking and four
   simulated cameras, 67% of the medium and 74% of the low changes matched a true change within 2 s, against 94% of
   the high ones. On the held-out Metrica game 3, pooled the same way, 87% of the high changes (489 of 564) and 70% of
   the uncertain ones (452 of 643) were right, so 'high' is 83-92% right per variant on data nobody tuned on, not 94%. A
   record made by possession-3 still says 'medium' (drawn solid) or 'low' (drawn as uncertain). Every change is shown
   by default ("Show uncertain changes" is ticked when the view is made): hiding the uncertain ones would drop 4 to 5 in
   10 of the right changes on Metrica (game 1: 325 of 769; game 3: 452 of 941). An uncertain change is a hollow
   triangle with "uncertain" in its tooltip and on its won frame's line, a high one a solid triangle. Unticked, an
   uncertain change is hidden everywhere: no marker, no "won the ball" line and no winner's ring on its won frame (the
   strip keeps the record's team on every frame, so the colour still turns there). A one-line key under the strip says
   that every change is an estimate: on the broadcast run 42-222 s (dev data) 2 of the 6 changes were false and only 2
   within 1 s; on the held-out broadcast run 222-303 s 1 of the 2 changes reported was right and 3 of the 4 real
   changes were missed.

   Fit. The panel's height is reserved under the board (index.html #viewBoard.pz-on / .pz-line), so its rows never
   wrap on a desktop width: the live line gives way first (cut with an ellipsis, the whole line in its tooltip), the
   share drops its 'Possession' label on a narrow panel, the key is one line (also cut, whole in its tooltip) and the
   one-line message of an unavailable record is held to two lines (the whole reason in its tooltip). */
(function (global) {
'use strict';

const DEFAULT_COLOURS = {A: '#b98cff', B: '#f6b564'};   // tactical.js DEFAULT_COLOURS, until the board says otherwise
const SHADE = {unknown: '#4a5650', dead: '#050807', neutral: '#8d99a6', hatch: 'rgba(5,9,8,.62)', halo: '#07131bb3'};
const STRIP_HEIGHT = 16;       // px, the strip's CSS height (#pzStrip)
const RUN_TAIL = 1.5;          // frames: a time this far past a run's last frame still belongs to it (the next run starts a frame later)
const LOST_APART = .05;        // s: a change whose loss and win are further apart than this names both times
const SEEK_INTO = .2;          // frames into the won frame a marker seeks to (the board draws observed markers there)
const WON_EARLY = .1;          // frames: the won frame's line starts this early (a wonAt not found among the frames is rounded to the ms)
const MARK_HALF = 7;           // px: half a change marker's button (index.html .pz-mark, 14 px) ...
const MARK_ICON = 10;          // ... and its triangle's width (.pz-mark i)
const RING = 1.85;             // the carrier's ring, in marker radii (the selection ring is 1.5, the held ring 1.35)
const TRAVEL_ALPHA = .58;      // the team's flights, loose and contested balls: fainter than its control
const CAUSES = {tackle_or_duel: 'tackle or duel', interception: 'interception', out_of_play: 'ball out of play', unclear: 'cause unclear'};
const CONFIDENCE = {high: 'high confidence', uncertain: 'uncertain', medium: 'medium confidence', low: 'uncertain (low confidence)'};
const UNCERTAIN = ['uncertain', 'low'];   // the change confidences drawn hollow (see Change confidence above) ...
const SHOW_UNCERTAIN = true;   // ... and shown by default: "Show uncertain changes" ticked when the view is made
const KEY_TEXT = 'Changes of possession are estimates: they can be wrong or a second or two off.';
const KEY_TITLE = 'Each triangle marks where the 2D tracking estimates that the other team won the ball, in that team’s ' +
  'colour: an estimate, not a counted turnover; read it as the boundary between attack and defence. A hollow triangle is ' +
  'an uncertain change; untick “Show uncertain changes” to hide the uncertain ones.';
const UNPLACED = ['excluded', 'withdrawn'];   // match groups of identities that are no longer players (coaching.js labelMap)

const finite = (v) => typeof v === 'number' && Number.isFinite(v);
const team = (v) => v === 'A' || v === 'B' ? v : null;

function clock(t) {
  // mm:ss.ss of absolute source time, as the page's timestamp(), rounded to the hundredth first (59.999 s is 01:00.00)
  t = Math.round(Math.max(0, t) * 100) / 100;
  return `${Math.floor(t / 60).toString().padStart(2, '0')}:${(t % 60).toFixed(2).padStart(5, '0')}`;
}
function seconds(v) {
  return v >= 90 ? `${(v / 60).toFixed(1)} min` : `${v.toFixed(1)} s`;
}
function sentence(text) {
  const s = String(text || '').trim();
  return !s ? '' : /[.!?]$/.test(s) ? s : s + '.';
}

function labelMap(result) {
  // id -> display of the result as it is now: an identity a correction excluded or the tracker withdrew shows as
  // 'excluded (was A-03)' (its bare label belongs to another player after the renumbering); a merged identity shows
  // the label of the one it was merged into. The rule of coaching.js labelMap, kept here so neither needs the other.
  const out = new Map();
  if (!result) return out;
  const m = result.match || {};
  for (const key of ['players', 'goalkeepers', 'referees']) for (const e of m[key] || []) if (e && typeof e.id === 'string') out.set(e.id, e.display || e.id);
  for (const key of UNPLACED) {
    for (const e of m[key] || []) {
      if (!e || typeof e.id !== 'string' || out.has(e.id)) continue;
      out.set(e.id, `${key} (was ${typeof e.display === 'string' && e.display && e.display !== 'OUT' ? e.display : e.id})`);
    }
  }
  for (const s of ((result.roster || {}).slots) || []) if (s && typeof s.id === 'string' && !out.has(s.id)) out.set(s.id, s.display || s.id);
  for (const e of m.retired || []) if (e && typeof e.id === 'string' && !out.has(e.id)) out.set(e.id, out.get(e.retiredInto) || e.retiredInto || e.id);
  return out;
}

/* The index of the run at time t, or -1 (before the first run, past a run's last frame with no run yet, or after the
   last one). runs: time-sorted {t0, t1}; step: one frame in seconds; hint: a run to try first (the last one found). */
function runAt(runs, t, step, hint = -1) {
  const n = runs.length, tail = RUN_TAIL * step;
  if (!n || !finite(t)) return -1;
  const holds = (k) => runs[k].t0 - step * .5 <= t && t <= runs[k].t1 + tail && (k + 1 >= n || t < runs[k + 1].t0 - step * .5);
  if (hint >= 0 && hint < n) {
    if (holds(hint)) return hint;
    if (hint + 1 < n && holds(hint + 1)) return hint + 1;
  }
  if (t < runs[0].t0 - step * .5) return -1;
  let lo = 0, hi = n - 1;
  while (lo < hi) { const mid = (lo + hi + 1) >> 1; if (runs[mid].t0 - step * .5 <= t) lo = mid; else hi = mid - 1; }
  return t <= runs[lo].t1 + tail ? lo : -1;
}

/* The tooltip of a change, display labels from `label` (id -> shown label):
   "Team B won the ball · A-07 lost it, B-05 won it · 01:23.40 · tackle or duel · medium confidence". */
function describeChange(c, label) {
  const to = team(c.toTeam), name = (id) => label(id);
  const who = [];
  if (typeof c.lostBy === 'string' && c.lostBy) who.push(`${name(c.lostBy)} lost it`);
  if (typeof c.wonBy === 'string' && c.wonBy) who.push(`${name(c.wonBy)} won it`);
  const parts = [to ? `Team ${to} won the ball` : 'Possession changed'];
  if (who.length) parts.push(who.join(', '));
  parts.push(finite(c.lostAt) && finite(c.wonAt) && c.wonAt - c.lostAt > LOST_APART ? `lost ${clock(c.lostAt)}, won ${clock(c.wonAt)}` : clock(c.wonAt));
  parts.push(CAUSES[c.cause] || CAUSES.unclear);
  parts.push(CONFIDENCE[c.confidence] || 'confidence not given');
  return parts.join(' · ');
}

/* The live line of a run (null: nothing judged at that time), and the colour key of its dot. */
function describeRun(run, spell, label) {
  if (!run) return {text: 'Possession unknown', key: 'unknown'};
  const side = team(run.team), carrier = typeof run.carrier === 'string' && run.carrier ? label(run.carrier) : null;
  const flags = (text) => text + (run.estimated ? ' · estimated' : '') + (run.confidence === 'low' ? ' · low confidence' : '');
  const held = side ? `Team ${side} in possession` : null;
  switch (run.state) {
    case 'control': return {text: flags(held ? held + (carrier ? ` · ${carrier} on the ball` : '') : carrier ? `${carrier} on the ball` : 'Ball under control'), key: side || 'neutral'};
    case 'flight': return {text: flags(held ? held + ' · ball in flight' : 'Ball in flight'), key: side || 'neutral'};
    case 'loose': return {text: flags(held ? held + ' · loose ball' : 'Loose ball'), key: side || 'neutral'};
    case 'contested': return {text: flags(held ? held + ' · contested ball' : 'Contested ball'), key: side || 'neutral'};
    case 'dead': {
      const restart = spell ? team(spell.restartTeam) : null;
      return {text: flags('Ball out of play' + (restart ? ` · Team ${restart} to restart` : '')), key: 'dead'};
    }
    // Unknown is said plainly: the engine's unknown runs are always 'low' and often 'estimated' (an interpolated ball
    // the posterior could not decide on), which would only repeat that nothing is named.
    default: return {text: 'Possession unknown', key: 'unknown'};
  }
}

/* The live line on the frame where a change says the ball was won: the engine's posterior is usually below the
   operating point right at a change (on the broadcast run 42-222 s, 5 of the 6 won frames are in an 'unknown' run), so
   the run alone would answer a click on a 'ball won' marker with 'Possession unknown'. */
function describeWon(c, label) {
  const to = team(c.toTeam), who = typeof c.wonBy === 'string' && c.wonBy ? label(c.wonBy) : null;
  return {text: (to ? `Team ${to} won the ball` : 'Possession changed') + (who ? ` · ${who} won it` : '') +
                (uncertain(c) ? ' · uncertain' : ''), key: to || 'neutral'};
}

// An uncertain change: hollow, and hidden when "Show uncertain changes" is unticked.
function uncertain(c) { return !!c && UNCERTAIN.includes(c.confidence); }

function textNode(text) { return global.document.createTextNode(text); }
function element(tag, cls, text) {
  const e = global.document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined) e.textContent = text;
  return e;
}

class PossessionView {
  constructor(options) {
    this.panel = options.panel; this.host = options.host || null; this.rows = options.rows || [];
    this.live = options.live; this.share = options.share; this.track = options.track; this.strip = options.strip;
    this.marks = options.marks; this.now = options.now; this.axis = options.axis || []; this.legend = options.legend;
    this.message = options.message; this.board = options.board || null;
    this.key = options.key || null; this.toggle = options.uncertain || null; this.count = options.uncertainCount || null;
    this.api = options.api; this.runName = options.run || (() => null); this.seekTo = options.seek || (() => {});
    this.dpr = () => Math.min(3, global.devicePixelRatio || 1);
    this.liveDot = element('i'); this.liveText = element('span');
    if (this.live) this.live.replaceChildren(this.liveDot, this.liveText);
    this.result = null; this.info = null; this.labels = new Map(); this.record = null;
    this.runs = []; this.changes = []; this.shown = []; this.won = []; this.spells = []; this.span = [0, 0]; this.step = 1 / 25;
    this.gen = 0; this.mode = null; this.cursor = -1; this.time = null; this.lastT = null; this.lastLine = null; this.lastLeft = undefined;
    for (const input of options.colourInputs || []) input.addEventListener('input', () => this.repaint());
    if (this.toggle) {
      if (SHOW_UNCERTAIN) this.toggle.checked = true;
      this.toggle.addEventListener('change', () => this.reselect());
    }
    if (this.strip) {
      this.strip.addEventListener('click', (e) => this.stripClick(e));
      this.strip.addEventListener('mousemove', (e) => this.stripHover(e));
    }
    // The ring: a layer of the board's overlays, drawn under the markers and labels.
    if (this.board && Array.isArray(this.board.overlays)) this.board.overlays.push((ctx, g, state) => this.drawBoard(ctx, g, state));
    if (typeof global.ResizeObserver === 'function' && this.track) new global.ResizeObserver(() => { this.drawStrip(); this.fitMarks(); }).observe(this.track);
    this.show('off');
  }

  label(id) { return this.labels.get(id) || id; }
  colours() { return this.board && this.board.colours ? this.board.colours : DEFAULT_COLOURS; }
  colour(key) {
    const c = this.colours();
    return key === 'A' || key === 'B' ? c[key] : key === 'dead' ? SHADE.dead : key === 'neutral' ? SHADE.neutral : SHADE.unknown;
  }
  fps() {
    const r = this.record, v = this.info && this.info.video;
    return r && finite(r.fps) && r.fps > 0 ? r.fps : v && finite(v.fps) && v.fps > 0 ? v.fps : 25;
  }
  invalidateBoard() { if (this.board && typeof this.board.invalidate === 'function') this.board.invalidate(); }

  // The panel: 'off' (no result), 'line' (one plain line: loading, unavailable, failed) or 'full'. The board view's
  // class reserves the panel's height under the board (index.html #viewBoard.pz-on / .pz-line).
  show(mode, text = '') {
    this.mode = mode;
    if (this.panel) this.panel.hidden = mode === 'off';
    for (const row of this.rows) row.hidden = mode !== 'full';
    if (this.message) {
      // Held to two lines by the page (the height it reserves); the whole text is its tooltip.
      this.message.hidden = mode !== 'line'; this.message.textContent = mode === 'line' ? text : ''; this.message.title = mode === 'line' ? text : '';
    }
    if (this.host && this.host.classList) { this.host.classList.toggle('pz-on', mode === 'full'); this.host.classList.toggle('pz-line', mode === 'line'); }
    this.lastT = null; this.lastLine = null; this.lastLeft = undefined;
  }

  // ---------- data ----------
  load(result, info) {
    // Every showResult (a new run, or the same run after a correction): the earlier record is dropped and asked again;
    // the answer of an older load that arrives late is dropped.
    const gen = ++this.gen;
    this.result = result || null; this.info = info || null; this.labels = labelMap(result);
    this.record = null; this.runs = []; this.changes = []; this.shown = []; this.won = []; this.spells = []; this.cursor = -1;
    if (this.marks) this.marks.replaceChildren();
    this.invalidateBoard();
    if (!result) { this.show('off'); return Promise.resolve(); }
    this.show('line', 'Possession: judging who has the ball from the 2D tracking…');
    const run = this.runName(), url = '/api/possession' + (run ? '?run=' + encodeURIComponent(run) : '');
    return Promise.resolve().then(() => this.api(url)).then(
      (record) => { if (gen === this.gen) this.apply(record); },
      (error) => { if (gen === this.gen) this.show('line', `Possession is not shown. ${sentence(error && error.message || error) || 'The server did not answer.'}`); });
  }

  apply(record) {
    if (!record || typeof record !== 'object') { this.show('line', 'Possession is not shown. The server sent no possession record.'); return; }
    if (record.available !== true) {
      this.show('line', `Possession is not shown for this run. ${sentence(record.reason) || 'The tracking does not support judging it.'}`);
      this.invalidateBoard();
      return;
    }
    const runs = (Array.isArray(record.timeline) ? record.timeline : [])
      .filter((r) => r && finite(r.t0) && finite(r.t1) && r.t1 >= r.t0).slice().sort((a, b) => a.t0 - b.t0);
    this.record = record; this.runs = runs; this.cursor = -1;
    this.changes = (Array.isArray(record.changes) ? record.changes : []).filter((c) => c && finite(c.wonAt)).slice().sort((a, b) => a.wonAt - b.wonAt);
    this.spells = (Array.isArray(record.deadSpells) ? record.deadSpells : []).filter((s) => s && finite(s.t0) && finite(s.t1)).slice().sort((a, b) => a.t0 - b.t0);
    this.step = 1 / this.fps();
    const t0 = finite(record.t0) ? record.t0 : runs.length ? runs[0].t0 : 0;
    const t1 = finite(record.t1) ? record.t1 : runs.length ? runs[runs.length - 1].t1 : t0;
    this.span = [t0, Math.max(t0 + this.step, t1 + this.step)];   // the last frame has a frame's width
    this.select();
    this.show('full');
    this.renderShares(); this.renderLegend(); this.renderKey(); this.renderMarks(); this.renderAxis(); this.drawStrip();
    this.invalidateBoard();
  }

  // ---------- which changes are shown ----------
  showUncertain() { return !!(this.toggle && this.toggle.checked); }
  select() {
    // The changes shown (uncertain ones only when ticked) and their won frames, which the live line and the rings read.
    const all = this.showUncertain();
    this.shown = this.changes.filter((c) => all || !uncertain(c));
    this.won = this.shown.map((c) => { const at = this.frameTime(c); return {c, at: at === null ? c.wonAt : at}; }).sort((a, b) => a.at - b.at);
  }
  reselect() {
    // "Show uncertain changes" ticked or cleared: the markers, the live line and the ring follow on the next frame.
    if (this.mode !== 'full') return;
    this.select(); this.renderMarks();
    this.lastT = null; this.lastLine = null;
    if (this.time !== null) this.tick(this.time);
    this.invalidateBoard();
  }

  // ---------- lookups ----------
  indexAt(t) {
    const k = runAt(this.runs, t, this.step, this.cursor);
    if (k >= 0) this.cursor = k;
    return k;
  }
  runAtTime(t) {
    if (this.mode !== 'full') return null;
    const k = this.indexAt(t);
    return k >= 0 ? this.runs[k] : null;
  }
  spellAt(t) {
    const s = this.spells;
    let lo = 0, hi = s.length - 1, found = null;
    while (lo <= hi) { const mid = (lo + hi) >> 1; if (s[mid].t0 - this.step * .5 <= t) { found = s[mid]; lo = mid + 1; } else hi = mid - 1; }
    return found && t <= found.t1 + RUN_TAIL * this.step ? found : null;
  }
  changeAt(t) {
    // The change whose won frame is shown at time t (the page shows the last frame at or before t), or null.
    const w = this.won;
    let lo = 0, hi = w.length - 1, found = null;
    while (lo <= hi) { const mid = (lo + hi) >> 1; if (w[mid].at - this.step * WON_EARLY <= t) { found = w[mid]; lo = mid + 1; } else hi = mid - 1; }
    return found && t < found.at + this.step ? found.c : null;
  }
  holder(t) {
    // Whom the rings mark at time t: the run's carrier, else, on the frame a change says the ball was won, its winner
    // (dashed: the record does not name him in control there).
    const run = this.runAtTime(t);
    if (run && typeof run.carrier === 'string' && run.carrier) return {id: run.carrier, team: team(run.team), dashed: run.estimated || run.state !== 'control'};
    const c = run && run.state === 'dead' ? null : this.changeAt(t);
    return c && typeof c.wonBy === 'string' && c.wonBy ? {id: c.wonBy, team: team(c.toTeam), dashed: true} : null;
  }
  inside(t) { return t >= this.span[0] - this.step * .5 && t <= this.span[1]; }
  line(t, run) {
    if (!this.inside(t)) return {text: 'Outside the analysed segment', key: 'unknown'};
    const won = run && run.state === 'dead' ? null : this.changeAt(t);
    if (won) return describeWon(won, (id) => this.label(id));
    return describeRun(run, run && run.state === 'dead' ? this.spellAt(t) : null, (id) => this.label(id));
  }
  pct(t) { return 100 * (t - this.span[0]) / Math.max(1e-9, this.span[1] - this.span[0]); }

  // ---------- every frame ----------
  tick(t) {
    if (this.mode !== 'full' || t === this.lastT) return;
    this.lastT = t; this.time = t;
    const run = this.runAtTime(t), line = this.line(t, run), key = line.text + '|' + line.key;
    if (key !== this.lastLine) {
      this.lastLine = key;
      this.liveText.textContent = line.text;
      if (this.live) this.live.title = line.text;   // the whole line, where a narrow panel cuts it
      this.liveDot.style.background = this.colour(line.key);
    }
    const left = this.inside(t) ? Math.max(0, Math.min(100, this.pct(t))).toFixed(2) : null;
    if (left !== this.lastLeft) {
      this.lastLeft = left;
      if (this.now) { this.now.hidden = left === null; if (left !== null) this.now.style.left = left + '%'; }
    }
  }

  // ---------- the panel ----------
  renderShares() {
    if (!this.share) return;
    const r = this.record, s = r.shares || {}, q = r.quality || {}, num = (v) => finite(v) && v > 0 ? v : 0;
    const a = num(s.A), b = num(s.B), named = a + b, nodes = [element('span', 'pz-k', 'Possession')];
    if (named > 0) {
      const pa = Math.round(100 * a / named);
      for (const [side, value] of [['A', pa], ['B', 100 - pa]]) {
        const item = element('span'), dot = element('i');
        dot.style.background = this.colour(side);
        item.append(dot, textNode(`Team ${side} `), element('b', '', `${value}%`));
        nodes.push(item);
      }
    } else nodes.push(element('span', '', 'no team named'));
    const coverage = finite(q.coverage) ? Math.round(100 * Math.max(0, Math.min(1, q.coverage))) : null;
    nodes.push(element('span', '', coverage === null ? 'coverage not given' : `coverage ${coverage}%`));
    if (q.level === 'limited') nodes.push(element('span', 'pz-tag', 'limited'));
    this.share.replaceChildren(...nodes);
    const notes = (Array.isArray(q.notes) ? q.notes : []).filter((n) => typeof n === 'string' && n).map(sentence);
    this.share.title = `Over the analysed segment (${clock(this.span[0])}–${clock(finite(r.t1) ? r.t1 : this.span[1])}): ` +
      `Team A had the ball ${seconds(a)}, Team B ${seconds(b)} (the shares are of that time); ball out of play ${seconds(num(s.dead))}; ` +
      `unknown ${seconds(num(s.unknown))}.` + (coverage === null ? '' : ` A team is named on ${coverage}% of the frames.`) +
      (q.level === 'limited' ? ' Limited quality.' : '') + (notes.length ? ' ' + notes.join(' ') : '');
  }

  renderLegend() {
    if (!this.legend) return;
    const item = (background, text) => { const s = element('span'), i = element('i'); i.style.background = background; s.append(i, textNode(text)); return s; };
    const c = this.colours();
    this.legend.replaceChildren(item(c.A, 'Team A'), item(c.B, 'Team B'),
      item(`repeating-linear-gradient(135deg,${SHADE.neutral} 0 2px,${SHADE.dead} 2px 4px)`, 'estimated'),
      item(SHADE.unknown, 'unknown'), item(SHADE.dead, 'out of play'));
    this.legend.title = 'Each team in its board colour: solid while a player controls the ball, fainter while its pass or clearance travels ' +
      'or the ball is loose; hatched where the record estimates; grey where it cannot say; dark while the ball is out of play.';
  }

  renderKey() {
    // The markers' key, one line: a change is an estimate; a hollow triangle is uncertain.
    if (!this.key) return;
    this.key.replaceChildren(element('i', 'pz-tri'), textNode(KEY_TEXT), element('i', 'pz-tri uncertain'), textNode('uncertain'));
    this.key.title = KEY_TITLE;
  }

  renderAxis() {
    const [a, b] = this.axis;
    if (a) a.textContent = clock(this.span[0]);
    if (b) b.textContent = clock(this.record && finite(this.record.t1) ? this.record.t1 : this.span[1]);
  }

  renderMarks() {
    if (!this.marks) return;
    const nodes = [], hidden = this.changes.length - this.shown.length, doubtful = this.changes.filter(uncertain).length;
    if (this.toggle) this.toggle.disabled = !doubtful;
    if (this.count) {
      this.count.textContent = `(${doubtful})`;
      this.count.title = doubtful ? `${doubtful} uncertain change${doubtful === 1 ? '' : 's'} of ${this.changes.length}` : 'No uncertain change in this segment';
    }
    this.marks.setAttribute('aria-label', `Changes of possession: ${this.shown.length} shown` + (hidden ? `, ${hidden} uncertain hidden` : ''));
    for (const c of this.shown) {
      const text = describeChange(c, (id) => this.label(id)), b = element('button', 'pz-mark' + (uncertain(c) ? ' uncertain' : '')), i = element('i');
      b.type = 'button';
      b.style.left = Math.max(0, Math.min(100, this.pct(c.wonAt))).toFixed(2) + '%';
      b.title = text;
      b.setAttribute('aria-label', text);
      if (typeof c.id === 'string') b.dataset.change = c.id;
      i.style.background = this.colour(team(c.toTeam) || 'neutral');
      b.append(i);
      b.addEventListener('click', () => this.jump(c));
      nodes.push(b);
    }
    this.marks.replaceChildren(...nodes);
    this.fitMarks();
  }
  fitMarks() {
    // Markers closer than a marker's width share the space between them: each one's button reaches halfway to its
    // neighbour, its triangle stays on its own time. Otherwise the later of a close pair covers the earlier's centre
    // (on the broadcast run 42-222 s the changes at 139.4 and 140.53 s are 4 px apart on a 640 px strip, which would
    // leave the earlier one clickable on 4 of its 14 px; possession-2's pairs there were 3-4 px apart).
    const w = this.marks && this.marks.clientWidth, nodes = this.marks ? this.marks.children : [];
    if (!(w > 0) || !nodes.length) return;
    const xs = this.shown.map((c) => Math.max(0, Math.min(100, this.pct(c.wonAt))) / 100 * w);
    for (let k = 0; k < nodes.length && k < xs.length; k++) {
      const left = k > 0 ? Math.min(MARK_HALF, (xs[k] - xs[k - 1]) / 2) : MARK_HALF;
      const right = k + 1 < xs.length ? Math.min(MARK_HALF, (xs[k + 1] - xs[k]) / 2) : MARK_HALF;
      const b = nodes[k], i = b.children && b.children[0];
      b.style.marginLeft = `${(-left).toFixed(1)}px`; b.style.width = `${(left + right).toFixed(1)}px`;
      if (i) i.style.left = `${(left - MARK_ICON / 2).toFixed(1)}px`;
    }
  }

  // The strip: one canvas, every run from its first frame to the next run (or one frame past its last frame).
  drawStrip() {
    const c = this.strip;
    if (!c || this.mode !== 'full' || typeof c.getContext !== 'function') return;
    const dpr = this.dpr(), w = Math.max(1, Math.round((c.clientWidth || 600) * dpr)), h = Math.max(1, Math.round(STRIP_HEIGHT * dpr));
    if (c.width !== w) c.width = w;
    if (c.height !== h) c.height = h;
    const ctx = c.getContext('2d'), span = this.span, x = (t) => (t - span[0]) / Math.max(1e-9, span[1] - span[0]) * w;
    ctx.globalAlpha = 1;
    ctx.fillStyle = SHADE.unknown; ctx.fillRect(0, 0, w, h);   // nothing judged: unknown
    let hatch = null;
    for (let k = 0; k < this.runs.length; k++) {
      const r = this.runs[k], next = this.runs[k + 1], end = next ? Math.min(next.t0, r.t1 + this.step) : r.t1 + this.step;
      const a = Math.max(0, x(r.t0)), b = Math.min(w, x(Math.max(end, r.t0 + this.step)));
      if (b <= a) continue;
      const side = team(r.team);
      let fill = SHADE.unknown, alpha = 1;
      if (r.state === 'dead') fill = SHADE.dead;
      else if (side) { fill = this.colour(side); alpha = r.state === 'control' ? 1 : TRAVEL_ALPHA; }
      else if (r.state === 'loose' || r.state === 'contested' || r.state === 'flight' || r.state === 'control') { fill = SHADE.neutral; alpha = .45; }
      if (alpha < 1) { ctx.globalAlpha = 1; ctx.fillStyle = SHADE.unknown; ctx.fillRect(a, 0, b - a, h); }
      ctx.globalAlpha = alpha; ctx.fillStyle = fill; ctx.fillRect(a, 0, b - a, h);
      if (r.estimated && r.state !== 'dead' && r.state !== 'unknown') {
        if (hatch === null) hatch = this.pattern(ctx, dpr);
        ctx.globalAlpha = 1; ctx.fillStyle = hatch || SHADE.hatch; ctx.fillRect(a, 0, b - a, h);
      }
    }
    ctx.globalAlpha = 1;
  }
  pattern(ctx, dpr) {
    // Diagonal dark lines over the run's colour: the muted mark of an estimate.
    if (typeof ctx.createPattern !== 'function') return null;
    const tile = element('canvas'), n = Math.max(4, Math.round(6 * dpr));
    tile.width = n; tile.height = n;
    const t = tile.getContext && tile.getContext('2d');
    if (!t) return null;
    t.strokeStyle = SHADE.hatch; t.lineWidth = Math.max(1, dpr * 1.4);
    t.beginPath(); t.moveTo(0, n); t.lineTo(n, 0); t.moveTo(-n / 2, n / 2); t.lineTo(n / 2, -n / 2); t.moveTo(n / 2, n * 1.5); t.lineTo(n * 1.5, n / 2); t.stroke();
    return ctx.createPattern(tile, 'repeat') || null;
  }

  repaint() {
    // The board's colours changed: the strip, the markers, the shares, the legend and the live dot follow.
    if (this.mode !== 'full') return;
    this.drawStrip(); this.renderMarks(); this.renderShares(); this.renderLegend();
    this.lastLine = null; this.lastT = null;
  }

  // ---------- seeking ----------
  frameTime(c) {
    // The time of the frame the ball was won, found by its absolute index in the result; null when it is not there.
    const frames = this.result && this.result.frames;
    if (Number.isInteger(c.frame) && Array.isArray(frames) && frames.length) {
      let lo = 0, hi = frames.length - 1;
      while (lo < hi) { const mid = (lo + hi) >> 1; if (frames[mid].frame < c.frame) lo = mid + 1; else hi = mid; }
      if (frames[lo].frame === c.frame && finite(frames[lo].time)) return frames[lo].time;
    }
    return null;
  }
  changeTime(c) {
    // 0.2 frame into the frame the ball was won, else the record's time.
    const at = this.frameTime(c);
    return at === null ? c.wonAt : at + SEEK_INTO / this.fps();
  }
  jump(c) { this.seekTo(this.changeTime(c), true); }
  timeAtX(e) {
    const c = this.strip, r = c && typeof c.getBoundingClientRect === 'function' ? c.getBoundingClientRect() : null;
    if (!r || !(r.width > 0) || !finite(e.clientX)) return null;
    const u = Math.max(0, Math.min(1, (e.clientX - r.left) / r.width));
    return Math.min(this.span[1] - this.step * .5, this.span[0] + u * (this.span[1] - this.span[0]));
  }
  stripClick(e) {
    if (this.mode !== 'full') return;
    const t = this.timeAtX(e);
    if (t !== null) this.seekTo(t, false);
  }
  stripHover(e) {
    if (this.mode !== 'full') return;
    const t = this.timeAtX(e);
    if (t === null) return;
    const k = runAt(this.runs, t, this.step);
    this.strip.title = `${clock(t)} · ${this.line(t, k >= 0 ? this.runs[k] : null).text}`;
  }

  // ---------- rings ----------
  drawBoard(ctx, g, state) {
    // On the board, around the carrier's marker, only where the board draws it.
    const who = state ? this.holder(state.time) : null;
    if (!who) return;
    const p = (state.players || []).find((q) => q.id === who.id);
    if (!p || (this.board && typeof this.board.visible === 'function' && !this.board.visible(p))) return;
    const [x, y] = g.toPx(p.x, p.y), r = g.playerRadius * RING, dashed = who.dashed;
    ctx.save();
    ctx.globalAlpha = finite(p.alpha) ? Math.max(.35, p.alpha) : 1;
    ctx.lineWidth = Math.max(2.5, g.playerRadius * .36); ctx.strokeStyle = SHADE.halo; ctx.setLineDash([]);
    ctx.beginPath(); ctx.arc(x, y, r, 0, Math.PI * 2); ctx.stroke();
    ctx.lineWidth = Math.max(1.5, g.playerRadius * .2); ctx.strokeStyle = this.colour(who.team || team(p.team) || 'neutral');
    if (dashed) ctx.setLineDash([g.playerRadius * .55, g.playerRadius * .4]);
    ctx.beginPath(); ctx.arc(x, y, r, 0, Math.PI * 2); ctx.stroke();
    ctx.restore();
  }
  drawVideo(ctx, frame, t, width, height, colours) {
    // On the tracking video, a thin ellipse at the carrier's feet (the bottom of their box on this frame), in the
    // boxes' colours; nothing when the carrier is not detected on this frame.
    const who = frame ? this.holder(t) : null;
    if (!who) return;
    const p = (frame.people || []).find((q) => q.id === who.id);
    if (!p || !Array.isArray(p.box) || p.box.length !== 4 || !p.box.every(finite)) return;
    const [bx, by, bw, bh] = p.box, rx = Math.max(10, bw * width * .85), ry = Math.max(4, rx * .3);
    const side = who.team || team(p.team), palette = colours || DEFAULT_COLOURS;
    ctx.save();
    ctx.globalAlpha = .85; ctx.lineWidth = Math.max(2, width / 640);
    ctx.strokeStyle = side && palette[side] ? palette[side] : '#ffffff';
    ctx.setLineDash(who.dashed ? [8, 6] : []);
    ctx.beginPath(); ctx.ellipse((bx + bw / 2) * width, (by + bh) * height, rx, ry, 0, 0, Math.PI * 2); ctx.stroke();
    ctx.restore();
  }
}

global.Possession = {create: (options) => new PossessionView(options), labelMap, runAt, describeChange, describeRun, describeWon, uncertain, clock};
})(window);
