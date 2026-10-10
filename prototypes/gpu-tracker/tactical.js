/* PitchIQ · 2D match view: a top-down recreation of the match from the saved tracking result.

   Data. The pipeline's result.json. Every observed person with a global identity carries `pitch`: the
   normalized pitch position of the feet (x 0..1 from the left goal line to the right goal line, y 0..1
   from the far touchline to the near touchline, on a 105 x 68 m pitch), computed by soccer/calibration.py
   from the user's landmark calibration and carried through pans and zooms by the camera-motion estimate.
   The ball carries `pitch` the same way. Nothing here is derived from screen coordinates: without a
   calibration there are no positions, and none are invented.

   Structure, so that analytics overlays (trails, heat maps, formations, lanes) can be added later:
     Timeline  built once per result from the finalized frames and the final registry (teams and roles
               as the identity manager settled them, retroactive relabelling included): per identity,
               time-sorted samples with a centred smoothing window; the ball likewise.
     Sampler   positions at any video time: the observation of that frame, a linear interpolation across
               a short gap, or the last position held briefly (fading) before the marker hides.
     Renderer  canvas layers in order: pitch (cached), overlays (hooks), players, labels, ball, debug.
   The <video> element is the only clock: the page calls view.render(video.currentTime) on every animation
   frame, and the view redraws only when the time or a setting changed. */
(function (global) {
'use strict';

const PITCH = {length: 105, width: 68, penaltyDepth: 16.5, penaltyHalf: 20.16, goalDepth: 5.5, goalHalf: 9.16,
               spot: 11, circle: 9.15, goalWidth: 7.32, goalNet: 2.2, cornerArc: 1.0, margin: 4.5};
const RULES = {
  interpolateGap: 1.0,     // s: an identity observed on both sides of a gap this long is interpolated across it
  holdAfter: 0.6,          // s: after its last observation a marker stays (fading) this long, then hides
  ballHold: 0.2,           // s: the ball tracker already predicts while hidden; hold only a little longer
  ballInterpolateGap: 0.15, // s: a lost ball is never bridged by a straight line; only a dropped frame or two is
  smoothWindow: 0.14,      // s: centred averaging window over a player's observations (offline, no lag)
  ballSmoothWindow: 0.06,
  ballPredictedMin: 0.5,   // ball confidence needed to draw a predicted (hidden) ball
  outside: 0.1,            // normalized: how far beyond the lines a position may still be drawn
};
const DEFAULT_COLOURS = {A: '#b98cff', B: '#f6b564'};
const NEUTRAL = '#8d99a6';   // a goalkeeper whose team is not known yet

// ---------- timeline ----------
class Track {
  constructor(info) {
    this.info = info;
    this.t = []; this.x = []; this.y = []; this.c = []; this.predicted = [];
  }
  push(t, x, y, confidence, predicted) {
    if (this.t.length && t <= this.t[this.t.length - 1]) return;   // frames arrive in time order
    this.t.push(t); this.x.push(x); this.y.push(y); this.c.push(confidence); this.predicted.push(!!predicted);
  }
  smooth(window) {
    // Centred average of the samples within ±window/2: the whole history is known, so no lag.
    const n = this.t.length, xs = new Float64Array(n), ys = new Float64Array(n);
    let lo = 0, hi = 0;
    for (let i = 0; i < n; i++) {
      while (this.t[i] - this.t[lo] > window / 2) lo++;
      if (hi < i) hi = i;
      while (hi + 1 < n && this.t[hi + 1] - this.t[i] <= window / 2) hi++;
      let sx = 0, sy = 0;
      for (let j = lo; j <= hi; j++) { sx += this.x[j]; sy += this.y[j]; }
      xs[i] = sx / (hi - lo + 1); ys[i] = sy / (hi - lo + 1);
    }
    this.x = xs; this.y = ys; this.t = Float64Array.from(this.t); this.c = Float64Array.from(this.c);
    return this;
  }
}

function validPosition(p) {
  return Array.isArray(p) && p.length === 2 && Number.isFinite(p[0]) && Number.isFinite(p[1]) &&
    p[0] >= -RULES.outside && p[0] <= 1 + RULES.outside && p[1] >= -RULES.outside && p[1] <= 1 + RULES.outside;
}

/* The final registry decides who is drawn: team players and goalkeepers. Referees, merged (retired) and
   unknown people are left out, and an identity's team and role are the settled ones, not the ones a frame
   may have carried before a correction. */
function finalRoster(result) {
  const roster = new Map();
  for (const p of result.match.players || []) if (p.team === 'A' || p.team === 'B') roster.set(p.id, {id: p.id, display: p.display || p.id, team: p.team, role: 'PLAYER'});
  for (const p of result.match.goalkeepers || []) roster.set(p.id, {id: p.id, display: p.display || p.id, team: p.team || null, role: 'GOALKEEPER'});
  return roster;
}

function buildTimeline(result, info) {
  const roster = finalRoster(result), tracks = new Map(), ball = new Track({id: 'BALL'});
  const frames = result.frames || [], frameTimes = new Float64Array(frames.length), calibrated = new Uint8Array(frames.length);
  let observed = 0, ballFrames = 0;
  frames.forEach((f, i) => {
    frameTimes[i] = f.time;
    calibrated[i] = f.calibrated ? 1 : 0;
    for (const p of f.people || []) {
      if (!p.id || !roster.has(p.id) || !validPosition(p.pitch)) continue;
      let track = tracks.get(p.id);
      if (!track) tracks.set(p.id, track = new Track(roster.get(p.id)));
      track.push(f.time, p.pitch[0], p.pitch[1], Number.isFinite(p.identityConfidence) ? p.identityConfidence : .5, false);
      observed++;
    }
    const b = f.ball;
    if (b && validPosition(b.pitch) && (b.state === 'TRACKED' || (b.state === 'MISSING' && b.confidence >= RULES.ballPredictedMin))) {
      ball.push(f.time, b.pitch[0], b.pitch[1], b.confidence, b.state !== 'TRACKED');
      ballFrames++;
    }
  });
  for (const track of tracks.values()) track.smooth(RULES.smoothWindow);
  ball.smooth(RULES.ballSmoothWindow);
  let step = info && info.video && info.video.fps > 0 ? 1 / info.video.fps : 0;
  if (!step && frames.length > 1) step = (frames[frames.length - 1].time - frames[0].time) / (frames.length - 1);
  const calibratedFrames = calibrated.reduce((a, v) => a + v, 0);
  return {tracks, ball, frameTimes, calibrated, frameStep: step || .04, observed, ballFrames, calibratedFrames, frames: frames.length,
          goalEnds: (result.tactical && result.tactical.goalEnds) || {}};
}

// ---------- sampler ----------
/* Position of one track at time t: OBSERVED (that frame's observation), INTERPOLATED (between two
   observations at most interpolateGap apart), HELD (the last observation, fading, for holdAfter seconds)
   or null (unknown: the marker hides). The ball's predicted samples report PREDICTED. */
function sampleAt(track, t, frameStep, hold = RULES.holdAfter, gapLimit = RULES.interpolateGap) {
  const n = track.t.length, same = frameStep * .25;   // this close to an observation's time, it is that observation
  if (!n || t < track.t[0] - same) return null;
  let lo = 0, hi = n - 1;
  while (lo < hi) { const mid = (lo + hi + 1) >> 1; if (track.t[mid] <= t) lo = mid; else hi = mid - 1; }
  const k = lo, t0 = track.t[k], dt = t - t0;
  const observed = (i) => ({x: track.x[i], y: track.y[i], confidence: track.c[i], source: track.predicted[i] ? 'PREDICTED' : 'OBSERVED', alpha: 1});
  if (dt < 0 || dt <= same) return observed(dt < 0 ? 0 : k);
  if (k + 1 < n) {
    const t1 = track.t[k + 1], gap = t1 - t0;
    if (t1 - t <= same) return observed(k + 1);
    if (gap <= gapLimit) {
      const u = dt / gap;
      return {x: track.x[k] + (track.x[k + 1] - track.x[k]) * u, y: track.y[k] + (track.y[k + 1] - track.y[k]) * u,
              confidence: Math.min(track.c[k], track.c[k + 1]), source: track.predicted[k] || track.predicted[k + 1] ? 'PREDICTED' : 'INTERPOLATED', alpha: 1};
    }
  }
  if (dt <= hold) return {x: track.x[k], y: track.y[k], confidence: track.c[k] * (1 - dt / hold), source: 'HELD', alpha: 1 - .7 * dt / hold};
  return null;
}

function frameAt(timeline, t) {
  const times = timeline.frameTimes, n = times.length;
  if (!n) return -1;
  let lo = 0, hi = n - 1;
  while (lo < hi) { const mid = (lo + hi + 1) >> 1; if (times[mid] <= t) lo = mid; else hi = mid - 1; }
  return Math.abs(times[lo] - t) <= 1.6 * timeline.frameStep ? lo : -1;
}

// ---------- renderer ----------
function luminance(hex) {
  const v = parseInt(hex.slice(1), 16);
  return (.299 * (v >> 16 & 255) + .587 * (v >> 8 & 255) + .114 * (v & 255)) / 255;
}

class TacticalView {
  constructor(options) {
    this.canvas = options.canvas;
    this.ctx = this.canvas.getContext('2d');
    this.controls = options.controls || {};
    this.notice = options.notice || null;
    this.status = options.status || null;
    this.timeline = null;
    this.overlays = [];        // (ctx, geometry, state) => void, drawn between the pitch and the players
    this.pitchLayer = null;
    this.lastKey = null;
    this.last = {time: 0, players: [], ball: null};
    this.storageKey = null;
    this.colours = {...DEFAULT_COLOURS};
    const redraw = () => { this.lastKey = null; };
    for (const key of ['colourA', 'colourB']) {
      const input = this.controls[key];
      if (!input) continue;
      input.value = this.colours[key === 'colourA' ? 'A' : 'B'];
      input.addEventListener('input', () => { this.colours[key === 'colourA' ? 'A' : 'B'] = input.value; this.persist(); redraw(); });
    }
    for (const key of ['labels', 'debug']) if (this.controls[key]) this.controls[key].addEventListener('change', () => { this.persist(); redraw(); });
    if (typeof ResizeObserver === 'function') new ResizeObserver(() => this.resize()).observe(this.canvas);
    this.resize();
  }

  // Colour choices are the viewer's (the tracking's team A / B never changes); kept per video.
  persist() {
    if (!this.storageKey) return;
    try {
      localStorage.setItem(this.storageKey, JSON.stringify({A: this.colours.A, B: this.colours.B, labels: this.showLabels()}));
    } catch (e) { /* storage unavailable: colours last for the page */ }
  }
  restore(key) {
    this.storageKey = key;
    try {
      const saved = JSON.parse(localStorage.getItem(key) || 'null');
      if (saved && /^#[0-9a-f]{6}$/i.test(saved.A || '') && /^#[0-9a-f]{6}$/i.test(saved.B || '')) {
        this.colours = {A: saved.A, B: saved.B};
        if (this.controls.colourA) this.controls.colourA.value = saved.A;
        if (this.controls.colourB) this.controls.colourB.value = saved.B;
        if (this.controls.labels && typeof saved.labels === 'boolean') this.controls.labels.checked = saved.labels;
      }
    } catch (e) { /* ignore */ }
  }
  showLabels() { return this.controls.labels ? this.controls.labels.checked : true; }
  showDebug() { return this.controls.debug ? this.controls.debug.checked : false; }

  load(result, info) {
    this.timeline = result && result.frames ? buildTimeline(result, info) : null;
    if (info && info.videoKey) this.restore('pitchiq.tactical.' + info.videoKey);
    this.pitchLayer = null;   // the board names the goal ends once the result says who defends them
    this.lastKey = null;
    if (this.status) {
      const tl = this.timeline;
      this.status.textContent = !tl ? '' : !tl.calibratedFrames ? 'No pitch coordinates in this result.' :
        `Pitch calibrated on ${Math.round(100 * tl.calibratedFrames / Math.max(1, tl.frames))}% of frames · ${tl.tracks.size} players with positions · ball placed on ${Math.round(100 * tl.ballFrames / Math.max(1, tl.frames))}% of frames`;
    }
  }

  resize() {
    const width = this.canvas.clientWidth || 960;
    const ratio = (PITCH.length + 2 * PITCH.margin) / (PITCH.width + 2 * PITCH.margin);
    const dpr = Math.min(3, global.devicePixelRatio || 1);
    const w = Math.max(200, Math.round(width * dpr)), h = Math.round(w / ratio);
    if (this.canvas.width !== w || this.canvas.height !== h) { this.canvas.width = w; this.canvas.height = h; }
    this.pitchLayer = null;
    this.lastKey = null;
  }

  geometry() {
    const scale = this.canvas.width / (PITCH.length + 2 * PITCH.margin);
    const ox = PITCH.margin * scale, oy = PITCH.margin * scale;
    return {scale, ox, oy, width: this.canvas.width, height: this.canvas.height,
            metres: (x, y) => [ox + x * scale, oy + y * scale],
            toPx: (nx, ny) => [ox + nx * PITCH.length * scale, oy + ny * PITCH.width * scale],
            playerRadius: Math.max(5, Math.min(13, PITCH.length * scale * .011))};
  }

  // Layer 1 + 2: the board, drawn once per size and cached.
  drawPitch(g) {
    const layer = document.createElement('canvas');
    layer.width = this.canvas.width; layer.height = this.canvas.height;
    const c = layer.getContext('2d'), m = g.metres, L = PITCH.length, W = PITCH.width;
    c.fillStyle = '#1e6b36'; c.fillRect(0, 0, layer.width, layer.height);
    const stripes = 10, [fx, fy] = m(0, 0), [tx, ty] = m(L, W);
    for (let i = 0; i < stripes; i++) {
      c.fillStyle = i % 2 ? '#2b8a46' : '#30944d';
      const x0 = fx + (tx - fx) * i / stripes, x1 = fx + (tx - fx) * (i + 1) / stripes;
      c.fillRect(x0, fy, x1 - x0 + .5, ty - fy);
    }
    c.strokeStyle = '#f4f7f4'; c.lineWidth = Math.max(1, g.scale * .13); c.lineCap = 'round'; c.lineJoin = 'round';
    const rect = (x, y, w, h) => { const [a, b] = m(x, y); c.strokeRect(a, b, w * g.scale, h * g.scale); };
    const line = (x0, y0, x1, y1) => { const [a, b] = m(x0, y0), [d, e] = m(x1, y1); c.beginPath(); c.moveTo(a, b); c.lineTo(d, e); c.stroke(); };
    const arc = (x, y, r, from, to) => { const [a, b] = m(x, y); c.beginPath(); c.arc(a, b, r * g.scale, from, to); c.stroke(); };
    const dot = (x, y) => { const [a, b] = m(x, y); c.fillStyle = '#f4f7f4'; c.beginPath(); c.arc(a, b, Math.max(1.5, g.scale * .25), 0, Math.PI * 2); c.fill(); };
    rect(0, 0, L, W);
    line(L / 2, 0, L / 2, W);
    arc(L / 2, W / 2, PITCH.circle, 0, Math.PI * 2);
    dot(L / 2, W / 2);
    for (const [x0, sign] of [[0, 1], [L, -1]]) {
      rect(sign > 0 ? 0 : L - PITCH.penaltyDepth, W / 2 - PITCH.penaltyHalf, PITCH.penaltyDepth, 2 * PITCH.penaltyHalf);
      rect(sign > 0 ? 0 : L - PITCH.goalDepth, W / 2 - PITCH.goalHalf, PITCH.goalDepth, 2 * PITCH.goalHalf);
      dot(x0 + sign * PITCH.spot, W / 2);
      const spread = Math.acos((PITCH.penaltyDepth - PITCH.spot) / PITCH.circle);   // the arc outside the penalty area
      arc(x0 + sign * PITCH.spot, W / 2, PITCH.circle, sign > 0 ? -spread : Math.PI - spread, sign > 0 ? spread : Math.PI + spread);
      // Goal: a net box behind the goal line.
      const [gx, gy] = m(sign > 0 ? -PITCH.goalNet : L, W / 2 - PITCH.goalWidth / 2);
      c.fillStyle = '#ffffff22'; c.fillRect(gx, gy, PITCH.goalNet * g.scale, PITCH.goalWidth * g.scale);
      c.strokeRect(gx, gy, PITCH.goalNet * g.scale, PITCH.goalWidth * g.scale);
    }
    for (const [x, y, from] of [[0, 0, 0], [L, 0, Math.PI / 2], [L, W, Math.PI], [0, W, -Math.PI / 2]]) arc(x, y, PITCH.cornerArc, from, from + Math.PI / 2);
    // Which team defends which end, when the tracking settled it.
    const ends = this.timeline ? this.timeline.goalEnds : {};
    c.font = `${Math.max(10, g.scale * 1.9)}px system-ui, sans-serif`; c.fillStyle = '#d9e3dc'; c.textAlign = 'center';
    if (ends.left) { const [a, b] = m(0, W + PITCH.margin * .55); c.fillText(`Team ${ends.left} goal`, a + 10 * g.scale, b); }
    if (ends.right) { const [a, b] = m(L, W + PITCH.margin * .55); c.fillText(`Team ${ends.right} goal`, a - 10 * g.scale, b); }
    return layer;
  }

  // Positions of everyone at time t, from the timeline (nothing is drawn without a pitch position).
  positions(t) {
    const tl = this.timeline, players = [];
    if (!tl) return {time: t, frame: -1, calibrated: false, players, ball: null};
    for (const track of tl.tracks.values()) {
      const s = sampleAt(track, t, tl.frameStep);
      if (s) players.push({...track.info, ...s});
    }
    players.sort((a, b) => a.y - b.y || (a.id < b.id ? -1 : 1));   // far players first, so near ones draw on top
    const ball = sampleAt(tl.ball, t, tl.frameStep, RULES.ballHold, RULES.ballInterpolateGap);
    const frame = frameAt(tl, t);
    return {time: t, frame, calibrated: frame >= 0 && !!tl.calibrated[frame], players, ball};
  }

  render(t, options = {}) {
    const selected = options.selected || null;
    const key = `${t.toFixed(4)}|${this.colours.A}|${this.colours.B}|${this.showLabels()}|${this.showDebug()}|${selected}|${this.timeline ? 1 : 0}|${this.canvas.width}`;
    if (key === this.lastKey) return this.last;
    this.lastKey = key;
    const g = this.geometry(), ctx = this.ctx;
    if (!this.pitchLayer || this.pitchLayer.width !== this.canvas.width) this.pitchLayer = this.drawPitch(g);
    ctx.clearRect(0, 0, g.width, g.height);
    ctx.drawImage(this.pitchLayer, 0, 0);
    const state = this.positions(t);
    for (const overlay of this.overlays) overlay(ctx, g, state);   // layer 3: future analytics
    this.drawPlayers(ctx, g, state, selected);                     // layers 4 and 5
    this.drawBall(ctx, g, state);                                  // layer 6
    if (this.showDebug()) this.drawDebug(ctx, g, state);
    this.updateNotice(state);
    this.last = state;
    return state;
  }

  drawPlayers(ctx, g, state, selected) {
    const r = g.playerRadius, labels = this.showLabels();
    ctx.lineWidth = Math.max(1, r * .16);
    for (const p of state.players) {
      const [x, y] = g.toPx(p.x, p.y), colour = p.team ? this.colours[p.team] : NEUTRAL;
      ctx.globalAlpha = p.alpha * (selected && selected !== p.id ? .35 : 1);
      ctx.fillStyle = colour; ctx.strokeStyle = '#0d1a24b8';
      ctx.beginPath(); ctx.arc(x, y, r, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
      if (p.source === 'HELD') { ctx.setLineDash([r * .5, r * .4]); ctx.strokeStyle = '#ffffffaa'; ctx.beginPath(); ctx.arc(x, y, r * 1.35, 0, Math.PI * 2); ctx.stroke(); ctx.setLineDash([]); }
      if (selected === p.id) { ctx.strokeStyle = '#ffffff'; ctx.beginPath(); ctx.arc(x, y, r * 1.5, 0, Math.PI * 2); ctx.stroke(); }
      if (p.role === 'GOALKEEPER') {
        ctx.font = `bold ${Math.round(r * .95)}px system-ui, sans-serif`; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
        ctx.fillStyle = luminance(colour) > .55 ? '#10202a' : '#ffffff'; ctx.fillText('GK', x, y + r * .05);
      }
    }
    ctx.globalAlpha = 1;
    if (!labels) return;
    ctx.font = `${Math.max(9, Math.round(r * .95))}px system-ui, sans-serif`; ctx.textAlign = 'center'; ctx.textBaseline = 'alphabetic';
    for (const p of state.players) {
      const [x, y] = g.toPx(p.x, p.y);
      ctx.globalAlpha = p.alpha * (selected && selected !== p.id ? .35 : 1);
      ctx.lineWidth = 3; ctx.strokeStyle = '#07131bcc'; ctx.strokeText(p.display, x, y - r - 3);
      ctx.fillStyle = '#f4f7f4'; ctx.fillText(p.display, x, y - r - 3);
    }
    ctx.globalAlpha = 1;
  }

  drawBall(ctx, g, state) {
    const b = state.ball;
    if (!b) return;
    const [x, y] = g.toPx(b.x, b.y), r = Math.max(3.5, g.playerRadius * .45);
    ctx.save();
    ctx.globalAlpha = b.source === 'PREDICTED' || b.source === 'HELD' ? .7 : 1;
    ctx.shadowColor = 'rgba(0,0,0,.55)'; ctx.shadowBlur = r; ctx.shadowOffsetY = r * .3;
    ctx.fillStyle = '#ffffff'; ctx.beginPath(); ctx.arc(x, y, r, 0, Math.PI * 2); ctx.fill();
    ctx.shadowColor = 'transparent';
    ctx.lineWidth = Math.max(1, r * .35); ctx.strokeStyle = '#141c24';
    if (b.source !== 'OBSERVED' && b.source !== 'INTERPOLATED') ctx.setLineDash([r * .6, r * .5]);
    ctx.stroke();
    ctx.restore();
  }

  drawDebug(ctx, g, state) {
    const size = Math.max(9, Math.round(g.playerRadius * .8));
    ctx.font = `${size}px ui-monospace, monospace`; ctx.textAlign = 'left'; ctx.textBaseline = 'top';
    const block = (x, y, lines) => {
      const w = Math.max(...lines.map(l => ctx.measureText(l).width)) + 8, h = lines.length * (size + 2) + 6;
      ctx.fillStyle = '#07131bd0'; ctx.fillRect(x, y, w, h);
      ctx.fillStyle = '#dbe6ef'; lines.forEach((l, i) => ctx.fillText(l, x + 4, y + 3 + i * (size + 2)));
    };
    for (const p of state.players) {
      const [x, y] = g.toPx(p.x, p.y);
      block(x + g.playerRadius + 3, y - size, [p.display, `x ${p.x.toFixed(3)} y ${p.y.toFixed(3)}`, p.source, `conf ${p.confidence.toFixed(2)}`]);
    }
    if (state.ball) {
      const [x, y] = g.toPx(state.ball.x, state.ball.y);
      block(x + g.playerRadius, y + g.playerRadius * .5, ['BALL', `x ${state.ball.x.toFixed(3)} y ${state.ball.y.toFixed(3)}`, state.ball.source, `conf ${state.ball.confidence.toFixed(2)}`]);
    }
    const tl = this.timeline;
    block(6, 6, [`video ${state.time.toFixed(2)} s · frame ${state.frame >= 0 ? state.frame : '-'} · ${state.calibrated ? 'calibrated' : 'no calibration here'}`,
                 `${state.players.length} players placed · ball ${state.ball ? state.ball.source : 'hidden'}${tl ? ` · timeline ${tl.tracks.size} identities, ${tl.observed} observations` : ''}`]);
  }

  updateNotice(state) {
    if (!this.notice) return;
    const tl = this.timeline;
    const text = !tl ? 'The 2D match view appears here after a run.' :
      !tl.calibratedFrames ? 'No pitch coordinates in this result. Add calibration keyframes (Mark landmarks, four or more) and process the segment again.' :
      !state.calibrated && state.frame >= 0 && !state.players.length ? 'The camera is not calibrated at this moment: positions are unknown until the next calibration keyframe.' : '';
    if (this.notice.textContent !== text) { this.notice.textContent = text; this.notice.hidden = !text; }
  }

  state() { return this.last; }
}

global.Tactical = {PITCH, RULES, buildTimeline, sampleAt, finalRoster, create: (options) => new TacticalView(options)};
})(window);
