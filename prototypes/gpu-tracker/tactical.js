/* PitchIQ · 2D match view: a top-down recreation of the match from the saved tracking result.

   Data. The pipeline's result.json. Every observed person with a global identity carries `pitch`: the
   normalized pitch position of the feet (x 0..1 from the left goal line to the right goal line, y 0..1
   from the far touchline to the near touchline, on a 105 x 68 m pitch), computed by soccer/calibration.py
   from the user's landmark calibration and carried through pans and zooms by the camera-motion estimate.
   The ball carries `pitch` the same way. Nothing here is derived from screen coordinates: without a
   calibration there are no positions, and none are invented.
   When the result has a `stabilized` block (soccer/trajectory.py: per identity the foot-point noise of the
   detector boxes filtered out of the observations, imputation's estimates made on the filtered ones), the
   players are drawn from it, as given: those positions are smoothed once, there, and here they are only
   joined by straight lines between frames. The raw positions stay for the developer comparison (Debug
   positions: raw, stabilized or both).

   Structure, so that analytics overlays (trails, heat maps, formations, lanes) can be added later:
     Timeline  built once per result from the finalized frames and the final registry (teams and roles
               as the identity manager settled them, retroactive relabelling included): per identity,
               time-sorted samples with a centred smoothing window; the ball likewise; the stabilized
               tracks, when the result has them, unsmoothed (built on first use: the analytics views load a
               timeline for the pitch only).
     Sampler   positions at any video time: the observation of that frame, a linear interpolation across
               a short gap, or the last position held briefly (fading) before the marker hides (sampleAt);
               a stabilized track by the straight line between its samples at every time (sampleStable).
     Renderer  canvas layers in order: pitch (cached), overlays (hooks), players, labels, ball, debug
               (the comparison 'both': the raw positions as crosses over the players).
   The <video> element is the only clock: the page calls view.render(video.currentTime) on every animation
   frame, and the view redraws only when the time or a setting changed. */
(function (global) {
'use strict';

const PITCH = {length: 105, width: 68, penaltyDepth: 16.5, penaltyHalf: 20.16, goalDepth: 5.5, goalHalf: 9.16,
               spot: 11, circle: 9.15, goalWidth: 7.32, goalNet: 2.2, cornerArc: 1.0, margin: 4.5};
const RULES = {
  interpolateGap: 1.0,     // s: an identity observed on both sides of a gap this long is interpolated across it
  holdAfter: 0.6,          // s: after its last observation a marker stays (fading) this long, then hides
  anonymousGap: 0.3,       // s: an unidentified person's observations are joined only across this much (a stabilizer's bridge too)
  anonymousHold: 0.3,      // s: ... and held this long
  ballHold: 0.2,           // s: the ball tracker already predicts while hidden; hold only a little longer
  ballInterpolateGap: 0.15, // s: a lost ball is never bridged by a straight line; only a dropped frame or two is
  smoothWindow: 0.14,      // s: centred averaging window over a player's observations (offline, no lag)
  ballSmoothWindow: 0.06,
  ballPredictedMin: 0.5,   // ball confidence needed to draw a predicted (hidden) ball
  outside: 0.1,            // normalized: how far beyond the lines a position may still be drawn
  uncertainSigma: 8.0,     // m: a prediction less certain than this is faded and marked, and hidden unless the full lineup is shown
};
const DEFAULT_COLOURS = {A: '#b98cff', B: '#f6b564'};
const NEUTRAL = '#8d99a6';   // a goalkeeper whose team is not known yet
const POSITION_MODES = ['raw', 'stabilized', 'both'];   // the developer comparison of the drawn positions (Debug positions)
const POSITIONS_KEY = 'pitchiq.tactical.positions';      // ... remembered per viewer, not per video
const SOURCE = {O: 'OBSERVED', I: 'INTERPOLATED', P: 'PREDICTED'};

// ---------- timeline ----------
class Track {
  constructor(info) {
    this.info = info;
    this.t = []; this.x = []; this.y = []; this.c = []; this.predicted = []; this.q = []; this.k = []; this.s = [];
  }
  push(t, x, y, confidence, predicted, quality = 1, kind = null, sigma = null) {
    if (this.t.length && t <= this.t[this.t.length - 1]) return;   // frames arrive in time order
    this.t.push(t); this.x.push(x); this.y.push(y); this.c.push(confidence); this.predicted.push(!!predicted);
    this.q.push(Number.isFinite(quality) ? quality : 1);   // pitch position confidence: the registration's, times the local projection certainty
    this.k.push(kind || (predicted ? 'P' : 'O'));            // the sample's kind: O observed, I interpolated, P predicted
    this.s.push(Number.isFinite(sigma) ? sigma : null);      // its sigma in metres, when the timeline gives one
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
    this.x = xs; this.y = ys; this.t = Float64Array.from(this.t); this.c = Float64Array.from(this.c); this.q = Float64Array.from(this.q);
    return this;
  }
  smoothObserved(window) {
    // The smoothing of smooth(), restricted to observed samples: estimates are left as the timeline gives them.
    const n = this.t.length, xs = Float64Array.from(this.x), ys = Float64Array.from(this.y);
    for (let i = 0; i < n; i++) {
      if (this.k[i] !== 'O') continue;
      let sx = 0, sy = 0, m = 0;
      for (let j = i; j >= 0 && this.t[i] - this.t[j] <= window / 2 && this.k[j] === 'O'; j--) { sx += this.x[j]; sy += this.y[j]; m++; }
      for (let j = i + 1; j < n && this.t[j] - this.t[i] <= window / 2 && this.k[j] === 'O'; j++) { sx += this.x[j]; sy += this.y[j]; m++; }
      xs[i] = sx / m; ys[i] = sy / m;
    }
    this.x = xs; this.y = ys; this.t = Float64Array.from(this.t); this.c = Float64Array.from(this.c); this.q = Float64Array.from(this.q);
    return this;
  }
}

function validXY(x, y) {
  return Number.isFinite(x) && Number.isFinite(y) && x >= -RULES.outside && x <= 1 + RULES.outside && y >= -RULES.outside && y <= 1 + RULES.outside;
}
function validPosition(p) {
  return Array.isArray(p) && p.length === 2 && validXY(p[0], p[1]);
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
  let observed = 0, ballFrames = 0, anonymous = 0, estimated = 0;
  const block = result.timeline && result.timeline.ids && result.timeline.frameTimes && result.timeline.frameTimes.length === frames.length ? result.timeline : null;
  if (block) {
    // The finished timeline: every roster slot on every frame with its kind and sigma (estimates included).
    for (const [id, cols] of Object.entries(block.ids)) {
      if (!roster.has(id)) continue;
      const track = new Track(roster.get(id));
      for (let i = 0; i < frames.length; i++) {
        const kind = cols.kind[i];
        if (kind === 'U' || cols.x[i] === null || cols.y[i] === null || !validPosition([cols.x[i], cols.y[i]])) continue;
        const sigma = Array.isArray(cols.sigma[i]) ? Math.max(cols.sigma[i][0], cols.sigma[i][1]) : null;
        track.push(frames[i].time, cols.x[i], cols.y[i], kind === 'O' ? 1 : .5, kind === 'P', kind === 'O' ? 1 : Math.max(.3, 1 - (sigma || 0) / 20), kind, sigma);
        if (kind === 'O') observed++; else estimated++;
      }
      if (track.t.length) tracks.set(id, track);
    }
  }
  frames.forEach((f, i) => {
    frameTimes[i] = f.time;
    calibrated[i] = f.calibrated ? 1 : 0;
    for (const p of f.people || []) {
      if (!validPosition(p.pitch)) continue;
      if (!p.id) {
        // A tracked person with a team but no identity: a real observation, shown as a '?' of that team.
        if (p.state !== 'uncertain' || (p.team !== 'A' && p.team !== 'B')) continue;
        const key = 'anon:' + p.track;
        let track = tracks.get(key);
        if (!track) tracks.set(key, track = new Track({id: null, display: '?', team: p.team, role: 'PLAYER', anonymous: true, track: p.track}));
        track.push(f.time, p.pitch[0], p.pitch[1], .3, false, p.pitchConfidence);
        anonymous++;
        continue;
      }
      if (!roster.has(p.id) || block) continue;   // with a timeline block the identities are already built from it
      let track = tracks.get(p.id);
      if (!track) tracks.set(p.id, track = new Track(roster.get(p.id)));
      track.push(f.time, p.pitch[0], p.pitch[1], Number.isFinite(p.identityConfidence) ? p.identityConfidence : .5, false, p.pitchConfidence);
      observed++;
    }
    const b = f.ball;
    if (b && validPosition(b.pitch) && (b.state === 'TRACKED' || (b.state === 'MISSING' && b.confidence >= RULES.ballPredictedMin))) {
      ball.push(f.time, b.pitch[0], b.pitch[1], b.confidence, b.state !== 'TRACKED', b.pitchConfidence);
      ballFrames++;
    }
  });
  for (const track of tracks.values()) if (block && !track.info.anonymous) track.smoothObserved(RULES.smoothWindow); else track.smooth(RULES.smoothWindow);
  ball.smooth(RULES.ballSmoothWindow);
  // The stabilized positions, when the result has them for these frames (stableTracks), built on first use: the
  // analytics and team-shape views load a timeline only for the pitch and its goal ends, and never draw a player.
  // The tracks above stay as they are: they are the raw comparison.
  const stableBlock = result.stabilized && result.stabilized.ids && Array.isArray(result.stabilized.frameTimes) &&
    result.stabilized.frameTimes.length === frames.length ? result.stabilized : null;
  let stable;
  const stabilized = () => stable !== undefined ? stable : (stable = stableBlock ? stableTracks(stableBlock, frameTimes, roster) : null);
  let step = info && info.video && info.video.fps > 0 ? 1 / info.video.fps : 0;
  if (!step && frames.length > 1) step = (frames[frames.length - 1].time - frames[0].time) / (frames.length - 1);
  const calibratedFrames = calibrated.reduce((a, v) => a + v, 0);
  return {tracks, ball, frameTimes, calibrated, frameStep: step || .04, observed, estimated, anonymous, ballFrames, calibratedFrames, frames: frames.length,
          identities: [...tracks.values()].filter(t => !t.info.anonymous).length, hasTimeline: !!block, frames_: frames,
          goalEnds: (result.tactical && result.tactical.goalEnds) || {},
          get stable() { const s = stabilized(); return s ? s.tracks : null; },
          get stableCounts() { return stabilized(); },   // what the stabilized tracks hold, counted as the raw timeline's counts are
          get stableMethod() { return stabilized() ? String(stableBlock.method || '') : null; }};
}

/* The stabilized tracks: per roster identity the stabilizer's observations and bridges with imputation's estimates on
   them, and the anonymous '?' tracks, kept exactly as given (they were smoothed once, by the stabilizer). A '?' is
   joined across a hole only as far as RULES.anonymousGap allows, the stabilizer's bridge included: a bridged stretch
   between two observations further apart is left out, so the marker is held briefly and hides, as without the block
   (a '?' carries no kind on the board, and a bridge drawn would look observed). Returns the tracks and their counts
   (identities, observed and estimated samples, '?' observations), or null when no track has a position. The frames
   are given by their times (frameTimes: the same numbers, and much faster to read than the frames' own field). */
function stableTracks(block, times, roster) {
  const tracks = new Map(), counts = {identities: 0, observed: 0, estimated: 0, anonymous: 0, hasTimeline: true};
  const add = (key, trackInfo, cols) => {
    const kinds = cols.kind || [], xs = cols.x || [], ys = cols.y || [], sigmas = cols.sigma || [], weights = cols.w || [];
    let shown = [];   // the frames with a position
    for (let i = 0; i < times.length; i++) if (kinds[i] && kinds[i] !== 'U' && validXY(xs[i], ys[i])) shown.push(i);
    if (trackInfo.anonymous) {
      const kept = [];
      for (let j = 0, last = -1; j < shown.length;) {   // last: the frame of the latest observation
        if (kinds[shown[j]] === 'O') { last = shown[j]; kept.push(shown[j++]); continue; }
        let next = j;
        while (next < shown.length && kinds[shown[next]] !== 'O') next++;
        if (last >= 0 && next < shown.length && times[shown[next]] - times[last] <= RULES.anonymousGap) kept.push(...shown.slice(j, next));
        j = next;
      }
      shown = kept;
    }
    const track = new Track(trackInfo);
    track.f = [];   // the frame of each sample: two samples of consecutive frames have nothing unknown between them
    track.w = [];   // the stabilizer's weight of the observation (1 normal, below 1 down-weighted, 0 rejected; null: none)
    for (const i of shown) {
      const kind = kinds[i], s = sigmas[i], sigma = Array.isArray(s) ? Math.max(s[0], s[1]) : null, w = weights[i], n = track.t.length;
      track.push(times[i], xs[i], ys[i], trackInfo.anonymous ? .3 : kind === 'O' ? 1 : .5, kind === 'P', kind === 'O' ? 1 : Math.max(.3, 1 - (sigma || 0) / 20), kind, sigma);
      if (track.t.length > n) { track.f.push(i); track.w.push(Number.isFinite(w) ? w : null); }
    }
    if (!track.t.length) return;
    tracks.set(key, track);
    const observations = track.k.filter(k => k === 'O').length;
    if (trackInfo.anonymous) counts.anonymous += observations;
    else { counts.identities++; counts.observed += observations; counts.estimated += track.t.length - observations; }
  };
  for (const [id, cols] of Object.entries(block.ids)) if (roster.has(id)) add(id, roster.get(id), cols);
  for (const [key, cols] of Object.entries(block.anonymous || {})) {
    if (cols.team !== 'A' && cols.team !== 'B') continue;
    const local = Number.isFinite(Number(key)) ? Number(key) : key;   // the raw '?' tracks carry the number
    add('anon:' + local, {id: null, display: '?', team: cols.team, role: 'PLAYER', anonymous: true, track: local}, cols);
  }
  return tracks.size ? {tracks, ...counts} : null;
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
  const quality = (i) => track.q && track.q.length > i ? track.q[i] : 1;
  const kindOf = (i) => track.k && track.k.length > i ? track.k[i] : (track.predicted[i] ? 'P' : 'O');
  const sigmaOf = (i) => track.s && track.s.length > i ? track.s[i] : null;
  const observed = (i) => ({x: track.x[i], y: track.y[i], confidence: track.c[i], source: kindOf(i) === 'P' ? 'PREDICTED' : kindOf(i) === 'I' ? 'INTERPOLATED' : 'OBSERVED',
                            kind: kindOf(i), sigma: sigmaOf(i), alpha: 1, quality: quality(i)});
  if (dt < 0 || dt <= same) return observed(dt < 0 ? 0 : k);
  if (k + 1 < n) {
    const t1 = track.t[k + 1], gap = t1 - t0;
    if (t1 - t <= same) return observed(k + 1);
    if (gap <= gapLimit) {
      const u = dt / gap;
      const kind = kindOf(k) === 'P' || kindOf(k + 1) === 'P' ? 'P' : 'I';
      return {x: track.x[k] + (track.x[k + 1] - track.x[k]) * u, y: track.y[k] + (track.y[k + 1] - track.y[k]) * u,
              confidence: Math.min(track.c[k], track.c[k + 1]), source: kind === 'P' ? 'PREDICTED' : 'INTERPOLATED', kind, alpha: 1,
              sigma: Math.max(sigmaOf(k) || 0, sigmaOf(k + 1) || 0) || null, quality: Math.min(quality(k), quality(k + 1))};
    }
  }
  if (dt <= hold) return {x: track.x[k], y: track.y[k], confidence: track.c[k] * (1 - dt / hold), source: 'HELD', kind: kindOf(k), sigma: sigmaOf(k), alpha: 1 - .7 * dt / hold, quality: quality(k)};
  return null;
}

/* Position of one stabilized track at time t. Its samples are already smoothed, so they are drawn as given: at a
   sample's time that sample, between two samples at most gapLimit apart the straight line between them at the
   actual frame times, at every t (no snapping to the nearer frame as sampleAt does: that was a quarter-frame
   stutter), and after the last sample of a run sampleAt's fading hold. Between two samples the kind is the pair's:
   observations of consecutive frames draw as observed (no interpolated ring between frames), a prediction on either
   side as predicted, anything else (a bridged sample, or frames without a position between the two) as
   interpolated. Only the first sample of a run may show up to a quarter frame early: the video clock and the frame
   times are rounded differently, and there is no earlier position to jump from. */
function sampleStable(track, t, frameStep, hold = RULES.holdAfter, gapLimit = RULES.interpolateGap) {
  const n = track.t.length, early = frameStep * .25;
  if (!n || t < track.t[0] - early) return null;
  const weightOf = (i) => track.w && Number.isFinite(track.w[i]) ? track.w[i] : null;
  const sample = (i) => ({x: track.x[i], y: track.y[i], confidence: track.c[i], source: SOURCE[track.k[i]], kind: track.k[i], sigma: track.s[i],
                          alpha: 1, quality: track.q[i], weight: weightOf(i)});
  if (t <= track.t[0]) return sample(0);
  let lo = 0, hi = n - 1;
  while (lo < hi) { const mid = (lo + hi + 1) >> 1; if (track.t[mid] <= t) lo = mid; else hi = mid - 1; }
  const k = lo, t0 = track.t[k], dt = t - t0;
  if (dt === 0) return sample(k);
  if (k + 1 < n) {
    const t1 = track.t[k + 1], gap = t1 - t0;
    if (gap <= gapLimit) {
      const u = dt / gap, mix = (a, b) => a + (b - a) * u, k0 = track.k[k], k1 = track.k[k + 1];
      const consecutive = track.f ? track.f[k + 1] === track.f[k] + 1 : gap <= 1.5 * frameStep;
      const kind = k0 === 'P' || k1 === 'P' ? 'P' : k0 === 'O' && k1 === 'O' && consecutive ? 'O' : 'I';
      const s0 = track.s[k], s1 = track.s[k + 1], w0 = weightOf(k), w1 = weightOf(k + 1);
      return {x: mix(track.x[k], track.x[k + 1]), y: mix(track.y[k], track.y[k + 1]), confidence: mix(track.c[k], track.c[k + 1]), source: SOURCE[kind], kind,
              sigma: s0 !== null && s1 !== null ? mix(s0, s1) : s0 !== null ? s0 : s1, alpha: 1, quality: mix(track.q[k], track.q[k + 1]),
              weight: w0 !== null && w1 !== null ? Math.min(w0, w1) : w0 !== null ? w0 : w1};
    }
    if (t1 - t <= early) return sample(k + 1);
  }
  if (dt <= hold) return {...sample(k), confidence: track.c[k] * (1 - dt / hold), source: 'HELD', alpha: 1 - .7 * dt / hold};
  return null;
}

// Every track's position at time t through the sampler, far players first so near ones draw on top.
function place(tracks, t, frameStep, sampler) {
  const players = [];
  for (const track of tracks.values()) {
    const anonymous = !!track.info.anonymous;
    const s = sampler(track, t, frameStep, anonymous ? RULES.anonymousHold : RULES.holdAfter, anonymous ? RULES.anonymousGap : RULES.interpolateGap);
    if (s) players.push({...track.info, ...s});
  }
  players.sort((a, b) => a.y - b.y || (a.id < b.id ? -1 : 1));
  return players;
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
    for (const key of ['labels', 'debug', 'lineup', 'footprint']) if (this.controls[key]) this.controls[key].addEventListener('change', () => { this.persist(); redraw(); });
    // The developer comparison of the drawn positions: offered with Debug positions when the result has stabilized ones.
    const positions = this.controls.positions;
    if (positions) {
      try {
        const saved = localStorage.getItem(POSITIONS_KEY);
        if (POSITION_MODES.includes(saved)) positions.value = saved;
      } catch (e) { /* storage unavailable: the default */ }
      positions.addEventListener('change', () => {
        try { localStorage.setItem(POSITIONS_KEY, positions.value); } catch (e) { /* storage unavailable: the choice lasts for the page */ }
        this.updateStatus();
        redraw();
      });
    }
    if (this.controls.debug) this.controls.debug.addEventListener('change', () => { this.syncPositionControl(); this.updateStatus(); });
    this.syncPositionControl();
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
  showLineup() { return this.controls.lineup ? this.controls.lineup.checked : false; }
  showFootprint() { return this.controls.footprint ? this.controls.footprint.checked : true; }
  // Which positions are drawn: the stabilized ones when the result has them; with Debug positions on, the comparison
  // may ask for the raw ones (the drawing before stabilization) or both.
  positionSource() {
    if (!this.timeline || !this.timeline.stable) return 'raw';
    const choice = this.showDebug() && this.controls.positions ? this.controls.positions.value : 'stabilized';
    return POSITION_MODES.includes(choice) ? choice : 'stabilized';
  }
  syncPositionControl() {
    if (this.controls.positionsWrap) this.controls.positionsWrap.hidden = !(this.showDebug() && this.timeline && this.timeline.stable);
  }
  // The counts of the positions drawn (the status line, the debug summary): the stabilized tracks' unless the
  // comparison asks for the raw ones.
  drawnCounts() {
    return this.positionSource() === 'raw' ? this.timeline : this.timeline.stableCounts;
  }

  load(result, info) {
    this.timeline = result && result.frames ? buildTimeline(result, info) : null;
    if (info && info.videoKey) this.restore('pitchiq.tactical.' + info.videoKey);
    this.syncPositionControl();
    this.pitchLayer = null;   // the board names the goal ends once the result says who defends them
    this.lastKey = null;
    this.updateStatus();
  }
  updateStatus() {
    if (!this.status) return;
    const tl = this.timeline, c = tl && tl.calibratedFrames ? this.drawnCounts() : null;
    this.status.textContent = !tl ? '' : !tl.calibratedFrames ? 'No pitch coordinates in this result.' :
      `Pitch calibrated on ${Math.round(100 * tl.calibratedFrames / Math.max(1, tl.frames))}% of frames · ${c.identities} players with positions` +
      (c.hasTimeline ? ` · ${c.observed} observed and ${c.estimated} estimated samples` : '') +
      (c.anonymous ? ` · ${c.anonymous} observations of unidentified people shown as ?` : '') +
      ` · ball placed on ${Math.round(100 * tl.ballFrames / Math.max(1, tl.frames))}% of frames`;
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

  // Positions of everyone at time t, from the timeline (nothing is drawn without a pitch position). With stabilized
  // positions in the result the state says which were drawn, and the comparison 'both' adds the raw ones (state.raw).
  positions(t) {
    const tl = this.timeline;
    if (!tl) return {time: t, frame: -1, calibrated: false, players: [], ball: null};
    const mode = this.positionSource();
    const players = mode === 'raw' ? place(tl.tracks, t, tl.frameStep, sampleAt) : place(tl.stable, t, tl.frameStep, sampleStable);
    const ball = sampleAt(tl.ball, t, tl.frameStep, RULES.ballHold, RULES.ballInterpolateGap);
    const frame = frameAt(tl, t);
    const state = {time: t, frame, calibrated: frame >= 0 && !!tl.calibrated[frame], players, ball};
    if (tl.stable) {
      state.positions = mode;
      if (mode === 'both') state.raw = place(tl.tracks, t, tl.frameStep, sampleAt);
    }
    return state;
  }

  render(t, options = {}) {
    const selected = options.selected || null;
    const key = `${t.toFixed(4)}|${this.colours.A}|${this.colours.B}|${this.showLabels()}|${this.showDebug()}|${this.showLineup()}|${this.showFootprint()}|${selected}|${this.timeline ? 1 : 0}|${this.canvas.width}|${this.positionSource()}`;
    if (key === this.lastKey) return this.last;
    this.lastKey = key;
    const g = this.geometry(), ctx = this.ctx;
    if (!this.pitchLayer || this.pitchLayer.width !== this.canvas.width) this.pitchLayer = this.drawPitch(g);
    ctx.clearRect(0, 0, g.width, g.height);
    ctx.drawImage(this.pitchLayer, 0, 0);
    const state = this.positions(t);
    if (this.showFootprint()) this.drawFootprint(ctx, g, state);   // layer 2b: what the camera sees of the pitch
    for (const overlay of this.overlays) overlay(ctx, g, state);   // layer 3: future analytics
    this.drawPlayers(ctx, g, state, selected);                     // layers 4 and 5
    if (state.raw) this.drawRaw(ctx, g, state, selected);          // the comparison 'both': raw positions on top
    this.drawBall(ctx, g, state);                                  // layer 6
    if (this.showDebug()) this.drawDebug(ctx, g, state);
    this.updateNotice(state);
    this.last = state;
    return state;
  }

  drawFootprint(ctx, g, state) {
    const tl = this.timeline;
    if (!tl || state.frame < 0 || !tl.frames_) return;
    const view = tl.frames_[state.frame] && tl.frames_[state.frame].view;
    if (!Array.isArray(view) || view.length < 3) return;
    ctx.save();
    ctx.beginPath();
    view.forEach(([vx, vy], i) => { const [px, py] = g.toPx(vx, vy); if (i) ctx.lineTo(px, py); else ctx.moveTo(px, py); });
    ctx.closePath();
    ctx.fillStyle = '#ffffff14'; ctx.fill();
    ctx.setLineDash([g.scale * .8, g.scale * .6]); ctx.lineWidth = Math.max(1, g.scale * .12); ctx.strokeStyle = '#ffffff77'; ctx.stroke();
    ctx.restore();
  }

  visible(p) {
    // Estimates are shown when they are reasonably certain; the full lineup shows every estimate the timeline has.
    if (p.kind !== 'P') return true;
    return this.showLineup() || (Number.isFinite(p.sigma) ? p.sigma <= RULES.uncertainSigma : true);
  }

  drawPlayers(ctx, g, state, selected) {
    const r = g.playerRadius, labels = this.showLabels();
    ctx.lineWidth = Math.max(1, r * .16);
    for (const p of state.players) {
      if (!this.visible(p)) continue;
      const [x, y] = g.toPx(p.x, p.y), colour = p.team ? this.colours[p.team] : NEUTRAL;
      const q = Number.isFinite(p.quality) ? p.quality : 1, uncertain = q < .5;
      if (p.anonymous) {
        // An observation without an identity: the team's colour, hollow with a dashed ring and a '?'.
        ctx.globalAlpha = p.alpha * .8 * (selected ? .35 : 1);
        ctx.setLineDash([r * .45, r * .35]); ctx.fillStyle = colour + '66'; ctx.strokeStyle = colour;
        ctx.beginPath(); ctx.arc(x, y, r * .9, 0, Math.PI * 2); ctx.fill(); ctx.stroke(); ctx.setLineDash([]);
        ctx.font = `bold ${Math.round(r * 1.1)}px system-ui, sans-serif`; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
        ctx.fillStyle = luminance(colour) > .55 ? '#10202a' : '#ffffff'; ctx.fillText('?', x, y + r * .05);
        continue;
      }
      const predicted = p.kind === 'P', interpolated = p.kind === 'I', shaky = predicted && Number.isFinite(p.sigma) && p.sigma > RULES.uncertainSigma;
      ctx.globalAlpha = p.alpha * (.45 + .55 * q) * (selected && selected !== p.id ? .35 : 1) * (shaky ? .45 : predicted ? .75 : 1);
      // Kinds: observed solid; interpolated solid with a thin inner ring; predicted translucent with a dashed outline;
      // a very uncertain prediction faded with a '~'.
      ctx.fillStyle = uncertain || predicted ? colour + (predicted ? '77' : '55') : colour; ctx.strokeStyle = uncertain ? colour : predicted ? colour : '#0d1a24b8';
      if (predicted) ctx.setLineDash([r * .5, r * .4]);
      ctx.beginPath(); ctx.arc(x, y, r, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
      ctx.setLineDash([]);
      if (interpolated) { ctx.strokeStyle = '#ffffff99'; ctx.beginPath(); ctx.arc(x, y, r * .6, 0, Math.PI * 2); ctx.stroke(); }
      if (shaky) { ctx.font = `bold ${Math.round(r * 1.2)}px system-ui, sans-serif`; ctx.textAlign = 'center'; ctx.textBaseline = 'middle'; ctx.fillStyle = '#ffffffcc'; ctx.fillText('~', x, y + r * .05); }
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
      if (p.anonymous || !this.visible(p)) continue;
      const [x, y] = g.toPx(p.x, p.y);
      ctx.globalAlpha = p.alpha * (selected && selected !== p.id ? .35 : 1) * (p.kind === 'P' ? .75 : 1);
      ctx.lineWidth = 3; ctx.strokeStyle = '#07131bcc'; ctx.strokeText(p.display, x, y - r - 3);
      ctx.fillStyle = '#f4f7f4'; ctx.fillText(p.display, x, y - r - 3);
    }
    ctx.globalAlpha = 1;
  }

  // The comparison 'both': the raw positions (the drawing before stabilization) as small crosses, each tied by a thin
  // line to the stabilized marker of the same person. A cross, not a ring: on top of its own marker a ring would read
  // as the interpolated marker's inner ring.
  drawRaw(ctx, g, state, selected) {
    const d = g.playerRadius * .32, width = Math.max(1, g.playerRadius * .18), key = (p) => p.anonymous ? 'anon:' + p.track : p.id, drawn = new Map();
    for (const p of state.players) if (this.visible(p)) drawn.set(key(p), p);
    ctx.save();
    ctx.lineCap = 'round';
    for (const p of state.raw) {
      if (!this.visible(p)) continue;
      const [x, y] = g.toPx(p.x, p.y), s = drawn.get(key(p));
      ctx.globalAlpha = p.alpha * (selected && selected !== p.id ? .35 : 1);
      if (s) { const [sx, sy] = g.toPx(s.x, s.y); ctx.lineWidth = 1; ctx.strokeStyle = '#ffffffb3'; ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(sx, sy); ctx.stroke(); }
      ctx.beginPath(); ctx.moveTo(x - d, y - d); ctx.lineTo(x + d, y + d); ctx.moveTo(x - d, y + d); ctx.lineTo(x + d, y - d);
      ctx.lineWidth = width + 2; ctx.strokeStyle = '#07131bcc'; ctx.stroke();   // a dark edge: visible on its own team's marker too
      ctx.lineWidth = width; ctx.strokeStyle = p.team ? this.colours[p.team] : NEUTRAL; ctx.stroke();
    }
    ctx.restore();
  }

  drawBall(ctx, g, state) {
    const b = state.ball;
    if (!b) return;
    const [x, y] = g.toPx(b.x, b.y), r = Math.max(3.5, g.playerRadius * .45);
    ctx.save();
    ctx.globalAlpha = (b.source === 'PREDICTED' || b.source === 'HELD' ? .7 : 1) * (.5 + .5 * (Number.isFinite(b.quality) ? b.quality : 1));
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
    // A stabilized sigma below 0.05 m (stored rounded, it may read 0) is small, not zero. (The raw ones print as before.)
    const metres = (s) => state.positions === 'stabilized' && s < .05 ? '<0.05' : s.toFixed(1);
    // The comparison 'both' is about the markers: per-player boxes would cover them.
    for (const p of state.positions === 'both' ? [] : state.players) {
      const [x, y] = g.toPx(p.x, p.y);
      if (!this.visible(p)) continue;
      block(x + g.playerRadius + 3, y - size, [p.anonymous ? `? track ${p.track}` : p.display, `x ${p.x.toFixed(3)} y ${p.y.toFixed(3)}`,
            p.source + (Number.isFinite(p.sigma) ? ` ± ${metres(p.sigma)} m` : '') + (Number.isFinite(p.weight) ? ` · weight ${p.weight.toFixed(2)}` : ''),
            p.anonymous ? `no identity · position ${(Number.isFinite(p.quality) ? p.quality : 1).toFixed(2)}` : `identity ${p.confidence.toFixed(2)} · position ${(Number.isFinite(p.quality) ? p.quality : 1).toFixed(2)}`]);
    }
    if (state.ball) {
      const [x, y] = g.toPx(state.ball.x, state.ball.y);
      block(x + g.playerRadius, y + g.playerRadius * .5, ['BALL', `x ${state.ball.x.toFixed(3)} y ${state.ball.y.toFixed(3)}`, state.ball.source, `conf ${state.ball.confidence.toFixed(2)} · position ${(Number.isFinite(state.ball.quality) ? state.ball.quality : 1).toFixed(2)}`]);
    }
    const tl = this.timeline, c = tl ? this.drawnCounts() : null;
    const anonymous = state.players.filter(p => p.anonymous).length;
    const summary = [`video ${state.time.toFixed(2)} s · frame ${state.frame >= 0 ? state.frame : '-'} · ${state.calibrated ? 'calibrated' : 'no calibration here'}`,
                     `${state.players.length - anonymous} players placed, ${anonymous} unidentified · ball ${state.ball ? state.ball.source : 'hidden'}${tl ? ` · timeline ${c.identities} identities, ${c.observed} observations` : ''}`];
    if (state.positions) summary.push(`positions ${state.positions}${state.positions === 'both' ? ' (×: raw, tied to the stabilized marker)' : ''}` +
                                      (tl.stableMethod ? ` · ${tl.stableMethod.split(':')[0]}` : ''));   // the method's name, not its description
    block(6, 6, summary);
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

  // The next render draws again even at the same time and options: a layer in this.overlays changed (the coaching
  // evidence shown or cleared while the video is paused), which the render cache key cannot see.
  invalidate() { this.lastKey = null; }
}

global.Tactical = {PITCH, RULES, buildTimeline, sampleAt, sampleStable, finalRoster, create: (options) => new TacticalView(options)};
})(window);
