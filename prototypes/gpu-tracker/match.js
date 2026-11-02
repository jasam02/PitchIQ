/* PitchIQ · Full Match Mode in the review page: the dock's full-match panel and the match player (served at /match.js
   by fullmatch/routes.py; the backend is fullmatch/).

   The dock. Its head gets a SEGMENT | FULL MATCH switch, remembered in localStorage 'pitchiq.dockMode' (default
   Segment, which is the dock as it always was). In Full match the trim timeline and the range row give way to
   #matchPanel; the processing settings stay, they apply to both:
     setup     the match time range (default the whole video), an optional halftime break and the workers; every change
               is planned again (POST /api/match/plan, debounced) for the segments, the workers, the calibration
               coverage, the estimated time and the disk space; ANALYZE FULL MATCH starts it (POST /api/match/start)
     progress  while a match runs or assembles (GET /api/match/status every 2 s): progress, segments, workers, video
               minutes, stage, elapsed and remaining time, why no new segment starts, errors; PAUSE and STOP
     resume    an incomplete match of this video (GET /api/match: resumable): RESUME ANALYSIS, or start over
     open      the newest assembled match (GET /api/match/latest): OPEN FULL MATCH, here and in the hero when no
               Segment result is shown

   The match player (window.matchMode while a match is open). The whole source video plays; the result shown is the
   VIEW of the segment that owns the video's time (index.segments: start <= t < end), a normal Segment-run result with
   the match's names. render() asks frame(t) first: when the owner changes it swaps the shown view with a light swap
   (the result, the board's timeline, the presence map and the pitch index; no showResult, no panel rebuilt, the
   selection kept). A view is fetched when it is needed ("Loading match data…"; meanwhile the shown view's frames
   beyond its cut are drawn while they reach) and ahead of a cut (PREFETCH_S before the shown one's end, the previous
   one after a step or seek backwards near a start), at most MAX_VIEWS are kept, and a view fetched ahead is prepared in
   idle time (its timeline and stabilized tracks, presence and pitch index) so that the swap at the cut only assigns.
   Between the periods and outside the match the last view stays and nothing is drawn. The panels get one match-level
   object at entry (not per swap), and index.html's api() adds match=<id> to their requests (scope()).
   A key moment being watched (index.html watchMoment, the coaching report's moment player) may lie in a segment whose
   view is not loaded: the player asks waiting(t) until the view owning that time is shown, and a swap keeps its mark
   (coachMarkRebase: the key frame is found again in the new view) where any other coaching mark is cleared.

   Corrections. The match index comes with the coach's corrections applied (aliases: merged identities -> the final
   one; excluded; teamOverrides; suggestions). Every view is renamed through them when it arrives (renameView), from
   what it was as assembled, so a correction made or undone renames the loaded views in place. The roster drawer lists
   the match identities with their corrections (join fragments of one player, not a player, change team, the
   suggestions) and the corrections list offers Undo (POST /api/match/correction, /api/match/correction/remove).

   The page's own state and functions are used by name: index.html's inline script loads after this file and declares
   them (info, result, presence, pitchIndex, selected, busy, mode, loadedRun, calibrations, anchors, video, tactical,
   analytics, teamShape, coaching, possession, possessionStats (when the page has it), drawers, api, poll, select, settings, buildPresence, closeEditor,
   clearCoachMark, eventList, pitchIssueList, roster, correctionList, matchMeta, stageStatus, dockState, fitStage,
   frameIndex). Nothing here runs before FullMatch.init() but the dock switch. */
(function (global) {
'use strict';

const MODE_KEY = 'pitchiq.dockMode';
const POLL_MS = 2000;          // the match status
const OVERVIEW_MS = 15000;     // the video's matches and the newest assembled one (besides every change of state)
const PLAN_MS = 400;           // the setup is planned again this long after its last change
const PREFETCH_S = 30;         // the next view is fetched this long before the shown one's end
const MAX_VIEWS = 3;           // views kept (one is 80-100 MB of JSON)
const RETRY_MS = 5000;         // a view that failed to load is asked again after this long
const ANCHOR_STEP = 25;        // frames between the anchors of the panels' match-level object
const OUTSIDE = -1e9;          // a time no frame is near: the overlay and the board draw nothing
const SCOPED = ['/api/analytics', '/api/team-shape', '/api/possession', '/api/possession2', '/api/coaching'];
const BUSY_MODES = ['running', 'pausing', 'stopping', 'assembling'];
const ORIG = Symbol('assembled');   // what renameView changed, as the view was assembled (not enumerable, not serialized)

// ---------- small helpers ----------
const byId = (id) => document.getElementById(id);
const isNum = (v) => typeof v === 'number' && Number.isFinite(v);
const pad2 = (n) => String(n).padStart(2, '0');
const plural = (n, one, many) => `${n} ${n === 1 ? one : (many || one + 's')}`;
const idle = (fn) => (typeof global.requestIdleCallback === 'function' ? global.requestIdleCallback(fn, {timeout: 500}) : setTimeout(fn, 16));
function mk(tag, cls, content) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (content !== undefined && content !== null) e.textContent = content;
  return e;
}
function btn(label, onclick, cls) {
  const b = mk('button', cls || null, label);
  b.type = 'button';
  if (onclick) b.addEventListener('click', onclick);
  return b;
}
function show(el, on) { if (el) el.hidden = !on; }
function setText(el, value) { if (el && el.textContent !== value) el.textContent = value; }
// the Analytics tab's ball possession dashboard (index.html possessionStats), loaded like the other panels where the page has one
function statsPanel() { try { return typeof possessionStats !== 'undefined' && possessionStats ? possessionStats : null; } catch (e) { return null; } }
function callPage(name, ...args) {
  const fn = global[name];
  if (typeof fn !== 'function') return undefined;
  try { return fn(...args); } catch (error) { console.error(`Full match: ${name}() failed`, error); return undefined; }
}
// mm:ss (minutes may pass 59: a 96-minute video ends at 96:00)
function clockText(t) { t = Math.max(0, Math.round(isNum(t) ? t : 0)); return `${pad2(Math.floor(t / 60))}:${pad2(t % 60)}`; }
// h:mm:ss from an hour on
function videoText(t) {
  t = Math.max(0, Math.round(isNum(t) ? t : 0));
  const h = Math.floor(t / 3600), m = Math.floor(t % 3600 / 60), s = t % 60;
  return h ? `${h}:${pad2(m)}:${pad2(s)}` : `${pad2(m)}:${pad2(s)}`;
}
// a duration: "45 s", "12 min", "1 h 05 min"; estimates (round) to 5 minutes from an hour on
function spanText(seconds, round) {
  const s = Math.max(0, isNum(seconds) ? seconds : 0);
  if (s < 60) return `${Math.round(s)} s`;
  let m = Math.round(s / 60);
  if (round && m >= 60) m = Math.round(m / 5) * 5;
  if (m < 60) return `${m} min`;
  const h = Math.floor(m / 60), r = m % 60;
  return r ? `${h} h ${pad2(r)} min` : `${h} h`;
}
function rangeText(r) {
  if (!r || !isNum(r.likely)) return '-';
  return `about ${spanText(r.likely, true)} (${spanText(r.low, true)} – ${spanText(r.high, true)})`;
}
// "mm:ss", "m:ss", "h:mm:ss" or seconds; null when it is none of them
function parseClock(text) {
  const s = String(text === undefined || text === null ? '' : text).trim();
  if (!s) return null;
  const parts = s.split(':');
  if (parts.length > 3) return null;
  let v = 0;
  for (const p of parts) { if (!/^\d+(\.\d+)?$/.test(p)) return null; v = v * 60 + Number(p); }
  return v;
}
function whenText(value) {
  const d = value ? new Date(value) : null;
  if (!d || Number.isNaN(d.getTime())) return value ? String(value) : '';
  return d.toLocaleString(undefined, {day: 'numeric', month: 'short', hour: '2-digit', minute: '2-digit'});
}
const naturally = (a, b) => String(a).localeCompare(String(b), undefined, {numeric: true});

// ---------- state ----------
const st = {
  ready: false, timer: 0,
  overview: null,            // GET /api/match
  status: null,              // GET /api/match/status (null: this server has processed no match)
  statusError: null, lastState: null, wasBusy: false, overviewAt: 0, overviewAsk: null,
  latest: null,              // GET /api/match/latest .match
  plan: null, planError: null, planSeq: 0, planTimer: 0, planning: false,
  setup: {start: 0, end: null, half: false, b0: null, b1: null, workers: 'auto'},   // end null: the end of the video
  acting: null,              // the action in flight: start, resume, pause, stop, open
  message: '', messageError: false,
  calSig: null,
  labels: null,              // the range selects' own texts, while the match panels relabel them
};
let dockMode = 'segment';
let player = null;           // the open full match (also window.matchMode)
global.matchMode = null;

const hasVideo = () => { try { return !!(info && info.video); } catch (e) { return false; } };
const duration = () => (hasVideo() && isNum(Number(info.video.duration)) ? Number(info.video.duration) : 0);
const matchBusy = () => !!(st.status && BUSY_MODES.includes(st.status.mode));
const pageBusy = () => { try { return !!busy; } catch (e) { return false; } };
const drawing = () => { try { return !!mode; } catch (e) { return false; } };

// ---------- the dock switch ----------
function setDockMode(next, byUser) {
  dockMode = next === 'match' ? 'match' : 'segment';
  if (byUser) { try { localStorage.setItem(MODE_KEY, dockMode); } catch (e) { /* storage blocked: the choice lasts for the page */ } }
  const on = dockMode === 'match', dock = byId('dock'), title = byId('dockTitle');
  if (dock) { dock.classList.toggle('match-mode', on); dock.setAttribute('aria-label', on ? 'Analyze full match' : 'Analyze segment'); }
  setText(title, on ? 'Analyze full match' : 'Analyze segment');
  for (const b of document.querySelectorAll('#dockMode [data-mode]')) b.setAttribute('aria-pressed', String(b.dataset.mode === dockMode));
  show(byId('matchPanel'), on);
  if (!st.ready) return;
  if (on) { refreshOverview(); schedulePlan(0); }
  paint();
  callPage('dockState');
  callPage('fitStage');
}
function wireSwitch() {
  for (const b of document.querySelectorAll('#dockMode [data-mode]')) b.addEventListener('click', () => setDockMode(b.dataset.mode, true));
  let saved = null;
  try { saved = localStorage.getItem(MODE_KEY); } catch (e) { /* storage blocked: Segment */ }
  setDockMode(saved === 'match' ? 'match' : 'segment', false);   // before the page's script runs: no flash of the other mode
}

// ---------- the server ----------
async function refreshOverview() {
  if (st.overviewAsk) return st.overviewAsk;
  st.overviewAt = Date.now();
  st.overviewAsk = (async () => {
    const [overview, latest] = await Promise.allSettled([api('/api/match'), api('/api/match/latest')]);
    if (overview.status === 'fulfilled') { st.overview = overview.value; fillWorkers(); }
    if (latest.status === 'fulfilled') st.latest = (latest.value && latest.value.match) || null;
    paint();
  })().finally(() => { st.overviewAsk = null; });
  return st.overviewAsk;
}
async function pollStatus() {
  try {
    const s = await api('/api/match/status');
    st.status = s && s.state && s.state !== 'none' ? s : null;
    st.statusError = null;
  } catch (error) {
    st.statusError = error.message;
  }
  const busyNow = matchBusy(), state = st.status ? `${st.status.matchId}|${st.status.state}|${st.status.mode}` : null;
  // the server refuses Segment runs while a match runs: ANALYZE is off at once (the page's poll keeps it so, and turns it on)
  if (busyNow) { const run = byId('run'); if (run) run.disabled = true; }
  if (busyNow !== st.wasBusy || state !== st.lastState || Date.now() - st.overviewAt > OVERVIEW_MS) {
    const finished = st.wasBusy && !busyNow;
    st.wasBusy = busyNow; st.lastState = state;
    await refreshOverview();
    if (finished && dockMode === 'match') schedulePlan(0);   // the disk space and the measured speed changed
  }
  const sig = calibrationSig();
  if (sig !== st.calSig) { const first = st.calSig === null; st.calSig = sig; if (!first && dockMode === 'match' && !busyNow) schedulePlan(); }
  paint();
  callPage('dockState');
}
function loop() {
  clearTimeout(st.timer);
  pollStatus().catch((error) => console.error('Full match status:', error)).finally(() => { st.timer = setTimeout(loop, POLL_MS); });
}
function calibrationSig() {
  try { return JSON.stringify([(calibrations || []).map((c) => [c.time, (c.points || []).length]), (anchors || []).map((a) => a.time)]); }
  catch (e) { return ''; }
}

// ---------- setup ----------
function resetSetup() {
  const s = st.setup;
  s.start = 0; s.end = null;
  writeSetup();
}
function writeSetup() {
  const s = st.setup, end = s.end === null ? duration() : s.end;
  // a time that does not parse stays as typed (marked invalid) so that it can be corrected
  const put = (id, v) => { const el = byId(id); if (el && document.activeElement !== el && (isNum(v) || v === null)) el.value = isNum(v) ? clockText(v) : ''; };
  put('mpStartIn', s.start); put('mpEndIn', end); put('mpB0In', s.b0); put('mpB1In', s.b1);
  const half = byId('mpHalf');
  if (half) half.checked = s.half;
  for (const id of ['mpB0In', 'mpB1In']) { const el = byId(id); if (el) el.disabled = !s.half; }
}
function defaultBreak() {
  // 45:00-60:00 into the match when it is long enough for two halves, else a tenth around its middle
  const s = st.setup, a = s.start, b = s.end === null ? duration() : s.end, len = b - a;
  if (len >= 105 * 60) { s.b0 = a + 45 * 60; s.b1 = a + 60 * 60; }
  else { const mid = a + len / 2; s.b0 = Math.round(mid - len / 20); s.b1 = Math.round(mid + len / 20); }
}
function readTime(id, current) {
  const el = byId(id);
  if (!el) return current;
  const v = parseClock(el.value);
  return v === null ? NaN : v;
}
function readSetup() {
  const s = st.setup, dur = duration();
  s.start = readTime('mpStartIn', s.start);
  if (isNum(s.start) && s.start < .5) s.start = 0;
  const end = readTime('mpEndIn', s.end);
  s.end = isNum(end) && dur && Math.abs(end - dur) < 1 ? null : end;   // the end of the video, as rounded on screen
  if (s.half) { s.b0 = readTime('mpB0In', s.b0); s.b1 = readTime('mpB1In', s.b1); }
  const w = byId('mpWorkers');
  if (w) s.workers = w.value || 'auto';
}
function validate() {
  // The server's rules (fullmatch/plan.periods), said before asking it.
  const s = st.setup, dur = duration(), bad = new Set();
  let message = null;
  const start = s.start, end = s.end === null ? dur : s.end;
  if (!isNum(start)) { bad.add('mpStartIn'); message = 'Enter the match start as mm:ss.'; }
  else if (!isNum(end)) { bad.add('mpEndIn'); message = 'Enter the match end as mm:ss.'; }
  else if (end > dur + .5) { bad.add('mpEndIn'); message = `The video ends at ${videoText(dur)}: the match cannot end after it.`; }
  else if (start >= end) { bad.add('mpStartIn'); bad.add('mpEndIn'); message = 'The match must end after it starts.'; }
  else if (end - start < 5) { bad.add('mpEndIn'); message = 'A match must be at least 5 seconds long.'; }
  else if (s.half) {
    if (!isNum(s.b0)) { bad.add('mpB0In'); message = 'Enter the break start as mm:ss.'; }
    else if (!isNum(s.b1)) { bad.add('mpB1In'); message = 'Enter the break end as mm:ss.'; }
    else if (!(start < s.b0 && s.b0 <= s.b1 && s.b1 < end)) { bad.add('mpB0In'); bad.add('mpB1In'); message = 'The halftime break must lie inside the match range (start < break start ≤ break end < end).'; }
    else if (s.b0 - start < 5 || end - s.b1 < 5) { bad.add('mpB0In'); bad.add('mpB1In'); message = 'Each half must be at least 5 seconds long.'; }
  }
  for (const id of ['mpStartIn', 'mpEndIn', 'mpB0In', 'mpB1In']) { const el = byId(id); if (el) el.setAttribute('aria-invalid', String(bad.has(id))); }
  return message;
}
function requestBody() {
  const s = st.setup, dur = duration(), end = s.end === null ? dur : Math.min(s.end, dur);
  let processing = {};
  try { processing = settings(); } catch (e) { /* the page's settings are not ready: the server's defaults */ }
  return {start: s.start, end, halftime: s.half ? [s.b0, s.b1] : null, workers: s.workers === 'auto' ? 'auto' : Number(s.workers), settings: processing};
}
function schedulePlan(delay) {
  clearTimeout(st.planTimer);
  st.planTimer = setTimeout(plan, delay === undefined ? PLAN_MS : delay);
}
async function plan() {
  if (!st.ready || dockMode !== 'match' || !hasVideo()) return;
  readSetup();
  const problem = validate();
  if (problem) { st.planSeq++; st.plan = null; st.planError = problem; st.planning = false; paint(); return; }
  const seq = ++st.planSeq;
  st.planning = true; paint();
  try {
    const answer = await api('/api/match/plan', requestBody());
    if (seq !== st.planSeq) return;
    st.plan = answer; st.planError = null;
  } catch (error) {
    if (seq !== st.planSeq) return;
    st.plan = null; st.planError = error.message || String(error);
  } finally {
    if (seq === st.planSeq) { st.planning = false; paint(); }
  }
}
function fillWorkers() {
  const sel = byId('mpWorkers'), d = (st.overview && st.overview.defaults) || {};
  if (!sel) return;
  const auto = isNum(d.autoWorkers) ? d.autoWorkers : null, max = isNum(d.maxWorkers) ? d.maxWorkers : 4, autoMax = isNum(d.autoMax) ? d.autoMax : 3;
  const sig = `${auto}|${max}|${autoMax}`;
  if (sel.dataset.sig === sig || document.activeElement === sel) return;   // not rebuilt under the viewer's hand
  sel.dataset.sig = sig;
  const keep = String(st.setup.workers || 'auto'), options = [['auto', auto ? `Auto (recommended: ${auto})` : 'Auto (recommended)']];
  for (let n = 1; n <= max; n++) options.push([String(n), n > autoMax ? `${n} (experimental)` : String(n)]);
  sel.replaceChildren(...options.map(([value, label]) => { const o = mk('option', null, label); o.value = value; return o; }));
  sel.value = options.some(([value]) => value === keep) ? keep : 'auto';
}

// ---------- actions ----------
function canStart() {
  return st.ready && hasVideo() && !matchBusy() && !pageBusy() && !drawing() && !st.acting && !!st.plan && !st.planError && !st.planning &&
    !(st.plan.disk && st.plan.disk.ok === false);
}
async function act(name, path, body, after) {
  if (st.acting) return;
  st.acting = name; st.message = ''; st.messageError = false; paint();
  try {
    const answer = await api(path, body);
    if (after) after(answer);
  } catch (error) {
    st.message = error.message || String(error); st.messageError = true;
  } finally {
    st.acting = null;
    await pollStatus().catch(() => {});
    paint();
  }
}
function startMatch() {
  readSetup();
  if (validate()) { schedulePlan(0); return; }
  if (!canStart()) return;
  act('start', '/api/match/start', requestBody(), (a) => { st.message = `Full match analysis started (${a.matchId}).`; });
}
function resumeMatch() {
  const r = st.overview && st.overview.resumable;
  if (!r || matchBusy() || pageBusy()) return;
  readSetup();
  // the workers only when chosen here: else the server continues with the match's own choice
  const body = {matchId: r.matchId};
  if (st.setup.workers && st.setup.workers !== 'auto') body.workers = Number(st.setup.workers);
  act('resume', '/api/match/resume', body, () => { st.message = 'Resuming: only the missing segments are processed.'; });
}
function pauseMatch() { if (st.status && st.status.mode === 'running') act('pause', '/api/match/pause', {}, () => { st.message = 'Pausing: the running segments finish and are kept.'; }); }
function stopMatch() { if (st.status && ['running', 'pausing'].includes(st.status.mode)) act('stop', '/api/match/stop', {}, () => { st.message = 'Stopping: the running segments are cancelled; finished ones are kept.'; }); }
function openCalibration() {
  try { drawers.open('advanced'); } catch (e) { return; }
  setTimeout(() => { const b = byId('calibrate'); if (b) { b.scrollIntoView({block: 'center', behavior: 'smooth'}); b.focus({preventScroll: true}); } }, 60);
}

// ---------- painting the panel ----------
function paint() {
  if (!st.ready) return;
  try {
    paintSetup(); paintPlan(); paintProgress(); paintResume(); paintReady(); paintHero();
  } catch (error) {
    console.error('Full match panel:', error);
  }
}
function paintSetup() {
  const v = hasVideo() ? info.video : null, busyNow = matchBusy();
  show(byId('mpSetup'), !busyNow);
  setText(byId('mpSource'), v ? sourceName() : 'No source video');
  setText(byId('mpDuration'), v ? videoText(duration()) : '-');
  const go = byId('mpAnalyze');
  if (go) {
    go.disabled = !canStart();
    go.title = pageBusy() ? 'A segment run is processing: wait for it or stop it first.' : st.planError ? st.planError :
      (st.plan && st.plan.disk && st.plan.disk.ok === false) ? 'Not enough free disk space.' : 'Process the whole match range on this computer';
  }
  const msg = byId('mpMsg');
  if (msg) {
    const text = st.message || (st.planError ? st.planError : pageBusy() ? 'A segment run is processing: the full match can start once it has finished.' : '');
    setText(msg, text);
    msg.classList.toggle('error', !!(st.messageError || (!st.message && st.planError)));
  }
}
function sourceName() {
  try {
    const upload = (info.uploads || []).find((u) => info.videoKey && info.videoKey.startsWith(String(u.file).split('.')[0]));
    return upload ? upload.name : info.video.name;
  } catch (e) { return ''; }
}
function periodsText(periods) {
  if (!Array.isArray(periods) || !periods.length) return '';
  if (periods.length === 1) return `one period, ${clockText(periods[0].start)}–${clockText(periods[0].end)}`;
  return periods.map((p) => `${p.name || 'period'} ${clockText(p.start)}–${clockText(p.end)}`).join(' · ');
}
function paintPlan() {
  // rebuilt only when the plan changed (a rebuild every poll would take the focus off its button)
  const p = st.plan, box = byId('mpPlan'), sig = [p, st.planError, !p && st.planning];
  if (box) box.classList.toggle('stale', !!(p && st.planning));
  if (st.planShown && st.planShown.every((v, i) => v === sig[i])) return;
  st.planShown = sig;
  const dd = (id, parts) => { const el = byId(id); if (el) el.replaceChildren(...parts); };
  const sub = (text) => mk('span', 'sub', text), warn = (text) => mk('span', 'warn', text);
  if (!p) {
    const filler = [sub(st.planning ? 'Planning…' : st.planError ? 'Correct the setup to see the plan.' : '-')];
    for (const id of ['mpPlanText', 'mpWorkersText', 'mpCal', 'mpEta', 'mpDisk']) dd(id, id === 'mpPlanText' ? filler : [sub('-')]);
    return;
  }
  dd('mpPlanText', [document.createTextNode(`${plural(p.segments, 'segment')} of up to ${spanText(p.segmentSeconds)} · ${spanText(p.matchSeconds)} of match`),
    mk('br'), sub(`${periodsText(p.periods)} · ${spanText(p.processedSeconds)} processed with ${p.padSeconds} s of context on each side`)]);
  const w = p.workers || {}, sys = p.system || {}, gpu = sys.gpu;
  const limits = w.mode === 'auto' && Array.isArray(w.binding) && w.binding.length ? ` (limited by the ${w.binding.join(', ')})` : '';
  const machine = [isNum(sys.ramAvailableGB) ? `${sys.ramAvailableGB} GB memory free` : null, isNum(sys.cpus) ? `${sys.cpus} CPU threads` : null,
    gpu ? `${gpu.name}${isNum(gpu.freeGB) ? ` (${gpu.freeGB} GB free)` : ''}` : null].filter(Boolean).join(' · ');
  const workerParts = [document.createTextNode(`${w.mode === 'auto' ? 'Auto: ' : ''}${plural(w.count || 1, 'worker')} at once${limits}`)];
  if (w.note) workerParts.push(mk('br'), warn(w.note));
  if (w.experimental) workerParts.push(mk('br'), warn('Experimental: a fourth worker was about 7% faster on short segments, with about 2.5 GB more memory in use; it has not run a whole match.'));
  if (machine) workerParts.push(mk('br'), sub(machine));
  dd('mpWorkersText', workerParts);
  const c = p.calibration || {}, calParts = [document.createTextNode(`${plural(c.keyframes || 0, 'calibration keyframe')} · ${c.anchored || 0} of ${c.segments || p.segments} segments have a keyframe within reach`)];
  if (c.note) calParts.push(mk('br'), (c.automatic && !(c.discovery && c.discovery.runs) ? warn : sub)(c.note));
  const tool = btn('Mark landmarks…', openCalibration, 'quiet');
  tool.title = 'Open the calibration keyframe tool (Advanced): mark four or more landmarks on a clear frame';
  calParts.push(mk('br'), tool);
  dd('mpCal', calParts);
  const e = p.estimate || {};
  dd('mpEta', [document.createTextNode(rangeText(e)), mk('br'),
    sub(`${e.basis === 'this computer' ? 'measured on this computer' : 'from the benchmark'}${isNum(e.fps) ? ` · ${e.fps} frames/s with ${plural(w.count || 1, 'worker')}` : ''}`)]);
  const d = p.disk || {};
  dd('mpDisk', d.ok === false ? [warn(`Needs about ${d.neededText}, only ${d.freeText} free: free some space before starting.`)] :
    [document.createTextNode(`needs about ${d.neededText || '-'}`), mk('br'), sub(`${d.freeText || '-'} free`)]);
}
function stat(label, value, note, cls) {
  const box = mk('div', 'mp-stat' + (cls ? ' ' + cls : ''));
  box.append(mk('span', 'mp-k', label), mk('b', null, value));
  if (note) box.append(mk('small', null, note));
  return box;
}
function paintProgress() {
  const s = st.status, on = matchBusy(), box = byId('mpProgress');
  show(box, on);
  if (!on || !box) return;
  const progress = Math.max(0, Math.min(1, Number(s.progress) || 0));
  const bar = byId('mpBar');
  if (bar) { bar.value = progress; bar.setAttribute('aria-valuetext', `${Math.round(progress * 100)}%`); }
  setText(byId('mpPct'), `${Math.round(progress * 100)}%`);
  setText(byId('mpStage'), `Stage: ${s.stage || s.mode}`);
  const seg = s.segments || {}, w = s.workers || {}, v = s.video || {}, eta = s.eta;
  const minutes = (x) => (isNum(x) ? Math.floor(x / 60) : 0);
  const workers = Array.isArray(w.list) ? w.list.filter((x) => x.state === 'busy' && x.segment !== null && x.segment !== undefined) : [];
  const stats = [
    stat('Completed segments', `${seg.done || 0} / ${seg.total || 0}`, seg.failed ? `${seg.failed} failed` : seg.running ? `${seg.running} running` : null),
    stat('Workers', `${w.active || 0} active`, workers.length ? workers.map((x) => `segment ${x.segment + 1}${isNum(x.progress) ? ` ${Math.round(x.progress * 100)}%` : ''}${isNum(x.fps) ? ` · ${x.fps} fps` : ''}`).join(' · ') :
      (isNum(w.target) && w.target !== w.active ? `${w.target} planned` : null)),
    stat('Processed video', `${minutes(v.doneSeconds)} of ${Math.max(1, Math.round((v.totalSeconds || 0) / 60))} min`),
    stat('Elapsed', spanText(s.elapsed)),
    stat('Remaining', eta ? rangeText(eta) : '-', eta && !eta.measured ? 'estimate from the benchmark' : eta ? 'measured on this match' : null),
  ];
  const cal = calibrationStat(s);
  if (cal) stats.splice(2, 0, cal);
  const grid = byId('mpStats');
  if (grid) grid.replaceChildren(...stats);
  const held = byId('mpHeld');
  if (held) { setText(held, s.held ? `No new segment starts: ${s.held}.` : ''); show(held, !!s.held); }
  const errors = byId('mpErrors'), list = Array.isArray(s.errors) ? s.errors.slice(-3) : [];
  if (errors) {
    errors.replaceChildren(...list.map((e) => mk('li', null, `${isNum(e.segment) ? `Segment ${e.segment + 1}: ` : isNum(e.worker) ? `Worker ${e.worker}: ` : ''}${e.error || e.kind || 'error'}`)));
    show(errors, list.length > 0);
  }
  const pause = byId('mpPause'), stop = byId('mpStop');
  if (pause) { pause.disabled = s.mode !== 'running' || !!st.acting; setText(pause, s.mode === 'pausing' ? 'PAUSING…' : 'PAUSE'); }
  if (stop) { stop.disabled = !['running', 'pausing'].includes(s.mode) || !!st.acting; setText(stop, s.mode === 'stopping' ? 'STOPPING…' : 'STOP'); }
  setText(byId('mpActNote'), s.mode === 'assembling' ? 'Assembling the match: the segments are stitched into one.' :
    s.mode === 'pausing' ? 'Pausing: the running segments finish; RESUME continues later.' : s.mode === 'stopping' ? 'Stopping: finished segments are kept.' : (st.message || ''));
}
// The calibration discovery before tracking: chunks solved while it runs, then what it found (or why it did not).
function calibrationStat(s) {
  const d = s.discovery;
  if (!d || d.state === 'off') return null;
  if (s.phase === 'discovery' || d.state === 'running' || d.state === 'pending') {
    return stat('Calibration', `${d.done || 0} / ${d.total || 0} chunks`, `finding keyframes automatically${isNum(d.sampled) ? ` · ${d.sampled} frames sampled` : ''}${isNum(d.solving) && d.solving ? ` · ${d.solving} processes` : ''}`);
  }
  const sum = d.summary || {}, c = s.calibration || {};
  if (d.state === 'complete') return stat('Calibration', `${plural(sum.keyframes || 0, 'automatic keyframe')}`, c.note || null);
  return stat('Calibration', d.state === 'refused' ? 'not automatic' : 'discovery failed', c.note || sum.refused || sum.error || null, 'mp-warn');
}
function paintResume() {
  const r = st.overview && st.overview.resumable, on = !!r && !matchBusy() && hasVideo(), box = byId('mpResume');
  show(box, on);
  if (!on || !box) return;
  const seg = r.segments || {}, done = seg.done || 0, total = r.total || 0;
  box.classList.toggle('mp-fail', r.state === 'failed');
  setText(byId('mpResumeTitle'), `An analysis of this match is incomplete (${done} of ${total} segments done)`);
  const words = {paused: 'Paused', stopped: 'Stopped', failed: 'Failed', planned: 'Planned, not started'};
  const s = st.status && st.status.matchId === r.matchId ? st.status : null;
  const errors = s && Array.isArray(s.errors) && s.errors.length ? ` · last error: ${s.errors[s.errors.length - 1].error}` : '';
  setText(byId('mpResumeMeta'), `${words[r.state] || r.state}${r.reason ? `: ${r.reason}` : ''} · started ${whenText(r.created)}${errors}. RESUME ANALYSIS processes only the missing segments; Start over runs a new analysis with the setup below.`);
  const resume = byId('mpResumeBtn'), over = byId('mpStartOver');
  if (resume) { resume.disabled = !!st.acting || pageBusy() || r.compatible === false; setText(resume, st.acting === 'resume' ? 'RESUMING…' : 'RESUME ANALYSIS'); }
  if (over) over.disabled = !canStart();
}
function paintReady() {
  const box = byId('mpReady'), latest = st.latest, open = !!player;
  show(box, hasVideo() && (open || !!latest));
  if (!box) return;
  const kicker = byId('mpReadyKicker'), meta = byId('mpReadyMeta'), go = byId('mpOpen'), back = byId('mpLeave');
  if (open) {
    setText(kicker, 'Full match open');
    setText(meta, player.segs ? `${plural(player.segs.length, 'segment')} · ${span() || ''}${player.current >= 0 ? ` · showing segment ${player.current + 1}` : ' · loading'}` : 'Loading the match…');
  } else if (latest) {
    setText(kicker, 'Full match ready');
    const runtime = latest.runtime || {}, took = isNum(runtime.totalSeconds) ? runtime.totalSeconds : runtime.trackingSeconds;
    setText(meta, `${plural(latest.segments || 0, 'segment')}${Array.isArray(latest.periods) && latest.periods.length > 1 ? ` · ${latest.periods.length} periods` : ''} · started ${whenText(latest.created)}` +
      `${isNum(took) ? ` · processed in ${spanText(took)}${isNum(runtime.workers) ? ` with ${plural(runtime.workers, 'worker')}` : ''}` : ''}`);
  }
  show(go, !open); show(back, open);
  if (go) { go.disabled = !!st.acting; setText(go, st.acting === 'open' ? 'OPENING…' : 'OPEN FULL MATCH'); }
}
function paintHero() {
  let shown = null;
  try { shown = result; } catch (e) { shown = null; }
  const on = hasVideo() && !!st.latest && !player && !shown;
  show(byId('heroMatch'), on);
  if (on) setText(byId('heroMatchMeta'), `A full match is ready · ${plural(st.latest.segments || 0, 'segment')}`);
  const heroBtn = byId('heroOpenMatch');
  if (heroBtn) heroBtn.disabled = !!st.acting;
  const badge = byId('matchBadge'), back = byId('matchBack');
  show(badge, !!player); show(back, !!player);
}

// ---------- the dock state chip, the stage status (index.html asks) ----------
function chip() {
  const s = st.status;
  if (!s) return null;
  const words = {running: 'Full match running', pausing: 'Full match pausing', stopping: 'Full match stopping', assembling: 'Full match assembling'};
  if (words[s.mode]) return {text: words[s.mode], busy: true, error: false};
  if (dockMode !== 'match') return null;
  if (s.state === 'paused') return {text: 'Full match paused', busy: false, error: false};
  if (s.state === 'stopped') return {text: 'Full match stopped', busy: false, error: false};
  if (s.state === 'failed') return {text: 'Full match failed', busy: false, error: true};
  return null;
}
function gapText(m, t) {
  const segs = m.segs || [];
  if (!segs.length || t < segs[0].start || t >= segs[segs.length - 1].end) return 'Outside the analysed match';
  const next = segs.find((s) => s.start > t), periods = (m.index && m.index.periods) || [];
  return next && next.period === 1 && periods.length <= 2 ? 'Halftime · not analysed' : 'Break between periods · not analysed';
}
function stage() {
  const m = player;
  if (!m) return null;
  if (!m.segs || m.current < 0) return {text: 'Loading match data', hero: 'Opening the full match…'};
  if (m.owner < 0) return {text: gapText(m, video.currentTime).split(' · ')[0], hero: 'Full match'};
  const period = ((m.index.periods || [])[m.segs[m.owner].period] || {}).name;
  return {text: `Segment ${m.owner + 1} of ${m.segs.length}`, hero: `Full match${period && period !== 'match' ? ` · ${period}` : ''}`};
}
function span() {
  const m = player;
  if (!m || !m.segs || !m.segs.length) return null;
  return `${clockText(m.segs[0].start)}–${clockText(m.segs[m.segs.length - 1].end)}`;
}
function scope(path) {
  const m = player;
  if (!m || typeof path !== 'string' || /[?&]match=/.test(path)) return path;
  if (!SCOPED.some((p) => path === p || path.startsWith(p + '?') || path.startsWith(p + '/'))) return path;
  return path + (path.includes('?') ? '&' : '?') + 'match=' + encodeURIComponent(m.id);
}

// ---------- renaming a view through the corrections ----------
function contextOf(index) {
  const aliases = index && index.aliases && typeof index.aliases === 'object' ? index.aliases : {};
  const memo = new Map(), own = Object.prototype.hasOwnProperty;
  const resolve = (id) => {
    if (id === null || id === undefined) return id;
    if (memo.has(id)) return memo.get(id);
    let f = id;
    for (let n = 0; n < 32 && own.call(aliases, f) && aliases[f] !== f && aliases[f] !== null && aliases[f] !== undefined; n++) f = aliases[f];
    memo.set(id, f);
    return f;
  };
  const roster = new Map();
  for (const p of (index && index.roster) || []) if (p && typeof p.matchId === 'string') roster.set(p.matchId, p);
  const excluded = new Set(Array.isArray(index && index.excluded) ? index.excluded : []);
  for (const p of roster.values()) if (p.excluded) excluded.add(p.matchId);
  const overrides = new Map(Object.entries((index && index.teamOverrides) || {}).filter(([, v]) => v === 'A' || v === 'B'));
  const out = (a, f) => excluded.has(f) || excluded.has(a);
  return {aliases, resolve, roster, excluded, overrides, out};
}
function assembledOf(view) {
  const m = view.match || {}, clone = (list) => (Array.isArray(list) ? list.map((e) => ({...e})) : []);
  return {
    timeline: view.timeline && view.timeline.ids && typeof view.timeline.ids === 'object' ? view.timeline.ids : null,
    stabilized: view.stabilized && view.stabilized.ids && typeof view.stabilized.ids === 'object' ? view.stabilized.ids : null,
    match: {players: clone(m.players), goalkeepers: clone(m.goalkeepers), excluded: clone(m.excluded)},
    slots: view.roster && Array.isArray(view.roster.slots) ? clone(view.roster.slots) : null,
    visibility: view.frames.map((f) => f.visibility || null),
    labels: view.labels && typeof view.labels === 'object' ? view.labels : null,
  };
}
// per frame the better of two samples: observed over interpolated over predicted over none
function sampleRank(col, i) {
  const x = col.x ? col.x[i] : null, k = col.kind ? col.kind[i] : null;
  if (x === null || x === undefined || !k || k === 'U') return 0;
  return k === 'O' ? 3 : k === 'I' ? 2 : 1;
}
function mergeColumns(cols) {
  // cols[0] is the final identity's own column when this view has one; the others fill the frames it has no sample on
  const base = cols[0], n = Array.isArray(base.x) ? base.x.length : 0, out = {...base};
  const fields = Object.keys(base).filter((k) => Array.isArray(base[k]) && base[k].length === n);
  for (const k of fields) out[k] = base[k].slice();
  for (let c = 1; c < cols.length; c++) {
    const other = cols[c];
    for (let i = 0; i < n; i++) {
      if (sampleRank(other, i) <= sampleRank(out, i)) continue;
      for (const k of fields) if (Array.isArray(other[k]) && other[k].length === n) out[k][i] = other[k][i];
    }
  }
  return out;
}
function renameEntries(list, ctx, keepGroup) {
  // match.players / goalkeepers / roster slots: the final ids, one entry per identity (the final one's own kept), the
  // corrected teams and the match's displays; excluded identities are returned apart
  const kept = [], at = new Map(), gone = [];
  for (const e of list) {
    if (!e || e.id === null || e.id === undefined) { kept.push({...e}); continue; }
    const a = e.id, f = ctx.resolve(a), r = ctx.roster.get(f);
    const team = ctx.overrides.get(f) || (r && (r.team === 'A' || r.team === 'B') ? r.team : e.team);
    const entry = {...e, id: f, team, display: r && r.display ? r.display : e.display};
    if (f !== a) entry.joinedFrom = a;
    if (ctx.out(a, f)) { if (!gone.some((g) => g.id === f)) gone.push({...entry, excludedByYou: true}); continue; }
    if (at.has(f)) { if (a === f) kept[at.get(f)] = entry; continue; }
    at.set(f, kept.length);
    kept.push(entry);
  }
  void keepGroup;
  return {kept, gone};
}
// what a rename changed is kept on the object itself, under a symbol that is not enumerable (spreads and JSON skip it)
function keepOriginal(target, value) { Object.defineProperty(target, ORIG, {value, enumerable: false, configurable: true, writable: true}); }
function renameView(view, ctx) {
  // In place, from the view as assembled (kept on its first rename): the corrections of the current index. Numbers are
  // not touched; ids, displays, teams and the excluded people change, and the 2D columns of joined fragments merge.
  if (!view || !Array.isArray(view.frames)) return view;
  if (!view[ORIG]) Object.defineProperty(view, ORIG, {value: assembledOf(view), enumerable: false});
  const base = view[ORIG];
  // the people of every frame
  for (const f of view.frames) {
    const people = f.people;
    if (!Array.isArray(people)) continue;
    for (const p of people) {
      const o = p[ORIG], a = o ? o.id : p.id;
      if (a === null || a === undefined) continue;
      const fid = ctx.resolve(a);
      let want;
      if (ctx.out(a, fid)) {
        // as a Segment run's exclusion marks them (corrections.py): no identity, shown only with Identity details
        want = {id: null, display: 'OUT', team: null, role: 'UNKNOWN', label: 'EXCLUDED_BY_YOU', state: 'rejected', reason: 'marked by you as not a player', excludedId: fid};
      } else {
        const src = o || p, r = ctx.roster.get(fid);
        want = {id: fid, display: r && r.display ? r.display : src.display, team: ctx.overrides.get(fid) || src.team, role: src.role, label: src.label, state: src.state, reason: src.reason, excludedId: src.excludedId};
      }
      if (p.id === want.id && p.display === want.display && p.team === want.team && p.role === want.role && p.label === want.label &&
          p.state === want.state && p.reason === want.reason && p.excludedId === want.excludedId) continue;
      if (!o) keepOriginal(p, {id: p.id, display: p.display, team: p.team, role: p.role, label: p.label, state: p.state, reason: p.reason, excludedId: p.excludedId});
      for (const [k, v] of Object.entries(want)) { if (v === undefined) delete p[k]; else p[k] = v; }
    }
  }
  // the 2D columns (the timeline and the stabilized positions): renamed, joined fragments merged per frame
  for (const key of ['timeline', 'stabilized']) {
    const block = view[key], ids0 = base[key];
    if (!block || !ids0) continue;
    const groups = new Map();
    for (const [a, col] of Object.entries(ids0)) {
      const fid = ctx.resolve(a);
      if (ctx.out(a, fid) || !col) continue;
      let g = groups.get(fid);
      if (!g) groups.set(fid, g = []);
      if (a === fid) g.unshift(col); else g.push(col);
    }
    const ids = {};
    for (const [fid, cols] of groups) {
      let col = cols.length > 1 ? mergeColumns(cols) : cols[0];
      const team = ctx.overrides.get(fid);
      if (team && col.team !== team) col = cols.length > 1 ? Object.assign(col, {team}) : {...col, team};
      ids[fid] = col;
    }
    block.ids = ids;
  }
  // the identity lists
  if (view.match) {
    const people = renameEntries(base.match.players, ctx), keepers = renameEntries(base.match.goalkeepers, ctx);
    const inKeepers = new Set(keepers.kept.map((e) => e.id));
    view.match.players = people.kept.filter((e) => !inKeepers.has(e.id) || e.id === null);
    view.match.goalkeepers = keepers.kept;
    view.match.excluded = [...base.match.excluded.map((e) => ({...e})), ...people.gone, ...keepers.gone];
  }
  if (view.roster && base.slots) view.roster.slots = renameEntries(base.slots, ctx).kept;
  // visibility per frame (the roster strip under the board)
  const renamed = (k) => ctx.resolve(k) !== k || ctx.out(k, ctx.resolve(k));
  view.frames.forEach((f, i) => {
    const v0 = base.visibility[i];
    if (!v0) return;
    let touched = false;
    for (const k in v0) if (renamed(k)) { touched = true; break; }
    if (!touched) { if (f.visibility !== v0) f.visibility = v0; return; }
    const v = {};
    for (const k in v0) {
      const fid = ctx.resolve(k);
      if (ctx.out(k, fid)) continue;
      if (k === fid || !(fid in v)) v[fid] = v0[k];
    }
    f.visibility = v;
  });
  if (base.labels) {
    const labels = {};
    for (const [a, v] of Object.entries(base.labels)) { const fid = ctx.resolve(a); if (ctx.out(a, fid)) continue; if (a === fid || !(fid in labels)) labels[fid] = v; }
    view.labels = labels;
  }
  for (const e of view.events || []) {
    const a = e[ORIG] !== undefined ? e[ORIG] : e.playerId;
    if (a === null || a === undefined) continue;
    const fid = ctx.resolve(a);
    if (fid !== e.playerId) { if (e[ORIG] === undefined) keepOriginal(e, e.playerId); e.playerId = fid; }
  }
  return view;
}

// ---------- the match player ----------
function adoptIndex(m, index) {
  m.index = index;
  m.segs = (Array.isArray(index.segments) ? index.segments : []).filter((s) => s && isNum(s.start) && isNum(s.end)).slice().sort((a, b) => a.start - b.start);
  m.ctx = contextOf(index);
  m.ctxGen++;
}
function fpsOf(m) {
  const v = (m.index && m.index.video) || {};
  return Number(v.fps) > 0 ? Number(v.fps) : (hasVideo() && Number(info.video.fps) > 0 ? Number(info.video.fps) : 25);
}
function ownerAt(m, t) {
  const segs = m.segs;
  for (let i = 0; i < segs.length; i++) { const s = segs[i]; if (t < s.start) return -1; if (t < s.end) return i; }
  return -1;
}
function covers(m, t) {
  // the shown view has frames at t (its context beyond the cut)
  if (m.current < 0) return false;
  const s = m.segs[m.current], tol = 1.6 / fpsOf(m);
  return isNum(s.firstTime) && isNum(s.lastTime) && t >= s.firstTime - tol && t <= s.lastTime + tol;
}
function pitchIndexOf(view) {
  const out = [];
  let last = -1;
  view.frames.forEach((f, i) => { if (f.pitch) last = i; out.push(last); });
  return out;
}
function presenceOf(view) {
  // the page's buildPresence() on another result, the shown one untouched
  const keepResult = result, keepPresence = presence;
  try { result = view; buildPresence(); return presence; }
  finally { result = keepResult; presence = keepPresence; }
}
function boardTakesTimeline() {
  return !!tactical && typeof tactical.syncPositionControl === 'function' && typeof tactical.updateStatus === 'function' && 'timeline' in tactical;
}
function keep(m, k, view) {
  m.views.set(k, view);
  while (m.views.size > MAX_VIEWS) {
    let worst = -1, far = -1;
    const near = m.owner >= 0 ? m.owner : m.current;
    for (const j of m.views.keys()) {
      if (j === m.current || j === m.owner || j === k) continue;
      const d = Math.abs(j - near);
      if (d > far) { far = d; worst = j; }
    }
    if (worst < 0) break;
    m.views.delete(worst); m.prepared.delete(worst);
  }
}
async function fetchView(m, k, signal, kind) {
  const seg = m.segs[k], t0 = performance.now();
  const response = await fetch(`/api/match/view?id=${encodeURIComponent(m.id)}&segment=${encodeURIComponent(seg.index)}`, {signal});
  if (!response.ok) {
    let message = `${response.status} ${response.statusText}`;
    try { const body = await response.json(); if (body && body.error) message = body.error; } catch (e) { /* not JSON */ }
    throw new Error(message);
  }
  const body = await response.text();
  const t1 = performance.now();
  const view = JSON.parse(body);
  const t2 = performance.now();
  if (!view || view.version !== 3 || !view.match || !Array.isArray(view.frames) || !view.frames.length) throw new Error('the server sent no usable view');
  if (view.matchView && isNum(view.matchView.segment) && view.matchView.segment !== seg.index) throw new Error('the server sent another segment');
  renameView(view, m.ctx);
  view[ORIG].ctxGen = m.ctxGen;
  const t3 = performance.now();
  m.loads.push({segment: k, kind, transferMs: Math.round(t1 - t0), parseMs: Math.round(t2 - t1), renameMs: Math.round(t3 - t2), megabytes: Math.round(body.length / 1e5) / 10, frames: view.frames.length});
  if (m.loads.length > 50) m.loads.shift();
  return view;
}
function load(m, k, kind) {
  // The view of segment k: kept, in flight, or fetched now. A demand cancels the other loads (a seek moved on); a
  // look-ahead waits while anything is in flight (one 80-100 MB parse at a time).
  if (k < 0 || !m.segs || k >= m.segs.length) return null;
  if (m.views.has(k)) return Promise.resolve(m.views.get(k));
  const failed = m.failed.get(k);
  if (failed && Date.now() - failed < RETRY_MS) return null;
  const live = m.inflight.get(k);
  if (live) { if (kind === 'demand') live.kind = 'demand'; return live.promise; }
  if (kind === 'demand') { for (const [j, other] of m.inflight) { other.ctrl.abort(); m.inflight.delete(j); } }
  else if (m.inflight.size) return null;
  const entry = {k, kind, ctrl: new AbortController(), promise: null};
  m.inflight.set(k, entry);
  entry.promise = fetchView(m, k, entry.ctrl.signal, kind).then((view) => {
    if (m.inflight.get(k) === entry) m.inflight.delete(k);
    if (player !== m) return null;
    if (view[ORIG].ctxGen !== m.ctxGen) { renameView(view, m.ctx); view[ORIG].ctxGen = m.ctxGen; }   // a correction meanwhile
    m.failed.delete(k); m.error = null;
    keep(m, k, view);
    if (k !== m.owner) prepare(m, k);
    return view;
  }, (error) => {
    if (m.inflight.get(k) === entry) m.inflight.delete(k);
    if (player !== m || (error && error.name === 'AbortError')) return null;
    m.failed.set(k, Date.now());
    m.error = `Segment ${k + 1}: ${error.message || error}`;
    console.warn('Full match: a view could not be loaded:', m.error);
    return null;
  });
  return entry.promise;
}
function prepare(m, k) {
  // A view fetched ahead: its board timeline (the stabilized tracks too), presence and pitch index built in idle time,
  // in three steps, so that the swap at the cut only assigns.
  const view = m.views.get(k), gen = m.ctxGen;
  if (!view || (m.prepared.has(k) && m.prepared.get(k).view === view && m.prepared.get(k).gen === gen)) return;
  const valid = () => player === m && m.views.get(k) === view && m.ctxGen === gen && m.current !== k;
  const out = {view, gen, ms: 0};
  const steps = [
    () => { out.timeline = Tactical.buildTimeline(view, info); },
    () => { void out.timeline.stable; },
    () => { out.pitchIndex = pitchIndexOf(view); out.presence = presenceOf(view); },
  ];
  let i = 0;
  const step = () => {
    if (!valid()) return;
    const t = performance.now();
    try { steps[i++](); } catch (error) { console.warn('Full match: preparing a view failed', error); return; }
    out.ms += performance.now() - t;
    if (i < steps.length) idle(step);
    else { out.ms = Math.round(out.ms); m.prepared.set(k, out); }
  };
  idle(step);
}
function swap(m, k, why) {
  // The light swap: the shown result becomes view k (no showResult: the panels keep the match-level object, nothing is
  // seeked, the selection stays).
  const view = m.views.get(k);
  if (!view) return false;
  const prep = m.prepared.get(k), ready = !!prep && prep.view === view && prep.gen === m.ctxGen && boardTakesTimeline();
  console.time('match view swap');
  const t0 = performance.now();
  result = view;
  if (ready) {
    tactical.load(null, info);              // the board's per-result state reset (pitch layer, cache key, settings)
    tactical.timeline = prep.timeline;
    tactical.syncPositionControl(); tactical.updateStatus();
    presence = prep.presence; pitchIndex = prep.pitchIndex;
  } else {
    tactical.load(view, info);
    buildPresence();
    pitchIndex = pitchIndexOf(view);
  }
  closeEditor();
  if (callPage('coachMarkRebase') !== true) clearCoachMark();   // a key moment being watched stays (its key frame is looked up in this view)
  m.current = k;
  const ms = performance.now() - t0;
  console.timeEnd('match view swap');
  if (!ready && tactical.timeline) m.prepared.set(k, {view, gen: m.ctxGen, timeline: tactical.timeline, presence, pitchIndex, ms: Math.round(ms)});   // a swap back is light
  m.swaps.push({segment: k, ms: Math.round(ms * 100) / 100, prepared: ready, why: why || 'cut', at: Math.round(video.currentTime * 100) / 100});
  if (m.swaps.length > 200) m.swaps.shift();
  idle(() => { if (player === m && m.current === k) { callPage('eventList'); callPage('pitchIssueList'); } });
  stageChanged(m, true);
  return true;
}
function waiting(t) {
  // The moment player waits before it plays or places a mark: true while the view of the segment owning t is not the
  // one shown and the shown one has no frames at t (render() is fetching it), also while the match is opening. A time
  // no segment owns (a break, outside the match) has nothing to wait for.
  const m = player;
  if (!m) return false;
  if (!m.segs || m.current < 0) return true;
  const k = ownerAt(m, t);
  return k >= 0 && k !== m.current && !covers(m, t);
}
function note(m, text) {
  const el = byId('matchNote');
  if (!el) return;
  if (el.textContent !== text) el.textContent = text;
  if (el.hidden !== !text) el.hidden = !text;
}
function stageChanged(m, force) {
  const key = `${m.current}|${m.owner}|${m.segs ? 1 : 0}`;
  if (!force && key === m.stageKey) return;
  m.stageKey = key;
  callPage('stageStatus');
  paintReady();
}
function frame(t) {
  // Every render: the view of the segment that owns t, the look-ahead, the notice. Returns the time to draw (OUTSIDE:
  // nothing).
  const m = player;
  if (!m) return t;
  if (!m.segs) { note(m, 'Loading match data…'); return OUTSIDE; }
  if (isNum(m.lastT)) { if (t < m.lastT - 1e-3) m.dir = -1; else if (t > m.lastT + 1e-3) m.dir = 1; }
  m.lastT = t;
  const k = ownerAt(m, t);
  m.owner = k;
  if (k >= 0 && k !== m.current) {
    if (m.views.has(k)) swap(m, k, video.paused ? 'seek' : 'playback'); else load(m, k, 'demand');
  }
  let shown = OUTSIDE, text = '';
  if (k >= 0) {
    if (k === m.current) shown = t;
    else if (covers(m, t)) shown = t;    // the owner's view is on its way: the shown one's frames beyond its cut meanwhile
    else text = m.failed.has(k) ? `Match data for segment ${k + 1} could not be loaded (${m.error || 'error'}); retrying…` : 'Loading match data…';
    if (k === m.current) {
      const s = m.segs[k];
      if (s.end - t <= PREFETCH_S && k + 1 < m.segs.length) load(m, k + 1, 'ahead');
      else if (m.dir < 0 && t - s.start <= PREFETCH_S && k > 0) load(m, k - 1, 'ahead');
    }
  } else {
    text = gapText(m, t);
    const next = m.segs.findIndex((s) => s.start > t);
    if (next >= 0 && m.segs[next].start - t <= PREFETCH_S) load(m, next, 'ahead');
  }
  note(m, text);
  stageChanged(m, false);
  reviewTick(m, t);
  return shown;
}
function rosterPeople(m) {
  // the match's identities after the corrections: merged ones (aliases) are in their final identity
  return ((m.index && m.index.roster) || []).filter((p) => p && typeof p.matchId === 'string' && m.ctx.resolve(p.matchId) === p.matchId);
}
function matchObject(m) {
  // What the panels (analytics, team shape, coaching, possession) load once per entry or correction: the match's people
  // and sparse frame anchors over every segment's owned frames; the server answers their requests for the match.
  const fps = fpsOf(m), people = rosterPeople(m);
  // a short fragment (display A-?) is listed but not confirmed: the server analyses numbered identities only
  // (fullmatch/analytics.roster_people), so the analytics player list leaves it out
  const entry = (p) => ({id: p.matchId, display: p.display || p.matchId, team: p.team === 'A' || p.team === 'B' ? p.team : null,
                         role: p.role === 'GOALKEEPER' ? 'GOALKEEPER' : 'PLAYER', status: 'ACTIVE', firstSeen: p.firstSeen, lastSeen: p.lastSeen,
                         confirmed: !String(p.display || '').endsWith('?')});
  const active = people.filter((p) => !m.ctx.out(p.matchId, p.matchId));
  const players = active.filter((p) => p.role !== 'GOALKEEPER').map(entry), keepers = active.filter((p) => p.role === 'GOALKEEPER').map(entry);
  const excluded = people.filter((p) => m.ctx.out(p.matchId, p.matchId)).map((p) => ({...entry(p), status: 'EXCLUDED'}));
  const frames = [];
  for (const s of m.segs) {
    if (!isNum(s.cut0) || !isNum(s.cut1)) continue;
    for (let f = s.cut0; f < s.cut1; f += ANCHOR_STEP) frames.push({frame: f, time: s.start + (f - s.cut0) / fps, people: []});
    const last = s.cut1 - 1;
    if (last > s.cut0 && (last - s.cut0) % ANCHOR_STEP) frames.push({frame: last, time: s.start + (last - s.cut0) / fps, people: []});
  }
  // the goal ends of the first period: the coach's directions when the match has them (directionPeriods), else the plan's
  const directions = directionPeriodsOf(m), first = m.first || {}, tactical0 = {...(first.tactical || {})};
  const p0 = ((directions && directions.length ? directions : m.index.periods) || [])[0] || {};
  if (p0.defendsLeft === 'A' || p0.defendsLeft === 'B') tactical0.goalEnds = {left: p0.defendsLeft, right: p0.defendsLeft === 'A' ? 'B' : 'A'};
  return {version: 3, video: m.index.video || first.video, matchLevel: {matchId: m.id, segments: m.segs.length},
          match: {players, goalkeepers: keepers, referees: [], retired: [], excluded, withdrawn: []},
          roster: {slots: [...players, ...keepers].map((e) => ({...e})), expected: {A: 11, B: 11, goalkeepers: 2}},
          frames, tactical: tactical0, summary: {}, teams: first.teams || {}};
}
// The match's attacking directions by period (halftime.periods: from the coach's halftime and first-half direction):
// the match state's own as soon as it says them (a halftime just saved), else the index's.
function directionPeriodsOf(m) {
  const fromIntel = mi.m === m && mi.intel && Array.isArray(mi.intel.directionPeriods) ? mi.intel.directionPeriods : null;
  const fromIndex = m.index && Array.isArray(m.index.directionPeriods) ? m.index.directionPeriods : null;
  return (fromIntel && fromIntel.length ? fromIntel : fromIndex) || null;
}
function goalEndsAt(m, t) {
  // The board's goal-end labels at time t: the team attacking right in that period defends the LEFT goal. Between two
  // periods (a halftime break) nobody defends either end ({}); null (the segment view's own labels) when the match
  // says no direction.
  if (player !== m || !isNum(t)) return null;
  const periods = (directionPeriodsOf(m) || []).filter((p) => p && isNum(p.start) && isNum(p.end));
  if (!periods.length) return null;
  const p = periods.find((q) => t >= q.start - 1e-6 && t <= q.end + 1e-6);
  if (!p) return t > periods[0].start && t < periods[periods.length - 1].end ? {} : null;
  return p.attacksRight === 'A' || p.attacksRight === 'B' ? {left: p.attacksRight, right: p.attacksRight === 'A' ? 'B' : 'A'} : null;
}
function boardEnds(m) {
  // in match playback the 2D board names the goal ends by time (they change at halftime and follow the coach's
  // direction), not by each segment view's tactical.goalEnds; only the labels: no coordinate is flipped
  try { if (tactical && typeof tactical.setGoalEnds === 'function') tactical.setGoalEnds(m ? (t) => goalEndsAt(m, t) : null); } catch (e) { /* an older board keeps the views' labels */ }
}
function relabelRanges(on) {
  // the panels' "whole range" options say the full match while it is open
  const items = [['anRange', 'full', 'Full match (all segments)'], ['tsRange', 'full', 'Entire full match'], ['crRange', 'whole', 'Whole match']];
  if (on && !st.labels) st.labels = {};
  for (const [id, value, label] of items) {
    const option = byId(id) && [...byId(id).options].find((o) => o.value === value);
    if (!option) continue;
    if (on) { if (!(id in st.labels)) st.labels[id] = option.textContent; option.textContent = label; }
    else if (st.labels && id in st.labels) option.textContent = st.labels[id];
  }
  if (!on) st.labels = null;
}
const ESTIMATE = /[^-?]\?$/;   // a provisional slot label ('A-04?', 'GK-A?': the best estimate of the place), not 'A-?'
function analyticsObject(m) {
  // The analytics panel's copy of the match object: a provisional person (a slot's best estimate, 'A-04?') is
  // analysable too (the server answers any person of the match), named as an estimate with when it was seen, so that
  // the list, sorted by its labels, groups the people of a slot ('A-04', then 'A-04? (estimate) \u00b7 12:30-15:10', ...).
  // The unresolved ('A-?', no slot) stay out of the list, as do the people left out. The other panels keep the object.
  const o = m.object, seen = new Map(rosterPeople(m).map((p) => [p.matchId, p]));
  const relabel = (e) => {
    const d = String(e.display || ''), p = seen.get(e.id);
    if (!ESTIMATE.test(d)) return e;
    const span = p && isNum(p.firstSeen) && isNum(p.lastSeen) ? ` \u00b7 ${clockText(p.firstSeen)}\u2013${clockText(p.lastSeen)}` : '';
    return {...e, confirmed: true, estimate: true, display: `${d} (estimate)${span}`};
  };
  return {...o, match: {...o.match, players: o.match.players.map(relabel), goalkeepers: o.match.goalkeepers.map(relabel)}};
}
function loadPanels(m) {
  m.object = matchObject(m);
  relabelRanges(true);
  for (const [name, panel] of [['analytics', analytics], ['team shape', teamShape], ['coaching', coaching], ['possession', possession], ['ball possession', statsPanel()]]) {
    if (!panel) continue;
    try {
      const answer = panel.load(name === 'analytics' ? analyticsObject(m) : m.object, info);
      if (answer && typeof answer.catch === 'function') answer.catch((error) => console.warn(`Full match: ${name}:`, error));
    } catch (error) {
      console.warn(`Full match: the ${name} panel could not load the match:`, error);
    }
  }
}
function summaryText(m) {
  const idx = m.index, people = rosterPeople(m), count = (pred) => people.filter(pred).length;
  const cov = idx.coverage || {}, periods = idx.periods || [];
  const corrections = Array.isArray(idx.corrections) ? idx.corrections.length : 0;
  return `Full match ${m.id} · ${plural(m.segs.length, 'segment')} over ${span()} (${periods.length > 1 ? `${periods.length} periods` : 'one period'}) · assembled ${whenText(idx.assembledAt)}.\n` +
    `Match identities: team A ${count((p) => p.team === 'A' && p.role !== 'GOALKEEPER' && !p.excluded)}, team B ${count((p) => p.team === 'B' && p.role !== 'GOALKEEPER' && !p.excluded)}, goalkeepers ${count((p) => p.role === 'GOALKEEPER' && !p.excluded)}` +
    `${count((p) => p.team !== 'A' && p.team !== 'B' && !p.excluded) ? `, not assigned to a team ${count((p) => p.team !== 'A' && p.team !== 'B' && !p.excluded)}` : ''}${count((p) => p.excluded) ? `, not players ${count((p) => p.excluded)}` : ''} (stitched over the segments' overlaps${corrections ? `; ${plural(corrections, 'correction')} by you` : ''}).\n` +
    `${isNum(cov.seconds) ? `${spanText(cov.seconds)} of match analysed` : ''}${isNum(cov.underConstrainedShare) ? `; the pitch registration is under-constrained on ${Math.round(cov.underConstrainedShare * 100)}% of it` : ''}.\n` +
    'The views of the segments are shown as they were processed; the shown frame belongs to the segment that owns it.';
}
function seekInto(t) { try { video.currentTime = Math.max(0, Math.min(isNum(video.duration) ? video.duration : t, t)); } catch (e) { /* no media yet */ } }
async function open(id) {
  if (player || st.acting) return;
  id = id || (st.latest && st.latest.matchId);
  if (!id || !hasVideo()) return;
  const m = {id, index: null, segs: null, ctx: null, ctxGen: 0, views: new Map(), prepared: new Map(), inflight: new Map(), failed: new Map(),
             current: -1, owner: -1, lastT: null, dir: 0, swaps: [], loads: [], error: null, object: null, first: null, stageKey: null, opened: false};
  player = m; global.matchMode = m;
  document.body.classList.add('match-mode');
  st.acting = 'open'; st.message = ''; st.messageError = false;
  callPage('closeEditor'); callPage('clearCoachMark');
  paint(); stageChanged(m, true);
  try {
    const index = await api(`/api/match/index?id=${encodeURIComponent(id)}`);
    if (player !== m) return;
    if (!index || !Array.isArray(index.segments) || !index.segments.length) throw new Error('the match index has no segments');
    adoptIndex(m, index);
    intelOpen(m);   // the match state (timeline, events, halftime) is asked for while the first view loads
    boardEnds(m);
    // the video's time: inside the match; before or after it, its start; in a break, the next period
    let k = ownerAt(m, video.currentTime);
    if (k < 0) {
      const next = m.segs.findIndex((s) => s.start > video.currentTime);
      if (next > 0) k = next;
      else { k = 0; seekInto(m.segs[0].start + .5 / fpsOf(m)); }
    }
    m.owner = k;
    // the first view (a seek meanwhile asks for another one: then that one is waited for; render() may swap it in first)
    for (let tries = 0; player === m && m.current < 0 && tries < 6; tries++) {
      const wanted = m.owner >= 0 ? m.owner : k, asked = load(m, wanted, 'demand');
      const view = asked ? await asked : null;
      if (player !== m) return;
      if (m.current >= 0) break;
      if (view && m.views.get(wanted) === view) { swap(m, wanted, 'open'); break; }
      if (!asked || m.failed.has(wanted)) break;
    }
    if (player !== m) return;
    if (m.current < 0) throw new Error(m.error || 'the first view did not load');
    const shown = m.views.get(m.current);
    m.first = {tactical: shown.tactical, teams: shown.teams, video: shown.video};
    loadPanels(m);
    const download = byId('download');
    if (download) download.hidden = true;   // the tracking JSON of the Segment run: not this match
    setText(byId('summary'), summaryText(m));
    callPage('roster'); callPage('correctionList'); callPage('matchMeta'); callPage('stageStatus');
    m.opened = true;
  } catch (error) {
    if (player === m) {
      leave({reload: true});
      st.message = `The full match could not be opened: ${error.message || error}`; st.messageError = true;
    }
  } finally {
    if (st.acting === 'open') st.acting = null;
    paint();
  }
}
function leave(options) {
  const m = player, reload = !options || options.reload !== false;
  if (!m) return;
  intelStop();
  boardEnds(null);
  player = null; global.matchMode = null;
  for (const entry of m.inflight.values()) entry.ctrl.abort();
  m.inflight.clear(); m.views.clear(); m.prepared.clear();
  document.body.classList.remove('match-mode');
  note(null, '');
  callPage('closeEditor'); callPage('clearCoachMark');
  if (m.current < 0) {
    // no view was shown (the opening failed or was cancelled): the Segment result and the panels are still the page's
    let shown = null;
    try { shown = result; } catch (e) { shown = null; }
    if (shown) callPage('roster');
    callPage('correctionList'); callPage('matchMeta'); callPage('stageStatus');
    paint();
    return;
  }
  // nothing is shown until the Segment result is loaded again the normal way (poll)
  result = null;
  try { tactical.load(null, info); } catch (e) { console.error(e); }
  callPage('buildPresence');
  pitchIndex = [];
  for (const panel of [analytics, teamShape, coaching, possession, statsPanel()]) { if (!panel) continue; try { panel.load(null, info); } catch (e) { console.warn(e); } }
  relabelRanges(false);
  const box = byId('roster');
  if (box) box.textContent = 'No result yet.';
  setText(byId('summary'), 'Run a segment to inspect detections and identities. No accuracy score is claimed.');
  const events = byId('events');
  if (events) events.replaceChildren(mk('span', 'muted', 'Run a segment to see identity decisions.'));
  callPage('pitchIssueList'); callPage('correctionList'); callPage('matchMeta'); callPage('stageStatus');
  paint();
  if (reload) {
    loadedRun = null;
    Promise.resolve().then(() => poll()).then(() => {
      if (player || !result || !Array.isArray(result.frames) || !result.frames.length) return;
      if (frameIndex(video.currentTime) < 0) { const first = result.frames.find((f) => f.people && f.people.length) || result.frames[0]; seekInto(first.time); }
    }).catch((error) => console.error('Full match: reloading the segment result failed', error));
  }
}

// ---------- corrections (the roster drawer in match playback) ----------
const drawerState = {busy: false, message: '', error: false, open: new Set(), rowError: new Map()};
// how an identity is named in the drawer: its display, and its id when they differ (several fragments show A-?); a
// fragment left out by its fragment id (an internal track id, never shown as a slot label)
function nameOf(p) { return p ? (p.excluded ? `fragment ${p.matchId}` : p.display && p.display !== p.matchId ? `${p.display} \u00b7 ${p.matchId}` : p.matchId) : '?'; }
function displayOf(m, id) { const r = m.ctx.roster.get(m.ctx.resolve(id)); return r ? nameOf(r) : id; }
function applyIndex(m, index) {
  // A correction made or undone: the new index renames the loaded views in place; the shown one is swapped again (its
  // board, presence and pitch index), the others prepared again, the panels get the new match-level object.
  if (!index || !Array.isArray(index.segments)) return;
  const t0 = performance.now();
  adoptIndex(m, index);
  for (const view of m.views.values()) { renameView(view, m.ctx); view[ORIG].ctxGen = m.ctxGen; }
  m.prepared.clear();
  if (m.current >= 0 && m.views.has(m.current)) swap(m, m.current, 'correction');
  for (const k of m.views.keys()) if (k !== m.current) prepare(m, k);
  try {
    if (selected) {
      const f = m.ctx.resolve(selected);
      if (m.ctx.out(selected, f)) selected = null;
      else if (f !== selected) select(f);
    }
  } catch (e) { /* the page has no selection yet */ }
  loadPanels(m);
  setText(byId('summary'), summaryText(m));
  callPage('roster'); callPage('correctionList');
  m.applyMs = Math.round(performance.now() - t0);
  intelIndexChanged(m);   // the match state follows the people (made again when a correction changed the slots)
}
async function indexAfter(m, answer) {
  // the index a correction route answers with (asked again when the answer has none)
  if (answer && answer.index && Array.isArray(answer.index.segments)) return answer.index;
  return api(`/api/match/index?id=${encodeURIComponent(m.id)}`);
}
async function correct(m, body, done, rowId) {
  if (drawerState.busy || player !== m) return;
  drawerState.busy = true; drawerState.message = ''; drawerState.error = false;
  if (rowId) drawerState.rowError.delete(rowId);
  callPage('roster');
  try {
    const answer = await api('/api/match/correction', {match: m.id, ...body});
    const index = await indexAfter(m, answer);
    if (player !== m) return;
    drawerState.message = done; drawerState.busy = false;
    if (rowId) {
      drawerState.open.delete(rowId);
      for (const k of [...mi.rv.open]) if (k === rowId || k.startsWith(`${rowId}:`)) mi.rv.open.delete(k);   // a roster review row's tools close too
    }
    applyIndex(m, index);
  } catch (error) {
    if (player !== m) return;
    if (rowId) drawerState.rowError.set(rowId, error.message || String(error));
    else { drawerState.message = error.message || String(error); drawerState.error = true; }
  } finally {
    drawerState.busy = false;
    if (player === m) callPage('roster');
  }
}
async function undo(m, c) {
  if (drawerState.busy || player !== m) return;
  drawerState.busy = true; drawerState.message = ''; drawerState.error = false;
  callPage('correctionList');
  try {
    const answer = await api('/api/match/correction/remove', {match: m.id, id: c.id});
    const index = await indexAfter(m, answer);
    if (player !== m) return;
    drawerState.message = `Undone: ${correctionText(m, c)}.`; drawerState.busy = false;
    applyIndex(m, index);
  } catch (error) {
    if (player !== m) return;
    drawerState.message = error.message || String(error); drawerState.error = true;
  } finally {
    drawerState.busy = false;
    if (player === m) { callPage('roster'); callPage('correctionList'); }
  }
}
function correctionText(m, c) {
  // a correction names fragments (the server keeps a person's first fragment): written as fragments, with the person
  // each one is in now
  const frags = m.index && Array.isArray(m.index.fragments) ? fragmentMap(m) : null;
  const who = (id) => {
    if (!id) return '?';
    if (frags && frags.has(id)) {
      const r = m.ctx.roster.get(m.ctx.resolve(id));
      return r && !r.excluded ? `fragment ${id} (${nameOf(r)})` : `fragment ${id}`;
    }
    return m.ctx.roster.has(id) ? displayOf(m, id) : id;
  };
  // a correction aimed at a person carries the person's other fragments ('also') and applies to all of them
  const also = Array.isArray(c.also) && c.also.length ? ` and its ${plural(c.also.length, 'other fragment')}` : '';
  const whom = (id) => `${who(id)}${also}`;
  if (c.kind === 'merge') return `${whom(c.target)} joined into ${who(c.into)} (same player)`;
  if (c.kind === 'exclude') return `${whom(c.target)} marked as not a player`;
  if (c.kind === 'include') return `${whom(c.target)} counted as a player again`;
  if (c.kind === 'team') return `${whom(c.target)} moved to team ${c.team}`;
  if (c.kind === 'keeper') return `${whom(c.target)} ${c.on === false ? 'is not' : 'is'} a goalkeeper`;
  if (c.kind === 'split') return `${who(c.target)} is not the same player as ${(Array.isArray(c.from) ? c.from : []).map(who).join(' or ') || '?'}`;
  if (c.kind === 'confirm') return `${whom(c.target)} is the player of slot ${isNum(c.slot) ? pad2(c.slot) : '?'}`;
  if (c.kind === 'substitution') return `Substitution ${who(c.out)} \u2192 ${who(c.in)} ${c.status === 'rejected' ? 'rejected' : 'confirmed'}`;
  return `${c.kind || 'correction'} of ${who(c.target)}`;
}
function seekIdentity(m, id) {
  const r = m.ctx.roster.get(m.ctx.resolve(id));
  if (!r) return;
  try { video.pause(); } catch (e) { /* no media */ }
  if (isNum(r.firstSeen)) seekInto(r.firstSeen);
  if (!m.ctx.out(r.matchId, r.matchId)) callPage('select', r.matchId);
}
function teamWord(team) { return team === 'A' || team === 'B' ? `Team ${team}` : 'No team'; }
function personMeta(m, p) {
  const parts = [teamWord(p.team) + (p.teamCorrected ? ' (set by you)' : ''), `${spanText(p.seconds)} observed`];
  if (isNum(p.firstSeen) && isNum(p.lastSeen)) parts.push(`${clockText(p.firstSeen)}–${clockText(p.lastSeen)}`);
  const how = p.how ? `${p.how}${p.confidence ? ` (${p.confidence} confidence)` : ''}` : p.confidence ? `${p.confidence} confidence` : '';
  if (how) parts.push(how);
  if (Array.isArray(p.mergedFrom) && p.mergedFrom.length) parts.push(`joined from ${p.mergedFrom.join(', ')}`);
  return parts.join(' · ');
}
function candidates(m, p) {
  // the identities it could be the same player as: its team (any, without one), nearest in time first
  const list = rosterPeople(m).filter((q) => q.matchId !== p.matchId && !m.ctx.out(q.matchId, q.matchId) && (p.team !== 'A' && p.team !== 'B' ? true : q.team === p.team));
  const gap = (q) => {
    if (!isNum(p.firstSeen) || !isNum(p.lastSeen) || !isNum(q.firstSeen) || !isNum(q.lastSeen)) return Infinity;
    if (q.firstSeen > p.lastSeen) return q.firstSeen - p.lastSeen;
    if (p.firstSeen > q.lastSeen) return p.firstSeen - q.lastSeen;
    return -1;   // their times overlap: rarely the same player
  };
  return list.map((q) => ({q, g: gap(q)})).sort((a, b) => (a.g < 0) - (b.g < 0) || a.g - b.g || naturally(a.q.display, b.q.display));
}
function suggestionsOf(m, p) {
  const raw = (m.index.suggestions && m.index.suggestions[p.matchId]) || [];
  return (Array.isArray(raw) ? raw : []).filter((s) => s && s.into && m.ctx.resolve(s.into) !== p.matchId && m.ctx.roster.has(m.ctx.resolve(s.into)) &&
    !m.ctx.out(s.into, m.ctx.resolve(s.into))).slice(0, 3);
}
function personRow(m, p) {
  const row = mk('div', 'mr' + (m.ctx.out(p.matchId, p.matchId) ? ' out' : '')), out = m.ctx.out(p.matchId, p.matchId);
  row.dataset.person = p.matchId;
  const head = mk('div', 'mr-head'), who = btn(nameOf(p), () => seekIdentity(m, p.matchId), 'mr-who');
  who.dataset.id = p.matchId;
  who.title = `${p.matchId} · jump to the first observation (${clockText(p.firstSeen)})`;
  try { if (selected === p.matchId) who.classList.add('active'); } catch (e) { /* no selection yet */ }
  head.append(who, mk('span', 'mr-meta', personMeta(m, p)));
  row.append(head);
  if (p.substitution && p.substitution.status === 'possible') {
    const notes = Array.isArray(p.substitution.notes) ? p.substitution.notes.join('; ') : '';
    row.append(mk('p', 'mr-sub', `Possible substitution${notes ? `: ${notes}` : ''}. ${p.substitution.note || ''}`.trim()));
  }
  if (out) return row;
  // the best suggestion on the row; the weaker ones inside its Correct panel (they are weak evidence: the coach decides)
  const suggestions = suggestionsOf(m, p), suggestion = (s) => {
    const line = mk('div', 'mr-sugg'), into = m.ctx.resolve(s.into);
    const facts = [isNum(s.gapSeconds) ? `starts ${Math.round(s.gapSeconds)} s later` : null, isNum(s.metres) ? `${Math.round(s.metres)} m away` : null].filter(Boolean).join(', ');
    line.append(mk('span', null, `Possibly the same as ${displayOf(m, into)}${facts ? ` (${facts})` : ''}`));
    const join = btn('Join', () => correct(m, {kind: 'merge', target: p.matchId, into: s.into}, `Joined ${nameOf(p)} and ${displayOf(m, into)}: one player.`, p.matchId));
    join.title = `Join ${nameOf(p)} and ${displayOf(m, into)} as one player`;
    join.disabled = drawerState.busy;
    const look = btn('Show', () => seekIdentity(m, into), 'quiet');
    look.title = `Jump to ${displayOf(m, into)}'s first observation`;
    line.append(join, look);
    return line;
  };
  if (suggestions.length) row.append(suggestion(suggestions[0]));
  const open = drawerState.open.has(p.matchId), toggle = btn(open ? 'Close' : 'Correct', () => {
    if (drawerState.open.has(p.matchId)) drawerState.open.delete(p.matchId); else drawerState.open.add(p.matchId);
    callPage('roster');
  }, 'quiet mr-edit');
  toggle.setAttribute('aria-expanded', String(open));
  head.append(toggle);
  if (open) {
    const tools = mk('div', 'mr-tools'), sel = mk('select'), label = mk('label', 'mr-same');
    sel.setAttribute('aria-label', `Same player as (for ${nameOf(p)})`);
    const first = mk('option', null, 'Same player as…'); first.value = '';
    sel.append(first);
    for (const {q, g} of candidates(m, p)) {
      const o = mk('option', null, `${nameOf(q)} · ${clockText(q.firstSeen)}–${clockText(q.lastSeen)}${g < 0 ? ' · overlaps in time' : ''}`);
      o.value = q.matchId;
      sel.append(o);
    }
    label.append(sel);
    const join = btn('Join', () => { if (sel.value) correct(m, {kind: 'merge', target: p.matchId, into: sel.value}, `Joined ${nameOf(p)} and ${displayOf(m, sel.value)}: one player.`, p.matchId); });
    join.disabled = true;
    sel.addEventListener('change', () => { join.disabled = !sel.value || drawerState.busy; });
    const exclude = btn('Not a player', () => correct(m, {kind: 'exclude', target: p.matchId}, `${nameOf(p)} is marked as not a player.`, p.matchId));
    exclude.title = 'Referee, staff, a spectator: hidden on the board and left out of the match analysis';
    tools.append(label, join, exclude);
    const teams = p.team === 'A' ? ['B'] : p.team === 'B' ? ['A'] : ['A', 'B'];
    for (const team of teams) {
      const move = btn(p.team === 'A' || p.team === 'B' ? `Change team to ${team}` : `Team ${team}`, () => correct(m, {kind: 'team', target: p.matchId, team}, `${nameOf(p)} moved to team ${team}.`, p.matchId));
      tools.append(move);
    }
    for (const b of tools.querySelectorAll('button')) if (b !== join) b.disabled = drawerState.busy;
    for (const s of suggestions.slice(1)) tools.append(suggestion(s));
    row.append(tools);
  }
  const error = drawerState.rowError.get(p.matchId);
  if (error) { const e = mk('p', 'mr-msg error', error); e.setAttribute('role', 'alert'); row.append(e); }
  return row;
}
function rosterDrawer(box) {
  const m = player;
  if (!m || !box) return false;
  box.replaceChildren();
  if (!m.index || !m.ctx) { box.append(mk('span', 'muted', 'Loading the match roster…')); return true; }
  const people = rosterPeople(m), out = (p) => m.ctx.out(p.matchId, p.matchId);
  const bar = mk('div', 'mr-bar');
  const all = btn('Show all', () => { try { selected = null; } catch (e) { /* no selection yet */ } document.querySelectorAll('#roster .active').forEach((e) => e.classList.remove('active')); });
  const fragments = people.filter((p) => !out(p)).length;
  bar.append(all, mk('span', 'muted', `${plural(fragments, 'match identity', 'match identities')} over ${plural(m.segs.length, 'segment')}. Fragments of one player can be joined.`));
  box.append(bar);
  if (drawerState.message) { const msg = mk('p', 'mr-msg' + (drawerState.error ? ' error' : ''), drawerState.message); msg.setAttribute('role', 'status'); box.append(msg); }
  if (drawerState.busy) box.append(mk('p', 'mr-msg', 'Saving the correction…'));
  // the roster review (the match's people and slots, once built): the slots at the playback time, the unresolved
  // fragments, the ones left out; the list of every identity below it on request
  if (hasPlaces(m)) {
    const rv = mk('div', 'rv');
    box.append(rv);
    reviewSection(m, rv);
    const count = people.filter((p) => !out(p)).length, toggle = btn(mi.rv.all ? 'Hide the list of every identity' : `Every match identity \u00b7 ${count}`, () => {
      mi.rv.all = !mi.rv.all;
      callPage('roster');
    }, 'quiet rv-all');
    toggle.dataset.act = 'all';
    toggle.setAttribute('aria-expanded', String(mi.rv.all));
    box.append(toggle);
    if (!mi.rv.all) return true;
  } else mi.rv.box = null;
  const sorted = people.slice().sort((a, b) => naturally(a.display || a.matchId, b.display || b.matchId));
  const groups = [['Team A players', (p) => !out(p) && p.role !== 'GOALKEEPER' && p.team === 'A'], ['Team B players', (p) => !out(p) && p.role !== 'GOALKEEPER' && p.team === 'B'],
                  ['Goalkeepers', (p) => !out(p) && p.role === 'GOALKEEPER'], ['Not assigned to a team', (p) => !out(p) && p.role !== 'GOALKEEPER' && p.team !== 'A' && p.team !== 'B'],
                  ['Not players (set by you)', out]];
  for (const [title, test] of groups) {
    const list = sorted.filter(test);
    if (!list.length) continue;
    box.append(mk('h3', null, `${title} · ${list.length}`));
    const wrap = mk('div', 'mr-list');
    for (const p of list) wrap.append(personRow(m, p));
    box.append(wrap);
  }
  return true;
}
function correctionsDrawer(box) {
  const m = player;
  if (!m || !box) return false;
  box.replaceChildren();
  if (!m.index || !m.ctx) { box.append(mk('span', 'muted', 'Loading…')); return true; }
  const list = Array.isArray(m.index.corrections) ? m.index.corrections : [];
  if (!list.length) box.append(mk('span', 'muted', hasPlaces(m) ? 'No match corrections yet. In the roster review above: join a fragment into the player it is, say whose slot it is, split one off a slot, mark someone as not a player, change a team or a goalkeeper, confirm or reject a substitution.' :
    'No match corrections yet. In Match identities above: join the fragments of one player, mark someone as not a player, or change a team.'));
  for (const c of list) {
    const row = mk('div', 'row'), u = btn('Undo', () => undo(m, c));
    u.disabled = drawerState.busy;
    row.append(mk('span', null, `${correctionText(m, c)}${c.at ? ` · ${whenText(c.at)}` : ''}`), u);
    box.append(row);
  }
  return true;
}

// ---------- the match's state (Full Match Intelligence): the transport's strip, markers and halftime; events; roster review ----------
/* What the server knows of the match beyond the tracking (fullmatch/intel.py: the state timeline and the events of
   gamestate.py, the halftime of halftime.py, the people and slots of persons.py), shown and corrected in match playback
   only. A Segment run shows nothing of it; a server without these routes (404) or a state that fails to build leaves
   everything else as it was.

   The transport. In match playback the one range (#timeline) maps the SOURCE video's time (min 0, max its duration);
   its thumb's centre travels inset by half the thumb (THUMB_PX) at each end, so timeToX / xToTime use that geometry
   and the layers of #mt (index.html, over the range) are laid out in the same inset lane: a marker at P% stands where
   the thumb stands at that time. The transport grows (.mt-on) by the lanes above the range.
     strip     under the range: the state timeline (consecutive intervals of one state merged), coloured by state
     markers   above it, in two rows: goals and substitutions large; kick-offs, the other restarts and the stoppages
               small. A removed event is not drawn; a candidate is hollow and dimmer. Hover or focus: a tooltip (type,
               video time, team with its kit label, status and confidence, the evidence); click: pause and seek through
               the page's playback controller (playback.seek, the range's own path). One tab stop: the arrow keys move
               along the markers. A goal stands out: a ball in its team's ring with a tick down to the track, the running
               score above it ('2-1', team A first) where the previous label leaves room, and the score and where its
               analysis is (the key-moments report's Goals) in its tooltip.
     filter    #mtf beside the range: only the kinds of events it shows are drawn (so only they are roved, named under
               "Also here" and seeked); its button says how many ("Goals only", "5 of 11") and opens a popover of one
               checkbox per type, the play-state band, Goals only / All / None. One choice for every match, in
               localStorage (MARKS_KEY), goals only by default; a change draws the strip and the markers again at once.
     halftime  a green line with its handle above the bar, "HALFTIME - mm:ss" (two, FIRST HALF END and SECOND HALF START,
               with the break shaded, when the break is included). Drag it (pointer events: mouse, pen and touch) or focus
               it: Left / Right 1 s, Shift 10 s, Page Up / Down 60 s, Home / End. Enter, or a click on the label, opens the
               halftime editor (#mtPop): the times typed (mm:ss, h:mm:ss or seconds), the break, which way team A attacks
               in the first half (and whether the goal ends and the kick-offs agree; directionCheck.note first), Save,
               Cancel (the edits) and Reset (the match plan's break). An edit is a draft ("not saved yet") until Save; then
               the server makes the state again ("Updating the match state..."): asked every INTEL_POLL_MS until ready,
               then the index is read again (applyIndex) and everything is drawn from the new state. Never past the match
               range (halftime.range), and strictly inside it as the server requires.
   The state is asked for when the match opens; when the index says it is missing or was made for other people
   (index.intel.state 'missing' or 'people', e.g. after a roster correction that changed the slots) it is built first
   (POST /api/match/intel/build) under a status line ("Recognising the match: <stage>").
   The events drawer (#matchEvents, beside the match identities): the shares of the match per state, the direction
   check, the events in time order with a type filter (seek, confirm, type, set to now, remove, the evidence), Add at
   current time, and the event corrections with Undo. Reviewing is never required.
   The roster review (the Match identities drawer, above the list of every identity): per team the slots 01-10 and GK at
   the playback time (index.places: the node covering it, else the last one before it), the person holding it, when it
   was last observed, how sure the slot is, the slot's substitutions; then the fragments no slot claims and the ones
   left out. Its corrections are roster corrections (correct(), undone in Corrections). */
const INTEL_POLL_MS = 2000;   // the match state while it is made
const THUMB_PX = 14;          // the range thumb's width (index.html .transport input[type=range]): its centre travels inset by half of it
const NUDGE_PCT = .8;         // large markers closer than this (% of the timeline) stand a few pixels apart
const SCORE_PX = 28;          // a goal's running score stands above it when the previous score label is this far away (px) ...
const SCORE_PCT = 3;          // ... or this far (% of the timeline) while the lane's width is not known
const PANELS_MS = 600;      // the panels get the match again this long after the last event correction
const HALF_KEYS = {ArrowLeft: -1, ArrowDown: -1, ArrowRight: 1, ArrowUp: 1, PageDown: -60, PageUp: 60};
const STATE_CLASS = {IN_PLAY: 's-play', OUT_OF_PLAY: 's-stop', RESTART_PREPARATION: 's-prep', POSSIBLE_STOPPAGE: 's-maybe', HALFTIME: 's-half',
                     NON_MATCH_FOOTAGE: 's-nopic', UNKNOWN: 's-unknown'};
const STATE_WORDS = {IN_PLAY: 'In play', OUT_OF_PLAY: 'Stoppage', RESTART_PREPARATION: 'Restart preparation', POSSIBLE_STOPPAGE: 'Possible stoppage',
                     HALFTIME: 'Halftime', NON_MATCH_FOOTAGE: 'No match picture', UNKNOWN: 'Unknown'};
const EVENT_WORDS = {GOAL: 'Goal', SUBSTITUTION: 'Substitution', KICK_OFF: 'Kick-off', CORNER_KICK: 'Corner kick', GOAL_KICK: 'Goal kick',
                     THROW_IN: 'Throw-in', FREE_KICK: 'Free kick', PENALTY_KICK: 'Penalty kick', UNKNOWN_RESTART: 'Unknown restart',
                     FOUL_OR_WHISTLE_STOPPAGE: 'Foul or whistle stoppage', OTHER_STOPPAGE: 'Other stoppage'};
const EVENT_TYPES = Object.keys(EVENT_WORDS);
const TEAM_EVENTS = ['GOAL', 'SUBSTITUTION'];   // added with a team (A or B)
function kindOf(type) {
  return type === 'GOAL' ? 'goal' : type === 'SUBSTITUTION' ? 'sub' : type === 'KICK_OFF' ? 'ko' :
    type === 'FOUL_OR_WHISTLE_STOPPAGE' || type === 'OTHER_STOPPAGE' ? 'stop' : 'restart';
}
const isBig = (e) => e.type === 'GOAL' || e.type === 'SUBSTITUTION';
const EVENT_FILTERS = [
  ['all', 'All events', () => true], ['GOAL', 'Goals', (e) => e.type === 'GOAL'], ['SUBSTITUTION', 'Substitutions', (e) => e.type === 'SUBSTITUTION'],
  ['KICK_OFF', 'Kick-offs', (e) => e.type === 'KICK_OFF'], ['restart', 'Other restarts', (e) => kindOf(e.type) === 'restart'],
  ['stop', 'Stoppages', (e) => kindOf(e.type) === 'stop'], ['candidate', 'To review (candidates)', (e) => e.status === 'candidate'],
  ['confirmed', 'Confirmed or added by you', (e) => e.status === 'confirmed'], ['removed', 'Removed by you', (e) => e.status === 'removed']];
// The timeline filter (#mtf beside the range): which kinds of events the markers show, and whether the play-state band
// (the strip) is drawn. One choice for every match, kept in localStorage under MARKS_KEY; the default is the clean-up:
// goals only, with the band. MARK_TYPES is the order of its checkboxes (EVENT_WORDS names them).
const MARKS_KEY = 'pitchiq.timelineMarkers.v1';
const MARK_TYPES = ['GOAL', 'KICK_OFF', 'CORNER_KICK', 'GOAL_KICK', 'THROW_IN', 'FREE_KICK', 'PENALTY_KICK', 'UNKNOWN_RESTART', 'SUBSTITUTION',
                    'FOUL_OR_WHISTLE_STOPPAGE', 'OTHER_STOPPAGE'];
const MARKS_DEFAULT = {show: ['GOAL'], strip: true};

function freshIntel(m) {
  return {m, intel: null, ready: null, off: false, error: null, phase: 'idle', timer: 0, seq: 0, builds: 0, fails: 0, busy: false,
          message: '', messageError: false, panelsTimer: 0, on: false, drawn: null, drawnMax: null, tipOwner: null, rows: new Map(), fpop: false,
          draft: null, drag: null, pop: false, popWhich: 'one', popError: '', halfMessage: '', halfError: false,
          filter: 'all', evidence: new Set(), addType: 'GOAL', addTeam: 'A',
          rv: {box: null, title: null, sig: null, second: null, index: null, events: null, open: new Set(), all: false}};
}
const mi = freshIntel(null);   // the open match's state (phase: idle | loading | building | updating | reloading)
let mt = null;                 // the layers built in #mt (mtParts)
let timelineFilter = null;     // the timeline filter in use: {show: Set of event types, strip} (marksFilter)
let mtf = null;                // the filter's button and popover in #mtf (filterParts)

const round2 = (v) => Math.round(v * 100) / 100;
const pct = (p) => `${Math.round(p * 1000) / 1000}%`;
const upperFirst = (s) => (s ? String(s).charAt(0).toUpperCase() + String(s).slice(1) : '');
const lowerFirst = (s) => (s ? String(s).charAt(0).toLowerCase() + String(s).slice(1) : '');
const article = (word) => (/^[aeiou]/i.test(word) ? 'an' : 'a');
function nowTime() { try { return round2(Math.max(0, Number(video.currentTime) || 0)); } catch (e) { return 0; } }
// a time as the halftime editor writes it: mm:ss, with its tenth when it has one (a typed time keeps its precision)
function fieldText(t) {
  const r = Math.round((isNum(t) ? t : 0) * 10) / 10, whole = Math.floor(r), tenth = Math.round((r - whole) * 10);
  return tenth ? `${clockText(whole)}.${tenth}` : clockText(whole);
}
function teamLabel(team) {
  // the team with its kit: the match index's teamKits ('Team A (white)'), else the coaching report's when it has one
  if (team !== 'A' && team !== 'B') return 'No team';
  const kits = player && player.index && player.index.teamKits, kit = kits && kits[team];
  if (kit && typeof kit.label === 'string' && kit.label) return kit.label;
  try { const k = coaching && coaching.mKits && coaching.mKits[team]; if (k && typeof k.label === 'string' && k.label) return k.label; } catch (e) { /* no kit labels on the page */ }
  return `Team ${team}`;
}
function teamColour(team) {
  try { const c = tactical && tactical.colours && tactical.colours[team]; if (typeof c === 'string' && c) return c; } catch (e) { /* the board's defaults */ }
  return team === 'A' ? '#b98cff' : '#f6b564';
}
function personName(m, fragment, person) {
  // a person by its slot label and id, found through a fragment where one is known (person ids are renumbered after corrections)
  const id = fragment ? m.ctx.resolve(fragment) : person, r = id ? m.ctx.roster.get(id) : null;
  return r ? nameOf(r) : (person || (fragment ? `fragment ${fragment}` : '?'));
}
function seekTo(t) {
  // a marker or a row: paused at its time through the page's playback controller (the range's own path)
  try { if (mode) return; } catch (e) { /* the page has no drawing mode */ }
  try { video.pause(); } catch (e) { /* no media yet */ }
  try { if (playback && typeof playback.seek === 'function') { playback.seek(t); return; } } catch (e) { /* no controller: the time is set directly */ }
  seekInto(t);
}

// --- the range's geometry: value = source-video seconds; the thumb's centre travels inset by THUMB_PX / 2 at each end
function seekScale() {
  const range = byId('timeline'), dur = duration();
  let min = range ? Number(range.min) : 0, max = range ? Number(range.max) : dur;
  if (!isNum(min)) min = 0;
  if (!isNum(max) || max <= min + 1) max = Math.max(min + 1, dur);   // the page sets the max to the video's duration at start
  let box = null;
  try { box = range && typeof range.getBoundingClientRect === 'function' ? range.getBoundingClientRect() : null; } catch (e) { box = null; }
  const left = box && isNum(box.left) ? box.left : 0, width = box && isNum(box.width) ? box.width : 0;
  return {min, max, left: left + THUMB_PX / 2, width: Math.max(1, width - THUMB_PX)};
}
function timeToX(t, s) { s = s || seekScale(); return s.left + (Math.max(s.min, Math.min(s.max, t)) - s.min) / (s.max - s.min) * s.width; }
function xToTime(x, s) { s = s || seekScale(); return s.min + Math.max(0, Math.min(1, (x - s.left) / s.width)) * (s.max - s.min); }
function timePct(t, s) { s = s || seekScale(); return (Math.max(s.min, Math.min(s.max, t)) - s.min) / (s.max - s.min) * 100; }

// --- the layers in #mt
function mtParts() {
  const box = byId('mt');
  if (!box) return null;
  if (mt && mt.box === box) return mt;
  const lane = (cls) => mk('span', `mt-lane ${cls}`);
  // the score labels' lane comes before the halftime's: a handle stands over a label (drawScores leaves out one it meets)
  const p = {box, strip: lane('mt-strip'), scores: lane('mt-scores'), half: lane('mt-half'), marks: lane('mt-marks'), tips: lane('mt-tips'), band: mk('span', 'mt-band'),
             tip: mk('span', 'mt-tip'), pop: mk('div', 'mt-pop'), handles: {}, lines: {}, labels: {}, pe: null, goals: []};
  p.strip.setAttribute('aria-hidden', 'true');
  p.scores.setAttribute('aria-hidden', 'true');   // the markers' own labels say the score
  p.marks.setAttribute('role', 'toolbar');
  p.marks.setAttribute('aria-label', 'Match events on the timeline: the arrow keys move, Enter seeks');
  p.tip.id = 'mtTip'; p.tip.setAttribute('role', 'tooltip'); p.tip.hidden = true;
  p.pop.id = 'mtPop'; p.pop.setAttribute('role', 'dialog'); p.pop.setAttribute('aria-label', 'Halftime'); p.pop.hidden = true;
  p.band.hidden = true;
  p.half.append(p.band);
  p.tips.append(p.tip);
  box.replaceChildren(p.strip, p.scores, p.half, p.marks, p.tips, p.pop);
  mt = p;
  const range = byId('timeline');
  if (range) {
    range.addEventListener('pointermove', rangeHover);
    range.addEventListener('pointerleave', () => hideTip('range'));
  }
  document.addEventListener('pointerdown', (e) => {
    // the halftime editor closes on a press elsewhere (its draft stays)
    if (!mi.pop || !mt) return;
    const t = e.target;
    if ((t && mt.pop.contains(t)) || Object.values(mt.handles).some((h) => h && t && h.contains(t))) return;
    closePop(false);
  }, true);
  if (typeof global.addEventListener === 'function') global.addEventListener('resize', () => { placeHandles(); placePop(); drawScores(p); if (mi.fpop && mtf) filterPlace(mtf); });
  // the score labels are placed again when the lane gets its width (the transport shown again after the Analytics or
  // Coaching view, where it had none and the handles could not be measured)
  if (typeof global.ResizeObserver === 'function') { try { new global.ResizeObserver(() => drawScores(p)).observe(p.scores); } catch (e) { /* the next draw places them */ } }
  return p;
}

// --- the strip and the markers
function stripRuns(timeline) {
  const runs = [];
  for (const r of Array.isArray(timeline) ? timeline : []) {
    if (!r || !isNum(r.t0) || !isNum(r.t1) || r.t1 <= r.t0) continue;
    const last = runs[runs.length - 1];
    if (last && last.state === r.state && Math.abs(last.t1 - r.t0) < 1e-6) last.t1 = r.t1;
    else runs.push({state: r.state, t0: r.t0, t1: r.t1});
  }
  return runs;
}
function shownEvents(a) {
  return (a && Array.isArray(a.events) ? a.events : []).filter((e) => e && isNum(e.t) && e.status !== 'removed').slice().sort((x, y) => x.t - y.t);
}
// the events the timeline draws: not removed, of a kind the timeline filter shows
function markedEvents(a) { const show = marksFilter().show; return shownEvents(a).filter((e) => show.has(e.type)); }
function scoresAfter(a) {
  // the running score after each goal, team A's goals first ('2-1'): the goals not removed that have a team, in time
  // order (event id -> text)
  const out = new Map(), n = {A: 0, B: 0};
  for (const e of shownEvents(a)) if (e.type === 'GOAL' && (e.team === 'A' || e.team === 'B')) { n[e.team]++; out.set(String(e.id), `${n.A}-${n.B}`); }
  return out;
}
function stateAt(t) {
  const tl = mi.ready && Array.isArray(mi.ready.timeline) ? mi.ready.timeline : null;
  if (!tl || !tl.length) return null;
  let lo = 0, hi = tl.length - 1;
  while (lo < hi) { const mid = Math.ceil((lo + hi) / 2); if (tl[mid].t0 <= t) lo = mid; else hi = mid - 1; }
  const r = tl[lo];
  return r && r.t0 <= t && t <= r.t1 ? r : null;
}
function drawStrip(p, a, s) {
  const runs = marksFilter().strip ? stripRuns(a ? a.timeline : null) : [];   // the play-state band, unless the timeline filter hides it
  p.strip.replaceChildren(...runs.map((r) => {
    const el = mk('span', `mt-run ${STATE_CLASS[r.state] || 's-unknown'}`), x0 = timePct(r.t0, s), x1 = timePct(r.t1, s);
    el.dataset.state = String(r.state);
    el.style.left = pct(x0);
    el.style.width = pct(Math.max(0, x1 - x0));
    return el;
  }));
  p.strip.hidden = !runs.length;
}
function eventTitle(e) { return `${EVENT_WORDS[e.type] || e.type} \u00b7 ${clockText(e.t)}`; }
function statusText(e) {
  const by = e.status === 'confirmed' ? (e.source === 'you' ? 'added by you' : 'confirmed by you') : e.status === 'candidate' ? 'candidate, to review' :
    e.status === 'removed' ? 'removed by you' : 'detected';
  const conf = e.confidence && e.confidence !== 'confirmed' ? `${e.confidence} confidence` : null;
  return [conf, by].filter(Boolean).join(' \u00b7 ');
}
function statusShort(e) {
  if (e.status === 'confirmed') return e.source === 'you' ? 'added by you' : 'confirmed';
  if (e.status === 'removed') return 'removed';
  return `${e.status === 'candidate' ? 'to review' : 'detected'}${e.confidence && e.confidence !== 'confirmed' ? ` \u00b7 ${e.confidence}` : ''}`;
}
function subSlots(e) {
  // a substitution's slots as the state saw them, 'A-03 -> A-04' (outSlot, inSlot: null for a player no slot claims);
  // null from a state that does not say them
  if (!e || (!('outSlot' in e) && !('inSlot' in e))) return null;
  const team = e.team === 'A' || e.team === 'B' ? e.team : '?', label = (s) => (Number.isInteger(s) ? `${team}-${pad2(s)}` : `${team}-?`);
  return `${label(e.outSlot)} \u2192 ${label(e.inSlot)}`;
}
function subPeople(m, e) {
  // the two people by their ids now (found through their fragments)
  const id = (f, p) => (f ? m.ctx.resolve(f) : p) || p || '?';
  return `${id(e.outFragment, e.outPerson)} \u2192 ${id(e.inFragment, e.inPerson)}`;
}
function eventWho(m, e) {
  // the team's part in the event (with its kit label when the page has one); a substitution's slots and people
  const team = e.team === 'A' || e.team === 'B' ? teamLabel(e.team) : null;
  if (e.type === 'SUBSTITUTION') {
    const slots = subSlots(e);
    if (slots) return [team, slots, subPeople(m, e)].filter(Boolean).join(' \u00b7 ');
    const parts = [team, Number.isInteger(e.slot) ? `slot ${pad2(e.slot)}` : null].filter(Boolean);
    const out = e.outFragment || e.outPerson ? personName(m, e.outFragment, e.outPerson) : null;
    const inn = e.inFragment || e.inPerson ? personName(m, e.inFragment, e.inPerson) : null;
    if (out || inn) parts.push(`${out || '?'} \u2192 ${inn || '?'}`);
    return parts.join(' \u00b7 ');
  }
  if (!team) return '';
  return e.type === 'GOAL' ? `${team} scored` : e.type === 'KICK_OFF' ? `${team} kicked off` : team;
}
function eventLabel(m, e) { return [`${EVENT_WORDS[e.type] || e.type} at ${clockText(e.t)}`, eventWho(m, e), statusText(e)].filter(Boolean).join(', '); }
function drawMarkers(m, p, a, s) {
  // only the kinds the timeline filter shows: a hidden kind is not drawn, so it takes no part in the nudging, the roving
  // tab stop, the arrow keys or a tooltip's "Also here"
  let prevBig = null, nudge = 0;
  const score = scoresAfter(a), goals = [];
  const marks = markedEvents(a).map((e) => {
    const big = isBig(e), b = mk('button', `mt-ev k-${kindOf(e.type)} st-${e.status || 'detected'} ${big ? 'big' : 'small'}`), x = timePct(e.t, s);
    b.type = 'button'; b.tabIndex = -1;
    b.dataset.ev = String(e.id); b.dataset.t = String(e.t);
    b.style.left = pct(x);
    if (big) {
      nudge = prevBig !== null && x - prevBig < NUDGE_PCT ? (nudge + 1) % 3 : 0;
      prevBig = x;
      if (nudge) b.style.transform = `translateX(${nudge * 7}px)`;
    }
    if (e.type === 'GOAL' && (e.team === 'A' || e.team === 'B')) b.style.setProperty('--team', teamColour(e.team));
    const after = e.type === 'GOAL' ? score.get(String(e.id)) : null;
    if (after) goals.push({id: String(e.id), x, nudge, text: after});
    b.setAttribute('aria-label', after ? `${eventLabel(m, e)}, score ${teamLabel('A')} ${after} ${teamLabel('B')}` : eventLabel(m, e));
    b.append(mk('i'));
    b.addEventListener('click', () => { hideTip(); markerSeek(m, e); });
    b.addEventListener('pointerenter', () => markerTip(m, e));
    b.addEventListener('pointerleave', () => hideTip('marker'));
    b.addEventListener('focus', () => { rove(b); markerTip(m, e); });
    b.addEventListener('blur', () => hideTip('marker'));
    b.addEventListener('keydown', (ev) => markerKey(ev, b));
    return b;
  });
  p.marks.replaceChildren(...marks);
  p.goals = goals;
  drawScores(p);
  const now = nowTime(), first = marks.find((b) => Number(b.dataset.t) >= now) || marks[0];   // the one tab stop: the next event
  if (first) first.tabIndex = 0;
}
function drawScores(p) {
  // the running score above the goals drawn, each where the previous label is SCORE_PX away (SCORE_PCT while the
  // lane's width is not known), so that two labels never overlap, and none where a halftime handle stands (in the same
  // row, it would hide the label or half of it); again whenever the handles are drawn (a drag, a resize)
  if (!p || !p.scores) return;
  let box = null;
  try { box = p.scores.getBoundingClientRect(); } catch (e) { box = null; }
  const laneW = (box && box.width) || p.scores.clientWidth || 0, room = laneW ? SCORE_PX / laneW * 100 : SCORE_PCT, labels = [];
  const handles = box && box.width ? Object.values(p.handles).filter((h) => h && !h.hidden).map((h) => h.getBoundingClientRect()).filter((r) => r && r.width > 0) : [];
  let last = -Infinity;
  for (const g of p.goals || []) {
    const x = g.x + (laneW ? g.nudge * 7 / laneW * 100 : 0), cx = box ? box.left + x / 100 * laneW : 0;
    if (x - last < room) continue;
    if (handles.some((r) => cx + SCORE_PX / 2 > r.left && cx - SCORE_PX / 2 < r.right)) continue;
    last = x;
    const label = mk('span', 'mt-score', g.text);
    label.dataset.ev = g.id;
    label.style.left = pct(g.x);
    if (g.nudge) label.style.transform = `translateX(-50%) translateX(${g.nudge * 7}px)`;   // above its nudged marker
    labels.push(label);
  }
  p.scores.replaceChildren(...labels);
}
function rove(b) { if (!mt) return; for (const x of mt.marks.children) x.tabIndex = x === b ? 0 : -1; }
function markerKey(ev, b) {
  if (!mt || ev.altKey || ev.ctrlKey || ev.metaKey) return;
  const list = [...mt.marks.children], i = list.indexOf(b);
  let j = -1;
  if (ev.key === 'ArrowRight' || ev.key === 'ArrowDown') j = Math.min(list.length - 1, i + 1);
  else if (ev.key === 'ArrowLeft' || ev.key === 'ArrowUp') j = Math.max(0, i - 1);
  else if (ev.key === 'Home') j = 0;
  else if (ev.key === 'End') j = list.length - 1;
  else if (ev.key === 'Escape') { hideTip(); return; }
  else return;
  ev.preventDefault(); ev.stopPropagation();   // the arrows move along the markers here, they never also step the video
  if (j >= 0 && list[j]) { rove(list[j]); list[j].focus(); }
}
function markerSeek(m, e) {
  seekTo(e.t);
  // the drawer's row, when the drawer shows it
  const row = mi.rows.get(String(e.id));
  let open = false;
  try { open = drawers.current === 'edit'; } catch (err) { open = false; }
  if (row && open) {
    row.classList.remove('flash'); void row.offsetWidth; row.classList.add('flash');
    try { row.scrollIntoView({block: 'nearest'}); } catch (err) { /* no scrolling here */ }
  }
}
function markerTip(m, e) {
  const s = seekScale(), x = timePct(e.t, s), lines = [eventTitle(e)];
  const who = eventWho(m, e), after = e.type === 'GOAL' ? scoresAfter(mi.ready).get(String(e.id)) : null;
  if (who) lines.push(who);
  if (after) lines.push(`Score after the goal: ${teamLabel('A')} ${after} ${teamLabel('B')}`);
  lines.push(upperFirst(statusText(e)));
  if (e.timeUncertain) lines.push('The time is uncertain');
  for (const line of (Array.isArray(e.evidence) ? e.evidence : []).slice(0, 4)) lines.push(`\u2022 ${line}`);
  if (e.type === 'GOAL') lines.push('Goal analysis: Coaching \u203a Key moments \u203a Goals');
  // the events drawn on top of each other are named too, so that none hides behind another (a kind the filter hides is
  // not drawn, so not named)
  const near = markedEvents(mi.ready).filter((o) => o.id !== e.id && isBig(o) === isBig(e) && Math.abs(timePct(o.t, s) - x) < .6);
  if (near.length) lines.push(`Also here: ${near.slice(0, 4).map((o) => `${EVENT_WORDS[o.type] || o.type} ${clockText(o.t)}`).join(', ')}${near.length > 4 ? ` and ${near.length - 4} more` : ''}`);
  tipShow(lines, x, 'marker');
}
function rangeHover(ev) {
  if (!mi.on || mi.drag || (mi.tipOwner && mi.tipOwner !== 'range')) return;
  // the time under the pointer, and its state while the play-state band is shown
  const s = seekScale(), t = xToTime(ev.clientX, s), r = marksFilter().strip ? stateAt(t) : null, lines = [`${clockText(t)}${r ? ` \u00b7 ${STATE_WORDS[r.state] || r.state}` : ''}`];
  if (r && r.why) lines.push(upperFirst(r.why));
  if (r && r.review) lines.push('Uncertain: worth a look');
  tipShow(lines, timePct(t, s), 'range');
}
function tipShow(lines, x, owner) {
  const p = mtParts();
  if (!p) return;
  const tip = p.tip;
  tip.replaceChildren(...lines.map((line, k) => mk(k ? 'span' : 'b', null, line)));
  tip.style.left = pct(x);
  tip.style.marginLeft = '0px';
  tip.hidden = false;
  mi.tipOwner = owner;
  // inside the transport when the layout is known
  const laneW = p.tips.clientWidth || 0, w = tip.offsetWidth || 0;
  if (laneW && w) {
    const left = x / 100 * laneW - w / 2, fixed = Math.max(-7, Math.min(laneW + 7 - w, left));
    if (fixed !== left) tip.style.marginLeft = `${Math.round(fixed - left)}px`;
  }
}
function hideTip(owner) {
  if (!mt || (owner && mi.tipOwner && owner !== mi.tipOwner)) return;
  mt.tip.hidden = true;
  mi.tipOwner = null;
}
function drawLayers(m) {
  // the strip and the markers, drawn again only when the state shown, the range's max or the timeline filter changed
  // (filterChanged clears the memo)
  const p = mt;
  if (!p) return;
  const a = mi.on ? mi.ready : null, s = seekScale();
  if (mi.drawn === a && mi.drawnMax === s.max) return;
  drawStrip(p, a, s); drawMarkers(m, p, a, s); mi.drawn = a; mi.drawnMax = s.max;
}

// --- the timeline filter (#mtf beside the range: #mtfBtn and its popover #mtfPop)
function marksFilter() {
  // the choice in use, read from localStorage once (absent, unreadable or storage blocked: the default, goals only)
  if (timelineFilter) return timelineFilter;
  let saved = null;
  try { saved = JSON.parse(localStorage.getItem(MARKS_KEY) || 'null'); } catch (e) { saved = null; }
  const ok = !!saved && typeof saved === 'object' && Array.isArray(saved.show);
  timelineFilter = {show: new Set((ok ? saved.show : MARKS_DEFAULT.show).filter((t) => MARK_TYPES.includes(t))),
                    strip: ok && typeof saved.strip === 'boolean' ? saved.strip : MARKS_DEFAULT.strip};
  return timelineFilter;
}
function marksSave() {
  const f = marksFilter();
  try { localStorage.setItem(MARKS_KEY, JSON.stringify({show: MARK_TYPES.filter((t) => f.show.has(t)), strip: f.strip})); } catch (e) { /* storage blocked: the choice lasts for the page */ }
}
function typesWord(type) { return type === 'FOUL_OR_WHISTLE_STOPPAGE' ? 'Fouls or whistle stoppages' : `${EVENT_WORDS[type] || type}s`; }
function marksText() {
  // what the button says: 'Goals only', '5 of 11', 'All markers', 'No markers'
  const f = marksFilter(), n = MARK_TYPES.filter((t) => f.show.has(t)).length;
  if (n === MARK_TYPES.length) return 'All markers';
  if (!n) return 'No markers';
  return n === 1 && f.show.has('GOAL') ? 'Goals only' : `${n} of ${MARK_TYPES.length}`;
}
function filterParts() {
  // the button and its popover (index.html #mtf), built and wired once: the button opens and closes the popover; Escape
  // closes it (the focus back on the button), so does a press elsewhere
  const box = byId('mtf'), button = byId('mtfBtn'), pop = byId('mtfPop');
  if (!box || !button || !pop) return null;
  if (mtf && mtf.box === box) return mtf;
  const p = {box, button, pop, text: byId('mtfText'), boxes: new Map(), counts: new Map(), rows: new Map(), quick: new Map(), strip: null};
  const head = mk('div', 'mtf-head'), close = btn('\u00d7', () => filterOpen(false, true), 'quiet');
  close.setAttribute('aria-label', 'Close the timeline markers');
  head.append(mk('b', null, 'TIMELINE MARKERS'), close);
  const quick = mk('div', 'mtf-quick');
  quick.setAttribute('role', 'group'); quick.setAttribute('aria-label', 'Show at once');
  for (const [key, label] of [['goals', 'Goals only'], ['all', 'All'], ['none', 'None']]) {
    const b = btn(label, () => filterSet(key));
    b.dataset.mtf = key;
    p.quick.set(key, b);
    quick.append(b);
  }
  const list = mk('div', 'mtf-list');
  list.setAttribute('role', 'group'); list.setAttribute('aria-label', 'Events on the timeline');
  for (const type of MARK_TYPES) {
    // a checkbox, the marker as the timeline draws it, the kind and how many of them the match has
    const row = mk('label', 'mtf-row'), input = mk('input'), key = mk('span', `me-key k-${kindOf(type)}`), n = mk('span', 'mtf-n');
    input.type = 'checkbox'; input.dataset.type = type;
    input.addEventListener('change', () => filterType(type, input.checked));
    key.append(mk('i'));
    row.append(input, key, mk('span', 'mtf-name', typesWord(type)), n);
    p.boxes.set(type, input); p.counts.set(type, n); p.rows.set(type, row);
    list.append(row);
  }
  const band = mk('label', 'mtf-row mtf-band');
  p.strip = mk('input');
  p.strip.type = 'checkbox'; p.strip.dataset.mtf = 'strip';
  p.strip.addEventListener('change', () => filterStrip(p.strip.checked));
  band.append(p.strip, mk('i', 'me-sw s-play'), mk('span', 'mtf-name', 'Play-state band'));
  pop.replaceChildren(head, quick, list, band);
  button.addEventListener('click', () => filterOpen(!mi.fpop, false));
  box.addEventListener('keydown', (e) => { if (e.key === 'Escape' && mi.fpop) { e.preventDefault(); e.stopPropagation(); filterOpen(false, true); } });
  document.addEventListener('pointerdown', (e) => {
    if (!mi.fpop || !mtf) return;
    const t = e.target;
    if (t && mtf.box.contains(t)) return;
    filterOpen(false, false);
  }, true);
  mtf = p;
  return p;
}
function filterPaint() {
  // only while a match's state is shown (mi.on, like #mt): the button's text, the checkboxes, the count of each kind
  // (removed events aside); 'part' marks a choice that hides something (a dot where the text is hidden)
  const p = filterParts();
  if (!p) return;
  if (!mi.on) mi.fpop = false;
  const f = marksFilter(), text = marksText(), counts = new Map(), n = MARK_TYPES.filter((t) => f.show.has(t)).length;
  for (const e of shownEvents(mi.on ? mi.ready : null)) counts.set(e.type, (counts.get(e.type) || 0) + 1);
  p.box.hidden = !mi.on;
  p.pop.hidden = !mi.fpop;
  p.box.classList.toggle('part', n < MARK_TYPES.length || !f.strip);
  setText(p.text, text);
  const said = `Timeline markers: ${text}${f.strip ? '' : ', play-state band hidden'}`;
  p.button.setAttribute('aria-expanded', String(!!mi.fpop));
  p.button.setAttribute('aria-label', said);
  p.button.title = `${said}. Choose what the time control shows.`;
  for (const type of MARK_TYPES) {
    const c = counts.get(type) || 0;
    p.boxes.get(type).checked = f.show.has(type);
    setText(p.counts.get(type), String(c));
    p.rows.get(type).classList.toggle('none', !c);
  }
  p.strip.checked = f.strip;
  const pressed = {goals: n === 1 && f.show.has('GOAL'), all: n === MARK_TYPES.length, none: !n};
  for (const [key, b] of p.quick) b.setAttribute('aria-pressed', String(pressed[key]));
}
function filterOpen(on, focusButton) {
  const p = filterParts();
  if (!p) return;
  mi.fpop = !!on && mi.on;
  filterPaint();
  if (mi.fpop) filterPlace(p);
  try {
    if (mi.fpop) p.boxes.get(MARK_TYPES[0]).focus();
    else if (focusButton) p.button.focus();
  } catch (e) { /* not focusable here */ }
}
function filterPlace(p) {
  // above the button when it fits under the sticky header (the page's header), else below it; never taller than the
  // room on the side it opens to (it scrolls inside)
  let b = null, h = 0, top = 0;
  try {
    b = p.button.getBoundingClientRect();
    h = p.pop.scrollHeight || 0;
    const head = document.querySelector('header');
    top = head && typeof head.getBoundingClientRect === 'function' ? Math.max(0, head.getBoundingClientRect().bottom) : 0;
  } catch (e) { b = null; }
  if (!b || !isNum(b.top) || !isNum(global.innerHeight)) return;
  const above = b.top - top - 16, below = global.innerHeight - b.bottom - 16, down = h > above && below > above;
  p.pop.classList.toggle('down', down);
  p.pop.style.maxHeight = `${Math.max(160, Math.floor(down ? below : above))}px`;
}
function filterSet(key) { marksFilter().show = new Set(key === 'all' ? MARK_TYPES : key === 'goals' ? ['GOAL'] : []); filterChanged(); }
function filterType(type, on) { const f = marksFilter(); if (on) f.show.add(type); else f.show.delete(type); filterChanged(); }
function filterStrip(on) { marksFilter().strip = !!on; filterChanged(); }
function filterChanged() {
  // kept for the next visit and drawn at once: the memo of the strip and the markers is cleared
  marksSave();
  mi.drawn = null; mi.drawnMax = null;
  if (player && mi.m === player) drawLayers(player);
  hideTip('marker');
  filterPaint();
}

// --- the halftime
function halfSaved() {
  // the halftime as the server has it: saved by the coach, else the match plan's break and the goal ends' direction
  const h = (mi.intel && mi.intel.halftime) || (mi.m && mi.m.index && mi.m.index.halftime) || null;
  return h && typeof h === 'object' ? h : null;
}
function inferredSide(h) { const f = h && h.inferred && h.inferred.firstHalf; return f === 'A' ? 'right' : f === 'B' ? 'left' : null; }
function halfRange() {
  const h = halfSaved(), r = h && Array.isArray(h.range) ? h.range : null;
  if (r && isNum(r[0]) && isNum(r[1]) && r[1] > r[0] + 2) return [r[0], r[1]];
  const segs = mi.m && mi.m.segs;
  return segs && segs.length ? [segs[0].start, segs[segs.length - 1].end] : null;
}
function halfLimits() {
  // strictly inside the match range, as the server requires (fullmatch/halftime.clean)
  const r = halfRange();
  return r ? [Math.floor(r[0]) + 1, Math.ceil(r[1]) - 1] : null;
}
function halfBase() {
  // what the handles show when nothing is edited: {mode, end, start, side, set}
  const h = halfSaved(), lim = halfLimits();
  if (!lim) return null;
  let end = h && isNum(h.firstHalfEnd) ? h.firstHalfEnd : null, start = h && isNum(h.secondHalfStart) ? h.secondHalfStart : null;
  const set = end !== null;
  if (end === null) end = Math.round((lim[0] + lim[1]) / 2);
  if (start === null || start < end) start = end;
  const mode = h && h.mode === 'break' && start > end ? 'break' : 'marker';
  const side = h && (h.teamAFirstHalf === 'left' || h.teamAFirstHalf === 'right') ? h.teamAFirstHalf : inferredSide(h);
  return {mode, end, start: mode === 'break' ? start : end, side, set};
}
function halfShown() { return mi.draft || halfBase(); }
function halfDirty() {
  const d = mi.draft, b = halfBase();
  if (!d || !b) return false;
  return !b.set || d.mode !== b.mode || Math.abs(d.end - b.end) > 1e-6 || (d.mode === 'break' && Math.abs(d.start - b.start) > 1e-6) || d.side !== b.side;
}
function halfDraft() {
  if (!mi.draft) {
    const b = halfBase();
    if (!b) return null;
    mi.draft = {mode: b.mode, end: b.end, start: b.start, side: b.side, lastBreak: null};
  }
  return mi.draft;
}
function halfBounds(which, d) {
  const lim = halfLimits() || [0, 0];
  if (d.mode !== 'break' || which === 'one') return lim;
  return which === 'start' ? [Math.max(lim[0], d.end), lim[1]] : [lim[0], Math.min(lim[1], d.start)];
}
function setHalf(which, t) {
  const d = halfDraft();
  if (!d || !isNum(t)) return false;
  const [lo, hi] = halfBounds(which, d);
  t = Math.max(lo, Math.min(hi, t));
  if (d.mode !== 'break') { d.end = t; d.start = t; } else if (which === 'start') d.start = t; else d.end = t;
  drawHalf(); updatePop();
  return true;
}
function setBreak(on) {
  const d = halfDraft(), lim = halfLimits();
  if (!d || !lim) return;
  if (on && d.mode !== 'break') {
    const b = halfBase();
    // the break as it was before it was switched off, else the saved one, else a fifteen-minute break
    const start = d.lastBreak && d.lastBreak.end === d.end ? d.lastBreak.start : b && b.mode === 'break' && b.start > d.end ? b.start : d.end + 15 * 60;
    d.mode = 'break';
    d.start = Math.min(lim[1], Math.max(d.end, start));
  } else if (!on && d.mode === 'break') {
    d.lastBreak = {end: d.end, start: d.start};
    d.mode = 'marker';
    d.start = d.end;
  }
  mi.popError = '';
  drawHalf(); updatePop();
}
function setSide(side) {
  const d = halfDraft();
  if (!d) return;
  d.side = side;
  mi.popError = '';
  drawHalf(); updatePop();
}
function halfParts(which) {
  const p = mtParts();
  if (!p) return null;
  if (!p.handles[which]) {
    const el = mk('span', 'mt-hh'), word = mk('b'), time = mk('span'), line = mk('span', 'mt-hline');
    el.dataset.half = which; line.dataset.half = which;
    el.setAttribute('role', 'slider'); el.setAttribute('aria-haspopup', 'dialog');
    el.tabIndex = 0;
    el.append(word, time);
    el.addEventListener('pointerdown', (e) => halfDown(e, which));
    el.addEventListener('pointermove', halfMove);
    el.addEventListener('pointerup', halfUp);
    el.addEventListener('pointercancel', halfCancel);
    el.addEventListener('lostpointercapture', halfUp);
    el.addEventListener('keydown', (e) => halfKey(e, which));
    el.addEventListener('pointerenter', () => { if (!mi.drag) halfTip(which); });
    el.addEventListener('pointerleave', () => hideTip('half'));
    el.addEventListener('focus', () => { if (!mi.drag) halfTip(which); });
    el.addEventListener('blur', () => hideTip('half'));
    p.handles[which] = el; p.lines[which] = line; p.labels[which] = {word, time};
    p.half.append(line, el);
  }
  return {el: p.handles[which], line: p.lines[which], label: p.labels[which]};
}
function drawHalf() {
  const p = mt || mtParts();
  if (!p) return;
  const d = mi.on ? halfShown() : null, want = !d ? [] : d.mode === 'break' ? ['end', 'start'] : ['one'];
  const s = seekScale(), dirty = halfDirty(), updating = mi.busy || mi.phase === 'updating' || mi.phase === 'reloading';
  for (const which of ['one', 'end', 'start']) {
    if (!want.includes(which)) { if (p.handles[which]) { p.handles[which].hidden = true; p.lines[which].hidden = true; } continue; }
    const {el, line, label} = halfParts(which), v = which === 'start' ? d.start : d.end, [lo, hi] = halfBounds(which, d), x = pct(timePct(v, s));
    el.hidden = false; line.hidden = false;
    el.style.left = x; line.style.left = x;
    el.classList.toggle('a-end', which === 'end'); el.classList.toggle('a-start', which === 'start');
    el.classList.toggle('dirty', dirty); line.classList.toggle('dirty', dirty);
    el.classList.toggle('busy', updating && !dirty);
    el.classList.toggle('dragging', !!(mi.drag && mi.drag.which === which && mi.drag.moved));
    const word = which === 'end' ? 'First half end' : which === 'start' ? 'Second half start' : 'Halftime';
    setText(label.word, word.toUpperCase());
    setText(label.time, !mi.draft && d.set === false ? ' \u2014 not set' : ` \u2014 ${clockText(v)}`);
    el.setAttribute('aria-label', word);
    el.setAttribute('aria-valuemin', String(lo)); el.setAttribute('aria-valuemax', String(hi)); el.setAttribute('aria-valuenow', String(round2(v)));
    el.setAttribute('aria-valuetext', `${videoText(v)}${dirty ? ', not saved yet' : ''}`);
  }
  const brk = !!d && d.mode === 'break' && d.start > d.end;
  p.band.hidden = !brk;
  if (brk) { const x0 = timePct(d.end, s); p.band.style.left = pct(x0); p.band.style.width = pct(timePct(d.start, s) - x0); }
  placeHandles();
  drawScores(p);   // the score labels clear of the handles where they stand now
}
function placeHandles() {
  // a handle stays inside the transport when the layout is known (its line stays at its time); the two of a break say
  // END and START when their full words would overlap (a narrow transport), their tooltips and labels keep the names
  const p = mt;
  if (!p) return;
  const laneW = p.half.clientWidth || 0;
  const place = () => {
    for (const which of ['one', 'end', 'start']) {
      const el = p.handles[which];
      if (!el || el.hidden) continue;
      el.style.marginLeft = '0px';
      const w = el.offsetWidth || 0;
      if (!laneW || !w) continue;
      const left = parseFloat(el.style.left) / 100 * laneW + (which === 'end' ? -w : which === 'start' ? 0 : -w / 2);
      const fixed = Math.max(-7, Math.min(laneW + 7 - w, left));
      if (fixed !== left) el.style.marginLeft = `${Math.round(fixed - left)}px`;
    }
  };
  const words = (short) => {
    for (const [which, long, brief] of [['end', 'FIRST HALF END', 'END'], ['start', 'SECOND HALF START', 'START']]) if (p.labels[which]) setText(p.labels[which].word, short ? brief : long);
  };
  const end = p.handles.end, start = p.handles.start, pair = !!end && !!start && !end.hidden && !start.hidden;
  if (pair) words(false);
  place();
  if (!pair || !laneW) return;
  if (end.getBoundingClientRect().right > start.getBoundingClientRect().left + .5) { words(true); place(); }   // they touch by design (each beside its line)
}
function halfTip(which) {
  const d = halfShown();
  if (!d || !mt || !mt.handles[which]) return;
  const v = which === 'start' ? d.start : d.end, word = which === 'end' ? 'First half end' : which === 'start' ? 'Second half start' : 'Halftime';
  tipShow([`${word} \u00b7 ${clockText(v)}${halfDirty() ? ' \u00b7 not saved yet' : ''}`,
           'Marks where the teams change ends: it moves only the attacking directions, never positions or identities.',
           'Drag it, use the arrow keys (Shift: 10 s) or press Enter to type a time.'], timePct(v), 'half');
}
function halfDown(e, which) {
  if ((e.button !== undefined && e.button !== 0) || mi.busy) return;
  e.preventDefault(); e.stopPropagation();
  const el = mt && mt.handles[which];
  if (!el) return;
  try { el.setPointerCapture(e.pointerId); } catch (err) { /* no capture: the moves arrive while over the handle */ }
  const d0 = halfShown();
  // relative to where it was grabbed: a handle's label stands beside its line, so the pointer is rarely on the line itself
  mi.drag = {which, id: e.pointerId, x0: e.clientX, t0: which === 'start' ? d0.start : d0.end, moved: false, el};
  hideTip();
  try { el.focus({preventScroll: true}); } catch (err) { /* not focusable here */ }
}
function halfMove(e) {
  const d = mi.drag;
  if (!d || (d.id !== undefined && e.pointerId !== undefined && e.pointerId !== d.id)) return;
  if (!d.moved && Math.abs(e.clientX - d.x0) < 3) return;   // a click on the label opens the precise entry
  d.moved = true;
  if (e.preventDefault) e.preventDefault();
  const s = seekScale();
  setHalf(d.which, Math.round(d.t0 + (e.clientX - d.x0) / s.width * (s.max - s.min)));   // whole seconds, as the label shows them
}
function halfUp(e) {
  const d = mi.drag;
  if (!d || (d.id !== undefined && e && e.pointerId !== undefined && e.pointerId !== d.id)) return;
  mi.drag = null;
  try { d.el.releasePointerCapture(d.id); } catch (err) { /* released already */ }
  drawHalf();
  openPop(d.which, !d.moved);   // after a drag: Save is one click away; a click: the time to type
}
function halfCancel() { mi.drag = null; drawHalf(); }
function halfKey(e, which) {
  if (e.altKey || e.ctrlKey || e.metaKey) return;
  const k = e.key;
  if (k === 'Enter') { e.preventDefault(); e.stopPropagation(); openPop(which, true); return; }
  if (k === 'Escape') { if (mi.pop) { e.preventDefault(); e.stopPropagation(); closePop(true); } return; }
  if (!(k in HALF_KEYS) && k !== 'Home' && k !== 'End') return;
  e.preventDefault(); e.stopPropagation();   // the arrows move the handle here, they never also step the video
  if (mi.busy) return;
  const d = halfDraft();
  if (!d) return;
  const [lo, hi] = halfBounds(which, d), v = which === 'start' ? d.start : d.end;
  const step = HALF_KEYS[k], t = k === 'Home' ? lo : k === 'End' ? hi : v + step * (e.shiftKey && Math.abs(step) === 1 ? 10 : 1);
  setHalf(which, t);
  openPop(which, false);
}
// the halftime editor
function popBuild(p) {
  if (p.pe) return p.pe;
  const pe = {};
  const head = mk('div', 'mt-pop-head');
  pe.close = btn('\u00d7', () => closePop(true), 'quiet');
  pe.close.setAttribute('aria-label', 'Close the halftime editor');
  head.append(mk('b', null, 'HALFTIME'), pe.close);
  pe.note = mk('p', 'mt-note');
  pe.note.setAttribute('role', 'note');
  const chk = mk('label', 'mt-chk');
  pe.brk = mk('input');
  pe.brk.type = 'checkbox';
  pe.brk.dataset.mt = 'break';
  pe.brk.addEventListener('change', () => setBreak(pe.brk.checked));
  chk.append(pe.brk, document.createTextNode('Include the break (first half end and second half start)'));
  const field = (which) => {
    const wrap = mk('label', 'mt-time'), label = mk('span'), row = mk('span'), input = mk('input');
    const now = btn('Now', () => { mi.popError = ''; setHalf(mi.draft && mi.draft.mode === 'break' ? which : 'one', nowTime()); });
    input.type = 'text'; input.placeholder = 'mm:ss';
    input.setAttribute('inputmode', 'decimal'); input.setAttribute('autocomplete', 'off'); input.setAttribute('spellcheck', 'false');
    input.dataset.half = which; now.dataset.half = which;
    now.title = 'Use the current playback time';
    input.addEventListener('change', () => popEnter(which, input));
    input.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') { e.preventDefault(); e.stopPropagation(); popEnter(which, input); }
      else if (e.key === 'Escape') { e.preventDefault(); e.stopPropagation(); closePop(true); }
    });
    row.append(input, now);
    wrap.append(label, row);
    return {wrap, label, input, now};
  };
  pe.f1 = field('end'); pe.f2 = field('start');
  const times = mk('div', 'mt-times');
  times.append(pe.f1.wrap, pe.f2.wrap);
  pe.err = mk('p', 'mt-err');
  pe.err.setAttribute('role', 'alert');
  const dir = mk('div', 'mt-dir'), seg = mk('span', 'mt-seg');
  seg.setAttribute('role', 'group'); seg.setAttribute('aria-label', 'Team A attacks in the first half');
  pe.left = btn('Left', () => setSide('left')); pe.right = btn('Right', () => setSide('right'));
  pe.left.dataset.side = 'left'; pe.right.dataset.side = 'right';
  seg.append(pe.left, pe.right);
  pe.hint = mk('span', 'mt-hint');
  dir.append(mk('span', null, 'Team A attacks'), seg, mk('span', null, 'in the first half'), pe.hint);
  pe.check = mk('div', 'mt-check');
  const explain = mk('p', 'mt-explain', 'It marks where the teams change ends. Only the attacking directions follow it (team shape, the defensive line, the reports); positions, player identities and detections never change.');
  const foot = mk('div', 'mt-foot');
  pe.save = btn('Save', () => halfSave(), 'primary');
  pe.cancel = btn('Cancel', () => halfDiscard());
  pe.reset = btn('Reset', () => halfReset(), 'quiet');
  pe.save.dataset.mt = 'save'; pe.cancel.dataset.mt = 'cancel'; pe.reset.dataset.mt = 'reset';
  pe.cancel.title = 'Forget the edits';
  pe.reset.title = 'Back to the match plan\u2019s halftime and the direction the goal ends suggest';
  pe.state = mk('span', 'mt-state');
  pe.state.setAttribute('role', 'status');
  foot.append(pe.save, pe.cancel, pe.reset, pe.state);
  p.pop.replaceChildren(head, pe.note, chk, times, pe.err, dir, pe.check, explain, foot);
  p.pe = pe;
  return pe;
}
function openPop(which, entry) {
  const p = mt || mtParts();
  if (!p || !mi.on || !halfShown()) return;
  popBuild(p);
  mi.pop = true; mi.popWhich = which;
  updatePop();
  if (entry) {
    const d = halfShown(), f = d.mode === 'break' && which === 'start' ? p.pe.f2 : p.pe.f1;
    try { f.input.focus(); if (typeof f.input.select === 'function') f.input.select(); } catch (e) { /* not focusable here */ }
  }
}
function closePop(focusHandle) {
  if (!mi.pop) return;
  mi.pop = false; mi.popError = '';
  if (mt) mt.pop.hidden = true;
  if (focusHandle && mt) {
    const h = [mt.handles[mi.popWhich], mt.handles.one, mt.handles.end].find((x) => x && !x.hidden);
    try { if (h) h.focus(); } catch (e) { /* not focusable here */ }
  }
}
function popEnter(which, input) {
  // a typed time: mm:ss, h:mm:ss or seconds, inside the match (and on its side of the other handle)
  const d = halfDraft();
  if (!d) return;
  const w = d.mode === 'break' ? which : 'one', v = parseClock(input.value), [lo, hi] = halfBounds(w, d);
  let problem = null;
  if (v === null) problem = 'Enter a time as mm:ss, h:mm:ss or seconds.';
  else if (v < lo || v > hi) {
    problem = w === 'one' ? `Halftime must lie inside the match: ${videoText(lo)}\u2013${videoText(hi)}.` :
      w === 'end' ? `The first half must end between ${videoText(lo)} and ${videoText(hi)}.` : `The second half must start between ${videoText(lo)} and ${videoText(hi)}.`;
  }
  input.setAttribute('aria-invalid', String(!!problem));
  mi.popError = problem || '';
  if (problem) { updatePop(); return; }
  setHalf(w, v);
  input.value = fieldText(v);
}
function updatePop() {
  const p = mt;
  if (!p) return;
  const open = mi.pop && mi.on && !!halfShown();
  p.pop.hidden = !open;
  if (!open) return;
  const pe = popBuild(p), d = halfShown(), base = halfBase(), saved = halfSaved(), dirty = halfDirty(), brk = d.mode === 'break';
  const updating = mi.phase === 'updating' || mi.phase === 'reloading';
  pe.brk.checked = brk; pe.brk.disabled = mi.busy;
  setText(pe.f1.label, brk ? 'First half end' : 'Halftime');
  setText(pe.f2.label, 'Second half start');
  pe.f2.wrap.hidden = !brk;
  for (const [f, v] of [[pe.f1, d.end], [pe.f2, d.start]]) {
    if (document.activeElement !== f.input) { f.input.value = fieldText(v); f.input.setAttribute('aria-invalid', 'false'); }
    f.input.disabled = mi.busy; f.now.disabled = mi.busy;
  }
  pe.f1.input.setAttribute('aria-label', `${brk ? 'First half end' : 'Halftime'} (mm:ss, h:mm:ss or seconds)`);
  pe.f2.input.setAttribute('aria-label', 'Second half start (mm:ss, h:mm:ss or seconds)');
  setText(pe.err, mi.popError); pe.err.hidden = !mi.popError;
  pe.left.setAttribute('aria-pressed', String(d.side === 'left')); pe.right.setAttribute('aria-pressed', String(d.side === 'right'));
  pe.left.disabled = mi.busy; pe.right.disabled = mi.busy;
  setText(pe.hint, d.side ? `In the second half team A attacks ${d.side === 'left' ? 'right' : 'left'}.` : 'Choose the side team A attacks in the first half.');
  const check = mi.ready ? mi.ready.directionCheck : null, note = check && check.note;
  setText(pe.note, note || ''); pe.note.hidden = !note;
  pe.check.replaceChildren(...directionLines(saved, check, d.side).slice(1).map((l) => mk('p', l.warn ? 'warn' : null, l.text)));
  pe.save.disabled = mi.busy || !dirty || (d.side !== 'left' && d.side !== 'right');
  setText(pe.save, mi.busy ? 'Saving\u2026' : 'Save');
  pe.cancel.hidden = !dirty; pe.cancel.disabled = mi.busy;
  pe.reset.disabled = mi.busy || !(saved && saved.confirmed);
  setText(pe.state, mi.halfError ? mi.halfMessage : mi.busy ? 'Saving the halftime\u2026' : updating ? 'Updating the match state\u2026' : dirty ? 'Not saved yet' :
    saved && saved.confirmed ? 'Saved' : base && base.set ? 'From the match plan and the goal ends: not confirmed yet' : 'Not set: drag the handle or type a time');
  pe.state.className = `mt-state${mi.halfError ? ' error' : dirty ? ' warn' : ''}`;
  placePop();
}
function placePop() {
  const p = mt;
  if (!p || p.pop.hidden) return;
  const W = p.box.clientWidth || 0, w = p.pop.offsetWidth || 0, h = [p.handles[mi.popWhich], p.handles.one, p.handles.end].find((x) => x && !x.hidden);
  if (!W || !w || !h) { p.pop.style.left = '50%'; p.pop.style.transform = 'translateX(-50%)'; return; }
  const x = THUMB_PX / 2 + parseFloat(h.style.left) / 100 * (W - THUMB_PX);
  p.pop.style.left = `${Math.round(Math.max(0, Math.min(W - w, x - w / 2)))}px`;
  p.pop.style.transform = 'none';
}
function directionLines(h, check, side) {
  // the first-half direction in words, and whether the goal ends and the kick-offs agree with it: [{text, warn}]
  const out = [];
  if (!h) return out;
  if (side === undefined) side = h.teamAFirstHalf;
  const saved = h.teamAFirstHalf, inf = h.inferred || {}, infSide = inferredSide(h);
  if (side === 'left' || side === 'right') {
    out.push({text: `Team A attacks ${side} in the first half and ${side === 'left' ? 'right' : 'left'} in the second${h.confirmed ? ' (saved)' : ' (from the goal ends, not confirmed)'}.`});
  } else out.push({text: 'Which way team A attacks in the first half is not known: set it on the HALFTIME handle of the timeline.', warn: true});
  const k = check && isNum(check.kickOffs) ? check.kickOffs : 0;
  if ((side === 'left' || side === 'right') && (saved === 'left' || saved === 'right') && side !== saved) {
    out.push({text: `Not saved yet: the kick-offs were checked with team A attacking ${saved} in the first half.`, warn: true});
  } else if (k) {
    const all = k === 1 ? 'the kick-off' : `all ${k} kick-offs`;
    if (!check.disagree) out.push({text: infSide && infSide === saved && inf.consistent ? `The inferred direction and ${all} agree.` : `${upperFirst(all)} agree${k === 1 ? 's' : ''} with it.`});
    else out.push({text: `${check.agree || 0} of ${k} kick-offs agree with it.`, warn: true});
  } else if (check) out.push({text: 'No kick-off was found to check it against.'});
  if (infSide && (side === 'left' || side === 'right') && infSide !== side) out.push({text: `The goal ends suggest team A attacks ${infSide} in the first half.`, warn: true});
  else if (inf.consistent === false && inf.firstHalf && inf.secondHalf) out.push({text: 'The goal ends do not show the teams changing ends: check the direction.', warn: true});
  return out;
}
async function halfSave() {
  const m = player, d = mi.draft;
  if (!m || mi.m !== m || mi.busy || !d || !halfDirty()) return;
  if (d.side !== 'left' && d.side !== 'right') { mi.popError = 'Choose which way team A attacks in the first half.'; updatePop(); return; }
  await halfPost(m, '/api/match/halftime', {match: m.id, mode: d.mode, firstHalfEnd: round2(d.end), secondHalfStart: round2(d.mode === 'break' ? d.start : d.end),
                                            teamAFirstHalf: d.side});
}
async function halfReset() {
  const m = player;
  if (!m || mi.m !== m || mi.busy) return;
  await halfPost(m, '/api/match/halftime/reset', {match: m.id});
}
function halfDiscard() { mi.draft = null; mi.popError = ''; mi.halfError = false; drawHalf(); updatePop(); }
async function halfPost(m, path, body) {
  // saved: the server makes the state again on a thread ("Updating the match state..."), asked until ready
  mi.busy = true; mi.halfError = false; mi.halfMessage = ''; mi.popError = '';
  intelDraw(m);
  try {
    const answer = await api(path, body);
    if (player !== m || mi.m !== m) return;
    mi.draft = null; mi.busy = false; mi.phase = 'updating'; mi.builds = 0; mi.seq++;
    if (answer && answer.intel) intelTake(m, answer.intel); else intelPoll(m);
  } catch (error) {
    if (player !== m || mi.m !== m) return;
    mi.halfError = true; mi.halfMessage = error.message || String(error);
  } finally {
    mi.busy = false;
    if (player === m && mi.m === m) intelDraw(m);
  }
}

// --- the status line beside the range
function statusLine() {
  if (!mi.m || mi.off) return null;
  const a = mi.intel, b = a && a.build, progress = b && isNum(b.progress) && b.progress > 0 ? Math.round(b.progress * 100) : null;
  if (mi.phase === 'updating' || mi.phase === 'reloading') {
    return {text: 'Updating the match state\u2026', busy: true, title: b && b.stage ? `${b.stage}${progress !== null ? ` (${progress}%)` : ''}` : ''};
  }
  if (mi.phase === 'building' || (a && a.state === 'building')) {
    return {text: `Recognising the match: ${b && b.stage ? lowerFirst(String(b.stage)) : 'starting'}${progress !== null ? ` \u00b7 ${progress}%` : ''}`, busy: true, title: ''};
  }
  if (mi.error) return {text: 'Match state unavailable', error: true, title: mi.error};
  if (mi.phase === 'loading') return {text: 'Reading the match state\u2026', busy: true, title: ''};
  return null;
}
function drawStatus() {
  const el = byId('mtMsg');
  if (!el) return;
  const s = mi.on || mi.error ? statusLine() : null;
  el.hidden = !s;
  if (!s) return;
  el.classList.toggle('error', !!s.error);
  el.replaceChildren(mk('i'), mk('span', null, s.text));
  el.title = s.title || s.text;
}

// --- asking for the state, building it, following it
function intelDraw(m) {
  // everything that shows the match state: the transport's layers, the halftime, the status line, the drawers
  if (!m || player !== m || mi.m !== m) return;
  const p = mtParts(), on = !mi.off && (!!mi.intel || !!halfSaved());
  mi.on = on;
  if (p) p.box.hidden = !on;
  const tr = byId('transport');
  if (tr && tr.classList.contains('mt-on') !== on) { tr.classList.toggle('mt-on', on); callPage('fitStage'); }
  if (p) {
    p.box.classList.toggle('stale', on && !!mi.ready && !(mi.intel && mi.intel.state === 'ready' && mi.phase === 'idle'));
    drawLayers(m);
  }
  filterPaint();
  drawHalf(); updatePop(); drawStatus(); eventsDrawer(byId('matchEvents'));
  if (mi.rv.box && (mi.rv.index !== m.index || mi.rv.events !== mi.ready)) reviewRedraw(m);
}
function intelOpen(m) {
  clearTimeout(mi.timer); clearTimeout(mi.panelsTimer);
  Object.assign(mi, freshIntel(m));
  mi.phase = 'loading';
  intelDraw(m);
  const state = m.index && m.index.intel && m.index.intel.state;
  if (state === 'missing' || state === 'people') intelBuild(m, 'building');
  else intelFetch(m);
}
function intelStop() {
  clearTimeout(mi.timer); clearTimeout(mi.panelsTimer);
  Object.assign(mi, freshIntel(null));
  if (mt) {
    mt.strip.replaceChildren(); mt.marks.replaceChildren(); mt.scores.replaceChildren(); mt.goals = [];
    for (const k of Object.keys(mt.handles)) { mt.handles[k].hidden = true; mt.lines[k].hidden = true; }
    mt.band.hidden = true; mt.tip.hidden = true; mt.pop.hidden = true;
  }
  filterPaint();   // mi.on is false now: the filter's button and popover are hidden
  const box = byId('mt');
  if (box) box.hidden = true;
  const tr = byId('transport');
  if (tr && tr.classList.contains('mt-on')) { tr.classList.remove('mt-on'); callPage('fitStage'); }
  const msg = byId('mtMsg');
  if (msg) msg.hidden = true;
  const ev = byId('matchEvents');
  if (ev) ev.replaceChildren(mk('span', 'muted', 'Open a full match to review its events.'));
}
function intelPoll(m) {
  clearTimeout(mi.timer);
  mi.timer = setTimeout(() => { if (player === m && mi.m === m) intelFetch(m); }, INTEL_POLL_MS);
}
async function intelFetch(m) {
  const seq = ++mi.seq;
  let answer;
  try { answer = await api(`/api/match/intel?id=${encodeURIComponent(m.id)}`); } catch (error) {
    if (player === m && mi.m === m && seq === mi.seq) intelFailed(m, error);
    return;
  }
  if (player === m && mi.m === m && seq === mi.seq) intelTake(m, answer);
}
async function intelBuild(m, phase) {
  mi.phase = phase; mi.builds++;
  const seq = ++mi.seq;
  intelDraw(m);
  let answer;
  try { answer = await api('/api/match/intel/build', {match: m.id}); } catch (error) {
    if (player === m && mi.m === m && seq === mi.seq) intelFailed(m, error);
    return;
  }
  if (player === m && mi.m === m && seq === mi.seq) intelTake(m, answer);
}
function intelFailed(m, error) {
  clearTimeout(mi.timer);
  if (error && error.status === 404 && !mi.intel) {
    mi.off = true; mi.phase = 'idle';   // no match state on this server (or for this match): the page as before
    console.warn('Full match: no match state:', error.message || error);
  } else if ((mi.phase === 'building' || mi.phase === 'updating') && mi.fails < 3) {
    mi.fails++;
    intelPoll(m);                       // a request lost while the state is made: asked again
  } else {
    mi.phase = 'idle';
    mi.error = `The match state could not be read: ${(error && error.message) || error}`;
  }
  intelDraw(m);
}
function intelTake(m, a) {
  if (player !== m || mi.m !== m) return;
  if (!a || typeof a !== 'object') { intelFailed(m, new Error('the server sent no match state')); return; }
  mi.intel = a; mi.fails = 0;
  if (a.state === 'ready') {
    mi.ready = a; mi.error = null;
    if (mi.phase === 'building' || mi.phase === 'updating') { mi.phase = 'reloading'; intelReload(m); }   // the index (people, halftime, directions) follows the state
    else if (mi.phase !== 'reloading') mi.phase = 'idle';
  } else if (a.state === 'building') {
    mi.error = null;
    if (mi.phase !== 'updating') mi.phase = 'building';
    intelPoll(m);
  } else if (a.state === 'missing') {
    if (mi.builds < 6) { intelBuild(m, mi.phase === 'updating' ? 'updating' : 'building'); return; }
    mi.phase = 'idle'; mi.error = 'The match state was not made (asked several times).';
  } else {
    mi.phase = 'idle';
    mi.error = a.error || (a.build && a.build.error) || 'The match state could not be made.';
  }
  intelDraw(m);
}
async function intelReload(m) {
  const seq = ++mi.seq;
  let index;
  try { index = await api(`/api/match/index?id=${encodeURIComponent(m.id)}`); } catch (error) {
    if (player === m && mi.m === m && seq === mi.seq) { mi.phase = 'idle'; mi.error = `The match state is ready but the match could not be read again: ${error.message || error}`; intelDraw(m); }
    return;
  }
  if (player !== m || mi.m !== m || seq !== mi.seq) return;
  mi.phase = 'idle';
  if (index && Array.isArray(index.segments)) applyIndex(m, index); else intelDraw(m);
}
function intelIndexChanged(m) {
  // a new index (a correction made or undone, or the state made again): a state made for other people is made again
  if (mi.m !== m || player !== m) return;
  const state = m.index && m.index.intel && m.index.intel.state;
  if (!mi.off && mi.phase === 'idle' && (state === 'missing' || state === 'people')) {
    if (mi.builds < 6) {
      // the correction route started the state's build already: it is asked for until ready (one still missing is
      // built then: intelTake)
      mi.builds++; mi.phase = mi.ready ? 'updating' : 'building';
      intelDraw(m);
      intelFetch(m);
      return;
    }
    mi.error = 'The match state keeps falling behind the corrections: reopen the match.';
  }
  if (state === 'ready') mi.builds = 0;
  intelDraw(m);
}
function panelsLater(m) {
  // an event correction changes what the analyses count (the masks): the panels get the match again, once
  clearTimeout(mi.panelsTimer);
  mi.panelsTimer = setTimeout(() => { if (player === m && mi.m === m) loadPanels(m); }, PANELS_MS);
}

// --- the events drawer
function canWrite() { return !!mi.ready && !!mi.intel && mi.intel.state === 'ready' && mi.phase === 'idle' && !mi.busy; }
async function eventPost(m, path, body, done) {
  if (mi.busy || player !== m || mi.m !== m) return;
  mi.busy = true; mi.message = ''; mi.messageError = false;
  intelDraw(m);
  try {
    const answer = await api(path, {match: m.id, ...body});
    if (player !== m || mi.m !== m) return;
    mi.busy = false; mi.message = done; mi.seq++;
    if (answer && answer.intel) intelTake(m, answer.intel);
    panelsLater(m);
  } catch (error) {
    if (player !== m || mi.m !== m) return;
    mi.message = error.message || String(error); mi.messageError = true;
  } finally {
    mi.busy = false;
    if (player === m && mi.m === m) intelDraw(m);
  }
}
function eventFixText(c) {
  const events = (mi.ready && mi.ready.events) || [], name = (type) => (EVENT_WORDS[type] || type || 'event').toLowerCase();
  const e = (c.kind === 'add' ? c.event : null) || events.find((x) => x.id === c.id) || null, what = e ? `${name(e.type)} at ${clockText(e.t)}` : `event ${c.id || ''}`.trim();
  if (c.kind === 'confirm') return `Confirmed the ${what}`;
  if (c.kind === 'remove') return `Removed the ${what}`;
  if (c.kind === 'retype') return `The event at ${e ? clockText(e.t) : '?'} is ${article(name(c.type))} ${name(c.type)}`;
  if (c.kind === 'move') return `${upperFirst(e ? name(e.type) : 'event')} moved to ${clockText(c.t)}`;
  if (c.kind === 'add') {
    const type = name(c.event && c.event.type), team = c.event && (c.event.team === 'A' || c.event.team === 'B') ? ` (${teamLabel(c.event.team)})` : '';
    return `Added ${article(type)} ${type} at ${clockText(c.event && c.event.t)}${team}`;
  }
  if (c.kind === 'state') return `${clockText(c.t0)}\u2013${clockText(c.t1)} set to ${(STATE_WORDS[c.state] || String(c.state || '')).toLowerCase()}`;
  return `${c.kind || 'A correction'} of the ${what}`;
}
function readyText(a) {
  const ev = shownEvents(a), review = ev.filter((e) => e.status === 'candidate').length, mine = ev.filter((e) => e.status === 'confirmed').length;
  return `Match state ready \u00b7 ${plural(ev.length, 'event')}${review ? ` \u00b7 ${review} to review` : ''}${mine ? ` \u00b7 ${mine} confirmed or added by you` : ''}`;
}
function statsLine(a) {
  const share = a && a.stats && a.stats.share;
  if (!share) return null;
  const box = mk('div', 'me-stats');
  for (const [k, word, cls] of [['IN_PLAY', 'in play', 's-play'], ['STOPPAGE', 'stoppage', 's-stop'], ['POSSIBLE_STOPPAGE', 'possible stoppage', 's-maybe'],
                                ['UNKNOWN', 'unknown', 's-unknown'], ['HALFTIME', 'halftime', 's-half']]) {
    if (!isNum(share[k]) || (k === 'HALFTIME' && !share[k])) continue;
    const s = mk('span');
    s.dataset.share = k;
    s.append(mk('i', `me-sw ${cls}`), mk('b', null, `${Math.round(share[k] * 100)}%`), document.createTextNode(` ${word}`));
    box.append(s);
  }
  if (isNum(a.stats.seconds)) box.append(mk('span', null, `of ${spanText(a.stats.seconds)} of match`));
  return box;
}
function legendLine() {
  const box = mk('div', 'me-legend');
  const sw = (cls, word) => { const s = mk('span'); s.append(mk('i', `me-sw ${cls}`), document.createTextNode(word)); return s; };
  const key = (kind, word, hollow) => { const s = mk('span'), k = mk('span', `me-key k-${kind}${hollow ? ' st-candidate' : ''}`); k.append(mk('i')); s.append(k, document.createTextNode(` ${word}`)); return s; };
  box.append(mk('span', 'mp-k', 'On the timeline'), sw('s-play', 'in play'), sw('s-stop', 'stoppage'), sw('s-prep', 'restart preparation'), sw('s-maybe', 'possible stoppage'),
             sw('s-half', 'halftime'), sw('s-nopic', 'no picture'), sw('s-unknown', 'unknown'), key('goal', 'goal'), key('sub', 'substitution'), key('ko', 'kick-off'),
             key('restart', 'other restart'), key('stop', 'stoppage'), key('restart', 'hollow: to review', true));
  return box;
}
function eventRow(m, e) {
  const kind = kindOf(e.type), row = mk('div', `me-row k-${kind} st-${e.status || 'detected'}`), word = EVENT_WORDS[e.type] || e.type, can = canWrite();
  row.dataset.ev = String(e.id);
  mi.rows.set(String(e.id), row);
  const head = mk('div', 'me-head'), time = btn(clockText(e.t), () => seekTo(e.t), 'me-time'), type = mk('span', 'me-type'), icon = mk('span', `me-key k-${kind}`);
  time.dataset.act = 'seek';
  time.title = `Seek to ${clockText(e.t)} (video time)`;
  time.setAttribute('aria-label', `Seek to the ${word.toLowerCase()} at ${clockText(e.t)}`);
  icon.append(mk('i'));
  type.append(icon, document.createTextNode(word));
  head.append(time, type);
  const who = eventWho(m, e);
  if (who) head.append(mk('span', 'me-team', who));
  head.append(mk('span', 'me-st', statusShort(e)));
  row.append(head);
  if (e.status === 'removed') {
    const fix = ((mi.intel && mi.intel.corrections) || []).find((c) => c.kind === 'remove' && c.id === e.id);
    if (fix) {
      const tools = mk('div', 'me-tools'), undo = btn('Undo the removal', () => eventPost(m, '/api/match/event/remove', {cid: fix.cid}, `Undone: ${eventFixText(fix)}.`));
      undo.dataset.act = 'undo'; undo.disabled = !can;
      tools.append(undo);
      row.append(tools);
    }
    return row;
  }
  const tools = mk('div', 'me-tools');
  const confirm = btn(e.status === 'confirmed' ? 'Confirmed' : 'Confirm', () => eventPost(m, '/api/match/event', {kind: 'confirm', id: e.id}, `Confirmed the ${word.toLowerCase()} at ${clockText(e.t)}.`));
  confirm.dataset.act = 'confirm'; confirm.disabled = !can || e.status === 'confirmed';
  const sel = mk('select');
  sel.dataset.act = 'type';
  sel.setAttribute('aria-label', `Type of the event at ${clockText(e.t)}`);
  for (const t of EVENT_TYPES) { const o = mk('option', null, EVENT_WORDS[t]); o.value = t; sel.append(o); }
  sel.value = e.type; sel.disabled = !can;
  sel.addEventListener('change', () => {
    if (sel.value && sel.value !== e.type) eventPost(m, '/api/match/event', {kind: 'retype', id: e.id, type: sel.value}, `The event at ${clockText(e.t)} is ${article(EVENT_WORDS[sel.value])} ${EVENT_WORDS[sel.value].toLowerCase()}.`);
  });
  const move = btn('Set to now', () => { const t = nowTime(); eventPost(m, '/api/match/event', {kind: 'move', id: e.id, t}, `${word} moved to ${clockText(t)}.`); });
  move.dataset.act = 'move'; move.disabled = !can;
  move.title = 'Move the event to the current playback time';
  const remove = btn('Remove', () => eventPost(m, '/api/match/event', {kind: 'remove', id: e.id}, `Removed the ${word.toLowerCase()} at ${clockText(e.t)}.`), 'quiet');
  remove.dataset.act = 'remove'; remove.disabled = !can;
  remove.title = 'Not a real event: hidden from the timeline (Undo below brings it back)';
  tools.append(confirm, sel, move, remove);
  const evidence = Array.isArray(e.evidence) ? e.evidence : [];
  if (evidence.length) {
    const open = mi.evidence.has(String(e.id)), list = mk('ul', 'me-ev'), toggle = btn(`Evidence (${evidence.length})`, () => {
      const on = !mi.evidence.has(String(e.id));
      if (on) mi.evidence.add(String(e.id)); else mi.evidence.delete(String(e.id));
      list.hidden = !on;
      toggle.setAttribute('aria-expanded', String(on));
    }, 'quiet');
    toggle.dataset.act = 'evidence';
    toggle.setAttribute('aria-expanded', String(open));
    for (const line of evidence) list.append(mk('li', null, String(line)));
    list.hidden = !open;
    tools.append(toggle);
    row.append(tools, list);
  } else row.append(tools);
  return row;
}
function addForm(m) {
  const box = mk('div', 'me-add'), type = mk('select'), team = mk('select');
  box.append(mk('span', 'mp-k', 'Add a missing event at the playback time'));
  type.dataset.add = 'type'; team.dataset.add = 'team';
  type.setAttribute('aria-label', 'Type of the event to add');
  team.setAttribute('aria-label', 'Team of the event to add (for a goal: the team that scored)');
  for (const t of EVENT_TYPES) { const o = mk('option', null, EVENT_WORDS[t]); o.value = t; type.append(o); }
  type.value = mi.addType;
  const fillTeam = () => {
    const need = TEAM_EVENTS.includes(mi.addType);
    if (need && mi.addTeam !== 'A' && mi.addTeam !== 'B') mi.addTeam = 'A';
    team.replaceChildren(...[['A', teamLabel('A')], ['B', teamLabel('B')]].concat(need ? [] : [['', 'No team']]).map(([v, l]) => { const o = mk('option', null, l); o.value = v; return o; }));
    team.value = mi.addTeam;
  };
  fillTeam();
  type.addEventListener('change', () => { mi.addType = type.value; fillTeam(); });
  team.addEventListener('change', () => { mi.addTeam = team.value; });
  const go = btn('Add at current time', () => {
    const t = nowTime(), body = {kind: 'add', type: mi.addType, t}, word = (EVENT_WORDS[mi.addType] || mi.addType).toLowerCase();
    if (mi.addTeam === 'A' || mi.addTeam === 'B') body.team = mi.addTeam;
    eventPost(m, '/api/match/event', body, `Added ${article(word)} ${word} at ${clockText(t)}${body.team ? ` (${teamLabel(body.team)})` : ''}.`);
  }, 'primary');
  go.dataset.act = 'add'; go.disabled = !canWrite();
  go.title = 'Adds the event at the current playback time';
  box.append(type, team, go);
  return box;
}
function eventFixes(m) {
  const list = mi.intel && Array.isArray(mi.intel.corrections) ? mi.intel.corrections : [], box = mk('div', 'me-fixes');
  box.append(mk('h3', null, `Corrections of the events \u00b7 ${list.length}`));
  if (!list.length) { box.append(mk('p', 'muted', 'None. Everything works without a review; a correction made here can be undone.')); return box; }
  for (const c of list) {
    const row = mk('div', 'me-fix'), undo = btn('Undo', () => eventPost(m, '/api/match/event/remove', {cid: c.cid}, `Undone: ${eventFixText(c)}.`));
    undo.dataset.act = 'undo'; undo.dataset.cid = String(c.cid || ''); undo.disabled = !canWrite() || !c.cid;
    row.append(mk('span', null, `${eventFixText(c)}${c.at ? ` \u00b7 ${whenText(c.at)}` : ''}`), undo);
    box.append(row);
  }
  return box;
}
function eventsDrawer(box) {
  const m = player;
  mi.rows = new Map();
  if (!box) return false;
  if (!m || mi.m !== m) { box.replaceChildren(mk('span', 'muted', 'Open a full match to review its events.')); return false; }
  box.replaceChildren();
  if (mi.off) { box.append(mk('span', 'muted', 'This server has no match state for this match: no events and no halftime control.')); return true; }
  const s = statusLine(), a = mi.ready, status = mk('div', `me-status${s && s.busy ? ' busy' : ''}${s && s.error ? ' error' : ''}`);
  status.setAttribute('role', 'status');
  status.append(mk('i'), mk('span', null, s ? (s.error ? `${s.text}: ${s.title}` : s.text) : a ? readyText(a) : 'No match state yet.'));
  box.append(status);
  if (!a) {
    box.append(mk('p', 'muted', mi.error ? 'Everything else keeps working: the tracking, the 2D match, analytics and the reports.' : 'The events appear here when the match state is ready.'));
    return true;
  }
  const stats = statsLine(a);
  if (stats) box.append(stats);
  box.append(legendLine());
  const dir = mk('div', 'me-dir'), note = a.directionCheck && a.directionCheck.note;
  if (note) { const n = mk('p', 'me-note', note); n.setAttribute('role', 'note'); dir.append(n); }
  for (const line of directionLines(halfSaved(), a.directionCheck)) dir.append(mk('p', line.warn ? 'warn' : null, line.text));
  box.append(dir);
  if (mi.message) { const msg = mk('p', `me-msg${mi.messageError ? ' error' : ''}`, mi.message); msg.setAttribute('role', mi.messageError ? 'alert' : 'status'); box.append(msg); }
  if (mi.phase !== 'idle') box.append(mk('p', 'me-msg', 'The match state is being updated: corrections wait until it is ready.'));
  const all = (Array.isArray(a.events) ? a.events : []).filter((e) => e && isNum(e.t)).slice().sort((x, y) => x.t - y.t);
  const keep = (key, test) => (e) => (key === 'removed' ? e.status === 'removed' : e.status !== 'removed') && test(e);
  const bar = mk('div', 'me-bar'), sel = mk('select');
  sel.dataset.me = 'filter';
  sel.setAttribute('aria-label', 'Show the events of this kind');
  for (const [key, label, test] of EVENT_FILTERS) {
    const n = all.filter(keep(key, test)).length;
    if (key !== 'all' && key !== mi.filter && !n) continue;
    const o = mk('option', null, `${label} (${n})`);
    o.value = key;
    sel.append(o);
  }
  sel.value = mi.filter;
  sel.addEventListener('change', () => { mi.filter = sel.value; eventsDrawer(byId('matchEvents')); });
  bar.append(sel);
  box.append(bar);
  const [key, , test] = EVENT_FILTERS.find(([k]) => k === mi.filter) || EVENT_FILTERS[0], shown = all.filter(keep(key, test)), list = mk('div', 'me-list');
  for (const e of shown) list.append(eventRow(m, e));
  if (!shown.length) list.append(mk('p', 'muted', 'No events of this kind.'));
  box.append(list, addForm(m), eventFixes(m));
  return true;
}

// --- the roster review (the Match identities drawer)
function hasPlaces(m) { const p = m && m.index && m.index.places; return !!p && typeof p === 'object' && (Array.isArray(p.A) || Array.isArray(p.B)); }
function fragmentMap(m) {
  if (!m.fragMap || m.fragMap.index !== m.index) {
    const map = new Map();
    for (const f of (m.index && m.index.fragments) || []) if (f && typeof f.matchId === 'string') map.set(f.matchId, f);
    m.fragMap = {index: m.index, map};
  }
  return m.fragMap.map;
}
function nodeAt(nodes, t) {
  // the node covering t, else the last one before it
  let best = -1;
  for (let k = 0; k < nodes.length; k++) {
    const n = nodes[k];
    if (!n || !isNum(n.t0) || !isNum(n.t1)) continue;
    if (n.t0 <= t && t <= n.t1) return {j: k, covering: true};
    if (n.t1 < t && (best < 0 || n.t1 >= nodes[best].t1)) best = k;
  }
  return {j: best, covering: false};
}
function slotsAt(m, team, t) {
  const places = m.index && m.index.places && Array.isArray(m.index.places[team]) ? m.index.places[team] : [], out = [];
  for (let slot = 1; slot <= 10; slot++) {
    const place = places.find((p) => p && p.slot === slot), nodes = place && Array.isArray(place.nodes) ? place.nodes : [], at = nodeAt(nodes, t);
    const next = nodes.filter((n) => n && isNum(n.t0) && n.t0 > t).sort((a, b) => a.t0 - b.t0)[0] || null;
    out.push({team, slot, nodes, j: at.j, covering: at.covering, node: at.j >= 0 ? nodes[at.j] : null, next});
  }
  return out;
}
function keeperAt(m, team, t) {
  // the team's goalkeeper at t: the keeper person observed then (the sure one first), else the one seen last before it
  const frags = fragmentMap(m), keepers = rosterPeople(m).filter((p) => p.team === team && p.slot === 'GK' && !p.excluded && !m.ctx.out(p.matchId, p.matchId));
  let best = null;
  for (const p of keepers) {
    const spans = (Array.isArray(p.fragments) ? p.fragments : []).map((f) => frags.get(f)).filter((f) => f && isNum(f.firstSeen) && isNum(f.lastSeen)).map((f) => [f.firstSeen, f.lastSeen]);
    if (!spans.length && isNum(p.firstSeen) && isNum(p.lastSeen)) spans.push([p.firstSeen, p.lastSeen]);
    const covering = spans.some(([a, b]) => a <= t && t <= b), before = spans.filter(([, b]) => b < t).map(([, b]) => b);
    const last = before.length ? Math.max(...before) : null, first = spans.length ? Math.min(...spans.map(([a]) => a)) : null;
    const rank = covering ? [2, p.provisional ? 0 : 1] : last !== null ? [1, last] : [0, first === null ? -Infinity : -first];
    if (!best || rank[0] > best.rank[0] || (rank[0] === best.rank[0] && rank[1] > best.rank[1])) best = {p, covering, last, first, rank};
  }
  return best ? {person: best.p, covering: best.covering, last: best.last, first: best.first, others: keepers.filter((q) => q !== best.p)} : {person: null, others: []};
}
function slotsSig(m, t) {
  const parts = [];
  for (const team of ['A', 'B']) {
    for (const r of slotsAt(m, team, t)) parts.push(`${r.node ? r.node.node : '-'}${r.covering ? '+' : ''}`);
    const k = keeperAt(m, team, t);
    parts.push(`${k.person ? k.person.matchId : '-'}${k.covering ? '+' : ''}`);
  }
  return parts.join('|');
}
function reviewRedraw(m) { if (mi.rv.box && mi.m === m) reviewSection(m, mi.rv.box); }
function reviewTick(m, t) {
  // the review follows the playback time: its title every second, its rows when a slot changes hands or is lost
  const rv = mi.rv;
  if (mi.m !== m || !rv.box || !hasPlaces(m)) return;
  if (rv.box.isConnected === false) { rv.box = null; return; }
  let shown = true;
  try { shown = drawers.current === 'edit'; } catch (e) { shown = true; }
  const sec = Math.floor(t);
  if (!shown || sec === rv.second) return;
  rv.second = sec;
  if (slotsSig(m, t) !== rv.sig) { reviewSection(m, rv.box); return; }
  if (rv.title) setText(rv.title, `Roster review \u00b7 at ${clockText(t)}`);
}
function reviewSection(m, rv) {
  const t = nowTime();
  Object.assign(mi.rv, {box: rv, sig: slotsSig(m, t), second: Math.floor(t), index: m.index, events: mi.ready});
  rv.replaceChildren();
  mi.rv.title = mk('h3', 'rv-title', `Roster review \u00b7 at ${clockText(t)}`);
  rv.append(mi.rv.title, mk('p', 'muted rv-lede', 'Per team, who holds each on-field slot at the playback time: the match-wide person, when they were last observed, and how sure the slot is (sure: the player who began it, or confirmed by you; estimate: the best continuation of the place).'));
  for (const team of ['A', 'B']) {
    const box = mk('div', 'rv-team');
    box.dataset.team = team;
    box.append(mk('h4', null, `${teamLabel(team)} \u00b7 slots 01\u201310 and GK`));
    for (const r of slotsAt(m, team, t)) box.append(slotRow(m, r));
    box.append(keeperRow(m, team, t));
    rv.append(box);
  }
  rv.append(reviewUnresolved(m), reviewLeftOut(m));
}
function rowToggle(m, key) {
  const open = mi.rv.open.has(key), b = btn(open ? 'Close' : 'Correct', () => { if (mi.rv.open.has(key)) mi.rv.open.delete(key); else mi.rv.open.add(key); reviewRedraw(m); }, 'quiet');
  b.dataset.act = 'correct';
  b.setAttribute('aria-expanded', String(open));
  return b;
}
function rowError(row, key) {
  const error = drawerState.rowError.get(key);
  if (error) { const e = mk('p', 'mr-msg error', error); e.setAttribute('role', 'alert'); row.append(e); }
}
function reviewShow(m, p, t) {
  // a person of the review: highlighted on every view, and the video at its last observation when it is not on now
  if (isNum(t)) seekTo(t);
  if (p && !p.excluded && !m.ctx.out(p.matchId, p.matchId)) callPage('select', p.matchId);
}
function slotRow(m, r) {
  const key = `rv:${r.team}:${r.slot}`, row = mk('div', 'rv-row');
  row.dataset.rv = 'slot'; row.dataset.team = r.team; row.dataset.slot = String(r.slot);
  row.append(mk('span', 'rv-slot', pad2(r.slot)));
  if (!r.node) { row.append(mk('span', 'rv-none', r.next ? `Nobody yet \u00b7 first held from ${clockText(r.next.t0)}` : 'Nobody holds this slot in the match')); return row; }
  const n = r.node, frs = Array.isArray(n.fragments) ? n.fragments : [], pid = m.ctx.resolve(n.person || frs[0]), p = m.ctx.roster.get(pid) || null;
  const who = btn(p ? nameOf(p) : (n.person || `fragment ${frs[0]}`), () => reviewShow(m, p, r.covering ? null : n.t1), 'rv-who');
  who.dataset.id = pid || '';
  who.title = `Held slot ${pad2(r.slot)} ${clockText(n.t0)}\u2013${clockText(n.t1)}${r.covering ? '' : ` \u00b7 jump to its last observation (${clockText(n.t1)})`}`;
  try { if (selected && selected === pid) who.classList.add('active'); } catch (e) { /* no selection yet */ }
  const openKey = `${key}:${n.node}`, toggle = rowToggle(m, openKey);
  toggle.disabled = !p;
  row.append(who, toggle);
  const meta = mk('span', 'rv-meta');
  meta.append(mk('span', 'rv-seen', r.covering ? 'Seen now' : `Not seen since ${clockText(n.t1)}`), document.createTextNode(' \u00b7 '),
              mk('span', n.sure ? 'rv-sure' : 'rv-est', n.sure ? 'sure' : `estimate${isNum(n.p) ? ` (link ${Math.round(n.p * 100)}%)` : ''}`),
              document.createTextNode(` \u00b7 ${frs.length > 1 ? 'fragments' : 'fragment'} ${frs.join(', ')}`));
  row.append(meta);
  const subs = slotSubs(m, r.team, r.slot, key);
  if (subs) row.append(subs);
  if (p && mi.rv.open.has(openKey)) {
    const tools = mk('div', 'mr-tools rv-tools'), prev = r.j > 0 ? r.nodes[r.j - 1] : null, next = r.j + 1 < r.nodes.length ? r.nodes[r.j + 1] : null;
    const from = [prev && prev.fragments && prev.fragments[prev.fragments.length - 1], next && next.fragments && next.fragments[0]].filter(Boolean);
    const split = btn('Not this player', () => correct(m, {kind: 'split', target: frs[0], from}, `${nameOf(p)} no longer continues slot ${pad2(r.slot)}.`, key));
    split.dataset.act = 'split'; split.disabled = drawerState.busy || !from.length || !frs.length;
    split.title = from.length ? `Fragment ${frs[0]} is not the same player as ${from.map((f) => `fragment ${f}`).join(' or ')} (before and after it in this slot)` : 'Nobody before or after it in this slot';
    tools.append(joinTool(m, p, key), split, slotConfirm(m, p.matchId, r.slot, p, key), ...personTools(m, p, key));
    row.append(tools);
  }
  rowError(row, key);
  return row;
}
function slotSubs(m, team, slot, key) {
  // the slot's substitutions: the state's SUBSTITUTION events and the ones confirmed or rejected by you
  const events = (mi.ready && Array.isArray(mi.ready.events) ? mi.ready.events : []).filter((e) => e && e.type === 'SUBSTITUTION' && e.team === team && Number(e.slot) === slot && e.status !== 'removed');
  const said = m.index.people && Array.isArray(m.index.people.substitutions) ? m.index.people.substitutions : [];
  const who = (f, p) => (f ? m.ctx.resolve(f) : p || null);
  const decisionOf = (out, outP, inn, inP) => said.find((s) => (s.outFragment === out && s.inFragment === inn) ||
    (who(s.outFragment, s.out) && who(s.outFragment, s.out) === who(out, outP) && who(s.inFragment, s.in) && who(s.inFragment, s.in) === who(inn, inP))) || null;
  const items = events.map((e) => ({t: e.t, event: e, outRef: e.outFragment || e.outPerson, inRef: e.inFragment || e.inPerson, slots: subSlots(e), people: subPeople(m, e),
    out: personName(m, e.outFragment, e.outPerson), inn: personName(m, e.inFragment, e.inPerson), decision: decisionOf(e.outFragment, e.outPerson, e.inFragment, e.inPerson)}));
  const frags = fragmentMap(m);
  for (const s of said) {
    if (items.some((x) => x.decision === s)) continue;
    const f = frags.get(s.inFragment), place = f && f.place;
    if (place && place.team === team && place.slot === slot) {
      items.push({t: f.firstSeen, event: null, outRef: s.outFragment || s.out, inRef: s.inFragment || s.in, out: personName(m, s.outFragment, s.out), inn: personName(m, s.inFragment, s.in), decision: s});
    }
  }
  if (!items.length) return null;
  items.sort((a, b) => (a.t || 0) - (b.t || 0));
  const box = mk('div', 'rv-subs');
  for (const it of items) {
    const line = mk('div', 'rv-sub');
    line.dataset.rv = 'sub';
    const state = it.decision ? (it.decision.status === 'rejected' ? 'rejected by you' : 'confirmed by you') :
      `${it.event.status === 'candidate' ? 'candidate' : it.event.status}${it.event.confidence && it.event.confidence !== 'confirmed' ? `, ${it.event.confidence} confidence` : ''}`;
    line.append(mk('span', null, `Substitution${isNum(it.t) ? ` ${clockText(it.t)}` : ''} \u00b7 ${it.slots ? `${it.slots} \u00b7 ${it.people}` : `${it.out} \u2192 ${it.inn}`} \u00b7 ${state}`));
    for (const [status, label] of [['confirmed', 'Confirm'], ['rejected', 'Reject']]) {
      const b = btn(label, () => correct(m, {kind: 'substitution', out: it.outRef, in: it.inRef, status}, `Substitution ${it.out} \u2192 ${it.inn} ${status} (${teamLabel(team)}, slot ${pad2(slot)}).`, key), 'quiet');
      b.dataset.act = status === 'confirmed' ? 'sub-confirm' : 'sub-reject';
      b.disabled = drawerState.busy || !it.outRef || !it.inRef || !!(it.decision && it.decision.status === status);
      line.append(b);
    }
    box.append(line);
  }
  return box;
}
function joinTool(m, p, key) {
  const wrap = mk('span', 'rv-join'), sel = mk('select'), first = mk('option', null, 'Same player as\u2026');
  sel.dataset.act = 'join-into';
  sel.setAttribute('aria-label', `Same player as (for ${nameOf(p)})`);
  first.value = '';
  sel.append(first);
  for (const {q, g} of candidates(m, p)) {
    const o = mk('option', null, `${nameOf(q)} \u00b7 ${clockText(q.firstSeen)}\u2013${clockText(q.lastSeen)}${g < 0 ? ' \u00b7 overlaps in time' : ''}`);
    o.value = q.matchId;
    sel.append(o);
  }
  const join = btn('Join', () => { if (sel.value) correct(m, {kind: 'merge', target: p.matchId, into: sel.value}, `Joined ${nameOf(p)} and ${displayOf(m, sel.value)}: one player.`, key); });
  join.dataset.act = 'join'; join.disabled = true;
  sel.addEventListener('change', () => { join.disabled = !sel.value || drawerState.busy; });
  sel.disabled = drawerState.busy;
  wrap.append(sel, join);
  return wrap;
}
function slotConfirm(m, target, slot, p, key) {
  // target: the person (the server applies it to all of the person's fragments)
  const wrap = mk('span', 'rv-confirm'), sel = mk('select');
  sel.dataset.act = 'slot-of';
  sel.setAttribute('aria-label', `The slot whose player ${nameOf(p)} is`);
  for (let s = 1; s <= 10; s++) { const o = mk('option', null, `slot ${pad2(s)}`); o.value = String(s); sel.append(o); }
  sel.value = String(slot || 1); sel.disabled = drawerState.busy;
  const go = btn('Confirm', () => {
    const n = Number(sel.value);
    correct(m, {kind: 'confirm', target, slot: n}, `${nameOf(p)} is the player of slot ${pad2(n)} (${teamLabel(p.team)}).`, key);
  });
  go.dataset.act = 'confirm-slot'; go.disabled = drawerState.busy || !target;
  go.title = 'This person is the player who began that slot: one person with him';
  wrap.append(mk('span', null, 'This is the player of'), sel, go);
  return wrap;
}
function personTools(m, p, key) {
  const out = [], exclude = btn('Not a player', () => correct(m, {kind: 'exclude', target: p.matchId}, `${nameOf(p)} is marked as not a player.`, key));
  exclude.dataset.act = 'exclude';
  exclude.title = 'A referee, staff, a spectator or a graphic: left out of the people and the analysis (Include under Left out takes it back)';
  out.push(exclude);
  for (const team of p.team === 'A' ? ['B'] : p.team === 'B' ? ['A'] : ['A', 'B']) {
    const b = btn(p.team === 'A' || p.team === 'B' ? `Change team to ${team}` : `Team ${team}`, () => correct(m, {kind: 'team', target: p.matchId, team}, `${nameOf(p)} moved to team ${team}.`, key));
    b.dataset.act = `team-${team}`;
    out.push(b);
  }
  const keeper = p.role === 'GOALKEEPER' || p.slot === 'GK';
  const k = btn(keeper ? 'Not the goalkeeper' : 'Goalkeeper', () => correct(m, {kind: 'keeper', target: p.matchId, on: !keeper}, `${nameOf(p)} ${keeper ? 'is no longer' : 'is now'} a goalkeeper.`, key));
  k.dataset.act = keeper ? 'keeper-off' : 'keeper-on';
  out.push(k);
  for (const b of out) b.disabled = drawerState.busy;
  return out;
}
function keeperRow(m, team, t) {
  const key = `rv:${team}:GK`, row = mk('div', 'rv-row'), k = keeperAt(m, team, t);
  row.dataset.rv = 'slot'; row.dataset.team = team; row.dataset.slot = 'GK';
  row.append(mk('span', 'rv-slot', 'GK'));
  if (!k.person) { row.append(mk('span', 'rv-none', 'No goalkeeper of this team found')); return row; }
  const p = k.person, frs = Array.isArray(p.fragments) ? p.fragments : [];
  const who = btn(nameOf(p), () => reviewShow(m, p, k.covering ? null : k.last), 'rv-who');
  who.dataset.id = p.matchId;
  try { if (selected && selected === p.matchId) who.classList.add('active'); } catch (e) { /* no selection yet */ }
  const openKey = `${key}:${p.matchId}`;
  row.append(who, rowToggle(m, openKey));
  const meta = mk('span', 'rv-meta');
  meta.append(mk('span', 'rv-seen', k.covering ? 'Seen now' : k.last !== null ? `Not seen since ${clockText(k.last)}` : isNum(k.first) ? `Not seen yet \u00b7 first at ${clockText(k.first)}` : 'Not seen'),
              document.createTextNode(' \u00b7 '), mk('span', p.provisional ? 'rv-est' : 'rv-sure', p.provisional ? 'estimate' : 'sure'),
              document.createTextNode(` \u00b7 ${plural(frs.length, 'fragment')}`));
  row.append(meta);
  if (k.others.length) row.append(mk('span', 'rv-meta', `Other goalkeeper ${k.others.length === 1 ? 'person' : 'people'} of this team: ${k.others.map(nameOf).join(', ')}`));
  if (mi.rv.open.has(openKey)) { const tools = mk('div', 'mr-tools rv-tools'); tools.append(joinTool(m, p, key), ...personTools(m, p, key)); row.append(tools); }
  rowError(row, key);
  return row;
}
function reviewUnresolved(m) {
  const list = rosterPeople(m).filter((p) => !p.excluded && !m.ctx.out(p.matchId, p.matchId) && (p.slot === null || p.slot === undefined) && p.role !== 'GOALKEEPER')
    .sort((a, b) => naturally(a.team || 'Z', b.team || 'Z') || (a.firstSeen || 0) - (b.firstSeen || 0));
  const box = mk('div', 'rv-group');
  box.dataset.rv = 'unresolved';
  box.append(mk('h4', null, `Unresolved fragments \u00b7 ${list.length}`));
  if (!list.length) { box.append(mk('p', 'muted rv-lede', 'Every person of the teams holds a slot.')); return box; }
  box.append(mk('p', 'muted rv-lede', 'People no slot claims (A-? and B-? on the board): join one into the player it is, say whose slot it is, or mark it as not a player.'));
  for (const p of list) {
    const key = `rv:u:${p.matchId}`, row = mk('div', 'rv-row'), frs = Array.isArray(p.fragments) ? p.fragments : [];
    row.dataset.rv = 'unresolved'; row.dataset.person = p.matchId;
    row.append(mk('span', 'rv-slot', p.team === 'A' || p.team === 'B' ? p.team : '\u2013'));
    const who = btn(nameOf(p), () => seekIdentity(m, p.matchId), 'rv-who');
    who.dataset.id = p.matchId;
    who.title = `Jump to its first observation (${clockText(p.firstSeen)})`;
    row.append(who, rowToggle(m, key));
    row.append(mk('span', 'rv-meta', `${teamWord(p.team)} \u00b7 ${frs.length > 1 ? 'fragments' : 'fragment'} ${frs.join(', ')} \u00b7 ${clockText(p.firstSeen)}\u2013${clockText(p.lastSeen)} \u00b7 ${spanText(p.seconds)} observed`));
    // a stitched track that joins two people (a keeper and an outfield player, or two kits): no slot unless you say
    if (p.mixed) {
      const at = isNum(p.mixed.at) ? `at ${clockText(p.mixed.at)}` : 'at a segment start';
      const what = p.mixed.why === 'keeper' ? 'a keeper and an outfield player' : 'two different kits';
      const n = mk('span', 'rv-meta rv-mixed', `Joins two people ${at} (${what}), so no slot takes it; saying whose slot it is places it anyway.`);
      n.setAttribute('role', 'note');
      row.append(n);
    }
    if (mi.rv.open.has(key)) { const tools = mk('div', 'mr-tools rv-tools'); tools.append(joinTool(m, p, key), slotConfirm(m, p.matchId, 1, p, key), ...personTools(m, p, key)); row.append(tools); }
    rowError(row, key);
    box.append(row);
  }
  return box;
}
function reviewLeftOut(m) {
  const list = ((m.index && m.index.roster) || []).filter((p) => p && p.excluded).sort((a, b) => (a.firstSeen || 0) - (b.firstSeen || 0));
  const box = mk('div', 'rv-group');
  box.dataset.rv = 'left-out';
  box.append(mk('h4', null, `Left out \u00b7 ${list.length}`));
  if (!list.length) { box.append(mk('p', 'muted rv-lede', 'Nobody is left out.')); return box; }
  box.append(mk('p', 'muted rv-lede', 'Fragments that are not players (a broadcast graphic, an official or a coach, someone beyond the lines, or marked by you), kept apart with the reason: Include takes one back.'));
  for (const p of list) {
    const key = `rv:x:${p.matchId}`, row = mk('div', 'rv-row');
    row.dataset.rv = 'left-out'; row.dataset.fragment = p.matchId;
    row.append(mk('span', 'rv-slot', p.team === 'A' || p.team === 'B' ? p.team : '\u2013'));
    const show = btn(`fragment ${p.matchId}`, () => seekTo(p.firstSeen), 'rv-who quiet');
    show.title = `Jump to it (${clockText(p.firstSeen)})`;
    const include = btn('Include', () => correct(m, {kind: 'include', target: p.matchId}, `Fragment ${p.matchId} counts as a player again.`, key));
    include.dataset.act = 'include'; include.disabled = drawerState.busy;
    row.append(show, include, mk('span', 'rv-meta', `${upperFirst(p.excludedReason || 'left out')} \u00b7 ${clockText(p.firstSeen)}\u2013${clockText(p.lastSeen)} \u00b7 ${spanText(p.seconds)}`));
    rowError(row, key);
    box.append(row);
  }
  return box;
}

// ---------- wiring ----------
function wire() {
  const on = (id, event, fn) => { const el = byId(id); if (el) el.addEventListener(event, fn); };
  const changed = () => { st.message = ''; st.messageError = false; readSetup(); schedulePlan(); paint(); };
  for (const id of ['mpStartIn', 'mpEndIn', 'mpB0In', 'mpB1In']) {
    on(id, 'input', changed);
    on(id, 'change', () => { changed(); writeSetup(); });
    on(id, 'keydown', (e) => { if (e.key === 'Enter') { e.preventDefault(); e.target.blur(); } });
    on(id, 'blur', () => writeSetup());
  }
  on('mpWhole', 'click', () => { st.setup.start = 0; st.setup.end = null; if (st.setup.half) defaultBreak(); writeSetup(); changed(); });
  on('mpHalf', 'change', () => { st.setup.half = byId('mpHalf').checked; if (st.setup.half && (!isNum(st.setup.b0) || !isNum(st.setup.b1))) defaultBreak(); writeSetup(); changed(); });
  on('mpWorkers', 'change', changed);
  on('mpAnalyze', 'click', startMatch);
  on('mpStartOver', 'click', startMatch);
  on('mpResumeBtn', 'click', resumeMatch);
  on('mpPause', 'click', pauseMatch);
  on('mpStop', 'click', stopMatch);
  on('mpOpen', 'click', () => open());
  on('mpLeave', 'click', () => leave({reload: true}));
  on('heroOpenMatch', 'click', () => open());
  on('matchBack', 'click', () => leave({reload: true}));
  // the processing settings feed the plan too (the server validates them; keyframes set the calibration coverage)
  const proc = byId('procSettings');
  if (proc) for (const event of ['change', 'input']) proc.addEventListener(event, () => { if (dockMode === 'match') schedulePlan(); });
}
function init() {
  if (st.ready) return;
  st.ready = true;
  try { wire(); } catch (error) { console.error('Full match: wiring failed', error); }
  resetSetup();
  st.calSig = calibrationSig();
  setDockMode(dockMode, false);
  refreshOverview();
  loop();
}

wireSwitch();
global.FullMatch = {
  init, open, leave, frame, scope, chip, stage, span, renameView, contextOf, waiting,
  // the coaching panel's run in match playback: the match report's ('match-<id>': its frames and PDF are asked by run)
  coachingRun: () => (player ? `match-${player.id}` : null),
  busy: () => matchBusy(),
  roster: (box) => rosterDrawer(box),
  corrections: (box) => correctionsDrawer(box),
  // the match's state in match playback: the events drawer, and the transport's geometry and state (also for the tests)
  events: (box) => eventsDrawer(box || byId('matchEvents')),
  intel: {
    get state() { return mi; },
    draw: () => { if (player) intelDraw(player); },
    timeToX: (t) => timeToX(t), xToTime: (x) => xToTime(x), timePct: (t) => timePct(t),
    slotsAt: (team, t) => (player && player.index ? slotsAt(player, team, isNum(t) ? t : nowTime()) : null),
    keeperAt: (team, t) => (player && player.index ? keeperAt(player, team, isNum(t) ? t : nowTime()) : null),
  },
  get state() { return st; },
  get dockMode() { return dockMode; },
  setDockMode: (m) => setDockMode(m, true),
};
})(window);
