"""The Coaching report tab: coaching.js run in node (skipped without node) on a report in the contract's shape, and the
page's wiring of the fourth tab in index.html.

The view renders every section of a report (overview, summary, strengths, areas to improve, reviews, breakdown,
priorities, training plan, confidence and limitations), says plainly when there is no report, when one is being built,
when it failed and when nothing met the evidence bar, and when a stored report is stale it says why and offers to
regenerate it. Players are shown by the display label of the result the page shows (the report keys them by id). An
evidence timestamp jumps to the frame's centre, (frame + 0.5)/fps; a verdict is posted with the finding's id and the
report's fingerprint. The page has four views on its loop (the controller counts them instead of assuming three), the
arrow keys on the view switcher switch views without also stepping the video, playback keys are off on the coaching
view, and a jump pauses, seeks, selects and shows the finding's overlay until the video plays or leaves the frame.
Those v1 tests run the page against a server without the coaching v2 routes (legacy mode).

Coaching v2 (DESIGN 9; classes V2*). The mock reports start from schema.example('report') and are checked with
schema.validate('report') first (V2Mocks), so the page is built against the contract. The report shows the seven
sections of DESIGN 9.2 in order; cards follow the level's register (level 1: no counts, no measurements, and no word
level 1 bans anywhere on the page; level 2: counts only; level 4: metres, no shares); a key moment jumps to the frame's
centre with its window, caption and follow list (an image-tier moment opens on the 2D match); the side column shows
the moment's screenshot, overlay and diagram (no suggested area over a frame below 'world'), the validated
alternative or why there is none, and lets the coach adjust the area (the posted polygon is schema-valid). The setup
follows the age group's format default until the coach overrides it, offers formations and roles by format (equal to
profiles.roles_for), refuses a save without age group and level, and posts a schema-valid TeamSettings block; on PROFILE's
real catalogue (profiles.catalog()) it takes the server's ages, formats, defaults, roles and level information. The
palette reproduces the 14 CIEDE2000 test pairs of Sharma et al. (2005) to 4 decimals and keeps its colours 20 or more
away from the team colours (or flags a fallback). The page keeps a v2 annotation through its window (key frame: whole
overlay; elsewhere: the followed people; leaving it, Escape or the chip clear it; Replay plays from the window start)
and scrolls the stage below the sticky header. A mutation check (20 of 21 single-rule mutants of coaching.js and
index.html fail these tests; the survivor is a redundant request guard) was run on a scratch copy.

Wave 7 (classes V2Marks, V2Spans, V2Progression, V2ServerShapes, V2Real and three PageWiring tests). The coach's
possession marks: "our ball / their ball / dead ball from here" and "End" at the video's time (to the hundredth, inside
the analysed segment), the newest mark winning where it overlaps an older one, the list's typed times and removals, a
two-click "Remove all", a failed save kept on the page with "Save again", the team letters following the report's team,
every posted record valid against schema 'coach_spans'; the lane under the possession strip (positions on the strip's
time axis, team colours, the playhead, a click seeks); the report's way to the marks where the phase could not be
followed, and the banner after a save. The answers as SERVER built them (progress detail under 'progress', the match
date, removing a saved area by its adj- id, coach-marked shares not called estimates, the settings note). REAL reports
(coaching.interpret with every stage on EVAL's scenes through TEXT's fixtures) render all seven sections with 0 unmeasured
events and, at level 1, 0 banned words outside the quoted source titles; measurement chips only from the level that may
state them; a key moment's caption once; a real jump to the frame centre. The page: a controls row may serve several
views; the chip's Why shows the finding's explanation and Report goes back.

Run:  python -m unittest test_coaching_panel          (from this directory)
"""
import datetime
import json
import os
import re
import shutil
import subprocess
import tempfile
import unittest

from test_coaching_store import fixture_report, synthetic_result
from soccer.coaching import pdf, store
from soccer.coaching import schema as S
try:                    # PROFILE's catalogue (GET /api/coaching/profiles); built in parallel, so its tests skip without it
    from soccer.coaching import profiles as PROFILES
    CATALOG = PROFILES.catalog()
except Exception:       # noqa: BLE001 - any failure of the other role's module only skips the catalogue tests
    PROFILES = CATALOG = None

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAGE = os.path.join(ROOT, 'index.html')
FILES = {name: os.path.join(ROOT, name) for name in ('tactical.js', 'coaching.js')}
NODE = shutil.which('node')

HARNESS = r"""
const fs = require('fs'), vm = require('vm');
const input = JSON.parse(fs.readFileSync(0, 'utf8'));
const round = (x) => typeof x === 'number' ? Math.round(x * 100) / 100 : typeof x === 'string' ? x : null;
function recorder(log) {
  return new Proxy({}, {get: (o, k) => k in o ? o[k] : k === 'measureText' ? (s) => ({width: String(s).length * 6}) : (...a) => { log.push([k, ...a.map(round)]); },
                        set: (o, k, v) => { o[k] = v; return true; }});
}
class Node {
  constructor(tag) {
    this.tagName = String(tag).toUpperCase(); this.children = []; this.parentElement = null; this.attrs = {}; this.dataset = {}; this.style = {};
    this.listeners = {}; this.hidden = false; this.disabled = false; this.value = ''; this.own = ''; this.cls = ''; this.log = null;
    const self = this;
    this.classList = {add: (...c) => c.forEach((x) => self.mark(x, true)), remove: (...c) => c.forEach((x) => self.mark(x, false)),
                      toggle: (c, on) => { const want = on === undefined ? !self.classes().includes(c) : !!on; self.mark(c, want); return want; },
                      contains: (c) => self.classes().includes(c)};
  }
  get className() { return this.cls; }
  set className(v) { this.cls = String(v); }
  classes() { return this.cls.split(/\s+/).filter(Boolean); }
  mark(c, on) { const s = new Set(this.classes()); if (on) s.add(c); else s.delete(c); this.cls = [...s].join(' '); }
  get textContent() { return this.own + this.children.map((c) => c.textContent).join(''); }
  set textContent(v) { this.own = String(v); this.clear(); }
  clear() { for (const c of this.children) c.parentElement = null; this.children = []; }
  append(...cs) { for (const c of cs) { if (c.parentElement) c.parentElement.removeChild(c); c.parentElement = this; this.children.push(c); } }
  removeChild(c) { const i = this.children.indexOf(c); if (i >= 0) this.children.splice(i, 1); c.parentElement = null; }
  replaceChildren(...cs) { this.own = ''; this.clear(); this.append(...cs); }
  after(c) { const p = this.parentElement; if (!p) return; if (c.parentElement) c.parentElement.removeChild(c); p.children.splice(p.children.indexOf(this) + 1, 0, c); c.parentElement = p; }
  remove() { if (this.parentElement) this.parentElement.removeChild(this); }
  setAttribute(k, v) { this.attrs[k] = String(v); }
  getAttribute(k) { return k in this.attrs ? this.attrs[k] : null; }
  removeAttribute(k) { delete this.attrs[k]; }
  getBoundingClientRect() { return {left: 100, top: 0, width: 1000, height: 10}; }
  addEventListener(t, f) { (this.listeners[t] = this.listeners[t] || []).push(f); }
  fire(t) { const e = {type: t, target: this, preventDefault() {}, stopPropagation() {}}; if (typeof this['on' + t] === 'function') this['on' + t](e); for (const f of this.listeners[t] || []) f(e); }
  click() { this.fire('click'); }
  getContext() { return recorder(this.log = this.log || []); }
}
const all = (n, out = []) => { out.push(n); for (const c of n.children || []) all(c, out); return out; };
const find = (root, pred) => all(root).filter((n) => n instanceof Node && pred(n));
const cls = (n) => (n.attrs.class || n.className || '');
// setTimeout runs at once, unless a job holds the polls (sandbox.hold): they wait in sandbox.held until it runs them.
const sandbox = {document: {createElement: (t) => new Node(t), createElementNS: (ns, t) => new Node(t), body: new Node('body'),
                            createTextNode: (t) => ({nodeType: 3, textContent: String(t), parentElement: null})},
                 console, setTimeout: (f) => sandbox.hold ? (sandbox.held.push(f), 0) : setImmediate(f), clearTimeout: (h) => { if (h) clearImmediate(h); },
                 URLSearchParams, Map, Set, JSON, hold: false, held: [],
                 localStorage: {getItem: () => null, setItem() {}}, devicePixelRatio: 1, addEventListener() {}};
sandbox.window = sandbox;
vm.createContext(sandbox);
for (const path of Object.values(input.files)) vm.runInContext(fs.readFileSync(path, 'utf8'), sandbox, {filename: path});
const settle = async () => { for (let k = 0; k < 40; k++) await new Promise((r) => setImmediate(r)); };   // setTimeout(0) waits a 15 ms tick on Windows
const copy = (x) => JSON.parse(JSON.stringify(x));

function setup(job, answers) {
  const grid = new Node('div'), main = new Node('div'), side = new Node('aside');
  grid.append(main, side);
  const select = (v) => Object.assign(new Node('select'), {value: v});
  const c = {team: select(job.team || 'A'), range: select(job.range || 'whole'), custom: new Node('span'), t0: new Node('input'), t1: new Node('input'),
             setup: new Node('div'), generate: new Node('button'), pdf: new Node('button')};
  if (job.contextSelect) c.context = select(job.context || 'competitive');
  const s = {grid, main, side, c, asked: [], meta: [], posted: [], jumps: [], downloads: [], answers: answers.slice(), pending: [], now: job.now === undefined ? 0 : job.now,
             shown: [], seeks: [], fits: 0, posts: 0};
  // The coaching v2 routes (catalogue, settings) answer from the job, or fail as on a v1 server (legacy mode); report
  // requests are listed in s.asked. job.manual: every report request waits until the scenario answers it (s.pending).
  const meta = (url) => {
    const kind = url.startsWith('/api/coaching/profiles') ? 'profiles' : url.startsWith('/api/coaching/settings') ? 'settings'
      : url.startsWith('/api/coaching/spans') ? 'spans' : url.startsWith('/api/coaching/progress') ? 'progress' : null;
    if (!kind) return null;
    s.meta.push(url);
    // index.html's api() puts the HTTP status on its error (also when the body is an HTML page that does not parse).
    if (job[kind] && job[kind].__error) return Promise.reject(Object.assign(new Error(job[kind].__error), job[kind].__name ? {name: job[kind].__name} : {}, job[kind].__status ? {status: job[kind].__status} : {}));
    return job[kind] ? Promise.resolve(copy(job[kind])) : Promise.reject(Object.assign(new Error('NOT FOUND'), {status: 404}));
  };
  const api = job.manual ? (url) => meta(url) || (s.asked.push(url), new Promise((res) => s.pending.push({url, res})))
    : async (url) => { const m = meta(url); if (m) return m; s.asked.push(url); const a = s.answers.length > 1 ? s.answers.shift() : s.answers[0]; if (a && a.__error) throw new Error(a.__error); return copy(a); };
  const post = async (url, body) => {
    s.posted.push({url, body: copy(body)});
    s.posts++;
    if (job.postError) throw new Error(job.postError);
    if (job.postErrorOnce && job.postErrorOnce[url] && !s['failed' + url]) { s['failed' + url] = true; throw new Error(job.postErrorOnce[url]); }
    if (job.holdPosts === url) await new Promise((res) => (s.postWait = s.postWait || []).push(res));   // released by {do: 'release'}
    if (url === '/api/coaching/spans' && !(job.postAnswers || {})[url]) return {ok: true, spans: copy(body.spans), dropped: 0, stale: ['measurement']};
    const a = (job.postAnswers || {})[url];
    return a ? copy(a) : {ok: true};
  };
  s.banner = new Node('div');
  // The possession marks' elements as index.html gives them: the bar, a toggle host per view row, the lane.
  const marks = job.noMarks ? null : {bar: new Node('div'), toggles: [new Node('div'), new Node('div')], lane: new Node('div'), track: new Node('div'), note: new Node('div'),
                                      host: new Node('section'), now: () => s.now, seek: (t) => s.seeks.push(t), fit: () => { s.fits++; }, show: (v) => s.shown.push(v)};
  if (marks) { marks.lane.append(marks.track, marks.note); marks.bar.className = 'controls cm off'; }
  s.marks = marks;
  s.view = sandbox.Coaching.create({main, side, banner: s.banner, controls: c, api, post, run: () => 'run1', video: {currentTime: 0}, marks,
                                    board: {colours: {A: '#b98cff', B: '#f6b564'}}, jump: (ev) => { s.jumps.push(ev); return true; },
                                    download: async (u) => { s.downloads.push(u); if (job.downloadError) throw new Error(job.downloadError); }, media: {matches: !!job.narrow, addEventListener() {}}});
  return s;
}
const summary = (s) => {
  const sections = find(s.main, (n) => n.dataset.section).map((n) => ({key: n.dataset.section, text: n.textContent}));
  const cards = find(s.main, (n) => n.tagName === 'ARTICLE').map((n) => ({observation: n.dataset.observation, cls: n.className, text: n.textContent}));
  const setupText = s.c.setup ? s.c.setup.textContent : '';
  const figures = find(s.side, (n) => n.tagName === 'FIGURE');
  return {sections, cards, banner: {hidden: s.banner.hidden, cls: s.banner.className, text: s.banner.textContent},
          generate: {text: s.c.generate.textContent, disabled: s.c.generate.disabled}, pdf: {disabled: s.c.pdf.disabled},
          side: {text: s.side.textContent, parent: s.side.parentElement === s.grid ? 'grid' : s.side.parentElement ? 'main' : null, figures: figures.length,
                 images: find(s.side, (n) => n.tagName === 'IMG').map((n) => n.src),
                 overlays: find(s.side, (n) => cls(n).includes('cr-overlay')).length, diagrams: find(s.side, (n) => cls(n).includes('cr-diagram')).length},
          asked: s.asked, posted: s.posted, jumps: s.jumps.map((j) => ({frame: j.frame, seek: j.seek, view: j.view, focus: j.focus, diagram: !!j.diagram, overlay: !!j.overlay})),
          downloads: s.downloads, custom: {hidden: s.c.custom.hidden, t0: s.c.t0.value, t1: s.c.t1.value}, setup: setupText, meta: s.meta};
};
const buttonsIn = (root, text) => find(root, (n) => n.tagName === 'BUTTON' && n.textContent === text);

// ---------- coaching v2 ----------
const svgInfo = (root) => ({
  viewBox: root.attrs ? root.attrs.viewBox || null : null,
  shapes: find(root, (n) => ['CIRCLE', 'ELLIPSE', 'RECT', 'POLYGON', 'POLYLINE'].includes(n.tagName)).map((n) => ({tag: n.tagName, stroke: n.attrs.stroke || null, fill: n.attrs.fill || null,
    dash: n.attrs['stroke-dasharray'] || null, opacity: n.attrs['stroke-opacity'] || null, fillOpacity: n.attrs['fill-opacity'] || null, points: n.attrs.points || null,
    cx: n.attrs.cx ? Number(n.attrs.cx) : null, cy: n.attrs.cy ? Number(n.attrs.cy) : null, x: n.attrs.x ? Number(n.attrs.x) : null})),
  texts: find(root, (n) => n.tagName === 'TEXT').map((n) => ({text: n.textContent, fill: n.attrs.fill, x: Number(n.attrs.x), y: Number(n.attrs.y), anchor: n.attrs['text-anchor']}))});
// A section's text without the source credits (data-credit: the titles of federation documents are quoted, not PitchIQ's
// words) and the list of what PitchIQ does not measure (data-unmeasured: it names passes as not measured).
const textOf = (n) => !(n instanceof Node) ? String(n.textContent || '') : n.dataset.credit || n.dataset.unmeasured ? '' : n.own + n.children.map(textOf).join('');
const v2summary = (s) => {
  const setupSelects = {};
  for (const n of find(s.c.setup, (n) => n.tagName === 'SELECT' && n.dataset.setup)) setupSelects[n.dataset.setup] = {value: n.value, options: n.children.map((o) => o.value), texts: n.children.map((o) => o.textContent)};
  const roles = {};
  for (const n of find(s.c.setup, (n) => n.tagName === 'SELECT' && n.dataset.role)) roles[n.dataset.role] = {value: n.value, options: n.children.map((o) => o.value)};
  const gk = find(s.c.setup, (n) => n.dataset.setup === 'goalkeeper')[0];
  const save = find(s.c.setup, (n) => n.dataset.setup === 'save')[0];
  const sideSvgs = find(s.side, (n) => n.tagName === 'SVG');
  return {
    sections: find(s.main, (n) => n.dataset.section).map((n) => ({key: n.dataset.section, text: n.textContent, plain: textOf(n)})),
    cards: find(s.main, (n) => n.tagName === 'ARTICLE').map((n) => ({finding: n.dataset.finding, verdict: n.dataset.verdict, cls: n.className, text: n.textContent,
                                                                    metrics: find(n, (m) => cls(m).split(' ').includes('cr-metric')).length, selected: cls(n).includes('selected')})),
    moments: find(s.main, (n) => n.tagName === 'LI' && cls(n).includes('cr-km-row')).map((n) => ({evidence: n.dataset.evidence, cls: n.className, text: n.textContent})),
    setup: {hidden: s.c.setup.hidden, text: s.c.setup.textContent, selects: setupSelects, roles, gk: gk ? gk.checked : null, save: save ? {disabled: save.disabled, cls: save.className} : null},
    side: {text: s.side.textContent, images: find(s.side, (n) => n.tagName === 'IMG').map((n) => n.src), svgs: sideSvgs.map((n) => cls(n)),
           diagram: sideSvgs.filter((n) => cls(n).includes('cr-diagram')).map(svgInfo)[0] || null, overlay: sideSvgs.filter((n) => cls(n).includes('cr-overlay')).map(svgInfo)[0] || null,
           buttons: find(s.side, (n) => n.tagName === 'BUTTON').map((n) => n.textContent)},
    banner: {hidden: s.banner.hidden, cls: s.banner.className, text: s.banner.textContent},
    generate: {text: s.c.generate.textContent, disabled: s.c.generate.disabled}, pdf: {disabled: s.c.pdf.disabled},
    asked: s.asked.slice(), meta: s.meta.slice(), posted: copy(s.posted), downloads: s.downloads.slice(),
    jumps: s.jumps.map((j) => ({frame: j.frame, seek: j.seek, time: j.time, view: j.view, focus: j.focus, tier: j.tier, caption: j.caption, window: j.window, follow: j.follow,
                                replay: j.replay, evidence: j.evidence, finding: j.finding, diagram: !!j.diagram, overlay: !!j.overlay, explanation: j.explanation || null})),
    selected: s.view.selected, selectedMoment: s.view.selectedMoment, mode: s.view.mode,
    chips: find(s.main, (n) => n.tagName === 'ARTICLE').map((n) => find(n, (m) => m.dataset.measure).map((m) => m.dataset.measure)),
    marks: s.marks ? {bar: {off: cls(s.marks.bar).split(' ').includes('off'), text: s.marks.bar.textContent,
                            buttons: find(s.marks.bar, (n) => n.tagName === 'BUTTON').map((n) => ({text: n.textContent, mark: n.dataset.mark || n.dataset.marks || null, disabled: n.disabled, on: cls(n).includes('on')})),
                            rows: find(s.marks.bar, (n) => n.dataset.span !== undefined).map((n) => n.textContent + '|' + find(n, (m) => m.tagName === 'INPUT').map((m) => m.value).join('|'))},
                      toggles: s.marks.toggles.map((t) => ({hidden: t.hidden, text: t.textContent})),
                      lane: {hidden: s.marks.lane.hidden, note: s.marks.note.textContent, label: s.marks.track.attrs['aria-label'] || null,
                             spans: find(s.marks.track, (n) => n.tagName === 'I').map((n) => ({cls: n.className, left: n.style.left, width: n.style.width, background: n.style.background || null, state: n.dataset.state || null, title: n.title || null})),
                             now: (find(s.marks.track, (n) => n.tagName === 'B')[0] || {style: {}}).style.left || null},
                      host: cls(s.marks.host), shown: s.shown.slice(), seeks: s.seeks.slice(), fits: s.fits,
                      spans: copy(s.view.marks ? s.view.marks.spans : [])} : null,
    posts: s.posts};
};
async function act(s, step) {
  // One user action of a v2 scenario: click a button found by its dataset or text, choose a value, type, call a method.
  const roots = {main: s.main, side: s.side, setup: s.c.setup, bar: s.marks && s.marks.bar, toggle: s.marks && s.marks.toggles[0], lane: s.marks && s.marks.track};
  let root = roots[step.in || 'main'];
  if (step.within) { root = find(root, (n) => Object.entries(step.within).every(([k, v]) => n.dataset[k] === v))[0]; if (!root) throw new Error('no ' + JSON.stringify(step.within)); }
  const match = (n) => (step.data ? Object.entries(step.data).every(([k, v]) => n.dataset[k] === v) : true) && (step.text === undefined || n.textContent === step.text)
    && (step.tag === undefined || n.tagName === step.tag);
  if (step.do === 'click') { const n = find(root, (n) => (n.tagName === 'BUTTON' || n.tagName === 'LI') && match(n))[step.index || 0]; if (!n) throw new Error('no button ' + JSON.stringify(step)); n.click(); }
  else if (step.do === 'choose') { const n = find(root, (n) => (n.tagName === 'SELECT' || n.tagName === 'INPUT' || n.tagName === 'TEXTAREA') && match(n))[0]; if (!n) throw new Error('no field ' + JSON.stringify(step));
    if ('checked' in step) n.checked = step.checked; else n.value = step.value; n.fire(step.event || 'change'); }
  else if (step.do === 'team') { s.c.team.value = step.value; s.c.team.fire('change'); }
  else if (step.do === 'method') s.view[step.name](...(step.args || []));
  else if (step.do === 'time') s.now = step.value;                                     // the video's clock (marks: "from here")
  else if (step.do === 'tick') s.view.tick(step.value);
  else if (step.do === 'lane') s.marks.track.fire('click');                            // a click without a position: nothing happens
  else if (step.do === 'laneAt') { const e = {type: 'click', clientX: step.x, target: s.marks.track}; for (const f of s.marks.track.listeners.click || []) f(e); }
  else if (step.do === 'active') s.view.setActive(step.value);
  else if (step.do === 'release') { const w = s.postWait || []; s.postWait = []; w.forEach((f) => f()); }
}
async function v2scenario(job) {
  sandbox.hold = !!job.hold; sandbox.held = [];   // job.hold: polls (setTimeout) wait, as a busy server would make them
  if (job.result === 'shared') job.result = input.result;
  const s = setup(job, job.answers);
  s.view.load(job.result, job.info);
  s.view.setActive(true);
  await settle();
  const out = {first: v2summary(s)};
  for (const step of job.steps || []) { await act(s, step); await settle(); if (step.name) out[step.name] = v2summary(s); }
  out.last = v2summary(s);
  sandbox.hold = false; sandbox.held = [];
  return out;
}
async function paint2(job) {
  // The v2 element drawing: SVG diagram and overlay (attributes), the canvas layers (calls), the follow layer.
  const labels = new Map(Object.entries(job.labels || {})), colours = job.colours || {A: '#b98cff', B: '#f6b564'};
  const out = {};
  if (job.diagram) out.diagram = svgInfo(sandbox.Coaching.svgDiagram(job.diagram, colours, labels, job.opts || {}));
  if (job.overlay) out.overlay = svgInfo(sandbox.Coaching.svgOverlay(job.overlay, colours, labels));
  if (job.overlay) { const log = []; sandbox.Coaching.drawVideo(recorder(log), job.overlay, 960, 540, colours, labels); out.video = log; }
  if (job.diagram && job.state) {
    const board = sandbox.Tactical.create({canvas: new Node('canvas'), controls: {}}), log = [];
    sandbox.Coaching.drawBoard(recorder(log), board.geometry(), job.diagram, colours, {players: job.state}, labels); out.board = log;
  }
  if (job.follow) {
    const log = [];
    sandbox.Coaching.drawFollow(recorder(log), job.follow, {mode: 'video', people: new Map(Object.entries(job.people || {})), width: 1000, height: 500, tier: job.tier || 'world', colours, labels});
    out.followVideo = log;
    if (job.state) {
      const board = sandbox.Tactical.create({canvas: new Node('canvas'), controls: {}}), blog = [];
      sandbox.Coaching.drawFollow(recorder(blog), job.follow, {mode: 'board', g: board.geometry(), state: {players: job.state}, colours, labels}); out.followBoard = blog;
    }
    if (job.ev) out.followOf = sandbox.Coaching.followOf(job.ev);
  }
  return out;
}
async function palette(job) {
  const C = sandbox.Coaching;
  return {sharma: job.sharma.map(([a, b]) => C.ciede2000(a, b)), palettes: job.pairs.map(([a, b]) => C.annotationPalette(a, b)),
          dE: job.dE.map(([a, b]) => C.ciede2000(C.labOf(a), C.labOf(b))), contrast: job.contrast.map(([a, b]) => C.contrastRatio(a, b)),
          clock: job.clock.map(C.clock), catalogue: C.CATALOGUE, tiers: C.TIER_ELEMENTS, budget: C.ELEMENT_BUDGET,
          defaults: job.defaults.map(([age, fed]) => C.formatDefault(age, fed)), typical: job.defaults.map(([age]) => C.typicalLevel(age)),
          slots: job.slots.map(([f, formation]) => C.relativeSlots(f, formation))};
}

async function scenario(job) {
  if (job.result === 'shared') job.result = input.result;   // the synthetic result is sent once for every job
  sandbox.hold = !!job.hold; sandbox.held = [];
  const s = setup(job, job.answers);
  s.view.load(job.result, job.info);
  s.view.setActive(true);
  await settle();
  const out = {first: summary(s)};
  const area = () => find(s.main, (n) => n.tagName === 'TEXTAREA')[0], image = () => find(s.side, (n) => n.tagName === 'IMG')[0];
  if (job.kind === 'race') {
    s.pending.shift().res(copy(job.answers[0])); await settle();                     // team A shown
    s.c.team.value = 'B'; s.c.team.fire('change'); await settle();                   // B asked, still in flight
    s.c.team.value = 'A'; s.c.team.fire('change'); await settle();                   // back to A: from the cache
    out.during = {team: s.view.answer.report.team, loading: s.view.loading, generate: s.c.generate.disabled, pdf: s.c.pdf.disabled};
    s.pending.shift().res(copy(job.answers[1])); await settle();                     // B's late answer
    out.after = summary(s);
    out.after.team = s.view.answer.report.team; out.after.loading = s.view.loading;
    s.c.pdf.fire('click'); await settle();
    out.downloads = s.downloads;
  } else if (job.kind === 'resize') {
    const a0 = area(), i0 = image(), asked0 = s.asked.length;
    for (let k = 0; k < 5; k++) { s.view.setActive(true); await settle(); }          // what five resizes used to do
    out.resize = {sameTextarea: area() === a0, sameImage: image() === i0, asked: s.asked.length - asked0};
    s.view.setActive(false); s.view.setActive(true); await settle();                 // shown again: asked again, same answer
    out.reshown = {sameTextarea: area() === a0, sameImage: image() === i0, asked: s.asked.length - asked0};
  } else if (job.kind === 'busy') {
    out.held = sandbox.held.length;                                                  // the poll waiting for the other build
    sandbox.hold = false;
    for (const f of sandbox.held.splice(0)) f();
    await settle();
    out.after = summary(s);
  } else if (job.kind === 'rounding' || job.kind === 'clamp') {
    s.c.range.value = 'custom'; s.c.range.fire('change'); await settle();
    s.c.t0.value = job.t0; s.c.t1.value = job.t1; s.c.t0.fire('change'); await settle();
    if (job.kind === 'rounding') { s.c.generate.fire('click'); await settle(); await settle(); }
    out.after = summary(s);
    out.after.customTitle = s.c.custom.title || '';
  } else if (job.kind === 'jump') {
    const card = find(s.main, (n) => n.dataset.observation === job.observation)[0];
    find(card, (n) => cls(n).includes('cr-time'))[0].click();
    buttonsIn(s.side, 'Show on 2D')[0].click();
    out.after = summary(s);
  } else if (job.kind === 'feedback') {
    const box = find(s.main, (n) => n.dataset.feedback === job.observation)[0];
    const saveBefore = buttonsIn(box, 'Save review')[0].disabled;
    find(box, (n) => n.dataset.verdict === 'partial')[0].click();
    const area = find(box, (n) => n.tagName === 'TEXTAREA')[0];
    area.value = 'The gap was a pressing trigger'; area.fire('input');
    buttonsIn(box, 'Save review')[0].click();
    await settle();
    const after = find(s.main, (n) => n.dataset.feedback === job.observation)[0];
    out.saveBefore = saveBefore;
    out.afterSave = {text: after.textContent, update: buttonsIn(after, 'Update review').length, pressed: find(after, (n) => n.attrs['aria-pressed'] === 'true').map((n) => n.dataset.verdict)};
    buttonsIn(after, 'Remove')[0].click();
    await settle();
    out.after = summary(s);
  } else if (job.kind === 'generate') {
    s.c.generate.fire('click');
    await settle(); await settle();
    out.after = summary(s);
  } else if (job.kind === 'pdf') {
    s.c.pdf.fire('click');
    await settle();
    out.after = summary(s);
  } else if (job.kind === 'custom') {
    s.c.range.value = 'custom'; s.c.range.fire('change');
    await settle();
    out.after = summary(s);
  } else if (job.kind === 'reactivate') {
    s.view.setActive(false); s.view.setActive(true);
    await settle();
    out.after = summary(s);
  } else if (job.kind === 'badrange') {
    s.c.range.value = 'custom'; s.c.range.fire('change');
    await settle();
    s.c.t0.value = 'soon'; s.c.t0.fire('change');
    await settle();
    s.c.generate.fire('click');
    await settle();
    out.after = summary(s);
  } else if (job.kind === 'reload') {
    s.view.load(job.result, job.info);
    await settle();
    out.reloaded = summary(s);
    s.view.load(null, job.info);
    await settle();
    out.after = summary(s);
  } else if (job.kind === 'select') {
    find(s.main, (n) => n.dataset.observation === job.observation)[0];
    const card = find(s.main, (n) => n.dataset.observation === job.observation)[0];
    buttonsIn(card, 'Evidence (1)')[0].click();
    const parent = s.side.parentElement;
    out.after = summary(s);
    out.placement = {sameParent: parent === card.parentElement, nextToCard: parent ? parent.children.indexOf(s.side) === parent.children.indexOf(card) + 1 : false};
  }
  sandbox.hold = false; sandbox.held = [];
  return out;
}

async function drawing(job) {
  const labels = new Map(Object.entries(job.labels)), colours = {A: '#b98cff', B: '#f6b564'};
  const board = sandbox.Tactical.create({canvas: new Node('canvas'), controls: {}});
  const g = board.geometry(), boardLog = [], videoLog = [];
  sandbox.Coaching.drawBoard(recorder(boardLog), g, job.diagram, colours, {players: job.state}, labels);
  sandbox.Coaching.drawVideo(recorder(videoLog), job.overlay, 1280, 720, colours, labels);
  const svg = sandbox.Coaching.svgDiagram(job.diagram, colours, labels), over = sandbox.Coaching.svgOverlay(job.overlay, colours, labels);
  const texts = (root) => find(root, (n) => n.tagName === 'TEXT').map((n) => n.textContent);
  return {board: boardLog, video: videoLog, svg: {circles: find(svg, (n) => n.tagName === 'CIRCLE').length, texts: texts(svg), viewBox: svg.attrs.viewBox,
                                                   polylines: find(svg, (n) => n.tagName === 'POLYLINE').length, polygons: find(svg, (n) => n.tagName === 'POLYGON').length},
          overlay: {viewBox: over.attrs.viewBox, aspect: over.attrs.preserveAspectRatio, texts: texts(over), ellipses: find(over, (n) => n.tagName === 'ELLIPSE').length},
          plain: sandbox.Coaching.plainLabels('A-03 passed B-07 and A-77; GK-1 stayed', labels),
          corrected: job.corrected ? Object.fromEntries(sandbox.Coaching.labelMap(job.corrected)) : null};
}

async function brackets(job) {
  // The SVG diagram's lines (points, dash) and texts (position, anchor): where a bracket and its value stand.
  const svg = sandbox.Coaching.svgDiagram(job.diagram, {A: '#b98cff', B: '#f6b564'}, new Map());
  const pts = (n) => String(n.attrs.points || '').split(' ').filter(Boolean).map((p) => p.split(',').map(Number));
  return {lines: find(svg, (n) => n.tagName === 'POLYLINE').map((n) => ({points: pts(n), dash: n.attrs['stroke-dasharray'] || null})),
          texts: find(svg, (n) => n.tagName === 'TEXT').map((n) => ({text: n.textContent, x: Number(n.attrs.x), y: Number(n.attrs.y), anchor: n.attrs['text-anchor']}))};
}

(async () => {
  const out = [];
  const kinds = {drawing, brackets, v2: v2scenario, paint2, palette};
  for (const job of input.jobs) out.push(await (kinds[job.kind] || scenario)(job));
  process.stdout.write(JSON.stringify(out));
})().catch((error) => { process.stderr.write(String(error && error.stack || error)); process.exit(1); });
"""


def run(*jobs):
    done = subprocess.run([NODE, '-e', HARNESS], input=json.dumps({'files': FILES, 'jobs': list(jobs), 'result': RESULT}), capture_output=True, text=True,
                          encoding='utf-8', timeout=300)
    if done.returncode:
        raise AssertionError(done.stderr[-3000:])
    return json.loads(done.stdout)


RESULT = synthetic_result(seconds=12)
INFO = {'video': RESULT['video'], 'videoKey': '0'*64}
REPORT = fixture_report(RESULT)
FP = 'f'*64
GAP = REPORT['improvements'][0]['observation']
REVIEW = REPORT['reviews'][0]['observation']
# The page's sections in the PDF's print order (soccer/coaching/pdf.py section_order(); the statistics and the evidence
# index are the PDF's alone, and the page lists the feedback that sits on no finding where the PDF prints its own).
PAGE_SECTIONS = ('overview', 'summary', 'strengths', 'improvements', 'reviews', 'breakdown', 'priorities', 'training', 'confidence', 'feedback')
PAGE_ORDER = [k for k in pdf.section_order() if k in PAGE_SECTIONS]


def corrected_result():
    """RESULT after "A-03 is not part of the match" (as corrections.apply_result leaves it): the entry moves to
    match.excluded with the display it had (A-02 here: the fixture swaps A-02 and A-03), and the labels renumber."""
    result = json.loads(json.dumps(RESULT))
    match = result['match']
    gone = next(e for e in match['players'] if e['id'] == 'A-03')
    match['players'].remove(gone)
    match['excluded'] = [dict(gone, excluded=True, status='EXCLUDED')]
    for e in match['players']:
        if e['team'] == 'A' and e['id'] >= 'A-04':
            e['display'] = f"A-{int(e['id'][2:])-1:02d}"
    match['players'][[e['id'] for e in match['players']].index('A-02')]['display'] = 'A-02'
    result['roster']['slots'] = [s for s in result['roster']['slots'] if s['id'] != 'A-03']
    return result


def answer(state='ready', report=REPORT, **kw):
    return dict({'state': state, 'fingerprint': FP if report else None, 'current': FP, 'reasons': [], 'reasonCodes': [], 'report': report, 'feedback': [],
                 'saved': '2026-10-18T19:00:00', 'options': {'team': 'A', 't0': None, 't1': None, 'context': 'competitive'}, 'run': 'run1',
                 'error': None, 'busy': False}, **kw)


def job(kind='render', answers=None, **kw):
    return dict({'kind': kind, 'answers': answers or [answer()], 'result': 'shared', 'info': INFO}, **kw)


def section(out, key):
    return next(s['text'] for s in out['sections'] if s['key'] == key)


@unittest.skipUnless(NODE, 'needs node to run the page scripts')
class ReportView(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        entry = {'id': 'f20261018-190312-abcdef', 'run': 'run1', 'made': '2026-10-18T19:03:12', 'report': 'f'*16, 'verdict': 'partial',
                 'text': 'The gap was a pressing trigger', 'thisRun': True, 'attached': GAP,
                 'observation': {'key': GAP, 'detector': 'unit_gap_dm', 'team': 'A', 't0': 1.0, 't1': 13.0, 'players': [], 'headline': 'x'}}
        loose = dict(entry, id='f20261018-190313-abcdef', attached=None, thisRun=False, run='run0', verdict='incorrect',
                     observation=dict(entry['observation'], headline='An old finding'))
        stale = answer('stale', reasons=['1 correction made since this report: the tracking result changed.', 'The report engine was updated.'],
                       reasonCodes=['result', 'engine'])
        # Verdicts that sit on no finding: one from an earlier run (its text names that run's ids), one of this run on a
        # finding this report does not have, and one of this run on team B's report (not listed under team A's).
        earlier = dict(loose, id='f20261018-190314-abcdef', observation=dict(entry['observation'], team='B', headline='Gap behind A-03'),
                       text='Wrong side')
        gone = dict(loose, id='f20261018-190315-abcdef', thisRun=True, run='run1', verdict='correct', observation=dict(entry['observation'], headline='Old width finding'))
        other_team = dict(gone, id='f20261018-190316-abcdef', observation=dict(entry['observation'], team='B', headline='Team B finding'))
        team_b = answer(report=fixture_report(RESULT, 'B'), options={'team': 'B', 't0': None, 't1': None, 'context': 'competitive'})
        clamped = answer(options={'team': 'A', 't0': 1.0, 't1': 8.0, 'context': 'competitive'})
        cls.out = run(
            job(), job('jump', observation=GAP), job('feedback', observation=GAP, answers=[answer(), answer(feedback=[entry]), answer()]),
            job(answers=[answer('missing', report=None)]), job(answers=[answer('building', report=None), answer()]), job(answers=[stale]),
            job(answers=[answer('error', report=None, error='The coaching report engine is not available in this version yet.')]),
            job(answers=[answer(report=fixture_report(RESULT, empty=True))]),
            job('generate', answers=[answer('missing', report=None), answer('building', report=None), answer()]),
            job('pdf', downloadError='PDF export is not available in this version yet.'),
            job('custom'), job('reload', answers=[answer(feedback=[loose])]), job('select', observation=REVIEW, narrow=True),
            job(answers=[{'__error': 'There is no result to report on yet.'}]),
            job('busy', hold=True, answers=[answer(busy=True, state='missing', report=None), answer('missing', report=None)]),
            job('reactivate'), job('badrange'),
            job('race', manual=True, answers=[answer(), team_b]), job('resize'),
            job('rounding', t0='00:02.125', t1='00:10.125', answers=[answer('missing', report=None)]),
            job('clamp', t0='00:00.00', t1='00:08.00', answers=[clamped]),
            job(answers=[answer(feedback=[dict(entry), earlier, gone, other_team])]),
            job(result=corrected_result()))
        (cls.ready, cls.jump, cls.feedback, cls.missing, cls.building, cls.stale, cls.error, cls.empty, cls.generate, cls.pdf, cls.custom,
         cls.reload, cls.narrow, cls.failed, cls.busy, cls.reactivate, cls.badrange, cls.race, cls.resize, cls.rounding, cls.clamp,
         cls.apart, cls.corrected) = cls.out

    def test_every_section_is_rendered(self):
        out = self.ready['first']
        self.assertEqual([s['key'] for s in out['sections']],
                         ['overview', 'summary', 'strengths', 'improvements', 'reviews', 'breakdown', 'priorities', 'training', 'confidence'])
        self.assertEqual([s['key'] for s in out['sections']], [k for k in PAGE_ORDER if k != 'feedback'], "the PDF's order")
        self.assertEqual(PAGE_ORDER, list(PAGE_SECTIONS), 'every page section is in the PDF, in the same order')
        apart = [s['key'] for s in self.apart['first']['sections']]
        self.assertEqual(apart, PAGE_ORDER, 'with reviews that sit on no finding: the Expert feedback section last, as in the PDF')
        self.assertIn('SEGMENT COACHING REPORT', section(out, 'overview'))
        self.assertIn('Team A (as labelled in this run)', section(out, 'overview'))
        self.assertIn('changes the wording, never the measurements', section(out, 'overview'))
        self.assertIn('Segment report: fewer than 20 minutes measured', section(out, 'confidence'))
        self.assertIn('Observationstotal 4 \u00b7 shown 3', section(out, 'overview'), 'a group of counts is one row')
        self.assertIn('Measured secondsspacing 11.9 s \u00b7 movement 4.0 s', section(out, 'overview'))
        self.assertIn('ContextThe ball was in the middle third', section(out, 'overview'), 'block height and ball: the context')
        self.assertIn('Strengths \u00b7 1', section(out, 'strengths'))
        self.assertIn('Areas to improve \u00b7 1', section(out, 'improvements'))
        training = section(out, 'training')
        for text in ('Compact lines', 'Purpose. Keep the distance', 'Players. 8\u201312 players. Two lines of four.', 'Duration. 15\u201320 min. Four blocks.',
                     'Coaching pointsMove together', 'How it runsShift with the ball', 'SafetyWarm up first', 'At the competitive level. Play 7 v 7'):
            self.assertIn(text, training)
        self.assertIn('Close the space between the lines', section(out, 'priorities'))
        self.assertIn('No possession record', section(out, 'breakdown'), 'an unavailable section says why')
        self.assertIn('Lateral shift was measured for only 4.0 s', section(out, 'confidence'), 'measured too briefly: a limitation, not a weakness')
        self.assertNotIn('Lateral shift', section(out, 'improvements'))
        self.assertEqual([c['observation'] for c in out['cards']], [r['observation'] for r in REPORT['strengths']+REPORT['improvements']+REPORT['reviews']])
        gap = next(c for c in out['cards'] if c['observation'] == GAP)
        self.assertIn('k-improvement', gap['cls'])
        self.assertIn('Potential improvement', gap['text'])
        self.assertIn('What to improve.', gap['text'])
        self.assertIn('On the video. Watch the space behind A-02 when the ball is wide.', gap['text'])
        self.assertIn('Confidence. Registration medium on part of the measured time.', gap['text'])
        self.assertIn('Drill. Compact lines', gap['text'])
        self.assertIn('Defence to midfield gap 21.4 m (est.)', gap['text'], 'metres are marked as estimated')
        self.assertIn('Your review', gap['text'])
        self.assertEqual((out['banner']['hidden'], out['generate'], out['pdf']['disabled']), (True, {'text': 'Report up to date', 'disabled': True}, False))
        self.assertTrue(out['asked'][0].startswith('/api/coaching?run=run1&team=A&context=competitive'), out['asked'])

    def test_players_are_shown_by_their_current_display_label(self):
        gap = next(c for c in self.ready['first']['cards'] if c['observation'] == GAP)['text']
        # The report names A-03 (shown as A-02 in this result) and B-07 (shown as B-05).
        self.assertIn('behind A-02', gap)
        self.assertIn('B-05 were nearest', gap)
        self.assertNotIn('B-07', gap)
        side = self.ready['first']['side']['text']
        self.assertIn('A-02, A-03', side, 'the evidence lists the players by display label')

    def test_evidence_of_the_selected_finding(self):
        side = self.ready['first']['side']
        self.assertEqual(side['parent'], 'grid')
        self.assertEqual(side['figures'], 2, 'the first improvement is selected: its two evidence frames')
        frames = [e['frame'] for e in REPORT['evidence'] if e['observation'] == GAP]
        self.assertEqual(side['images'], [f'/api/coaching/frame?run=run1&frame={f}&w=640' for f in frames])
        self.assertEqual((side['overlays'], side['diagrams']), (2, 2))
        self.assertIn("Shapes drawn from the players\u2019 feet on this frame", side['text'])
        self.assertIn('distances estimated (assumed 105 \u00d7 68 m pitch)', side['text'])
        self.assertIn('Registration confidence0.82', side['text'])

    def test_an_evidence_timestamp_jumps_to_the_frame_centre(self):
        ev = [e for e in REPORT['evidence'] if e['observation'] == GAP][0]
        jumps = self.jump['after']['jumps']
        self.assertEqual(jumps[0], {'frame': ev['frame'], 'seek': ev['seek'], 'view': 'tracking', 'focus': ev['players'], 'diagram': True, 'overlay': True})
        self.assertAlmostEqual(jumps[0]['seek'], (ev['frame']+.5)/30, places=5)
        self.assertEqual(jumps[1]['view'], 'board')
        self.assertEqual(jumps[1]['frame'], ev['frame'])

    def test_feedback_is_posted_and_shown(self):
        out = self.feedback
        self.assertTrue(out['saveBefore'], 'nothing to save before a verdict is chosen')
        post = out['after']['posted'][0]
        self.assertEqual(post['url'], '/api/coaching/feedback')
        self.assertEqual(post['body'], {'run': 'run1', 'observation': GAP, 'verdict': 'partial', 'text': 'The gap was a pressing trigger', 'fingerprint': FP, 'team': 'A'})
        self.assertIn('Saved 2026-10-18 19:03:12: Partially correct', out['afterSave']['text'])
        self.assertEqual((out['afterSave']['update'], out['afterSave']['pressed']), (1, ['partial']))
        self.assertEqual(out['after']['posted'][1], {'url': '/api/coaching/feedback/remove', 'body': {'id': 'f20261018-190312-abcdef'}})
        self.assertGreaterEqual(len(out['after']['asked']), 3, 'the report is asked again after each save')

    def test_no_report_yet(self):
        out = self.missing['first']
        self.assertEqual([s['key'] for s in out['sections']], ['empty'])
        self.assertIn('No coaching report yet for Team A (whole analysed segment, competitive level)', out['sections'][0]['text'])
        self.assertIn('Only what the tracking measured is reported', out['sections'][0]['text'])
        self.assertEqual((out['generate'], out['pdf']['disabled']), ({'text': 'Generate report', 'disabled': False}, True))
        self.assertIn('appears here', out['side']['text'])

    def test_building_polls_until_ready(self):
        out = self.building['first']
        self.assertEqual(out['generate'], {'text': 'Report up to date', 'disabled': True}, 'the poll found the finished report')
        self.assertGreaterEqual(len(out['asked']), 2)
        self.assertEqual(out['sections'][0]['key'], 'overview')

    def test_a_stale_report_says_why_and_offers_to_regenerate(self):
        out = self.stale['first']
        self.assertFalse(out['banner']['hidden'])
        self.assertIn('stale', out['banner']['cls'])
        self.assertIn('This report is out of date.', out['banner']['text'])
        self.assertIn('1 correction made since this report', out['banner']['text'])
        self.assertIn('The report engine was updated.', out['banner']['text'])
        self.assertEqual(out['generate'], {'text': 'Regenerate', 'disabled': False})
        self.assertEqual(out['sections'][0]['key'], 'overview', 'the stale report stays readable')
        self.assertFalse(out['pdf']['disabled'])

    def test_errors_and_nothing_to_report(self):
        out = self.error['first']
        self.assertIn('error', out['banner']['cls'])
        self.assertIn('engine is not available', out['banner']['text'])
        self.assertEqual(out['generate']['text'], 'Generate report')
        empty = self.empty['first']
        self.assertIn('No strength met the evidence bar', section(empty, 'strengths'))
        self.assertIn('Nothing went beyond the professional level more often than professional teams do over the same measured time',
                      section(empty, 'improvements'))
        self.assertNotIn('upper level', section(empty, 'strengths')+section(empty, 'improvements'), 'the rules as the detectors apply them')
        self.assertNotIn('repeated', section(empty, 'improvements'))
        self.assertIn('Nothing in this range met the evidence bar', section(empty, 'summary'))
        self.assertIn('No priority', section(empty, 'priorities'))
        self.assertIn('Not enough measured time.', section(empty, 'breakdown'))
        self.assertEqual(empty['cards'], [])
        self.assertIn('Select a finding', empty['side']['text'])
        failed = self.failed['first']
        self.assertIn('There is no result to report on yet.', failed['banner']['text'])
        busy = self.busy['first']
        self.assertIn('Another coaching report is being built', busy['banner']['text'])
        self.assertTrue(busy['generate']['disabled'])

    def test_busy_is_asked_again_until_the_other_build_ends(self):
        # A 'busy' answer was cached: after the other build ended the banner stayed and Generate stayed disabled.
        self.assertEqual(self.busy['held'], 1, 'a poll is waiting')
        after = self.busy['after']
        self.assertEqual(len(after['asked']), 2)
        self.assertEqual(after['asked'][0], after['asked'][1], 'the same selection')
        self.assertTrue(after['banner']['hidden'])
        self.assertEqual(after['generate'], {'text': 'Generate report', 'disabled': False})

    def test_generate_builds_then_follows_the_build(self):
        out = self.generate['after']
        self.assertEqual(out['posted'], [{'url': '/api/coaching/build', 'body': {'run': 'run1', 'team': 'A', 'context': 'competitive'}}])
        self.assertEqual(len(out['asked']), 3, 'missing, building, ready')
        self.assertEqual(out['generate']['text'], 'Report up to date')

    def test_export_pdf_downloads_the_selection(self):
        out = self.pdf['after']
        self.assertEqual(out['downloads'], ['/api/coaching/pdf?run=run1&team=A&context=competitive'])
        self.assertIn('The PDF could not be made: PDF export is not available in this version yet.', out['banner']['text'], 'a refusal is said, not downloaded')
        self.assertFalse(out['pdf']['disabled'], 'and the button is usable again')

    def test_custom_range(self):
        out = self.custom['after']
        self.assertEqual(out['custom'], {'hidden': False, 't0': '00:01.00', 't1': '00:12.97'})
        self.assertIn('&t0=1.00&t1=12.97', out['asked'][-1])

    def test_shown_again_asks_again(self):
        # A team-shape save on the Analytics tab reloads no result, so the report's staleness is asked again.
        self.assertEqual(len(self.reactivate['after']['asked']), 2)

    def test_an_unreadable_custom_range_is_said(self):
        out = self.badrange['after']
        self.assertIn('Enter the custom range as two times (mm:ss.ss, source-video time)', out['banner']['text'])
        self.assertEqual(out['posted'], [], 'nothing is built for it')
        self.assertEqual(len(out['asked']), 2, 'the whole segment, then the filled-in custom range; not the unreadable one')

    def test_load_clears_the_cache(self):
        self.assertEqual(len(self.reload['reloaded']['asked']), 2, 'a reloaded result asks again')
        self.assertIn('An old finding \u00b7 Incorrect \u00b7 \u201cThe gap was a pressing trigger\u201d (run run0, Team A as labelled in that run',
                      section(self.reload['first'], 'feedback'), 'a review from an earlier run is listed apart')
        self.assertIn('Coaching reports appear here after a run.', self.reload['after']['sections'][0]['text'])

    def test_reviews_on_no_finding_are_listed_apart(self):
        out = self.apart['first']
        fb = section(out, 'feedback')
        self.assertIn('From an earlier run', fb)
        self.assertIn('Gap behind A-03 \u00b7 Incorrect \u00b7 \u201cWrong side\u201d (run run0, Team B as labelled in that run', fb,
                      "an earlier run's text keeps that run's ids (A-03 is shown as A-02 in this run)")
        self.assertIn('Your reviews of this run that match no finding here', fb)
        self.assertIn('Old width finding \u00b7 Correct', fb)
        self.assertNotIn('Team B finding', fb, "this run's team B reviews belong to team B's report")
        gap = next(c for c in out['cards'] if c['observation'] == GAP)['text']
        self.assertIn('Saved 2026-10-18 19:03:12: Partially correct', gap)
        self.assertNotIn('earlier run', gap, 'no other run on the finding')
        self.assertNotIn('Wrong side', gap)
        self.assertNotIn('feedback', [s['key'] for s in self.ready['first']['sections']], 'no section when every review sits on its finding')

    def test_a_late_answer_never_replaces_the_selection(self):
        # Team B asked, then A shown from the cache: B's late answer used to replace A's report while the selector said A.
        self.assertEqual(self.race['during'], {'team': 'A', 'loading': False, 'generate': True, 'pdf': False})
        after = self.race['after']
        self.assertEqual((after['team'], after['loading']), ('A', False))
        self.assertIn('Team A (as labelled in this run)', section(after, 'overview'))
        self.assertEqual(self.race['downloads'], ['/api/coaching/pdf?run=run1&team=A&context=competitive'], "the PDF of the report on screen")

    def test_nothing_is_rebuilt_while_the_view_stays_shown(self):
        # Every resize used to call setActive(true): the note being typed and the screenshots were replaced 5 times.
        self.assertEqual(self.resize['resize'], {'sameTextarea': True, 'sameImage': True, 'asked': 0})
        self.assertEqual(self.resize['reshown'], {'sameTextarea': True, 'sameImage': True, 'asked': 1},
                         'shown again: asked again, and the same answer keeps the drawn report')

    def test_status_and_build_ask_for_the_same_hundredths(self):
        out = self.rounding['after']
        self.assertIn('&t0=2.13&t1=10.13', out['asked'][-1])
        build = out['posted'][0]['body']
        self.assertEqual((build['t0'], build['t1']), (2.13, 10.13))
        # Both reach the store as the same option, so the built report is the one the status request finds.
        query = dict(p.split('=') for p in out['asked'][-1].split('?')[1].split('&'))
        bounds = (RESULT['frames'][0]['time'], RESULT['frames'][-1]['time'])
        self.assertEqual(store.clean_options('A', query['t0'], query['t1'], None, bounds), store.clean_options('A', build['t0'], build['t1'], None, bounds))
        self.assertNotEqual(round(10.125, 2), 10.13, 'what rounding twice did')

    def test_the_custom_range_is_source_video_time_and_a_clamp_is_said(self):
        out = self.clamp['after']
        self.assertIn('Source-video time', out['customTitle'])
        self.assertIn('The analysed segment runs 00:01.00\u201300:12.97', out['customTitle'])
        self.assertFalse(out['banner']['hidden'])
        self.assertIn('The range 00:00.00\u201300:08.00 reaches outside the analysed segment (00:01.00\u201300:12.97, source-video time), '
                      'so the report covers 00:01.00\u201300:08.00.', out['banner']['text'])
        self.assertIn('covers the whole analysed segment', self.custom['after']['banner']['text'], 'a range over the whole segment says so')
        self.assertIn('(mm:ss.ss, source-video time)', self.badrange['after']['banner']['text'])

    def test_an_excluded_player_is_never_shown_with_another_players_label(self):
        gap = next(c for c in self.corrected['first']['cards'] if c['observation'] == GAP)['text']
        self.assertIn('behind excluded (was A-02)', gap, 'A-03 was excluded; its label A-02 belongs to A-02 now')
        self.assertIn('Watch the space behind excluded (was A-02) when the ball is wide.', gap)
        side = self.corrected['first']['side']['text']
        self.assertIn('Playersexcluded (was A-02), A-02', side)

    def test_narrow_screens_put_the_evidence_under_its_finding(self):
        self.assertEqual(self.narrow['placement'], {'sameParent': True, 'nextToCard': True})
        self.assertEqual(self.narrow['after']['side']['figures'], 1)


@unittest.skipUnless(NODE, 'needs node to run the page scripts')
class Drawing(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ev = REPORT['evidence'][0]
        frame = next(f for f in RESULT['frames'] if f['frame'] == ev['frame'])
        state = [{'id': p['id'], 'x': p['pitch'][0]+.001, 'y': p['pitch'][1], 'team': p['team']} for p in frame['people']]
        labels = {e['id']: e['display'] for e in RESULT['match']['players']}
        cls.ev, cls.out = ev, run({'kind': 'drawing', 'diagram': ev['diagram'], 'overlay': ev['videoOverlay'], 'state': state, 'labels': labels})[0]

    def test_board_layer(self):
        log = self.out['board']
        self.assertTrue(any(c[0] == 'fillText' and c[1] == '21.4 m (est.)' for c in log), 'the measure is labelled')
        self.assertFalse(any(c[0] == 'fillText' and c[1] in ('A-02', 'A-03') for c in log), 'the board labels its own markers')
        rings = [c for c in log if c[0] == 'arc']
        self.assertGreaterEqual(len(rings), 2, 'the players involved are ringed')
        self.assertTrue(any(c[0] == 'fillText' and c[1] == 'COACHING ILLUSTRATION' for c in log))

    def test_video_layer_and_svgs(self):
        self.assertTrue(any(c[0] == 'fillText' and c[1] == 'A-02' for c in self.out['video']), 'the involved player A-03 is labelled A-02')
        self.assertTrue(any(c[0] == 'ellipse' for c in self.out['video']))
        svg = self.out['svg']
        players = sum(1 for e in self.ev['diagram']['elements'] if e['type'] == 'player')
        self.assertGreaterEqual(svg['circles'], players)
        self.assertEqual(svg['viewBox'], '0 0 114 81')
        self.assertIn('A-02', svg['texts'])
        self.assertIn('Team A attacks \u2192', svg['texts'])
        self.assertGreaterEqual(svg['polygons'], 2, 'the zone and the band')
        o = self.out['overlay']
        self.assertEqual((o['viewBox'], o['aspect'], o['ellipses']), ('0 0 1920 1080', 'none', 4))
        self.assertIn('A-02', o['texts'])
        self.assertEqual(self.out['plain'], 'A-02 passed B-05 and A-77; GK-1 stayed', 'unknown ids stay as they are')


def quality_report():
    """REPORT with the fields soccer/coaching/report.py adds: the data-quality note at the head of the limitations
    (overview.dataQualityLines), the bare per-measurement lines of a footage with a few players per team in view, a
    block-height context line, a strength's "To build on it:" and the drill library's area, equipment, regressions and
    success criteria."""
    r = json.loads(json.dumps(REPORT))
    quality = ['Players appear small in this footage, so positions and identities are less certain than on broadcast footage.',
               'What would help: set 2-3 pitch keyframes on frames where the pitch lines are clearly visible.']
    r['overview'].update(limitations=quality+['Team width: not assessed on this footage.', 'Team length: not assessed on this footage.',
                                              'Lateral shift was measured for only 4.0 s, below the 20 s needed, so it is not assessed.'],
                         dataQualityLines=2, context=["Block height: Team A's back line stood a median 30.6 m (est.) from its own goal over 60 s."])
    r['strengths'][0]['whatToImprove'] = 'To build on it: keep the width this compact when the ball is switched.'
    r['training'][0]['drill'].update(area='About 40 x 30 m, split into three zones.', equipment=['Flat markers', 'Bibs in two colours'],
                                     regressions=['Walk it through first.'],
                                     successCriteria=['In the game: the lines stay within 12 m at every freeze.',
                                                      'Back in the match: the next PitchIQ coaching report shows the gap at or below the professional median.'])
    return r


@unittest.skipUnless(NODE, 'needs node to run the page scripts')
class ReportFields(unittest.TestCase):
    """The report fields the page shows as report.py and drills.json write them."""

    @classmethod
    def setUpClass(cls):
        cls.out = run(job(answers=[answer(report=quality_report())]))[0]['first']

    def test_the_data_quality_note_is_shown_once_with_the_overview(self):
        overview, confidence = section(self.out, 'overview'), section(self.out, 'confidence')
        self.assertIn('Data qualityPlayers appear small in this footage', overview)
        self.assertIn('What would help: set 2-3 pitch keyframes', overview)
        self.assertNotIn('Players appear small', confidence, 'stated once, in the overview')
        self.assertIn('The data-quality note of this footage is in the overview.', confidence)
        self.assertIn("ContextBlock height: Team A's back line stood a median 30.6 m (est.)", overview)

    def test_bare_not_assessed_lines_are_one_line(self):
        confidence = section(self.out, 'confidence')
        self.assertIn('LimitationsNot assessed on this footage: Team width, team length.Lateral shift was measured for only 4.0 s', confidence)
        self.assertNotIn('Team width: not assessed on this footage.', confidence)

    def test_a_strength_says_how_to_build_on_it(self):
        card = next(c for c in self.out['cards'] if 'k-strength' in c['cls'])['text']
        self.assertIn('To build on it. Keep the width this compact when the ball is switched.', card)
        self.assertNotIn('What to improve.', card)
        self.assertNotIn('To build on it:', card)

    def test_the_drill_fields(self):
        training = section(self.out, 'training')
        for text in ('Area. About 40 x 30 m, split into three zones.', 'EquipmentFlat markersBibs in two colours',
                     'RegressionsWalk it through first.', 'Success looks likeIn the game: the lines stay within 12 m at every freeze.',
                     'Back in the match: the next PitchIQ coaching report shows the gap'):
            self.assertIn(text, training)
        self.assertLess(training.index('Progressions'), training.index('Regressions'))
        self.assertLess(training.index('Regressions'), training.index('Success looks like'))


@unittest.skipUnless(NODE, 'needs node to run the page scripts')
class Brackets(unittest.TestCase):
    """A team width or length is a bracket beside the team (render.py's drawing): ticks back at the team, dashed
    leaders to the two end players, the value outside; a 'bracket' element is moved beyond the team on its side."""
    L, W, M = 105.0, 68.0, 4.5

    @classmethod
    def setUpClass(cls):
        players = [{'type': 'player', 'id': f'A-0{k}', 'team': 'A', 'x': .4+.05*(k % 3), 'y': .2+.15*k, 'kind': 'O'} for k in range(5)]
        players.append({'type': 'player', 'id': 'B-07', 'team': 'B', 'x': .7, 'y': .5, 'kind': 'O'})
        base = {'preset': 'width_depth', 'title': 'Team width', 'team': 'A', 'attack': 'right', 'metresEstimated': True,
                'pitch': {'length': 105, 'width': 68}}
        width = dict(base, elements=players+[{'type': 'measure', 'a': [.545, .2], 'b': [.545, .8], 'bracket': True, 'ids': ['A-00', 'A-04'],
                                               'text': '40.8 m (est.)'}])
        length = dict(base, title='Team length', elements=players+[{'type': 'bracket', 'a': [.4, .5], 'b': [.5, .5], 'side': 'bottom',
                                                                      'text': '10.5 m (est.)'}])
        cls.width, cls.length = run({'kind': 'brackets', 'diagram': width}, {'kind': 'brackets', 'diagram': length})

    def px(self, x, y):
        return self.M+x*self.L, self.M+y*self.W

    def test_a_placed_width_bracket(self):
        x, _ = self.px(.545, 0)
        ends = sorted(self.px(.4+.05*(k % 3), 0)[0] for k in (0, 4))     # A-00 and A-04 set the extent
        text = next(t for t in self.width['texts'] if t['text'] == '40.8 m (est.)')
        self.assertEqual(text['anchor'], 'start', 'the value outside the bracket, away from the team')
        self.assertGreater(text['x'], x)
        solid = [ln for ln in self.width['lines'] if not ln['dash'] and len(ln['points']) == 2]
        self.assertTrue([ln for ln in solid if all(abs(p[0]-x) < .01 for p in ln['points'])], 'the line stands where evidence.py put it')
        ticks = [ln for ln in solid if abs(ln['points'][0][0]-x) < .01 and ln['points'][1][0] < x-.1]
        self.assertGreaterEqual(len(ticks), 2, 'end ticks point back at the team')
        leaders = [ln for ln in self.width['lines'] if ln['dash']]
        self.assertEqual(len(leaders), 2, 'a dashed leader to each end player')
        for ln in leaders:
            self.assertLess(ln['points'][1][0], ln['points'][0][0], 'the leader runs toward the team')
        stops = sorted(ln['points'][1][0] for ln in leaders)
        for stop, end in zip(stops, ends):
            self.assertTrue(end < stop < end+3, f'the leader stops just outside its end player ({stop} vs {end})')

    def test_a_bracket_element_goes_beyond_the_team_on_its_side(self):
        bottom = max(self.px(0, .2+.15*k)[1] for k in range(5))
        flat = [ln for ln in self.length['lines'] if not ln['dash'] and len(ln['points']) == 2
                and abs(ln['points'][0][1]-ln['points'][1][1]) < .01 and abs(ln['points'][0][0]-ln['points'][1][0]) > 5]
        self.assertTrue(flat)
        self.assertTrue(all(ln['points'][0][1] > bottom for ln in flat), 'below the lowest team player (side bottom)')
        text = next(t for t in self.length['texts'] if t['text'] == '10.5 m (est.)')
        self.assertGreater(text['y'], flat[0]['points'][0][1], 'the value below the bracket')


# ================= coaching v2 (DESIGN 9: setup, seven sections, key moments, annotations) =================
# The mock reports below start from schema.example('report') (the contract's own consistent mini-run) and are checked
# with schema.validate('report') before any test uses them, so the page is built against the contract, not a guess.
FPS = 30.0
SHARMA = [  # Sharma, Wu and Dalal (2005), CIEDE2000 test data: (Lab 1, Lab 2, dE00)
    ([50, 2.6772, -79.7751], [50, 0, -82.7485], 2.0425), ([50, 3.1571, -77.2803], [50, 0, -82.7485], 2.8615),
    ([50, 2.8361, -74.02], [50, 0, -82.7485], 3.4412), ([50, -1.3802, -84.2814], [50, 0, -82.7485], 1.0),
    ([50, 0, 0], [50, -1, 2], 2.3669), ([50, 2.49, -0.001], [50, -2.49, 0.0009], 7.1792), ([50, 2.5, 0], [73, 25, -18], 27.1492),
    ([50, 2.5, 0], [61, -5, 29], 22.8977), ([50, 2.5, 0], [56, -27, -3], 31.903), ([50, 2.5, 0], [58, 24, 15], 19.4535),
    ([60.2574, -34.0099, 36.2677], [60.4626, -34.1751, 39.4387], 1.2644), ([63.0109, -31.0961, -5.8663], [62.8187, -29.7946, -4.0864], 1.263),
    ([22.7233, 20.0904, -46.694], [23.0331, 14.973, -42.5619], 2.0373), ([90.8027, -2.0831, 1.441], [91.1528, -1.6435, 0.0447], 1.4441)]
L1_BANNED = re.compile(r'\b(mistakes?|balance|cover\w*|press|pressing|pressure|rest defen[cs]e|transitions?)\b', re.I)   # DESIGN 6.3, level 1


def v2_result(measured=False):
    """A small result in the page's shape: the frames of the example segment (one per second) and the people the mock
    reports name, with display labels that differ from the ids (A-04 is shown as A-03 after a renumbering)."""
    players = [{'id': i, 'display': d, 'team': i[0], 'role': 'PLAYER', 'status': 'ACTIVE'}
               for i, d in (('A-02', 'A-02'), ('A-04', 'A-03'), ('A-05', 'A-05'), ('A-06', 'A-06'), ('B-10', 'B-07'), ('B-11', 'B-08'))]
    players.append({'id': 'A-09', 'display': 'A-?', 'team': 'A', 'role': 'PLAYER', 'status': 'ACTIVE', 'confirmed': False})   # beyond the slots
    frames = [{'time': float(t), 'frame': int(t*FPS), 'people': []} for t in range(720, 900)]
    return {'version': 3, 'video': {'fps': FPS, 'width': 1920, 'height': 1080}, 'frames': frames,
            'match': {'players': players, 'goalkeepers': [{'id': 'GK-1', 'display': 'GK-A', 'team': 'A', 'role': 'GOALKEEPER'}], 'referees': []},
            'tactical': {'pitchSize': {'length': 104.0 if measured else 105.0, 'width': 67.0 if measured else 68.0, 'source': 'landmarks' if measured else 'standard'}}}


V2_INFO = {'video': {'fps': FPS, 'width': 1920, 'height': 1080}, 'videoKey': '1'*64}


def _moment(src, *, t, finding, opportunity, tier='world', detector=None, caption='A moment.', diagram=None, overlay=None, alternative=None,
            players=None):
    ev = json.loads(json.dumps(src))
    frame = int(round(t*FPS))
    ev.update(id=S.record_id('ev', opportunity, frame), opportunity=opportunity, finding=finding, time=t, frame=frame, seek=S.seek_of(frame, FPS),
              fps=FPS, segmentTime=round(t-720.0, 2), tier=tier, alternative=alternative, diagram=diagram, videoOverlay=overlay, caption=caption)
    if detector:
        ev['detector'] = detector
    if players is not None:
        ev['players'] = players
    ev['annotation'] = {'finding': finding, 'window': {'t0': round(t-3.0, 2), 't1': round(t+2.0, 2)}, 'keyFrame': frame, 'keyTime': t, 'caption': caption[:90]}
    return ev


def _stamp(ev):
    return {'evidence': ev['id'], 'opportunity': ev['opportunity'], 't': ev['time'], 'frame': ev['frame'], 'seek': ev['seek']}


def v2_base():
    return S.example('report')


def v2_rich():
    """The example report plus a strength (D6, its own moment), a review whose role could not be confirmed (D3, an
    image-tier moment with a suggested area on its diagram only), a described finding, a moment with no reachable
    area, a formation diagram, an optional drill, a progression and a withheld finding."""
    r = S.example('report')
    work, plain = r['needsWork'][0], r['keyMoments'][1]
    s_opps = [S.record_id('opp', 'D6', k) for k in range(4)]
    s_key = S.finding_key('attacking_shape', 'A', 'team', 724.0, 748.0)
    s_id = S.record_id('fnd', s_key, 4)
    s_ev = _moment(plain, t=731.0, finding=s_id, opportunity=s_opps[0], detector='attacking_shape', players=['A-04', 'A-05'],
                   caption='The team stood wide and deep with the ball.')
    strength = json.loads(json.dumps(work))
    strength.update(id=s_id, key=s_key, observation='attacking_shape', verdict='strength', priority=None,
                    players={'primary': ['A-04'], 'supporting': ['A-05'], 'opponents': []}, roles={}, timestamps=[_stamp(s_ev)],
                    opportunities=s_opps, measurements={'widthM': 48.2, 'depthM': 31.0, 'metres': 'estimated'},
                    aggregate={'opportunities': 4, 'demonstrated': 3, 'notDemonstrated': 1, 'unclear': 0, 'distinctFailingSpells': 1,
                               'distinctDemonstratedSpells': 3, 'phaseSeconds': 30.0, 'eligibleShare': .6, 'ratePer10': None},
                    suggestedAction=None, claims=[], drills=[], limitations=[],
                    text={'headline': 'The team spread wide and deep when it had the ball',
                          'explanation': 'In 3 of 4 moments with the ball (estimated) A-04 and A-05 held the width on both sides.',
                          'short': 'Wide and deep', 'action': 'Keep one player high and one wide on each side.'})
    strength['context'] = dict(work['context'], phase={'state': 'own', 'estimated': True, 'coverage': .79})
    strength['guardrails'] = dict(work['guardrails'], alternative='not_applicable')

    rv_opps = [S.record_id('opp', 'D3', k) for k in range(3)]
    rv_key = S.finding_key('wide_defender_advance', 'A', 'L-side', 800.0, 840.0)
    rv_id = S.record_id('fnd', rv_key, 4)
    alt = json.loads(json.dumps(r['alternatives'][0]))
    alt_id = S.record_id('alt', rv_id, 'A-06')
    zone = [[0.12, 0.10], [0.20, 0.10], [0.20, 0.22], [0.12, 0.22]]
    diagram = {'preset': 'wide_defender_advance', 'title': 'Advance and the channel behind', 't': 812.4, 'frame': int(round(812.4*FPS)), 'team': 'A',
               'attack': 'right', 'metresEstimated': True, 'pitch': {'length': 105.0, 'width': 68.0},
               'elements': [{'type': 'channel', 'polygon': [[0.05, 0.0], [0.30, 0.0], [0.30, 0.33], [0.05, 0.33]], 'kind': 'vacated'},
                            {'type': 'player', 'id': 'B-11', 'team': 'B', 'x': 0.25, 'y': 0.12, 'kind': 'O', 'emphasis': False},
                            {'type': 'focus', 'id': 'A-02', 'x': 0.45, 'y': 0.10, 'tone': 'review', 'label': True},
                            {'type': 'support', 'id': 'A-06', 'team': 'A', 'x': 0.22, 'y': 0.38, 'why': 'nearest centre-back'},
                            {'type': 'relation', 'a': 'A-02', 'b': 'A-06', 'kind': 'channel', 'ok': False, 'text': '24 m'},
                            {'type': 'suggestedZone', 'polygon': zone, 'kind': 'zone', 'alternative': alt_id, 'mover': 'A-06', 'label': 'possible covering area'},
                            {'type': 'moveArrow', 'id': 'A-06', 'from': [0.22, 0.38], 'to': [0.17, 0.20]},
                            {'type': 'caption', 'text': 'Left defender advanced; the channel behind stayed empty.', 'anchor': 'top-left'}]}
    overlay = {'tier': 'image', 'width': 1920, 'height': 1080,
               'elements': [{'type': 'focus', 'id': 'A-02', 'x': 900.0, 'y': 600.0, 'tone': 'review', 'label': True, 'box': [880.0, 520.0, 40.0, 80.0]},
                            {'type': 'support', 'id': 'A-06', 'team': 'A', 'x': 600.0, 'y': 700.0, 'why': 'nearest centre-back', 'box': [580.0, 620.0, 40.0, 80.0]},
                            {'type': 'caption', 'text': 'Left defender advanced; the channel behind stayed empty.', 'anchor': 'top-left'}]}
    rv_ev = _moment(plain, t=812.4, finding=rv_id, opportunity=rv_opps[0], tier='image', detector='wide_defender_advance', diagram=diagram,
                    overlay=overlay, alternative=alt_id, players=['A-02', 'A-06', 'B-11'], caption='Left defender advanced; the channel behind stayed empty.')
    alt.update(id=alt_id, finding=rv_id, evidence=rv_ev['id'], t=812.4, mover='A-06', principle='cover_behind_advancing_defender', rejected=[],
               **{'from': [0.22, 0.38]})
    alt['zone']['polygon'] = zone
    review = json.loads(json.dumps(work))
    review.update(id=rv_id, key=rv_key, observation='wide_defender_advance', verdict='review', priority=None,
                  players={'primary': ['A-02'], 'supporting': ['A-06'], 'opponents': ['B-11']},
                  roles={'A-02': {'role': 'LB', 'source': 'window', 'confidence': 'low'}}, timestamps=[_stamp(rv_ev)], opportunities=rv_opps,
                  measurements={'advanceM': 21.3, 'behindN': 1, 'channelOccupied': False, 'coverPresentShare': .25, 'metres': 'estimated'},
                  aggregate={'opportunities': 3, 'demonstrated': 1, 'notDemonstrated': 2, 'unclear': 0, 'distinctFailingSpells': 2,
                             'distinctDemonstratedSpells': 1, 'phaseSeconds': 40.0, 'eligibleShare': .5, 'ratePer10': None},
                  limitations=['role_unconfirmed'], claims=[], drills=[],
                  suggestedAction={'principle': 'cover_behind_advancing_defender', 'mode': 'zone', 'alternatives': [alt_id]},
                  text={'headline': 'A-02 advanced with nobody in the channel behind',
                        'explanation': 'When we had the ball (estimated), A-02 moved about 21 m forward while the left channel behind stayed empty.',
                        'short': 'Advance without cover', 'action': 'A-06 could move across into the channel, or A-02 could delay the run.'})
    review['guardrails'] = dict(work['guardrails'], role='fail')
    review['context'] = dict(work['context'], phase={'state': 'own', 'estimated': True, 'coverage': .79}, ball={'zone': 'middle_third', 'channel': 'left'})

    d_key = S.finding_key('defensive_spacing', 'A', 'team', 760.0, 880.0)
    described = json.loads(json.dumps(work))
    described.update(id=S.record_id('fnd', d_key, 4), key=d_key, observation='defensive_spacing', verdict='describe', priority=None,
                     players={'primary': [], 'supporting': [], 'opponents': []}, roles={}, timestamps=[], opportunities=[], measurements={},
                     aggregate={'opportunities': 0, 'demonstrated': 0, 'notDemonstrated': 0, 'unclear': 0, 'distinctFailingSpells': 0,
                                'distinctDemonstratedSpells': 0, 'phaseSeconds': 0.0, 'eligibleShare': None, 'ratePer10': None},
                     suggestedAction=None, claims=[], drills=[], limitations=[],
                     text={'headline': 'The back line stayed level', 'short': 'Level back line', 'action': None,
                           'explanation': 'The defenders kept about the same height across the pitch when the other team had the ball (estimated).'})
    described['guardrails'] = {g: 'not_applicable' for g in S.GUARDRAILS}

    r['strengths'], r['reviews'], r['described'] = [strength], [review], [described]
    r['keyMoments'] = sorted(r['keyMoments']+[s_ev, rv_ev], key=lambda m: m['time'])
    r['alternatives'].append(alt)
    second = work['timestamps'][1]
    r['noAlternatives'] = [{'finding': work['id'], 'evidence': second['evidence'], 't': second['t'], 'principle': 'cover_first_defender',
                            'reason': 'mover_unseen', 'rejected': []}]
    formation = {'preset': 'formation', 'title': 'Measured shape', 't': 760.0, 'frame': 22800, 'team': 'A', 'attack': 'right', 'metresEstimated': True,
                 'pitch': {'length': 105.0, 'width': 68.0},
                 'elements': [{'type': 'player', 'id': f'A-0{k}', 'team': 'A', 'x': .2+.05*k, 'y': .2+.1*k, 'kind': 'O'} for k in (2, 4, 5, 6)]}
    r['diagrams'] += [{'id': S.record_id('dg', rv_id), 'kind': 'finding', 'finding': rv_id, 'evidence': rv_ev['id'], 'caption': 'Advance and the channel behind',
                       'diagram': diagram},
                      {'id': S.record_id('dg', 'formation', 0), 'kind': 'formation', 'finding': None, 'evidence': None,
                       'caption': 'Measured shape 12:00-15:00 (likely 4-3-3)', 'diagram': formation}]
    r['drills'].append({'finding': None, 'drill': 'l4_defensive_unit_6v6', 'optional': True,
                        'fields': {'trains': 'Moving as a unit.', 'why': 'For the next objective of this level.', 'moments': [], 'easier': 'Play 4v4.',
                                   'harder': 'Add a target player.', 'progress': 'The line stays level in more moments.'}})
    r['progression'] = {'squad': 'sq-0123456789ab', 'note': 'Three matches linked to U14 Blue.',
                        'skills': [{'skill': 'pressure_and_cover', 'name': 'Pressure and cover', 'state': 'DEVELOPING', 'opportunities': 31, 'matches': 3, 'trend': 'up'},
                                   {'skill': 'wide_defender_support', 'name': 'Support behind advancing wide defenders', 'state': 'NOT ENOUGH EVIDENCE',
                                    'opportunities': 4, 'matches': 1, 'trend': 'unknown'}]}
    ov = r['overview']
    ov['detectors'] = [{'detector': 'pressure_cover', 'status': 'assessed', 'reason': None, 'eligibleSeconds': 34.0, 'phaseSeconds': 88.2, 'opportunities': 9},
                       {'detector': 'wide_defender_advance', 'status': 'not_assessed', 'reason': 'role_unconfirmed', 'eligibleSeconds': 0.0, 'phaseSeconds': 70.0,
                        'opportunities': 0},
                       {'detector': 'weak_side_balance', 'status': 'not_assessed', 'reason': 'out_of_view', 'eligibleSeconds': 0.0, 'phaseSeconds': 88.2,
                        'opportunities': 0},
                       {'detector': 'defensive_spacing', 'status': 'off', 'reason': None, 'eligibleSeconds': 0.0, 'phaseSeconds': None, 'opportunities': 0}]
    ov['unavailable'] = [{'category': 'passing_support', 'reason': 'No lane or body-orientation model yet.', 'unlock': 'A lane model and labels.'}]
    ov['notices'] = [{'code': 'formation_mismatch', 'text': 'The shape measured looked like 4-4-2; your selected 4-3-3 is used as the plan.'}]
    r['withheld'] = [{'finding': S.record_id('fnd', 'withheld'), 'reason': 'text_failed'}]
    return r


def v2_level1():
    """Level 1 (U8, 4v4 without a keeper): numbers none, try_this, the level's own skill names; every text in level-1
    words, so a banned word on the page can only come from the page itself."""
    r = S.example('report')
    p = r['profile']
    p.update(ageGroup='U8', level=1, levelName='FOUNDATION', formation='1-2-1', format={'players': 4, 'goalkeeper': False, 'outfield': 4, 'name': '4v4'},
             principles=[], objectives=[{'text': 'Everyone tries every position and has fun with the ball.', 'tag': 'FEDERATION', 'sources': ['USS-CG-U78']}],
             drillPool=['l1_help_your_teammate_2v2'], sources=['USS-CG-U78'])
    for d in p['detectors'].values():
        d.update(verdicts=['strength', 'try_this'], framing='go to the ball')
    p['evidence'].update(maxWorkItems=1, naming='never', minFailureShare=.6)
    p['register'] = {'tone': 'praise first', 'numbers': 'none', 'maxChallenges': 1, 'concepts': ['spread out'], 'banned': ['mistake']}
    p['safety'] = {'heading': 'none', 'maxLoad': 'low', 'tackling': False, 'sprintRaces': False, 'notes': []}
    p['skills'] = {s: None for s in S.SKILLS}
    p['skills'].update(pressure_and_cover='Go to the ball and help a teammate', someone_back_when_attacking='Someone ready to protect our goal',
                       spread_when_attacking='Spread out: tall, wide, deep')
    f = r['needsWork'][0]
    f.update(verdict='try_this', level=1, measurements={}, drills=['l1_help_your_teammate_2v2'],
             text={'headline': 'Going to the ball together', 'explanation': 'Someone went to the ball quickly in several moments.',
                   'short': 'Go to the ball', 'action': 'Who can stand close behind the teammate who goes to the ball?'})
    f['context']['format'] = '4v4'
    for m in r['keyMoments']:
        m['caption'] = 'Someone went to the ball; a teammate was far away.'
        m['annotation']['caption'] = 'Someone went to the ball.'
    for spec in (r['keyMoments'][0]['diagram'], r['keyMoments'][0]['videoOverlay'], r['diagrams'][0]['diagram']):
        spec['title'] = 'Going to the ball' if 'title' in spec else None
        if spec['title'] is None:
            del spec['title']
        for e in spec['elements']:
            if e['type'] == 'caption':
                e['text'] = 'Someone went to the ball.'
            if e['type'] == 'suggestedZone':
                e['label'] = 'a place to help'
            if e['type'] == 'relation':
                e.pop('text', None)
    r['diagrams'][0]['caption'] = 'Going to the ball'
    r['drills'][0].update(drill='l1_help_your_teammate_2v2', fields={'trains': 'Going to the ball and helping a teammate.', 'why': 'Seen in this match.',
                                                                      'moments': [r['keyMoments'][0]['id']], 'easier': 'Play 1v1.', 'harder': 'Add a small goal.',
                                                                      'progress': 'A teammate close behind more often.'})
    r['nextObjective'] = {'skill': 'pressure_and_cover', 'text': 'Going to the ball and helping a teammate.', 'fromMatch': True, 'finding': f['id']}
    r['overview']['detectors'] = [{'detector': d, 'status': 'assessed' if d == 'pressure_cover' else 'off', 'reason': None, 'eligibleSeconds': 34.0 if d == 'pressure_cover' else 0.0,
                                   'phaseSeconds': None, 'opportunities': 9 if d == 'pressure_cover' else 0} for d in S.DETECTORS]
    r['overview']['unavailable'] = [{'category': 'finding_space', 'reason': 'No model of open space yet.', 'unlock': None}]
    r['sources'] = [{'id': 'USS-CG-U78', 'title': 'U7-U8 coaching guide', 'url': 'https://www.ussoccer.com/', 'publisher': 'U.S. Soccer'}]
    return r


def v2_level2():
    """Level 2 (U10, 7v7): counts but no metres, so the measurements stay off the card."""
    r = v2_level1()
    p = r['profile']
    p.update(ageGroup='U10', level=2, levelName='EARLY DEVELOPMENT', formation='1-3-2-1', format={'players': 7, 'goalkeeper': True, 'outfield': 6, 'name': '7v7'})
    p['register'].update(numbers='counts', maxChallenges=2)
    p['evidence'].update(maxWorkItems=2, naming='relative')
    f = r['needsWork'][0]
    f.update(level=2, measurements={'d1Median': 3.1, 'coverPresentShare': .33, 'metres': 'estimated'})
    f['context']['format'] = '7v7'
    return r


def v2_drill():
    d = S.example('drill')
    return {d['drill_id']: d}


def v2_answer(report, **kw):
    return dict({'state': 'ready', 'tiers': {'measurement': 'ready', 'role': 'ready', 'interpretation': 'ready'}, 'report': report, 'feedback': [], 'reasons': [],
                 'saved': '2026-10-21T15:00:00', 'options': {'team': 'A', 't0': None, 't1': None}, 'run': 'run1', 'error': None, 'busy': False,
                 'drills': v2_drill()}, **kw)


SQUAD = {'id': 'sq-0123456789ab', 'name': 'U14 Blue', 'ageGroup': 'U14', 'level': 4, 'format': {'players': 11, 'goalkeeper': True}, 'created': '2026-10-01T10:00:00'}


def v2_settings(team_a=None):
    return {'settings': {'version': 1, 'teams': {'A': team_a, 'B': None}, 'updated': None},
            'inferred': {'A': [{'id': 'A-04', 'role': 'LCB', 'source': 'stable', 'confidence': 'medium'}, {'id': 'A-02', 'role': None, 'source': 'none', 'confidence': None}],
                         'B': []},
            'squads': [SQUAD]}


def saved_block(**kw):
    b = S.example('team_settings')
    b.update(kw)
    return b


def v2job(steps=None, report=None, answers=None, **kw):
    return dict({'kind': 'v2', 'answers': answers or [v2_answer(report if report is not None else v2_rich())], 'result': v2_result(), 'info': V2_INFO,
                 'profiles': {}, 'settings': v2_settings(saved_block()), 'steps': steps or []}, **kw)


def sec(out, key):
    return next(s['text'] for s in out['sections'] if s['key'] == key)


V2_SECTIONS = ['overview', 'wentWell', 'needsWork', 'keyMoments', 'diagrams', 'drills', 'progression']


class V2Mocks(unittest.TestCase):
    """The mock reports and records the page tests use follow the contract (schema.py)."""

    def test_the_mock_reports_follow_the_schema(self):
        for name, r in (('base', v2_base()), ('rich', v2_rich()), ('level1', v2_level1()), ('level2', v2_level2())):
            self.assertEqual(S.validate('report', r), [], name)
        self.assertEqual(S.validate('drill', next(iter(v2_drill().values()))), [])
        self.assertEqual(S.validate('team_settings', saved_block()), [])
        self.assertEqual(S.validate('squads', {'version': 1, 'squads': [SQUAD]}), [])


@unittest.skipUnless(NODE, 'needs node to run the page scripts')
class V2Report(unittest.TestCase):
    """The v2 report: the seven sections of DESIGN 9.2, the cards in the level's register, the key moments and their
    jump, the side column, the drills, the progression and the coach's feedback and adjusted areas."""

    @classmethod
    def setUpClass(cls):
        rich = v2_rich()
        work, review = rich['needsWork'][0], rich['reviews'][0]
        image_ev = next(m for m in rich['keyMoments'] if m['tier'] == 'image')
        cls.rich, cls.work, cls.review, cls.image_ev = rich, work, review, image_ev
        key_ev = work['timestamps'][0]['evidence']
        fb = [{'id': 'f1', 'findingKey': work['key'], 'attached': work['key'], 'thisRun': True, 'verdict': 'partial', 'text': 'Fair', 'made': '2026-10-21T16:00:00', 'run': 'run1'},
              {'id': 'f2', 'findingKey': 'D2|A|team|700.0|710.0', 'attached': None, 'thisRun': False, 'verdict': 'incorrect', 'text': '', 'made': '2026-10-20T10:00:00',
               'run': 'run0', 'finding': {'headline': 'An earlier far-side finding'}},
              {'id': 'f3', 'observation': {'headline': 'Gap behind A-03', 'key': 'obs-123'}, 'attached': None, 'thisRun': False, 'verdict': 'correct', 'text': '',
               'made': '2026-10-19T10:00:00', 'run': 'run0'},
              {'id': 'f4', 'findingKey': work['key'], 'attached': work['key'], 'thisRun': True, 'verdict': None, 'made': '2026-10-21T16:05:00', 'run': 'run1',
               'adjustment': {'evidence': key_ev, 'polygon': [[0.15, 0.30], [0.19, 0.30], [0.19, 0.34], [0.15, 0.34]], 'mover': 'A-05', 'note': None}}]
        cls.out = run(
            v2job(),
            v2job(steps=[{'do': 'click', 'data': {'evidence': key_ev}, 'name': 'show'}, {'do': 'click', 'data': {'replay': key_ev}, 'name': 'replay'},
                         {'do': 'click', 'data': {'board': key_ev}, 'name': 'board'}, {'do': 'click', 'data': {'evidence': image_ev['id']}, 'name': 'image'}]),
            v2job(steps=[{'do': 'click', 'within': {'feedback': work['key']}, 'data': {'verdict': 'correct'}},
                         {'do': 'click', 'within': {'feedback': work['key']}, 'text': 'Save review', 'name': 'saved'}]),
            v2job(answers=[v2_answer(v2_rich(), feedback=fb)]),
            v2job(steps=[{'do': 'click', 'in': 'side', 'data': {'adjust': 'start'}, 'name': 'adjusting'}, {'do': 'click', 'in': 'side', 'data': {'adjust': 'right'}},
                         {'do': 'click', 'in': 'side', 'data': {'adjust': 'down'}}, {'do': 'click', 'in': 'side', 'data': {'adjust': 'larger'}, 'name': 'moved'},
                         {'do': 'choose', 'in': 'side', 'tag': 'TEXTAREA', 'value': 'Deeper, nearer the centre-back', 'event': 'input'},
                         {'do': 'click', 'in': 'side', 'data': {'adjust': 'save'}, 'name': 'saved'}]),
            v2job(report=v2_level1()),
            v2job(answers=[v2_answer(None, state='missing')]),
            v2job(steps=[{'do': 'click', 'in': 'main', 'tag': 'LI', 'data': {'evidence': image_ev['id']}, 'name': 'image'}]),
            v2job(answers=[v2_answer(v2_rich(), busy=True)], hold=True, steps=[{'do': 'method', 'name': 'exportPdf'}]),
            v2job(report=v2_level2()))
        (cls.ready, cls.jumps, cls.feedback, cls.apart, cls.adjust, cls.level1, cls.missing, cls.select_image, cls.busy, cls.level2) = cls.out

    def test_the_seven_sections_in_order(self):
        out = self.ready['first']
        self.assertEqual(out['mode'], 'v2')
        self.assertEqual([s['key'] for s in out['sections']], V2_SECTIONS)
        self.assertEqual([s['key'] for s in self.apart['first']['sections']], V2_SECTIONS+['feedback'], 'feedback on no finding comes last')
        meta = [u for u in out['meta'] if not u.startswith('/api/coaching/spans')]
        self.assertTrue(meta[0].startswith('/api/coaching/profiles'))
        self.assertEqual(meta[1], '/api/coaching/settings?run=run1')
        self.assertIn('/api/coaching/spans?run=run1', out['meta'], 'the coach\u2019s possession marks of the run are asked with the result')
        self.assertEqual(out['asked'][0], '/api/coaching?run=run1&team=A', 'a v2 request carries no level: the settings are on the server')
        self.assertEqual((out['generate'], out['pdf']['disabled'], out['banner']['hidden']), ({'text': 'Report up to date', 'disabled': True}, False, True))

    def test_match_overview(self):
        ov = sec(self.ready['first'], 'overview')
        for text in ('Coaching profileU14 \u00b7 level 4 Intermediate', 'Match format11v11 \u00b7 10 outfield players',
                     'Intended formation4-3-3 (a starting model, not where every player must stand)', 'Style of playUnknown: no intended style is assumed',
                     'Who had the ballTeam A about 45%, Team B about 55% (estimated from 79% of the time)',
                     'Shares are estimates; broadcast stoppages are mostly missed.', '105 \u00d7 68 m, assumed; distances are estimates',
                     'The shape measured looked like 4-4-2', '12:00.00\u201314:59.80: 4-3-3 (likely)',
                     'checked in 9 moments (34.0 s usable)', 'not assessed: the players\u2019 roles are not confirmed (confirm them under Player roles',
                     'not assessed: the players involved were often out of the camera view', 'Back line coordination: not part of this level\u2019s checks',
                     'Passing support: No lane or body-orientation model yet. What would make it possible: A lane model and labels.',
                     'Context from this match', 'The back line stayed level', '1 finding was held back'):
            self.assertIn(text, ov)

    def test_cards_in_the_level_register(self):
        cards = {c['finding']: c for c in self.ready['first']['cards']}
        work, review = cards[self.work['id']], cards[self.review['id']]
        self.assertIn('k-improvement', work['cls'])
        for text in ('Work on', 'Priority 1', 'Pressure on the ball came without cover behind it', 'What could help. One presses, the nearest teammate covers.',
                     'Missing in 5 of 9 moments (3 separate spells), 1 unclear, when the other team had the ball (estimated), over 88.2 s of that time.',
                     'Nearest player to the ball 3.1 m (est.)', 'A possible better area is drawn on the key moment', 'Moments (3)', 'Drill: 3v3 Cover the Line'):
            self.assertIn(text, work['text'])
        self.assertEqual(work['metrics'], 1, 'level 4 shows metres, not shares (coverPresentShare is left out)')
        self.assertIn('Your review', work['text'])
        self.assertIn('k-review', review['cls'])
        for text in ('Possible issue to review', 'Could not be confirmed: the player\u2019s tactical role.', 'Limits: the players\u2019 roles are not confirmed',
                     'A-02 advanced with nobody in the channel behind', 'Advance 21.3 m (est.)', 'Behind the ball 1', 'Channel behind covered no'):
            self.assertIn(text, review['text'])
        strength = next(c for c in self.ready['first']['cards'] if c['verdict'] == 'strength')
        self.assertIn('Shown in 3 of 4 moments (3 separate spells), when we had the ball (estimated)', strength['text'])
        self.assertIn('To build on it.', strength['text'])
        self.assertNotIn('Priority', strength['text'])
        self.assertIn('The team spread wide and deep', sec(self.ready['first'], 'wentWell'))
        nw = sec(self.ready['first'], 'needsWork')
        self.assertIn('At most 3 items are shown at this level', nw)
        self.assertLess(nw.index('Pressure on the ball came'), nw.index('Possible issues to review'), 'work items, then the reviews')
        self.assertIn('In 3 of 4 moments with the ball (estimated) A-03 and A-05', strength['text'], 'ids shown as display labels')

    def test_key_moments_are_chronological_with_their_tier(self):
        rows = self.ready['first']['moments']
        self.assertEqual([r['evidence'] for r in rows], [m['id'] for m in self.rich['keyMoments']])
        image = next(r for r in rows if r['evidence'] == self.image_ev['id'])
        self.assertIn('Boxes only', image['text'])
        self.assertIn('A-02, A-06, B-08', image['text'], 'players by display label')
        self.assertIn('k-review', image['cls'])
        first = next(r for r in rows if r['evidence'] == self.work['timestamps'][0]['evidence'])
        self.assertIn('Pitch-accurate', first['text'])
        self.assertIn('selected', first['cls'], 'the first work item\u2019s first moment is selected')

    def test_a_key_moment_jumps_with_its_annotation(self):
        out, ev = self.jumps, self.rich['keyMoments'][1]
        show = out['show']['jumps'][0]
        self.assertEqual((show['frame'], show['seek'], show['view'], show['tier']), (ev['frame'], ev['seek'], 'tracking', 'world'))
        self.assertAlmostEqual(show['seek'], (ev['frame']+.5)/FPS, places=3)
        self.assertEqual(show['focus'], ['A-04', 'A-05', 'B-10'], 'the primary player first')
        self.assertEqual(show['window'], {'t0': 758.0, 't1': 763.0})
        self.assertEqual(show['caption'], 'Pressure on the ball; nobody covered behind it.')
        self.assertTrue(show['explanation'].startswith('When the other team had the ball (estimated) in our third, A-03 closed the carrier'),
                        'the chip\u2019s explanation names players by their display label (A-04 is shown as A-03)')
        self.assertEqual(show['follow'], {'focus': [{'id': 'A-04', 'tone': 'review', 'secondary': False}], 'support': [{'id': 'A-05', 'team': 'A'}],
                                          'relations': [{'a': 'A-04', 'b': 'A-05', 'kind': 'cover', 'ok': False}]})
        self.assertEqual((show['replay'], show['diagram'], show['overlay']), (False, True, True))
        self.assertTrue(out['replay']['jumps'][1]['replay'])
        self.assertEqual(out['board']['jumps'][2]['view'], 'board')
        image = out['image']['jumps'][3]
        self.assertEqual((image['view'], image['tier']), ('board', 'image'), 'an image-tier moment opens on the 2D match (DESIGN 9.3)')
        self.assertEqual(out['image']['selectedMoment'], self.image_ev['id'])
        self.assertEqual(out['image']['selected'], self.review['id'])

    def test_the_side_column_shows_the_selected_moment(self):
        side = self.ready['first']['side']
        ev = self.rich['keyMoments'][1]
        self.assertEqual(side['images'], [f'/api/coaching/frame?run=run1&frame={ev["frame"]}&w=640'])
        self.assertEqual(sorted(side['svgs']), ['cr-diagram', 'cr-overlay'])
        for text in ('Pitch-accurate: Drawn through this frame', 'Pressure on the ball; nobody covered behind it.',
                     'The dotted area is one possibility for A-05, with everyone else kept where they were', 'within reach of where A-05 stood 2.0 s earlier at up to 5.6 m/s',
                     'cover holds in 92% of it', 'PlayersA-03, A-05, B-07', 'Registration confidence0.82'):
            self.assertIn(text, side['text'])
        self.assertIn('Adjust the area', side['buttons'])
        self.assertIn('Replay', side['buttons'])
        image = self.select_image['image']['side']
        self.assertIn('Suggested area shown on the 2D diagram.', image['text'])
        self.assertIn('Boxes only:', image['text'])
        zone_on_video = [s for s in image['overlay']['shapes'] if s['dash'] and s['stroke'] == '#7dffb0']
        self.assertEqual(zone_on_video, [], 'no suggested area over an image-tier frame')
        self.assertTrue(any(s['stroke'] == '#7dffb0' for s in image['diagram']['shapes']), 'the area is on the diagram')

    def test_a_pinned_moment_says_where_the_area_is(self):
        r = v2_rich()
        ev = next(m for m in r['keyMoments'] if m['tier'] == 'image')
        ev['tier'] = 'pinned'
        ev['videoOverlay']['tier'] = 'pinned'
        self.assertEqual(S.validate('report', r), [])
        jump = run(v2job(report=r, steps=[{'do': 'click', 'data': {'evidence': ev['id']}, 'name': 'show'}]))[0]['show']['jumps'][0]
        self.assertEqual(jump['view'], 'tracking')
        self.assertTrue(jump['caption'].endswith('. Suggested area shown on the 2D diagram.'), jump['caption'])
        self.assertNotIn('2D diagram', self.jumps['show']['jumps'][0]['caption'], 'a world-tier moment draws its area on the video')

    def test_no_reachable_area_is_said(self):
        out = run(v2job(steps=[{'do': 'click', 'data': {'evidence': self.work['timestamps'][1]['evidence']}, 'name': 'second'}]))[0]
        self.assertIn('No suggested area: the teammates who could help were not in view just before. The principle to review is given instead.', out['second']['side']['text'])

    def test_diagrams_drills_and_progression(self):
        out = self.ready['first']
        dg = sec(out, 'diagrams')
        for text in ('Tactical diagrams \u00b7 3', 'Finding Pressure without cover', 'Shape Measured shape', 'A dotted area is a possible better position'):
            self.assertIn(text, dg)
        dr = sec(out, 'drills')
        for text in ('3v3 Cover the Line', 'For: Pressure on the ball came without cover behind it', 'What it trains. Pressure and cover.',
                     'Why it was chosen. Cover was missing in 5 of 9 moments.', 'Moments it addresses: 12:41.00', 'Make it easier. Play 2v2.',
                     'Make it harder. Add a fourth attacker.', 'What progress looks like next match. Cover present in more than 5 of 9 pressured moments.',
                     'L4 defensive unit 6v6', 'Optional: for the next objective, not from this match\u2019s evidence.'):
            self.assertIn(text, dr)
        self.assertLess(dr.index('Why it was chosen'), dr.index('Moments it addresses'))
        self.assertLess(dr.index('Moments it addresses'), dr.index('Make it easier'))
        drill = next(iter(v2_drill().values()))
        for text in (f'Players. {drill["player_count"]["min"]}', 'Duration.', 'Safety. no heading', 'Source.'):
            self.assertIn(text, dr)
        pr = sec(out, 'progression')
        for text in ('Pressure and coverDetailsDeveloping313improving', 'Not enough evidence yet', 'Three matches linked to U14 Blue.',
                     'Next development objective', 'The finding: Pressure on the ball came', 'proposals, not validated'):
            self.assertIn(text, pr)

    def test_feedback_is_keyed_by_the_finding_key(self):
        post = self.feedback['saved']['posted'][0]
        self.assertEqual(post, {'url': '/api/coaching/feedback', 'body': {'run': 'run1', 'team': 'A', 'findingKey': self.work['key'], 'verdict': 'correct', 'text': '',
                                                                          'range': {'t0': 720.0, 't1': 899.8}}})
        work = next(c for c in self.apart['first']['cards'] if c['finding'] == self.work['id'])
        self.assertIn('Saved 2026-10-21 16:00:00: Partially correct', work['text'])
        fb = sec(self.apart['first'], 'feedback')
        self.assertIn('From an earlier runAn earlier far-side finding \u00b7 Incorrect (run run0', fb)
        self.assertIn('Feedback from the previous report engineGap behind A-03 \u00b7 Correct', fb)
        side = self.apart['first']['side']['diagram']
        self.assertTrue(any(t['text'] == 'coach\u2019s area' for t in side['texts']), 'the coach\u2019s saved area is drawn beside PitchIQ\u2019s')

    def test_a_suggested_area_can_be_adjusted_and_saved(self):
        out = self.adjust
        self.assertIn('Drag the area on the diagram', out['adjusting']['side']['text'])
        self.assertTrue(any(t['text'] == 'your adjusted area' for t in out['moved']['side']['diagram']['texts']))
        post = out['saved']['posted'][0]
        self.assertEqual(post['url'], '/api/coaching/feedback')
        adj = post['body']['adjustment']
        zone = [[0.17, 0.36], [0.21, 0.36], [0.21, 0.4], [0.17, 0.4]]
        cx, cy = .19+1/105, .38+1/68
        want = [[round(cx+(x+1/105-cx)/.85, 4), round(cy+(y+1/68-cy)/.85, 4)] for x, y in zone]
        for (x, y), (wx, wy) in zip(adj['polygon'], want):
            self.assertAlmostEqual(x, wx, places=3)
            self.assertAlmostEqual(y, wy, places=3)
        self.assertEqual((adj['evidence'], adj['mover'], adj['note']), (self.rich['keyMoments'][1]['id'], 'A-05', 'Deeper, nearer the centre-back'))
        self.assertEqual(post['body']['findingKey'], self.work['key'])
        self.assertEqual(S.validate('adjustment', {'id': 'adj-0123456789ab', 'findingKey': self.work['key'], 'evidence': adj['evidence'], 'polygon': adj['polygon'],
                                                   'mover': adj['mover'], 'note': adj['note'], 'created': '2026-10-21T16:00:00'}), [])
        self.assertIn('Saved: your area is kept with this video', out['saved']['side']['text'])

    def test_level_1_shows_no_numbers_and_adds_no_banned_word(self):
        out = self.level1['first']
        card = out['cards'][0]
        self.assertIn('Try this', card['text'])
        self.assertIn('Who can stand close behind the teammate', card['text'])
        self.assertNotIn('Missing in', card['text'])
        self.assertEqual(card['metrics'], 0)
        self.assertIn('Looked at when the other team had the ball (estimated).', card['text'])
        main = ' '.join(s['text'] for s in out['sections'])
        self.assertEqual(L1_BANNED.findall(main), [], 'the page adds no word level 1 bans (DESIGN 6.3)')
        self.assertIn('Go to the ball and help a teammate: checked', main, 'the level\u2019s own name of the check')
        self.assertIn('Try this', sec(out, 'needsWork'))
        self.assertIn('At most 1 item is shown', sec(out, 'needsWork'))

    def test_level_2_counts_without_measurements(self):
        card = self.level2['first']['cards'][0]
        self.assertIn('Missing in 5 of 9 moments (3 separate spells), 1 unclear, when the other team had the ball (estimated).', card['text'])
        self.assertNotIn(' s of that time', card['text'], 'no seconds below level 3')
        self.assertEqual(card['metrics'], 0, 'no metres at level 2 (DESIGN 6.3)')

    def test_no_text_names_an_unmeasured_event(self):
        for out in (self.ready['first'], self.level1['first']):
            main = ' '.join(s['text'] for s in out['sections'])
            self.assertEqual(S.event_problems(main), [])

    def test_no_report_yet_and_busy(self):
        out = self.missing['first']
        self.assertEqual([s['key'] for s in out['sections']], ['empty'])
        self.assertIn('No coaching report yet for Team A (whole analysed segment; U14 \u00b7 level 4 Intermediate \u00b7 11v11 \u00b7 4-3-3', out['sections'][0]['text'])
        self.assertIn('only the match format, the field size or the possession you mark measures them again', out['sections'][0]['text'])
        self.assertEqual(out['generate'], {'text': 'Generate report', 'disabled': False})
        busy = self.busy['first']
        self.assertTrue(busy['banner']['hidden'], 'a ready report never shows "another report is being built" (critic C4)')
        self.assertEqual(busy['generate']['disabled'], True)

    def test_the_pdf_carries_the_team_colours(self):
        self.assertEqual(self.busy['last']['downloads'], ['/api/coaching/pdf?run=run1&team=A&colourA=%23b98cff&colourB=%23f6b564'])

    def test_a_measurement_change_says_generate(self):
        out = run(v2job(answers=[v2_answer(None, state='missing', tiers={'measurement': 'missing', 'role': 'missing', 'interpretation': 'missing'})]))[0]['first']
        self.assertTrue(out['banner']['hidden'], 'a run never generated: the empty report says what Generate does, no "changed" banner')
        why = run(v2job(answers=[v2_answer(None, state='missing', reasons=['The possession you marked changed.'],
                                           tiers={'measurement': 'missing', 'role': 'missing', 'interpretation': 'missing'})]))[0]['first']
        self.assertIn('The possession you marked changed. This run\u2019s situations are measured again', why['banner']['text'], 'the server\u2019s reason, when it gives one')


@unittest.skipUnless(NODE, 'needs node to run the page scripts')
class V2Setup(unittest.TestCase):
    """The coaching setup (DESIGN 9.1): age group, playing level, format by age (U.S. defaults, overridable), formation
    by format, style, federation, field size, player roles, squad; a save posts a block that follows the schema."""

    @classmethod
    def setUpClass(cls):
        empty = v2_settings(None)
        cls.out = run(
            v2job(settings=empty, steps=[
                {'do': 'choose', 'in': 'setup', 'data': {'setup': 'age'}, 'value': 'U10', 'name': 'u10'},
                {'do': 'choose', 'in': 'setup', 'data': {'setup': 'level'}, 'value': '2', 'name': 'level2'},
                {'do': 'choose', 'in': 'setup', 'data': {'setup': 'formation'}, 'value': '1-3-2-1'},
                {'do': 'click', 'in': 'setup', 'data': {'setup': 'roles'}, 'name': 'roles'},
                {'do': 'choose', 'in': 'setup', 'data': {'role': 'A-04'}, 'value': 'back-1/3', 'name': 'role'},
                {'do': 'click', 'in': 'setup', 'data': {'setup': 'save'}, 'name': 'saved'}],
                postAnswers={'/api/coaching/settings': {'ok': True, 'settings': None, 'stale': ['measurement', 'role', 'interpretation']}}),
            v2job(settings=empty, steps=[
                {'do': 'choose', 'in': 'setup', 'data': {'setup': 'age'}, 'value': 'U13', 'name': 'u13'},
                {'do': 'choose', 'in': 'setup', 'data': {'setup': 'federation'}, 'value': 'England', 'name': 'england'},
                {'do': 'choose', 'in': 'setup', 'data': {'setup': 'format'}, 'value': '11v11', 'name': 'override'},
                {'do': 'choose', 'in': 'setup', 'data': {'setup': 'age'}, 'value': 'U11', 'name': 'kept'},
                {'do': 'choose', 'in': 'setup', 'data': {'setup': 'format'}, 'value': '7v7', 'name': 'back'},
                {'do': 'choose', 'in': 'setup', 'data': {'setup': 'age'}, 'value': 'U7', 'name': 'u7'},
                {'do': 'choose', 'in': 'setup', 'data': {'setup': 'format'}, 'value': '4v4', 'name': 'four'},
                {'do': 'choose', 'in': 'setup', 'data': {'setup': 'goalkeeper'}, 'checked': True, 'name': 'keeper'},
                {'do': 'choose', 'in': 'setup', 'data': {'setup': 'level'}, 'value': '6', 'name': 'level6'},
                {'do': 'choose', 'in': 'setup', 'data': {'setup': 'field-length'}, 'value': '40', 'name': 'half'},
                {'do': 'click', 'in': 'setup', 'data': {'setup': 'save'}, 'name': 'refused'},
                {'do': 'choose', 'in': 'setup', 'data': {'setup': 'field-width'}, 'value': '25'},
                {'do': 'choose', 'in': 'setup', 'data': {'setup': 'squad'}, 'value': 'new', 'name': 'newsquad'},
                {'do': 'choose', 'in': 'setup', 'data': {'setup': 'squad-name'}, 'value': 'U7 Green', 'event': 'input'},
                {'do': 'click', 'in': 'setup', 'data': {'setup': 'save'}, 'name': 'saved'}],
                postAnswers={'/api/coaching/squads': {'squad': {'id': 'sq-aaaaaaaaaaaa', 'name': 'U7 Green'}},
                             '/api/coaching/settings': {'ok': True, 'settings': None, 'stale': ['measurement']}}),
            v2job(steps=[{'do': 'click', 'in': 'setup', 'data': {'setup': 'toggle'}, 'name': 'opened'},
                         {'do': 'choose', 'in': 'setup', 'data': {'setup': 'style'}, 'value': 'possession', 'name': 'style'},
                         {'do': 'click', 'in': 'setup', 'data': {'setup': 'info'}, 'name': 'info'},
                         {'do': 'click', 'in': 'setup', 'data': {'setup': 'save'}, 'name': 'saved'}],
                  postAnswers={'/api/coaching/settings': {'ok': True, 'settings': None, 'stale': ['interpretation']}}),
            v2job(settings=empty, steps=[{'do': 'click', 'in': 'setup', 'data': {'setup': 'save'}, 'name': 'refused'}]),
            v2job(profiles=None, settings=None, answers=[answer()], result='shared', info=INFO, steps=[
                {'do': 'choose', 'in': 'setup', 'data': {'setup': 'age'}, 'value': 'U9'}, {'do': 'choose', 'in': 'setup', 'data': {'setup': 'level'}, 'value': '2'},
                {'do': 'click', 'in': 'setup', 'data': {'setup': 'save'}, 'name': 'saved'}]),
            v2job(settings=empty, profiles={'formatDefaults': {'auto': {'U10': {'players': 9, 'goalkeeper': True}}}},
                  steps=[{'do': 'choose', 'in': 'setup', 'data': {'setup': 'age'}, 'value': 'U10', 'name': 'u10'}]),
            v2job(settings=v2_settings(saved_block(squad={'id': SQUAD['id'], 'name': SQUAD['name']}, runLinks=[{'run': 'run0', 'team': 'B', 'confirmed': True}])),
                  steps=[{'do': 'click', 'in': 'setup', 'data': {'setup': 'toggle'}, 'name': 'opened'}, {'do': 'click', 'in': 'setup', 'data': {'setup': 'link'}, 'name': 'linked'}]),
            v2job(settings=empty, result=v2_result(measured=True)))
        (cls.flow, cls.formats, cls.saved, cls.unset, cls.legacy, cls.server, cls.link, cls.measured) = cls.out

    def test_a_new_team_opens_the_setup_with_nothing_guessed(self):
        out = self.flow['first']
        self.assertFalse(out['setup']['hidden'])
        sel = out['setup']['selects']
        self.assertEqual((sel['age']['value'], sel['level']['value'], sel['format']['value'], sel['style']['value'], sel['federation']['value']),
                         ('', '', '11v11', 'unknown', 'auto'))
        self.assertEqual(sel['age']['options'], ['']+list(S.AGE_GROUPS))
        self.assertEqual(sel['level']['options'], ['']+[str(k) for k in range(1, 8)])
        self.assertIn('2 \u00b7 Early development', sel['level']['texts'])
        self.assertEqual(sel['format']['options'], list(S.FORMAT_NAMES))
        self.assertEqual(sel['formation']['options'], ['']+list(S.FORMATIONS[11]))
        self.assertIn('Not saved for this team yet', out['setup']['text'])
        self.assertEqual(out['setup']['save']['disabled'], True, 'nothing to save before a choice')

    def test_the_format_follows_the_age_until_the_coach_changes_it(self):
        out = self.flow
        self.assertEqual(out['u10']['setup']['selects']['format']['value'], '7v7')
        self.assertIn('7v7 is the default for U10 (U.S. Soccer); you can change it.', out['u10']['setup']['text'])
        self.assertEqual(out['u10']['setup']['selects']['formation']['options'], ['']+list(S.FORMATIONS[7]))
        f = self.formats
        self.assertEqual(f['u13']['setup']['selects']['format']['value'], '11v11')
        self.assertEqual(f['england']['setup']['selects']['format']['value'], '9v9', 'England plays 9v9 at U13 (DESIGN 2.2)')
        self.assertEqual(f['override']['setup']['selects']['format']['value'], '11v11')
        self.assertEqual(f['kept']['setup']['selects']['format']['value'], '11v11', 'the coach\u2019s format is kept when the age changes')
        self.assertIn('U11 usually plays 7v7 (England); your choice, 11v11, is used.', f['kept']['setup']['text'])
        self.assertIn('7v7 is the default for U11 (England); you can change it.', f['back']['setup']['text'])
        self.assertEqual(f['u7']['setup']['selects']['format']['value'], '5v5', 'a format equal to the earlier default follows the age')
        self.assertIn('5v5 with goalkeeper is the default for U7 (England); you can change it.', f['u7']['setup']['text'],
                      'England plays 5v5 at U7-U8 (FA-PITCH-U78; SOURCES F1)')
        self.assertEqual(f['four']['setup']['gk'], False, 'U.S. 4v4 has no goalkeeper; the toggle is offered')
        self.assertEqual(f['keeper']['setup']['gk'], True)
        self.assertIn('U7 usually plays 5v5 with goalkeeper (England); your choice, 4v4 with goalkeeper, is used.', f['keeper']['setup']['text'])
        self.assertIn('Small-sided matches are never judged against 11v11 positions', f['four']['setup']['text'])
        self.assertIn('Level 6 is well above the typical level for U7 (level 1)', f['level6']['setup']['text'])
        self.assertEqual(self.server['u10']['setup']['selects']['format']['value'], '9v9', 'the server\u2019s format table wins')

    def test_roles_by_format(self):
        roles = self.flow['roles']['setup']
        self.assertIn('Measured: LCB (stable, medium)', roles['text'])
        self.assertIn('Measured: no stable role', roles['text'])
        self.assertEqual(roles['roles']['A-04']['options'], ['', 'back-1/3', 'back-2/3', 'back-3/3', 'mid-1/2', 'mid-2/2', 'front-1/1'],
                         '7v7 1-3-2-1: relative slots, never 11v11 role names (DESIGN 2.5); no GK: the team has a tracked keeper')
        self.assertEqual(roles['roles']['GK-1']['options'], ['', 'GK'], 'the tracked goalkeeper is offered the goalkeeper only (CODE L9)')
        self.assertEqual(set(roles['roles']), {'GK-1', 'A-02', 'A-04', 'A-05', 'A-06'}, 'A-09 (shown as A-?) is not listed')
        self.assertIn('1 Team A identity without a numbered slot (shown as A-?) is not listed', roles['text'])
        self.assertIn('Player roles (optional) \u00b7 1 confirmed', self.flow['role']['setup']['text'])
        opened = self.saved['opened']['setup']
        self.assertEqual(opened['selects']['formation']['value'], '4-3-3')

    def test_a_save_posts_a_schema_valid_block_and_asks_again(self):
        out = self.flow['saved']
        post = next(p for p in out['posted'] if p['url'] == '/api/coaching/settings')
        self.assertEqual((post['body']['run'], post['body']['team']), ('run1', 'A'))
        block = post['body']['settings']
        self.assertEqual(S.validate('team_settings', block), [])
        self.assertEqual((block['ageGroup'], block['playingLevel'], block['format'], block['formation']['intended']), ('U10', 2, {'players': 7, 'goalkeeper': True}, '1-3-2-1'))
        self.assertEqual([s['role'] for s in block['roles']['A-04']], ['back-1/3'])
        self.assertIn('Saved. This run was already measured with this match format and field size: the report below is up to date.',
                      out['setup']['text'], 'the answer after the save is ready: no "press Generate" (CODE L3)')
        self.assertGreater(len(out['asked']), len(self.flow['first']['asked']), 'the report is asked again after a save')
        self.assertEqual(out['setup']['selects'], {}, 'collapsed after the save')
        saved = self.saved['saved']
        body = saved['posted'][-1]['body']['settings']
        self.assertEqual(body['style'], 'possession')
        self.assertIn('Saved. The report below was re-interpreted for this profile from the same measurements.', saved['setup']['text'])

    def test_a_squad_is_created_then_linked(self):
        out = self.formats
        self.assertIn('Field size: a length of 15 to 130 m', out['refused']['setup']['text'])
        self.assertEqual(out['refused']['posted'], [])
        posts = out['saved']['posted']
        self.assertEqual([p['url'] for p in posts], ['/api/coaching/squads', '/api/coaching/settings'])
        self.assertEqual(posts[0]['body'], {'name': 'U7 Green', 'ageGroup': 'U7', 'level': 6, 'format': {'players': 4, 'goalkeeper': True}})
        block = posts[1]['body']['settings']
        self.assertEqual(S.validate('team_settings', block), [])
        self.assertEqual((block['squad'], block['runLinks'], block['field']), ({'id': 'sq-aaaaaaaaaaaa', 'name': 'U7 Green'}, [{'run': 'run1', 'team': 'A', 'confirmed': True}],
                                                                              {'length': 40, 'width': 25}))
        self.assertIn('Link Team A of this run to U14 Blue', self.link['opened']['setup']['text'])
        self.assertEqual(self.link['linked']['setup']['save']['disabled'], False, 'linking this run is a change to save')

    def test_age_and_level_are_required(self):
        out = self.unset['refused']
        self.assertEqual(out['posted'], [])

    def test_the_level_information_and_sources(self):
        info = self.saved['info']['setup']['text']
        for text in ('Level 4 \u00b7 Intermediate', 'Cover is the angle and distance', 'Federation: EF-COVER-RECOVER',
                     'PitchIQ\u2019s rules for this level (proposals, not validated)', 'A work item needs at least 10 moments and 2 separate spells',
                     'Heading practice only within the stricter of the U.S. Soccer and England limits', 'Sources that informed this level',
                     'Cover and recover \u00b7 England Football [EF-COVER-RECOVER]', 'No federation reviewed or endorses PitchIQ.'):
            self.assertIn(text, info)

    def test_a_v1_server_keeps_the_setup_on_the_page(self):
        out = self.legacy
        self.assertEqual(out['first']['mode'], 'legacy')
        self.assertTrue(out['first']['asked'][0].endswith('&context=competitive'))
        self.assertIn('/api/coaching?run=run1&team=A&context=development', out['saved']['asked'], 'level 2 asks the v1 development wording')
        self.assertEqual([p['url'] for p in out['saved']['posted']], [], 'nothing is posted to a server without the route')
        self.assertIn('Kept on this page', out['saved']['setup']['text'])

    def test_a_measured_field_is_said(self):
        self.assertIn('Field measured by calibration: about 104 \u00d7 67 m.', self.measured['first']['setup']['text'])


@unittest.skipUnless(NODE and CATALOG, 'needs node and profiles.catalog()')
class V2Catalog(unittest.TestCase):
    """The setup on the catalogue the server will send (profiles.catalog(), PROFILE's shape): ages, formats and
    formations, format defaults by age and federation, roles per formation, the level information and its sources."""

    @classmethod
    def setUpClass(cls):
        cls.out = run(v2job(profiles=CATALOG, settings=v2_settings(None), report=v2_level1(), steps=[
            {'do': 'choose', 'in': 'setup', 'data': {'setup': 'age'}, 'value': 'U11', 'name': 'u11'},
            {'do': 'choose', 'in': 'setup', 'data': {'setup': 'federation'}, 'value': 'England', 'name': 'england'},
            {'do': 'choose', 'in': 'setup', 'data': {'setup': 'level'}, 'value': '3'},
            {'do': 'choose', 'in': 'setup', 'data': {'setup': 'formation'}, 'value': '1-3-2-1'},
            {'do': 'click', 'in': 'setup', 'data': {'setup': 'roles'}, 'name': 'roles'},
            {'do': 'click', 'in': 'setup', 'data': {'setup': 'info'}, 'name': 'info'},
            {'do': 'choose', 'in': 'setup', 'data': {'setup': 'age'}, 'value': 'U8'},
            {'do': 'choose', 'in': 'setup', 'data': {'setup': 'format'}, 'value': '4v4', 'name': 'four'}]))[0]

    def test_the_server_catalogue_drives_the_setup(self):
        sel = self.out['first']['setup']['selects']
        self.assertEqual(sel['age']['options'], ['']+[a['id'] for a in CATALOG['ageGroups']])
        self.assertEqual(sel['format']['options'], [f['name'] for f in CATALOG['formats']])
        self.assertEqual(sel['formation']['options'], ['']+[f['name'] for f in CATALOG['formats'][-1]['formations']])
        self.assertEqual(sel['style']['options'], [s['id'] for s in CATALOG['styles']])
        self.assertEqual(sel['federation']['options'], CATALOG['federations'])
        name = lambda f: '{0}v{0}'.format(f['players'])
        self.assertEqual(self.out['u11']['setup']['selects']['format']['value'], name(PROFILES.default_format('U11', 'auto')))
        self.assertEqual(self.out['england']['setup']['selects']['format']['value'], name(PROFILES.default_format('U11', 'England')))
        self.assertIn('7v7 is the default for U11 (England); you can change it.', self.out['england']['setup']['text'],
                      'the federation whose format it is (S-13)')
        self.assertIn(next(f['note'] for f in CATALOG['formats'] if f['players'] == 7), self.out['england']['setup']['text'])

    def test_roles_and_level_information_from_the_catalogue(self):
        roles = self.out['roles']['setup']['roles']
        want = next(f for f in CATALOG['formats'] if f['players'] == 7)['roles']['1-3-2-1']
        self.assertEqual(roles['A-04']['options'], ['']+[r for r in want if r != 'GK'], 'outfield roles: the team has a tracked keeper')
        self.assertEqual(roles['GK-1']['options'], ['', 'GK'])
        info = self.out['info']['setup']['text']
        lv = CATALOG['levels'][2]
        self.assertIn(lv['objectives'][0]['text'], info)
        self.assertIn('A work item needs at least %d moments' % lv['evidence']['minOpportunities'], info)
        self.assertIn(CATALOG['disclaimer'], info)
        self.assertIn(next(a for a in CATALOG['ageGroups'] if a['id'] == 'U11')['heading']['text'], info)
        source = next(s for s in CATALOG['sources'] if s['id'] == lv['objectives'][0]['sources'][0])
        self.assertIn(source['title']+' \u00b7 '+source['publisher']+' ['+source['id']+']', info)

    def test_the_goalkeeper_choice_follows_the_catalogue(self):
        self.assertEqual(self.out['four']['setup']['gk'], next(f for f in CATALOG['formats'] if f['players'] == 4)['goalkeeperDefault'])
        self.assertIsNone(self.out['england']['setup']['gk'], 'no goalkeeper toggle for 7v7')


@unittest.skipUnless(NODE, 'needs node to run the page scripts')
class V2FixRound1(unittest.TestCase):
    """Fix round 1 on the page (critic S-11, B-11, CODE M3, L2, L3, L9, L11)."""

    @classmethod
    def setUpClass(cls):
        rich = v2_rich()
        inferred = [{'id': 'A-04', 'role': 'LB', 'source': 'stable', 'confidence': 'high'}]
        one = v2_rich()
        one['overview']['possession'] = {'available': True, 'quality': 'limited', 'reason': None, 'roughShares': {'A': 1.0, 'B': 0.0}, 'coverage': .3,
                                         'caveat': 'Who had the ball comes from the spans you marked (coach-marked); PitchIQ could not follow it on this footage.'}
        one['overview']['detectors'][0].update(status='not_assessed', reason='coach_marks_one_side', eligibleSeconds=0.0, opportunities=0)
        both = json.loads(json.dumps(one))
        both['overview']['possession']['roughShares'] = {'A': .4, 'B': .6}
        both['overview']['detectors'][0].update(status='assessed', reason=None, eligibleSeconds=12.0, opportunities=4)
        missing = v2_answer(None, state='missing', tiers={'measurement': 'missing', 'role': 'missing', 'interpretation': 'missing'})
        (cls.roles, cls.broken, cls.html404, cls.one, cls.both, cls.press, cls.squad, cls.oneReport) = run(
            v2job(answers=[v2_answer(rich, inferred=inferred)], steps=[{'do': 'click', 'in': 'setup', 'data': {'setup': 'toggle'}},
                                                                        {'do': 'click', 'in': 'setup', 'data': {'setup': 'roles'}, 'name': 'roles'}]),
            v2job(profiles={'__error': 'The adaptive coaching report is not available in this version: broken profiles'}),
            v2job(profiles={'__error': 'Unexpected token < in JSON', '__name': 'SyntaxError', '__status': 404}),
            marks_job([], report=one, answers=[v2_answer(one, coachSpans=True)], spans=spans_answer([{'t0': 722.0, 't1': 740.0, 'state': 'A'}])),
            marks_job([], report=both, answers=[v2_answer(both, coachSpans=True)],
                      spans=spans_answer([{'t0': 722.0, 't1': 740.0, 'state': 'A'}, {'t0': 740.0, 't1': 760.0, 'state': 'B'}])),
            v2job(answers=[v2_answer(rich), missing], steps=[{'do': 'click', 'in': 'setup', 'data': {'setup': 'toggle'}},
                                                             {'do': 'choose', 'in': 'setup', 'data': {'setup': 'format'}, 'value': '9v9'},
                                                             {'do': 'click', 'in': 'setup', 'data': {'setup': 'save'}, 'name': 'saved'}],
                  postAnswers={'/api/coaching/settings': {'ok': True, 'settings': None, 'stale': ['measurement', 'role', 'interpretation']}}),
            v2job(steps=[{'do': 'click', 'in': 'setup', 'data': {'setup': 'toggle'}}, {'do': 'choose', 'in': 'setup', 'data': {'setup': 'squad'}, 'value': 'new'},
                         {'do': 'choose', 'in': 'setup', 'data': {'setup': 'squad-name'}, 'value': 'U10 \u00c9toiles', 'event': 'input'},
                         {'do': 'click', 'in': 'setup', 'data': {'setup': 'save'}, 'name': 'refused'}]),
            marks_job([], report=one, answers=[v2_answer(one, coachSpans=True)]))

    def test_the_roles_editor_shows_the_roles_the_answer_measured(self):
        # CODE M3: the settings answer of the page load named A-04 LCB; the report answer after Generate names LB.
        text = self.roles['roles']['setup']['text']
        self.assertIn('Measured: LB (stable, high)', text)
        self.assertNotIn('LCB (stable', text)
        self.assertEqual(sum(1 for u in self.roles['last']['meta'] if u.startswith('/api/coaching/settings')), 1, 'no second settings request')

    def test_the_v1_report_only_for_a_server_without_the_route(self):
        # CODE L2: any failure of the catalogue route fell back to the professional-reference v1 report.
        broken = self.broken['last']
        self.assertEqual(broken['mode'], 'v2')
        self.assertTrue(broken['asked'] and all('context=' not in u for u in broken['asked']), broken['asked'])
        self.assertIn('The coaching setup catalogue could not be read (The adaptive coaching report is not available in this version: broken profiles)',
                      broken['setup']['text'])
        self.assertEqual(self.html404['last']['mode'], 'legacy', 'Flask' + chr(39) + 's HTML 404 page: a server without the v2 routes')
        self.assertTrue(all('context=' in u for u in self.html404['last']['asked']))

    def test_one_sided_marks_give_no_share_and_say_what_is_missing(self):
        # Critic N6 on the AM copy: Team A's two spells only gave "A 1.0, B 0.0" and the defending checks were not
        # assessed with nothing said.
        ov = sec(self.one['first'], 'overview')
        self.assertIn('Who had the ballNo share given: you marked only Team A\u2019s possession (30% of the time); mark Team B\u2019s spells too for a share', ov)
        self.assertNotIn('100%', ov)
        self.assertIn('You marked only Team A\u2019s possession. PitchIQ could not follow the ball on this footage, so the checks played while Team B had the '
                      'ball are not assessed until you mark Team B\u2019s spells too.', ov)
        self.assertIn('not assessed: you marked only Team A\u2019s possession: mark Team B\u2019s spells too to run this check', ov)
        # The page's list empty (spans saved outside it, or not loaded yet): the report itself says one side.
        ov = sec(self.oneReport['first'], 'overview')
        self.assertIn('Who had the ballNo share given: you marked only Team AQs possession (30% of the time)'.replace('Q', chr(8217)), ov)
        self.assertNotIn('100%', ov)

    def test_shares_from_the_coach_marks_alone_are_the_coach_marks(self):
        ov = sec(self.both['first'], 'overview')
        self.assertIn('Who had the ballTeam A about 40%, Team B about 60% (from your marks, 30% of the time)', ov)
        self.assertNotIn('estimated', ov.split('Who had the ball')[1].split('Data quality')[0])
        self.assertNotIn('You marked only', ov)

    def test_the_save_message_follows_the_answer_after_it(self):
        # CODE L3: the message was chosen from the tiers the save made stale, before the answer that followed.
        self.assertIn('Saved. Press Generate to measure this run' + chr(8217) + 's situations with this setup (no new tracking).',
                      self.press['saved']['setup']['text'])

    def test_a_squad_name_the_server_would_refuse_is_said_before_the_save(self):
        out = self.squad['refused']
        self.assertIn('The squad name can use plain letters, digits, spaces and punctuation only (no accents or symbols).', out['setup']['text'])
        self.assertEqual([p for p in out['posted'] if p['url'] in ('/api/coaching/squads', '/api/coaching/settings')], [], 'nothing posted (CODE L11)')


def r3_spacing(rich, mode, k):
    """A level-4 back-line spacing 'moments worth a look' review of rich's review, cited for the gap between defence
    and midfield (mode 'dm_gap': the SFS review of CODE L-4, 12.2 m with one opponent in it) or for a lateral gap."""
    f = json.loads(json.dumps(rich['reviews'][0]))
    key = S.finding_key('defensive_spacing', 'A', 'team', 724.0 + k, 748.0)
    f.update(id=S.record_id('fnd', key, 9), key=key, observation='defensive_spacing', verdict='review', priority=None, roles={},
             limitations=['moments_worth_a_look'], drills=[], suggestedAction=None, timestamps=[],
             measurements={'failureMode': mode, 'lateralGapM': 8.8, 'gapBallSide': False, 'opponentInGap': mode == 'lateral_gap',
                           'dmGapM': 12.2, 'dmGapOpponents': 1, 'lineStdM': 1.4, 'metres': 'estimated',
                           'clearMoments': 7, 'clearSpells': 6, 'decidedMoments': 10, 'failingMoments': 8},
             text={'headline': f'Moments worth a look: back-line spacing ({mode})', 'short': 'Back-line spacing',
                   'explanation': 'An opponent stood in a large gap in our back line, or between it and midfield.',
                   'action': 'Watch whether the back line stayed close to midfield.'})
    return f


@unittest.skipUnless(NODE, 'needs node to run the page scripts')
class V2FixRound3(unittest.TestCase):
    """Fix round 3 on the page (CODE M-1, L-3, L-4, L-5, L-7 of verification round 2): legacy (v1) mode only on HTTP
    404, any other failure of the catalogue stays v2 with the error shown; a strength's action without a second "To
    build on it"; the chips of the cited back-line failure mode; the drills section's empty sentence by case; a role
    confirmation stamped in local time."""

    @classmethod
    def setUpClass(cls):
        rich = v2_rich()
        strength = next(f for f in rich['strengths'] if f['verdict'] == 'strength')
        strength['text']['action'] = 'To build on it, keep one player high and one wide on each side.'
        rich['reviews'] = rich['reviews'] + [r3_spacing(rich, 'dm_gap', 0), r3_spacing(rich, 'lateral_gap', 1)]
        rich['overview']['detectors'][0].update(status='not_assessed', reason='too_few_situations', opportunities=4, eligibleSeconds=226.0)
        work_no_drill = json.loads(json.dumps(rich))
        work_no_drill['drills'] = []
        nothing = json.loads(json.dumps(work_no_drill))
        nothing['needsWork'] = []
        general = json.loads(json.dumps(nothing))
        general['nextObjective'] = {'skill': None, 'text': 'Every player experiences all positions.', 'fromMatch': False, 'finding': None}
        # critic S-3 (COACH F-4 / FACTS F-9): an objective chosen from this match's checks never reads "not from this match"
        nothing['nextObjective'] = {'skill': 'spread_when_attacking', 'text': 'From this match' + chr(39) + 's checks (not a work item): Spread out.',
                                    'fromMatch': False, 'finding': None}
        table = ('The adaptive coaching report is not available: profiles.json does not validate (parameters.coverAngleDeg: 70.0 acts on '
                 'nothing beyond the measurement bound (max 60.0)). Fix or restore the file, then reload the page.')
        role = [{'do': 'click', 'in': 'setup', 'data': {'setup': 'toggle'}}, {'do': 'click', 'in': 'setup', 'data': {'setup': 'roles'}, 'name': 'roles'},
                {'do': 'choose', 'in': 'setup', 'data': {'role': 'A-04'}, 'value': 'LB'}, {'do': 'click', 'in': 'setup', 'data': {'setup': 'save'}, 'name': 'saved'}]
        (cls.html500, cls.json503, cls.cards, cls.workNoDrill, cls.nothing, cls.general, cls.role) = run(
            v2job(profiles={'__error': "Unexpected token '<', \"<!doctype \"... is not valid JSON", '__name': 'SyntaxError', '__status': 500}),
            v2job(profiles={'__error': table, '__status': 503}, answers=[{'__error': table}]),
            v2job(report=rich),
            v2job(report=work_no_drill), v2job(report=nothing), v2job(report=general),
            v2job(steps=role, postAnswers={'/api/coaching/settings': {'ok': True, 'settings': None, 'stale': ['role', 'interpretation']}}))
        cls.local = datetime.datetime.now()
        cls.rich = rich
        cls.reports = {'workNoDrill': work_no_drill, 'nothing': nothing, 'general': general}

    def test_only_a_404_is_a_server_without_the_v2_routes(self):
        # CODE M-1: profiles.json with coverAngleDeg 70 (beyond its bound) answered Flask's HTML 500; the page could not
        # parse it, took the SyntaxError for a missing route and built the v1 professional-reference report.
        for out, said in ((self.html500['last'], 'could not be read (the server answered HTTP 500 without a readable message)'),
                          (self.json503['last'], 'profiles.json does not validate (parameters.coverAngleDeg: 70.0')):
            self.assertEqual(out['mode'], 'v2')
            self.assertIn(said, out['setup']['text'])
            self.assertTrue(out['asked'] and all('context=' not in u for u in out['asked']), 'never the v1 report: ' + str(out['asked']))
        out = self.json503['last']
        self.assertIn('profiles.json does not validate', out['banner']['text'], 'the report answer\u2019s error is shown too')
        self.assertIn('profiles.json does not validate', sec(out, 'empty'))

    def test_a_strength_says_to_build_on_it_once(self):
        strength = next(c for c in self.cards['first']['cards'] if c['verdict'] == 'strength')
        self.assertIn('To build on it. Keep one player high and one wide on each side.', strength['text'])
        self.assertEqual(strength['text'].count('To build on it'), 1, strength['text'])

    def test_the_chips_of_the_cited_back_line_failure_mode(self):
        out = self.cards['first']
        chips = {c['finding']: keys for c, keys in zip(out['cards'], out['chips'])}
        cards = {c['finding']: c['text'] for c in out['cards']}
        dm, lateral = self.rich['reviews'][-2]['id'], self.rich['reviews'][-1]['id']
        self.assertEqual(chips[dm], ['lateralGapM', 'dmGapM', 'dmGapOpponents'], 'level 4: the dm gap and its opponents, no lateral yes/no')
        self.assertIn('Defence to midfield gap 12.2 m (est.)', cards[dm])
        # (pre-freeze, critic N-4: the count is of the ball's side only, as contextual measures it)
        self.assertIn('Opponents between defence and midfield on the ball\u2019s side 1', cards[dm])
        self.assertNotIn('Opponent in the gap', cards[dm])
        self.assertEqual(chips[lateral], ['lateralGapM', 'gapBallSide', 'opponentInGap'], 'the lateral mode as before (dmGapM from level 5)')
        self.assertNotIn('Opponents between defence and midfield', cards[lateral])
        # The counts add up (critic B-3, FACTS F-4): 8 failing of the 10 judged, 7 of them clear.
        self.assertIn('Clear in 7 of the 10 moments PitchIQ could judge (6 separate spells), 1 more fell short less clearly', cards[dm])

    def test_the_objective_origin_never_contradicts_its_text(self):
        prog = sec(self.nothing['first'], 'progression')
        self.assertIn('checks (not a work item): Spread out.', prog)
        self.assertNotIn('not from this match', prog)

    def test_too_few_situations_is_not_a_short_range(self):
        # critic S-2 / COACH F-14: 4 clear advances in 6.1 minutes (226 s of our possession) read "the range is too short".
        ov = sec(self.cards['first'], 'overview')
        self.assertIn('not assessed: only 4 situations of this kind happened in this range, too few to judge', ov)
        self.assertNotIn('the range is too short', ov)

    def test_the_empty_drills_section_says_why(self):
        say = {'workNoDrill': 'No drill: PitchIQ has none that fits the work items in this range for this team\u2019s age, level and format.',
               'nothing': 'No drill: there is no work item in this range, and PitchIQ has none that fits the next objective for this team\u2019s age, '
                          'level, format and range.',
               'general': 'No drill: there is no work item in this range, and the next objective is a general one with no drill tied to it.'}
        for name, text in say.items():
            drills = sec(getattr(self, name)['first'], 'drills')
            self.assertIn(text, drills, name)
            self.assertNotIn('never at random', drills)
            self.assertEqual(pdf.no_drill_text(self.reports[name]), text.replace('\u2019', "'"), 'the PDF says the same: ' + name)

    def test_a_role_confirmation_is_stamped_in_local_time(self):
        self.assertIn('LB', self.role['roles']['setup']['roles']['A-04']['options'])
        posted = [p for p in self.role['saved']['posted'] if p['url'] == '/api/coaching/settings']
        self.assertEqual(len(posted), 1)
        stamp = posted[0]['body']['settings']['roles']['A-04'][0]['set']
        self.assertRegex(stamp, r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}$')
        made = datetime.datetime.strptime(stamp, '%Y-%m-%dT%H:%M:%S')
        self.assertLess(abs((made - self.local).total_seconds()), 600, f'local time ({self.local:%H:%M}), not UTC: {stamp}')


@unittest.skipUnless(NODE, 'needs node to run the page scripts')
class V2Drawing(unittest.TestCase):
    """The new annotation elements (DESIGN 5.5): colours away from the team colours, tier gating (5.4), the element
    budget, the follow layer of the replay (5.6)."""

    @classmethod
    def setUpClass(cls):
        rich = v2_rich()
        ev = rich['keyMoments'][1]
        rv = next(m for m in rich['keyMoments'] if m['tier'] == 'image')
        pinned = dict(ev['videoOverlay'], tier='pinned')
        crowded = json.loads(json.dumps(ev['diagram']))
        crowded['elements'] += [{'type': 'focus', 'id': f'A-0{k}', 'x': .5, 'y': .1*k, 'tone': 'review', 'label': True} for k in range(6, 9)]
        state = [{'id': 'A-04', 'x': .22, 'y': .31, 'team': 'A'}, {'id': 'A-05', 'x': .18, 'y': .45, 'team': 'A'}, {'id': 'B-10', 'x': .21, 'y': .29, 'team': 'B'}]
        people = {'A-04': [.40, .50, .02, .08], 'A-05': [.30, .52, .02, .08]}
        cls.rv_diagram = rv['diagram']
        cls.out = run({'kind': 'paint2', 'diagram': ev['diagram'], 'overlay': ev['videoOverlay'], 'state': state, 'follow': {}, 'labels': {'A-04': 'A-03'}},
                      {'kind': 'paint2', 'overlay': pinned}, {'kind': 'paint2', 'overlay': rv['videoOverlay']}, {'kind': 'paint2', 'diagram': crowded},
                      {'kind': 'paint2', 'diagram': rv['diagram']},
                      {'kind': 'paint2', 'follow': {'focus': [{'id': 'A-04', 'tone': 'review', 'secondary': False}], 'support': [{'id': 'A-05', 'team': 'A'}],
                                                    'relations': [{'a': 'A-04', 'b': 'A-05', 'kind': 'cover', 'ok': False}]},
                       'people': people, 'state': state, 'ev': ev},
                      {'kind': 'paint2', 'follow': {'focus': [{'id': 'A-04', 'tone': 'review', 'secondary': False}], 'support': [{'id': 'A-05', 'team': 'A'}],
                                                    'relations': [{'a': 'A-04', 'b': 'A-05', 'kind': 'cover', 'ok': False}]}, 'people': people, 'tier': 'image'},
                      {'kind': 'paint2', 'diagram': ev['diagram'], 'colours': {'A': '#ff0000', 'B': '#00ff00'}},
                      {'kind': 'paint2', 'diagram': ev['diagram'], 'colours': {'A': '#ff3c9f', 'B': '#ffffff'}})
        (cls.world, cls.pinned, cls.image, cls.crowded, cls.channel, cls.follow, cls.follow_image, cls.adversarial, cls.fallback) = cls.out

    def test_the_diagram_draws_every_new_element(self):
        d = self.world['diagram']
        zone = [s for s in d['shapes'] if s['tag'] == 'POLYGON' and s['stroke'] == '#7dffb0']
        self.assertEqual(len(zone), 1, 'one dotted area in the zone colour')
        self.assertTrue(zone[0]['dash'].startswith('0.12'), 'dotted: dots of 0.35 x the line width')
        self.assertGreater(len([s for s in d['shapes'] if s['tag'] == 'POLYLINE' and s['stroke'] == '#7dffb0' and s['opacity'] == '0.45']), 0, 'a light hatch')
        rel = [s for s in d['shapes'] if s['stroke'] == '#ffffff' and s['opacity'] == '0.45']
        self.assertEqual(len(rel), 1)
        self.assertIsNotNone(rel[0]['dash'], 'a missing relation is dashed')
        self.assertTrue(any(s['tag'] == 'CIRCLE' and s['stroke'] == '#ff4d6d' for s in d['shapes']), 'the reviewed player is ringed in the review colour')
        arrow = [s for s in d['shapes'] if s['tag'] == 'POLYLINE' and s['stroke'] == '#7dffb0' and s['dash'] and s['opacity'] is None]
        self.assertEqual(len(arrow), 1, 'one dashed movement arrow')
        texts = [t['text'] for t in d['texts']]
        for text in ('A-03', 'A-05', 'B-10', 'possible covering area', '14 m', 'Pressure on the ball; nobody covered behind it.'):
            self.assertIn(text, texts)
        caption = next(t for t in d['texts'] if t['text'].startswith('Pressure on the ball'))
        self.assertEqual(caption['anchor'], 'start')
        self.assertAlmostEqual(caption['x'], 4.5+3.1*.6, places=2, msg='the caption at the top left of the pitch')
        faint = [s for s in d['shapes'] if s['tag'] == 'CIRCLE' and s['fill'] == '#f6b564']
        self.assertEqual(faint[0]['fillOpacity'], '0.5', 'a player the annotation is not about is faint')

    def test_tier_gating_on_the_video(self):
        world = self.world['overlay']['shapes']
        self.assertTrue(any(s['tag'] == 'POLYGON' and s['stroke'] == '#7dffb0' for s in world), 'world: the area over the video')
        self.assertTrue(any(s['tag'] == 'ELLIPSE' and s['stroke'] == '#ff4d6d' for s in world), 'world: an outline at the feet')
        pinned = self.pinned['overlay']['shapes']
        self.assertFalse(any(s['stroke'] == '#7dffb0' for s in pinned), 'pinned: no area, no arrow')
        self.assertTrue(any(s['tag'] == 'ELLIPSE' and s['stroke'] == '#ff4d6d' for s in pinned))
        image = self.image['overlay']['shapes']
        self.assertEqual([s['tag'] for s in image if s['stroke'] == '#ff4d6d'], ['RECT'], 'image: the box is outlined, nothing on the pitch')
        self.assertFalse(any(s['opacity'] == '0.45' and s['stroke'] == '#ffffff' and s['tag'] == 'POLYLINE' for s in image), 'image: no relation line')
        self.assertTrue(any(c[0] == 'ellipse' for c in self.world['video']), 'the page canvas draws the same overlay')

    def test_the_caption_stands_above_the_pitch_and_labels_stay_inside(self):
        d = self.channel['diagram']
        self.assertEqual(d['viewBox'], '0 -4.6 114 85.6', 'a strip above the pitch holds the caption')
        caption = next(t for t in d['texts'] if t['text'].startswith('Left defender advanced'))
        self.assertLess(caption['y'], 0, 'the caption is in the strip, not over the players')
        label = next(t for t in d['texts'] if t['text'] == 'possible covering area')
        width = len(label['text'])*3.1*.66*.52
        x0 = label['x']-(width if label['anchor'] == 'end' else width/2 if label['anchor'] == 'middle' else 0)
        self.assertGreaterEqual(x0, 0, 'the area label stays inside the drawing')
        self.assertLessEqual(x0+width, 114)
        self.assertEqual(self.out[0]['diagram']['viewBox'], '0 -4.6 114 85.6')

    def test_the_element_budget(self):
        rings = [s for s in self.crowded['diagram']['shapes'] if s['tag'] == 'CIRCLE' and s['stroke'] == '#ff4d6d']
        self.assertEqual(len(rings), 2, 'at most two focus elements are drawn (DESIGN 5.5)')

    def test_the_channel_is_drawn_faint_in_the_review_colour(self):
        ch = [s for s in self.channel['diagram']['shapes'] if s['tag'] == 'POLYGON' and s['fill'] == '#ff4d6d']
        self.assertEqual(len(ch), 1)
        self.assertEqual(ch[0]['fillOpacity'], '0.12')

    def test_adversarial_team_colours_change_the_palette(self):
        d = self.adversarial['diagram']['shapes']
        self.assertFalse(any(s['stroke'] == '#ff4d6d' for s in d), 'rose is too close to a red team')
        self.assertTrue(any(s['tag'] == 'CIRCLE' and s['stroke'] == '#ff2fd1' for s in d), 'magenta instead')
        self.assertTrue(any(s['tag'] == 'POLYGON' and s['stroke'] == '#5cf2ff' for s in d), 'aqua: the colour farthest from a green team')

    def test_a_white_fallback_outline_has_a_dashed_dark_halo(self):
        d = self.fallback['diagram']['shapes']
        rings = [s for s in d if s['tag'] == 'CIRCLE' and s['stroke'] == '#ffffff' and s['fill'] == 'none' and s['opacity'] is None]
        halos = [s for s in d if s['tag'] == 'CIRCLE' and s['stroke'] == '#07131bd9' and s['dash']]
        self.assertTrue(rings and halos, 'the focus ring is white over a dashed dark halo')

    def test_the_follow_layer(self):
        fo = self.follow['followOf']
        self.assertEqual(fo['relations'], [{'a': 'A-04', 'b': 'A-05', 'kind': 'cover', 'ok': False}])
        video = self.follow['followVideo']
        ellipses = [c for c in video if c[0] == 'ellipse']
        self.assertEqual([(c[1], c[2]) for c in ellipses], [(310.0, 300.0), (310.0, 300.0), (410.0, 290.0), (410.0, 290.0)],
                         'support then focus, at the feet of the live boxes (1000 x 500)')
        self.assertTrue(any(c[0] == 'lineTo' and (c[1], c[2]) == (310.0, 300.0) for c in video), 'the relation joins the feet')
        self.assertFalse(any(c[0] == 'fillText' and c[1] == '14 m' for c in video), 'no stale value while following')
        image = self.follow_image['followVideo']
        self.assertTrue(any(c[0] == 'rect' for c in image))
        self.assertFalse(any(c[0] == 'lineTo' for c in image), 'no relation line at the image tier')
        board = self.follow['followBoard']
        self.assertGreaterEqual(len([c for c in board if c[0] == 'arc']), 2, 'rings around the board markers')


@unittest.skipUnless(NODE, 'needs node to run the page scripts')
class V2Palette(unittest.TestCase):
    """CIEDE2000, the annotation palette, the setup catalogue against schema.py, the clock."""

    @classmethod
    def setUpClass(cls):
        pairs = [('#b98cff', '#f6b564'), ('#ff0000', '#00ff00'), ('#e53935', '#43a047'), ('#ffffff', '#ff00ff'), ('#ff4d6d', '#7dffb0'), ('#7dffb0', '#5cf2ff'),
                 ('#c6ff4d', '#ff2fd1'), ('#ff3c9f', '#ffffff')]
        defaults = [(a, f) for a in S.AGE_GROUPS for f in S.FEDERATIONS]
        slots = [({'players': 7, 'goalkeeper': True}, '1-3-2-1'), ({'players': 9, 'goalkeeper': True}, '1-3-2-3'), ({'players': 4, 'goalkeeper': False}, '1-2-1'),
                 ({'players': 7, 'goalkeeper': True}, None), ({'players': 4, 'goalkeeper': True}, '1-2-1'), ({'players': 9, 'goalkeeper': True}, '1-4-3-1')]
        cls.slots = slots
        cls.pairs, cls.defaults = pairs, defaults
        cls.out = run({'kind': 'palette', 'sharma': [[a, b] for a, b, _ in SHARMA], 'pairs': pairs, 'dE': [['#ff4d6d', '#ff5fa2'], ['#c6ff4d', '#c6ff3d']],
                       'contrast': [['#ffffff', '#000000'], ['#7dffb0', '#2b8a46']], 'clock': [59.999, 0, 61.005, 3599.995, -2],
                       'defaults': defaults, 'slots': slots})[0]

    def test_ciede2000_reproduces_the_published_pairs(self):
        for (a, b, want), got in zip(SHARMA, self.out['sharma']):
            self.assertAlmostEqual(got, want, places=4, msg=f'{a} {b}')

    def test_the_palette_rule(self):
        p = self.out['palettes']
        self.assertEqual((p[0]['review'], p[0]['zone'], p[0]['reviewFallback'], p[0]['zoneFallback']), ('#ff4d6d', '#7dffb0', False, False),
                         'defaults: rose and mint (lime is the ball marker)')
        self.assertEqual((p[1]['review'], p[1]['zone'], p[1]['zoneFallback']), ('#ff2fd1', '#5cf2ff', True), 'red and green teams')
        self.assertEqual(p[3]['review'], '#ff4d6d')
        self.assertEqual((p[7]['review'], p[7]['reviewFallback']), ('#ffffff', True), 'a pink team and a white team leave no candidate 20 away: white with a dashed halo')
        self.assertLess(self.out['dE'][1], 10, 'lime is the ball marker\u2019s colour')
        for (a, b), pal in zip(self.pairs, p):
            if not pal['reviewFallback']:
                self.assertGreaterEqual(pal['scores']['review'], 20, (a, b))
            if not pal['zoneFallback']:
                self.assertGreaterEqual(pal['scores']['zoneContrast'], 3, (a, b))
                self.assertGreaterEqual(pal['scores']['zoneTeam'], 20, (a, b))
        self.assertAlmostEqual(self.out['contrast'][0], 21.0, places=6)

    def test_the_catalogue_matches_the_schema(self):
        c = self.out['catalogue']
        self.assertEqual(c['ages'], list(S.AGE_GROUPS))
        self.assertEqual([lv['name'] for lv in c['levels']], [S.LEVEL_NAMES[k] for k in range(1, 8)])
        self.assertEqual({int(k): v for k, v in c['formations'].items()}, {k: list(v) for k, v in S.FORMATIONS.items()})
        self.assertEqual([f['name'] for f in c['formats']], list(S.FORMAT_NAMES))
        self.assertEqual([s['id'] for s in c['styles']], ['unknown', 'possession', 'counter', 'high_press', 'compact'])
        self.assertEqual(set(s['id'] for s in c['styles']), set(S.STYLES))
        self.assertEqual([f['id'] for f in c['federations']], list(S.FEDERATIONS))
        self.assertEqual(set(c['roles']), set(S.TEMPLATE_ROLES))
        self.assertEqual({k: v for k, v in self.out['tiers'].items()}, {k: list(v) for k, v in S.TIER_ELEMENTS.items()})
        self.assertEqual(self.out['budget'], S.ELEMENT_BUDGET)

    def test_format_defaults_by_age(self):
        got = {(a, f): d for (a, f), d in zip(self.defaults, self.out['defaults'])}
        want = {'U6': 4, 'U7': 4, 'U8': 4, 'U9': 7, 'U10': 7, 'U11': 9, 'U12': 9, 'U13': 11, 'U14': 11, 'U18': 11, 'Adult': 11}
        for age, players in want.items():
            self.assertEqual(got[(age, 'auto')]['players'], players, age)
            self.assertEqual(got[(age, 'US')], got[(age, 'auto')])
        self.assertEqual((got[('U11', 'England')]['players'], got[('U11', 'Canada')]['players'], got[('U13', 'England')]['players']), (7, 7, 9))
        self.assertFalse(got[('U8', 'auto')]['goalkeeper'], 'U.S. 4v4 has no goalkeeper')
        for (age, fed), d in got.items():
            block = saved_block(ageGroup=age, format=d, formation={'intended': None, 'periods': []}, roles={})
            self.assertEqual(S.validate('team_settings', block), [], (age, fed))
        self.assertEqual(self.out['typical'][:len(S.FEDERATIONS)], [1]*len(S.FEDERATIONS))
        if PROFILES is not None:
            for (age, fed), d in got.items():
                self.assertEqual(d, PROFILES.default_format(age, fed), (age, fed, 'the page copy equals profiles.json'))

    def test_england_and_canada_defaults_follow_the_research(self):
        # SOURCES F1 / critic S-13. Research L1L2 format[4]: the FA plays 5v5 at U7/U8 (FA-PITCH-U78) and 7v7 at U9-U10;
        # format[3]: Canada 3v3 at U6-U7 and 4v4 without a goalkeeper or 5v5 with one at U8-U9 (CS-STANDARDS-2026); L2
        # format[4]: Canada 7v7 at U10-U11; L3L4: England 7v7 at U11 and 9v9 at U12-U13, Canada 9v9 at U12-U13.
        got = {(a, f): d for (a, f), d in zip(self.defaults, self.out['defaults'])}
        research = {'England': {'U7': 5, 'U8': 5, 'U9': 7, 'U10': 7, 'U11': 7, 'U12': 9, 'U13': 9, 'U14': 11},
                    'Canada': {'U8': (4, 5), 'U9': (4, 5), 'U10': 7, 'U11': 7, 'U12': 9, 'U13': 9}}
        for fed, rows in research.items():
            for age, players in rows.items():
                self.assertIn(got[(age, fed)]['players'], players if isinstance(players, tuple) else (players,), (fed, age))
                if PROFILES is not None:
                    self.assertIsNone(PROFILES.default_format_basis(age, fed), (fed, age, 'the federation' + chr(39) + 's own format'))
        self.assertEqual((got[('U8', 'England')], got[('U9', 'Canada')]), ({'players': 5, 'goalkeeper': True},)*2)
        if PROFILES is not None:
            for age in ('U6', 'U7'):
                self.assertIn('Canada plays 3v3', PROFILES.default_format_basis(age, 'Canada'), 'Canada' + chr(39) + 's 3v3: the nearest PitchIQ format')
            notice = lambda block: [n['text'] for n in PROFILES.resolve(block)['notices'] if n['code'] == 'format_above_default']
            self.assertEqual(notice({'ageGroup': 'U8', 'playingLevel': 2, 'federation': 'England', 'format': {'players': 5, 'goalkeeper': True}}), [],
                             'England' + chr(39) + 's own 5v5 is not called above England' + chr(39) + 's format')
            self.assertEqual(notice({'ageGroup': 'U9', 'playingLevel': 2, 'federation': 'Canada', 'format': {'players': 4, 'goalkeeper': False}}), [])
            above = notice({'ageGroup': 'U6', 'playingLevel': 1, 'federation': 'Canada', 'format': {'players': 7, 'goalkeeper': True}})[0]
            self.assertTrue(above.startswith("PitchIQ's default for U6 with Canada Soccer selected is 4v4"), above)
            self.assertNotIn('The Canada Soccer format', above)

    def test_relative_slots(self):
        s = self.out['slots']
        self.assertEqual(s[0], ['back-1/3', 'back-2/3', 'back-3/3', 'mid-1/2', 'mid-2/2', 'front-1/1'])
        self.assertEqual(len(s[1]), 8)
        self.assertEqual(s[2], ['back-1/1', 'mid-1/2', 'mid-2/2', 'front-1/1'])
        self.assertEqual(s[3], s[0], 'no formation: the generic blocks of 6 outfield players')
        self.assertEqual(s[4], ['back-1/1', 'mid-1/1', 'front-1/1'], '4v4 with a keeper: 3 outfield players')
        self.assertEqual(s[5], ['back-1/4', 'back-2/4', 'back-3/4', 'back-4/4', 'mid-1/3', 'mid-2/3', 'mid-3/3', 'front-1/1'])
        for slot in sum(s, []):
            self.assertRegex(slot, '^'+S.RELATIVE_ROLE+'$')
        if PROFILES is not None:
            for (fmt, formation), got in zip(self.slots, s):
                self.assertEqual(got, [r for r in PROFILES.roles_for(fmt, formation) if r != 'GK'], (fmt, formation))

    def test_the_clock_rounds_before_splitting_minutes(self):
        self.assertEqual(self.out['clock'], ['01:00.00', '00:00.00', '01:01.01', '60:00.00', '00:00.00'])


# ---------- wave 7: the coach's possession marks, real reports, the progression detail ----------
SPANS_EMPTY = {'run': 'run1', 'spans': {'version': 1, 'kind': 'coach-possession-spans', 'source': 'coach', 'spans': []},
               'segment': {'t0': 720.0, 't1': 899.0}, 'saved': None}


def spans_answer(spans, saved='2026-10-22T10:00:00'):
    a = json.loads(json.dumps(SPANS_EMPTY))
    a['spans']['spans'] = spans
    a['saved'] = saved
    return a


def phase_off_report():
    """The rich mock on footage where possession-4 abstains and nobody marked possession: the notice, and the
    phase-dependent checks not assessed for phase_unknown."""
    r = v2_rich()
    ov = r['overview']
    ov['possession'] = {'available': False, 'quality': 'unavailable', 'reason': 'The ball and who had it could not be followed reliably on this footage.',
                        'roughShares': None, 'coverage': None, 'caveat': None}
    ov['notices'] = [{'code': 'phase_unavailable', 'text': 'Who had the ball could not be followed on this footage, so the checks that need it were not made.'}]
    ov['detectors'][0].update(status='not_assessed', reason='phase_unknown', eligibleSeconds=0.0, opportunities=0)
    return r


def marks_job(steps, spans=None, **kw):
    return v2job(steps=steps, spans=SPANS_EMPTY if spans is None else spans, now=0, **kw)


@unittest.skipUnless(NODE, 'needs node to run the page scripts')
class V2Marks(unittest.TestCase):
    """The coach's possession marks (user decision 1, Oct 21): "our ball / their ball / dead ball from here", "end",
    a list to edit or remove them, saved per run as a schema 'coach_spans' record with team letters, drawn on a lane of
    their own under the possession strip, and the report's way to them where the phase could not be followed."""

    @classmethod
    def setUpClass(cls):
        mark = lambda kind: {'do': 'click', 'in': 'bar', 'data': {'mark': kind}}
        at = lambda t: {'do': 'time', 'value': t}
        cls.flow = run(marks_job([
            {'do': 'method', 'name': 'openMarks'}, {'do': 'tick', 'value': 0, 'name': 'opened'},
            {'do': 'click', 'in': 'toggle', 'data': {'marks': 'toggle'}, 'name': 'closed'},
            {'do': 'method', 'name': 'openMarks'}, {'do': 'tick', 'value': 0, 'name': 'open'},
            at(725.0), dict(mark('own'), name='own1'),
            at(731.2), dict(mark('opp'), name='second'),
            at(735.555), dict(mark('end'), name='ended'),
            at(728.0), mark('dead'), at(729.0), dict(mark('end'), name='over'),
            {'do': 'click', 'in': 'bar', 'data': {'marks': 'list'}, 'name': 'list'},
            {'do': 'choose', 'in': 'bar', 'within': {'span': '3'}, 'data': {'field': 't1'}, 'value': '12:20.00', 'name': 'edited'},
            {'do': 'click', 'in': 'bar', 'within': {'span': '1'}, 'data': {'marks': 'remove'}, 'name': 'removed'},
            {'do': 'click', 'in': 'bar', 'within': {'span': '0'}, 'data': {'marks': 'go'}},
            {'do': 'click', 'in': 'bar', 'data': {'marks': 'remove-all'}, 'name': 'ask'},
            {'do': 'click', 'in': 'bar', 'data': {'marks': 'remove-all'}, 'name': 'cleared'}]))[0]
        cls.saved = spans_answer([{'t0': 722.0, 't1': 740.0, 'state': 'A'}, {'t0': 740.0, 't1': 752.5, 'state': 'dead'}, {'t0': 760.0, 't1': 790.0, 'state': 'B'}])
        (cls.missing, cls.outside, cls.team_b, cls.failed, cls.lane, cls.phase_off, cls.after_save) = run(
            v2job(steps=[{'do': 'method', 'name': 'openMarks'}]),
            marks_job([{'do': 'method', 'name': 'openMarks'}, at(700.0), dict(mark('own'), name='early'), at(910.0), dict(mark('dead'), name='late'),
                       at(730.0), mark('own'), at(725.0), dict(mark('end'), name='back'), at(740.0), mark('opp'), at(740.1), dict(mark('end'), name='short')]),
            marks_job([{'do': 'method', 'name': 'openMarks'}, {'do': 'choose', 'in': 'bar', 'data': {'marks': 'team'}, 'value': 'B', 'name': 'us'},
                       at(725.0), mark('own'), at(727.0), dict(mark('end'), name='ended')]),
            marks_job([{'do': 'method', 'name': 'openMarks'}, at(725.0), mark('own'), at(727.0), dict(mark('end'), name='failed'),
                       {'do': 'click', 'in': 'bar', 'data': {'marks': 'retry'}, 'name': 'retried'}], postErrorOnce={'/api/coaching/spans': 'disk full'}),
            marks_job([{'do': 'tick', 'value': 749.5, 'name': 'ticked'}, {'do': 'laneAt', 'x': 600, 'name': 'clicked'},
                       {'do': 'click', 'in': 'setup', 'data': {'setup': 'toggle'}, 'name': 'setup'}, {'do': 'click', 'in': 'toggle', 'data': {'marks': 'toggle'}, 'name': 'open'},
                       at(770.0), mark('dead'), {'do': 'tick', 'value': 780.0, 'name': 'marking'}], spans=cls.saved),
            marks_job([{'do': 'click', 'in': 'main', 'data': {'marks': 'open'}, 'name': 'opened'}], report=phase_off_report()),
            marks_job([{'do': 'method', 'name': 'openMarks'}, at(725.0), mark('own'), at(727.0), dict(mark('end'), name='saved')],
                      answers=[v2_answer(v2_rich()), v2_answer(None, state='missing', tiers={'measurement': 'missing', 'role': 'missing', 'interpretation': 'missing'})]))

    def test_marking_while_watching(self):
        f = self.flow
        self.assertTrue(f['first']['marks']['bar']['off'], 'closed until the coach opens it')
        # The toggle in the tracking and 2D controls rows shows once the marks are used (critic S-11: always there, it
        # wrapped those rows onto another line at 1440 and 800 px); the report's buttons open the bar before that.
        self.assertEqual([t['hidden'] for t in f['first']['marks']['toggles']], [True, True], 'no marks, bar closed: no toggle')
        self.assertEqual([(t['hidden'], t['text']) for t in f['opened']['marks']['toggles']], [(False, 'Hide possession marks')]*2,
                         'a button in each view\u2019s controls while the bar is open')
        self.assertTrue(f['closed']['marks']['bar']['off'])
        self.assertEqual([t['hidden'] for t in f['closed']['marks']['toggles']], [True, True], 'closed again with nothing marked')
        self.assertEqual([t['text'] for t in f['ended']['marks']['toggles']], ['Hide possession marks']*2)
        self.assertFalse(f['open']['marks']['bar']['off'])
        self.assertIn('Who has the ball', f['open']['marks']['bar']['text'])
        self.assertIn('Team A is us', f['open']['marks']['bar']['text'])
        b = {x['mark']: x for x in f['open']['marks']['bar']['buttons']}
        self.assertEqual([b[k]['text'] for k in ('own', 'opp', 'dead', 'end')], ['Our ball from here', 'Their ball from here', 'Dead ball from here', 'End'])
        self.assertTrue(b['end']['disabled'], 'nothing to end yet')
        self.assertIn('Our ball (Team A) from 12:05.00', f['own1']['marks']['bar']['text'])
        self.assertTrue(next(x for x in f['own1']['marks']['bar']['buttons'] if x['mark'] == 'own')['on'])
        self.assertEqual(f['own1']['posted'], [], 'an open phase is not saved until it ends')
        posts = [p['body'] for p in f['last']['posted'] if p['url'] == '/api/coaching/spans']
        self.assertEqual(posts[0]['spans']['spans'], [{'t0': 725.0, 't1': 731.2, 'state': 'A'}], 'the next phase closes the open one')
        self.assertEqual(posts[1]['spans']['spans'], [{'t0': 725.0, 't1': 731.2, 'state': 'A'}, {'t0': 731.2, 't1': 735.56, 'state': 'B'}],
                         'End closes it at the video time, to the hundredth')
        self.assertEqual(posts[2]['spans']['spans'], [{'t0': 725.0, 't1': 728.0, 'state': 'A'}, {'t0': 728.0, 't1': 729.0, 'state': 'dead'},
                                                      {'t0': 729.0, 't1': 731.2, 'state': 'A'}, {'t0': 731.2, 't1': 735.56, 'state': 'B'}],
                         'marking over an earlier phase wins there and splits it')
        for body in posts:
            self.assertEqual(body['run'], 'run1')
            self.assertEqual(S.validate('coach_spans', body['spans']), [], body['spans'])
        self.assertIn('Saved with this video', f['ended']['marks']['bar']['text'])

    def test_the_list_edits_and_removes(self):
        f = self.flow
        rows = f['list']['marks']['bar']['rows']
        self.assertEqual(len(rows), 4)
        self.assertIn('12:05.00|12:08.00', rows[0])
        self.assertIn('Ours 5.2 s \u00b7 theirs 4.4 s \u00b7 dead ball 1.0 s', f['list']['marks']['bar']['text'])
        posts = [p['body']['spans']['spans'] for p in f['last']['posted'] if p['url'] == '/api/coaching/spans']
        self.assertEqual(posts[3][-1], {'t0': 731.2, 't1': 740.0, 'state': 'B'}, 'a typed end time (mm:ss.ss, source-video time)')
        self.assertEqual(posts[4], [{'t0': 725.0, 't1': 728.0, 'state': 'A'}, {'t0': 729.0, 't1': 731.2, 'state': 'A'}, {'t0': 731.2, 't1': 740.0, 'state': 'B'}])
        self.assertEqual(f['last']['marks']['seeks'], [725.0], 'Go shows the phase on the video')
        ask = next(x for x in f['ask']['marks']['bar']['buttons'] if x['mark'] == 'remove-all')
        self.assertEqual(ask['text'], 'Remove all 3? Click again')
        self.assertEqual(posts[5], [], 'the second click removes every phase')
        self.assertEqual(len(posts), 6)

    def test_no_route_no_tool(self):
        out = self.missing['last']
        self.assertTrue(all(t['hidden'] for t in out['marks']['toggles']), 'a server without the spans route: no marking button')
        self.assertTrue(out['marks']['bar']['off'])
        self.assertTrue(out['marks']['lane']['hidden'])
        self.assertEqual(out['marks']['shown'], [], 'openMarks does nothing there')
        self.assertNotIn('Mark who had the ball', ' '.join(s['text'] for s in out['sections']))

    def test_outside_the_segment_nothing_is_marked(self):
        out = self.outside
        self.assertIn('The video is outside the analysed segment (12:00.00\u201314:59.00)', out['early']['marks']['bar']['text'])
        self.assertIn('outside the analysed segment', out['late']['marks']['bar']['text'])
        self.assertIn('The video is before the start of that phase (12:10.00), so it was not kept.', out['back']['marks']['bar']['text'])
        self.assertIn('That phase was too short to keep.', out['short']['marks']['bar']['text'])
        self.assertEqual(out['last']['posted'], [], 'nothing is saved from a phase ended before its start or after 0.1 s')
        self.assertEqual(out['last']['marks']['spans'], [])

    def test_our_team_follows_the_report_team(self):
        out = self.team_b
        self.assertIn('/api/coaching?run=run1&team=B', out['us']['asked'], 'choosing "Team B is us" switches the report to team B')
        posts = [p['body']['spans']['spans'] for p in out['last']['posted'] if p['url'] == '/api/coaching/spans']
        self.assertEqual(posts, [[{'t0': 725.0, 't1': 727.0, 'state': 'B'}]], '"our ball" is stored as the team letter')

    def test_a_failed_save_keeps_the_marks(self):
        out = self.failed
        self.assertIn('Not saved: disk full. Your marks stay on this page.', out['failed']['marks']['bar']['text'])
        self.assertIn('retry', [b['mark'] for b in out['failed']['marks']['bar']['buttons']])
        self.assertEqual(out['failed']['marks']['spans'], [{'t0': 725.0, 't1': 727.0, 'state': 'A'}])
        posts = [p for p in out['last']['posted'] if p['url'] == '/api/coaching/spans']
        self.assertEqual(len(posts), 2)
        self.assertEqual(posts[0]['body'], posts[1]['body'], 'Save again sends the same list')
        self.assertIn('Saved with this video', out['retried']['marks']['bar']['text'])

    def test_the_lane_under_the_possession_strip(self):
        out = self.lane
        lane = out['first']['marks']['lane']
        self.assertFalse(lane['hidden'])
        self.assertIn('pz-coach-on', out['first']['marks']['host'], 'the board reserves the lane\u2019s height')
        spans = [s for s in lane['spans'] if 'pz-cspan' in s['cls']]
        self.assertEqual([(s['state'], s['background']) for s in spans], [('A', '#b98cff'), ('dead', '#050807'), ('B', '#f6b564')])
        span = 899.0+1/30-720.0
        self.assertAlmostEqual(float(spans[0]['left'].rstrip('%')), 100*2.0/span, places=2)
        self.assertAlmostEqual(float(spans[0]['width'].rstrip('%')), 100*18.0/span, places=2)
        self.assertIn('s-own', spans[0]['cls'])
        self.assertIn('s-opp', spans[2]['cls'])
        self.assertIn('Our ball (Team A) 12:02.00\u201312:20.00, marked by the coach', spans[0]['title'])
        self.assertIn('Coach', lane['note'])
        self.assertIn('as you marked it', lane['note'])
        self.assertIn('Team A (us)', lane['note'])
        self.assertAlmostEqual(float(out['ticked']['marks']['lane']['now'].rstrip('%')), 100*29.5/span, places=2, msg='the playhead follows the video')
        self.assertEqual(out['ticked']['marks']['seeks'], [])
        self.assertEqual(len(out['clicked']['marks']['seeks']), 1, 'a click on the lane seeks')
        self.assertAlmostEqual(out['clicked']['marks']['seeks'][0], 720.0+.5*span, places=3)
        marking = [s for s in out['marking']['marks']['lane']['spans'] if 'open' in s['cls']][0]
        self.assertAlmostEqual(float(marking['left'].rstrip('%')), 100*50.0/span, places=2)
        self.assertAlmostEqual(float(marking['width'].rstrip('%')), 100*10.0/span, places=2, msg='the open phase grows with the video')
        self.assertIn('Who had the ball: 3 phases marked by you for this run', out['setup']['setup']['text'])

    def test_the_report_leads_to_the_marks_where_the_phase_is_unknown(self):
        out = self.phase_off
        ov = sec(out['first'], 'overview')
        self.assertIn('You can mark who had the ball yourself while you watch', ov)
        self.assertIn('not assessed: the ball and who had it could not be followed reliably on this footage (marking who had the ball under the video would unlock it)', ov)
        self.assertEqual(out['opened']['marks']['shown'], ['tracking'], 'the button opens the video')
        self.assertFalse(out['opened']['marks']['bar']['off'], 'with the marking bar open')

    def test_one_save_at_a_time(self):
        # A phase closed while the previous save is still on its way is saved right after it, with the whole list.
        mark = lambda kind: {'do': 'click', 'in': 'bar', 'data': {'mark': kind}}
        out = run(marks_job([{'do': 'method', 'name': 'openMarks'}, {'do': 'time', 'value': 725.0}, mark('own'),
                             {'do': 'time', 'value': 727.0}, dict(mark('opp'), name='first'), {'do': 'time', 'value': 729.0}, dict(mark('end'), name='second'),
                             {'do': 'release', 'name': 'released'}, {'do': 'release', 'name': 'done'}], holdPosts='/api/coaching/spans'))[0]
        self.assertEqual([out[k]['posts'] for k in ('first', 'second', 'released', 'done')], [1, 1, 2, 2], 'the second save waits for the first')
        posts = [p['body']['spans']['spans'] for p in out['last']['posted']]
        self.assertEqual(posts[1], [{'t0': 725.0, 't1': 727.0, 'state': 'A'}, {'t0': 727.0, 't1': 729.0, 'state': 'B'}])
        self.assertIn('Saving', out['second']['marks']['bar']['text'])

    def test_a_saved_mark_asks_for_generate(self):
        out = self.after_save
        self.assertEqual(len(out['saved']['asked']), 2, 'the report is asked again after the save')
        self.assertIn('The possession you marked changed, so this run\u2019s situations are measured again: Generate (no new tracking).', out['saved']['banner']['text'])


@unittest.skipUnless(NODE, 'needs node to run the page scripts')
class V2Spans(unittest.TestCase):
    """cleanSpans (situations.clean_coach_spans in the page) and placeSpan (the newest mark wins)."""

    def test_rules(self):
        script = r"""
const fs = require('fs'), vm = require('vm');
const sandbox = {console, Map, Set, JSON, URLSearchParams, document: {}, setTimeout, clearTimeout}; sandbox.window = sandbox; vm.createContext(sandbox);
vm.runInContext(fs.readFileSync(process.argv[1], 'utf8'), sandbox);
const C = sandbox.Coaching, out = {};
out.clean = C.cleanSpans({spans: [{t0: 10, t1: 20, state: 'A'}, {t0: 15, t1: 25, state: 'B'}, {t0: 9.7, t1: 11, state: 'dead'}, {t0: 30, t1: 29, state: 'A'},
                                  {t0: 40, t1: 50.4, state: 'B'}, {t0: 50.6, t1: 52, state: 'A'}, {t0: 44, t1: 45, state: 'X'}]}, 10, 50);
out.split = C.placeSpan([{t0: 0, t1: 10, state: 'A'}], {t0: 4, t1: 6, state: 'B'});
out.cover = C.placeSpan([{t0: 2, t1: 3, state: 'A'}, {t0: 3, t1: 9, state: 'dead'}], {t0: 1, t1: 8.9, state: 'B'});
out.join = C.placeSpan([{t0: 0, t1: 5, state: 'A'}], {t0: 5, t1: 8, state: 'A'});
out.tiny = C.placeSpan([{t0: 0, t1: 5, state: 'A'}], {t0: 5.05, t1: 5.1, state: 'B'});
out.record = C.spanRecord([{t0: 1.004, t1: 2.006, state: 'A'}]);
process.stdout.write(JSON.stringify(out));
"""
        done = subprocess.run([NODE, '-e', script, FILES['coaching.js']], capture_output=True, text=True, encoding='utf-8', timeout=60)
        self.assertEqual(done.returncode, 0, done.stderr[-2000:])
        out = json.loads(done.stdout)
        self.assertEqual(out['clean'], [{'t0': 10, 't1': 11, 'state': 'dead'}, {'t0': 15, 't1': 25, 'state': 'B'}, {'t0': 40, 't1': 50, 'state': 'B'}],
                         'clipped within 0.5 s of the segment, later overlaps, bad times, unknown states and spans beyond the tolerance dropped')
        self.assertEqual(out['split'], [{'t0': 0, 't1': 4, 'state': 'A'}, {'t0': 4, 't1': 6, 'state': 'B'}, {'t0': 6, 't1': 10, 'state': 'A'}])
        self.assertEqual(out['cover'], [{'t0': 1, 't1': 8.9, 'state': 'B'}], 'a 0.1 s remnant is dropped')
        self.assertEqual(out['join'], [{'t0': 0, 't1': 8, 'state': 'A'}])
        self.assertEqual(out['tiny'], [{'t0': 0, 't1': 5, 'state': 'A'}], 'a phase under 0.2 s is not kept')
        self.assertEqual(out['record'], {'version': 1, 'kind': 'coach-possession-spans', 'source': 'coach', 'spans': [{'t0': 1.0, 't1': 2.01, 'state': 'A'}]})
        self.assertEqual(S.validate('coach_spans', out['record']), [])


def progress_detail():
    return {'squad': 'sq-0123456789ab', 'level': 4, 'rules': {'version': 1, 'sha': '0'*16, 'label': 'Unvalidated PitchIQ rules; review with a coach.'},
            'matches': 3, 'shortClips': 1, 'note': 'x', 'separate': {'earlierEngine': [], 'needsRules': []},
            'skills': {'pressure_and_cover': {'skill': 'pressure_and_cover', 'state': 'DEVELOPING', 'reason': 'lower_bound', 'opportunities': 31, 'demonstrated': 24,
                                              'notDemonstrated': 7, 'unclear': 3, 'share': .774, 'lower': .66, 'upper': .86, 'matches': 3, 'coachMarkedShare': .25,
                                              'history': [{'video': 'v1', 'matchDate': '2026-09-20', 'opportunities': 12, 'demonstrated': 8, 'seconds': 300.0,
                                                           'qualifies': True, 'inWindow': True},
                                                          {'video': 'v2', 'matchDate': '2026-10-04', 'opportunities': 4, 'demonstrated': 4, 'seconds': 60.0,
                                                           'qualifies': False, 'inWindow': True}]}}}


@unittest.skipUnless(NODE, 'needs node to run the page scripts')
class V2Progression(unittest.TestCase):
    """The progression rows open the skill's detail from GET /api/coaching/progress (PROGRESS's mastery())."""

    def test_details(self):
        step = {'do': 'click', 'in': 'main', 'data': {'progress': 'pressure_and_cover'}, 'name': 'open'}
        out, low = run(v2job(steps=[step], progress=progress_detail()), v2job(steps=[step]))
        pr = sec(out['open'], 'progression')
        self.assertIn('/api/coaching/progress?squad=sq-0123456789ab&run=run1&team=A', out['open']['meta'])
        for text in ('Why this state. Developing: read from how often it was shown over the recent matches.',
                     'Moments. Shown in 24 of 31 judged moments over 3 matches; 3 more could not be judged clearly.',
                     '25% of these moments use possession marked by the coach.', 'Match by match', '2026-09-20', '8 of 12 shown', 'too few moments to count'):
            self.assertIn(text, pr)
        self.assertNotIn('likely between', pr, 'no shares at level 4 (register: metres)')
        self.assertIn('The details could not be read', sec(low['open'], 'progression'), 'a server without the route says so')


@unittest.skipUnless(NODE, 'needs node to run the page scripts')
class V2Real(unittest.TestCase):
    """The page on REAL v2 reports (coaching.interpret with every real stage on EVAL's scenes, through TEXT's
    fixtures): the wave-1 page was built on mocks, and the first real reports printed raw measurement keys ("Threat n
    1", "Weak count 0") and repeated every key moment's caption under its title."""

    @classmethod
    def setUpClass(cls):
        import coaching_text_fixtures as TF
        from soccer.coaching import drills as DR
        lib = DR.load_library()
        cases = [('unsupported_advance', 'none_behind', 1), ('unsupported_advance', 'none_behind', 2), ('unsupported_advance', 'none_behind', 4),
                 ('unsupported_advance', 'none_behind', 7), ('amateur_coach_spans', 'spans', 3), ('amateur_coach_spans', 'no_spans', 3),
                 # (fix round 2, ASSESS) a level-1 report with a drill: none_behind's level-1 report has no next objective now
                 # (critic S-8: no check decided enough and not shown), so no optional drill
                 ('unsupported_advance', 'final_third', 1)]
        cls.reports, jobs = [], []
        for scene, variant, level in cases:
            rep = TF.report(scene, variant, level=level)
            built = TF.tiers(scene, variant)[0]
            res = built.result
            slim = {'version': 3, 'video': res['video'], 'match': res['match'], 'frames': [{'time': f['time'], 'frame': f['frame'], 'people': []}
                                                                                           for f in (res['frames'][0], res['frames'][-1])]}
            drills = {p['drill']: DR.get(lib, p['drill']) for p in rep['drills']}
            first = rep['keyMoments'][0]['id'] if rep['keyMoments'] else None
            steps = [{'do': 'click', 'in': 'main', 'tag': 'BUTTON', 'data': {'evidence': first}, 'name': 'jump'}] if first else []
            cls.reports.append((scene, variant, level, rep))
            jobs.append(v2job(report=rep, answers=[v2_answer(rep, drills=drills)], result=slim, info={'video': res['video'], 'videoKey': '0'*64},
                              profiles=CATALOG, settings=v2_settings(saved_block(ageGroup=rep['profile']['ageGroup'], playingLevel=level)), steps=steps,
                              spans=SPANS_EMPTY))
        cls.out = run(*jobs)
        with open(FILES['coaching.js'], encoding='utf-8') as handle:
            src = handle.read()
        block = src[src.index('const MEASURE_UI = {'):]
        cls.measure_keys = set(re.findall(r"(\w+): \['", block[:block.index('};')]))

    def test_every_real_report_renders_its_seven_sections(self):
        for (scene, variant, level, rep), out in zip(self.reports, self.out):
            o = out['first']
            self.assertEqual([s['key'] for s in o['sections']][:7], V2_SECTIONS, (scene, variant, level))
            self.assertEqual(len(o['cards']), sum(len(rep[k]) for k in ('strengths', 'needsWork', 'reviews', 'described')), (scene, level))
            main = ' '.join(s['text'] for s in o['sections'])
            plain = ' '.join(s['plain'] for s in o['sections'])
            self.assertEqual(S.event_problems(plain), [], (scene, level))
            self.assertNotRegex(main, r'\b(?:Threat n|Weak count|Open width m|Central count|Rest margin|Min behind)\b', 'no raw measurement key on the page')
            if level == 1:
                self.assertEqual(L1_BANNED.findall(plain), [], 'level 1: no word level 1 bans, also on real reports (source titles are quoted apart)')
                if rep['drills']:
                    self.assertIn('Source.', main, 'the drill keeps its source credit')
        self.assertTrue(any(rep['drills'] for _, _, level, rep in self.reports if level == 1), 'a level-1 report with a drill is checked')

    def test_measurement_chips_follow_the_level(self):
        for (scene, variant, level, rep), out in zip(self.reports, self.out):
            chips = [k for card in out['first']['chips'] for k in card]
            self.assertTrue(set(chips) <= self.measure_keys, (scene, level, set(chips)-self.measure_keys))
            if level <= 2:
                self.assertEqual(chips, [], 'no measurement below level 3')
            if level < 5:
                self.assertFalse({'widthShare', 'depthShare', 'coverPresentShare', 'goalSideShare', 'threatN', 'restMargin'} & set(chips), (scene, level))
        l7 = self.out[3]['first']
        self.assertTrue(any(l7['chips']), 'level 7 shows measurements')

    def test_a_key_moment_row_says_its_caption_once(self):
        for (scene, variant, level, rep), out in zip(self.reports, self.out):
            for row, ev in zip(out['first']['moments'], rep['keyMoments']):
                if ev['caption']:
                    self.assertEqual(row['text'].count(ev['caption']), 1, (scene, level, ev['caption']))

    def test_a_real_moment_jumps_to_its_frame_centre(self):
        for (scene, variant, level, rep), out in zip(self.reports, self.out):
            if not rep['keyMoments']:
                continue
            ev, j = rep['keyMoments'][0], out['jump']['jumps'][0]
            self.assertAlmostEqual(j['seek'], (ev['frame']+.5)/ev['fps'], places=6)
            self.assertEqual(j['view'], 'board' if ev['tier'] == 'image' else 'tracking')
            self.assertEqual(j['caption'].split(' Suggested area')[0], ev['annotation']['caption'])

    def test_coach_marked_possession_is_said(self):
        out = self.out[4]['first']
        cards = ' '.join(c['text'] for c in out['cards'])
        self.assertIn('marked by the coach', cards)
        self.assertIn('Who had the ball: marked by the coach', cards, 'the chip on a finding resting on coach-marked phases (level 3)')
        ov = sec(out, 'overview')
        self.assertIn('Who had the ball', ov)
        none = self.out[5]['first']
        self.assertNotIn('marked by the coach', ' '.join(c['text'] for c in none['cards']))
        self.assertIn('Mark who had the ball', sec(none, 'overview'), 'without marks or an engine estimate the report offers the marking')


@unittest.skipUnless(NODE, 'needs node to run the page scripts')
class V2ServerShapes(unittest.TestCase):
    """The answers as SERVER built them in wave 7 (server.py, store.py): progress under 'progress', the match date, a
    saved area removed by its adj- id, the coach-span flag of the report answer, the settings note."""

    @classmethod
    def setUpClass(cls):
        rich = v2_rich()
        work = rich['needsWork'][0]
        key_ev = work['timestamps'][0]['evidence']
        area = {'id': 'adj-0123456789ab', 'engine': 2, 'run': 'run1', 'team': 'A', 'findingKey': work['key'], 'range': {'t0': 720.0, 't1': 899.8},
                'made': '2026-10-22T11:00:00', 'finding': {'headline': None}, 'verdict': None, 'text': 'deeper', 'attached': work['key'], 'thisRun': True,
                'adjustment': {'id': 'adj-0123456789ab', 'findingKey': work['key'], 'evidence': key_ev, 'polygon': [[0.15, 0.30], [0.19, 0.30], [0.19, 0.34], [0.15, 0.34]],
                               'mover': 'A-05', 'note': 'deeper', 'created': '2026-10-22T11:00:00'}}
        settings = v2_settings(saved_block(squad={'id': SQUAD['id'], 'name': SQUAD['name']}, runLinks=[{'run': 'run1', 'team': 'A', 'confirmed': True}]))
        settings['matchDate'] = None
        cls.progress, cls.coach, cls.area, cls.date, cls.note = run(
            v2job(steps=[{'do': 'click', 'in': 'main', 'data': {'progress': 'pressure_and_cover'}, 'name': 'open'}],
                  progress={'squad': SQUAD, 'progress': progress_detail(), 'entries': 2, 'rederived': 0}),
            v2job(answers=[v2_answer(v2_rich(), coachSpans=True)]),
            v2job(answers=[v2_answer(rich, feedback=[area])], steps=[{'do': 'click', 'in': 'side', 'data': {'adjust': 'remove'}, 'name': 'removed'}]),
            v2job(settings=settings, steps=[{'do': 'click', 'in': 'setup', 'data': {'setup': 'toggle'}},
                                            {'do': 'choose', 'in': 'setup', 'data': {'setup': 'match-date'}, 'value': '2026-10-04', 'name': 'dated'}],
                  postAnswers={'/api/coaching/match-date': {'ok': True, 'date': '2026-10-04', 'updated': 1}}),
            v2job(steps=[{'do': 'click', 'in': 'setup', 'data': {'setup': 'toggle'}}, {'do': 'choose', 'in': 'setup', 'data': {'setup': 'style'}, 'value': 'counter'},
                         {'do': 'click', 'in': 'setup', 'data': {'setup': 'save'}, 'name': 'saved'}],
                  postAnswers={'/api/coaching/settings': {'ok': True, 'settings': None, 'stale': ['interpretation'], 'squad': None,
                                                          'note': 'The squad was not updated: no such squad.'}}))

    def test_the_progress_detail_answer_of_the_server(self):
        pr = sec(self.progress['open'], 'progression')
        self.assertIn('Moments. Shown in 24 of 31 judged moments over 3 matches', pr, 'mastery() under "progress"')

    def test_coach_marked_shares_are_not_called_estimates(self):
        ov = sec(self.coach['first'], 'overview')
        self.assertIn('Who had the ballTeam A about 45%, Team B about 55% (your marks where you made them, PitchIQ\u2019s estimate elsewhere; from 79% of the time)', ov)
        self.assertNotIn('estimated from 79%', ov)

    def test_a_saved_area_can_be_removed(self):
        side = self.area['first']['side']
        self.assertIn('Your area saved 2026-10-22 11:00:00 (deeper).', side['text'])
        self.assertIn('Remove my area', side['buttons'])
        post = self.area['removed']['posted'][-1]
        self.assertEqual(post, {'url': '/api/coaching/feedback/remove', 'body': {'id': 'adj-0123456789ab'}})
        self.assertIn('Your area was removed.', self.area['removed']['side']['text'])

    def test_the_match_date(self):
        post = [p for p in self.date['dated']['posted'] if p['url'] == '/api/coaching/match-date']
        self.assertEqual(post, [{'url': '/api/coaching/match-date', 'body': {'run': 'run1', 'date': '2026-10-04'}}])
        self.assertIn('Match date saved: 2026-10-04.', self.date['dated']['setup']['text'])

    def test_mixed_phases_and_description_only_reports(self):
        r = v2_rich()
        r['strengths'][0]['measurements']['phaseSource'] = 'mixed'
        d = v2_rich()
        moment = d['needsWork'][0]['timestamps'][0]
        d['described'][0]['timestamps'] = [moment]
        d['strengths'], d['needsWork'], d['reviews'] = [], [], []
        d['keyMoments'] = [m for m in d['keyMoments'] if m['id'] == moment['evidence']]
        for m in d['keyMoments']:
            m['finding'] = d['described'][0]['id']
            m['annotation']['finding'] = d['described'][0]['id']
        d['diagrams'], d['alternatives'], d['noAlternatives'], d['drills'] = [], [], [], [d['drills'][1]]
        d['nextObjective'] = {'skill': 'pressure_and_cover', 'text': 'Pressure and cover.', 'fromMatch': False, 'finding': None}
        mixed, described = run(v2job(report=r), v2job(report=d))
        strength = next(c for c in mixed['first']['cards'] if c['verdict'] == 'strength')
        self.assertIn('when we had the ball (estimated, partly marked by the coach)', strength['text'])
        self.assertIn('Who had the ball: partly marked by the coach', strength['text'])
        self.assertEqual(described['first']['selectedMoment'], moment['evidence'], 'a report of descriptions only still opens its first moment')

    def test_the_settings_note_is_said(self):
        self.assertIn('Saved. The report below was re-interpreted for this profile from the same measurements. The squad was not updated: no such squad.',
                      self.saved_text())

    def saved_text(self):
        return self.note['saved']['setup']['text']

    def test_an_area_note_is_folded_to_ascii(self):
        # SERVER refuses a note that is not printable ASCII (it reaches the PDF); smart punctuation is folded first.
        out = run(v2job(steps=[{'do': 'click', 'in': 'side', 'data': {'adjust': 'start'}},
                               {'do': 'choose', 'in': 'side', 'tag': 'TEXTAREA', 'value': 'Deeper \u2019near\u2019 the CB \u2013 \u201cyes\u201d\u2026  ok', 'event': 'input'},
                               {'do': 'click', 'in': 'side', 'data': {'adjust': 'save'}, 'name': 'saved'}]))[0]
        note = out['saved']['posted'][-1]['body']['adjustment']['note']
        self.assertEqual(note, 'Deeper \'near\' the CB - "yes"... ok')


# ---------- the page ----------
def page():
    with open(PAGE, encoding='utf-8') as handle:
        return handle.read()


def page_script(text):
    return '\n'.join(re.findall(r'<script>(.*?)</script>', text, re.S))


def span(src, start, end_marker):
    a = src.index(start)
    return src[a:src.index(end_marker, a)+len(end_marker)]


def js_function(src, name):
    start = src.index(f'function {name}(')
    depth = 0
    for k in range(src.index('{', start), len(src)):
        depth += {'{': 1, '}': -1}.get(src[k], 0)
        if depth == 0:
            return src[start:k+1]
    raise ValueError(name)


def node(script):
    done = subprocess.run([NODE, '-e', script], capture_output=True, text=True, encoding='utf-8', timeout=120)
    if done.returncode:
        raise AssertionError(done.stderr[-3000:])
    return json.loads(done.stdout)


COACH_BLOCK_END = "document.addEventListener('keydown',e=>{if(e.key==='Escape'&&coachMark&&!editor)clearCoachMark();});"


class PageWiring(unittest.TestCase):
    """index.html: the fourth tab and view, the module, the overlay canvas, the controller and the keyboard."""

    def test_the_stage_keeps_its_height(self):
        # CODE M1 / critic S-11: the closed marking bar (.controls.cm.off, display none) took 14 px and the marks toggle
        # 4 px more (34 against the 30 px chips): 2D 274 against 288 px, tracking 403 against 421 px at 1024x768 (measured
        # again after the fix in the in-app browser: 288 / 421 px, the checkout's, and equal at 800, 1440 and 1920 px).
        src = page_script(self.page)
        fit = js_function(src, 'fitStage')
        self.assertIn('(el.offsetHeight>0?el.offsetHeight+14:0)', fit)
        self.assertIn('.controls .cm-toggle{height:30px}', self.page)
        self.assertIn('.controls.cm.off{display:none}', self.page)

    @classmethod
    def setUpClass(cls):
        cls.page = page()
        cls.src = page_script(cls.page)

    @unittest.skipUnless(NODE, 'needs node to check the page script')
    def test_scripts_parse(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'page.js')
            with open(path, 'w', encoding='utf-8') as handle:
                handle.write(self.src)
            for script in (path, FILES['coaching.js']):
                done = subprocess.run([NODE, '--check', script], capture_output=True, text=True)
                self.assertEqual(done.returncode, 0, done.stderr)

    def test_tab_view_and_module(self):
        p = self.page
        tabs = re.findall(r'<button role="tab" type="button" data-view="(\w+)"', p)
        self.assertEqual(tabs, ['tracking', 'board', 'analytics', 'coaching'])
        self.assertIn('id="tabCoaching">Coaching<span class="tab-more"> report</span></button>', p)
        self.assertRegex(p, r'<section class="view" data-view="coaching" id="viewCoaching" role="tabpanel" aria-labelledby="tabCoaching"><div id="coachingPanel"')
        for control in ('crTeam', 'crRange', 'crCustom', 'crT0', 'crT1', 'crSetup', 'crGenerate', 'crPdf', 'crBanner', 'crMain', 'crSide'):
            self.assertIn(f'id="{control}"', p, control)
            self.assertIn(f"$('{control}')", self.src, control)
        self.assertNotIn('id="crContext"', p, 'the coaching setup replaces the Level select (DESIGN 9.1)')
        self.assertNotIn('compared with a professional reference', p, 'the v1 lede is gone (DESIGN 11.2)')
        self.assertIn('<div class="cr-setup" id="crSetup" aria-label="Coaching setup" hidden></div><div class="cr-banner" id="crBanner"', p)
        self.assertRegex(p, r'<div class="stage" id="stageWrap"><div class="coach-chip" id="coachChip" role="status" aria-live="polite" hidden>')
        for rule in ('.coach-chip[hidden],.stage.coaching-view .coach-chip,.stage.analytics-view .coach-chip', '.cr-setup[hidden]{display:none}',
                     '.cr-km-row.selected{', '.cr-dg-grid{', '.cr-prog-row{', '.cr-adjust-wrap.adjusting{cursor:move;touch-action:none}'):
            self.assertIn(rule, p, rule)
        self.assertRegex(p, r'<script src="/teamshape.js"></script>\s*<script src="/coaching.js"></script>')
        self.assertIn('<canvas id="overlay"></canvas><canvas id="coachOverlay" aria-hidden="true"></canvas></div></section>', p)
        self.assertIn('#coachOverlay{pointer-events:none}', p, 'clicks reach the boxes, also while drawing')
        self.assertIn('.stage.coaching-view .transport{display:none}', p)
        self.assertIn('.cr-custom[hidden]{display:none}', p)
        self.assertIn('teamShape.load(null,info);coaching.load(null,info);', self.src)
        self.assertIn('teamShape.load(data,info);coaching.load(data,info);', self.src)
        self.assertIn('trim.playhead(video.currentTime);drawCoachMark(i);}', self.src)
        self.assertIn("if((active==='coaching')!==coachingShown){coachingShown=active==='coaching';coaching.setActive(coachingShown);}", self.src,
                      'told only when the shown view changes, not on every resize')
        self.assertIn('function showResult(data){clearCoachMark();', self.src, 'a new result clears a coaching overlay still shown')
        self.assertIn('<span class="cr-tz">Source-video time</span>', p)
        self.assertIn('aria-label="From, source-video time (mm:ss.ss)"', p)
        self.assertIn('aria-label="To, source-video time (mm:ss.ss)"', p)
        self.assertNotRegex(p, r'innerHTML')
        with open(FILES['coaching.js'], encoding='utf-8') as handle:
            self.assertNotIn('innerHTML', handle.read(), 'report text is never parsed as HTML')

    def test_the_controller_counts_its_views(self):
        self.assertIn("const order=['tracking','board','analytics','coaching'],", self.src)
        views = span(self.src, 'const views=(()=>{', '})();')
        self.assertNotRegex(views, r'%\s*3|\+\s*3\)', 'no view count of three left')
        self.assertIn('order.length', views)
        self.assertNotIn('.view-tabs', self.src, 'the keyboard handler checks the switcher that exists')

    @unittest.skipUnless(NODE, 'needs node to run the page script')
    def test_views_loop_and_switcher_arrows(self):
        views = span(self.src, 'const views=(()=>{', '})();')
        script = r"""
const out = {};
function make(reducedMotion) {
  const mk = (id) => ({id, dataset: {}, style: {}, listeners: {}, added: [], offsetWidth: 80, offsetLeft: 0, offsetHeight: 100, textContent: '',
    classList: {add(...c) { this.owner.added.push(...c); }, remove() {}, toggle() {}}, setAttribute() {},
    addEventListener(t, f) { (this.listeners[t] = this.listeners[t] || []).push(f); }, removeEventListener() {}, click() { if (this.onclick) this.onclick(); }});
  const fix = (n) => { n.classList.owner = n; return n; };
  const els = {}, $ = (id) => els[id] || (els[id] = fix(mk(id)));
  const order = ['tracking', 'board', 'analytics', 'coaching'];
  const tabs = order.map((v) => Object.assign(fix(mk('tab-' + v)), {dataset: {view: v}})), panels = {};
  for (const v of order) panels[v] = fix(mk('panel-' + v));
  const seg = fix(mk('seg')), active = [];
  const document = {querySelectorAll: (s) => s === '.seg [role=tab]' ? tabs : [], querySelector: (s) => { const m = s.match(/data-view="(\w+)"/); return m ? panels[m[1]] : s === '.seg' ? seg : null; }};
  const resizes = [], window = {matchMedia: () => ({matches: reducedMotion}), addEventListener: (t, f) => { if (t === 'resize') resizes.push(f); }};
  const fitStage = () => {}, getComputedStyle = () => ({transform: '', getPropertyValue: () => '420'}), localStorage = {setItem() {}};
  const setTimeout = () => 0, coaching = {setActive: (on) => active.push(on)};
  VIEWS
  return {views, $, seg, tabs, panels, active, resizes};
}
const r = make(true);
r.$('viewPrev').click();                                            // to the coaching view
const told = r.active.length;
for (let k = 0; k < 5; k++) r.resizes.forEach((f) => f());
out.resize = {before: r.active.slice(0, told), added: r.active.length - told};
const a = make(true), seen = [];
for (let k = 0; k < 4; k++) { a.$('viewNext').click(); seen.push(a.views.active); }
a.$('viewPrev').click(); seen.push(a.views.active);
let stopped = 0, prevented = 0;
a.seg.listeners.keydown[0]({key: 'ArrowLeft', preventDefault() { prevented++; }, stopPropagation() { stopped++; }}); seen.push(a.views.active);
a.seg.listeners.keydown[0]({key: 'Enter', preventDefault() { prevented++; }, stopPropagation() { stopped++; }});
a.tabs[1].listeners.click[0](); seen.push(a.views.active);
out.loop = seen; out.keys = {stopped, prevented}; out.active = a.active;
const b = make(false); b.tabs[2].listeners.click[0](); out.towardsAnalytics = b.panels.analytics.added;
const c = make(false); c.tabs[3].listeners.click[0](); out.towardsCoaching = c.panels.coaching.added;
const d = make(false); d.$('viewPrev').click(); out.prevFromTracking = [d.views.active, d.panels.coaching.added];
process.stdout.write(JSON.stringify(out));
""".replace('VIEWS', views)
        out = node(script)
        self.assertEqual(out['loop'], ['board', 'analytics', 'coaching', 'tracking', 'coaching', 'analytics', 'board'])
        self.assertEqual(out['keys'], {'stopped': 1, 'prevented': 1}, 'an arrow on the switcher stops there; other keys pass')
        self.assertEqual(out['active'][-1], False)
        self.assertIn(True, out['active'], 'the coaching view is told when it is shown')
        self.assertEqual(out['active'], [False, True, False, True, False], 'told on each change of the shown view, and only then')
        self.assertEqual(out['resize'], {'before': [False, True], 'added': 0}, 'five resizes on the coaching view tell it nothing')
        self.assertIn('from-right', out['towardsAnalytics'], 'two ahead of four: forward')
        self.assertIn('from-left', out['towardsCoaching'], 'three ahead of four is one back')
        self.assertEqual(out['prevFromTracking'][0], 'coaching')
        self.assertIn('from-left', out['prevFromTracking'][1])

    @unittest.skipUnless(NODE, 'needs node to run the page script')
    def test_a_failing_coaching_script_never_stops_the_page(self):
        line = next(l for l in self.src.split('\n') if l.startswith('const coaching=(()=>{try{return Coaching.create('))
        script = r"""
const out = [];
for (const kind of ['missing', 'throws', 'works']) {
  const els = {}, $ = (id) => els[id] || (els[id] = {id, textContent: '', disabled: false});
  const api = () => null, jumpToEvidence = () => true, video = {}, tactical = {}, resultRun = null, document = {querySelectorAll: () => []};
  const console = {error: () => {}};
  const Coaching = kind === 'missing' ? undefined : {create: () => { if (kind === 'throws') throw new Error('boom'); return {load() {}, setActive() {}, labels: new Map(), real: true}; }};
  LINE
  coaching.load(null, {}); coaching.setActive(true);
  out.push({kind, real: !!coaching.real, labels: coaching.labels instanceof Map, main: $('crMain').textContent, generate: $('crGenerate').disabled, pdf: $('crPdf').disabled});
}
process.stdout.write(JSON.stringify(out));
""".replace('LINE', line)
        out = node(script)
        self.assertEqual([o['real'] for o in out], [False, False, True])
        for o in out[:2]:
            self.assertTrue(o['labels'])
            self.assertIn('The coaching report could not be loaded', o['main'])
            self.assertIn('the other views are unaffected', o['main'])
            self.assertEqual((o['generate'], o['pdf']), (True, True))
        self.assertIn('boom', out[1]['main'])
        self.assertEqual(out[2]['main'], '')

    @unittest.skipUnless(NODE, 'needs node to run the page script')
    def test_playback_keys(self):
        line = next(l for l in self.src.split('\n') if "views.active==='coaching')return;" in l and 'playback.toggle()' in l)
        script = r"""
let handler = null; const document = {addEventListener: (t, f) => { if (t === 'keydown') handler = f; }};
let mode = null, editor = null; const views = {active: 'tracking'}, video = {currentTime: 10}, calls = [];
const playback = {toggle: () => calls.push('toggle'), step: (n) => calls.push('step ' + n), seek: (t) => calls.push('seek ' + t)}, toggleFullscreen = () => calls.push('full');
LINE
const key = (k, target, extra) => handler(Object.assign({key: k, target, preventDefault() {}, altKey: false, ctrlKey: false, metaKey: false, shiftKey: false}, extra || {}));
const body = {tagName: 'BODY', closest: () => null}, onSeg = {tagName: 'BUTTON', closest: (s) => s === '.seg' ? {} : null};
key('ArrowRight', body); key('ArrowRight', onSeg); views.active = 'coaching'; key('ArrowRight', body); key(' ', body); views.active = 'analytics'; key('k', body);
views.active = 'board'; key('ArrowLeft', body, {shiftKey: true});
process.stdout.write(JSON.stringify(calls));
""".replace('LINE', line)
        self.assertEqual(node(script), ['step 1', 'seek 5'])

    @unittest.skipUnless(NODE, 'needs node to run the page script')
    def test_jump_pauses_seeks_selects_and_shows_until_the_frame_changes(self):
        block = span(self.src, 'let coachMark=null;', COACH_BLOCK_END)
        frame_index = js_function(self.src, 'frameIndex')
        frames = [{'time': round(f/30, 6), 'frame': f} for f in range(30, 60)]
        script = r"""
const result = {frames: FRAMES}, info = {video: {fps: 30}};
let mode = null, editor = null; const shown = [], seeks = [], selected = [], drawn = [], listeners = {};
let invalidated = 0, cleared = 0;
const closeEditor = () => {}, views = {show: (v) => shown.push(v)}, playback = {seek: (t) => { seeks.push(t); video.currentTime = t; }}, select = (id) => selected.push(id);
const video = {paused: false, currentTime: 0, pause() { this.paused = true; }, addEventListener: (t, f) => { listeners[t] = f; }};
const document = {querySelector: () => ({offsetHeight: 60}), addEventListener() {}};
const canvas = {width: 1920, height: 1080}, ctx = {clearRect: () => { cleared++; }}, $ = () => ({width: 0, height: 0, style: {}, getContext: () => ctx, setAttribute() {}});
const tactical = {overlays: [], colours: {A: '#b98cff', B: '#f6b564'}, invalidate: () => { invalidated++; }};
const COLORS = {A: '#b98cff', B: '#f6b564'}, coaching = {labels: new Map()};
const Coaching = {drawVideo: (c, spec, w, h, colours) => drawn.push(['video', spec.tag, w, h, colours.A]), drawBoard: (c, g, spec, colours) => drawn.push(['board', spec.tag, colours.B])};
FRAMEINDEX
BLOCK
const out = {};
out.ok = jumpToEvidence({frame: 40, seek: 40.5 / 30, view: 'tracking', focus: ['A-03', 'B-07'], overlay: {tag: 'o'}, diagram: {tag: 'd'}});
out.after = {shown: shown.slice(), seeks: seeks.slice(), selected: selected.slice(), paused: video.paused, invalidated};
const i = frameIndex(video.currentTime); out.index = i;
drawCoachMark(i); drawCoachMark(i); out.drawnOnce = drawn.slice();
tactical.overlays[0](null, null, {frame: i}); tactical.overlays[0](null, null, {frame: i + 1}); out.board = drawn.slice(1);
drawCoachMark(i + 1); out.leftFrame = {mark: coachMark === null, invalidated, cleared: cleared > 0};
jumpToEvidence({frame: 41, seek: 41.5 / 30, view: 'board', focus: [], overlay: null, diagram: {tag: 'd2'}}); out.board2 = {seek: seeks[seeks.length - 1], shown: shown[shown.length - 1], selected: selected[selected.length - 1]};
listeners.play(); out.played = coachMark === null;
out.missing = jumpToEvidence({frame: 5, seek: 5.5 / 30, view: 'tracking'}); mode = 'boundary'; out.drawing = jumpToEvidence({frame: 40, seek: 40.5 / 30});
out.seeksTotal = seeks.length;
process.stdout.write(JSON.stringify(out));
""".replace('FRAMES', json.dumps(frames)).replace('FRAMEINDEX', frame_index).replace('BLOCK', block)
        out = node(script)
        self.assertTrue(out['ok'])
        self.assertEqual(out['after']['shown'], ['tracking'])
        self.assertAlmostEqual(out['after']['seeks'][0], 40.5/30, places=9, msg='the frame centre')
        self.assertEqual((out['after']['selected'], out['after']['paused'], out['index']), (['A-03'], True, 10))
        self.assertEqual(out['drawnOnce'], [['video', 'o', 1920, 1080, '#b98cff']], "drawn once while the frame stays, in the boxes' colours")
        self.assertEqual(out['board'], [['board', 'd', '#f6b564']], 'the board layer draws only at the evidence frame')
        self.assertEqual(out['leftFrame'], {'mark': True, 'invalidated': 2, 'cleared': True})
        self.assertAlmostEqual(out['board2']['seek'], round(41/30, 6)+.2/30, places=9, msg='0.2 frame into the frame on the board')
        self.assertEqual((out['board2']['shown'], out['board2']['selected']), ('board', None))
        self.assertTrue(out['played'], 'cleared when the video plays')
        self.assertEqual((out['missing'], out['drawing'], out['seeksTotal']), (False, False, 2), 'no jump outside the result or while drawing')

    @unittest.skipUnless(NODE, 'needs node to run the page script')
    def test_a_key_moment_annotation_lives_through_its_window(self):
        # DESIGN 5.6: inside the window the mark stays while playing or scrubbing; the key frame draws the whole overlay,
        # the other frames the people it follows; leaving the window, Escape or the chip clear it; Replay starts at the
        # window and plays through the one transport; the stage is scrolled below the sticky header (critic S4).
        block = span(self.src, 'let coachMark=null;', COACH_BLOCK_END)
        frame_index = js_function(self.src, 'frameIndex')
        frames = [{'time': round(f/30, 6), 'frame': f, 'people': [{'id': 'A-04', 'box': [.4, .5, .02, .08]}, {'id': 'A-05', 'box': [.3, .5, .02, .08]}]}
                  for f in range(430, 470)]
        script = r"""
const result = {frames: FRAMES}, info = {video: {fps: 30}};
let mode = null, editor = null; const shown = [], seeks = [], selected = [], drawn = [], listeners = {}, keys = [], scrolled = [];
let invalidated = 0, toggles = 0;
const closeEditor = () => {}, views = {show: (v) => shown.push(v)}, select = (id) => selected.push(id);
const video = {paused: false, currentTime: 0, pause() { this.paused = true; }, addEventListener: (t, f) => { listeners[t] = f; }};
const playback = {seek: (t) => { seeks.push(t); video.currentTime = t; }, toggle: () => { toggles++; video.paused = !video.paused; }};
const document = {querySelector: (s) => s === '.app-header' ? {offsetHeight: 72} : null, addEventListener: (t, f) => { if (t === 'keydown') keys.push(f); }};
const els = {};
const $ = (id) => els[id] || (els[id] = {id, width: 0, height: 0, hidden: true, textContent: '', style: {}, getContext: () => ({clearRect() {}}), attrs: {},
                                         setAttribute(k, v) { this.attrs[k] = String(v); }, scrollIntoView(o) { scrolled.push([id, o.block, this.style.scrollMarginTop]); }});
const canvas = {width: 1000, height: 500};
const tactical = {overlays: [], colours: {A: '#b98cff', B: '#f6b564'}, invalidate: () => { invalidated++; }};
const COLORS = {A: '#b98cff', B: '#f6b564'}, coaching = {labels: new Map()};
const Coaching = {drawVideo: (c, spec) => drawn.push(['video', spec.tag]), drawBoard: (c, g, spec) => drawn.push(['board', spec.tag]),
                  drawFollow: (c, follow, o) => drawn.push(['follow', o.mode, o.mode === 'video' ? [...o.people.keys()].join(',') : 'state', o.tier || null])};
FRAMEINDEX
BLOCK
const out = {};
const ev = {frame: 450, seek: 450.5 / 30, view: 'tracking', focus: ['A-04'], overlay: {tag: 'o'}, diagram: {tag: 'd'}, tier: 'world', caption: 'Pressure on the ball; nobody covered behind it.',
            window: {t0: 447 / 30, t1: 452 / 30}, follow: {focus: [{id: 'A-04'}], support: [{id: 'A-05'}], relations: []}};
out.ok = jumpToEvidence(ev);
out.chip = {hidden: $('coachChip').hidden, text: $('coachChipText').textContent};
out.scrolled = scrolled.slice();
const at = (t) => { video.currentTime = t; const i = frameIndex(t); drawCoachMark(i); return i; };
const key = at(450.5 / 30); out.key = drawn.slice();
listeners.play(); video.paused = false; out.afterPlay = !!coachMark;
at(448.5 / 30); out.inside = drawn.slice(1);
tactical.overlays[0](null, null, {frame: 10, time: 448.5 / 30, players: []}); out.board = drawn.slice(2);
tactical.overlays[0](null, null, {frame: key, time: 450.5 / 30, players: []}); out.boardKey = drawn.slice(3);
at(460.5 / 30); out.left = {mark: coachMark === null, chip: $('coachChip').hidden};
jumpToEvidence(Object.assign({}, ev, {replay: true})); out.replay = {seek: seeks[seeks.length - 1], toggles, paused: video.paused};
keys.forEach((f) => f({key: 'Escape'})); out.escape = coachMark === null;
jumpToEvidence(ev); $('coachChipClose').onclick(); out.closed = {mark: coachMark === null, chip: $('coachChip').hidden};
jumpToEvidence(Object.assign({}, ev, {window: null, follow: null, caption: ''})); listeners.play(); out.v1Play = coachMark === null;
process.stdout.write(JSON.stringify(out));
""".replace('FRAMES', json.dumps(frames)).replace('FRAMEINDEX', frame_index).replace('BLOCK', block)
        out = node(script)
        self.assertTrue(out['ok'])
        self.assertEqual(out['chip'], {'hidden': False, 'text': 'Pressure on the ball; nobody covered behind it.'})
        self.assertEqual(out['scrolled'], [['stageWrap', 'start', '80px']], 'scrolled below the 72 px header')
        self.assertEqual(out['key'], [['video', 'o']], 'the key frame: the whole overlay')
        self.assertTrue(out['afterPlay'], 'playing inside the window keeps the annotation')
        self.assertEqual(out['inside'], [['follow', 'video', 'A-04,A-05', 'world']], 'elsewhere in the window the people follow their live boxes')
        self.assertEqual(out['board'], [['follow', 'board', 'state', None]])
        self.assertEqual(out['boardKey'], [['board', 'd']])
        self.assertEqual(out['left'], {'mark': True, 'chip': True}, 'leaving the window clears it')
        self.assertAlmostEqual(out['replay']['seek'], 447/30, places=9, msg='Replay starts at the window')
        self.assertEqual((out['replay']['toggles'], out['replay']['paused']), (1, False), 'and plays through the page transport')
        self.assertTrue(out['escape'])
        self.assertEqual(out['closed'], {'mark': True, 'chip': True})
        self.assertTrue(out['v1Play'], 'a v1 mark still clears when the video plays')

    def test_the_possession_marks_markup(self):
        # Wave 7 (user decision 1): the marking bar under the video and the board, a button in each view's controls,
        # the coach lane inside the possession panel (between the engine strip and its axis), its height reserved like
        # the strip's (and given back in full screen), the elements handed to coaching.js, the lane's playhead ticked.
        p, src = self.page, self.src
        self.assertIn('<div class="controls cm off" id="coachMarks" data-for="tracking board" role="group" aria-label="Possession marked by the coach" hidden></div>', p)
        self.assertEqual(p.count('<div class="group" data-cm-toggle hidden></div>'), 2)
        tracking = span(p, '<div class="controls" data-for="tracking">', '</div></div>')
        self.assertIn('data-cm-toggle', tracking)
        board = span(p, '<div class="controls" data-for="board" hidden>', 'id="tacticalStatus"></span></div>')
        self.assertIn('data-cm-toggle', board)
        panel = span(p, '<div class="pz" id="possession"', '<p class="pz-msg" id="pzMsg" hidden></p></div>')
        self.assertLess(panel.index('id="pzTrack"'), panel.index('id="pzCoach"'))
        self.assertLess(panel.index('id="pzCoach"'), panel.index('id="pzFoot"'))
        self.assertIn('<div class="pz-coach" id="pzCoach" hidden><div class="pz-ctrack" id="pzCoachTrack" role="img"', panel)
        self.assertIn('.controls .group[hidden],.controls.cm.off{display:none}', p, 'an empty toggle host takes no room')
        self.assertLess(p.index('#viewBoard.pz-on.pz-coach-on{--pz-h:144px}'), p.index('.stage:fullscreen #viewBoard,.stage.full #viewBoard{--pz-h:0px}'),
                        'full screen still gives the whole stage to the board')
        self.assertIn("marks:{bar:$('coachMarks'),toggles:[...document.querySelectorAll('[data-cm-toggle]')],lane:$('pzCoach'),track:$('pzCoachTrack'),note:$('pzCoachNote'),"
                      "host:$('viewBoard'),now:()=>video.currentTime,seek:t=>{if(mode)return;playback.seek(t);},fit:()=>fitStage(),show:v=>views.show(v)}", src)
        self.assertIn('rosterStrip(boardState,i);coaching.tick(video.currentTime);analytics.tick(video.currentTime);possession.tick(video.currentTime);playback.tick();', src,
                      'ticked before the other views, so the call sequence test_possession_panel checks stays as it is')
        self.assertIn('return {load(){},setActive(){},tick(){},labels:new Map()};', src, 'the fenced fallback ticks too')
        self.assertIn("el.hidden=!el.dataset.for.split(' ').includes(active)", src)

    @unittest.skipUnless(NODE, 'needs node to run the page script')
    def test_a_controls_row_can_serve_several_views(self):
        views = span(self.src, 'const views=(()=>{', '})();')
        script = r"""
const mk = (id) => ({id, dataset: {}, style: {}, listeners: {}, offsetWidth: 80, offsetLeft: 0, offsetHeight: 100, textContent: '', hidden: false,
  classList: {add() {}, remove() {}, toggle() {}}, setAttribute() {}, addEventListener(t, f) { (this.listeners[t] = this.listeners[t] || []).push(f); }, removeEventListener() {},
  click() { if (this.onclick) this.onclick(); }});
const els = {}, $ = (id) => els[id] || (els[id] = mk(id));
const order = ['tracking', 'board', 'analytics', 'coaching'], tabs = order.map((v) => Object.assign(mk('tab-' + v), {dataset: {view: v}})), panels = {};
for (const v of order) panels[v] = mk('panel-' + v);
const rows = [Object.assign(mk('t'), {dataset: {for: 'tracking'}}), Object.assign(mk('b'), {dataset: {for: 'board'}}), Object.assign(mk('m'), {dataset: {for: 'tracking board'}})];
const document = {querySelectorAll: (s) => s === '.seg [role=tab]' ? tabs : s === '[data-for]' ? rows : [], querySelector: (s) => { const m = s.match(/data-view="(\w+)"/); return m ? panels[m[1]] : mk('seg'); }};
const window = {matchMedia: () => ({matches: true}), addEventListener() {}};
const fitStage = () => {}, getComputedStyle = () => ({transform: '', getPropertyValue: () => '420'}), localStorage = {setItem() {}}, setTimeout = () => 0, coaching = {setActive() {}};
VIEWS
const out = {};
const state = () => rows.map((r) => r.hidden);
out.tracking = state(); views.show('board'); out.board = state(); views.show('analytics'); out.analytics = state(); views.show('coaching'); out.coaching = state();
process.stdout.write(JSON.stringify(out));
""".replace('VIEWS', views)
        out = node(script)
        self.assertEqual(out['tracking'], [False, True, False])
        self.assertEqual(out['board'], [True, False, False], 'the marking bar stays under the 2D match')
        self.assertEqual(out['analytics'], [True, True, True])
        self.assertEqual(out['coaching'], [True, True, True])

    @unittest.skipUnless(NODE, 'needs node to run the page script')
    def test_the_chip_explains_the_moment_and_leads_back(self):
        block = span(self.src, 'let coachMark=null;', COACH_BLOCK_END)
        frame_index = js_function(self.src, 'frameIndex')
        frames = [{'time': round(f/30, 6), 'frame': f, 'people': []} for f in range(430, 470)]
        script = r"""
const result = {frames: FRAMES}, info = {video: {fps: 30}};
let mode = null, editor = null; const shown = [];
const closeEditor = () => {}, views = {show: (v) => shown.push(v)}, select = () => {};
const video = {paused: false, currentTime: 0, pause() { this.paused = true; }, addEventListener() {}};
const playback = {seek: (t) => { video.currentTime = t; }, toggle() {}};
const document = {querySelector: () => ({offsetHeight: 60}), addEventListener() {}};
const els = {};
const $ = (id) => els[id] || (els[id] = {id, width: 0, height: 0, hidden: true, textContent: '', style: {}, attrs: {}, getContext: () => ({clearRect() {}}),
                                         setAttribute(k, v) { this.attrs[k] = String(v); }, scrollIntoView() {}});
const canvas = {width: 1000, height: 500}, tactical = {overlays: [], colours: {}, invalidate() {}}, COLORS = {A: '#b98cff', B: '#f6b564'}, coaching = {labels: new Map()};
const Coaching = {drawVideo() {}, drawBoard() {}, drawFollow() {}};
FRAMEINDEX
BLOCK
const out = {};
const ev = {frame: 450, seek: 450.5 / 30, view: 'tracking', focus: [], overlay: null, diagram: null, tier: 'world', caption: 'Cover behind a forward run',
            explanation: 'When we had the ball (estimated), the left defender pushed up while nobody covered the channel behind.', window: {t0: 447 / 30, t1: 452 / 30}, follow: null};
jumpToEvidence(ev);
const chip = () => ({why: $('coachChipWhy').hidden, more: $('coachChipMore').hidden, text: $('coachChipMore').textContent, expanded: $('coachChipWhy').attrs['aria-expanded']});
out.shown = chip();
$('coachChipWhy').onclick(); out.open = chip();
$('coachChipWhy').onclick(); out.closed = chip();
$('coachChipBack').onclick(); out.back = shown.slice();
jumpToEvidence(Object.assign({}, ev, {explanation: null})); out.none = chip();
process.stdout.write(JSON.stringify(out));
""".replace('FRAMES', json.dumps(frames)).replace('FRAMEINDEX', frame_index).replace('BLOCK', block)
        out = node(script)
        self.assertEqual(out['shown'], {'why': False, 'more': True, 'text': 'When we had the ball (estimated), the left defender pushed up while nobody covered the channel behind.',
                                        'expanded': 'false'}, 'the explanation waits behind Why')
        self.assertEqual((out['open']['more'], out['open']['expanded']), (False, 'true'))
        self.assertEqual((out['closed']['more'], out['closed']['expanded']), (True, 'false'))
        self.assertEqual(out['back'], ['tracking', 'coaching'], 'Report goes back to the coaching report')
        self.assertTrue(out['none']['why'], 'no explanation: no Why button')
        self.assertIn('<button type="button" class="quiet wide" id="coachChipWhy" aria-expanded="false" aria-controls="coachChipMore"', self.page)
        self.assertIn('<p class="coach-chip-more" id="coachChipMore" hidden></p></div>', self.page)


if __name__ == '__main__':
    unittest.main()
