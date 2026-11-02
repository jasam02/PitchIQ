"""Full Match Intelligence in the review page: match.js run in node, with a small DOM, on the real CAN-ISL answers of
the backend (tests/match_intel_index.json.gz = GET /api/match/index and match_intel_state.json.gz = GET /api/match/intel
of match 20261028-062138-83f2a6, its people and state built).

- the transport: time <-> x on the seek bar (the range thumb's inset lane), the state strip (consecutive intervals of
  one state merged, a class per state) and the event markers (a removed event is not drawn, a candidate is hollow:
  st-candidate), the tooltip, a click that pauses and seeks through the page's playback controller, the arrow keys;
- the timeline filter (class TimelineFilter): goals only on a first visit, a kind turned on or off drawn at once and
  kept in localStorage, All / None / Goals only, the play-state band, storage that throws or holds junk; hidden markers
  out of the arrow keys and "Also here"; the goal marker, its running score (only where it has room) and its tooltip;
- the halftime: the two handles of the break (or the one marker), dragged by pointer (the label follows the drag),
  moved by the keys, typed in the editor (mm:ss, h:mm:ss, seconds), the break switched off, the direction set; the POST
  bodies of Save and Reset, the draft ("not saved yet"), the state made again and asked for until ready, then the
  index read again; never past the match range;
- the events drawer: the shares, the direction check and its note, the rows in time order and the type filter, the POST
  bodies of confirm, type, set to now, remove, add and undo;
- the roster review: the slots 01-10 and GK at the playback time from index.places, the unresolved fragments, the ones
  left out, the slot's substitutions, the POST bodies of its corrections, the corrections' texts;
- the build: a match whose state is missing (or made for other people) is built, polled and drawn; a state that fails
  or a server without the routes leaves the rest of the page alone; leaving the match removes it all;
- tactical.js draws a provisional slot label ('A-04?') tentative, 'A-?' as before.

Run:  python -m unittest test_match_intel_panel      (from this directory; about 20 s)
"""
import collections
import gzip
import json
import math
import os
import re
import shutil
import subprocess
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
NODE = shutil.which('node')
MATCH_JS = os.path.join(ROOT, 'match.js')
TACTICAL_JS = os.path.join(ROOT, 'tactical.js')
ANALYTICS_JS = os.path.join(ROOT, 'analytics.js')
PAGE = os.path.join(ROOT, 'index.html')
INDEX_GZ = os.path.join(HERE, 'match_intel_index.json.gz')
STATE_GZ = os.path.join(HERE, 'match_intel_state.json.gz')
DOT, DASH, NDASH, ARROW, ELL = chr(0xb7), chr(0x2014), chr(0x2013), chr(0x2192), chr(0x2026)


def load(path):
    with gzip.open(path, 'rt', encoding='utf-8') as handle:
        return json.load(handle)


INDEX, INTEL = load(INDEX_GZ), load(STATE_GZ)
MATCH = INDEX['matchId']
DUR = INDEX['video']['duration']
RECT = {'left': 100, 'width': 1000}          # the range's box in the harness
ROSTER = {p['matchId']: p for p in INDEX['roster']}
EVENTS = sorted(INTEL['events'], key=lambda e: e['t'])
BY_ID = {e['id']: e for e in EVENTS}
THROW = next(e['id'] for e in EVENTS if e['type'] == 'THROW_IN')


def js_round(x):
    return math.floor(x + .5)


def to_x(t):
    return RECT['left'] + 7 + max(0.0, min(DUR, t)) / DUR * (RECT['width'] - 14)


def to_t(x):
    return max(0.0, min(1.0, (x - RECT['left'] - 7) / (RECT['width'] - 14))) * DUR


def clock(t):
    t = max(0, js_round(t))
    return f'{t // 60:02d}:{t % 60:02d}'


def num(left):
    return float(left.rstrip('%'))


def name_of(p):
    if p.get('excluded'):
        return f"fragment {p['matchId']}"
    return f"{p['display']} {DOT} {p['matchId']}" if p.get('display') and p['display'] != p['matchId'] else p['matchId']


def runs_of(timeline):
    runs = []
    for r in timeline:
        if r['t1'] <= r['t0']:
            continue
        if runs and runs[-1]['state'] == r['state'] and abs(runs[-1]['t1'] - r['t0']) < 1e-6:
            runs[-1]['t1'] = r['t1']
        else:
            runs.append({'state': r['state'], 't0': r['t0'], 't1': r['t1']})
    return runs


def holder(nodes, t):
    best = -1
    for k, n in enumerate(nodes):
        if n['t0'] <= t <= n['t1']:
            return k, True
        if n['t1'] < t and (best < 0 or n['t1'] >= nodes[best]['t1']):
            best = k
    return best, False


def place(team, slot):
    return next(p for p in INDEX['places'][team] if p['slot'] == slot)


KIND = {'GOAL': 'goal', 'SUBSTITUTION': 'sub', 'KICK_OFF': 'ko', 'FOUL_OR_WHISTLE_STOPPAGE': 'stop', 'OTHER_STOPPAGE': 'stop'}
WORDS = {'GOAL': 'Goal', 'SUBSTITUTION': 'Substitution', 'KICK_OFF': 'Kick-off', 'CORNER_KICK': 'Corner kick', 'GOAL_KICK': 'Goal kick', 'THROW_IN': 'Throw-in',
         'FREE_KICK': 'Free kick', 'PENALTY_KICK': 'Penalty kick', 'UNKNOWN_RESTART': 'Unknown restart', 'FOUL_OR_WHISTLE_STOPPAGE': 'Foul or whistle stoppage',
         'OTHER_STOPPAGE': 'Other stoppage'}
STATE_CLASS = {'IN_PLAY': 's-play', 'OUT_OF_PLAY': 's-stop', 'RESTART_PREPARATION': 's-prep', 'POSSIBLE_STOPPAGE': 's-maybe', 'HALFTIME': 's-half',
               'NON_MATCH_FOOTAGE': 's-nopic', 'UNKNOWN': 's-unknown'}
# the timeline filter: its localStorage key, the kinds in the popover's order, the stored choices
MARKS_KEY = 'pitchiq.timelineMarkers.v1'
MARK_TYPES = ['GOAL', 'KICK_OFF', 'CORNER_KICK', 'GOAL_KICK', 'THROW_IN', 'FREE_KICK', 'PENALTY_KICK', 'UNKNOWN_RESTART', 'SUBSTITUTION',
              'FOUL_OR_WHISTLE_STOPPAGE', 'OTHER_STOPPAGE']


def stored(show, strip=True):
    return {MARKS_KEY: json.dumps({'show': list(show), 'strip': strip})}


ALL_MARKERS = stored(MARK_TYPES)
GOALS = [e for e in EVENTS if e['type'] == 'GOAL']
SEP = chr(0x203a)
SCORE_PX = 28          # match.js: the room a goal's score label needs after the previous one

HARNESS = r"""
const fs = require('fs'), vm = require('vm'), zlib = require('zlib');
const input = JSON.parse(fs.readFileSync(0, 'utf8'));
const gz = (p) => JSON.parse(zlib.gunzipSync(fs.readFileSync(p)).toString('utf8'));
const INDEX_TEXT = JSON.stringify(gz(input.index)), INTEL_TEXT = JSON.stringify(gz(input.intel)), SOURCE = fs.readFileSync(input.match, 'utf8');
const copy = (x) => (x === undefined ? undefined : JSON.parse(JSON.stringify(x)));
const settle = async (n = 90) => { for (let k = 0; k < n; k++) await new Promise((r) => setImmediate(r)); };

// '@index' / '@intel': the fixtures; {'@index': patch} / {'@intel': patch}: top-level keys replaced (__events patches
// events by id, __halftime and __check patch halftime and directionCheck); {__status, __error}: an HTTP error
function token(v) {
  if (v === '@index') return JSON.parse(INDEX_TEXT);
  if (v === '@intel') return JSON.parse(INTEL_TEXT);
  if (v && typeof v === 'object' && !Array.isArray(v)) {
    if ('@index' in v || '@intel' in v) {
      const base = '@index' in v ? JSON.parse(INDEX_TEXT) : JSON.parse(INTEL_TEXT), patch = copy(v['@index'] || v['@intel']);
      if (patch.__events) { for (const e of base.events) if (patch.__events[e.id]) Object.assign(e, patch.__events[e.id]); delete patch.__events; }
      if (patch.__halftime) { Object.assign(base.halftime, patch.__halftime); delete patch.__halftime; }
      if (patch.__check) { Object.assign(base.directionCheck, patch.__check); delete patch.__check; }
      return Object.assign(base, patch);
    }
    const out = {};
    for (const [k, x] of Object.entries(v)) out[k] = token(x);
    return out;
  }
  return copy(v);
}

async function scenario(job) {
  let active = null, hold = !!job.hold, timers = 0;
  const held = [], docListeners = {}, posts = [], gets = [], seeks = [], selects = [], fits = [], warns = [], keyLog = [], geo = [];
  class Node {
    constructor(tag) {
      this.tagName = String(tag).toUpperCase(); this.children = []; this.parentElement = null; this.attrs = {}; this.dataset = {};
      this.style = {setProperty(k, v) { this[k] = String(v); }, removeProperty(k) { delete this[k]; }};
      this.listeners = {}; this.hidden = false; this.disabled = false; this.value = ''; this.own = ''; this.cls = ''; this.title = ''; this.id = '';
      this.type = ''; this.checked = false; this.tabIndex = 0; this.placeholder = ''; this.rect = null; this.width = 0;
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
    append(...cs) { for (const c of cs) { if (c.parentElement && c.parentElement.removeChild) c.parentElement.removeChild(c); c.parentElement = this; this.children.push(c); } }
    removeChild(c) { const i = this.children.indexOf(c); if (i >= 0) this.children.splice(i, 1); c.parentElement = null; }
    replaceChildren(...cs) { this.own = ''; this.clear(); this.append(...cs); }
    remove() { if (this.parentElement) this.parentElement.removeChild(this); }
    setAttribute(k, v) { this.attrs[k] = String(v); }
    getAttribute(k) { return k in this.attrs ? this.attrs[k] : null; }
    removeAttribute(k) { delete this.attrs[k]; }
    hasAttribute(k) { return k in this.attrs; }
    getBoundingClientRect() { return this.rect || {left: 0, top: 0, width: 0, height: 0, right: 0, bottom: 0}; }
    get clientWidth() { return this.width; }
    get offsetWidth() { return this.width; }
    get offsetHeight() { return 0; }
    get isConnected() { for (let x = this; x; x = x.parentElement) if (x === document.body) return true; return false; }
    get options() { return this.children.filter((c) => c.tagName === 'OPTION'); }
    contains(n) { for (let x = n; x; x = x.parentElement) if (x === this) return true; return false; }
    addEventListener(t, f) { (this.listeners[t] = this.listeners[t] || []).push(f); }
    removeEventListener(t, f) { this.listeners[t] = (this.listeners[t] || []).filter((g) => g !== f); }
    dispatch(type, props) {
      props = props || {};
      const ev = Object.assign({type, target: this, currentTarget: this, defaultPrevented: false, propagationStopped: false, bubbles: true,
                                preventDefault() { this.defaultPrevented = true; }, stopPropagation() { this.propagationStopped = true; }}, props);
      for (const f of (docListeners[type + ':capture'] || []).slice()) f(ev);
      for (let n = this; n; n = ev.bubbles ? n.parentElement : null) {
        ev.currentTarget = n;
        if (typeof n['on' + type] === 'function') n['on' + type](ev);
        for (const f of (n.listeners[type] || []).slice()) f(ev);
        if (ev.propagationStopped || !ev.bubbles) break;
      }
      if (ev.bubbles && !ev.propagationStopped) for (const f of (docListeners[type] || []).slice()) f(ev);
      return ev;
    }
    click() { if (!this.disabled) this.dispatch('click'); }
    focus() { if (active === this) return; const prev = active; active = this; if (prev) prev.dispatch('blur', {bubbles: false}); this.dispatch('focus', {bubbles: false}); }
    blur() { if (active === this) { active = null; this.dispatch('blur', {bubbles: false}); } }
    select() {}
    setPointerCapture() {}
    releasePointerCapture() {}
    scrollIntoView() {}
    getContext() { return new Proxy({}, {get: () => () => {}}); }
  }
  const all = (n, out = []) => { out.push(n); for (const c of n.children || []) all(c, out); return out; };
  const find = (root, pred) => (root ? all(root).filter((n) => n instanceof Node && pred(n)) : []);
  const one = (root, pred) => find(root, pred)[0] || null;
  const has = (n, c) => !!n && n instanceof Node && n.classes().includes(c);
  const text = (n) => (n ? n.textContent : null);
  const body = new Node('body');
  const document = {
    body, createElement: (t) => new Node(t), createElementNS: (ns, t) => new Node(t), createTextNode: (t) => ({nodeType: 3, textContent: String(t), parentElement: null}),
    getElementById: (id) => one(body, (n) => n.id === id), querySelectorAll: () => [], querySelector: () => null,
    addEventListener: (t, f, capture) => { const key = capture ? t + ':capture' : t; (docListeners[key] = docListeners[key] || []).push(f); },
    removeEventListener() {},
    get activeElement() { return active; },
  };
  const byId = (id) => document.getElementById(id);
  const el = (tag, id, cls) => { const n = new Node(tag); if (id) n.id = id; if (cls) n.className = cls; return n; };
  // the page: the transport (the range in a .seek), the drawers' boxes
  const INDEX0 = JSON.parse(INDEX_TEXT);
  const transport = el('div', 'transport', 'transport'), seek = el('span', null, 'seek'), mt = el('span', 'mt', 'mt'), range = el('input', 'timeline');
  mt.hidden = true; range.type = 'range'; range.min = '0'; range.max = String(INDEX0.video.duration); range.rect = {left: 100, top: 0, width: 1000, height: 4, right: 1100, bottom: 4};
  const msg = el('span', 'mtMsg', 'mt-msg'); msg.hidden = true;
  // the timeline filter beside the range (index.html #mtf: the button with its text, the popover match.js fills)
  const mtf = el('span', 'mtf', 'mtf'), mtfBtn = el('button', 'mtfBtn', 'mtf-btn'), mtfText = el('span', 'mtfText'), mtfPop = el('div', 'mtfPop', 'mtf-pop');
  mtf.hidden = true; mtfPop.hidden = true; mtfText.textContent = 'Goals only'; mtfBtn.append(mtfText); mtf.append(mtfBtn, mtfPop);
  seek.append(mt, range); transport.append(seek, mtf, msg);
  // localStorage: job.storage at the start (a map of keys to strings), or 'throw': storage that refuses every access
  const store = job.storage === 'throw' ? null : Object.assign({}, job.storage || {});
  const refuse = () => { throw new Error('SecurityError: storage is blocked'); };
  const storage = store ? {getItem: (k) => (k in store ? store[k] : null), setItem: (k, v) => { store[k] = String(v); }, removeItem: (k) => { delete store[k]; }}
    : {getItem: refuse, setItem: refuse, removeItem: refuse};
  const drawer = el('aside', 'drawer');
  drawer.append(el('div', 'matchEvents'), el('div', 'roster'), el('div', 'corrections'));
  body.append(transport, drawer, el('div', 'summary'), el('a', 'download'), el('div', 'matchNote'));
  // the answers of the routes: per route a list given in turn, the last one repeated
  const answers = Object.assign({
    index: ['@index'], intel: ['@intel'],
    build: [{'@intel': {state: 'building', build: {state: 'building', progress: .3, stage: 'Reading the match data'}, events: [], timeline: null, stats: null}}],
    halftime: [{ok: true, intel: {'@intel': {state: 'building', build: {state: 'building', progress: .8, stage: 'Recognising stoppages, restarts and goals'}, events: [], timeline: null}}, index: '@index'}],
    reset: [{ok: true, intel: {'@intel': {state: 'building', events: [], timeline: null}}, index: '@index'}],
    event: [{ok: true, correction: {cid: 'c-new'}, intel: '@intel'}], eventRemove: [{ok: true, intel: '@intel'}],
    correction: [{ok: true, correction: {id: 'r-new'}, index: '@index'}], correctionRemove: [{ok: true, index: '@index'}]}, copy(job.answers || {}));
  const ROUTES = {'/api/match/intel/build': 'build', '/api/match/halftime': 'halftime', '/api/match/halftime/reset': 'reset', '/api/match/event': 'event',
                  '/api/match/event/remove': 'eventRemove', '/api/match/correction': 'correction', '/api/match/correction/remove': 'correctionRemove'};
  const api = async (path, payload) => {
    if (payload === undefined) gets.push(path); else posts.push({url: path, body: copy(payload)});
    const p = path.split('?')[0], key = payload === undefined ? ({'/api/match/index': 'index', '/api/match/intel': 'intel'})[p] : ROUTES[p], list = key && answers[key];
    if (!list || !list.length) throw Object.assign(new Error('NOT FOUND ' + path), {status: 404});
    const a = list.length > 1 ? list.shift() : list[0];
    if (a && a.__status) throw Object.assign(new Error(a.__error || 'failed'), {status: a.__status});
    return token(a);
  };
  const fetch = async (url) => {
    const k = Number(new URLSearchParams(url.split('?')[1] || '').get('segment')), s = INDEX0.segments.find((x) => x.index === k);
    const view = {version: 3, matchView: {segment: k}, match: {players: [], goalkeepers: [], excluded: []}, roster: {slots: []}, tactical: {}, teams: {}, video: INDEX0.video, events: [],
                  frames: [{frame: s.cut0, time: s.start, people: []}, {frame: s.cut1 - 1, time: s.end - .04, people: []}]};
    return {ok: true, status: 200, statusText: 'OK', text: async () => JSON.stringify(view), json: async () => view};
  };
  const video = {currentTime: job.time, duration: INDEX0.video.duration, paused: false, pause() { this.paused = true; }, play() { this.paused = false; return Promise.resolve(); }};
  const quiet = {log() {}, info() {}, debug() {}, time() {}, timeEnd() {}, warn: (...a) => warns.push(a.map(String).join(' ')),
                 error: (...a) => warns.push('ERROR ' + a.map((x) => (x && x.stack) || String(x)).join(' '))};
  const sandbox = {
    document, console: quiet, performance, AbortController, URLSearchParams, fetch, api,
    setTimeout: (f) => { if (++timers > 5000) return 0; if (hold) { const h = {f, cleared: false}; held.push(h); return h; } return setImmediate(f); },
    clearTimeout: (h) => { if (!h) return; if (h.f) h.cleared = true; else clearImmediate(h); },
    localStorage: storage, addEventListener() {},
    info: {video: Object.assign({}, INDEX0.video), videoKey: '3a043f106ecf56c2' + '0'.repeat(48), uploads: []},
    video, result: null, presence: new Map(), pitchIndex: [], selected: null, busy: false, mode: null, loadedRun: null,
    tactical: {load() {}, timeline: null, colours: {A: '#b98cff', B: '#f6b564'}, invalidate() {}, ends: null, endsCalls: [],
               setGoalEnds(fn) { this.ends = typeof fn === 'function' ? fn : null; this.endsCalls.push(this.ends ? 'fn' : null); }},
    analytics: {load(o) { this.last = o; }}, teamShape: {load() {}}, possession: {load() {}}, coaching: {load() {}, mKits: job.kits || {A: null, B: null}},
    buildPresence: () => { sandbox.presence = new Map(); }, closeEditor() {}, clearCoachMark() {}, stageStatus() {}, matchMeta() {},
    playback: {seek: (t) => { seeks.push(t); video.currentTime = t; }},
    drawers: {current: job.drawer === undefined ? 'edit' : job.drawer, open() {}},
    select: (id) => { selects.push(id); sandbox.selected = id; },
    roster: () => sandbox.FullMatch.roster(byId('roster')),
    correctionList: () => sandbox.FullMatch.corrections(byId('corrections')),
    fitStage: () => { fits.push(1); },
  };
  sandbox.window = sandbox;
  vm.createContext(sandbox);
  if (job.realAnalytics) {
    // the page's own analytics panel (tactical.js + analytics.js) instead of the stand-in: its player list
    for (const path of [input.tactical, input.analytics]) vm.runInContext(fs.readFileSync(path, 'utf8'), sandbox, {filename: path});
    const ctl = (tag, value) => { const n = el(tag); n.value = value || ''; return n; };
    sandbox.analytics = sandbox.Analytics.create({canvas: el('canvas'), notice: el('div'), status: el('div'), legend: el('div'), insights: el('div'), api, video,
      board: sandbox.tactical, colourInputs: [], controls: {tabs: [], team: ctl('select', 'A'), player: ctl('select'), range: ctl('select', 'full'), custom: el('span'),
                                                           t0: ctl('input'), t1: ctl('input'), scope: ctl('select', 'all'), scopeWrap: el('span')}});
  }
  vm.runInContext(SOURCE, sandbox, {filename: 'match.js'});
  const fire = (n, type, props) => { if (!n) throw new Error('no element for ' + type + ' ' + JSON.stringify(props || {})); return n.dispatch(type, props || {}); };
  const rvRow = (s) => one(byId('roster'), (n) => has(n, 'rv-row') && (s.slot !== undefined ? n.dataset.rv === 'slot' && n.dataset.team === s.team && n.dataset.slot === String(s.slot) :
    s.person ? n.dataset.rv === 'unresolved' && n.dataset.person === s.person : n.dataset.rv === 'left-out' && n.dataset.fragment === s.fragment));
  const need = (n, what) => { if (!n) throw new Error('not found: ' + what); return n; };
  async function act(step) {
    const mtEl = byId('mt'), ev = byId('matchEvents');
    const handle = () => need(one(mtEl, (n) => has(n, 'mt-hh') && n.dataset.half === step.half && !n.hidden), 'handle ' + step.half);
    if (step.do === 'geometry') {
      const I = sandbox.FullMatch.intel;
      geo.push({toX: step.ts.map((t) => I.timeToX(t)), toT: step.xs.map((x) => I.xToTime(x)), pct: step.ts.map((t) => I.timePct(t))});
    } else if (step.do === 'marker') {
      const b = need(one(mtEl, (n) => has(n, 'mt-ev') && n.dataset.ev === step.id), 'marker ' + step.id);
      if (step.how === 'focus') b.focus();
      else if (step.how === 'enter') fire(b, 'pointerenter', {bubbles: false});
      else if (step.how === 'key') { const e = fire(b, 'keydown', {key: step.key}); keyLog.push({prevented: e.defaultPrevented, stopped: e.propagationStopped}); }
      else b.click();
    } else if (step.do === 'pointer') {
      const h = handle();
      if (step.down !== false) fire(h, 'pointerdown', {clientX: step.path[0], pointerId: 7, button: 0});
      for (const x of step.path.slice(step.down !== false ? 1 : 0)) fire(h, 'pointermove', {clientX: x, pointerId: 7});
      if (step.up !== false) fire(h, 'pointerup', {clientX: step.path[step.path.length - 1], pointerId: 7});
    } else if (step.do === 'key') {
      const h = handle();
      h.focus();
      const e = fire(h, 'keydown', {key: step.key, shiftKey: !!step.shift});
      keyLog.push({prevented: e.defaultPrevented, stopped: e.propagationStopped});
    } else if (step.do === 'pop-input') {
      const i = need(one(byId('mtPop'), (n) => n.tagName === 'INPUT' && n.dataset.half === step.half), 'input ' + step.half);
      i.focus(); i.value = step.value;
      if (step.how === 'change') fire(i, 'change'); else fire(i, 'keydown', {key: 'Enter'});
    } else if (step.do === 'pop') {
      const pop = byId('mtPop');
      const b = need(one(pop, (n) => (step.mt && n.dataset.mt === step.mt) || (step.side && n.dataset.side === step.side) || (step.now && n.tagName === 'BUTTON' && n.dataset.half === step.now)), 'pop ' + JSON.stringify(step));
      if (step.mt === 'break') { b.checked = !!step.on; fire(b, 'change'); } else b.click();
    } else if (step.do === 'press-outside') {
      fire(byId('summary'), 'pointerdown', {clientX: 0, pointerId: 9});
    } else if (step.do === 'event') {
      const row = need(one(ev, (n) => has(n, 'me-row') && n.dataset.ev === step.id), 'row ' + step.id), e2 = need(one(row, (n) => n.dataset.act === step.act), step.act);
      if (e2.tagName === 'SELECT') { e2.value = step.value; fire(e2, 'change'); } else e2.click();
    } else if (step.do === 'filter') {
      const s = need(one(ev, (n) => n.dataset.me === 'filter'), 'filter'); s.value = step.value; fire(s, 'change');
    } else if (step.do === 'add') {
      const t = need(one(ev, (n) => n.dataset.add === 'type'), 'type'); t.value = step.type; fire(t, 'change');
      const tm = need(one(ev, (n) => n.dataset.add === 'team'), 'team');
      if (step.team !== undefined) { tm.value = step.team; fire(tm, 'change'); }
      need(one(ev, (n) => n.dataset.act === 'add'), 'add').click();
    } else if (step.do === 'undo') {
      need(one(ev, (n) => n.dataset.act === 'undo' && (!step.cid || n.dataset.cid === step.cid)), 'undo').click();
    } else if (step.do === 'time') {
      video.currentTime = step.t; sandbox.FullMatch.frame(step.t);
    } else if (step.do === 'drawer') {
      sandbox.drawers.current = step.page;
    } else if (step.do === 'roster') {
      sandbox.roster();
    } else if (step.do === 'all') {
      need(one(byId('roster'), (n) => n.dataset.act === 'all'), 'all').click();
    } else if (step.do === 'rv') {
      if (step.open) need(one(need(rvRow(step), 'row'), (n) => n.dataset.act === 'correct'), 'correct').click();
      const target = need(one(need(rvRow(step), 'row'), (n) => n.dataset.act === step.act), step.act);
      if (target.tagName === 'SELECT') {
        target.value = String(step.value); fire(target, 'change');
        if (step.then) need(one(need(rvRow(step), 'row'), (n) => n.dataset.act === step.then), step.then).click();
      } else target.click();
    } else if (step.do === 'release') {
      hold = !!step.hold;
      const list = held.splice(0);
      for (const h of list) if (!h.cleared) h.f();
    } else if (step.do === 'hover') {
      fire(range, 'pointermove', {clientX: step.x});
    } else if (step.do === 'leave') {
      sandbox.FullMatch.leave({reload: false});
    } else if (step.do === 'an-team') {
      const c = sandbox.analytics.c; c.team.value = step.team; fire(c.team, 'change');
    } else if (step.do === 'mtf') {
      byId('mtfBtn').click();
    } else if (step.do === 'mtf-type') {
      const box = need(one(byId('mtfPop'), (n) => n.tagName === 'INPUT' && n.dataset.type === step.type), 'checkbox ' + step.type);
      box.checked = !!step.on; fire(box, 'change');
    } else if (step.do === 'mtf-quick') {
      need(one(byId('mtfPop'), (n) => n.tagName === 'BUTTON' && n.dataset.mtf === step.key), 'quick ' + step.key).click();
    } else if (step.do === 'mtf-strip') {
      const box = need(one(byId('mtfPop'), (n) => n.dataset.mtf === 'strip'), 'strip'); box.checked = !!step.on; fire(box, 'change');
    } else if (step.do === 'mtf-key') {
      const e = fire(active || byId('mtfBtn'), 'keydown', {key: step.key}); keyLog.push({prevented: e.defaultPrevented, stopped: e.propagationStopped});
    } else if (step.do === 'lane') {
      need(one(mtEl, (n) => has(n, 'mt-scores')), 'scores lane').width = step.width;
    } else if (step.do === 'rects') {
      // the layout's boxes (getBoundingClientRect) of the score lane and the halftime handles
      if (step.lane) need(one(mtEl, (n) => has(n, 'mt-scores')), 'scores lane').rect = step.lane;
      for (const [which, r] of Object.entries(step.handles || {})) need(one(mtEl, (n) => has(n, 'mt-hh') && n.dataset.half === which), 'handle ' + which).rect = r;
    } else throw new Error('unknown step ' + step.do);
  }
  function snap() {
    const mtEl = byId('mt'), pop = byId('mtPop'), tip = byId('mtTip'), ev = byId('matchEvents'), ro = byId('roster');
    const hiddenUp = (n, root) => { for (let x = n; x && x !== root; x = x.parentElement) if (x.hidden) return true; return false; };
    const marks = find(mtEl, (n) => has(n, 'mt-ev')).map((n) => ({id: n.dataset.ev, cls: n.className, left: n.style.left, t: Number(n.dataset.t), tab: n.tabIndex,
                                                                 label: n.attrs['aria-label'] || '', transform: n.style.transform || '', team: n.style['--team'] || null}));
    const runs = find(mtEl, (n) => has(n, 'mt-run')).map((n) => ({state: n.dataset.state, cls: n.className, left: n.style.left, width: n.style.width}));
    const handles = find(mtEl, (n) => has(n, 'mt-hh') && !n.hidden).map((n) => ({half: n.dataset.half, text: n.textContent, left: n.style.left, cls: n.className, role: n.attrs.role, tab: n.tabIndex,
      now: Number(n.attrs['aria-valuenow']), min: Number(n.attrs['aria-valuemin']), max: Number(n.attrs['aria-valuemax']), valuetext: n.attrs['aria-valuetext'], label: n.attrs['aria-label']}));
    const lines = find(mtEl, (n) => has(n, 'mt-hline') && !n.hidden).map((n) => ({half: n.dataset.half, left: n.style.left, cls: n.className}));
    const band = one(mtEl, (n) => has(n, 'mt-band'));
    let popOut = null;
    if (pop && !pop.hidden) {
      const bt = (key) => { const b = one(pop, (n) => n.dataset.mt === key); return b ? {text: b.textContent, disabled: !!b.disabled, hidden: hiddenUp(b, pop)} : null; };
      const shown = (cls) => { const e = one(pop, (n) => has(n, cls)); return e && !e.hidden ? e.textContent : ''; };
      popOut = {text: pop.textContent, brk: !!(one(pop, (n) => n.dataset.mt === 'break') || {}).checked, save: bt('save'), cancel: bt('cancel'), reset: bt('reset'),
                inputs: find(pop, (n) => n.tagName === 'INPUT' && n.type === 'text').map((n) => ({half: n.dataset.half, value: n.value, hidden: hiddenUp(n, pop), invalid: n.attrs['aria-invalid'] || null, focused: active === n})),
                sides: find(pop, (n) => n.dataset.side).map((n) => [n.dataset.side, n.attrs['aria-pressed']]),
                state: shown('mt-state'), err: shown('mt-err'), note: shown('mt-note'), check: shown('mt-check'), hint: shown('mt-hint')};
    }
    const evOut = {
      text: text(ev), status: text(one(ev, (n) => has(n, 'me-status'))), stats: text(one(ev, (n) => has(n, 'me-stats'))), dir: text(one(ev, (n) => has(n, 'me-dir'))),
      note: text(one(ev, (n) => has(n, 'me-note'))), msgs: find(ev, (n) => has(n, 'me-msg')).map(text),
      filter: (() => { const s = one(ev, (n) => n.dataset.me === 'filter'); return s ? {value: s.value, options: s.children.map((o) => [o.value, o.textContent])} : null; })(),
      rows: find(ev, (n) => has(n, 'me-row')).map((r) => ({id: r.dataset.ev, cls: r.className, head: text(one(r, (n) => has(n, 'me-head'))),
        acts: find(r, (n) => n.dataset.act).map((n) => ({act: n.dataset.act, disabled: !!n.disabled, text: n.textContent, value: n.tagName === 'SELECT' ? n.value : null})),
        evidence: (() => { const u = one(r, (n) => n.tagName === 'UL'); return u ? {hidden: u.hidden, items: u.children.map(text)} : null; })()})),
      add: (() => { const t = one(ev, (n) => n.dataset.add === 'type'), tm = one(ev, (n) => n.dataset.add === 'team'), b = one(ev, (n) => n.dataset.act === 'add');
                    return t ? {type: t.value, team: tm.value, teams: tm.children.map((o) => o.value), disabled: !!b.disabled} : null; })(),
      fixes: find(ev, (n) => has(n, 'me-fix')).map(text)};
    const rv = one(ro, (n) => has(n, 'rv'));
    const rvOut = rv ? {
      title: text(one(rv, (n) => has(n, 'rv-title'))), heads: find(rv, (n) => n.tagName === 'H4').map(text),
      slots: find(rv, (n) => has(n, 'rv-row') && n.dataset.rv === 'slot').map((r) => { const who = one(r, (n) => has(n, 'rv-who'));
        return {team: r.dataset.team, slot: r.dataset.slot, who: text(who), id: who ? who.dataset.id : null, meta: find(r, (n) => has(n, 'rv-meta')).map(text),
                none: text(one(r, (n) => has(n, 'rv-none'))), subs: find(r, (n) => n.dataset.rv === 'sub').map(text), acts: find(r, (n) => n.dataset.act).map((n) => n.dataset.act)}; }),
      unresolved: find(rv, (n) => has(n, 'rv-row') && n.dataset.rv === 'unresolved').map((r) => ({person: r.dataset.person, text: r.textContent})),
      leftOut: find(rv, (n) => has(n, 'rv-row') && n.dataset.rv === 'left-out').map((r) => ({fragment: r.dataset.fragment, text: r.textContent}))} : null;
    const fbox = byId('mtf'), fbtn = byId('mtfBtn'), fpop = byId('mtfPop');
    const filterOut = {hidden: fbox.hidden, text: text(byId('mtfText')), expanded: fbtn.attrs['aria-expanded'] || null, label: fbtn.attrs['aria-label'] || null, part: has(fbox, 'part'),
      pop: fpop.hidden, head: text(one(fpop, (n) => has(n, 'mtf-head'))), names: find(fpop, (n) => has(n, 'mtf-name')).map(text),
      types: find(fpop, (n) => n.tagName === 'INPUT' && n.dataset.type).map((n) => [n.dataset.type, n.checked]),
      counts: find(fpop, (n) => has(n, 'mtf-row') && !has(n, 'mtf-band')).map((r) => [text(one(r, (n) => has(n, 'mtf-n'))), has(r, 'none')]),
      icons: find(fpop, (n) => has(n, 'me-key')).map((n) => n.className), strip: !!(one(fpop, (n) => n.dataset.mtf === 'strip') || {}).checked,
      quick: find(fpop, (n) => n.tagName === 'BUTTON' && n.dataset.mtf).map((n) => [n.dataset.mtf, n.textContent, n.attrs['aria-pressed'] || null])};
    const scoresOut = find(mtEl, (n) => has(n, 'mt-score')).map((n) => ({ev: n.dataset.ev, text: n.textContent, left: n.style.left, transform: n.style.transform || ''}));
    const stripLane = one(mtEl, (n) => has(n, 'mt-strip'));
    const S = sandbox.FullMatch.intel.state, T = sandbox.tactical, A = sandbox.analytics && sandbox.analytics.c;
    const ends = T.ends ? Object.fromEntries((job.endsAt || []).map((t) => [String(t), copy(T.ends(t))])) : null;
    const analyticsOut = A ? {team: A.team.value, options: A.player.children.map((o) => [o.value, o.textContent])} :
      sandbox.analytics.last ? {goalEnds: copy((sandbox.analytics.last.tactical || {}).goalEnds || null)} : null;
    return {filter: filterOut, scores: scoresOut, stripHidden: stripLane ? stripLane.hidden : null, storage: store ? copy(store) : 'throw',
            focus: active ? (active.id || active.dataset.type || active.dataset.ev || active.tagName) : null,
            mtHidden: mtEl.hidden, on: has(transport, 'mt-on'), stale: has(mtEl, 'stale'), marks, runs, handles, lines, ends, endsCalls: T.endsCalls.slice(), analytics: analyticsOut,
            band: band ? {hidden: band.hidden, left: band.style.left, width: band.style.width} : null, pop: popOut,
            tip: tip && !tip.hidden ? tip.children.map(text) : null, msg: msg.hidden ? null : {text: msg.textContent, cls: msg.className, title: msg.title},
            events: evOut, review: rvOut, rosterText: text(ro), corrections: text(byId('corrections')), list: find(ro, (n) => has(n, 'mr')).length,
            posts: copy(posts), gets: gets.slice(), seeks: seeks.slice(), selects: selects.slice(), video: {t: video.currentTime, paused: video.paused}, fits: fits.length,
            phase: S.phase, error: S.error, off: S.off, draft: copy(S.draft), active: active ? {tag: active.tagName, half: active.dataset.half || null, ev: active.dataset.ev || null} : null,
            keys: keyLog.slice(), geo: copy(geo), warns: warns.slice()};
  }
  await sandbox.FullMatch.open(job.match);
  await settle();
  const out = {first: snap()};
  for (const step of job.steps || []) { await act(step); await settle(); if (step.name) out[step.name] = snap(); }
  out.last = snap();
  return out;
}
(async () => {
  const out = [];
  for (const job of input.jobs) out.push(await scenario(job));
  process.stdout.write(JSON.stringify(out), () => process.exit(0));
})().catch((error) => { process.stderr.write(String((error && error.stack) || error)); process.exit(1); });
"""


def job(**kw):
    """One page scenario: match.js opens the match at `time` (s), the routes answer from `answers` (per route a list
    given in turn, the last one repeated; '@index' and '@intel' are the fixtures), then the steps run; a named step keeps
    a snapshot. hold: polls wait until a 'release' step. storage: the page's localStorage at the start; ALL_MARKERS here
    by default (every kind of event on the timeline, as these scenarios were written before the timeline filter), {} a
    first visit (the page's default: goals only), 'throw' storage that refuses every access."""
    return dict({'match': MATCH, 'time': 2000.0, 'answers': {}, 'steps': [], 'hold': False, 'kits': None, 'storage': ALL_MARKERS}, **kw)


def run(*jobs):
    done = subprocess.run([NODE, '-e', HARNESS], input=json.dumps({'match': MATCH_JS, 'tactical': TACTICAL_JS, 'analytics': ANALYTICS_JS, 'index': INDEX_GZ, 'intel': STATE_GZ,
                                                                    'jobs': list(jobs)}),
                          capture_output=True, text=True, encoding='utf-8', timeout=600)
    if done.returncode:
        raise AssertionError(done.stderr[-4000:])
    return json.loads(done.stdout)


def errors(snap):
    return [w for w in snap['warns'] if w.startswith('ERROR')]


def post_bodies(snap, url):
    return [p['body'] for p in snap['posts'] if p['url'] == url]


def building(progress, stage, **extra):
    return {'@intel': dict({'state': 'building', 'build': {'state': 'building', 'progress': progress, 'stage': stage}, 'events': [], 'timeline': None, 'stats': None}, **extra)}


@unittest.skipUnless(NODE, 'needs node to run match.js')
class Transport(unittest.TestCase):
    """The strip, the markers, the tooltip and the seek on the transport, from the real state of the match."""

    @classmethod
    def setUpClass(cls):
        base = job(steps=[
            {'do': 'geometry', 'ts': [0, 2854, DUR, 99999], 'xs': [50, 107, 600, 1093, 2000], 'name': 'geo'},
            {'do': 'hover', 'x': to_x(40), 'name': 'hover'},
            {'do': 'marker', 'id': 'goal-5080', 'how': 'enter', 'name': 'tip'},
            {'do': 'marker', 'id': 'goal-5080', 'name': 'clicked'},
            {'do': 'marker', 'id': 'goal-5080', 'how': 'focus'},
            {'do': 'marker', 'id': 'goal-5080', 'how': 'key', 'key': 'ArrowRight', 'name': 'keyed'},
            {'do': 'marker', 'id': 'substitution-54514', 'how': 'enter', 'name': 'close'}])
        removed = job(answers={'intel': [{'@intel': {'__events': {'goal-12780': {'status': 'removed'}, THROW: {'status': 'confirmed'}}}}]})
        kits = job(kits={'A': {'label': 'Team A (white)'}, 'B': {'label': 'Team B (red)'}}, steps=[{'do': 'marker', 'id': 'goal-41824', 'how': 'enter', 'name': 'tip'}])
        cls.base, cls.removed, cls.kits = run(base, removed, kits)

    def test_time_and_x_on_the_seek_bar(self):
        g = self.base['geo']['geo'][0]
        for t, x in zip([0, 2854, DUR, 99999], g['toX']):
            self.assertAlmostEqual(x, to_x(t), places=6, msg=f'time {t}: the thumb centre, inset by half the 14 px thumb')
        self.assertAlmostEqual(g['toX'][0], 107)
        self.assertAlmostEqual(g['toX'][2], 1093)
        for x, t in zip([50, 107, 600, 1093, 2000], g['toT']):
            self.assertAlmostEqual(t, to_t(x), places=6, msg=f'x {x}')
        self.assertEqual((g['toT'][0], g['toT'][4]), (0, DUR), 'clamped to the video')
        for t, p in zip([0, 2854, DUR], g['pct']):
            self.assertAlmostEqual(p, t / DUR * 100, places=9)
        with open(PAGE, encoding='utf-8') as handle:
            page = handle.read()
        self.assertIn('.mt-lane{position:absolute;left:7px;right:7px;', page, 'the layers stand in the thumb-inset lane')
        self.assertIn('::-webkit-slider-thumb{-webkit-appearance:none;width:14px;', page, 'the thumb the inset is half of')

    def test_markers_are_the_events_not_removed_and_candidates_are_hollow(self):
        o = self.base['first']
        self.assertEqual(errors(o), [])
        self.assertTrue(o['on'] and not o['mtHidden'], 'the layers are shown on the transport in match playback')
        self.assertGreater(o['fits'], 0, 'the stage fits the taller transport')
        self.assertEqual([m['id'] for m in o['marks']], [e['id'] for e in EVENTS])
        for m in o['marks']:
            e = BY_ID[m['id']]
            with self.subTest(event=e['id']):
                kind = KIND.get(e['type'], 'restart')
                self.assertIn(f'k-{kind}', m['cls'].split())
                self.assertIn(f"st-{e['status']}", m['cls'].split())
                self.assertIn('big' if kind in ('goal', 'sub') else 'small', m['cls'].split())
                self.assertAlmostEqual(num(m['left']), e['t'] / DUR * 100, places=3)
                self.assertIn(clock(e['t']), m['label'])
        cands = [m for m in o['marks'] if 'st-candidate' in m['cls']]
        self.assertEqual(len(cands), sum(1 for e in EVENTS if e['status'] == 'candidate'))
        self.assertEqual(len(cands), 32)
        goals = [m for m in o['marks'] if 'k-goal' in m['cls']]
        self.assertEqual([g['team'] for g in goals], ['#b98cff', '#b98cff', '#f6b564', '#f6b564'], 'a goal ringed in the scoring team\'s board colour')
        first = next(e for e in EVENTS if e['t'] >= 2000)
        self.assertEqual([m['id'] for m in o['marks'] if m['tab'] == 0], [first['id']], 'one tab stop: the next event after the playback time')
        r = self.removed['first']
        ids = [m['id'] for m in r['marks']]
        self.assertNotIn('goal-12780', ids, 'a removed event is not drawn')
        self.assertEqual(len(ids), len(EVENTS) - 1)
        self.assertIn('st-confirmed', next(m for m in r['marks'] if m['id'] == THROW)['cls'])

    def test_close_large_markers_stand_apart(self):
        o = self.base['first']
        subs = {m['id']: m for m in o['marks']}
        self.assertEqual(subs['substitution-54514']['transform'], '', 'the first of two substitutions 2 s apart')
        self.assertEqual(subs['substitution-54536']['transform'], 'translateX(7px)', 'the second stands 7 px aside')
        tip = self.base['close']['tip']
        self.assertIn(f'Also here: Substitution {clock(5453.58)}', tip[-1], 'the tooltip names the event drawn beside it')

    def test_the_state_strip(self):
        o = self.base['first']
        want = runs_of(INTEL['timeline'])
        self.assertEqual(len(o['runs']), len(want))
        self.assertLess(len(want), len(INTEL['timeline']), 'consecutive intervals of one state are merged')
        for got, r in zip(o['runs'], want):
            self.assertEqual(got['state'], r['state'])
            self.assertIn(STATE_CLASS[r['state']], got['cls'].split())
            self.assertAlmostEqual(num(got['left']), r['t0'] / DUR * 100, places=3)
            self.assertAlmostEqual(num(got['width']), (r['t1'] - r['t0']) / DUR * 100, places=3)
        self.assertEqual({r['state'] for r in o['runs']}, {'IN_PLAY', 'POSSIBLE_STOPPAGE', 'OUT_OF_PLAY', 'RESTART_PREPARATION', 'UNKNOWN', 'HALFTIME'})

    def test_the_tooltip(self):
        tip = self.base['tip']['tip']
        e = BY_ID['goal-5080']
        self.assertEqual(tip[0], f'Goal {DOT} {clock(e["t"])}')
        self.assertEqual(tip[1], 'Team A scored')
        self.assertEqual(tip[2], 'Score after the goal: Team A 1-0 Team B', 'a goal adds the score after it')
        self.assertEqual(tip[3], f'High confidence {DOT} detected')
        self.assertEqual(tip[4:-1], [f'{chr(0x2022)} {line}' for line in e['evidence']])
        self.assertEqual(tip[-1], f'Goal analysis: Coaching {SEP} Key moments {SEP} Goals', 'and where its analysis is')
        kit = self.kits['tip']['tip']
        self.assertEqual(kit[1], 'Team B (red) scored', 'the team with its kit label when the page has one')
        self.assertEqual(kit[2], 'Score after the goal: Team A (white) 2-1 Team B (red)')
        self.assertIn('The time is uncertain', run(job(steps=[{'do': 'marker', 'id': 'kick_off-28934', 'how': 'enter', 'name': 't'}]))[0]['t']['tip'])
        hover = self.base['hover']['tip']
        self.assertEqual(hover, [f'00:40 {DOT} Possible stoppage', 'Players walking, the ball not dead', 'Uncertain: worth a look'], 'the range shows the time and state under the pointer')

    def test_a_click_seeks_through_the_playback_controller(self):
        c = self.base['clicked']
        self.assertEqual(c['seeks'], [BY_ID['goal-5080']['t']])
        self.assertEqual(c['video'], {'t': BY_ID['goal-5080']['t'], 'paused': True})
        self.assertIsNone(c['tip'], 'the tooltip goes')

    def test_the_arrow_keys_move_along_the_markers(self):
        k = self.base['keyed']
        nxt = EVENTS[[e['id'] for e in EVENTS].index('goal-5080') + 1]['id']
        self.assertEqual(k['active']['ev'], nxt)
        self.assertEqual([m['id'] for m in k['marks'] if m['tab'] == 0], [nxt], 'the roving tab stop follows the focus')
        self.assertEqual(k['keys'][-1], {'prevented': True, 'stopped': True}, 'the page does not also step the video')
        self.assertEqual(k['tip'][0], f"{WORDS[BY_ID[nxt]['type']]} {DOT} {clock(BY_ID[nxt]['t'])}", 'and shows its tooltip')


EXTRA_GOAL = {'id': 'goal-5380', 'type': 'GOAL', 't': 537.99, 'team': 'B', 'status': 'detected', 'confidence': 'medium', 'timeUncertain': False, 'evidence': []}


def ids_of(*types):
    return [e['id'] for e in EVENTS if e['type'] in types]


@unittest.skipUnless(NODE, 'needs node to run match.js')
class TimelineFilter(unittest.TestCase):
    """The timeline filter beside the range (goals only by default, kept in localStorage) and the goal marker with its
    running score, on the real state of the match."""

    @classmethod
    def setUpClass(cls):
        first = job(storage={}, steps=[
            {'do': 'marker', 'id': 'goal-5080', 'how': 'focus'},
            {'do': 'marker', 'id': 'goal-5080', 'how': 'key', 'key': 'ArrowRight', 'name': 'roved'},
            {'do': 'marker', 'id': 'goal-12780', 'how': 'key', 'key': 'End', 'name': 'end'},
            {'do': 'marker', 'id': 'goal-41824', 'how': 'enter', 'name': 'tip'},
            {'do': 'mtf', 'name': 'opened'},
            {'do': 'mtf-type', 'type': 'FREE_KICK', 'on': True, 'name': 'free'},
            {'do': 'mtf-type', 'type': 'GOAL', 'on': False, 'name': 'noGoals'},
            {'do': 'mtf-quick', 'key': 'all', 'name': 'all'},
            {'do': 'mtf-quick', 'key': 'none', 'name': 'none'},
            {'do': 'mtf-quick', 'key': 'goals', 'name': 'goals'},
            {'do': 'mtf-strip', 'on': False, 'name': 'noStrip'},
            {'do': 'hover', 'x': to_x(40), 'name': 'hover'},
            {'do': 'mtf-strip', 'on': True, 'name': 'strip'},
            {'do': 'mtf-key', 'key': 'Escape', 'name': 'escaped'},
            {'do': 'mtf', 'name': 'again'},
            {'do': 'press-outside', 'name': 'outside'}])
        later = job(storage=stored(['FREE_KICK', 'SUBSTITUTION'], strip=False))
        blocked = job(storage='throw', steps=[{'do': 'mtf'}, {'do': 'mtf-type', 'type': 'THROW_IN', 'on': True, 'name': 'throws'}])
        unreadable = job(storage={MARKS_KEY: 'not json'})
        junk = job(storage={MARKS_KEY: json.dumps({'show': ['GOAL', 'YELLOW_CARD', 7], 'strip': 'yes'})})
        near_all = job(steps=[{'do': 'marker', 'id': 'foul_or_whistle_stoppage-26407', 'how': 'enter', 'name': 'tip'}])
        near_some = job(storage=stored(['FOUL_OR_WHISTLE_STOPPAGE', 'THROW_IN']), steps=[{'do': 'marker', 'id': 'foul_or_whistle_stoppage-26407', 'how': 'enter', 'name': 'tip'}])
        crowded = job(storage={}, answers={'intel': [{'@intel': {'events': INTEL['events'] + [EXTRA_GOAL]}}]},
                      steps=[{'do': 'lane', 'width': 300}, {'do': 'mtf-quick', 'key': 'goals', 'name': 'narrow'}])
        edited = job(storage={}, answers={'intel': [{'@intel': {'__events': {'goal-12780': {'status': 'removed'}, 'goal-41824': {'status': 'candidate'}}}}]})
        box = lambda left, width: {'left': left, 'top': 0, 'width': width, 'height': 15, 'right': left + width, 'bottom': 15}
        handled = job(storage={}, steps=[{'do': 'rects', 'lane': box(107, 986), 'handles': {'end': box(400, 140), 'start': box(790, 60)}},
                                         {'do': 'mtf-quick', 'key': 'goals', 'name': 'laid'}])
        (cls.first, cls.later, cls.blocked, cls.unreadable, cls.junk, cls.near_all, cls.near_some, cls.crowded, cls.edited, cls.handled) = run(
            first, later, blocked, unreadable, junk, near_all, near_some, crowded, edited, handled)

    def test_a_first_visit_shows_only_the_goals(self):
        o = self.first['first']
        self.assertEqual(errors(o), [])
        self.assertEqual([m['id'] for m in o['marks']], [e['id'] for e in GOALS], 'goals only: the clean-up')
        self.assertEqual(len(GOALS), 4)
        f = o['filter']
        self.assertEqual((f['hidden'], f['text'], f['expanded'], f['pop']), (False, 'Goals only', 'false', True), 'beside the range, saying what it shows')
        self.assertEqual(f['label'], 'Timeline markers: Goals only')
        self.assertTrue(f['part'], 'it hides some kinds')
        self.assertTrue(o['runs'] and o['stripHidden'] is False, 'the play-state band as before')
        self.assertEqual([m['id'] for m in o['marks'] if m['tab'] == 0], ['goal-41824'], 'the one tab stop: the next goal after the playback time')
        self.assertEqual(o['storage'], {}, 'nothing is written before a change')

    def test_the_popover(self):
        o = self.first['opened']
        f = o['filter']
        self.assertEqual((f['pop'], f['expanded']), (False, 'true'))
        self.assertEqual(o['focus'], 'GOAL', 'the focus on its first checkbox')
        self.assertEqual(f['names'], ['Goals', 'Kick-offs', 'Corner kicks', 'Goal kicks', 'Throw-ins', 'Free kicks', 'Penalty kicks', 'Unknown restarts',
                                      'Substitutions', 'Fouls or whistle stoppages', 'Other stoppages', 'Play-state band'])
        self.assertEqual(f['types'], [[t, t == 'GOAL'] for t in MARK_TYPES])
        counts = collections.Counter(e['type'] for e in EVENTS if e['status'] != 'removed')
        self.assertEqual(f['counts'], [[str(counts[t]), not counts[t]] for t in MARK_TYPES], 'how many of each kind the match has (none: dimmed)')
        self.assertEqual(f['counts'][MARK_TYPES.index('PENALTY_KICK')], ['0', True])
        self.assertEqual(f['icons'], ['me-key k-goal', 'me-key k-ko'] + ['me-key k-restart'] * 6 + ['me-key k-sub', 'me-key k-stop', 'me-key k-stop'],
                         'each kind with its marker')
        self.assertTrue(f['strip'])
        self.assertEqual(f['quick'], [['goals', 'Goals only', 'true'], ['all', 'All', 'false'], ['none', 'None', 'false']])
        self.assertIn('TIMELINE MARKERS', f['head'])

    def test_a_kind_turned_on_or_off_is_drawn_at_once_and_kept(self):
        free = self.first['free']
        self.assertEqual([m['id'] for m in free['marks']], ids_of('GOAL', 'FREE_KICK'), 'the free kicks join the goals, in time order')
        self.assertEqual(free['filter']['text'], f'2 of {len(MARK_TYPES)}')
        self.assertEqual(json.loads(free['storage'][MARKS_KEY]), {'show': ['GOAL', 'FREE_KICK'], 'strip': True}, 'one versioned key')
        self.assertEqual(free['filter']['quick'][0][2], 'false')
        no = self.first['noGoals']
        self.assertEqual([m['id'] for m in no['marks']], ids_of('FREE_KICK'))
        self.assertEqual((no['scores'], no['filter']['text']), ([], f'1 of {len(MARK_TYPES)}'), 'no goal drawn: no score labels')
        later = self.later['first']
        self.assertEqual([m['id'] for m in later['marks']], ids_of('FREE_KICK', 'SUBSTITUTION'), 'the choice of an earlier visit')
        self.assertEqual((later['runs'], later['stripHidden']), ([], True))
        self.assertEqual(later['filter']['types'], [[t, t in ('FREE_KICK', 'SUBSTITUTION')] for t in MARK_TYPES])
        self.assertEqual(later['filter']['label'], f'Timeline markers: 2 of {len(MARK_TYPES)}, play-state band hidden')
        self.assertEqual(errors(later), [])

    def test_all_none_and_goals_only(self):
        a = self.first['all']
        self.assertEqual([m['id'] for m in a['marks']], [e['id'] for e in EVENTS])
        self.assertEqual(a['filter']['text'], 'All markers')
        self.assertFalse(a['filter']['part'])
        self.assertEqual([q[2] for q in a['filter']['quick']], ['false', 'true', 'false'])
        self.assertEqual(json.loads(a['storage'][MARKS_KEY])['show'], MARK_TYPES)
        n = self.first['none']
        self.assertEqual((n['marks'], n['scores'], n['filter']['text']), ([], [], 'No markers'))
        self.assertEqual([q[2] for q in n['filter']['quick']], ['false', 'false', 'true'])
        self.assertEqual(json.loads(n['storage'][MARKS_KEY]), {'show': [], 'strip': True})
        g = self.first['goals']
        self.assertEqual([m['id'] for m in g['marks']], [e['id'] for e in GOALS])
        self.assertEqual(g['filter']['text'], 'Goals only')
        self.assertTrue(g['runs'], 'the band is a choice of its own')

    def test_the_play_state_band_toggle(self):
        off = self.first['noStrip']
        self.assertEqual((off['runs'], off['stripHidden']), ([], True))
        self.assertEqual([m['id'] for m in off['marks']], [e['id'] for e in GOALS], 'the markers stay')
        self.assertEqual(json.loads(off['storage'][MARKS_KEY]), {'show': ['GOAL'], 'strip': False})
        self.assertTrue(off['filter']['label'].endswith('play-state band hidden'))
        self.assertFalse(off['filter']['strip'])
        self.assertEqual(self.first['hover']['tip'], ['00:40'], 'the range says the time, not the state the band no longer shows')
        on = self.first['strip']
        self.assertEqual(len(on['runs']), len(runs_of(INTEL['timeline'])))
        self.assertFalse(on['stripHidden'])

    def test_escape_and_a_press_elsewhere_close_it(self):
        e = self.first['escaped']
        self.assertEqual((e['filter']['pop'], e['filter']['expanded'], e['focus']), (True, 'false', 'mtfBtn'), 'Escape: closed, the focus back on the button')
        self.assertEqual(e['keys'][-1], {'prevented': True, 'stopped': True}, 'the page does not also leave full screen or close a drawer')
        self.assertFalse(self.first['again']['filter']['pop'])
        self.assertTrue(self.first['outside']['filter']['pop'])
        self.assertEqual(errors(self.first['last']), [])

    def test_storage_that_throws_or_lies(self):
        b = self.blocked
        self.assertEqual([m['id'] for m in b['first']['marks']], [e['id'] for e in GOALS], 'storage blocked: the default')
        self.assertEqual([m['id'] for m in b['throws']['marks']], ids_of('GOAL', 'THROW_IN'), 'and the filter still works for the page')
        self.assertEqual(b['throws']['filter']['text'], f'2 of {len(MARK_TYPES)}')
        self.assertEqual(b['throws']['storage'], 'throw')
        self.assertEqual(errors(b['last']), [])
        for out in (self.unreadable, self.junk):
            o = out['first']
            self.assertEqual([m['id'] for m in o['marks']], [e['id'] for e in GOALS], 'unreadable or unknown kinds: goals only')
            self.assertTrue(o['runs'], 'a strip that is not true or false: the band')
            self.assertEqual(errors(o), [])

    def test_hidden_markers_take_no_part_in_the_keys_or_also_here(self):
        roved = self.first['roved']
        self.assertEqual(roved['active']['ev'], 'goal-12780', 'the arrow moves to the next goal, over the hidden kick-offs and restarts')
        self.assertEqual([m['id'] for m in roved['marks'] if m['tab'] == 0], ['goal-12780'])
        self.assertEqual(roved['keys'][-1], {'prevented': True, 'stopped': True})
        self.assertEqual(self.first['end']['active']['ev'], GOALS[-1]['id'], 'End: the last marker drawn')
        tip = self.near_all['tip']['tip']
        self.assertEqual(tip[-1], f'Also here: Free kick {clock(2647.31)}, Throw-in {clock(2665.91)}')
        tip = self.near_some['tip']['tip']
        self.assertEqual(tip[-1], f'Also here: Throw-in {clock(2665.91)}', 'a hidden kind is not named')
        self.assertEqual([m['id'] for m in self.near_some['first']['marks']], ids_of('FOUL_OR_WHISTLE_STOPPAGE', 'THROW_IN'))

    def test_the_goal_marker_and_its_score(self):
        o = self.first['first']
        for m, e in zip(o['marks'], GOALS):
            with self.subTest(goal=e['id']):
                self.assertEqual(m['cls'].split(), ['mt-ev', 'k-goal', f"st-{e['status']}", 'big'])
                self.assertAlmostEqual(num(m['left']), e['t'] / DUR * 100, places=3)
        self.assertEqual([m['team'] for m in o['marks']], ['#b98cff', '#b98cff', '#f6b564', '#f6b564'], "ringed in the scoring team's colour")
        self.assertEqual([(s['ev'], s['text']) for s in o['scores']], [('goal-5080', '1-0'), ('goal-12780', '2-0'), ('goal-41824', '2-1'), ('goal-46834', '2-2')],
                         'the running score, team A first')
        for s, e in zip(o['scores'], GOALS):
            self.assertAlmostEqual(num(s['left']), e['t'] / DUR * 100, places=3, msg='above its goal')
        self.assertIn('score Team A 2-1 Team B', next(m for m in o['marks'] if m['id'] == 'goal-41824')['label'], 'the marker says the score')
        tip = self.first['tip']['tip']
        self.assertEqual(tip[:3], [f'Goal {DOT} {clock(4182.38)}', 'Team B scored', 'Score after the goal: Team A 2-1 Team B'])
        self.assertEqual(tip[-1], f'Goal analysis: Coaching {SEP} Key moments {SEP} Goals')

    def test_a_score_label_only_where_there_is_room(self):
        c = self.crowded['first']
        self.assertEqual(errors(c), [])
        marks = {m['id']: m for m in c['marks']}
        self.assertEqual(marks['goal-5380']['transform'], 'translateX(7px)', 'a goal 30 s after another stands aside')
        self.assertIn('score Team A 1-1 Team B', marks['goal-5380']['label'])
        self.assertEqual([(s['ev'], s['text']) for s in c['scores']], [('goal-5080', '1-0'), ('goal-12780', '2-1'), ('goal-41824', '2-2'), ('goal-46834', '2-3')],
                         'the second of two close goals has no label: it would overlap the first')
        narrow = self.crowded['narrow']
        self.assertEqual([s['ev'] for s in narrow['scores']], ['goal-5080', 'goal-12780', 'goal-41824'],
                         f'a 300 px lane: {SCORE_PX} px of room each (78:03 stands 26 px after 69:42)')
        # laid out: 69:42 stands at x 819 of a 986 px lane from x 107, under the SECOND HALF START handle (790-850)
        laid = self.handled['laid']
        self.assertEqual([s['ev'] for s in laid['scores']], ['goal-5080', 'goal-12780', 'goal-46834'], 'no label where a halftime handle would hide it')
        self.assertEqual(len(laid['marks']), 4, 'the goal itself is drawn')

    def test_removed_goals_do_not_count_and_candidates_stay_hollow(self):
        o = self.edited['first']
        self.assertEqual([m['id'] for m in o['marks']], ['goal-5080', 'goal-41824', 'goal-46834'])
        self.assertIn('st-candidate', next(m for m in o['marks'] if m['id'] == 'goal-41824')['cls'].split())
        self.assertEqual([(s['ev'], s['text']) for s in o['scores']], [('goal-5080', '1-0'), ('goal-41824', '1-1'), ('goal-46834', '1-2')])


@unittest.skipUnless(NODE, 'needs node to run match.js')
class Halftime(unittest.TestCase):
    """The halftime handles and editor: pointer, keys, typed times, the break, the direction, Save, Reset."""

    @classmethod
    def setUpClass(cls):
        new = {'firstHalfEnd': 2800.0, 'confirmed': True}
        drag = job(hold=True, answers={
            'intel': ['@intel', building(.8, 'Recognising stoppages, restarts and goals', __halftime=new), {'@intel': {'__halftime': new}}],
            'index': ['@index', {'@index': {'__halftime': new}}],
            'halftime': [{'ok': True, 'intel': building(.55, 'Finding the people and their places', __halftime=new), 'index': '@index'}]},
            steps=[{'do': 'pointer', 'half': 'end', 'path': [to_x(2854), to_x(2830)], 'up': False, 'name': 'dragging'},
                   {'do': 'pointer', 'half': 'end', 'path': [to_x(2800)], 'down': False, 'name': 'dropped'},
                   {'do': 'pop', 'mt': 'save', 'name': 'saved'},
                   {'do': 'release', 'hold': True, 'name': 'polled'},
                   {'do': 'release', 'hold': False, 'name': 'ready'}])
        keys = job(steps=[{'do': 'key', 'half': 'end', 'key': 'ArrowRight', 'name': 'right'},
                          {'do': 'key', 'half': 'end', 'key': 'ArrowLeft', 'shift': True, 'name': 'shift'},
                          {'do': 'key', 'half': 'end', 'key': 'Home', 'name': 'home'},
                          {'do': 'key', 'half': 'end', 'key': 'End', 'name': 'end'},
                          {'do': 'key', 'half': 'end', 'key': 'PageDown', 'name': 'page'},
                          {'do': 'key', 'half': 'start', 'key': 'End', 'name': 'startEnd'},
                          {'do': 'key', 'half': 'start', 'key': 'Enter', 'name': 'enter'}])
        entry = job(steps=[{'do': 'key', 'half': 'end', 'key': 'Enter', 'name': 'opened'},
                           {'do': 'pop-input', 'half': 'end', 'value': '47:00', 'name': 'mmss'},
                           {'do': 'pop-input', 'half': 'start', 'value': '1:00:00', 'how': 'change', 'name': 'hmmss'},
                           {'do': 'pop-input', 'half': 'end', 'value': '2830.5', 'name': 'seconds'},
                           {'do': 'pop-input', 'half': 'end', 'value': 'abc', 'name': 'bad'},
                           {'do': 'pop-input', 'half': 'start', 'value': '99:00', 'name': 'outside'},
                           {'do': 'pop', 'mt': 'save', 'name': 'saved'}])
        label = job(steps=[{'do': 'pointer', 'half': 'end', 'path': [to_x(2854)], 'name': 'clicked'}])
        marker = job(steps=[{'do': 'key', 'half': 'end', 'key': 'Enter'},
                            {'do': 'pop', 'mt': 'break', 'on': False, 'name': 'one'},
                            {'do': 'pointer', 'half': 'one', 'path': [to_x(2854), to_x(2880), to_x(2900)], 'name': 'moved'},
                            {'do': 'pop', 'side': 'right', 'name': 'sided'},
                            {'do': 'pop', 'mt': 'save', 'name': 'saved'}])
        grab = job(steps=[{'do': 'pointer', 'half': 'end', 'path': [to_x(2854) - 70, to_x(2854) - 71], 'up': False, 'name': 'still'},
                          {'do': 'pointer', 'half': 'end', 'path': [to_x(2854) - 80], 'down': False, 'name': 'moved'}])
        clamp = job(steps=[{'do': 'pointer', 'half': 'end', 'path': [to_x(2854), to_x(3000)], 'name': 'pastStart'},
                           {'do': 'pointer', 'half': 'start', 'path': [to_x(2885), to_x(2000)], 'name': 'pastEnd'},
                           {'do': 'pointer', 'half': 'end', 'path': [to_x(2854), 0], 'name': 'pastMatch'}])
        note = 'Two of 5 kick-offs had the teams on the other sides than the halftime setting says: check which way team A attacks in the first half'
        noted = job(answers={'intel': [{'@intel': {'__check': {'note': note, 'agree': 3, 'disagree': 2}}}]}, steps=[{'do': 'key', 'half': 'end', 'key': 'Enter', 'name': 'opened'}])
        saved = job(answers={'intel': [{'@intel': {'__halftime': {'confirmed': True}}}]},
                    steps=[{'do': 'key', 'half': 'end', 'key': 'Enter', 'name': 'opened'},
                           {'do': 'pointer', 'half': 'end', 'path': [to_x(2854), to_x(2700)], 'name': 'edited'},
                           {'do': 'pop', 'mt': 'cancel', 'name': 'cancelled'},
                           {'do': 'press-outside', 'name': 'outside'},
                           {'do': 'key', 'half': 'end', 'key': 'Enter'},
                           {'do': 'pop', 'mt': 'reset', 'name': 'reset'}])
        none = {'firstHalfEnd': None, 'secondHalfStart': None, 'mode': 'marker', 'teamAFirstHalf': None,
                'inferred': {'firstHalf': None, 'secondHalf': None, 'agreement': [None, None], 'consistent': False}}
        unset = job(answers={'intel': [{'@intel': {'__halftime': none}}], 'index': [{'@index': {'__halftime': none}}]},
                    steps=[{'do': 'key', 'half': 'one', 'key': 'ArrowRight', 'name': 'moved'}, {'do': 'pop', 'mt': 'save', 'name': 'blocked'},
                           {'do': 'pop', 'side': 'right', 'name': 'sided'}, {'do': 'pop', 'mt': 'save', 'name': 'saved'}])
        (cls.drag, cls.keys, cls.entry, cls.label, cls.marker, cls.clamp, cls.noted, cls.saved, cls.unset, cls.grab) = run(
            drag, keys, entry, label, marker, clamp, noted, saved, unset, grab)
        cls.note = note

    def test_the_break_has_two_handles_and_is_shaded(self):
        o = self.drag['first']
        self.assertEqual(errors(o), [])
        h = {x['half']: x for x in o['handles']}
        self.assertEqual(sorted(h), ['end', 'start'])
        self.assertEqual(h['end']['text'], f'FIRST HALF END {DASH} 47:34')
        self.assertEqual(h['start']['text'], f'SECOND HALF START {DASH} 48:05')
        self.assertAlmostEqual(num(h['end']['left']), 2854 / DUR * 100, places=3)
        self.assertAlmostEqual(num(h['start']['left']), 2885 / DUR * 100, places=3)
        self.assertIn('a-end', h['end']['cls'].split(), 'first half end extends to the left of its line')
        self.assertIn('a-start', h['start']['cls'].split(), 'second half start to the right: the two never overlap')
        self.assertEqual((h['end']['role'], h['end']['tab'], h['end']['label']), ('slider', 0, 'First half end'))
        self.assertEqual((h['end']['min'], h['end']['max'], h['end']['now']), (11, 2885, 2854), 'strictly inside the match, before the second half')
        self.assertEqual((h['start']['min'], h['start']['max']), (2854, 5791))
        self.assertEqual(len(o['lines']), 2)
        self.assertFalse(o['band']['hidden'])
        self.assertAlmostEqual(num(o['band']['width']), 31 / DUR * 100, places=3)
        self.assertIsNone(o['pop'], 'compact: the editor opens on demand')

    def test_drag_label_follows_then_save_posts_and_waits_for_the_state(self):
        d = self.drag
        mid = {x['half']: x for x in d['dragging']['handles']}
        self.assertEqual(mid['end']['text'], f'FIRST HALF END {DASH} {clock(round(to_t(to_x(2830))))}', 'the label follows the drag')
        self.assertIn('dragging', mid['end']['cls'].split())
        self.assertIn('dirty', mid['end']['cls'].split())
        drop = d['dropped']
        h = {x['half']: x for x in drop['handles']}
        self.assertEqual(h['end']['now'], round(to_t(to_x(2800))))
        self.assertEqual(h['end']['now'], 2800)
        self.assertEqual(h['end']['valuetext'], '46:40, not saved yet')
        p = drop['pop']
        self.assertIsNotNone(p, 'after a drag the editor shows Save')
        self.assertEqual(p['state'], 'Not saved yet')
        self.assertFalse(p['save']['disabled'])
        self.assertFalse(p['cancel']['hidden'])
        self.assertEqual([i['value'] for i in p['inputs']], ['46:40', '48:05'])
        self.assertEqual(drop['posts'], [], 'nothing is saved before Save')
        s = d['saved']
        self.assertEqual(post_bodies(s, '/api/match/halftime'), [{'match': MATCH, 'mode': 'break', 'firstHalfEnd': 2800, 'secondHalfStart': 2885, 'teamAFirstHalf': 'left'}])
        self.assertEqual(s['phase'], 'updating')
        self.assertEqual(s['msg']['text'], f'Updating the match state{ELL}')
        self.assertTrue(s['stale'], 'the old markers stay, dimmed, until the new state is ready')
        self.assertEqual(len(s['marks']), len(EVENTS))
        self.assertEqual({x['half']: x['now'] for x in s['handles']}, {'end': 2800, 'start': 2885}, 'the saved halftime')
        self.assertEqual(s['pop']['state'], f'Updating the match state{ELL}')
        self.assertIn('The match state is being updated: corrections wait until it is ready.', s['events']['msgs'])
        polled = d['polled']
        self.assertEqual(polled['gets'].count(f'/api/match/intel?id={MATCH}'), 2, 'asked again while it is made')
        self.assertEqual(polled['phase'], 'updating')
        r = d['ready']
        self.assertEqual(r['gets'][-1], f'/api/match/index?id={MATCH}', 'then the index is read again')
        self.assertEqual(r['gets'].count(f'/api/match/intel?id={MATCH}'), 3)
        self.assertEqual(r['phase'], 'idle')
        self.assertIsNone(r['msg'])
        self.assertFalse(r['stale'])
        self.assertEqual(r['pop']['state'], 'Saved')
        self.assertNotIn('dirty', ' '.join(x['cls'] for x in r['handles']))
        self.assertEqual(errors(r), [])

    def test_keys_move_the_handle(self):
        k = self.keys
        val = lambda o, half='end': next(x['now'] for x in o['handles'] if x['half'] == half)
        self.assertEqual(val(k['right']), 2855)
        self.assertEqual(val(k['shift']), 2845, 'Shift: 10 s')
        self.assertEqual(val(k['home']), 11, 'Home: the start of the match (strictly inside it)')
        self.assertEqual(val(k['end']), 2885, 'End: the second half start')
        self.assertEqual(val(k['page']), 2825, 'Page Down: 60 s')
        self.assertEqual(val(k['startEnd'], 'start'), 5791, 'End: the end of the match (strictly inside it)')
        self.assertTrue(all(x == {'prevented': True, 'stopped': True} for x in k['last']['keys']), 'the page never also steps the video')
        self.assertEqual(k['right']['pop']['state'], 'Not saved yet')
        self.assertEqual(k['enter']['active'], {'tag': 'INPUT', 'half': 'start', 'ev': None}, 'Enter: the typed entry of that handle')
        self.assertEqual(k['last']['posts'], [])

    def test_precise_entry(self):
        e = self.entry
        o = e['opened']['pop']
        self.assertEqual(e['opened']['active'], {'tag': 'INPUT', 'half': 'end', 'ev': None})
        self.assertEqual([(i['half'], i['value'], i['hidden']) for i in o['inputs']], [('end', '47:34', False), ('start', '48:05', False)])
        self.assertTrue(o['brk'])
        val = lambda s, half: next(x['now'] for x in s['handles'] if x['half'] == half)
        self.assertEqual(val(e['mmss'], 'end'), 2820, 'mm:ss')
        self.assertEqual(next(x['text'] for x in e['mmss']['handles'] if x['half'] == 'end'), f'FIRST HALF END {DASH} 47:00')
        self.assertEqual(val(e['hmmss'], 'start'), 3600, 'h:mm:ss')
        self.assertEqual(next(x['text'] for x in e['hmmss']['handles'] if x['half'] == 'start'), f'SECOND HALF START {DASH} 60:00')
        self.assertEqual(val(e['seconds'], 'end'), 2830.5, 'seconds, with a decimal')
        self.assertEqual(next(i['value'] for i in e['seconds']['pop']['inputs'] if i['half'] == 'end'), '47:10.5', 'written back with its tenth')
        bad = e['bad']['pop']
        self.assertEqual(bad['err'], 'Enter a time as mm:ss, h:mm:ss or seconds.')
        self.assertEqual(next(i['invalid'] for i in bad['inputs'] if i['half'] == 'end'), 'true')
        self.assertEqual(val(e['bad'], 'end'), 2830.5, 'unchanged')
        self.assertEqual(e['outside']['pop']['err'], 'The second half must start between 47:11 and 1:36:31.')
        self.assertEqual(val(e['outside'], 'start'), 3600)
        self.assertEqual(post_bodies(e['saved'], '/api/match/halftime'),
                         [{'match': MATCH, 'mode': 'break', 'firstHalfEnd': 2830.5, 'secondHalfStart': 3600, 'teamAFirstHalf': 'left'}])

    def test_a_click_on_the_label_opens_the_entry(self):
        c = self.label['clicked']
        self.assertIsNotNone(c['pop'])
        self.assertEqual(c['active'], {'tag': 'INPUT', 'half': 'end', 'ev': None})
        self.assertIsNone(c['draft'], 'a click is not an edit')
        self.assertEqual(c['pop']['state'], 'From the match plan and the goal ends: not confirmed yet')

    def test_one_marker_and_the_direction(self):
        o = self.marker['one']
        self.assertEqual([(x['half'], x['text']) for x in o['handles']], [('one', f'HALFTIME {DASH} 47:34')])
        self.assertTrue(o['band']['hidden'])
        self.assertEqual([(i['half'], i['hidden']) for i in o['pop']['inputs']], [('end', False), ('start', True)])
        self.assertFalse(o['pop']['brk'])
        self.assertEqual(o['handles'][0]['min'], 11)
        self.assertEqual(o['handles'][0]['max'], 5791)
        self.assertEqual(self.marker['moved']['handles'][0]['now'], 2900)
        sided = self.marker['sided']['pop']
        self.assertEqual(sided['sides'], [['left', 'false'], ['right', 'true']])
        self.assertEqual(sided['hint'], 'In the second half team A attacks left.')
        self.assertIn('Not saved yet: the kick-offs were checked with team A attacking left in the first half.', sided['check'])
        self.assertIn('The goal ends suggest team A attacks left in the first half.', sided['check'])
        self.assertEqual(post_bodies(self.marker['saved'], '/api/match/halftime'),
                         [{'match': MATCH, 'mode': 'marker', 'firstHalfEnd': 2900, 'secondHalfStart': 2900, 'teamAFirstHalf': 'right'}])

    def test_a_drag_moves_from_where_the_handle_was_grabbed(self):
        # the label stands beside its line: grabbed 70 px left of the line, the line does not jump to the pointer
        g = self.grab
        self.assertEqual(next(x['now'] for x in g['still']['handles'] if x['half'] == 'end'), 2854, 'a pixel is not a drag yet')
        self.assertIsNone(g['still']['draft'])
        moved = next(x['now'] for x in g['moved']['handles'] if x['half'] == 'end')
        self.assertEqual(moved, js_round(2854 - 10 / (RECT['width'] - 14) * DUR), 'ten pixels to the left: the time of ten pixels earlier')

    def test_never_past_the_match_range_or_the_other_handle(self):
        c = self.clamp
        val = lambda s, half: next(x['now'] for x in s['handles'] if x['half'] == half)
        self.assertEqual(val(c['pastStart'], 'end'), 2885)
        self.assertEqual(val(c['pastEnd'], 'start'), 2885)
        self.assertEqual(val(c['pastMatch'], 'end'), 11)

    def test_the_direction_check_and_its_note(self):
        o = self.drag['first']
        self.assertIn('Team A attacks left in the first half and right in the second (from the goal ends, not confirmed).', o['events']['dir'])
        self.assertIn('The inferred direction and all 5 kick-offs agree.', o['events']['dir'])
        p = self.keys['enter']['pop']
        self.assertIn('The inferred direction and all 5 kick-offs agree.', p['check'])
        self.assertEqual(p['note'], '')
        n = self.noted['opened']
        self.assertEqual(n['pop']['note'], self.note, 'the note first in the editor')
        self.assertEqual(n['events']['note'], self.note, 'and in the events drawer')
        self.assertIn('3 of 5 kick-offs agree with it.', n['pop']['check'])

    def test_cancel_outside_and_reset(self):
        s = self.saved
        self.assertEqual(s['opened']['pop']['state'], 'Saved')
        self.assertFalse(s['opened']['pop']['reset']['disabled'], 'Reset once the coach saved a halftime')
        self.assertEqual(next(x['now'] for x in s['edited']['handles'] if x['half'] == 'end'), 2700)
        c = s['cancelled']
        self.assertEqual({x['half']: x['now'] for x in c['handles']}, {'end': 2854, 'start': 2885})
        self.assertIsNone(c['draft'])
        self.assertIsNone(s['outside']['pop'], 'a press elsewhere closes the editor')
        self.assertEqual(post_bodies(s['reset'], '/api/match/halftime/reset'), [{'match': MATCH}])
        self.assertEqual(s['reset']['gets'][-2:], [f'/api/match/intel?id={MATCH}', f'/api/match/index?id={MATCH}'], 'the state asked for until ready, then the index')
        self.assertEqual(s['reset']['phase'], 'idle')
        self.assertTrue(self.drag['first']['handles'] and self.keys['enter']['pop']['reset']['disabled'], 'nothing to reset before a save')

    def test_a_match_without_a_halftime(self):
        u = self.unset
        self.assertEqual([(x['half'], x['text']) for x in u['first']['handles']], [('one', f'HALFTIME {DASH} not set')])
        mid = round((11 + 5791) / 2)
        self.assertEqual(u['moved']['handles'][0]['now'], mid + 1)
        self.assertIsNotNone(u['moved']['pop'])
        self.assertTrue(u['moved']['pop']['save']['disabled'], 'no direction known or chosen yet')
        self.assertEqual(u['moved']['pop']['sides'], [['left', 'false'], ['right', 'false']])
        self.assertEqual(u['moved']['pop']['hint'], 'Choose the side team A attacks in the first half.')
        self.assertEqual(post_bodies(u['blocked'], '/api/match/halftime'), [])
        self.assertFalse(u['sided']['pop']['save']['disabled'])
        self.assertEqual(post_bodies(u['saved'], '/api/match/halftime'),
                         [{'match': MATCH, 'mode': 'marker', 'firstHalfEnd': mid + 1, 'secondHalfStart': mid + 1, 'teamAFirstHalf': 'right'}])


@unittest.skipUnless(NODE, 'needs node to run match.js')
class EventsDrawer(unittest.TestCase):
    """The match events drawer: the shares, the rows, the filter, and the POST bodies of its actions."""

    @classmethod
    def setUpClass(cls):
        confirmed = {'@intel': {'__events': {'goal-5080': {'status': 'confirmed'}},
                                'corrections': [{'kind': 'confirm', 'id': 'goal-5080', 'cid': 'c1', 'at': '2026-10-31T07:00:00'}]}}
        acts = job(answers={'event': [{'ok': True, 'correction': {'cid': 'c1'}, 'intel': confirmed}, {'ok': True, 'intel': confirmed}],
                            'eventRemove': [{'ok': True, 'intel': '@intel'}]},
                   steps=[{'do': 'event', 'id': 'goal-5080', 'act': 'confirm', 'name': 'confirmed'},
                          {'do': 'undo', 'cid': 'c1', 'name': 'undone'},
                          {'do': 'event', 'id': 'goal_kick-1469', 'act': 'type', 'value': 'CORNER_KICK', 'name': 'retyped'},
                          {'do': 'time', 't': 150.5},
                          {'do': 'event', 'id': 'goal_kick-1469', 'act': 'move', 'name': 'moved'},
                          {'do': 'event', 'id': THROW, 'act': 'remove', 'name': 'removed'},
                          {'do': 'time', 't': 1234.56},
                          {'do': 'add', 'type': 'GOAL', 'team': 'B', 'name': 'goal'},
                          {'do': 'add', 'type': 'OTHER_STOPPAGE', 'team': '', 'name': 'stoppage'},
                          {'do': 'add', 'type': 'SUBSTITUTION', 'name': 'sub'},
                          {'do': 'event', 'id': 'goal-5080', 'act': 'evidence', 'name': 'evidence'},
                          {'do': 'filter', 'value': 'GOAL', 'name': 'goals'},
                          {'do': 'event', 'id': 'goal-41824', 'act': 'seek', 'name': 'seeked'}])
        removed = job(answers={'intel': [{'@intel': {'__events': {THROW: {'status': 'removed'}},
                                                     'corrections': [{'kind': 'remove', 'id': THROW, 'cid': 'c9', 'at': '2026-10-31T07:01:00'}]}}]},
                      steps=[{'do': 'filter', 'value': 'removed', 'name': 'removed'}, {'do': 'event', 'id': THROW, 'act': 'undo', 'name': 'undo'}])
        cls.acts, cls.removed = run(acts, removed)

    def test_status_shares_and_rows_in_time_order(self):
        o = self.acts['first']['events']
        self.assertEqual(o['status'], f'Match state ready {DOT} 55 events {DOT} 32 to review')
        share = INTEL['stats']['share']
        for key, word in (('IN_PLAY', 'in play'), ('STOPPAGE', 'stoppage'), ('POSSIBLE_STOPPAGE', 'possible stoppage'), ('UNKNOWN', 'unknown'), ('HALFTIME', 'halftime')):
            self.assertIn(f'{js_round(share[key] * 100)}% {word}', o['stats'])
        self.assertIn('46% in play', o['stats'])
        self.assertEqual([r['id'] for r in o['rows']], [e['id'] for e in EVENTS])
        g = next(r for r in o['rows'] if r['id'] == 'goal-5080')
        self.assertEqual(g['head'], f'08:28GoalTeam A scoreddetected {DOT} high', 'time (seek), type, team, status')
        acts = {a['act']: a for a in g['acts']}
        self.assertEqual(sorted(acts), ['confirm', 'evidence', 'move', 'remove', 'seek', 'type'])
        self.assertEqual((acts['type']['value'], acts['evidence']['text'], acts['seek']['text']), ('GOAL', 'Evidence (2)', '08:28'))
        self.assertTrue(g['evidence']['hidden'], 'the evidence on request')
        sub = next(r for r in o['rows'] if r['id'] == 'substitution-7384')
        self.assertIn(f"slot 02 {DOT} {name_of(ROSTER['A-P29'])} {ARROW} {name_of(ROSTER['A-P34'])}", sub['head'])
        self.assertIn('to review', sub['head'])
        self.assertEqual(o['filter']['value'], 'all')
        self.assertIn(['GOAL', 'Goals (4)'], o['filter']['options'])
        self.assertIn(['candidate', 'To review (candidates) (32)'], o['filter']['options'])
        self.assertNotIn('removed', [k for k, _ in o['filter']['options']], 'no empty kinds offered')
        self.assertEqual(o['add'], {'type': 'GOAL', 'team': 'A', 'teams': ['A', 'B'], 'disabled': False})
        self.assertEqual(o['fixes'], [])
        self.assertIn('Corrections of the events', o['text'])

    def test_the_post_bodies_of_the_actions(self):
        a = self.acts
        self.assertEqual(post_bodies(a['confirmed'], '/api/match/event'), [{'match': MATCH, 'kind': 'confirm', 'id': 'goal-5080'}])
        c = a['confirmed']['events']
        self.assertIn('confirmed', next(r for r in c['rows'] if r['id'] == 'goal-5080')['cls'])
        self.assertEqual(len(c['fixes']), 1)
        self.assertTrue(c['fixes'][0].startswith('Confirmed the goal at 08:28'))
        self.assertIn('Confirmed the goal at 08:28.', c['msgs'])
        self.assertIn('st-confirmed', next(m for m in a['confirmed']['marks'] if m['id'] == 'goal-5080')['cls'], 'the marker follows the answer')
        self.assertEqual(post_bodies(a['undone'], '/api/match/event/remove'), [{'match': MATCH, 'cid': 'c1'}])
        self.assertEqual(post_bodies(a['retyped'], '/api/match/event')[-1], {'match': MATCH, 'kind': 'retype', 'id': 'goal_kick-1469', 'type': 'CORNER_KICK'})
        self.assertEqual(post_bodies(a['moved'], '/api/match/event')[-1], {'match': MATCH, 'kind': 'move', 'id': 'goal_kick-1469', 't': 150.5})
        self.assertEqual(post_bodies(a['removed'], '/api/match/event')[-1], {'match': MATCH, 'kind': 'remove', 'id': THROW})
        self.assertEqual(post_bodies(a['goal'], '/api/match/event')[-1], {'match': MATCH, 'kind': 'add', 'type': 'GOAL', 't': 1234.56, 'team': 'B'})
        self.assertEqual(post_bodies(a['stoppage'], '/api/match/event')[-1], {'match': MATCH, 'kind': 'add', 'type': 'OTHER_STOPPAGE', 't': 1234.56})
        self.assertEqual(a['stoppage']['events']['add']['teams'], ['A', 'B', ''], 'a stoppage may have no team')
        self.assertEqual(post_bodies(a['sub'], '/api/match/event')[-1], {'match': MATCH, 'kind': 'add', 'type': 'SUBSTITUTION', 't': 1234.56, 'team': 'A'},
                         'a substitution always has a team')
        self.assertEqual(a['sub']['events']['add']['teams'], ['A', 'B'])
        self.assertEqual(errors(a['last']), [])

    def test_evidence_filter_and_seek(self):
        ev = next(r for r in self.acts['evidence']['events']['rows'] if r['id'] == 'goal-5080')['evidence']
        self.assertEqual(ev, {'hidden': False, 'items': BY_ID['goal-5080']['evidence']})
        goals = self.acts['goals']['events']
        self.assertEqual([r['id'] for r in goals['rows']], [e['id'] for e in EVENTS if e['type'] == 'GOAL'])
        self.assertEqual(goals['filter']['value'], 'GOAL')
        s = self.acts['seeked']
        self.assertEqual(s['seeks'][-1], BY_ID['goal-41824']['t'])
        self.assertTrue(s['video']['paused'])

    def test_a_removed_event_and_its_undo(self):
        r = self.removed
        self.assertNotIn(THROW, [m['id'] for m in r['first']['marks']])
        self.assertNotIn(THROW, [x['id'] for x in r['first']['events']['rows']], 'not among the events')
        self.assertIn(['removed', 'Removed by you (1)'], r['first']['events']['filter']['options'])
        rows = r['removed']['events']['rows']
        self.assertEqual([x['id'] for x in rows], [THROW])
        self.assertEqual([a['act'] for a in rows[0]['acts']], ['seek', 'undo'])
        self.assertTrue(r['first']['events']['fixes'][0].startswith('Removed the throw-in at'))
        self.assertEqual(post_bodies(r['undo'], '/api/match/event/remove'), [{'match': MATCH, 'cid': 'c9'}])


@unittest.skipUnless(NODE, 'needs node to run match.js')
class RosterReview(unittest.TestCase):
    """The roster review in the Match identities drawer: slots at the playback time, the unresolved, the left out, and
    its corrections."""

    @classmethod
    def setUpClass(cls):
        a2 = place('A', 2)
        j, _ = holder(a2['nodes'], 2000)
        cls.a2node = a2['nodes'][j]
        cls.a2from = [a2['nodes'][j - 1]['fragments'][-1], a2['nodes'][j + 1]['fragments'][0]]
        review = job(steps=[{'do': 'time', 't': 2000.4, 'name': 'tick'}, {'do': 'time', 't': 100.0, 'name': 'early'},
                            {'do': 'all', 'name': 'all'}, {'do': 'drawer', 'page': 'advanced'}, {'do': 'time', 't': 5000.0, 'name': 'closed'}])
        unresolved = next(p for p in INDEX['roster'] if p['matchId'] == 'A-P24')
        cls.unresolved = unresolved
        fixes = job(steps=[
            {'do': 'rv', 'team': 'A', 'slot': 2, 'open': True, 'act': 'join-into', 'value': 'A-P01', 'then': 'join', 'name': 'merge'},
            {'do': 'rv', 'team': 'A', 'slot': 2, 'open': True, 'act': 'split', 'name': 'split'},
            {'do': 'rv', 'team': 'A', 'slot': 2, 'open': True, 'act': 'slot-of', 'value': 2, 'then': 'confirm-slot', 'name': 'confirm'},
            {'do': 'rv', 'team': 'A', 'slot': 2, 'open': True, 'act': 'exclude', 'name': 'exclude'},
            {'do': 'rv', 'team': 'A', 'slot': 2, 'open': True, 'act': 'team-B', 'name': 'team'},
            {'do': 'rv', 'team': 'A', 'slot': 2, 'open': True, 'act': 'keeper-on', 'name': 'keeper'},
            {'do': 'rv', 'team': 'A', 'slot': 'GK', 'open': True, 'act': 'keeper-off', 'name': 'keeperOff'},
            {'do': 'rv', 'team': 'A', 'slot': 2, 'act': 'sub-confirm', 'name': 'subConfirm'},
            {'do': 'rv', 'team': 'A', 'slot': 2, 'act': 'sub-reject', 'name': 'subReject'},
            {'do': 'rv', 'person': 'A-P24', 'open': True, 'act': 'slot-of', 'value': 3, 'then': 'confirm-slot', 'name': 'unresolved'},
            {'do': 'rv', 'fragment': 'A-21', 'act': 'include', 'name': 'include'}])
        listed = [{'kind': 'split', 'target': 'A-51', 'from': ['A-44', 'A-63'], 'id': 'k1', 'at': '2026-10-31T07:00:00'},
                  {'kind': 'confirm', 'target': 'A-51', 'slot': 2, 'id': 'k2', 'at': '2026-10-31T07:00:01'},
                  {'kind': 'substitution', 'out': 'A-31', 'in': 'A-36', 'status': 'rejected', 'id': 'k3', 'at': '2026-10-31T07:00:02'},
                  {'kind': 'include', 'target': 'A-21', 'id': 'k4', 'at': '2026-10-31T07:00:03'},
                  {'kind': 'keeper', 'target': 'A-51', 'on': True, 'id': 'k5', 'at': '2026-10-31T07:00:04'}]
        said = {'@index': {'corrections': listed, 'people': dict(INDEX['people'], substitutions=[
            {'outFragment': 'A-31', 'inFragment': 'A-36', 'out': 'A-P29', 'in': 'A-P34', 'status': 'rejected'}])}}
        texts = job(answers={'index': [said]})
        people = job(hold=True, answers={'correction': [{'ok': True, 'correction': {'id': 'r1'}, 'index': {'@index': {'intel': {'state': 'people'}}}}],
                                         'index': ['@index'], 'intel': ['@intel', building(.8, 'Recognising stoppages, restarts and goals'),
                                                                        building(.9, 'Recognising stoppages, restarts and goals'), '@intel']},
                     steps=[{'do': 'rv', 'team': 'B', 'slot': 3, 'open': True, 'act': 'split', 'name': 'corrected'},
                            {'do': 'release', 'hold': True, 'name': 'polled'}, {'do': 'release', 'hold': False, 'name': 'ready'}])
        cls.review, cls.fixes, cls.texts, cls.people = run(review, fixes, texts, people)

    def expected_slots(self, t):
        out = []
        for team in 'AB':
            for slot in range(1, 11):
                nodes = place(team, slot)['nodes']
                j, covering = holder(nodes, t)
                n = nodes[j]
                p = ROSTER[n['person']]
                out.append((team, str(slot), name_of(p), n, covering))
        return out

    def check_slots(self, review, t):
        got = [s for s in review['slots'] if s['slot'] != 'GK']
        want = self.expected_slots(t)
        self.assertEqual(len(got), 20)
        for g, (team, slot, who, n, covering) in zip(got, want):
            with self.subTest(team=team, slot=slot, t=t):
                self.assertEqual((g['team'], g['slot'], g['who'], g['id']), (team, slot, who, n['person']))
                seen = 'Seen now' if covering else f"Not seen since {clock(n['t1'])}"
                conf = 'sure' if n['sure'] else f"estimate (link {js_round(n['p'] * 100)}%)"
                self.assertEqual(g['meta'][0], f"{seen} {DOT} {conf} {DOT} fragment {', '.join(n['fragments'])}")

    def test_slots_at_the_playback_time(self):
        o = self.review['first']
        self.assertEqual(errors(o), [])
        r = o['review']
        self.assertEqual(r['title'], f'Roster review {DOT} at 33:20')
        self.assertEqual(r['heads'][:2], [f'Team A {DOT} slots 01{NDASH}10 and GK', f'Team B {DOT} slots 01{NDASH}10 and GK'])
        self.check_slots(r, 2000)
        a3 = next(s for s in r['slots'] if s['team'] == 'A' and s['slot'] == '3')
        self.assertTrue(a3['meta'][0].startswith('Not seen since 32:47'), 'the last node before the time when none covers it')
        self.assertEqual(o['list'], 0, 'the list of every identity waits behind its toggle')

    def test_the_goalkeepers(self):
        r = self.review['first']['review']
        gk = {s['team']: s for s in r['slots'] if s['slot'] == 'GK'}
        self.assertEqual((gk['A']['who'], gk['A']['id']), (name_of(ROSTER['A-P20']), 'A-P20'))
        self.assertEqual((gk['B']['who'], gk['B']['id']), (name_of(ROSTER['B-P10']), 'B-P10'))
        self.assertTrue(gk['A']['meta'][0].endswith(f"{DOT} sure {DOT} {len(ROSTER['A-P20']['fragments'])} fragments"))
        self.assertIn('Other goalkeeper person of this team: GK-A?', gk['A']['meta'][1])

    def test_the_review_follows_the_playback_time(self):
        self.assertEqual(self.review['tick']['review']['title'], f'Roster review {DOT} at 33:20', 'the same second: nothing redrawn')
        early = self.review['early']['review']
        self.assertEqual(early['title'], f'Roster review {DOT} at 01:40')
        self.check_slots(early, 100)
        a1 = next(s for s in early['slots'] if s['team'] == 'A' and s['slot'] == '1')
        self.assertEqual(a1['who'], f'A-01 {DOT} A-P04')
        self.assertTrue(a1['meta'][0].startswith(f'Seen now {DOT} sure'))
        self.assertEqual(self.review['closed']['review']['title'], f'Roster review {DOT} at 01:40', 'not followed while the drawer shows another page')
        self.assertGreater(self.review['all']['list'], 300, 'every identity on request')

    def test_unresolved_and_left_out(self):
        r = self.review['first']['review']
        want = [p for p in INDEX['roster'] if not p.get('excluded') and p.get('slot') is None and p['role'] != 'GOALKEEPER']
        self.assertEqual(len(r['unresolved']), len(want))
        self.assertEqual(len(want), 27)
        self.assertEqual(sorted(u['person'] for u in r['unresolved']), sorted(p['matchId'] for p in want))
        p = self.unresolved
        row = next(u for u in r['unresolved'] if u['person'] == 'A-P24')
        self.assertIn(name_of(p), row['text'])
        self.assertIn(f"fragment {p['fragments'][0]}", row['text'])
        self.assertIn(f"Unresolved fragments {DOT} 27", r['heads'])
        out = [p for p in INDEX['roster'] if p.get('excluded')]
        self.assertEqual([x['fragment'] for x in r['leftOut']], [p['matchId'] for p in sorted(out, key=lambda p: p['firstSeen'])])
        a21 = next(x for x in r['leftOut'] if x['fragment'] == 'A-21')
        self.assertIn('fragment A-21', a21['text'])
        self.assertIn('A box that stays put on a broadcast graphic, not a person', a21['text'])
        self.assertIn(f'Left out {DOT} 32', r['heads'])

    def test_the_slot_substitutions(self):
        a2 = next(s for s in self.review['first']['review']['slots'] if s['team'] == 'A' and s['slot'] == '2')
        e = BY_ID['substitution-7384']
        self.assertEqual(a2['subs'], [f"Substitution {clock(e['t'])} {DOT} {name_of(ROSTER['A-P29'])} {ARROW} {name_of(ROSTER['A-P34'])} {DOT} candidate, low confidenceConfirmReject"])
        rejected = next(s for s in self.texts['first']['review']['slots'] if s['team'] == 'A' and s['slot'] == '2')
        self.assertTrue(rejected['subs'][0].startswith(f"Substitution {clock(e['t'])}"))
        self.assertIn('rejected by you', rejected['subs'][0])

    def test_the_post_bodies_of_its_corrections(self):
        f = self.fixes
        url = '/api/match/correction'
        n = self.a2node
        self.assertEqual(post_bodies(f['merge'], url), [{'match': MATCH, 'kind': 'merge', 'target': n['person'], 'into': 'A-P01'}])
        self.assertEqual(post_bodies(f['split'], url)[-1], {'match': MATCH, 'kind': 'split', 'target': n['fragments'][0], 'from': self.a2from})
        self.assertEqual(post_bodies(f['confirm'], url)[-1], {'match': MATCH, 'kind': 'confirm', 'target': n['person'], 'slot': 2},
                         'the person: the server applies it to all of its fragments')
        self.assertEqual(post_bodies(f['exclude'], url)[-1], {'match': MATCH, 'kind': 'exclude', 'target': n['person']})
        self.assertEqual(post_bodies(f['team'], url)[-1], {'match': MATCH, 'kind': 'team', 'target': n['person'], 'team': 'B'})
        self.assertEqual(post_bodies(f['keeper'], url)[-1], {'match': MATCH, 'kind': 'keeper', 'target': n['person'], 'on': True})
        self.assertEqual(post_bodies(f['keeperOff'], url)[-1], {'match': MATCH, 'kind': 'keeper', 'target': 'A-P20', 'on': False})
        self.assertEqual(post_bodies(f['subConfirm'], url)[-1], {'match': MATCH, 'kind': 'substitution', 'out': 'A-31', 'in': 'A-36', 'status': 'confirmed'})
        self.assertEqual(post_bodies(f['subReject'], url)[-1], {'match': MATCH, 'kind': 'substitution', 'out': 'A-31', 'in': 'A-36', 'status': 'rejected'})
        self.assertEqual(post_bodies(f['unresolved'], url)[-1], {'match': MATCH, 'kind': 'confirm', 'target': 'A-P24', 'slot': 3})
        self.assertEqual(post_bodies(f['include'], url)[-1], {'match': MATCH, 'kind': 'include', 'target': 'A-21'})
        self.assertEqual(len(post_bodies(f['last'], url)), 11)
        merged = f['merge']
        self.assertIn(f"Joined {name_of(ROSTER[n['person']])} and {name_of(ROSTER['A-P01'])}: one player.", merged['rosterText'], 'the message of the correction')
        acts = next(s['acts'] for s in merged['review']['slots'] if s['team'] == 'A' and s['slot'] == '2')
        self.assertEqual(acts, ['correct', 'sub-confirm', 'sub-reject'], "the row's tools close after the correction")
        self.assertEqual(errors(f['last']), [])

    def test_the_corrections_say_what_they_did(self):
        c = self.texts['first']['corrections']
        self.assertIn(f"fragment A-51 ({name_of(ROSTER['A-P49'])}) is not the same player as fragment A-44 ({name_of(ROSTER['A-P42'])}) or fragment A-63 ({name_of(ROSTER['A-P61'])})", c)
        self.assertIn(f"fragment A-51 ({name_of(ROSTER['A-P49'])}) is the player of slot 02", c)
        self.assertIn(f"Substitution fragment A-31 ({name_of(ROSTER['A-P29'])}) {ARROW} fragment A-36 ({name_of(ROSTER['A-P34'])}) rejected", c)
        self.assertIn('fragment A-21 counted as a player again', c)
        self.assertIn(f"fragment A-51 ({name_of(ROSTER['A-P49'])}) is a goalkeeper", c)
        self.assertEqual(c.count('Undo'), 5)

    def test_a_correction_that_changes_the_slots_makes_the_state_again(self):
        p = self.people
        c = p['corrected']
        bodies = [x['url'] for x in c['posts']]
        self.assertEqual(bodies, ['/api/match/correction'], 'the correction route starts the build itself: no second POST')
        self.assertEqual(c['gets'][-1], f'/api/match/intel?id={MATCH}', 'the index says the state is being made again: it is asked for')
        self.assertEqual(c['phase'], 'updating')
        self.assertEqual(c['msg']['text'], f'Updating the match state{ELL}')
        self.assertEqual(p['polled']['phase'], 'updating')
        self.assertEqual(p['polled']['gets'].count(f'/api/match/intel?id={MATCH}'), 3, 'asked again while it is made')
        r = p['ready']
        self.assertEqual(r['phase'], 'idle')
        self.assertEqual(r['gets'][-1], f'/api/match/index?id={MATCH}')
        self.assertIsNone(r['msg'])
        self.assertEqual(errors(r), [])


@unittest.skipUnless(NODE, 'needs node to run match.js')
class Build(unittest.TestCase):
    """A match whose state is missing is built and polled; a failing state or a server without the routes leaves the
    page alone; leaving the match removes it all."""

    @classmethod
    def setUpClass(cls):
        missing = job(hold=True, answers={'index': [{'@index': {'intel': {'state': 'missing'}}}, '@index'],
                                          'build': [building(.3, 'Reading the match data (about a minute the first time)')],
                                          'intel': [building(.55, 'Finding the people and their places'), '@intel']},
                      steps=[{'do': 'release', 'hold': True, 'name': 'polled'}, {'do': 'release', 'hold': False, 'name': 'ready'}])
        people = job(answers={'index': [{'@index': {'intel': {'state': 'people'}}}, '@index'], 'build': ['@intel']})
        failed = job(answers={'intel': [{'@intel': {'state': 'error', 'error': 'The match state could not be made: boom', 'events': [], 'timeline': None}}]})
        absent = job(answers={'intel': [{'__status': 404, '__error': 'NOT FOUND'}], 'index': [{'@index': {'intel': None, 'halftime': None}}]})
        leave = job(steps=[{'do': 'leave', 'name': 'left'}])
        cls.missing, cls.people, cls.failed, cls.absent, cls.leave = run(missing, people, failed, absent, leave)

    def test_a_missing_state_is_built_polled_then_drawn(self):
        m = self.missing
        f = m['first']
        self.assertEqual(f['posts'], [{'url': '/api/match/intel/build', 'body': {'match': MATCH}}])
        self.assertEqual(f['msg']['text'], f'Recognising the match: reading the match data (about a minute the first time) {DOT} 30%')
        self.assertEqual(f['marks'], [], 'nothing drawn before the state is ready')
        self.assertEqual(f['events']['status'], f['msg']['text'])
        self.assertEqual([x['half'] for x in f['handles']], ['end', 'start'], 'the halftime from the index meanwhile')
        p = m['polled']
        self.assertEqual(p['msg']['text'], f'Recognising the match: finding the people and their places {DOT} 55%')
        self.assertEqual(p['gets'], [f'/api/match/index?id={MATCH}', f'/api/match/intel?id={MATCH}'])
        r = m['ready']
        self.assertEqual(r['gets'], [f'/api/match/index?id={MATCH}', f'/api/match/intel?id={MATCH}', f'/api/match/intel?id={MATCH}', f'/api/match/index?id={MATCH}'])
        self.assertEqual(len(r['marks']), len(EVENTS))
        self.assertIsNone(r['msg'])
        self.assertEqual(r['phase'], 'idle')
        self.assertEqual(errors(r), [])

    def test_a_state_made_for_other_people_is_built(self):
        p = self.people['last']
        self.assertEqual([x['url'] for x in p['posts']], ['/api/match/intel/build'])
        self.assertEqual(len(p['marks']), len(EVENTS))
        self.assertEqual(p['gets'][-1], f'/api/match/index?id={MATCH}')

    def test_a_failed_state_says_so_and_the_rest_works(self):
        f = self.failed['last']
        self.assertEqual(f['msg']['text'], 'Match state unavailable')
        self.assertEqual(f['msg']['title'], 'The match state could not be made: boom')
        self.assertIn('error', f['msg']['cls'])
        self.assertEqual(f['events']['status'], 'Match state unavailable: The match state could not be made: boom')
        self.assertIn('Everything else keeps working', f['events']['text'])
        self.assertEqual(f['marks'], [])
        self.assertEqual([x['half'] for x in f['handles']], ['end', 'start'], 'the halftime can still be set')
        self.assertEqual(len(f['review']['slots']), 22, 'the roster review works')

    def test_a_server_without_the_routes_shows_nothing_new(self):
        a = self.absent['last']
        self.assertTrue(a['mtHidden'])
        self.assertFalse(a['on'])
        self.assertIsNone(a['msg'])
        self.assertTrue(a['off'])
        self.assertTrue(a['filter']['hidden'], 'no timeline filter without a match state')
        self.assertEqual(a['events']['text'], 'This server has no match state for this match: no events and no halftime control.')
        self.assertEqual(errors(a), [])

    def test_leaving_the_match_removes_it(self):
        o = self.leave['first']
        self.assertTrue(o['on'])
        self.assertFalse(o['filter']['hidden'])
        left = self.leave['left']
        self.assertTrue(left['mtHidden'])
        self.assertFalse(left['on'])
        self.assertEqual((left['marks'], left['handles'], left['msg'], left['scores']), ([], [], None, []))
        self.assertTrue(left['filter']['hidden'] and left['filter']['pop'], 'the filter goes with the match')
        self.assertEqual(left['events']['text'], 'Open a full match to review its events.')


KITS = {'A': {'kit': 'white', 'label': 'Team A (white)'}, 'B': {'kit': 'red', 'label': 'Team B (red)'}}
SUB_SLOTS = {'substitution-29896': {'slot': 3, 'outSlot': 3, 'inSlot': 4}, 'substitution-3656': {'slot': None, 'outSlot': None, 'inSlot': None}}
MARKER = {'mode': 'marker', 'firstHalfEnd': 2854.0, 'secondHalfStart': 2854.0}


@unittest.skipUnless(NODE, 'needs node to run match.js')
class NewContract(unittest.TestCase):
    """The backend's second round: the team kits in the index, a substitution's slots (outSlot / inSlot, slot = the
    outgoing player's), a person's correction that carries its other fragments ('also'), one halftime marker by default."""

    @classmethod
    def setUpClass(cls):
        kits = job(answers={'index': [{'@index': {'teamKits': KITS}}], 'intel': [{'@intel': {'__events': SUB_SLOTS}}]},
                   steps=[{'do': 'marker', 'id': 'goal-41824', 'how': 'enter', 'name': 'goal'},
                          {'do': 'marker', 'id': 'substitution-29896', 'how': 'enter', 'name': 'sub'}])
        also = [{'kind': 'exclude', 'target': 'A-51', 'also': ['A-90', 'A-91'], 'id': 'k1', 'at': '2026-10-31T08:00:00'},
                {'kind': 'confirm', 'target': 'A-51', 'slot': 2, 'also': ['A-90'], 'id': 'k2', 'at': '2026-10-31T08:00:01'},
                {'kind': 'team', 'target': 'A-51', 'team': 'B', 'id': 'k3', 'at': '2026-10-31T08:00:02'}]
        texts = job(answers={'index': [{'@index': {'corrections': also}}]})
        marker = job(answers={'intel': [{'@intel': {'__halftime': MARKER}}], 'index': [{'@index': {'__halftime': MARKER}}]},
                     steps=[{'do': 'key', 'half': 'one', 'key': 'Enter', 'name': 'opened'}, {'do': 'pop', 'mt': 'break', 'on': True, 'name': 'brk'}])
        cls.kits, cls.texts, cls.marker = run(kits, texts, marker)

    def test_the_index_kits_name_the_teams(self):
        k = self.kits
        self.assertEqual(k['goal']['tip'][1], 'Team B (red) scored', 'from index.teamKits, without the coaching report')
        row = next(r for r in k['first']['events']['rows'] if r['id'] == 'goal-41824')
        self.assertIn('Team B (red) scored', row['head'])
        self.assertEqual(k['first']['review']['heads'][:2], [f'Team A (white) {DOT} slots 01{NDASH}10 and GK', f'Team B (red) {DOT} slots 01{NDASH}10 and GK'])
        self.assertEqual(errors(k['last']), [])

    def test_a_substitution_names_its_slots(self):
        k = self.kits
        self.assertEqual(k['sub']['tip'][1], f'Team A (white) {DOT} A-03 {ARROW} A-04 {DOT} A-P95 {ARROW} A-P102', 'the slots, then the people by their ids')
        slots = {(s['team'], s['slot']): s for s in k['first']['review']['slots']}
        e = BY_ID['substitution-29896']
        self.assertIn(f"Substitution {clock(e['t'])} {DOT} A-03 {ARROW} A-04 {DOT} A-P95 {ARROW} A-P102 {DOT} candidate, low confidenceConfirmReject", slots[('A', '3')]['subs'],
                      "under the outgoing player's slot")
        self.assertFalse([t for t in slots[('A', '4')]['subs'] if 'A-P102' in t], "not under the incoming player's")
        none = next(r for r in k['first']['events']['rows'] if r['id'] == 'substitution-3656')
        self.assertIn(f'Team B (red) {DOT} B-? {ARROW} B-? {DOT} B-P13 {ARROW} B-P20', none['head'], 'neither player in a slot')
        self.assertFalse([t for s in k['first']['review']['slots'] if s['team'] == 'B' for t in s['subs'] if f'B-P13 {ARROW}' in t], 'and under no slot of the review')

    def test_a_person_correction_names_its_other_fragments(self):
        c = self.texts['first']['corrections']
        who = f"fragment A-51 ({name_of(ROSTER['A-P49'])})"
        self.assertIn(f'{who} and its 2 other fragments marked as not a player', c)
        self.assertIn(f'{who} and its 1 other fragment is the player of slot 02', c)
        self.assertIn(f'{who} moved to team B', c, 'no others: as before')

    def test_one_halftime_marker_by_default(self):
        m = self.marker
        self.assertEqual([(x['half'], x['text']) for x in m['first']['handles']], [('one', f'HALFTIME {DASH} 47:34')])
        self.assertTrue(m['first']['band']['hidden'])
        o = m['opened']['pop']
        self.assertFalse(o['brk'])
        self.assertEqual([(i['half'], i['hidden']) for i in o['inputs']], [('end', False), ('start', True)])
        b = m['brk']
        self.assertEqual({x['half']: x['now'] for x in b['handles']}, {'end': 2854, 'start': 2854 + 15 * 60}, 'Include the break: a fifteen-minute break to adjust')
        self.assertFalse(b['band']['hidden'])
        self.assertEqual(b['pop']['state'], 'Not saved yet')


@unittest.skipUnless(NODE, 'needs node to run match.js')
class MixedTrack(unittest.TestCase):
    """A stitched track that joins two people (the people's 'mixed': a keeper and an outfield player at a segment
    start): its unresolved row says why no slot takes it, and no other row does."""

    @classmethod
    def setUpClass(cls):
        roster = [dict(p, mixed={'segment': 26, 'why': 'keeper', 'at': 4594.8}) if p['matchId'] == 'A-P24' else p
                  for p in INDEX['roster']]
        (cls.mixed,) = run(job(answers={'index': [{'@index': {'roster': roster}}]}))

    def test_the_unresolved_row_says_why(self):
        r = self.mixed['first']['review']
        self.assertEqual(errors(self.mixed['first']), [])
        row = next(u for u in r['unresolved'] if u['person'] == 'A-P24')
        self.assertIn('Joins two people at 76:35 (a keeper and an outfield player), so no slot takes it', row['text'])
        self.assertFalse([u['person'] for u in r['unresolved'] if u['person'] != 'A-P24' and 'Joins two people' in u['text']])


@unittest.skipUnless(NODE, 'needs node to run match.js')
class BoardGoalEnds(unittest.TestCase):
    """The 2D board names the goal ends by time from the match's direction periods (the team attacking right defends
    the left goal), follows a saved direction at once, and gives the views their own labels back on leaving."""

    @classmethod
    def setUpClass(cls):
        flip = lambda team: 'B' if team == 'A' else 'A'
        flipped = [dict(p, attacksRight=flip(p['attacksRight']), defendsLeft=flip(p['defendsLeft'])) for p in INTEL['directionPeriods']]
        saved = {'ok': True, 'index': '@index', 'intel': building(.5, 'Recognising stoppages, restarts and goals', directionPeriods=flipped,
                                                                   __halftime={'teamAFirstHalf': 'right', 'confirmed': True})}
        cls.out = run(job(endsAt=[100, 2870, 4000, 9999], hold=True, answers={'halftime': [saved]},
                          steps=[{'do': 'key', 'half': 'end', 'key': 'Enter'}, {'do': 'pop', 'side': 'right'}, {'do': 'pop', 'mt': 'save', 'name': 'saved'},
                                 {'do': 'leave', 'name': 'left'}]))[0]

    def test_the_goal_ends_by_time(self):
        o = self.out['first']
        self.assertEqual([p['attacksRight'] for p in INTEL['directionPeriods']], ['B', 'A'])
        self.assertEqual(o['ends'], {'100': {'left': 'B', 'right': 'A'}, '2870': {}, '4000': {'left': 'A', 'right': 'B'}, '9999': None},
                         'first half: B attacks right and defends the left goal; the break: nobody; after the match: the views\' own')
        self.assertEqual(o['endsCalls'], ['fn'])
        self.assertEqual(o['analytics'], {'goalEnds': {'left': 'B', 'right': 'A'}}, 'the panels\' pitch: the first period\'s ends')

    def test_a_saved_direction_is_followed_at_once_and_leaving_gives_the_labels_back(self):
        s = self.out['saved']
        self.assertEqual(post_bodies(s, '/api/match/halftime')[0]['teamAFirstHalf'], 'right')
        self.assertEqual(s['ends']['100'], {'left': 'A', 'right': 'B'}, 'the state being made already says the new directions')
        self.assertEqual(s['ends']['4000'], {'left': 'B', 'right': 'A'})
        self.assertEqual(self.out['left']['endsCalls'], ['fn', None])
        self.assertIsNone(self.out['left']['ends'])


@unittest.skipUnless(NODE, 'needs node to run match.js and analytics.js')
class AnalyticsList(unittest.TestCase):
    """The analytics panel's player list in match playback (the page's analytics.js): the provisional people are
    analysable estimates, grouped with their slot; the unresolved and the left out stay out."""

    @classmethod
    def setUpClass(cls):
        cls.out = run(job(realAnalytics=True, steps=[{'do': 'an-team', 'team': 'B', 'name': 'b'}]))[0]

    def expected(self, team):
        out = {}
        for p in INDEX['roster']:
            if p.get('excluded') or p.get('team') != team or p['display'].endswith('-?'):
                continue
            if p['display'].endswith('?'):
                out[p['matchId']] = f"{p['display']} (estimate) {DOT} {clock(p['firstSeen'])}{NDASH}{clock(p['lastSeen'])} {DOT} {p['matchId']}"
            else:
                out[p['matchId']] = f"{p['display']} {DOT} {p['matchId']}"
        return out

    def test_the_list_holds_the_estimates_and_not_the_unresolved(self):
        for team, snap in (('A', self.out['first']), ('B', self.out['b'])):
            with self.subTest(team=team):
                got = dict(snap['analytics']['options'])
                self.assertEqual(snap['analytics']['team'], team)
                self.assertEqual(got, self.expected(team))
                self.assertTrue(any('(estimate)' in t for t in got.values()))
                self.assertFalse(any(t.startswith(f'{team}-? ') for t in got.values()), 'no unresolved person')
                unresolved = [p['matchId'] for p in INDEX['roster'] if p.get('team') == team and p['display'] == f'{team}-?' and not p.get('excluded')]
                self.assertTrue(unresolved and not set(unresolved) & set(got))
                self.assertFalse(set(got) & {p['matchId'] for p in INDEX['roster'] if p.get('excluded')}, 'nobody left out')
        self.assertEqual(errors(self.out['last']), [])

    def test_grouped_by_slot_sure_first(self):
        texts = [t for _, t in self.out['first']['analytics']['options']]
        slots = [int(t[2:4]) for t in texts if re.match(r'A-\d\d', t)]
        self.assertEqual(slots, sorted(slots), 'the people of a slot together, slot by slot')
        self.assertEqual(texts.index(f"A-01 {DOT} A-P04"), 0, 'the sure player before the estimates of the slot')
        keepers = [t for t in texts if t.startswith('GK-')]
        self.assertEqual(texts[-len(keepers):], keepers, 'the goalkeepers after the outfield players')
        self.assertTrue(keepers[0].startswith('GK-A ') and any(t.startswith('GK-A? (estimate)') for t in keepers))


BOARD = r"""
const fs = require('fs'), vm = require('vm');
const sandbox = {document: {}, console, devicePixelRatio: 1, localStorage: {getItem: () => null, setItem() {}}};
sandbox.window = sandbox;
vm.createContext(sandbox);
vm.runInContext(fs.readFileSync(process.argv[1], 'utf8'), sandbox);
const log = [];
const ctx = new Proxy({}, {get: (o, k) => (k in o ? o[k] : (...a) => log.push([k, ...a])), set: (o, k, v) => { o[k] = v; log.push(['=' + k, v]); return true; }});
const view = sandbox.Tactical.create({canvas: {getContext: () => ctx, clientWidth: 960, width: 0, height: 0}});
const g = view.geometry();
const players = ['A-04?', 'GK-A?', 'A-?', 'A-04'].map((d, k) => ({id: 'p' + k, display: d, x: .2 + .1 * k, y: .5, team: 'A', kind: 'O', alpha: 1, quality: 1, role: d.startsWith('GK') ? 'GOALKEEPER' : 'PLAYER'}));
log.length = 0;
view.drawPlayers(ctx, g, {players}, null);
const out = {};
let font = null, alpha = null, fill = null;
for (const e of log) {
  if (e[0] === '=font') font = e[1];
  else if (e[0] === '=globalAlpha') alpha = e[1];
  else if (e[0] === '=fillStyle') fill = e[1];
  else if (e[0] === 'fillText' && players.some((p) => p.display === e[1])) out[e[1]] = {font, alpha, fill};
}
process.stdout.write(JSON.stringify(out));
"""


BOARD_ENDS = r"""
const fs = require('fs'), vm = require('vm');
const layers = [];
const recorder = (log) => new Proxy({}, {get: (o, k) => (k in o ? o[k] : (...a) => log.push([k, ...a])), set: (o, k, v) => { o[k] = v; return true; }});
const document = {createElement: () => { const layer = {width: 0, height: 0, log: []}; layer.getContext = () => recorder(layer.log); layers.push(layer); return layer; }};
const sandbox = {document, console, devicePixelRatio: 1, localStorage: {getItem: () => null, setItem() {}}};
sandbox.window = sandbox;
vm.createContext(sandbox);
vm.runInContext(fs.readFileSync(process.argv[1], 'utf8'), sandbox);
const view = sandbox.Tactical.create({canvas: {getContext: () => recorder([]), clientWidth: 960, width: 0, height: 0}});
view.load({version: 3, frames: [{frame: 0, time: 0, people: []}, {frame: 25, time: 1, people: []}], match: {players: [], goalkeepers: []},
           tactical: {goalEnds: {left: 'A', right: 'B'}}, video: {fps: 25, width: 1920, height: 1080}}, {video: {fps: 25}});
const labels = () => layers[layers.length - 1].log.filter((e) => e[0] === 'fillText' && / goal$/.test(e[1])).sort((a, b) => a[2] - b[2]).map((e) => e[1]);
const out = {};
view.render(10); out.own = labels(); out.n0 = layers.length;
view.setGoalEnds((t) => (t < 50 ? {left: 'B', right: 'A'} : t < 60 ? {} : null));
view.render(10); out.first = labels(); out.n1 = layers.length;
view.render(20); out.n2 = layers.length;
view.render(55); out.gap = labels(); out.n3 = layers.length;
view.render(70); out.back = labels(); out.n4 = layers.length;
view.setGoalEnds(null); view.render(10); out.cleared = labels();
process.stdout.write(JSON.stringify(out));
"""


@unittest.skipUnless(NODE, 'needs node to run tactical.js')
class BoardLabels(unittest.TestCase):
    def test_the_goal_ends_follow_the_page_by_time(self):
        done = subprocess.run([NODE, '-e', BOARD_ENDS, TACTICAL_JS], capture_output=True, text=True, encoding='utf-8', timeout=60)
        self.assertEqual(done.returncode, 0, done.stderr[-2000:])
        o = json.loads(done.stdout)
        self.assertEqual(o['own'], ['Team A goal', 'Team B goal'], "the result's own ends, left to right")
        self.assertEqual(o['first'], ['Team B goal', 'Team A goal'], "the page's ends for that time")
        self.assertEqual((o['n0'], o['n1'], o['n2']), (1, 2, 2), 'the pitch drawn again only when the ends change')
        self.assertEqual(o['gap'], [], 'between two periods: no label')
        self.assertEqual(o['back'], ['Team A goal', 'Team B goal'], "no answer for that time: the result's own")
        self.assertEqual((o['n3'], o['n4']), (3, 4))
        self.assertEqual(o['cleared'], ['Team A goal', 'Team B goal'])

    def test_a_provisional_slot_label_is_drawn_tentative(self):
        done = subprocess.run([NODE, '-e', BOARD, TACTICAL_JS], capture_output=True, text=True, encoding='utf-8', timeout=60)
        self.assertEqual(done.returncode, 0, done.stderr[-2000:])
        out = json.loads(done.stdout)
        for label in ('A-04?', 'GK-A?'):
            self.assertTrue(out[label]['font'].startswith('italic '), label)
            self.assertEqual(out[label]['fill'], '#c8d3cc')
            self.assertAlmostEqual(out[label]['alpha'], .8)
        for label in ('A-?', 'A-04'):
            self.assertFalse(out[label]['font'].startswith('italic'), label)
            self.assertEqual((out[label]['fill'], out[label]['alpha']), ('#f4f7f4', 1), label)


class Markup(unittest.TestCase):
    """index.html and match.js: the transport's layers, the drawer section, the styles, ASCII and CRLF."""

    @classmethod
    def setUpClass(cls):
        with open(PAGE, encoding='utf-8') as handle:
            cls.page = handle.read()
        with open(MATCH_JS, encoding='utf-8') as handle:
            cls.js = handle.read()

    def test_the_transport_and_the_drawer(self):
        p = self.page
        self.assertIn('<span class="seek"><span class="mt" id="mt" hidden></span><input id="timeline" type="range"', p, 'the layers over the one range')
        self.assertIn('</span><span class="mt-msg" id="mtMsg" role="status" aria-live="polite" hidden></span><span class="sep"></span>', p)
        self.assertRegex(p, r'<div class="page" data-page="edit"><section class="match-only" id="matchEventsSection"><h2>Match events</h2>.*?<div id="matchEvents"')
        for rule in ('#mt[hidden],.mt [hidden],#mtMsg[hidden],#matchEvents [hidden],.rv [hidden]{display:none!important}', '.transport.mt-on{padding-top:28px}',
                     '.mt{position:absolute;left:0;right:0;top:-24px;bottom:0;z-index:3;pointer-events:none}', '.mt-hh{position:absolute;', 'touch-action:none',
                     '.mt-ev.st-candidate{opacity:.72}', '.mt-band{', '.mt-pop{'):
            self.assertIn(rule, p, rule)
        self.assertNotRegex(p, r'innerHTML')
        self.assertNotIn('innerHTML', self.js)

    def test_the_timeline_filter_and_the_goal_marker(self):
        p = self.page
        self.assertIn('aria-label="Match time"></span><span class="mtf" id="mtf" hidden><button id="mtfBtn" type="button" class="mtf-btn" aria-haspopup="dialog" '
                      'aria-expanded="false" aria-controls="mtfPop"', p, 'beside the range, hidden until a match state is shown')
        self.assertIn('<div class="mtf-pop" id="mtfPop" role="dialog" aria-label="Timeline markers" hidden></div></span><span class="mt-msg"', p)
        self.assertIn('<symbol id="i-filter" viewBox="0 0 24 24">', p)
        for rule in ('.mt-ev.k-goal{top:-2px;width:18px;height:18px;margin-left:-9px;z-index:1}', '.mt-ev.k-goal::after{content:\'\';position:absolute;left:50%;top:100%;',
                     'background:var(--team,var(--green));pointer-events:none}', '.mt-ev.k-goal.st-candidate i{background:transparent;box-shadow:inset 0 0 0 2px #F1F5F2',
                     '.mt-scores{', '.mt-score{position:absolute;', '.mtf[hidden],.mtf [hidden]{display:none!important}', '.mtf-pop{position:absolute;right:0;bottom:calc(100% + 8px);',
                     '.mtf-quick button[aria-pressed=true]{', '.mtf.part .mtf-btn::after{'):
            self.assertIn(rule, p, rule)
        self.assertLess(p.index('.mt-ev.k-goal i{'), p.index('.mt-ev.k-goal.st-candidate i{'), 'a candidate goal stays hollow')
        self.assertIn("const MARKS_KEY = 'pitchiq.timelineMarkers.v1';", self.js, 'one versioned key')
        self.assertIn("const MARKS_DEFAULT = {show: ['GOAL'], strip: true};", self.js, 'goals only by default')

    def test_the_hollow_kick_off_triangle_has_a_hole(self):
        found = re.search(r'\.k-ko\.st-candidate i\{clip-path:polygon\(([^)]*)\)\}', self.page)
        self.assertTrue(found)
        points = [tuple(float(v.rstrip('%') or 0) for v in p.split()) for p in found.group(1).split(',')]
        self.assertEqual(points[0], points[3])
        self.assertEqual(points[4], points[7])

        def area(pts):
            return sum(x0*y1-x1*y0 for (x0, y0), (x1, y1) in zip(pts, pts[1:]+pts[:1]))/2
        self.assertLess(area(points[:3])*area(points[4:7]), 0, 'the inner loop runs the other way: a hole')

    def test_ascii_and_crlf(self):
        new = self.js[self.js.index("// ---------- the match's state (Full Match Intelligence)"):self.js.index('// ---------- wiring ----------')]
        self.assertEqual([c for c in new if ord(c) > 127], [], 'the new source is ASCII (typographic characters as escapes)')
        css = self.page[self.page.index('/* Full Match Intelligence (match.js'):self.page.index('</style></head>')]
        self.assertEqual([c for c in css if ord(c) > 127], [])
        for name in ('index.html', 'match.js', 'tactical.js'):
            with open(os.path.join(ROOT, name), 'rb') as handle:
                data = handle.read()
            self.assertEqual(data.count(b'\n'), data.count(b'\r\n'), f'{name} keeps CRLF')
        self.assertIn('init, open, leave, frame, scope, chip, stage, span, renameView, contextOf, waiting,', self.js)


if __name__ == '__main__':
    unittest.main()
