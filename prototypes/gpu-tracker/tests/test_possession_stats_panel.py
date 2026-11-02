"""Ball possession in the review page's Analytics tab: possession_stats.js run in node (skipped without node) with a small
DOM, on the real CAN-ISL answer of GET /api/possession2?match= (tests/possession2_match_example.json.gz, which carries the
range of 600-900 s), on variants of it, on a Segment-run answer cut from it (halves [], the stats of 600-900 s as its
whole range) and on an empty Segment record; and the page's wiring of the third category in index.html, match.js and
server.py.

What is checked:
- the primary bar: the team labels of the answer, the server's percentages shown whole (Team B's the complement of Team
  A's, so the two add up to 100), the segments in the board's team colours at the server's shares, the line "of N:NN
  resolved controlled in-play time (X% of in-play time)"; with no percentages (insufficient) the bar stays empty and
  neutral with "Not enough resolved possession to give percentages", never 50/50;
- the coverage: shareOfLive as five stacked segments and a legend with percentages, the time left out (confirmed and
  possible stoppages, halftime), the record's estimate note where inferred control exceeds observed;
- the range filter: Full match (Entire analysed segment in a Segment run), the halves the record names (none without),
  Custom range in mm:ss or seconds asked of the server with t0 / t1 and shown from its `range`; an invalid custom range
  says why and asks nothing; a range asked once is not asked again; a failed range says why and keeps the shown one;
- the halves compared (a card each, which also chooses the half), the breakdown per team (controlled, observed /
  inferred, sequences, their mean only when there are some, changes won in the range) and of the time neither team had;
- the timeline: one canvas drawn by pixel columns (no DOM node per interval; every interval wide enough shows its own
  kind; inferred at half strength), the change ticks (filled 'high', hollow 'uncertain'), the interval at a time
  (intervalAt), the hover tooltip, a click that selects (the details under the strip) and seeks through the seek
  callback with a pause, Left / Right / Home / End, a change tick's hover and click;
- the states: lazy loading (nothing asked until the category is shown), building (progress, asked again until ready,
  never while hidden), missing and needs-state ("Work out possession" posts {match: <id>} and then asks until ready),
  error and a failed request (one line; a match offers Try again, which posts the build again), an older load's answer
  dropped;
- index.html: the third category tab, its toolbar and dashboard, the script, the fenced creation, the category switch
  (analytics paused unless Player, team shape active on Team shape, possession on Ball possession, the grid hidden),
  the loads from showResult; match.js scopes /api/possession2 and loads the panel with the others; server.py serves
  /possession_stats.js.

Run:  python -m unittest test_possession_stats_panel          (from this directory)
"""
import copy
import gzip
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
NODE = shutil.which('node')
JS = os.path.join(ROOT, 'possession_stats.js')
PAGE = os.path.join(ROOT, 'index.html')
MATCH_JS = os.path.join(ROOT, 'match.js')
SERVER = os.path.join(ROOT, 'server.py')
FIXTURE_GZ = os.path.join(HERE, 'possession2_match_example.json.gz')
A, B = '#b98cff', '#f6b564'   # the board's default team colours
MATCH = '20261028-062138-83f2a6'
DOT, NDASH, ELL = chr(0xb7), chr(0x2013), chr(0x2026)
SHADE = {'contested': '#d5dde6', 'free': '#97a3ad', 'unknown': '#2e3833', 'possible_stoppage': '#1c2621', 'stoppage': '#161e1a',
         'halftime': 'rgba(143,179,217,.22)', 'non_match': '#111815'}
OVERLAYS = ('PATTERN', 'rgba(143,179,217,.6)', 'rgba(241,245,242,.42)')   # the hatch, the half-time edges, the half divider

with gzip.open(FIXTURE_GZ, 'rt', encoding='utf-8') as _handle:
    FIX = json.load(_handle)
REC = FIX['record']
TL = REC['timeline']
N = len(TL['t0'])
FPS = REC['fps']
CODES = REC['codes']
LO, HI = min(REC['t0'], TL['t0'][0]), max(REC['t1'], TL['t1'][-1])
TEAMS = {t: FIX['teams'][t]['label'] for t in 'AB'}


# ---------- the page's formats, in Python (JS Math.round rounds halves up) ----------
def js_round(x):
    return math.floor(x + .5)


def js_num(x):
    """A number as JavaScript's String(x) writes it (37.0 -> '37', 0.0 -> '0')."""
    return str(int(x)) if float(x) == int(x) else repr(float(x))


def percent(p):
    if p is None:
        return NDASH
    q = js_round(p * 10) / 10
    r = js_round(q)
    if q > 0 and r == 0:
        return '<1%'
    if q < 100 and r == 100:
        return '>99%'
    return f'{r}%'


def share(v):
    return percent(None if v is None else v * 100)


def pair(a):
    q = js_round(a * 10) / 10
    r = js_round(q)
    if q > 0 and r == 0:
        return '<1%', '>99%'
    if q < 100 and r == 100:
        return '>99%', '<1%'
    return f'{r}%', f'{100 - r}%'


def span(s):
    s = max(0, js_round(s or 0))
    return f'{s // 60}:{s % 60:02d}'


def clock(t):
    t = max(0, js_round(t))
    return f'{t // 60:02d}:{t % 60:02d}'


def clock_fine(t):
    t = js_round(max(0, t) * 100) / 100
    m = math.floor(t / 60)
    return f'{m:02d}:{t - m * 60:05.2f}'


def brief(s):
    return f'{max(0, s):.1f} s' if s < 59.95 else span(s)


def interval_at(t):
    """The interval holding t (t0 <= t < t1; the last one its own end too), else -1."""
    if t < TL['t0'][0] or t > TL['t1'][-1]:
        return -1
    lo, hi = 0, N - 1
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if TL['t0'][mid] <= t:
            lo = mid
        else:
            hi = mid - 1
    return lo if t < TL['t1'][lo] or (lo == N - 1 and t <= TL['t1'][lo]) else -1


def seek_time(k):
    return max(TL['t0'][k], min(TL['t0'][k] + .7 / FPS, TL['t1'][k] - .1 / FPS))


STATE_TITLE = {'contested': 'Contested ball', 'free': 'Free ball', 'unknown': 'Unresolved', 'possible_stoppage': 'Possible stoppage',
               'stoppage': 'Confirmed stoppage', 'halftime': 'Half-time', 'non_match': 'Not match footage'}


def describe(k, teams=TEAMS):
    """The title and lines of interval k (its tooltip, and the selection under the strip)."""
    name = CODES[TL['state'][k]]
    if name in 'AB':
        title = f"{teams[name]} in possession {DOT} {'observed' if TL['designation'][k] == 1 else 'inferred'}"
    else:
        title = STATE_TITLE[name]
    if name == 'possible_stoppage' and TL['decoded'][k] >= 0:
        d = CODES[TL['decoded'][k]]
        title += f" (play read as {teams[d] if d in 'AB' else STATE_TITLE[d].lower()})"
    parts = [f"{clock_fine(TL['t0'][k])} {NDASH} {clock_fine(TL['t1'][k])}", brief(TL['t1'][k] - TL['t0'][k])]
    if TL['confidence'][k] >= 0:
        parts.append(f"confidence {TL['confidence'][k]}%")
    if TL['possessor'][k] >= 0:
        p = TL['people'][TL['possessor'][k]]
        parts.append(f"on the ball: {p['label'] or p['id']}")
    if TL['ballSeen'][k] >= 0:
        parts.append(f"ball seen on {TL['ballSeen'][k]}% of its frames")
    if name in ('possible_stoppage', 'stoppage', 'halftime', 'non_match'):
        parts.append('left out of the shares')
    return title, parts


def won(lo=None, hi=None):
    out = {'A': 0, 'B': 0}
    for c in REC['changes']:
        if c['to'] in out and (lo is None or lo <= c['wonAt'] < hi):
            out[c['to']] += 1
    return out


def team_rows(S, team, wins):
    s = S['sequences'][team]
    rows = [['Controlled time', span(S['seconds'][team])], ['Observed / inferred', f"{span(S['observed'][team])} / {span(S['inferred'][team])}"],
            ['Possession sequences', str(s['count'])]]
    if s['count'] > 0 and s['meanSeconds'] is not None:
        rows.append(['Mean sequence', brief(s['meanSeconds'])])
    rows.append(['Changes won', str(wins[team])])
    return rows


def neither_rows(S):
    return [['Contested', span(S['seconds']['contested'])], ['Free ball', span(S['seconds']['free'])], ['Unresolved', span(S['seconds']['unknown'])]]


FULL_PCTS = list(pair(REC['stats']['match']['percent']['A']))   # the whole match's two percentages as shown
COVERAGE = [('observed', 'Observed control'), ('inferred', 'Inferred control'), ('contested', 'Contested'), ('free', 'Free ball'), ('unknown', 'Unresolved')]


def left_out(S):
    sec = S['seconds']
    return f"Left out: confirmed stoppages {span(sec['stoppage'])}, possible stoppages {span(sec['possible_stoppage'])}, halftime {span(sec['halftime'])}"


# ---------- answers ----------
def segment_answer(lo=600.0, hi=900.0):
    """A Segment run's answer cut from the match: the intervals of [lo, hi) (the ends clipped), no halves, the stats of
    that range (the fixture's own `range`) as its whole range, Team A / Team B labels."""
    idx = [i for i in range(N) if TL['t1'][i] > lo and TL['t0'][i] < hi]
    cols = {k: [TL[k][i] for i in idx] for k in ('t0', 't1', 'state', 'designation', 'confidence', 'possessor', 'ballSeen', 'decoded')}
    cols['t0'][0], cols['t1'][-1] = max(cols['t0'][0], lo), min(cols['t1'][-1], hi)
    used = sorted({p for p in cols['possessor'] if p >= 0})
    remap = {p: k for k, p in enumerate(used)}
    cols['possessor'] = [remap[p] if p >= 0 else -1 for p in cols['possessor']]
    cols['people'] = [TL['people'][p] for p in used]
    stats = {k: v for k, v in FIX['range'].items() if k not in ('t0', 't1')}
    record = dict(copy.deepcopy(REC), t0=cols['t0'][0] + .02, t1=cols['t1'][-1] - .02, halves=[], timeline=cols, stats={'match': stats},
                  changes=[c for c in REC['changes'] if lo <= c['wonAt'] < hi],
                  source={'kind': 'segment', 'state': "the run's dead-ball spells", 'people': 'segment identities'})
    record.pop('fingerprint', None)
    return {'state': 'ready', 'record': record, 'definition': FIX['definition'], 'teams': {t: {'kit': None, 'label': f'Team {t}'} for t in 'AB'}}


def empty_answer():
    """A Segment record with nothing in it (what possession_v2.stats_from_runs gives over no runs)."""
    zero = {k: 0.0 for k in CODES}
    stats = {'seconds': zero, 'live': 0.0, 'resolved': 0.0, 'observed': {'A': 0.0, 'B': 0.0}, 'inferred': {'A': 0.0, 'B': 0.0}, 'percent': None,
             'insufficient': True, 'coverage': None, 'shareOfLive': None, 'sequences': {t: {'count': 0, 'meanSeconds': None} for t in 'AB'}, 'minResolved': 30.0}
    record = {'version': REC['version'], 'model': REC['model'], 'definition': REC['definition'], 'settings': REC['settings'], 't0': 0.0, 't1': 0.0, 'fps': 25.0,
              'halves': [], 'codes': CODES, 'timeline': {k: [] for k in ('t0', 't1', 'state', 'designation', 'confidence', 'possessor', 'ballSeen', 'decoded', 'people')},
              'changes': [], 'stats': {'match': stats}, 'ballStates': {}, 'notes': ['Less than 30 s of resolved controlled time: no percentages.'],
              'source': {'kind': 'segment', 'state': 'none (no stoppages known)', 'people': 'segment identities'}}
    return {'state': 'ready', 'record': record, 'definition': FIX['definition'], 'teams': {t: {'kit': None, 'label': f'Team {t}'} for t in 'AB'}}


def state(kind, **extra):
    return dict({'state': kind, 'teams': FIX['teams'], 'definition': FIX['definition']}, **extra)


INSUFFICIENT = {'@base': 'match', 'set': [[['record', 'stats', 'match', 'percent'], None], [['record', 'stats', 'match', 'insufficient'], True]]}
MATCH_RESULT = {'version': 3, 'matchLevel': {'matchId': MATCH, 'segments': 33}, 'match': {'players': []}, 'frames': [], 'tactical': {}}
SEGMENT_RESULT = {'version': 3, 'match': {'players': []}, 'frames': [{'frame': 15000, 'time': 600.0, 'people': []}], 'tactical': {}}

HARNESS = r"""
const fs = require('fs'), vm = require('vm'), zlib = require('zlib');
const input = JSON.parse(fs.readFileSync(0, 'utf8'));
const BASES = {match: JSON.stringify(JSON.parse(zlib.gunzipSync(fs.readFileSync(input.fixture)).toString('utf8'))), segment: JSON.stringify(input.segment),
               empty: JSON.stringify(input.empty)};
const copy = (x) => (x === undefined ? undefined : JSON.parse(JSON.stringify(x)));
const settle = async (n = 60) => { for (let k = 0; k < n; k++) await new Promise((r) => setImmediate(r)); };
// '@match' / '@segment' / '@empty': a base answer; {'@base', set: [[path, value]], drop: [key]}: a variant of one
function token(v) {
  if (typeof v === 'string' && v[0] === '@') return JSON.parse(BASES[v.slice(1)]);
  if (v && typeof v === 'object' && v['@base']) {
    const out = JSON.parse(BASES[v['@base']]);
    for (const [path, value] of v.set || []) { let o = out; for (const k of path.slice(0, -1)) o = o[k]; o[path[path.length - 1]] = value; }
    for (const k of v.drop || []) delete out[k];
    return out;
  }
  return copy(v);
}
// A 2D context that logs every call with the state it was made in.
function recorder(log) {
  const s = {fillStyle: '#000000', strokeStyle: '#000000', globalAlpha: 1, lineWidth: 1, dash: []};
  return new Proxy(s, {
    get: (o, k) => k in o ? o[k] : k === 'createPattern' ? () => 'PATTERN' : k === 'setLineDash' ? (d) => { o.dash = d.slice(); } : k === 'getLineDash' ? () => o.dash.slice()
      : (...a) => { log.push([k, ...a.map((x) => typeof x === 'number' ? Math.round(x * 1000) / 1000 : typeof x === 'string' ? x : null),
                              {fill: String(o.fillStyle), stroke: String(o.strokeStyle), alpha: o.globalAlpha, dash: o.dash.slice()}]); },
    set: (o, k, v) => { o[k] = v; return true; }});
}

class Node {
  constructor(tag) {
    this.tagName = String(tag).toUpperCase(); this.children = []; this.parentElement = null; this.attrs = {}; this.dataset = {}; this.style = {};
    this.listeners = {}; this.hidden = false; this.disabled = false; this.value = ''; this.own = ''; this.cls = ''; this.title = ''; this.id = '';
    this.type = ''; this.clientWidth = 0; this.rect = null; this.log = null; this.width = 0; this.height = 0; this.max = 1;
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
  setAttribute(k, v) { this.attrs[k] = String(v); }
  getAttribute(k) { return k in this.attrs ? this.attrs[k] : null; }
  removeAttribute(k) { delete this.attrs[k]; }
  addEventListener(t, f) { (this.listeners[t] = this.listeners[t] || []).push(f); }
  fire(t, extra) {
    const e = Object.assign({type: t, target: this, defaultPrevented: false, propagationStopped: false,
                             preventDefault() { this.defaultPrevented = true; }, stopPropagation() { this.propagationStopped = true; }}, extra || {});
    for (const f of (this.listeners[t] || []).slice()) f(e);
    return e;
  }
  click() { if (!this.disabled) this.fire('click'); }
  getContext() { return recorder(this.log = this.log || []); }
  getBoundingClientRect() { return this.rect || {left: 0, top: 0, width: 0, height: 0}; }
}
const all = (n, out = []) => { out.push(n); for (const c of n.children || []) all(c, out); return out; };
const find = (root, pred) => all(root).filter((n) => n instanceof Node && pred(n));
const has = (n, c) => n instanceof Node && n.classes().includes(c);
let hold = false, held = [];
const sandbox = {document: {createElement: (t) => new Node(t), createTextNode: (t) => ({nodeType: 3, textContent: String(t), parentElement: null})},
                 console, URLSearchParams, devicePixelRatio: 1,
                 setTimeout: (f) => { if (hold) { held.push(f); return {held: held.length - 1}; } return setImmediate(f); },
                 clearTimeout: (h) => { if (h && h.held !== undefined) held[h.held] = null; else if (h) clearImmediate(h); }};
sandbox.window = sandbox;
vm.createContext(sandbox);
vm.runInContext(fs.readFileSync(input.js, 'utf8'), sandbox, {filename: input.js});
const INFO = {video: {fps: 25, duration: 5800, width: 1920, height: 1080}};

function setup(job) {
  const el = (tag) => new Node(tag || 'div');
  const s = {root: el('div'), range: el('select'), custom: el('span'), t0: el('input'), t1: el('input'), asked: [], posted: [], seeks: [], keys: [],
             answers: copy(job.answers || ['@match']), builds: copy(job.builds || [])};
  s.custom.hidden = true;
  const next = (list) => {
    if (!list.length) throw Object.assign(new Error('NOT FOUND'), {status: 404});
    const a = list.length > 1 ? list.shift() : list[0];
    if (a && a.__status) throw Object.assign(new Error(a.__error || 'failed'), {status: a.__status});
    return token(a);
  };
  const api = async (url) => { s.asked.push(url); return next(s.answers); };
  const post = async (url, body) => { s.posted.push({url, body: copy(body)}); return next(s.builds); };
  s.view = sandbox.PossessionStats.create({root: s.root, controls: {range: s.range, custom: s.custom, t0: s.t0, t1: s.t1}, api, post,
                                           run: () => (job.run === undefined ? 'run1' : job.run), match: () => null, board: {colours: {A: input.A, B: input.B}},
                                           colourInputs: [], seek: (t, pause) => s.seeks.push([t, pause])});
  s.strip = find(s.root, (n) => has(n, 'ps-strip'))[0];
  s.strip.clientWidth = 1000; s.strip.rect = {left: 100, top: 0, width: 1000, height: 42};
  return s;
}
const X = (s, t) => { const v = s.view.shown; return 100 + (t - v.lo) / (v.hi - v.lo) * 1000; };
function strip(s) {
  const log = s.strip.log || [];
  let k = log.length - 1;
  while (k >= 0 && log[k][0] !== 'clearRect') k--;
  const draw = k >= 0 ? log.slice(k) : [];
  const state = (c) => c[c.length - 1];
  return {width: s.strip.width, height: s.strip.height, draws: log.filter((c) => c[0] === 'clearRect').length,
          fills: draw.filter((c) => c[0] === 'fillRect').map((c) => ({x: c[1], y: c[2], w: c[3], h: c[4], fill: state(c).fill, alpha: state(c).alpha})),
          ticksFilled: draw.filter((c) => c[0] === 'fill').map((c) => state(c).fill),
          strokes: draw.filter((c) => c[0] === 'stroke').map((c) => ({stroke: state(c).stroke, dash: state(c).dash})),
          outlines: draw.filter((c) => c[0] === 'strokeRect').map((c) => ({x: c[1], w: c[3], stroke: state(c).stroke}))};
}
function summary(s) {
  const v = s.view, E = v.el, kids = (n) => n.children.filter((c) => c instanceof Node);
  return {
    state: {hidden: E.state.hidden, text: E.state.textContent, kinds: kids(E.state).map((c) => c.dataset.ps || c.className),
            buttons: find(E.state, (n) => n.tagName === 'BUTTON').map((b) => ({ps: b.dataset.ps, text: b.textContent, disabled: b.disabled})),
            progress: find(E.state, (n) => n.tagName === 'PROGRESS').map((p) => p.value)},
    bodyHidden: E.body.hidden, bodyCls: E.body.className,
    hero: {range: E.range.textContent, names: [E.a.label.textContent, E.b.label.textContent], swatches: [E.a.sw.style.background, E.b.sw.style.background],
           pcts: [E.a.pct.textContent, E.b.pct.textContent], empty: E.bar.classList.contains('empty'),
           segs: [E.segA, E.segB].map((n) => ({hidden: n.hidden, flex: n.style.flex || null, background: n.style.background || null})),
           none: {hidden: E.none.hidden, text: E.none.textContent}, sub: E.sub.textContent, text: E.range.parentElement.parentElement.textContent},
    coverage: {live: E.live.textContent, stackHidden: E.stack.hidden, segs: E.covSegs.map((n) => ({key: n.dataset.key, flex: n.style.flex || null, hidden: n.hidden})),
               legend: kids(E.covLegend).map((c) => ({key: c.dataset.key || null, text: c.textContent})), left: E.left.textContent,
               est: {hidden: E.est.hidden, text: E.est.textContent}},
    halves: {hidden: E.halves.hidden, row: E.row.className,
             cards: kids(E.halfList).map((b) => ({half: b.dataset.half, pressed: b.attrs['aria-pressed'], parts: Object.fromEntries(kids(b).map((c) => [c.className, c.textContent]))}))},
    timeline: {range: E.tlRange.textContent, axis: [E.axis0.textContent, E.axis1.textContent], detail: E.detail.textContent, key: kids(E.key).map((c) => c.textContent),
               aria: Object.assign({}, E.strip.attrs), tabIndex: E.strip.tabIndex},
    tip: {hidden: E.tip.hidden, lines: kids(E.tip).map((c) => c.textContent), left: E.tip.style.left || null, cls: E.tip.className},
    tables: kids(E.tables).map((t) => t.tagName === 'TABLE' ? {team: t.dataset.team, caption: t.children[0].textContent,
                                                                  rows: kids(t.children[1]).map((tr) => kids(tr).map((c) => c.textContent))} : {text: t.textContent}),
    how: E.howBody.textContent,
    options: kids(s.range).map((o) => [o.value, o.textContent]), rangeValue: s.range.value, customHidden: s.custom.hidden, inputs: [s.t0.value, s.t1.value],
    selected: v.selected, filter: v.filter, shown: v.shown ? {lo: v.shown.lo, hi: v.shown.hi, kind: v.shown.kind} : null,
    asked: s.asked.slice(), posted: copy(s.posted), seeks: s.seeks.slice(), keys: s.keys.slice(), nodes: all(s.root).length, rootText: s.root.textContent,
    strip: strip(s)};
}
async function act(s, step, job) {
  const v = s.view;
  if (step.do === 'filter') { s.range.value = step.value; s.range.fire('change'); }
  else if (step.do === 'type') { s.t0.value = step.t0; s.t1.value = step.t1; if (step.change !== false) s.t1.fire('change'); }
  else if (step.do === 'enter') { s.t0.value = step.t0; s.t0.fire('keydown', {key: 'Enter'}); }
  else if (step.do === 'click') s.strip.fire('click', {clientX: step.x !== undefined ? step.x : X(s, step.t), clientY: step.y === undefined ? 30 : step.y});
  else if (step.do === 'hover') s.strip.fire('pointermove', {clientX: step.x !== undefined ? step.x : X(s, step.t), clientY: step.y === undefined ? 30 : step.y});
  else if (step.do === 'leave') s.strip.fire('pointerleave');
  else if (step.do === 'key') { const e = s.strip.fire('keydown', {key: step.key}); s.keys.push({key: step.key, prevented: e.defaultPrevented, stopped: e.propagationStopped}); }
  else if (step.do === 'button') {
    const b = find(s.root, (n) => n.tagName === 'BUTTON' && n.dataset.ps === step.ps)[0];
    if (!b) throw new Error('no button ' + step.ps + ' in ' + s.root.textContent.slice(0, 300));
    b.click();
  } else if (step.do === 'half') find(s.root, (n) => has(n, 'ps-half') && n.dataset.half === String(step.k))[0].click();
  else if (step.do === 'active') v.setActive(step.value);
  else if (step.do === 'load') v.load(step.result === undefined ? job.result : step.result, INFO);
  else if (step.do === 'release') { hold = step.hold !== false; for (const f of held.splice(0)) if (f) f(); }
  else if (step.do === 'answers') s.answers = copy(step.answers);
  else throw new Error('unknown step ' + step.do);
}
async function scenario(job) {
  hold = !!job.hold; held = [];
  const s = setup(job), out = {};
  s.view.load(job.result, INFO);
  await settle();
  out.inactive = summary(s);
  if (job.activate !== false) { s.view.setActive(true); await settle(); }
  out.first = summary(s);
  for (const step of job.steps || []) { await act(s, step, job); await settle(); if (step.name) out[step.name] = summary(s); }
  out.last = summary(s);
  out.held = held.filter(Boolean).length;
  hold = false; held = [];
  return out;
}
(async () => {
  const out = [];
  for (const job of input.jobs) out.push(await scenario(job));
  process.stdout.write(JSON.stringify(out));
})().catch((error) => { process.stderr.write(String(error && error.stack || error)); process.exit(1); });
"""


def run(*jobs):
    payload = {'js': JS, 'fixture': FIXTURE_GZ, 'segment': segment_answer(), 'empty': empty_answer(), 'A': A, 'B': B,
               'jobs': [dict({'result': MATCH_RESULT}, **j) for j in jobs]}
    done = subprocess.run([NODE, '-e', HARNESS], input=json.dumps(payload), capture_output=True, text=True, encoding='utf-8', timeout=300)
    if done.returncode:
        raise AssertionError(done.stderr[-3000:])
    return json.loads(done.stdout)


def node(script):
    done = subprocess.run([NODE, '-e', script], capture_output=True, text=True, encoding='utf-8', timeout=120)
    if done.returncode:
        raise AssertionError(done.stderr[-3000:])
    return json.loads(done.stdout)


def bar_columns(fills, width=1000):
    """The colour each pixel column of the bar was filled with (the runs, without the overlays drawn over them)."""
    cols = [None] * width
    for f in fills:
        if f['y'] != 12 or f['h'] != 30 or f['fill'] in OVERLAYS:
            continue
        for c in range(int(round(f['x'])), int(round(f['x'] + f['w']))):
            if 0 <= c < width:
                cols[c] = (f['fill'], f['alpha'])
    return cols


def kind_colour(k):
    name = CODES[TL['state'][k]]
    if name in 'AB':
        return ({'A': A, 'B': B}[name], 1 if TL['designation'][k] == 1 else .5)
    return (SHADE[name], 1)


def longest(name, designation=None):
    best = max((i for i in range(N) if CODES[TL['state'][i]] == name and (designation is None or TL['designation'][i] == designation)),
               key=lambda i: TL['t1'][i] - TL['t0'][i])
    return best


@unittest.skipUnless(NODE, 'needs node to run the page scripts')
class FullMatch(unittest.TestCase):
    """The CAN-ISL answer as it is shown: the primary bar, the coverage, the halves, the timeline, the breakdown."""

    @classmethod
    def setUpClass(cls):
        cls.out = run({})[0]
        cls.o = cls.out['first']

    def test_nothing_is_asked_until_the_category_is_shown(self):
        lazy = self.out['inactive']
        self.assertEqual(lazy['asked'], [], 'loaded while inactive: nothing asked')
        self.assertTrue(lazy['bodyHidden'])
        self.assertEqual(self.o['asked'], ['/api/possession2'], 'a match: no run, the page adds match=<id> (match.js SCOPED)')
        self.assertFalse(self.o['bodyHidden'])
        self.assertTrue(self.o['state']['hidden'], 'nothing to say above a ready dashboard')

    def test_the_primary_bar(self):
        h, S = self.o['hero'], REC['stats']['match']
        self.assertEqual(h['names'], ['Team A (white)', 'Team B (red)'], 'the labels of the answer')
        self.assertEqual(h['swatches'], [A, B], 'beside each label, the board colour')
        self.assertFalse(S['insufficient'])
        self.assertEqual(h['pcts'], list(pair(S['percent']['A'])))
        self.assertEqual(int(h['pcts'][0].rstrip('%')) + int(h['pcts'][1].rstrip('%')), 100, 'the two shown add up to 100')
        self.assertEqual(h['pcts'][0], percent(S['percent']['A']))
        self.assertFalse(h['empty'])
        self.assertEqual(h['segs'], [{'hidden': False, 'flex': f"{js_num(S['percent']['A'])} 1 0px", 'background': A},
                                     {'hidden': False, 'flex': f"{js_num(S['percent']['B'])} 1 0px", 'background': B}])
        self.assertTrue(h['none']['hidden'])
        self.assertEqual(h['sub'], f"of {span(S['resolved'])} resolved controlled in-play time ({percent(100 * S['resolved'] / S['live'])} of in-play time)")
        self.assertRegex(h['sub'], r'^of \d+:\d\d resolved controlled in-play time \(\d+% of in-play time\)$')
        self.assertEqual(h['range'], f'Full match {DOT} {clock(LO)}{NDASH}{clock(HI)}')
        self.assertNotIn('50%', h['text'])

    def test_the_coverage(self):
        c, S = self.o['coverage'], REC['stats']['match']
        sh = S['shareOfLive']
        self.assertEqual([s['key'] for s in c['segs']], [k for k, _ in COVERAGE])
        self.assertEqual([s['flex'] for s in c['segs']], [f'{js_num(sh[k])} 1 0px' for k, _ in COVERAGE], 'stacked at the shares of live time')
        self.assertEqual([s['hidden'] for s in c['segs']], [not sh[k] > 0 for k, _ in COVERAGE])
        self.assertEqual(c['legend'], [{'key': k, 'text': f'{name} {share(sh[k])}'} for k, name in COVERAGE])
        self.assertAlmostEqual(sum(sh.values()), 1, places=2)
        self.assertEqual(c['left'], left_out(S))
        self.assertRegex(c['left'], r'^Left out: confirmed stoppages \d+:\d\d, possible stoppages \d+:\d\d, halftime \d+:\d\d$')
        self.assertEqual(c['live'], f"{span(S['live'])} in play {DOT} coverage {share(S['coverage'])}")
        note = next((n for n in REC['notes'] if 'inferred' in n), 'More of the controlled time is inferred than directly observed: the shares are an estimate.')
        self.assertEqual(c['est'], {'hidden': False, 'text': note} if sh['inferred'] > sh['observed'] else {'hidden': True, 'text': ''},
                         "where inferred control exceeds observed, the record's note: the shares are an estimate")

    def test_the_halves_compared(self):
        hv = self.o['halves']
        self.assertFalse(hv['hidden'])
        self.assertNotIn('one', hv['row'])
        self.assertEqual(len(hv['cards']), 2)
        for card, half in zip(hv['cards'], REC['halves']):
            S = REC['stats'][half['name']]
            a, b = pair(S['percent']['A'])
            self.assertEqual(card['parts'], {'ps-half-name': half['name'].capitalize(), 'ps-half-when': f"{clock(half['t0'])}{NDASH}{clock(half['t1'])}", 'ps-mini': '',
                                             'ps-half-pct': f'A {a} / B {b}', 'ps-half-cov': f"coverage {share(S['coverage'])} of {span(S['live'])} in play"})
            self.assertEqual(card['pressed'], 'false')
        self.assertEqual([c['parts']['ps-half-name'] for c in hv['cards']], ['First half', 'Second half'])

    def test_the_breakdown(self):
        S, wins = REC['stats']['match'], won()
        self.assertEqual(sum(wins.values()), len(REC['changes']), 'the whole match counts every change')
        self.assertEqual(self.o['tables'], [{'team': 'A', 'caption': TEAMS['A'], 'rows': team_rows(S, 'A', wins)},
                                            {'team': 'B', 'caption': TEAMS['B'], 'rows': team_rows(S, 'B', wins)},
                                            {'team': 'none', 'caption': 'Neither team', 'rows': neither_rows(S)}])
        self.assertEqual([r[0] for r in self.o['tables'][0]['rows']], ['Controlled time', 'Observed / inferred', 'Possession sequences', 'Mean sequence', 'Changes won'])

    def test_how_this_is_counted(self):
        how = self.o['how']
        self.assertIn(FIX['definition'], how)
        prov = REC['model']['provenance']
        self.assertIn(f"Model {REC['version']}, fitted {prov['fitted']} on {prov['data']} ({prov['tool']}); model id {REC['model']['id']}.", how)
        self.assertIn(f"at least {percent(100 * REC['settings']['minPosterior'])}, else the time is unresolved", how)
        self.assertIn('percentages need at least 0:30 of resolved controlled time', how)
        bs = REC['ballStates']
        self.assertIn('The ball over in-play time: ' + ', '.join(f'{k.lower()} {share(v)}' for k, v in bs.items()) + '.', how)
        self.assertIn('Source: the whole match (33 segments); stoppages from match state; players from match people.', how)
        for note in REC['notes']:
            self.assertIn(note, how)

    def test_the_timeline_is_drawn_by_pixel_columns(self):
        st, tl = self.o['strip'], self.o['timeline']
        self.assertEqual((st['width'], st['height']), (1000, 42))
        self.assertEqual(tl['range'], f'Full match {DOT} {clock(LO)}{NDASH}{clock(HI)}')
        self.assertEqual(tl['axis'], [clock(LO), clock(HI)])
        self.assertLess(self.o['nodes'], 400, f'no DOM node per interval ({N} intervals)')
        runs = [f for f in st['fills'] if f['y'] == 12 and f['h'] == 30 and f['fill'] not in OVERLAYS]
        self.assertLessEqual(len(runs), 1000, 'at most one rectangle per pixel column')
        self.assertLess(len(runs), N / 4)
        xs = sorted(runs, key=lambda f: f['x'])
        for p, q in zip(xs, xs[1:]):
            self.assertLessEqual(p['x'] + p['w'], q['x'] + 1e-6, 'runs never overlap')
        cols = bar_columns(st['fills'])
        checked = 0
        for i in range(N):
            xa, xb = (TL['t0'][i] - LO) / (HI - LO) * 1000, (TL['t1'][i] - LO) / (HI - LO) * 1000
            for c in range(math.ceil(xa), math.floor(xb)):
                self.assertEqual(cols[c], kind_colour(i), f'interval {i} ({CODES[TL["state"][i]]}) at column {c}')
                checked += 1
        self.assertGreater(checked, 100, 'the long intervals show their own kind on every column they fill')
        hatch = [f for f in st['fills'] if f['fill'] == 'PATTERN']
        self.assertEqual(bool(hatch), (SHADE['unknown'], 1) in cols, 'unresolved runs are hatched ...')
        self.assertTrue(all(cols[int(round(f['x']))] == (SHADE['unknown'], 1) for f in hatch), '... and only they')
        dotted = [s for s in st['strokes'] if s['stroke'] == '#9aaa9f']
        self.assertEqual(bool(dotted), (SHADE['possible_stoppage'], 1) in cols, 'a possible stoppage drawn has a dotted top edge')
        self.assertTrue(all(s['dash'] == [2, 2] for s in dotted))
        self.assertEqual(tl['tabIndex'], 0)
        self.assertEqual(tl['aria']['role'], 'slider')

    def test_the_change_ticks(self):
        st = self.o['strip']
        high = [c for c in REC['changes'] if c['confidence'] == 'high']
        uncertain = [c for c in REC['changes'] if c['confidence'] != 'high']
        self.assertEqual(sorted(st['ticksFilled']), sorted({'A': A, 'B': B}[c['to']] for c in high), 'a high-confidence change: a filled tick in the winner\'s colour')
        hollow = [s['stroke'] for s in st['strokes'] if s['stroke'] in (A, B)]
        self.assertEqual(sorted(hollow), sorted({'A': A, 'B': B}[c['to']] for c in uncertain), 'an uncertain one: hollow')
        self.assertTrue(high and uncertain, 'the example has both')


@unittest.skipUnless(NODE, 'needs node to run the page scripts')
class Filter(unittest.TestCase):
    """The range filter: Full match, the halves, a custom range (mm:ss or seconds), and the half cards."""

    @classmethod
    def setUpClass(cls):
        cls.out = run({'steps': [{'do': 'filter', 'value': 'half:0', 'name': 'firstHalf'}, {'do': 'filter', 'value': 'half:1', 'name': 'secondHalf'},
                                 {'do': 'type', 't0': '10:00', 't1': '15:00', 'name': 'typed'}, {'do': 'filter', 'value': 'custom', 'name': 'custom'},
                                 {'do': 'filter', 'value': 'full', 'name': 'full'}, {'do': 'filter', 'value': 'custom', 'name': 'again'},
                                 {'do': 'type', 't0': 'abc', 't1': '15:00', 'name': 'invalid'}, {'do': 'type', 't0': '600', 't1': '900', 'name': 'seconds'},
                                 {'do': 'half', 'k': 1, 'name': 'card'}]},
                      {'steps': [{'do': 'filter', 'value': 'half:0'}, {'do': 'filter', 'value': 'custom', 'name': 'prefilled'},
                                 {'do': 'filter', 'value': 'half:1'}, {'do': 'load', 'name': 'reloaded'}]},
                      {'answers': ['@match', {'__status': 400, '__error': 't0 must be a number of seconds.'}],
                       'steps': [{'do': 'type', 't0': '10:00', 't1': '15:00'}, {'do': 'filter', 'value': 'custom', 'name': 'failed'}]})

    def test_the_options(self):
        o = self.out[0]['first']
        self.assertEqual(self.out[0]['inactive']['options'], [['full', 'Full match'], ['custom', 'Custom range']], 'before the record: no halves')
        self.assertEqual(o['options'], [['full', 'Full match'], ['half:0', 'First half'], ['half:1', 'Second half'], ['custom', 'Custom range']])
        self.assertEqual(self.out[0]['first']['rangeValue'], 'full')

    def test_the_halves(self):
        for name, k in (('firstHalf', 0), ('secondHalf', 1)):
            o, half = self.out[0][name], REC['halves'][k]
            S = REC['stats'][half['name']]
            with self.subTest(half=half['name']):
                self.assertEqual(o['hero']['pcts'], list(pair(S['percent']['A'])))
                self.assertEqual(o['hero']['segs'][0]['flex'], f"{S['percent']['A']} 1 0px")
                self.assertEqual(o['hero']['range'], f"{half['name'].capitalize()} {DOT} {clock(half['t0'])}{NDASH}{clock(half['t1'])}")
                self.assertEqual(o['timeline']['range'], o['hero']['range'])
                self.assertEqual(o['shown'], {'lo': half['t0'], 'hi': half['t1'], 'kind': 'half'})
                self.assertEqual(o['coverage']['legend'][0]['text'], f"Observed control {share(S['shareOfLive']['observed'])}")
                self.assertEqual(o['coverage']['left'], left_out(S))
                wins = won(half['t0'], half['t1'])
                self.assertEqual(o['tables'][0]['rows'], team_rows(S, 'A', wins))
                self.assertEqual(o['tables'][1]['rows'], team_rows(S, 'B', wins))
                self.assertEqual([c['pressed'] for c in o['halves']['cards']], ['true' if j == k else 'false' for j in range(2)])
                self.assertEqual(o['asked'], ['/api/possession2'], "the halves' stats are in the record: nothing asked")
                self.assertTrue(o['customHidden'])

    def test_a_custom_range_is_asked_with_t0_t1_and_shown_from_its_range(self):
        typed, o = self.out[0]['typed'], self.out[0]['custom']
        self.assertEqual(typed['asked'], ['/api/possession2'], 'times typed while another range is chosen ask nothing')
        R = FIX['range']
        self.assertEqual((R['t0'], R['t1']), (600.0, 900.0))
        self.assertEqual(o['asked'], ['/api/possession2', '/api/possession2?t0=600&t1=900'])
        self.assertFalse(o['customHidden'])
        self.assertEqual(o['hero']['pcts'], list(pair(R['percent']['A'])))
        self.assertNotEqual(R['percent'], REC['stats']['match']['percent'], 'the range differs from the whole match')
        self.assertEqual(o['hero']['sub'], f"of {span(R['resolved'])} resolved controlled in-play time ({percent(100 * R['resolved'] / R['live'])} of in-play time)")
        self.assertEqual(o['hero']['range'], f'Custom range {DOT} 10:00{NDASH}15:00')
        self.assertEqual(o['shown'], {'lo': 600, 'hi': 900, 'kind': 'custom'})
        self.assertEqual(o['coverage']['legend'], [{'key': k, 'text': f"{name} {share(R['shareOfLive'][k])}"} for k, name in COVERAGE])
        self.assertEqual(o['tables'][0]['rows'], team_rows(R, 'A', won(600, 900)))
        self.assertEqual(o['tables'][2]['rows'], neither_rows(R))
        self.assertEqual(o['timeline']['axis'], ['10:00', '15:00'])
        self.assertEqual([c['pressed'] for c in o['halves']['cards']], ['false', 'false'])
        cols = bar_columns(o['strip']['fills'])
        self.assertIn((A, 1), cols, 'zoomed in, observed control shows in full colour ...')
        self.assertIn((A, .5), cols, '... and inferred at half strength')

    def test_back_to_the_full_match_and_the_range_asked_once(self):
        full, again = self.out[0]['full'], self.out[0]['again']
        self.assertEqual(full['hero']['pcts'], FULL_PCTS)
        self.assertTrue(full['customHidden'])
        self.assertEqual(again['hero']['pcts'], list(pair(FIX['range']['percent']['A'])))
        self.assertEqual(len(again['asked']), 2, 'the range asked once is not asked again')

    def test_an_invalid_custom_range_says_why_and_asks_nothing(self):
        o = self.out[0]['invalid']
        self.assertEqual(o['state']['kinds'], ['notice'])
        self.assertIn('Enter both times of the custom range, as mm:ss or seconds.', o['state']['text'])
        self.assertEqual(len(o['asked']), 2)
        self.assertEqual(o['hero']['pcts'], list(pair(FIX['range']['percent']['A'])), 'the range shown stays')
        s = self.out[0]['seconds']
        self.assertTrue(s['state']['hidden'])
        self.assertEqual(len(s['asked']), 2, '600 to 900 in seconds is the range already asked')
        self.assertEqual(s['hero']['range'], f'Custom range {DOT} 10:00{NDASH}15:00')

    def test_a_half_card_chooses_its_half(self):
        o = self.out[0]['card']
        self.assertEqual((o['filter'], o['rangeValue']), ('half:1', 'half:1'))
        self.assertEqual(o['hero']['pcts'], list(pair(REC['stats']['second half']['percent']['A'])))
        self.assertTrue(o['customHidden'])

    def test_a_first_custom_range_starts_as_the_range_shown(self):
        o = self.out[1]['prefilled']
        h = REC['halves'][0]
        self.assertEqual(o['inputs'], [clock(h['t0']), clock(h['t1'])])
        self.assertEqual(o['asked'][-1], f"/api/possession2?t0={js_round(h['t0'])}&t1={js_round(h['t1'])}")

    def test_a_reload_keeps_the_range_chosen(self):
        o = self.out[1]['reloaded']
        self.assertEqual(o['asked'][-1], '/api/possession2', 'a correction (a new load) asks again')
        self.assertEqual((o['filter'], o['rangeValue']), ('half:1', 'half:1'), 'and the half chosen comes back with the new record')
        self.assertEqual(o['hero']['range'].split(f' {DOT} ')[0], 'Second half')

    def test_a_failed_range_says_why_and_keeps_the_range_shown(self):
        o = self.out[2]['failed']
        self.assertEqual(o['state']['kinds'], ['notice'])
        self.assertIn('The range could not be worked out. t0 must be a number of seconds.', o['state']['text'])
        self.assertFalse(o['bodyHidden'])
        self.assertEqual(o['hero']['pcts'], FULL_PCTS, 'the full match stays shown')
        self.assertNotIn('ps-wait', o['bodyCls'])


@unittest.skipUnless(NODE, 'needs node to run the page scripts')
class Timeline(unittest.TestCase):
    """The interval at a time, hover, click to select and seek, the keys, and the change ticks."""

    @classmethod
    def setUpClass(cls):
        cls.stop = longest('stoppage')
        cls.binf = longest('B')
        span_px = (HI - LO) / 1000
        isolated = [c for c in REC['changes'] if c['confidence'] == 'high' and LO < c['wonAt'] < HI
                    and all(d is c or abs(d['wonAt'] - c['wonAt']) > 3 * span_px for d in REC['changes'])]
        cls.change = isolated[0]
        mid = lambda k: (TL['t0'][k] + TL['t1'][k]) / 2  # noqa: E731
        cls.out = run({'steps': [{'do': 'click', 't': mid(cls.stop), 'name': 'clicked'}, {'do': 'key', 'key': 'ArrowRight', 'name': 'right'},
                                 {'do': 'key', 'key': 'ArrowLeft'}, {'do': 'key', 'key': 'ArrowLeft', 'name': 'left'},
                                 {'do': 'key', 'key': 'Home', 'name': 'home'}, {'do': 'key', 'key': 'ArrowLeft', 'name': 'homeLeft'},
                                 {'do': 'key', 'key': 'End', 'name': 'end'}, {'do': 'key', 'key': 'Escape', 'name': 'other'},
                                 {'do': 'hover', 't': mid(cls.binf), 'name': 'hover'}, {'do': 'leave', 'name': 'left2'},
                                 {'do': 'hover', 't': cls.change['wonAt'], 'y': 5, 'name': 'hoverChange'},
                                 {'do': 'click', 't': cls.change['wonAt'], 'y': 5, 'name': 'clickChange'},
                                 {'do': 'filter', 'value': 'half:1', 'name': 'otherHalf'}]})[0]

    def test_the_interval_at_a_time(self):
        times = sorted({TL['t0'][0] - 1, TL['t0'][0], TL['t1'][-1], TL['t1'][-1] + .01, 600.0, 900.0, 2854.0, 1234.567, float('nan')}
                       | {TL['t0'][k] for k in range(0, N, 97)} | {TL['t1'][k] for k in range(0, N, 89)}
                       | {(TL['t0'][k] + TL['t1'][k]) / 2 for k in range(0, N, 61)}, key=lambda t: (t != t, t))
        script = r"""
const fs = require('fs'), vm = require('vm'), zlib = require('zlib');
const sandbox = {document: {}, console};
sandbox.window = sandbox;
vm.createContext(sandbox);
vm.runInContext(fs.readFileSync(__JS__, 'utf8'), sandbox);
const P = sandbox.PossessionStats, rec = JSON.parse(zlib.gunzipSync(fs.readFileSync(__FIX__)).toString('utf8')).record, cols = P.columns(rec);
const gap = P.columns({timeline: {t0: [0, 1, 5], t1: [1, 2, 6], state: [0, 1, 2], designation: [1, 2, 0], confidence: [90, 80, -1], possessor: [-1, -1, -1], ballSeen: [100, 0, 50], decoded: [-1, -1, -1], people: []}});
process.stdout.write(JSON.stringify({at: __TIMES__.map((t) => P.intervalAt(cols, t === null ? NaN : t)), n: cols.n,
  gap: [-.1, 0, .5, 1, 1.999, 2, 3, 5, 6, 6.01].map((t) => P.intervalAt(gap, t)), empty: P.intervalAt(P.columns({timeline: {t0: [], t1: [], state: [], designation: [], confidence: [], possessor: [], ballSeen: [], decoded: []}}), 1),
  broken: P.columns({timeline: {t0: [1, 2], t1: [2]}}), parse: ['10:00', '600', '1:02:03', '96:31.5', ' 7 ', '10:60', 'abc', '', '1:2:3:4', '-5', '5.'].map(P.parseTime)}));
""".replace('__JS__', json.dumps(JS)).replace('__FIX__', json.dumps(FIXTURE_GZ)).replace('__TIMES__', json.dumps([None if t != t else t for t in times]))
        out = node(script)
        self.assertEqual(out['n'], N)
        self.assertEqual(out['at'], [interval_at(t) if t == t else -1 for t in times])
        self.assertEqual(out['gap'], [-1, 0, 0, 1, 1, -1, -1, 2, 2, -1], 'half-open, the last one holds its end, a gap holds nothing')
        self.assertEqual((out['empty'], out['broken']), (-1, None))
        self.assertEqual(out['parse'], [600, 600, 3723, 5791.5, 7, None, None, None, None, None, None], 'mm:ss, seconds, h:mm:ss; nothing else')

    def test_a_click_selects_the_interval_and_seeks_to_its_start(self):
        o, k = self.out['clicked'], self.stop
        self.assertGreater(TL['t1'][k] - TL['t0'][k], 30, 'the longest stoppage spans several pixels')
        self.assertEqual(o['selected'], k)
        self.assertEqual(len(o['seeks']), 1)
        self.assertAlmostEqual(o['seeks'][0][0], seek_time(k), places=9)
        self.assertAlmostEqual(o['seeks'][0][0], TL['t0'][k] + .7 / FPS, places=9, msg='.2 frame into its first frame')
        self.assertIs(o['seeks'][0][1], True, 'paused, through the seek possession.js gets')
        title, parts = describe(k)
        self.assertEqual(o['timeline']['detail'], f"{title} {DOT} {f' {DOT} '.join(parts)}. Playback was moved to {clock_fine(seek_time(k))}: open Tracking or 2D match to watch it.")
        self.assertTrue(o['timeline']['detail'].startswith('Confirmed stoppage'))
        self.assertEqual(o['timeline']['aria']['aria-valuenow'], str(js_round(TL['t0'][k])))
        outline = o['strip']['outlines']
        self.assertEqual(len(outline), 1)
        self.assertAlmostEqual(outline[0]['x'], (TL['t0'][k] - LO) / (HI - LO) * 1000 + 1, delta=.01)
        self.assertEqual(self.out['first']['timeline']['detail'][:20], 'Hover the timeline t', 'before a selection: how to use it')

    def test_the_keys_step_through_the_intervals(self):
        k = self.stop
        self.assertEqual(self.out['right']['selected'], k + 1)
        self.assertAlmostEqual(self.out['right']['seeks'][-1][0], seek_time(k + 1), places=9)
        self.assertEqual(self.out['left']['selected'], k - 1)
        self.assertEqual(self.out['home']['selected'], 0)
        self.assertEqual(self.out['homeLeft']['selected'], 0, 'never before the first interval of the range')
        self.assertEqual(self.out['end']['selected'], N - 1)
        self.assertEqual(self.out['other']['selected'], N - 1, 'other keys do nothing')
        self.assertEqual(len(self.out['end']['seeks']), 1 + 6, 'each step seeks (Left on the first interval stays there)')
        keys = self.out['last']['keys']
        self.assertTrue(all(x['prevented'] and x['stopped'] for x in keys if x['key'] != 'Escape'), 'the page never also steps the video')
        self.assertFalse(any(x['prevented'] for x in keys if x['key'] == 'Escape'))
        tip = self.out['end']['tip']
        self.assertFalse(tip['hidden'], 'the keys show what hover shows')
        self.assertEqual(tip['lines'], [describe(N - 1)[0], *describe(N - 1)[1]])

    def test_hover_names_the_interval_under_the_pointer(self):
        o, k = self.out['hover'], self.binf
        self.assertGreater((TL['t1'][k] - TL['t0'][k]) / (HI - LO) * 1000, 1, 'wider than the pixel it is hovered on')
        self.assertFalse(o['tip']['hidden'])
        title, parts = describe(k)
        self.assertEqual(o['tip']['lines'], [title, *parts])
        self.assertTrue(title.startswith('Team B (red) in possession'))
        self.assertTrue(any(p.startswith('confidence ') for p in parts) and any(p.startswith('ball seen on ') for p in parts))
        self.assertTrue(self.out['left2']['tip']['hidden'])
        self.assertEqual(len(o['seeks']), len(self.out['end']['seeks']), 'hover never seeks')

    def test_a_change_tick_hovers_and_seeks_to_where_the_ball_was_won(self):
        c, o = self.change, self.out['hoverChange']
        lost = c.get('lostByLabel') or c.get('lostBy')
        won = c.get('wonByLabel') or c.get('wonBy')
        lines = [f"{TEAMS[c['to']]} won the ball"] + ([f'{lost} lost it'] if lost else []) + ([f'{won} won it'] if won else [])
        when = f"lost {clock_fine(c['lostAt'])}, won {clock_fine(c['wonAt'])}" if c['lostAt'] is not None and c['wonAt'] - c['lostAt'] > .05 else clock_fine(c['wonAt'])
        self.assertEqual(o['tip']['lines'], lines + [when, 'high confidence'])
        k = self.out['clickChange']
        self.assertAlmostEqual(k['seeks'][-1][0], c['wonAt'] + .2 / FPS, places=6)
        self.assertIs(k['seeks'][-1][1], True)
        self.assertEqual(k['selected'], interval_at(c['wonAt']))

    def test_a_selection_outside_the_new_range_is_dropped(self):
        o = self.out['otherHalf']
        k = interval_at(self.change['wonAt'])
        inside = TL['t1'][k] > REC['halves'][1]['t0'] and TL['t0'][k] < REC['halves'][1]['t1']
        self.assertEqual(o['selected'], k if inside else -1)
        self.assertFalse(self.out['clickChange']['tip']['hidden'])
        self.assertTrue(o['tip']['hidden'], 'a tooltip placed for the strip as it was is gone once the range changes')


@unittest.skipUnless(NODE, 'needs node to run the page scripts')
class States(unittest.TestCase):
    """Building, missing, needs-state, error, a failed request, the insufficient case, a Segment run, an empty record."""

    @classmethod
    def setUpClass(cls):
        building = [state('building', progress=.4, stage='Reading segment 3 of 33'), state('building', progress=.85, stage='Working out possession over the whole match'), '@match']
        cls.out = run(
            {'hold': True, 'answers': building, 'steps': [{'do': 'release', 'name': 'poll2'}, {'do': 'release', 'name': 'ready'}]},
            {'hold': True, 'answers': building, 'steps': [{'do': 'active', 'value': False}, {'do': 'release', 'name': 'hidden'},
                                                          {'do': 'active', 'value': True, 'name': 'shown'}]},
            {'answers': [state('missing', error=None), state('building', progress=.2, stage='Reading segment 1 of 33'), '@match'],
             'builds': [state('building', progress=0.0, stage='starting')], 'steps': [{'do': 'button', 'ps': 'build', 'name': 'built'}]},
            {'answers': [state('needs-state', error=None), '@match'], 'builds': [state('building', progress=0.0, stage='starting')], 'hold': True,
             'steps': [{'do': 'button', 'ps': 'build', 'name': 'built'}, {'do': 'release', 'hold': False, 'name': 'ready'}]},
            {'answers': [state('error', error='Possession could not be worked out: boom')], 'builds': [state('building', progress=0.0, stage='starting')], 'hold': True,
             'steps': [{'do': 'button', 'ps': 'retry', 'name': 'retried'}]},
            {'answers': [{'__status': 500, '__error': 'The possession record could not be served'}]},
            {'answers': [INSUFFICIENT]},
            {'result': SEGMENT_RESULT, 'answers': ['@segment'], 'steps': [{'do': 'type', 't0': '11:00', 't1': '12:00'}, {'do': 'filter', 'value': 'custom', 'name': 'custom'}]},
            {'result': SEGMENT_RESULT, 'answers': ['@empty']},
            {'result': None},
            {'hold': True, 'answers': [state('building', progress=.4, stage='x'), '@segment'],
             'steps': [{'do': 'load', 'result': SEGMENT_RESULT}, {'do': 'release', 'name': 'reloaded'}]},
        )

    def test_building_shows_its_progress_and_is_asked_again_until_ready(self):
        o = self.out[0]
        f = o['first']['state']
        self.assertEqual(f['kinds'], ['building'])
        self.assertEqual(f['progress'], [.4])
        self.assertIn('Working out possession', f['text'])
        self.assertIn('Reading segment 3 of 33.', f['text'])
        self.assertIn('40%', f['text'])
        self.assertTrue(o['first']['bodyHidden'])
        self.assertEqual(o['poll2']['state']['progress'], [.85])
        self.assertEqual(len(o['poll2']['asked']), 2)
        self.assertEqual(len(o['ready']['asked']), 3)
        self.assertFalse(o['ready']['bodyHidden'])
        self.assertEqual(o['ready']['hero']['pcts'], FULL_PCTS)
        self.assertEqual(o['held'], 0, 'no poll once ready')

    def test_no_polling_while_the_category_is_hidden(self):
        o = self.out[1]
        self.assertEqual(len(o['hidden']['asked']), 1, 'hidden: the poll stops')
        self.assertEqual(len(o['shown']['asked']), 2, 'shown again: asked at once')

    def test_missing_offers_to_work_out_possession_and_posts_the_match(self):
        o = self.out[2]
        f = o['first']['state']
        self.assertEqual(f['kinds'], ['missing'])
        self.assertIn('Possession has not been worked out for this match yet.', f['text'])
        self.assertEqual(f['buttons'], [{'ps': 'build', 'text': 'Work out possession', 'disabled': False}])
        b = o['built']
        self.assertEqual(b['posted'], [{'url': '/api/possession2/build', 'body': {'match': MATCH}}])
        self.assertEqual(b['asked'][0], '/api/possession2')
        self.assertGreaterEqual(len(b['asked']), 3, 'asked until ready')
        self.assertFalse(b['bodyHidden'])
        self.assertEqual(b['hero']['pcts'], FULL_PCTS)

    def test_needs_state_says_the_match_state_comes_first(self):
        o = self.out[3]
        f = o['first']['state']
        self.assertEqual(f['kinds'], ['needs-state'])
        self.assertIn('Possession needs the match state first.', f['text'])
        self.assertIn('match state (stoppages, half-time, restarts)', f['text'])
        b = o['built']
        self.assertEqual(b['posted'], [{'url': '/api/possession2/build', 'body': {'match': MATCH}}])
        self.assertEqual(b['state']['kinds'], ['building'])
        self.assertIn('Starting.', b['state']['text'])
        self.assertEqual(b['state']['progress'], [0])
        self.assertEqual(len(b['asked']), 1, 'the poll waits its turn')
        self.assertEqual(len(o['ready']['asked']), 2)
        self.assertEqual(o['ready']['hero']['pcts'], FULL_PCTS)

    def test_an_error_is_one_line_and_a_match_can_try_again(self):
        o = self.out[4]
        f = o['first']['state']
        self.assertEqual(f['kinds'], ['error'])
        self.assertEqual(f['text'], 'Possession could not be worked out: boom.Try again')
        self.assertEqual(f['buttons'], [{'ps': 'retry', 'text': 'Try again', 'disabled': False}])
        self.assertTrue(o['first']['bodyHidden'])
        self.assertEqual(o['retried']['posted'], [{'url': '/api/possession2/build', 'body': {'match': MATCH}}])
        self.assertEqual(o['retried']['state']['kinds'], ['building'])

    def test_a_failed_request_is_one_line(self):
        f = self.out[5]['first']
        self.assertEqual(f['state']['kinds'], ['error'])
        self.assertEqual(f['state']['text'], 'Possession is not shown. The possession record could not be served.Try again')
        self.assertTrue(f['bodyHidden'])

    def test_not_enough_resolved_possession_gives_no_percentages(self):
        o = self.out[6]['first']
        h, S = o['hero'], REC['stats']['match']
        self.assertEqual(h['pcts'], [NDASH, NDASH])
        self.assertTrue(h['empty'], 'an empty neutral bar')
        self.assertEqual([s['hidden'] for s in h['segs']], [True, True])
        self.assertEqual(h['none'], {'hidden': False, 'text': 'Not enough resolved possession to give percentages'})
        self.assertEqual(h['sub'], f"{span(S['resolved'])} of resolved controlled in-play time in this range; percentages need at least 0:30.")
        self.assertNotIn('50', h['text'])
        self.assertNotIn(FULL_PCTS[0], h['text'])

    def test_a_segment_run_has_no_halves(self):
        o = self.out[7]
        f, seg = o['first'], segment_answer()
        S = seg['record']['stats']['match']
        self.assertEqual(o['inactive']['asked'], [])
        self.assertEqual(f['asked'], ['/api/possession2?run=run1'], 'a Segment run by its name')
        self.assertEqual(f['options'], [['full', 'Entire analysed segment'], ['custom', 'Custom range']])
        self.assertTrue(f['halves']['hidden'])
        self.assertIn('one', f['halves']['row'])
        self.assertEqual(f['hero']['names'], ['Team A', 'Team B'])
        self.assertEqual(f['hero']['pcts'], list(pair(S['percent']['A'])))
        self.assertTrue(f['hero']['range'].startswith(f'Entire analysed segment {DOT} 10:00{NDASH}'))
        self.assertEqual(f['tables'][0]['rows'], team_rows(S, 'A', won(600, 900)))
        self.assertIn('Source: this segment run;', f['how'])
        self.assertNotIn('Half-time', f['timeline']['key'], 'no half-time in the key without halves')
        self.assertEqual(o['custom']['asked'][-1], '/api/possession2?run=run1&t0=660&t1=720')

    def test_an_empty_segment_record(self):
        f = self.out[8]['first']
        self.assertFalse(f['bodyHidden'])
        self.assertEqual(f['options'], [['full', 'Entire analysed segment'], ['custom', 'Custom range']])
        self.assertEqual(f['hero']['pcts'], [NDASH, NDASH])
        self.assertTrue(f['hero']['empty'])
        self.assertEqual(f['hero']['none']['text'], 'Not enough resolved possession to give percentages')
        self.assertTrue(f['coverage']['stackHidden'])
        self.assertEqual(f['coverage']['legend'], [{'key': None, 'text': 'No in-play time in this range.'}])
        self.assertTrue(f['halves']['hidden'])
        self.assertEqual(f['tables'][0]['rows'], [['Controlled time', '0:00'], ['Observed / inferred', '0:00 / 0:00'], ['Possession sequences', '0'], ['Changes won', '0']])
        self.assertEqual(f['strip']['ticksFilled'], [])
        self.assertFalse([x for x in f['strip']['fills'] if x['fill'] not in OVERLAYS], 'nothing to draw')

    def test_no_result(self):
        o = self.out[9]
        self.assertEqual(o['first']['asked'], [])
        self.assertTrue(o['first']['bodyHidden'])
        self.assertEqual(o['first']['state']['text'], 'Ball possession appears here after a run.')

    def test_an_older_loads_answer_is_dropped(self):
        o = self.out[10]
        self.assertEqual(o['first']['state']['kinds'], ['building'])
        r = o['reloaded']
        self.assertEqual(r['asked'], ['/api/possession2', '/api/possession2?run=run1'], 'the new load asks for its own run at once')
        self.assertEqual(r['hero']['names'], ['Team A', 'Team B'])
        self.assertEqual(o['held'], 0, "the old load's poll was dropped")


# ---------- the page ----------
def page():
    with open(PAGE, encoding='utf-8') as handle:
        return handle.read()


def page_script(text):
    return '\n'.join(re.findall(r'<script>(.*?)</script>', text, re.S))


class PageWiring(unittest.TestCase):
    """index.html, match.js and server.py around the module."""

    @classmethod
    def setUpClass(cls):
        cls.page = page()
        cls.src = page_script(cls.page)
        with open(MATCH_JS, encoding='utf-8') as handle:
            cls.match = handle.read()
        with open(SERVER, encoding='utf-8') as handle:
            cls.server = handle.read()
        with open(JS, encoding='utf-8') as handle:
            cls.js = handle.read()

    def test_the_markup(self):
        p = self.page
        self.assertIn('<button type="button" data-cat="team" role="tab" aria-selected="false">Team shape</button>'
                      '<button type="button" data-cat="possession" role="tab" aria-selected="false">Ball possession</button></div>', p, 'the third category tab')
        bar = re.search(r'<div class="toolbar" data-cat-bar="possession" hidden>(.*?)</div></div><div class="an-grid">', p)
        self.assertTrue(bar, 'its toolbar, after the team toolbar in the head')
        self.assertIn('<select id="psRange"><option value="full">Entire analysed segment</option><option value="custom">Custom range</option></select>', bar.group(1))
        self.assertRegex(bar.group(1), r'<span id="psCustom" hidden><input type="text" id="psT0"[^>]*> ' + NDASH + r' <input type="text" id="psT1"[^>]*></span>$')
        self.assertIn('<p id="anStatus"></p></aside></div><div class="ps" id="psDash" role="region" aria-label="Ball possession" hidden></div></div></section>', p,
                      'the dashboard after the grid, inside the analytics panel')
        self.assertRegex(p, r'<script src="/analytics.js"></script>\s*<script src="/possession_stats.js"></script>\s*<script src="/teamshape.js"></script>')
        self.assertRegex(p, r'<script src="/coaching.js"></script>\s*<script src="/possession.js"></script>\s*<script>')
        self.assertIn('.an-grid[hidden],#psDash[hidden],.ps [hidden],#psCustom[hidden]{display:none!important}', p, 'the flex and grid rows hide')
        self.assertNotRegex(p, r'innerHTML')
        self.assertNotIn('innerHTML', self.js, 'labels and reasons are never parsed as HTML')

    def test_the_wiring(self):
        src = self.src
        line = next(l for l in src.split('\n') if l.startswith('const possessionStats=(()=>{try{return PossessionStats.create('))
        for part in ("root:$('psDash')", "controls:{range:$('psRange'),custom:$('psCustom'),t0:$('psT0'),t1:$('psT1')}", 'api:path=>api(path)', 'post:(path,body)=>api(path,body)',
                     'run:()=>resultRun', 'match:()=>(window.matchMode&&window.matchMode.id)||null', 'board:tactical', "colourInputs:[$('teamAColour'),$('teamBColour')]",
                     'seek:(t,pause)=>{if(mode)return;if(pause)video.pause();playback.seek(t);}'):
            self.assertIn(part, line)
        self.assertLess(src.index('const possessionStats='), src.index('const analyticsCats='), 'made before the categories that switch it')
        self.assertIn("analytics.setPaused(cat!=='player');teamShape.setActive(cat==='team');possessionStats.setActive(cat==='possession');", src)
        self.assertIn("if(grid)grid.hidden=cat==='possession';$('psDash').hidden=cat!=='possession';", src)
        self.assertIn('teamShape.load(null,info);coaching.load(null,info);possession.load(null,info);possessionStats.load(null,info);', src)
        self.assertIn('teamShape.load(data,info);coaching.load(data,info);possession.load(data,info);possessionStats.load(data,info);', src)

    @unittest.skipUnless(NODE, 'needs node to check the scripts')
    def test_the_scripts_parse(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'page.js')
            with open(path, 'w', encoding='utf-8') as handle:
                handle.write(self.src)
            for script in (path, JS, MATCH_JS):
                done = subprocess.run([NODE, '--check', script], capture_output=True, text=True)
                self.assertEqual(done.returncode, 0, done.stderr)

    @unittest.skipUnless(NODE, 'needs node to run the page script')
    def test_the_category_switch(self):
        line = next(l for l in self.src.split('\n') if l.startswith('const analyticsCats=(()=>{'))
        script = r"""
const calls = [];
const mk = (attrs) => Object.assign({hidden: false, dataset: {}, attrs: {}, cls: new Set(), listeners: {},
  classList: {toggle(c, on) { on ? this.owner.cls.add(c) : this.owner.cls.delete(c); }}, setAttribute(k, v) { this.attrs[k] = v; },
  addEventListener(t, f) { this.listeners[t] = f; }}, attrs);
const own = (e) => { e.classList.owner = e; return e; };
const buttons = ['player', 'team', 'possession'].map((c) => own(mk({dataset: {cat: c}})));
const tabs = ['player', 'team'].map((c) => own(mk({dataset: {catTabs: c}}))), bars = ['player', 'team', 'possession'].map((c) => own(mk({dataset: {catBar: c}})));
const grid = own(mk({})), els = {analyticsPanel: own(mk({})), psDash: own(mk({hidden: true}))};
const document = {querySelectorAll: (q) => q.includes('.an-cats') ? buttons : q.includes('data-cat-tabs') ? tabs : bars, querySelector: () => grid};
const $ = (id) => els[id];
const analytics = {setPaused: (v) => calls.push('paused ' + v)}, teamShape = {setActive: (v) => calls.push('team ' + v)};
const possessionStats = {setActive: (v) => calls.push('possession ' + v)};
const setTimeout = (f) => f(), fitStage = () => calls.push('fit');
LINE
const out = {};
for (const c of ['possession', 'team', 'player', 'nonsense']) {
  calls.length = 0;
  buttons.find((b) => b.dataset.cat === c) ? buttons.find((b) => b.dataset.cat === c).listeners.click() : analyticsCats.set(c);
  out[c] = {cat: analyticsCats.get(), calls: calls.slice(), grid: grid.hidden, dash: els.psDash.hidden, bars: bars.map((b) => b.hidden), tabs: tabs.map((b) => b.hidden),
            selected: buttons.map((b) => b.attrs['aria-selected']), mode: [...els.analyticsPanel.cls].sort()};
}
process.stdout.write(JSON.stringify(out));
""".replace('LINE', line)
        out = node(script)
        self.assertEqual(out['possession'], {'cat': 'possession', 'calls': ['paused true', 'team false', 'possession true', 'fit'], 'grid': True, 'dash': False,
                                             'bars': [True, True, False], 'tabs': [True, True], 'selected': ['false', 'false', 'true'], 'mode': ['possession-mode']})
        self.assertEqual(out['team'], {'cat': 'team', 'calls': ['paused true', 'team true', 'possession false', 'fit'], 'grid': False, 'dash': True,
                                       'bars': [True, False, True], 'tabs': [True, False], 'selected': ['false', 'true', 'false'], 'mode': ['team-mode']})
        self.assertEqual(out['player']['calls'], ['paused false', 'team false', 'possession false', 'fit'])
        self.assertEqual((out['player']['grid'], out['player']['dash'], out['player']['mode']), (False, True, []))
        self.assertEqual(out['nonsense']['cat'], 'player')

    @unittest.skipUnless(NODE, 'needs node to run the page script')
    def test_a_failing_script_never_stops_the_page(self):
        line = next(l for l in self.src.split('\n') if l.startswith('const possessionStats=(()=>{'))
        script = r"""
const out = [];
for (const kind of ['missing', 'throws', 'works']) {
  const els = {}, $ = (id) => els[id] || (els[id] = {id, textContent: ''});
  const asked = [], api = (path, body) => { asked.push([path, body === undefined ? null : body]); return null; }, tactical = {}, resultRun = 'run1';
  const window = {matchMode: null};
  let mode = null; const calls = [];
  const video = {pause: () => calls.push('pause')}, playback = {seek: (t) => calls.push('seek ' + t)};
  const console = {error: () => {}};
  let options = null;
  const PossessionStats = kind === 'missing' ? undefined : {create: (o) => { if (kind === 'throws') throw new Error('boom'); options = o; return {load() {}, setActive() {}, real: true}; }};
  LINE
  possessionStats.load(null, {}); possessionStats.setActive(true);
  const entry = {kind, real: !!possessionStats.real, message: els.psDash ? els.psDash.textContent : ''};
  if (options) {
    options.seek(12.5, true); options.seek(3, false); mode = 'boundary'; options.seek(7, true);
    options.api('/api/possession2'); options.post('/api/possession2/build', {match: 'm1'});
    entry.calls = calls; entry.asked = asked; entry.run = options.run(); entry.match = [options.match()];
    window.matchMode = {id: 'm1'}; entry.match.push(options.match());
    entry.board = options.board === tactical; entry.root = options.root.id;
    entry.controls = Object.fromEntries(Object.entries(options.controls).map(([k, v]) => [k, v.id])); entry.inputs = options.colourInputs.map((r) => r.id);
  }
  out.push(entry);
}
process.stdout.write(JSON.stringify(out));
""".replace('LINE', line)
        out = node(script)
        self.assertEqual([o['real'] for o in out], [False, False, True], 'missing or throwing: a stand-in that does nothing')
        self.assertIn('Ball possession could not be loaded (boom); the other views are unaffected.', out[1]['message'])
        w = out[2]
        self.assertEqual(w['calls'], ['pause', 'seek 12.5', 'seek 3'], 'pauses when asked, seeks; nothing while a drawing mode is on')
        self.assertEqual(w['asked'], [['/api/possession2', None], ['/api/possession2/build', {'match': 'm1'}]])
        self.assertEqual((w['run'], w['match'], w['board'], w['root']), ('run1', [None, 'm1'], True, 'psDash'))
        self.assertEqual(w['controls'], {'range': 'psRange', 'custom': 'psCustom', 't0': 'psT0', 't1': 'psT1'})
        self.assertEqual(w['inputs'], ['teamAColour', 'teamBColour'])

    def test_match_js_scopes_the_route_and_loads_the_panel(self):
        m = self.match
        scoped = json.loads(re.search(r"const SCOPED = (\[[^\]]*\]);", m).group(1).replace("'", '"'))
        self.assertIn('/api/possession2', scoped)

        def scope(path):   # match.js scope()
            return any(path == p or path.startswith(p + '?') or path.startswith(p + '/') for p in scoped)
        self.assertTrue(scope('/api/possession2') and scope('/api/possession2?t0=600&t1=900') and scope('/api/possession2/build'))
        self.assertIn("function statsPanel() { try { return typeof possessionStats !== 'undefined' && possessionStats ? possessionStats : null; } catch (e) { return null; } }", m)
        self.assertIn("['possession', possession], ['ball possession', statsPanel()]]) {\r\n    if (!panel) continue;", m.replace('\r\n', '\n').replace('\n', '\r\n'))
        self.assertIn('for (const panel of [analytics, teamShape, coaching, possession, statsPanel()]) { if (!panel) continue; try { panel.load(null, info); }', m)

    def test_the_server_serves_the_script(self):
        self.assertIn("@app.get('/possession_stats.js')\n    def possession_stats_script():\n        return send_file(ROOT/'possession_stats.js', mimetype='application/javascript')",
                      self.server.replace('\r\n', '\n'))
        if HERE not in sys.path:
            sys.path.insert(0, HERE)
        try:
            from test_possession_server import Scratch
        except ImportError as error:   # the server's own dependencies (Flask, OpenCV) are missing here
            self.skipTest(f'cannot load the server: {error}')
        s = Scratch()
        self.addCleanup(s.tmp.cleanup)
        self.addCleanup(s.gate.set)
        shutil.copyfile(JS, s.root/'possession_stats.js')
        with s.get('/possession_stats.js') as r:
            self.assertEqual((r.status_code, r.mimetype), (200, 'application/javascript'))
            with open(JS, 'rb') as handle:
                self.assertEqual(r.data, handle.read())
            self.assertEqual(r.headers['Cache-Control'], 'no-store')


if __name__ == '__main__':
    unittest.main()
