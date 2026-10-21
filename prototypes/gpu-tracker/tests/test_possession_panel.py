"""The possession panel of the 2D match view: possession.js run in node (skipped without node) on a hand-written record
in the contract's shape (soccer/possession.py's compute(); the record here is a fixture, the engine has its own tests),
with the real tactical.js board, and the page's wiring of the panel in index.html.

The panel shows the live line at the video time (team in possession and the carrier by display label, ball in flight,
loose, contested, out of play with the restarting team, unknown, estimated and low confidence said, outside the analysed
segment), the possession share per team of the named time with the coverage, a strip in the board's team colours
(control solid, flights and loose balls fainter, estimates hatched, unknown grey, dead ball dark) that repaints when the
colours change, and one marker per change whose tooltip names the players by the display label of the result shown now
(after a correction, the new labels), and seeks the shared playback 0.2 frame into the frame the ball was won (pausing);
on that frame the line says who won the ball and the ring marks the winner, even where the run there is unknown (the
engine's posterior is below the operating point right at a change); a click on the strip seeks there. Unknown is said
plainly (the engine's unknown runs are low confidence and often estimated). Markers closer than their width share
the space between them, each triangle on its own time. An uncertain change (possession-4's 'uncertain', or a
possession-3 record's 'low'; possession-3's 'medium' stays solid) is a hollow triangle with "uncertain" in its tooltip
and on its won frame's line, shown by default: the view ticks "Show uncertain changes" when it is made (the markup
leaves the box unticked); unticked, an uncertain change is hidden (no marker, no won line, no ring) until it is ticked
again; the count of uncertain changes sits by the option, which is disabled where there is none. A one-line key under
the strip says that changes are estimates. A record that is not available, and a
request that failed, say why in one line (held to two lines by the page, the whole text in its tooltip). The answer of an
older load is dropped. The ring is drawn on the board under the carrier's marker (an overlays layer of the real board),
dashed for an estimate, and at the carrier's feet on the video. A frame of playback writes nothing to the page unless
the line or the playhead moved; the live line's tooltip is the whole line. The page fences the panel like the coaching
report: without possession.js the rest of the page works. The heights the page reserves for the panel add up to its rows,
and the uncertain triangle is a hollow outline (its clip path's inner loop runs the other way).

Run:  python -m unittest test_possession_panel          (from this directory)
"""
import json
import os
import re
import shutil
import subprocess
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAGE = os.path.join(ROOT, 'index.html')
FILES = {name: os.path.join(ROOT, name) for name in ('tactical.js', 'possession.js')}
NODE = shutil.which('node')
FPS = 30
A, B = '#b98cff', '#f6b564'   # the board's default team colours

HARNESS = r"""
const fs = require('fs'), vm = require('vm');
const input = JSON.parse(fs.readFileSync(0, 'utf8'));
const round = (x) => typeof x === 'number' ? Math.round(x * 100) / 100 : typeof x === 'string' ? x : Array.isArray(x) ? x.map(round) : null;
const writes = {text: 0, left: 0};
// A 2D context that logs every call with the state it was made in (fill, stroke, alpha, dash).
function recorder(log) {
  const s = {fillStyle: '#000000', strokeStyle: '#000000', globalAlpha: 1, lineWidth: 1, dash: []};
  return new Proxy(s, {
    get: (o, k) => k in o ? o[k] : k === 'measureText' ? (t) => ({width: String(t).length * 6}) : k === 'createPattern' ? () => 'PATTERN'
      : k === 'setLineDash' ? (d) => { o.dash = d.slice(); } : k === 'getLineDash' ? () => o.dash.slice()
      : (...a) => { log.push([k, ...a.map(round), {fill: o.fillStyle, stroke: o.strokeStyle, alpha: round(o.globalAlpha), dash: round(o.dash), width: round(o.lineWidth)}]); },
    set: (o, k, v) => { o[k] = v; return true; }});
}
function styles() {
  return new Proxy({}, {set: (o, k, v) => { if (k === 'left') writes.left++; o[k] = v; return true; }});
}
class Node {
  constructor(tag) {
    this.tagName = String(tag).toUpperCase(); this.children = []; this.parentElement = null; this.attrs = {}; this.dataset = {}; this.style = styles();
    this.listeners = {}; this.hidden = false; this.disabled = false; this.checked = false; this.value = ''; this.own = ''; this.cls = ''; this.log = null; this.title = '';
    this.clientWidth = 600;
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
  set textContent(v) { writes.text++; this.own = String(v); this.clear(); }
  clear() { for (const c of this.children) c.parentElement = null; this.children = []; }
  append(...cs) { for (const c of cs) { if (c.parentElement) c.parentElement.removeChild(c); c.parentElement = this; this.children.push(c); } }
  removeChild(c) { const i = this.children.indexOf(c); if (i >= 0) this.children.splice(i, 1); c.parentElement = null; }
  replaceChildren(...cs) { this.own = ''; this.clear(); this.append(...cs); }
  setAttribute(k, v) { this.attrs[k] = String(v); }
  getAttribute(k) { return k in this.attrs ? this.attrs[k] : null; }
  addEventListener(t, f) { (this.listeners[t] = this.listeners[t] || []).push(f); }
  fire(t, extra) { const e = Object.assign({type: t, target: this, preventDefault() {}, stopPropagation() {}}, extra || {}); for (const f of this.listeners[t] || []) f(e); }
  click() { this.fire('click'); }
  getContext() { return recorder(this.log = this.log || []); }
  getBoundingClientRect() { return {left: 100, top: 0, width: 600, height: 16}; }
}
const sandbox = {document: {createElement: (t) => new Node(t), body: new Node('body'),
                            createTextNode: (t) => ({nodeType: 3, textContent: String(t), parentElement: null})},
                 console, setTimeout: (f) => setImmediate(f), clearTimeout: () => {}, Map, Set, JSON, Promise,
                 localStorage: {getItem: () => null, setItem() {}}, devicePixelRatio: 1, addEventListener() {}};
sandbox.window = sandbox;
vm.createContext(sandbox);
for (const path of Object.values(input.files)) vm.runInContext(fs.readFileSync(path, 'utf8'), sandbox, {filename: path});
const settle = async () => { for (let k = 0; k < 20; k++) await new Promise((r) => setImmediate(r)); };
const copy = (x) => JSON.parse(JSON.stringify(x));

function setup(job) {
  const el = (tag) => new Node(tag || 'div');
  const s = {panel: el(), host: el('section'), top: el(), live: el('span'), share: el('span'), track: el(), strip: el('canvas'), marks: el(), now: el(),
             foot: el(), axis: [el('span'), el('span')], legend: el('span'), message: el('p'), inputs: [el('input'), el('input')],
             note: el(), key: el('span'), toggle: el('input'), count: el('span'),
             asked: [], seeks: [], pending: [], answers: (job.answers || []).slice(), invalidated: 0};
  s.toggle.checked = false;   // "Show uncertain changes" as the markup leaves it: the view ticks it when it is made
  s.board = job.realBoard ? sandbox.Tactical.create({canvas: new Node('canvas'), controls: {}}) : {colours: {A: input.A, B: input.B}, overlays: [], visible: () => true};
  const invalidate = s.board.invalidate ? s.board.invalidate.bind(s.board) : () => {};
  s.board.invalidate = () => { s.invalidated++; invalidate(); };
  const api = job.manual ? (url) => { s.asked.push(url); return new Promise((res, rej) => s.pending.push({url, res, rej})); }
    : async (url) => { s.asked.push(url); const a = s.answers.length > 1 ? s.answers.shift() : s.answers[0]; if (a && a.__error) throw new Error(a.__error); return copy(a); };
  s.view = sandbox.Possession.create({panel: s.panel, host: s.host, rows: [s.top, s.track, s.foot, s.note], live: s.live, share: s.share, track: s.track, strip: s.strip,
                                      marks: s.marks, now: s.now, axis: s.axis, legend: s.legend, key: s.key, uncertain: s.toggle, uncertainCount: s.count,
                                      message: s.message, api, run: () => job.run === undefined ? 'run1' : job.run,
                                      board: s.board, colourInputs: s.inputs, seek: (t, pause) => s.seeks.push([t, pause])});
  return s;
}
const strip = (s, from = 0) => (s.strip.log || []).slice(from).filter((c) => c[0] === 'fillRect').map((c) => ({x: c[1], w: c[3], fill: c[5].fill, alpha: c[5].alpha}));
const panel = (s) => ({hidden: s.panel.hidden, host: s.host.className, rows: [s.top, s.track, s.foot, s.note].map((r) => r.hidden),
                       message: {hidden: s.message.hidden, text: s.message.textContent, title: s.message.title}, share: {text: s.share.textContent, title: s.share.title},
                       legend: s.legend.textContent, legendColours: s.legend.children.filter((c) => c.children && c.children[0]).map((c) => c.children[0].style.background),
                       key: {text: s.key.textContent, title: s.key.title, marks: s.key.children.filter((c) => c.tagName === 'I').map((c) => c.className)},
                       toggle: {checked: s.toggle.checked, disabled: s.toggle.disabled, count: s.count.textContent, countTitle: s.count.title},
                       axis: s.axis.map((a) => a.textContent), marksLabel: s.marks.getAttribute('aria-label'),
                       marks: s.marks.children.map((b) => ({left: b.style.left, title: b.title, aria: b.attrs['aria-label'], cls: b.className, colour: b.children[0].style.background,
                                                            change: b.dataset.change, type: b.type, width: b.style.width, margin: b.style.marginLeft, icon: b.children[0].style.left})),
                       asked: s.asked.slice(), invalidated: s.invalidated});
const live = (s, t) => { s.view.tick(t); return {t, text: s.live.children[1].textContent, title: s.live.title, dot: s.live.children[0].style.background, now: {hidden: s.now.hidden, left: s.now.style.left}}; };
const ringsAt = (s, t, players) => {
  const log = [], ctx = recorder(log);
  s.view.drawBoard(ctx, {toPx: (x, y) => [x * 1000, y * 500], playerRadius: 10}, {time: t, players});
  return log.filter((c) => c[0] === 'arc').map((c) => [c[1], c[2], c[3], c[6].stroke, c[6].dash.length > 0]);
};

async function scenario(job) {
  const s = setup(job), out = {};
  s.view.load(job.result, job.info);
  if (job.kind === 'late') {
    s.view.load(job.result, job.info);
    await settle();
    s.pending[1].res(copy(job.record)); await settle();
    s.pending[0].res(copy(job.unavailable)); await settle();
    out.after = panel(s);
    return out;
  }
  out.loading = panel(s);
  await settle();
  out.first = panel(s);
  if (job.kind === 'render') {
    out.live = (job.times || []).map((t) => live(s, t));
    out.strip = strip(s);
  } else if (job.kind === 'seek') {
    for (const b of s.marks.children) b.click();
    s.strip.fire('click', {clientX: 100 + 300});
    s.strip.fire('click', {clientX: 100 + 600});
    s.strip.fire('mousemove', {clientX: 100 + 60});
    out.seeks = s.seeks; out.hover = s.strip.title;
  } else if (job.kind === 'colours') {
    const before = (s.strip.log || []).length;
    s.board.colours.A = '#ff0000';
    s.inputs[0].fire('input');
    out.strip = strip(s, before); out.after = panel(s); out.live = live(s, 10.5);
  } else if (job.kind === 'reload') {
    s.view.load(job.corrected, job.info); await settle();
    out.after = panel(s); out.live = live(s, 15.0);
    s.view.load(null, job.info); await settle();
    out.cleared = panel(s);
    s.view.tick(15.0);
    out.clearedTick = s.live.children[1].textContent;
  } else if (job.kind === 'quiet') {
    live(s, 10.5);
    const w0 = {...writes};
    s.view.tick(10.5);
    const w1 = {...writes};
    s.view.tick(10.6);
    const w2 = {...writes};
    s.view.tick(13.5);
    const w3 = {...writes};
    out.writes = {same: w1.text - w0.text + w1.left - w0.left, moved: {text: w2.text - w1.text, left: w2.left - w1.left}, changed: {text: w3.text - w2.text, left: w3.left - w2.left}};
    out.asked = s.asked.length; out.stripCalls = (s.strip.log || []).length;
    for (let k = 0; k < 50; k++) s.view.tick(10 + k * .1);
    out.askedAfter = s.asked.length; out.stripCallsAfter = (s.strip.log || []).length;
  } else if (job.kind === 'rings') {
    const log = [], ctx = recorder(log), g = {toPx: (x, y) => [x * 1000, y * 500], playerRadius: 10};
    const players = job.players;
    for (const t of job.times) { log.length = 0; s.view.drawBoard(ctx, g, {time: t, players}); out['board' + t] = log.filter((c) => c[0] === 'arc').map((c) => [c[1], c[2], c[3], c[6].stroke, c[6].dash.length > 0]); }
    const frame = job.result.frames.find((f) => Math.abs(f.time - 10.5) < 1e-6);
    for (const t of job.times) { log.length = 0; s.view.drawVideo(ctx, frame, t, 1920, 1080, {A: '#b98cff', B: '#f6b564'}); out['video' + t] = log.filter((c) => c[0] === 'ellipse').map((c) => [c[1], c[2], c[3], c[4], c[8].stroke, c[8].dash.length > 0]); }
    s.board.visible = () => false; log.length = 0; s.view.drawBoard(ctx, g, {time: 10.5, players});
    out.hiddenMarker = log.filter((c) => c[0] === 'arc').length;
  } else if (job.kind === 'toggle') {
    // "Show uncertain changes": ticked when the view is made, cleared, ticked again; the line and the ring on the
    // uncertain change's won frame.
    const t = job.at;
    out.on = {line: live(s, t), rings: ringsAt(s, t, job.players)};
    let before = s.invalidated;
    s.toggle.checked = false; s.toggle.fire('change');
    out.off = {panel: panel(s), line: s.live.children[1].textContent, rings: ringsAt(s, t, job.players), redraws: s.invalidated - before};
    out.off.lineAfterTick = live(s, t).text;
    before = s.invalidated;
    s.toggle.checked = true; s.toggle.fire('change');
    out.onAgain = {panel: panel(s), line: s.live.children[1].textContent, rings: ringsAt(s, t, job.players), redraws: s.invalidated - before};
  } else if (job.kind === 'board') {
    // The real board: the ring is one of its overlays, drawn under the carrier's marker.
    s.board.load(job.result, job.info);
    const canvas = s.board.canvas, g = s.board.geometry();
    for (const t of job.times) {
      canvas.log.length = 0;   // the board's context logs into this array (made when the board was created)
      s.board.invalidate();
      const state = s.board.render(t);
      const log = canvas.log, arcs = log.map((c, i) => [i, c]).filter(([, c]) => c[0] === 'arc');
      const ring = arcs.filter(([, c]) => Math.abs(c[3] - round(g.playerRadius * 1.85)) < .02);
      const carrier = state.players.find((p) => p.id === job.carrier[t]);
      const at = carrier ? g.toPx(carrier.x, carrier.y).map(round) : null;
      const marker = carrier ? arcs.filter(([, c]) => Math.abs(c[3] - round(g.playerRadius)) < .02 && Math.abs(c[1] - at[0]) < .02 && Math.abs(c[2] - at[1]) < .02) : [];
      out['t' + t] = {rings: ring.map(([i, c]) => ({i, x: c[1], y: c[2], stroke: c[6].stroke, dashed: c[6].dash.length > 0})), at, marker: marker.map(([i]) => i)};
    }
  }
  return out;
}

(async () => {
  const out = [];
  for (const job of input.jobs) out.push(await scenario(job));
  process.stdout.write(JSON.stringify(out));
})().catch((error) => { process.stderr.write(String(error && error.stack || error)); process.exit(1); });
"""


def frames(first=300, last=599):
    """The fixture result's frames: absolute source frames, time = frame/fps, four players standing still with their
    boxes (the video ring is drawn at the bottom of the box) and pitch positions (the board draws them), and the ball."""
    people = [('A-07', 'A', [.40, .50], [.30, .40, .02, .08]), ('A-02', 'A', [.45, .30], [.40, .30, .02, .08]),
              ('B-07', 'B', [.55, .50], [.55, .40, .02, .08]), ('B-09', 'B', [.60, .70], [.60, .55, .02, .08])]
    return [{'time': round(f/FPS, 6), 'frame': f, 'calibrated': True,
             'people': [{'id': pid, 'team': team, 'role': 'PLAYER', 'state': 'confirmed', 'pitch': pitch, 'box': box, 'track': k+1,
                         'identityConfidence': .9, 'pitchConfidence': 1} for k, (pid, team, pitch, box) in enumerate(people)],
             'ball': {'state': 'TRACKED', 'pitch': [.41, .50], 'confidence': .9, 'pitchConfidence': 1}} for f in range(first, last+1)]


# Display labels differ from the ids (A-07 shown as A-03, B-07 as B-05, B-09 as B-01): the panel must show the labels.
RESULT = {'version': 3, 'frames': frames(),
          'match': {'players': [{'id': 'A-07', 'display': 'A-03', 'team': 'A'}, {'id': 'A-02', 'display': 'A-02', 'team': 'A'},
                                {'id': 'B-07', 'display': 'B-05', 'team': 'B'}, {'id': 'B-09', 'display': 'B-01', 'team': 'B'}],
                    'goalkeepers': [], 'referees': []},
          'roster': {'slots': []}}
INFO = {'video': {'fps': FPS, 'width': 1920, 'height': 1080}, 'videoKey': '0'*64}


def run_(t0, t1, team, state, carrier=None, estimated=False, confidence='high'):
    return {'t0': t0, 't1': t1, 'f0': round(t0*FPS), 'f1': round(t1*FPS), 'team': team, 'state': state, 'carrier': carrier,
            'estimated': estimated, 'confidence': confidence}


LAST = round(599/FPS, 6)   # 19.966667
RECORD = {
    'version': 'fixture-1', 'fps': FPS, 't0': 10.0, 't1': LAST, 'available': True, 'reason': None,
    'quality': {'level': 'limited', 'coverage': .75, 'notes': ['The ball is hidden on 12% of the frames']},
    'shares': {'A': 5.5, 'B': 2.0, 'dead': 1.0, 'unknown': 1.5},
    'timeline': [run_(10.0, 11.966667, 'A', 'control', 'A-07'), run_(12.0, 12.966667, 'A', 'flight'),
                 run_(13.0, 13.966667, 'A', 'control', 'A-02', estimated=True, confidence='medium'),
                 run_(14.0, 14.466667, None, 'contested', confidence='low'), run_(14.5, 15.966667, 'B', 'control', 'B-07', confidence='medium'),
                 run_(16.0, 16.466667, 'B', 'loose'), run_(16.5, 17.466667, None, 'dead'), run_(17.5, 18.466667, None, 'unknown', estimated=True, confidence='low'),
                 run_(18.5, LAST, 'A', 'control', 'A-07', confidence='low')],
    'changes': [{'id': 'chg-0001', 'lostAt': 13.95, 'wonAt': 14.5, 'frame': 435, 'fromTeam': 'A', 'toTeam': 'B', 'lostBy': 'A-02', 'wonBy': 'B-07',
                 'cause': 'tackle_or_duel', 'confidence': 'medium'},
                {'id': 'chg-0002', 'lostAt': 16.4, 'wonAt': 18.5, 'frame': 555, 'fromTeam': 'B', 'toTeam': 'A', 'lostBy': None, 'wonBy': 'A-07',
                 'cause': 'out_of_play', 'confidence': 'low'},
                {'id': 'chg-0003', 'lostAt': 19.5, 'wonAt': 19.5, 'frame': 585, 'fromTeam': 'A', 'toTeam': 'B', 'lostBy': None, 'wonBy': None,
                 'cause': 'mystery', 'confidence': 'high'}],
    'deadSpells': [{'t0': 16.5, 't1': 17.466667, 'reason': 'out_of_play', 'restartTeam': 'A'}],
}
UNAVAILABLE = {'version': 'fixture-1', 'fps': FPS, 't0': 10.0, 't1': LAST, 'available': False,
               'reason': 'The ball is placed on too few frames of this run (12%) to judge who has it',
               'quality': {'level': 'unavailable', 'coverage': 0.0, 'notes': []}, 'shares': {'A': 0, 'B': 0, 'dead': 0, 'unknown': 10.0},
               'timeline': [], 'changes': [], 'deadSpells': []}
# chg-0001's won frame (14.5 s, frame 435) and the two after it in an unknown run, as the engine leaves most won frames.
WON = json.loads(json.dumps(RECORD))
WON['timeline'][4:5] = [run_(14.5, 14.566667, None, 'unknown', estimated=True, confidence='low'), run_(14.6, 15.966667, 'B', 'control', 'B-07', confidence='medium')]
ON_WON, NEXT_FRAME = 14.5+.2/FPS, 14.5+1.2/FPS
# Two changes 0.1 s apart (6 px on the 600 px strip of the harness), as the broadcast's back-and-forth pairs.
CLOSE = json.loads(json.dumps(RECORD))
CLOSE['changes'][1] = {'id': 'chg-close', 'lostAt': 14.55, 'wonAt': 14.6, 'frame': 438, 'fromTeam': 'B', 'toTeam': 'A', 'lostBy': 'B-07',
                       'wonBy': 'A-02', 'cause': 'interception', 'confidence': 'medium'}
SPAN = LAST + 1/FPS - 10.0   # the strip: the analysed segment, its last frame one frame wide
# RECORD as possession-4 writes it: a change is 'high' or 'uncertain' (possession-3's 'medium' and 'low' merged);
# timeline runs keep high / medium / low.
V4 = json.loads(json.dumps(RECORD))
V4['version'] = 'fixture-4'
for _change, _level in zip(V4['changes'], ('uncertain', 'uncertain', 'high')):
    _change['confidence'] = _level
# The uncertain change chg-0002 (won at 18.5 s, frame 555) with its won frame and the next in an unknown run.
LOW_WON = json.loads(json.dumps(V4))
LOW_WON['timeline'][8:9] = [run_(18.5, 18.533333, None, 'unknown', estimated=True, confidence='low'), run_(18.566667, LAST, 'A', 'control', 'A-07', confidence='low')]
ON_LOW = 18.5+.2/FPS


def corrected():
    """RESULT after two corrections, as corrections.apply_result leaves it: B-07 is shown as B-04 now, and A-07 was
    marked as not part of the match (its entry moves to match.excluded with the label it had)."""
    r = json.loads(json.dumps(RESULT))
    players = r['match']['players']
    players[2]['display'] = 'B-04'
    gone = players.pop(0)
    r['match']['excluded'] = [dict(gone, excluded=True)]
    return r


def run(*jobs):
    done = subprocess.run([NODE, '-e', HARNESS], input=json.dumps({'files': FILES, 'jobs': list(jobs), 'A': A, 'B': B}), capture_output=True,
                          text=True, encoding='utf-8', timeout=300)
    if done.returncode:
        raise AssertionError(done.stderr[-3000:])
    return json.loads(done.stdout)


def job(kind='render', answers=None, **kw):
    return dict({'kind': kind, 'answers': answers or [RECORD], 'result': RESULT, 'info': INFO}, **kw)


def pct(t):
    return f'{max(0, min(100, 100*(t-10.0)/SPAN)):.2f}%'


@unittest.skipUnless(NODE, 'needs node to run the page scripts')
class Panel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        times = [10.0, 10.5, 11.98, 12.5, 13.5, 14.2, 15.0, 16.2, 17.0, 18.0, 19.0, LAST, 9.0, 21.0]
        players = [{'id': 'A-07', 'team': 'A', 'x': .4, 'y': .5, 'alpha': 1}, {'id': 'A-02', 'team': 'A', 'x': .45, 'y': .3, 'alpha': .6},
                   {'id': 'B-07', 'team': 'B', 'x': .55, 'y': .5, 'alpha': 1}]
        # Every job starts with "Show uncertain changes" as the view leaves it (ticked), so every change is shown.
        cls.out = run(job(times=times), job('seek'), job('colours'), job('reload', corrected=corrected()),
                      job('quiet'), job(answers=[UNAVAILABLE]),
                      job(answers=[{'__error': 'Possession tracking is not available in this version: No module named soccer.possession'}]),
                      job('late', manual=True, record=RECORD, unavailable=UNAVAILABLE), job(run=None),
                      job('rings', times=[10.5, 12.5, 13.5, 15.2, 17.2], players=players),
                      job('board', realBoard=True, times=[10.5, 12.5, 15.2], carrier={'10.5': 'A-07', '12.5': None, '15.2': 'B-07'}),
                      job(answers=[WON], times=[14.5-.9/FPS, ON_WON, 14.5+.97/FPS, NEXT_FRAME, 15.0, 18.5+.2/FPS, 19.5+.2/FPS]),
                      job('rings', answers=[WON], times=[ON_WON, NEXT_FRAME, 15.2], players=players),
                      job(answers=[CLOSE]), job(answers=[V4], times=[14.5+.2/FPS, 18.5+.2/FPS, 19.5+.2/FPS]),
                      job('toggle', answers=[LOW_WON], at=ON_LOW, players=players))
        (cls.render, cls.seek, cls.colours, cls.reload, cls.quiet, cls.unavailable, cls.error, cls.late, cls.norun, cls.rings, cls.board,
         cls.won, cls.wonRings, cls.close, cls.v4, cls.toggle) = cls.out

    def test_loading_then_the_full_panel(self):
        loading, first = self.render['loading'], self.render['first']
        self.assertEqual((loading['hidden'], loading['host'], loading['rows']), (False, 'pz-line', [True, True, True, True]))
        self.assertIn('judging who has the ball', loading['message']['text'])
        self.assertEqual(loading['message']['title'], loading['message']['text'], 'the whole message is its tooltip (the page holds it to two lines)')
        self.assertEqual((first['hidden'], first['host'], first['rows'], first['message']['hidden'], first['message']['title']),
                         (False, 'pz-on', [False, False, False, False], True, ''))
        self.assertEqual(first['asked'], ['/api/possession?run=run1'])
        self.assertEqual(self.norun['first']['asked'], ['/api/possession'], 'without a run name: the shown run')
        self.assertGreaterEqual(first['invalidated'], 2, 'the board redraws when the record arrives (the ring)')
        self.assertEqual(first['axis'], ['00:10.00', '00:19.97'])

    def test_the_live_line(self):
        lines = {round(x['t'], 4): x['text'] for x in self.render['live']}
        self.assertEqual(lines[10.0], 'Team A in possession \u00b7 A-03 on the ball', 'the display label, not the id A-07')
        self.assertEqual(lines[10.5], 'Team A in possession \u00b7 A-03 on the ball')
        self.assertEqual(lines[11.98], 'Team A in possession \u00b7 A-03 on the ball', 'between the last frame of a run and the next run')
        self.assertEqual(lines[12.5], 'Team A in possession \u00b7 ball in flight', "the team's possession persists through its pass")
        self.assertEqual(lines[13.5], 'Team A in possession \u00b7 A-02 on the ball \u00b7 estimated')
        self.assertEqual(lines[14.2], 'Contested ball \u00b7 low confidence')
        self.assertEqual(lines[15.0], 'Team B in possession \u00b7 B-05 on the ball')
        self.assertEqual(lines[16.2], 'Team B in possession \u00b7 loose ball')
        self.assertEqual(lines[17.0], 'Ball out of play \u00b7 Team A to restart')
        self.assertEqual(lines[18.0], 'Possession unknown', 'an estimated, low-confidence unknown run is said plainly')
        self.assertEqual(lines[19.0], 'Team A in possession \u00b7 A-03 on the ball \u00b7 low confidence')
        self.assertEqual(lines[round(LAST, 4)], 'Team A in possession \u00b7 A-03 on the ball \u00b7 low confidence')
        self.assertEqual((lines[9.0], lines[21.0]), ('Outside the analysed segment', 'Outside the analysed segment'))
        self.assertTrue(all(x['title'] == x['text'] for x in self.render['live']), 'the whole line is the tooltip of the live chip (a narrow panel cuts it)')
        dots = {round(x['t'], 4): x['dot'] for x in self.render['live']}
        self.assertEqual((dots[10.5], dots[15.0], dots[17.0], dots[18.0]), (A, B, '#050807', '#4a5650'), 'the team colour, dark for dead ball, grey for unknown')
        now = {round(x['t'], 4): x['now'] for x in self.render['live']}
        self.assertEqual(now[15.0], {'hidden': False, 'left': pct(15.0)})
        self.assertTrue(now[9.0]['hidden'] and now[21.0]['hidden'], 'no playhead outside the analysed segment')

    def test_shares_and_coverage(self):
        share = self.render['first']['share']
        self.assertEqual(share['text'], 'PossessionTeam A 73%Team B 27%coverage 75%limited', 'of the named time: 5.5 s and 2.0 s')
        for text in ('Team A had the ball 5.5 s, Team B 2.0 s', 'ball out of play 1.0 s', 'unknown 1.5 s', 'A team is named on 75% of the frames',
                     'Limited quality.', 'The ball is hidden on 12% of the frames.', '(00:10.00\u201300:19.97)'):
            self.assertIn(text, share['title'])

    def test_the_strip(self):
        rects = self.render['strip']
        w = 600

        def at(t):
            # The rectangles drawn over time t, last on top.
            x = (t-10.0)/SPAN*w
            return [(r['fill'], r['alpha']) for r in rects if r['x'] <= x < r['x']+r['w']]
        self.assertEqual(rects[0], {'x': 0, 'w': w, 'fill': '#4a5650', 'alpha': 1}, 'nothing judged is unknown')
        self.assertEqual(at(10.5)[-1], (A, 1), 'control: solid team colour')
        self.assertEqual(at(12.5)[-1], (A, .58), 'the team in possession, ball in flight: fainter')
        self.assertEqual(at(13.5)[-2:], [(A, 1), ('PATTERN', 1)], 'estimated: hatched over the team colour')
        self.assertEqual(at(14.2)[-1], ('#8d99a6', .45), 'contested with no team: neutral')
        self.assertEqual(at(15.0)[-1], (B, 1))
        self.assertEqual(at(16.2)[-1], (B, .58))
        self.assertEqual(at(17.0)[-1], ('#050807', 1), 'dead ball: dark')
        self.assertEqual(at(18.0)[-1], ('#4a5650', 1), 'unknown: grey')
        self.assertAlmostEqual(sum(r['w'] for r in rects[1:] if r['fill'] in (A, B) and r['alpha'] == 1),
                               w*(2.0+1.0+1.5+1.5)/SPAN, delta=.2, msg='every control run, estimates included, solid over its frames')
        legend = self.render['first']
        self.assertEqual(legend['legend'], 'Team ATeam Bestimatedunknownout of play', "the strip's colours (the markers have their own key)")
        self.assertEqual(legend['legendColours'][:2], [A, B])

    def test_change_markers(self):
        marks = self.render['first']['marks']
        self.assertEqual([m['change'] for m in marks], ['chg-0001', 'chg-0002', 'chg-0003'], 'every change is shown by default')
        self.assertEqual([m['left'] for m in marks], [pct(14.5), pct(18.5), pct(19.5)])
        self.assertEqual([m['colour'] for m in marks], [B, A, B], "the winning team's colour")
        self.assertEqual([m['cls'] for m in marks], ['pz-mark', 'pz-mark uncertain', 'pz-mark'],
                         "a possession-3 record: medium and high solid, the low one hollow")
        self.assertEqual(marks[0]['title'], 'Team B won the ball \u00b7 A-02 lost it, B-05 won it \u00b7 lost 00:13.95, won 00:14.50 \u00b7 tackle or duel \u00b7 medium confidence')
        self.assertEqual(marks[1]['title'], 'Team A won the ball \u00b7 A-03 won it \u00b7 lost 00:16.40, won 00:18.50 \u00b7 ball out of play \u00b7 uncertain (low confidence)')
        self.assertEqual(marks[2]['title'], 'Team B won the ball \u00b7 00:19.50 \u00b7 cause unclear \u00b7 high confidence', 'an unknown cause is unclear')
        self.assertTrue(all(m['aria'] == m['title'] and m['type'] == 'button' for m in marks))
        self.assertEqual(self.render['first']['marksLabel'], 'Changes of possession: 3 shown')

    def test_a_possession_4_record(self):
        # possession-4 rates a change 'high' or 'uncertain': an uncertain one is hollow and says so in its tooltip and
        # on its won frame's line; a high one is solid.
        first = self.v4['first']
        self.assertEqual([m['change'] for m in first['marks']], ['chg-0001', 'chg-0002', 'chg-0003'])
        self.assertEqual([m['cls'] for m in first['marks']], ['pz-mark uncertain', 'pz-mark uncertain', 'pz-mark'])
        self.assertEqual(first['marks'][0]['title'], 'Team B won the ball \u00b7 A-02 lost it, B-05 won it \u00b7 lost 00:13.95, won 00:14.50 \u00b7 tackle or duel \u00b7 uncertain')
        self.assertTrue(first['marks'][1]['title'].endswith(' \u00b7 ball out of play \u00b7 uncertain'), first['marks'][1]['title'])
        self.assertTrue(first['marks'][2]['title'].endswith(' \u00b7 high confidence'))
        self.assertEqual(first['toggle'], {'checked': True, 'disabled': False, 'count': '(2)', 'countTitle': '2 uncertain changes of 3'})
        self.assertEqual(first['marksLabel'], 'Changes of possession: 3 shown')
        lines = [x['text'] for x in self.v4['live']]
        self.assertEqual(lines, ['Team B won the ball \u00b7 B-05 won it \u00b7 uncertain', 'Team A won the ball \u00b7 A-03 won it \u00b7 uncertain', 'Team B won the ball'],
                         'the won frames: uncertain said, high not')

    def test_the_key_and_the_option(self):
        first = self.render['first']
        self.assertEqual(first['key']['text'], 'Changes of possession are estimates: they can be wrong or a second or two off.uncertain', 'one line')
        self.assertEqual(first['key']['marks'], ['pz-tri', 'pz-tri uncertain'], 'a solid and a hollow triangle, drawn like the markers')
        self.assertIn('not a counted turnover', first['key']['title'])
        self.assertIn('A hollow triangle is an uncertain change; untick \u201cShow uncertain changes\u201d to hide the uncertain ones.', first['key']['title'])
        self.assertEqual(first['toggle'], {'checked': True, 'disabled': False, 'count': '(1)', 'countTitle': '1 uncertain change of 3'},
                         'ticked when the view is made (the markup leaves it unticked)')
        self.assertEqual(self.close['first']['toggle'], {'checked': True, 'disabled': True, 'count': '(0)', 'countTitle': 'No uncertain change in this segment'},
                         'no uncertain change: the option is disabled')
        self.assertEqual(self.close['first']['marksLabel'], 'Changes of possession: 3 shown')

    def test_uncertain_changes_are_shown_until_unticked(self):
        t = self.toggle
        self.assertEqual(t['first']['toggle']['checked'], True)
        self.assertEqual([m['cls'] for m in t['first']['marks']], ['pz-mark uncertain', 'pz-mark uncertain', 'pz-mark'])
        self.assertEqual(t['on']['line']['text'], 'Team A won the ball \u00b7 A-03 won it \u00b7 uncertain', 'shown: its won frame says so')
        self.assertEqual(t['on']['rings'], [[400, 250, 18.5, '#07131bb3', False], [400, 250, 18.5, A, True]], 'the winner ringed, dashed')
        off = t['off']
        self.assertEqual([m['change'] for m in off['panel']['marks']], ['chg-0003'], 'unticked: only the high change')
        self.assertEqual(off['panel']['marksLabel'], 'Changes of possession: 1 shown, 2 uncertain hidden')
        self.assertEqual(off['line'], 'Possession unknown', 'hidden: no "won the ball" line on its won frame, at once')
        self.assertEqual(off['lineAfterTick'], off['line'])
        self.assertEqual(off['rings'], [], 'and no ring around its winner')
        self.assertGreaterEqual(off['redraws'], 1, 'the board draws again (the ring)')
        self.assertEqual((off['panel']['toggle']['checked'], off['panel']['toggle']['count']), (False, '(2)'))
        again = t['onAgain']
        self.assertEqual([m['change'] for m in again['panel']['marks']], ['chg-0001', 'chg-0002', 'chg-0003'], 'ticked again: every change')
        self.assertEqual((again['line'], again['rings']), (t['on']['line']['text'], t['on']['rings']))
        self.assertGreaterEqual(again['redraws'], 1)

    def test_seeking(self):
        seeks = self.seek['seeks']
        for (t, pause), frame in zip(seeks[:3], (435, 555, 585)):
            self.assertAlmostEqual(t, round(frame/FPS, 6)+.2/FPS, places=9, msg='0.2 frame into the frame the ball was won')
            self.assertTrue(pause, 'a marker pauses')
        self.assertAlmostEqual(seeks[3][0], 10.0+.5*SPAN, places=6, msg='the strip seeks to the time clicked')
        self.assertFalse(seeks[3][1], 'and leaves playback as it was')
        self.assertFalse(seeks[4][1])
        self.assertLess(seeks[4][0], 10.0+SPAN, 'the right edge: inside the last frame')
        self.assertGreater(seeks[4][0], LAST)
        self.assertEqual(self.seek['hover'], '00:11.00 \u00b7 Team A in possession \u00b7 A-03 on the ball')

    def test_the_frame_the_ball_was_won(self):
        lines = {round(x['t'], 4): (x['text'], x['dot']) for x in self.won['live']}
        self.assertEqual(lines[round(ON_WON, 4)], ('Team B won the ball \u00b7 B-05 won it', B),
                         'where a marker seeks: the change, though the run there is unknown')
        self.assertEqual(lines[round(14.5+.97/FPS, 4)][0], 'Team B won the ball \u00b7 B-05 won it', 'the whole won frame')
        self.assertEqual(lines[round(14.5-.9/FPS, 4)][0], 'Contested ball \u00b7 low confidence', 'the frame before is the run before')
        self.assertEqual(lines[round(NEXT_FRAME, 4)][0], 'Possession unknown', 'the next frame is the run again')
        self.assertEqual(lines[15.0][0], 'Team B in possession \u00b7 B-05 on the ball')
        self.assertEqual(lines[round(18.5+.2/FPS, 4)][0], 'Team A won the ball \u00b7 A-03 won it \u00b7 uncertain', 'a shown low-confidence change says so')
        self.assertEqual(lines[round(19.5+.2/FPS, 4)][0], 'Team B won the ball', 'no winner named: the team only')
        r = self.wonRings
        self.assertEqual(r['board' + repr(ON_WON)], [[550, 250, 18.5, '#07131bb3', False], [550, 250, 18.5, B, True]],
                         'the winner is ringed on the won frame, dashed (not named in control there)')
        self.assertEqual(r['video' + repr(ON_WON)][0][4:], [B, True])
        self.assertEqual((r['board' + repr(NEXT_FRAME)], r['video' + repr(NEXT_FRAME)]), ([], []), 'no carrier on the next frame: no ring')
        self.assertEqual(r['board15.2'][1], [550, 250, 18.5, B, False], 'the carrier again, solid')

    def test_close_markers_share_the_space_between_them(self):
        marks = self.close['first']['marks']
        self.assertEqual([m['change'] for m in marks], ['chg-0001', 'chg-close', 'chg-0003'])
        self.assertEqual([m['left'] for m in marks], [pct(14.5), pct(14.6), pct(19.5)], 'each on its own time')
        self.assertEqual([(m['margin'], m['width'], m['icon']) for m in marks],
                         [('-7.0px', '10.0px', '2.0px'), ('-3.0px', '10.0px', '-2.0px'), ('-7.0px', '14.0px', '2.0px')],
                         'the two buttons meet halfway (3 px each side of the 6 px gap), the triangles stay centred on their times')
        far = self.render['first']['marks']
        self.assertEqual({(m['margin'], m['width'], m['icon']) for m in far}, {('-7.0px', '14.0px', '2.0px')}, 'apart: the full 14 px')

    def test_board_colours_repaint_the_panel(self):
        out = self.colours
        self.assertIn({'x': 0, 'w': 600, 'fill': '#4a5650', 'alpha': 1}, out['strip'], 'the strip is drawn again')
        self.assertIn('#ff0000', [r['fill'] for r in out['strip']])
        self.assertNotIn(A, [r['fill'] for r in out['strip']])
        self.assertEqual([m['colour'] for m in out['after']['marks']], [B, '#ff0000', B])
        self.assertEqual(out['after']['legendColours'][:2], ['#ff0000', B])
        self.assertEqual(out['live']['dot'], '#ff0000')

    def test_labels_follow_a_correction_and_a_new_load_asks_again(self):
        after = self.reload['after']
        self.assertEqual(after['asked'], ['/api/possession?run=run1', '/api/possession?run=run1'])
        self.assertEqual(after['marks'][0]['title'], 'Team B won the ball \u00b7 A-02 lost it, B-04 won it \u00b7 lost 00:13.95, won 00:14.50 \u00b7 tackle or duel \u00b7 medium confidence')
        self.assertIn('excluded (was A-03) won it', after['marks'][1]['title'], 'an excluded identity never shows another player\u2019s label')
        self.assertEqual(self.reload['live']['text'], 'Team B in possession \u00b7 B-04 on the ball')
        cleared = self.reload['cleared']
        self.assertEqual((cleared['hidden'], cleared['host'], cleared['marks'], len(cleared['asked'])), (True, '', [], 2), 'no result: no panel, nothing asked')
        self.assertEqual(self.reload['clearedTick'], 'Team B in possession \u00b7 B-04 on the ball', 'a hidden panel is not updated')

    def test_a_frame_of_playback_writes_nothing_unless_something_moved(self):
        w = self.quiet['writes']
        self.assertEqual(w['same'], 0, 'the same time again: nothing')
        self.assertEqual(w['moved'], {'text': 0, 'left': 1}, 'the same run later: only the playhead')
        self.assertEqual(w['changed'], {'text': 1, 'left': 1}, 'another run: the line and the playhead')
        self.assertEqual((self.quiet['askedAfter'], self.quiet['stripCallsAfter']), (self.quiet['asked'], self.quiet['stripCalls']),
                         'playback never asks the server or redraws the strip')

    def test_unavailable_and_failed_say_why_in_one_line(self):
        u = self.unavailable['first']
        self.assertEqual((u['hidden'], u['host'], u['rows'], u['marks']), (False, 'pz-line', [True, True, True, True], []))
        self.assertEqual(u['message']['text'], 'Possession is not shown for this run. The ball is placed on too few frames of this run (12%) to judge who has it.')
        self.assertEqual(u['message']['title'], u['message']['text'])
        e = self.error['first']
        self.assertEqual((e['host'], e['message']['text']),
                         ('pz-line', 'Possession is not shown. Possession tracking is not available in this version: No module named soccer.possession.'))

    def test_a_late_answer_of_an_older_load_is_dropped(self):
        after = self.late['after']
        self.assertEqual((after['host'], len(after['marks']), after['message']['hidden']), ('pz-on', 3, True), 'the record (every change shown)')

    def test_rings(self):
        r = self.rings
        self.assertEqual(r['board10.5'], [[400, 250, 18.5, '#07131bb3', False], [400, 250, 18.5, A, False]], 'a dark edge, then the team colour, solid')
        self.assertEqual(r['board12.5'], [], 'no carrier while the ball is in flight')
        self.assertEqual(r['board13.5'][1], [450, 150, 18.5, A, True], 'an estimated carrier: dashed')
        self.assertEqual(r['board15.2'][1], [550, 250, 18.5, B, False])
        self.assertEqual(r['board17.2'], [])
        self.assertEqual(r['hiddenMarker'], 0, 'nothing where the board does not draw the marker')
        self.assertEqual(r['video10.5'], [[595.2, 518.4, 32.64, 9.79, A, False]],
                         "an ellipse at the carrier's feet, in the boxes' colours")
        self.assertEqual(r['video13.5'][0][4:], [A, True])
        self.assertEqual(r['video15.2'][0][:2], [1075.2, 518.4])
        self.assertEqual((r['video12.5'], r['video17.2']), ([], []))

    def test_the_ring_is_a_layer_of_the_real_board_under_the_markers(self):
        for t in ('10.5', '15.2'):
            out = self.board['t'+t]
            self.assertEqual(len(out['rings']), 2, out)
            self.assertEqual([[x['x'], x['y']] for x in out['rings']], [out['at']]*2, 'around the marker the board draws')
            self.assertEqual(out['rings'][1]['stroke'], A if t == '10.5' else B)
            self.assertTrue(out['marker'] and out['rings'][1]['i'] < out['marker'][0], 'drawn before (under) the marker')
        self.assertEqual(self.board['t12.5']['rings'], [])


def page():
    with open(PAGE, encoding='utf-8') as handle:
        return handle.read()


def page_script(text):
    return '\n'.join(re.findall(r'<script>(.*?)</script>', text, re.S))


def node(script):
    done = subprocess.run([NODE, '-e', script], capture_output=True, text=True, encoding='utf-8', timeout=120)
    if done.returncode:
        raise AssertionError(done.stderr[-3000:])
    return json.loads(done.stdout)


class PageWiring(unittest.TestCase):
    """index.html: the panel under the board, the module, its fence, and its calls from the page's load and render."""

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
            for script in (path, FILES['possession.js']):
                done = subprocess.run([NODE, '--check', script], capture_output=True, text=True)
                self.assertEqual(done.returncode, 0, done.stderr)

    def test_panel_markup_module_and_calls(self):
        p, src = self.page, self.src
        self.assertRegex(p, r'<script src="/coaching.js"></script>\s*<script src="/possession.js"></script>\s*<script>')
        self.assertRegex(p, r'<section class="view" data-view="board" id="viewBoard"[^>]*><div class="board media"><canvas id="tactical"[^>]*></canvas>'
                            r'<div class="notice" id="tacticalNotice">[^<]*</div></div><div class="pz" id="possession" role="group" aria-label="Possession" hidden>')
        for control in ('possession', 'viewBoard', 'pzTop', 'pzTrack', 'pzFoot', 'pzNote', 'pzLive', 'pzShare', 'pzStrip', 'pzMarks', 'pzNow', 'pzAxis0', 'pzAxis1',
                        'pzLegend', 'pzKey', 'pzUncertain', 'pzUncertainN', 'pzMsg'):
            self.assertIn(f'id="{control}"', p, control)
            self.assertIn(f"$('{control}')", src, control)
        self.assertIn('<div class="pz-row pz-note" id="pzNote"><span class="pz-key" id="pzKey"></span><label class="pz-opt"', p, 'the key and the option under the strip')
        self.assertIn('<input type="checkbox" id="pzUncertain">Show uncertain changes<span id="pzUncertainN"></span></label>', p,
                      'unticked in the markup; possession.js ticks it when the view is made')
        self.assertIn("key:$('pzKey'),uncertain:$('pzUncertain'),uncertainCount:$('pzUncertainN'),", src)
        self.assertIn('teamShape.load(null,info);coaching.load(null,info);possession.load(null,info);', src)
        self.assertIn('teamShape.load(data,info);coaching.load(data,info);possession.load(data,info);', src)
        self.assertIn('analytics.tick(video.currentTime);possession.tick(video.currentTime);playback.tick();trim.playhead(video.currentTime);drawCoachMark(i);}', src)
        self.assertIn('ctx.globalAlpha=1;possession.drawVideo(ctx,f,video.currentTime,canvas.width,canvas.height,COLORS);drawBall(i,notes);}', src,
                      'the video ring is drawn with the boxes, in their colours, under the ball')
        self.assertIn("($('possession')||board.parentElement).after(strip);", src, 'the roster line comes after the possession panel')
        self.assertIn('board:tactical,', src)
        for rule in ('.pz[hidden],.pz-row[hidden],.pz-track[hidden],.pz-msg[hidden],.pz-now[hidden]{display:none}', '#viewBoard.pz-on{--pz-h:110px}',
                     '#viewBoard.pz-line{--pz-h:46px}', '.stage:fullscreen #viewBoard,.stage.full #viewBoard{--pz-h:0px}', '.stage:fullscreen .pz,.stage.full .pz{display:none}',
                     '#viewBoard .board.media{width:min(100%,calc(max(var(--stage-h) - var(--pz-h),var(--stage-h) * .72) * var(--ar)))}',
                     'container:pz/inline-size;width:min(100%,max(720px,', '@container pz (max-width:859px){.pz-share .pz-k{display:none}}',
                     '-webkit-line-clamp:2;line-clamp:2;overflow:hidden;', '.pz-key{flex:1 1 auto;min-width:0;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}'):
            self.assertIn(rule, p, rule)
        self.assertNotRegex(p, r'innerHTML')
        with open(FILES['possession.js'], encoding='utf-8') as handle:
            self.assertNotIn('innerHTML', handle.read(), 'labels and reasons are never parsed as HTML')

    @unittest.skipUnless(NODE, 'needs node to run the page script')
    def test_a_failing_possession_script_never_stops_the_page(self):
        line = next(l for l in self.src.split('\n') if l.startswith('const possession=(()=>{try{return Possession.create('))
        script = r"""
const out = [];
for (const kind of ['missing', 'throws', 'works']) {
  const els = {}, $ = (id) => els[id] || (els[id] = {id});
  const api = () => null, tactical = {}, resultRun = 'run1', COLORS = {};
  let mode = null; const calls = [];
  const video = {pause: () => calls.push('pause')}, playback = {seek: (t) => calls.push('seek ' + t)};
  const console = {error: () => {}};
  let options = null;
  const Possession = kind === 'missing' ? undefined : {create: (o) => { if (kind === 'throws') throw new Error('boom'); options = o; return {load() {}, tick() {}, drawVideo() {}, real: true}; }};
  LINE
  possession.load(null, {}); possession.tick(1); possession.drawVideo({}, {}, 1, 10, 10, COLORS);
  const entry = {kind, real: !!possession.real};
  if (options) {
    options.seek(12.5, true); options.seek(3, false); mode = 'boundary'; options.seek(7, true);
    entry.calls = calls; entry.run = options.run(); entry.board = options.board === tactical; entry.rows = options.rows.map((r) => r.id);
    entry.host = options.host.id; entry.panel = options.panel.id; entry.inputs = options.colourInputs.map((r) => r.id);
    entry.key = [options.key, options.uncertain, options.uncertainCount].map((r) => r.id);
  }
  out.push(entry);
}
process.stdout.write(JSON.stringify(out));
""".replace('LINE', line)
        out = node(script)
        self.assertEqual([o['real'] for o in out], [False, False, True], 'missing or throwing: a stand-in that does nothing')
        works = out[2]
        self.assertEqual(works['calls'], ['pause', 'seek 12.5', 'seek 3'], 'a marker pauses and seeks; the strip seeks; nothing while drawing')
        self.assertEqual((works['run'], works['board'], works['host'], works['panel']), ('run1', True, 'viewBoard', 'possession'))
        self.assertEqual(works['rows'], ['pzTop', 'pzTrack', 'pzFoot', 'pzNote'])
        self.assertEqual(works['inputs'], ['teamAColour', 'teamBColour'])
        self.assertEqual(works['key'], ['pzKey', 'pzUncertain', 'pzUncertainN'])

    def test_the_uncertain_triangle_is_hollow(self):
        # The clip path traces the triangle, then a smaller one inside it the other way round: under the nonzero rule
        # the inner one is a hole, so what is left is an outline in the team's colour.
        found = re.search(r'\.pz-mark\.uncertain i,\.pz-tri\.uncertain\{clip-path:polygon\(([^)]*)\)\}', self.page)
        self.assertTrue(found, 'the hollow clip path of the uncertain marker and key')
        points = [tuple(float(v.rstrip('%') or 0) for v in p.split()) for p in found.group(1).split(',')]
        self.assertEqual(points[0], points[3], 'the outer loop closes')
        self.assertEqual(points[4], points[7], 'the inner loop closes')

        def area(pts):
            return sum(x0*y1-x1*y0 for (x0, y0), (x1, y1) in zip(pts, pts[1:]+pts[:1]))/2
        outer, inner = points[:3], points[4:7]
        self.assertLess(area(outer)*area(inner), 0, 'the inner loop runs the other way')
        self.assertGreater(abs(area(inner)), .1*abs(area(outer)), 'a real hole')
        self.assertIn('.pz-mark i{position:absolute;top:1px;left:2px;display:block;width:10px;height:9px;clip-path:polygon(0 0,100% 0,50% 100%)', self.page,
                      'high and medium: the solid triangle')

    def test_the_reserved_heights_add_up(self):
        # The page reserves the panel's height under the board (#viewBoard.pz-on / .pz-line); if a row grows the
        # transport moves down by as much (measured in the browser on the broadcast run 42-222 s: 110 px from 1024 x 768
        # to 1920 x 1080). This adds the rows up from the CSS: the panel's top margin, the top row, the track (marker
        # band and strip), the legend row and the key row (11 px text at the page's line height), three gaps; and two
        # lines of the message.
        def number(pattern):
            m = re.search(pattern, self.page)
            self.assertTrue(m, pattern)
            return float(m.group(1))
        line = number(r':root\{[^}]*font:14px/([\d.]+) var\(--font\)')
        gap, margin = number(r'\.pz\{display:flex;flex-direction:column;gap:(\d+)px'), number(r'\.pz\{[^}]*margin:(\d+)px auto 0')
        top = number(r'\.pz-row\{[^}]*min-height:(\d+)px')
        band, strip = number(r'\.pz-track\{position:relative;padding-top:(\d+)px'), number(r'#pzStrip\{[^}]*height:(\d+)px')
        foot = max(number(r'\.pz-foot\{[^}]*min-height:(\d+)px'), number(r'\.pz-foot\{[^}]*font-size:(\d+)px')*line)
        note = max(number(r'\.pz-note\{[^}]*min-height:(\d+)px'), number(r'\.pz-note\{[^}]*font-size:(\d+)px')*line)
        full = margin + top + band + strip + foot + note + 3*gap
        self.assertEqual((top, band + strip, foot, note), (26, 28, 16.5, 16.5))
        self.assertAlmostEqual(number(r'#viewBoard\.pz-on\{--pz-h:(\d+)px\}'), full, delta=.5)
        msg = 2*number(r'\.pz-msg\{[^}]*font-size:([\d.]+)px')*number(r'\.pz-msg\{[^}]*line-height:([\d.]+)')
        self.assertAlmostEqual(number(r'#viewBoard\.pz-line\{--pz-h:(\d+)px\}'), margin + msg, delta=.5)


@unittest.skipUnless(NODE, 'needs node to run the page scripts')
class RunAt(unittest.TestCase):
    def test_the_run_at_a_time(self):
        script = r"""
const fs = require('fs'), vm = require('vm');
const sandbox = {document: {}, console};
sandbox.window = sandbox;
vm.createContext(sandbox);
vm.runInContext(fs.readFileSync(FILE, 'utf8'), sandbox);
const P = sandbox.Possession, step = 1 / 30;
const runs = [{t0: 1, t1: 1.9667}, {t0: 2, t1: 2.9667}, {t0: 4, t1: 4.5}];
const at = (t, hint) => P.runAt(runs, t, step, hint);
process.stdout.write(JSON.stringify({
  before: at(.9), edge: at(1 - step * .4), first: at(1), between: at(1.98), second: at(2.5), lastFrame: at(2.9667), gap: at(3.5), third: at(4.2),
  after: at(4.5 + 2 * step), tail: at(4.5 + step), wayAfter: at(9), hinted: at(2.5, 0), wrongHint: at(4.2, 0), badHint: at(1.5, 7), empty: P.runAt([], 1, step), nan: at(NaN),
  clock: [P.clock(83.4), P.clock(0), P.clock(3599.999)]}));
""".replace('FILE', json.dumps(FILES['possession.js']))
        out = node(script)
        self.assertEqual(out, {'before': -1, 'edge': 0, 'first': 0, 'between': 0, 'second': 1, 'lastFrame': 1, 'gap': -1, 'third': 2, 'after': -1,
                               'tail': 2, 'wayAfter': -1, 'hinted': 1, 'wrongHint': 2, 'badHint': 0, 'empty': -1, 'nan': -1,
                               'clock': ['01:23.40', '00:00.00', '60:00.00']})


if __name__ == '__main__':
    unittest.main()
