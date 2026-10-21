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

Run:  python -m unittest test_coaching_panel          (from this directory)
"""
import json
import os
import re
import shutil
import subprocess
import tempfile
import unittest

from test_coaching_store import fixture_report, synthetic_result
from soccer.coaching import pdf, store

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
             context: select(job.context || 'competitive'), generate: new Node('button'), pdf: new Node('button')};
  const s = {grid, main, side, c, asked: [], posted: [], jumps: [], downloads: [], answers: answers.slice(), pending: []};
  // job.manual: every request waits until the scenario answers it (s.pending), to order late answers.
  const api = job.manual ? (url) => { s.asked.push(url); return new Promise((res) => s.pending.push({url, res})); }
    : async (url) => { s.asked.push(url); const a = s.answers.length > 1 ? s.answers.shift() : s.answers[0]; if (a && a.__error) throw new Error(a.__error); return copy(a); };
  const post = async (url, body) => { s.posted.push({url, body: copy(body)}); if (job.postError) throw new Error(job.postError); return {ok: true}; };
  s.banner = new Node('div');
  s.view = sandbox.Coaching.create({main, side, banner: s.banner, controls: c, api, post, run: () => 'run1', video: {currentTime: 0},
                                    board: {colours: {A: '#b98cff', B: '#f6b564'}}, jump: (ev) => { s.jumps.push(ev); return true; },
                                    download: async (u) => { s.downloads.push(u); if (job.downloadError) throw new Error(job.downloadError); }, media: {matches: !!job.narrow, addEventListener() {}}});
  return s;
}
const summary = (s) => {
  const sections = find(s.main, (n) => n.dataset.section).map((n) => ({key: n.dataset.section, text: n.textContent}));
  const cards = find(s.main, (n) => n.tagName === 'ARTICLE').map((n) => ({observation: n.dataset.observation, cls: n.className, text: n.textContent}));
  const figures = find(s.side, (n) => n.tagName === 'FIGURE');
  return {sections, cards, banner: {hidden: s.banner.hidden, cls: s.banner.className, text: s.banner.textContent},
          generate: {text: s.c.generate.textContent, disabled: s.c.generate.disabled}, pdf: {disabled: s.c.pdf.disabled},
          side: {text: s.side.textContent, parent: s.side.parentElement === s.grid ? 'grid' : s.side.parentElement ? 'main' : null, figures: figures.length,
                 images: find(s.side, (n) => n.tagName === 'IMG').map((n) => n.src),
                 overlays: find(s.side, (n) => cls(n).includes('cr-overlay')).length, diagrams: find(s.side, (n) => cls(n).includes('cr-diagram')).length},
          asked: s.asked, posted: s.posted, jumps: s.jumps.map((j) => ({frame: j.frame, seek: j.seek, view: j.view, focus: j.focus, diagram: !!j.diagram, overlay: !!j.overlay})),
          downloads: s.downloads, custom: {hidden: s.c.custom.hidden, t0: s.c.t0.value, t1: s.c.t1.value}};
};
const buttonsIn = (root, text) => find(root, (n) => n.tagName === 'BUTTON' && n.textContent === text);

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
  for (const job of input.jobs) out.push(await (job.kind === 'drawing' ? drawing : job.kind === 'brackets' ? brackets : scenario)(job));
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
        self.assertIn('Observationstotal 4 · shown 3', section(out, 'overview'), 'a group of counts is one row')
        self.assertIn('Measured secondsspacing 11.9 s · movement 4.0 s', section(out, 'overview'))
        self.assertIn('ContextThe ball was in the middle third', section(out, 'overview'), 'block height and ball: the context')
        self.assertIn('Strengths · 1', section(out, 'strengths'))
        self.assertIn('Areas to improve · 1', section(out, 'improvements'))
        training = section(out, 'training')
        for text in ('Compact lines', 'Purpose. Keep the distance', 'Players. 8–12 players. Two lines of four.', 'Duration. 15–20 min. Four blocks.',
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
        self.assertIn("Shapes drawn from the players’ feet on this frame", side['text'])
        self.assertIn('distances estimated (assumed 105 × 68 m pitch)', side['text'])
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
        self.assertIn('An old finding · Incorrect · “The gap was a pressing trigger” (run run0, Team A as labelled in that run',
                      section(self.reload['first'], 'feedback'), 'a review from an earlier run is listed apart')
        self.assertIn('Coaching reports appear here after a run.', self.reload['after']['sections'][0]['text'])

    def test_reviews_on_no_finding_are_listed_apart(self):
        out = self.apart['first']
        fb = section(out, 'feedback')
        self.assertIn('From an earlier run', fb)
        self.assertIn('Gap behind A-03 · Incorrect · “Wrong side” (run run0, Team B as labelled in that run', fb,
                      "an earlier run's text keeps that run's ids (A-03 is shown as A-02 in this run)")
        self.assertIn('Your reviews of this run that match no finding here', fb)
        self.assertIn('Old width finding · Correct', fb)
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
        self.assertIn('The analysed segment runs 00:01.00–00:12.97', out['customTitle'])
        self.assertFalse(out['banner']['hidden'])
        self.assertIn('The range 00:00.00–00:08.00 reaches outside the analysed segment (00:01.00–00:12.97, source-video time), '
                      'so the report covers 00:01.00–00:08.00.', out['banner']['text'])
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
        self.assertIn('Team A attacks →', svg['texts'])
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


class PageWiring(unittest.TestCase):
    """index.html: the fourth tab and view, the module, the overlay canvas, the controller and the keyboard."""

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
        for control in ('crTeam', 'crRange', 'crCustom', 'crT0', 'crT1', 'crContext', 'crGenerate', 'crPdf', 'crBanner', 'crMain', 'crSide'):
            self.assertIn(f'id="{control}"', p, control)
            self.assertIn(f"$('{control}')", self.src, control)
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
  const api = () => null, jumpToEvidence = () => true, video = {}, tactical = {}, resultRun = null;
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
        block = span(self.src, 'let coachMark=null;', "video.addEventListener('play',clearCoachMark);")
        frame_index = js_function(self.src, 'frameIndex')
        frames = [{'time': round(f/30, 6), 'frame': f} for f in range(30, 60)]
        script = r"""
const result = {frames: FRAMES}, info = {video: {fps: 30}};
let mode = null; const shown = [], seeks = [], selected = [], drawn = [], listeners = {};
let invalidated = 0, cleared = 0;
const closeEditor = () => {}, views = {show: (v) => shown.push(v)}, playback = {seek: (t) => { seeks.push(t); video.currentTime = t; }}, select = (id) => selected.push(id);
const video = {paused: false, currentTime: 0, pause() { this.paused = true; }, addEventListener: (t, f) => { listeners[t] = f; }};
const canvas = {width: 1920, height: 1080}, ctx = {clearRect: () => { cleared++; }}, $ = () => ({width: 0, height: 0, getContext: () => ctx});
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


if __name__ == '__main__':
    unittest.main()
