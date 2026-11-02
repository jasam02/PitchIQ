"""The KEY MOMENTS coaching report of a full match in the review page: coaching.js run in node on the two example
reports of the CAN-ISL match (soccer/moments/report.py, kind 'coaching-moments' v1; moments_example_A/B.json.gz, as
the lead's backend built them at level 7), the page's moment player (index.html watchMoment, run in node with the
page's own coach-mark block) and what match.js does for a moment in a segment whose view is not loaded.

coaching.js (classes MomentsReport, MomentsStates):
- in match playback (run 'match-<id>') the tab shows the eight sections in order (summary, did well, needs
  improvement, key match moments, patterns, training, metrics, coverage) and More moments collapsed; the v2 report's
  options bar and evidence column give way to the summary's own team choice and Generate;
- a card carries its badge (Positive / Review), time and half, the players with their roles shown by the display labels
  of the result shown (never the internal ids), the finding, an improvement's suggestion and alternatives or "No
  validated alternative position", the top-down diagram, WATCH, and its details on demand;
- the timeline has one marker per key moment, in its half, coloured by category, at its time;
- WATCH hands the page the moment (window t0 / t1 / key, the follow list, the video overlay and the diagram, the seek of
  the key frame's centre, the chip's text with display labels); Previous / Next move along the list it came from;
- the team switch asks for the other team; a build is shown with its progress and asked again until ready; Generate
  posts the build, and is posted again while another build runs; an error is said; the v2 report stays reachable and
  leads back; a Segment run never asks for key moments, and a server without the route keeps the v2 report;
- the Goals section (class GoalsSection: report.goals, cut from the CAN-ISL report): right after the summary, the goals
  scored (what went well) and conceded (what went wrong), each card with its time, the score after it (the report team
  first), how it came about in words, "time uncertain", the findings with their source chips and time links (the video
  paused there), the notes of a goal with no findings, WATCH (its build-up on the moment player, Previous / Next along
  its group; the v2 jump without the player); no section for an older report, a quiet line before the match state.

index.html (class MomentPlayer): the moment player on the page's one clock: seek to the window's start, play the
lead-up with the people followed, pause on the key frame (the frame's centre) and draw the whole overlay there (the
diagram on the board), Continue to the window's end, Replay, Video / 2D, Previous / Next and Back, the chip's text and
explanation; in match playback it waits for the view owning the time and keeps the moment across a view swap.

match.js (class MatchJump): waiting(t) and a swap that keeps a moment being watched (any other coaching mark is cleared).

Run:  python -m unittest test_moments_panel          (from this directory; about 10 s)
"""
import copy
import gzip
import json
import os
import re
import shutil
import subprocess
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for _p in (ROOT, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from test_coaching_panel import (CATALOG, COACH_BLOCK_END, js_function, page, page_script, span, v2_answer,  # noqa: E402
                                 v2_base, V2_SECTIONS)

NODE = shutil.which('node')
FILES = {name: os.path.join(ROOT, name) for name in ('tactical.js', 'coaching.js')}
MATCH_JS = os.path.join(ROOT, 'match.js')
RUN = 'match-20261028-062138-83f2a6'
SECTIONS = ['mSummary', 'mStrengths', 'mImprovements', 'mKey', 'mPatterns', 'mTraining', 'mMetrics', 'mCoverage', 'mMore']
DOT = chr(0xb7)


def example(team):
    with gzip.open(os.path.join(HERE, f'moments_example_{team}.json.gz'), 'rt', encoding='utf-8') as handle:
        return json.load(handle)


REPORT_A, REPORT_B = example('A'), example('B')


def ids_of(*reports):
    """Every player id the reports name (moments, roles, diagrams, overlays), by team."""
    out = set()
    for r in reports:
        for m in r['moments'].values():
            for group in m['players'].values():
                out.update(group)
            out.update(m['roles'])
            for spec in (m['evidence']['diagram'], m['evidence']['videoOverlay']):
                out.update(e['id'] for e in spec['elements'] if isinstance(e.get('id'), str))
    return sorted(i for i in out if re.fullmatch(r'[AB]-\d+', i))


def relabelled():
    """The display labels of a corrected roster: every id of a team renumbered A-01, A-02, ... (never its own id)."""
    labels = {}
    for team in 'AB':
        ids = [i for i in ids_of(REPORT_A, REPORT_B) if i[0] == team]
        for k, i in enumerate(ids):
            labels[i] = f'{team}-{k+1:02d}'
    return labels


LABELS = relabelled()
assert all(LABELS[i] != i for i in ('A-62', 'A-63', 'B-45'))


def match_result(labels=None):
    """The match-level object match.js hands the panels (matchObject): the people with their display labels."""
    labels = labels or {}
    people = [{'id': i, 'display': labels.get(i, i), 'team': i[0], 'role': 'PLAYER', 'status': 'ACTIVE'} for i in ids_of(REPORT_A, REPORT_B)]
    return {'version': 3, 'video': {'fps': 25.0, 'width': 1920, 'height': 1080, 'duration': 5791.66},
            'matchLevel': {'matchId': RUN[6:], 'segments': 33},
            'match': {'players': people, 'goalkeepers': [], 'referees': [], 'retired': [], 'excluded': [], 'withdrawn': []},
            'frames': [{'frame': 250, 'time': 10.0, 'people': []}, {'frame': 144700, 'time': 5788.0, 'people': []}], 'tactical': {}}


INFO = {'video': {'fps': 25.0, 'width': 1920, 'height': 1080, 'duration': 5791.66}, 'videoKey': '3'*64}
BLOCK = {'squad': None, 'ageGroup': 'Adult', 'playingLevel': 7, 'format': {'players': 11, 'goalkeeper': True}, 'field': None, 'federation': 'auto',
         'formation': {'intended': None, 'periods': []}, 'style': 'unknown', 'roles': {}, 'runLinks': []}
SETTINGS = {'run': RUN, 'settings': {'version': 1, 'teams': {'A': BLOCK, 'B': None}, 'updated': None}, 'inferred': {'A': None, 'B': None}, 'squads': []}
PROFILE = {'level': 7, 'ageGroup': 'Adult', 'format': '11v11', 'source': 'team'}


def ready(report):
    return {'state': 'ready', 'report': report, 'profile': dict(PROFILE, source='team' if report['team'] == 'A' else 'other team')}


def mjob(steps=None, moments=None, **kw):
    """One page scenario: the coaching view in match playback with the moments routes answering from `moments` (by team,
    a list of answers given in turn, the last one repeated) and `build` (the same, for POST)."""
    return dict({'run': RUN, 'result': match_result(LABELS), 'info': INFO, 'profiles': CATALOG or {}, 'settings': SETTINGS,
                 'moments': moments if moments is not None else {'A': [ready(REPORT_A)], 'B': [ready(REPORT_B)]}, 'build': {}, 'v2': [],
                 'steps': steps or []}, **kw)


HARNESS = r"""
const fs = require('fs'), vm = require('vm');
const input = JSON.parse(fs.readFileSync(0, 'utf8'));
class Node {
  constructor(tag) {
    this.tagName = String(tag).toUpperCase(); this.children = []; this.parentElement = null; this.attrs = {}; this.dataset = {}; this.style = {};
    this.listeners = {}; this.hidden = false; this.disabled = false; this.value = ''; this.own = ''; this.cls = ''; this.title = ''; this.id = '';
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
  prepend(...cs) { for (const c of cs.slice().reverse()) { if (c.parentElement) c.parentElement.removeChild(c); c.parentElement = this; this.children.unshift(c); } }
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
  scrollIntoView(o) { scrolled.push([this.dataset.moment || this.tagName, o && o.block]); }
  getContext() { return new Proxy({}, {get: () => () => {}}); }
}
const scrolled = [];
const all = (n, out = []) => { out.push(n); for (const c of n.children || []) all(c, out); return out; };
const find = (root, pred) => all(root).filter((n) => n instanceof Node && pred(n));
const cls = (n) => (n.attrs.class || n.className || '');
const has = (n, c) => cls(n).split(' ').includes(c);
const sandbox = {document: {createElement: (t) => new Node(t), createElementNS: (ns, t) => new Node(t), body: new Node('body'),
                            createTextNode: (t) => ({nodeType: 3, textContent: String(t), parentElement: null})},
                 console, setTimeout: (f) => sandbox.hold ? (sandbox.held.push(f), 0) : setImmediate(f), clearTimeout: (h) => { if (h) clearImmediate(h); },
                 URLSearchParams, Map, Set, JSON, hold: false, held: [], localStorage: {getItem: () => null, setItem() {}}, devicePixelRatio: 1, addEventListener() {}};
sandbox.window = sandbox;
vm.createContext(sandbox);
for (const path of Object.values(input.files)) vm.runInContext(fs.readFileSync(path, 'utf8'), sandbox, {filename: path});
const settle = async () => { for (let k = 0; k < 60; k++) await new Promise((r) => setImmediate(r)); };
const copy = (x) => JSON.parse(JSON.stringify(x));

function setup(job) {
  const grid = new Node('div'), main = new Node('div'), side = new Node('aside'), bar = new Node('div');
  grid.append(main, side);
  const select = (v) => Object.assign(new Node('select'), {value: v});
  const c = {team: select(job.team || 'A'), range: select('whole'), custom: new Node('span'), t0: new Node('input'), t1: new Node('input'), setup: new Node('div'),
             generate: new Node('button'), pdf: new Node('button')};
  bar.append(c.team, c.range, c.custom, c.generate, c.pdf);
  const heading = {title: new Node('h2'), lede: new Node('p')};
  heading.title.textContent = 'Tactical observations from this run'; heading.lede.textContent = 'Coaching for your team.';
  const s = {grid, main, side, bar, c, heading, banner: new Node('div'), asked: [], posted: [], watches: [], jumps: []};
  const next = (map, key, what) => {
    const list = map && map[key];
    if (!list || !list.length) throw Object.assign(new Error('NOT FOUND'), {status: 404});
    const a = list.length > 1 ? list.shift() : list[0];
    if (a && a.__status) throw Object.assign(new Error(a.__error || 'failed'), {status: a.__status});
    return copy(a);
  };
  const api = async (url) => {
    s.asked.push(url);
    if (url.startsWith('/api/coaching/moments')) return next(job.moments, new URLSearchParams(url.split('?')[1] || '').get('team'));
    if (url.startsWith('/api/coaching?')) return next({v2: job.v2}, 'v2');
    const kind = url.startsWith('/api/coaching/profiles') ? 'profiles' : url.startsWith('/api/coaching/settings') ? 'settings' : null;
    if (kind && job[kind]) return copy(job[kind]);
    throw Object.assign(new Error('NOT FOUND'), {status: 404});
  };
  const post = async (url, body) => {
    s.posted.push({url, body: copy(body)});
    if (url === '/api/coaching/moments/build') return next(job.build, body.team);
    return {ok: true};
  };
  s.view = sandbox.Coaching.create({main, side, banner: s.banner, controls: c, api, post, run: () => job.run, video: {currentTime: 0},
                                    board: {colours: {A: '#b98cff', B: '#f6b564'}}, jump: (ev) => { s.jumps.push(ev); return true; },
                                    watch: job.noWatch ? undefined : (ev) => { s.watches.push(ev); return true; }, heading,
                                    media: {matches: false, addEventListener() {}}});
  return s;
}
const text = (n) => (n ? n.textContent : null);
function summary(s) {
  const one = (root, pred) => find(root, pred)[0] || null;
  const cards = find(s.main, (n) => n.tagName === 'ARTICLE' && n.dataset.moment).map((n) => {
    const det = one(n, (m) => has(m, 'km-details'));
    return {moment: n.dataset.moment, from: n.dataset.from, cls: n.className, label: n.attrs['aria-labelledby'] || null, text: n.textContent,
            badge: text(one(n, (m) => has(m, 'cr-badge'))), when: text(one(n, (m) => has(m, 'km-when'))), title: text(one(n, (m) => has(m, 'km-title'))),
            finding: text(one(n, (m) => m.dataset.km === 'finding')), who: text(one(n, (m) => m.dataset.km === 'players')),
            suggestion: find(n, (m) => m.dataset.km === 'suggestion').map(text), alternatives: find(n, (m) => m.dataset.km === 'alternative').map(text),
            noSuggestion: find(n, (m) => m.dataset.km === 'no-suggestion').map(text), uncertain: find(n, (m) => m.dataset.km === 'uncertain').length,
            diagrams: find(n, (m) => has(m, 'cr-diagram')).length, ids: find(n, (m) => cls(m) === 'cr-id').map((m) => [m.textContent, m.title]),
            details: det ? {hidden: det.hidden, text: det.textContent} : null, watch: find(n, (m) => m.tagName === 'BUTTON' && m.dataset.watch).length,
            selected: has(n, 'selected')};
  });
  const names = find(s.main, (n) => has(n, 'km-tl-ax')).map((n) => text(n.children[1]));
  const halves = find(s.main, (n) => has(n, 'km-tl-half')).map((n, k) => ({flex: n.style.flex, name: names[k],
    height: n.style.height, marks: n.children.filter((c) => c.dataset && c.dataset.mark).map((c) => ({id: c.dataset.mark, cls: c.className, left: c.style.left, bottom: c.style.bottom, label: c.attrs['aria-label']}))}));
  const build = one(s.main, (n) => has(n, 'km-build')), gen = build && one(build, (n) => n.dataset.km === 'generate'), status = build && one(build, (n) => n.dataset.km === 'status');
  const bar = build && one(build, (n) => n.tagName === 'PROGRESS'), more = one(s.main, (n) => has(n, 'km-more-list'));
  // the Goals section: its groups and goal cards (a finding: its own text, the source chip, the time link)
  const goals = find(s.main, (n) => n.tagName === 'ARTICLE' && n.dataset.goal).map((n) => {
    const score = one(n, (m) => m.dataset.km === 'score'), named = n.attrs['aria-labelledby'];
    return {goal: n.dataset.goal, group: n.dataset.group, cls: n.className, labelled: !!named && !!one(n, (m) => m.id === named),
            badge: text(one(n, (m) => has(m, 'cr-badge'))), when: text(one(n, (m) => has(m, 'km-when'))), score: text(score), scoreTitle: score ? score.title : null,
            title: text(one(n, (m) => m.dataset.km === 'goal-title')), tags: find(n, (m) => has(m, 'km-tag')).map(text),
            findings: find(n, (m) => m.dataset.km === 'finding').map((r) => {
              const body = one(r, (m) => has(m, 'km-gf-body')), chip = one(r, (m) => m.dataset.km === 'source'), t = one(r, (m) => m.tagName === 'BUTTON');
              return {text: body.children.filter((c) => c !== chip).map((c) => c.textContent).join(''), chip: text(chip), tone: chip ? chip.className : null,
                      time: text(t), seek: t ? t.dataset.seek : null, ids: find(r, (m) => cls(m) === 'cr-id').map((m) => [m.textContent, m.title])};
            }),
            notes: find(n, (m) => m.dataset.km === 'note').map(text), none: find(n, (m) => has(m, 'cr-none')).map(text),
            watch: find(n, (m) => m.tagName === 'BUTTON' && m.dataset.goal).map(text), selected: has(n, 'selected')};
  });
  const goalGroups = find(s.main, (n) => n.dataset.goals).map((n) => ({group: n.dataset.goals, head: text(n.children[0]), intro: text(n.children[1]), cls: n.children[1].className}));
  return {
    sections: find(s.main, (n) => n.dataset.section).map((n) => ({key: n.dataset.section, text: n.textContent})),
    goals, goalGroups, jumpList: s.jumps.map((j) => ({seek: j.seek, time: j.time, start: j.start, window: j.window, replay: j.replay, caption: j.caption, explanation: j.explanation})),
    cards, halves,
    rows: find(s.main, (n) => n.tagName === 'LI' && has(n, 'km-row')).map((n) => ({moment: n.dataset.moment, text: n.textContent, cls: n.className})),
    build: build ? {state: build.dataset.state, button: text(gen), disabled: gen.disabled, primary: has(gen, 'primary'), status: text(status), error: has(status, 'error'),
                    bar: {hidden: bar.hidden, value: bar.value, max: bar.max}} : null,
    profile: text(one(s.main, (n) => n.dataset.km === 'profile')), summaryText: text(one(s.main, (n) => n.dataset.km === 'summary')),
    counts: text(one(s.main, (n) => n.dataset.km === 'counts')),
    teams: find(s.main, (n) => n.tagName === 'BUTTON' && n.dataset.team).map((n) => ({team: n.dataset.team, text: n.textContent, pressed: n.attrs['aria-pressed']})),
    more: more ? {hidden: more.hidden, cards: more.children.length} : null,
    metrics: find(s.main, (n) => n.tagName === 'TR').map((tr) => tr.children.map((c) => c.textContent)),
    watches: s.watches.map((w) => Object.assign({}, w, {diagram: !!w.diagram, overlay: w.overlay ? w.overlay.elements.length : 0})),
    jumps: s.jumps.length, asked: s.asked.slice(), posted: copy(s.posted), barHidden: s.bar.hidden, heading: [s.heading.title.textContent, s.heading.lede.textContent],
    sideHidden: s.side.hidden, grid: s.grid.className, back: find(s.main, (n) => n.dataset.km === 'back').map(text),
    banner: {hidden: s.banner.hidden, text: s.banner.textContent}, mView: s.view.mView, gone: s.view.mGone, scrolled: scrolled.slice(),
    setup: s.c.setup.textContent};
}
async function act(s, step) {
  const match = (n) => (step.data ? Object.entries(step.data).every(([k, v]) => n.dataset[k] === v) : true) && (step.text === undefined || n.textContent === step.text)
    && (step.tag === undefined || n.tagName === step.tag);
  if (step.do === 'click') {
    const root = step.within ? find(s.main, (m) => m.dataset.section === step.within)[0] : s.main;
    if (!root) throw new Error('no section ' + step.within);
    const n = find(root, (n) => (n.tagName === 'BUTTON' || n.tagName === 'LI') && match(n))[step.index || 0];
    if (!n) throw new Error('no button ' + JSON.stringify(step));
    n.click();
  }
  else if (step.do === 'method') s.view[step.name](...(step.args || []));
  else if (step.do === 'release') { sandbox.hold = !!step.hold; for (const f of sandbox.held.splice(0)) f(); }
  else if (step.do === 'active') s.view.setActive(step.value);
}
async function scenario(job) {
  sandbox.hold = !!job.hold; sandbox.held = [];
  const s = setup(job);
  s.view.load(job.result, job.info);
  s.view.setActive(true);
  await settle();
  const out = {first: summary(s)};
  for (const step of job.steps || []) { await act(s, step); await settle(); if (step.name) out[step.name] = summary(s); }
  out.last = summary(s);
  sandbox.hold = false; sandbox.held = [];
  return out;
}
(async () => {
  const out = [];
  for (const job of input.jobs) out.push(await scenario(job));
  process.stdout.write(JSON.stringify(out));
})().catch((error) => { process.stderr.write(String(error && error.stack || error)); process.exit(1); });
"""


def run(*jobs):
    done = subprocess.run([NODE, '-e', HARNESS], input=json.dumps({'files': FILES, 'jobs': list(jobs)}), capture_output=True, text=True,
                          encoding='utf-8', timeout=300)
    if done.returncode:
        raise AssertionError(done.stderr[-3000:])
    return json.loads(done.stdout)


def sec(out, key):
    return next(s['text'] for s in out['sections'] if s['key'] == key)


def card(out, moment, from_=None):
    return next(c for c in out['cards'] if c['moment'] == moment and (from_ is None or c['from'] == from_))


def shown(text):
    """A text as the page shows it: every id the example names replaced by its display label."""
    return re.sub(r'(?<![\w-])[AB]-\d+(?!\w)', lambda m: LABELS.get(m.group(0), m.group(0)), text)


@unittest.skipUnless(NODE, 'needs node to run the page scripts')
class MomentsReport(unittest.TestCase):
    """The report as it is shown in match playback, from the example report of Team A (level 7)."""

    @classmethod
    def setUpClass(cls):
        cls.out = run(mjob())[0]['last']
        cls.r = REPORT_A

    def test_the_eight_sections_then_more_moments(self):
        self.assertEqual([s['key'] for s in self.out['sections']], SECTIONS)
        for key, title in (('mSummary', 'Match coaching summary'), ('mStrengths', 'What the team did well'), ('mImprovements', 'What needs improvement'),
                           ('mKey', 'Key match moments'), ('mPatterns', 'Repeated tactical patterns'), ('mTraining', 'Training priorities'),
                           ('mMetrics', 'Supporting team metrics'), ('mCoverage', 'Data coverage and limitations'), ('mMore', 'More moments')):
            self.assertTrue(sec(self.out, key).startswith(title), key)
        self.assertIn(f'What the team did well {DOT} 8', sec(self.out, 'mStrengths'))
        self.assertIn(f'What needs improvement {DOT} 5', sec(self.out, 'mImprovements'))
        self.assertIn(f'Key match moments {DOT} 11', sec(self.out, 'mKey'))
        self.assertEqual(self.out['more'], {'hidden': True, 'cards': 0}, 'more moments are collapsed and drawn on request')

    def test_the_summary(self):
        o = self.out
        self.assertEqual(o['summaryText'], self.r['summary']['text'])
        self.assertEqual(o['teams'], [{'team': 'A', 'text': 'Team A (white)', 'pressed': 'true'}, {'team': 'B', 'text': 'Team B (red)', 'pressed': 'false'}])
        self.assertIn('Read for Adult', o['profile'])
        self.assertIn('level 7', o['profile'])
        self.assertIn("from Team A\u2019s coaching setup", o['profile'])
        self.assertIn('8 done well', o['counts'])
        self.assertIn('5 to improve', o['counts'])
        self.assertIn('11 key moments', o['counts'])
        self.assertIn('4 repeated patterns', o['counts'])
        self.assertIn('27 more moments', o['counts'])
        self.assertEqual((o['build']['state'], o['build']['button'], o['build']['disabled'], o['build']['primary']), ('ready', 'Refresh', False, False))
        self.assertIn('Up to date for this team and level', o['build']['status'])
        self.assertIn('Previous report (team-shape checks)', sec(o, 'mSummary'))

    def test_the_chrome_of_the_moments_report(self):
        o = self.out
        self.assertTrue(o['barHidden'], 'the v2 options bar (range, PDF) is hidden: the summary has the team and Generate')
        self.assertTrue(o['sideHidden'])
        self.assertIn('km-one', o['grid'], 'one column: no evidence column')
        self.assertEqual(o['heading'][0], 'Key moments of the match')
        self.assertFalse(any('/api/coaching?' in a for a in o['asked']), 'the v2 report is not asked for')
        self.assertTrue(any(a.startswith('/api/coaching/moments?') and 'team=A' in a and f'run={RUN}' in a for a in o['asked']))

    def test_cards_carry_badge_time_players_finding_with_display_labels(self):
        o, r = self.out, self.r
        strengths = [c for c in o['cards'] if c['from'] == 'strengths']
        improvements = [c for c in o['cards'] if c['from'] == 'improvements']
        self.assertEqual([c['moment'] for c in strengths], r['strengths'])
        self.assertEqual([c['moment'] for c in improvements], r['improvements'])
        for c in strengths + improvements:
            m = r['moments'][c['moment']]
            with self.subTest(moment=c['moment']):
                self.assertEqual(c['badge'], 'Positive' if m['category'] == 'positive' else 'Review')
                self.assertIn('k-strength' if m['category'] == 'positive' else 'k-improvement', c['cls'])
                self.assertEqual(c['when'], f"{m['clock']} {DOT} {m['half']}")
                self.assertEqual(c['title'], shown(m['title']))
                self.assertEqual(c['finding'], shown(m['finding']))
                self.assertEqual(c['diagrams'], 1, 'the top-down diagram')
                self.assertEqual(c['watch'], 1)
                self.assertEqual(c['details']['hidden'], True, 'details on demand')
                self.assertTrue(c['label'], 'the card is named by its title')
                for pid in [p for g in m['players'].values() for p in g]:
                    self.assertIn(LABELS[pid], c['who'])
                    self.assertNotIn(pid, c['who'].replace(LABELS[pid], ''), 'never the internal id')
                for role in m['roles'].values():
                    self.assertIn(role, c['who'])
        e = card(o, 'mo-e67e0da74a14')
        self.assertEqual(e['who'], f"Team A (white) {DOT} {LABELS['A-62']} central midfielder {DOT} with {LABELS['A-63']} left centre-back {DOT} against {LABELS['B-45']} striker")
        self.assertIn([LABELS['A-62'], f"{LABELS['A-62']} (internal id A-62)"], e['ids'], 'the id is in the tooltip')
        self.assertNotIn('A-62', e['finding'])

    def test_a_full_matchs_person_ids_by_the_reports_labels(self):
        # a full match names people by person ids ('A-P281'), not the track-id shape the text pass rewrites: the card looks
        # each one up in the page's label map, else shows the label the report gives it (report.labels), the id in the
        # tooltip
        r = copy.deepcopy(REPORT_A)
        m = r['moments']['mo-e67e0da74a14']
        rename = {'A-62': 'A-P281', 'A-63': 'A-P279', 'B-45': 'B-P286'}
        m['players'] = {k: [rename.get(i, i) for i in v] for k, v in m['players'].items()}
        m['roles'] = {rename.get(i, i): v for i, v in m['roles'].items()}
        r['labels'] = {'A-P281': 'A-07?', 'A-P279': 'A-03', 'B-P286': 'B-08?'}
        e = card(run(mjob(moments={'A': [ready(r)], 'B': [ready(REPORT_B)]}))[0]['last'], 'mo-e67e0da74a14')
        self.assertEqual(e['who'], f"Team A (white) {DOT} A-07? central midfielder {DOT} with A-03 left centre-back {DOT} against B-08? striker")
        self.assertIn(['A-07?', 'A-07? (internal id A-P281)'], e['ids'])
        self.assertNotIn('A-P', e['who'])
        result = match_result(LABELS)
        result['match']['players'].append({'id': 'A-P281', 'display': 'A-11', 'team': 'A', 'role': 'PLAYER', 'status': 'ACTIVE'})
        e = card(run(mjob(result=result, moments={'A': [ready(r)], 'B': [ready(REPORT_B)]}))[0]['last'], 'mo-e67e0da74a14')
        self.assertTrue(e['who'].startswith(f"Team A (white) {DOT} A-11 central midfielder"), 'the page\'s own label first (the roster now)')
        self.assertIn(['A-11', 'A-11 (internal id A-P281)'], e['ids'])

    def test_an_improvement_shows_its_suggestion_or_that_there_is_none(self):
        o, r = self.out, self.r
        for mid in r['improvements']:
            m, c = r['moments'][mid], card(o, mid, 'improvements')
            with self.subTest(moment=mid):
                if m['suggestion']:
                    self.assertEqual(c['suggestion'], [f"Suggestion. {shown(m['suggestion']['text'])}"])
                    alts = [a['text'] for a in m['suggestion'].get('alternatives') or []]
                    self.assertEqual(c['alternatives'], [f'Or. {shown(t[0].upper() + t[1:])}' for t in alts])
                    self.assertEqual(c['noSuggestion'], [])
                else:
                    self.assertTrue(m['noSuggestion'])
                    self.assertEqual(c['noSuggestion'], ['No validated alternative position.'])
        self.assertTrue(any(card(o, mid, 'improvements')['noSuggestion'] for mid in r['improvements']), 'the example has one without')
        self.assertTrue(any(card(o, mid, 'improvements')['alternatives'] for mid in r['improvements']), 'and one with an alternative')

    def test_the_timeline_has_one_marker_per_key_moment(self):
        o, r = self.out, self.r
        self.assertEqual([h['name'] for h in o['halves']], ['first half', 'second half'])
        periods = r['match']['periods']
        self.assertEqual([h['flex'] for h in o['halves']], [f"{round(p['end'] - p['start'])} 1 0" for p in periods], 'widths by the halves\' lengths')
        marks = [m for h in o['halves'] for m in h['marks']]
        self.assertEqual(sorted(m['id'] for m in marks), sorted(r['keyMoments']))
        self.assertEqual(len(marks), len(r['keyMoments']))
        for h, p in zip(o['halves'], periods):
            for mk in h['marks']:
                m = r['moments'][mk['id']]
                with self.subTest(moment=mk['id']):
                    self.assertTrue(p['start'] <= m['time'] <= p['end'], 'in its half')
                    self.assertAlmostEqual(float(mk['left'].rstrip('%')), 100*(m['time'] - p['start'])/(p['end'] - p['start']), places=1)
                    self.assertIn('k-strength' if m['category'] == 'positive' else 'k-improvement', mk['cls'])
                    self.assertIn(m['clock'], mk['label'])
                    self.assertIn('Watch the moment', mk['label'])
        self.assertEqual(sum(len(h['marks']) for h in o['halves'][1:]), 3, 'three key moments in the second half')
        for h in o['halves']:
            near = [(a, b) for a in h['marks'] for b in h['marks'] if a['id'] < b['id'] and abs(float(a['left'][:-1]) - float(b['left'][:-1])) < 2.6]
            for a, b in near:
                self.assertNotEqual(a['bottom'], b['bottom'], 'close markers stand on separate lanes')
        self.assertEqual({h['height'] for h in o['halves']}, {'54px'}, 'three lanes of markers (15:44, 16:07 and 16:27 are close)')
        times = [r['moments'][row['moment']]['time'] for row in o['rows']]
        self.assertEqual(times, sorted(times), 'the list is in match order')
        self.assertEqual(sorted(row['moment'] for row in o['rows']), sorted(r['keyMoments']))

    def test_patterns_training_metrics_coverage(self):
        o, r = self.out, self.r
        pats = sec(o, 'mPatterns')
        for p in r['patterns']:
            self.assertIn(p['title'], pats)
            self.assertIn(p['text'], pats)
        self.assertIn('Block Compactness: Shadow to 8v8', pats, 'the drill of the pattern that has one')
        train = sec(o, 'mTraining')
        t = r['training'][0]
        for part in (t['title'], t['objective'], t['drill']['name'], '25 min', t['drill']['setup'], *t['drill']['coachingPoints']):
            self.assertIn(part, train)
        self.assertEqual(o['metrics'][0], ['Measure', 'With the ball', 'Without the ball'])
        self.assertIn(['Team width', '52.1 m', '34.9 m'], o['metrics'])
        self.assertIn(['Back line to midfield', '15.1 m', '10.5 m'], o['metrics'])
        met = sec(o, 'mMetrics')
        self.assertIn('4-3-3 21%', met)
        self.assertIn('Team A (white) 08:45 (9%)', met, 'possession in minutes and shares of the estimate')
        self.assertIn(r['metrics']['note'], met)
        cov = sec(o, 'mCoverage')
        for item in r['coverage']['items']:
            self.assertIn(item, cov)
        self.assertIn('PitchIQ found 49 moments of this team (127 for both teams)', cov)


@unittest.skipUnless(NODE, 'needs node to run the page scripts')
class MomentsStates(unittest.TestCase):
    """Watching, the team switch, the build states, the v2 report, and the runs that never show key moments."""

    def test_watch_hands_the_page_the_whole_moment(self):
        mid = 'mo-e67e0da74a14'
        out = run(mjob(steps=[{'do': 'click', 'data': {'watch': mid}, 'tag': 'BUTTON', 'within': 'mImprovements', 'name': 'watched'},
                              {'do': 'method', 'name': 'momentNav', 'args': [1]},
                              {'do': 'method', 'name': 'setMomentView', 'args': ['board']},
                              {'do': 'method', 'name': 'momentNav', 'args': [-1]},
                              {'do': 'click', 'data': {'mark': 'mo-019ebf9fd8a8'}, 'tag': 'BUTTON', 'name': 'marked'}]))[0]
        w = out['watched']['watches'][0]
        m, ev = REPORT_A['moments'][mid], REPORT_A['moments'][mid]['evidence']
        self.assertEqual((w['moment'], w['start'], w['view'], w['tier']), (mid, 'leadup', 'tracking', 'world'))
        self.assertEqual(w['window'], ev['window'])
        self.assertEqual((w['frame'], w['seek'], w['time']), (ev['frame'], ev['seek'], ev['time']), 'the key frame\'s centre')
        self.assertEqual(w['follow'], ev['follow'])
        self.assertEqual(w['focus'], ev['players'])
        self.assertTrue(w['diagram'])
        self.assertEqual(w['overlay'], len(ev['videoOverlay']['elements']), 'the video overlay as the report made it')
        c = w['chip']
        self.assertEqual((c['title'], c['finding'], c['badge'], c['kind'], c['counter'], c['prev'], c['next']),
                         (shown(m['title']), shown(m['finding']), 'Review', 'improvement', '1 of 5', False, True))
        self.assertEqual(c['when'], f'29:31 {DOT} first half')
        leads = [d[0] for d in c['details']]
        self.assertEqual(leads, ['What happened.', 'Why it mattered.', 'Assessment.', 'Suggestion.', 'Or.', 'Coaching pointer.', 'Data reliability.'])
        self.assertTrue(all('A-62' not in d[1] for d in c['details']), 'the chip names players by their display labels')
        self.assertIn(LABELS['A-63'], dict(c['details'])['Suggestion.'])
        self.assertIn('selected', card(out['watched'], mid, 'improvements')['cls'])
        later = out['last']['watches']
        self.assertEqual([x['moment'] for x in later[1:3]], [REPORT_A['improvements'][1], mid], 'Next, then Previous, along the list it came from')
        self.assertEqual(later[1]['chip']['counter'], '2 of 5')
        self.assertEqual(later[2]['view'], 'board', 'the coach\'s 2D choice is kept')
        key = sorted(REPORT_A['keyMoments'], key=lambda i: REPORT_A['moments'][i]['time'])
        self.assertEqual(later[3]['moment'], 'mo-019ebf9fd8a8')
        self.assertEqual(later[3]['chip']['counter'], f"{key.index('mo-019ebf9fd8a8') + 1} of {len(key)}", 'a timeline marker watches along the key moments')
        self.assertEqual(out['last']['jumps'], 0, 'the moment player, not the v2 jump')

    def test_watch_without_the_moment_player_falls_back_to_the_v2_jump(self):
        out = run(mjob(noWatch=True, steps=[{'do': 'click', 'data': {'watch': 'mo-e67e0da74a14'}, 'tag': 'BUTTON', 'within': 'mImprovements'}]))[0]
        self.assertEqual(out['last']['jumps'], 1)

    def test_the_team_switch_asks_for_the_other_team(self):
        out = run(mjob(steps=[{'do': 'click', 'data': {'team': 'B'}, 'tag': 'BUTTON', 'name': 'b'}]))[0]
        b = out['b']
        self.assertTrue(any(a.startswith('/api/coaching/moments?') and 'team=B' in a for a in b['asked']))
        self.assertEqual(b['summaryText'], REPORT_B['summary']['text'])
        self.assertEqual([t['pressed'] for t in b['teams']], ['false', 'true'])
        self.assertEqual([c['moment'] for c in b['cards'] if c['from'] == 'strengths'], REPORT_B['strengths'])
        self.assertIn("Team B has no coaching setup of its own", b['profile'])
        self.assertIn('Not saved for this team yet: its key moments are read for the other team', b['setup'], 'the setup says what the moments use')
        self.assertNotIn('Age group', b['setup'], 'and starts closed under the key moments')
        self.assertIn('Team A', out['first']['setup'], 'the saved team is summarised')

    def test_a_build_in_progress_is_shown_and_asked_again_until_ready(self):
        building = {'state': 'building', 'progress': {'stage': 'reading the match', 'done': 12, 'of': 33}, 'profile': PROFILE}
        out = run(mjob(hold=True, moments={'A': [building, building, ready(REPORT_A)]},
                       steps=[{'do': 'release', 'hold': True, 'name': 'polled'}, {'do': 'release', 'hold': False}]))[0]
        first = out['first']
        self.assertEqual([s['key'] for s in first['sections']], ['mSummary'], 'only the summary while building')
        self.assertEqual((first['build']['state'], first['build']['button'], first['build']['disabled']), ('building', 'Building\u2026', True))
        self.assertEqual(first['build']['status'], 'Reading the match: 12 of 33 segments')
        self.assertEqual(first['build']['bar'], {'hidden': False, 'value': 12, 'max': 33})
        self.assertIn('Read for Adult', first['profile'], 'the profile it is built for')
        asked = lambda o: sum(1 for a in o['asked'] if a.startswith('/api/coaching/moments?'))
        self.assertEqual(asked(out['polled']), asked(first) + 1, 'asked again')
        self.assertEqual([s['key'] for s in out['last']['sections']], SECTIONS)

    def test_missing_then_generate(self):
        missing = {'state': 'missing', 'error': None, 'profile': PROFILE}
        out = run(mjob(moments={'A': [missing, ready(REPORT_A)]},
                       build={'A': [{'state': 'building', 'progress': {'stage': 'starting'}, 'profile': PROFILE}]},
                       hold=True, steps=[{'do': 'click', 'data': {'km': 'generate'}, 'tag': 'BUTTON', 'name': 'posted'}, {'do': 'release', 'hold': False}]))[0]
        first = out['first']
        self.assertEqual((first['build']['state'], first['build']['button'], first['build']['disabled'], first['build']['primary']),
                         ('missing', 'Generate key moments', False, True))
        self.assertIn('Not built yet for this team and level', first['build']['status'])
        self.assertEqual(out['posted']['posted'], [{'url': '/api/coaching/moments/build', 'body': {'run': RUN, 'team': 'A'}}])
        self.assertEqual((out['posted']['build']['state'], out['posted']['build']['status']), ('building', 'Starting\u2026'))
        self.assertEqual([s['key'] for s in out['last']['sections']], SECTIONS)

    def test_generate_waits_while_another_build_runs(self):
        missing = {'state': 'missing', 'error': None, 'profile': PROFILE}
        busy = {'state': 'building', 'progress': None, 'busy': True}
        out = run(mjob(moments={'A': [missing, ready(REPORT_A)]}, build={'A': [busy, {'state': 'building', 'progress': {'stage': 'starting'}}]},
                       hold=True, steps=[{'do': 'click', 'data': {'km': 'generate'}, 'tag': 'BUTTON', 'name': 'queued'}, {'do': 'release', 'hold': True, 'name': 'again'},
                                         {'do': 'release', 'hold': False}]))[0]
        q = out['queued']['build']
        self.assertEqual((q['state'], q['button'], q['disabled']), ('queued', 'Waiting\u2026', True))
        self.assertIn('Another key-moments report is being built', q['status'])
        self.assertEqual(len([p for p in out['again']['posted'] if p['url'] == '/api/coaching/moments/build']), 2, 'posted again')
        self.assertEqual([s['key'] for s in out['last']['sections']], SECTIONS)

    def test_a_failed_build_is_said(self):
        failed = {'state': 'missing', 'error': 'The key-moments report could not be built: boom', 'profile': PROFILE}
        out = run(mjob(moments={'A': [failed]}))[0]
        self.assertEqual(out['last']['build']['status'], 'The key-moments report could not be built: boom')
        self.assertTrue(out['last']['build']['error'])
        self.assertEqual(out['last']['build']['button'], 'Generate key moments')
        err = run(mjob(moments={'A': [{'__status': 500, '__error': 'HTTP 500'}]}))[0]['last']
        self.assertIn('The key moments could not be read (HTTP 500)', err['build']['status'])
        self.assertEqual(err['mView'], 'moments', 'a failure is not a missing route')

    def test_the_v2_report_stays_reachable_and_leads_back(self):
        v2 = v2_answer(v2_base(), run=RUN)
        out = run(mjob(v2=[v2], steps=[{'do': 'click', 'data': {'km': 'v2'}, 'tag': 'BUTTON', 'name': 'v2'},
                                       {'do': 'click', 'data': {'km': 'moments'}, 'tag': 'BUTTON', 'name': 'back'}]))[0]
        v = out['v2']
        self.assertTrue(any(a.startswith('/api/coaching?') and f'run={RUN}' in a for a in v['asked']), 'the v2 report of the match is asked')
        self.assertEqual([s['key'] for s in v['sections']], V2_SECTIONS, 'the v2 renderer, unchanged')
        self.assertEqual(len(v['back']), 1)
        self.assertIn('Key moments of the match', v['back'][0])
        self.assertFalse(v['barHidden'])
        self.assertEqual(v['heading'][0], 'Tactical observations from this run')
        self.assertNotIn('km-one', v['grid'])
        b = out['back']
        self.assertEqual([s['key'] for s in b['sections']], SECTIONS)
        self.assertTrue(b['barHidden'])

    def test_a_segment_run_never_asks_for_key_moments(self):
        out = run(mjob(run='20261026-101010-aaaaaa', v2=[v2_answer(v2_base())]))[0]['last']
        self.assertFalse(any('/api/coaching/moments' in a for a in out['asked']))
        self.assertEqual([s['key'] for s in out['sections']], V2_SECTIONS)
        self.assertEqual(out['back'], [])
        self.assertFalse(out['barHidden'])

    def test_a_server_without_the_route_keeps_the_v2_report(self):
        out = run(mjob(moments={'A': [{'__status': 404, '__error': 'Not available for a full match yet.'}]}, v2=[v2_answer(v2_base(), run=RUN)]))[0]['last']
        self.assertTrue(out['gone'])
        self.assertEqual([s['key'] for s in out['sections']], V2_SECTIONS)
        self.assertEqual(out['back'], [], 'no way back to a report the server cannot make')

    def test_more_moments_open_on_request_and_mark_uncertain_context(self):
        out = run(mjob(steps=[{'do': 'click', 'data': {'km': 'more'}, 'tag': 'BUTTON', 'name': 'open'},
                              {'do': 'click', 'data': {'details': REPORT_A['moreMoments'][0]}, 'tag': 'BUTTON', 'name': 'details'}]))[0]
        o, r = out['open'], REPORT_A
        self.assertEqual(o['more'], {'hidden': False, 'cards': len(r['moreMoments'])})
        more = [c for c in o['cards'] if c['from'] == 'more']
        self.assertEqual([c['moment'] for c in more], r['moreMoments'])
        for c in more:
            m = r['moments'][c['moment']]
            self.assertIn('km-compact', c['cls'])
            self.assertEqual(c['uncertain'], 1 if m['uncertainContext'] else 0)
            self.assertEqual(c['diagrams'], 0, 'a compact card draws its diagram when opened')
        self.assertTrue(any(c['uncertain'] for c in more))
        d = card(out['details'], r['moreMoments'][0], 'more')
        self.assertFalse(d['details']['hidden'])
        self.assertEqual(d['diagrams'], 1)
        m = r['moments'][r['moreMoments'][0]]
        for part in ('What happened.', shown(m['what']), 'Data reliability.', m['reliability']['text']):
            self.assertIn(part, d['details']['text'])


# ---------- the Goals section (soccer/moments/goals.py: report.goals) ----------
GOALS_NOTE = ('Goals come from the match state (a kick-off after a goal, law 10). The build-up is measured from the tracking at five samples a second; '
              'who had the ball is PitchIQ' + chr(0x2019) + 's possession estimate. There is no ball height and no pass event, so no scorer or assist is named.')


def goal(gid, t, team, kind, watch, findings, notes=(), restart=None, uncertain=False, status='detected', confidence='high', start=None, start_kind='turnover'):
    return {'id': gid, 't': t, 'team': team, 'conceding': 'B' if team == 'A' else 'A', 'confidence': confidence, 'timeUncertain': uncertain, 'status': status,
            'kind': kind, 'start': start, 'startKind': start_kind, 'restart': restart, 'watch': {'t0': watch[0], 't1': watch[1]}, 'quality': None, 'facts': {},
            'findings': [{'text': x, 't': at, 'source': source, 'sure': sure} for x, at, source, sure in findings], 'notes': list(notes)}


# the four goals of CAN-ISL as the backend analysed them for Team A (report-A-*.json, cut, 2026-11-01), the first with one
# more finding of a key moment that names a player by his track id; a goal whose making the tracking did not see (KINDS)
UNSEEN = 'The tracking placed both teams on 0% of the 30 s before this goal: too little to analyse the build-up.'
G1 = goal('goal-508.0', 507.99, 'A', 'open_play', (499.75, 510.17), [
    ('Won the ball in the middle third (centre), about 52 m from goal and scored 5 s later.', 502.75, 'possession estimate', 'estimated'),
    ('Moved the ball about 36 m towards goal in 5 s.', 502.75, 'tracking', 'measured'),
    ('Struck towards goal with no Team B (red) outfield player within 5 m of the ball.', 507.39, 'tracking', 'estimated'),
    ('A-62: ran in behind the defence (key moment at 08:24).', 504.6, 'key moment', 'plausible')], start=502.75)
G2 = goal('goal-1275.9', 1275.91, 'A', 'open_play', (1269.87, 1278.16), [
    ('Won the ball in the attacking third (right), about 34 m from goal and scored 3 s later.', 1272.87, 'possession estimate', 'estimated'),
    ('Moved the ball about 22 m towards goal in 3 s.', 1272.87, 'tracking', 'measured'),
    ('On the ball in the build-up (estimated): A-02?.', 1272.87, 'possession estimate', 'estimated')], start=1272.87)
G3 = goal('goal-4180.6', 4180.58, 'B', 'penalty', (4172.38, 4182.58), [
    ('Gave away a penalty at 67:39: red B-01? had the ball in the penalty area, about 5 m from goal when the attack stopped. Watch what happened '
     'there.', 4059.54, 'possession estimate', 'estimated')],
    restart={'t': 4180.58, 'type': 'PENALTY_KICK', 'team': 'B', 'how': 'layout'}, confidence='medium', start=4175.38, start_kind='restart')
G4 = goal('goal-4684.8', 4684.78, 'B', 'penalty', (4680.58, 4686.78), [
    ('Gave away a penalty at 76:30: red B-01? had the ball in the penalty area, about 13 m from goal when the attack stopped. Watch what happened '
     'there.', 4590.9, 'possession estimate', 'estimated')],
    restart={'t': 4684.78, 'type': 'PENALTY_KICK', 'team': 'B', 'how': 'layout'}, start=4683.58, start_kind='restart')
GOALS_A = {'scored': [G1, G2], 'conceded': [G3, G4], 'note': GOALS_NOTE}
GOALS_B = {'scored': [G3, G4], 'conceded': [G1, G2], 'note': GOALS_NOTE}
KINDS = {'scored': [goal('goal-600.0', 600.0, 'A', 'set_piece', (590.0, 602.0), [], restart={'t': 595.0, 'type': 'CORNER_KICK', 'team': 'A'}),
                    goal('goal-1200.0', 1200.0, 'A', 'penalty', (1190.0, 1202.0), [], restart={'t': 1197.0, 'type': 'PENALTY_KICK', 'team': 'A'}),
                    goal('goal-1800.0', 1800.0, 'A', 'kick_off', (1790.0, 1802.0), [], restart={'t': 1785.0, 'type': 'KICK_OFF', 'team': 'A'})],
         'conceded': [goal('goal-2400.0', 2400.0, 'B', 'unknown', (2390.0, 2402.0), [], notes=[UNSEEN], uncertain=True, confidence='medium'),
                      goal('goal-3000.0', 3000.0, 'B', 'set_piece', (2990.0, 3002.0), [], restart={'t': 2994.0, 'type': 'FREE_KICK', 'team': 'B'}, status='candidate')],
         'note': GOALS_NOTE}


def with_goals(report, goals):
    r = copy.deepcopy(report)
    r['goals'] = copy.deepcopy(goals)
    return r


def goals_job(goals_a, goals_b=None, **kw):
    return mjob(moments={'A': [ready(with_goals(REPORT_A, goals_a))], 'B': [ready(with_goals(REPORT_B, goals_b))]}, **kw)


@unittest.skipUnless(NODE, 'needs node to run the page scripts')
class GoalsSection(unittest.TestCase):
    """The key-moments report's Goals section: the report team's goals scored (what went well) and conceded (what went
    wrong), each with the score after it, how it came about, its findings with their sources and times, its notes and
    WATCH; and the report without goals (an older one), before the match state, with none recognised."""

    @classmethod
    def setUpClass(cls):
        # (a 'method' step's snapshot is kept under the method's name)
        shown_ = goals_job(GOALS_A, GOALS_B, steps=[
            {'do': 'click', 'data': {'goal': 'goal-508.0'}, 'tag': 'BUTTON', 'within': 'mGoals', 'name': 'watched'},
            {'do': 'method', 'name': 'momentNav', 'args': [1]},
            {'do': 'method', 'name': 'momentNav', 'args': [1]},
            {'do': 'click', 'data': {'seek': '504.6'}, 'tag': 'BUTTON', 'within': 'mGoals', 'name': 'seeked'},
            {'do': 'method', 'name': 'momentNav', 'args': [1]},
            {'do': 'method', 'name': 'revealMoment', 'args': ['goal-508.0']},
            {'do': 'click', 'data': {'team': 'B'}, 'tag': 'BUTTON', 'name': 'b'}])
        fallback = goals_job(GOALS_A, GOALS_B, noWatch=True, steps=[
            {'do': 'click', 'data': {'goal': 'goal-508.0'}, 'tag': 'BUTTON', 'within': 'mGoals'},
            {'do': 'click', 'data': {'seek': '502.75'}, 'tag': 'BUTTON', 'within': 'mGoals'}])
        cls.out, cls.fallback, cls.older, cls.none, cls.empty, cls.kinds = run(
            shown_, fallback, mjob(), goals_job(None), goals_job({'scored': [], 'conceded': [], 'note': GOALS_NOTE}), goals_job(KINDS))

    def goal_card(self, out, gid):
        return next(g for g in out['goals'] if g['goal'] == gid)

    def test_the_section_follows_the_summary(self):
        o = self.out['first']
        self.assertEqual([s['key'] for s in o['sections']], ['mSummary', 'mGoals'] + SECTIONS[1:])
        g = sec(o, 'mGoals')
        self.assertTrue(g.startswith(f'Goals {DOT} 4'))
        self.assertIn(GOALS_NOTE, g, 'the section note once, at the top')
        self.assertEqual(g.count(GOALS_NOTE), 1)
        self.assertEqual(o['goalGroups'], [
            {'group': 'scored', 'head': f'Goals scored {DOT} 2', 'intro': 'What went well: how the team made each goal.', 'cls': 'cr-intro'},
            {'group': 'conceded', 'head': f'Goals conceded {DOT} 2', 'intro': 'What went wrong: why each goal was allowed.', 'cls': 'cr-intro'}])
        self.assertIn('4 goals', o['counts'], 'a goals chip in the summary')

    def test_each_goal_card(self):
        o = self.out['first']
        self.assertEqual([(g['goal'], g['group']) for g in o['goals']], [('goal-508.0', 'scored'), ('goal-1275.9', 'scored'), ('goal-4180.6', 'conceded'), ('goal-4684.8', 'conceded')])
        want = [('Scored', 'k-strength', '08:28', '1-0', 'Team A (white) scored from open play'),     # rounded, as the timeline says it
                ('Scored', 'k-strength', '21:16', '2-0', 'Team A (white) scored from open play'),
                ('Conceded', 'k-improvement', '69:41', '2-1', 'Team B (red) scored from a penalty'),
                ('Conceded', 'k-improvement', '78:05', '2-2', 'Team B (red) scored from a penalty')]
        for g, (badge, kind, when, score, title) in zip(o['goals'], want):
            with self.subTest(goal=g['goal']):
                self.assertEqual((g['badge'], g['when'], g['score'], g['title']), (badge, when, score, title))
                self.assertIn(kind, g['cls'].split())
                self.assertTrue(g['labelled'], 'the card is named by its title')
                self.assertEqual(g['watch'], ['Watch goal'])
        self.assertEqual(self.goal_card(o, 'goal-508.0')['scoreTitle'], 'After this goal: Team A (white) 1, Team B (red) 0')
        self.assertEqual(self.goal_card(o, 'goal-4684.8')['tags'], [])
        self.assertEqual(self.goal_card(self.kinds['first'], 'goal-2400.0')['tags'], ['Time uncertain'])
        self.assertEqual(self.goal_card(o, 'goal-508.0')['tags'], [])

    def test_findings_chips_and_time_links(self):
        g = self.goal_card(self.out['first'], 'goal-508.0')
        self.assertEqual([f['text'] for f in g['findings']], [shown(f['text']) for f in G1['findings']])
        self.assertEqual([f['chip'] for f in g['findings']], [f'possession estimate {DOT} estimated', f'tracking {DOT} measured', f'tracking {DOT} estimated',
                                                              f'key moment {DOT} plausible'])
        self.assertEqual([f['tone'].split()[-1] for f in g['findings']], ['s-est', 's-ok', 's-est', 's-est'], 'measured reads surer than estimated')
        self.assertEqual([(f['time'], f['seek']) for f in g['findings']], [('08:22', '502.75'), ('08:22', '502.75'), ('08:27', '507.39'), ('08:24', '504.6')])
        last = g['findings'][3]
        self.assertTrue(last['text'].startswith(f"{LABELS['A-62']}: ran in behind"), 'a track id shown as the display label')
        self.assertEqual(last['ids'], [[LABELS['A-62'], f"{LABELS['A-62']} (internal id A-62)"]])
        self.assertEqual(g['notes'], [])

    def test_a_goal_without_findings_shows_its_notes(self):
        unseen = self.goal_card(self.kinds['first'], 'goal-2400.0')
        self.assertEqual((unseen['findings'], unseen['notes'], unseen['none']), ([], [UNSEEN], []))
        g3 = self.goal_card(self.out['first'], 'goal-4180.6')
        self.assertEqual((len(g3['findings']), g3['notes']), (1, []))
        self.assertEqual((g3['findings'][0]['chip'], g3['findings'][0]['time']), (f'possession estimate {DOT} estimated', '67:39'),
                         'the play that won the penalty, two minutes before the kick')

    def test_watch_plays_the_build_up_on_the_moment_player(self):
        w = self.out['watched']['watches'][0]
        self.assertEqual((w['moment'], w['start'], w['seek'], w['time'], w['view']), ('goal-508.0', 'leadup', 507.99, 507.99, 'tracking'))
        self.assertEqual(w['window'], {'t0': 499.75, 't1': 510.17, 'key': 507.99}, 'watch.t0 to watch.t1, paused on the goal')
        self.assertEqual((w['focus'], w['follow'], w['diagram'], w['overlay'], w['tier']), ([], None, False, 0, None), 'no episode: nothing to draw')
        c = w['chip']
        self.assertEqual(c['title'], f'Goal {DOT} Team A (white) {DOT} 08:28')
        self.assertEqual((c['badge'], c['kind'], c['when'], c['counter'], c['prev'], c['next']), ('Scored', 'positive', f'Score 1-0 {DOT} Open play', '1 of 2', False, True))
        self.assertEqual(c['finding'], G1['findings'][0]['text'])
        self.assertEqual(c['details'][0], ['08:22.', f"{G1['findings'][0]['text']} (possession estimate, estimated)"])
        self.assertEqual(c['details'][3][1], f"{shown(G1['findings'][3]['text'])} (key moment, plausible)", 'display labels in the chip')
        self.assertIn('selected', self.goal_card(self.out['watched'], 'goal-508.0')['cls'])
        later = self.out['last']['watches']
        self.assertEqual([(x['moment'], x['start']) for x in later], [('goal-508.0', 'leadup'), ('goal-1275.9', 'leadup'), ('goal-508.0', 'key')],
                         'Next along the goals scored, nothing past the last; then a finding of the first goal')
        self.assertEqual((later[1]['chip']['counter'], later[1]['chip']['prev'], later[1]['chip']['next'], later[1]['chip']['when']),
                         ('2 of 2', True, False, f'Score 2-0 {DOT} Open play'))

    def test_a_finding_time_pauses_the_video_there(self):
        w = self.out['seeked']['watches'][-1]
        self.assertEqual((w['moment'], w['start'], w['seek'], w['time']), ('goal-508.0', 'key', 504.6, 504.6))
        self.assertEqual(w['window'], {'t0': 499.75, 't1': 510.17, 'key': 504.6}, 'in the goal\'s window: Continue plays on to the goal')
        c = w['chip']
        self.assertEqual((c['finding'], c['when'], c['counter'], c['prev'], c['next']), (shown(G1['findings'][3]['text']), f'08:24 {DOT} key moment, plausible', '', False, False))
        self.assertEqual(len(self.out['last']['watches']), 3, 'Previous / Next do nothing from a finding')
        self.assertEqual(self.out['last']['jumps'], 0)

    def test_without_the_moment_player_the_v2_jump(self):
        j = self.fallback['last']['jumpList']
        self.assertEqual(len(j), 2)
        self.assertEqual((j[0]['replay'], j[0]['seek'], j[0]['window']['t0'], j[0]['window']['t1']), (True, 507.99, 499.75, 510.17), 'replayed from watch.t0')
        self.assertEqual(j[0]['caption'], f'Goal {DOT} Team A (white) {DOT} 08:28')
        self.assertEqual((j[1]['replay'], j[1]['seek']), (False, 502.75), 'a finding: the video at its time')
        self.assertEqual(self.fallback['last']['watches'], [])

    def test_back_from_the_player_and_the_other_team(self):
        self.assertIn(['ARTICLE', 'center'], self.out['revealMoment']['scrolled'], 'Report brings the goal card into view')
        self.assertIn('selected', self.goal_card(self.out['revealMoment'], 'goal-508.0')['cls'])
        b = self.out['b']
        self.assertEqual([(g['goal'], g['group'], g['score']) for g in b['goals']],
                         [('goal-4180.6', 'scored', '1-2'), ('goal-4684.8', 'scored', '2-2'), ('goal-508.0', 'conceded', '0-1'), ('goal-1275.9', 'conceded', '0-2')],
                         'the score after each goal, the report team first')
        self.assertEqual(self.goal_card(b, 'goal-508.0')['scoreTitle'], 'After this goal: Team B (red) 0, Team A (white) 1')
        self.assertEqual(self.goal_card(b, 'goal-4180.6')['badge'], 'Scored')

    def test_kinds_in_words_and_a_candidate(self):
        k = self.kinds['first']
        self.assertEqual([g['title'] for g in k['goals']], ['Team A (white) scored from a set piece (corner kick)', 'Team A (white) scored from a penalty',
                                                            'Team A (white) scored from a kick-off', 'Team B (red) scored',
                                                            'Team B (red) scored from a set piece (free kick)'])
        self.assertEqual(self.goal_card(k, 'goal-3000.0')['tags'], ['To review'])
        self.assertEqual(self.goal_card(k, 'goal-600.0')['none'], ['PitchIQ could not say anything for sure about how this goal came about.'])

    def test_an_older_report_no_match_state_and_no_goals(self):
        self.assertNotIn('mGoals', [s['key'] for s in self.older['last']['sections']], 'a report made before goals were analysed: no section')
        self.assertNotIn('goal', self.older['last']['counts'])
        n = self.none['last']
        self.assertEqual([s['key'] for s in n['sections']], ['mSummary', 'mGoals'] + SECTIONS[1:])
        self.assertEqual(sec(n, 'mGoals'), 'GoalsGoals appear here once the match state is built.')
        self.assertNotIn('goal', n['counts'])
        e = self.empty['last']
        self.assertEqual(sec(e, 'mGoals'), f'Goals {DOT} 0{GOALS_NOTE}No goals were recognised in this match.')
        self.assertIn('0 goals', e['counts'])
        self.assertEqual(e['goals'], [])

    def test_the_styles(self):
        p = page()
        for rule in ('.km-goals>.cr-sub-h{', '.km-goal .km-score{', '.km-gf{list-style:none;', '.km-gf-row{display:flex;', '.km-gf .km-gf-t{',
                     '.km-gf-src{', '.km-gf-src.s-ok{', '.km-gf-src.s-est{', '.km-gf-src.s-low{', '.cr-sec p.km-goal-note{'):
            self.assertIn(rule, p, rule)
        css = p[p.index('.km-goals>.cr-sub-h{'):p.index('.cr-sec p.km-goal-note{')]
        self.assertEqual([c for c in css if ord(c) > 127], [], 'ASCII')


# ---------- the page: the moment player in index.html ----------
PLAYER = r"""
const out = {log: []};
const FPS = 25, frameOf = (f) => ({frame: f, time: f / FPS, people: [{id: 'A-62', box: [.4, .5, .02, .08]}, {id: 'A-63', box: [.3, .5, .02, .08]}]});
const viewOf = (a, b) => ({frames: Array.from({length: b - a}, (_, k) => frameOf(a + k))});
let result = viewOf(VIEW0), info = {video: {fps: FPS}};
let mode = null, editor = null; const shown = [], seeks = [], selected = [], drawn = [], listeners = {};
let toggles = 0, invalidated = 0;
const closeEditor = () => {}, views = {show: (v) => shown.push(v)}, select = (id) => selected.push(id);
const video = {paused: true, currentTime: 0, pause() { this.paused = true; }, play() { this.paused = false; }, addEventListener: (t, f) => { listeners[t] = f; }};
const playback = {seek: (t) => { seeks.push(Math.round(t * 1e4) / 1e4); video.currentTime = t; }, toggle: () => { toggles++; if (video.paused) video.play(); else video.pause(); }};
class El {
  constructor(id) { this.id = id; this.hidden = true; this.own = ''; this.kids = []; this.title = ''; this.disabled = false; this.attrs = {}; this.style = {}; this.className = '';
    const set = new Set(); this.classList = {add: (...c) => c.forEach((x) => set.add(x)), remove: (...c) => c.forEach((x) => set.delete(x)), toggle: (c, on) => { if (on === undefined ? !set.has(c) : on) set.add(c); else set.delete(c); }, has: (c) => set.has(c), contains: (c) => set.has(c), list: () => [...set]}; }
  get textContent() { return this.own + this.kids.map((k) => k.textContent).join(''); }
  set textContent(v) { this.own = String(v); this.kids = []; }
  append(...cs) { this.kids.push(...cs); }
  setAttribute(k, v) { this.attrs[k] = String(v); }
  getContext() { return {clearRect() {}}; }
  scrollIntoView() {}
  getBoundingClientRect() { return {left: 0, top: 0, right: 1000, bottom: 500, width: 1000, height: 500}; }
  get offsetHeight() { return 40; }
}
const els = {}, $ = (id) => els[id] || (els[id] = new El(id));
const document = {querySelector: (s) => s === '.app-header' ? {offsetHeight: 60} : null, addEventListener() {}, createElement: (t) => new El(t), createTextNode: (t) => ({textContent: String(t)})};
const canvas = {width: 1000, height: 500};
const tactical = {overlays: [], colours: {A: '#b98cff', B: '#f6b564'}, invalidate: () => { invalidated++; }};
const COLORS = {A: '#b98cff', B: '#f6b564'}, nav = [], revealed = [], viewsChosen = [];
const coaching = {labels: new Map(), momentNav: (d) => nav.push(d), revealMoment: (id) => revealed.push(id), setMomentView: (v) => viewsChosen.push(v)};
const Coaching = {drawVideo: (c, spec) => drawn.push(['video', spec.tag]), drawBoard: (c, g, spec) => drawn.push(['board', spec.tag]),
                  drawFollow: (c, follow, o) => drawn.push(['follow', o.mode, o.mode === 'video' ? [...o.people.keys()].join(',') : 'state'])};
const timestamp = (t) => `${String(Math.floor(t / 60)).padStart(2, '0')}:${(t % 60).toFixed(2).padStart(5, '0')}`;
let waiting = false;
const FullMatch = {waiting: () => waiting}, window = {matchMode: MATCH ? {} : null, FullMatch};
FRAMEINDEX
BLOCK
const draw = () => { const i = frameIndex(video.currentTime); drawCoachMark(i); return i; };
// the clock: plays in steps of one frame while not paused, a render after each step
const play = (seconds) => { for (let k = 0; k < Math.round(seconds * FPS) && !video.paused; k++) { video.currentTime += 1 / FPS; draw(); } };
const chip = () => ({hidden: $('coachChip').hidden, moment: $('coachChipMoment').hidden === false, text: $('coachChipText').textContent, badge: $('coachChipBadge').textContent,
                     when: $('coachChipWhen').textContent, phase: $('coachChipPhase').textContent, finding: $('coachChipFinding').textContent, note: $('coachChipNote').textContent,
                     why: !$('coachChipWhy').hidden, more: !$('coachChipMore').hidden, details: $('coachChipMore').textContent,
                     buttons: Object.fromEntries(['Replay', 'Continue', 'View', 'Prev', 'Next'].map((b) => [b, {disabled: $('coachChip' + b).disabled, text: $('coachChip' + b).textContent}])),
                     cls: $('coachChip').classList.list().sort()});
SCRIPT
process.stdout.write(JSON.stringify(out));
"""

EV = {'moment': 'mo-1', 'start': 'leadup', 'frame': 450, 'seek': (450 + .5)/25, 'time': 450/25, 'view': 'tracking', 'focus': ['A-62', 'A-63'],
      'overlay': {'tag': 'o', 'width': 1920, 'height': 1080, 'elements': []}, 'diagram': {'tag': 'd', 'elements': []}, 'tier': 'world',
      'window': {'t0': 444/25, 't1': 460/25, 'key': 450/25}, 'follow': {'focus': [{'id': 'A-62'}], 'support': [{'id': 'A-63'}], 'relations': []},
      'chip': {'title': 'Opponent free between defence and midfield', 'finding': 'Red B-03 was free between the lines.', 'badge': 'Review', 'kind': 'improvement',
               'when': '00:18 \u00b7 first half', 'counter': '1 of 5', 'prev': False, 'next': True, 'note': '',
               'details': [['What happened.', 'At 00:18, red B-03 stood unmarked.'], ['Coaching pointer.', 'Compact the vertical distance.']]}}


def player(script, view0=(400, 520), match=False):
    src = page_script(page())
    block = span(src, 'let coachMark=null;', COACH_BLOCK_END)
    text = (PLAYER.replace('BLOCK', block).replace('FRAMEINDEX', js_function(src, 'frameIndex')).replace('VIEW0', f'{view0[0]}, {view0[1]}')
            .replace('MATCH', 'true' if match else 'false').replace('SCRIPT', script).replace('EVJSON', json.dumps(EV)))
    done = subprocess.run([NODE, '-e', text], capture_output=True, text=True, encoding='utf-8', timeout=120)
    if done.returncode:
        raise AssertionError(done.stderr[-3000:])
    return json.loads(done.stdout)


@unittest.skipUnless(NODE, 'needs node to run the page script')
class MomentPlayer(unittest.TestCase):
    """index.html: the moment player on the page's one clock (the coach-mark block run in node)."""

    def test_markup_and_wiring(self):
        p, src = page(), page_script(page())
        self.assertIn("jump:ev=>jumpToEvidence(ev),watch:ev=>watchMoment(ev),heading:{title:$('crTitle'),lede:$('crLede')},marks:{", src)
        self.assertIn('<h2 id="crTitle">Tactical observations from this run</h2><p class="cr-lede" id="crLede">', p)
        chip = span(p, '<div class="coach-chip" id="coachChip"', '<p class="coach-chip-more" id="coachChipMore" hidden></p></div>')
        for control in ('coachChipMoment', 'coachChipBadge', 'coachChipWhen', 'coachChipPhase', 'coachChipFinding', 'coachChipNote', 'coachChipReplay',
                        'coachChipContinue', 'coachChipView', 'coachChipPrev', 'coachChipNext'):
            self.assertIn(f'id="{control}"', chip, control)
        self.assertLess(chip.index('id="coachChipMoment"'), chip.index('id="coachChipMore"'), 'the explanation stays last')
        for rule in ('.cr-bar[hidden]', '.cr-grid.km-one{grid-template-columns:minmax(0,1fr)}', '.coach-chip .ccm-ctl button{width:auto;', '.km-tl .km-mark{'):
            self.assertIn(rule, p, rule)
        self.assertNotRegex(p, r'innerHTML')
        with open(FILES['coaching.js'], encoding='utf-8') as handle:
            js = handle.read()
        self.assertNotIn('innerHTML', js)
        new = js[js.index('// ---------- KEY MOMENTS:'):js.index('// ---------- the view ----------')] + js[js.index('// ================= KEY MOMENTS'):js.index('  placeSide() {')]
        self.assertEqual([c for c in new if ord(c) > 127], [], 'the new source is ASCII (typographic characters as \\u escapes)')
        for name in ('index.html', 'coaching.js', 'match.js'):
            with open(os.path.join(ROOT, name), 'rb') as handle:
                data = handle.read()
            self.assertEqual(data.count(b'\n'), data.count(b'\r\n'), f'{name} keeps CRLF')

    def test_lead_up_pause_on_the_key_frame_then_continue(self):
        out = player(r"""
const ev = EVJSON;
out.ok = watchMoment(ev);
out.start = {shown: shown.slice(), seeks: seeks.slice(), selected: selected.slice(), paused: video.paused, chip: chip()};
draw(); out.started = {toggles, paused: video.paused, phase: chip().phase};
play(2); out.leadup = drawn.slice(); drawn.length = 0;
play(10); out.key = {paused: video.paused, seek: seeks[seeks.length - 1], chip: chip()};
const k = draw(); out.keyIndex = k; out.keyDrawn = drawn.slice(); drawn.length = 0;
tactical.overlays[0](null, null, {frame: k, time: result.frames[k].time}); out.board = drawn.slice(); drawn.length = 0;
$('coachChipContinue').onclick(); out.cont = {paused: video.paused, toggles, chip: chip()};
play(20); out.end = {paused: video.paused, time: Math.round(video.currentTime * 1e4) / 1e4, chip: chip(), mark: !!coachMark};
$('coachChipReplay').onclick(); out.replay = {seek: seeks[seeks.length - 1], chip: chip().phase, paused: video.paused};
draw(); out.replayed = {paused: video.paused, phase: chip().phase};
""")
        self.assertTrue(out['ok'])
        s = out['start']
        self.assertEqual(s['shown'], ['tracking'])
        self.assertEqual(s['seeks'], [round(444/25, 4)], 'seeks to the window start')
        self.assertEqual((s['selected'], s['paused']), (['A-62'], True))
        c = s['chip']
        self.assertEqual((c['hidden'], c['moment'], c['text'], c['badge'], c['finding']), (False, True, EV['chip']['title'], 'Review', EV['chip']['finding']))
        self.assertEqual(c['when'], f"00:18 {DOT} first half {DOT} 1 of 5")
        self.assertEqual(c['buttons']['Prev']['disabled'], True)
        self.assertEqual(c['buttons']['Next']['disabled'], False)
        self.assertEqual(c['buttons']['View']['text'], '2D')
        self.assertTrue(c['why'])
        self.assertFalse(c['more'], 'the explanation waits until the key frame')
        self.assertIn('moment', c['cls'])
        self.assertEqual(out['started'], {'toggles': 1, 'paused': False, 'phase': 'Lead-up from 00:17.76'})
        self.assertTrue(out['leadup'])
        self.assertTrue(all(d == ['follow', 'video', 'A-62,A-63'] for d in out['leadup']), 'the lead-up follows the people at their live boxes')
        k = out['key']
        self.assertTrue(k['paused'])
        self.assertAlmostEqual(k['seek'], round(450.5/25, 4), places=4, msg="paused on the key frame's centre")
        self.assertEqual(k['chip']['phase'], 'Paused on the key moment')
        self.assertTrue(k['chip']['more'], 'the explanation is opened on the key frame')
        self.assertIn('What happened. At 00:18, red B-03 stood unmarked.', k['chip']['details'])
        self.assertEqual(out['keyIndex'], 50)
        self.assertEqual(out['keyDrawn'], [['video', 'o']], 'the whole overlay on the key frame')
        self.assertEqual(out['board'], [['board', 'd']], 'and the diagram on the board')
        self.assertEqual((out['cont']['paused'], out['cont']['chip']['phase'], out['cont']['chip']['buttons']['Continue']['disabled']), (False, 'Follow-up', True))
        e = out['end']
        self.assertEqual((e['paused'], e['time'], e['chip']['phase'], e['mark']), (True, round(460/25, 4), 'End of the moment', True), 'paused at the window\'s end')
        self.assertTrue(e['chip']['buttons']['Continue']['disabled'])
        self.assertEqual((out['replay']['seek'], out['replay']['paused']), (round(444/25, 4), True))
        self.assertEqual(out['replayed'], {'paused': False, 'phase': 'Lead-up from 00:17.76'}, 'Replay plays the lead-up again')

    def test_video_2d_prev_next_and_back(self):
        out = player(r"""
const ev = EVJSON;
watchMoment(ev); draw(); play(12); draw();
$('coachChipView').onclick(); out.board = {shown: shown.slice(), seek: seeks[seeks.length - 1], view: chip().buttons.View.text, chosen: viewsChosen.slice()};
$('coachChipView').onclick(); out.video = {seek: seeks[seeks.length - 1], view: chip().buttons.View.text};
$('coachChipNext').onclick(); $('coachChipPrev').onclick(); out.nav = nav.slice();
video.play(); $('coachChipBack').onclick(); out.back = {shown: shown[shown.length - 1], paused: video.paused, revealed: revealed.slice()};
""")
        self.assertEqual(out['board']['shown'][-1], 'board')
        self.assertAlmostEqual(out['board']['seek'], round(450/25 + .2/25, 4), places=4, msg='0.2 frame into the key frame on the board')
        self.assertEqual((out['board']['view'], out['board']['chosen']), ('Video', ['board']))
        self.assertAlmostEqual(out['video']['seek'], round(450.5/25, 4), places=4)
        self.assertEqual(out['video']['view'], '2D')
        self.assertEqual(out['nav'], [1, -1])
        self.assertEqual(out['back'], {'shown': 'coaching', 'paused': True, 'revealed': ['mo-1']})

    def test_a_moment_in_a_view_not_loaded_yet(self):
        # match playback: the shown view (frames 100-200) does not have the moment; the player waits for the view owning
        # the window's start, a swap (match.js) keeps the moment, and the key frame is found in the new view
        out = player(r"""
const ev = EVJSON;
waiting = true;
out.ok = watchMoment(ev);
draw(); draw(); out.waiting = {toggles, paused: video.paused, phase: chip().phase, key: coachMark.i, seek: seeks[0]};
result = viewOf(380, 520); out.rebase = coachMarkRebase(); waiting = false;
draw(); out.started = {toggles, paused: video.paused, phase: chip().phase};
play(12); const k = draw(); out.key = {paused: video.paused, index: k, drawn: drawn.filter((d) => d[0] === 'video')};
""", view0=(100, 200), match=True)
        self.assertTrue(out['ok'], 'placed although the frame is not in the view shown')
        self.assertEqual(out['waiting']['toggles'], 0, 'nothing plays before the view owning the time is shown')
        self.assertEqual(out['waiting']['phase'], 'Loading the match data for this moment\u2026')
        self.assertEqual(out['waiting']['key'], -1)
        self.assertAlmostEqual(out['waiting']['seek'], round(444/25, 4), places=4)
        self.assertTrue(out['rebase'], 'a swap keeps a moment')
        self.assertEqual(out['started'], {'toggles': 1, 'paused': False, 'phase': 'Lead-up from 00:17.76'})
        self.assertEqual(out['key']['index'], 70, 'the key frame in the new view')
        self.assertTrue(out['key']['paused'])
        self.assertEqual(out['key']['drawn'], [['video', 'o']])

    def test_the_chip_stands_above_the_picture(self):
        # over the video the opened explanation hid the annotation (1920 x 1080) or most of the picture (the 800 px
        # pane): the moment's chip stands above the picture, in the stage, except in full screen; the explanation opens
        # by itself on the key frame from a 760 px wide picture, on a phone it waits behind Why
        out = player(r"""
const ev = EVJSON;
$('coachChip').parentElement = $('stageWrap');
$('screen').getBoundingClientRect = () => ({left: 0, top: 0, right: 343, bottom: 193, width: 343, height: 193});
watchMoment(ev); draw(); out.phone = chip().cls;
play(12); draw(); out.phoneKey = {phase: chip().phase, more: chip().more, why: chip().why};
$('screen').getBoundingClientRect = () => ({left: 0, top: 0, right: 785, bottom: 441, width: 785, height: 441});
$('coachChipReplay').onclick(); draw(); play(12); draw(); out.desk = {cls: chip().cls, more: chip().more};
$('stageWrap').classList.add('full'); coachChipPlace(); out.full = chip().cls;
$('stageWrap').classList.remove('full'); coachChipPlace(); out.back = chip().cls;
""")
        self.assertIn('flow', out['phone'])
        self.assertEqual(out['phoneKey'], {'phase': 'Paused on the key moment', 'more': False, 'why': True})
        self.assertIn('flow', out['desk']['cls'])
        self.assertTrue(out['desk']['more'])
        self.assertNotIn('flow', out['full'], 'over the video in full screen')
        self.assertIn('flow', out['back'])

    def test_other_marks_are_unchanged(self):
        out = player(r"""
out.rebase = coachMarkRebase();
jumpToEvidence({frame: 450, seek: 450.5 / 25, view: 'tracking', focus: [], overlay: {tag: 'o'}, diagram: null, window: {t0: 447 / 25, t1: 452 / 25}, caption: 'A v2 moment'});
out.v2 = {rebase: coachMarkRebase(), chip: chip()};
""")
        self.assertFalse(out['rebase'])
        self.assertFalse(out['v2']['rebase'], 'a v2 mark is cleared by a swap as before')
        self.assertEqual((out['v2']['chip']['text'], out['v2']['chip']['moment']), ('A v2 moment', False))
        self.assertNotIn('moment', out['v2']['chip']['cls'])

    def test_a_goal_plays_without_an_episode(self):
        # a goal of the key-moments report (coaching.js kmGoalPayload): no frame, diagram, overlay or follow list; its
        # build-up from the window's start, paused on the goal's time with nothing drawn; a finding's time: paused there
        out = player(r"""
const goal = {moment: 'goal-18.0', start: 'leadup', seek: 18, time: 18, fps: 25, view: 'tracking', focus: [], diagram: null, overlay: null, tier: null,
              window: {t0: 444 / 25, t1: 460 / 25, key: 18}, follow: null,
              chip: {title: 'Goal - Team A (white) - 00:18', finding: 'Won the ball and scored 5 s later.', badge: 'Scored', kind: 'positive', when: 'Score 1-0',
                     counter: '1 of 2', prev: false, next: true, note: '', details: [['00:17.', 'Won the ball (possession estimate, estimated)']]}};
out.ok = watchMoment(goal);
out.start = {seeks: seeks.slice(), selected: selected.slice(), chip: chip()};
draw(); play(10);
out.key = {paused: video.paused, seek: seeks[seeks.length - 1], phase: chip().phase, drawn: drawn.slice()};
const k = draw(); tactical.overlays[0](null, null, {frame: k, time: result.frames[k].time}); out.board = drawn.slice();
$('coachChipContinue').onclick(); play(20); out.end = {paused: video.paused, phase: chip().phase, time: Math.round(video.currentTime * 1e4) / 1e4};
const before = toggles, finding = Object.assign({}, goal, {start: 'key', seek: 17.9, time: 17.9, window: {t0: 444 / 25, t1: 460 / 25, key: 17.9}});
out.ok2 = watchMoment(finding); draw();
out.finding = {seek: seeks[seeks.length - 1], paused: video.paused, phase: chip().phase, played: toggles - before};
""")
        self.assertTrue(out['ok'], 'the moment player takes a window without an episode')
        self.assertEqual(out['start']['seeks'], [round(444/25, 4)], 'from the build-up\'s start')
        self.assertEqual(out['start']['selected'], [None], 'nobody selected')
        c = out['start']['chip']
        self.assertEqual((c['text'], c['badge'], c['finding']), ('Goal - Team A (white) - 00:18', 'Scored', 'Won the ball and scored 5 s later.'))
        self.assertIn('k-positive', c['cls'])
        k = out['key']
        self.assertEqual((k['paused'], k['seek'], k['phase'], k['drawn']), (True, 18.0, 'Paused on the key moment', []), 'paused on the goal, nothing to draw')
        self.assertEqual(out['board'], [])
        self.assertEqual(out['end'], {'paused': True, 'phase': 'End of the moment', 'time': round(460/25, 4)})
        self.assertTrue(out['ok2'])
        self.assertEqual(out['finding'], {'seek': 17.9, 'paused': True, 'phase': 'Paused on the key moment', 'played': 0}, 'a finding: the video paused at its time')


# ---------- match.js: the cross-segment jump ----------
@unittest.skipUnless(NODE, 'needs node to run match.js functions')
class MatchJump(unittest.TestCase):
    def test_waiting_and_a_swap_that_keeps_a_moment(self):
        with open(MATCH_JS, encoding='utf-8') as handle:
            src = handle.read()
        fns = '\n'.join(js_function(src, name) for name in ('ownerAt', 'covers', 'fpsOf', 'waiting', 'swap'))
        script = r"""
console.time = console.timeEnd = () => {};   // swap() times itself on the console
let player = null, result = null, presence = null, pitchIndex = null, rebase = true, cleared = 0;
const info = {video: {fps: 25}}, hasVideo = () => true, isNum = (v) => typeof v === 'number' && Number.isFinite(v);
const tactical = {load() {}, timeline: null}, video = {currentTime: 0}, closeEditor = () => {}, clearCoachMark = () => { cleared++; };
const callPage = (name) => name === 'coachMarkRebase' ? rebase : undefined, boardTakesTimeline = () => false, buildPresence = () => {}, pitchIndexOf = () => [];
const idle = () => {}, stageChanged = () => {};
FNS
const segs = [{start: 10, end: 190, firstTime: 7, lastTime: 193}, {start: 190, end: 370, firstTime: 187, lastTime: 373}];
const out = {};
out.none = waiting(100);
player = {segs: null, current: -1, index: {video: {fps: 25}}};
out.opening = waiting(100);
player = {segs, current: 0, owner: 0, index: {video: {fps: 25}}, views: new Map([[0, {frames: []}], [1, {frames: []}]]), prepared: new Map(), swaps: [], ctxGen: 0};
out.shown = waiting(100); out.context = waiting(192); out.other = waiting(250); out.gap = waiting(5);
swap(player, 1, 'seek'); out.kept = cleared;
rebase = false; swap(player, 0, 'seek'); out.cleared = cleared;
process.stdout.write(JSON.stringify(out));
""".replace('FNS', fns)
        done = subprocess.run([NODE, '-e', script], capture_output=True, text=True, encoding='utf-8', timeout=60)
        self.assertEqual(done.returncode, 0, done.stderr[-2000:])
        out = json.loads(done.stdout)
        self.assertEqual((out['none'], out['opening']), (False, True), 'no match: nothing to wait for; a match opening: wait')
        self.assertEqual((out['shown'], out['context'], out['other'], out['gap']), (False, False, True, False),
                         "the shown view's segment or its context frames: no wait; another segment's: wait; a break: no wait")
        self.assertEqual((out['kept'], out['cleared']), (0, 1), 'a swap keeps a moment being watched and clears any other mark')
        self.assertIn("if (callPage('coachMarkRebase') !== true) clearCoachMark();", src)
        self.assertIn('init, open, leave, frame, scope, chip, stage, span, renameView, contextOf, waiting,', src)


if __name__ == '__main__':
    unittest.main()
