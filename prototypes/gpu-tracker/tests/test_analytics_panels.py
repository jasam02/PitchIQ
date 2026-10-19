"""The Analytics panels' registration note: analytics.js and teamshape.js run in node (skipped without node) on the
reports soccer/analytics.py and soccer/team_shape.py make from synthetic results.

Since the analytics count MEDIUM registrations (analytics.MIN_PITCH_CONFIDENCE = pitch_model.MEDIUM), each report says
how much of the selection's observed time rests on a HIGH and on a MEDIUM registration (`quality`; the player report
also `averagesQuality` for the players the average tab draws). When MEDIUM holds more than half of it the panel's
status line, where the low-coverage note is, says the pitch registration places those positions with only medium
confidence and they are less certain; on a HIGH registration it says nothing of the kind. The pages run with the real tactical.js board in
a vm context whose canvases accept every call and whose elements are plain objects.

Run:  python -m unittest test_analytics_panels          (from this directory)
"""
import json
import os
import shutil
import subprocess
import unittest

import scene  # noqa: F401  (adds the prototype directory to sys.path)
from soccer import analytics, team_shape as ts
import team_shape_synth as syn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FILES = {name: os.path.join(ROOT, name) for name in ('tactical.js', 'analytics.js', 'teamshape.js')}
NODE = shutil.which('node')
NOTE = 'the pitch registration places with only medium confidence'

HARNESS = r"""
const fs = require('fs'), vm = require('vm');
const input = JSON.parse(fs.readFileSync(0, 'utf8'));
const context = () => new Proxy({}, {
  get: (o, k) => k === 'measureText' ? (s) => ({width: String(s).length * 6})
    : k === 'createImageData' ? (w, h) => ({data: new Uint8ClampedArray(w * h * 4)}) : () => {},
  set: () => true,
});
function element(tag) {
  return {tagName: tag, children: [], style: {setProperty() {}}, dataset: {}, hidden: false, disabled: false, value: '', textContent: '',
          className: '', width: 0, height: 0, clientWidth: 960, offsetWidth: 960, classList: {toggle() {}, add() {}, remove() {}},
          append(...c) { this.children.push(...c); }, replaceChildren(...c) { this.children = c; }, setAttribute() {}, addEventListener() {},
          getContext: () => context(), getBoundingClientRect: () => ({left: 0, top: 0, width: 960, height: 124})};
}
function select(value) {
  // A <select>: replacing its options keeps the value only when an option still has it, else the first option's.
  const e = element('select');
  e.value = value;
  e.replaceChildren = function (...c) { this.children = c; if (!c.some((o) => o.value === this.value)) this.value = c.length ? c[0].value : ''; };
  return e;
}
const sandbox = {document: {createElement: element, createTextNode: (text) => ({textContent: text})}, console, setTimeout, clearTimeout, URLSearchParams,
                 localStorage: {getItem: () => null, setItem() {}}, devicePixelRatio: 1, addEventListener() {}};
sandbox.window = sandbox;
vm.createContext(sandbox);
for (const path of Object.values(input.files)) vm.runInContext(fs.readFileSync(path, 'utf8'), sandbox, {filename: path});
const settle = async () => { for (let k = 0; k < 20; k++) await new Promise((r) => setTimeout(r, 0)); };
const colours = {A: '#b98cff', B: '#f6b564'};

async function player(job) {
  const tabs = ['heatmap', 'average', 'trail'].map((tab) => Object.assign(element('button'), {dataset: {tab}}));
  const controls = {tabs, team: select(job.team), player: select(job.player), range: select('full'), custom: element('div'),
                    t0: element('input'), t1: element('input'), scope: select(job.scope), scopeWrap: element('div')};
  const status = element('div'), notice = element('div'), asked = [];
  const view = sandbox.Analytics.create({canvas: element('canvas'), notice, status, legend: element('div'), insights: element('div'), controls,
                                        api: async (url) => { asked.push(url); return job.report; }, video: {currentTime: 0}, board: {colours}});
  view.load(job.result, job.info);
  await settle();
  const out = {asked, player: controls.player.value};
  for (const tab of ['heatmap', 'average', 'trail']) { view.setTab(tab); out[tab] = status.textContent; }
  return out;
}

async function shape(job) {
  const tabs = ['width', 'line', 'formation'].map((tsTab) => Object.assign(element('button'), {dataset: {tsTab}}));
  const controls = {tabs, team: select(job.team), mode: select('observed'), range: select('full'), custom: element('div'), t0: element('input'),
                    t1: element('input'), formationView: select('average'), modeWrap: element('div'), formationViewWrap: element('div')};
  const status = element('div'), notice = element('div'), asked = [];
  const api = async (url) => {
    asked.push(url);
    if (url.startsWith('/api/team-shape/snapshot')) return job.snapshot;
    return new URLSearchParams(url.split('?')[1]).get('team') === job.team ? job.report : job.other;
  };
  const view = sandbox.TeamShape.create({canvas: element('canvas'), notice, status, legend: element('div'), insights: element('div'), controls, api,
                                        post: async () => ({}), video: {currentTime: job.at}, board: {colours}, under: element('div'),
                                        chart: element('canvas'), timeline: element('div'), snap: element('input'), snapTime: element('span')});
  view.load(job.result, job.info);
  view.setActive(true);
  await settle();
  const out = {asked};
  for (const tab of ['width', 'line', 'formation']) { view.setTab(tab); await settle(); out[tab] = status.textContent; }
  return out;
}

(async () => {
  const out = [];
  for (const job of input.jobs) out.push(await (job.kind === 'shape' ? shape : player)(job));
  process.stdout.write(JSON.stringify(out));
})().catch((error) => { process.stderr.write(String(error && error.stack || error)); process.exit(1); });
"""


def run(*jobs):
    done = subprocess.run([NODE, '-e', HARNESS], input=json.dumps({'files': FILES, 'jobs': list(jobs)}), capture_output=True, text=True,
                          encoding='utf-8', timeout=300)
    if done.returncode:
        raise AssertionError(done.stderr[-3000:])
    return json.loads(done.stdout)


FPS = 30.0


def player_result(confidences):
    """Four seconds of three players standing apart; their pitch confidence per id."""
    teams = {'B-07': 'B', 'A-02': 'A', 'A-03': 'A'}
    spots = {'B-07': (.3, .4), 'A-02': (.6, .6), 'A-03': (.7, .3)}
    frames = []
    for k in range(int(4*FPS)):
        t = round(k/FPS, 4)
        people = [{'time': t, 'track': n+1, 'box': [.5, .5, .02, .1], 'id': pid, 'display': pid, 'team': team, 'role': 'PLAYER', 'state': 'confirmed',
                   'identityConfidence': .9, 'pitch': list(spots[pid]), 'pitchConfidence': confidences[pid], 'pitchSigma': [.5, .5]}
                  for n, (pid, team) in enumerate(teams.items())]
        frames.append({'time': t, 'frame': k, 'calibrated': True, 'people': people})
    return {'version': 3, 'video': {'width': 1920, 'height': 1080, 'fps': FPS}, 'frames': frames,
            'match': {'players': [{'id': pid, 'display': pid, 'team': team} for pid, team in teams.items()], 'goalkeepers': []},
            'tactical': {'pitchSize': {'length': 105.0, 'width': 68.0}, 'goalEnds': {'left': 'A', 'right': 'B'}}}


def player_job(confidences, scope='both'):
    result = player_result(confidences)
    s = analytics.Samples(result)
    a, b = analytics.clamp_range(s)
    report = json.loads(json.dumps(analytics.report(s, 'B-07', a, b, scope)))
    return {'kind': 'player', 'team': 'B', 'player': 'B-07', 'scope': scope, 'result': result, 'info': {'video': result['video']}, 'report': report}


def shape_job(confidence):
    times, teams, ends = syn.match('4-4-2', '4-3-3', seconds=40, fps=5.0, seed=31)
    result = syn.make_result(times, teams, ends, pitch_confidence=confidence)
    shape = ts.TeamShape(result)
    dump = lambda v: json.loads(json.dumps(v))
    return {'kind': 'shape', 'team': 'A', 'at': 20.0, 'result': result, 'info': {'video': result['video']},
            'report': dump(shape.report('A')), 'other': dump(shape.report('B')), 'snapshot': dump(shape.snapshot('A', 20.0))}


@unittest.skipUnless(NODE, 'needs node to run the page scripts')
class RegistrationNote(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.medium_player, cls.high_player, cls.medium_shape, cls.high_shape, cls.all_medium = run(
            player_job({'B-07': .4, 'A-02': .8, 'A-03': .8}), player_job({'B-07': .8, 'A-02': .8, 'A-03': .8}), shape_job(.4), shape_job(.9),
            player_job({'B-07': .4, 'A-02': .4, 'A-03': .4}))

    def test_player_panel_says_a_medium_registration_is_less_certain(self):
        out = self.medium_player
        self.assertEqual(out['player'], 'B-07')
        self.assertTrue(any(url.startswith('/api/analytics?') for url in out['asked']), out['asked'])
        for tab in ('heatmap', 'trail'):
            self.assertIn(NOTE, out[tab], tab)
            self.assertIn('100%', out[tab], 'the measured share is said')
            self.assertIn('less certain', out[tab])

    def test_average_tab_speaks_for_the_players_it_draws(self):
        # The average tab draws all three players: two of them on a HIGH registration, so MEDIUM is a third of the time.
        self.assertNotIn(NOTE, self.medium_player['average'])
        # All three on a MEDIUM registration: the average tab says so, for the players it draws.
        self.assertIn(NOTE, self.all_medium['average'])
        self.assertIn("Of the drawn players' observed time, 100%", self.all_medium['average'])

    def test_no_note_on_a_high_registration(self):
        for tab in ('heatmap', 'average', 'trail'):
            self.assertNotIn(NOTE, self.high_player[tab], tab)

    def test_team_shape_panel_says_a_medium_registration_is_less_certain(self):
        for tab in ('width', 'line', 'formation'):
            self.assertIn(NOTE, self.medium_shape[tab], tab)
            self.assertIn('100%', self.medium_shape[tab])
            self.assertNotIn(NOTE, self.high_shape[tab], tab)
        self.assertTrue(any(url.startswith('/api/team-shape/snapshot?') for url in self.medium_shape['asked']))


if __name__ == '__main__':
    unittest.main()
