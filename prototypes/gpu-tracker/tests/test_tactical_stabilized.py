"""The 2D match view's stabilized positions: tactical.js run in node (skipped without node).

When the result has a `stabilized` block (soccer/trajectory.py's smoothed observations with imputation's estimates
made on them), the board draws it as given: no second smoothing, the straight line between two samples at the
actual frame times at every display time (no snap to the nearer frame), the interpolated ring decided by the kinds
of the two samples, the anonymous '?' tracks from the block (a stabilizer's bridge on one only within
RULES.anonymousGap), nothing across a gap longer than RULES.interpolateGap, the ball untouched, the status line
counting what is drawn, and the stabilized tracks built only for a view that draws them (the analytics views load a
timeline for the pitch only). With Debug positions on, the developer comparison draws the raw positions, the
stabilized ones or both (raw as crosses, never a ring), and is remembered per viewer. Without the block the view is
the unmodified one, call for call: the states, the canvas calls and the status line of tactical.js at BASELINE, read
from git (those tests are skipped without git or the commit).

The clip is synthetic: three identities, '?' tracks and the ball over 4 s at 30 fps with jittered frame times (a
variable-frame-rate source), raw positions with a 0.5 m zigzag, stabilized ones with a 0.06 m zigzag that any second
smoothing would flatten.

Run:  python -m unittest test_tactical_stabilized          (from this directory)
"""
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TACTICAL = os.path.join(ROOT, 'tactical.js')
PAGE = os.path.join(ROOT, 'index.html')
NODE = shutil.which('node')
GIT = shutil.which('git')
BASELINE = '7fe810db666dfab48fc8c39671b92c7262c93479'   # the last tactical.js before the stabilized positions

# The page code in node: a vm context with a window, a document whose canvases record every call and assignment of
# their 2D context, a localStorage (or one that throws), and the board's controls as plain objects.
HARNESS = r"""
const fs = require('fs'), vm = require('vm');
const input = JSON.parse(fs.readFileSync(0, 'utf8'));
function recorder(log) {
  return new Proxy({}, {
    get: (o, k) => k === 'measureText' ? (s) => ({width: String(s).length * 6})
      : (...args) => { log.push([String(k), ...args.map((a) => a && typeof a === 'object' && !Array.isArray(a) ? 'object' : a)]); },
    set: (o, k, v) => { log.push(['=' + String(k), v]); return true; },
  });
}
function load(path, saved) {
  let storage;
  if (saved === 'throw') storage = {getItem() { throw new Error('denied'); }, setItem() { throw new Error('denied'); }};
  else {
    const m = new Map(Object.entries(saved || {}));
    storage = {getItem: (k) => m.has(k) ? m.get(k) : null, setItem: (k, v) => { m.set(k, String(v)); }, dump: () => Object.fromEntries(m)};
  }
  const sandbox = {window: {devicePixelRatio: 1}, document: {createElement: () => ({width: 0, height: 0, getContext: () => recorder([])})}, localStorage: storage, console};
  vm.createContext(sandbox);
  vm.runInContext(fs.readFileSync(path, 'utf8'), sandbox, {filename: path});
  return {T: sandbox.window.Tactical, storage};
}
function control(props) {
  const listeners = [];
  return Object.assign({addEventListener: (type, f) => listeners.push(f), change: () => listeners.forEach((f) => f())}, props);
}
function view(job) {
  const {T, storage} = load(job.tactical, job.storage);
  const log = [], canvas = {clientWidth: 960, width: 0, height: 0, getContext: () => recorder(log)}, status = {textContent: ''};
  const controls = {labels: control({checked: job.labels !== false}), debug: control({checked: false}), lineup: control({checked: !!job.lineup}),
                    footprint: control({checked: true}), positions: control({value: 'stabilized'}), positionsWrap: {hidden: true}};
  const board = T.create({canvas, controls, status});
  const out = {restored: controls.positions.value};
  board.load(job.result, job.info);
  out.hiddenBeforeDebug = controls.positionsWrap.hidden;
  out.statusLoaded = status.textContent;
  if (job.debug) { controls.debug.checked = true; controls.debug.change(); }
  if (job.mode) { controls.positions.value = job.mode; controls.positions.change(); }
  out.hidden = controls.positionsWrap.hidden;
  out.status = status.textContent;
  out.storage = storage.dump ? storage.dump() : null;
  const g = board.geometry();
  Object.assign(out, {radius: g.playerRadius, scale: g.scale, ox: g.ox, oy: g.oy});
  out.states = job.times.map((t) => {
    const start = log.length, state = board.render(t, {selected: job.selected || null});
    return job.logs === false ? {state} : {state, log: log.slice(start)};
  });
  return out;
}
function sample(job) {
  const {T} = load(job.tactical);
  const tl = T.buildTimeline(job.result, job.info), R = T.RULES;
  const at = (sampler, track) => job.times.map((t) => sampler(track, t, tl.frameStep, track.info.anonymous ? R.anonymousHold : R.holdAfter,
                                                              track.info.anonymous ? R.anonymousGap : R.interpolateGap));
  const dump = (tracks, sampler) => tracks ? Object.fromEntries([...tracks].map(([key, track]) => [key,
    {t: Array.from(track.t), x: Array.from(track.x), y: Array.from(track.y), k: Array.from(track.k), at: at(sampler, track)}])) : null;
  return {frameStep: tl.frameStep, stable: dump(tl.stable, T.sampleStable), raw: dump(tl.tracks, T.sampleAt)};
}
function compare(job) {
  // The same view jobs on two tactical.js files: every render's state and canvas calls, compared. With `prefix` the
  // calls are compared up to the debug summary box (its third line names the positions drawn) and the states without
  // the fields the comparison adds. The status lines are compared too.
  const a = view(job.a), b = view(job.b);
  const cut = (log) => { if (!job.prefix) return log; const i = log.findIndex((c) => c[0] === 'fillRect' && c[1] === 6 && c[2] === 6); return i < 0 ? log : log.slice(0, i); };
  const strip = (s) => job.prefix ? {time: s.time, frame: s.frame, calibrated: s.calibrated, players: s.players, ball: s.ball} : s;
  let states = 0, logs = 0, calls = 0, first = null;
  a.states.forEach((r, i) => {
    const q = b.states[i], sa = JSON.stringify(strip(r.state)), sb = JSON.stringify(strip(q.state)), la = JSON.stringify(cut(r.log)), lb = JSON.stringify(cut(q.log));
    calls += cut(r.log).length;
    if (sa !== sb) { states++; first = first || {i, a: sa.slice(0, 1500), b: sb.slice(0, 1500)}; }
    if (la !== lb) { logs++; first = first || {i, a: la.slice(0, 1500), b: lb.slice(0, 1500)}; }
  });
  return {renders: a.states.length, states, logs, calls, first, status: [a.status, b.status]};
}
function lazy(job) {
  // How often the block's identities are read: by two views as the analytics and team-shape panels create them
  // (controls {}, the pitch drawn, no player), then by the board, which draws them over many renders.
  const {T} = load(job.tactical);
  let reads = 0;
  job.result.stabilized.ids = new Proxy(job.result.stabilized.ids, {ownKeys: (o) => { reads++; return Reflect.ownKeys(o); }});
  const canvas = () => ({clientWidth: 960, width: 0, height: 0, getContext: () => recorder([])});
  for (let v = 0; v < 2; v++) { const panel = T.create({canvas: canvas(), controls: {}}); panel.load(job.result, job.info); panel.drawPitch(panel.geometry()); }
  const panels = reads, status = {textContent: ''};
  const board = T.create({canvas: canvas(), controls: {}, status});
  board.load(job.result, job.info);
  const drawn = job.times.map((t) => board.render(t).players.length);
  return {panels, board: reads - panels, same: board.timeline.stable === board.timeline.stable, drawn, status: status.textContent};
}
const kinds = {sample, compare, lazy, view};
const out = input.jobs.map((job) => (kinds[job.kind] || view)(job));
process.stdout.write(JSON.stringify(out));
"""


def run(*jobs):
    done = subprocess.run([NODE, '-e', HARNESS], input=json.dumps({'jobs': list(jobs)}), capture_output=True, text=True, encoding='utf-8', timeout=300)
    if done.returncode:
        raise AssertionError(done.stderr[-3000:])
    return json.loads(done.stdout)


# ---------- a synthetic clip ----------
FPS, N, L, WD = 30.0, 120, 105.0, 68.0
STEP = 1/FPS
INFO = {'video': {'fps': FPS}}


def frame_times():
    rng = np.random.default_rng(7)
    t = np.arange(N)/FPS + rng.uniform(-.004, .004, N)   # a variable-frame-rate source: the frame times are not a grid
    t[0] = 0.0
    return [round(float(v), 6) for v in t]


TIMES = frame_times()
PATHS = {'A-01': lambda t: (30+2*t, 20+t), 'B-01': lambda t: (60-1.5*t, 40+.5*t), 'GK-1': lambda t: (8+.5*t, 34+.3*math.sin(t)),
         'X-99': lambda t: (70.0, 10.0), 7: lambda t: (50+t, 50-.5*t), 8: lambda t: (70.0, 20.0), 9: lambda t: (20+3*t, 60-.5*t)}
KINDS = {'A-01': ['I' if 40 <= i < 50 else 'P' if 80 <= i < 95 else 'O' for i in range(N)],
         # unknown for 1.7 s (frames 20-69: never bridged) and for 0.73 s (95-115: the straight line)
         'B-01': ['U' if 20 <= i < 70 or 95 <= i < 116 else 'O' for i in range(N)],
         'GK-1': list('O'*10 + 'I'*5 + 'O'*5 + 'I' + 'O'*9 + 'P'*5) + ['O']*(N-35),
         'X-99': ['O']*N}                                      # not on the roster: never drawn
RAW_SEEN = {pid: [k == 'O' for k in KINDS[pid]] for pid in ('A-01', 'B-01')} | {'GK-1': [True]*N, 'X-99': [True]*N}
LONG_GAP, SHORT_GAP = (19, 70), (94, 116)                      # B-01's samples on either side of its gaps
ANON_RAW = {7: range(10, 60), 8: range(60, 90)}                # '?' tracks in the frames
ANON_BLOCK = [10 <= i < 30 or 45 <= i < 60 for i in range(N)]  # track 7 in the block: a 0.5 s hole (> RULES.anonymousGap)
# Track 9, only in the block: stabilizer bridges (kind I) over 0.57 s (frames 20-35, between the observations of
# frames 19 and 36: beyond RULES.anonymousGap) and over 0.23 s (50-55, between 49 and 56: within it), and three I
# samples after its last observation (71-73: no observation to join).
ANON_BRIDGED = ['O' if i < 20 or 36 <= i < 50 or 56 <= i < 71 else 'I' if i < 74 else None for i in range(N)]
LONG_BRIDGE, SHORT_BRIDGE, ANON_END = (19, 36), (49, 56), 70
ZERO_SIGMA = 60                                                # A-01's sigma there is [0, 0]: stored to 0.1 m, it was below 0.05


def norm(x, y, digits):
    return [round(x/L, digits), round(y/WD, digits)]


def make_result(timeline=True, stabilized=True):
    rng = np.random.default_rng(3)
    frames = []
    for i, t in enumerate(TIMES):
        zig, people = (1 if i % 2 else -1), []

        def person(pid, track, team, state='confirmed'):
            x, y = PATHS[pid if pid else track](t)
            people.append({'time': t, 'track': track, 'id': pid, 'team': team, 'state': state, 'identityConfidence': .9, 'pitchConfidence': .8,
                           'pitch': norm(x+.5*zig+rng.normal(0, .1), y+.3*zig+rng.normal(0, .1), 4), 'pitchSigma': [.4, .4]})
        for n, (pid, team) in enumerate((('A-01', 'A'), ('B-01', 'B'), ('GK-1', 'A'), ('X-99', 'B'))):
            if RAW_SEEN[pid][i]:
                person(pid, n+1, team)
        for track, team in ((7, 'B'), (8, 'A')):
            if i in ANON_RAW[track]:
                person(None, track, team, 'uncertain')
        bx, by = 40+5*t, 30+2*t
        ball = {'state': 'TRACKED' if not 50 <= i < 56 else 'MISSING', 'confidence': .9 if not 50 <= i < 56 else .3, 'pitch': norm(bx, by, 4), 'pitchConfidence': .9}
        frames.append({'time': t, 'frame': i, 'calibrated': True, 'people': people, 'ball': ball, 'view': [[.2, .1], [.8, .1], [.9, .9], [.1, .9]]})
    result = {'version': 3, 'video': {'fps': FPS, 'width': 1280, 'height': 720}, 'frames': frames, 'tactical': {'goalEnds': {'left': 'A', 'right': 'B'}},
              'match': {'players': [{'id': 'A-01', 'display': '7', 'team': 'A'}, {'id': 'B-01', 'display': '9', 'team': 'B'}],
                        'goalkeepers': [{'id': 'GK-1', 'display': 'GK-A', 'team': 'A'}]}}
    if timeline:
        # As soccer/imputation.py writes it: O the frame's raw position, I and P estimates with a sigma, U nothing.
        ids = {}
        for pid, team in (('A-01', 'A'), ('B-01', 'B'), ('GK-1', 'A')):
            cols = {'team': team, 'x': [], 'y': [], 'kind': [], 'sigma': []}
            for i, f in enumerate(frames):
                kind = KINDS[pid][i] if pid != 'GK-1' else 'O'
                raw = next((p['pitch'] for p in f['people'] if p['id'] == pid), None)
                xy = raw if kind == 'O' else None if kind == 'U' else norm(*PATHS[pid](f['time']), 4)
                cols['kind'].append(kind)
                cols['x'].append(xy[0] if xy else None)
                cols['y'].append(xy[1] if xy else None)
                cols['sigma'].append(None if kind == 'U' else [.4, .4] if kind == 'O' else [1.0, 1.0] if kind == 'I' else [3.0, 3.0])
            ids[pid] = cols
        result['timeline'] = {'frameTimes': [round(t, 3) for t in TIMES], 'ids': ids, 'method': 'test'}
    if stabilized:
        ids = {}
        for pid, team in (('A-01', 'A'), ('B-01', 'B'), ('GK-1', 'A'), ('X-99', 'B')):
            cols = {'team': team, 'x': [], 'y': [], 'kind': [], 'sigma': [], 'v': [], 'w': []}
            for i, t in enumerate(TIMES):
                kind = KINDS[pid][i]
                cols['kind'].append(kind)
                if kind == 'U':
                    for key in ('x', 'y', 'sigma', 'v', 'w'):
                        cols[key].append(None)
                    continue
                x, y = PATHS[pid](t)
                zig = .06 if i % 2 else -.06
                xy = norm(x+zig, y-zig, 6)
                cols['x'].append(xy[0])
                cols['y'].append(xy[1])
                cols['sigma'].append([0.0, 0.0] if (pid, i) == ('A-01', ZERO_SIGMA) else [.3, .8] if kind == 'O' else [.6, 1.2] if kind == 'I' else [2.5, 3.5])
                cols['v'].append([2.0, 1.0])
                cols['w'].append(None if kind != 'O' else .4 if i == 10 else 0.0 if i == 11 else 1.0)
            ids[pid] = cols
        anonymous = {}
        for track, team, kinds in ((7, 'B', ['O' if s else None for s in ANON_BLOCK]), (9, 'A', ANON_BRIDGED)):
            anon = {'team': team, 'x': [], 'y': [], 'kind': [], 'sigma': []}
            for i, t in enumerate(TIMES):
                xy = norm(*PATHS[track](t), 6) if kinds[i] else [None, None]
                anon['x'].append(xy[0])
                anon['y'].append(xy[1])
                anon['kind'].append(kinds[i])
                anon['sigma'].append(None if not kinds[i] else [.5, 1.0] if kinds[i] == 'O' else [.8, 1.5])
            anonymous[str(track)] = anon
        result['stabilized'] = {'version': 1, 'method': 'test smoother', 'frameTimes': [round(t, 3) for t in TIMES], 'ids': ids,
                                'anonymous': anonymous, 'flags': [], 'summary': {}, 'noise': {}, 'constants': {}}
    return result


def view_job(result, times, tactical=TACTICAL, **options):
    return {'kind': 'view', 'tactical': tactical, 'result': result, 'info': INFO, 'times': list(times), **options}


def by_key(players):
    return {('anon:%s' % p['track']) if p.get('anonymous') else p['id']: p for p in players}


def baseline_file(directory):
    """tactical.js at BASELINE (the unmodified view), written into the directory; None without git or the commit."""
    if not GIT:
        return None
    done = subprocess.run([GIT, 'show', f'{BASELINE}:./tactical.js'], cwd=ROOT, capture_output=True)
    if done.returncode or not done.stdout:
        return None
    path = os.path.join(directory, 'tactical_baseline.js')
    with open(path, 'wb') as handle:
        handle.write(done.stdout)
    return path


@unittest.skipUnless(NODE, 'needs node to run the page script')
class StabilizedBoardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = make_result()
        cls.plain = make_result(stabilized=False)
        cls.frames_only = make_result(timeline=False, stabilized=False)
        cls.tmp = tempfile.mkdtemp(prefix='pitchiq-tactical-')
        cls.baseline = baseline_file(cls.tmp)
        cls.block = cls.result['stabilized']

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def block_samples(self, key):
        """(times, x, y, kinds) of a block track's samples (its frames with a position)."""
        cols = self.block['ids'][key] if key in self.block['ids'] else self.block['anonymous'][key]
        rows = [(TIMES[i], cols['x'][i], cols['y'][i], cols['kind'][i]) for i in range(N) if cols['kind'][i] not in (None, 'U')]
        return tuple(map(list, zip(*rows)))

    def drawn_counts(self):
        """The status line's counts of the stabilized positions, from the block: the roster's observed and estimated
        samples (X-99 is not on the roster) and the '?' observations (a dropped bridge holds none)."""
        roster = [self.block['ids'][pid]['kind'] for pid in ('A-01', 'B-01', 'GK-1')]
        observed, estimated = sum(k.count('O') for k in roster), sum(k.count('I')+k.count('P') for k in roster)
        anonymous = sum(cols['kind'].count('O') for cols in self.block['anonymous'].values())
        self.assertEqual((observed, estimated, anonymous), (95+49+109, 25+0+11, 35+49))
        return f' · 3 players with positions · {observed} observed and {estimated} estimated samples · {anonymous} observations of unidentified people shown as ? · '

    def test_stabilized_samples_are_drawn_as_given(self):
        sampled, board = run({'kind': 'sample', 'tactical': TACTICAL, 'result': self.result, 'info': INFO, 'times': TIMES}, view_job(self.result, TIMES, logs=False))
        self.assertEqual(set(sampled['stable']), {'A-01', 'B-01', 'GK-1', 'anon:7', 'anon:9'})   # X-99 is not on the roster
        for key in ('A-01', 'B-01', 'GK-1'):
            ts, xs, ys, kinds = self.block_samples(key)
            track = sampled['stable'][key]
            self.assertEqual((track['t'], track['x'], track['y'], track['k']), (ts, xs, ys, kinds))   # bit for bit: not smoothed
            at = [s for s, t in zip(track['at'], TIMES) if t in set(ts)]
            self.assertEqual([(s['x'], s['y'], s['kind']) for s in at], list(zip(xs, ys, kinds)))
        # The default board (Debug positions off) draws them; the raw path still smooths its observations.
        drawn = 0
        for i, frame in enumerate(board['states']):
            players = by_key(frame['state']['players'])
            self.assertEqual(frame['state']['positions'], 'stabilized')
            self.assertNotIn('X-99', players)
            for key in ('A-01', 'GK-1'):
                cols = self.block['ids'][key]
                self.assertEqual((players[key]['x'], players[key]['y']), (cols['x'][i], cols['y'][i]))
                drawn += 1
        self.assertEqual(drawn, 2*N)
        raw = sampled['raw']['A-01']
        observed = [i for i in range(N) if KINDS['A-01'][i] == 'O']
        timeline = self.result['timeline']['ids']['A-01']['x']
        changed = sum(abs(raw['x'][raw['t'].index(TIMES[i])] - timeline[i]) > 1e-9 for i in observed)
        self.assertGreater(changed, .9*len(observed))
        # The zigzag survives: consecutive stabilized differences keep alternating in sign on the observed run.
        ts, xs, _, _ = self.block_samples('GK-1')
        d = np.diff(np.array(xs[40:100]) - np.array([PATHS['GK-1'](t)[0]/L for t in ts[40:100]]))
        self.assertTrue(np.all(d[:-1]*d[1:] < 0))

    def test_any_time_is_the_straight_line_between_samples(self):
        rng = np.random.default_rng(11)
        times = sorted(rng.uniform(TIMES[0], TIMES[-1], 600).tolist())
        out, = run({'kind': 'sample', 'tactical': TACTICAL, 'result': self.result, 'info': INFO, 'times': times})
        self.assertGreater(np.ptp(np.diff(TIMES)), .005)   # the frame times are not a grid: interpolation must use them
        for key in ('A-01', 'GK-1'):
            ts, xs, ys, _ = self.block_samples(key)
            got = out['stable'][key]['at']
            self.assertTrue(all(s is not None for s in got))
            np.testing.assert_allclose([s['x'] for s in got], np.interp(times, ts, xs), rtol=0, atol=1e-12)
            np.testing.assert_allclose([s['y'] for s in got], np.interp(times, ts, ys), rtol=0, atol=1e-12)
            # The sigma too (the larger axis): continuous, so a prediction's fading does not step at the frames.
            sigma = [max(self.block['ids'][key]['sigma'][TIMES.index(t)]) for t in ts]
            np.testing.assert_allclose([s['sigma'] for s in got], np.interp(times, ts, sigma), rtol=0, atol=1e-12)
            # On the grid i/30 instead of the actual times the straight line would be elsewhere.
            grid = np.interp(times, np.arange(N)/FPS, [self.block['ids'][key]['x'][i] for i in range(N)])
            self.assertGreater(np.max(np.abs(np.array([s['x'] for s in got]) - grid)), 1e-5)
        # B-01's 0.73 s gap (at most RULES.interpolateGap) is bridged by the straight line between its two samples.
        (a, b), cols = SHORT_GAP, self.block['ids']['B-01']
        inside = [(t, s) for t, s in zip(times, out['stable']['B-01']['at']) if TIMES[a] < t < TIMES[b]]
        self.assertGreater(len(inside), 50)
        for t, s in inside:
            u = (t-TIMES[a])/(TIMES[b]-TIMES[a])
            self.assertAlmostEqual(s['x'], cols['x'][a]+(cols['x'][b]-cols['x'][a])*u, places=12)
            self.assertEqual((s['kind'], s['source']), ('I', 'INTERPOLATED'))   # two observations, unknown frames between them

    def test_no_snap_to_the_nearer_frame(self):
        frames = range(55, 75)   # A-01 observed throughout
        offsets = (.02, .1, .2, .24)   # frames: inside the raw path's snap (a quarter frame) on either side of a frame
        times = [TIMES[k]+d*STEP for k in frames for d in offsets] + [TIMES[k+1]-d*STEP for k in frames for d in offsets]
        out, = run({'kind': 'sample', 'tactical': TACTICAL, 'result': self.result, 'info': INFO, 'times': times})
        ts, xs, _, _ = self.block_samples('A-01')
        expected = np.interp(times, ts, xs)
        got = np.array([s['x'] for s in out['stable']['A-01']['at']])
        np.testing.assert_allclose(got, expected, rtol=0, atol=1e-12)
        samples = set(xs)
        self.assertFalse(any(x in samples for x in got))   # never a frame's own position between frames
        # The raw path keeps its quarter-frame snap (the old behaviour): there it returns the nearer sample itself.
        raw = out['raw']['A-01']
        snapped = sum(s['x'] in set(raw['x']) for s, t in zip(raw['at'], times))
        self.assertEqual(snapped, len(times))

    def test_interpolated_ring_follows_the_two_samples(self):
        ts, _, _, kinds = self.block_samples('GK-1')
        mids = [(ts[k]+ts[k+1])/2 for k in range(40)]
        expected = ['P' if 'P' in (kinds[k], kinds[k+1]) else 'O' if kinds[k] == kinds[k+1] == 'O' else 'I' for k in range(40)]
        board, raw = run(view_job(self.result, mids + ts[:40]), view_job(self.result, mids, debug=True, mode='raw'))
        r = board['radius']

        def ring(frame, key):
            p = by_key(frame['state']['players'])[key]
            px, py = board['ox']+p['x']*L*board['scale'], board['oy']+p['y']*WD*board['scale']
            return p, any(c[0] == 'arc' and abs(c[1]-px) < 1e-6 and abs(c[2]-py) < 1e-6 and abs(c[3]-.6*r) < 1e-9 for c in frame['log'])
        for k, frame in enumerate(board['states'][:40]):
            p, drawn = ring(frame, 'GK-1')
            self.assertEqual(p['kind'], expected[k], k)
            self.assertEqual(p['source'], {'O': 'OBSERVED', 'I': 'INTERPOLATED', 'P': 'PREDICTED'}[expected[k]])
            self.assertEqual(drawn, expected[k] == 'I', k)
        for k, frame in enumerate(board['states'][40:]):   # at a sample's own time: that sample's kind
            self.assertEqual(ring(frame, 'GK-1')[0]['kind'], kinds[k])
        # The raw path (unchanged) still rings between two observed frames: the flicker the stabilized path removes.
        rings = sum(by_key(f['state']['players'])['GK-1']['kind'] == 'I' for f, e in zip(raw['states'], expected) if e == 'O')
        self.assertEqual(rings, expected.count('O'))

    def test_anonymous_question_marks_come_from_the_block(self):
        times = TIMES + [TIMES[29]+.2, TIMES[29]+.35]
        board, plain = run(view_job(self.result, times, logs=False), view_job(self.plain, TIMES, logs=False))
        cols = self.block['anonymous']['7']
        for i, frame in enumerate(board['states'][:N]):
            anon = {p['track']: p for p in frame['state']['players'] if p.get('anonymous')}
            self.assertNotIn(8, anon)   # only in the frames: with a block the '?' tracks are the block's
            if ANON_BLOCK[i]:
                self.assertEqual((anon[7]['x'], anon[7]['y'], anon[7]['team'], anon[7]['display']), (cols['x'][i], cols['y'][i], 'B', '?'))
            elif TIMES[i] < TIMES[10]-.25*STEP or TIMES[29]+.3 < TIMES[i] < TIMES[45]-.25*STEP or TIMES[i] > TIMES[59]+.3:
                self.assertNotIn(7, anon)   # the 0.5 s hole is not bridged (RULES.anonymousGap) and is held only briefly
        held, gone = board['states'][N]['state'], board['states'][N+1]['state']
        p = [q for q in held['players'] if q.get('anonymous') and q['track'] == 7][0]
        self.assertEqual((p['source'], p['x']), ('HELD', cols['x'][29]))
        self.assertFalse([q for q in gone['players'] if q.get('anonymous') and q['track'] == 7])
        # Without the block: the raw '?' tracks, both of them, smoothed as before.
        seen = {p['track'] for f in plain['states'] for p in f['state']['players'] if p.get('anonymous')}
        self.assertEqual(seen, {7, 8})

    def test_anonymous_bridges_obey_the_anonymous_gap(self):
        (a, b), (c, e) = LONG_BRIDGE, SHORT_BRIDGE
        long_, short = np.linspace(TIMES[a], TIMES[b], 200)[1:-1].tolist(), np.linspace(TIMES[c], TIMES[e], 60)[1:-1].tolist()
        tail = np.linspace(TIMES[ANON_END], TIMES[ANON_END+5], 40)[1:].tolist()
        sampled, board = run({'kind': 'sample', 'tactical': TACTICAL, 'result': self.result, 'info': INFO, 'times': long_ + short + tail},
                             view_job(self.result, TIMES, logs=False))
        cols, track = self.block['anonymous']['9'], sampled['stable']['anon:9']
        # A '?' carries no kind on the board: a bridge beyond RULES.anonymousGap (or after the last observation) would
        # be drawn as observed, so it is left out; the bridge within it is kept as given.
        kept = [i for i in range(N) if ANON_BRIDGED[i] and not a < i < b and i <= ANON_END]
        self.assertEqual((track['t'], track['x'], track['k']), ([TIMES[i] for i in kept], [cols['x'][i] for i in kept], [ANON_BRIDGED[i] for i in kept]))
        self.assertEqual(track['k'].count('I'), e-c-1)
        at = iter(track['at'])
        for t, s in zip(long_, at):   # held for RULES.anonymousHold, then hidden: never a fabricated position
            if t-TIMES[a] <= .3:
                self.assertEqual((s['source'], s['x']), ('HELD', cols['x'][a]))
            elif t < TIMES[b]-.25*STEP:
                self.assertIsNone(s, t)
        for t, s in zip(short, at):   # the stabilizer's bridge, its samples joined by straight lines
            self.assertNotEqual(s['source'], 'HELD')
            self.assertAlmostEqual(s['x'], float(np.interp(t, track['t'], track['x'])), places=12)
        for t, s in zip(tail, at):
            self.assertTrue(s is None if t-TIMES[ANON_END] > .3 else (s['source'], s['x']) == ('HELD', cols['x'][ANON_END]), t)
        for i, frame in enumerate(board['states']):
            anon = {p['track']: p for p in frame['state']['players'] if p.get('anonymous')}
            if c < i < e:
                self.assertEqual((anon[9]['x'], anon[9]['kind']), (cols['x'][i], 'I'))
            elif (a < i < b and TIMES[i]-TIMES[a] > .3) or TIMES[i]-TIMES[ANON_END] > .3:
                self.assertNotIn(9, anon, i)

    def test_debug_comparison_draws_raw_stabilized_or_both(self):
        times = TIMES[::4]
        key = 'pitchiq.tactical.positions'
        default, off, raw, stable, both, plain, denied, absent = run(
            view_job(self.result, times),
            view_job(self.result, times, storage={key: 'raw'}),                 # remembered, but Debug positions is off
            view_job(self.result, times, debug=True, mode='raw'),
            view_job(self.result, times, debug=True, mode='stabilized'),
            view_job(self.result, times, debug=True, mode='both'),
            view_job(self.plain, times, debug=True),
            view_job(self.result, times, debug=True, mode='both', storage='throw'),
            view_job(self.plain, times, debug=True, mode='both'))
        # The selector: hidden until Debug positions is on, default 'stabilized', remembered per viewer.
        self.assertEqual((default['restored'], default['hiddenBeforeDebug'], default['hidden']), ('stabilized', True, True))
        self.assertEqual((off['restored'], off['hidden']), ('raw', True))
        self.assertEqual((both['hidden'], both['storage']), (False, {key: 'both'}))
        self.assertEqual((denied['restored'], denied['hidden'], denied['storage']), ('stabilized', False, None))
        self.assertEqual(absent['hidden'], True)   # no stabilized block: nothing to compare
        for k in range(len(times)):
            s = {name: run_['states'][k]['state'] for name, run_ in
                 (('default', default), ('off', off), ('raw', raw), ('stable', stable), ('both', both), ('plain', plain), ('denied', denied), ('absent', absent))}
            self.assertEqual([s[n]['positions'] for n in ('default', 'off', 'raw', 'stable', 'both', 'denied')],
                             ['stabilized', 'stabilized', 'raw', 'stabilized', 'both', 'both'])
            self.assertNotIn('positions', s['plain'])
            self.assertEqual(s['off']['players'], s['default']['players'])
            self.assertEqual(s['stable']['players'], s['default']['players'])
            self.assertEqual(s['raw']['players'], s['plain']['players'])            # raw: exactly the drawing without the block
            self.assertEqual(s['both']['players'], s['default']['players'])
            self.assertEqual(s['both']['raw'], s['plain']['players'])
            self.assertEqual(s['denied']['players'], s['both']['players'])
            self.assertEqual(s['absent']['players'], s['plain']['players'])
            self.assertNotIn('raw', s['stable'])
            for name in ('off', 'raw', 'stable', 'both', 'plain'):                     # the ball path is untouched
                self.assertEqual(s[name]['ball'], s['default']['ball'])
        # 'both': each raw position a small cross, tied by a thin line to the stabilized marker; never a ring, which on
        # its own marker would read as the interpolated marker's inner ring.
        r, lines, markers, rings = both['radius'], 0, 0, 0
        d = .32*r
        px = lambda p: (both['ox']+p['x']*L*both['scale'], both['oy']+p['y']*WD*both['scale'])
        segment = lambda log, a, b: any(c[0] == 'moveTo' and abs(c[1]-a[0]) < 1e-6 and abs(c[2]-a[1]) < 1e-6 and log[j+1][0] == 'lineTo' and
                                        abs(log[j+1][1]-b[0]) < 1e-6 and abs(log[j+1][2]-b[1]) < 1e-6 for j, c in enumerate(log[:-1]))
        for frame in both['states']:
            stable_ = by_key(frame['state']['players'])
            log = frame['log']
            for key_, p in by_key(frame['state']['raw']).items():
                (x, y), q = px(p), stable_.get(key_)
                markers += segment(log, (x-d, y-d), (x+d, y+d)) and segment(log, (x-d, y+d), (x+d, y-d))
                rings += any(c[0] == 'arc' and abs(c[1]-x) < 1e-6 and abs(c[2]-y) < 1e-6 for c in log)
                if q:
                    lines += segment(log, (x, y), px(q))
        raw_shown = sum(len(f['state']['raw']) for f in both['states'])
        self.assertEqual(markers, raw_shown)
        self.assertEqual(rings, 0)
        self.assertGreater(lines, .8*raw_shown)
        for job in (stable, both):   # radius .45 r: only the ball's
            small = sum(c[0] == 'arc' and abs(c[3]-.45*r) < 1e-9 for f in job['states'] for c in f['log'])
            self.assertEqual(small, sum(1 for f in job['states'] if f['state']['ball']))
        texts = lambda job: [c[1] for f in job['states'] for c in f['log'] if c[0] == 'fillText']
        self.assertIn('positions both (×: raw, tied to the stabilized marker) · test smoother', texts(both))
        self.assertIn('positions stabilized · test smoother', texts(stable))
        # The per-player debug boxes would cover the compared markers: 'both' leaves them out (the summary stays).
        self.assertTrue(any(s.startswith('identity ') for s in texts(stable)))
        self.assertFalse(any(s.startswith(('identity ', 'no identity')) for s in texts(both)))

    def test_no_line_across_a_long_gap(self):
        a, b = LONG_GAP
        ta, tb = TIMES[a], TIMES[b]
        times = np.linspace(ta, tb-.3*STEP, 400)[1:-1].tolist() + [ta+.3, ta+.59, ta+.61, tb-.3*STEP, tb-.2*STEP]
        out, = run({'kind': 'sample', 'tactical': TACTICAL, 'result': self.result, 'info': INFO, 'times': times})
        cols, at = self.block['ids']['B-01'], out['stable']['B-01']['at']
        for t, s in zip(times[:398], at):
            self.assertTrue(s is None or (s['x'], s['source']) == (cols['x'][a], 'HELD'), t)   # held, then hidden; never a blend
        held, late, gone, early, edge = at[398:]
        self.assertEqual((held['source'], held['x']), ('HELD', cols['x'][a]))
        self.assertAlmostEqual(held['alpha'], 1-.7*.3/.6, places=9)
        self.assertEqual(late['source'], 'HELD')
        self.assertIsNone(gone)
        self.assertIsNone(early)
        self.assertEqual((edge['x'], edge['source']), (cols['x'][b], 'OBSERVED'))   # a run's first sample up to a quarter frame early

    def test_without_the_block_the_view_is_unmodified(self):
        if not self.baseline:
            self.skipTest('needs git and the baseline commit of tactical.js')
        grid = sorted(set(TIMES[::2] + [t+d*STEP for t in TIMES[::3] for d in (.1, .25, .3, .5, .8)] + [-.05, TIMES[-1]+.3, TIMES[-1]+1.0]))
        jobs = []
        for result in (self.plain, self.frames_only):
            for options in ({}, {'debug': True}, {'lineup': True, 'selected': 'A-01'}, {'labels': False, 'debug': True, 'mode': 'both'}):
                jobs.append({'kind': 'compare', 'a': view_job(result, grid, **options), 'b': view_job(result, grid, tactical=self.baseline, **options)})
        for out in run(*jobs):
            self.assertEqual((out['states'], out['logs']), (0, 0), out['first'])
            self.assertEqual(out['renders'], len(grid))
            self.assertGreater(out['calls'], 50*len(grid))   # the comparison covers real drawing
            self.assertEqual(out['status'][0], out['status'][1])
            self.assertIn('players with positions', out['status'][0])

    def test_raw_comparison_is_the_old_drawing(self):
        if not self.baseline:
            self.skipTest('needs git and the baseline commit of tactical.js')
        grid = sorted(set(TIMES[::2] + [t+d*STEP for t in TIMES[::3] for d in (.1, .25, .5)]))
        # The unmodified view ignores the block: its drawing is what 'raw' must reproduce.
        out, = run({'kind': 'compare', 'prefix': True, 'a': view_job(self.result, grid, debug=True, mode='raw'),
                    'b': view_job(self.result, grid, tactical=self.baseline, debug=True)})
        self.assertEqual((out['states'], out['logs']), (0, 0), out['first'])
        self.assertGreater(out['calls'], 50*len(grid))
        self.assertEqual(out['status'][0], out['status'][1])   # the status line counts the raw positions it draws

    def test_status_line_counts_what_is_drawn(self):
        default, raw, both, plain = run(view_job(self.result, [1.0]), view_job(self.result, [1.0], debug=True, mode='raw'),
                                        view_job(self.result, [1.0], debug=True, mode='both'), view_job(self.plain, [1.0]))
        self.assertIn(self.drawn_counts(), default['status'])
        for out in (default, both):
            self.assertEqual(out['status'], default['statusLoaded'])
        # The raw positions drawn: the raw timeline's counts, the status line of the view without the block.
        self.assertEqual(raw['status'], plain['status'])
        self.assertNotEqual(raw['status'], default['status'])
        self.assertEqual(raw['statusLoaded'], default['status'])   # before Debug positions turned on: the stabilized ones

    def test_stable_tracks_are_built_only_for_a_view_that_draws_them(self):
        times = TIMES[::10]
        out, = run({'kind': 'lazy', 'tactical': TACTICAL, 'result': self.result, 'info': INFO, 'times': times})
        # The analytics and team-shape panels load a timeline for the pitch: the block's identities are never read.
        self.assertEqual(out['panels'], 0)
        self.assertEqual(out['board'], 1)   # the board reads them once, for its status line and every render after it
        self.assertTrue(out['same'])
        self.assertIn(self.drawn_counts(), out['status'])
        board, = run(view_job(self.result, times, logs=False))
        self.assertEqual(out['drawn'], [len(f['state']['players']) for f in board['states']])

    def test_debug_box_never_prints_a_zero_sigma(self):
        t = TIMES[ZERO_SIGMA]
        stable, raw = run(view_job(self.result, [t, TIMES[61]], debug=True, mode='stabilized'), view_job(self.result, [t], debug=True, mode='raw'))
        texts = lambda frame: [c[1] for c in frame['log'] if c[0] == 'fillText']
        # A-01 (display 7) in its debug box, drawn after the labels: name, position, then source and sigma.
        player = lambda frame: texts(frame)[len(texts(frame))-1-texts(frame)[::-1].index('7')+2]
        self.assertEqual(player(stable['states'][0]), 'OBSERVED ± <0.05 m · weight 1.00')   # stored [0, 0]: below 0.05 m, not none
        self.assertEqual(player(stable['states'][1]), 'OBSERVED ± 0.8 m · weight 1.00')
        self.assertRegex(player(raw['states'][0]), r'^OBSERVED ± \d\.\d m$')                # the raw ones print as before


class PageWiringTests(unittest.TestCase):
    """The board's debug control in index.html: the page script still parses, and the selector is wired to the view."""

    def setUp(self):
        with open(PAGE, encoding='utf-8') as handle:
            self.page = handle.read()

    @unittest.skipUnless(NODE, 'needs node to check the page script')
    def test_page_scripts_parse(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'page.js')
            with open(path, 'w', encoding='utf-8') as handle:
                handle.write('\n'.join(re.findall(r'<script>(.*?)</script>', self.page, re.S)))
            for script in (path, TACTICAL):
                done = subprocess.run([NODE, '--check', script], capture_output=True, text=True)
                self.assertEqual(done.returncode, 0, done.stderr)

    def test_selector_next_to_debug_positions(self):
        control = re.search(r'<input type="checkbox" id="tacticalDebug"> Debug positions</label>(<label class="chip" id="tacticalPositionsWrap" hidden[^>]*>.*?</label>)', self.page)
        self.assertIsNotNone(control)
        options = re.findall(r'<option value="(\w+)"( selected)?>', control.group(1))
        self.assertEqual(options, [('raw', ''), ('stabilized', ' selected'), ('both', '')])
        self.assertIn('<select id="tacticalPositions"', control.group(1))
        self.assertIn("positions:$('tacticalPositions'),positionsWrap:$('tacticalPositionsWrap')", self.page)
        self.assertIn('#tacticalPositionsWrap[hidden]{display:none}', self.page)   # .chip's display would otherwise show it


if __name__ == '__main__':
    unittest.main()
