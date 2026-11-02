"""Full Match Mode's coaching report (fullmatch/coaching.py): one synthetic match (EVAL's coaching scene
'unsupported_advance/none_behind', tools/coaching_synth.py: 11v11, possession-4 available, planted wide-defender
advances for the role tier) assembled as segment views with context frames beyond their cuts, measured segment by
segment over the owned frames and merged, against the SAME result measured as one run.

What is checked:
- the merged obs passes the schema (both teams) and counts every owned sample exactly once: its samples equal the single
  run's, every opportunity, moment, frame tier, phase span and stored role sample lies inside one segment's owned
  range, no sample time repeats, and the phase seconds add up to the samples;
- the merged statistics are the sums of the segments' (counts, eligible, phase and abstention seconds) and the
  non-additive ones are recomputed from the pooled samples (the p90 speed, the lineup), never averaged;
- against the single run: phase quality, rough shares, coverage and the data-quality block are equal; the opportunity
  count per detector differs by at most BOUND_OPPS per boundary and the distinct spells by at most BOUND_SPELLS (an
  eligible stretch across a cut is measured as two: a piece can start at the cut, a 3 s shape span can fall below 3 s
  on one side, team_shape's windows restart there), the phase seconds by at most BOUND_PHASE_S per boundary and state
  (the possession decoder sees a view, not the match: at its edges a change's window can move); the reports' verdicts
  are the same;
- nothing of the context frames is counted: with the context frames of every view scrambled (and the possession record
  fixed, the one documented reader of the context), every count of the merged obs is unchanged;
- a match in two periods: the break is a 'dead' span without a source, the interpretation's unknown seconds leave it
  out, the report says the played minutes and that the break is not analysed;
- build_match_report / report_state: missing -> ready, the other team's report from the stored obs at once, a level
  change is the interpretation tier only, the notices validate at every level, the records are store records;
- the settings a match uses, the roster corrections on a view, the per-frame columns cut with the frames, the merge
  refusals and the formation change on a boundary.

Run:  python -m unittest test_fullmatch_coaching          (from this directory)
"""
import copy
import json
import shutil
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for p in (str(ROOT), str(ROOT/'tools')):
    if p not in sys.path:
        sys.path.insert(0, p)
import coaching_synth as cs  # noqa: E402
from fullmatch import assemble, coaching as MC  # noqa: E402
from soccer import coaching, possession_amateur  # noqa: E402
from soccer.coaching import assess as AS, measuring, schema as S, situations as SI, store  # noqa: E402

SCENE = ('unsupported_advance', 'none_behind')
PAD = 30                    # context frames beyond each cut (3 s at the scene's 10 fps), as a real view has
BOUND_OPPS = 2              # opportunities per detector and boundary (module docstring)
BOUND_SPELLS = 2            # distinct spells per detector and boundary
BOUND_PHASE_S = 6.0         # s per phase state and boundary: a change window (2 x 2 s) and a decoder edge step or two
_SCENE = {}


def scene():
    if 'built' not in _SCENE:
        _SCENE['built'] = cs.build(*SCENE)
    return _SCENE['built']


def make_match(folder, result, edges, periods=None, pad=PAD, match_id='synth', quality=None):
    """An assembled match in folder/assembly from one result: segment k owns frames [edges[k][0], edges[k][1]) of the
    result, its view holds its owned frames and `pad` frames of context either side inside its period (assemble.
    write_view, gz), the index (match.json) lists the segments, their owned times, the periods."""
    frames = result['frames']
    times = {f['frame']: f['time'] for f in frames}
    fps = float(result['video']['fps'])
    periods = periods or [0]*len(edges)
    bounds = {}
    for (o0, o1), p in zip(edges, periods):
        lo, hi = bounds.get(p, (o0, o1))
        bounds[p] = (min(lo, o0), max(hi, o1))
    a = Path(folder)/'assembly'
    (a/'views').mkdir(parents=True, exist_ok=True)
    segs = []
    for k, ((o0, o1), p) in enumerate(zip(edges, periods)):
        p0, p1 = max(bounds[p][0], o0-pad), min(bounds[p][1], o1+pad)
        view = dict(result)
        view['frames'] = [f for f in frames if p0 <= f['frame'] < p1]
        view['matchView'] = {'matchId': match_id, 'segment': k, 'period': p, 'own0': o0, 'own1': o1, 'cut0': o0,
                             'cut1': o1}
        name = f's{k:03d}-f{o0}-{o1}.json.gz'
        assemble.write_view(a/'views'/name, view)
        segs.append({'index': k, 'key': name[:-8], 'period': p, 'own0': o0, 'own1': o1, 'proc0': p0, 'proc1': p1,
                     'cut0': o0, 'cut1': o1, 'start': times[o0], 'end': times.get(o1, times[o1-1]+1/fps),
                     'view': f'views/{name}',
                     'quality': dict(quality or {'keyframeAnchors': 1, 'automaticAnchors': 0})})
    index = {'version': 1, 'matchId': match_id, 'source': {'key': 'f'*64, 'name': None}, 'video': result['video'],
             'periods': [{'index': p, 'start': times[lo], 'end': times[hi-1], 'name': f'period {p}',
                          'defendsLeft': 'A'} for p, (lo, hi) in sorted(bounds.items())],
             'segments': segs, 'pitchSize': result['tactical']['pitchSize']}
    (a/'match.json').write_text(json.dumps(index), encoding='utf-8')
    return index


def halves(result, k=2):
    """k segments of equal owned length (a multiple of 10 frames: the 5 Hz grid and the 1 Hz track stay aligned)."""
    f0, f1 = result['frames'][0]['frame'], result['frames'][-1]['frame']+1
    step = (f1-f0)//k//10*10
    cuts = [f0+step*i for i in range(k)]+[f1]
    return list(zip(cuts, cuts[1:]))


def owned_ranges(index):
    return [(float(s['start']), float(s['end'])) for s in index['segments']]


def inside(t, ranges):
    return sum(1 for a, b in ranges if a-1e-6 <= t < b-1e-6)


class Fixture:
    """The synthetic match measured once per process: the parts, the merged obs and the single run's obs."""
    data = None

    @classmethod
    def get(cls):
        if cls.data is None:
            b = scene()
            folder = Path(tempfile.mkdtemp(prefix='fm_coaching_'))
            index = make_match(folder, b.result, halves(b.result))
            settings = cs.settings_for(b, level=4)
            shape = MC.shape_settings_of(index)
            parts = MC.measure_segments(folder, index, None, settings, shape_settings=shape)
            merged = {tm: MC.merge_obs(parts[tm], shape_settings=shape, team_settings=S.team_block(settings, tm))
                      for tm in ('A', 'B')}
            single = measuring.measure(b.result, result_sha='0'*64, shape_settings=shape, settings=settings,
                                       spans=None, team=None)
            cls.data = {'built': b, 'folder': folder, 'index': index, 'settings': settings, 'shape': shape,
                        'parts': parts, 'merged': merged, 'single': single}
        return cls.data


def tearDownModule():
    if Fixture.data is not None:
        shutil.rmtree(Fixture.data['folder'], ignore_errors=True)


class MergedObs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.d = Fixture.get()

    def test_schema_and_every_owned_sample_counted_once(self):
        d = self.d
        ranges = owned_ranges(d['index'])
        for tm in ('A', 'B'):
            m, one = d['merged'][tm], d['single'][tm]
            self.assertEqual(S.validate('obs', m), [], tm)
            json.dumps(m, allow_nan=False)
            tl = m['timeline']
            self.assertEqual(tl['samples'], one['timeline']['samples'])
            self.assertEqual(tl['samples'], sum(p['obs']['timeline']['samples'] for p in d['parts'][tm]))
            ps = tl['phaseSeconds']
            self.assertAlmostEqual(ps['own']+ps['opp']+ps['dead']+ps['unknown'], tl['samples']*tl['step'], places=1)
            times = [t for o in m['opportunities'] for t in o['relations']['t']]
            self.assertTrue(times)
            for t in times:
                self.assertEqual(inside(t, ranges), 1, t)
            for o in m['opportunities']:      # no opportunity sample counted twice (left / right variants aside)
                self.assertEqual(len(o['relations']['t']), len(set(o['relations']['t'])))
            for mo in m['moments']:
                self.assertEqual(inside(mo['t'], ranges), 1)
            for r in m['frameTiers']:
                self.assertEqual(inside(r['t0'], ranges), 1)
                self.assertEqual(inside(r['t1'], ranges), 1)
            geo = m['roleInputs']['detect']['samples']
            self.assertEqual(len(geo['t']), len(set(geo['t'])))
            self.assertTrue(all(inside(t, ranges) == 1 for t in geo['t']))
            for name in ('X', 'Y', 'vx'):
                self.assertEqual(len(geo[name]), len(m['roleInputs']['detect']['ids']))
                self.assertTrue(all(len(col) == len(geo['t']) for col in geo[name]))
            self.assertTrue(all(len(bits) == len(m['roleInputs']['detect']['ids']) for bits in geo['ok']))
            track = m['roleInputs']['context']['track']
            self.assertEqual(len(track['t']), len(set(track['t'])))
            self.assertTrue(all(len(col) == len(track['t']) for col in track['x']))
            spans = [(s['t0'], s['t1']) for s in tl['phaseSpans']]
            self.assertEqual(spans, sorted(spans))
            self.assertTrue(all(b0 >= a1 for (_, a1), (b0, _) in zip(spans, spans[1:])))

    def test_statistics_are_sums_and_pooled_recomputations(self):
        d = self.d
        for tm in ('A', 'B'):
            m, parts = d['merged'][tm], d['parts'][tm]
            for det, cov in m['detectors'].items():
                covs = [p['obs']['detectors'][det] for p in parts]
                self.assertEqual(cov['opportunities'], sum(c['opportunities'] for c in covs))
                self.assertAlmostEqual(cov['eligibleSeconds'], sum(c['eligibleSeconds'] for c in covs), places=1)
                self.assertAlmostEqual(cov['phaseSeconds'], sum(c['phaseSeconds'] for c in covs), places=1)
                for reason, sec in cov['abstain'].items():
                    self.assertAlmostEqual(sec, sum(c['abstain'].get(reason, 0.0) for c in covs), places=1)
            for k in ('own', 'opp', 'dead', 'unknown', 'nearChange'):
                self.assertAlmostEqual(m['timeline']['phaseSeconds'][k],
                                       sum(p['obs']['timeline']['phaseSeconds'][k] for p in parts), places=1)
            # the p90 speed of the pooled speed samples, not a mean of the segments' p90s
            for pid, rec in m['timeline']['players'].items():
                if rec['goalkeeper']:
                    continue
                pooled = np.concatenate([p['extra']['speeds'][pid] for p in parts if pid in p['extra']['speeds']])
                want = round(min(float(np.percentile(pooled, 90)), 15.0), 2) if len(pooled) >= SI.P90_MIN_SAMPLES \
                    else None
                self.assertEqual(rec['p90SpeedMS'], want, pid)
                self.assertAlmostEqual(rec['observedSeconds'], sum(p['obs']['timeline']['players'][pid]
                                                                   ['observedSeconds'] for p in parts
                                                                   if pid in p['obs']['timeline']['players']),
                                       places=1)
            self.assertEqual(m['fingerprint'][:16], m['engine']['measurement'])
            self.assertEqual(len(m['timeline']['formationPeriods']),
                             sum(len(p['obs']['timeline']['formationPeriods']) for p in parts))
            self.assertEqual([q['index'] for q in m['roleInputs']['context']['periods']],
                             list(range(len(m['roleInputs']['context']['periods']))))

    def test_close_to_the_single_run_within_the_boundary_bounds(self):
        d = self.d
        boundaries = len(d['index']['segments'])-1
        for tm in ('A', 'B'):
            m, one = d['merged'][tm], d['single'][tm]
            for k in ('available', 'level', 'coverage', 'roughShares'):
                self.assertEqual(m['timeline']['phaseQuality'][k], one['timeline']['phaseQuality'][k], (tm, k))
            for k in ('dataQuality', 'footage', 'scale', 'completeShare', 'observedMean', 'ballReliableShare'):
                self.assertEqual(m['timeline'][k], one['timeline'][k], (tm, k))
            for k in ('own', 'opp', 'dead', 'unknown'):
                self.assertLessEqual(abs(m['timeline']['phaseSeconds'][k]-one['timeline']['phaseSeconds'][k]),
                                     BOUND_PHASE_S*boundaries, (tm, k))
            cm = Counter(o['detector'] for o in m['opportunities'])
            c1 = Counter(o['detector'] for o in one['opportunities'])
            for det in set(cm) | set(c1):
                self.assertLessEqual(abs(cm[det]-c1[det]), BOUND_OPPS*boundaries, (tm, det, cm[det], c1[det]))
                spells = [len(SI.distinct([{'t0': o['t0']} for o in obs['opportunities'] if o['detector'] == det]))
                          for obs in (m, one)]
                self.assertLessEqual(abs(spells[0]-spells[1]), BOUND_SPELLS*boundaries, (tm, det, spells))
            for pid, rec in one['timeline']['players'].items():
                got = m['timeline']['players'][pid]
                self.assertLessEqual(abs(got['observedSeconds']-rec['observedSeconds']), 2.0*boundaries, pid)
                if rec['p90SpeedMS'] is not None:
                    self.assertLessEqual(abs(got['p90SpeedMS']-rec['p90SpeedMS']), .25, pid)
            # the role tier and the reports
            roles, roles1 = coaching.build_roles(m, d['settings'], tm), coaching.build_roles(one, d['settings'], tm)
            self.assertEqual(S.validate('roles', roles), [])
            rm = Counter(o['detector'] for o in roles['opportunities'])
            r1 = Counter(o['detector'] for o in roles1['opportunities'])
            for det in set(rm) | set(r1):
                self.assertLessEqual(abs(rm[det]-r1[det]), BOUND_OPPS*boundaries, (tm, det))
            rep, rep1 = coaching.interpret(m, roles, d['settings'], tm), coaching.interpret(one, roles1, d['settings'],
                                                                                            tm)
            verdicts = [sorted((f['observation'], f['verdict']) for k in ('strengths', 'needsWork', 'reviews',
                                                                          'described') for f in r[k])
                        for r in (rep, rep1)]
            self.assertEqual(verdicts[0], verdicts[1], tm)
        self.assertTrue(any(o['detector'] == 'wide_defender_advance'
                            for o in coaching.build_roles(d['merged']['A'], d['settings'], 'A')['opportunities']))

    def test_the_owned_possession_shares_cover_the_owned_frames(self):
        d = self.d
        fps = float(d['built'].result['video']['fps'])
        for p in d['parts']['A']:
            sh = p['extra']['possession']['shares']
            owned = (p['segment']['cut1']-p['segment']['cut0'])/fps
            self.assertAlmostEqual(sum(sh.values()), owned, places=1)
        # one whole-result record: the owned shares of every frame are the record's own shares
        r = d['built'].result
        rec = possession_amateur.compute(r)
        _, facts = MC.owned_possession(rec, r['frames'], r['frames'][0]['frame'], r['frames'][-1]['frame']+1, fps)
        for k in ('A', 'B', 'dead', 'unknown'):
            self.assertAlmostEqual(facts['shares'][k], rec['shares'][k], places=1)
        self.assertAlmostEqual(facts['coverage'], rec['quality']['coverage'], places=2)


class Context(unittest.TestCase):
    def test_context_frames_are_never_counted(self):
        """The views' context frames scrambled (every person moved and the ball dropped) with the possession record
        fixed: the merged obs is the same in every count, so the measurement reads the owned frames only."""
        d = Fixture.get()
        record = possession_amateur.compute(d['built'].result)
        fixed = lambda view: copy.deepcopy(record)                      # noqa: E731
        clean = MC.measure_segments(d['folder'], d['index'], 'A', d['settings'], shape_settings=d['shape'],
                                    possession=fixed)
        scrambled_dir = Path(tempfile.mkdtemp(prefix='fm_coaching_ctx_'))
        try:
            index = make_match(scrambled_dir, d['built'].result, halves(d['built'].result))
            a = scrambled_dir/'assembly'
            rng = np.random.default_rng(7)
            for s in index['segments']:
                view = assemble.load_view(a, s['view'])
                for f in view['frames']:
                    if s['cut0'] <= f['frame'] < s['cut1']:
                        continue
                    f['ball'] = None
                    for p in f.get('people') or ():
                        if isinstance(p.get('pitch'), list):
                            p['pitch'] = [float(rng.uniform(0, 1)), float(rng.uniform(0, 1))]
                assemble.write_view(a/s['view'], view)
            dirty = MC.measure_segments(scrambled_dir, index, 'A', d['settings'], shape_settings=d['shape'],
                                        possession=fixed)
        finally:
            shutil.rmtree(scrambled_dir, ignore_errors=True)
        m0 = MC.merge_obs(clean, shape_settings=d['shape'])
        m1 = MC.merge_obs(dirty, shape_settings=d['shape'])
        self.assertEqual([o['key'] for o in m0['opportunities']], [o['key'] for o in m1['opportunities']])
        self.assertEqual([o['relations'] for o in m0['opportunities']], [o['relations'] for o in m1['opportunities']])
        for k in ('samples', 'phaseSeconds', 'phaseSpans', 'dataQuality', 'completeShare', 'players'):
            self.assertEqual(m0['timeline'][k], m1['timeline'][k], k)
        self.assertEqual(m0['detectors'], m1['detectors'])


class Periods(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        b = scene()
        f0, f1 = b.result['frames'][0]['frame'], b.result['frames'][-1]['frame']+1
        mid = f0+(f1-f0)//2//10*10
        cls.folder = Path(tempfile.mkdtemp(prefix='fm_coaching_periods_'))
        # two periods: the 30 s after the first period's end are owned by nobody (the break)
        cls.index = make_match(cls.folder, b.result, [(f0, mid), (mid+300, f1)], periods=[0, 1])
        cls.settings = cs.settings_for(b, level=4)
        cls.built = MC.build_match_report(cls.folder, cls.index, 'A', cls.settings)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.folder, ignore_errors=True)

    def test_the_break_is_dead_time_and_never_play(self):
        keys = self.built['keys']
        obs = store.load_tier(store.tier_path(self.folder/'assembly', 'obs', keys))
        tl = obs['timeline']
        breaks = [s for s in tl['phaseSpans'] if 'source' not in s]
        self.assertEqual(len(breaks), 1)
        self.assertEqual(breaks[0]['state'], 'dead')
        a, b = self.index['segments'][0], self.index['segments'][1]
        self.assertTrue(a['end']-1e-6 <= breaks[0]['t0'] < breaks[0]['t1'] < b['start'])
        lo, hi = float(obs['segment']['t0']), float(obs['segment']['t1'])
        self.assertAlmostEqual(AS._phase_seconds(tl, 'unknown', lo, hi), tl['phaseSeconds']['unknown'], delta=.5)
        self.assertEqual(obs['roleInputs']['match']['breaks'], [[breaks[0]['t0'], breaks[0]['t1']]])

    def test_the_report_states_the_played_time_and_the_break(self):
        rep = self.built['record']['report']
        self.assertEqual(S.validate('report', rep), [])
        played = sum(s['end']-s['start'] for s in self.index['segments'])
        self.assertLess(played, rep['range']['seconds']-10)
        from soccer.coaching import report as RP
        self.assertTrue(rep['overview']['text'].startswith(f'This report covers {RP.duration_words(played)} of play'),
                        rep['overview']['text'])
        notice = next(n for n in rep['overview']['notices'] if n['code'] == 'match_segments')
        self.assertIn('2 periods', notice['text'])
        self.assertIn('between them is not analysed', notice['text'])


class Reports(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        b = scene()
        cls.folder = Path(tempfile.mkdtemp(prefix='fm_coaching_reports_'))
        cls.index = make_match(cls.folder, b.result, halves(b.result, 3))
        cls.settings = cs.settings_for(b, level=4)
        cls.before = MC.report_state(cls.folder, cls.index, 'A', cls.settings)
        cls.first = MC.build_match_report(cls.folder, cls.index, 'A', cls.settings)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.folder, ignore_errors=True)

    def test_states_tiers_and_records(self):
        st = self.before
        self.assertEqual((st['state'], st['tiers']['measurement'], st['report']), ('missing', 'missing', None))
        built = self.first
        self.assertTrue(built['measured'])
        again = MC.build_match_report(self.folder, self.index, 'A', self.settings)
        self.assertTrue(again['cached'])
        self.assertEqual(again['record']['report'], built['record']['report'])
        record = built['record']
        rep = record['report']
        self.assertEqual(S.validate('report', rep), [])
        obs = store.load_tier(store.tier_path(self.folder/'assembly', 'obs', built['keys']))
        roles = store.load_tier(store.tier_path(self.folder/'assembly', 'roles', built['keys']))
        self.assertEqual(S.cross_check(rep, obs, roles), [])
        self.assertEqual(rep['run'], 'match-synth')
        codes = {n['code'] for n in rep['overview']['notices']}
        self.assertIn('match_segments', codes)
        self.assertNotIn('match_phase', codes)          # the engine named the phase in every segment
        self.assertEqual(len(record['match']['segments']), 3)
        self.assertEqual(store.load_record_v2(store.tier_path(self.folder/'assembly', 'report', built['keys']))
                         ['keys']['interpretation'], built['keys']['ifp'])
        st = MC.report_state(self.folder, self.index, 'A', self.settings)
        self.assertEqual(st['state'], 'ready')
        self.assertEqual(st['fingerprints']['measurement'], built['keys']['mfp'])
        self.assertEqual(st['report']['fingerprints'], rep['fingerprints'])
        self.assertIsNotNone(st['inferred'])
        # the other team: its obs was measured in the same pass, its roles and report are made at once
        other = MC.report_state(self.folder, self.index, 'B', self.settings)
        self.assertEqual(other['state'], 'ready')
        # a level change is the interpretation tier only; the stored record of the old level is found stale
        lv = MC.build_match_report(self.folder, self.index, 'A', self.settings, level=6, age='U18')
        self.assertFalse(lv['measured'])
        self.assertEqual(lv['keys']['mfp'], built['keys']['mfp'])
        self.assertNotEqual(lv['keys']['ifp'], built['keys']['ifp'])
        state = MC.report_state(self.folder, self.index, 'A', MC.with_profile(self.settings, 'A', style='possession'),
                                make=False)
        self.assertEqual(state['state'], 'stale')
        self.assertIn('profile', state['reasonCodes'])

    def test_notices_validate_at_every_level(self):
        built = MC.build_match_report(self.folder, self.index, 'A', self.settings)
        for level in range(1, 8):
            settings = MC.with_profile(self.settings, 'A', level=level)
            out = MC.build_match_report(self.folder, self.index, 'A', settings)
            self.assertFalse(out['measured'], level)          # the format kept: the interpretation tier only
            rep = out['record']['report']
            self.assertEqual(S.validate('report', rep), [], level)
            codes = [n['code'] for n in rep['overview']['notices']]
            self.assertIn('match_segments', codes, level)
            self.assertEqual(len(codes), len(set(codes)))
        self.assertEqual(built['keys']['mfp'], out['keys']['mfp'])

    def test_coverage_notices_name_what_was_assessable(self):
        """Segments where nothing named who had the ball, and segments without a pitch keyframe: the notices say how
        many and where, at every level (the level's text gate accepts them)."""
        from soccer.coaching import profiles as PR
        obs = copy.deepcopy(Fixture.get()['merged']['A'])
        rows = obs['roleInputs']['match']['segments']
        rows[0].update(named=0.0, keyframe=False)
        for level in range(1, 8):
            profile = PR.resolve({'playingLevel': level, 'ageGroup': PR.default_age(level),
                                  'format': {'players': 11, 'goalkeeper': True}}, team='A')
            notices = {n['code']: n['text'] for n in MC.match_notices(obs, profile)}
            self.assertIn('match_segments', notices, level)
            self.assertIn('1 of the 2 segments had no pitch keyframe', notices.get('match_calibration', ''), level)
            self.assertIn('named in 1 of the 2 segments', notices.get('match_phase', ''), level)
        for row in rows:
            row['named'] = 0.0
        self.assertIn('not named in any of the 2 segments',
                      {n['code']: n['text'] for n in MC.match_notices(obs, profile)}['match_phase'])

    def test_background_build(self):
        folder = Path(tempfile.mkdtemp(prefix='fm_coaching_bg_'))
        try:
            index = make_match(folder, scene().result, halves(scene().result))
            done = []
            out = MC.start_build(folder, index, 'B', self.settings, on_done=lambda r, e: done.append((r, e)))
            self.assertEqual(out['state'], 'building')
            busy = MC.start_build(self.folder, self.index, 'A', MC.with_profile(self.settings, 'A', level=3))
            self.assertIn(busy['state'], ('busy', 'ready', 'building'))   # 'busy' unless the first build ended
            st = MC.report_state(folder, index, 'B', self.settings, make=False)
            self.assertIn(st['state'], ('building', 'ready'))
            self.assertTrue(MC.wait(120))
            self.assertIsNone(done[0][1])
            self.assertEqual(MC.report_state(folder, index, 'B', self.settings)['state'], 'ready')
            self.assertEqual(MC.start_build(folder, index, 'B', self.settings)['state'], 'ready')
        finally:
            shutil.rmtree(folder, ignore_errors=True)


class Pieces(unittest.TestCase):
    def test_match_settings(self):
        with tempfile.TemporaryDirectory() as tmp:
            video, match = Path(tmp)/'video', Path(tmp)/'video'/'matches'/'m1'
            (match/'assembly').mkdir(parents=True)
            (match/'assembly'/'match.json').write_text('{}', encoding='utf-8')
            settings, info = MC.match_settings(video, match)
            self.assertEqual((info['source'], settings['teams']), ('default', {'A': None, 'B': None}))
            block = {'ageGroup': 'U12', 'playingLevel': 3, 'format': {'players': 9, 'goalkeeper': True},
                     'roles': {'A-07': [{'role': 'back-1/3', 'from': None, 'to': None, 'set': '2026-10-20T10:00:00'}]}}
            store.save_team_settings(video, 'A', block)
            settings, info = MC.match_settings(video, match)
            self.assertEqual(info['source'], 'video')
            self.assertEqual(info['droppedRoles'], {'A': ['A-07']})
            self.assertEqual(settings['teams']['A']['roles'], {})
            self.assertEqual(settings['teams']['A']['playingLevel'], 3)
            self.assertEqual(S.validate('settings', settings), [])
            store.save_team_settings(MC.settings_dir(match), 'B', {'ageGroup': 'U14', 'playingLevel': 4})
            settings, info = MC.match_settings(video, match)
            self.assertEqual(info['source'], 'match')
            self.assertIsNone(settings['teams']['A'])
            self.assertEqual(settings['teams']['B']['ageGroup'], 'U14')

    def test_owned_result_cuts_every_per_frame_column(self):
        frames = [{'frame': k, 'time': k/10.0, 'people': []} for k in range(100, 120)]
        block = {'frameTimes': [f['time'] for f in frames], 'method': 'x',
                 'ids': {'A-01': {'team': 'A', 'x': list(range(20)), 'y': list(range(20)), 'kind': ['O']*20}}}
        view = {'frames': frames, 'timeline': block, 'stabilized': dict(block, anonymous={'7': {'x': list(range(20))}},
                                                                         flags=[{'id': 'A-01', 'time': 10.1}])}
        out = MC.owned_result(view, 105, 112)
        self.assertEqual([f['frame'] for f in out['frames']], list(range(105, 112)))
        self.assertEqual(out['timeline']['ids']['A-01']['x'], list(range(5, 12)))
        self.assertEqual(out['timeline']['ids']['A-01']['team'], 'A')
        self.assertEqual(out['stabilized']['anonymous']['7']['x'], list(range(5, 12)))
        self.assertEqual(out['stabilized']['flags'], view['stabilized']['flags'])
        self.assertEqual(len(view['frames']), 20)                       # the view itself is untouched

    def test_corrections_on_a_view(self):
        view = {'frames': [{'frame': 1, 'time': 0.1, 'people': [{'id': 'A-03', 'team': 'A', 'state': 'confirmed'},
                                                                 {'id': 'A-09', 'team': 'A', 'state': 'confirmed'},
                                                                 {'id': 'B-02', 'team': 'B', 'state': 'confirmed'}],
                            'visibility': {'A-03': 1, 'A-09': 1}}],
                'match': {'players': [{'id': 'A-01', 'team': 'A'}, {'id': 'A-03', 'team': 'A'},
                                      {'id': 'A-09', 'team': 'A'}, {'id': 'B-02', 'team': 'B'}]},
                'roster': {'slots': [{'id': 'A-01', 'team': 'A', 'firstSeen': 50.0},
                                     {'id': 'A-03', 'team': 'A', 'firstSeen': 10.0},
                                     {'id': 'A-09', 'team': 'A', 'firstSeen': 5.0}]},
                'timeline': {'frameTimes': [0.0, 0.1], 'ids': {'A-01': {'team': 'A', 'x': [None, .5], 'y': [None, .5]},
                                                              'A-03': {'team': 'A', 'x': [.2, None], 'y': [.3, None]}}},
                'issues': [{'id': 'A-03', 'time': 0.1, 'reason': 'Crossed'}], 'events': [{'playerId': 'A-03'}]}
        fixes = {'aliases': {'A-03': 'A-01'}, 'excluded': ['A-09'], 'teams': {'B-02': 'A'}}
        MC.apply_fixes(view, fixes)
        people = view['frames'][0]['people']
        self.assertEqual([p['id'] for p in people], ['A-01', None, 'B-02'])
        self.assertEqual((people[1]['state'], people[1]['team']), ('nontarget', None))
        self.assertEqual(people[2]['team'], 'A')
        self.assertEqual(sorted(e['id'] for e in view['match']['players']), ['A-01', 'B-02'])
        self.assertEqual(view['roster']['slots'], [{'id': 'A-01', 'team': 'A', 'firstSeen': 10.0, 'left': None}])
        self.assertEqual(view['timeline']['ids']['A-01']['x'], [.2, .5])
        self.assertEqual((view['issues'][0]['id'], view['events'][0]['playerId']), ('A-01', 'A-01'))
        self.assertEqual(view['frames'][0]['visibility'], {'A-01': 1})

    def test_segment_spans_are_clipped_to_the_owned_range(self):
        spans = {'version': 1, 'kind': 'coach-possession-spans', 'source': 'coach',
                 'spans': [{'t0': 10.0, 't1': 30.0, 'state': 'A'}, {'t0': 31.0, 't1': 32.0, 'state': 'dead'},
                           {'t0': 40.0, 't1': 50.0, 'state': 'B'}]}
        out = MC.segment_spans(spans, 20.0, 35.0)
        self.assertEqual(out['spans'], [{'t0': 20.0, 't1': 30.0, 'state': 'A'},
                                        {'t0': 31.0, 't1': 32.0, 'state': 'dead'}])
        self.assertEqual(S.validate('coach_spans', out), [])
        self.assertIsNone(MC.segment_spans(spans, 60.0, 70.0))

    def test_merge_refusals(self):
        d = Fixture.get()
        parts = d['parts']['A']
        overlap = copy.deepcopy(parts)
        overlap[1]['segment']['cut0'] -= 10
        with self.assertRaises(MC.MergeError):
            MC.merge_obs(overlap)
        with self.assertRaises(MC.MergeError):
            MC.merge_obs([parts[0], d['parts']['B'][1]])
        mixed = copy.deepcopy(parts)
        pq = mixed[1]['obs']['timeline']['phaseQuality']
        pq['available'] = False
        pq['imageEstimate'] = {'version': 'possession-amateur-1', 'used': True, 'reason': None, 'agreement': 1.0,
                               'checkedSeconds': 30.0, 'offeredSeconds': 10.0}
        with self.assertRaises(MC.MergeError):
            MC.merge_obs(mixed)
        with self.assertRaises(MC.MergeError):
            MC.merge_obs(parts, shape_settings={'direction': {'A': 'right'}})

    def test_a_formation_change_on_a_cut_suppresses_like_one_run(self):
        """Two segments whose formation periods read 4-4-2 (clear, 100 s) up to the cut and 3-5-2 (clear, 100 s) from
        it: neither segment sees a change; the merge flags the opportunities within 15 s of the cut, and only them."""
        def part(k, t0, t1, label):
            q = {'index': 0, 't0': t0, 't1': t1, 'label': label, 'confidence': 'clear', 'corrected': False}
            return {'segment': {'index': k, 'period': 0, 'start': t0, 'end': t1, 'cut0': k, 'cut1': k+1},
                    'obs': {'timeline': {'samples': int((t1-t0)*5), 'step': .2},
                            'roleInputs': {'context': {'periods': [q]}}}}
        parts = [part(0, 0.0, 100.0, '4-4-2'), part(1, 100.0, 200.0, '3-5-2')]
        opp = lambda t: {'relations': {'t': [t, t+1.0]}, 'intentSignals': {'formationChangeNear': False}}  # noqa
        merged = {'roleInputs': {'context': {'periods': [dict(parts[0]['obs']['roleInputs']['context']['periods'][0]),
                                                         dict(parts[1]['obs']['roleInputs']['context']['periods'][0],
                                                              index=1)]},
                                 'detect': {'samples': {'t': [50.0, 90.0, 110.0, 150.0], 'fcn': [False]*4}}},
                  'opportunities': [opp(40.0), opp(88.0), opp(113.0), opp(160.0)]}
        windows = MC._boundary_formation_changes(merged, parts, [0, 1])
        self.assertEqual(windows, [[85.0, 115.0]])
        self.assertEqual([o['intentSignals']['formationChangeNear'] for o in merged['opportunities']],
                         [False, True, True, False])
        self.assertEqual(merged['roleInputs']['detect']['samples']['fcn'], [False, True, True, False])


if __name__ == '__main__':
    unittest.main()
