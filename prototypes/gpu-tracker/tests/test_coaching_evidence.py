"""The evidence of the coaching observations (soccer/coaching/evidence.py): the strongest distinct moments are
selected (at least 10 s apart, at most three, best tier and coverage first), each carries its exact frame, absolute
and segment time and the browser seek time (frame + 0.5) / fps, the overlay tier follows the measured rule (world only
at confidence >= .7 with the paint checked within +-0.5 s, never on keyless, small-scene or lens-bent footage; pinned
from .35; image below; never world where the registration chains disagree by more than 3 m), overlays draw only
what their tier allows, diagrams show observed positions of real ids (a team width or length as a band with a bracket
between the extreme players outside the team), an evidence record names only players observed on its frame, a typical
moment lies inside the measurement's interquartile range, a formation moment needs the whole lineup in view for a
diagram, ids are stable and insufficient observations have no evidence.

Synthetic results from test_coaching_detectors (tests/team_shape_synth.py layouts) with a simple affine camera: world
metres -> pixels x = 16 X + 100, y = 10 Y + 200, the painted lines projected through it (or shifted to fail the
paint check), and people boxes standing on their projected feet.

Run:  python -m unittest test_coaching_evidence          (from this directory)
"""
import copy
import json
import math
import os
import sys
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (ROOT, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)
import soccer.coaching as coaching  # noqa: E402
from soccer import field  # noqa: E402
from soccer.coaching import detectors as det, evidence as E, render, series as cs  # noqa: E402
from test_coaching_detectors import (FPS, SYNTHETIC_REFERENCE, build, gap_pushes, pick, quantiles,  # noqa: E402
                                     ramp, rates, synthetic_reference)

W, H = 1920, 1080
CAMERA = np.array([[16.0, 0.0, 100.0], [0.0, 10.0, 200.0], [0.0, 0.0, 1.0]])    # world metres -> pixels
SIZE = field.PitchSize()
PAINTED = ('far touchline', 'near touchline', 'halfway line', 'left penalty area front', 'right penalty area front')


def markings(offset=0.0):
    """The painted lines as a tick frame's detected segments (normalized), shifted by offset pixels."""
    out = []
    for name, a, b in field.lines(SIZE):
        if name not in PAINTED:
            continue
        pa, pb = field.project(CAMERA, [a])[0], field.project(CAMERA, [b])[0]
        out.append([(pa[0]+offset)/W, (pa[1]+offset)/H, (pb[0]+offset)/W, (pb[1]+offset)/H])
    return out


def calibrate(result, confidence=lambda t: .9, offset=lambda t: 0.0, anchors=('landmarks',), lens=None):
    """Give a synthetic result a registration: per frame its confidence and H, paint on every frame (each a tick),
    boxes standing on the people's projected feet, and the run summary's anchors / lens."""
    for f in result['frames']:
        t = f['time']
        f['calibration'] = {'confidence': confidence(t), 'H': CAMERA.ravel().tolist(), 'disagreement': None,
                            'source': 'carried'}
        f['calibrated'] = True
        f['pitch'] = {'markings': markings(offset(t))}
        for p in f['people']:
            X, Y = p['pitch'][0]*SIZE.length, p['pitch'][1]*SIZE.width
            fx, fy = field.project(CAMERA, [(X, Y)])[0]
            p['box'] = [(fx-10)/W, (fy-50)/H, 20/W, 50/H]
    reg = {'anchors': [{'source': s} for s in anchors]}
    if lens is not None:
        reg['lens'] = lens
    result['summary'] = {'pitch': {'registration': reg}}
    return result


def analysis(result, reference=SYNTHETIC_REFERENCE, team='A', **kw):
    out = coaching.build_observations(result, team=team, reference=reference, keep_series=True, **kw)
    return out, {e['id']: e for e in out['evidence']}


class Tiers(unittest.TestCase):
    """One result, four registration spells: good paint at .9, paint 20 px off at .9, .5, .3."""

    @classmethod
    def setUpClass(cls):
        cls.result = calibrate(build(seconds=80.0),
                               confidence=lambda t: .3 if t >= 60 else (.5 if t >= 40 else .9),
                               offset=lambda t: 20.0 if 20 <= t < 40 else 0.0)
        cls.cd = cs.build_series(cls.result)
        cls.fx = E.Frames(cls.cd)

    def tier(self, t):
        return self.fx.tier(int(np.argmin(np.abs(self.cd.t-t))))

    def test_the_paint_check(self):
        good = E.paint_check(CAMERA, SIZE, {'pitch': {'markings': markings()}}, W, H)
        self.assertEqual((good['segments'], good['matched'], good['wideN']), (5, 5, 5))
        self.assertAlmostEqual(good['wide'], 0.0, places=6)
        off = E.paint_check(CAMERA, SIZE, {'pitch': {'markings': markings(20.0)}}, W, H)
        self.assertAlmostEqual(off['wide'], 20.0, places=4)
        self.assertEqual(E.paint_check(CAMERA, SIZE, {'pitch': {'markings': []}}, W, H)['wide'], None)

    def test_world_needs_confidence_and_the_paint_checked(self):
        tier, info = self.tier(10.0)
        self.assertEqual(tier, 'world')
        self.assertEqual((info['confidence'], info['paintTicks']), (.9, 5))
        self.assertLessEqual(info['paintResidualPx'], E.PAINT_MAX_RESIDUAL_PX)
        self.assertEqual(self.tier(30.0)[0], 'pinned')          # confident but the paint is 20 px off
        self.assertEqual(self.tier(30.0)[1]['paintResidualPx'], 20.0)
        self.assertEqual(self.tier(19.8)[0], 'pinned')          # a bad tick within 0.5 s is enough to refuse
        self.assertEqual(self.tier(50.0)[0], 'pinned')          # MEDIUM
        self.assertEqual(self.tier(70.0)[0], 'image')           # below .35

    def test_never_world_on_keyless_small_scene_or_lens_bent_footage(self):
        def tier_of(res):
            cd = cs.build_series(res)
            return E.Frames(cd).tier(int(np.argmin(np.abs(cd.t-10.0))))[0]
        self.assertEqual(tier_of(calibrate(build(seconds=30.0))), 'world')
        self.assertEqual(tier_of(calibrate(build(seconds=30.0), anchors=('auto-midfield',))), 'pinned')
        self.assertEqual(tier_of(calibrate(build(seconds=30.0), lens={'k': .58, 'gain': .32})), 'pinned')

        def small(k, t, f):
            f['profile'] = {'smallScene': True}
        self.assertEqual(tier_of(calibrate(build(seconds=30.0, edit=small))), 'pinned')


class Selection(unittest.TestCase):
    """Five spells of a large back line to midfield gap: two of them 6 s apart, one with a forward unseen."""
    SPELLS = [(20.0, 30.0), (36.0, 46.0), (80.0, 90.0), (130.0, 140.0), (180.0, 190.0)]

    @classmethod
    def setUpClass(cls):
        times = np.arange(0, 240.0, 1/FPS)
        visible = {'A': np.ones((len(times), 10), bool)}
        visible['A'][(times >= 125) & (times < 145), 9] = False      # A-10 (a forward) unseen in the fourth spell
        cls.result = build(overrides_a=gap_pushes(cls.SPELLS), visible=visible)
        cls.out, cls.ev = analysis(cls.result)
        cls.gap = pick(cls.out['observations'], 'unit_gap_dm')[0]

    def test_the_strongest_distinct_moments_are_selected(self):
        o = self.gap
        self.assertEqual(o['occurrenceCount'], 5)
        evs = [self.ev[e] for e in o['evidence']]
        self.assertTrue(1 <= len(evs) <= E.MAX_EVIDENCE)
        self.assertEqual(len(evs), 3)
        times = [e['time'] for e in evs]
        for i in range(len(times)):
            for j in range(i+1, len(times)):
                self.assertGreaterEqual(abs(times[i]-times[j]), E.MIN_SEPARATION_SECONDS)
        # each moment lies inside its own occurrence, at the occurrence's peak when nothing better exists
        occ = {x['id']: x for x in o['occurrences']}
        for e in evs:
            x = occ[e['occurrence']]
            self.assertTrue(x['t0'] <= e['time'] <= x['t1'])
            self.assertEqual(e['frame'], x['peakFrame'])
        # the spell with a player unseen ranks below the fully seen ones
        chosen = {occ[e['occurrence']]['t0'] for e in evs}
        self.assertFalse(any(125 <= t <= 145 for t in chosen))

    def test_moments_closer_than_ten_seconds_give_one(self):
        cd = self.out['series']
        fx = E.Frames(cd)
        s = cd.teams['A']
        at = lambda t: int(np.argmin(np.abs(cd.t-t)))          # noqa: E731
        players = s.units[at(60.0)]['D']+s.units[at(60.0)]['M1']
        hint = {'preset': 'defensive_gap', 'metric': 'gapDM', 'units': ('D', 'M1'), 'player': None, 'typical': None,
                'occurrences': [{'occurrence': f'occ-{k}', 'peak': at(t), 'candidates': [at(t)], 'values': [v],
                                 'players': players, 'strength': v}
                                for k, (t, v) in enumerate(((60.0, 22.0), (66.0, 25.0), (100.0, 21.0), (150.0, 20.5),
                                                            (200.0, 20.2)))]}
        picked = E.select(fx, s, hint)
        self.assertEqual([p['occurrence'] for p in picked], ['occ-1', 'occ-2', 'occ-3'])

    def test_insufficient_and_suppressed_observations_have_no_evidence(self):
        for o in self.out['observations']:
            if o['classification'] == 'insufficient':
                self.assertEqual(o['evidence'], [], o['detector'])
        obs = det.detect(self.out['series'], 'A', SYNTHETIC_REFERENCE)
        for o in obs:
            o['status'] = 'suppressed'
        self.assertEqual(E.attach(obs, self.out['series']), [])
        self.assertFalse(any('_ev' in o for o in obs))

    def test_strengths_get_typical_moments(self):
        out, ev = analysis(build(seconds=120.0))
        s = pick(out['observations'], 'unit_gap_dm')[0]
        self.assertEqual(s['classification'], 'strength')
        self.assertTrue(1 <= len(s['evidence']) <= E.TYPICAL_EVIDENCE)
        for e in s['evidence']:
            self.assertEqual((ev[e]['kind'], ev[e]['occurrence']), ('typical', None))

    def test_ids_are_stable(self):
        again, _ = analysis(copy.deepcopy(self.result))
        self.assertEqual([e['id'] for e in self.out['evidence']], [e['id'] for e in again['evidence']])
        for e in self.out['evidence']:
            self.assertRegex(e['id'], r'^ev-[0-9a-f]{12}$')
        self.assertEqual(len({e['id'] for e in self.out['evidence']}), len(self.out['evidence']))
        json.dumps({k: self.out[k] for k in ('observations', 'evidence', 'meta')}, allow_nan=False)


class Time(unittest.TestCase):
    """A segment that starts 100 s into the video."""

    @classmethod
    def setUpClass(cls):
        cls.result = build(seconds=120.0, start=100.0, overrides_a=gap_pushes([(130.0, 142.0), (170.0, 182.0)]))
        cls.out, cls.ev = analysis(cls.result)

    def test_absolute_and_segment_times_and_the_seek(self):
        frames = {f['frame']: f for f in self.result['frames']}
        self.assertTrue(self.out['evidence'])
        for e in self.out['evidence']:
            f = frames[e['frame']]
            self.assertEqual(e['time'], round(f['time'], 3))
            self.assertEqual(e['frame'], int(round(f['time']*FPS)))
            self.assertAlmostEqual(e['segmentTime'], f['time']-100.0, places=3)
            self.assertAlmostEqual(e['seek'], (e['frame']+.5)/FPS, places=4)
            self.assertEqual(e['fps'], FPS)
            self.assertTrue(100.0 <= e['time'] <= self.result['frames'][-1]['time'])
            self.assertEqual(e['diagram']['frame'], e['frame'])
        gap = pick(self.out['observations'], 'unit_gap_dm')[0]
        self.assertEqual(gap['window']['t0'], 100.0)
        self.assertEqual(gap['window']['segmentT0'], 0.0)
        self.assertAlmostEqual(gap['window']['segmentT1'], gap['window']['t1']-100.0, places=3)
        for x in gap['occurrences']:
            self.assertAlmostEqual(x['segmentT0'], x['t0']-100.0, places=3)
            self.assertEqual(x['frame0'], int(round(x['t0']*FPS)))
            self.assertEqual(x['peakFrame'], int(round(x['peakT']*FPS)))
        self.assertEqual(self.out['meta']['segmentStart'], 100.0)

    def test_a_custom_range_is_clamped_to_the_segment(self):
        out = coaching.build_observations(self.result, team='A', reference=SYNTHETIC_REFERENCE, t0=50.0, t1=500.0)
        w = out['observations'][0]['window']
        self.assertEqual((w['t0'], w['segmentT0']), (100.0, 0.0))
        self.assertEqual(w['t1'], round(self.result['frames'][-1]['time'], 3))
        self.assertEqual(out['meta']['range']['t0'], 100.0)


class Pictures(unittest.TestCase):
    """Diagrams and overlays of a gap observation and an isolation observation on a calibrated result."""

    @classmethod
    def setUpClass(cls):
        spells = [(20.0, 32.0), (70.0, 82.0), (120.0, 132.0), (170.0, 182.0)]

        def overrides(times, pos):
            pos = gap_pushes(spells)(times, pos)
            pos[:, 9, 0] += ramp(times, [(45.0, 60.0), (95.0, 110.0), (145.0, 160.0)], 25.0)
            return pos
        ref = synthetic_reference(nearestTeammate=quantiles(14.0, 23.0, 25.0))
        ref['metrics']['nearestTeammate']['episodes']['all'] = rates(p50=.1, p90=.2)
        cls.ref = ref
        cls.result = calibrate(build(seconds=200.0, overrides_a=overrides),
                               confidence=lambda t: .5 if (t % 50) < 25 else .9)
        cls.out, cls.ev = analysis(cls.result, reference=ref)
        cls.cd = cls.out['series']

    def test_diagrams_show_observed_positions_of_real_ids(self):
        s, other = self.cd.teams['A'], self.cd.teams['B']
        ids = set(s.ids) | set(other.ids)
        for e in self.out['evidence']:
            d = e['diagram']
            k = int(np.argmin(np.abs(self.cd.t-e['time'])))
            pos = {**s.positions(k), **other.positions(k)}
            players = {el['id']: el for el in d['elements'] if el['type'] == 'player'}
            self.assertTrue(players)
            for pid, el in players.items():
                self.assertIn(pid, ids)
                self.assertEqual(el['kind'], 'O')
                self.assertAlmostEqual(el['x'], pos[pid][0], places=4)
                self.assertAlmostEqual(el['y'], pos[pid][1], places=4)
            for el in d['elements']:
                if el['type'] == 'unit':
                    self.assertTrue(set(el['ids']) <= set(players))
                if el['type'] == 'arrow':
                    ts = [p[2] for p in el['points']]
                    self.assertTrue(all(b-a <= E.ARROW_GAP_SECONDS+1e-9 for a, b in zip(ts, ts[1:])))
                    self.assertLessEqual(ts[-1], e['time']+1e-6)
            self.assertEqual((d['team'], d['attack'], d['metresEstimated']), ('A', 'right', True))
            self.assertEqual(d['pitch'], {'length': 105.0, 'width': 68.0})
            # every element is one the renderer knows, and nothing is skipped
            drawn = render.diagram_ops(d)
            self.assertEqual(drawn['skipped'], [], drawn['skipped'])

    def test_the_gap_diagram_measures_the_gap(self):
        gap = pick(self.out['observations'], 'unit_gap_dm')[0]
        self.assertEqual(gap['classification'], 'improvement')
        e = self.ev[gap['evidence'][0]]
        d = e['diagram']
        self.assertEqual(d['preset'], 'defensive_gap')
        units = {el['unit']: el['ids'] for el in d['elements'] if el['type'] == 'unit'}
        self.assertEqual(set(units), {'D', 'M1'})
        measure = [el for el in d['elements'] if el['type'] == 'measure'][0]
        self.assertAlmostEqual(abs(measure['b'][0]-measure['a'][0])*105.0, e['metrics']['gapDM'], delta=.15)
        self.assertEqual(measure['text'], f"{e['metrics']['gapDM']:.1f} m (est.)")
        self.assertTrue(any(el['type'] == 'zone' for el in d['elements']))

    def test_overlays_draw_only_what_their_tier_allows(self):
        tiers = set()
        for e in self.out['evidence']:
            ov = e['videoOverlay']
            self.assertIsNotNone(ov)
            self.assertEqual((ov['tier'], ov['width'], ov['height']), (e['tier'], W, H))
            tiers.add(ov['tier'])
            kinds = {el['type'] for el in ov['elements']}
            self.assertTrue(kinds <= render.TIER_ELEMENTS[ov['tier']], (ov['tier'], kinds))
            if ov['tier'] != 'world':
                self.assertNotIn('zone', kinds)
            k = int(np.argmin(np.abs(self.cd.t-e['time'])))
            people = {p['id']: p for p in self.result['frames'][int(self.cd.shape.observed.frame_index[k])]['people']}
            for el in ov['elements']:
                if el['type'] == 'player':
                    bx, by, bw, bh = people[el['id']]['box']
                    self.assertAlmostEqual(el['x'], (bx+bw/2)*W, places=0)       # the box's bottom centre
                    self.assertAlmostEqual(el['y'], (by+bh)*H, places=0)
            ops = render.overlay_ops(ov, 0, 0, 320, 180)
            self.assertEqual(ops['skipped'], [], ops['skipped'])
        self.assertEqual(tiers, {'world'})              # the confident spells are preferred over the MEDIUM ones
        gap = pick(self.out['observations'], 'unit_gap_dm')[0]
        world = self.ev[gap['evidence'][0]]['videoOverlay']
        self.assertTrue({'zone', 'measure', 'unit'} <= {el['type'] for el in world['elements']})

    def test_an_image_tier_overlay_has_no_pitch_geometry(self):
        res = calibrate(build(seconds=120.0, overrides_a=gap_pushes([(30.0, 42.0), (80.0, 92.0)])),
                        confidence=lambda t: .5)
        out, ev = analysis(res)
        fx = E.Frames(out['series'])
        gap = pick(out['observations'], 'unit_gap_dm')[0]
        hint = {'preset': 'defensive_gap', 'metric': 'gapDM', 'units': ('D', 'M1'), 'player': None}
        s = int(np.argmin(np.abs(out['series'].t-gap['occurrences'][0]['peakT'])))
        players = gap['occurrences'][0]['players']
        image = E.video_overlay(fx, gap, hint, s, players, 'image')
        self.assertEqual({el['type'] for el in image['elements']}, {'player'})
        pinned = E.video_overlay(fx, gap, hint, s, players, 'pinned')
        self.assertEqual({el['type'] for el in pinned['elements']}, {'player', 'unit'})
        self.assertEqual(ev[gap['evidence'][0]]['tier'], 'pinned')
        self.assertEqual(ev[gap['evidence'][0]]['confidence'], 'medium')

    def test_the_isolation_moment_shows_the_player_and_his_nearest_team_mate(self):
        iso = [o for o in self.out['observations'] if o['detector'] == 'isolation' and o['players'] == ['A-10']]
        self.assertEqual(len(iso), 1)
        e = self.ev[iso[0]['evidence'][0]]
        d = e['diagram']
        self.assertEqual(d['preset'], 'player_movement')
        emph = [el['id'] for el in d['elements'] if el['type'] == 'player' and el['emphasis']]
        self.assertIn('A-10', emph)
        self.assertEqual(len(emph), 2)
        m = [el for el in d['elements'] if el['type'] == 'measure'][0]
        dist = math.hypot((m['b'][0]-m['a'][0])*105.0, (m['b'][1]-m['a'][1])*68.0)
        self.assertAlmostEqual(dist, e['metrics']['nearestTeammate'], delta=.5)
        self.assertTrue(any(el['type'] == 'arrow' and el['id'] == 'A-10' for el in d['elements']))
        self.assertEqual(e['players'], ['A-10'])


class Rules(unittest.TestCase):
    """The evidence rules the fact check asked for: the registration's 3 m rule for world overlays, observed players
    only, typical moments inside the IQR, formation diagrams only with the whole lineup, a width bracket."""

    @classmethod
    def setUpClass(cls):
        cls.result = calibrate(build(seconds=120.0))
        cls.out, cls.ev = analysis(cls.result)
        cls.cd = cls.out['series']
        cls.fx = E.Frames(cls.cd)
        cls.s = cls.cd.teams['A']

    def at(self, t):
        return int(np.argmin(np.abs(self.cd.t-t)))

    def test_world_needs_the_chains_within_three_metres(self):
        res = calibrate(build(seconds=30.0))
        for f in res['frames']:
            f['calibration']['disagreement'] = 4.5 if f['time'] < 15.0 else 2.0
        cd = cs.build_series(res)
        fx = E.Frames(cd)

        def tier(t):
            return fx.tier(int(np.argmin(np.abs(cd.t-t))))
        self.assertEqual(tier(8.0)[0], 'pinned')
        self.assertEqual(tier(8.0)[1]['disagreement'], 4.5)
        self.assertEqual(tier(22.0)[0], 'world')

    def test_an_evidence_record_names_only_players_observed_on_its_frame(self):
        gap = pick(self.out['observations'], 'unit_gap_dm')[0]
        hint = {'preset': 'defensive_gap', 'metric': 'gapDM', 'units': ('D', 'M1'), 'player': None, 'typical': None,
                'occurrences': []}
        s = self.at(60.0)
        seen = list(self.s.positions(s))
        item = {'s': s, 'occurrence': None, 'players': seen[:2]+['A-99'], 'complete': True}
        ev = E.build(self.fx, gap, hint, item)
        self.assertEqual(ev['players'], seen[:2])
        for e in self.out['evidence']:
            k = self.at(e['time'])
            self.assertTrue(set(e['players']) <= set(self.cd.teams[e['diagram']['team']].positions(k)), e['id'])

    def test_a_typical_moment_lies_inside_the_interquartile_range(self):
        cands = [self.at(t) for t in np.arange(10.0, 70.0, .2)]
        values = list(np.linspace(0.0, 100.0, len(cands)))          # the IQR is 25..75, the median 50
        hint = {'preset': 'width_depth', 'metric': 'width', 'units': None, 'player': None, 'occurrences': [],
                'typical': {'candidates': cands, 'values': values, 'target': 50.0}}
        picked = E.select(self.fx, self.s, hint)
        self.assertTrue(picked)
        value = dict(zip(cands, values))
        for p in picked:
            self.assertTrue(25.0 <= value[p['s']] <= 75.0, value[p['s']])
        # a stretch whose samples are all beyond the IQR never gives the typical moment, however long it is
        spike = [100.0 if i < 60 else 10.0+(i % 7) for i in range(len(cands))]     # 20% of the time, one stretch
        hint['typical'] = {'candidates': cands, 'values': spike, 'target': float(np.median(spike))}
        value = dict(zip(cands, spike))
        picked = E.select(self.fx, self.s, hint)
        self.assertTrue(picked)
        self.assertTrue(all(value[p['s']] < 100.0 for p in picked))

    def test_a_formation_moment_needs_the_whole_lineup_for_a_diagram(self):
        times = np.arange(0, 120.0, 1/FPS)
        visible = {'A': np.ones((len(times), 10), bool)}
        visible['A'][times < 60.0, 3] = False                        # A-04 unseen in the first minute
        cd = cs.build_series(build(seconds=120.0, visible=visible))
        fx, s = E.Frames(cd), cd.teams['A']
        early = [int(i) for i in np.flatnonzero(cd.t < 60.0)]
        late = [int(i) for i in np.flatnonzero(cd.t >= 60.0)]
        hint = {'preset': 'formation_shape', 'metric': None, 'units': ('D', 'M', 'F'), 'typical': None,
                'occurrences': [{'occurrence': 'occ-a', 'peak': early[len(early)//2], 'candidates': early+late[:50],
                                 'values': [0.0]*(len(early)+50), 'players': [], 'strength': 1.0},
                                {'occurrence': 'occ-b', 'peak': early[10], 'candidates': early,
                                 'values': [0.0]*len(early), 'players': [], 'strength': 1.0}]}
        picked = {p['occurrence']: p for p in E.select(fx, s, hint)}
        self.assertTrue(picked['occ-a']['complete'])
        self.assertGreaterEqual(cd.t[picked['occ-a']['s']], 60.0)         # moved to where all ten are in view
        self.assertFalse(picked['occ-b']['complete'])
        obs = {'id': 'obs-x', 'team': 'A', 'pattern': 'formation_change', 'detector': 'formation_stability',
               'metrics': {'label': '4-4-2'}, 'occurrences': [], 'confidence': {'overall': 'medium'}}
        self.assertIsNone(E.build(fx, obs, hint, picked['occ-b'])['diagram'])
        self.assertIsNotNone(E.build(fx, obs, hint, picked['occ-a'])['diagram'])

    def test_a_team_width_is_a_bracket_between_the_extreme_players(self):
        width = pick(self.out['observations'], 'compactness', 'team_width')[0]
        self.assertTrue(width['evidence'])
        for eid in width['evidence']:
            d = self.ev[eid]['diagram']
            players = {el['id']: el for el in d['elements'] if el['type'] == 'player' and el['team'] == 'A'}
            band = [el for el in d['elements'] if el['type'] == 'band']
            measure = [el for el in d['elements'] if el['type'] == 'measure']
            self.assertEqual(len(band), 1)
            self.assertEqual(band[0]['axis'], 'y')
            self.assertEqual(len(measure), 1)
            m = measure[0]
            self.assertTrue(m['bracket'])
            a, b = (players[p] for p in m['ids'])
            self.assertEqual(sorted((m['a'][1], m['b'][1])), sorted((a['y'], b['y'])))
            self.assertEqual((band[0]['from'], band[0]['to']), tuple(sorted((a['y'], b['y']))))
            xs = [el['x'] for el in players.values()]
            self.assertTrue(m['a'][0] > max(xs) or m['a'][0] < min(xs))         # outside the team
            self.assertAlmostEqual(abs(a['y']-b['y'])*68.0, self.ev[eid]['metrics']['width'], delta=.2)
            self.assertEqual(render.diagram_ops(d)['skipped'], [])


if __name__ == '__main__':
    unittest.main()
