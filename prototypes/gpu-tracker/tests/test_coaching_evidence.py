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

v2 (evidence.moments / moments_from_obs / annotate, DESIGN v2 5.4-5.6) on EVAL's synthetic scenes through the real
tiers (coaching_detect_fixtures): one moment per opportunity at a sample of its dominant outcome (an identity doubt in
the run is never hidden), frames exact and the scene the timeline's, the frame tiers and pixel anchors the role tier
needs, the role tier's moments rebuilt from the obs file, the moments independent of the coaching profile; the
annotations: every cited moment has valid evidence, elements follow the tier table and the budget, labels follow the
level's naming, relation values its register, captions pass explain's gate, a suggested zone only on the
alternative's moment, projected through the frame's homography, kept off the video by the disagreement rule, off the
picture or below world (DESIGN 13.1 'calibration below world'), a direction arrow no longer than the reach.

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


# =====================================================================================================================
# v2 (DESIGN v2 5.4-5.6): evidence.moments / moments_from_obs / annotate
# =====================================================================================================================
import coaching_detect_fixtures as FX  # noqa: E402  (EVAL's scenes through the real tiers; cached per process)
from soccer.coaching import assess as AS, contextual as CX, explain as EX, schema as S  # noqa: E402
from soccer.coaching import situations as SI  # noqa: E402

_V2 = {}


def v2_tiers(scene_name, variant=None):
    """(built, settings, cd, timeline, obs, roles, anchors) of an EVAL scene through build_observations / build_roles
    with the REAL moments stage, the anchors stored as the orchestrator is asked to (obs['roleInputs']['evidence']);
    cached for the process (callers that change a record copy it)."""
    key = (scene_name, variant)
    if key not in _V2:
        b, st, cd, poss = FX.prepared(scene_name, variant)
        held = {}

        def moments(cd_, tl, opps):
            out = E.moments(cd_, tl, opps)
            held['tl'], held['ri'] = tl, out['roleInputs']
            return out
        stages = {'moments': moments}
        if b.coach_spans is not None:
            spans = b.coach_spans
            stages['situations'] = lambda cd_, p_, tm, meas: SI.build_timeline(cd_, p_, tm, meas, coach_spans=spans)
        obs = coaching.build_observations(b.result, 'A', coaching=st, series=cd, possession=poss, result_sha='0'*64,
                                          stages=stages)
        obs['roleInputs'].update(held['ri'])
        roles = coaching.build_roles(obs, st, 'A')
        _V2[key] = (b, st, cd, held['tl'], obs, roles, held['ri']['evidence'])
    return _V2[key]


def sample_of(tl, t):
    return int(np.argmin(np.abs(np.asarray(tl.t, float)-t)))


def d6_opportunity(widths, depths=None, channels=None):
    """A hand-made attacking_shape opportunity (outcome per sample: width share >= .55, depth share >= .30, three
    channels -> demonstrated, else not)."""
    n = len(widths)
    depths = depths or [.35]*n
    channels = channels or [3]*n
    t = [round(100.0+.2*k, 3) for k in range(n)]
    return {'detector': 'attacking_shape', 'team': 'A', 'abstain': None,
            'relations': {'t': t, 'widthShare': widths, 'depthShare': depths, 'channelsOccupied': channels,
                          'widthM': [w*68.0 if w is not None else None for w in widths],
                          'depthM': [d*105.0 if d is not None else None for d in depths]},
            'actors': {'primary': [], 'supporting': [], 'opponents': [], 'carrier': None}}


SCALE = {'lengthM': 105.0, 'widthM': 68.0, 'source': 'assumed', 'metresUsable': True}


class V2Selection(unittest.TestCase):
    """Which sample of an opportunity is its moment (key_run, _choose): pure rules on hand-made records."""

    def test_the_dominant_outcome_decides_the_run(self):
        good, bad = .7, .3
        demo = d6_opportunity([good]*4+[bad]*3+[good]*6)           # 10 demonstrated, a 3-sample failing stretch
        self.assertEqual(E.key_run(demo, SCALE), (1, 7, 12))
        k = CX.key_sample(demo, SCALE)                                # the old pick: the failing stretch
        self.assertEqual(CX.sample_outcomes(demo, None, SCALE)[0][k], -1)
        fail = d6_opportunity([bad]*6+[good]*4)
        self.assertEqual(E.key_run(fail, SCALE), (-1, 0, 5))
        tie = d6_opportunity([good]*3+[bad]*3)
        self.assertEqual(E.key_run(tie, SCALE)[0], -1)                 # a tie shows the failing run
        split = d6_opportunity([bad, bad, good, bad, bad, bad, good, good, good, good])
        self.assertEqual(E.key_run(split, SCALE), (-1, 3, 5))          # 5 and 5: the longest failing run
        none = d6_opportunity([None]*5)
        self.assertEqual(E.key_run(none, SCALE), (0, 0, 4))

    def test_a_tie_on_length_takes_the_earliest_run(self):
        o = d6_opportunity([.3, .3, .7, .3, .3])
        self.assertEqual(E.key_run(o, SCALE), (-1, 0, 1))

    def test_missing_relations_give_no_outcome_instead_of_an_error(self):
        o = d6_opportunity([.7]*4)
        del o['relations']['depthShare']
        self.assertEqual(E.outcomes(o, None, SCALE), ([None]*4, [None]*4))
        self.assertEqual(E.key_run(o, SCALE), (0, 0, 3))

    def test_the_sample_inside_the_run(self):
        mid = 5.0
        cands = list(range(11))
        # nearest the middle, the earlier one on a tie
        self.assertEqual(E._choose(cands, mid, lambda k: None, lambda k: False, lambda k: 0), 5)
        self.assertEqual(E._choose([4, 6], mid, lambda k: None, lambda k: False, lambda k: 0), 4)
        # a better tier wins over the middle
        self.assertEqual(E._choose(cands, mid, lambda k: None, lambda k: False, lambda k: 2 if k == 9 else 1), 9)
        # a stabilizer flag is avoided while another candidate has none, kept when all are flagged
        self.assertEqual(E._choose(cands, mid, lambda k: None, lambda k: k == 5, lambda k: 0), 4)
        self.assertEqual(E._choose([5], mid, lambda k: None, lambda k: True, lambda k: 0), 5)
        # an identity doubt anywhere in the run is never hidden by a cleaner frame (counterexample 'identity crossing')
        self.assertEqual(E._choose(cands, mid, lambda k: k not in (1, 2), lambda k: False,
                                   lambda k: 2 if k == 5 else 1), 2)

    def test_at_most_max_ranked_tiers_are_evaluated(self):
        seen = []

        def rank(k):
            seen.append(k)
            return 0
        k = E._choose(list(range(200)), 100.0, lambda k: None, lambda k: False, rank)
        self.assertEqual(k, 100)
        self.assertLessEqual(len(set(seen)), E.MAX_RANKED)


class V2Moments(unittest.TestCase):
    """The measurement and role tiers' moments on EVAL's unsupported-advance scene (16 uncovered left-back advances,
    team A defending left, world-tier broadcast frames)."""

    @classmethod
    def setUpClass(cls):
        cls.b, cls.st, cls.cd, cls.tl, cls.obs, cls.roles, cls.anchors = v2_tiers('unsupported_advance', 'none_behind')

    def test_the_files_validate_with_one_moment_per_opportunity(self):
        self.assertEqual(S.validate('obs', self.obs), [])
        self.assertEqual(S.validate('roles', self.roles), [])
        for f in (self.obs, self.roles):
            opps = [o['id'] for o in f['opportunities'] if o['abstain'] is None]
            self.assertEqual(sorted(m['opportunity'] for m in f['moments']), sorted(opps))
        self.assertTrue(self.roles['moments'])
        ids = [m['id'] for m in self.obs['moments']+self.roles['moments']]
        self.assertEqual(len(ids), len(set(ids)))

    def test_frames_are_exact_and_the_scene_is_the_timelines(self):
        tl, fps = self.tl, float(self.b.result['video']['fps'])
        for m in self.obs['moments']:
            s = sample_of(tl, m['t'])
            self.assertAlmostEqual(float(tl.t[s]), m['t'], places=3)
            self.assertEqual(m['frame'], int(tl.frame[s]))
            self.assertEqual(m['fps'], fps)
            self.assertAlmostEqual(m['seek'], (m['frame']+.5)/fps, places=4)
            o = next(o for o in self.obs['opportunities'] if o['id'] == m['opportunity'])
            self.assertAlmostEqual(o['relations']['t'][m['sample']], m['t'], places=3)
            want = {pid for j, pid in enumerate(tl.ids) if tl.posOk[s, j]} | \
                {pid for j, pid in enumerate(tl.oppIds) if tl.oppPosOk[s, j]}
            got = {p for p, e in m['scene']['players'].items() if not e['goalkeeper']}
            self.assertEqual(got, want)
            for pid, e in m['scene']['players'].items():
                if pid in tl.col:
                    self.assertAlmostEqual(e['x'], float(tl.x[s, tl.col[pid]]), places=4)
                    self.assertAlmostEqual(e['y'], float(tl.y[s, tl.col[pid]]), places=4)
            self.assertTrue(set(m['players']) <= set(m['scene']['players']))
            involved = set(E._involved(o, m['sample']))
            for pid, e in m['scene']['players'].items():
                if e['team'] != 'A' and e['box'] is not None:
                    self.assertIn(pid, involved)             # opponents keep a box only when the moment is about them
            if m['before'] is not None:
                self.assertAlmostEqual(m['before']['t'], m['t']-E.REACH_H, delta=float(tl.step)/2+1e-6)
                self.assertLess(m['before']['frame'], m['frame'])
                sb = sample_of(tl, m['before']['t'])
                own = {pid for j, pid in enumerate(tl.ids) if tl.posOk[sb, j]}
                self.assertEqual({p for p, e in m['before']['players'].items() if not e['goalkeeper']}, own)
                self.assertTrue(all(e['team'] == 'A' and e['box'] is None for e in m['before']['players'].values()))
            self.assertEqual(m['calibration']['side'], str(tl.side[s]))

    def test_the_moment_shows_the_dominant_outcome(self):
        for f in (self.obs, self.roles):
            opps = {o['id']: o for o in f['opportunities']}
            for m in f['moments']:
                o = opps[m['opportunity']]
                want, i0, i1 = E.key_run(o, self.obs['timeline']['scale'])
                self.assertTrue(i0 <= m['sample'] <= i1)
                if want:
                    outs, _ = CX.sample_outcomes(o, None, self.obs['timeline']['scale'])
                    self.assertEqual(outs[m['sample']], want)

    def test_world_moments_carry_the_frames_homography(self):
        fx = E.frames_of(self.cd)
        for m in self.obs['moments']:
            cal = m['calibration']
            s = sample_of(self.tl, m['t'])
            self.assertEqual(m['tier'], fx.tier(s)[0])
            has_h = all(f'h{i}' in cal for i in range(9))
            self.assertEqual(has_h, m['tier'] == 'world')
            if has_h:
                H = np.array([cal[f'h{i}'] for i in range(9)]).reshape(3, 3)
                np.testing.assert_allclose(H, E.oriented_h(fx.frame(s), fx.width, fx.height), rtol=1e-8)
            self.assertEqual((cal['pitchLengthM'], cal['pitchWidthM']), (105.0, 68.0))

    def test_frame_tiers_cover_every_sample_the_role_tier_stores(self):
        fx = E.frames_of(self.cd)
        spans = self.obs['frameTiers']
        self.assertTrue(spans)
        for a, b in zip(spans, spans[1:]):
            self.assertLess(a['frame1'], b['frame0'])
        geo = self.obs['roleInputs']['detect']['samples']
        for t, frame in zip(geo['t'], geo['frame']):
            span = [r for r in spans if r['frame0'] <= frame <= r['frame1']]
            self.assertEqual(len(span), 1, frame)
            self.assertEqual(span[0]['tier'], fx.tier(sample_of(self.tl, t))[0])
        stored = set(geo['frame'])
        grid = [int(f) for f in self.tl.frame]
        for r in spans:                       # a run never claims a tier for a grid frame the role tier does not store
            inside = [f for f in grid if r['frame0'] <= f <= r['frame1']]
            self.assertTrue(inside and set(inside) <= stored, r)

    def test_role_moments_rebuild_the_timeline_from_the_obs_file(self):
        L = self.obs['timeline']['scale']['lengthM']
        fx = E.frames_of(self.cd)
        for m in self.roles['moments']:
            s = sample_of(self.tl, m['t'])
            self.assertEqual(m['frame'], int(self.tl.frame[s]))
            self.assertEqual(m['calibration']['side'], str(self.tl.side[s]))
            for pid, e in m['scene']['players'].items():
                if pid in self.tl.col:                  # stored at 0.1 m in the attack frame
                    self.assertAlmostEqual(e['x'], float(self.tl.x[s, self.tl.col[pid]]), delta=.06/L+1e-4)
            if str(m['frame']) in self.anchors['frames']:
                people = fx.people(s)
                for pid, e in m['scene']['players'].items():
                    if e['team'] == 'A':
                        self.assertEqual(e['box'], [round(v, 4) for v in people[pid]['box']])
            self.assertEqual(m['tier'], fx.tier(s)[0])

    def test_role_moments_without_the_anchors_have_no_pixels(self):
        obs = copy.deepcopy(self.obs)
        del obs['roleInputs']['evidence']
        out = E.moments_from_obs(obs, self.roles['opportunities'])['moments']
        self.assertEqual(len(out), len(self.roles['moments']))
        for m in out:
            self.assertEqual(S.validate('moment', m), [])
            self.assertTrue(all(e['box'] is None for e in m['scene']['players'].values()))
            self.assertFalse(any(k.startswith('h') and k[1:].isdigit() for k in m['calibration']))

    def test_anchors_are_sparse_and_in_the_stored_samples(self):
        geo = self.obs['roleInputs']['detect']['samples']
        frames = {str(f) for f in geo['frame']}
        self.assertTrue(set(self.anchors['frames']) <= frames)
        self.assertLessEqual(len(self.anchors['frames']), len(frames)//E.ANCHOR_STRIDE+1)
        for a in self.anchors['frames'].values():
            self.assertTrue(a['h'] is None or len(a['h']) == 9)

    def test_the_moments_never_change_with_the_coaching_profile(self):
        """Invariant 2: the measurement stages see only the format and field; the bytes of obs are the same."""
        b, st, cd, poss = FX.prepared('unsupported_advance', 'none_behind')
        dumps = set()
        for level, style in ((1, 'unknown'), (7, 'compact')):
            s2 = copy.deepcopy(st)
            s2['teams']['A'].update(playingLevel=level, ageGroup=FX.PR.default_age(level), style=style)
            obs = coaching.build_observations(b.result, 'A', coaching=s2, series=cd, possession=poss,
                                              result_sha='0'*64)
            dumps.add(json.dumps(obs['moments'], sort_keys=True)+json.dumps(obs['frameTiers'], sort_keys=True))
        self.assertEqual(len(dumps), 1)
        self.assertTrue(dumps.pop().startswith(json.dumps(self.obs['moments'], sort_keys=True)))


class V2Annotate(unittest.TestCase):
    """The annotations of the findings on the unsupported-advance scene at every level."""

    @classmethod
    def setUpClass(cls):
        cls.b, cls.st, cls.cd, cls.tl, cls.obs, cls.roles, _ = v2_tiers('unsupported_advance', 'none_behind')
        cls.out, cls.ev, cls.prof = {}, {}, {}
        for level in range(1, 8):
            prof = FX.profile('unsupported_advance', 'none_behind', level)
            out = AS.assess(cls.obs, cls.roles, prof)
            cls.prof[level], cls.out[level] = prof, out
            cls.ev[level] = E.annotate(out['findings'], cls.obs, cls.roles, prof, alternatives=[])
        cls.moments = {m['id']: m for m in cls.obs['moments']+cls.roles['moments']}

    def finding(self, level, detector, verdicts=('work_on', 'try_this', 'review')):
        return next(f for f in self.out[level]['findings'] if f['observation'] == detector and
                    f['verdict'] in verdicts)

    def test_every_cited_moment_has_valid_evidence(self):
        for level, ev in self.ev.items():
            cited = {ts['evidence'] for f in self.out[level]['findings'] for ts in f['timestamps']}
            self.assertEqual({e['id'] for e in ev}, cited)
            self.assertEqual([e['time'] for e in ev], sorted(e['time'] for e in ev))
            seg = self.obs['segment']
            for e in ev:
                self.assertEqual(S.validate('evidence', e), [], (level, e['id']))
                m = self.moments[e['id']]
                self.assertEqual((e['frame'], e['seek'], e['opportunity'], e['time']),
                                 (m['frame'], m['seek'], m['opportunity'], m['t']))
                # DESIGN 13.4 #8: the outlined player stands where he was observed on that frame, inside the spell
                o = next(x for x in self.obs['opportunities']+self.roles['opportunities']
                         if x['id'] == m['opportunity'])
                self.assertTrue(o['t0'] <= e['time'] <= o['t1'])
                for el in e['diagram']['elements']:
                    if el['type'] in ('focus', 'support'):
                        seen = m['scene']['players'][el['id']]
                        self.assertEqual((el['x'], el['y'], seen['kind']), (seen['x'], seen['y'], 'O'))
                if e['videoOverlay'] is None:
                    self.assertEqual(e['tier'], m['tier'])        # nothing drawn over the video: its tier stands
                else:
                    self.assertLessEqual(E.TIER_RANK[e['tier']], E.TIER_RANK[m['tier']])
                a = e['annotation']
                self.assertEqual(a['keyFrame'], e['frame'])
                self.assertAlmostEqual(a['window']['t0'], max(seg['t0'], m['t']-3.0), places=3)
                self.assertAlmostEqual(a['window']['t1'], min(seg['t1'], m['t']+2.0), places=3)
                self.assertLessEqual(len(a['caption']), S.CAPTION_MAX)

    def test_elements_follow_the_tier_and_the_budget(self):
        for ev in self.ev.values():
            for e in ev:
                ov = e['videoOverlay']
                for spec, gate in ((e['diagram'], 'diagram'), (ov, (ov or {}).get('tier'))):
                    if spec is None:
                        continue
                    counts = {}
                    for el in spec['elements']:
                        self.assertIn(gate, S.TIER_ELEMENTS[el['type']], (el['type'], gate))
                        counts[el['type']] = counts.get(el['type'], 0)+1
                    for t, n in counts.items():
                        self.assertLessEqual(n, S.ELEMENT_BUDGET.get(t, 10**6))
                if e['videoOverlay']:
                    self.assertEqual(e['videoOverlay']['tier'], e['tier'])

    def test_the_wide_defender_annotation(self):
        f = self.finding(4, 'wide_defender_advance')
        self.assertEqual(f['verdict'], 'work_on')
        e = next(x for x in self.ev[4] if x['id'] == f['timestamps'][0]['evidence'])
        els = {el['type']: el for el in e['diagram']['elements']}
        self.assertEqual(els['focus']['id'], f['players']['primary'][0])
        self.assertEqual((els['focus']['tone'], els['channel']['kind']), ('review', 'vacated'))
        rel = els['relation']
        self.assertEqual((rel['kind'], rel['ok']), ('cover', False))         # nobody covered: dashed
        xs = [p[0] for p in els['channel']['polygon']]
        self.assertLess(max(xs), els['focus']['x'])                          # the space behind him (A defends left)
        ys = [p[1] for p in els['channel']['polygon']]
        self.assertAlmostEqual(min(ys), 0.0, places=3)
        self.assertAlmostEqual(max(ys), 1/3, places=3)                        # his outer third (team's left)
        ov = e['videoOverlay']
        self.assertEqual(ov['tier'], 'world')
        types = [el['type'] for el in ov['elements']]
        self.assertIn('channel', types)
        focus = next(el for el in ov['elements'] if el['type'] == 'focus')
        box = focus['box']
        self.assertAlmostEqual(focus['x'], box[0]+box[2]/2, places=0)          # on the feet
        self.assertAlmostEqual(focus['y'], box[1]+box[3], places=0)

    def test_any_cover_test_turns_the_wide_defenders_line_solid(self):
        f = self.finding(4, 'wide_defender_advance')
        m = self.moments[f['timestamps'][0]['evidence']]
        o = copy.deepcopy(next(x for x in self.roles['opportunities'] if x['id'] == m['opportunity']))
        k = m['sample']

        def line(opp):
            an = E._Annotation(f, m, opp, None, self.obs, self.prof[4])
            return next(r for r in an.relations if r['kind'] == 'cover')
        self.assertFalse(line(o)['ok'])
        for test in ('cbAdjusted', 'channelOccupied', 'interchange'):    # tests (c), (a), (e)
            o2 = copy.deepcopy(o)
            o2['relations'][test][k] = True
            self.assertTrue(line(o2)['ok'], test)

    def test_labels_follow_the_levels_naming(self):
        for level in range(1, 8):
            naming = self.prof[level]['evidence']['naming']
            for e in self.ev[level]:
                f = next(x for x in self.out[level]['findings'] if x['id'] == e['finding'])
                for el in e['diagram']['elements']:
                    if el['type'] == 'focus':
                        if naming in ('never', 'relative', 'role_words'):
                            self.assertFalse(el['label'], (level, f['observation']))
                        if f['guardrails']['identity'] == 'fail':
                            self.assertFalse(el['label'])
        f = self.finding(5, 'wide_defender_advance')
        e = next(x for x in self.ev[5] if x['id'] == f['timestamps'][0]['evidence'])
        self.assertTrue(next(el for el in e['diagram']['elements'] if el['type'] == 'focus')['label'])

    def test_the_cover_line_only_where_the_level_requires_cover(self):
        # DETECT fix round 1 (critic S-2): this scene's background pressing sits at 3-5 m without closing ('unclear'
        # under the three-state call: level 4 decides 2 of 20 opportunities and reports no pressure finding), so the
        # level-4 half reads the press-with-cover scene, whose level-4 strength shows the cover.
        _, _, _, _, obs_c, roles_c, _ = v2_tiers('defending', 'press_cover')
        prof_c = FX.profile('defending', 'press_cover', 4)
        ev4 = E.annotate(AS.assess(obs_c, roles_c, prof_c)['findings'], obs_c, roles_c, prof_c, alternatives=[])
        for level, prof, evs in ((1, self.prof[1], self.ev[1]), (4, prof_c, ev4)):
            needs = prof['detectors']['pressure_cover']['params'].get('coverRequired', True)
            seen = False
            for e in evs:
                if e['detector'] != 'pressure_cover':
                    continue
                kinds = [el['kind'] for el in e['diagram']['elements'] if el['type'] == 'relation']
                self.assertIn('pressure', kinds)
                seen = True
                if not needs:
                    self.assertNotIn('cover', kinds)
            self.assertTrue(seen)
        self.assertFalse(self.prof[1]['detectors']['pressure_cover']['params']['coverRequired'])
        self.assertTrue(any(el['kind'] == 'cover' for e in ev4 if e['detector'] == 'pressure_cover'
                            for el in e['diagram']['elements'] if el['type'] == 'relation'))

    def test_the_behind_the_ball_annotation_is_judged_with_the_levels_parameters(self):
        # this scene's ball never reaches the final third: judge the middle third too (profile param behindBallZone)
        L = self.obs['timeline']['scale']['lengthM']
        by_opp = {m['opportunity']: m for m in self.obs['moments']}
        o = next(x for x in self.obs['opportunities'] if x['detector'] == 'behind_ball' and x['id'] in by_opp and
                 (x['relations']['ballX'][by_opp[x['id']]['sample']] or 0) >= L/3 and
                 x['relations']['behindN'][by_opp[x['id']]['sample']] >= 2)
        m = by_opp[o['id']]
        f = copy.deepcopy(self.finding(4, 'wide_defender_advance'))
        f.update(observation='behind_ball', verdict='review', opportunities=[o['id']],
                 timestamps=[{'evidence': m['id'], 'opportunity': o['id'], 't': m['t'], 'frame': m['frame'],
                              'seek': m['seek']}])

        def an(prof):
            return E._Annotation(f, m, o, None, self.obs, prof)
        prof = copy.deepcopy(self.prof[4])
        prof['detectors']['behind_ball']['params']['behindBallZone'] = 'middle_or_final'
        a = an(prof)
        line = next(r for r in a.relations if r['kind'] == 'behind_ball')
        self.assertEqual((line['b'], line['pb'], line['ok']), ('ball', m['scene']['ball'], True))
        self.assertEqual(a.focus[0][0], o['relations']['deepest'][m['sample']])
        behind = (o['relations']['behindIds'][m['sample']] or '').split('|')
        self.assertTrue({p for p, _ in a.support} <= set(behind))
        strict = copy.deepcopy(prof)
        strict['detectors']['behind_ball']['params']['minBehind'] = 11
        self.assertFalse(next(r for r in an(strict).relations if r['kind'] == 'behind_ball')['ok'])
        e = E.evidence_of(f, m, o, None, self.obs, strict)
        self.assertEqual(S.validate('evidence', e), [])

    def test_relation_values_follow_the_register(self):
        metres = E._metres_word(self.obs['timeline']['scale'])
        for level in range(1, 8):
            mode = self.prof[level]['register']['numbers']
            for e in self.ev[level]:
                for el in e['diagram']['elements']:
                    if el['type'] != 'relation' or el['kind'] not in ('pressure', 'cover', 'gap'):
                        continue
                    if mode in ('none', 'counts'):
                        self.assertNotIn('text', el)
                    elif 'text' in el:
                        about = mode == 'rounded_metres' or metres == 'estimated'
                        self.assertRegex(el['text'], r'^about \d+ m$' if about else r'^\d+ m$')

    def test_captions_pass_the_text_gate_at_every_level(self):
        for level in range(1, 8):
            prof = self.prof[level]
            for f in self.out[level]['findings']:
                cap = E.caption_of(f, prof)
                self.assertLessEqual(len(cap), S.CAPTION_MAX)
                claims = EX.build_claims(f, prof)
                text = {'headline': cap, 'explanation': cap, 'short': cap,
                        'action': None if f['verdict'] == 'describe' else cap}
                self.assertEqual(EX.validate(text, claims, f, prof, short=True), [], (level, f['observation'], cap))
                low = cap.lower()
                for word in prof['register']['banned']:
                    self.assertNotIn(word.lower(), low)

    def test_describe_findings_outline_nobody(self):
        found = False
        for level, ev in self.ev.items():
            for e in ev:
                f = next(x for x in self.out[level]['findings'] if x['id'] == e['finding'])
                if f['verdict'] == 'describe':
                    found = True
                    self.assertFalse([el for el in e['diagram']['elements'] if el['type'] == 'focus'])
        self.assertTrue(found)

    def test_the_attacking_shape_draws_its_width_or_depth(self):
        found = overlaid = False
        for level, ev in self.ev.items():
            for e in ev:
                if e['detector'] != 'attacking_shape':
                    continue
                ms = [el for el in e['diagram']['elements'] if el['type'] == 'measure']
                self.assertEqual(len(ms), 1)
                players = {el['id']: el for el in e['diagram']['elements'] if el['type'] == 'player'}
                self.assertTrue(all(p in players and players[p]['emphasis'] for p in ms[0]['ids']))
                self.assertFalse([el for el in e['diagram']['elements'] if el['type'] == 'focus'])
                ov = e['videoOverlay']
                if ov is not None:                                  # between the two players' feet on the video
                    vm = [el for el in ov['elements'] if el['type'] == 'measure']
                    self.assertEqual([x['ids'] for x in vm], [ms[0]['ids']])
                    m = self.moments[e['id']]
                    for end, pid in zip((vm[0]['a'], vm[0]['b']), ms[0]['ids']):
                        bx = m['scene']['players'][pid]['box']
                        self.assertAlmostEqual(end[0], (bx[0]+bx[2]/2)*1920, delta=.2)
                    overlaid = True
                found = True
        self.assertTrue(found and overlaid)

    def test_a_finding_citing_an_unmeasured_moment_is_refused(self):
        f = copy.deepcopy(self.finding(4, 'wide_defender_advance'))
        f['timestamps'][0]['evidence'] = 'ev-000000000000'
        with self.assertRaises(ValueError):
            E.annotate([f], self.obs, self.roles, self.prof[4])

    def test_apply_captions_takes_the_findings_short_text(self):
        f = copy.deepcopy(self.finding(4, 'wide_defender_advance'))
        ev = E.annotate([f], self.obs, self.roles, self.prof[4])
        f['text'] = {'headline': 'x', 'explanation': 'x', 'short': 'The left back pushed on; check the space behind.',
                     'action': None}
        out = E.apply_captions(ev, [f])
        for old, new in zip(ev, out):
            self.assertEqual(new['annotation']['caption'], f['text']['short'])
            caps = [el['text'] for spec in (new['diagram'], new['videoOverlay']) if spec for el in spec['elements']
                    if el['type'] == 'caption']
            self.assertTrue(caps and all(c == f['text']['short'] for c in caps))
            self.assertEqual(S.validate('evidence', new), [])
            self.assertNotEqual(old['annotation']['caption'], new['annotation']['caption'])
        self.assertEqual(E.apply_captions(ev, []), ev)
        noted = copy.deepcopy(ev[0])
        noted['caption'] = E._with_note(noted['annotation']['caption'])
        new = E.apply_captions([noted], [f])[0]
        self.assertEqual(new['caption'], f['text']['short']+' '+E.DIAGRAM_NOTE)


def alternative_for(f, m, obs, mover, centre, half=.012, display='zone', radius=8.0):
    """A schema-valid Alternative of a finding at a moment (a square zone around centre, normalized)."""
    cx, cy = centre
    poly = [[cx-half, cy-half], [cx+half, cy-half], [cx+half, cy+half], [cx-half, cy+half]]
    e = m['scene']['players'][mover]
    alt = {'id': S.record_id('alt', f['id'], m['id'], mover), 'finding': f['id'], 'evidence': m['id'], 't': m['t'],
           'mover': mover, 'principle': 'cover_behind_advancing_defender',
           'zone': {'polygon': poly, 'areaM2': 9.0, 'kind': 'zone', 'radiusM': None}, 'display': display,
           'from': [e['x'], e['y']], 'reach': {'horizonS': 2.0, 'speedMS': radius/2.0, 'radiusM': radius},
           'improves': {'relation': 'channel_cover', 'before': False, 'after': True, 'shareOfZone': .9},
           'checks': {'pressureKept': True, 'centralKept': True, 'gapOk': True, 'noNewFreeOpponent': True,
                      'noStacking': True}, 'assumption': 'single_mover', 'rejected': []}
    assert S.validate('alternative', alt) == [], S.validate('alternative', alt)
    return alt


class V2Zones(unittest.TestCase):
    """A validated alternative on the wide-defender finding: where its zone and arrow are drawn, and when not."""

    @classmethod
    def setUpClass(cls):
        cls.b, cls.st, cls.cd, cls.tl, cls.obs, cls.roles, _ = v2_tiers('unsupported_advance', 'none_behind')
        cls.prof = FX.profile('unsupported_advance', 'none_behind', 4)
        cls.out = AS.assess(cls.obs, cls.roles, cls.prof)
        cls.f = next(f for f in cls.out['findings'] if f['observation'] == 'wide_defender_advance')
        cls.moments = {m['id']: m for m in cls.obs['moments']+cls.roles['moments']}
        cls.m = cls.moments[cls.f['timestamps'][0]['evidence']]
        sc = cls.m['scene']['players']
        cls.mover = next(p for p in cls.f['players']['supporting'] if p in sc)
        # a zone 6 m deeper and nearer the touchline than the mover (inside the vacated channel's direction)
        e = sc[cls.mover]
        cls.alt = alternative_for(cls.f, cls.m, cls.obs, cls.mover, (e['x']-6/105, max(.08, e['y']-.12)))

    def annotate(self, alt, obs=None, roles=None):
        return E.annotate([self.f], obs or self.obs, roles or self.roles, self.prof, alternatives=[alt])

    def test_the_zone_is_drawn_on_the_alternatives_moment_only(self):
        ev = self.annotate(self.alt)
        for e in ev:
            self.assertEqual(S.validate('evidence', e), [])
            zones = [el for spec in (e['diagram'], e['videoOverlay']) if spec for el in spec['elements']
                     if el['type'] in ('suggestedZone', 'moveArrow')]
            if e['id'] == self.alt['evidence']:
                self.assertEqual(e['alternative'], self.alt['id'])
                self.assertEqual(len(zones), 4)                      # zone and arrow, diagram and video
            else:
                self.assertEqual((zones, e['alternative']), ([], None))
        e = next(x for x in ev if x['id'] == self.alt['evidence'])
        d = {el['type']: el for el in e['diagram']['elements']}
        np.testing.assert_allclose(d['suggestedZone']['polygon'], self.alt['zone']['polygon'], atol=1e-4)
        self.assertEqual(d['suggestedZone']['mover'], self.mover)
        mover = self.m['scene']['players'][self.mover]
        self.assertEqual(d['moveArrow']['from'], [mover['x'], mover['y']])  # from where he stood at the moment
        moved = copy.deepcopy(self.alt)
        moved['from'] = [mover['x']+.05, mover['y']]                      # (alt.from: his position 2 s earlier)
        e2 = next(x for x in self.annotate(moved) if x['id'] == moved['evidence'])
        arrow = next(el for el in e2['diagram']['elements'] if el['type'] == 'moveArrow')
        self.assertEqual(arrow['from'], [mover['x'], mover['y']])
        cx = sum(p[0] for p in self.alt['zone']['polygon'])/4
        self.assertAlmostEqual(d['moveArrow']['to'][0], cx, places=4)
        sup = [el['id'] for el in e['diagram']['elements'] if el['type'] == 'support']
        self.assertEqual(sup[0], self.mover)
        # the video zone is the diagram zone through the frame's homography
        cal = self.m['calibration']
        H = np.array([cal[f'h{i}'] for i in range(9)]).reshape(3, 3)
        ov = next(el for el in e['videoOverlay']['elements'] if el['type'] == 'suggestedZone')
        want = field.project(H, [[x*cal['pitchLengthM'], y*cal['pitchWidthM']]
                                 for x, y in d['suggestedZone']['polygon']])
        np.testing.assert_allclose(np.array(ov['polygon']), want, atol=.06)
        self.assertNotIn(E.DIAGRAM_NOTE, e['caption'])

    def test_a_direction_shows_only_an_arrow_no_longer_than_the_reach(self):
        far = copy.deepcopy(self.alt)
        far['display'] = 'direction'
        sc = self.m['scene']['players'][self.mover]
        far['zone']['polygon'] = [[x-.25, y] for x, y in far['zone']['polygon']]
        ev = self.annotate(far)
        e = next(x for x in ev if x['id'] == far['evidence'])
        self.assertEqual(S.validate('evidence', e), [])
        types = [el['type'] for el in e['diagram']['elements']]
        self.assertNotIn('suggestedZone', types)
        arrow = next(el for el in e['diagram']['elements'] if el['type'] == 'moveArrow')
        L, W = self.obs['timeline']['scale']['lengthM'], self.obs['timeline']['scale']['widthM']
        length = math.hypot((arrow['to'][0]-sc['x'])*L, (arrow['to'][1]-sc['y'])*W)
        self.assertLessEqual(length, far['reach']['radiusM']+.05)
        self.assertEqual(e['alternative'], far['id'])

    def test_the_disagreement_rule_keeps_the_zone_off_a_misregistered_frame(self):
        obs, roles = copy.deepcopy(self.obs), copy.deepcopy(self.roles)
        for m in obs['moments']+roles['moments']:
            if m['id'] == self.m['id']:
                cal = m['calibration']
                H = np.array([cal[f'h{i}'] for i in range(9)]).reshape(3, 3)
                shift = np.array([[1.0, 0.0, 60.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])   # 60 px off
                for i, v in enumerate((shift @ H).ravel()):
                    cal[f'h{i}'] = float(v)
        ev = self.annotate(self.alt, obs, roles)
        e = next(x for x in ev if x['id'] == self.alt['evidence'])
        self.assertEqual(S.validate('evidence', e), [])
        self.assertEqual((e['tier'], e['videoOverlay']['tier']), ('pinned', 'pinned'))
        types = [el['type'] for el in e['videoOverlay']['elements']]
        for t in ('suggestedZone', 'channel', 'moveArrow'):
            self.assertNotIn(t, types)
        self.assertIn('suggestedZone', [el['type'] for el in e['diagram']['elements']])
        self.assertTrue(e['caption'].endswith(E.DIAGRAM_NOTE))
        self.assertEqual(e['seek'], self.m['seek'])

    def test_without_pixels_the_overlay_is_absent_and_the_tier_stands(self):
        obs = copy.deepcopy(self.obs)
        del obs['roleInputs']['evidence']
        roles = copy.deepcopy(self.roles)
        roles['moments'] = E.moments_from_obs(obs, roles['opportunities'])['moments']
        out = AS.assess(obs, roles, self.prof)
        f = next(f for f in out['findings'] if f['observation'] == 'wide_defender_advance')
        moments = {m['id']: m for m in roles['moments']}
        for e in E.annotate([f], obs, roles, self.prof):
            self.assertIsNone(e['videoOverlay'])
            self.assertEqual(e['tier'], moments[e['id']]['tier'])
            self.assertEqual(e['tier'], 'world')
            self.assertTrue(e['diagram']['elements'])

    def test_a_zone_behind_the_camera_or_off_the_picture_stays_on_the_diagram(self):
        off = copy.deepcopy(self.alt)
        off['zone']['polygon'] = [[1.05, -.08], [1.09, -.08], [1.09, -.04], [1.05, -.04]]
        e = next(x for x in self.annotate(off) if x['id'] == off['evidence'])
        self.assertNotIn('suggestedZone', [el['type'] for el in (e['videoOverlay'] or {}).get('elements', ())])
        self.assertIn('suggestedZone', [el['type'] for el in e['diagram']['elements']])
        self.assertTrue(e['caption'].endswith(E.DIAGRAM_NOTE))

    def test_the_zone_label_respects_the_levels_words(self):
        prof1 = FX.profile('unsupported_advance', 'none_behind', 1)
        self.assertIn('cover', prof1['register']['banned'])
        an = E._Annotation(self.f, self.m, next(o for o in self.roles['opportunities']
                                                 if o['id'] == self.m['opportunity']), self.alt, self.obs, prof1)
        self.assertEqual(an.zone[4], E.ZONE_LABEL_PLAIN)


class V2LowCalibration(unittest.TestCase):
    """DESIGN 13.1 'calibration below world at every candidate frame': no zone on the video, the zone on the
    diagram, the seek still exact."""

    @classmethod
    def setUpClass(cls):
        cls.b, cls.st, cls.cd, cls.tl, cls.obs, cls.roles, _ = v2_tiers('low_calibration')
        cls.prof = FX.profile('low_calibration', None, 4)
        cls.out = AS.assess(cls.obs, cls.roles, cls.prof)

    def test_no_zone_on_the_video_and_the_seek_exact(self):
        tiers = {m['tier'] for m in self.obs['moments']+self.roles['moments']}
        self.assertNotIn('world', tiers)
        for m in self.obs['moments']:
            self.assertFalse(any(k in m['calibration'] for k in ('h0', 'h8')))     # no homography below world
        f = next(f for f in self.out['findings'] if f['verdict'] in ('work_on', 'review', 'try_this'))
        moments = {m['id']: m for m in self.obs['moments']+self.roles['moments']}
        m = moments[f['timestamps'][0]['evidence']]
        mover = next(p for p in m['scene']['players'] if p != (f['players']['primary'] or [None])[0] and
                     m['scene']['players'][p]['team'] == 'A')
        e0 = m['scene']['players'][mover]
        alt = alternative_for(f, m, self.obs, mover, (e0['x']-.03, e0['y']))
        ev = E.annotate([f], self.obs, self.roles, self.prof, alternatives=[alt])
        e = next(x for x in ev if x['id'] == m['id'])
        self.assertEqual(S.validate('evidence', e), [])
        self.assertIn('suggestedZone', [el['type'] for el in e['diagram']['elements']])
        for el in (e['videoOverlay'] or {}).get('elements', ()):
            self.assertNotIn(el['type'], ('suggestedZone', 'channel', 'moveArrow'))
        self.assertTrue(e['caption'].endswith(E.DIAGRAM_NOTE))
        fps = float(self.b.result['video']['fps'])
        self.assertAlmostEqual(e['seek'], (e['frame']+.5)/fps, places=4)
        s = sample_of(self.tl, e['time'])
        self.assertEqual(e['frame'], int(self.tl.frame[s]))


class V2IdentityCrossing(unittest.TestCase):
    """DESIGN 13.1 'identity crossing at the evidence moment': the moment keeps the doubt, nobody is labelled."""

    @classmethod
    def setUpClass(cls):
        cls.b, cls.st, cls.cd, cls.tl, cls.obs, cls.roles, _ = v2_tiers('identity_crossing')

    def test_the_crossing_stays_visible_to_the_identity_guardrail(self):
        # Fix round 3 (critic B-1): level 2 judges an advance left exposed (an opponent in the vacated channel) wherever
        # the ball is, so this scene's advances (ball at 36-42 m, attack frame) are judged at level 2 as well; the
        # crossing keeps the identity guardrail failed there as at levels 3-7: a review, no work item, and every focus
        # ring drawn at the moment without a label (nobody is named while two identities cross).
        for level in (2, 3, 4, 6):
            prof = FX.profile('identity_crossing', None, level)
            out = AS.assess(self.obs, self.roles, prof)
            fs = [f for f in out['findings'] if f['observation'] == 'wide_defender_advance']
            with self.subTest(level=level):
                self.assertFalse([f for f in fs if f['verdict'] in ('work_on', 'try_this')])
                f = next(f for f in fs if f['verdict'] != 'strength')
                self.assertEqual((f['verdict'], f['guardrails']['identity']), ('review', 'fail'))
                focus = [el for e in E.annotate([f], self.obs, self.roles, prof) for el in e['diagram']['elements']
                         if el['type'] == 'focus']
                self.assertTrue(focus)
                for el in focus:
                    self.assertFalse(el['label'])


if __name__ == '__main__':
    unittest.main()
