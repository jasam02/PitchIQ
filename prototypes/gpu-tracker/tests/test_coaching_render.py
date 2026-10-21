"""Coaching figures as drawing operations (soccer/coaching/render.py): every DiagramSpec element type becomes the
operations the PDF writes and the PNG preview rasterizes; estimates and suggestions are never drawn as facts; an
arrow is cut where its samples are more than a second apart; a team width or length is a bracket beside the team, not
a line through it; a video overlay draws only what its tier allows, scaled to its placement and captioned as an
analysis overlay.

Run:  python -m unittest test_coaching_render          (from this directory)
"""
import json
import os
import sys
import unittest

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from soccer.coaching import render  # noqa: E402


def player(pid, x, y, team=None, kind='O', emphasis=False):
    return {'type': 'player', 'id': pid, 'team': team or pid[0], 'x': x, 'y': y, 'kind': kind, 'emphasis': emphasis}


def spec(*elements, **extra):
    s = {'preset': 'defensive_gap', 'title': 'test', 't': 24.433, 'frame': 733, 'team': 'A', 'attack': 'right',
         'metresEstimated': True, 'pitch': {'length': 105, 'width': 68}, 'elements': list(elements)}
    s.update(extra)
    return s


def ops_of(d, kind):
    return [o for o in d['ops'] if o['op'] == kind]


def at(d, x, y, width=None):
    """Points of a normalized pitch position in a diagram drawn at d's width."""
    L, W = 105.0, 68.0
    scale = d['width']/(L+2*render.MARGIN)
    return render.MARGIN*scale+x*L*scale, render.MARGIN*scale+y*W*scale


class DiagramTests(unittest.TestCase):
    def test_players_are_drawn_at_their_positions_in_team_colours(self):
        d = render.diagram_ops(spec(player('A-03', .4, .3), player('B-07', .6, .7)), 243)
        circles = ops_of(d, 'circle')
        ax, ay = at(d, .4, .3)
        a = [c for c in circles if abs(c['cx']-ax) < .02 and abs(c['cy']-ay) < .02]
        self.assertEqual(len(a), 1)
        self.assertEqual(a[0]['fill'], render.COLOURS['A'])
        self.assertEqual(a[0]['alpha'], 1.0)
        bx, by = at(d, .6, .7)
        self.assertTrue(any(c['fill'] == render.COLOURS['B'] and abs(c['cx']-bx) < .02 for c in circles))
        self.assertEqual(d['drawn'], {'player': 2})
        # the analysed team's players are labelled (by the display label when given), opponents are not
        d = render.diagram_ops(spec(player('A-03', .4, .3), player('B-07', .6, .7)), 243, labels={'A-03': 'A-09'})
        texts = [t['text'] for t in ops_of(d, 'text')]
        self.assertIn('A-09', texts)
        self.assertNotIn('A-03', texts)
        self.assertNotIn('B-07', texts)

    def test_an_estimate_is_hollow_and_dashed_and_an_emphasis_gets_a_ring(self):
        d = render.diagram_ops(spec(player('A-03', .4, .3, kind='P'), player('A-04', .5, .3, emphasis=True)), 243)
        est = [c for c in ops_of(d, 'circle') if abs(c['cx']-at(d, .4, .3)[0]) < .02]
        self.assertEqual(len(est), 1)
        self.assertLess(est[0]['alpha'], 1.0)
        self.assertTrue(est[0]['dash'])
        ex = at(d, .5, .3)[0]
        rings = [c for c in ops_of(d, 'circle') if abs(c['cx']-ex) < .02 and not c['fill']]
        self.assertEqual(len(rings), 1)
        self.assertEqual(rings[0]['stroke'], render.BRAND_DARK)
        self.assertIn('estimated position', d['legend'])
        self.assertIn('highlighted', d['legend'])

    def test_a_unit_runs_across_the_pitch_through_its_observed_players(self):
        els = [player('A-01', .30, .80), player('A-02', .31, .20), player('A-03', .29, .50),
               player('A-04', .30, .65, kind='P'),
               {'type': 'unit', 'unit': 'D', 'ids': ['A-01', 'A-02', 'A-03', 'A-04']}]
        d = render.diagram_ops(spec(*els), 243)
        line = [p for p in ops_of(d, 'path') if p['stroke'] == render.COLOURS['A']][0]
        ys = [round(y, 1) for _, y in line['points']]
        self.assertEqual(len(ys), 3)                                 # the estimate is not part of the line
        self.assertEqual(ys, sorted(ys))                             # touchline to touchline, never a zigzag
        d = render.diagram_ops(spec(player('A-01', .3, .5), {'type': 'unit', 'unit': 'D', 'ids': ['A-01', 'A-09']}))
        self.assertIn('unit', [s['type'] for s in d['skipped']])

    def test_a_measure_is_a_line_with_ticks_and_the_spec_text(self):
        d = render.diagram_ops(spec({'type': 'measure', 'a': [.3, .5], 'b': [.5, .5], 'text': '21.0 m (est.)'}))
        self.assertEqual(len([p for p in ops_of(d, 'path') if p['stroke'] == render.INK]), 3)
        label = [t for t in ops_of(d, 'text') if t['text'] == '21.0 m (est.)']
        self.assertEqual(len(label), 1)
        self.assertEqual(label[0]['halo'], render.WHITE)
        self.assertAlmostEqual(label[0]['x'], (at(d, .3, .5)[0]+at(d, .5, .5)[0])/2, places=1)

    def test_zones_and_bands_are_translucent(self):
        d = render.diagram_ops(spec({'type': 'zone', 'polygon': [[.3, .2], [.5, .2], [.5, .8]], 'style': 'space'},
                                    {'type': 'band', 'axis': 'x', 'from': .52, 'to': .38, 'style': 'gap'}))
        fills = [p for p in ops_of(d, 'path') if p['fill'] == render.ZONE_STYLES['space'][0]]
        self.assertEqual(len(fills), 1)
        self.assertTrue(fills[0]['closed'])
        self.assertLess(fills[0]['alpha'], 1)
        band = [r for r in ops_of(d, 'rect') if r['fill'] == render.ZONE_STYLES['gap'][0]
                and r['alpha'] == render.ZONE_STYLES['gap'][1]]                  # not the legend's swatch
        self.assertEqual(len(band), 1)
        self.assertLess(band[0]['alpha'], 1)
        x0, y0 = at(d, .38, 0)
        x1, y1 = at(d, .52, 1)
        self.assertAlmostEqual(band[0]['x'], x0, places=1)          # from/to in either order
        self.assertAlmostEqual(band[0]['w'], x1-x0, places=1)
        self.assertAlmostEqual(band[0]['h'], y1-y0, places=1)        # across the whole pitch
        self.assertEqual(d['drawn'], {'zone': 1, 'band': 1})

    def test_an_arrow_is_cut_where_its_samples_are_more_than_a_second_apart(self):
        pts = [[.5, .5, 10.0], [.52, .5, 10.4], [.54, .5, 10.8], [.6, .52, 12.5], [.62, .53, 12.9]]
        d = render.diagram_ops(spec(player('A-07', .5, .5), {'type': 'arrow', 'id': 'A-07', 'points': pts}))
        lines = [p for p in ops_of(d, 'path') if p['stroke'] == render.COLOURS['A']]
        self.assertEqual([len(p['points']) for p in lines], [3, 2])  # never joined across the 1.7 s gap
        heads = [p for p in ops_of(d, 'path') if p['closed'] and p['fill'] == render._shade(render.COLOURS['A'], .3)]
        self.assertEqual(len(heads), 1)
        self.assertEqual(heads[0]['points'][0], [round(v, 2) for v in at(d, .62, .53)])   # the tip at the end
        self.assertIn('arrow', [s['type'] for s in d['skipped']])    # the cut is reported
        d = render.diagram_ops(spec({'type': 'arrow', 'id': 'A-07', 'points': [[.5, .5, 1.0], [.6, .5, 1.5]]}))
        self.assertEqual(d['skipped'], [])

    def test_a_suggested_position_is_dashed_and_labelled_as_an_illustration(self):
        d = render.diagram_ops(spec(player('A-05', .5, .4),
                                    {'type': 'suggested', 'id': 'A-05', 'x': .45, 'y': .4, 'label': 'move here'}))
        sx, sy = at(d, .45, .4)
        ring = [c for c in ops_of(d, 'circle') if abs(c['cx']-sx) < .02]
        self.assertEqual(len(ring), 1)
        self.assertTrue(ring[0]['dash'])
        self.assertIsNone(ring[0]['fill'])
        self.assertIn(render.ILLUSTRATION, [t['text'] for t in ops_of(d, 'text')])
        link = [p for p in ops_of(d, 'path') if p['dash'] and p['stroke'] == render.BRAND_DARK]
        self.assertEqual(len(link), 1)                               # from the real position to the suggestion
        self.assertIn('coaching illustration', d['legend'])

    def test_labels_and_the_attack_direction(self):
        d = render.diagram_ops(spec({'type': 'label', 'at': [.4, .9], 'text': 'gap'}))
        self.assertIn('gap', [t['text'] for t in ops_of(d, 'text')])
        self.assertIn('Team A attacks', [t['text'] for t in ops_of(d, 'text')])
        d = render.diagram_ops(spec(), team_names={'A': 'Riverside'})
        self.assertIn('Riverside attacks', [t['text'] for t in ops_of(d, 'text')])

    def test_invalid_elements_are_skipped_never_clamped(self):
        els = [player('A-01', 1.4, .5), player('A-02', float('nan'), .5), {'type': 'wobble'}, 'junk',
               player('A-03', .4, .5), player('A-03', .6, .5),
               {'type': 'zone', 'polygon': [[.1, .1], [.2, .1]]}, {'type': 'band', 'axis': 'z', 'from': 0, 'to': 1},
               {'type': 'measure', 'a': [.1, .1], 'b': [2, 2], 'text': 'x'}, {'type': 'label', 'at': [.5, .5]}]
        d = render.diagram_ops(spec(*els))
        reasons = [s['reason'] for s in d['skipped']]
        self.assertEqual(len(reasons), 9)
        self.assertIn('duplicate player id', reasons)
        self.assertEqual(d['drawn'], {'player': 1})
        right = at(d, 1.0, .5)[0]
        self.assertFalse(any(c['cx'] >= right-.5 and c['fill'] == render.COLOURS['A'] for c in ops_of(d, 'circle')))

    def test_pitch_size_and_metres_note(self):
        small = render.diagram_ops(spec(pitch={'length': 100, 'width': 64}), 243, legend=False)
        std = render.diagram_ops(spec(), 243, legend=False)
        self.assertAlmostEqual(small['height'], 243*(64+8)/(100+8), places=1)
        self.assertAlmostEqual(std['height'], 243*(68+8)/(105+8), places=1)
        bad = render.diagram_ops(spec(pitch={'length': 'x', 'width': 9}), 243, legend=False)
        self.assertEqual(bad['height'], std['height'])
        self.assertIn('metres estimated (assumed 105 x 68 m pitch)', render.diagram_ops(spec())['legend'])
        self.assertIn('metres estimated (assumed 105 x 68 m pitch)',
                      render.diagram_ops(spec(metresEstimated=None))['legend'])
        self.assertFalse(any('metres' in x for x in render.diagram_ops(spec(metresEstimated=False))['legend']))

    def test_viewer_colours_are_used_only_when_valid(self):
        d = render.diagram_ops(spec(player('A-03', .4, .3), player('B-07', .6, .7)),
                               colours={'A': 'red', 'B': '#00FF00', 'C': '#123456'})
        fills = {c['fill'] for c in ops_of(d, 'circle')}
        self.assertIn(render.COLOURS['A'], fills)
        self.assertIn('#00ff00', fills)
        self.assertEqual(render.team_colours(None), render.COLOURS)

    def test_ops_are_deterministic_json_and_ascii(self):
        s = spec(player('A-03', .4, .3), {'type': 'arrow', 'id': 'A-03', 'points': [[.4, .3, 1], [.5, .3, 1.5]]})
        a, b = render.diagram_ops(s), render.diagram_ops(s)
        self.assertEqual(json.dumps(a, allow_nan=False), json.dumps(b, allow_nan=False))
        self.assertTrue(json.dumps(a).isascii())


class BracketTests(unittest.TestCase):
    """A team width or length bracket: beside the analysed team (clear of its markers and labels), on the asked
    side or the one covering fewer opponents, with end ticks, leaders to the end players and its value."""

    def team(self):
        return [player('A-01', .40, .12), player('A-02', .55, .30), player('A-03', .45, .50), player('A-04', .60, .88)]

    @staticmethod
    def line_of(d, vertical):
        """The bracket's main line: the INK path of width .9 along its direction (the ticks are short)."""
        k = 0 if vertical else 1
        return [p for p in ops_of(d, 'path') if p['stroke'] == render.INK and p['width'] == .9
                and len(p['points']) == 2 and abs(p['points'][0][k]-p['points'][1][k]) < .01
                and abs(p['points'][0][1-k]-p['points'][1][1-k]) > 10]

    def test_a_width_bracket_stands_beside_the_team_with_its_value(self):
        b = {'type': 'bracket', 'a': [.5, .12], 'b': [.5, .88], 'side': 'left', 'text': '51.7 m (est.)'}
        d = render.diagram_ops(spec(*self.team(), b), 243)
        self.assertEqual(d['drawn'].get('bracket'), 1)
        self.assertEqual(d['skipped'], [])
        scale = d['width']/(105+2*render.MARGIN)
        line = self.line_of(d, True)
        self.assertEqual(len(line), 1)
        x = line[0]['points'][0][0]
        left_marker = at(d, .40, .12)[0]-render.MIN_RADIUS
        self.assertLess(x, left_marker-render.BRACKET_GAP*scale*.9)          # outside the team, not through it
        ys = sorted(p[1] for p in line[0]['points'])
        self.assertAlmostEqual(ys[0], at(d, 0, .12)[1], places=1)             # spans exactly the measured extent
        self.assertAlmostEqual(ys[1], at(d, 0, .88)[1], places=1)
        value = [t for t in ops_of(d, 'text') if t['text'] == '51.7 m (est.)']
        self.assertEqual(len(value), 1)
        x0, _, x1, _ = render.text_box(value[0])
        self.assertTrue(0 <= x0 and x1 <= x+.01)                               # outside the bracket, in the box
        leaders = [p for p in ops_of(d, 'path') if p['dash'] and p['stroke'] == render.INK_2]
        self.assertEqual(len(leaders), 2)                                      # back to A-01 and A-04
        right = render.diagram_ops(spec(*self.team(), dict(b, side='right')), 243)
        rx = self.line_of(right, True)[0]['points'][0][0]
        self.assertGreater(rx, at(d, .60, .88)[0]+render.BRACKET_GAP*scale*.9)

    def test_a_length_bracket_keeps_clear_of_the_labels(self):
        b = {'type': 'bracket', 'a': [.40, .5], 'b': [.60, .5], 'side': 'top', 'text': '21.0 m (est.)'}
        d = render.diagram_ops(spec(*self.team(), b), 243)
        line = self.line_of(d, False)
        self.assertEqual(len(line), 1)
        y = line[0]['points'][0][1]
        top_label = min(render.text_box(t)[1] for t in ops_of(d, 'text') if t['text'] == 'A-01')
        self.assertLess(y, top_label)                                          # above A-01's label, not on it
        self.assertEqual([t['text'] for t in ops_of(d, 'text') if t['text'] == '21.0 m (est.)'], ['21.0 m (est.)'])

    def test_without_a_side_it_avoids_opponents_and_flips_at_the_edge(self):
        opponents = [player(f'B-0{k}', .30, .15+.12*k) for k in range(1, 6)]
        b = {'type': 'bracket', 'a': [.5, .12], 'b': [.5, .88], 'text': '51.7 m (est.)'}
        d = render.diagram_ops(spec(*self.team(), *opponents, b), 243)
        x = self.line_of(d, True)[0]['points'][0][0]
        self.assertGreater(x, at(d, .60, 0)[0])                                # the side without the opponents
        edge = [player('A-01', .01, .12), player('A-04', .08, .88)]
        d = render.diagram_ops(spec(*edge, dict(b, a=[.05, .12], b=[.05, .88], side='left')), 243)
        x = self.line_of(d, True)[0]['points'][0][0]
        self.assertGreater(x, at(d, .08, 0)[0])                                # no room on the left: the right

    def test_evidence_placed_bracket_is_drawn_where_it_stands(self):
        # evidence.py emits a measure with bracket True, already BRACKET_OFFSET outside the team, and the two end ids
        m = {'type': 'measure', 'a': [.66, .12], 'b': [.66, .88], 'bracket': True, 'ids': ['A-01', 'A-04'],
             'text': '51.7 m (est.)'}
        d = render.diagram_ops(spec(*self.team(), m), 243)
        self.assertEqual(d['skipped'], [])
        line = self.line_of(d, True)
        self.assertEqual(len(line), 1)
        self.assertAlmostEqual(line[0]['points'][0][0], at(d, .66, 0)[0], places=1)   # not moved
        ticks = [p for p in ops_of(d, 'path') if p['stroke'] == render.INK and len(p['points']) == 2
                 and abs(p['points'][0][1]-p['points'][1][1]) < .01]
        self.assertEqual(len(ticks), 2)
        self.assertTrue(all(p['points'][1][0] < p['points'][0][0] for p in ticks))   # pointing back at the team
        leaders = [p for p in ops_of(d, 'path') if p['dash'] and p['stroke'] == render.INK_2]
        self.assertEqual(len(leaders), 2)
        ends = sorted(p['points'][1][0] for p in leaders)
        self.assertLess(ends[0], at(d, .45, 0)[0])                             # to A-01 (x .40) ...
        self.assertGreater(ends[1], at(d, .55, 0)[0])                          # ... and to A-04 (x .60)
        value = [t for t in ops_of(d, 'text') if t['text'] == '51.7 m (est.)'][0]
        self.assertGreater(render.text_box(value)[0], line[0]['points'][0][0])  # outside, beyond the bracket
        plain = render.diagram_ops(spec({'type': 'measure', 'a': [.3, .5], 'b': [.5, .5], 'text': '21.0 m (est.)'}))
        self.assertEqual(len([p for p in ops_of(plain, 'path') if p['stroke'] == render.INK]), 3)  # unchanged

    def test_invalid_brackets_are_skipped(self):
        d = render.diagram_ops(spec({'type': 'bracket', 'a': [.5, .5], 'b': [.5, .5], 'text': 'x'},
                                    {'type': 'bracket', 'a': [.5, .1], 'b': [2.0, .9]},
                                    {'type': 'bracket', 'b': [.5, .9]}))
        self.assertEqual([s['type'] for s in d['skipped']], ['bracket']*3)
        self.assertNotIn('bracket', d['drawn'])


class OverlayTests(unittest.TestCase):
    def overlay(self, tier, *elements):
        return {'tier': tier, 'width': 1920, 'height': 1080, 'elements': list(elements)}

    def test_scaled_to_the_placement_clipped_and_captioned(self):
        ov = self.overlay('image', {'type': 'player', 'id': 'A-03', 'team': 'A', 'x': 960, 'y': 540,
                                    'box': [940, 480, 40, 60]})
        o = render.overlay_ops(ov, 100, 200, 240, 135)
        self.assertEqual(o['ops'][0], render.clip(100, 200, 240, 135))
        self.assertEqual(o['ops'][-1], render.unclip())
        rings = ops_of(o, 'circle')
        self.assertTrue(rings and all(abs(c['cx']-220) < .01 and abs(c['cy']-267.5) < .01 for c in rings))
        boxes = [r for r in ops_of(o, 'rect') if r['stroke']]
        self.assertAlmostEqual(boxes[0]['x'], 100+940/8, places=2)
        self.assertAlmostEqual(boxes[0]['w'], 40/8, places=2)
        self.assertIn(render.OVERLAY_CAPTION, [t['text'] for t in ops_of(o, 'text')])
        self.assertEqual(o['note'], render.TIER_NOTES['image'])

    def test_each_tier_draws_only_what_it_allows(self):
        els = [{'type': 'player', 'id': 'A-01', 'team': 'A', 'x': 100, 'y': 500},
               {'type': 'player', 'id': 'A-02', 'team': 'A', 'x': 300, 'y': 520},
               {'type': 'unit', 'unit': 'D', 'ids': ['A-01', 'A-02']},
               {'type': 'measure', 'a': [100, 500], 'b': [300, 520], 'text': '12.0 m (est.)'},
               {'type': 'zone', 'polygon': [[100, 500], [300, 520], [200, 600]]},
               {'type': 'suggested', 'id': 'A-01', 'x': 150, 'y': 480},
               {'type': 'band', 'axis': 'x', 'from': .3, 'to': .4},
               {'type': 'arrow', 'id': 'A-01', 'points': [[100, 500], [140, 490]]},
               {'type': 'label', 'at': [500, 500], 'text': 'note'}]
        skipped = {t: sorted(s['type'] for s in render.overlay_ops(self.overlay(t, *els), 0, 0, 480, 270)['skipped'])
                   for t in ('image', 'pinned', 'world', 'bogus')}
        self.assertEqual(skipped['image'], ['band', 'measure', 'suggested', 'unit', 'zone'])
        self.assertEqual(skipped['pinned'], ['band', 'suggested'])
        self.assertEqual(skipped['world'], ['band'])
        self.assertEqual(skipped['bogus'], skipped['image'])         # an unknown tier is the most restrictive
        image = render.overlay_ops(self.overlay('image', *els), 0, 0, 480, 270)
        self.assertNotIn('12.0 m (est.)', [t['text'] for t in ops_of(image, 'text')])   # no metres on T1 video
        world = render.overlay_ops(self.overlay('world', *els), 0, 0, 480, 270)
        self.assertIn(render.ILLUSTRATION, [t['text'] for t in ops_of(world, 'text')])

    def test_a_bracket_on_the_video_needs_the_pinned_tier(self):
        els = [{'type': 'bracket', 'a': [100, 400], 'b': [100, 900], 'text': '40.1 m (est.)'}]
        image = render.overlay_ops(self.overlay('image', *els), 0, 0, 480, 270)
        self.assertEqual([s['type'] for s in image['skipped']], ['bracket'])
        pinned = render.overlay_ops(self.overlay('pinned', *els), 0, 0, 480, 270)
        self.assertEqual(pinned['skipped'], [])
        white = [p for p in ops_of(pinned, 'path') if p['stroke'] == render.WHITE]
        self.assertEqual(len(white), 3)                                        # the line and two end ticks
        self.assertIn('40.1 m (est.)', [t['text'] for t in ops_of(pinned, 'text')])

    def test_the_caption_says_what_the_overlay_drew(self):
        feet = [{'type': 'player', 'id': 'A-01', 'team': 'A', 'x': 100, 'y': 500},
                {'type': 'player', 'id': 'A-02', 'team': 'A', 'x': 300, 'y': 520}]
        unit = {'type': 'unit', 'unit': 'D', 'ids': ['A-01', 'A-02']}
        zone = {'type': 'zone', 'polygon': [[100, 500], [300, 520], [200, 600]]}
        note = lambda tier, *els: render.overlay_ops(self.overlay(tier, *els), 0, 0, 480, 270)['note']   # noqa
        self.assertEqual(note('world', *feet, unit, zone), render.TIER_NOTES['world'])
        self.assertEqual(note('world', *feet, unit), render.WORLD_LINES_NOTE)       # feet only: no pitch claim
        self.assertEqual(note('world', *feet), render.PLAYERS_NOTE)
        self.assertEqual(note('pinned', *feet, unit), render.TIER_NOTES['pinned'])
        self.assertEqual(note('pinned', *feet), render.PLAYERS_NOTE)
        self.assertEqual(note('image', *feet, unit), render.TIER_NOTES['image'])

    def test_an_overlay_without_an_image_size_draws_nothing(self):
        o = render.overlay_ops({'tier': 'world', 'elements': [{'type': 'label', 'at': [1, 1], 'text': 'x'}]},
                               0, 0, 100, 50)
        self.assertEqual(o['ops'], [])
        self.assertEqual(len(o['skipped']), 1)


class PngTests(unittest.TestCase):
    def test_colours_translucency_clip_and_dashes(self):
        ops = [render.rect(0, 0, 100, 50, fill='#ffffff'),
               render.circle(20, 25, 6, fill='#ff0000'),
               render.rect(50, 0, 50, 50, fill='#0000ff', alpha=.5),
               render.clip(0, 0, 40, 50), render.rect(30, 40, 60, 10, fill='#00ff00'), render.unclip(),
               render.path([(0, 5), (100, 5)], stroke='#000000', width=1, dash=[4, 4])]
        img = render.ops_to_png(ops, 100, 50, 2.0)
        self.assertEqual(img.shape, (100, 200, 3))
        self.assertEqual(tuple(int(v) for v in img[50, 40]), (0, 0, 255))           # BGR red at the circle
        b, g, r = (int(v) for v in img[60, 150])
        self.assertTrue(120 <= r <= 135 and 120 <= g <= 135 and b > 250)              # half-transparent blue
        self.assertEqual(tuple(int(v) for v in img[90, 70]), (0, 255, 0))            # inside the clip
        self.assertEqual(tuple(int(v) for v in img[90, 150]), tuple(int(v) for v in img[60, 150]))   # outside it
        row = img[10, :, 0]
        dark = row < 100
        self.assertTrue(dark.any() and not dark.all())                                # a dashed line has gaps

    def test_an_image_op_places_pixels(self):
        pic = np.zeros((40, 80, 3), np.uint8)
        pic[:, :40] = (0, 0, 255)
        ok, data = cv2.imencode('.jpg', pic, [cv2.IMWRITE_JPEG_QUALITY, 95])
        img = render.ops_to_png([{'op': 'image', 'x': 10, 'y': 10, 'w': 80, 'h': 40, 'jpeg': data.tobytes()}],
                                100, 60, 1.0)
        self.assertGreater(int(img[30, 20, 2]), 200)
        self.assertLess(int(img[30, 80, 2]), 60)
        self.assertEqual(tuple(int(v) for v in img[5, 5]), (255, 255, 255))

    def test_a_diagram_png_has_the_pitch_and_the_players(self):
        s = spec(player('A-03', .4, .3), player('B-07', .6, .7))
        img = render.diagram_png(s, 720)
        d = render.diagram_ops(s)
        self.assertEqual(img.shape[1], 720)
        x, y = at(d, .4, .3)
        k = 720/d['width']
        b, g, r = (int(v) for v in img[int(round(y*k)), int(round(x*k))])
        self.assertEqual((r, g, b), (0xb9, 0x8c, 0xff))                              # team A's colour
        self.assertGreater(len(np.unique(img.reshape(-1, 3), axis=0)), 20)


if __name__ == '__main__':
    unittest.main()
