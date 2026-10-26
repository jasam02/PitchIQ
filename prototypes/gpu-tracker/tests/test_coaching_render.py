"""Coaching figures as drawing operations (soccer/coaching/render.py): every DiagramSpec element type becomes the
operations the PDF writes and the PNG preview rasterizes; estimates and suggestions are never drawn as facts; an
arrow is cut where its samples are more than a second apart; a team width or length is a bracket beside the team, not
a line through it; a video overlay draws only what its tier allows, scaled to its placement and captioned as an
analysis overlay.

Coaching v2 annotations: the page's colour rule (CIEDE2000 to Sharma's 14 pairs, the shared test vectors) and its
print twins, the focus / support / relation / channel / suggested-area / movement-arrow / caption drawings, the
element budget and the tier table shared with schema.py, naming, the coach's adjusted area, and dotted lines in
the PNG preview.

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


# ---------------------------------------------------------------------------------------------------------------
# Coaching v2 annotations (DESIGN 5.4-5.5): colours, the new elements, tier gating, budget

from soccer.coaching import schema as S  # noqa: E402

SHARMA = [  # Sharma, Wu and Dalal (2005), CIEDE2000 test data: (Lab 1, Lab 2, dE00)
    ([50, 2.6772, -79.7751], [50, 0, -82.7485], 2.0425), ([50, 3.1571, -77.2803], [50, 0, -82.7485], 2.8615),
    ([50, 2.8361, -74.02], [50, 0, -82.7485], 3.4412), ([50, -1.3802, -84.2814], [50, 0, -82.7485], 1.0),
    ([50, 0, 0], [50, -1, 2], 2.3669), ([50, 2.49, -0.001], [50, -2.49, 0.0009], 7.1792),
    ([50, 2.5, 0], [73, 25, -18], 27.1492), ([50, 2.5, 0], [61, -5, 29], 22.8977),
    ([50, 2.5, 0], [56, -27, -3], 31.903),
    ([50, 2.5, 0], [58, 24, 15], 19.4535), ([60.2574, -34.0099, 36.2677], [60.4626, -34.1751, 39.4387], 1.2644),
    ([63.0109, -31.0961, -5.8663], [62.8187, -29.7946, -4.0864], 1.263),
    ([22.7233, 20.0904, -46.694], [23.0331, 14.973, -42.5619], 2.0373),
    ([90.8027, -2.0831, 1.441], [91.1528, -1.6435, 0.0447], 1.4441)]


def v2spec(*elements, **extra):
    """A v2 finding diagram: the reviewed player A-04 (focus), the mover A-05 (support), an opponent, the rest faint."""
    base = [player('A-02', .3, .6), player('B-10', .21, .29),
            {'type': 'focus', 'id': 'A-04', 'x': .22, 'y': .31, 'tone': 'review', 'label': True},
            {'type': 'support', 'id': 'A-05', 'team': 'A', 'x': .18, 'y': .45, 'why': 'nearest teammate'}]
    return spec(*(base+list(elements)), preset='pressure_cover', **extra)


ZONE = {'type': 'suggestedZone', 'polygon': [[.17, .36], [.21, .36], [.21, .40], [.17, .40]], 'kind': 'zone',
        'alternative': 'alt-ff134036e746', 'mover': 'A-05', 'label': 'possible covering area'}


def strokes(d, colour):
    return [o for o in d['ops'] if o.get('stroke') == colour]


class PaletteTests(unittest.TestCase):
    def test_ciede2000_reproduces_the_published_pairs(self):
        for a, b, want in SHARMA:
            self.assertAlmostEqual(render.ciede2000(a, b), want, 4)
            self.assertAlmostEqual(render.ciede2000(b, a), want, 4)

    def test_the_page_rule_and_its_test_vectors(self):
        p = render.annotation_palette('#b98cff', '#f6b564')         # the default teams
        self.assertEqual((p['review'], p['zone'], p['reviewFallback'], p['zoneFallback']),
                         ('#ff4d6d', '#7dffb0', False, False))
        self.assertEqual(p['scores']['review'], 32.1)
        p = render.annotation_palette('#ff0000', '#00ff00')
        self.assertEqual((p['review'], p['zone'], p['zoneFallback']), ('#ff2fd1', '#5cf2ff', True))
        p = render.annotation_palette('#ff3c9f', '#ffffff')
        self.assertEqual((p['review'], p['reviewFallback'], p['scores']['review']), ('#ffffff', True, 13.1))
        self.assertLess(render.delta_e('#c6ff4d', render.PALETTE['ball']), 10)   # lime is the ball's colour
        self.assertAlmostEqual(render.contrast_ratio('#ffffff', '#000000'), 21.0)

    def test_the_print_twins_keep_the_hue_and_pass_the_rule_on_the_print_pitch(self):
        greens = (render.PITCH_FILL, render.PITCH_STRIPE)
        for screen, twin in render.PRINT_ZONE.items():
            self.assertTrue(all(render.contrast_ratio(screen, g) < 3 for g in greens))   # the page's colour fails
            self.assertTrue(all(render.contrast_ratio(twin, g) >= 3.3 for g in greens))
            self.assertLess(abs(render.lab_of(twin)[2]/max(1e-6, abs(render.lab_of(twin)[1]))
                                - render.lab_of(screen)[2]/max(1e-6, abs(render.lab_of(screen)[1]))), 1.0)
        p = render.print_palette({'A': '#b98cff', 'B': '#f6b564'})
        self.assertEqual((p['review'], p['zone'], p['zoneFallback']), ('#ff4d6d', '#448a5f', False))
        self.assertEqual(p['halo'], render.WHITE)
        for a, b in (('#ff0000', '#00ff00'), ('#448a5f', '#678528'), ('#ffffff', '#ff3c9f'), ('#34888f', '#ff4d6d')):
            p = render.print_palette({'A': a, 'B': b})
            ok = min(render.contrast_ratio(p['zone'], g) for g in greens) >= 3 and \
                min(render.delta_e(p['zone'], t) for t in (a, b)) >= 20
            self.assertTrue(ok or p['zoneFallback'], (a, b))
            self.assertNotEqual(p['review'], render.WHITE)              # white vanishes on paper: the ink instead
        self.assertEqual(render.print_palette({'A': '#ff3c9f', 'B': '#ffffff'})['review'], render.INK)


class V2DiagramTests(unittest.TestCase):
    def test_the_reviewed_player_is_ringed_in_the_review_colour_and_the_others_are_faint(self):
        d = render.diagram_ops(v2spec(), 243)
        rings = [o for o in strokes(d, '#ff4d6d') if o['op'] == 'circle' and o['cy'] < d['height']-40]   # no legend
        self.assertEqual(len(rings), 1)
        self.assertAlmostEqual(rings[0]['cx'], at(d, .22, .31)[0], 1)
        support = [o for o in d['ops'] if o['op'] == 'circle' and o.get('dash') and o.get('stroke') == render.INK_2
                   and o['cy'] < d['height']-40]
        self.assertEqual(len(support), 1)
        self.assertAlmostEqual(support[0]['cy'], at(d, .18, .45)[1], 1)
        faint = [o for o in d['ops'] if o['op'] == 'circle' and o.get('fill') and o['alpha'] == .5]
        self.assertEqual(len(faint), 2)                                 # A-02 and B-10 are not about the moment
        texts = [o['text'] for o in ops_of(d, 'text')]
        self.assertIn('A-04', texts)
        self.assertNotIn('A-02', texts)
        self.assertIn('player the moment is about', d['legend'])

    def test_a_level_that_names_nobody_gets_no_label_of_the_people(self):
        s = v2spec()
        s['elements'][2]['label'] = False
        texts = [o['text'] for o in ops_of(render.diagram_ops(s, 243), 'text')]
        self.assertNotIn('A-04', texts)
        self.assertNotIn('A-05', texts)

    def test_relations_are_dashed_when_missing_and_find_their_ends(self):
        rel = {'type': 'relation', 'a': 'A-04', 'b': 'A-05', 'kind': 'cover', 'ok': False, 'text': '14 m'}
        d = render.diagram_ops(v2spec(rel), 243)
        line = [o for o in d['ops'] if o['op'] == 'path' and o.get('stroke') == render.INK and o['alpha'] == .6]
        self.assertEqual(len(line), 1)
        self.assertTrue(line[0]['dash'])
        self.assertEqual([round(v, 1) for v in line[0]['points'][0]], [round(v, 1) for v in at(d, .22, .31)])
        self.assertIn('14 m', [o['text'] for o in ops_of(d, 'text')])
        d = render.diagram_ops(v2spec(dict(rel, ok=True)), 243)
        self.assertFalse([o for o in d['ops'] if o.get('stroke') == render.INK and o['alpha'] == .6][0]['dash'])
        d = render.diagram_ops(v2spec(ZONE, dict(rel, b='zone'), dict(rel, b='goal'), dict(rel, b='ball', pb=[.5, .5])),
                               243)
        ends = [o['points'][1] for o in d['ops'] if o.get('stroke') == render.INK and o['alpha'] == .6]
        self.assertEqual(len(ends), 3)
        self.assertAlmostEqual(ends[0][0], at(d, .19, .38)[0], 1)       # the area's centre
        self.assertAlmostEqual(ends[1][0], at(d, 0.0, .5)[0], 1)        # attacking right: the own goal is on the left
        self.assertAlmostEqual(ends[2][1], at(d, .5, .5)[1], 1)
        d = render.diagram_ops(v2spec(dict(rel, b='A-99')), 243)
        self.assertEqual(d['skipped'][0]['reason'], 'relation end not placed on the pitch')

    def test_a_suggested_area_is_dotted_hatched_and_faintly_filled_over_a_halo(self):
        d = render.diagram_ops(v2spec(ZONE), 243)
        zone = '#448a5f'
        outline = [o for o in strokes(d, zone) if o['op'] == 'path' and o.get('closed')]
        self.assertEqual(len(outline), 1)
        self.assertEqual(outline[0]['dash'][0], render.DOT)             # round dots (round caps)
        hatch = [o for o in strokes(d, zone) if o['op'] == 'path' and not o.get('closed')]
        self.assertGreater(len(hatch), 1)
        x0, y0 = at(d, .17, .36)
        x1, y1 = at(d, .21, .40)
        for o in hatch:
            for x, y in o['points']:
                self.assertTrue(x0-.02 <= x <= x1+.02 and y0-.02 <= y <= y1+.02)
        self.assertTrue([o for o in d['ops'] if o.get('fill') == zone and o['alpha'] == .1])
        self.assertTrue([o for o in d['ops'] if o.get('stroke') == render.WHITE and o.get('closed')])
        self.assertIn('possible covering area', [o['text'] for o in ops_of(d, 'text')])
        self.assertIn('possible area (a suggestion, not tracked data)', d['legend'])
        point = render.diagram_ops(v2spec(dict(ZONE, kind='point')), 243)
        self.assertTrue([o for o in strokes(point, zone) if o['op'] == 'circle' and o['dash']])
        self.assertIn('possible spot (a suggestion, not tracked data)', point['legend'])

    def test_a_movement_arrow_is_dashed_with_its_head_at_the_area(self):
        arrow = {'type': 'moveArrow', 'id': 'A-05', 'from': [.18, .45], 'to': [.19, .38]}
        d = render.diagram_ops(v2spec(arrow), 243)
        line = [o for o in strokes(d, '#448a5f') if o['op'] == 'path']
        self.assertTrue(line[0]['dash'])
        head = [o for o in d['ops'] if o.get('fill') == '#448a5f' and o['op'] == 'path' and o['closed']
                and o['points'][0][1] < d['height']-40]
        self.assertEqual(len(head), 1)
        self.assertEqual([round(v, 1) for v in head[0]['points'][0]], [round(v, 1) for v in at(d, .19, .38)])

    def test_the_channel_is_the_review_colour_faint(self):
        ch = {'type': 'channel', 'polygon': [[.3, 0], [.45, 0], [.45, .33], [.3, .33]], 'kind': 'vacated'}
        d = render.diagram_ops(v2spec(ch), 243)
        self.assertTrue([o for o in d['ops'] if o.get('fill') == '#ff4d6d' and o['alpha'] == .12])
        self.assertIn('area the check is about', d['legend'])

    def test_the_element_budget_is_kept(self):
        extra = [{'type': 'support', 'id': f'A-{k:02d}', 'team': 'A', 'x': .1*k, 'y': .5, 'why': 'x'}
                 for k in (6, 7, 8)]
        d = render.diagram_ops(v2spec(*extra, ZONE, dict(ZONE)), 243)
        self.assertEqual(d['drawn']['support'], 3)
        self.assertEqual(d['drawn']['suggestedZone'], 1)
        self.assertEqual(sorted(s['type'] for s in d['skipped']), ['suggestedZone', 'support'])
        self.assertEqual(render.V2_BUDGET, S.ELEMENT_BUDGET)
        self.assertEqual(render.V2_TIERS, S.TIER_ELEMENTS)

    def test_the_caption_stands_in_a_strip_above_the_pitch(self):
        cap = {'type': 'caption', 'text': 'Pressure on the ball; nobody covered behind it.', 'anchor': 'top-left'}
        plain = render.diagram_ops(v2spec(), 243)
        d = render.diagram_ops(v2spec(cap), 243)
        strip = d['height']-plain['height']
        self.assertGreater(strip, 5)
        text = next(o for o in ops_of(d, 'text') if o['text'].startswith('Pressure on the ball'))
        self.assertLess(text['y'], strip)                                # above the pitch, never over it
        self.assertAlmostEqual(next(o for o in ops_of(d, 'circle') if o.get('stroke') == '#ff4d6d')['cy'],
                               at(plain, .22, .31)[1]+strip, 1)
        self.assertEqual(render.diagram_ops(v2spec(cap), 243, caption=False)['height'], plain['height'])
        strong = v2spec(cap)
        strong['elements'][2]['tone'] = 'strength'
        d = render.diagram_ops(strong, 243)
        self.assertTrue([o for o in ops_of(d, 'path')
                         if o.get('stroke') == render.BRAND_DARK and o['points'][0][1] < strip])

    def test_the_coachs_adjusted_area_is_thin_and_solid_beside_pitchiqs(self):
        d = render.diagram_ops(v2spec(ZONE), 243, adjustments=[{'polygon': [[.16, .35], [.22, .35], [.22, .41]]}])
        adj = [o for o in d['ops'] if o.get('stroke') == render.INK_2 and o['op'] == 'path' and o.get('closed')]
        self.assertEqual(len(adj), 1)
        self.assertFalse(adj[0]['dash'])
        self.assertIn(render.COACH_AREA, [o['text'] for o in ops_of(d, 'text')])

    def test_v1_diagrams_are_unchanged_by_the_v2_code(self):
        s = spec(player('A-03', .4, .3, emphasis=True), player('B-07', .6, .7))
        d = render.diagram_ops(s, 243)
        self.assertIn('highlighted', d['legend'])
        self.assertTrue([o for o in d['ops'] if o.get('stroke') == render.BRAND_DARK and o['op'] == 'circle'])


class V2OverlayTests(unittest.TestCase):
    OV = {'tier': 'world', 'width': 1920, 'height': 1080, 'elements': [
        {'type': 'focus', 'id': 'A-04', 'x': 787.2, 'y': 626.4, 'tone': 'review', 'label': True,
         'box': [768.0, 540.0, 38.4, 86.4]},
        {'type': 'support', 'id': 'A-05', 'team': 'A', 'x': 640.0, 'y': 700.0, 'why': 'x', 'box': [622, 610, 36, 90]},
        {'type': 'relation', 'a': 'A-04', 'b': 'A-05', 'kind': 'cover', 'ok': False, 'text': '14 m',
         'pa': [787.2, 626.4], 'pb': [640.0, 700.0]},
        {'type': 'channel', 'polygon': [[500, 500], [900, 500], [900, 800], [500, 800]], 'kind': 'vacated'},
        {'type': 'suggestedZone', 'polygon': [[600.0, 700.0], [680.0, 700.0], [690.0, 760.0], [590.0, 760.0]],
         'kind': 'zone', 'alternative': 'alt-ff134036e746', 'mover': 'A-05', 'label': 'possible covering area'},
        {'type': 'moveArrow', 'id': 'A-05', 'from': [640, 700], 'to': [640, 730]},
        {'type': 'caption', 'text': 'Pressure without cover', 'anchor': 'top-left'}]}

    def overlay(self, tier, **kw):
        return render.overlay_ops(dict(self.OV, tier=tier), 0, 0, 243, 136.7, **kw)

    def test_each_tier_draws_only_what_it_allows(self):
        world = self.overlay('world')
        self.assertEqual(world['skipped'], [])
        self.assertTrue(strokes(world, '#7dffb0'))                      # the page's mint over the video
        self.assertEqual(world['note'], render.V2_TIER_NOTES['world'])
        pinned = self.overlay('pinned')
        self.assertEqual(sorted(s['type'] for s in pinned['skipped']), ['channel', 'moveArrow', 'suggestedZone'])
        self.assertFalse(strokes(pinned, '#7dffb0'))
        self.assertEqual(pinned['note'], render.V2_TIER_NOTES['pinned'])
        image = self.overlay('image')
        self.assertEqual(sorted(s['type'] for s in image['skipped']),
                         ['channel', 'moveArrow', 'relation', 'suggestedZone'])
        box = [o for o in strokes(image, '#ff4d6d') if o['op'] == 'rect']
        self.assertEqual(len(box), 1)                                   # the box outlined, no ellipse
        self.assertLess(box[0]['x'], 768*243/1920)
        self.assertNotIn('Pressure without cover', [o['text'] for o in ops_of(world, 'text')])   # printed beside

    def test_the_reviewed_player_is_ringed_at_the_feet_with_a_label(self):
        world = self.overlay('world', labels={'A-04': 'A-03'})
        ring = [o for o in strokes(world, '#ff4d6d') if o['op'] == 'path' and o['closed'] and o['alpha'] == 1]
        self.assertEqual(len(ring), 1)
        ys = [p[1] for p in ring[0]['points']]
        self.assertAlmostEqual((min(ys)+max(ys))/2, 626.4*136.7/1080, 0)
        self.assertIn('A-03', [o['text'] for o in ops_of(world, 'text')])
        quiet = dict(self.OV)
        quiet['elements'] = [dict(e, label=False) if e['type'] == 'focus' else e for e in self.OV['elements']]
        texts = [o['text'] for o in ops_of(render.overlay_ops(quiet, 0, 0, 243, 136.7), 'text')]
        self.assertNotIn('A-04', texts)

    def test_strokes_keep_their_printed_weight_when_the_frame_is_magnified(self):
        a = self.overlay('world', unit=.6)
        b = render.overlay_ops(self.OV, -100, -50, 486, 273.4, unit=.6)
        wa = sorted({o['width'] for o in strokes(a, '#ff4d6d')})
        wb = sorted({o['width'] for o in strokes(b, '#ff4d6d')})
        self.assertEqual(wa, wb)

    def test_a_projected_area_may_reach_past_the_frame_edge(self):
        far = [[-900, 500], [900, 500], [900, 2600], [-900, 2600]]
        ov = dict(self.OV, elements=[{'type': 'channel', 'polygon': far, 'kind': 'weak_side'}])
        self.assertEqual(render.overlay_ops(ov, 0, 0, 243, 136.7)['skipped'], [])
        ov['elements'][0]['polygon'][0] = [-1e7, 0]
        self.assertEqual(len(render.overlay_ops(ov, 0, 0, 243, 136.7)['skipped']), 1)


class DotTests(unittest.TestCase):
    def test_a_dash_shorter_than_its_line_rasterizes_as_round_dots(self):
        ops = [render.path([(5, 10), (95, 10)], stroke='#000000', width=2.0, dash=[render.DOT, 4.0])]
        img = render.ops_to_png(ops, 100, 20, 4)
        row = (img[38:42, :, 0] < 128).any(axis=0)
        runs = int(np.sum(row[1:] & ~row[:-1]))
        self.assertGreaterEqual(runs, 15)                               # about 90 / 4 dots, each a separate blob


if __name__ == '__main__':
    unittest.main()
