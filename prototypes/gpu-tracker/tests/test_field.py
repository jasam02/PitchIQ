import unittest

import numpy as np

import broadcast
from scene import H, W, draw_person, person_box, pitch_frame
from soccer.pitch import FieldFilter, PitchDetector, zone_of
from soccer.relevance import assess


def detect(frame):
    return PitchDetector().analyze(frame, 0.0)


def det(foot_x, foot_y, height=.12, score=.9):
    return {'box': person_box(foot_x, foot_y, height), 'score': score, 'cls': 'player'}


class PitchTests(unittest.TestCase):
    def test_touchlines_clip_the_run_off(self):
        model = detect(pitch_frame())
        self.assertTrue(model.reliable)
        self.assertEqual({l['side'] for l in model.lines}, {'near', 'far'})
        f = FieldFilter()
        self.assertEqual(zone_of(model, person_box(.4, .6), f)[0], 'inside')
        self.assertEqual(zone_of(model, person_box(.4, .905), f)[0], 'boundary')   # feet on the touchline
        self.assertEqual(zone_of(model, person_box(.4, .95), f)[0], 'outside')     # bench on the run-off grass
        self.assertEqual(zone_of(model, person_box(.4, .29), f)[0], 'outside')     # behind the far touchline
        self.assertEqual(zone_of(model, person_box(.4, .15), f)[0], 'outside')     # stands

    def test_region_is_redetected_after_the_camera_moves(self):
        wide, zoomed = detect(pitch_frame()), detect(pitch_frame(far=.15, near=1.2))
        top = lambda m: min(p[1] for p in m.polygon)
        bottom = lambda m: max(p[1] for p in m.polygon)
        self.assertLess(top(zoomed), top(wide)-.1)
        self.assertGreater(bottom(zoomed), .99)  # near touchline out of view: grass runs to the bottom
        self.assertEqual(zone_of(zoomed, person_box(.4, .95), FieldFilter())[0], 'inside')

    def test_no_pitch_in_a_crowd_shot(self):
        crowd = np.random.default_rng(1).integers(40, 200, size=(H, W, 3), dtype=np.uint8)
        model = detect(crowd)
        self.assertFalse(model.reliable)
        self.assertEqual(zone_of(model, person_box(.5, .5), FieldFilter())[0], 'unknown')

    def test_green_stands_behind_boards_are_not_pitch(self):
        frame = pitch_frame()
        frame[:int(.27*H)] = (55, 135, 50)  # green seats above the board band
        model = detect(frame)
        self.assertTrue(model.reliable)
        self.assertGreater(min(p[1] for p in model.polygon), .3)
        self.assertEqual(zone_of(model, person_box(.5, .2), FieldFilter())[0], 'outside')

    def test_touchline_tolerance_scales_with_the_slider(self):
        model = detect(pitch_frame())
        box = person_box(.4, .912)
        self.assertEqual(zone_of(model, box, FieldFilter.with_margin(.35))[0], 'boundary')
        self.assertEqual(zone_of(model, box, FieldFilter.with_margin(0))[0], 'outside')


class BroadcastPitchTests(unittest.TestCase):
    """Perspective views with thin painted lines, a grass run-off strip and white advertising boards."""

    def model(self, P, **kw):
        return PitchDetector().analyze(broadcast.render(P, **kw), 0.0)

    def test_boundary_sits_on_the_touchline_not_the_boards(self):
        P = broadcast.camera()
        model = self.model(P)
        far = next(l for l in model.lines if l['side'] == 'far')
        touchline = broadcast.image_point(P, 52.5, 0)[1]
        self.assertAlmostEqual(far['a'][1], touchline, delta=.004)
        self.assertAlmostEqual(min(p[1] for p in model.polygon), touchline, delta=.004)
        f = FieldFilter()
        height = broadcast.player_px(P, 40, 0)/broadcast.H  # a person's box height at the far touchline
        board_side = broadcast.image_point(P, 40, -2.5)  # a coach on the run-off, in front of the boards
        self.assertEqual(zone_of(model, person_box(board_side[0], board_side[1], height), f)[0], 'outside')
        on_line = broadcast.image_point(P, 40, .2)
        self.assertIn(zone_of(model, person_box(on_line[0], on_line[1], height), f)[0], ('inside', 'boundary'))

    def test_goal_line_found_and_grass_behind_the_goal_excluded(self):
        P = broadcast.camera(centre_x=20, span=50)
        model = self.model(P)
        self.assertEqual(model.sources()['left'], 'line')
        self.assertTrue(any(g['side'] == 'left' and g['kind'] == 'goal-line' for g in model.goals))
        behind = broadcast.image_point(P, -3, 34)
        self.assertEqual(zone_of(model, person_box(behind[0], behind[1], .1), FieldFilter())[0], 'outside')

    def test_penalty_box_front_is_not_a_goal_line(self):
        model = self.model(broadcast.camera())  # both penalty boxes visible, goal lines out of view
        self.assertEqual(model.sources()['left'], 'grass edge')
        self.assertEqual(model.sources()['right'], 'grass edge')
        self.assertEqual(sorted(g['side'] for g in model.goals if g['kind'] == 'box-front'), ['left', 'right'])

    def test_boundary_is_stable_over_time(self):
        P = broadcast.camera()
        detector = PitchDetector()
        first = detector.analyze(broadcast.render(P), 0.0)
        # A frame where the far touchline is hidden (e.g. by players): the boundary is carried, not dropped.
        frame = broadcast.render(P)
        y = int(broadcast.image_point(P, 52.5, 0)[1]*broadcast.H)
        frame[y-3:y+4, :] = broadcast.GRASS
        carried = detector.analyze(frame, .2, np.eye(3), True)
        far = next(l for l in carried.lines if l['side'] == 'far')
        self.assertGreater(far['age'], 0)
        self.assertAlmostEqual(far['a'][1], next(l for l in first.lines if l['side'] == 'far')['a'][1], delta=.002)


class RelevanceTests(unittest.TestCase):
    def setUp(self):
        self.frame = pitch_frame()
        self.model = detect(self.frame)

    def test_audience_and_staff_are_rejected_with_reasons(self):
        players = [det(.2+.1*i, .5+.05*(i % 4)) for i in range(7)]
        fans = [det(.3+.03*i, .2, .06) for i in range(4)]
        coach = det(.5, .97)
        accepted, continuation, rejected, size = assess(players+fans+[coach], self.model, FieldFilter())
        self.assertEqual(len(accepted), 7)
        reasons = [r['reason'] for r in rejected]
        self.assertEqual(reasons.count('audience'), 4)
        self.assertIn('outside-pitch', reasons)
        self.assertEqual([d['box'] for d in continuation], [coach['box']])  # may continue a known player, never start one
        self.assertTrue(size.reliable)

    def test_assistant_referee_on_the_touchline_is_kept_as_an_official(self):
        players = [det(.2+.1*i, .5+.05*(i % 4)) for i in range(7)]
        assistant = dict(det(.3, .95), cls='referee', score=.8)  # beyond the touchline tolerance
        unsure = dict(det(.6, .95), cls='referee', score=.3)
        bench = [dict(det(.75+.03*i, .96), cls='referee', score=.8) for i in range(3)]  # bibs in a group
        accepted, _, rejected, _ = assess(players+[assistant, unsure]+bench, self.model, FieldFilter())
        kept = next(d for d in accepted if d['box'] == assistant['box'])
        self.assertEqual((kept['zone'], kept.get('official')), ('boundary', True))
        self.assertEqual(len(accepted), 8)
        self.assertEqual({r['reason'] for r in rejected}, {'outside-pitch'})
        self.assertEqual(len(rejected), 4)

    def test_foot_point_not_box_overlap_decides(self):
        # A tall box that overlaps the pitch, but whose feet are on the run-off beyond the near touchline.
        tall = det(.5, .96, height=.4)
        accepted, _, rejected, _ = assess([tall], self.model, FieldFilter())
        self.assertFalse(accepted)
        self.assertEqual(rejected[0]['reason'], 'outside-pitch')

    def test_implausible_size_is_rejected(self):
        players = [det(.15+.1*i, .45+.07*(i % 5), .1+.06*(.45+.07*(i % 5)-.45)) for i in range(8)]
        giant = det(.5, .6, .45)
        lying = {'box': [.45, .66, .08, .02], 'score': .8, 'cls': 'player'}
        accepted, _, rejected, _ = assess(players+[giant, lying], self.model, FieldFilter())
        self.assertIn('implausible-size', [r['reason'] for r in rejected])
        self.assertIn(lying['box'], [d['box'] for d in accepted])

    def test_user_boundary_is_stricter(self):
        inside = det(.2, .6)
        polygon = [(.3, .3), (.9, .3), (.9, .95), (.3, .95)]
        accepted, _, rejected, _ = assess([inside, det(.6, .6)], self.model, FieldFilter(), user_polygon=polygon)
        self.assertEqual(len(accepted), 1)
        self.assertEqual(rejected[0]['box'], inside['box'])

    def test_disabled_filter_keeps_everyone(self):
        accepted, _, rejected, _ = assess([det(.4, .2), det(.4, .6)], self.model, FieldFilter(enabled=False))
        self.assertEqual(len(accepted), 2)
        self.assertFalse(rejected)


class DrawnPeopleTests(unittest.TestCase):
    def test_people_do_not_break_the_pitch(self):
        frame = pitch_frame()
        for i in range(12):
            draw_person(frame, person_box(.1+.07*i, .45+.04*(i % 5)), 'A' if i % 2 else 'B')
        model = detect(frame)
        self.assertTrue(model.reliable)
        self.assertEqual({l['side'] for l in model.lines}, {'near', 'far'})


if __name__ == '__main__':
    unittest.main()
