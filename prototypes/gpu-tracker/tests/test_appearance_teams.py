import unittest

import numpy as np

from scene import H, W, draw_person, person_box, pitch_frame
from soccer.appearance import (Descriptor, add_to_gallery, compare, decode_descriptor, describe, descriptor_distance,
                               encode_descriptor, grass_reference)
from soccer.teams import RoleEvidence, TeamModel, decide_role, fit_team_model, kit_vote, with_referee


def px(box):
    return [box[0]*W, box[1]*H, (box[0]+box[2])*W, (box[1]+box[3])*H]


class AppearanceTests(unittest.TestCase):
    def setUp(self):
        self.frame = pitch_frame()
        self.ref = grass_reference(self.frame)

    def describe(self, kit, foot=(.5, .6), height=.15, others=()):
        box = person_box(*foot, height)
        draw_person(self.frame, box, kit)
        return describe(self.frame, px(box), [px(b) for b in others], None, self.ref)

    def test_part_descriptor_separates_kits_and_ignores_grass(self):
        a1, a2, b = self.describe('A', (.3, .6)), self.describe('A', (.5, .7)), self.describe('B', (.7, .6))
        self.assertGreater(a1.quality, .45)
        self.assertLess(descriptor_distance(a1, a2)['jersey'], .15)
        self.assertGreater(descriptor_distance(a1, b)['jersey'], .5)
        self.assertGreater(descriptor_distance(a1, b)['shorts'], .5)

    def test_green_jersey_is_not_discarded_as_grass(self):
        kit = {'jersey': (40, 200, 20), 'shorts': (240, 240, 240), 'socks': (40, 200, 20)}
        d = self.describe(kit, (.4, .6))
        self.assertTrue(d.has_jersey)

    def test_occlusion_and_truncation_lower_quality(self):
        clean = self.describe('A', (.3, .6))
        front = person_box(.305, .63, .15)
        hidden = self.describe('A', (.6, .6), others=[person_box(.605, .63, .15)])
        draw_person(self.frame, front, 'B')
        edge = describe(self.frame, [-20, 300, 25, 420], [], None, self.ref)
        self.assertLess(hidden.quality, clean.quality)
        self.assertLess(edge.quality, clean.quality)

    def test_gallery_keeps_several_diverse_good_samples(self):
        g = []
        rng = np.random.default_rng(0)
        for i in range(12):
            e = rng.normal(size=512)
            d = Descriptor(np.r_[1, np.zeros(23)], np.r_[1, np.zeros(11)], np.r_[1, np.zeros(7)], np.full(24, .5), .5+.04*(i % 5), e/np.linalg.norm(e))
            g = add_to_gallery(g, d, i*1.0, 5)
        g = add_to_gallery(g, Descriptor(np.zeros(24), np.zeros(12), np.zeros(8), np.full(24, -1.0), .1), 20.0, 5)
        self.assertEqual(len(g), 5)
        self.assertTrue(all(s.d.quality >= .45 for s in g))
        self.assertIsNotNone(compare(g, [g[0].d])['appearance'])

    def test_descriptor_round_trip(self):
        d = self.describe('B')
        e = np.random.default_rng(3).normal(size=512)
        d.embedding = e/np.linalg.norm(e)
        back = decode_descriptor(encode_descriptor(d))
        self.assertAlmostEqual(float(np.dot(back.embedding, d.embedding)), 1.0, places=3)
        self.assertLess(descriptor_distance(back, d)['total'], .01)


def hist(i, j=None, w=.7):
    v = np.zeros(24)
    v[i] = w if j is not None else 1
    if j is not None:
        v[j] = 1-w
    return v


class TeamTests(unittest.TestCase):
    def test_two_kits_learned_from_players(self):
        rng = np.random.default_rng(1)
        samples = []
        for i in range(8):
            samples.append({'jersey': hist(4, 5, .6+.1*rng.random()), 'weight': 1, 'order': i})
            samples.append({'jersey': hist(16, 17, .6+.1*rng.random()), 'weight': 1, 'order': i+.5})
        samples.append({'jersey': hist(8, 9), 'weight': 1, 'order': 99})  # one odd crop
        model = fit_team_model(samples)
        self.assertTrue(model.ready)
        self.assertEqual(kit_vote(model, hist(4, 5)).team, 'A')
        self.assertEqual(kit_vote(model, hist(16, 17)).team, 'B')
        again = fit_team_model([dict(s, order=-s['order']) for s in samples], model)
        self.assertEqual(kit_vote(again, hist(4, 5)).team, 'A', 'refits keep the A/B naming')

    def test_one_kit_is_not_two_teams(self):
        samples = [{'jersey': hist(4, 5, .5+.03*i), 'weight': 1, 'order': i} for i in range(12)]
        self.assertFalse(fit_team_model(samples).ready)

    def test_referee_kit_is_an_outlier(self):
        model = with_referee(TeamModel(hist(4, 5), hist(16, 17), None, .1, 20), [hist(8, 9), hist(8, 9, .65)])
        v = kit_vote(model, hist(8, 9))
        self.assertTrue(v.outlier and v.ref_like)

    def test_role_needs_temporal_evidence(self):
        model = TeamModel(hist(4, 5), hist(16, 17), None, .1, 20)
        e = RoleEvidence()
        e.add('player', .9, kit_vote(model, hist(4, 5)), False)
        self.assertEqual(decide_role(e, model).label, 'CANDIDATE')
        for _ in range(6):
            e.add('player', .9, kit_vote(model, hist(4, 5)), False)
        self.assertEqual(decide_role(e, model).label, 'PLAYER_TEAM_A')
        e.add('player', .9, kit_vote(model, hist(16, 17)), False)  # one bad frame
        self.assertEqual(decide_role(e, model).label, 'PLAYER_TEAM_A')

    def test_referee_from_detector_votes(self):
        model = with_referee(TeamModel(hist(4, 5), hist(16, 17), None, .1, 20), [hist(8, 9), hist(8, 9)])
        e = RoleEvidence()
        for _ in range(6):
            e.add('referee', .8, kit_vote(model, hist(8, 9)), False)
        self.assertEqual(decide_role(e, model).label, 'REFEREE')
        confused = RoleEvidence()
        for _ in range(6):
            confused.add('referee', .5, kit_vote(model, hist(4, 5)), False)
        for _ in range(2):
            confused.add('player', .5, kit_vote(model, hist(4, 5)), False)
        self.assertEqual(decide_role(confused, model).label, 'CANDIDATE', 'a referee class on a team kit is held back')

    def test_goalkeeper_from_detector_votes(self):
        model = TeamModel(hist(4, 5), hist(16, 17), None, .1, 20)
        e = RoleEvidence()
        for _ in range(6):
            e.add('goalkeeper', .8, kit_vote(model, hist(12, 13)), True)
        d = decide_role(e, model)
        self.assertEqual((d.label, d.role, d.team), ('GOALKEEPER', 'goalkeeper', None))


if __name__ == '__main__':
    unittest.main()
