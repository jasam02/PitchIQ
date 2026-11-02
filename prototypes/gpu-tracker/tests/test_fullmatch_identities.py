"""Full Match Mode's match-wide identities (fullmatch/identities.py) and assembly (fullmatch/assemble.py) on synthetic
segment artifacts with known truth (fullmatch_synth.py).

Identity: the same people in two neighbouring segments under different local ids and the other team letters become
one match identity each, and the second segment's letters are swapped back (decided by the overlap); a person whose run
had not yet confirmed an identity on the shared frames is matched through the track's identity; two players crossing in
the overlap keep their own identities; a referee standing on a player is never matched; a goalkeeper keeps the keeper id
and team across the boundary; a player off camera at the boundary who reappears later with the same appearance, where
he could have walked, is linked, while a look-alike teammate somewhere else is not merged; within one segment no two
local identities ever share a match identity; halftime: the segments after a break are mapped by kit colour (no
overlap) and the attacking direction is per period; a long-playing player who disappears early is a possible
substitution, never a confirmed one.
Assembly: the views keep every number (boxes, pitch positions, timeline columns) and carry match names and the frames
they own; the cut sits at the planned boundary when the balls agree there and moves to where they agree when they do not;
the match index lists the periods, segments, boundaries and roster; a match of compressed results (as the workers keep
them) assembles to the same views and index as one of plain results (as they were kept before).

Run:  python -m unittest test_fullmatch_identities          (from this directory)
"""
import copy
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fullmatch import assemble, identities, manifest as mf, plan  # noqa: E402
import fullmatch_synth as synth  # noqa: E402
from fullmatch_synth import Person, walker  # noqa: E402

PAD = 25   # frames of context on each side (1 s)


def two_segments():
    """Frames 0-499 and 500-999 owned, each processing 25 frames of context."""
    return plan.plan(1000, synth.FPS, [(0.0, 40.0)], segment_s=20.0, pad_s=1.0)


class Scene:
    def __init__(self, segments, people, swap=(False, False), late=(0, 0), goal_ends=None, balls=None):
        self.root = Path(tempfile.mkdtemp())
        self.segments = segments
        self.folders, self.local = {}, {}
        for k, s in enumerate(segments):
            result, snapshot, local = synth.make_segment(s, people, swap=swap[k] if k < len(swap) else False,
                                                         late=late[k] if k < len(late) else 0,
                                                         goal_ends=(goal_ends or {}).get(s.period),
                                                         ball=(balls or {}).get(k))
            folder = self.root/'segments'/s.key
            synth.write_artifact(folder, result, snapshot)
            self.folders[s.index], self.local[s.index] = folder, local

    def resolve(self):
        return identities.resolve(self.segments, self.folders, PAD, synth.FPS)

    def match_id(self, out, segment, person):
        local = self.local[segment][person]
        return out['ids'][segment].get(local)


class IdentityTests(unittest.TestCase):
    def test_the_same_people_under_other_ids_and_letters_become_one_identity_each(self):
        people = [Person('a1', 'A', seed=1, path=walker(.2, .3, .0001)), Person('a2', 'A', seed=2, path=walker(.4, .6)),
                  Person('b1', 'B', seed=3, path=walker(.6, .4, -.0001)), Person('k', 'B', 'GOALKEEPER', seed=4, path=walker(.95, .5))]
        scene = Scene(two_segments(), people, swap=(False, True))
        out = scene.resolve()
        self.assertFalse(out['mapping'][0]['swap'])
        self.assertTrue(out['mapping'][1]['swap'])
        self.assertEqual(out['mapping'][1]['method'], 'overlap')
        for name in ('a1', 'a2', 'b1', 'k'):
            self.assertEqual(scene.match_id(out, 0, name), scene.match_id(out, 1, name), name)
        self.assertEqual(len(out['roster']), 4)
        teams = {p['matchId']: p['team'] for p in out['roster']}
        self.assertEqual(teams[scene.match_id(out, 0, 'b1')], 'B')
        self.assertEqual(teams[scene.match_id(out, 1, 'a2')], 'A')
        keeper = scene.match_id(out, 1, 'k')
        self.assertEqual(keeper, 'GK-2')     # team B's keeper
        self.assertEqual(next(p for p in out['roster'] if p['matchId'] == keeper)['display'], 'GK-B')
        self.assertEqual(out['boundaries'][0]['accepted'], 4)

    def test_a_person_not_yet_identified_on_the_shared_frames_is_matched_through_the_track(self):
        people = [Person('a1', 'A', seed=1, path=walker(.2, .3)), Person('b1', 'B', seed=3, path=walker(.6, .4)),
                  Person('b2', 'B', seed=5, path=walker(.7, .7))]
        scene = Scene(two_segments(), people, late=(0, 40))   # segment 1 confirms people after 40 frames (> the 50 shared)
        out = scene.resolve()
        for name in ('a1', 'b1', 'b2'):
            self.assertEqual(scene.match_id(out, 0, name), scene.match_id(out, 1, name), name)

    def test_crossing_players_keep_their_identities_and_a_referee_is_never_matched(self):
        # a1 walks right through b1's path in the overlap (they cross at frame ~500); the referee stands on a2.
        people = [Person('a1', 'A', seed=1, path=walker(.30, .5, .0004)), Person('b1', 'B', seed=3, path=walker(.70, .5, -.0004)),
                  Person('a2', 'A', seed=2, path=walker(.2, .8)), Person('ref', 'A', seed=9, referee=True, path=walker(.2, .8))]
        scene = Scene(two_segments(), people)
        out = scene.resolve()
        self.assertEqual(scene.match_id(out, 0, 'a1'), scene.match_id(out, 1, 'a1'))
        self.assertEqual(scene.match_id(out, 0, 'b1'), scene.match_id(out, 1, 'b1'))
        self.assertNotEqual(scene.match_id(out, 0, 'a1'), scene.match_id(out, 0, 'b1'))
        self.assertEqual(scene.match_id(out, 0, 'a2'), scene.match_id(out, 1, 'a2'))
        self.assertIsNone(scene.local[0]['ref'])   # the referee never had an identity to match

    def test_a_player_off_camera_at_the_boundary_is_linked_but_a_look_alike_elsewhere_is_not(self):
        segments = plan.plan(1500, synth.FPS, [(0.0, 60.0)], segment_s=20.0, pad_s=1.0)   # three segments
        gone = lambda f: not (400 <= f < 600)          # off camera around the first boundary (500)
        twin = Person('twin', 'A', seed=1, path=walker(.8, .2, visible=lambda f: f >= 1100))   # a1's appearance, far away, later
        people = [Person('a1', 'A', seed=1, path=walker(.3, .3, .00002, visible=gone)), Person('b1', 'B', seed=3, path=walker(.6, .5)), twin]
        scene = Scene(segments, people)
        out = scene.resolve()
        self.assertEqual(scene.match_id(out, 0, 'a1'), scene.match_id(out, 1, 'a1'))
        self.assertEqual(out['links'], 1)
        self.assertNotEqual(scene.match_id(out, 2, 'twin'), scene.match_id(out, 2, 'a1'))
        person = next(p for p in out['roster'] if p['matchId'] == scene.match_id(out, 0, 'a1'))
        self.assertEqual(person['how'], 'linked across gaps')
        self.assertEqual(person['confidence'], 'low')

    def test_no_two_local_identities_of_one_segment_share_a_match_identity(self):
        people = [Person(f'p{i}', 'AB'[i % 2], seed=i, path=walker(.1+.08*i, .2+.05*(i % 5))) for i in range(10)]
        scene = Scene(plan.plan(1500, synth.FPS, [(0.0, 60.0)], segment_s=20.0, pad_s=1.0), people, swap=(False, True, False))
        out = scene.resolve()
        for segment, ids in out['ids'].items():
            values = [v for v in ids.values()]
            self.assertEqual(len(values), len(set(values)), segment)
        self.assertEqual(len(out['roster']), 10)

    def test_halftime_maps_letters_by_kit_and_keeps_directions_per_period(self):
        segments = plan.plan(2000, synth.FPS, [(0.0, 20.0), (40.0, 60.0)], segment_s=20.0, pad_s=1.0)
        people = [Person('a1', 'A', seed=1, path=walker(.3, .3)), Person('b1', 'B', seed=3, path=walker(.7, .5))]
        scene = Scene(segments, people, swap=(False, True), goal_ends={0: {'left': 'A', 'right': 'B'}, 1: {'left': 'B', 'right': 'A'}})
        out = scene.resolve()
        self.assertTrue(out['mapping'][1]['swap'])
        self.assertEqual(out['mapping'][1]['method'], 'kit')
        self.assertEqual(out['periods'][0]['defendsLeft'], 'A')
        self.assertEqual(out['periods'][1]['defendsLeft'], 'B')
        self.assertEqual(out['boundaries'], [])     # nothing is stitched across the break

    def test_a_long_player_who_vanishes_early_is_only_a_possible_substitution(self):
        segments = plan.plan(12000, synth.FPS, [(0.0, 480.0)], segment_s=160.0, pad_s=1.0)
        people = [Person('stays', 'A', seed=1, path=walker(.3, .3)),
                  Person('leaves', 'A', seed=2, path=walker(.5, .5, visible=lambda f: f < 5000)),
                  Person('b1', 'B', seed=3, path=walker(.7, .5))]
        scene = Scene(segments, people)
        out = scene.resolve()
        leaves = next(p for p in out['roster'] if p['matchId'] == scene.match_id(out, 0, 'leaves'))
        self.assertEqual(leaves['substitution']['status'], 'possible')
        self.assertIn('Not confirmed', leaves['substitution']['note'])
        stays = next(p for p in out['roster'] if p['matchId'] == scene.match_id(out, 0, 'stays'))
        self.assertIsNone(stays['substitution'])


class AssemblyTests(unittest.TestCase):
    def build(self, balls=None, swap=(False, True), compressed=False):
        segments = two_segments()
        people = [Person('a1', 'A', seed=1, path=walker(.2, .3, .0001)), Person('b1', 'B', seed=3, path=walker(.6, .4))]
        root = Path(tempfile.mkdtemp())/'matches'/'m'
        video = {'name': 'v', 'width': 1920, 'height': 1080, 'fps': synth.FPS, 'frames': 1000, 'duration': 40.0}
        m = mf.Manifest.create(root, {'key': 'k'*64, 'path': 'v.mp4'}, video, [(0.0, 40.0)], segments, {'imageSize': 1280}, {}, 1.0, 20.0)
        originals = []
        for k, s in enumerate(segments):
            result, snapshot, local = synth.make_segment(s, people, swap=swap[k], ball=(balls or {}).get(k))
            originals.append(copy.deepcopy(result))
            synth.write_artifact(m.artifact_folder(s.index), result, snapshot, compressed=compressed)
            m.update_segment(s.index, state='done', artifact='segments/'+s.key)
        return m, segments, originals

    def test_compressed_results_assemble_to_the_same_match(self):
        balls = {0: lambda f: (.5, .5) if f < 498 else (.1, .1), 1: lambda f: (.9, .9) if f < 490 else (.5, .5)}
        plain, _, _ = self.build(balls=balls)
        packed, segments, _ = self.build(balls=balls, compressed=True)
        self.assertTrue(all((packed.artifact_folder(s.index)/mf.RESULT_GZ).is_file() for s in segments))
        self.assertFalse(any((packed.artifact_folder(s.index)/mf.RESULT).exists() for s in segments))
        a, b = assemble.assemble(plain), assemble.assemble(packed)
        for s in a['segments']:
            self.assertEqual(assemble.load_view(plain.folder/'assembly', s['view']), assemble.load_view(packed.folder/'assembly', s['view']))
        volatile = ('matchId', 'assembledAt', 'assemblySeconds', 'assemblyTimings')
        self.assertEqual({k: v for k, v in a.items() if k not in volatile}, {k: v for k, v in b.items() if k not in volatile})
        self.assertEqual(b['boundaries'][0]['cutFrame'], 497)

    def test_views_keep_every_number_and_carry_match_names(self):
        m, segments, originals = self.build(balls={0: lambda f: (.5, .5), 1: lambda f: (.5, .5)})
        index = assemble.assemble(m)
        self.assertEqual([s['index'] for s in index['segments']], [0, 1])
        self.assertEqual(len(index['roster']), 2)
        view = assemble.load_view(m.folder/'assembly', index['segments'][1]['view'])
        original = originals[1]
        self.assertEqual(view['matchView']['cut0'], 500)
        self.assertTrue(view['matchView']['teamsSwapped'])
        for fv, fo in zip(view['frames'], original['frames']):
            self.assertEqual([p['box'] for p in fv['people']], [p['box'] for p in fo['people']])
            self.assertEqual([p['pitch'] for p in fv['people']], [p['pitch'] for p in fo['people']])
        a_id = next(p['matchId'] for p in index['roster'] if p['team'] == 'A')
        person = next(p for p in view['frames'][100]['people'] if p['id'] == a_id)
        self.assertEqual(person['team'], 'A')                       # the run's 'B' is the match's A
        self.assertEqual(person['label'], 'PLAYER_TEAM_A')
        self.assertIn(a_id, view['timeline']['ids'])
        self.assertIn(a_id, view['stabilized']['ids'])
        self.assertEqual(view['timeline']['ids'][a_id]['team'], 'A')
        self.assertEqual(view['teams']['a'], original['teams']['b'])
        self.assertTrue(all(k in {p['matchId'] for p in index['roster']} for k in view['frames'][100]['visibility']))
        first = assemble.load_view(m.folder/'assembly', index['segments'][0]['view'])
        a0 = next(p for p in first['frames'][10]['people'] if p['team'] == 'A')
        self.assertEqual(a0['id'], a_id)
        self.assertEqual(index['boundaries'][0]['cutFrame'], 500)
        self.assertEqual(m.data['assembly']['state'], 'complete')

    def test_the_match_state_is_made_at_the_end(self):
        # step 7: the people, places and state of fullmatch/intel.py, so that the page opens with them
        m, segments, _ = self.build(balls={0: lambda f: (.5, .5), 1: lambda f: (.5, .5)})
        assemble.assemble(m)
        self.assertEqual(m.data['assembly'].get('intel'), 'ready')
        self.assertTrue(list((m.folder/'assembly'/'intel').glob('state-*.json')))

    def test_the_cut_moves_to_where_the_balls_agree(self):
        # the true ball is at (.5, .5): segment 0 follows it until frame 497 and then a white thing at (.1, .1); segment 1
        # follows it from frame 490 on. At the planned boundary (500) they disagree; 490-497 they agree.
        balls = {0: lambda f: (.5, .5) if f < 498 else (.1, .1), 1: lambda f: (.9, .9) if f < 490 else (.5, .5)}
        m, segments, _ = self.build(balls=balls)
        index = assemble.assemble(m)
        b = index['boundaries'][0]
        self.assertEqual(b['plannedFrame'], 500)
        self.assertEqual(b['cutFrame'], 497)
        self.assertIn('moved', b['how'])
        self.assertEqual(index['segments'][0]['cut1'], b['cutFrame'])
        self.assertEqual(index['segments'][1]['cut0'], b['cutFrame'])

    def test_a_disagreeing_boundary_is_flagged(self):
        balls = {0: lambda f: (.1, .1), 1: lambda f: (.9, .9)}
        m, segments, _ = self.build(balls=balls)
        index = assemble.assemble(m)
        self.assertTrue(index['boundaries'][0]['ball']['flagged'])
        self.assertEqual(index['boundaries'][0]['cutFrame'], 500)


if __name__ == '__main__':
    unittest.main()
