import contextlib
import unittest

import numpy as np

import scene  # noqa: F401  (adds the prototype directory to sys.path)
from soccer.appearance import Descriptor
from soccer.identity import IdentityManager, Options, Stab
from soccer.teams import TeamModel, hellinger, kit_vote

RNG = np.random.default_rng(7)


def unit(v):
    return v/np.linalg.norm(v)


TEAM_BASE = {k: unit(RNG.normal(size=512)) for k in ('A', 'B', 'REF', 'GK')}
JERSEY = {k: np.zeros(24) for k in ('A', 'B', 'REF', 'GK')}
JERSEY['A'][4], JERSEY['A'][5] = .7, .3      # red
JERSEY['B'][4+2*6], JERSEY['B'][5+2*6] = .6, .4  # blue
JERSEY['REF'][4+2*2], JERSEY['REF'][5+2*2] = .5, .5  # yellow
JERSEY['GK'][4+2*4], JERSEY['GK'][5+2*4] = .5, .5  # green
SHORTS = {'A': 2, 'B': 2, 'REF': 0, 'GK': 1}  # shorts colour bin: white, white, black, grey


def onehot(i, n=12):
    v = np.zeros(n)
    v[i] = 1
    return v


MODEL = TeamModel(JERSEY['A'], JERSEY['B'], JERSEY['REF'], .1, 20, shorts_a=onehot(SHORTS['A']), shorts_b=onehot(SHORTS['B']),
                  referee_shorts=onehot(SHORTS['REF']))


class Person:
    """kit: jersey colours; shorts: shorts colour (default: the kit's); look: the OSNet appearance
    (default: the kit's); cls: what the detector calls this person."""

    def __init__(self, kit, unique=None, cls=None, shorts=None, look=None):
        self.kit, self.shorts, self.look = kit, shorts or kit, look or kit
        self.unique = unit(RNG.normal(size=512)) if unique is None else unique
        self.cls = cls or {'REF': 'referee', 'GK': 'goalkeeper'}.get(kit, 'player')
        self.cycle = None   # optional sequence of detector classes, one per observation

    def detector_class(self):
        if self.cycle:
            self.cls = self.cycle[0]
            self.cycle = self.cycle[1:]+self.cycle[:1]
        return self.cls

    def embedding(self):
        return unit(.9*TEAM_BASE[self.look]+.45*self.unique+.12*unit(RNG.normal(size=512)))

    def descriptor(self, quality=.8):
        shorts = onehot(SHORTS[self.shorts])
        return Descriptor(JERSEY[self.kit].copy(), shorts, shorts[:8].copy(), np.full(24, .5), quality, self.embedding())


@contextlib.contextmanager
def pink_kit(shorts='B'):
    """A 'PINK' kit: a jersey inside team B's colour spread but near its edge (a pink referee next to a
    garnet kit), the given shorts, and the referees' learned appearance."""
    limit = min(.5, 2.5*MODEL.team_spread('B')+.1)
    for w in np.linspace(.05, .95, 91):
        jersey = (1-w)*JERSEY['B']+w*JERSEY['REF']
        if .85*limit <= hellinger(jersey, MODEL.b) < .97*limit and hellinger(jersey, MODEL.a) > limit:
            break
    else:
        raise AssertionError('no blend lands near the edge of the team spread')
    JERSEY['PINK'], SHORTS['PINK'], TEAM_BASE['PINK'] = jersey, SHORTS[shorts], TEAM_BASE['REF']
    try:
        yield
    finally:
        for table in (JERSEY, SHORTS, TEAM_BASE):
            table.pop('PINK', None)


def same_uniform(other):
    """An OSNet appearance close to another person's: the officials wear one uniform."""
    return unit(other.unique+.3*unit(RNG.normal(size=512)))


def box_at(x, y, h=.12):
    w = h*.38*9/16
    return [x-w/2, y-h, w, h]


class Driver:
    def __init__(self, options=None, saved=None):
        self.ids = IdentityManager(options or Options(), saved)
        self.model = MODEL
        self.t = 0.0
        self.events = []
        self.people = []

    def step(self, visible, ended=(), zone='inside', segment=0, dt=.2, cues=None, ball=None):
        """visible: {local id: (Person, x, y)}; zone and cues: a value or a function of the local id; ball: its
        camera-stabilized (x, y) when known."""
        samples = []
        boxes = {tid: box_at(x, y) for tid, (_, x, y) in visible.items()}
        for tid, (p, x, y) in visible.items():
            d = p.descriptor()
            b = boxes[tid]
            occluded = any(abs(b[0]-o[0]) < b[2] and abs(b[1]-o[1]) < b[3] for k, o in boxes.items() if k != tid)
            z = zone(tid) if callable(zone) else zone
            c = cues(tid) if callable(cues) else cues
            samples.append({'id': tid, 'box': b, 'score': .9, 'cls': p.detector_class(), 'zone': z, 'occluded': occluded,
                            'stab': Stab(x, y, .12), 'descriptor': d, 'vote': kit_vote(self.model, d.jersey, d.shorts), 'cues': c or {}})
        out = self.ids.update({'time': self.t, 'tick': True, 'samples': samples, 'ended': [{'id': i} for i in ended],
                               'model': self.model, 'pitch_reliable': True, 'filter_enabled': True, 'aspect': 16/9,
                               'segment': segment, 'drift': 0, 'ball': {'stab': ball, 'pitch': None} if ball else None})
        self.events += out['events']
        self.people.append(out['people'])
        self.t += dt
        return {p['track']: p for p in out['people']}

    def run(self, seconds, visible, **kw):
        out = None
        for _ in range(round(seconds/.2)):
            out = self.step(visible, **kw)
        return out

    def kinds(self, kind):
        return [e for e in self.events if e['kind'] == kind]


def roster(team, n, x0=.2):
    return [(Person(team), x0+i*.06, .5+.03*(i % 3)) for i in range(n)]


class IdentityTests(unittest.TestCase):
    def test_candidate_needs_several_observations(self):
        d = Driver()
        p = Person('A')
        out = d.step({1: (p, .5, .6)})
        self.assertIsNone(out[1]['id'])
        self.assertEqual(out[1]['label'], 'CANDIDATE')
        out = d.run(1.2, {1: (p, .5, .6)})
        self.assertEqual(out[1]['id'], 'A-01')
        self.assertEqual((out[1]['team'], out[1]['role'], out[1]['label']), ('A', 'PLAYER', 'PLAYER_TEAM_A'))
        # The observations collected before promotion are labelled retroactively.
        first = d.people[0][0]
        self.assertEqual(first['id'], 'A-01')
        self.assertEqual(first['evidence'], 'reidentified')

    def test_one_frame_detection_never_becomes_a_player(self):
        d = Driver()
        d.step({1: (Person('A'), .5, .6)})
        d.step({}, ended=[1])
        d.run(2, {})
        self.assertEqual(d.ids.registry, {})

    def test_returning_player_keeps_global_identity_on_new_local_track(self):
        d = Driver()
        team = roster('A', 4)
        visible = {i+1: t for i, t in enumerate(team)}
        d.run(2, visible)
        ids = {tid: d.ids.tracks[tid].player_id for tid in visible}
        self.assertEqual(sorted(ids.values()), ['A-01', 'A-02', 'A-03', 'A-04'])
        leaving = team[3]
        stay = {k: v for k, v in visible.items() if k != 4}
        d.step(stay, ended=[4])
        d.run(10, stay)  # 10 s off screen
        back = dict(stay)
        back[91] = (leaving[0], leaving[1]+.01, leaving[2])
        out = d.run(3, back)
        self.assertEqual(out[91]['id'], ids[4])
        self.assertEqual(len(d.ids.registry), 4, 'no duplicate identity')
        reid = [e for e in d.kinds('reid') if e['track'] == 91]
        self.assertEqual(len(reid), 1)
        self.assertIn('RE-ID EVENT', reid[0]['message'])
        self.assertIn(f"Matched Global Player: TeamA_Player_{ids[4][2:]}", reid[0]['message'])
        self.assertGreaterEqual(reid[0]['scores']['final'], .72)

    def test_ambiguous_return_is_deferred_not_guessed(self):
        d = Driver()
        twin = unit(RNG.normal(size=512))
        a, b = Person('A', twin), Person('A', unit(twin+.05*RNG.normal(size=512)))
        others = roster('A', 9, x0=.1)
        visible = {i+10: t for i, t in enumerate(others)}
        visible[1], visible[2] = (a, .4, .5), (b, .45, .5)
        d.run(2, visible)
        pa, pb = d.ids.tracks[1].player_id, d.ids.tracks[2].player_id
        self.assertEqual(len(d.ids.registry), 11)
        rest = {k: v for k, v in visible.items() if k not in (1, 2)}
        d.step(rest, ended=[1, 2])
        d.run(20, rest, segment=1)  # camera cut: no spatial evidence
        back = dict(rest)
        back[50] = (a, .7, .5)
        out = d.run(3, back, segment=1)
        self.assertIsNone(out[50]['id'], 'two look-alike missing players: identity must stay unresolved')
        self.assertEqual(out[50]['label'], 'IDENTITY_UNCERTAIN')
        self.assertTrue(any(e['track'] == 50 for e in d.kinds('deferred')))
        self.assertNotIn(pa, [p['id'] for p in out.values() if p['track'] == 50])
        self.assertEqual(len(d.ids.registry), 11, 'no new identity while missing players could be this person')
        del pb

    def test_referee_is_separate_from_players(self):
        d = Driver()
        visible = {1: (Person('REF'), .5, .5), 2: (Person('A'), .3, .6), 3: (Person('B'), .7, .6)}
        out = d.run(2, visible)
        self.assertEqual(out[1]['id'], 'REF-1')
        self.assertEqual((out[1]['role'], out[1]['team'], out[1]['display']), ('REFEREE', None, 'REF-1'))
        groups = d.ids.summary()
        self.assertEqual([g['id'] for g in groups['referees']], ['REF-1'])
        self.assertEqual(sorted(g['id'] for g in groups['players']), ['A-01', 'B-01'])

    def test_people_outside_the_pitch_are_never_promoted(self):
        d = Driver()
        out = d.run(5, {1: (Person('A'), .5, .95)}, zone='outside')
        self.assertIsNone(out[1]['id'])
        self.assertIn(out[1]['state'], ('candidate', 'rejected'))
        self.assertEqual(d.ids.registry, {})

    def test_new_person_on_the_touchline_needs_stronger_evidence(self):
        d = Driver()
        out = d.run(1.2, {1: (Person('A'), .5, .9)}, zone='boundary')
        self.assertIsNone(out[1]['id'])
        self.assertIn('born on the touchline', d.ids.tracks[1].reason)
        out = d.run(2.4, {1: (Person('A'), .5, .7)})
        self.assertEqual(out[1]['id'], 'A-01')

    def test_team_cap_is_a_sanity_check(self):
        d = Driver(Options(max_per_team=3))
        team = roster('A', 4)
        out = d.run(3, {i+1: t for i, t in enumerate(team)})
        self.assertEqual(len([p for p in out.values() if p['id']]), 3)
        self.assertTrue(any('already has 3 identities' in e['message'] for e in d.kinds('sanity')))

    def test_crossing_swap_is_corrected_by_appearance(self):
        d = Driver()
        a, b = Person('A'), Person('A')
        others = roster('A', 3, x0=.1)
        visible = {10+i: t for i, t in enumerate(others)}
        visible.update({1: (a, .40, .5), 2: (b, .60, .5)})
        d.run(2, visible)
        ida, idb = d.ids.tracks[1].player_id, d.ids.tracks[2].player_id
        # They meet, overlap, and the local tracker swaps them while they separate.
        for x in (.45, .49, .5, .5):
            d.step({**visible, 1: (a, x, .5), 2: (b, 1-x, .5)})
        for x in (.45, .4, .35, .3):
            d.step({**visible, 1: (b, x, .5), 2: (a, 1-x, .5)})
        self.assertTrue(d.kinds('swap-corrected'), [e['message'] for e in d.events[-6:]])
        self.assertEqual(d.ids.tracks[1].player_id, idb)
        self.assertEqual(d.ids.tracks[2].player_id, ida)

    def test_identities_continue_into_the_next_run(self):
        d = Driver()
        team = roster('B', 3)
        d.run(2, {i+1: t for i, t in enumerate(team)})
        saved = d.ids.snapshot()
        nxt = Driver(saved=saved)
        nxt.t = 40.0
        for g in nxt.ids.registry.values():
            self.assertIn(g.status, ('missing', 'offscreen'))
        out = nxt.run(3, {7: (team[1][0], .8, .6)})
        self.assertEqual(out[7]['id'], 'B-02')
        self.assertEqual(len(nxt.ids.registry), 3)

    def test_goalkeeper_team_from_the_defenders_next_to_them(self):
        d = Driver()
        keeper = Person('GK')
        visible = {1: (keeper, .05, .55)}
        for i, (team, x) in enumerate((('A', .15), ('A', .2), ('B', .3), ('B', .45), ('A', .55), ('B', .7))):
            visible[10+i] = (Person(team), x, .5+.04*i)
        out = d.run(6, visible)
        self.assertEqual(out[1]['id'], 'GK-1')
        self.assertEqual((out[1]['role'], out[1]['team'], out[1]['display'], out[1]['label']), ('GOALKEEPER', 'A', 'GK-A', 'GOALKEEPER_TEAM_A'))
        self.assertEqual(len(d.ids.summary()['goalkeepers']), 1)
        self.assertTrue(any('GOALKEEPER TEAM' in e['message'] and 'Team: A' in e['message'] for e in d.kinds('role')))

    def test_player_identity_that_turns_out_to_be_a_referee_is_converted(self):
        # A referee whose jersey looks like team B's. Early on the kit model has no shorts prototypes
        # and the detector calls them a player: they become a team B player. Then the black shorts and
        # the detector's referee votes reveal the referee.
        d = Driver()
        d.model = MODEL.replace(shorts_a=None, shorts_b=None, referee_shorts=None)
        ref = Person('B', shorts='REF', look='REF', cls='player')
        visible = {1: (ref, .5, .5), 2: (Person('A'), .3, .6), 3: (Person('B'), .7, .6)}
        out = d.run(3, visible)
        first = out[1]['id']
        self.assertTrue(first and first.startswith('B-'), out[1])
        d.model, ref.cls = MODEL, 'referee'
        out = d.run(12, visible)
        self.assertEqual((out[1]['id'], out[1]['role'], out[1]['team'], out[1]['display']), ('REF-1', 'REFEREE', None, 'REF-1'))
        update = [e for e in d.kinds('role') if 'New role: REFEREE' in e['message']]
        self.assertEqual(len(update), 1, [e['message'] for e in d.kinds('role')])
        for line in ('ROLE UPDATE', f'Global ID: {first} -> REF-1', 'Old role: PLAYER (team B)', 'Team A similarity', 'Team B similarity', 'Referee confidence'):
            self.assertIn(line, update[0]['message'])
        old = d.ids.registry[first]
        self.assertEqual((old.status, old.retired_into), ('retired', 'REF-1'))
        # Every observation of this person, also those labelled before the conversion, is the referee.
        mine = [p for frame in d.people for p in frame if p['track'] == 1 and p['id']]
        self.assertTrue(mine)
        self.assertEqual({(p['id'], p['role'], p['team']) for p in mine}, {('REF-1', 'REFEREE', None)})
        groups = d.ids.summary()
        self.assertEqual([g['id'] for g in groups['referees']], ['REF-1'])
        self.assertEqual([(g['id'], g['retiredInto']) for g in groups['retired']], [(first, 'REF-1')])
        self.assertNotIn(first, [g['id'] for g in groups['players']])
        self.assertTrue(groups['referees'][0]['roleLocked'])

    def test_returning_referee_is_reconnected_not_made_a_player(self):
        # shorts: black shorts reveal the referee; look-only: same kit as team B, only their own
        # appearance (OSNet) tells them apart from the team B players.
        for variant, shorts in (('shorts', 'REF'), ('look-only', 'B')):
            with self.subTest(variant):
                d = Driver()
                ref = Person('B', shorts=shorts, look='REF', cls='referee')
                team = {10+i: t for i, t in enumerate(roster('A', 3, x0=.1)+roster('B', 3, x0=.6))}
                out = d.run(4, {1: (ref, .45, .4), **team})
                self.assertEqual(out[1]['id'], 'REF-1')
                d.step(team, ended=[1])
                d.run(6, team)
                ref.cls = 'player'  # back on a new local track, and the detector now calls them a player
                out = d.run(5, {**team, 40: (ref, .5, .45)})
                self.assertEqual(out[40]['id'], 'REF-1')
                ids = {p['id'] for frame in d.people for p in frame if p['track'] == 40 and p['id']}
                self.assertEqual(ids, {'REF-1'})
                self.assertEqual(len(d.ids.summary()['referees']), 1)
                self.assertEqual(sorted(g['id'] for g in d.ids.summary()['players']), ['A-01', 'A-02', 'A-03', 'B-01', 'B-02', 'B-03'])

    def test_track_switch_onto_the_referee_never_relabels_the_player(self):
        d = Driver()
        player, ref = Person('A'), Person('REF')
        team = {10+i: t for i, t in enumerate(roster('A', 2, x0=.1)+roster('B', 3, x0=.6))}
        d.run(3, {1: (player, .4, .5), **team})
        pid = d.ids.tracks[1].player_id
        self.assertTrue(pid and pid.startswith('A-'))
        switched = len(d.people)
        d.run(10, {1: (ref, .42, .5), **team})  # the local tracker now follows the referee on track 1
        self.assertEqual(d.ids.tracks[1].player_id, 'REF-1')
        self.assertNotEqual(d.ids.registry[pid].status, 'retired')
        before = {p['id'] for frame in d.people[:switched] for p in frame if p['track'] == 1 and p['id']}
        self.assertEqual(before, {pid}, 'the player observations before the switch keep the player identity')
        out = d.run(3, {**team, 1: (ref, .42, .5), 30: (player, .3, .55)})  # the player is back on a new track
        self.assertEqual(out[30]['id'], pid)

    def test_assistant_referee_on_the_touchline_is_an_official(self):
        d = Driver()
        visible = {1: (Person('REF', cls='player'), .5, .93)}  # the detector calls them a player
        visible.update({10+i: t for i, t in enumerate(roster('A', 3)+roster('B', 3, x0=.55))})
        out = d.run(5, visible, zone=lambda tid: 'boundary' if tid == 1 else 'inside')
        self.assertEqual((out[1]['id'], out[1]['role']), ('REF-1', 'REFEREE'))
        found = [e for e in d.kinds('role') if e['message'].startswith('REFEREE IDENTIFIED')]
        self.assertEqual(len(found), 1)
        for line in ('Global ID: REF-1', 'Team A similarity', 'Team B similarity', 'Referee confidence'):
            self.assertIn(line, found[0]['message'])
        self.assertEqual({p['id'] for frame in d.people for p in frame if p['track'] == 1 and p['id']}, {'REF-1'})

    def test_odd_kit_alone_never_makes_a_referee(self):
        # A kit matching neither team (a team kit in shadow does that too) with the detector never saying
        # referee: nobody becomes REF-n on colour alone.
        d = Driver()
        odd = Person('GK', cls='player')
        visible = {1: (odd, .5, .5), 2: (Person('A'), .3, .6), 3: (Person('B'), .7, .6)}
        out = d.run(6, visible)
        self.assertIsNone(out[1]['id'], out[1])
        self.assertEqual(d.ids.summary()['referees'], [])
        self.assertEqual(sorted(g['id'] for g in d.ids.summary()['players']), ['A-01', 'B-01'])

    def test_loosely_matching_kit_with_detector_referee_votes_is_a_referee(self):
        # A referee kit close to team B's colours (inside B's spread, but near its edge) and a detector
        # that calls them a referee most of the time: the detector is not held back by the loose match.
        with pink_kit():
            d = Driver()
            ref = Person('PINK', cls='referee')
            ref.cycle = ['referee', 'referee', 'player', 'referee', 'referee', 'referee', 'player', 'referee', 'referee', 'referee']
            visible = {1: (ref, .5, .5), 2: (Person('A'), .3, .6), 3: (Person('B'), .7, .6)}
            out = d.run(6, visible)
            self.assertEqual((out[1]['id'], out[1]['role']), ('REF-1', 'REFEREE'), out[1])
            self.assertGreaterEqual(len(d.ids.referee_samples(MODEL)), 3, "a loosely matching kit the detector calls referee seeds the referee kit")
            # The same kit with the detector mostly saying player stays a team B player.
            d = Driver()
            ref = Person('PINK', cls='player')
            ref.cycle = ['player', 'player', 'referee', 'player', 'player']
            out = d.run(6, {1: (ref, .5, .5), 2: (Person('A'), .3, .6), 3: (Person('B'), .7, .6)})
            self.assertEqual((out[1]['team'], out[1]['role']), ('B', 'PLAYER'), out[1])
            self.assertEqual(d.ids.referee_samples(MODEL), [])

    def test_referee_in_a_team_like_kit_is_corrected_by_the_known_referees_appearance(self):
        # The centre referee in a kit near team B's colours, the detector calling them a player: with no
        # referee known they become a team B player. Once the other official (REF-1) is confirmed, their
        # whole appearance says "like the known referees, not like team B's players" (the officials wear one
        # uniform), the player identity is converted into REF-2 and retired, and its observations follow.
        with pink_kit():
            d = Driver()
            ref1 = Person('REF')
            centre = Person('PINK', cls='player', unique=same_uniform(ref1))
            team = {10+i: t for i, t in enumerate(roster('A', 3, x0=.1)+roster('B', 3, x0=.6))}
            out = d.run(3, {1: (centre, .5, .45), **team})
            pid = out[1]['id']
            self.assertTrue(pid and pid.startswith('B-'), out[1])
            view = {1: (centre, .5, .45), 2: (ref1, .35, .4), **team}
            out = d.run(8, view, ball=(.56, .55))  # the ball a couple of metres from the referee, never at their feet
            self.assertEqual((out[1]['id'], out[1]['role'], out[1]['team']), ('REF-2', 'REFEREE', None), out[1])
            self.assertEqual(out[2]['id'], 'REF-1')
            self.assertEqual((d.ids.registry[pid].status, d.ids.registry[pid].retired_into), ('retired', 'REF-2'))
            self.assertEqual({p['id'] for frame in d.people for p in frame if p['track'] == 1 and p['id']}, {'REF-2'})
            update = [e for e in d.kinds('role') if 'New role: REFEREE' in e['message'] and f'Global ID: {pid} -> REF-2' in e['message']]
            self.assertEqual(len(update), 1, [e['message'] for e in d.kinds('role')])
            for line in ('Appearance vs known referees / team A / team B:', 'Ball-following behaviour:', 'Team formation consistency:', 'Final: '):
                self.assertIn(line, update[0]['message'])
            players = d.ids.summary()['players']
            self.assertEqual(sum(1 for g in players if g['team'] == 'B'), 3)
            self.assertNotIn(pid, [g['id'] for g in players])
            referee = next(g for g in d.ids.summary()['referees'] if g['id'] == 'REF-2')
            self.assertIsNotNone(referee['roleEvidence'].get('refereeAppearance'))
            self.assertIn('looks ref', out[1]['why'])
            # Settled: the detector calling them a player for a long while never makes them B-04 again.
            out = d.run(12, {**view, 1: (centre, .55, .5)})
            self.assertEqual((out[1]['id'], out[1]['role']), ('REF-2', 'REFEREE'))
            self.assertEqual(len(d.ids.summary()['referees']), 2)

    def test_touchline_official_in_a_team_like_kit_is_an_assistant_referee(self):
        # The officials at the near touchline: a kit near team B's colours, the detector calling them a
        # player, on or just outside the line and moving along it. With REF-1 known, the whole appearance
        # and the touchline behaviour make them an ASSISTANT REFEREE, never a team B player.
        with pink_kit():
            d = Driver()
            ref1 = Person('REF')
            official = Person('PINK', cls='player', unique=same_uniform(ref1))
            team = {10+i: t for i, t in enumerate(roster('A', 3, x0=.1)+roster('B', 3, x0=.6))}
            d.run(2, {2: (ref1, .35, .4), **team})
            self.assertEqual(d.ids.tracks[2].player_id, 'REF-1')
            edge = {'edge': {'side': 'near', 'metres': 1.0, 'dir': (1.0, 0.0)}}
            x = .3
            for _ in range(25):  # 5 s walking along the near touchline, a metre outside it
                x += .006
                out = d.step({1: (official, x, .93), 2: (ref1, .35, .4), **team}, zone=lambda tid: 'boundary' if tid == 1 else 'inside',
                             cues=lambda tid: edge if tid == 1 else {})
            self.assertEqual((out[1]['id'], out[1]['role'], out[1]['team']), ('REF-2', 'REFEREE', None), out[1])
            self.assertEqual(d.ids.registry['REF-2'].official, 'assistant')
            self.assertEqual(next(g for g in d.ids.summary()['referees'] if g['id'] == 'REF-2')['official'], 'ASSISTANT_REFEREE')
            found = [e for e in d.kinds('role') if e['message'].startswith('REFEREE IDENTIFIED') and 'Global ID: REF-2' in e['message']]
            self.assertEqual(len(found), 1, [e['message'] for e in d.kinds('role')])
            for line in ('Appearance vs known referees / team A / team B:', 'Time on or outside the touchline: 100%', 'movement along it: 1.00',
                         'Final: ASSISTANT REFEREE'):
                self.assertIn(line, found[0]['message'])
            self.assertIn('ASSISTANT REFEREE', out[1]['why'])
            self.assertEqual(sum(1 for g in d.ids.summary()['players'] if g['team'] == 'B'), 3)

    def test_staff_pacing_the_touchline_without_referee_evidence_stay_unidentified(self):
        # A coach in clothes matching neither team walks along the touchline. Touchline behaviour alone is
        # not a referee: with no referee look, no referee kit and no detector votes, nobody is promoted.
        d = Driver()
        coach = Person('GK', cls='player')
        visible = {2: (Person('REF'), .35, .4), **{10+i: t for i, t in enumerate(roster('A', 3, x0=.1)+roster('B', 3, x0=.6))}}
        d.run(2, visible)
        edge = {'edge': {'side': 'near', 'metres': 2.0, 'dir': (1.0, 0.0)}}
        x = .3
        for _ in range(30):
            x += .005
            out = d.step({1: (coach, x, .93), **visible}, zone=lambda tid: 'boundary' if tid == 1 else 'inside', cues=lambda tid: edge if tid == 1 else {})
        self.assertIsNone(out[1]['id'], out[1])
        self.assertEqual(out[1]['role'], 'UNKNOWN')
        self.assertEqual(len(d.ids.summary()['referees']), 1)

    def test_full_team_doubts_another_team_kit_player_the_detector_calls_a_referee(self):
        # Team B already has its full count of identities. Another person in team B's kit whom the detector
        # calls a referee more often than not is then a referee, not one more B player; with room in the
        # team the same evidence, held back by the team kit, leaves them a team B player.
        for cap, expected in ((3, ('REF', 'REFEREE', None)), (11, ('B-', 'PLAYER', 'B'))):
            with self.subTest(cap=cap):
                d = Driver(Options(max_per_team=cap))
                team = {10+i: t for i, t in enumerate(roster('A', 3, x0=.1)+roster('B', 3, x0=.6))}
                d.run(2, team)
                extra = Person('B', cls='player')
                extra.cycle = ['referee', 'referee', 'player', 'referee', 'player']
                out = d.run(4, {1: (extra, .5, .45), **team})
                self.assertTrue(out[1]['id'] and out[1]['id'].startswith(expected[0]), out[1])
                self.assertEqual((out[1]['role'], out[1]['team']), expected[1:], out[1])
                if cap == 3:
                    self.assertIn('Team already has its full count of identities', [e for e in d.kinds('role') if 'REFEREE IDENTIFIED' in e['message']][0]['message'])

    def test_locked_referee_is_never_turned_into_a_player(self):
        d = Driver()
        ref = Person('REF')
        visible = {1: (ref, .5, .5), 2: (Person('B'), .7, .6), 3: (Person('A'), .3, .6)}
        d.run(3, visible)
        self.assertTrue(d.ids.registry['REF-1'].role_locked)
        ref.cls = 'player'  # the detector changes its mind for a long while
        out = d.run(12, visible)
        self.assertEqual((out[1]['id'], out[1]['role'], out[1]['display']), ('REF-1', 'REFEREE', 'REF-1'))
        self.assertEqual(len(d.ids.registry), 3)

    @staticmethod
    def keeper_scene(keeper, x=.04):
        visible = {1: (keeper, x, .55)}
        for i, (team, px) in enumerate((('A', .15), ('A', .2), ('B', .3), ('B', .45), ('A', .55), ('B', .7))):
            visible[10+i] = (Person(team), px, .5+.04*i)
        return visible

    def test_goalkeeper_found_from_position_and_kit_without_the_detector(self):
        d = Driver()
        keeper = Person('GK', cls='player')  # the detector never says goalkeeper
        at_goal = {'near_goal': True, 'goal_side': 'left', 'extreme': True, 'isolated': True}
        out = d.run(6, self.keeper_scene(keeper), cues=lambda tid: at_goal if tid == 1 else {})
        self.assertEqual(out[1]['id'], 'GK-1')
        self.assertEqual((out[1]['role'], out[1]['team'], out[1]['display'], out[1]['label']), ('GOALKEEPER', 'A', 'GK-A', 'GOALKEEPER_TEAM_A'))
        self.assertTrue(d.ids.registry['GK-1'].role_locked)
        self.assertGreaterEqual(d.ids.registry['GK-1'].goalkeeper_confidence, .7)
        found = [e for e in d.kinds('role') if e['message'].startswith('GOALKEEPER IDENTIFIED')]
        self.assertEqual(len(found), 1)
        for line in ('Global ID: GK-1', 'Goal proximity score: 1.00', 'Penalty-area residence: 100%', 'Uniform difference score: 1.00', 'Temporal confidence'):
            self.assertIn(line, found[0]['message'])

    def test_goalkeeper_without_goal_hints_is_not_a_referee(self):
        # No goal line or box in view: a kit matching nobody, always the deepest person, alone.
        d = Driver()
        keeper = Person('GK', cls='player')
        out = d.run(6, self.keeper_scene(keeper), cues=lambda tid: {'extreme': True, 'isolated': True} if tid == 1 else {})
        self.assertEqual((out[1]['id'], out[1]['role']), ('GK-1', 'GOALKEEPER'))
        self.assertEqual(d.ids.summary()['referees'], [])

    def test_unconfirmed_referee_that_is_the_goalkeeper_is_converted(self):
        d = Driver()
        keeper = Person('GK', cls='player')
        keeper.cycle = ['referee', 'player', 'player', 'referee', 'player']  # the detector sometimes takes the odd kit for a referee
        out = d.run(3, self.keeper_scene(keeper, x=.38))  # away from the goal: an unconfirmed referee
        self.assertEqual(out[1]['id'], 'REF-1')
        keeper.cycle, keeper.cls = None, 'player'
        self.assertFalse(d.ids.registry['REF-1'].role_locked)
        at_goal = {'near_goal': True, 'goal_side': 'left', 'extreme': True, 'isolated': True}
        out = d.run(12, self.keeper_scene(keeper), cues=lambda tid: at_goal if tid == 1 else {})
        self.assertEqual((out[1]['id'], out[1]['display'], out[1]['role']), ('GK-1', 'GK-A', 'GOALKEEPER'))
        found = [e for e in d.kinds('role') if e['message'].startswith('GOALKEEPER IDENTIFIED')]
        self.assertEqual(len(found), 1)
        self.assertIn('Global ID: REF-1 -> GK-1 (REF-1 retired', found[0]['message'])
        self.assertEqual((d.ids.registry['REF-1'].status, d.ids.registry['REF-1'].retired_into), ('retired', 'GK-1'))
        self.assertEqual({p['display'] for frame in d.people for p in frame if p['track'] == 1 and p['id']}, {'GK-A'})
        self.assertEqual(d.ids.summary()['referees'], [])

    def test_player_identity_becomes_goalkeeper_and_keeps_the_role(self):
        d = Driver()
        keeper = Person('A', cls='player')  # a keeper in a kit close to their team's
        out = d.run(3, self.keeper_scene(keeper, x=.35))
        pid = out[1]['id']
        self.assertTrue(pid and pid.startswith('A-'), out[1])
        keeper.cls = 'goalkeeper'
        at_goal = {'near_goal': True, 'goal_side': 'left', 'extreme': True, 'isolated': True}
        out = d.run(12, self.keeper_scene(keeper), cues=lambda tid: at_goal if tid == 1 else {})
        self.assertEqual((out[1]['id'], out[1]['display'], out[1]['role'], out[1]['label']), (pid, 'GK-A', 'GOALKEEPER', 'GOALKEEPER_TEAM_A'))
        found = [e for e in d.kinds('role') if e['message'].startswith('GOALKEEPER IDENTIFIED')]
        self.assertEqual(len(found), 1)
        for line in ('Goal proximity score', 'Penalty-area residence', 'Uniform difference score', 'Temporal confidence'):
            self.assertIn(line, found[0]['message'])
        # Earlier observations follow the role; leaving the goal area does not remove it.
        self.assertEqual(d.people[0][0]['display'], 'GK-A')
        keeper.cls = 'player'
        out = d.run(8, self.keeper_scene(keeper, x=.3))
        self.assertEqual((out[1]['id'], out[1]['display'], out[1]['role']), (pid, 'GK-A', 'GOALKEEPER'))
        self.assertTrue(d.ids.registry[pid].role_locked)

    def test_identity_is_never_deleted(self):
        d = Driver()
        p = Person('B')
        d.run(2, {1: (p, .5, .5)})
        d.step({}, ended=[1])
        d.run(30, {})
        g = d.ids.registry['B-01']
        self.assertIn(g.status, ('missing', 'offscreen'))
        self.assertTrue(g.gallery)


if __name__ == '__main__':
    unittest.main()
