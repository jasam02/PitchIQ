import contextlib
import unittest

import numpy as np

import scene  # noqa: F401  (adds the prototype directory to sys.path)
from soccer.appearance import Descriptor, encode_descriptor
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
        self.pitch_reliable = True   # False: no usable pitch on these frames (the registration failed, not the tracking)

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
                               'model': self.model, 'pitch_reliable': self.pitch_reliable, 'filter_enabled': True, 'aspect': 16/9,
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

    def test_referee_is_ignored_never_an_identity(self):
        # Referees are not match entities: no REF-n, no team, no gallery; a local track that is NON_TARGET.
        d = Driver()
        visible = {1: (Person('REF'), .5, .5), 2: (Person('A'), .3, .6), 3: (Person('B'), .7, .6)}
        out = d.run(2, visible)
        self.assertEqual((out[1]['id'], out[1]['display'], out[1]['role'], out[1]['state'], out[1]['team'], out[1]['hint']),
                         (None, 'IGNORE', 'NON_TARGET', 'nontarget', None, 'REFEREE_LIKE'), out[1])
        groups = d.ids.summary()
        self.assertEqual(groups['referees'], [])
        self.assertFalse([g for g in d.ids.registry.values() if g.role == 'referee' or g.pid.startswith('REF')])
        self.assertEqual(sorted(g['id'] for g in groups['players']), ['A-01', 'B-01'])
        found = [e for e in d.kinds('nontarget') if e['message'].startswith('NON-TARGET PERSON')]
        self.assertEqual(len(found), 1)
        for line in ('Local Track: 1', 'Hint: REFEREE_LIKE', 'Global identity: none', 'Non-target confidence'):
            self.assertIn(line, found[0]['message'])
        self.assertFalse(d.ids.tracks[1].player_id)

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
        # The detector's votes alone never confirm a goalkeeper: the keeper is also the deepest person.
        out = d.run(6, visible, cues=lambda tid: {'extreme': True} if tid == 1 else {})
        self.assertEqual(out[1]['id'], 'GK-1')
        self.assertEqual((out[1]['role'], out[1]['team'], out[1]['display'], out[1]['label']), ('GOALKEEPER', 'A', 'GK-A', 'GOALKEEPER_TEAM_A'))
        self.assertEqual(len(d.ids.summary()['goalkeepers']), 1)
        self.assertTrue(any('GOALKEEPER TEAM' in e['message'] and 'Team: A' in e['message'] for e in d.kinds('role')))

    def test_referee_promoted_as_a_player_by_mistake_is_withdrawn(self):
        # A referee whose jersey looks like team B's. Early on the kit model has no shorts prototypes and the
        # detector calls them a player: they become a team B player. Then the black shorts and the detector's
        # referee votes reveal the referee. The identity never became an established team B player and showed
        # this same person from its start: it is withdrawn, and every frame it labelled becomes NON_TARGET.
        d = Driver()
        d.model = MODEL.replace(shorts_a=None, shorts_b=None, referee_shorts=None)
        ref = Person('B', shorts='REF', look='REF', cls='player')
        visible = {1: (ref, .5, .5), 2: (Person('A'), .3, .6), 3: (Person('B'), .7, .6)}
        out = d.run(3, visible)
        first = out[1]['id']
        self.assertTrue(first and first.startswith('B-'), out[1])
        d.model, ref.cls = MODEL, 'referee'
        out = d.run(12, visible)
        self.assertEqual((out[1]['id'], out[1]['display'], out[1]['state'], out[1]['team']), (None, 'IGNORE', 'nontarget', None), out[1])
        withdrawn = [e for e in d.kinds('nontarget') if e['message'].startswith('PLAYER IDENTITY WITHDRAWN')]
        self.assertEqual(len(withdrawn), 1, [e['message'][:70] for e in d.events if e['kind'] in ('role', 'nontarget')])
        for line in (f'Global ID: {first} (was team B)', 'Team history:', 'Same person since the identity was created: yes',
                     'Observations relabelled NON_TARGET', 'Referee-like evidence: 4 clean checks in a row'):
            self.assertIn(line, withdrawn[0]['message'])
        g = d.ids.registry[first]
        self.assertEqual(g.status, 'retired')
        self.assertTrue(g.withdrawn)
        self.assertFalse([p for frame in d.people for p in frame if p['track'] == 1 and p['id']], 'no frame of the referee keeps a player identity')
        groups = d.ids.summary()
        self.assertEqual([x['id'] for x in groups['withdrawn']], [first])
        self.assertNotIn(first, [x['id'] for x in groups['players']])
        self.assertEqual(groups['referees'], [])
        self.assertFalse([e for e in d.kinds('role') if 'New role: REFEREE' in e['message']])

    def test_returning_referee_is_ignored_again_not_made_a_player(self):
        # shorts: black shorts give the referee away; look-only: same kit as team B, only their own
        # appearance (OSNet, like the ignored official seen before) and the detector tell them apart.
        for variant, shorts in (('shorts', 'REF'), ('look-only', 'B')):
            with self.subTest(variant):
                d = Driver()
                ref = Person('B', shorts=shorts, look='REF', cls='referee')
                team = {10+i: t for i, t in enumerate(roster('A', 3, x0=.1)+roster('B', 3, x0=.6))}
                out = d.run(4, {1: (ref, .45, .4), **team})
                self.assertEqual((out[1]['id'], out[1]['state']), (None, 'nontarget'), out[1])
                d.step(team, ended=[1])
                d.run(6, team)
                out = d.run(5, {**team, 40: (ref, .5, .45)})
                self.assertIsNone(out[40]['id'], out[40])
                self.assertEqual(out[40]['state'], 'nontarget', out[40])
                self.assertFalse({p['id'] for frame in d.people for p in frame if p['track'] in (1, 40) and p['id']})
                self.assertEqual(sorted(g['id'] for g in d.ids.summary()['players']), ['A-01', 'A-02', 'A-03', 'B-01', 'B-02', 'B-03'])
                self.assertFalse([g for g in d.ids.registry.values() if g.role == 'referee'])

    def test_track_switch_onto_the_referee_never_relabels_the_player(self):
        d = Driver()
        player, ref = Person('A'), Person('REF')
        team = {10+i: t for i, t in enumerate(roster('A', 2, x0=.1)+roster('B', 3, x0=.6))}
        d.run(3, {1: (player, .4, .5), **team})
        pid = d.ids.tracks[1].player_id
        self.assertTrue(pid and pid.startswith('A-'))
        switched = len(d.people)
        d.run(10, {1: (ref, .42, .5), **team})  # the local tracker now follows the referee on track 1
        self.assertIsNone(d.ids.tracks[1].player_id)
        self.assertEqual(d.ids.tracks[1].state, 'nontarget')   # the referee is ignored; the player's identity is not theirs
        self.assertNotEqual(d.ids.registry[pid].status, 'retired')
        self.assertTrue([e for e in d.kinds('switch') if e['track'] == 1])
        before = {p['id'] for frame in d.people[:switched] for p in frame if p['track'] == 1 and p['id']}
        self.assertEqual(before, {pid}, 'the player observations before the switch keep the player identity')
        out = d.run(3, {**team, 1: (ref, .42, .5), 30: (player, .3, .55)})  # the player is back on a new track
        self.assertEqual(out[30]['id'], pid)

    def test_assistant_referee_on_the_touchline_is_ignored(self):
        d = Driver()
        visible = {1: (Person('REF', cls='player'), .5, .93)}  # the detector calls them a player
        visible.update({10+i: t for i, t in enumerate(roster('A', 3)+roster('B', 3, x0=.55))})
        out = d.run(5, visible, zone=lambda tid: 'boundary' if tid == 1 else 'inside')
        self.assertEqual((out[1]['id'], out[1]['state'], out[1]['role']), (None, 'nontarget', 'NON_TARGET'), out[1])
        found = [e for e in d.kinds('nontarget') if e['message'].startswith('NON-TARGET PERSON')]
        self.assertEqual(len(found), 1)
        for line in ('Local Track: 1', 'Team A similarity', 'Team B similarity', 'Non-target confidence', 'Global identity: none'):
            self.assertIn(line, found[0]['message'])
        self.assertFalse({p['id'] for frame in d.people for p in frame if p['track'] == 1 and p['id']})
        self.assertFalse([g for g in d.ids.registry.values() if g.role == 'referee'])

    def test_odd_kit_alone_never_makes_anyone_ignored(self):
        # A kit matching neither team (a team kit in shadow does that too) with the detector never saying
        # referee: nobody is ignored as referee-like on colour alone; they stay an undecided candidate.
        d = Driver()
        odd = Person('GK', cls='player')
        visible = {1: (odd, .5, .5), 2: (Person('A'), .3, .6), 3: (Person('B'), .7, .6)}
        out = d.run(6, visible)
        self.assertIsNone(out[1]['id'], out[1])
        self.assertIn(out[1]['state'], ('candidate', 'unknown'), out[1])
        self.assertFalse([e for e in d.kinds('nontarget')])
        self.assertEqual(sorted(g['id'] for g in d.ids.summary()['players']), ['A-01', 'B-01'])

    def test_loosely_matching_kit_with_detector_referee_votes_is_ignored(self):
        # A referee kit close to team B's colours (inside B's spread, but near its edge) and a detector
        # that calls them a referee most of the time: the detector is not held back by the loose match.
        with pink_kit():
            d = Driver()
            ref = Person('PINK', cls='referee')
            ref.cycle = ['referee', 'referee', 'player', 'referee', 'referee', 'referee', 'player', 'referee', 'referee', 'referee']
            visible = {1: (ref, .5, .5), 2: (Person('A'), .3, .6), 3: (Person('B'), .7, .6)}
            out = d.run(6, visible)
            self.assertEqual((out[1]['id'], out[1]['state']), (None, 'nontarget'), out[1])
            self.assertGreaterEqual(len(d.ids.referee_samples(MODEL)), 3, "a loosely matching kit the detector calls referee seeds the referee kit")
            # The same kit with the detector mostly saying player stays a team B player.
            d = Driver()
            ref = Person('PINK', cls='player')
            ref.cycle = ['player', 'player', 'referee', 'player', 'player']
            out = d.run(6, {1: (ref, .5, .5), 2: (Person('A'), .3, .6), 3: (Person('B'), .7, .6)})
            self.assertEqual((out[1]['team'], out[1]['role']), ('B', 'PLAYER'), out[1])
            self.assertEqual(d.ids.referee_samples(MODEL), [])

    def test_referee_in_a_team_like_kit_is_withdrawn_once_the_officials_look_is_known(self):
        # The centre referee in a kit near team B's colours, the detector calling them a player: with no
        # official seen yet they become a team B player. Once the other official is seen (and ignored), their
        # whole appearance says "like the officials, not like team B's players" (the officials wear one
        # uniform) and they follow play without playing it: never an established team B player (the kit only
        # loosely matches), the same person from the start, so the identity is withdrawn.
        with pink_kit():
            d = Driver()
            ref1 = Person('REF')
            centre = Person('PINK', cls='player', unique=same_uniform(ref1))
            team = {10+i: t for i, t in enumerate(roster('A', 3, x0=.1)+roster('B', 3, x0=.6))}
            out = d.run(3, {1: (centre, .5, .45), **team})
            pid = out[1]['id']
            self.assertTrue(pid and pid.startswith('B-'), out[1])
            # From here the detector calls them a referee now and then (as the real detector does). The officials'
            # look alone would not be enough: it must be backed by something else (see IdentityManager._supported).
            centre.cycle = ['player', 'referee', 'player', 'player', 'referee']
            view = {1: (centre, .5, .45), 2: (ref1, .35, .4), **team}
            out = d.run(12, view, ball=(.56, .55))  # the ball a couple of metres from the referee, never at their feet
            self.assertEqual((out[1]['id'], out[1]['state'], out[1]['team']), (None, 'nontarget', None), out[1])
            self.assertEqual(out[2]['state'], 'nontarget')
            self.assertTrue(d.ids.registry[pid].withdrawn)
            self.assertFalse({p['id'] for frame in d.people for p in frame if p['track'] == 1 and p['id']})
            withdrawn = [e for e in d.kinds('nontarget') if e['message'].startswith('PLAYER IDENTITY WITHDRAWN')]
            self.assertEqual(len(withdrawn), 1, [e['message'][:70] for e in d.events if e['kind'] in ('role', 'nontarget')])
            self.assertIn('looks ref', withdrawn[0]['message'])
            players = d.ids.summary()['players']
            self.assertEqual(sum(1 for g in players if g['team'] == 'B'), 3)
            self.assertNotIn(pid, [g['id'] for g in players])
            # Settled: the detector calling them a player for a long while never makes them a B player again.
            centre.cycle = None
            out = d.run(12, {**view, 1: (centre, .55, .5)})
            self.assertEqual((out[1]['id'], out[1]['state']), (None, 'nontarget'))
            self.assertFalse([g for g in d.ids.registry.values() if g.role == 'referee'])

    def test_touchline_official_in_a_team_like_kit_is_ignored_as_a_sideline_official(self):
        # The officials at the near touchline: a kit near team B's colours, the detector calling them a
        # player, on or just outside the line and moving along it. With another official already ignored, the
        # whole appearance and the touchline behaviour make them a SIDELINE_OFFICIAL: ignored, never a team B
        # player.
        with pink_kit():
            d = Driver()
            ref1 = Person('REF')
            official = Person('PINK', cls='player', unique=same_uniform(ref1))
            team = {10+i: t for i, t in enumerate(roster('A', 3, x0=.1)+roster('B', 3, x0=.6))}
            d.run(2, {2: (ref1, .35, .4), **team})
            self.assertEqual(d.ids.tracks[2].state, 'nontarget')
            edge = {'edge': {'side': 'near', 'metres': 1.0, 'dir': (1.0, 0.0)}}
            x = .3
            for _ in range(25):  # 5 s walking along the near touchline, a metre outside it
                x += .006
                out = d.step({1: (official, x, .93), 2: (ref1, .35, .4), **team}, zone=lambda tid: 'boundary' if tid == 1 else 'inside',
                             cues=lambda tid: edge if tid == 1 else {})
            self.assertEqual((out[1]['id'], out[1]['state'], out[1]['team'], out[1]['hint']), (None, 'nontarget', None, 'SIDELINE_OFFICIAL'), out[1])
            found = [e for e in d.kinds('nontarget') if e['message'].startswith('NON-TARGET PERSON') and e['track'] == 1]
            self.assertEqual(len(found), 1, [e['message'][:60] for e in d.kinds('nontarget')])
            for line in ("Appearance vs the officials' look / team A / team B:", 'Time on or outside the touchline: 100%', 'movement along it: 1.00',
                         'Final: ASSISTANT REFEREE', 'Hint: SIDELINE_OFFICIAL'):
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
        self.assertEqual(d.ids.tracks[2].state, 'nontarget')   # the referee is ignored, the coach is not referee-like
        self.assertEqual(d.ids.summary()['referees'], [])

    def test_full_team_doubts_another_team_kit_player_the_detector_calls_a_referee(self):
        # Team B already has its full count of identities on the pitch. Another person in team B's kit whom the
        # detector calls a referee more often than not is then ignored as referee-like, not one more B player;
        # with room in the team the same evidence, held back by the team kit, leaves them a team B player.
        for cap in (3, 11):
            with self.subTest(cap=cap):
                d = Driver(Options(max_per_team=cap))
                team = {10+i: t for i, t in enumerate(roster('A', 3, x0=.1)+roster('B', 3, x0=.6))}
                d.run(2, team)
                extra = Person('B', cls='player')
                extra.cycle = ['referee', 'referee', 'player', 'referee', 'player']
                out = d.run(4, {1: (extra, .5, .45), **team})
                if cap == 3:
                    self.assertEqual((out[1]['id'], out[1]['state'], out[1]['team']), (None, 'nontarget', None), out[1])
                    found = [e for e in d.kinds('nontarget') if e['message'].startswith('NON-TARGET PERSON')]
                    self.assertIn('Team already has its full count of identities', found[0]['message'])
                else:
                    self.assertTrue(out[1]['id'] and out[1]['id'].startswith('B-'), out[1])
                    self.assertEqual((out[1]['role'], out[1]['team']), ('PLAYER', 'B'), out[1])

    def test_ignored_referee_stays_ignored_while_the_detector_wavers(self):
        d = Driver()
        ref = Person('REF')
        visible = {1: (ref, .5, .5), 2: (Person('B'), .7, .6), 3: (Person('A'), .3, .6)}
        d.run(3, visible)
        self.assertEqual(d.ids.tracks[1].state, 'nontarget')
        ref.cls = 'player'  # the detector changes its mind for a long while; the kit is still the referee's
        out = d.run(12, visible)
        self.assertEqual((out[1]['id'], out[1]['state']), (None, 'nontarget'))
        self.assertEqual(len(d.ids.registry), 2)

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

    def test_keeper_first_ignored_as_referee_like_becomes_the_goalkeeper(self):
        d = Driver()
        keeper = Person('GK', cls='player')
        keeper.cycle = ['referee', 'player', 'player', 'referee', 'player']  # the detector sometimes takes the odd kit for a referee
        out = d.run(3, self.keeper_scene(keeper, x=.38))  # away from the goal: referee-like, ignored
        self.assertEqual((out[1]['id'], out[1]['state']), (None, 'nontarget'), out[1])
        keeper.cycle, keeper.cls = None, 'player'
        at_goal = {'near_goal': True, 'goal_side': 'left', 'extreme': True, 'isolated': True}
        out = d.run(15, self.keeper_scene(keeper), cues=lambda tid: at_goal if tid == 1 else {})
        self.assertEqual((out[1]['id'], out[1]['display'], out[1]['role']), ('GK-1', 'GK-A', 'GOALKEEPER'), out[1])
        found = [e for e in d.kinds('role') if e['message'].startswith('GOALKEEPER IDENTIFIED')]
        self.assertEqual(len(found), 1)
        self.assertTrue([e for e in d.kinds('nontarget') if e['message'].startswith('NON-TARGET PERSON RECONSIDERED')])
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

    @staticmethod
    def ends_match():
        # Two teams in view with team B's defenders at the right end (team B defends the right goal).
        return {10+i: (Person(team), x, .5+.03*i) for i, (team, x) in enumerate((('A', .2), ('A', .26), ('A', .32), ('B', .62), ('B', .7), ('B', .78)))}

    def test_sideline_official_the_detector_calls_goalkeeper_never_reserves_the_slot(self):
        # The stale-slot failure: a person on the pitch just inside the touchline, next to team B's
        # defenders, whom the detector calls a goalkeeper. They used to become GK-B within seconds and
        # kept team B's goalkeeper slot after they were found to be OUT, so the real keeper was GK-?.
        d = Driver()
        team = self.ends_match()
        d.run(2, team)
        official = Person('GK', cls='goalkeeper')
        corridor = {'edge': {'side': 'near', 'metres': 1.0, 'dir': (1.0, 0.0)}}
        out = d.run(6, {1: (official, .9, .88), **team}, cues=lambda tid: corridor if tid == 1 else {})
        self.assertIsNone(out[1]['id'], out[1])
        # Never a goalkeeper: at most a candidate while the sideline evidence builds, then unknown.
        self.assertIn(out[1]['label'], ('UNKNOWN', 'CANDIDATE', 'GOALKEEPER_CANDIDATE'))
        self.assertNotIn(out[1]['display'], ('GK-B', 'GK-A'))
        self.assertIn('touchline', out[1]['reason'])
        self.assertIn('gk candidate', out[1]['why'])
        self.assertEqual(d.ids.keeper_slots(), {'A': None, 'B': None})
        self.assertEqual(d.ids.summary()['goalkeepers'], [])
        self.assertFalse([e for e in d.kinds('role') if e['message'].startswith('GOALKEEPER IDENTIFIED')])
        # Then they stand outside the pitch: OUT; the slot stays free.
        out = d.run(7, {1: (official, .9, .97), **team}, zone=lambda tid: 'outside' if tid == 1 else 'inside', cues=lambda tid: corridor if tid == 1 else {})
        self.assertEqual((out[1]['id'], out[1]['display'], out[1]['label']), (None, 'OUT', 'REJECTED_OUTSIDE_FIELD'), out[1])
        self.assertEqual(d.ids.keeper_slots()['B'], None)
        # The real team B goalkeeper appears at the right goal, is confirmed, and takes GK-B.
        keeper = Person('GK', cls='goalkeeper', look='B')
        at_goal = {'near_goal': True, 'goal_side': 'right', 'extreme': True, 'isolated': True}
        out = d.run(8, {2: (keeper, .95, .55), **team}, cues=lambda tid: at_goal if tid == 2 else {})
        self.assertEqual((out[2]['id'], out[2]['display'], out[2]['role'], out[2]['team']), ('GK-1', 'GK-B', 'GOALKEEPER', 'B'), out[2])
        self.assertEqual(d.ids.keeper_slots()['B'], 'GK-1')
        found = [e for e in d.kinds('role') if e['message'].startswith('GOALKEEPER IDENTIFIED')]
        self.assertEqual(len(found), 1)
        self.assertIn('Goalkeeper confirmation: CONFIRMED (goal-area residence', found[0]['message'])

    def test_midfield_person_the_detector_calls_goalkeeper_stays_a_candidate_with_its_evidence_logged(self):
        d = Driver()
        team = self.ends_match()
        d.run(2, team)
        odd = Person('GK', cls='goalkeeper')   # inside the pitch near the middle, never at a goal
        out = d.run(6, {1: (odd, .5, .45), **team})
        self.assertIsNone(out[1]['id'], out[1])
        self.assertEqual((out[1]['display'], out[1]['label'], out[1]['role']), ('GK-?', 'GOALKEEPER_CANDIDATE', 'GOALKEEPER'))
        self.assertEqual(d.ids.keeper_slots(), {'A': None, 'B': None})
        candidates = [e for e in d.kinds('role') if e['message'].startswith('GOALKEEPER CANDIDATE') and 'DROPPED' not in e['message'][:30]]
        self.assertTrue(candidates)
        for line in ('Local Track: 1', 'Role detector: GK 100%', 'Pitch state: INSIDE 100%', 'Goal proximity score: 0.00', 'Penalty-area residence: 0%',
                     'Goalkeeper confirmation: CANDIDATE (no goal-area or deepest-person evidence yet', 'State: GK_CANDIDATE'):
            self.assertIn(line, candidates[0]['message'], candidates[0]['message'])
        self.assertLessEqual(len(candidates), 2)   # logged every few seconds, not every tick


    def test_confirmed_goalkeeper_is_revoked_after_sustained_contrary_evidence_and_the_slot_reused(self):
        d = Driver()
        team = self.ends_match()
        wrong = Person('GK', cls='goalkeeper')
        at_goal = {'near_goal': True, 'goal_side': 'right', 'extreme': True, 'isolated': True}
        out = d.run(6, {1: (wrong, .95, .55), **team}, cues=lambda tid: at_goal if tid == 1 else {})
        self.assertEqual((out[1]['id'], out[1]['display']), ('GK-1', 'GK-B'))
        self.assertTrue(d.ids.registry['GK-1'].role_locked)
        # A few odd frames do not shake a confirmed keeper; a long time outside the pitch does.
        out = d.run(2, {1: (wrong, .95, .97), **team}, zone=lambda tid: 'outside' if tid == 1 else 'inside')
        self.assertEqual(d.ids.keeper_slots()['B'], 'GK-1')
        out = d.run(9, {1: (wrong, .95, .97), **team}, zone=lambda tid: 'outside' if tid == 1 else 'inside')
        g = d.ids.registry['GK-1']
        self.assertEqual((g.status, g.role_locked, bool(g.revoked)), ('unknown', False, True))
        self.assertEqual(d.ids.keeper_slots()['B'], None)
        self.assertIsNone(out[1]['id'])
        revoked = [e for e in d.kinds('role') if e['message'].startswith('GOALKEEPER REVOKED')]
        self.assertEqual(len(revoked), 1)
        for line in ('Global ID: GK-1 (was GK-B)', 'outside the pitch', 'Team B goalkeeper slot: released'):
            self.assertIn(line, revoked[0]['message'])
        # The real keeper claims the slot; the revoked identity is kept but holds nothing.
        keeper = Person('GK', cls='goalkeeper', look='B')
        out = d.run(8, {2: (keeper, .95, .55), **team}, cues=lambda tid: at_goal if tid == 2 else {})
        self.assertEqual((out[2]['id'], out[2]['display']), ('GK-2', 'GK-B'))
        self.assertEqual(d.ids.keeper_slots()['B'], 'GK-2')
        self.assertEqual([(x['id'], x['status'], bool(x['revoked'])) for x in d.ids.summary()['goalkeepers']], [('GK-1', 'UNKNOWN', True), ('GK-2', 'ACTIVE', False)])
        self.assertTrue(all(p['label'] == 'GOALKEEPER_REVOKED' for frame in d.people for p in frame if p['track'] == 1 and p.get('id') == 'GK-1'))

    def test_stronger_goalkeeper_candidate_takes_the_slot_from_a_weaker_absent_owner(self):
        d = Driver()
        team = self.ends_match()
        weak = Person('GK', cls='goalkeeper')
        deep = {'extreme': True, 'isolated': True}   # the deepest person, alone, but never seen at a goal
        out = d.run(6, {1: (weak, .95, .55), **team}, cues=lambda tid: deep if tid == 1 else {})
        self.assertEqual((out[1]['id'], out[1]['display']), ('GK-1', 'GK-B'))
        d.step(team, ended=[1])
        strong = Person('A', cls='goalkeeper')   # the real keeper, in a kit near their team's
        at_goal = {'near_goal': True, 'goal_side': 'right', 'extreme': True, 'isolated': True}
        out = d.run(8, {2: (strong, .95, .55), **team}, cues=lambda tid: at_goal if tid == 2 else {})
        self.assertEqual((out[2]['id'], out[2]['display']), ('GK-2', 'GK-B'), out[2])
        self.assertEqual(d.ids.keeper_slots()['B'], 'GK-2')
        self.assertTrue(d.ids.registry['GK-1'].revoked)
        self.assertTrue(any('a stronger goalkeeper' in e['message'] for e in d.kinds('role') if e['message'].startswith('GOALKEEPER REVOKED')))

    def test_goalkeeper_slots_start_empty_and_a_late_keeper_still_claims_one(self):
        d = Driver()
        team = self.ends_match()
        d.run(20, team)   # midfield only: nobody near a goal
        self.assertEqual(d.ids.keeper_slots(), {'A': None, 'B': None})
        keeper = Person('GK', cls='player')   # the detector never says goalkeeper
        at_goal = {'near_goal': True, 'goal_side': 'right', 'extreme': True, 'isolated': True}
        out = d.run(8, {2: (keeper, .95, .55), **team}, cues=lambda tid: at_goal if tid == 2 else {})
        self.assertEqual((out[2]['id'], out[2]['display']), ('GK-1', 'GK-B'))

    def test_revoked_goalkeeper_stays_released_when_identities_are_continued(self):
        d = Driver()
        team = self.ends_match()
        wrong = Person('GK', cls='goalkeeper')
        at_goal = {'near_goal': True, 'goal_side': 'right', 'extreme': True, 'isolated': True}
        d.run(6, {1: (wrong, .95, .55), **team}, cues=lambda tid: at_goal if tid == 1 else {})
        d.run(11, {1: (wrong, .95, .97), **team}, zone=lambda tid: 'outside' if tid == 1 else 'inside')
        self.assertTrue(d.ids.registry['GK-1'].revoked)
        nxt = Driver(saved=d.ids.snapshot())
        self.assertEqual(nxt.ids.registry['GK-1'].status, 'unknown')
        self.assertEqual(nxt.ids.keeper_slots(), {'A': None, 'B': None})
        keeper = Person('GK', cls='goalkeeper', look='B')
        out = nxt.run(8, {2: (keeper, .95, .55), **team}, cues=lambda tid: at_goal if tid == 2 else {})
        self.assertEqual((out[2]['id'], out[2]['display']), ('GK-2', 'GK-B'))


    AT_RIGHT_GOAL = {'near_goal': True, 'goal_side': 'right', 'extreme': True, 'isolated': True}

    def confirmed_keeper(self, kit='GK', cls='goalkeeper', look=None):
        """A keeper confirmed and locked as GK-1 / GK-B at the right goal, with the match in view."""
        d = Driver()
        team = self.ends_match()
        keeper = Person(kit, cls=cls, look=look)
        out = d.run(8, {1: (keeper, .95, .55), **team}, cues=lambda tid: self.AT_RIGHT_GOAL if tid == 1 else {})
        self.assertEqual((out[1]['id'], out[1]['display']), ('GK-1', 'GK-B'), out[1])
        self.assertTrue(d.ids.registry['GK-1'].role_locked)
        return d, team, keeper

    def test_confirmed_keeper_walking_upfield_as_a_player_to_the_detector_keeps_the_role(self):
        d, team, keeper = self.confirmed_keeper(kit='B', cls='goalkeeper')   # a keeper in a kit near their team's
        keeper.cls = 'player'   # upfield for a corner: the detector calls them a player for 20 s
        out = d.run(20, {1: (keeper, .5, .5), **team})
        self.assertEqual((out[1]['id'], out[1]['display'], out[1]['role']), ('GK-1', 'GK-B', 'GOALKEEPER'), out[1])
        self.assertTrue(d.ids.registry['GK-1'].role_locked)
        self.assertFalse([e for e in d.kinds('role') if e['message'].startswith('GOALKEEPER REVOKED')])
        self.assertEqual(d.ids.keeper_slots()['B'], 'GK-1')

    def test_confirmed_keeper_fetching_the_ball_behind_the_goal_line_keeps_the_role(self):
        d, team, keeper = self.confirmed_keeper()
        behind = {'near_goal': True, 'goal_side': 'right', 'extreme': True, 'isolated': True}
        out = d.run(12, {1: (keeper, .99, .55), **team}, zone=lambda tid: 'outside' if tid == 1 else 'inside', cues=lambda tid: behind if tid == 1 else {})
        self.assertEqual(d.ids.registry['GK-1'].status, 'active')
        self.assertEqual(d.ids.keeper_slots()['B'], 'GK-1')
        self.assertFalse([e for e in d.kinds('role') if e['message'].startswith('GOALKEEPER REVOKED')])
        out = d.run(4, {1: (keeper, .95, .55), **team}, cues=lambda tid: self.AT_RIGHT_GOAL if tid == 1 else {})
        self.assertEqual((out[1]['id'], out[1]['display']), ('GK-1', 'GK-B'))

    def test_keeper_confirmed_at_a_goal_only_loses_the_team_to_a_later_keeper_there(self):
        d, team, keeper = self.confirmed_keeper()
        d.run(12, {1: (keeper, .5, .5), **team})    # upfield for a while (last window: no goal area)
        d.step(team, ended=[1])                      # then out of view
        d.run(12, team)
        other = Person('A', cls='goalkeeper')         # a different person is confirmed in the same goal area
        out = d.run(8, {2: (other, .95, .55), **team}, cues=lambda tid: self.AT_RIGHT_GOAL if tid == 2 else {})
        self.assertEqual((out[2]['id'], out[2]['display']), ('GK-2', 'GK-B'), out[2])
        g = d.ids.registry['GK-1']
        self.assertEqual((g.role, g.team, g.status, g.role_locked, g.revoked), ('goalkeeper', None, 'missing', True, ''))
        self.assertTrue(any(e['message'].startswith('GOALKEEPER TEAM RELEASED') for e in d.kinds('role')))
        self.assertFalse([e for e in d.kinds('role') if e['message'].startswith('GOALKEEPER REVOKED')])
        # The first keeper returns to that goal and is re-identified as GK-1, still a goalkeeper.
        d.step(team, ended=[2])
        d.run(12, team)
        out = d.run(8, {3: (keeper, .95, .55), **team}, cues=lambda tid: self.AT_RIGHT_GOAL if tid == 3 else {})
        self.assertEqual((out[3]['id'], out[3]['role']), ('GK-1', 'GOALKEEPER'), out[3])

    def test_real_keeper_in_a_similar_kit_is_decided_within_seconds_behind_a_weak_absent_owner(self):
        d = Driver()
        team = self.ends_match()
        weak = Person('GK', cls='goalkeeper')
        d.run(6, {1: (weak, .95, .55), **team}, cues=lambda tid: {'extreme': True, 'isolated': True} if tid == 1 else {})
        self.assertEqual(d.ids.keeper_slots()['B'], 'GK-1')
        d.step(team, ended=[1])
        d.run(12, team)
        real = Person('GK', cls='goalkeeper')   # the same kit, another person: re-ID is undecided
        out = d.run(14, {2: (real, .95, .55), **team}, cues=lambda tid: self.AT_RIGHT_GOAL if tid == 2 else {})
        self.assertIsNotNone(out[2]['id'], out[2])
        self.assertEqual((out[2]['display'], out[2]['role']), ('GK-B', 'GOALKEEPER'), out[2])
        self.assertEqual(d.ids.keeper_slots()['B'], out[2]['id'])

    def test_vetoed_sideline_candidate_is_never_matched_to_a_missing_keeper(self):
        d, team, keeper = self.confirmed_keeper()
        d.step(team, ended=[1])
        d.run(2, team)
        sub = Person('GK', cls='goalkeeper')   # same kit, warming up along the near touchline
        corridor = {'edge': {'side': 'near', 'metres': 1.0, 'dir': (1.0, 0.0)}}
        zones = ['inside', 'inside', 'boundary', 'inside', 'inside', 'inside', 'boundary', 'inside', 'inside', 'boundary']
        k = [0]
        def zone(tid):
            k[0] += 1
            return zones[k[0] % len(zones)] if tid == 2 else 'inside'
        out = d.run(12, {2: (sub, .5, .9), **team}, zone=zone, cues=lambda tid: corridor if tid == 2 and k[0] % 10 < 4 else {})
        self.assertIsNone(out[2]['id'], out[2])
        self.assertNotEqual(out[2]['display'], 'GK-B')
        self.assertEqual((d.ids.registry['GK-1'].status, d.ids.registry['GK-1'].revoked), ('missing', ''))
        self.assertEqual(d.ids.keeper_slots()['B'], 'GK-1')

    def test_keeper_kit_team_vote_does_not_bypass_the_slot(self):
        d, team, keeper = self.confirmed_keeper()
        d.model = d.model.replace(keepers=[{'team': 'B', 'jersey': JERSEY['GK']}])   # GK-1's kit is now team B's keeper kit
        other = Person('GK', cls='goalkeeper')   # the other team's keeper in a similar kit, at the LEFT goal, while GK-1 is in view
        at_left = {'near_goal': True, 'goal_side': 'left', 'extreme': True, 'isolated': True}
        out = d.run(8, {1: (keeper, .95, .55), 2: (other, .05, .55), **team}, cues=lambda tid: self.AT_RIGHT_GOAL if tid == 1 else at_left if tid == 2 else {})
        self.assertEqual((out[1]['id'], out[1]['display']), ('GK-1', 'GK-B'))
        self.assertNotEqual(out[2]['display'], 'GK-B', out[2])
        self.assertLessEqual(sum(1 for g in d.ids.summary()['goalkeepers'] if g['team'] == 'B' and not g['revoked']), 1)


    @staticmethod
    def remembered(pid, role, team, who, excluded=False):
        """A saved identity as a correction in the review page leaves it: the person's appearance, the
        role and team you gave them, marked manual (see corrections.py)."""
        gallery = [{'t': float(k), 'd': encode_descriptor(who.descriptor())} for k in range(6)]
        return {'id': pid, 'role': role, 'team': team, 'status': 'unknown' if excluded else 'missing', 'gallery': gallery,
                'roleLocked': not excluded, 'manual': True, 'excluded': excluded, 'firstSeen': 0.0, 'lastSeen': 0.0}

    def test_referee_you_corrected_is_recognised_and_left_out_in_a_later_run(self):
        # The referee wears a jersey like team B's and the detector calls them a player: on their own they
        # would become a team B player. You corrected them to a referee (REF-4) in an earlier version: referees
        # are not identities now, so they are recognised by appearance and left out, never a B player first.
        ref = Person('B', look='REF', cls='player')
        d = Driver(saved={'identities': [self.remembered('REF-4', 'referee', None, ref)], 'counters': {}})
        team = {10+i: t for i, t in enumerate(roster('A', 3, x0=.1)+roster('B', 3, x0=.6))}
        out = d.run(4, {1: (ref, .45, .5), **team})
        self.assertEqual((out[1]['id'], out[1]['display'], out[1]['label'], out[1].get('excludedId')), (None, 'OUT', 'EXCLUDED_BY_YOU', 'REF-4'), out[1])
        self.assertFalse({p['id'] for frame in d.people for p in frame if p['track'] == 1 and p['id']})   # never a B player first
        self.assertTrue(any(e['message'].startswith('RECOGNISED: NOT PART OF THE MATCH') for e in d.kinds('manual')))
        self.assertTrue(any(e['message'].startswith('REMEMBERED FROM YOUR CORRECTIONS') and 'REF-4: not part of the match' in e['message'] for e in d.kinds('manual')))
        self.assertEqual(sum(1 for g in d.ids.summary()['players'] if g['team'] == 'B'), 3)
        self.assertEqual([g['id'] for g in d.ids.summary()['excluded']], ['REF-4'])
        out = d.run(12, {1: (ref, .45, .5), **team})
        self.assertEqual((out[1]['id'], out[1]['display']), (None, 'OUT'))
        self.assertTrue(d.ids.registry['REF-4'].manual)

    def test_player_you_corrected_is_never_converted_to_a_referee(self):
        mid = Person('B', cls='referee')   # the detector keeps calling this team B player a referee
        d = Driver(saved={'identities': [self.remembered('B-15', 'player', 'B', mid)], 'counters': {}})
        team = {10+i: t for i, t in enumerate(roster('A', 3, x0=.1)+roster('B', 3, x0=.6))}
        out = d.run(16, {1: (mid, .45, .5), **team})
        self.assertEqual((out[1]['id'], out[1]['role'], out[1]['team']), ('B-15', 'PLAYER', 'B'), out[1])
        self.assertEqual(d.ids.summary()['referees'], [])
        self.assertFalse([e for e in d.kinds('role') if 'New role: REFEREE' in e['message'] or 'REFEREE IDENTIFIED' in e['message']])

    def test_person_you_marked_as_not_part_of_the_match_is_left_out(self):
        coach = Person('GK', cls='player')   # a coach standing on the pitch in a kit matching neither team
        d = Driver(saved={'identities': [self.remembered('B-08', 'player', 'B', coach, excluded=True)], 'counters': {}})
        team = {10+i: t for i, t in enumerate(roster('A', 3, x0=.1)+roster('B', 3, x0=.6))}
        out = d.run(8, {1: (coach, .45, .5), **team})
        self.assertEqual((out[1]['id'], out[1]['display'], out[1]['label'], out[1]['state'], out[1].get('excludedId')), (None, 'OUT', 'EXCLUDED_BY_YOU', 'rejected', 'B-08'), out[1])
        self.assertFalse({p['id'] for frame in d.people for p in frame if p['track'] == 1 and p['id']})
        self.assertTrue(any(e['message'].startswith('RECOGNISED: NOT PART OF THE MATCH') for e in d.kinds('manual')))
        self.assertEqual(d.ids.summary()['goalkeepers'], [])
        self.assertEqual([g['id'] for g in d.ids.summary()['excluded']], ['B-08'])

    def test_recognising_a_remembered_person_needs_appearance_that_points_clearly_to_them(self):
        # The shortcut to a person you corrected skips the usual promotion and re-identification, so it is
        # taken only when the crops look clearly more like that person than like anyone the tracker knows.
        # Someone halfway between the remembered referee and a known teammate (very like both) does not
        # qualify, however similar.
        team = {10+i: t for i, t in enumerate(roster('A', 3, x0=.1)+roster('B', 3, x0=.6))}
        mate = team[13][0]
        ref = Person('B', cls='player')
        d = Driver(saved={'identities': [self.remembered('REF-4', 'referee', None, ref)], 'counters': {}})
        d.run(3, team)
        between = Person('B', unique=unit(ref.unique+mate.unique), cls='player')
        checks = []
        original = d.ids._manual_check
        def spy(tr, t):
            before = tr.manual_checked
            found = original(tr, t)
            if tr.id == 1 and tr.manual_checked != before:
                checks.append((d.ids._best_appearance(tr, lambda o: o.pid == 'REF-4'), tr.manual_hits, found))
            return found
        d.ids._manual_check = spy
        d.run(4, {1: (between, .45, .5), **team})
        self.assertTrue(checks)
        self.assertGreaterEqual(max(similarity for similarity, _, _ in checks), .8)
        self.assertEqual({(hits, found) for _, hits, found in checks}, {(0, None)})
        self.assertFalse([e for e in d.events if e['message'].startswith('RECOGNISED')])

    def test_manual_flags_survive_the_saved_identities(self):
        ref = Person('B', look='REF', cls='player')
        d = Driver(saved={'identities': [self.remembered('REF-4', 'referee', None, ref), self.remembered('B-08', 'player', 'B', Person('GK'), excluded=True)], 'counters': {}})
        snap = d.ids.snapshot()
        nxt = Driver(saved=snap)
        # A referee you corrected is remembered as someone to leave out (referees are not identities).
        self.assertEqual((nxt.ids.registry['REF-4'].manual, nxt.ids.registry['REF-4'].excluded, nxt.ids.registry['REF-4'].status), (True, True, 'unknown'))
        self.assertEqual((nxt.ids.registry['B-08'].excluded, nxt.ids.registry['B-08'].status), (True, 'unknown'))

    def test_identity_is_never_deleted(self):
        d = Driver()
        p = Person('B')
        d.run(2, {1: (p, .5, .5)})
        d.step({}, ended=[1])
        d.run(30, {})
        g = d.ids.registry['B-01']
        self.assertIn(g.status, ('missing', 'offscreen'))
        self.assertTrue(g.gallery)


class UnknownPitchTests(unittest.TestCase):
    """A registration failure is not a tracking failure: without a usable pitch, image evidence alone
    promotes people (more slowly) and nobody is rejected for a zone that was never measured."""

    def test_people_are_promoted_on_image_evidence_and_never_rejected_without_a_pitch(self):
        d = Driver()
        d.pitch_reliable = False
        visible = {i+1: p for i, p in enumerate(roster('A', 5)+roster('B', 5, x0=.6))}
        people = d.run(8.0, visible, zone='unknown')
        states = {tid: p['state'] for tid, p in people.items()}
        self.assertNotIn('rejected', states.values(), states)
        self.assertGreaterEqual(sum(1 for p in people.values() if p['id']), 8, states)
        self.assertTrue(any('unresolved' in (p.get('reason') or '') or p['id'] for p in d.people[3]), d.people[3])
        with_pitch = Driver()
        quick = with_pitch.run(1.4, visible, zone='inside')
        blind = Driver()
        blind.pitch_reliable = False
        slow = blind.run(1.4, visible, zone='unknown')
        self.assertGreater(sum(1 for p in quick.values() if p['id']), sum(1 for p in slow.values() if p['id']))   # image evidence alone takes longer

    def test_zone_evidence_from_earlier_frames_still_counts(self):
        d = Driver()
        visible = {i+1: p for i, p in enumerate(roster('A', 4))}
        d.run(3.0, visible, zone='outside')          # measured outside the pitch while it was known
        d.pitch_reliable = False
        people = d.run(4.0, visible, zone='unknown')
        self.assertTrue(all(p['state'] == 'rejected' for p in people.values()), {k: p['state'] for k, p in people.items()})


if __name__ == '__main__':
    unittest.main()
