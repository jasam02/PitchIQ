"""Identity on amateur footage (far camera, small players, kit votes that drift with scale). A kit change re-births its
local track at the change, so the person it shows now is compared with the missing players of their team instead of
being ruled out as "seen together" with everyone seen since the track's first frame (the duplicate identities A-12,
B-11 and A-13 of amateur 3, 118-182 s). In small scenes only (SoccerTracker.small_scene): small crops never release
an identity to the other team alone, never feed a gallery with the other team's kit, and a provisional pitch delays
only the creation of new identities, not the comparison with the missing ones. Driven on the synthetic people of
test_identity.py."""
import unittest

import numpy as np

import scene  # noqa: F401  (sys.path)
from soccer import tracklets
from soccer.appearance import Sample
from soccer.identity import RELEASE_BACK, VOTE_HISTORY, VOTES, GlobalPlayer, IdentityManager, Options, Stab, TrackState, Trail
from soccer.teams import kit_vote
from test_identity import MODEL, Driver, Person, box_at, roster


class Sized(Person):
    """A person whose crops have a fixed descriptor quality (a small, distant crop scores .2-.3)."""

    def __init__(self, kit, quality=.8, **kw):
        super().__init__(kit, **kw)
        self.quality = quality

    def descriptor(self, quality=None):
        return super().descriptor(self.quality)


class AmateurDriver(Driver):
    """Driver whose identity manager sees extra per-frame context (small_scene, pitch_provisional), as
    SoccerTracker.step supplies it."""

    def __init__(self, options=None, saved=None, **flags):
        super().__init__(options, saved)
        self.flags = dict(flags)
        update = self.ids.update
        self.ids.update = lambda ctx: update({**ctx, **self.flags})


def track_frames(d, tid):
    """Every person dict emitted for local track tid, in time order."""
    return [p for frame in d.people for p in frame if p['track'] == tid]


class TeamChangeRebirthTests(unittest.TestCase):
    """The amateur 3 bug: track 8 carried B-04, the local tracker moved it to a team A player behind an occlusion, 12 of
    15 kit votes later the identity was released, and the track (born at 121 s) ruled out every team A identity seen
    since then as "seen together" (score 0%) and became the duplicate A-12, over the cap without the 5 s wait."""

    def scene(self, b4_at=(.42, .6), max_per_team=4, swap_look=None, stray=0, walk=0.0, back_at=None):
        """Team B: B-01..B-03 on tracks 1-3 and the player on track 4; team A: the player on track 5 and three more.
        At 20 s track 4 ends next to track 5 and the local tracker hands track 5 to that team B player, whose crops are
        small there (quality .25: below the switch and mismatch checks, above the kit votes). stray: one crop of the
        team A player on track 5 reads team B that many ticks before the hand-over (one tick before it: ~19.8 s); walk:
        the team B player then walks right at this many image units per second; back_at: the team A player comes
        back on track 7, next to where they were, this many seconds after the hand-over."""
        d = AmateurDriver(Options(max_per_team=max_per_team))
        bs = {i+1: t for i, t in enumerate(roster('B', 3, x0=.1))}
        As = {11+i: t for i, t in enumerate(roster('A', 3, x0=.7))}
        b4, a5 = Sized('B'), Sized('A')
        hidden_b4 = Sized('B', quality=.25, unique=b4.unique if swap_look is None else swap_look)
        before = {**bs, **As, 4: (b4, *b4_at), 5: (a5, .45, .6)}
        pitch = lambda v: {tid: (x, y) for tid, (_, x, y) in v.items()}
        d.run(20-.2*stray, before, pitch=pitch(before))
        if stray:
            odd = {**before, 5: (Sized('B', unique=a5.unique, look='A'), .45, .6)}   # the same person: only the kit reads B
            d.step(odd, pitch=pitch(odd))
            d.run(.2*(stray-1), before, pitch=pitch(before))
        ids = {tid: d.ids.tracks[tid].player_id for tid in before}
        self.assertTrue(all(ids.values()), ids)
        n = len(d.ids.registry)
        t0 = d.t
        for k in range(41):
            after = {**bs, **As, 5: (hidden_b4, .45+walk*.2*k, .6)}
            if back_at is not None and d.t >= t0+back_at-1e-6:
                after[7] = (a5, .43, .6)
            d.step(after, ended=[4] if k == 0 else (), pitch=pitch(after))
        return d, ids, n, t0

    def test_kit_change_reidentifies_the_missing_player_instead_of_a_duplicate(self):
        d, ids, n, t0 = self.scene()
        changes = [e for e in d.kinds('team-change') if e['track'] == 5]
        self.assertEqual(len(changes), 1, [e['message'] for e in d.kinds('team-change')])
        self.assertEqual(changes[0]['playerId'], ids[5])
        self.assertEqual(d.ids.tracks[5].player_id, ids[4], (d.ids.tracks[5].reason, [e['message'][:140] for e in d.events if e['track'] == 5]))
        self.assertEqual(len(d.ids.registry), n, 'no duplicate identity after the kit change')
        self.assertFalse([e for e in d.kinds('new-identity') if e['time'] >= t0])
        self.assertFalse([e for e in d.kinds('reid-rejected') if e['track'] == 5 and 'score 0%' in e['message']], 'nobody ruled out as seen together')
        reid = [e for e in d.kinds('reid') if e['track'] == 5 and e['time'] >= t0]
        self.assertTrue(reid and reid[0]['playerId'] == ids[4], reid)
        self.assertGreaterEqual(reid[0]['scores']['spatial'], .5, reid[0]['scores'])
        # The track was re-born at the change: its birth and first position are those of the first other-team vote.
        tr = d.ids.tracks[5]
        self.assertGreaterEqual(tr.born, t0-1e-6)
        self.assertLess(tr.born, changes[0]['time'])
        self.assertEqual(len(tr.switches), 1)
        self.assertEqual(tr.cut_reasons[tr.switches[0]], 'kit')

    def test_labels_since_the_change_are_withdrawn_from_the_released_identity(self):
        d, ids, n, t0 = self.scene()
        change = next(e for e in d.kinds('team-change') if e['track'] == 5)
        g = d.ids.registry[ids[5]]
        self.assertEqual(g.status, 'missing')
        self.assertLess(g.last_seen, t0, 'the released identity was last seen before the change')
        self.assertFalse([p for p in g.emitted_all if p['time'] >= t0-1e-6], 'no observation of the other-team player stays with it')
        frames = track_frames(d, 5)
        after = [p for p in frames if p['time'] >= t0-1e-6]
        self.assertFalse([p for p in after if p.get('id') == ids[5]], [(p['time'], p.get('display')) for p in after if p.get('id') == ids[5]][:5])
        # Once re-identified, the withdrawn frames take the person the track showed (labelled retroactively).
        between = [p for p in after if p['time'] <= change['time']+1e-6]
        self.assertTrue(between and all(p.get('id') == ids[4] for p in between), [(p['time'], p.get('id'), p.get('display')) for p in between][:6])
        self.assertTrue(all(p.get('evidence') == 'reidentified' for p in between))
        self.assertTrue(all(p['id'] == ids[5] for p in frames if p['time'] < t0-1e-6), 'labels before the change are kept')

    def test_over_the_cap_a_new_identity_waits_five_seconds_from_the_change(self):
        # The team B player missing is 50 m away: nobody of team B can be the person on track 5 now. Team B is at its cap,
        # so a new identity needs 5 s of observation since the change, not since the track's first frame 20 s earlier.
        d, ids, n, t0 = self.scene(b4_at=(.92, .6), swap_look=Sized('B').unique)
        created = [e for e in d.kinds('new-identity') if e['track'] == 5 and e['time'] >= t0]
        self.assertTrue(created, [e['message'][:140] for e in d.events if e['track'] == 5])
        born = d.ids.tracks[5].born
        self.assertGreaterEqual(born, t0-1e-6)
        self.assertGreaterEqual(created[0]['time'], born+d.ids.o.over_cap_seconds-1e-6, (born, created[0]['time']))
        self.assertTrue([e for e in d.kinds('sanity') if e['track'] == 5 and 'stays unidentified' in e['message']])
        # The run's metrics count that new identity as the outcome of the kit change, however late it came.
        frames = [{'time': round(k*.2, 3), 'people': list(f)} for k, f in enumerate(d.people)]
        report = tracklets.refine(d.ids, frames, [(np.eye(3), True, False)]*len(frames), 1280, 720)
        m = tracklets.metrics(d.ids, frames, d.events+report['events'], report)
        self.assertEqual((m['teamChanges'], m['newIdentitiesAfterTeamChange']), (1, 1))

    def test_a_stray_other_team_vote_before_the_hand_over_does_not_move_the_change_back(self):
        # One crop of the team A player reads team B one, two or three ticks before the hand-over (its own crops in
        # between). The change is where the votes turn, not that stray vote: the re-born track is not "seen together"
        # with the team B player whose track ended at the hand-over.
        for stray in (1, 2, 3):
            with self.subTest(stray=stray):
                d, ids, n, t0 = self.scene(stray=stray)
                tr = d.ids.tracks[5]
                self.assertGreaterEqual(tr.born, t0-.2-1e-6)
                self.assertEqual(tr.player_id, ids[4], [e['message'][:140] for e in d.events if e['track'] == 5 and e['time'] >= t0])
                self.assertEqual(len(d.ids.registry), n)
                self.assertFalse([e for e in d.kinds('reid-rejected') if e['track'] == 5 and 'score 0%' in e['message']])

    def test_the_returning_player_is_labelled_from_their_new_track_s_first_frame(self):
        # The team A player comes back on track 7 one second after the hand-over, while track 5 still carries their
        # identity (its kit votes take ~2 s to turn). Once the kit change releases the identity, it remembers the
        # player as they were before the change (not the team B player walking away on track 5), and track 7 takes it
        # with every frame since it began: none were labelled with that identity on track 5 any more.
        d, ids, n, t0 = self.scene(walk=.05, back_at=1.0)
        change = next(e for e in d.kinds('team-change') if e['track'] == 5)
        self.assertEqual(d.ids.tracks[7].player_id, ids[5], [e['message'][:140] for e in d.events if e['track'] == 7])
        frames = track_frames(d, 7)
        early = [p for p in frames if p['time'] < change['time']-1e-6]
        self.assertTrue(early and all(p['id'] == ids[5] for p in early), [(p['time'], p['id']) for p in early])
        self.assertEqual(len(d.ids.registry), n)

    def test_the_offline_pass_cuts_the_track_at_the_kit_change(self):
        d, ids, n, t0 = self.scene()
        frames = [{'time': round(k*.2, 3), 'people': list(f)} for k, f in enumerate(d.people)]
        report = tracklets.refine(d.ids, frames, [(np.eye(3), True, False)]*len(frames), 1280, 720)
        split = [e['message'] for e in report['events'] if e['message'].startswith('TRACKLET SPLIT') and e['track'] == 5]
        self.assertTrue(split and 'kit change: its kit votes turned to the other team' in split[0], split)
        m = tracklets.metrics(d.ids, frames, d.events+report['events'], report)
        self.assertEqual((m['teamChanges'], m['newIdentitiesAfterTeamChange']), (1, 0))

    def test_onset_is_never_more_than_release_back_before_the_change(self):
        m = IdentityManager()
        tr = TrackState(7, 0.0, 0, Stab(.5, .5, .1), (.5, .5), 'inside', '', box_at(.5, .5), Stab(.5, .5, .1), 0)
        tr.state, tr.team, tr.role, tr.hits = 'uncertain', 'A', 'player', 80
        for k in range(30):
            tr.trail.append(Trail(k*.2, 0, Stab(.5+.001*k, .5, .1), (.5+.001*k, .5), (.005, 0.0), (.001, 0.0), box_at(.5+.001*k, .5)))
        # 12 B votes spread over 6 s (the window fills slowly when crops are contaminated), then 3 more.
        for k, when in enumerate([0.0, .5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 5.6, 5.8, 6.0]):
            tr.team_votes.append(('B' if k < 12 else '-', when, .8, .1))
        m.tracks[7] = tr
        m.time = 6.0
        out = {'events': [], 'issues': []}
        m._check_team(tr, {'time': 6.0, 'segment': 0}, out)
        self.assertEqual((tr.team, tr.state), ('B', 'uncertain'))
        self.assertAlmostEqual(tr.switches[0], 6.0-RELEASE_BACK, places=3)
        self.assertAlmostEqual(tr.born, 3.0, places=3)
        self.assertAlmostEqual(tr.first_pitch[0], .5+.001*15, places=6)

    def test_the_change_is_where_the_votes_turn(self):
        onset = IdentityManager._kit_onset
        votes = lambda labels: [(x, .2*k, .8, .1) for k, x in enumerate(labels)]
        self.assertAlmostEqual(onset(votes('ABAABBBBBBBBBBB'), 'A', 'B'), .8)   # a stray B before the change: where the B run begins
        self.assertAlmostEqual(onset(votes('AABBBBABBBBBBBB'), 'A', 'B'), .4)   # a stray A after it does not move it forward
        self.assertAlmostEqual(onset(votes('BABBBBBBBBBBBBB'), 'A', 'B'), .4)   # a tie: the later split (never back past who left)
        self.assertAlmostEqual(onset(votes('A-xBBBBBBBBBBBB'), 'A', 'B'), .6)   # votes for neither team say nothing


class ReleasedIdentityMemoryTests(unittest.TestCase):
    """The identity a kit change releases forgets the person the track showed since the change: where they stood and
    how fast they moved (from the track's last clean tick before it), the gallery samples and the history taken since,
    and its last observation, so its return is reached from the right place and labelled from its first frame."""

    def manager(self, trail_from=0.0):
        """A confirmed team A identity on track 7: the player stood at x .30 until 6.8 s; from 7.0 s the track showed a
        team B player walking right (15 B votes), whose position, speed and crops the identity took until 9.8 s."""
        m = IdentityManager()
        tr = TrackState(7, 0.0, 0, Stab(.3, .5, .1), (.3, .5), 'inside', '', box_at(.3, .5), Stab(.3, .5, .1), 0)
        tr.state, tr.team, tr.role, tr.player_id, tr.hits = 'confirmed', 'A', 'player', 'A-01', 50
        g = GlobalPlayer('A-01', 'player', 'A', status='active', track=7, last_seen=9.8)
        for k in range(50):
            t = round(k*.2, 3)
            x = .3 if t < 7 else .3+.05*(t-7)
            if t >= trail_from-1e-6:
                tr.trail.append(Trail(t, 0, Stab(x, .5, .1), (x, .5), (0.0 if t < 7 else .05, 0.0), (0.0, 0.0) if t < 7 else (.03, 0.0), box_at(x, .5)))
            if t >= 7:
                tr.team_votes.append(('B', t, .8, .1))
        g.stab, g.velocity, g.last_pitch, g.pitch_velocity, g.last_box = Stab(.44, .5, .1), (.05, 0.0), (.44, .5), (.03, 0.0), box_at(.44, .5)
        g.clean_seen = g.last_observed = 9.8
        d = Person('A').descriptor()
        g.gallery = [Sample(d, 2.0), Sample(d, 8.0)]
        g.history.extend({'t': t, 'box': box_at(.3, .5)} for t in (5.0, 8.0, 9.8))
        m.registry['A-01'], m.tracks[7] = g, tr
        m.time = 10.0
        m._check_team(tr, {'time': 10.0, 'segment': 0}, {'events': [], 'issues': []})
        return g

    def test_the_memory_goes_back_to_the_last_clean_tick_before_the_change(self):
        g = self.manager()
        self.assertEqual((g.track, g.status), (None, 'missing'))
        self.assertAlmostEqual(g.stab.x, .3)
        self.assertEqual((g.velocity, g.last_pitch, g.pitch_velocity), ((0.0, 0.0), (.3, .5), (0.0, 0.0)))
        self.assertEqual(g.last_box, box_at(.3, .5))
        self.assertAlmostEqual(g.clean_seen, 6.8)
        self.assertLess(g.last_observed, 7.0)
        self.assertLess(g.last_seen, 7.0)
        self.assertEqual([s.time for s in g.gallery], [2.0])
        self.assertEqual([h['t'] for h in g.history], [5.0])

    def test_without_a_clean_tick_before_the_change_no_speed_is_kept(self):
        g = self.manager(trail_from=7.0)
        self.assertAlmostEqual(g.stab.x, .44)   # the best position left
        self.assertEqual((g.velocity, g.pitch_velocity), ((0.0, 0.0), None))
        self.assertLess(g.clean_seen, 7.0)
        self.assertLess(g.last_observed, 7.0)


class AnonymousKitChangeTests(unittest.TestCase):
    """A kit change on a track that carries no identity: nothing is protected by holding its old team, so it is released
    in small scenes too; the observations since the change lose the old team and take the identity found for them."""

    def test_small_votes_release_a_track_without_an_identity_in_a_small_scene(self):
        m = IdentityManager()
        tr = TrackState(7, 0.0, 0, Stab(.5, .5, .1), None, 'inside', '', box_at(.5, .5), Stab(.5, .5, .1), 0)
        tr.state, tr.team, tr.role, tr.hits = 'uncertain', 'A', 'player', 80
        tr.team_heights.extend([.1]*20)
        for k in range(VOTES):
            tr.team_votes.append(('B', 7.0+.2*k, .25, .04))   # small, poor crops
        m.tracks[7] = tr
        m.time = 10.0
        out = {'events': [], 'issues': []}
        m._check_team(tr, {'time': 10.0, 'segment': 0, 'small_scene': True}, out)
        self.assertEqual((tr.team, tr.state), ('B', 'uncertain'))
        self.assertEqual([e['kind'] for e in out['events']], ['team-change'])

    def test_pending_observations_since_the_change_take_the_identity_found_for_them(self):
        # Team A is full and in view, so the fifth team A person on track 5 stays anonymous (A-?). At 10 s the local
        # tracker hands track 5 to the team B player whose own track ended there: the kit change re-births track 5,
        # it is re-identified as that player, and its frames since the hand-over are labelled with them.
        d = AmateurDriver(Options(max_per_team=4))
        bs = {i+1: t for i, t in enumerate(roster('B', 3, x0=.1))}
        As = {11+i: t for i, t in enumerate(roster('A', 4, x0=.65))}
        b4, a5 = Sized('B'), Sized('A')
        before = {**bs, **As, 4: (b4, .42, .6), 5: (a5, .45, .6)}
        after = {**bs, **As, 5: (Sized('B', quality=.25, unique=b4.unique), .45, .6)}
        d.run(2, {k: v for k, v in before.items() if k != 5})   # team A's four identities first
        d.run(8, before)
        b4_id = d.ids.tracks[4].player_id
        self.assertTrue(b4_id)
        self.assertIsNone(d.ids.tracks[5].player_id, d.ids.tracks[5].reason)
        t0 = d.t
        d.step(after, ended=[4])
        d.run(6, after)
        change = next(e for e in d.kinds('team-change') if e['track'] == 5)
        self.assertIsNone(change['playerId'])
        self.assertEqual(d.ids.tracks[5].player_id, b4_id)
        frames = track_frames(d, 5)
        since = [p for p in frames if t0-1e-6 <= p['time'] <= change['time']+1e-6]
        self.assertTrue(since and all(p['id'] == b4_id and p['evidence'] == 'reidentified' for p in since),
                        [(p['time'], p['id'], p.get('reason')) for p in since][:5])
        self.assertTrue(all(p['id'] is None for p in frames if p['time'] < t0-1e-6))
        self.assertFalse([e for e in d.kinds('new-identity') if e['time'] >= t0])


class TeamChangeMetricTests(unittest.TestCase):
    """newIdentitiesAfterTeamChange counts the kit changes whose outcome was a new identity: the first identity decision
    on that track after the change, however late, unless the track switched person or changed kit again first."""

    def count(self, kinds):
        events = [{'time': t, 'kind': kind, 'track': track, 'playerId': 'B-05' if kind != 'team-change' else None, 'message': ''}
                  for t, kind, track in kinds]
        return tracklets.metrics(IdentityManager(), [], events, {})['newIdentitiesAfterTeamChange']

    def test_outcomes(self):
        self.assertEqual(self.count([(10.0, 'team-change', 5), (14.6, 'new-identity', 5)]), 1)   # over the cap: 4.6 s later
        self.assertEqual(self.count([(10.0, 'team-change', 5), (10.4, 'reid', 5), (30.0, 'new-identity', 5)]), 0)
        self.assertEqual(self.count([(10.0, 'team-change', 5), (11.0, 'switch', 5), (12.0, 'new-identity', 5)]), 0)
        self.assertEqual(self.count([(10.0, 'team-change', 5), (11.0, 'new-identity', 6)]), 0)
        self.assertEqual(self.count([(10.0, 'team-change', 5), (12.0, 'team-change', 5), (13.0, 'new-identity', 5)]), 1)


class KitDriftTests(unittest.TestCase):
    """Track 12 of amateur 3: one person, A-04, shrinking from 40 to 15 px with no overlap; the kit reading drifted
    ('looks A .43 -> .09, B .21 -> .33') and 12 of 15 votes released him to team B. In a small scene, small crops alone
    never release an identity; larger crops, or crops that also look like someone else, still do."""

    def drift(self, small):
        d = AmateurDriver(small_scene=small)
        team = {i+1: t for i, t in enumerate(roster('A', 3, x0=.1)+roster('B', 3, x0=.6))}
        player = Sized('A')
        far = Sized('B', quality=.25, unique=player.unique, look='A')   # the same person, small: the kit reads as team B
        visible = {**team, 9: (player, .45, .6)}
        d.run(8, visible)
        pid = d.ids.tracks[9].player_id
        self.assertTrue(pid and pid.startswith('A-'))
        small_box = {9: box_at(.45, .6, h=.05)}
        d.run(6, {**team, 9: (far, .45, .6)}, boxes=small_box)
        return d, pid

    def test_votes_from_small_crops_do_not_release_the_identity_in_a_small_scene(self):
        d, pid = self.drift(small=True)
        self.assertEqual(d.ids.tracks[9].player_id, pid, d.ids.tracks[9].reason)
        self.assertFalse(d.kinds('team-change'))
        held = d.kinds('team-hold')
        self.assertTrue(held and held[0]['message'].startswith('TEAM CHANGE IGNORED: small crops (drift)'), held)
        self.assertIn('0 from crops of quality >= 0.35', held[0]['message'])
        self.assertEqual(len(d.ids.registry), 7)

    def test_the_same_votes_release_the_identity_outside_small_scenes(self):
        d, pid = self.drift(small=False)   # broadcast: unchanged
        self.assertTrue([e for e in d.kinds('team-change') if e['track'] == 9])
        self.assertFalse(d.kinds('team-hold'))
        self.assertNotEqual(d.ids.tracks[9].player_id, pid)

    def manager(self, quality, height, crops=None):
        """A confirmed team A identity on track 7 established at box height .1, whose last 15 votes name team B."""
        m = IdentityManager()
        person = Person('A')
        tr = TrackState(7, 0.0, 0, Stab(.5, .5, .1), None, 'inside', '', box_at(.5, .5), Stab(.5, .5, .1), 0)
        tr.state, tr.team, tr.role, tr.player_id, tr.hits = 'confirmed', 'A', 'player', 'A-01', 80
        tr.first = [person.descriptor(), person.descriptor()]
        from soccer.identity import GlobalPlayer
        g = GlobalPlayer('A-01', 'player', 'A', status='active', track=7, last_seen=9.8)
        g.kit_heights.extend([.1]*20)
        m.registry['A-01'], m.tracks[7] = g, tr
        for k in range(VOTES):
            tr.team_votes.append(('B', 7.0+.2*k, quality, height))
        for k, d in enumerate(crops or []):
            tr.descs.append((d, 7.0+.2*k, False))
        m.time = 10.0
        out = {'events': [], 'issues': []}
        m._check_team(tr, {'time': 10.0, 'segment': 0, 'small_scene': True}, out)
        return tr, out

    def test_votes_from_crops_the_size_the_team_was_established_at_release_in_a_small_scene(self):
        tr, out = self.manager(.6, .09)
        self.assertEqual((tr.team, tr.player_id), ('B', None))
        self.assertEqual([e['kind'] for e in out['events']], ['team-change'])

    def test_small_votes_release_when_the_crops_also_show_someone_else(self):
        other = Person('B')
        tr, out = self.manager(.25, .04, crops=[other.descriptor(.4) for _ in range(3)])
        self.assertEqual((tr.team, tr.player_id), ('B', None), [e['message'] for e in out['events']])

    def test_small_votes_with_crops_of_the_same_person_are_held(self):
        tr, out = self.manager(.6, .04)   # good descriptors, but from boxes far smaller than the team was established at
        self.assertEqual((tr.team, tr.player_id), ('A', 'A-01'))
        self.assertEqual([e['kind'] for e in out['events']], ['team-hold'])


class KitConsistentGalleryTests(unittest.TestCase):
    """A09 of amateur 3 learned the crops of a team B player (looks A .44 -> .12, B .01 -> .48) because OSNet's cosine
    admits the other team at these sizes. In a small scene a crop whose own kit vote names the other team never joins a
    player identity's gallery."""

    def scene(self, small, kit='B'):
        d = AmateurDriver(small_scene=small)
        team = {i+1: t for i, t in enumerate(roster('A', 2, x0=.1)+roster('B', 2, x0=.6))}
        player = Person('A')
        d.run(6, {**team, 9: (player, .45, .6)})
        g = d.ids.registry[d.ids.tracks[9].player_id]
        gallery = list(g.gallery)
        other_kit = Person(kit, unique=player.unique, look='A', cls='player')   # the same OSNet look, another kit
        d.t += 1.0   # well past the gallery spacing
        out = d.step({**team, 9: (other_kit, .45, .6)})
        return out[9], g, gallery

    def test_a_crop_voting_the_other_team_is_not_stored_in_a_small_scene(self):
        person, g, gallery = self.scene(small=True)
        self.assertEqual(person['memory'], 'frozen: kit votes team B')
        self.assertEqual([s.time for s in g.gallery], [s.time for s in gallery])

    def test_a_crop_matching_neither_team_is_not_stored_in_a_small_scene(self):
        person, g, gallery = self.scene(small=True, kit='REF')   # the referee's kit: an outlier vote ('x')
        self.assertEqual(person['memory'], 'frozen: kit matches neither team')
        self.assertEqual([s.time for s in g.gallery], [s.time for s in gallery])

    def test_outside_small_scenes_the_gallery_rule_is_unchanged(self):
        person, g, gallery = self.scene(small=False)
        self.assertEqual(person['memory'], 'updated')


class ProvisionalPitchPromotionTests(unittest.TestCase):
    """Without a usable pitch under a provisional (low, not automatic) registration, a track born on the touchline waited
    20 observations before anyone compared it with the missing players. In a small scene it is compared at 10; a new
    identity still waits for the 20."""

    def run_scene(self, small, recover_at=None):
        """(observations at promotion, observations at the new identity) of a touchline-born newcomer; recover_at: the
        pitch is usable again from the newcomer's that-many-th observation."""
        d = AmateurDriver(pitch_provisional=True, small_scene=small)
        d.pitch_reliable = False
        team = {i+1: t for i, t in enumerate(roster('A', 3, x0=.1)+roster('B', 3, x0=.6))}
        d.run(4, team, zone='unknown')
        newcomer = {**team, 9: (Person('A'), .45, .9)}
        promoted = created = None
        for k in range(30):
            # Born on the touchline, measured inside the pitch for three ticks, then the registration fails.
            zone = 'boundary' if k == 0 else 'inside' if k < 4 else 'unknown'
            if recover_at is not None and k+1 >= recover_at:
                d.pitch_reliable = True
            out = d.step(newcomer, zone=lambda tid: zone if tid == 9 else 'unknown')
            tr = d.ids.tracks[9]
            if promoted is None and out[9]['state'] in ('uncertain', 'confirmed'):
                promoted = tr.hits
            if created is None and out[9]['id']:
                created = tr.hits
        return promoted, created

    def test_reidentification_starts_at_the_undoubled_count_in_a_small_scene(self):
        promoted, created = self.run_scene(small=True)
        self.assertEqual(promoted, 10)
        self.assertEqual(created, 20)

    def test_outside_small_scenes_promotion_waits_for_the_doubled_count(self):
        promoted, created = self.run_scene(small=False)
        self.assertEqual(promoted, 20)
        self.assertEqual(created, 20)

    def test_the_wait_for_a_new_identity_ends_when_the_pitch_is_usable_again(self):
        # Promoted early (no usable pitch), then the registration recovers at the 12th observation: the doubled wait
        # belonged to the frames without a pitch, so the new identity is made at once.
        promoted, created = self.run_scene(small=True, recover_at=12)
        self.assertEqual(promoted, 10)
        self.assertEqual(created, 12)


class SwitchMemoryTests(unittest.TestCase):
    """A TRACK SWITCH releases the identity the track carried. In a small scene the identity also forgets the person the
    track showed since the switch (as after a kit change); elsewhere (broadcast) its memory is left as it always was."""

    def scene(self, small):
        d = AmateurDriver(small_scene=small)
        team = {i+1: t for i, t in enumerate(roster('A', 3, x0=.1)+roster('B', 3, x0=.6))}
        player, other = Person('B'), Person('B', look='A')   # a teammate who looks nothing like the player
        d.run(6, {**team, 9: (player, .40, .5)})
        pid = d.ids.tracks[9].player_id
        swap = d.t
        for k in range(6):
            d.step({**team, 9: (other, .40+.01*(k+1), .5)})   # the track now shows the teammate, walking right
        switch = [e for e in d.kinds('switch') if e['track'] == 9]
        self.assertTrue(switch and switch[0]['playerId'] == pid, [e['message'][:80] for e in d.events if e['track'] == 9])
        return d.ids.registry[pid], swap

    def test_the_released_identity_forgets_the_other_person_in_a_small_scene(self):
        g, swap = self.scene(small=True)
        self.assertAlmostEqual(g.stab.x, .40)
        self.assertEqual(g.velocity, (0.0, 0.0))
        self.assertLess(g.last_observed, swap)

    def test_outside_small_scenes_the_memory_is_unchanged(self):
        g, swap = self.scene(small=False)
        self.assertGreater(g.stab.x, .40)
        self.assertGreater(g.last_observed, swap)


class VoteHistoryTests(unittest.TestCase):
    """The per-track kit votes kept for the offline small-fragment rule: only on small-scene frames (nothing grows on a
    broadcast run), only votes naming a team (the rule reads nothing else), the latest VOTE_HISTORY of a long track."""

    def votes(self, small):
        d = AmateurDriver(small_scene=small)
        team = {i+1: t for i, t in enumerate(roster('A', 3, x0=.1)+roster('B', 3, x0=.6))}
        d.run(4, {**team, 9: (Person('A'), .45, .6), 10: (Person('REF', cls='player'), .45, .3)})
        return d.ids.tracks

    def test_kept_on_small_scene_frames_only(self):
        tracks = self.votes(small=True)
        self.assertEqual({v for _, v in tracks[9].vote_history}, {'A'})
        self.assertGreaterEqual(len(tracks[9].vote_history), 15)
        self.assertFalse(tracks[10].vote_history, 'outlier votes are not kept')
        self.assertEqual(tracks[9].vote_history.maxlen, VOTE_HISTORY)
        self.assertFalse(any(tr.vote_history for tr in self.votes(small=False).values()))


if __name__ == '__main__':
    unittest.main()
