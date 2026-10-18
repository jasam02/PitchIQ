"""Off-screen estimation (soccer/imputation.py): role-anchored centroid voting recovers the team centre
where the visible mean is biased, nothing is estimated without support, sigma grows with the time hidden,
a gap with both ends observed meets both observations, and a never-observed player stays unknown."""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from soccer import imputation  # noqa: E402

K = imputation.KINDS


def formation(seed=0, players=10, seconds=120.0, fps=5.0):
    """A team whose centre drifts along the pitch while every player keeps a role offset plus a small
    random walk: (times, true positions (T, P, 2))."""
    rng = np.random.default_rng(seed)
    T = int(seconds*fps)
    times = np.arange(T)/fps
    offsets = np.array([(x, y) for x in (-20, -10, 0, 10, 20) for y in (-12, 12)], np.float64)[:players]
    centre = np.c_[52.5+15*np.sin(times/20), 34+5*np.cos(times/30)]
    walk = np.cumsum(rng.normal(0, .15, size=(T, players, 2)), axis=0)
    return times, centre[:, None, :]+offsets[None]+walk, centre, offsets


def window(pos, x_lo, x_hi):
    inside = (pos[:, :, 0] >= x_lo) & (pos[:, :, 0] <= x_hi)
    obs = np.where(inside[:, :, None], pos, np.nan)
    sig = np.where(inside, 1.0, np.nan)
    return obs, sig, inside


class VotingTests(unittest.TestCase):
    def test_voting_recovers_the_centre_where_the_visible_mean_is_biased(self):
        times, pos, centre, offsets = formation()
        # Learn offsets while everyone is visible, then hide the left half of the team.
        obs, sig, _ = window(pos, -1e9, 1e9)
        offsets_l, var_l, centre_l, _ = imputation.learn_offsets(obs, sig, times)
        half = pos.copy()
        half[:, pos[len(pos)//2, :, 0] < centre[len(pos)//2, 0]] = np.nan   # the players left of the centre are hidden
        sig_half = np.where(np.isfinite(half[:, :, 0]), 1.0, np.nan)
        t = len(times)-1
        visible_mean = np.nanmean(half[t], axis=0)
        self.assertGreater(abs(visible_mean[0]-centre[t, 0]), 5.0)         # the raw mean is biased towards the camera side
        voted, _ = imputation.vote_centres(half, sig_half, offsets_l, var_l)
        self.assertLess(abs(voted[t, 0]-centre[t, 0]), 1.5)                   # the voted centre is not
        self.assertLess(abs(voted[t, 1]-centre[t, 1]), 1.5)

    def test_no_centre_without_enough_voters(self):
        times, pos, centre, offsets = formation()
        obs, sig, _ = window(pos, -1e9, 1e9)
        offsets_l, var_l, _, _ = imputation.learn_offsets(obs, sig, times)
        few = pos.copy()
        few[:, 3:] = np.nan                                                  # three visible players
        voted, _ = imputation.vote_centres(few, np.where(np.isfinite(few[:, :, 0]), 1.0, np.nan), offsets_l, var_l)
        self.assertTrue(np.isnan(voted).all())


class EstimateTests(unittest.TestCase):
    def test_hidden_players_are_estimated_with_a_growing_sigma_and_a_gap_meets_both_observations(self):
        times, pos, centre, offsets = formation(seed=1)
        obs, sig, inside = window(pos, -1e9, 1e9)
        obs = obs.copy()
        sig = sig.copy()
        hide = (times >= 60) & (times <= 72)                                 # player 0 hidden for 12 s
        obs[hide, 0] = np.nan
        sig[hide, 0] = np.nan
        est, sigma, kind, age = imputation.estimate_team(obs, sig, times)
        t0 = int(np.argmax(hide))
        t1 = t0+int(hide.sum())-1
        self.assertEqual(kind[t0-1, 0], K['O'])
        self.assertTrue((kind[t0:t1+1, 0] == K['P']).all())
        errors = np.hypot(*(est[t0:t1+1, 0]-pos[t0:t1+1, 0]).T)
        self.assertLess(errors.max(), 4.0, errors)                           # the formation holds the player within metres
        self.assertLess(sigma[t0, 0], sigma[t0+30, 0])                       # the uncertainty grows while hidden
        self.assertAlmostEqual(float(np.hypot(*(est[t0, 0]-pos[t0-1, 0]))), 0.0, delta=1.0)   # meets the last observation
        self.assertAlmostEqual(float(np.hypot(*(est[t1, 0]-pos[t1+1, 0]))), 0.0, delta=1.0)   # ... and the next one
        # One-sided (online) estimates exist too, and are no better than the blended ones on average.
        est_f, _, kind_f, _ = imputation.estimate_team(obs, sig, times, bidirectional=False)
        self.assertTrue((kind_f[t0:t1+1, 0] == K['P']).all())
        online = np.hypot(*(est_f[t0:t1+1, 0]-pos[t0:t1+1, 0]).T).mean()
        self.assertLessEqual(errors.mean(), online+.5)

    def test_short_gaps_are_interpolated_and_unsupported_players_are_unknown(self):
        times, pos, centre, offsets = formation(seed=2)
        obs, sig, _ = window(pos, -1e9, 1e9)
        obs = obs.copy()
        sig = sig.copy()
        obs[10:13, 1] = np.nan                                               # 0.6 s gap
        sig[10:13, 1] = np.nan
        obs[:, 9] = np.nan                                                   # never observed
        sig[:, 9] = np.nan
        est, sigma, kind, age = imputation.estimate_team(obs, sig, times)
        self.assertTrue((kind[10:13, 1] == K['I']).all())
        self.assertTrue((kind[:, 9] == K['U']).all())
        self.assertTrue(np.isnan(est[:, 9]).all())
        # Hidden for longer than MAX_GAP from both sides: unknown.
        obs2 = obs.copy()
        sig2 = sig.copy()
        long = (times >= 20) & (times <= 100)
        obs2[long, 2] = np.nan
        sig2[long, 2] = np.nan
        est2, sigma2, kind2, age2 = imputation.estimate_team(obs2, sig2, times)
        middle = int(np.flatnonzero(long)[len(np.flatnonzero(long))//2])
        self.assertEqual(kind2[middle, 2], K['U'])
        self.assertEqual(kind2[int(np.argmax(long))+2, 2], K['P'])          # just after it vanished: predicted

    def test_the_result_timeline_block(self):
        frames = []
        for i in range(40):
            t = round(i*.2, 3)
            people = [{'id': 'A-01', 'team': 'A', 'pitch': [.5+.001*i, .5], 'pitchConfidence': .8, 'pitchSigma': [1.0, 1.0]}]
            if i < 20:
                people.append({'id': 'A-02', 'team': 'A', 'pitch': [.6, .4], 'pitchConfidence': .8})
            frames.append({'time': t, 'calibrated': i != 10, 'people': people if i != 10 else []})   # frame 10: registration lost
        result = {'frames': frames, 'roster': {'slots': [{'id': 'A-01', 'team': 'A', 'status': 'ACTIVE'}, {'id': 'A-02', 'team': 'A', 'status': 'UNKNOWN'},
                                                         {'id': 'B-01', 'team': 'B', 'status': 'UNKNOWN'}, {'id': 'A-09', 'team': 'A', 'status': 'RETIRED'}]}}
        block = imputation.estimate(result)
        self.assertEqual(len(block['frameTimes']), 40)
        self.assertEqual(set(block['ids']), {'A-01', 'A-02', 'B-01'})
        self.assertEqual(block['ids']['A-01']['kind'][5], 'O')
        self.assertEqual((block['ids']['A-01']['kind'][10], block['ids']['A-01']['x'][10]), ('U', None))   # nobody is placed on an uncalibrated frame
        self.assertEqual(block['ids']['A-01']['kind'][11], 'O')
        self.assertEqual(block['ids']['A-02']['kind'][25], 'P')              # 1.2 s after it vanished: motion carries it
        self.assertIsNotNone(block['ids']['A-02']['x'][25])
        self.assertTrue(all(k == 'U' for k in block['ids']['B-01']['kind']))
        self.assertTrue(all(x is None for x in block['ids']['B-01']['x']))
        self.assertAlmostEqual(sum(block['kinds'].values()), 1.0, places=2)

    def test_a_precise_far_position_on_a_small_scene_frame_stays_an_observation(self):
        """pitchConfidence under .2 only because a few pixels are worth many metres at the far side: on small-scene frames
        with a registration of at least MEDIUM and a sigma of at most 6 m it is still observed; never on other frames."""
        def run(small, sigma=(2.5, 4.0), frame_confidence=.6):
            frames = []
            for i in range(30):
                people = [{'id': 'A-01', 'team': 'A', 'pitch': [.5, .1], 'pitchConfidence': .15, 'pitchSigma': None if sigma is None else list(sigma)},
                          {'id': 'A-02', 'team': 'A', 'pitch': [.5, .6], 'pitchConfidence': .8, 'pitchSigma': [.5, .5]}]
                f = {'time': round(i*.2, 3), 'calibrated': True, 'people': people, 'calibration': {'confidence': frame_confidence}}
                if small is not None:
                    f['profile'] = {'smallScene': small}
                frames.append(f)
            result = {'frames': frames, 'roster': {'slots': [{'id': 'A-01', 'team': 'A', 'status': 'ACTIVE'}, {'id': 'A-02', 'team': 'A', 'status': 'ACTIVE'}]}}
            return imputation.estimate(result)['ids']
        ids = run(True)
        self.assertTrue(all(k == 'O' for k in ids['A-01']['kind']))
        self.assertEqual((ids['A-01']['x'][3], ids['A-01']['y'][3], ids['A-01']['sigma'][3]), (.5, .1, [4.0, 4.0]))
        self.assertTrue(all(k == 'O' for k in run(True, frame_confidence=imputation.OBS_MIN_FRAME)['A-01']['kind']))
        # Not on other frames; not beyond 6 m; not without a sigma; not on a frame whose registration is calibrated (it
        # wrote positions: >= .2) but LOW, where the registration's own decay, which the sigma leaves out, is part of the doubt.
        for kwargs in ({'small': False}, {'small': None}, {'small': True, 'sigma': (2.5, 8.0)}, {'small': True, 'sigma': None},
                       {'small': True, 'frame_confidence': .3}):
            ids = run(**kwargs)
            self.assertNotIn('O', ids['A-01']['kind'], kwargs)        # dropped as before: estimated or unknown
            self.assertTrue(all(k == 'O' for k in ids['A-02']['kind']))

    def test_the_small_scene_frame_floor_is_the_medium_registration_level(self):
        from soccer import pitch_model, registration
        self.assertEqual(imputation.OBS_MIN_FRAME, pitch_model.MEDIUM)
        self.assertGreater(imputation.OBS_MIN_FRAME, registration.MIN_POSITION)    # a floor every positioned frame passes tests nothing


def override_result():
    """60 frames (0.2 s) of four team A players: A-01 observed throughout but under the confidence gate on frames
    20-22, A-02 hidden on frames 30-44, A-03 on a small-scene frame range with a precise far position, A-04
    unconfirmed with a gap; frame 12 without a registration."""
    frames = []
    for i in range(60):
        t = round(i*.2, 3)
        people = [{'id': 'A-01', 'team': 'A', 'pitch': [.4+.002*i, .5], 'pitchConfidence': .1 if 20 <= i < 23 else .8, 'pitchSigma': [1.2, .6]},
                  {'id': 'A-03', 'team': 'A', 'pitch': [.6, .1+.001*i], 'pitchConfidence': .15, 'pitchSigma': [2.5, 4.0]},
                  {'id': 'A-05', 'team': 'A', 'pitch': [.2, .2], 'pitchConfidence': .9}]   # not a roster slot
        if not 30 <= i < 45:
            people.append({'id': 'A-02', 'team': 'A', 'pitch': [.55, .3-.001*i], 'pitchConfidence': .7})
        if not 8 <= i < 11:
            people.append({'id': 'A-04', 'team': 'A', 'pitch': [.7, .7], 'pitchConfidence': .9, 'pitchSigma': [.8, .8]})
        frames.append({'time': t, 'calibrated': i != 12, 'people': people, 'calibration': {'confidence': .6}, 'profile': {'smallScene': i >= 40}})
    slots = [{'id': f'A-0{k}', 'team': 'A', 'status': 'ACTIVE'} for k in (1, 2, 3)]+[{'id': 'A-04', 'team': 'A', 'status': 'ACTIVE', 'confirmed': False}]
    return {'frames': frames, 'roster': {'slots': slots}}


def raw_observations(result, length=105.0, width=68.0, min_confidence=.2):
    """What estimate() reads of the people, as an override: per slot the positions in metres and the pitchSigma."""
    T = len(result['frames'])
    obs = {s['id']: ([None]*T, [None]*T, [None]*T) for s in result['roster']['slots']}
    for t, f in enumerate(result['frames']):
        for p in f['people']:
            if p['id'] not in obs or not f['calibrated']:
                continue
            if p['pitchConfidence'] < min_confidence and not imputation._precise_on_small_scene(p, f):
                continue
            xs, ys, sigmas = obs[p['id']]
            xs[t], ys[t], sigmas[t] = p['pitch'][0]*length, p['pitch'][1]*width, p.get('pitchSigma')
    return obs


class ObservationOverrideTests(unittest.TestCase):
    """estimate(observations=...): the 2D replay's stabilized positions in place of the people's (soccer/stabilized.py), and
    observation(), the rule estimate() reads the people with (soccer/stabilized.py reads the raw observations through it)."""

    def test_without_an_override_the_timeline_is_the_same(self):
        result = override_result()
        default = imputation.estimate(result)
        self.assertEqual(imputation.estimate(result, observations=None), default)
        self.assertEqual(imputation.estimate(result, observations={}), default)
        self.assertEqual(imputation.estimate(result, observations={'B-09': ([1.0]*60, [1.0]*60, [None]*60)}), default)   # not a slot: ignored
        # The override handed exactly what the default path reads gives the same block, number for number.
        self.assertEqual(imputation.estimate(result, observations=raw_observations(result)), default)
        self.assertEqual(default['ids']['A-01']['kind'][22], 'I')          # under the gate: interpolated
        self.assertEqual(default['ids']['A-03']['kind'][45], 'O')          # precise on a small-scene frame
        self.assertEqual(default['ids']['A-04']['kind'][9], 'U')           # unconfirmed: no estimate in its gap

    def test_an_override_replaces_the_positions_of_its_slots_only(self):
        result = override_result()
        default = imputation.estimate(result)
        obs = raw_observations(result)
        xs, ys, sigmas = obs['A-01']
        moved = ([None if x is None else x+2.0 for x in xs], ys, [[.4, .3] if s is not None else None for s in sigmas])
        for t in range(20, 23):                                             # the caller decides: under the gate, still observed
            moved[0][t], moved[1][t] = (.4+.002*t)*105.0+2.0, .5*68.0
        moved[0][12], moved[1][12] = 50.0, 34.0                             # a frame without a registration places nobody
        moved[0][40], moved[1][40] = None, None                             # not observed here: estimated as usual
        block = imputation.estimate(result, observations={'A-01': moved})
        a01 = block['ids']['A-01']
        self.assertEqual(a01['kind'][5], 'O')
        self.assertEqual((a01['x'][5], a01['y'][5], a01['sigma'][5]), (round(((.4+.01)*105.0+2.0)/105.0, 4), .5, [.4, .4]))
        self.assertEqual([a01['kind'][t] for t in range(20, 23)], ['O']*3)
        self.assertEqual((a01['kind'][12], a01['x'][12]), ('U', None))
        self.assertEqual(a01['kind'][40], 'I')
        self.assertEqual(a01['sigma'][41], [.4, .4])
        # Every other slot's observations are its people's.
        for pid in ('A-02', 'A-03', 'A-04'):
            for t in range(60):
                if default['ids'][pid]['kind'][t] == 'O':
                    self.assertEqual((block['ids'][pid]['kind'][t], block['ids'][pid]['x'][t], block['ids'][pid]['y'][t]),
                                     ('O', default['ids'][pid]['x'][t], default['ids'][pid]['y'][t]), (pid, t))
        # An override without sigmas takes the default observation sigma.
        bare = imputation.estimate(result, observations={'A-01': (moved[0], moved[1], None)})
        self.assertEqual(bare['ids']['A-01']['sigma'][5], [imputation.DEFAULT_SIGMA, imputation.DEFAULT_SIGMA])


    def test_observation_is_what_the_timeline_observes(self):
        result = override_result()
        default = imputation.estimate(result)
        for t, f in enumerate(result['frames']):
            for p in f['people']:
                if p['id'] not in default['ids']:
                    continue
                seen = imputation.observation(p, f)
                kind = default['ids'][p['id']]['kind'][t]
                if f['calibrated']:
                    self.assertEqual(seen is not None, kind == 'O', (p['id'], t))
                if seen is not None and kind == 'O':
                    self.assertEqual((round(seen[0]/105.0, 4), round(seen[1]/68.0, 4)), (default['ids'][p['id']]['x'][t], default['ids'][p['id']]['y'][t]))
                    self.assertEqual([round(seen[2], 1)]*2, default['ids'][p['id']]['sigma'][t])
        a01 = result['frames'][5]['people'][0]
        self.assertEqual(imputation.observation(a01, result['frames'][5], 100.0, 50.0), (a01['pitch'][0]*100.0, a01['pitch'][1]*50.0, 1.2))
        self.assertIsNone(imputation.observation(result['frames'][21]['people'][0], result['frames'][21]))          # under the gate
        self.assertIsNotNone(imputation.observation(result['frames'][45]['people'][1], result['frames'][45]))      # precise, small scene
        self.assertEqual(imputation.observation({'pitch': [.5, .5], 'pitchConfidence': .9}, {})[2], imputation.DEFAULT_SIGMA)
        self.assertIsNone(imputation.observation({'pitch': None, 'pitchConfidence': .9}, {}))


if __name__ == '__main__':
    unittest.main()
