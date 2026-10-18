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


if __name__ == '__main__':
    unittest.main()
