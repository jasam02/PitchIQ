"""Player analytics (soccer/analytics.py) on synthetic trajectories: time-weighted heatmaps, average positions,
movement trails, coverage, the validity rules and the final roster (corrections) deciding who is analysed."""
import math
import unittest

import scene  # noqa: F401  (adds the prototype directory to sys.path)
from soccer import analytics, pitch_model

FPS = 30.0


def person(pid, t, x, y, team='B', role='PLAYER', state='confirmed', conf=.9, pitch_conf=.8, track=1, **extra):
    return {'time': t, 'track': track, 'box': [.5, .5, .02, .1], 'cls': 'player', 'zone': 'inside', 'evidence': 'observed', 'id': pid,
            'display': pid, 'team': team, 'role': role, 'label': f'{role}_TEAM_{team}', 'state': state, 'identityConfidence': conf,
            'roleConfidence': .8, 'pitch': [x, y], 'pitchConfidence': pitch_conf, 'pitchSource': 'carried', **extra}


def entry(pid, team='B', role='PLAYER', **extra):
    return {'id': pid, 'display': pid, 'team': team, 'role': role, 'status': 'ACTIVE', **extra}


def result(frames, players=(), goalkeepers=(), seconds=None):
    """A result with the given frames (list of (time, [people])) and roster entries."""
    out_frames = [{'time': round(t, 4), 'frame': i, 'people': people, 'calibrated': True} for i, (t, people) in enumerate(frames)]
    return {'version': 3, 'video': {'width': 1920, 'height': 1080, 'fps': FPS}, 'frames': out_frames,
            'match': {'players': list(players), 'goalkeepers': list(goalkeepers), 'referees': [], 'retired': [], 'excluded': [], 'withdrawn': []},
            'tactical': {'pitchSize': {'length': 105.0, 'width': 68.0}}}


def regular(pid, t0, t1, where, fps=FPS, **kw):
    """Frames at `fps` from t0 to t1 with pid at `where` (a point or a function of time)."""
    frames = []
    n = int(round((t1-t0)*fps))
    for k in range(n):
        t = t0+k/fps
        x, y = where(t) if callable(where) else where
        frames.append((t, [person(pid, t, x, y, **kw)]))
    return frames


def cell_seconds(hm, x, y):
    """Smoothed seconds in the cells within 7 m of a normalized point."""
    cx, cy = x*105/hm['cell'], y*68/hm['cell']
    r = 7/hm['cell']
    return sum(hm['grid'][j][i] for j in range(hm['rows']) for i in range(hm['cols']) if math.hypot(i+.5-cx, j+.5-cy) <= r)


class HeatmapTests(unittest.TestCase):
    def test_more_observed_time_means_more_density(self):
        # 6 s at A (right touchline), 2 s at B (own half): A's region holds three times the seconds.
        frames = regular('B-07', 0, 6, (.85, .9)) + regular('B-07', 6, 8, (.3, .3))
        s = analytics.Samples(result(frames, players=[entry('B-07')]))
        hm = analytics.heatmap(s, 'B-07', 0, 8)
        a, b = cell_seconds(hm, .85, .9), cell_seconds(hm, .3, .3)
        self.assertAlmostEqual(hm['seconds'], 8.0, delta=.2)
        self.assertAlmostEqual(a/b, 3.0, delta=.3, msg=(a, b))
        self.assertAlmostEqual(sum(map(sum, hm['grid'])), hm['seconds'], delta=.05, msg='smoothing keeps the total observed time')

    def test_sampling_rate_does_not_change_occupancy(self):
        # 4 s at A sampled at 30 fps and 4 s at B sampled at 6 fps: the same observed time in both regions.
        frames = regular('B-07', 0, 4, (.2, .5), fps=30) + regular('B-07', 4, 8, (.8, .5), fps=6)
        s = analytics.Samples(result(frames, players=[entry('B-07')]))
        hm = analytics.heatmap(s, 'B-07', 0, 8)
        a, b = cell_seconds(hm, .2, .5), cell_seconds(hm, .8, .5)
        self.assertAlmostEqual(a, b, delta=.35, msg=(a, b))

    def test_a_long_absence_is_not_counted_as_standing_still(self):
        # Seen 3 s at A, unseen 15 s, seen 3 s at B: about 6 s of observed time (the samples on either side of the
        # absence stand for a quarter second of it each, no more), nothing during the absence.
        frames = regular('B-07', 0, 3, (.2, .5)) + regular('B-07', 18, 21, (.8, .5))
        s = analytics.Samples(result(frames, players=[entry('B-07')]))
        hm = analytics.heatmap(s, 'B-07', 0, 21)
        self.assertAlmostEqual(hm['seconds'], 6.5, delta=.3)
        cov = analytics.coverage(s, 'B-07', 0, 21)
        self.assertAlmostEqual(cov['observedSeconds'], 6.5, delta=.3)
        self.assertAlmostEqual(cov['coverage'], 6.5/21, delta=.02)
        self.assertLess(cell_seconds(hm, .5, .5), .05, 'nothing in between')
        self.assertTrue(cov['low'])

    def test_the_time_range_selects_the_samples(self):
        frames = regular('B-07', 0, 5, (.2, .5)) + regular('B-07', 5, 10, (.8, .5))
        s = analytics.Samples(result(frames, players=[entry('B-07')]))
        hm = analytics.heatmap(s, 'B-07', 0, 5)
        self.assertGreater(cell_seconds(hm, .2, .5), 3.0)
        self.assertLess(cell_seconds(hm, .8, .5), .05)   # the one sample at exactly 5.0 s
        self.assertEqual(analytics.heatmap(s, 'B-07', 20, 30)['max'], 0.0)

    def test_invalid_and_untrusted_samples_are_excluded_not_clamped(self):
        frames = regular('B-07', 0, 2, (.5, .5))
        t = 2.0
        bad = [person('B-07', t, 1.4, .5), person('B-07', t+.1, .5, .5, pitch_conf=.3), person('B-07', t+.2, .5, .5, pitchClipped=True),
               person('B-07', t+.3, .5, .5, conf=.1), person('B-07', t+.4, .5, .5, state='uncertain')]
        frames += [(p['time'], [p]) for p in bad]
        s = analytics.Samples(result(frames, players=[entry('B-07')]))
        self.assertEqual(s.excluded, {'offPitch': 1, 'lowPitchConfidence': 1, 'clipped': 1, 'lowIdentityConfidence': 1, 'notConfirmed': 1, 'notInRoster': 0})
        self.assertEqual(len(s.tracks['B-07']['t']), 60)
        hm = analytics.heatmap(s, 'B-07', 0, 3)
        self.assertLess(hm['grid'][34][104], 1e-6, 'nothing was clamped onto the right goal line')


class AveragePositionTests(unittest.TestCase):
    def test_time_weighted_mean(self):
        # 6 s at (.2, .4), 2 s at (.6, .8): the mean is three quarters of the way to the first.
        frames = regular('B-07', 0, 6, (.2, .4)) + regular('B-07', 6, 8, (.6, .8))
        s = analytics.Samples(result(frames, players=[entry('B-07')]))
        a = analytics.average(s, 'B-07', 0, 8)
        self.assertAlmostEqual(a['position'][0], .3, delta=.01)
        self.assertAlmostEqual(a['position'][1], .5, delta=.01)
        self.assertAlmostEqual(a['coverage'], 1.0, delta=.02)
        self.assertFalse(a['low'])

    def test_two_local_tracks_of_one_identity_are_one_player(self):
        frames = regular('B-07', 0, 4, (.2, .4), track=20) + regular('B-07', 4, 8, (.6, .8), track=88)
        s = analytics.Samples(result(frames, players=[entry('B-07')]))
        a = analytics.averages(s, 0, 8, ('B',))
        self.assertEqual([x['id'] for x in a], ['B-07'])
        self.assertAlmostEqual(a[0]['position'][0], .4, delta=.01)

    def test_low_coverage_is_flagged_and_missing_time_is_not_an_observation(self):
        frames = regular('B-07', 0, 3, (.2, .4))
        s = analytics.Samples(result(frames, players=[entry('B-07')]))
        a = analytics.average(s, 'B-07', 0, 10)
        self.assertAlmostEqual(a['position'][0], .2, delta=.01)
        self.assertAlmostEqual(a['coverage'], .3, delta=.02)
        self.assertTrue(a['low'])
        self.assertIsNone(analytics.average(s, 'B-07', 20, 30)['position'])

    def test_all_players_by_team_with_the_final_roster(self):
        frames = []
        for k in range(60):
            t = k/FPS
            frames.append((t, [person('A-01', t, .3, .5, team='A'), person('B-01', t, .7, .5), person('GK-2', t, .95, .5, role='GOALKEEPER'),
                               person('B-09', t, .6, .2), person('REF-1', t, .5, .5, team=None, role='REFEREE')]))
        r = result(frames, players=[entry('A-01', 'A'), entry('B-01'), entry('B-09', confirmed=False)], goalkeepers=[entry('GK-2', 'B', 'GOALKEEPER')])
        s = analytics.Samples(r)
        self.assertEqual(sorted(s.people), ['A-01', 'B-01', 'GK-2'], 'referees and unconfirmed identities are not analysed')
        both = analytics.averages(s, 0, 2, ('A', 'B'))
        self.assertEqual([(a['id'], a['team'], a['role']) for a in both], [('A-01', 'A', 'PLAYER'), ('B-01', 'B', 'PLAYER'), ('GK-2', 'B', 'GOALKEEPER')])
        self.assertEqual([a['id'] for a in analytics.averages(s, 0, 2, ('B',))], ['B-01', 'GK-2'])
        self.assertEqual(s.excluded['notInRoster'], 120)

    def test_a_corrected_result_drops_the_referee_and_keeps_the_relabelled_player_apart(self):
        # B-07 was a referee: excluded; another identity (B-11) is now shown as B-07. Its positions are its own.
        frames = []
        for k in range(60):
            t = k/FPS
            frames.append((t, [person('B-07', t, .5, .5), person('B-11', t, .8, .3)]))
        r = result(frames, players=[entry('B-11', display='B-07')])
        r['match']['excluded'] = [entry('B-07')]
        s = analytics.Samples(r)
        self.assertEqual(list(s.people), ['B-11'])
        a = analytics.average(s, 'B-11', 0, 2)
        self.assertEqual((a['display'], a['position']), ('B-07', [.8, .3]))


class TrailTests(unittest.TestCase):
    def test_consecutive_observations_are_joined_and_long_gaps_break_the_trail(self):
        path = lambda t: (.2+.05*t, .5)
        frames = regular('B-07', 0, 4, path) + regular('B-07', 19, 23, lambda t: (.9, .2+.02*(t-19)))
        s = analytics.Samples(result(frames, players=[entry('B-07')]))
        tr = analytics.trail(s, 'B-07', 0, 23)
        self.assertEqual(len(tr['segments']), 2)
        self.assertEqual(len(tr['breaks']), 1)
        self.assertEqual(tr['breaks'][0]['why'], 'unseen')
        self.assertAlmostEqual(tr['breaks'][0]['seconds'], 15.0, delta=.1)
        self.assertAlmostEqual(tr['start']['x'], .2, delta=.01)
        self.assertAlmostEqual(tr['end']['y'], .28, delta=.01)
        self.assertEqual(len(tr['segments'][0]['t']), 120)

    def test_a_dropped_frame_does_not_break_the_trail_but_a_jump_does(self):
        frames = regular('B-07', 0, 2, lambda t: (.2+.05*t, .5+.006*(round(t*FPS) % 2)))   # half a metre of jitter a frame: not a jump
        frames = [f for f in frames if abs(f[0]-1.0) > 1e-6]   # one frame missing at 1.0 s
        frames += regular('B-07', 2, 3, lambda t: (.9, .9))     # 60 m away in one frame step: a registration glitch
        s = analytics.Samples(result(frames, players=[entry('B-07')]))
        tr = analytics.trail(s, 'B-07', 0, 3)
        self.assertEqual([b['why'] for b in tr['breaks']], ['jump'])
        self.assertEqual(len(tr['segments']), 2)

    def test_the_range_cuts_the_trail(self):
        frames = regular('B-07', 0, 10, lambda t: (.1+.08*t, .5))
        s = analytics.Samples(result(frames, players=[entry('B-07')]))
        tr = analytics.trail(s, 'B-07', 2, 4)
        self.assertEqual(len(tr['segments']), 1)
        self.assertAlmostEqual(tr['start']['t'], 2.0, delta=.04)
        self.assertAlmostEqual(tr['end']['t'], 4.0, delta=.04)
        self.assertAlmostEqual(tr['start']['x'], .26, delta=.01)

    def test_positions_keep_the_board_convention(self):
        # A point near the right touchline of the 2D view (x .9, y .95 near touchline) stays there: no axis flip.
        frames = regular('B-07', 0, 1, (.9, .95))
        s = analytics.Samples(result(frames, players=[entry('B-07')]))
        a = analytics.average(s, 'B-07', 0, 1)
        self.assertEqual(a['position'], [.9, .95])
        hm = analytics.heatmap(s, 'B-07', 0, 1)
        j, i = max(((j, i) for j in range(hm['rows']) for i in range(hm['cols'])), key=lambda ji: hm['grid'][ji[0]][ji[1]])
        self.assertTrue(i >= 90 and j >= 60, (i, j))


class RangeAndReportTests(unittest.TestCase):
    def test_ranges_are_clipped_to_the_segment_and_the_current_block_follows_the_playhead(self):
        frames = regular('B-07', 100, 160, (.5, .5))
        s = analytics.Samples(result(frames, players=[entry('B-07')]))
        self.assertEqual(analytics.clamp_range(s), (100.0, s.end))
        self.assertEqual(analytics.clamp_range(s, 0, 130), (100.0, 130.0))
        self.assertEqual(analytics.current_block(s, 120), (100.0, s.end))
        self.assertEqual(analytics.current_block(s, 620, window=300), (s.end, s.end))

    def test_report_has_everything_the_page_draws(self):
        frames = regular('B-07', 0, 4, (.3, .4)) + regular('A-02', 0, 4, (.6, .6), team='A')
        s = analytics.Samples(result(frames, players=[entry('B-07'), entry('A-02', 'A')]))
        rep = analytics.report(s, 'B-07', 0, 4, scope='both')
        self.assertEqual(rep['player']['id'], 'B-07')
        self.assertEqual(set(rep), {'player', 'range', 'rules', 'excluded', 'pitch', 'coverage', 'heatmap', 'average', 'trail', 'averages',
                                    'quality', 'averagesQuality'})
        self.assertEqual([a['id'] for a in rep['averages']], ['A-02', 'B-07'])
        self.assertEqual(rep['coverage']['samples'], 120)
        none = analytics.report(s, 'B-99', 0, 4)
        self.assertIsNone(none['player'])
        self.assertEqual(none['averages'], [])


class RegistrationLevelTests(unittest.TestCase):
    """The floor is the pitch model's MEDIUM level (a run without keyframes is capped there, as the user's amateur runs
    are), LOW stays out, and every report says how much of its observed time rests on a HIGH and on a MEDIUM
    registration."""

    def test_the_floor_is_the_pitch_models_medium_level(self):
        self.assertEqual(analytics.MIN_PITCH_CONFIDENCE, pitch_model.MEDIUM)
        self.assertEqual(analytics.HIGH_PITCH_CONFIDENCE, pitch_model.HIGH)
        frames = regular('B-07', 0, 1, (.5, .5), pitch_conf=pitch_model.MEDIUM) + regular('B-07', 1, 2, (.5, .5), pitch_conf=pitch_model.MEDIUM-.01)
        s = analytics.Samples(result(frames, players=[entry('B-07')]))
        self.assertEqual(len(s.tracks['B-07']['t']), 30, 'MEDIUM counts')
        self.assertEqual(s.excluded['lowPitchConfidence'], 30, 'just below it is LOW: excluded')

    def test_a_medium_confidence_run_gets_player_analytics_and_says_medium(self):
        # Pitch confidence .4 throughout, as on an amateur run without keyframes (p50 .39): nothing reached the old 0.5.
        r = result(regular('B-07', 0, 6, lambda t: (.2+.05*t, .5), pitch_conf=.4), players=[entry('B-07')])
        self.assertEqual(analytics.Samples(r, min_pitch_confidence=.5).tracks, {}, 'the old floor left the panel empty')
        s = analytics.Samples(r)
        rep = analytics.report(s, 'B-07', 0, 6, scope='both')
        self.assertEqual(rep['coverage']['samples'], 180)
        self.assertAlmostEqual(rep['coverage']['coverage'], 1.0, delta=.02)
        self.assertAlmostEqual(rep['heatmap']['seconds'], 6.0, delta=.1)
        self.assertAlmostEqual(rep['average']['position'][0], .35, delta=.01)
        self.assertEqual(len(rep['trail']['segments']), 1)
        q = rep['quality']
        self.assertEqual((q['level'], q['high'], q['medium'], q['low'], q['mostlyMedium']), ('MEDIUM', 0.0, 1.0, 0.0, True))
        self.assertAlmostEqual(q['mediumSeconds'], 6.0, delta=.1)
        self.assertEqual(rep['average']['quality'], q)
        self.assertEqual(rep['averagesQuality'], q, 'the average tab draws only this player')
        self.assertEqual((rep['rules']['minPitchConfidence'], rep['rules']['highPitchConfidence']), (pitch_model.MEDIUM, pitch_model.HIGH))
        self.assertEqual(rep['excluded']['lowPitchConfidence'], 0)

    def test_a_low_sample_is_excluded(self):
        # Every other frame at .3 (LOW): those are left out, the .4 ones count, and none of them is reported as LOW time.
        frames = regular('B-07', 0, 2, (.5, .5), pitch_conf=.4)
        frames = [(t, [dict(people[0], pitchConfidence=.3)]) if k % 2 else (t, people) for k, (t, people) in enumerate(frames)]
        s = analytics.Samples(result(frames, players=[entry('B-07')]))
        self.assertEqual(s.excluded['lowPitchConfidence'], 30)
        self.assertEqual(len(s.tracks['B-07']['t']), 30)
        q = analytics.quality(s, 'B-07', 0, 2)
        self.assertEqual((q['low'], q['medium'], q['level']), (0.0, 1.0, 'MEDIUM'))
        # A caller that lowers the floor gets the LOW time reported as such, never as MEDIUM.
        lowered = analytics.Samples(result(frames, players=[entry('B-07')]), min_pitch_confidence=.2)
        q = analytics.report(lowered, 'B-07', 0, 2)['quality']
        self.assertAlmostEqual(q['low'], .5, delta=.02)
        self.assertAlmostEqual(q['medium'], .5, delta=.02)
        self.assertEqual(analytics.report(lowered, 'B-07', 0, 2)['rules']['minPitchConfidence'], .2)

    def test_quality_shares_are_time_weighted(self):
        # 6 s on a HIGH registration at 30 fps, then 2 s on a MEDIUM one at 6 fps: 180 samples against 12, but the time
        # is three quarters HIGH.
        frames = regular('B-07', 0, 6, (.3, .5), pitch_conf=.8) + regular('B-07', 6, 8, (.6, .5), fps=6, pitch_conf=.4)
        s = analytics.Samples(result(frames, players=[entry('B-07')]))
        q = analytics.quality(s, 'B-07', 0, 8)
        self.assertAlmostEqual(q['high'], .75, delta=.02)
        self.assertAlmostEqual(q['medium'], .25, delta=.02)
        self.assertAlmostEqual(q['highSeconds']+q['mediumSeconds'], q['seconds'], delta=.02)
        self.assertEqual((q['level'], q['mostlyMedium']), ('HIGH', False))
        late = analytics.quality(s, 'B-07', 6.5, 8)
        self.assertEqual((late['level'], late['medium'], late['mostlyMedium']), ('MEDIUM', 1.0, True))
        self.assertEqual(analytics.quality(s, 'B-07', 20, 30), {'seconds': 0.0, 'highSeconds': 0.0, 'mediumSeconds': 0.0, 'lowSeconds': 0.0,
                                                                 'high': 0.0, 'medium': 0.0, 'low': 0.0, 'level': None, 'mostlyMedium': False})

    def test_the_average_tab_quality_pools_the_players_it_draws(self):
        frames = []
        for k in range(120):
            t = k/FPS
            frames.append((t, [person('B-07', t, .3, .4, pitch_conf=.4), person('A-02', t, .6, .6, team='A', pitch_conf=.8),
                               person('A-03', t, .7, .3, team='A', pitch_conf=.8)]))
        s = analytics.Samples(result(frames, players=[entry('B-07'), entry('A-02', 'A'), entry('A-03', 'A')]))
        rep = analytics.report(s, 'B-07', 0, 4, scope='both')
        self.assertTrue(rep['quality']['mostlyMedium'])
        pooled = rep['averagesQuality']
        self.assertAlmostEqual(pooled['medium'], 1/3, delta=.01)
        self.assertAlmostEqual(pooled['high'], 2/3, delta=.01)
        self.assertFalse(pooled['mostlyMedium'])
        self.assertEqual({a['id']: a['quality']['level'] for a in rep['averages']}, {'A-02': 'HIGH', 'A-03': 'HIGH', 'B-07': 'MEDIUM'})
        self.assertEqual(analytics.report(s, 'B-07', 0, 4, scope='B')['averagesQuality'], rep['quality'])


if __name__ == '__main__':
    unittest.main()
