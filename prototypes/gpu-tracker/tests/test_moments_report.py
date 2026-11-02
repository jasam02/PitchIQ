"""The key-moments report (soccer/moments/report.assemble) on synthetic scenes: its sections and their references, a
moment's coaching structure (title, players, time, what / why / assessment, suggestion, pointer, reliability, the
evidence the page draws), the level's register (level 1 carries no number, level 7 metres), the suggested area (a
reachable zone, never the player on the ball, the simplest answer at the youngest levels), the drills chosen from the
library by level, and the possession wording (an estimate, said once per moment).

Run: python -m unittest test_moments_report        (from this directory)
"""
import re
import unittest

import moments_synth as ms
from soccer.moments import engine as en
from soccer.moments import report as rp
from soccer.moments.detectors import detect_all


def scene():
    """Team A: an exposed channel behind its advanced left back (own ball, 50-90 s) and, earlier, pressure with cover on
    B's carrier; team B: nothing special."""
    s = ms.Scene(80, 'A')
    a, b = s.add_team('A'), s.add_team('B')
    s.possession('B', 10, 50, carrier=b['RCM'])
    s.place(b['RCM'], 20, 105-40, 0)
    s.place(a['LCM'], 20, 41.5, 1)
    s.place(a['RCB'], 20, 32, -2)
    s.place(a['LCM'], 34, 44, 7)
    s.place(a['RCB'], 34, 30, -8)
    s.place(b['RCM'], 34, 44, -7)
    s.ball_with(b['RCM'], 10, 50)                      # after the moves: the ball stays with its carrier
    s.possession('A', 50, 90, carrier=a['LB'])
    s.move(a['LB'], 50, 56, 55, 26)
    s.ball_with(a['LB'], 50, 90)
    s.move(b['RM'], 50, 56, 105-34, -21)
    return s, a, b


def build(level, team='A', rotation_in_reach=False):
    s, a, b = scene()
    if rotation_in_reach:
        s.place(a['LCM'], 50, 38, 16)                  # the left central midfielder close enough to rotate behind
    data = s.build()
    engine = en.Engine(data)
    eps = [rp._episode_ids(e, data) for e in detect_all(engine)]
    prof = rp._profile({'level': level, 'ageGroup': 'Adult', 'format': '11v11'})
    return rp.assemble(data, engine, eps, team, prof, {'A': 'white', 'B': 'red'}, {}, 'test'), s, a, b


class ReportTests(unittest.TestCase):
    def test_sections_and_references(self):
        r, s, a, b = build(7)
        self.assertEqual((r['kind'], r['version'], r['team'], r['teamLabel']), ('coaching-moments', 1, 'A', 'Team A (white)'))
        for key in ('summary', 'strengths', 'improvements', 'keyMoments', 'patterns', 'training', 'metrics', 'coverage',
                    'moreMoments', 'moments', 'runtime', 'counts', 'profile'):
            self.assertIn(key, r)
        for key in ('strengths', 'improvements', 'keyMoments', 'moreMoments'):
            for i in r[key]:
                self.assertIn(i, r['moments'])
        self.assertTrue(r['improvements'], r['counts'])
        self.assertTrue(r['strengths'], r['counts'])
        times = [r['moments'][i]['time'] for i in r['keyMoments']]
        self.assertEqual(times, sorted(times))

    def test_a_moment_has_the_coaching_structure_and_its_evidence(self):
        r, s, a, b = build(7)
        m = r['moments'][r['improvements'][0]]
        self.assertEqual(m['kind'], 'exposed_channel')
        for key in ('title', 'finding', 'what', 'why', 'assessment', 'pointer', 'clock', 'half', 'players', 'roles'):
            self.assertTrue(m[key], key)
        self.assertEqual(m['players']['focus'], ['A-02'])
        self.assertEqual(m['roles']['A-02'], 'left back')
        self.assertTrue(m['what'].startswith(f"At {m['clock']}"))
        ev = m['evidence']
        self.assertLess(ev['window']['t0'], ev['window']['key'])
        self.assertLess(ev['window']['key'], ev['window']['t1'])
        self.assertAlmostEqual(ev['window']['key']-ev['window']['t0'], 8.0, delta=4.01)
        types = [e['type'] for e in ev['diagram']['elements']]
        for t in ('focus', 'channel', 'caption', 'player'):
            self.assertIn(t, types)
        self.assertEqual(ev['follow']['focus'][0]['id'], 'A-02')
        self.assertEqual(ev['tier'], 'image')                # no homography in the synthetic data: boxes only on video
        self.assertIn('estimate', m['reliability']['text'])

    def test_the_suggestion_is_a_reachable_zone_and_never_the_player_on_the_ball(self):
        # level 5: a rotation into the channel (started with the run), by a player who is not on the ball, in time
        r, s, a, b = build(5)
        m = r['moments'][r['improvements'][0]]
        sug = m['suggestion']
        self.assertEqual(sug['label'], 'rotate_cover')
        self.assertNotEqual(sug['mover'], 'A-02')          # the left back has the ball
        self.assertLessEqual(sug['reachS'], sug['budgetS'])
        self.assertIn('as left back A-02 went forward', sug['text'])
        # level 2: the simplest answer, the decision before the run (hold a deeper position)
        r2, *_ = build(2)
        sug2 = r2['moments'][r2['improvements'][0]]['suggestion']
        self.assertEqual((sug2['label'], sug2['mover']), ('hold_deeper', 'A-02'))
        for x, y in sug['zone']:
            self.assertTrue(-.05 <= x <= 1.05 and -.05 <= y <= 1.05)
        self.assertIn('suggestedZone', [e['type'] for e in m['evidence']['diagram']['elements']])
        self.assertIn('Assuming everyone else stayed where they were', sug['text'])

    def test_level_one_carries_no_number_and_level_seven_metres(self):
        r1, *_ = build(1)
        r7, *_ = build(7)
        m1 = r1['moments'][r1['improvements'][0]]
        m7 = r7['moments'][r7['improvements'][0]]
        for key in ('what', 'finding'):
            body = re.sub(r'[AB]-\d+|\d\d:\d\d', '', m1[key])
            self.assertIsNone(re.search(r'\d', body), m1[key])
        self.assertRegex(m7['what'], r'\d+ m')
        self.assertNotEqual(m1['pointer'], m7['pointer'])
        self.assertEqual(r1['profile']['band'], 'development')

    def test_training_takes_a_library_drill_for_the_level(self):
        r7, *_ = build(7)
        r2, *_ = build(2)
        d7 = [t['drill'] for t in r7['training'] if t['drill']]
        d2 = [t['drill'] for t in r2['training'] if t['drill']]
        self.assertTrue(d7 and d2)
        self.assertTrue(any(7 in d['levels'] or 6 in d['levels'] for d in d7), d7)
        self.assertTrue(all(min(abs(2-v) for v in d['levels']) <= 1 for d in d2), d2)

    def test_every_person_a_moment_involves_has_its_label(self):
        # report.labels: the page shows a full match's person ids by these (its own label map holds one view's people)
        rep, s, a, b = build(7)
        ids = {i for m in rep['moments'].values() for group in m['players'].values() for i in group}
        self.assertTrue(ids)
        self.assertEqual(set(rep['labels']), ids)
        shown = {p['id']: p['display'] for p in s.people}
        self.assertEqual(rep['labels'], {i: shown[i] for i in ids})

    def test_the_other_teams_report_has_none_of_this_teams_moments(self):
        r, *_ = build(7, team='B')
        self.assertTrue(all(m['team'] == 'B' for m in r['moments'].values()))
        self.assertEqual(r['teamLabel'], 'Team B (red)')


if __name__ == '__main__':
    unittest.main()
