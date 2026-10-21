"""The professional reference (soccer/coaching/reference.json, built by tools/coaching_bench.py from Metrica game 1):
the shipped file has the contract's schema, every quantile set is monotone and says how many samples it rests on, the
default episode rule and its baselines are present and consistent with the quantiles they were counted against, the
window distributions of the episode count (the improvement test's baseline) are well formed, say which come from both
teams and are never below a length measured directly between them, every metric with a strength has its decorrelation
time and the bound's coverage check, the distributions are those of all the time (dead ball included) with the in-play
variant kept, and reference.load_reference reads it. window_baseline reads the next longer length, window_tail is an
upper estimate, Holm's procedure and the share's upper bound behave as documented. The bench refuses to touch the
held-out game without --validate (and before reading any data), refuses to validate on the tuning game, and runs end
to end, build and validation, on synthetic Metrica-format files (never the real data here), its window counts equal a
detector's count on the window. The shipped file carries its validation on the held-out game 2 (2026-10-19, PASS),
naming the reference by its content identity, which recording a validation leaves unchanged; --record writes only
that block and refuses a foreign acceptance summary or a second record of one reference. The lateral-shift functions both the bench and the detector use measure a known lag
and drop a lag at the search bound.

Run:  python -m unittest test_coaching_reference          (from this directory)
"""
import contextlib
import datetime
import importlib.util
import io
import json
import math
import os
import subprocess
import sys
import tempfile
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (ROOT, os.path.join(ROOT, 'tools'), HERE):
    if p not in sys.path:
        sys.path.insert(0, p)
from soccer.coaching import episodes as ep, reference as rf  # noqa: E402
import team_shape_synth as syn  # noqa: E402

BENCH = os.path.join(ROOT, 'tools', 'coaching_bench.py')
SHIPPED = os.path.join(ROOT, 'soccer', 'coaching', 'reference.json')
METRICS = ('width', 'depth', 'hull', 'spread', 'gapDM', 'gapMF', 'lineStd', 'lineHeight', 'nearestTeammate', 'lagDM',
           'corrDM')
THIRDS = ('third:own', 'third:middle', 'third:final')
BACK_LINES = ('backline:3', 'backline:4', 'backline:5')
UNITS = ('unit:D', 'unit:M', 'unit:F')
UNIT_OF = {'width': 'm', 'depth': 'm', 'hull': 'm2', 'spread': 'm', 'gapDM': 'm', 'gapMF': 'm', 'lineStd': 'm',
           'lineHeight': 'm', 'nearestTeammate': 'm', 'lagDM': 's', 'corrDM': 'ratio'}


def load_bench():
    spec = importlib.util.spec_from_file_location('tools_coaching_bench', BENCH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bench = load_bench()


def strict_json(path):
    """The file parsed with NaN / Infinity refused (allow_nan=False on the way out means none may be in it)."""
    def refuse(name):
        raise ValueError(f'non-finite constant {name} in {path}')
    with open(path, encoding='utf-8') as fh:
        return json.load(fh, parse_constant=refuse)


def write_metrica_game(directory, game, seconds=120.0, fps=25, layouts=('4-4-2', '3-5-2'), seed=0,
                       dead=((30.0, 34.0), (80.0, 86.0))):
    """A synthetic match in the Metrica sample format (3 header rows; Period, Frame, Time [s], x/y per player with the
    goalkeeper first, the ball; positions 0..1): two halves of `seconds`, the sides swapped at half time, two unused
    substitutes, and the ball dead (NaN) during `dead` spells of each half."""
    T = int(seconds*fps)
    times = np.arange(2*T)/fps+1.0/fps
    period = np.r_[np.ones(T, int), np.full(T, 2)]
    ball = np.stack([.5+.3*np.sin(times/9), .5+.35*np.sin(times/6+1)], axis=1)
    for a, b in dead:
        for off in (0.0, seconds):
            ball[(times >= a+off) & (times < b+off)] = np.nan
    for name, layout, first, s in (('Home', layouts[0], 'left', seed), ('Away', layouts[1], 'right', seed+1)):
        second = 'right' if first == 'left' else 'left'
        out = np.concatenate([syn.team_track(times[:T], layout, first, s), syn.team_track(times[T:], layout, second, s+10)])
        gk = np.concatenate([syn.goalkeeper_track(times[:T], first, s), syn.goalkeeper_track(times[T:], second, s+10)])
        pos = np.concatenate([gk[:, None, :], out, np.full((2*T, 2, 2), np.nan)], axis=1)
        P = pos.shape[1]
        rows = [','.join(['', '', '']+[x for _ in range(P) for x in (name, '')]+['', '']),
                ','.join(['', '', '']+[x for j in range(P) for x in (str(j+1), '')]+['', '']),
                ','.join(['Period', 'Frame', 'Time [s]']+[x for j in range(P) for x in (f'Player{j+1}', '')]+['Ball', ''])]
        cells = np.concatenate([pos.reshape(2*T, -1), ball], axis=1)
        for k in range(2*T):
            rows.append(f'{period[k]},{k+1},{times[k]:.2f},'+','.join('NaN' if not np.isfinite(v) else f'{v:.5f}' for v in cells[k]))
        with open(os.path.join(directory, f'Sample_Game_{game}_RawTrackingData_{name}_Team.csv'), 'w', encoding='utf-8') as fh:
            fh.write('\n'.join(rows)+'\n')


def check_schema(case, ref):
    """The contract's reference schema (section 2), shared by the shipped file and a synthetic build."""
    case.assertEqual(ref['version'], 1)
    case.assertTrue(ref['source'].startswith('metrica-sample-game-'))
    case.assertEqual(ref['sampleRate'], 5.0)
    datetime.datetime.fromisoformat(ref['created'])
    case.assertIsInstance(ref['notes'], str)
    case.assertEqual(ref['rules'], {'default': {'enter': 'p95', 'exit': 'p90', 'minSeconds': 5.0, 'mergeSeconds': 3.0}})
    case.assertEqual(set(ref['metrics']), set(METRICS))
    for metric, m in ref['metrics'].items():
        case.assertEqual(m['unit'], UNIT_OF[metric], metric)
        case.assertIsInstance(m['definition'], str)
        case.assertIn('all', m['contexts'])
        for context, q in m['contexts'].items():
            case.assertIsInstance(q['n'], int)
            if q['n']:
                case.assertEqual(set(q), {'n'} | set(rf.QUANTILES), (metric, context))
        for context, rules in m['episodes'].items():
            case.assertIn(context, m['contexts'])
            case.assertEqual(set(rules), {'default'})
            b = rules['default']
            for key in ('perMinute', 'perHalf', 'durationSeconds', 'shareOfTime', 'count', 'measuredMinutes'):
                case.assertIn(key, b, (metric, context, key))
            case.assertEqual(set(b['perMinute']), {'p10', 'p50', 'p90'})
            case.assertLessEqual({'p50', 'p90'}, set(b['durationSeconds']))
    families = {'third': ('width', 'depth', 'hull', 'spread', 'gapDM', 'gapMF'), 'line': ('lineStd', 'lineHeight'),
                'unit': ('nearestTeammate',), 'window': ('lagDM', 'corrDM')}
    for metric in families['third']:
        case.assertEqual(set(ref['metrics'][metric]['contexts']), {'all', *THIRDS}, metric)
    for metric in families['line']:
        case.assertEqual(set(ref['metrics'][metric]['contexts']), {'all', *THIRDS, *BACK_LINES}, metric)
    case.assertEqual(set(ref['metrics']['nearestTeammate']['contexts']), {'all', *UNITS})
    for metric in families['window']:
        case.assertEqual(set(ref['metrics'][metric]['contexts']), {'all', 'window:20'})
        case.assertEqual(ref['metrics'][metric]['episodes'], {})
    for metric, m in ref['metrics'].items():
        case.assertIn('inPlay', m, metric)
        key = 'playerWindowCounts' if metric == 'nearestTeammate' else 'windowCounts'
        if metric == 'corrDM':
            continue
        case.assertIn(key, m, metric)
        for context, rules in m[key].items():
            case.assertIn(context, m['contexts'])
            block = rules['default']
            case.assertEqual(block['slideSeconds'], rf.WINDOW_SLIDE_SECONDS)
            entries = block['entries']
            case.assertEqual([e['minutes'] for e in entries], sorted(e['minutes'] for e in entries), (metric, context))
            for e in entries:
                case.assertEqual(sum(e['hist']), e['n'], (metric, context, e['minutes']))
                case.assertEqual(sum(e['nByTeam'].values()), e['n'])
                levels = [e[k] for k in ('p50', 'p90', 'p95', 'p99', 'p999', 'max')]
                case.assertEqual(levels, sorted(levels), (metric, context, e['minutes']))
                both = all(k >= rf.WINDOW_MIN_TEAM_SHARE*e['n'] and k > 0 for k in e['nByTeam'].values()) \
                    and len(e['nByTeam']) == 2
                case.assertEqual(e['bothTeams'], both, (metric, context, e['minutes']))
    json.dumps(ref, allow_nan=False)


class ShippedReference(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ref = strict_json(SHIPPED)

    def test_the_schema_is_the_contract(self):
        check_schema(self, self.ref)
        self.assertEqual(self.ref['source'], 'metrica-sample-game-1')
        self.assertEqual(self.ref['build']['game'], 1)
        v = self.ref['validation']                                         # game 2, once, thresholds frozen
        self.assertEqual((v['status'], v['verdict'], v['game']), ('run', 'PASS', 2))
        self.assertEqual(v['reference'], rf.reference_id(rf.load_reference(SHIPPED)), 'the record names this reference')
        self.assertTrue(v['referenceFileSha256'].startswith('e8dc19cdfecc8219'))  # the file as validated
        fp = v['falsePositiveTest']
        self.assertEqual((fp['verdict'], fp['game']), ('PASS', 2))
        self.assertEqual({(c['version'], str(c['length'])) for c in fp['cells']},
                         {(x, str(n)) for x in ('truth', 'cam60') for n in (60.0, 180.0, 600.0, 'half')})
        for c in fp['cells']:
            self.assertLessEqual(c['improvementsPer10TeamMinutes'], .2, c)
            self.assertEqual(c['highConfidenceConcerns'], 0, c)
        self.assertAlmostEqual(v['summary']['shareAtOrAboveAll']['width']['p90'], .078, delta=.001)
        previous = v['previous']                                           # the replaced reference's verdict
        self.assertEqual((previous[0]['reference'], previous[0]['verdict']),
                         ('metrica-sample-game-1@e8410fd48fca5ddc', 'FAIL'))

    def test_the_identity_leaves_the_validation_out(self):
        ref = strict_json(SHIPPED)
        sha = rf.content_sha(ref)
        self.assertEqual(rf.load_reference(SHIPPED)['_sha'], sha)
        self.assertEqual(rf.content_sha(dict(ref, validation={'status': 'not run'})), sha)
        changed = json.loads(json.dumps(ref))
        changed['metrics']['width']['contexts']['all']['p90'] += .1
        self.assertNotEqual(rf.content_sha(changed), sha, 'a threshold is part of the identity')

    def test_quantiles_are_monotone(self):
        for metric, m in self.ref['metrics'].items():
            for context, q in m['contexts'].items():
                values = [q[k] for k in rf.QUANTILES if k in q]
                self.assertEqual(values, sorted(values), (metric, context))
                for v in values:
                    self.assertTrue(math.isfinite(v), (metric, context))

    def test_n_per_context(self):
        for metric, m in self.ref['metrics'].items():
            c = m['contexts']
            self.assertGreaterEqual(c['all']['n'], rf.MIN_SAMPLES, metric)
            for family in (THIRDS, BACK_LINES, UNITS):
                present = [k for k in family if k in c]
                if present:
                    # The contexts split the 'all' samples: they never hold more between them.
                    self.assertLessEqual(sum(c[k]['n'] for k in present), c['all']['n'], (metric, family))
        per_sample = ('width', 'depth', 'hull', 'spread', 'gapDM', 'gapMF', 'lineStd', 'lineHeight')
        for metric in per_sample:      # one match, both teams, in play: tens of thousands of samples
            self.assertGreater(self.ref['metrics'][metric]['contexts']['all']['n'], 20000, metric)
        width = self.ref['metrics']['width']['contexts']
        self.assertEqual(sum(width[k]['n'] for k in THIRDS), width['all']['n'])   # every complete sample has a third
        self.assertGreaterEqual(self.ref['metrics']['lagDM']['contexts']['all']['n'], rf.MIN_SAMPLES)
        self.assertEqual(self.ref['metrics']['lagDM']['contexts']['all']['n'], self.ref['metrics']['corrDM']['contexts']['all']['n'])
        by_team = self.ref['metrics']['width']['byTeam']['all']
        self.assertEqual(by_team['A']['n']+by_team['B']['n'], width['all']['n'])

    def test_the_rule_and_its_baselines(self):
        rule = rf.rule(self.ref)
        self.assertEqual(rule, {'enter': 'p95', 'exit': 'p90', 'minSeconds': 5.0, 'mergeSeconds': 3.0})
        for metric in ('width', 'depth', 'hull', 'spread', 'gapDM', 'gapMF', 'lineStd', 'lineHeight', 'nearestTeammate'):
            m = self.ref['metrics'][metric]
            self.assertIn('all', m['episodes'], metric)
            for context, rules in m['episodes'].items():
                b, q = rules['default'], m['contexts'][context]
                self.assertGreaterEqual(q['n'], rf.MIN_SAMPLES, (metric, context))
                self.assertEqual(b['thresholds'], {'enter': q['p95'], 'exit': q['p90']}, (metric, context))
                p = b['perMinute']
                self.assertTrue(p['p10'] <= p['p50'] <= p['p90'], (metric, context))
                self.assertGreaterEqual(b['halvesInSpread'], 2)
                self.assertEqual(b['count'], sum(r['count'] for r in b['perHalf']), (metric, context))
                self.assertAlmostEqual(b['measuredMinutes'], sum(r['measuredMinutes'] for r in b['perHalf']), delta=.05)
                self.assertLessEqual({(r['team'], r['half']) for r in b['perHalf']}, {('A', 1), ('A', 2), ('B', 1), ('B', 2)})
                # By construction about 5% / 10% of the context's samples lie at or above its p95 / p90.
                self.assertAlmostEqual(b['shareAboveEnter'], .05, delta=.01, msg=(metric, context))
                self.assertAlmostEqual(b['shareAboveExit'], .10, delta=.01, msg=(metric, context))
                self.assertTrue(0 < b['shareOfTime'] < .15, (metric, context))
                self.assertGreaterEqual(b['durationSeconds']['p50'], rule['minSeconds'])
        self.assertEqual(len(self.ref['metrics']['width']['episodes']['all']['default']['perHalf']), 4)
        self.assertIn('perPlayerMinute', self.ref['metrics']['nearestTeammate']['episodes']['unit:D']['default'])

    def test_load_reference_reads_the_shipped_file(self):
        ref = rf.load_reference()
        self.assertIs(ref, rf.load_reference(rf.REFERENCE_PATH))
        self.assertTrue(rf.reference_id(ref).startswith('metrica-sample-game-1@'))
        self.assertEqual(len(ref['_sha']), 16)
        for metric in METRICS:
            value, used = rf.quantile(ref, metric, 'p90')
            self.assertEqual(used, 'all', metric)
            self.assertTrue(math.isfinite(value), metric)
        self.assertEqual(rf.quantile(ref, 'gapDM', 'p90', 'third:final')[1], 'third:final')
        self.assertEqual(rf.quantile(ref, 'lineStd', 'p90', 'backline:3')[1], 'backline:3')
        self.assertLess(rf.quantile(ref, 'lineStd', 'p90', 'backline:3')[0], rf.quantile(ref, 'lineStd', 'p90', 'backline:4')[0])
        self.assertEqual(rf.quantile(ref, 'nearestTeammate', 'p90', 'unit:F')[1], 'unit:F')
        self.assertEqual(rf.quantile(ref, 'gapDM', 'p90', 'no:such')[1], 'all')
        self.assertEqual(rf.episode_baseline(ref, 'gapDM', 'default', 'third:middle')['context'], 'third:middle')
        self.assertEqual(rf.episode_baseline(ref, 'lineStd', 'default', 'backline:5')['context'], 'all')   # sparse
        self.assertIsNone(rf.episode_baseline(ref, 'lagDM'))

    def test_the_file_is_ascii_with_crlf_line_endings(self):
        with open(SHIPPED, 'rb') as fh:
            data = fh.read()
        data.decode('ascii')
        lines = data.split(b'\n')
        self.assertEqual(lines[-1], b'')
        self.assertTrue(all(line.endswith(b'\r') for line in lines[:-1]))

    def test_the_professional_numbers_match_the_audit(self):
        # Metrica game 1 audit (2026-10-18, its own scripts): pooled in-play p10 / p50 / p90, the in-play variant.
        c = self.ref['metrics']
        for metric, expected, tol in (('width', (30.8, 41.3, 57.7), .2), ('depth', (23.2, 32.5, 43.1), .2),
                                      ('hull', (550, 930, 1379), 5), ('spread', (13.6, 17.4, 21.3), .2),
                                      ('gapDM', (7.5, 13.1, 20.2), .5), ('gapMF', (6.0, 12.0, 20.7), .5)):
            q = c[metric]['inPlay']['contexts']['all']
            for key, v in zip(('p10', 'p50', 'p90'), expected):
                self.assertAlmostEqual(q[key], v, delta=tol, msg=(metric, key))

    def test_the_distributions_include_the_dead_ball(self):
        # All the time with the full lineup observed: more samples than in play, the set pieces' crowding at the low end.
        c = self.ref['metrics']
        for metric in ('width', 'hull', 'gapDM', 'nearestTeammate'):
            q, ip = c[metric]['contexts']['all'], c[metric]['inPlay']['contexts']['all']
            self.assertGreater(q['n'], 1.3*ip['n'], metric)
            self.assertLess(q['p10'], ip['p10'], metric)
        self.assertEqual((c['hull']['contexts']['all']['p10'], c['width']['contexts']['all']['p90']), (479.0, 56.1))
        self.assertGreater(c['lagDM']['contexts']['all']['n'], 3*c['lagDM']['inPlay']['contexts']['all']['n'])
        self.assertIn('dead ball included', self.ref['notes'])

    def test_lags_at_the_sample_step_and_none_at_the_search_bound(self):
        for metric in ('lagDM',):
            for context, q in self.ref['metrics'][metric]['contexts'].items():
                for k in rf.QUANTILES:
                    self.assertAlmostEqual(q[k]/.2, round(q[k]/.2), places=6, msg=(context, k))
                    self.assertLess(abs(q[k]), rf.LAG_MAX_SECONDS-.2)
            for context, w in self.ref['metrics'][metric]['windows'].items():
                self.assertGreater(w['droppedAtSearchBound'], 0, context)

    def test_the_window_distributions(self):
        m = self.ref['metrics']
        # Game 1 has one team per back-line size: no both-team distribution there at any length.
        for context in ('backline:3', 'backline:4', 'backline:5'):
            entries = m['lineStd']['windowCounts'][context]['default']['entries']
            self.assertFalse(any(e['bothTeams'] for e in entries), context)
            self.assertEqual(rf.window_baseline(self.ref, 'lineStd', context, 3.0)['reason'], 'one_team_baseline')
        for metric in ('width', 'depth', 'hull', 'gapDM', 'gapMF'):
            for context in ('all', 'third:own', 'third:middle'):
                entries = m[metric]['windowCounts'][context]['default']['entries']
                self.assertEqual([e['minutes'] for e in entries][:5], [1.0, 1.5, 2.0, 2.5, 3.0], (metric, context))
                self.assertTrue(all(e['bothTeams'] for e in entries[:5]), (metric, context))
            half = [e for e in m[metric]['windowCounts']['all']['default']['entries'] if e.get('half')]
            self.assertEqual(len(half), 1, metric)
            self.assertGreater(half[0]['minutes'], 30.0)
        for context in ('unit:D', 'unit:M', 'unit:F'):
            self.assertTrue(m['nearestTeammate']['playerWindowCounts'][context]['default']['entries'][0]['bothTeams'])
        self.assertTrue(m['lagDM']['windowCounts']['all']['default']['entries'])
        # The next longer length is never below a length measured directly between two of them.
        check = self.ref['build']['windowBoundCheck']
        self.assertTrue(check)
        for key, rows in check.items():
            for row in rows:
                self.assertGreaterEqual(row['estimate'], row['direct']-1e-9, (key, row))

    def test_the_window_baseline_reads_the_next_longer_length(self):
        w = rf.window_baseline(self.ref, 'gapDM', 'all', 3.4)
        self.assertEqual((w['available'], w['method'], w['from']), (True, 'next_longer', [4.0]))
        entries = {e['minutes']: e for e in self.ref['metrics']['gapDM']['windowCounts']['all']['default']['entries']}
        self.assertEqual(w['p99'], max(entries[x]['p99'] for x in entries if x <= 4.0))
        self.assertEqual(rf.window_baseline(self.ref, 'gapDM', 'all', 1.0)['method'], 'exact')
        self.assertEqual(rf.window_baseline(self.ref, 'gapDM', 'all', .4)['from'], [1.0])
        longest = max(entries)
        far = rf.window_baseline(self.ref, 'gapDM', 'all', 2*longest)
        self.assertEqual(far['method'], 'scaled')
        self.assertAlmostEqual(far['p99'], 2*max(e['p99'] for e in entries.values()), places=2)
        self.assertEqual(rf.window_baseline(self.ref, 'gapDM', 'no:such', 3.0)['reason'], 'no_window_baseline')
        # the empirical tail grows with the count's rarity and is at least the share at the length read
        p1 = rf.window_tail(self.ref, 'gapDM', 'all', 3.4, 1)
        p4 = rf.window_tail(self.ref, 'gapDM', 'all', 3.4, 4)
        self.assertTrue(0 <= p4 <= p1 <= 1)
        e4 = entries[4.0]
        self.assertGreaterEqual(p4, sum(e4['hist'][4:])/e4['n']-1e-12)
        self.assertIsNone(rf.window_tail(self.ref, 'lineStd', 'backline:4', 3.0, 5))

    def test_tau_and_the_coverage_of_the_strength_bound(self):
        for metric in ('width', 'depth', 'hull', 'spread', 'gapDM', 'gapMF', 'lineStd', 'lineHeight'):
            tau = self.ref['metrics'][metric]['tau']
            self.assertTrue(5.0 <= tau['seconds'] <= 20.0, (metric, tau))
            self.assertGreater(tau['iatSeconds'], tau['seconds'])
            self.assertEqual(rf.tau_seconds(self.ref, metric), tau['seconds'])
            for T in ('120', '300'):           # the nominal 80% bound covers the team-half's own share about 80%
                self.assertTrue(.72 <= tau['coverage80'][T] <= .9, (metric, T, tau['coverage80']))
                self.assertGreater(tau['coverage80Integrated'][T], tau['coverage80'][T])
        self.assertTrue(10.0 <= rf.tau_seconds(self.ref, 'lagDM') <= 30.0)
        self.assertIsNone(rf.tau_seconds(self.ref, 'nearestTeammate'))


class Statistics(unittest.TestCase):
    def test_the_share_upper_bound(self):
        self.assertAlmostEqual(rf.share_upper_bound(0.0, 6.0), 1-.2**(1/6), places=9)
        self.assertGreater(rf.share_upper_bound(0.0, 5.0), .25)                # 0% over 5 looks is not enough
        self.assertLess(rf.share_upper_bound(0.0, 5.6), .25)
        self.assertLess(rf.share_upper_bound(.1, 60.0), .25)
        self.assertGreater(rf.share_upper_bound(.2, 60.0), .25)
        self.assertGreater(rf.share_upper_bound(.1, 60.0), .1)
        self.assertEqual(rf.share_upper_bound(1.0, 10.0), 1.0)
        self.assertEqual(rf.share_upper_bound(.0, 0.0), 1.0)
        a, b = rf.share_upper_bound(.05, 20.0), rf.share_upper_bound(.05, 80.0)
        self.assertGreater(a, b)                                                 # more looks, tighter bound

    def test_holm(self):
        self.assertEqual(rf.holm([], .01), [])
        self.assertEqual(rf.holm([.001, .02, .003], .01), [0, 2])            # .001 <= .01/3, .003 <= .01/2
        self.assertEqual(rf.holm([.004, .004, .004], .01), [])               # .004 > .01/3
        self.assertEqual(rf.holm([None, .0], .05), [1])

    def test_the_lag_resolution(self):
        self.assertEqual([rf.quantize_lag(x, .2) for x in (-.15, -.09, .1, .31, 1.06, None)],
                         [-.2, 0.0, 0.0, .4, 1.0, None])


class Lateral(unittest.TestCase):
    def test_a_known_lag_is_measured(self):
        step = .2
        t = np.arange(0, 60, step)
        lead = np.sin(2*math.pi*t/9)+.4*np.sin(2*math.pi*t/4.1)
        for shift in (3, -4):           # samples: +0.6 s (follows later), -0.8 s (moves earlier)
            follow = np.roll(lead, shift)
            lag, r = rf.lateral_lag(lead[10:-10], follow[10:-10], step)
            self.assertAlmostEqual(lag, shift*step, delta=.03)
            self.assertGreater(r, .99)

    def test_player_velocity_never_bridges(self):
        step = .2
        t = np.arange(0, 20, step)
        Y = np.stack([2.0*t, 30-1.5*t], axis=1)
        V = rf.player_velocity(t, Y)
        ok = np.isfinite(V)
        self.assertTrue(np.allclose(V[ok[:, 0], 0], 2.0) and np.allclose(V[ok[:, 1], 1], -1.5))
        self.assertTrue(np.isnan(V[:3]).all() and np.isnan(V[-3:]).all())      # the span runs off the data
        Y[40, 0] = np.nan                                                      # one unobserved sample
        V = rf.player_velocity(t, Y)
        self.assertTrue(np.isnan(V[37:44, 0]).all() and np.isfinite(V[36, 0]) and np.isfinite(V[44, 0]))
        self.assertTrue(np.isfinite(V[37:44, 1]).all())
        t2 = t.copy()
        t2[50:] += 3.0                                                          # a time gap
        V = rf.player_velocity(t2, np.stack([2.0*t2, 2.0*t2], axis=1))
        self.assertTrue(np.isnan(V[47:53]).all())
        self.assertTrue(np.allclose(V[np.isfinite(V[:, 0]), 0], 2.0))

    def test_a_unit_change_does_not_jump(self):
        step = .2
        t = np.arange(0, 20, step)
        Y = np.stack([10+t, 20+t, 50+t], axis=1)      # everyone moves sideways at 1 m/s
        members = np.zeros((len(t), 3), bool)
        members[:, 0] = True
        members[:50, 1] = True                        # player 1 leaves the unit, player 2 joins
        members[50:, 2] = True
        v = rf.unit_velocity(t, Y, members)
        self.assertTrue(np.allclose(v[np.isfinite(v)], 1.0))
        self.assertGreater(np.isfinite(v).sum(), len(t)-8)
        none = rf.unit_velocity(t, Y, np.zeros_like(members))
        self.assertTrue(np.isnan(none).all())

    def test_windows_lie_inside_continuous_eligible_stretches(self):
        step = .2
        t = np.arange(0, 100, step)
        lead = np.sin(2*math.pi*t/9)
        follow = np.roll(lead, 2)
        ok = np.ones(len(t), bool)
        ok[(t >= 40) & (t < 45)] = False
        ws = rf.lateral_windows(t, lead, follow, ok, 30.0, 5.0)
        for w in ws:
            self.assertFalse(w['t0'] < 45 and w['t1'] >= 40)
            self.assertAlmostEqual(w['t1']-w['t0'], 30.0-step, places=6)
            self.assertAlmostEqual(w['lag'], .4, delta=.05)
        self.assertEqual([round(w['t0'], 1) for w in ws][:4], [0.0, 5.0, 10.0, 45.0])
        flat = rf.lateral_windows(t, np.full(len(t), .1), follow, ok)           # one unit not moving: no lag
        self.assertEqual(flat, [])


class Refusals(unittest.TestCase):
    """Every refusal comes before any data is read: the data directory does not even exist."""
    MISSING = os.path.join(tempfile.gettempdir(), 'pitchiq-no-such-metrica-dir')

    def refused(self, argv, isolated=True):
        with self.assertRaises(bench.Refused) as cm, contextlib.redirect_stdout(io.StringIO()):
            bench.main(['--data', self.MISSING]+argv, isolated=isolated)
        return cm.exception.message

    def test_the_held_out_game_needs_validate(self):
        self.assertIn('held out', self.refused(['--game', '2']))
        self.assertIn('held out', self.refused(['--game', '2', '--out', os.path.join(self.MISSING, 'r.json')]))
        self.assertIn('game 1 only', self.refused(['--game', '3']))

    def test_validate_never_on_the_tuning_game_and_never_writes_the_reference(self):
        self.assertIn('proves nothing', self.refused(['--game', '1', '--validate']))
        self.assertIn('never writes', self.refused(['--game', '2', '--validate', '--out', SHIPPED]))
        self.assertIn('overwrite', self.refused(['--game', '2', '--validate', '--json', SHIPPED]))
        self.assertIn('belong to --validate', self.refused(['--game', '1', '--record']))
        self.assertIn('recorded with --record', self.refused(['--game', '2', '--validate', '--acceptance', SHIPPED]))
        with tempfile.TemporaryDirectory() as d:
            bad = os.path.join(d, 'fp.json')
            with open(bad, 'w', encoding='utf-8') as fh:
                json.dump({'verdict': 'PASS', 'referenceFileSha256': '0'*64}, fh)
            self.assertIn('another reference file', self.refused(['--game', '2', '--validate', '--record',
                                                                  '--acceptance', bad]))

    def test_downloaded_data_only_under_an_isolated_interpreter(self):
        self.assertIn('python -I -B', self.refused(['--game', '1'], isolated=False))
        self.assertIn('python -I -B', self.refused(['--game', '2', '--validate'], isolated=False))
        r = subprocess.run([sys.executable, '-B', BENCH, '--data', self.MISSING, '--game', '1'], capture_output=True,
                           text=True, timeout=120, env=dict(os.environ, CUDA_VISIBLE_DEVICES='-1'))
        self.assertNotEqual(r.returncode, 0)
        self.assertIn('python -I -B', r.stderr)


class EndToEnd(unittest.TestCase):
    """Build and validation on synthetic files in a temporary directory (never the Metrica data)."""

    @classmethod
    def setUpClass(cls):
        cls.dir = tempfile.TemporaryDirectory()
        d = cls.dir.name
        write_metrica_game(d, 1, seed=0)
        write_metrica_game(d, 2, seed=5)
        cls.ref_path = os.path.join(d, 'reference.json')
        cls.val_path = os.path.join(d, 'validation.json')
        keep = bench.MIN_HALF_MINUTES
        bench.MIN_HALF_MINUTES = 1.0             # 2-minute halves: let them into the spread for the schema checks
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                cls.built = bench.main(['--data', d, '--game', '1', '--out', cls.ref_path], isolated=True)
                cls.validation = bench.main(['--data', d, '--game', '2', '--validate', '--reference', cls.ref_path,
                                             '--json', cls.val_path], isolated=True)
        finally:
            bench.MIN_HALF_MINUTES = keep

    @classmethod
    def tearDownClass(cls):
        cls.dir.cleanup()

    def test_the_built_reference_has_the_schema(self):
        ref = strict_json(self.ref_path)
        check_schema(self, ref)
        self.assertEqual(ref['build']['game'], 1)
        self.assertEqual(set(ref['build']['dataSha256']), {'Sample_Game_1_RawTrackingData_Home_Team.csv',
                                                          'Sample_Game_1_RawTrackingData_Away_Team.csv'})
        self.assertEqual(rf.load_reference(self.ref_path)['source'], 'metrica-sample-game-1')
        with open(self.ref_path, 'rb') as fh:
            self.assertTrue(all(line.endswith(b'\r') for line in fh.read().split(b'\n')[:-1]))

    def test_the_baselines_are_the_episodes_module_counting(self):
        cd, play, halves, _ = bench.load(self.dir.name, 1)
        ref = strict_json(self.ref_path)
        rule = ref['rules']['default']
        checked = 0
        for metric, key, gate in (('width', 'width', 'width'), ('lineHeight', 'lineHeight', 'lineStd'),
                                  ('nearestTeammate', 'nearestTeammate', 'nearestTeammate')):
            b = ref['metrics'][metric]['episodes']['all']['default']
            for row in b['perHalf']:
                s = cd.teams[row['team']]
                base = bench.team_base(s, np.ones(len(cd.t), bool)) & (halves == row['half'])      # dead ball too
                v = s.values[key]
                e = s.eligible[gate] & (base[:, None] if v.ndim == 2 else base)
                cols = [(v[:, j], e[:, j]) for j in range(v.shape[1])] if v.ndim == 2 else [(v, e)]
                n = sum(len(ep.episodes(cd.t, vv, ee, b['thresholds']['enter'], b['thresholds']['exit'],
                                        rule['minSeconds'], rule['mergeSeconds'], 'above', cd.step)) for vv, ee in cols if ee.any())
                self.assertEqual(n, row['count'], (metric, row))
                checked += n
        self.assertGreater(checked, 0)

    def test_the_window_counts_are_a_detectors_count_on_the_window(self):
        cd, play, halves, _ = bench.load(self.dir.name, 1)
        ref = strict_json(self.ref_path)
        rule = ref['rules']['default']
        samples = bench.collect(cd, np.ones(len(cd.t), bool))
        q = ref['metrics']['width']['contexts']['all']
        units = bench.context_units(cd.t, halves, samples['width'], 'all', q['p95'], q['p90'], rule, cd.step)
        checked = 0
        for u in units:
            spans = [(x['i0'], x['i1']) for x in u['episodes']]
            counts = bench.span_counts(cd.t, cd.step, u['mask'], spans, 1.0, rule['minSeconds'])
            if counts is None:
                continue
            idx = np.flatnonzero(u['mask'])
            K = int(round(60/cd.step))
            every = int(round(rf.WINDOW_SLIDE_SECONDS/cd.step))
            v = cd.teams[u['team']].values['width']
            for k, s0 in enumerate(range(0, len(idx)-K+1, every)):
                m = np.zeros(len(cd.t), bool)
                m[idx[s0:s0+K]] = True
                exact = len(ep.episodes(cd.t, v, m, q['p95'], q['p90'], rule['minSeconds'], rule['mergeSeconds'],
                                        'above', cd.step))
                self.assertLessEqual(exact, counts[k])                  # never fewer than a detector finds
                self.assertLessEqual(counts[k]-exact, 1)
                checked += 1
        self.assertGreater(checked, 10)
        entries = ref['metrics']['width']['windowCounts']['all']['default']['entries']
        self.assertTrue(entries and entries[0]['bothTeams'])

    def test_the_validation_report(self):
        v = strict_json(self.val_path)
        self.assertEqual(v['game'], 2)
        self.assertTrue(v['reference'].startswith('metrica-sample-game-1@'))
        self.assertEqual(set(v['metrics']), set(METRICS))
        width = v['metrics']['width']['all']
        self.assertEqual(set(width['shareAtOrAbove']), set(rf.QUANTILES))
        self.assertAlmostEqual(width['shareAtOrAbove']['p90']['expected'], .1)
        near = v['metrics']['nearestTeammate']['all']
        self.assertIn('episodes', near)
        for row in near['episodes']['perHalf']:
            self.assertIn(row['withinTuningRange'], (True, False, None))
        rows = v['metrics']['width']['all']['windowCounts']
        self.assertTrue(rows)
        for r in rows:
            self.assertTrue(0 <= r['beyondP99'] <= 1)
        self.assertIn('playerWindowCounts', v['metrics']['nearestTeammate']['unit:D'])
        # The validation read the reference and wrote only its own report.
        self.assertEqual(strict_json(self.ref_path)['validation']['status'], 'not run')

    def test_record_writes_only_the_validation_block(self):
        import hashlib
        import shutil
        d = self.dir.name
        path = os.path.join(d, 'recorded.json')
        shutil.copyfile(self.ref_path, path)
        with open(path, 'rb') as fh:
            sha = hashlib.sha256(fh.read()).hexdigest()
        acc = os.path.join(d, 'acceptance.json')
        with open(acc, 'w', encoding='utf-8') as fh:
            json.dump({'game': 2, 'verdict': 'FAIL', 'referenceFileSha256': sha, 'cells': []}, fh)
        before = strict_json(path)
        with contextlib.redirect_stdout(io.StringIO()):
            bench.main(['--data', d, '--game', '2', '--validate', '--reference', path, '--record', '--acceptance', acc],
                       isolated=True)
        after = strict_json(path)
        self.assertEqual([k for k in before if k != 'validation'], [k for k in after if k != 'validation'])
        self.assertTrue(all(before[k] == after[k] for k in before if k != 'validation'), 'nothing but the record')
        v = after['validation']
        self.assertEqual((v['status'], v['verdict'], v['referenceFileSha256'], v['game']), ('run', 'FAIL', sha, 2))
        self.assertEqual(v['reference'], rf.reference_id(rf.load_reference(path)), 'the identity is unchanged')
        self.assertEqual(rf.content_sha(after), rf.content_sha(before))
        self.assertEqual(v['previous'], before['validation']['previous'])
        self.assertIn('width', v['summary']['shareAtOrAboveAll'])
        with open(path, 'rb') as fh:
            self.assertTrue(all(line.endswith(b'\r') for line in fh.read().split(b'\n')[:-1]))
        with open(acc, 'w', encoding='utf-8') as fh:     # the record's file changed: a summary of it now needs its sha
            with open(path, 'rb') as ref_fh:
                json.dump({'verdict': 'PASS', 'referenceFileSha256': hashlib.sha256(ref_fh.read()).hexdigest()}, fh)
        with self.assertRaises(bench.Refused) as cm, contextlib.redirect_stdout(io.StringIO()):
            bench.main(['--data', d, '--game', '2', '--validate', '--reference', path, '--record', '--acceptance', acc],
                       isolated=True)
        self.assertIn('validated already', cm.exception.message)

    def test_the_command_line_under_python_i_b(self):
        out = os.path.join(self.dir.name, 'cli.json')
        r = subprocess.run([sys.executable, '-I', '-B', BENCH, '--data', self.dir.name, '--game', '1', '--out', out],
                           capture_output=True, text=True, timeout=300, env=dict(os.environ, CUDA_VISIBLE_DEVICES='-1'))
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn('written to', r.stdout)
        ref = strict_json(out)
        self.assertEqual(ref['source'], 'metrica-sample-game-1')
        self.assertEqual(ref['metrics']['width']['contexts'], strict_json(self.ref_path)['metrics']['width']['contexts'])


if __name__ == '__main__':
    unittest.main()
