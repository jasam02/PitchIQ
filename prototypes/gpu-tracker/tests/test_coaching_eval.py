"""tools/coaching_eval.py's own parts that test_coaching_counterexamples and test_coaching_levels do not cover: the
blind label sheets of DESIGN 13.3 (MOMENT, ROLE, ADVANCE, ZONE) are blind, complete, deterministic and recorded in a
manifest; the scorer turns filled sheets into the success criteria, says "not measured" when a sheet is thin, records
the frozen rules it scored against and refuses a key that changed after the sheets were made; the summary table reads a
`runs` folder; the StatsBomb reader measures the nearest-opponent distance on a hand-made 360 frame and skips malformed
input (the real download is untrusted and never read by a test); nothing is written under runs/ or uploads/.

The sheets are built here from one synthetic scene (both teams of unsupported_advance/none_behind, 366 s at 10 fps)
treated as a broadcast clip; the real sheets (tools/annotations/coaching/labels/) come from the five run copies.

Fix round 1 (critic N1 / B-9, N3, section 6 G1+): every key a sheet writes carries the frozen block of the rules it was
computed with; `rekey` recomputes PitchIQ's side of the same MOMENT / ROLE / ADVANCE items and, on unchanged code,
reproduces every engine field (0 of 3 sets changed here); the scorer refuses a key without a frozen block (the
committed wave-8 sheets: rekey after the freeze) and one whose block differs after a one-line change of a frozen file;
mock labels score identically through the rekeyed keys; BC60 0-60 s and BC303 222-303 s are 'seen' for D3 / D4; and
each new invariant of coaching_eval.fix1_problems flags a report doctored to break it (presser, phase words, level-1
"someone back" off the final third and on a covered sample, a strength's short run, a reason left out).

Fix round 2 (LEAD decisions (b) and (d), game-3 G1+): coaching_eval.fix2_problems finds nothing on the scene reports
with a 'moments worth a look' review (small_field_scale/declared, levels 4 and 5) and flags each doctored one: the
form (a verdict, a priority, a drill, below level 3, a failed guardrail besides the evidence one, a second finding of
the family), more reviews than the level's cap, each stated count off by one, an opportunity whose identity is no
longer high, a cited moment whose player is not named or that lies outside the family, the count of all
opportunities, a missing or wrong share at level 5, a cited time left out, the possible-issue headline, verdict and
blame words, and an overview counting it as a possible issue.

Fix round 3 (the round-2 critic's B-1, B-2, B-3 validators; LEAD decision (2)): the amended level-1/2 rule accepts an
exposed failure wherever the ball is and still refuses a covered one, a not-exposed one off the final third and one
keeping enough players behind the ball; the moments recount applies the settled-play, contested-ball, line and
marking gates (7 clear of the 11 judged in 5 spells on small_field_scale/declared, 10 in 8 before) and the base-rate
test (scipy's binomial tail equals assess.binom_tail to 12 places); fix3_problems flags a doctored actor, gap pair,
third, aggregate, all-opportunity strength count, unmargined behind-the-ball count, report-level text and drill.

Run:  python -m unittest test_coaching_eval        (from this directory; about 45 s cold)
"""
import copy
import csv
import hashlib
import json
import math
import re
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for p in (str(ROOT), str(HERE), str(ROOT/'tools')):
    if p not in sys.path:
        sys.path.insert(0, p)
import coaching_detect_fixtures as FX  # noqa: E402
import coaching_eval as E  # noqa: E402

SCENE = ('unsupported_advance', 'none_behind')
ITEM_KEYS = {'moment': {'id', 'footage', 'video', 'time', 'clock', 'team', 'teamHint', 'from', 'to', 'marks'},
             'role': {'id', 'footage', 'video', 'time', 'clock', 'team', 'player', 'clip', 'from', 'to', 'marks'},
             'advance': {'id', 'footage', 'video', 'time', 'clock', 'team', 'player', 'from', 'to', 'marks'},
             'zone': {'id', 'source', 'video', 'time', 'clock', 'team', 'side', 'scale', 'players', 'ball', 'carrier',
                      'ringed', 'problem'}}
# words of PitchIQ's coaching output that a blind sheet must not carry (verdicts, outcomes, detector and record ids)
LEAKS = re.compile(r'work_on|try_this|not_demonstrated|demonstrated|strength|wide_defender_advance|pressure_cover|'
                   r'weak_side_balance|behind_ball|defensive_spacing|attacking_shape|"zone"|"mover"|principle|'
                   r'\bopp-[0-9a-f]{6}|\bev-[0-9a-f]{6}|\bfnd-[0-9a-f]{6}')


def agreeing_labels(keys, labels):
    """Write moment_labels.csv and zone_labels.json into `labels` with the answers PitchIQ's keys hold (a coach who
    agrees with every measured outcome): the scorer must then report agreement 1.0."""
    labels.mkdir(parents=True, exist_ok=True)
    rows = []
    for k in keys['moment']:
        e = k['engineByRun']['SYN']
        ph = (e['phase'] or {}).get('state')
        row = {'id': k['id'], 'ball': {'own': 'this team', 'opp': 'other team', 'dead': 'nobody'}.get(ph, 'unclear')}
        d4 = e['detectors'].get(E.D4)
        for b in E.BANDS:
            if d4 and d4.get(b) in (1, -1):
                row[f'behind_{b}'] = 'yes' if d4[b] == 1 else 'no'
        for det, q in ((E.D2, 'farside'), (E.D5, 'line'), (E.D6, 'spread')):
            x = e['detectors'].get(det)
            if x and x.get('intermediate') in (1, -1):
                row[q] = 'yes' if x['intermediate'] == 1 else 'no'
        rows.append(row)
    cols = ['id', 'ball', 'farside', 'line', 'spread']+[f'behind_{b}' for b in E.BANDS]
    with open(labels/'moment_labels.csv', 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, cols)
        w.writeheader()
        w.writerows(rows)
    zones = {k['id']: {'polygon': k['zone']['polygon'], 'mover': k['mover'], 'agree': 'yes'} for k in keys['zone']}
    (labels/'zone_labels.json').write_text(json.dumps({'set': 'zone', 'answers': zones}), encoding='utf-8')
    return rows, zones


def sheet_data(path):
    text = Path(path).read_text(encoding='ascii')
    m = re.search(r'<script id="data" type="application/json">(.*?)</script>', text, re.S)
    return json.loads(m.group(1).replace('<\\/', '</'))


class Sheets(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        teams = {}
        for team in ('A', 'B'):
            b, st, obs, roles = E.scene_tiers(*SCENE, team=team, prepared=FX.prepared)
            teams[team] = (obs, roles)
        cls.built = b
        obs = teams['A'][0]
        cls.src = {'run': 'SYN', 'video': b.result['video']['name'], 'footage': 'broadcast', 'split': 'holdout',
                   'segment': (obs['segment']['t0'], obs['segment']['t1']), 'index': E.frame_index(b.result),
                   'teams': teams}
        cls.tmp = tempfile.TemporaryDirectory()
        cls.dir = Path(cls.tmp.name)/'sheets'
        cls.man = E.make_sheets([cls.src], cls.dir)
        cls.keys = {name: json.loads((cls.dir/'key'/f'{name}_key.json').read_text(encoding='ascii'))['keys']
                    for name in ITEM_KEYS}

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_every_set_is_written_with_its_files(self):
        for name in ('moment', 'role', 'advance'):
            for ext in ('html', 'csv'):
                self.assertTrue((self.dir/f'{name}_sheet.{ext}').is_file(), (name, ext))
        for rel in ('zone_sheet.html', 'zone_items.csv', 'INSTRUCTIONS.txt', 'key/zone_reveal.html', 'key/README.txt'):
            self.assertTrue((self.dir/rel).is_file(), rel)
        self.assertGreater(self.man['counts']['moment'], 40)
        self.assertGreater(self.man['counts']['role'], 10)
        self.assertGreater(self.man['counts']['advance'], 0)

    def test_the_sheets_are_blind(self):
        for name, allowed in ITEM_KEYS.items():
            data = sheet_data(self.dir/f'{name}_sheet.html')
            self.assertEqual(len(data['items']), self.man['counts'][name], name)
            for it in data['items']:
                self.assertLessEqual(set(it), allowed, (name, it['id']))
            blob = json.dumps(data['items'])
            self.assertIsNone(LEAKS.search(blob), (name, LEAKS.search(blob) and LEAKS.search(blob).group(0)))
        for name in ('moment', 'role', 'advance'):
            text = (self.dir/f'{name}_sheet.csv').read_text(encoding='ascii')
            self.assertIsNone(LEAKS.search(text), name)
        # the ids PitchIQ gave its records never reach a sheet
        obs, roles = self.src['teams']['A']
        ids = {o['id'] for o in obs['opportunities']+roles['opportunities']}
        ids |= {m['id'] for m in obs['moments']+roles['moments']}
        for rel in ('moment_sheet.html', 'role_sheet.html', 'advance_sheet.html', 'zone_sheet.html'):
            text = (self.dir/rel).read_text(encoding='ascii')
            self.assertFalse([i for i in ids if i in text], rel)

    def test_the_keys_hold_what_the_sheets_hide(self):
        for name, keys in self.keys.items():
            self.assertEqual(sorted(k['id'] for k in keys),
                             sorted(it['id'] for it in sheet_data(self.dir/f'{name}_sheet.html')['items']), name)
        for k in self.keys['moment']:
            self.assertIn('SYN', k['engineByRun'])
        for k in self.keys['zone']:
            self.assertGreaterEqual(len(k['zone']['polygon']), 3)

    def test_moments_are_stratified_spaced_and_thirty_percent_random(self):
        keys = self.keys['moment']
        rand = sum(k['stratum']['kind'] == 'random' for k in keys)
        self.assertAlmostEqual(rand/len(keys), E.RANDOM_SHARE, delta=1.5/len(keys)+.01)
        strata = {(k['stratum']['detector'], k['stratum']['outcome']) for k in keys if k['stratum']['kind'] != 'random'}
        self.assertGreaterEqual(len(strata), 3)
        by_team = {}
        for k in keys:
            by_team.setdefault(k['team'], []).append(k['time'])
        for ts in by_team.values():
            ts.sort()
            self.assertGreaterEqual(min(b-a for a, b in zip(ts, ts[1:])), E.MOMENT_SPACING_S-1e-6)

    def test_the_manifest_hashes_every_file_and_the_sheets_are_deterministic(self):
        for rel, sha in self.man['files'].items():
            self.assertEqual(hashlib.sha256((self.dir/rel).read_bytes()).hexdigest(), sha, rel)
        again = E.make_sheets([self.src], Path(self.tmp.name)/'again')
        self.assertEqual(again['files'], self.man['files'])

    def test_the_scorer_on_labels_that_agree_with_pitchiq(self):
        labels = Path(self.tmp.name)/'labels'
        rows, zones = agreeing_labels(self.keys, labels)
        res = E.score_labels(labels, self.dir)
        allm = res['sets']['moment']['bySplit']['all']
        self.assertEqual(res['sets']['moment']['labelled'], len(rows))
        for name, a in allm.items():
            if name.startswith(('phase', 'D4.', 'D2.', 'D5.', 'D6.')):
                self.assertEqual(a['agreement'], 1.0, name)
        zres = res['sets']['zone']['holdout']        # the scene source is a held-out clip
        self.assertEqual((zres['sameMover'], zres['centroidInside']), (1.0, 1.0))
        n_zone = len(zones)
        self.assertEqual(zres['drawn'], n_zone)
        self.assertEqual(res['sets']['zone']['tune']['drawn'], 0)
        crit = res['criteria']['9 coach centroid inside PitchIQ zone >= .7 (held out)']
        if n_zone >= 10:
            self.assertTrue(crit['pass'])
        else:
            self.assertTrue(str(crit).startswith('not measured'))
        self.assertTrue(str(res['criteria']['6 held-out precision of "unsupported" >= .8']).startswith('not measured'))
        self.assertIn('soccer/coaching/profiles.json', res['frozen'])
        self.assertIn('moment_labels.csv', res['labelFiles'])

    def test_the_scorer_refuses_a_key_changed_after_the_sheets(self):
        d = Path(self.tmp.name)/'tampered'
        E.make_sheets([self.src], d)
        path = d/'key'/'moment_key.json'
        before = path.read_bytes()
        path.write_bytes(before.replace(b'"holdout"', b'"tune"', 1))
        self.assertNotEqual(path.read_bytes(), before)
        (d/'moment_labels.csv').write_text('id,ball\n', encoding='utf-8')
        with self.assertRaises(SystemExit):
            E.score_labels(d, d)

    def test_every_key_names_the_rules_it_was_computed_with(self):
        now = E.frozen_state()
        for name in ITEM_KEYS:
            doc = json.loads((self.dir/'key'/f'{name}_key.json').read_text(encoding='ascii'))
            self.assertEqual(doc['frozen'], now, name)
        self.assertEqual(E.frozen_diff(now), [])
        self.assertTrue(E.frozen_diff(None)[0].startswith('no frozen block'))

    def test_rekey_on_unchanged_code_reproduces_the_keys(self):
        out = Path(self.tmp.name)/'rekey'
        man = E.rekey_keys(self.dir, [self.src], out)
        self.assertEqual({k: v['engineFieldsChanged'] for k, v in man['changed'].items()},
                         {'moment': 0, 'role': 0, 'advance': 0})
        self.assertIn('zone', man['notRekeyed'])
        for name in E.REKEY_SETS:
            old = {k['id']: k for k in self.keys[name]}
            new = json.loads((out/f'{name}_key.json').read_text(encoding='ascii'))
            self.assertEqual(new['frozen'], E.frozen_state())
            self.assertEqual(sorted(k['id'] for k in new['keys']), sorted(old))
            for k in new['keys']:
                field = {'moment': 'engineByRun', 'role': 'records', 'advance': 'd3'}[name]
                self.assertEqual(k[field], old[k['id']][field], (name, k['id']))
                if name == 'moment' and k['stratum']['kind'] == 'outcome':
                    self.assertEqual((k['engineStratum']['outcome'], k['engineStratum']['evidence']),
                                     (k['stratum']['outcome'], k['stratum']['evidence']), k['id'])

    def test_mock_labels_score_identically_through_the_rekey(self):
        labels = Path(self.tmp.name)/'labels_rekey'
        agreeing_labels(self.keys, labels)
        out = Path(self.tmp.name)/'rekey2'
        E.rekey_keys(self.dir, [self.src], out)
        a, b = E.score_labels(labels, self.dir), E.score_labels(labels, self.dir, keys_dir=out)
        self.assertEqual(a['sets'], b['sets'])
        self.assertEqual(a['criteria'], b['criteria'])

    def test_the_scorer_refuses_a_key_of_other_rules(self):
        # a deliberate one-line change of a frozen file (a copy of the repository's frozen files, so nothing real is
        # touched): the same sheets' keys are then refused, the unchanged copy is accepted
        labels = Path(self.tmp.name)/'labels_rules'
        agreeing_labels(self.keys, labels)
        fake = Path(self.tmp.name)/'repo_copy'
        for rel in E.FROZEN_FILES:
            (fake/rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(E.REPO/rel, fake/rel)
        for p in (E.REPO/'soccer'/'coaching').glob('*.py'):
            shutil.copyfile(p, fake/'soccer'/'coaching'/p.name)
        with mock.patch.object(E, 'REPO', fake):
            E.score_labels(labels, self.dir)
            path = fake/'soccer'/'coaching'/'profiles.json'
            text = path.read_bytes()
            i = text.index(b'"value": ')+len(b'"value": ')
            path.write_bytes(text[:i]+b'1'+text[i:])          # one rule value changed (0.6 -> 10.6, say)
            with self.assertRaises(SystemExit) as cm:
                E.score_labels(labels, self.dir)
            self.assertIn('profiles.json', str(cm.exception))

    def test_a_rekey_for_other_items_is_refused(self):
        out = Path(self.tmp.name)/'rekey3'
        E.rekey_keys(self.dir, [self.src], out)
        path = out/'moment_key.json'
        doc = json.loads(path.read_text(encoding='ascii'))
        doc['keys'] = doc['keys'][1:]
        path.write_text(json.dumps(doc), encoding='ascii')
        man = json.loads((out/'rekey_manifest.json').read_text(encoding='ascii'))
        man['files']['moment_key.json'] = hashlib.sha256(path.read_bytes()).hexdigest()
        (out/'rekey_manifest.json').write_text(json.dumps(man), encoding='ascii')
        labels = Path(self.tmp.name)/'labels_items'
        agreeing_labels(self.keys, labels)
        with self.assertRaises(SystemExit):
            E.score_labels(labels, self.dir, keys_dir=out)

    def test_nothing_is_written_under_runs_or_uploads(self):
        for bad in ('runs', 'uploads'):
            with self.assertRaises(SystemExit):
                E.write_json(Path(self.tmp.name)/bad/'x.json', {})
            with self.assertRaises(SystemExit):
                E.make_sheets([self.src], Path(self.tmp.name)/bad/'sheets')


LABELS = ROOT/'tools'/'annotations'/'coaching'/'labels'


@unittest.skipUnless((LABELS/'manifest.json').is_file(), 'no committed label sheets')
class CommittedSheets(unittest.TestCase):
    """The sheets handed to the user (tools/annotations/coaching/labels/, made Oct 22 from the five run copies and 45
    Metrica game 1 positions): every file is the one the manifest hashed (so the keys the scorer reads are the keys
    made with the sheets), and no sheet shows PitchIQ's coaching output."""

    def test_the_files_are_the_manifests(self):
        man = json.loads((LABELS/'manifest.json').read_text(encoding='ascii'))
        for rel, sha in man['files'].items():
            self.assertEqual(hashlib.sha256((LABELS/rel).read_bytes()).hexdigest(), sha, rel)
        self.assertEqual(man['counts']['moment'], sum(man['countsByFootage']['moment'].values()))
        self.assertGreaterEqual(man['countsByFootage']['moment']['broadcast'], 190)

    def test_the_committed_keys_predate_the_freeze_and_are_refused(self):
        # critic N1: the wave-8 keys hold the engine of Oct 22; a score must rekey them after the freeze first
        keys = json.loads((LABELS/'key'/'moment_key.json').read_text(encoding='ascii'))
        if 'frozen' in keys:
            self.skipTest('the committed sheets were regenerated with a frozen block')
        with tempfile.TemporaryDirectory() as d:
            Path(d, 'moment_labels.csv').write_text('id,ball\n', encoding='utf-8')
            with self.assertRaises(SystemExit) as cm:
                E.score_labels(d, LABELS)
            self.assertIn('frozen', str(cm.exception))

    def test_the_committed_sheets_are_blind(self):
        for name, allowed in ITEM_KEYS.items():
            data = sheet_data(LABELS/f'{name}_sheet.html')
            for it in data['items']:
                self.assertLessEqual(set(it), allowed, (name, it['id']))
            self.assertIsNone(LEAKS.search(json.dumps(data['items'])), name)
        for name in ('moment', 'role', 'advance'):
            self.assertIsNone(LEAKS.search((LABELS/f'{name}_sheet.csv').read_text(encoding='ascii')), name)


class SeenSpans(unittest.TestCase):
    """Critic N3: the stretches the verification round studied are 'seen' for D3 and D4 and scored apart."""

    def test_seen_detectors(self):
        self.assertEqual(E._seen_dets('BC60', 30.0), {E.D3, E.D4})
        self.assertEqual(E._seen_dets('BC60+coach', 59.0), {E.D3, E.D4})
        self.assertEqual(E._seen_dets('BC303', 226.0), {E.D3, E.D4})
        self.assertEqual(E._seen_dets('BC303', 200.0), set())
        self.assertEqual(E._seen_dets('BC180', 30.0), set())
        self.assertEqual(E._comparison_detector('coach:D4.youth'), E.D4)
        self.assertEqual(E._comparison_detector('cover'), E.D1)
        self.assertEqual(E._comparison_detector('strength.D3'), E.D3)
        self.assertIsNone(E._comparison_detector('raise.senior'))


class Fix1Invariants(unittest.TestCase):
    """coaching_eval.fix1_problems (game-3 G1+) on real scene reports: none on the reports as made, and each rule
    flags a report doctored to break it."""

    def kinds(self, rep, obs, roles):
        return {k for k, _ in E.fix1_problems(rep, obs, roles)}

    def test_the_presser_and_the_cover(self):
        b, st, obs, roles = E.scene_tiers('defending', 'press_no_cover', prepared=FX.prepared)
        rep = E.scene_report('defending', 'press_no_cover', 4, prepared=FX.prepared)
        self.assertEqual(E.fix1_problems(rep, obs, roles), [])
        f = next((f for f in E.findings_of(rep) if f['observation'] == E.D1 and
                  any(c['type'] == 'actor_state' and c['facts'].get('player') for c in f['claims'])), None)
        self.assertIsNotNone(f)
        bad = copy.deepcopy(rep)
        g = next(x for x in E.findings_of(bad) if x['id'] == f['id'])
        c = next(c for c in g['claims'] if c['type'] == 'actor_state' and c['facts'].get('player'))
        c['facts']['player'] = 'A-99'
        self.assertIn('G1+presser', self.kinds(bad, obs, roles))

    def test_phase_words_with_the_phase_unknown(self):
        b, st, obs, roles = E.scene_tiers('covered_advance', 'cb_shift', prepared=FX.prepared)
        rep = E.scene_report('covered_advance', 'cb_shift', 4, prepared=FX.prepared)
        self.assertEqual(E.fix1_problems(rep, obs, roles), [])
        bad = copy.deepcopy(rep)
        f = E.findings_of(bad)[0]
        f.setdefault('context', {}).setdefault('phase', {})['state'] = 'unknown'
        f['text']['headline'] = 'Team shape with the ball'
        self.assertIn('G1+phase', self.kinds(bad, obs, roles))

    def test_a_strength_needs_its_demonstrated_run(self):
        b, st, obs, roles = E.scene_tiers('covered_advance', 'cb_shift', prepared=FX.prepared)
        rep = E.scene_report('covered_advance', 'cb_shift', 4, prepared=FX.prepared)
        self.assertTrue([f for f in rep['strengths'] if f['observation'] == E.D3])
        bad = copy.deepcopy(rep)
        bad['profile']['detectors'][E.D3]['params']['spellMinS'] = 60.0
        self.assertIn('G1+strength_run', self.kinds(bad, obs, roles))

    def test_a_check_not_assessed_names_its_reason(self):
        b, st, obs, roles = E.scene_tiers('covered_advance', 'cb_shift', prepared=FX.prepared)
        bad = copy.deepcopy(E.scene_report('covered_advance', 'cb_shift', 4, prepared=FX.prepared))
        bad['overview']['detectors'][0].update(status='not_assessed', reason=None)
        self.assertIn('G1+reason', self.kinds(bad, obs, roles))

    def test_level_1_someone_back_off_the_final_third_or_covered(self):
        from soccer.coaching import contextual as CX
        b, st, obs, roles = E.scene_tiers('unsupported_advance', 'final_third', prepared=FX.prepared)
        rep = E.scene_report('unsupported_advance', 'final_third', 1, prepared=FX.prepared)
        crit = [f for f in rep['needsWork']+rep['reviews'] if f['observation'] == E.D4
                and f['key'].split('|')[2] == 'advance']
        self.assertTrue(crit)
        self.assertEqual(E.fix1_problems(rep, obs, roles), [])
        ids = {t['opportunity'] for f in crit for t in f['timestamps']}

        def all_fail(o, params, scale, as_detector=None):    # the pre-round rule: judged wherever the ball is
            n = len(o['relations']['t'])
            return [-1]*n, [None]*n
        # fix round 3 (critic B-1 / N3): an exposed sample (an opponent in the vacated channel) fails wherever the ball
        # is, so "off the final third" is a problem only where nobody of theirs stood in the channel
        for change, edits in (('off the final third, not exposed', {'ballX': 40.0, 'opponentsInChannel': 0}),
                              ('covered', {'channelOccupied': True}),
                              ('final third, enough behind', {'opponentsInChannel': 0, 'behindN': 4})):
            r2 = copy.deepcopy(roles)
            for o in r2['opportunities']:
                if o['id'] in ids:
                    for field, value in edits.items():
                        o['relations'][field] = [value]*len(o['relations']['t'])
            with mock.patch.object(CX, 'sample_outcomes', all_fail):
                got = E.fix1_problems(rep, obs, r2)
            self.assertTrue([p for k, p in got if k == 'G1+someone_back'], (change, got))

    def test_levels_1_2_an_exposed_advance_fails_wherever_the_ball_is(self):
        # fix round 3 (critic B-1, COACH F-1): none_behind's centre-back carries the ball into midfield, the channel
        # left open with an opponent in it: levels 1-2 criticise it, and the amended G1+someone_back accepts it; the
        # same samples with the opponent removed (not exposed, the ball in midfield) are refused
        from soccer.coaching import contextual as CX
        b, st, obs, roles = E.scene_tiers(*SCENE, prepared=FX.prepared)
        for lv in (1, 2):
            rep = E.scene_report(*SCENE, lv, prepared=FX.prepared)
            crit = [f for f in rep['needsWork']+rep['reviews'] if f['observation'] in (E.D3, E.D4)]
            self.assertTrue(crit, lv)
            self.assertEqual(E.fix1_problems(rep, obs, roles), [], lv)
            ids = {t['opportunity'] for f in crit for t in f['timestamps']}
            r2 = copy.deepcopy(roles)
            for o in r2['opportunities']:
                if o['id'] in ids:
                    o['relations']['opponentsInChannel'] = [0]*len(o['relations']['t'])

            def outcomes(o, params, scale, as_detector=None, real=CX.sample_outcomes):
                x = copy.deepcopy(o)
                x['relations']['opponentsInChannel'] = [1]*len(o['relations']['t'])   # judged as exposed
                return real(x, params, scale, as_detector=as_detector)
            with mock.patch.object(CX, 'sample_outcomes', outcomes):
                got = [p for k, p in E.fix1_problems(rep, obs, r2) if k == 'G1+someone_back']
            self.assertTrue(got and 'outside the final third' in got[0], (lv, got))


class Fix2MomentsInvariants(unittest.TestCase):
    """coaching_eval.fix2_problems (game-3 G1+ of fix round 2, LEAD decisions (b) and (d)) on real scene reports with a
    'moments worth a look' review (small_field_scale/declared: D5, 7 clear of the 11 judged in 5 spells at levels 4
    and 5 since fix round 3's settled-play gates; 10 in 8 before): none on the reports as made, and each rule flags a
    report (or obs file) doctored to break it."""

    SCENE = ('small_field_scale', 'declared')

    @classmethod
    def setUpClass(cls):
        cls.b, cls.st, cls.obs, cls.roles = E.scene_tiers(*cls.SCENE, prepared=FX.prepared)
        cls.rep4 = E.scene_report(*cls.SCENE, 4, prepared=FX.prepared)
        cls.rep5 = E.scene_report(*cls.SCENE, 5, prepared=FX.prepared)

    def mw(self, rep):
        return next(f for f in rep['reviews'] if E.is_moments_review(f))

    def kinds(self, rep, obs=None, roles=None):
        return {k for k, _ in E.fix2_problems(rep, obs or self.obs, roles or self.roles)}

    def test_the_reports_as_made(self):
        for rep in (self.rep4, self.rep5):
            self.assertEqual(len([f for f in rep['reviews'] if E.is_moments_review(f)]), 1)
            self.assertEqual(E.fix2_problems(rep, self.obs, self.roles), [])
            self.assertEqual(E.report_problems(rep, self.obs, self.roles), [])

    def test_the_form(self):
        for change in ('verdict', 'priority', 'drill', 'level', 'guardrail', 'twin'):
            bad = copy.deepcopy(self.rep4)
            f = self.mw(bad)
            if change == 'verdict':
                f['verdict'] = 'work_on'
            elif change == 'priority':
                f['priority'] = 1
            elif change == 'drill':
                f['drills'] = ['some-drill']
            elif change == 'level':
                bad['profile']['level'] = 2
            elif change == 'guardrail':
                f['guardrails']['identity'] = 'fail'
            else:
                twin = copy.deepcopy(f)
                twin.update(id='fnd-0000000twin1', verdict='strength', limitations=[])
                bad['strengths'].append(twin)
            self.assertIn('G1+moments_form', self.kinds(bad), change)

    def test_the_cap(self):
        bad = copy.deepcopy(self.rep4)
        extra = copy.deepcopy(self.mw(bad))
        extra['id'] = 'fnd-0000000extra'
        bad['reviews'].append(extra)
        self.assertIn('G1+moments_cap', self.kinds(bad))

    def test_the_counts_are_recomputed(self):
        for key, delta in (('clearMoments', -1), ('decidedMoments', 1), ('failingMoments', -1), ('clearSpells', 1)):
            bad = copy.deepcopy(self.rep4)
            self.mw(bad)['measurements'][key] += delta
            self.assertIn('G1+moments_counts', self.kinds(bad), key)
        # an opportunity whose identity is not high is no clear failure: the stated counts no longer hold
        f = self.mw(self.rep4)
        roles, obs = copy.deepcopy(self.roles), copy.deepcopy(self.obs)
        for o in obs['opportunities']+roles['opportunities']:
            if o['id'] == f['timestamps'][0]['opportunity']:
                o['quality']['identity'] = 'medium'
        got = self.kinds(self.rep4, obs, roles)
        self.assertIn('G1+moments_counts', got)
        self.assertIn('G1+moments_clear', got)

    def test_every_cited_moment_is_a_clear_failure(self):
        f = self.mw(self.rep4)
        obs, roles = copy.deepcopy(self.obs), copy.deepcopy(self.roles)
        ts = f['timestamps'][0]
        o = next(o for o in obs['opportunities']+roles['opportunities'] if o['id'] == ts['opportunity'])
        m = next(m for m in obs['moments']+roles['moments'] if m['id'] == ts['evidence'])
        o['relations']['primaryNamed'] = [True]*len(o['relations']['t'])
        o['relations']['primaryNamed'][m['sample']] = False        # the moment's player not named
        self.assertIn('G1+moments_clear', self.kinds(self.rep4, obs, roles))
        bad = copy.deepcopy(self.rep4)
        other = next(x for x in self.obs['moments']+self.roles['moments'] if x['opportunity'] not in f['opportunities'])
        self.mw(bad)['timestamps'][0].update(evidence=other['id'], opportunity=other['opportunity'], t=other['t'])
        self.assertIn('G1+moments_clear', self.kinds(bad))

    def test_the_count_and_the_share_are_stated(self):
        from soccer.coaching import language as LG
        # the counts as the report states them (fix round 3: 7 clear of the 11 judged in 5 spells, against 10 in 8
        # before the settled-play gates; the test reads them, the recount above checks them)
        m4, m5 = self.mw(self.rep4)['measurements'], self.mw(self.rep5)['measurements']
        of4 = f'{m4["clearMoments"]} of the {m4["decidedMoments"]}'
        pct = f'about {round(100*m5["clearMoments"]/m5["decidedMoments"])}%'
        for rep, old, new in ((self.rep4, of4, f'{m4["clearMoments"]} of the {self.mw(self.rep4)["aggregate"]["opportunities"]}'),
                              (self.rep4, of4, 'most of the'), (self.rep5, pct, 'about 28%'), (self.rep5, f' ({pct})', '')):
            bad = copy.deepcopy(rep)
            t = self.mw(bad)['text']
            self.assertIn(old, t['explanation'])
            t['explanation'] = t['explanation'].replace(old, new)
            self.assertIn('G1+moments_text', self.kinds(bad), (old, new))
        # fix round 3 (FACTS F-4): the failing moments that were not clear are counted ("In 4 more it fell short less
        # clearly"); a text that drops or miscounts them is flagged
        more = m4['failingMoments']-m4['clearMoments']
        self.assertGreater(more, 0)
        for new in ('', f'In {more+1} more it fell short less clearly; those are not cited.'):
            bad = copy.deepcopy(self.rep4)
            t = self.mw(bad)['text']
            t['explanation'] = re.sub(r'[^.]*less clearly[^.]*\.\s*', new+(' ' if new else ''), t['explanation'])
            self.assertIn('G1+moments_text', self.kinds(bad), new)
        bad = copy.deepcopy(self.rep4)
        t = self.mw(bad)['text']
        t['explanation'] = t['explanation'].replace(LG.fmt_time(self.mw(bad)['timestamps'][1]['t']), 'later')
        self.assertIn('G1+moments_text', self.kinds(bad))
        bad = copy.deepcopy(self.rep4)
        self.mw(bad)['text']['headline'] = 'Possible issue to review: back-line spacing'
        self.assertIn('G1+moments_text', self.kinds(bad))

    def test_the_base_rate_test(self):
        # fix round 3 (LEAD decision (2)): the clear spells must be significantly above the level's base share; a
        # review stated with another base or tail, or one whose base makes its spells unremarkable, is flagged
        for key, value in (('baseShare', .5), ('baseTail', .2)):
            bad = copy.deepcopy(self.rep4)
            self.mw(bad)['measurements'][key] = value
            self.assertIn('G1+moments_base', self.kinds(bad), key)
        bad = copy.deepcopy(self.rep4)
        bad['profile']['detectors'][self.mw(bad)['observation']]['params']['baseFailShare'] = .9
        self.assertIn('G1+moments_base', self.kinds(bad))
        bad = copy.deepcopy(self.rep4)
        bad['profile']['detectors'][self.mw(bad)['observation']]['params'].pop('baseFailShare')
        self.assertIn('G1+moments_base', self.kinds(bad), 'no measured base: no review')
        # the independent tail (scipy) equals assess's
        from soccer.coaching import assess as AS
        for k, n, p in ((5, 11, .048), (3, 62, .212), (4, 4, .654), (1, 11, .004), (0, 9, .3), (12, 11, .5)):
            self.assertAlmostEqual(E.binomial_tail(k, n, p), AS.binom_tail(k, n, p), places=12)

    def test_the_settled_play_and_marking_gates(self):
        # fix round 3 (LEAD decision (2), FACTS F-3, critic N1): a clear moment lies in settled play, away from a loose
        # or contested ball; the recount drops a cited moment the obs file puts 1.5 s after a dead ball, or 0.5 s from
        # a contested ball, and the stated counts no longer hold
        f = self.mw(self.rep4)
        ts = f['timestamps'][0]
        for change in ('dead ball', 'contested'):
            obs, roles = copy.deepcopy(self.obs), copy.deepcopy(self.roles)
            o = next(o for o in obs['opportunities']+roles['opportunities'] if o['id'] == ts['opportunity'])
            if change == 'dead ball':
                t = float(ts['t'])
                obs['timeline']['phaseSpans'].append({'t0': t-3.0, 't1': t-1.5, 'state': 'dead'})
                obs['timeline']['phaseSpans'].sort(key=lambda s: s['t0'])
            else:
                o['relations']['contestS'] = [.5]*len(o['relations']['t'])
            got = self.kinds(self.rep4, obs, roles)
            self.assertIn('G1+moments_clear', got, change)
        # the gates themselves, on hand-made samples with the profile's values (level 4)
        p = self.rep4['profile']['detectors'][E.D5]['params']
        R = {'t': [10.0], 'contestS': [None], 'lineSize': [4], 'lineStdM': [3.9], 'gapOppFreeM': [4.5]}
        idx = E.unsettled_intervals({'step': .2, 'phaseSpans': [{'t0': 0.0, 't1': 8.1, 'state': 'opp'},
                                                                 {'t0': 8.2, 't1': 9.0, 'state': 'dead'},
                                                                 {'t0': 9.1, 't1': 20.0, 'state': 'opp'}]},
                                    p['momentsSettleS'])
        self.assertTrue(E._unsettled(idx, 10.0))           # 1.0 s after the dead ball
        self.assertFalse(E._unsettled(idx, 11.1))          # 2.1 s after it
        self.assertTrue(E._sample_gates(E.D5, R, 0, 'lateral_gap', p, None, 11))
        for key, value in (('contestS', .9), ('lineSize', 3), ('lineStdM', 4.1), ('gapOppFreeM', 3.9)):
            R2 = dict(R, **{key: [value]})
            self.assertFalse(E._sample_gates(E.D5, R2, 0, 'lateral_gap', p, None, 11), key)
        self.assertTrue(E._sample_gates(E.D5, dict(R, lineSize=[3]), 0, 'lateral_gap', p, None, 7))   # 11v11 only
        self.assertTrue(E._sample_gates(E.D5, dict(R, gapOppFreeM=[3.9]), 0, 'dm_gap', p, None, 11))   # lateral only

    def test_no_verdict_words(self):
        for field, text in (('action', 'The back line should close the gap.'), ('short', 'Poor back-line spacing'),
                            ('explanation', 'Work on the gaps. '), ('action', 'Try this drill next time.')):
            bad = copy.deepcopy(self.rep4)
            t = self.mw(bad)['text']
            t[field] = text+(t[field] if field == 'explanation' else '')
            self.assertIn('G1+moments_words', self.kinds(bad), text)
        bad = copy.deepcopy(self.rep4)
        bad['overview']['text'] += ' There is one possible issue to review.'
        self.assertIn('G1+moments_words', self.kinds(bad), 'the overview counts it as a possible issue')


class Fix3Invariants(unittest.TestCase):
    """coaching_eval.fix3_problems (game-3 G1+ of fix round 3, critic B-3 validators): none on the scene reports as
    made, and each rule flags a report doctored to break it (the moment's actor and gap pair, its third, the aggregate
    and a strength counted over all its opportunities, a count behind the ball without its margin, TEXT's report-level
    invariants, a drill from far below the level, a repetition word in a drill, an optional drill comparing with a
    match that decided nothing of its check)."""

    def tiers(self, scene, variant):
        return E.scene_tiers(scene, variant, prepared=FX.prepared)[2:]

    def kinds(self, rep, scene, variant):
        obs, roles = self.tiers(scene, variant)
        return {k for k, _ in E.fix3_problems(rep, obs, roles)}

    def test_the_reports_as_made(self):
        for scene, variant, lv in (('unsupported_advance', 'none_behind', 4), ('small_field_scale', 'declared', 4),
                                   ('unsupported_advance', 'final_third', 3), ('covered_advance', 'cb_shift', 5),
                                   ('defending', 'press_cover', 6), ('amateur_coach_spans', 'spans', 1)):
            rep = E.scene_report(scene, variant, lv, prepared=FX.prepared)
            obs, roles = self.tiers(scene, variant)
            self.assertEqual(E.fix3_problems(rep, obs, roles), [], (scene, variant, lv))

    def test_the_moments_own_players_and_third(self):
        rep = E.scene_report('unsupported_advance', 'none_behind', 4, prepared=FX.prepared)
        f0 = next(f for f in rep['needsWork'] if f['observation'] == E.D3)
        self.assertTrue(E.ID_RE.fullmatch(f0['measurements']['actorId']))
        for change in ('actor', 'primary', 'zone'):
            bad = copy.deepcopy(rep)
            f = next(x for x in bad['needsWork'] if x['id'] == f0['id'])
            if change == 'actor':
                f['measurements']['actorId'] = 'A-99'
            elif change == 'primary':
                f['players']['primary'] = ['A-99']
            else:
                f['measurements']['momentZone'] = 'final_third' if f['measurements']['momentZone'] != 'final_third' \
                    else 'own_third'
            want = 'G1+moment_zone' if change == 'zone' else 'G1+moment_actor'
            self.assertIn(want, self.kinds(bad, 'unsupported_advance', 'none_behind'), change)
        rep = E.scene_report('small_field_scale', 'declared', 4, prepared=FX.prepared)
        mw = next(f for f in rep['reviews'] if E.is_moments_review(f))
        self.assertIn(mw['measurements']['actorId'], (mw['measurements']['gapLeftId'], mw['measurements']['gapRightId']))
        for change in ('pair', 'supporting'):
            bad = copy.deepcopy(rep)
            f = next(x for x in bad['reviews'] if x['id'] == mw['id'])
            if change == 'pair':
                f['measurements']['gapRightId'] = 'A-99'
            else:
                f['players']['supporting'] = ['A-99']+f['players']['supporting'][1:]
            self.assertIn('G1+moment_actor', self.kinds(bad, 'small_field_scale', 'declared'), change)

    def test_the_aggregate_and_a_strength_over_all_opportunities(self):
        rep = E.scene_report('small_field_scale', 'declared', 4, prepared=FX.prepared)
        s = next(f for f in rep['strengths'] if f['observation'] == E.D1)
        a = s['aggregate']
        self.assertGreater(a['unclear'], 0)
        bad = copy.deepcopy(rep)
        next(x for x in bad['strengths'] if x['id'] == s['id'])['aggregate']['unclear'] += 1
        self.assertIn('G1+aggregate', self.kinds(bad, 'small_field_scale', 'declared'))
        bad = copy.deepcopy(rep)
        t = next(x for x in bad['strengths'] if x['id'] == s['id'])['text']
        t['explanation'] += f' In {a["demonstrated"]} of the {a["opportunities"]} defending moments the ball was pressed.'
        self.assertIn('G1+aggregate', self.kinds(bad, 'small_field_scale', 'declared'))

    def test_a_count_behind_the_ball_states_its_margin(self):
        rep = E.scene_report('unsupported_advance', 'final_third', 3, prepared=FX.prepared)
        f = next(x for x in E.findings_of(rep) if 'at least 10 m behind the ball' in (x['text'] or {}).get(
            'explanation', ''))
        bad = copy.deepcopy(rep)
        t = next(x for x in E.findings_of(bad) if x['id'] == f['id'])['text']
        t['explanation'] = t['explanation'].replace('at least 10 m behind the ball', 'behind the ball')
        self.assertIn('G1+behind_margin', self.kinds(bad, 'unsupported_advance', 'final_third'))

    def test_the_report_level_text(self):
        rep = E.scene_report('covered_advance', 'cb_shift', 5, prepared=FX.prepared)
        bad = copy.deepcopy(rep)
        bad['nextObjective'] = {'skill': 'pressure_and_cover', 'fromMatch': False, 'finding': None,
                                'text': "From the level's objectives, not from this match: Pressure and cover."}
        self.assertIn('G1+report_text', self.kinds(bad, 'covered_advance', 'cb_shift'))

    def test_the_drill_rules(self):
        from soccer.coaching import drills as DR, schema as S
        lib = DR.load_library()
        rep = E.scene_report('unsupported_advance', 'none_behind', 4, prepared=FX.prepared)
        pick = next(p for p in rep['drills'] if p['finding'])
        low = next(d for d in lib['drills'] if max(d['development_level']) <= 2)    # (criterion 12 checks the check)
        bad = copy.deepcopy(rep)
        next(p for p in bad['drills'] if p['drill'] == pick['drill'])['drill'] = low['drill_id']
        self.assertIn('G1+drills', self.kinds(bad, 'unsupported_advance', 'none_behind'), 'two levels down')
        bad = copy.deepcopy(rep)
        p = next(p for p in bad['drills'] if p['drill'] == pick['drill'])
        p['fields']['why'] = (p['fields'].get('why') or '')+' This happened often.'
        self.assertIn('G1+drills', self.kinds(bad, 'unsupported_advance', 'none_behind'), 'repetition word')
        # an optional drill comparing with "this match" for a check the reported findings decided nothing of
        found = None
        for scene, variant in (('covered_advance', 'cb_shift'), ('professional', 'default'), ('defending', 'press_cover'),
                               ('covered_buildup', 'default'), ('low_block', 'default')):
            for lv in (3, 4, 5, 6):
                r = E.scene_report(scene, variant, lv, prepared=FX.prepared)
                opt = [p for p in r['drills'] if p['optional'] and p['finding'] is None]
                if opt and r['nextObjective'].get('skill'):
                    found = (scene, variant, r, opt[0])
                    break
            if found:
                break
        self.assertIsNotNone(found)
        scene, variant, r, opt = found
        bad = copy.deepcopy(r)
        det = S.SKILLS[bad['nextObjective']['skill']]
        for k in E.FINDING_SECTIONS:
            bad[k] = [f for f in bad[k] if f['observation'] != det]
        p = next(x for x in bad['drills'] if x['drill'] == opt['drill'])
        p['fields']['progress'] = 'Look for the same picture in more of these moments than in this match.'
        self.assertIn('G1+drills', self.kinds(bad, scene, variant), (scene, variant))


class StatsBombReader(unittest.TestCase):
    """A hand-made match in StatsBomb's layout (no real data): the distances, the 'certain' rule and the skips."""

    def test_distances_certainty_and_malformed_input(self):
        events = [
            {'id': 'e1', 'type': {'name': 'Pass'}, 'under_pressure': True, 'location': [60.0, 40.0]},
            {'id': 'e2', 'type': {'name': 'Carry'}, 'location': [100.0, 40.0]},
            {'id': 'e3', 'type': {'name': 'Pressure'}, 'location': [10.0, 10.0]},       # not a carrier event
            {'id': 'e4', 'type': {'name': 'Shot'}, 'location': ['x', 1]},                # malformed, no frame actor
            'not an event',
        ]
        frames = [
            {'event_uuid': 'e1', 'visible_area': [40, 20, 80, 20, 80, 60, 40, 60, 40, 20],
             'freeze_frame': [{'location': [60.0, 40.0], 'actor': True, 'teammate': True, 'keeper': False},
                              {'location': [63.0, 40.0], 'actor': False, 'teammate': False, 'keeper': False},
                              {'location': [61.0, 40.0], 'actor': False, 'teammate': False, 'keeper': True},
                              {'location': [60.5, 40.0], 'actor': False, 'teammate': True, 'keeper': False},
                              {'location': 'bad'}]},
            {'event_uuid': 'e2', 'visible_area': [90, 35, 120, 35, 120, 45, 90, 45, 90, 35],
             'freeze_frame': [{'location': [100.0, 40.0], 'actor': True, 'teammate': True},
                              {'location': [100.0, 52.0], 'actor': False, 'teammate': False, 'keeper': False}]},
            {'event_uuid': 'e4', 'freeze_frame': [{'location': [1, 2], 'teammate': False}]},
        ]
        with tempfile.TemporaryDirectory() as d:
            Path(d, 'events_1.json').write_text(json.dumps(events), encoding='utf-8')
            Path(d, 'three-sixty_1.json').write_text(json.dumps(frames), encoding='utf-8')
            Path(d, 'events_x.json').write_text('[]', encoding='utf-8')         # not a match id: ignored
            rows = E.statsbomb_rows(d)
            res = E.statsbomb_check(d)
        self.assertEqual([r['type'] for r in rows], ['Pass', 'Carry'])
        # 3 StatsBomb units along = 3 x 105/120 m; the keeper and the team-mate do not count
        self.assertAlmostEqual(rows[0]['d'], 3*105/120, delta=.01)
        self.assertTrue(rows[0]['pressured'])
        self.assertTrue(rows[0]['certain'])
        # 12 units across = 12 x 68/80 m, beyond the visible area's edge (5 units away): not certain
        self.assertAlmostEqual(rows[1]['d'], 12*68/80, delta=.01)
        self.assertFalse(rows[1]['certain'])
        self.assertAlmostEqual(rows[1]['x'], 100*105/120, delta=.01)
        t5 = res['subsets']['all']['thresholds']['5']
        self.assertEqual((t5['pressuredWithin'], t5['notPressuredWithin']), (1.0, 0.0))
        self.assertEqual(res['subsets']['defendingThird']['events'], 1)
        self.assertTrue(math.isclose(res['subsets']['all']['pressured']['p50'], 2.62, abs_tol=.01))


class SummaryTable(unittest.TestCase):
    def test_the_summary_reads_a_runs_folder(self):
        b, st, obs, roles = E.scene_tiers(*SCENE, prepared=FX.prepared)
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            reps = {}
            for tag in E.TABLE_TAGS:
                lv = int(tag[1])
                reps[tag] = E.interpret(obs, roles, st, 'A', lv)
                E.write_json(out/'SYN'/'A'/f'{tag}.json', reps[tag])
            summary = {'generated': 'test', 'notes': [], 'runs': {'SYN': {
                'coachSpans': 0, 'coachSpansSynthetic': False, 'remeasured': 0, 'teams': {'A': {
                    'phase': obs['timeline']['phaseSeconds'], 'opportunities': len(obs['opportunities']),
                    'possession': {k: obs['timeline']['phaseQuality'][k] for k in ('available', 'level', 'coverage')},
                    'roleOpportunities': len(roles['opportunities']), 'moments': 0, 'problems': [],
                    'reports': {tag: E.summary_row(r) for tag, r in reps.items()}}}}}}
            E.write_json(out/'summary.json', summary)
            text = E.runs_markdown(out)
        rows = [ln for ln in text.splitlines() if re.match(r'[|] L[1-7]', ln)]
        self.assertEqual(len(rows), 2*len(E.TABLE_TAGS))       # the run's table and the totals per level
        l4 = E.report_stats(reps['L4-U14-unknown'])
        self.assertGreaterEqual(l4['mix']['W'], 1)
        self.assertIn(f"| L4-U14-unknown | {l4['findings']} |", text)


if __name__ == '__main__':
    unittest.main()
