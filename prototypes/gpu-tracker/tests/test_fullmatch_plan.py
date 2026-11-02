"""Full Match Mode's segment plan (fullmatch/plan.py) and match manifest (fullmatch/manifest.py), without a video or a GPU.

The plan: a 90-minute match at 25 fps is 30 segments of exactly 180 s; the CAN-ISL source (144,792 frames at
25.00008 fps) is 33 equal segments; a 29.97 fps match keeps every segment at or under 180 s; the owned frames of a
period follow each other without a gap or an overlap and every frame of a period has exactly one owner; the processed
frames add PAD_S of context on each side but never leave their period (nothing crosses halftime); a period just over
one segment becomes two halves rather than a full segment and a sliver; bad ranges are refused.
The manifest: a round trip; atomic writes leave no temporary files; resume compatibility (source, frames, fps,
processing settings; the plan's start/length and the calibration keyframes are not settings); which segments a
calibration change reaches; artifact verification (complete, untruncated, unmodified, covering the owned frames) and
the reconciliation a resume runs (done but broken -> pending, interrupted -> pending, a verified orphan -> done).
The stored results: a compressed one reads back as the run wrote it, verifies, is the same bytes every time, and is
caught when cut short or changed; the original stays when the copy does not read back; an artifact from before the
compression (a plain result.json, a segment.json naming no file) still verifies and reads. The disk estimate covers
what the whole CAN-ISL match took with its results compressed, without being several times that.

Run:  python -m unittest test_fullmatch_plan          (from this directory)
"""
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fullmatch import manifest as mf, plan, system  # noqa: E402

CAN_ISL = {'frames': 144792, 'fps': 25.000080239880965}


def owners_ok(test, segments, f0, f1):
    """Every frame of [f0, f1) has exactly one owner (checked at every boundary and a stride inside)."""
    probes = set(range(f0, f1, 997)) | {s.own0 for s in segments} | {s.own1-1 for s in segments} | {f0, f1-1}
    for f in sorted(probes):
        owners = [s for s in segments if s.own0 <= f < s.own1]
        test.assertEqual(len(owners), 1, f)
        test.assertIs(plan.owner(segments, f), owners[0])


class PlanTests(unittest.TestCase):
    def test_a_90_minute_match_at_25_fps_is_30_segments_of_180_seconds(self):
        frames = 90*60*25
        segments = plan.plan(frames, 25.0, plan.periods(frames/25.0))
        self.assertEqual(len(segments), 30)
        self.assertTrue(all(s.own1-s.own0 == 4500 for s in segments))
        self.assertEqual((segments[0].own0, segments[-1].own1), (0, frames))
        pad = round(plan.PAD_S*25)
        self.assertEqual((segments[0].proc0, segments[0].proc1), (0, 4500+pad))
        self.assertEqual((segments[5].proc0, segments[5].proc1), (5*4500-pad, 6*4500+pad))
        self.assertEqual(segments[-1].proc1, frames)
        owners_ok(self, segments, 0, frames)
        self.assertEqual(len(plan.neighbours(segments)), 29)
        self.assertEqual(plan.processed_frames(segments), frames+2*pad*29)

    def test_the_can_isl_source_is_33_equal_segments(self):
        fps, frames = CAN_ISL['fps'], CAN_ISL['frames']
        segments = plan.plan(frames, fps, plan.periods(frames/fps))
        self.assertEqual(len(segments), 33)
        lengths = [s.own1-s.own0 for s in segments]
        self.assertLessEqual(max(lengths)-min(lengths), 1)
        self.assertLessEqual(max(lengths)/fps, plan.SEGMENT_S+1e-6)
        owners_ok(self, segments, 0, frames)

    def test_a_29_97_fps_match_keeps_every_segment_at_most_180_seconds(self):
        fps = 30000/1001
        frames = int(round(95.5*60*fps))
        segments = plan.plan(frames, fps, plan.periods(frames/fps))
        self.assertEqual(len(segments), 32)
        self.assertTrue(all((s.own1-s.own0)/fps <= plan.SEGMENT_S+1/fps for s in segments))
        owners_ok(self, segments, 0, frames)

    def test_halftime_is_a_gap_nothing_crosses(self):
        fps, frames = 25.0, 100*60*25
        periods = plan.periods(frames/fps, start=60, end=98*60, halftime=(48*60, 63*60))
        self.assertEqual(periods, [(60.0, 2880.0), (3780.0, 5880.0)])
        segments = plan.plan(frames, fps, periods)
        first = [s for s in segments if s.period == 0]
        second = [s for s in segments if s.period == 1]
        self.assertEqual((first[0].own0, first[-1].own1), (1500, 72000))
        self.assertEqual((second[0].own0, second[-1].own1), (94500, 147000))
        self.assertEqual(first[-1].proc1, 72000)     # no context into the break
        self.assertEqual(second[0].proc0, 94500)
        self.assertIsNone(plan.owner(segments, 80000))   # in the break
        self.assertIsNone(plan.owner(segments, 100))     # before the match
        self.assertIsNone(plan.owner(segments, 147000))  # after it
        self.assertNotIn((first[-1], second[0]), plan.neighbours(segments))
        owners_ok(self, first, 1500, 72000)
        owners_ok(self, second, 94500, 147000)

    def test_each_half_can_have_its_own_range(self):
        periods = plan.periods(6000, ranges=[(120, 2900), (3800, 5900)])
        segments = plan.plan(150000, 25.0, periods)
        self.assertEqual({s.period for s in segments}, {0, 1})
        self.assertEqual(segments[0].own0, 3000)
        self.assertEqual(segments[-1].own1, 147500)

    def test_a_period_just_over_one_segment_is_two_halves_not_a_sliver(self):
        segments = plan.plan(181*25, 25.0, plan.periods(181.0))
        self.assertEqual([s.own1-s.own0 for s in segments], [2262, 2263])

    def test_a_short_match_is_one_segment_without_context(self):
        segments = plan.plan(60*25, 25.0, plan.periods(60.0))
        self.assertEqual(len(segments), 1)
        self.assertEqual((segments[0].proc0, segments[0].proc1), (0, 1500))

    def test_bad_ranges_are_refused(self):
        with self.assertRaises(ValueError):
            plan.periods(5400, start=100, end=5000, halftime=(5100, 5200))   # break outside the match
        with self.assertRaises(ValueError):
            plan.periods(5400, ranges=[(0, 2800), (2700, 5400)])              # overlapping halves
        with self.assertRaises(ValueError):
            plan.periods(5400, ranges=[(0, 3)])                               # a slip of the marker
        with self.assertRaises(ValueError):
            plan.periods(5400, start=0, end=6000)                             # past the end of the video
        with self.assertRaises(ValueError):
            plan.periods(5400, ranges=[(0, 10), (20, 30), (40, 50), (60, 70), (80, 90)])
        with self.assertRaises(ValueError):
            plan.plan(0, 25.0, [(0, 10)])

    def test_check_refuses_gaps_overlaps_and_context_outside_the_period(self):
        S = plan.Segment
        with self.assertRaises(ValueError):
            plan.check([S(0, 0, 0, 100, 0, 110), S(1, 0, 101, 200, 90, 200)])   # a gap
        with self.assertRaises(ValueError):
            plan.check([S(0, 0, 0, 100, 0, 110), S(1, 0, 99, 200, 90, 200)])    # an overlap
        with self.assertRaises(ValueError):
            plan.check([S(0, 0, 0, 100, 0, 120), S(1, 0, 100, 110, 90, 110)])   # context past the period's end
        with self.assertRaises(ValueError):
            plan.check([S(0, 0, 10, 100, 5, 100)])                                 # context before the period
        self.assertTrue(plan.check([S(0, 0, 0, 100, 0, 110), S(1, 0, 100, 200, 90, 200)]))

    def test_segment_json_round_trip_and_keys(self):
        s = plan.plan(10000, 25.0, plan.periods(400.0))[1]
        data = s.to_json(25.0)
        self.assertEqual(plan.Segment.from_json(json.loads(json.dumps(data))), s)
        self.assertEqual(data['key'], f's001-f{s.own0}-{s.own1}')
        self.assertAlmostEqual(data['ownStart'], s.own0/25.0)


def fake_result(first, last, status='complete'):
    return {'status': status, 'frames': [{'frame': f, 'time': f/25.0, 'people': []} for f in range(first, last+1)]}


def write_artifact(folder, result, compressed=False):
    folder.mkdir(parents=True)
    (folder/'result.json').write_text(json.dumps(result), encoding='utf-8')
    (folder/'identities.json').write_text('{}', encoding='utf-8')
    (folder/'registration.json').write_text('{}', encoding='utf-8')
    stored = mf.compress_result(folder)[0] if compressed else folder/'result.json'
    (folder/mf.SIDECAR).write_text(json.dumps(mf.sidecar(result, stored)), encoding='utf-8')


SETTINGS = {'imageSize': 1280, 'ballTiles': False, 'boundaries': [], 'calibrations': [{'time': 30.0, 'points': []}],
            'fieldFilter': True, 'touchlineMargin': .35, 'maxPerTeam': 11, 'debug': True, 'start': 0, 'seconds': 60}
VIDEO = {'name': 'm.mp4', 'width': 1920, 'height': 1080, 'fps': 25.0, 'frames': 20000, 'duration': 800.0}


class ManifestTests(unittest.TestCase):
    def setUp(self):
        self.folder = Path(tempfile.mkdtemp())/'matches'/'m1'
        self.segments = plan.plan(VIDEO['frames'], 25.0, plan.periods(800.0))
        self.m = mf.Manifest.create(self.folder, {'key': 'k'*64, 'path': 'x.mp4', 'bytes': 1}, VIDEO, plan.periods(800.0),
                                    self.segments, SETTINGS, {'mode': 'auto', 'count': 3}, plan.PAD_S, plan.SEGMENT_S)

    def test_round_trip_and_no_temporary_files(self):
        loaded = mf.Manifest.load(self.folder/'manifest.json')
        self.assertEqual(loaded.segments(), self.segments)
        self.assertEqual(loaded.data['state'], 'planned')
        self.assertEqual(loaded.counts()['pending'], len(self.segments))
        self.m.update_segment(2, state='running', attempts=1)
        self.m.set_state('running')
        again = mf.Manifest.load(self.folder/'manifest.json')
        self.assertEqual((again.entry(2)['state'], again.data['state']), ('running', 'running'))
        self.assertEqual([p.name for p in self.folder.iterdir() if p.name.endswith('.tmp')], [])
        with self.assertRaises(ValueError):
            self.m.set_state('bogus')

    def test_compatibility(self):
        self.assertIsNone(self.m.compatible('k'*64, 20000, 25.0, SETTINGS | {'start': 99, 'seconds': 5, 'calibrations': []}))
        self.assertIn('different file', self.m.compatible('j'*64, 20000, 25.0, SETTINGS))
        self.assertIn('frame rate', self.m.compatible('k'*64, 20001, 25.0, SETTINGS))
        self.assertIn('settings', self.m.compatible('k'*64, 20000, 25.0, SETTINGS | {'imageSize': 1536}))

    def test_a_calibration_change_reaches_only_nearby_segments(self):
        old = [{'time': 30.0, 'points': [1]}]
        new = old+[{'time': 400.0, 'points': [2]}]
        hit = mf.calibration_changes(old, new, self.segments, 25.0)
        reach = [s.index for s in self.segments if s.proc0/25-mf.ROLL_SECONDS <= 400 <= s.proc1/25+mf.ROLL_SECONDS]
        self.assertEqual(hit, reach)
        self.assertTrue(1 <= len(hit) <= 2)
        self.assertEqual(mf.calibration_changes(old, old, self.segments, 25.0), [])

    def test_artifact_verification(self):
        s = self.segments[1]
        good = self.folder/'segments'/s.key
        write_artifact(good, fake_result(s.proc0, s.proc1-1))
        self.assertIsNone(mf.verify_artifact(good, s))
        # cut short: the size no longer matches segment.json
        with (good/'result.json').open('ab') as out:
            out.write(b' ')
        self.assertIn('size', mf.verify_artifact(good, s))
        # same size, different content: the checksum catches it
        text = (good/'result.json').read_text(encoding='utf-8')[:-1]
        (good/'result.json').write_text(text.replace('"complete"', '"complexe"'), encoding='utf-8')
        self.assertIn('checksum', mf.verify_artifact(good, s))
        self.assertIsNone(mf.verify_artifact(good, s, deep=False))   # the quick check trusts a same-size file (resume runs deep)
        short = self.folder/'segments'/'short'
        write_artifact(short, fake_result(s.own0+10, s.own1))
        self.assertIn('covers frames', mf.verify_artifact(short, s))
        cancelled = self.folder/'segments'/'cancelled'
        write_artifact(cancelled, fake_result(s.proc0, s.proc1-1, status='cancelled'))
        self.assertIn('cancelled', mf.verify_artifact(cancelled, s))
        gappy = self.folder/'segments'/'gappy'
        result = fake_result(s.proc0, s.proc1-1)
        del result['frames'][50]
        write_artifact(gappy, result)
        self.assertIn('skips frames', mf.verify_artifact(gappy, s))
        unfinished = self.folder/'segments'/'unfinished'
        write_artifact(unfinished, fake_result(s.proc0, s.proc1-1))
        (unfinished/mf.SIDECAR).unlink()
        self.assertIn('segment.json', mf.verify_artifact(unfinished, s))

    def test_a_compressed_result_reads_back_verifies_and_is_caught_when_damaged(self):
        s = self.segments[1]
        folder = self.folder/'segments'/s.key
        result = fake_result(s.proc0, s.proc1-1)
        write_artifact(folder, result, compressed=True)
        self.assertEqual(sorted(p.name for p in folder.iterdir()),
                         ['identities.json', 'registration.json', 'result.json.gz', mf.SIDECAR])   # no plain copy, no temporary
        self.assertEqual(mf.read_result(folder), result)
        self.assertIsNone(mf.verify_artifact(folder, s))
        self.assertEqual(json.loads((folder/mf.SIDECAR).read_text(encoding='utf-8'))['resultFile'], mf.RESULT_GZ)
        packed = (folder/mf.RESULT_GZ).read_bytes()
        again = self.folder/'segments'/'again'
        write_artifact(again, result, compressed=True)
        self.assertEqual((again/mf.RESULT_GZ).read_bytes(), packed)   # the same bytes: no name or time in the header
        (folder/mf.RESULT_GZ).write_bytes(packed[:-1])
        self.assertIn('size', mf.verify_artifact(folder, s))
        (folder/mf.RESULT_GZ).write_bytes(packed[:-1]+bytes([packed[-1] ^ 1]))
        self.assertIn('checksum', mf.verify_artifact(folder, s))
        (folder/mf.RESULT_GZ).unlink()
        self.assertIn('result.json.gz is missing', mf.verify_artifact(folder, s))

    def test_the_original_stays_when_the_compressed_copy_does_not_read_back(self):
        folder = self.folder/'segments'/'x'
        write_artifact(folder, fake_result(0, 99))
        before = (folder/mf.RESULT).read_bytes()
        with mock.patch.object(mf.gzip, 'open', lambda *a, **k: io.BytesIO(b'{}')):
            with self.assertRaises(OSError):
                mf.compress_result(folder)
        self.assertEqual((folder/mf.RESULT).read_bytes(), before)
        self.assertEqual(sorted(p.name for p in folder.iterdir()), ['identities.json', 'registration.json', 'result.json', mf.SIDECAR])

    def test_an_artifact_from_before_the_compression_still_verifies_and_reads(self):
        s = self.segments[1]
        folder = self.folder/'segments'/s.key
        result = fake_result(s.proc0, s.proc1-1)
        write_artifact(folder, result)
        facts = json.loads((folder/mf.SIDECAR).read_text(encoding='utf-8'))
        del facts['resultFile']   # segment.json as the workers wrote it before the compression
        (folder/mf.SIDECAR).write_text(json.dumps(facts), encoding='utf-8')
        self.assertIsNone(mf.verify_artifact(folder, s))
        self.assertEqual(mf.read_result(folder), result)
        facts['resultFile'] = '../../elsewhere.json'
        (folder/mf.SIDECAR).write_text(json.dumps(facts), encoding='utf-8')
        self.assertIn('unknown result file', mf.verify_artifact(folder, s))
        del facts['resultFile']
        (folder/mf.SIDECAR).write_text(json.dumps(facts), encoding='utf-8')
        (folder/mf.RESULT).unlink()
        self.assertIn('result.json is missing', mf.verify_artifact(folder, s))

    def test_reconcile_after_a_crash(self):
        a, b, c = self.segments[0], self.segments[1], self.segments[2]
        write_artifact(self.m.artifact_folder(a.index), fake_result(a.proc0, a.proc1-1))
        write_artifact(self.m.artifact_folder(c.index), fake_result(c.proc0, c.proc1-1))
        (self.m.artifact_folder(c.index)/'result.json').write_text('{}', encoding='utf-8')   # broken after it was done
        self.m.update_segment(a.index, state='pending')    # the manifest write was lost: the artifact is verified
        self.m.update_segment(b.index, state='running')    # the app stopped mid-run
        self.m.update_segment(c.index, state='done', artifact='segments/'+c.key)
        changed = self.m.reconcile(deep=True)
        self.assertEqual(sorted(changed), [0, 1, 2])
        self.assertEqual([self.m.entry(i)['state'] for i in (0, 1, 2)], ['done', 'pending', 'pending'])
        self.assertIn('no longer valid', self.m.entry(2)['error'])
        (self.folder/'segments'/'s009-f1-2.partial-abc').mkdir(parents=True)
        self.assertEqual(self.m.remove_partials(), ['s009-f1-2.partial-abc'])

    def test_find_lists_matches_newest_first(self):
        for name, created in (('m0', '2026-10-28T06:21:38'), ('dev12', '2026-10-28T03:54:45')):
            other = mf.Manifest.create(self.folder.parent/name, {'key': 'k'*64}, VIDEO, plan.periods(800.0), self.segments,
                                       SETTINGS, {}, 3.0, 180.0)
            other.data['created'] = created
            other.save()
        self.m.data['created'] = '2026-10-28T02:00:00'
        self.m.save()
        (self.folder.parent/'broken').mkdir()
        (self.folder.parent/'broken'/'manifest.json').write_text('{not json', encoding='utf-8')
        found = mf.find(self.folder.parent.parent)
        self.assertEqual([m.data['matchId'] for m in found], ['m0', 'dev12', 'm1'])   # by creation, not by folder name


class DiskEstimateTests(unittest.TestCase):
    def test_the_estimate_covers_the_measured_match(self):
        # The whole CAN-ISL match (148,417 processed frames, 33 segments) with its results compressed: 638,764,214 bytes
        # once done (measured 2026-10-28: its segments, assembly with the coaching report's cache, calibration, logs;
        # 3,077,580,978 plain). While it runs, the plain results of the segments running at once are there too: at most
        # MAX_WORKERS of them, the largest measured 93 MiB (a broadcast segment of 4,642 frames).
        need = 638_764_214+system.MAX_WORKERS*93*2**20
        estimate = system.disk_needed(148417, 33)
        self.assertGreater(estimate, need)
        self.assertLess(estimate, 1.5*need)


if __name__ == '__main__':
    unittest.main()
