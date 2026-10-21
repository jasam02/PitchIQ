"""The coaching report PDF (soccer/coaching/pdf.py): a valid PDF 1.4 file (every xref offset, stream length and
reference checked by parsing the output), the same bytes for the same report, screenshots passed through as JPEG,
links only where a valid playback URL is given, text limited to WinAnsi, every timestamp printed as absolute and
segment-relative text, display labels instead of internal ids (never inside a coach's note), and honest empty
sections; a professional comparison claimed only where the baseline was available, each limitation printed once and
in words, the page's section order, feedback of another run shown as that run's, rows taller than a page continued
on the next one, a linear wrap(), and a download name with the context level.

Run:  python -m unittest test_coaching_pdf          (from this directory)
"""
import copy
import re
import sys
import tempfile
import time
import unittest
import zlib
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from soccer.coaching import pdf, render  # noqa: E402

URL = 'http://127.0.0.1:8765/?video={video}&run={run}&t={seek}&view=coaching'


def evidence(n, obs, frame, t, seg, tier='pinned', overlay=True):
    players = [{'type': 'player', 'id': f'A-0{k}', 'team': 'A', 'x': .3+.01*k, 'y': .15+.18*k, 'kind': 'O',
                'role': 'D', 'emphasis': k == 1} for k in range(1, 5)]
    players += [{'type': 'player', 'id': f'A-0{k+4}', 'team': 'A', 'x': .5, 'y': .05+.18*k, 'kind': 'O', 'role': 'M'}
                for k in range(1, 5)]
    players += [{'type': 'player', 'id': 'B-07', 'team': 'B', 'x': .42, 'y': .45, 'kind': 'O'}]
    diagram = {'preset': 'defensive_gap', 'title': 'Space between defence and midfield', 't': t, 'frame': frame,
               'team': 'A', 'attack': 'right', 'metresEstimated': True, 'pitch': {'length': 105, 'width': 68},
               'elements': players + [{'type': 'unit', 'unit': 'D', 'ids': ['A-01', 'A-02', 'A-03', 'A-04']},
                                      {'type': 'measure', 'a': [.32, .5], 'b': [.5, .5], 'text': '18.9 m (est.)'},
                                      {'type': 'band', 'axis': 'x', 'from': .33, 'to': .5, 'style': 'gap'}]}
    feet = [{'type': 'player', 'id': f'A-0{k}', 'team': 'A', 'x': 300+180*k, 'y': 700-40*k,
             'box': [285+180*k, 610-40*k, 30, 90]} for k in range(1, 5)]
    video = {'tier': tier, 'width': 1920, 'height': 1080,
             'elements': feet+[{'type': 'unit', 'unit': 'D', 'ids': ['A-01', 'A-02', 'A-03', 'A-04']}]}
    return {'id': f'ev-{n:012x}', 'observation': obs, 'occurrence': f'occ-{n:012x}', 'time': t, 'frame': frame,
            'segmentTime': seg, 'seek': round((frame+.5)/30, 4), 'fps': 30.0, 'players': ['A-01', 'A-02'],
            'metrics': {'gapDM': 18.9}, 'confidence': 'high', 'tier': tier,
            'calibration': {'confidence': .82, 'paintResidualPx': 4.1, 'disagreement': None},
            'diagram': diagram, 'videoOverlay': video if overlay else None,
            'caption': 'The gap between A-01 and the midfield at its widest.'}


def fixture_report():
    """A report in the shape of the contract (section 7), with every section filled."""
    ev = [evidence(1, 'obs-000000000001', 2533, 84.433, 42.0),
          evidence(2, 'obs-000000000001', 3010, 100.333, 57.9, tier='world'),
          evidence(3, 'obs-000000000002', 2000, 66.667, 24.234, overlay=False),
          evidence(4, 'obs-000000000001', 3500, 116.667, 74.234)]
    finding = {'observation': 'obs-000000000001', 'label': 'Potential improvement',
               'headline': 'Space between defence and midfield opened three times',
               'whatHappened': 'In 3 spells the gap between A-03 and the midfield stayed above the professional '
                               'range, the longest for 8.2 s.',
               'whyItMatters': 'A large gap between the units leaves room between the lines.',
               'whatToImprove': 'Keep the defence and the midfield within reach of each other.',
               'howToPractice': 'A compactness game with two lines.', 'drill': 'defensive_compactness',
               'evidence': [ev[0]['id'], ev[1]['id'], ev[3]['id']],
               'metrics': [{'name': 'Longest spell', 'value': 8.2, 'unit': 's'},
                           {'name': 'Peak gap', 'value': 21.4, 'unit': 'm'},
                           {'name': 'Share above the professional p90', 'value': .31, 'unit': 'share'}]}
    return {
        'version': 1, 'engineVersion': '1', 'fingerprint': 'a1b2c3d4e5f60718', 'generated': '2026-10-18T19:30:00',
        'kind': 'segment', 'context': 'competitive', 'team': 'A', 'teamName': None,
        'video': {'name': '2bac432528d0e568.mp4', 'key': '2bac432528d0e568'}, 'run': '20261018-024712-3485a2',
        'range': {'t0': 42.433, 't1': 222.4, 'segmentT0': 0.0, 'segmentT1': 179.967, 'seconds': 179.97},
        'overview': {'title': 'SEGMENT COACHING REPORT',
                     'coverage': {'measuredSeconds': 151.2, 'eligibleShare': .84,
                                  'calibration': {'high': .435, 'medium': .43, 'low': .135}},
                     'limitations': ['No registration from 190.60 s to 212.43 s.',
                                     'Metre values are estimated from an assumed 105 x 68 m pitch.']},
        'summary': ['Team A kept an even back line for most of the measured time.',
                    'The space between A-03 and the midfield opened three times.'],
        'strengths': [{'observation': 'obs-000000000003', 'label': 'Observed strength',
                       'headline': 'The back line stayed even', 'whatHappened': 'Measured over 151.2 s.',
                       'whyItMatters': 'An even line is harder to play through.', 'whatToImprove': None,
                       'howToPractice': None, 'drill': None, 'evidence': [], 'metrics': []}],
        'improvements': [finding],
        'reviews': [{'observation': 'obs-000000000002', 'label': 'Review opportunity',
                     'headline': 'A high line together with a large gap', 'whatHappened': 'One observed instance.',
                     'whyItMatters': 'Worth a look on video.', 'whatToImprove': None, 'howToPractice': None,
                     'drill': None, 'evidence': [ev[2]['id']], 'metrics': []}],
        'breakdown': {'defensiveOrganization': {'available': True, 'reason': None,
                                                'findings': ['obs-000000000003'], 'notes': []},
                      'spacing': {'available': True, 'reason': None, 'findings': ['obs-000000000001'],
                                  'notes': ['Measured on 151.2 s.']},
                      'formation': {'available': False, 'reason': 'no stable formation label', 'findings': [],
                                    'notes': []},
                      'attackingStructure': {'available': False, 'reason': None, 'findings': [], 'notes': []},
                      'movementSupport': {'available': True, 'reason': None, 'findings': [], 'notes': []}},
        'priorities': [{'title': 'Connect defence and midfield', 'finding': 'obs-000000000001',
                        'drill': 'defensive_compactness'}],
        'training': [{'drill': {'id': 'defensive_compactness', 'title': 'Defensive compactness game',
                                'objective': 'Keep two lines within 15 m.', 'setup': '40 x 30 m, 8 v 8.',
                                'coachingPoints': ['Move together.', 'Talk.']},
                      'findings': ['obs-000000000001']}],
        'confidence': {'overall': 'medium', 'notes': ['Registration was MEDIUM on 43% of the frames.']},
        'evidence': ev,
        'observations': [
            {'id': 'obs-000000000001', 'detector': 'unit_gap_dm', 'pattern': 'space_between_defence_and_midfield',
             'classification': 'improvement', 'severity': 'potential_improvement', 'status': 'reported',
             'measuredSeconds': 151.2, 'eligibleShare': .84, 'occurrenceCount': 3, 'ratePerMeasuredMinute': 1.19,
             'baseline': {'reference': 'metrica-sample-game-1', 'rule': 'default', 'context': 'third:middle',
                          'professionalRatePerMinute': {'p10': .1, 'p50': .4, 'p90': .9}, 'available': True},
             'confidence': {'tracking': 'high', 'calibration': 'medium', 'identity': 'high', 'ball': 'n/a',
                            'overall': 'medium', 'reasons': ['registration MEDIUM on 43% of the frames']}},
            {'id': 'obs-000000000002', 'detector': 'line_height_and_gap', 'pattern': 'high_line_with_gap',
             'classification': 'review', 'severity': 'review_opportunity', 'status': 'reported',
             'measuredSeconds': 90.0, 'occurrenceCount': 1, 'confidence': {'overall': 'medium'},
             'baseline': {'reference': None, 'rule': None, 'context': 'all', 'available': False}},
            {'id': 'obs-000000000003', 'detector': 'back_line_coordination', 'pattern': 'uneven_back_line',
             'classification': 'strength', 'severity': 'observed_strength', 'status': 'reported',
             'measuredSeconds': 151.2, 'confidence': {'overall': 'high'},
             'baseline': {'reference': 'metrica-sample-game-1', 'rule': 'default', 'context': 'backline:4',
                          'available': True}},
            {'id': 'obs-000000000004', 'detector': 'lateral_shift', 'pattern': 'lateral_lag',
             'classification': 'insufficient', 'status': 'reported', 'measuredSeconds': 14.0,
             'confidence': {'overall': 'low', 'reasons': ['short_stretch', 'keyless']}}],
        'feedback': [{'id': 'f20261018-190312-1a2b3c', 'run': '20261018-024712-3485a2', 'made': '2026-10-18T19:03:12',
                      'verdict': 'partial', 'observation': {'headline': 'Space between defence and midfield'},
                      'text': 'Two of the three spells were set pieces.'}]}


class Frames:
    """A frame provider: a deterministic 640 x 360 JPEG per frame (the frame number drawn in), with call records."""

    def __init__(self, missing=()):
        self.missing = set(missing)
        self.calls, self.prefetched = [], []

    def prefetch(self, frames, widths):
        self.prefetched.append((list(frames), list(widths)))

    def __call__(self, frame, width):
        self.calls.append((frame, width))
        if frame in self.missing:
            return None
        img = np.full((360, 640, 3), (40, 120, 60), np.uint8)
        cv2.putText(img, str(frame), (40, 200), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 5)
        ok, data = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return data.tobytes()


def parse(data):
    """{number: (dictionary bytes, stream bytes or None)} through the xref, checking offsets and stream lengths."""
    assert data.startswith(b'%PDF-1.4\n'), 'header'
    assert data.endswith(b'%%EOF\n'), 'trailer end'
    sx = data.rindex(b'startxref\n')
    xref = int(data[sx+10:].split()[0])
    assert data[xref:xref+5] == b'xref\n', 'startxref must point at the xref table'
    lines = data[xref:].split(b'\n')
    first, count = (int(v) for v in lines[1].split())
    assert first == 0 and lines[2] == b'0000000000 65535 f '
    objs = {}
    for num in range(1, count):
        entry = lines[2+num]
        assert len(entry)+1 == 20, 'xref entries are 20 bytes'
        off = int(entry[:10])
        head = b'%d 0 obj\n' % num
        assert data[off:off+len(head)] == head, f'object {num}: offset {off}'
        start = off+len(head)
        s, e = data.find(b'\nstream\n', start), data.find(b'\nendobj\n', start)
        if s != -1 and s < e:
            body = data[start:s]
            length = int(re.search(rb'/Length (\d+)', body).group(1))
            stream = data[s+8:s+8+length]
            assert data[s+8+length:s+8+length+18] == b'\nendstream\nendobj\n', f'object {num}: /Length'
            objs[num] = (body, stream)
        else:
            objs[num] = (data[start:e], None)
    trailer = data[data.rindex(b'trailer\n'):sx]
    assert int(re.search(rb'/Size (\d+)', trailer).group(1)) == count
    return objs, trailer


def page_texts(data):
    """The text shown on each page, in page order (decompressed content streams, Tj strings unescaped)."""
    objs, _ = parse(data)
    pages_obj = next(b for b, s in objs.values() if b.startswith(b'<< /Type /Pages'))
    kids = [int(k) for k in re.findall(rb'(\d+) 0 R', re.search(rb'/Kids \[(.*?)\]', pages_obj).group(1))]
    out = []
    for k in kids:
        content = int(re.search(rb'/Contents (\d+) 0 R', objs[k][0]).group(1))
        stream = zlib.decompress(objs[content][1])
        runs = re.findall(rb'\(((?:\\.|[^\\)])*)\) Tj', stream)
        out.append([re.sub(rb'\\(.)', rb'\1', r).decode('cp1252') for r in runs])
    return out


def build(report=None, frames=None, **options):
    return pdf.build_report_pdf(fixture_report() if report is None else report, frames, options)


class StructureTests(unittest.TestCase):
    def test_xref_offsets_lengths_references_and_page_count(self):
        data, info = build(frames=Frames(), playback_url=URL)
        objs, trailer = parse(data)
        for num, (body, _) in objs.items():
            for ref in re.findall(rb'(\d+) 0 R', body):
                self.assertIn(int(ref), objs, f'object {num} refers to a missing object')
        root = int(re.search(rb'/Root (\d+) 0 R', trailer).group(1))
        self.assertTrue(objs[root][0].startswith(b'<< /Type /Catalog'))
        pages = [n for n, (b, _) in objs.items() if b.startswith(b'<< /Type /Page ')]
        tree = next(b for b, _ in objs.values() if b.startswith(b'<< /Type /Pages'))
        self.assertEqual(int(re.search(rb'/Count (\d+)', tree).group(1)), len(pages))
        self.assertEqual(info['pages'], len(pages))
        self.assertGreaterEqual(len(pages), 4)
        self.assertEqual(info['bytes'], len(data))
        for n in pages:
            self.assertIn(b'/MediaBox [0 0 595.28 841.89]', objs[n][0])
        for body, stream in objs.values():                                  # every content stream inflates
            if stream is not None and b'/FlateDecode' in body:
                zlib.decompress(stream)

    def test_the_same_report_gives_the_same_bytes(self):
        a, _ = build(frames=Frames(), playback_url=URL)
        b, _ = build(frames=Frames(), playback_url=URL)
        self.assertEqual(a, b)
        later = fixture_report()
        later['generated'] = '2026-10-19T08:00:00'
        c, _ = build(later, Frames(), playback_url=URL)
        self.assertNotEqual(a, c)
        self.assertIn(b'/CreationDate (D:20261018193000)', a)

    def test_every_page_is_numbered_and_footed(self):
        data, info = build(frames=Frames())
        texts = page_texts(data)
        n = len(texts)
        for i, page in enumerate(texts, 1):
            self.assertIn(f'Page {i} of {n}', page)
            self.assertIn('PitchIQ coaching report | 2bac432528d0e568.mp4 | Team A', page)

    def test_a_long_report_breaks_pages(self):
        r = fixture_report()
        base = r['improvements'][0]
        r['improvements'] = []
        for k in range(18):
            f = copy.deepcopy(base)
            f['headline'] = f'Finding number {k}'
            r['improvements'].append(f)
        data, info = build(r, Frames())
        self.assertGreater(info['pages'], 12)
        joined = [' '.join(p) for p in page_texts(data)]
        self.assertTrue(all('Page' in p for p in joined))
        self.assertIn('Finding number 17', ' '.join(joined))

    def test_an_empty_report_is_still_a_valid_honest_pdf(self):
        data, info = build({}, None)
        parse(data)
        text = ' '.join(t for p in page_texts(data) for t in p)
        for key in ('strengths', 'improvements', 'reviews', 'training', 'evidence', 'priorities'):
            self.assertIn(pdf.EMPTY[key], text)
        # No wording of the module claims what PitchIQ does not record.
        self.assertIsNone(re.search(r'\b(possession|pressing|turnovers?|pass(es|ing)?|shots?)\b', text, re.I))

    def test_malformed_fields_never_break_the_export(self):
        base = fixture_report()
        for key in list(base):
            for junk in (None, 'text', 7, [None, 3, 'x']):
                r = copy.deepcopy(base)
                r[key] = junk
                data, _ = build(r, Frames())
                parse(data)
        r = copy.deepcopy(base)
        r['evidence'][0].update(frame=2533.0, diagram='x', videoOverlay=[1], metrics=None, time=None)
        r['observations'][0].update(confidence='high', baseline=[], measuredSeconds=float('nan'))
        r['improvements'][0].update(metrics={'a': 1})
        r['reviews'][0].update(evidence='ev-1')
        r['breakdown']['spacing'] = 'x'
        data, info = build(r, Frames())
        parse(data)
        self.assertEqual(info['screenshots'], 2)                           # 2533.0 (an integral float) and 3010

    def test_write_is_atomic(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'out'/'report.pdf'
            info = pdf.write_report_pdf(fixture_report(), path, Frames())
            self.assertEqual(Path(info['path']), path)
            self.assertEqual(path.read_bytes(), build(frames=Frames())[0])
            self.assertEqual([p.name for p in path.parent.iterdir()], ['report.pdf'])


class ContentTests(unittest.TestCase):
    def test_timestamps_are_absolute_and_segment_relative_text(self):
        data, _ = build(frames=Frames())
        text = ' '.join(t for p in page_texts(data) for t in p)
        self.assertIn('01:24.43  (segment +00:42.00)', text)
        self.assertIn('frame 2533', text)
        rows = [p for p in page_texts(data) if 'Evidence timestamps' in p][0]
        for s in ('01:24.43', '+00:42.00', '01:06.67', '+00:24.23', '01:56.67', '+01:14.23', '2533', '3500'):
            self.assertIn(s, rows)
        self.assertIn('00:42.43 - 03:42.40 of the source video (3.0 min); segment +00:00.00 to +02:59.97', text)

    def test_images_are_embedded_as_jpeg_and_capped(self):
        frames = Frames()
        data, info = build(frames=frames)
        objs, _ = parse(data)
        images = [(b, s) for b, s in objs.values() if b'/Subtype /Image' in b]
        self.assertEqual(len(images), 3)                                    # two rows per finding: 2533, 3010, 2000
        self.assertTrue(all(b'/Filter /DCTDecode' in b and b'/Width 640 /Height 360' in b for b, _ in images))
        for frame in (2533, 3010, 2000):
            self.assertIn(frames(frame, 640), [s for _, s in images])       # passed through byte for byte
        self.assertEqual(info['screenshots'], 3)
        self.assertEqual(info['overlays'], 2)                               # the third has no overlay
        self.assertEqual(frames.prefetched, [([2000, 2533, 3010], [640])])  # one sorted request first
        capped, info = build(frames=Frames(), max_screenshots=1)
        self.assertEqual(info['screenshots'], 1)
        text = ' '.join(t for p in page_texts(capped) for t in p)
        self.assertIn('Screenshot unavailable: screenshot left out to keep the file small', text)

    def test_without_frames_the_report_says_so(self):
        data, info = build(frames=None)
        self.assertNotIn(b'/Subtype /Image', data)
        text = ' '.join(t for p in page_texts(data) for t in p)
        self.assertIn('Screenshot unavailable: no source video was available when this PDF was made.', text)
        data, info = build(frames=Frames(missing={2533}))
        self.assertEqual(info['missingFrames'], [2533])
        self.assertEqual(info['screenshots'], 2)

    def test_an_overlay_of_another_size_is_left_out(self):
        r = fixture_report()
        r['evidence'][0]['videoOverlay']['height'] = 1920                 # a square overlay over a 16:9 frame
        data, info = build(r, Frames())
        self.assertEqual(info['overlays'], 1)
        text = ' '.join(t for p in page_texts(data) for t in p)
        self.assertIn('overlay was left out', text)

    def test_ids_are_shown_as_the_current_display_labels(self):
        labels = {'A-03': 'A-07', 'A-07': 'A-01', 'A-01': 'A-03'}       # a renumbering: one pass, no chains
        data, _ = build(frames=Frames(), labels=labels)
        text = ' '.join(t for p in page_texts(data) for t in p)
        self.assertIn('The space between A-07 and the midfield opened three times.', text)
        self.assertIn('the gap between A-03 and the midfield at its widest', text.replace('The gap', 'the gap'))
        plain, _ = build(frames=Frames(), labels=labels, substitute_text=False)
        self.assertIn('The space between A-03 and the midfield opened three times.',
                      ' '.join(t for p in page_texts(plain) for t in p))

    def test_the_store_entry_point_and_an_out_of_date_report(self):
        data = pdf.report_pdf(fixture_report(), labels={'A-03': 'A-07'}, frame=Frames(),
                              stale=['2 corrections since this report'])
        pages = page_texts(data)
        first = ' '.join(pages[0])
        self.assertIn('OUT OF DATE', first)
        self.assertIn('2 corrections since this report', first)
        self.assertTrue(all(any(t.startswith('OUT OF DATE') for t in p) for p in pages[1:]))   # every header
        self.assertIn(b'/Subtype /Image', data)
        current = pdf.report_pdf(fixture_report(), labels=None, frame=None, stale=None)
        self.assertNotIn('OUT OF DATE', ' '.join(t for p in page_texts(current) for t in p))

    def test_report_assembly_fields(self):
        # The fields soccer/coaching/report.py adds: a finding's video and confidence notes, the overview's context
        # notes, and the drill record of drills.json (name, ranges, safety, the level's variation).
        r = fixture_report()
        r['improvements'][0].update(video='Moments to watch: 01:24.43 (A-03).',
                                    confidenceNote='Medium confidence: registration MEDIUM.')
        r['overview']['context'] = ['Ball location, for context: middle third 45%.']
        drill = r['training'][0]['drill']
        del drill['title']
        drill.update(name='Compact game', players={'min': 12, 'max': 18, 'note': 'Two teams.'},
                     durationMinutes={'min': 15, 'max': 15}, safety=['Warm up first.'], variation='Play 7 v 7.',
                     recommendationKeys=['compact_lines'], patterns=['team_width'])
        data, _ = build(r, Frames(), labels={'A-03': 'A-07'})
        text = ' '.join(t for p in page_texts(data) for t in p)
        for s in ('ON THE VIDEO', 'Moments to watch: 01:24.43 (A-07).', 'Medium confidence: registration MEDIUM.',
                  'Ball location, for context: middle third 45%.', 'Compact game', '12-18 players. Two teams.',
                  '15 minutes.', 'Warm up first.', 'AT THIS LEVEL', 'Play 7 v 7.', 'Suggested practice: Compact game'):
            self.assertIn(s, text)
        self.assertNotIn('Confidence notes: registration MEDIUM on 43%', text)    # the finding's own note instead
        self.assertNotIn('compact_lines', text)                                     # internal fields stay out

    def test_sections_and_honest_labels(self):
        data, _ = build(frames=Frames())
        text = ' '.join(t for p in page_texts(data) for t in p)
        for s in ('SEGMENT COACHING REPORT', 'Team A (as labelled in this run)', 'Observed strengths',
                  'Potential improvements', 'Review opportunities', 'Tactical breakdown', 'Key statistics',
                  'Training recommendations', 'Evidence timestamps', 'Coverage and confidence', 'Expert feedback',
                  'Not assessed: no stable formation label', 'Partially correct', 'Defensive compactness game',
                  'Sideways shift of back line and midfield: not assessed; measured 14.0 s; no continuous measured '
                  'stretch was long enough; no pitch keyframes were set', 'Competitive (the level changes the wording',
                  'Further evidence moments for this finding: E3', '21.4 m (est.)', '31.0%',
                  'metrica-sample-game-1: one professional match, a sanity range and not a standard',
                  'High 43.5%, Medium 43.0%, Low 13.5%'):
            self.assertIn(s, text)


class LinkTests(unittest.TestCase):
    def test_uri_links_only_for_a_valid_playback_url(self):
        data, info = build(frames=Frames(), playback_url=URL)
        uris = re.findall(rb'/URI \((.*?)\)', data)
        self.assertEqual(info['links'], len(uris))
        self.assertIn(b'http://127.0.0.1:8765/?video=2bac432528d0e568&run=20261018-024712-3485a2&t=84.4500'
                      b'&view=coaching', uris)
        self.assertEqual(len({u for u in uris}), 4)                        # each evidence moment
        none, info = build(frames=Frames())
        self.assertNotIn(b'/URI', none)
        for bad in ('javascript:alert(1)', 'file:///C:/x.mp4?{frame}', 'http://bad host/{frame}',
                    'http://127.0.0.1:8765/?t={nope}'):
            data, info = build(frames=Frames(), playback_url=bad)
            self.assertNotIn(b'/URI', data, bad)
            self.assertEqual(info['links'], 0)
        data, info = build(frames=Frames(), playback_url=lambda ev: None)
        self.assertNotIn(b'/URI', data)

    def test_internal_links_point_at_pages(self):
        data, info = build(frames=Frames())
        objs, _ = parse(data)
        pages = {n for n, (b, _) in objs.items() if b.startswith(b'<< /Type /Page ')}
        dests = [int(d) for d in re.findall(rb'/Dest \[(\d+) 0 R /XYZ', data)]
        self.assertGreater(info['internalLinks'], 20)
        self.assertTrue(dests and all(d in pages for d in dests))
        self.assertIn(b'/Outlines', data)


class TextTests(unittest.TestCase):
    def test_winansi_is_enforced(self):
        self.assertEqual(pdf.encode_text('A\u2013B \u20ac \xe9')[0], b'A\x96B \x80 \xe9')
        self.assertEqual(pdf.encode_text('x\u2192y\u2713'), (b'x?y?', 2))
        with self.assertRaises(pdf.WinAnsiError):
            pdf.encode_text('x\u2192y', strict=True)
        self.assertEqual(pdf.winansi_problems('fine \u2013 text'), [])
        self.assertEqual(pdf.winansi_problems('\u2192 \u6771 \x81'), ['\x81', '\u2192', '\u6771'])
        r = fixture_report()
        r['teamName'] = '\u6771\u4eac FC'
        data, info = build(r, Frames())
        self.assertGreater(info['replaced'], 0)
        self.assertIn('?? FC', ' '.join(t for p in page_texts(data) for t in p))
        with self.assertRaises(pdf.WinAnsiError):
            build(r, Frames(), strict=True)

    def test_the_modules_own_text_is_ascii(self):
        for name in ('pdf.py', 'render.py', 'frames.py'):
            data = (ROOT/'soccer'/'coaching'/name).read_bytes()
            self.assertTrue(all(b < 128 for b in data), name)

    def test_widths_and_wrapping(self):
        self.assertAlmostEqual(pdf.text_width('Hello', 10), (722+556+222+222+556)/100)
        self.assertAlmostEqual(pdf.text_width('Hello', 10, 'bold'), (722+556+278+278+611)/100)
        self.assertAlmostEqual(pdf.text_width("'", 10), 1.91)                 # quotesingle, not quoteright
        lines = pdf.wrap('the quick brown fox jumps over the lazy dog ' * 6, 9, 120)
        self.assertTrue(all(pdf.text_width(line, 9) <= 120 for line in lines))
        self.assertEqual(' '.join(lines), ('the quick brown fox jumps over the lazy dog ' * 6).strip())
        broken = pdf.wrap('x' * 300, 9, 100)
        self.assertTrue(len(broken) > 1 and all(pdf.text_width(line, 9) <= 100 for line in broken))

    def test_jpeg_info(self):
        img = np.zeros((90, 160, 3), np.uint8)
        for progressive in (0, 1):
            ok, data = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_PROGRESSIVE, progressive])
            self.assertEqual(pdf.jpeg_info(data.tobytes()), (160, 90, 3))
        ok, grey = cv2.imencode('.jpg', img[:, :, 0])
        self.assertEqual(pdf.jpeg_info(grey.tobytes()), (160, 90, 1))
        for bad in (b'', b'\x89PNG\r\n', b'\xff\xd8\xff\xd9'):
            with self.assertRaises(ValueError):
                pdf.jpeg_info(bad)

    def test_clock_and_values(self):
        self.assertEqual(pdf.clock(84.433), '01:24.43')
        self.assertEqual(pdf.clock(3725.0), '62:05.00')
        self.assertEqual(pdf.offset(42.0), '+00:42.00')
        self.assertEqual(pdf.clock(float('nan')), 'n/a')
        self.assertEqual(pdf.value_text(.957, 'share'), '95.7%')
        self.assertEqual(pdf.value_text(.62, 'ratio'), '0.62')                # a correlation, not a share
        self.assertEqual(pdf.value_text(3.0, 'count'), '3')
        self.assertEqual(pdf.value_text(1.234, 'per_min'), '1.23 per min')
        self.assertEqual(pdf.value_text(21.44, 'm', True), '21.4 m (est.)')
        self.assertEqual(pdf.value_text(None), 'n/a')


class SelfCheckTests(unittest.TestCase):
    def test_pages_rasterize(self):
        doc, info = pdf.layout_report(fixture_report(), Frames(), {'playback_url': URL})
        for ops in doc.pages[:3]:
            img = render.ops_to_png(ops, doc.size[0], doc.size[1], .5)
            self.assertEqual(img.shape[:2], (421, 298))
            self.assertGreater(int((img < 200).any(axis=2).sum()), 2000)    # ink on the page


def all_text(data):
    return ' '.join(t for p in page_texts(data) for t in p)


def old_wrap(text, size, width, font='regular'):
    """The first wrap() (quadratic in a long word), kept as the reference the linear one must reproduce."""
    lines, line = [], ''
    for word in str(text).split():
        trial = f'{line} {word}' if line else word
        if pdf.text_width(trial, size, font) <= width:
            line = trial
            continue
        if line:
            lines.append(line)
        while pdf.text_width(word, size, font) > width and len(word) > 1:
            k = len(word)
            while k > 1 and pdf.text_width(word[:k], size, font) > width:
                k -= 1
            lines.append(word[:k])
            word = word[k:]
        line = word
    if line:
        lines.append(line)
    return lines


def feedback_entry(text, run='20261018-024712-3485a2', this_run=None, attached='obs-000000000001',
                   headline='Space between defence and midfield', key='obs-000000000001'):
    e = {'id': 'f20261018-190312-1a2b3c', 'run': run, 'made': '2026-10-18T19:03:12', 'verdict': 'incorrect',
         'observation': {'key': key, 'detector': 'unit_gap_dm', 'team': 'A', 'headline': headline}, 'text': text,
         'attached': attached}
    if this_run is not None:
        e['thisRun'] = this_run
    return e


class HonestyTests(unittest.TestCase):
    def test_a_professional_comparison_is_claimed_only_where_the_baseline_was_available(self):
        text = all_text(build(frames=Frames())[0])
        # the improvement (available baseline) says so; the review (available False) says it has none
        self.assertIn("Compared with the professional reference with the team's centre in the middle third", text)
        self.assertIn('no professional comparison: described from the tracked positions', text)
        self.assertNotIn('is compared with professional tracking data under the same rule. Times', text)
        self.assertIn('line height and gap has no professional comparison', text)
        # a formation-like report: no baseline anywhere, so no comparison claim and no reference row
        r = fixture_report()
        for o in r['observations']:
            o['baseline'] = {'available': False, 'reference': None, 'context': 'all',
                             'rule': 'team_shape periods'}
        text = all_text(build(r, Frames())[0])
        self.assertIsNone(re.search('compared with the professional reference', text, re.I))
        self.assertNotIn('metrica-sample-game-1', text)
        self.assertIn('No finding of this report has a professional comparison', text)
        # a baseline record that carries a reference but says unavailable (lateral shift without a stretch)
        r = fixture_report()
        r['observations'][0]['baseline']['available'] = False
        text = all_text(build(r, Frames())[0])
        self.assertNotIn("Compared with the professional reference with the team's centre", text)

    def test_limitations_are_printed_once_in_words(self):
        r = fixture_report()
        line = ('Back-line evenness: not assessed; only 20 s could be measured (11% of the time), with a longest '
                'continuous stretch of 6.0 s; no pitch keyframes were set, so distances rely on the automatic '
                'registration.')
        r['overview']['limitations'].append(line)
        r['breakdown']['defensiveOrganization']['notes'] = ['Measured for up to 2.5 minutes.', line]
        r['breakdown']['formation'] = {'available': False, 'findings': [], 'notes': [],
                                       'reason': 'Not enough reliable measurement. ' + line}
        r['observations'].append({'id': 'obs-000000000005', 'detector': 'isolation', 'pattern': 'player_isolation',
                                  'classification': 'insufficient', 'status': 'reported', 'measuredSeconds': 0.0,
                                  'insufficientReasons': ['short_measured', 'small_scene_veto', 'broadcast_only'],
                                  'confidence': {'overall': 'low', 'reasons': ['keyless', 'short_stretch',
                                                                               'small_scene']}})
        data, _ = build(r, Frames())
        pages = page_texts(data)
        text = ' '.join(t for p in pages for t in p)
        self.assertEqual(text.count('only 20 s could be measured (11% of the time)'), 1)        # was 3 times
        self.assertIn('This report has 3 limitations', ' '.join(pages[0]))                     # page 1: a pointer
        self.assertNotIn('No registration from 190.60 s', ' '.join(pages[0]))
        self.assertEqual(text.count('No registration from 190.60 s to 212.43 s.'), 1)
        self.assertIn('1 measurement of this area could not be assessed or was limited: see Limitations', text)
        self.assertIn('Support distances: not assessed; measured 0.0 s; too little of the play could be measured; '
                      'players appear too small in this footage for this measurement', text)
        from soccer.coaching import language
        for code in language.REASONS:                     # no raw reason code, no internal pattern name
            if '_' in code:
                self.assertNotIn(code, text)
        for internal in ('Lateral lag', 'Player isolation', 'Space between defence and midfield: measured',
                         '(window:20)'):
            self.assertNotIn(internal, text)

    def test_the_section_order_is_the_page_order(self):
        self.assertEqual(pdf.section_order(pdf.PAGE_ORDER),
                         ('overview', 'summary', 'strengths', 'improvements', 'reviews', 'breakdown', 'statistics',
                          'priorities', 'training', 'confidence', 'evidence', 'feedback'))
        # a published order is followed; a list of the breakdown's subsections is not an order of the report
        self.assertEqual(pdf.section_order(['overview', 'summary', 'strengths', 'improvements', 'reviews',
                                            'priorities', 'breakdown', 'training', 'confidence'])[5:8],
                         ('priorities', 'breakdown', 'statistics'))
        self.assertEqual(pdf.section_order(['defensiveOrganization', 'spacing']), pdf.section_order())
        doc, _ = pdf.layout_report(fixture_report(), Frames(), {})
        titles = dict(pdf.SECTIONS)
        heads = [op['text'] for p in doc.pages for op in p                  # the level-1 headings, in print order
                 if op.get('op') == 'text' and op.get('size') == 15.0 and op.get('text') in titles.values()]
        self.assertEqual(heads, [titles[k] for k in pdf.section_order() if k in titles])
        # the Studio page (coaching.js renderMain) shows the shared sections in the same order
        js = (ROOT/'coaching.js').read_text(encoding='utf-8')
        m = re.search(r'renderMain\(\)\s*\{(.*?)\n  \}', js, re.S)
        if m is None:
            self.skipTest('coaching.js renderMain not found')
        groups = re.search(r'const GROUPS = \[(.*?)\n\];', js, re.S)
        groups = re.findall(r"^\s*\['(\w+)'", groups.group(1), re.M) if groups else []
        calls = re.findall(r'this\.(overview|summary|group|breakdown|priorities|training|confidence)\(', m.group(1))
        page = []
        for c in calls:
            page += groups if c == 'group' else [c]
        shared = [k for k in pdf.section_order() if k in page]
        self.assertEqual(page, shared)


class FeedbackTests(unittest.TestCase):
    def test_a_coach_note_is_printed_as_typed(self):
        r = fixture_report()
        r['feedback'] = [feedback_entry('A-05 was the one out of line, not A-03.', this_run=True)]
        text = all_text(build(r, Frames(), labels={'A-05': 'A-07', 'A-03': 'A-01'})[0])
        self.assertIn('A-05 was the one out of line, not A-03.', text)      # never relabelled
        self.assertNotIn('A-07 was the one out of line', text)

    def test_a_verdict_from_another_run_is_shown_as_that_runs(self):
        r = fixture_report()
        r['feedback'] = [feedback_entry('Right call.', this_run=True),
                         feedback_entry('Wrong team.', run='20261017-101010-aaaaaa', this_run=False,
                                        headline='Space between A-03 and midfield'),
                         feedback_entry('Unknown origin.', run=None)]
        doc, _ = pdf.layout_report(r, Frames(), {'labels': {'A-03': 'A-07'}})
        data = doc.build()
        text = all_text(data)
        self.assertIn('Feedback recorded on earlier runs of this video', text)
        self.assertIn('run 20261017-101010-aaaaaa', text)
        self.assertIn('Space between A-03 and midfield', text)              # the ids of that run, as recorded
        self.assertNotIn('Space between A-07 and midfield', text)
        self.assertIn('2026-10-18, run not recorded', text)
        # only the verdict of this run links to the finding
        start = next(i for i, p in enumerate(doc.pages) if any(op.get('text') == 'Expert feedback' for op in p))
        after = []
        for i, p in enumerate(doc.pages[start:]):
            k = next(j for j, op in enumerate(p) if op.get('text') == 'Expert feedback') if i == 0 else 0
            after += p[k:]
        heading = 'Feedback recorded on earlier runs of this video'
        mark = next(k for k, op in enumerate(after) if op.get('text') == heading)
        links = [k for k, op in enumerate(after) if op.get('op') == 'link' and op.get('dest') == 'obs-obs-000000000001']
        self.assertTrue(links and all(k < mark for k in links))             # the row of this run, none after
        self.assertLess(text.index('Right call.'), text.index('Feedback recorded on earlier runs'))
        self.assertLess(text.index('Feedback recorded on earlier runs'), text.index('Wrong team.'))

    def test_a_row_taller_than_a_page_continues_on_the_next_page(self):
        for note in ('W'*2000, ('WWWWWWW ' * 250)[:2000]):
            r = fixture_report()
            r['feedback'] = [feedback_entry(note, this_run=True)]
            t0 = time.perf_counter()
            doc, _ = pdf.layout_report(r, Frames(), {})
            self.assertLess(time.perf_counter()-t0, 5.0)                     # 34 s with the quadratic wrap
            H = doc.size[1]
            footer = H-pdf.MARGINS[3]+26
            for page in doc.pages:
                for op in page:
                    if op.get('op') == 'text' and abs(op['y']-footer) > .01:
                        self.assertLessEqual(op['y'], H-pdf.MARGINS[3]+.01)
            r['feedback'] = [feedback_entry('-', this_run=True)]
            base, _ = pdf.layout_report(r, Frames(), {})
            self.assertEqual(self.ws(doc)-self.ws(base), note.count('W'))   # every character printed

    @staticmethod
    def ws(doc):
        return sum(op['text'].count('W') for p in doc.pages for op in p if op.get('op') == 'text')

    def test_the_file_name_carries_the_level(self):
        self.assertEqual(pdf.pdf_filename('2bac432528d0e568.mp4', '20261018-024712-3485a2', 'A', 'development'),
                         'pitchiq-coaching-2bac432528d0e568-20261018-024712-3485a2-A-development.pdf')
        names = {pdf.pdf_filename('m.mp4', 'r1', 'A', level) for level in ('development', 'competitive', 'advanced')}
        self.assertEqual(len(names), 3)
        self.assertEqual(pdf.pdf_filename('../my match (final).mp4', 'r/1', 'B', 'nope'),
                         'pitchiq-coaching-my-match-final-r-1-B.pdf')
        self.assertEqual(pdf.pdf_filename(None, 'r1', 'A', 'advanced', key='abcdef0123456789ffff'),
                         'pitchiq-coaching-abcdef0123456789-r1-A-advanced.pdf')
        self.assertEqual(pdf.report_filename(fixture_report()),
                         'pitchiq-coaching-2bac432528d0e568-20261018-024712-3485a2-A-competitive.pdf')


def fields_report():
    """fixture_report() with the fields soccer/coaching/report.py writes: the data-quality note at the head of the
    limitations (overview.dataQualityLines), the bare per-measurement lines of a footage with a few players per team
    in view and the observations they cover (overview.notAssessed), a grouped strength (its 'observations' and every
    member's evidence) whose whatToImprove starts "To build on it:", and a drill with regressions."""
    r = fixture_report()
    quality = ['Players appear small in this footage, so positions and identities are less certain than on broadcast '
               'footage.', 'What would help: set 2-3 pitch keyframes on frames where the pitch lines are clearly '
               'visible.']
    r['overview']['limitations'] = quality+['Team width: not assessed on this footage.',
                                            'Team length: not assessed on this footage.',
                                            'No registration from 190.60 s to 212.43 s.']
    r['overview']['dataQualityLines'] = 2
    r['overview']['notAssessed'] = [{'observation': 'obs-000000000004', 'topic': 'Sideways shift'}]
    member = {'id': 'obs-000000000006', 'detector': 'compactness', 'pattern': 'team_area', 'team': 'A',
              'classification': 'strength', 'severity': 'observed_strength', 'status': 'reported',
              'measuredSeconds': 151.2, 'confidence': {'overall': 'high'},
              'baseline': {'reference': 'metrica-sample-game-1', 'rule': 'default', 'context': 'third:middle',
                           'available': True}}
    r['observations'].append(member)
    ev = evidence(5, member['id'], 3600, 120.0, 77.567)
    r['evidence'].append(ev)
    strength = r['strengths'][0]
    strength.update(observations=[strength['observation'], member['id']], evidence=[ev['id']],
                    whatToImprove='To build on it: keep the line this level when the ball is switched.')
    r['training'][0]['drill']['regressions'] = ['Walk the shifts through first.']
    return r


class ReportFieldTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = fields_report()
        data, _ = build(cls.report, Frames())
        cls.pages = page_texts(data)
        cls.text = ' '.join(t for p in cls.pages for t in p)
        cls.doc, _ = pdf.layout_report(cls.report, Frames(), {})

    def test_the_data_quality_note_is_printed_once_on_page_1(self):
        first = ' '.join(self.pages[0])
        self.assertIn('Players appear small in this footage', first)
        self.assertIn('What would help: set 2-3 pitch keyframes', first)
        self.assertEqual(self.text.count('Players appear small in this footage'), 1)
        self.assertIn('This report has 2 further limitations', first)
        self.assertIn('The data-quality note of this footage is on page 1', self.text)

    def test_bare_lines_are_one_line_and_covered_observations_get_none(self):
        self.assertIn('Not assessed on this footage: Team width, team length.', self.text)
        self.assertNotIn('Team width: not assessed on this footage.', self.text)
        self.assertEqual(pdf.join_bare(['a', 'X: not assessed on this footage.', 'b']),
                         ['a', 'X: not assessed on this footage.', 'b'])
        # obs-000000000004 (lateral shift, insufficient) is covered by the report's own lines: no extra line
        self.assertNotIn('not assessed; measured 14.0 s', self.text)
        r = fields_report()
        r['overview']['notAssessed'] = []
        self.assertIn('not assessed; measured 14.0 s', all_text(build(r, Frames())[0]), 'uncovered: a line of its own')

    def test_every_member_of_a_grouped_finding_is_linked(self):
        anchors = self.doc.anchors
        self.assertIn('obs-obs-000000000003', anchors)
        self.assertIn('obs-obs-000000000006', anchors, "the member's anchor sits on the grouped finding")
        self.assertEqual(anchors['obs-obs-000000000003'], anchors['obs-obs-000000000006'])
        dests = [op.get('dest') for page in self.doc.pages for op in page if op.get('op') == 'link' and op.get('dest')]
        self.assertIn('obs-obs-000000000006', dests, "the member's evidence row links to the finding")
        self.assertTrue(all(d in anchors for d in dests), [d for d in dests if d not in anchors])

    def test_a_strength_says_how_to_build_on_it(self):
        self.assertIn('TO BUILD ON IT', self.text)
        self.assertIn('Keep the line this level when the ball is switched.', self.text)
        self.assertNotIn('To build on it:', self.text)
        self.assertIn('REGRESSIONS', self.text)
        self.assertIn('Walk the shifts through first.', self.text)


class WrapTests(unittest.TestCase):
    def test_wrap_is_linear_and_gives_the_same_lines(self):
        import random
        rng = random.Random(3)
        alphabet = 'abcdefghijklmnopqrstuvwxyzWMil.,;()-ABC\u00e9\u2013\u2192   '
        for _ in range(300):
            s = ''.join(rng.choice(alphabet) for _ in range(rng.randint(0, 160)))
            if rng.random() < .2:
                s += ' ' + 'W'*rng.randint(20, 60)
            size, width = rng.choice([6.6, 7.8, 9.5, 21]), rng.uniform(8, 400)
            font = rng.choice(['regular', 'bold', 'oblique'])
            self.assertEqual(pdf.wrap(s, size, width, font), old_wrap(s, size, width, font), (s, size, width))
        t0 = time.perf_counter()
        lines = pdf.wrap('W'*2000, 7.8, 150)
        self.assertLess(time.perf_counter()-t0, .5)                          # 34.9 s before
        self.assertEqual(''.join(lines), 'W'*2000)
        self.assertTrue(all(pdf.text_width(x, 7.8) <= 150 for x in lines))


if __name__ == '__main__':
    unittest.main()
