"""The coaching report PDF (soccer/coaching/pdf.py): a valid PDF 1.4 file (every xref offset, stream length and
reference checked by parsing the output), the same bytes for the same report, screenshots passed through as JPEG,
links only where a valid playback URL is given, text limited to WinAnsi, every timestamp printed as absolute and
segment-relative text, display labels instead of internal ids (never inside a coach's note), and honest empty
sections; a professional comparison claimed only where the baseline was available, each limitation printed once and
in words, the page's section order, feedback of another run shown as that run's, rows taller than a page continued
on the next one, a linear wrap(), and a download name with the context level.

Coaching v2 reports (schema 'report' version 2): the seven sections in the page's order and words, key moments in
time order with exact frames zoomed on the players involved (clipped, never re-encoded), overlays at what their tier
allows with the area kept on the diagram, the dotted-area sentence only when a report holds one, the level's
register for every added number, coach-marked possession labelled, no player named where the level names nobody,
drills with their six fields, library card and source credit, the progression table or how to get one, sources
linked, coach notes as typed with adjusted areas drawn, the viewer's team colours, fixed words that pass the event
and banned-word checks at levels 1-7, real scene reports through every stage, and critic N1 / N2 / N5 / N8 / R4.

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


# ---------------------------------------------------------------------------------------------------------------
# Coaching v2 reports (schema 'report' version 2): the page's seven sections, for print

from soccer.coaching import schema as S  # noqa: E402

COLUMN = (pdf.A4[0]-pdf.MARGINS[0]-pdf.MARGINS[2]-pdf.COLUMN_GAP)/2


def v2_report():
    """The schema's example report (one consistent mini-run; S.cross_check passes on it) with a strength, a review,
    a described finding, a formation diagram, a progression and a coach review added; still schema-valid."""
    r = S.example('report')
    work = r['needsWork'][0]
    strength = copy.deepcopy(work)
    strength.update(id='fnd-00000000aaaa', key='D1|A|team|775.0|790.0', verdict='strength', priority=None, drills=[],
                    suggestedAction=None, timestamps=work['timestamps'][1:2], opportunities=work['opportunities'][1:2])
    strength['aggregate'] = dict(work['aggregate'], opportunities=1, demonstrated=1, notDemonstrated=0, unclear=0,
                                 distinctFailingSpells=0, distinctDemonstratedSpells=1)
    strength['text'] = {'headline': 'A-05 covered behind A-04', 'explanation': 'A-05 stood behind A-04 at 12:55.00.',
                        'short': 'Cover in place', 'action': 'To build on it, keep the cover close.'}
    r['strengths'] = [strength]
    r['progression'] = {'squad': 'sq-0123456789ab', 'note': 'Read per moment, never on match totals (2 matches).',
                        'skills': [{'skill': 'pressure_and_cover', 'name': 'Pressure and cover', 'state': 'DEVELOPING',
                                    'opportunities': 31, 'matches': 3, 'trend': 'up'}]}
    return r


class Frames2(Frames):
    """Frames of the asked width in the video's 16:9 (a zoomed screenshot asks for 1280 px)."""

    def __call__(self, frame, width):
        self.calls.append((frame, width))
        if frame in self.missing:
            return None
        img = np.full((width*9//16, width, 3), (40, 120, 60), np.uint8)
        cv2.putText(img, str(frame), (40, width//4), cv2.FONT_HERSHEY_SIMPLEX, width/200, (255, 255, 255), 5)
        return cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()


def build2(report=None, frames=None, **options):
    return pdf.build_report_pdf(v2_report() if report is None else report, frames, options)


def text_of(data):
    return ' '.join(t for p in page_texts(data) for t in p)


def doc_ops(report, frames=None, **options):
    doc, info = pdf.layout_report(report, frames, options)
    return doc, info, [op for page in doc.pages for op in page]


class V2StructureTests(unittest.TestCase):
    def test_a_v2_report_is_a_valid_deterministic_pdf_with_the_seven_sections_in_order(self):
        r = v2_report()
        self.assertEqual(S.validate('report', r), [])
        data, info = build2(r, Frames2(), playback_url=URL)
        objs, _ = parse(data)
        for num, (body, _) in objs.items():
            for ref in re.findall(rb'(\d+) 0 R', body):
                self.assertIn(int(ref), objs)
        self.assertEqual(data, build2(r, Frames2(), playback_url=URL)[0])
        self.assertEqual(info['version'], 2)
        self.assertEqual(info['droppedLinks'], 0)                       # every internal link has its anchor
        self.assertEqual(info['replaced'], 0)
        text = text_of(data)
        heads = ['1. Match overview', '2. What went well (1)', '3. What needs work (1)', '4. Key moments (3)',
                 '5. Tactical diagrams (1)', '6. Training drills (1)', '7. Development progression', 'Sources (1)',
                 'About this report']
        at = text.index('How to read this report')                  # past the contents, which name them too
        for h in heads:
            at = text.index(h, at)
        for title in heads:                                             # the viewer's sidebar has them all
            self.assertIn(pdf._string(title.encode('ascii')), data)
        texts = page_texts(data)
        for i, page in enumerate(texts, 1):
            self.assertIn(f'Page {i} of {len(texts)}', page)
        self.assertNotIn('COACHING ILLUSTRATION', text)

    def test_the_key_moments_are_chronological_with_absolute_and_segment_times_and_exact_frames(self):
        r = v2_report()
        r['keyMoments'] = list(reversed(r['keyMoments']))               # given out of order
        data, info = build2(r, Frames2())
        text = text_of(data)
        times = []
        for ev in sorted(r['keyMoments'], key=lambda e: e['time']):
            n = len(times)+1
            line = (f'E{n}  |  {pdf.clock(ev["time"])}  (segment {pdf.offset(ev["segmentTime"])})  |  '
                    f'frame {ev["frame"]}')
            self.assertIn(line, text)
            times.append(text.index(line))
        self.assertEqual(times, sorted(times))
        self.assertEqual(info['evidenceShown'], 3)
        self.assertIn('Replay window 12:38.00 - 12:43.00.', text)       # the annotation window (DESIGN 5.6)
        self.assertIn('Key moments: E1 12:41.00 E2 12:55.00 E3 13:09.00', text)

    def test_screenshots_pass_through_capped_and_zoomed_on_the_players_involved(self):
        frames = Frames2()
        data, info = build2(frames=frames)
        self.assertEqual(info['screenshots'], 3)
        self.assertEqual(info['images'], 3)
        self.assertGreaterEqual(info['zoomed'], 1)
        self.assertIn((22830, pdf.ZOOM_WIDTH), frames.calls)            # the zoomed moment asks for the 1280 px frame
        self.assertIn(frames(22830, pdf.ZOOM_WIDTH), data)              # passed through as it came (/DCTDecode)
        doc, info, ops = doc_ops(v2_report(), Frames2())
        image = next(op for op in ops if op['op'] == 'image' and op['frame'] == 22830)
        self.assertGreater(image['w'], COLUMN*1.3)                      # magnified, then clipped to the column
        clip = next(op for op in ops if op['op'] == 'clip' and abs(op['w']-COLUMN) < .01)
        self.assertLessEqual(image['x'], clip['x'])
        self.assertGreaterEqual(image['x']+image['w'], clip['x']+clip['w'])
        _, plain, _ = doc_ops(v2_report(), Frames2(), zoom=False)
        self.assertEqual(plain['zoomed'], 0)
        _, few, _ = doc_ops(v2_report(), Frames2(), max_screenshots=1)
        self.assertEqual(few['screenshots'], 1)
        _, none, _ = doc_ops(v2_report(), None)
        self.assertEqual(none['screenshots'], 0)
        self.assertIn('Screenshot unavailable: no source video was available when this PDF was made.',
                      text_of(build2()[0]))

    def test_the_zoom_keeps_every_person_of_the_annotation_in_the_crop(self):
        ov = v2_report()['keyMoments'][0]['videoOverlay']
        box = pdf.focus_box(ov)
        crop = pdf.zoom_crop(box, ov['width'], ov['height'])
        x0, y0, w, h, zoom = crop
        self.assertTrue(x0 <= box[0] and box[2] <= x0+w and y0 <= box[1] and box[3] <= y0+h)
        self.assertAlmostEqual(w/h, 16/9, 3)
        self.assertLessEqual(zoom, 1/pdf.ZOOM_MIN_SHARE+1e-9)
        self.assertGreaterEqual(zoom, pdf.ZOOM_FROM)
        self.assertIsNone(pdf.zoom_crop((0, 0, 1900, 1000), 1920, 1080))   # people across the frame: no zoom
        tight = pdf.zoom_crop((900, 500, 920, 540), 1920, 1080)          # one player: the zoom is capped
        self.assertAlmostEqual(tight[4], 1/pdf.ZOOM_MIN_SHARE, 6)
        self.assertTrue(tight[0] <= 900 and 920 <= tight[0]+tight[2] and tight[1] <= 500 and 540 <= tight[1]+tight[3])
        self.assertIsNone(pdf.focus_box({'width': 1920, 'height': 1080, 'elements': []}))

    def test_an_overlay_draws_only_what_its_tier_allows_and_the_diagram_keeps_the_area(self):
        r = v2_report()
        ev = r['keyMoments'][0]
        ev['tier'] = ev['videoOverlay']['tier'] = 'pinned'
        doc, info, ops = doc_ops(r, Frames2())
        skipped = [s for s in info['skippedElements'] if s['evidence'] == ev['id']]
        zone = [s for s in skipped if s['type'] == 'suggestedZone']
        self.assertEqual([s['figure'] for s in zone], ['overlay'])
        self.assertIn('pinned-tier frame (shown on the 2D diagram)', zone[0]['reason'])
        text = ' '.join(op['text'] for op in ops if op['op'] == 'text')
        self.assertIn("From the players' feet", text)
        self.assertIn('possible covering area', text)                   # still on the diagram

    def test_the_dotted_area_is_described_only_when_the_report_has_one(self):
        sentence = 'A dotted area is a possible better position'
        self.assertIn(sentence, text_of(build2()[0]))
        r = v2_report()
        for ev in r['keyMoments']+[d for d in r['diagrams']]:
            for spec in (ev.get('diagram'), ev.get('videoOverlay')):
                if spec:
                    spec['elements'] = [e for e in spec['elements'] if e['type'] not in ('suggestedZone', 'moveArrow')]
        self.assertNotIn(sentence, text_of(build2(r)[0]))

    def test_an_out_of_date_report_says_so_once_without_a_double_full_stop(self):
        data, _ = build2(stale=['The tracking result changed.', 'The team settings changed.'])
        text = text_of(data)
        self.assertIn('OUT OF DATE  |  Team A', text)
        self.assertIn('made: The tracking result changed; The team settings changed. Regenerate', text)
        self.assertNotIn('..', text.replace('...', ''))

    def test_malformed_fields_never_break_the_export(self):
        base = v2_report()
        for key in list(base):
            for junk in (None, 'text', 7, [None, 3, 'x'], {'x': 1}):
                r = copy.deepcopy(base)
                r[key] = junk
                parse(build2(r, Frames2())[0])
        r = copy.deepcopy(base)
        r['keyMoments'][0].update(frame=None, diagram='x', videoOverlay=[1], time=None, annotation=7)
        r['keyMoments'][1]['videoOverlay'] = {'tier': 'world', 'width': 1920, 'height': 1080, 'elements': [
            None, {'type': 'focus'}, {'type': 'relation', 'pa': 'x'}, {'type': 'suggestedZone', 'polygon': 'x'}]}
        r['needsWork'][0].update(text=None, timestamps='x', aggregate=[], measurements=None, drills=[3])
        r['drills'][0]['fields'] = None
        r['overview'].update(possession=None, detectors=[None, {'detector': 7}], notices='x')
        r['profile'].update(level='4', register=None, skills=[])
        data, info = build2(r, Frames2())
        parse(data)
        parse(build2({'kind': 'coaching-report', 'version': 2}, None)[0])


class V2ContentTests(unittest.TestCase):
    def test_cards_follow_the_level_register(self):
        r = v2_report()
        text = text_of(build2(r)[0])
        self.assertIn('Missing in 5 of 9 moments (3 separate spells), 1 unclear, when the other team had the ball '
                      '(estimated), over 88.2 s of that time.', text)
        self.assertIn('At the first key moment: Nearest player to the ball 3.1 m (est.).', text)   # share hidden at L4
        self.assertIn('WORK ON', text)
        self.assertIn('Priority 1', text)
        r['profile']['register']['numbers'] = 'none'
        quiet = text_of(build2(r)[0])
        for s in ('Missing in', 'At the first key moment', 'checked in'):
            self.assertNotIn(s, quiet)
        self.assertIn('Looked at when the other team had the ball (estimated).', quiet)
        r = v2_report()                                                 # a level-5 measure stays hidden at level 4
        r['needsWork'][0]['measurements']['nearCbShiftM'] = 4.2
        self.assertNotIn('Near centre-back shift', text_of(build2(r)[0]))
        r['profile']['level'] = r['needsWork'][0]['level'] = 5
        self.assertIn('Near centre-back shift 4.2 m (est.)', text_of(build2(r)[0]))
        r = v2_report()
        r['profile']['register']['numbers'] = 'rates'
        r['profile']['level'] = r['needsWork'][0]['level'] = 7
        self.assertIn('per 10 minutes of it', text_of(build2(r)[0]))

    def test_coach_marked_phases_are_labelled(self):
        r = v2_report()
        r['needsWork'][0]['measurements']['phaseSource'] = 'coach'
        text = text_of(build2(r)[0])
        self.assertIn('when the other team had the ball (marked by the coach)', text)
        self.assertIn('who had the ball: marked by the coach', text)
        self.assertIn('Where you marked who had the ball, your marks are used and labelled as marked by the coach.',
                      text)
        self.assertNotIn('labelled as marked by the coach', text_of(build2()[0]))

    def test_ids_are_shown_as_display_labels_but_a_level_that_names_nobody_gets_no_labels(self):
        labels = {'A-04': 'A-03', 'A-05': 'A-07', 'B-10': 'B-02'}
        data, _ = build2(None, Frames2(), labels=labels)
        text = text_of(data)
        self.assertIn('A-07 covered behind A-03', text)
        self.assertIn('Players: A-03, A-07, B-02.', text)
        r = v2_report()
        for ev in r['keyMoments']:
            for spec in (ev['diagram'], ev['videoOverlay']):
                for el in (spec or {}).get('elements', []):
                    if el['type'] == 'focus':
                        el['label'] = False
        doc, info, ops = doc_ops(r, Frames2(), labels=labels)
        shown = [op['text'] for op in ops if op['op'] == 'text']
        self.assertNotIn('A-03', shown)                                 # no marker or overlay label of the player
        self.assertNotIn('Players:', ' '.join(shown))
        self.assertIn('one possibility for the teammate who could move', ' '.join(shown))

    def test_drills_print_the_six_fields_the_library_card_and_the_source_credit(self):
        r = v2_report()
        r['drills'].append(copy.deepcopy(r['drills'][0]))               # the same drill for a second finding
        text = text_of(build2(r)[0])
        self.assertEqual(text.count('Cover the Line'), 1+text.count('Drill: '+'3v3 Cover the Line'))
        for title in ('WHAT IT TRAINS', 'WHY IT WAS CHOSEN', 'MAKE IT EASIER', 'MAKE IT HARDER',
                      'WHAT PROGRESS LOOKS LIKE NEXT MATCH', 'HOW IT RUNS', 'COACHING POINTS'):
            self.assertIn(title, text)
        self.assertIn('Moments it addresses: E1 12:41.00', text)
        self.assertIn('Source: Adapted from', text)
        self.assertIn('PitchIQ proposal, not validated by a coach', text)
        self.assertIn('Safety at this age', text)
        r = v2_report()
        r['drills'][0].update(finding=None, optional=True)
        self.assertIn("Optional: for the next objective, not from this match's evidence.", text_of(build2(r)[0]))

    def test_progression_table_or_the_way_to_get_one_and_the_next_objective(self):
        text = text_of(build2()[0])
        for s in ('Pressure and cover', 'Developing', '31', 'improving', 'Next development objective',
                  'Read per moment, never on match totals'):
            self.assertIn(s, text)
        r = v2_report()
        r['progression'] = None
        self.assertIn('Link this team to a squad in the coaching setup to follow its skills across matches.',
                      text_of(build2(r)[0]))

    def test_sources_are_credited_with_working_links(self):
        data, info = build2()
        self.assertIn(b'/URI (https://learn.englandfootball.com/)', data)
        text = text_of(data)
        self.assertIn('[EF-COVER-RECOVER] Cover and recover (England Football)', text)
        self.assertIn('No federation reviewed or endorses PitchIQ', text)
        goal = ('Cover is the angle and distance that make you the next protection behind the presser. '
                '[EF-COVER-RECOVER]')
        self.assertGreater(text.index(goal), text.rindex('Sources ('))  # the level's guidance stands with its sources

    def test_feedback_notes_are_printed_as_typed_and_an_adjusted_area_is_drawn(self):
        r = v2_report()
        ev = r['keyMoments'][0]['id']
        fb = [{'id': 'f1', 'findingKey': r['needsWork'][0]['key'], 'attached': r['needsWork'][0]['key'],
               'thisRun': True, 'verdict': 'partial', 'text': 'A-04 was screened, B-10 unlikely', 'made':
               '2026-10-22T10:00:00', 'run': 'run-x'},
              {'id': 'f2', 'findingKey': r['needsWork'][0]['key'], 'attached': None, 'thisRun': True, 'verdict': None,
               'text': '', 'made': '2026-10-22T10:01:00', 'run': 'run-x',
               'adjustment': {'evidence': ev, 'polygon': [[.16, .35], [.22, .35], [.22, .41], [.16, .41]],
                              'mover': 'A-05', 'note': None}},
              {'id': 'f3', 'findingKey': None, 'thisRun': False, 'verdict': 'correct', 'text': 'old engine',
               'made': '2026-10-01', 'run': 'run-old', 'finding': {'headline': 'Old'}}]
        doc, info, ops = doc_ops(r, None, feedback=fb, labels={'A-04': 'A-09'})
        text = ' '.join(op['text'] for op in ops if op['op'] == 'text')
        self.assertIn('Your review: Partially correct - "A-04 was screened, B-10 unlikely"', text)   # never relabelled
        self.assertIn("coach's area", text)
        self.assertIn('Feedback from the previous report engine', text)
        self.assertIn('(adjusted area on E1)', text)

    def test_the_viewers_team_colours_choose_the_annotation_colours(self):
        _, _, ops = doc_ops(v2_report(), Frames2())
        strokes = {op.get('stroke') for op in ops}
        self.assertIn('#ff4d6d', strokes)                               # default teams: rose review outline
        self.assertIn('#448a5f', strokes)                               # the print twin of mint on the diagram
        self.assertIn('#7dffb0', strokes)                               # mint over the screenshot
        _, _, ops = doc_ops(v2_report(), Frames2(), colours={'A': '#ff0000', 'B': '#00ff00'})
        strokes = {op.get('stroke') for op in ops}
        self.assertIn('#ff2fd1', strokes)
        self.assertNotIn('#ff4d6d', strokes)

    def test_fixed_words_never_name_an_unmeasured_event_or_a_banned_word(self):
        import coaching_text_fixtures as TX
        for level in (1, 2, 4, 7):
            r = TX.report('unsupported_advance', 'none_behind', level=level)
            r['overview']['unavailable'] = []                       # profiles.json's reasons (INTERFACES: PDF)
            data, info = pdf.build_report_pdf(r, None, {})
            text = text_of(data)
            text = text[:text.rindex('Sources (')]+text[text.rindex('About this report'):]   # credits and guidance
            text = re.sub(r'Source: .*?PitchIQ proposal, not validated', '', text)            # drill credits
            self.assertEqual(S.event_problems(text), [], level)
            banned = [w for w in r['profile']['register']['banned']
                      if re.search(r'(?<![a-z])'+re.escape(w)+r'(?![a-z])', text, re.I)]
            self.assertEqual(banned, [], level)
            self.assertEqual(info['skippedElements'], [], level)
            self.assertEqual(info['evidenceShown'], len(r['keyMoments']))

    def test_real_scene_reports_through_every_stage(self):
        import coaching_text_fixtures as TX
        for scene, variant, level in (('unsupported_advance', 'none_behind', 4), ('low_calibration', 'default', 4),
                                      ('defending', 'press_no_cover', 2), ('kids_swarm', 'default', 1)):
            r = TX.report(scene, variant, level=level)
            data, info = pdf.build_report_pdf(r, Frames2(), {})
            parse(data)
            self.assertEqual(info['droppedLinks'], 0, scene)
            self.assertEqual(info['evidenceShown'], len(r['keyMoments']), scene)
            text = text_of(data)
            for f in r['needsWork']+r['reviews']+r['strengths']:
                self.assertIn(f['text']['headline'][:40], text, scene)
        r = TX.report('low_calibration', 'default', level=4)
        zones = [ev for ev in r['keyMoments'] if ev['alternative']]
        self.assertTrue(zones and all(ev['tier'] != 'world' for ev in zones))
        _, info, _ = doc_ops(r, Frames2())
        self.assertTrue(all(s['figure'] == 'overlay' for s in info['skippedElements'] if s['type'] == 'suggestedZone'))

    def test_the_page_and_the_pdf_cover_the_same_codes(self):
        self.assertEqual(set(pdf.REASON_UI), set(S.ABSTAIN_REASONS) | {'off'})
        self.assertEqual(set(pdf.GUARDRAIL_UI), set(S.GUARDRAILS))
        self.assertEqual(set(pdf.SKILL_OF.values()), set(S.SKILLS))
        self.assertEqual(set(pdf.DETECTOR_UI), set(S.DETECTORS))
        self.assertEqual(set(pdf.V2_VERDICTS), set(S.FINDING_VERDICTS))
        js = (ROOT/'coaching.js').read_text(encoding='utf-8')
        for name, table in (('REASON_UI', pdf.REASON_UI), ('GUARDRAIL_UI', pdf.GUARDRAIL_UI),
                            ('MEASURE_UI', pdf.MEASURE_UI), ('MASTERY_UI', pdf.MASTERY_UI),
                            ('ALT_REASON_UI', pdf.ALT_REASON_UI), ('DRILL_SOURCE_UI', pdf.DRILL_SOURCE_UI)):
            m = re.search(r'const '+name+r' = \{(.*?)\};', js, re.S)
            if m is None:
                self.skipTest(f'coaching.js has no {name} (FRONTEND changed it)')
            keys = set(re.findall(r"(?:^|[{,\s])'?([A-Za-z_][A-Za-z0-9_ ]*)'?:", m.group(1)))
            self.assertEqual(set(table) - keys, set(), name)


class V2FixesTests(unittest.TestCase):
    def test_clock_rounds_before_splitting_minutes(self):
        self.assertEqual(pdf.clock(59.999), '01:00.00')                 # critic N8 ('00:60.00' before)
        self.assertEqual(pdf.clock(3599.995), '60:00.00')
        self.assertEqual(pdf.clock(61.005), '01:01.01')
        self.assertEqual(pdf.clock(-2), '-00:02.00')
        self.assertEqual(pdf.offset(0.004), '+00:00.00')

    def test_a_long_video_name_never_runs_into_the_page_number(self):
        r = v2_report()
        r['video']['name'] = 'a very long upload name '*8+'.mp4'
        doc, _, _ = doc_ops(r)
        for page in doc.pages:
            foot = [op for op in page if op['op'] == 'text' and op['size'] == 7 and op['y'] > pdf.A4[1]-40]
            left = next(op for op in foot if op['align'] == 'left')
            right = next(op for op in foot if op['align'] == 'right')
            self.assertTrue(left['text'].endswith('...'))
            self.assertLess(left['x']+pdf.text_width(left['text'], 7), right['x']-pdf.text_width(right['text'], 7))
        fb = fixture_report()
        fb['video']['name'] = 'b'*300+'.mp4'
        doc, _ = pdf.layout_report(fb, None, {})
        foot = [op for op in doc.pages[0] if op['op'] == 'text' and op['size'] == 7 and op['y'] > pdf.A4[1]-40]
        self.assertTrue(next(op for op in foot if op['align'] == 'left')['text'].endswith('...'))

    def test_a_heading_keeps_the_block_that_follows_it(self):
        doc = pdf.Document()
        L = pdf._Layout(doc, 'header')
        L.y = L.bottom-80                                             # room for the heading, not for a card
        L.heading('Observed strengths', 1, keep=90)
        self.assertEqual(L.page, 1)                                   # critic N5: moved to the next page
        L.y = L.bottom-80
        L.heading('Short section', 1)
        self.assertEqual(L.page, 1)                                   # nothing to keep: it stays

    def test_a_section_heading_is_never_the_last_thing_on_a_page(self):
        for report in (v2_report(), fixture_report()):
            doc, _ = pdf.layout_report(report, Frames2(), {})
            for page in doc.pages[:-1]:
                body = [op for op in page if op['op'] == 'text' and op['y'] < pdf.A4[1]-pdf.MARGINS[3]]
                self.assertFalse(body and body[-1]['size'] >= 11.5, body[-1]['text'] if body else '')

    def test_the_v1_cover_mentions_illustrations_only_when_one_is_drawn(self):
        text = text_of(build(frames=Frames())[0])
        self.assertNotIn('COACHING ILLUSTRATION', text)
        r = fixture_report()
        r['evidence'][0]['diagram']['elements'].append({'type': 'suggested', 'id': 'A-01', 'x': .3, 'y': .3})
        self.assertIn('COACHING ILLUSTRATION', text_of(build(r, Frames())[0]))

    def test_the_file_name_carries_the_level_and_a_custom_range(self):
        r = v2_report()
        self.assertEqual(pdf.report_filename(r), 'pitchiq-coaching-example-run-20261021-A-level4.pdf')
        r['range'].update(t0=742.43, t1=800.01, whole=False)
        self.assertEqual(pdf.report_filename(r), 'pitchiq-coaching-example-run-20261021-A-level4-742-801s.pdf')
        self.assertEqual(pdf.pdf_filename('clip.avi', 'r1', 'A', 'competitive'),
                         'pitchiq-coaching-clip-r1-A-competitive.pdf')


COACH_ONLY = ('Who had the ball comes from the spans you marked (coach-marked); PitchIQ could not follow it on this '
              'footage.')


class V2FixRound3Tests(unittest.TestCase):
    """Verification round 2's CODE findings on the PDF: M-2 (the coach's possession marks were called an estimate),
    L-3 ("To build on it" twice), L-4 (the chips of the cited back-line failure mode), L-5 (the stale "No drill"
    sentence)."""

    @staticmethod
    def lay(report=None, **options):
        return pdf._ReportV2(v2_report() if report is None else report, None, options)

    def test_the_possession_row_says_where_the_shares_come_from(self):
        po = {'available': True, 'quality': 'limited', 'reason': None, 'roughShares': {'A': .75, 'B': .25},
              'coverage': .42, 'caveat': COACH_ONLY, 'source': 'coach'}
        say = self.lay(coach_spans=True, marked='both').possession_text
        self.assertEqual(say(po), 'Team A about 75%, Team B about 25% (from your marks, 42% of the time); limited quality')
        one = dict(po, roughShares=None, coverage=.31)
        self.assertEqual(self.lay(coach_spans=True, marked='A').possession_text(one),
                         "No share given: you marked only Team A's possession (31% of the time); mark Team B's spells too "
                         'for a share; limited quality')
        self.assertEqual(self.lay(coach_spans=True).possession_text(one), 'From your marks, 31% of the time; limited quality',
                         'the side unknown: still the coach\'s marks, never an estimate')
        mixed = dict(po, source='mixed', roughShares={'A': .35, 'B': .65}, coverage=1.0, quality='good')
        self.assertEqual(say(mixed), "Team A about 35%, Team B about 65% (your marks where you made them, PitchIQ's estimate "
                                     'elsewhere; from 100% of the time)')
        engine = dict(po, source='engine', roughShares={'A': .45, 'B': .55}, coverage=.79, quality='good')
        self.assertEqual(say(engine), 'Team A about 45%, Team B about 55% (estimated from 79% of the time)')
        old = {k: v for k, v in po.items() if k != 'source'}                    # a report stored before 'source' existed
        self.assertEqual(self.lay(coach_spans=True).possession_text(old),
                         'Team A about 75%, Team B about 25% (from your marks, 42% of the time); limited quality')
        self.assertEqual(self.lay().possession_text(dict(old, caveat=None)),
                         'Team A about 75%, Team B about 25% (estimated from 42% of the time); limited quality')
        full = dict(old, roughShares={'A': 1.0, 'B': 0.0})                       # the report's own one-sided signal
        self.assertTrue(self.lay(coach_spans=True).possession_text(full).startswith("No share given: you marked only Team A's"))

    def test_the_printed_overview_of_one_sided_marks(self):
        r = v2_report()
        r['overview']['possession'] = {'available': True, 'quality': 'limited', 'reason': None, 'roughShares': None,
                                       'coverage': .31, 'caveat': COACH_ONLY, 'source': 'coach'}
        r['overview']['detectors'][0].update(status='not_assessed', reason='coach_marks_one_side', opportunities=0)
        text = text_of(build2(r, coach_spans=True, marked='A')[0])
        self.assertIn("No share given: you marked only Team A's possession (31% of the time)", text)
        self.assertIn("not assessed: you marked only Team A's possession: mark Team B's spells too to run this check", text)
        self.assertNotIn('stimated from', text.split('Who had the ball')[1][:200])

    def test_a_strength_says_to_build_on_it_once(self):
        self.assertEqual(pdf.without_lead('To build on it, keep the cover close.'), 'Keep the cover close.')
        self.assertEqual(pdf.without_lead('Keep the cover close.'), 'Keep the cover close.')
        text = text_of(build2()[0])
        self.assertIn('TO BUILD ON IT Keep the cover close.', text)
        self.assertNotIn('To build on it,', text)

    def test_the_chips_of_the_cited_back_line_failure_mode(self):
        lay = self.lay()
        m = {'lateralGapM': 8.8, 'gapBallSide': False, 'opponentInGap': False, 'dmGapM': 12.2, 'dmGapOpponents': 1,
             'metres': 'estimated'}
        dm = {'observation': 'defensive_spacing', 'verdict': 'review', 'level': 4, 'measurements': dict(m, failureMode='dm_gap')}
        # (pre-freeze, critic N-4: the count is of the ball's side only, as contextual measures it)
        self.assertEqual(lay.measure_line(dm), 'At the first key moment: Widest gap in the back line 8.8 m (est.); Defence '
                                               'to midfield gap 12.2 m (est.); Opponents between defence and midfield '
                                               "on the ball's side 1.")
        lateral = dict(dm, measurements=dict(m, failureMode='lateral_gap', opponentInGap=True))
        self.assertEqual(lay.measure_line(lateral), 'At the first key moment: Widest gap in the back line 8.8 m (est.); Gap on '
                                                    'the ball side no; Opponent in the gap yes.')
        inferred = dict(dm, measurements={k: v for k, v in m.items() if k != 'lateralGapM'})   # no failureMode, no lateral gap
        self.assertIn("Opponents between defence and midfield on the ball's side 1", lay.measure_line(inferred))
        strength = dict(lateral, verdict='strength', level=5)
        self.assertNotIn('Opponents between', lay.measure_line(strength))

    def test_the_objective_origin_never_contradicts_its_text(self):
        # critic S-3 (COACH F-4 / FACTS F-9): an objective chosen from this match's checks must not read "not from this
        # match"; nor one from a work item whose finding is not shown.
        say = pdf.origin_line
        self.assertIsNone(say({'skill': 'spread_when_attacking', 'fromMatch': False,
                               'text': "From this match's checks (not a work item): Spread out."}, None))
        self.assertIsNone(say({'skill': 'pressure_and_cover', 'fromMatch': True, 'text': 'Pressure and cover.'}, None))
        self.assertEqual(say({'skill': 'pressure_and_cover', 'fromMatch': True, 'text': 'Pressure and cover.'}, {'id': 'f'}),
                         'From this match: see the finding linked below.')
        self.assertIsNone(say({'skill': 'pressure_and_cover', 'fromMatch': False,
                               'text': "From the level's objectives, not from this match: Pressure and cover."}, None))
        self.assertEqual(say({'skill': 'pressure_and_cover', 'fromMatch': False, 'text': 'Pressure and cover.'}, None),
                         "From the level's objectives, not from this match.")
        self.assertIsNone(say({'skill': None, 'fromMatch': False, 'text': 'Every player experiences all positions.'}, None))
        r = v2_report()
        r['nextObjective'] = {'skill': 'spread_when_attacking', 'fromMatch': False, 'finding': None,
                              'text': "From this match's checks (not a work item): Spread out."}
        self.assertNotIn('not from this match', text_of(build2(r)[0]))

    def test_too_few_situations_is_not_a_short_range(self):
        # critic S-2 / COACH F-14: 4 clear advances in 6.1 minutes read "the range is too short".
        lay = self.lay()
        d = {'detector': 'wide_defender_advance', 'status': 'not_assessed', 'reason': 'too_few_situations', 'opportunities': 4}
        self.assertEqual(lay.checked_text(d), 'not assessed: only 4 situations of this kind happened in this range, too few to judge')
        self.assertEqual(lay.checked_text(dict(d, opportunities=None)),
                         'not assessed: too few of these situations happened in this range to judge')
        lay.numbers = 'none'
        self.assertEqual(lay.checked_text(d), 'not assessed: too few of these situations happened in this range to judge')

    def test_the_moments_counts_add_up(self):
        # critic B-3 (FACTS F-4 / COACH F-10, round 2): failures that were not clear were left out of the count.
        f = {'observation': 'defensive_spacing', 'verdict': 'review', 'level': 4, 'limitations': ['moments_worth_a_look'],
             'aggregate': {'opportunities': 34, 'unclear': 24},
             'measurements': {'clearMoments': 7, 'clearSpells': 6, 'decidedMoments': 10, 'failingMoments': 8}}
        self.assertTrue(self.lay().fact_line(f).startswith('Clear in 7 of the 10 moments PitchIQ could judge (6 separate spells), 1 more '
                                                           'fell short less clearly, 24 unclear'))
        f['measurements']['failingMoments'] = 7
        self.assertNotIn('more fell short', self.lay().fact_line(f))

    def test_the_empty_drills_section_says_why(self):
        r = v2_report()
        r['drills'] = []
        self.assertIn("No drill: PitchIQ has none that fits the work items in this range for this team's age, level and "
                      'format.', text_of(build2(r)[0]))
        r['needsWork'] = []
        text = text_of(build2(r)[0])
        self.assertIn('No drill: there is no work item in this range, and PitchIQ has none that fits the next objective for '
                      "this team's age, level, format and range.", text)
        self.assertNotIn('never at random', text)
        r['nextObjective'] = {'skill': None, 'text': 'Every player experiences all positions.', 'fromMatch': False,
                              'finding': None}
        self.assertEqual(pdf.no_drill_text(r), 'No drill: there is no work item in this range, and the next objective is a '
                                               'general one with no drill tied to it.')
        r['nextObjective'] = None
        self.assertEqual(pdf.no_drill_text(r), 'No drill: there is no work item in this range and no next objective.')


if __name__ == '__main__':
    unittest.main()
