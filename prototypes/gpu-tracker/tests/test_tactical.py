"""The 2D match view.

1. The report's `tactical` block (pitch-coordinate description, calibrated share, goal ends).
2. The page itself, in a headless browser, against a synthetic calibrated clip: a panning, zooming
   broadcast camera over a rendered pitch with people drawn where known pitch positions project, and a
   result.json whose `pitch` fields are those positions. Checks the critical transformation test (a
   player on the centre spot stays mid-board while the camera pans), exclusion of referees and rejected
   people, interpolation, hold-then-hide for people the camera leaves, a dropped calibration, the ball's
   observed / predicted / lost states, play-pause-seek synchronisation and the colour pickers. Needs
   Playwright with a Chromium and ffmpeg; it is skipped otherwise.

Run:  python -m unittest test_tactical          (from this directory)
      PITCHIQ_CHROMIUM=C:\\path\\chrome.exe to use a particular browser executable.
"""
import json
import math
import os
import shutil
import subprocess
import tempfile
import threading
import unittest
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

import scene  # noqa: F401  (adds the prototype directory to sys.path)
import broadcast as B

ROOT = Path(__file__).resolve().parent.parent
try:
    import pipeline
    PIPELINE = True
except Exception:  # ultralytics / torch missing: the report builder cannot be imported
    PIPELINE = False
try:
    from playwright.sync_api import sync_playwright
    PLAYWRIGHT = True
except ImportError:
    PLAYWRIGHT = False


def chromium():
    """A browser executable: the one named in PITCHIQ_CHROMIUM, a pre-installed Playwright Chromium, or
    None for Playwright's own download."""
    for candidate in (os.environ.get('PITCHIQ_CHROMIUM'), '/opt/pw-browsers/chromium'):
        if candidate and os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    return None


class ReportTests(unittest.TestCase):
    @unittest.skipUnless(PIPELINE, 'needs the pipeline module (ultralytics, torch)')
    def test_report_describes_pitch_coordinates_calibration_and_goal_ends(self):
        tracker = SimpleNamespace(ids=SimpleNamespace(summary=lambda: {'players': [], 'goalkeepers': [], 'referees': [], 'retired': []},
                                                      goal_sides=lambda: {'left': {'A': 20, 'B': 2}, 'right': {'B': 16}}, keeper_slots=lambda: {}),
                                  events=[], issues=[], model=SimpleNamespace(to_json=lambda: {'learned': False}),
                                  ball=SimpleNamespace(track=0, rejections=Counter()), rejected_counts={}, cuts=0)
        frames = [{'time': i/25, 'frame': i, 'people': [], 'ball': {'state': 'UNKNOWN', 'confidence': 0.0}, 'calibrated': i < 40} for i in range(50)]
        meta = {'name': 'x.mp4', 'width': 1280, 'height': 720, 'fps': 25, 'frames': 50, 'duration': 2}
        report = pipeline.build_report(frames, tracker, meta, {'name': 'none'}, {}, 1.0, False, None)
        t = report['tactical']
        self.assertEqual((t['calibratedFrames'], t['calibratedShare']), (40, .8))
        self.assertEqual(t['goalEnds'], {'left': 'A', 'right': 'B'})
        self.assertIn('left goal line', t['coordinates'])
        self.assertEqual(report['version'], 3)


# ---------- a synthetic calibrated clip ----------
FPS, SECONDS, W, H = 25, 7.0, 1280, 720
GK_KIT = {'jersey': (40, 200, 40), 'shorts': (30, 30, 30), 'socks': (30, 30, 30)}
UNCALIBRATED, BALL_MISSING, BALL_UNKNOWN = (4.0, 5.5), (2.4, 3.0), (5.8, 6.4)


def path(points):
    def at(t):
        for (t0, p0), (t1, p1) in zip(points, points[1:]):
            if t <= t1:
                u = max(0.0, (t-t0)/(t1-t0))
                return (p0[0]+(p1[0]-p0[0])*u, p0[1]+(p1[1]-p0[1])*u)
        return points[-1][1]
    return at


PEOPLE = {'A-01': ('A', 'PLAYER', 'A', lambda t: (52.5, 34.0)),                                        # on the centre spot
          'A-02': ('A', 'PLAYER', 'A', path([(0, (52.5, 12)), (SECONDS, (52.5, 56))])),
          'A-03': ('A', 'PLAYER', 'A', path([(0, (20, 50)), (SECONDS, (16, 56))])),                     # the camera pans away from them
          'GK-1': ('A', 'GOALKEEPER', GK_KIT, path([(0, (3, 34)), (SECONDS, (4, 31))])),
          'B-01': ('B', 'PLAYER', 'B', path([(0, (45, 30)), (SECONDS, (62, 40))])),
          'B-02': ('B', 'PLAYER', 'B', lambda t: (70+8*math.cos(t/2), 34+8*math.sin(t/2))),
          'GK-2': ('B', 'GOALKEEPER', GK_KIT, path([(0, (102, 34)), (SECONDS, (101, 36))])),
          'REF-1': (None, 'REFEREE', 'REF', path([(0, (55, 40)), (SECONDS, (65, 35))]))}
BALL = path([(0, (31, 21)), (2, (46, 31)), (3.5, (55, 25)), (5, (52.5, 46)), (SECONDS, (75, 44))])
DISPLAY = {'GK-1': 'GK-A', 'GK-2': 'GK-B'}


def camera(t):
    u = t/SECONDS
    return B.camera(30+45*(3*u*u-2*u*u*u), 60+15*math.sin(math.pi*u))


def make_clip(out):
    rng = np.random.default_rng(1)
    frames_dir = out/'frames'
    frames_dir.mkdir()
    frames, truth = [], []
    for i in range(int(SECONDS*FPS)):
        t, P = i/FPS, camera(i/FPS)
        img = B.render(P, seed=0)
        calibrated = not UNCALIBRATED[0] <= t < UNCALIBRATED[1]
        people, gt = [], {}

        def place(x, y):
            fx, fy = B.image_point(P, x, y)
            h = B.player_px(P, x, y)/H
            return fx, fy, h, .02 <= fx <= .98 and .05 <= fy <= .985
        for n, (pid, (team, role, kit, at)) in enumerate(PEOPLE.items()):
            x, y = at(t)
            fx, fy, h, inside = place(x, y)
            gt[pid] = {'x': x/105, 'y': y/68, 'visible': inside}
            if not inside:
                continue
            w = h*.38*H/W
            box = [fx-w/2, fy-h, w, h]
            scene.draw_person(img, box, kit)
            person = {'time': round(t, 4), 'track': n+1, 'box': [round(v, 5) for v in box], 'score': .9, 'cls': role.lower(), 'zone': 'inside', 'evidence': 'observed',
                      'id': pid, 'display': DISPLAY.get(pid, pid), 'team': team, 'role': role, 'label': f'{role}_TEAM_{team}' if team else role,
                      'state': 'confirmed', 'identityConfidence': .9, 'roleConfidence': 1.0}
            if calibrated:
                person['pitch'] = [round((x+rng.normal(0, .2))/105, 4), round((y+rng.normal(0, .2))/68, 4)]
            people.append(person)
        fx, fy, h, inside = place(40, 70)   # bench staff on the near touchline: never in the tactical view
        if inside:
            w = h*.38*H/W
            scene.draw_person(img, [fx-w/2, fy-h, w, h], 'FAN')
            people.append({'time': round(t, 4), 'track': 99, 'box': [fx-w/2, fy-h, w, h], 'score': .8, 'cls': 'player', 'zone': 'outside', 'evidence': 'observed',
                           'id': None, 'display': 'OUT', 'team': None, 'role': 'UNKNOWN', 'label': 'REJECTED_OUTSIDE_FIELD', 'state': 'rejected',
                           'identityConfidence': 0, 'roleConfidence': 0, 'pitch': [40/105, 1.02]})
        bx, by = BALL(t)
        bfx, bfy, _, binside = place(bx, by)
        missing, unknown = BALL_MISSING[0] <= t < BALL_MISSING[1], BALL_UNKNOWN[0] <= t < BALL_UNKNOWN[1]
        if binside and not missing and not unknown:
            cv2.circle(img, (int(bfx*W), int(bfy*H)-4), 5, (245, 245, 245), -1, cv2.LINE_AA)
        if unknown or not binside:
            ball = {'state': 'UNKNOWN', 'phase': 'SEARCHING', 'confidence': 0.0}
        else:
            d = .012
            ball = {'state': 'MISSING' if missing else 'TRACKED', 'phase': 'OCCLUDED' if missing else 'LOCKED', 'observed': not missing,
                    'confidence': .7 if missing else .93, 'track': 1, 'box': [bfx-d/2, bfy-d, d, d*W/H], 'center': [bfx, bfy-d/2], 'missingFor': 0.2}
            if calibrated:
                ball['pitch'] = [round(bx/105, 4), round(by/68, 4)]
        gt['BALL'] = {'x': bx/105, 'y': by/68}
        frames.append({'time': round(t, 4), 'frame': i, 'people': people, 'ball': ball, 'ballCandidates': [], 'rejected': [], 'pitch': None, 'boundary': None,
                       'cameraReliable': True, 'cut': False, 'segment': 0, 'calibrated': calibrated})
        truth.append({'time': round(t, 4), **gt})
        cv2.imwrite(str(frames_dir/f'{i:04d}.png'), img)
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-framerate', str(FPS), '-i', str(frames_dir/'%04d.png'), '-c:v', 'libvpx-vp9', '-b:v', '1500k',
                    '-pix_fmt', 'yuv420p', str(out/'demo.webm')], check=True)
    shutil.rmtree(frames_dir)

    def identity(pid):
        team, role = PEOPLE[pid][0], PEOPLE[pid][1]
        return {'id': pid, 'longId': pid, 'display': DISPLAY.get(pid, pid), 'team': team, 'role': role, 'label': f'{role}_TEAM_{team}' if team else role,
                'status': 'ACTIVE', 'firstSeen': 0, 'lastSeen': SECONDS, 'observations': 10, 'identityConfidence': .9, 'teamConfidence': .9, 'roleConfidence': 1.0,
                'roleLocked': role != 'PLAYER', 'refereeConfidence': 0.0, 'goalkeeperConfidence': 0.0, 'roleHistory': [], 'gallerySize': 3,
                'lastBox': [.5, .5, .02, .05], 'lastPitch': None, 'velocity': [0, 0], 'exitEdge': '', 'restored': False, 'history': [],
                'official': 'CENTER_REFEREE' if role == 'REFEREE' else None, 'roleEvidence': {}}
    match = {'players': [identity(p) for p, v in PEOPLE.items() if v[1] == 'PLAYER'], 'goalkeepers': [identity(p) for p, v in PEOPLE.items() if v[1] == 'GOALKEEPER'],
             'referees': [identity(p) for p, v in PEOPLE.items() if v[1] == 'REFEREE'], 'retired': []}
    share = lambda n: round(n/len(frames), 3)
    states = Counter(f['ball']['state'] for f in frames)
    calibrated_frames = sum(1 for f in frames if f['calibrated'])
    meta = {'name': 'demo.webm', 'width': W, 'height': H, 'fps': FPS, 'frames': len(frames), 'duration': SECONDS}
    summary = {'processedFrames': len(frames), 'localTrackSegments': 9, 'globalIdentities': {'teamA': 3, 'teamB': 2, 'goalkeepers': 2, 'referees': 1, 'merged': 0},
               'reidentifications': 0, 'deferredDecisions': 0, 'newIdentities': 8, 'swapCorrections': 0, 'crossingWarnings': 0, 'sanityWarnings': 0,
               'identifiedShare': .97, 'rejectedDetections': {}, 'cameraCuts': 0, 'meanVisiblePeople': 7.0,
               'ball': {'trackedShare': share(states['TRACKED']), 'missingShare': share(states['MISSING']), 'unknownShare': share(states['UNKNOWN']),
                        'meanConfidence': .9, 'tracks': 1, 'reacquisitions': 1, 'lost': 1, 'phases': {}, 'rejectedCandidates': {}},
               'roles': {'refereeConversions': 0, 'goalkeepersIdentified': 2, 'goalkeeperTeams': 2, 'mergedIdentities': 0}, 'goalSides': {},
               'elapsedSeconds': 1.0, 'processingFPS': 10.0, 'peakGPUMemoryGB': 0}
    report = {'version': 3, 'video': meta, 'gpu': {'name': 'test', 'memoryGB': 0}, 'config': {}, 'frames': frames, 'match': match, 'tracks': {}, 'events': [],
              'issues': [], 'teams': {'learned': True}, 'continuedFrom': None, 'status': 'complete', 'summary': summary, 'labels': {}, 'limitations': [],
              'tactical': {'coordinates': 'normalized pitch position of the feet', 'calibratedFrames': calibrated_frames, 'calibratedShare': share(calibrated_frames),
                           'goalEnds': {'left': 'A', 'right': 'B'}}}
    (out/'result.json').write_text(json.dumps(report, separators=(',', ':')))
    (out/'info.json').write_text(json.dumps({'gpu': report['gpu'], 'landmarks': ['centre-spot'], 'uploads': [], 'video': meta, 'videoKey': 'test-key',
                                             'settings': {'start': 0, 'seconds': SECONDS, 'imageSize': 1280, 'maxPerTeam': 11, 'ballTiles': False, 'fieldFilter': True,
                                                          'touchlineMargin': .35, 'continueIdentities': False, 'boundaries': [], 'calibrations': []},
                                             'previousIdentities': None}))
    return truth


def serve(directory):
    """The review page and module from the repository with the clip's result and video (byte ranges for seeking)."""
    status = json.dumps({'state': 'complete', 'stage': 'Previous result loaded', 'progress': 1, 'hasResult': True, 'run': 'demo', 'gpu': {'name': 'test', 'memoryGB': 0}}).encode()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def reply(self, body, kind, code=200, extra=()):
            self.send_response(code)
            self.send_header('Content-Type', kind)
            self.send_header('Content-Length', str(len(body)))
            for k, v in extra:
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            route = self.path.split('?')[0]
            files = {'/': (ROOT/'index.html', 'text/html; charset=utf-8'), '/tactical.js': (ROOT/'tactical.js', 'application/javascript'),
                     '/api/info': (directory/'info.json', 'application/json'), '/api/result': (directory/'result.json', 'application/json')}
            if route in files:
                return self.reply(files[route][0].read_bytes(), files[route][1])
            if route == '/api/status':
                return self.reply(status, 'application/json')
            if route == '/video':
                data = (directory/'demo.webm').read_bytes()
                header = self.headers.get('Range', '')
                if header.startswith('bytes='):
                    a, b = header[6:].split('-')
                    start, end = int(a or 0), min(int(b) if b else len(data)-1, len(data)-1)
                    return self.reply(data[start:end+1], 'video/webm', 206, [('Content-Range', f'bytes {start}-{end}/{len(data)}'), ('Accept-Ranges', 'bytes')])
                return self.reply(data, 'video/webm', 200, [('Accept-Ranges', 'bytes')])
            self.reply(b'{}', 'application/json', 404)
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


@unittest.skipUnless(PLAYWRIGHT and shutil.which('ffmpeg'), 'needs Playwright (pip install playwright) and ffmpeg')
class PageTests(unittest.TestCase):
    TOL = (1.2/105, 1.2/68)   # 1.2 m

    @classmethod
    def setUpClass(cls):
        cls.directory = Path(tempfile.mkdtemp(prefix='pitchiq-tactical-'))
        cls.truth = make_clip(cls.directory)
        cls.server = serve(cls.directory)

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        shutil.rmtree(cls.directory, ignore_errors=True)

    def truth_at(self, t):
        return min(self.truth, key=lambda r: abs(r['time']-t))

    def near(self, m, gt):
        return m is not None and abs(m['x']-gt['x']) <= self.TOL[0] and abs(m['y']-gt['y']) <= self.TOL[1]

    @staticmethod
    def marker(state, pid):
        return next((p for p in state['players'] if p['id'] == pid), None)

    def test_page_recreates_the_match_from_pitch_coordinates(self):
        with sync_playwright() as p:
            try:
                browser = p.chromium.launch(executable_path=chromium())
            except Exception as error:
                self.skipTest(f'no Chromium for Playwright: {str(error)[:120]}')
            errors = []
            page = browser.new_page(viewport={'width': 1400, 'height': 1200})
            page.on('pageerror', lambda e: errors.append(str(e)))
            page.goto(f'http://127.0.0.1:{self.server.server_address[1]}/')
            page.wait_for_function("document.getElementById('summary').textContent.includes('frames')", timeout=60000)
            page.wait_for_function("document.getElementById('video').readyState >= 2", timeout=60000)
            page.evaluate("document.getElementById('video').muted = true")

            def seek(t):
                page.evaluate("t => new Promise(r => {const v=document.getElementById('video'); v.pause(); v.addEventListener('seeked', () => r(), {once: true}); v.currentTime=t;})", t)
                page.wait_for_timeout(150)
                return page.evaluate("() => tactical.state()")
            # The critical test: the player on the centre spot stays mid-board while the camera pans across the pitch.
            screen = []
            for t in (.5, 3.0, 6.5):
                s = seek(t)
                m = self.marker(s, 'A-01')
                box = page.evaluate("t => {const p=result.frames[frameIndex(t)].people.find(p=>p.id==='A-01'); return p ? p.box : null;}", t)
                self.assertTrue(m is not None and abs(m['x']-.5) <= self.TOL[0] and abs(m['y']-.5) <= self.TOL[1], (t, m))
                screen.append(box[0]+box[2]/2)
            self.assertGreater(max(screen)-min(screen), .3, 'the camera pan moved the player across the frame')
            # Everyone visible is drawn where the truth says; the referee and the rejected person are not.
            for t in (1.0, 6.5):
                s, gt = seek(t), self.truth_at(t)
                ids = {q['id'] for q in s['players']}
                expected = {pid for pid, v in gt.items() if pid not in ('time', 'BALL') and v['visible'] and not pid.startswith('REF')}
                self.assertTrue(expected <= ids, (t, expected-ids))
                self.assertNotIn('REF-1', ids)
                self.assertTrue(all(q['id'] for q in s['players']))
                self.assertEqual([pid for pid in expected if not self.near(self.marker(s, pid), gt[pid])], [], t)
                self.assertEqual({q['id']: (q['team'], q['role']) for q in s['players'] if q['role'] == 'GOALKEEPER'},
                                 {pid: (PEOPLE[pid][0], 'GOALKEEPER') for pid in expected if pid.startswith('GK')})
            self.assertEqual(self.marker(seek(1.0), 'A-02')['source'], 'OBSERVED')
            self.assertEqual(self.marker(seek(1.02), 'A-02')['source'], 'INTERPOLATED')
            # A person the camera pans away from: held briefly, then hidden rather than invented.
            last = max(r['time'] for r in self.truth if r['A-03']['visible'])
            self.assertEqual(self.marker(seek(last+.3), 'A-03')['source'], 'HELD')
            self.assertIsNone(self.marker(seek(last+1.0), 'A-03'))
            # The calibration is dropped between 4.0 and 5.5 s: held, then nothing, then back.
            self.assertEqual(self.marker(seek(4.3), 'A-01')['source'], 'HELD')
            s = seek(5.0)
            self.assertEqual((s['calibrated'], s['players']), (False, []))
            self.assertTrue(self.near(self.marker(seek(5.7), 'A-01'), {'x': .5, 'y': .5}))
            # The ball: observed, predicted while hidden, absent while lost.
            s = seek(1.6)   # a frame time (frame 40): the ball of that frame, not an interpolation
            self.assertTrue(s['ball'] and s['ball']['source'] == 'OBSERVED' and self.near(s['ball'], self.truth_at(1.6)['BALL']), s['ball'])
            self.assertEqual(seek(2.7)['ball']['source'], 'PREDICTED')
            self.assertIsNone(seek(6.1)['ball'])
            # Play, pause and seek follow the video clock.
            seek(.2)
            page.evaluate("document.getElementById('video').play()")
            page.wait_for_timeout(1200)
            video_t, s1 = page.evaluate("document.getElementById('video').currentTime"), page.evaluate("tactical.state()")
            self.assertGreater(video_t, .5)
            self.assertLessEqual(abs(s1['time']-video_t), .1)
            self.assertTrue(self.near(self.marker(s1, 'A-02'), self.truth_at(s1['time'])['A-02']))
            page.evaluate("document.getElementById('video').pause()")
            page.wait_for_timeout(200)
            a = page.evaluate("tactical.state().time")
            page.wait_for_timeout(300)
            self.assertEqual(a, page.evaluate("tactical.state().time"))
            self.assertLessEqual(abs(seek(1.0)['time']-1.0), .05)
            # Colour pickers recolour the markers at once and are remembered for this video.
            page.evaluate("() => {const i=document.getElementById('teamAColour'); i.value='#ff0000'; i.dispatchEvent(new Event('input'));}")
            page.wait_for_timeout(100)
            rgb = page.evaluate("""() => {const p=tactical.state().players.find(p=>p.id==='A-01'), c=document.getElementById('tactical'), T=Tactical.PITCH,
                scale=c.width/(T.length+2*T.margin), d=c.getContext('2d').getImageData(Math.round((T.margin+p.x*T.length)*scale), Math.round((T.margin+p.y*T.width)*scale)+3, 1, 1).data;
                return [d[0], d[1], d[2]];}""")
            self.assertTrue(rgb[0] > 200 and rgb[1] < 80 and rgb[2] < 80, rgb)
            self.assertIn('ff0000', page.evaluate("localStorage.getItem('pitchiq.tactical.test-key')"))
            browser.close()
            self.assertEqual(errors, [])


if __name__ == '__main__':
    unittest.main()
