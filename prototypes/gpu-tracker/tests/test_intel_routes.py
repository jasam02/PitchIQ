"""The Full Match Intelligence routes (fullmatch/routes.py with fullmatch/intel.py) through Flask's test client, on a
synthetic assembled match in a scratch folder (tests/intel_synth.py: the match data cache written where the stages look
for it, so no video and no views are read): the state built on request, its events (a goal and its kick-off), the
people's roster in the index, the halftime saved inside a segment and found again by a new server, wrong halftimes
refused, event corrections and their undo, roster corrections of the people.

Run:  python -m unittest test_intel_routes          (from this directory)
"""
import gzip
import json
import os
import sys
import tempfile
import threading
import time
import types
import unittest
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import intel_synth as S  # noqa: E402
from fullmatch import routes  # noqa: E402
from soccer.moments import data as md  # noqa: E402

MATCH = '20990101-000000-synth'


def write_match(root):
    """A synthetic assembled match under root/runs/<video>/matches/<id>: (video folder, match folder)."""
    m = S.Match(seconds=200.0)
    m.team('A')
    m.team('B')
    S.kickoff_layout(m, 10, 16, 'A')
    S.live(m, 17, 90)
    S.ball_path(m, 86, 90, (85.0, 33.0), (104.5, 34.0))            # team A scores at the right goal
    S.kickoff_layout(m, 100, 112, 'B')
    S.live(m, 113, 210)
    S.dead(m, 91, 112, 'stoppage', 'B')
    data, index = m.build()
    video = root/'runs'/'synthvideo'
    folder = video/'matches'/MATCH
    asm = folder/'assembly'
    (asm/'views').mkdir(parents=True)
    (asm/'views'/'seg0.json.gz').write_bytes(gzip.compress(b'{"frames": []}'))
    index['segments'] = [{'index': 0, 'key': 'seg0', 'start': float(data.t[0]), 'end': float(data.t[-1]), 'period': 0,
                          'cut0': 250, 'cut1': int(data.t[-1]*25), 'view': 'views/seg0.json.gz'}]
    index['video'] = {'fps': 25.0, 'width': 1920, 'height': 1080, 'name': 'synth.mp4'}
    (asm/'match.json').write_text(json.dumps(index), encoding='utf-8')
    (asm/'possession.json').write_text(json.dumps({'deadSpells': m.spells, 'timeline': []}), encoding='utf-8')
    (asm/'together.json').write_text('{}', encoding='utf-8')
    fp = md.source_fingerprint(folder)
    data.meta = dict(data.meta, fingerprint=fp, version=md.VERSION)
    md.save(data, asm/'coaching'/'moments'/f'data-{fp[:16]}.npz')
    return video, folder


def app_for(video_dir):
    from flask import Flask
    app = Flask(__name__)
    v = types.SimpleNamespace(directory=video_dir, source=None, key='synthvideo', meta={'fps': 25.0, 'name': 'synth.mp4'})
    context = {'video': lambda: v, 'runs': str(video_dir.parent), 'single_busy': lambda: False,
               'validate': lambda payload, meta: payload, 'release_models': lambda: None, 'preload_models': lambda: None,
               'seeds': lambda video: None}
    routes.register(app, context)
    return app.test_client()


class Routes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.video, cls.folder = write_match(Path(cls.tmp.name))
        cls.client = app_for(cls.video)

    @classmethod
    def tearDownClass(cls):
        # a correction can start the state's rebuild on a thread: let the builds finish before their folder goes
        for t in threading.enumerate():
            if t.name in ('match-intel', 'match-possession2', 'moments-build'):
                t.join(timeout=60)
        cls.tmp.cleanup()

    def intel(self, client=None):
        r = (client or self.client).get(f'/api/match/intel?id={MATCH}')
        self.assertEqual(r.status_code, 200)
        return r.get_json()

    def wait(self, client=None):
        for _ in range(300):
            d = self.intel(client)
            if d['state'] in ('ready', 'error'):
                return d
            time.sleep(.1)
        return d

    def post(self, path, body, client=None):
        r = (client or self.client).post(path, json=dict(body, match=MATCH))
        return r.status_code, r.get_json()

    def test_1_build_state_and_index(self):
        first = self.intel()
        self.assertIn(first['state'], ('missing', 'ready'))
        if first['state'] == 'missing':
            code, d = self.post('/api/match/intel/build', {})
            self.assertEqual(code, 200)
        d = self.wait()
        self.assertEqual(d['state'], 'ready', d.get('error'))
        goals = [e for e in d['events'] if e['type'] == 'GOAL']
        self.assertEqual([(g['team'], round(g['t'])) for g in goals], [('A', 90)])
        self.assertTrue(any(e['type'] == 'KICK_OFF' and e.get('periodStart') for e in d['events']))
        self.assertTrue(d['masks'])
        self.assertEqual(d['directionCheck']['disagree'], 0)
        index = self.client.get(f'/api/match/index?id={MATCH}').get_json()
        self.assertEqual(index['intel']['state'], 'ready')
        shown = {p['display'] for p in index['roster'] if not p.get('excluded')}
        self.assertTrue(all(len(x) <= 5 for x in shown), shown)          # 'A-10?' at most: never A-11 or more
        self.assertEqual(len(index['places']['A']), 10)

    def test_2_halftime_inside_a_segment_and_persistence(self):
        self.wait()
        code, d = self.post('/api/match/halftime', {'mode': 'marker', 'firstHalfEnd': 150.0, 'teamAFirstHalf': 'right'})
        self.assertEqual(code, 200, d)
        self.assertEqual([(p['start'], p['end'], p['attacksRight']) for p in d['intel']['directionPeriods']][1][2], 'B')
        d = self.wait()
        self.assertEqual(d['halftime']['firstHalfEnd'], 150.0)
        again = app_for(self.video)                                          # the server started again
        self.assertEqual(self.intel(again)['halftime']['firstHalfEnd'], 150.0)
        code, d = self.post('/api/match/halftime', {'mode': 'marker', 'firstHalfEnd': 9999.0, 'teamAFirstHalf': 'right'})
        self.assertEqual(code, 400)
        code, d = self.post('/api/match/halftime/reset', {})
        self.assertEqual(code, 200)
        self.assertIsNone(self.wait()['halftime']['firstHalfEnd'])          # one period: no halftime until set

    def test_3_event_corrections_and_undo(self):
        d = self.wait()
        goal = next(e for e in d['events'] if e['type'] == 'GOAL')
        code, r = self.post('/api/match/event', {'kind': 'confirm', 'id': goal['id']})
        self.assertEqual(code, 200, r)
        self.assertEqual(next(e for e in r['intel']['events'] if e['id'] == goal['id'])['status'], 'confirmed')
        code, r2 = self.post('/api/match/event/remove', {'cid': r['correction']['cid']})
        self.assertEqual(code, 200)
        self.assertNotEqual(next(e for e in r2['intel']['events'] if e['id'] == goal['id'])['status'], 'confirmed')
        code, r3 = self.post('/api/match/event', {'kind': 'add', 'type': 'SUBSTITUTION', 't': 120.0, 'team': 'B'})
        self.assertEqual(code, 200)
        self.assertIn(r3['correction']['event']['id'], {e['id'] for e in r3['intel']['events']})
        code, r4 = self.post('/api/match/event', {'kind': 'retype', 'id': goal['id'], 'type': 'NOT_A_TYPE'})
        self.assertEqual(code, 400)

    def test_4_roster_corrections_of_the_people(self):
        self.wait()
        index = self.client.get(f'/api/match/index?id={MATCH}').get_json()
        person = next(p for p in index['roster'] if p.get('team') == 'A' and p.get('slot') and not p.get('excluded'))
        code, r = self.post('/api/match/correction', {'kind': 'exclude', 'target': person['matchId']})
        self.assertEqual(code, 200, r)
        frag = person['fragments'][0]
        self.assertIn(frag, r['index']['excluded'])
        code, r2 = self.post('/api/match/correction', {'kind': 'include', 'target': frag})
        self.assertEqual(code, 200, r2)
        code, r3 = self.post('/api/match/correction', {'kind': 'confirm', 'target': frag, 'slot': 11})
        self.assertEqual(code, 400)


if __name__ == '__main__':
    unittest.main()
