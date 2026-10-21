"""tools/possession_truth.py and tools/possession_sim.py on Metrica's two formats. A synthetic game written both ways,
as games 1-2 come (a tracking CSV per team and an events CSV) and as game 3 comes (FIFA EPTS metadata XML, a tracking
text file with one line per frame, JSON events), loads into the same arrays and gives the same truth payload byte
for byte. The EPTS reader maps each line's slots to players by the data format range of the frame (a substitute's
column is NaN until he comes on), joins subtype lists as the CSV spells them, takes Home from the local team,
corrects a listed identity swap over its frames only, and leaves CARRY to no rule. It refuses what it cannot read
safely: a document type declaration, another frame rate or pitch, a first column that is not the goalkeeper, an
event player who is not the metadata's, rows out of order.

With the Metrica data on this machine, games 1-2's truth payloads and simulated results are checked against the bytes
the tools gave before game 3 was added (sha256 recorded 2026-10-20 from the frozen v3 tools, possession_truth.py
52dcd523..., possession_sim.py cf944528...), and game 3's payload against its hash when it was built:

    set PITCHIQ_METRICA_TRACKING=<games 1-2 tracking folder>   PITCHIQ_METRICA_EVENTS=<games 1-2 events folder>
    set PITCHIQ_METRICA_GAME3=<game 3 folder>
    python -I -B -m unittest discover -s . -t . -p test_possession_truth.py     (from this directory; -I because the
                                                                                 data are untrusted downloads)
The real-data tests take about 30 s; without the variables they are skipped and the rest takes under a second.
"""
import hashlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

TOOLS = Path(__file__).resolve().parents[1]/'tools'
sys.path.insert(0, str(TOOLS))
import possession_truth as mt  # noqa: E402

FPS = 25.0
GAME = 7                  # the synthetic game's number (the files are Sample_Game_7_*)
T = 400
HALF = 200                # period 1 is frames 1-200, period 2 frames 201-400
HOME = [('P1', 'Player 11', 11, 'Goalkeeper'), ('P2', 'Player 1', 1, 'Right Back'), ('P3', 'Player 2', 2, 'Left Back'),
        ('P4', 'Player 12', 12, 'Left Back')]
AWAY = [('P5', 'Player 25', 25, 'Goalkeeper'), ('P6', 'Player 15', 15, 'Right Back'), ('P7', 'Player 16', 16, 'Left Back')]
SLOTS = [(1, HALF, ['P1', 'P2', 'P3', 'P5', 'P6', 'P7']), (HALF+1, T, ['P1', 'P2', 'P4', 'P5', 'P6', 'P7'])]   # Player 12 on for Player 2
BALL_OUT = (121, 160)     # untracked: a ball out of play, then the throw-in at 161
HALF_TIME = (196, 205)    # untracked across the period boundary


def positions():
    """Normalized (x, y) per frame for every player id and the ball, as 5-decimal strings (both formats print them so),
    'NaN' while untracked. Everyone stays within a few metres of the centre spot so no event is displaced."""
    f = np.arange(1, T+1)
    out = {}
    for k, (pid, *_rest) in enumerate(HOME+AWAY):
        x = .5+.03*np.sin(f/37.0+k)
        y = .5+.04*np.cos(f/29.0+2*k)
        out[pid] = [(f'{a:.5f}', f'{b:.5f}') for a, b in zip(x, y)]
    bx, by = .5+.02*np.sin(f/13.0), .5+.02*np.cos(f/17.0)
    ball = [(f'{a:.5f}', f'{b:.5f}') for a, b in zip(bx, by)]
    for a, b in (BALL_OUT, HALF_TIME):
        for i in range(a-1, b):
            ball[i] = ('NaN', 'NaN')
    out['ball'] = ball
    return out


# (team, type, subtypes (list), start, end, from, to): the events of both formats; CARRY only in the JSON.
EVENTS = [
    ('Home', 'SET PIECE', ['KICK OFF'], 5, 5, 'Player 1', None),
    ('Home', 'PASS', [], 5, 20, 'Player 1', 'Player 2'),
    ('Home', 'PASS', [], 40, 60, 'Player 2', 'Player 1'),
    ('Home', 'BALL LOST', ['INTERCEPTION'], 80, 85, 'Player 1', None),
    ('Away', 'RECOVERY', ['INTERCEPTION'], 86, 86, 'Player 15', None),
    ('Away', 'CHALLENGE', ['AERIAL', 'WON'], 90, 90, 'Player 15', None),
    ('Home', 'CHALLENGE', ['AERIAL', 'LOST'], 90, 90, 'Player 2', None),
    ('Away', 'PASS', ['HEAD'], 90, 110, 'Player 15', 'Player 16'),
    ('Away', 'BALL OUT', [], 115, 120, 'Player 16', None),
    ('Home', 'SET PIECE', ['THROW IN'], 161, 161, 'Player 2', None),
    ('Home', 'PASS', [], 161, 175, 'Player 2', 'Player 1'),
    ('Home', 'BALL LOST', ['END HALF'], 195, 195, 'Player 1', None),
    ('Away', 'SET PIECE', ['KICK OFF'], 206, 206, 'Player 15', None),
    ('Away', 'PASS', [], 206, 230, 'Player 15', 'Player 16'),
    ('Away', 'SHOT', ['OFF TARGET', 'OUT'], 250, 260, 'Player 16', None),
    ('Home', 'PASS', ['GOAL KICK'], 300, 330, 'Player 11', 'Player 12'),
    ('Home', 'BALL LOST', ['HEAD', 'INTERCEPTION'], 350, 360, 'Player 12', None),
    ('Away', 'RECOVERY', [], 361, 361, 'Player 16', None),
]
CARRIES = [('Home', 'CARRY', [], 20, 40, 'Player 2', None), ('Away', 'CARRY', [], 361, 390, 'Player 16', None)]
IDS = {name: pid for pid, name, *_rest in HOME+AWAY}
TEAM_ID = {'Home': 'FIFATMA', 'Away': 'FIFATMB'}


def write_csv_game(folder):
    """The game as Metrica's games 1-2 files."""
    pos = positions()
    for team, players in (('Home', HOME), ('Away', AWAY)):
        lines = [',,,'+''.join(f'{team},,' for _ in players)+',',
                 ',,,'+''.join(f'{j},,' for _, _, j, _ in players)+',',
                 'Period,Frame,Time [s],'+''.join(f'{n.replace(" ", "")},,' for _, n, _, _ in players)+'Ball,']
        for f in range(1, T+1):
            on = next(s for a, b, s in SLOTS if a <= f <= b)
            cells = []
            for pid, *_rest in players:
                cells += list(pos[pid][f-1]) if pid in on else ['NaN', 'NaN']
            lines.append(','.join([str(1 if f <= HALF else 2), str(f), repr(f/FPS)]+cells+list(pos['ball'][f-1])))
        (Path(folder)/f'Sample_Game_{GAME}_RawTrackingData_{team}_Team.csv').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    rows = ['Team,Type,Subtype,Period,Start Frame,Start Time [s],End Frame,End Time [s],From,To,Start X,Start Y,End X,End Y']
    for team, typ, sub, s, e, who, to in EVENTS:
        rows.append(','.join([team, typ, '-'.join(sub), str(1 if s <= HALF else 2), str(s), repr(s/FPS), str(e), repr(e/FPS),
                              who.replace(' ', ''), (to or '').replace(' ', ''), 'NaN', 'NaN', 'NaN', 'NaN']))
    (Path(folder)/f'Sample_Game_{GAME}_RawEventsData.csv').write_text('\n'.join(rows)+'\n', encoding='utf-8')


def metadata_xml(local='FIFATMA', fps=25, size=(105, 68), home=None, doctype=False):
    home = HOME if home is None else home
    players = ''.join(
        f'<Player id="{pid}" teamId="{tid}"><Name>{name}</Name><ShirtNumber>{j}</ShirtNumber><ProviderPlayerParameters>'
        f'<ProviderParameter><Name>position_type</Name><Value>{pos}</Value></ProviderParameter></ProviderPlayerParameters></Player>'
        for tid, group in (('FIFATMA', home), ('FIFATMB', AWAY)) for pid, name, j, pos in group)
    channels = ''.join(f'<PlayerChannel channelId="{c}" id="{pid}_{c}" playerId="{pid}"/>' for pid, *_rest in HOME+AWAY for c in 'xy')
    specs = ''.join(
        f'<DataFormatSpecification endFrame="{b}" separator=":" startFrame="{a}"><StringRegister name="frameCount"/>'
        '<SplitRegister separator=";">'
        + ''.join(f'<SplitRegister separator=","><PlayerChannelRef playerChannelId="{pid}_x"/><PlayerChannelRef playerChannelId="{pid}_y"/></SplitRegister>'
                  for pid in slots)
        + '</SplitRegister><SplitRegister separator=","><BallChannelRef channelId="x"/><BallChannelRef channelId="y"/></SplitRegister>'
        '</DataFormatSpecification>' for a, b, slots in SLOTS)
    visiting = 'FIFATMB' if local == 'FIFATMA' else 'FIFATMA'
    params = ''.join(f'<ProviderParameter><Name>{k}</Name><Value>{v}</Value></ProviderParameter>'
                     for k, v in (('first_half_start', 1), ('first_half_end', HALF), ('second_half_start', HALF+1), ('second_half_end', T),
                                  ('first_extra_half_start', ''), ('first_extra_half_end', '')))
    head = '<?xml version="1.0" encoding="utf-8"?>'+('<!DOCTYPE main [<!ENTITY a "b">]>' if doctype else '')
    return (head+'<main><Metadata><GlobalConfig><FrameRate>'+str(fps)+'</FrameRate><ProviderGlobalParameters>'+params+'</ProviderGlobalParameters>'
            '</GlobalConfig><Sessions><Session id="s"><MatchParameters><Score idLocalTeam="'+local+'" idVisitingTeam="'+visiting+'">'
            '<LocalTeamScore>0</LocalTeamScore><VisitingTeamScore>0</VisitingTeamScore></Score></MatchParameters>'
            f'<FieldSize><Width>{size[0]}</Width><Height>{size[1]}</Height></FieldSize></Session></Sessions>'
            '<Teams><Team id="FIFATMA"><Name>Team A</Name><ProviderTeamsParameters><ProviderParameter><Name>attack_direction_first_half</Name>'
            '<Value>left_to_right</Value></ProviderParameter></ProviderTeamsParameters></Team><Team id="FIFATMB"><Name>Team B</Name></Team></Teams>'
            '<Players>'+players+'</Players><PlayerChannels>'+channels+'</PlayerChannels></Metadata>'
            '<DataFormatSpecifications>'+specs+'</DataFormatSpecifications></main>')


def event_json(index, team, typ, sub, s, e, who, to, team_ids=TEAM_ID):
    shape = None if not sub else ({'name': sub[0], 'id': 1} if len(sub) == 1 else [{'name': x, 'id': 1} for x in sub])
    return {'index': index, 'team': {'name': 'Team A' if team == 'Home' else 'Team B', 'id': team_ids[team]}, 'type': {'name': typ, 'id': 1},
            'subtypes': shape, 'start': {'frame': s, 'time': s/FPS, 'x': None, 'y': None}, 'end': {'frame': e, 'time': e/FPS, 'x': None, 'y': None},
            'period': 1 if s <= HALF else 2, 'from': {'name': who, 'id': IDS[who]}, 'to': {'name': to, 'id': IDS[to]} if to else None}


def write_epts_game(folder, events=None, xml=None, lines=None):
    """The game as Metrica's game 3 files (EVENTS, then the carries, unless events is given)."""
    pos = positions()
    (Path(folder)/f'Sample_Game_{GAME}_metadata.xml').write_text(xml or metadata_xml(), encoding='utf-8')
    if lines is None:
        lines = []
        for f in range(1, T+1):
            on = next(s for a, b, s in SLOTS if a <= f <= b)
            lines.append(f'{f}:'+';'.join(','.join(pos[pid][f-1]) for pid in on)+':'+','.join(pos['ball'][f-1]))
    (Path(folder)/f'Sample_Game_{GAME}_tracking.txt').write_text('\n'.join(lines)+'\n', encoding='ascii')
    rows = events if events is not None else [event_json(k+1, *ev) for k, ev in enumerate(EVENTS+CARRIES)]
    (Path(folder)/f'Sample_Game_{GAME}_events.json').write_text(json.dumps({'data': rows, 'metadata': []}), encoding='utf-8')


def payload_bytes(folder, tmp):
    *_rest, payload = mt.build(GAME, folder, folder)
    path = Path(tmp)/'payload.json'
    mt.save(payload, path)
    return path.read_bytes()


class TwoFormats(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        cls.csv, cls.epts = root/'csv', root/'epts'
        cls.csv.mkdir()
        cls.epts.mkdir()
        write_csv_game(cls.csv)
        write_epts_game(cls.epts)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_same_arrays(self):
        a, b = mt.load_tracking(GAME, self.csv), mt.load_tracking(GAME, self.epts)
        self.assertNotIn('metadata', a)
        self.assertEqual(set(b)-set(a), {'metadata'})
        for k in ('frame', 'time', 'period', 'ball', 'Home', 'Away'):
            self.assertEqual(a[k].dtype, b[k].dtype, k)
            np.testing.assert_array_equal(a[k], b[k], err_msg=k)
        for k in ('names', 'jerseys', 'ids', 'team_of', 'column', 'game'):
            self.assertEqual(a[k], b[k], k)
        self.assertEqual(b['ids']['Home'], ['GK-A', 'A-01', 'A-02', 'A-03'])
        self.assertEqual(b['names']['Away'], ['Player25', 'Player15', 'Player16'])

    def test_same_events_carries_aside(self):
        a = mt.load_events(GAME, self.csv)
        b = mt.load_events(GAME, self.epts)
        self.assertEqual(len(b), len(a)+len(CARRIES))
        for x, y in zip(a, b):
            self.assertEqual(set(x), set(y))
            for k in x:
                if k in ('sx', 'sy', 'ex', 'ey'):
                    self.assertTrue(np.isnan(x[k]) and np.isnan(y[k]))
                else:
                    self.assertEqual(x[k], y[k], k)
        self.assertEqual([e['type'] for e in b[len(a):]], ['CARRY', 'CARRY'])
        self.assertEqual(b[6]['subtype'], 'AERIAL-LOST')          # a list joined as the CSV spells it
        self.assertEqual(b[3]['subtype'], 'INTERCEPTION')         # one {name}
        self.assertEqual(b[1]['subtype'], '')                     # none

    def test_same_truth_byte_for_byte(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(payload_bytes(self.csv, tmp), payload_bytes(self.epts, tmp))
        g = mt.load_game(GAME, self.epts, self.epts)
        kinds = [(c['kind'], c['from'], c['to']) for c in g['changes']]
        self.assertEqual(kinds, [('open', 'Home', 'Away'), ('restart', 'Away', 'Home'), ('restart', 'Away', 'Home'), ('open', 'Home', 'Away')])
        self.assertEqual(g['possession'][BALL_OUT[0]-1], mt.DEAD)
        self.assertEqual(g['possession'][HALF_TIME[0]-1], mt.DEAD)

    def test_substitute_column(self):
        b = mt.load_tracking(GAME, self.epts)
        j12, j2 = b['column']['Player12'][1], b['column']['Player2'][1]
        self.assertTrue(np.isnan(b['Home'][:HALF, j12]).all())
        self.assertTrue(np.isfinite(b['Home'][HALF:, j12]).all())
        self.assertTrue(np.isnan(b['Home'][HALF:, j2]).all())
        self.assertEqual(float(b['Home'][HALF, j12, 0]), float(positions()['P4'][HALF][0])*mt.LENGTH)
        self.assertEqual(b['period'].tolist(), [1]*HALF+[2]*(T-HALF))
        self.assertEqual(float(b['time'][0]), .04)

    def test_carries_change_no_truth(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_epts_game(tmp, events=[event_json(k+1, *ev) for k, ev in enumerate(EVENTS)])
            plain = mt.load_game(GAME, tmp, tmp)
        rows = sorted(EVENTS+CARRIES, key=lambda e: e[3])          # carries interleaved, as in game 3
        with tempfile.TemporaryDirectory() as tmp:
            write_epts_game(tmp, events=[event_json(k+1, *ev) for k, ev in enumerate(rows)])
            mixed = mt.load_game(GAME, tmp, tmp)
        np.testing.assert_array_equal(plain['possession'], mixed['possession'])
        np.testing.assert_array_equal(plain['phase'], mixed['phase'])
        keys = ('kind', 'from', 'to', 'wonAt', 'lostAt', 'wonBy', 'lostBy', 'wonType', 'lostType', 'oneTouch')
        self.assertEqual([[c[k] for k in keys] for c in plain['changes']], [[c[k] for k in keys] for c in mixed['changes']])

    def test_listed_swap_only_over_its_frames(self):
        plain = mt.load_tracking(GAME, self.epts)
        with mock.patch.dict(mt.EPTS_SWAPS, {GAME: (('Away', 'Player15', 'Player16', 301, 350),)}):
            swapped = mt.load_tracking(GAME, self.epts)
        a, b = plain['column']['Player15'][1], plain['column']['Player16'][1]
        np.testing.assert_array_equal(swapped['Away'][300:350, a], plain['Away'][300:350, b])
        np.testing.assert_array_equal(swapped['Away'][300:350, b], plain['Away'][300:350, a])
        np.testing.assert_array_equal(swapped['Away'][:300], plain['Away'][:300])
        np.testing.assert_array_equal(swapped['Away'][350:], plain['Away'][350:])
        np.testing.assert_array_equal(swapped['Home'], plain['Home'])

    def test_local_team_is_home(self):
        ids = {'Home': 'FIFATMB', 'Away': 'FIFATMA'}
        with tempfile.TemporaryDirectory() as tmp:
            write_epts_game(tmp, xml=metadata_xml(local='FIFATMB'), events=[event_json(1, 'Home', 'PASS', [], 5, 9, 'Player 15', 'Player 16', ids)])
            meta = mt.read_metadata(GAME, tmp)
            self.assertEqual(meta['teams']['Home']['id'], 'FIFATMB')
            self.assertEqual(meta['teams']['Home']['name'], 'Team B')
            ev = mt.load_events(GAME, tmp)
            self.assertEqual(ev[0]['team'], 'Home')
            self.assertEqual(mt.load_tracking(GAME, tmp)['names']['Home'], ['Player25', 'Player15', 'Player16'])

    def test_refusals(self):
        bad_order = [event_json(2, *EVENTS[0]), event_json(1, *EVENTS[1])]
        wrong_id = [dict(event_json(1, *EVENTS[1]), **{'from': {'name': 'Player 1', 'id': 'P3'}})]
        wrong_team = [dict(event_json(1, *EVENTS[1]), team={'name': 'Team B', 'id': 'FIFATMB'})]
        pos = positions()
        short = [f'{f}:'+';'.join(','.join(pos[pid][f-1]) for pid in SLOTS[0][2][:-1])+':NaN,NaN' for f in range(1, T+1)]
        gap = [f'{f}:'+';'.join(','.join(pos[pid][f-1]) for pid in next(s for a, b, s in SLOTS if a <= f <= b))+':NaN,NaN'
               for f in range(1, T+1) if f != 50]
        cases = {
            'doctype': dict(xml=metadata_xml(doctype=True)),
            'frame rate': dict(xml=metadata_xml(fps=30)),
            'pitch': dict(xml=metadata_xml(size=(100, 64))),
            'goalkeeper first': dict(xml=metadata_xml(home=HOME[1:2]+HOME[:1]+HOME[2:])),
            'slots': dict(lines=short),
            'frame gap': dict(lines=gap),
        }
        for name, kw in cases.items():
            with self.subTest(name), tempfile.TemporaryDirectory() as tmp:
                write_epts_game(tmp, **kw)
                with self.assertRaises(ValueError):
                    mt.load_tracking(GAME, tmp)
        for name, rows in (('order', bad_order), ('id', wrong_id), ('team', wrong_team)):
            with self.subTest(name), tempfile.TemporaryDirectory() as tmp:
                write_epts_game(tmp, events=rows)
                with self.assertRaises(ValueError):
                    mt.load_events(GAME, tmp)

    def test_simulator_takes_the_epts_game(self):
        import possession_sim as cs
        g = mt.load_game(GAME, self.epts, self.epts)
        for mode in ('full', 'camera'):
            r = cs.simulate(g, mode=mode, fps=30, seed=0).result()
            self.assertEqual(r['video']['frames'], len(r['frames']))
            self.assertEqual(len(r['roster']['slots']), 7)
            self.assertEqual({s['id'] for s in r['roster']['slots'] if s['role'] == 'GOALKEEPER'}, {'GK-A', 'GK-B'})
            ids = {p['id'] for f in r['frames'] for p in f['people'] if p['id']}
            self.assertLessEqual(ids, set(g['ids']['Home'])|set(g['ids']['Away']))
            sub = [s for s in r['roster']['slots'] if s['id'] == 'A-03'][0]
            self.assertGreater(sub['firstSeen'], (HALF-1)/FPS)


# ----------------------------------------------------------------------------------------------- the real games
TRACKING = os.environ.get('PITCHIQ_METRICA_TRACKING')
EVENTS_DIR = os.environ.get('PITCHIQ_METRICA_EVENTS')
GAME3 = os.environ.get('PITCHIQ_METRICA_GAME3')
# sha256 of mt.save(build(n)[3]) and of json.dumps(sim.result(600, 660)) with the v3 tools (before game 3 was added).
BEFORE = {1: {'payload': '490ea7ac523d472c112e12bb706ab90ed92d4661baf877a0e5eb3397ca80f776',
              'full': '17e4921e54b2e960f7c1820de95fe60aafff4d6f94ec681db1f0837f39274c0c',
              'camera': '3ab72cb5dd9eaf07db202ec29299b210585cc110d360cbf6dca619e188bbe43c'},
          2: {'payload': '78a609dd663eb3ce62e3dd0eda9764b9fae0e55dca15daa846b13c81e5d47461',
              'full': '7463d766ee02317e1a45c72d611fa1b8b78eea82a65348e3fca5a67c0ecad025',
              'camera': '5b57fab49b7956701e042896b9be1427d013597ae1fc7ce6006372f03cd225eb'}}
# sha256 of game 3's payload as built on 2026-10-20 (EPTS_SWAPS applied; scratch possession/truth/game3.json adds
# the validation block to these bytes).
GAME3_PAYLOAD = 'c79f3e3e0bc40f00bd4700eb82ab0349024bce8d2e38ae8d7b07dcd5fa7345a8'


def sha(b):
    return hashlib.sha256(b).hexdigest()


@unittest.skipUnless(TRACKING and EVENTS_DIR, 'set PITCHIQ_METRICA_TRACKING and PITCHIQ_METRICA_EVENTS to the games 1-2 folders')
class GamesOneTwoUnchanged(unittest.TestCase):
    def test_payloads_and_simulations_byte_identical(self):
        import possession_sim as cs
        for game, want in BEFORE.items():
            with self.subTest(game=game), tempfile.TemporaryDirectory() as tmp:
                *_rest, payload = mt.build(game, TRACKING, EVENTS_DIR)
                mt.save(payload, Path(tmp)/'p.json')
                self.assertEqual(sha((Path(tmp)/'p.json').read_bytes()), want['payload'])
                g = mt.load_game(game, TRACKING, EVENTS_DIR)
                for mode in ('full', 'camera'):
                    r = cs.simulate(g, mode=mode, fps=30, seed=0, dead_fill='hold', noise='camera').result(600.0, 660.0)
                    self.assertEqual(sha(json.dumps(r, sort_keys=False).encode()), want[mode], mode)


@unittest.skipUnless(GAME3, 'set PITCHIQ_METRICA_GAME3 to the game 3 folder')
class GameThree(unittest.TestCase):
    def test_truth_as_built(self):
        trk, ev, tr, payload = mt.build(3, GAME3, GAME3)
        self.assertEqual((len(trk['frame']), trk['Home'].shape[1], trk['Away'].shape[1]), (143761, 17, 18))
        self.assertEqual(tr['log']['displacedEvents'], [])
        on = {t: set(np.isfinite(trk[t][..., 0]).sum(1).tolist()) for t in mt.TEAMS}
        self.assertEqual(on, {'Home': {11}, 'Away': {11}})
        with tempfile.TemporaryDirectory() as tmp:
            mt.save(payload, Path(tmp)/'p.json')
            self.assertEqual(sha((Path(tmp)/'p.json').read_bytes()), GAME3_PAYLOAD)


if __name__ == '__main__':
    unittest.main()
