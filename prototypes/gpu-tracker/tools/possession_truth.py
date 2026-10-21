"""The possession truth of the Metrica Sports sample games 1, 2 and 3 (per-frame team possession, the changes of
possession with lostAt / wonAt and who lost / won the ball, the dead-ball spells), derived from Metrica's events files
and checked against the tracking, and the reference scorer of the pre-declared metrics. tools/possession_bench.py
scores soccer/possession.py against it; tools/possession_sim.py turns the same games into PitchIQ results.

    python -I -B tools/possession_truth.py --tracking DIR --events DIR [--game 1] [--out DIR]   (writes game<n>.json)

    import sys; sys.path.insert(0, '<repo>/tools')
    import possession_truth as mt
    g = mt.load_game(1, TRACKING, EVENTS)    # tracking + truth: numpy arrays at 25 Hz, pitch metres (about 2.5 s)
    g['possession']                          # (T,) int8: 0 dead ball, 1 Home, 2 Away
    g['changes']                             # possession changes with lostAt / wonAt frames and players

TRACKING holds Sample_Game_<n>_RawTrackingData_{Home,Away}_Team.csv and EVENTS Sample_Game_<n>_RawEventsData.csv (the
Metrica Sports sample data, games 1-2). Game 3 was published in FIFA's EPTS format instead: one folder (pass it as both
TRACKING and EVENTS) with Sample_Game_3_metadata.xml, Sample_Game_3_tracking.txt and Sample_Game_3_events.json;
load_epts and load_events_json read it into exactly the structures the CSV loaders give (see load_epts for every
difference and how it is handled), so the derivation below runs unchanged on it. They are downloaded files, so run
this module and anything that imports it with python -I -B (-I keeps the script's own folder and the current
directory off sys.path), keep the scripts outside the data folders, and pass the folders as arguments. The truth is
derived on the fly (as fast as reading a cache: the tracking dominates); --out writes it as JSON (run lengths,
changes, dead spells, the derivation log) for inspection.

Conventions (PREDECLARED.md): possession persists from a team's control through its passes, shots and clearances
until the other team gains control (a change, timed twice: lostAt = the losing team's last action on the ball,
wonAt = the gaining team's first control) or the ball goes out of play (dead until the restart, which belongs to
the restarting team). Metrica's coding is followed literally: a RECOVERY is a gain of control even when the same
player loses the ball on that frame (a first-time interception or header; such changes are flagged oneTouch), so
the truth keeps every short possession the annotators saw and the evaluation can leave them out by the flag.

Index k of every per-frame array is Metrica frame k+1: the tracking frames run 1..T without gaps (checked) and the
events' Start / End Frame use the same numbering. Positions: x 0..105 m from the left goal line, y 0..68 m from the
top touchline of Metrica's picture (PitchIQ's far touchline, y 0), NaN while untracked. Period 2 continues the clock;
times are made monotonic as tools/metrica_bench.py does.

Metrica leaves the ball untracked (NaN) exactly while it is out of play: in game 1 the ball is NaN on 39.1% of the
frames and every long NaN run starts on the frame after an event that stops play (foul 110001 -> NaN from 110002,
goal 2309 -> 2310, end of half 70578 -> 70579) and ends on the frame before the restart (kick-off 3675 <- 3674).
The truth is derived from the events alone wherever they are consistent; the tracking is used (and logged) only to
place a stoppage the events leave open.

Checks on game 1, where every rule was developed (game 2 ran the same code afterwards; its counts in brackets): the
ball is within 1 m of the coded player on 91.6% of the 2513 events (92.4%; median .11 m, the 32 beyond 5 m are
timing apart from 4 miscodes, 3 dropped as displaced); 99.4% of the 320 changes (294) follow a losing action of the
losing team (94.6%: game 2 codes more gains implicitly, 12 against 2); 99.1% of the untracked ball frames are dead
(98.3%) and 86.1% of the dead frames untracked (96.5%); spell boundaries meet an untracked run within 2 frames on 85%
of the spells. Game 1: 248 changes in play and 72 across a stoppage (6.0 per in-play minute), 79 one-touch
possessions, a loose ball (after the loser's last action, before the winner's control: still the loser's) on 11% of
the in-play time. A trivial rule (the team whose nearest player is within 1 m, the other's beyond, else the last
team; dead while untracked) agrees on .914 of the in-play frames at .992 coverage with change precision .52 and
recall .85 within 2 s: what any engine must beat. These checks were made by a validation script that is not part of
the repository (possession truth role, 2026-10-20).

Game 3 is held out for the engine; its truth was built on 2026-10-20 by this unchanged derivation and checked with
counts and consistency checks only (no engine output was looked at). Once EPTS_SWAPS corrects its tracking, no event
is displaced (57 are without it). The ball is within 1 m of the coded player on 91.5% of 3334 coded positions
(carries aside; median .10 m; the 8 beyond 5 m are all duels). 98.6% of the 367 changes follow a losing action. 99.3%
of the untracked ball frames are dead and 99.7% of the dead frames untracked, and all 104 spells meet an untracked
run (within 2 frames on 93% of the starts and 99% of the ends; only the half-time pair is more than 2 s off). The 1395
carries, which no rule reads, lie under the carrier's team on 99.6% of their in-play frames. Game 3 has 284 changes
in play and 83 across a stoppage (5.6 per in-play minute), 101 one-touch possessions, and a loose ball on 10% of the
in-play time. The CSV games' derivation, payloads and simulations are byte-identical to before game 3 was added.
"""
import argparse
import csv
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

FPS = 25.0
LENGTH, WIDTH = 105.0, 68.0
TEAMS = ('Home', 'Away')
DEAD, HOME, AWAY = 0, 1, 2
CODE = {'Home': HOME, 'Away': AWAY}
NAME = {DEAD: 'dead', HOME: 'Home', AWAY: 'Away'}
LETTER = {'Home': 'A', 'Away': 'B'}   # tools/team_shape_bench.py: home is team A, away team B
OTHER = {'Home': 'Away', 'Away': 'Home'}
# Phase codes (a second per-frame array): what the ball is doing under the possession label.
PH_DEAD, PH_HELD, PH_FLIGHT, PH_LOOSE = 0, 1, 2, 3
PHASES = {PH_DEAD: 'dead', PH_HELD: 'held', PH_FLIGHT: 'pass', PH_LOOSE: 'loose'}
# Action priorities on one frame: an end of play first (a RECOVERY coded on the frame of the recovering player's own
# foul is no gain: game 1 frame 38457, Away RECOVERY + TACKLE-FAULT-LOST, Home free kick), then the restart, then
# gains, then other controls, then the reception of a pass, then touches and losing actions.
P_DEAD, P_RESTART, P_GAIN, P_CONTROL, P_RECEIVE, P_TOUCH = range(6)
LATE_LOSS = 25         # frames (1 s): a BALL LOST of the team that just lost the ball, coded up to this long after the
                       # other team's RECOVERY, is the same loss coded late, not a regain (game 1: THEFT coded 1 frame
                       # after the RECOVERY at 5627/5628, 0.84 s after it at 115976/115997, 0.52 s at 116601/116614)
STOP_BEFORE = 25       # frames: an untracked-ball run that starts within this many frames before a stoppage ...
STOP_AFTER = 75        # ... or this many after it confirms that play stopped there (only consulted when the events leave
                       # the stoppage open: play coded between it and the next restart, or a restart with no stoppage)
MIN_STOP = 25          # frames: the shortest untracked run taken as a stoppage (a 1 s dropout of the ball is not one)
PERIOD_REASONS = ('before kick-off', 'end of period')
CLUSTER = 50           # frames (2 s): dead actions this close are one incident (a CHALLENGE ...FAULT-LOST, its FAULT-WON
                       # and the FAULT RECEIVED: up to 21 frames apart in game 1)
DISPLACED_M = 15.0     # metres: an event whose From player is further than this from the tracked ball on every frame
DISPLACED_SEARCH = 12  # within this many frames of its start is dropped as miscoded (game 1: event 1529, an away
                       # goal kick coded 58 m from the ball while the home goal kick was in flight; 1527 at 83 m)
DISPLACED_TYPES = ('RECOVERY', 'PASS', 'BALL LOST', 'SHOT', 'BALL OUT', 'SET PIECE')


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return math.nan


def _name(v):
    return (v or '').replace(' ', '')     # game 2's away header has "Player 26", its events "Player26"


# ----------------------------------------------------------------------------------------------- loading

def load_tracking(game, data):
    """Tracking of one game: dict(frame (T,) int Metrica frame numbers 1..T, time (T,) s monotonic, period (T,),
    ball (T, 2) m, Home (T, P, 2) m, Away (T, Q, 2) m (NaN while a player is off), names {team: [PlayerN, ...]},
    jerseys {team: [n, ...]}, ids {team: [id, ...]} (tools/team_shape_bench.py ids: GK-A / GK-B for the first
    column, the goalkeeper's, else A-01 ... by column), team_of {PlayerN: team}, column {PlayerN: (team, j)}).
    The ball comes from the home file (both files carry the same one). A folder that holds the game in the EPTS
    format instead (game 3) is read by load_epts."""
    if _is_epts(game, data):
        return load_epts(game, data)
    out = {'game': int(game), 'names': {}, 'jerseys': {}, 'ids': {}, 'team_of': {}, 'column': {}}
    for team in TEAMS:
        path = Path(data)/f'Sample_Game_{game}_RawTrackingData_{team}_Team.csv'
        with open(path, encoding='utf-8', newline='') as fh:
            reader = csv.reader(fh)
            next(reader)
            jerseys = next(reader)
            header = next(reader)
            rows = [r for r in reader if len(r) >= len(header)-1]
        names = [_name(h) for h in header[3:-2:2]]
        jer = [int(j) if j.strip().isdigit() else None for j in jerseys[3:-2:2]]
        width = 3+2*len(names)+2
        body = np.array([[_num(v) for v in r[:width]] for r in rows], np.float64)
        period = body[:, 0].astype(np.int64)
        frame = body[:, 1].astype(np.int64)
        times = body[:, 2].copy()
        pos = body[:, 3:3+2*len(names)].reshape(len(body), -1, 2).copy()
        ball = body[:, 3+2*len(names):width].copy()
        pos[:, :, 0] *= LENGTH
        pos[:, :, 1] *= WIDTH
        ball[:, 0] *= LENGTH
        ball[:, 1] *= WIDTH
        for k in range(1, len(times)):
            if times[k] <= times[k-1]:
                times[k:] += times[k-1]-times[k]+1/FPS
        if frame[0] != 1 or not (np.diff(frame) == 1).all():
            raise ValueError(f'{path.name}: frames are not 1..T without gaps')
        out[team] = pos
        out['names'][team] = names
        out['jerseys'][team] = jer
        letter = LETTER[team]
        out['ids'][team] = [('GK-'+letter) if j == 0 else f'{letter}-{j:02d}' for j in range(len(names))]
        for j, n in enumerate(names):
            out['team_of'][n] = team
            out['column'][n] = (team, j)
        if team == 'Home':
            out['frame'], out['time'], out['period'], out['ball'] = frame, times, period, ball
        elif len(frame) != len(out['frame']):
            raise ValueError('home and away files differ in length')
    return out


def load_events(game, events, metadata=None):
    """The events of one game in file order: dicts with i (row), team, type, subtype, period, start, end (Metrica
    frames; an end before the start, the kick-off's 0, becomes the start), From, To, sx, sy, ex, ey (metres, NaN when
    not given). Game 3's Sample_Game_3_events.json is read by load_events_json when the folder has no CSV; it needs
    the game's metadata (in `metadata`, default the events folder) for which team is Home."""
    path = Path(events)/f'Sample_Game_{game}_RawEventsData.csv'
    if not path.is_file() and (Path(events)/f'Sample_Game_{game}_events.json').is_file():
        return load_events_json(game, events, events if metadata is None else metadata)
    out = []
    with open(path, encoding='utf-8', newline='') as fh:
        for i, r in enumerate(csv.DictReader(fh)):
            s = int(float(r['Start Frame']))
            e = int(float(r['End Frame'] or 0))
            out.append({'i': i, 'team': r['Team'], 'type': r['Type'], 'subtype': r['Subtype'] or '', 'period': int(r['Period']),
                        'start': s, 'end': e if e >= s else s, 'From': _name(r['From']) or None, 'To': _name(r['To']) or None,
                        'sx': _num(r['Start X'])*LENGTH, 'sy': _num(r['Start Y'])*WIDTH,
                        'ex': _num(r['End X'])*LENGTH, 'ey': _num(r['End Y'])*WIDTH})
    return out


# ----------------------------------------------------------------------------------------------- game 3: EPTS + JSON

EPTS_PERIODS = ('first_half', 'second_half', 'first_extra_half', 'second_extra_half')   # metadata period names, 1..4
# Identity corrections of the EPTS tracking, measured, not guessed: {game: ((team, name, name, first frame, last
# frame), ...)} swaps the two players' tracking columns over those frames. Game 3's tracking labels Player6 and
# Player13 the wrong way round for the whole second half: all 187 of their period-2 events with both players tracked
# lie at the other one's column (ball within .03 m of it at the median, 18.99 m from the named player's own), none at
# their own, while every other event player of games 1 and 3 sits at his own column (no other team-mate within 2 m
# when the named one is beyond 3 m). Without the swap 57 events (41 of Player13, 16 of Player6) would be dropped as
# displaced. The data cannot prove which file is wrong. The columns are swapped rather than the events relabelled
# because the metadata sides with the events: it lists Player 13 (on for Player 8 at half-time) as a right centre
# midfielder, and the column labelled Player6 is the one on the right after half-time (29.2 m from the attacker's
# right touchline on average, against 41.9 m). Relabelling the events instead would give the same possession, dead
# spells and change times; only the ids these two players carry in period 2 (A-06 / A-12) depend on the choice.
EPTS_SWAPS = {3: (('Home', 'Player6', 'Player13', 69662, 143761),)}


def _is_epts(game, folder):
    """Game <n> is in this folder as EPTS (metadata XML + tracking text) and not as Metrica's CSV pair."""
    folder = Path(folder)
    return (not (folder/f'Sample_Game_{game}_RawTrackingData_Home_Team.csv').is_file()
            and (folder/f'Sample_Game_{game}_metadata.xml').is_file())


def read_metadata(game, folder):
    """The EPTS metadata of one game (Sample_Game_<n>_metadata.xml) as a dict: fps, pitch [length, width] m, periods
    {period: [first frame, last frame]}, teams {'Home'|'Away': {'id', 'name', 'attackFirstHalf'}} (Home is the
    idLocalTeam of the score), players [{'id', 'team', 'name', 'jersey', 'position'}] in file order and specs
    [{'start', 'end', 'slots': [player id, ...]}]: which player each ';'-separated slot of a tracking line holds over
    a range of frames (a substitution starts a new range). The file is untrusted: a document type declaration (the
    vehicle of entity expansion) is refused, and any structure other than Metrica's is an error, not a guess."""
    path = Path(folder)/f'Sample_Game_{game}_metadata.xml'
    raw = path.read_bytes()
    if b'<!DOCTYPE' in raw or b'<!ENTITY' in raw:
        raise ValueError(f'{path.name}: a document type declaration is not accepted')
    root = ET.fromstring(raw)
    md = root.find('Metadata')
    params = {p.findtext('Name'): (p.findtext('Value') or '').strip() for p in md.findall('GlobalConfig/ProviderGlobalParameters/ProviderParameter')}
    periods = {}
    for k, key in enumerate(EPTS_PERIODS, 1):
        a, b = params.get(key+'_start', ''), params.get(key+'_end', '')
        if a and b:
            periods[k] = [int(a), int(b)]
    session = md.find('Sessions/Session')
    score = session.find('MatchParameters/Score')
    side = {score.get('idLocalTeam'): 'Home', score.get('idVisitingTeam'): 'Away'}
    teams = {}
    for t in md.findall('Teams/Team'):
        tp = {p.findtext('Name'): p.findtext('Value') for p in t.findall('ProviderTeamsParameters/ProviderParameter')}
        teams[side[t.get('id')]] = {'id': t.get('id'), 'name': t.findtext('Name'), 'attackFirstHalf': tp.get('attack_direction_first_half')}
    if sorted(teams) != sorted(TEAMS):
        raise ValueError(f'{path.name}: expected a local and a visiting team, got {sorted(side)}')
    players = []
    for p in md.findall('Players/Player'):
        pp = {q.findtext('Name'): q.findtext('Value') for q in p.findall('ProviderPlayerParameters/ProviderParameter')}
        players.append({'id': p.get('id'), 'team': side[p.get('teamId')], 'name': p.findtext('Name'), 'jersey': int(p.findtext('ShirtNumber')),
                        'position': pp.get('position_type')})
    channels = {c.get('id'): (c.get('playerId'), c.get('channelId')) for c in md.findall('PlayerChannels/PlayerChannel')}
    specs = []
    for s in root.findall('DataFormatSpecifications/DataFormatSpecification'):
        kids = list(s)
        if (s.get('separator') != ':' or len(kids) != 3 or kids[0].tag != 'StringRegister' or kids[0].get('name') != 'frameCount'
                or kids[1].get('separator') != ';' or kids[2].get('separator') != ','
                or [b.get('channelId') for b in kids[2]] != ['x', 'y']):
            raise ValueError(f'{path.name}: unexpected data format at frames {s.get("startFrame")}-{s.get("endFrame")}')
        slots = []
        for reg in kids[1]:
            xy = [channels[r.get('playerChannelId')] for r in reg]
            if reg.get('separator') != ',' or len(xy) != 2 or xy[0][0] != xy[1][0] or [c for _, c in xy] != ['x', 'y']:
                raise ValueError(f'{path.name}: a player slot is not one player x,y')
            slots.append(xy[0][0])
        specs.append({'start': int(s.get('startFrame')), 'end': int(s.get('endFrame')), 'slots': slots})
    return {'fps': float(md.findtext('GlobalConfig/FrameRate')), 'pitch': [float(session.findtext('FieldSize/Width')), float(session.findtext('FieldSize/Height'))],
            'periods': periods, 'teams': teams, 'players': players, 'specs': specs}


def load_epts(game, data):
    """load_tracking's dict for a game in the EPTS format (Metrica sample game 3), made to match the CSV games'
    conventions exactly. The differences and how they are handled:
    - One text file, a line per frame 'frame:x,y;x,y;...:ball x,ball y', instead of a CSV per team. A line holds
      only the 22 players on the pitch, in slots whose player the metadata's DataFormatSpecification of that frame
      range names; a substitution starts a new range (game 3: 11 ranges, 13 substitutions). Each player gets his
      own column for the whole game, NaN outside the ranges he is in, as a CSV substitute's column is NaN until he
      comes on.
    - Columns: the metadata's player order per team, which is the CSV games' (the goalkeeper, the starters by shirt
      number, then the substitutes: game 3 Home Player11, Player1 ... Player17, Away Player28, Player18 ...
      Player35); the first column must be the goalkeeper (GK-A / GK-B), else this refuses the file. Names lose
      their space as in games 1-2 ("Player 11" -> "Player11"); jerseys are the metadata's shirt numbers.
    - No period or time column: the period comes from the metadata's first_half / second_half frame ranges and the
      time is frame / 25, which is what games 1-2's Time [s] column holds (frame 1 at 0.04 s, period 2 continuing
      the clock).
    - Home is the metadata's local team (idLocalTeam: "Team A"), Away the visiting one.
    - The frame rate and the pitch must be the metadata's 25 Hz and 105 x 68 m (the truth, the scorer and
      tools/possession_sim.py assume them), else this refuses the file. Positions are normalized with y from the
      top touchline of Metrica's picture, as in the CSV games (game 3: every right back plays on the attacker's
      right of that convention in both halves, and the events' own x, y lie a median .07 m from the tracked ball).
    - The tracking's identity errors listed in EPTS_SWAPS are corrected (game 3: Player6 and Player13 over the
      second half), so that the events' players and the tracking columns name the same person, as they do in game 1.
    The dict also carries 'metadata' (read_metadata, plus 'playerIds' {PlayerN: metadata id}); the CSV games have
    none."""
    meta = read_metadata(game, data)
    path = Path(data)/f'Sample_Game_{game}_tracking.txt'
    if meta['fps'] != FPS or meta['pitch'] != [LENGTH, WIDTH]:
        raise ValueError(f'{path.name}: {meta["fps"]} Hz and a {meta["pitch"]} m pitch, not {FPS} Hz and {[LENGTH, WIDTH]} m')
    cols = {t: [p for p in meta['players'] if p['team'] == t] for t in TEAMS}
    where = {p['id']: (t, j) for t in TEAMS for j, p in enumerate(cols[t])}
    frames, slots, balls = [], [], []
    with open(path, encoding='ascii', newline='') as fh:
        for line in fh:
            line = line.rstrip('\r\n')
            if not line:
                continue
            f, people, ball = line.split(':')
            frames.append(int(f))
            slots.append([_num(v) for v in people.replace(';', ',').split(',')])
            balls.append([_num(v) for v in ball.split(',')])
    frame = np.array(frames, np.int64)
    if frame[0] != 1 or not (np.diff(frame) == 1).all():
        raise ValueError(f'{path.name}: frames are not 1..T without gaps')
    T = len(frame)
    out = {'game': int(game), 'names': {}, 'jerseys': {}, 'ids': {}, 'team_of': {}, 'column': {}}
    norm = {t: np.full((T, len(cols[t]), 2), np.nan) for t in TEAMS}
    covered = np.zeros(T, np.int64)
    for s in meta['specs']:
        rows = np.flatnonzero((frame >= s['start']) & (frame <= s['end']))
        covered[rows] += 1
        if any(len(slots[k]) != 2*len(s['slots']) for k in rows):
            raise ValueError(f'{path.name}: a line of frames {s["start"]}-{s["end"]} does not have {len(s["slots"])} players')
        block = np.array([slots[k] for k in rows], np.float64).reshape(len(rows), -1, 2)
        for k, pid in enumerate(s['slots']):
            team, j = where[pid]
            norm[team][rows, j] = block[:, k]
    if (covered != 1).any():
        raise ValueError(f'{path.name}: {int((covered != 1).sum())} frames are in no data format range or in several')
    names_of = {t: [_name(p['name']) for p in cols[t]] for t in TEAMS}
    for team, a, b, f0, f1 in EPTS_SWAPS.get(int(game), ()):
        ja, jb = names_of[team].index(a), names_of[team].index(b)
        rows = (frame >= f0) & (frame <= f1)
        norm[team][rows, ja], norm[team][rows, jb] = norm[team][rows, jb].copy(), norm[team][rows, ja].copy()
    period = np.zeros(T, np.int64)
    for p, (a, b) in meta['periods'].items():
        period[(frame >= a) & (frame <= b)] = p
    if (period == 0).any():
        raise ValueError(f'{path.name}: {int((period == 0).sum())} frames are in no period')
    times = frame/FPS
    for k in range(1, len(times)):
        if times[k] <= times[k-1]:
            times[k:] += times[k-1]-times[k]+1/FPS
    if any(len(b) != 2 for b in balls):
        raise ValueError(f'{path.name}: a line has no ball x,y')
    ball = np.array(balls, np.float64)
    ball[:, 0] *= LENGTH
    ball[:, 1] *= WIDTH
    for team in TEAMS:
        pos = norm[team]
        pos[:, :, 0] *= LENGTH
        pos[:, :, 1] *= WIDTH
        out[team] = pos
        if cols[team][0]['position'] != 'Goalkeeper':
            raise ValueError(f'{path.name}: the first {team} player is not the goalkeeper')
        names = [_name(p['name']) for p in cols[team]]
        out['names'][team] = names
        out['jerseys'][team] = [p['jersey'] for p in cols[team]]
        letter = LETTER[team]
        out['ids'][team] = [('GK-'+letter) if j == 0 else f'{letter}-{j:02d}' for j in range(len(names))]
        for j, n in enumerate(names):
            out['team_of'][n] = team
            out['column'][n] = (team, j)
    out['frame'], out['time'], out['period'], out['ball'] = frame, times, period, ball
    meta['playerIds'] = {_name(p['name']): p['id'] for p in meta['players']}
    out['metadata'] = meta
    return out


def load_events_json(game, events, metadata):
    """load_events' list for Metrica's JSON events (Sample_Game_<n>_events.json, game 3). The differences from the
    CSV games and how they are handled:
    - The team is "Team A" / "Team B" with an id: Home is the metadata's local team (read_metadata of `metadata`).
    - Subtypes are absent, one {name} or a list of them: joined with '-' in the file's order, which gives the CSV
      games' strings (each of the 21 multi-part subtypes game 3 shares with game 1 comes out spelt as game 1 spells
      it: AERIAL-FAULT-LOST, HEAD-INTERCEPTION, ON TARGET-SAVED ...).
    - A player is {name, id}: the name loses its space as in the CSV ("Player 10" -> "Player10"); the id must be
      the metadata's player of that name and team, else this refuses the file.
    - Frames are given with their time; only the frames are read (the times equal frame / 25 on all 3620 events of
      game 3). The rows must be in file order (index 1..N): `i` is the row, 0-based, as in the CSV.
    The rules that turn events into actions are the CSV games' and are not changed. What that means for game 3's
    codings that games 1-2 do not use:
    - CARRY (a player running with the ball; 1395 of the 3620 events) has no rule, so the truth is derived from the
      vocabulary games 1-2 were derived from. The carries serve as an independent check of it instead (reading
      them as controls would change no frame and no change of the truth).
    - PASS ...OFFSIDE (2 events; the CSV games code an offside as BALL LOST OFFSIDE) stays a pass. Play was coded
      after both (a shot), so an offside stoppage there would be dropped by pass 1 anyway (the truth is identical
      with one), and the untracked run before the free kick gives the spell.
    - BALL LOST GOAL KICK-END HALF (the half ended with the goal kick in the air) restarts play, and the period
      ends with its final untracked run: only an exact END HALF subtype stops play at its start frame, in the CSV
      games as well. Stopping play there would make 53 more frames dead and remove one restart change."""
    meta = read_metadata(game, metadata)
    path = Path(events)/f'Sample_Game_{game}_events.json'
    rows = json.loads(path.read_text(encoding='utf-8'))['data']
    side = {meta['teams'][t]['id']: t for t in TEAMS}
    known = {p['id']: p for p in meta['players']}
    out = []
    for i, r in enumerate(rows):
        if r.get('index') != i+1:
            raise ValueError(f'{path.name}: row {i} has index {r.get("index")}: not in file order')
        team = side[r['team']['id']]
        sub = r.get('subtypes')
        sub = [sub] if isinstance(sub, dict) else (sub or [])
        who = {}
        for key in ('from', 'to'):
            p = r.get(key)
            if not p:
                who[key] = None
                continue
            m = known.get(p.get('id'))
            if m is None or m['name'] != p.get('name') or (key == 'from' and m['team'] != team):
                raise ValueError(f'{path.name}: row {i}: {key} {p} is not that team\'s metadata player')
            who[key] = _name(p['name']) or None
        st, en = r.get('start') or {}, r.get('end') or {}
        s = int(st['frame'])
        e = int(en.get('frame') or 0)
        out.append({'i': i, 'team': team, 'type': r['type']['name'], 'subtype': '-'.join(x['name'] for x in sub), 'period': int(r['period']),
                    'start': s, 'end': e if e >= s else s, 'From': who['from'], 'To': who['to'],
                    'sx': _num(st.get('x'))*LENGTH, 'sy': _num(st.get('y'))*WIDTH,
                    'ex': _num(en.get('x'))*LENGTH, 'ey': _num(en.get('y'))*WIDTH})
    return out


# ----------------------------------------------------------------------------------------------- actions

def is_foul(e):
    """A foul that stops play: CHALLENGE ...FAULT... without ADVANTAGE, and FAULT RECEIVED."""
    if e['type'] == 'FAULT RECEIVED':
        return True
    return e['type'] == 'CHALLENGE' and 'FAULT' in e['subtype'] and 'ADVANTAGE' not in e['subtype']


def _label(e):
    return f"{e['type']}/{e['subtype']}" if e['subtype'] else e['type']


def actions(events):
    """The events as actions, sorted by frame, priority and file order. Kinds:
    dead     play stops on this frame (BALL OUT and SHOT ...OUT / ...GOAL at their End Frame, a foul, an offside,
             END HALF at their Start Frame);
    restart  SET PIECE, and an event whose subtype names a GOAL KICK;
    gain     RECOVERY;
    control  the From player plays a ball he holds: PASS, BALL LOST, SHOT at their Start Frame;
    receive  PASS End Frame, the To player;
    lose     the losing action (BALL LOST, SHOT, BALL OUT start, CHALLENGE ...LOST), for lostAt;
    touch    BALL OUT start (a touch, not a control: a deflection out by the defending team gains nothing)."""
    acts = []
    for e in events:
        typ, sub, team, s, end = e['type'], e['subtype'], e['team'], e['start'], e['end']

        def add(f, p, kind, player, reason=None):
            acts.append((f, p, e['i'], {'frame': f, 'prio': p, 'kind': kind, 'team': team, 'player': player, 'event': e, 'reason': reason}))
        if typ == 'SET PIECE':
            add(s, P_RESTART, 'restart', e['From'], sub)
        if 'GOAL KICK' in sub and typ in ('PASS', 'BALL LOST', 'BALL OUT'):
            add(s, P_RESTART, 'restart', e['From'], 'GOAL KICK')
        if typ == 'PASS':
            add(s, P_CONTROL, 'control', e['From'])
            add(end, P_RECEIVE, 'receive', e['To'])
        elif typ == 'BALL LOST':
            add(s, P_CONTROL, 'control', e['From'])
            add(s, P_TOUCH, 'lose', e['From'])
            if sub == 'OFFSIDE':
                add(s, P_DEAD, 'dead', e['From'], 'offside')
            elif sub == 'END HALF':
                add(s, P_DEAD, 'dead', e['From'], 'end of period')
        elif typ == 'SHOT':
            add(s, P_CONTROL, 'control', e['From'])
            add(s, P_TOUCH, 'lose', e['From'])
            if sub.endswith('OUT') or sub.endswith('GOAL'):
                add(end, P_DEAD, 'dead', e['From'], 'goal' if sub.endswith('GOAL') else 'ball out')
        elif typ == 'RECOVERY':
            add(s, P_GAIN, 'gain', e['From'])
        elif typ == 'BALL OUT':
            add(s, P_TOUCH, 'touch', e['From'])
            add(s, P_TOUCH, 'lose', e['From'])
            add(end, P_DEAD, 'dead', e['From'], 'goal' if sub.endswith('GOAL') else 'ball out')
        elif typ == 'CHALLENGE':
            if is_foul(e):
                add(s, P_DEAD, 'dead', e['From'], 'foul')
            elif sub.endswith('LOST'):
                add(s, P_TOUCH, 'lose', e['From'], 'duel')
        elif typ == 'FAULT RECEIVED':
            add(s, P_DEAD, 'dead', e['From'], 'foul')
    acts.sort(key=lambda a: (a[0], a[1], a[2]))
    return [a[3] for a in acts]


# ----------------------------------------------------------------------------------------------- pass 1: dead spells

def _untracked_runs(ball):
    """[(first, last)] Metrica frames of the runs where the ball is untracked."""
    nan = ~np.isfinite(ball[:, 0])
    if not nan.any():
        return []
    d = np.diff(np.concatenate([[0], nan.astype(np.int8), [0]]))
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1)-1
    return [(int(a)+1, int(b)+1) for a, b in zip(starts, ends)]


def dead_spells(trk, acts):
    """Pass 1: the dead-ball spells [{start, end (inclusive frames), reason, startEvent, restart (frame), restartTeam,
    restartPlayer, restartEvent, restartType, source}] and a log. A stoppage runs from its dead action to the next
    restart action. When play is coded in between (a PASS or a SHOT: the stoppage is not where the events say, or the
    restart was never coded) the untracked-ball run that confirms the stoppage ends it instead, and the restart is the
    first gain or control from that run's end (source 'tracking'); with no such run the stoppage is dropped (play went
    on). The same happens when a second stoppage (a dead action more than CLUSTER frames later: not the same foul's
    FAULT RECEIVED) comes before the next restart: the restart of the first was never coded (game 1 frame 36212, a
    goal kick coded as the keeper's RECOVERY + BALL LOST, then a throw-in); without an untracked run the first gain
    or control after the stoppage restarts it (source 'events'). A restart that no stoppage precedes takes the
    untracked run ending just before it as its spell."""
    T = len(trk['frame'])
    period = trk['period']
    first_of = {int(p): int(trk['frame'][period == p][0]) for p in np.unique(period)}
    last_of = {int(p): int(trk['frame'][period == p][-1]) for p in np.unique(period)}
    runs = [r for r in _untracked_runs(trk['ball']) if r[1]-r[0]+1 >= MIN_STOP]
    restarts = [a for a in acts if a['kind'] == 'restart']
    deads = [a for a in acts if a['kind'] == 'dead']
    plays = [a for a in acts if a['kind'] in ('control', 'receive') and a['event']['type'] in ('PASS', 'SHOT')]
    controls = [a for a in acts if a['kind'] in ('gain', 'control')]
    log = {'droppedStoppages': [], 'trackingRestarts': [], 'trackingStoppages': [], 'unusedRestarts': [], 'uncodedRestarts': []}
    spells = []
    used_restart = set()

    def run_near(f):
        for a, b in runs:
            if f-STOP_BEFORE <= a <= f+STOP_AFTER:
                return a, b
        return None

    def per(f):
        return int(period[min(max(f, 1), T)-1])

    # Before each period's kick-off, and from END HALF (or the last untracked run) to the period's end.
    kickoffs = {}
    for a in restarts:
        if a['reason'] == 'KICK OFF' and per(a['frame']) not in kickoffs:
            kickoffs[per(a['frame'])] = a
    for p, a in sorted(kickoffs.items()):
        used_restart.add(id(a))
        if a['frame'] > first_of[p]:
            s = {'start': first_of[p], 'end': a['frame']-1, 'reason': 'before kick-off', 'startEvent': None, 'source': 'events'}
            _restart_fields(s, a)
            spells.append(s)
    deads.sort(key=lambda a: (a['frame'], a['prio']))
    covered_until = 0
    for d in deads:
        f = d['frame']
        if f <= covered_until:
            continue                                   # a second foul action, a card time: already dead
        p = per(f)
        if d['reason'] == 'end of period':
            spells.append({'start': f, 'end': last_of[p], 'reason': 'end of period', 'startEvent': d['event']['i'], 'source': 'events'})
            covered_until = last_of[p]
            continue
        nxt = next((r for r in restarts if r['frame'] > f and id(r) not in used_restart), None)
        if nxt is not None and per(nxt['frame']) != p:
            nxt = None
        limit = nxt['frame'] if nxt else last_of[p]+1
        between = [a for a in plays if f < a['frame'] < limit]
        second = next((x for x in deads if f+CLUSTER < x['frame'] < limit), None)
        if nxt is not None and not between and second is None:
            used_restart.add(id(nxt))
            s = {'start': f, 'end': nxt['frame']-1, 'reason': d['reason'], 'startEvent': d['event']['i'], 'source': 'events'}
            _restart_fields(s, nxt)
            spells.append(s)
            covered_until = s['end']
            continue
        stop = min([limit]+[a['frame'] for a in between[:1]]+([second['frame']] if second else []))
        run = run_near(f)
        if run is not None and run[1] < stop:
            r_frame = run[1]+1
            starter = next((a for a in controls if a['frame'] >= r_frame-2), None)
            if starter is not None and starter['frame'] >= stop and second is not None:
                starter = None
            s = {'start': f, 'end': max(f, (starter['frame'] if starter else r_frame)-1), 'reason': d['reason'], 'startEvent': d['event']['i'],
                 'source': 'tracking', 'untracked': list(run)}
            if starter is not None:
                _restart_fields(s, starter)
                s['restartType'] = 'not coded: '+_label(starter['event'])
            spells.append(s)
            covered_until = s['end']
            log['trackingRestarts'].append({'deadFrame': f, 'reason': d['reason'], 'event': d['event']['i'], 'untracked': list(run),
                                            'restart': s.get('restart'), 'codedRestart': nxt['frame'] if nxt else None,
                                            'playCodedBetween': [a['event']['i'] for a in between[:5]]})
            continue
        if second is not None and not between:
            starter = next((a for a in controls if f+CLUSTER < a['frame'] < second['frame']), None)
            if starter is not None:
                s = {'start': f, 'end': starter['frame']-1, 'reason': d['reason'], 'startEvent': d['event']['i'], 'source': 'events'}
                _restart_fields(s, starter)
                s['restartType'] = 'not coded: '+_label(starter['event'])
                spells.append(s)
                covered_until = s['end']
                log['uncodedRestarts'].append({'deadFrame': f, 'reason': d['reason'], 'event': d['event']['i'], 'restart': starter['frame'],
                                               'secondStoppage': second['frame']})
                continue
        log['droppedStoppages'].append({'frame': f, 'reason': d['reason'], 'event': d['event']['i'], 'type': _label(d['event']),
                                        'codedRestart': nxt['frame'] if nxt else None, 'playCodedBetween': [a['event']['i'] for a in between[:5]]})
    # Restarts that closed no stoppage: a stoppage the events never coded.
    for r in restarts:
        if id(r) in used_restart or any(s['start'] <= r['frame'] <= s['end']+1 for s in spells):
            continue
        run = next((x for x in runs if r['frame']-STOP_BEFORE <= x[1]+1 <= r['frame']+2), None)
        if run is not None:
            s = {'start': run[0], 'end': r['frame']-1, 'reason': 'stoppage not coded', 'startEvent': None, 'source': 'tracking', 'untracked': list(run)}
            _restart_fields(s, r)
            spells.append(s)
            log['trackingStoppages'].append({'restart': r['frame'], 'event': r['event']['i'], 'type': r['reason'], 'untracked': list(run)})
        else:
            log['unusedRestarts'].append({'frame': r['frame'], 'event': r['event']['i'], 'type': r['reason'], 'team': r['team']})
    # A period that ends in play: dead from the final untracked run that reaches its last frame.
    for p, last in last_of.items():
        if any(s['start'] <= last <= s['end'] for s in spells):
            continue
        run = next((x for x in runs if x[1] == last or (x[1] >= last-1 and x[0] <= last)), None)
        if run is not None:
            spells.append({'start': run[0], 'end': last, 'reason': 'end of period', 'startEvent': None, 'source': 'tracking', 'untracked': list(run)})
    spells = [s for s in spells if s['end'] >= s['start']]
    spells.sort(key=lambda s: s['start'])
    # Overlaps (a stoppage inside another, e.g. a goal kick spell inside an end-of-period one): keep the earlier.
    merged = []
    for s in spells:
        if merged and s['start'] <= merged[-1]['end']:
            merged[-1]['end'] = max(merged[-1]['end'], s['end'])
            continue
        merged.append(s)
    used = {s['restartEvent'] for s in merged if s.get('restartEvent') is not None} | {a['event']['i'] for a in kickoffs.values()}
    return merged, log, used


def _restart_fields(spell, a):
    if spell is None or a is None:
        return
    spell.update(restart=a['frame'], restartTeam=a['team'], restartPlayer=a['player'], restartEvent=a['event']['i'],
                 restartType=a['reason'] if a['kind'] == 'restart' else _label(a['event']))


# ----------------------------------------------------------------------------------------------- pass 2: possession

def derive(trk, events):
    """Possession truth: per-frame arrays and the change, dead-spell and possession lists (the module docstring)."""
    T = len(trk['frame'])
    dropped = displaced(trk, events)
    gone = {d['event'] for d in dropped}
    acts = actions([e for e in events if e['i'] not in gone])
    spells, log, used = dead_spells(trk, acts)
    log['displacedEvents'] = dropped
    dead = np.zeros(T+2, bool)
    for s in spells:
        dead[s['start']:s['end']+1] = True              # index = Metrica frame
    ends = {}
    for s in spells:
        if s.get('restart') is not None:
            ends.setdefault(s['restart'], s)
    marks = [{'frame': s['start'], 'prio': -1, 'kind': 'spell start', 'spell': s} for s in spells]
    marks += [{'frame': f, 'prio': P_RESTART-.5, 'kind': 'spell end', 'spell': s} for f, s in ends.items()]
    stream = sorted(marks+acts, key=lambda a: (a['frame'], a['prio']))
    log.update(ignoredInDead=[], implicitGains=[], lateLosses=[], receiveByOther=[])
    owner, dead_owner = None, None
    segments = [(1, None)]
    changes = []
    last_ctrl = {t: None for t in TEAMS}
    last_lose = {t: None for t in TEAMS}
    last_change = None
    for a in stream:
        f, kind = a['frame'], a['kind']
        if kind == 'spell start':
            dead_owner = owner if a['spell']['reason'] not in PERIOD_REASONS else None
            owner = None
            segments.append((f, None))
            a['spell']['ownerBefore'] = dead_owner
            continue
        if kind == 'spell end':
            s = a['spell']
            team = s['restartTeam']
            if dead_owner is not None and team != dead_owner and s['reason'] not in PERIOD_REASONS:
                fake = {'frame': f, 'player': s.get('restartPlayer'), 'event': events[s['restartEvent']], 'kind': 'restart'}
                c = _change(dead_owner, team, f, fake, 'restart', last_ctrl[dead_owner], last_lose[dead_owner])
                c.update(deadSpell=[s['start'], s['end']], deadReason=s['reason'], restartType=s.get('restartType'), restartSource=s['source'],
                         startEvent=s.get('startEvent'))
                changes.append(c)
                last_change = c
            owner, dead_owner = team, None
            segments.append((f, team))
            last_ctrl[team] = {'frame': f, 'player': s.get('restartPlayer'), 'event': events[s['restartEvent']]}
            continue
        team = a['team']
        if dead[f]:
            if kind in ('gain', 'control', 'receive'):
                log['ignoredInDead'].append({'frame': f, 'event': a['event']['i'], 'type': _label(a['event']), 'team': team})
            continue
        if kind == 'dead':
            continue                                     # pass 1 placed it
        if kind == 'restart':
            if a['reason'] == 'KICK OFF' and owner is None:
                owner = team                             # a kick-off on a period's first frame: no spell before it
                segments.append((f, team))
                last_ctrl[team] = a
                continue
            if a['event']['i'] in used:
                continue                                 # pass 1 ended a spell with it
            kind = 'gain'                                # a set piece no stoppage could be placed for: the team has the ball
        if kind == 'gain':
            if owner != team:
                c = _change(owner, team, f, a, 'open', last_ctrl.get(owner), last_lose.get(owner)) if owner else None
                if c:
                    changes.append(c)
                    last_change = c
                owner = team
                segments.append((f, team))
            last_ctrl[team] = a
        elif kind == 'control':
            if owner != team:
                if (a['event']['type'] == 'BALL LOST' and last_change is not None and last_change['from'] == team and last_change['to'] == owner
                        and f-last_change['wonAt'] <= LATE_LOSS):
                    log['lateLosses'].append({'frame': f, 'event': a['event']['i'], 'type': _label(a['event']), 'team': team,
                                              'afterGainAt': last_change['wonAt'], 'frames': f-last_change['wonAt']})
                    continue
                log['implicitGains'].append({'frame': f, 'event': a['event']['i'], 'type': _label(a['event']), 'team': team, 'owner': owner})
                c = _change(owner, team, f, a, 'open', last_ctrl.get(owner), last_lose.get(owner)) if owner else None
                if c:
                    c['implicit'] = True
                    changes.append(c)
                    last_change = c
                owner = team
                segments.append((f, team))
            last_ctrl[team] = a
        elif kind == 'receive':
            if owner != team:
                if last_change is not None and last_change['from'] == team and last_change['to'] == owner and f-last_change['wonAt'] <= LATE_LOSS:
                    # The pass was won on its way and its "reception" coded just after (game 1 139851 / 139852: a
                    # RECOVERY THEFT, the reception a frame later, the receiver's BALL LOST 6 frames later).
                    log['lateLosses'].append({'frame': f, 'event': a['event']['i'], 'type': 'PASS reception', 'team': team,
                                              'afterGainAt': last_change['wonAt'], 'frames': f-last_change['wonAt']})
                    continue
                # A PASS is a ball that reached a team-mate: the passer's team has it on the reception frame even
                # when the other team's touch on the way was coded as a RECOVERY (game 1 frames 15300 / 15336).
                log['receiveByOther'].append({'frame': f, 'event': a['event']['i'], 'team': team, 'owner': owner})
                c = _change(owner, team, f, a, 'open', last_ctrl.get(owner), last_lose.get(owner)) if owner else None
                if c:
                    c['implicit'] = True
                    changes.append(c)
                    last_change = c
                owner = team
                segments.append((f, team))
            last_ctrl[team] = a
        elif kind == 'lose':
            last_lose[team] = a
        elif kind == 'touch':
            if team == owner:
                last_ctrl[team] = a
    poss = np.zeros(T, np.int8)
    segments.sort(key=lambda x: x[0])
    for (f0, team), nxt in zip(segments, segments[1:]+[(T+1, None)]):
        lo, hi = max(1, f0), min(T, nxt[0]-1)
        if hi >= lo:
            poss[lo-1:hi] = DEAD if team is None else CODE[team]
    poss[dead[1:T+1]] = DEAD
    _finish_changes(changes, trk, events, poss)
    phase = np.where(poss == DEAD, PH_DEAD, PH_HELD).astype(np.int8)
    for e in events:
        if e['type'] == 'PASS' and e['end'] > e['start']+1:
            sl = slice(e['start'], e['end']-1)           # frames after the kick to the one before the reception
            seg = phase[sl]
            seg[poss[sl] == CODE[e['team']]] = PH_FLIGHT
    for c in changes:
        if c['kind'] == 'open' and c['wonAt'] > c['lostAt']:
            sl = slice(c['lostAt'], c['wonAt']-1)        # after the losing action up to the frame before the gain
            seg = phase[sl]
            seg[poss[sl] != DEAD] = PH_LOOSE
    return {'possession': poss, 'phase': phase, 'changes': changes, 'deadSpells': spells, 'log': log, 'actions': len(acts)}


def displaced(trk, events):
    """Events (DISPLACED_TYPES) whose From player is further than DISPLACED_M from the tracked ball on every frame
    within DISPLACED_SEARCH of the start; events whose ball or player is untracked there are kept."""
    out = []
    T = len(trk['frame'])
    for e in events:
        if e['type'] not in DISPLACED_TYPES or not e['From'] or e['From'] not in trk['column']:
            continue
        team, j = trk['column'][e['From']]
        lo, hi = max(1, e['start']-DISPLACED_SEARCH), min(T, e['start']+DISPLACED_SEARCH)
        d = np.hypot(*(trk['ball'][lo-1:hi]-trk[team][lo-1:hi, j]).T)
        d = d[np.isfinite(d)]
        if len(d) and d.min() > DISPLACED_M:
            out.append({'event': e['i'], 'type': _label(e), 'team': e['team'], 'player': e['From'], 'frame': e['start'],
                        'closestMetres': round(float(d.min()), 1)})
    return out


def _change(prev, team, f, act, kind, ctrl, lose):
    return {'kind': kind, 'from': prev, 'to': team, 'wonAt': f, 'wonBy': act.get('player'), 'wonType': _label(act['event']),
            'wonEvent': act['event']['i'], '_ctrl': ctrl, '_lose': lose}


def _finish_changes(changes, trk, events, poss):
    """lostAt / lostBy of every change, in this order of preference: the losing team's BALL LOST, SHOT or BALL OUT at
    or after its last control; a BALL LOST of that team coded up to LATE_LOSS frames after the gain (clamped to the
    gain frame, flagged lostCodedAfterGain); a duel it lost (CHALLENGE ...LOST) after its last control (lostType
    CHALLENGE); for a free kick the losing team's foul; else its last control (lossImplicit). Times, period, the ball
    at the gain, ids, the oneTouch flags and the seconds the new team then held the ball are added."""
    T = len(trk['frame'])
    by_frame = {}
    for e in events:
        by_frame.setdefault(e['start'], []).append(e)
    for c in changes:
        ctrl, lose = c.pop('_ctrl'), c.pop('_lose')
        won = c['wonAt']
        ctrl_f = ctrl['frame'] if ctrl else -1
        cand = None
        if lose is not None and lose['frame'] >= ctrl_f and lose['event']['type'] in ('BALL LOST', 'SHOT', 'BALL OUT'):
            cand = (lose['frame'], lose['player'], _label(lose['event']), lose['event']['i'])
        if cand is None and c['kind'] == 'open':
            late = next((e for f in range(won, won+LATE_LOSS+1) for e in by_frame.get(f, [])
                         if e['team'] == c['from'] and e['type'] == 'BALL LOST'), None)
            if late is not None:
                cand = (min(late['start'], won), late['From'], _label(late), late['i'])
                c['lostCodedAfterGain'] = late['start'] > won
        if cand is None and lose is not None and lose['frame'] >= ctrl_f and lose['event']['type'] == 'CHALLENGE':
            cand = (lose['frame'], lose['player'], _label(lose['event']), lose['event']['i'])
        if c['kind'] == 'restart' and c.get('deadReason') == 'offside':
            e = events[c['startEvent']] if c.get('startEvent') is not None else None
            if e is not None and e['team'] == c['from']:
                cand = (e['start'], e['From'], _label(e), e['i'])
        if c['kind'] == 'restart' and c.get('deadReason') == 'foul':
            foul = next((e for e in events if c['deadSpell'][0] <= e['start'] <= c['deadSpell'][0]+LATE_LOSS and is_foul(e)
                         and e['team'] == c['from']), None)
            if foul is not None:
                cand = (foul['start'], foul['From'], _label(foul), foul['i'])
        if cand is None and ctrl is not None:
            cand = (ctrl['frame'], ctrl.get('player'), 'last control ('+_label(ctrl['event'])+')', ctrl['event']['i'])
            c['lossImplicit'] = True
        if cand is None:
            cand = (won, None, 'unknown', None)
            c['lossImplicit'] = True
        c['lostAt'], c['lostBy'], c['lostType'], c['lostEvent'] = int(min(cand[0], won)), cand[1], cand[2], cand[3]
        c['lostTime'] = round(float(trk['time'][min(c['lostAt'], T)-1]), 2)
        c['wonTime'] = round(float(trk['time'][min(won, T)-1]), 2)
        c['period'] = int(trk['period'][min(won, T)-1])
        b = trk['ball'][min(won, T)-1]
        c['ballAtWon'] = [round(float(b[0]), 2), round(float(b[1]), 2)] if np.isfinite(b).all() else None
        for side, key in (('lostBy', 'lostById'), ('wonBy', 'wonById')):
            col = trk['column'].get(c[side]) if c[side] else None
            c[key] = trk['ids'][col[0]][col[1]] if col else None
        c['oneTouch'] = False
    # A one-touch possession: the gaining player loses the ball on the very frame he gained it (a first-time
    # interception or header that went on to the other team, or out of play).
    for c, nxt in zip(changes, changes[1:]):
        if nxt['from'] == c['to'] and nxt['lostAt'] == c['wonAt'] and nxt['lostBy'] == c['wonBy']:
            c['oneTouch'] = True
            nxt['afterOneTouch'] = True
    for c, nxt in zip(changes, changes[1:]+[None]):
        end = nxt['wonAt'] if nxt is not None else len(poss)+1
        seg = poss[c['wonAt']-1:end-1]
        c['heldSeconds'] = round(float((seg == CODE[c['to']]).sum()/FPS), 2)


# ----------------------------------------------------------------------------------------------- run lengths, cache

def run_lengths(codes):
    """[[first frame, last frame, code], ...] in Metrica frames (1-based, inclusive) for a per-frame int array."""
    codes = np.asarray(codes)
    if not len(codes):
        return []
    cut = np.flatnonzero(np.diff(codes))+1
    starts = np.concatenate([[0], cut])
    ends = np.concatenate([cut, [len(codes)]])
    return [[int(s)+1, int(e), int(codes[s])] for s, e in zip(starts, ends)]


def from_run_lengths(runs, T=None):
    T = T or (runs[-1][1] if runs else 0)
    out = np.zeros(T, np.int8)
    for a, b, v in runs:
        out[a-1:b] = v
    return out


def possessions(poss):
    """Possession spells: [{'team', 'start', 'end' (frames), 'seconds'}] for every in-play run of one team."""
    return [{'team': NAME[v], 'start': a, 'end': b, 'seconds': round((b-a+1)/FPS, 2)} for a, b, v in run_lengths(poss) if v != DEAD]


def build(game, tracking_dir, events_dir):
    """Derive the truth of one game: (tracking, events, truth, the JSON payload with run lengths, lists, the log)."""
    trk = load_tracking(game, tracking_dir)
    ev = load_events(game, events_dir, tracking_dir)
    tr = derive(trk, ev)
    payload = {
        'game': int(game), 'fps': FPS, 'frames': int(len(trk['frame'])), 'pitch': [LENGTH, WIDTH],
        'note': 'Metrica frame numbers (1-based, inclusive). possession codes 0 dead / 1 Home / 2 Away; phase 0 dead / 1 held / '
                '2 own pass in flight / 3 loose (after the loser\'s last action, before the winner\'s control). See tools/possession_truth.py.',
        'codes': {'possession': {str(k): v for k, v in NAME.items()}, 'phase': {str(k): v for k, v in PHASES.items()}},
        'periods': {str(int(p)): [int(trk['frame'][trk['period'] == p][0]), int(trk['frame'][trk['period'] == p][-1])] for p in np.unique(trk['period'])},
        'possession': run_lengths(tr['possession']), 'phase': run_lengths(tr['phase']),
        'changes': tr['changes'], 'deadSpells': tr['deadSpells'], 'possessions': possessions(tr['possession']),
        'ids': {t: dict(zip(trk['names'][t], trk['ids'][t])) for t in TEAMS},
        'log': tr['log'],
    }
    return trk, ev, tr, payload


def save(payload, path):
    text = json.dumps(payload, indent=1, default=_json_default, ensure_ascii=True)
    Path(path).write_bytes(text.replace('\n', '\r\n').encode('ascii'))


def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def load_game(game, tracking_dir, events_dir):
    """Tracking and truth of one game as numpy arrays at 25 Hz in pitch metres: frame, time, period, ball (T, 2),
    Home (T, P, 2), Away (T, Q, 2), names / ids / jerseys / team_of / column, in_play (T,) bool (truth: not dead),
    ball_tracked (T,) bool, possession (T,) int8 (0 dead, 1 Home, 2 Away), phase (T,) int8 (0 dead, 1 held, 2 own
    pass in flight, 3 loose), changes, deadSpells, possessions, events (lists), derived from the events on the fly."""
    trk = load_tracking(game, tracking_dir)
    events = load_events(game, events_dir, tracking_dir)
    tr = derive(trk, events)
    changes = json.loads(json.dumps(tr['changes'], default=_json_default))
    spells = json.loads(json.dumps(tr['deadSpells'], default=_json_default))
    g = dict(trk)
    g.update(possession=tr['possession'], phase=tr['phase'], in_play=tr['possession'] != DEAD, ball_tracked=np.isfinite(trk['ball'][:, 0]),
             changes=changes, deadSpells=spells, possessions=possessions(tr['possession']), events=events)
    return g


def truth_at(g, times):
    """(possession code, 25 Hz index) at arbitrary times on the tracking clock (s): the nearest frame. A 30 fps
    result maps through this."""
    t = np.asarray(times, float)
    k = np.clip(np.searchsorted(g['time'], t), 1, len(g['time'])-1)
    k = np.where(np.abs(g['time'][k-1]-t) <= np.abs(g['time'][k]-t), k-1, k)
    return g['possession'][k], k


def score(g, times, team, changes=None, kinds=('open', 'restart'), one_touch=True, tolerances=(1.0, 2.0)):
    """PREDECLARED.md's metrics of an engine against this truth (a reference implementation).
    times (N,): the engine's frame times on the tracking clock (s); team (N,): its label per frame, 0 dead ball, 1 Home
    (team A), 2 Away (team B), -1 no answer. changes: the engine's possession changes [{'wonAt': s, 'to': 'Home'|'Away'
    (or 'A'|'B'), optional 'lostAt': s, 'wonBy' / 'lostBy': ids}]; truth changes of the given kinds count ('open' in
    play, 'restart' across a stoppage; one_touch=False leaves out a one-touch possession's change in and change out).
    1 accuracy: truth in-play frames with an engine team where the team is right; 2 coverage: truth in-play frames with
    an engine team; 4 dead: truth-dead frames called dead (recall) and engine-dead frames truly dead (precision);
    3 changes: one-to-one matching of wonAt within each tolerance (same new team), precision / recall / F1, the median
    |wonAt error| and |lostAt error| of the matches at the widest tolerance, and player attribution where both name one."""
    t = np.asarray(times, float)
    e = np.asarray(team, int)
    truth, _ = truth_at(g, t)
    play = truth != DEAD
    named = play & (e > 0)
    out = {'frames': int(len(t)), 'inPlayFrames': int(play.sum()),
           'accuracy': round(float((e[named] == truth[named]).mean()), 4) if named.any() else None,
           'coverage': round(float(named.sum()/max(1, play.sum())), 4),
           'deadRecall': round(float((e[~play] == DEAD).mean()), 4) if (~play).any() else None,
           'deadPrecision': round(float((truth[e == DEAD] == DEAD).mean()), 4) if (e == DEAD).any() else None}
    if changes is None:
        return out
    lo, hi = t.min(), t.max()
    norm = {'A': 'Home', 'B': 'Away', 'Home': 'Home', 'Away': 'Away'}
    tc = [c for c in g['changes'] if c['kind'] in kinds and lo <= c['wonTime'] <= hi
          and (one_touch or not (c.get('oneTouch') or c.get('afterOneTouch')))]
    ec = [dict(c, to=norm.get(c.get('to'), c.get('to'))) for c in changes if lo <= c['wonAt'] <= hi]
    ids = {}
    for team_name in TEAMS:
        for name, pid in zip(g['names'][team_name], g['ids'][team_name]):
            ids[name] = pid
    out['changes'] = {'truth': len(tc), 'engine': len(ec)}
    for tol in tolerances:
        pairs = []
        cand = sorted(((abs(x['wonAt']-y['wonTime']), i, j) for i, x in enumerate(ec) for j, y in enumerate(tc)
                       if x['to'] == y['to'] and abs(x['wonAt']-y['wonTime']) <= tol))
        used_e, used_t = set(), set()
        for d, i, j in cand:
            if i in used_e or j in used_t:
                continue
            used_e.add(i)
            used_t.add(j)
            pairs.append((i, j, d))
        p = len(pairs)/max(1, len(ec))
        r = len(pairs)/max(1, len(tc))
        res = {'matched': len(pairs), 'precision': round(p, 4), 'recall': round(r, 4), 'f1': round(2*p*r/max(1e-9, p+r), 4)}
        if tol == max(tolerances) and pairs:
            res['medianWonError'] = round(float(np.median([d for _, _, d in pairs])), 3)
            lost = [abs(ec[i]['lostAt']-tc[j]['lostTime']) for i, j, _ in pairs if ec[i].get('lostAt') is not None]
            res['medianLostError'] = round(float(np.median(lost)), 3) if lost else None
            for key, tkey in (('wonBy', 'wonById'), ('lostBy', 'lostById')):
                both = [(ec[i].get(key), tc[j].get(tkey)) for i, j, _ in pairs if ec[i].get(key) and tc[j].get(tkey)]
                res[key+'Accuracy'] = round(sum(a == b for a, b in both)/len(both), 4) if both else None
                res[key+'Compared'] = len(both)
        out['changes'][f'within{tol:g}s'] = res
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--tracking', required=True, help='folder of Sample_Game_<n>_RawTrackingData_{Home,Away}_Team.csv (games 1-2), '
                    'or of Sample_Game_<n>_metadata.xml + Sample_Game_<n>_tracking.txt (game 3, EPTS)')
    ap.add_argument('--events', required=True, help='folder of Sample_Game_<n>_RawEventsData.csv (games 1-2) or Sample_Game_<n>_events.json '
                    '(game 3: usually the --tracking folder)')
    ap.add_argument('--game', type=int, action='append', default=[])
    ap.add_argument('--out', help='write game<n>.json here')
    a = ap.parse_args(argv)
    for game in a.game or [1, 2]:
        trk, ev, tr, payload = build(game, a.tracking, a.events)
        if a.out:
            save(payload, Path(a.out)/f'game{game}.json')
        kinds = [c['kind'] for c in tr['changes']]
        print(json.dumps({'game': game, 'frames': payload['frames'], 'changes': len(kinds), 'open': kinds.count('open'),
                          'restart': kinds.count('restart'), 'oneTouch': sum(1 for c in tr['changes'] if c.get('oneTouch')),
                          'deadSpells': len(tr['deadSpells']), 'inPlayShare': round(float((tr['possession'] != DEAD).mean()), 4)}))


if __name__ == '__main__':
    main()
