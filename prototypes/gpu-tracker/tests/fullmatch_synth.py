"""Synthetic segment artifacts for the Full Match Mode stitching and assembly tests: what independent runs of neighbouring
segments write (result.json, identities.json, registration.json, segment.json), with the truth known.

A Person is one real person: team (the match's letter), role, an appearance (a seeded OSNet-like unit vector and a kit
colour), and where they are on each frame (box and pitch position) or None when off camera. make_segment() renders one
segment run: its own local ids (counted from 1 per team, in the order people appear: never the match's), its own team
letters (swap=True names the kits the other way round, as a run whose first kit cluster was the other team), people
without an identity for their first `late` frames (a run's first second), tracks with their dominant identity, a kit
model, goal ends, a ball, a calibration per frame, and the identities.json galleries.
"""
import base64
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from fullmatch import manifest as mf

FPS = 25.0
KITS = {'A': 3, 'B': 17, 'GK': 9}   # the bin of the 24-bin jersey histogram each kit fills


@dataclass
class Person:
    name: str
    team: str            # 'A' or 'B' in the match's letters (a goalkeeper's team too)
    role: str = 'PLAYER'
    seed: int = 0
    path: object = None  # frame -> (box (x, y, w, h), pitch (x, y)) or None (off camera)
    kit: int = None
    referee: bool = False
    embedding: np.ndarray = field(default=None, repr=False)

    def __post_init__(self):
        if self.embedding is None:
            v = np.random.default_rng(self.seed).normal(size=512)
            self.embedding = v/np.linalg.norm(v)
        if self.kit is None:
            self.kit = KITS['GK'] if self.role == 'GOALKEEPER' else KITS[self.team]


def walker(x0, y0, vx=0.0, vy=0.0, visible=None):
    """A person walking at constant speed (normalized pitch units per frame); box from the pitch position."""
    def at(frame):
        if visible is not None and not visible(frame):
            return None
        x, y = x0+vx*frame, y0+vy*frame
        return (round(.05+.9*x, 5), round(.2+.6*y, 5), .012, .05), (round(x, 5), round(y, 5))
    return at


def _hist(bin_, n=24):
    h = np.full(n, .01)
    h[bin_] = 1.0
    return (h/h.sum()).round(4).tolist()


def _descriptor(person, rng):
    e = person.embedding+rng.normal(scale=.012, size=512)   # same-person cosine ~.93 (real OSNet: ~.7-.9)
    e = (e/np.linalg.norm(e)).astype(np.float16)
    return {'q': .9, 'j': _hist(person.kit), 's': _hist(person.kit % 12, 12), 'k': _hist(person.kit % 8, 8),
            'l': [.5]*24, 'e': base64.b64encode(e.tobytes()).decode('ascii')}


def make_segment(segment, people, swap=False, late=0, goal_ends=None, ball=None, H=None):
    """(result, snapshot) of one segment run. ball: frame -> (x, y) normalized image centre or None."""
    local = {'A': 'B' if swap else 'A', 'B': 'A' if swap else 'B'}
    ids, tracks, first_seen = {}, {}, {}
    counters = {'A': 0, 'B': 0, 'GK': 0}
    frames = []
    H = H or [1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0]
    for f in range(segment.proc0, segment.proc1):
        rows = []
        for p in people:
            where = p.path(f) if p.path else None
            if where is None:
                continue
            box, pitch = where
            if p.name not in tracks:
                tracks[p.name] = len(tracks)+1
                first_seen[p.name] = f
                if not p.referee:
                    if p.role == 'GOALKEEPER':
                        counters['GK'] += 1
                        ids[p.name] = f"GK-{counters['GK']}"
                    else:
                        t = local[p.team]
                        counters[t] += 1
                        ids[p.name] = f'{t}-{counters[t]:02d}'
            known = not p.referee and f-first_seen[p.name] >= late
            row = {'time': round(f/FPS-.0119, 6), 'track': tracks[p.name], 'box': list(box), 'score': .9, 'cls': 2,
                   'id': ids.get(p.name) if known else None,
                   'team': None if p.referee else local[p.team], 'role': 'NON_TARGET' if p.referee else p.role,
                   'state': 'nontarget' if p.referee else 'confirmed' if known else 'candidate',
                   'label': 'NON_TARGET' if p.referee else f'PLAYER_TEAM_{local[p.team]}',
                   'display': (ids.get(p.name) if known else 'CAND') if not p.referee else 'IGNORE',
                   'pitch': list(pitch), 'pitchConfidence': .9}
            rows.append(row)
        b = ball(f) if ball else None
        frames.append({'time': round(f/FPS-.0119, 6), 'frame': f, 'people': rows,
                       'ball': {'state': 'TRACKED', 'center': list(b), 'origin': 'OBSERVATION'} if b else {'state': 'UNKNOWN'},
                       'calibrated': True, 'calibration': {'H': H, 'confidence': .8},
                       'visibility': {ids[p.name]: 'C' for p in people if p.name in ids and p.path and p.path(f)}})
    groups = {'players': [], 'goalkeepers': [], 'referees': [], 'retired': [], 'excluded': [], 'withdrawn': []}
    for p in people:
        if p.name not in ids:
            continue
        group = 'goalkeepers' if p.role == 'GOALKEEPER' else 'players'
        groups[group].append({'id': ids[p.name], 'display': ids[p.name], 'team': local[p.team], 'role': p.role,
                              'label': f'PLAYER_TEAM_{local[p.team]}', 'status': 'ACTIVE', 'observations': 1})
    timeline = {'frameTimes': [f['time'] for f in frames],
                'ids': {ids[p.name]: {'team': local[p.team], 'x': [None]*len(frames), 'y': [None]*len(frames)} for p in people if p.name in ids}}
    result = {'version': 3, 'status': 'complete', 'video': {'fps': FPS, 'width': 1920, 'height': 1080}, 'frames': frames, 'match': groups,
              'tracks': {str(tracks[p.name]): {'ids': {ids[p.name]: 10} if p.name in ids else {}} for p in people if p.name in tracks},
              'teams': {'learned': True, 'a': _hist(KITS['B' if swap else 'A']), 'b': _hist(KITS['A' if swap else 'B']), 'samples': 100,
                        'shortsA': [0]*12, 'shortsB': [1]*12, 'spreadA': .1, 'spreadB': .2, 'keepers': []},
              'tactical': {'goalEnds': {k: local[v] for k, v in (goal_ends or {}).items()}, 'pitchSize': {'length': 105.0, 'width': 68.0}},
              'roster': {'slots': [{'id': ids[p.name], 'display': ids[p.name], 'team': local[p.team]} for p in people if p.name in ids],
                         'expected': {'A': 11, 'B': 11, 'goalkeepers': 2}, 'counts': {'A': 1, 'B': 2, 'goalkeepers': 0}},
              'timeline': timeline, 'stabilized': {'version': 1, 'frameTimes': timeline['frameTimes'], 'ids': json.loads(json.dumps(timeline['ids']))},
              'labels': {ids[p.name]: f'Team{local[p.team]}_Player_1' for p in people if p.name in ids},
              'events': [{'time': 0, 'kind': 'reid', 'playerId': ids[p.name], 'track': tracks[p.name]} for p in people if p.name in ids][:2],
              'summary': {'goalSides': {'left': {'A': 3}}, 'goalkeeperSlots': {'A': None, 'B': None},
                          'globalIdentities': {'teamA': 1, 'teamB': 2}, 'profile': {}, 'ball': {}, 'pitch': {}}}
    rng = np.random.default_rng(segment.index+100)
    snapshot = {'registry': {'identities': [{'id': ids[p.name], 'gallery': [{'t': 0.0, 'd': _descriptor(p, rng)} for _ in range(4)]}
                                            for p in people if p.name in ids]}}
    return result, snapshot, {p.name: ids.get(p.name) for p in people}


def write_artifact(folder, result, snapshot, compressed=False):
    """A segment artifact: compressed as a worker keeps it now (result.json.gz), or plain as workers wrote it before."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    (folder/'result.json').write_text(json.dumps(result), encoding='utf-8')
    (folder/'identities.json').write_text(json.dumps(snapshot), encoding='utf-8')
    (folder/'registration.json').write_text('{}', encoding='utf-8')
    stored = mf.compress_result(folder)[0] if compressed else folder/'result.json'
    mf.write_json(folder/mf.SIDECAR, mf.sidecar(result, stored))
