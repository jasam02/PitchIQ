"""Match-wide identities: the people of independently tracked segments joined into one match roster.

Each segment run names its own teams (A/B is arbitrary per run: the first kit cluster found) and its own identities
(A-07 in one segment is nobody in particular in the next). The resolver, over the finished segment artifacts:

1. Digests every segment once (result.json and identities.json, then dropped): per identity its team, role, owned
   observations (frames, first/last, mean pitch position, a position every second), its appearance gallery; the people
   of the frames it shares with each neighbour (the overlap: 2 x PAD_S around every boundary).
2. Matches the people of every boundary. Both runs processed the same source frames with the same detector, so the same
   person has nearly the same box in both: per shared frame, a one-to-one assignment by box overlap (IoU >= MATCH_IOU),
   votes per pair of identities over the shared frames, and a pair is accepted when it is the mutual best, holds for
   most of the frames both were seen (>= SHARE of the fewer) and at least MIN_VOTES frames. People without an identity
   on a shared frame (a run's first second, before its identity was confirmed) count through their local track's
   identity in that run.
3. Maps every segment's team letters to the match's: from the accepted pairs' own letters where the overlap decides
   (>= 3 pairs, >= 80% agreeing), else from the kit colours (Bhattacharyya coefficient of the jersey histograms against
   the match's team prototypes, updated segment by segment). Across a halftime break only the kit colours tell. A pair
   whose teams disagree after mapping is refused (different teams are never merged).
4. Joins accepted pairs into chains (one identity per segment at most: a chain is a path along the boundaries).
5. Links chains across a gap without overlap evidence (a player off camera at a boundary): within a period, a chain
   that ends before the period does and a chain of the same team that starts later, never seen together, scored on
   appearance (the galleries: soccer.appearance.batch_similarity), the distance between where one was last and the other
   first seen against the time between (12 m/s at most), and where each played on average. Accepted only with
   LINK_SCORE and LINK_MARGIN over the runner-up both ways; everything else stays a separate match identity: an
   unlinked fragment is better than two players made one.
6. Numbers the match roster: per team in order of first appearance, players A-01, A-02, ... (B-..), goalkeepers GK-1,
   GK-2 (by team). Chains observed less than DISPLAY_MIN_S show A-? (as a run shows identities beyond its slots).
   Halftime: the attacking direction per period from the segments' goal ends (mapped); coordinates are never flipped.
   Substitutions are never asserted: a chain with real participation that ends well before its period does, and one of
   the same team that starts well after the period began, are noted as possible changes with the times they were last
   and first seen; a person can be confirmed or corrected by the coach.
"""
import base64
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from . import manifest as mf

MATCH_IOU = .5        # box overlap of the same person in a shared frame of two runs (whole-frame detections are identical)
MIN_VOTES = 8         # shared frames an accepted pair must hold for at least (a third of a second at 25 fps)
SHARE = .4            # ... and this share of the frames the less-seen of the two was seen in the overlap
TEAM_PAIRS = 3        # accepted pairs with both teams known that let the overlap decide a segment's team letters
TEAM_AGREE = .8       # ... when this share of them agree
LINK_SCORE = .72      # gap linking: the combined score a link needs
LINK_MARGIN = .12     # ... and its lead over the runner-up on both sides
LINK_MAX_SPEED = 12.0 # m/s: farther than this from where it was last seen, in the time between, is someone else
LINK_MAX_GAP_S = 600.0
DISPLAY_MIN_S = 10.0  # seconds observed (owned frames) for a numbered display label
SUB_MIN_S = 120.0     # participation that makes the end of a chain worth noting as a possible substitution
SUB_EDGE_S = 120.0    # ... when it ends this long before its period does (or starts this long after it began)
PITCH_MIN_CONF = .35
POSITION_STEP = 25    # frames between position history samples


def _bc(p, q):
    p, q = np.asarray(p, float), np.asarray(q, float)
    if p.sum() <= 0 or q.sum() <= 0:
        return 0.0
    return float(np.sum(np.sqrt(p/p.sum()*q/q.sum())))


def _iou(a, b):
    ax0, ay0, aw, ah = a
    bx0, by0, bw, bh = b
    ix = max(0.0, min(ax0+aw, bx0+bw)-max(ax0, bx0))
    iy = max(0.0, min(ay0+ah, by0+bh)-max(ay0, by0))
    inter = ix*iy
    union = aw*ah+bw*bh-inter
    return inter/union if union > 0 else 0.0


def _swap(letter, swap):
    if not swap or letter not in ('A', 'B'):
        return letter
    return 'B' if letter == 'A' else 'A'


def _decode_embedding(text):
    try:
        e = np.frombuffer(base64.b64decode(text), np.float16).astype(np.float32)
        n = float(np.linalg.norm(e))
        return e/n if n > 0 else None
    except (ValueError, TypeError):
        return None


# ---- 1. digests ------------------------------------------------------------------------------------------------------
ACTIVE_GROUPS = ('players', 'goalkeepers')


def digest(segment, result, snapshot, pad):
    """What the resolver needs of one finished segment (the result itself is dropped afterwards)."""
    entries = {}
    for group in ('players', 'goalkeepers', 'retired', 'excluded', 'withdrawn'):
        for e in (result.get('match') or {}).get(group, []):
            entries[e['id']] = {'id': e['id'], 'group': group, 'team': e.get('team'),
                                'role': 'GOALKEEPER' if group == 'goalkeepers' or e.get('role') == 'GOALKEEPER' else 'PLAYER',
                                'display': e.get('display'), 'status': e.get('status')}
    dominant = {}
    for t, v in (result.get('tracks') or {}).items():
        ids = v.get('ids') or {}
        if ids:
            dominant[int(t)] = max(ids.items(), key=lambda kv: kv[1])[0]
    head_end, tail_start = segment.own0+pad, segment.own1-pad
    has_head, has_tail = segment.proc0 < segment.own0, segment.proc1 > segment.own1
    head, tail = {}, {}
    owned = defaultdict(lambda: {'frames': 0, 'first': None, 'last': None, 'sx': 0.0, 'sy': 0.0, 'n': 0, 'history': [],
                                 'firstPitch': None, 'lastPitch': None, 'firstPitchFrame': None, 'lastPitchFrame': None,
                                 'confirmed': 0})
    frame_people = defaultdict(set)   # owned frame -> ids seen (who was seen together)
    for f in result.get('frames') or []:
        index = f['frame']
        rows = []
        for p in f.get('people') or []:
            if p.get('state') in ('nontarget', 'rejected'):
                continue
            pid = p.get('id')
            if pid is None and p.get('track') is not None:
                pid = dominant.get(int(p['track']))
            if pid is None or pid not in entries or entries[pid]['group'] not in ACTIVE_GROUPS:
                continue
            rows.append((pid, tuple(p['box']), p.get('team'), p.get('role')))
            if segment.own0 <= index < segment.own1 and p.get('id') == pid:
                o = owned[pid]
                o['frames'] += 1
                o['confirmed'] += p.get('state') == 'confirmed'
                o['first'] = index if o['first'] is None else o['first']
                o['last'] = index
                frame_people[index].add(pid)
                pitch = p.get('pitch')
                if pitch and (p.get('pitchConfidence') or 0) >= PITCH_MIN_CONF:
                    x, y = float(pitch[0]), float(pitch[1])
                    o['sx'] += x
                    o['sy'] += y
                    o['n'] += 1
                    if o['firstPitch'] is None:
                        o['firstPitch'], o['firstPitchFrame'] = (x, y), index
                    o['lastPitch'], o['lastPitchFrame'] = (x, y), index
                    if not o['history'] or index-o['history'][-1][0] >= POSITION_STEP:
                        o['history'].append((index, round(x, 4), round(y, 4)))
        if has_head and index < head_end:
            head[index] = rows
        if has_tail and index >= tail_start:
            tail[index] = rows
    from soccer import appearance
    galleries = {}
    for e in ((snapshot or {}).get('registry') or {}).get('identities', []):
        decoded = [appearance.decode_descriptor(s.get('d') or {}) for s in e.get('gallery') or []]
        galleries[e['id']] = [g for g in decoded if g is not None]
    together = Counter()
    for ids in frame_people.values():
        ids = sorted(ids)
        for i, a in enumerate(ids):
            for b in ids[i+1:]:
                together[(a, b)] += 1
    teams = result.get('teams') or {}
    return {'index': segment.index, 'period': segment.period, 'own0': segment.own0, 'own1': segment.own1,
            'proc0': segment.proc0, 'proc1': segment.proc1, 'entries': entries, 'owned': dict(owned), 'head': head, 'tail': tail,
            'galleries': galleries, 'together': together,
            'kit': {'learned': bool(teams.get('learned')), 'a': teams.get('a'), 'b': teams.get('b'), 'samples': teams.get('samples', 0)},
            'goalEnds': (result.get('tactical') or {}).get('goalEnds') or {}, 'fps': float((result.get('video') or {}).get('fps') or 25.0),
            'pitchSize': (result.get('tactical') or {}).get('pitchSize') or {'length': 105.0, 'width': 68.0}}


def load_digest(segment, folder, pad):
    folder = Path(folder)
    result = mf.read_result(folder)
    try:
        snapshot = json.loads((folder/'identities.json').read_text(encoding='utf-8'))
    except (OSError, ValueError):
        snapshot = None
    out = digest(segment, result, snapshot, pad)
    del result
    return out


# ---- 2. boundary matching ------------------------------------------------------------------------------------------
def boundary_votes(a, b):
    """Votes per (identity in a, identity in b) over the frames both processed: a one-to-one assignment by box overlap
    on each shared frame."""
    from scipy.optimize import linear_sum_assignment
    shared = sorted(set(a['tail']) & set(b['head']))
    votes, seen_a, seen_b, ious = Counter(), Counter(), Counter(), defaultdict(list)
    for f in shared:
        A, B = a['tail'][f], b['head'][f]
        for row in A:
            seen_a[row[0]] += 1
        for row in B:
            seen_b[row[0]] += 1
        if not A or not B:
            continue
        M = np.array([[_iou(x[1], y[1]) for y in B] for x in A])
        rows, cols = linear_sum_assignment(-M)
        for i, j in zip(rows, cols):
            if M[i, j] >= MATCH_IOU:
                votes[(A[i][0], B[j][0])] += 1
                ious[(A[i][0], B[j][0])].append(float(M[i, j]))
    return {'frames': len(shared), 'votes': votes, 'seenA': seen_a, 'seenB': seen_b, 'iou': ious}


def accept_pairs(v):
    """The accepted pairs of a boundary: mutual best, enough frames, most of the frames both were seen."""
    best_a, best_b = {}, {}
    for (x, y), n in v['votes'].items():
        if n > best_a.get(x, (None, 0))[1]:
            best_a[x] = (y, n)
        if n > best_b.get(y, (None, 0))[1]:
            best_b[y] = (x, n)
    pairs = []
    for x, (y, n) in best_a.items():
        if best_b.get(y, (None, 0))[0] != x:
            continue
        fewer = min(v['seenA'][x], v['seenB'][y])
        if n >= MIN_VOTES and n >= SHARE*fewer:
            pairs.append({'a': x, 'b': y, 'votes': n, 'share': round(n/max(1, fewer), 3),
                          'iou': round(float(np.mean(v['iou'][(x, y)])), 3)})
    return pairs


# ---- 3. team letters -----------------------------------------------------------------------------------------------
def team_mapping(digests, boundaries):
    """{segment: swap} so that the match's A is one kit throughout: the overlap's matched people decide where they can,
    the kit colours (against the running match prototypes) everywhere else; with the method and its margin."""
    out = {}
    protos = {'A': None, 'B': None}
    weight = {'A': 0.0, 'B': 0.0}

    def update(d, swap):
        kit = d['kit']
        if not kit['learned'] or kit['a'] is None or kit['b'] is None:
            return
        for letter, hist in (('A', kit['a']), ('B', kit['b'])):
            mapped = _swap(letter, swap)
            h = np.asarray(hist, float)
            w = float(max(1, kit.get('samples') or 1))
            protos[mapped] = h*w if protos[mapped] is None else protos[mapped]+h*w
            weight[mapped] += w

    def by_kit(d):
        kit = d['kit']
        if protos['A'] is None or protos['B'] is None or not kit['learned'] or kit['a'] is None or kit['b'] is None:
            return False, 0.0
        same = _bc(kit['a'], protos['A'])+_bc(kit['b'], protos['B'])
        cross = _bc(kit['a'], protos['B'])+_bc(kit['b'], protos['A'])
        return cross > same, abs(cross-same)/2

    by_index = {d['index']: d for d in digests}
    for d in digests:
        i = d['index']
        b = boundaries.get(i)   # the boundary with the previous segment, if they share frames
        decided = None
        if b is not None:
            prev = by_index[b['previous']]
            agree = Counter()
            for p in b['pairs']:
                ta = _swap(prev['entries'][p['a']]['team'], out[prev['index']]['swap'])
                tb = d['entries'][p['b']]['team']
                if ta in ('A', 'B') and tb in ('A', 'B'):
                    agree['same' if ta == tb else 'swap'] += 1
            total = sum(agree.values())
            if total >= TEAM_PAIRS and max(agree.values()) >= TEAM_AGREE*total:
                swap = agree['swap'] > agree['same']
                decided = {'swap': swap, 'method': 'overlap', 'evidence': dict(agree)}
        if decided is None:
            swap, margin = by_kit(d)
            decided = {'swap': swap, 'method': 'kit' if protos['A'] is not None else 'first', 'margin': round(margin, 3)}
        out[i] = decided
        update(d, decided['swap'])
    return out, {k: (v/weight[k]).round(4).tolist() if v is not None and weight[k] else None for k, v in protos.items()}


# ---- 4-6. chains, links, roster --------------------------------------------------------------------------------------
class _Union:
    def __init__(self):
        self.parent = {}

    def find(self, x):
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def _chain_stats(chain, digests_by_index, mapping, fps):
    members = sorted(chain, key=lambda m: m[0])
    frames = 0
    sx = sy = n = 0.0
    first = last = None
    history = []
    roles = Counter()
    teams = Counter()
    for seg, pid in members:
        d = digests_by_index[seg]
        o = d['owned'].get(pid)
        e = d['entries'][pid]
        roles[e['role']] += (o or {}).get('frames', 0) or 1
        team = _swap(e['team'], mapping[seg]['swap'])
        if team in ('A', 'B'):
            teams[team] += (o or {}).get('frames', 0) or 1
        if not o:
            continue
        frames += o['frames']
        sx += o['sx']
        sy += o['sy']
        n += o['n']
        first = o['first'] if first is None else min(first, o['first'])
        last = o['last'] if last is None else max(last, o['last'])
        history.extend(o['history'])
    role = roles.most_common(1)[0][0] if roles else 'PLAYER'
    team = teams.most_common(1)[0][0] if teams else None
    return {'members': members, 'frames': frames, 'seconds': round(frames/fps, 2), 'first': first, 'last': last,
            'mean': (sx/n, sy/n) if n else None, 'history': sorted(history), 'role': role, 'team': team,
            'teamVotes': dict(teams), 'roleVotes': dict(roles)}


def _endpoint(chain, digests_by_index, which):
    """(segment, id, owned record) of a chain's first or last member."""
    seg, pid = chain['members'][0] if which == 'first' else chain['members'][-1]
    return seg, pid, digests_by_index[seg]['owned'].get(pid)


def _appearance(d1, id1, d2, id2):
    """Similarity of two identities' galleries in [0, 1] (OSNet cosine and kit colours, as the runs' own re-ID)."""
    from soccer import appearance
    g1, g2 = d1['galleries'].get(id1) or [], d2['galleries'].get(id2) or []
    if not g1 or not g2:
        return None
    S = appearance.batch_similarity(g1, g2)
    return float(np.median(S))


def link_gaps(chains, digests_by_index, mapping, fps, period_bounds):
    """Pairs (earlier chain, later chain) linked across a gap without overlap evidence (see the module docstring)."""
    candidates = []
    ends = [c for c in chains if c['last'] is not None]
    for x in ends:
        seg_x, id_x, o_x = _endpoint(x, digests_by_index, 'last')
        px = digests_by_index[seg_x]['period']
        if x['last'] >= period_bounds[px][1]-1:
            continue       # seen to the end of its period
        for y in chains:
            if y is x or y['first'] is None or y['team'] != x['team'] or y['role'] != x['role'] or x['team'] is None:
                continue
            seg_y, id_y, o_y = _endpoint(y, digests_by_index, 'first')
            if digests_by_index[seg_y]['period'] != px or seg_y <= seg_x or y['first'] <= x['last']:
                continue
            gap = (y['first']-x['last'])/fps
            if gap > LINK_MAX_GAP_S:
                continue
            if any(s in {m[0] for m in y['members']} for s in {m[0] for m in x['members']}):
                continue       # both in one segment: that run kept them apart
            app = _appearance(digests_by_index[seg_x], id_x, digests_by_index[seg_y], id_y)
            if app is None:
                continue
            size = digests_by_index[seg_x]['pitchSize']
            L, W = float(size.get('length', 105.0)), float(size.get('width', 68.0))
            reach, home = .5, .5
            if o_x and o_x['lastPitch'] and o_y and o_y['firstPitch']:
                dx = (o_x['lastPitch'][0]-o_y['firstPitch'][0])*L
                dy = (o_x['lastPitch'][1]-o_y['firstPitch'][1])*W
                dist = math.hypot(dx, dy)
                dt = max(1/fps, (o_y['firstPitchFrame']-o_x['lastPitchFrame'])/fps)
                if dist > LINK_MAX_SPEED*dt+3.0:
                    continue
                reach = max(0.0, 1-dist/(LINK_MAX_SPEED*dt+3.0))
            if x['mean'] and y['mean']:
                d = math.hypot((x['mean'][0]-y['mean'][0])*L, (x['mean'][1]-y['mean'][1])*W)
                home = max(0.0, 1-d/25.0)
            score = .6*app+.2*reach+.2*home
            candidates.append((score, x, y, {'appearance': round(app, 3), 'reach': round(reach, 3), 'home': round(home, 3),
                                             'gapSeconds': round(gap, 1)}))
    links = []
    taken_x, taken_y = set(), set()
    by_x, by_y = defaultdict(list), defaultdict(list)
    for c in candidates:
        by_x[id(c[1])].append(c[0])
        by_y[id(c[2])].append(c[0])
    for score, x, y, why in sorted(candidates, key=lambda c: -c[0]):
        if id(x) in taken_x or id(y) in taken_y or score < LINK_SCORE:
            continue
        rivals_x = sorted(by_x[id(x)], reverse=True)
        rivals_y = sorted(by_y[id(y)], reverse=True)
        second = max(rivals_x[1] if len(rivals_x) > 1 else 0.0, rivals_y[1] if len(rivals_y) > 1 else 0.0)
        if score-second < LINK_MARGIN:
            continue
        taken_x.add(id(x))
        taken_y.add(id(y))
        links.append((x, y, round(score, 3), why))
    return links


def resolve(segments, folders, pad, fps, progress=None):
    """The match roster and every segment's mapping. segments: plan.Segment list; folders: {index: artifact folder}."""
    digests = []
    for k, s in enumerate(segments):
        digests.append(load_digest(s, folders[s.index], pad))
        if progress:
            progress((k+1)/len(segments)*.6, 'Reading the segments')
    by_index = {d['index']: d for d in digests}
    boundaries = {}
    for a, b in zip(segments, segments[1:]):
        if a.period != b.period or a.own1 != b.own0:
            continue
        v = boundary_votes(by_index[a.index], by_index[b.index])
        pairs = accept_pairs(v)
        boundaries[b.index] = {'previous': a.index, 'next': b.index, 'frames': v['frames'], 'pairs': pairs,
                               'seenA': len(v['seenA']), 'seenB': len(v['seenB'])}
    mapping, prototypes = team_mapping(digests, boundaries)
    # Refuse pairs whose teams disagree after mapping, and keeper-with-player pairs whose roles are both firm.
    union = _Union()
    for b in boundaries.values():
        da, db = by_index[b['previous']], by_index[b['next']]
        kept = []
        for p in b['pairs']:
            ea, eb = da['entries'][p['a']], db['entries'][p['b']]
            ta, tb = _swap(ea['team'], mapping[da['index']]['swap']), _swap(eb['team'], mapping[db['index']]['swap'])
            if ta in ('A', 'B') and tb in ('A', 'B') and ta != tb:
                p['refused'] = 'teams differ'
                continue
            union.union((da['index'], p['a']), (db['index'], p['b']))
            kept.append(p)
        b['accepted'] = len(kept)
    nodes = [(d['index'], pid) for d in digests for pid, e in d['entries'].items() if e['group'] in ACTIVE_GROUPS]
    groups = defaultdict(list)
    for node in nodes:
        groups[union.find(node)].append(node)
    chains = [_chain_stats(members, by_index, mapping, fps) for members in groups.values()]
    period_bounds = {}
    for s in segments:
        lo, hi = period_bounds.get(s.period, (s.own0, s.own1))
        period_bounds[s.period] = (min(lo, s.own0), max(hi, s.own1))
    if progress:
        progress(.75, 'Linking players across the boundaries')
    links = link_gaps(chains, by_index, mapping, fps, period_bounds)
    merged = _Union()
    for c in chains:
        merged.find(id(c))
    for x, y, _, _ in links:
        merged.union(id(x), id(y))
    final = defaultdict(list)
    for c in chains:
        final[merged.find(id(c))].append(c)
    link_notes = {(id(x), id(y)): (score, why) for x, y, score, why in links}
    persons = []
    for parts in final.values():
        parts.sort(key=lambda c: (c['first'] is None, c['first'] or 0))
        members = [m for c in parts for m in c['members']]
        person = _chain_stats(members, by_index, mapping, fps)
        person['links'] = [{'score': link_notes[(id(a), id(b))][0], **link_notes[(id(a), id(b))][1]}
                           for a, b in zip(parts, parts[1:]) if (id(a), id(b)) in link_notes]
        person['how'] = ('linked across gaps' if person['links'] else
                         'matched over the overlaps' if len({m[0] for m in members}) > 1 else 'one segment')
        persons.append(person)
    roster = number(persons, fps, period_bounds)
    ids = defaultdict(dict)
    for person in roster:
        for seg, pid in person['members']:
            ids[seg][pid] = person['matchId']
    # Identities that are not match people (retired, excluded, withdrawn): a name unique in the match, their group kept.
    for d in digests:
        for pid, e in d['entries'].items():
            if pid not in ids[d['index']]:
                ids[d['index']][pid] = f"S{d['index']:02d}-{pid}"
    periods = attacking_directions(digests, mapping, segments)
    # Who was seen with whom (owned frames), by match id: two identities seen in one frame are two people (the coach's
    # merges are checked against it).
    together = Counter()
    for d in digests:
        names = ids[d['index']]
        for (a, b), n in d['together'].items():
            ma, mb = names.get(a), names.get(b)
            if ma and mb and ma != mb:
                together['|'.join(sorted((ma, mb)))] += n
    if progress:
        progress(1.0, 'Players stitched')
    return {'mapping': {i: {'swap': m['swap'], 'method': m['method'], **{k: v for k, v in m.items() if k not in ('swap', 'method')}}
                        for i, m in mapping.items()},
            'ids': {i: dict(v) for i, v in ids.items()}, 'roster': roster, 'prototypes': prototypes,
            'boundaries': [boundaries[k] for k in sorted(boundaries)], 'links': len(links), 'periods': periods,
            'together': dict(together)}


def number(persons, fps, period_bounds):
    """Match ids and display labels: per team by first appearance (A-01.., B-01..), goalkeepers GK-1/GK-2 by team order
    (GK-3.. for more), the rest of the unteamed GK-n; short chains show A-?."""
    out = []
    counters = Counter()
    keepers = set()
    first_keeper = {'A': 1, 'B': 2}   # the first keeper of team A is GK-1, of B GK-2; any later one GK-3, GK-4, ...
    for p in sorted(persons, key=lambda p: (p['first'] is None, p['first'] or 0, p['members'][0])):
        team, role = p['team'], p['role']
        if role == 'GOALKEEPER':
            n = first_keeper.pop(team, None) or 3
            while f'GK-{n}' in keepers:
                n += 1
            match_id = f'GK-{n}'
            keepers.add(match_id)
            display = f'GK-{team}' if team in ('A', 'B') else 'GK'
        else:
            letter = team if team in ('A', 'B') else 'U'
            counters[letter] += 1
            match_id = f'{letter}-{counters[letter]:02d}'
            display = match_id if p['seconds'] >= DISPLAY_MIN_S and letter != 'U' else f'{letter}-?'
        entry = {'matchId': match_id, 'display': display, 'team': team, 'role': role, 'seconds': p['seconds'],
                 'firstFrame': p['first'], 'lastFrame': p['last'],
                 'firstSeen': round(p['first']/fps, 2) if p['first'] is not None else None,
                 'lastSeen': round(p['last']/fps, 2) if p['last'] is not None else None,
                 'members': [list(m) for m in p['members']], 'segments': sorted({m[0] for m in p['members']}),
                 'how': p['how'], 'links': p['links'], 'mean': [round(v, 4) for v in p['mean']] if p['mean'] else None,
                 'history': p['history'][::4], 'teamVotes': p['teamVotes'], 'roleVotes': p['roleVotes'],
                 'confidence': 'high' if p['how'] == 'matched over the overlaps' else 'medium' if p['how'] == 'one segment' else 'low'}
        out.append(entry)
    substitutions(out, fps, period_bounds)
    return out


def substitutions(roster, fps, period_bounds):
    """Possible changes, never asserted: a chain with real participation ending well before its period does; one
    starting well after its period began."""
    for p in roster:
        p['substitution'] = None
        if p['role'] != 'PLAYER' or p['firstFrame'] is None or p['seconds'] < SUB_MIN_S:
            continue
        for lo, hi in period_bounds.values():
            if lo <= p['firstFrame'] < hi:
                notes = []
                if (hi-p['lastFrame'])/fps > SUB_EDGE_S:
                    notes.append(f"last seen at {p['lastSeen']:.0f} s, {(hi-p['lastFrame'])/fps/60:.0f} min before the period ends")
                if (p['firstFrame']-lo)/fps > SUB_EDGE_S:
                    notes.append(f"first seen at {p['firstSeen']:.0f} s, {(p['firstFrame']-lo)/fps/60:.0f} min into the period")
                if notes:
                    p['substitution'] = {'status': 'possible', 'notes': notes,
                                         'note': 'Not confirmed: a player can be off camera for long stretches; confirm or correct it.'}


def attacking_directions(digests, mapping, segments):
    """Per period: which match team defends the left goal, by the segments' goal ends (mapped letters), and so attacks
    to the right. Coordinates stay physical; views that want 'attacking to the right' derive it from this."""
    votes = defaultdict(Counter)
    for d in digests:
        ends = d['goalEnds'] or {}
        left = _swap(ends.get('left'), mapping[d['index']]['swap'])
        right = _swap(ends.get('right'), mapping[d['index']]['swap'])
        if left in ('A', 'B'):
            votes[d['period']][('left', left)] += 1
        if right in ('A', 'B'):
            votes[d['period']][('right', right)] += 1
    out = {}
    for period in sorted({s.period for s in segments}):
        v = votes.get(period, Counter())
        left_a = v[('left', 'A')]+v[('right', 'B')]
        left_b = v[('left', 'B')]+v[('right', 'A')]
        if left_a == left_b == 0:
            out[period] = {'defendsLeft': None, 'votes': {}}
        else:
            out[period] = {'defendsLeft': 'A' if left_a >= left_b else 'B', 'votes': {'A': left_a, 'B': left_b},
                           'agreement': round(max(left_a, left_b)/max(1, left_a+left_b), 3)}
    return out
