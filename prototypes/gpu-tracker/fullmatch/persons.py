"""Match-wide PEOPLE and on-field SLOTS over the stitched fragments of a full match (Full Match Intelligence).

Three identities, kept apart:
    fragment  a match chain of fullmatch/identities.py: one segment run's identity, joined over the segment overlaps
              where both runs saw the player ('A-57'). An internal track: no cap, never shown as a number.
    person    one real person as far as the evidence goes: fragments joined only where the link is certain enough
              ('A-P07'). A real substitute is always a different person; nothing here joins two people to save a label.
    slot      a place in the team's on-field eleven: 01-10 for outfield players, GK for the keeper. The board shows the
              slot: 'A-04' for the person that began in it, 'A-04?' for a later person whose continuation of that place
              is a best estimate, 'A-?' for a fragment no place claims. Never a number past 10.

Why the old labels reached A-197 (measured on CAN-ISL, scratch audit): every fragment was numbered. A player is followed
across a 180 s segment boundary only when both runs saw him in the ~6 s overlap (a median 12 of ~22 people per
boundary), so 295 of 392 fragments begin at a segment start; and 27 'players' were two static detections on a broadcast
graphic in the corner of the picture.

1. Fragments that are not players: a SCREEN OVERLAY (the box stays put in the picture, within OVERLAY_SPREAD of the
   frame, while real people move; the hot spots where such boxes sit), people OFF THE PITCH (OFFPITCH_SHARE of their
   positions beyond the lines) and fragments NOT IN THE TEAM'S KIT (shirt and shorts unlike the team's: an official, a
   coach). Left out of the people and the analytics with the reason, never deleted; the coach can take them back.
2. Keepers: a keeper's kit is not the team's, so its team is the GOAL it stands at (with the periods' attacking
   directions); a team's keeper fragments never seen together are one person.
3. Outfield players, per team: the ten PLACES of the team over the whole match, solved at once (offline, so later
   evidence counts as much as earlier). Every fragment is scored against every fragment that ended before it began
   (GAP_MAX_S) with a logistic model (LINK_MODEL) of
       role   where each stands relative to the teammates seen with it (team coordinates, attacking to the right): the
              offsets to the same anchor fragments near the junction; the strongest cue on a broadcast camera
       place  each one's mean place relative to the teammates seen with it
       reach  the distance between where one was last and the other first seen, against the time between
       look   the runs' own appearance galleries (every descriptor's best match, both ways); weak between teammates
   fitted on CAN-ISL's overlap-matched fragments with the overlap hidden (scratch fit_links.py). The ten places are then
   the minimum-cost set of ten time-ordered paths through the fragments (a min-cost flow: covering a fragment earns
   REWARD, a link costs its negative log-odds), so at any moment a fragment can only continue a place that is free
   then: the team's roster is the constraint the pairwise scores alone lack (they pick the right continuation of ~5
   candidates 60% of the time at a 20 s gap).
   A link's CONFIDENCE is the softmax of its score against the other places that were free when the fragment began.
   Measured with hidden overlap links: a link scored .5-.8 is right 57-72% of the time, .8 and over 79%. People are the
   runs of a place joined by links of P_PERSON or more; a later person in a place shows the place with '?'.
4. Slot numbers: the places numbered by the line of the person that began each (defence, midfield, attack, deepest
   first) and from left to right.
5. Substitutions are candidates, never facts: where a place passes to someone who looks different (LOOK_SUB), with
   where and when it happened; the incoming player is a different person whatever the coach decides about the slot.

Nothing here changes a fragment, a position or a segment artifact: build() writes the automatic result (cached by the
inputs' fingerprint), apply() turns it and the coach's corrections (fullmatch/corrections.py: merge, split, exclude,
include, team, keeper, substitution) into the roster the page and the analytics read; a correction that changes the
places solves them again from the stored link scores (no video, no tracking).
"""
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

VERSION = 2
# 1. not players
OVERLAY_SPREAD = .004         # box centre spread (fraction of the frame) of a box that stays put in the picture
OVERLAY_MIN_SAMPLES = 25      # 5 s at 5 Hz
OVERLAY_RADIUS = .015         # a box centre this close to a hot spot sits on the overlay
OVERLAY_SHARE = .5            # share of a fragment's boxes on hot spots that makes it an overlay
OFFPITCH_MARGIN_M = 1.0
OFFPITCH_SHARE = .5
KIT_MIN_BC = .4               # shirt and shorts likeness to the team's kit (Bhattacharyya) below which it is not a player
KIT_MIN_BC_ONE = .25          # ... for a fragment with a single gallery sample
MIXED_BC = .4                 # two stitched members less alike than this in shirt AND shorts are two people
# 3. places
BIN_S = 60.0
ROLE_WINDOW_S = 900.0         # anchor offsets compared within this of the later fragment's span
ANCHOR_MIN = 10               # co-present samples an anchor needs on each side (2 s at 5 Hz)
REL_MIN = 25                  # samples behind a fragment's place relative to the teammates seen with it
REACH_SPEED = 9.0             # m/s: farther than this (plus the slack) in the time between is someone else
REACH_SLACK_M = 6.0
REACH_VETO_S = 120.0          # beyond this gap reach vetoes nothing
GAP_MAX_S = 1800.0
PREDECESSORS = 25             # the best-scored earlier fragments kept per fragment
LINK_MODEL = {'bias': -2.3024, 'role': -.6411, 'noRole': -1.3883, 'logAnchors': .8301, 'rel': -.4751,
              'noRel': -.0586, 'lat': -.4214, 'dep': -.3754, 'look': .8923, 'noLook': -1.2661, 'reachShort': -3.2911,
              'reachMid': .2144, 'noReach': -2.5044, 'logGap': .0435, 'short': 1.4967}
OUTFIELD_SLOTS = 10
REWARD = 8.0                  # what covering a fragment is worth (log-odds units), plus REWARD_PER_S per second seen
REWARD_PER_S = .01             # (uncapped: a cap at 300 s left two real 20-minute players of CAN-ISL out of the
                              # places for a few seconds of conflict; without it the hidden-link test is unchanged)
GALLERY_CAP = 48
# 5. substitutions
LOOK_SUB = .76                # a place passing to someone this unlike the person before is a possible substitution
SUB_P = .5                    # ... when the link was less confident than this
TOUCHLINE_M = 8.0

SUB_LINK = 30.0               # a substitution you confirmed: the incoming player continues the outgoing one's place

TEAMS = ('A', 'B')


# ---------------------------------------------------------------- inputs
def galleries(folder, index):
    """{(segment, local id): [(quality, embedding)], ('kit', segment, local id): [(jersey, shorts histograms)]} from
    every segment's identities.json (the runs' own galleries)."""
    from soccer import appearance
    folder = Path(folder)
    by_local = {}
    for s in index['segments']:
        try:
            snap = json.loads((folder/'segments'/s['key']/'identities.json').read_text(encoding='utf-8'))
        except (OSError, ValueError, KeyError):
            continue
        for e in (snap.get('registry') or {}).get('identities', []):
            embs, kits = [], []
            for g in e.get('gallery') or []:
                raw = g.get('d') or {}
                if len(raw.get('j') or []) and len(raw.get('s') or []):
                    kits.append((np.asarray(raw['j'], float), np.asarray(raw['s'], float)))
                d = appearance.decode_descriptor(raw)
                if d is not None and d.embedding is not None:
                    embs.append((float(getattr(d, 'quality', 0.0) or 0.0), np.asarray(d.embedding, np.float32)))
            by_local[(s['index'], e['id'])] = embs
            by_local[('kit', s['index'], e['id'])] = kits
    return by_local


def _stack(embs):
    if not embs:
        return None
    embs = sorted(embs, key=lambda q: -q[0])[:GALLERY_CAP]
    size = len(embs[0][1])
    E = np.stack([e for _, e in embs if len(e) == size]).astype(np.float32)
    n = np.linalg.norm(E, axis=1, keepdims=True)
    E = E/np.maximum(n, 1e-9)
    return E


def _look(E1, E2):
    """Appearance similarity of two galleries: every descriptor's best match in the other gallery, averaged both ways
    (top-1 55% across a boundary on CAN-ISL, where the median of all pairs gets 41%)."""
    if E1 is None or E2 is None or not len(E1) or not len(E2) or E1.shape[1] != E2.shape[1]:
        return None
    S = E1 @ E2.T
    return float((S.max(axis=1).mean()+S.max(axis=0).mean())/2)


class Fragment:
    __slots__ = ('id', 'team', 'role', 'kit', 'rows', 's', 't', 'x', 'y', 'z', 'w', 'cx', 'cy', 't0', 't1', 'n',
                 'seconds', 'flag', 'reason', 'goal_team', 'rel', 'members', 'mean', 'history', 'first_frame',
                 'last_frame', 'E', 'period', 'kits', 'pf', 'pl', 'mixed')


def mixed(members, gals, seg_start=None):
    """Where a stitched chain joins two people, or None: the first two consecutive members of which one was a keeper
    and the other an outfield player in their runs (local ids GK-n), or whose shirts and shorts are both unlike
    (MIXED_BC). On CAN-ISL two chains of 392, both a keeper stitched to a player at a segment start (the kit likeness
    of one 0.25, of the other 0.75: lime and white; of the other 406 links the lowest 0.67, 99% above 0.85); by making
    eleven of a team at once they held two long real players out of the places."""
    def kit(m):
        ks = (gals or {}).get(('kit', m[0], m[1])) or []
        if not ks:
            return None
        return np.mean([k[0] for k in ks], axis=0), np.mean([k[1] for k in ks], axis=0)
    for m1, m2 in zip(members, members[1:]):
        why = None
        if str(m1[1]).startswith('GK') != str(m2[1]).startswith('GK'):
            why = 'keeper'
        else:
            k1, k2 = kit(m1), kit(m2)
            if k1 is not None and k2 is not None and max(_bc(k1[0], k2[0]), _bc(k1[1], k2[1])) < MIXED_BC:
                why = 'kit'
        if why:
            at = (seg_start or {}).get(m2[0])
            return {'segment': int(m2[0]), 'why': why, 'at': None if at is None else round(float(at), 1)}
    return None


def fragments(index, data, gals=None, split=None, gap_cut=0.0):
    """The fragments of the match with their observations (data: soccer.moments.data.MatchData over the owned frames).
    split / gap_cut: a measurement hook (scratch evaluation only): {(matchId, segment)} cuts that chain before that
    segment into two fragments ('<id>~<segment>'), dropping gap_cut/2 seconds of observations on each side."""
    L, W = data.length, data.width
    obs_s = data.obs_sample()
    cx = data.box[:, 0]+data.box[:, 2]/2
    cy = data.box[:, 1]+data.box[:, 3]/2
    seg_of = data.segment[obs_s]
    seg_start = {s['index']: float(s['start']) for s in index['segments']}
    dirs = {t: _directions(data, t) for t in TEAMS}
    out = []
    split = split or set()
    for r in index['roster']:
        pid = data.index_of.get(r['matchId'])
        rows = np.flatnonzero(data.pid == pid) if pid is not None else np.zeros(0, np.int64)
        cuts = sorted(seg for (mid, seg) in split if mid == r['matchId'])
        pieces = []
        if cuts:
            bounds = [None]+cuts+[None]
            for a, b in zip(bounds, bounds[1:]):
                m = np.ones(len(rows), bool)
                if a is not None:
                    m &= seg_of[rows] >= a
                    m &= data.t[obs_s[rows]] >= seg_start[a]+gap_cut/2
                if b is not None:
                    m &= seg_of[rows] < b
                    m &= data.t[obs_s[rows]] < seg_start[b]-gap_cut/2
                name = r['matchId'] if a is None else f"{r['matchId']}~{a}"
                members = [mm for mm in r['members'] if (a is None or mm[0] >= a) and (b is None or mm[0] < b)]
                pieces.append((name, rows[m], members))
        else:
            pieces.append((r['matchId'], rows, r['members']))
        for name, rr, members in pieces:
            f = Fragment()
            f.id, f.kit, f.role = name, r.get('team'), r.get('role') or 'PLAYER'
            f.team = f.kit if f.kit in TEAMS else None
            f.rows = rr
            f.s = obs_s[rr]
            f.t = data.t[f.s]
            f.x, f.y = data.x[rr].astype(np.float64), data.y[rr].astype(np.float64)
            f.cx, f.cy = cx[rr], cy[rr]
            f.n = len(rr)
            f.t0 = float(f.t[0]) if f.n else None
            f.t1 = float(f.t[-1]) if f.n else None
            f.seconds = float(r.get('seconds') or 0.0) if name == r['matchId'] and not cuts else round(f.n/5.0, 2)
            f.flag, f.reason, f.goal_team, f.rel = None, None, None, None
            f.members = [list(m) for m in members]
            f.mean, f.history = r.get('mean'), r.get('history') or []
            f.first_frame, f.last_frame = r.get('firstFrame'), r.get('lastFrame')
            f.E = _stack([e for seg, local in members for e in (gals or {}).get((seg, local), [])])
            f.kits = [k for seg, local in members for k in (gals or {}).get(('kit', seg, local), [])]
            f.mixed = mixed(members, gals, seg_start)
            f.period = int(Counter(data.period[f.s].tolist()).most_common(1)[0][0]) if f.n else -1
            f.pf = int(data.period[f.s[0]]) if f.n else -1
            f.pl = int(data.period[f.s[-1]]) if f.n else -1
            f.z = f.w = None
            if f.team and f.n:
                d = dirs[f.team][f.s]
                f.z = np.where(d >= 0, f.x, L-f.x)
                f.w = d*(W/2-f.y)
            out.append(f)
    return out


def _directions(data, team):
    out = np.zeros(data.samples, np.int8)
    for p in data.periods:
        if p.get('attacksRight') not in TEAMS:
            continue
        out[data.period == int(p['index'])] = 1 if p['attacksRight'] == team else -1
    return out


# ---------------------------------------------------------------- 1. not players
def flag_non_players(frags, data):
    """Screen overlays and people off the pitch (see the module doc), in place; returns the overlay hot spots."""
    L, W = data.length, data.width
    still = []
    for f in frags:
        if f.n < OVERLAY_MIN_SAMPLES:
            continue
        spread = max(float(np.nanstd(f.cx)), float(np.nanstd(f.cy)))
        if spread <= OVERLAY_SPREAD:
            moved = math.hypot(float(np.std(f.x)), float(np.std(f.y)))
            still.append((float(np.nanmedian(f.cx)), float(np.nanmedian(f.cy)), moved))
    # a hot spot: where still boxes sit while the camera moved (their pitch position changed), or where two or more
    # still fragments sat at different times
    spots = []
    for x, y, moved in still:
        near = [q for q in still if math.hypot(q[0]-x, q[1]-y) <= OVERLAY_RADIUS]
        if moved >= 2.0 or len(near) >= 2:
            if not any(math.hypot(sx-x, sy-y) <= OVERLAY_RADIUS for sx, sy in spots):
                spots.append((x, y))
    for f in frags:
        if not f.n:
            continue
        if spots:
            on = np.zeros(f.n, bool)
            for sx, sy in spots:
                on |= np.hypot(f.cx-sx, f.cy-sy) <= OVERLAY_RADIUS
            if on.mean() >= OVERLAY_SHARE and f.n >= 5:
                f.flag, f.reason = 'overlay', 'a box that stays put on a broadcast graphic, not a person'
                continue
        off = (f.x < -OFFPITCH_MARGIN_M) | (f.x > L+OFFPITCH_MARGIN_M) | (f.y < -OFFPITCH_MARGIN_M) | (f.y > W+OFFPITCH_MARGIN_M)
        if off.mean() >= OFFPITCH_SHARE:
            f.flag, f.reason = 'offpitch', 'seen mostly beyond the lines (bench, warm-up, staff)'
    flag_kits(frags)
    return [list(s) for s in spots]


def _bc(p, q):
    p, q = np.asarray(p, float), np.asarray(q, float)
    if p.sum() <= 0 or q.sum() <= 0:
        return 0.0
    return float(np.sum(np.sqrt(p/p.sum()*q/q.sum())))


def flag_kits(frags):
    """In place: outfield fragments whose shirt AND shorts do not look like their team's kit (the team's own fragments
    averaged, by observed time): a coach, a steward or an official the runs took for a player. KIT_MIN_BC on the worse
    of the two; a fragment with a single gallery sample must be clearer (KIT_MIN_BC_ONE)."""
    protos = {}
    for team in TEAMS:
        js, ss, ws = [], [], []
        for f in frags:
            if f.team == team and f.role != 'GOALKEEPER' and not f.flag and f.kits:
                js.append(np.mean([k[0] for k in f.kits], axis=0))
                ss.append(np.mean([k[1] for k in f.kits], axis=0))
                ws.append(min(f.seconds or 0.0, 300.0)+1.0)
        if js:
            protos[team] = (np.average(js, axis=0, weights=ws), np.average(ss, axis=0, weights=ws))
    for f in frags:
        if f.flag or f.role == 'GOALKEEPER' or f.team not in protos or not f.kits:
            continue
        j = np.mean([k[0] for k in f.kits], axis=0)
        sh = np.mean([k[1] for k in f.kits], axis=0)
        worst = min(_bc(j, protos[f.team][0]), _bc(sh, protos[f.team][1]))
        if worst < (KIT_MIN_BC if len(f.kits) >= 2 else KIT_MIN_BC_ONE):
            f.flag, f.reason = 'kit', 'not in the team kit (an official, a coach or staff)'


# ---------------------------------------------------------------- 2. keepers
def keeper_teams(frags, data):
    """In place: every keeper fragment's team by the goal it stands at (the periods' attacking directions)."""
    L = data.length
    defends_left = {}
    for p in data.periods:
        a = p.get('attacksRight')
        if a in TEAMS:
            defends_left[int(p['index'])] = a
    for f in frags:
        if f.role != 'GOALKEEPER' or not f.n or f.flag:
            continue
        votes = Counter()
        for q, x in zip(data.period[f.s].tolist(), f.x.tolist()):
            dl = defends_left.get(int(q))
            if dl is None:
                continue
            votes[dl if x < L/2 else ('B' if dl == 'A' else 'A')] += 1
        if votes:
            team, n = votes.most_common(1)[0]
            if n >= .7*sum(votes.values()):
                f.goal_team = team
        f.team = f.goal_team or f.team


# ---------------------------------------------------------------- co-presence and offsets
class Pairs:
    """Per team: which fragments were seen together (5 Hz) and their offsets in team coordinates, per BIN_S bin."""

    def __init__(self, frags, data):
        self.nbins = int(math.ceil((float(data.t[-1]) if data.samples else 1.0)/BIN_S))+1
        self.sums = {}            # (i, j) i < j -> (nbins, 3): sum dz (i - j), sum dw, count
        self.rel = {}             # i -> [sum dz to the visible centroid, sum dw, count]
        by_sample = defaultdict(list)
        for i, f in enumerate(frags):
            if f.flag or f.team is None or f.role == 'GOALKEEPER' or not f.n:
                continue
            for k in range(f.n):
                by_sample[int(f.s[k])].append((i, k))
        keys_i, keys_j, bins, dzs, dws = [], [], [], [], []
        rel = defaultdict(lambda: np.zeros(3))
        for s, items in by_sample.items():
            if len(items) < 2:
                continue
            b = int(data.t[s]//BIN_S)
            for team in TEAMS:
                sel = [(i, k) for i, k in items if frags[i].team == team]
                if len(sel) < 2:
                    continue
                ii = np.array([i for i, _ in sel])
                z = np.array([frags[i].z[k] for i, k in sel])
                w = np.array([frags[i].w[k] for i, k in sel])
                cz, cw = z.mean(), w.mean()
                for i, zz, ww in zip(ii, z, w):
                    rel[int(i)] += (zz-cz, ww-cw, 1)
                a, c = np.triu_indices(len(ii), 1)
                lo = np.minimum(ii[a], ii[c])
                hi = np.maximum(ii[a], ii[c])
                sign = np.where(ii[a] < ii[c], 1.0, -1.0)
                keys_i.append(lo)
                keys_j.append(hi)
                bins.append(np.full(len(a), b))
                dzs.append(sign*(z[a]-z[c]))
                dws.append(sign*(w[a]-w[c]))
        self.rel = {i: v for i, v in rel.items()}
        if keys_i:
            ki, kj, bb = np.concatenate(keys_i), np.concatenate(keys_j), np.concatenate(bins)
            dz, dw = np.concatenate(dzs), np.concatenate(dws)
            order = np.lexsort((bb, kj, ki))
            ki, kj, bb, dz, dw = ki[order], kj[order], bb[order], dz[order], dw[order]
            starts = np.flatnonzero(np.r_[True, (np.diff(ki) != 0) | (np.diff(kj) != 0)])
            ends = np.r_[starts[1:], len(ki)]
            for a, e in zip(starts, ends):
                arr = np.zeros((self.nbins, 3))
                np.add.at(arr[:, 0], bb[a:e], dz[a:e])
                np.add.at(arr[:, 1], bb[a:e], dw[a:e])
                np.add.at(arr[:, 2], bb[a:e], 1.0)
                self.sums[(int(ki[a]), int(kj[a]))] = arr


# ---------------------------------------------------------------- 3. link scores
class Scorer:
    """Log-odds that a fragment z, first seen after fragment p was last seen, is p (see the module doc, step 3)."""

    def __init__(self, frags, pairs):
        self.frags, self.pairs = frags, pairs
        self.nbr = defaultdict(set)
        for (i, j) in pairs.sums:
            self.nbr[i].add(j)
            self.nbr[j].add(i)

    def _offsets(self, a, b, lo, hi):
        """Mean offset (depth, lateral) of fragment a to fragment b over bins lo..hi, and the count."""
        arr = self.pairs.sums.get((a, b) if a < b else (b, a))
        if arr is None:
            return None
        part = arr[lo:hi+1]
        n = part[:, 2].sum()
        if n <= 0:
            return None
        sgn = 1.0 if a < b else -1.0
        return sgn*part[:, 0].sum()/n, sgn*part[:, 1].sum()/n, n

    def role(self, p, z):
        """Weighted distance (m) between z's and p's offsets to the same anchors near z's span, and the anchors."""
        Z = self.frags[z]
        lo = max(0, int((Z.t0-ROLE_WINDOW_S)//BIN_S))
        hi = min(self.pairs.nbins-1, int((Z.t1+ROLE_WINDOW_S)//BIN_S))
        num = den = 0.0
        anchors = 0
        for q in self.nbr[z]:
            if q == p:
                continue
            oz = self._offsets(z, q, lo, hi)
            if oz is None or oz[2] < ANCHOR_MIN:
                continue
            op = self._offsets(p, q, lo, hi)
            if op is None or op[2] < ANCHOR_MIN:
                continue
            w = math.sqrt(min(oz[2], op[2]))
            num += w*math.hypot(oz[0]-op[0], oz[1]-op[1])
            den += w
            anchors += 1
        if anchors < 2:
            return None, anchors
        return num/den, anchors

    def score(self, p, z):
        """(log-odds, features), or None when z cannot be p (it began before p ended, or the gap cannot be run)."""
        P, Z = self.frags[p], self.frags[z]
        gap = float(Z.t0-P.t1)
        if gap <= 0:
            return None
        dist = math.hypot(Z.x[0]-P.x[-1], Z.y[0]-P.y[-1]) if Z.pf == P.pl else None
        if dist is not None and gap <= REACH_VETO_S and dist > REACH_SPEED*gap+REACH_SLACK_M:
            return None
        role, anchors = self.role(p, z)
        a, b = self.pairs.rel.get(z), self.pairs.rel.get(p)
        known = a is not None and b is not None and a[2] >= REL_MIN and b[2] >= REL_MIN
        dz = abs(a[0]/a[2]-b[0]/b[2]) if known else None
        dw = abs(a[1]/a[2]-b[1]/b[2]) if known else None
        rel = math.hypot(dz, dw) if known else None
        look = _look(Z.E, P.E)
        g = max(gap, .2)
        if dist is not None and g <= 60:
            reach = min(dist/(3.0+2.5*math.sqrt(g)), 6.0)
            rs, rm, nr = reach*math.exp(-g/10), reach*(1-math.exp(-g/10)), 0.0
        else:
            reach, rs, rm, nr = None, 0.0, 0.0, 1.0
        x = {'bias': 1.0, 'role': min(role, 40.0)/10 if role is not None else 0.0, 'noRole': float(role is None),
             'logAnchors': math.log1p(anchors), 'rel': min(rel, 40.0)/10 if rel is not None else 0.0,
             'noRel': float(rel is None), 'lat': min(dw, 40.0)/10 if dw is not None else 0.0,
             'dep': min(dz, 40.0)/10 if dz is not None else 0.0, 'look': (look-.77)*20 if look is not None else 0.0,
             'noLook': float(look is None), 'reachShort': rs, 'reachMid': rm, 'noReach': nr, 'logGap': math.log(g),
             'short': float(g <= 30)}
        s = sum(LINK_MODEL[k]*v for k, v in x.items())
        return s, {'role': None if role is None else round(role, 1), 'anchors': anchors,
                   'place': None if rel is None else round(rel, 1), 'reach': None if reach is None else round(reach, 2),
                   'look': None if look is None else round(look, 3), 'gap': round(gap, 1)}


# ---------------------------------------------------------------- 3. the places: K paths of minimum cost
def min_cost_tracks(order, edges, K, reward):
    """K node-disjoint paths through the nodes (`order`: ids in time order; every edge goes forward in it), a covered
    node earning reward[id] and a link (a, b) costing edges[(a, b)]: successive shortest paths with Johnson potentials.
    Returns ({id: successor}, [path starts])."""
    import heapq
    n = len(order)
    pos = {v: k for k, v in enumerate(order)}
    S, T = 0, 1
    N = 2+2*n
    to, cap, cost, nxt = [], [], [], []
    head = [-1]*N

    def add(u, v, c):
        for a, b, cc, cp in ((u, v, c, 1), (v, u, -c, 0)):
            to.append(b)
            cap.append(cp)
            cost.append(cc)
            nxt.append(head[a])
            head[a] = len(to)-1
    for k, v in enumerate(order):
        add(S, 2+2*k, 0.0)
        add(2+2*k, 3+2*k, -float(reward[v]))
        add(3+2*k, T, 0.0)
    for (a, b), c in edges.items():
        add(3+2*pos[a], 2+2*pos[b], float(c))
    INF = float('inf')
    h = [INF]*N
    h[S] = 0.0
    for u in [S]+[x for k in range(n) for x in (2+2*k, 3+2*k)]+[T]:
        if h[u] == INF:
            continue
        e = head[u]
        while e != -1:
            if cap[e] > 0 and h[u]+cost[e] < h[to[e]]:
                h[to[e]] = h[u]+cost[e]
            e = nxt[e]
    h = [0.0 if x == INF else x for x in h]
    for _ in range(K):
        dist = [INF]*N
        prev = [-1]*N
        dist[S] = 0.0
        heap = [(0.0, S)]
        while heap:
            d, u = heapq.heappop(heap)
            if d > dist[u]:
                continue
            e = head[u]
            while e != -1:
                if cap[e] > 0:
                    v = to[e]
                    nd = d+cost[e]+h[u]-h[v]
                    if nd < dist[v]-1e-12:
                        dist[v] = nd
                        prev[v] = e
                        heapq.heappush(heap, (nd, v))
                e = nxt[e]
        if dist[T] == INF:
            break
        for v in range(N):
            if dist[v] < INF:
                h[v] += dist[v]
        v = T
        while v != S:
            e = prev[v]
            cap[e] -= 1
            cap[e ^ 1] += 1
            v = to[e ^ 1]
    succ, starts = {}, []
    for k, v in enumerate(order):
        e = head[3+2*k]
        while e != -1:
            if e % 2 == 0 and cap[e] == 0 and to[e] >= 2 and to[e] % 2 == 0:
                succ[v] = order[(to[e]-2)//2]
            e = nxt[e]
    e = head[S]
    while e != -1:
        if e % 2 == 0 and cap[e] == 0:
            starts.append(order[(to[e]-2)//2])
        e = nxt[e]
    starts.sort(key=lambda v: pos[v])
    return succ, starts


def solve_places(nodes, scores, K=OUTFIELD_SLOTS, forbidden=()):
    """The team's places: nodes {id: {'t0', 't1', 'seconds'}}; scores {(a, b): log-odds} between nodes (a ended before b
    began). Returns (tracks [[ids]], confidence {(a, b): p})."""
    order = sorted(nodes, key=lambda v: (nodes[v]['t0'], v))
    forbidden = set(forbidden)
    edges = {k: -s for k, s in scores.items() if k not in forbidden and k[0] in nodes and k[1] in nodes}
    reward = {v: REWARD+REWARD_PER_S*float(nodes[v]['seconds'] or 0.0) for v in nodes}
    succ, starts = min_cost_tracks(order, edges, K, reward)
    tracks = []
    for s in starts:
        tr = [s]
        while tr[-1] in succ:
            tr.append(succ[tr[-1]])
        tracks.append(tr)
    return _confidence(tracks, nodes, {k: s for k, s in scores.items() if k not in forbidden})


def number_places(tracks, rel):
    """Slot numbers 1..K: the places by the line of the person that began each (deepest first: 3-5 defenders,
    midfield, 1-3 forwards) and from left to right (positive lateral first); places without a known place last."""
    known = [k for k, _ in enumerate(tracks) if rel.get(k) is not None]
    numbers = {}
    if len(known) >= 6:
        zs = sorted(known, key=lambda k: rel[k][0])
        n = len(zs)
        best = None
        for a in range(3, 6):
            for c in range(1, 4):
                m = n-a-c
                if m < 1:
                    continue
                parts = (zs[:a], zs[a:a+m], zs[a+m:])
                cost = sum(float(np.var([rel[k][0] for k in part])*len(part)) for part in parts)
                if best is None or cost < best[0]:
                    best = (cost, a, c)
        if best is not None:
            _, a, c = best
            m = n-a-c
            num = 1
            for line in (zs[:a], zs[a:a+m], zs[a+m:]):
                for k in sorted(line, key=lambda k: -rel[k][1]):
                    numbers[k] = num
                    num += 1
    num = max(numbers.values(), default=0)
    for k in range(len(tracks)):
        if k not in numbers:
            num += 1
            numbers[k] = num
    return numbers


def link_keepers(frags, team, together):
    """Keeper fragments of one team (by goal) never seen together: one person; a fragment seen with another keeper of
    the team stays apart (one of the two is not the keeper)."""
    ids = sorted((i for i, f in enumerate(frags) if f.role == 'GOALKEEPER' and f.team == team and not f.flag and f.n),
                 key=lambda i: -frags[i].n)
    groups = []
    for i in ids:
        for g in groups:
            if not any(together.get('|'.join(sorted((frags[i].id, frags[j].id)))) or
                       set(frags[i].s.tolist()) & set(frags[j].s.tolist()) for j in g):
                g.append(i)
                break
        else:
            groups.append([i])
    return groups


# ---------------------------------------------------------------- build
def fingerprint(folder, data=None):
    """What the automatic result depends on: the match data's source fingerprint (the index, the views, the possession
    record), the index and who was seen with whom, this module's version."""
    from soccer.moments import data as md
    asm = Path(folder)/'assembly'
    h = hashlib.sha256(f'persons-{VERSION}'.encode())
    h.update(((data.meta.get('fingerprint') if data is not None else None) or md.source_fingerprint(folder)).encode())
    for name in ('match.json', 'together.json'):
        p = asm/name
        if p.is_file():
            h.update(hashlib.sha256(p.read_bytes()).digest())
    return h.hexdigest()


def build(folder, data=None, progress=None, split=None, gap_cut=0.0):
    """The automatic result of an assembled match (see the module doc). data: its MatchData when the caller has it;
    split / gap_cut: the measurement hook of fragments() (scratch evaluation only)."""
    from soccer.moments import data as md
    folder = Path(folder)
    asm = folder/'assembly'
    index = json.loads((asm/'match.json').read_text(encoding='utf-8'))
    try:
        together = json.loads((asm/'together.json').read_text(encoding='utf-8'))
    except (OSError, ValueError):
        together = {}
    if split:
        together = {}
    if data is None:
        data = md.match_data(folder)
    out = build_from(index, data, galleries(folder, index), together, progress=progress, split=split, gap_cut=gap_cut)
    out['fingerprint'] = fingerprint(folder, data)
    return out


def build_from(index, data, gals, together, progress=None, split=None, gap_cut=0.0):
    """build() on what was read: the index (its roster are the fragments), the MatchData, the galleries
    ({(segment, local id): [(quality, embedding)], ('kit', segment, local id): [(jersey, shorts)]}) and who was seen
    with whom ({'id|id': frames})."""
    frags = fragments(index, data, gals, split=split, gap_cut=gap_cut)
    if progress:
        progress(.2, 'Reading the fragments')
    spots = flag_non_players(frags, data)
    keeper_teams(frags, data)
    pairs = Pairs(frags, data)
    scorer = Scorer(frags, pairs)
    teams = {}
    for k, team in enumerate(TEAMS):
        ids = [i for i, f in enumerate(frags) if f.team == team and f.role != 'GOALKEEPER' and not f.flag and f.n]
        scores, feats = {}, {}
        for z in ids:
            cand = []
            for p in ids:
                if p == z or not (0 < frags[z].t0-frags[p].t1 <= GAP_MAX_S):
                    continue
                if p in scorer.nbr[z] or together.get('|'.join(sorted((frags[p].id, frags[z].id)))):
                    continue
                sc = scorer.score(p, z)
                if sc is not None:
                    cand.append((sc[0], p, sc[1]))
            for s, p, f in sorted(cand, key=lambda c: -c[0])[:PREDECESSORS]:
                scores[(p, z)] = s
                feats[(p, z)] = f
        nodes = {i: {'t0': frags[i].t0, 't1': frags[i].t1, 'seconds': frags[i].seconds} for i in ids
                 if not frags[i].mixed}
        tracks, conf = solve_places(nodes, {k: v for k, v in scores.items() if k[0] in nodes and k[1] in nodes})
        teams[team] = {'nodes': [frags[i].id for i in ids],
                       'scores': [[frags[a].id, frags[b].id, round(s, 3)] for (a, b), s in scores.items()],
                       'feats': {f'{frags[a].id}>{frags[b].id}': f for (a, b), f in feats.items()
                                 if any(a == x and b == y for tr in tracks for x, y in zip(tr, tr[1:]))},
                       'tracks': [[frags[i].id for i in tr] for tr in tracks],
                       'keepers': [[frags[i].id for i in g] for g in link_keepers(frags, team, together)]}
        if progress:
            progress(.3+.3*(k+1), f'Placing team {team}')
    frag_out = {}
    for i, f in enumerate(frags):
        rel = pairs.rel.get(i)
        frag_out[f.id] = {'team': f.team, 'kit': f.kit, 'role': f.role, 'seconds': f.seconds, 'samples': f.n,
                          't0': f.t0, 't1': f.t1, 'flag': f.flag, 'reason': f.reason, 'goalTeam': f.goal_team,
                          'mixed': f.mixed,
                          'rel': None if rel is None else [round(float(rel[0]), 1), round(float(rel[1]), 1), int(rel[2])],
                          'first': None if not f.n else [round(float(f.x[0]), 2), round(float(f.y[0]), 2)],
                          'last': None if not f.n else [round(float(f.x[-1]), 2), round(float(f.y[-1]), 2)],
                          'emb': _mean_embedding(f.E)}
    return {'version': VERSION, 'fingerprint': None, 'fragments': frag_out, 'teams': teams,
            'overlaySpots': spots, 'pitch': [data.length, data.width], 'k': OUTFIELD_SLOTS}


def _mean_embedding(E):
    """A fragment's gallery as 8 float16 descriptors (base64), for the look of the people apply() forms."""
    if E is None or not len(E):
        return None
    import base64
    return base64.b64encode(np.asarray(E[:8], np.float16).tobytes()).decode('ascii')


def _decode_embedding(text, dim=512):
    if not text:
        return None
    import base64
    E = np.frombuffer(base64.b64decode(text), np.float16).astype(np.float32)
    if len(E) % dim:
        return None
    E = E.reshape(-1, dim)
    return E/np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-9)


def cached(folder, data=None, progress=None):
    """build() through the cache assembly/intel/persons-<fp16>.json."""
    from soccer.moments import data as md
    folder = Path(folder)
    if data is None:
        data = md.match_data(folder)
    fp = fingerprint(folder, data)
    path = folder/'assembly'/'intel'/f'persons-{fp[:16]}.json'
    if path.is_file():
        try:
            out = json.loads(path.read_text(encoding='utf-8'))
            if out.get('fingerprint') == fp and out.get('version') == VERSION:
                return out
        except (OSError, ValueError):
            pass
    out = build(folder, data=data, progress=progress)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f'.{path.name}.tmp')
    tmp.write_text(json.dumps(out, separators=(',', ':')), encoding='utf-8')
    tmp.replace(path)
    return out


def latest(folder):
    """The cached automatic result of the match as it is now (without building one), or None."""
    try:
        fp = fingerprint(folder)
    except (OSError, ValueError, KeyError):
        return None
    path = Path(folder)/'assembly'/'intel'/f'persons-{fp[:16]}.json'
    try:
        out = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None
    return out if out.get('fingerprint') == fp and out.get('version') == VERSION else None


# ---------------------------------------------------------------- apply: people, places, the roster
PLACE_KINDS = ('merge', 'split', 'exclude', 'include', 'team', 'keeper')


def resolve(auto, items):
    """People and places from the automatic result and the coach's corrections (in order):
        merge {target, into}         the two fragments are one person (the places are solved with them as one)
        split {target, from: [...]}  the target does not continue / is not continued by those fragments
        confirm {target, slot}       the target is the player who began that place (one person with him)
        exclude / include {target}   not a player / a player after all (an automatic flag lifted; a mixed fragment,
                                     one joining two people, placed after all)
        team {target, team}, keeper {target, on}
        substitution {out, in, status: confirmed | rejected}: confirmed, the incoming player continues the outgoing
                                     one's place (shown sure) and stays a different person
    A correction aimed at a person carries its other fragments in 'also' and applies to all of them. A person's id is
    the number of its earliest fragment among the match's fragments by first sight ('A-P007'): it survives corrections
    that do not move that fragment.
    Returns {'person': {fragment: person id}, 'people': {id: {...}}, 'excluded': {fragment: reason},
    'teams': {fragment: team where it differs from the kit}, 'places': {team: [...]}, 'links': {...},
    'substitutions': [...]}."""
    frag = auto['fragments']
    excluded = {f: v['reason'] for f, v in frag.items() if v.get('flag')}
    team_of = {f: v['team'] for f, v in frag.items()}
    keeper = {f: v['role'] == 'GOALKEEPER' for f, v in frag.items()}
    parent = {}

    def find(x):
        while parent.get(x, x) != x:
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    forbid, confirms, subs_said, structural, included = set(), {}, {}, False, set()
    for c in items or []:
        kind, t = c.get('kind'), c.get('target')
        structural |= kind in PLACE_KINDS
        targets = [x for x in [t]+list(c.get('also') or []) if x in frag]     # a person: every fragment of it
        if kind == 'exclude':
            for x in targets:
                excluded[x] = 'marked by you as not a player'
        elif kind == 'include':
            for x in targets:
                excluded.pop(x, None)
                included.add(x)
        elif kind == 'team' and c.get('team') in TEAMS:
            for x in targets:
                team_of[x] = c['team']
        elif kind == 'keeper':
            for x in targets:
                keeper[x] = bool(c.get('on', True))
        elif kind == 'merge' and c.get('into') in frag:
            for x in targets:
                union(x, c['into'])
        elif kind == 'split' and t in frag:
            for other in c.get('from') or []:
                forbid |= {(other, t), (t, other)}
        elif kind == 'confirm' and isinstance(c.get('slot'), int):
            for x in targets:
                confirms[x] = c['slot']
                included.add(x)                     # the coach's player: placed even when it looked mixed
        elif kind == 'substitution' and c.get('out') in frag and c.get('in') in frag:
            subs_said[(c['out'], c['in'])] = c.get('status', 'confirmed')
            structural |= c.get('status', 'confirmed') == 'confirmed'
    people, places, links, substitutions = {}, {}, {}, []
    person_parent = dict(parent)          # people: the merges, plus the confirmations below

    def pfind(x):
        while person_parent.get(x, x) != x:
            x = person_parent[x]
        return x

    def punion(a, b):
        ra, rb = pfind(a), pfind(b)
        if ra != rb:
            person_parent[ra] = rb
    node_info = {}
    for team in TEAMS:
        data = auto['teams'].get(team) or {}
        members = defaultdict(list)
        for f in data.get('nodes', []):
            if f not in excluded and team_of.get(f) == team and not keeper.get(f) and (not frag[f].get('mixed') or f in included):
                members[find(f)].append(f)
        nodes = {}
        for key, fs in members.items():
            fs.sort(key=lambda f: frag[f]['t0'])
            nodes[key] = {'t0': frag[fs[0]]['t0'], 't1': max(frag[f]['t1'] for f in fs),
                          'seconds': sum(frag[f]['seconds'] or 0 for f in fs), 'frags': fs}
        by_first = {n['frags'][0]: k for k, n in nodes.items()}
        by_last = {max(n['frags'], key=lambda f: frag[f]['t1']): k for k, n in nodes.items()}
        scores = {}
        for a, b, s in data.get('scores', []):
            ka, kb = by_last.get(a), by_first.get(b)
            if ka is None or kb is None or ka == kb or (a, b) in forbid or nodes[kb]['t0'] <= nodes[ka]['t1']:
                continue
            scores[(ka, kb)] = s
        node_of = {f: k for k, n in nodes.items() for f in n['frags']}
        sub_in = set()
        for (o, i), said in subs_said.items():
            ko, ki = node_of.get(o), node_of.get(i)
            if said == 'confirmed' and ko is not None and ki is not None and ko != ki and nodes[ki]['t0'] > nodes[ko]['t1']:
                scores[(ko, ki)] = SUB_LINK
                sub_in.add(ki)
        if structural:
            tracks, conf = solve_places(nodes, scores, K=auto.get('k', OUTFIELD_SLOTS))
        else:
            tracks = [[find(f) for f in tr if find(f) in nodes] for tr in data.get('tracks', [])]
            tracks = [list(dict.fromkeys(tr)) for tr in tracks if tr]
            _, conf = _confidence(tracks, nodes, scores)
        rel = {}
        for k, tr in enumerate(tracks):
            sums = np.zeros(3)
            for f in nodes[tr[0]]['frags']:
                r = frag[f].get('rel')                  # sums of the offsets to the teammates' centroid, and the count
                if r:
                    sums += (r[0], r[1], r[2])
            rel[k] = (sums[0]/sums[2], sums[1]/sums[2]) if sums[2] >= REL_MIN else None
        numbers = number_places(tracks, rel)
        team_places = []
        for k, tr in enumerate(tracks):
            slot = numbers[k]
            starter = tr[0]
            entries = []
            for j, key in enumerate(tr):
                fs = nodes[key]['frags']
                confirmed = j > 0 and any(confirms.get(f) == slot for f in fs)
                sure = j == 0 or confirmed or (key in sub_in)        # the substitute you confirmed holds the slot
                if confirmed:
                    punion(fs[0], nodes[starter]['frags'][0])
                p = None if j == 0 else round(conf.get((tr[j-1], key), 0.0), 3)
                if j > 0:
                    links[f'{nodes[tr[j-1]]["frags"][-1]}>{fs[0]}'] = p
                entries.append({'node': key, 'fragments': fs, 'sure': sure, 'p': p, 't0': nodes[key]['t0'],
                                't1': nodes[key]['t1']})
                node_info[key] = (team, slot, sure)
            team_places.append({'slot': slot, 'nodes': entries})
        team_places.sort(key=lambda p: p['slot'])
        places[team] = team_places
        # keepers: the team's keeper fragments (by goal) never seen together are one person; the coach's keepers join it
        kgroups = [[f for f in g if f not in excluded and keeper.get(f, True) and team_of.get(f) == team]
                   for g in data.get('keepers', [])]
        extra = [f for f, on in keeper.items() if on and team_of.get(f) == team and f not in excluded
                 and frag[f]['role'] != 'GOALKEEPER']
        if extra:
            if kgroups:
                kgroups[0] += extra
            else:
                kgroups.append(extra)
        split_groups = []
        for g in kgroups:
            cuts = sorted(frag[i]['t0'] for (o, i), st in subs_said.items() if st == 'confirmed' and o in g and i in g)
            if not cuts:
                split_groups.append(g)
                continue
            bounds = [-1e18]+cuts+[1e18]
            for a, b in zip(bounds, bounds[1:]):
                part = [f for f in g if a <= (frag[f]['t0'] or 0) < b]
                if part:
                    split_groups.append(part)
        kgroups = split_groups
        # the keeper in goal when: the main keeper first, then each one replacing him (a later group by the coach's
        # substitution) is the keeper too; groups that clash with the main one stay apart ('GK-A?')
        kgroups = sorted([g for g in kgroups if g], key=lambda g: -sum(frag[f]['seconds'] or 0 for f in g))
        replaced = {i for (o, i), st in subs_said.items() if st == 'confirmed'}
        for j, g in enumerate(kgroups):
            for f in g[1:]:
                punion(f, g[0])
            for f in g:
                node_info[f] = (team, 'GK', j == 0 or any(x in replaced for x in g))
    # people: the person of every fragment, numbered per team by first sight
    groups = defaultdict(list)
    for f in frag:
        if f in excluded:
            continue
        groups[pfind(f)].append(f)
    by_team = defaultdict(list)
    for key, fs in groups.items():
        fs.sort(key=lambda f: frag[f]['t0'] if frag[f]['t0'] is not None else 1e12)
        info = [node_info.get(find(f)) or node_info.get(f) for f in fs]
        info = [i for i in info if i]
        team = info[0][0] if info else team_of.get(fs[0])
        by_team[team].append((fs, info))
    out_person = {}
    home = {f: k for k, f in enumerate(sorted(frag, key=lambda f: (frag[f]['t0'] if frag[f]['t0'] is not None else 1e12, f)), 1)}
    for team, plist in by_team.items():
        plist.sort(key=lambda p: frag[p[0][0]]['t0'] if frag[p[0][0]]['t0'] is not None else 1e12)
        for fs, info in plist:
            # a person keeps its id through corrections: the number of its earliest fragment (unique in the match)
            n = min(home[f] for f in fs)
            pid = f'{team}-P{n:03d}' if team in TEAMS else f'U-P{n:03d}'
            sure = [i for i in info if i[2]]
            place = (sure or info or [None])[0]
            if place is None:
                display, slot, provisional = (f'{team}-?' if team in TEAMS else '?'), None, False
            elif place[1] == 'GK':
                display, slot, provisional = f'GK-{team}'+('' if place[2] else '?'), 'GK', not place[2]
            else:
                display, slot, provisional = f'{team}-{place[1]:02d}'+('' if place[2] else '?'), place[1], not place[2]
            people[pid] = {'id': pid, 'team': team, 'fragments': fs, 'display': display, 'slot': slot,
                           'provisional': provisional, 'keeper': slot == 'GK',
                           'how': 'one fragment' if len(fs) == 1 else 'joined automatically' if slot == 'GK' else
                                  'joined by you'}
            mx = next((frag[f]['mixed'] for f in fs if frag[f].get('mixed') and f not in included), None)
            if mx and slot is None:
                where = f"where the segment at {mx['at']} s begins" if mx.get('at') is not None else 'at a segment start'
                what = 'a keeper and an outfield player' if mx['why'] == 'keeper' else 'unlike kits'
                people[pid]['note'] = f'joins two people {where} ({what}): its label is left open'
                people[pid]['mixed'] = mx
            for f in fs:
                out_person[f] = pid
    for team in places:
        for place in places[team]:
            for e in place['nodes']:
                e['person'] = out_person.get(e['fragments'][0])
    for (o, i), status in subs_said.items():
        substitutions.append({'outFragment': o, 'inFragment': i, 'out': out_person.get(o), 'in': out_person.get(i),
                              'status': status})
    return {'person': out_person, 'people': people, 'excluded': excluded,
            'teams': {f: t for f, t in team_of.items() if t in TEAMS and t != frag[f]['kit']},
            'places': places, 'links': links, 'substitutions': substitutions,
            'mixed': {f: v['mixed'] for f, v in frag.items() if v.get('mixed') and f not in included}}


def _confidence(tracks, nodes, scores):
    """Each link's confidence: its score's softmax against the places free when the later node began (covered nodes
    scored to it whose own successor began no earlier)."""
    succ = {tr[i]: tr[i+1] for tr in tracks for i in range(len(tr)-1)}
    covered = {v for tr in tracks for v in tr}
    by_end = defaultdict(dict)
    for (a, b), s in scores.items():
        if a in covered:
            by_end[b][a] = s
    conf = {}
    for tr in tracks:
        for a, b in zip(tr, tr[1:]):
            alts = {q: s for q, s in by_end[b].items() if q == a or q not in succ or nodes[succ[q]]['t0'] >= nodes[b]['t0']}
            if a not in alts:
                conf[(a, b)] = 0.0
                continue
            m = max(alts.values())
            conf[(a, b)] = math.exp(alts[a]-m)/sum(math.exp(v-m) for v in alts.values())
    return tracks, conf


def _group_look(fs1, fs2, frag):
    E1 = [_decode_embedding(frag[f].get('emb')) for f in fs1]
    E2 = [_decode_embedding(frag[f].get('emb')) for f in fs2]
    E1 = [e for e in E1 if e is not None]
    E2 = [e for e in E2 if e is not None]
    if not E1 or not E2:
        return None
    return _look(np.concatenate(E1), np.concatenate(E2))


def corrected_index(index, people, items=None):
    """The assembled index as the page and the analytics read it with the people (resolve()'s result): the roster is
    one entry per person (matchId = the person's id, the slot as its display, its fragments), the alias map takes every
    fragment to its person, the excluded fragments keep their own entries (excluded, with the reason); 'fragments' is
    the stitched roster as assembled (each with its person and place) and 'places' the slots over time, for the roster
    review."""
    frag_roster = {r['matchId']: r for r in index.get('roster') or []}
    place_of = {}
    for team, plist in (people.get('places') or {}).items():
        for place in plist:
            for j, node in enumerate(place['nodes']):
                for f in node['fragments']:
                    place_of[f] = {'team': team, 'slot': place['slot'], 'order': j, 'sure': node['sure'], 'p': node['p']}
    roster = []
    for pid, p in people['people'].items():
        parts = [frag_roster[f] for f in p['fragments'] if f in frag_roster]
        if not parts:
            continue
        base = parts[0]
        firsts = [q['firstSeen'] for q in parts if q.get('firstSeen') is not None]
        lasts = [q['lastSeen'] for q in parts if q.get('lastSeen') is not None]
        f0 = [q['firstFrame'] for q in parts if q.get('firstFrame') is not None]
        f1 = [q['lastFrame'] for q in parts if q.get('lastFrame') is not None]
        weighted = [(q['mean'], float(q.get('seconds') or 0.0)) for q in parts if q.get('mean')]
        wsum = sum(w for _, w in weighted)
        mean = [round(sum(m[0]*w for m, w in weighted)/wsum, 4), round(sum(m[1]*w for m, w in weighted)/wsum, 4)] if wsum else None
        entry = dict(base) | {
            'matchId': pid, 'personId': pid, 'display': p['display'], 'team': p['team'],
            'role': 'GOALKEEPER' if p['keeper'] else 'PLAYER', 'slot': p['slot'], 'provisional': p['provisional'],
            'seconds': round(sum(float(q.get('seconds') or 0.0) for q in parts), 2),
            'firstSeen': min(firsts) if firsts else None, 'lastSeen': max(lasts) if lasts else None,
            'firstFrame': min(f0) if f0 else None, 'lastFrame': max(f1) if f1 else None,
            'members': [m for q in parts for m in q['members']],
            'segments': sorted({s for q in parts for s in q.get('segments') or []}),
            'history': sorted(h for q in parts for h in q.get('history') or []), 'mean': mean,
            'fragments': [q['matchId'] for q in parts], 'mergedFrom': [q['matchId'] for q in parts[1:]],
            'how': p['how'], 'links': [], 'note': p.get('note'), 'mixed': p.get('mixed'),
            'confidence': ('confirmed' if p['how'] == 'joined by you' else 'high' if not p['provisional'] and p['slot'] else
                           'estimate' if p['slot'] else 'unresolved')}
        roster.append(entry)
    for f, reason in people['excluded'].items():
        r = frag_roster.get(f)
        if r is not None:
            roster.append(dict(r) | {'excluded': True, 'excludedReason': reason, 'fragments': [f]})
    roster.sort(key=lambda p: (p.get('firstSeen') is None, p.get('firstSeen') or 0))
    out = dict(index)
    out['roster'] = roster
    out['aliases'] = {f: pid for f, pid in people['person'].items() if pid != f and f not in people['excluded']}
    out['excluded'] = sorted(people['excluded'])
    kit = {r['matchId']: r.get('team') for r in index.get('roster') or []}
    out['teamOverrides'] = {pid: p['team'] for pid, p in people['people'].items()
                            if p['team'] in TEAMS and any(kit.get(f) != p['team'] for f in p['fragments'])}
    out['corrections'] = items or []
    out['fragments'] = [dict(r) | {'person': people['person'].get(r['matchId']), 'place': place_of.get(r['matchId']),
                                    'excludedReason': people['excluded'].get(r['matchId']),
                                    'mixed': (people.get('mixed') or {}).get(r['matchId'])}
                        for r in index.get('roster') or []]
    out['places'] = people.get('places')
    out['people'] = {'version': VERSION, 'count': len(people['people']),
                     'slots': sorted({p['display'] for p in people['people'].values() if p['slot']}),
                     'substitutions': people.get('substitutions') or []}
    return out
