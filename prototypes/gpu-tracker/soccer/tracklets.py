"""Offline tracklet refinement (global tracklet association), run after the causal pass over the whole
segment.

The causal identity manager can leave a real person in pieces: B-02 on one local track, then a new
local track it could not reconnect in time (B-?), or a stretch where the person was ignored as
referee-like (IGNORE, NON_TARGET). And a local track can itself be two people: the tracker handed it
from one to the other when they crossed. With the video known end to end, every local track is cut
where its identity label changes (a mixed tracklet), where the causal pass found a TRACK SWITCH, and
where its clean crops stop looking like the ones before (TRACKLET SPLIT); a fragment after such a
cut is unidentified whatever label it still wears. Every fragment without a confident player
identity is compared with the dormant player identities around it in time: appearance (its own clean
crops against the identity's gallery), camera-compensated image distance from the identity's last
sighting before the fragment and first sighting after it, the time gap, and team compatibility. An
ignored fragment becomes a player again only when its crops clearly look like that player (never on
position alone): NON_TARGET is "not interpreted as a player now", not "deleted". There are no
referee identities to connect anything to. A fragment whose identity was clearly seen elsewhere at
the same time can never be that identity. Connected fragments take the identity's id, team and role
on every frame (so the 2D view keeps the player).

The same local track is itself evidence: an anonymous stretch of a track (the kit undecided on a
small, distant player: UNK-8; or B-? before the identity was decided) that lies within CONTINUE_GAP of
an identified stretch of the same track, with no cut between them, is that identity, whatever OSNet
makes of thirty-pixel crops, unless the crops clearly contradict it. This is what resolves a player
first identified late from the clearer frames: his earlier frames take the identity (future frames
correct earlier labels). Anonymous fragments of UNRESOLVED_MIN or more that stay anonymous are logged
(TRACKLET UNRESOLVED) with the best candidate and why it was not enough.
"""
import bisect
import math
from collections import Counter, defaultdict

import numpy as np
from scipy.optimize import linear_sum_assignment

from .appearance import add_to_gallery, compare, embedding_cosine
from .geometry import clamp, edge_of
from .identity import PLAYER_HEIGHT, ROLE_OUT, combine, long_id, r3

MAX_GAP = 10.0        # seconds between an identity's sighting and a fragment it may continue
MIN_FRAGMENT = .4     # seconds a fragment must last to be considered
ACCEPT = .6           # combined score to connect a fragment to an identity
LEAD = .1             # ... and its lead over the next identity
MIN_APPEARANCE = .5   # appearance similarity below which no connection is made, whatever the position says
MIN_APPEARANCE_IGNORED = .65   # ... for a fragment that was ignored as referee-like: it must clearly look like the player
MOTION_ONLY_GAP = 3.0 # seconds: beyond this gap a fragment without usable crops is not connected on position alone
SPLIT_COSINE = .55    # clean crops on the two sides of a cut are this unlike each other on average (OSNet on small crops:
SPLIT_SIDE = 3        # another kit .36-.5, the same person .75-.95) over up to this many crops a side, each side agreeing
SPLIT_WITHIN = .6     # with itself (every pair inside a side at least this alike)
SCENE_TRACKS = 4      # appearance cuts on this many tracks within SCENE_WINDOW are a scene change (zoom, lighting), not swaps
SCENE_WINDOW = .75
PITCH_MIN = .2        # registration confidence from which pitch positions (metres) replace camera-compensated image distances
ASSIGN_ROUNDS = 5     # rounds of the global assignment of the remaining anonymous fragments
CONTINUE_GAP = 1.0    # s: an anonymous fragment this close to an identified fragment of the SAME local track, with no cut between, is
                      # that identity (the track never changed hands); only crops that contradict (SPLIT_COSINE) stop it
UNRESOLVED_MIN = 1.0  # s: anonymous fragments at least this long that stay anonymous are logged with their best candidate


class _Scorer:
    """What the refinement and the global assignment share: when each identity was observed (its own
    observations), the camera-motion chain, a fragment's own clean crops, and the scoring of an identity for
    a fragment: appearance (crops against the gallery), spatial (the identity's nearest sighting before and
    after the fragment: in metres on the pitch when both carry a registered position, else the
    camera-compensated image distance in body heights), the time gap."""

    def __init__(self, manager, frames, motions, width, height, size=None):
        self.manager, self.frames, self.width, self.height, self.size = manager, frames, width, height, size
        self.frame_times = [f['time'] for f in frames]
        self.times = {pid: sorted(person['time'] for person in g.emitted_all if person.get('id') == pid) for pid, g in manager.registry.items()}
        self.T, self.breaks = _chain(motions)

    def track_state(self, tid):
        return self.manager.ended.get(tid) or self.manager.tracks.get(tid)

    def observed_between(self, pid, t0, t1):
        ts = self.times.get(pid, [])
        k = bisect.bisect_left(ts, t0)
        return k < len(ts) and ts[k] <= t1

    def sighting(self, g, before, t):
        """The identity's own observation nearest to t on the given side: (time, frame index, box, pitch,
        pitch confidence) or None."""
        candidates = [person for person in g.emitted_all if person.get('id') == g.pid and (person['time'] < t if before else person['time'] > t)]
        if not candidates:
            return None
        person = max(candidates, key=lambda q: q['time']) if before else min(candidates, key=lambda q: q['time'])
        i = bisect.bisect_left(self.frame_times, person['time'])
        if i >= len(self.frames) or abs(self.frame_times[i]-person['time']) > 1e-3:
            i = min(range(len(self.frames)), key=lambda k: abs(self.frame_times[k]-person['time']))
        return person['time'], i, person['box'], person.get('pitch'), person.get('pitchConfidence') or 0.0

    def spatial(self, g, i_frag, box_frag, before, pitch_frag=None, conf_frag=0.0):
        s = self.sighting(g, before, self.frame_times[i_frag])
        if s is None:
            return None, None
        t_s, i_s, box_s, pitch_s, conf_s = s
        gap = abs(self.frame_times[i_frag]-t_s)
        if gap > MAX_GAP or any(self.breaks[min(i_s, i_frag)+1:max(i_s, i_frag)+1]):
            return None, gap
        sigma = 3+7*min(gap, 2.0)
        if self.size is not None and pitch_frag is not None and pitch_s is not None and conf_frag >= PITCH_MIN and conf_s >= PITCH_MIN:
            metres = math.hypot((pitch_frag[0]-pitch_s[0])*self.size.length, (pitch_frag[1]-pitch_s[1])*self.size.width)
        else:
            x, y, h = _centre(box_s, self.width, self.height)
            A = self.T[i_frag] @ np.linalg.inv(self.T[i_s])
            p = A @ np.array([x, y, 1.0])
            px, py = p[0]/p[2], p[1]/p[2]
            fx, fy, fh = _centre(box_frag, self.width, self.height)
            metres = math.hypot(px-fx, py-fy)/max((h+fh)/2, 1.0)*PLAYER_HEIGHT
        return math.exp(-.5*(metres/sigma)**2), gap

    def descs_of(self, tid, t0, t1):
        """A fragment's own clean crops: those of its time span from the track's thinned history, else the track's
        last clean crops (a mixed track's end may show another person; the history keeps it apart)."""
        tr = self.track_state(tid)
        history = [d for when, d in getattr(tr, 'clean_history', []) if t0-.1 <= when <= t1+.1] if tr is not None else []
        return history or [d for d, _, occluded in (tr.descs if tr is not None else []) if not occluded] or [d for d, _, _ in (tr.descs if tr is not None else [])]

    def score(self, g, rows, descs, ignored=False):
        """The identity's score for a fragment (rows of (frame index, person)), or None when nothing can be compared."""
        c = compare(g.gallery, descs) if g.gallery and descs else None
        appearance = None if c is None else (c['appearance'] if c['appearance'] is not None else clamp(1-c['total']/.6))
        first, last = rows[0], rows[-1]
        fwd, gap_f = self.spatial(g, first[0], first[1]['box'], True, first[1].get('pitch'), first[1].get('pitchConfidence') or 0.0)
        bwd, gap_b = self.spatial(g, last[0], last[1]['box'], False, last[1].get('pitch'), last[1].get('pitchConfidence') or 0.0)
        if fwd is None and bwd is None:
            return None
        gap = min(x for x in (gap_f, gap_b) if x is not None)
        sp = max(x for x in (fwd, bwd) if x is not None)
        if appearance is None and (gap > MOTION_ONLY_GAP or ignored):
            return None   # position alone says little after a few seconds, and never revives an ignored person
        temporal = clamp(math.exp(-gap/30), .2, 1)
        final = combine({'appearance': appearance, 'uniform': None, 'spatial': sp, 'movement': None, 'temporal': temporal, 'team': True})
        return {'g': g, 'final': final, 'appearance': appearance, 'spatial': sp, 'forward': fwd, 'backward': bwd, 'gap': gap,
                'cosine': c.get('cosine') if c else None}

    def adopt(self, g, rows, old, conf, evidence, track):
        """Relabel a fragment's observations as identity g."""
        for _, person in rows:
            person['firstPassLabel'] = person.get('display')
            if old is not None and person in old.emitted_all:
                old.emitted_all.remove(person)
                old.observations -= 1
            self.manager._style(person, g, conf)
            person['evidence'] = evidence
            g.emitted_all.append(person)
            g.observations += 1
        if track not in g.tracks:
            g.tracks.append(track)
        t0, t1 = self.frames[rows[0][0]]['time'], self.frames[rows[-1][0]]['time']
        self.times[g.pid] = sorted(self.times.get(g.pid, [])+[self.frames[i]['time'] for i, _ in rows])
        if old is not None:
            self.times[old.pid] = [x for x in self.times.get(old.pid, []) if not (t0-1e-6 <= x <= t1+1e-6)]


def _chain(motions):
    T = [np.eye(3)]
    breaks = [False]*len(motions)
    for i in range(1, len(motions)):
        M, reliable, cut = motions[i]
        breaks[i] = bool(cut)
        T.append((np.asarray(M, np.float64) @ T[-1]) if (M is not None and reliable) else T[-1].copy())
    return T, breaks


def _centre(box, width, height):
    return (box[0]+box[2]/2)*width, (box[1]+box[3])*height, max(box[3]*height, 1.0)


def appearance_cuts(history):
    """Times inside a local track where its clean crops stop looking like the ones before: the pairs across
    the cut (up to SPLIT_SIDE crops a side) average below SPLIT_COSINE while each side agrees with itself.
    Returns [(time, mean cosine across the cut)]. history: [(time, descriptor)] in time order."""
    cuts = []
    for i in range(2, len(history)-1):
        before, after = history[max(0, i-SPLIT_SIDE):i], history[i:i+SPLIT_SIDE]
        cross = [c for c in (embedding_cosine(a.embedding, b.embedding) for _, a in before for _, b in after) if c is not None]
        if not cross or float(np.mean(cross)) >= SPLIT_COSINE:
            continue
        within = [c for side in (before, after) for k, (_, a) in enumerate(side) for _, b in side[k+1:]
                  for c in (embedding_cosine(a.embedding, b.embedding),) if c is not None]
        if within and min(within) < SPLIT_WITHIN:
            continue
        cut = (history[i-1][0]+history[i][0])/2
        if not cuts or cut-cuts[-1][0] > 1.0:
            cuts.append((cut, float(np.mean(cross))))
    return cuts


def _build_fragments(manager, frames, motions, width, height, size, report):
    """Every local track cut where its identity label changes, where the causal pass found a TRACK SWITCH and where its
    clean crops stop looking like the ones before (unless many tracks change together: a scene change). Returns the
    scorer, the fragments (track, rows, id, segment, cut, stale) and the splits; counts into the report."""
    by_track = defaultdict(list)
    for i, f in enumerate(frames):
        for p in f.get('people', []):
            if p.get('state') == 'rejected':
                continue
            by_track[p['track']].append((i, p))
    report['tracks'] = len(by_track)
    sc = _Scorer(manager, frames, motions, width, height, size)
    track_state = sc.track_state
    fragments, splits = [], []
    proposed = {}
    for tid in by_track:
        tr = track_state(tid)
        causal = sorted(getattr(tr, 'switches', []) or []) if tr is not None else []
        cuts = [(c, None, 'online') for c in causal]
        for c, cos in appearance_cuts(getattr(tr, 'clean_history', []) if tr is not None else []):
            if all(abs(c-x[0]) > 1.0 for x in cuts):
                cuts.append((c, cos, 'offline'))
        proposed[tid] = cuts
    # Many tracks changing appearance at the same moment is the scene (a zoom, a replay, the light), not swaps.
    offline_times = sorted((c, tid) for tid, cuts in proposed.items() for c, _, how in cuts if how == 'offline')
    scene = set()
    for c, tid in offline_times:
        near = {other for x, other in offline_times if abs(x-c) <= SCENE_WINDOW}
        if len(near) >= SCENE_TRACKS:
            scene.add((tid, c))
    if scene:
        moments = sorted({round(c, 1) for _, c in scene})
        report['events'].append({'time': r3(min(c for _, c in scene)), 'kind': 'tracklet', 'track': None, 'playerId': None,
                                 'message': f"TRACKLET SPLIT SKIPPED\n{len(scene)} tracks changed appearance together around {', '.join(f'{m:.1f}' for m in moments)} s: a scene change, not swaps; no cut made"})
    for tid, rows in by_track.items():
        cuts = sorted(x for x in proposed[tid] if not (x[2] == 'offline' and (tid, x[0]) in scene))
        cut_times = [c[0] for c in cuts]
        seg_of = lambda row: bisect.bisect_right(cut_times, frames[row[0]]['time'])
        key = lambda row: (row[1].get('id'), seg_of(row))
        start = 0
        for k in range(1, len(rows)+1):
            if k == len(rows) or key(rows[k]) != key(rows[start]):
                seg = seg_of(rows[start])
                frag = {'track': tid, 'rows': rows[start:k], 'id': rows[start][1].get('id'), 'segment': seg,
                        'cut': cuts[seg-1] if seg > 0 else None, 'stale': False}
                # A label that continues across a cut is stale: it was earned by the person before the cut.
                if seg > 0 and frag['id'] is not None and fragments and fragments[-1]['track'] == tid and fragments[-1]['id'] == frag['id'] \
                        and fragments[-1]['segment'] == seg-1:
                    frag['stale'] = True
                fragments.append(frag)
                if k < len(rows) and rows[k][1].get('id') != rows[start][1].get('id'):
                    report['mixed'] += 1
                start = k
        for c, cos, how in cuts:
            before = [p for i, p in rows if frames[i]['time'] < c]
            after = [p for i, p in rows if frames[i]['time'] >= c]
            if before and after:
                splits.append({'track': tid, 'time': c, 'cosine': cos, 'how': how, 'before': before, 'after': after, 'outcome': []})
    report['fragments'], report['splits'] = len(fragments), len(splits)
    return sc, fragments, splits


def _continue_fragments(manager, frames, fragments, sc, report, note=None):
    """The same local track is evidence: an anonymous fragment within CONTINUE_GAP of an identified fragment of the same
    track in the same segment (no track switch or appearance cut between) is that identity, whatever OSNet makes of
    small crops, unless the crops contradict it (SPLIT_COSINE), the fragment voted the other team, the identity was seen
    elsewhere meanwhile, or the two sides name different identities (the track changed hands unnoticed). Runs until
    nothing changes, so a continued fragment can carry the identity on to its other neighbour. A refusal is kept on the
    fragment ('continueNote') for the TRACKLET UNRESOLVED log. Returns the number continued."""
    done = 0
    changed = True
    while changed:
        changed = False
        for k, frag in enumerate(fragments):
            if frag['id'] is not None or frag.get('continued'):
                continue
            rows = frag['rows']
            t0, t1 = frames[rows[0][0]]['time'], frames[rows[-1][0]]['time']
            if t1-t0 < MIN_FRAGMENT or any(p.get('state') == 'nontarget' for _, p in rows):
                continue
            found = []
            for side, j in (('before', k-1), ('after', k+1)):
                if j < 0 or j >= len(fragments):
                    continue
                other = fragments[j]
                if other['track'] != frag['track'] or other['segment'] != frag['segment'] or other['id'] is None or other['stale']:
                    continue
                gap = t0-frames[other['rows'][-1][0]]['time'] if side == 'before' else frames[other['rows'][0][0]]['time']-t1
                if gap > CONTINUE_GAP:
                    continue
                g = manager.registry.get(other['id'])
                if g is None or g.role != 'player' or g.team not in ('A', 'B') or g.excluded or g.withdrawn or g.status in ('retired', 'substituted', 'unknown'):
                    continue
                found.append((g, side, gap))
            if not found:
                continue
            if len({g.pid for g, _, _ in found}) > 1:
                frag['continueNote'] = f"the same track carries {found[0][0].pid} before and {found[1][0].pid} after it: it changed hands unnoticed"
                continue
            g, side, gap = found[0]
            descs = sc.descs_of(frag['track'], t0, t1)
            teams_here = Counter(p.get('team') for _, p in rows if p.get('team') in ('A', 'B'))
            vote = teams_here.most_common(1)[0][0] if teams_here else None
            c = compare(g.gallery, descs) if g.gallery and descs else None
            cosine = c.get('cosine') if c else None
            if vote is not None and vote != g.team:
                frag['continueNote'] = f'same track as {g.pid} but voted team {vote}'
            elif sc.observed_between(g.pid, t0+1e-3, t1-1e-3):
                # Strictly inside the fragment's span: the neighbouring fragment's own frames lie a frame outside it.
                frag['continueNote'] = f'same track as {g.pid}, who was seen elsewhere meanwhile'
            elif cosine is not None and cosine < SPLIT_COSINE:
                frag['continueNote'] = f'same track as {g.pid} but the crops contradict it (OSNet cosine {cosine:.2f})'
            if frag.get('continueNote'):
                if note is not None:
                    note(frag, f"{t0:.2f}-{t1:.2f} s: {frag['continueNote']}; not continued")
                continue
            conf = r3(g.identity_confidence or ACCEPT)
            was = f"{rows[0][1].get('display')}"
            states = Counter(p.get('state') for _, p in rows)
            sc.adopt(g, rows, None, conf, 'connected offline', frag['track'])
            frag['id'], frag['continued'] = g.pid, True
            done += 1
            changed = True
            report['connected'] = report.get('connected', 0)+1
            report['sameTrack'] = report.get('sameTrack', 0)+1
            report['framesRestored'] = report.get('framesRestored', 0)+len(rows)
            lines = ['TRACKLET CONNECT', f"Tracklet {frag['track']} ({t0:.2f}-{t1:.2f} s, {len(rows)} frames, first pass: {was}) -> {long_id(g.pid)}",
                     f'Global Identity: {g.pid} ({ROLE_OUT[g.role]}, team {g.team})',
                     f"Same local track as {g.pid} {side} it ({gap:.2f} s apart), no track switch or appearance cut between: the track never changed hands",
                     'Appearance similarity: ' + ('not usable (small or dirty crops); not required here' if cosine is None else f'OSNet cosine {cosine:.2f} (no contradiction)'),
                     f"Team consistency: {'yes' if vote == g.team else 'no team vote (kit undecided on these frames)'}",
                     f"First pass states: {', '.join(f'{k} {v}' for k, v in states.most_common())}",
                     f'Final: continued; identity confidence {conf:.2f}']
            report.setdefault('events', []).append({'time': r3(t0), 'kind': 'tracklet', 'track': frag['track'], 'playerId': g.pid, 'message': '\n'.join(lines)})
            report.setdefault('connections', []).append({'track': frag['track'], 'start': r3(t0), 'end': r3(t1), 'to': g.pid, 'from': None, 'final': conf, 'sameTrack': True})
            if note is not None and frag['cut'] is not None:
                note(frag, f"{t0:.2f}-{t1:.2f} s -> {g.pid} (same track)")
    return done


def continue_same_track(manager, frames, motions, width, height, size=None):
    """The same-track continuation alone, for a second pass after the global assignment: a fragment identified there
    hands its identity to the anonymous stretches of its own track. Returns {'connected', 'sameTrack', 'framesRestored',
    'events', 'connections'}."""
    report = {'connected': 0, 'sameTrack': 0, 'framesRestored': 0, 'events': [], 'connections': []}
    if not frames or len(motions) != len(frames):
        return report
    scratch = {'tracks': 0, 'fragments': 0, 'mixed': 0, 'splits': 0, 'events': []}
    sc, fragments, _ = _build_fragments(manager, frames, motions, width, height, size, scratch)
    _continue_fragments(manager, frames, fragments, sc, report, None)
    report['events'].sort(key=lambda e: e['time'])
    return report


def refine(manager, frames, motions, width, height, size=None):
    """Split and connect. manager: the IdentityManager after the causal pass (its registry, galleries and
    ended tracks); frames: the per-frame observations (people dicts are relabelled in place); motions: per
    frame (M pixels previous->this or None, reliable, cut); size: the pitch size, when the frames carry
    registered pitch positions (spatial evidence in metres). Returns a report with the counts and events."""
    report = {'tracks': 0, 'fragments': 0, 'mixed': 0, 'splits': 0, 'connected': 0, 'ignoredRestored': 0,
              'stripped': 0, 'framesRestored': 0, 'sameTrack': 0, 'unresolved': 0, 'events': [], 'connections': []}
    if not frames or len(motions) != len(frames):
        return report
    sc, fragments, splits = _build_fragments(manager, frames, motions, width, height, size, report)
    times, observed_between = sc.times, sc.observed_between

    def unknown(person, why):
        person['firstPassLabel'] = person.get('display')
        person.update(id=None, display='?', label='IDENTITY_UNCERTAIN', team=None, role='UNKNOWN', state='uncertain', identityConfidence=0.0,
                      evidence='split offline', reason=why)
        for key in ('stableRole', 'memory'):
            person.pop(key, None)

    def note(frag, text):
        for sp in splits:
            if sp['track'] == frag['track'] and abs(sp['time']-frag['cut'][0]) < 1e-6:
                sp['outcome'].append(text)

    # The track itself first: it outranks an appearance match of another track to the same identity.
    _continue_fragments(manager, frames, fragments, sc, report, note)

    for frag in fragments:
        rows = frag['rows']
        t0, t1 = frames[rows[0][0]]['time'], frames[rows[-1][0]]['time']
        if t1-t0 < MIN_FRAGMENT:
            if frag['cut'] is not None:
                note(frag, f'{t0:.2f}-{t1:.2f} s: too short to compare')
            continue
        pid = frag['id']
        if frag.get('continued'):
            continue
        if pid is not None and not frag['stale']:
            if frag['cut'] is not None:
                note(frag, f'{t0:.2f}-{t1:.2f} s: {pid}, identified during the run after the cut; kept')
            continue
        descs = sc.descs_of(frag['track'], t0, t1)
        # Labels earned before a cut say nothing about the person after it: no team vote then. (After a
        # TRACK SWITCH the causal pass restarted its evidence: those labels are fresh.)
        after_cut = frag['segment'] > 0 and not (frag['cut'] is not None and frag['cut'][2] == 'online' and not frag['stale'])
        teams = Counter(p.get('team') for _, p in rows if p.get('team') in ('A', 'B')) if not after_cut else Counter()
        team = teams.most_common(1)[0][0] if teams else None
        # Ignored as referee-like on some frames: a player again only if the crops clearly look like one.
        ignored = any(p.get('state') == 'nontarget' for _, p in rows)
        old = manager.registry.get(pid) if pid is not None else None
        scored = []
        for g in manager.registry.values():
            if g.status in ('retired', 'substituted', 'unknown') or g.excluded or g.withdrawn or g.pid == pid:
                continue
            if g.role != 'player' or g.team not in ('A', 'B') or (team is not None and g.team != team):
                continue
            if observed_between(g.pid, t0-.05, t1+.05):
                continue
            s = sc.score(g, rows, descs, ignored)
            if s is not None:
                scored.append(s)
        scored.sort(key=lambda s: -s['final'])
        best, second = (scored[0] if scored else None), (scored[1] if len(scored) > 1 else None)
        floor = MIN_APPEARANCE_IGNORED if ignored else MIN_APPEARANCE
        if best is None or best['final'] < ACCEPT or (best['appearance'] is not None and best['appearance'] < floor) or \
                (second is not None and best['final']-second['final'] < LEAD):
            if frag['stale'] and old is not None:
                # The label after the cut is the earlier person's. It stays only if these crops still look like
                # that identity; otherwise the fragment is shown as unidentified rather than as the wrong person.
                c = compare(old.gallery, descs) if old.gallery and descs else None
                if c is not None and c.get('cosine') is not None and c['cosine'] < SPLIT_COSINE:
                    for _, person in rows:
                        if person in old.emitted_all:
                            old.emitted_all.remove(person)
                            old.observations -= 1
                        unknown(person, f"the crops after the cut do not look like {old.pid} (OSNet cosine {c['cosine']:.2f})")
                    times[old.pid] = [x for x in times.get(old.pid, []) if not (t0-1e-6 <= x <= t1+1e-6)]
                    report['stripped'] += 1
                    note(frag, f"{t0:.2f}-{t1:.2f} s: {old.pid} removed, shown as ? (cosine {c['cosine']:.2f} to {old.pid}, no dormant identity matched)")
                else:
                    note(frag, f"{t0:.2f}-{t1:.2f} s: kept {old.pid}" + (f" (cosine {c['cosine']:.2f})" if c is not None and c.get('cosine') is not None else ''))
            elif frag['cut'] is not None:
                # A team label (B-?) after a cut was earned by the person before it: unidentified, not B-?.
                stale_team = [person for _, person in rows if person.get('state') == 'uncertain' and person.get('team') and after_cut]
                for person in stale_team:
                    unknown(person, 'the team label was earned before the cut')
                note(frag, f"{t0:.2f}-{t1:.2f} s: no dormant player matched; " + ('shown as ?' if stale_team else 'label kept') +
                     (f" (best {best['g'].pid} {best['final']:.2f})" if best else ''))
            if pid is None and t1-t0 >= UNRESOLVED_MIN:
                report['unresolved'] += 1
                why = ('no dormant identity of the team to compare with' if best is None else
                       f"best {best['g'].pid} {best['final']:.2f}" + (f" (needs {ACCEPT:.2f})" if best['final'] < ACCEPT else '') +
                       (f", appearance {best['appearance']:.2f} under the floor {floor:.2f}" if best['appearance'] is not None and best['appearance'] < floor else '') +
                       (f", lead over {second['g'].pid} {best['final']-second['final']:.2f} under {LEAD:.2f}" if second is not None and best['final']-second['final'] < LEAD else ''))
                if frag.get('continueNote'):
                    why += f"; {frag['continueNote']}"
                states = Counter(p.get('state') for _, p in rows)
                lines = ['TRACKLET UNRESOLVED', f"Tracklet {frag['track']} ({t0:.2f}-{t1:.2f} s, {len(rows)} frames, first pass: {rows[0][1].get('display')}) stays without an identity",
                         f"Team vote: {team or 'none (kit undecided)'}; first pass states: {', '.join(f'{k} {v}' for k, v in states.most_common())}",
                         f'Why: {why}', 'Decision: left as it is (no identity is forced; the global assignment may still place it)']
                report['events'].append({'time': r3(t0), 'kind': 'tracklet', 'track': frag['track'], 'playerId': None, 'message': '\n'.join(lines)})
            continue
        g = best['g']
        conf = r3(min(best['final'], g.identity_confidence or best['final']))
        sc.adopt(g, rows, old, conf, 'connected offline', frag['track'])
        report['connected'] += 1
        report['framesRestored'] += len(rows)
        if ignored:
            report['ignoredRestored'] += 1
        was = f"{rows[0][1]['firstPassLabel']}" + (f' ({pid})' if pid else '')
        lines = ['TRACKLET CONNECT', f"Tracklet {frag['track']} ({t0:.2f}-{t1:.2f} s, {len(rows)} frames, first pass: {was}) -> {long_id(g.pid)}",
                 f'Global Identity: {g.pid} ({ROLE_OUT[g.role]}' + (f', team {g.team})' if g.team else ')'),
                 'Appearance similarity: ' + ('n/a' if best['appearance'] is None else f"{best['appearance']:.2f}" + (f" (OSNet cosine {best['cosine']:.2f})" if best['cosine'] is not None else '')),
                 'Motion consistency: ' + ', '.join(x for x in (f"before {best['forward']:.2f}" if best['forward'] is not None else '', f"after {best['backward']:.2f}" if best['backward'] is not None else '') if x),
                 f"Team consistency: {'yes' if team == g.team else 'no team vote'}", f"Time gap: {best['gap']:.2f} s",
                 f"Final: {best['final']:.2f}" + (f" (next {second['g'].pid} {second['final']:.2f})" if second else '')]
        if ignored:
            lines.append(f'Was ignored as referee-like on {sum(1 for _, p in rows if p.get("firstPassLabel") == "IGNORE")} frames: the crops clearly show {g.pid}')
        if frag['stale']:
            lines.append(f"{old.pid}'s label on these frames was earned before the cut and is dropped")
        report['events'].append({'time': r3(t0), 'kind': 'tracklet', 'track': frag['track'], 'playerId': g.pid, 'message': '\n'.join(lines)})
        report['connections'].append({'track': frag['track'], 'start': r3(t0), 'end': r3(t1), 'to': g.pid, 'from': pid, 'final': r3(best['final'])})
        if frag['cut'] is not None:
            note(frag, f"{t0:.2f}-{t1:.2f} s -> {g.pid}")
    for sp in splits:
        first_before, last_before, first_after = sp['before'][0], sp['before'][-1], sp['after'][0]
        label = lambda p: p.get('firstPassLabel') or p.get('display')
        how = 'found during the run (TRACK SWITCH: the crops stopped looking like the track\'s own first crops)' if sp['how'] == 'online' else \
            f"the clean crops after it do not show the person before it (mean OSNet cosine {sp['cosine']:.2f} across the cut)"
        lines = ['TRACKLET SPLIT', f"Local track {sp['track']} cut at {sp['time']:.2f} s: {how}",
                 f"Before the cut: {label(first_before)}" + (f" -> {label(last_before)}" if label(last_before) != label(first_before) else '') +
                 f" ({first_before['time']:.2f}-{last_before['time']:.2f} s); after it: {label(first_after)} ({first_after['time']:.2f}-{sp['after'][-1]['time']:.2f} s)"]
        lines += ['After the cut: '+x for x in sp['outcome']] or ['After the cut: nothing to compare']
        report['events'].append({'time': r3(sp['time']), 'kind': 'tracklet', 'track': sp['track'], 'playerId': None, 'message': '\n'.join(lines)})
    report['events'].sort(key=lambda e: e['time'])
    return report


def assign_fragments(manager, frames, motions, width, height, size=None):
    """Global assignment of the fragments still without an identity after refine(): runs of a local track
    in state 'uncertain' (no id) at least MIN_FRAGMENT long, against the team players not observed during
    the fragment, scored like the refinement (appearance, spatial in metres where registered, time gap).
    Fragments and identities are matched one-to-one per round (Hungarian), and a pair is accepted only at
    ACCEPT with the appearance floor, with the usual motion-only limit, and with a lead of LEAD over the
    fragment's next-best identity AND over the identity's next-best fragment: two teammates returning
    together take two identities or none, never the same one twice. Never creates an identity. Returns the
    report: counts, events, assignments."""
    report = {'proposed': 0, 'accepted': 0, 'rejectedByLead': 0, 'rejectedByScore': 0, 'rounds': 0, 'framesAssigned': 0, 'events': [], 'assignments': []}
    if not frames or len(motions) != len(frames):
        return report
    by_track = defaultdict(list)
    for i, f in enumerate(frames):
        for p in f.get('people', []):
            if p.get('state') != 'rejected':
                by_track[p['track']].append((i, p))
    fragments = []
    for tid, rows in by_track.items():
        start = None
        for k in range(len(rows)+1):
            p = rows[k][1] if k < len(rows) else None
            anonymous = p is not None and p.get('id') is None and p.get('state') == 'uncertain'
            if anonymous and start is None:
                start = k
            if not anonymous and start is not None:
                seg = rows[start:k]
                t0, t1 = frames[seg[0][0]]['time'], frames[seg[-1][0]]['time']
                if t1-t0 >= MIN_FRAGMENT:
                    teams = Counter(q.get('team') for _, q in seg if q.get('team') in ('A', 'B'))
                    fragments.append({'track': tid, 'rows': seg, 't0': t0, 't1': t1, 'team': teams.most_common(1)[0][0] if teams else None, 'done': False})
                start = None
    if not fragments:
        return report
    sc = _Scorer(manager, frames, motions, width, height, size)
    candidates = [g for g in manager.registry.values() if g.role == 'player' and g.team in ('A', 'B') and g.gallery
                  and g.status not in ('retired', 'substituted', 'unknown') and not g.excluded and not g.withdrawn]
    if not candidates:
        return report
    for _ in range(ASSIGN_ROUNDS):
        open_frags = [fr for fr in fragments if not fr['done']]
        if not open_frags:
            break
        report['rounds'] += 1
        S = np.zeros((len(open_frags), len(candidates)))
        info = {}
        for fi, frag in enumerate(open_frags):
            descs = sc.descs_of(frag['track'], frag['t0'], frag['t1'])
            for gi, g in enumerate(candidates):
                if frag['team'] is not None and g.team != frag['team']:
                    continue
                if sc.observed_between(g.pid, frag['t0']-.05, frag['t1']+.05):
                    continue
                s = sc.score(g, frag['rows'], descs)
                if s is None:
                    continue
                if s['appearance'] is not None and s['appearance'] < MIN_APPEARANCE:
                    continue
                S[fi, gi] = s['final']
                info[(fi, gi)] = s
        if not S.any():
            break
        rows_i, cols_i = linear_sum_assignment(-S)
        progress = False
        for fi, gi in zip(rows_i, cols_i):
            value = S[fi, gi]
            if value <= 0:
                continue
            report['proposed'] += 1
            if value < ACCEPT:
                report['rejectedByScore'] += 1
                continue
            row_second = max((S[fi, j] for j in range(S.shape[1]) if j != gi), default=0.0)
            col_second = max((S[i, gi] for i in range(S.shape[0]) if i != fi), default=0.0)
            if value-max(row_second, col_second) < LEAD:
                report['rejectedByLead'] += 1
                continue
            frag, s, g = open_frags[fi], info[(fi, gi)], candidates[gi]
            conf = r3(min(value, g.identity_confidence or value))
            was = frag['rows'][0][1].get('display')
            sc.adopt(g, frag['rows'], None, conf, 'assigned offline', frag['track'])
            frag['done'] = True
            progress = True
            report['accepted'] += 1
            report['framesAssigned'] += len(frag['rows'])
            rival = f" (next {candidates[int(np.argsort(-S[fi])[1])].pid} {row_second:.2f})" if row_second > 0 else ''
            lines = ['TRACKLET ASSIGNED', f"Tracklet {frag['track']} ({frag['t0']:.2f}-{frag['t1']:.2f} s, {len(frag['rows'])} frames, first pass: {was}) -> {long_id(g.pid)}",
                     f'Global Identity: {g.pid} ({ROLE_OUT[g.role]}, team {g.team})',
                     'Appearance similarity: ' + ('n/a' if s['appearance'] is None else f"{s['appearance']:.2f}" + (f" (OSNet cosine {s['cosine']:.2f})" if s['cosine'] is not None else '')),
                     'Motion consistency: ' + ', '.join(x for x in (f"before {s['forward']:.2f}" if s['forward'] is not None else '', f"after {s['backward']:.2f}" if s['backward'] is not None else '') if x),
                     f"Time gap: {s['gap']:.2f} s", f"Final: {value:.2f}{rival}; no other fragment fits {g.pid} within {LEAD:.2f} (best other {col_second:.2f})",
                     'Assigned jointly with the other fragments of the segment (one identity per fragment, one fragment per identity at a time)']
            report['events'].append({'time': r3(frag['t0']), 'kind': 'tracklet', 'track': frag['track'], 'playerId': g.pid, 'message': '\n'.join(lines)})
            report['assignments'].append({'track': frag['track'], 'start': r3(frag['t0']), 'end': r3(frag['t1']), 'to': g.pid, 'final': r3(value)})
        if not progress:
            break
    report['events'].sort(key=lambda e: e['time'])
    return report


MERGE_APPEARANCE = .7    # mutual gallery similarity two identities need before a merge is proposed
MERGE_MAX_SPEED = 8.0    # m/s: the move from the last sighting of one to the first of the other must be plausible
MERGE_MIN_GAP = .5       # s: the gap used for that speed is never shorter than this
MERGE_APPLY_GAP = 3.0    # s: a later identity that begins this soon after the earlier one vanished inside the frame (not through an
                         # edge), continuous and mutually most similar, is the same player hidden for a moment: merged into the earlier
                         # one (apply_merges); a longer gap, an exit, or an identity you set stays a proposal for review


def merge_proposals(manager, frames, motions, width, height, size=None):
    """Pairs of identities that may be one person, proposed for review and never applied: same team, both
    players with galleries, never observed on the same frame, one observed entirely before the other, galleries
    mutually the most similar at MERGE_APPEARANCE or above, and a plausible continuity between the last
    sighting of the earlier one and the first of the later one (pitch speed at most MERGE_MAX_SPEED when both
    are registered; else the camera-compensated image distance scored like a fragment). Returns
    {'proposals': [...], 'pairsCompared', 'rejectedByAppearance', 'rejectedByContinuity', 'events'}."""
    report = {'proposals': [], 'pairsCompared': 0, 'rejectedByAppearance': 0, 'rejectedByContinuity': 0, 'interleaved': 0, 'events': []}
    if not frames or len(motions) != len(frames):
        return report
    sc = _Scorer(manager, frames, motions, width, height, size)
    ids = [g for g in manager.registry.values() if g.role == 'player' and g.team in ('A', 'B') and g.gallery
           and g.status != 'retired' and not g.excluded and not g.withdrawn and sc.times.get(g.pid)]
    similarity = {}
    for i, a in enumerate(ids):
        for b in ids[i+1:]:
            if a.team != b.team:
                continue
            ta, tb = sc.times[a.pid], sc.times[b.pid]
            if ta[-1] >= tb[0] and tb[-1] >= ta[0]:
                report['interleaved'] += 1   # overlapping spans: two people seen in turn, or one person the manager flip-flopped on; not proposed
                continue
            report['pairsCompared'] += 1
            ab = compare(a.gallery, [s.d for s in b.gallery])
            ba = compare(b.gallery, [s.d for s in a.gallery])
            values = [c['appearance'] for c in (ab, ba) if c is not None and c.get('appearance') is not None]
            if not values:
                continue
            similarity[(a.pid, b.pid)] = similarity[(b.pid, a.pid)] = min(values)
    best = {}
    for (pa, pb), v in similarity.items():
        if v > best.get(pa, (None, -1))[1]:
            best[pa] = (pb, v)
    seen = set()
    for pa, (pb, v) in best.items():
        if best.get(pb, (None, -1))[0] != pa or (pb, pa) in seen:
            continue
        seen.add((pa, pb))
        if v < MERGE_APPEARANCE:
            report['rejectedByAppearance'] += 1
            continue
        a, b = manager.registry[pa], manager.registry[pb]
        if sc.times[a.pid][0] > sc.times[b.pid][0]:
            a, b = b, a
        last_a = sc.sighting(a, True, sc.times[b.pid][0])
        first_b = sc.sighting(b, False, sc.times[a.pid][-1])
        gap = max(MERGE_MIN_GAP, first_b[0]-last_a[0])
        speed = spatial = None
        if size is not None and last_a[3] is not None and first_b[3] is not None and last_a[4] >= PITCH_MIN and first_b[4] >= PITCH_MIN:
            metres = math.hypot((first_b[3][0]-last_a[3][0])*size.length, (first_b[3][1]-last_a[3][1])*size.width)
            speed = metres/gap
            plausible = speed <= MERGE_MAX_SPEED
        else:
            spatial, _ = sc.spatial(a, first_b[1], first_b[2], True, first_b[3], first_b[4])
            plausible = spatial is None or spatial >= .1
        proposal = {'earlier': a.pid, 'later': b.pid, 'team': a.team, 'appearance': r3(v), 'gap': r3(first_b[0]-last_a[0]),
                    'speed': None if speed is None else r3(speed), 'imageSpatial': None if spatial is None else r3(spatial),
                    'continuity': 'pitch' if speed is not None else ('image' if spatial is not None else 'unmeasured'), 'plausible': bool(plausible),
                    'vanishedInside': not edge_of(last_a[2]), 'at': r3(first_b[0]), 'applied': False}
        if not plausible:
            report['rejectedByContinuity'] += 1
            continue
        report['proposals'].append(proposal)
        how = f"{speed:.1f} m/s on the pitch" if speed is not None else (f"image continuity {spatial:.2f}" if spatial is not None else 'continuity unmeasured')
        report['events'].append({'time': r3(first_b[0]), 'kind': 'merge-proposal', 'playerId': b.pid,
                                 'message': f"MERGE PROPOSED (not applied)\n{long_id(a.pid)} (last seen {last_a[0]:.2f} s) and {long_id(b.pid)} (first seen {first_b[0]:.2f} s) may be one person: "
                                            f"galleries mutually most similar at {v:.2f}, never seen together, {how}. Review it in the page; nothing is merged automatically."})
    report['proposals'].sort(key=lambda x: -x['appearance'])
    return report


def apply_merges(manager, proposals, time):
    """Apply the proposals that describe a short occlusion: the later identity began within MERGE_APPLY_GAP of the
    earlier one's last sighting, which was inside the frame (an occlusion or a missed detection, not an exit), the
    move between them is plausible and measured, neither identity is one you set, and the later one was created in
    this run. The later identity is retired into the earlier one: every observation labelled with it takes the
    earlier identity (evidence 'merged offline', the first-pass label kept), its gallery samples join the earlier
    gallery, its local tracks are recorded on the earlier identity. The other proposals stay for review. Returns
    {'merged', 'events'}."""
    report = {'merged': 0, 'events': []}
    for p in proposals:
        a, b = manager.registry.get(p['earlier']), manager.registry.get(p['later'])
        if a is None or b is None or not p['plausible'] or p['continuity'] == 'unmeasured' or not p.get('vanishedInside') or p['gap'] > MERGE_APPLY_GAP:
            continue
        if a.manual or b.manual or b.restored or b.created_on is None or a.status in ('retired', 'substituted') or b.status in ('retired', 'substituted'):
            continue
        moved = 0
        for person in list(b.emitted_all):
            if person.get('id') != b.pid:
                continue
            person['firstPassLabel'] = person.get('firstPassLabel') or person.get('display')
            manager._style(person, a, min(person.get('identityConfidence', 1), a.identity_confidence or 1))
            person['evidence'] = 'merged offline'
            a.emitted_all.append(person)
            moved += 1
        b.emitted_all, b.observations = [], 0
        a.observations += moved
        for s in b.gallery:
            a.gallery = add_to_gallery(a.gallery, s.d, s.time, manager.o.gallery_size)
        a.tracks += [t for t in b.tracks if t not in a.tracks]
        if b.last_seen > a.last_seen:
            a.last_seen, a.last_box, a.last_pitch, a.exit_edge, a.status = b.last_seen, list(b.last_box), b.last_pitch, b.exit_edge, b.status
            a.stab, a.velocity, a.pitch_velocity, a.segment, a.drift, a.clean_seen = b.stab, b.velocity, b.pitch_velocity, b.segment, b.drift, b.clean_seen
            a.track, a.here = b.track, b.here
            live = manager.tracks.get(b.track) if b.track is not None else None
            if live is not None and live.player_id == b.pid:
                live.player_id = a.pid
        a.kit_clear, a.kit_total = a.kit_clear+b.kit_clear, a.kit_total+b.kit_total
        a.history.extend(b.history)
        b.status, b.retired_into, b.track, b.here = 'retired', a.pid, None, False
        b.role_history.append((r3(time), 'retired', f'was {a.pid}: the same player, hidden for {p["gap"]:.1f} s'))
        p['applied'] = True
        report['merged'] += 1
        how = f"{p['speed']:.1f} m/s on the pitch" if p['speed'] is not None else f"image continuity {p['imageSpatial']:.2f}"
        lines = ['MERGE APPLIED', f"{long_id(b.pid)} is {long_id(a.pid)}: created {p['gap']:.2f} s after {a.pid} vanished inside the frame, galleries mutually "
                 f"most similar at {p['appearance']:.2f}, never seen together, {how}.",
                 f"{moved} observations of {b.pid} relabelled {a.pid} (first-pass label kept); {b.pid} retired."]
        report['events'].append({'time': p.get('at', r3(time)), 'kind': 'merge', 'track': None, 'playerId': a.pid, 'message': '\n'.join(lines)})
    return report


def metrics(manager, frames, events, report):
    """Identity metrics of a run: fragmentation, what happened to confirmed players next to ignored people,
    the ignored (non-target) people themselves, and what the offline pass restored."""
    ids = [g for g in manager.registry.values() if not g.withdrawn]
    switches = sum(max(0, len(g.tracks)-1) for g in ids)
    role_events = [e for e in events if e['kind'] == 'role']
    ignored_updates = sum(1 for e in role_events if e['message'].startswith('ROLE UPDATE IGNORED'))
    deferred = sum(1 for e in role_events if e['message'].startswith('ROLE UPDATE DEFERRED'))
    withdrawn = [e for e in events if e['kind'] == 'nontarget' and e['message'].startswith('PLAYER IDENTITY WITHDRAWN')]
    online_switches = sum(1 for e in events if e['kind'] == 'switch')
    # A confirmed player's local track that continues as an ignored person (final labels), and whether a
    # track switch or a released identity (the local track moved to someone else) explains it.
    explained = defaultdict(list)
    for e in events:
        if e['kind'] == 'switch' or (e['kind'] == 'reid-rejected' and 'no longer' in e['message']):
            explained[e['track']].append(e['time'])
    by_track = defaultdict(list)
    nontarget_frames, nontarget_tracks, hints = 0, set(), Counter()
    for f in frames:
        for p in f.get('people', []):
            by_track[p['track']].append((f['time'], p))
            if p.get('state') == 'nontarget':
                nontarget_frames += 1
                nontarget_tracks.add(p['track'])
                hints[p.get('hint') or 'REFEREE_LIKE'] += 1
    to_nontarget = unexplained = 0
    for tid, rows in by_track.items():
        last = None
        for t, p in rows:
            if p.get('id') and p.get('role') in ('PLAYER', 'GOALKEEPER'):
                last = t
            elif p.get('state') == 'nontarget' and last is not None:
                to_nontarget += 1
                if not any(last-.5 <= x <= t+.5 for x in explained.get(tid, [])):
                    unexplained += 1
                last = None
    encounters = list(manager.encounters.values())
    preserved = reid_after = 0
    for e in encounters:
        player = manager.registry.get(e['ids'][0])
        if player is not None and player.status != 'retired' and any(person['time'] >= e['start']+2.0 for person in player.emitted_all if person.get('id') == player.pid):
            preserved += 1
        if any(ev['kind'] == 'reid' and ev.get('playerId') == e['ids'][0] and e['start'] <= ev['time'] <= e['start']+10.0 for ev in events):
            reid_after += 1
    lost_after = 0
    for g in ids:
        if g.role != 'player' or g.status == 'retired':
            continue
        mine = sorted(person['time'] for person in g.emitted_all if person.get('id') == g.pid)
        for e in encounters:
            if g.pid in e['ids'] and mine and mine[-1] < e['start']+3.0:
                lost_after += 1
                break
    shown = Counter()
    for f in frames:
        for p in f.get('people', []):
            if p.get('evidence') == 'connected offline':
                shown[p.get('firstPassLabel') or '?'] += 1
    players = [g for g in ids if g.role == 'player' and g.status not in ('retired', 'substituted')]
    over = {team: n for team in ('A', 'B') for n in [sum(1 for g in players if g.team == team)] if n > manager.o.max_per_team}
    return {'localTracks': report.get('tracks', 0), 'fragments': report.get('fragments', 0), 'mixedTracklets': report.get('mixed', 0),
            'trackSwitchesFoundOnline': online_switches, 'trackletsSplit': report.get('splits', 0), 'fragmentsShownAsUnknown': report.get('stripped', 0),
            'idSwitches': switches, 'globalIdFragmentation': round(switches/max(1, sum(1 for g in ids if g.tracks)), 2),
            'playerTrackToNonTarget': to_nontarget, 'playerToNonTargetUnexplained': unexplained,
            'refereeIdentities': sum(1 for g in manager.registry.values() if g.role == 'referee'),
            'nonTargetTracks': len(nontarget_tracks), 'nonTargetFrames': nontarget_frames, 'nonTargetHints': dict(hints),
            'playerIdentitiesWithdrawn': len(withdrawn), 'roleUpdatesIgnored': ignored_updates, 'roleUpdatesDeferred': deferred,
            'closeEncounters': len(encounters), 'playersPreservedThroughEncounters': preserved, 'playersLostAfterEncounter': lost_after,
            'playerReidentifiedAfterEncounter': reid_after,
            'newPlayerIdentities': sum(1 for e in events if e['kind'] == 'new-identity' and e.get('playerId', '')[:2] in ('A-', 'B-')),
            'teamsOverCap': over,
            'trackletsConnected': report.get('connected', 0), 'ignoredFragmentsRestored': report.get('ignoredRestored', 0),
            'sameTrackContinued': report.get('sameTrack', 0), 'unresolvedFragments': report.get('unresolved', 0),
            'framesRestoredToPlayers': report.get('framesRestored', 0), 'restoredFromLabels': dict(shown),
            'assignedOffline': report.get('assigned', 0), 'framesAssignedOffline': report.get('framesAssigned', 0),
            'assignment': report.get('assignment'), 'mergedOffline': report.get('merged', 0)}
