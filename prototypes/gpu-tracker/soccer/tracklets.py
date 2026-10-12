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
"""
import bisect
import math
from collections import Counter, defaultdict

import numpy as np

from .appearance import compare, embedding_cosine
from .geometry import clamp
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


def refine(manager, frames, motions, width, height):
    """Split and connect. manager: the IdentityManager after the causal pass (its registry, galleries and
    ended tracks); frames: the per-frame observations (people dicts are relabelled in place); motions: per
    frame (M pixels previous->this or None, reliable, cut). Returns a report with the counts and events."""
    report = {'tracks': 0, 'fragments': 0, 'mixed': 0, 'splits': 0, 'connected': 0, 'ignoredRestored': 0,
              'stripped': 0, 'framesRestored': 0, 'events': [], 'connections': []}
    if not frames or len(motions) != len(frames):
        return report
    by_track = defaultdict(list)
    for i, f in enumerate(frames):
        for p in f.get('people', []):
            if p.get('state') == 'rejected':
                continue
            by_track[p['track']].append((i, p))
    report['tracks'] = len(by_track)
    frame_times = [f['time'] for f in frames]

    def track_state(tid):
        return manager.ended.get(tid) or manager.tracks.get(tid)

    # ---- cuts: identity changes, TRACK SWITCH times from the causal pass, appearance changes in the clean crops
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
    # When each identity was observed (its own observations only), for the time-overlap test.
    times = {}
    for pid, g in manager.registry.items():
        times[pid] = sorted(person['time'] for person in g.emitted_all if person.get('id') == pid)
    T, breaks = _chain(motions)

    def observed_between(pid, t0, t1):
        ts = times.get(pid, [])
        k = bisect.bisect_left(ts, t0)
        return k < len(ts) and ts[k] <= t1

    def sighting(g, before, t):
        """The identity's own observation nearest to t on the given side: (time, frame index, box) or None."""
        candidates = [person for person in g.emitted_all if person.get('id') == g.pid and (person['time'] < t if before else person['time'] > t)]
        if not candidates:
            return None
        person = max(candidates, key=lambda q: q['time']) if before else min(candidates, key=lambda q: q['time'])
        i = bisect.bisect_left(frame_times, person['time'])
        if i >= len(frames) or abs(frame_times[i]-person['time']) > 1e-3:
            i = min(range(len(frames)), key=lambda k: abs(frame_times[k]-person['time']))
        return person['time'], i, person['box']

    def spatial(g, i_frag, box_frag, before):
        s = sighting(g, before, frame_times[i_frag])
        if s is None:
            return None, None
        t_s, i_s, box_s = s
        gap = abs(frame_times[i_frag]-t_s)
        if gap > MAX_GAP or any(breaks[min(i_s, i_frag)+1:max(i_s, i_frag)+1]):
            return None, gap
        x, y, h = _centre(box_s, width, height)
        A = T[i_frag] @ np.linalg.inv(T[i_s])
        p = A @ np.array([x, y, 1.0])
        px, py = p[0]/p[2], p[1]/p[2]
        fx, fy, fh = _centre(box_frag, width, height)
        metres = math.hypot(px-fx, py-fy)/max((h+fh)/2, 1.0)*PLAYER_HEIGHT
        sigma = 3+7*min(gap, 2.0)
        return math.exp(-.5*(metres/sigma)**2), gap

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

    for frag in fragments:
        rows = frag['rows']
        t0, t1 = frames[rows[0][0]]['time'], frames[rows[-1][0]]['time']
        if t1-t0 < MIN_FRAGMENT:
            if frag['cut'] is not None:
                note(frag, f'{t0:.2f}-{t1:.2f} s: too short to compare')
            continue
        pid = frag['id']
        if pid is not None and not frag['stale']:
            if frag['cut'] is not None:
                note(frag, f'{t0:.2f}-{t1:.2f} s: {pid}, identified during the run after the cut; kept')
            continue
        tr = track_state(frag['track'])
        # The fragment's own clean crops: those of its time span from the track's thinned history, else the
        # track's last clean crops (a mixed track's end may show another person; the history keeps it apart).
        history = [d for when, d in getattr(tr, 'clean_history', []) if t0-.1 <= when <= t1+.1] if tr is not None else []
        descs = history or [d for d, _, occluded in (tr.descs if tr is not None else []) if not occluded] or [d for d, _, _ in (tr.descs if tr is not None else [])]
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
            c = compare(g.gallery, descs) if g.gallery and descs else None
            appearance = None if c is None else (c['appearance'] if c['appearance'] is not None else clamp(1-c['total']/.6))
            fwd, gap_f = spatial(g, rows[0][0], rows[0][1]['box'], True)
            bwd, gap_b = spatial(g, rows[-1][0], rows[-1][1]['box'], False)
            if fwd is None and bwd is None:
                continue
            gap = min(x for x in (gap_f, gap_b) if x is not None)
            sp = max(x for x in (fwd, bwd) if x is not None)
            if appearance is None and (gap > MOTION_ONLY_GAP or ignored):
                continue   # position alone says little after a few seconds, and never revives an ignored person
            temporal = clamp(math.exp(-gap/30), .2, 1)
            final = combine({'appearance': appearance, 'uniform': None, 'spatial': sp, 'movement': None, 'temporal': temporal, 'team': True})
            scored.append({'g': g, 'final': final, 'appearance': appearance, 'spatial': sp, 'forward': fwd, 'backward': bwd, 'gap': gap, 'cosine': c.get('cosine') if c else None})
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
            continue
        g = best['g']
        conf = r3(min(best['final'], g.identity_confidence or best['final']))
        for _, person in rows:
            person['firstPassLabel'] = person.get('display')
            if old is not None and person in old.emitted_all:
                old.emitted_all.remove(person)
                old.observations -= 1
            manager._style(person, g, conf)
            person['evidence'] = 'connected offline'
            g.emitted_all.append(person)
            g.observations += 1
        if frag['track'] not in g.tracks:
            g.tracks.append(frag['track'])
        times[g.pid] = sorted(times.get(g.pid, [])+[frames[i]['time'] for i, _ in rows])
        if old is not None:
            times[old.pid] = [x for x in times.get(old.pid, []) if not (t0-1e-6 <= x <= t1+1e-6)]
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
            'framesRestoredToPlayers': report.get('framesRestored', 0), 'restoredFromLabels': dict(shown)}
