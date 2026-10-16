"""The active roster: one slot per global identity of the match (team players and goalkeepers), with its
status at the end of the segment, derived from a finished result (pure: the same result gives the same
roster, after a correction too).

A slot is an active participant, never a guaranteed location. A slot beyond the team's active outfield
slots (soccer/labels.py: the least participation evidence when a team holds more identities than it
fields) is `confirmed: False`: shown as A-?, its observations kept, no positions estimated for it.
Statuses at the end of the segment:
ACTIVE (observed within the last second), OFF_SCREEN (last seen leaving through a frame edge),
UNKNOWN (missing without an exit seen), LEFT (you marked the player as substituted or sent off) and
RETIRED (merged into another identity). The expected sizes of the teams come from the run's
maxPerTeam; a team with more identities than that is reported as over the cap (duplicates after a
failed re-identification) rather than hidden. People tracked with a team but no identity are counted as
anonymous observations: real measurements the 2D view shows as '?' markers, never as identities.
"""
from collections import Counter

ACTIVE_WITHIN = 1.0   # seconds before the end of the segment within which a last observation counts as active
STATUSES = ('ACTIVE', 'OFF_SCREEN', 'UNKNOWN', 'LEFT', 'RETIRED')


def _valid(p):
    return isinstance(p, list) and len(p) == 2 and all(isinstance(v, (int, float)) for v in p) and -.1 <= p[0] <= 1.1 and -.1 <= p[1] <= 1.1


def observations(result):
    """Per identity: frame count, first and last time, the local tracks; and the anonymous fragments
    (runs of a tracked, team-labelled person without an identity) with their frame count."""
    seen, first, last, tracks = Counter(), {}, {}, {}
    anonymous_frames, fragments, open_runs = 0, 0, {}
    for f in result.get('frames', []):
        t = f['time']
        current = set()
        for p in f.get('people', []):
            pid = p.get('id')
            if pid:
                seen[pid] += 1
                first.setdefault(pid, t)
                last[pid] = t
                tracks.setdefault(pid, set()).add(p.get('track'))
            elif p.get('state') == 'uncertain' and p.get('team') in ('A', 'B'):
                anonymous_frames += 1
                current.add(p.get('track'))
                if p.get('track') not in open_runs:
                    open_runs[p.get('track')] = t
                    fragments += 1
        for tid in list(open_runs):
            if tid not in current:
                del open_runs[tid]
    return {'seen': seen, 'first': first, 'last': last, 'tracks': tracks, 'anonymousFrames': anonymous_frames, 'anonymousFragments': fragments}


def build(result):
    """The roster of a result: {'expected', 'slots', 'overCap', 'anonymous', 'assignment'}."""
    frames = result.get('frames', [])
    end = frames[-1]['time'] if frames else 0.0
    match = result.get('match', {})
    config = result.get('config', {}) or {}
    per_team = int(config.get('maxPerTeam', 11) or 11)
    obs = observations(result)
    slots = []
    left = {e.get('id'): e.get('left') for key in ('players', 'goalkeepers') for e in match.get(key, []) if e.get('left')}
    for key, role in (('players', 'PLAYER'), ('goalkeepers', 'GOALKEEPER')):
        for e in match.get(key, []):
            pid = e.get('id')
            if not pid:
                continue
            n = obs['seen'].get(pid, 0)
            last = obs['last'].get(pid)
            if pid in left:
                status = 'LEFT'
            elif n == 0:
                status = 'UNKNOWN'
            elif end-last <= ACTIVE_WITHIN:
                status = 'ACTIVE'
            elif e.get('exitEdge'):
                status = 'OFF_SCREEN'
            else:
                status = 'UNKNOWN'
            slots.append({'id': pid, 'display': e.get('display', pid), 'team': e.get('team'), 'role': role, 'status': status,
                          'firstSeen': obs['first'].get(pid), 'lastSeen': last, 'observedFrames': n,
                          'tracks': sorted(t for t in obs['tracks'].get(pid, ()) if t is not None),
                          'lastPitch': e.get('lastPitch'), 'exitEdge': e.get('exitEdge') or '', 'left': left.get(pid), 'manual': bool(e.get('manual')),
                          # Beyond the team's active slots (soccer/labels.py): shown as A-?, no estimated positions, data kept.
                          'confirmed': bool(e.get('confirmed', True)), 'participation': e.get('participation')})
    for e in match.get('retired', []):
        if e.get('id'):
            slots.append({'id': e['id'], 'display': e.get('display', e['id']), 'team': e.get('team'), 'role': 'PLAYER' if e.get('role') != 'GOALKEEPER' else 'GOALKEEPER',
                          'status': 'RETIRED', 'into': e.get('retiredInto'), 'observedFrames': obs['seen'].get(e['id'], 0)})
    counts = Counter(s['team'] for s in slots if s['role'] == 'PLAYER' and s['status'] not in ('RETIRED', 'LEFT') and s['team'] in ('A', 'B'))
    keepers = sum(1 for s in slots if s['role'] == 'GOALKEEPER' and s['status'] not in ('RETIRED', 'LEFT'))
    expected = {'A': per_team, 'B': per_team, 'goalkeepers': 2}
    over = {team: n-per_team for team, n in counts.items() if n > per_team}
    if keepers > 2:
        over['goalkeepers'] = keepers-2
    return {'expected': expected, 'counts': {'A': counts.get('A', 0), 'B': counts.get('B', 0), 'goalkeepers': keepers},
            'slots': slots, 'overCap': over,
            'anonymous': {'fragments': obs['anonymousFragments'], 'frames': obs['anonymousFrames']},
            'statuses': dict(Counter(s['status'] for s in slots)),
            'assignment': ((result.get('summary') or {}).get('identity') or {}).get('assignment'),
            # Proposed merges (two identities that may be one person) are kept across re-derivations; never applied here.
            'mergeProposals': list((result.get('roster') or {}).get('mergeProposals') or [])}


def summary(roster):
    return {'expected': roster['expected'], 'counts': roster['counts'], 'overCap': roster['overCap'], 'statuses': roster['statuses'],
            'anonymousFragments': roster['anonymous']['fragments'], 'anonymousFrames': roster['anonymous']['frames'],
            'mergeProposals': len(roster.get('mergeProposals') or [])}
