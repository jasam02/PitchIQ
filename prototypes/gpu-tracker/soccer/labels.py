"""Display labels of the players (A-01, B-07), regenerated from the confirmed roster, apart from the
internal identity.

`id` (A-07, B-12) is the internal global identity: created once by the identity manager, never
renumbered, never reused. Appearance memory, trajectories, tracklet associations, the timeline,
corrections and the saved identities of a run all key on it. `display` is what the overlay, the
roster and the 2D view show, and it is regenerated here: per team, the confirmed outfield players in
the order their identities were created take A-01, A-02, ... with no gaps, up to OUTFIELD players
active at a time. A player who left the match (substituted or sent off: a correction) keeps their
number and frees a slot, so a substitute takes the next number. When a team holds more active
identities than it fields, the slots go to the identities with the most participation evidence over
the whole segment (seconds observed clear of the frame edges and of the outside of the pitch, compared
in two-second steps; ties in creation order), never to the first created: a person seen for four seconds at the bottom edge
among the staff does not keep a slot from a player tracked for a minute on the far touchline whose
kit took longer to read. In a small-scene run (far camera, amateur footage: at least half the frames have
profile.smallScene) the slots go first to the identities whose clean kit votes clearly established them as team
players (teamHistory.established, or identified by you), then by participation: there a duplicate made by a kit
misread on small crops collects as much presence as a real player, but not his kit history (amateur 3: B-11, 7 of
29 clear votes, numbered over the established B-01, 64 of 75). Identities beyond the active slots stay unconfirmed (`A-?`), with all
their tracking data kept for re-identification and corrections, until a slot frees up, for instance
when you delete a false player: then the remaining players are renumbered without gaps and the
person who was B-11 is shown as B-09 while remaining the same internal identity. The roster count is
supporting evidence only: nothing is merged, deleted or invented from it. Goalkeepers are GK-A / GK-B
as before and are not counted.

Labels are regenerated at the end of a run and after every correction, never while tracking.
"""
import re
from collections import Counter

OUTFIELD = 10      # outfield players a team fields at a time in a normal match (config 'outfieldPerTeam')
EXTRA = '?'        # the display of a confirmed identity beyond the active slots: unconfirmed until a slot frees up
EDGE = .005        # a box this close to a frame edge (normalized) is clipped: not evidence of being on the pitch
PRESENCE_STEP = 2.0   # seconds: presence is compared in steps of this size, so near-equal evidence keeps the creation order
                      # (a few frames more or less between runs never move a number)
SMALL_SCENE_SHARE = .5   # a run is a small scene when at least this share of its frames are (frames[].profile.smallScene)


def long_label(display):
    if display and display[:2] in ('A-', 'B-'):
        return f'Team{display[0]}_Player_{display[2:]}'
    return display


def outfield_of(result):
    config = result.get('config') or {}
    try:
        return max(1, int(config.get('outfieldPerTeam', OUTFIELD)))
    except (TypeError, ValueError):
        return OUTFIELD


def _order(pid):
    m = re.fullmatch(r'[AB]-(\d+)', pid or '')
    return (int(m.group(1)) if m else 10**6, pid or '')


def small_scene_run(frames):
    """A small-scene run (far camera, amateur footage): at least SMALL_SCENE_SHARE of its frames were small scenes
    (SoccerTracker.small_scene, written as frames[].profile.smallScene). Never true on broadcast footage."""
    small = sum(1 for f in frames if (f.get('profile') or {}).get('smallScene'))
    return bool(frames) and small >= SMALL_SCENE_SHARE*len(frames)


def established(entry):
    """A match entry whose clean kit votes clearly established it as a team player (IdentityManager._established),
    or someone you identified."""
    return bool(entry.get('manual') or (entry.get('teamHistory') or {}).get('established'))


def participation(result):
    """Per player identity, from the frames: seconds observed, seconds of presence (observed with the box clear
    of the frame edges and the person not outside the pitch), the share of observations at an edge or outside,
    and the most teammates in view at the same time. Evidence of being a fielded player; a sideline person is
    seen briefly, at an edge, below the touchline."""
    fps = float((result.get('video') or {}).get('fps') or 0.0) or 25.0
    dt = 1.0/fps
    stats = {}
    for f in result.get('frames', []):
        people = f.get('people', [])
        in_view = Counter(p.get('team') for p in people if p.get('id') and p.get('role') == 'PLAYER' and p.get('state') == 'confirmed')
        for p in people:
            pid = p.get('id')
            if not pid or p.get('role') != 'PLAYER':
                continue
            s = stats.setdefault(pid, {'frames': 0, 'edge': 0, 'teammates': 0})
            b = p.get('box') or [0.0, 0.0, 0.0, 0.0]
            clipped = b[0] <= EDGE or b[0]+b[2] >= 1-EDGE or b[1]+b[3] >= 1-EDGE
            s['frames'] += 1
            s['edge'] += 1 if clipped or p.get('zone') == 'outside' else 0
            s['teammates'] = max(s['teammates'], in_view.get(p.get('team'), 1)-1)
    return {pid: {'seconds': round(s['frames']*dt, 2), 'presence': round((s['frames']-s['edge'])*dt, 2),
                  'edgeShare': round(s['edge']/s['frames'], 3) if s['frames'] else 0.0, 'teammatesInView': s['teammates']}
            for pid, s in stats.items()}


def plan(result):
    """{internal id: display} for every team player of the result, and per team how many are numbered,
    active, left and unconfirmed (with the unconfirmed identities and their participation). Writes
    `confirmed` and `participation` on the match entries."""
    match = result.get('match') or {}
    cap = outfield_of(result)
    part = participation(result)
    small = small_scene_run(result.get('frames') or [])
    mapping, counts = {}, {}
    for team in ('A', 'B'):
        players = sorted((e for e in match.get('players', []) if e.get('team') == team and e.get('id')), key=lambda e: _order(e['id']))
        active_players = [e for e in players if not e.get('left')]
        # Over the cap, the slots go to the identities with the most participation evidence; ties keep the creation order.
        # A small-scene run ranks the identities with an established kit history first.
        presence = lambda e: (-int(part.get(e['id'], {}).get('presence', 0.0)//PRESENCE_STEP), _order(e['id']))
        ranked = sorted(active_players, key=(lambda e: (not established(e),)+presence(e)) if small else presence)
        confirmed = {e['id'] for e in ranked[:cap]}
        numbered = active = left = extra = 0
        waiting = []
        for e in players:
            gone = bool(e.get('left'))
            e['participation'] = part.get(e['id'], {'seconds': 0.0, 'presence': 0.0, 'edgeShare': 0.0, 'teammatesInView': 0})
            e['confirmed'] = gone or e['id'] in confirmed
            if e['confirmed']:
                numbered += 1
                mapping[e['id']] = f'{team}-{numbered:02d}'
                if gone:
                    left += 1
                else:
                    active += 1
            else:
                mapping[e['id']] = f'{team}-{EXTRA}'
                extra += 1
                waiting.append(e['id'])
        counts[team] = {'numbered': numbered, 'active': active, 'left': left, 'unconfirmed': extra,
                        'unconfirmedIds': waiting, 'participation': {pid: part.get(pid) for pid in waiting},
                        'weakestConfirmed': min((part.get(e['id'], {}).get('presence', 0.0) for e in active_players if e['id'] in confirmed), default=None)}
        if small:
            counts[team]['ranking'] = 'established kit, then participation'
            counts[team]['teamHistory'] = {e['id']: e.get('teamHistory') for e in players if e['id'] in waiting}
    return mapping, counts


def assign(result):
    """Regenerate the display labels of a result in place: the match entries, every frame, the labels map
    and summary.labelling. Internal identities are untouched. Returns {'changed': {id: (old, new)},
    'counts': per team, 'outfield': cap}."""
    mapping, counts = plan(result)
    changed = {}
    match = result.get('match') or {}
    for e in match.get('players', []):
        new = mapping.get(e.get('id'))
        if new is None:
            continue
        old = e.get('display')
        e['display'], e['numbered'], e['longId'] = new, not new.endswith(EXTRA), long_label(new)
        if old != new:
            changed[e['id']] = (old, new)
    for f in result.get('frames', []):
        for p in f.get('people', []):
            new = mapping.get(p.get('id'))
            if new is not None and p.get('role') == 'PLAYER':
                p['display'] = new
    labels = result.setdefault('labels', {})
    for pid, new in mapping.items():
        labels[pid] = long_label(new)
    cap = outfield_of(result)
    summary = result.setdefault('summary', {})
    summary['labelling'] = {'outfieldPerTeam': cap, 'teams': counts,
                            'note': 'display labels are regenerated from the confirmed roster; id is the internal identity and never changes'}
    return {'changed': changed, 'counts': counts, 'outfield': cap}


def describe(outcome):
    """One line per team for the log: what is numbered, what waits, what was renumbered."""
    lines = []
    for team, c in outcome['counts'].items():
        line = f"Team {team}: {c['numbered']} numbered ({c['active']} active" + (f", {c['left']} left the match" if c['left'] else '') + ')'
        if c['unconfirmed']:
            line += f", {c['unconfirmed']} unconfirmed beyond the {outcome['outfield']} active slots (shown as {team}-{EXTRA}, kept for re-identification and corrections)"
        lines.append(line)
        for pid in c.get('unconfirmedIds', []):
            q = (c.get('participation') or {}).get(pid) or {}
            lines.append(f"  {pid} unconfirmed: seen {q.get('seconds', 0):.1f} s, {q.get('presence', 0):.1f} s clear of the frame edges and the outside"
                         f" ({q.get('edgeShare', 0):.0%} at an edge), at most {q.get('teammatesInView', 0)} teammates in view with it; the slots went to"
                         f" identities present {c['weakestConfirmed']:.1f} s or more" if c.get('weakestConfirmed') is not None else
                         f"  {pid} unconfirmed: seen {q.get('seconds', 0):.1f} s")
            if 'teamHistory' in c:
                h = c['teamHistory'].get(pid) or {}
                lines[-1] += (f"; kit history {h.get('clearKitVotes', 0)} of {h.get('kitVotes', 0)} clean votes clearly team {team} "
                              f"({'established' if h.get('established') else 'not established'}; {c['ranking']})")
    moves = [f'{pid}: {old} -> {new}' for pid, (old, new) in sorted(outcome['changed'].items(), key=lambda kv: _order(kv[0])) if old is not None]
    if moves:
        lines.append('Renumbered (internal identity: old label -> new label): ' + ', '.join(moves))
    return lines
