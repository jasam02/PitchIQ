"""The coach's corrections of a match roster (Full Match Mode), and the merge suggestions that make them quick.

Why they matter: on a wide camera the segments' own runs lose identities (a player unidentified for a stretch comes
back under a new identity), and across a segment boundary a player can only be followed when both runs identified him on
the frames they share. Measured on CAN-ISL (12 minutes, 4 segments, the overlap-matched pairs as truth): the appearance
model does not tell teammates apart across segments (gallery similarity median .42 for the same person vs .35 for
teammates, the true partner the most similar only 49% of the time) and neither does where they play (19-30%). So the
automatic stitching links only what the overlaps prove; the rest stays separate fragments, and the coach joins them.

Corrections (assembly/corrections.json), applied wherever the match is read, never by rewriting a segment:
    merge    {target, into}: target's observations are into's (both match ids). Refused when they were seen in the same
             frame (two people), when their teams differ (correct the team first), when either is a goalkeeper and the
             other is not.
    exclude  {target}: not a player (a coach, a steward, a ball boy): left out of the roster and the analytics.
    team     {target, team}: the person plays for that team.
The result is an alias map (every merged id -> the id it now is, chains followed), the excluded ids and team overrides;
the index's roster, the analytics samples and the possession record are served with them applied, and the page renames
the ids of a view with the alias map when it loads it.

Suggestions (computed at assembly): for every fragment that ends before its period does, the fragments of the same team
and role that start after it within SUGGEST_MAX_GAP_S, never seen together with it, reachable at a running speed in
the time between; ranked by how soon and how near they start (appearance only breaks ties, being weak here). They are
possibilities to look at, not decisions.
"""
import json
import math
import uuid
from collections import defaultdict
from pathlib import Path

from . import manifest as mf

SUGGEST_MAX_GAP_S = 240.0
SUGGEST_SPEED = 9.0        # m/s: farther than this (plus SUGGEST_SLACK_M) in the time between is someone else
SUGGEST_SLACK_M = 5.0
SUGGEST_PER_FRAGMENT = 3
FILE = 'corrections.json'


def load(folder):
    try:
        return json.loads((Path(folder)/'assembly'/FILE).read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return []


def save(folder, items):
    mf.write_json(Path(folder)/'assembly'/FILE, items)


def resolve_aliases(items):
    """(aliases {id: final id}, excluded set, teams {id: team}) from the corrections in order."""
    parent = {}

    def find(x):
        while parent.get(x, x) != x:
            x = parent[x]
        return x
    excluded, teams = set(), {}
    for c in items:
        if c['kind'] == 'merge':
            a, b = find(c['target']), find(c['into'])
            if a != b:
                parent[a] = b
        elif c['kind'] == 'exclude':
            excluded.add(c['target'])
        elif c['kind'] == 'team':
            teams[c['target']] = c['team']
    aliases = {x: find(x) for x in list(parent)}
    return {k: v for k, v in aliases.items() if k != v}, {find(x) for x in excluded}, {find(k): v for k, v in teams.items()}


def together_key(a, b):
    return '|'.join(sorted((a, b)))


def check_merge(index, together, items, target, into):
    """Why merging target into into is refused (None when allowed), given the corrections already made."""
    aliases, excluded, teams = resolve_aliases(items)
    a, b = aliases.get(target, target), aliases.get(into, into)
    if a == b:
        return 'they are already the same person'
    roster = {p['matchId']: p for p in index['roster']}
    if a not in roster or b not in roster:
        return 'unknown identity'
    ta, tb = teams.get(a, roster[a]['team']), teams.get(b, roster[b]['team'])
    if ta != tb:
        return f'they play for different teams ({ta} and {tb}): correct the team first'
    if (roster[a]['role'] == 'GOALKEEPER') != (roster[b]['role'] == 'GOALKEEPER'):
        return 'one is a goalkeeper and the other is not'
    members = defaultdict(set)
    for x in roster:
        members[aliases.get(x, x)].add(x)
    for x in members[a]:
        for y in members[b]:
            if together.get(together_key(x, y)):
                return f'{x} and {y} were seen at the same time ({together[together_key(x, y)]} frames): two different people'
    return None


def add(folder, index, together, request, people=None):
    """One correction appended (validated). index: the assembled index (its roster are the fragments); people: the
    match's people (fullmatch/persons.resolve) when they are built: then a target may name a person ('A-P07', its
    first fragment is meant), and the kinds of the people's roster review are accepted (split, confirm, include,
    keeper, substitution; fullmatch/persons.resolve applies them)."""
    kind = request.get('kind')
    known = {p['matchId'] for p in index['roster']}

    def fragment(name):
        if people and name in (people.get('people') or {}):
            return people['people'][name]['fragments'][0]
        return name

    def others(name):
        """The other fragments of a person target (a person's correction applies to all of its fragments)."""
        if people and name in (people.get('people') or {}):
            return list(people['people'][name]['fragments'][1:])
        return []
    target = fragment(request.get('target'))
    also = others(request.get('target'))
    items = load(folder)
    if kind == 'merge':
        into = fragment(request.get('into'))
        problem = check_merge(index, together, items, target, into)
        if problem:
            raise ValueError(f'Cannot merge {target} into {into}: {problem}.')
        entry = {'kind': 'merge', 'target': target, 'into': into}
        if also and people is not None:
            entry['also'] = also
    elif kind == 'exclude':
        if target not in known:
            raise ValueError('Unknown identity.')
        entry = {'kind': 'exclude', 'target': target}
        if also:
            entry['also'] = also
    elif kind == 'team':
        if request.get('team') not in ('A', 'B') or target not in known:
            raise ValueError('A team correction needs a known identity and team A or B.')
        entry = {'kind': 'team', 'target': target, 'team': request['team']}
        if also:
            entry['also'] = also
    elif kind in ('split', 'confirm', 'include', 'keeper') and people is not None:
        if target not in known:
            raise ValueError('Unknown fragment.')
        entry = {'kind': kind, 'target': target}
        if also and kind in ('confirm', 'include', 'keeper'):
            entry['also'] = also
        if kind == 'split':
            others = [fragment(x) for x in request.get('from') or []]
            if not others or any(x not in known for x in others):
                raise ValueError('A split needs the fragments it does not continue (from).')
            entry['from'] = others
        elif kind == 'confirm':
            slot = request.get('slot')
            if not isinstance(slot, int) or not 1 <= slot <= 10:
                raise ValueError('confirm needs the slot (1-10) the fragment is the player of.')
            entry['slot'] = slot
        elif kind == 'include':
            if target not in (people.get('excluded') or {}):
                raise ValueError(f'{target} is not left out.')
        elif kind == 'keeper':
            entry['on'] = bool(request.get('on', True))
    elif kind == 'substitution' and people is not None:
        out_f, in_f = fragment(request.get('out')), fragment(request.get('in'))
        if out_f not in known or in_f not in known or out_f == in_f:
            raise ValueError('A substitution needs two different known players (out, in).')
        status = request.get('status', 'confirmed')
        if status not in ('confirmed', 'rejected'):
            raise ValueError('status must be confirmed or rejected.')
        entry = {'kind': 'substitution', 'out': out_f, 'in': in_f, 'status': status}
    else:
        raise ValueError('kind must be merge, exclude or team'+(', split, confirm, include, keeper or substitution'
                                                                 if people is not None else ''))
    entry |= {'id': uuid.uuid4().hex[:10], 'at': mf.now()}
    items.append(entry)
    save(folder, items)
    return entry


def remove(folder, cid):
    items = load(folder)
    kept = [c for c in items if c['id'] != cid]
    if len(kept) == len(items):
        raise ValueError('No such correction.')
    save(folder, kept)
    return kept


def apply_index(index, items):
    """The index with the corrections applied: the roster merged, excluded and re-teamed, plus the alias map."""
    aliases, excluded, teams = resolve_aliases(items)
    out = dict(index)
    groups = defaultdict(list)
    for p in index['roster']:
        groups[aliases.get(p['matchId'], p['matchId'])].append(p)
    roster = []
    for final, parts in groups.items():
        base = next((p for p in parts if p['matchId'] == final), parts[0])
        firsts = [p['firstSeen'] for p in parts if p['firstSeen'] is not None]
        lasts = [p['lastSeen'] for p in parts if p['lastSeen'] is not None]
        entry = dict(base) | {'seconds': round(sum(p['seconds'] for p in parts), 2),
                              'firstSeen': min(firsts) if firsts else None, 'lastSeen': max(lasts) if lasts else None,
                              'members': [m for p in parts for m in p['members']],
                              'segments': sorted({s for p in parts for s in p['segments']}),
                              'mergedFrom': sorted(p['matchId'] for p in parts if p['matchId'] != final)}
        if final in teams:
            entry['team'] = teams[final]
            entry['teamCorrected'] = True
        if final in excluded:
            entry['excluded'] = True
        if entry['mergedFrom']:
            entry['how'] = 'joined by you'
            entry['confidence'] = 'confirmed'
        roster.append(entry)
    roster.sort(key=lambda p: (p['firstSeen'] is None, p['firstSeen'] or 0))
    out['roster'] = roster
    out['aliases'] = aliases
    out['excluded'] = sorted(excluded)
    out['teamOverrides'] = teams
    out['corrections'] = items
    return out


def suggestions(roster, together, periods, fps, pitch=(105.0, 68.0)):
    """Possible same-player pairs (see the module docstring): {matchId: [{into, gapSeconds, metres, score}]}."""
    L, W = pitch
    by_period = []
    for p in periods:
        by_period.append((float(p['start']), float(p['end'])))
    def period_of(t):
        return next((i for i, (a, b) in enumerate(by_period) if a-1e-6 <= t <= b+1e-6), None)
    out = {}
    people = [p for p in roster if p.get('firstSeen') is not None and p.get('team') in ('A', 'B')]
    for x in people:
        px = period_of(x['lastSeen'])
        if px is None or x['lastSeen'] >= by_period[px][1]-5:
            continue
        last_pos = x['history'][-1][1:] if x.get('history') else None
        found = []
        for y in people:
            if y is x or y['team'] != x['team'] or y['role'] != x['role'] or period_of(y['firstSeen']) != px:
                continue
            gap = y['firstSeen']-x['lastSeen']
            if gap <= 0 or gap > SUGGEST_MAX_GAP_S or together.get(together_key(x['matchId'], y['matchId'])):
                continue
            metres = None
            if last_pos and y.get('history'):
                first_pos = y['history'][0][1:]
                metres = math.hypot((last_pos[0]-first_pos[0])*L, (last_pos[1]-first_pos[1])*W)
                if metres > SUGGEST_SPEED*gap+SUGGEST_SLACK_M:
                    continue
            score = math.exp(-gap/60.0)*(math.exp(-(metres or 20.0)/20.0))
            found.append({'into': y['matchId'], 'gapSeconds': round(gap, 1), 'metres': round(metres, 1) if metres is not None else None,
                          'score': round(score, 3)})
        found.sort(key=lambda s: -s['score'])
        if found:
            out[x['matchId']] = found[:SUGGEST_PER_FRAGMENT]
    return out
