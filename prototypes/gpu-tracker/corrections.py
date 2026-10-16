"""Corrections made in the review page: "B-08 is really a goalkeeper", "this is the same person as A-03",
"this person is not a player" (a referee, staff: not part of the match). Referees are not tracked as
identities: a request for "referee" (from an older page) is "not a player". Results of earlier versions
that still contain REF-n identities can be corrected like any other identity.

A correction is stored per video in runs/<video>/corrections.json and does three things:

1. The run it was made on shows it at once. result.json and identities.json of that run are rebuilt
   from the untouched originals (result.raw.json, identities.raw.json, copied on the first
   correction) with every correction of the run applied in the order they were made, so a correction
   can always be undone exactly. Every frame of the person is relabelled, the roster, counts and
   labels follow, and a CORRECTED BY YOU event is added.
2. Continuing identities from that run starts from the corrected identities.
3. Every later run of the video starts with the corrected people as *manual identities*: their
   appearance gallery with the role and team you gave them, locked against automatic role changes and
   recognised by appearance before anything else (soccer/identity.py). Someone you marked as not part
   of the match is recognised the same way and left out.

A person without a global identity yet (only a local track) has no saved appearance: correcting them
changes that run's result only, and the page says so.
"""
import copy
import json
import os
import re
import shutil
import time
import uuid
from pathlib import Path

from soccer import imputation, labels, roster, visibility

ROLES = ('player', 'goalkeeper', 'referee', 'excluded', 'left')   # 'referee' (older pages) means not a player: 'excluded'; 'left': substituted or sent off
GROUPS = ('players', 'goalkeepers', 'referees')         # 'referees' only in results of earlier versions
ALL_GROUPS = GROUPS+('retired', 'excluded', 'withdrawn')
ROLE_OUT = {'player': 'PLAYER', 'goalkeeper': 'GOALKEEPER', 'referee': 'REFEREE'}
GROUP_OF = {'player': 'players', 'goalkeeper': 'goalkeepers', 'referee': 'referees'}
ROLE_IN = {'PLAYER': 'player', 'GOALKEEPER': 'goalkeeper', 'REFEREE': 'referee'}
GALLERY = 6


def long_id(pid):
    if pid.startswith('A-') or pid.startswith('B-'):
        return f'Team{pid[0]}_Player_{pid[2:]}'
    if pid.startswith('GK-'):
        return f'Goalkeeper_{int(pid[3:]):02d}'
    if pid.startswith('REF-'):
        return f'Referee_{int(pid[4:]):02d}'
    return pid


def describe(role, team=None):
    """'team B player', 'goalkeeper (team A)', 'referee (not a player)', 'not part of the match', 'unknown'."""
    role = ROLE_IN.get(role, role)
    if role == 'player':
        return f'team {team} player' if team else 'player (team unknown)'
    if role == 'goalkeeper':
        return f'goalkeeper (team {team})' if team else 'goalkeeper (team unknown)'
    if role == 'referee':
        return 'referee (not a player)'
    if role == 'excluded':
        return 'not part of the match'
    if role == 'nontarget':
        return 'ignored (not a player)'
    if role == 'left':
        return 'left the match (substituted or sent off)'
    return 'unknown'


def timestamp(t):
    return f'{int(t//60):02d}:{t % 60:05.2f}'


def _person_fields(role, team, pid):
    if role == 'player':
        return {'id': pid, 'display': pid, 'team': team, 'role': 'PLAYER', 'label': f'PLAYER_TEAM_{team}', 'state': 'confirmed'}
    if role == 'goalkeeper':
        return {'id': pid, 'display': f'GK-{team}', 'team': team, 'role': 'GOALKEEPER', 'label': f'GOALKEEPER_TEAM_{team}', 'state': 'confirmed'}
    return {'id': pid, 'display': pid, 'team': None, 'role': 'REFEREE', 'label': 'REFEREE', 'state': 'confirmed'}


# ---------- storage ----------
def path_of(video_dir):
    return Path(video_dir)/'corrections.json'


def load(video_dir):
    path = path_of(video_dir)
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
        return [c for c in data.get('corrections', []) if isinstance(c, dict) and c.get('id') and c.get('run')]
    except (ValueError, OSError, AttributeError):
        return []


def save(video_dir, items):
    path = path_of(video_dir)
    if not items:
        path.unlink(missing_ok=True)
        return
    _write(path, {'version': 1, 'corrections': items}, indent=1)


def _write(path, data, indent=None):
    """Write JSON atomically; a reader on Windows may hold the old file for a moment, so retry."""
    tmp = path.with_name(path.name+f'.{uuid.uuid4().hex[:6]}.tmp')
    tmp.write_text(json.dumps(data, indent=indent, separators=None if indent else (',', ':'), allow_nan=False), encoding='utf-8')
    for attempt in range(20):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if attempt == 19:
                tmp.unlink(missing_ok=True)
                raise
            time.sleep(.1)


def _restore_file(raw, live):
    for attempt in range(20):
        try:
            shutil.copyfile(raw, live)
            return
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(.1)


def originals(run_dir):
    """(result, snapshot or None) as the pipeline wrote them; the raw copies are made on first use."""
    run_dir = Path(run_dir)
    for name in ('result', 'identities'):
        live, raw = run_dir/f'{name}.json', run_dir/f'{name}.raw.json'
        if live.exists() and not raw.exists():
            shutil.copyfile(live, raw)
    result = json.loads((run_dir/'result.raw.json').read_text(encoding='utf-8'))
    raw_ids = run_dir/'identities.raw.json'
    snapshot = json.loads(raw_ids.read_text(encoding='utf-8')) if raw_ids.exists() else None
    return result, snapshot


def rebuild(run_dir, corrections):
    """Rewrite result.json and identities.json of a run: the originals with these corrections applied
    (byte-for-byte the originals when there are none). Returns the corrected result."""
    run_dir = Path(run_dir)
    result, snapshot = originals(run_dir)
    if not corrections:
        _restore_file(run_dir/'result.raw.json', run_dir/'result.json')
        if snapshot is not None:
            _restore_file(run_dir/'identities.raw.json', run_dir/'identities.json')
        return result
    apply_result(result, corrections)
    _write(run_dir/'result.json', result)
    if snapshot is not None:
        _write(run_dir/'identities.json', apply_snapshot(snapshot, corrections))
    return result


# ---------- the result of a run ----------
def _entries(match):
    for key in ALL_GROUPS:
        for e in match.get(key, []):
            yield key, e


def find_entry(match, pid):
    for key, e in _entries(match):
        if e.get('id') == pid:
            return key, e
    return None


def _targets(p, pid, track):
    return p.get('id') == pid if pid else (p.get('track') == track and not p.get('id'))


def apply_result(result, corrections):
    """Apply corrections (in order) to a result dict in place; returns it."""
    match = result.setdefault('match', {})
    for key in ALL_GROUPS:
        match.setdefault(key, [])
    applied = []
    for c in corrections:
        applied.append(listing_entry(c, applied=_apply_one(result, c)))
    for key in ALL_GROUPS:
        match[key].sort(key=lambda e: (e.get('id') or '', e.get('track') or 0))
    _recount(result, sum(1 for a in applied if a['applied']))
    result['corrections'] = applied
    # The display labels follow the corrected roster: no gaps, the outfield slots filled in order, internal
    # identities untouched (soccer/labels.py).
    labelling = labels.assign(result)
    moved = {pid: (old, new) for pid, (old, new) in labelling['changed'].items() if old is not None}
    if moved and applied:
        when = max((float(c.get('time') or 0.0) for c in corrections), default=0.0)
        event = {'time': round(when, 6), 'kind': 'labels', 'track': None, 'playerId': None, 'accepted': True,
                 'message': '\n'.join(['PLAYER LABELS RENUMBERED', 'Internal identities unchanged; only the shown labels moved.']+labels.describe(labelling))}
        events = result.setdefault('events', [])
        at = len(events)
        while at > 0 and events[at-1].get('time', 0) > event['time']:
            at -= 1
        events.insert(at, event)
    # The roster and the per-frame visibility follow from the corrected result.
    if result.get('frames') and result.get('video'):
        result['roster'] = roster.build(result)
        result.setdefault('summary', {})['roster'] = roster.summary(result['roster'])
        size = (result.get('tactical') or {}).get('pitchSize') or {}
        result['timeline'] = imputation.estimate(result, float(size.get('length', 105.0)), float(size.get('width', 68.0)))
        result['summary']['timeline'] = {'method': result['timeline']['method'], 'kinds': result['timeline']['kinds'], 'slots': len(result['timeline']['ids'])}
        result['summary']['visibility'] = visibility.annotate(result)
    return result


def _apply_left(result, c):
    """The player left the match at the correction's time (substituted or sent off): the roster slot is closed
    from then on; the observations are untouched (an identity continued on someone else is corrected separately)."""
    pid = c['target'].get('id')
    entry = find_entry(result['match'], pid) if pid else None
    if entry is None or entry[0] not in ('players', 'goalkeepers'):
        return False
    e = entry[1]
    when = float(c.get('time') or e.get('lastSeen') or 0.0)
    e['left'] = {'time': round(when, 3), 'reason': 'substituted or sent off (marked by you)', 'correction': c['id']}
    e.update(status='LEFT', manual=True, correction=c['id'])
    lines = ['CORRECTED BY YOU', f'Global ID: {pid} left the match at {timestamp(when)}', f"Was: {describe(c['before'].get('role'), c['before'].get('team'))}",
             'Now: left the match (substituted or sent off); the roster slot is closed from then on, the observations before it are kept']
    event = {'time': round(when, 6), 'kind': 'manual', 'track': c['target'].get('trackAt'), 'playerId': pid, 'message': '\n'.join(lines), 'accepted': True}
    events = result.setdefault('events', [])
    at = len(events)
    while at > 0 and events[at-1].get('time', 0) > event['time']:
        at -= 1
    events.insert(at, event)
    return True


def _apply_one(result, c):
    pid, track = c['target'].get('id'), c['target'].get('track')
    role, team, new, into = c['after']['role'], c['after'].get('team'), c['after'].get('id'), c['after'].get('into')
    if role == 'left':
        return _apply_left(result, c)
    match = result['match']
    fields = None if role == 'excluded' else _person_fields(role, team, new)
    hit, first, last, last_box = 0, None, None, None
    for f in result.get('frames', []):
        for p in f.get('people', []):
            mine = _targets(p, pid, track)
            joined = bool(into) and not mine and p.get('id') == new   # the identity merged into takes the corrected role too
            if not (mine or joined):
                continue
            if mine:
                hit += 1
                first = f['time'] if first is None else first
                last, last_box = f['time'], p.get('box')
            if role == 'excluded':
                p.update(id=None, display='OUT', team=None, role='UNKNOWN', label='EXCLUDED_BY_YOU', state='rejected',
                         identityConfidence=0.0, reason='marked by you as not part of this match')
                if pid:
                    p['excludedId'] = pid
            else:
                p.update(fields)
                p.pop('reason', None)
                p['roleConfidence'] = 1.0
                if not pid and mine:
                    p['identityConfidence'] = 1.0
            p['manual'], p['correction'] = True, c['id']
    if not hit:
        return False
    when = c.get('time') if c.get('time') is not None else first
    source = find_entry(match, pid) if pid else None
    if source is not None and source[0] in GROUPS:
        match[source[0]].remove(source[1])
    if role == 'excluded':
        entry = copy.deepcopy(source[1]) if source else {'id': None, 'track': track, 'longId': f'Track {track}', 'display': 'OUT'}
        entry.update(excluded=True, manual=True, correction=c['id'], status='EXCLUDED', firstSeen=first, lastSeen=last, observations=hit)
        match['excluded'].append(entry)
    else:
        group = GROUP_OF[role]
        existing = find_entry(match, new)
        person = {'team': fields['team'], 'role': fields['role'], 'label': fields['label'], 'display': fields['display']}
        if existing is not None and existing[0] in GROUPS:
            key, e = existing
            if key != group:
                match[key].remove(e)
                match[group].append(e)
            e.update(person, roleLocked=True, manual=True, correction=c['id'], revoked=None)
            seen = int(e.get('observations', 0)) > 0   # an identity only remembered, never seen in this run, has no times here
            e['observations'] = int(e.get('observations', 0))+hit
            e['firstSeen'] = min(e.get('firstSeen', first), first) if seen else first
            e['lastSeen'] = max(e.get('lastSeen', last), last) if seen else last
            e['roleHistory'] = list(e.get('roleHistory') or [])[-5:]+[[round(when, 3), role, f"{pid or f'track {track}'} merged in by you"]]
        else:
            e = copy.deepcopy(source[1]) if source else {
                'status': 'MISSING', 'identityConfidence': 1.0, 'teamConfidence': 1.0 if team else 0.0, 'refereeConfidence': 0.0,
                'goalkeeperConfidence': 0.0, 'gallerySize': 0, 'lastBox': last_box, 'lastPitch': None, 'velocity': [0.0, 0.0], 'exitEdge': '',
                'restored': False, 'history': [], 'official': None, 'roleEvidence': {}, 'roleHistory': [], 'trackOnly': track}
            e.pop('retiredInto', None)
            e.update(person, id=new, longId=long_id(new), roleLocked=True, roleConfidence=1.0, manual=True, correction=c['id'], revoked=None,
                     observations=hit, firstSeen=first, lastSeen=last)
            if role == 'referee':
                e['official'] = e.get('official') if (source and source[1].get('role') == 'REFEREE') else 'UNKNOWN_OFFICIAL'
            else:
                e['official'] = None
            e['roleHistory'] = list(e.get('roleHistory') or [])[-5:]+[[round(when, 3), role, 'set by you']]
            match[group].append(e)
    if source is not None and source[0] in GROUPS:
        retired = copy.deepcopy(source[1])
        retired.update(status='RETIRED' if role != 'excluded' else 'EXCLUDED', retiredInto=new, manual=True, correction=c['id'])
        if role != 'excluded':
            match['retired'].append(retired)
    lines = ['CORRECTED BY YOU', f"{'Global ID' if pid else 'Local Track'}: {pid or track} -> {new or 'not part of the match'}",
             f"Was: {describe(c['before'].get('role'), c['before'].get('team'))}",
             f"Now: {describe(role, team)}" + (f' (the same person as {new})' if into else ''),
             f'Observations relabelled: {hit}' + (' (and those of the identity it joined)' if into else ''),
             'Remembered for later runs of this video: ' + ('yes, recognised by appearance' if c.get('identity') else 'no (this person has no saved appearance)')]
    event = {'time': round(float(when), 6), 'kind': 'manual', 'track': c['target'].get('trackAt', track), 'playerId': new or pid,
             'message': '\n'.join(lines), 'accepted': True}
    events = result.setdefault('events', [])
    at = len(events)
    while at > 0 and events[at-1].get('time', 0) > event['time']:
        at -= 1
    events.insert(at, event)
    return True


def _recount(result, applied):
    match, summary = result['match'], result.setdefault('summary', {})
    identities = summary.setdefault('globalIdentities', {})
    identities.update(teamA=sum(1 for p in match['players'] if p.get('team') == 'A'),
                      teamB=sum(1 for p in match['players'] if p.get('team') == 'B'),
                      goalkeepers=sum(1 for p in match['goalkeepers'] if not p.get('revoked')),
                      referees=len(match['referees']), merged=len(match['retired']), excluded=len(match['excluded']))
    roles = summary.setdefault('roles', {})
    roles['mergedIdentities'] = len(match['retired'])
    summary['corrections'] = applied
    slots = dict(summary.get('goalkeeperSlots') or {})
    for team in ('A', 'B'):
        keepers = [p for p in match['goalkeepers'] if p.get('team') == team and not p.get('revoked')]
        manual = [p['id'] for p in keepers if p.get('manual')]
        ids = [p['id'] for p in keepers]
        slots[team] = manual[-1] if manual else slots.get(team) if slots.get(team) in ids else (ids[0] if ids else None)
    summary['goalkeeperSlots'] = slots
    result['labels'] = {e['id']: long_id(e['id']) for _, e in _entries(match) if e.get('id')}


# ---------- saved identities of a run (continuing identities) ----------
def _merge_galleries(a, b, size=GALLERY):
    """Up to `size` samples taken alternately from the newest of each gallery, in time order."""
    a, b = sorted(a or [], key=lambda s: s.get('t', 0)), sorted(b or [], key=lambda s: s.get('t', 0))
    out = []
    while (a or b) and len(out) < size:
        for source in (b, a):
            if source and len(out) < size:
                out.append(source.pop())
    return sorted(out, key=lambda s: s.get('t', 0))


def _key_of(pid):
    m = re.fullmatch(r'(REF|GK|A|B)-(\d+)', pid or '')
    return (m.group(1), int(m.group(2))) if m else (None, 0)


def _as_manual(entry, role, team, pid):
    e = copy.deepcopy(entry)
    e.update(id=pid, role=role, team=team if role in ('player', 'goalkeeper') else None, manual=True, excluded=False, roleLocked=True,
             revoked='', retiredInto=None, roleConfidence=1.0)
    if e.get('status') in ('retired', 'unknown', None):
        e['status'] = 'missing'
    if role == 'goalkeeper':
        e['keeperGoal'], e['gkVotes'] = 1.0, {}
    if role != 'referee':
        e['official'] = ''
    return e


def apply_snapshot(snapshot, corrections):
    """Apply corrections to a run's saved identities (what 'continue identities' starts from)."""
    snapshot = copy.deepcopy(snapshot)
    registry = snapshot.setdefault('registry', {'version': 1, 'identities': [], 'counters': {}})
    items = registry.setdefault('identities', [])
    counters = registry.setdefault('counters', {})
    for c in corrections:
        pid = c['target'].get('id')
        if not pid:
            continue   # a local track only: nothing saved to correct
        role, team, new, into = c['after']['role'], c['after'].get('team'), c['after'].get('id'), c['after'].get('into')
        by_id = {e['id']: e for e in items}
        source = by_id.get(pid)
        if source is None:
            continue
        if role == 'excluded':
            source.update(manual=True, excluded=True, status='unknown', roleLocked=False)
            continue
        if role == 'left':
            source.update(manual=True, status='substituted')   # a later run never continues a player who left
            continue
        target = by_id.get(new) if into else None
        if target is not None:
            merged = _as_manual(target, role, team, new)
            merged['gallery'] = _merge_galleries(target.get('gallery'), source.get('gallery'))
            target.clear()
            target.update(merged)
        else:
            items.append(_as_manual(source, role, team, new))
        source.update(status='retired', retiredInto=new)
        key, number = _key_of(new)
        if key:
            counters[key] = max(int(counters.get(key, 0)), number)
    return snapshot


# ---------- making a correction ----------
def _taken_ids(result, snapshot, items):
    taken = {e.get('id') for _, e in _entries(result.get('match', {})) if e.get('id')}
    if snapshot:
        taken |= {e.get('id') for e in snapshot.get('registry', {}).get('identities', [])}
    for c in items:
        taken |= {c['target'].get('id'), c['after'].get('id'), (c.get('identity') or {}).get('id')}
    return {t for t in taken if t}


def next_id(role, team, taken):
    prefix = 'REF-' if role == 'referee' else 'GK-' if role == 'goalkeeper' else f'{team}-'
    numbers = [int(m.group(1)) for pid in taken for m in [re.fullmatch(re.escape(prefix)+r'(\d+)', pid)] if m]
    n = max(numbers, default=0)+1
    return f'{prefix}{n:02d}' if role == 'player' else f'{prefix}{n}'


def _frames_of(result, pid, track):
    return [i for i, f in enumerate(result.get('frames', [])) if any(_targets(p, pid, track) for p in f.get('people', []))]


def make(result, snapshot, request, items, run, now=None):
    """Validate a correction request against the run's current (corrected) result and saved identities
    and return the correction to store. request: {'target': {'id' | 'track', 'time'}, 'role', 'team',
    'into'}. Raises ValueError with a message for the page."""
    if not isinstance(request, dict) or not isinstance(request.get('target'), dict):
        raise ValueError('Expected a person and what they are.')
    target = request['target']
    pid = target.get('id') or None
    track = target.get('track')
    if pid is not None and (not isinstance(pid, str) or len(pid) > 16):
        raise ValueError('Unknown identity.')
    if pid is None and (not isinstance(track, int) or isinstance(track, bool)):
        raise ValueError('Choose a person on the video.')
    role, team, into = request.get('role'), request.get('team'), request.get('into') or None
    if role not in ROLES:
        raise ValueError('Choose team A or B player, goalkeeper, or not a player.')
    if role == 'referee':
        role = 'excluded'   # referees are not tracked: "a referee" is "not a player", recognised and left out
    if role == 'left':
        if not pid:
            raise ValueError('Only an identified player can leave the match; identify the person first.')
        into = None
    if role in ('player', 'goalkeeper'):
        if team not in ('A', 'B'):
            raise ValueError('Choose team A or team B.')
    else:
        team = None
    if role == 'excluded':
        into = None
    if into is not None and (not isinstance(into, str) or len(into) > 16):
        raise ValueError('Unknown identity to join.')
    time_at = target.get('time')
    time_at = float(time_at) if isinstance(time_at, (int, float)) and not isinstance(time_at, bool) else None
    match = result.get('match', {})
    mine = _frames_of(result, pid, track)
    name = pid or f'Track {track}'
    if not mine:
        raise ValueError(f'{name} is not in this result (it may have been corrected already); reload the page.')
    entry = find_entry(match, pid) if pid else None
    if pid and (entry is None or entry[0] not in GROUPS):
        raise ValueError(f'{pid} is not a current identity of this result; reload the page.')
    frames = result['frames']
    if entry is not None:
        e = entry[1]
        before = {'id': pid, 'track': track, 'role': ROLE_IN.get(e.get('role'), 'unknown'), 'team': e.get('team'), 'display': e.get('display')}
    else:
        sample = next(p for p in frames[mine[0]]['people'] if _targets(p, None, track))
        before = {'id': None, 'track': track, 'role': ROLE_IN.get(sample.get('role'), 'unknown'), 'team': sample.get('team'), 'display': sample.get('display')}
        if sample.get('label') == 'EXCLUDED_BY_YOU':
            before['role'] = 'excluded'
        elif sample.get('label') == 'NON_TARGET':
            before['role'] = 'nontarget'
    if into is not None:
        joined = find_entry(match, into)
        if joined is None or joined[0] not in GROUPS or into == pid:
            raise ValueError(f'{into} is not another identity of this result.')
        j = joined[1]
        same_group = (role == 'referee' and j.get('role') == 'REFEREE') or \
                     (role == 'player' and j.get('role') == 'PLAYER' and j.get('team') == team) or \
                     (role == 'goalkeeper' and j.get('role') == 'GOALKEEPER' and j.get('team') in (team, None))
        if not same_group:
            raise ValueError(f'{into} is a {describe(j.get("role"), j.get("team"))}, not a {describe(role, team)}.')
        theirs = set(_frames_of(result, into, None))
        both = sorted(theirs.intersection(mine))
        if both:
            raise ValueError(f'{name} and {into} are both on the video at {timestamp(frames[both[0]]["time"])}: they are two different people.')
    elif role == 'left':
        if entry[1].get('left'):
            raise ValueError(f'{pid} already left the match at {timestamp(entry[1]["left"]["time"])}.')
    elif pid and before['role'] == role and (before['team'] == team or role == 'referee'):
        raise ValueError(f'{pid} already is a {describe(role, team)}. To say they are someone else, choose "same person as".')
    taken = _taken_ids(result, snapshot, items)
    new = into if into else None if role in ('excluded', 'left') else next_id(role, team, taken)
    if role == 'left':
        new, team = pid, before.get('team')
    identity, teams = None, None
    if pid and snapshot and role != 'left':
        saved = {e['id']: e for e in snapshot.get('registry', {}).get('identities', [])}
        source = saved.get(pid)
        if source is not None and source.get('gallery'):
            if role == 'excluded':
                identity = copy.deepcopy(source)
                identity.update(manual=True, excluded=True, status='unknown', roleLocked=False, lastPitch=None, exitEdge='', observations=0)
            else:
                base = saved.get(into) if into else None
                identity = _as_manual(base or source, role, team, new)
                if base is not None:
                    identity['gallery'] = _merge_galleries(base.get('gallery'), source.get('gallery'))
                identity.update(lastPitch=None, exitEdge='', observations=0)
            if role in ('player', 'goalkeeper'):
                teams = snapshot.get('teams')
    now = now or time.time()
    stamp = time.strftime('%Y%m%d-%H%M%S', time.localtime(now))
    track_at = target.get('trackAt') if isinstance(target.get('trackAt'), int) else track
    return {'id': f'c{stamp}-{uuid.uuid4().hex[:6]}', 'run': run, 'made': time.strftime('%Y-%m-%dT%H:%M:%S', time.localtime(now)),
            'time': time_at if time_at is not None else frames[mine[0]]['time'],
            'target': {'id': pid, 'track': None if pid else track, 'trackAt': track_at}, 'before': before,
            'after': {'role': role, 'team': team, 'id': new, 'into': bool(into)}, 'identity': identity, 'teams': teams}


def remove(items, cid):
    """Remove a correction and the later corrections of the same run that act on what it produced
    (a correction of the referee it created, a merge into it). Returns (kept, removed ids)."""
    first = next((c for c in items if c['id'] == cid), None)
    if first is None:
        raise ValueError('That correction no longer exists.')
    gone, produced = {cid}, {first['after'].get('id')} - {None}
    started = False
    for c in items:
        if c['id'] == cid:
            started = True
            continue
        if not started or c['run'] != first['run']:
            continue
        if c['target'].get('id') in produced or (c['after'].get('into') and c['after'].get('id') in produced):
            gone.add(c['id'])
            if c['after'].get('id'):
                produced.add(c['after']['id'])
    return [c for c in items if c['id'] not in gone], gone


# ---------- what later runs remember ----------
def seeds(items):
    """The people later runs start with: every corrected person with a saved appearance, the latest
    correction of a person winning (a referee you later changed back to a player is that player).
    Returns {'identities': [snapshot entries], 'teams': team model or None}."""
    memory, teams = {}, None
    for c in items:
        source, new = c['target'].get('id'), c['after'].get('id')
        if source and source in memory and source != new:
            memory.pop(source)
        identity = c.get('identity')
        if identity:
            memory[identity['id']] = identity
        if c.get('teams'):
            teams = c['teams']
    out = []
    for e in memory.values():
        e = copy.deepcopy(e)
        e.update(manual=True, lastPitch=None, exitEdge='', observations=0)
        e['status'] = 'unknown' if e.get('excluded') else 'missing'
        out.append(e)
    team_bound = any(e.get('role') in ('player', 'goalkeeper') and e.get('team') and not e.get('excluded') for e in out)
    return {'identities': sorted(out, key=lambda e: e['id']), 'teams': teams if team_bound else None}


def seed_state(previous, items):
    """What a new run starts from: the continued identities (if any) plus the remembered corrected
    people not already among them. Team letters only mean something with the kit model they were
    given under, so a run that does not continue identities starts from that model when a remembered
    person has a team."""
    memory = seeds(items)
    if not memory['identities']:
        return previous
    saved = dict(previous) if previous else {}
    registry = copy.deepcopy(saved.get('registry')) if saved.get('registry') else {'version': 1, 'identities': [], 'counters': {}}
    present = {e.get('id') for e in registry.get('identities', [])}
    added = [e for e in memory['identities'] if e['id'] not in present]
    registry['identities'] = list(registry.get('identities', []))+added
    saved['registry'] = registry
    if not saved.get('teams') and memory['teams']:
        saved['teams'] = memory['teams']
    saved['manualSeeds'] = [e['id'] for e in added]
    return saved


def listing_entry(c, applied=None, current_run=None):
    """A correction as the page lists it."""
    out = {'id': c['id'], 'run': c['run'], 'made': c.get('made'), 'time': c.get('time'),
           'from': c['target'].get('id') or f"track {c['target'].get('track')}", 'to': c['after'].get('id') or 'OUT',
           'what': describe(c['after']['role'], c['after'].get('team')), 'was': describe(c['before'].get('role'), c['before'].get('team')),
           'into': bool(c['after'].get('into')), 'remembered': bool(c.get('identity'))}
    if applied is not None:
        out['applied'] = applied
    if current_run is not None:
        out['thisRun'] = c['run'] == current_run
    return out


def add(video_dir, run_dir, request):
    """Validate and store a correction made on run_dir (the run the page shows) and rebuild the run's
    files. Returns the stored correction's listing entry."""
    items = load(video_dir)
    run = Path(run_dir).name
    mine = [c for c in items if c['run'] == run]
    result, snapshot = originals(run_dir)
    apply_result(result, mine)
    if snapshot is not None:
        snapshot = apply_snapshot(snapshot, mine)
    c = make(result, snapshot, request, items, run)
    items.append(c)
    save(video_dir, items)
    rebuild(run_dir, mine+[c])
    return listing_entry(c, applied=True, current_run=run)


def drop(video_dir, run_dir, cid):
    """Remove a correction (and its dependents); rebuild the shown run when it had any of them.
    Returns the removed correction ids."""
    before = load(video_dir)
    items, gone = remove(before, cid)
    save(video_dir, items)
    run = Path(run_dir).name if run_dir else None
    if run and any(c['run'] == run for c in before if c['id'] in gone):
        rebuild(run_dir, [c for c in items if c['run'] == run])
    return gone


def listing(video_dir, current_run):
    """Every correction of the video, newest first, for the page."""
    return [listing_entry(c, current_run=current_run) for c in reversed(load(video_dir))]
