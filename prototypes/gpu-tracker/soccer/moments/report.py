"""The key-moments report of one team ('coaching-moments', version 1): what the team did well, what needs improvement,
the key moments as a browsable timeline, repeated patterns, training priorities, then the team-shape numbers as
supporting context and what could not be assessed.

    build(folder, team, profile)   -> report (dict), cached in assembly/coaching/moments/

Stages and caches (each keyed by everything it depends on):
    data        soccer/moments/data.match_data: the match as 5 Hz arrays (built once from the views, ~1 min on CAN-ISL)
    episodes    every detector over the whole match, both teams (data + the coach's roster corrections + ENGINE_VERSION)
    report      selection, suggestions, words, evidence for one team and coaching profile (level, age, format)
A level change re-uses the episodes: it only re-selects, re-words and re-suggests.

Selection (per team). Main lists take episodes observed with 'high' or 'medium' reliability and read 'clear' or
'plausible' (MAIN_RELIABILITY, MAIN_INTERPRETATION); the rest of the medium-or-better ones go to 'more moments' (marked
'uncertain context' when the reading was uncertain). Duplicates: episodes of one team whose key times are within DEDUP_S
and that share a person keep only the higher score. Patterns: one kind of moment that happened PATTERN_MIN or more times,
PATTERN_SPACING_S apart. Strengths and improvements: by score, at most MAX_PER_KIND of one kind and MAX_PER_PATTERN of
one pattern among the shown ones, up to MAX_STRENGTHS / MAX_IMPROVEMENTS; fewer when fewer met the bar (never padded).

Every moment carries an evidence record the page's annotation drawing (coaching.js paint / drawFollow, index.html
jumpToEvidence) takes as it is: the 2D diagram (normalized pitch coordinates), the video overlay of the key frame (source
pixels; areas only when the key frame's calibration put the involved players' pitch positions within PROJECT_MAX_PX of
their feet), the window (lead-up and follow-up) and the people to follow.
"""
import hashlib
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from . import data as md
from . import engine as en
from . import geometry as geo
from . import goals as gl
from . import kits as kt
from . import language as lg
from . import roles as rl
from . import suggest as sg
from .detectors import common as cm
from .detectors import detect_all

ENGINE_VERSION = '1'
KIND = 'coaching-moments'
MAIN_RELIABILITY = ('high', 'medium')
MAIN_INTERPRETATION = ('clear', 'plausible')
DEDUP_S = 4.0
SEQUENCE_S = 20.0
PATTERN_MIN, PATTERN_SPACING_S = 3, 20.0
MAX_STRENGTHS = MAX_IMPROVEMENTS = 8
MAX_PER_KIND, MAX_PER_PATTERN = 3, 2
MAX_PATTERNS, MAX_KEY, MAX_MORE, MAX_TRAINING = 5, 12, 80, 3
PROJECT_MAX_PX = 20.0          # at 1920 px wide (soccer/coaching/evidence.py's disagreement rule)
TRAIL_S = 4.0
SKILLS = {'exposed_channel': ('wide_defender_support', 'someone_back_when_attacking'),
          'rotated_to_cover': ('wide_defender_support', 'someone_back_when_attacking'),
          'covered_advance': ('wide_defender_support', 'someone_back_when_attacking'),
          'stepped_out_open': ('back_line_coordination', 'pressure_and_cover'),
          'stepped_out_covered': ('back_line_coordination', 'pressure_and_cover'),
          'gap_attacker': ('back_line_coordination',), 'dropped_off': ('back_line_coordination',),
          'stepped_up_together': ('back_line_coordination',),
          'free_between_lines': ('back_line_coordination', 'far_side_protection'),
          'double_press_free': ('pressure_and_cover',), 'press_no_cover': ('pressure_and_cover',),
          'press_and_cover': ('pressure_and_cover',),
          'slow_reorganisation': ('someone_back_when_attacking', 'pressure_and_cover'),
          'recovery_run': ('someone_back_when_attacking', 'pressure_and_cover'),
          'carrier_isolated': ('spread_when_attacking',), 'offered_support': ('spread_when_attacking',),
          'run_in_behind': ('spread_when_attacking',), 'created_separation': ('spread_when_attacking',)}


# ---------------------------------------------------------------- stages
def _sha(*parts):
    h = hashlib.sha256()
    for p in parts:
        h.update(json.dumps(p, sort_keys=True, default=str).encode('utf-8'))
    return h.hexdigest()


def _code_sha():
    here = Path(__file__).resolve().parent
    h = hashlib.sha256()
    for p in sorted(list(here.glob('*.py'))+list((here/'detectors').glob('*.py'))):
        h.update(p.read_bytes())
    return h.hexdigest()[:16]


def _corrections(folder):
    """(key items, aliases, excluded, teams, intel snapshot): the match's people when they are built (fullmatch/intel.py:
    their fingerprint, the halftime's and the masks' join the key), else the coach's corrections."""
    try:
        from fullmatch import corrections as fc
        items = fc.load(folder)
    except Exception:   # noqa: BLE001 - a match without the module or the file has no corrections
        return [], {}, set(), {}, None
    try:
        from fullmatch import halftime as ht, intel
        snap = intel.snapshot(folder)
    except Exception:   # noqa: BLE001 - without the match's state the corrections alone apply
        snap = None
    if snap is not None and snap.get('state') is not None:
        aliases, excluded, teams = intel.aliases_of(snap['people'])
        key = {'corrections': items, 'people': snap['auto'].get('fingerprint'), 'halftime': ht.fingerprint(snap['halftime']),
               'masks': intel.masks_fingerprint(snap)}
        rec = possession_v2(folder, snap)
        if rec is not None:
            key['possession2'] = rec.get('fingerprint')
        key['goals'] = gl.signature(snap['state'])
        return key, aliases, excluded, teams, snap
    aliases, excluded, teams = fc.resolve_aliases(items)
    return items, aliases, excluded, teams, None


def possession_v2(folder, snap):
    """The match's possession v2 record for this snapshot (fullmatch/possession_match.py: controlled in-play
    possession over the whole match, through hidden-ball gaps), or None: then the segments' possession record stays the
    phase source."""
    try:
        from fullmatch import possession_match as pm
        return pm.latest(folder, snap)
    except Exception:   # noqa: BLE001 - without it the old record applies
        return None


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (np.floating,)):
        return None if not math.isfinite(float(x)) else float(x)
    if isinstance(x, float):
        return None if not math.isfinite(x) else x
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x


def _episode_ids(ep, data):
    """The episode with person indices replaced by match ids (so a cached episode survives a re-indexing)."""
    out = dict(ep)
    out['players'] = {g: [data.people[int(p)]['id'] for p in v] for g, v in ep['players'].items()}
    out['roles'] = {data.people[int(p)]['id']: v for p, v in ep['roles'].items()}
    out['id'] = 'mo-'+hashlib.sha256(f"{ep['detector']}|{ep['kind']}|{ep['team']}|{ep['tKey']:.2f}|{out['players']}".encode()).hexdigest()[:12]
    return _jsonable(out)


def analyse(folder, progress=None, cache=True):
    """(data, engine, episodes with match ids, timings): the match's data and every detector's episodes (cached)."""
    folder = Path(folder)
    timings = {}
    started = time.perf_counter()
    data = md.match_data(folder, progress=progress)
    timings['data'] = round(time.perf_counter()-started, 2)
    items, aliases, excluded, teams, snap = _corrections(folder)
    md.apply_corrections(data, aliases, excluded, teams)
    if snap is not None:
        # the report names people as the page shows them: the match's slots (fullmatch/intel.py), not the assembly's labels
        shown = (snap.get('people') or {}).get('people') or {}
        for p in data.people:
            label = (shown.get(p['id']) or {}).get('display')
            if label:
                p['display'] = label
        # the coach's halftime gives the directions; the stoppages, set pieces and halftime are no open play
        from fullmatch import gamestate as gs, halftime as ht
        ht.apply_to_data(data, snap['index'], snap['halftime'])
        data.open_play = gs.eligible(gs.mask_at(snap['masks'], data.t), 'open_play')
        rec = possession_v2(folder, snap)
        if rec is not None:
            md.apply_possession_v2(data, rec)
    started = time.perf_counter()
    engine = en.Engine(data)
    timings['context'] = round(time.perf_counter()-started, 2)
    key = _sha(data.meta['fingerprint'], items, ENGINE_VERSION, _code_sha())[:16]
    path = folder/'assembly'/'coaching'/'moments'/f'episodes-{key}.json'
    if cache and path.is_file():
        try:
            stored = json.loads(path.read_text(encoding='utf-8'))
            timings['detect'] = 0.0
            timings['detectCached'] = stored.get('seconds')
            return data, engine, stored['episodes'], timings, key
        except (OSError, ValueError, KeyError):
            pass
    started = time.perf_counter()
    raw = detect_all(engine)
    eps = [_episode_ids(e, data) for e in raw]
    timings['detect'] = round(time.perf_counter()-started, 2)
    if cache:
        _write(path, {'version': 1, 'key': key, 'seconds': timings['detect'], 'episodes': eps})
    return data, engine, eps, timings, key


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f'.{path.name}.{hashlib.sha256(str(time.time()).encode()).hexdigest()[:8]}.tmp')
    tmp.write_text(json.dumps(value, separators=(',', ':')), encoding='utf-8')
    for attempt in range(20):
        try:
            tmp.replace(path)
            return
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(.05*(attempt+1))


# ---------------------------------------------------------------- selection
def _dedup(eps):
    """One entry per situation: an episode whose key time is within DEDUP_S of a better one sharing a person, or within
    SEQUENCE_S of a better one of the same kind (one pressing sequence, one spell of the line stepping up), is kept only
    as that one's 'related' moment. CAN-ISL had three 'pressure with cover' strengths in 15 s of one sequence."""
    keep = []
    for e in sorted(eps, key=lambda e: -e['score']):
        people = {p for g in ('focus', 'support') for p in e['players'][g]}
        clash = False
        for k in keep:
            near = abs(k['tKey']-e['tKey'])
            shared = people & {p for g in ('focus', 'support') for p in k['players'][g]}
            if (near <= DEDUP_S and shared) or (near <= SEQUENCE_S and k['kind'] == e['kind']):
                clash = True
                k.setdefault('related', []).append(e['id'])
                break
        if not clash:
            keep.append(e)
    return sorted(keep, key=lambda e: e['tKey'])


def _patterns(eps):
    """{kind: [episode ids]} for kinds that happened PATTERN_MIN+ times PATTERN_SPACING_S apart."""
    by_kind = defaultdict(list)
    for e in sorted(eps, key=lambda e: e['tKey']):
        if by_kind[e['kind']] and e['tKey']-by_kind[e['kind']][-1]['tKey'] < PATTERN_SPACING_S:
            continue
        by_kind[e['kind']].append(e)
    return {k: v for k, v in by_kind.items() if len(v) >= PATTERN_MIN}


def _pick(eps, limit, pattern_of):
    out, kinds, pats = [], Counter(), Counter()
    for e in sorted(eps, key=lambda e: -e['score']):
        if kinds[e['kind']] >= MAX_PER_KIND:
            continue
        p = pattern_of.get(e['id'])
        if p and pats[p] >= MAX_PER_PATTERN:
            continue
        out.append(e)
        kinds[e['kind']] += 1
        if p:
            pats[p] += 1
        if len(out) >= limit:
            break
    return out


# ---------------------------------------------------------------- evidence
def _norm(data, x, y):
    return [round(float(x)/data.length, 4), round(float(y)/data.width, 4)]


def _frame_facts(data, s, ids):
    """Oriented homography, frame size, and {id: (box px, feet px, pitch xy)} of the people `ids` at sample s."""
    from soccer import field
    W, Hh = data.meta.get('videoSize') or [1920, 1080]
    h = data.H[s]
    H = field.orient(np.asarray(h, float).reshape(3, 3), W, Hh) if np.all(np.isfinite(h)) else None
    people = {}
    r = data.rows(s)
    for i in range(r.start, r.stop):
        pid = data.people[int(data.pid[i])]['id']
        if pid not in ids:
            continue
        b = data.box[i]
        if not np.all(np.isfinite(b)):
            continue
        px = [float(b[0]*W), float(b[1]*Hh), float(b[2]*W), float(b[3]*Hh)]
        people[pid] = (px, (px[0]+px[2]/2, px[1]+px[3]), (float(data.x[i]), float(data.y[i])))
    return H, (W, Hh), people


def _project(H, pts):
    if H is None:
        return None
    from soccer import field
    p = np.asarray(pts, float).reshape(-1, 2)
    if not (field.depth(H, p) > 1e-9).all():
        return None
    return field.project(H, p)


def _tier(data, s, H, size, people):
    """('world' / 'pinned' / 'image', projection error px at 1920): world when calibrated with confidence >= .8 and the
    involved people's pitch positions project within PROJECT_MAX_PX of their box feet."""
    if not data.calibrated[s] or float(data.calibConf[s]) < .5 or H is None:
        return 'image', None
    errs = []
    for box, feet, xy in people.values():
        q = _project(H, [xy])
        if q is not None:
            errs.append(float(np.hypot(q[0][0]-feet[0], q[0][1]-feet[1])))
    err = float(np.median(errs))*1920/size[0] if errs else None
    if float(data.calibConf[s]) >= .8 and err is not None and err <= PROJECT_MAX_PX:
        return 'world', round(err, 1)
    return 'pinned', None if err is None else round(err, 1)


def _trail(data, pid_index, s, seconds=TRAIL_S):
    pts = []
    s0 = data.sample_at(float(data.t[s])-seconds)
    for k in range(s0, s+1, 2):
        p = cm.where(data, k, pid_index)
        if p:
            pts.append((p[0], p[1]))
    return pts if len(pts) >= 3 and math.hypot(pts[-1][0]-pts[0][0], pts[-1][1]-pts[0][1]) >= 3.0 else None


def evidence(data, team, e, suggestion, caption):
    """The moment's evidence record for the page (see the module doc)."""
    s = int(e['sKey'])
    idx = data.index_of
    ids = {p for g in e['players'].values() for p in g}
    diagram, overlay = [], []
    r = data.rows(s)
    for i in range(r.start, r.stop):
        p = data.people[int(data.pid[i])]
        if p['id'] in ids and p['id'] not in e['players']['opponents']:
            continue
        diagram.append({'type': 'player', 'id': p['id'], 'team': p.get('team'), 'x': _norm(data, data.x[i], data.y[i])[0],
                        'y': _norm(data, data.x[i], data.y[i])[1], 'kind': 'O', 'emphasis': p['id'] in ids})
    tone = 'strength' if e['category'] == 'positive' else 'review'
    pos = {}
    for g in ('focus', 'support', 'opponents'):
        for pid in e['players'][g]:
            p = cm.where(data, s, idx[pid]) if pid in idx else None
            if p:
                pos[pid] = (p[0], p[1])
    for k, pid in enumerate(e['players']['focus'][:2]):
        if pid in pos:
            diagram.append({'type': 'focus', 'id': pid, 'team': team, 'x': _norm(data, *pos[pid])[0],
                            'y': _norm(data, *pos[pid])[1], 'tone': tone, 'secondary': k > 0})
    for pid in e['players']['support'][:3]:
        if pid in pos:
            diagram.append({'type': 'support', 'id': pid, 'team': team, 'x': _norm(data, *pos[pid])[0],
                            'y': _norm(data, *pos[pid])[1]})
    rel = []
    f0 = e['players']['focus'][0] if e['players']['focus'] else None
    if f0:
        for pid in e['players']['support'][:1]:
            rel.append({'type': 'relation', 'a': f0, 'b': pid, 'kind': 'support', 'ok': e['category'] == 'positive'})
        for pid in e['players']['opponents'][:2]:
            rel.append({'type': 'relation', 'a': f0, 'b': pid, 'kind': 'opponent', 'ok': e['category'] == 'positive'})
    diagram.extend(rel[:3])
    zones = e.get('zones') or {}
    if isinstance(zones.get('channel'), list) and len(zones['channel']) >= 3 and e['category'] == 'improvement':
        diagram.append({'type': 'channel', 'polygon': [_norm(data, x, y) for x, y in zones['channel']]})
    for pid in e['players']['focus'][:1]:
        if pid in idx:
            tr = _trail(data, idx[pid], s)
            if tr:
                diagram.append({'type': 'arrow', 'id': pid, 'points': [_norm(data, x, y) for x, y in tr]})
    if suggestion:
        diagram.append({'type': 'suggestedZone', 'polygon': [_norm(data, x, y) for x, y in suggestion['zone']],
                        'kind': 'zone', 'label': 'suggested area'})
        if suggestion.get('from'):
            diagram.append({'type': 'moveArrow', 'from': _norm(data, *suggestion['from']), 'to': _norm(data, *suggestion['point'])})
    diagram.append({'type': 'caption', 'text': caption})
    ctx_dir = data.direction(team, s)
    # the key frame's video overlay
    H, size, people = _frame_facts(data, s, ids)
    tier, err = _tier(data, s, H, size, people)
    for k, pid in enumerate(e['players']['focus'][:2]):
        if pid in people:
            overlay.append({'type': 'focus', 'id': pid, 'x': people[pid][1][0], 'y': people[pid][1][1], 'box': people[pid][0],
                            'tone': tone, 'secondary': k > 0})
    for pid in e['players']['support'][:3]:
        if pid in people:
            overlay.append({'type': 'support', 'id': pid, 'x': people[pid][1][0], 'y': people[pid][1][1], 'box': people[pid][0]})
    for pid in e['players']['opponents'][:2]:
        if pid in people:
            overlay.append({'type': 'player', 'id': pid, 'x': people[pid][1][0], 'y': people[pid][1][1], 'box': people[pid][0],
                            'kind': 'O', 'emphasis': True})
    if tier in ('world', 'pinned'):
        overlay.extend(r_ for r_ in rel[:3] if r_['a'] in people and r_['b'] in people)
    if tier == 'world':
        polys = []
        if any(d['type'] == 'channel' for d in diagram):
            polys.append(('channel', zones['channel']))
        if suggestion:
            polys.append(('suggestedZone', suggestion['zone']))
        for kind_, poly in polys:
            q = _project(H, poly)
            if q is not None and 0 <= q[:, 0].mean() <= size[0] and 0 <= q[:, 1].mean() <= size[1]:
                el = {'type': kind_, 'polygon': [[round(float(a), 1), round(float(b), 1)] for a, b in q]}
                if kind_ == 'suggestedZone':
                    el.update(kind='zone', label='suggested area')
                overlay.append(el)
        if suggestion and suggestion.get('from'):
            q = _project(H, [suggestion['from'], suggestion['point']])
            if q is not None:
                overlay.append({'type': 'moveArrow', 'from': [round(float(q[0][0]), 1), round(float(q[0][1]), 1)],
                                'to': [round(float(q[1][0]), 1), round(float(q[1][1]), 1)]})
    overlay.append({'type': 'caption', 'text': caption})
    t_key = float(e['tKey'])
    fps = data.fps
    frame = int(data.frame[s])
    follow = {'focus': [{'id': p, 'tone': tone, 'secondary': k > 0} for k, p in enumerate(e['players']['focus'][:2])],
              'support': [{'id': p, 'team': team} for p in e['players']['support'][:3]],
              'relations': [{'a': r_['a'], 'b': r_['b'], 'kind': r_['kind'], 'ok': r_['ok']} for r_ in rel[:3]]}
    return {'id': 'ev-'+e['id'][3:], 'time': round(t_key, 3), 'frame': frame, 'seek': round((frame+.5)/fps, 4), 'fps': fps,
            'tier': tier, 'projectionErrorPx': err,
            'window': {'t0': round(max(0.0, t_key-float(e['before'])), 3), 't1': round(t_key+float(e['after']), 3),
                       'key': round(t_key, 3)},
            'players': [p for g in ('focus', 'support', 'opponents') for p in e['players'][g]],
            'follow': follow,
            'diagram': {'preset': e['kind'], 'title': caption, 't': round(t_key, 3), 'frame': frame, 'team': team,
                        'attack': 'right' if ctx_dir > 0 else 'left', 'metresEstimated': False,
                        'pitch': {'length': data.length, 'width': data.width}, 'elements': diagram},
            'videoOverlay': {'width': size[0], 'height': size[1], 'tier': tier, 'elements': overlay}}


# ---------------------------------------------------------------- report pieces
def _profile(profile):
    level = int((profile or {}).get('level') or 5)
    return {'level': max(1, min(7, level)), 'band': lg.band(level), 'ageGroup': (profile or {}).get('ageGroup'),
            'format': (profile or {}).get('format') or '11v11', 'source': (profile or {}).get('source', 'default')}


def _drill(kind, prof):
    try:
        from soccer.coaching import drills as dr  # noqa: F401 - the library's own loader when present
    except Exception:   # noqa: BLE001
        pass
    path = Path(__file__).resolve().parent.parent/'coaching'/'drills.json'
    try:
        library = json.loads(path.read_text(encoding='utf-8'))['drills']
    except (OSError, ValueError, KeyError):
        return None, 'The drill library could not be read.'
    skills = SKILLS.get(kind, ())
    level, fmt = prof['level'], prof['format']
    best = None
    for d in library:
        prim = d.get('primary_skill')
        sec = d.get('secondary_skills') or []
        rank = next((k for k, sk in enumerate(skills) if sk == prim), None)
        rank2 = next((k for k, sk in enumerate(skills) if sk in sec), None)
        if rank is None and rank2 is None:
            continue
        levels = d.get('development_level') or []
        dist = min(abs(level-v) for v in levels) if levels else 9
        if dist > 1:
            continue
        fmt_ok = fmt in (d.get('recommended_format') or [])
        key = (dist, 0 if rank is not None else 1, rank if rank is not None else rank2, 0 if fmt_ok else 1)
        if best is None or key < best[0]:
            best = (key, d)
    if not best:
        return None, f'The drill library has no exercise for this at level {level}.'
    d = best[1]
    return {'id': d['drill_id'], 'name': d['name'], 'levels': d.get('development_level'),
            'minutes': (d.get('duration') or {}).get('minutes'), 'setup': (d.get('field_setup') or {}).get('notes'),
            'coachingPoints': (d.get('coaching_points') or [])[:3], 'source': d.get('source_reference'),
            'levelMatch': 'exact' if best[0][0] == 0 else 'nearest'}, None


def metrics(engine, team):
    """Team-shape numbers as SUPPORTING context (medians over settled possession spells with 8+ outfield players in
    view): width, length, back-line height, back line to midfield; the shapes by lines; the possession estimate."""
    data, ctx, roles = engine.data, engine.ctx[team], engine.roles[team]
    out = []
    for phase, code, name in (('own', 1, 'with the ball'), ('opp', 2, 'without the ball')):
        samples = np.flatnonzero(ctx.settled & (ctx.phase == code) & data.calibrated)
        wid, lng, line, gap = [], [], [], []
        for s in samples[::2]:
            idx, pids = ctx.players(int(s))
            if len(idx) < 8:
                continue
            d = ctx.depth(data.x[idx], int(s))
            lat = ctx.lateral(data.y[idx], int(s))
            wid.append(float(lat.max()-lat.min()))
            lng.append(float(d.max()-d.min()))
            recs = [engine.role(team, p, int(s)) for p in pids]
            dd = [d[k] for k, r_ in enumerate(recs) if r_ and r_['line'] == 'D']
            mm = [d[k] for k, r_ in enumerate(recs) if r_ and r_['role'] in ('DM', 'CM')]
            if len(dd) >= 3:
                line.append(float(np.median(dd)))
                if len(mm) >= 2:
                    gap.append(float(np.median(mm)-np.median(dd)))
        n = len(wid)
        for key, vals, label in (('width', wid, 'Team width'), ('length', lng, 'Team length'),
                                 ('lineHeight', line, 'Back-line height (from own goal)'),
                                 ('gapDM', gap, 'Back line to midfield')):
            if len(vals) >= 25:
                out.append({'key': key, 'phase': phase, 'label': f'{label} {name}', 'value': round(float(np.median(vals)), 1),
                            'unit': 'm', 'samples': len(vals)})
    shapes = sorted(roles.shapes().items(), key=lambda kv: -kv[1])[:3]
    total = sum(roles.shapes().values()) or 1
    return {'items': out, 'shapes': [{'shape': k, 'share': round(v/total, 2)} for k, v in shapes],
            'possession': (data.meta.get('possession') or {}).get('shares'),
            'note': ('Medians over the settled possession spells (as PitchIQ estimates who had the ball) with 8 or more outfield '
                     'players in view. Shapes by lines are counted in 60 s windows that saw 10 outfield players; '
                     'identities split by the camera can make a line look bigger.')}


def coverage(engine, team, eps_team):
    data, ctx = engine.data, engine.ctx[team]
    live = (ctx.phase != 3)
    named = (ctx.phase == 1) | (ctx.phase == 2)
    shares = {'calibrated': float(data.calibrated.mean()), 'deadBall': float((ctx.phase == 3).mean()),
              'possessionNamedOfLive': float(named.sum()/max(1, live.sum())),
              'settledOfLive': float(ctx.settled.sum()/max(1, live.sum())),
              'ballPlaced': float((data.ballState == 1).mean()), 'people': len(data.people)}
    views = [md.view_share(data.view[s], data.length, data.width, step=6.0) for s in range(0, data.samples, 50)]
    shares['viewMedian'] = float(np.median(views)) if views else 0.0
    items = [
        f"Calibrated frames: {shares['calibrated']*100:.0f}%. The camera showed a median {shares['viewMedian']*100:.0f}% of the "
        'pitch; areas it did not show were never judged (a moment needs the relevant area in view).',
        f"Who had the ball is PitchIQ\u2019s estimate from the tracking: named on {shares['possessionNamedOfLive']*100:.0f}% of "
        f"live play, settled (well inside one team\u2019s spell) on {shares['settledOfLive']*100:.0f}%. Checks that depend on "
        'it only use those stretches.',
        f"The ball was placed on the pitch in {shares['ballPlaced']*100:.0f}% of the frames; defensive judgements need it (or "
        'the named carrier) nearby.',
        f"Dead-ball time ({shares['deadBall']*100:.0f}% of the frames, as estimated) is not reviewed: set pieces are not "
        'assessed.',
        f"Players are named by their match label; the camera split identities into {shares['people']} labels, so one player "
        'can appear under several labels over the match (merge them in the roster to follow one player).',
        'Not assessed: passes and passing decisions, body orientation and scanning, cover shadows, pressing triggers, effort '
        'or intent. Roles are inferred from positions (not confirmed by the coach).']
    counts = Counter((e['category'], e['interpretation']) for e in eps_team)
    return {'shares': {k: (round(v, 3) if isinstance(v, float) else v) for k, v in shares.items()}, 'items': items,
            'candidates': {f'{a}|{b}': n for (a, b), n in sorted(counts.items())}}


def _summary(team_word, strengths, improvements, patterns, level, minutes):
    parts = [f"{team_word}: {minutes:.0f} minutes reviewed." if level >= 2 else f"{team_word}: the whole match reviewed."]
    if strengths:
        tops = Counter(e['kind'] for e in strengths).most_common(2)
        parts.append('Best: ' + '; '.join(lg.PATTERN_TITLES.get(k, k).lower() for k, _ in tops) + '.')
    else:
        parts.append('No strength met the evidence bar.')
    imp = [p for p in patterns if p['category'] == 'improvement']
    if imp:
        parts.append('To work on: ' + '; '.join(p['title'].lower() for p in imp[:2]) + '.')
    elif improvements:
        tops = Counter(e['kind'] for e in improvements).most_common(2)
        parts.append('To look at: ' + '; '.join(lg.PATTERN_TITLES.get(k, k).lower() for k, _ in tops) + '.')
    else:
        parts.append('Nothing to improve met the evidence bar (that is not proof that everything went well).')
    return ' '.join(parts)


# ---------------------------------------------------------------- build
def report_key(analysis_key, team, prof):
    return _sha(analysis_key, team, prof['level'], prof['ageGroup'], prof['format'], ENGINE_VERSION)[:16]


def expected_path(folder, team, profile=None):
    """Where the report for this match state, team and profile is (or would be) stored: computed without loading the
    match (the data fingerprint hashes the index, the views' sizes and the possession record)."""
    items = _corrections(folder)[0]
    akey = _sha(md.source_fingerprint(folder), items, ENGINE_VERSION, _code_sha())[:16]
    return report_path(folder, team, report_key(akey, team, _profile(profile)))


def cached(folder, team, profile=None):
    """The stored report for the match as it is now, or None."""
    path = expected_path(folder, team, profile)
    if path.is_file():
        try:
            return json.loads(path.read_text(encoding='utf-8'))
        except (OSError, ValueError):
            return None
    return None


def profile_from_settings(settings, team):
    """{level, ageGroup, format, source} from coaching settings (schema 'settings': teams.<letter>.playingLevel /
    ageGroup / format.players); a team without a block takes the other team's (a match is usually one level), else the
    defaults (level 5), and says so."""
    teams = (settings or {}).get('teams') or {}
    block, source = teams.get(team), 'team'
    if not isinstance(block, dict):
        block, source = teams.get('B' if team == 'A' else 'A'), 'other team'
    if not isinstance(block, dict):
        return {'level': 5, 'ageGroup': None, 'format': '11v11', 'source': 'default'}
    players = (block.get('format') or {}).get('players') if isinstance(block.get('format'), dict) else None
    return {'level': block.get('playingLevel') or 5, 'ageGroup': block.get('ageGroup'),
            'format': f'{players}v{players}' if players else '11v11', 'source': source}


def report_path(folder, team, key):
    return Path(folder)/'assembly'/'coaching'/'moments'/f'report-{team}-{key}.json'


def build(folder, team, profile=None, progress=None, cache=True, match_index=None):
    """The team's key-moments report (see the module doc)."""
    started = time.perf_counter()
    folder = Path(folder)
    prof = _profile(profile)
    data, engine, eps_all, timings, akey = analyse(folder, progress=progress, cache=cache)
    path = report_path(folder, team, report_key(akey, team, prof))
    if cache and path.is_file():
        try:
            return json.loads(path.read_text(encoding='utf-8'))
        except (OSError, ValueError):
            pass
    index = match_index or json.loads((folder/'assembly'/'match.json').read_text(encoding='utf-8'))
    kits = kt.team_kits(index)
    goals_section = None
    snap = _corrections(folder)[4]
    if snap is not None and snap.get('state') is not None:
        # every goal of the match state: what the team did well before the goals it scored, what went wrong before those
        # it conceded (soccer/moments/goals.py)
        found = gl.analyse(data, engine, eps_all, snap['state'], possession_v2(folder, snap))
        goals_section = gl.for_team(found, team, lg.Names(data, team, kits, {}), lg.clock)
    report = assemble(data, engine, eps_all, team, prof, kits, timings, akey, started, goals_section=goals_section)
    if cache:
        _write(path, report)
    return report


def _labels(data, moments):
    """{person id: the label the report gives that person} for everyone its moments involve: a full match's ids
    ('A-P281') are no labels; the page shows its own label for an id first (the roster as it is now) and these when it
    has none."""
    out = {}
    for m in moments.values():
        for ids in (m.get('players') or {}).values():
            for pid in ids if isinstance(ids, list) else []:
                i = data.index_of.get(pid)
                if i is not None and pid not in out:
                    out[pid] = data.people[i].get('display') or pid
    return out


def assemble(data, engine, eps_all, team, prof, kits, timings=None, akey='', started=None, goals_section=None):
    """The report of one team from the analysed match (data, engine, every episode with match ids): selection,
    suggestions, words, evidence, patterns, training, metrics, coverage. No file is read or written here."""
    started = started or time.perf_counter()
    timings = dict(timings or {})
    rkey = report_key(akey, team, prof)
    t_report = time.perf_counter()
    roles_now = {}
    eps = [e for e in eps_all if e['team'] == team and e['category'] in ('positive', 'improvement')]
    eps = _dedup(eps)
    good = [e for e in eps if e['reliability'] in MAIN_RELIABILITY]
    main = [e for e in good if e['interpretation'] in MAIN_INTERPRETATION]
    pats = _patterns(main)
    pattern_of = {}
    for kind, items in pats.items():
        for e in items:
            pattern_of[e['id']] = f'pt-{kind}'
    strengths = _pick([e for e in main if e['category'] == 'positive'], MAX_STRENGTHS, pattern_of)
    improvements = _pick([e for e in main if e['category'] == 'improvement'], MAX_IMPROVEMENTS, pattern_of)
    shown = {e['id'] for e in strengths+improvements}
    for items in pats.values():
        for e in sorted(items, key=lambda e: -e['score'])[:3]:
            shown.add(e['id'])
    more = sorted([e for e in good if e['id'] not in shown], key=lambda e: -e['score'])[:MAX_MORE]
    moments = {}
    for e in [e for e in good if e['id'] in shown]+more:
        ep = dict(e)
        for p, v in (ep.get('roles') or {}).items():
            if p in data.index_of:
                roles_now[data.index_of[p]] = v
        names = lg.Names(data, team, kits, roles_now)
        ep_idx = dict(ep, players={g: [data.index_of[p] for p in v if p in data.index_of] for g, v in ep['players'].items()})
        suggestion, alternatives = None, []
        if ep['category'] == 'improvement':
            ep_eng = dict(ep_idx, zones={k: (tuple(v) if isinstance(v, list) and len(v) == 2 and not isinstance(v[0], list) else v)
                                         for k, v in (ep.get('zones') or {}).items()})
            try:
                opts = sg.options(engine, ep_eng)
            except Exception:   # noqa: BLE001 - a suggestion that cannot be computed is left out, never guessed
                opts = []
            suggestion, alternatives = sg.choose(opts, prof['level'])
        words = lg.moment_text(ep_idx, names, prof['level'], suggestion, alternatives)
        sug_out = None
        if suggestion:
            sug_out = {'text': words['suggestion'], 'mover': data.people[suggestion['mover']]['id'],
                       'zone': [_norm(data, x, y) for x, y in suggestion['zone']], 'label': suggestion['label'],
                       'reachS': suggestion['reachS'], 'budgetS': suggestion['budgetS'],
                       'alternatives': [{'text': t, 'mover': data.people[a['mover']]['id'], 'label': a['label'],
                                         'zone': [_norm(data, x, y) for x, y in a['zone']]} for t, a in zip(words['alternatives'], alternatives)]}
        caption = words['finding'] if len(words['finding']) <= 90 else words['title']
        ev = evidence(data, team, ep, suggestion, caption)
        moments[ep['id']] = _jsonable({
            'id': ep['id'], 'detector': ep['detector'], 'kind': ep['kind'], 'category': ep['category'], 'team': team,
            'title': words['title'], 'finding': words['finding'], 'what': words['what'], 'why': words['why'],
            'assessment': words['assessment'], 'after': words.get('after'), 'suggestion': sug_out,
            'noSuggestion': (ep['category'] == 'improvement' and not sug_out) or None,
            'pointer': words['pointer'], 'players': ep['players'],
            'roles': {p: rl.role_word(v['role']) for p, v in (ep.get('roles') or {}).items()},
            'time': round(float(ep['tKey']), 3), 'clock': lg.clock(ep['tKey']),
            'half': next((p.get('name') for p in data.periods if p['start']-1 <= ep['tKey'] <= p['end']+1), None),
            'window': ev['window'], 'phase': ep['phase'], 'measures': ep['measures'], 'outcome': ep.get('outcome'),
            'reliability': {'observation': ep['reliability'], 'interpretation': ep['interpretation'],
                            'text': words['reliability'], 'reasons': ep.get('reasons') or []},
            'pattern': pattern_of.get(ep['id']), 'related': ep.get('related') or [], 'score': ep['score'],
            'uncertainContext': ep['interpretation'] not in MAIN_INTERPRETATION, 'evidence': ev})
    names = lg.Names(data, team, kits, roles_now)
    team_word = names.team_word()
    patterns = []
    for kind, items in sorted(pats.items(), key=lambda kv: -sum(e['score'] for e in kv[1])):
        cat = items[0]['category']
        halves = (sum(1 for e in items if (data.periods[0]['end'] >= e['tKey'])),
                  sum(1 for e in items if (data.periods[0]['end'] < e['tKey'])))
        title = lg.PATTERN_TITLES.get(kind, kind)
        count_text = (f"{len(items)} times" + (f" ({halves[0]} in the first half, {halves[1]} in the second)" if all(halves) else '')
                      if prof['level'] >= 2 else 'Several times')
        examples = [e['id'] for e in sorted(items, key=lambda e: -e['score'])[:3] if e['id'] in moments]
        drill, note = _drill(kind, prof) if cat == 'improvement' else (None, None)
        patterns.append({'id': f'pt-{kind}', 'kind': kind, 'category': cat, 'title': title,
                         'text': f"{count_text}. {lg.POINTERS.get(kind, {}).get(prof['band'], '')}", 'count': len(items),
                         'halves': list(halves), 'examples': examples, 'drill': drill, 'drillNote': note})
    patterns = patterns[:MAX_PATTERNS]
    training = []
    sources = [p for p in patterns if p['category'] == 'improvement']
    if not sources:
        sources = [{'kind': k, 'title': lg.PATTERN_TITLES.get(k, k), 'id': None}
                   for k, _ in Counter(e['kind'] for e in improvements).most_common(MAX_TRAINING)]
    for p in sources[:MAX_TRAINING]:
        drill, note = _drill(p['kind'], prof)
        training.append({'id': f"tr-{p['kind']}", 'title': p['title'], 'pattern': p.get('id'),
                         'objective': lg.POINTERS.get(p['kind'], {}).get(prof['band'], ''), 'drill': drill, 'drillNote': note})
    key_moments = sorted([e['id'] for e in strengths[:MAX_KEY//2]+improvements[:MAX_KEY//2]],
                         key=lambda i: moments[i]['time'])
    minutes = sum(float(p['end'])-float(p['start']) for p in data.periods)/60
    report = {
        'kind': KIND, 'version': 1, 'engine': {'version': ENGINE_VERSION, 'code': _code_sha(), 'analysis': akey, 'report': rkey},
        'match': {'matchId': data.meta.get('matchId'), 'video': data.meta.get('videoName'),
                  'periods': [{k: p.get(k) for k in ('index', 'name', 'start', 'end', 'attacksRight')} for p in data.periods],
                  'teams': {t: {'kit': kits.get(t), 'label': names.team_word(t)} for t in 'AB'}},
        'team': team, 'teamLabel': team_word, 'profile': prof,
        'summary': {'text': _summary(team_word, strengths, improvements, patterns, prof['level'], minutes)},
        'strengths': [e['id'] for e in strengths], 'improvements': [e['id'] for e in improvements],
        'keyMoments': key_moments, 'patterns': patterns, 'training': training,
        'moreMoments': [e['id'] for e in more if e['id'] in moments], 'moments': moments,
        'metrics': metrics(engine, team), 'coverage': coverage(engine, team, eps), 'goals': goals_section,
        'labels': _labels(data, moments),
        'counts': {'episodesAllTeams': len(eps_all), 'episodesTeam': len(eps), 'main': len(main), 'more': len(more),
                   'strengths': len(strengths), 'improvements': len(improvements), 'patterns': len(patterns)},
        'runtime': timings | {'report': round(time.perf_counter()-t_report, 2), 'total': round(time.perf_counter()-started, 2)},
        'generated': time.strftime('%Y-%m-%dT%H:%M:%S')}
    return _jsonable(report)
