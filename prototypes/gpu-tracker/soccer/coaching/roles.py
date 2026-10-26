"""The role book (DESIGN v2 section 3.4): which tactical role each player had, when, how sure that is and where it came
from, kept apart from his global identity (spec Part 4: "do not assume a player's label such as B-07 automatically
means left back").

Why sources and confidence instead of one role per player. The audit and the feasibility study measured how far a
role read from positions can be trusted, and the answer depends on the team and the phase:
- Defending in a back four it is good: on Metrica game 1 the window role of a full-back agrees with his half role
  .91 / .84 (recall / precision, full tracking), a centre-back's .95 / .95, and the LB / RB holder changes on only 4%
  of 5 s window steps. Broadcast 0-60 s: both teams 'clear' with every role held for the whole minute.
- A team in possession reads in its attacking structure: Barcelona's full-backs read as LWB / LW, and across the three
  broadcast clips Barcelona had 229 s of its own possession and 0 s of it in a confident back-four reading (broadcast
  42-222 s: 4-3-3 ambiguous, the LB holder changing on 26% of window steps).
- A holding midfielder is identified about half the time (precision .43-.46 on team B of game 1).
So a role is a claim with a source, ordered by how much a criticism may lean on it:
  coach     the coach confirmed it for that time (settings 'roles'): confidence high. When the player's observed
            lateral position contradicts it over the span (a confirmed left-sided role with the median position on
            the right half, at least 10 s seen), `conflict` = 'confirmed_role_contradicted' and the overview asks
            the coach to check the setting; the role is still the coach's.
  stable    11v11, team_shape period 'clear' (high) or 'likely' (medium), the player's role in at least .7 of the
            period's windows where he was classified (two at least), and that role's holder changing on at most .15
            of consecutive window steps: role words may be used.
  carried   11v11: his stable role from the windows where the opponents had the ball (at least .6 of the window's
            samples), in the same period or within 120 s, carried into windows where his team had the ball (at least
            .5): medium; the text says "role read when defending". This is the feasibility study's answer to the
            possession team's full-back reading as a winger.
  window    team_shape's role in the 15 s window only: low; never named in a criticism.
  relative  small formats (4v4, 5v5, 7v7, 9v9: no team_shape template exists for them and DESIGN 2.5 forbids 11v11
            role names there), and 11v11 windows without a template role when the coach gave an intended formation:
            the players ranked by depth into the formation's back / mid / front blocks (profiles.blocks_of) and by
            lateral position within a block, over 15 s windows every 5 s of the 1 Hz track; medium when every outfield
            player was seen on at least .8 of the window and the blocks stand at least 2 m apart in mean depth (on
            EVAL's synthetic 7v7 scenes the closest blocks are p10 5.2 m apart, in the 4v4 "swarm" p90 0.7 m), else
            low ('back-1/3' = the left of a back three). Measured on the synthetic scenes against their planted
            roles: 7v7 (small field declared or assumed; the amateur scene with coach spans) 100% of the relative
            role-seconds right; the 4v4 swarm 30% (by design: a swarm has no positions to rank).
  none      nothing usable (no record is written: RoleIndex.at answers None).
Records are 5 s slices (the window step) merged while nothing changes; a coach span cuts whatever it covers.

Measured on Metrica game 1 (feasibility caches; full tracking with the truth phase, and the simulated camera with
possession-4's phase under the +-2 s exclusion), against each player's HALF ROLE (his modal window role over the half
on full tracking: what a coach would confirm; the same tracking, so consistency, not truth), per second of named
phase (team A 4-4-2 / team B back three):
                       full tracking A / B          camera A / B
  stable, own phase    .94 / .89 exact              .74 / .83
  stable, opp phase    .94 / .83                    .81 / .82
  carried, own phase   .60 / .54 (4226 / 2342 s)    .77 / .56 (250 / 720 s)
  window, own phase    .50 / .44                    .58 / .43
A carried role is better than a window role (+.10 to +.19) and well below a stable one, hence 'medium' with the
wording "role read when defending". For the question it exists for, the full-backs: on team A the half-role full-backs
in own possession read as LB / RB on 342 of 342 carried seconds (camera 56 of 56) against 135 of 201 window seconds
(camera 391 of 505). Most of the camera's own-possession player-seconds have no role at all (10054 of 17199 for team
A: no classified window), which is why D3 asks for a coach role on teams that rotate.

Interchanges (never a finding, only a suppressor of "out of position", DESIGN 3.4): on 11v11, a template role whose
holder changes between consecutive windows of one formation (the old and the new holder both classified in both
windows); on the relative slots, a slot whose holder changes after each held it for at least 3 s with at most 10 s
between them. Rotation recognition from positions relabels 4-25% of window steps even on full tracking, so these are
generous by design: they only stop a criticism, they never make one. A label swap alone is not cover either (fix round
1, critic S-4: 40-101 swaps per team and 10 minutes on Metrica full tracking, and overlap plants were "covered" by
them with the channel empty): the wide-defender check counts an interchange only when the player who took the
defender's slot stands in the vacated slot's geometry at that sample (contextual.interchange_cover).

Formation checks (11v11): per team_shape period, the measured formation and its confidence against the coach's intended
formation (the period's own when the coach set periods, else the intended one); `mismatch` when the reading is 'clear',
differs and is not a related pair (team_shape.RELATED): the overview then says the measured shape differs and
role-based criticism of players without a coach role is downgraded (assess.py). Small formats have no measured
formation (no templates), so no check.

The role tier reads only the obs file (INTERFACES.md LEAD section 4): everything here comes from
obs['roleInputs']['context'] (situations.SituationTimeline.role_inputs(), stored there by the detect stage) and the
coach's role settings. Every threshold is a PITCHIQ-PROPOSAL, unvalidated: no role truth exists anywhere (the Metrica
half roles are derived from the same tracking); the ROLE label sheet of DESIGN 13.3 is what can validate them.
"""
import math
import warnings
from collections import Counter

import numpy as np

from .. import team_shape as ts
from . import profiles as PR
from . import schema as S

STABLE_SHARE = .7               # of the period's classified windows in one role
STABLE_MIN_WINDOWS = 2          # windows the player must be classified in
STABLE_MAX_CHANGE = .15         # holder changes per consecutive window step of that role
STABLE_CONFIDENCE = {'clear': 'high', 'likely': 'medium'}
DEFEND_OPP_SHARE = .6           # a window read while defending: the opponents had the ball this share of its samples
ATTACK_OWN_SHARE = .5           # a window of own possession: this share of its samples
CARRY_HORIZON_S = 120.0         # s: defending windows this close (or in the same period) lend their stable role
SLICE_S = ts.WINDOW_STEP/2      # s: a window's own slice either side of its centre (2.5 s)
WINDOW_GAP_S = ts.WINDOW_STEP+1e-3   # consecutive windows are at most this far apart
RELATIVE_WINDOW_S = ts.WINDOW_SECONDS   # 15 s relative windows ...
RELATIVE_STEP_S = ts.WINDOW_STEP        # ... every 5 s
RELATIVE_PRESENT = .5           # share of a relative window's track points a player needs to be ranked
RELATIVE_MEDIUM = .8            # every outfield player seen this share of the window: medium confidence ...
RELATIVE_SEPARATION_M = 2.0     # ... and consecutive blocks at least this far apart in mean depth (coaching-scale m;
                                # synthetic 7v7 scenes p10 5.2-5.3 m, the 4v4 swarm p90 0.7 m)
INTERCHANGE_HOLD_S = 3.0        # s a relative slot must be held before and after a change
INTERCHANGE_GAP_S = 10.0        # s at most between the old holder's last and the new holder's first held second
CONFLICT_MIN_S = 10.0           # s of track inside a coach span before a contradiction is judged
CONFLICT_CODE = 'confirmed_role_contradicted'
SOURCE_ORDER = ('coach', 'stable', 'carried', 'window', 'relative', 'none')
GROUP_OF = {'GK': 'GK', 'LB': 'D', 'LCB': 'D', 'CB': 'D', 'RCB': 'D', 'RB': 'D', 'LWB': 'D', 'RWB': 'D',
            'LM': 'M', 'LCM': 'M', 'CM': 'M', 'RCM': 'M', 'RM': 'M', 'CDM': 'M', 'LDM': 'M', 'RDM': 'M', 'CAM': 'M',
            'LW': 'F', 'RW': 'F', 'CF': 'F', 'LCF': 'F', 'RCF': 'F'}   # a coach-confirmed name's group (wing-backs: D)
BLOCK_GROUP = {'back': 'D', 'mid': 'M', 'front': 'F'}
DEFAULT_OUTFIELD = {4: 4, 5: 4, 7: 6, 9: 8, 11: 10}   # when the role inputs do not say (schema defaults)


def role_group(role):
    """'D' / 'M' / 'F' / 'GK' of a role name (template, coach or relative slot), None for none."""
    if not role:
        return None
    if role in GROUP_OF:
        return GROUP_OF[role]
    return BLOCK_GROUP.get(role.split('-')[0])


def role_side(role):
    """'L' / 'C' / 'R' of a role name: template names by their first letter (LB, RCM; CB, CDM, CF central), relative
    slots by their place in the block (the first of n is the left, the n-th the right, any other central)."""
    if not role:
        return None
    if '-' in role and '/' in role:
        k, n = role.split('-')[1].split('/')
        k, n = int(k), int(n)
        return 'C' if n == 1 or (k != 1 and k != n) else ('L' if k == 1 else 'R')
    if role == 'GK':
        return 'C'
    return role[0] if role[0] in 'LR' else 'C'


def _slot_names(blocks):
    names = []
    for block, n in zip(('back', 'mid', 'front'), blocks):
        names += [f'{block}-{k}/{n}' for k in range(1, n+1)]
    return names


def blocks_11(formation):
    """Back / mid / front block sizes of an 11v11 template name ('4-2-3-1' -> (4, 5, 1)), None when unknown."""
    if formation not in ts.TEMPLATES:
        return None
    lines = ts.TEMPLATES[formation]['lines']
    return (lines[0], sum(lines[1:-1]), lines[-1])


def intended_at(formation, t0, t1):
    """The coach's intended formation over [t0, t1]: a formation period covering at least half of it, else the
    intended formation (None when none)."""
    if not isinstance(formation, dict):
        return None
    for p in formation.get('periods') or ():
        if isinstance(p, dict) and isinstance(p.get('t0'), (int, float)) and isinstance(p.get('t1'), (int, float)):
            a, b = max(float(p['t0']), t0), min(float(p['t1']), t1)
            if b > a and b-a >= .5*max(t1-t0, 1e-6):
                return p.get('formation')
    return formation.get('intended')


# ---------- slices -> records ----------
def _record(pid, t0, t1, role, source, confidence, support=None, period=None, ambiguous=(), conflict=None):
    return {'id': pid, 't0': round(float(t0), 2), 't1': round(float(t1), 2), 'role': role, 'group': role_group(role),
            'side': role_side(role), 'source': source, 'confidence': confidence,
            'support': None if support is None else round(float(min(max(support, 0.0), 1.0)), 3),
            'period': period, 'ambiguousWith': [a for a in dict.fromkeys(ambiguous) if a and a != role],
            'conflict': conflict}


def _merge(records):
    """Consecutive records of one player that touch and say the same thing become one."""
    out = []
    keys = ('id', 'role', 'source', 'confidence', 'period', 'ambiguousWith', 'conflict')
    for r in sorted(records, key=lambda r: (r['id'], r['t0'])):
        if out and all(out[-1][k] == r[k] for k in keys) and r['t0'] <= out[-1]['t1']+1e-6:
            prev = out[-1]
            if prev['support'] is not None and r['support'] is not None:
                prev['support'] = round(min(prev['support'], r['support']), 3)
            prev['t1'] = max(prev['t1'], r['t1'])
        else:
            out.append(dict(r, ambiguousWith=list(r['ambiguousWith'])))
    return out


def _cut(records, pid, a, b):
    """The player's records with [a, b] removed (a coach span takes its place)."""
    out = []
    for r in records:
        if r['id'] != pid or r['t1'] <= a or r['t0'] >= b:
            out.append(r)
            continue
        if r['t0'] < a:
            out.append(dict(r, t1=round(a, 2)))
        if r['t1'] > b:
            out.append(dict(r, t0=round(b, 2)))
    return out


# ---------- 11v11: team_shape windows ----------
def _holder_changes(wins, role):
    """(changes, pairs): consecutive windows (at most one step apart) where `role` was held in either, and those where
    its holder set changed."""
    changes = pairs = 0
    for w1, w2 in zip(wins, wins[1:]):
        if w2['centre']-w1['centre'] > WINDOW_GAP_S:
            continue
        h1 = {p for p, r in w1['roles'].items() if r == role}
        h2 = {p for p, r in w2['roles'].items() if r == role}
        if h1 or h2:
            pairs += 1
            changes += int(h1 != h2)
    return changes, pairs


def _modal(wins, pid):
    """(role, share, windows) of a player over windows (only where he was classified)."""
    c = Counter(w['roles'][pid] for w in wins if pid in w['roles'])
    if not c:
        return None, 0.0, 0
    role, n = c.most_common(1)[0]
    return role, n/sum(c.values()), sum(c.values())


def _stable_roles(wins, pids):
    """{pid: (role, share)} of the players whose modal role passes the stability rule over the windows."""
    out = {}
    rates = {}
    for pid in pids:
        role, share, n = _modal(wins, pid)
        if role is None or n < STABLE_MIN_WINDOWS or share < STABLE_SHARE-1e-9:
            continue
        if role not in rates:
            ch, pairs = _holder_changes(wins, role)
            rates[role] = ch/pairs if pairs else 0.0
        if rates[role] <= STABLE_MAX_CHANGE+1e-9:
            out[pid] = (role, share)
    return out


def _alt_modal(wins, pid, role):
    c = Counter(w['alt'].get(pid) for w in wins if w.get('alt') and w['alt'].get(pid))
    if not c:
        return []
    alt = c.most_common(1)[0][0]
    return [alt] if alt != role else []


def _template_records(ctx, seg):
    windows = sorted(ctx.get('windows') or [], key=lambda w: w['centre'])
    periods = {p['index']: p for p in ctx.get('periods') or []}
    pids = list(ctx.get('ids') or [])
    by_period = {}
    for w in windows:
        p = periods.get(w.get('period'))
        if p is not None and p.get('label') == w.get('label'):
            by_period.setdefault(p['index'], []).append(w)
    stable = {}
    for k, wins in by_period.items():
        p = periods[k]
        if p.get('confidence') in STABLE_CONFIDENCE:
            stable[k] = _stable_roles(wins, pids)
    defending = [w for w in windows if w.get('opp', 0.0) >= DEFEND_OPP_SHARE-1e-9]
    records = []
    for w in windows:
        c = float(w['centre'])
        a = seg[0] if float(w['t0']) <= seg[0]+1e-6 else max(seg[0], c-SLICE_S)
        b = seg[1] if float(w['t1']) >= seg[1]-1e-6 else min(seg[1], c+SLICE_S)
        if b <= a:
            continue
        k = w.get('period')
        p = periods.get(k) or {}
        st = stable.get(k, {})
        attacking = w.get('own', 0.0) >= ATTACK_OWN_SHARE-1e-9
        carried = {}
        if attacking:
            near = [d for d in defending if d.get('period') == k and abs(float(d['centre'])-c) <= CARRY_HORIZON_S+1e-6]
            carried = _stable_roles(near, [pid for pid in pids if pid not in st]) if near else {}
        for pid in pids:
            if pid in st:
                role, share = st[pid]
                amb = _alt_modal(by_period.get(k, []), pid, role) if p.get('related') else []
                records.append(_record(pid, a, b, role, 'stable', STABLE_CONFIDENCE[p['confidence']], share, k, amb))
            elif pid in carried:
                role, share = carried[pid]
                records.append(_record(pid, a, b, role, 'carried', 'medium', share, k))
            elif pid in w['roles']:
                role = w['roles'][pid]
                alt = w.get('alt', {}).get(pid)
                records.append(_record(pid, a, b, role, 'window', 'low', None, k, [alt] if alt else ()))
    return records


def _template_interchanges(ctx):
    out = []
    windows = sorted(ctx.get('windows') or [], key=lambda w: w['centre'])
    for w1, w2 in zip(windows, windows[1:]):
        if w2['centre']-w1['centre'] > WINDOW_GAP_S or w1.get('label') != w2.get('label'):
            continue
        both = set(w1['roles']) & set(w2['roles'])
        h1 = {r: p for p, r in w1['roles'].items() if p in both}
        h2 = {r: p for p, r in w2['roles'].items() if p in both}
        for role in sorted(set(h1) & set(h2)):
            if h1[role] != h2[role]:
                out.append({'t': round(float(w2['centre'])-SLICE_S, 2), 'out': h1[role], 'in': h2[role], 'slot': role})
    return out


# ---------- relative roles (small formats; 11v11 fallback with an intended formation) ----------
def _track(ctx):
    tr = ctx.get('track') or {}
    t = np.asarray(tr.get('t') or [], float)
    pids = list(ctx.get('ids') or [])
    K, N = len(t), len(pids)
    X, Y = np.full((K, N), np.nan), np.full((K, N), np.nan)
    for j in range(min(N, len(tr.get('x') or []))):
        X[:, j] = [np.nan if v is None else v for v in tr['x'][j]]
        Y[:, j] = [np.nan if v is None else v for v in tr['y'][j]]
    return t, pids, X, Y


def _rank(X, Y, cols, blocks):
    """{col: slot} ranking the columns by depth into blocks and by lateral position within a block."""
    order = sorted(cols, key=lambda j: X[j])
    names = {}
    i = 0
    for block, n in zip(('back', 'mid', 'front'), blocks):
        members = sorted(order[i:i+n], key=lambda j: Y[j])
        for k, j in enumerate(members, 1):
            names[j] = f'{block}-{k}/{n}'
        i += n
    return names


def _relative_records(ctx, seg, blocks_at, covered):
    """Relative slots over 15 s windows of the track; blocks_at(t0, t1) gives the block sizes (None: skip); covered
    {pid: [(a, b)]} are slices already holding a better role (skipped)."""
    t, pids, X, Y = _track(ctx)
    out = []
    if not len(t) or not pids:
        return out
    outfield = int(ctx.get('outfield') or len(pids))
    c = float(t[0])+RELATIVE_WINDOW_S/2
    end = float(t[-1])
    last = False
    while not last:
        lo, hi = c-RELATIVE_WINDOW_S/2, c+RELATIVE_WINDOW_S/2
        last = hi >= end-1e-6
        rows = np.flatnonzero((t >= lo-1e-6) & (t <= hi+1e-6))
        blocks = blocks_at(lo, hi)
        if len(rows) and blocks is not None and sum(blocks) == outfield:
            seen = np.isfinite(X[rows]) & np.isfinite(Y[rows])
            share = seen.mean(axis=0)
            cols = [j for j in range(len(pids)) if share[j] >= RELATIVE_PRESENT-1e-9]
            if len(cols) == outfield:
                # (a player of `pids` not seen in this window gives a NaN mean; numpy warns "Mean of empty slice"
                # for it: CODE round-2 L-7, silenced, the values are unchanged)
                with np.errstate(invalid='ignore'), warnings.catch_warnings():
                    warnings.simplefilter('ignore', RuntimeWarning)
                    mx, my = np.nanmean(X[rows], axis=0), np.nanmean(Y[rows], axis=0)
                slots = _rank(mx, my, cols, blocks)
                depth = sorted(float(mx[j]) for j in cols)
                cuts = np.cumsum(blocks)[:-1]
                gap = min((depth[k]-depth[k-1] for k in cuts), default=np.inf)
                conf = 'medium' if min(share[j] for j in cols) >= RELATIVE_MEDIUM-1e-9 and                     gap >= RELATIVE_SEPARATION_M-1e-9 else 'low'
                a = seg[0] if lo <= float(t[0])+1e-6 else max(seg[0], c-SLICE_S)
                b = seg[1] if last else min(seg[1], c+SLICE_S)
                for j, slot in slots.items():
                    pid = pids[j]
                    if any(x0 <= (a+b)/2 <= x1 for x0, x1 in covered.get(pid, ())):
                        continue
                    out.append(_record(pid, a, b, slot, 'relative', conf, float(share[j])))
        c += RELATIVE_STEP_S
    return out


def _relative_interchanges(ctx, blocks_at):
    t, pids, X, Y = _track(ctx)
    out = []
    if not len(t) or not pids:
        return out
    outfield = int(ctx.get('outfield') or len(pids))
    step = float((ctx.get('track') or {}).get('step') or 1.0)
    holder = {}                       # slot -> [(k, pid)]
    for k in range(len(t)):
        cols = [j for j in range(len(pids)) if np.isfinite(X[k, j]) and np.isfinite(Y[k, j])]
        blocks = blocks_at(float(t[k]), float(t[k]))
        if blocks is None or len(cols) != outfield or sum(blocks) != outfield:
            continue
        for j, slot in _rank(X[k], Y[k], cols, blocks).items():
            holder.setdefault(slot, []).append((k, pids[j]))
    need = max(1, int(round(INTERCHANGE_HOLD_S/step)))
    for slot, seq in holder.items():
        runs = []                     # (pid, k0, k1) of held runs (consecutive points, same holder)
        for k, pid in seq:
            if runs and runs[-1][0] == pid and k == runs[-1][2]+1:
                runs[-1][2] = k
            else:
                runs.append([pid, k, k])
        held = [r for r in runs if r[2]-r[1]+1 >= need]
        for r1, r2 in zip(held, held[1:]):
            if r1[0] != r2[0] and (t[r2[1]]-t[r1[2]]) <= INTERCHANGE_GAP_S+1e-6:
                out.append({'t': round(float(t[r2[1]]), 2), 'out': r1[0], 'in': r2[0], 'slot': slot})
    return sorted(out, key=lambda i: (i['t'], i['slot']))


# ---------- coach roles ----------
def _coach_records(ctx, roles, seg, pids_ok):
    t, pids, X, Y = _track(ctx)
    col = {pid: j for j, pid in enumerate(pids)}
    W = float((ctx.get('scale') or {}).get('widthM') or 68.0)
    out = []
    for pid, spans in sorted((roles or {}).items()):
        if pid not in pids_ok or not isinstance(spans, list):
            continue
        for sp in spans:
            if not isinstance(sp, dict) or not sp.get('role'):
                continue
            a = float(sp['from']) if isinstance(sp.get('from'), (int, float)) else seg[0]
            b = float(sp['to']) if isinstance(sp.get('to'), (int, float)) else seg[1]
            a, b = max(a, seg[0]), min(b, seg[1])
            if b < a:
                continue
            conflict = None
            side = role_side(sp['role'])
            j = col.get(pid)
            if j is not None and side in ('L', 'R') and len(t):
                m = (t >= a-1e-6) & (t <= b+1e-6) & np.isfinite(Y[:, j])
                step = float((ctx.get('track') or {}).get('step') or 1.0)
                if m.sum()*step >= CONFLICT_MIN_S-1e-9:
                    med = float(np.median(Y[m, j]))
                    if (side == 'L' and med > W/2) or (side == 'R' and med < W/2):
                        conflict = CONFLICT_CODE
            out.append(_record(pid, a, b, sp['role'], 'coach', 'high', None, None, (), conflict))
    return out


# ---------- the stage ----------
def build_rolebook(obs, role):
    """The RoleBook (schema 'rolebook') of an obs file's team from its role inputs (obs['roleInputs']['context']) and
    the coach's role settings role = {'roles': {pid: [RoleSpan]}, 'formation': {'intended', 'periods'}} (the role tier
    stage, coaching.build_roles)."""
    team = obs['team']
    tl = obs.get('timeline') or {}
    fmt_name = tl.get('format') or '11v11'
    players = int(fmt_name.split('v')[0])
    ctx = (obs.get('roleInputs') or {}).get('context') or {}
    seg = (float(tl.get('t0', 0.0)), float(tl.get('t1', 0.0)))
    if ctx.get('segment'):
        seg = (float(ctx['segment']['t0']), float(ctx['segment']['t1']))
    role = role if isinstance(role, dict) else {}
    formation = role.get('formation') if isinstance(role.get('formation'), dict) else {'intended': None, 'periods': []}
    outfield = int(ctx['outfield']) if isinstance(ctx.get('outfield'), int) else DEFAULT_OUTFIELD.get(players, players-1)
    fmt = {'players': players, 'goalkeeper': players-outfield == 1}
    pids = list(ctx.get('ids') or [])
    gks = [pid for pid, p in (tl.get('players') or {}).items() if p.get('team') == team and p.get('goalkeeper')]
    pids_ok = set(pids)|set(gks)
    if players == 11:
        records = _template_records(ctx, seg)
        interchanges = _template_interchanges(ctx)

        def blocks_at(a, b):
            return blocks_11(intended_at(formation, a, b))
        covered = {}
        for r in records:
            covered.setdefault(r['id'], []).append((r['t0'], r['t1']))
        if formation.get('intended') or formation.get('periods'):
            records += _relative_records(ctx, seg, blocks_at, covered)
        model = 'team_shape'
    else:
        def blocks_at(a, b):
            return PR.blocks_of(fmt, intended_at(formation, a, b))
        records = _relative_records(ctx, seg, blocks_at, {})
        interchanges = _relative_interchanges(ctx, blocks_at)
        model = 'relative'
    coach = _coach_records(ctx, role.get('roles'), seg, pids_ok)
    for r in coach:
        records = _cut(records, r['id'], r['t0'], r['t1'])
    records = _merge(records+coach)
    checks = []
    if players == 11:
        for p in ctx.get('periods') or []:
            intended = intended_at(formation, float(p['t0']), float(p['t1']))
            intended = intended if intended in S.FORMATIONS_11 else None
            measured = p.get('label')
            pair = frozenset((measured, intended)) if measured and intended else None
            related = bool(pair is not None and pair in ts.RELATED)
            checks.append({'period': int(p['index']), 't0': float(p['t0']), 't1': float(p['t1']), 'measured': measured,
                           'measuredConfidence': p.get('confidence'), 'intended': intended,
                           'mismatch': bool(p.get('confidence') == 'clear' and measured and intended and
                                            measured != intended and not related),
                           'related': related})
    return {'team': team, 'format': fmt_name, 'model': model, 'records': records,
            'interchanges': sorted(interchanges, key=lambda i: (i['t'], i['slot'])), 'formationChecks': checks}


# ---------- reading a role book (DETECT, EVIDENCE, TEXT) ----------
class RoleIndex:
    """Fast questions to a RoleBook: the role of a player at a time, the holders of some roles, the interchanges near a
    time, whether the measured formation contradicts the coach's plan."""

    def __init__(self, rolebook):
        self.book = rolebook
        self.by_id = {}
        for r in rolebook.get('records') or ():
            self.by_id.setdefault(r['id'], []).append(r)
        for v in self.by_id.values():
            v.sort(key=lambda r: (r['t0'], SOURCE_ORDER.index(r['source'])))
        self.interchanges = list(rolebook.get('interchanges') or ())
        self.checks = list(rolebook.get('formationChecks') or ())

    def at(self, pid, t):
        """The player's record at time t (the most trusted source when two touch), None when nothing covers it."""
        best = None
        for r in self.by_id.get(pid, ()):
            if r['t0']-1e-6 <= t <= r['t1']+1e-6 and (best is None or
                                                      SOURCE_ORDER.index(r['source']) < SOURCE_ORDER.index(best['source'])):
                best = r
        return best

    def holders(self, t, roles=None, sources=None):
        """{pid: record} of the players whose record at t has one of `roles` (any when None) from one of `sources`."""
        out = {}
        for pid in self.by_id:
            r = self.at(pid, t)
            if r is None or r['role'] is None:
                continue
            if roles is not None and r['role'] not in roles:
                continue
            if sources is not None and r['source'] not in sources:
                continue
            out[pid] = r
        return out

    def interchanges_near(self, t, horizon=INTERCHANGE_GAP_S, pid=None):
        """Interchanges within `horizon` seconds of t (involving pid when given)."""
        return [i for i in self.interchanges if abs(i['t']-t) <= horizon+1e-6 and
                (pid is None or pid in (i['out'], i['in']))]

    def mismatch_at(self, t):
        """The formation check covering t when it is a mismatch (clear reading against the coach's plan), else None."""
        for c in self.checks:
            if c['t0']-1e-6 <= t <= c['t1']+1e-6 and c['mismatch']:
                return c
        return None


__all__ = ['build_rolebook', 'RoleIndex', 'role_group', 'role_side', 'blocks_11', 'intended_at', 'SOURCE_ORDER',
           'CONFLICT_CODE']
