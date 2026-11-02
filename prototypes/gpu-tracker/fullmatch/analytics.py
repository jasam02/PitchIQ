"""Match-level analytics from the assembled segments, counting every frame once (the frames between each segment's cuts).

Player samples (heatmaps, average positions, movement trails: soccer/analytics.py). Per segment the unchanged
analytics.Samples is built from the view's owned frames; its raw per-identity arrays (time, pitch x/y, identity and
pitch confidence, sigma) are appended in match order (assembly writes them to assembly/samples.npz). MatchSamples then
recomputes what Samples derives from those arrays (breaks, the smoothing per continuous run, the time weights) over the
whole match, so a match report is exactly what analytics.report would give on one result holding every owned frame:
sums of seconds per cell, weighted means, trails broken where the samples are (a halftime break is a gap like any
other). Physical coordinates are kept; orient='attack' gives every team attacking to the right in every period (the
period's attacking direction from the match index), as a separate view of the same samples.

Possession (soccer/possession_amateur.py dispatch, possession-4 where it judges the footage): each segment's view is
computed whole (its context helps the decoder at the edges), then trimmed to its owned frames; the match record joins
them: spans and changes concatenated, shares and coverage recomputed from the joined spans, the engines used listed.
"""
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

from soccer import analytics

FIELDS = ('t', 'x', 'y', 'identity', 'quality', 'sigma')


class SampleStore:
    """Collects the raw samples of the owned frames of each segment, in match order (assembly)."""

    def __init__(self):
        self.parts = {k: {} for k in FIELDS}
        self.excluded = Counter()
        self.meta = None

    def add(self, view, cut0, cut1):
        owned = dict(view)
        owned['frames'] = [f for f in view.get('frames') or [] if cut0 <= f['frame'] < cut1]
        if not owned['frames']:
            return
        s = analytics.Samples(owned)
        if self.meta is None:
            self.meta = {'length': s.length, 'width': s.width, 'fps': s.fps, 'start': s.start}
        self.meta['end'] = s.end
        self.excluded.update(s.excluded)
        for pid, tr in s.tracks.items():
            for key, source in (('t', 't'), ('x', 'rawX'), ('y', 'rawY'), ('identity', 'identity'), ('quality', 'quality'), ('sigma', 'sigma')):
                self.parts[key].setdefault(pid, []).append(np.asarray(tr[source]))

    def save(self, path, people):
        arrays = {}
        for key in FIELDS:
            for pid, chunks in self.parts[key].items():
                arrays[f'{pid}|{key}'] = np.concatenate(chunks).astype(np.float64 if key in ('t', 'x', 'y') else np.float32)
        meta = (self.meta or {'length': analytics.LENGTH, 'width': analytics.WIDTH, 'fps': 25.0, 'start': 0.0, 'end': 0.0}) | \
            {'people': people, 'excluded': dict(self.excluded)}
        np.savez_compressed(path, __meta__=np.frombuffer(json.dumps(meta).encode('utf-8'), np.uint8), **arrays)
        return path


class MatchSamples(analytics.Samples):
    """analytics.Samples over the whole match, built from assembly/samples.npz (see the module docstring)."""

    def __init__(self, path, periods=None, orient=None, min_pitch_confidence=analytics.MIN_PITCH_CONFIDENCE,
                 aliases=None, excluded_ids=(), teams=None, keep=None, breaks_at=(), people=None):
        """aliases / excluded_ids / teams: the coach's corrections (fullmatch/corrections.py) or the match's people
        (fullmatch/intel.aliases_of): a merged identity's samples join the identity it now is, an excluded one is left
        out, a re-teamed one counts for its new team. keep: times -> bool (fullmatch/intel.keep_function: the samples
        this kind of analysis counts; the others are kept on disk, not counted); breaks_at: times a trail never runs
        across (the halftime); people: {id: {'display', 'team', 'role'}} names for the people the aliases create."""
        data = np.load(path)
        meta = json.loads(bytes(data['__meta__']).decode('utf-8'))
        self.length, self.width, self.fps = float(meta['length']), float(meta['width']), float(meta['fps'])
        self.cap = max(analytics.SAMPLE_CAP, 2.5/self.fps)
        self.start, self.end = float(meta['start']), float(meta['end'])
        aliases, teams = aliases or {}, teams or {}
        self.people = {}
        for pid, info in meta['people'].items():
            final = aliases.get(pid, pid)
            if final in excluded_ids or pid in excluded_ids:
                continue
            if final not in self.people:
                base = meta['people'].get(final, info)
                self.people[final] = dict(base) | {'id': final} | ((people or {}).get(final) or {})
            if final in teams:
                self.people[final]['team'] = teams[final]
        self.min_pitch_confidence = min_pitch_confidence
        self.excluded = meta['excluded']
        self.orient = orient
        raw = {}
        for name in data.files:
            if name == '__meta__':
                continue
            pid, key = name.split('|')
            final = aliases.get(pid, pid)
            if final in excluded_ids or pid in excluded_ids:
                continue
            raw.setdefault(final, {}).setdefault(key, []).append(data[name])
        raw = {pid: {k: np.concatenate(v) for k, v in a.items()} for pid, a in raw.items()}
        self.tracks = {}
        for pid, a in raw.items():
            t = a['t'].astype(np.float64)
            x, y = a['x'].astype(np.float64), a['y'].astype(np.float64)
            if orient == 'attack' and periods and pid in self.people:
                x, y = attack_right(t, x, y, self.people[pid]['team'], periods)
            order = np.argsort(t, kind='stable')
            if keep is not None:
                order = order[keep(t[order])]
                if not len(order):
                    continue                 # nothing of this person's is counted by this kind of analysis
            t, x, y = t[order], x[order], y[order]
            c, q, sg = a['identity'][order].astype(np.float64), a['quality'][order].astype(np.float64), a['sigma'][order].astype(np.float64)
            why = analytics._breaks(t, x, y, self.length, self.width)
            for b in breaks_at or ():
                k = int(np.searchsorted(t, float(b)))
                if 0 < k < len(t) and why[k] == '':
                    why[k] = 'halftime'
            run = np.cumsum(why != '')
            sx, sy = x.copy(), y.copy()
            for k in np.unique(run):
                m = run == k
                sx[m], sy[m] = analytics._smooth(t[m], x[m], y[m], analytics.SMOOTH_WINDOW)
            self.tracks[pid] = {'t': t, 'x': sx, 'y': sy, 'w': analytics._weights(t, self.cap, 1/self.fps), 'run': run, 'why': why,
                                'rawX': x, 'rawY': y, 'identity': c, 'quality': q, 'sigma': sg}
        if keep is not None:
            self.people = {pid: p for pid, p in self.people.items() if pid in self.tracks}


def attack_right(t, x, y, team, periods):
    """Positions turned so that `team` attacks to the right in every period (a 180-degree turn in the periods where it
    defends the right goal). periods: [{'start', 'end', 'defendsLeft'}]; unknown directions are left as they are."""
    x, y = x.copy(), y.copy()
    for p in periods:
        left = p.get('defendsLeft')
        if left not in ('A', 'B') or team not in ('A', 'B') or left == team:
            continue
        m = (t >= float(p['start'])-1e-6) & (t <= float(p['end'])+1e-6)
        x[m], y[m] = 1-x[m], 1-y[m]
    return x, y


def roster_people(roster):
    """analytics.roster's shape from the match roster: numbered players and goalkeepers of team A or B."""
    out = {}
    for p in roster:
        if p.get('team') not in ('A', 'B') or str(p.get('display', '')).endswith('?'):
            continue
        out[p['matchId']] = {'id': p['matchId'], 'display': p['display'], 'team': p['team'], 'role': p['role']}
    return out


# ---- possession ------------------------------------------------------------------------------------------------------
def possession_segment(view, cut0, cut1, engine=None):
    """The possession record of one segment's view, trimmed to its owned frames [cut0, cut1)."""
    if engine is None:
        from soccer import possession_amateur as engine
    record = engine.compute(view)
    fps = float(record.get('fps') or (view.get('video') or {}).get('fps') or 25.0)
    times = {f['frame']: f['time'] for f in view.get('frames') or []}
    t_cut0 = times.get(cut0, cut0/fps)
    t_cut1 = times.get(cut1, times.get(cut1-1, (cut1-1)/fps)+1/fps)
    runs = []
    for r in record.get('timeline') or []:
        f0, f1 = r.get('f0'), r.get('f1')
        if f0 is None or f1 is None or f1 < cut0 or f0 >= cut1:
            continue
        r = dict(r)
        if f0 < cut0:
            r['f0'], r['t0'] = cut0, t_cut0
        if f1 >= cut1:
            r['f1'] = cut1-1
            r['t1'] = times.get(cut1-1, r['t1'])
        runs.append(r)
    changes = [c for c in record.get('changes') or [] if c.get('frame') is not None and cut0 <= c['frame'] < cut1]
    dead = []
    for d in record.get('deadSpells') or []:
        a, b = max(float(d['t0']), t_cut0), min(float(d['t1']), t_cut1)
        if b > a:
            dead.append(dict(d) | {'t0': round(a, 3), 't1': round(b, 3)})
    return {'version': record.get('version'), 'available': record.get('available'), 'reason': record.get('reason'),
            'fps': fps, 't0': t_cut0, 't1': t_cut1, 'timeline': runs, 'changes': changes, 'deadSpells': dead,
            'quality': record.get('quality'), 'basis': record.get('basis')}


def possession_with(record, aliases, excluded=()):
    """The match possession record with the coach's merges applied to the carriers (an excluded person carries nothing:
    such spans keep their team, without a carrier)."""
    rename = lambda pid: None if pid is None else (None if aliases.get(pid, pid) in excluded else aliases.get(pid, pid))
    out = dict(record)
    out['timeline'] = [dict(r) | {'carrier': rename(r.get('carrier'))} for r in record.get('timeline') or []]
    out['changes'] = [dict(c) | {'lostBy': rename(c.get('lostBy')), 'wonBy': rename(c.get('wonBy'))} for c in record.get('changes') or []]
    return out


def _share(runs):
    seconds = Counter()
    for r in runs:
        span = max(0.0, float(r['t1'])-float(r['t0']))
        state = r.get('state')
        key = r['team'] if r.get('team') in ('A', 'B') and state not in ('dead', 'unknown') else ('dead' if state == 'dead' else 'unknown')
        seconds[key] += span
    return {k: round(seconds.get(k, 0.0), 2) for k in ('A', 'B', 'dead', 'unknown')}


def merge_possession(parts, periods):
    """The match's possession record from the trimmed segment records (in match order)."""
    runs = [r for p in parts for r in p['timeline']]
    changes = [c for p in parts for c in p['changes']]
    dead = [d for p in parts for d in p['deadSpells']]
    shares = _share(runs)
    total = sum(shares.values()) or 1.0
    engines = Counter(p.get('version') or 'unavailable' for p in parts)
    available = any(p.get('available') for p in parts)
    notes = [f"{n} of {len(parts)} segments: {v}" for v, n in engines.most_common()]
    unavailable = [i for i, p in enumerate(parts) if not p.get('available')]
    if unavailable:
        notes.append(f'{len(unavailable)} segments have no possession estimate (see each one\'s reason).')
    return {'version': 'match-1', 'available': available, 'reason': None if available else 'No segment has a possession estimate.',
            'fps': parts[0]['fps'] if parts else 25.0, 't0': parts[0]['t0'] if parts else 0.0, 't1': parts[-1]['t1'] if parts else 0.0,
            'timeline': runs, 'changes': changes, 'deadSpells': dead, 'shares': shares,
            'quality': {'level': 'limited' if any((p.get('quality') or {}).get('level') == 'limited' for p in parts) else
                        ((parts[0].get('quality') or {}).get('level') if parts else 'unavailable'),
                        'coverage': round((shares['A']+shares['B']+shares['dead'])/total, 3), 'notes': notes},
            'engines': dict(engines), 'periods': periods,
            'segments': [{'t0': p['t0'], 't1': p['t1'], 'version': p.get('version'), 'available': p.get('available'),
                          'reason': p.get('reason')} for p in parts]}
