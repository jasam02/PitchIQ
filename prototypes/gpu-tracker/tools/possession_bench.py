"""Benchmark of the possession engine (soccer/possession.py): the pre-declared metrics on the Metrica possession truth
(tools/possession_truth.py) under full tracking and a simulated broadcast camera (tools/possession_sim.py: two seeds and
three dead-ball fills), the accuracy/coverage curve of the operating point, the ablations (each v3 and v4 rule off among
them, and the possession-3 rules), a registration hole, short windows, the one-touch possessions under both conventions,
the in-play seconds called dead and the dead seconds missed, and the hand-labelled PitchIQ runs. Every Metrica and
hand-label number in soccer/possession.py's docstring comes from this script.

    python -I -B tools/possession_bench.py --tracking DIR --events DIR [--quick] [--json out.json]      game 1 (TUNE)
    python -I -B tools/possession_bench.py --tracking DIR --events DIR --heldout --frozen NOTE          game 2
    python -I -B tools/possession_bench.py --tracking DIR --events DIR --heldout --game 3 --frozen NOTE   game 3 (EPTS)
    python -I -B tools/possession_bench.py --labels tools/annotations/possession --run BC180=RUN/result.json [--run ...]

--tracking holds the Metrica sample tracking (Sample_Game_<n>_RawTrackingData_{Home,Away}_Team.csv), --events its
events (Sample_Game_<n>_RawEventsData.csv): downloaded files, so run with -I -B (the repository root and tools/ are
added to sys.path explicitly; no data folder ever is). --heldout reads game 2 (or --game N) and refuses to run unless the
freeze note given by --frozen exists: a held-out game is read once, after the engine is frozen (game 3 is Metrica's EPTS
release, one folder passed as both --tracking and --events: tools/possession_truth.py reads it). --run scores a PitchIQ result against the hand
labels of the same name in --labels (BC60, BC180, AM, VEO: tools/annotations/possession/README.md); the videos are not
needed.

What is measured (PREDECLARED): 1 frame accuracy (truth in-play frames where the engine names a team: the team is right),
2 coverage (truth in-play frames with a named team), 3 changes (one-to-one matching of wonAt within 1 s and 2 s with the
same new team: precision, recall, F1; median wonAt / lostAt error; who lost / won where both name a player), 4 dead ball
(truth-dead frames called dead; engine-dead frames truly dead). Changes are scored against every truth change (Metrica
codes a first-time interception as a gain: one-touch possessions in), without the one-touch possessions' change in and
out (the truth's one_touch=False), and 'hybrid' (precision against every change, recall over the changes outside
one-touch possessions; defined after the plan, reported for context only). The engine is scored from its own record
(the timeline runs expanded to frames, the changes as listed) on consecutive windows of --window seconds, each a
separate result as a PitchIQ run is. The hand labels are points every 0.5 s (team, dead, unclear), changes (lostAt,
wonAt) and dead spells: accuracy / coverage on the team points, dead points called dead and engine-dead points labelled
dead, changes matched as above, the dead spells' overlap with the engine's.
"""
import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'tools'))
from soccer import possession  # noqa: E402

VARIANTS = {   # name: (mode, dead_fill, seed, noise)
    'full': ('full', 'hold', 0, 'camera'),
    'camera': ('camera', 'hold', 0, 'camera'),
    'camera-seed1': ('camera', 'hold', 1, 'camera'),
    'camera-visible': ('camera', 'visible', 0, 'camera'),
    'camera-none': ('camera', 'none', 0, 'camera'),
    'camera-independent': ('camera', 'hold', 0, 'independent'),
}
CURVE = (0.0, .5, .6, .7, .8, .9, .95, .99)
V3_RULES = {'live_hold': False, 'keeper_hands': False, 'knock_past': False, 'origin_fallback': False, 'fp_rows': 1.0,
            'still_since': False, 'cut_fresh': False, 'replays': False, 'conf_rule': 'v3'}   # (features: kit_team has no
                                                                                             # effect on a simulated game)
V2_RULES = dict(V3_RULES, start_on='any', won_by='first', won_w=1.0, conf_rule='v2', carry_by='taker', keep_prev=False,
                restart_gap=True, out_consistent=False, slow_s=float('inf'))
ABLATIONS = {
    'switch .10': {'switch': .10}, 'switch .20': {'switch': .20}, 'switch .30': {'switch': .30},
    'no image agreement': {'use_image': False}, 'no nearTrack': {'near_track': False},
    'FAST: blocked': {'fast': 'block'}, 'FAST: like any ball': {'fast': 'none'},
    'no restart taker': {'restart_w': 0.0}, 'no gap restart': {'gap': 1e9}, 'no carry over gaps': {'carry_gap': False},
    'no still-ball stoppage': {'still_s': 1e9}, 'pitch radius 1.0-2.0 m': {'r0': 1.0, 'r1': 2.0},
    'v2 rules (possession-2)': V2_RULES, 'v3 rules (possession-3)': V3_RULES,
    'fix 1 off: changes start on any evidence': {'start_on': 'any'},
    'fix 1 strict: observed frames only, decoded on them': {'start_on': 'observed', 'decode_on': 'observed'},
    'fix 1 alt: observed frames only': {'start_on': 'observed'},
    'fix 2 off: every excursion is out of play': {'out_consistent': False, 'carry_by': 'taker', 'restart_gap': True},
    'fix 3 off: no slowdown stoppage': {'slow_s': float('inf'), 'keep_prev': False},
    'fix 4 off: wonBy the first evidence': {'won_by': 'first', 'won_w': 1.0},
    'fix 5 off: v2 change confidence': {'conf_rule': 'v2'},
    'v4 fix 2 off: a held ball can be a stoppage': {'live_hold': False}, 'v4 fix 3 off: no keeper hands': {'keeper_hands': False},
    'v4 fix 4 off: a ball running past is his': {'knock_past': False}, 'v4 fix 7 off: still ball against the unseen frame': {'still_since': False},
    'v4 fix 10 off: high / medium / low': {'conf_rule': 'v3'}, 'knock-past 1.0 m': {'pass_m': 1.0}, 'knock-past 2.0 m': {'pass_m': 2.0},
    'hold look .5 s': {'hold_look': .5}, 'hold look 2 s': {'hold_look': 2.0},
    'activity median': {'act_q': .5}, 'activity of each team': {'act_teams': 'both'},
}
TRUTH_TYPES = {'interception': ('BALL LOST/INTERCEPTION', 'BALL LOST/HEAD-INTERCEPTION', 'BALL LOST/CROSS-INTERCEPTION'),
               'tackle_or_duel': ('BALL LOST/THEFT', 'CHALLENGE/'), 'out_of_play': ('BALL OUT', 'SHOT/', 'BALL LOST/')}

_TRUTH = {}


def _game(tracking, events, game):
    import possession_truth as mt
    key = (str(tracking), str(events), game)
    if key not in _TRUTH:
        _TRUTH[key] = mt.load_game(game, tracking, events)
    return _TRUTH[key]


def windows(g, seconds):
    t0, t1 = float(g['time'][0]), float(g['time'][-1])
    out, a = [], t0
    while a < t1:
        out.append((a, min(t1, a+seconds)-1e-6))
        a += seconds
    return out


def build(args):
    """Features of every window of one simulated variant (run in a worker process)."""
    tracking, events, game, variant, seconds = args
    import possession_sim as cs
    g = _game(tracking, events, game)
    mode, fill, seed, noise = VARIANTS[variant]
    sim = cs.simulate(g, mode=mode, fps=30, seed=seed, dead_fill=fill, noise=noise)
    Fs = []
    for a, b in windows(g, seconds):
        r = sim.result(a, b)
        if r['frames']:
            Fs.append(possession.features(r))
    return variant, Fs


def expand(rec, frames):
    """Per-frame team codes of a record on the result's absolute frame indices: 1 A, 2 B, 0 dead, -1 no team named."""
    out = np.full(len(frames), -1, int)
    for run in rec['timeline']:
        a = int(np.searchsorted(frames, run['f0']))
        b = int(np.searchsorted(frames, run['f1'], side='right'))
        out[a:b] = 0 if run['state'] == 'dead' else {'A': 1, 'B': 2}.get(run['team'], -1)
    return out


def run_engine(Fs, settings=None, abstained=None):
    """(times, codes, changes) of the engine's records over the windows; abstained (a list) collects the reasons of the
    windows the engine declined."""
    T, C, changes = [], [], []
    for F in Fs:
        rec = possession.from_features(F, settings)
        T.append(F['t'])
        C.append(expand(rec, F['frame']))
        changes += rec['changes']
        if not rec['available'] and abstained is not None:
            abstained.append(rec['reason'])
    return np.concatenate(T), np.concatenate(C), changes


def match(g, changes, one_touch, lo, hi, tol=2.0):
    """possession_truth.score's greedy one-to-one matching: (pairs, missed truth changes)."""
    tc = [c for c in g['changes'] if lo <= c['wonTime'] <= hi and (one_touch or not (c.get('oneTouch') or c.get('afterOneTouch')))]
    to = {'A': 'Home', 'B': 'Away'}
    cand = sorted((abs(x['wonAt']-y['wonTime']), i, j) for i, x in enumerate(changes) for j, y in enumerate(tc)
                  if to[x['toTeam']] == y['to'] and abs(x['wonAt']-y['wonTime']) <= tol)
    ue, ut, pairs = set(), set(), []
    for _, i, j in cand:
        if i not in ue and j not in ut:
            ue.add(i)
            ut.add(j)
            pairs.append((changes[i], tc[j]))
    return pairs, [c for j, c in enumerate(tc) if j not in ut]


def one_touch_frames(g):
    """25 Hz frames inside a one-touch possession (a oneTouch change's wonAt to the next change)."""
    m = np.zeros(len(g['time']), bool)
    ch = sorted(g['changes'], key=lambda c: c['wonAt'])
    for c, nxt in zip(ch, ch[1:]+[None]):
        if c.get('oneTouch'):
            m[c['wonAt']-1:(nxt['wonAt']-1) if nxt is not None else c['wonAt']+24] = True
    return m


def evaluate(mt, g, Fs, settings=None):
    abstained = []
    times, codes, changes = run_engine(Fs, settings, abstained)
    sc = [{'wonAt': c['wonAt'], 'lostAt': c['lostAt'], 'to': c['toTeam'], 'wonBy': c['wonBy'], 'lostBy': c['lostBy']} for c in changes]
    full = mt.score(g, times, codes, sc)
    clean = mt.score(g, times, codes, sc, one_touch=False)
    lo, hi = float(times.min()), float(times.max())
    every, _ = match(g, changes, True, lo, hi)
    outside, missed = match(g, changes, False, lo, hi)
    P = len(every)/max(1, len(changes))
    R = len(outside)/max(1, len(outside)+len(missed))
    truth, k = mt.truth_at(g, times)
    fps_run = 1.0/float(np.median(np.diff(times))) if len(times) > 1 else 30.0
    phase = g['phase'][k]
    ot = one_touch_frames(g)[k]
    play = truth != 0
    named = play & (codes > 0)
    right = named & (codes == truth)
    acc = lambda m: round(float(right[m].sum()/max(1, named[m].sum())), 4)
    cov = lambda m: round(float(named[m].sum()/max(1, (play & m).sum())), 4)
    got = {id(t) for _, t in every}
    tc = [c for c in g['changes'] if lo <= c['wonTime'] <= hi]
    groups = {'intoOneTouch': [c for c in tc if c.get('oneTouch')],
              'outOfOneTouch': [c for c in tc if c.get('afterOneTouch') and not c.get('oneTouch')],
              'rest': [c for c in tc if not (c.get('oneTouch') or c.get('afterOneTouch'))]}
    out = {'windows': len(Fs), 'abstained': len(abstained),
           'accuracy': full['accuracy'], 'coverage': full['coverage'], 'deadRecall': full['deadRecall'], 'deadPrecision': full['deadPrecision'],
           'inPlayCalledDead_s': round(float(((codes == 0) & (truth != 0)).sum())/fps_run, 1),
           'deadMissed_s': round(float(((codes != 0) & (truth == 0)).sum())/fps_run, 1),
           'phases': {name: {'accuracy': acc(play & (phase == v)), 'coverage': cov(phase == v)} for v, name in ((1, 'held'), (2, 'pass'), (3, 'loose'))},
           'accuracyOutsideOneTouch': acc(play & ~ot), 'accuracyWithoutLoose': acc(play & (phase != 3)),
           'oneTouchRecall2s': {k2: f"{sum(id(c) in got for c in v)}/{len(v)}" for k2, v in groups.items()},
           'changes': {'engine': len(changes), 'all': full['changes'], 'noOneTouch': clean['changes'],
                       'hybrid2s': {'precision': round(P, 4), 'recall': round(R, 4), 'f1': round(2*P*R/max(1e-9, P+R), 4)}}}
    levels = {}
    for lv in dict.fromkeys(list(getattr(possession, 'CHANGE_LEVELS', ()))+[c['confidence'] for c in changes]):
        mine = [c for c in changes if c['confidence'] == lv]
        hit, _ = match(g, mine, True, lo, hi)
        levels[lv] = {'changes': len(mine), 'precision': round(len(hit)/len(mine), 4) if mine else None}
    out['changeConfidence'] = levels
    causes = {}
    for c, tc_ in every:
        lt = tc_.get('lostType') or ''
        truth_cause = 'out_of_play' if tc_['kind'] == 'restart' else next(
            (name for name, prefixes in TRUTH_TYPES.items() if name != 'out_of_play' and any(lt.startswith(p) for p in prefixes)), 'other')
        key = f"{c['cause']} / truth {truth_cause}"
        causes[key] = causes.get(key, 0)+1
    out['causes'] = dict(sorted(causes.items(), key=lambda kv: -kv[1]))
    return out


def curve(mt, g, Fs, thresholds=CURVE):
    """(min_posterior, accuracy, coverage) along the operating point."""
    rows = []
    for thr in thresholds:
        times, codes, _ = run_engine(Fs, {'min_posterior': thr})
        truth, _ = mt.truth_at(g, times)
        play = truth != 0
        named = play & (codes > 0)
        rows.append((thr, round(float((codes[named] == truth[named]).mean()), 4), round(float(named.sum()/play.sum()), 4)))
    return rows


def degrade(Fs, every=180.0, hole=20.0):
    """A registration hole of `hole` seconds in every `every`: the ball's and the players' pitch positions unusable (as
    on the broadcast at 190-215 s), the image unchanged (the players' speeds then come from the image: none in the
    simulator, which records no camera motion)."""
    out = []
    for F in Fs:
        G = dict(F)
        m = (F['t'] % every) < hole
        G['bconf'] = np.where(m, 0.0, F['bconf'])
        G['dP'] = np.where(m[None, :], np.inf, F['dP'])
        G['iP'] = np.where(m[None, :], -1, F['iP'])
        G['cP'] = np.where(m[None, None, :], np.inf, F['cP'])
        G['uP'] = np.where(m, np.inf, F['uP'])
        G['spd'] = np.where(m[:, None], np.nan, F['spd']).astype(np.float32)
        out.append(G)
    return out


def brief(r):
    c = r['changes']
    return {'abstained': f"{r['abstained']}/{r['windows']}", 'accuracy': r['accuracy'], 'coverage': r['coverage'], 'deadRecall': r['deadRecall'],
            'deadPrecision': r['deadPrecision'], 'inPlayDead_s': r['inPlayCalledDead_s'], 'deadMissed_s': r['deadMissed_s'], 'engine': c['engine'],
            'P@2s all': c['all']['within2s']['precision'], 'R@2s all': c['all']['within2s']['recall'],
            'F1@2s all': c['all']['within2s']['f1'], 'F1@1s all': c['all']['within1s']['f1'],
            'F1@2s noOneTouch': c['noOneTouch']['within2s']['f1'], 'F1@1s noOneTouch': c['noOneTouch']['within1s']['f1'],
            'F1@2s hybrid': c['hybrid2s']['f1'], 'won/lost error': [c['all']['within2s'].get('medianWonError'), c['all']['within2s'].get('medianLostError')],
            'wonBy/lostBy': [c['all']['within2s'].get('wonByAccuracy'), c['all']['within2s'].get('lostByAccuracy')],
            'high': r['changeConfidence'].get('high'), 'uncertain': r['changeConfidence'].get('uncertain')}


# ------------------------------------------------------------------------------------------------ hand-labelled runs

CODE = {'A': 1, 'B': 2}


def label_points(rec, F, lab, lo=None):
    """Points every 0.5 s of the labels against the engine's code on the nearest frame."""
    codes = expand(rec, F['frame'])
    t = F['t']
    rows = []
    for p in lab['points']:
        if lo is not None and p['t'] < lo:
            continue
        i = int(np.argmin(np.abs(t-p['t'])))
        rows.append((p['t'], p['team'], int(codes[i])))
    team = [r for r in rows if r[1] in CODE]
    named = [r for r in team if r[2] > 0]
    right = [r for r in named if CODE[r[1]] == r[2]]
    dead = [r for r in rows if r[1] == 'dead']
    edead = [r for r in rows if r[2] == 0]
    return {'teamPoints': len(team), 'named': len(named), 'right': len(right),
            'accuracy': round(len(right)/len(named), 4) if named else None, 'coverage': round(len(named)/len(team), 4) if team else None,
            'deadPoints': f"{sum(1 for r in dead if r[2] == 0)}/{len(dead)} called dead",
            'engineDeadPoints': f"{sum(1 for r in edead if r[1] == 'dead')}/{len(edead)} labelled dead",
            'wrong': [(r[0], r[1]) for r in named if CODE[r[1]] != r[2]]}


def label_changes(rec, lab, tol):
    """The engine's changes against the labels' (wonAt within tol, the same new team, one to one)."""
    eng = rec['changes']
    tc = [c for c in lab.get('changes') or [] if c.get('wonAt') is not None]
    cand = sorted((abs(x['wonAt']-y['wonAt']), i, j) for i, x in enumerate(eng) for j, y in enumerate(tc)
                  if x['toTeam'] == y['toTeam'] and abs(x['wonAt']-y['wonAt']) <= tol)
    ue, ut, pairs = set(), set(), []
    for _, i, j in cand:
        if i not in ue and j not in ut:
            ue.add(i)
            ut.add(j)
            pairs.append((i, j))
    hit = {i for i, _ in pairs}
    return {'engine': len(eng), 'truth': len(tc), 'matched': len(pairs),
            'precision': round(len(pairs)/len(eng), 4) if eng else None, 'recall': round(len(pairs)/len(tc), 4) if tc else None,
            'pairs': [{'truthWon': tc[j]['wonAt'], 'won': eng[i]['wonAt'], 'truthLost': tc[j].get('lostAt'), 'lost': eng[i]['lostAt'],
                       'confidence': eng[i]['confidence']} for i, j in pairs],
            'byConfidence': {lv: f"{sum(1 for i, c in enumerate(eng) if c['confidence'] == lv and i in hit)}/{sum(1 for c in eng if c['confidence'] == lv)}"
                             for lv in dict.fromkeys(list(getattr(possession, 'CHANGE_LEVELS', ()))+[c['confidence'] for c in eng])},
            'unmatched': [(c['wonAt'], c['toTeam'], c['confidence']) for i, c in enumerate(eng) if i not in hit]}


def score_run(name, path, labels):
    lab = json.loads((Path(labels)/f'{name}.json').read_text(encoding='ascii'))
    result = json.loads(Path(path).read_text())
    t0 = time.time()
    F = possession.features(result)
    rec = possession.from_features(F)
    out = {'available': rec['available'], 'reason': rec['reason'], 'quality': rec['quality']['level'], 'liveCoverage': rec['quality']['coverage'],
           'seconds': round(time.time()-t0, 2), 'points': label_points(rec, F, lab),
           'changes1s': label_changes(rec, lab, 1.0), 'changes2s': label_changes(rec, lab, 2.0),
           'deadSpells': [{'truth': [d['from'], d['to']], 'overlap': round(sum(max(0.0, min(d['to'], s['t1'])-max(d['from'], s['t0'])) for s in rec['deadSpells'])
                                                                          / max(1e-9, d['to']-d['from']), 3)} for d in lab.get('deadSpells') or []]}
    if lab.get('overlapWithBC60'):
        out['pointsWithoutBC60Overlap'] = label_points(rec, F, lab, lo=float(lab['overlapWithBC60'][1]))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--tracking', help='folder of the Metrica sample tracking files')
    ap.add_argument('--events', help='folder of the Metrica sample events files')
    ap.add_argument('--heldout', action='store_true', help='game 2: refuses unless the --frozen note exists')
    ap.add_argument('--frozen', help='the freeze note that unlocks --heldout')
    ap.add_argument('--game', type=int, help='with --heldout: the held-out game (default 2)')
    ap.add_argument('--window', type=float, default=300.0, help='seconds per scored window (each a separate result)')
    ap.add_argument('--quick', action='store_true', help='the main table only (full tracking and the camera)')
    ap.add_argument('--labels', help='folder of the PitchIQ hand labels (tools/annotations/possession)')
    ap.add_argument('--run', action='append', default=[], help='NAME=path/to/result.json, scored against --labels/NAME.json')
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--json', help='write every number here')
    a = ap.parse_args(argv)
    started = time.time()
    report = {'version': possession.VERSION, 'settings': possession.DEFAULTS, 'variants': {}, 'runs': {}}
    if a.run:
        if not a.labels:
            sys.exit('--run needs --labels')
        for spec in a.run:
            name, path = spec.split('=', 1)
            report['runs'][name] = score_run(name, path, a.labels)
            r = report['runs'][name]
            print(f"{name}: available {r['available']} quality {r['quality']} live coverage {r['liveCoverage']} | points {json.dumps({k: r['points'][k] for k in ('accuracy', 'coverage', 'right', 'named', 'teamPoints', 'deadPoints', 'engineDeadPoints')})}")
            if 'pointsWithoutBC60Overlap' in r:
                print(f"{name}: points without the BC60 overlap {json.dumps({k: r['pointsWithoutBC60Overlap'][k] for k in ('accuracy', 'coverage', 'deadPoints')})}")
            for tol in ('1s', '2s'):
                c = r[f'changes{tol}']
                print(f"{name}: changes {tol} {json.dumps({k: c[k] for k in ('engine', 'truth', 'matched', 'precision', 'recall', 'byConfidence')})}")
            print(f"{name}: matched {json.dumps(r['changes2s']['pairs'])} unmatched {json.dumps(r['changes2s']['unmatched'])} dead spells {json.dumps(r['deadSpells'])}", flush=True)
    if a.tracking or a.events:
        if not (a.tracking and a.events):
            sys.exit('--tracking and --events go together')
        game = 1
        if a.game is not None and not a.heldout:
            sys.exit('--game goes with --heldout: game 1 is the tuning game, every other game is held out')
        if a.heldout:
            if not a.frozen or not Path(a.frozen).exists():
                sys.exit(f'Refusing --heldout: the freeze note {a.frozen} does not exist. A held-out game is read once, after the engine is frozen.')
            game = a.game if a.game is not None else 2
            if game == 1:
                sys.exit('game 1 is the tuning game, not held out')
        import possession_truth as mt
        g = _game(a.tracking, a.events, game)
        report['game'], report['window'] = game, a.window
        names = ['full', 'camera'] if a.quick else list(VARIANTS)
        feats = {}
        with ProcessPoolExecutor(max(1, a.workers)) as ex:
            for name, Fs in ex.map(build, [(a.tracking, a.events, game, v, a.window) for v in names]):
                feats[name] = Fs
        for name in names:
            report['variants'][name] = {'operating': evaluate(mt, g, feats[name]), 'allNamed': brief(evaluate(mt, g, feats[name], {'min_posterior': 0.0}))}
            print(f'{name:20s} operating {json.dumps(brief(report["variants"][name]["operating"]))}', flush=True)
            print(f'{"":20s} all named {json.dumps(report["variants"][name]["allNamed"])}', flush=True)
        if 'camera-independent' in feats:   # the stress test also with the run-level agreement gate off
            report['variants']['camera-independent']['gateOff'] = brief(evaluate(mt, g, feats['camera-independent'], {'agree_min': 0.0}))
            print(f'{"camera-independent":20s} gate off  {json.dumps(report["variants"]["camera-independent"]["gateOff"])}', flush=True)
        for name in names:
            if name == 'camera-independent':
                continue
            r = report['variants'][name]['operating']
            print(f'{name}: phases {json.dumps(r["phases"])} outside one-touch {r["accuracyOutsideOneTouch"]} without loose {r["accuracyWithoutLoose"]}')
            print(f'{name}: changes 2 s all {json.dumps(r["changes"]["all"]["within2s"])}')
            print(f'{name}: changes 2 s no one-touch {json.dumps(r["changes"]["noOneTouch"]["within2s"])}')
            print(f'{name}: changes 1 s all / no one-touch {json.dumps(r["changes"]["all"]["within1s"])} {json.dumps(r["changes"]["noOneTouch"]["within1s"])}')
            print(f'{name}: change confidence {json.dumps(r["changeConfidence"])} one-touch recall at 2 s {json.dumps(r["oneTouchRecall2s"])}')
            print(f'{name}: causes {json.dumps(r["causes"])}', flush=True)
        if not a.quick:
            report['curve'] = {name: curve(mt, g, feats[name]) for name in ('full', 'camera')}
            for name, rows in report['curve'].items():
                print(f'curve {name}: ' + ' '.join(f'{thr:g}:{acc:.3f}/{cov:.3f}' for thr, acc, cov in rows))
            report['ablations'] = {}
            for label, s in ABLATIONS.items():
                report['ablations'][label] = {name: brief(evaluate(mt, g, feats[name], s)) for name in ('full', 'camera', 'camera-visible')}
                print(f'ablation {label:52s} ' + ' | '.join(f'{name} {json.dumps(v)}' for name, v in report['ablations'][label].items()), flush=True)
            hole = degrade(feats['camera'])
            report['registrationHole'] = {'image fallback': brief(evaluate(mt, g, hole)), 'no image fallback': brief(evaluate(mt, g, hole, {'image_only_w': 0.0}))}
            print(f'registration hole (20 s in 180 s): {json.dumps(report["registrationHole"])}')
            with ProcessPoolExecutor(max(1, a.workers)) as ex:
                short = dict(ex.map(build, [(a.tracking, a.events, game, 'camera', 60.0)]))
            report['window60'] = brief(evaluate(mt, g, short['camera']))
            print(f'camera, 60 s windows: {json.dumps(report["window60"])}')
    report['seconds'] = round(time.time()-started, 1)
    print(f'done in {report["seconds"]} s')
    if a.json:
        Path(a.json).write_text(json.dumps(report, indent=1, default=float), encoding='ascii')


if __name__ == '__main__':
    main()
