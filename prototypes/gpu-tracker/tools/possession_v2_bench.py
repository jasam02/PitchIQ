"""Fit and benchmark soccer/possession_v2.py on a Metrica game through tools/possession_sim.py's camera.

    python -I -B tools/possession_v2_bench.py --data DIR [--game 3] [--fit out.json] [--variant broadcast|tactical]
                                              [--seed 0] [--json report.json]

--data holds the game (game 3: Sample_Game_3_metadata.xml, _tracking.txt, _events.json; games 1-2 the CSV files with
their events in the same folder). Downloaded data: run with -I -B (this script adds the repository and tools/ to
sys.path itself; no data folder is ever on it).

The truth (tools/possession_truth.py) is turned into the v2 definition per frame: held and the flight of a completed
pass -> the team; 'loose' (after the losing action, before the gain) -> contested where players of both teams are within
DUEL_M of the ball (Metrica's own positions), else free; dead -> not live.

--fit: the model (emission table, generator, gamma, initial distributions) is fitted on the game's FIRST half and the
report scores the SECOND half with it (the fitted constants are what soccer/possession_v2.py carries; the report states
which game and variant they came from). Without --fit the module's own constants are scored on both halves.

Variants of the simulated ball (possession_sim.DETECT: per context, P(seen | seen before), P(missed | missed before)):
'broadcast' is the simulator's default (fitted to the broadcast ball audit); 'tactical' sees the ball less, as on the
CAN-ISL tactical-wide match (its live play: about half the frames tracked, 45% bridged, 5% unplaced).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'tools'))
from soccer import possession_v2 as pv  # noqa: E402

DUEL_M = 1.5
VARIANTS = {
    'broadcast': None,
    'tactical': {'feet': (.86, .985), 'free': (.86, .55), 'fast': (.62, .70), 'air': (.55, .70)},
    # longer hidden spells (misses decay over MISS_DECAY_GAPS frames instead of possession_sim's 140), set to the CAN-ISL
    # match's unplaced share of live play: game 3 seed 0 gives tracked .368, bridged .545, unplaced .087 (CAN-ISL .461 /
    # .447 / .092: the simulation bridges more of what CAN-ISL tracks)
    'tactical_gaps': {'feet': (.86, .987), 'free': (.86, .68), 'fast': (.62, .78), 'air': (.55, .78)},
}
MISS_DECAY_GAPS = 300.0
THRESHOLDS = (.5, .6, .7, .8, .9, .95)


def truth_states(g, times):
    """Per time the v2 definition's state as the engine's six states: -1 dead, 0 A (Home), 1 B (Away), 2 / 3 contested
    after A's / B's control, 4 / 5 free after A's / B's control (the loser is the team the truth still credits during a
    loose ball), and the 25 Hz index. outcome() maps them to the output codes (0 A, 1 B, 2 contested, 3 free)."""
    import possession_truth as mt
    code, k = mt.truth_at(g, times)
    phase = g['phase'][k]
    out = np.full(len(k), -1, np.int8)
    out[(phase == 1) | (phase == 2)] = np.where(code[(phase == 1) | (phase == 2)] == 1, 0, 1)
    loose = phase == 3
    if loose.any():
        ball = g['ball'][k[loose]]
        dh = np.nanmin(np.hypot(g['Home'][k[loose], :, 0]-ball[:, None, 0], g['Home'][k[loose], :, 1]-ball[:, None, 1]), axis=1)
        da = np.nanmin(np.hypot(g['Away'][k[loose], :, 0]-ball[:, None, 0], g['Away'][k[loose], :, 1]-ball[:, None, 1]), axis=1)
        both = (dh <= DUEL_M) & (da <= DUEL_M)
        loser_b = code[loose] == 2
        idx = np.flatnonzero(loose)
        out[idx] = np.where(both, np.where(loser_b, pv.CB, pv.CA), np.where(loser_b, pv.FB, pv.FA))
    return out, k


def outcome(truth):
    """Six-state truth -> output codes: -1 dead, 0 A, 1 B, 2 contested, 3 free."""
    out = truth.copy()
    out[(truth == pv.CA) | (truth == pv.CB)] = 2
    out[(truth == pv.FA) | (truth == pv.FB)] = 3
    return out


def restart_teams(g, T, live):
    """{first frame of a live sequence: restarting team index or None} from the truth's dead spells."""
    starts = {}
    t = T['t']
    for a, b in pv._runs(live):
        starts[a] = None
    for s in g.get('deadSpells') or []:
        team = s.get('restartTeam')
        if not s.get('restart'):
            continue
        end_t = float(g['time'][min(int(s['restart'])-1, len(g['time'])-1)])
        i = int(np.searchsorted(t, end_t))
        for a in list(starts):
            if abs(a-i) <= 5 and starts[a] is None and team in ('Home', 'Away'):
                starts[a] = 0 if team == 'Home' else 1
    return starts


SMOOTH = .5    # each state's class probabilities are shrunk toward the pooled class frequencies by this weight, so a class
               # rarely seen in the simulation (a hidden ball: 1% of its live frames) cannot make a state impossible


def fit(M, truth, T, live, starts, smooth=SMOOTH):
    """Emission table (mirror-pooled soft counts, shrunk toward the pooled class frequencies), generator (mirror-pooled
    transition counts per second in a state), initial distributions; gamma and rate_scale by the grid in main()."""
    E = np.zeros((pv.NS, pv.K))
    for s in range(pv.NS):
        sel = live & (truth == s)
        E[s] += M[sel].sum(axis=0)
        ms = pv.MIRROR_STATE[s]
        E[ms] += M[sel][:, pv.MIRROR_CLASS].sum(axis=0)
    pooled = E.sum(axis=0)/max(E.sum(), 1e-9)
    E = E/np.maximum(E.sum(axis=1, keepdims=True), 1e-9)
    E = (E+smooth*pooled[None, :])/(1.0+smooth)
    t = T['t']
    dt = np.diff(t)
    counts = np.zeros((pv.NS, pv.NS))
    time_in = np.zeros(pv.NS)
    for a, b in pv._runs(live):
        for i in range(a+1, b+1):
            s0, s1 = truth[i-1], truth[i]
            if s0 < 0 or s1 < 0 or dt[i-1] > .2:
                continue
            time_in[s0] += dt[i-1]
            time_in[pv.MIRROR_STATE[s0]] += dt[i-1]
            if s1 != s0:
                counts[s0, s1] += 1
                counts[pv.MIRROR_STATE[s0], pv.MIRROR_STATE[s1]] += 1
    G = counts/np.maximum(time_in, 1e-9)[:, None]
    np.fill_diagonal(G, 0.0)
    np.fill_diagonal(G, -G.sum(axis=1))
    init_r = np.full(pv.NS, .5)
    init_u = np.full(pv.NS, .5)
    for a, team in starts.items():
        s = truth[a]
        if s < 0:
            continue
        if team is None:
            init_u[s] += 1
            init_u[pv.MIRROR_STATE[s]] += 1
        else:
            init_r[s if team == 0 else pv.MIRROR_STATE[s]] += 1
    return {'emission': E.tolist(), 'generator': G.tolist(), 'init_restart': (init_r/init_r.sum()).tolist(),
            'init_unknown': (init_u/init_u.sum()).tolist()}


def score(T, out, post, path, truth6, live, curve=THRESHOLDS):
    """Team accuracy / coverage (truth A or B), neutral (contested + free) recall and precision, the frames whose ball is
    not placed, and the accuracy-coverage curve along min_posterior."""
    truth = outcome(truth6)
    team_t = live & (truth >= 0) & (truth <= 1)
    neu_t = live & (truth >= 2)
    inner = out['inner']
    named = (inner == 0) | (inner == 1)
    res = {}
    def acc_cov(mask):
        n = int(mask.sum())
        nm = mask & named
        right = int((nm & (inner == truth)).sum())
        return {'frames': n, 'coverage': round(float(nm.sum())/max(1, n), 3), 'accuracy': round(right/max(1, int(nm.sum())), 4)}
    res['team'] = acc_cov(team_t)
    unplaced = T['bst'] == 0
    bridged = T['bst'] == 2
    res['team_ball_unplaced'] = acc_cov(team_t & unplaced)
    res['team_ball_bridged'] = acc_cov(team_t & bridged)
    res['team_ball_seen'] = acc_cov(team_t & (T['bst'] == 1))
    neu_e = (inner == 2) | (inner == 3)
    nl = live & named
    res['namedPrecision'] = round(float((nl & (inner == truth)).sum())/max(1, int(nl.sum())), 4)
    res['neutral'] = {'truthFrames': int(neu_t.sum()), 'recall': round(float((neu_t & neu_e).sum())/max(1, int(neu_t.sum())), 3),
                      'precision': round(float((neu_t & neu_e).sum())/max(1, int((live & neu_e).sum())), 3),
                      'namedTeamOnNeutral': round(float((neu_t & named).sum())/max(1, int(neu_t.sum())), 3)}
    pa, pb = post[:, 0], post[:, 1]
    res['curve'] = []
    for th in curve:
        a_ = team_t & (pa >= th)
        b_ = team_t & (pb >= th)
        nm = int((a_ | b_).sum())
        right = int((a_ & (truth == 0)).sum()+(b_ & (truth == 1)).sum())
        res['curve'].append({'min_posterior': th, 'coverage': round(nm/max(1, int(team_t.sum())), 3), 'accuracy': round(right/max(1, nm), 4)})
    # calibration of the team posterior
    bins = [(.5, .6), (.6, .7), (.7, .8), (.8, .9), (.9, .95), (.95, 1.01)]
    cal = []
    pmax = np.maximum(pa, pb)
    pick = np.where(pa >= pb, 0, 1)
    for lo, hi in bins:
        m = team_t & (pmax >= lo) & (pmax < hi)
        if m.sum():
            cal.append({'bin': [lo, hi], 'frames': int(m.sum()), 'right': round(float((pick[m] == truth[m]).mean()), 3)})
    res['calibration'] = cal
    return res


def score_changes(g, out, sel_t, tol=(1.0, 2.0)):
    """Open-play truth changes (wonAt) inside the selected time range matched one-to-one to the engine's changes of the
    same new team: precision, recall, median |wonAt error|."""
    lo, hi = sel_t
    truth = []
    for c in g['changes']:
        if c.get('kind') != 'open' or c.get('wonAt') is None:
            continue
        tw = float(g['time'][int(c['wonAt'])-1])
        if lo <= tw <= hi:
            truth.append((tw, 0 if c['to'] == 'Home' else 1))
    eng = [(c['wonAt'], 0 if c['to'] == 'A' else 1, c['confidence']) for c in out['changes'] if lo <= c['wonAt'] <= hi]
    res = {'truth': len(truth), 'engine': len(eng), 'engineHigh': sum(1 for e in eng if e[2] == 'high')}
    for tl in tol:
        used = set()
        errs = []
        for tw, k in truth:
            best = None
            for j, (ew, ek, _) in enumerate(eng):
                if j in used or ek != k or abs(ew-tw) > tl:
                    continue
                if best is None or abs(ew-tw) < abs(eng[best][0]-tw):
                    best = j
            if best is not None:
                used.add(best)
                errs.append(abs(eng[best][0]-tw))
        res[f'within{tl:g}s'] = {'precision': round(len(used)/max(1, len(eng)), 3), 'recall': round(len(used)/max(1, len(truth)), 3),
                                 'medianError': round(float(np.median(errs)), 2) if errs else None}
    return res


def logloss(post, truth, live):
    m = live & (truth >= 0) & np.isfinite(post[:, 0])
    p = post[m, truth[m]]
    return float(-np.mean(np.log(np.clip(p, 1e-6, 1))))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--game', type=int, default=3)
    ap.add_argument('--variant', default='tactical', choices=sorted(VARIANTS))
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--fit')
    ap.add_argument('--json')
    ap.add_argument('--fps', type=float, default=25.0)
    ap.add_argument('--r-ctrl', type=float, help='the control zone (soccer/possession_v2 DEFAULTS r_ctrl) to fit and score with')
    args = ap.parse_args(argv)
    import possession_truth as mt
    import possession_sim as cs
    t0 = time.time()
    g = mt.load_game(args.game, args.data, args.data)
    if args.variant == 'tactical_gaps':
        cs.MISS_DECAY = MISS_DECAY_GAPS
    sim = cs.simulate(g, mode='camera', fps=args.fps, seed=args.seed, detect=VARIANTS[args.variant])
    result = sim.result()
    ends = result['tactical']['goalEnds']
    T = pv.resolve(pv.table(result), ends=(ends.get('left'), ends.get('right')))
    print(f'table {T["n"]} frames in {time.time()-t0:.1f} s', flush=True)
    truth, k = truth_states(g, T['t'])
    live = truth >= 0
    period = g['period'][k]
    first, second = live & (period == 1), live & (period == 2)
    st = T['bst'][live]
    shares = {s: round(float((st == v).mean()), 3) for s, v in (('tracked', 1), ('bridged', 2), ('unplaced', 0))}
    print('ball in live play', shares, flush=True)
    t1 = time.time()
    S = {'r_ctrl': args.r_ctrl} if args.r_ctrl else None
    M, aux = pv.classes(T, S)
    print(f'classes in {time.time()-t1:.1f} s', flush=True)
    report = {'game': args.game, 'variant': args.variant, 'seed': args.seed, 'fps': args.fps, 'ballLiveShares': shares,
              'truthLiveShares': {n: round(float((truth[live] == i).mean()), 3) for i, n in enumerate(pv.STATE_NAMES)}}
    starts_all = restart_teams(g, T, live)
    if args.fit:
        params = fit(M, truth, T, first, {a: v for a, v in starts_all.items() if first[a]})
        # the operating constants: gamma (evidence per frame) and rate_scale (how fast the belief may move), chosen on the
        # FIRST half by the objective declared before it was run (possv2/PREDECLARED.md addendum): maximise coverage at
        # min_posterior .8 plus change F1 within 2 s, among settings whose team accuracy is at least .97 and whose named
        # frames are the truth's team on at least .95 of them
        best = None
        st_first = {a: v for a, v in starts_all.items() if first[a]}
        ts = T['t'][first]
        grid = []
        for gamma, scale, og in [(g_, s_, o_) for g_ in (.2, .3, .5) for s_ in (3.0, 10.0) for o_ in (1.0, .3, .1)]:
            params['gamma'], params['rate_scale'], params['open_gamma'] = gamma, scale, og
            mod = pv.model(params)
            post, path = pv.decode(T, M, first, mod, starts=st_first)
            live_code = np.where(first, -1, pv.STOPPAGE).astype(np.int8)
            o = pv.label(T, post, path, M, aux, live_code, S)
            sc = score(T, o, post, path, truth, first)
            ch = score_changes(g, o, (float(ts.min()), float(ts.max())))
            f1 = 2*ch['within2s']['precision']*ch['within2s']['recall']/max(1e-9, ch['within2s']['precision']+ch['within2s']['recall'])
            obj = sc['team']['coverage']+f1
            okay = sc['team']['accuracy'] >= .97 and sc['namedPrecision'] >= .95
            grid.append({'gamma': gamma, 'rate_scale': scale, 'open_gamma': og, 'coverage': sc['team']['coverage'], 'accuracy': sc['team']['accuracy'],
                         'namedPrecision': sc['namedPrecision'], 'changeF1': round(f1, 3), 'logloss': round(logloss(post, truth, first), 4),
                         'eligible': okay})
            print(grid[-1], flush=True)
            if okay and (best is None or obj > best[0]):
                best = (obj, gamma, scale, og)
        if best is None:
            best = (None, .2, 3.0, 1.0)
        params['gamma'], params['rate_scale'], params['open_gamma'] = best[1], best[2], best[3]
        params['provenance'] = {'game': args.game, 'variant': args.variant, 'seed': args.seed, 'fps': args.fps, 'r_ctrl': args.r_ctrl,
                                'fittedOn': 'first half', 'objective': 'coverage@.8 + change F1 (2 s), accuracy >= .97, named precision >= .95',
                                'grid': grid}
        Path(args.fit).write_text(json.dumps(params, indent=1), encoding='utf-8')
        report['fit'] = params
        mod = pv.model(params)
    else:
        mod = pv.model()
    t2 = time.time()
    post, path = pv.decode(T, M, live, mod, starts=starts_all)
    live_code = np.where(live, -1, pv.STOPPAGE).astype(np.int8)
    out = pv.label(T, post, path, M, aux, live_code, S)
    report['decodeSeconds'] = round(time.time()-t2, 2)
    for name, sel in (('first', first), ('second', second)):
        report[name] = score(T, out, post, path, truth, sel)
        report[name]['logloss'] = round(logloss(post, truth, sel), 4)
        ts = T['t'][sel]
        report[name]['changes'] = score_changes(g, out, (float(ts.min()), float(ts.max())))
    print(json.dumps({k: report[k] for k in ('ballLiveShares', 'truthLiveShares', 'first', 'second')}, indent=1))
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main()
