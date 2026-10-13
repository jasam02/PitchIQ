"""Benchmark of the off-screen estimation (soccer/imputation.py) on full-pitch tracking data with simulated
camera views.

    python tools/metrica_bench.py --data DIR --game 1 [--game 2] [--fps 5] [--width 45] [--lag 1.5] [--low-angle] [--json out]

DIR holds the Metrica Sports sample data (Sample_Game_<n>_RawTrackingData_{Home,Away}_Team.csv; 25 Hz,
positions 0..1 on a 105 x 68 m pitch, NaN while a player is off the pitch). A side camera is simulated in
pitch space: its centre follows the ball along the length with a first-order lag, it sees a window of
--width metres (the whole width of the pitch, or only its far 70% with --low-angle), and every ~25 s it
pans for 3 s away from the ball and back. Players outside the window are hidden; the visible ones get
Gaussian noise (0.5-2 m) as their observation. Every method then estimates the hidden players from the
visible ones, and the error against the known true positions is reported by how long the player had been
hidden: last position, constant velocity, linear interpolation (offline), a fixed formation anchor, the
biased visible-mean offsets, role-anchored centroid voting, and the one-sided (online) variant of the
voting. Also the share of hidden samples each method estimates at all (coverage) and, for methods with a
sigma, how well calibrated it is (p68 of |error|/sigma: 1 is perfect).

Use one game to choose constants and keep the other game untouched for the reported numbers.
"""
import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from soccer import imputation  # noqa: E402

LENGTH, WIDTH = 105.0, 68.0
BUCKETS = ((0, 1), (1, 3), (3, 8), (8, 20), (20, 1e9))
METHODS = ('last', 'cv', 'linear', 'anchor', 'mean', 'voting', 'voting-online', 'shift', 'shiftrole', 'shiftrole-online', 'hybrid',
           'last-online', 'cv-online', 'shift-online')


def load_team(path):
    """(times (T,), positions (T, P, 2) metres with NaN off the pitch, ball (T, 2) metres or NaN, player names)."""
    with open(path, encoding='utf-8', newline='') as fh:
        rows = list(csv.reader(fh))
    header = rows[2]
    names = [h for h in header[3:-2:2]]
    times, pos, ball = [], [], []
    for r in rows[3:]:
        if len(r) < len(header):
            continue
        times.append(float(r[2]))
        values = np.array([float(v) if v not in ('', 'NaN') else np.nan for v in r[3:]], np.float64)
        players = values[:-2].reshape(-1, 2)
        pos.append(players)
        ball.append(values[-2:])
    times = np.array(times)
    pos = np.array(pos)
    ball = np.array(ball)
    pos[:, :, 0] *= LENGTH
    pos[:, :, 1] *= WIDTH
    ball[:, 0] *= LENGTH
    ball[:, 1] *= WIDTH
    # The second period restarts the clock in some exports: make time monotonic.
    for k in range(1, len(times)):
        if times[k] <= times[k-1]:
            times[k:] += times[k-1]-times[k]+.04
    return times, pos, ball, names


def simulate_view(times, ball, width_m, lag_s, low_angle=False, seed=0):
    """The camera's visible window per frame: (x0, x1, y_max) arrays."""
    rng = np.random.default_rng(seed)
    T = len(times)
    cx = np.zeros(T)
    target = np.where(np.isfinite(ball[:, 0]), ball[:, 0], LENGTH/2)
    cx[0] = target[0]
    pan_until, pan_target = -1.0, None
    next_pan = 25.0+rng.uniform(0, 10)
    for t in range(1, T):
        dt = times[t]-times[t-1]
        if times[t] >= next_pan and pan_target is None:
            pan_target = float(np.clip(target[t]+rng.choice([-1, 1])*rng.uniform(15, 30), 10, LENGTH-10))
            pan_until = times[t]+3.0
            next_pan = times[t]+25.0+rng.uniform(0, 15)
        goal = pan_target if (pan_target is not None and times[t] < pan_until) else target[t]
        if pan_target is not None and times[t] >= pan_until:
            pan_target = None
        rate = 15.0*dt if pan_target is not None else abs(goal-cx[t-1])*(1-math.exp(-dt/lag_s))
        cx[t] = cx[t-1]+np.sign(goal-cx[t-1])*min(abs(goal-cx[t-1]), rate)
    x0, x1 = cx-width_m/2, cx+width_m/2
    y_max = np.full(T, WIDTH*(.7 if low_angle else 1.0))
    return x0, x1, y_max


def observe(pos, x0, x1, y_max, seed=0):
    """Observed positions (noise added) and sigmas where the true position lies in the window; NaN elsewhere."""
    rng = np.random.default_rng(seed+1)
    T, P, _ = pos.shape
    inside = np.isfinite(pos[:, :, 0]) & (pos[:, :, 0] >= x0[:, None]) & (pos[:, :, 0] <= x1[:, None]) & (pos[:, :, 1] <= y_max[:, None])
    sig = rng.uniform(.5, 2.0, size=(T, P))
    noise = rng.normal(size=(T, P, 2))*sig[:, :, None]
    obs = np.where(inside[:, :, None], pos+noise, np.nan)
    sig = np.where(inside, sig, np.nan)
    return obs, sig, inside


def evaluate(true, obs, sig, inside, times, method):
    """Errors of the hidden samples (true known, not visible): per bucket of time since the last observation."""
    bidirectional = not method.endswith('-online')
    base = method[:-7] if method.endswith('-online') else method
    est, sigma, kind, age = imputation.estimate_team(obs, sig, times, base, bidirectional)
    hidden = np.isfinite(true[:, :, 0]) & ~inside
    # Time since the last past observation, for the buckets (also for bidirectional methods).
    T, P = hidden.shape
    since = np.full((T, P), np.inf)
    last = np.full(P, -np.inf)
    for t in range(T):
        last = np.where(inside[t], times[t], last)
        since[t] = times[t]-last
    out = {'coverage': 0.0, 'buckets': {}, 'calibration': None, 'samples': int(hidden.sum())}
    estimated = hidden & np.isfinite(est[:, :, 0])
    out['coverage'] = round(float(estimated.sum()/max(1, hidden.sum())), 3)
    err = np.hypot(*(est-true).transpose(2, 0, 1))
    for lo, hi in BUCKETS:
        mask = estimated & (since >= lo) & (since < hi)
        n = int(mask.sum())
        total = int((hidden & (since >= lo) & (since < hi)).sum())
        if n:
            e = np.sort(err[mask])
            out['buckets'][f'{lo}-{hi if hi < 1e9 else ""}s'] = {'n': n, 'of': total, 'mae': round(float(e.mean()), 2), 'median': round(float(e[len(e)//2]), 2),
                                                                 'p90': round(float(e[int(.9*len(e))-1]), 2)}
        else:
            out['buckets'][f'{lo}-{hi if hi < 1e9 else ""}s'] = {'n': 0, 'of': total}
    ratio = err[estimated]/np.maximum(sigma[estimated], 1e-6)
    if len(ratio):
        r = np.sort(ratio)
        out['calibration'] = round(float(r[int(.68*len(r))-1]), 2)
    all_e = np.sort(err[estimated])
    if len(all_e):
        out['all'] = {'mae': round(float(all_e.mean()), 2), 'median': round(float(all_e[len(all_e)//2]), 2), 'p90': round(float(all_e[int(.9*len(all_e))-1]), 2)}
    return out


def run(data, games, fps, width_m, lag_s, low_angle, methods, seed=0):
    report = {'games': {}, 'settings': {'fps': fps, 'width': width_m, 'lag': lag_s, 'lowAngle': low_angle}}
    step = max(1, int(round(25/fps)))
    for game in games:
        report['games'][game] = {}
        for side in ('Home', 'Away'):
            path = Path(data)/f'Sample_Game_{game}_RawTrackingData_{side}_Team.csv'
            times, pos, ball, names = load_team(path)
            times, pos, ball = times[::step], pos[::step], ball[::step]
            x0, x1, y_max = simulate_view(times, ball, width_m, lag_s, low_angle, seed)
            obs, sig, inside = observe(pos, x0, x1, y_max, seed)
            visible_share = float(inside.sum()/max(1, np.isfinite(pos[:, :, 0]).sum()))
            team = {'frames': int(len(times)), 'players': int(pos.shape[1]), 'visibleShare': round(visible_share, 3), 'methods': {}}
            for method in methods:
                started = time.perf_counter()
                team['methods'][method] = evaluate(pos, obs, sig, inside, times, method)
                team['methods'][method]['seconds'] = round(time.perf_counter()-started, 1)
            report['games'][game][side] = team
    return report


def table(report):
    lines = []
    for game, sides in report['games'].items():
        for side, team in sides.items():
            lines.append(f"\n### game {game} {side}: {team['frames']} frames, {team['players']} players, {team['visibleShare']:.0%} of player-frames visible")
            buckets = list(next(iter(team['methods'].values()))['buckets'])
            lines.append('| method | coverage | '+' | '.join(f'{b} MAE/med/p90 (n)' for b in buckets)+' | all MAE | calibration p68 |')
            lines.append('|---|---|'+'---|'*len(buckets)+'---|---|')
            for method, r in team['methods'].items():
                cells = []
                for b in buckets:
                    v = r['buckets'][b]
                    cells.append(f"{v['mae']}/{v['median']}/{v['p90']} ({v['n']})" if v.get('n') else f"- ({v.get('of', 0)})")
                lines.append(f"| {method} | {r['coverage']} | "+' | '.join(cells)+f" | {r.get('all', {}).get('mae', '-')} | {r['calibration']} |")
    return '\n'.join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', required=True)
    ap.add_argument('--game', type=int, action='append', default=[])
    ap.add_argument('--fps', type=float, default=5.0)
    ap.add_argument('--width', type=float, default=45.0)
    ap.add_argument('--lag', type=float, default=1.5)
    ap.add_argument('--low-angle', action='store_true')
    ap.add_argument('--methods', default=','.join(METHODS))
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--json')
    a = ap.parse_args(argv)
    report = run(a.data, a.game or [1], a.fps, a.width, a.lag, a.low_angle, a.methods.split(','), a.seed)
    print(table(report))
    if a.json:
        Path(a.json).write_text(json.dumps(report, indent=1), encoding='utf-8')


if __name__ == '__main__':
    sys.exit(main())
