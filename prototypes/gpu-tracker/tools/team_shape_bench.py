"""Team shape analytics (soccer/team_shape.py) on full-pitch tracking data: the Metrica Sports sample games turned into
a PitchIQ result, with or without a simulated broadcast camera.

    python tools/team_shape_bench.py --data DIR --game 1 [--rate 5] [--camera WIDTH] [--noise-tau S] [--max-missing N]
                                     [--seed K] [--minutes M] [--no-switch] [--json out]

DIR holds the Metrica sample data (Sample_Game_<n>_RawTrackingData_{Home,Away}_Team.csv; 25 Hz, positions 0..1 on a
105 x 68 m pitch, the goalkeeper in each team's first column, NaN while a player is off the pitch). Home is team A, away
team B. The result has one frame per 1/rate s, every player on the pitch as a confirmed observation, the goal ends of
the first period and, unless --no-switch, the user setting "sides swap at the start of the second period" (the way a
coach would set it for a full match).

--camera WIDTH hides everyone outside a simulated side camera (tools/metrica_bench.py simulate_view: a window WIDTH
metres long following the ball with a lag, with occasional pans), adds observation noise, and builds the off-screen
estimates with soccer/imputation.py, as the pipeline does. The full-coverage run is then the reference: for every
sample the camera run measured, the error of its width / depth against the true value is reported by status
(complete, reliable, partial), which is what the coverage rules promise.

The noise is correlated in time, as a tracked position's error is (camera_error: sigma 0.5-2 m per axis, an AR(1)
process with a --noise-tau time constant, 1.5 s by default). --noise-tau 0 draws it independently on every sample, as
the bench did until 2026-10-18: steps of several metres every 0.2 s, which analytics._breaks takes for registration
jumps, so TeamShape dropped about half of the players in view as glitches and the coverage measured the noise model.
--max-missing N classifies formation windows missing up to N players of the lineup (team_shape.FORMATION_MAX_MISSING
otherwise), to measure what a looser rule would label.

Reported: per team the width / depth / line aggregates, the formation periods per half and the share of time per label,
how often the label changes, and with --camera what was inside the window against what TeamShape kept, the errors by
status, the formation agreement and the camera run's own formation periods.
"""
import argparse
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'tools'))

from soccer import analytics, imputation, team_shape  # noqa: E402
from metrica_bench import LENGTH, WIDTH, load_team, simulate_view  # noqa: E402

NOISE_SIGMA = (.5, 2.0)       # m: the simulated camera's position error per axis, drawn per sighting
NOISE_TAU = 1.5               # s: its time constant (AR(1)); 0 draws it independently on every sample
SIGHTING_GAP = 2.0            # s: a player out of view longer than this comes back as a new sighting (a new sigma)


def camera_error(state, pid, t, rng, tau=NOISE_TAU):
    """The simulated camera's position error (x, y metres) of a player in view at time t: sigma * z, with z a unit
    AR(1) process per player (z_t = phi z_s + sqrt(1 - phi^2) n, phi = exp(-(t - s) / tau), s the player's previous
    sample in view, which is exact across the time out of view) and sigma ~ U(NOISE_SIGMA) per sighting. Every sample's
    error has the same distribution as with tau = 0, which draws sigma and the error anew on every sample (the bench
    before 2026-10-18, with the same random numbers, so its results reproduce)."""
    if not tau:
        sigma = rng.uniform(*NOISE_SIGMA)
        return rng.normal(0, sigma), rng.normal(0, sigma)
    n = rng.normal(size=2)
    prev = state.get(pid)
    if prev is None:
        z, sigma = n, rng.uniform(*NOISE_SIGMA)
    else:
        s, z, sigma = prev
        phi = math.exp(-(t-s)/tau)
        z = phi*z+math.sqrt(1-phi*phi)*n
        if t-s > SIGHTING_GAP:
            sigma = rng.uniform(*NOISE_SIGMA)
    state[pid] = (t, z, sigma)
    return float(sigma*z[0]), float(sigma*z[1])


def load_game(data, game):
    teams = {}
    for team, side in (('A', 'Home'), ('B', 'Away')):
        times, pos, ball, names = load_team(Path(data)/f'Sample_Game_{game}_RawTrackingData_{side}_Team.csv')
        teams[team] = (times, pos, ball, names)
    periods = _periods(Path(data)/f'Sample_Game_{game}_RawTrackingData_Home_Team.csv')
    return teams, periods


def _periods(path):
    out = []
    with open(path, encoding='utf-8') as fh:
        for k, line in enumerate(fh):
            if k < 3:
                continue
            out.append(int(line.split(',', 1)[0]))
    return np.array(out)


def build_result(teams, periods, rate=5.0, minutes=None, camera=None, seed=0, noise_tau=NOISE_TAU):
    """A PitchIQ-style result: frames, match roster, roster slots, goal ends, pitch size. With a camera, only the
    players inside its window, each with camera_error added."""
    times, _, ball, _ = teams['A']
    step = max(1, int(round(25/rate)))
    keep = np.arange(0, len(times), step)
    if minutes:
        keep = keep[times[keep] <= minutes*60]
    view = None
    if camera:
        x0, x1, ymax = simulate_view(times, ball, camera, 1.5, seed=seed)
        view = (x0, x1)
    rng = np.random.default_rng(seed)
    noise = {}
    frames = []
    seen = {}
    for k in keep:
        people = []
        for team in ('A', 'B'):
            _, pos, _, names = teams[team]
            for j in range(pos.shape[1]):
                x, y = pos[k, j]
                if not (np.isfinite(x) and np.isfinite(y)):
                    continue
                pid = ('GK-' if j == 0 else f'{team}-')+(team if j == 0 else f'{j:02d}')
                if view is not None and not (view[0][k] <= x <= view[1][k]):
                    continue
                if view is not None:
                    ex, ey = camera_error(noise, pid, float(times[k]), rng, noise_tau)
                    x, y = x+ex, y+ey
                nx, ny = float(x/LENGTH), float(y/WIDTH)
                if not (-.02 <= nx <= 1.02 and -.02 <= ny <= 1.02):
                    continue
                people.append({'id': pid, 'display': pid, 'team': team, 'role': 'GOALKEEPER' if j == 0 else 'PLAYER', 'state': 'confirmed',
                               'identityConfidence': .9, 'pitch': [round(nx, 5), round(ny, 5)], 'pitchConfidence': .9,
                               'pitchSigma': [.3, .3], 'track': j+(0 if team == 'A' else 100)})
                first, last = seen.get(pid, (float(times[k]), float(times[k])))
                seen[pid] = (min(first, float(times[k])), float(times[k]))
        frames.append({'time': round(float(times[k]), 3), 'frame': int(k), 'people': people, 'calibrated': True})
    players, gks, slots = [], [], []
    for team in ('A', 'B'):
        _, pos, _, names = teams[team]
        for j in range(pos.shape[1]):
            pid = ('GK-' if j == 0 else f'{team}-')+(team if j == 0 else f'{j:02d}')
            if not np.isfinite(pos[keep, j, 0]).any():
                continue
            entry = {'id': pid, 'display': pid, 'team': team, 'confirmed': True}
            (gks if j == 0 else players).append(entry)
            present = np.flatnonzero(np.isfinite(pos[keep, j, 0]))
            slots.append({'id': pid, 'team': team, 'role': 'GOALKEEPER' if j == 0 else 'PLAYER', 'confirmed': True,
                          'firstSeen': float(times[keep][present[0]]), 'left': float(times[keep][present[-1]]) if present[-1] < len(keep)-1 else None})
    gk_x = np.nanmean(teams['A'][1][keep[periods[keep] == 1], 0, 0])
    ends = {'left': 'A', 'right': 'B'} if gk_x < LENGTH/2 else {'left': 'B', 'right': 'A'}
    result = {'version': 3, 'video': {'fps': rate, 'width': 1920, 'height': 1080}, 'frames': frames,
              'match': {'players': players, 'goalkeepers': gks}, 'roster': {'expected': {'A': 11, 'B': 11}, 'slots': slots},
              'tactical': {'goalEnds': ends, 'pitchSize': {'length': LENGTH, 'width': WIDTH, 'source': 'metrica'}}}
    if camera:
        result['timeline'] = imputation.estimate(result, LENGTH, WIDTH)
    second = keep[periods[keep] == 2]
    switch = float(times[second[0]]) if len(second) else None
    return result, ends, switch


def in_view(teams, result, camera, seed=0):
    """What the simulated camera had in its window, before noise and before TeamShape's rules, per team over the
    result's frames: the mean of outfield players inside it, the share of frames with every outfield player on the
    pitch inside it, and with at least team_shape.RELIABLE_MIN inside it."""
    times, _, ball, _ = teams['A']
    x0, x1, _ = simulate_view(times, ball, camera, 1.5, seed=seed)
    k = np.array([f['frame'] for f in result['frames']], np.int64)
    out = {}
    for team in ('A', 'B'):
        pos = teams[team][1][k, 1:]
        on = np.isfinite(pos[:, :, 0])
        inside = (on & (pos[:, :, 0] >= x0[k, None]) & (pos[:, :, 0] <= x1[k, None])).sum(axis=1)
        out[team] = {'meanPlayers': round(float(inside.mean()), 2), 'fullShare': round(float((inside == on.sum(axis=1)).mean()), 3),
                     'reliableShare': round(float((inside >= team_shape.RELIABLE_MIN).mean()), 3)}
    return out


def summarize(shape, team, switch):
    out = {}
    halves = [(None, switch), (switch, None)] if switch else [(None, None)]
    for h, (a, b) in enumerate(halves):
        rep = shape.report(team, a, b)
        periods = rep['formation']['periods']
        dur = Counter()
        for p in periods:
            lo, hi = p['clipped']
            dur[p['label']] += hi-lo
        total = sum(dur.values()) or 1
        changes = sum(1 for p, q in zip(periods, periods[1:]) if p['label'] in team_shape.TEMPLATES and q['label'] in team_shape.TEMPLATES
                      and p['label'] != q['label'])
        conf = Counter(p.get('confidence') for p in periods if p['label'] in team_shape.TEMPLATES)
        out[f'half{h+1}'] = {
            'range': rep['range'], 'direction': rep['direction']['side'], 'directionStatus': rep['direction']['status'],
            'width': {k: rep['widthDepth']['width'][k] for k in ('mean', 'median', 'min', 'max', 'reliableShare')},
            'depth': {k: rep['widthDepth']['depth'][k] for k in ('mean', 'median', 'min', 'max', 'reliableShare')},
            'line': {k: rep['line']['summary'].get(k) for k in ('height', 'deepest', 'measuredShare', 'disruptedShare', 'trend')},
            'lineSources': rep['line']['sources'],
            'formationShare': {k: round(v/total, 3) for k, v in dur.most_common()},
            'periods': len(periods), 'formationChanges': changes, 'confidence': dict(conf),
            'longest': max(((p['clipped'][1]-p['clipped'][0], p['label']) for p in periods), default=None),
        }
    return out


def errors_by_status(ref_shape, cam_shape, team):
    """Width / depth error of the camera run against the full-coverage reference, by the camera run's status."""
    ref = {r['t']: r for r in team_shape.width_depth_series(ref_shape.data, team, ref_shape.data.start, ref_shape.data.end)}
    cam = team_shape.width_depth_series(cam_shape.data, team, cam_shape.data.start, cam_shape.data.end)
    out = {}
    for key in ('width', 'depth'):
        by = {}
        for r in cam:
            st = r.get(key+'Status')
            truth = ref.get(r['t'])
            if st == 'unavailable' or not truth or not truth.get(key) or not r.get(key):
                by.setdefault(st, []).append(None)
                continue
            by.setdefault(st, []).append(r[key]['metres']-truth[key]['metres'])
        out[key] = {}
        n_all = len(cam)
        for st, errs in by.items():
            e = np.array([v for v in errs if v is not None])
            out[key][st] = {'share': round(len(errs)/max(n_all, 1), 3)}
            if len(e):
                out[key][st].update(meanError=round(float(e.mean()), 2), medianAbs=round(float(np.median(np.abs(e))), 2),
                                    p90Abs=round(float(np.percentile(np.abs(e), 90)), 2), underBy5=round(float((e < -5).mean()), 3))
    return out


def formation_agreement(ref_shape, cam_shape, team):
    """How the camera run's formation labels compare with the full-coverage run, window by window: the share of windows
    labelled, and among the labelled ones the agreement (exact, and within a related pair such as 3-5-2 / 5-3-2), broken
    down by the number of players the camera run had in the window; `ownAgreement` is the window's own best fit against
    the reference label, before the stabilization lends it its neighbours' label."""
    ref_w = {w['centre']: lab for w, lab in zip(ref_shape.shapes[team]['windows'], ref_shape.shapes[team]['labels'])}
    by = {}
    total = labelled = agree = related = 0
    for w, lab in zip(cam_shape.shapes[team]['windows'], cam_shape.shapes[team]['labels']):
        total += 1
        if lab not in team_shape.TEMPLATES:
            continue
        truth = ref_w.get(w['centre'])
        if truth not in team_shape.TEMPLATES:
            continue
        labelled += 1
        same = truth == lab
        rel = same or frozenset((truth, lab)) in team_shape.RELATED
        agree += same
        related += rel
        k = by.setdefault(int(w.get('players') or 0), [0, 0, 0, 0])
        k[0] += 1
        k[1] += same
        k[2] += rel
        k[3] += w.get('best') == truth
    return {'windows': total, 'labelledShare': round(labelled/max(total, 1), 3), 'agreement': round(agree/max(labelled, 1), 3),
            'agreementRelated': round(related/max(labelled, 1), 3),
            'byPlayers': {n: {'windows': v[0], 'agreement': round(v[1]/v[0], 3), 'agreementRelated': round(v[2]/v[0], 3),
                              'ownAgreement': round(v[3]/v[0], 3)} for n, v in sorted(by.items())}}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', required=True)
    ap.add_argument('--game', type=int, action='append', required=True)
    ap.add_argument('--rate', type=float, default=5.0)
    ap.add_argument('--minutes', type=float)
    ap.add_argument('--camera', type=float, help='simulated camera window length in metres')
    ap.add_argument('--noise-tau', type=float, default=NOISE_TAU, help='time constant (s) of the camera noise; 0 = independent per sample')
    ap.add_argument('--max-missing', type=int, help='formation windows may miss this many players (default team_shape.FORMATION_MAX_MISSING)')
    ap.add_argument('--seed', type=int, default=0, help="the camera's pans and noise")
    ap.add_argument('--no-switch', action='store_true', help='do not tell the analysis that the sides swap at half time')
    ap.add_argument('--json')
    a = ap.parse_args(argv)
    if a.max_missing is not None:
        team_shape.FORMATION_MAX_MISSING = a.max_missing
    report = {}
    for game in a.game:
        t0 = time.perf_counter()
        teams, periods = load_game(a.data, game)
        result, ends, switch = build_result(teams, periods, a.rate, a.minutes)
        settings = {'direction': {'A': 'left' if ends['left'] == 'A' else 'right', 'switchAt': None if a.no_switch else switch}}
        shape = team_shape.TeamShape(result, settings=settings)
        entry = {'goalEnds': ends, 'switchAt': switch, 'seconds': round(time.perf_counter()-t0, 1),
                 'A': summarize(shape, 'A', switch), 'B': summarize(shape, 'B', switch)}
        if a.camera:
            cam_result, _, _ = build_result(teams, periods, a.rate, a.minutes, camera=a.camera, seed=a.seed, noise_tau=a.noise_tau)
            cam = team_shape.TeamShape(cam_result, settings=settings)
            view = in_view(teams, cam_result, a.camera, a.seed)
            entry['camera'] = {'width': a.camera, 'noiseTau': a.noise_tau, 'seed': a.seed, 'formationMaxMissing': team_shape.FORMATION_MAX_MISSING}
            for team in ('A', 'B'):
                rep = cam.report(team)
                entry['camera'][team] = {'inView': view[team], 'coverage': rep['coverage'], 'excluded': rep['excluded'],
                                         'errors': errors_by_status(shape, cam, team), 'formation': formation_agreement(shape, cam, team),
                                         'halves': summarize(cam, team, switch)}
        report[f'game{game}'] = entry
        print(json.dumps(entry, indent=1, default=float), flush=True)
    if a.json:
        Path(a.json).write_text(json.dumps(report, indent=1, default=float), encoding='utf-8')
    return report


if __name__ == '__main__':
    main()
