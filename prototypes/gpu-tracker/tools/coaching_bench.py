"""The professional reference of the coaching report (soccer/coaching/reference.json), measured on real professional
tracking: the Metrica Sports sample game 1 (both teams, 25 Hz, every player on the pitch, the ball), and its one-off
validation on the held-out game 2.

    python -I -B tools/coaching_bench.py --data DIR --game 1 --out soccer/coaching/reference.json
    python -I -B tools/coaching_bench.py --data DIR --game 2 --validate [--reference PATH] [--json OUT]
                                         [--record [--acceptance FP.json]]

DIR holds Sample_Game_<n>_RawTrackingData_{Home,Away}_Team.csv (downloaded data: the bench runs only under python -I
-B, so nothing next to the data or in the working directory is ever imported and no bytecode is written).

Why it is built this way:
- The same computation as a PitchIQ run. The game is turned into a PitchIQ result by tools/team_shape_bench.py
  (build_result at 5 samples a second, home = team A, the half-time side swap set as a coach would set it) and measured
  by soccer/coaching/series.build_series, the code every detector reads, so a team's value and its professional
  baseline differ only in the footage, never in the definition. The lateral lag is reference.lateral_windows on
  reference.dm_velocities, also the detector's code, and the episodes are soccer.coaching.episodes.episodes under the
  default rule, exactly as a detector counts them.
- Measured like PitchIQ measures: all the time, dead ball included. A PitchIQ result has no in-play mask, so a run's
  numbers include every stoppage the camera shows; the distributions it is compared with must too. A sample counts
  when the team's whole lineup of 10 outfield players is observed and its direction is known. The in-play variant
  (the ball tracked and on the pitch, 1 m margin: 59.9% of the samples; Metrica tracks the ball only while it is
  live, 39% of the frames have none in 155 gaps of median 7.8 s) stays in the file as metrics[m].inPlay, for the
  record of what the dead ball changes. On game 1 it adds 62% more samples (a team's full lineup is observed in
  94.7% / 96.9% of all samples) and widens the low end most: set pieces crowd the players, so the hull's p10 is 479
  against 550 m2 in play (own third 261 against 433 m2), the nearest-teammate distance's p10 3.9 against 5.0 m and the
  width's p10 28.9 against 30.8 m, while the p90s move by 1-2% (width 56.1 against 57.7 m). The lateral lag has 1,331
  instead of 390 windows of 30 s, p10 / p50 / p90 -0.8 / 0.2 / 1.0 s against -0.6 / 0.2 / 0.8 s: a reference without
  the dead ball made a PitchIQ run's stoppages look late (0.06-0.12 lateral findings per 10 team-minutes on professional
  play, none without the dead ball, validation of 2026-10-18).
- Both teams pooled. Two teams of one match is the whole base, and their p90s differ by 1.0 m (width), 1.9 m (D-M
  gap), 2.5 m (M-F gap) and 46 m2 (hull), about what one team's two halves differ by (0.5-2.5 m, the audit of
  2026-10-18), except the back line, which depends on how many defenders form it (A plays a back four, B a back three
  in 84% of its line samples; the depth std's p90 is 7.8 m behind a back four and 4.9 m behind a back three): line
  metrics carry the back-line size as a context (backline:3|4|5), pooled across the teams within one size.
- Contexts. The pitch third of the team's centre (series.third: own below 35 m, final above 70 m from its own goal
  line): the D-M gap's p90 is 15.7 m with the centre in its own third, 19.4 m in the middle and 24.9 m in the final
  third, so one threshold for all would flag every attacking spell. The nearest-teammate distance per unit (D, M, F).
- Quantiles p05 ... p975, n per context. reference.quantile falls back to 'all' below MIN_SAMPLES (300) samples. Lag
  quantiles are kept at the 5 Hz resolution (multiples of 0.2 s, reference.quantize_lag), and a lag window at the
  search bound is no measurement (22 of the 30 s and 61 of the 20 s windows of game 1 sat at +-3.8 s or beyond).
- Window distributions (the improvement test's baseline). Per metric and context the default rule's episodes are
  found per team and half (an episode lasts only while its context holds: leaving the context, like any unmeasured
  stretch, ends it, censored, after 3 s), and every window of W measured minutes (reference.WINDOW_MINUTES and a
  whole half), started every 5 s of measured time, counts the episodes with at least 5 s of their time inside it,
  which is what a detector run on that window finds (2,991 windows re-detected from scratch: 99.8% the same count, the
  rest one more here, never fewer). The counts' p50/p90/p95/p99/p999, maximum and histogram, with the windows each
  team gave (a distribution needs both teams, reference.WINDOW_MIN_TEAM_SHARE each: game 1's back-line sizes are
  one team's each, so its back-line evenness has none). The per-player distance has per-player windows
  (playerWindowCounts). The lateral lag's count is the number of late windows among the non-overlapping ones
  (reference.independent_windows) inside the measured window, as the detector counts. A detector reads the next
  longer length's levels (reference.window_baseline); windowBoundCheck in the file sets those against lengths
  measured directly in between (WINDOW_CHECK_MINUTES): never below them.
- Episode rates per measured minute over the four team-halves (perMinute p10/p50/p90) stay as a description.
- Decorrelation time tau per metric: the e-folding time of the autocorrelation of the professional "beyond the p75"
  indicator (all team-halves pooled, pairs inside continuous measured stretches; game 1: 7.6 s for the length to
  13.6 s for the back-line height, 15 s for the lateral lag's windows): the strength rule's effective number of
  independent looks is measured / tau. The e-folding time and not the integrated autocorrelation time (about twice as
  long: 16.8-25.7 s): on game 1 the one-sided 80% upper bound of the share beyond the p75 over 120-300 s windows covers
  the team-half's own share in 76-84% of the windows with the e-folding time (nominal 80%) but 89-99% with the
  integrated time, which would demand twice the evidence the data needs. Both coverages are written to the file.

--validate reads the held-out game with every threshold frozen (the reference file is only read): the share of its
samples above each stored quantile (by construction about 1 - q on game 1), its per-half episode rates against
game 1's range and, per window distribution, the share of the held-out team-windows beyond the stored p99 (expected
at most 1%). Game 2 stays untouched until the validation stage; the bench refuses to read it without --validate,
refuses --validate on the game the reference was built from, and never changes a threshold in that mode. With --record
it writes the validation's summary (validation_summary) into the reference's 'validation' block and nothing else:
status 'run', the date, the reference's name (reference.content_sha, which leaves the block out, so it stays the name
every observation carries), the sha of the file as validated, and, with --acceptance, the summary of the pre-declared
false-positive test (the detectors on the held-out game as PitchIQ-shaped results: improvement findings per 10
team-minutes at most 0.2 at every window length and no high-confidence concern), whose verdict is the record's. The
identity is checked unchanged before the file is replaced, and an acceptance summary made against another file is
refused. The validation of 2026-10-19 (game 2, PASS) is recorded so; see the README section "Coaching report".
"""
import argparse
import datetime
import hashlib
import json
import math
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT/'tools')):
    if p not in sys.path:
        sys.path.insert(0, p)

import team_shape_bench as tsb  # noqa: E402
from soccer.coaching import episodes as ep, reference as rf, series as cs  # noqa: E402

TUNING_GAME = 1             # the only game a reference is built from
HELD_OUT = (2,)             # read only by --validate, after the thresholds are frozen
RATE = 5.0                  # samples a second (team_shape.SAMPLE_RATE: the grid every coaching series lives on)
OUTFIELD = 10               # outfield players of a full lineup: a sample counts when all of them are observed
IN_PLAY_MARGIN = 1.0        # m: the ball this far beyond a line is still in play (its position jitters at the line)
QUANTILE_LEVELS = (5.0, 10.0, 25.0, 50.0, 75.0, 90.0, 95.0, 97.5)    # the reference.QUANTILES keys, in order
DIGITS = {'m': 1, 'm2': 0, 's': 2, 'ratio': 3}                     # rounding per unit (metres 0.1, seconds 0.01)
LAG_WINDOWS = {'all': rf.LAG_WINDOW_SECONDS, 'window:20': 20.0}     # lagDM / corrDM contexts: window seconds
THIRDS = ('own', 'middle', 'final')
BACK_LINES = (3, 4, 5)
UNITS = ('D', 'M', 'F')
DEFAULT_RULE = {'enter': 'p95', 'exit': 'p90', 'minSeconds': 5.0, 'mergeSeconds': 3.0}
MIN_HALF_MINUTES = 5.0      # measured minutes a team-half needs to enter the per-minute spread: at the professional rate
                            # of about 0.3 episodes a minute a shorter one holds one or two (team A's 0.37 minutes behind
                            # a back three read 2.7 a minute from one episode); a context with fewer than two such
                            # team-halves has no per-minute baseline (reference.episode_baseline falls back to 'all')
WINDOW_LEVELS = (50.0, 90.0, 95.0, 99.0, 99.9)          # the window entries' p50 / p90 / p95 / p99 / p999
HALF_MIN_MINUTES = 20.0     # measured minutes a team-half (a player-half) needs in a context to give a whole-half window
WINDOW_CHECK_MINUTES = (1.25, 3.5, 7.0, 17.0)          # lengths measured directly to check reference.window_baseline
TAU_QUANTILE = 'p75'        # the strength rule's level: tau is the decorrelation time of the share beyond it
TAU_MAX_SECONDS = 300.0     # s: the autocorrelation is followed this far at most
TAU_COVERAGE_SECONDS = (60.0, 120.0, 300.0)   # window lengths of the bound's coverage check
STRENGTH_CONFIDENCE = .8    # the strength rule's one-sided upper bound (detectors.STRENGTH_BOUND_CONFIDENCE)

# The reference metrics: unit, the series value and the eligibility metric it is gated by, its context family, and
# what it is (written into the file, so a reader never has to guess the definition behind a threshold).
METRICS = {
    'width': ('m', 'width', 'width', 'third',
              "Across-pitch extent of the team's outfield players (series.width: team_shape.width_depth_series, "
              'complete or reliable samples).'),
    'depth': ('m', 'depth', 'depth', 'third',
              "Along-pitch extent of the team's outfield players (series.depth)."),
    'hull': ('m2', 'hull', 'hull', 'third',
             'Convex hull area of the outfield players (series.hull, complete samples).'),
    'spread': ('m', 'spread', 'spread', 'third',
               'RMS distance of the outfield players to their centroid (series.spread, complete samples).'),
    'gapDM': ('m', 'gapDM', 'gapDM', 'third',
              "Mean depth of the first midfield line (the formation template's line index 1) minus the defensive "
              "unit's, both units complete and every member observed (series.gapDM)."),
    'gapMF': ('m', 'gapMF', 'gapMF', 'third',
              'Mean depth of the forward line minus that of every middle line (series.gapMF).'),
    'lineStd': ('m', 'lineStd', 'lineStd', 'line',
                "Standard deviation of the defenders' distances from their own goal line: team_shape.defensive_line "
                'complete samples of the formation back line, not disrupted (series.lineStd).'),
    'lineHeight': ('m', 'lineHeight', 'lineStd', 'line',
                   "Median distance of the defenders from their own goal line (series.lineHeight)."),
    'nearestTeammate': ('m', 'nearestTeammate', 'nearestTeammate', 'unit',
                        'Distance from an outfield player to the nearest outfield teammate, per player and sample '
                        '(series.nearestTeammate); unit:D|M|F by the player\'s unit (series.group).'),
    'lagDM': ('s', None, None, 'window',
              "Seconds the defence's sideways movement trails the midfield's over a 30 s window (positive: the "
              'defence moves later): the lag of reference.lateral_windows(t, midfield, defence) on '
              'reference.dm_velocities, both units complete, windows at the search bound left out; window:20 for '
              '20 s windows. Quantiles at the 5 Hz resolution (multiples of 0.2 s).'),
    'corrDM': ('ratio', None, None, 'window',
               "Peak correlation of the defence's and the midfield's sideways velocities over the window (the same "
               'windows as lagDM).'),
}
CONTEXTS = {'third': ('all',)+tuple(f'third:{x}' for x in THIRDS),
            'line': ('all',)+tuple(f'third:{x}' for x in THIRDS)+tuple(f'backline:{n}' for n in BACK_LINES),
            'unit': ('all',)+tuple(f'unit:{u}' for u in UNITS),
            'window': tuple(LAG_WINDOWS)}
CONTEXT_NOTES = {
    'all': 'every eligible sample, dead ball included (PitchIQ results have no in-play mask)',
    'third': "the pitch third of the team's outfield centre from its own goal line (series.third: own below L/3, "
             'final above 2L/3)',
    'backline': "the number of defenders in the back line (series.lineSize); team A's back four and team B's back "
                'three',
    'unit': "the player's unit at the sample (series.group: D the back line, M every middle line, F the front line)",
    'window': 'lagDM / corrDM: all = 30 s windows, window:20 = 20 s windows, starting 5 s apart',
}


class Refused(SystemExit):
    """The bench will not do what was asked (held-out data, validating on the tuning game, ...)."""

    def __init__(self, message):
        super().__init__(f'coaching_bench: {message}')
        self.message = message


# ---------- loading ----------
def in_play(ball):
    """Per sample: the ball is tracked and on the pitch (IN_PLAY_MARGIN beyond the lines still counts)."""
    b = np.asarray(ball, float)
    with np.errstate(invalid='ignore'):
        return (np.isfinite(b).all(axis=1) & (b[:, 0] >= -IN_PLAY_MARGIN) & (b[:, 0] <= tsb.LENGTH+IN_PLAY_MARGIN)
                & (b[:, 1] >= -IN_PLAY_MARGIN) & (b[:, 1] <= tsb.WIDTH+IN_PLAY_MARGIN))


def load(data, game, rate=RATE):
    """The game as the coaching layer sees it: (CoachingData, in-play mask, half per sample, info)."""
    clock = time.perf_counter()
    teams, periods = tsb.load_game(data, game)
    result, ends, switch = tsb.build_result(teams, periods, rate)
    settings = {'direction': {'A': 'left' if ends['left'] == 'A' else 'right', 'switchAt': switch}}
    cd = cs.build_series(result, settings=settings)
    k = np.asarray(cd.frame, np.int64)                  # the 25 Hz row of each sample's frame
    ball = teams['A'][2][k] if len(k) else np.zeros((0, 2))
    play = in_play(ball)
    half = np.asarray(periods, np.int64)[k] if len(k) else np.zeros(0, np.int64)
    info = {'game': int(game), 'rate': float(rate), 'samples': int(len(cd.t)),
            'seconds': round(float(cd.end-cd.start), 2) if len(cd.t) else 0.0,
            'switchAt': None if switch is None else round(float(switch), 2), 'goalEnds': ends,
            'inPlayShare': round(float(play.mean()), 3) if len(play) else 0.0,
            'buildSeconds': round(time.perf_counter()-clock, 1)}
    return cd, play, half, info


def data_files(data, game):
    """sha256 of the two tracking files (which data the reference was measured on)."""
    out = {}
    for side in ('Home', 'Away'):
        p = Path(data)/f'Sample_Game_{game}_RawTrackingData_{side}_Team.csv'
        h = hashlib.sha256()
        with open(p, 'rb') as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b''):
                h.update(chunk)
        out[p.name] = h.hexdigest()
    return out


# ---------- the samples a reference rests on ----------
def team_base(s, play):
    """The time mask `play` (everything for the primary distributions, in play for the documentation variant), the
    whole lineup of OUTFIELD players observed, the direction known."""
    known = (s.side == 'left') | (s.side == 'right')
    return np.asarray(play, bool) & s.complete & (s.expected == OUTFIELD) & known


def context_mask(s, family, context):
    """The samples (S, or S x N for a per-player metric) of a team in a context."""
    if context == 'all':
        return np.ones(s.nearestTeammate.shape if family == 'unit' else len(s.t), bool)
    kind, value = context.split(':', 1)
    if kind == 'third':
        m = s.third == value
        return np.repeat(m[:, None], len(s.ids), axis=1) if family == 'unit' else m
    if kind == 'backline':
        with np.errstate(invalid='ignore'):
            return s.lineSize == float(value)
    if kind == 'unit':
        return s.group == value
    raise ValueError(f'unknown context {context!r}')


def collect(cd, play):
    """Per metric (not the windowed lag) and team: (values, eligible) with the base mask applied, and the context
    masks: {metric: {team: (values, eligible, {context: mask})}}."""
    out = {}
    for metric, (_, key, gate, family, _) in METRICS.items():
        if key is None:
            continue
        out[metric] = {}
        for team, s in cd.teams.items():
            base = team_base(s, play)
            v = np.asarray(s.values[key], float)
            e = s.eligible[gate] & (base[:, None] if v.ndim == 2 else base) & np.isfinite(v)
            out[metric][team] = (v, e, {c: context_mask(s, family, c) for c in CONTEXTS[family]})
    return out


def lateral(cd, play, halves, drop_bound=True):
    """Per window context and team: the lateral windows (reference.lateral_windows on the base samples, those at the
    search bound left out unless drop_bound is False), half by half (the clock runs on across half time, so a window
    must not): {context: {team: [window]}} with each window's half in 'half'."""
    out = {}
    for context, seconds in LAG_WINDOWS.items():
        out[context] = {}
        for team, s in cd.teams.items():
            t, defence, midfield, ok = rf.dm_velocities(cd, team)
            ok = ok & team_base(s, play)
            out[context][team] = [dict(w, half=int(h)) for h in np.unique(halves[halves > 0])
                                  for w in rf.lateral_windows(t, midfield, defence, ok & (halves == h), seconds,
                                                              drop_bound=drop_bound)]
    return out


# ---------- statistics ----------
def _round(x, digits):
    return None if x is None or not np.isfinite(x) else round(float(x), digits)


def _round_metric(metric, unit, x, step=1.0/RATE):
    if metric == 'lagDM':
        return None if x is None or not np.isfinite(x) else rf.quantize_lag(x, step)
    return _round(x, DIGITS[unit])


def quantiles(values, unit, metric=None):
    """{'n', 'p05', ..., 'p975'} of the finite values, rounded for the unit (monotone, as rounding is); a lag at the
    sample step."""
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    q = {'n': int(len(v))}
    if not len(v):
        return q
    for key, x in zip(rf.QUANTILES, np.percentile(v, QUANTILE_LEVELS)):
        q[key] = _round_metric(metric, unit, x)
    return q


def _pool(per_team, context):
    """The pooled values of both teams in a context: (values, per-team values)."""
    parts = {team: v[e & m[context]] for team, (v, e, m) in per_team.items()}
    return np.concatenate(list(parts.values())) if parts else np.zeros(0), parts


def _spread(rates):
    r = np.array([x for x in rates if x is not None], float)
    if not len(r):
        return {'p10': None, 'p50': None, 'p90': None}
    p = np.percentile(r, [10, 50, 90])
    return {'p10': _round(p[0], 3), 'p50': _round(p[1], 3), 'p90': _round(p[2], 3)}


def context_units(t, halves, per_team, context, enter, exit_, rule, step):
    """The measured units of a context: one per team and half (and per player column for a per-player metric), each
    {'team', 'half', 'column', 'mask', 'episodes', 'measured'} with the default rule's episodes over its samples."""
    out = []
    for team, (v, e, masks) in per_team.items():
        ec = e & masks[context]
        for h in np.unique(halves[halves > 0]):
            m = ec & (halves == h)[:, None] if v.ndim == 2 else ec & (halves == h)
            cols = range(v.shape[1]) if v.ndim == 2 else (None,)
            for j in cols:
                mj = m[:, j] if j is not None else m
                vj = v[:, j] if j is not None else v
                if not mj.any():
                    continue
                found = ep.episodes(t, vj, mj, enter, exit_, rule['minSeconds'], rule['mergeSeconds'], 'above', step)
                out.append({'team': team, 'half': int(h), 'column': j, 'mask': mj, 'episodes': found,
                            'measured': ep.measured_seconds(t, mj, step)})
    return out


def team_half_episodes(t, v, e, half, enter, exit_, rule, step):
    """Episodes of one team in one half (v, e: S or S x N; a per-player metric counts every player's own episodes):
    (episodes, measured seconds, [(count, measured seconds) per player column with measured time])."""
    m = e & (half[:, None] if v.ndim == 2 else half)
    if v.ndim == 1:
        v, m = v[:, None], m[:, None]
    found, measured, per = [], 0.0, []
    for j in range(v.shape[1]):
        if not m[:, j].any():
            continue
        seconds = ep.measured_seconds(t, m[:, j], step)
        mine = ep.episodes(t, v[:, j], m[:, j], enter, exit_, rule['minSeconds'], rule['mergeSeconds'], 'above', step)
        found += mine
        measured += seconds
        per.append((len(mine), seconds))
    return found, measured, per


def episode_baseline(t, halves, per_team, context, q, rule, step, allow_sparse=False, units=None):
    """The professional rate of episodes under a rule in a context: per team-half counts and rates, their spread over
    the team-halves with at least MIN_HALF_MINUTES measured, durations and the share of measured time inside episodes.
    None without the thresholds, or (unless allow_sparse) with fewer than two team-halves in the spread. units: the
    context's measured units (context_units), computed here when not given."""
    enter, exit_ = q.get(rule['enter']), q.get(rule['exit'])
    if enter is None or exit_ is None:
        return None
    if units is None:
        units = context_units(t, halves, per_team, context, enter, exit_, rule, step)
    per_player = any(u['column'] is not None for u in units)
    rows, durations, inside, censored, total, players = [], [], 0.0, 0, 0.0, []
    for team in per_team:
        for h in np.unique(halves[halves > 0]):
            mine = [u for u in units if u['team'] == team and u['half'] == int(h)]
            found = [x for u in mine for x in u['episodes']]
            measured = sum(u['measured'] for u in mine)
            players += [ep.rate_per_minute(len(u['episodes']), u['measured']) for u in mine
                        if u['column'] is not None and u['measured'] >= MIN_HALF_MINUTES*60]
            if measured <= 0:
                continue
            rows.append({'team': team, 'half': int(h), 'count': len(found), 'measuredMinutes': round(measured/60, 2),
                         'perMinute': _round(ep.rate_per_minute(len(found), measured), 3),
                         'inSpread': bool(measured >= MIN_HALF_MINUTES*60)})
            durations += [x['seconds'] for x in found]
            inside += sum(x['samples'] for x in found)*step
            censored += sum(1 for x in found if x['censoredStart'] or x['censoredEnd'])
            total += measured
    pooled, _ = _pool(per_team, context)
    d = np.array(durations, float)
    spread = [r['perMinute'] for r in rows if r['inSpread']]
    if len(spread) < 2 and not allow_sparse:
        return None
    out = {'thresholds': {'enter': enter, 'exit': exit_},
           'perMinute': _spread(spread), 'halvesInSpread': len(spread), 'perHalf': rows,
           'durationSeconds': {'p50': _round(np.median(d), 2) if len(d) else None,
                               'p90': _round(np.percentile(d, 90), 2) if len(d) else None,
                               'max': _round(d.max(), 2) if len(d) else None},
           'shareOfTime': _round(inside/total, 3) if total else None,
           'count': int(len(d)), 'censored': int(censored), 'measuredMinutes': round(total/60, 2),
           'pooledPerMinute': _round(ep.rate_per_minute(len(d), total), 3),
           'shareAboveEnter': _round(ep.share_beyond(pooled, np.ones(len(pooled), bool), enter), 3),
           'shareAboveExit': _round(ep.share_beyond(pooled, np.ones(len(pooled), bool), exit_), 3)}
    if per_player:      # one player's own rate (per measured minute of his) across the player-halves
        out['perPlayerMinute'] = dict(_spread(players), n=len(players))
    return out


# ---------- window distributions ----------
def span_counts(t, step, mask, spans, minutes, need_seconds, slide_seconds=rf.WINDOW_SLIDE_SECONDS):
    """The episode count of every window of `minutes` measured minutes in one unit (mask: its measured samples;
    spans: [(i0, i1)] of its episodes), the windows starting every slide_seconds of measured time: an episode counts
    when at least need_seconds of it lie inside the window (what a detector run on the window alone counts). None
    when the unit is shorter than the window."""
    idx = np.flatnonzero(mask)
    K = int(round(minutes*60/step))
    if K < 1 or K > len(idx):
        return None
    every = max(1, int(round(slide_seconds/step)))
    s = np.arange(0, len(idx)-K+1, every)
    ws, we = idx[s], idx[s+K-1]
    if not spans:
        return np.zeros(len(s), int)
    i0 = np.array([a for a, _ in spans], int)
    i1 = np.array([b for _, b in spans], int)
    a = np.maximum(i0[:, None], ws[None, :])
    b = np.minimum(i1[:, None], we[None, :])
    last = len(t)-1
    dur = np.where(b >= a, t[np.minimum(b, last)]-t[np.minimum(a, last)]+step, 0.0)
    return (dur >= need_seconds-1e-9).sum(axis=0)


def lateral_counts(t, step, cover, windows, length, enter, minutes, slide_seconds=rf.WINDOW_SLIDE_SECONDS,
                   min_corr=rf.LAG_MIN_CORR):
    """The lateral detector's count over every window of `minutes` measured minutes of one team-half (cover: the
    samples its lag windows cover, windows: its lag windows in time order): the late windows (reference.late_windows)
    among the non-overlapping ones (reference.independent_windows) lying wholly inside the measured window."""
    idx = np.flatnonzero(cover)
    K = int(round(minutes*60/step))
    if K < 1 or K > len(idx):
        return None
    every = max(1, int(round(slide_seconds/step)))
    starts = range(0, len(idx)-K+1, every)
    if not windows:
        return np.zeros(len(starts), int)
    n = max(2, int(round(length/step)))
    rank = np.cumsum(cover)-1
    r0 = np.array([rank[w['i0']] for w in windows], int)
    t0 = np.array([w['t0'] for w in windows], float)
    late = np.array([w['lag'] >= enter and w['corr'] >= min_corr for w in windows], bool)
    nxt = np.searchsorted(t0, t0+length-1e-6, side='left')
    out = []
    for s in starts:
        a = int(np.searchsorted(r0, s, side='left'))
        b = int(np.searchsorted(r0, s+K-n+1, side='left'))       # windows a..b-1 lie wholly inside
        c, j = 0, a
        while j < b:
            c += int(late[j])
            j = int(nxt[j])
        out.append(c)
    return np.array(out, int)


def window_entry(minutes, counts, measured, half=False):
    """One window length's distribution: counts {team: [arrays]}, measured {team: [minutes of the units used]}."""
    by_team = {t: (np.concatenate(c) if c else np.zeros(0, int)) for t, c in counts.items()}
    allc = np.concatenate(list(by_team.values())) if by_team else np.zeros(0, int)
    n = int(len(allc))
    if not n:
        return None
    q = np.percentile(allc, WINDOW_LEVELS)
    n_by = {t: int(len(c)) for t, c in by_team.items()}
    both = len(n_by) >= 2 and all(k >= rf.WINDOW_MIN_TEAM_SHARE*n and k > 0 for k in n_by.values())
    hist = np.bincount(allc.astype(int)).tolist()
    total = sum(sum(v) for v in measured.values())
    out = {'minutes': round(float(minutes), 2), 'n': n, 'nByTeam': n_by, 'bothTeams': bool(both),
           'independent': round(total/minutes, 1) if minutes > 0 else None,
           **{k: round(float(x), 2) for k, x in zip(('p50', 'p90', 'p95', 'p99', 'p999'), q)},
           'max': int(allc.max()), 'hist': [int(x) for x in hist]}
    if half:
        out['half'] = True
    return out


def window_distribution(units, count_fn, minutes_list=rf.WINDOW_MINUTES, extra=()):
    """The window entries of a context: units [{'team', 'measured' (s), ...}], count_fn(unit, minutes) -> counts or
    None. Lengths: minutes_list, the extra lengths (checks), and a whole half: the shortest unit among those with at
    least HALF_MIN_MINUTES measured (those units only)."""
    entries = []
    teams = sorted({u['team'] for u in units})
    for W in tuple(minutes_list)+tuple(extra):
        counts, measured = {t: [] for t in teams}, {t: [] for t in teams}
        for u in units:
            c = count_fn(u, W)
            if c is not None and len(c):
                counts[u['team']].append(c)
                measured[u['team']].append(u['measured']/60)
        e = window_entry(W, counts, measured)
        if e is not None:
            entries.append(e)
    long_units = [u for u in units if u['measured']/60 >= HALF_MIN_MINUTES]
    if long_units:
        W = math.floor(min(u['measured'] for u in long_units)/60*10)/10
        if W >= HALF_MIN_MINUTES and all(abs(e['minutes']-W) > 1e-6 for e in entries):
            counts, measured = {t: [] for t in teams}, {t: [] for t in teams}
            for u in long_units:
                c = count_fn(u, W)
                if c is not None and len(c):
                    counts[u['team']].append(c)
                    measured[u['team']].append(u['measured']/60)
            e = window_entry(W, counts, measured, half=True)
            if e is not None:
                entries.append(e)
    return sorted(entries, key=lambda e: e['minutes'])


# ---------- decorrelation ----------
def _autocorrelation(t, step, series, max_seconds=TAU_MAX_SECONDS):
    """The pooled autocorrelation rho[k-1] (k = 1, 2, ... samples) of indicator series [(x, ok)] over pairs of
    measured samples k steps apart in time; stops after the integrated time's initial positive sequence ends."""
    xs = [(np.asarray(x, float), np.asarray(ok, bool)) for x, ok in series if np.asarray(ok, bool).sum() > 1]
    if not xs:
        return np.zeros(0)
    vals = np.concatenate([x[ok] for x, ok in xs])
    mu, var = float(vals.mean()), float(vals.var())
    if var <= 0:
        return np.zeros(0)
    rho = []
    for k in range(1, int(max_seconds/step)+1):
        num = den = 0.0
        for x, ok in xs:
            if len(x) <= k:
                continue
            both = ok[:-k] & ok[k:] & (np.abs(t[k:]-t[:-k]-k*step) < 1e-6)
            if both.any():
                num += float(((x[:-k][both]-mu)*(x[k:][both]-mu)).sum())
                den += float(both.sum())
        if den < 50:
            break
        rho.append(num/(den*var))
        if len(rho) >= 2 and len(rho) % 2 == 0 and rho[-2]+rho[-1] <= 0:
            break
    return np.array(rho)


def decorrelation(t, step, series):
    """{'seconds' (e-folding: the first lag where the autocorrelation is below 1/e), 'iatSeconds' (the integrated
    autocorrelation time, Geyer's initial positive sequence)} of indicator series [(x, ok)]."""
    rho = _autocorrelation(t, step, series)
    if not len(rho):
        return {'seconds': None, 'iatSeconds': None}
    below = np.flatnonzero(rho < math.exp(-1))
    efold = (int(below[0])+1)*step if len(below) else len(rho)*step
    s = 1.0
    for k in range(0, len(rho)-1, 2):
        pair = rho[k]+rho[k+1]
        if pair <= 0:
            break
        s += 2*pair
    return {'seconds': round(float(efold), 1), 'iatSeconds': round(float(s*step), 1)}


def bound_coverage(t, step, units, values, level, tau, lengths=TAU_COVERAGE_SECONDS):
    """The share of professional windows whose one-sided STRENGTH_CONFIDENCE upper bound of the share beyond `level`
    (n_eff = seconds / tau) covers the unit's own share: {seconds: coverage}."""
    out = {}
    for T in lengths:
        K = int(round(T/step))
        hit = []
        for u in units:
            v = values[u['team']] if u['column'] is None else values[u['team']][:, u['column']]
            idx = np.flatnonzero(u['mask'])
            if len(idx) < K:
                continue
            beyond = v[idx] > level
            truth = float(beyond.mean())
            c = np.r_[0, np.cumsum(beyond)]
            for s0 in range(0, len(idx)-K+1, int(round(rf.WINDOW_SLIDE_SECONDS/step))):
                share = (c[s0+K]-c[s0])/K
                hit.append(truth <= rf.share_upper_bound(share, T/tau, STRENGTH_CONFIDENCE)+1e-12)
        if hit:
            out[str(int(T))] = round(float(np.mean(hit)), 3)
    return out


def _team_summary(values, unit, metric=None):
    q = quantiles(values, unit, metric)
    return {k: q.get(k) for k in ('n', 'p10', 'p50', 'p90')}


def _lite(q):
    return {k: q.get(k) for k in ('n', 'p10', 'p50', 'p75', 'p90', 'p95')}


def lateral_units(t, step, windows, halves, context):
    """Per team and half the lateral unit for the window counts: its windows, the samples they cover, the measured
    seconds."""
    n = max(2, int(round(LAG_WINDOWS[context]/step)))
    out = []
    for team, ws in windows[context].items():
        for h in np.unique(halves[halves > 0]):
            mine = [w for w in ws if w['half'] == int(h)]
            if not mine:
                continue
            cover = np.zeros(len(t), bool)
            for w in mine:
                cover[w['i0']:w['i0']+n] = True
            out.append({'team': team, 'half': int(h), 'column': None, 'windows': mine, 'mask': cover,
                        'measured': ep.measured_seconds(t, cover, step)})
    return out


def lateral_tau(units, level, step_seconds=rf.LAG_STEP_SECONDS):
    """The e-folding and integrated time of the beyond-`level` indicator over the lag windows: each run of
    consecutive windows (step_seconds apart) is one series; runs are kept apart so no pair spans two of them."""
    T, X, off = [], [], 0.0
    for u in units:
        ws = u['windows']
        k = 0
        while k < len(ws):
            j = k+1
            while j < len(ws) and abs(ws[j]['t0']-ws[j-1]['t0']-step_seconds) < 1e-6:
                j += 1
            run = ws[k:j]
            T.append(off+step_seconds*np.arange(len(run)))
            X.append(np.array([w['lag'] > level for w in run], float))
            off += step_seconds*len(run)+10*TAU_MAX_SECONDS
            k = j
    if not T:
        return {'seconds': None, 'iatSeconds': None}
    tt = np.concatenate(T)
    return decorrelation(tt, step_seconds, [(np.concatenate(X), np.ones(len(tt), bool))])


def build_reference(data, game=TUNING_GAME, rate=RATE):
    """The reference dict of a game (the tuning game only; see main for the refusals)."""
    clock = time.perf_counter()
    cd, play, halves, info = load(data, game, rate)
    t, step = cd.t, cd.step
    rule = dict(DEFAULT_RULE)
    everything = np.ones(len(t), bool)
    samples = collect(cd, everything)               # the primary distributions: dead ball included
    samples_play = collect(cd, play)                # documentation: in play only
    metrics, checks = {}, {}
    for metric, per_team in samples.items():
        unit, key, _, family, definition = METRICS[metric]
        contexts, by_team, episodes, windows, in_play_q = {}, {}, {}, {}, {}
        per_player = metric == 'nearestTeammate'
        for context in CONTEXTS[family]:
            pooled, parts = _pool(per_team, context)
            q = quantiles(pooled, unit)
            contexts[context] = q
            by_team[context] = {team: _team_summary(v, unit) for team, v in parts.items()}
            in_play_q[context] = _lite(quantiles(_pool(samples_play[metric], context)[0], unit))
            if q['n'] < rf.MIN_SAMPLES:
                continue
            units = context_units(t, halves, per_team, context, q[rule['enter']], q[rule['exit']], rule, step)
            base = episode_baseline(t, halves, per_team, context, q, rule, step, units=units)
            if base is not None:
                episodes[context] = {'default': base}

            def count(u, W):
                return span_counts(t, step, u['mask'], [(x['i0'], x['i1']) for x in u['episodes']], W,
                                   rule['minSeconds'])
            extra = WINDOW_CHECK_MINUTES if context in ('all', 'third:middle') and not per_player else ()
            entries = window_distribution(units, count, extra=extra)
            if extra:
                checks[f'{metric}|{context}'] = [e for e in entries if e['minutes'] in WINDOW_CHECK_MINUTES]
                entries = [e for e in entries if e['minutes'] not in WINDOW_CHECK_MINUTES]
            windows[context] = {'default': {'slideSeconds': rf.WINDOW_SLIDE_SECONDS, 'needSeconds': rule['minSeconds'],
                                            'minTeamShare': rf.WINDOW_MIN_TEAM_SHARE, 'entries': entries}}
        m = {'unit': unit, 'definition': definition, 'contexts': contexts, 'byTeam': by_team, 'episodes': episodes,
             ('playerWindowCounts' if per_player else 'windowCounts'): windows,
             'inPlay': {'contexts': in_play_q}}
        if per_player:
            m['perPlayer'] = True     # n counts player-samples, episodes and windows are each player's own
            m['tau'] = {'seconds': None, 'note': 'no strength is ever claimed on it (isolation is at most a review)'}
        else:
            level = contexts['all'].get(TAU_QUANTILE)
            series = [((v > level).astype(float), e & (halves == h)) for v, e, _ in per_team.values()
                      for h in np.unique(halves[halves > 0])]
            tau = decorrelation(t, step, series)
            all_units = context_units(t, halves, per_team, 'all', contexts['all']['p95'], contexts['all']['p90'], rule,
                                      step)
            values = {team: v for team, (v, _, _) in per_team.items()}
            tau['basis'] = f"share beyond the 'all' context's {TAU_QUANTILE} ({level})"
            if tau['seconds']:
                tau['coverage80'] = bound_coverage(t, step, all_units, values, level, tau['seconds'])
            if tau['iatSeconds']:
                tau['coverage80Integrated'] = bound_coverage(t, step, all_units, values, level, tau['iatSeconds'])
            m['tau'] = tau
        metrics[metric] = m
    windows_all = lateral(cd, everything, halves)
    windows_bound = lateral(cd, everything, halves, drop_bound=False)
    windows_play = lateral(cd, play, halves)
    for metric, key in (('lagDM', 'lag'), ('corrDM', 'corr')):
        unit, _, _, family, definition = METRICS[metric]
        contexts, by_team, cover, in_play_q, wcounts = {}, {}, {}, {}, {}
        for context, per_team in windows_all.items():
            parts = {team: np.array([w[key] for w in ws], float) for team, ws in per_team.items()}
            contexts[context] = quantiles(np.concatenate(list(parts.values())), unit, metric)
            by_team[context] = {team: _team_summary(v, unit, metric) for team, v in parts.items()}
            in_play_q[context] = _lite(quantiles(np.array([w[key] for ws in windows_play[context].values()
                                                           for w in ws], float), unit, metric))
            seconds = LAG_WINDOWS[context]
            covered = sum(_covered(ws, step) for ws in per_team.values())
            dropped = sum(len(ws) for ws in windows_bound[context].values())-sum(len(ws) for ws in per_team.values())
            cover[context] = {'windowSeconds': seconds, 'stepSeconds': rf.LAG_STEP_SECONDS,
                              'coveredMinutes': round(covered/60, 2), 'independentWindows': round(covered/seconds, 1),
                              'droppedAtSearchBound': int(dropped)}
            if metric == 'lagDM':
                units = lateral_units(t, step, windows_all, halves, context)
                enter = contexts[context].get(rule['enter'])

                def count(u, W, seconds=seconds, enter=enter):
                    return lateral_counts(t, step, u['mask'], u['windows'], seconds, enter, W)
                wcounts[context] = {'default': {'slideSeconds': rf.WINDOW_SLIDE_SECONDS, 'enter': enter,
                                                'minCorr': rf.LAG_MIN_CORR, 'minTeamShare': rf.WINDOW_MIN_TEAM_SHARE,
                                                'count': 'late windows among the non-overlapping ones',
                                                'entries': window_distribution(units, count)}}
        m = {'unit': unit, 'definition': definition, 'contexts': contexts, 'byTeam': by_team, 'episodes': {},
             'windows': cover, 'inPlay': {'contexts': in_play_q}}
        if metric == 'lagDM':
            m['windowCounts'] = wcounts
            units = lateral_units(t, step, windows_all, halves, 'all')
            tau = lateral_tau(units, contexts['all'].get(TAU_QUANTILE))
            level = contexts['all'].get(TAU_QUANTILE)
            tau['basis'] = f"share of 30 s windows beyond the 'all' context's {TAU_QUANTILE} ({level})"
            m['tau'] = tau
        metrics[metric] = m
    base_share = {team: _round(team_base(s, everything).mean(), 3) if len(t) else None for team, s in cd.teams.items()}
    base_share_play = {team: _round(team_base(s, play)[play].mean(), 3) if play.any() else None
                       for team, s in cd.teams.items()}
    build = dict(info, tool='tools/coaching_bench.py', engineVersion=ep.ENGINE_VERSION, baseShare=base_share,
                 baseShareOfInPlay=base_share_play, dataSha256=data_files(data, game),
                 windowBoundCheck=bound_check(metrics, checks), totalSeconds=round(time.perf_counter()-clock, 1))
    return {'version': 1, 'source': f'metrica-sample-game-{game}', 'sampleRate': float(rate),
            'created': datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0).isoformat(),
            'notes': notes(game), 'contexts': CONTEXT_NOTES, 'metrics': metrics, 'rules': {'default': rule},
            'build': build,
            'validation': {'status': 'not run', 'heldOut': [f'metrica-sample-game-{g}' for g in HELD_OUT],
                           'how': 'python -I -B tools/coaching_bench.py --data DIR --game 2 --validate',
                           'previous': PREVIOUS_VALIDATIONS}}


# The validations of the references a rebuild replaces (their ids differ; recorded so the history travels with the
# file).
# A rebuild is a new reference (its 'created' and build times differ, so its name does): it starts 'not run'.
PREVIOUS_VALIDATIONS = [
    {'reference': 'metrica-sample-game-1@e8410fd48fca5ddc', 'date': '2026-10-18', 'verdict': 'FAIL',
     'summary': 'in-play reference with a Poisson rate test at the professional p90: on professional play 1.31 '
                '(game 2) and 1.44 (game 1) improvement findings per 10 team-minutes with 10-minute windows on '
                'truth, most from the isolation detector (one player against team-pooled rates, no correction for '
                'ten players), one-team baselines (back four, own-third M-F gap) and dead-ball lateral windows'},
    {'reference': 'metrica-sample-game-1@5e3b95a6d2534532', 'file': 'e8dc19cdfecc8219', 'date': '2026-10-19',
     'verdict': 'PASS',
     'summary': 'dead-ball-included reference with window-count baselines (improvement beyond the professional p99 '
                'count of the next longer window length): on game 2 0.00 / 0.05 / 0.00 / 0.11 (truth) and 0.00 / '
                '0.05 / 0.00 / 0.05 (60 m camera) improvement findings per 10 team-minutes with 1, 3, 10-minute '
                'and whole-half windows, no high-confidence concern'}]
ACCEPTANCE_MAX_PER_10_TEAM_MINUTES = .2     # the pre-declared false-positive test: improvements on professional play
ACCEPTANCE_CONCERNS = 0                     # ... and high-confidence concerns allowed


SUMMARY_WINDOW_SHARE = .01     # a window distribution whose pooled held-out share beyond its p99 exceeds this is listed
SUMMARY_WINDOW_ROW = .05       # ... and so is any length beyond this with at least SUMMARY_WINDOW_MIN_N windows
SUMMARY_WINDOW_MIN_N = 100     # (a whole-half row has 4 team-halves: one count beyond reads .25)


def validation_summary(report):
    """The validation report's headline numbers for the record: per metric the share of the held-out samples at or
    above the stored p90 and p95 in the 'all' context (expected .10 / .05), the metric-context pairs whose share
    above the p90 is beyond .15 (looser than expected: more flags) or below .05 (stricter), the team-halves whose
    episode rate lies in the tuning game's range, and per both-team window distribution (the improvement test's
    baseline) the held-out team-windows beyond the stored p99 count pooled over its lengths (expected at most .01;
    the distributions above SUMMARY_WINDOW_SHARE are listed with their worst well-populated lengths). On game 2 the
    pooled shares were 0-0.6% for width, depth, area, spread, both unit gaps, the D and M players' distances and the
    30 s lag windows (1.3% for 20 s windows), but 2.8-10.5% for the back-line height (only the level of a review),
    2.9-4.0% for the back-line spread in the 'all' and middle-third contexts (the back-line detector compares by
    back-line size, one team's each on game 1, so no improvement rests on them) and 2.8% for the forwards'
    nearest-teammate distance (a role-check review)."""
    shares, looser, stricter, within, pooled, listed = {}, [], [], {}, {}, []
    for metric, res in (report.get('metrics') or {}).items():
        for context, r in res.items():
            at = r.get('shareAtOrAbove') or {}
            p90 = (at.get('p90') or {}).get('observed')
            if context == 'all':
                shares[metric] = {'p90': p90, 'p95': (at.get('p95') or {}).get('observed'), 'n': r.get('n')}
                halves = [h.get('withinTuningRange') for h in (r.get('episodes') or {}).get('perHalf') or ()]
                if halves:
                    within[metric] = f'{sum(1 for h in halves if h)}/{len(halves)}'
            if isinstance(p90, (int, float)) and (r.get('n') or 0) >= rf.MIN_SAMPLES:
                (looser if p90 > .15 else stricter if p90 < .05 else []).append(f'{metric}|{context} {p90}')
            for key in ('windowCounts', 'playerWindowCounts'):
                rows = [x for x in r.get(key) or () if x.get('bothTeams') and x.get('n')
                        and isinstance(x.get('beyondP99'), (int, float))]
                if not rows:
                    continue
                share = sum(x['n']*x['beyondP99'] for x in rows)/sum(x['n'] for x in rows)
                pooled[f'{metric}|{context}'] = _round(share, 4)
                worst = [f"{x['minutes']:g} min {x['beyondP99']}" for x in rows
                         if x['beyondP99'] > SUMMARY_WINDOW_ROW and x['n'] >= SUMMARY_WINDOW_MIN_N]
                if share > SUMMARY_WINDOW_SHARE or worst:
                    listed.append(f'{metric}|{context} pooled {_round(share, 4)}' + (f" ({', '.join(worst)})" if worst
                                                                                       else ''))
    return {'shareAtOrAboveAll': shares, 'looserThanExpected': looser, 'stricterThanExpected': stricter,
            'episodeHalvesWithinTuningRange': within, 'windowShareBeyondP99': pooled,
            'windowShareBeyondP99Listed': listed}


def validation_record(reference, report, acceptance=None, file_sha=None, date=None):
    """The 'validation' block for a validation report of the frozen reference (see the module docstring, --record).
    acceptance: the pre-declared false-positive test's summary ({'verdict', 'referenceFileSha256', 'cells', ...})."""
    old = reference.get('validation') or {}
    verdict = 'not judged'
    if acceptance is not None:
        if file_sha is not None and str(acceptance.get('referenceFileSha256') or '')[:16] != file_sha[:16]:
            raise Refused('the acceptance summary was made against another reference file')
        verdict = acceptance.get('verdict') if acceptance.get('verdict') in ('PASS', 'FAIL') else 'not judged'
    return {'status': 'run', 'verdict': verdict,
            'date': date or datetime.date.today().isoformat(),
            'reference': rf.reference_id(dict(reference, _sha=rf.content_sha(reference))),
            'referenceFileSha256': file_sha, 'game': report.get('game'),
            'heldOut': old.get('heldOut') or [f'metrica-sample-game-{g}' for g in HELD_OUT],
            'how': 'python -I -B tools/coaching_bench.py --data DIR --game 2 --validate --record --acceptance FP.json',
            'summary': validation_summary(report),
            'falsePositiveTest': acceptance,
            'previous': old.get('previous') or PREVIOUS_VALIDATIONS[:1]}


def record_validation(path, block):
    """Write block as the reference file's 'validation' and change nothing else: the reference's identity
    (reference.content_sha) is checked unchanged before the file is replaced (write_json, CRLF)."""
    path = Path(path)
    ref = json.loads(path.read_bytes().decode('utf-8'))
    before = rf.content_sha(ref)
    old = ref.get('validation') or {}
    if old.get('status') == 'run' and old.get('reference') == block.get('reference'):
        raise Refused(f"{block.get('reference')} is validated already: the held-out game is read once per reference")
    ref['validation'] = block
    if rf.content_sha(ref) != before:
        raise Refused('recording the validation would change the reference')
    write_json(path, ref)
    return before


def bound_check(metrics, checks):
    """reference.window_baseline's estimate at lengths measured directly (WINDOW_CHECK_MINUTES) against the direct
    levels: {key: [{minutes, level, direct, estimate}]}, estimate >= direct where the bound is conservative."""
    out = {}
    ref = {'metrics': metrics}
    for key, entries in checks.items():
        metric, context = key.split('|')
        rows = []
        for e in entries:
            if not e.get('bothTeams'):
                continue
            est = rf.window_baseline(ref, metric, context, e['minutes'])
            if not est.get('available'):
                continue
            for level in ('p50', 'p99', 'max'):
                rows.append({'minutes': e['minutes'], 'level': level, 'direct': e[level], 'estimate': est[level]})
        out[key] = rows
    return out


def _covered(windows, step):
    """Seconds covered by the union of the windows."""
    total, end = 0.0, -np.inf
    for w in windows:
        a, b = w['t0'], w['t1']+step
        if b > end:
            total += b-max(a, end)
            end = b
    return total


def notes(game):
    return (f'Metrica Sports sample game {game}: one professional match, both teams pooled (team A a back four, team B '
            'a back three), measured by soccer.coaching.series on the result tools/team_shape_bench.build_result makes '
            'of it (5 samples a second, the half-time side swap set). All the time with all 10 outfield players of '
            'the team observed, dead ball included, because a PitchIQ result has no in-play mask; the in-play '
            'variant is kept per metric as inPlay. No possession phase: Metrica has no event record here and '
            'PitchIQ records none, so every distribution mixes the phases (a team is about 10 m wider with the '
            'ball). Each team spends about 10% of its time above the p90 by construction: a count is compared with '
            'the professional window distribution of the same rule, context and measured minutes (windowCounts), '
            'never read on its own. A sanity range from one match, not a standard; game 2 is held out for '
            'validation.')


# ---------- validation on the held-out game (thresholds frozen) ----------
def validate(reference, data, game, rate=None):
    """The held-out game measured against the frozen reference: per metric and context the share of its samples at or
    above each stored quantile (expected about 1 - q), per episode baseline its per-team-half rates against the
    tuning game's range, and per window distribution the share of its team-windows (players' windows for the
    nearest-teammate distance) whose count is beyond the stored p99 and p999 (expected at most 1% and 0.1%). All the
    time, dead ball included, as the reference. Nothing is tuned and nothing is written to the reference."""
    rate = float(rate or reference.get('sampleRate') or RATE)
    cd, play, halves, info = load(data, game, rate)
    t, step = cd.t, cd.step
    rule = rf.rule(reference, 'default')
    everything = np.ones(len(t), bool)
    samples = collect(cd, everything)
    windows = lateral(cd, everything, halves)
    out = {'reference': rf.reference_id(reference), 'game': int(game), 'build': info, 'metrics': {}}
    for metric, entry in (reference.get('metrics') or {}).items():
        if metric not in METRICS:
            continue
        res = {}
        for context, q in (entry.get('contexts') or {}).items():
            if METRICS[metric][1] is None:
                key = 'lag' if metric == 'lagDM' else 'corr'
                pooled = np.array([w[key] for ws in windows.get(context, {}).values() for w in ws], float)
            elif metric in samples and context in next(iter(samples[metric].values()))[2]:
                pooled, _ = _pool(samples[metric], context)
            else:
                continue
            shares = {}
            for key, level in zip(rf.QUANTILES, QUANTILE_LEVELS):
                if q.get(key) is not None and len(pooled):
                    shares[key] = {'expected': round(1-level/100, 3), 'observed': _round(np.mean(pooled >= q[key]), 3)}
            res[context] = {'n': int(len(pooled)), 'nReference': q.get('n'), 'shareAtOrAbove': shares}
        for context, rules in (entry.get('episodes') or {}).items():
            base = (rules or {}).get('default')
            if not base or metric not in samples or context not in res:
                continue
            thr = base.get('thresholds') or {}
            q = {rule['enter']: thr.get('enter'), rule['exit']: thr.get('exit')}
            now = episode_baseline(t, halves, samples[metric], context, q, rule, step, allow_sparse=True)
            if now is None:
                continue
            ref_rates = [r['perMinute'] for r in base.get('perHalf') or ()
                         if r.get('inSpread') and r.get('perMinute') is not None]
            lo, hi = (min(ref_rates), max(ref_rates)) if ref_rates else (None, None)
            for r in now['perHalf']:
                r['withinTuningRange'] = (None if lo is None or r['perMinute'] is None or not r['inSpread']
                                          else bool(lo <= r['perMinute'] <= hi))
            res[context]['episodes'] = {'tuningRange': {'min': lo, 'max': hi}, 'tuningPerMinute': base.get('perMinute'),
                                        'perHalf': now['perHalf'], 'perMinute': now['perMinute'],
                                        'shareOfTime': now['shareOfTime'], 'tuningShareOfTime': base.get('shareOfTime')}
        for wkey in ('windowCounts', 'playerWindowCounts'):
            for context, rules in (entry.get(wkey) or {}).items():
                block = (rules or {}).get('default') or {}
                if context not in res or not block.get('entries'):
                    continue
                if metric == 'lagDM':
                    units = lateral_units(t, step, windows, halves, context)
                    enter = block.get('enter')

                    def count(u, W, seconds=LAG_WINDOWS[context], enter=enter):
                        return lateral_counts(t, step, u['mask'], u['windows'], seconds, enter, W)
                else:
                    q = entry['contexts'][context]
                    units = context_units(t, halves, samples[metric], context, q[rule['enter']], q[rule['exit']], rule,
                                          step)

                    def count(u, W):
                        return span_counts(t, step, u['mask'], [(x['i0'], x['i1']) for x in u['episodes']], W,
                                           rule['minSeconds'])
                rows = []
                for e in block['entries']:
                    c = [count(u, e['minutes']) for u in units]
                    c = np.concatenate([x for x in c if x is not None and len(x)]) if any(
                        x is not None and len(x) for x in c) else np.zeros(0)
                    if len(c):
                        rows.append({'minutes': e['minutes'], 'n': int(len(c)), 'bothTeams': e.get('bothTeams'),
                                     'p99': e['p99'], 'beyondP99': _round(np.mean(c > e['p99']), 4),
                                     'beyondP999': _round(np.mean(c > e['p999']), 4)})
                res[context][wkey] = rows
        out['metrics'][metric] = res
    return out


def summary_lines(reference=None, validation=None):
    """A short printed table: the 'all' quantiles, the window p99s and tau, or the validation's headline."""
    lines = []
    if reference is not None:
        for metric, m in reference['metrics'].items():
            q = m['contexts'].get('all') or {}
            w = ((m.get('windowCounts') or m.get('playerWindowCounts') or {}).get('all') or {}).get('default') or {}
            p99 = ' '.join(f"{e['minutes']:g}:{e['p99']:g}{'' if e['bothTeams'] else '*'}" for e in w.get('entries') or ())
            tau = (m.get('tau') or {})
            lines.append(f"{metric:16s} n={q.get('n')} p10/p50/p90/p95 {q.get('p10')}/{q.get('p50')}/{q.get('p90')}/"
                         f"{q.get('p95')} {m['unit']}  window p99 {p99 or '-'}  tau {tau.get('seconds')} s "
                         f"(integrated {tau.get('iatSeconds')}, coverage {tau.get('coverage80')})")
            ip = (m.get('inPlay') or {}).get('contexts', {}).get('all') or {}
            if ip:
                lines.append(f"{'':16s} in play p10/p50/p90 {ip.get('p10')}/{ip.get('p50')}/{ip.get('p90')}")
    if validation is not None:
        for metric, res in validation['metrics'].items():
            a = res.get('all') or {}
            s90 = (a.get('shareAtOrAbove') or {}).get('p90') or {}
            eps = a.get('episodes') or {}
            within = [r.get('withinTuningRange') for r in eps.get('perHalf') or ()]
            wins = a.get('windowCounts') or a.get('playerWindowCounts') or []
            beyond = ' '.join(f"{r['minutes']:g}:{r['beyondP99']}" for r in wins)
            lines.append(f"{metric:16s} n={a.get('n')} share >= p90 {s90.get('observed')} (expected {s90.get('expected')})"
                         f"  team-halves within the game-1 range {sum(1 for w in within if w)}/{len(within)}"
                         f"  windows beyond p99 {beyond or '-'}")
    return lines


def write_json(path, obj):
    """JSON (allow_nan=False) with CRLF line endings, written atomically."""
    text = json.dumps(obj, indent=1, allow_nan=False, ensure_ascii=True)+'\n'
    path = Path(path)
    fd, tmp = tempfile.mkstemp(prefix=path.name+'.', suffix='.tmp', dir=str(path.parent))
    try:
        with os.fdopen(fd, 'w', encoding='utf-8', newline='\r\n') as fh:
            fh.write(text)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


# ---------- command line ----------
def check(a, reference_game=None):
    """The refusals, before any data is read."""
    if not a.validate and (a.record or a.acceptance):
        raise Refused('--record and --acceptance belong to --validate')
    if a.acceptance and not a.record:
        raise Refused('--acceptance is recorded with --record')
    if not a.validate:
        if a.game in HELD_OUT:
            raise Refused(f'game {a.game} is held out: only --validate reads it, with the thresholds frozen')
        if a.game != TUNING_GAME:
            raise Refused(f'the reference is built from game {TUNING_GAME} only (got --game {a.game})')
        if a.json:
            raise Refused('--json is the validation report; the reference goes to --out')
        return
    if a.out:
        raise Refused('--validate never writes the reference (it is frozen); use --json for the validation report')
    if reference_game is not None and a.game == reference_game:
        raise Refused(f'the reference was built from game {a.game}: validating on it proves nothing')
    if a.json and Path(a.json).resolve() == Path(a.reference).resolve():
        raise Refused('--json would overwrite the reference')


def main(argv=None, isolated=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', required=True, help='directory of the Metrica sample tracking files')
    ap.add_argument('--game', type=int, required=True)
    ap.add_argument('--out', help='where to write the reference (build mode)')
    ap.add_argument('--rate', type=float, default=RATE, help='samples a second (build mode)')
    ap.add_argument('--validate', action='store_true', help='measure the held-out game against the frozen reference')
    ap.add_argument('--reference', default=str(rf.REFERENCE_PATH), help='the frozen reference (--validate)')
    ap.add_argument('--json', help='where to write the validation report (--validate)')
    ap.add_argument('--record', action='store_true',
                    help="write the validation's summary into the reference's 'validation' block (--validate)")
    ap.add_argument('--acceptance', help="the pre-declared false-positive test's summary JSON, recorded with --record")
    a = ap.parse_args(argv)
    reference = None
    if a.validate:
        check(a)
        reference = rf.load_reference(a.reference)
        built = (reference.get('build') or {}).get('game')
        check(a, int(built) if isinstance(built, int) else TUNING_GAME)
    else:
        check(a)
    if isolated is None:
        isolated = bool(sys.flags.isolated and sys.flags.dont_write_bytecode)
    if not isolated:
        raise Refused('the Metrica files are downloaded data: run the bench as python -I -B tools/coaching_bench.py ...')
    if a.validate:
        acceptance = json.loads(Path(a.acceptance).read_bytes().decode('utf-8')) if a.acceptance else None
        file_sha = hashlib.sha256(Path(a.reference).read_bytes()).hexdigest()
        if acceptance is not None:
            validation_record(reference, {}, acceptance, file_sha)        # refuse a foreign summary before reading
        report = validate(reference, a.data, a.game)
        if a.json:
            write_json(a.json, report)
        else:
            json.dumps(report, allow_nan=False)
        print('\n'.join(summary_lines(validation=report)))
        if a.record:
            block = validation_record(reference, report, acceptance, file_sha)
            record_validation(a.reference, block)
            print(f"recorded in {a.reference}: {block['reference']} {block['verdict']} ({block['date']})")
        return report
    ref = build_reference(a.data, a.game, a.rate)
    if a.out:
        write_json(a.out, ref)
    else:
        json.dumps(ref, allow_nan=False)
    print('\n'.join(summary_lines(reference=ref)))
    print(f"built in {ref['build']['totalSeconds']} s" + (f'; written to {a.out}' if a.out else ' (not written: no --out)'))
    return ref


if __name__ == '__main__':
    main()
