"""Coaching observations: what one team did over an analysed range, measured on the coaching series
(soccer/coaching/series.py), counted as episodes (soccer/coaching/episodes.py) and compared with the professional
reference (soccer/coaching/reference.py). Each detector returns Observation records (contract section 5) whose
classification follows the explicit rules below; the words come later (language.py), the evidence moments from
evidence.py.

Why a count is never a finding on its own. On the Metrica Sports professional sample (game 1, 5 samples a second,
both teams) a team spends about 10% of its time above the game's own p90 of every metric by construction, and even
the strict default rule (enter above p95, leave below p90, at least 5 s, gaps under 3 s merged) fires 3-16 times per
team and half; the episodes cluster (counts differ up to 3x between the halves of one team, and within a half one
10-minute stretch can hold two and a half times the half's rate). A Poisson test at the professional p90 rate called
professional play a weakness 1.3-1.4 times per 10 team-minutes on game 1 itself. So a count is set against the
distribution of the counts professional team-windows of the same measured length showed, under the same rule and in
the same context (reference.window_baseline), and an observation says how long it could be measured. Below
MIN_MEASURED_SECONDS nothing is assessed. Measured on game 1 over non-overlapping 1-, 3- and 10-minute windows and the
four team-halves (the validation stage's false-positive scripts), dead ball included: 0.00 / 0.05 / 0.00 / 0.00
improvement findings per 10 team-minutes on the truth (before: 5.52 / 3.92 / 1.44 / 0.78) and 0.00 / 0.11 / 0.06 / 0.05
under a simulated 60 m camera with correlated position noise (before: 1.09 / 1.40 / 0.44 / 0.16), no high-confidence
concern in either (before: up to 0.25). On the held-out game 2, run once with the frozen reference (2026-10-19, the
validation block of reference.json): 0.00 / 0.05 / 0.00 / 0.11 on the truth and 0.00 / 0.05 / 0.00 / 0.05 under the
camera, no high-confidence concern, against the pre-declared limit of 0.2 (the replaced detectors: 1.31 on the truth
and 0.44 under the camera with 10-minute windows); the remaining findings are team width and length spells counted
one or two beyond the professional p99 count, at most the professional maximum.

Contexts. The reference is split where the professional ranges differ: the D-M gap's p90 is 12-13 m with the team's
centre in its own third and 26-28 m in the final third, and the back line's depth spread is 2.8/7.8 m (p50/p90) behind
Metrica team A's back four but 2.2/4.9 m behind team B's back three. So an observation is made per context: the team's
third (width, depth, area, unit gaps, line height with a gap), the back-line size (back-line evenness) and the
player's unit (support distance). Its samples are those measured in that context only, its thresholds and window
distribution are that context's, and its text names it; a context the reference does not have (fewer than
reference.MIN_SAMPLES samples) falls back to 'all' and is pooled with the others that do. A context with less than
MIN_MEASURED_SECONDS of measurement is left out (its seconds are reported as otherContextSeconds) unless no context
reaches it, in which case one observation pools every sample under the 'all' thresholds.

Classification (constants below; every rule is tested in tests/test_coaching_detectors.py):
  insufficient  no professional reference for the metric ('no_reference'), less than MIN_MEASURED_SECONDS measured
                ('short_measured'), no continuous measured stretch as long as the rule's minimum duration
                ('short_stretch'), or an overall confidence of low ('low_confidence'). Listed under limitations,
                never as strength or weakness.
  improvement   at least MIN_OCCURRENCES occurrences and more than the professional IMPROVEMENT_LEVEL (p99) count of
                windows of the same measured time in the same context (reference.window_baseline: an upper estimate of
                that level, so the comparison never favours a finding), on a window distribution of both professional
                teams ('one_team_baseline' otherwise: no improvement can be claimed there). No single occurrence is
                ever an improvement, however long. Severity 'high_confidence_concern' only with high confidence, at
                least SHORT_SEGMENT_SECONDS measured, a count at least CONCERN_MARGIN times that p99 and beyond every
                professional window of that length (the maximum). Otherwise 'potential_improvement'.
  strength      (strength_check) measured at least max(STRENGTH_MIN_SECONDS, min(STRENGTH_FULL_SECONDS,
                STRENGTH_RANGE_SHARE x the analysed range)); the median on the favourable side of the professional p50
                in the same context (narrower, shorter, smaller than the median professional team: a wide team is
                never called compact); the share of the measured time beyond the professional p75, over n_eff =
                measured / tau independent looks (tau: the metric's decorrelation time in the reference), with a
                one-sided STRENGTH_BOUND_CONFIDENCE upper bound below STRENGTH_PRO_SHARE (the professional share beyond
                its own p75); high or medium confidence; no excess of episodes (none, or no more than the professional
                window median at that measured time). A pooled observation over several contexts is a strength only
                when every context with enough measurement for a strength passes in its own context, and it quotes
                the strictest of those comparisons (the highest upper bound). Worded as "maintained", never
                "excellent".
  review        context-dependent readings that are not mistakes: a sustained change between unrelated formations, a
                formation reading ambiguous between unrelated families, a high back line together with a large gap to
                the first midfield line, a player repeatedly far from his nearest team-mate (isolation: always a role
                check, never an improvement).
  neutral       measured enough, and neither of the above. Its 'state' says which: 'absent' (no spell beyond the
                professional level), 'one_spell' (one spell, its times and peak in metrics.spell), 'within_range'
                (several spells, within the professional window range), 'under_measured' (no spell, but too little
                measurement for a strength); formation and ball readings are 'descriptive' or 'related_shapes'.
Low confidence never yields an improvement or a concern, and a short segment (under 3 minutes measured) never claims a
recurring pattern: language.py words one occurrence as one observed instance.

Confidence per aspect (contract section 5): calibration from the share of the measured samples on a HIGH registration
(>= .7; broadcast HIGH frames are 0.5 m median off, MEDIUM 1.6 m with a 5 m p90), capped at medium on a keyless run
(the amateur clip stays at .38-.41 whatever the error, which is a size bias of up to 4 m on the far touchline);
tracking from how much of the range the metric was measured and its longest stretch, capped at medium on a
small-scene run and where the stabilizer flagged the involved players' boxes on more than FLAG_MEDIUM_SHARE of the
samples, or where the units come from an ambiguous formation reading on at least half of them (broadcast 42-222 s team
A: 4-3-3 ambiguous, role stability .50, eight back-line group changes); identity from the share of the involved
players' time within IDENTITY_TAINT_SECONDS of a crossing, re-identification or uncertain swap (identityConfidence
plateaus at .85 on 83% of broadcast observations and is no probability, so it is not used). Overall is the worst
aspect.

Detectors (registry DETECTORS, in this order):
  compactness            width, depth and convex-hull area of the outfield team (patterns team_width, team_depth,
                         team_area) by third; complete or reliable samples only (Metrica camera simulation: p50
                         1.2-1.5 m, p90 3-4 m off; a partial value is 3-5 m short, a partial hull 40% too small).
  unit_gap_dm            back line to the first midfield line (template line index 1), centre to centre, by third;
                         fully seen units only (p90 2.1-2.6 m off; one member missing reads 2.1-2.6 m short).
  unit_gap_mf            midfield (every middle line) to the forward line, by third.
  back_line_coordination std of the defenders' distance from their goal, by back-line size, with the coherent /
                         uneven / stretched shares of team_shape's line quality. Game 1 has one team per back-line size
                         (A a back four, B a back three), so its window distributions are one-team and no improvement
                         is claimed on it (the stored reason says so); a strength still compares with that team.
  line_height_and_gap    a back line higher than the professional p90 line height while the D-M gap is beyond the
                         context's enter level: review, never "a high line is bad". Otherwise a neutral context note
                         with the team's and the opponent's median back-line height (block height).
  formation_stability    team_shape's stabilized periods (registered samples only): label confidence, each player's
                         share of time in his dominant role, back-line group changes between complete windows. No
                         professional distribution exists for these, so the rules are team_shape's own confidence
                         levels, a reading is at most medium confidence (FORMATION_MAX_CONFIDENCE), shapes of one
                         family (team_shape.RELATED: 4-3-3 / 4-1-4-1, ...) are one shape, and a stable reading is a
                         strength only over FORMATION_STRENGTH_SECONDS (holding a shape for a minute is the baseline).
  lateral_shift          the delay of the back line's sideways movement behind the midfield's (positive: the back line
                         later), computed by reference.py's own functions so that a team and the professional windows
                         are the same code: 30 s windows every 5 s (20 s windows when no stretch holds 30 s); the count
                         is the late windows among the non-overlapping ones, against the professional window
                         distribution of that count; lags stated at the 5 Hz resolution.
  isolation              per player, the certain nearest-teammate distance (the disc in view; broadcast only:
                         small-scene runs are vetoed by the series), against the per-player professional window
                         distribution, with Holm's correction over the players tested (ISOLATION_ALPHA family-wise):
                         a rejected player with at least MIN_OCCURRENCES spells is a review (a winger staying wide is a
                         role, not a fault); everyone else is summarised in one team-level observation.
  ball_context           where the measured ball was (team-relative thirds, channels): descriptive, always neutral.

Identities are internal ids everywhere (people[].id); display labels are rendered at view time. Ids of observations
and occurrences are stable (episodes.stable_id) over the detector, the team, the pattern and context (and the player)
and the rounded times, so regenerating the same analysis gives the same ids.
"""
import math
from collections import Counter

import numpy as np

from .. import pitch_model, team_shape as ts
from . import episodes as ep
from . import reference as refs
from . import series as cs

ENGINE_VERSION = ep.ENGINE_VERSION

# ---------- classification rules ----------
MIN_MEASURED_SECONDS = 20.0     # s: less measured time is not assessed (contract section 5)
SHORT_SEGMENT_SECONDS = 180.0   # s: below 3 min measured nothing is recurring and nothing is a concern
MIN_OCCURRENCES = 2             # occurrences an improvement (and an isolation review) needs: never a single one
IMPROVEMENT_LEVEL = 'p99'       # the professional window level an improvement's count must exceed
CONCERN_MARGIN = 1.5            # a high-confidence concern: at least this times the professional p99 count, and beyond
                                # the professional maximum (on game 1 the maximum is the p99 or one more: 1.0-1.5x it)
STRENGTH_MIN_SECONDS = 30.0     # s: a strength needs at least this much measured time ...
STRENGTH_FULL_SECONDS = 60.0    # s: ... and this much, or STRENGTH_RANGE_SHARE of a shorter analysed range
STRENGTH_RANGE_SHARE = .8
STRENGTH_LEVEL = 'p75'          # the professional level a strength's share is counted beyond
STRENGTH_PRO_SHARE = .25        # the professional share beyond its own p75 (by construction)
STRENGTH_BOUND_CONFIDENCE = .8  # one-sided upper bound of the share that must stay below STRENGTH_PRO_SHARE
TAU_FALLBACK_SECONDS = 10.0     # s: the decorrelation time when the reference has none (game 1: 7.6-13.6 s per metric)
PRO_SHARE_ABOVE_P90 = .10       # share of professional time beyond its own p90: 10% by construction of the quantile
DEFAULT_RULE = {'enter': 'p95', 'exit': 'p90', 'minSeconds': 5.0, 'mergeSeconds': 3.0}   # reference.json's default
NO_REFERENCE = 'no_reference'
DOMINANT_CONTEXT_SHARE = .9     # one context with this share of the measured time is the observation's context
ISOLATION_ALPHA = .01           # family-wise level of the isolation test over a team's players (Holm), the same 1% the
                                # team tests' p99 stands for

# ---------- confidence ----------
HIGH_CALIBRATION = pitch_model.HIGH     # .7: a HIGH registration
CAL_HIGH_SHARE = .6                     # share of measured samples on HIGH for a 'high' calibration aspect
TRACK_HIGH_SHARE = .5                   # share of the range measured ...
TRACK_HIGH_STRETCH = 20.0               # ... and longest stretch (s) for a 'high' tracking aspect
TRACK_MEDIUM_STRETCH = 5.0              # s: below this longest stretch tracking is 'low'
FLAG_MEDIUM_SHARE = .25                 # stabilizer flags on more of the measured samples cap tracking at medium
AMBIGUOUS_LOW_SHARE = .5                # units from an ambiguous formation reading on this share: tracking low
IDENTITY_TAINT_SECONDS = 2.0            # s around a crossing / re-identification / uncertain swap (data audit)
IDENTITY_HIGH, IDENTITY_MEDIUM = .05, .15            # tainted share (team-level, per player weighted)
PLAYER_IDENTITY_HIGH, PLAYER_IDENTITY_MEDIUM = .10, .30   # tainted share of one player's measured time
TAINT_ISSUES = ('crossed', 're-identified', 'stopped matching', 'kit')
TAINT_EVENTS = ('swap-uncertain', 'switch', 'team-change')
FEW_PLAYERS = 4.0                       # mean observed outfield players per team below this: 'few_players' (Veo 2-2.6)
REASON_MIN_SHARE = .10                  # a not-measured reason is listed when it covers this share of the range

# ---------- detector constants ----------
THIRDS = ('own', 'middle', 'final')
BACKLINE_SIZES = (3, 4, 5)
FORMATION_MIN_WINDOWS = 3               # classified formation windows (15 s, 5 s apart) for any formation reading
FORMATION_MAX_SET_CHANGES = 1           # back-line group changes (between complete windows) a stable reading may have
FORMATION_MIN_ROLE_STABILITY = ts.CLEAR_ROLE_STABILITY    # .7: share of players keeping one role for a stable reading
FORMATION_ROLE_SHARE = .75              # a player kept one role when it was his for this share of the time (team_shape)
FORMATION_STRENGTH_SECONDS = 300.0      # s: a stable formation is a strength only over this much measured time
FORMATION_SUSTAINED_SECONDS = 60.0      # s: each of two unrelated formations must hold this long for a change review
FORMATION_MEDIUM_IDENTITY = .45         # mean identity confidence of the periods below which identity is low (a 'clear'
                                        # period needs team_shape.CLEAR_IDENTITY .6; AM's offline-assigned ids are .6-.74)
FORMATION_MAX_CONFIDENCE = 'medium'     # a formation reading is never high confidence: labels have no ground truth
                                        # (README: no formation accuracy claimed), and the same physical team read
                                        # 3-4-3 'clear' on broadcast 0-60 s but 4-3-3 'ambiguous' on 42-222 s (both
                                        # three deep defenders, a holding midfielder, four, then two, by mean depth)
LAG_WINDOW_SECONDS = refs.LAG_WINDOW_SECONDS     # 30 s: the reference's lag windows (every 5 s)
LAG_SHORT_WINDOW_SECONDS = 20.0         # s: the contract's shortest continuous stretch; its levels are 'window:20'
LAG_SHORT_CONTEXT = 'window:20'         # the reference context of 20 s windows (lag p90 1.0 s against 1.0 s at 30 s)
LAG_MIN_CORR = refs.LAG_MIN_CORR        # a window whose best correlation is lower is never counted as late
NEAREST_CONTEXT = 'unit:{}'             # reference context of the nearest-teammate distance by the player's unit
LINE_HEIGHT_ENTER, LINE_HEIGHT_EXIT = 'p90', 'p75'   # a high line: beyond the professional p90 height, until p75

DETECTOR_KEYS = {   # detector -> (recommendation key, category)
    'compactness': ('compact_shape', 'spacing'),
    'unit_gap_dm': ('compact_lines', 'spacing'),
    'unit_gap_mf': ('compact_lines', 'spacing'),
    'back_line_coordination': ('back_line_coordination', 'defensive_line'),
    'line_height_and_gap': ('line_height_gap', 'defensive_line'),
    'formation_stability': ('formation_organisation', 'formation'),
    'lateral_shift': ('lateral_shift', 'movement'),
    'isolation': ('support_distances', 'support'),
    'ball_context': ('ball_context', 'ball_context'),
}
UNITS_OF = {'width': 'm', 'depth': 'm', 'hull': 'm2', 'spread': 'm', 'gapDM': 'm', 'gapMF': 'm', 'lineStd': 'm',
            'lineHeight': 'm', 'nearestTeammate': 'm', 'lagDM': 's', 'corrDM': 'ratio'}


# ---------- small helpers ----------
def _num(x):
    return isinstance(x, (int, float, np.floating, np.integer)) and not isinstance(x, bool) and math.isfinite(float(x))


def _round(x, digits):
    return None if x is None or not _num(x) else round(float(x), digits)


def _t(x):
    return _round(x, 3)         # an absolute or segment time (frame exact at 30 fps)


def _sec(x):
    return _round(x, 2)         # a duration


def _share(x):
    return _round(x, 3)


def _val(metric, x):
    unit = UNITS_OF.get(metric, 'm')
    if metric == 'lagDM':
        return refs.quantize_lag(x, 1.0/ts.SAMPLE_RATE) if _num(x) else None
    return _round(x, 0 if unit == 'm2' else (2 if unit in ('s',) else (3 if unit == 'ratio' else 1)))


def poisson_tail(count, lam):
    """P(N >= count) for N ~ Poisson(lam): how often a team at a constant rate shows at least this many (kept for
    the record; the improvement test is the empirical window distribution's)."""
    if count <= 0:
        return 1.0
    if lam <= 0:
        return 0.0
    term = math.exp(-lam)
    below = term
    for i in range(1, int(count)):
        term *= lam/i
        below += term
    return max(0.0, 1.0-below)


def binomial_tail(count, n, p):
    """P(X >= count) for X ~ Binomial(n, p)."""
    if count <= 0:
        return 1.0
    if count > n:
        return 0.0
    return float(sum(math.comb(n, k)*p**k*(1-p)**(n-k) for k in range(int(count), int(n)+1)))


def resolve_reference(reference=None):
    """The reference dict to compare with: the given dict, or reference.json (soccer/coaching/reference.py), or None
    when there is none ("no professional reference available"). reference=False means none on purpose."""
    if reference is False:
        return None
    if isinstance(reference, dict):
        return reference
    try:
        return refs.load_reference(reference) if reference is not None else refs.load_reference()
    except (OSError, ValueError):
        return None


def rule_of(reference, name='default'):
    """The episode rule (reference.json 'rules'), DEFAULT_RULE when the reference names none."""
    if reference:
        for n in (name, 'default'):
            try:
                return dict(DEFAULT_RULE, **refs.rule(reference, n))
            except KeyError:
                continue
    return dict(DEFAULT_RULE)


def thresholds(reference, metric, context, rule):
    """{'enter', 'exit', 'p90', 'p75', 'p50', 'p10', 'context' (the one used), 'samples'} or None when the reference
    lacks them."""
    if not reference:
        return None
    enter, used = refs.quantile(reference, metric, rule['enter'], context)
    exit_, used2 = refs.quantile(reference, metric, rule['exit'], used or context)
    if enter is None or exit_ is None or used2 != used:
        return None
    levels = {q: refs.quantile(reference, metric, q, used)[0] for q in ('p90', 'p75', 'p50', 'p10')}
    n = (((reference.get('metrics') or {}).get(metric) or {}).get('contexts') or {}).get(used, {}).get('n')
    return dict(levels, enter=float(enter), exit=float(min(exit_, enter)), context=used, samples=n)


def strength_seconds(range_seconds):
    """The measured time a strength needs over an analysed range: max(30 s, min(60 s, 0.8 x the range))."""
    return max(STRENGTH_MIN_SECONDS, min(STRENGTH_FULL_SECONDS, STRENGTH_RANGE_SHARE*float(range_seconds or 0.0)))


def run_facts(cd):
    """Run-level facts the confidence and the evidence tiers use: small scene, keyless registration (no landmark
    anchor: the automatic pitch is provisional), a measured lens bend (wide-angle footage: no single homography fits),
    the stabilized block, the mean observed outfield players per team and the frame size."""
    result = cd.result or {}
    summary = result.get('summary') or {}
    reg = ((summary.get('pitch') or {}).get('registration')) or {}
    anchors = reg.get('anchors')
    has_cal = bool(cd.available.get('calibration'))
    keyless = bool(has_cal and isinstance(anchors, list) and not any(
        isinstance(a, dict) and a.get('source') == 'landmarks' for a in anchors))
    observed = {team: float(np.mean(s.observed)) if len(s.observed) else 0.0 for team, s in cd.teams.items()}
    video = result.get('video') or {}
    return {'smallScene': bool(cd.small_scene), 'keyless': keyless, 'lens': reg.get('lens') is not None,
            'stabilized': bool(cd.available.get('stabilized')), 'observed': observed,
            'fewPlayers': bool(observed) and max(observed.values()) < FEW_PLAYERS,
            'width': video.get('width'), 'height': video.get('height')}


# ---------- the analysed range of one team ----------
class Analysis:
    """One team's series restricted to the analysed range [t0, t1] (absolute seconds, clamped to the segment), with
    the reference, the run facts and the identity taint the detectors share."""

    def __init__(self, cd, team, reference=None, t0=None, t1=None, library=None):
        if team not in cd.teams:
            raise ValueError('team must be A or B')
        self.cd, self.team = cd, team
        self.s = cd.teams[team]
        self.other = cd.teams['B' if team == 'A' else 'A']
        self.reference = reference
        frames = (cd.result or {}).get('frames') or []
        first = float(frames[0]['time']) if frames else float(cd.start)
        last = float(frames[-1]['time']) if frames else float(cd.end)
        self.segment_start = first
        lo = first if t0 is None else min(max(float(t0), first), last)
        hi = last if t1 is None else max(min(float(t1), last), first)
        if hi < lo:
            lo, hi = hi, lo
        self.t0, self.t1 = lo, hi
        self.rows = np.flatnonzero((cd.t >= lo-1e-6) & (cd.t <= hi+1e-6))
        self.t = cd.t[self.rows]
        self.step = cd.step
        self.n = len(self.rows)
        self.range_seconds = self.n*self.step
        self.run = run_facts(cd)
        self.library = library
        self._taint = None

    # Series accessors restricted to the range.
    def val(self, name):
        return self.s.values[name][self.rows]

    def ok(self, metric):
        return self.s.eligible[metric][self.rows]

    def why(self, metric):
        return self.s.reasons[metric][self.rows]

    def g(self, i):
        """The global sample index of range sample i."""
        return int(self.rows[i])

    def window(self):
        return {'t0': _t(self.t0), 't1': _t(self.t1), 'segmentT0': _t(self.t0-self.segment_start),
                'segmentT1': _t(self.t1-self.segment_start)}

    def measured(self, mask):
        return ep.measured_seconds(self.t, mask, self.step)

    def longest(self, mask):
        return ep.longest_stretch(self.t, mask, self.step)

    def registered(self):
        """Range samples whose frame's final registration supports positions (the series' registration gates:
        calibrated, at least MEDIUM, chains within 3 m); every sample when the result has no registration."""
        cal = self.cd.calibration or {}
        S = len(self.cd.t)
        ok = np.ones(S, bool)
        if 'calibrated' in cal:
            ok &= np.asarray(cal['calibrated'], bool)
        conf = np.asarray(cal.get('confidence'), float) if cal.get('confidence') is not None else None
        present = np.asarray(cal.get('present'), bool) if cal.get('present') is not None else None
        if conf is not None and present is not None:
            with np.errstate(invalid='ignore'):
                ok &= ~present | (conf >= cs.MIN_CALIBRATION)
        dis = np.asarray(cal.get('disagreement'), float) if cal.get('disagreement') is not None else None
        if dis is not None:
            with np.errstate(invalid='ignore'):
                ok &= ~(np.isfinite(dis) & (dis > cs.MAX_DISAGREEMENT))
        return ok[self.rows]

    def taint(self):
        """(range samples x players) bool: the player is within IDENTITY_TAINT_SECONDS of an identity issue."""
        if self._taint is None:
            result = self.cd.result or {}
            times = {}
            for i in result.get('issues') or ():
                if isinstance(i, dict) and isinstance(i.get('id'), str) and _num(i.get('time')) and \
                        any(k in str(i.get('reason') or '').lower() for k in TAINT_ISSUES):
                    times.setdefault(i['id'], []).append(float(i['time']))
            for e in result.get('events') or ():
                if isinstance(e, dict) and e.get('kind') in TAINT_EVENTS and isinstance(e.get('playerId'), str) \
                        and _num(e.get('time')):
                    times.setdefault(e['playerId'], []).append(float(e['time']))
            T = np.zeros((self.n, len(self.s.ids)), bool)
            for pid, ts_ in times.items():
                j = self.s.col.get(pid)
                if j is None:
                    continue
                arr = np.sort(np.array(ts_))
                k = np.searchsorted(arr, self.t-IDENTITY_TAINT_SECONDS, side='left')
                hit = k < len(arr)
                T[hit, j] = arr[k[hit]] <= self.t[hit]+IDENTITY_TAINT_SECONDS
            self._taint = T
        return self._taint

    def obs_id(self, detector, *parts):
        return ep.stable_id('obs', detector, self.team, *parts, self.t0, self.t1)

    def occurrence(self, detector, key, i0, i1, peak, metrics, players, censored_start=False, censored_end=False,
                   seconds=None):
        """An occurrence record over range samples i0..i1 with its peak sample."""
        g0, g1, gk = self.g(i0), self.g(i1), self.g(peak)
        t0, t1, tp = float(self.t[i0]), float(self.t[i1]), float(self.t[peak])
        frame = self.cd.frame
        return {'id': ep.stable_id('occ', detector, self.team, key, t0, t1), 't0': _t(t0), 't1': _t(t1),
                'peakT': _t(tp), 'segmentT0': _t(t0-self.segment_start), 'segmentT1': _t(t1-self.segment_start),
                'segmentPeakT': _t(tp-self.segment_start), 'frame0': int(frame[g0]), 'frame1': int(frame[g1]),
                'peakFrame': int(frame[gk]), 'seconds': _sec(seconds if seconds is not None else t1-t0+self.step),
                'censored': bool(censored_start or censored_end), 'censoredStart': bool(censored_start),
                'censoredEnd': bool(censored_end), 'players': list(players), 'metrics': metrics}


# ---------- confidence ----------
_RANK = {'high': 0, 'medium': 1, 'low': 2}


def _worst(*levels):
    levels = [x for x in levels if x in _RANK]
    return max(levels, key=_RANK.get) if levels else 'low'


def _cap(level, cap):
    return _worst(level, cap)


def confidence(a, mask, cover=None, players=None, player=None, structure=False, ball=None):
    """The confidence record {tracking, calibration, identity, ball, overall, reasons} of a measurement over the range
    samples in `mask`. cover: the metric's measured samples over the whole range (all contexts; default mask);
    players: the ids involved (team-level: per-sample share of them tainted), player: one player's measurement;
    structure: the units come from the formation reading; ball: the ball aspect ('n/a' when None)."""
    reasons = []
    rows = a.rows[mask]
    # Calibration (the final registration of the measured samples).
    conf = a.cd.calibration.get('confidence') if a.cd.calibration else None
    cal = 'high'
    if conf is not None and len(rows):
        c = np.asarray(conf, float)[rows]
        c = c[np.isfinite(c)]
        if len(c) and float(np.mean(c >= HIGH_CALIBRATION)) < CAL_HIGH_SHARE:
            cal = 'medium'
            reasons.append('medium_registration')
    if a.run['keyless']:
        cal = _cap(cal, 'medium')
        reasons.append('keyless')
    # Tracking (how much and how continuously the metric was measured).
    cover = mask if cover is None else cover
    share = float(np.count_nonzero(cover))/a.n if a.n else 0.0
    longest = a.longest(cover)
    if share >= TRACK_HIGH_SHARE and longest >= TRACK_HIGH_STRETCH:
        track = 'high'
    elif longest >= TRACK_MEDIUM_STRETCH:
        track = 'medium'
        reasons.append('short_stretch' if longest < TRACK_HIGH_STRETCH else 'short_measured')
    else:
        track = 'low'
        reasons.append('short_stretch')
    if a.run['smallScene']:
        track = _cap(track, 'medium')
        reasons.append('small_scene')
    cols = [a.s.col[p] for p in (players or ([player] if player else [])) if p in a.s.col]
    if a.run['stabilized'] and cols and len(rows):
        flagged = a.s.values['flagged'][np.ix_(rows, cols)].any(axis=1)
        if float(np.mean(flagged)) > FLAG_MEDIUM_SHARE:
            track = _cap(track, 'medium')
            reasons.append('stabilizer_flags')
    if structure and len(rows):
        amb = float(np.mean(a.s.values['formationConfidence'][rows] == 'ambiguous'))
        if amb >= AMBIGUOUS_LOW_SHARE:
            track = 'low'
            reasons.append('ambiguous_formation')
    # Identity (crossings, re-identifications, uncertain swaps of the involved players).
    identity = 'high'
    if cols and np.count_nonzero(mask):
        T = a.taint()[np.flatnonzero(mask)][:, cols]
        tainted = float(T.mean()) if player is None else float(T.any(axis=1).mean())
        hi_, med_ = (IDENTITY_HIGH, IDENTITY_MEDIUM) if player is None else (PLAYER_IDENTITY_HIGH, PLAYER_IDENTITY_MEDIUM)
        identity = 'high' if tainted < hi_ else ('medium' if tainted < med_ else 'low')
        if identity != 'high':
            reasons.append('crossings')
    out = {'tracking': track, 'calibration': cal, 'identity': identity, 'ball': ball or 'n/a'}
    out['overall'] = _worst(track, cal, identity, *([ball] if ball else []))
    out['reasons'] = list(dict.fromkeys(reasons))
    return out


def not_measured(a, metric, player=None):
    """Reason codes for the range samples where the metric was not measured, largest first, those covering at least
    REASON_MIN_SHARE of the range (series reason codes; language.REASONS words the known ones)."""
    r = a.why(metric)
    if r.ndim == 2:
        j = a.s.col.get(player) if player is not None else None
        r = r[:, j] if j is not None else r.ravel()
    codes = Counter(x for x in r.tolist() if x)
    total = max(1, a.n*(1 if r.ndim == 1 or player is not None else len(a.s.ids)))
    out = [c for c, k in codes.most_common() if k/total >= REASON_MIN_SHARE]
    if a.run['fewPlayers'] and 'few_players' not in out:
        out.append('few_players')
    return out


# ---------- classification ----------
def strength_check(reference, metric, values, measured, range_seconds, context, low=None):
    """The strength comparison of a measurement in a context (module docstring): {'context', 'measuredSeconds',
    'median', 'p50', 'p75', 'p90', 'favourable', 'shareAboveP75', 'nEff', 'tauSeconds', 'upperBound',
    'coverageSeconds', 'enough', 'passes', 'failed' [reasons]}. values: the measured values; low: the level below which
    the median is no longer favourable (the lateral lag's p10: a back line far ahead of its midfield is not "moving
    together"), None for a one-sided metric."""
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    need = strength_seconds(range_seconds)
    out = {'context': context, 'measuredSeconds': _sec(measured), 'coverageSeconds': _sec(need),
           'enough': bool(measured >= need-1e-9)}
    p50 = refs.quantile(reference, metric, 'p50', context)[0] if reference else None
    p75 = refs.quantile(reference, metric, STRENGTH_LEVEL, context)[0] if reference else None
    p90 = refs.quantile(reference, metric, 'p90', context)[0] if reference else None
    if not len(v) or p50 is None or p75 is None:
        return dict(out, passes=False, failed=['no_reference' if p50 is None else 'not_measured'])
    median = float(np.median(v))
    favourable = median <= p50+1e-9 and (low is None or median >= low-1e-9)
    share = float(np.mean(v > p75))
    tau = (refs.tau_seconds(reference, metric) if reference else None) or TAU_FALLBACK_SECONDS
    n_eff = measured/tau if tau > 0 else 0.0
    bound = refs.share_upper_bound(share, n_eff, STRENGTH_BOUND_CONFIDENCE)
    failed = []
    if not out['enough']:
        failed.append('short_for_strength')
    if not favourable:
        failed.append('unfavourable_side')
    if bound >= STRENGTH_PRO_SHARE:
        failed.append('evidence_bound')
    return dict(out, median=_val(metric, median), p50=_val(metric, p50), p75=_val(metric, p75),
                p90=_val(metric, p90), favourable=bool(favourable), shareAboveP75=_share(share),
                nEff=_round(n_eff, 1), tauSeconds=_round(tau, 1), upperBound=_share(bound),
                passes=not failed, failed=failed)


def classify(measured, longest, count, rule, window, strength, overall, has_reference=True):
    """(classification, severity, reasons, stats) of a count-based observation (module docstring). count: the
    occurrences counted; window: the professional window levels at the measured time (reference.window_baseline) or
    None; strength: strength_check's record or None; overall: the observation's overall confidence."""
    minutes = measured/60.0
    stats = {'rate': (count/minutes) if minutes > 0 else None, 'windowP99': None, 'windowP50': None, 'windowMax': None,
             'windowTail': None, 'improvementBlocked': None}
    if not has_reference:
        return 'insufficient', None, [NO_REFERENCE], stats
    if measured < MIN_MEASURED_SECONDS-1e-9:
        return 'insufficient', None, ['short_measured'], stats
    if longest < float(rule.get('minSeconds', DEFAULT_RULE['minSeconds']))-1e-9:
        return 'insufficient', None, ['short_stretch'], stats
    if overall == 'low':
        return 'insufficient', None, ['low_confidence'], stats
    usable = bool(window and window.get('available'))
    if usable:
        stats.update(windowP99=window.get(IMPROVEMENT_LEVEL), windowP50=window.get('p50'), windowMax=window.get('max'))
        p99 = window.get(IMPROVEMENT_LEVEL)
        if count >= MIN_OCCURRENCES and _num(p99) and count > float(p99)+1e-9:
            peak = window.get('max')
            concern = (overall == 'high' and measured >= SHORT_SEGMENT_SECONDS-1e-9
                       and count >= CONCERN_MARGIN*float(p99)-1e-9 and _num(peak) and count > float(peak)+1e-9)
            return 'improvement', 'high_confidence_concern' if concern else 'potential_improvement', [], stats
    else:
        stats['improvementBlocked'] = (window or {}).get('reason') or 'no_window_baseline'
    p50 = window.get('p50') if usable else None
    no_excess = count == 0 or (_num(p50) and count <= float(p50)+1e-9)
    if strength and strength.get('passes') and overall in ('high', 'medium') and no_excess:
        return 'strength', 'observed_strength', [], stats
    return 'neutral', None, [], stats


def neutral_state(occurrences, strength=None, under=None):
    """(state, spell) of a neutral observation: 'one_spell' with that spell's times and peak, 'within_range' for
    several, else 'under_measured' (too little measurement for a strength) or 'absent'."""
    occ = sorted(occurrences or [], key=lambda o: o['t0'])
    if len(occ) == 1:
        o = occ[0]
        peak = next(iter((o.get('metrics') or {}).values()), {}) if o.get('metrics') else {}
        return 'one_spell', {'t0': o['t0'], 't1': o['t1'], 'segmentT0': o.get('segmentT0'),
                             'segmentT1': o.get('segmentT1'), 'peakT': o.get('peakT'), 'seconds': o.get('seconds'),
                             'peak': (peak or {}).get('peak') if isinstance(peak, dict) else None,
                             'censored': bool(o.get('censored'))}
    if len(occ) >= 2:
        return 'within_range', None
    if under is None:
        under = not (strength or {}).get('enough', False)
    return ('under_measured' if under else 'absent'), None


# ---------- records ----------
def _drills(a, key, pattern, classification, detector=None):
    if classification not in ('improvement', 'review') or detector == 'isolation':
        return []               # an isolation review is a role check, not a training need
    lib = a.library
    if lib is None:
        try:
            from . import drills as drill_lib
            lib = a.library = drill_lib.load_drills()
        except (ImportError, OSError, ValueError):
            a.library = {'drills': []}
            return []
    try:
        from . import drills as drill_lib
        return [d['id'] for d in drill_lib.drills_for(lib, key, pattern)]
    except (ImportError, KeyError, TypeError):
        return []


def record(a, detector, pattern, key_parts, classification, severity, measured, longest, occurrences, baseline,
           metrics, players, conf, metric=None, reasons=(), evidence_hint=None, rate=None, rated=True, state=None):
    """An Observation record (contract section 5). Private '_ev' carries the evidence hints for evidence.py."""
    key, category = DETECTOR_KEYS[detector]
    occurrences = sorted(occurrences, key=lambda o: o['t0'])
    count = len(occurrences)
    if not rated:
        rate = None
    elif rate is None:
        rate = ep.rate_per_minute(count, measured) if measured > 0 else None
    obs = {'id': a.obs_id(detector, *key_parts), 'engineVersion': ENGINE_VERSION, 'team': a.team,
           'detector': detector, 'category': category, 'pattern': pattern, 'metric': metric,
           'classification': classification, 'severity': severity, 'window': a.window(),
           'measuredSeconds': _sec(measured), 'eligibleShare': _share(measured/a.range_seconds if a.range_seconds else 0.0),
           'longestStretchSeconds': _sec(longest), 'occurrences': occurrences, 'occurrenceCount': count,
           'ratePerMeasuredMinute': _round(rate, 3), 'baseline': baseline, 'metrics': metrics,
           'players': list(players), 'confidence': conf, 'evidence': [], 'recommendationKey': key,
           'drillIds': _drills(a, key, pattern, classification, detector), 'status': 'reported',
           'suppressionReason': None,
           'insufficientReasons': list(dict.fromkeys(reasons)) if classification == 'insufficient' else [],
           'segmentKind': 'short' if measured < SHORT_SEGMENT_SECONDS else 'long'}
    if classification == 'neutral':
        obs['state'] = state or 'absent'
    if evidence_hint is not None:
        obs['_ev'] = evidence_hint
    return obs


def public(obs):
    """The observation without its private evidence hints."""
    return {k: v for k, v in obs.items() if not k.startswith('_')}


def _window_summary(window):
    if not window:
        return None
    keys = ('available', 'reason', 'minutes', 'method', 'from', 'p50', 'p90', 'p99', 'max', 'n', 'nByTeam',
            'bothTeams', 'perPlayer')
    return {k: window.get(k) for k in keys if k in window}


def _baseline(a, metric, rule_name, rule, thr, pro, window=None):
    if thr is None:
        return {'reference': refs.reference_id(a.reference) if a.reference else None, 'rule': rule_name,
                'context': None, 'threshold': None, 'professionalRatePerMinute': None,
                'professionalShareAbove': None, 'professionalWindow': _window_summary(window), 'available': False}
    rates = (pro or {}).get('perMinute') or {}
    dur = (pro or {}).get('durationSeconds') or {}
    return {'reference': refs.reference_id(a.reference), 'rule': rule_name, 'context': thr['context'],
            'threshold': {'enter': _val(metric, thr['enter']), 'exit': _val(metric, thr['exit']),
                          'p90': _val(metric, thr['p90']), 'p75': _val(metric, thr.get('p75')),
                          'p50': _val(metric, thr['p50']), 'enterQuantile': rule['enter'],
                          'exitQuantile': rule['exit']},
            'minSeconds': float(rule['minSeconds']), 'mergeSeconds': float(rule['mergeSeconds']),
            'professionalRatePerMinute': {k: _round(rates.get(k), 3) for k in ('p10', 'p50', 'p90')} if pro else None,
            'professionalDurationSeconds': {k: _sec(dur.get(k)) for k in ('p50', 'p90')} if dur else None,
            'professionalShareAbove': PRO_SHARE_ABOVE_P90, 'referenceLevel': 'p90',
            'professionalWindow': _window_summary(window), 'referenceSamples': thr.get('samples'), 'available': True}


def _summary(metric, v, mask, thr):
    vals = v[mask]
    vals = vals[np.isfinite(vals)]
    out = {'median': _val(metric, np.median(vals)) if len(vals) else None,
           'p90': _val(metric, np.percentile(vals, 90)) if len(vals) else None,
           'mean': _val(metric, np.mean(vals)) if len(vals) else None}
    if thr is not None and _num(thr.get('p90')) and len(vals):
        out['shareAboveProfessionalP90'] = _share(np.mean(vals > float(thr['p90'])))
    return out


def _strength_metrics(strength):
    """The strength comparison's numbers as flat observation metrics."""
    if not strength:
        return {}
    return {'professionalP50': strength.get('p50'), 'professionalP75': strength.get('p75'),
            'favourableSide': strength.get('favourable'), 'shareAboveProfessionalP75': strength.get('shareAboveP75'),
            'upperBoundShareAboveP75': strength.get('upperBound'), 'nEff': strength.get('nEff'),
            'tauSeconds': strength.get('tauSeconds'), 'strengthSeconds': strength.get('coverageSeconds'),
            'strengthFailed': list(strength.get('failed') or [])}


def _modal_players(lists, limit=None):
    """Players ordered by how often they appear in the lists (ties by id)."""
    c = Counter(p for lst in lists for p in lst if isinstance(p, str) and p)
    ids = sorted(c, key=lambda p: (-c[p], p))
    return ids[:limit] if limit else ids


# ---------- the generic metric detector ----------
def _context_groups(groups, a, metric, rule):
    """Merge candidate (context, mask) groups by the context the reference actually has (a missing one falls back to
    'all'). Returns (kept: [(context, mask, thresholds)] with at least MIN_MEASURED_SECONDS each, in third order;
    union: every group's samples; split: whether the reference distinguishes a context other than 'all' here)."""
    merged = {}
    union = np.zeros(a.n, bool)
    for name, m in groups:
        union |= m
        thr = thresholds(a.reference, metric, name, rule)
        if thr is None:
            continue
        used = thr['context']
        if used in merged:
            merged[used] = (merged[used][0] | m, merged[used][1])
        else:
            merged[used] = (m.copy(), thr)
    kept = [(c, m, thr) for c, (m, thr) in merged.items() if a.measured(m) >= MIN_MEASURED_SECONDS-1e-9]
    order = {('third:'+x): i for i, x in enumerate(THIRDS)}
    kept.sort(key=lambda k: (order.get(k[0], 9), k[0]))
    return kept, union, any(c != 'all' for c in merged)


def third_groups(a, mask):
    """[(context, mask)] of a metric's measured samples by the team's third (the voted centre's distance from the own
    goal line); samples without a known third go to 'all'."""
    third, ok = a.val('third'), a.ok('third')
    out = [('third:'+x, mask & ok & (third == x)) for x in THIRDS]
    rest = mask & ~ok
    if rest.any():
        out.append(('all', rest))
    return out


def _analyse(a, detector, metric, value, m, thr, rule, rule_name, key, players_at, extra_metrics):
    """Episodes of one metric over the samples m beyond the thresholds thr, with their occurrence records and
    evidence hints, the professional window levels at its measured time, the strength comparison and the summary
    metrics."""
    measured, longest = a.measured(m), a.longest(m)
    eps = ep.episodes(a.t, value, m, thr['enter'], thr['exit'], rule['minSeconds'], rule['mergeSeconds'], 'above',
                      step=a.step)
    occ, hints = [], []
    for e in eps:
        idx = np.arange(e['i0'], e['i1']+1)
        idx = idx[m[idx]]
        players = players_at(e['peakIndex'])
        met = {metric: {'peak': _val(metric, e['peak']), 'mean': _val(metric, e['mean'])}}
        if extra_metrics:
            met.update(extra_metrics(idx, e['peakIndex']))
        o = a.occurrence(detector, '|'.join(map(str, key)), e['i0'], e['i1'], e['peakIndex'], met, players,
                         e['censoredStart'], e['censoredEnd'], e['seconds'])
        o['context'] = thr['context']
        occ.append(o)
        hints.append({'occurrence': o['id'], 'peak': a.g(e['peakIndex']), 'candidates': [a.g(i) for i in idx],
                      'values': [float(value[i]) for i in idx], 'players': players, 'strength': float(e['peak'])})
    pro = refs.episode_baseline(a.reference, metric, rule_name, thr['context'])
    if pro and pro.get('context') != thr['context']:
        pro = None              # a rate of another context is not this one's baseline (contract: same rule and context)
    window = refs.window_baseline(a.reference, metric, thr['context'], measured/60.0, rule_name)
    strength = strength_check(a.reference, metric, value[m], measured, a.range_seconds, thr['context'])
    metrics = _summary(metric, value, m, thr)
    if extra_metrics:
        metrics.update({k: v for k, v in (extra_metrics(np.flatnonzero(m), None) or {}).items()
                        if k not in metrics and not isinstance(v, dict)})
    return {'context': thr['context'], 'mask': m, 'thr': thr, 'measured': measured, 'longest': longest, 'occ': occ,
            'hints': hints, 'pro': pro, 'window': window, 'strength': strength, 'metrics': metrics, 'key': key}


def pooled_strength(a, metric, value, results, measured):
    """The strength comparison of a pooled observation over several contexts: every sample is compared with its own
    context's levels (beyond that context's p75; the median of every context on the favourable side of its own p50),
    the upper bound over the pooled measured time, and every context with enough measurement for a strength of its
    own must pass on its own (module docstring). Without any context measured MIN_MEASURED_SECONDS the 'all' levels
    are the only comparison there is (strength_check over every sample).
    """
    if not results:
        v = np.asarray(value, float)
        return strength_check(a.reference, metric, v[np.isfinite(v)], measured, a.range_seconds, 'all')
    beyond, failed = [], []
    for r in results:
        p75 = refs.quantile(a.reference, metric, STRENGTH_LEVEL, r['context'])[0]
        v = np.asarray(value, float)[r['mask']]
        v = v[np.isfinite(v)]
        if p75 is not None:
            beyond.append(v > p75)
    b = np.concatenate(beyond) if beyond else np.zeros(0, bool)
    need = strength_seconds(a.range_seconds)
    tau = refs.tau_seconds(a.reference, metric) or TAU_FALLBACK_SECONDS
    share = float(b.mean()) if len(b) else 1.0
    n_eff = measured/tau
    bound = refs.share_upper_bound(share, n_eff, STRENGTH_BOUND_CONFIDENCE)
    favourable = all(r['strength'].get('favourable') for r in results)
    own = [r for r in results if r['strength'].get('enough')]
    if measured < need-1e-9:
        failed.append('short_for_strength')
    if not favourable:
        failed.append('unfavourable_side')
    if bound >= STRENGTH_PRO_SHARE:
        failed.append('evidence_bound')
    if not all(r['strength'].get('passes') for r in own):
        failed.append('context_fails')
    return {'context': 'contexts', 'measuredSeconds': _sec(measured), 'coverageSeconds': _sec(need),
            'enough': bool(measured >= need-1e-9), 'favourable': bool(favourable), 'shareAboveP75': _share(share),
            'nEff': _round(n_eff, 1), 'tauSeconds': _round(tau, 1), 'upperBound': _share(bound),
            'passes': not failed, 'failed': failed, 'p50': None, 'p75': None}


def metric_detector(a, detector, pattern, metric, value, mask, groups, players_at, typical_players, preset,
                    extra_metrics=None, structure=False, units=None, key_suffix=(), player=None, rule_name='default',
                    cover=None):
    """Observations of one metric (module docstring, "Contexts"): episodes beyond each context's enter level and
    their count against that context's professional window distribution. A context with an improvement is an
    observation of its own; otherwise one observation pools every measured sample under the 'all' levels, never an
    improvement (the levels it would compare with are not the contexts'), and a strength only when every context with
    enough measurement for one passes the strength rule in its own context (the strictest of those comparisons is the
    one quoted). players_at(i) -> ids involved at range sample i; typical_players(mask) -> ids of a measurement
    without occurrences."""
    rule = rule_of(a.reference, rule_name)
    value = np.asarray(value, float)
    mask = mask & np.isfinite(value)
    cover = mask if cover is None else cover
    kept, union, split = _context_groups(groups, a, metric, rule) if a.reference else ([], mask, False)
    thr_all = thresholds(a.reference, metric, 'all', rule) if a.reference else None
    suffix = tuple(key_suffix)

    def insufficient(reasons, m=mask):
        conf = confidence(a, m, cover, players=typical_players(m), player=player, structure=structure)
        return record(a, detector, pattern, (pattern, 'all')+suffix, 'insufficient', None, a.measured(m),
                      a.longest(m), [], _baseline(a, metric, rule_name, rule, None, None),
                      _summary(metric, value, m, None), typical_players(m), conf, metric,
                      reasons+not_measured(a, metric, player))

    if thr_all is None and not kept:
        return [insufficient([NO_REFERENCE] if not a.reference or not split else ['short_measured'])]

    def emit(r, pooled=None):
        involved = _modal_players([o['players'] for o in r['occ']]) if r['occ'] else typical_players(r['mask'])
        conf = confidence(a, r['mask'], cover, players=involved, player=player, structure=structure)
        cls, sev, why, stats = classify(r['measured'], r['longest'], len(r['occ']), rule, r['window'], r['strength'],
                                        conf['overall'])
        base = _baseline(a, metric, rule_name, rule, r['thr'], r['pro'], r['window'])
        metrics = dict(r['metrics'], **_strength_metrics(r['strength']),
                       professionalWindowP99=_round(stats['windowP99'], 2),
                       professionalWindowP50=_round(stats['windowP50'], 2),
                       professionalWindowMax=_round(stats['windowMax'], 2),
                       improvementBlocked=stats['improvementBlocked'])
        if pooled is not None:
            cls, sev, why, base, metrics = pooled(cls, sev, why, base, metrics)
        typical = None
        if not r['occ'] and cls != 'insufficient':
            idx = np.flatnonzero(r['mask'])
            typical = {'candidates': [a.g(i) for i in idx], 'values': [float(value[i]) for i in idx],
                       'target': metrics.get('pooledMedian', metrics.get('median'))}
        hint = {'preset': preset, 'metric': metric, 'units': units, 'player': player, 'occurrences': r['hints'],
                'typical': typical, 'enter': r['thr']['enter']}
        reasons = why+(not_measured(a, metric, player) if cls == 'insufficient' else [])
        state, spell = (None, None)
        if cls == 'neutral':
            state, spell = neutral_state(r['occ'], r['strength'])
            if spell:
                metrics['spell'] = spell
        return cls, record(a, detector, pattern, r['key'], cls, sev, r['measured'], r['longest'], r['occ'], base,
                           metrics, involved, conf, metric, reasons, hint if cls != 'insufficient' else None,
                           stats['rate'], state=state)

    results = [_analyse(a, detector, metric, value, m, thr, rule, rule_name, (pattern, c)+suffix, players_at,
                        extra_metrics) for c, m, thr in kept]
    if not split:
        # The reference has no contexts here: the 'all' analysis is the observation.
        if not results:
            return [insufficient(['short_measured'])]
        return [emit(r)[1] for r in results]
    found = [obs for cls, obs in (emit(r) for r in results) if cls == 'improvement']
    if found:
        return found
    if len(results) == 1 and results[0]['measured'] >= DOMINANT_CONTEXT_SHARE*a.measured(union)-1e-9:
        return [emit(results[0])[1]]        # one context holds (nearly) all of it: its own levels, its own words
    if thr_all is None:
        thr_all = kept[0][2]
    pooled = _analyse(a, detector, metric, value, union, dict(thr_all), rule, rule_name, (pattern, 'all')+suffix,
                      players_at, extra_metrics)
    # The pooled observation keeps the contexts' own occurrences (their levels are the context's).
    pooled['occ'] = sorted((o for r in results for o in r['occ']), key=lambda o: o['t0'])
    pooled['hints'] = [h for r in results for h in r['hints']]
    pooled['strength'] = pooled_strength(a, metric, value, results, pooled['measured'])
    contexts = {}
    for r in results:
        s = r['strength']
        contexts[r['context']] = {'measuredSeconds': _sec(r['measured']), 'median': r['metrics'].get('median'),
                                  'shareAboveProfessionalP90': r['metrics'].get('shareAboveProfessionalP90'),
                                  'occurrences': len(r['occ']), 'enter': _val(metric, r['thr']['enter']),
                                  'p50': s.get('p50'), 'p75': s.get('p75'), 'p90': _val(metric, r['thr']['p90']),
                                  'shareAboveProfessionalP75': s.get('shareAboveP75'),
                                  'upperBoundShareAboveP75': s.get('upperBound'), 'favourableSide': s.get('favourable'),
                                  'enoughForStrength': s.get('enough'), 'passesStrength': s.get('passes'),
                                  'professionalWindowP99': (r['window'] or {}).get('p99')}
    pooled['metrics']['contexts'] = contexts
    # The strictest context comparison (the most time beyond its own p75, then the median nearest its own p50) is
    # the one a pooled observation quotes, never the loosest 'all' levels (broadcast 42-222 s team B's width: 0%
    # beyond the all-context p90 of 57.7 m, but 9% beyond its own third's 45.3 m).
    strict = max(results, key=lambda r: (r['strength'].get('shareAboveP75') or 0.0,
                                         (r['strength'].get('median') or 0.0)-(r['strength'].get('p50') or 0.0)),
                 default=None)

    def quote(cls, sev, why, base, metrics):
        if cls == 'improvement':
            cls, sev, why = 'neutral', None, []
        if strict is None:
            return cls, sev, why, base, metrics
        sm = strict['strength']
        out = dict(metrics, pooledMedian=metrics.get('median'),
                   pooledShareAboveProfessionalP90=metrics.get('shareAboveProfessionalP90'),
                   median=strict['metrics'].get('median'),
                   shareAboveProfessionalP90=strict['metrics'].get('shareAboveProfessionalP90'),
                   strictestContext=strict['context'], comparisonSeconds=_sec(strict['measured']),
                   professionalP50=sm.get('p50'), professionalP75=sm.get('p75'),
                   shareAboveProfessionalP75=sm.get('shareAboveP75'),
                   contextShareAboveP75=pooled['strength'].get('shareAboveP75'))
        return cls, sev, why, _baseline(a, metric, rule_name, rule, strict['thr'], strict['pro'], strict['window']), out
    return [emit(pooled, pooled=quote)[1]]


# ---------- the detectors ----------
def _unit_players(a, names):
    units = a.s.units

    def at(i):
        u = units[a.g(i)]
        if not u:
            return []
        return [p for n in names for p in u.get(n) or ()]

    def typical(mask):
        lists = [at(i) for i in np.flatnonzero(mask)]
        if not lists:
            return []
        common = Counter(tuple(x) for x in lists).most_common(1)[0][0]
        return list(common)
    return at, typical


def detect_compactness(a):
    """Width, depth and convex-hull area of the outfield team by third (patterns team_width, team_depth, team_area)."""
    out = []
    frm = {k: a.val(k+'From') for k in ('width', 'depth')}
    to = {k: a.val(k+'To') for k in ('width', 'depth')}
    for metric, pattern, axes in (('width', 'team_width', ('width',)), ('depth', 'team_depth', ('depth',)),
                                  ('hull', 'team_area', ('width', 'depth'))):
        value = a.val(metric)
        mask = a.ok(metric) & np.isfinite(value)

        def at(i, axes=axes):
            ids = []
            for ax in axes:
                for p in (frm[ax][i], to[ax][i]):
                    if isinstance(p, str) and p and p not in ids:
                        ids.append(p)
            return ids
        out += metric_detector(a, 'compactness', pattern, metric, value, mask, third_groups(a, mask), at,
                               lambda m: [], 'width_depth', units=None)
    return out


def detect_unit_gap_dm(a):
    """Back line to the first midfield line (template line index 1), centre to centre, by third."""
    value = a.val('gapDM')
    mask = a.ok('gapDM') & np.isfinite(value)
    at, typical = _unit_players(a, ('D', 'M1'))
    height = a.val('lineHeight')

    def extra(idx, peak):
        h = height[idx]
        h = h[np.isfinite(h)]
        return {'lineHeight': {'peak': _val('lineHeight', h.max()), 'mean': _val('lineHeight', h.mean())}} if len(h) \
            else {}
    return metric_detector(a, 'unit_gap_dm', 'space_between_defence_and_midfield', 'gapDM', value, mask,
                           third_groups(a, mask), at, typical, 'defensive_gap', structure=True, units=('D', 'M1'),
                           extra_metrics=lambda idx, peak: extra(idx, peak) if peak is not None else {})


def detect_unit_gap_mf(a):
    """Midfield (every middle line) to the forward line, centre to centre, by third."""
    value = a.val('gapMF')
    mask = a.ok('gapMF') & np.isfinite(value)
    at, typical = _unit_players(a, ('M', 'F'))
    return metric_detector(a, 'unit_gap_mf', 'space_between_midfield_and_forwards', 'gapMF', value, mask,
                           third_groups(a, mask), at, typical, 'defensive_gap', structure=True, units=('M', 'F'))


def detect_back_line(a):
    """Back-line evenness: the std of the defenders' depth, by back-line size, with the line-quality shares."""
    value = a.val('lineStd')
    mask = a.ok('lineStd') & np.isfinite(value)
    size = a.val('lineSize')
    quality = a.s.values['lineQuality'][a.rows]
    defenders = a.s.values['lineDefenders']
    height = a.val('lineHeight')
    groups = [('backline:%d' % n, mask & (size == n)) for n in BACKLINE_SIZES]
    other = mask & ~np.isin(size, BACKLINE_SIZES)
    if other.any():
        groups.append(('all', other))

    def at(i):
        return list(defenders[a.g(i)])

    def typical(m):
        lists = [tuple(defenders[a.g(i)]) for i in np.flatnonzero(m)]
        return list(Counter(lists).most_common(1)[0][0]) if lists else []

    def extra(idx, peak):
        if peak is not None:
            return {}
        q = quality[idx]
        sizes = size[idx]
        h = height[idx]
        out = {k+'Share': _share(np.mean(q == k)) if len(q) else None for k in ('coherent', 'uneven', 'stretched')}
        if len(sizes):
            out['lineSize'] = int(Counter(int(x) for x in sizes if np.isfinite(x)).most_common(1)[0][0])
        if len(h):
            out['lineHeight'] = _val('lineHeight', np.nanmedian(h))
        return out
    return metric_detector(a, 'back_line_coordination', 'uneven_back_line', 'lineStd', value, mask, groups, at,
                           typical, 'line_movement', extra_metrics=extra, structure=True, units=('D',))


def block_heights(a):
    """The team's and the opponent's median back-line height (m from their own goal line) over the range, with
    their measured seconds: the block height a report's context paragraph quotes. The opponent's is quoted only when
    it was measured at least MIN_MEASURED_SECONDS, the rule the team's own note has (amateur 118-182 s: team B's line
    was measured 18 s, so team B's report has no block height, and team A's report no longer quotes B's 67.9 m)."""
    out = {}
    for name, s in (('', a.s), ('opponent', a.other)):
        h = np.asarray(s.values['lineHeight'][a.rows], float)
        ok = s.eligible['lineHeight'][a.rows] & np.isfinite(h)
        key = 'lineHeight' if not name else 'opponentLineHeight'
        seconds = a.measured(ok)
        out[key] = _val('lineHeight', np.median(h[ok])) if ok.any() and \
            (not name or seconds >= MIN_MEASURED_SECONDS-1e-9) else None
        out[key+'Seconds'] = _sec(seconds)
    return out


def detect_line_height_gap(a):
    """A back line higher than the professional p90 line height while the D-M gap is beyond its context's enter level:
    review only (a high line is a valid choice; with a large gap in front of it the space between the lines is worth a
    look). Without that combination: one neutral context note with the team's and the opponent's median back-line
    height (block_heights), when the line was measured long enough."""
    rule = rule_of(a.reference, 'default')
    gap, height = a.val('gapDM'), a.val('lineHeight')
    mask = a.ok('gapDM') & a.ok('lineHeight') & np.isfinite(gap) & np.isfinite(height)
    at, typical = _unit_players(a, ('D', 'M1'))
    defenders = a.s.values['lineDefenders']
    blocks = block_heights(a)
    line_ok = a.ok('lineHeight') & np.isfinite(height)
    h_enter, used_h = refs.quantile(a.reference, 'lineHeight', LINE_HEIGHT_ENTER, 'all') if a.reference else (None, None)
    h_exit, _ = refs.quantile(a.reference, 'lineHeight', LINE_HEIGHT_EXIT, 'all') if a.reference else (None, None)

    def context_note(reasons=None):
        measured, longest = a.measured(line_ok), a.longest(line_ok)
        if measured < MIN_MEASURED_SECONDS:
            return []           # nothing measured long enough: the back-line detector reports the coverage
        conf = confidence(a, line_ok, players=typical(line_ok), structure=True)
        p50 = refs.quantile(a.reference, 'lineHeight', 'p50', 'all')[0] if a.reference else None
        metrics = dict(blocks, gapDM=_val('gapDM', np.median(gap[mask])) if mask.any() else None,
                       professionalLineHeightP50=_val('lineHeight', p50))
        cls = 'insufficient' if reasons else 'neutral'
        base = _baseline(a, 'gapDM', 'default', rule, None, None)
        if h_enter is not None:
            base['lineHeightThreshold'] = {'enter': _val('lineHeight', h_enter), 'exit': _val('lineHeight', h_exit),
                                           'context': used_h}
        return [record(a, 'line_height_and_gap', 'high_line_with_gap', ('high_line_with_gap', 'context'), cls, None,
                       measured, longest, [], base, metrics, typical(line_ok), conf, 'lineHeight', reasons or [],
                       None, rated=False, state='absent')]

    if h_enter is None or h_exit is None or thresholds(a.reference, 'gapDM', 'all', rule) is None:
        return context_note([NO_REFERENCE])
    h_exit = min(h_exit, h_enter)
    kept, _, _ = _context_groups(third_groups(a, mask), a, 'gapDM', rule)
    out = []
    for context, m, thr in kept:
        measured, longest = a.measured(m), a.longest(m)
        if measured < MIN_MEASURED_SECONDS:
            continue
        with np.errstate(invalid='ignore', divide='ignore'):
            zh = (height-h_exit)/max(h_enter-h_exit, 1e-6)
            zg = (gap-thr['exit'])/max(thr['enter']-thr['exit'], 1e-6)
        z = np.minimum(zh, zg)
        eps = ep.episodes(a.t, z, m, 1.0, 0.0, rule['minSeconds'], rule['mergeSeconds'], 'above', step=a.step)
        if not eps:
            continue
        key = ('high_line_with_gap', context)
        occ, hints = [], []
        for e in eps:
            idx = np.arange(e['i0'], e['i1']+1)
            idx = idx[m[idx]]
            k = int(idx[np.argmax(gap[idx])])
            players = list(defenders[a.g(k)])+[p for p in at(k) if p not in defenders[a.g(k)]]
            met = {'gapDM': {'peak': _val('gapDM', gap[idx].max()), 'mean': _val('gapDM', gap[idx].mean())},
                   'lineHeight': {'peak': _val('lineHeight', height[idx].max()),
                                  'mean': _val('lineHeight', height[idx].mean())}}
            o = a.occurrence('line_height_and_gap', '|'.join(key), e['i0'], e['i1'], k, met, players,
                             e['censoredStart'], e['censoredEnd'], e['seconds'])
            occ.append(o)
            hints.append({'occurrence': o['id'], 'peak': a.g(k), 'candidates': [a.g(i) for i in idx],
                          'values': [float(gap[i]) for i in idx], 'players': players, 'strength': float(gap[idx].max())})
        involved = _modal_players([o['players'] for o in occ])
        conf = confidence(a, m, players=involved, structure=True)
        base = _baseline(a, 'gapDM', 'default', rule, thr, None)
        base['lineHeightThreshold'] = {'enter': _val('lineHeight', h_enter), 'exit': _val('lineHeight', h_exit),
                                       'context': used_h}
        metrics = dict(blocks, gapDM=_val('gapDM', np.median(gap[m])),
                       lineHeightInContext=_val('lineHeight', np.median(height[m])))
        if conf['overall'] == 'low':
            cls, sev, reasons = 'insufficient', None, ['low_confidence']
        else:
            cls, sev, reasons = 'review', 'review_opportunity', []
        hint = {'preset': 'defensive_gap', 'metric': 'gapDM', 'units': ('D', 'M1'), 'occurrences': hints,
                'typical': None, 'enter': thr['enter']}
        out.append(record(a, 'line_height_and_gap', 'high_line_with_gap', key, cls, sev, measured, longest, occ,
                          base, metrics, involved, conf, 'gapDM', reasons, hint if cls == 'review' else None))
    if not any(o['classification'] == 'review' for o in out):
        out += context_note()
    return out


def _related(a_label, b_label):
    """Two formation labels of one family (team_shape.RELATED: they differ in one detail)."""
    return bool(a_label and b_label and a_label != b_label and frozenset((a_label, b_label)) in ts.RELATED)


def detect_formation(a):
    """Formation stability from team_shape's stabilized periods (module docstring), on registered samples only. A
    review is either a sustained change between unrelated formations (each held at least FORMATION_SUSTAINED_SECONDS)
    or a reading ambiguous between unrelated families for at least half the measured time: ambiguity or a change
    between shapes of one family (4-3-3 / 4-1-4-1, 3-5-2 / 5-3-2, 3-4-3 / 5-4-1, 4-4-2 / 4-2-3-1) is one shape read two
    ways, a neutral note ('related_shapes'). Role stability is each player's share of the time in his dominant role
    (the periods' role shares, weighted by their measured seconds), with the number of players at
    FORMATION_ROLE_SHARE or more; back-line group changes count between consecutive complete windows only (a window
    fitted with 9 of 10 players has a 3-man back line by construction)."""
    s = a.s
    t = a.t
    labels_at = np.full(a.n, '', dtype='<U12')
    period_at = np.full(a.n, -1, int)
    for k, p in enumerate(s.periods):
        inside = (t >= float(p['t0'])-1e-6) & (t < float(p['t1'])-1e-6) if k < len(s.periods)-1 else \
            (t >= float(p['t0'])-1e-6) & (t <= float(p['t1'])+1e-6)
        labels_at[inside] = str(p.get('label') or '')
        period_at[inside] = k
    registered = a.registered()
    formation = np.isin(labels_at, list(ts.TEMPLATES)) & registered
    measured, longest = a.measured(formation), a.longest(formation)
    win = [w for w in s.windows if a.t0-1e-6 <= float(w['centre']) <= a.t1+1e-6]
    classified = [w for w in win if w.get('status') == 'ok']
    by_centre = {float(w['centre']): w for w in s.windows if w.get('status') == 'ok'}
    roles = [wr for wr in s.window_roles if a.t0-1e-6 <= float(wr[0]) <= a.t1+1e-6]

    def complete(wr):
        w = by_centre.get(float(wr[0])) or {}
        exp = w.get('expected')
        return _num(exp) and len(wr[2]) >= int(exp)
    full_roles = [wr for wr in roles if complete(wr)]
    sets = [frozenset(p for p, g in wr[2].items() if g == 'D') for wr in full_roles]
    changes = sum(1 for x, y in zip(sets, sets[1:]) if x != y)
    distinct = len(set(sets))
    # The formation periods in range, by their measured (registered) seconds.
    secs = Counter()
    for k in np.unique(period_at[formation]):
        secs[int(k)] = a.measured(formation & (period_at == k))
    periods = sorted(secs, key=lambda k: float(s.periods[k]['t0']))
    labels = list(dict.fromkeys(str(s.periods[k]['label']) for k in periods))
    by_label = Counter()
    for k in periods:
        by_label[str(s.periods[k]['label'])] += secs[k]
    dominant = by_label.most_common(1)[0][0] if by_label else None
    main = max((k for k in periods if s.periods[k]['label'] == dominant), key=lambda k: secs[k], default=None)
    P = s.periods[main] if main is not None else {}
    tot = sum(secs.values())
    # Each player's share of the time in his dominant role.
    share_sum, weight = Counter(), Counter()
    for k in periods:
        for pid, r in (s.periods[k].get('roles') or {}).items():
            if isinstance(r, dict) and _num(r.get('share')):
                share_sum[pid] += secs[k]*float(r['share'])
                weight[pid] += secs[k]
    role_shares = {pid: _share(share_sum[pid]/weight[pid]) for pid in sorted(weight) if weight[pid] > 0}
    kept_role = sum(1 for v in role_shares.values() if v is not None and v >= FORMATION_ROLE_SHARE-1e-9)
    role_stability = kept_role/len(role_shares) if role_shares else None
    mean_share = float(np.mean(list(role_shares.values()))) if role_shares else None
    identity = [s.periods[k].get('identity') for k in periods if _num(s.periods[k].get('identity'))]
    ident = float(np.mean(identity)) if identity else None
    ambiguous = sum(secs[k] for k in periods if s.periods[k].get('confidence') == 'ambiguous')
    ambiguous_unrelated = sum(secs[k] for k in periods if s.periods[k].get('confidence') == 'ambiguous'
                              and not _related(str(s.periods[k].get('label')), s.periods[k].get('alternative')))
    unclear = a.measured(np.isin(labels_at, ('unclear', 'transition')) & registered)
    # A sustained change: two labels of different families, each held long enough.
    held = [lab for lab, sec in by_label.items() if sec >= FORMATION_SUSTAINED_SECONDS-1e-9]
    unrelated_pairs = [(x, y) for i, x in enumerate(held) for y in held[i+1:] if not _related(x, y)]
    related_only = len(labels) >= 2 and not unrelated_pairs
    alt = P.get('alternative')
    metrics = {'label': dominant, 'labelConfidence': P.get('confidence'), 'alternative': alt,
               'related': bool(_related(dominant, alt)) if alt else False,
               'margin': _round(P.get('margin'), 3), 'support': _share(P.get('support')),
               'roleStability': _share(role_stability), 'playersKeptRole': int(kept_role),
               'playersAssessed': len(role_shares), 'roleShareThreshold': FORMATION_ROLE_SHARE,
               'meanRoleShare': _share(mean_share), 'roleShares': role_shares,
               'defenderSetChanges': int(changes), 'distinctDefenderSets': int(distinct),
               'completeWindows': len(full_roles), 'labels': labels,
               'labelSeconds': {k: _sec(v) for k, v in by_label.items()}, 'classifiedWindows': len(classified),
               'windows': len(win), 'unclearSeconds': _sec(unclear), 'ambiguousSeconds': _sec(ambiguous),
               'ambiguousUnrelatedSeconds': _sec(ambiguous_unrelated),
               'unregisteredSeconds': _sec(a.measured(np.isin(labels_at, list(ts.TEMPLATES)) & ~registered)),
               'identity': _share(ident), 'strengthSeconds': FORMATION_STRENGTH_SECONDS}
    D = sorted(set().union(*sets)) if sets else []
    id_level = None
    if ident is not None:
        id_level = 'high' if ident >= ts.CLEAR_IDENTITY else ('medium' if ident >= FORMATION_MEDIUM_IDENTITY else 'low')
    conf = confidence(a, formation, players=list(s.ids))
    if id_level and id_level != 'high':
        conf['identity'] = _worst(conf['identity'], id_level)
        conf['overall'] = _worst(conf['overall'], id_level)
        conf['reasons'] = list(dict.fromkeys(conf['reasons']+['identity']))
    if _RANK[conf['overall']] < _RANK[FORMATION_MAX_CONFIDENCE]:
        conf['overall'] = FORMATION_MAX_CONFIDENCE
        conf['reasons'] = list(dict.fromkeys(conf['reasons']+['formation_unverified']))
    base = {'reference': None, 'rule': 'team_shape periods', 'context': 'all', 'threshold': None,
            'professionalRatePerMinute': None, 'professionalShareAbove': None, 'available': False,
            'note': 'team_shape confidence levels (clear / likely / ambiguous); no professional distribution'}
    statuses = Counter(w.get('status') for w in win if w.get('status') != 'ok')
    reasons = []
    pattern, cls, sev, occ, hints, state = 'formation_stable', 'neutral', None, [], [], 'descriptive'
    if measured < MIN_MEASURED_SECONDS-1e-9 or len(classified) < FORMATION_MIN_WINDOWS:
        cls = 'insufficient'
        reasons = ['no_formation']+[{'coverage': 'incomplete_team', 'direction': 'no_direction', 'lineup': 'few_players',
                                     'empty': 'incomplete_team'}.get(st, 'no_formation') for st, _ in statuses.most_common()]
        if metrics['unregisteredSeconds'] and metrics['unregisteredSeconds'] >= MIN_MEASURED_SECONDS:
            reasons.append('not_calibrated')
        if a.run['fewPlayers']:
            reasons.append('few_players')
    elif conf['overall'] == 'low':
        cls, reasons = 'insufficient', ['low_confidence']
    elif unrelated_pairs:
        pattern, cls, sev = 'formation_change', 'review', 'review_opportunity'
    elif len(labels) >= 2:
        pattern, state = 'formation_change', ('related_shapes' if related_only else 'descriptive')
    elif P.get('confidence') == 'ambiguous' and ambiguous_unrelated >= .5*tot:
        pattern, cls, sev = 'formation_ambiguous', 'review', 'review_opportunity'
    elif P.get('confidence') == 'ambiguous' and ambiguous >= .5*tot:
        pattern, state = 'formation_ambiguous', 'related_shapes'
    elif (measured >= FORMATION_STRENGTH_SECONDS-1e-9 and P.get('confidence') == 'clear' and len(labels) == 1
          and _num(role_stability) and role_stability >= FORMATION_MIN_ROLE_STABILITY
          and changes <= FORMATION_MAX_SET_CHANGES and conf['overall'] in ('high', 'medium')):
        cls, sev = 'strength', 'observed_strength'
    elif measured < FORMATION_STRENGTH_SECONDS-1e-9:
        state = 'under_measured'
    if cls == 'insufficient':
        metrics['alternative'] = None
    if cls == 'review':
        for k in periods:
            p = s.periods[k]
            m = formation & (period_at == k)
            idx = np.flatnonzero(m)
            if not len(idx):
                continue
            mid = idx[np.argmin(np.abs(t[idx]-(t[idx[0]]+t[idx[-1]])/2))]
            pd = sorted(pid for pid, r in (p.get('roles') or {}).items() if isinstance(r, dict) and r.get('group') == 'D')
            o = a.occurrence('formation_stability', f'period|{p["label"]}', int(idx[0]), int(idx[-1]), int(mid),
                             {}, pd, bool(idx[0] == 0 and float(p['t0']) < a.t0-1e-6),
                             bool(idx[-1] == a.n-1 and float(p['t1']) > a.t1+1e-6))
            o['label'], o['labelConfidence'] = p.get('label'), p.get('confidence')
            occ.append(o)
            hints.append({'occurrence': o['id'], 'peak': a.g(int(mid)), 'candidates': [a.g(i) for i in idx],
                          'values': [float(a.s.observed[a.g(i)]) for i in idx], 'players': pd,
                          'strength': float(secs[k])})
    typical = None
    if cls in ('strength', 'neutral'):
        idx = np.flatnonzero(formation & (period_at == main)) if main is not None else np.zeros(0, int)
        typical = {'candidates': [a.g(i) for i in idx], 'values': [float(a.s.observed[a.g(i)]) for i in idx],
                   'target': None}
    hint = {'preset': 'formation_shape', 'metric': None, 'units': ('D', 'M', 'F'), 'occurrences': hints,
            'typical': typical, 'enter': None} if cls != 'insufficient' else None
    return [record(a, 'formation_stability', pattern, ('formation',), cls, sev, measured, longest, occ, base, metrics,
                   D, conf, None, reasons, hint, rated=False, state=state)]


def _independent(windows, seconds):
    """The windows that do not overlap (reference.independent_windows)."""
    return refs.independent_windows(windows, seconds)


def detect_lateral_shift(a):
    """The back line's sideways movement against the midfield's (module docstring): the lag and correlation of the
    reference's own computation (reference.dm_velocities, reference.lateral_windows: 30 s windows every 5 s inside
    continuous measured stretches, the units' velocity the mean of their members' velocities, windows at the lag
    search's bound left out), 20 s windows (context 'window:20') when no stretch holds a 30 s one. Overlapping windows
    beyond the enter level form one occurrence; the count set against the professional window distribution is that
    of the late windows among the non-overlapping ones (the reference counts the same). A window whose correlation is
    below LAG_MIN_CORR measures, but never counts as late. The strength rule's favourable side is a median lag between
    the professional p10 and p50 (neither later than the typical professional back line nor unusually far ahead of
    its midfield). Lags are stated at the 5 Hz resolution (reference.quantize_lag)."""
    t_all, defence, midfield, eligible = refs.dm_velocities(a.cd, a.team)
    in_range = np.zeros(len(t_all), bool)
    in_range[a.rows] = True
    windows, length, context = [], LAG_WINDOW_SECONDS, 'all'
    for length, context in ((LAG_WINDOW_SECONDS, 'all'), (LAG_SHORT_WINDOW_SECONDS, LAG_SHORT_CONTEXT)):
        windows = refs.lateral_windows(t_all, midfield, defence, eligible & in_range, window_seconds=length)
        if windows:
            break
    n_samples = max(2, int(round(length/a.step)))
    pos = {int(g): i for i, g in enumerate(a.rows)}
    cover = np.zeros(a.n, bool)
    for w in windows:
        cover[[pos[g] for g in range(w['i0'], w['i0']+n_samples) if g in pos]] = True
    measured, longest = a.measured(cover), a.longest(cover)
    mask = a.ok('unitD') & a.ok('unitM')
    units = a.s.units

    def unit_ids(g):
        u = units[g] or {}
        return list(u.get('D') or ())+[p for p in u.get('M') or () if p not in (u.get('D') or ())]
    players = _modal_players([unit_ids(a.g(i)) for i in np.flatnonzero(cover if cover.any() else mask)])
    conf = confidence(a, cover if cover.any() else mask, players=players, structure=True)
    lags = np.array([w['lag'] for w in windows])
    corrs = np.array([w['corr'] for w in windows])
    rule = rule_of(a.reference, 'default')
    thr = thresholds(a.reference, 'lagDM', context, rule)
    if thr is not None and thr['context'] != context:
        thr = None              # 30 s levels are not those of 20 s windows (and the reverse)
    indep = _independent(windows, length)
    q = (lambda x: refs.quantize_lag(x, a.step))
    metrics = {'lag': q(np.median(lags)) if len(lags) else None,
               'corr': _round(np.median(corrs), 3) if len(corrs) else None, 'windows': len(windows),
               'independentWindows': len(indep), 'windowSeconds': float(length), 'lagResolutionSeconds': _sec(a.step)}
    metrics['median'] = metrics['lag']
    if thr is None or not windows:
        reasons = ([NO_REFERENCE] if a.reference is None or (windows and thr is None) else [])
        reasons += ['short_stretch']+not_measured(a, 'unitD') if not windows else []
        return [record(a, 'lateral_shift', 'lateral_lag', ('lateral_lag', context), 'insufficient', None, measured,
                       longest, [], _baseline(a, 'lagDM', 'default', rule, None, None), metrics, players, conf,
                       'lagDM', reasons or [NO_REFERENCE])]
    late = refs.late_windows(windows, thr['enter'], LAG_MIN_CORR)
    # Overlapping late windows are one situation.
    runs = []
    for w in late:
        if runs and w['t0'] <= runs[-1][-1]['t1']+a.step+1e-6:
            runs[-1].append(w)
        else:
            runs.append([w])
    occ, hints = [], []
    for run in runs:
        g0, g1 = run[0]['i0'], run[-1]['i0']+n_samples-1
        idx = [pos[g] for g in range(g0, g1+1) if g in pos and cover[pos[g]]]
        if not idx:
            continue
        best = max(run, key=lambda w: w['lag'])
        peak = pos.get(best['i0']+n_samples//2, idx[len(idx)//2])
        who = unit_ids(a.g(peak))
        o = a.occurrence('lateral_shift', f'lateral_lag|{context}', idx[0], idx[-1], peak,
                         {'lagDM': {'peak': q(best['lag']), 'mean': q(np.mean([w['lag'] for w in run]))},
                          'corrDM': {'peak': _round(best['corr'], 3),
                                     'mean': _round(np.mean([w['corr'] for w in run]), 3)}}, who)
        o['windows'] = len(run)
        occ.append(o)
        hints.append({'occurrence': o['id'], 'peak': a.g(peak), 'candidates': [a.g(i) for i in idx],
                      'values': [float(a.s.observed[a.g(i)]) for i in idx], 'players': who, 'strength': best['lag']})
    k = len(refs.late_windows(indep, thr['enter'], LAG_MIN_CORR))
    window = refs.window_baseline(a.reference, 'lagDM', context, measured/60.0)
    strength = strength_check(a.reference, 'lagDM', lags, measured, a.range_seconds, context, low=thr.get('p10'))
    metrics.update(windowsBeyond=k, shareAboveProfessionalP90=_share(np.mean(lags > float(thr['p90'])))
                   if _num(thr.get('p90')) else None)
    cls, sev, why, stats = classify(measured, longest, k, rule, window, strength, conf['overall'])
    metrics.update(_strength_metrics(strength), professionalWindowP99=_round(stats['windowP99'], 2),
                   professionalWindowP50=_round(stats['windowP50'], 2),
                   professionalWindowMax=_round(stats['windowMax'], 2), improvementBlocked=stats['improvementBlocked'])
    base = _baseline(a, 'lagDM', 'default', rule, thr, None, window)
    state = None
    if cls == 'neutral':
        state, spell = neutral_state(occ, strength)
        if spell:
            metrics['spell'] = spell
    typical_hint = None
    if not occ and cls != 'insufficient':
        idx = np.flatnonzero(cover)
        typical_hint = {'candidates': [a.g(i) for i in idx], 'values': [float(a.s.observed[a.g(i)]) for i in idx],
                        'target': None}
    hint = {'preset': 'line_movement', 'metric': 'lagDM', 'units': ('D', 'M'), 'occurrences': hints,
            'typical': typical_hint, 'enter': thr['enter'], 'arrows': True}
    involved = _modal_players([o['players'] for o in occ]) if occ else players
    return [record(a, 'lateral_shift', 'lateral_lag', ('lateral_lag', context), cls, sev, measured, longest, occ,
                   base, metrics, involved, conf, 'lagDM', why, hint if cls != 'insufficient' else None, state=state)]


def detect_isolation(a):
    """Per player, the certain nearest-teammate distance against the per-player professional window distribution of
    his unit's context (module docstring): each player measured at least MIN_MEASURED_SECONDS in his dominant unit is
    one test, its p-value the share of professional player-windows of his measured time with at least his count of
    spells (reference.window_tail, an upper estimate), and Holm's procedure over the players tested keeps the
    family-wise level at ISOLATION_ALPHA. A rejected player with at least MIN_OCCURRENCES spells becomes a review
    ("repeatedly far from the nearest teammate; check whether this is the player's role"); never an improvement or a
    concern, and never from a single spell. Everyone else is summarised in one team-level observation."""
    s = a.s
    near = a.val('nearestTeammate')
    ok = a.ok('nearestTeammate')
    group = s.values['group'][a.rows]
    rule_name = 'isolation' if a.reference and 'isolation' in (a.reference.get('rules') or {}) else 'default'
    rule = rule_of(a.reference, rule_name)
    tests, quiet, measured_any = [], [], np.zeros(a.n, bool)
    for j, pid in enumerate(s.ids):
        m = ok[:, j] & np.isfinite(near[:, j])
        if not m.any():
            continue
        measured_any |= m
        groups_j = Counter(g for g in group[m, j].tolist() if g)
        unit = groups_j.most_common(1)[0][0] if groups_j else None
        mm = m & (group[:, j] == unit) if unit else m
        context = NEAREST_CONTEXT.format(unit) if unit else 'all'
        thr = thresholds(a.reference, 'nearestTeammate', context, rule) if a.reference else None
        measured = a.measured(mm)
        if thr is None or measured < MIN_MEASURED_SECONDS-1e-9:
            quiet.append(pid)
            continue
        eps = ep.episodes(a.t, near[:, j], mm, thr['enter'], thr['exit'], rule['minSeconds'], rule['mergeSeconds'],
                          'above', step=a.step)
        conf = confidence(a, mm, cover=m, player=pid)
        window = refs.window_baseline(a.reference, 'nearestTeammate', thr['context'], measured/60.0, rule_name,
                                      per_player=True)
        p = refs.window_tail(a.reference, 'nearestTeammate', thr['context'], measured/60.0, len(eps), rule_name,
                             per_player=True) if window.get('available') else None
        testable = (p is not None and conf['overall'] != 'low' and a.longest(mm) >= rule['minSeconds']-1e-9)
        tests.append({'pid': pid, 'j': j, 'mask': mm, 'thr': thr, 'eps': eps, 'conf': conf, 'window': window,
                      'p': p if testable else None, 'measured': measured})
    rejected = set(refs.holm([x['p'] for x in tests], ISOLATION_ALPHA))
    tested = sum(1 for x in tests if x['p'] is not None)
    out = []
    for i, x in enumerate(tests):
        if i not in rejected or len(x['eps']) < MIN_OCCURRENCES:
            quiet.append(x['pid'])
            continue
        pid, j, mm, thr = x['pid'], x['j'], x['mask'], x['thr']
        occ, hints = [], []
        key = ('player_isolation', thr['context'], pid)
        for e in x['eps']:
            idx = np.arange(e['i0'], e['i1']+1)
            idx = idx[mm[idx]]
            o = a.occurrence('isolation', '|'.join(key), e['i0'], e['i1'], e['peakIndex'],
                             {'nearestTeammate': {'peak': _val('nearestTeammate', e['peak']),
                                                  'mean': _val('nearestTeammate', e['mean'])}}, [pid],
                             e['censoredStart'], e['censoredEnd'], e['seconds'])
            o['context'] = thr['context']
            occ.append(o)
            hints.append({'occurrence': o['id'], 'peak': a.g(e['peakIndex']), 'candidates': [a.g(k) for k in idx],
                          'values': [float(near[k, j]) for k in idx], 'players': [pid], 'strength': float(e['peak'])})
        rank = sorted(t['p'] for t in tests if t['p'] is not None).index(x['p'])
        metrics = dict(_summary('nearestTeammate', near[:, j], mm, thr), pValue=_round(x['p'], 5),
                       holmLevel=_round(ISOLATION_ALPHA/(tested-rank), 5), testedPlayers=tested,
                       professionalWindowP99=_round(x['window'].get('p99'), 2),
                       professionalWindowP50=_round(x['window'].get('p50'), 2), roleCheck=True)
        hint = {'preset': 'player_movement', 'metric': 'nearestTeammate', 'units': None, 'player': pid,
                'occurrences': hints, 'typical': None, 'enter': thr['enter']}
        out.append(record(a, 'isolation', 'player_isolation', key, 'review', 'review_opportunity', x['measured'],
                          a.longest(mm), occ, _baseline(a, 'nearestTeammate', rule_name, rule, thr, None, x['window']),
                          metrics, [pid], x['conf'], 'nearestTeammate', [], hint))
    # The team-level summary of everyone without a review.
    cols = [s.col[p] for p in quiet]
    vals = near[:, cols][ok[:, cols]] if cols else np.zeros(0)
    team_rows = (ok[:, cols] & np.isfinite(near[:, cols])).any(axis=1) if cols else np.zeros(a.n, bool)
    measured, longest = a.measured(team_rows), a.longest(team_rows)
    thr = thresholds(a.reference, 'nearestTeammate', 'all', rule)
    metrics = {'median': _val('nearestTeammate', np.median(vals)) if len(vals) else None,
               'p90': _val('nearestTeammate', np.percentile(vals, 90)) if len(vals) else None,
               'players': len([p for p in quiet if (ok[:, s.col[p]]).any()]), 'testedPlayers': tested,
               'reviewPlayers': len(out), 'familyAlpha': ISOLATION_ALPHA}
    if thr is not None and len(vals) and _num(thr.get('p90')):
        metrics['shareAboveProfessionalP90'] = _share(np.mean(vals > float(thr['p90'])))
    conf = confidence(a, team_rows, players=quiet)
    if a.reference is None or thr is None:
        cls, reasons = 'insufficient', [NO_REFERENCE]
    elif measured < MIN_MEASURED_SECONDS-1e-9:
        cls, reasons = 'insufficient', ['short_measured']
    elif conf['overall'] == 'low':
        cls, reasons = 'insufficient', ['low_confidence']
    else:
        cls, reasons = 'neutral', []
    if cls == 'insufficient':
        reasons += not_measured(a, 'nearestTeammate')
        if a.run['smallScene']:
            reasons += ['small_scene_veto', 'broadcast_only']
    if out and cls != 'insufficient':
        return out              # the reviews stand; a team-level "no finding" note would contradict them
    if out and not measured_any.any():
        return out
    rec = record(a, 'isolation', 'player_isolation', ('player_isolation', 'team'), cls, None, measured, longest, [],
                 _baseline(a, 'nearestTeammate', rule_name, rule, thr, None), metrics,
                 [p for p in quiet if (ok[:, s.col[p]]).any()], conf, 'nearestTeammate', reasons, None,
                 state='under_measured' if measured < strength_seconds(a.range_seconds)-1e-9 else 'absent')
    return out+[rec]


def detect_ball(a):
    """Where the measured ball was, from this team's point of view: descriptive, always neutral (or insufficient)."""
    third, channel = a.val('ballThird'), a.val('ballChannel')
    ok = a.ok('ballThird') & (third != '')
    measured, longest = a.measured(ok), a.longest(ok)
    share = measured/a.range_seconds if a.range_seconds else 0.0
    ball_level = 'high' if share >= .7 else ('medium' if share >= .3 else 'low')
    conf = confidence(a, ok, ball=ball_level, players=[])
    conf['tracking'] = 'n/a'
    conf['overall'] = _worst(conf['calibration'], ball_level)
    metrics = {'thirds': {x: _share(np.mean(third[ok] == x)) if ok.any() else None for x in THIRDS},
               'channels': {x: _share(np.mean(channel[ok] == x)) if ok.any() else None
                            for x in ('left', 'centre', 'right')},
               'measuredShare': _share(share)}
    if measured < MIN_MEASURED_SECONDS-1e-9:
        cls, reasons = 'insufficient', ['no_ball']+not_measured(a, 'ballThird')
    else:
        cls, reasons = 'neutral', []
    base = {'reference': None, 'rule': None, 'context': 'all', 'threshold': None, 'professionalRatePerMinute': None,
            'professionalShareAbove': None, 'available': False, 'note': 'descriptive: no professional comparison'}
    return [record(a, 'ball_context', 'ball_location', ('ball_location',), cls, None, measured, longest, [], base,
                   metrics, [], conf, None, reasons, None, rated=False, state='descriptive')]


DETECTORS = {
    'compactness': detect_compactness,
    'unit_gap_dm': detect_unit_gap_dm,
    'unit_gap_mf': detect_unit_gap_mf,
    'back_line_coordination': detect_back_line,
    'line_height_and_gap': detect_line_height_gap,
    'formation_stability': detect_formation,
    'lateral_shift': detect_lateral_shift,
    'isolation': detect_isolation,
    'ball_context': detect_ball,
}


def detect(cd, team, reference=None, t0=None, t1=None, only=None, library=None):
    """The observations of one team ('A' or 'B') over [t0, t1] (absolute seconds; the whole segment by default) from
    the coaching series cd (series.build_series). reference: a reference dict, a path, None (reference.json) or False
    (none). only: detector names to run (default all, in registry order). The records keep private '_ev' evidence
    hints until evidence.attach has used them (public() strips them)."""
    a = Analysis(cd, team, resolve_reference(reference), t0, t1, library)
    out = []
    for name, fn in DETECTORS.items():
        if only and name not in only:
            continue
        out += fn(a)
    return out
