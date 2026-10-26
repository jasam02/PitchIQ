"""The coaching v2 opportunity finders (DESIGN v2 section 4.0-4.6): where in the match a coaching behaviour was
relevant, and what was measured there, sample by sample, for any level.

Why opportunities instead of verdicts. The v1 engine judged each metric against one professional match and turned a
threshold into a canned sentence; the critic found it counted time it never compared (B1), mixed units (B2) and
credited phase-blind compactness (B3), and the audit measured identical observations at every level. Here the
detectors only FIND the situations where a behaviour matters (the other team has the ball and a carrier is known; our
wide defender stands well beyond the centre-backs while we have it; ...) and record the relations there (the first
defender's distance, the cover behind him, who occupies the vacated channel, ...). Whether a relation "held" is decided
later by soccer/coaching/assess.py with the level's own parameters, so the same measured opportunities serve every
age and level and never change with them (invariant 2: these records are profile-independent).

Detectors (S.DETECTORS) and the tier each runs in (DESIGN 1.2):
  D1 pressure_cover       measurement  the other team's phase, a carrier from possession-4's NON-estimated control
                          runs; the first defender (our outfield player nearest the carrier: the disc of that radius in
                          view and nobody of our kit unplaced, so nobody unseen could be nearer); his distance d1,
                          closing speed, goal-side flag; the nearest team-mate within 60 deg of his line to our goal
                          (coverDist / coverAngle) and whether the 15 m (x min(1, L/105)) disc around him was in view.
                          Opportunities: one per carrier spell of at least 1 s (cut at 10 s).
  D2 weak_side_balance    measurement  the other team's phase, the ball reliable in an outer third, our team complete:
                          defenders goal-side of the ball on the far side, the open far width, central protection.
  D4 behind_ball          measurement  our phase in spans of 3 s or more, the ball reliable beyond our third, our team
                          complete: players at least 10 m (x L/105) goal-side of the ball, the opponents that far on our
                          side of it and the margin (both teams complete).
  D5 defensive_spacing    measurement  the other team's phase in spans of 3 s or more (11v11): the recognised back line
                          (series' defensive line: complete, from the formation, not displaced), its depth std, the
                          largest lateral gap, whether it is on the ball's side and holds an opponent, the D-M gap
                          (NaN with interleaved units, critic B2) and the block height.
                          role (small formats): the same from the relative back block of the role book.
  D6 attacking_shape      measurement  our phase in spans of 3 s or more, our team complete: width, depth, their shares
                          of the field, the lateral channels and length thirds occupied. With no phase anywhere (no
                          possession record, no coach span) the complete samples become describe-only opportunities
                          (phase 'unknown').
  D3 wide_defender_advance role        our phase in spans of 3 s or more, outside change windows: a wide defender (LB /
                          RB / LWB / RWB from a coach, stable or carried role; on small formats the outer members of a
                          relative back block of three or more) at least 15 m (x L/105) beyond the centre-back line
                          (or, on small formats, 0.1 L beyond halfway); the cover tests (a) channel occupied, (b) the
                          holding midfielder dropped, (c) the near centre-back shifted, (d) players behind the ball,
                          (e) an interchange, (f) opponents in the vacated channel, and the rest-defence margin. Samples
                          where the defender runs back at 3 m/s or more are never an opportunity (counterexample
                          "recovering"); with the team incomplete the relaxed gate asks the vacated channel to be in
                          view.
Every opportunity carries its phase source ('possession-4' estimated, or 'coach' marked), the ball zone and channel,
the actors, per-sample relations, quality (identity, position, visibility, ball) and intent signals; every detector
reports its eligible seconds and, per abstention reason (S.ABSTAIN_REASONS), the seconds lost (spec Part 6: say what
was not assessed rather than fabricate it). Measured coverage on the run copies and Metrica game 1 is in INTERFACES.md
(DETECT, wave 3).

Phase rules (possession v4 docs, binding): the phase is the timeline's (team per frame only; +-2 s around every change
removed; dead spells removed; coach-marked spans trusted as given); shape and spacing checks (D3-D6) use spans of 3 s
or more; nothing here counts or names a change of possession, and change windows only ever SUPPRESS (they are
'near_change' abstentions, never opportunities).

The role tier reads only the obs file (INTERFACES.md LEAD 4), so detect() stores what D3 and the small-format D5 need
in obs['roleInputs']['detect'] (role_geometry(): the own-phase samples of spans of 3 s or more, and the other team's
on small formats, with every player's attack-frame position, velocity, observed / identity bits, the camera view of
the outer-third channels, the ball and each player's median lateral position per formation period) beside CONTEXT's
obs['roleInputs']['context'] (situations.SituationTimeline.role_inputs()).

sample_outcomes() classifies an opportunity's samples with a parameter set (assess.py passes the profile's; key_sample()
uses REFERENCE_PARAMS, DESIGN Appendix A's values, so EVIDENCE can pick a typical moment profile-independently).

Every constant is a PITCHIQ-PROPOSAL (origin PITCHIQ-PROPOSAL, unvalidated) unless a measured basis is quoted.

Fix round 1 (2026-10-23, the critic's B-1, B-5, B-7, S-2, S-4, S-16): relations were ADDED, never changed (the far
side with a level tolerance, the ball's depth on an advance, a geometric interchange, free opponents between the lines,
the space to the opponents' last line, coach-marked change windows) and sample_outcomes judges with them: the
three-state pressure call, no far-side judgement within 20 m of the own goal, "someone back" at levels 1-2 only with
the ball in the final third and a covered advance protected, the D-M gap only with a free opponent between the lines
in reach of the ball, depth by the field share or the available length. Measured on the five run copies (196 reports,
14 profiles x 2 teams x 7 variants): the 18 criticisms of the old rules (BC60 / BC303 Barcelona "nobody stayed back"
at level 1, BC180 Barcelona "far side open" at levels 3-7 and "someone high and someone deep" at levels 1-2, BC303 A's
compact-style ball-side gap) -> 0. On Metrica game 1 (full tracking, 3-minute clips) the professional D-M gap reviews
at levels 6-7 fell from 8 of 62 clips (the plain 18 m rule; "any opponent between the lines" still 8) to 3 with the
free-opponent gate and 0 with assess's review floor; 0 of 218 professional "pressed without cover" failing samples at
level 4 (15-minute clips) sit at 3-5 m with the defender moving away (the STATS verifier found 101 of 270 on its
10-minute clips under the two-state call).

Fix round 3 (2026-10-24, critic B-1 / COACH F-1, LEAD decision (2), critic N1): relations ADDED again, none changed
(the before and after tiers of Metrica game 1, full tracking, truth phase and camera, are equal once the new keys and
the code fingerprints are dropped): contestS on every opportunity (situations.contestS; the role tier through
roleInputs.detect.samples.contest) and gapOppFreeM on D5 (gap_opponent_free_m: the least-marked opponent in the
gap, his distance to the nearest of ours goal-side of him). sample_outcomes at levels 1-2: an uncovered advance with
an opponent in the vacated channel fails wherever the ball is and a covered one is +1 wherever it is (levels 1-2
judged only the final third: 9 of the 10 planted unsupported advances, the full-back 20 m up with B-10 in the empty
channel, were never mentioned at U8 / U10; the probe of COACH round 2 found the round-1 real-clip cases BC60 B x3,
BC303 B, BC180 B demonstrated under it); the failure modes 'exposed' / 'uncovered' / 'nobody_behind' are recorded for
D3 and level 1's reframed advance. On the 196 run-copy reports and the user's 9 runs (252 reports) no criticism
appears at levels 1-2 after the change (0 before); on Metrica game 1 none on the 62 three-minute and 12 fifteen-minute
clips, the halves or the whole match, full tracking or camera (0 before).
"""
import math
import re
from collections import Counter

import numpy as np

from . import schema as S
from . import situations as SI
from . import tactics_geom as G

DETECT_VERSION = 1              # roleInputs['detect'] layout version
OPP_MERGE_S = .6                # s: an eligibility gap this short does not split an opportunity (its samples are left
                                # out)
OPP_MAX_S = 10.0                # s: longer eligible stretches are cut into opportunities of at most this (D1, D2,
                                # D4-D6);
                                # recurrence is then counted in DESIGN 3.6's distinct spells (10 s apart)
D3_MAX_S = 30.0                 # s: a wide defender's advance is one opportunity up to this long (a wing-back holding
                                # high through a possession is one spell, not three)
MIN_S = {'pressure_cover': 1.0, 'weak_side_balance': 2.0, 'wide_defender_advance': 3.0, 'behind_ball': 2.0,
         'defensive_spacing': 2.0, 'attacking_shape': 2.0}     # DESIGN 4.1-4.6 "Spell >= ..."
POSITION_HIGH, POSITION_MEDIUM = .7, .35    # median registration confidence of the samples (pitch_model HIGH / MEDIUM)
VIEW_STATION_M = 5.0            # m: the outer-third channel's view is stored at depth stations this far apart
VIEW_EDGE_M = 1.0               # m: the channel's corners are tested this far inside the touchline
MEDIAN_MIN_SAMPLES = 25         # samples (5 s) a player needs for a median lateral position (fe_geom)
CONFIDENT = ('coach', 'stable', 'carried')   # role sources a role-based rule may lean on (DESIGN 3.4)
WIDE_ROLES = {'left': ('LB', 'LWB'), 'right': ('RB', 'RWB')}
CB_ROLES = ('LCB', 'CB', 'RCB')
DM_ROLES = ('CDM', 'LDM', 'RDM')
INTERCHANGE_S = 10.0            # s: an interchange this close to a sample re-occupies the slot (test e; DESIGN 3.4)
NA = 'n/a'                      # a sample where the behaviour was not relevant (not an abstention)
PHASE_REASONS = ('phase_unknown', 'near_change', 'possession_unavailable')
REASON_DTYPE = '<U24'
REASON_ORDER = ('phase_unknown', 'near_change', 'possession_unavailable', 'registration_low', 'scale_unknown',
                'format_not_supported', 'role_unconfirmed', 'no_carrier', 'ball_unreliable', 'out_of_view',
                'team_incomplete', 'units_interleaved', 'recovering', 'identity_uncertain', 'short_range')
D4_JUDGED_ZONE = 'final_third'  # behind_ball is JUDGED with the ball in the final third only (opportunities are still
                                # found from the end of our own third, DESIGN 4.4, and keep their ball depth): the
                                # professional
                                # basis is the final third (margin p10 / p50 +1 / +2, a negative margin never lasted 2 s),
                                # while in the middle third professionals have a margin of 0 or less on 27-35% of samples
                                # and broadcast 0-60 s Barcelona building up (ball 40-60 m) had nobody 10 m behind the
                                # ball on most samples, their centre-backs 5 m behind it
SERIES_LINE_REASON = {'not_calibrated': 'registration_low', 'low_calibration': 'registration_low',
                      'disagreement': 'registration_low'}   # every other back-line code: 'team_incomplete'

# DESIGN Appendix A's values (the level-4 profile's), for key_sample() only: assess.py always uses the profile's.
REFERENCE_PARAMS = {
    'pressure_cover': {'pressureM': 5.0, 'pressureCertainM': 3.0, 'pressureClosingMS': 1.0, 'goToBallClosingMS': 1.5,
                       'goToBallMaxM': 8.0, 'coverRadiusM': 12.0, 'coverRadiusMinM': 6.0, 'coverAngleDeg': 60.0,
                       'pressureRule': 'distance', 'coverRequired': True, 'carrierZone': 'any',
                       'unpressuredZone': 'own_third'},
    'weak_side_balance': {'minWeakCount': 1, 'excludeOwnGoalM': 20.0, 'excludeOwnGoalScale': 'length'},
    'wide_defender_advance': {'advancedM': 15.0, 'advancedScale': 'length', 'advancedRule': 'cb_line',
                              'advancedHalfwayShare': .1, 'countOnly': 'uncovered', 'useDmRotation': False,
                              'recoveringMS': 3.0, 'cbShiftM': 4.0},
    'behind_ball': {'behindRule': 'count', 'minBehind': 2, 'minMargin': 1, 'behindBallZone': 'final_third'},
    'defensive_spacing': {'lateralGapM': 20.0, 'lateralGapScale': 'width', 'dmGapM': 18.0, 'dmGapScale': 'length',
                          'dmGapStrengthLoM': 8.0, 'dmGapStrengthLoScale': 'length', 'dmGapStrengthHiM': 14.0,
                          'dmGapStrengthHiScale': 'length', 'lineStdStrengthM': 2.5},
    'attacking_shape': {'widthShare': .55, 'depthShare': .30, 'judgeWidth': True, 'depthRule': 'available_or_field',
                        'depthAvailShare': .75},
}


# ---------- small helpers ----------
def _py(v, digits=None):
    """A JSON-safe Python scalar (numpy types converted, NaN -> None, floats rounded when digits is given)."""
    if v is None:
        return None
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        f = float(v)
        if not math.isfinite(f):
            return None
        return round(f, digits) if digits is not None else f
    if isinstance(v, (str, np.str_)):
        return str(v)
    return v


def _modal(values, exclude=(None, '')):
    """The most frequent value (first seen wins a tie), None when there is none."""
    c = Counter(v for v in values if v not in exclude)
    return c.most_common(1)[0][0] if c else None


def _ranked(values, exclude=(None, ''), limit=3):
    """Distinct values by frequency (first seen wins a tie)."""
    c = Counter(v for v in values if v not in exclude)
    return [v for v, _ in c.most_common(limit)]


def segments(t, eligible, keys=None, merge_s=OPP_MERGE_S, min_s=1.0, max_s=OPP_MAX_S, step=.2):
    """Opportunity sample sets: runs of eligible samples with one key, bridged across gaps of at most merge_s (the gap's
    samples are left out), cut into pieces of at most max_s (a short tail joins the piece before it), kept from min_s
    (t1 - t0 + step). Returns [np.ndarray of sample indices] in time order."""
    t = np.asarray(t, float)
    idx = np.flatnonzero(np.asarray(eligible, bool))
    runs, cur = [], []
    for i in idx:
        i = int(i)
        if cur and (t[i]-t[cur[-1]]-step > merge_s+.5*step or (keys is not None and keys[i] != keys[cur[0]])):
            runs.append(cur)
            cur = []
        cur.append(i)
    if cur:
        runs.append(cur)
    out = []
    for run in runs:
        pieces, piece = [], [run[0]]
        for i in run[1:]:
            if t[i]-t[piece[0]]+step > max_s+1e-9:
                pieces.append(piece)
                piece = []
            piece.append(i)
        pieces.append(piece)
        if len(pieces) > 1 and t[pieces[-1][-1]]-t[pieces[-1][0]]+step < min_s-1e-9:
            tail = pieces.pop()
            pieces[-1] = pieces[-1]+tail
        out += [np.asarray(p, int) for p in pieces if t[p[-1]]-t[p[0]]+step >= min_s-1e-9]
    return out


def _coach_near(tl):
    """Per sample: within CHANGE_EXCLUDE_S of a coach-marked change between the two teams' spans (situations
    coachNear; False when the timeline has none)."""
    try:
        m = np.asarray(tl.coachNear, bool)
    except AttributeError:
        return np.zeros(len(tl.t), bool)
    return m if len(m) == len(tl.t) else np.zeros(len(tl.t), bool)


def _contest(tl):
    """Per sample: situations' contestS (seconds to the nearest ball the possession engine calls loose or contested;
    NaN where it names none or the timeline has no such field)."""
    try:
        a = np.asarray(tl.contestS, float)
    except AttributeError:
        return np.full(len(tl.t), np.nan)
    return a if len(a) == len(tl.t) else np.full(len(tl.t), np.nan)


def _contest_rel(contest, idx):
    """The 'contestS' relation of an opportunity's samples (None where not measured)."""
    return [_py(contest[i], 2) for i in idx]


def _phase_reasons(phase, near, state):
    """Per sample: '' where the phase is `state` (to be gated further), 'near_change' / 'phase_unknown' where it is
    unknown (inside a change window / elsewhere), NA for the other phases and dead ball. A sample of the phase inside a
    change window is 'near_change' too: the engine never names one, but a coach-marked change between the two teams'
    spans keeps its phase and still suppresses (fix round 1, critic S-16)."""
    r = np.full(len(phase), NA, dtype=REASON_DTYPE)
    r[phase == state] = ''
    r[(phase == state) & near] = 'near_change'
    unk = phase == 'unknown'
    r[unk & near] = 'near_change'
    r[unk & ~near] = 'phase_unknown'
    return r


def _gate(reason, fail, code):
    """Give `code` to the still-eligible samples where `fail` holds (in gate order: the first failed gate names it)."""
    reason[(reason == '') & np.asarray(fail, bool)] = code


def _seconds(counts, step):
    return {k: round(float(v)*step, 2) for k, v in counts.items() if v}


def _reason_counts(reason):
    codes, n = np.unique(reason[(reason != '') & (reason != NA)], return_counts=True)
    return {str(c): int(k) for c, k in zip(codes, n) if str(c) in S.ABSTAIN_REASONS}


def eligible_seconds(opps, step):
    """Seconds covered by the opportunities' samples, each sample time counted once (a left and a right wide-defender
    opportunity may share their samples)."""
    return len({round(t, 3) for o in opps for t in o['relations']['t']})*step


def coverage(detector, tier, opps, step, phase_seconds, abstain_seconds):
    """A DetectorCoverage record: eligible seconds = the samples inside emitted opportunities (each stands for one
    step), their spans (merged), the relevant phase seconds and the seconds lost per abstention reason with the
    dominant one."""
    abstain = {k: round(float(v), 2) for k, v in sorted(abstain_seconds.items()) if v > 0 and k in S.ABSTAIN_REASONS}
    spans = []
    for o in sorted(opps, key=lambda o: o['t0']):
        if spans and o['t0'] <= spans[-1][1]+step+1e-6:
            spans[-1][1] = max(spans[-1][1], o['t1'])
        else:
            spans.append([o['t0'], o['t1']])
    # The dominant reason is the largest loss among the detector's own gates on its relevant phase; the phase
    # reasons (unknown phase, change windows) name it only when nothing of that phase was lost to a gate.
    dominant = None
    gates = {k: v for k, v in abstain.items() if k not in PHASE_REASONS}
    pool = gates or abstain
    if pool:
        dominant = max(pool, key=lambda k: (pool[k], -REASON_ORDER.index(k) if k in REASON_ORDER else -99))
    return {'detector': detector, 'tier': tier,
            'eligibleSeconds': round(eligible_seconds(opps, step), 2),
            'phaseSeconds': None if phase_seconds is None else round(float(phase_seconds), 2),
            'opportunities': len(opps), 'eligibleSpans': [[round(a, 3), round(b, 3)] for a, b in spans],
            'abstain': abstain, 'dominantReason': dominant}


def _level_of(share):
    return 'high' if share < SI.TAINT_HIGH else ('medium' if share < SI.TAINT_MEDIUM else 'low')


def _position_level(conf):
    c = np.asarray(conf, float)
    c = c[np.isfinite(c)]
    if not len(c):
        return 'high'           # a result without registration (Metrica-shaped): nothing speaks against it
    m = float(np.median(c))
    return 'high' if m >= POSITION_HIGH else ('medium' if m >= POSITION_MEDIUM else 'low')


# ---------- the two sources of samples: the timeline (measurement tier) and the stored geometry (role tier) ----------
class _TimelineView:
    """The fields an opportunity is built from, read from a SituationTimeline."""

    def __init__(self, tl):
        self.tl = tl
        self.team = tl.team
        self.t, self.frame = np.asarray(tl.t, float), np.asarray(tl.frame, np.int64)
        self.phase, self.source = tl.phase, tl.phaseSource
        self.span, self.near = tl.phaseSpan, np.asarray(tl.nearChange, bool) | _coach_near(tl)
        self.third, self.channel, self.ball_ok = tl.ballThird, tl.ballChannel, tl.ballReliable
        self.fcn = tl.formationChangeNear
        cal = (tl.cd.calibration or {}).get('confidence') if tl.cd is not None else None
        self.cal = np.asarray(cal, float) if cal is not None and len(cal) == len(self.t) else np.full(len(self.t), np.nan)
        self.step = float(tl.step)

    def identity_level(self, pid, rows):
        lv = self.tl.identity_level(pid, rows)
        return lv or 'high'

    def position_level(self, rows):
        return _position_level(self.cal[rows])


class _StoredView:
    """The same fields read from obs['roleInputs']['detect'] (role_geometry())."""

    def __init__(self, geo):
        s = geo['samples']
        self.geo = geo
        self.team = geo['team']
        self.t = np.asarray(s['t'], float)
        self.frame = np.asarray(s['frame'], np.int64)
        n = len(self.t)
        self.phase = np.asarray(s['phase'], dtype='<U7')
        self.source = np.asarray(s['source'], dtype='<U12')
        self.span = np.asarray(s['span'], float)
        # within 2 s of a coach-marked change between the two teams' spans (situations.coachNear; fix round 1, S-16)
        self.near = np.asarray(s['near'], bool) if len(s.get('near') or ()) == n else np.zeros(n, bool)
        # seconds to a loose / contested ball (situations.contestS, fix round 3); NaN in an older file
        cs_ = s.get('contest') or ()
        self.contest = np.asarray([np.nan if v is None else v for v in cs_], float) if len(cs_) == n else \
            np.full(n, np.nan)
        self.third = np.asarray([v or '' for v in s['ballThird']], dtype='<U12')
        self.channel = np.asarray([v or '' for v in s['ballChannel']], dtype='<U6')
        self.ball_ok = np.asarray(s['ballOk'], bool)
        self.fcn = np.asarray(s['fcn'], bool)
        self.cal = np.asarray([np.nan if v is None else v for v in s['cal']], float)
        self.step = float(geo['step'])
        self.ids, self.oppIds = list(geo['ids']), list(geo['oppIds'])
        N, M = len(self.ids), len(self.oppIds)

        def mat(rows, k):
            a = np.asarray(rows, float) if k else np.zeros((0, n))
            return a.T.reshape(n, k) if k else np.zeros((n, 0))
        self.X, self.Y, self.vx = mat(s['X'], N), mat(s['Y'], N), mat(s['vx'], N)
        full = bool(s.get('oX'))
        self.OX = mat(s['oX'], M) if full else np.full((n, M), np.nan)
        self.OY = mat(s['oY'], M) if full else np.full((n, M), np.nan)
        self.oppSide = {'left': s['oppL'], 'right': s['oppR']}
        self.threatN = list(s['threatN'])
        self.ok = np.array([[c == '1' for c in b] for b in s['ok']], bool).reshape(n, N)
        self.taint = np.array([[c == '1' for c in b] for b in s['taint']], bool).reshape(n, N)
        self.named = self.ok & ~self.taint
        self.complete = np.asarray(s['complete'], bool)
        self.oppComplete = np.asarray(s['oppComplete'], bool)
        self.unplaced = np.asarray(s['unplaced'], int)
        self.oppUnplaced = np.asarray(s['oppUnplaced'], int)
        self.period = np.asarray(s['period'], int)
        self.ballX = np.asarray([np.nan if v is None else v for v in s['ballX']], float)
        self.ballY = np.asarray([np.nan if v is None else v for v in s['ballY']], float)
        self.viewL, self.viewR = list(s['viewL']), list(s['viewR'])
        self.col = {pid: j for j, pid in enumerate(self.ids)}

    def identity_level(self, pid, rows):
        j = self.col.get(pid)
        if j is None:
            return 'high'
        rows = np.asarray(rows, int)
        return _level_of(float(self.taint[rows, j].mean()) if len(rows) else 0.0)

    def position_level(self, rows):
        return _position_level(self.cal[rows])


def _phase_of(view, idx):
    src = str(view.source[idx[0]])
    state = str(view.phase[idx[0]])
    span = float(np.min(view.span[idx])) if state in ('own', 'opp') else None
    return {'state': state, 'source': src if src in ('possession-4', 'coach') else 'none',
            'estimated': src == 'possession-4', 'spanSeconds': None if span is None else round(span, 2),
            'nearChange': bool(np.any(view.near[idx]))}


def _opportunity(view, detector, idx, *, tier, relations, actors, quality, intent=None, roles=None, variant=None):
    """One Opportunity record (schema 'opportunity') over the sample indices idx of a view."""
    t0, t1 = round(float(view.t[idx[0]]), 3), round(float(view.t[idx[-1]]), 3)
    key = S.opportunity_key(detector, view.team, t0, t1, variant)
    zone = _modal(str(z) for z in view.third[idx])
    chan = _modal(str(c) for c in view.channel[idx])
    rel = {'t': [round(float(v), 3) for v in view.t[idx]]}
    rel.update(relations)
    q = {'identity': 'high', 'position': view.position_level(idx), 'visibility': 'complete', 'ball': 'not_needed',
         'samples': int(len(idx))}
    q.update(quality)
    opp = {'id': S.record_id('opp', key), 'key': key, 'detector': detector, 'team': view.team, 'tier': tier,
           't0': t0, 't1': t1, 'frame0': int(view.frame[idx[0]]), 'frame1': int(view.frame[idx[-1]]),
           'phase': _phase_of(view, idx),
           'ball': {'zone': zone, 'channel': chan, 'reliable': bool(np.mean(view.ball_ok[idx]) >= .5)},
           'actors': actors, 'roles': roles or {}, 'relations': rel, 'quality': q,
           'intentSignals': intent or {'recovering': False, 'interchange': None,
                                       'formationChangeNear': bool(np.any(view.fcn[idx]))},
           'abstain': None}
    if variant:
        opp['variant'] = variant
    return opp


def _actors(primary, supporting=(), opponents=(), carrier=None):
    prim = [p for p in primary if p]
    sup = [p for p in dict.fromkeys(supporting) if p and p not in prim]
    return {'primary': prim, 'supporting': sup[:3], 'opponents': [p for p in dict.fromkeys(opponents) if p][:3],
            'carrier': carrier or None}


def _id(ids, j):
    return None if j is None else str(ids[j])


def valid_id(pid):
    """An internal player id (schema PLAYER_ID: A-07, GK-1); display forms such as GK-A are not."""
    return isinstance(pid, str) and re.fullmatch(S.PLAYER_ID, pid) is not None


# ---------- measurement-tier detectors ----------
def _own_points(tl, s):
    ok = tl.posOk[s]
    return np.where(ok, tl.X[s], np.nan), np.where(ok, tl.Y[s], np.nan)


def _opp_points(tl, s):
    ok = tl.oppPosOk[s]
    return np.where(ok, tl.oppX[s], np.nan), np.where(ok, tl.oppY[s], np.nan)


def _d1(tl, view):
    """Pressure on the carrier and cover behind the first defender (DESIGN 4.1)."""
    S_ = len(view.t)
    L, W = tl.scale['lengthM'], tl.scale['widthM']
    kl = L/G.REF_LENGTH
    reason = _phase_reasons(view.phase, view.near, 'opp')
    _gate(reason, tl.regGate != '', 'registration_low')
    if not tl.scale['metresUsable']:
        _gate(reason, np.ones(S_, bool), 'scale_unknown')
    known = np.array([valid_id(str(c)) for c in tl.carrier], bool) if len(view.t) else np.zeros(0, bool)
    _gate(reason, ~known | (tl.carrierTeam != tl.other), 'no_carrier')
    _gate(reason, ~(np.isfinite(tl.carrierX) & np.isfinite(tl.carrierY)), 'out_of_view')
    view_r = G.COVER_VIEW_M*min(1.0, kl)
    vals = {k: [None]*S_ for k in ('d1', 'fd', 'fdGoalSide', 'closingMS', 'coverId', 'coverDist', 'coverAngle',
                                   'coverVisible', 'secondId', 'secondDist', 'goalSideShare', 'primaryNamed')}
    for s in np.flatnonzero(reason == ''):
        s = int(s)
        cx, cy = float(tl.carrierX[s]), float(tl.carrierY[s])
        X, Y = _own_points(tl, s)
        j, d1 = G.first_defender(cx, cy, X, Y)
        if j is None or not tl.local_context(s, cx, cy, d1, teams=('own',)):
            reason[s] = 'out_of_view'
            continue
        fx, fy = float(X[j]), float(Y[j])
        (cj, cd, ca), (sj, sd, _) = G.cover_candidate(fx, fy, X, Y, W, exclude=[j])
        vals['d1'][s] = _py(d1, 2)
        vals['fd'][s] = tl.ids[j]
        vals['fdGoalSide'][s] = bool(fx < cx)
        vals['closingMS'][s] = _py(G.closing_speed(fx, fy, float(tl.velX[s, j]), float(tl.velY[s, j]), cx, cy), 2)
        vals['coverId'][s] = _id(tl.ids, cj)
        vals['coverDist'][s] = _py(cd, 2)
        vals['coverAngle'][s] = _py(ca, 1)
        vals['coverVisible'][s] = bool(tl.disc_in_view(s, fx, fy, view_r))
        vals['secondId'][s] = _id(tl.ids, sj)
        vals['secondDist'][s] = _py(sd, 2)
        vals['goalSideShare'][s] = _py(G.goal_side_share(X, cx, exclude=[j]), 3)
        vals['primaryNamed'][s] = bool(tl.named_ok(tl.ids[j], s))
    keys = np.array([f'{c}|{p}' for c, p in zip(tl.carrier, view.source)], dtype=object)
    contest = _contest(tl)
    opps = []
    for idx in segments(view.t, reason == '', keys, min_s=MIN_S['pressure_cover'], step=view.step):
        rel = {k: [vals[k][i] for i in idx] for k in vals}
        rel['carrierX'] = [_py(tl.carrierX[i], 2) for i in idx]
        rel['carrierY'] = [_py(tl.carrierY[i], 2) for i in idx]
        rel['carrierZone'] = [G.zone_of(tl.carrierX[i], L) for i in idx]
        rel['complete'] = [bool(tl.complete[i]) for i in idx]
        rel['contestS'] = _contest_rel(contest, idx)
        fd = _modal(rel['fd'])
        carrier = str(tl.carrier[idx[0]])
        actors = _actors([fd], _ranked(rel['coverId'], exclude=(None, '', fd), limit=2), [carrier], carrier)
        opps.append(_opportunity(view, 'pressure_cover', idx, tier='measurement', relations=rel, actors=actors,
                                 quality={'identity': view.identity_level(fd, idx), 'visibility': 'disc'}))
    ab = _seconds(_reason_counts(reason), view.step)
    return opps, coverage('pressure_cover', 'measurement', opps, view.step, float((view.phase == 'opp').sum())*view.step,
                          ab)


D2_TOL_KEYS = ('weakGoalCount', 'weakLevelCount', 'centralGoalCount', 'weakRecovering')


def _ball_sigma(tl):
    """Per sample: the ball's pitch sigma in metres (0 where the result gives none)."""
    try:
        s = np.asarray(tl.ballSigma, float)
    except AttributeError:
        return np.zeros(len(tl.t))
    return np.where(np.isfinite(s), s, 0.0) if len(s) == len(tl.t) else np.zeros(len(tl.t))


def _d2(tl, view):
    """Defenders on the far side when the ball is wide (DESIGN 4.2), with the level tolerance of fix round 1 (critic
    B-5): besides the plain depth counts (weakCount, centralCount: X <= ball depth, unchanged) the relations count the
    far-half defenders goal-side by depth OR by the distance to the own goal centre with a tolerance of max(1 m, ball
    sigma + player sigma) (weakGoalCount, centralGoalCount), those level with the ball within it (weakLevelCount) and
    the far-half defenders not yet goal-side running back at 3 m/s or more (weakRecovering). On BC180 at 188.233 s the
    plain rule counts 0 goal-side and 0 central (the back four 0.01-2.3 m upfield of the ball at the box edge); the
    tolerant one counts the centre-back A-12 (0.01 m) central and goal-side by his goal distance."""
    S_ = len(view.t)
    W = tl.scale['widthM']
    reason = _phase_reasons(view.phase, view.near, 'opp')
    _gate(reason, tl.regGate != '', 'registration_low')
    _gate(reason, ~(tl.ballReliable & np.isfinite(tl.ballAX) & np.isfinite(tl.ballAY)), 'ball_unreliable')
    reason[(reason == '') & ~np.isin(tl.ballChannel, ('left', 'right'))] = NA
    _gate(reason, ~tl.complete, 'team_incomplete')
    vals = {k: [None]*S_ for k in ('ballSide', 'weakCount', 'openWidthM', 'centralCount', 'nearestWeak',
                                   'primaryNamed')+D2_TOL_KEYS}
    bsig = _ball_sigma(tl)
    for s in np.flatnonzero(reason == ''):
        s = int(s)
        X, Y = _own_points(tl, s)
        bx, by = float(tl.ballAX[s]), float(tl.ballAY[s])
        w = G.weak_side(bx, by, X, Y, W, tl.scale, side=str(tl.ballChannel[s]))
        vals['ballSide'][s] = w['ballSide']
        vals['weakCount'][s] = w['weakCount']
        vals['openWidthM'][s] = _py(w['openWidthM'], 2)
        vals['centralCount'][s] = w['centralCount']
        vals['nearestWeak'][s] = _id(tl.ids, w['nearestWeak'])
        vals['primaryNamed'][s] = bool(w['nearestWeak'] is not None and tl.named_ok(tl.ids[w['nearestWeak']], s))
        # The level tolerance (critic B-5): goal-side by depth or by the distance to the own goal centre, a player
        # within max(1 m, ball sigma + his sigma) of either is level (the sample is then unclear, never "nobody").
        psig = np.where(np.isfinite(tl.sigma[s]), tl.sigma[s], 0.0)
        tol = np.maximum(G.LEVEL_TOL_MIN_M, bsig[s]+psig)
        vx = np.where(tl.velOk[s], tl.velX[s], np.nan)
        f = G.far_side_protection(bx, by, X, Y, W, tl.scale, side=str(tl.ballChannel[s]), tol=tol, vx=vx)
        if f is not None:
            vals['weakGoalCount'][s], vals['weakLevelCount'][s] = f['weakGoal'], f['weakLevel']
            vals['centralGoalCount'][s], vals['weakRecovering'][s] = f['centralGoal'], f['weakRecovering']
    keys = np.array([f'{c}|{p}' for c, p in zip(tl.ballChannel, view.source)], dtype=object)
    contest = _contest(tl)
    opps = []
    for idx in segments(view.t, reason == '', keys, min_s=MIN_S['weak_side_balance'], step=view.step):
        rel = {k: [vals[k][i] for i in idx] for k in vals}
        rel['ballX'] = [_py(tl.ballAX[i], 2) for i in idx]
        rel['ballY'] = [_py(tl.ballAY[i], 2) for i in idx]
        rel['contestS'] = _contest_rel(contest, idx)
        prim = _modal(rel['nearestWeak'])
        carrier = _modal(str(c) for c in tl.carrier[idx] if valid_id(str(c)))
        actors = _actors([prim], (), [carrier] if carrier else [], carrier)
        opps.append(_opportunity(view, 'weak_side_balance', idx, tier='measurement', relations=rel, actors=actors,
                                 quality={'identity': view.identity_level(prim, idx) if prim else 'high',
                                          'ball': 'reliable'}))
    return opps, coverage('weak_side_balance', 'measurement', opps, view.step,
                          float((view.phase == 'opp').sum())*view.step, _seconds(_reason_counts(reason), view.step))


def _d4(tl, view):
    """Players behind the ball while we have it (DESIGN 4.4)."""
    S_ = len(view.t)
    L = tl.scale['lengthM']
    margin = G.BEHIND_BALL_M*L/G.REF_LENGTH
    reason = _phase_reasons(view.phase, view.near, 'own')
    _gate(reason, tl.regGate != '', 'registration_low')
    reason[(reason == '') & (view.span < SI.MIN_SHAPE_SPAN_S-1e-6)] = NA
    _gate(reason, ~(tl.ballReliable & np.isfinite(tl.ballAX)), 'ball_unreliable')
    with np.errstate(invalid='ignore'):
        reason[(reason == '') & (tl.ballAX < L/3.0)] = NA
    _gate(reason, ~tl.complete, 'team_incomplete')
    vals = {k: [None]*S_ for k in ('ballX', 'behindN', 'threatN', 'restMargin', 'behindSpreadM', 'deepest',
                                   'behindIds', 'oppComplete')}
    for s in np.flatnonzero(reason == ''):
        s = int(s)
        X, Y = _own_points(tl, s)
        bx = float(tl.ballAX[s])
        n, mask = G.behind_ball(X, bx, margin)
        vals['ballX'][s] = _py(bx, 2)
        vals['behindN'][s] = n
        ys = Y[mask]
        vals['behindSpreadM'][s] = _py(float(ys.max()-ys.min()), 2) if n >= 2 else None
        order = [j for j in np.argsort(np.where(mask, X, np.inf)) if mask[j]]
        vals['behindIds'][s] = '|'.join(tl.ids[j] for j in order[:3]) or None
        seen = np.isfinite(X)
        vals['deepest'][s] = tl.ids[int(np.argmin(np.where(seen, X, np.inf)))] if seen.any() else None
        vals['oppComplete'][s] = bool(tl.oppComplete[s])
        if tl.oppComplete[s]:
            OX, _ = _opp_points(tl, s)
            _, thr, mar = G.rest_margin(X, OX, bx, margin)
            vals['threatN'][s], vals['restMargin'][s] = thr, mar
    # a change of the ball's third starts a new opportunity: the margin is judged in the final third only
    keys = np.array([f'{z}|{p}' for z, p in zip(tl.ballThird, view.source)], dtype=object)
    contest = _contest(tl)
    opps = []
    for idx in segments(view.t, reason == '', keys, min_s=MIN_S['behind_ball'], step=view.step):
        rel = {k: [vals[k][i] for i in idx] for k in vals}
        rel['contestS'] = _contest_rel(contest, idx)
        deepest = _modal(rel['deepest'])
        behind = []
        for b in rel['behindIds']:
            behind += (b or '').split('|')
        actors = _actors([deepest], _ranked(behind, exclude=(None, '', deepest)))
        opps.append(_opportunity(view, 'behind_ball', idx, tier='measurement', relations=rel, actors=actors,
                                 quality={'identity': 'high', 'ball': 'reliable'}))
    return opps, coverage('behind_ball', 'measurement', opps, view.step, float((view.phase == 'own').sum())*view.step,
                          _seconds(_reason_counts(reason), view.step))


def gap_opponent_free_m(OX, OY, opp, AX, AY):
    """The marking of the opponents in a back-line gap (fix round 3, critic N1: the only review on the user's own
    runs cited a 'free' opponent whom our midfielder stood 2.8-2.9 m from, goal-side): for each opponent index in
    `opp`, the distance to the nearest of our seen outfield players goal-side of him (a smaller X; AX / AY all of our
    outfield players, attack frame), FREE_CAP_M when nobody of ours is goal-side of him; the largest of these (the
    least marked opponent in the gap), rounded to 0.01 m; None without an opponent in the gap or our positions."""
    if not opp or AX is None:
        return None
    AX, AY = np.asarray(AX, float), np.asarray(AY, float)
    seen = np.isfinite(AX) & np.isfinite(AY)
    worst = 0.0
    for k in opp:
        ox, oy = float(OX[k]), float(OY[k])
        with np.errstate(invalid='ignore'):
            mates = seen & (AX < ox)
        d = float(np.hypot(AX[mates]-ox, AY[mates]-oy).min()) if mates.any() else FREE_CAP_M
        worst = max(worst, min(d, FREE_CAP_M))
    return round(worst, 2)


def _gap_relations(X, Y, OX, OY, ids, ball_y, ball_ok, scale, AX=None, AY=None):
    """The back-line gap relations of one sample from the line's seen positions (X, Y aligned with ids); with our
    whole team's positions (AX, AY) also gapOppFreeM (gap_opponent_free_m)."""
    W = scale['widthM']
    gap, a, b, gy = G.lateral_gaps(Y)
    out = {'lateralGapM': _py(gap, 2), 'gapLeft': _id(ids, a), 'gapRight': _id(ids, b), 'gapY': _py(gy, 2),
           'gapBallSide': None, 'opponentInGap': False, 'gapOpponents': 0, 'ballSide': None, 'gapPrimary': None,
           'gapOppFreeM': None}
    if a is None:
        return out
    opp = G.opponent_in_gap(OX, OY, (float(X[a]), float(Y[a])), (float(X[b]), float(Y[b])), scale)
    out['opponentInGap'] = bool(opp)
    out['gapOpponents'] = len(opp)
    out['gapOppFreeM'] = gap_opponent_free_m(OX, OY, opp, AX, AY)
    prim = a
    if ball_ok and math.isfinite(ball_y):
        side = 'left' if ball_y < W/2.0 else 'right'
        out['ballSide'] = side
        out['gapBallSide'] = bool((gy < W/2.0) == (side == 'left'))
        prim = a if abs(Y[a]-ball_y) <= abs(Y[b]-ball_y) else b
    out['gapPrimary'] = _id(ids, prim)
    out['gapOther'] = _id(ids, b if prim == a else a)
    return out


D5_KEYS = ('lineStdM', 'lineSize', 'lateralGapM', 'gapLeft', 'gapRight', 'gapY', 'gapBallSide', 'opponentInGap',
           'gapOpponents', 'ballSide', 'dmGapM', 'dmGapState', 'lineHeightM', 'blockHeight', 'primaryNamed',
           'dmGapOpponents', 'dmGapFree', 'gapOppFreeM')
DM_FREE_M = 4.0                 # m: an opponent between the lines with nobody of ours this close is free to receive [P]
DM_REACH_M = 25.0               # m x L/105: ... and within this of the ball, in reach of a pass now [P]
FREE_CAP_M = 50.0               # m: gapOppFreeM of an opponent with nobody of ours goal-side of him (fix round 3)


def between_lines(OX, OY, back_x, mid_x, ball_x, ball_y, width, X=None, Y=None, reach_m=DM_REACH_M,
                  free_m=DM_FREE_M):
    """(between, free): opponents between our back line (depth back_x) and our midfield line (mid_x) on the ball's
    lateral half, and those of them FREE to receive (nobody of ours, positions X / Y, within free_m) within reach_m of
    the ball: the condition of the midfield-to-defence gap criticism (fix round 1, critic B-7: DESIGN 4.5's own basis
    says a plain D-M gap above 18 m "fires 6-11 times per 10 min on professional play and is not a mistake signal"; on
    Metrica game 1 (full tracking, 3-minute clips) an opponent stood between the lines on the ball side on 760 of the
    825 samples with a gap above 18 m, mostly with a high block, so only a free one in reach of the ball is the
    "receiving or able to receive" of the lateral-gap rule). (None, None) when the ball or a line is unknown."""
    if not (math.isfinite(back_x) and math.isfinite(mid_x) and math.isfinite(ball_x) and math.isfinite(ball_y)):
        return None, None
    OX, OY = np.asarray(OX, float), np.asarray(OY, float)
    with np.errstate(invalid='ignore'):
        side = (OY < width/2.0) if ball_y < width/2.0 else (OY >= width/2.0)
        m = np.isfinite(OX) & np.isfinite(OY) & (OX > back_x) & (OX < mid_x) & side
    free = 0
    if X is not None and m.any():
        X, Y = np.asarray(X, float), np.asarray(Y, float)
        seen = np.isfinite(X) & np.isfinite(Y)
        for k in np.flatnonzero(m):
            near = np.hypot(X[seen]-OX[k], Y[seen]-OY[k]).min() if seen.any() else np.inf
            if near >= free_m and math.hypot(OX[k]-ball_x, OY[k]-ball_y) <= reach_m:
                free += 1
    return int(m.sum()), int(free)


def _d5(tl, view):
    """Back-line evenness, lateral gaps and the D-M gap while the other team has the ball (DESIGN 4.5), 11v11 from the
    series' recognised defensive line. Small formats: detect_roles (relative back block)."""
    S_ = len(view.t)
    L, W = tl.scale['lengthM'], tl.scale['widthM']
    me = tl.series
    rl = float((tl.cd.pitch or {}).get('length') or 105.0)
    k_len = L/rl                                   # series metres (result scale) -> coaching scale
    reason = _phase_reasons(view.phase, view.near, 'opp')
    _gate(reason, tl.regGate != '', 'registration_low')
    reason[(reason == '') & (view.span < SI.MIN_SHAPE_SPAN_S-1e-6)] = NA
    line_ok = np.asarray(me.eligible['lineStd'], bool)
    line_reason = np.asarray(me.reasons['lineStd'])
    for s in np.flatnonzero((reason == '') & ~line_ok):
        reason[s] = SERIES_LINE_REASON.get(str(line_reason[s]), 'team_incomplete')
    vals = {k: [None]*S_ for k in D5_KEYS+('gapPrimary', 'gapOther')}
    dm_ok = np.asarray(me.eligible['gapDM'], bool)
    dm_reason = np.asarray(me.reasons['gapDM'])
    for s in np.flatnonzero(reason == ''):
        s = int(s)
        cols = [tl.col[p] for p in me.lineDefenders[s] if p in tl.col]
        if len(cols) < 3 or not all(tl.posOk[s, c] for c in cols):
            reason[s] = 'team_incomplete'
            continue
        X, Y = tl.X[s, cols], tl.Y[s, cols]
        OX, OY = _opp_points(tl, s)
        AX, AY = _own_points(tl, s)
        g = _gap_relations(X, Y, OX, OY, [tl.ids[c] for c in cols], float(tl.ballAY[s]), bool(tl.ballReliable[s]),
                           tl.scale, AX, AY)
        for k, v in g.items():
            vals[k][s] = v
        vals['lineStdM'][s] = _py(float(me.lineStd[s])*k_len, 2)
        vals['lineSize'][s] = len(cols)
        vals['dmGapM'][s] = _py(float(me.gapDM[s])*k_len, 2) if dm_ok[s] else None
        if dm_ok[s] and tl.ballReliable[s]:
            vals['dmGapOpponents'][s], vals['dmGapFree'][s] = between_lines(
                OX, OY, float(me.DX[s])*k_len, float(me.M1X[s])*k_len, float(tl.ballAX[s]), float(tl.ballAY[s]), W,
                AX, AY, reach_m=DM_REACH_M*L/G.REF_LENGTH)
        vals['dmGapState'][s] = '' if dm_ok[s] else ('units_interleaved' if dm_reason[s] == 'units_interleaved'
                                                    else 'not_measured')
        h = float(me.lineHeight[s])*k_len if np.isfinite(me.lineHeight[s]) else math.nan
        vals['lineHeightM'][s] = _py(h, 2)
        vals['blockHeight'][s] = G.block_height(h, L)
        vals['primaryNamed'][s] = bool(g['gapPrimary'] is not None and tl.named_ok(g['gapPrimary'], s))
    # a change of the ball's half starts a new opportunity: a gap matters on the ball's side (DESIGN 4.5)
    keys = np.array([f'{b}|{p}' for b, p in zip(vals['ballSide'], view.source)], dtype=object)
    contest = _contest(tl)
    opps = []
    for idx in segments(view.t, reason == '', keys, min_s=MIN_S['defensive_spacing'], step=view.step):
        rel = {k: [vals[k][i] for i in idx] for k in D5_KEYS}
        rel['contestS'] = _contest_rel(contest, idx)
        prim = _modal(vals['gapPrimary'][i] for i in idx)
        other = _modal(vals['gapOther'][i] for i in idx if vals['gapPrimary'][i] == prim)
        actors = _actors([prim], [other])
        opps.append(_opportunity(view, 'defensive_spacing', idx, tier='measurement', relations=rel, actors=actors,
                                 quality={'identity': view.identity_level(prim, idx) if prim else 'high'}))
    return opps, coverage('defensive_spacing', 'measurement', opps, view.step,
                          float((view.phase == 'opp').sum())*view.step, _seconds(_reason_counts(reason), view.step))


D6_SPACE_KEYS = ('oppLastLineX', 'availableDepthM', 'heightGapM', 'depthAvailShare', 'deepN')
MIN_AVAILABLE_M = 5.0           # m: below this the space to the opponents' last line is no length to judge depth by
AVAIL_SHARE_CAP = 2.0           # a team stretched beyond the opponents' last line is capped here (a forward offside)


def _d6(tl, view):
    """Height, width and depth while we have the ball (DESIGN 4.6); describe-only samples without any phase. Besides
    the field shares (widthShare, depthShare: unchanged) the relations measure the space actually available (fix round
    1, critic B-7 / COACH B4: against a low block Barcelona's forwards stood on Espanyol's last line with the
    centre-backs at halfway, about 30 m of available length, and the field share .25 read as "short from back to
    front"): oppLastLineX (the opponents' deepest outfield player, attack frame), availableDepthM (from our deepest
    outfield player to it), heightGapM (from our highest to it) and depthAvailShare (our depth / the available length,
    capped at 2), all only where the opponents are complete, and deepN (our players 10 m x L/105 or more goal-side of
    a reliable ball; context only). Measured on Metrica game 1 (full tracking, own possession, 15-minute clips):
    depthAvailShare p10 / p50 .87 / .95 (camera .86 / .95), below .75 on 2.3% (2.2%) of the samples, while the field
    share is below .30 on 30% of them; heightGapM p10 / p50 / p90 -1.2 / 1.6 / 4.6 m. On BC180 Barcelona (field share
    .20-.31 against Espanyol's low block) depthAvailShare was about 1.0."""
    S_ = len(view.t)
    L, W = tl.scale['lengthM'], tl.scale['widthM']
    no_phase = not np.isin(view.phase, ('own', 'opp')).any()
    if no_phase:
        reason = np.full(S_, NA, dtype=REASON_DTYPE)
        reason[view.phase == 'unknown'] = ''
    else:
        reason = _phase_reasons(view.phase, view.near, 'own')
        reason[(reason == '') & (view.span < SI.MIN_SHAPE_SPAN_S-1e-6)] = NA
    _gate(reason, tl.regGate != '', 'registration_low')
    _gate(reason, ~tl.complete, 'team_incomplete')
    vals = {k: [None]*S_ for k in ('widthM', 'depthM', 'widthShare', 'depthShare', 'channelsOccupied',
                                   'bandsOccupied')+D6_SPACE_KEYS}
    for s in np.flatnonzero(reason == ''):
        s = int(s)
        X, Y = _own_points(tl, s)
        w, d = G.team_extent(X, Y)
        vals['widthM'][s], vals['depthM'][s] = _py(w, 2), _py(d, 2)
        vals['widthShare'][s], vals['depthShare'][s] = _py(w/W, 3), _py(d/L, 3)
        vals['channelsOccupied'][s] = G.channels_occupied(Y, W)
        vals['bandsOccupied'][s] = G.bands_occupied(X, L)
        # The space actually available (critic B-7): from our deepest outfield player to the opponents' last line
        # (their deepest outfield player), measured only with the opponents complete (nobody unseen behind them).
        if tl.oppComplete[s] and np.isfinite(d):
            OX, _ = _opp_points(tl, s)
            seen = np.isfinite(X)
            if seen.any() and np.isfinite(OX).any():
                last, deep, high = float(np.nanmax(OX)), float(np.min(X[seen])), float(np.max(X[seen]))
                avail = last-deep
                vals['oppLastLineX'][s], vals['heightGapM'][s] = _py(last, 2), _py(last-high, 2)
                vals['availableDepthM'][s] = _py(avail, 2)
                if avail >= MIN_AVAILABLE_M:
                    vals['depthAvailShare'][s] = _py(min(d/avail, AVAIL_SHARE_CAP), 3)
        if tl.ballReliable[s] and np.isfinite(tl.ballAX[s]):
            vals['deepN'][s] = G.behind_ball(X, float(tl.ballAX[s]), G.BEHIND_BALL_M*L/G.REF_LENGTH)[0]
    keys = np.array(list(view.source), dtype=object)
    contest = _contest(tl)
    opps = []
    for idx in segments(view.t, reason == '', keys, min_s=MIN_S['attacking_shape'], step=view.step):
        rel = {k: [vals[k][i] for i in idx] for k in vals}
        rel['contestS'] = _contest_rel(contest, idx)
        opps.append(_opportunity(view, 'attacking_shape', idx, tier='measurement', relations=rel, actors=_actors([]),
                                 quality={}, variant='unknown' if no_phase else None))
    # Without any phase every unknown sample is lost for a verdict (the describe-only opportunities included).
    ab = {'phase_unknown': int((view.phase == 'unknown').sum())} if no_phase else _reason_counts(reason)
    return opps, coverage('attacking_shape', 'measurement', opps, view.step,
                          float((view.phase == 'own').sum())*view.step, _seconds(ab, view.step))


# ---------- role-tier inputs (obs['roleInputs']['detect']) ----------
def _bits(row):
    return ''.join('1' if b else '0' for b in row)


def _view_mask(tl, s, side):
    """Bit k: the outer third of `side` is in the camera's view at depth k x VIEW_STATION_M (both corners, 1 m inside
    the touchline and on the third's inner line); None when the result has no view field (all counts as in view)."""
    v = tl.view[s] if 0 <= s < len(tl.view) else None
    if v is None:
        return None
    if v is False:
        return 0
    L, W = tl.scale['lengthM'], tl.scale['widthM']
    xs = np.arange(0.0, L+1e-6, VIEW_STATION_M)
    y0, y1 = ((VIEW_EDGE_M, W*G.WIDE_SHARE) if side == 'left' else (W*(1.0-G.WIDE_SHARE), W-VIEW_EDGE_M))
    X = np.r_[xs, xs]
    Y = np.r_[np.full(len(xs), y0), np.full(len(xs), y1)]
    p = tl.to_pitch(X, Y, s)
    if p is None:
        return 0
    from . import series as cs
    inside = cs._disc_inside(v, np.asarray(p[0], float), np.asarray(p[1], float), np.zeros(len(X)), L, W)
    both = inside[:len(xs)] & inside[len(xs):]
    return int(sum(1 << k for k in np.flatnonzero(both)))


def channel_in_view(mask, x0, x1):
    """Is the depth range [x0, x1] of a channel in view by a stored station mask (None = no view field = yes)? Every
    station inside the range must be in view, and the nearest station to each end."""
    if mask is None:
        return True
    lo, hi = sorted((float(x0), float(x1)))
    k0, k1 = int(math.floor(lo/VIEW_STATION_M)), int(math.ceil(hi/VIEW_STATION_M))
    for k in range(max(0, k0), k1+1):
        if not (mask >> k) & 1:
            return False
    return True


def role_geometry(tl):
    """obs['roleInputs']['detect'] (DETECT_VERSION): the samples the role tier's detectors read (module docstring): our
    phase in spans of 3 s or more on frames that pass the registration gates (D3), and on small formats the other
    team's too (D5), with the seconds lost before them per reason."""
    L = tl.scale['lengthM']
    small = tl.format['players'] != 11
    phase, near, span, step = tl.phase, tl.nearChange, tl.phaseSpan, float(tl.step)
    long_span = span >= SI.MIN_SHAPE_SPAN_S-1e-6
    reg_ok = tl.regGate == ''
    keep = ((phase == 'own') | ((phase == 'opp') & small)) & long_span & reg_ok
    idx = np.flatnonzero(keep)
    coach_near = _coach_near(tl)
    unk = phase == 'unknown'
    lost = {}
    for st in ('own', 'opp'):
        lost[st] = _seconds({'phase_unknown': int((unk & ~near).sum()), 'near_change': int((unk & near).sum()),
                             'registration_low': int(((phase == st) & long_span & ~reg_ok).sum())}, step)
    cal = (tl.cd.calibration or {}).get('confidence') if tl.cd is not None else None
    cal = np.asarray(cal, float) if cal is not None and len(cal) == len(tl.t) else np.full(len(tl.t), np.nan)
    X = np.where(tl.posOk, tl.X, np.nan)
    Y = np.where(tl.posOk, tl.Y, np.nan)
    OX = np.where(tl.oppPosOk, tl.oppX, np.nan)
    OY = np.where(tl.oppPosOk, tl.oppY, np.nan)
    medians = {}
    period = np.asarray(tl.formationPeriod, int) if tl.format['players'] == 11 else np.full(len(tl.t), -1, int)
    for j, pid in enumerate(tl.ids):
        m = {}
        ok = np.isfinite(Y[:, j])
        if ok.sum() >= MEDIAN_MIN_SAMPLES:
            m['-1'] = round(float(np.median(Y[ok, j])), 2)
        for p in np.unique(period[period >= 0]):
            sel = ok & (period == p)
            if sel.sum() >= MEDIAN_MIN_SAMPLES:
                m[str(int(p))] = round(float(np.median(Y[sel, j])), 2)
        if m:
            medians[pid] = m

    def col(a, j, digits=1):
        return [_py(a[i, j], digits) for i in idx]
    # The opponents D3 reads (role-free parts precomputed, so the large-format file does not carry every opponent):
    # those in each outer third of our own half ([id, X, Y]: tests (f) and the channel), and the rest-defence threat
    # count (opponents goal-side of the ball by the behind-the-ball margin, both teams complete, ball reliable).
    margin = G.BEHIND_BALL_M*L/G.REF_LENGTH
    side_opp = {'left': [], 'right': []}
    threat = []
    for i in idx:
        for side in ('left', 'right'):
            with np.errstate(invalid='ignore'):
                m = np.isfinite(OX[i]) & G.in_outer_third(OY[i], side, tl.scale['widthM']) & (OX[i] < L/2.0)
            side_opp[side].append([[tl.oppIds[k], round(float(OX[i, k]), 1), round(float(OY[i, k]), 1)]
                                   for k in np.flatnonzero(m)])
        ok_t = tl.complete[i] and tl.oppComplete[i] and tl.ballReliable[i] and np.isfinite(tl.ballAX[i])
        threat.append(G.behind_ball(OX[i], float(tl.ballAX[i]), margin)[0] if ok_t else None)
    samples = {
        't': [round(float(tl.t[i]), 3) for i in idx], 'frame': [int(tl.frame[i]) for i in idx],
        'side': [str(tl.side[i]) for i in idx],     # the side defended: attack frame -> pitch (EVIDENCE scenes)
        'phase': [str(phase[i]) for i in idx], 'source': [str(tl.phaseSource[i]) for i in idx],
        'span': [round(float(span[i]), 2) for i in idx],
        'complete': [bool(tl.complete[i]) for i in idx], 'oppComplete': [bool(tl.oppComplete[i]) for i in idx],
        'unplaced': [int(tl.unplaced[i]) for i in idx], 'oppUnplaced': [int(tl.oppUnplaced[i]) for i in idx],
        'fcn': [bool(tl.formationChangeNear[i]) for i in idx], 'period': [int(period[i]) for i in idx],
        'near': [bool(v) for v in coach_near[idx]],
        'contest': _contest_rel(_contest(tl), idx),     # fix round 3 (situations.contestS); older files lack it
        'ballOk': [bool(tl.ballReliable[i]) for i in idx],
        'ballX': [_py(tl.ballAX[i], 2) if tl.ballReliable[i] else None for i in idx],
        'ballY': [_py(tl.ballAY[i], 2) if tl.ballReliable[i] else None for i in idx],
        'ballThird': [str(tl.ballThird[i]) or None for i in idx],
        'ballChannel': [str(tl.ballChannel[i]) or None for i in idx],
        'cal': [_py(cal[i], 3) for i in idx],
        'viewL': [None if tl.complete[i] else _view_mask(tl, int(i), 'left') for i in idx],
        'viewR': [None if tl.complete[i] else _view_mask(tl, int(i), 'right') for i in idx],
        'ok': [_bits(tl.posOk[i]) for i in idx], 'taint': [_bits(tl.tainted[i]) for i in idx],
        'X': [col(X, j) for j in range(len(tl.ids))], 'Y': [col(Y, j) for j in range(len(tl.ids))],
        'vx': [col(np.where(tl.velOk, tl.velX, np.nan), j, 2) for j in range(len(tl.ids))],
        'threatN': threat, 'oppL': side_opp['left'], 'oppR': side_opp['right'],
        'oX': [col(OX, j) for j in range(len(tl.oppIds))] if small else [],
        'oY': [col(OY, j) for j in range(len(tl.oppIds))] if small else [],
    }
    return {'version': DETECT_VERSION, 'team': tl.team, 'format': tl.format['name'],
            'players': int(tl.format['players']), 'outfield': int(tl.format['outfield']), 'scale': dict(tl.scale),
            'step': step, 'fps': float(tl.fps), 'ids': list(tl.ids), 'oppIds': list(tl.oppIds),
            'phaseSeconds': {'own': round(float((phase == 'own').sum())*step, 2),
                             'opp': round(float((phase == 'opp').sum())*step, 2)},
            'lost': lost, 'medianY': medians, 'stationM': VIEW_STATION_M, 'samples': samples,
            'lengthM': L}


MEASUREMENT_DETECTORS = (('pressure_cover', _d1), ('weak_side_balance', _d2), ('behind_ball', _d4),
                         ('defensive_spacing', _d5), ('attacking_shape', _d6))


def detect(timeline):
    """The measurement tier's 'detect' stage (INTERFACES.md LEAD 4): {'opportunities': [Opportunity, tier
    measurement], 'detectors': {name: DetectorCoverage}, 'roleInputs': {'context': CONTEXT's role inputs, 'detect':
    role_geometry()}} of one team's SituationTimeline. D5 is here only for 11v11 (small formats: detect_roles); D3 is
    always in the role tier."""
    tl = timeline
    view = _TimelineView(tl)
    opps, cov = [], {}
    for name, fn in MEASUREMENT_DETECTORS:
        if name == 'defensive_spacing' and tl.format['players'] != 11:
            continue
        found, c = fn(tl, view)
        opps += found
        cov[name] = c
    opps.sort(key=lambda o: (o['t0'], S.DETECTORS.index(o['detector'])))
    return {'opportunities': opps, 'detectors': cov,
            'roleInputs': {'context': tl.role_inputs(), 'detect': role_geometry(tl)}}


# ---------- role tier ----------
SOURCE_RANK = {s: k for k, s in enumerate(('coach', 'stable', 'carried', 'window', 'relative', 'none'))}


def roles_at(rolebook, ids, t):
    """(role, source, confidence) arrays (S x N) of the players `ids` at the times t from a RoleBook: the most
    trusted record covering each time (roles.RoleIndex.at, vectorized); '' where none covers it."""
    t = np.asarray(t, float)
    n, N = len(t), len(ids)
    role = np.full((n, N), '', dtype='<U10')
    src = np.full((n, N), '', dtype='<U8')
    conf = np.full((n, N), '', dtype='<U6')
    rank = np.full((n, N), 99, int)
    col = {pid: j for j, pid in enumerate(ids)}
    for r in rolebook.get('records') or ():
        j = col.get(r['id'])
        if j is None or not r.get('role'):
            continue
        a, b = np.searchsorted(t, r['t0']-1e-6, 'left'), np.searchsorted(t, r['t1']+1e-6, 'right')
        if a >= b:
            continue
        k = SOURCE_RANK.get(r['source'], 99)
        better = rank[a:b, j] > k
        rows = np.arange(a, b)[better]
        role[rows, j] = r['role']
        src[rows, j] = r['source']
        conf[rows, j] = r.get('confidence') or ''
        rank[rows, j] = k
    return role, src, conf


def _slot(role):
    """(block, k, n) of a relative slot 'back-2/3', None otherwise."""
    if not role or '-' not in role or '/' not in role:
        return None
    block, rest = role.split('-', 1)
    k, n = rest.split('/')
    return block, int(k), int(n)


def _relative_ok(src, conf):
    return src == 'coach' or (src == 'relative' and conf in ('medium', 'high'))


def _wide_and_centre(view, s, R, Sr, C, small):
    """At sample s: ({side: wide defender column or None}, centre-back columns, holding-midfielder columns, reason per
    side when the wide defender is unknown, the back block's size)."""
    N = R.shape[1]
    wd, why = {'left': None, 'right': None}, {'left': None, 'right': None}
    if small:
        back = []
        weak = False
        for j in range(N):
            sl = _slot(R[s, j])
            if sl and sl[0] == 'back':
                if _relative_ok(Sr[s, j], C[s, j]):
                    back.append((j, sl[1], sl[2]))
                else:
                    weak = True
        if not back:
            return wd, [], [], {'left': 'role_unconfirmed', 'right': 'role_unconfirmed'}, 0
        n = max(x[2] for x in back)
        if n < 3:
            return wd, [], [], {'left': 'format_not_supported', 'right': 'format_not_supported'}, n
        for j, k, _ in back:
            if k == 1:
                wd['left'] = j
            elif k == n:
                wd['right'] = j
        cbs = [j for j, k, _ in back if 1 < k < n]
        for side in wd:
            if wd[side] is None:
                why[side] = 'role_unconfirmed'
        return wd, cbs, [], why, n
    cbs = [j for j in range(N) if R[s, j] in CB_ROLES]
    dms = [j for j in range(N) if R[s, j] in DM_ROLES and Sr[s, j] in ('coach', 'stable')]
    for side, names in WIDE_ROLES.items():
        cand = [j for j in range(N) if R[s, j] in names]
        good = [j for j in cand if Sr[s, j] in CONFIDENT]
        if good:
            good.sort(key=lambda j: (SOURCE_RANK.get(Sr[s, j], 99), -(view.X[s, j] if np.isfinite(view.X[s, j])
                                                                       else -1e9)))
            wd[side] = good[0]
        else:
            why[side] = 'role_unconfirmed'
    return wd, cbs, dms, why, len(cbs)+2


def _role_ref(R, Sr, C, s, j):
    role = str(R[s, j]) or None
    src = str(Sr[s, j]) or 'none'
    return {'role': role if src != 'none' else None, 'source': src, 'confidence': (str(C[s, j]) or None)
            if src != 'none' else None}


D3_KEYS = ('advanceM', 'beyondHalfM', 'channelOccupied', 'channelBy', 'dmDropped', 'cbAdjusted', 'nearCb',
           'nearCbShiftM', 'behindN', 'interchange', 'opponentsInChannel', 'threatN', 'restMargin', 'dmCbSpacingM',
           'wdVelX', 'complete', 'relaxed', 'primaryNamed', 'wdSource', 'wdConfidence', 'ballX', 'interchangeCover',
           'interchangeBy', 'contestS')


def interchange_cover(inter, wd_id, t, X, Y, ok, col, j, side, width, behind_m, horizon=INTERCHANGE_S):
    """Test (e) with geometry (fix round 1, critic S-4): (covered, the incoming player's id) at one sample. An
    interchange record within `horizon` s in which the wide defender LEFT his slot ('out') counts only when the player
    who took it ('in') stands, at this sample, seen, on the defender's lateral half and at least `behind_m` behind
    him: the vacated slot re-occupied, not a template label swapped (on Metrica full tracking the role book holds
    40-101 slot swaps per team and 10 minutes, and the STATS verifier's overlap plants at 839 and 938 s were 'covered'
    by a label swap alone with the channel empty)."""
    for rec in inter:
        if rec.get('out') != wd_id or abs(float(rec['t'])-t) > horizon+1e-9:
            continue
        k = col.get(rec.get('in'))
        if k is None or k == j or not ok[k] or not (np.isfinite(X[k]) and np.isfinite(Y[k])):
            continue
        lateral = Y[k] <= width/2.0 if side == 'left' else Y[k] >= width/2.0
        if lateral and X[k] <= X[j]-behind_m:
            return True, str(rec['in'])
    return False, None


def _d3(view, book):
    """Wide-defender advances and their cover tests (DESIGN 4.3) on the stored samples."""
    geo = view.geo
    n = len(view.t)
    L, W = geo['scale']['lengthM'], geo['scale']['widthM']
    kl = L/G.REF_LENGTH
    players = int(geo['players'])
    small = players != 11
    step = view.step
    own = view.phase == 'own'
    lost = dict(geo['lost'].get('own') or {})
    if players in (4, 5):
        lost['format_not_supported'] = lost.get('format_not_supported', 0.0)+round(float(own.sum())*step, 2)
        return [], coverage('wide_defender_advance', 'role', [], step, geo['phaseSeconds']['own'], lost)
    R, Sr, C = roles_at(book, view.ids, view.t)
    inter = list(book.get('interchanges') or ())
    vals = {side: {k: [None]*n for k in D3_KEYS} for side in ('left', 'right')}
    reason = {side: np.full(n, NA, dtype=REASON_DTYPE) for side in ('left', 'right')}
    wd_of = {side: np.full(n, '', dtype=object) for side in ('left', 'right')}
    supp = {side: [None]*n for side in ('left', 'right')}
    opp_ids = {side: [()]*n for side in ('left', 'right')}
    adv_m, behind_m = G.ADVANCED_M*kl, G.BEHIND_BALL_M*kl
    chan_m = G.CHANNEL_BEHIND_M*kl
    for s in np.flatnonzero(own):
        s = int(s)
        if view.near[s]:
            # within 2 s of a coach-marked change between the two teams' spans (fix round 1, critic S-16): like the
            # engine's change windows, never an opportunity
            reason['left'][s] = reason['right'][s] = 'near_change'
            continue
        wd, cbs, dms, why, _ = _wide_and_centre(view, s, R, Sr, C, small)
        X, Y = view.X[s], view.Y[s]
        ok = view.ok[s]
        for side in ('left', 'right'):
            j = wd[side]
            if j is None:
                reason[side][s] = why[side] or 'role_unconfirmed'
                continue
            if not cbs:
                reason[side][s] = 'role_unconfirmed'
                continue
            if not ok[j] or not all(ok[c] for c in cbs):
                reason[side][s] = 'out_of_view'
                continue
            cbx = float(np.mean(X[cbs]))
            adv, half = float(X[j]-cbx), float(X[j]-L/2.0)
            if not (adv >= adv_m or (small and half >= G.ADVANCED_HALFWAY_SHARE*L)):
                continue                                    # not advanced: not relevant (NA)
            if np.isfinite(view.vx[s, j]) and view.vx[s, j] <= -G.RECOVERING_MS:
                reason[side][s] = 'recovering'
                continue
            relaxed = False
            if not view.complete[s]:
                x0, x1, _, _ = G.vacated_channel(cbx, float(X[j]), side, W, chan_m)
                mask = (view.viewL if side == 'left' else view.viewR)[s]
                relaxed = view.unplaced[s] == 0 and view.oppUnplaced[s] == 0 and channel_in_view(mask, x0, x1)
                if not relaxed:
                    reason[side][s] = 'team_incomplete'
                    continue
            reason[side][s] = ''
            wd_of[side][s] = view.ids[j]
            v = vals[side]
            occ, by = G.channel_occupied(X, Y, j, side, W, chan_m)
            v['advanceM'][s], v['beyondHalfM'][s] = _py(adv, 2), _py(half, 2)
            v['channelOccupied'][s], v['channelBy'][s] = bool(occ), _id(view.ids, by)
            seen_dm = [d for d in dms if ok[d]]
            v['dmDropped'][s] = bool(any(X[d] <= cbx+G.DM_DROP_M for d in seen_dm)) if seen_dm else None
            near = min(cbs, key=lambda c: Y[c]) if side == 'left' else max(cbs, key=lambda c: Y[c])
            med = (geo['medianY'].get(view.ids[near]) or {})
            m = med.get(str(int(view.period[s])), med.get('-1'))
            shift = G.cb_shift(Y[near], m if m is not None else math.nan, side)
            v['nearCb'][s] = view.ids[near]
            v['nearCbShiftM'][s] = _py(shift, 2)
            v['cbAdjusted'][s] = None if not math.isfinite(shift) else bool(shift >= G.CB_SHIFT_M)
            if seen_dm:
                v['dmCbSpacingM'][s] = _py(min(math.hypot(X[d]-X[near], Y[d]-Y[near]) for d in seen_dm), 2)
            bx = view.ballX[s]
            if view.complete[s] and math.isfinite(bx):
                v['behindN'][s] = G.behind_ball(X, bx, behind_m)[0]
                if view.threatN[s] is not None:
                    v['threatN'][s] = int(view.threatN[s])
                    v['restMargin'][s] = v['behindN'][s]-v['threatN'][s]
            near_opp = view.oppSide[side][s] or []
            if near_opp:
                OX = np.array([o[1] for o in near_opp], float)
                OY = np.array([o[2] for o in near_opp], float)
                ch = G.opponents_in_channel(OX, OY, float(X[j]), side, W, L)
            else:
                ch = []
            v['opponentsInChannel'][s] = len(ch)
            opp_ids[side][s] = tuple(near_opp[k][0] for k in ch)
            t = float(view.t[s])
            v['interchange'][s] = bool(any(abs(i['t']-t) <= INTERCHANGE_S and view.ids[j] in (i['out'], i['in'])
                                           for i in inter))
            v['interchangeCover'][s], v['interchangeBy'][s] = interchange_cover(inter, view.ids[j], t, X, Y, ok,
                                                                                view.col, j, side, W, chan_m)
            v['ballX'][s] = _py(bx, 2) if math.isfinite(bx) else None
            v['wdVelX'][s] = _py(view.vx[s, j], 2)
            v['complete'][s], v['relaxed'][s] = bool(view.complete[s]), bool(relaxed)
            v['contestS'][s] = _py(view.contest[s], 2)
            v['primaryNamed'][s] = bool(view.named[s, j])
            v['wdSource'][s], v['wdConfidence'][s] = str(Sr[s, j]) or None, str(C[s, j]) or None
            supp[side][s] = (view.ids[near], v['channelBy'][s],
                             view.ids[seen_dm[0]] if seen_dm else None)
    opps = []
    for side in ('left', 'right'):
        keys = np.array([f'{w}|{p}' for w, p in zip(wd_of[side], view.source)], dtype=object)
        for idx in segments(view.t, reason[side] == '', keys, min_s=MIN_S['wide_defender_advance'],
                            max_s=D3_MAX_S, step=step):
            rel = {k: [vals[side][k][i] for i in idx] for k in D3_KEYS}
            wd = str(wd_of[side][idx[0]])
            near = _modal(supp[side][i][0] for i in idx)
            by = _modal(supp[side][i][1] for i in idx)
            dm = _modal(supp[side][i][2] for i in idx)
            chan_opp = []
            for i in idx:
                chan_opp += list(opp_ids[side][i])
            actors = _actors([wd], [near, by, dm], _ranked(chan_opp, limit=2))
            mid = int(idx[len(idx)//2])
            refs = {}
            for pid in actors['primary']+actors['supporting']:
                j = view.col.get(pid)
                if j is not None:
                    refs[pid] = _role_ref(R, Sr, C, mid, j)
            t0, t1 = float(view.t[idx[0]]), float(view.t[idx[-1]])
            rec = bool(np.any((reason[side] == 'recovering') & (view.t >= t0-1e-6) & (view.t <= t1+1e-6)))
            # The interchange intent signal needs the slot really re-occupied at a sample of the advance (critic S-4):
            # the swap record whose incoming player stood in the vacated slot geometry (interchange_cover).
            by_in = next((rel['interchangeBy'][k] for k in range(len(idx)) if rel['interchangeCover'][k]), None)
            ich = next((dict(i) for i in inter if t0-INTERCHANGE_S <= i['t'] <= t1+INTERCHANGE_S and
                        i['out'] == wd and i['in'] == by_in), None) if by_in else None
            intent = {'recovering': rec, 'interchange': ich, 'formationChangeNear': bool(np.any(view.fcn[idx]))}
            vis = 'complete' if all(rel['complete']) else 'partial'
            opps.append(_opportunity(view, 'wide_defender_advance', idx, tier='role', relations=rel, actors=actors,
                                     roles=refs, intent=intent, variant=side,
                                     quality={'identity': view.identity_level(wd, idx), 'visibility': vis}))
    # Per sample, the reason the stored own-phase sample was lost (a sample with an eligible side is not lost).
    counts = Counter()
    for s in np.flatnonzero(own):
        rs = (reason['left'][s], reason['right'][s])
        if '' in rs:
            continue
        cand = [r for r in rs if r in ('recovering', 'team_incomplete')]
        if rs == ('near_change', 'near_change'):
            counts['near_change'] += 1
        elif cand:
            counts[cand[0]] += 1
        elif all(r == 'role_unconfirmed' for r in rs):
            counts['role_unconfirmed'] += 1
        elif 'out_of_view' in rs:
            counts['out_of_view'] += 1
        elif 'format_not_supported' in rs:
            counts['format_not_supported'] += 1
        elif 'role_unconfirmed' in rs:
            counts['role_unconfirmed'] += 1
    for k, v in _seconds({str(k): v for k, v in counts.items()}, step).items():
        lost[k] = round(lost.get(k, 0.0)+v, 2)
    opps.sort(key=lambda o: o['t0'])
    return opps, coverage('wide_defender_advance', 'role', opps, step, geo['phaseSeconds']['own'], lost)


def _d5_small(view, book):
    """D5 on small formats from the relative back block (role tier)."""
    geo = view.geo
    n = len(view.t)
    step = view.step
    players = int(geo['players'])
    opp_phase = view.phase == 'opp'
    lost = dict(geo['lost'].get('opp') or {})
    if players in (4, 5):
        lost['format_not_supported'] = lost.get('format_not_supported', 0.0)+round(float(opp_phase.sum())*step, 2)
        return [], coverage('defensive_spacing', 'role', [], step, geo['phaseSeconds']['opp'], lost)
    L = geo['scale']['lengthM']
    R, Sr, C = roles_at(book, view.ids, view.t)
    reason = np.full(n, NA, dtype=REASON_DTYPE)
    vals = {k: [None]*n for k in D5_KEYS+('gapPrimary', 'gapOther')}
    for s in np.flatnonzero(opp_phase):
        s = int(s)
        back, mid, weak = [], [], False
        for j in range(len(view.ids)):
            sl = _slot(R[s, j])
            if not sl:
                continue
            if not _relative_ok(Sr[s, j], C[s, j]):
                weak = True
                continue
            (back if sl[0] == 'back' else mid if sl[0] == 'mid' else []).append(j)
        if len(back) < 3:
            reason[s] = 'role_unconfirmed' if weak or not back else 'format_not_supported'
            continue
        if not all(view.ok[s, j] for j in back):
            reason[s] = 'team_incomplete'
            continue
        reason[s] = ''
        X, Y = view.X[s, back], view.Y[s, back]
        g = _gap_relations(X, Y, view.OX[s], view.OY[s], [view.ids[j] for j in back], float(view.ballY[s]),
                           bool(view.ball_ok[s] and np.isfinite(view.ballY[s])), geo['scale'],
                           np.where(view.ok[s], view.X[s], np.nan), np.where(view.ok[s], view.Y[s], np.nan))
        for k, v in g.items():
            vals[k][s] = v
        vals['lineStdM'][s] = _py(G.line_std(X), 2)
        vals['lineSize'][s] = len(back)
        if mid and all(view.ok[s, j] for j in mid):
            mx = float(np.mean(view.X[s, mid]))
            if float(np.max(X)) > mx:
                vals['dmGapState'][s] = 'units_interleaved'
            else:
                vals['dmGapM'][s] = _py(mx-float(np.mean(X)), 2)
                vals['dmGapState'][s] = ''
                if view.ball_ok[s] and np.isfinite(view.ballY[s]):
                    vals['dmGapOpponents'][s], vals['dmGapFree'][s] = between_lines(
                        view.OX[s], view.OY[s], float(np.mean(X)), mx, float(view.ballX[s]), float(view.ballY[s]),
                        float(geo['scale']['widthM']), view.X[s], view.Y[s], reach_m=DM_REACH_M*L/G.REF_LENGTH)
        else:
            vals['dmGapState'][s] = 'not_measured'
        h = float(np.mean(X))
        vals['lineHeightM'][s] = _py(h, 2)
        vals['blockHeight'][s] = G.block_height(h, L)
        vals['primaryNamed'][s] = bool(g['gapPrimary'] is not None and view.named[s, view.col[g['gapPrimary']]])
    keys = np.array([f'{b}|{p}' for b, p in zip(vals['ballSide'], view.source)], dtype=object)
    opps = []
    for idx in segments(view.t, reason == '', keys, min_s=MIN_S['defensive_spacing'], step=step):
        rel = {k: [vals[k][i] for i in idx] for k in D5_KEYS}
        rel['contestS'] = [_py(view.contest[i], 2) for i in idx]
        prim = _modal(vals['gapPrimary'][i] for i in idx)
        other = _modal(vals['gapOther'][i] for i in idx if vals['gapPrimary'][i] == prim)
        actors = _actors([prim], [other])
        mid_s = int(idx[len(idx)//2])
        refs = {pid: _role_ref(R, Sr, C, mid_s, view.col[pid]) for pid in actors['primary']+actors['supporting']
                if pid in view.col}
        opps.append(_opportunity(view, 'defensive_spacing', idx, tier='role', relations=rel, actors=actors, roles=refs,
                                 quality={'identity': view.identity_level(prim, idx) if prim else 'high'}))
    for k, v in _seconds(_reason_counts(reason), step).items():
        lost[k] = round(lost.get(k, 0.0)+v, 2)
    return opps, coverage('defensive_spacing', 'role', opps, step, geo['phaseSeconds']['opp'], lost)


def detect_roles(obs, rolebook):
    """The role tier's 'detect_roles' stage: {'opportunities': [Opportunity, tier role], 'detectors': {name:
    DetectorCoverage}} from the obs file's stored geometry (roleInputs['detect']) and the RoleBook: D3 always, D5 on
    small formats."""
    geo = (obs.get('roleInputs') or {}).get('detect')
    if not isinstance(geo, dict) or geo.get('version') != DETECT_VERSION:
        own = float(((obs.get('timeline') or {}).get('phaseSeconds') or {}).get('own') or 0.0)
        cov = coverage('wide_defender_advance', 'role', [], .2, own, {})
        return {'opportunities': [], 'detectors': {'wide_defender_advance': cov}}
    view = _StoredView(geo)
    opps, cov = _d3(view, rolebook)
    out = {'wide_defender_advance': cov}
    if int(geo['players']) != 11:
        found, c = _d5_small(view, rolebook)
        opps += found
        out['defensive_spacing'] = c
    opps.sort(key=lambda o: (o['t0'], S.DETECTORS.index(o['detector'])))
    return {'opportunities': opps, 'detectors': out}


# ---------- sample outcomes (the profile's parameters; assess.py and EVIDENCE) ----------
def _scaled_param(p, name, default, scale):
    v = p.get(name, default)
    if v is None:
        return None
    return G.scaled(v, p.get(name[:-1]+'Scale') if name.endswith('M') else None, scale)


def pressure_cover_flags(R, k, params, scale):
    """(pressed, cover) at sample k of a pressure_cover opportunity's relations R under the parameters. pressed is the
    three-state call's 'pressed' (fix round 1, critic S-2, LEAD decision): the first defender within pressureCertainM
    (3 m), or within pressureM (5 m) while closing at pressureClosingMS (1 m/s) or more, or (the 'distance_or_go_to_ball'
    rule of levels 1-2) closing at goToBallClosingMS or more within goToBallMaxM. Beyond pressureM without the
    go-to-ball rule the carrier is 'not pressed'; between the two without closing the call is 'unclear' (see
    sample_outcomes). cover = the nearest team-mate within the cover angle stands within the scaled cover radius.
    Basis (StatsBomb Euro 2020, 4 matches, data by StatsBomb; 8,576 carriers with a visible opponent): "within 5 m"
    flags 43.9% of the carriers StatsBomb calls not under pressure (precision .322, base rate .198; 72.9% in the subset
    whose nearest opponent is fully in view), 3 m gives precision .421 and the best Youden J (.431); "nobody within
    5 m" agrees with "not under pressure" on .935 (.911 in the defending third). StatsBomb freeze frames carry no
    velocity, so the closing part is a PitchIQ proposal it does not validate."""
    p = dict(REFERENCE_PARAMS['pressure_cover'])
    p.update({k_: v for k_, v in (params or {}).items() if v is not None})
    d1, closing = R['d1'][k], R['closingMS'][k]
    outer = float(p['pressureM'])
    certain = outer if p.get('pressureCertainM') is None else min(float(p['pressureCertainM']), outer)
    need = p.get('pressureClosingMS')
    closes = need is not None and closing is not None and closing >= float(need)
    pressure = d1 is not None and (d1 <= certain or (d1 <= outer and closes) or (
        p['pressureRule'] == 'distance_or_go_to_ball' and closing is not None and closing >= p['goToBallClosingMS']
        and d1 <= p['goToBallMaxM']))
    radius = G.cover_radius(scale, p['coverRadiusM'], p['coverRadiusMinM'])
    angle = min(float(p['coverAngleDeg']), G.COVER_ANGLE_DEG)
    cd, ca = R['coverDist'][k], R['coverAngle'][k]
    cover = cd is not None and ca is not None and cd <= radius and ca <= angle
    return bool(pressure), bool(cover)


def sample_outcomes(opp, params, scale, as_detector=None):
    """Per sample of an opportunity: +1 (the behaviour was demonstrated), -1 (not demonstrated), 0 (unclear: a needed
    part not in view, or a situation the parameters do not judge) or None (outside what the parameters assess, e.g.
    a carrier outside the level's zone); and per sample the failure mode of a -1 (pressure_cover 'no_cover' /
    'no_pressure', defensive_spacing 'lateral_gap' / 'dm_gap', attacking_shape 'width' / 'depth' / 'shape' (both) /
    'channels', else None). params: a DetectorProfile's params (any missing one at REFERENCE_PARAMS); scale: the
    timeline's {lengthM, widthM}; as_detector: judge the samples with another detector's rule on the same relations
    (level 1 reframes a wide-defender advance as "someone ready to protect our goal": the behind_ball rule on the
    advance's behindN / restMargin, DESIGN 4.3).

    Fix round 1 (each rule on the relations measured; a relation an older record lacks never makes a failure):
    - pressure_cover: the three-state pressure call (pressure_cover_flags): 3-5 m without closing is 0, never "pressed
      without cover" (on Metrica 184 of 270 professional no-cover failing samples sat at 3-5 m, 101 with the
      "presser" moving away); "nobody pressed" keeps d1 > pressureM (5 m).
    - weak_side_balance: not judged with the ball within excludeOwnGoalM (20 m x L/105) of the own goal line (box
      defending, not switch protection); goal-side with the level tolerance (weakGoalCount); a far-half defender level
      with the ball or running back at 3 m/s or more makes the sample 0, never "nobody".
    - wide_defender_advance: covered by (a) the channel, (c) the near centre-back's shift of cbShiftM or more
      (nearCbShiftM), (e) a geometric interchange (interchangeCover) or (b) at L5+; level 2's team-level rule
      (requireNobodyBehind; fix round 3, critic B-1): covered +1 wherever the ball is, uncovered -1 when EXPOSED (an
      opponent in the vacated channel, opponentsInChannel >= 1) wherever the ball is or with nobody behind the ball
      in the final third (D4_JUDGED_ZONE), else 0 (no ball depth: out of scope).
    - behind_ball: the level-1 reframed advance: covered +1 (the channel held, the stored cbAdjusted flag or an
      interchange) and exposed -1 (one of theirs in the vacated channel) wherever the ball is; otherwise the
      behind-the-ball rule in the final third (behindN against minBehind, or the margin rule), 0 elsewhere; a sample of
      an older roles file without the ball's depth is out of scope. This is NOT level 2's rule (corrected before the
      freeze, critic round-3 N-1): it judges every sample of the advance without the wide-defender check's scope
      (advancedM, or level 2's small-format halfway rule, USS-F-7V7), its recovering exemption (recoveringMS) or its
      cbShiftM cover test. On 11v11 the two levels agree on every failing sample (every synthetic scene, Metrica game
      1 F and C0: freeze gate verify3/gate/l1_scope_probe.txt); in 7v7, 996 of the 7829 level-1 failing samples of
      that probe (amateur_coach_spans spans and veo, 16 opportunities each) are ones level 2 leaves out of scope, so
      a U8 report can criticise an advance a U10 report does not judge (accepted for the frozen version).
    Fix round 3 (critic B-1, COACH F-1): levels 1-2 had judged an advance only with the ball in the final third, so
    9 of the 10 planted unsupported advances (the full-back 20 m up, the channel empty, an opponent standing in it)
    were never mentioned at U8 / U10; the exposed relation was measured and unused there.
    - defensive_spacing: the D-M gap fails only with a free opponent between the lines on the ball's half within
      reach of the ball (dmGapFree; dmGapOpponents counts every one there).
    - attacking_shape: depth is the field share OR the share of the space available (our length over the length from
      our deepest outfield player to the opponents' last line, depthAvailShare); depthRule 'available' (11v11): 0
      where the field share fails and the available length cannot be measured (the opponents not complete);
      'available_or_field' (small formats): the field share decides there; 'field': the field share alone."""
    d = as_detector or opp['detector']
    p = dict(REFERENCE_PARAMS.get(d, {}))
    p.update({k: v for k, v in (params or {}).items() if v is not None or k in ('minMargin', 'minBehind')})
    R = opp['relations']
    n = len(R['t'])
    L, W = float(scale['lengthM']), float(scale['widthM'])
    out, modes = [None]*n, [None]*n
    if d == 'pressure_cover':
        for k in range(n):
            d1, cx = R['d1'][k], R['carrierX'][k]
            if d1 is None or cx is None:
                continue
            if p['carrierZone'] == 'own_half' and not cx < L/2.0:
                continue
            pressure, cover = pressure_cover_flags(R, k, p, scale)
            unpressed = not pressure and d1 > float(p['pressureM'])
            in_zone = cx < (L/3.0 if p['unpressuredZone'] == 'own_third' else L/2.0)
            if not p['coverRequired']:
                out[k] = 1 if pressure else (-1 if unpressed and in_zone else 0)
            elif pressure:
                if cover:
                    out[k] = 1
                elif not R['coverVisible'][k]:
                    out[k] = 0
                elif p.get('coverExemptZone') == 'final_third' and cx >= 2.0*L/3.0:
                    out[k] = 0
                else:
                    out[k] = -1
            else:
                out[k] = -1 if unpressed and in_zone else 0
            if out[k] == -1:
                modes[k] = 'no_cover' if pressure else 'no_pressure'
    elif d == 'weak_side_balance':
        box_x = _scaled_param(p, 'excludeOwnGoalM', 20.0, scale)
        need = p['minWeakCount']
        tolerant = 'weakGoalCount' in R
        for k in range(n):
            bx = R['ballX'][k] if 'ballX' in R else None
            if box_x is not None and bx is not None and bx < box_x-1e-9:
                continue
            g = R['weakGoalCount'][k] if tolerant else None
            if g is not None:
                level, rec = R['weakLevelCount'][k] or 0, R['weakRecovering'][k] or 0
                out[k] = 1 if g >= need else (0 if (g+level >= need or rec >= 1) else -1)
            else:
                w = R['weakCount'][k]
                if w is not None:
                    out[k] = 1 if w >= need else -1
    elif d == 'wide_defender_advance':
        adv_m = _scaled_param(p, 'advancedM', 15.0, scale)
        cb_m = p.get('cbShiftM')
        for k in range(n):
            if p['advancedRule'] == 'halfway':
                b = R['beyondHalfM'][k]
                if b is None or b < p['advancedHalfwayShare']*L:
                    continue
            elif R['advanceM'][k] is None or R['advanceM'][k] < adv_m:
                continue
            v = R['wdVelX'][k]
            if v is not None and v <= -float(p.get('recoveringMS', G.RECOVERING_MS)):
                continue
            team_rule = bool(p.get('requireNobodyBehind'))
            if team_rule and 'ballX' not in R:
                continue                    # a roles file from before fix round 1: never judged at level 2
            bx = R['ballX'][k] if 'ballX' in R else None
            shift = R['nearCbShiftM'][k] if 'nearCbShiftM' in R else None
            cb = (shift >= float(cb_m)) if (shift is not None and cb_m is not None) else bool(R['cbAdjusted'][k])
            swap = bool(R['interchangeCover'][k]) if 'interchangeCover' in R else False
            covered = bool(R['channelOccupied'][k] or cb or swap or (p.get('useDmRotation') and R['dmDropped'][k]))
            exposed = not covered and (R['opponentsInChannel'][k] or 0) >= 1
            if covered:
                out[k] = 1
            elif team_rule:
                # level 2 (fix round 3, critic B-1): uncovered fails with one of theirs in the vacated channel
                # anywhere, or with nobody behind the ball in the final third; otherwise unclear (no depth: none)
                if exposed:
                    out[k] = -1
                elif bx is None:
                    continue
                elif bx >= (2.0*L/3.0 if D4_JUDGED_ZONE == 'final_third' else L/3.0)-1e-9:
                    out[k] = -1 if R['behindN'][k] == 0 else 0
                else:
                    out[k] = 0
            elif p['countOnly'] == 'exposed':
                out[k] = -1 if exposed else 0
            else:
                out[k] = -1
            if out[k] == -1:                # the failure mode (fix round 3, for TEXT's level 1-2 wording)
                modes[k] = 'exposed' if exposed else ('nobody_behind' if team_rule else 'uncovered')
    elif d == 'behind_ball':
        # Judged with the ball in the final third only (D4_JUDGED_ZONE; DESIGN 4.4 deviation: see the constant), the
        # level-1 reframed advance too (critic B-1: BC60 / BC303 Barcelona building up at 42-61 m with the channel
        # held were told "nobody stayed back"); a reframed sample without the ball's depth is out of scope.
        zone_x = 2.0*L/3.0 if p.get('behindBallZone', D4_JUDGED_ZONE) == 'final_third' else L/3.0
        reframed = opp.get('detector') == 'wide_defender_advance'
        for k in range(n):
            bx = R['ballX'][k] if 'ballX' in R else None
            if reframed:
                # level 1 (fix round 3, critic B-1, the spec's EARLY DEVELOPMENT example "nobody stayed behind to
                # cover"): a covered advance is protected (+1) and an uncovered one with one of theirs in the
                # vacated channel is not (-1) wherever the ball is; otherwise the behind-the-ball rule in the final
                # third, unclear (0) outside it, out of scope without the ball's depth (and in an older roles file)
                if 'ballX' not in R:
                    continue
                covered = bool(R['channelOccupied'][k] or R['cbAdjusted'][k] or
                               ('interchangeCover' in R and R['interchangeCover'][k]))
                if covered:
                    out[k] = 1              # someone covers the space the defender left: protected
                    continue
                if (R['opponentsInChannel'][k] or 0) >= 1:
                    out[k], modes[k] = -1, 'exposed'
                    continue
                if bx is None:
                    continue
                if bx < zone_x-1e-9:
                    out[k] = 0
                    continue
            elif bx is not None and bx < zone_x-1e-9:
                continue
            if p['behindRule'] == 'margin':
                m = R['restMargin'][k]
                if m is None or p.get('minMargin') is None:
                    out[k] = 0
                else:
                    out[k] = 1 if m >= p['minMargin'] else -1
            else:
                b = R['behindN'][k]
                if b is not None and p.get('minBehind') is not None:
                    out[k] = 1 if b >= p['minBehind'] else -1
            if reframed and out[k] == -1:
                modes[k] = 'nobody_behind'
    elif d == 'defensive_spacing':
        lat = _scaled_param(p, 'lateralGapM', 20.0, scale)
        dmg = _scaled_param(p, 'dmGapM', 18.0, scale)
        lo = _scaled_param(p, 'dmGapStrengthLoM', 8.0, scale)
        hi = _scaled_param(p, 'dmGapStrengthHiM', 14.0, scale)
        for k in range(n):
            gap, dm, std = R['lateralGapM'][k], R['dmGapM'][k], R['lineStdM'][k]
            fail_lat = gap is not None and gap > lat and R['gapBallSide'][k] is True and R['opponentInGap'][k] is True
            between = R['dmGapFree'][k] if 'dmGapFree' in R else None
            fail_dm = dm is not None and dm > dmg and between is not None and between >= 1
            if fail_lat or fail_dm:
                out[k], modes[k] = -1, ('lateral_gap' if fail_lat else 'dm_gap')
            elif std is not None and std <= p['lineStdStrengthM'] and dm is not None and lo <= dm <= hi:
                out[k] = 1
            else:
                out[k] = 0
    elif d == 'attacking_shape':
        rule = p.get('depthRule') or 'field'
        for k in range(n):
            ws, ds, ch = R['widthShare'][k], R['depthShare'][k], R['channelsOccupied'][k]
            if ws is None or ds is None:
                continue
            w_ok = ws >= p['widthShare'] or p.get('judgeWidth') is False
            c_ok = ch == 3
            if ds >= p['depthShare'] or rule == 'field':
                d_ok = ds >= p['depthShare']
            else:
                da = R['depthAvailShare'][k] if 'depthAvailShare' in R else None
                if da is None:
                    d_ok = False if rule == 'available_or_field' else None
                else:
                    d_ok = da >= p['depthAvailShare']
            if w_ok and c_ok and d_ok:
                out[k] = 1
            elif not w_ok or not c_ok or d_ok is False:
                out[k] = -1
                if not w_ok:
                    modes[k] = 'shape' if d_ok is False else 'width'
                else:
                    modes[k] = 'depth' if d_ok is False else 'channels'
            else:
                out[k] = 0                  # width and channels held; the depth could not be measured
    return out, modes


def key_sample(opp, scale):
    """A typical sample of an opportunity, profile-independently (REFERENCE_PARAMS): the middle of its longest run of
    not-demonstrated samples, else of demonstrated ones, else its middle sample. EVIDENCE may use it as the moment."""
    out, _ = sample_outcomes(opp, None, scale)
    n = len(out)
    for want in (-1, 1):
        best, cur = None, None
        for k in range(n+1):
            if k < n and out[k] == want:
                cur = (cur[0], k) if cur else (k, k)
            else:
                if cur and (best is None or cur[1]-cur[0] > best[1]-best[0]):
                    best = cur
                cur = None
        if best:
            return (best[0]+best[1])//2
    return n//2


__all__ = ['detect', 'detect_roles', 'role_geometry', 'roles_at', 'sample_outcomes', 'pressure_cover_flags',
           'key_sample', 'segments',
           'coverage', 'channel_in_view', 'REFERENCE_PARAMS', 'DETECT_VERSION', 'MIN_S']
