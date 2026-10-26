"""The situation timeline (DESIGN v2 section 3.1-3.3, 3.5, 3.6): per team and per sample of TeamShape's 5 Hz grid,
the match situation a coaching check may rest on, and nothing it may not.

Why a timeline of its own. The v1 detectors judged every sample against one professional match whatever the phase, and
the audit measured that phase moves the very quantities they judged: on Metrica game 1 (full tracking, truth phase)
the team width is 47.6 / 50.4 m in possession against 36.0 / 39.0 m out of it, the back line's depth spread 4.6 / 2.9
against 1.4 / 1.7 m (team A / B). So every v2 detector (soccer/coaching/contextual.py) asks first: who had the ball,
is that known, where was the ball and who carried it, which players were really seen where, and was the whole team in
the picture. This module answers those questions once, profile-independently (the measurement tier: it reads only the
coaching series, the possession-4 record, the format, the field size and the coach-marked spans), and states for each
answer where it is unknown instead of guessing (spec Part 4: "if the ball or phase of play cannot be identified
reliably, do not fabricate it").

Phase from possession-4 (DESIGN 3.2; the binding rules of the possession v4 report, possession/v4/reports/docs.md):
1. Used only where the record is available; quality 'limited' is accepted and its coverage stated (on the held-out
   game 3 limited windows were only .004-.022 less accurate, and the fresh broadcast segment 222-303 s, limited,
   scored .949 at coverage .924).
2. [min(lostAt, wonAt - 2 s), wonAt + 2 s] around EVERY change (high and uncertain) is excluded: the team is not named
   there (`phase` unknown, `nearChange` true). Change timing is the engine's weak point (F1 .69 full tracking, .60
   camera; within 1 s .64 / .49-.50); after the exclusion the named frames are right .975-.977 of the time on game 3
   and .996 on broadcast 42-222 s. The windows are kept as `nearChange` only to SUPPRESS criticism (a player sprinting
   back may be reacting to a change); a change is never an event here: no turnover, no "won / lost the ball".
3. The run's confidence is not used as a filter (it lowered accuracy on the fresh segment: .946 against .949).
4. The engine's dead spells (and dead runs) are dead. Broadcast stoppages are mostly missed (about 16% of the time
   kept by rule 2 on the fresh segment was really dead ball; 15 of its 55 dead points stay named): a note says so.
5. `phaseSpan` is the length of the contiguous span of one phase containing the sample; shape and spacing checks use
   spans of 3 s or more (schema MIN_SHAPE_SPAN_S).
6./8. No turnover, no "won / lost by", only "when we had the ball (estimated)": TEXT's rules; nothing here names a
   change as an event (the notes are checked event-free).
7. Shares only roughly: `roughShares` rounded to 5%, with the engine's coverage (game 3: per-300 s window error median
   .031 / .019, worst .07).
Measured here on the five run copies (team A's grid; DESIGN 3.1 phase shares are of the whole segment): broadcast
0-60 s named .977 (no change, nothing excluded), 42-222 s .887 raw -> .790 after the exclusion, 222-303 s .778 ->
.672 (dead .183); the amateur and Veo clips abstain (small, far scenes on 60% / 61% of their frames).

Coach-marked spans (user decision 1, 2026-10-21): a coach may mark "team A had the ball from t0 to t1" or "dead ball
from t0 to t1" (schema 'coach_spans'). Where a span covers a sample it decides the phase instead of possession-4
(`phaseSource` 'coach'): the coach is trusted as given, so no +-2 s exclusion is applied at a coach boundary, while
the engine's windows still apply outside the spans. A span covers [t0, t1) (two spans marked end to end do not share
their boundary instant; one ending at the segment's end covers it). A span must lie inside the analysed segment (half a
second of tolerance at either end for a click at the first or last frame); a span outside it, an overlapping one or a
malformed one is not used and is counted (`phaseQuality.coach.dropped`). The engine's agreement inside the spans is
measured (`agreement`) and stated, never used to overrule the coach. On amateur and Veo footage, where possession-4
abstains, coach spans are what unlocks the phase; without them every sample there is `unknown` and the phase-dependent
checks report `phase_unknown`. Measured with spans made from the blind possession hand labels (0.5 s points joined into
runs of A / B / dead of at least 1 s): the amateur clip goes from 0 to .70 of its 64 s named (19.6 s / 25.4 s own
possession for team A / B), the Veo clip from 0 to .85 of 30 s (A had the ball throughout); on the broadcast runs the
engine names the coach's team on 1.0 (0-60 s), .972 (42-222 s) and .801 (222-303 s, where it names stoppages as
possession) of the seconds both name. They unlock only what needs the phase alone: on these clips the carrier (from
the engine) and the ball (vetoed on small scenes) stay unknown, so the pressure, weak-side and behind-the-ball checks
still have no eligible time; the wide-defender, spacing and shape checks get some (team A on the amateur clip: 19.6 s
of own possession in spans of 3 s or more, 5.8 s of it with the whole team in view).

Ball and carrier (DESIGN 3.3). The ball is the series' ball (final path TRACKED or bridged MISSING, ball pitch
confidence MEDIUM or better, sigma at most 3 m, on the pitch, registration gates passed, vetoed on small-scene runs:
18-20% of the amateur clip's shown ball is the end of a painted arc). Its third and channel are recomputed here on the
coaching scale (a declared small field changes the metres) with the series' 3 m hysteresis, and named as the schema's
zones (own_third / middle_third / final_third; left / centre / right in the team's attack frame). The carrier is
possession-4's carrier on a `control` run that is NOT `estimated` (DESIGN 3.1: an estimated run is decoded across an
interpolated ball), kept only where the phase names the carrier's team (own: our player, opp: theirs), so a carrier
never contradicts the phase a check runs on. That halves the carrier's coverage: the opponents' carrier is known on
.23-.32 of their named possession on the broadcast runs (BC60 A .32, BC180 A / B .23 / .26, BC303 A .24), against
.45-.59 counting the estimated runs too (DESIGN's .33-.58 counted every control run). The estimated runs' carrier is
kept apart (`estimatedCarrier`, same phase rule) for a consumer that states the lower confidence; `carrier` is never
filled from it.

Players (DESIGN 3.1, 3.5). Positions are the series' observed outfield positions (kind 'O': analytics-validated,
identity >= .3, sigma <= 3 m), both teams, in the evaluated team's ATTACK FRAME on the coaching scale (X metres from
its own goal line towards the goal it attacks, Y from its left touchline); goalkeepers apart. `posOk` adds the
registration gates (calibrated, at least MEDIUM, chains within 3 m) and sigma <= 3 m: the pitch-position gate.
Velocities are the series' (+-0.8 s differences; camera error p50 0.30, p90 0.88 m/s), rescaled to the coaching scale
and turned into this team's frame for the opponents. `tainted` marks a player within 2 s of a crossing, a
re-identification, an uncertain swap or a kit change (the result's issues and events; the identity gate: a player is
never named in a criticism while tainted). `stabFlag` is the stabilizer's veto (evidence only).
`complete`: every outfield player of the FORMAT observed and nobody extra (11v11: team_shape's lineup, 9 after a
dismissal; small formats: the format's outfield count, because team_shape's lineup accepts only 7-10 and the roster's
expected count minus one is wrong for a 4v4 without a keeper). DESIGN 3.1 wrote ">= outfield"; an extra observed
outfield player is a duplicate identity or a departed one and would inflate a behind-the-ball count, so `excess`
samples are not complete (v1's series rule too). Measured completeness (A / B): broadcast 0-60 s .99 / .96, 42-222 s
.56 / .79, 222-303 s .25 / .56, amateur .16 / .18, Veo 0 / 0. `unplaced` counts the people of a team's kit in the
picture without a usable position (anyone of them could be the nearest player: the local-context gate).

Scale (DESIGN 2.5). A coach-declared field wins (the result writes a youth field as 105 x 68 'standard', so every metre
is 105/L too long until it is declared), then a field measured from landmarks, else the assumed 105 x 68. On an
assumed scale body-size distances are usable only on 11v11 (estimated metres, as today); on a small format they are
not (`metresUsable` false: D1's pressure distance and cover radius are not assessed).

Formation periods (11v11 only; small formats never use team_shape's templates, DESIGN 2.5): team_shape's stabilized
periods, the coach's team-shape label where one applies, and `formationChangeNear`: within +-15 s of a change between
two different template labels (transition and data-gap periods between them skipped), where role-based criticism is
suppressed (counterexample "a formation deliberately changes").

Run level (`summary()`, schema 'timeline'): phase seconds and spans (split by source), phase quality with notes, the
footage kind (broadcast; amateur: small scene; veo: small scene with a few players per team in view, as the Veo clip's
2.0 / 2.6), the data-quality chain lineup -> in kit -> confirmed -> positioned with its dominant cause (critic B4: the
Veo clip had 5.7 / 6.8 players in kit, 2.9 / 3.5 confirmed, 2.0 / 2.6 positioned: most of the lineup was out of
view), the scale, completeness, ball and carrier shares, the formation periods and each player's observed seconds and
p90 speed (the reach of a suggested zone, DESIGN 5.1).

Role inputs (`role_inputs()`, stored by the detect stage under obs['roleInputs']['context']; INTERFACES.md CONTEXT
wave 2): the role tier reads only the obs file, so what soccer/coaching/roles.py needs is kept here in a compact form:
team_shape's classified windows (15 s, every 5 s) with each player's template role under the deciding formation and
under the period's runner-up, and the window's own / opponent phase shares; the periods; and a 1 Hz track of each
outfield player's mean attack-frame position (the samples within +-0.5 s, at least 2 of 5) with the phase, for
relative roles, interchanges and the coach-role contradiction check. Measured 0.14-0.22 KB per second of video on the
five runs (13 KB for broadcast 0-60 s, 39 KB for 42-222 s): about 1.2 MB for 90 minutes of 11v11.

Repetition (DESIGN 3.6): `spells()` (contiguous samples of one outcome, merged across interruptions of at most 1 s,
kept from a minimum duration: 2 s, 3 s for D3/D4), `distinct()` (spells at least 10 s apart start to start), and
`rate_per_10min()` (per 10 minutes of the relevant phase, never per clip). Every number here is a PitchIQ proposal
(origin PITCHIQ-PROPOSAL, unvalidated) except where a measured basis is quoted.

Fix round 1 (2026-10-23, the critic's S-3, S-16, B-11): a formation change suppresses role-based criticism only between
two label runs that each hold 60 s and read clear / likely, never with more than 6 changes per 10 minutes (Metrica game
1, 15-minute clips: D3 opportunities flagged 10 of 125 -> 4 on full tracking, 12 of 32 -> 0 on the simulated camera,
whose readings change about 29 times per 15 minutes); +-2 s around a coach-marked change between the two teams' spans
the samples suppress (`coachNear`) while the phase stays the coach's; the rough shares are the merged per-sample
phase's when the coach marked spans and the engine is available (CODE H2: Team A 5-25 s and Team B 25-33.5 s on BC60
read "A 0%, B 100%" from the engine), and with the engine unavailable a note says when only one team was marked
(`one_sided`); `ballSigma` per sample feeds the far-side level tolerance.

Fix round 3 (2026-10-24, LEAD decision (2), FACTS F-3): `contestS` per sample, the seconds to the nearest stretch the
possession engine calls the ball 'loose' (between the loser's last touch and the winner's first: the ball between the
teams) or 'contested' (0 inside one, NaN with none): a measurement of the ball beside the phase, never changed by a
coach's marks. assess.py keeps 'moments worth a look' away from such balls (a clear moment of round 2 at 4557.67 s on
Metrica game 1 lay inside a loose ball of 3.8 s, one at 3217.07 s 0.4-0.8 s from two contested ones). Measured on
Metrica game 1 (engine on 300 s windows): loose / contested on .036 / .030 of the engine's frames on full tracking,
.031 / .023 on the simulated camera. Every other field is unchanged (the before and after whole-match tiers are equal
once contestS and the code fingerprints are dropped).
"""
import math
import re

import numpy as np

from .. import team_shape as ts
from . import episodes as ep
from . import schema as S
from . import series as cs

# ---------- constants (PITCHIQ-PROPOSAL unless a measured basis is given) ----------
CHANGE_EXCLUDE_S = 2.0          # s around every change of possession (possession docs rule 2)
MIN_SHAPE_SPAN_S = S.MIN_SHAPE_SPAN_S   # 3 s: shape and spacing checks use phase spans this long (rule 5)
ROUGH_SHARE_STEP = .05          # possession shares rounded to 5% (rule 7)
COACH_EDGE_S = .5               # s: a coach span may start / end this far outside the segment (a click on an end frame)
TAINT_S = 2.0                   # s around an identity issue (data audit; detectors.IDENTITY_TAINT_SECONDS)
TAINT_ISSUES = ('crossed', 're-identified', 'stopped matching', 'kit')   # result issues[].reason keywords
TAINT_EVENTS = ('swap-uncertain', 'switch', 'team-change')               # result events[].kind
TAINT_HIGH, TAINT_MEDIUM = .10, .30     # a player's tainted share of the rows: high below .10, medium below .30 (v1)
MAX_SIGMA_M = 3.0               # m: the pitch-position gate (analytics / team_shape MAX_SIGMA)
FORMATION_CHANGE_S = 15.0       # s around a formation change (DESIGN 3.4)
FORMATION_PERSIST_S = 60.0      # s: a change suppresses only when both formations around it hold this long (S-3)
FORMATION_MAX_RATE = 6.0        # changes per 10 minutes above which the readings are churn, not changes (S-3)
FORMATION_CONFIDENT = ('clear', 'likely')   # a change suppresses only between readings this confident (or coach-set)
DQ_OK_SHARE = .9                # positioned >= this share of the lineup: no dominant cause is named
FEW_PLAYERS = 4.0               # mean observed outfield players per team below this on a small scene: 'veo' footage
P90_MIN_SAMPLES = 25            # velocity samples (5 s) a player needs for a p90 speed
SPELL_MERGE_S = 1.0             # s: interruptions this short do not split a spell (DESIGN 3.6)
SPELL_MIN_S = 2.0               # s: the default minimum spell (D3 / D4: MIN_SHAPE_SPAN_S)
DISTINCT_S = 10.0               # s between the starts of two distinct spells
RATE_SECONDS = 600.0            # rates are per 10 minutes of the relevant phase
SHORT_RANGE_S = 180.0           # a range shorter than this never claims a pattern (DESIGN 3.6)
TRACK_STEP_S = 1.0              # s: the role inputs' track step
TRACK_HALF_S = .5               # s: a track point is the mean of the player's positions this close to it
TRACK_MIN_SAMPLES = 2           # ... at least this many of them
ZONE_NAMES = ('own_third', 'middle_third', 'final_third')
ZONE_OF = {'own': 'own_third', 'middle': 'middle_third', 'final': 'final_third'}
CHANNEL_NAMES = ('left', 'centre', 'right')
PHASE_DTYPE = '<U7'
SOURCE_DTYPE = '<U12'
ENGINE = 'possession-4'         # Opportunity.phase.source of an engine-named sample
COACH = 'coach'
NONE = 'none'
CONTEXT_VERSION = 1             # roleInputs['context'] layout version
CONTEST_STATES = ('loose', 'contested')     # possession-4 run states with the ball between the teams (contestS)
OPP_NAME = {'x': 'oppx', 'y': 'oppy'}   # the opponents' normalized positions (oppX / oppY are attack-frame metres)

STOPPAGE_NOTE = ('Stoppages are mostly missed on broadcast footage: on a fresh test segment about 16% of the time '
                 'kept was really dead ball.')


def _number(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _ascii(text, n=300):
    """Printable ASCII, at most n characters (schema Text)."""
    out = re.sub(r'[^ -~]', ' ', str(text or '')).strip()
    return out[:n] if out else None


def _r(x, d=3):
    return None if x is None or not _number(float(x)) else round(float(x), d)


def team_of(pid, gk_teams=None):
    """The team of an internal id ('A-07' -> 'A'); goalkeepers through gk_teams {id: team}."""
    if not isinstance(pid, str):
        return None
    if pid[:2] in ('A-', 'B-'):
        return pid[0]
    return (gk_teams or {}).get(pid)


# ---------- repetition (DESIGN 3.6) ----------
def spells(t, outcome, eligible=None, min_s=SPELL_MIN_S, merge_s=SPELL_MERGE_S, step=None):
    """The spells of an outcome: runs of samples where `outcome` (and `eligible`) is true, merged across interruptions
    (ineligible or other-outcome samples) of at most merge_s seconds, kept when they last at least min_s (t1 - t0 +
    step). Returns [{'t0', 't1', 'i0', 'i1', 'samples' (outcome samples), 'seconds'}] in time order."""
    t = np.asarray(t, float)
    m = np.asarray(outcome, bool)
    if eligible is not None:
        m = m & np.asarray(eligible, bool)
    step = ep.sample_step(t) if step is None else float(step)
    idx = np.flatnonzero(m)
    out = []
    if not len(idx):
        return out
    runs = []
    start = prev = int(idx[0])
    count = 1
    for i in idx[1:]:
        i = int(i)
        if t[i]-t[prev]-step > merge_s+.5*step:
            runs.append((start, prev, count))
            start, count = i, 0
        prev = i
        count += 1
    runs.append((start, prev, count))
    for i0, i1, n in runs:
        seconds = float(t[i1]-t[i0]+step)
        if seconds >= min_s-1e-9:
            out.append({'t0': float(t[i0]), 't1': float(t[i1]), 'i0': i0, 'i1': i1, 'samples': int(n),
                        'seconds': round(seconds, 3)})
    return out


def distinct(found, apart=DISTINCT_S):
    """The distinct spells: in start order, a spell counts when it starts at least `apart` seconds after the last
    counted one (DESIGN 3.6: recurrence = their number)."""
    out, last = [], None
    for sp in sorted(found, key=lambda s: s['t0']):
        if last is None or sp['t0']-last >= apart-1e-9:
            out.append(sp)
            last = sp['t0']
    return out


def rate_per_10min(count, seconds):
    """Occurrences per 10 minutes of the relevant phase time (None without any)."""
    return None if not seconds or seconds <= 0 else float(count)*RATE_SECONDS/float(seconds)


def short_range(seconds):
    """A range under 3 minutes: no 'repeated' / 'pattern' / 'often' wording, work items need 3 distinct spells."""
    return float(seconds) < SHORT_RANGE_S


# ---------- coach-marked spans ----------
def clean_coach_spans(raw, t0, t1, edge=COACH_EDGE_S):
    """(kept spans [{t0, t1, state}], dropped count): the usable coach spans of a 'coach_spans' record (or a plain list
    of spans) for a segment [t0, t1]. A span is kept when its times are numbers with t0 < t1, its state is A, B or
    dead, it lies inside the segment (edge seconds of tolerance; then clipped to it) and it does not overlap a span kept
    before it (in start order)."""
    if raw is None:
        return [], 0
    spans = raw.get('spans') if isinstance(raw, dict) else raw
    if not isinstance(spans, (list, tuple)):
        return [], 1
    cand, dropped = [], 0
    for s in spans:
        ok = isinstance(s, dict) and _number(s.get('t0')) and _number(s.get('t1')) and s.get('state') in ('A', 'B', 'dead')
        if not ok or not float(s['t1']) > float(s['t0']) or float(s['t0']) < t0-edge or float(s['t1']) > t1+edge:
            dropped += 1
            continue
        cand.append({'t0': max(float(s['t0']), t0), 't1': min(float(s['t1']), t1), 'state': s['state']})
    cand.sort(key=lambda s: (s['t0'], s['t1']))
    kept = []
    for s in cand:
        if kept and s['t0'] < kept[-1]['t1']-1e-9:
            dropped += 1
            continue
        kept.append(s)
    return kept, dropped


# ---------- the timeline ----------
class SituationTimeline:
    """One team's situation timeline on the coaching grid (module docstring). Per sample (length S): t, frame, side;
    phase ('own', 'opp', 'dead', 'unknown'), phaseSource ('possession-4', 'coach', 'none'), phaseSpan (s, 0 where
    unknown), nearChange, changeConfidence ('high', 'uncertain', ''); ballX/ballY (normalized pitch), ballAX/ballAY
    (attack-frame metres), ballReliable, ballThird (schema zone or ''), ballChannel; carrier (id or ''), carrierTeam,
    carrierX/carrierY (attack-frame metres, NaN unseen), estimatedCarrier (an estimated run's carrier, never in
    `carrier`); contestS (seconds to a loose or contested ball, NaN without one: fix round 3); coachMarked (a coach
    span decides the sample), coachNear; observedCount, outfield, complete, excess, unplaced (this
    team), oppObservedCount, oppComplete, oppUnplaced; regGate (series registration code, '' passes);
    formationPeriod (index, -1), formationLabel, formationConfidence, formationChangeNear; inKit, confirmedInKit.
    Per player (S x N, `ids` / `col`; the opponents' S x M with the prefix opp: oppIds, oppCol, oppx, oppy, oppX,
    oppY, oppSigma, ...): x, y (normalized, NaN unless observed), X, Y (this team's attack frame, metres), sigma,
    identity, tainted, posOk, velX, velY, speed (m/s, this team's frame), velOk, stabFlag (own only). Goalkeepers: gk, oppGk (S x 2 normalized), gkX/gkY, oppGkX/oppGkY. Also: team, other,
    format {players, goalkeeper, outfield, name}, scale {lengthM, widthM, source, metresUsable}, step, fps, view (per
    sample: polygon in pitch metres on the coaching scale, None, False), series / oppSeries (the TeamSeries), cd."""

    def __init__(self):
        self.team = self.other = None
        self.arrays = {}

    def __getattr__(self, name):
        arrays = self.__dict__.get('arrays')
        if arrays is not None and name in arrays:
            return arrays[name]
        raise AttributeError(name)

    def _put(self, **kw):
        self.arrays.update(kw)

    # ----- geometry helpers -----
    def to_pitch(self, X, Y, s):
        """This team's attack-frame metres at sample s -> pitch metres on the coaching scale (None if no side)."""
        side = self.side[s]
        if side == 'left':
            return X, Y
        if side == 'right':
            return self.scale['lengthM']-X, self.scale['widthM']-Y
        return None

    def disc_in_view(self, s, X, Y, r):
        """Is the disc of radius r (metres) around (X, Y) (attack frame) inside the camera's view of the pitch at
        sample s (series._disc_inside: only the pitch matters)? True when the result has no view field."""
        v = self.view[s] if 0 <= s < len(self.view) else None
        if v is None:
            return True
        if v is False:
            return False
        p = self.to_pitch(float(X), float(Y), s)
        if p is None or not (_number(p[0]) and _number(p[1]) and _number(r)):
            return False
        return bool(cs._disc_inside(v, np.array([p[0]]), np.array([p[1]]), np.array([float(r)]),
                                    self.scale['lengthM'], self.scale['widthM'])[0])

    def local_context(self, s, X, Y, r, teams=('own',)):
        """The local-context gate (DESIGN 3.5): the disc in view and nobody of the named teams ('own', 'opp') in the
        picture without a usable position."""
        for who in teams:
            if (self.unplaced if who == 'own' else self.oppUnplaced)[s] > 0:
                return False
        return self.disc_in_view(s, X, Y, r)

    def rows(self, t0=None, t1=None):
        """Sample indices inside [t0, t1] (absolute seconds; None = open)."""
        m = np.ones(len(self.t), bool)
        if t0 is not None:
            m &= self.t >= float(t0)-1e-6
        if t1 is not None:
            m &= self.t <= float(t1)+1e-6
        return np.flatnonzero(m)

    def identity_level(self, pid, rows):
        """'high' / 'medium' / 'low' identity of a player over the rows from his tainted share (v1 per-player levels:
        below .10 high, below .30 medium); None when he is of neither team."""
        if pid in self.col:
            T = self.tainted[:, self.col[pid]]
        elif pid in self.oppCol:
            T = self.oppTainted[:, self.oppCol[pid]]
        else:
            return None
        rows = np.asarray(rows, int)
        share = float(T[rows].mean()) if len(rows) else 0.0
        return 'high' if share < TAINT_HIGH else ('medium' if share < TAINT_MEDIUM else 'low')

    def named_ok(self, pid, s):
        """May the player be named at sample s (identity gate: observed and not within 2 s of an identity issue)?"""
        if pid in self.col:
            j = self.col[pid]
            return bool(self.posOk[s, j] and not self.tainted[s, j])
        if pid in self.oppCol:
            j = self.oppCol[pid]
            return bool(self.oppPosOk[s, j] and not self.oppTainted[s, j])
        return False

    def phase_spells(self, state, min_s=0.0):
        """The contiguous spans of one phase state as spells (spells() with no merging)."""
        return spells(self.t, self.phase == state, min_s=min_s, merge_s=0.0, step=self.step)

    # ----- outputs -----
    def summary(self):
        return self._summary

    def role_inputs(self):
        """JSON-safe inputs of soccer/coaching/roles.py (module docstring "Role inputs"); the detect stage stores them
        under obs['roleInputs']['context']."""
        return self._role_inputs


# ---------- building ----------
def _format(measurement):
    fmt = (measurement or {}).get('format') if isinstance(measurement, dict) else None
    if not (isinstance(fmt, dict) and fmt.get('players') in S.FORMAT_PLAYERS):
        fmt = dict(S.DEFAULT_FORMAT)
    players = int(fmt['players'])
    keeper = bool(fmt.get('goalkeeper')) if players in (4, 5) else True
    fmt = {'players': players, 'goalkeeper': keeper}
    return dict(fmt, outfield=S.outfield(fmt), name=S.format_name(fmt))


def scale_of(cd, fmt, field):
    """The coaching scale (module docstring "Scale"): {lengthM, widthM, source, metresUsable}."""
    pitch = cd.pitch or {}
    rl, rw = float(pitch.get('length') or 105.0), float(pitch.get('width') or 68.0)
    if isinstance(field, dict) and _number(field.get('length')) and _number(field.get('width')) and \
            15.0 <= field['length'] <= 130.0 and 10.0 <= field['width'] <= 100.0:
        L, W, src = float(field['length']), float(field['width']), 'declared'
    elif pitch.get('source') == 'landmarks':
        L, W, src = rl, rw, 'measured'
    else:
        L, W, src = rl, rw, 'assumed'
    L, W = min(max(L, 15.0), 130.0), min(max(W, 10.0), 100.0)
    usable = src != 'assumed' or fmt['players'] == 11
    return {'lengthM': L, 'widthM': W, 'source': src, 'metresUsable': bool(usable)}


def _ball_sigma(cd, frames, S_):
    """Per sample: the ball's larger pitch sigma (m) on the sample's result frame, NaN where none is given (the
    D2 level tolerance, contextual._d2)."""
    out = np.full(S_, np.nan)
    data = cd.shape.observed if cd.shape is not None else None
    fidx = getattr(data, 'frame_index', None) if data is not None else None
    if fidx is None or len(fidx) != S_:
        return out
    for s, i in enumerate(fidx):
        i = int(i)
        b = frames[i].get('ball') if 0 <= i < len(frames) and isinstance(frames[i], dict) else None
        sig = b.get('pitchSigma') if isinstance(b, dict) else None
        if isinstance(sig, (list, tuple)) and sig and all(_number(v) for v in sig):
            out[s] = float(max(sig))
        elif _number(sig):
            out[s] = float(sig)
    return out


def _engine_phase(poss, frame, t, team):
    """Per sample: the engine's team after possession docs rules 1-4 ('own', 'opp', 'dead', 'unknown'), the raw team
    before the change exclusion, the carrier (non-estimated control runs) and its team, nearChange and the change
    confidence."""
    S_ = len(t)
    phase = np.full(S_, 'unknown', dtype=PHASE_DTYPE)
    raw = np.full(S_, '', dtype='<U4')          # 'A', 'B', 'dead' before the exclusion
    carrier = np.full(S_, '', dtype='<U8')
    estimated = np.full(S_, '', dtype='<U8')     # the carrier of an estimated control run (ball interpolated)
    near = np.zeros(S_, bool)
    conf = np.full(S_, '', dtype='<U9')
    ok = isinstance(poss, dict) and poss.get('available') is True and isinstance(poss.get('timeline'), list)
    if not ok or not S_:
        return phase, raw, carrier, estimated, near, conf, ok
    runs = [r for r in poss['timeline'] if isinstance(r, dict) and _number(r.get('f0')) and _number(r.get('f1'))]
    runs.sort(key=lambda r: r['f0'])
    if runs:
        f0 = np.array([r['f0'] for r in runs], float)
        f1 = np.array([r['f1'] for r in runs], float)
        k = np.searchsorted(f0, frame, side='right')-1
        inside = (k >= 0) & (frame <= f1[np.clip(k, 0, len(runs)-1)])
        for s in np.flatnonzero(inside):
            r = runs[k[s]]
            if r.get('state') == 'dead':
                raw[s] = 'dead'
            elif r.get('team') in ('A', 'B') and r.get('state') != 'unknown':
                raw[s] = r['team']
                if r.get('state') == 'control' and isinstance(r.get('carrier'), str):
                    (carrier if r.get('estimated') is False else estimated)[s] = r['carrier']
    for d in poss.get('deadSpells') or ():
        if isinstance(d, dict) and _number(d.get('t0')) and _number(d.get('t1')):
            m = (t >= float(d['t0'])-1e-6) & (t <= float(d['t1'])+1e-6)
            raw[m] = 'dead'
            carrier[m] = estimated[m] = ''
    best = np.full(S_, np.inf)
    for c in poss.get('changes') or ():
        if not (isinstance(c, dict) and _number(c.get('wonAt'))):
            continue
        won = float(c['wonAt'])
        lost = float(c['lostAt']) if _number(c.get('lostAt')) else won
        m = (t >= min(lost, won-CHANGE_EXCLUDE_S)-1e-6) & (t <= won+CHANGE_EXCLUDE_S+1e-6)
        near |= m
        dist = np.abs(t-won)
        upd = m & (dist < best)
        best[upd] = dist[upd]
        conf[upd] = 'high' if c.get('confidence') == 'high' else 'uncertain'
    named = (raw == 'A') | (raw == 'B')
    keep = named & ~near
    phase[keep & (raw == team)] = 'own'
    phase[keep & (raw != team) & named] = 'opp'
    phase[raw == 'dead'] = 'dead'
    return phase, raw, carrier, estimated, near, conf, ok


def _contest_seconds(poss, t):
    """Per sample: seconds to the nearest stretch the possession engine calls the ball 'loose' (last touched by the
    team not in possession, between the loser's last touch and the winner's first: the ball between the teams) or
    'contested' (both teams at it), 0 inside one; NaN where the engine names no such stretch or is unavailable (fix
    round 3, LEAD decision (2): a 'moments worth a look' moment is never one of these, assess._clear_moments). A
    measurement of the ball's state, kept beside the phase: a coach's marks decide the phase, never this."""
    S_ = len(t)
    out = np.full(S_, np.nan)
    ok = isinstance(poss, dict) and poss.get('available') is True and isinstance(poss.get('timeline'), list)
    if not ok or not S_:
        return out
    iv = sorted((float(r['t0']), float(r['t1'])) for r in poss['timeline']
                if isinstance(r, dict) and r.get('state') in CONTEST_STATES and _number(r.get('t0'))
                and _number(r.get('t1')) and float(r['t1']) >= float(r['t0']))
    if not iv:
        return out
    a = np.array([x[0] for x in iv])
    b = np.maximum.accumulate(np.array([x[1] for x in iv]))     # runs are disjoint; this keeps it safe if not
    t = np.asarray(t, float)
    k = np.searchsorted(a, t, side='right')-1
    before = np.where(k >= 0, np.maximum(0.0, t-b[np.clip(k, 0, len(a)-1)]), np.inf)
    after = np.where(k+1 < len(a), a[np.clip(k+1, 0, len(a)-1)]-t, np.inf)
    d = np.minimum(before, after)
    out[np.isfinite(d)] = d[np.isfinite(d)]
    return out


def _runs(mask):
    """[(i0, i1)] of the true runs of a boolean array."""
    m = np.asarray(mask, bool)
    if not m.any():
        return []
    d = np.diff(np.r_[0, m.astype(np.int8), 0])
    return list(zip(np.flatnonzero(d == 1).tolist(), (np.flatnonzero(d == -1)-1).tolist()))


def _span_lengths(t, phase, step):
    """Per sample: the length of the contiguous run of its phase state (own / opp / dead), 0 where unknown."""
    out = np.zeros(len(t))
    for state in ('own', 'opp', 'dead'):
        for i0, i1 in _runs(phase == state):
            # a gap in the grid longer than 1.5 steps breaks the run
            cut = np.flatnonzero(np.diff(t[i0:i1+1]) > 1.5*step)+i0
            starts = [i0]+(cut+1).tolist()
            ends = cut.tolist()+[i1]
            for a, b in zip(starts, ends):
                out[a:b+1] = t[b]-t[a]+step
    return out


def _taint(result, ids, t):
    """(S x N) bool: the player is within TAINT_S of an identity issue (crossing, re-identification, kit change,
    uncertain swap)."""
    times = {}
    for i in result.get('issues') or ():
        if isinstance(i, dict) and isinstance(i.get('id'), str) and _number(i.get('time')) and \
                any(k in str(i.get('reason') or '').lower() for k in TAINT_ISSUES):
            times.setdefault(i['id'], []).append(float(i['time']))
    for e in result.get('events') or ():
        if isinstance(e, dict) and e.get('kind') in TAINT_EVENTS and isinstance(e.get('playerId'), str) \
                and _number(e.get('time')):
            times.setdefault(e['playerId'], []).append(float(e['time']))
    T = np.zeros((len(t), len(ids)), bool)
    for j, pid in enumerate(ids):
        arr = np.sort(np.array(times.get(pid) or [], float))
        if not len(arr):
            continue
        k = np.searchsorted(arr, t-TAINT_S, side='left')
        hit = k < len(arr)
        T[hit, j] = arr[k[hit]] <= t[hit]+TAINT_S
    return T


def _attack(xn, yn, side, L, W):
    X, Y = ts.attack_frame(xn, yn, side)
    return X*L, Y*W


def _side_sign(side):
    return np.where(side == 'left', 1.0, np.where(side == 'right', -1.0, np.nan))


def _footage(cd, frames):
    small = sum(1 for f in frames if (f.get('profile') or {}).get('smallScene'))
    share = small/len(frames) if frames else 0.0
    reg = (((cd.result or {}).get('summary') or {}).get('pitch') or {}).get('registration') or {}
    anchors = reg.get('anchors') if isinstance(reg, dict) else None
    has_cal = bool((cd.available or {}).get('calibration'))
    keyless = bool(has_cal and isinstance(anchors, list) and not any(
        isinstance(a, dict) and a.get('source') == 'landmarks' for a in anchors))
    lens = isinstance(reg, dict) and reg.get('lens') is not None
    observed = [float(np.mean(s.observed)) for s in cd.teams.values() if len(s.observed)]
    if not cd.small_scene:
        kind = 'broadcast'
    elif observed and max(observed) < FEW_PLAYERS:
        kind = 'veo'
    else:
        kind = 'amateur'
    return {'kind': kind, 'smallSceneShare': round(float(share), 3), 'keyless': keyless, 'lensBent': bool(lens)}


def _formation(shape, team, t, fmt):
    """Per sample period index / effective template label / confidence / changeNear, the period summaries, and the
    role-input windows and periods (11v11 only)."""
    S_ = len(t)
    idx = np.full(S_, -1, np.int32)
    label = np.full(S_, '', dtype='<U8')
    conf = np.full(S_, '', dtype='<U10')
    near = np.zeros(S_, bool)
    if fmt['players'] != 11 or shape is None or team not in getattr(shape, 'shapes', {}):
        return idx, label, conf, near, [], [], []
    F = shape.shapes[team]
    periods = F.get('periods') or []
    eff = []
    for p in periods:
        lab = p.get('label')
        corr = (p.get('correction') or {}).get('formation') if p.get('correctedRoles') else None
        e = corr if corr in ts.TEMPLATES else (lab if lab in ts.TEMPLATES else None)
        c = p.get('confidence') if e is not None else ('unclear' if lab == 'unclear' else None)
        eff.append((e, c, bool(corr)))
    if periods:
        starts = np.array([float(p['t0']) for p in periods])
        k = np.clip(np.searchsorted(starts, t+1e-6, side='right')-1, 0, len(periods)-1)
        for s in range(S_):
            p = periods[k[s]]
            if float(p['t0'])-1e-6 <= t[s] <= float(p['t1'])+1e-6:
                idx[s] = k[s]
                label[s] = eff[k[s]][0] or ''
                conf[s] = eff[k[s]][1] or ''
    # Runs of one template label (transition and data-gap periods between them skipped); a change between two runs
    # suppresses role-based criticism within FORMATION_CHANGE_S only when both runs hold FORMATION_PERSIST_S and read
    # clear / likely (or the coach set them), and never when the labels change more than FORMATION_MAX_RATE times per
    # 10 minutes of the segment (fix round 1, critic S-3: on the simulated camera the label changed 18.8 times per
    # 10-minute clip against 4.6 on full tracking and the old any-change rule flagged 55-57% of the D3 opportunities,
    # 8% on full tracking; the broadcast BC303 B had 3 periods in 81 s).
    runs = []
    for i in range(len(periods)):
        lab = eff[i][0]
        if lab is None:
            continue
        sure = eff[i][1] in FORMATION_CONFIDENT or eff[i][2]
        if runs and runs[-1][0] == lab:
            runs[-1][2] = float(periods[i]['t1'])
            runs[-1][3] = runs[-1][3] or sure
        else:
            runs.append([lab, float(periods[i]['t0']), float(periods[i]['t1']), sure])
    seg_s = float(t[-1]-t[0]) if S_ else 0.0
    churn = len(runs) > 1 and seg_s > 0 and (len(runs)-1)*600.0/seg_s > FORMATION_MAX_RATE+1e-9
    for a, b in zip(runs, runs[1:]):
        held = a[2]-a[1] >= FORMATION_PERSIST_S-1e-6 and b[2]-b[1] >= FORMATION_PERSIST_S-1e-6
        if churn or not held or not (a[3] and b[3]):
            continue
        lo, hi = min(a[2], b[1]), max(a[2], b[1])
        near |= (t >= lo-FORMATION_CHANGE_S-1e-6) & (t <= hi+FORMATION_CHANGE_S+1e-6)
    summaries = [{'index': i, 't0': round(float(p['t0']), 2), 't1': round(float(p['t1']), 2), 'label': eff[i][0],
                  'confidence': eff[i][1]} for i, p in enumerate(periods)]
    # Role-input windows: team_shape's classified windows, roles under the deciding formation and the runner-up.
    windows = F.get('windows') or []
    labels = F.get('labels') or []
    out_w = []
    for i, w in enumerate(windows):
        if w.get('status') != 'ok' or '_fits' not in w:
            continue
        c = float(w['centre'])
        kp = next((k_ for k_, p in enumerate(periods) if float(p['t0'])-1e-6 <= c <= float(p['t1'])+1e-6), None)
        decides = None
        if kp is not None and eff[kp][0] in w['_fits']:
            decides = eff[kp][0]
        elif i < len(labels) and labels[i] in w['_fits']:
            decides = labels[i]
        if decides is None:
            continue
        tpl = ts.TEMPLATES[decides]
        fit = w['_fits'][decides]
        roles = {pid: tpl['roles'][r] for pid, r in zip(w['ids'], fit['roles'])}
        alt_name = periods[kp].get('alternative') if kp is not None else None
        alt = {}
        if alt_name in w['_fits']:
            atpl = ts.TEMPLATES[alt_name]
            alt = {pid: atpl['roles'][r] for pid, r in zip(w['ids'], w['_fits'][alt_name]['roles'])}
        rows = w.get('_rows')
        out_w.append({'centre': round(c, 2), 't0': round(float(w['t0']), 2), 't1': round(float(w['t1']), 2),
                      'period': kp, 'label': decides, 'alternative': alt_name if alt else None, 'roles': roles,
                      'alt': alt,
                      '_rows': np.asarray(rows, int) if rows is not None else np.zeros(0, int)})
    out_p = [{'index': i, 't0': round(float(p['t0']), 2), 't1': round(float(p['t1']), 2), 'label': eff[i][0],
              'rawLabel': _ascii(p.get('label'), 20), 'confidence': eff[i][1], 'corrected': eff[i][2],
              'alternative': p.get('alternative') if p.get('alternative') in ts.TEMPLATES else None,
              'related': bool(p.get('related')), 'margin': _r(p.get('margin'), 4),
              'roleStability': _r(p.get('roleStability'), 3)} for i, p in enumerate(periods)]
    return idx, label, conf, near, summaries, out_w, out_p


def _dominant_cause(lineup, in_kit, confirmed, positioned):
    if lineup is None or positioned >= DQ_OK_SHARE*lineup:
        return None
    losses = [('out_of_view', lineup-in_kit), ('identity', in_kit-confirmed), ('registration', confirmed-positioned)]
    cause, loss = max(losses, key=lambda c: c[1])
    return cause if loss > 0 else None


def _track(t, X, Y, ok, phase, step=TRACK_STEP_S, half=TRACK_HALF_S, min_n=TRACK_MIN_SAMPLES):
    """The 1 Hz track: times, per player the mean attack-frame position of his usable samples within `half` s of each
    point (None with fewer than min_n), and the phase of the nearest sample."""
    if not len(t):
        return {'t': [], 'x': [], 'y': [], 'phase': []}
    t0 = float(t[0])
    k_last = int(math.floor((float(t[-1])-t0)/step+1e-9))
    times = t0+step*np.arange(k_last+1)
    pos = np.clip(np.searchsorted(t, times), 0, len(t)-1)
    prev = np.clip(pos-1, 0, len(t)-1)
    nearest = np.where(np.abs(t[prev]-times) <= np.abs(t[pos]-times), prev, pos)
    near_k = np.rint((t-t0)/step).astype(int)
    close = np.abs(t-(t0+near_k*step)) <= half+1e-6
    K, N = len(times), X.shape[1]
    sx, sy, n = np.zeros((K, N)), np.zeros((K, N)), np.zeros((K, N))
    use = ok & close[:, None] & np.isfinite(X) & np.isfinite(Y)
    rows, cols = np.nonzero(use)
    kk = near_k[rows]
    inside = (kk >= 0) & (kk < K)
    np.add.at(sx, (kk[inside], cols[inside]), X[rows[inside], cols[inside]])
    np.add.at(sy, (kk[inside], cols[inside]), Y[rows[inside], cols[inside]])
    np.add.at(n, (kk[inside], cols[inside]), 1)
    with np.errstate(invalid='ignore', divide='ignore'):
        mx, my = sx/n, sy/n
    good = n >= min_n
    xs = [[round(float(mx[k, j]), 1) if good[k, j] else None for k in range(K)] for j in range(N)]
    ys = [[round(float(my[k, j]), 1) if good[k, j] else None for k in range(K)] for j in range(N)]
    return {'t': [round(float(v), 2) for v in times], 'x': xs, 'y': ys, 'phase': [str(phase[i]) for i in nearest]}


def build_timeline(cd, possession, team, measurement, coach_spans=None):
    """The situation timeline of `team` ('A' or 'B') from the coaching series cd (series.build_series), the possession-4
    record of the same result (soccer.possession.compute; None = unavailable), the measurement settings {format,
    field} (schema.tier_settings(...)['measurement']) and the coach-marked spans (schema 'coach_spans' record or a list
    of spans; also read from measurement['coachSpans'] when present). Returns a SituationTimeline (`summary()`: schema
    'timeline'; `role_inputs()`: obs['roleInputs']['context'])."""
    if team not in ('A', 'B'):
        raise ValueError('team must be A or B')
    other = 'B' if team == 'A' else 'A'
    if coach_spans is None and isinstance(measurement, dict):
        coach_spans = measurement.get('coachSpans')
    fmt = _format(measurement)
    field = measurement.get('field') if isinstance(measurement, dict) else None
    tl = SituationTimeline()
    tl.team, tl.other, tl.format, tl.cd = team, other, fmt, cd
    result = cd.result or {}
    frames = result.get('frames') or []
    t = np.asarray(cd.t, float)
    frame = np.asarray(cd.frame, np.int64)
    S_ = len(t)
    step = float(cd.step)
    fps = float(cd.fps or 0.0)
    tl.step, tl.fps = step, fps
    scale = scale_of(cd, fmt, field)
    tl.scale = scale
    L, W = scale['lengthM'], scale['widthM']
    rl = float((cd.pitch or {}).get('length') or 105.0)
    rw = float((cd.pitch or {}).get('width') or 68.0)
    me, op = cd.teams[team], cd.teams[other]
    tl.series, tl.oppSeries = me, op
    side = np.asarray(me.side).astype('<U5') if S_ else np.zeros(0, '<U5')
    oside = np.asarray(op.side).astype('<U5') if S_ else np.zeros(0, '<U5')
    data = cd.shape.observed if cd.shape is not None else None
    gk_teams = {}
    if data is not None:
        for tm in ('A', 'B'):
            for g in data.teams[tm].get('gks') or ():
                gk_teams[g] = tm

    # Registration gate (series' frame gates) as one code per sample.
    reg = np.full(S_, '', dtype=cs.REASON_DTYPE)
    cal = cd.calibration or {}
    if S_ and cal:
        present = np.asarray(cal.get('present'), bool)
        conf_c = np.asarray(cal.get('confidence'), float)
        dis = np.asarray(cal.get('disagreement'), float)
        calibrated = np.asarray(cal.get('calibrated'), bool)
        with np.errstate(invalid='ignore'):
            low = present & calibrated & (conf_c < cs.MIN_CALIBRATION)
            disagree = np.isfinite(dis) & (dis > cs.MAX_DISAGREEMENT)
        reg = cs._gates(S_, [(~calibrated, cs.NOT_CALIBRATED), (low, cs.LOW_CALIBRATION), (disagree, cs.DISAGREEMENT)])
    frame_ok = reg == ''

    # Players of both teams.
    def players(series, tm, sd):
        ids = list(series.ids)
        N = len(ids)
        x = np.asarray(series.values['x'], float).reshape(S_, N)
        y = np.asarray(series.values['y'], float).reshape(S_, N)
        obs = np.isfinite(x) & np.isfinite(y)
        d = data.teams[tm] if data is not None else {}
        sigma = np.asarray(d.get('sigma'), float).reshape(S_, N) if data is not None else np.full((S_, N), np.nan)
        ident = np.asarray(d.get('identity'), float).reshape(S_, N) if data is not None else np.zeros((S_, N))
        with np.errstate(invalid='ignore'):
            pos_ok = obs & frame_ok[:, None] & ~(np.isfinite(sigma) & (sigma > MAX_SIGMA_M))
        X, Y = _attack(x, y, side[:, None], L, W)
        X, Y = np.where(obs, X, np.nan), np.where(obs, Y, np.nan)
        vx = np.asarray(series.values['velX'], float).reshape(S_, N)*(L/rl)
        vy = np.asarray(series.values['velY'], float).reshape(S_, N)*(W/rw)
        vok = np.asarray(series.eligible['velocity'], bool).reshape(S_, N)
        if tm != team:      # the opponents' velocity in this team's frame
            f = (_side_sign(side)*_side_sign(sd))[:, None]
            vx, vy = vx*f, vy*f
            vok = vok & np.isfinite(f)
        vx, vy = np.where(vok, vx, np.nan), np.where(vok, vy, np.nan)
        gk = np.asarray(d.get('gk'), float).reshape(S_, 2) if data is not None else np.full((S_, 2), np.nan)
        gX, gY = _attack(gk[:, 0], gk[:, 1], side, L, W)
        return {'ids': ids, 'col': {pid: j for j, pid in enumerate(ids)}, 'x': np.where(obs, x, np.nan),
                'y': np.where(obs, y, np.nan), 'X': X, 'Y': Y, 'sigma': sigma, 'identity': ident,
                'tainted': _taint(result, ids, t), 'posOk': pos_ok, 'velX': vx, 'velY': vy,
                'speed': np.hypot(vx, vy), 'velOk': vok & np.isfinite(vx), 'gk': gk, 'gkX': gX, 'gkY': gY,
                'observedCount': pos_ok.sum(axis=1).astype(int) if N else np.zeros(S_, int)}
    mine, theirs = players(me, team, side), players(op, other, oside)
    tl.ids, tl.col = mine['ids'], mine['col']
    tl.oppIds, tl.oppCol = theirs['ids'], theirs['col']
    for k in ('x', 'y', 'X', 'Y', 'sigma', 'identity', 'tainted', 'posOk', 'velX', 'velY', 'speed', 'velOk', 'gk',
              'gkX', 'gkY', 'observedCount'):
        tl._put(**{k: mine[k], OPP_NAME.get(k, 'opp'+k[0].upper()+k[1:]): theirs[k]})
    tl._put(stabFlag=np.asarray(me.values['flagged'], bool).reshape(S_, len(tl.ids)) if S_ else
            np.zeros((0, len(tl.ids)), bool))

    def lineup(series):
        if fmt['players'] == 11:
            return np.clip(np.asarray(series.expected, int), 1, 10) if S_ else np.zeros(0, int)
        return np.full(S_, fmt['outfield'], int)
    outfield, opp_outfield = lineup(me), lineup(op)
    complete = frame_ok & (tl.observedCount == outfield) & (outfield > 0)
    tl._put(t=t, frame=frame, side=side, regGate=reg, outfield=outfield, complete=complete,
            excess=tl.observedCount > outfield,
            oppComplete=frame_ok & (tl.oppObservedCount == opp_outfield) & (opp_outfield > 0),
            unplaced=np.asarray(cd.unplaced.get(team, np.zeros(S_, int)), int),
            oppUnplaced=np.asarray(cd.unplaced.get(other, np.zeros(S_, int)), int),
            inKit=np.asarray(cd.in_kit.get(team, np.zeros(S_, int)), int),
            confirmedInKit=np.asarray(cd.confirmed_in_kit.get(team, np.zeros(S_, int)), int))
    views = []
    for v in (cd.view or [None]*S_):
        views.append(v if v is None or v is False else np.asarray(v, float)/np.array([rl, rw])*np.array([L, W]))
    tl.view = views

    # Ball (series gates; zones on the coaching scale).
    bx = np.asarray((cd.ball or {}).get('x', np.full(S_, np.nan)), float)
    by = np.asarray((cd.ball or {}).get('y', np.full(S_, np.nan)), float)
    ball_ok = np.asarray(me.eligible['ball'], bool) if S_ else np.zeros(0, bool)
    bX, bY = _attack(bx, by, side, L, W)
    zone_ok = ball_ok & np.isfinite(bX)
    third = cs._zones(t, bX, zone_ok, (L/3, 2*L/3), ('own', 'middle', 'final')) if S_ else np.zeros(0, '<U6')
    third = np.array([ZONE_OF.get(str(z), '') for z in third], dtype='<U12')
    channel = cs._zones(t, bY, zone_ok, (W/3, 2*W/3), CHANNEL_NAMES) if S_ else np.zeros(0, '<U6')
    tl._put(ballSigma=_ball_sigma(cd, frames, S_))
    tl._put(ballX=np.where(ball_ok, bx, np.nan), ballY=np.where(ball_ok, by, np.nan),
            ballAX=np.where(zone_ok, bX, np.nan), ballAY=np.where(zone_ok, bY, np.nan), ballReliable=ball_ok,
            ballThird=np.asarray(third, dtype='<U12'), ballChannel=np.asarray(channel, dtype='<U6'))

    # Phase: possession-4 under the binding rules, coach spans over it.
    phase, raw, carrier, est_carrier, near, cconf, available = _engine_phase(possession, frame, t, team)
    source = np.where(phase == 'unknown', NONE, ENGINE).astype(SOURCE_DTYPE)
    seg_t0 = float(t[0]) if S_ else 0.0
    seg_t1 = float(t[-1]) if S_ else 0.0
    kept, dropped = clean_coach_spans(coach_spans, seg_t0, seg_t1)
    coach_mask = np.zeros(S_, bool)
    engine_named_in = agree_in = 0
    for sp in kept:
        # [t0, t1): two spans marked end to end do not share their boundary instant; the last one reaches the end
        end = (t <= sp['t1']+1e-6) if sp['t1'] >= seg_t1-1e-6 else (t < sp['t1']-1e-6)
        m = (t >= sp['t0']-1e-6) & end
        state = 'dead' if sp['state'] == 'dead' else ('own' if sp['state'] == team else 'opp')
        eng = m & ((raw == 'A') | (raw == 'B')) & ~near
        engine_named_in += int(eng.sum())
        agree_in += int((eng & (raw == sp['state'])).sum())
        phase[m], source[m] = state, COACH
        near[m] = False
        cconf[m] = ''
        coach_mask |= m
    # +-CHANGE_EXCLUDE_S around every coach-marked change between the two teams' spans (fix round 1, critic S-16 /
    # FACTS L10: on the amateur clip with marks 12 opportunity samples lay within 2 s of a marked change): the phase
    # stays the coach's, the samples only SUPPRESS (contextual: 'near_change'), as around the engine's changes. Spans
    # of A and B count as one change when the gap between them is at most 2 x CHANGE_EXCLUDE_S; a dead span between
    # them is a restart, not a change.
    coach_near = np.zeros(S_, bool)
    for a, b in zip(kept, kept[1:]):
        if {a['state'], b['state']} == {'A', 'B'} and b['t0']-a['t1'] <= 2*CHANGE_EXCLUDE_S+1e-6:
            coach_near |= (t >= a['t1']-CHANGE_EXCLUDE_S-1e-6) & (t <= b['t0']+CHANGE_EXCLUDE_S+1e-6)
    if kept or dropped:
        coach = {'spans': len(kept), 'dropped': int(dropped), 'seconds': round(float(coach_mask.sum())*step, 2),
                 'own': round(float((coach_mask & (phase == 'own')).sum())*step, 2),
                 'opp': round(float((coach_mask & (phase == 'opp')).sum())*step, 2),
                 'dead': round(float((coach_mask & (phase == 'dead')).sum())*step, 2),
                 'engineNamedSeconds': round(engine_named_in*step, 2),
                 'agreement': round(agree_in/engine_named_in, 3) if engine_named_in else None}
    else:
        coach = None
    # The carrier only where the phase names his team.
    def agrees(ids):
        tm = np.array([team_of(c, gk_teams) or '' for c in ids], dtype='<U1') if S_ else np.zeros(0, '<U1')
        return ((phase == 'own') & (tm == team)) | ((phase == 'opp') & (tm == other)), tm
    consistent, cteam = agrees(carrier)
    carrier = np.where(consistent, carrier, '')
    cteam = np.where(consistent, cteam, '')
    est_carrier = np.where(agrees(est_carrier)[0], est_carrier, '')
    cX, cY = np.full(S_, np.nan), np.full(S_, np.nan)
    for s in np.flatnonzero(carrier != ''):
        pid = str(carrier[s])
        if pid in tl.col:
            cX[s], cY[s] = tl.X[s, tl.col[pid]], tl.Y[s, tl.col[pid]]
        elif pid in tl.oppCol:
            cX[s], cY[s] = tl.oppX[s, tl.oppCol[pid]], tl.oppY[s, tl.oppCol[pid]]
        elif gk_teams.get(pid) == team:
            cX[s], cY[s] = tl.gkX[s], tl.gkY[s]
        elif gk_teams.get(pid) == other:
            cX[s], cY[s] = tl.oppGkX[s], tl.oppGkY[s]
    tl._put(phase=phase, phaseSource=source, phaseSpan=_span_lengths(t, phase, step), nearChange=near,
            changeConfidence=cconf, carrier=carrier.astype('<U8'), carrierTeam=cteam, carrierX=cX, carrierY=cY,
            estimatedCarrier=est_carrier.astype('<U8'),
            coachMarked=coach_mask, coachNear=coach_near, contestS=_contest_seconds(possession, t))
    tl._raw_engine = raw

    # Formation periods (11v11 only).
    fidx, flabel, fconf, fnear, fsummaries, fwindows, fperiods = _formation(cd.shape, team, t, fmt)
    tl._put(formationPeriod=fidx, formationLabel=flabel, formationConfidence=fconf, formationChangeNear=fnear)

    tl._summary = _summarize(tl, possession, available, coach, kept, dropped, frames, fsummaries)
    tl._role_inputs = _role_inputs(tl, fwindows, fperiods)
    return tl


def _phase_spans(tl):
    out = []
    t, frame, phase, source = tl.t, tl.frame, tl.phase, tl.phaseSource
    for state in ('own', 'opp', 'dead'):
        for src in (ENGINE, COACH):
            for i0, i1 in _runs((phase == state) & (source == src)):
                out.append({'state': state, 't0': round(float(t[i0]), 3), 't1': round(float(t[i1]), 3),
                            'frame0': int(frame[i0]), 'frame1': int(frame[i1]), 'source': src})
    out.sort(key=lambda s: s['t0'])
    return out


def one_sided(coach, available, team):
    """The team letter whose possession alone the coach marked, when the engine is unavailable and the marks name
    only one team (critic N6: the overview then read "A 1.0, B 0.0" and the defending checks said phase_unknown); None
    otherwise. coach: the timeline's phaseQuality.coach of `team` (own / opp seconds from that team's view)."""
    if available or not coach:
        return None
    own, opp = float(coach.get('own') or 0.0), float(coach.get('opp') or 0.0)
    if (own > 0.0) == (opp > 0.0):
        return None
    return team if own > 0.0 else ('B' if team == 'A' else 'A')


def _summarize(tl, poss, available, coach, kept, dropped, frames, fsummaries):
    S_ = len(tl.t)
    step = tl.step
    phase = tl.phase
    sec = {k: round(float((phase == k).sum())*step, 2) for k in ('own', 'opp', 'dead', 'unknown')}
    sec['nearChange'] = round(float(tl.nearChange.sum())*step, 2)
    poss = poss if isinstance(poss, dict) else {}
    q = poss.get('quality') if isinstance(poss.get('quality'), dict) else {}
    level = q.get('level') if q.get('level') in ('good', 'limited', 'unavailable') else 'unavailable'
    cover = q.get('coverage') if _number(q.get('coverage')) else None
    shares = poss.get('shares') or {}
    rough = None
    if available and coach is not None and coach['spans'] and sec['own']+sec['opp'] > 0:
        # The coach marked spans: the shares are those of the merged per-sample phase (the coach's spans where
        # marked, the engine elsewhere), so the overview never contradicts the marks (fix round 1, critic B-11 / CODE
        # H2: Team A 5-25 s and Team B 25-33.5 s marked on BC60 read "A about 0%, B about 100%" from the engine).
        x = round(sec['own']/(sec['own']+sec['opp'])/ROUGH_SHARE_STEP)*ROUGH_SHARE_STEP
        rough = {tl.team: round(x, 2), tl.other: round(1.0-x, 2)}
        rough = {'A': rough['A'], 'B': rough['B']}
    elif available and _number(shares.get('A')) and _number(shares.get('B')) and shares['A']+shares['B'] > 0:
        a = round(shares['A']/(shares['A']+shares['B'])/ROUGH_SHARE_STEP)*ROUGH_SHARE_STEP
        rough = {'A': round(a, 2), 'B': round(1.0-a, 2)}
    footage = _footage(tl.cd, frames)
    notes = []
    changes = [c for c in poss.get('changes') or () if isinstance(c, dict)]
    if available:
        if changes:
            notes.append(f'Who had the ball is estimated by the possession engine; the {CHANGE_EXCLUDE_S:.0f} s either '
                         f'side of each of its {len(changes)} changes of possession are left out '
                         f'({sec["nearChange"]:.0f} s).')
        else:
            notes.append('Who had the ball is estimated by the possession engine (it found no change of possession).')
        if level == 'limited' and cover is not None:
            notes.append(f'Possession quality is limited on this run (coverage {cover*100:.0f}%).')
        if footage['kind'] == 'broadcast':
            notes.append(STOPPAGE_NOTE)
    else:
        notes.append('The possession engine did not name who had the ball on this footage, so checks that need the '
                     'phase are not assessed unless the coach marks who had the ball.')
    if coach is not None:
        if coach['spans']:
            notes.append(f'{coach["spans"]} coach-marked spans decide the phase for {coach["seconds"]:.0f} s'
                         + ('; elsewhere the estimate is used.' if available else '.'))
        if dropped:
            notes.append(f'{dropped} coach-marked spans were not used (outside the analysed video, overlapping or '
                         'incomplete).')
        side = one_sided(coach, available, tl.team)
        if side is not None:
            missing = 'B' if side == 'A' else 'A'
            notes.append(f'You marked only Team {side}\'s possession; mark Team {missing}\'s spells (or dead balls) '
                         f'too to run the checks that need Team {missing} on the ball.')
    reason = _ascii(poss.get('reason'), 300) if not available else None
    if not available and reason is None:
        reason = 'No possession record for this run.'
    quality = {'available': bool(available), 'reason': reason, 'level': level if available else 'unavailable',
               'coverage': round(float(cover), 3) if cover is not None else None,
               'roughShares': rough, 'notes': [n[:300] for n in notes], 'coach': coach}
    lineup = int(np.bincount(tl.outfield).argmax()) if S_ and len(tl.outfield) else None
    in_kit = float(np.mean(tl.inKit)) if S_ else None
    conf = float(np.mean(tl.confirmedInKit)) if S_ else None
    posd = float(np.mean(tl.observedCount)) if S_ else None
    dq = {'inKit': _r(in_kit, 1), 'confirmed': _r(conf, 1), 'positioned': _r(posd, 1),
          'lineup': max(1, min(11, lineup)) if lineup is not None else None,
          'dominantCause': _dominant_cause(lineup, in_kit, conf, posd) if S_ else None}
    opp_n = int((phase == 'opp').sum())
    carrier_share = round(float(((phase == 'opp') & (tl.carrier != '')).sum())/opp_n, 3) if opp_n else None
    players = {}
    for prefix, ids, team in (('', tl.ids, tl.team), ('opp', tl.oppIds, tl.other)):
        ok = tl.posOk if not prefix else tl.oppPosOk
        speed = tl.speed if not prefix else tl.oppSpeed
        vok = tl.velOk if not prefix else tl.oppVelOk
        for j, pid in enumerate(ids):
            if not re.fullmatch(S.PLAYER_ID, str(pid)):
                continue
            v = speed[:, j][vok[:, j]]
            players[pid] = {'team': team, 'goalkeeper': False,
                            'observedSeconds': round(float(ok[:, j].sum())*step, 2),
                            'p90SpeedMS': round(min(float(np.percentile(v, 90)), 15.0), 2)
                            if len(v) >= P90_MIN_SAMPLES else None}
    data = tl.cd.shape.observed if tl.cd.shape is not None else None
    if data is not None:
        for team, key in ((tl.team, 'gk'), (tl.other, 'oppGk')):
            gks = data.teams[team].get('gks') or []
            if len(gks) == 1 and re.fullmatch(S.PLAYER_ID, str(gks[0])):
                g = getattr(tl, key)
                players[gks[0]] = {'team': team, 'goalkeeper': True,
                                   'observedSeconds': round(float(np.isfinite(g[:, 0]).sum())*step, 2),
                                   'p90SpeedMS': None}
    return {'team': tl.team, 'samples': int(S_), 'step': float(step),
            't0': round(float(tl.t[0]), 3) if S_ else 0.0, 't1': round(float(tl.t[-1]), 3) if S_ else 0.0,
            'frame0': int(tl.frame[0]) if S_ else 0, 'frame1': int(tl.frame[-1]) if S_ else 0,
            'fps': float(tl.fps), 'format': tl.format['name'], 'phaseSeconds': sec, 'phaseSpans': _phase_spans(tl),
            'phaseQuality': quality, 'footage': footage, 'dataQuality': dq, 'scale': dict(tl.scale),
            'completeShare': round(float(tl.complete.mean()), 3) if S_ else 0.0,
            'observedMean': round(float(np.mean(tl.observedCount)), 2) if S_ else 0.0,
            'ballReliableShare': round(float(tl.ballReliable.mean()), 3) if S_ else 0.0,
            'carrierShare': carrier_share, 'formationPeriods': fsummaries, 'players': players}


def _role_inputs(tl, windows, periods):
    """obs['roleInputs']['context'] (module docstring "Role inputs")."""
    t = tl.t
    phase = tl.phase
    out_w = []
    for w in windows:
        rows = w.pop('_rows')
        rows = rows[(rows >= 0) & (rows < len(t))] if len(rows) else rows
        n = len(rows)
        own = float((phase[rows] == 'own').sum())/n if n else 0.0
        opp = float((phase[rows] == 'opp').sum())/n if n else 0.0
        out_w.append(dict(w, own=round(own, 3), opp=round(opp, 3), samples=int(n)))
    ok = tl.posOk & np.isfinite(tl.X)
    track = _track(t, tl.X, tl.Y, ok, phase)
    complete_k = []
    if track['t']:
        tt = np.asarray(track['t'])
        pos = np.clip(np.searchsorted(t, tt), 0, len(t)-1)
        complete_k = [bool(tl.complete[i]) for i in pos]
    return {'version': CONTEXT_VERSION, 'team': tl.team, 'format': tl.format['name'],
            'outfield': int(tl.format['outfield']), 'scale': dict(tl.scale), 'ids': list(tl.ids),
            'segment': {'t0': round(float(t[0]), 3) if len(t) else 0.0, 't1': round(float(t[-1]), 3) if len(t) else 0.0},
            'track': dict(track, step=TRACK_STEP_S, complete=complete_k), 'windows': out_w, 'periods': periods}


__all__ = ['build_timeline', 'SituationTimeline', 'spells', 'distinct', 'rate_per_10min', 'short_range',
           'clean_coach_spans', 'scale_of', 'team_of', 'one_sided', 'CHANGE_EXCLUDE_S', 'MIN_SHAPE_SPAN_S', 'SPELL_MIN_S',
           'SPELL_MERGE_S', 'DISTINCT_S', 'CONTEXT_VERSION']
