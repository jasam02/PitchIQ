"""Synthetic coaching scenes (DESIGN 13.1): deterministic PitchIQ-shaped results whose tactical truth is known, so the
coaching v2 pipeline (series -> situations -> detectors -> assess -> report) can be tested on counterexamples no real
clip provides.

    import sys; sys.path.insert(0, '<repo>/tools')
    import coaching_synth as cs
    built = cs.build('unsupported_advance', 'none_behind')     # a Built: .result, .truth, .expected, .coach_spans
    settings = cs.settings_for(built, level=4)                 # a schema-valid coaching_settings.json for that scene
    cs.names()                                                 # [(scene, variant), ...] of the catalogue

    python tools/coaching_synth.py list
    python tools/coaching_synth.py check [SCENE[/VARIANT] ...]     # load through analytics, team shape, possession, series
    python tools/coaching_synth.py write SCENE[/VARIANT] --out DIR  # result.json, truth.json, coaching_settings.json, ...
    python tools/coaching_synth.py catalogue                        # the scene catalogue (Markdown) on stdout

Why synthetic scenes, and why built this way:
- The four real runs cannot show most counterexamples of the spec (Part 16): a fullback advancing with and without
  cover in the same match, a formation change, a crossing at the evidence moment, a 4v4 swarm, an amateur clip with
  coach-marked possession. A scene plants each situation with a known truth (truth.json) and a stated expectation
  (DESIGN 13.1, the `expected` list), so a detector's decision can be checked against what was planted, not against
  what the detector itself thinks.
- They run through the REAL pipeline. A scene is a result.json dict with everything soccer/analytics.py,
  soccer/team_shape.py, soccer/possession.py and soccer/coaching/series.py read: frames with people (box, pitch,
  confidences, identities), a ball as soccer/ball_path.py's final path records it (TRACKED or UNKNOWN, origin,
  motion, speed, nearTrack, image box and pitch position), the final registration (calibration.H, confidence, disagreement),
  the camera's view polygon, paint ticks (frames[].pitch.markings) for the evidence's world tier, the small-scene
  profile, the roster, goal ends and pitch size, and the identity issues and events the detectors' identity taint
  reads. Nothing in a scene is a stand-in for a stage: possession-4 decodes the ball itself.
- The camera is affine (tests/test_possession.py's convention): x pixels = u0 + px * metres along the pitch, y pixels
  = v0 + px / 2 * metres across (rows foreshortened 2x), people are 1.8 m tall boxes at the across scale, so the image
  metric of ball_path.within_reach measures the same metres as the pitch and possession's pitch-vs-image agreement is
  the scene's, not a projection artefact. calibration.H is that affine map (field.orient accepts it), and the paint
  ticks are the predicted markings themselves (residual 0 px), so a HIGH registration is a world-tier frame.
- Players move as a team: each team holds a formation (hand-placed layouts, deliberately not team_shape's templates,
  from tests/team_shape_synth.py) whose block follows the ball (deeper and narrower when defending, higher and wider
  in possession), smoothed so a block never moves faster than BLOCK_VMAX, with a slow per-player wander. Situations
  are PLANTS on top: a wide defender's advance (with or without each cover test of DESIGN 4.3), the team pushed up so
  that nobody (or one player) is behind the ball, an opponent left in the vacated channel, an identity crossing, a
  sprint back after a change, a teammate standing where a cover zone would be (stacking), pressure and cover on the
  ball carrier, an emptied weak side, an open ball-side gap. Every plant blends in and out with smoothstep ramps.
- The ball is scripted from possession spans: controls at a carrier's feet (BALL_AHEAD in front of him), passes at
  PASS_SPEED between team-mates, a live change as a pass the other team's nearest player intercepts, an out-of-play
  spell as a kick over the nearest touchline, the ball unseen and then placed on the line while the restarting team's
  player walks to it. Carriers are chosen among the players no plant moves at that time, so a plant never drags the
  ball and the ball never drags a plant.
- Deterministic: numpy Generators seeded by the scene's name and seed (one per purpose, so twins share their motion),
  no clocks, no dict-order effects; the same call gives the same bytes (tested). FPS is 10 by default (PitchIQ writes
  25-30): the consumers work in seconds (team shape samples at 5 Hz), and a 6-minute 11v11 scene stays at 3,660 frames
  (about 80,000 people records). At fps=30 possession-4 reads the same scenes as well (agreement 1.0 at coverage
  .94-.96 against .995 at 10 fps; every change found).
- The truth is scripted, not measured by the coaching code: truth.json holds the possession spans, the changes, each
  plant's interval and parameters, and relations recomputed by this module's own simple geometry on the true
  (noise-free) positions at 5 Hz (advance margin over the centre-back line, the cover tests, players behind the ball,
  opponents in the channel, the goalwards speed, the crossing distance, the team-mates in view, the maximum speeds),
  so a scene that no longer plants what it claims fails its own test before any detector is blamed.

What a scene does NOT simulate: detection misses inside the view (everyone in the image is detected), registration
drift (positions carry AR(1) noise of NOISE_SD only), identity switches (a crossing is reported, never executed), team
errors, referees and staff, replays and cuts, lens bend in the image (a Veo-like scene only declares one), bridged
(MISSING) ball frames, the ball tracker's first-pass and candidate records, the stabilized 2D stream, and the imputation
timeline block (TeamShape's estimated mode has no estimates). Small-sided fields are written as PitchIQ
would write them: normalized on the real field with tactical.pitchSize 105 x 68 'standard', because the pitch model
knows only full-size markings (DESIGN 2.5); the field's real size is in truth.json and in the coach's settings.

Measured on 2026-10-22 (main venv, no GPU; `python tools/coaching_synth.py check --all` reproduces the consumer numbers,
truth.json the plant numbers; the table per scene is in tools/annotations/coaching/synthetic_scenes.md):
- possession-4 on the broadcast-like scenes: available, quality good, the scripted team on .998-1.000 of the frames it
  names, coverage .972-.997 of the live frames (.821 in the zoomed view, where the ball leaves the picture), every
  scripted change found (15 of 15 per 6-minute scene, 23 of 23 in sprint_back, 1 of 1 in the 60 s segment). It
  abstains where a scene says it must: the sparse ball ("placed on only 12% of the frames"), the small-scene footage
  (100% of the frames), and the 4v4 field written as 105 x 68 (the pitch and the image disagree: distances read 3.5x too
  long); on the same field written at its real size it is available (.983 at coverage .70: a swarm is contested).
- team_shape reads the scripted shape on every full-view 11v11 scene (4-4-2, 4-3-3, 3-5-2, and the 4-4-2 -> 3-5-2
  change as two periods), 'insufficient' in the zoomed view (a median 3 of A's 10 outfield players in it), 'lineup' on
  the small formats (it has no 6- or 3-outfield templates).
- The evidence tier (soccer/coaching/evidence.Frames) is world on the broadcast scenes (paint residual .05 px over 17
  matched segments) and pinned under the .55 registration, on the amateur clip and in the zoomed view (2 paint segments
  in view, world needs 3).
- The plants hold on the true positions (truth.json): covered spells pass a cover test on 94-100% of their advanced
  own-possession samples, uncovered ones on 0-6% (20% in the formation change's first wing-back spell, during its
  ramps); the planted number behind the ball on 78-100% (median 100%); the opponent in the channel on 100%; 5.2-6.8 s
  advanced in own possession per short spell, 28-30 s per held wing-back spell, 0.2 s in sprint_back (inside the change
  windows), where the left-back then runs back at up to 5.8-7.2 m/s; crossings at .3 m. The defending variants press
  the carrier on 96% of the settled samples (1% for 'no press'), cover 95% of the pressed ones (12% for 'no cover'),
  empty the weak side on 100% and open the ball-side gap on 95% of the settled wide-ball samples. No player moves faster
  than 8.5 m/s (SPEED_CAP), no ball faster than about 25 m/s.
- A scene builds in .1-.6 s (about 2.5 s with the zoomed camera's per-tick paint) and loads through the four consumers
  in .4-3 s.

Fix round 1 (2026-10-23; added by DETECT/ASSESS while EVAL had not started): the coach verifier's scenes
(covered_buildup, box_edge, low_block, unsupported_advance/final_third), Spec.blocks (per-team overrides of the block
model) and PRESS_D 2.0 m (the three-state pressure call: the 3.0 m plant measured a first-defender distance of p10 /
p50 / p90 2.84 / 3.45 / 4.21 m drifting away, unclear on 27 of 31 planted opportunities; at 2.0 m 2.06 / 2.51 /
4.07 m). Measured through the real tiers: covered_buildup's ball at the carrying centre-back about 31 m from A's goal;
final_third's D3 samples with the ball at 75.8-76.3 m (p10-p90) and nobody behind it on 176 of 176; box_edge/box the
ball at 14.8-20.6 m (p10-p90) with the plain depth rule counting nobody goal-side on the far side on 388 of 388
samples; low_block/default A's field depth share .22-.25 against an available-length share p50 .92, compressed .13-.16
and .56.
EVAL's part of the round (2026-10-23): COACH verifier scene 2 as channel_threat (deep / high), stacking/reachable,
and Advance.opponent_at, where the channel opponent waits: 'deep' 4 m deeper than the centre-back line (truth: -4.1 m
on all 16 spells of stacking/default and channel_threat/deep), 'high' halfway up to the advanced wide defender (+9.9
m; low_calibration +6.5 to +9.9 m, its centre-backs pushed up). With him deep no cover spot level with or goal-side of
him is reachable in 2 s (alternatives' critic-B-8 rule): the scenes keep the principle only (no_reachable_fix at
levels 3-7). With him high, low_calibration, stacking/reachable and channel_threat/high draw one D3 zone at each of
levels 3-7 (15 in all), every one goal-side of him (coaching_eval.zone_side_problems) and clear of the stacking
team-mate; before, low_calibration's diagram-only row and stacking's zone check had no zone left to test. The scenes'
zones at the default profiles: 25 (D3 15, D1 6, D2 4), 0 property problems.
Fix round 3 (2026-10-24, EVAL; only expectations changed: every frame of every scene is as before, and the result's
sourceSHA256, a hash of the scene spec with its expectations, differs for the 12 variants whose expectations changed;
the truth without its expectations is identical for all 39): levels 1-2 now judge an exposed advance
wherever the ball is (critic B-1), so every UNSUPPORTED plant (an opponent left in the vacated channel) expects a
criticism at levels 1-2 ('criticised'; none_behind the team-level try_this at both, which fix round 1 had asserted
away), the covered scenes keep 'no_criticism'; spells4 expects a 'moments worth a look' review at levels 3-7 (critic
S-2: its 4 clear advances in 6.1 minutes), single_advance 'too_few_situations'; amateur_coach_spans expects level 1's
reframed advance assessed and criticised without a ball (level 2's advanced rule leaves these 7v7 advances out of
scope); wrong_formation/default keeps the role downgrade at levels 2-7 and expects level 1's team-level try_this.
Measured through the real engine (coaching_eval scenes): 477 checks pass, 0 fail, 65 not applicable, 10 vacuous.
"""
import argparse
import hashlib
import json
import math
import os
import sys
from dataclasses import dataclass, field as dc_field, replace
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
CATALOGUE = REPO/'tools'/'annotations'/'coaching'/'synthetic_scenes.md'
VERSION = 1                 # truth / catalogue layout version (bump when a field changes meaning)

# ---------- time and image ----------
FPS = 10.0                  # frames per second (PitchIQ writes 25-30; consumers work in seconds; see the docstring)
T_START = 600.0             # s: absolute source time of the first frame (not 0: a consumer that mixes segment and source
                            # time fails on a scene)
IMG_W, IMG_H = 1920, 1080   # px: the image the boxes and the homography refer to
FULL_SPAN_PX = 1785.0       # px the field's length takes in the static full-field camera (105 m -> 17 px per metre)
FULL_TOP_PX = 400.0         # px: image row of the far touchline in the full-field camera (boxes 1.8 m tall fit above)
BOX_H_M, BOX_W_M = 1.8, .7  # m: a person's box (height at the across scale, as ball_path's pixels-per-metre assumes)
BALL_D_M = .35              # m: the ball's box side in the image (a 22 cm ball plus blur at broadcast resolution)
PAINT_EVERY = .2            # s between paint ticks (PitchIQ: every 6th frame at 30 fps)

# ---------- motion ----------
BLOCK_TAU = .9              # s: each of the two cascaded smoothing stages of a team block's centre and stretch
BLOCK_VMAX = 5.0            # m/s: a block's centre never moves faster (a team shifting, not sprinting)
WANDER_SD, WANDER_TAU = .9, 4.0     # m, s: a player's slow drift around his spot (AR(1))
NOISE_SD, NOISE_TAU = .2, .4        # m, s: observation noise on a position (AR(1)); pitchSigma says .3 m
BALL_NOISE_SD = .1          # m: observation noise on the ball's pitch position
FORMATION_BLEND = 8.0       # s: players walk from their old to their new slot over this after a formation change
FORMATION_NEAR = 15.0       # s: DESIGN 3.4 suppresses role-based criticism this close to a formation change
GK_LINE = 5.0               # m: a goalkeeper's depth from his goal line (plus a share of his block's movement)

# Team blocks, in metres on a 105 x 68 m pitch (scaled to the field), in the team's attack frame (X from its own goal
# line, Y from its own left touchline). The ball focus F is where the ball roughly is.
ATT_BEHIND, ATT_X = 10.0, (35.0, 66.0)  # in possession: block centre F.X - 10, within these depths
DEF_BEHIND, DEF_X = 22.0, (25.0, 52.0)  # out of possession: F.X - 22 (deeper), within these
ATT_SX, ATT_SY, ATT_KY = 1.15, 1.0, .2  # depth and width stretch of the layout; lateral pull towards the ball
DEF_SX, DEF_SY, DEF_KY = .85, .72, .4

# ---------- the ball ----------
BALL_AHEAD = .45            # m: the ball in front of its carrier (possession: full evidence within .7 m)
PASS_SPEED = 15.0           # m/s
KICK_OUT_SPEED = 14.0       # m/s: a ball played over the touchline
OUT_BEYOND = 3.0            # m beyond the line where an out-of-play ball is last seen (possession: 1 m for .2 s)
OUT_SEEN = .5               # s it is seen there before it leaves the picture
PLACED_S = 2.0              # s the ball lies still on the line before a restart (possession: still .3 s = placed)
TAKER_BACK = 4.0            # s the restart's taker takes to rejoin his team's shape after the restart
CARRY = (1.6, 3.4)          # s a carrier keeps the ball (uniform)
FLIGHT = (.5, 2.4)          # s a pass may take (a 40 m switch at most about 17 m/s)
SPARSE_ON, SPARSE_EVERY = .5, 4.0   # s: a 'sparse' ball is seen this long every this many seconds (12.5% of frames:
                                    # possession needs a usable ball on 20% to be available)

# ---------- plants ----------
ADVANCE_BEYOND = 20.0       # m beyond the centre-back line (DESIGN 4.3: advanced = 15 m x L/105)
ADVANCED_M = 15.0           # m: the advanced threshold the truth checks (DESIGN 4.3)
CHANNEL_Y = 5.0             # m from the touchline: an advanced wide defender's lane
CB_SHIFT = 6.0              # m: the near centre-back's shift towards the touchline (cover test c needs 4 m)
CHANNEL_BEHIND = 15.0       # m: a team-mate covering the channel stands this far behind the advanced player (test a: 10)
BEHIND_M = 10.0             # m: 'behind the ball' (DESIGN 4.4: goal-side by 10 m x L/105)
PUSH_MARGIN = 6.0           # m: pushed-up centre-backs stand this far behind the block centre (the carrier 4 m: about
                            # 2.5 m goal-side of the ball, so not 'behind' it by 10 m)
DEEP_MARGIN = 16.0          # m: the one centre-back left behind stands DEEP_MARGIN + 1 behind the block centre (about
                            # 13.5 m goal-side of the ball)
PRESS_D = 2.0               # m: a pressing first defender's distance to the carrier (DESIGN 4.1: pressure <= 5 m; fix
                            # round 1: the three-state call presses within 3 m, or 3-5 m only while closing, and a 3.0 m
                            # plant measured p50 3.45 m drifting away, unclear on 27 of 31 opportunities)
NATURAL_PRESS_D = 4.0       # m: the first defender of a 'natural' defence
UNPRESSED_D = 9.5           # m: nobody nearer the carrier than this when a defence does not press (DESIGN: > 8 m)
PRESSED_M = 5.0             # m: the truth calls the carrier pressed when a defender is this close (DESIGN 4.1)
PRESS_REACH = 12.0          # m: a defender further than this from the carrier does not go to press (body-scale
                            # distances are metres on every field size, DESIGN 2.5)
PRESS_VMAX = 6.5            # m/s: a defender's run to press (or to cover) is timed so that its smoothstep peaks at this
                            # speed, starting while the pass to the carrier is in the air
COVER_D, COVER_ANGLE = 7.0, 25.0    # m, degrees: the covering defender behind the first one (DESIGN: R 12 m, A 60 deg;
                                    # at most .8 R on a small field)
COVER_CLEAR_ANGLE = 70.0    # degrees: 'no cover' moves every defender within R + 1 m at <= this angle out to R + 3 m
OPPONENT_VMAX = 6.0         # m/s: the opponent who waits in the vacated channel jogs there and back at most this fast
STACK_GAP = 2.5             # m: the stacking team-mate stands this far outside the outer third, 13 m behind
RECOVER_SPEED = 3.0         # m/s: goalwards speed that counts as recovering (DESIGN 4.3 counterexample 2)
WIDE_BALL_TAU = 1.0         # s: a defence reacts to the ball entering a wide channel with this smoothing (weak, gap)
WIDE_HELD = 3.5             # s: the truth checks the weak side and the gap once the ball has been wide this long
GAP_OPEN_Y = (2.0, 4.0)     # m: an open ball-side gap: the wide defender at the touchline (2 m), the near centre-back
                            # 4 m from the middle (a gap of about 28 m: DESIGN 4.5 needs 20 m x W/68)

BASE_VMAX = 7.0             # m/s: a player's formation movement (block + wander) is clamped to this before any plant
SPEED_CAP = 8.5             # m/s: the final positions are clamped to this, except on the frames a player carries the ball
                            # (the ball is at his feet there); truth.speeds.capped says how many player-frames it moved
LIMIT_PLAYER_SPEED = 9.5    # m/s: the truth flags a scene whose true positions move faster (sanity, tested)
LIMIT_BALL_SPEED = 32.0     # m/s

LEVEL_AGE = {1: 'U8', 2: 'U10', 3: 'U12', 4: 'U14', 5: 'U16', 6: 'Adult', 7: 'Adult'}   # DESIGN 2.2 typical ages

# Formation layouts: (role, X, Y) in metres on 105 x 68, attack frame, the defending reference shape. 11v11 layouts are
# tests/team_shape_synth.py's (placed by hand, spacings unlike soccer/team_shape.py's templates); role names are the
# template roles in the same order. Small formats use the relative names of DESIGN 3.4 (block-k/n, k-th from the left).
LAYOUTS = {
    '4-4-2': [('LB', 28, 7), ('LCB', 24, 24), ('RCB', 24, 44), ('RB', 28, 61), ('LM', 43, 9), ('LCM', 39, 26),
              ('RCM', 39, 42), ('RM', 43, 59), ('LCF', 55, 28), ('RCF', 56, 41)],
    '4-3-3': [('LB', 27, 8), ('LCB', 24, 25), ('RCB', 24, 43), ('RB', 27, 60), ('LCM', 38, 21), ('CDM', 34, 34),
              ('RCM', 38, 47), ('LW', 53, 9), ('CF', 58, 33), ('RW', 53, 59)],
    '3-5-2': [('LCB', 25, 17), ('CB', 23, 34), ('RCB', 25, 51), ('LWB', 41, 4), ('LCM', 38, 22), ('CDM', 35, 34),
              ('RCM', 38, 46), ('RWB', 41, 64), ('LCF', 55, 28), ('RCF', 55, 40)],
    '1-3-2-3': [('back-1/3', 22, 12), ('back-2/3', 20, 34), ('back-3/3', 22, 56), ('mid-1/2', 38, 24), ('mid-2/2', 38, 44),
                ('front-1/3', 55, 10), ('front-2/3', 58, 34), ('front-3/3', 55, 58)],
    '1-3-2-1': [('back-1/3', 24, 12), ('back-2/3', 22, 34), ('back-3/3', 24, 56), ('mid-1/2', 42, 22), ('mid-2/2', 42, 46),
                ('front-1/1', 60, 34)],
    '1-2-1': [('back-1/1', 22, 34), ('mid-1/2', 38, 14), ('mid-2/2', 38, 54), ('front-1/1', 55, 34)],
}
# A formation change moves each player from his old slot to a new one (by role).
CHANGES = {('4-4-2', '3-5-2'): {'LB': 'LWB', 'LCB': 'LCB', 'RCB': 'RCB', 'RB': 'RWB', 'LM': 'LCM', 'LCM': 'CB',
                                'RCM': 'CDM', 'RM': 'RCM', 'LCF': 'LCF', 'RCF': 'RCF'}}
CENTRE_BACKS = ('LCB', 'CB', 'RCB', 'back-2/3')
WIDE_DEFENDERS = {'LB': 'left', 'RB': 'right', 'LWB': 'left', 'RWB': 'right', 'back-1/3': 'left', 'back-3/3': 'right'}
HOLDING = ('CDM', 'LDM', 'RDM', 'mid-1/2')


# ---------- small helpers ----------
def _smoothstep(u):
    u = np.clip(u, 0.0, 1.0)
    return u*u*(3-2*u)


def ramp(t, t0, up, hold, down):
    """Weight 0..1 over time: smoothstep up from t0 over `up` s, 1 for `hold` s, smoothstep down over `down` s."""
    a = _smoothstep((t-t0)/max(up, 1e-9))
    b = _smoothstep((t-(t0+up+hold))/max(down, 1e-9))
    return a*(1-b)


def smooth(v, dt, tau=BLOCK_TAU, vmax=None):
    """Two cascaded exponential stages (a critically damped lag), then a rate limit of vmax per second; v is (T,) or
    (T, k)."""
    out = np.array(v, np.float64, copy=True)
    a = 1.0-math.exp(-dt/tau)
    for _ in range(2):
        for k in range(1, len(out)):
            out[k] = out[k-1]+a*(out[k]-out[k-1])
    if vmax is not None:
        step = vmax*dt
        for k in range(1, len(out)):
            out[k] = out[k-1]+np.clip(out[k]-out[k-1], -step, step)
    return out


def ar1(rng, T, shape, dt, tau, sd):
    """AR(1) noise of standard deviation sd and correlation time tau: (T,)+shape."""
    phi = math.exp(-dt/tau)
    e = rng.normal(0.0, sd*math.sqrt(1-phi*phi), (T,)+tuple(shape))
    out = np.empty_like(e)
    out[0] = rng.normal(0.0, sd, tuple(shape))
    for k in range(1, T):
        out[k] = phi*out[k-1]+e[k]
    return out


def limit_speed(P, step, exempt=None):
    """Positions (T, N, 2) whose frame-to-frame step is clamped to `step` metres, sequentially per player; frames where
    exempt (T, N) is True keep their position (and the clamp resumes from there)."""
    out = np.array(P, np.float64, copy=True)
    for k in range(1, len(out)):
        d = out[k]-out[k-1]
        n = np.hypot(d[:, 0], d[:, 1])
        over = n > step
        if exempt is not None:
            over &= ~exempt[k]
        if over.any():
            out[k, over] = out[k-1, over]+d[over]*(step/n[over])[:, None]
    return out


def attack(P, side, L, W):
    """Pitch metres (x from the left goal line, y from the far touchline) <-> a team's attack-frame metres (X from its
    own goal line, Y from its own left touchline). The map is its own inverse."""
    P = np.asarray(P, np.float64)
    if side == 'left':
        return P.copy()
    out = np.empty_like(P)
    out[..., 0] = L-P[..., 0]
    out[..., 1] = W-P[..., 1]
    return out


def _r(v, d=4):
    return round(float(v), d)


def _sha(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(',', ':')).encode('ascii')).hexdigest()


# ---------- the scene specification ----------
@dataclass(frozen=True)
class Span:
    """A stretch of the script: team 'A' or 'B' has the ball, or the ball is 'dead' (out of play until the next span's
    team restarts). focus: (fraction of the span, X, Y) in the possessing team's attack frame, metres on 105 x 68;
    a path without a point at 0 starts where the previous span's ball was."""
    t0: float
    t1: float
    state: str
    focus: tuple = ()


@dataclass(frozen=True)
class Advance:
    """A wide defender's advance beyond the centre-back line (DESIGN 4.3), with its context plants.
    cover: 'none' | 'cb_shift' (the near centre-back shifts towards the touchline: test c) | 'channel' (a team-mate in
    the vacated outer third, behind him: test a) | 'dm_drop' (the holding midfielder drops to the centre-back line:
    test b, L5+). behind: None (natural) or the number of team-mates left goal-side of the ball by BEHIND_M or more (0 or
    1; the others are pushed up). opponent: an opponent waits in the vacated channel in the team's own half (exposed).
    crossing: an opponent crosses the wide defender at the middle of the hold (an identity issue is reported).
    recover: the spell ends with a lost ball and he sprints back. stacking: a team-mate stands just outside the outer
    third where the natural cover zone is. opponent_at: where the channel opponent waits, "deep" (4 m deeper than the
    centre-back line: no team-mate reaches a spot level with or goal-side of him in 2 s, so the honest output is the
    principle only, critic B-8) or "high" (halfway between the centre-back line and the advanced wide defender: a
    winger left near halfway, with a reachable cover spot goal-side of him; fix round 1, COACH verifier scene 2)."""
    team: str
    role: str
    t0: float
    up: float = 3.5
    hold: float = 4.0
    down: float = 3.5
    beyond: float = ADVANCE_BEYOND
    cover: str = 'none'
    behind: int | None = None
    opponent: bool = False
    crossing: bool = False
    recover: bool = False
    stacking: bool = False
    opponent_at: str = 'deep'


@dataclass(frozen=True)
class Defend:
    """How a team defends while the other team has the ball. press: 'natural' (the nearest defender goes to 4 m),
    'cover' (3 m, and a second defender covers him 7 m behind at 25 deg), 'press' (3 m and nobody covers: no other
    defender within 13 m at <= 70 deg), 'none' (nobody nearer the carrier than 9.5 m). weak: 'natural' | 'empty' (with
    the ball in a wide channel nobody stays in the far half). gap: 'natural' | 'open' (with the ball in a wide channel
    the near wide defender holds the touchline and the near centre-back the middle: a ball-side gap of about 28 m)."""
    press: str = 'natural'
    weak: str = 'natural'
    gap: str = 'natural'


@dataclass(frozen=True)
class Spec:
    scene: str
    variant: str
    title: str
    spans: tuple
    formations: dict                      # {'A': ((t, name), ...), 'B': ...}
    players: int = 11                     # format (DESIGN 2.1): 4, 7, 9, 11
    goalkeeper: bool = True
    field: tuple = (105.0, 68.0)          # the real field, metres
    measured: bool = False                # tactical.pitchSize source 'landmarks' (else 'standard': assumed)
    declared: bool = False                # the coach declares the field in the settings
    plants: tuple = ()
    defend: dict = dc_field(default_factory=lambda: {'A': Defend(), 'B': Defend()})
    blocks: dict = dc_field(default_factory=dict)   # per team, overrides of the block model (fix round 1): att_behind,
                                                    # att_x (lo, hi), att_sx, def_behind, def_x (lo, hi), def_sx
    footage: str = 'broadcast'            # 'broadcast' | 'amateur' (small scene, keyless) | 'veo' (small scene, lens)
    camera: str = 'full'                  # 'full' | 'follow' (a zoomed camera on the team's left channel)
    follow_px: float = 72.0               # px per metre of the follow camera (27 m across, 30 m down)
    calibration: float = .86              # the final registration's confidence (HIGH >= .7: world tier with paint)
    paint: bool = True                    # paint ticks (world tier needs them; broadcast full-size only)
    ball: str = 'tracked'                 # 'tracked' | 'sparse' (placed on 12.5% of the frames: possession abstains)
    swarm: bool = False                   # everyone near the ball (a 4v4 of young players)
    carry: tuple = CARRY
    seconds: float = 0.0                  # 0: the end of the last span
    fps: float = FPS
    t_start: float = T_START
    seed: int = 0
    intended: str | None = 'auto'         # the coach's intended formation ('auto': the scripted starting formation)
    coach_roles: tuple = ()               # ((pid, role), ...) roles the coach confirms
    coach_spans: bool = False             # the coach marked the possession and dead spans (user decision 1)
    age: str | None = None                # default age group of the settings (else by level, DESIGN 2.2)
    expected: tuple = ()
    notes: tuple = ()


@dataclass
class Built:
    spec: Spec
    result: dict | None                   # None from build_truth (the truth without the observed result)
    truth: dict
    coach_spans: dict | None

    @property
    def key(self):
        return f'{self.spec.scene}/{self.spec.variant}'

    @property
    def expected(self):
        return [dict(e) for e in self.spec.expected]


# ---------- the world: positions, plants, ball ----------
class World:
    """Everything a scene computes, in pitch metres of the real field (x from the left goal line, y from the far
    touchline). Team A defends the left goal, team B the right one, for the whole scene."""

    def __init__(self, spec):
        self.spec = spec
        self.L, self.W = float(spec.field[0]), float(spec.field[1])
        self.kx, self.ky = self.L/105.0, self.W/68.0
        self.fps = float(spec.fps)
        self.dt = 1.0/self.fps
        end = spec.seconds or (spec.spans[-1].t1-spec.t_start)
        self.T = int(round(end*self.fps))
        self.t = spec.t_start+np.arange(self.T)*self.dt
        base = int(hashlib.sha256(f'{spec.scene}|{spec.seed}'.encode('ascii')).hexdigest()[:8], 16)
        # One generator per purpose, seeded by the scene (not the variant): twins share their wander, noise and the
        # carriers' rhythm wherever their plants do not differ.
        self.rngs = {k: np.random.default_rng([base, i]) for i, k in enumerate(('wander', 'ball', 'noise', 'misc'))}
        self.side = {'A': 'left', 'B': 'right'}
        self.outfield = spec.players-(1 if spec.goalkeeper else 0)
        self.ids = {team: [f'{team}-{j+1:02d}' for j in range(self.outfield)] for team in ('A', 'B')}
        self.gk = {'A': 'GK-1', 'B': 'GK-2'} if spec.goalkeeper else {}
        self.locks = {}            # (team, pid) -> [(t_from, t_to)]: moved by a plant, never chosen to carry the ball
        self.forced = []           # (team, pid, ta, tb): a plant's carrier
        self.issues, self.events, self.low_identity = [], [], []
        self.plant_log = []

    # ----- indices and frames
    def k_of(self, time):
        return int(np.clip(round((time-self.spec.t_start)*self.fps), 0, self.T-1))

    def att(self, team, P):
        return attack(P, self.side[team], self.L, self.W)

    @staticmethod
    def other(team):
        return 'B' if team == 'A' else 'A'

    def lock(self, team, pid, t0, t1):
        self.locks.setdefault((team, pid), []).append((float(t0), float(t1)))

    def locked(self, team, pid, t0, t1):
        return any(a < t1 and t0 < b for a, b in self.locks.get((team, pid), ()))

    # ----- the script
    def script(self):
        s, t, L, W = self.spec, self.t, self.L, self.W
        self.state = np.full(self.T, 'dead', dtype='<U4')
        focus = np.full((self.T, 2), np.nan)
        last = np.array([L/2, W/2])
        for span in s.spans:
            m = (t >= span.t0-1e-9) & (t < span.t1-1e-9)
            self.state[m] = span.state
            if span.state in ('A', 'B'):
                side = self.side[span.state]
                pts = [(float(f), X*self.kx, Y*self.ky) for f, X, Y in span.focus]
                if not pts or pts[0][0] > 0:
                    st = attack(last, side, L, W)
                    pts = [(0.0, float(st[0]), float(st[1]))]+pts
                if pts[-1][0] < 1:
                    pts.append((1.0,)+pts[-1][1:])
                fr = (t[m]-span.t0)/max(span.t1-span.t0, 1e-9)
                X = np.interp(fr, [p[0] for p in pts], [p[1] for p in pts])
                Y = np.interp(fr, [p[0] for p in pts], [p[2] for p in pts])
                focus[m] = attack(np.stack([X, Y], axis=1), side, L, W)
                last = attack(np.array([pts[-1][1], pts[-1][2]]), side, L, W)
            else:
                focus[m] = last
        self.state[t >= s.spans[-1].t1-1e-9] = s.spans[-1].state
        bad = ~np.isfinite(focus[:, 0])
        if bad.any():
            focus[bad] = last
        self.focus = smooth(focus, self.dt, tau=.6, vmax=12.0)
        # A team's mode: in possession (+1), defending (-1), or (dead ball) the mode it had before.
        self.mode = {}
        for team in ('A', 'B'):
            mode = np.where(self.state == team, 1.0, np.where(self.state == self.other(team), -1.0, np.nan))
            prev = -1.0
            for k in range(self.T):
                if np.isnan(mode[k]):
                    mode[k] = prev
                prev = mode[k]
            self.mode[team] = mode

    # ----- formations and blocks
    def layout(self, name, roles):
        rows = {r: (x, y) for r, x, y in LAYOUTS[name]}
        xy = np.array([rows[r] for r in roles], np.float64)
        centre = np.array([[x, y] for _, x, y in LAYOUTS[name]], np.float64).mean(axis=0)
        return (xy-centre)*[self.kx, self.ky]

    def formations(self):
        self.roles, self.offsets, self.periods = {}, {}, {}
        for team in ('A', 'B'):
            periods = self.spec.formations[team]
            first = periods[0][1]
            current = [r for r, _, _ in LAYOUTS[first]]
            if len(current) != self.outfield:
                raise ValueError(f'{self.spec.scene}: {first} has {len(current)} outfield players, the format {self.outfield}')
            off = np.repeat(self.layout(first, current)[None], self.T, axis=0)
            roles = np.empty((self.T, self.outfield), dtype=object)
            roles[:] = current
            prev = first
            spans = [{'t0': _r(self.t[0], 3), 'formation': first}]
            for tc, name in periods[1:]:
                mapping = CHANGES[(prev, name)]
                new = [mapping[r] for r in current]
                w = _smoothstep((self.t-tc)/FORMATION_BLEND)[:, None, None]
                off = (1-w)*off+w*self.layout(name, new)[None]
                roles[self.t >= tc+FORMATION_BLEND/2] = new
                spans[-1]['t1'] = _r(tc, 3)
                spans.append({'t0': _r(tc, 3), 'formation': name})
                current, prev = new, name
            spans[-1]['t1'] = _r(self.t[-1], 3)
            self.roles[team], self.offsets[team], self.periods[team] = roles, off, spans

    def formation_at(self, team, time):
        name = self.spec.formations[team][0][1]
        for tc, n in self.spec.formations[team]:
            if time >= tc+FORMATION_BLEND/2:
                name = n
        return name

    def role_index(self, team, role, time):
        row = list(self.roles[team][self.k_of(time)])
        return row.index(role) if role in row else None

    def blocks(self):
        kx, ky, W = self.kx, self.ky, self.W
        self.block = {}
        for team in ('A', 'B'):
            F = self.att(team, self.focus)
            att = self.mode[team] > 0
            o = (self.spec.blocks or {}).get(team) or {}
            ab, ax, asx = o.get('att_behind', ATT_BEHIND), o.get('att_x', ATT_X), o.get('att_sx', ATT_SX)
            db, dx, dsx = o.get('def_behind', DEF_BEHIND), o.get('def_x', DEF_X), o.get('def_sx', DEF_SX)
            cx = np.where(att, np.clip(F[:, 0]-ab*kx, ax[0]*kx, ax[1]*kx),
                          np.clip(F[:, 0]-db*kx, dx[0]*kx, dx[1]*kx))
            sx = np.where(att, asx, dsx)
            sy = np.where(att, ATT_SY, DEF_SY)
            pull = np.where(att, ATT_KY, DEF_KY)
            cy = W/2+pull*(F[:, 1]-W/2)
            self.block[team] = {'cx': smooth(cx, self.dt, vmax=BLOCK_VMAX*kx), 'cy': smooth(cy, self.dt, vmax=BLOCK_VMAX*ky),
                                'sx': smooth(sx, self.dt), 'sy': smooth(sy, self.dt)}

    def base_positions(self):
        kx, ky, L, W = self.kx, self.ky, self.L, self.W
        k = min(kx, ky)
        rng = self.rngs['wander']
        self.pos, self.gkpos = {}, {}
        for team in ('A', 'B'):
            b = self.block[team]
            off = self.offsets[team]
            wander = ar1(rng, self.T, (self.outfield, 2), self.dt, WANDER_TAU, WANDER_SD*k)
            if self.spec.swarm:
                # Everyone around the ball: a per-player offset that wanders (sd 3 m, kept within 7 m), the defending
                # team a little goal-side.
                F = self.att(team, self.focus)
                spread = ar1(rng, self.T, (self.outfield, 2), self.dt, 2.5, 3.0*k)
                X = F[:, None, 0]+np.clip(spread[..., 0], -7*k, 7*k)-1.5*k*(self.mode[team][:, None] < 0)
                Y = F[:, None, 1]+np.clip(spread[..., 1], -7*k, 7*k)
            else:
                X = b['cx'][:, None]+b['sx'][:, None]*off[..., 0]+wander[..., 0]
                Y = b['cy'][:, None]+b['sy'][:, None]*off[..., 1]+wander[..., 1]
            X = np.clip(X, 1.0*kx, L-1.0*kx)
            Y = np.clip(Y, .8*ky, W-.8*ky)
            self.pos[team] = limit_speed(self.att(team, np.stack([X, Y], axis=2)), BASE_VMAX*self.dt)
            if team in self.gk:
                F = self.att(team, self.focus)
                gw = ar1(rng, self.T, (2,), self.dt, 3.0, .3*k)
                gX = np.clip(GK_LINE*kx+.06*(b['cx']-35*kx), 2*kx, 14*kx)+gw[:, 0]
                gY = W/2+np.clip(.3*(F[:, 1]-W/2), -7*ky, 7*ky)+gw[:, 1]
                self.gkpos[team] = self.att(team, np.stack([gX, gY], axis=1))

    # ----- moving one player towards a target, in his team's attack frame
    def att_pos(self, team):
        return self.att(team, self.pos[team])

    def blend(self, team, j, w, target_att, axes=(0, 1)):
        """pos = (1-w) pos + w target in the team's attack frame, where w > 0; target_att is (T, 2) or (2,)."""
        m = w > 0
        if not m.any():
            return
        A = self.att(team, self.pos[team][:, j])
        tgt = np.broadcast_to(np.asarray(target_att, np.float64), A.shape)
        for a in axes:
            A[m, a] = (1-w[m])*A[m, a]+w[m]*tgt[m, a]
        self.pos[team][:, j] = self.att(team, A)

    # ----- plants that do not depend on the ball
    def apply_advance(self, p):
        team, opp = p.team, self.other(p.team)
        kx, ky, L, W = self.kx, self.ky, self.L, self.W
        t, T = self.t, self.T
        total = p.up+p.hold+p.down
        w = ramp(t, p.t0, p.up, p.hold, p.down)
        mid = p.t0+p.up+p.hold/2
        j = self.role_index(team, p.role, mid)
        if j is None:
            raise ValueError(f'{self.spec.scene}: no {p.role} in team {team} at {mid}')
        pid = self.ids[team][j]
        lane = WIDE_DEFENDERS[p.role]
        sgn = 1.0 if lane == 'left' else -1.0          # +1: the lane's touchline is at small Y (attack frame)
        lane_y = CHANNEL_Y*ky if lane == 'left' else W-CHANNEL_Y*ky
        outer = W/3 if lane == 'left' else 2*W/3       # the inner edge of the lane's outer third
        win = (p.t0-.5, p.t0+total+.5)
        self.lock(team, pid, *win)
        roles = list(self.roles[team][self.k_of(mid)])
        layout_y = {r: y*ky for r, _, y in LAYOUTS[self.formation_at(team, mid)]}
        on_lane_side = lambda r: sgn*(layout_y.get(r, W/2)-W/2) < 0
        cbs = [i for i, r in enumerate(roles) if r in CENTRE_BACKS]
        cb_near = min(cbs, key=lambda i: sgn*layout_y.get(roles[i], W/2)) if cbs else None
        cx = self.block[team]['cx']
        log = {'kind': 'wide_advance', 'team': team, 'player': pid, 'role': p.role, 'lane': lane, 't0': _r(p.t0, 3),
               't1': _r(p.t0+total, 3), 'holdT0': _r(p.t0+p.up, 3), 'holdT1': _r(p.t0+p.up+p.hold, 3),
               'beyondM': _r(p.beyond*kx, 2), 'cover': p.cover, 'behind': p.behind, 'opponent': p.opponent,
               'crossing': None, 'recover': p.recover, 'stacking': p.stacking, 'movers': {},
               'opponentAt': p.opponent_at if p.opponent else None}
        zeros = np.zeros(T)
        # 1. The team pushed up so that nobody (or one player) is goal-side of the ball: the near centre-back carries it
        #    into midfield (a forced carrier), the far full-back steps up and, for nobody behind, the other centre-backs.
        if p.behind is not None and cb_near is not None:
            far = [i for i, r in enumerate(roles) if r in WIDE_DEFENDERS and WIDE_DEFENDERS[r] != lane]
            moves = [(cb_near, -4.0)]+[(i, -8.0) for i in far]
            rest = [i for i in cbs if i != cb_near]
            # behind 1: the most central other centre-back stays (one player goal-side of the ball); the rest step up
            keep = min(rest, key=lambda i: abs(layout_y.get(roles[i], W/2)-W/2)) if p.behind == 1 and rest else None
            moves += [(i, -DEEP_MARGIN-1.0 if i == keep else -PUSH_MARGIN) for i in rest]
            for i, dx in moves:
                self.blend(team, i, w, np.stack([cx+dx*kx, zeros], axis=1), axes=(0,))
                self.lock(team, self.ids[team][i], *win)
                log['movers'][self.ids[team][i]] = ('carrier' if i == cb_near else 'stays_behind' if i == keep
                                                    else 'pushed_up')
            self.forced.append((team, self.ids[team][cb_near], p.t0+.4*p.up, p.t0+p.up+p.hold+.6*p.down))
        A = self.att_pos(team)
        cbx = A[:, cbs, 0].mean(axis=1) if cbs else cx-15*kx
        # 2. The wide defender beyond the centre-back line, in his lane.
        self.blend(team, j, w, np.stack([cbx+p.beyond*kx, np.full(T, lane_y)], axis=1))
        A = self.att_pos(team)
        wide = A[:, j].copy()
        # 3. Cover (DESIGN 4.3 tests a, b, c) or its absence.
        if p.cover == 'cb_shift' and cb_near is not None:
            self.blend(team, cb_near, w, np.stack([zeros, A[:, cb_near, 1]-sgn*CB_SHIFT*ky], axis=1), axes=(1,))
            self.lock(team, self.ids[team][cb_near], *win)
            log['movers'][self.ids[team][cb_near]] = 'cb_shift'
        elif p.cover == 'channel':
            mates = [i for i, r in enumerate(roles) if r in ('LCM', 'RCM', 'CDM', 'CM', 'mid-1/2', 'mid-2/2') and i != j]
            if mates:
                i = min(mates, key=lambda i: sgn*layout_y.get(roles[i], W/2))
                self.blend(team, i, w, np.stack([wide[:, 0]-CHANNEL_BEHIND*kx, np.full(T, lane_y+sgn*4*ky)], axis=1))
                self.lock(team, self.ids[team][i], *win)
                log['movers'][self.ids[team][i]] = 'channel_cover'
        elif p.cover == 'dm_drop':
            dms = [i for i, r in enumerate(roles) if r in HOLDING and i != j]
            if dms:
                self.blend(team, dms[0], w, np.stack([cbx, zeros], axis=1), axes=(0,))
                self.lock(team, self.ids[team][dms[0]], *win)
                log['movers'][self.ids[team][dms[0]]] = 'dm_drop'
        if p.cover == 'none':
            # No holding midfielder dropping to the centre-back line either (test b): he stays 7 m or more ahead of it.
            for i, r in enumerate(roles):
                if r in HOLDING and i != j:
                    x = self.att_pos(team)[:, i, 0]
                    self.blend(team, i, w, np.stack([np.maximum(x, cbx+7*kx), zeros], axis=1), axes=(0,))
                    self.lock(team, self.ids[team][i], *win)
                    log['movers'].setdefault(self.ids[team][i], 'holding_ahead')
        if p.cover in ('none', 'dm_drop'):
            # Nobody else in the vacated outer third: the team-mates of that half keep inside it (the wide midfielder
            # tucks in as the full-back overlaps), with the spell's own ramp.
            A = self.att_pos(team)
            edge = outer+sgn*3.5*ky
            for i, r in enumerate(roles):
                if i == j or not on_lane_side(r) or (r in HOLDING and p.cover == 'dm_drop'):
                    continue
                y = A[:, i, 1]
                target = np.maximum(y, edge) if lane == 'left' else np.minimum(y, edge)
                if np.allclose(target[w > 0], y[w > 0]):
                    continue
                self.blend(team, i, w, np.stack([zeros, target], axis=1), axes=(1,))
                self.lock(team, self.ids[team][i], *win)
                log['movers'].setdefault(self.ids[team][i], 'tucked_inside')
        # 4. A team-mate just outside the outer third where a cover zone would be (stacking).
        if p.stacking:
            mates = [i for i, r in enumerate(roles) if r in ('LCM', 'RCM', 'mid-1/2', 'mid-2/2') and i != j]
            if mates:
                i = min(mates, key=lambda i: sgn*layout_y.get(roles[i], W/2))
                self.blend(team, i, w, np.stack([wide[:, 0]-13*kx, np.full(T, outer+sgn*STACK_GAP*ky)], axis=1))
                self.lock(team, self.ids[team][i], *win)
                log['movers'][self.ids[team][i]] = 'stacking'
        used = set()
        # 5. An opponent waiting in the vacated channel in the team's own half (exposed).
        if p.opponent:
            hold = w > .5
            depth = float(np.median(cbx[hold])) if hold.any() else 40*kx
            if p.opponent_at not in ('deep', 'high'):
                raise ValueError(f'{self.spec.scene}: opponent_at {p.opponent_at!r}')
            # 'deep': 4 m deeper than the centre-back line; 'high': halfway up to the advanced wide defender
            ox = depth-4*kx if p.opponent_at == 'deep' else depth+.5*p.beyond*kx
            spot = np.array([min(ox, L/2-6*kx), lane_y+sgn*4*ky])
            O = self.att(team, self.pos[opp])
            k0 = self.k_of(p.t0)
            d = np.hypot(O[k0, :, 0]-spot[0], O[k0, :, 1]-spot[1])
            d += 1e3*np.array([self.locked(opp, q, p.t0-6.0, p.t0+total+6.0) for q in self.ids[opp]])
            i = int(np.argmin(d))
            # he jogs there (smoothstep peak OPPONENT_VMAX), in place before the wide defender is advanced, and back
            ramp_s = float(np.clip(1.5*d[i]/OPPONENT_VMAX, 3.0, 6.0))
            start = p.t0+.6*p.up-ramp_s
            stay = (p.t0+p.up+p.hold+.4*p.down)-(start+ramp_s)
            self.blend(opp, i, ramp(t, start, ramp_s, stay, ramp_s), self.att(opp, self.att(team, spot)))
            self.lock(opp, self.ids[opp][i], start-.5, start+2*ramp_s+stay+.5)
            used.add(i)
            log['movers'][self.ids[opp][i]] = 'opponent_in_channel'
        # 6. An opponent crossing the wide defender at the middle of the hold (an identity crossing).
        if p.crossing:
            tc = p.t0+p.up+p.hold/2
            k0 = self.k_of(tc-3.0)
            A = self.att_pos(team)
            O = self.att(team, self.pos[opp])
            d = np.hypot(O[k0, :, 0]-A[k0, j, 0], O[k0, :, 1]-A[k0, j, 1])
            for i in used:
                d[i] = np.inf
            d += 1e3*np.array([self.locked(opp, q, tc-3.5, tc+3.0) for q in self.ids[opp]])
            i = int(np.argmin(d))
            target = A[:, j]+[.3*kx, 0.0]
            self.blend(opp, i, ramp(t, tc-3.0, 2.8, .4, 2.0), self.att(opp, self.att(team, target)))
            self.lock(opp, self.ids[opp][i], tc-3.5, tc+3.0)
            q = self.ids[opp][i]
            log['crossing'] = {'t': _r(tc, 3), 'opponent': q}
            log['movers'][q] = 'crossing'
            for who in (pid, q):
                self.issues.append({'id': who, 'time': _r(tc, 3), 'reason': 'Crossed a similar player: check identities.'})
            self.events.append({'time': _r(tc, 3), 'kind': 'swap-uncertain', 'track': None, 'playerId': pid,
                                'message': f'SWAP UNCERTAIN\n{pid} crossed {q}: identities kept, not confirmed'})
            self.low_identity.append((pid, tc-1.5, tc+1.5))
        self.plant_log.append(log)

    def limit_plants(self):
        """Clamp every player to SPEED_CAP after the ball-independent plants (before any carrier is chosen)."""
        self.capped = {}
        for team in ('A', 'B'):
            before = self.pos[team]
            self.pos[team] = limit_speed(before, SPEED_CAP*self.dt)
            self.capped[team] = int((np.hypot(*(self.pos[team]-before).transpose(2, 0, 1)) > .05).sum())

    # ----- the ball
    def carry_pos(self, team, pid, k):
        """The ball at a carrier's feet on frame k: BALL_AHEAD in front of him (his attack direction)."""
        j = self.ids[team].index(pid)
        p = self.pos[team][k, j]
        d = 1.0 if self.side[team] == 'left' else -1.0
        return np.array([p[0]+d*BALL_AHEAD, p[1]])

    def choose(self, team, time, near, exclude=(), horizon=4.0, ball=None):
        """A team-mate to receive or win the ball at `time`: nearest to `near` (pitch metres), among the players no
        plant moves until `horizon` seconds later, preferring a pass of 6-30 m from `ball`."""
        k = self.k_of(time)
        rng = self.rngs['ball']
        best, score = None, math.inf
        for j, pid in enumerate(self.ids[team]):
            if pid in exclude or self.locked(team, pid, time-1.0, time+horizon):
                continue
            p = self.pos[team][k, j]
            s = math.hypot(p[0]-near[0], p[1]-near[1])+rng.uniform(0.0, 4.0)
            if ball is not None:
                pass_m = math.hypot(p[0]-ball[0], p[1]-ball[1])
                s += 10.0*(pass_m < 6.0)+.5*max(0.0, pass_m-30.0)
            if s < score:
                best, score = pid, s
        if best is None:      # everyone is held by a plant: take the nearest anyway
            P = self.pos[team][k]
            best = self.ids[team][int(np.argmin(np.hypot(P[:, 0]-near[0], P[:, 1]-near[1])))]
        return best

    def flight_time(self, start, team, pid, t_release):
        f = 1.0
        for _ in range(3):
            end = self.carry_pos(team, pid, self.k_of(t_release+f))
            f = float(np.clip(math.hypot(end[0]-start[0], end[1]-start[1])/PASS_SPEED, *FLIGHT))
        return f

    def plan_ball(self):
        """Controls (team, pid, ta, tb), flights (t0, t1, start, end: a point or a (team, pid) carrier), out-of-play
        spells and restarts from the spans; takers walk to the restart spot (a plant applied here, before the ball)."""
        s = self.spec
        rng = self.rngs['ball']
        spans = list(s.spans)
        self.controls, self.flights, self.outs, self.changes, self.takers = [], [], [], [], []
        forced = sorted(self.forced, key=lambda f: f[2])
        pending = None                 # (team, pid, t_arrival): the first control of the next possession span
        for n, span in enumerate(spans):
            nxt = spans[n+1] if n+1 < len(spans) else None
            if span.state == 'dead':
                continue
            P = span.state
            if pending is not None and pending[0] == P:
                c, tc = pending[1], pending[2]
            else:
                c, tc = self.choose(P, span.t0, self.focus[self.k_of(span.t0)]), span.t0
            pending = None
            end = span.t1 if nxt is not None else float(self.t[-1])+self.dt
            while True:
                fz = next((f for f in forced if f[0] == P and f[3] > tc+1.5 and f[2] < end-1.0), None)
                tb = tc+rng.uniform(*s.carry)
                if fz is not None and (fz[1] == c or tb+FLIGHT[1] >= fz[2]):
                    # a plant's carrier: the ball goes to him (or stays with him) for the plant's window
                    forced.remove(fz)
                    if fz[1] != c:
                        start = self.carry_pos(P, c, self.k_of(fz[2]-1.0))
                        f = self.flight_time(start, P, fz[1], fz[2]-1.0)
                        tb = max(tc+.4, fz[2]-f)
                        self.controls.append((P, c, tc, tb))
                        self.flights.append((tb, tb+f, (P, c), (P, fz[1])))
                        c, tc = fz[1], tb+f
                    nb = min(fz[3], end-1.5)
                    if nb <= tc+.4:
                        continue
                    ball = self.carry_pos(P, c, self.k_of(nb))
                    r = self.choose(P, nb, self.focus[self.k_of(nb+.8)], exclude=(c,), ball=ball)
                    f = self.flight_time(ball, P, r, nb)
                    if nb+f+CARRY[0] >= end-.3:
                        break                      # he keeps it to the end of the span
                    self.controls.append((P, c, tc, nb))
                    self.flights.append((nb, nb+f, (P, c), (P, r)))
                    c, tc = r, nb+f
                    continue
                if tb+FLIGHT[1]+CARRY[0] >= end-.3:
                    break
                ball = self.carry_pos(P, c, self.k_of(tb))
                r = self.choose(P, tb, self.focus[self.k_of(tb+.8)], exclude=(c,), ball=ball)
                f = self.flight_time(ball, P, r, tb)
                self.controls.append((P, c, tc, tb))
                self.flights.append((tb, tb+f, (P, c), (P, r)))
                c, tc = r, tb+f
            # closing: the last control and what ends the span
            if nxt is None:
                self.controls.append((P, c, tc, end))
            elif nxt.state in ('A', 'B') and nxt.state != P:
                Q = nxt.state
                ball = self.carry_pos(P, c, self.k_of(span.t1-1.0))
                q = self.choose(Q, span.t1, ball, horizon=3.0)
                f = self.flight_time(ball, Q, q, span.t1-1.0)
                tb = max(tc+.4, span.t1-f)       # the flight lands on the scripted change (later only if the
                won = tb+f                         # last control began too late for it)
                self.controls.append((P, c, tc, tb))
                self.flights.append((tb, won, (P, c), (Q, q)))
                self.changes.append({'lostAt': _r(tb, 3), 'wonAt': _r(won, 3), 'from': P, 'to': Q, 'lostBy': c, 'wonBy': q,
                                     'kind': 'interception'})
                pending = (Q, q, won)
            elif nxt.state in ('A', 'B'):
                self.controls.append((P, c, tc, span.t1))     # the same team keeps it: carry on
                pending = (P, c, span.t1)
            else:
                # out of play over the nearest touchline, then a restart by the team of the span after the dead one
                d0, d1 = nxt.t0, nxt.t1
                after = spans[n+2] if n+2 < len(spans) else None
                R = after.state if after is not None and after.state in ('A', 'B') else self.other(P)
                ball = self.carry_pos(P, c, self.k_of(d0-1.0))
                far = ball[1] < self.W/2
                line_y = 0.0 if far else self.W
                beyond = np.array([np.clip(ball[0]+(4.0 if P == 'A' else -4.0)*self.kx, 2.0, self.L-2.0),
                                   -OUT_BEYOND if far else self.W+OUT_BEYOND])
                f = float(np.clip(math.hypot(beyond[0]-ball[0], beyond[1]-ball[1])/KICK_OUT_SPEED, .6, 3.0))
                tb = max(tc+.4, d0+OUT_BEYOND/KICK_OUT_SPEED-f)
                arrive = tb+f
                self.controls.append((P, c, tc, tb))
                self.flights.append((tb, arrive, (P, c), beyond))
                spot = np.array([beyond[0], line_y])
                # the taker walks to the spot (1 m beyond the line) and stays until the restart
                k0 = self.k_of(d0-.5)
                Pr = self.pos[R][k0]
                cand = [j for j, q in enumerate(self.ids[R]) if not self.locked(R, q, d0-1.0, d1+TAKER_BACK+.5)]
                jt = min(cand or range(len(self.ids[R])), key=lambda j: math.hypot(Pr[j, 0]-spot[0], Pr[j, 1]-spot[1]))
                taker = self.ids[R][jt]
                off = min(1.0, self.kx, self.ky)      # 1 m beyond the line (less on a small field: still on it
                stand = np.array([spot[0], -off if far else self.W+off])   # within the analytics' 2% margin)
                walk = max(2.0, d1-PLACED_S+.3-(d0-.5))
                wt = ramp(self.t, d0-.5, walk, (d1+.3)-(d0-.5+walk), TAKER_BACK)
                self.blend(R, jt, wt, self.att(R, stand))
                self.lock(R, taker, d0-1.0, d1+TAKER_BACK+.5)
                self.takers.append({'team': R, 'player': taker, 't0': _r(d0-.5, 3), 't1': _r(d1+.3+TAKER_BACK, 3)})
                self.outs.append({'t0': _r(d0, 3), 't1': _r(d1, 3), 'beyond': beyond, 'spot': spot, 'arrive': arrive,
                                  'side': 'far' if far else 'near', 'restart': R, 'taker': taker})
                r = self.choose(R, d1, self.focus[self.k_of(d1+1.0)], exclude=(taker,), ball=spot)
                f = self.flight_time(spot, R, r, d1)
                self.flights.append((d1, d1+f, spot, (R, r)))
                if R != P:
                    self.changes.append({'lostAt': _r(tb, 3), 'wonAt': _r(d1, 3), 'from': P, 'to': R, 'lostBy': c, 'wonBy': None,
                                         'kind': 'restart'})
                pending = (R, r, d1+f)

    def ball_track(self):
        """Per frame: the ball's pitch position (NaN unseen), state, the carrier's id ('' none) and whether it is still."""
        T = self.T
        xy = np.full((T, 2), np.nan)
        state = np.full(T, 'UNKNOWN', dtype='<U8')
        carrier = np.full(T, '', dtype='<U8')
        placed = np.zeros(T, bool)
        wob = ar1(self.rngs['ball'], T, (2,), self.dt, .5, .12)
        for P, pid, ta, tb in self.controls:
            k0, k1 = self.k_of(ta), self.k_of(tb)
            for k in range(k0, max(k0, k1)+(1 if tb >= self.t[-1] else 0)):
                xy[k] = self.carry_pos(P, pid, k)+wob[k]
                state[k], carrier[k] = 'TRACKED', pid
        for t0, t1, start, end in self.flights:
            k0, k1 = self.k_of(t0), self.k_of(t1)
            a = self.carry_pos(*start, k0) if isinstance(start, tuple) else np.asarray(start, np.float64)
            b = self.carry_pos(*end, k1) if isinstance(end, tuple) else np.asarray(end, np.float64)
            for k in range(k0, k1):
                u = (self.t[k]-t0)/max(t1-t0, 1e-9)
                xy[k] = a+np.clip(u, 0, 1)*(b-a)
                state[k], carrier[k] = 'TRACKED', ''
        for o in self.outs:
            k_arr, k_seen = self.k_of(o['arrive']), self.k_of(o['arrive']+OUT_SEEN)
            xy[k_arr:k_seen] = o['beyond']
            state[k_arr:k_seen], carrier[k_arr:k_seen] = 'TRACKED', ''
            k_unseen, k_place, k_kick = k_seen, self.k_of(o['t1']-PLACED_S), self.k_of(o['t1'])
            xy[k_unseen:k_place] = np.nan
            state[k_unseen:k_place], carrier[k_unseen:k_place] = 'UNKNOWN', ''
            xy[k_place:k_kick] = o['spot']
            state[k_place:k_kick], carrier[k_place:k_kick] = 'TRACKED', ''
            placed[k_place:k_kick] = True
        state[~np.isfinite(xy[:, 0])] = 'UNKNOWN'
        if self.spec.ball == 'sparse':
            phase = (self.t-self.t[0]) % SPARSE_EVERY
            hide = phase >= SPARSE_ON
            xy[hide] = np.nan
            state[hide] = 'UNKNOWN'
        self.ball, self.ball_state, self.carrier, self.ball_placed = xy, state, carrier, placed

    # ----- plants that depend on the ball: how a team defends
    def fade(self, team, before=3.5, gap=1.0, after=.5, back=2.0):
        """(T, N) weights 0..1 with which a ball-dependent plant may move each player: 0 from `gap` s before each of his
        controls until `after` s after it (the ball is at his feet: he stands where the ball was put), a smoothstep
        down over the (before - gap) s before that and up over `back` s after; the same around a plant's lock."""
        out = np.ones((self.T, self.outfield))
        col = {pid: j for j, pid in enumerate(self.ids[team])}
        lead = before-gap

        def cut(j, a, b):
            down = 1.0-_smoothstep((self.t-(a-lead))/lead)
            up = _smoothstep((self.t-b)/back)
            out[:, j] = np.minimum(out[:, j], np.where(self.t < a, down, np.where(self.t > b, up, 0.0)))
        for P, pid, ta, tb in self.controls:
            if P == team:
                cut(col[pid], ta-gap, tb+after)
        for (tm, pid), wins in self.locks.items():
            if tm == team:
                for a, b in wins:
                    cut(col[pid], a, b)
        return out

    def apply_defence(self):
        L, W = self.L, self.W
        radius = max(6.0, 12.0*min(1.0, L/105.0))      # DESIGN 4.1: R = 12 m x min(1, L/105), at least 6 m
        cover_d = min(COVER_D, .8*radius)
        self.pressure_log = []
        for Q in ('A', 'B'):
            d = self.spec.defend.get(Q) or Defend()
            P = self.other(Q)
            G = np.array([0.0, W/2]) if self.side[Q] == 'left' else np.array([L, W/2])
            fade = self.fade(Q)
            prev_end = {}
            for P_, c, ta, tb in self.controls:
                lead = ta-prev_end.get(P_, ta-1.0)       # the pass to this carrier was in the air this long
                prev_end[P_] = tb
                if P_ != P or tb-ta < .6:
                    continue
                k0, k1 = self.k_of(ta), self.k_of(tb)
                jc = self.ids[P].index(c)
                Cxy = self.pos[P][:, jc].copy()
                Cxy[:, 0] += BALL_AHEAD*(1.0 if self.side[P] == 'left' else -1.0)
                Pq = self.pos[Q]
                avail = [j for j in range(self.outfield) if fade[k0:k1+1, j].min() > .99]
                dist = {j: math.hypot(Pq[k0, j, 0]-Cxy[k0, 0], Pq[k0, j, 1]-Cxy[k0, 1]) for j in avail}

                def window(D, ta=ta, tb=tb, lead=lead):
                    up = float(np.clip(1.5*D/PRESS_VMAX, .8, 3.0))
                    start = ta-min(.6*up, max(0.0, lead))
                    return ramp(self.t, start, up, max(0.0, tb-(start+up)), float(np.clip(1.5*D/PRESS_VMAX, 1.2, 3.0)))
                if d.press == 'none':
                    for j in range(self.outfield):
                        rel = Pq[:, j]-Cxy
                        r = np.hypot(rel[:, 0], rel[:, 1])
                        need = window(max(0.0, UNPRESSED_D-float(r[k0])))*(r < UNPRESSED_D)
                        if need.max() <= 0:
                            continue
                        need = np.maximum(smooth(need, self.dt, tau=.3), need)*fade[:, j]
                        u = rel/np.maximum(r, 1e-6)[:, None]
                        self.blend(Q, j, need, self.att(Q, Cxy+UNPRESSED_D*u))
                    self.pressure_log.append({'team': Q, 'carrier': c, 't0': _r(ta, 3), 't1': _r(tb, 3), 'press': 'none',
                                              'first': None, 'cover': None})
                    continue
                if not avail:
                    continue
                fd = min(avail, key=dist.get)
                if dist[fd] > PRESS_REACH:
                    continue
                g = G-Cxy
                g /= np.maximum(np.hypot(g[:, 0], g[:, 1]), 1e-6)[:, None]
                dp = PRESS_D if d.press in ('cover', 'press') else NATURAL_PRESS_D
                fd_target = Cxy+dp*g
                w = window(math.hypot(Pq[k0, fd, 0]-fd_target[k0, 0], Pq[k0, fd, 1]-fd_target[k0, 1]))*fade[:, fd]
                self.blend(Q, fd, w, self.att(Q, fd_target))
                entry = {'team': Q, 'carrier': c, 't0': _r(ta, 3), 't1': _r(tb, 3), 'press': d.press,
                         'first': self.ids[Q][fd], 'cover': None}
                rest = [j for j in avail if j != fd]
                if d.press == 'cover' and rest:
                    kk = min(k0+5, self.T-1)
                    d2 = min(rest, key=lambda j: math.hypot(Pq[kk, j, 0]-fd_target[kk, 0], Pq[kk, j, 1]-fd_target[kk, 1]))
                    gg = G-fd_target
                    gg /= np.maximum(np.hypot(gg[:, 0], gg[:, 1]), 1e-6)[:, None]
                    # the covering player stands behind the first one, turned towards the pitch's middle
                    ca, sa = math.cos(math.radians(COVER_ANGLE)), math.sin(math.radians(COVER_ANGLE))
                    v1 = np.stack([gg[:, 0]*ca-gg[:, 1]*sa, gg[:, 0]*sa+gg[:, 1]*ca], axis=1)
                    v2 = np.stack([gg[:, 0]*ca+gg[:, 1]*sa, -gg[:, 0]*sa+gg[:, 1]*ca], axis=1)
                    p1, p2 = fd_target+cover_d*v1, fd_target+cover_d*v2
                    pick = (np.abs(p1[:, 1]-W/2) <= np.abs(p2[:, 1]-W/2))[:, None]
                    tgt = np.where(pick, p1, p2)
                    tgt[:, 0] = np.clip(tgt[:, 0], 1.0, L-1.0)
                    w2 = window(math.hypot(Pq[k0, d2, 0]-tgt[k0, 0], Pq[k0, d2, 1]-tgt[k0, 1]))*fade[:, d2]
                    self.blend(Q, d2, w2, self.att(Q, tgt))
                    entry['cover'] = self.ids[Q][d2]
                elif d.press == 'press':
                    # nobody covers: every other defender within the cover radius (+1 m) at <= COVER_CLEAR_ANGLE from
                    # the first defender towards the goal is moved out along his bearing to the radius + 3 m
                    ga = math.cos(math.radians(COVER_CLEAR_ANGLE))
                    for j in range(self.outfield):
                        if j == fd:
                            continue
                        rel = Pq[:, j]-fd_target
                        r = np.hypot(rel[:, 0], rel[:, 1])
                        cosang = (rel[:, 0]*g[:, 0]+rel[:, 1]*g[:, 1])/np.maximum(r, 1e-6)
                        need = window(radius+3.0-float(r[k0]))*((r < radius+1.0) & (cosang >= ga))
                        if need.max() <= 0:
                            continue
                        need = np.maximum(smooth(need, self.dt, tau=.3), need)*fade[:, j]
                        u = rel/np.maximum(r, 1e-6)[:, None]
                        tgt = fd_target+(radius+3.0)*u
                        tgt[:, 0] = np.clip(tgt[:, 0], 1.0, L-1.0)
                        tgt[:, 1] = np.clip(tgt[:, 1], .8, W-.8)
                        self.blend(Q, j, need, self.att(Q, tgt))
                self.pressure_log.append(entry)
            if d.weak == 'empty' or d.gap == 'open':
                self.apply_wide_ball(Q, d, self.fade(Q, before=5.0))

    def apply_wide_ball(self, Q, d, fade):
        """With the other team's ball in a wide channel: nobody of Q in the far half (weak 'empty'), or Q's near wide
        defender on the touchline and the near centre-back in the middle with an opponent between them (gap 'open')."""
        ky, W = self.ky, self.W
        P = self.other(Q)
        B = self.att(Q, np.where(np.isfinite(self.ball), self.ball, self.focus))
        live = self.state == P
        wl = smooth((live & (B[:, 1] < W/3)).astype(float), self.dt, tau=WIDE_BALL_TAU)
        wr = smooth((live & (B[:, 1] > 2*W/3)).astype(float), self.dt, tau=WIDE_BALL_TAU)
        roles = self.roles[Q]
        zeros = np.zeros(self.T)
        if d.weak == 'empty':
            A = self.att_pos(Q)
            for j in range(self.outfield):
                far_l = A[:, j, 1] > W/2-3*ky       # ball on the left: the far half is Y > W/2
                far_r = A[:, j, 1] < W/2+3*ky
                self.blend(Q, j, wl*far_l*fade[:, j], np.stack([zeros, np.full(self.T, W/2-4*ky)], axis=1), axes=(1,))
                self.blend(Q, j, wr*far_r*fade[:, j], np.stack([zeros, np.full(self.T, W/2+4*ky)], axis=1), axes=(1,))
        if d.gap == 'open':
            fadeP = self.fade(P, before=5.0, after=1.0, back=3.0)
            Ap = self.att(Q, self.pos[P])
            for lane, wgt in (('left', wl), ('right', wr)):
                sgn = 1.0 if lane == 'left' else -1.0
                wide_y = GAP_OPEN_Y[0]*ky if lane == 'left' else W-GAP_OPEN_Y[0]*ky
                cb_y = W/2-sgn*GAP_OPEN_Y[1]*ky
                for j in range(self.outfield):
                    r = roles[:, j]
                    is_wide = np.array([WIDE_DEFENDERS.get(x) == lane for x in r])
                    is_cb = np.array([x in CENTRE_BACKS and sgn*(self.layout_y(Q, x)-W/2) < 0 for x in r])
                    self.blend(Q, j, wgt*is_wide*fade[:, j], np.stack([zeros, np.full(self.T, wide_y)], axis=1), axes=(1,))
                    self.blend(Q, j, wgt*is_cb*fade[:, j], np.stack([zeros, np.full(self.T, cb_y)], axis=1), axes=(1,))
                # per wide-ball episode, the opponent nearest the gap among those not about to carry the ball steps into
                # it at the back line's depth
                A = self.att_pos(Q)
                backs = [j for j in range(self.outfield) if roles[0, j] in CENTRE_BACKS or roles[0, j] in WIDE_DEFENDERS]
                depth = A[:, backs, 0].mean(axis=1) if backs else self.block[Q]['cx']-12*self.kx
                gap = np.stack([depth+1.0*self.kx, np.full(self.T, (wide_y+cb_y)/2)], axis=1)
                on = wgt > .02
                edges = np.flatnonzero(np.diff(np.r_[0, on.astype(np.int8), 0]))
                for a, b in zip(edges[::2], edges[1::2]):
                    kk = a+int(np.argmax(wgt[a:b]))
                    if wgt[kk] < .5:
                        continue
                    cand = [j for j in range(self.outfield) if fadeP[a:b, j].min() > .99]
                    if not cand:
                        continue
                    jp = min(cand, key=lambda j: math.hypot(Ap[kk, j, 0]-gap[kk, 0], Ap[kk, j, 1]-gap[kk, 1]))
                    seg = np.zeros(self.T)
                    seg[a:b] = wgt[a:b]
                    self.blend(P, jp, seg, self.att(P, self.att(Q, gap)))

    def limit_speeds(self):
        """The final clamp (SPEED_CAP) on every player except while he carries the ball; counts the moved frames."""
        for team in ('A', 'B'):
            exempt = np.zeros((self.T, self.outfield), bool)
            col = {pid: j for j, pid in enumerate(self.ids[team])}
            for P, pid, ta, tb in self.controls:
                if P == team:
                    exempt[(self.t >= ta-1e-9) & (self.t <= tb+1e-9), col[pid]] = True
            before = self.pos[team]
            after = limit_speed(before, SPEED_CAP*self.dt, exempt)
            self.capped[team] += int((np.hypot(*(after-before).transpose(2, 0, 1)) > .05).sum())
            self.pos[team] = after

    def layout_y(self, team, role):
        for name in [n for _, n in self.spec.formations[team]]:
            for r, _, y in LAYOUTS[name]:
                if r == role:
                    return y*self.ky
        return self.W/2

    # ----- the truth recomputed from the true positions (5 Hz)
    def truth(self):
        kx, ky, L, W = self.kx, self.ky, self.L, self.W
        step = max(1, int(round(self.fps/5.0)))
        ks = np.arange(0, self.T, step)
        t = self.t[ks]
        windows = [(min(c['lostAt'], c['wonAt']-2.0), c['wonAt']+2.0) for c in self.changes]
        near = np.zeros(len(ks), bool)
        for a, b in windows:
            near |= (t >= a) & (t <= b)
        lag = max(1, int(round(.4*self.fps)))
        plants = []
        for log in self.plant_log:
            team, opp = log['team'], self.other(log['team'])
            j = self.ids[team].index(log['player'])
            m = (t >= log['t0']) & (t <= log['t1'])
            kk = ks[m]
            A = self.att_pos(team)[kk]
            O = self.att(team, self.pos[opp])[kk]
            ball = self.att(team, self.ball)[kk]
            roles = self.roles[team][kk]
            sgn = 1.0 if log['lane'] == 'left' else -1.0
            outer = W/3 if log['lane'] == 'left' else 2*W/3
            cb = np.array([[r in CENTRE_BACKS for r in row] for row in roles])
            hold = np.array([[r in HOLDING for r in row] for row in roles])
            cbx = np.where(cb.any(axis=1), (A[..., 0]*cb).sum(axis=1)/np.maximum(cb.sum(axis=1), 1), np.nan)
            margin = A[:, j, 0]-cbx
            adv = margin >= ADVANCED_M*kx
            own = self.state[kk] == team
            use = adv & own & ~near[m]
            others = np.ones(self.outfield, bool)
            others[j] = False
            in_outer = sgn*(A[..., 1]-outer) < 0
            test_a = ((A[..., 0] <= A[:, j, 0][:, None]-10*kx) & in_outer & others[None]).any(axis=1)
            test_b = (hold & (A[..., 0] <= cbx[:, None]+5*kx)).any(axis=1)
            Afull = self.att_pos(team)
            cb_near = [jj for jj in range(self.outfield) if log['movers'].get(self.ids[team][jj]) == 'cb_shift']
            near_cb = None
            if not cb_near:
                cands = [jj for jj in range(self.outfield) if roles[0, jj] in CENTRE_BACKS]
                if cands:
                    near_cb = min(cands, key=lambda jj: sgn*self.layout_y(team, roles[0, jj]))
            else:
                near_cb = cb_near[0]
            if near_cb is not None:
                med = float(np.median(Afull[:, near_cb, 1]))
                test_c = sgn*(med-A[:, near_cb, 1]) >= 4*ky
            else:
                test_c = np.zeros(len(kk), bool)
            seen = np.isfinite(ball[:, 0])
            behind = ((A[..., 0] <= ball[:, 0][:, None]-BEHIND_M*kx)).sum(axis=1)
            opp_in = ((O[..., 0] < L/2) & (sgn*(O[..., 1]-outer) < 0)).sum(axis=1)
            # the deepest opponent in the vacated channel (outer third, our half, goal-side of the wide defender)
            # against the centre-back line, on the advanced own-possession samples (fix round 1, critic B-8)
            chan = (O[..., 0] < L/2) & (sgn*(O[..., 1]-outer) < 0) & (O[..., 0] < A[:, j, 0][:, None])
            deep_x = np.where(chan, O[..., 0], np.inf).min(axis=1)
            threat = use & np.isfinite(deep_x)
            Xj = Afull[:, j, 0]
            vel = np.zeros(len(kk))
            for n, k_ in enumerate(kk):
                a, b = max(0, k_-lag), min(self.T-1, k_+lag)
                vel[n] = -(Xj[b]-Xj[a])/max((b-a)*self.dt, 1e-9)
            share = lambda v: _r(float(np.mean(v[use])), 3) if use.any() else None
            entry = dict(log)
            entry['movers'] = dict(sorted(log['movers'].items()))
            bseen = use & seen
            values, counts = np.unique(behind[bseen], return_counts=True) if bseen.any() else ([], [])
            changes = [tc for team_, periods in self.spec.formations.items() if team_ == team for tc, _ in periods[1:]]
            near_form = adv & np.array([any(abs(x-tc) <= FORMATION_NEAR for tc in changes) for x in t[m]], bool)
            entry['measured'] = {
                'advancedSeconds': _r(adv.sum()*step*self.dt, 2),
                'advancedNearFormationChangeSeconds': _r(near_form.sum()*step*self.dt, 2),
                'advancedOwnSeconds': _r(use.sum()*step*self.dt, 2),
                'advancedNearChangeSeconds': _r((adv & near[m]).sum()*step*self.dt, 2),
                'advancedNotOwnSeconds': _r((adv & ~own).sum()*step*self.dt, 2),
                'maxMarginM': _r(np.nanmax(margin), 2) if np.isfinite(margin).any() else None,
                'coverShare': {'channel': share(test_a), 'holding': share(test_b), 'cbShift': share(test_c),
                               'any': share(test_a | test_b | test_c)},
                'behindBall': {str(int(v)): int(c) for v, c in zip(values, counts)},
                'behindShare': (_r(float(np.mean(behind[bseen] == log['behind'])), 3)
                                if bseen.any() and log['behind'] is not None else None),
                'opponentShare': share(opp_in > 0),
                'channelOpponentVsCbM': (_r(float(np.median(deep_x[threat]-cbx[threat])), 2)
                                         if threat.any() else None),
                'recoveringShare': _r(float(np.mean(vel[adv] >= RECOVER_SPEED)), 3) if adv.any() else None,
                'maxGoalwardsMS': _r(float(vel.max()), 2) if len(vel) else None,
            }
            if log['crossing']:
                q = self.ids[opp].index(log['crossing']['opponent'])
                tc = log['crossing']['t']
                kc = (self.t >= tc-1.0) & (self.t <= tc+1.0)
                dd = np.hypot(*(self.pos[team][kc, j]-self.pos[opp][kc, q]).T)
                entry['measured']['crossingMinM'] = _r(float(dd.min()), 2)
            plants.append(entry)
        speeds = {}
        for team in ('A', 'B'):
            P = self.pos[team][ks]
            v = np.hypot(*(np.diff(P, axis=0)).transpose(2, 0, 1))/(step*self.dt)
            speeds[team] = _r(float(v.max()), 2) if v.size else 0.0
        bv = np.hypot(*np.diff(self.ball, axis=0).T)/self.dt
        bv = bv[np.isfinite(bv)]
        return {'plants': plants, 'defence': self.defence_truth(), 'speeds': {'playersMaxMS': speeds, 'ballMaxMS': _r(float(bv.max()), 2) if bv.size else 0.0,
                                             'capped': dict(self.capped)}}

    def defence_truth(self):
        """Per defending team, over the settled samples (5 Hz) where the other team's carrier has the ball (he has had
        it 1.5 s and keeps it .5 s more; no change or dead ball within 3 s): the share pressed
        (nearest outfield defender within PRESSED_M), the share of those covered (a second defender within the cover
        radius at <= 60 deg from the first one towards the own goal: DESIGN 4.1), and with the ball in a wide channel the
        share with nobody goal-side in the far half (DESIGN 4.2) and with a ball-side back-line gap over 20 m holding an
        opponent (DESIGN 4.5). The planted defending behaviours are checked against these."""
        kx, ky, L, W = self.kx, self.ky, self.L, self.W
        step = max(1, int(round(self.fps/5.0)))
        ks = np.arange(0, self.T, step)
        radius = max(6.0, 12.0*min(1.0, L/105.0))
        out = {}
        for Q in ('A', 'B'):
            P = self.other(Q)
            # settled samples: the carrier has had the ball 1.5 s and keeps it .5 s more, no change or dead ball within 3 s
            settled = np.zeros(self.T, bool)
            for P_, _, ta, tb in self.controls:
                if P_ == P:
                    settled[(self.t >= ta+1.5) & (self.t <= tb-.5)] = True
            for c in self.changes:
                settled[(self.t >= c['lostAt']-3.0) & (self.t <= c['wonAt']+3.0)] = False
            for o in self.outs:
                settled[(self.t >= float(o['t0'])-3.0) & (self.t <= float(o['t1'])+3.0)] = False
            sel = ks[settled[ks]]
            if not len(sel):
                out[Q] = None
                continue
            A = self.att(Q, self.pos[Q][sel])
            O = self.att(Q, self.pos[P][sel])
            ball = self.att(Q, self.ball[sel])
            cidx = np.array([self.ids[P].index(c) for c in self.carrier[sel]])
            C = O[np.arange(len(sel)), cidx]
            d = np.hypot(A[..., 0]-C[:, None, 0], A[..., 1]-C[:, None, 1])
            fd = np.argmin(d, axis=1)
            d1 = d[np.arange(len(sel)), fd]
            F = A[np.arange(len(sel)), fd]
            g = np.stack([-F[:, 0], W/2-F[:, 1]], axis=1)
            g /= np.maximum(np.hypot(g[:, 0], g[:, 1]), 1e-9)[:, None]
            rel = A-F[:, None]
            r = np.hypot(rel[..., 0], rel[..., 1])
            cos = (rel[..., 0]*g[:, None, 0]+rel[..., 1]*g[:, None, 1])/np.maximum(r, 1e-9)
            cov = (r <= radius) & (cos >= math.cos(math.radians(60.0))) & (r > 1e-6)
            covered = cov.any(axis=1)
            pressed = d1 <= PRESSED_M
            # wide: the ball has been in that outer third for WIDE_HELD s or more (a defence needs time to shift)
            Bf = self.att(Q, self.ball)
            run_l = _held(Bf[:, 1] < W/3, int(round(WIDE_HELD*self.fps)))
            run_r = _held(Bf[:, 1] > 2*W/3, int(round(WIDE_HELD*self.fps)))
            wide_l, wide_r = run_l[sel], run_r[sel]
            far = np.where(wide_l[:, None], A[..., 1] > W/2, A[..., 1] < W/2)
            goal_side = A[..., 0] <= ball[:, None, 0]
            weak = (far & goal_side).sum(axis=1)
            wide = wide_l | wide_r
            roles = self.roles[Q][sel]
            gaps, holders = np.zeros(len(sel)), np.zeros(len(sel), bool)
            for n in np.flatnonzero(wide):
                back = [j for j in range(self.outfield) if roles[n, j] in CENTRE_BACKS or roles[n, j] in WIDE_DEFENDERS]
                if len(back) < 3:
                    continue
                ys = np.sort(A[n, back, 1])
                depth = float(np.mean(A[n, back, 0]))
                half = ys < W/2 if wide_l[n] else ys > W/2
                pairs = [(ys[i], ys[i+1]) for i in range(len(ys)-1) if half[i] or half[i+1]]
                if wide_l[n]:
                    pairs.append((0.0, ys[0]))
                else:
                    pairs.append((ys[-1], W))
                a, b = max(pairs, key=lambda p: p[1]-p[0])
                gaps[n] = b-a
                holders[n] = bool(((O[n, :, 1] > a) & (O[n, :, 1] < b) & (np.abs(O[n, :, 0]-depth) <= 5*kx)).any())
            share = lambda v, m: _r(float(np.mean(v[m])), 3) if m.any() else None
            out[Q] = {'samples': int(len(sel)), 'pressedShare': share(pressed, np.ones(len(sel), bool)),
                      'coveredShareOfPressed': share(covered, pressed), 'medianD1M': _r(float(np.median(d1)), 2),
                      'wideSamples': int(wide.sum()), 'weakEmptyShare': share(weak == 0, wide),
                      'gapOpenShare': share((gaps > 20*ky) & holders, wide)}
        return out

    # ----- the camera, the observations and the result
    def camera(self):
        s = self.spec
        if s.camera == 'full':
            px = FULL_SPAN_PX/self.L
            self.px = px
            self.u0 = np.full(self.T, (IMG_W-px*self.L)/2)
            self.v0 = np.full(self.T, FULL_TOP_PX)
            return
        px = s.follow_px
        self.px = px
        half_w, half_h = IMG_W/px/2, IMG_H/(px/2)/2
        target = np.where(np.isfinite(self.ball), self.ball, self.focus)
        xc = np.clip(target[:, 0], half_w-6, self.L-half_w+6)
        yc = np.clip(target[:, 1]-10*self.ky, half_h-4, self.W-half_h+4)
        c = smooth(np.stack([xc, yc], axis=1), self.dt, tau=.8, vmax=9.0)
        self.u0 = IMG_W/2-px*c[:, 0]
        self.v0 = IMG_H/2-px/2*c[:, 1]

    def image(self, P):
        """Pixels of pitch points P (..., 2) on every frame (the leading axis is time)."""
        shape = (self.T,)+(1,)*(P.ndim-2)
        u = self.u0.reshape(shape)+self.px*P[..., 0]
        v = self.v0.reshape(shape)+self.px/2*P[..., 1]
        return u, v

    def written_size(self):
        if self.spec.measured:
            return {'length': self.L, 'width': self.W, 'source': 'landmarks'}
        return {'length': 105.0, 'width': 68.0, 'source': 'standard'}

    def homography(self, k):
        size = self.written_size()
        return [self.px*self.L/size['length'], 0.0, float(self.u0[k]), 0.0, self.px/2*self.W/size['width'], float(self.v0[k]),
                0.0, 0.0, 1.0]

    def view(self, k):
        out = []
        for u, v in ((IMG_W, 0), (IMG_W, IMG_H), (0, IMG_H), (0, 0)):
            x = (u-self.u0[k])/self.px/self.L
            y = (v-self.v0[k])/(self.px/2)/self.W
            out.append([_r(x), _r(y)])
        return out

    def markings(self, k):
        """The painted lines the frame's homography predicts, clipped to the image (normalized endpoints)."""
        fld = _soccer('field')
        size = fld.PitchSize(self.L, self.W, 'landmarks')
        out = []
        for _, a, b in fld.lines(size):
            p0 = (self.u0[k]+self.px*a[0], self.v0[k]+self.px/2*a[1])
            p1 = (self.u0[k]+self.px*b[0], self.v0[k]+self.px/2*b[1])
            seg = _clip(p0, p1)
            if seg is None or math.hypot(seg[1][0]-seg[0][0], seg[1][1]-seg[0][1]) < 30.0:
                continue
            out.append([_r(seg[0][0]/IMG_W), _r(seg[0][1]/IMG_H), _r(seg[1][0]/IMG_W), _r(seg[1][1]/IMG_H)])
        return out


def _held(mask, n):
    """mask, True only where it has been True for the last n frames."""
    out = np.zeros(len(mask), bool)
    run = 0
    for k, v in enumerate(mask):
        run = run+1 if v else 0
        out[k] = run >= n
    return out


def _soccer(name):
    """A soccer.* module of this repository (imported on first use: building a scene needs only numpy)."""
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    import importlib
    return importlib.import_module(f'soccer.{name}')


def _clip(p0, p1):
    """Liang-Barsky: the part of segment p0-p1 inside the image, or None."""
    x0, y0 = p0
    dx, dy = p1[0]-x0, p1[1]-y0
    lo, hi = 0.0, 1.0
    for p, q in ((-dx, x0), (dx, IMG_W-x0), (-dy, y0), (dy, IMG_H-y0)):
        if abs(p) < 1e-12:
            if q < 0:
                return None
            continue
        r = q/p
        if p < 0:
            lo = max(lo, r)
        else:
            hi = min(hi, r)
        if lo > hi:
            return None
    return (x0+lo*dx, y0+lo*dy), (x0+hi*dx, y0+hi*dy)

# ---------- observations and the result ----------
def _observe(w):
    """The result.json dict of a world whose positions, ball and camera are final."""
    s = w.spec
    T, L, W, fps = w.T, w.L, w.W, w.fps
    rng = w.rngs['noise']
    k = min(w.kx, w.ky)
    small = s.footage in ('amateur', 'veo')
    noise = NOISE_SD*k*(3.0 if small else 1.0)
    pc = _r(max(.2, s.calibration-.01), 3)
    sigma = {'broadcast': [.3, .3], 'amateur': [1.2, 1.6], 'veo': [.9, 1.3]}[s.footage]
    teams = [('A', w.ids['A']), ('B', w.ids['B'])]
    track = {}
    for team, ids in teams:
        for pid in ids:
            track[pid] = 2+len(track)
    for team in ('A', 'B'):
        if team in w.gk:
            track[w.gk[team]] = 2+len(track)
    rows = []          # (pid, team, role, true (T, 2), noisy (T, 2))
    for team, ids in teams:
        obs = w.pos[team]+ar1(rng, T, (len(ids), 2), w.dt, NOISE_TAU, noise)
        for j, pid in enumerate(ids):
            rows.append((pid, team, 'PLAYER', w.pos[team][:, j], obs[:, j]))
    for team in ('A', 'B'):
        if team in w.gk:
            obs = w.gkpos[team]+ar1(rng, T, (2,), w.dt, NOISE_TAU, noise)
            rows.append((w.gk[team], team, 'GOALKEEPER', w.gkpos[team], obs))
    low = {}
    for pid, a, b in w.low_identity:
        low.setdefault(pid, []).append((a, b))
    per = []
    seen_at = {}
    for pid, team, role, true, obs in rows:
        u, v = w.u0+w.px*true[:, 0], w.v0+w.px/2*true[:, 1]
        vis = (u >= 1) & (u <= IMG_W-1) & (v >= 1) & (v <= IMG_H-1)
        h, bw = BOX_H_M*w.px, BOX_W_M*w.px
        left = np.clip(u-bw/2, 0, IMG_W)
        right = np.clip(u+bw/2, 0, IMG_W)
        top = np.clip(v-h, 0, IMG_H)
        box = np.stack([left/IMG_W, top/IMG_H, (right-left)/IMG_W, (v-top)/IMG_H], axis=1).round(5).tolist()
        pitch = np.stack([obs[:, 0]/L, obs[:, 1]/W], axis=1).round(4).tolist()
        ident = np.full(T, .85)
        for a, b in low.get(pid, ()):
            ident[(w.t >= a) & (w.t <= b)] = .45
        per.append((pid, team, role, vis, box, pitch, ident.round(3).tolist()))
        idx = np.flatnonzero(vis)
        seen_at[pid] = (float(w.t[idx[0]]), float(w.t[idx[-1]]), int(len(idx))) if len(idx) else None
    # the ball
    bt = w.ball
    bvis = np.isfinite(bt[:, 0])
    bu, bv = w.u0+w.px*np.nan_to_num(bt[:, 0]), w.v0+w.px/2*np.nan_to_num(bt[:, 1])
    bvis &= (bu >= 1) & (bu <= IMG_W-1) & (bv >= 1) & (bv <= IMG_H-1)
    bobs = bt+rng.normal(0.0, BALL_NOISE_SD, bt.shape)
    speed = np.zeros(T)
    d = np.hypot(*np.diff(bt, axis=0).T)/w.dt
    speed[1:] = np.where(np.isfinite(d), d, 0.0)
    speed[0] = speed[1] if T > 1 else 0.0
    speed[w.ball_placed] = 0.0
    bd = max(4.0, BALL_D_M*w.px)
    ball_pc = .9 if s.footage == 'broadcast' else pc
    paint_every = max(1, int(round(PAINT_EVERY*fps)))
    # paint only on a full-size field (the markings of field.py are full-size ones)
    paint = s.paint and s.footage == 'broadcast' and (L, W) == (105.0, 68.0)
    static = s.camera == 'full'
    static_marks = w.markings(0) if paint and static else None
    static_h = [_r(x, 6) for x in w.homography(0)] if static else None
    static_view = w.view(0) if static else None
    tlist = [round(float(x), 6) for x in w.t]
    ages = [round(float(x-w.t[0]), 3) for x in w.t]
    frame0 = int(round(w.t[0]*fps))
    frames = []
    for kk in range(T):
        tk = tlist[kk]
        people = []
        for pid, team, role, vis, box, pitch, ident in per:
            if not vis[kk]:
                continue
            gk = role == 'GOALKEEPER'
            people.append({'time': tk, 'track': track[pid], 'box': box[kk], 'score': .85, 'cls': 'player', 'zone': 'inside',
                           'evidence': 'observed', 'id': pid, 'display': (f'GK-{team}' if gk else pid), 'team': team,
                           'role': role, 'label': f'{role}_TEAM_{team}', 'state': 'confirmed', 'identityConfidence': ident[kk],
                           'roleConfidence': 1.0, 'pitch': pitch[kk], 'pitchConfidence': pc, 'pitchSource': 'carried',
                           'pitchSigma': list(sigma)})
        if bvis[kk] and w.ball_state[kk] == 'TRACKED':
            sp = float(speed[kk])
            motion = 'STATIONARY' if sp < 1.0 else 'ROLLING' if sp < 8.0 else 'FAST'
            cu, cv = float(bu[kk]), float(bv[kk])
            near = track.get(w.carrier[kk]) if w.carrier[kk] else None
            ball = {'state': 'TRACKED', 'phase': 'PATH', 'observed': True, 'confidence': .9, 'identity': .95, 'visual': .9,
                    'trajectory': 0.0, 'track': 1, 'source': 'PATH', 'bridged': False, 'origin': 'OBSERVATION',
                    'motion': motion, 'box': [_r((cu-bd/2)/IMG_W, 5), _r((cv-bd)/IMG_H, 5), _r(bd/IMG_W, 5), _r(bd/IMG_H, 5)],
                    'center': [_r(cu/IMG_W, 5), _r((cv-bd/2)/IMG_H, 5)], 'missingFor': 0.0, 'speed': _r(sp, 2),
                    'nearTrack': near, 'pitch': [_r(bobs[kk, 0]/L), _r(bobs[kk, 1]/W)], 'pitchConfidence': ball_pc,
                    'pitchSigma': [.3, .3] if s.footage == 'broadcast' else list(sigma)}
        else:
            ball = {'state': 'UNKNOWN', 'phase': 'PATH', 'observed': False, 'origin': 'UNKNOWN', 'motion': 'UNKNOWN'}
        du = float(w.u0[kk]-w.u0[kk-1]) if kk else 0.0
        dv = float(w.v0[kk]-w.v0[kk-1]) if kk else 0.0
        marks = None
        if paint and kk % paint_every == 0:
            marks = {'reliable': True, 'source': 'synthetic', 'confidence': s.calibration,
                     'level': 'HIGH' if s.calibration >= .7 else 'MEDIUM',
                     'markings': static_marks if static_marks is not None else w.markings(kk)}
        frames.append({'time': tk, 'frame': frame0+kk, 'ball': ball, 'people': people, 'pitch': marks, 'boundary': None,
                       'cameraReliable': True, 'cut': False, 'segment': 0, 'calibrated': True,
                       'motion': [1.0, 0.0, _r(du, 3), 0.0, 1.0, _r(dv, 3)], 'pitchCoordinates': 'predicted',
                       'pitchMode': 'automatic', 'pitchConfidence': s.calibration, 'profile': {'smallScene': small},
                       'calibration': {'confidence': s.calibration, 'temporal': .9, 'observability': .95, 'source': 'carried',
                                       'direction': 'forward', 'anchor': frame0, 'anchorAge': ages[kk],
                                       'disagreement': None,
                                       'H': list(static_h) if static else [_r(x, 6) for x in w.homography(kk)]},
                       'view': [list(v) for v in static_view] if static else w.view(kk), 'visibility': {}})
    players, keepers, slots, labels = [], [], [], {}
    for pid, team, role, *_ in per:
        se = seen_at[pid]
        gk = role == 'GOALKEEPER'
        n = int(pid.split('-')[1])
        long_id = f'Goalkeeper_{n:02d}' if gk else f'Team{team}_Player_{n:02d}'
        entry = {'id': pid, 'longId': long_id, 'display': f'GK-{team}' if gk else pid, 'team': team, 'role': role,
                 'label': f'{role}_TEAM_{team}', 'status': 'ACTIVE', 'firstSeen': _r(se[0], 3) if se else None,
                 'lastSeen': _r(se[1], 3) if se else None, 'observations': se[2] if se else 0, 'identityConfidence': .85,
                 'teamConfidence': .9, 'roleConfidence': 1.0, 'confirmed': True}
        (keepers if gk else players).append(entry)
        labels[pid] = long_id
        slots.append({'id': pid, 'display': entry['display'], 'team': team, 'role': role, 'status': 'ACTIVE',
                      'firstSeen': entry['firstSeen'], 'lastSeen': entry['lastSeen'], 'observedFrames': entry['observations'],
                      'tracks': [track[pid]], 'exitEdge': '', 'left': None, 'manual': False, 'confirmed': True})
    if s.footage == 'amateur':
        anchors = [{'frame': frame0, 'time': _r(w.t[0], 3), 'observability': .3, 'source': 'midfield', 'root': frame0}]
    else:
        anchors = [{'frame': frame0, 'time': _r(w.t[0], 3), 'observability': .95, 'source': 'landmarks', 'root': frame0,
                    'landmarks': 8, 'rmsMetres': .1}]
    lens = {'k': .58, 'frames': 46, 'gain': .32, 'agreement': 1.0} if s.footage == 'veo' else None
    size = w.written_size()
    result = {
        'version': 3,
        'video': {'name': f'synthetic-{s.scene}-{s.variant}.mp4', 'width': IMG_W, 'height': IMG_H, 'fps': fps,
                  'frames': T, 'duration': _r(T/fps, 3)},
        'config': {'start': _r(w.t[0], 3), 'seconds': _r(T/fps, 3), 'maxPerTeam': s.players, 'outfieldPerTeam': w.outfield,
                   'boundaries': [], 'calibrations': []},
        'frames': frames,
        'match': {'players': players, 'goalkeepers': keepers, 'referees': [], 'retired': [], 'excluded': [], 'withdrawn': []},
        'tracks': {}, 'events': sorted(w.events, key=lambda e: (e['time'], e['playerId'])),
        'issues': sorted(w.issues, key=lambda i: (i['time'], i['id'])), 'pitchIssues': [],
        'tactical': {'coordinates': 'normalized pitch position of the feet: x 0..1 from the left goal line to the right goal '
                                    'line, y 0..1 from the far touchline to the near touchline (synthetic scene)',
                     'calibratedFrames': T, 'calibratedShare': 1.0, 'goalEnds': {'left': 'A', 'right': 'B'}, 'pitchSize': size},
        'status': 'complete',
        'summary': {'processedFrames': T, 'pitch': {'registration': {'anchors': anchors, 'lens': lens, 'frames': T,
                                                                     'calibratedFrames': T, 'calibratedShare': 1.0}}},
        'labels': labels,
        'limitations': ['Synthetic scene built by tools/coaching_synth.py: positions are scripted, not tracked.'],
        'roster': {'expected': {'A': s.players, 'B': s.players, 'goalkeepers': 2 if s.goalkeeper else 0},
                   'counts': {'A': len(w.ids['A']), 'B': len(w.ids['B']), 'goalkeepers': len(keepers)}, 'slots': slots},
        'synthetic': {'tool': 'tools/coaching_synth.py', 'version': VERSION, 'scene': s.scene, 'variant': s.variant,
                      'note': 'A synthetic scene: engines must not read this block; its truth is truth.json.'},
    }
    result['sourceSHA256'] = _sha(_spec_json(s))
    return result


def _spec_json(spec):
    from dataclasses import asdict
    return json.loads(json.dumps(asdict(spec), default=str))


def _visibility(w):
    """Outfield players of each team in the camera's view at 5 Hz: min / median / max."""
    step = max(1, int(round(w.fps/5.0)))
    out = {}
    for team in ('A', 'B'):
        P = w.pos[team][::step]
        u = w.u0[::step, None]+w.px*P[..., 0]
        v = w.v0[::step, None]+w.px/2*P[..., 1]
        n = ((u >= 1) & (u <= IMG_W-1) & (v >= 1) & (v <= IMG_H-1)).sum(axis=1)
        out[team] = {'min': int(n.min()), 'median': float(np.median(n)), 'max': int(n.max())}
    return out


def _roles(w):
    out = {}
    for team in ('A', 'B'):
        R = w.roles[team]
        for j, pid in enumerate(w.ids[team]):
            col, spans, start = R[:, j], [], 0
            for k in range(1, w.T+1):
                if k == w.T or col[k] != col[start]:
                    spans.append({'role': str(col[start]), 't0': _r(w.t[start], 3), 't1': _r(w.t[k-1], 3)})
                    start = k
            out[pid] = spans
        if team in w.gk:
            out[w.gk[team]] = [{'role': 'GK', 't0': _r(w.t[0], 3), 't1': _r(w.t[-1], 3)}]
    return out


def _coach_spans(w):
    """The possession and dead spans as a coach would mark them for this run (user decision 1, 2026-10-21): clipped to
    the analysed range. Layout proposed in INTERFACES.md (EVAL-synth); CONTEXT / SERVER own the stored one."""
    first, last = float(w.t[0]), float(w.t[-1])
    spans = []
    for sp in w.spec.spans:
        a, b = max(sp.t0, first), min(sp.t1, last)
        if b > a:
            spans.append({'t0': _r(a, 3), 't1': _r(b, 3), 'state': sp.state})
    return {'version': 1, 'kind': 'coach-possession-spans', 'source': 'coach', 'spans': spans}


def _truth(w, measured):
    s = w.spec
    first, last = float(w.t[0]), float(w.t[-1])
    press = {}
    for e in w.pressure_log:
        key = f"{e['team']}:{e['press']}"
        press[key] = press.get(key, 0)+1
    return {
        'version': VERSION, 'scene': s.scene, 'variant': s.variant, 'title': s.title,
        'segment': {'t0': _r(first, 3), 't1': _r(last, 3), 'fps': w.fps, 'frames': w.T, 'frame0': int(round(first*w.fps)),
                    'seconds': _r(w.T/w.fps, 3)},
        'format': {'players': s.players, 'goalkeeper': s.goalkeeper, 'outfield': w.outfield},
        'field': {'length': w.L, 'width': w.W, 'declared': s.declared, 'written': w.written_size()},
        'footage': s.footage, 'camera': s.camera, 'calibration': s.calibration, 'paint': bool(s.paint and s.footage == 'broadcast'),
        'ball': s.ball, 'sides': {'A': 'left', 'B': 'right'}, 'goalkeepers': dict(w.gk),
        'formations': w.periods, 'roles': _roles(w),
        'possession': [{'t0': _r(sp.t0, 3), 't1': _r(sp.t1, 3), 'state': sp.state} for sp in s.spans],
        'changes': w.changes,
        'outOfPlay': [{'t0': o['t0'], 't1': o['t1'], 'side': o['side'], 'restart': o['restart'], 'taker': o['taker']}
                      for o in w.outs],
        'plants': measured['plants'],
        'pressure': {'controls': press, 'log': w.pressure_log},
        'defending': measured['defence'],
        'defend': {team: vars(d) for team, d in sorted(s.defend.items())},
        'inView': _visibility(w),
        'speeds': measured['speeds'],
        'coachSpans': bool(s.coach_spans),
        'coachRoles': [list(r) for r in s.coach_roles],
        'expected': [dict(e) for e in s.expected],
        'notes': list(s.notes),
    }


def render(spec, observe=True):
    """Spec -> Built (the world computed in order: script, formations, blocks, positions, plants, ball, defence);
    observe=False leaves the result out (about half the time: the truth and the coach spans only)."""
    w = World(spec)
    w.script()
    w.formations()
    w.blocks()
    w.base_positions()
    for p in spec.plants:
        w.apply_advance(p)
    w.limit_plants()
    w.plan_ball()
    w.ball_track()
    w.apply_defence()
    w.limit_speeds()
    w.camera()
    measured = w.truth()
    result = _observe(w) if observe else None
    return Built(spec, result, _truth(w, measured), _coach_spans(w) if spec.coach_spans else None)


# ---------- coaching settings for a scene ----------
def settings_for(built, level=4, age=None, style='unknown', federation='auto', team='A', roles=True, declared=None,
                 intended='scene'):
    """A complete coaching_settings.json (schema kind 'settings') for one team of a scene: its format, its field when
    the scene declares it, the scene's intended formation ('auto': the scripted starting one), the coach-confirmed roles
    of the scene (roles=False drops them), the age (default: the level's typical age, DESIGN 2.2)."""
    s = built.spec
    if intended == 'scene':
        intended = s.formations[team][0][1] if s.intended == 'auto' else s.intended
    show_field = s.declared if declared is None else declared
    block = {'squad': None, 'ageGroup': age or s.age or LEVEL_AGE[level], 'playingLevel': int(level),
             'format': {'players': s.players, 'goalkeeper': s.goalkeeper},
             'field': {'length': float(s.field[0]), 'width': float(s.field[1])} if show_field else None,
             'federation': federation, 'formation': {'intended': intended, 'periods': []}, 'style': style,
             'roles': {pid: [{'role': role, 'from': None, 'to': None, 'set': '2026-10-22T00:00:00'}]
                       for pid, role in s.coach_roles} if roles else {},
             'runLinks': []}
    return {'version': 1, 'teams': {team: block, ('B' if team == 'A' else 'A'): None}, 'updated': None}


# ---------- the scenes (DESIGN 13.1 and the user's coach-span decision) ----------
D1, D2, D3, D4, D5, D6 = ('pressure_cover', 'weak_side_balance', 'wide_defender_advance', 'behind_ball', 'defensive_spacing',
                          'attacking_shape')
ALL = (1, 2, 3, 4, 5, 6, 7)
PHASED = (D1, D2, D3, D4, D5, D6)
# Ball focus paths, metres on 105 x 68 in the possessing team's attack frame (fraction of the span, X, Y).
OWN_FOCUS = ((.25, 50.0, 28.0), (.5, 56.0, 22.0), (.75, 54.0, 38.0), (1.0, 50.0, 32.0))     # middle third, both sides
OPP_FOCUS = ((.3, 52.0, 36.0), (.7, 60.0, 30.0), (1.0, 56.0, 40.0))
OWN_LEFT = ((.25, 50.0, 14.0), (.5, 56.0, 10.0), (.75, 54.0, 16.0), (1.0, 50.0, 12.0))     # A's left wing
OPP_RIGHT = ((.3, 52.0, 56.0), (.7, 60.0, 58.0), (1.0, 56.0, 54.0))                        # B's right = A's left
OPP_WIDE = ((.2, 50.0, 10.0), (.45, 58.0, 8.0), (.55, 56.0, 34.0), (.8, 58.0, 60.0), (1.0, 52.0, 58.0))
OPP_DEEP = ((.3, 64.0, 30.0), (.7, 72.0, 38.0), (1.0, 68.0, 34.0))          # B attacking into A's own half
OPP_DEEP_WIDE = ((.1, 62.0, 9.0), (.45, 70.0, 8.0), (.55, 66.0, 58.0), (.9, 72.0, 60.0), (1.0, 64.0, 56.0))
# One wing for a whole span (gap_open, wave 8): with OPP_DEEP_WIDE's switch each wing lasts about 8 s, of which the
# defence needs about 3.5 s to open the gap, so the gap was open on about half of every D5 opportunity (7 of 25
# failed at level 4, under its 40% failure share): the plant was not what the scene claimed.
OPP_DEEP_LEFT = ((.15, 64.0, 9.0), (.5, 70.0, 8.0), (1.0, 66.0, 10.0))
OPP_DEEP_RIGHT = ((.15, 64.0, 59.0), (.5, 70.0, 60.0), (1.0, 66.0, 58.0))
# Fix round 1 (the coach verifier's scenes): build-up in A's own half; a high possession whose carrier stands in the
# final third; B's wide carrier at A's box edge (17.5 m from A's goal line, 8 m from A's right touchline) and 27 m out;
# A attacking against B's low block.
OWN_BUILDUP = ((.25, 40.0, 28.0), (.5, 46.0, 22.0), (.75, 44.0, 38.0), (1.0, 40.0, 32.0))
OWN_HIGH = ((.25, 88.0, 28.0), (.5, 90.0, 22.0), (.75, 89.0, 38.0), (1.0, 88.0, 32.0))
OPP_BOX_RIGHT = ((.15, 84.0, 9.0), (.5, 87.5, 8.0), (1.0, 86.0, 10.0))
OPP_FLAT_RIGHT = ((.15, 76.0, 9.0), (.5, 78.0, 8.0), (1.0, 77.0, 10.0))
OWN_LOW_BLOCK = ((.25, 80.0, 28.0), (.5, 84.0, 22.0), (.75, 82.0, 40.0), (1.0, 80.0, 34.0))


def cycles(n, own=32.0, opp=12.0, dead=7.0, dead_every=3, own_focus=OWN_FOCUS, opp_focus=OPP_FOCUS, start=T_START):
    """n cycles of A in possession `own` s then B `opp` s; every `dead_every`-th A spell ends with the ball out of play
    (B restarts), the others with a live change (an interception); B always loses the ball to an interception."""
    spans, t = [], float(start)
    for c in range(n):
        spans.append(Span(t, t+own, 'A', own_focus))
        t += own
        if dead_every and c % dead_every == dead_every-1 and c < n-1:
            spans.append(Span(t, t+dead, 'dead'))
            t += dead
        spans.append(Span(t, t+opp, 'B', opp_focus))
        t += opp
    return tuple(spans)


def advances(spans, role='LB', offsets=(3.0, 17.0), limit=None, **kw):
    """Advance plants at these offsets into every A possession span (two distinct spells 14 s apart per 32 s span)."""
    out = []
    for sp in spans:
        if sp.state != 'A':
            continue
        for off in offsets:
            if limit is not None and len(out) >= limit:
                return tuple(out)
            out.append(Advance('A', role, sp.t0+off, **kw))
    return tuple(out)


def long_advances(spans, role='LWB', start=0.0, ramp_s=3.0, end_gap=-.5, **kw):
    """One advance held through each A possession span (a wing-back stands high all the time: his natural spot is
    already about 17 m beyond the back three, so separate short spells would merge into one). The ramps lie in the
    +-2 s windows around the changes, which the phase rules exclude anyway."""
    out = []
    for sp in spans:
        if sp.state == 'A':
            hold = (sp.t1-sp.t0)-start-2*ramp_s-end_gap
            out.append(Advance('A', role, sp.t0+start, up=ramp_s, hold=hold, down=ramp_s, **kw))
    return tuple(out)


def _e(levels, expect, detectors=None, value=None, source='DESIGN 13.1', note=None, style=None):
    """One expectation. style: checked on the report made with that playing style declared (DESIGN 2.6), else with the
    scene's settings (style unknown)."""
    out = {'levels': list(levels), 'expect': expect, 'detectors': list(detectors) if detectors else None, 'value': value,
           'source': source}
    if note:
        out['note'] = note
    if style:
        out['style'] = style
    return out


F442 = {'A': ((T_START, '4-4-2'),), 'B': ((T_START, '4-4-2'),)}
UNSUPPORTED = dict(cover='none', opponent=True)
# Level 7 (wave 8): DESIGN 13.1 says work_on at L4-L7, but DESIGN 2.3 (L7 "with style: unknown, criticisms stay
# review") and 2.6 say a professional team is judged against its declared style: with the style unknown the L7
# criticism is a review, with a declared style a work_on. Both are asserted.
# Fix round 3 (critic B-1, spec Part 5 EARLY DEVELOPMENT / Part 16): levels 1-2 judge an uncovered advance with an
# opponent in the vacated channel (exposed) wherever the ball is; every UNSUPPORTED plant leaves one there, so the
# planted advances are criticised at levels 1-2 as well (fix round 1 asserted the opposite while "someone back" was
# judged in the final third only).
EXPOSED_L12 = _e((1, 2), 'criticised', (D3,), source='critic B-1 (fix round 3)',
                 note='uncovered, an opponent in the vacated channel (exposed): judged wherever the ball is')
EXPECT_UNSUPPORTED = (
    EXPOSED_L12,
    _e((3,), 'review_or_work_on', (D3,), note='per DESIGN 4.9'),
    _e((4, 5, 6), 'work_on', (D3,), note='level-specific claims; at L6 because the channel is exposed'),
    _e((7,), 'review_or_work_on', (D3,), source='DESIGN 2.3 L7, 2.6', note='style unknown: a review'),
    _e((7,), 'at_most_review', (D3,), source='DESIGN 2.3 L7, 2.6', note='style unknown: criticisms stay review'),
    _e((7,), 'work_on', (D3,), source='DESIGN 2.3 L7, 2.6', style='possession',
       note='judged against the declared style; the channel is exposed'),
    _e(ALL, 'same_evidence_across_levels', (D3,)),
)


def _advance_scene(variant):
    spans = cycles(8)
    base = dict(scene='', variant=variant, title='', spans=spans, formations=F442)
    return spans, base


def scene_covered_advance(variant='cb_shift'):
    spans, base = _advance_scene(variant)
    cover = {'cb_shift': 'cb_shift', 'channel': 'channel'}[variant]
    return Spec(**{**base, 'scene': 'covered_advance',
                   'title': 'Left-back advances 20 m with cover (near centre-back shifts 6 m, or a team-mate takes the channel)'},
                plants=advances(spans, cover=cover),
                expected=(_e(ALL, 'no_criticism', (D3,)), _e((3, 4, 5, 6, 7), 'strength_allowed', (D3,))),
                notes=('16 covered advances in 8 possession spans of 32 s (256 s of own possession), 14 s apart.',
                       'cb_shift passes cover test (c) and, with the centre-back inside the outer third, test (a); '
                       'channel passes test (a) only.'))


def scene_unsupported_advance(variant='none_behind'):
    spans, base = _advance_scene(variant)
    behind = {'none_behind': 0, 'one_behind': 1, 'spells4': 0, 'final_third': 0}[variant]
    limit = 4 if variant == 'spells4' else None
    expected = EXPECT_UNSUPPORTED
    notes = ['No local cover test passes: the same-half team-mates tuck inside the outer third, the near centre-back '
             'does not shift; an opponent waits in the vacated channel in A\'s half (exposed).',
             'behind the ball: the near centre-back carries the ball into midfield while the far full-back steps up; '
             '(ii) the other centre-back too (nobody >= 10 m goal-side of the ball), (i) he stays (one player).']
    if variant == 'none_behind':
        expected = (_e((1,), 'try_this_team', (D3, D4), source='critic B-1 (fix round 3)',
                       note='the centre-back carries the ball into midfield, but the advance is exposed: "someone '
                            'ready to protect our goal" for the team, no player named (fix round 1 asserted no '
                            'criticism here)'),
                    _e((2,), 'try_this_team', (D3,), source='critic B-1 (fix round 3)',
                       note='level 2\'s team-level rule: exposed wherever the ball is'))+EXPECT_UNSUPPORTED
    elif variant == 'one_behind':
        notes.append('DESIGN 13.1 states L1-L2 outcomes only for variant (ii); none is asserted here for (i).')
    elif variant == 'final_third':
        spans = cycles(8, own_focus=OWN_HIGH)
        base = dict(base, spans=spans, blocks={'A': {'att_x': (35.0, 80.0)}})
        expected = (_e((1,), 'try_this_team', (D3, D4), source='DESIGN 4.3 L1, critic B-1 (fix round 1)',
                       note='"someone ready to protect our goal"; no player named'),
                    _e((2,), 'try_this_team', (D3,), source='DESIGN 4.3 L2, critic B-1 (fix round 1)'),
                    _e((4, 5, 6), 'review_or_work_on', (D3,),
                       note='uncovered, an opponent in the vacated channel in our half (exposed)'),
                    _e((1, 2, 3), 'no_player_named', (D4,),
                       note='levels 1-3 name nobody (naming never / relative / role words); from level 4 the '
                            'evidence may label the player (naming evidence_only / yes)'))
        notes = ['Possession high in B\'s half (block centre up to 80 m): the near centre-back carries the ball at about '
                 '74-76 m (the final third) while the left-back stands 20 m beyond the centre-back line; nobody of A '
                 'is 10 m goal-side of the ball; no cover test passes; an opponent waits in the vacated channel in '
                 'A\'s half. The advances start 8 and 19 s into each possession, once the block has moved up.']
        return Spec(**{**base, 'scene': 'unsupported_advance',
                       'title': 'Left-back advances 20 m without cover, nobody behind, the ball in the final third'},
                    plants=advances(spans, offsets=(8.0, 19.0), behind=0, **UNSUPPORTED), expected=expected,
                    notes=tuple(notes))
    else:
        expected = (_e(ALL, 'at_most_review', (D3,), source='DESIGN 4.9',
                       note='4 spells are fewer than every level\'s minimum opportunities (8-15)'),
                    _e((3, 4, 5, 6, 7), 'moments_review', (D3,), source='critic S-2 (fix round 3)',
                       note='4 clear advances in 6.1 minutes: the wide-defender check counts its clear advances '
                            'against its own floor (momentsMinSpells), not "the range is too short"'),
                    _e((1, 2), 'criticised', (D3,), source='critic B-1 (fix round 3)',
                       note='exposed advances; 4 are fewer than the level\'s minimum, so a review'))
        notes.append('DESIGN 13.1 asks for 4 spells, but DESIGN 4.9 needs 8-15 opportunities for a work item: with 4 '
                     'the 13.1 outcomes are unreachable. The default variants plant 16; this one is the literal 4.')
    return Spec(**{**base, 'scene': 'unsupported_advance',
                   'title': 'Left-back advances 20 m without cover, opponent in the vacated channel'},
                plants=advances(spans, limit=limit, behind=behind, **UNSUPPORTED), expected=tuple(expected), notes=tuple(notes))


def scene_single_advance(variant='default'):
    spans, base = _advance_scene(variant)
    return Spec(**{**base, 'scene': 'single_advance', 'title': 'One unsupported advance only'},
                plants=advances(spans, limit=1, behind=0, **UNSUPPORTED),
                expected=(_e((3, 4, 5, 6, 7), 'not_assessed', (D3,), 'too_few_situations',
                             source='critic S-1 (fix round 1), S-2 (fix round 3)',
                             note='one advance in 256 s of own possession: under half the level\'s minimum '
                                  'opportunities with enough of the phase, so too few situations to judge, not a short '
                                  'range (DESIGN 13.1 row 3 asked for a one-instance review)'),
                          _e(ALL, 'no_drill', (D3,)),
                          _e((1, 2), 'no_criticism', (D3, D4), source='critic B-1 (fix rounds 1 and 3)',
                             note='one exposed advance is fewer situations than levels 1-2 judge (too few '
                                  'situations): no criticism (fix round 1: the ball was in midfield)')),
                notes=('The first spell of unsupported_advance/none_behind; the left-back stays home otherwise.',))


def scene_identity_crossing(variant='default'):
    spans, base = _advance_scene(variant)
    return Spec(**{**base, 'scene': 'identity_crossing', 'title': 'Unsupported advances with an identity crossing at each'},
                plants=advances(spans, behind=0, crossing=True, **UNSUPPORTED),
                expected=(_e(ALL, 'no_player_named', (D3,)), _e(ALL, 'at_most_review', (D3,))),
                notes=('An opponent runs through the left-back at the middle of every hold (closest .3 m); the result '
                       'reports "Crossed a similar player" for both and a swap-uncertain event, and the left-back\'s '
                       'identity confidence is .45 for 1.5 s either side.',))


def scene_possession_unknown(variant='default'):
    spans, base = _advance_scene(variant)
    return Spec(**{**base, 'scene': 'possession_unknown', 'title': 'Unsupported advances, possession unavailable'},
                plants=advances(spans, behind=0, **UNSUPPORTED), ball='sparse',
                expected=(_e(ALL, 'possession_unavailable', source='DESIGN 3.2'),
                          _e(ALL, 'not_assessed', (D3, D4), 'phase_unknown'),
                          _e(ALL, 'not_assessed', (D1, D2, D5), 'phase_unknown', source='DESIGN 3.2'),
                          _e(ALL, 'describe_only', (D6,), source='DESIGN 4.6')),
                notes=('The ball is seen .5 s every 4 s (12.5% of the frames): possession-4 needs a usable ball on 20% '
                       'and abstains for the whole run.',))


def scene_three_visible(variant='default'):
    spans = cycles(8, own_focus=OWN_LEFT, opp_focus=OPP_RIGHT)
    return Spec(scene='three_visible', variant=variant,
                title='Unsupported advances seen by a zoomed camera (about 3 team-mates in view)',
                spans=spans, formations=F442, camera='follow', plants=advances(spans, behind=1, **UNSUPPORTED),
                expected=(_e(ALL, 'no_team_claims'), _e(ALL, 'not_assessed', (D2, D4, D6), 'team_incomplete', source='DESIGN 3.5')),
                notes=('A camera 27 m wide follows the ball on A\'s left wing (left-biased by 10 m): the left-back, the '
                       'ball carrier and one or two team-mates are in view; truth.inView gives the counts.',
                       'Geometry of unsupported_advance/one_behind with the ball on the left.'))


def scene_low_calibration(variant='default'):
    spans, base = _advance_scene(variant)
    return Spec(**{**base, 'scene': 'low_calibration', 'title': 'Unsupported advances under a MEDIUM registration everywhere'},
                plants=advances(spans, behind=0, opponent_at='high', **UNSUPPORTED), calibration=.55, paint=False,
                expected=(EXPOSED_L12, _e(ALL, 'no_video_zone', (D3,)), _e((4, 5, 6, 7), 'diagram_zone', (D3,)),
                          _e(ALL, 'seek_exact'),
                          _e(ALL, 'zone_goal_side_of_threat', (D3,), source='critic B-8 (fix round 1)')),
                notes=('Registration confidence .55 on every frame and no paint ticks: the best evidence tier is pinned.',
                       'Fix round 1: the channel opponent waits halfway between the centre-back line and the advanced '
                       'left-back (opponent_at high). With him 4 m deeper than the centre-backs, as before, no cover spot '
                       'goal-side of him is reachable in 2 s (critic B-8), no area is proposed at all, and the row\'s '
                       '"diagram only, never on the video" had nothing left to test.'))


def scene_sprint_back(variant='default'):
    spans = cycles(12, own=16.0, opp=10.0, dead_every=0)
    plants = tuple(Advance('A', 'LB', sp.t1-4.5, up=3.5, hold=1.0, down=6.0, recover=True, **UNSUPPORTED)
                   for sp in spans if sp.state == 'A')
    return Spec(scene='sprint_back', variant=variant, title='Left-back caught upfield at every loss, sprinting back',
                spans=spans, formations=F442, plants=plants,
                expected=(_e(ALL, 'no_criticism', (D3,)),),
                notes=('Every A possession ends with an interception while the left-back is 18-20 m beyond the '
                       'centre-back line; he then runs back at 5-8 m/s (>= 3 m/s goalwards) inside the change window.',
                       'Advanced own-possession time outside the +-2 s change windows is under 3 s per spell (truth).'))


def scene_wrong_formation(variant='default'):
    spans = cycles(8)
    roles = (('A-04', 'LWB'),) if variant == 'coach_roles' else ()
    expected = [_e(ALL, 'notice', value='formation_mismatch')]
    if variant == 'coach_roles':
        expected.append(_e(ALL, 'not_downgraded_by_mismatch', (D3,), source='DESIGN 3.4',
                           note='the left wing-back is coach-confirmed'))
    else:
        expected.append(_e((2, 3, 4, 5, 6, 7), 'at_most_review', (D3,),
                           note='role criticism without coach roles downgraded to review'))
        # fix round 3 (critic B-1): level 1 judges the exposure for the team ("someone ready to protect our goal"),
        # with no role or formation word (criterion 3); DESIGN 3.4 downgrades role-based criticism of players, which
        # level 1 never makes, so its try_this stands (a LEAD / coach question, reported in the round's answer)
        expected.append(_e((1,), 'try_this_team', (D3, D4), source='critic B-1 (fix round 3), DESIGN 3.4',
                           note='the exposed advance for the team; no player, role or formation named'))
    return Spec(scene='wrong_formation', variant=variant, title='Coach selects 4-4-2, the team plays 3-5-2',
                spans=spans, formations={'A': ((T_START, '3-5-2'),), 'B': ((T_START, '4-4-2'),)}, intended='4-4-2',
                coach_roles=roles, plants=long_advances(spans, behind=1, **UNSUPPORTED), expected=tuple(expected),
                notes=('A holds 3-5-2 throughout; its left wing-back (A-04) stands 20 m beyond the back three through '
                       'each possession (one long spell per span: 8), uncovered, an opponent in the channel.',))


def scene_formation_change(variant='default'):
    spans = cycles(8)
    a_spans = [sp for sp in spans if sp.state == 'A']
    tc = a_spans[3].t0+10.0
    plants = []
    for n, sp in enumerate(a_spans):
        if n < 3:
            plants += [Advance('A', 'LB', sp.t0+off, behind=1, **UNSUPPORTED) for off in (3.0, 17.0)]
        elif n == 3:
            plants.append(Advance('A', 'LB', sp.t0+3.0, behind=1, **UNSUPPORTED))
            plants.append(Advance('A', 'LWB', sp.t0+17.0, up=3.0, hold=8.0, down=3.0, behind=1, **UNSUPPORTED))
        else:
            plants += long_advances((sp,), behind=1, **UNSUPPORTED)
    return Spec(scene='formation_change', variant=variant, title='Formation change mid-segment (4-4-2 to 3-5-2)',
                spans=spans, formations={'A': ((T_START, '4-4-2'), (tc, '3-5-2')), 'B': ((T_START, '4-4-2'),)},
                plants=tuple(plants), expected=(_e(ALL, 'suppressed_near_formation_change', (D3,), FORMATION_NEAR),),
                notes=(f'A changes shape at {tc:.1f} s (players walk to their new slots over {FORMATION_BLEND:.0f} s; the '
                       'left-back becomes the left wing-back). Before it: two left-back spells per span; after it: the '
                       'wing-back held high through each span. truth.plants[].measured.advancedNearFormationChangeSeconds '
                       'says which spells lie within 15 s of the scripted change; team_shape finds the boundary itself (a '
                       'clear change takes it 15-30 s), so the suppression window is around the measured boundary.',))


def scene_short_segment(variant='default'):
    spans = (Span(T_START, T_START+42.0, 'A', OWN_FOCUS), Span(T_START+42.0, T_START+60.0, 'B', OPP_FOCUS))
    plants = tuple(Advance('A', 'LB', T_START+off, up=4.0, hold=3.0, down=4.0, behind=1, **UNSUPPORTED)
                   for off in (2.0, 15.0, 28.0))
    return Spec(scene='short_segment', variant=variant, title='A 60 s segment with three unsupported advances',
                spans=spans, formations=F442, plants=plants,
                expected=(_e(ALL, 'no_repetition_wording'), _e(ALL, 'work_items_need_spells', value=3),
                          _e(ALL, 'no_work_items', source='DESIGN 4.9',
                             note='42 s of own possession is under every level\'s minimum relevant phase time (60-180 s)')),
                notes=('Three distinct spells 13 s apart in 42 s of possession.',))


def scene_kids_swarm(variant='default'):
    rng = np.random.default_rng(404)
    spans, t = [], T_START
    for c in range(16):
        for team in ('A', 'B'):
            pts = tuple((f, float(rng.uniform(25, 85)), float(rng.uniform(14, 54))) for f in (.5, 1.0))
            spans.append(Span(t, t+7.0, team, pts))
            t += 7.0
        if c % 4 == 3 and c < 15:
            spans.append(Span(t, t+4.0, 'dead'))
            t += 4.0
    measured = variant == 'measured_size'
    expected = [_e((1, 2), 'no_position_criticism'), _e((1, 2), 'max_try_this', value=1),
                _e((1, 2), 'strength_only_if_demonstrated')]
    notes = ['A 30 x 20 m field (declared); all eight players within about 7 m of the ball; the ball changes team every 7 s.']
    if measured:
        notes.append('tactical.pitchSize is the real 30 x 20 m (landmarks), which today\'s pitch model cannot produce '
                     '(soccer/field.py learns sizes of 60-120 x 40-90 m only): possession-4 then measures real metres.')
    else:
        expected = [_e(ALL, 'possession_unavailable', source='DESIGN 2.5 (assumed size)'),
                    _e((1, 2), 'phase_from_coach', source='user decision 1 (2026-10-21)')]+expected
        notes.append('Written as PitchIQ writes a small field: 105 x 68 standard, so possession-4 reads every distance 3.5x '
                     'too long and abstains (the pitch and the image disagree); the coach marked the spans.')
    return Spec(scene='kids_swarm', variant=variant, title='4v4 (no keepers): everyone near the ball', spans=tuple(spans),
                formations={'A': ((T_START, '1-2-1'),), 'B': ((T_START, '1-2-1'),)}, players=4, goalkeeper=False,
                field=(30.0, 20.0), declared=True, measured=measured, swarm=True, age='U8', coach_spans=not measured,
                expected=tuple(expected), notes=tuple(notes))


def scene_small_field_scale(variant='assumed'):
    spans = cycles(8, own=14.0, opp=22.0, dead_every=4)
    declared = variant == 'declared'
    return Spec(scene='small_field_scale', variant=variant, title='7v7 on a 55 x 36 m field, with or without the field declared',
                spans=spans, formations={'A': ((T_START, '1-3-2-1'),), 'B': ((T_START, '1-3-2-1'),)}, players=7,
                field=(55.0, 36.0), declared=declared, paint=False, age='U10',
                defend={'A': Defend(press='cover'), 'B': Defend()},
                expected=((_e(ALL, 'assessed', (D1,), source='DESIGN 2.5'),) if declared else
                          (_e(ALL, 'not_assessed', (D1,), 'scale_unknown'),)),
                notes=('A presses the ball with cover (D1 opportunities in B\'s 22 s spans). Without a declared field the '
                       'scale is the assumed 105 x 68 (positions normalized on the real 55 x 36 m field): body-scale '
                       'detectors must not be assessed.',))


def scene_professional(variant='default'):
    spans = cycles(8)
    plants = []
    if variant != 'no_advances':
        for sp in spans:
            if sp.state == 'A':
                plants.append(Advance('A', 'LB', sp.t0+3.0, cover='cb_shift'))
                plants.append(Advance('A', 'RB', sp.t0+17.0, cover='channel'))
    expected = [_e(ALL, 'no_work_items'), _e(ALL, 'strength_allowed', (D1,), source='EVAL')]
    if variant == 'no_advances':
        expected.append(_e(ALL, 'no_strength', (D3,), source='DESIGN 13.4 #11', note='mutation twin: the demonstrations removed'))
    else:
        expected.append(_e((3, 4, 5, 6, 7), 'strength_allowed', (D3,)))
    return Spec(scene='professional', variant=variant, title='Professional-like team: covered advances, pressure with cover',
                spans=spans, formations={'A': ((T_START, '4-3-3'),), 'B': ((T_START, '4-4-2'),)}, plants=tuple(plants),
                defend={'A': Defend(press='cover'), 'B': Defend()}, expected=tuple(expected),
                notes=('A (4-3-3) alternates left-back (centre-back shift) and right-back (a midfielder takes the '
                       'channel) advances, always covered; when defending its nearest player presses at 3 m with a '
                       'second one 7 m behind him.',))


def scene_stacking(variant='default'):
    spans, base = _advance_scene(variant)
    notes = [f'The left centre-midfielder stands {STACK_GAP:.1f} m outside the outer third, 13 m behind the advanced '
             'left-back: not cover (test a), but inside the natural cover zone\'s 3 m disc.']
    if variant == 'reachable':
        expected = (EXPOSED_L12, _e((4, 5, 6, 7), 'zone_avoids_stacking', (D3,)),
                    _e(ALL, 'zone_goal_side_of_threat', (D3,), source='critic B-8 (fix round 1)'),
                    _e((4, 5, 6), 'work_on', (D3,), note='the same plant as the default: uncovered, exposed'))
        notes.append('Fix round 1: the channel opponent waits halfway between the centre-back line and the advanced '
                     'left-back (opponent_at high), so a cover spot goal-side of him is reachable and the zone must '
                     'keep 3 m off the stacking team-mate.')
        title = 'Unsupported advances with a team-mate where the cover zone would be, a reachable cover spot'
    else:
        expected = (EXPOSED_L12, _e((4, 5, 6, 7), 'zone_avoids_stacking', (D3,)),
                    _e((4, 5, 6, 7), 'no_zone', (D3,), 'no_reachable_fix', source='critic B-8 (fix round 1)',
                       note='the channel opponent is deeper than every spot reachable in 2 s: the principle only'))
        notes.append('Fix round 1 (critic B-8): the channel opponent waits 4 m deeper than the centre-back line, so no '
                     'cover spot level with or goal-side of him is reachable; variant reachable is the twin with a '
                     'reachable spot.')
        title = 'Unsupported advances with a team-mate where the cover zone would be'
    return Spec(**{**base, 'scene': 'stacking', 'title': title},
                plants=advances(spans, behind=1, stacking=True, opponent_at='high' if variant == 'reachable' else 'deep',
                                **UNSUPPORTED),
                expected=expected, notes=tuple(notes))


def scene_channel_threat(variant='deep'):
    """COACH verifier scene 2 (fix round 1, critic B-8): the same uncovered, exposed left-back advances (one
    centre-back stays back, the near one carries the ball into midfield), the channel opponent either deeper than the
    cover region ('deep': the honest output is the principle only) or upfield of it ('high': a cover spot goal-side of
    him is reachable, and the zone must stay there)."""
    spans, base = _advance_scene(variant)
    if variant == 'deep':
        title = 'Uncovered left-back advances, the channel opponent deeper than the cover region'
        expected = (EXPOSED_L12,
                    _e((4, 5, 6), 'work_on', (D3,), note='uncovered, an opponent in the vacated channel (exposed)'),
                    _e((3, 4, 5, 6, 7), 'no_zone', (D3,), 'no_reachable_fix', source='COACH B2 / critic B-8 (fix round 1)',
                       note='nobody reaches a spot level with or goal-side of the opponent in 2 s: principle only'),
                    _e(ALL, 'same_evidence_across_levels', (D3,)))
        notes = ('The opponent waits 4 m deeper than A\'s centre-back line in the outer third the left-back left; the '
                 'cover region (centre-back line - 5 m to the left-back - 10 m) lies upfield of him.',)
    else:
        title = 'Uncovered left-back advances, the channel opponent upfield of the cover region'
        expected = (EXPOSED_L12,
                    _e((4, 5, 6), 'work_on', (D3,), note='uncovered, an opponent in the vacated channel (exposed)'),
                    _e((4, 5, 6, 7), 'diagram_zone', (D3,), source='COACH B2 / critic B-8 (fix round 1)'),
                    _e(ALL, 'zone_goal_side_of_threat', (D3,), source='COACH B2 / critic B-8 (fix round 1)'),
                    _e(ALL, 'same_evidence_across_levels', (D3,)))
        notes = ('The opponent waits halfway between A\'s centre-back line and the advanced left-back (a winger left '
                 'near halfway): a cover spot goal-side of him is reachable.',)
    return Spec(**{**base, 'scene': 'channel_threat', 'title': title},
                plants=advances(spans, behind=1, opponent_at=variant, **UNSUPPORTED), expected=expected, notes=notes)


def scene_amateur_coach_spans(variant='spans'):
    spans = cycles(8)
    footage = 'veo' if variant == 'veo' else 'amateur'
    coach = variant != 'no_spans'
    if coach:
        expected = (_e(ALL, 'possession_unavailable', source='DESIGN 3.2'),
                    _e(ALL, 'phase_from_coach', source='user decision 1 (2026-10-21)'),
                    _e(ALL, 'assessed', (D3, D5, D6), source='user decision 1; DESIGN 4.3, 4.5, 4.6',
                       note='phase from the coach spans; these detectors need no ball'),
                    _e(ALL, 'not_assessed', (D1,), 'no_carrier', source='DESIGN 3.3'),
                    _e((2, 3, 4, 5, 6, 7), 'not_assessed', (D2, D4), 'ball_unreliable',
                       source='DESIGN 3.3 (ball vetoed on small scenes)'),
                    # fix round 3 (critic B-1): level 1's reframed advance is judged on the exposure, which needs no
                    # ball, so behind_ball is assessed there and the exposed advances are criticised (level 2's
                    # advanced rule leaves these 7v7 advances out of scope: no level-2 expectation)
                    _e((1,), 'criticised', (D3,), source='critic B-1 (fix round 3)',
                       note='the exposed advance needs no ball depth'),
                    _e((1,), 'assessed', (D4,), source='critic B-1 (fix round 3)',
                       note='the reframed advance is judged on the exposure, without the ball'))
    else:
        expected = (_e(ALL, 'possession_unavailable', source='DESIGN 3.2'),
                    _e(ALL, 'not_assessed', (D1, D2, D3, D4, D5), 'phase_unknown', source='DESIGN 3.2'),
                    _e(ALL, 'describe_only', (D6,), source='DESIGN 4.6'))
    return Spec(scene='amateur_coach_spans', variant=variant,
                title='Amateur 7v7 clip: possession abstains, the coach marked the spans',
                spans=spans, formations={'A': ((T_START, '1-3-2-1'),), 'B': ((T_START, '1-3-2-1'),)}, players=7,
                field=(55.0, 36.0), declared=True, footage=footage, calibration=.39 if footage == 'amateur' else .45,
                paint=False, coach_spans=coach, age='U10',
                plants=advances(spans, role='back-1/3', **UNSUPPORTED), expected=expected,
                notes=('Small-scene footage (frames[].profile.smallScene on every frame), a keyless MEDIUM registration '
                       '(amateur) or a lens bend (veo): possession-4 abstains (small_run) and the series vetoes the ball.',
                       'coach_spans.json marks every A, B and dead span of the script (state A|B|dead, t0, t1).'))


def scene_defending(variant='press_cover'):
    wide = variant in ('weak_empty', 'gap_open')
    spans = cycles(8, own=12.0, opp=22.0, dead_every=4, opp_focus=OPP_DEEP_WIDE if wide else OPP_DEEP)
    if variant == 'gap_open':
        out, k = [], 0
        for sp in spans:
            if sp.state == 'B':
                sp = replace(sp, focus=(OPP_DEEP_LEFT, OPP_DEEP_RIGHT)[k % 2])
                k += 1
            out.append(sp)
        spans = tuple(out)
    d = {'press_cover': Defend(press='cover'), 'press_no_cover': Defend(press='press'), 'no_press': Defend(press='none'),
         'weak_empty': Defend(press='cover', weak='empty'), 'gap_open': Defend(press='cover', gap='open')}[variant]
    det, outcome = {'press_cover': (D1, 'demonstrated'), 'press_no_cover': (D1, 'not_demonstrated'),
                    'no_press': (D1, 'not_demonstrated'), 'weak_empty': (D2, 'not_demonstrated'),
                    'gap_open': (D5, 'not_demonstrated')}[variant]
    src = 'EVAL (DESIGN 4.1, 4.2, 4.5 definitions)'
    # Wave 8: D2 and D5 are off at levels 1-2 (DESIGN 2.3), so their planted outcome is checked from level 3; at level
    # 1 D1's rule is "goes to the ball" (DESIGN 4.1: demonstrated = pressure), so pressure without cover is a level-1
    # demonstration.
    expected = {'press_no_cover': (_e((1,), 'planted_outcome', (D1,), 'demonstrated', source='DESIGN 4.1 (level 1: '
                                      'demonstrated = pressure)'),
                                   _e((2, 3, 4, 5, 6, 7), 'planted_outcome', (D1,), 'not_demonstrated', source=src)),
                'weak_empty': (_e((3, 4, 5, 6, 7), 'planted_outcome', (D2,), 'not_demonstrated', source=src),
                               _e((3, 4, 5, 6, 7), 'review_or_work_on', (D2,), source='DESIGN 4.2, 4.9')),
                'gap_open': (_e((3, 4, 5, 6, 7), 'planted_outcome', (D5,), 'not_demonstrated', source=src),
                             _e((4, 5, 6, 7), 'review_or_work_on', (D5,), source='DESIGN 4.5, 4.9'))
                }.get(variant, (_e(ALL, 'planted_outcome', (det,), outcome, source=src),))
    return Spec(scene='defending', variant=variant, title='Team A defending: pressure, cover, weak side, back-line gap',
                spans=spans, formations=F442, defend={'A': d, 'B': Defend()}, carry=(3.0, 5.0),
                expected=expected,
                notes=('Not a row of DESIGN 13.1: planted D1 / D2 / D5 outcomes for the detectors "implemented first". '
                       'B keeps the ball 22 s of every 34 s in A\'s half (wide variants: on one wing, then the other), '
                       'carries last 3-5 s so the first defender arrives. truth.defending gives the shares on the settled '
                       'samples.',))


def scene_covered_buildup(variant='default'):
    spans = cycles(8, own_focus=OWN_BUILDUP)
    return Spec(scene='covered_buildup', variant=variant,
                title='Covered left-back advances while the near centre-back carries the ball in the own half',
                spans=spans, formations=F442, plants=advances(spans, cover='channel', behind=0),
                expected=(_e(ALL, 'no_criticism', (D3, D4), source='COACH B1 / critic B-1 (fix round 1)'),
                          _e((3, 4, 5, 6, 7), 'strength_allowed', (D3,))),
                notes=('Build-up: the near centre-back carries the ball at about 31 m (the block held at 35 m), the '
                       'other centre-back and the far full-back step up, nobody is 10 m goal-side of the ball; a '
                       'midfielder takes the vacated channel 15 m behind the left-back (test a) on every advance.',
                       'Before fix round 1 level 1 read these as "nobody stayed back to protect our goal" (the BC60 / '
                       'BC303 Barcelona build-up the verifiers found).'))


def scene_box_edge(variant='box'):
    focus, lo, hi = {'box': (OPP_BOX_RIGHT, 31.0, 84.0), 'flat_line': (OPP_FLAT_RIGHT, 41.0, 74.0)}[variant]
    spans = cycles(8, own=12.0, opp=22.0, dead_every=4, opp_focus=focus)
    where = {'box': 'at the edge of A\'s box, 17.5 m from its goal line', 'flat_line': '27 m from A\'s goal line'}[variant]
    return Spec(scene='box_edge', variant=variant,
                title=f'B\'s wide carrier {where}; A\'s back four level with the ball',
                spans=spans, formations=F442, carry=(3.0, 5.0),
                blocks={'A': {'def_x': (lo, 52.0)}, 'B': {'att_behind': 4.0, 'att_x': (35.0, hi)}},
                expected=(_e((3, 4, 5, 6, 7), 'no_criticism', (D2,), source='COACH B3 / critic B-5 (fix round 1)'),),
                notes=(f'B keeps the ball 22 s of every 34 s on A\'s right wing {where} (its block pushed up so its '
                       'wide midfielder carries there); A\'s block is held so its centre-backs stand level with the '
                       'ball\'s depth (the full-backs about 3 m higher): the far side is protected by players between '
                       'the ball and the goal, never deeper than the ball.',
                       'The plain depth rule (X <= ball) counts nobody goal-side there; the fixed rule counts the far '
                       'full-back goal-side by his distance to the goal centre and calls the centre-backs level.'))


def scene_low_block(variant='default'):
    spans = cycles(8, own_focus=OWN_LOW_BLOCK)
    blocks = {'default': {'A': {'att_behind': 6.0, 'att_x': (35.0, 78.0), 'att_sx': .75},
                          'B': {'def_x': (30.0, 52.0)}},
              'compressed': {'A': {'att_behind': 14.0, 'att_x': (35.0, 70.0), 'att_sx': .45},
                             'B': {'def_x': (30.0, 52.0)}}}[variant]
    if variant == 'default':
        title = 'Positional attack against a low block: forwards on the last line, about 24 m of length'
        expected = (_e((1, 2, 3), 'no_criticism', (D6,), source='COACH B4 / critic B-7 (fix round 1)',
                       note='tall and deep for the space available'), _e((1, 2, 3), 'strength_allowed', (D6,)))
        notes = ('B defends with its back line about 16.5 m from its goal (A frame 88.5 m); A attacks with its forwards '
                 'on that line and its centre-backs about 24 m behind them: a field share of about .23, the whole '
                 'available length.',)
    else:
        title = 'Compressed attack against a low block: forwards 15 m short of the last line'
        # (fix round 2, ASSESS) LEAD decision (c), the coach's recommendation: 11v11 attacking depth is descriptive
        # only at levels 1-3, so the compressed attack is described, never a try_this (fix round 1 expected one)
        expected = (_e((1, 2, 3), 'no_criticism', (D6,), source='LEAD decision (c), fix round 2: 11v11 depth '
                       'described at levels 1-3', note='about half of the available length used'),
                    _e((1, 2, 3), 'no_strength', (D6,), source='LEAD decision (c), fix round 2',
                       note='a flat team is not praised for its shape either'))
        notes = ('The twin of low_block/default with A compressed: about 14 m from back to front, its forwards about '
                 '15 m short of B\'s last line (about half of the available length).',)
    return Spec(scene='low_block', variant=variant, title=title, spans=spans, formations=F442, blocks=blocks,
                expected=expected, notes=notes)


SCENES = {
    'covered_advance': (scene_covered_advance, ('cb_shift', 'channel'), 'DESIGN 13.1 row 1'),
    'unsupported_advance': (scene_unsupported_advance, ('none_behind', 'one_behind', 'spells4', 'final_third'),
                            'DESIGN 13.1 row 2'),
    'single_advance': (scene_single_advance, ('default',), 'DESIGN 13.1 row 3'),
    'identity_crossing': (scene_identity_crossing, ('default',), 'DESIGN 13.1 row 4'),
    'possession_unknown': (scene_possession_unknown, ('default',), 'DESIGN 13.1 row 5'),
    'three_visible': (scene_three_visible, ('default',), 'DESIGN 13.1 row 6'),
    'low_calibration': (scene_low_calibration, ('default',), 'DESIGN 13.1 row 7'),
    'sprint_back': (scene_sprint_back, ('default',), 'DESIGN 13.1 row 8'),
    'wrong_formation': (scene_wrong_formation, ('default', 'coach_roles'), 'DESIGN 13.1 row 9'),
    'formation_change': (scene_formation_change, ('default',), 'DESIGN 13.1 row 10'),
    'short_segment': (scene_short_segment, ('default',), 'DESIGN 13.1 row 11'),
    'kids_swarm': (scene_kids_swarm, ('default', 'measured_size'), 'DESIGN 13.1 row 12'),
    'small_field_scale': (scene_small_field_scale, ('assumed', 'declared'), 'DESIGN 13.1 row 13'),
    'professional': (scene_professional, ('default', 'no_advances'), 'DESIGN 13.1 row 14 (+ 13.4 #11 mutation twin)'),
    'stacking': (scene_stacking, ('default', 'reachable'), 'DESIGN 13.1 row 15'),
    'amateur_coach_spans': (scene_amateur_coach_spans, ('spans', 'no_spans', 'veo'), 'user decision 1 (2026-10-21)'),
    'defending': (scene_defending, ('press_cover', 'press_no_cover', 'no_press', 'weak_empty', 'gap_open'),
                  'EVAL: D1 / D2 / D5 planted outcomes'),
    'covered_buildup': (scene_covered_buildup, ('default',), 'COACH verifier scene 1 (fix round 1, critic B-1)'),
    'box_edge': (scene_box_edge, ('box', 'flat_line'), 'COACH verifier scene 3 (fix round 1, critic B-5)'),
    'low_block': (scene_low_block, ('default', 'compressed'), 'COACH verifier scene 4 (fix round 1, critic B-7)'),
    'channel_threat': (scene_channel_threat, ('deep', 'high'), 'COACH verifier scene 2 (fix round 1, critic B-8)'),
}


def names():
    """[(scene, variant), ...] of the whole catalogue, the default variant of each scene first."""
    return [(scene, v) for scene, (_, variants, _) in SCENES.items() for v in variants]


def spec(scene, variant=None, **overrides):
    if scene not in SCENES:
        raise KeyError(f'unknown scene {scene!r}: one of {", ".join(SCENES)}')
    fn, variants, _ = SCENES[scene]
    variant = variant or variants[0]
    if variant not in variants:
        raise KeyError(f'unknown variant {variant!r} of {scene}: one of {", ".join(variants)}')
    out = fn(variant)
    return replace(out, **overrides) if overrides else out


def build(scene, variant=None, **overrides):
    """Build one scene (overrides: any Spec field, e.g. fps=30.0, seed=1)."""
    return render(spec(scene, variant, **overrides))


def build_truth(scene, variant=None, **overrides):
    """The truth (and coach spans) of a scene without its result (Built.result is None)."""
    return render(spec(scene, variant, **overrides), observe=False)


# ---------- loading a scene through the real consumers ----------
def check(built):
    """Run a scene through what the server runs on a result: analytics.Samples and its reports, TeamShape (observed,
    default settings, the server's slim result), possession.compute (serialized as the server does) and the coaching
    series. Returns numbers (deterministic) for the catalogue and the tests; raises on any error."""
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    from soccer import analytics, possession, team_shape
    from soccer.coaching import series
    r = built.result
    samples = analytics.Samples(r)
    lo, hi = analytics.clamp_range(samples, None, None)
    for scope in ('A', 'B', 'both'):
        json.dumps(analytics.report(samples, '', lo, hi, scope), allow_nan=False)
    first = sorted(p for p, e in analytics.roster(r).items() if e['team'] == 'A' and e['role'] == 'PLAYER')[0]
    json.dumps(analytics.report(samples, first, lo, hi, 'selected'), allow_nan=False)
    slim = {'frames': [{'time': f['time'], 'view': f.get('view')} for f in r['frames']], 'roster': r.get('roster'),
            'tactical': r.get('tactical'), 'match': r.get('match'), 'timeline': r.get('timeline'),
            'corrections': r.get('corrections')}
    shape = team_shape.TeamShape(slim, samples=samples, mode='observed', settings=team_shape.clean_settings({}))
    formations = {}
    for team in ('A', 'B'):
        rep = shape.report(team)
        json.dumps(rep, allow_nan=False)
        json.dumps(shape.snapshot(team, (lo+hi)/2), allow_nan=False)
        periods = ((rep.get('formation') or {}).get('periods') or [])
        formations[team] = [p.get('label') for p in periods]
    rec = possession.compute(r)
    json.dumps(rec, separators=(',', ':'), allow_nan=False)
    cd = series.build_series(r, samples=samples, shape=shape)     # the server passes its cached ones too
    json.dumps(cd.summary(), allow_nan=False)
    truth = built.truth
    # possession against the script, per frame
    times = np.array([f['time'] for f in r['frames']])
    script = np.full(len(times), '', dtype='<U4')
    for sp in truth['possession']:
        script[(times >= sp['t0']-1e-6) & (times < sp['t1']-1e-6)] = sp['state']
    script[times >= truth['possession'][-1]['t1']-1e-6] = truth['possession'][-1]['state']
    named = np.full(len(times), '', dtype='<U4')
    for run in rec.get('timeline') or ():
        if run.get('team') in ('A', 'B'):
            named[(times >= run['t0']-1e-6) & (times <= run['t1']+1e-6)] = run['team']
    live = np.isin(script, ('A', 'B'))
    hit = named != ''
    agree = float(np.mean(named[hit & live] == script[hit & live])) if (hit & live).any() else None
    return {'frames': len(times), 'seconds': truth['segment']['seconds'],
            'analytics': {'identities': len(samples.tracks), 'excluded': {k: v for k, v in samples.excluded.items() if v}},
            'teamShape': formations,
            'possession': {'available': bool(rec.get('available')), 'reason': rec.get('reason'),
                           'level': (rec.get('quality') or {}).get('level'),
                           'coverage': _r(float(np.mean(hit[live])), 3) if live.any() else None,
                           'agreement': _r(agree, 3) if agree is not None else None,
                           'changes': len(rec.get('changes') or ()), 'scriptChanges': len(truth['changes'])},
            'series': {'samples': int(len(cd.t)), 'smallScene': bool(cd.small_scene)}}


# ---------- the catalogue ----------
EXPECT_WORDS = {
    'no_criticism': 'no work_on, try_this or review finding of the detector(s) for team A',
    'criticised': 'a work_on, try_this or review finding of the detector(s) for team A (level 1: the reframed advance '
                  'included)',
    'moments_review': 'a \'moments worth a look\' review of the detector (limitation moments_worth_a_look)',
    'strength_allowed': 'a strength of the detector may appear (a candidate: the scene alone does not require it)',
    'no_strength': 'no strength of the detector',
    'try_this_team': 'a try_this finding phrased for the team (no player named)',
    'team_level': 'the findings name no player (team-level wording)',
    'review_or_work_on': 'the detector\'s criticism is a review or a work_on (DESIGN 4.9 decides which)',
    'work_on': 'a work_on finding of the detector',
    'at_most_review': 'no work_on or try_this of the detector (review, describe or strength only)',
    'review_one_instance': 'a review that says it is one observed instance',
    'no_drill': 'no drill linked to the detector\'s findings',
    'no_player_named': 'the detector\'s findings name no player',
    'possession_unavailable': 'possession-4 abstains for the run (available false)',
    'not_assessed': 'the detector\'s status is not_assessed with the reason in value',
    'assessed': 'the detector\'s status is assessed (opportunities may still be 0)',
    'describe_only': 'only a describe finding (never a verdict)',
    'no_team_claims': 'team-level detectors are not assessed; local relations only',
    'no_video_zone': 'no suggestedZone / moveArrow / channel element on any video overlay',
    'diagram_zone': 'a suggested zone, when one is proposed, is drawn on the top-down diagram',
    'seek_exact': 'every evidence seek equals (frame + 0.5) / fps',
    'notice': 'the overview carries the notice named in value',
    'not_downgraded_by_mismatch': 'the formation-mismatch rule does not downgrade the coach-confirmed role',
    'suppressed_near_formation_change': 'no role-based criticism from opportunities within value s of a measured formation '
                                        'period boundary',
    'no_repetition_wording': 'no "repeated", "pattern" or "often" in any text (range under 3 minutes)',
    'work_items_need_spells': 'every work item has at least value distinct failing spells',
    'no_work_items': 'no work_on or try_this finding at all',
    'no_position_criticism': 'no criticism of positions (D2-D5) for team A',
    'max_try_this': 'at most value try_this findings',
    'strength_only_if_demonstrated': 'every strength rests on demonstrated spells',
    'same_evidence_across_levels': 'the findings of one opportunity family cite the same evidence ids at every level',
    'zone_avoids_stacking': 'no suggested-zone point within 3 m of a team-mate, or no zone',
    'no_zone': 'the detector\'s criticisms draw no suggested area and record the reason in value (vacuous without one)',
    'zone_goal_side_of_threat': 'every suggested area of the detector reaches at most 2 m upfield of the deepest opponent '
                                'in the vacated channel at its moment (critic B-8), or no zone',
    'phase_from_coach': 'opportunities carry phase.source "coach" where the coach marked the span',
    'planted_outcome': 'the detector classifies the planted opportunities as value (DESIGN 4.8 step 2)',
}
GENERATED_BEGIN = '<!-- BEGIN GENERATED: python tools/coaching_synth.py catalogue -->'
GENERATED_END = '<!-- END GENERATED -->'


def _spec_summary(s):
    secs = (s.seconds or (s.spans[-1].t1-s.t_start))
    own = sum(sp.t1-sp.t0 for sp in s.spans if sp.state == 'A')
    plants = len(s.plants)
    fmt = f'{s.players}v{s.players}'+('' if s.goalkeeper else ' (no GK)')
    forms = ' -> '.join(n for _, n in s.formations['A'])
    extra = []
    if s.footage != 'broadcast':
        extra.append(s.footage)
    if s.camera != 'full':
        extra.append(f'{s.camera} camera')
    if s.calibration < .7:
        extra.append(f'registration {s.calibration:.2f}')
    if s.ball != 'tracked':
        extra.append(f'ball {s.ball}')
    if s.coach_spans:
        extra.append('coach spans')
    if s.coach_roles:
        extra.append('coach roles')
    if s.declared:
        extra.append(f'field {s.field[0]:.0f} x {s.field[1]:.0f} m declared')
    elif tuple(s.field) != (105.0, 68.0):
        extra.append(f'field {s.field[0]:.0f} x {s.field[1]:.0f} m not declared')
    return secs, own, plants, fmt, forms, extra


def catalogue_markdown():
    """The generated part of tools/annotations/coaching/synthetic_scenes.md (the scene table and every scene's plants and
    expectations), from the registry: the doc cannot drift from the code (a test compares them)."""
    lines = [GENERATED_BEGIN, '', '## Scenes', '',
             '| Scene / variant | What is planted | Format, A\'s shape | Seconds (A has the ball) | Spells | Footage and settings |',
             '|---|---|---|---|---|---|']
    for scene, variant in names():
        s = spec(scene, variant)
        secs, own, plants, fmt, forms, extra = _spec_summary(s)
        lines.append(f'| `{scene}/{variant}` | {s.title} | {fmt}, {forms} | {secs:.0f} ({own:.0f}) | {plants} | '
                     f'{", ".join(extra) or "broadcast, full view"} |')
    lines += ['', '## Expectations', '',
              'Tokens (each expectation is `{levels, expect, detectors, value, source}` in truth.json and in '
              '`Built.expected`):', '']
    for k, v in EXPECT_WORDS.items():
        lines.append(f'- `{k}`: {v}.')
    for scene, (_, variants, row) in SCENES.items():
        lines += ['', f'### `{scene}` ({row})', '']
        for variant in variants:
            s = spec(scene, variant)
            lines.append(f'- **{variant}**: {s.title}.')
            for e in s.expected:
                lv = 'all levels' if tuple(e['levels']) == ALL else 'L'+','.join(str(x) for x in e['levels'])
                if e.get('style'):
                    lv += f' (style {e["style"]} declared)'
                det = ', '.join(e['detectors']) if e['detectors'] else 'report'
                val = f' = {e["value"]}' if e['value'] is not None else ''
                note = f' ({e["note"]})' if e.get('note') else ''
                lines.append(f'  - {lv}: `{e["expect"]}`{val} for {det} [{e["source"]}]{note}')
            for n in s.notes:
                lines.append(f'  - Note: {n}')
    lines += ['', GENERATED_END]
    return '\n'.join(lines)+'\n'


# ---------- command line ----------
def _safe_dir(path):
    p = Path(path).resolve()
    if any(part.lower() in ('runs', 'uploads') for part in p.parts):
        raise SystemExit(f'refusing to write under a runs/ or uploads/ folder: {p}')
    return p


def _dump(obj, path):
    data = json.dumps(obj, allow_nan=False, ensure_ascii=True, separators=(',', ':'))
    path.write_text(data, encoding='ascii')


def _parse_key(key):
    scene, _, variant = key.partition('/')
    return scene, variant or None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    sub.add_parser('list', help='every scene/variant with its title')
    c = sub.add_parser('check', help='build scenes and load them through analytics, team shape, possession, series')
    c.add_argument('keys', nargs='*', help='scene or scene/variant (default: every default variant)')
    c.add_argument('--all', action='store_true', help='every variant')
    c.add_argument('--fps', type=float, default=None)
    c.add_argument('--json', default=None, help='also write the numbers here (not under runs/ or uploads/)')
    wr = sub.add_parser('write', help='write a scene: result.json, truth.json, coaching_settings.json, coach_spans.json')
    wr.add_argument('key')
    wr.add_argument('--out', required=True)
    wr.add_argument('--level', type=int, default=4)
    wr.add_argument('--fps', type=float, default=None)
    cat = sub.add_parser('catalogue', help='print the generated catalogue (Markdown)')
    cat.add_argument('--write', action='store_true', help=f'replace the generated part of {CATALOGUE.relative_to(REPO)}')
    a = ap.parse_args(argv)
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')
    over = {'fps': a.fps} if getattr(a, 'fps', None) else {}
    if a.cmd == 'list':
        for scene, variant in names():
            s = spec(scene, variant)
            secs = s.seconds or (s.spans[-1].t1-s.t_start)
            print(f'{scene+"/"+variant:<36} {secs:6.0f} s  {s.title}')
        return 0
    if a.cmd == 'catalogue':
        text = catalogue_markdown()
        if not a.write:
            sys.stdout.write(text)
            return 0
        old = CATALOGUE.read_text(encoding='ascii') if CATALOGUE.exists() else ''
        old = old.replace('\r\n', '\n')
        if GENERATED_BEGIN in old and GENERATED_END in old:
            head = old[:old.index(GENERATED_BEGIN)]
            tail = old[old.index(GENERATED_END)+len(GENERATED_END)+1:]
            new = head+text+tail
        else:
            new = old+('\n' if old else '')+text
        CATALOGUE.parent.mkdir(parents=True, exist_ok=True)
        CATALOGUE.write_bytes(new.replace('\n', '\r\n').encode('ascii'))
        print(f'wrote {CATALOGUE}')
        return 0
    if a.cmd == 'write':
        out = _safe_dir(a.out)
        built = build(*_parse_key(a.key), **over)
        out.mkdir(parents=True, exist_ok=True)
        _dump(built.result, out/'result.json')
        _dump(built.truth, out/'truth.json')
        _dump(settings_for(built, level=a.level), out/'coaching_settings.json')
        if built.coach_spans is not None:
            _dump(built.coach_spans, out/'coach_spans.json')
        print(f'wrote {built.key} to {out}')
        return 0
    keys = [_parse_key(k) for k in a.keys] if a.keys else [(s, v) for s, v in names() if a.all or v == SCENES[s][1][0]]
    rows = {}
    for scene, variant in keys:
        built = build(scene, variant, **over)
        rows[built.key] = got = check(built)
        p = got['possession']
        print(f'{built.key:<36} {got["frames"]:5d} frames  possession {"yes" if p["available"] else "no "} '
              f'cov {p["coverage"]} agree {p["agreement"]} changes {p["changes"]}/{p["scriptChanges"]}  '
              f'shape A {got["teamShape"]["A"]}  speed {built.truth["speeds"]["playersMaxMS"]}', flush=True)
    if a.json:
        path = _safe_dir(a.json)
        _dump(rows, path)
    return 0


if __name__ == '__main__':
    sys.exit(main())
