"""Stabilized player trajectories for the 2D replay: per identity, the observed pitch positions of the whole segment
smoothed offline, each observation weighted by what its foot pixel is worth in metres, robust to the boxes that lose
the legs.

Why (measured on the amateur, Veo and broadcast regression runs): a player's pitch position is the foot pixel
(geometry.foot_point, the box's bottom-centre) through the frame's homography. The bottom edge of a detector box is
noisy by about sqrt(0.4^2 + (0.011 h)^2) pixels (h: the box height) on every camera, but a low amateur camera sees the
ground at 5-7 degrees instead of the broadcast's 19, so a vertical pixel is worth about 0.3 m there instead of 0.1 m,
along the camera's line of sight (8-12 times the lateral metres per pixel): the markers bounce, nearly in one
dimension, towards and away from the camera. Half to three quarters of that variance comes from box-extent events
(1.7-3.7% of the samples): the box loses the legs behind a nearer player, ends at an overlapping player's head, or
splits into upper body and legs; the bottom edge jumps while the top stays, and the player is thrown 4-10 m away from
the camera for 0.1-0.7 s. The stored pitchSigma is not a calibrated uncertainty (it assumes FOOT_PIXELS everywhere).

What stabilize() does with a finished result:
    observations   per internal identity (never the display label) the frames the timeline marks O, and per anonymous
                   track (a team, state 'uncertain', no id) the same: the foot pixel, the 2x2 Jacobian of the
                   pixel -> pitch mapping there, the box's height and top, the detection score, the share of the box
                   covered by a nearer person (one whose box ends lower), and whether the frame's bottom cuts the box
    edge fits      per local-track run, the box's top and bottom edges of the neighbouring frames moved into each
                   frame's pixels through the frames' final homographies (H_t H_s^-1: between anchors exactly the camera
                   motion) and fitted locally (quadratic in time over +-FIT_HALF, leave-one-out, biweight), in blocks of
                   CHUNK observations (bounded memory), the windows sized by the typical frame step
    height         the box height an upright player has where the box's top is, from the ground's horizon (H^-T e3):
                   h_geo = r (top - horizon) / (1 - r), r = the identity's running median of h / (foot - horizon) over
                   +-RATIO_HALF (the player's height over the camera's); its own noise fitted per run. The reference
                   h_ref is the box's running median height where the two agree, the geometric one where they do not: a
                   box that loses the legs for over a second drags a running median along, never r
    noise model    the bottom edge's and the centre's pixel noise sqrt(a^2 + (b h)^2), a and b fitted per run on the
                   clean samples (no overlap, not clipped, a confident detection, no event) and shrunk towards the
                   priors, and the inflation for a nearer person in front and for a low score fitted the same way. The
                   measurement covariance is R = J diag(sx^2, sy^2) J^T, which keeps the depth direction's larger error,
                   plus a small registration floor: the smoothing follows the footage's own noise, and a broadcast run
                   is changed little
    box events     the bottom edge off its local fit (or, for a longer event, off its running median over
                   +-HEIGHT_HALF) by more than max(EVENT_PX, EVENT_SHARE h_ref), while the top stays (it moves less
                   than EVENT_TOP of that and less than the limit) and the box is shorter (the legs lost) or longer (it
                   reaches into a nearer player) than h_ref accordingly: the foot is repaired as top + h_ref with a sigma
                   of max(REPAIR_PX, REPAIR_SHARE h_ref), plus the share the repairs of one event have in common
                   (REPAIR_COMMON), and flagged. A box off its geometric height that is no event (a long one both scales
                   follow, a crouch) is doubtful: its bottom's sigma at least the difference, never moved. A whole-box
                   shift is not repaired: the robust weights deal with it
    smoother       a constant-velocity Kalman filter with a Rauch-Tung-Striebel smoother (state x, y, vx, vy; white-
                   noise acceleration; the actual time steps), batched over the identities, robust (iterated reweighting
                   on each observation's leave-one-out Mahalanobis distance, hard rejection far outside) and adaptive:
                   the process noise rises from Q_LOW towards Q_HIGH only where several consecutive good-quality
                   residuals lean the same way, and falls back to VETO_Q where following them would take a player
                   faster than V_PLAUSIBLE or accelerate him harder than A_PLAUSIBLE, so sprints, stops, cuts and
                   reversals are followed (the hardest only with some rounding), a standing player settles, and an
                   excursion that comes back is not followed
    segments       a gap longer than BRIDGE_GAP (the timeline's interpolation limit) ends a trajectory; shorter gaps
                   are bridged by the smoother (kind I, the sigma growing). A jump no player can run (beyond SWITCH_MIN
                   plus SWITCH_SPEED per second and SWITCH_NOISE sigmas of the observations, shown by good observations
                   on both sides and still there SWITCH_PERSIST later or up to the segment's end: an unresolved
                   identity switch) splits the trajectory where the observations start to follow the other player:
                   flagged, never glided across. Identities are never mixed and never averaged with teammates

The output carries only observed (O) and bridged (I) positions, on calibrated frames; what the camera does not see
stays soccer/imputation.py's (its estimate runs on these stabilized observations). A camera-wide error moves every
player alike, and per-player smoothing cannot tell it from motion: it is not removed here (that is the calibration's
to fix). The calibration's slow wobble is smoothed like motion. Its snaps are spread instead:
    step spreading the registration snaps where it re-anchors (the frame's homography H_t against the previous one carried
                   by the frame's camera motion M_t: between anchors the motion row's own error, 0.003-0.006 m, at the
                   regression runs' anchors up to 0.35-0.9 m on the grid and 1-4 m at the far players), moving every
                   player together. A frame whose step (registration_steps) exceeds SPREAD_STEP and stands out of the
                   steps around it (SPREAD_ISOLATION: the registration's gradual changes, the blend between two
                   disagreeing anchors, are 0.03-0.07 m a frame for a second on the amateur run and stay motion) has the
                   world -> world map of its step, A_t = G_t M_t H_{t-1} (registration_maps: a static ground point read
                   at X on frame t-1 is read at A_t X on frame t). Where the registration is anchored (anchor_frames:
                   the frame's anchorAge is 0, or the summary's anchors list it) a step ARRIVES at the anchor (on its
                   frame) or LEAVES it (on the frame after it, into the blend towards the next anchor: on the regression
                   runs every spread step but Veo's five 0.03-0.06 m blips is one of the two). How a step is spread
                   depends on whose anchor it is.
                   Pinned, a user keyframe (PINNED_SOURCES): shown exactly as registered (no correction at all on its
                   frame: the BC keyframes at 10.0 and 44.43 s were moved by 0.26 and 0.49 m on the landmarks when every
                   step was spread before it). A step arriving at it is spread over the frames before it: they move
                   towards A X linearly in time (0 at the ramp's start, the whole step on the frame before the
                   keyframe's), and from the keyframe's frame on the registration holds. A step leaving it is spread
                   over the frames after it: the frames from the step on carry the inverse of the step with a share
                   falling linearly to 0, so the keyframe's frame and the frames before it show its registration. A
                   keyframe that differs from both neighbours gets both ramps.
                   Trusted, a re-registration rooted at a keyframe (TRUSTED_SOURCES, the summary's anchors listing the
                   keyframe's frame as its root: the tracker matched the frame to the keyframe's view, directly or
                   through a relay view rooted at it, which the summary does not tell apart): its steps are spread as a
                   keyframe's, each starting an event of its own, the arriving one before it and the leaving one after
                   it, so a trusted frame no other ramp crosses is shown exactly as registered, good or bad. BC's 738
                   and 1032 (24.6 and 34.4 s) agree with the keyframes' landmarks to 0.12-0.13 m where the blend after
                   them is 0.41-0.46 m off (spread before them as automatic anchors' steps, their leaving steps pulled
                   the frames before them to 0.41 and 0.45 m), but as registered 768, a relay match, is 0.386 m off and
                   1062, a direct match, 0.423 m (shown at 0.19 and 0.23 m only because the leaving ramps of 739 and
                   1033 cross them). Unlike a keyframe it is no bound: other events' ramps run across it (Veo's 18.8 s
                   keyframe's 2.36 m arriving ramp runs back across the 14.4-16.4 s re-registrations rooted at the 13.3
                   s one: bounded there it would have 2.4 s for a 6 s ramp, and every player would snap 2.3 m), and a
                   squeezed ramp of its own turns the other way (moving its frame, never a keyframe) before it snaps. A
                   trusted anchor whose arriving and leaving steps cancel (their composite's step at the people on the
                   frame finite and under SPREAD_STEP at its largest: it disagrees with both neighbours the same way, an
                   outlier) is not honoured: its two steps are an automatic anchor's (honoured, a lone 0.6 m outlier
                   moved every player by more than 0.3 m for 3 s, towards it and back, five 1 s apart and 0.8 m off the
                   same way took the display 2.8 m off, and between two cuts 0.2 s apart it snapped: a 1.18 m spike).
                   The outlier test needs the leaving step on the next frame (an outlier on the frame before a cut or on
                   the last frame, or one over two trusted frames, is honoured: for 1.5-3.1 s every player moved by more
                   than 0.3 m), its cut at SPREAD_STEP is sharp (an outlier whose return leaves 0.025 m is not honoured,
                   one whose return leaves 0.035 m is: 2.8 s over 0.3 m), and the leaving step may be one the adjacency
                   below gives it, an automatic anchor's arrival on the next frame (a trusted anchor that brings a new
                   registration with an automatic outlier back to the old one right after it is then taken for one: the
                   three steps are one automatic event, its frame shown 0.01 m off). A keyframe is honoured always (the
                   user's ground truth). A step on an automatic anchor's frame right after a keyframe or a trusted
                   anchor is the honoured anchor's leaving step, no arrival at the automatic anchor, and it takes the
                   automatic steps after it, each within SPREAD_GAP of the one before (no start, keyframe or trusted
                   frame between), into its own event, spread as the honoured anchor's leaving step (after it): the
                   automatic anchor is passed over, not honoured. An automatic outlier right after a keyframe or a
                   trusted anchor (out on its frame, back on the next) then composes to about the identity, and the
                   honoured frame shows its registration (as an arrival grouped with the automatic anchor's leaving step
                   and spread before it, the event ran across a trusted frame: 0.4 m off its registration; with the
                   honoured anchor's leaving step an event of its own and the automatic anchor's leaving step another,
                   spread before it back across the trusted frame, the two ramps no longer cancelled: a 0.5 m outlier
                   after a trusted anchor that agrees with both neighbours took the display and the trusted frame 0.48 m
                   off for 4 s). A trusted outlier right after a keyframe or a trusted anchor is no such automatic
                   anchor: its arriving step stays its own, the one the outlier test composes with its leaving step
                   (taken for the honoured anchor's leaving step, the outlier went unseen and stayed honoured, its own
                   leaving step an event of its own: with room, two ramps after it that nearly cancelled; with a
                   registration start 0.1 s after it, both snapped, a 0.6 m spike), so once demoted its two steps are
                   one automatic event, which no honoured anchor's event takes right after it. A paint refinement, an
                   auto-midfield anchor, a re-registration rooted at an automatic anchor (all of the amateur run's) and
                   an anchor only a frame's calibration shows (no root known) are automatic.
                   Automatic anchors are not honoured (they disagree with each other by ~0.5 m on the amateur run, and
                   honouring each exactly forced glides): an automatic anchor's arriving and leaving steps and the other
                   automatic steps within SPREAD_GAP of them (no keyframe, trusted anchor or registration start between)
                   are one event [t_a, t_b], and so are the steps at no anchor (blips) within SPREAD_GAP of each other,
                   but for those an honoured anchor's leaving step takes (above). The frames inside an event are carried
                   through its remaining steps: they show the registration after it (an outlier anchor, a frame that
                   jumps away and the next that jumps back, composes to about the identity and disappears), and the
                   event's net step A_{t_b} ... A_{t_a} is spread before it as an arriving step is. (Spread after them
                   like a keyframe's, the leaving steps held the anchor's registration for seconds where the blend after
                   it was better: BC's 43.2 s paint anchor took A-02's hand mark from 0.22 to 0.65 m and the 44.43 s
                   keyframe's landmarks 1 s before it from 0.31 to 0.51 m; and an outlier anchor became two ramps every
                   player drifted along, towards it and back.)
                   A ramp is as long as the event's step at the people (the median of |A X - X| over the stored
                   positions of the people on the frame before it, everyone within PITCH_MARGIN of the pitch whatever
                   their identity, role or state, so a correction, which relabels people and never moves them, never
                   changes a ramp; the visible pitch's STEP_GRID without any) needs to glide at GLIDE_SPEED, within
                   RAMP_MIN and RAMP_MAX, and it is cut short only by a pinned frame, a camera cut or another
                   registration start (chain_starts), never by another event: other events' ramps run across automatic
                   and trusted anchors. Its room squeezes a ramp that is shorter than RAMP_FLOOR or glides the median
                   player faster than SQUEEZE times the larger of GLIDE_SPEED and its full ramp's glide (the step over
                   the ramp it asks for: faster than GLIDE_SPEED where RAMP_MAX caps that ramp), so a ramp at least
                   1/SQUEEZE as long as its step asks for is never squeezed. A squeezed ramp is spread the other way
                   instead when that way has the room and moves no pinned frame (an automatic event or a trusted
                   anchor's arriving step right after a registration start or a keyframe: after it; a trusted anchor's
                   leaving step right before one: before it), and else not at all: it keeps the jump freedom of an
                   unspread step, an honest snap instead of a fast glide (a keyframe's leaving step 0.2 s before a cut,
                   squeezed into that room, glided every player at 4 m/s). Only an event that cancels is spread when
                   neither way has the room: one of more than one step whose net step at the people is finite and under
                   SPREAD_STEP at its largest (an outlier there and back, composing to about the identity) is spread its
                   first way however little room that has, its frames carried through its steps, so no player moves by
                   SPREAD_STEP at once (between two cuts 0.2 s apart such an outlier snapped: a 1.16 m spike on its
                   frame). A single step never cancels (spread into a frame of room it moved no frame and lost its jump
                   freedom), nor does a net turn about the players' centre (0.018 m at the median person, 0.42 m at the
                   farthest: spread into one frame, a far point jumped 0.36 m). Overlapping ramps add their
                   displacements, each step's carried in its own registration: a position on frame t is carried to the
                   registration just before the step through the events' steps between them, so every step's jump is
                   cancelled exactly whatever overlaps it, and an event's steps add up to its net step (spread_steps).
                   Every observation, identified or anonymous, is corrected so before it is smoothed. A spread step gets
                   no jump freedom: its observations no longer jump (one above REGISTRATION_STEP is still never an
                   identity switch). A step that is not spread (above REGISTRATION_STEP, not standing out, or squeezed
                   both ways) keeps the jump freedom (STEP_FREEDOM: each player takes it from its own observations), and
                   a camera cut is a step of unknown size (CUT_STEP), never spread across. The raw observations are
                   never changed; the corrections are summarized (noise.registrationSteps: the events, why and which way
                   each is spread, whether it is a keyframe's or a trusted anchor's, the squeezed ones, the trusted
                   frames against the trusted anchors' events (a trusted outlier has none), the pinned frames' largest
                   correction; summary.spreadMetres). Known limits (synthetic probes; none occurs on the three
                   regression runs, which have three trusted spread steps between them, all BC's, no trusted arriving
                   and leaving pair and no automatic anchor right after an honoured one, and whose trusted leaving
                   ramps, BC's 739 and 1033, leave the registrations' range by at most 0.05 m at the median landmark): a
                   chain of near-outliers (returns that leave 0.05-0.1 m, honoured) takes the display 0.4-0.6 m beyond
                   the registrations; a leaving step spread after into a blend that converges back to the anchor on its
                   own overshoots (0.37 m for 0.6 m back in 1 s); the overlapping ramps of a staircase of trusted
                   anchors glide the fastest player at up to 1.5 m/s; and an automatic outlier right after a trusted
                   anchor that holds longer than SPREAD_GAP before it returns is two events again, one spread after the
                   trusted anchor and one before its return (0.5 m held 0.23-1 s: the display and the trusted frame
                   0.3-0.45 m off for 3-4 s; after a keyframe, a bound, nothing).
Pure: the same result gives the same trajectories (a corrected run may hand its original's noise model in). All
positions in metres on the pitch (x along the length, y across), times in seconds; smooth_tracks() is the array-level
smoother the Metrica bench (tools/trajectory_bench.py) also runs on simulated observations.
"""
import copy
import math
import numbers
import time

import numpy as np

from . import field
from .imputation import DEFAULT_SIGMA, INTERPOLATE_GAP
from .imputation import _precise_on_small_scene as _precise   # the timeline's own rule for what is an observation
from .registration import CLIPPED_FOOT, PITCH_MARGIN, clipped_below

VERSION = 1                    # of the result's stabilized block built from stabilize()
METHOD = ('robust adaptive RTS v1: constant-velocity Kalman + Rauch-Tung-Striebel smoother per identity, measurement '
          'covariance from the foot pixel through the homography with the pixel noise fitted per run, box-extent repair '
          'against the geometric height from the box top, iterated robust reweighting, process noise raised only for '
          'plausible manoeuvres')
BRIDGE_GAP = INTERPOLATE_GAP   # s: a longer gap between two observations ends a trajectory; shorter ones are bridged (I)
MIN_CONFIDENCE = .2            # pitchConfidence under which a position is no observation (imputation.estimate's default)

# Edge fits and box-extent events (the regression runs' measurements; see the module docstring).
FIT_HALF = .4          # s: half window of the local edge fits (the events last 0.1-0.7 s; the gait is slower than this)
FIT_SIDE = 3           # neighbours needed on each side of a sample for its fit (no extrapolation at a run's ends)
HEIGHT_HALF = 1.0      # s: half window of the running median box height h_ref (an event covers at most a third of it)
HEIGHT_MIN = 5         # samples in that window for h_ref
HEIGHT_STRIDE = 2      # every other neighbour is enough for those medians (the same events on the regression runs but a
                       # handful, at half the cost)
CHUNK = 8192           # observations per block of the edge fits (their neighbour matrices take 5-15 kB per observation)
LEVEL_ROWS = CHUNK     # observations the footage's noise level is measured on (an even subset of a larger set: a median
                       # over some 600 000 residuals, one block of the wide fit)
EVENT_PX = 3.0         # px: an event moves the bottom edge by at least this ...
EVENT_SHARE = .15      # ... and this share of h_ref (the diagnosis' rule: 1.7-3.7% of the samples, 47-73% of the variance) ...
EVENT_TOP = .5         # ... while the top edge moves less than this share of it
HEIGHT_AGREE = .5      # ... and the box is shorter (longer) than h_ref by this share of the limit at least
REPAIR_PX = 3.0        # px: the repaired foot's sigma is at least this ...
REPAIR_SHARE = .1      # ... and this share of h_ref (how well the reference height predicts the legs) ...
REPAIR_COMMON = .04    # ... and the repairs of one event share this share of it (r's own error: the identity's median
                       # ratio is the same for all of them), counted once per repair within +-HEIGHT_HALF
# The geometric height reference (geometric_heights): r = box height / the foot's distance below the horizon, per identity.
RATIO_HALF = 5.0       # s: r is the identity's running median over +-this (posture changes it, not the motion)
RATIO_MIN = 15         # clean samples that median needs (else the identity's median, else the run's)
RATIO_STEP = .5        # s: the grid the running median is evaluated on
RATIO_RANGE = (.01, .9)   # r = player height / camera height: a camera lower than about 2 m has no geometric height
GEO_PRIOR = (2.0, .05)    # px, per box height: the geometric height's noise prior (h - h_geo on the clean samples: a 1.5-3.7
                          # px, b .03-.07 on the regression runs)
GEO_A_RANGE, GEO_B_RANGE = (.5, 10.0), (.01, .2)
GEO_AGREE = 3.0        # sigmas of the geometric height within which the running median height is the reference (it is
                       # the more precise one where no long event moved it)
DOUBT_SIGMA = 2.0      # sigmas of the geometric height (and the event limit) a box with a steady top must be off it to be
                       # doubtful ...
DOUBT_SPELL = 1.0      # s: ... and, when it is taller than that, in a spell of taller boxes no longer than this (taller for
                       # longer says the identity's ratio is wrong there, not the boxes: another person after a switch, a
                       # far player where the horizon is uncertain; amateur A-03 at 160.5-161.6 s. A shorter box stays
                       # doubtful however long: a long truncation, or a crouch, whose position doubt does not move)
BIWEIGHT = 4.685       # Tukey's constant: neighbours beyond this many prior sigmas from the fit have no weight

# The pixel noise model sigma^2 = a^2 + (b h)^2 and its fit (sigma of the camera-compensated leave-one-out residuals).
PRIOR_BOTTOM = (.45, .011)     # px, per box height: the bottom edge on the three regression runs (a .32-.47, b .010-.011)
PRIOR_CENTRE = (.5, .008)      # px, per box height: the centre x (a .46-.52, b .008 on the amateur and broadcast runs)
PRIOR_SAMPLES = 400            # clean samples the prior counts for when it is blended with a run's fit
A_RANGE, B_RANGE = (.2, 2.0), (.003, .04)   # the fitted a and b are kept inside these
HEIGHT_BINS = (0, 10, 15, 20, 30, 45, 60, 90, 120, 180, 1e9)   # px: the box-height bins of the fit
BIN_MIN = 50                   # clean samples a bin needs to enter the fit
OCCLUDED = .2                  # share of the box covered by a nearer person from which the box counts as occluded
LOW_SCORE = .5                 # detection score under which a box counts as uncertain
PRIOR_OCCLUDED = 3.0           # the bottom edge's noise times this with a nearer person in front (measured 2.9-3.9)
PRIOR_LOW_SCORE = 2.5          # ... and with a low score (measured 1.8-4.1)
FACTOR_SAMPLES = 200           # samples the factors' priors count for
FACTOR_RANGE = (1.0, 6.0)
REGISTRATION_FLOOR = .05       # m: added in quadrature to every observation (the registration's own frame-to-frame jitter)

# The smoother. q is the spectral density of the white-noise acceleration (m^2/s^3). Chosen on the synthetic manoeuvres
# of tests/test_trajectory.py and on tools/trajectory_bench.py, game 1 (2 x 120 s, pole and broadcast cameras): there
# the stabilizer's RMSE was .201/.101 m with these values, .191/.099 with Q_LOW .3 (turns .74 m instead of .65, and the
# synthetic reversal and stop slightly worse), .212/.102 with Q_LOW 1; Q_HIGH 40-150 and LEAN_START 3-6 changed it by
# under .005 m.
Q_LOW = .5             # settled: a standing or jogging player (the jitter of a white-residual track is averaged out)
Q_HIGH = 80.0          # manoeuvring: sqrt(q 0.5 s) = a 6 m/s change of velocity within half a second
LEAN_WINDOW = .4       # s: residuals within this window around a step are averaged for the lean
LEAN_MIN = 4           # good residuals the window needs
LEAN_START, LEAN_FULL = 4.0, 12.0   # lean (n |mean whitened residual|^2, chi-square with 2 degrees of freedom when the
                                    # residuals are white) where q starts to rise and where it reaches Q_HIGH
V_PLAUSIBLE = 12.0     # m/s: a raised q that makes the trajectory faster than this ...
A_PLAUSIBLE = 40.0     # m/s^2: ... or change its velocity faster than this over +-ACC_HALF is withdrawn (it followed an
ACC_HALF = .1          # s       excursion, not a player: on the Metrica games the players' acceleration over +-0.1 s is under
                       # 4 m/s^2 in 99% of the samples and 12-17 m/s^2 at p99.9; a 3-4 m box event that snaps out and back
                       # needs well over 50. Not lower: at 20 the smoothed estimate of a hard stop or cut (the synthetic
                       # tests' 16-31 m/s^2, plus the estimate's own overshoot) was withdrawn too, and the stop glided
                       # 0.2 m, the cut was rounded by 1 m)
VETO_Q = 3.0           # the process noise a withdrawn raise falls back to (above the bench's fixed-q knee of 1, not Q_LOW: a
                       # real hard turn on low-noise footage trips the veto too, through the estimate's own noise). On the
                       # game-2 bench (tools/trajectory_bench.py, 4 x 300 s, clean samples) against Q_LOW: broadcast turns
                       # .181 m instead of .209, sprints .128 / .139, all .073 / .076; pole turns .326 / .339, sprints
                       # .146 / .156, all .139 / .138. Without the veto the pole camera's samples are .142 instead of .132
INLIER_D2 = 9.21       # leave-one-out Mahalanobis distance^2 up to which an observation keeps its full weight (chi2 99%)
REJECT_D2 = 80.0       # ... beyond which it is rejected (about 9 sigma); between them the weight is INLIER_D2 / d2
FLAG_WEIGHT = .5       # observations weighted under this are flagged as down-weighted
ITERATIONS = 3         # reweighting passes after the first (on the bench's pole camera .174/.213 m against .182/.216 with 1;
                       # equal on the broadcast camera)
DIFFUSE = 50.0         # m: the position prior of a segment's first step (the first observation decides it)
V0 = 5.0               # m/s: the velocity prior of a segment's first step
SWITCH_SPEED = 12.0    # m/s: a jump beyond this (plus SWITCH_MIN) between two observations ...
SWITCH_MIN = 3.0       # m
SWITCH_WINDOW = .5     # s: ... that the robust line fits over this much on each side confirm ...
SWITCH_SAMPLES = 5     # ... with at least this many observations a side ...
SWITCH_SIGMA = 5.0     # ... and this many sigmas of their endpoints ...
SWITCH_NOISE = 4.0     # ... and SWITCH_MIN + SWITCH_SPEED dt plus this many sigmas of the two observations along the jump
                       # (a far player's 3 px box step on the amateur run, 0.8 m a pixel in depth, was a 3.7 m 'jump') ...
SWITCH_PERSIST = 1.0   # s: ... and that has not come back this long after (longer than the longest box event, 0.7 s: an
                       # excursion that comes back is the robust weights'), splits the trajectory; one split per jump
REGISTRATION_STEP = .1 # m: a frame whose registration moved the visible pitch by more than this (the median over a grid of
                       # pixels against the previous frame's registration carried by the frame's camera motion; between
                       # anchors 0.003-0.006 m, at the regression runs' anchors up to 0.35-0.88 m) is a calibration step,
                       # and no jump across it is taken for an identity switch (every player jumps there)
STEP_GRID = (np.linspace(.1, .9, 9), np.linspace(.45, .95, 6))   # normalized pixels where the registration step is measured
STEP_FREEDOM = 3.0     # at a calibration step that is not spread every position may jump: (this times the step)^2 m^2 of
                       # position process noise per axis on that frame (the grid median understates the far players' step
                       # 2-3 times), so each player takes the calibration's step from its own observations at once instead
                       # of gliding across it with its observations rejected
CUT_STEP = 1.0         # m: a camera cut (frame cut, or a new segment) is a calibration step of unknown size, this one's
# Step spreading (the module docstring), measured on the amateur, Veo and broadcast control runs of Oct 18: the grid median
# step is 0.003 / 0.006 / 0.003 m on the frames between anchors (p99 0.05 / 0.02 / 0.03, the amateur and broadcast ones in
# blends that run at 0.03-0.07 m a frame for up to two seconds), and every frame these rules spread there lies at an
# anchor or the frame after one (the Veo run has 5 more, isolated 0.03-0.06 m steps).
SPREAD_STEP = .03      # m: a registration step spread for the display is at least this (5-10 times the motion row's own
                       # error between anchors; on the amateur camera a 0.03 m grid step is 0.1-0.4 m at the far players) ...
SPREAD_ISOLATION = 3.0 # ... and this many times the median step of the other frames within +-SPREAD_LOCAL (a snap, not a
SPREAD_LOCAL = .3      # s     gradual change of the registration: those are spread over many frames already)
SPREAD_GAP = .2        # s: steps this close that are no keyframe's or trusted anchor's form one event (an automatic
                       # anchor's arriving and leaving steps, a frame jumping away and the next one back), or join a
                       # keyframe's or trusted anchor's leaving step on an automatic anchor's frame (step_events)
GLIDE_SPEED = .2      # m/s: a ramp is as long as the event's step at the median person needs at this speed, within
RAMP_MIN = 1.0        # s    these. Overlapping ramps add: on the amateur run two anchors 1 s apart (162.2 and 163.2 s)
RAMP_MAX = 6.0        # s    moved the median player by 0.54 and 0.63 m the same way, and its glide there peaked at .41
                      # m/s, the fastest player's (the far players' step is 1.6-1.8 times the median one, up to 3.6) at
                      # .88 (at GLIDE_SPEED .3: .62 and 1.32; .5: 1.03 and 2.19). RAMP_MIN: the short ramps of neighbouring
                      # small steps add too (at .5 s the fastest player glided at 1.02 m/s on the amateur run, 141.6 s).
                      # RAMP_MAX: a step of 1.2 m or more at the median person takes the longest ramp (Veo's 18.8 s
                      # keyframe: 2.36 m, with room for 5.5 s back to the 13.3 s one: .43 m/s; at 4 s .59)
RAMP_FLOOR = .15      # s: a ramp with less room than this (a step right after a registration start or a pinned frame) ...
SQUEEZE = 2.0         # ... or one its room makes glide the median person faster than this times the larger of
                      # GLIDE_SPEED and its full ramp's glide (the larger where RAMP_MAX caps the ramp: Veo's 18.8 s
                      # keyframe, 2.36 m with room for 5.5 s of its 6, glides at .43 m/s, under 2 x .39, and is not
                      # squeezed; a ramp at least half as long as its step asks for never is) is squeezed: spread the
                      # other way when that way has the room and moves no pinned frame, else not at all (the jump
                      # freedom of an unspread step). A keyframe's leaving step 0.2 s before a cut, spread into that
                      # room, glided at 4 m/s. Only an event of several steps that cancels (step_events) is spread its
                      # first way when neither way has the room
PINNED_SOURCES = ('landmarks',)   # the anchors the display honours exactly: the user's keyframes
TRUSTED_SOURCES = ('reregistered',)   # ... and those whose steps it spreads as a keyframe's when a keyframe roots them
                                      # (anchor_frames: matched to its view, directly or through a relay view rooted at
                                      # it), but other ramps cross them, a squeezed ramp of their own turns the other
                                      # way (moving their frame) before it snaps, and one whose steps cancel (an
                                      # outlier) is automatic (step_events)

# ---------- the observations ----------
def _slot_ids(result):
    """The identities the timeline carries (roster team slots, not retired or left), or without a roster every id of a
    team person."""
    slots = (result.get('roster') or {}).get('slots')
    if slots is not None:
        return [s['id'] for s in slots if s.get('status') not in ('RETIRED', 'LEFT') and s.get('team') in ('A', 'B')]
    return sorted({p['id'] for f in result.get('frames', []) for p in f.get('people', []) if p.get('id') and p.get('team') in ('A', 'B')})


def _frame_times(frames):
    """Per frame its time in seconds and whether it has one: a frame without a numeric, finite time gets its
    neighbours' (interpolated by frame index, 0 when no frame has one) and carries no observation."""
    T = len(frames)
    t = np.full(T, np.nan)
    for i, f in enumerate(frames):
        v = f.get('time')
        if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v):
            t[i] = float(v)
    timed = np.isfinite(t)
    if timed.any() and not timed.all():
        index = np.arange(T)
        t[~timed] = np.interp(index[~timed], index[timed], t[timed])
    return np.nan_to_num(t), timed


def _homographies(frames, width=None, height=None):
    """Per frame the final homography (world metres -> pixels) and its inverse; NaN where the frame has none. With the
    video's width and height each is oriented (field.orient: the ground in front of the camera at positive depth), as
    a result written before the stored homographies were oriented may hold flipped ones (which see no ground)."""
    T = len(frames)
    H = np.full((T, 3, 3), np.nan)
    G = np.full((T, 3, 3), np.nan)
    for i, f in enumerate(frames):
        h = (f.get('calibration') or {}).get('H')
        if not isinstance(h, list) or len(h) != 9:
            continue
        m = np.asarray(h, np.float64).reshape(3, 3)
        if not np.isfinite(m).all() or abs(np.linalg.det(m)) < 1e-15:
            continue
        if width and height:
            m = field.orient(m, width, height)
            if m is None:
                continue
        H[i], G[i] = m, np.linalg.inv(m)
    return H, G


def _covered(boxes):
    """Per box (normalized [x, y, w, h]) the largest share of it covered by another box that ends lower in the image:
    a person nearer the camera, in front of it."""
    b = np.asarray(boxes, np.float64).reshape(-1, 4)
    if len(b) < 2:
        return np.zeros(len(b))
    x1, y1, x2, y2 = b[:, 0], b[:, 1], b[:, 0]+b[:, 2], b[:, 1]+b[:, 3]
    ix = np.clip(np.minimum(x2[:, None], x2[None]) - np.maximum(x1[:, None], x1[None]), 0, None)
    iy = np.clip(np.minimum(y2[:, None], y2[None]) - np.maximum(y1[:, None], y1[None]), 0, None)
    share = ix*iy/np.maximum(b[:, 2]*b[:, 3], 1e-12)[:, None]
    share[~(y2[None] > y2[:, None])] = 0.0   # only a box ending lower covers this one from in front
    np.fill_diagonal(share, 0.0)
    return share.max(axis=1)


def observations(result, width, height, size=None, min_confidence=MIN_CONFIDENCE):
    """Every observation the stabilizer smooths. Per identity (keyed by the internal id) the frames the timeline marks
    O: imputation.estimate's rule (a calibrated frame, a pitch position, pitchConfidence of at least min_confidence or
    precise on a small scene; of two people with one id on a frame the later one); per anonymous track (team A or B,
    state 'uncertain', no id) the same. Returns {'keys': [('id', pid) | ('anonymous', track)], and arrays over the
    observations sorted by key, local track and frame: key, frame, track, time, fx, fy (foot pixel), top (pixel), h
    (box height, pixels), score, cover (the largest share of the box a nearer person covers), clipped, z (N, 2) the
    stored position in metres, sigma (the stored pitchSigma's larger value, NaN without one)}."""
    size = size or field.PitchSize()
    frames = result.get('frames') or []
    times, timed = _frame_times(frames)
    ids = _slot_ids(result)
    keys = [('id', pid) for pid in ids]
    index = {pid: k for k, pid in enumerate(ids)}
    anonymous = {}
    picked = {}
    for i, f in enumerate(frames):
        if not f.get('calibrated') or not timed[i]:
            continue
        people = f.get('people') or ()
        found = []
        for n, p in enumerate(people):
            pid = p.get('id')
            if pid is not None:
                k = index.get(pid)
                if k is None:
                    continue
            elif p.get('state') == 'uncertain' and p.get('team') in ('A', 'B') and p.get('track') is not None:
                k = anonymous.get(p['track'])
                if k is None:
                    k = anonymous[p['track']] = -1-len(anonymous)
            else:
                continue
            if not isinstance(p.get('pitch'), list) or not isinstance(p.get('box'), list):
                continue
            if (p.get('pitchConfidence') or 0.0) < min_confidence and not _precise(p, f):
                continue
            found.append((k, n))
        if not found:
            continue
        boxed = [n for n, q in enumerate(people) if isinstance(q.get('box'), list)]
        shares = dict(zip(boxed, _covered([people[n]['box'] for n in boxed]).tolist()))
        for k, n in found:
            picked[(k, i)] = (people[n], shares[n])
    # Anonymous tracks after the identities, in track order.
    order = sorted(anonymous, key=lambda t: (str(type(t)), t))
    remap = {anonymous[t]: len(keys)+n for n, t in enumerate(order)}
    keys += [('anonymous', t) for t in order]
    n = len(picked)
    people = [p for p, _ in picked.values()]
    key = np.array([remap.get(k, k) for k, _ in picked], np.int64)
    frame = np.array([i for _, i in picked], np.int64)
    tracks = [p.get('track') if p.get('track') is not None else -1 for p in people]
    codes = {t: c for c, t in enumerate(sorted(set(tracks), key=lambda t: (str(type(t)), t)))}
    order = np.lexsort((frame, np.array([codes[t] for t in tracks], np.int64), key)) if n else np.zeros(0, np.int64)
    box = np.array([p['box'] for p in people], np.float64).reshape(-1, 4)[order]
    pitch = np.array([p['pitch'][:2] for p in people], np.float64).reshape(-1, 2)[order]

    def larger(s):
        return max(s) if isinstance(s, list) and s and all(v is not None for v in s) else np.nan
    out = {'keys': keys, 'key': key[order], 'track': np.array(tracks, object)[order], 'frame': frame[order],
           'time': times[frame[order]] if n else np.zeros(0),
           'fx': (box[:, 0]+box[:, 2]/2)*width, 'fy': (box[:, 1]+box[:, 3])*height, 'top': box[:, 1]*height, 'h': box[:, 3]*height,
           'score': np.array([float(p.get('score') or 0.0) for p in people], np.float64)[order],
           'cover': np.array([c for _, c in picked.values()], np.float64)[order],
           'clipped': np.array([bool(clipped_below(b)) for b in box], bool),
           'z': pitch*[size.length, size.width], 'sigma': np.array([larger(p.get('pitchSigma')) for p in people], np.float64)[order]}
    return out


def _to_world(G, x, y):
    """Pixels -> world metres through inverse homographies G (N, 3, 3): (X, Y, den); den <= 0 behind the horizon."""
    den = G[:, 2, 0]*x+G[:, 2, 1]*y+G[:, 2, 2]
    safe = np.where(np.abs(den) > 1e-12, den, 1e-12)
    return (G[:, 0, 0]*x+G[:, 0, 1]*y+G[:, 0, 2])/safe, (G[:, 1, 0]*x+G[:, 1, 1]*y+G[:, 1, 2])/safe, den


def jacobians(G, x, y):
    """The 2x2 Jacobian of pixel -> pitch metres at each pixel (x, y) through its inverse homography G (N, 3, 3);
    NaN where the pixel sees no ground in front of the camera."""
    X, Y, den = _to_world(G, x, y)
    J = np.empty((len(x), 2, 2))
    J[:, 0, 0] = (G[:, 0, 0]-X*G[:, 2, 0])/den
    J[:, 0, 1] = (G[:, 0, 1]-X*G[:, 2, 1])/den
    J[:, 1, 0] = (G[:, 1, 0]-Y*G[:, 2, 0])/den
    J[:, 1, 1] = (G[:, 1, 1]-Y*G[:, 2, 1])/den
    J[~(den > 1e-12)] = np.nan
    return J


# ---------- the edge fits, the noise model and the box events ----------
def _noise(prior, h):
    a, b = prior
    return np.sqrt(a*a+(b*h)**2)


def _row_medians(a):
    """The median of each row of a (..., M) over its finite values (NaN where none): a sort instead of np.nanmedian,
    which is several times slower on many short rows."""
    a = np.sort(a, axis=-1)                        # NaN last
    count = np.isfinite(a).sum(axis=-1)
    lo = np.clip((count-1)//2, 0, None)[..., None]
    hi = np.clip(count//2, 0, a.shape[-1]-1)[..., None]
    med = .5*(np.take_along_axis(a, lo, -1)+np.take_along_axis(a, hi, -1))[..., 0]
    return np.where(count > 0, med, np.nan)


def _frame_step(run, t):
    """The time step the neighbour windows are sized by: the 5th percentile of the steps within the runs above a
    quarter of their median (a duplicated or re-stamped frame, a step of a millisecond, must not widen every window of
    the segment tenfold: a window then reaches a little less far around it, its time mask is unchanged). None without
    steps."""
    steps = np.diff(t)[run[1:] == run[:-1]]
    steps = steps[steps > 1e-6]
    if not len(steps):
        return None
    steps = steps[steps >= .25*np.median(steps)]
    return float(np.percentile(steps, 5))


def _windows(run, t, half, own, stride=1, step=None, rows=None):
    """Neighbour index matrix (len(rows), M) over observations sorted by run and time: for each of rows (default: all)
    the samples of the same run within +-half seconds (the sample itself only when own; every stride-th neighbour),
    with a validity mask. step: the time step the window's width is sized by (_frame_step of the whole set when
    None)."""
    N = len(t)
    rows = np.arange(N) if rows is None else rows
    if not N or not len(rows):
        return np.zeros((len(rows), 1), np.int64), np.zeros((len(rows), 1), bool)
    step = _frame_step(run, t) if step is None else step
    reach = int(min(400, math.ceil(half/step)+1)) if step else 1
    offsets = np.r_[-np.arange(stride, reach+1, stride)[::-1], 0, np.arange(stride, reach+1, stride)]
    if not own:
        offsets = offsets[offsets != 0]
    j = rows[:, None]+offsets[None]
    valid = (j >= 0) & (j < N)
    j = np.clip(j, 0, N-1)
    valid &= (run[j] == run[rows, None]) & (np.abs(t[j]-t[rows, None]) <= half+1e-9)
    return j, valid


def _runs_of(obs):
    """Per observation its local-track run (one key on one local track), numbered in the observations' order."""
    N = len(obs['frame'])
    run = np.zeros(N, np.int64)
    if N > 1:
        change = (obs['key'][1:] != obs['key'][:-1]) | np.array([a != b for a, b in zip(obs['track'][1:], obs['track'][:-1])], bool)
        run[1:] = np.cumsum(change)
    return run


def edge_fits(obs, H, G, footage=None):
    """Camera-compensated local fits of every observation's box edges. Per local-track run (one key on one local
    track) the neighbours within +-FIT_HALF are moved into the sample's frame (H_t H_s^-1) and a quadratic in time is
    fitted to the centre x, the bottom and the top (tricube in time, biweight on the prior noise scaled to the run's own
    level, leaving the sample out); the running median of the camera-compensated box heights within +-HEIGHT_HALF.
    Computed in blocks of CHUNK observations, so the memory stays bounded however long the segment. Returns arrays
    over the observations: residual (N, 3) of the centre x, bottom, top (pixels, NaN without a fit), g (N, 3) the fit's
    variance factor (residual / sqrt(1 + g) has the noise's sigma), hmed (N,) the running median height, long (N, 2) the
    bottom and the top against their running medians over +-HEIGHT_HALF (pixels), horizon (N,) the image row of the
    ground's vanishing line at the foot's column (H^-T e3; NaN where the frame sees no ground below it), and footage
    {'level', 'span', 'step'}: the footage's own noise level per channel, the fits' window span and the frame step the
    windows are sized by, each measured over every observation. footage: those to use instead of measuring them (a
    corrected run keeps its original's, as its noise model: measured again without an excluded player's observations,
    the level moved 36 samples of two other players on the amateur run by up to 0.08 m)."""
    N = len(obs['frame'])
    if not (isinstance(footage, dict) and {'level', 'span', 'step'} <= set(footage) and isinstance(footage['level'], list) and
            len(footage['level']) == 3 and _number(footage['span'])):
        footage = None                                      # an older noise model, or none: measured
    out = {'residual': np.full((N, 3), np.nan), 'g': np.full((N, 3), np.nan), 'hmed': np.full(N, np.nan), 'long': np.full((N, 2), np.nan),
           'horizon': np.full(N, np.nan), 'footage': copy.deepcopy(footage)}
    if not N:
        return out
    run = _runs_of(obs)
    t, f = obs['time'], obs['frame']
    Hf, Gf = H[f], G[f]
    one = np.ones(N)
    foot_w = (Gf @ np.c_[obs['fx'], obs['fy'], one][:, :, None])[:, :, 0]   # homogeneous ground points (no division:
    top_w = (Gf @ np.c_[obs['fx'], obs['top'], one][:, :, None])[:, :, 0]   # a top above the horizon maps back exactly)
    with np.errstate(divide='ignore', invalid='ignore'):
        out['horizon'] = np.where(Gf[:, 2, 1] > 1e-12, -(Gf[:, 2, 0]*obs['fx']+Gf[:, 2, 2])/Gf[:, 2, 1], np.nan)
    if footage:
        step, span = footage['step'], footage['span']
    else:
        step = _frame_step(run, t)
        steps = np.diff(t)[(run[1:] == run[:-1])]
        steps = steps[steps > 1e-6]
        span = FIT_HALF+(float(np.median(steps)) if len(steps) else 0.0)   # the window's last neighbours keep some weight
    own_all = np.stack([obs['fx'], obs['fy'], obs['top']], axis=1)           # (N, 3)
    sigma_all = np.stack([_noise(PRIOR_CENTRE, obs['h']), _noise(PRIOR_BOTTOM, obs['h']), _noise(PRIOR_BOTTOM, obs['h'])], axis=1)
    ridge = np.eye(3)*1e-9

    def moved(rows, j):
        pf = Hf[rows] @ np.transpose(foot_w[j], (0, 2, 1))
        pt = Hf[rows] @ np.transpose(top_w[j], (0, 2, 1))
        with np.errstate(divide='ignore', invalid='ignore'):
            return pf[:, 0]/pf[:, 2], pf[:, 1]/pf[:, 2], pt[:, 1]/pt[:, 2]

    def medians(rows):
        # The running median of the box height, each neighbour's height measured in this frame's pixels; and the
        # bottom and the top against their own running medians there (the long scale: an event longer than half the
        # local fit's window is followed by the fit, not by a median over +-HEIGHT_HALF).
        j, valid = _windows(run, t, HEIGHT_HALF, True, HEIGHT_STRIDE, step, rows)
        _, bottom, top = moved(rows, j)
        valid &= np.isfinite(bottom) & np.isfinite(top)
        enough = valid.sum(axis=1) >= HEIGHT_MIN
        med = _row_medians(np.where(valid[:, None], np.stack([bottom-top, bottom, top], axis=1), np.nan))
        out['hmed'][rows] = np.where(enough, med[:, 0], np.nan)
        out['long'][rows] = np.where(enough[:, None], own_all[rows, 1:]-med[:, 1:], np.nan)

    def local_fit(rows, level):
        # Leaving the sample out: per sample a weighted quadratic in time for the centre x, the bottom and the top at
        # once (the channels' normal equations as one batched product). Without a level: the first (wide) fit's
        # residuals over the prior noise, per channel, which give the footage's own noise level.
        n = len(rows)
        j, valid = _windows(run, t, FIT_HALF, False, 1, step, rows)
        cx, bottom, top = moved(rows, j)
        values = np.stack([cx, bottom, top], axis=1)                          # (n, 3, M)
        valid &= np.isfinite(values).all(axis=1)
        tau = t[j]-t[rows, None]
        base = np.where(valid, np.clip(1-np.abs(tau/span)**3, 0, None)**3, 0.0)
        X = np.stack([np.ones_like(tau), tau, tau*tau], axis=-1)              # (n, M, 3)
        XX = (X[..., :, None]*X[..., None, :]).reshape(n, -1, 9)
        values = np.where(valid[:, None], values, 0.0)
        sigma = sigma_all[rows]
        scale = BIWEIGHT*np.maximum(sigma, 1.0)[:, :, None]                   # (n, 3, 1)
        weights = np.repeat(base[:, None], 3, axis=1)

        def solve(weights):
            A = (weights @ XX).reshape(n, 3, 3, 3)+ridge
            b = (weights*values) @ X
            return A, np.linalg.solve(A, b[..., None])[..., 0]
        for widen in (4.0, 2.0, 1.0):
            _, beta = solve(weights)
            r = values-beta @ np.transpose(X, (0, 2, 1))
            if widen == 4.0:
                if level is None:
                    return [np.abs(r[:, c][valid])/sigma[:, c, None].repeat(r.shape[2], 1)[valid] for c in range(3)]
                scale = scale*np.clip(1.4826*np.nan_to_num(level, nan=1.0), 1.0, 3.0)[None, :, None]
            u = r/(widen*scale)
            weights = base[:, None]*np.where(np.abs(u) < 1, (1-u*u)**2, 0.0)
        A, beta = solve(weights)
        Ainv = np.linalg.inv(A)
        B2 = ((weights*weights) @ XX).reshape(n, 3, 3, 3)
        g = (Ainv @ B2 @ Ainv)[..., 0, 0]
        used = weights[:, 1] > 0
        sides = ((tau < 0) & used).sum(axis=1) >= FIT_SIDE
        sides &= ((tau > 0) & used).sum(axis=1) >= FIT_SIDE
        out['residual'][rows] = np.where(sides[:, None], own_all[rows]-beta[..., 0], np.nan)
        out['g'][rows] = np.where(sides[:, None], g, np.nan)

    blocks = [np.arange(a, min(N, a+CHUNK)) for a in range(0, N, CHUNK)]
    for rows in blocks:
        medians(rows)
    # The footage's own noise level against the prior's, per channel, from the first (wide) fit's residuals (a noisier
    # camera would otherwise lose good neighbours to a biweight cut at the prior's noise): over every observation up to
    # LEVEL_ROWS of them, over an even subset beyond.
    if footage:
        level = np.array([np.nan if v is None else v for v in footage['level']], np.float64)
    else:
        subset = np.arange(0, N, max(1, math.ceil(N/LEVEL_ROWS)))
        parts = [local_fit(subset[a:a+CHUNK], None) for a in range(0, len(subset), CHUNK)]
        level = np.array([np.median(np.concatenate([p[c] for p in parts])) if sum(len(p[c]) for p in parts) else np.nan for c in range(3)])
        out['footage'] = {'level': [float(v) if np.isfinite(v) else None for v in level], 'span': float(span),
                          'step': None if step is None else float(step)}
    for rows in blocks:
        local_fit(rows, level)
    return out


def _key_medians(obs, values, use, half, least, step):
    """Per observation the median of values over its key's `use` samples within +-half seconds, evaluated on a grid of
    `step` seconds over the key's span (each observation takes its nearest grid point); where fewer than `least`
    samples are that near, the key's median over all its `use` samples (with `least` of them), else NaN."""
    out = np.full(len(values), np.nan)
    if not len(values):
        return out
    key, t = obs['key'], obs['time']
    order = np.lexsort((t, key))
    ks = key[order]
    bounds = np.flatnonzero(np.r_[True, ks[1:] != ks[:-1], True])
    for a, b in zip(bounds[:-1], bounds[1:]):
        idx = order[a:b]
        tt, m = t[idx], use[idx]
        tc, vc = tt[m], values[idx][m]
        if len(tc) < least:
            continue
        grid = np.arange(tt[0], tt[-1]+step, step)
        lo, hi = np.searchsorted(tc, grid-half, 'left'), np.searchsorted(tc, grid+half, 'right')
        j = lo[:, None]+np.arange(max(1, int((hi-lo).max())))[None]
        med = _row_medians(np.where(j < hi[:, None], vc[np.clip(j, 0, len(vc)-1)], np.nan))
        med = np.where(hi-lo >= least, med, float(np.median(vc)))
        out[idx] = med[np.clip(np.rint((tt-tt[0])/step).astype(np.int64), 0, len(grid)-1)]
    return out


def geometric_heights(obs, fits, exclude, run_ratio=None):
    """The box height an upright player of the observation's identity has where the box's top is: for a pinhole camera
    the box height is the foot's distance below the horizon (fits['horizon']) times r = the player's height / the
    camera's height, so from the top alone h_geo = r (top - horizon) / (1 - r). r is the identity's running median of
    h / (foot - horizon) over its clean samples (not clipped, no nearer person overlapping, a confident detection, not
    in exclude) within +-RATIO_HALF, else its median over all of them, else the run's (run_ratio when given). A box
    that keeps losing the legs for seconds moves a running median of the height itself (it covered over half of the
    +-HEIGHT_HALF window on Veo A-10 at 7-9.5 s, and the full boxes were then taken for boxes reaching into a nearer
    player); it moves r only when it covers half of the +-RATIO_HALF window. Returns (h_geo (N,) pixels, r (N,), the
    run's median ratio or None), NaN where the frame sees no ground below the box, r is outside RATIO_RANGE or the top
    is above the horizon."""
    below = obs['fy']-fits['horizon']
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = obs['h']/below
        clean = (np.isfinite(ratio) & (below > 1) & (ratio > RATIO_RANGE[0]) & (ratio < RATIO_RANGE[1]) & ~obs['clipped'] &
                 (obs['cover'] <= 0) & (obs['score'] >= LOW_SCORE) & ~exclude)
    r = _key_medians(obs, ratio, clean, RATIO_HALF, RATIO_MIN, RATIO_STEP)
    if run_ratio is None and clean.sum() >= RATIO_MIN:
        run_ratio = round(float(np.median(ratio[clean])), 4)   # as the noise model keeps it
    if run_ratio is not None:
        r = np.where(np.isfinite(r), r, run_ratio)
    above = obs['top']-fits['horizon']
    with np.errstate(invalid='ignore'):
        ok = np.isfinite(r) & (r > RATIO_RANGE[0]) & (r < RATIO_RANGE[1]) & (above > 0)
        return np.where(ok, r*above/(1-np.where(ok, r, 0.0)), np.nan), np.where(ok, r, np.nan), run_ratio


def _fit_level(h, values, clean, prior, samples, a_range, b_range):
    """sigma^2 = a^2 + (b h)^2 for values (N,) with noise of that size: per box-height bin (HEIGHT_BINS, BIN_MIN clean
    samples a bin) the robust sigma, fitted by weighted least squares (with a single bin, the prior scaled to it),
    blended with the prior by sample count (samples) and kept in a_range, b_range. Returns (a, b, fitted samples)."""
    edges = np.asarray(HEIGHT_BINS, np.float64)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = clean & (h >= lo) & (h < hi)
        if m.sum() >= BIN_MIN:
            rows.append((float(np.median(h[m])), _rstd(values[m]), int(m.sum())))
    a, b, n = prior[0], prior[1], 0
    if rows:
        rows = np.array(rows)
        if len(rows) >= 2:
            A = np.c_[np.ones(len(rows)), rows[:, 0]**2]*np.sqrt(rows[:, 2])[:, None]
            coef, *_ = np.linalg.lstsq(A, rows[:, 1]**2*np.sqrt(rows[:, 2]), rcond=None)
            fit_a, fit_b = math.sqrt(max(coef[0], 0.0)), math.sqrt(max(coef[1], 0.0))
        else:
            # One box size only: the level is measured, the split between a and b is not (the prior's).
            ratio = rows[0, 1]/float(_noise(prior, rows[0, 0]))
            fit_a, fit_b = prior[0]*ratio, prior[1]*ratio
        n = int(rows[:, 2].sum())
        u = n/(n+samples)
        a, b = (1-u)*prior[0]+u*fit_a, (1-u)*prior[1]+u*fit_b
    return float(np.clip(a, *a_range)), float(np.clip(b, *b_range)), n


def _short_spells(mask, run, t, longest):
    """mask (N,) over observations sorted by run and time, kept only on its spells (consecutive True samples of one run)
    that last at most `longest` seconds."""
    if not mask.any():
        return mask
    starts = mask & np.r_[True, ~mask[:-1] | (run[1:] != run[:-1])]
    label = np.cumsum(starts)-1
    idx = np.flatnonzero(mask)
    first = np.full(label[-1]+1 if len(label) else 0, np.inf)
    last = np.full(len(first), -np.inf)
    np.minimum.at(first, label[idx], t[idx])
    np.maximum.at(last, label[idx], t[idx])
    keep = mask.copy()
    keep[idx] = (last-first)[label[idx]] <= longest+1e-9
    return keep


def _event_rule(obs, fits, href):
    """The two-scale event rule against the height reference href: (event, limit, taller)."""
    limit = np.maximum(EVENT_PX, EVENT_SHARE*np.nan_to_num(href))
    judged = np.isfinite(href) & ~obs['clipped']
    taller = obs['h']-np.nan_to_num(href)
    event = np.zeros(len(href), bool)
    for rb, rt in ((fits['residual'][:, 1], fits['residual'][:, 2]), (fits['long'][:, 0], fits['long'][:, 1])):
        with np.errstate(invalid='ignore'):
            event |= (judged & np.isfinite(rb) & np.isfinite(rt) & (np.abs(rb) > limit) &
                      (np.abs(rt) < np.minimum(EVENT_TOP*np.abs(rb), limit)) & (np.sign(rb)*taller > HEIGHT_AGREE*limit))
    return event, limit, np.where(judged, taller, 0.0)


def box_events(obs, fits, model=None):
    """The box-extent events: the bottom edge off its local fit, or off its running median over +-HEIGHT_HALF (an
    event longer than half the fit's window, which the fit follows), by more than limit = max(EVENT_PX, EVENT_SHARE
    h_ref) while the top edge moves less than EVENT_TOP of that and less than the limit on the same scale (a box that
    jumps to a nearer person moves its top too: an identity switch is no event), and the box shorter than h_ref by
    HEIGHT_AGREE of the limit when the bottom went up (it lost the legs), longer when it went down (it reaches into a
    nearer player). A box cut by the frame's bottom is not judged. The foot is repaired as top + h_ref.

    h_ref is the running median height (fits['hmed']) where it agrees with the geometric height from the top
    (geometric_heights) within GEO_AGREE sigmas of the geometric height's own noise, the geometric height where it does
    not (a long event drags the running median with it), the running median where the frame has no geometry. The
    geometric height's noise sqrt(a^2 + (b h)^2) is fitted per run on the clean samples like the pixel noise
    (GEO_PRIOR; 1.5-2.2 px + 3.5-5% of h on the amateur and broadcast runs, 3.3-7.4 px on the Veo run, worse for boxes
    under 20 px). A box off its geometric height by more than the limit and DOUBT_SIGMA of that noise while its top is
    steady, that is no event (a long event both scales follow, or a change of posture: which edge moved is not known),
    is doubtful: not repaired, its bottom's sigma at least the difference; not when it is taller and one of a spell of
    taller boxes lasting over DOUBT_SPELL (the ratio is wrong there). Two passes: r and the noise again without the
    first pass's events and doubtful boxes. model: {'ratio', 'a', 'b', 'firstPass': the first pass's {'ratio', 'a',
    'b'}} the run's median ratio and geometric noise to use instead of fitting them (a corrected run keeps its
    original's), both passes replayed (a model without 'firstPass' is used in both). Returns {'event' (N,), 'repaired'
    (N,) foot y pixels (NaN where no event), 'href' (N,), 'geometric' (N,) h_ref is the geometric height, 'doubt' (N,)
    pixels (0 where not doubtful), 'ratio' (N,) r, 'sgeo' (N,) the geometric height's sigma (NaN without one), 'model'}.

    Measured on the amateur, Veo and broadcast runs against the trajectory smoothed with every candidate sample left
    out (bridged from the samples around it), with the running median as h_ref: the local scale's repairs come closer
    to it in 93-97% of its samples (raw 1.1-3.2 m off, repaired 0.2-0.45 m), the long scale's in 66-96%. The height's
    direction drops the samples next to an event, where the local fit still leans towards it; the top's own limit
    drops 15, 2 and 0 samples. Against the 99 hand-marked feet (tools/check_foot_truth.py) the geometric reference
    keeps the full boxes of a long event from being repaired to the cut ones' foot (Veo A-10 at 7.7 and 8.5 s: 5.8-5.9
    m off with the running median alone)."""
    hmed = fits['hmed']
    exclude = np.zeros(len(hmed), bool)
    fixed = model is not None
    run = _runs_of(obs)
    first = None
    for n_pass in range(2):
        # A given model replays both passes: its first pass's model ('firstPass') decides the first pass's events and
        # doubtful boxes, which the second pass leaves out of each identity's ratio, as the fit did (with the final
        # model in both passes, handing a run's own model back moved 1072 Veo samples by up to 0.25 m).
        use = (model.get('firstPass') or model) if fixed and n_pass == 0 else model
        hgeo, r, run_ratio = geometric_heights(obs, fits, exclude, use.get('ratio') if fixed else None)
        geometric = np.isfinite(hgeo)
        if not fixed:
            clean = geometric & ~exclude & ~obs['clipped'] & (obs['cover'] <= 0) & (obs['score'] >= LOW_SCORE)
            a, b, n = _fit_level(hgeo, obs['h']-np.nan_to_num(hgeo), clean, GEO_PRIOR, PRIOR_SAMPLES, GEO_A_RANGE, GEO_B_RANGE)
            use = model = {'ratio': None if run_ratio is None else round(run_ratio, 4), 'a': round(a, 3), 'b': round(b, 4), 'fittedSamples': n}
            if n_pass == 0:
                first = {key: model[key] for key in ('ratio', 'a', 'b')}
            else:
                model['firstPass'] = first
        sgeo = np.where(geometric, _noise((use['a'], use['b']), np.nan_to_num(hgeo)), np.nan)
        with np.errstate(invalid='ignore'):
            agree = np.abs(hmed-hgeo) <= GEO_AGREE*sgeo
        href = np.where(geometric & ~agree, hgeo, np.where(np.isfinite(hmed), hmed, hgeo))
        event, limit, taller = _event_rule(obs, fits, href)
        off = np.where(geometric, obs['h']-np.nan_to_num(hgeo), 0.0)
        with np.errstate(invalid='ignore'):
            steady = (np.abs(fits['residual'][:, 2]) < limit) | (np.abs(fits['long'][:, 1]) < limit)
            beyond = np.maximum(limit, DOUBT_SIGMA*np.nan_to_num(sgeo))
            doubtful = geometric & ~event & ~obs['clipped'] & steady & (np.abs(off) > beyond)
            # Taller than the geometric height for longer than DOUBT_SPELL (half the threshold is enough to carry the
            # spell on): the identity's ratio is wrong there, the boxes are not.
            tall = geometric & ~obs['clipped'] & (off > .5*beyond)
        doubtful &= ~(tall & ~_short_spells(tall, run, obs['time'], DOUBT_SPELL))
        exclude = event | doubtful
    return {'event': event, 'repaired': np.where(event, obs['top']+np.nan_to_num(href), np.nan), 'href': href,
            'geometric': geometric & ~agree, 'doubt': np.where(doubtful, np.abs(off), 0.0), 'ratio': r, 'sgeo': sgeo, 'model': model}


def _rstd(v):
    v = v[np.isfinite(v)]
    return float(1.4826*np.median(np.abs(v-np.median(v)))) if len(v) >= 5 else math.nan


def fit_noise(obs, fits, event):
    """The run's pixel noise model from the leave-one-out residuals of its clean samples (a fit, no overlap with
    another box, not clipped, score >= LOW_SCORE, no event): per box-height bin the robust sigma of residual /
    sqrt(1 + g), sigma^2 = a^2 + (b h)^2 fitted by weighted least squares (with a single bin, the prior scaled to it),
    blended with the prior by sample count (PRIOR_SAMPLES) and kept in A_RANGE, B_RANGE; the inflation factors (OCCLUDED cover, LOW_SCORE) as the bins'
    count-weighted ratio of robust sigmas, blended with their priors in the log (FACTOR_SAMPLES)."""
    norm = fits['residual']/np.sqrt(1+fits['g'])
    h = obs['h']
    fitted = np.isfinite(norm).all(axis=1) & ~event & ~obs['clipped']
    clean = fitted & (obs['cover'] <= 0) & (obs['score'] >= LOW_SCORE)
    edges = np.asarray(HEIGHT_BINS, np.float64)
    model = {'samples': int(clean.sum())}
    for name, column, prior in (('bottom', 1, PRIOR_BOTTOM), ('centre', 0, PRIOR_CENTRE)):
        a, b, n = _fit_level(h, norm[:, column], clean, prior, PRIOR_SAMPLES, A_RANGE, B_RANGE)
        model[name] = {'a': round(a, 3), 'b': round(b, 4), 'fittedSamples': n, 'prior': {'a': prior[0], 'b': prior[1]}}

    def factor(condition, prior):
        num = den = 0.0
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = fitted & condition & (h >= lo) & (h < hi)
            ref = clean & (h >= lo) & (h < hi)
            if m.sum() >= 30 and ref.sum() >= 30:
                num += m.sum()*_rstd(norm[m, 1])/max(_rstd(norm[ref, 1]), 1e-6)
                den += m.sum()
        measured = num/den if den else prior
        u = den/(den+FACTOR_SAMPLES)
        return round(float(np.clip(math.exp((1-u)*math.log(prior)+u*math.log(max(measured, 1e-3))), *FACTOR_RANGE)), 2), int(den)
    occluded, n_occluded = factor(obs['cover'] > OCCLUDED, PRIOR_OCCLUDED)
    low, n_low = factor(obs['score'] < LOW_SCORE, PRIOR_LOW_SCORE)
    model['inflation'] = {'occluderInFront': occluded, 'occluderSamples': n_occluded, 'lowScore': low, 'lowScoreSamples': n_low}
    return model


def measurement_covariance(obs, H, G, noise, boxes):
    """Per observation the position the smoother uses (metres; a repaired foot moves the stored position by the
    repair's own displacement) and its covariance R = J diag(sx^2, sy^2) J^T + REGISTRATION_FLOOR^2 I, with the
    run's noise model inflated for a nearer person in front or a low score, CLIPPED_FOOT of the height for a box cut by
    the frame, the repair's sigma for a repaired foot (boxes: box_events), at least the height's difference for a
    doubtful box. The repairs of one event share the reference's error: a repair with n repairs of its run within
    +-HEIGHT_HALF has sqrt(max(REPAIR_PX, REPAIR_SHARE h_ref)^2 + (n - 1) (REPAIR_COMMON h_ref)^2), so a long event's
    repairs together claim no more than that shared error. Where the pixel has no ground Jacobian, the stored sigma
    (or imputation.DEFAULT_SIGMA) in both axes. Returns (z (N, 2), R (N, 2, 2), repaired (N,), good (N,): the samples
    whose residuals may raise the process noise)."""
    N = len(obs['frame'])
    h = obs['h']
    event, repaired_y, doubt = boxes['event'], boxes['repaired'], boxes['doubt']
    inflation = np.ones(N)
    inflation = np.where(obs['cover'] > OCCLUDED, noise['inflation']['occluderInFront'], inflation)
    inflation = np.maximum(inflation, np.where(obs['score'] < LOW_SCORE, noise['inflation']['lowScore'], 1.0))
    sx = _noise((noise['centre']['a'], noise['centre']['b']), h)*inflation
    sy = _noise((noise['bottom']['a'], noise['bottom']['b']), h)*inflation
    sy = np.where(obs['clipped'], np.maximum(sy, CLIPPED_FOOT*h), sy)
    sy = np.maximum(sy, doubt)
    Gf = G[obs['frame']]
    z = obs['z'].copy()
    repaired = np.zeros(N, bool)
    foot_y = obs['fy'].copy()
    if event.any():
        X0, Y0, d0 = _to_world(Gf[event], obs['fx'][event], obs['fy'][event])
        X1, Y1, d1 = _to_world(Gf[event], obs['fx'][event], repaired_y[event])
        ok = (d0 > 1e-12) & (d1 > 1e-12) & np.isfinite(X1) & np.isfinite(Y1)
        where = np.flatnonzero(event)[ok]
        z[where] += np.c_[X1-X0, Y1-Y0][ok]
        foot_y[where] = repaired_y[where]
        repaired[where] = True
        run = _runs_of(obs)
        j, valid = _windows(run, obs['time'], HEIGHT_HALF, True, step=(noise.get('footage') or {}).get('step'), rows=where)
        n = (valid & repaired[j]).sum(axis=1)
        href = np.nan_to_num(repaired_y[where])-obs['top'][where]
        sy[where] = np.sqrt(np.maximum(REPAIR_PX, REPAIR_SHARE*href)**2+np.maximum(n-1, 0)*(REPAIR_COMMON*href)**2)
    J = jacobians(Gf, obs['fx'], foot_y)
    R = J @ (np.stack([sx*sx, np.zeros(N), np.zeros(N), sy*sy], axis=-1).reshape(N, 2, 2)) @ np.transpose(J, (0, 2, 1))
    R += np.eye(2)*REGISTRATION_FLOOR**2
    bad = ~np.isfinite(R).all(axis=(1, 2))
    fallback = np.where(np.isfinite(obs['sigma']), obs['sigma'], DEFAULT_SIGMA)
    R[bad] = np.eye(2)[None]*(fallback[bad]**2+REGISTRATION_FLOOR**2)[:, None, None]
    good = ~event & ~(doubt > 0) & ~bad & ~obs['clipped'] & (obs['cover'] <= OCCLUDED) & (obs['score'] >= LOW_SCORE)
    return z, R, repaired, good


# ---------- the smoother ----------
def _steps(times, jumps=None):
    """Per step i the transition from step i-1 (F (T, 4, 4), constant velocity over the actual time step), the white-
    noise-acceleration process covariance for a unit spectral density (Q (T, 4, 4), per axis [[dt^3/3, dt^2/2],
    [dt^2/2, dt]]; step 0 has dt = 0) and the extra position variance of a step where every position may jump
    (jumps (T,) m^2, zero by default)."""
    dt = np.r_[0.0, np.diff(times)]
    T = len(dt)
    F = np.tile(np.eye(4), (T, 1, 1))
    F[:, 0, 2] = F[:, 1, 3] = dt
    Q = np.zeros((T, 4, 4))
    Q[:, 0, 0] = Q[:, 1, 1] = dt**3/3
    Q[:, 0, 2] = Q[:, 2, 0] = Q[:, 1, 3] = Q[:, 3, 1] = dt**2/2
    Q[:, 2, 2] = Q[:, 3, 3] = dt
    jumps = np.zeros(T) if jumps is None else np.nan_to_num(np.asarray(jumps, np.float64))
    return F, Q, jumps


def _inv2(S):
    det = S[..., 0, 0]*S[..., 1, 1]-S[..., 0, 1]*S[..., 1, 0]
    out = np.empty_like(S)
    out[..., 0, 0], out[..., 1, 1] = S[..., 1, 1]/det, S[..., 0, 0]/det
    out[..., 0, 1], out[..., 1, 0] = -S[..., 0, 1]/det, -S[..., 1, 0]/det
    return out


def _rts(steps, z, Reff, update, q, seg, start):
    """One Kalman filter and Rauch-Tung-Striebel pass over every column at once. steps: _steps(times); z (T, K, 2);
    Reff (T, K, 2, 2) the covariance each observation is used with; update (T, K) the observations used; q (T, K) the
    process noise of the step into each frame; seg (T, K) the segment id (-1 outside a segment), start (T, K) its
    first step. Every column is carried on every step (a column outside its segments carries values nobody reads,
    which is cheaper than selecting the live ones; its covariance only grows, finitely, until the next start). Returns (xs (T, K, 4), Ps (T, K, 4, 4)), NaN outside the
    segments."""
    F, Qu, jumps = steps
    T, K = seg.shape
    x = np.zeros((K, 4))
    P = np.tile(np.eye(4), (K, 1, 1))
    xf = np.empty((T, K, 4))
    Pf = np.empty((T, K, 4, 4))
    xp = np.empty((T, K, 4))
    Pp = np.empty((T, K, 4, 4))
    prior = np.diag([DIFFUSE**2, DIFFUSE**2, V0**2, V0**2])
    for i in range(T):
        x = x @ F[i].T
        P = F[i] @ P @ F[i].T+q[i, :, None, None]*Qu[i]
        if jumps[i] > 0:
            P[:, 0, 0] += jumps[i]
            P[:, 1, 1] += jumps[i]
        born = start[i]
        if born.any():
            x[born] = 0.0
            x[born, :2] = z[i, born]
            P[born] = prior
        xp[i], Pp[i] = x, P
        u = update[i]
        if u.any():
            gain = P[:, :, :2] @ _inv2(P[:, :2, :2]+Reff[i])
            gain[~u] = 0.0
            x = x+(gain @ (z[i]-x[:, :2])[:, :, None])[:, :, 0]
            P = P-gain @ P[:, :2, :]
            P = .5*(P+np.transpose(P, (0, 2, 1)))
        xf[i], Pf[i] = x, P
    xs, Ps = xf.copy(), Pf.copy()
    link = (seg[:-1] >= 0) & (seg[1:] == seg[:-1])
    for i in range(T-2, -1, -1):
        if not link[i].any():
            continue
        C = np.transpose(np.linalg.solve(Pp[i+1], F[i+1] @ Pf[i]), (0, 2, 1))
        C[~link[i]] = 0.0
        xs[i] = xf[i]+(C @ (xs[i+1]-xp[i+1])[:, :, None])[:, :, 0]
        Ps[i] = Pf[i]+C @ (Ps[i+1]-Pp[i+1]) @ np.transpose(C, (0, 2, 1))
    inside = seg >= 0
    return np.where(inside[..., None], xs, np.nan), np.where(inside[..., None, None], Ps, np.nan)


def _segments(times, observed, starts, gap):
    """Segment ids (T, K): a segment runs from an observation to the last observation before the next forced start or
    the next gap longer than `gap`; -1 outside. Ids are unique over all columns. Returns (seg, start)."""
    T, K = observed.shape
    seg = np.full((T, K), -1, np.int64)
    start = np.zeros((T, K), bool)
    sid = 0
    for k in range(K):
        idx = np.flatnonzero(observed[:, k])
        if not len(idx):
            continue
        new = np.ones(len(idx), bool)
        new[1:] = (np.diff(times[idx]) > gap) | starts[idx[1:], k]
        label = np.cumsum(new)-1+sid
        sid = label[-1]+1
        first = np.flatnonzero(new)
        last = np.r_[first[1:]-1, len(idx)-1]
        for a, b, s in zip(idx[first], idx[last], label[first]):
            seg[a:b+1, k] = s
            start[a, k] = True
    return seg, start


def _line_end(t, z, var, t0):
    """A robust straight-line fit of positions z (n, 2) with per-axis variances var (n, 2) at times t, evaluated at t0:
    (position (2,), velocity (2,), the position's variance (2,)). Each axis on its own: three passes with Huber
    weights at 2 sigma, then two with Tukey's biweight (BIWEIGHT sigmas), which drops a short excursion at the window's
    end instead of bending the line towards it."""
    X = np.c_[np.ones_like(t), t-t0]
    w = 1/var
    for n in range(5):
        coef, out_var = np.empty((2, 2)), np.empty(2)
        res = np.empty_like(z)
        for a in range(2):
            A = X.T @ (X*w[:, a, None])
            coef[a] = np.linalg.solve(A, X.T @ (w[:, a]*z[:, a]))
            out_var[a] = np.linalg.inv(A)[0, 0]
            res[:, a] = (z[:, a]-X @ coef[a])/np.sqrt(var[:, a])
        if n < 2:
            w = np.where(np.abs(res) <= 2, 1, 2/np.maximum(np.abs(res), 1e-9))/var
        elif n < 4:
            u = res/BIWEIGHT
            biweight = np.where(np.abs(u) < 1, (1-u*u)**2, 0.0)
            # A fit must keep two points per axis (else the last Huber weights stand).
            w = np.where((biweight > 0).sum(axis=0) >= 2, biweight, np.where(np.abs(res) <= 2, 1, 2/np.maximum(np.abs(res), 1e-9)))/var
    return coef[:, 0], coef[:, 1], out_var


def _switches(times, z, R, good, seg, options, observed=None):
    """Unexplained jumps inside the segments, shown by the good-quality observations alone (a repaired or occluded
    box is no evidence of a switch: Veo A-10 from 7.0 s, a box cut by a nearer player for more than a second, would
    otherwise read as one): consecutive good observations further apart than SWITCH_MIN + SWITCH_SPEED times the time
    between them, confirmed by robust line fits over SWITCH_WINDOW on each side, whose jump also exceeds that plus
    SWITCH_NOISE sigmas of the two observations along it (a far player's box moving by 3 px is metres in depth: no
    switch) and SWITCH_SIGMA sigmas of the line ends, still there SWITCH_PERSIST later (the observations then are
    nearer the line after the jump than the line before it, both carried on at their own velocity; where the segment
    ends sooner, every good observation after the jump, at least SWITCH_SAMPLES of them, is), and not across a
    calibration step (options['calibrationSteps']). The new segment starts at the first observation of any quality
    (observed (T, K), default: good) from which on every observation before the jump's first good one is nearer the
    line after the jump than the line before it (a repaired or occluded box of the new player stays with him). Of the
    jumps confirmed within SWITCH_WINDOW of each other, the largest. Returns [(column, step where the new segment
    starts, jump in metres)]."""
    speed, least, window, persist = options['switchSpeed'], options['switchMin'], options['switchWindow'], options['switchPersist']
    noise = options.get('switchNoise', SWITCH_NOISE)
    steps = options.get('calibrationSteps')
    stepped = np.r_[0, np.cumsum(np.asarray(steps, bool))] if steps is not None else None
    observed = good if observed is None else observed
    found = []
    T, K = good.shape
    for k in range(K):
        idx = np.flatnonzero(good[:, k])
        if len(idx) < 2*SWITCH_SAMPLES:
            continue
        zz = z[idx, k]
        tt = times[idx]
        ss = seg[idx, k]
        step = np.hypot(*(zz[1:]-zz[:-1]).T)
        same = ss[1:] == ss[:-1]
        var = np.c_[R[idx, k, 0, 0], R[idx, k, 1, 1]]
        seen = np.flatnonzero(observed[:, k])
        confirmed = []
        for c in np.flatnonzero(same & (step > least+speed*np.diff(tt))):
            a, b = idx[c], idx[c+1]
            if stepped is not None and stepped[b+1]-stepped[a+1] > 0:
                continue
            mine = ss == ss[c]
            pre = (tt >= tt[c]-window) & (np.arange(len(idx)) <= c) & mine
            post = (tt <= tt[c+1]+window) & (np.arange(len(idx)) > c) & mine
            if min(pre.sum(), post.sum()) < SWITCH_SAMPLES:
                continue
            p0, v0, s0 = _line_end(tt[pre], zz[pre], var[pre], tt[c])
            p1, v1, s1 = _line_end(tt[post], zz[post], var[post], tt[c+1])
            jump = float(np.hypot(*(p1-p0)))
            u = (p1-p0)/max(jump, 1e-9)
            spread = math.sqrt(max(float(u @ R[a, k] @ u+u @ R[b, k] @ u), 0.0))
            if jump <= least+speed*(tt[c+1]-tt[c])+noise*spread or jump*jump <= SWITCH_SIGMA**2*float((s0+s1).sum()):
                continue

            def nearer_after(t_at, z_at):
                after = np.hypot(*(z_at-p1-np.outer(t_at-tt[c+1], v1)).T)
                before = np.hypot(*(z_at-p0-np.outer(t_at-tt[c], v0)).T)
                return after < before
            if tt[mine].max() >= tt[c+1]+persist:
                late = (tt >= tt[c+1]+persist-window) & (tt <= tt[c+1]+persist) & mine
                if late.sum() < SWITCH_SAMPLES or not nearer_after(np.array([np.median(tt[late])]), np.median(zz[late], axis=0)[None])[0]:
                    continue   # it came back: an excursion
            else:
                # The segment ends before SWITCH_PERSIST: its end is the persistence, if all the rest stays after the jump.
                rest = (np.arange(len(idx)) > c) & mine
                if rest.sum() < SWITCH_SAMPLES or not nearer_after(tt[rest], zz[rest]).all():
                    continue
            between = seen[(seen > a) & (seen < b) & (seg[seen, k] == ss[c])]
            start = b
            if len(between):
                after = nearer_after(times[between], z[between, k])
                for n in range(len(between)-1, -1, -1):
                    if not after[n]:
                        break
                    start = int(between[n])
            confirmed.append((int(start), jump))
        group = []
        for b, jump in confirmed+[(None, 0.0)]:
            if group and (b is None or times[b]-times[group[-1][0]] > window):
                best = max(group, key=lambda g: g[1])
                found.append((k, best[0], best[1]))
                group = []
            if b is not None:
                group.append((b, jump))
    return found


def camera_cuts(frames, H):
    """Per frame whether the camera cut to another shot there (the frame's cut, or a segment other than the previous
    frame's) with a homography on both sides: the registration starts again and its step is not measured (the frame
    has no camera motion), so every position may jump there by an unknown amount (CUT_STEP), never an identity switch.
    A frame merely without a motion row is no cut (results before the motion rows have none on any frame)."""
    T = len(frames)
    out = np.zeros(T, bool)
    have = np.isfinite(H).all(axis=(1, 2))
    for i in range(1, T):
        f, g = frames[i], frames[i-1]
        out[i] = have[i] and have[i-1] and (bool(f.get('cut')) or f.get('segment', 0) != g.get('segment', 0))
    return out


def _motion_rows(frames, H):
    """(t, M): the frames with a motion row (the camera motion from the previous frame) and a homography on them and on
    the previous frame, and those motions as 3x3 matrices (the row's affine part), invertible ones only."""
    rows = [f.get('motion') for f in frames]
    have = np.array([t > 0 and isinstance(m, list) and len(m) == 6 for t, m in enumerate(rows)], bool)
    have &= np.isfinite(H).all(axis=(1, 2)) & np.r_[False, np.isfinite(H[:-1]).all(axis=(1, 2))]
    t = np.flatnonzero(have)
    if not len(t):
        return t, np.zeros((0, 3, 3))
    M = np.tile(np.eye(3), (len(t), 1, 1))
    M[:, :2] = np.array([rows[i] for i in t], np.float64).reshape(-1, 2, 3)
    det = np.linalg.det(M)
    ok = np.isfinite(det) & (np.abs(det) > 1e-9)
    return t[ok], M[ok]


def registration_steps(frames, H, G, width, height, size):
    """Per frame how far the registration itself moved the visible pitch: the median distance, over STEP_GRID pixels
    that see the pitch (within registration.PITCH_MARGIN of it), between their ground points under this frame's homography and under the
    previous frame's carried by the frame's camera motion (its motion row, the affine part: exact enough in the
    middle of the view, where the median lies). NaN where a frame has no motion, or either frame no homography."""
    T = len(frames)
    out = np.full(T, np.nan)
    t, M = _motion_rows(frames, H)
    if not len(t):
        return out
    xs, ys = np.meshgrid(STEP_GRID[0]*width, STEP_GRID[1]*height)
    pts = np.c_[xs.ravel(), ys.ravel(), np.ones(xs.size)].T                      # (3, P)
    a = G[t] @ pts                                                                  # ground under this frame's homography
    b = G[t-1] @ np.linalg.inv(M) @ pts                                             # ... under the previous one, carried
    with np.errstate(divide='ignore', invalid='ignore'):
        A, B = a[:, :2]/a[:, 2:], b[:, :2]/b[:, 2:]
    use = ((a[:, 2] > 1e-12) & (b[:, 2] > 1e-12) & (A[:, 0] > -PITCH_MARGIN) & (A[:, 0] < size.length+PITCH_MARGIN) &
           (A[:, 1] > -PITCH_MARGIN) & (A[:, 1] < size.width+PITCH_MARGIN))
    d = np.where(use, np.hypot(*(A-B).transpose(1, 0, 2)), np.nan)
    out[t] = np.where(use.sum(axis=1) >= 4, _row_medians(d), np.nan)
    return out


def registration_maps(frames, H, G):
    """Per frame the world -> world map of its registration step, A_t = G_t M_t H_{t-1} (T, 3, 3), M_t the frame's
    camera motion (its motion row's affine part, as registration_steps): a static ground point read at X on the previous
    frame is read at A_t X on this one (about the identity where the registration only carried the camera motion);
    scaled to A[2, 2] = 1. NaN where the frame has no motion row, or either frame no homography."""
    out = np.full((len(frames), 3, 3), np.nan)
    t, M = _motion_rows(frames, H)
    if len(t):
        A = G[t] @ M @ H[t-1]
        scale = A[:, 2, 2]
        ok = np.isfinite(A).all(axis=(1, 2)) & (np.abs(scale) > 1e-12)
        out[t[ok]] = A[ok]/scale[ok, None, None]
    return out


def chain_starts(frames, H):
    """Per frame whether the registration starts again there: the first frame, a camera cut, a new segment, or the
    first frame with a homography after one without. No registration step is spread across one."""
    T = len(frames)
    have = np.isfinite(H).all(axis=(1, 2))
    out = np.zeros(T, bool)
    if T:
        out[0] = True
    for i in range(1, T):
        f, g = frames[i], frames[i-1]
        out[i] = bool(f.get('cut')) or f.get('segment', 0) != g.get('segment', 0) or bool(have[i] and not have[i-1])
    return out


def _normalized(A):
    return A/A[2, 2] if abs(A[2, 2]) > 1e-12 else A


def _number(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _integral(v):
    """Whether v is an integer, Python's or numpy's (a bool is none)."""
    return isinstance(v, numbers.Integral) and not isinstance(v, bool)


def anchor_frames(result):
    """Per frame whether the registration is anchored there, whether that anchor is pinned and whether it is trusted. An
    anchor frame: its calibration's anchorAge is 0 (the frame's own registration is the anchor's), or the summary's
    registration anchors (summary.pitch.registration.anchors: by frame index, checked against the anchor's time) list
    it, which also holds an anchor the registration's sweep did not take on its own frame (Veo's 13.3 s keyframe,
    interpolated there). Pinned: a user keyframe (source PINNED_SOURCES, the frame's calibration's or the listed
    anchor's); re-registered, paint and auto-midfield anchors are not. Trusted: a listed anchor of source
    TRUSTED_SOURCES whose root (the frame index of the anchor its chain started from: the keyframe whose view the
    tracker matched it to, directly or through a relay view rooted at it) is another frame, a pinned one. A root naming
    a listed anchor's frame index is the frames its listings resolved to (by their times, where an index disagrees with
    its listing's time): any pinned one roots it. A root that is missing, no integer (Python's or numpy's; a bool, a
    float or a string is none), outside the frames, the anchor's own frame or no pinned frame (a re-registration rooted
    at the auto-midfield anchor, or at a keyframe of the pre-roll) leaves the anchor automatic, and a pinned frame is
    never trusted. Returns (anchors, pinned, trusted), each (T,) bool."""
    frames = result.get('frames') or []
    T = len(frames)
    anchors, pinned, trusted = np.zeros(T, bool), np.zeros(T, bool), np.zeros(T, bool)
    for i, f in enumerate(frames):
        c = f.get('calibration') or {}
        if _number(c.get('anchorAge')) and c['anchorAge'] == 0:
            anchors[i] = True
            pinned[i] = c.get('source') in PINNED_SOURCES
    listed = (((result.get('summary') or {}).get('pitch') or {}).get('registration') or {}).get('anchors') or ()
    if T and listed:
        times, _ = _frame_times(frames)
        gaps = np.diff(times)
        gaps = gaps[gaps > 1e-6]
        tolerance = max(.5*float(np.median(gaps)), 1e-3) if len(gaps) else 1e-3
        rooted = []
        where = {}       # a listed frame index -> the frames its listings resolved to (by time: see above)
        for a in listed:
            if not isinstance(a, dict):
                continue
            f, t = a.get('frame'), a.get('time')
            i = int(f) if _integral(f) else -1
            ok = 0 <= i < T
            if ok and _number(t) and abs(times[i]-t) > tolerance:
                ok = False
            if not ok and _number(t):
                i = int(np.argmin(np.abs(times-t)))
                ok = abs(times[i]-t) <= tolerance
            if ok:
                anchors[i] = True
                pinned[i] |= a.get('source') in PINNED_SOURCES
                if _integral(f):
                    where.setdefault(int(f), set()).add(i)
                if a.get('source') in TRUSTED_SOURCES:
                    rooted.append((i, a))
        for i, a in rooted:          # once every listed keyframe is known (a root may be listed after the anchors it roots)
            r = a.get('root')
            if _integral(r) and any(0 <= j < T and j != i and pinned[j] for j in where.get(int(r), (int(r),))):
                trusted[i] = True
    return anchors, pinned, trusted & ~pinned


def _pitch_grid(size=None):
    """World points over the whole pitch: where an event's step is measured when nothing better is known."""
    size = size or field.PitchSize()
    xs, ys = np.meshgrid(np.linspace(0, size.length, 8), np.linspace(0, size.width, 5))
    return np.c_[xs.ravel(), ys.ravel()]


def _reach(A, X):
    """How far a world -> world map A (3, 3) moves world points X (N, 2): |A X - X| (N,), NaN where a point does not map
    in front."""
    return np.hypot(*(_carry(np.repeat(A[None], len(X), axis=0), X)-X).T)


def _cancels(reach, n):
    """Whether an event of n steps whose net step moves its points by reach (_reach) cancels: it has more than one step,
    and every point's step is under SPREAD_STEP, finite (a NaN, a point the net step does not map in front, is no
    cancel). An outlier there and back composes to about the identity; a single step is a step, however small."""
    return n > 1 and len(reach) > 0 and bool(np.all(reach < SPREAD_STEP))


def step_events(times, steps, maps, starts, anchors=None, pinned=None, points=None, trusted=None):
    """The registration snaps the display spreads (module docstring). A frame is a step where its registration step
    (registration_steps) exceeds SPREAD_STEP and SPREAD_ISOLATION times the median step of the other frames within
    +-SPREAD_LOCAL, it has a map (registration_maps) and the registration does not start again there (starts:
    chain_starts). anchors, pinned, trusted (T,) bool (anchor_frames; none by default): a step on an anchor frame
    arrives at it, one on the frame after an anchor leaves it, and one on an automatic anchor's frame right after a
    pinned or trusted one leaves that one. A trusted anchor whose arriving and leaving steps cancel (_cancels: their
    composite's step at points(the frame before it) finite and under SPREAD_STEP at its largest) is automatic, its
    arriving step still its own (not a leaving step of an honoured anchor before it: the one the outlier test composes),
    so that its two steps are one automatic event: right after a keyframe or a trusted anchor, one no honoured anchor's
    event takes (after an automatic anchor a leaving step passes over, that event takes them like any automatic steps,
    the honoured frame still exact). A step arriving at a pinned
    or trusted frame, or leaving one, starts an event of its own, spread before it (arriving) or after it (leaving): a
    keyframe's never the other way (that would move the keyframe), a trusted anchor's the other way when squeezed. The
    event is that step alone but for a leaving step on an automatic anchor's frame, which takes the other steps after it
    (at most SPREAD_GAP apart, with no start, pinned or trusted frame between them) into it: the automatic anchor is
    passed over (an outlier there composes to about nothing, and the honoured frame shows its registration). The other
    steps (an automatic anchor's arriving and leaving ones, blips) at most SPREAD_GAP apart, with no start, pinned or
    trusted frame between them, are one event, spread before it. The ramp: the event's step at points(frame before the
    event) (world points (N, 2); the pitch grid when None or empty: median |A X - X|) over GLIDE_SPEED, within RAMP_MIN
    and RAMP_MAX, shortened to end at the nearest pinned frame or registration start (never at a trusted one). Squeezed
    (shorter than RAMP_FLOOR, or its median glide faster than SQUEEZE times max(GLIDE_SPEED, the full ramp's)): spread
    the other way when it may and is not squeezed there, else not spread, unless the event cancels (_cancels: more than
    one step, its net step at the points finite and under SPREAD_STEP at its largest): then spread its first way
    whatever its room.
    Returns, in time order, [{'steps': [frame, ...], 'maps': their maps, 'start': the first step frame t_a, 'end': the
    last t_b, 'map': (3, 3) the event's composite A_{t_b} ... A_{t_a}, 'reason': 'arriving' | 'leaving' | 'blip' |
    'anchor' (an automatic anchor's arriving and leaving steps together; also an honoured anchor's leaving step that
    took an automatic anchor's arrival), 'keyframe' (a pinned frame's step), 'trusted' (a trusted anchor's step),
    'spread': 'before' | 'after' | None (not spread), 'squeezed': None | 'turned' (spread the other way) | 'snapped'
    (not spread), 'ramp': (from, to) seconds (before: share 0 at from, 1 at t_a; after: 1 at t_b, 0 at to) or None,
    'rampSeconds', 'needSeconds' (the ramp the step asks for), 'reach' and 'reachMax' (the step at the points, metres:
    median and largest)}]."""
    T = len(times)
    if not T:
        return []
    anchors = np.zeros(T, bool) if anchors is None else np.asarray(anchors, bool)
    pinned = np.zeros(T, bool) if pinned is None else np.asarray(pinned, bool)
    trusted = np.zeros(T, bool) if trusted is None else np.asarray(trusted, bool) & ~pinned
    with np.errstate(invalid='ignore'):
        candidates = np.flatnonzero(np.isfinite(steps) & (steps > SPREAD_STEP) & np.isfinite(maps).all(axis=(1, 2)) & ~starts &
                                    (np.arange(T) > 0))
    picked = []
    for i in candidates:
        a = int(np.searchsorted(times, times[i]-SPREAD_LOCAL-1e-9, 'left'))
        b = int(np.searchsorted(times, times[i]+SPREAD_LOCAL+1e-9, 'right'))
        around = np.r_[steps[a:i], steps[i+1:b]]
        around = around[np.isfinite(around)]
        if steps[i] > SPREAD_ISOLATION*(float(np.median(around)) if len(around) else 0.0):
            picked.append(int(i))

    def at(i):
        # The points the step of an event starting on frame i is measured at: points(i - 1), else the pitch grid.
        X = points(i-1) if points is not None else None
        return _pitch_grid() if X is None or not len(X) else np.asarray(X, np.float64).reshape(-1, 2)

    def classify(honoured, kept):
        # On an anchor's frame a step arrives at it, on the frame after one it leaves it; on an automatic anchor's
        # frame right after a keyframe or a trusted anchor it leaves that one (the honoured anchor's event, which takes
        # the automatic steps after it: below), unless the frame is a trusted outlier's (kept: honoured before the
        # outliers were found): there it is the outlier's.
        return {i: 'leaving' if anchors[i-1] and (not anchors[i] or honoured[i-1] and not kept[i]) else
                'arriving' if anchors[i] else 'blip' for i in picked}
    kept = pinned | trusted
    reasons = classify(kept, kept)
    for k in np.flatnonzero(trusted):     # a trusted outlier (its two steps cancel) is not honoured: automatic
        if (reasons.get(int(k)) == 'arriving' and reasons.get(int(k)+1) == 'leaving' and
                _cancels(_reach(_normalized(maps[k+1] @ maps[k]), at(int(k))), 2)):
            trusted[k] = False
    reasons = classify(pinned | trusted, kept)
    keyframe = {i: bool(pinned[i] if reasons[i] == 'arriving' else pinned[i-1] if reasons[i] == 'leaving' else False) for i in picked}
    trust = {i: bool(trusted[i] if reasons[i] == 'arriving' else trusted[i-1] if reasons[i] == 'leaving' else False) for i in picked}
    alone = {i: keyframe[i] or trust[i] for i in picked}             # a keyframe's or a trusted anchor's step: starts an event
    honoured = pinned | trusted
    groups = []
    for i in picked:
        last = groups[-1][-1] if groups else None
        # A keyframe's or a trusted anchor's leaving step on an automatic anchor's frame (classify) takes the automatic
        # steps after it into its event, the automatic anchor passed over: an outlier there composes to about nothing.
        passed = last is not None and alone[last] and reasons[last] == 'leaving' and anchors[last]
        if (last is not None and not alone[i] and (not alone[last] or passed) and times[i]-times[last] <= SPREAD_GAP+1e-9 and
                not starts[last+1:i+1].any() and not honoured[last:i].any()):
            groups[-1].append(i)
        else:
            groups.append([i])
    bound = np.asarray(starts, bool) | pinned
    index = np.arange(T)
    before_bound = np.maximum.accumulate(np.where(bound, index, 0))                 # the latest bound at or before a frame
    after_bound = np.minimum.accumulate(np.where(bound, index, T)[::-1])[::-1]       # the earliest at or after it (T: none)
    events = []
    for g in groups:
        a, b = g[0], g[-1]
        kinds = {reasons[u] for u in g}
        reason = 'anchor' if {'arriving', 'leaving'} <= kinds else 'arriving' if 'arriving' in kinds else 'leaving' if 'leaving' in kinds else 'blip'
        composite = np.eye(3)
        for u in g:
            composite = _normalized(maps[u] @ composite)
        reach = _reach(composite, at(a))
        cancels = _cancels(reach, len(g))
        reach = reach[np.isfinite(reach)]
        median, largest = (float(np.median(reach)), float(reach.max())) if len(reach) else (0.0, 0.0)
        need = float(np.clip(median/GLIDE_SPEED, RAMP_MIN, RAMP_MAX))
        lo = int(before_bound[a-1])
        hi = int(after_bound[b+1]) if b+1 < T else T
        room = {'before': float(times[a]-times[lo]), 'after': float(times[hi]-times[b]) if hi < T else math.inf}
        # A keyframe's step goes its one way; a trusted anchor's its way, the other when squeezed; an automatic event's
        # before it, after it only when squeezed before.
        if keyframe[a]:
            ways = ('before',) if reason == 'arriving' else ('after',)
        elif trust[a]:
            ways = ('before', 'after') if reason == 'arriving' else ('after', 'before')
        else:
            ways = ('before', 'after')
        shortest = max(RAMP_FLOOR, min(median/(SQUEEZE*GLIDE_SPEED), need/SQUEEZE))   # the shortest ramp not squeezed
        # The first way with the room; when neither has it, an event that cancels its first way (it moves no point by
        # SPREAD_STEP, so nothing glides however short the ramp) and any other none (it snaps).
        way = next((w for w in ways if min(need, room[w]) >= shortest-1e-9), ways[0] if cancels else None)
        squeezed = None if way == ways[0] else 'turned' if way else 'snapped'
        if way == 'before':
            ramp = (float(times[lo]) if room['before'] <= need else float(times[a])-need, float(times[a]))
        elif way == 'after':
            ramp = (float(times[b]), float(times[hi]) if room['after'] <= need else float(times[b])+need)
        else:
            ramp = None
        events.append({'steps': g, 'maps': [maps[u].copy() for u in g], 'start': a, 'end': b, 'map': composite, 'reason': reason,
                       'keyframe': keyframe[a], 'trusted': trust[a], 'spread': way, 'squeezed': squeezed, 'ramp': ramp,
                       'rampSeconds': None if ramp is None else ramp[1]-ramp[0], 'needSeconds': need, 'reach': median, 'reachMax': largest})
    return events


def _carry(A, z):
    """Projective maps A (N, 3, 3) applied to points z (N, 2): (N, 2), NaN where a point does not map in front."""
    p = (A[:, :, :2] @ z[:, :, None])[:, :, 0]+A[:, :, 2]
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where((p[:, 2:] > 1e-9) & np.isfinite(p).all(axis=1, keepdims=True), p[:, :2]/p[:, 2:], np.nan)


def _step_share(times, event, n):
    """The share of the display correction the n-th step v of a spread event carries, over the frames where it is not
    0: (lo, hi, share (hi - lo,)) for the frames lo <= t < hi. Spread before: rising linearly in time from 0 at the ramp's
    start to 1 at t_a, 1 from t_a up to the frame before v (the event's later frames show the registration after it).
    Spread after: -1 from v to t_b (the event's frames show the registration before it), then back to 0 linearly at the
    ramp's end. A ramp of no length (a cancelling event spread into no room) has no frame on it, so nothing is divided
    by its length."""
    v, a, b = event['steps'][n], event['start'], event['end']
    t0, t1 = event['ramp']
    if event['spread'] == 'before':
        lo = min(int(np.searchsorted(times, t0, 'right')), a)
        share = np.ones(v-lo)
        on = np.arange(lo, v) < a
        share[on] = np.clip((times[lo:v][on]-t0)/(t1-t0), 0.0, 1.0)
        return lo, v, share
    hi = max(int(np.searchsorted(times, t1, 'left')), b+1)
    share = -np.ones(hi-v)
    on = np.arange(v, hi) > b
    share[on] = -np.clip(1-(times[v:hi][on]-t0)/(t1-t0), 0.0, 1.0)
    return v, hi, share


def spread_shares(times, events):
    """Every spread step's share of the display correction on every frame (_step_share): (columns [(event index, step
    frame)], shares (T, columns)). A step's share s on frame t moves a position X there by s (A_v W - W), W the position
    carried to the registration before the step (spread_steps)."""
    columns = [(k, v) for k, e in enumerate(events) if e.get('spread') for v in e['steps']]
    out = np.zeros((len(times), len(columns)))
    c = 0
    for k, e in enumerate(events):
        if not e.get('spread'):
            continue
        for n in range(len(e['steps'])):
            lo, hi, share = _step_share(times, e, n)
            out[lo:hi, c] = share
            c += 1
    return columns, out


def _carries(chain, v, lo, hi):
    """Per frame lo <= t < hi the map from frame t's registration to the registration just before the step on frame v
    (S_{v-1} S_t^-1 with S the product of the registration steps chain {frame: A} up to a frame): through the chain's
    steps between t and v (after v: their inverses, v's own included)."""
    P = np.empty((max(hi-lo, 0), 3, 3))
    M = np.eye(3)
    for t in range(min(hi, v)-1, lo-1, -1):
        if t < v-1 and t+1 in chain:
            M = _normalized(M @ chain[t+1])
        P[t-lo] = M
    M = np.eye(3)
    for t in range(max(lo, v), hi):
        if t in chain:
            M = _normalized(M @ np.linalg.inv(chain[t]))
        P[t-lo] = M
    return P


def _spread(frame, z, times, events):
    """spread_steps, and per event the largest correction its own steps made (metres; 0 when not spread)."""
    out = z.copy()
    moved = np.zeros(len(z), bool)
    largest = [0.0]*len(events)
    if not any(e.get('spread') for e in events) or not len(z):
        return out, moved, largest
    chain = {u: A for e in events for u, A in zip(e['steps'], e['maps'])}
    order = np.argsort(frame, kind='stable')
    sorted_frames = frame[order]
    for k, e in enumerate(events):
        if not e.get('spread'):
            continue
        own = np.zeros(z.shape)
        for n, (v, A) in enumerate(zip(e['steps'], e['maps'])):
            lo, hi, share = _step_share(times, e, n)
            first, last = np.searchsorted(sorted_frames, [lo, hi])
            rows = order[first:last]
            if not len(rows):
                continue
            local = frame[rows]-lo
            W = _carry(_carries(chain, v, lo, hi)[local], z[rows])
            d = share[local, None]*(_carry(np.repeat(A[None], len(rows), axis=0), W)-W)
            ok = np.isfinite(d).all(axis=1) & (share[local] != 0)
            out[rows[ok]] += d[ok]
            own[rows[ok]] += d[ok]
            moved[rows[ok]] = True
        largest[k] = float(np.hypot(*own.T).max())
    return out, moved, largest


def spread_steps(frame, z, times, events):
    """The display correction of the step events (step_events) of observations: frame (N,) indices, z (N, 2) metres.
    Every spread step v moves a position X on frame t by its share s (spread_shares) of A_v W - W, W = S_{v-1} S_t^-1 X
    the position carried to the registration just before the step through the events' steps between t and v (S: their
    product up to a frame). The display's registration S_t + sum_v s_v J_v (J_v = S_v - S_{v-1}, the step's jump of a
    ground point) is then continuous wherever the shares are (each step's share falls by 1 on its frame, where S jumps
    by J_v), and the shares of overlapping ramps add. Returns (the corrected positions (N, 2), moved (N,) bool: the
    observations corrected); a position that does not map keeps its own."""
    out, moved, _ = _spread(frame, z, times, events)
    return out, moved


def _loo_d2(z, R, Reff, used, xs, Ps):
    """Squared Mahalanobis distance of each observation from the trajectory the other observations give, against the
    nominal R: exact for the linear smoother, z - x_loo = Reff (Reff - Ps)^-1 (z - xs) and P_loo = Ps (Reff - Ps)^-1
    Reff, where Reff is the covariance it was used with (an unused observation: x_loo = xs, P_loo = Ps). 0 where the
    other observations do not constrain the trajectory there (a lone sample)."""
    r = z-xs[..., :2]
    S = Ps[..., :2, :2]
    A = Reff-S
    detA = A[..., 0, 0]*A[..., 1, 1]-A[..., 0, 1]*A[..., 1, 0]
    detR = Reff[..., 0, 0]*Reff[..., 1, 1]-Reff[..., 0, 1]*Reff[..., 1, 0]
    lone = used & ~(detA > 1e-9*np.abs(detR))
    with np.errstate(all='ignore'):
        Ainv = _inv2(np.where(lone[..., None, None] | ~used[..., None, None], np.eye(2), A))
        r_loo = np.where(used[..., None], (Reff @ Ainv @ r[..., None])[..., 0], r)
        P_loo = np.where(used[..., None, None], S @ Ainv @ Reff, S)
        P_loo = .5*(P_loo+np.swapaxes(P_loo, -1, -2))
        C = R+P_loo
        d2 = (r_loo[..., None, :] @ _inv2(C) @ r_loo[..., :, None])[..., 0, 0]
    return np.where(lone | ~np.isfinite(d2), 0.0, d2)


def _whitened(z, R, xs):
    """Residuals z - xs whitened by the Cholesky factor of R: (T, K, 2), white with unit variance when the trajectory
    is right and R is the noise."""
    r = z-xs[..., :2]
    l11 = np.sqrt(R[..., 0, 0])
    l21 = R[..., 1, 0]/l11
    l22 = np.sqrt(np.maximum(R[..., 1, 1]-l21*l21, 1e-12))
    u1 = r[..., 0]/l11
    return np.stack([u1, (r[..., 1]-l21*u1)/l22], axis=-1)


def _window_bounds(times, seg, half):
    """Per step and column the first and last step of the same segment within +-half seconds: (lo, hi) (T, K)."""
    T, K = seg.shape
    lo = np.searchsorted(times, times-half-1e-9, side='left')
    hi = np.searchsorted(times, times+half+1e-9, side='right')-1
    first = np.zeros((T, K), np.int64)
    last = np.zeros((T, K), np.int64)
    for k in range(K):
        s = seg[:, k]
        on = s >= 0
        change = np.flatnonzero(np.r_[True, s[1:] != s[:-1]])
        ends = np.r_[change[1:], T]-1
        for a, b in zip(change, ends):
            if on[a]:
                first[a:b+1, k], last[a:b+1, k] = a, b
    return np.maximum(lo[:, None], first), np.minimum(hi[:, None], last)


def _window_sums(bounds, values):
    """Per step and column the sum of values (T, K, ...) over the window bounds (_window_bounds)."""
    lo, hi = bounds
    cs = np.concatenate([np.zeros((1,)+values.shape[1:]), np.cumsum(values, axis=0)])
    cols = np.arange(values.shape[1])[None]
    return cs[hi+1, cols]-cs[lo, cols]


def _implausible(times, xs, boost, accel_bounds, opt):
    """The steps of the raised-q runs (contiguous boost > 0 along time) whose trajectory is faster than vPlausible or
    changes its velocity faster than aPlausible somewhere (over the accel_bounds window): (T, K) bool."""
    runs = _runs(boost > 0)
    if not (runs >= 0).any():
        return np.zeros(boost.shape, bool)
    v = xs[..., 2:]
    lo, hi = accel_bounds
    cols = np.arange(v.shape[1])[None]
    with np.errstate(invalid='ignore', divide='ignore'):
        accel = np.hypot(*(v[hi, cols]-v[lo, cols]).transpose(2, 0, 1))/(times[hi]-times[lo])
        bad = (np.hypot(v[..., 0], v[..., 1]) > opt['vPlausible']) | (accel > opt['aPlausible'])
    bad = np.unique(runs[(runs >= 0) & bad])
    return np.isin(runs, bad) & (runs >= 0)


def _runs(mask):
    """Labels of the contiguous True runs of mask (T, K) along time, unique over columns; -1 where False."""
    T, K = mask.shape
    m = mask.T.reshape(-1)
    starts = m & np.r_[True, ~m[:-1]]
    starts[::T] = m[::T]
    label = np.cumsum(starts)-1
    return np.where(m, label, -1).reshape(K, T).T


DEFAULTS = {'qLow': Q_LOW, 'qHigh': Q_HIGH, 'iterations': ITERATIONS, 'bridgeGap': BRIDGE_GAP, 'leanWindow': LEAN_WINDOW,
            'leanStart': LEAN_START, 'leanFull': LEAN_FULL, 'inlierD2': INLIER_D2, 'rejectD2': REJECT_D2, 'vPlausible': V_PLAUSIBLE,
            'aPlausible': A_PLAUSIBLE, 'switch': True, 'switchSpeed': SWITCH_SPEED, 'switchMin': SWITCH_MIN, 'switchWindow': SWITCH_WINDOW, 'switchPersist': SWITCH_PERSIST,
            'switchNoise': SWITCH_NOISE, 'vetoQ': VETO_Q,
            'robust': True, 'adaptive': True}


def smooth_tracks(times, z, R, quality=None, options=None):
    """The smoother on arrays. times (T,) seconds, increasing; z (T, K, 2) metres, NaN where a column has no
    observation; R (T, K, 2, 2) the observations' covariance; quality (T, K) truthy where an observation's residual may
    raise the process noise (default: all); options overrides DEFAULTS and may carry 'starts' (T, K) bool: steps where
    a new segment must start (a column that holds several tracks one after another), 'calibrationSteps' (T,) bool:
    frames where the registration jumped (no jump across one splits a trajectory), and 'jumps' (T,) m^2: extra
    position process variance on those frames (every position may jump there). A column's observations form
    segments (BRIDGE_GAP, unexplained jumps); each segment is smoothed on its own. Returns {'x' (T, K, 2), 'v' (T, K, 2),
    'P' (T, K, 4, 4) the state covariance (NaN outside the segments), 'w' (T, K) the observation weights (1 full, < 1
    down-weighted, 0 rejected; NaN without an observation), 'kind' (T, K) 'O' | 'I' | '', 'q' (T, K) the process noise
    used, 'd2' (T, K) the leave-one-out Mahalanobis distance^2, 'segment' (T, K) ids (-1 outside), 'flags' [{'column',
    'step', 'reason', 'd2' or 'jump'}], 'passes'}."""
    opt = dict(DEFAULTS, **(options or {}))
    times = np.asarray(times, np.float64)
    z = np.asarray(z, np.float64)
    R = np.asarray(R, np.float64)
    T, K = z.shape[:2]
    if not T:
        return {'x': np.zeros((0, K, 2)), 'v': np.zeros((0, K, 2)), 'P': np.zeros((0, K, 4, 4)), 'w': np.zeros((0, K)), 'kind': np.zeros((0, K), '<U1'),
                'q': np.zeros((0, K)), 'd2': np.zeros((0, K)), 'segment': np.zeros((0, K), np.int64), 'flags': [], 'passes': 0}
    observed = np.isfinite(z).all(axis=-1) & np.isfinite(R).all(axis=(-1, -2))
    z = np.where(observed[..., None], z, 0.0)
    R = np.where(observed[..., None, None], R, np.eye(2))
    good = observed if quality is None else observed & np.asarray(quality, bool)
    starts = np.zeros((T, K), bool) if opt.get('starts') is None else np.asarray(opt['starts'], bool)
    seg, start = _segments(times, observed, starts, opt['bridgeGap'])
    flags = []
    if opt['switch']:
        jumps = _switches(times, z, R, good, seg, opt, observed)
        if jumps:
            starts = starts.copy()
            for k, b, jump in jumps:
                starts[b, k] = True
                flags.append({'column': k, 'step': b, 'reason': 'unexplained jump: trajectory split (an identity switch?)', 'jump': round(jump, 2)})
            seg, start = _segments(times, observed, starts, opt['bridgeGap'])
    q_low, q_high = opt['qLow'], opt['qHigh']
    vetoed = math.log(max(opt['vetoQ'], q_low)/q_low)/math.log(q_high/q_low) if q_high > q_low else 0.0   # the boost a veto leaves
    w = np.where(observed, 1.0, 0.0)
    boost = np.zeros((T, K))
    veto = np.zeros((T, K), bool)
    q = np.full((T, K), q_low)
    bounds = _window_bounds(times, seg, opt['leanWindow']/2)
    accel = _window_bounds(times, seg, ACC_HALF)
    steps = _steps(times, opt.get('jumps'))
    xs, Ps = _rts(steps, z, R, observed, q, seg, start)
    passes = 1
    for it in range(opt['iterations']+1):
        used = observed & (w > 0)
        Reff = R/np.where(used, w, 1.0)[..., None, None]
        d2 = _loo_d2(z, R, Reff, used, xs, Ps)
        if it == opt['iterations']:
            break
        if opt['robust']:
            w = np.where(observed, np.where(d2 <= opt['inlierD2'], 1.0, opt['inlierD2']/np.maximum(d2, 1e-9)), 0.0)
            w[observed & (d2 > opt['rejectD2'])] = 0.0
        if opt['adaptive']:
            # A raised q that took the trajectory faster than a player runs or accelerates was following an excursion:
            # withdrawn there for good, and those residuals no longer count.
            veto |= _implausible(times, xs, boost, accel, opt)
            lean_ok = good & (w >= FLAG_WEIGHT) & ~veto
            u = np.where(lean_ok[..., None], _whitened(z, R, xs), 0.0)
            su = _window_sums(bounds, u)
            n = _window_sums(bounds, lean_ok.astype(np.float64))
            with np.errstate(all='ignore'):
                lean = np.where(n >= LEAN_MIN, (su*su).sum(axis=-1)/np.maximum(n, 1), 0.0)
            # The boost only accumulates: a pass that follows the manoeuvre has white residuals there, which is the
            # raised q working, not a reason to lower it again (dropping it made the passes alternate).
            boost = np.maximum(boost, np.clip((lean-opt['leanStart'])/(opt['leanFull']-opt['leanStart']), 0, 1))
            boost[veto] = np.minimum(boost[veto], vetoed)
            q = q_low*(q_high/q_low)**boost
        xs, Ps = _rts(steps, z, R/np.where(observed & (w > 0), w, 1.0)[..., None, None], observed & (w > 0), q, seg, start)
        passes += 1
    if opt['adaptive'] and (boost > 0).any():
        # The last pass's manoeuvres checked once more: an implausible one is smoothed again without its raised q.
        bad = _implausible(times, xs, boost, accel, opt)
        if bad.any():
            boost[bad] = np.minimum(boost[bad], vetoed)
            q = q_low*(q_high/q_low)**boost
            used = observed & (w > 0)
            xs, Ps = _rts(steps, z, R/np.where(used, w, 1.0)[..., None, None], used, q, seg, start)
            d2 = _loo_d2(z, R, R/np.where(used, w, 1.0)[..., None, None], used, xs, Ps)
            passes += 1
    inside = seg >= 0
    kind = np.where(observed, 'O', np.where(inside, 'I', ''))
    for i, k in zip(*np.nonzero(observed & (w < FLAG_WEIGHT))):
        flags.append({'column': int(k), 'step': int(i), 'd2': round(float(d2[i, k]), 1),
                      'reason': 'rejected: far off the trajectory' if w[i, k] <= 0 else 'down-weighted: off the trajectory'})
    return {'x': np.where(inside[..., None], xs[..., :2], np.nan), 'v': np.where(inside[..., None], xs[..., 2:], np.nan),
            'P': np.where(inside[..., None, None], Ps, np.nan), 'w': np.where(observed, w, np.nan), 'kind': kind, 'q': q,
            'd2': np.where(observed, d2, np.nan), 'segment': seg, 'flags': flags, 'passes': passes}


# ---------- the result ----------
def _lanes(spans):
    """Columns for keys with (first, last) frame spans: keys whose spans do not overlap share a column (the forced
    segment start keeps them apart). Returns the column per key and the number of columns."""
    column, ends = {}, []
    for k in sorted(spans, key=lambda k: spans[k]):
        first, last = spans[k]
        for c, end in enumerate(ends):
            if end < first:
                column[k], ends[c] = c, last
                break
        else:
            column[k] = len(ends)
            ends.append(last)
    return column, len(ends)


def _series(values, digits):
    """A float array as a JSON list, None where NaN."""
    return [None if v != v else v for v in np.round(values, digits).tolist()]


def _constants():
    return {'bridgeGap': BRIDGE_GAP, 'minConfidence': MIN_CONFIDENCE, 'fitHalf': FIT_HALF, 'heightHalf': HEIGHT_HALF, 'eventPx': EVENT_PX,
            'eventShare': EVENT_SHARE, 'eventTop': EVENT_TOP, 'repairPx': REPAIR_PX, 'repairShare': REPAIR_SHARE,
            'priorBottom': list(PRIOR_BOTTOM), 'priorCentre': list(PRIOR_CENTRE), 'priorSamples': PRIOR_SAMPLES, 'occluded': OCCLUDED,
            'lowScore': LOW_SCORE, 'registrationFloor': REGISTRATION_FLOOR, 'clippedFoot': CLIPPED_FOOT, 'qLow': Q_LOW, 'qHigh': Q_HIGH,
            'leanWindow': LEAN_WINDOW, 'leanStart': LEAN_START, 'leanFull': LEAN_FULL, 'vPlausible': V_PLAUSIBLE, 'aPlausible': A_PLAUSIBLE, 'accHalf': ACC_HALF, 'inlierD2': INLIER_D2,
            'rejectD2': REJECT_D2, 'flagWeight': FLAG_WEIGHT, 'iterations': ITERATIONS, 'switchSpeed': SWITCH_SPEED, 'switchMin': SWITCH_MIN,
            'switchWindow': SWITCH_WINDOW, 'switchSigma': SWITCH_SIGMA, 'switchPersist': SWITCH_PERSIST, 'registrationStep': REGISTRATION_STEP,
            'stepFreedom': STEP_FREEDOM, 'heightAgree': HEIGHT_AGREE, 'repairCommon': REPAIR_COMMON, 'ratioHalf': RATIO_HALF,
            'ratioMin': RATIO_MIN, 'ratioRange': list(RATIO_RANGE), 'geoPrior': list(GEO_PRIOR), 'geoAgree': GEO_AGREE, 'doubtSigma': DOUBT_SIGMA,
            'doubtSpell': DOUBT_SPELL, 'switchNoise': SWITCH_NOISE, 'vetoQ': VETO_Q, 'cutStep': CUT_STEP, 'spreadStep': SPREAD_STEP,
            'spreadIsolation': SPREAD_ISOLATION, 'spreadLocal': SPREAD_LOCAL, 'spreadGap': SPREAD_GAP, 'glideSpeed': GLIDE_SPEED,
            'rampMin': RAMP_MIN, 'rampMax': RAMP_MAX, 'rampFloor': RAMP_FLOOR, 'squeeze': SQUEEZE, 'pinnedSources': list(PINNED_SOURCES),
            'trustedSources': list(TRUSTED_SOURCES)}


def _event_summary(e, times, steps, largest):
    """A step event for noise.registrationSteps: its times, the steps' grid medians, why and which way it is spread
    (None: not spread, it keeps the jump freedom), whether it is a keyframe's or a trusted anchor's (a trusted outlier's
    steps are an automatic anchor's) and whether its room squeezed it (turned: spread the other way; snapped: not
    spread), the ramp's length and the one its step asks for, the full step at the people on the frame before it
    (|A_event X - X|, metres: median and max; the visible grid's without anyone) and the largest correction its own
    steps made."""
    a, b = e['start'], e['end']
    return {'from': round(float(times[a]), 3), 'to': round(float(times[b]), 3), 'grid': [round(float(steps[u]), 3) for u in e['steps']],
            'reason': e['reason'], 'keyframe': e['keyframe'], 'trusted': e['trusted'], 'spread': e['spread'], 'squeezed': e['squeezed'],
            'rampSeconds': None if e['rampSeconds'] is None else round(e['rampSeconds'], 3),
            'needSeconds': round(e['needSeconds'], 3), 'stepMetres': {'median': round(e['reach'], 3), 'max': round(e['reachMax'], 3)},
            'largestCorrection': round(largest, 3)}


def _event_points(frames, G, width, height, size):
    """points(frame) for step_events: on a calibrated frame the stored pitch positions of its people within PITCH_MARGIN
    of the pitch, whoever they are (identified or not, any role or state: a correction relabels people and never moves
    them, so it never changes a ramp; measured at the observations only, excluding B-03 on the amateur run changed 7
    ramps and moved 8048 samples of the other players by up to 0.23 m, excluding B-01 on the broadcast run 10 ramps and
    12899 samples by up to 0.15 m), or without any the STEP_GRID pixels that see the pitch through the frame's
    homography (registration_steps' grid), or None. The largest step (the cancel test, _cancels) is over them too,
    bystanders near the lines included, which makes that test stricter where the step turns or scales."""
    xs, ys = np.meshgrid(STEP_GRID[0]*width, STEP_GRID[1]*height)
    pixels = np.c_[xs.ravel(), ys.ravel(), np.ones(xs.size)].T

    def on_pitch(X):
        return ((X[:, 0] > -PITCH_MARGIN) & (X[:, 0] < size.length+PITCH_MARGIN) & (X[:, 1] > -PITCH_MARGIN) &
                (X[:, 1] < size.width+PITCH_MARGIN))

    def points(i):
        if 0 <= i < len(frames) and frames[i].get('calibrated'):
            stored = [p['pitch'][:2] for p in frames[i].get('people') or () if isinstance(p.get('pitch'), list) and
                      len(p['pitch']) >= 2 and all(_number(v) for v in p['pitch'][:2])]
            X = np.array(stored, np.float64).reshape(-1, 2)*[size.length, size.width]
            X = X[on_pitch(X)]
            if len(X):
                return X
        if not (0 <= i < len(G)) or not np.isfinite(G[i]).all():
            return None
        g = G[i] @ pixels
        with np.errstate(divide='ignore', invalid='ignore'):
            X = (g[:2]/g[2:]).T
        use = (g[2] > 1e-12) & on_pitch(X)
        return X[use] if use.sum() >= 4 else None
    return points


def _display_events(result, times, H, G, width, height, size):
    frames = result.get('frames') or []
    steps = registration_steps(frames, H, G, width, height, size)
    anchors, pinned, trusted = anchor_frames(result)
    events = step_events(times, steps, registration_maps(frames, H, G), chain_starts(frames, H), anchors, pinned,
                         _event_points(frames, G, width, height, size), trusted)
    return steps, anchors, pinned, trusted, events


def display_events(result, width, height, size=None):
    """The registration step events stabilize() spreads on a result, as it computes them: (frame times (T,), events
    (step_events), pinned (T,) bool). spread_steps(frame, z, times, events) is then the display correction of any
    positions (metres) on those frames."""
    frames = result.get('frames') or []
    if size is None:
        size = field.PitchSize.from_json((result.get('tactical') or {}).get('pitchSize'))
    times, _ = _frame_times(frames)
    H, G = _homographies(frames, width, height)
    _, _, pinned, _, events = _display_events(result, times, H, G, width, height, size)
    return times, events, pinned


def stabilize(result, width, height, size=None, noise=None):
    """The stabilized observed positions of a finished result (see the module docstring). width, height: the video's
    pixels (the boxes are normalized); size: the pitch (default: the result's tactical pitchSize). Returns
    {'frameTimes', 'ids': {pid: track}, 'anonymous': {str(local track): track}, 'flags', 'noise', 'summary',
    'constants', 'method'}; a track is {'x', 'y' (metres), 'vx', 'vy' (m/s), 'sx', 'sy' (sigma, metres), 'kind' ('O'
    stabilized observation, 'I' bridged inside a gap of at most BRIDGE_GAP, None), 'w' (the observation's weight: 1
    normal, < 1 down-weighted, 0 rejected, None without one)}, each a list over the frames. Every roster slot is in
    'ids' (all None when never observed). A flag is {'id' (None on an anonymous track), 'track', 'frame' (index into
    the result's frames), 'time', 'reason', 'raw' [x, y] metres, 'used' [x, y] metres or None (rejected)}."""
    started = time.perf_counter()
    frames = result.get('frames') or []
    if size is None:
        size = field.PitchSize.from_json((result.get('tactical') or {}).get('pitchSize'))
    times, _ = _frame_times(frames)
    T = len(frames)
    obs = observations(result, width, height, size)
    H, G = _homographies(frames, width, height)
    given = copy.deepcopy(noise) if isinstance(noise, dict) and noise.get('bottom') and noise.get('centre') and noise.get('inflation') else None
    fits = edge_fits(obs, H, G, (given or {}).get('footage'))
    boxes = box_events(obs, fits, (given or {}).get('geometric'))
    event, repaired_y = boxes['event'], boxes['repaired']
    noise = given or fit_noise(obs, fits, event | (boxes['doubt'] > 0))
    noise['geometric'] = boxes['model']
    noise['footage'] = fits['footage']
    z, R, repaired, good = measurement_covariance(obs, H, G, noise, boxes)
    # The registration's snaps spread for the display: the observations the smoother takes, never the raw ones.
    steps, anchors, pinned, trusted, events = _display_events(result, times, H, G, width, height, size)
    z_repaired = z
    z, spread, largest = _spread(obs['frame'], z_repaired, times, events)
    keys = obs['keys']
    N = len(obs['frame'])
    spans = {}
    for k in np.unique(obs['key']) if N else []:
        f = obs['frame'][obs['key'] == k]
        spans[int(k)] = (int(f.min()), int(f.max()))
    column, K = _lanes(spans)
    col = np.array([column[int(k)] for k in obs['key']], np.int64) if N else np.zeros(0, np.int64)
    Z = np.full((T, max(K, 1), 2), np.nan)
    RR = np.tile(np.eye(2), (T, max(K, 1), 1, 1))
    Q = np.zeros((T, max(K, 1)), bool)
    starts = np.zeros((T, max(K, 1)), bool)
    Z[obs['frame'], col], RR[obs['frame'], col], Q[obs['frame'], col] = z, R, good
    for k, (first, _) in spans.items():
        starts[first, column[k]] = True
    stepped = steps > REGISTRATION_STEP
    cuts = camera_cuts(frames, H)
    spread_frames = np.zeros(T, bool)
    spread_frames[[u for e in events if e['spread'] for u in e['steps']]] = True
    free = stepped & ~spread_frames      # a step that is not spread keeps its jump freedom
    jumps = np.where(free, (STEP_FREEDOM*np.nan_to_num(steps))**2, 0.0)
    jumps = np.where(cuts, np.maximum(jumps, (STEP_FREEDOM*CUT_STEP)**2), jumps)
    smooth = smooth_tracks(times, Z, RR, Q, {'starts': starts, 'calibrationSteps': stepped | cuts, 'jumps': jumps})
    # Unpack the columns into the keys' tracks.
    owner = np.full((T, max(K, 1)), -1, np.int64)
    for k, (first, last) in spans.items():
        owner[first:last+1, column[k]] = k
    calibrated = np.array([bool(f.get('calibrated')) for f in frames], bool)
    empty = {'x': [None]*T, 'y': [None]*T, 'vx': [None]*T, 'vy': [None]*T, 'sx': [None]*T, 'sy': [None]*T, 'kind': [None]*T, 'w': [None]*T}
    out_ids, out_anonymous = {}, {}
    bridged = 0
    for k, (kind_name, name) in enumerate(keys):
        if k not in spans:
            if kind_name == 'id':
                out_ids[name] = {key: list(v) for key, v in empty.items()}
            continue
        c = column[k]
        mine = (owner[:, c] == k) & (smooth['kind'][:, c] != '') & calibrated
        x = np.where(mine[:, None], smooth['x'][:, c], np.nan)
        v = np.where(mine[:, None], smooth['v'][:, c], np.nan)
        P = smooth['P'][:, c]
        sig = np.where(mine[:, None], np.sqrt(np.maximum(np.stack([P[:, 0, 0], P[:, 1, 1]], axis=1), 0.0)), np.nan)
        kinds = [str(s) if m else None for s, m in zip(smooth['kind'][:, c], mine)]
        bridged += sum(1 for s in kinds if s == 'I')
        track = {'x': _series(x[:, 0], 3), 'y': _series(x[:, 1], 3), 'vx': _series(v[:, 0], 2), 'vy': _series(v[:, 1], 2),
                 'sx': _series(sig[:, 0], 3), 'sy': _series(sig[:, 1], 3), 'kind': kinds,
                 'w': _series(np.where(mine, smooth['w'][:, c], np.nan), 3)}
        (out_ids if kind_name == 'id' else out_anonymous)[name if kind_name == 'id' else str(name)] = track
    # Flags: the repairs, then the smoother's rejections, down-weightings and splits.
    at = {(int(c), int(f)): n for n, (c, f) in enumerate(zip(col, obs['frame']))}
    flags = []

    def describe(n, reason, used):
        kind_name, name = keys[int(obs['key'][n])]
        i = int(obs['frame'][n])
        track = obs['track'][n]
        return {'id': name if kind_name == 'id' else None, 'track': None if track == -1 else track, 'frame': i, 'time': round(float(times[i]), 3),
                'reason': reason, 'raw': [round(float(obs['z'][n, 0]), 2), round(float(obs['z'][n, 1]), 2)],
                'used': None if used is None else [round(float(used[0]), 2), round(float(used[1]), 2)]}
    rb = obs['fy']-np.nan_to_num(repaired_y)
    for n in np.flatnonzero(event):
        if repaired[n]:
            height = 'the geometric height' if boxes['geometric'][n] else 'the running height'
            reason = f'box lost the legs: foot repaired from the top and {height}' if rb[n] < 0 else \
                f'box reaches into a nearer player: foot repaired from the top and {height}'
            flags.append(describe(n, reason, z[n]))
        else:
            flags.append(describe(n, 'box-extent event: not repairable (the repaired foot sees no ground)', z[n]))
    for flag in smooth['flags']:
        n = at.get((flag['column'], flag['step']))
        if n is None:
            continue
        rejected = flag['reason'].startswith('rejected')
        flags.append(describe(n, flag['reason'], None if rejected else z[n]))
    flags.sort(key=lambda f: (f['frame'], str(f['id']), str(f['track'])))
    # Diagnostics.
    on = np.isfinite(smooth['w'])
    w_all = smooth['w'][on]
    correction = np.hypot(*(smooth['x'][obs['frame'], col]-obs['z']).T) if N else np.zeros(0)
    correction = correction[np.isfinite(correction)]
    segments = len(np.unique(smooth['segment'][smooth['segment'] >= 0]))
    shift = np.hypot(*(z-z_repaired).T) if N else np.zeros(0)   # the spread's own correction per observation
    spread &= shift > 1e-3     # counted from a millimetre (an event that composes to the identity ramps by nothing)
    on_pinned = pinned[obs['frame']] if N else np.zeros(0, bool)
    noise['registrationSteps'] = {'frames': int(np.sum(steps > REGISTRATION_STEP)), 'largestMetres': round(float(np.nanmax(steps)), 2) if np.isfinite(steps).any() else None,
                                  'cuts': int(cuts.sum()), 'spreadEvents': sum(1 for e in events if e['spread']),
                                  'spreadFrames': int(spread_frames.sum()),
                                  'jumpFrames': int(free.sum()),
                                  'events': [_event_summary(e, times, steps, s) for e, s in zip(events, largest)],
                                  'anchorFrames': int(anchors.sum()), 'pinnedFrames': int(pinned.sum()), 'trustedFrames': int(trusted.sum()),
                                  'trustedEvents': sum(1 for e in events if e['trusted']),
                                  'notSpread': sum(1 for e in events if not e['spread']),
                                  'squeezed': {k: sum(1 for e in events if e['squeezed'] == k) for k in ('turned', 'snapped')},
                                  'pinnedCorrectionMetres': round(float(shift[on_pinned].max()), 6) if on_pinned.any() else None}
    noise['events'] = {'legsLost': int((event & repaired & (rb < 0)).sum()), 'reachesNearer': int((event & repaired & (rb > 0)).sum()),
                       'notRepairable': int((event & ~repaired).sum()), 'doubtful': int((boxes['doubt'] > 0).sum()),
                       'geometricShare': round(float(boxes['geometric'].mean()), 4) if N else 0.0,
                       'judged': int(np.isfinite(fits['residual'][:, 1]).sum())}
    summary = {'observations': N, 'identities': sum(1 for k in spans if keys[k][0] == 'id'), 'anonymousTracks': sum(1 for k in spans if keys[k][0] == 'anonymous'),
               'columns': K, 'segments': segments, 'splits': sum(1 for f in smooth['flags'] if f['reason'].startswith('unexplained')),
               'bridgedSamples': bridged, 'rejectedShare': round(float(np.mean(w_all <= 0)), 4) if len(w_all) else 0.0,
               'downweightedShare': round(float(np.mean((w_all > 0) & (w_all < FLAG_WEIGHT))), 4) if len(w_all) else 0.0,
               'repairedShare': round(float(repaired.mean()), 4) if N else 0.0,
               'raisedProcessNoiseShare': round(float(np.mean(smooth['q'][on] > Q_LOW*1.01)), 4) if len(w_all) else 0.0,
               'correctionMetres': {'median': round(float(np.median(correction)), 3), 'p90': round(float(np.percentile(correction, 90)), 3),
                                    'max': round(float(correction.max()), 2)} if len(correction) else None,
               'spreadMetres': {'observations': int(spread.sum()), 'median': round(float(np.median(shift[spread])), 3),
                                'p90': round(float(np.percentile(shift[spread], 90)), 3), 'max': round(float(shift[spread].max()), 3)} if spread.any() else None,
               'passes': smooth['passes'], 'runtimeSeconds': round(time.perf_counter()-started, 3)}
    return {'frameTimes': [round(float(t), 3) for t in times], 'ids': out_ids, 'anonymous': out_anonymous, 'flags': flags,
            'noise': noise, 'summary': summary, 'constants': _constants(), 'method': METHOD}
