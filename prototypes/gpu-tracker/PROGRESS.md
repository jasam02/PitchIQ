# Progress: amateur footage and full-roster reconstruction

A running record for whoever continues this work (people or future sessions). Every number here comes
from a run that is named by its directory and the commit that produced it; nothing is copied from memory.
The approved plan (phases, acceptance criteria, research notes) lives outside the repository; its phase
order is: 0 harness → 1 calibration fixes → 2 identity coverage, roster, visibility → 3 adaptive detection
and tracking → 4 off-screen estimation (benchmarked on full-pitch data) → 5 tactical view kinds and export
→ 6 evidence-gated extras (learned calibration, lens undistortion, merges, tiling, speed).

## Evaluation harness (`tools/`)

- `tools/run_segment.py --video <upload key|path> --start S --seconds N --out DIR [--drop-keyframe T] [--no-calibrations] [--batch jobs.json]`
  runs the pipeline into a scratch directory with the saved settings of that video (`runs/<key>/settings.json`), never writing under `runs/`.
  A batch runs several segments in one process, so all of them use the same code even if files are edited meanwhile.
- `tools/compare.py A [B] [--json out]` prints the metrics of one run or two runs side by side with the agreement of B with A
  (people matched by box overlap; label agreement exact and under the best id mapping; pitch position distance in metres).
- `tools/holdout.py RUN_DIR [--json out]` rebuilds the registration without each calibration keyframe (and without the anchors
  rooted at it) and scores the rebuilt homography on that keyframe's landmarks; also pan stability and the disagreement share.
  Needs the run's `registration.json`.
- Unit tests: `.venv\Scripts\python.exe -m unittest discover -s tests -t tests` from `prototypes/gpu-tracker`.

Test clips (uploads, keys from `uploads/index.json`):

| clip | key | what it is | keyframes (settings.json) |
|---|---|---|---|
| broadcast | `2bac432528d0e568` | 1080p30 La Liga broadcast, crisp lines, pans and zooms | 10.0, 22.5, 32.3, 44.4 s |
| Veo | `f597d73d53f7b1fd` | 32 s wide-angle Veo recording of a youth match, faded lines, second pitch behind | 13.3, 18.8 s |
| amateur 3 | `6dfdc88ec6f2d100` | 295 s low side camera, turf, pans about 2 000 px and zooms 0.33x-1.56x | 0.0, 22.3 s |

Regression windows: broadcast 0.5-180 s, Veo 1-31 s, amateur 3 0-60 s, amateur 3 150-210 s (pan and zoom; no keyframe there yet, so
its calibration is unmeasured and only tracking is compared). Timing numbers are comparable only between runs that had the GPU to
themselves; runs made while another batch was processing are marked.

Ground truth available: your calibration keyframes (held out one at a time), your corrections (`runs/<key>/corrections.json`),
exact synthetic renders in the unit tests. Not available (reported as unmeasured): box-level tracking ground truth on the real clips
(HOTA / IDF1), real off-screen positions, calibration error away from the keyframes, detection precision.

## Phase 0 — harness, persistence, timing (no behaviour change) — commit `37b4eb5`

Added `registration.json` per run (motion log, anchors with their root keyframe, keyframes), `calibration.H` per frame,
`summary.timing`, the offline passes' events in `events.log`, and the three tools. Behaviour check: the Phase 0 runs of the amateur
and Veo windows agree 100% (boxes, labels, positions) with your own runs of 2026-10-12 (`20261012-102459-40fe2e`, `20261012-104858-ba6788`);
the broadcast baseline differs slightly from your run `20261012-122959-81676a` (label agreement 0.998 under the id mapping, identified
share 0.942 vs 0.934) because your review server had loaded the code of the previous day's identity work before the referee change
`a741be8` was committed, which only shows on the broadcast (its pink officials).

Baselines (Phase 0 code, scratch runs `p0/base/*`; the broadcast run shared the GPU with another batch, so its speed is not representative):

| window | tracked/frame | drawn/frame | anonymous with position | identified share | id switches | new identities | over cap | pitch conf | calibrated share | confident share | anchors | disagreement median m | ball tracked | elapsed s | fps | peak GPU GB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| broadcast | 20.87 | 17.84 | 0.46 | 0.942 | 12 | 26 | B 13 | 0.788 | 0.877 | 0.769 | landmarks 4, paint 92 | 0.27 | 0.673 | 1508 (shared GPU) | 3.58 | 0.45 |
| veo | 12.43 | 6.12 | 6.15 | 0.396 | 11 | 18 | - | 0.482 | 1.0 | 0.561 | landmarks 2, paint 8 | 0.21 | 0.483 | 190 | 4.72 | 0.49 |
| amateur3_0 | 17.17 | 13.38 | 3.75 | 0.696 | 18 | 23 | A 12 | 0.585 | 1.0 | 0.732 | landmarks 2, paint 55 | 0.23 | 0.241 | 427 | 4.22 | 0.36 |
| amateur3_150 | 16.66 | 0.0 | 0.0 | 0.688 | 43 | 23 | B 12 | 0.247 | 0.0 | 0.0 | - | - | 0.196 | 450 | 4.0 | 0.36 |

"tracked" = people in states confirmed or uncertain; "drawn" = identities of the final roster with a valid pitch position (what the
2D view shows); "anonymous with position" = tracked people with a team and a position but no identity (hidden by the 2D view today).

Hold-out calibration error of the baselines (`tools/holdout.py`; each keyframe rebuilt from the other anchors only):

| window | keyframe | px mean / max | metres mean / max | rebuilt from |
|---|---|---|---|---|
| broadcast | 10.0 s | 23.8 / 43.3 | 1.19 / 3.66 | carried backward from the 22.5 s keyframe (12.5 s) |
| broadcast | 22.5 s | 11.2 / 17.8 | 0.89 / 2.34 | interpolated, nearest anchor 4.6 s away |
| broadcast | 32.3 s | 19.2 / 35.7 | 0.80 / 1.80 | interpolated, nearest anchor 2.4 s away |
| broadcast | 44.4 s | 19.8 / 100.6 | 1.18 / 3.41 | carried forward, nearest anchor 0.7 s away |
| veo | 13.3 s | 78.0 / 111.7 | 4.78 / 7.71 | carried backward from 18.8 s (5.5 s) |
| veo | 18.8 s | 264.8 / 611.2 | 11.07 / 22.19 | carried forward from 13.3 s (5.5 s) |
| amateur3_0 | 0.0 s | 66.1 / 87.9 | 7.23 / 17.07 | carried backward from 22.3 s (22.3 s) |
| amateur3_0 | 22.3 s | 100.7 / 154.2 | 18.46 / 59.71 | carried forward from 0.0 s via 55 weak paint anchors (confidence reported 0.565) |

Pan stability (ground jump beyond the measured camera motion) was already small on every window (median 0 px, p95 under 0.2 px);
the baselines' paint anchors moved the view by up to 51 px (amateur 3) and 110 px (Veo) at once, which is the drift they were
"confirming".

## Phase 1 — calibration fixes for low-angle footage — commit `c2dbafa` (+ the anchor-fade removal in the Phase 3 commit)

What changed (`soccer/camera.py`, `pitch_model.py`, `calibration.py`, `registration.py`, `tracker.py`, `pipeline.py`, new `soccer/lens.py`):
direct re-registration of the pitch model to remembered keyframe views with relay views across pans; painted-line refinements anchor
the offline registration only with evidence of their own (observability >= 0.15); keyframes up to 15 s outside the segment are carried
in (pre-roll / post-roll); one clearly wrong click among six or more landmarks is left out; landmarks disagreeing by up to 4 m still
anchor at lower observability; every position carries `pitchSigma`; no position below a registration confidence of 0.2; feet above the
horizon are `horizon`; the forward/backward disagreement is measured over the visible pitch; the lens coefficient is measured;
uncovered sections are grouped per camera movement and name the missing landmark family. Tests: 181 (from 168).

Runs: `p0/p1/*` (same four windows, Phase 1 code at `c2dbafa`, GPU shared with nothing else for the amateur and Veo windows).

| window | drawn/frame base → P1 | identified share | id switches | label agreement (mapped) | position distance base↔P1 | pitch conf | confident share | paint anchors | re-registrations adopted / relays | fps |
|---|---|---|---|---|---|---|---|---|---|---|
| broadcast | 17.84 → 16.52 | 0.942 → 0.942 | 12 → 12 | 1.000 | 0.13 m mean | 0.788 → 0.806 (HIGH 94% → 97%) | 0.769 → 0.402 (see note) | 92 → 92 | 9 / 7 | 3.58 (shared) → 4.17 |
| veo | 6.12 → 6.12 | 0.396 → 0.398 | 11 → 11 | 1.000 | 1.40 m mean | 0.482 → 0.487 | 0.561 → 0.423 | 8 → 0 | 6 / 1 | 4.72 → 4.58 |
| amateur3_0 | 13.38 → 13.01 | 0.696 → 0.677 | 18 → 22 | 0.933 | 2.16 m mean | 0.585 → 0.605 | 0.732 → 0.831 | 55 → 8 | 59 / 6 | 4.22 → 4.18 |
| amateur3_150 | 0 → 0 (no keyframe) | 0.688 → 0.688 | 43 → 43 | 1.000 | - | 0.247 → 0.247 | - | - | 0 / 0 | 4.0 → 4.31 |

Hold-out calibration error (each keyframe rebuilt from the other anchors only), baseline → Phase 1:

| window | keyframe | px mean / max | metres mean / max |
|---|---|---|---|
| broadcast | 10.0 s | 23.8 / 43.3 → 23.8 / 43.3 | 1.19 / 3.66 → 1.19 / 3.66 |
| broadcast | 22.5 s | 11.2 / 17.8 → 6.9 / 13.0 | 0.89 / 2.34 → 0.53 / 1.15 |
| broadcast | 32.3 s | 19.2 / 35.7 → 7.1 / 9.9 | 0.80 / 1.80 → 0.75 / 1.52 |
| broadcast | 44.4 s | 19.8 / 100.6 → 20.7 / 104.1 | 1.18 / 3.41 → 1.21 / 3.51 |
| veo | 13.3 s | 78.0 / 111.7 → 78.0 / 111.7 | 4.78 / 7.71 → 4.78 / 7.71 |
| veo | 18.8 s | 264.8 / 611.2 → 265.5 / 649.3 | 11.07 / 22.19 → 12.09 / 26.47 |
| amateur3_0 | 0.0 s | 66.1 / 87.9 → 66.1 / 87.9 | 7.23 / 17.07 → 7.23 / 17.07 |
| amateur3_0 | 22.3 s | 100.7 / 154.2 → 48.6 / 133.1 | 18.46 / 59.71 → 4.66 / 14.29 |

Reading: the broadcast tracking is byte-for-byte preserved (labels 100%, same switches, same ball); the broadcast's own keyframes are
reached more accurately (22.5 s and 32.3 s) and its pitch is HIGH on 97% of frames. On the amateur clip the carry from the 0 s keyframe
to the one at 22.3 s improved fourfold in metres through the relay views; the carry in the other direction is unchanged (relays exist
forward in time only) and the Veo lens (k 0.58, sign agreement 1.0 over 46 frames) still defeats homography registration (6 of 17
attempts adopted). The identity labels of the amateur window changed on 7% of matched people because the online pitch geometry feeds the
re-identification's spatial evidence; the identified share moved from 0.696 to 0.677 and the far-tile pass stopped firing in automatic
mode (fixed in Phase 3: the trigger is now the expected size, whatever the mode). The broadcast "drawn per frame" fell by 1.3 because
positions are no longer written below a registration confidence of 0.2 (4 040 person-frames of the lost stretch at 146-157 s) and
because the first version of Phase 1 faded a paint anchor's certainty with time, which halved the broadcast's confident share (0.77 →
0.40) although its hold-out errors are about a metre: that fade was removed (an anchor that passes the evidence gate keeps the chain's
certainty); the confidence numbers with the fix appear under the combined run of the later phases. Measured and rejected: RANSAC at 2 m
dropped good clicks on every real keyframe (replaced by a clear-outlier rule), and a Jacobian condition number for point-only keyframes
was uninformative (column-normalised condition numbers 55-175 for all sets, degenerate sets already refused).

Unmeasured: calibration error away from the keyframes; whether the lens coefficient, applied, would help (not applied).

## Phase 2 — identity coverage, roster and visibility — commit `19a402f` (pipeline wiring in the Phase 3 commit)

What changed (`soccer/tracklets.py`, new `soccer/roster.py` and `soccer/visibility.py`, `pipeline.py`, `corrections.py`, `tactical.js`,
`index.html`, `tools/compare.py`): the offline passes now run the pitch registration first, so the tracklet refinement scores a fragment
against a dormant identity in metres on the pitch (sigma 3 m + 7 m per second of gap, capped at 2 s) wherever both carry a registered
position at confidence >= 0.2, and in camera-compensated image distance as before elsewhere. After the refinement a global assignment
(`assign_fragments`) matches the fragments still anonymous (>= 0.4 s in state uncertain) to the team's identities not observed during
the fragment, one-to-one per round (Hungarian), accepting a pair only at score >= 0.6, with appearance >= 0.5 where crops exist, motion
alone only for gaps <= 3 s, and a lead of 0.1 over the fragment's next-best identity and over the identity's next-best fragment; it never
creates an identity and never puts two fragments that overlap in time on one identity. Identities that may be one person (same team,
never on the same frame, one observed entirely before the other, galleries mutually the most similar at >= 0.7, a plausible move between
the last sighting of one and the first of the other) are listed as `roster.mergeProposals` and logged, never applied. `roster.build`
derives the active roster from the result (one slot per identity with status ACTIVE / OFF_SCREEN / UNKNOWN / LEFT / RETIRED, the
expected team sizes, `overCap` reported rather than trimmed, the anonymous observations); `visibility.annotate` classifies every
unobserved slot on every frame from the camera's view polygon (the frame cut at the horizon, mapped to the pitch) as C (expected in
frame, not seen), S (off-screen) or U (unknown); a correction choice "Left the match" marks a substitution or sending-off; the 2D view
draws tracked-but-unidentified people as team-coloured `?` markers (observations, never identities) and lists the slots not on the board
with their visibility; a correction re-derives roster and visibility. The online identity manager is unchanged. `tools/compare.py` now
also reports the agreement among the people labelled in both runs and the labels only one run has, since a label added offline is not a
disagreement about who someone is. Tests: +16 (roster, visibility, assignment exclusivity, identical-looking teammates stay anonymous,
a fragment overlapping the identity is never assigned, merge proposals, the correction, the comparison tool).

Runs: `p0/p2/*`, the same four windows with the Phase 2 code (batch started 18:12 on 2026-10-12, before the anchor-fade removal of
Phase 1, which concerns only the broadcast's confident share and is measured under Phase 3; the batch shared the CPU with the Metrica
benchmark, so its timing is marked "shared"). Baseline `p0/base/*`, Phase 1 `p0/p1/*`. The online detections and tracks of P1 and P2 are
identical on every window (box agreement 1.000, nobody only in one run), so every difference below is the offline passes.

| window | identified with a position per frame, base → P1 → P2 | anonymous with a position per frame (now drawn as `?`) | assigned offline: fragments / frames | roster at the end | labels vs P1: agreement where both labelled (added / removed) | online track switches base → P2 | new identities base → P2 | fps |
|---|---|---|---|---|---|---|---|---|
| broadcast | 17.84 → 16.52 → 16.52 | 0.46 → 0.38 | 1 / 726 | A 11, B 13 (over the cap by two, reported), 2 goalkeepers | 1.000 (726 / 0) | 1 → 1 | 26 → 26 | 3.58 (shared) → 2.13 (shared) |
| veo | 6.12 → 6.12 → 6.20 | 6.15 → 6.16 | 1 / 306 | A 10, B 8, no goalkeeper | 1.000 (679 / 604) | 3 → 3 | 18 → 18 | 4.72 → 3.99 (shared) |
| amateur3_0 | 13.38 → 13.01 → 13.31 | 3.75 → 3.81 | 4 / 2 930 | A 11, B 11, 2 goalkeepers (the baseline had 12 in team A) | 1.000 (2 935 / 2 386) | 0 → 0 | 23 → 22 | 4.22 → 3.88 (shared) |
| amateur3_150 | no positions (no keyframe); 16.66 tracked | 0 | 2 / 166 | A 11, B 12 (over the cap by one, reported) | 1.000 (166 / 0) | 3 → 3 | 23 → 23 | 4.00 → 4.23 |

Broadcast in detail: among the 108 875 people labelled in both the Phase 1 and the Phase 2 run every label agrees; 726 person-frames of
one fragment gained an identity (identified share 0.942 → 0.949) and none lost one; label agreement with the baseline is 0.995 overall,
the 0.5 % being those added labels. Online track switches 1 → 1, new identities 26 → 26, ball tracked 0.673 → 0.673, hold-out keyframe
errors identical to Phase 1, count-recall 1.0 on the six annotated frames. Your correction (B-08 became the referee REF-4 at 10 s) is
respected: that person is track 40 of this run, an ignored person on all 61 frames around 10 s, and no offline pass put a player identity
on it (`p0/check_correction.py`). The one broadcast number below the baseline is identified people with a position per frame, 17.84 →
16.52, unchanged since Phase 1 and caused there: positions are written only at a registration confidence of 0.2, and the broadcast's
last 34 s (146.2-180.5 s) have no keyframe and a paint-carried registration below that (the baseline was below 0.2 from 151.8 s and
between 0.15 and 0.2 from 146.2 s, and wrote positions there down to confidence 0.05). Those slots are now unknown instead of placed by a
registration nothing confirms; a keyframe around 150 s would restore them. The confident share (0.402 with the Phase 1 fade) is measured
with the fade removed under Phase 3.

Reading: the board gain the plan expected from drawing the anonymous observations is real, but it is a page change that applies to every
result, old or new: the Veo board goes from 6 to about 12 people per frame and the amateur board from 13 to 17, the `?` markers saying
that these are observations without an identity. The offline assignment itself put identities on 306 Veo and 2 930 amateur person-frames
(4 % and 10 % of the labelled people) without creating any; 18 of the 22 assignments proposed on the amateur window were refused for
lack of a lead, which is the intended behaviour when teammates look alike. The pitch-aware refinement is stricter than the image-based one
where positions carry metres of registration error (on the amateur window it reconnected 5 frames where the image rule had reconnected
2 386), and the global assignment then takes most of those fragments under its lead rule, so the labels added and removed between P1 and
P2 largely cover the same people with a stricter decision. Team A of the amateur window is no longer over the cap; the other over-cap
cases (broadcast team B by two, amateur 150-210 s team B by one) are reported, not trimmed. No frame draws one identity twice
(`tactical.duplicates` 0 everywhere). The identity-level "id switches" count rises with every reconnection or assignment by construction
(it counts local tracks per identity), so the honest switch metric is the online one, which the offline passes cannot change.
Tracked-but-undrawn people per frame after Phase 2: Veo 0.17, amateur 0-60 s 0.01, broadcast 3.97, of which 3.8 are the uncalibrated
last 34 s.

Acceptance: drawn (identified + anonymous) per frame Veo 12.4 (>= 10), amateur 17.1 (>= 16) — met; one identity drawn twice: 0 — met;
offline labels contradicting a correction on the broadcast: 0 — met; online switches and new identities not higher on any clip — met;
broadcast label agreement where both runs label a person 1.000, overall 0.995 — met; broadcast identified-with-position per frame
>= baseline — not met (16.52 vs 17.84, the Phase 1 confidence floor on the uncalibrated last 34 s, above).

Unmeasured: whether an assignment is correct (no box-level ground truth on the real clips); the visibility classes on the real clips
(checked on synthetic renders in `tests/test_visibility.py` only).

## Phase 3 — adaptive detection and tracking of small, distant players — commit `9f52ddf`

What changed (`pipeline.py`, `soccer/tracker.py`, `soccer/relevance.py`, `soccer/local.py`; the same commit wires Phases 2 and 4 into
the pipeline and removes the Phase 1 anchor fade in `soccer/pitch_model.py`): every addition is tied to a measured condition that
broadcast footage does not meet, and every trigger is counted in `summary.detection`. (1) The far-tile pass is triggered by the expected
size of a person at the far side of the pitch (under 30 px) whatever the pitch mode, instead of only in robust mode (Phase 1's
calibration had switched the amateur clip to automatic mode and silently stopped the tiles). (2) Tiles are merged with a cross-tile NMS
(IoU 0.5) before the three-frame confirmation, so a person on a tile border no longer counts twice. (3) A confirmed far detection is
handed to BoT-SORT with a tracking score of 0.45 (its own score from the upscaled tile is below the 0.40 new-track threshold), so it can
start a local track; the person keeps the true score. (4) Track-guided re-detection: for a confirmed identity whose local track was lost
less than a second ago with a box under 60 px, the detector runs on a 3x crop (>= 256 px) around the Kalman prediction, at most four
crops per frame in one batch, and a detection is accepted only at IoU >= 0.3 with the prediction; it continues a track and never creates
an identity. (5) Far-boundary continuation: a detection whose feet are at most 5 % of the frame height beyond the far edge of the field
and that continues a live box (IoU > 0.3) is kept as a continuation instead of being rejected as audience, only when the far edge is not
a painted line (the broadcast's far touchline is a painted line, so it never fires there). (6) BoT-SORT small-object tuning (second-stage
threshold 0.10 -> 0.05 and buffer 60 -> 90 frames while the far band exists) is implemented in `local.small_objects` behind
`SMALL_OBJECT_TRACKING`; measured after this commit on the Veo and amateur windows with the tuning on (`p0/p4b`); the outcome is recorded in the note at the end of this phase. Not implemented, by measurement: the team-unknown re-identification relaxation (its gate
was deferred decisions with unknown team >= 1 per frame after Phase 2; measured 0.17 per frame on the Veo clip and 0.01 on amateur 3:
the 2-3 team-unknown people per frame there are candidates in their first second of tracking, median 3-5 frames long, not deferred
decisions). Detector/CPU pipelining (optional) was not done. Tests: +6 (a confirmed far detection starts a track, a crop detection is
accepted only next to its prediction and never next to another person, no crop window next to a tracked neighbour, far-side
continuation only without a painted far edge, the far band triggers by size in every mode).

Measured and fixed before the numbers below: the first run of this code (`p0/p3/*`) kept the broadcast's boxes (box agreement 0.9999
with Phase 2) but changed the labels of 1.7 % of the people labelled in both runs, all of them two players: a crop re-detection bridged a
crossing at 35.5 s and another at 124.5 s, so the identity manager never registered those close encounters (4 instead of 6) and never
ran its swap check there (2 swap corrections instead of 4), and two team-A identities were carried onto team-B players for 25 s and
56 s; your own run of this clip calls the first of them B-05 on 702 of the 735 frames. The crop pass is therefore not made for a player
lost next to another tracked person, and a crop detection overlapping another detected person is not accepted (`cropSkippedCrowded`,
`cropRejectedCrowded`); a player lost behind someone is left to the encounter handling, which measures who comes out of it. The runs
below are the re-run with that guard.

Runs: `p0/p4/*`, the four windows with the final code of Phases 1-5, compared with the Phase 2 runs (`p0/p2/*`) and the baseline
(`p0/base/*`). Timing caveat: another GPU application (a game, cs2.exe, 50-90 % of the GPU) was running on this machine during the
Phase 2, 3 and 4 batches, so their fps and per-stage milliseconds are not comparable with the baseline or Phase 1, which had the GPU
alone; the whole-frame detector's own cost rose 1.3-1.8x on every window for that reason. The cost of the new passes themselves is
`localCropsMsPerFrame` and `farTilesMsPerFrame`.

| window | tracked per frame, base → P2 → final | identified with a position per frame (anonymous `?` drawn) | people under 30 px per frame: tracked / identified, base → final | far tiles: passes / detections / tracks started | far-boundary continuations | crops: passes / accepted / skipped (a neighbour) | online track switches base → final | new identities base → final | labels vs P2: agreement where both labelled (added / removed) | count-recall base → final | fps, GPU shared |
|---|---|---|---|---|---|---|---|---|---|---|---|
| broadcast | 20.87 → 20.87 → 20.87 | 17.84 → 16.52 → 17.14 (0.41) | 0.22 / 0.09 → 0.22 / 0.09 | 3 / 0 / 0 | 0 | 69 / 1 / 383 | 1 → 1 | 26 → 26 | 1.000 (0 / 0) | 1.0 → 1.0 | 3.58 → 2.13 → 3.83 |
| veo | 12.43 → 12.53 → 12.54 | 6.12 → 6.20 → 6.21 (6.16) | 1.75 / 0.19 → 1.75 / 0.40 | 189 / 38 / 1 | 70 | 68 / 4 / 642 | 3 → 3 | 18 → 18 | 1.000 (3 / 0) | 0.942 → 0.966 | 4.72 → 3.99 → 3.23 |
| amateur3_0 | 17.17 → 17.13 → 17.26 | 13.38 → 13.31 → 13.32 (3.90) | 0.92 / 0.35 → 0.97 / 0.33 | 8 / 0 / 0 | 138 | 61 / 0 / 885 | 0 → 0 | 23 → 22 | 1.000 (0 / 1) | 1.0 → 1.0 | 4.22 → 3.88 → 3.28 |
| amateur3_150 | 16.66 → 16.66 → 16.64 | no positions (no keyframe); identified share 0.688 → 0.693 → 0.695 | 6.15 / 3.54 → 6.15 / 3.58 | 1 227 / 149 / 88 | 66 | 179 / 3 / 1 039 | 3 → 2 | 23 → 23 | 0.912 (1 985 / 1 895) | 1.0 → 1.0 | 4.00 → 4.23 → 2.71 |

Reading: with the crop guard, the final code changes nothing in the Veo and amateur 0-60 s identities (labels identical to Phase 2
wherever both runs label a person; 3 and 0 labels added) and tracks 0.01-0.13 more people per frame, nearly all of them continuations
at the far edge of the field (70 / 138 / 66 person-frames). The far-tile pass fires where it was meant to: on the zoomed-out 150-210 s
window it ran on 1 227 of 1 800 frames, confirmed 149 detections and started 88 local tracks, and the identified share of 30-40 px
people rose from 0.664 to 0.709 (people under 30 px: 0.866 → 0.876); on the 0-60 s window and the broadcast the far band hardly exists
(8 and 3 passes). The crop re-detection is nearly inert once it keeps away from neighbours (4 / 0 / 3 accepted detections on the
Veo and amateur windows; 642-1 039 lost cases skipped because another tracked person was within half a body): on these clips a small
player is lost next to someone far more often than alone. Count-recall on the hand-counted frames is unchanged (0.966, 1.0, 1.0, with
over-counts rather than misses) and the number of identified small people per frame moves by at most 0.04. The 150-210 s window, with
its 88 far-tile track starts, re-labels 9 % of the people labelled in both runs, which no ground truth can judge; its online switches
fell from 3 to 2 and its identities stayed at 23. On the broadcast the final run is label-for-label identical to the Phase 2 run (109 601 people labelled in both, agreement 1.000, no label added or removed, boxes identical) and differs from the baseline only by the 726 labels the Phase 2 assignment added (agreement 0.9947 overall, 1.000 where both label a person); online switches 1, new identities 26, ball tracked 0.673, hold-out errors, count-recall and the respect of your correction unchanged. With the anchor fade removed the confident share is back at the baseline's 0.769 (HIGH 0.939, calibrated 0.877), and identified people with a position are 17.14 per frame against the baseline's 17.84: the remaining gap is the positions the baseline wrote on uncalibrated frames (Phase 1). Crops ran on 69 frames (1.3 %), accepted one detection and skipped 383 lost cases next to a neighbour; the far band existed on 3 frames and started no track.

Acceptance: count-recall up on small-player frames — not measurable as a gain, it was already 0.94-1.0 and stays so; over-count not
up — met (87 → 88, 40 → 39, 102 → 102 tracked on the counted frames); `teamsOverCap` and new identities not up — met; broadcast crop
passes <= 1 % of frames — passes on 1.3 % of frames (69), accepted detections on 0.02 % (1): the target is met for what enters the result, slightly missed for the passes; far-tile track starts on the broadcast = 0 — met (0); broadcast label agreement >= 99 % —
met (1.000 against Phase 2 and against the baseline where both label a person; 0.9947 overall against the baseline, the difference being labels added offline); fps >= 4.7 — not measurable here (GPU shared with another application; reported as measured).

Small-object tracker tuning, measured after the Phase 3 commit (`p0/p4b/*`: the Veo and the two amateur windows with
`SMALL_OBJECT_TRACKING` on, i.e. a second-stage threshold of 0.05 and a 90-frame buffer while the far band exists): on the Veo and the
amateur 0-60 s windows every box and label is identical to the final run (the tuning was active on 189 and 8 frames and changed no
association); on the 150-210 s window, where it was active on 1 227 frames, it tracked 0.33 more people per frame and raised the
identified share from 0.695 to 0.723, but it created one more identity (team B 12 → 13, over the cap by two), raised the identity
fragmentation from 1.83 to 2.54 (29 → 46 fragments reconnected offline) and relabelled 5 % of the co-labelled people, with online
switches unchanged at 2. The rule was to keep it only if fragmentation fell without more switches: it stays off (`False`), available
for a later measurement with ground truth.

## Phase 4 — off-screen estimation, benchmarked on full-pitch data — commit `278c716`

What changed (new `soccer/imputation.py`, `tools/metrica_bench.py`, `tests/test_imputation.py`; `pipeline.py` writes `timeline`,
`corrections.py` re-derives it after a correction): every roster slot gets, on every frame, a position with a kind and a sigma: O observed,
I interpolated (a straight line between two observations at most 1 s apart), P predicted, U unknown (no coordinates). Between two
observations further apart the straight line is kept as P, with the sigma of both sides plus 0.5·sqrt(gap) m. Where only one side is
observed (the player left the view and has not come back, or before the first sighting) the estimate is the last observed position moved
with the team: the visible teammates' displacement since that observation (each weighted by its uncertainties; at least 4 of them),
ramped in over 2 s so that the first seconds hold the last position, with a sigma growing at 0.8 m/s. An estimate whose sigma passes 15 m
or whose gap passes 25 s is U, and a never-observed slot is U throughout. Role-anchored centroid voting (an exponentially weighted offset
per player from the voted team centre, bootstrapped in three passes so that the visible-subset bias cancels) is implemented and benchmarked
but not used for the shipped estimate, because it lost (below). The visibility classification of Phase 2 now uses these estimates. Tests: 5
(voting recovers a centre the visible mean gets wrong by 5 m, no centre under 4 voters, sigma grows, the blend meets both observations, a
never-observed player is U).

Benchmark (`tools/metrica_bench.py`): the Metrica Sports sample games (25 Hz full-pitch positions of both teams, used at 5 fps, with
credit to Metrica Sports; the CSVs are not in the repository), a side camera simulated in pitch space: it follows the ball along the
length with a 1.5 s lag, sees 45 m of the length (also 30 m and 60 m, and a low-angle variant seeing only the far 70% of the width), and
pans away for 3 s every 25-40 s; the visible players get Gaussian noise of 0.5-2 m. Every hidden sample is scored against the true
position by the time since the last observation; coverage = the share of hidden samples a method estimates at all (so the "all" columns
compare different sample sets; the buckets give n); calibration = p68 of |error|/sigma (1 is perfect, below 1 conservative). Game 1 chose
the constants (ramp 2 s, sigma rates 0.8 and 1.8 m/s, the one-sided base), game 2 was held out: `p0/metrica_game1d.log`,
`p0/metrica_game2*.log`.

Held-out game 2, 45 m window, MAE in metres by time hidden (Home / Away team; the buckets hold 1.4-29 thousand samples each):

| method | coverage | 0-1 s | 1-3 s | 3-8 s | 8-20 s | > 20 s | all | calibration p68 |
|---|---|---|---|---|---|---|---|---|
| straight line between observations (offline) | .58 / .61 | 1.55 / 1.57 | 2.12 / 2.13 | 3.77 / 3.58 | 4.37 / 4.24 | 3.95 / 3.28 | 3.25 / 3.15 | .82 / .81 |
| last position held, both ways (offline) | .70 / .72 | 1.59 / 1.59 | 2.26 / 2.21 | 4.18 / 3.88 | 4.44 / 4.28 | 4.15 / 3.56 | 3.50 / 3.30 | .82 / .81 |
| **shipped: straight line, team shift where one-sided** | .73 / .76 | 1.59 / 1.60 | 2.27 / 2.25 | 4.26 / 4.08 | 5.31 / 5.54 | 5.71 / 5.71 | 3.99 / 4.00 | .89 / .90 |
| role-anchored centroid voting, both ways | .64 / .66 | 2.52 / 2.45 | 4.60 / 4.37 | 7.12 / 6.89 | 9.70 / 9.64 | 11.89 / 11.96 | 7.27 / 7.16 | 2.82 / 2.75 |
| last position held (one-sided) | .48 / .49 | 1.93 / 1.91 | 3.70 / 3.64 | 7.43 / 7.07 | 9.67 / 8.93 (n 1.4 k) | - | 5.38 / 5.15 | 1.10 / 1.10 |
| constant velocity (one-sided) | .48 / .49 | 2.51 / 2.48 | 4.70 / 4.63 | 8.47 / 8.22 | 10.86 / 10.22 (n 1.4 k) | - | 6.34 / 6.16 | 1.37 / 1.33 |
| team shift (one-sided) | .59 / .61 | 1.89 / 1.86 | 3.38 / 3.28 | 6.43 / 6.35 | 9.87 / 10.12 (n 13 k) | - | 5.71 / 5.72 | 1.25 / 1.23 |
| centroid voting (one-sided) | .53 / .55 | 2.52 / 2.45 | 4.97 / 4.76 | 8.45 / 8.26 | 11.88 / 11.79 | 14.28 / 13.88 | 7.62 / 7.53 | 2.28 / 2.29 |

The other views of the same held-out game (Home / Away):

| view | visible share | shipped: all MAE, p68 | straight line: all MAE | one-sided 3-8 s: team shift vs last held | voting both ways: all MAE |
|---|---|---|---|---|---|
| 60 m | .76 / .78 | 3.71 / 3.48, .85 / .80 | 3.00 / 2.70 | 6.22 / 6.21 vs 6.65 / 5.94 | 7.74 / 7.26 |
| 45 m | .65 / .66 | 3.99 / 4.00, .89 / .90 | 3.25 / 3.15 | 6.43 / 6.35 vs 7.43 / 7.07 | 7.27 / 7.16 |
| 30 m | .48 / .49 | 4.46 / 4.27, .96 / .92 | 3.67 / 3.65 | 7.29 / 7.01 vs 8.16 / 8.15 | 7.27 / 7.33 |
| low angle, far 70% of the width | .51 / .51 | 5.50 / 5.56, 1.11 / 1.10 | 4.45 / 4.34 | 7.12 / 6.84 vs 8.40 / 8.09 | 8.25 / 8.26 |

Reading: with observations on both sides the straight line is the best estimate tried, and any formation model only adds error. The
plan's expectation that role-anchored centroid voting beats the held position and constant velocity from 3 s on was not met: voting is
worse than simply holding the last position on every bucket, in every view, on the tuning game and on the held-out game, and its sigma is
overconfident (p68 2.3-3.3); a formation offset learned from minutes of partial views is too noisy against 0.5-2 m observation noise,
and the published figures for the method (3-4 m under 2 s, 15-17 m beyond 10 s) are no better than holding. The team-shift estimate,
the simplest use of the visible teammates, beats holding at 0-1, 1-3 and 3-8 s in 11 of the 12 view/team/bucket cells (the exception:
60 m view, Away, 3-8 s, 6.21 vs 5.94 m) and is the only one-sided estimate still available at 8-20 s (median 7.8-8.9 m), so it is the
shipped one-sided estimate. Bidirectional beats one-sided on every bucket, as expected. The shipped method's sigma is calibrated (p68
.80-1.11 across the views); the team-shift sigma alone is slightly overconfident in the narrow and low-angle views (1.27-1.31). Velocity
extrapolation hurts everywhere with noisy observations. Learned imputation models were not tried: no code or weights exist, and the
bar (15% better than the shipped estimate on 3-20 s) has nothing to be measured against.

Acceptance: "bidirectional beats one-sided on every bucket" met; "calibration ratio within .8-1.25" met for the shipped method; "voting
beats last-position and CV on every bucket >= 3 s" not met and reported; the real-clip checks (no P/I coordinates on uncalibrated frames,
no jump above 12 m/s at a reappearance, no slot with two ids) are under Phase 5, measured with `tools/compare.py` on the Phase 3 runs.

Unmeasured: the simulated window has no occlusion and no detector misses inside the view (on the real clips visibility comes from the
registration's view polygon and the tracker); only Metrica games 1-2 were used (DFL open data and SkillCorner visibility flags were not);
on the real clips the true off-screen positions are unknown, so only consistency is measured there.

## Phase 5 — tactical view kinds, full lineup, camera footprint, export — commit `7fe810d`

What changed (`tactical.js`, `index.html`, `tools/compare.py`, `tests/test_tactical.py`, and in `soccer/visibility.py` the off-screen
rule below): the 2D view draws every roster slot from the `timeline` when the result has one (older results are drawn from their
frames as before): an observed position as a solid marker, an interpolated one solid with a thin inner ring, a predicted one translucent
with a dashed outline, a prediction less certain than 8 m faded with a `~` and hidden unless "Full lineup (every estimate)" is on, an
unknown slot never drawn (only named in the roster strip with its visibility). Only observed samples are smoothed; estimates are drawn
as the timeline gives them, and because an estimate meets the observations a returning player never jumps. "Camera footprint" draws the
part of the pitch the camera sees on the current frame (`frames[i].view`); the debug overlay adds the sigma of an estimate; "Download
timeline CSV" exports, per slot and frame, position, kind, sigma along and across the pitch and visibility, built in the browser from
the result. `tools/compare.py` adds the consistency metrics: identities drawn twice on a frame, jumps above 12 m/s where a prediction
meets a measurement (consecutive samples) or over one second within a run of measurements (consecutive observations jitter by their own
noise, which says nothing about consistency), the kind shares and the footprint share. Visibility: on the amateur 0-60 s window none of
5 252 predicted positions lay further outside the camera's view than its own sigma, because a one-sided estimate holds a player near
where the view lost them, so a slot last seen leaving through a frame edge (the roster's exit edge) and not observed since is now `S`
instead of `U` on every later frame with a view. The page test's synthetic clip now carries a timeline, roster and visibility; a
browser-free test checks its kinds, sigmas and consistency, and the Playwright assertions (skipped here) follow the timeline: a player
the camera leaves is drawn as a prediction near the last observation, an unknown slot appears only in the strip.

Consistency on the final runs (`p0/p4/*`):

| window | timeline kinds O / I / P / U | visibility O / C / S / U | identities drawn twice | jumps > 12 m/s: prediction meets measurement / within measurements over 1 s | frames with a camera footprint |
|---|---|---|---|---|---|
| broadcast | .681 / .002 / .036 / .281 | .781 / .022 / .079 / .118 | 0 | 0 / 0 | 0.841 |
| veo | .345 / .049 / .309 / .297 | .345 / .149 / .144 / .363 | 0 | 0 / 1 | 0.909 |
| amateur3_0 | .558 / .033 / .128 / .281 | .555 / .131 / 0 / .314 | 0 | 0 / 20 | 1.000 |
| amateur3_150 | 0 / 0 / 0 / 1.000 (no calibration) | .578 / 0 / 0 / .422 | 0 | 0 / 0 | 0 |

Page check (read-only stub server with the final amateur result, built-in browser): the final amateur 0-60 s result (`p0/p4/amateur3_0`) at 30 s: 20 slots on the board, 17 observed (solid) and 3 predicted (translucent, dashed; sigmas 1.1-3.7 m, A-08, A-10, GK-2), 9 slots named in the roster strip as unknown and 5 anonymous `?` markers, the camera footprint drawn over the pitch; the CSV builder produced 41 400 rows (23 slots x 1 800 frames) with the header `time,id,team,x,y,kind,sigma_along_m,sigma_across_m,visibility` and kinds O 23 104 / I 1 366 / P 5 294 / U 11 636 (the browser download was intercepted for the check, nothing was saved)

Unmeasured: the Playwright page tests are skipped in this environment (Playwright is not installed); the CSV export was checked by
downloading it once and reading its header and row count, not by an analytics consumer.

## Phase 6 — evidence-gated extras: decisions, nothing applied without a measurement

(1) PnLCalib (GPL-2.0-only, so only ever as a separate process) was not trialled within this plan; the trial protocol stands: sampled
frames of the three clips scored with `tools/holdout.py` against the keyframes, integration as another `Anchor` source only if the
hold-out error drops on two clips. (2) Lens undistortion: the coefficient is measured and reported (`summary.pitch.lens`: Veo k ≈ 0.58
with sign agreement 1.0 over 46 frames; curvature 3.1 px per 1 000 px on the Veo clip, 2.2-2.7 on amateur 3, 0.9 on the
broadcast) but not applied: the gate is a lower hold-out error on two clips, and the clip where it could pay (Veo, 12 m at the 18.8 s
keyframe, dominated by the lens) is one; applying it means re-registering the keyframe clicks and the paint through the undistortion,
which is the next calibration step. (3) GTA-style merges: proposals are listed (`roster.mergeProposals`: 0 on all four windows: no two same-team identities were both never on the same frame and mutually most similar at 0.7, so nothing was proposed); automatic
application needs precision >= 0.9 against your corrections, and the three test videos hold one correction (a referee) and no merge, so
precision cannot be measured yet. (4) SAHI-style full tiling: count-recall on the annotated frames is 0.94-1.0 on all three clips with
over-counts rather than misses, so there is no measured miss for tiling to fix. (5) OSNet on CUDA and detector/CPU pipelining: not done;
the online cost per frame on the final runs is 253 on the broadcast and 297-364 on the Veo and amateur windows ms, of which the detector passes take 63 and 82-83 ms (with the GPU shared,
above). (6) SoccerTrack v2 with sn-trackeval: not done (registration required); it remains the only way to a real GS-HOTA number for the
amateur path.

## Short-occlusion re-identification — B-07 becoming a new identity behind an opponent (2026-10-14)

The failure (the user's clip, broadcast 40.8-42.8 s, run `runs/2bac432528d0e568/20261013-160036-fb1682`): B-07 on local
track 15 runs behind A-03 at 41.0 s, the local track ends, the player comes out 0.53 s later on local track 60 and is made a
new identity (B-11 in that run, B-12 in the baseline below) at 42.8 s with `RE-ID REJECTED ... spatially implausible (7%)`
although the appearance matched (0.725, OSNet cosine 0.854, final 0.582). The local tracker losing the track behind a
player is by design; the failure was in the global re-identification, and the offline pass had nothing to repair (a track
with a confident identity is never revisited).

Cause, replayed from the stored frames (`scratchpad` script, same numbers as the log): the pitch velocity of a track was
measured as the displacement over one evidence tick (0.2 s) divided by the interval since the last *frame* (1/30 s), so
every remembered pitch speed was six times too high at 30 fps; and the last two boxes before the occlusion were slivers
(height 0.033 and 0.026 instead of 0.044, IoU with A-03 only 0.04-0.11, so not "contaminated") whose foot point was 1-2 m
too far up the pitch. Remembered velocity (−5.7, −25.3) m/s, predicted position 15.3 m from where the player came out,
spatial 0.074; with the tick interval the same memory gives (−1.0, −4.2) m/s and 0.83.

What changed (`soccer/identity.py`, `soccer/tracker.py`, `soccer/tracklets.py`, `pipeline.py`, `index.html`, `README.md`):
the pitch speed is measured over the evidence tick and from clean boxes only; a partial box (contaminated, or next to someone
and shorter than 80% of the track's recent clean boxes) neither updates the track's velocity nor the identity's kinematic
memory, which keeps the last clean box and the time it was taken (`partial: true` on the frame; the reach of the remembered
position grows from that time); every sample carries the share of its box inside another person's box and above 0.5 the
crop is contaminated like an IoU above 0.15; the spatial plausibility takes the better of "kept moving at the remembered
speed" and "stopped", with the speed capped at 9 m/s; a new track that strongly resembles an identity lost less than 3 s ago
(appearance >= 0.65) is never ruled out on the predicted position alone and waits (`IDENTITY_UNCERTAIN`, "waiting for clean
frames") for up to 4 s of its own observation instead of becoming a new identity; and offline, a merge proposal that
describes a short occlusion (the later identity began within 3 s of the earlier one's last sighting, which was inside the
frame, continuity plausible and measured, neither identity set by you, the later one created in this run) is applied:
the later identity is retired into the earlier one, its frames relabelled (`evidence: merged offline`, `firstPassLabel`
kept), `MERGE APPLIED` logged, `summary.identity.mergedOffline` counted. Longer-gap proposals stay proposals.
Tests: `tests/test_identity.py::ShortOcclusionTests` (the tick-interval speed, the regression itself at 30 fps with a
sliver and a 0.5 s gap, the hold for a look-alike of a player lost moments ago), `tests/test_identity_encounters.py::
OfflineMergeTests` (applied for a 0.5 s gap, proposal only for a 6 s gap or after an exit through a frame edge); 211 tests
pass. Nothing is hard-coded to B-07 or B-11.

Measured on the four regression windows plus the clip's window (`base/*` = commit `7f24057` unmodified, `after/*` = this
change; the baseline batch shared the CPU with the unit-test suite, so timing is not compared):

| window | identified share | identities A / B / GK | team B over cap | anonymous frames | deferred decisions | re-ids | id switches | label agreement among people labelled in both runs | labels only after / only base | measured jumps > 12 m/s |
|---|---|---|---|---|---|---|---|---|---|---|
| clip (broadcast 10-50 s) | .973 → .973 | 10/12/2 → 10/11/2 | 1 → 0 | 0 → 0 | 3 → 1 | 1 → 2 | 1 → 2 | .990 | 0 / 0 | 0 → 0 |
| broadcast 0.5-180 s | .949 → .975 | 11/13/2 → 11/12/2 | 2 → 1 | 2 853 → 34 | 35 → 6 | 8 → 13 | 13 → 14 | .983 | 2 880 / 0 | 0 → 0 |
| veo 1-31 s | .404 → .404 | 10/8/0 same | 0 | 5 540 same | 92 same | 4 same | 14 same | 1.000 | 0 / 0 | 1 → 1 |
| amateur3 0-60 s | .690 → .682 | 11/11/2 → 11/12/2 | 0 → 1 | 7 071 → 7 592 | 122 → 109 | 5 → 8 | 23 → 24 | .947 | 1 871 / 2 121 | 20 → 36 |
| amateur3 150-210 s | .695 → .773 | 11/12/0 same | 1 same | 5 672 → 3 665 | 105 → 102 | 13 same | 42 → 59 | .975 | 3 542 / 794 | 0 → 0 |

Clip: track 60 is re-identified as B-07 at 42.8 s (spatial 0.997, appearance 0.725, final 0.868); B-07 is carried by
tracks 15 (10.0-41.0 s) and 60 (41.53-49.97 s), the 2D timeline is observed on both sides and interpolated across the
0.5 s gap (no jump), no team B identity is created after 40 s, no frame shows one identity on two people, team B has 11
identities. The offline merge was not needed on any window (0 applied; 0 proposals on all five).
Broadcast: the 1.7% disagreement among people labelled in both runs is the baseline's error, checked on the frames: from
the encounter at 86.5 s the baseline carried A-02 on the garnet (team B) player of local track 7 until the kit vote
released it at 124.5 s, then made that player a duplicate B-15, and left track 156 (the real B-08, then the real A-02)
anonymous for 33 s; this change re-identifies B-08 at 86.5 s (spatial 0.979 instead of a velocity-poisoned score) and the
crossing guard swaps the two identities back at 119.3 s. Every other label the change adds on the broadcast agrees with
the baseline. Veo is identical. Amateur 0-60 s: B-04 is re-identified at 10.8 s (appearance 0.748), so the anonymous
track 108 that the baseline deferred against B-04 for 4 s and dropped has no missing candidate left and becomes B-10
(team B then over the cap by one, as the roster reports); B-02 and B-08 are swapped between tracks 3 and 129 by the
crossing guard at 16.6 s where the baseline kept B-02 on a local track that itself jumped 17 m in a second at 17.3 s (both
runs show a jump there; which person is B-02 afterwards is unmeasured). Amateur 150-210 s (uncalibrated, image spatial
only): the track that appears 0.5 s after A-02 vanished, in A-02's place, with no other missing team A candidate, now
reaches the 0.72 threshold (appearance 0.406, spatial 0.994) where the baseline deferred it for 40 s; the later
A-05/A-08 difference cascades from that state and is unmeasured.

Decisions: the hold (SHORT_LOSS 3 s, STRONG_LOOK 0.65, SHORT_HOLD 4 s) and the merge gap (3 s) are first values, chosen so
that the professional footage is at least as good; they were not tuned on the amateur windows. Unmeasured: ground truth
for the amateur identity differences; timing (the baseline batch was polluted by the test suite); the page tooltip for
`merged offline` and `partial` (Playwright is not installed).

## False ball movement: the final path left a hidden ball for a ball boy's ball and came back (2026-10-15)

The failure (the user's clip `ball_tracking_wrong.mp4`, broadcast 22.1-24.3 s, run `runs/2bac432528d0e568/20261015-001926-88577e`):
the ball is last seen at 22.1 s at a player's feet (track 8); the causal pass keeps it OCCLUDED at that player and reacquires it
there at 24.27 s, 78 px from its prediction. The final path instead bridged 35 frames as a 12 m/s unseen flight down to a ball
standing still by the ball boy on the bottom touchline (detector 0.13-0.43, rejected by the causal pass as JUMP / LOW_SCORE), sat
on it for a second with `BALL LOCKED (final path)`, and started a new run on the real ball at 24.30 s: an excursion of 13 m and
back. Nothing in the causal tracker caused it; `ball_cleanup.py` (re-decoding a stored result, no detector) reproduces it exactly.

Cause, from a per-frame dump of the decoded path (scratchpad `path_dump.py`): a bridged frame costs .8 against 1.0 for NOT
VISIBLE, so a link earns .2 for every frame it bridges whatever its motion; a run may start on anything for 3.2; node
ball-likeness depends on size, shape, isolation and field (up to .6) far more than on detector confidence, so a resting ball
by the touchline scores .6-.9 and 29 such frames earn about 20 over leaving them unknown; and once a dead-stop penalty existed,
the decoder routed the flight through a single STATIC-rejected candidate at the bottom edge next to a person (a ball boy counted
as a player within reach) so that both legs looked like rolls. The stationary hypothesis (the ball stayed at the player for 2.2 s)
could not even be expressed: links bridged at most 1.2 s.

What changed (`soccer/ball_path.py`, `soccer/ball.py`, `pipeline.py`, `index.html`, `README.md`, tests in `tests/test_ball_path.py`
and `tests/test_ball.py`; 219 tests pass):

- A link whose implied speed is under 1 m/s may bridge up to 3 s (MAX_STILL_S): a ball seen again where it vanished most likely
  stayed there. Same price per bridged frame as any bridge (a cheaper one let static junk beat the real ball; measured).
- One ball: a candidate the causal pass saw while it confidently tracked the ball elsewhere keeps a fifth of its ball-likeness
  (ELSEWHERE .5 -> .2), so it never pays for a frame by itself.
- Physics the decoder had no word for: a flight (8 m/s or more) into or out of a sighting that stands still with no player
  within reach costs 12 (DEAD_STOP), and a link faster than a hard shot between sightings of two different things costs 12
  (JUMP_COST); both exceed what any bridge can earn. Node speeds come from tracklet neighbours within 0.3 s.
- People rejected as outside the pitch (ball boys, staff) are not players for the lower-body and within-reach logic.
- Trajectory review, up to 8 rounds: a still stretch of the path (0.5 m, 4+ sightings) that the causal pass ruled out (40% or more
  with JUMP, NO_ENTRY, OTHER_OBJECT, STATIC, FIELD_LINE or PLAYER_ATTACHED) and that nothing was seen rolling to from the last
  accepted sighting (a flight, 0.5 s or more unseen, or nothing; rejected waypoints are walked over) is cut out, logged as
  `BALL TRAJECTORY REVIEW` (frames, last accepted sighting, visual evidence, motion, the causal reasons, the decision
  `REJECTED_STATIC_UNANCHORED`), and the path is decoded again. UNCONFIRMED, LOW_SCORE and TRAJECTORY do not count: the real ball
  gets them while the causal pass re-acquires it (counting them cut 40-98 real stretches on the broadcast; measured and reverted).
- Every ball frame says where its position came from (`origin`: OBSERVATION, WEAK_DETECTION, INTERPOLATION, PREDICTION, UNKNOWN)
  and how the ball moved (`motion`: STATIONARY, ROLLING, FAST, OCCLUDED, UNKNOWN). The causal tracker reports STATIONARY only from
  sightings of the last 0.4 s that show the ball at rest and then carries no velocity into the next prediction; an unseen ball is
  OCCLUDED, never assumed still. `summary.ball` adds `staticRunsDropped`, `longUnseenLinks`, `stationaryLinks`.

Measured by re-decoding the stored results of the user's run and of the Phase-5-era regression windows (`after/*` of the 2026-10-14
session; scratchpad `ball_ablation.py`, one row per change added in order; "old decoder" is the committed code). There is no ball
ground truth, so the numbers are consistency metrics: shares of frames TRACKED / MISSING (bridged) / UNKNOWN, runs, teleports
(> 40 m/s between shown frames), out-and-back excursions (5 m away and back within seconds), bridged frames implying more than
8 m/s, and the distance of the shown ball from the first pass's prediction on frames where the first pass had the ball hidden at
a player (mean / max, metres).

| window | variant | tracked / missing / unknown | runs | teleports | excursions | bridged > 8 m/s | hidden drift mean / max m | review cuts |
|---|---|---|---|---|---|---|---|---|
| clip (10-50 s) | old decoder | .730 / .268 / .002 | 3 | 0 | 2 | 62 | 2.88 / 13.47 | 0 (visits the ball boy's ball, furthest 13.2 m) |
| clip | + stationary links | .730 / .268 / .002 | 3 | 0 | 2 | 62 | 2.88 / 13.47 | 0 (visits, 13.2 m) |
| clip | + one ball | .710 / .288 / .002 | 3 | 0 | 2 | 82 | 2.86 / 13.47 | 0 (visits, 13.2 m) |
| clip | + dead stop, jump | .711 / .287 / .003 | 4 | 0 | 2 | 52 | 3.35 / 14.27 | 0 (visits, 14.6 m) |
| clip | + review (shipped) | .672 / .327 / .001 | 2 | 0 | 2 | 49 | 1.54 / 7.75 | 4 (never visits, furthest 2.7 m) |
| broadcast 0.5-180 s | old decoder | .672 / .326 / .001 | 9 | 0 | 13 | 321 | 17.9 / 59.9 | 0 |
| broadcast | + stationary links | .668 / .331 / .001 | 8 | 0 | 12 | 245 | 18.2 / 59.9 | 0 |
| broadcast | + one ball | .607 / .392 / .001 | 6 | 0 | 12 | 253 | 17.4 / 59.9 | 0 |
| broadcast | + dead stop, jump | .604 / .395 / .001 | 6 | 0 | 12 | 229 | 17.4 / 59.9 | 0 |
| broadcast | + review (shipped) | .597 / .401 / .001 | 6 | 0 | 12 | 204 | 17.4 / 59.9 | 7 |
| veo 1-31 s | old decoder | .483 / .472 / .044 | 2 | 0 | 3 | 150 | 4.4 / 42.6 | 0 |
| veo | shipped | .477 / .522 / .001 | 2 | 0 | 3 | 130 | 4.2 / 42.6 | 1 |
| amateur3 0-60 s | old decoder | .245 / .586 / .169 | 3 | 0 | 2 | 94 | 1.95 / 11.2 | 0 |
| amateur3 0-60 s | shipped | .228 / .261 / .511 | 2 | 0 | 2 | 96 | 2.34 / 20.9 | 2 |
| amateur3 150-210 s | old decoder | .197 / .621 / .182 | 5 | 0 | 3 | 145 | 12.7 / 96.5 | 0 |
| amateur3 150-210 s | shipped | .181 / .738 / .081 | 4 | 0 | 2 | 83 | 9.7 / 36.1 | 1 |

Reading. The clip is fixed by the review, not by the penalties: with the penalties alone the decoder still reaches the ball boy's
ball through the stepping stone. On the broadcast the tracked share falls from .672 to .597; of the 419 frames the old path showed
as sightings and the new one does not, 327 are frames on which the causal pass had the ball hidden at a player and the old path sat
on something else, 50 are frames on which the old sighting was more than 3 m from a confident causal sighting, and 42 agreed with
the causal ball within 3 m (1.4 s lost). Checked by hand: at 39.3-42.2 s the old path followed a white object on the left (rejected
OTHER_OBJECT by the causal pass) for three seconds while the causal pass had the real ball dribbled along the bottom right; the new
path follows the causal ball. Amateur 0-60 s: the review cuts a 14 s chain of six weak sightings on a painted spot (detector .22,
four of six FIELD_LINE) that the stationary links had made a resting ball, so those frames are UNKNOWN. Amateur 150-210 s: the shown
hidden ball is never further than 36 m from the first pass's prediction instead of 96 m. The 180 s broadcast decode takes 46 s
instead of 8 s (stationary links are tried on every frame; a stride of three frames cost more than the margins decide on).

Known limitation, measured and left: a bridge may still ride over frames on which the causal pass had the ball confidently in sight
far from the bridge (broadcast: 79 such bridged frames before, 139 after, mostly 3 s stationary links on the left while the ball was
seen on the right, e.g. 21.3-22.5 s). A per-frame contradiction cost was tried; it removed those but produced an invented 23 m/s
diagonal flight elsewhere (21 teleports), so it is not shipped. Also tried and reverted, each measured: a per-frame travel charge on
bridges (cut real sparse flights on Veo and amateur footage, where the small ball is seen every 10-30 frames), pricing bridged frames
like NOT VISIBLE (amateur 0-60 s UNKNOWN .17 -> .54), a cheaper stationary bridge (.5: static junk on the touchline beat the real ball
through its occlusions on the broadcast), a place-based "something already stood there" test (the ball's own earlier sightings
match it), and a 1.5 m still region (cut dribbles).

Unmeasured: ball position error (no annotations exist; "no ball accuracy is claimed" still holds); the page overlay (Playwright is
not installed; the debug line now shows the motion state and the origin).

## Display labels apart from internal identities (2026-10-15)

The user deletes two team B "players" that were referees and the rest stay B-09..B-12: the shown number was the identity
itself (`GlobalPlayer.pid`, written as `display` on every frame), created in order and never reused, so a deletion left a
gap and a team of ten outfield players ended at B-12. New `soccer/labels.py`: `assign(result)` regenerates `display` from the
confirmed roster (match entries, every frame, the `labels` map, `summary.labelling`), per team in creation order, no gaps,
at most `outfieldPerTeam` (10) active numbers; a player who left (a correction) keeps the number and frees the slot; identities
beyond the slots are `B-?` with their data kept; goalkeepers untouched. Called at the end of `pipeline.build_report` (before
the roster, which takes `display` from the entries; the 2D view labels markers from the same entries) and in
`corrections.apply_result` after every correction, which logs `PLAYER LABELS RENUMBERED` with every move. `id` is never
changed: re-identification, galleries, tracklets, the timeline and corrections key on it. `tests/test_labels.py` runs the
user's scenario (B-01..B-12, delete B-07 and B-08 -> B-01..B-10, the former B-09..B-12 shown as B-07..B-10, same internal
ids on every frame, GK-A/GK-B untouched, roster and labels map updated, undo exact), a substitution (a player who left keeps
the number, the 11th takes B-11), fewer than ten (not padded) and stability (no change without a roster change). 224 tests
pass. Not changed: the online cap (`_create_or_hold` still creates an internal identity beyond the expected count after 5 s
when every missing identity is implausible, with the sanity warning) and the page's roster headings.

## Ball teleport to the penalty spot (2026-10-15)

The failure (the user's clip `BallTelport.mp4`, broadcast 12.0-15 s, run `runs/2bac432528d0e568/20261015-143540-2a22d2`): the final
path left the ball near midfield, sat on the right penalty spot (pitch (0.893, 0.505), 45.9 m away) from 12.57 to 13.37 s as
TRACKED / STATIONARY, and came back. The causal pass never went there: it had reacquired a boot at track 3's feet at 11.93 s
(detector 0.18, "4 px from the predicted position"), flagged the real ball OTHER_OBJECT while that lock was confident and JUMP
once it was hidden (`BALL SWITCH WARNING` at 12.567 s: required 119 m/s), and showed the ball OCCLUDED at that player.

Cause, from `path_dump.py` (the decoder re-run on the stored first pass, no detector): round 0 followed the real ball (tracklet
32, detector 0.78-0.93, 4-5 m/s, every sighting JUMP-rejected by the causal pass). The trajectory review then cut three
4-sighting stretches of it: a ball at 5 m/s passes through 0.5 m in four frames, so each stretch counted as "still", every
sighting was a strong negative (JUMP), and the entry was a flight. With the real ball removed the next round took tracklet 129,
the penalty spot (first sighting STATIC, the next 24 TRAJECTORY because the static flag flickered: the carried online
calibration drifted, so `_painted_spot` within 1 m and the settled drift test in pitch metres came and went), and nothing
cut it: TRAJECTORY is not a negative and one STATIC in 25 is under 40%. Three defects then: the review's stillness test had no
speed in it; the decoder did not know the painted spots or things standing for seconds; a marking shorter than a run could be a
stepping stone. Not a camera-motion or detector problem.

What changed (`soccer/ball_path.py`, `soccer/ball.py`, `soccer/geometry.py`, `pipeline.py`, `README.md`, tests in
`tests/test_ball_path.py` and `tests/test_ball.py`; 233 tests pass):

- Field markings in the decoder (`nodes_of`, `_standing_markings`): a candidate within 1 m of a penalty or centre spot of the
  offline registration (trusted, confidence 0.5 or more; `PAINTED_SPOTS` moved to `geometry.py`) keeps 40% of its
  ball-likeness unless the causal pass tracked the ball there; so does every sighting of a tracklet that spans 2 s and never
  leaves 0.6 m of its first sighting (pitch metres when registered, else camera-compensated image). `markingNodes` in
  `summary.ball`. Explicit penalty-spot handling with temporal evidence, not a mask: a ball tracked on the spot stays.
- Review stillness is a speed (`_review`): a stretch is still only when it moved under 1 m/s over its duration; the real ball
  rolling through 0.5 m is never chopped, whatever the causal pass said. Markings count as negatives; a stretch made only of
  markings is judged at any length (a stepping stone of two sightings is not a run), and a cut marking's other sightings go
  with it (the decoder re-found the next two sightings of the same still thing each round until the rounds ran out).
- Islands (`_review`): a run shorter than 1 s unreachable from the run before it and unable to reach the run after it
  (both over 40 m/s) is cut, `REJECTED_IMPOSSIBLE_JUMP`; the frames stay unknown. Future-frame validation for a moving thing.
- Causal `BALL SWITCH WARNING` (`_warn_switch`): elapsed time, required velocity, physical plausibility, near penalty spot
  YES/NO, static field marking evidence HIGH/MEDIUM/LOW with the stillness scores, detector confidence, `/ FIELD_MARKING`
  on a rejection of a known marking. The settled test (`_classify`) takes the smaller of the pitch and the camera-compensated
  image distance, so a thing fixed to the field stays settled while a carried calibration drifts.
- Tried and removed: a START penalty (JUMP_COST on a run that begins within 1 s of the last sighting on the path, faster than
  40 m/s away from it). It fixed nothing the review did not, and on the same 60 s it made the decoder keep a static thing
  outside the pitch (tracklet 1, pitch y 1.04) from 2.4 to 3.6 s and jump-link 36 m/s to the real ball at 4.47 s (13 teleports)
  rather than pay to restart on the real ball at 2.40 s: every alternative was "impossible", so the penalty only moved the
  jump. An impossible restart must cost what an impossible link costs and both must lose to leaving the gap unknown, which
  is what the island rule does after the fact.

Measurement (`ball_check.py`, re-decoding stored results from their first pass; the shipped Task 2 decoder is the "before").

| window | before: tracked / missing / unknown, runs, bridged > 8 m/s, drift mean / max, cuts | after |
|---|---|---|
| teleport run 0-60 s | .576 / .422 / .002, 4, 48, 15.7 / 44.9, 3 (visits the spot 12.57-12.70+, furthest 45.9 m) | .569 / .430 / .001, 2, 48, 15.4 / 44.9, 3 (never visits, furthest 2.8 m; 10294 marking nodes) |
| clip (ball boy, 10-50 s) | .672 / .327 / .001, 2, 49, 1.54 / 7.75, 4 (never visits) | .673 / .326 / .001, 2, 48, 1.54 / 7.75, 4 (never visits, furthest 5.7 m in 22-25 s) |
| veo | .477 / .522 / .001, 2, 130, 4.17 / 42.6, 1 | .482 / .517 / .001, 2, 130, 4.17 / 42.6, 0 (the cut stretch was a rolling ball) |
| amateur3_0 | .228 / .261 / .511, 2, 96, 2.34 / 20.9, 2 | identical |
| amateur3_150 | .181 / .738 / .081, 4, 83, 9.66 / 36.1, 1 | .182 / .569 / .248, 3, 84, 7.36 / 36.1, 2 (a 9 s bridge anchored on a marking is unknown now) |
| broadcast 0.5-180 s | .597 / .401 / .001, 6, 204, 17.4 / 59.9, 7 | .601 / .397 / .001, 6, 215, 17.4 / 59.9, 1 (six of the cuts were rolling-ball stretches) |

A GPU rerun of the same 60 s with the final code (scratchpad `task4/broadcast0`, 4.76 fps) gives the same first pass (.504 / .474 / .022, 9 teleports, 2 foot switches: the settled-test change did not alter it here) and a final path that never visits the spot (furthest 2.8 m from the last sighting in 12.4-13.5 s; identical to the re-decode), with the `BALL SWITCH WARNING` at 12.57 s now carrying the elapsed time, the penalty-spot and the marking lines. No island was cut on any window (the rule fires in the unit test only), no teleport anywhere, the fast passes of the Veo and
broadcast windows are unchanged (bridged > 8 m/s 130 and 215). Legitimate fast movement: links up to 40 m/s cost nothing extra,
a 36 m/s flight is still linked when nothing cheaper exists, and the island rule needs both ends impossible.

## A far-touchline player shown B-? for the whole run (2026-10-15)

The failure (the user's clip `B_wrong_track.mp4`, broadcast 1.1-12.8 s of their run `runs/2bac432528d0e568/20261015-162859-23f55f`,
0-60 s): a team B player on the far touchline is `B-?` at 1.6 s, drawn without any label from 2.8 to 7.8 s, `B-?` again after.
What the data says: he is ONE local track (8, no gap in 60 s) and ONE internal identity, B-14, created at 8.2 s ("every missing
identity of this group is implausible here; Team B now has 12 identities"), with his frames before 2.8 s labelled retroactively.
Three things went wrong, none of them a lost track:

- 2.8-7.8 s his kit read as neither team ("no consistent role or kit after 15 observations (kit matches neither team)"; a
  30-px person), so the track was `unknown` (`UNK-8`), which the page does not draw outside the debug view. Observations in
  that state were not kept for retroactive labelling (`identity.py` kept candidate, uncertain and nontarget ones), so when the
  track became B-14 the undecided frames stayed `UNK-8`; the offline refinement then scored that stretch like any anonymous
  fragment and the crops of a 30-px player did not pass the appearance floor.
- Team B held 12 identities. B-05 (local track 10) and B-13 (track 34) were created at 0.8 and 3.8 s at the bottom edge of
  the frame among the staff, seen 3.7 and 3.9 s, 63% of B-05's boxes cut by the frame edge, gone for good at 4.9 s; on 111
  frames 11-12 team B outfield identities were in view at once (team A: 11 on 224 frames, A-11 the same way). `labels.plan`
  numbered the first ten created, so B-14, the last, was `B-?` on every frame and his timeline slot carried `B-?`.
- The roster had no notion of which identities are fielded: 12 "ACTIVE / OFF_SCREEN" slots, the two sideline people with
  estimated positions for the 55 s they were not seen.

What changed (`soccer/identity.py`, `soccer/tracklets.py`, `soccer/labels.py`, `soccer/roster.py`, `soccer/imputation.py`,
`index.html`, `tools/compare.py`, `README.md`; tests in `tests/test_identity.py`, `tests/test_identity_encounters.py`,
`tests/test_labels.py`; 239 tests pass):

- Retroactive labelling covers the kit-undecided spell: 'unknown' observations stay pending (12 s buffer) and take the identity
  when the track is identified, like candidate and uncertain ones (the local track is the same person; a track switch still
  drops what came before it).
- Offline, the local track is evidence (`refine`): an anonymous stretch within 1 s of an identified stretch of the SAME track,
  with no track switch or appearance cut between, takes that identity without the appearance floor, unless the crops
  contradict it (OSNet cosine under 0.55), the stretch voted the other team, or the identity was seen elsewhere meanwhile.
  `TRACKLET CONNECT ... Same local track as B-14 after it`; `sameTrackContinued` in the identity metrics. Anonymous fragments
  of 1 s or more that stay anonymous are logged as `TRACKLET UNRESOLVED` with the best candidate, its score, the floor and the
  lead that stopped it (`unresolvedFragments`).
- Slots by participation (`labels.plan`): when a team holds more active identities than it fields, the numbered slots go to
  the identities with the most participation evidence over the whole segment (seconds observed with the box clear of the
  frame edges and the person not outside the pitch, compared in 2 s steps so that a few frames between runs never move a
  number; ties in creation order), numbers still in creation order. The entries and roster slots carry `confirmed` and
  `participation`; `PLAYER LABELS` names every unconfirmed identity with how long it was seen, how much of that at an edge and
  how many teammates were in view with it. The timeline keeps an unconfirmed identity's observed positions and estimates
  nothing for it; the 2D view's missing-players line leaves it out. The roster count is supporting evidence only: nothing is
  merged, deleted or invented from it (a referee classified as a B player with 19 s on the pitch cannot be told from a player
  by participation; that remains the role classifier's and the corrections' job).

Verification. Re-deriving the user's run with the new labels (scratch `task5/rederive.py`, no GPU): B-14 is `B-10` on all
1650 frames, roster slot confirmed, timeline O 1650 / P 150; B-05 and B-13 are `B-?`, unconfirmed, timeline O for their 112 /
116 observed frames and U elsewhere; A-11 the same. A GPU rerun of 0-20 s with the new identity code (scratch `task5/clipwin`):
track 8 is B-14 on all 600 frames (270 of them labelled retroactively, the former `UNK-8` spell included), shown `B-10`; 0 id
switches. That scratch run has no corrections applied (unlike the user's page), so the officials are 13 full-time team B
identities there and the slot ranking among them is creation order.

Two findings from the GPU regression batches (five windows, baseline = the Task 1 code's runs of 2026-10-14). The first batch
never continued a single fragment on real footage (`sameTrackContinued` 0 everywhere) although the cases exist: on amateur
150-210 s, track 100 is anonymous for 19.5 s (uncertain B-?) and then B-05 for 32 s on the same track. Two reasons, both fixed
and tested: the continuation ran inside the generic loop, after the refinement had connected a look-alike on another track to
B-05 by appearance, so B-05 counted as "seen elsewhere meanwhile" (the track itself outranks another track's appearance match:
it now runs first, and again after the global assignment, `tracklets.continue_same_track` from `pipeline.finalize_identities`);
and "seen elsewhere" was judged within +-50 ms of the fragment, which caught the neighbouring identified frame of the same track
33 ms away (now strictly inside the fragment's span). Final rerun (scratch `task5/win3`):

| window | identified share | id switches | fragmentation | tracklets connected | same-track continued | new identities |
|---|---|---|---|---|---|---|
| clip 10-50 s | .973 / .973 | 2 / 2 | .09 / .09 | 0 / 0 | 0 | 21 / 21 |
| veo 1-31 s | .404 / .406 | 14 / 14 | .78 / .78 | 9 / 10 | 1 | 18 / 18 |
| amateur 0-60 s | .682 / .684 | 24 / 24 | .96 / .96 | 17 / 17 | 0 | 23 / 23 |
| amateur 150-210 s | .773 / .789 | 59 / 58 | 2.57 / 2.52 | 43 / 46 | 4 | 23 / 23 |
| broadcast 0.5-180 s | .975 / .975 | 14 / 14 | .56 / .56 | 1 / 1 | 0 | 26 / 26 |

(clip, amateur 0 s and broadcast from the second batch, which already had the final causal code and the same offline outcome
for them: no same-track case exists on those windows.) Track 100 on amateur 150-210 s is B-05 from 158.67 s on (577 frames
connected offline, OSNet cosine 0.81 to the gallery, no contradiction), the 19.5 s of B-? before the causal pass decided are
corrected from the later frames; no frame carries one identity on two people. The occlusion regression of 2026-10-14 (B-07
behind A-03, now shown as B-06/B-07 after relabelling) still passes on the new clip run. The far-touchline player of the user's
clip is unchanged by the offline change: one track, B-14 on all 600 frames of the 0-20 s rerun, 270 of them retroactively.

## Player analytics: heatmap, average positions, movement trails (2026-10-15)

New panel between the 2D match view and the identity events (`index.html`), one pitch with three tabs (HEATMAP, AVERAGE
POSITION, MOVEMENT TRAIL) and a toolbar (team, player, range: processed segment / the five-minute block the playhead is
in / custom; scope for the averages: selected player, team A, team B, both). The computation is a Python module
(`soccer/analytics.py`), the page (`analytics.js`) only draws: `GET /api/analytics?player=&t0=&t1=&scope=` (`server.py`)
answers one selection from the observed samples of the saved result, extracted once per result file and cached by its
modification time (a correction rewrites the file). The page caches answers per result and clears them when the result
reloads, so corrections, exclusions and relabelling apply. Reuses tactical.js: the board (`drawPitch`, `geometry`,
goal-end labels) and the page's team colours. Clicking a player in Match identities selects them for the analytics too.

Rules (documented in the module and the README): only confirmed team players and goalkeepers of the final roster with a
confirmed slot; an observation counts when the person is confirmed, identity confidence >= .3, the position is on the
pitch (within 2% of the lines, never clamped), registration confidence >= .5 and the box is not cut by the frame edge;
estimates never count. Samples smoothed with the 2D view's 0.14 s window within runs. A sample stands for half the gap
to each neighbour, at most .25 s a side (half a frame on an open side): denser sampling is not more occupancy, a long
absence counts a quarter second at each end only. Coverage = the union of those seconds over the interval; < 50% is
flagged. Heatmap: 1 m grid, Gaussian sigma 2.5 m, total preserved, square-root colour ramp, legend in seconds per cell.
Average: the time-weighted mean; low coverage drawn dashed with the percentage. Trail: joined across gaps <= 1 s only;
a step over 2 m faster than 12 m/s is a registration glitch and a break (a far player's half-metre jitter is not: the
first rule without the distance broke B-14's path 82 times); START/END, arrowheads, a white dot following the video.

Verified in the browser on the user's 0-60 s run (server on port 8766): A-01's heatmap (coverage 98%), both teams'
average positions with GK-B dashed at 10% coverage, B-14's trail (two segments, one 5 s gap, 114 m) and a custom
10-20 s range (one segment, 19.4 m), the roster click selecting B-14; no console errors from the page. On the real
result: JSON load 0.3 s, extraction 0.04 s, one report 6 ms, 81 KB. `tests/test_analytics.py`: 16 tests on synthetic
trajectories (density proportional to observed time, sampling rate without effect, absence not counted, range, validity
rules, time-weighted mean, two local tracks one identity, roster and corrections, trail breaks and jitter, axis
convention, report shape). 258 tests pass.

## Review page redesign: one workspace, a carousel of three views (2026-10-15)

Styling and layout only (`index.html`; `tactical.js`, `analytics.js`, the server and every algorithm untouched; the page
script kept, with additive edits). White design system (background #FFFFFF / #F7F9F7, brand green #17834A, text #17211B,
muted #68736B, borders #E3E8E3; system sans-serif; subtle borders, one small shadow on the drawer and the popover; no
gradients, glows or cards). Header: a typographic PitchIQ wordmark with a small pitch mark, the match name, New match /
Edit tracking / Advanced. One workspace with TRACKING, 2D MATCH and ANALYTICS tabs (green underline) above it; the views
are three sections in a `.carousel`, all mounted, switched by CSS transitions on transform and opacity (the outgoing view
slides 62% aside and shrinks to .74 with a 6 degree turn while the incoming one does the reverse; the container's height
transitions between the two natural heights; forward is the loop order, a tab takes the shorter direction; arrows and the
keyboard arrows step the loop; reduced motion swaps instantly; phones slide without the off-screen previews). Per-view
controls under the workspace; the analytics toolbar stays above its pitch. Drawers for the identities and corrections,
for upload and processing, and for the summary, debug overlays, events, pitch confirmation and keyframes (a side column
on wide screens, an overlay with a backdrop under 1100 px). Added on the way: playback speed, full screen, the timeline
CSV button and the lineup/footprint checkboxes were not bound before and are now; the identity-debug overlay and the video
HUD are off by default. Verified in the browser on the user's run at 1440 px, tablet and phone widths: the tabs switch
the views with the video time preserved (paused at 00:35, the board shows 00:35), the Web Animations API lists the
transform and opacity transitions on both views and the height transition on the workspace (the harness's animation
clock does not advance while its tools run, so no midpoint photo), the roster click in the Edit tracking drawer selects
the player for the analytics, the Advanced drawer lists the summary and events, the correction popover opens on a player
box, no page errors. 258 tests pass (no Python changed).

## Review page, second pass: a bounded media stage, one transport, full screen (2026-10-15)

Still `index.html` only. What the user found after the first pass: plain white space, small generic type, browser-default
controls, the video growing past the viewport when the drawer closed (controls pushed off-screen), no playback controls on
the 2D view, no usable controls in full screen. Changes: a light canvas (#F8FAF8) with white surfaces, system Inter/Segoe UI
Variable stack, a match header (kicker, title, metadata chips, status chip), a 1560 px shell; the carousel kept as is. The
media stage: `.media{width:min(100%, var(--stage-h) * var(--ar))}` with `--ar` the video's aspect (from its metadata) or
the board's 1.48, and `--stage-h` measured by `fitStage()` (viewport height minus the stage's top, the transport and the
visible controls) on resize, view switch, drawer toggle and fullscreen change; the transport bar's width follows the
active media. One transport under the stage for every view: SVG icons, played portion in green, duration, speed, full
screen; the script's `#play/#back/#next/#timeline/#time` ids kept, its `video.onplay/onpause` and the per-frame time
update now call the `playback` controller (the video element stays the only clock; the board and analytics read it every
frame as before). Keyboard shortcuts (space/k, arrows, shift-arrows, f, escape) that yield to inputs, buttons, the popover
and the drawing modes. Full screen on the stage wrapper (view + transport) with idle-hiding controls; a fixed-layer
fallback with the same presentation when the Fullscreen API is refused (the in-app browser refuses it). Display options
as chips, custom selects, consistent 34 px controls, the identity popover restyled. Verified in the browser on the user's
run: 1920x1080 video 1274x717 with the transport ending at 1024 px (in view), 1440x900 and 1366x768 in view, the 2D view
plays/pauses from the shared bar with the board at the video's time, space toggles when nothing is focused, fallback full
screen shows the dark stage with the controls; tablet and phone without overflow; no page errors. 258 tests pass (no
Python changed).

## PitchIQ Studio: the third pass on the review page (2026-10-15)

`index.html` and the insight panel in `analytics.js`; nothing else. Design tokens in `:root` (colours, radii, shadows,
spacing, durations, control heights, the type stack). A light shell with a dark media stage (#111915 with a thin
#263429 border and a soft shadow) holding the video or the board and the transport; the Analytics view turns the stage
light and hides the transport (the user asked for no playback controls there). Match hero: kicker, 32 px title from the
upload's original name, one line of real context (status with a pulsing dot while processing, length, resolution and
fps, tracked identities, analysed stretch). Segmented view switcher with a sliding indicator (transform transition) and
the loop arrows; a stage label and a status (No analysis yet / Processing / Tracking available / Error) on the right.
Analytics as a two-column dashboard: pitch left (sized by `--stage-h`), insight panel right (`#anInsights`, rows written
by `describe()` in analytics.js: player, team swatch, interval, observed duration, coverage flagged when low, samples,
peak occupancy and smoothing, average X/Y, segments / distance / gaps, excluded samples), the legend and a note; a reveal
on tab change; stacked under 1180 px. Welcome state when no video is open (`.stage.welcome-view`): heading, one sentence,
Upload match video (opens the file picker) and Open an earlier upload (opens the New match drawer) with a pitch
illustration. Micro-interactions: pressed buttons and chips, drawer and popover entrance, selection transitions; reduced
motion disables them. Fixes on the way: the analytics canvas is redrawn when its layout width changes (the board module
clears it on resize), the stage's own padding is in the height fit, the switcher hides in the welcome state.

Verified in the in-app browser on the user's run at 1440x900: the dark stage with the video (913x514) and the transport
in view; the 2D match (board 710x480) at the video's time; Analytics with the insight rows and no transport, the canvas
drawn after the layout change; the welcome state on a server with no video; then 1920x1080, 1366x768, tablet and phone
without overflow and with the controls in view. 258 tests pass (no Python changed).

## Dark Studio and the Analyze Segment dock: the fourth pass on the review page (2026-10-15)

`index.html` only; no Python, no tactical.js, no analytics.js. The light shell became the dark graphite / forest-green
theme (tokens in `:root`: #101715 app, #19221F / #202B26 / #28342D surfaces, #090E0C media, #F1F5F2 / #9AAA9F / #718278
text, #334038 borders, #19B96B green with #22CF79 hover and #117B49 deep, #D96B6B danger; radii 12 / 10 / 7 / 6 / 5 px;
no gradients, glows or large shadows). Header 60 px with "Match studio / <file>" and New match (solid green), Edit
tracking, Advanced (dark secondary). Hero: kicker, 26 px title, one line of real data (analysis state with a dot, length,
resolution, FPS, analysed stretch). Segmented switcher with a lighter selected surface and a thin green indicator; the
carousel kept. The stage is at most 1450 px wide and height-fitted by `fitStage()` (which now also reserves the dock's
head on screens at least 860 px tall; shorter screens scroll). The transport: outline icons, a larger green play, green
progress, one clock; hidden on Analytics; fullscreen keeps it.

The permanent right-hand upload / processing drawer page is gone. Its controls moved into `section.dock#dock` under
the media: the source row (`#srcName`, `#srcMeta`, Change video toggling `#earlier` with `#uploads` + `#openUpload`,
`#upload` + `#file`, upload progress), the trim timeline (`#trimRuler`, `#trimStrip` with the `#trimThumbs` canvas,
`#trimRange`, handles `#trimStart` / `#trimEnd` as keyboard sliders, `#trimPlayhead`, `#trimTip`), the range row
(`#trimStartInput`, `#trimEndInput`, `#trimDuration`, `#run` restyled as ANALYZE, `#stop`), the status row (`#progress`,
`#status`, `#live`) and `details#procSettings` with the existing inputs (`#resolution`, `#perTeam`, `#margin`,
`#fieldFilter`, `#continueIds`, `#ballTiles`, `#previousIds`, `#gpu`). The hidden `#start` / `#seconds` inputs stay and
are written by the trim component, so `settings()` and the `#run` handler (`/api/run`) are unchanged. The `trim`
module keeps its own state (duration, fps, start, end): pointer events with capture for mouse and touch, a rAF render
during drags, frame snapping at the real fps, min 0.5 s, max 180 s (the backend's limit), range moves that keep the
length, keyboard steps (frame / shift second / page 10 s / Home / End), mm:ss.ff parsing, an adaptive ruler, thumbnails
from a hidden muted video element (sequential seeks, cached per width). `trim.init()` runs with the saved settings of
the open video, so the previous run's range comes back. `stageStatus()` feeds the hero line and the dock state;
`focusDock()` serves New match and the welcome's Open an earlier upload. The welcome state adds `dock.no-source`.

Verified in the in-app browser (two review servers: the broadcast upload on 8766, no video on 8767): dark theme on all
three views, drawers for Edit tracking and Advanced, New match focusing the dock; dragging the end handle (end 2:28.87,
replay untouched, no run started), dragging the range (start 23.10, length kept), keyboard steps (three frames and a
second), typed times, invalid text restored, the 0.5 s floor and the 3 min ceiling, seeking the replay without moving
the range; a real run from ANALYZE on the Veo clip (0-1 s at 960) with the progress in the dock and the hero and the
result loading afterwards (the clip's saved run files were restored afterwards); 1920x1080, 1440x900, 1366x768, 768x1024
and 375x812 without horizontal overflow (phone: full-width strip, 20 px handles with 44 px targets, stacked fields,
full-width ANALYZE). 258 tests pass (no Python changed).

## Faster without changing the tracking (2026-10-16)

The rule for this pass: every result stays bit for bit what the committed code produces. Profiling a 10 s broadcast
run (cProfile; 1536, ball scans on) put the pitch homography refinement first: `soccer/field.py`'s `_residuals` was
pure-Python loops over landmarks, lines and up to 600 centre-circle points, scipy's least-squares solver evaluates it
about 180 times per fit (finite-difference Jacobian), and the pipeline refines on most frames; that was about a third
of the whole run. The GPU detector itself was 39 ms of the 201 ms a frame took; the CPU tracking stage was 121 ms.

Two changes. (1) `_residuals` is vectorized: the observations become arrays once per fit (`_prepared`, cached in the
fit's own copy of the observations), the projections go through `_project_rows` (a stacked `np.matmul`, which numpy
computes with the same arithmetic as the per-point `H @ [x, y, 1]`; a `(n, 3) @ H.T` goes through BLAS and differs in
the last bit), and the norms through `_hypot`, a numpy port of CPython's accurately rounded `math.hypot` (numpy's
`hypot` differs from it by one ulp in a sixth of all cases, and that alone moved a fit's stopping point and flipped an
edge "outside the field" decision in 12 of 1200 frames of the first attempt). Checked: 800 of 800 synthetic cases and
19,354 real evaluations identical to the old function (largest difference 0.0), fits identical including the
Jacobian, the hypot port with 0 mismatches in 2.5 million pairs plus the special values; 4.8x faster per call on the
real inputs. (2) `pipeline.Prefetch`: a worker thread with its own detector instance decodes and runs the whole-frame
detection and the ball tile scans one frame ahead (queue depth 2) while the tracker works on the current frame; the
passes that depend on the tracker (far tiles, lost-track crops, the ball search crop) stay on the main thread after the
previous frame's tracking, so the frame order and every detection call are the ones the serial loop made.
`summary.timing`'s detect and ballTiles are now the worker's seconds and overlap the tracking; `elapsedSeconds` is
the wall clock.

Verification: the broadcast 10-50 s and Veo 0-20 s windows run with `tools/run_segment.py` on the committed code and
on each change, compared frame by frame (people: track, identity, label, team, role, state, box; pitch positions; ball;
mapping; match, roster, tracks, teams, labels, events): zero differences for both changes on both windows, and again on
the 10-30 s timing window. 258 tests pass. Timing could only be measured while a game held the GPU at 75-97% and
shared the CPU: on the 10-30 s broadcast window the committed code took 348 s and 351 s, the new code 246 s and 250 s
(1.4x); the residual's share is cleaner in-process (0.98 -> 0.20 ms per evaluation, about 160 evaluations a frame).
Measure again on an idle machine. Still on the table, all of which change numerics and so need a decision: the OSNet
embedder on the GPU (it runs on the CPU through ONNX Runtime, ~25 crops a frame; the torch thread count made no
difference), all six ball tiles in one FP16 call, TensorRT, a single detection pass, camera flow at half resolution,
refining the pitch less often.

### Round two: the rest of the list (2026-10-16, later)

The user then asked for the remaining optimizations with "make sure nothing breaks", and said the tracker will
move to cloud GPUs later (not now): nothing that ties the code to one GPU model or one machine. Done:

- **Ball and far tiles in one detector call** (`ball_tiles`, `far_tiles` in pipeline.py): six 640 px tiles
  per call instead of pairs; peak GPU memory had been 0.37 GB, so the old bound was far too cautious. FP16
  batching moves the ball candidates' scores in the last digits, which the offline ball decoder sees: on the
  broadcast 10-50 s window 26 of 1200 frames have a different ball record, no state or visibility change,
  centre shifts of at most 0.04 px; on the Veo window 38 frames, one 2.2 px shift. People, pitch positions,
  roster, tracks, labels and events are identical.
- **A batched Jacobian for the pitch fit** (`_residual_rows`, `_jacobian` in field.py): the residual now
  evaluates several parameter vectors at once (stacked products, batched inverses, the two BLAS/norm
  terms per row) and `fit` hands least_squares a `jac` that reproduces scipy's default 2-point finite
  differences exactly (EPS**0.5 steps with sign and max(1, |x|), the one-sided adjustment at the bounds,
  (x0+h)-x0 denominators, the F-ordered layout), evaluating all perturbations in one call. Verified:
  480 of 480 Jacobians identical to scipy's own on the old residual, including parameters sitting on the
  size bounds; 32 of 32 whole fits identical (H, rms, size, jac); 6.3x faster per fit; the two reference
  windows bit-identical to the batched-tile runs.

Tried and not kept: **the OSNet embedder on the GPU**. ONNX Runtime's CUDA provider (onnxruntime-gpu
1.22.0) runs every convolution in its "Fallback mode" on the RTX 5080 (compute capability 12.0): 249 ms a
frame against 12 ms on the CPU. Converting the ONNX graph to PyTorch (onnx2torch, with a rank-agnostic
BatchNorm for the final feature BN) gives descriptors within 4e-5 of the CPU ones, but eager execution of
its ~500 modules costs 26 ms a frame for two batches of 16, and CUDA-graph capture of the converted graph
fails. The embedder stays on the CPU through ONNX Runtime 1.22.1 (the environment was restored; nothing
new in requirements.txt). Also skipped: TensorRT (engines are built per GPU model and driver, the wrong
direction for the cloud plan), a single detection pass (the 960 pass keeps large foreground players this
checkpoint loses at 1536), refining the pitch less often (little left to gain), camera flow at lower
resolution (it already runs at 960 px wide). The torch CPU thread count makes no difference.

Clean timing (idle GPU) on the user's 0-60 s broadcast window at 1536 with ball scans: the committed code
436 s (4.13 fps), the new code 235 s (7.66 fps), 1.86x; same identified share, pitch confidence and ball
tracked share, people and pitch identical, the ball record differing only in the last digits as above. The
10-30 s window: 137 / 158 s before, 97 / 110 s after round one. 258 tests pass.

## Speed pass 3: batched detection, scans on demand, a vectorized ball decode, models kept loaded (2026-10-16)

Done in a cloud clone without a GPU, without `uploads/` and without a `.venv` (the task's fallback branch): CPU
torch and requirements.txt were installed to run the unit tests, tasks 1 to 4 were implemented, task 3 was
proven bit-identical on synthetic inputs, and the verification protocol was written into
`tools/verify_speed_pass_3.cmd`. **Nothing below was timed or regression-tested on the GPU**: the timing
table and the exact_diff / compare / ball-clip results for tasks 1, 2 and 4 are still to be produced on the
RTX 5080 (idle GPU, `nvidia-smi` first) by running that script, which does the whole protocol: a git
worktree of the committed code (`gpu-tracking-new`) next to this code, the two reference windows
(broadcast 10-50 s, Veo 0-20 s, 1536, ball tiles), exact_diff and compare on both, the 60 s broadcast window
back to back, two consecutive runs against two fresh ones, `tools/check_ball_decode.py` on the real frames
and `tools/ball_clips.py` on the three ball clips. The rules held: nothing per GPU model or per machine
(batching, threads and the same FP16 PyTorch calls only), every frame processed, identity, calibration and
the ball tracker's decisions untouched, requirements.txt unchanged, result version 3 and the page unchanged.
One commit per change on the branch; each message says what was and was not measured.

1. **Batched whole-frame detection** (`pipeline.detect_batch`, `Prefetch._work` / `_flush`). The worker
   collects the segment's frames in groups of `detectBatch` (default `DETECT_BATCH = 4`; the last group may
   be shorter; a group is flushed before any roll frame so the items keep the video's order) and runs one
   `model.predict` call per size pass (960 and the detail size) for the whole group, then merges the passes
   and suppresses duplicates per frame exactly as `detect()` did for one frame (`detect()` is now a batch of
   one; a one-element list and a bare array are the same input to Ultralytics, and same-shaped frames get
   the same letterbox in a batch as alone). The batch's seconds are shared out per frame for
   `summary.timing`. `tools/run_segment.py --detect-batch`, an optional `detectBatch` in the server's
   settings. Verified here only with a stub detector (`tests/test_prefetch.py`: batched rows identical to the
   serial ones, order, roll frames, tile scans); FP16 batching may move scores in the last digits, which the
   regression protocol is for. Not timed.
2. **Ball tile scans only when needed** (`Prefetch.scan_wanted`, `ball_locked`). The main thread writes a
   shared flag after every frame (the ball tracker's phase is LOCKED); the worker reads it when it decides a
   frame's scan: every other frame while not locked (SEARCHING, RECOVERING, OCCLUDED), once per
   `BALL_SCAN_S = 1` s while locked (`ballScanEvery`, `--ball-scan-every`, to tune on the ball clips). The
   hand-over became a semaphore of slots the consumer gives back after finishing each frame, and the worker
   decides a scan at the moment it gets a slot, so the word it reads is at most `depth` frames old (two
   detector batches: 8 frames, 0.32 s at 25 fps; smaller `detectBatch` shortens it). `summary.detection`
   gains `ballTileScans` and `detectBatch`. Tested with the stub (flag honoured with the lag, scans off, an
   early close). Not timed; the clips (12.0-15.0, 22.1-24.3, 40.8-42.8 s) not re-run.
3. **Vectorized offline ball decode** (`ball_path.decode`, `_layer_arrays`). The links into a frame were
   costed one source frame at a time (up to 75 of them at 25 fps), each with tiny numpy ops and four Python
   comprehensions over the node pairs; that loop was the 20 s offline ball pass. Now the source frames' nodes
   are stacked into rows (nearest frame first, each frame's nodes in order) and every term is computed once
   per frame with the same elementwise arithmetic: per-frame scalars (dt, 1+dt, skip and unreliability costs,
   the foot-switch fade) become per-row columns; player and tracklet comparisons use integer codes from one
   dictionary (equal exactly when the objects were); the consecutive-frame "all the same tracklet" exemption
   and the NaN behaviour (an undefined link lets its frame offer nothing to that candidate) are reproduced
   per source frame; the first row reaching a column's minimum is the frame and node the sequential
   strict-less updates accepted. The camera transform is still applied to each source frame's nodes alone
   (the same BLAS shapes; a stacked `(N, 3) @ (3, 3)` would not be bit-safe, see round one). A frame's
   reachability is recorded once instead of re-tested on every later frame. Measured on this machine's CPU,
   1500 synthetic frames of 2-10 candidates (`tests/ball_synth.py`): the frozen copy of the old decode
   (`tests/ball_path_reference.py`) 7.8-8.2 s, the new one 1.0-1.4 s (7x); `finalize()` end to end 8.4 s
   -> 1.8 s. Identical paths (nodes, link costs and speeds bit for bit) on 5 seeds x {with, without
   registration} at 1500 frames, 60 small random cases with cuts, unreliable frames, ties and empty frames,
   and `finalize`'s frames and report byte for byte (`tests/test_ball_path_speed.py`).
   `tools/check_ball_decode.py RUN_DIR` repeats the side-by-side on a real run (none was available here).
4. **Detectors and embedder kept between runs** (`pipeline.Resources`, `RESOURCES`). One pair of detector
   instances per process (the main thread's and the prefetch worker's) and the OSNet session, loaded on the
   first run and reused; `server.py` preloads them in a background thread at start-up; `run_segment.py`
   keeps them across `--batch` jobs and `--fresh-models` loads them anew per job (the two-consecutive-vs-
   two-fresh comparison of the protocol). A kept predictor re-derives its arguments on every call and its
   one-time warm-up is a dummy forward pass, so it gives what a fresh one gives. Tested with stubs
   (`tests/test_resources.py`). The ~10 s saving per run is not measured here.
5. **The pitch chain in a second process: not done.** The task allowed it only after 1 to 4 were verified,
   which this machine could not do; no timing of the chain was taken (no footage here, and synthetic frames
   would not say what real 1080p broadcast frames cost).

268 tests pass, 1 skipped (258 before, plus 10 new). Files: `pipeline.py`, `server.py`, `soccer/ball_path.py`,
`tools/run_segment.py`, new `tools/check_ball_decode.py`, `tools/ball_clips.py`, `tools/verify_speed_pass_3.py`
and `.cmd`, new tests `tests/test_prefetch.py`, `tests/test_resources.py`, `tests/test_ball_path_speed.py`
with the helpers `tests/ball_synth.py` and `tests/ball_path_reference.py`; README and this file.
