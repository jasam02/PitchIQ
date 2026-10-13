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
