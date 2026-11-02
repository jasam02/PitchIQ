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

### Speed pass 3 verified on the GPU and merged (2026-10-16, later)

The protocol (`tools/verify_speed_pass_3.cmd`) ran on the RTX 5080 against the committed code. Identities and
positions: roster, tracks, labels and events identical on the broadcast 10-50 s and 0-60 s windows, identified
share and id switches unchanged; with `detectBatch` 1 the people records are bit for bit the committed ones,
with 4 boxes moved by at most 3.6 px in 8 of 30,000 observations and pitch positions by at most 2 mm; on the
Veo window one internal track number shifts by one, every box and label agrees (compare.py 1.0). The ball:
the scan-on-demand policy changes the decoder's candidates, tracked share 0.569 -> 0.576 (0-60 s) and 0.673 ->
0.677 (10-50 s), unchanged on Veo; the three clips (penalty spot, ball boy, B-07 behind A-03) pass before and
after with no flagged frame. The vectorized decode is identical node for node on 1,200 and 1,800 real frames
and 6x faster (the offline ball pass 24 s -> 6 s per minute). Kept models: two consecutive runs identical to two
fresh ones. Timing was noisy (other processes on the CPU and GPU); alternating 60 s runs under the same
conditions: committed 282 / 299 s, this code 249 / 262 s (about 12%), the 10-50 s window 146 -> 133 s. Batch
sizes 1, 2 and 4 were indistinguishable within the noise, so the default is 1 (detections bit for bit as
before; the setting stays for other GPUs). Merged into gpu-tracking-new.

## Team shape: width and depth, defensive line height, formation recognition (2026-10-17)

Built on the team shape spec as a second Analytics category (Player / Team shape) with three tabs, from the saved
result only (`soccer/team_shape.py`, `teamshape.js`, `/api/team-shape*`); nothing in the tracking, ball, re-ID or
calibration changed. Everything uses the player analytics' validated samples (`analytics.Samples`), so identity
corrections apply. The README's "Team shape analytics" section has the rules; the decisions the measurements forced:

- **Templates from data, not from the name.** The 4-4-2, 4-2-3-1, 4-3-3, 3-5-2 and 3-4-3 templates are the mean role
  positions of SoccerCPD's 864 real formation periods (Kim et al., KDD 2022, `role_records.csv`, symmetrized); the
  others are composed from the same lines. A 4-4-1-1 template was dropped: it was the empirical 4-2-3-1 to within
  noise. Tests use hand-placed layouts deliberately unlike the templates, so recognising them is not the templates
  recognising themselves, and no formation accuracy is claimed (Metrica's sample games have no expert labels).
- **Stabilization tuned on game 1, judged on game 2.** Raw 15 s window labels changed about 250 times an hour on
  Metrica game 1; a Viterbi change-point search (switch penalty 0.45 of relative fit error, an "unclear" state at
  0.42) brought that to 3 / 0 (team A) and 3 / 9 (team B) changes per half. On the held-out game 2: team A 4-4-2 for
  61% / 59% of the halves (5 / 4 changes), team B 4-4-2 for 93% / 66% (0 / 1 change).
- **Coverage before claims.** Corrected on 2026-10-18 (see "Team shape bench: correlated camera noise" below): the
  numbers first written here ("a whole team in view only about 0.1% of the time", "8-player windows agreed 25 to 56%")
  measured the bench's noise model, not the camera. With a simulated 60 m side camera on Metrica game 1 and position
  errors that persist (AR(1), 1.5 s), a sample is complete 55% of the time (63% in play); complete and reliable widths
  and depths are off by about 1.1 m (median; 90th percentile 2.7 to 3.4 m), while partial widths are 2 to 4 m short on
  average (90th percentile 8 to 11 m) and partial depths 7 to 11 m short (19 to 23 m), so partial values are labelled,
  drawn faded and kept out of the averages. A formation window may miss at most one player of the lineup: an 8-player
  window's own fit matched the full-coverage label 21 to 38% of the time, and allowing it would mostly lend it its
  neighbours' label.
- **No silent side changes.** Without the half-time swap set, the second half of game 2 is reported as "direction
  uncertain" for 99.9% of its time and no defensive line is measured there. The first version still measured 0 to 3%
  of that half from the wrong goal: windows where the goalkeeper's side and the team ordering disagreed were
  believed. The goalkeeper now decides when seen (on Metrica the two kinds disagreed in 1 to 3% of the windows, as
  often under the right direction as under a wrong one), and a window without evidence next to a contradicted one is
  not believed either. With the swap set nothing changed. Line heights on game 2 (median, from the own goal): team A
  36.6 / 35.1 m, team B 36.7 / 36.2 m.

An adversarial review workflow (six lenses, each finding verified) confirmed 50 defects in the first version, among
them a crash when more players were eligible than a template has roles (a lineup of 9 with 10 on the pitch), the
estimated mode calling samples with unpositioned players reliable, a coach's label on an "unclear" period not deciding
the defensive unit, the depth fallback mixing both directions at a side swap, one lineup per video, the roster's
"left" being ignored, races between the samples and the slim result in the server cache, settings writes that could
wipe the file on Windows, and panel forms losing unsaved choices. The module was rewritten in one pass (v2) and every
finding has a regression test (26, 23 of them failing on the first version; the other three are controls). A second
review workflow (four reviewers, each new finding checked by a skeptic told to refute it) found 15 of the 50 fixes only
partial and 24 new defects; all 12 new ones it verified reproduced, none was refuted. Among them: the defensive-unit vote
tied for 5 s at every sustained role change and dropped both players; a coach's label on a 30 s period was outvoted by
the neighbouring periods; a short stored label took over a long period; a split identity's stale prediction let the
edge check call a width reliable; a flickering duplicate was never removed; a departed player's identity riding on his
substitute removed the substitute; two estimates of one person counted twice. All 24 were fixed, each with a test that
fails on the previous version (16 tests), and so were the partial fixes except the three listed below. `tests/test_team_shape.py` now has 90 tests; the full suite 358, 1 skipped.

Real footage: the broadcast 0-60 s gives team A a clear 4-4-2 (width 39 m, depth 19 m, line 31 m from its goal, deepest
defender 30 m) and team B a clear 3-4-3 (width 57 m, line 50 m, advancing); every metre is marked estimated (the pitch
size is the standard assumption). The amateur clip (4 to 5 players of a team observed) and the Veo clip (pitch
confidence under 0.5) report "not enough players observed" rather than a shape. All 20 saved results under `runs/`
analyse without an error, in under a second each.

Not done: possession phases (the result has no reliable possession record, so every shape is a positional structure
over both phases, and the page says so); a drag on the chart to set the range; formation accuracy against expert
labels; more than one lineup change per team (a second dismissal); keyboard navigation between the tabs (they carry
tab roles, no arrow keys); custom range fields that start at the segment's bounds. A roster "left" mark does not lower the lineup by itself (it may be a substitution): a
dismissal needs the lineup setting too.

## Amateur footage, round 2: A_wrong.mp4, keyless pitch, small players, the foreground ball (2026-10-17/18)

**The test clip.** `A_wrong.mp4` (Videos) is a 896x506 screen recording of the review page playing the user's run
`runs/6dfdc88ec6f2d100/20261017-174228-1eaac0` of amateur 3 (118.2-182 s): the clip at t is the source at t + ~121.0 s
(matched frame by frame: clip 27.0 s = source 148.07 s). All measurements below use the clean 1920x1080 source window
(118.2-182 s, "awsrc"); the recording (old boxes, labels and ball ring burned in, a yellow colour cast, a quarter of
the pixels) is a secondary low-resolution test only. The baseline run of the committed code on that window reproduces
the user's run exactly (label agreement 1.000, identical metrics). Hand-made references (this round, by eye on zoomed
source frames, +-3-4 px): `tools/annotations/ball_positions.json` (11 ball centres at 146.0-150.0 s, scored by
`tools/check_ball_truth.py`) and `tools/annotations/pitch_landmarks.json` (halfway line, circle crossings, centre spot,
far and near touchline points at 127.0 and 148.07 s, scored by `tools/check_pitch_truth.py`).

**Audit before the changes** (five parallel code audits plus GPU probes on the source):

1. Pitch calibration: per evidence tick the PitchDetector finds grass, paint ridges, Hough lines, boundaries and the
   centre circle; a live model is anchored by the user's keyframes (>= 4 landmarks) and refined by matched paint; camera
   motion (LK flow + RANSAC, phase-correlation fallback) carries it between ticks; SIFT re-registration returns to
   keyframe views; offline, forward and backward chains between anchors are blended and every position is remapped.
2. Semantic centre-circle geometry: the canonical model (field.py) has named landmarks and lines; the circle is found by
   RANSAC ellipse fitting around the halfway line; the automatic midfield anchor existed but needed a touchline and was
   refused on every amateur tick ("no touchline": 1255 times on amateur 3). Without user keyframes no frame of the A_wrong
   window had pitch coordinates (calibrated share 0, the 2D view empty).
3. Lens distortion: not a factor. The virtual follow camera's output is rectilinear (long halfway and touchline paint is
   straight; lens.estimate_k finds no estimate); the curved black corners are the edge of the panorama's coverage.
4. Why the projection failed: no anchor at all. On the recording also the colour cast (the fixed grass rule rejects the
   yellowed turf on 26 of 30 seconds) and the burned-in boundary line taken as a touchline.
5. Unresolved players: local tracks fragment (median 0.2 s), kit votes need crops of ~23 px or more, OSNet x0.25 does
   not separate teammates (within-identity cosine .71-.75 vs best other teammate .77-.80), galleries stay empty under
   ~35 px, far-tile detections (6x13 px) needed IoU > .5 on three frames to count (87% never did), and a kit-change
   release kept the track's old birth time, which zeroed every same-team candidate and created wrong-team duplicates.
6. The foreground ball (A_wrong 26-29 s = 146-150 s): (a) the YOLO ball class gives no candidate for the ~22 px ball at
   the dribbler's feet from 146.0 to 148.8 s at any input size (640-1536) or crop scale, even at confidence .03;
   (b) after 1.27 s hidden, the hidden-ball gate (converted to pixels with the scale of the last, near sighting: about
   21 m at the far depth) accepted a single detection on the painted penalty arc 180 px away; (c) the lock moved to a
   still 3-4 px speck far up the pitch; (d) that lock counted as mature and flagged the real ball OTHER_OBJECT for 2 s
   when the detector found it again (149.2-151.2 s, detector up to .95); (e) the offline decoder deferred to the causal
   lock. Result: 0 of 11 hand-marked ball frames on the ball, the path 500-980 px away from 148 s on.
7. So the ball was missing from the candidates (146.0-149.2 s) and then rejected afterwards (149.2-151.2 s).
8. Kept unchanged: the whole broadcast path: detector passes (960 + detail), BoT-SORT parameters, re-ID weights and
   thresholds, the keyframe calibration and paint refinement, the offline ball decoder's costs, the 2D renderer, analytics.
9. Modified: midfield.py, pitch.py, field.py, pitch_model.py, registration.py, geometry.py, goals.py; tracker.py,
   pipeline.py, local.py, relevance.py; ball.py, ball_path.py, new ball_proposals.py; identity.py, tracklets.py, labels.py,
   roster.py; imputation.py, visibility.py; index.html (debug overlay); tools (diagnose, compare, counts, check_*_truth).
10. Expected cost: none on broadcast (every amateur behaviour is behind a gate that is closed there); on amateur
   footage a 640 people pass on small-scene frames, the pinhole solve while there is no model (~30 ms per evidence
   tick), the classical ball proposer while the ball is unseen (~1-2 ms per frame).

**What was built** (the README's "Amateur footage (small scenes)" section has the rules and constants): the gates
(`small_scene`, `auto_landmarks`, `pitch_provisional`); a pinhole midfield anchor for runs without keyframes (circle on
the grass, halfway line, people's heights, paint and touchlines through the solved camera; capped MEDIUM unless
independent paint confirms it); far-tile rows confirmed by foot point, buffered IoU for tiny boxes, a near pass that
fills in missed near players; the kit-change re-birth (ungated bug fix), size-aware kit votes, kit-consistent galleries,
slots by kit establishment, a small-fragment rule, provisional persons, the kit change rejoined offline, an impossible
side vetoing an offline connection (ungated bug fix); classical near-feet ball proposals, the hidden-ball gate at the
candidate's depth, a provisional lock, the decoder's still-lock audit and review rules; the stored-H sign fix
(`registration.homography_json`, ungated: Veo's 272 misoriented frames) and visibility's re-orientation of older
results; `tools/diagnose.py`, `check_ball_truth.py`, `check_pitch_truth.py` and the "Registered pitch (final)" overlay.
P7 (temporal accumulation of landmarks before the first anchor) was not done; a Hough-line deduplication was measured
and gave nothing, so it was not kept; `tactical.js` was not changed.

**A regression the measurements caught, and its fix.** The first GPU batch with every change (identified share on the
source window .761 -> .70, local tracks 357 -> 524; Veo .406 -> .379 with 30 frames of a 15 m identity jump) was
traced by ablation on the source window (each small-scene part switched off by monkeypatching in a scratch runner, the
same seed): the near pass merged into the joint suppression was the cause (off: local tracks 524 -> 336, identified
.70 -> .741, ID switches 32 -> 28, ball truth 10/11), while switching off the buffered boxes (.673, ID switches 42) or
the foot-point rules for tiny tile rows (.659, ID switches 49) made things worse. On 38 sampled frames the 640 pass
replaced the main passes' box on about three people per frame and doubled about two more (24-60 px), and at Veo
20.7 s it drew one box over two crossing players that a local track then followed. It now only fills in people of 60 px
or more that no main-pass row covers 30% of (`pipeline._fill_in`): the 148.0 s ball carrier (60 x 102 px, no main-pass
row) is kept. On amateur 3 0-60 s (gate closed) B-10 jumped 14.5 m in 1 s: the kit-change fix rightly merged the
duplicate B-11 into B-10, which exposed an older weakness of the offline connector (a fragment scored by the better of
its two sides: "before 0.09, after 0.92" was connected at 0.94). An identity whose sighting on either side is beyond
12 m/s of the gap is now not that fragment (`tracklets.SIDE_MAX_SPEED`, ungated: broadcast identical). A step that
would retire the short identity a track had before a kit change into the later one when the crops look alike was built
and tested, but its guards rightly refused all three cases on the source window (the later identity was on another
track meanwhile, or the earlier one was), so it was removed.

**Results** (final code; base = committed code; `after1` = the first batch with every change; scratch runs):

| window | metric | base | after1 | final |
|---|---|---|---|---|
| amateur 3 118-182 s (keyless) | calibrated frames | 0% | 100% | 100% (all MEDIUM: one automatic anchor at 118.2 s, 63 re-registrations, 1 paint anchor) |
| | identified share / drawn identified per frame | .761 / 0 | .70 / 14.9 | .804 / 16.2 |
| | identified 0-30 / 30-40 / 40-60 / 60-90 px | .81 / .83 / .88 / .91 | .79 / .82 / .84 / .85 | .86 / .86 / .93 / .86 |
| | local tracks / unresolved fragments / ID switches | 357 / 72 / 33 | 524 / 123 / 32 | 344 / 69 / 29 |
| | identities A / B (cap 11) | 13 / 11 | 11 / 13 | 11 / 13 |
| | ball truth (11 marks 146-150 s) | 0 hit | 9 hit | 10 hit (the miss is the hidden ball at 146.4 s, 102 px) |
| | pitch truth (marks at 127 and 148.07 s) | none (no calibration) | median 3.0 px / 0.76 m | median 2.4 px / 0.69 m, max 5.6 m |
| amateur 3 0-60 s (keyframes) | identified / new identities / B identities | .684 / 23 / 12 | .688 / 21 / 10 | .687 / 21 / 10 |
| | identity jumps > 12 m/s (frames) | 36 | 48 | 36 |
| Veo 1-31 s (keyframes; small scene from 12.8 s) | identified / local tracks / ID switches | .406 / 272 / 14 | .379 / 325 / 19 | .419 / 270 / 15 |
| | identified 60-90 px / jumps > 12 m/s | .418 / 1 | .399 / 30 | .495 / 1 |
| broadcast 0-60 s (keyframes) | `tools/exact_diff.py` vs base | | | identical people, identities, labels, pitch positions, ball, calibration, roster, tracks; one extra PITCH MIDFIELD log line |
| broadcast 0-60 s, no keyframes | calibrated frames | 0% | 100% | 100% (all MEDIUM) |
| | vs the keyframe run, per frame (median / p90) | | 0.51 / 1.03 m | 0.51 / 1.03 m |
| | at the user's 4 keyframes' clicks (median, max) | | | 0.24-0.71 m, 1.0-2.9 m |
| A_wrong recording 0-30 s | calibrated / identified / local tracks | 0% / .612 / 159 | 100% / .778 / 169 | 100% / .833 / 121 |
| | ball tracked | 6.1% | 6.1% | 6.1% |

The broadcast log line: the HEAD code run twice in two processes differs from itself the same way (108 frames of the
internal circle debug values, one more PITCH MIDFIELD line, every output identical), as do two runs of this code; the
detector's stored outputs and the camera motion are identical, so the variation sits below the stored precision
(sub-pixel detector differences feeding the camera-motion masks, the tracked paint and the circle's pixel set). It is
the shipped pipeline's run-to-run noise, not a change.

On the source window the pitch is MEDIUM everywhere: the far touchline passes 1.0-5.7 px (0.8-5.6 m: about 0.8 m per
pixel there) from the marks and the halfway line's far end is 4.5-4.7 px (2.5-3.8 m) off, the circle crossings and the
centre spot are within 2.2 px (0.15-0.6 m), the halfway line within 3 px (0.16 m); the near
touchline runs parallel to its paint about 1 m inside it (30-140 px that close to the camera): the assumed 68 m width,
where the anchors measure 69.6-70.6 m. Without keyframes broadcast stays MEDIUM too (the P agent's CPU replay had
predicted 56% HIGH; the GPU run has none).

**Timing** (RTX 5080; cs2.exe shared the GPU from about 01:30 on 2026-10-18, so the final batch's broadcast, Veo and
A_wrong jobs ran 2-3x slow and are not used): broadcast with keyframes 191 / 178 / 177 ms per frame online (base,
after1, a second run of the HEAD code): no cost. Online ms per frame base -> final: broadcast without keyframes 174 ->
155, amateur 3 0-60 s 166 -> 160, the source window 237 -> 241 (+2%: the near pass +12 ms detection, ball crops
+7 ms, far tiles -12 ms) plus 11 s offline instead of 4 (the pitch chains and the identity pass on a calibrated run);
Veo 196 -> 215 in after1 (the final code was only measured with the GPU shared); the A_wrong recording 120 -> 285 in
after1 (now calibrated, its far band runs the far tiles on almost every frame, and the ball tiles are scaled for the
low resolution).

**Not done or not solved.** P7 (accumulating landmarks over time before the first anchor); a learned field width
(the measured width is recorded, H keeps 68 m); HIGH confidence on keyless footage; team B over its cap on the source
window (13 identities: three tracks changed kit within the first 8 s and each left a 2-6 s team B identity, B-? on
screen; the identities they turned into were on other tracks meanwhile, so which labels were wrong is not known); the ball on the A_wrong recording (6% tracked, 93% UNKNOWN, unchanged); `tactical.js`
uncertainty styling. Tests: 543 pass (1 skipped). Nothing committed.

## Bouncy 2D markers: a stabilized replay stream (2026-10-18)

**The complaint.** In the 2D Match view the player markers of the amateur runs bounced, reversed and jumped. The user's
recordings were `bouncy.mp4` (the Veo 1-31 s run, source 2.29-15.01 s), `bouncy_2.mp4` (the amateur 3 118.2-182 s run,
121.43-139.64 s) and `bouncy_p.mp4` (broadcast 0-60 s, 32.09-50.26 s, the good reference). The runs are
`20261018-024007-2d6370`, `20261018-023229-8d0948` and `20261018-024712-3485a2`. All the measurements below are on GPU
runs of the same three windows in a scratch directory: the committed code (control) and the final code.

**Diagnosis** (five parallel measurements on the three runs and the recordings):

1. The recordings show the data faithfully: markers detected in every recorded frame match a port of tactical.js run on
   the result at the same time to 3 cm (median), and the renderer adds no timing jitter (refresh rate, phase, speed:
   < 5%).
2. The bounce is per-player foot-point noise: the box's bottom-centre through the frame's homography explains 99% of
   the high-frequency pitch variance on the amateur and Veo runs (93-96% broadcast). The detector's pixel noise is
   about the same on every camera (bottom edge ~ sqrt(0.4^2 + (0.011 h)^2) px), but the low cameras see the ground at
   about 7 and 5 degrees (broadcast 19), so a vertical pixel is worth 0.32 / 0.29 m against 0.11 m, along the camera's
   line of sight (8-12x the lateral metres per pixel): the markers bounce towards and away from the camera.
3. Most of it is box-extent events: the bottom edge jumps by more than 15% of the box height while the top stays (the
   box lost the legs, ends at a nearer player's head, splits into upper body and legs). They are 1.7-3.7% of the
   samples and 47-73% of the high-frequency variance, last 0.1-0.7 s and throw the player 4-10 m away from the camera.
   Occlusion and low scores multiply them; the amateur boxes overlap another person 4-5x as often as broadcast.
4. The calibration contributes 6-10% of the wobble and about 1% of the high-frequency jitter (smoothing the homography
   over 1 s changes nothing): between anchors the final H follows the camera motion exactly; at anchor frames it snaps
   (the forward/backward blend is ~50/50 everywhere but the anchor frame takes 100% of the anchor), and every player
   steps together (the median player up to 0.7 m on the amateur run, 1.1 and 2.3 m at Veo's two keyframes, 0.6 m on
   broadcast; jitter.py counts 11, 10 and 10 frames where most players jump together).
5. The renderer's 0.14 s centred average passes 89% of the 1-2 Hz bounce. It stops at every non-observed neighbour, so
   the timeline's interpolated samples are anchored on raw edge observations. The quarter-frame snap adds a sub-pixel
   30 Hz stutter, and the 'interpolated' ring flickered on every other display frame. The stored pitchSigma is not a
   calibrated uncertainty (it assumes 3 px; rank correlation with the real error 0.16-0.33).
6. Why broadcast is stable: a 19-degree view (a third of the metres per pixel), 4-5x fewer occlusions, 10-19x fewer box
   events.

**What was built** (README "2D replay stabilization" has the rules, `soccer/trajectory.py`'s docstring the constants):

- `soccer/trajectory.py` `stabilize()`, a pure function of the finished result, per internal identity and per
  anonymous `?` track. It takes:
  - the timeline's observations;
  - a pixel noise model fitted per run, carried to the pitch through the homography's Jacobian (the depth direction
    keeps its larger error);
  - box-event repair from the box top and a running or geometric height (flagged).

  It then runs a robust (iteratively reweighted), adaptive constant-velocity Kalman filter with a Rauch-Tung-Striebel
  smoother, using the future frames. The process noise rises only for a plausible change of motion. Gaps of up to 1 s
  are bridged (kind I), and an impossible jump (an identity switch) splits the trajectory instead of being glided
  across.
- `soccer/stabilized.py` builds `result.stabilized` per roster slot, with imputation run on the stabilized
  observations for P/U. It is failure-isolated: the block is left out, and the view draws raw, if more than 1% of the
  observations would be drawn raw or the median correction exceeds 2 m.
- Where it plugs in:
  - `pipeline.run` attaches the block;
  - `corrections.apply_result` rebuilds it, keeping the original noise model;
  - `tactical.js` draws it with no further smoothing, plus the **Debug positions -> Positions** chip (Raw /
    Stabilized / Both);
  - `imputation.estimate` gained an observations override.
- New tools:
  - `tools/jitter.py`: RAW vs STABILIZED jitter, responsiveness against raw, and common-mode jumps;
  - `tools/trajectory_bench.py`: the Metrica truth seen through simulated pole and broadcast cameras;
  - `tools/check_foot_truth.py`, with `tools/annotations/foot_positions.json`: 99 hand-marked ground contacts on the
    three runs.
- The raw positions, `result.timeline` and every analytic are untouched, and so is the ball.

**The calibration snaps.** A registration that crossfaded between anchors removed the snaps at the source, but on Veo
it flipped one tracklet decision (552 frames) and so changed raw positions and analytics. It was reverted (the
registration and tracking are the committed code's). The snaps are spread in the stabilized stream instead: each step
standing out of the steps around it gets a ramp sized to glide the median player at 0.2 m/s (1-6 s), additive where
ramps overlap.

Five rounds of implement / verify / review on the three runs and synthetic probes settled whose anchor decides the
direction. The measure was the keyframes' clicked landmarks carried by the camera motion.

- **User keyframes** are shown exactly as registered: arriving steps are spread before them, leaving steps after.
- **Automatic anchors** (paint, auto-midfield, re-registrations rooted at them) are grouped with their neighbouring
  steps, and the net step is spread before them. An outlier anchor composes to nothing and disappears. Spread after,
  they held a worse anchor for seconds: BC 43.2 s took A-02's hand mark from 0.22 to 0.65 m.
- **Re-registrations rooted at a user keyframe** (BC 24.6 and 34.4 s agree with their keyframes to 0.12-0.13 m, the
  blend after them is 0.41-0.46 m off) are trusted. They are spread like keyframes, but they are no bound for other
  ramps (bounding Veo's 18.8 s keyframe ramp would snap every player 2.3 m).
- A trusted anchor that disagrees with both neighbours is treated as an outlier.
- Automatic steps right after a keyframe or a trusted anchor go with its leaving step.
- A squeezed ramp turns the other way or snaps.

Synthetic probes that still overshoot are listed as known limits in the docstring; none occurs on the three runs.

**Four integration bugs found on the final GPU runs and in their review, fixed:**

1. `pipeline.build_report` attached the block before `run()` wrote `summary.pitch.registration`. In production the
   stabilizer therefore saw only the frames' own anchors: 11 of BC's 23, 3 of its 4 keyframes and none of the trusted
   ones. The stored block differed from one rebuilt from the saved result in 9228 BC samples (up to 0.89 m), and BC's
   turn rounding against raw was 0.17 instead of 0.14 m.

   The block is now attached in `run()` after that line, and it equals the block rebuilt from `result.json` on all
   three runs (0 samples differ). `test_pipeline_run` asserts the summary lists the registration when the stabilizer
   is called.
2. Handing a run its own noise model back (what a correction does) moved 1072 Veo samples by up to 0.25 m. The
   geometric-height fit has two passes and a given model was used in both; the model now carries `firstPass`, and
   both passes are replayed.
3. A ramp's step was measured at the stabilizer's observations, so excluding someone changed the median: excluding
   B-03 on the amateur run changed 7 ramps and moved 8048 samples of the others by up to 0.23 m. It is now measured
   at every person on the frame within 5 m of the lines, whatever their role. That also makes the largest step (the
   cancel test) include bystanders, which is stricter; no decision changed on the three runs.
4. The box-edge fits scale their biweight by the footage's noise level, a median over every observation, which a
   correction re-measured: excluding B-08 on the amateur run moved 36 samples of two others by up to 0.08 m. The model
   now carries it (`footage`: the level, the fits' span and the frame step).

The fitted path is bit-identical through 2-4 (the new keys aside). With them, eleven exclusions on the three runs
move no observed sample of any other identity, and the kept model reproduces the fitted block exactly. A refit instead
of the kept model would move 535 samples by at most 0.017 m (A-11 on the amateur run). The final blocks are the
final code's rebuilt from the final GPU runs' results. The stored block equals the rebuilt one; fixes 2-4 change the
stabilizer only, by at most 0.10 / 0.36 / 0.10 m. On BC the landmark sweep moves from 1.13 s worse / 8.40 s better
to 1.13 / 8.37 s.

**Results** (final code, from the final GPU runs. RAW is the drawing before the stabilizer (`tools/jitter.py`'s port
of tactical.js at 88be58f); STABILIZED is the block as tactical.js now draws it. Players, observed samples, 60 Hz
display. Each cell is RAW -> STABILIZED):

| window | residual RMS cm | standing residual cm | reversals >= 25 cm /min | jumps > 1 m /min | > 10 m/s events /min | frames where most players jump together |
|---|---|---|---|---|---|---|
| bouncy.mp4 (Veo 2.29-15.01 s) | 44.8 -> 2.4 | 21.6 -> 2.6 | 58.5 -> 0.5 | 10.1 -> 0 | 45.4 -> 0 | 7 -> 0 |
| bouncy_2.mp4 (amateur 121.4-139.6 s) | 26.8 -> 3.5 | 17.6 -> 3.1 | 55.6 -> 0 | 0 -> 0 | 22.9 -> 0 | 0 -> 0 |
| bouncy_p.mp4 (broadcast 32.1-50.3 s) | 10.1 -> 2.4 | 7.2 -> 1.9 | 1.7 -> 0.3 | 0 -> 0 | 0.3 -> 0 | 6 -> 0 |
| amateur 118.2-182 s | 27.4 -> 3.9 | 16.1 -> 2.7 | 53.4 -> 0.06 | 0.9 -> 0 | 24.1 -> 0.06 | 11 -> 0 |
| Veo 1-31 s | 43.2 -> 3.4 | 20.4 -> 2.3 | 74.1 -> 0.3 | 7.9 -> 0 | 52.0 -> 0 | 10 -> 0 |
| broadcast 0-60 s | 9.6 -> 2.7 | 6.6 -> 1.8 | 2.8 -> 0.13 | 0 -> 0 | 0.4 -> 0 | 10 -> 0 |

Against the hand-marked feet. "Raw" is the stored position; each cell is median / p90 / max in metres. Normal boxes
are measured against a reference through the frame's own homography, which the snap spreading moves away from by
design.

| run | class | raw | stabilized |
|---|---|---|---|
| amateur | all 45 | .52 / 3.61 / 4.71 | .39 / .77 / 1.07 |
| | normal 26 | .35 / .57 / .77 | .39 / .69 / 1.07 |
| | box events 13 | 3.09 / 4.60 / 4.71 | .37 / .81 / 1.05 |
| Veo | all 36 | 3.82 / 9.37 / 12.13 | .42 / 1.33 / 1.67 |
| | normal 17 | .34 / .57 / .72 | .38 / .95 / 1.67 |
| | box events 18 | 7.96 / 10.58 / 12.13 | .47 / 1.22 / 1.41 |
| broadcast | all 18 | .19 / .74 / 1.41 | .22 / .41 / .63 |
| | normal 15 | .17 / .34 / .38 | .20 / .33 / .63 |
| | box events 2 | 1.38 / 1.41 / 1.41 | .20 / .25 / .26 |

The Metrica bench (truth known, game 2 held out, 3 seeds, 4 windows of 300 s, truth glitches excluded). RMSE in
metres for raw / the old renderer / Savitzky-Golay 0.5 s / RTS with fixed noise / the stabilizer; the bench has no
anchors, so the trusted-anchor rules are not exercised:

| scenario | camera | RMSE: raw / renderer / SG 0.5 / RTS / stabilizer | sprinting | turn RMSE | standing step p50 cm |
|---|---|---|---|---|---|
| base | pole | .83 / .70 / .69 / .43 / .13 | 1.02 -> .16 | 1.10 -> .33 | 23.3 -> .54 |
| base | broadcast | .20 / .17 / .17 / .15 / .06 | .24 -> .10 | .26 -> .19 | 7.7 -> .50 |
| calibration steps | pole | 1.02 / .91 / .91 / .70 / .42 | 1.45 -> .60 | 1.32 -> .61 | 23.5 -> .76 |
| calibration steps | broadcast | .26 / .24 / .23 / .22 / .14 | .34 -> .22 | .32 -> .24 | 7.8 -> .57 |
| every event kind + drift | pole | 1.27 / 1.15 / 1.15 / .85 / .57 | 1.64 -> .69 | 1.90 -> .84 | 23.6 -> .85 |
| every event kind + drift | broadcast | .31 / .28 / .28 / .26 / .17 | .37 -> .19 | .42 -> .29 | 7.8 -> .58 |

The lag stays within a few ms on every camera.

**What it costs on the broadcast run.** The tool measures turns and stops against the raw observations, and on the
amateur runs the raw observations' local noise (0.18-0.33 m) dominates those numbers. On broadcast:

- turns are rounded by 0.14 m median (0.06 m where no box event is near; raw 0.04);
- stops by 0.05 m (raw 0.02);
- normal-box foot marks are 0.03 m worse in median.

That is the price of the 2D view holding still and never snapping.

**Identity and tracking preserved.** The final code's GPU runs of the three windows are identical to the control's
apart from timing fields: 2.06M / 0.82M / 2.59M result values, plus `identities.json`, `registration.json` and the
event log byte-identical. The stabilizer never mixes identities.

**Overhead** (the final GPU runs; nothing else on the machine):

| run | block total | of which the stabilizer | share of the run | result.json growth | peak allocations |
|---|---|---|---|---|---|
| amateur | 3.4 s | 1.7 s | 1.1% | +2.3 MB (7.8%) | ~86 MB |
| Veo | 1.1 s | 0.7 s | 0.9% | +1.1 MB (9.0%) | ~75 MB |
| broadcast | 3.9 s | 1.6 s | 1.7% | +1.4 MB (3.9%) | ~89 MB |

**Not done or not solved.**

- Analytics stay on the raw positions; the spec's first stage only asked for the replay.
- The registration's own snaps are not fixed at the source.
- Veo's A-10 during its 1.5 s hip-cut spell (6.6-8.5 s): the repaired boxes put the player 0.5-1.4 m beyond the
  mark (raw 7-12 m). The two full boxes inside the spell, right in raw (0.00 and 0.41 m), are outvoted by the repairs
  and shown 1.4-1.7 m off; they are the stabilized normal-box maximum on Veo.
- The amateur run's A-03 is split at 160.37 s, where the raw identity jumps 18 m (a tracker identity switch, left to
  the tracker).
- Tests: 653 pass (1 skipped). Nothing committed.

## Team shape bench: correlated camera noise (2026-10-18)

**The artifact.** A coaching audit found that `tools/team_shape_bench.py --camera` added independent noise of 0.5-2 m
to every 5 Hz sample. Consecutive samples then differ by metres, which `analytics._breaks` (faster than 12 m/s over at
least 2 m) takes for registration jumps, and `TeamShapeData._fill` drops the short runs between them as glitches:
116,715 / 118,776 positions of team A / B on game 1. The 60 m window held 8.4 of the 10 outfield players on average
(9.0 in play) and the whole team 60% of the time (70% in play); TeamShape kept 4.3 of them. Every camera number of the
2026-10-17 section and of the README came from that noise model.

**The change.** `camera_error` makes a player's error sigma * z, with z a unit AR(1) process (time constant
`--noise-tau`, 1.5 s by default, exact across the time out of view) and sigma ~ U(0.5, 2) m per sighting (drawn again
after 2 s out of view). Each sample's error keeps the old distribution; only its persistence changed. `--noise-tau 0`
reproduces the old runs exactly (the same random numbers: 4.34 / 4.23 players kept, the same spike counts and errors).
Also new: `--max-missing`, `--seed`, and per team what the window held (`inView`), the exclusions and the camera
run's own formation periods. Three tests in `tests/test_tools.py`. The tracking, `team_shape` and `analytics` code is
unchanged (one comment rewritten).

**Why 1-2 s.** On real results TeamShape drops as spikes 0% of the observed outfield positions of the broadcast 0-60 s,
1.9% / 4.7% of amateur 3 118-182 s and 12-13% of Veo 1-31 s (about two players observed per sample). On game 1's first
20 minutes with the 60 m camera: white noise 49% / 51%, AR(1) with 1 s 2.3% / 2.6%, 1.5 s 0.7%, 2 s 0.3%.

**Re-measured on game 1** (game 2 not run: it stays held out). Team A / B, all samples (in play in brackets):

| noise | kept by TeamShape (in the window: 8.39 / 8.35) | complete | >= 8 kept | spikes dropped |
|---|---|---|---|---|
| white (the old bench) | 4.34 / 4.23 (4.53 / 4.44) | 0.1% / 0.1% | 4.6% / 4.3% | 116,715 / 118,776 |
| AR(1) 1 s | 8.18 / 8.12 | 46.9% / 47.0% | 73.0% / 73.2% | 5,123 / 5,789 |
| AR(1) 1.5 s | 8.31 / 8.27 (8.88 / 8.90) | 54.8% / 55.8% (63.4% / 62.9%) | 74.0% / 74.5% (82.0% / 83.4%) | 1,333 / 1,458 |
| AR(1) 2 s | 8.34 / 8.30 | 56.7% / 58.0% | 74.3% / 74.7% | 454 / 512 |

The window held the whole team 60.3% / 60.7% of the samples (69.9% / 68.9% in play). A 45 m camera (1.5 s): 7.2 in the
window, the whole team 34% / 36%, complete 31% / 34%.

Width and depth against full coverage by status, 60 m, 1.5 s (share of samples, median and 90th percentile absolute
error, mean error, metres):

| status | width | depth |
|---|---|---|
| complete | 54.8% / 55.8%; 1.12 / 1.18, 2.90 / 2.99; +0.09 / +0.11 | the same samples; 1.16 / 1.17, 2.98 / 3.09; +0.61 / +0.50 |
| reliable | 7.8% / 9.0%; 1.04 / 1.23, 2.71 / 3.04; +0.16 / +0.27 | 1.6% / 1.1%; 1.14 / 1.27, 3.19 / 3.34; +0.52 / +0.64 |
| partial | 23.2% / 18.9%; 2.25 / 1.85, 10.75 / 8.80; -3.52 / -2.54 | 29.4% / 26.8%; 8.97 / 7.74, 20.07 / 20.21; -9.73 / -9.14 |
| unavailable | 14.2% / 16.3% | 14.2% / 16.3% |

Over 1 to 2 s and the 45 m camera: complete and reliable 90th percentiles 2.7 to 3.4 m; partial widths 2.0 to 3.8 m
short on average (90th percentile 8.0 to 11.2 m), partial depths 7.0 to 10.7 m short (18.6 to 22.6 m). The white noise
had put partial depths at 3.4 to 3.7 m short: it removed players at random, while a window removes one end of the
team. Complete depths read 0.5 m long on average: the noise widens the extremes.

Formations (the shipped rule, at most one missing; 60 m, 1.5 s, camera seeds 0-3): 64-67% of the windows labelled
(white noise: 0.9-1.3%), 85-94% (A) and 65-73% (B) of them as with full coverage; 1 s and 2 s alike (A 92% / 83%, B
68% / 73%). Read against the base rates: with full coverage team A is 4-4-2 for 96% of its labelled time, team B
3-5-2 for 60%.

**FORMATION_MAX_MISSING stays 1.** Paired runs with 2 (16 team runs: 60 m 1.5 s seeds 0-3, 1 s and 2 s seed 0, 45 m
seeds 0-1):

- 40 to 117 eight-player windows per run. Their stabilized label agreed with full coverage 54-94% of the time (median
  71%; nine-player windows 55-100%, median 74%), but their own best fit only 21-38% (nine players 31-52%, ten 47-62%):
  the label is mostly lent by the neighbours.
- Labelled windows +3.5 to +5.2 points at 60 m, +8 to +10 at 45 m. Agreement up in 13 of 16 runs (by up to 15 points,
  mean +4.3), down in 3 (by at most 1 point). Much of the gain is on windows labelled anyway (team B's ten-player
  windows +4 to +14 points at 60 m): fewer coverage gaps cut the stabilization short (a gap ends a Viterbi run).
- Label changes per match at 60 m: A 0-2 -> 1-3, B 2-5 -> 4-9 (full coverage: 3 and 12).
- Time called clear (seeds 0 and 1): A 7.8% -> 4.2% and 6.3% -> 2.7%, B 5.6% -> 3.8% and 7.9% -> 5.1% (a clear period
  needs every window complete, and the bridged periods are longer).
- The real results (broadcast 0-60 s, amateur 3 118-182 s, Veo 1-31 s) do not change: the broadcast has all ten, the
  others fewer than eight.

The windows a second missing player admits carry little evidence of their own; the gain is continuity, which a
stabilization run that crosses a short coverage gap could give without labelling them. Identity errors, which the bench
does not have, are not measured either. The constant stays; its comment and the README carry these numbers.
Tests: 669 pass (1 skipped). Nothing committed.

## Possession tracking: who has the ball, and when it was lost (2026-10-19/20)

**The request.** Possession tracking from the 2D view, judging as accurately as possible who has the ball and when it
was lost, so that a later change to the coaching report can tell when either team is attacking or defending.

**Plan first.** The data roles, the conventions, the metrics and the targets were written down before any engine code
(PREDECLARED, 2026-10-19). TUNE: Metrica sample game 1, with a possession truth derived from its events, on full
tracking and through a simulated broadcast camera. DEV: hand labels of broadcast 0-60 s (120 points every 0.5 s).
HELD OUT, read once after the freeze: Metrica game 2, and adjudicated hand labels of broadcast 42.4-222.4 s (two
labellers, 360 points), amateur 3 (118.2-182 s) and Veo (1-31 s). Convention: a team keeps possession from its control
through its passes, shots, clearances and loose balls until the other team controls the ball or the ball goes out of
play; a change is timed at lostAt (the loser's last touch) and wonAt (the winner's first control).

**The truth** (scratch, not in the repository). The Metrica events give the possession on every frame: the ball is a
median 0.11 m from the event player (91.6% of 2513 events within 1 m), 99.4% of game 1's 320 changes follow a losing
action, and untracked ball frames match the truth's dead spells (99.1% of them are dead). 79 of the 320 changes go into
a one-touch possession (a first-time interception Metrica codes as a gain), so changes are scored with and without
them. The camera simulator was fitted to the ball audit of the broadcast run (tracked / missing .57 / .42 against
.56-.64 / .36-.44, pitch-vs-image agreement .94 / .88 against .92 / .87, player errors median 0.74 m); it has no
identity errors, replays or registration collapses, and nothing like the amateur or Veo clips.

**Stage A** (game 1 only): evidence that a team is at the ball (near on the pitch and in the image), a two-state
Viterbi with a 0.15 s cost per change and its posterior as the operating point (named where >= 0.8), restarts after
out of play, still balls, long unseen stretches. Two first ideas were measured and replaced: no evidence on fast
balls (accuracy .961 against .972, F1 .640 against .692) and control spans with hysteresis and a minimum hold
(F1 .612 / .571 on full tracking / the camera against .692 / .609; a draft measurement the final bench no longer
reproduces).

**Stage B** (the DEV labels and the runs' own data only): nearTrack is evidence only where the ball tracker's first
pass followed the same ball (the nearTrack person stood more than 2 m from the final ball on 50-66% of the broadcast's
interpolated frames); the carrier must be one player near on both metrics (BC60's 14 labelled carries: one carrier in
13, stage A 8; 16 of 16 random control frames right on the video); a person without a team is evidence for nobody; a
goalkeeper without a team takes the goal on his side; a camera cut is a gap. Game 1's headline numbers unchanged to
three decimals. Frozen (sha256 d95f29e1..., version possession-2) before any held-out label was read.

**Held out** (scored once with the frozen engine and settings=None):

| target | measured | verdict |
|---|---|---|
| Metrica full tracking: accuracy >= .95 at coverage >= .85 | .968 at .931 | met |
| Metrica full tracking: changes F1 >= .80 within 2 s | .685 (P .905, R .551); .773 without one-touch | missed |
| simulated camera: accuracy >= .90 at coverage >= .65 | .962 at .886 (seed 1 .968 at .888) | met |
| simulated camera: changes F1 >= .65 | .582; .628 without one-touch (seed 1 .605 / .654) | missed |
| broadcast 42-222 s: accuracy >= .90 at coverage >= .60 | .965 at .912 (.961 at .908 without the 42-60 s overlap) | met |
| broadcast 42-222 s: change precision >= .70 | .30 (3 of 10) | missed |
| amateur 3, Veo: >= .80 where named, else abstain | both abstain (small scene on 60% / 61% of frames) | met |

The 42-222 s labellers agree on 315 of 316 team points, and always answering the clip's main team scores .820. All 10
engine changes on it were checked on the video: 3 right, 1 more than 2 s off (a goal kick dated 3 s early), 6 wrong.
The causes: the ball path on an object beyond the touchline next to a steward (93.6-103.8 s, a false out-of-play spell
and a false pair of changes), the interpolated ball drawn at a defender's feet in a dribble (139.4 s, a false pair),
and a foul stoppage with the ball left rolling read as live play (199.45-213.77 s, two false changes). Dead balls: 0 of
41 dead points found. With its gates off amateur 3 would have scored .544 against the .567 of a constant answer.

**Integration** (this entry's last step; the engine was not changed: its sha256 still matches the freeze note).
- The record of every run was checked against the output contract and against what the page assumes (time-ordered
  contiguous runs whose t0/t1 are the times of f0/f1, changes[].frame = the frame of wonAt, internal ids only, plain
  JSON): all four runs pass. On scratch server roots (copies of the page files and of the four results, the worktree's
  server.py, the real videos read-only) /api/possession returned compute() byte for byte for all four, the cached
  answer again on the second request, 409 for another run, 403 for a foreign Host or Origin, and 503 on this route
  only with the engine made unimportable (the page, the board and /api/result still answered).
- In the browser: the strip, the shares and the live line during playback on broadcast 0-60 s and 42-222 s; a marker
  click pauses on the won frame + 0.2 frame (139.4067 s for the 139.4 s change); the ring on the board (a pixel diff
  around the carrier's marker) and the ellipse at the carrier's feet on the tracking video (Player boxes on); a team
  correction of the carrier B-08 at 140 s recomputed the record (the false 139.4 / 140.53 s pair disappeared, the run
  became team A with the new identity as carrier) and removing it gave back the record byte for byte; the one-line
  reason on amateur 3 and Veo; a 375 px wide page without horizontal scroll; no console error; Analytics, the
  Coaching report (a report generated, its Show on 2D jump) and the tracking overlay as before.
- Glue fixed in possession.js (the engine's output differed from what the page assumed):
  1. Right at a change the posterior is usually under the operating point: 9 of the 10 won frames on 42-222 s are in
     an 'unknown' run, so a click on "Team B won the ball" showed "Possession unknown" and no ring. On the won frame
     the line now names the change ("Team B won the ball · B-08 won it") and the ring marks the winner, dashed.
  2. The engine's unknown runs are always 'low' and often 'estimated'; the line said "Possession unknown · estimated ·
     low confidence". It now says "Possession unknown".
  3. Two changes 1.1-1.2 s apart are 3-4 px apart on a 640 px strip; the later 14 px marker covered the earlier one's
     centre (3-4 px of it clickable). Close markers now share the space between them (9-10 px each there), each
     triangle staying on its own time.
- compute() on 42-222 s: 0.42 s for 5400 frames (features 0.37 s, decoding 0.05 s); the server's first request 1.52 s
  (reading the 111 MB result 0.98 s), the cached one 0.03 s. Broadcast 0-60 s 0.14 s, amateur 3 0.10 s, Veo 0.04 s.
- README: a "Possession" section (method, conventions, the record, abstention, every measured number, what it cannot
  do), cross-references in Coaching report, Evaluation tools, Tests and Limits.
- Tests: 1068 pass (4 skipped): test_possession.py 16, test_possession_server.py 7, test_possession_panel.py 19 (two
  new: the won frame, close markers).

**Open.** Change detection on real footage (two of the three failure causes above come from the final ball path: an
off-pitch object and an interpolated ball drifting to a defender; the third, stoppages with the ball in view, from the
dead-ball rules); dead balls with the ball in view; a keeper holding the ball; the coaching report does not read the record yet; the
Metrica truth, the simulator and the hand labels are in scratch, not in the repository, so the bench needs them
supplied. Nothing committed.

### Round 3: possession-3, the verifiers' findings (2026-10-20)

**What changed.** Six rules, each measured alone on the dev data (game 1, BC60 and now BC180, whose labels were read at
the possession-2 test): a change starts only on a run of one player's evidence with an observed sighting on both
metrics; a ball beyond a line is out of play only if it got there plausibly and no other ball is in play (the steward's
ball of 93.6-103.8 s); a stoppage is also both teams slowing down (the 75th percentile of the players' speeds under
2.2 m/s for 3 s) while the ball is unseen, still or unplayed; the winner is the new team's player with the most evidence
over 0.5 s; changes are rated high / medium / low (back-and-forth pairs low); one-touch possessions are reported under
both conventions. Frozen as possession-3 (sha256 c5ea6461...).

| | possession-2 | possession-3 |
|---|---|---|
| game 1 full tracking (tuned): accuracy / coverage, F1 +-2 s | .972 / .928, .692 | .972 / .924, .689 |
| game 1 camera (tuned) | .961 / .887, .609 | .968 / .871, .607 |
| game 1 dead recall, full / camera | .43 / .71 | .93 / .87 |
| broadcast 42-222 s (dev from here on) | .965 at .912; 3 of 10 changes right; 0 of 41 dead points | .977 at .975; 4 of 6; 9 of 41 |
| game 2 (read once after the freeze: seen from here on), full / camera F1 | .685 / .582 | .685 / .577; dead recall full .278 -> .828 |

The page hid the low changes by default ("Show uncertain changes", hollow when shown), gained the key row and reserved
110 / 46 px under the board. The critic then found that results of 30 frames or fewer raised (np.convolve 'same'),
that the docs still described possession-2, that the coaching report says the tracking does not record who had the
ball while the 2D view now shows it, that out of play had never been checked on a real ball over a line in PitchIQ
footage, and that results without ball.origin got no changes; on game 2, medium and low were equally precise (camera .62
against .64), so hiding low changes hid real ones. Tests: 1080 pass (4 skipped).

### Round 4: possession-4, frozen and tested on held-out data (2026-10-20/21)

**Plan** (PREDECLARED_v4, before any change). Dev: game 1, BC60, BC180, AM, VEO. Seen: game 2, read once more after the
freeze as a regression check. Held out: Metrica sample game 3 (downloaded 2026-10-20, Metrica's EPTS release, never
looked at by anyone) and a new blind-labelled PitchIQ run of La Liga 03:42-05:03 (BC303). Targets unchanged. A rule
must be general and must not make game 1 worse.

**Ten fixes** (each can be switched off; with all of them off the records equal possession-3's on the four run copies
and the 120 game-1 windows, version aside). Before -> after on dev data:

| fix | possession-3 | possession-4 |
|---|---|---|
| 1 short results | 1-30 frames raised ValueError | a valid record; 31+ frames byte-identical |
| 2 a ball held in play is no stoppage | game 1 in-play time called dead 86.6 / 81.6 s (full / camera); 26 of 54 synthetic holds dead | 37.6 / 48.2 s; none (set pieces at a clip's start still dead); dead recall full .934 -> .929 |
| 3 a keeper's hands (synthetic) | change dated at the release (10.0 s), the hold dead or the shooting team's | change at the catch (3.9 s), his team throughout |
| 4 a ball knocked past a defender | 6 of 60 synthetic dribbles with 6 false changes each | 0 of 60; game 1 full F1 .6885 -> .6911 |
| 5 no ball.origin (BC180 with the field removed) | 0 changes, coverage .743 | 6 changes (4 right), .908, a note |
| 6 first-pass check, image rows doubled (synthetic ball boy 3-4.5 m beyond the line) | a false 7.7-7.9 s dead spell | none |
| 7 a goal kick after 3 s unseen (synthetic) | no dead ball | dead 0.3-5.5 s; camera dead recall .869 -> .876 |
| 8 a 6 s replay between two cuts (synthetic) | named for the other team, 2 false changes | unknown, no change |
| 9 an identity on another team's player (synthetic) | no change found | both changes found |
| 10 change confidence | high / medium / low (medium 67%, low 74% right: not ordered) | high / uncertain |

The UI default changed with fix 10: every change is shown, the uncertain ones hollow (hiding them would hide about 4 in
10 of the right changes on game 1). Fix 9 reads only a frame's own team: the identity tracker's per-frame kit readings
looked like swaps on BC60 / BC180, but on the video (93.0, 121.5, 198.5 s) the identities were right and the readings
misreads at crossings; PitchIQ never writes a frame team against the roster, so the rule is inert on real runs. Frozen
as possession-4 (sha256 75b65939...).

**Game 3's truth** was built before the freeze with counts and consistency checks only (no engine output). New readers
for the EPTS format (metadata XML with the players' frame ranges, one tracking line per frame, JSON events); they refuse
a document type declaration, another frame rate or pitch, a first column that is not the goalkeeper, an unknown player.
The tracking has Player6 and Player13 swapped over the whole second half (all 187 of their tracked second-half events
sit at the other one's column, a median 0.03 m from the ball against 18.99 m); EPTS_SWAPS corrects it. Checks, game 1 in
brackets: ball within 1 m of the coded player 91.5% (91.6%), changes preceded by a losing action 98.6% (99.4%),
untracked ball frames dead 99.3% (99.1%). 367 changes, 101 one-touch possessions, 67.9% in play. Games 1-2's truth and
simulations are byte-identical to before (270 sha256s).

**Held out, scored once each with the frozen engine:**

| target | game 3 | BC303 (broadcast 222-303 s) | verdict |
|---|---|---|---|
| accuracy at coverage (full .95 at .85; camera .90 at .65; broadcast .90 at .60) | full .964 at .942; camera .956 at .881, seed 1 .959 at .883 | .949 at .924 | met |
| changes (F1 .80 full, .65 camera; broadcast precision .70) | F1 .687 (95% .637-.739); .615 (.569-.666), .586 (.530-.646) | precision .50 (1 of 2), recall .25 (1 of 4) | missed |
| dead ball | recall / precision .897 / .965 full, .880 / .954 and .864 / .955 camera | 28 of 55 dead points, 28 of 30 engine-dead points right | (no target) |

- Confidence on game 3: 'high' 489 of 564 right (.867; .83-.92 per variant) against 444 of 474 (.937) on game 1, so
  game 1's "94%" overstated it; 'uncertain' 452 of 643 (.703), as on games 1-2. The two levels are ordered.
- possession-4 against possession-3 on game 3 (the bench's ablations, same run): F1 +.001 to +.007 (below the noise);
  in-play time called dead 85.4 -> 60.1 s (full), 91.6 -> 78.0 s (camera). Who lost the ball on the camera is .717,
  below possession-2's .741, as on game 2.
- Game 2 (seen): within .012 of possession-3 everywhere (full F1 .681, camera .581 / .617); but dead-ball errors rose on
  full tracking (+4.2 s) and with the ball shown (+23.9 s), so "they fall on every variant" holds on games 1 and 3 only.
- BC303 measures possession-3's rules: none of the ten fixes fires there (no keeper's hands, knock-past, kit frame or
  cut). Every engine output was checked on the video: the B->A change at 247.93 s is right (0.14 s early); the A->B
  change at 265.43 s is false, inside the throw-in stoppage 253.02-273.7 s, after the path left the real ball (in a
  player's hands behind the far touchline) for stray detections on the pitch; the out-of-play spell 253.07-262.17 s ends
  11.5 s early because an interpolated ball within 1 m of the line counts as back in play; the stoppage 268.97-274.6 s
  names the wrong restart team (from a weak detection at a Barcelona player's feet); the stoppage 283.15-290.4 s is
  missed: the path followed the real ball 1-8.7 m beyond the far touchline (286.43-288.67 s) but the excursion check
  threw it out (the first pass more than 2 m away on 94% of its frames).

**Verification.** Every frozen number reproduces exactly (five bench JSONs, 0 differences in 9,320 values), and the
logs show no engine or bench run on game 3 before FROZEN_v4 existed. Adversarial synthetic scenes broke four fixes
(each confirmed by switching the rule off): fix 2, the fouling team holding the ball until the free kick (two false
high changes, no dead ball) and team-mates passing during a stoppage; fix 7, a ball unseen 1-3 s after a reception with
the players walking (a false 3.7-5.2 s stoppage); fix 3, a cross along the keeper's image column at 8-12 m/s or an
attacker holding the ball 1.5-3.5 m behind him (a false pair); fix 8, a replay during an out-of-play stoppage with the
live ball beyond the line afterwards (a false high change). Fragile: fix 6 (a first pass 1.1-2 image metres off on a
real out of play turns it into a stoppage 0.5 s later) and fix 9 (a 0.7 s kit misread gives a false pair). Engine
mutants: 47 of 111 caught (42 of the survivors change some record: test gaps, most in fixes 2-4 and possession-3's
stoppage and abstention rules); server mutants 7 of 10.

**Docs and polish** (the last step; the engine is unchanged, sha256 still 75b65939...):
- tests/test_possession_panel.py: the four tests that asserted possession-3's hidden-by-default UI now assert the v4
  default (every change shown, uncertain ones hollow, unticking hides them, ticking shows them again); a new test
  draws a possession-4 record ('uncertain' hollow, its tooltip and won line saying so). 24 tests; 8 of 8 UI mutants are caught,
  among them the two the verifier found uncaught ('uncertain' drawn solid, the tooltip word gone) and the old hidden
  default.
- possession.js: the header's numbers (5 of the 6 won frames of 42-222 s in an unknown run under possession-4, not 9 of
  10; game 3's 87% / 70% beside game 1's; the held-out broadcast's 1 of 2), and two code comments with the same stale
  figures. index.html: the option's tooltip ("untick to hide them").
- tools/annotations/possession/*.json: a final CRLF (the JSON is identical).
- README: the Possession section describes possession-4 with the dev, seen and held-out numbers, the confidence, what
  it cannot do and how to reproduce; Tests and Limits updated.

**Phases for the coaching report** (the v3 critic's rules on possession-4; game 1 in 180 s windows, games 2-3 from the
test role's captured records in 300 s windows; BC303 is spent, descriptive only). R2 = every named frame minus
[min(lostAt, wonAt - 2 s), wonAt + 2 s] around every engine change:

| | every named frame (R0) | R2 |
|---|---|---|
| game 1 full / camera / seed 1 (dev) | .972 at .941 / .969 at .882 / .968 at .882 | .977 at .740 / .977 at .730 / .979 at .724 |
| game 2 full / camera / seed 1 (seen) | .968 at .933 / .965 at .888 / .972 at .884 | .974 at .756 / .979 at .744 / .984 at .751 |
| game 3 full / camera / seed 1 (held out) | .964 at .942 / .956 at .881 / .959 at .883 | .975 at .738 / .976 at .722 / .977 at .730 |
| BC180 (dev) | .977 at .975 | .996 at .874 |
| BC303 (held out, spent) | .949 at .924 | .956 at .857 |

Dead points named as a team's possession: BC180 11 of 41 (R2 8), BC303 26 of 55 (R2 17).

**Open** (in the frozen engine; a new version would need its own freeze and a new held-out test): the out-of-play veto
of a real ball beyond the line when the first pass is elsewhere (BC303 286.4-288.7 s; about 20% of BC303's 0.5 s windows
of tracked ball have the first pass more than 2 m away); an out-of-play spell ended by interpolated frames; the restart
team from a weak detection; the four adversarial failures above; the engine docstring's lines that are now out of date (out
of play "never checked on a real ball", game 3 "held out", knock_past()'s "the next touch must be the other team's"
where the code takes any touch of the other team within pass_after); a frame with a NaN time making a run out of time order; --heldout accepting any
existing note. The change targets are missed on three games and three versions; the coaching report still says the
tracking does not record who had the ball.

Tests: 1103 pass (6 skipped; test_possession.py 35, test_possession_panel.py 24, test_possession_truth.py 11, its 2
real-data tests skipped without the Metrica folders). Nothing committed; the demo checkout untouched.

## Coaching v2: the report adapts to age and playing level (2026-10-21/22)

**Why.** The first coaching report (commit `1262590`) failed its critic's review, with four blockers:
- pooled strengths counted time nobody had compared;
- the gaps between units were measured while the units were interleaved;
- compactness was credited whatever the phase of play;
- some sentences were false.

It also judged every team against one professional match. Its levels changed only the wording: the observation ids,
counts and drills were identical across levels on all 8 team-runs.

**What was built** (DESIGN v2 in the session scratch; worktree `.claude/worktrees/coaching-v2`, branch
`claude/coaching-v2`, uncommitted):
- **A coaching profile per team:** age, level 1-7, format, federation, field size, formation, style and roles.
- **Three cached tiers:** measurement, role and interpretation.
- **Six context checks** (D1-D6) with guardrails, explained by level.
- **Suggested areas** where one team-mate could have reached in about 2 s.
- **Evidence moments** on their exact frames.
- **A 14-drill library** behind an age safety filter.
- **Progression per squad.**
- **A seven-section page and PDF.**
- **A tool for coaches to mark who had the ball** while watching.

Waves 0-7 each wrote their own report with measurements; the README section "Coaching report" summarises them.

**Wave 8, the evaluation** (`tools/coaching_eval.py`, scratch `coaching2/build/eval/`):

| What | Result |
|---|---|
| Golden check: `result.json`, `/api/analytics`, `/api/team-shape`, `/api/possession` on BC60, BC180, BC303, AM and VEO | 170 outputs and 8 frozen modules byte-identical to the committed code |
| Synthetic counterexamples: 30 scene variants x levels 1-7 through `store.measure` and `coaching.interpret` (`tests/test_coaching_counterexamples.py`) | 380 checks pass, 0 fail, 1 known open defect (DETECT: level 1's reframing of D3 gives the identity-crossing scene a "try this"; DESIGN 13.1 says at most a review); 0 report invariants broken |
| DESIGN 13.4 #4: planted unsupported advances (16 spells) at level 4 | 6 of 6 scenes give a work item; covered twins and the professional-like scene get 0 criticisms at any level |
| #5: relation and opponent claims of the planted D3 finding | L2 1, L3 2, L4 3, L5-L7 4 (+2 over L2-L3) |
| #12: drill picks on every scene report | 210 picks, 0 trains another check, 0 heading drills at U11 or younger; 196 age/level/drill pairs the filter allows at U6-U11, 0 with heading |
| Same video, different levels: 30 scene variants x 14 profiles, 4 re-measured (`tests/test_coaching_levels.py`) | obs / roles bytes identical; enabled checks and verdicts match DESIGN 2.3 (written out independently); level-pair trigram Jaccard at most .368 outside 2 known defects (TEXT: the D6 describe sentence repeats, 184 scene pairs at .55-.76; DETECT: findings re-pick their moments when a level judges a different failing set, 3 families) |
| Real run copies BC60, BC180, BC303, AM, VEO (+ AM and VEO with a SYNTHETIC coach-marked phase from the blind possession labels), both teams, 14 profiles: 196 reports | 0 invariant problems; obs / roles identical under all 14 profiles; work items only on BC180 Barcelona's attacking depth at levels 1-2; every other criticism a review with a failed evidence guardrail; no suggested area drawn; amateur and Veo "not assessed" for every phase-dependent check without marks |
| StatsBomb open data, 4 Euro 2020 matches with 360 frames (8,576 carrier events with a visible opponent) | under pressure: nearest opponent median 2.12 m, 84% within PitchIQ's 5 m; not under pressure: 5.69 m, 44% within 5 m (precision .32, base rate .20); "nobody within 5 m" agrees with not-under-pressure on .94 (.91 in the defending third). Youden's best threshold is 3 m (J .43 vs .40 at 5 m) |

**Label sheets for a coach** (`tools/annotations/coaching/labels/` and scratch `build/labels/`):
- MOMENT: 266 moments (broadcast 200, amateur 47, Veo 19; stratified by check outcome, plus 30% random times).
- ROLE: 60 players.
- ADVANCE: 8 spells, from PitchIQ's own spells and a role-free rule.
- ZONE: 48 diagrams (45 from Metrica game 1, 3 from the clips).

The keys are hashed in `manifest.json`. `coaching_eval.py --labels DIR` scores them and records the sha of the frozen
rules.

**Not measured yet.**
- **Needs filled labels:** the coach-judged criteria. #6 (held-out precision of "unsupported"), #9 (coach's area vs
  PitchIQ's) and #11 (strength moments rated positive) wait for the labels.
- **Too little footage:** the broadcast clips hold only 4 held-out advance candidates. PitchIQ judged 3 of them, all
  as supported, so #6 (it needs 5 spells judged unsupported) stays unmeasurable on this footage even after labelling.
- **Game 3:** Metrica game 3 is reserved for one test declared in advance (`build/GAME3_PREDECLARED.md`), after the
  verification fixes and the freeze.

**Open.**
- **Defects in other modules:** the three known defects above.
- **Reasons not given (assess.py):** a check that is not assessed because its phase never occurred gives no reason (the
  page says "not enough data"); and "phase unknown" is given when coach marks name only one team's ball.
- **Youth footage:** no youth footage with keyframes exists yet, so levels 1-3 are judged only on synthetic scenes.

Tests: the worktree's full suite passes, 1853 tests OK (7 skipped) in 625 s. The new files are
`test_coaching_counterexamples.py`, `test_coaching_levels.py` and `test_coaching_eval.py`. Nothing is committed, and
the demo checkout is untouched.

## The penalty-arc tip shown as the ball on amateur 3 (2026-10-18)

**The complaint.** In the user's amateur 3 run `runs/6dfdc88ec6f2d100/20261018-143030-812a5f` (118.2-182 s, keyless)
the final ball sat on the tip of the painted penalty arc, pitch (0.83, 0.46), for 269 frames (18% of the shown ball):
118.2-120.1 s, where the first pass LOCKED on it, and 125.8-132.8 s, bridged through three single weak sightings by
3 s stationary links. The painted-spot discount needs a trusted registration (>= .5; keyless runs sit near .4), the
tip is not a spot, the lock earned the acceptance bonus, and a "still, nobody within 3 m, 1.5 s" rule caught only 2.0
of the 8.9 s.

**Diagnosis.** A re-decode of the stored first pass (registration.json motions, the frames' stored calibration)
reproduces the stored final path on every frame of all four runs below (state, origin, motion, phase, bridged, track
equal; centres within 0.02 px, the saved boxes' rounding), so the decoder was studied without the GPU. In the frames:

- 118.3-120.1 s is a throw-in from the right touchline. The ball is in the thrower's hands, then in the air, landing
  near (1590, 497); it is never detected. The arc tip is the only candidate, and the causal pass locks on it from
  SEARCHING at the run's first frames.
- At 125.4-125.7 s the causal pass takes the real ball at the far goal, 14 m away. The arc tip is seen again as
  tracklets 7, 12, 13 and 16 (rejected TRAJECTORY, JUMP, STATIC, FIELD_LINE) until 132.5 s.
- 126-132 s the keeper holds the ball and punts it at about 132.4 s.
- The camera compensation keeps the tip within 4.5 px (about 0.3 m) over the 14 s.
- The same tip was the stepping stone of an out-and-back excursion at 139.97 s (30 m/s each way) while the ball was
  dribbled, undetected, at a player's feet 139.3-140.8 s.

**The fix: fixed places** (`soccer/ball_path.py` `_places`, `_fixed_places`; README "Fixed places"). Static-object
evidence over time, with no registration needed. Every sighting, whatever its tracklet or status, joins the place it
stands at: within 0.6 m of where the place was first seen, camera-compensated (on the pitch when registered), unseen
for at most 6 s. There is one ball, and it does not come back to the exact spot it left by itself.

- **Fixed:** a place is a fixed object when a still thing stood there for 2 s (four sightings or more), then the
  causal pass took the ball at least 3 m away four times or more, then a still thing stood there again for 2 s.
- **Unattended:** at most a quarter of the place's sightings may be within a player's reach.
- **Stays:** the sightings are split into stays at those acceptances.
- **Kept (the ball on that spot):** a stay the ball was seen coming into or going out of. That means the nearest
  causal acceptance elsewhere within 1 s, at no more than 40 m/s, its tracklet in between no further from the place,
  and not turning back beyond it within 0.3 s, give or take a ball's diameter.
- **Kept (rolling over):** a sighting whose own tracklet is seen 1.2 m from the place within 0.3 s.
- **Demoted:** everything else is a field marking (MARKING_FACTOR, the causal lock included, without its acceptance
  bonus), and the review cuts it as one.

`BALL FIXED OBJECT` events log the places that withdrew causal acceptances; `summary.ball.fixedPlaces` counts the
places. It runs on every frame of a small-scene run: amateur 3's first small-scene frame is at 143.7 s, after the arc.

**Measured.** Ball truth is `tools/check_ball_truth.py` on the 28 hand marks. 17 new marks (118.8-140.3 s, this
session, 4x crops with a 10 px grid) were added to `tools/annotations/ball_positions.json` for the throw-in, the far
goal, the keeper holding the ball, the punt and the dribble.

| amateur 3 118.2-182 s | stored run (base) | GPU scratch run (final code) |
|---|---|---|
| TRACKED / MISSING / UNKNOWN | .346 / .434 / .220 | .317 / .330 / .354 |
| ball truth (28 marks) hit / wrong / absent / hidden | 13 / 15 / 0 / 0 | 13 / 4 / 8 / 3 |
| the 11 older marks (146-150 s) | 10 / 1 | 10 / 1 (unchanged) |
| frames on the arc tip | 269 + the 139.97 s excursion | 0 |

- The GPU run (`tools/run_segment.py`, the stored run's config, 328 s, GPU otherwise idle) has the same first pass
  and candidates as the user's run on every frame. Its final path equals the re-decode with the final code (centres
  equal to 4 decimals).
- 118.2-120.1 s and 139.3-140.8 s are now NOT VISIBLE: absent, not wrong.
- 125.8-130.8 s is NOT VISIBLE.
- The path changes nowhere else; the confidence and identity of the run that no longer contains the arc are
  recomputed.
- Four fixed places on amateur 3: the arc tip (121 sightings, six tracklets, 55 causal acceptances withdrawn) and
  three specks that the path never used.

**Broadcast and Veo unchanged.** These were re-decoded from their stored first passes with the final code:

- broadcast 0-60 s (`20261018-024712-3485a2`);
- broadcast 42.4-222.4 s (the audit's scratch run `BC_42_222_fixed`);
- Veo 1-31 s (`20261018-143758-8b0bab`, a small-scene run, 61%).

All three are byte-identical to the base decode on every field of every frame, with identical events. The report only
gains `fixedPlaces: 0`.

**Ungated, measured and left gated.** Without the small-scene gate, broadcast 0-60 s and Veo stay identical, but the
42-222 s window changes in four spans:

- 93.3-96.3 s and 100.7-103.8 s: the base path sat on a spare ball / painted dash 1.5 m beyond the near touchline,
  pitch (0.558, 1.021), seen 1365 times over 67 s. The real ball was in play in midfield and at the far side
  (verified on zoomed frames), so the audit's "genuine balls out of play" note for these spans is wrong. The rule
  corrects both.
- 42.4-43.6 s: one wrong place for another.
- 192.6-196.2 s: base on a fixed white object at the goal line (tracklet 1297, already seen at 192.3 s while the causal
  pass locked the real ball 10 m away; the keeper held the ball), about 50 px off. New: a bridge through the box,
  about 150 px off.

Taking the decoder off a wrong stepping stone lets it bridge through the next one, so the rule stays behind the gate.

**Tried and dropped** (each failed on these windows):

- A place needing only four sightings each side: on broadcast the real ball revisiting a spot during a dribble (2-3
  sightings, 3-4 m) became "fixed". Now 2 s and four sightings each side, and unattended.
- The review's entry rule ("rolling under 8 m/s within 0.5 s") as the only way in. It missed a ball crossing the left
  penalty-arc tip at 10 m/s (broadcast 199.0 s) and a shot into a place.
- A departure judged from the departing ball's next sighting alone: a ball pausing at a foot failed it.
- An arrival without a direction check: a ball kicked away from a mark, last seen 10 m off, counted as arriving at the
  mark's next stay.
- "Rolling over" when the tracklet was seen anywhere off the place: a still speck whose detections jitter across the
  0.6 m radius kept full weight, and the path moved onto it (amateur 132.87-133.1 s). Now the tracklet must be seen
  1.2 m away.

**Not solved.**

- 130.87-132.8 s (57 frames) is an invented flight. With the arc gone, the decoder bridges from a det .18 flicker on a
  player's white shirt (130.87 s) to a det .11 one (131.73 s) into the punted ball. The attachment test covers the
  lower body only, and a bridged frame (.8) is cheaper than NOT VISIBLE (1.0). These three marks were wrong before too
  (on the arc). It is the decoder's bridge economy, not this rule.
- The keeper's held ball and the dribbled ball at 139.3-140.8 s are never detected.

Tests: `tests/test_ball_path_fixed.py` (7: the arc-tip lock with the gate open and closed, a watched mark, a ball kicked
from a mark, a ball rolling over one, places across tracklets, gate closed equals no profile); the whole suite: 673
pass (1 skipped). Nothing committed.

## Amateur possession: an image-based estimate where possession-4 abstains (2026-10-26)

**The request.** Possession on amateur and Veo footage, which possession-4 declines (a small scene on 60% / 61% of the
frames of amateur 3 and the Veo clip), without touching possession-4 where it judges the footage, and a coaching
report that may use it only under rules. The possession audit (scratch amball/possession_audit) had found possession-4
forced past its gates at .574 on .600 of amateur 3's labelled team points (always the commoner team: .567), its image
metric blind to depth on a low camera, and its wrong calls coming from the ball path (an arc tip) and the people
layer (an undetected carrier near the camera, teamless keepers on reversed goal ends, an identity on the other kit).

**What was built.** `soccer/possession_amateur.py` (version `possession-amateur-1`): `compute()` is the dispatch
every consumer calls (the server's `/api/possession`, the coaching measurement tier), possession-4's own record where
it judges the footage, else (small scene or failed agreement) an estimate in the same record shape. A team is at the
ball when one of its eligible carriers (a confirmed identity whose kit readings agree, or a goalkeeper; teamless ones
from `tactical.goalEnds`) is within 1-2 m of a trusted ball (tracked, or bridged for 0.5 s at most, the first pass
agreeing or pointing to the same team) in a depth-aware reach (the local shape of each frame's calibration at the ball,
scaled by the people's box heights) and nobody else is; possession-4's decoder, with no carry across 1.5 s of a
trusted ball nobody is at, dead balls only with a line near the ball in view, abstention per span with a reason,
never `high`, changes as span boundaries with low confidence. `tools/possession_amateur_bench.py` scores it;
`tests/test_possession_amateur.py` checks it.

**Measured (DEV labels: AM and VEO were used before; amateur 3 0-60 s and 182-242 s are held out for blind labels and
were not looked at).** Before: no frame named (possession-4 abstains); after: amateur 3 1.0 at .356 (32 of 32 team
points), 2 changes reported (2 of the 4 within 2 s, 1 within 1 s), 20 of 22 dead points; Veo 1.0 at .196 (10 of 10),
no change, 0 of 8 dead points. Of the 23 amateur points the forced possession-4 had wrong, 22 are unknown and 1 right.
The BALL agent's interim reruns (small scene from the first frame): .968 at .344 (30 of 31) and 1.0 at .235. Each rule
off on amateur 3: first pass .886 at .389, no-carry .648 at .600, 1.2 s bridges .947 at .422, rows doubled instead of
the depth .952 at .467, wider reach .738-.750 at .62-.68. Forced on broadcast 0-60 / 42-222 s: 1.0 at .898 / .974 at
.852; on Metrica game 1 (tuned, forced): full .980 at .659, camera .965 at .697, the stress test possession-4 declines
.975 at .684. It names little because the ball and the people give it little (49 of amateur 3's 58 unknown team
points in `ball_untrusted` spans).

**possession-4 unchanged.** The dispatch is byte-identical to `possession.compute()` (sha256 of `json.dumps`) on the
run copies of broadcast 0-60, 42-222 and 222-303 s and on all 40 Metrica game-1 windows of full tracking and the
camera; `soccer/possession.py` still hashes 75b65939... (the v4 freeze).

**Coaching.** Coach-marked spans stay authoritative; the estimate fills only the time outside them, with the +-2 s
change exclusion and the source `possession-amateur` ("estimated, image-based"), and only when it named the coach's
team on 90% of the 20 s or more it named inside the marks; its carrier is never used (D1 stays 'no carrier'). Without
marks it is not used (decided: 141 DEV points on two
clips are too few to name a phase nobody checked; DESIGN 3.2's "phase unknown without marks" and every coaching test
hold). On the amateur and Veo copies with the blind labels as marks it named only 9-13 s and 5 s inside them, so it
was not used there.

**Not done / open.** The held-out blind labels (amateur 3 0-60 s, 182-242 s) are being made: the estimate is frozen
(scratch amball/poss/FROZEN.md) for them. `goalEnds` is reversed on amateur 3 as stored (no DEV frame rests on a
teamless keeper). A pass of 1.5 s or more breaks possession; midfield stoppages are never dead; an opponent pressing at
the ball while the true carrier is undetected is named (125.7 s on the interim rerun).

## Full Match Mode: a whole match analysed on this computer (2026-10-28)

**The request.** The user's 19-part spec: a whole recorded match (90 minutes or more) analysed locally in about 2-3
hours on the RTX 5080 by running bounded segment jobs of the unchanged pipeline in parallel and assembling them into
one match that plays, analyses and reports as one; no cloud, Segment mode unchanged (its 180 s cap kept), no frame
skipping in v1, the target claimed only after a measured end-to-end run, never an invented benchmark.

**Built** (worktree `.claude/worktrees/full-match`, uncommitted): `fullmatch/` (plan, manifest, system, worker,
scheduler, routes, calibration, identities, assemble, analytics, corrections, coaching), `match.js` and the page's
mode switch, match panel and match playback in `index.html`, the server's hooks (`server.py`: models released while a
match runs, Segment runs refused meanwhile, `?match=` on the analytics, possession and team-shape routes; the coaching
routes answer a match from `fullmatch/routes.py`), `tools/match_run.py`, and the penalty-spot lock fix in
`soccer/ball_path.py` (every run). README "Full match mode" describes each part.

**Measured.**
- A worker's segment equals the Segment runner's on the same window: exact_diff identical on CAN-ISL (20 s) and on the
  broadcast (148.7-304.4 s, 4641 frames: people, pitch positions, ball, mapping, match, roster, tracks, labels, events).
- Concurrency (CAN-ISL 40 s, results identical to a lone run): 1 / 2 / 3 workers 7.5 / 11.7 / 14.6 fps aggregate. Thread
  caps (3 workers x 3 x 40 s): defaults 13.09 fps, 4 threads 12.11, 2 threads 6.29 (contaminated by a replay): kept the
  defaults (Ultralytics already makes OpenCV single-threaded and sets OMP to 1).
- 3 vs 4 workers on one plan (CAN-ISL 440-920 s as 12 x 40 s segments, the 3 keyframes, no discovery, no assembly, a
  quiet GPU before each): 3 workers 13650 frames in 1022 s = 13.35 fps (a segment 4.09-6.17 fps), 4 workers 957 s =
  14.26 fps (3.34-5.06); peaks 3 / 4 workers: GPU 96% / 96%, 5.24 / 6.41 GB, workers 7.95 / 10.41 GB, system 21.4 /
  23.5 GB, CPU 65% / 77%.
- dev12 (CAN-ISL 420-1140 s, 4 segments, 3 workers): 18450 frames in 1695 s = 10.88 fps (the last segment alone 11 min at
  7 fps); peaks 8.56 GB workers, 22 GB system, GPU 4.79 GB at 96%, CPU 80%; assembly 32 s; views 8-9 MB gz (85 MB JSON).
- Broadcast 0-303 s (2 segments, 2 workers): 9283 frames in 603 s = 15.39 fps; assembly 16.8 s; 20 of 21 boundary pairs.
- Stitching on dev12: 9/9, 14/16, 17/17 people matched over the boundaries; ball agreeing at the cuts; calibration
  disagreement .5-1.3 m. Gap links measured and not automatic: gallery similarity .42 same person vs .35 teammates (the
  right one most similar 49%), position 19-30%: roster 54 entries for ~24 people -> coach merges with suggestions.
- Calibration discovery (research agent, scratch fullmatch/calib/; truth: 9 hand keyframes carried to every sampled
  frame by SIFT registration): of 1679 measured CAN-ISL anchors 8.6% more than 3 m off, all with their camera 20.4 m or
  more from the match's; the camera rule alone leaves none (largest 1.20 m), the neighbour rule alone 18; with all three
  rules 1259 kept, none of 1243 measured more than 1.5 m (median .26, p99 .58, max .92). Broadcast segment 148-303 s
  without a keyframe: the automatic route 27 of 58 frames > 3 m off (35 m at 150-180 s).
- Spot locks replayed from the saved first passes: only CAN-ISL 561.8-577.7 s changed (kf3 and dev12 s0); 0 frames on
  19 broadcast, 6 amateur 3 and 3 Veo runs and the other CAN-ISL segments.
- Match coaching (dev12, 4 segments): both teams measured in 18-25 s, a level change .06 s.
- **End to end** (CAN-ISL, the whole 96.5-minute recording, started from the page at 06:21:38 on a quiet machine;
  halves 00:10-47:34 and 48:05-96:32; 33 segments; AUTO = 3 workers; discovery on; the video's 3 saved keyframes):
  ready at 09:09:39, **2 h 48 min** (10081.5 s). The panel's estimate at the click: 3 h 15 min (2 h 35 - 4 h 05, the
  benchmark's 13.14 fps). Discovery 228.8 s (8 processes: 2876 frames sampled, 1723 anchors, 1261 kept, camera
  (53.75, 113.11) 24.27 m high, share .857, 86 keyframes, 33/33 segments anchored); tracking 9642.7 s (148417 frames,
  15.39 fps aggregate; segments 573-1112 s, 4.13-7.72 fps on one worker; the workers busy their whole span, no wait
  between segments; the tail: one worker idle from 09:01:33, a second from 09:04:58, the last segment ended 09:06:09); assembly 209.8 s (stitching ~47 s, writing
  the match 165 s: views 46, samples 5, possession 76). 33/33 segments on the first attempt, 0 restarts, 0 errors.
  Peaks CPU 87.8%, GPU 96% / 7.33 GB, RAM 23.3 GB system / 9.26 GB workers. 2.9 GB on disk (segments 2.6 GB,
  assembly 298 MB). Verified in the page: playback over the cuts (a prepared swap .04 ms) and halftime (shown as not analysed), seeks 0.1-0.4 s, the match analytics (heatmap), team shape (34.7 s for the match), possession, the
  coaching report (team A ~110 s, PDF 1.27 MB).
- Found in that run: 392 roster entries (203 seen in one segment; 407 of 439 boundary pairs matched); footage classed
  per segment (amateur 29, broadcast 2, Veo 2 on one camera), so possession came from possession-amateur-1 in 31
  segments and possession-4 in 2, and the coaching checks that need a named possession had 68 s; ball disagreement
  flagged at 9 of 31 cuts; calibration disagreement over 2 m at 5 of the 20 cuts with both sides calibrated; 9.1% of the
  owned frames uncalibrated (13048 of 143767), every sampled one a plain wide view.
- Keyframe placement for reach (after that run): a run given keyframes turns its automatic calibration off and a
  keyframe carries ~26.7 s (CAN-ISL segments 20, 23, 30: calibrated from 26.7 s before their lone keyframe). 309 s of
  the gaps had kept anchors inside, 211 s none (midfield views). Offline on the run's discovery chunks: 141 keyframes
  instead of 86, 88.0% of owned seconds within 25 s of one instead of 67.7%, 136 measured median .25, max .77 m (in-frame
  p90 max 1.27, none over 1.5). Segments 30, 13, 20 run again with them (the worker's config, only the synthetic
  keyframes changed): calibrated 49.5 -> 100%, 86.2 -> 100%, 49.2 -> 72.4%; against the reference on every sampled
  second 382 calibrated samples instead of 244, 3.4% over 1.5 m instead of 4.9%, none over 3 m; frames calibrated in
  both runs: median within .03 m, worst 2.18 -> .49, 1.69 -> .59, .66 -> .23 m; the new frames of segment 20 1.5-2.3 m
  off at worst where carried 5-17 s across a midfield pan (confidence .49-.85). Speed within the run-to-run spread
  (872 / 885 / 976 s against 825 / 789 / 1038 s; the detector's ms per frame moved the same way, +11 / +15 / -11%).

**Decisions forced by measurements.** Frames, not timestamps, identify a frame (irregular decoder timestamps); the
stdin pipe read through a private duplicate (a pending read deadlocked numpy's DLL load); no automatic gap links (see
above); calibration discovery with the camera-consensus rule (the temporal rule alone kept repeating wrong solutions);
keyframes placed for reach (a keyframe per third left segment starts uncalibrated); AUTO 3 workers on the 5080 (a
fourth measured 6.8% faster on 12 x 40 s with 2.5 GB more memory: not made the default on one short benchmark); spot
locks for every run (replay: nothing else moved). Fixed after the run: the latest match is the newest by creation (a
test match named dev12 sorted first and opened instead), and a stopped match a newer complete one replaced is no longer
offered for resume.

**Not done / open.** The whole match was not run again with the reach placement (three segments were). Roster
fragmentation needs the coach's merges. The footage class (and so the possession engine and the coaching's named
possession) is decided per segment. Midfield-only views longer than ~50 s stay uncalibrated (no single-frame solution;
a fixed-camera solve from the consensus camera would be the next step). Ball disagreement at cuts is flagged, not
resolved. No whole broadcast match has been run end to end. The view parse costs 176-361 ms per segment change.
Coaching for a match has no coach-marked possession, squads, feedback or match date; the estimated team-shape mode is
not available for a match.

**Then: the results stored compressed** (the same day, asked for after the report). The segments' plain results were
2.50 of the 2.87 GB the match took on disk; a worker now keeps its result as `result.json.gz` (gzip level 6, written
beside the plain file, read back and compared by length and sha256 before the plain file is deleted, the same bytes
every time); the readers, the verification (segment.json names the stored file) and resume accept both forms.
- Levels on 8 results (6 CAN-ISL, 2 broadcast; 70-93 MiB): 1: 6.4-7.9x in .26-.41 s; 5: 9.8-12.4x in .45-.74 s; 6:
  10.4-13.4x in .58-1.05 s; 9: 11.2-14.4x in 1.1-2.7 s. Decompressing .07-.11 s against a JSON parse of .50-.79 s.
- The whole CAN-ISL match, copied with hard links (the original only read: still 3,077,580,978 bytes) and assembled
  twice with the same code: the 33 results 2.504 GiB -> 238.1 MiB (10.77x, 9.9-13.4x per segment, .76-1.18 s each
  with the read-back); the match 3,077,580,978 -> 638,764,214 bytes (the coaching cache and the logs included);
  assembly 209.1 s plain, 214.6 s compressed; all 38 assembled files the same (views decompressed, the index without
  its timings, samples, possession, summaries); tools/exact_diff.py (it now reads .gz) on a compressed result against
  its plain original: identical.
- On the GPU through the page's route (CAN-ISL 600-780 s, 3 x 60 s, AUTO 3 workers, discovery on, the saved settings):
  ready in 430.7 s (the panel's estimate 411 s, 329-514); each worker stored its result in .36-.40 s (9.7-10.2x); every
  artifact verified with its checksum, a deep reconcile changed nothing, no plain result was left; that match (21.5 MB)
  and the older plain full match both opened in the page and loaded their views; the video's saved settings came back
  byte-identical.
- The disk estimate counts the compressed results (2.2 KB a frame), the other segment files, the assembled match (2.5
  KB a frame; 2.0-2.3 measured) and the plain results of the segments running at once: 1.2 GB for the whole CAN-ISL
  match (4.8 GB before; the match took 0.6 GB).
- The full suite then failed once, in a scheduler discovery test: the fake workers share one journal and read it to
  count attempts, and under load one read half a line another process was writing, crashed, and the match gave up
  (reproduced 2 of 15 runs with 14 busy CPU processes; 0 of 10 on a quiet machine). Not the product code: the journal's
  appends and reads now hold an OS file lock (tests/fullmatch_journal.py), 0 of 30 under the same load.

**Coaching intelligence v2: key moments** (2026-10-30, the user's 19-part spec; built in worktree coaching-moments).
The earlier report of CAN-ISL (team A, level 7) had 0 strengths, 0 improvements, 0 reviews and 2 attacking-shape
descriptions with no player named: every player-level check waited for possession-4, which named the phase for 68 s of
96 minutes on this camera. New package `soccer/moments` (README "Key moments"): a 5 Hz data cache from the assembled
views (59 s once), roles from pairwise relative positions, seven detectors, moments / patterns / suggested zones /
words by level, a report per team, routes, the page's moment player.
- Audit numbers (all 143,767 owned frames): both teams with 8+ positioned players in 80% of frames, the camera a median
  67% of the pitch, the ball placed in 37-39%, possession named on 42% of live play (settled 29%), dead ball 41%.
  Team letters of the match: A = white (Iceland), B = red (Canada) in all 33 segments (a Segment run at 435 s had them
  the other way: letters are per run).
- Video checks (validation sheets: board, key frame, crop, lead-up, follow-up) drove the rules: defensive judgements
  need the ball near (a gap 36 m from the ball and an opponent 'between the lines' with the ball already in the box
  meant nothing), a defender marking the deepest attacker is not 'dropped off', a lone striker pressing 73 m from her
  own goal is not 'pressure without cover', the player on the ball is not the cover and is never asked to move, a
  defender standing in its own zone is not a 'positive' cover, the nearest cover is the one named, a 'free' opponent
  needs an open line. Synthetic tests then found two more: a presser beside the ball counted as blocking every pass
  (now a pass is blocked only by a player in front along the line), and windows of the team's own possession
  re-labelled a pushed-on full back as a midfielder (now the defensive windows are preferred).
- Final reports (level 7): A 8 strengths / 4 improvements / 3 patterns, B 7 / 4 / 2; 4.5 s each once the data is
  cached. Of the 23 moments shown, 21 correct or plausible on video, 1 unclear, 1 not viewed: development data, not
  held out.

**Full match intelligence: people and slots, halftime, the match's state** (2026-10-31, the user's 21-part spec; built in
worktree match-intel). README "Full match intelligence".
- Audit of CAN-ISL: 392 match chains numbered as they came (A-197, B-150, GK-44); 295 begin at a segment start (a
  player is followed over a boundary only when both runs saw him in the ~6 s overlap: a median 12 of ~22 people);
  gap links accepted 2; within a segment 10-13 identities per team. The 11+ team A outfield players seen in 8.14% of the
  samples were 27 fragments of two static boxes on a broadcast graphic (bottom right of the picture).
- Identity cues across a boundary, the overlap-matched chains as truth with the overlap hidden (scratch matchintel
  link_cues / role_test / look_test / fit_links): appearance top-1 42% of ~5 candidates (55% with the best-match
  gallery similarity), the last position 77% at a 0 s gap and 28% at 60 s, offsets to the teammates seen on both sides
  70% -> 55%; a logistic model of them 80% -> 57%. Agglomerative and windowed-assignment linkers reached 75-81% precision
  at 14-45% recall; a 10-path min-cost flow over each team kept 60-64% of hidden continuations in their place, and its
  links scored 0.95 or more were right 54-74% (two folds). Decision: no automatic person joins; the slots show the
  flow's continuations as estimates ('A-04?'); the coach's merges and confirmations join people.
- Two video checks found data faults the layer handles: a referee inside a team-A run identity (an ID switch in the run;
  kit histograms do not catch it) and a coach in black taken for a player (caught by the kit check).
- State: dead spells from the possession record, restarts by the players' speed, kick-offs from the two teams in
  opposite halves; the 4 scoreboard goals found with the right teams (508 / 1278 / 4182 / 4683 s against score changes
  shown by 511 / 1281 / 4201 / 4721 s), 6 kick-offs. Restart kinds checked on 17 zoomed video frame sets: about half
  right with the ball's place alone (crowded boxes called goal kicks, a duel called a throw-in) -> a placed (still)
  ball and, for a goal kick, a near-empty penalty area are now required; 15 restarts stay UNKNOWN_RESTART.
- Tried and dropped: an automatic scoreboard reader (the see-through panel over the moving crowd), a direction-bound
  kick-off rule (a wrong halftime setting hid the goals; now direction-free and it checks the setting instead).
- Substitution candidates: a place passing to someone else near the halfway line gave 8, 5 of them plainly play or set
  pieces on video; the law-3 rule (a team's player last seen and one first seen by the halfway line on a touchline in
  one stoppage) gives 3, none confirmable at broadcast resolution, and a likely Iceland change around 4210 s is missed
  (they came on while the camera could not be placed). Candidates only; the coach confirms.
- Last checks on the 11+ players still seen (team B 2.33%, unchanged by the flags): video crops showed tracks that join
  two people where segments meet, two of them a keeper stitched to an outfield player (B-111: Iceland's keeper, then a
  Canada player; A-163: Iceland's keeper, then a white player) and one a player stitched to the assistant referee
  (A-41). The two keeper ones are now 'mixed' (a keeper member next to an outfield one, or shirts and shorts both under
  0.4 alike: 1 of 408 member links, the next lowest 0.67): no automatic place, a note in the roster review. A-163 had
  pushed a real 23-minute player (A-150) out of the places: the solver drops the longest fragment present at the
  eleven-at-once moments; unresolved outfield time 5.4% -> 4.4%. B-101 (20 min) and B-23 (21 min) stayed out for 2 and
  20 conflicting samples: the fragment reward stopped growing at 300 s; without the cap the hidden-link test is
  identical (64% / 59% same place) and the unresolved time fell to 2.7% (a larger reward per second cost same-place
  links: 61% / 58% at 0.05/s). Also fixed: the slot numbers read the offsets' sums as means (a place's line ordered by
  how long its starter was seen); now by the mean place, deepest first.
- Final CAN-ISL: 324 people for 392 fragments; 20 sure slot starters, 269 estimates, 31 unresolved, keepers GK-A (16
  fragments) and GK-B (20) plus one small group each shown GK-A? / GK-B?; 27 overlays, 4 kit, 1 off the pitch left out;
  of the 308 labels above 10 before, 262 continue a place, 27 left out, 19 unresolved. State: 45.8% in play, 20.9%
  stopped, 22.0% possible stoppage, 11.3% unknown; 4 goals, 6 kick-offs, 3 substitution candidates, direction check 5 of
  5; the people and the state built in about 4 s from the match data (about 12 s more the first time, when the pitch
  check of the blind stretches reads the video; then remembered).
- Tests: halftime 13, persons 27, state 17, routes 4, the page 49 (node).

## Ball possession v2: hidden-ball inference and full-match possession analytics (2026-10-31)

The user's spec asked for four things:
- better ball tracking where the detector loses the ball;
- team possession inferred through hidden-ball gaps;
- a possession timeline;
- possession percentages in the Analytics tab (match, halves, custom ranges) with coverage shown honestly.

Built in worktree `ball-possession`, merged uncommitted. Nothing was re-run on the GPU for the match.

### Audit (CAN-ISL, 143,767 owned frames)
- **Final ball path:** observed .350, weak .038, bridged .466, unknown .146.
- **Live play:** 1,183 s bridged against 133 s unknown.
- **Unseen frames:** 2.6% have no detection at all. 92% hold a candidate the tracker rejected with a detector score of .5 or more, mostly STATIC or OTHER_OBJECT.
- **Against blind hand marks (four dev windows, 276 ticks with the ball visible):**
  - the path was on the ball within 12 px on 96 ticks, bridged within 24 px on 54, wrong on 103, absent on 23;
  - of the 180 misses, 139 had no detector candidate within 8 px of the ball (detector misses);
  - 37 had a candidate the tracker rejected: 16 FIELD_LINE (the path bridged within 7 px anyway), 8 OTHER_OBJECT, 6 TRAJECTORY;
  - 88 more ticks had the ball hidden, unfindable or off screen;
  - wrong bridges are mostly dead balls carried in the hands.
- **The old match possession record:**
  - it ran per segment, so it restarted at all 32 boundaries;
  - it summed spans as t1 - t0, losing 153 s;
  - of the live time it named a team on .438, left .505 unknown, and counted .036 dead.
- **Ball tiles (scan the whole frame at 0.8x while not locked),** GPU runs of the four dev windows with tiles on and off:
  - ticks with a candidate on the ball 138 and 138;
  - hits 92 and 93;
  - wrong 146 and 136 (mostly one window);
  - 4-12% slower (machine shared with the agents).
  - Not a gain: the default stays.

### Possession v2
See the README section "Ball possession analytics (possession v2)".
- **Model:** six states, fitted on Metrica game 3's first half through possession_sim (variant `tactical_gaps`, set to CAN-ISL's unplaced share).
- **Operating constants:** chosen by rules written before each run, scratch `possv2/PREDECLARED.md`:
  - Metrica floors, plus a score on the dev windows W1 W3 W6 W8 (96 team ticks);
  - min_posterior .7: the lowest threshold that keeps dev accuracy at .92 or more AND passes every case test (at .5 a duel and an 8 s hidden ball were named a team).
- **What the dev windows changed:**
  - a 3 s pass between team-mates read as free, so contested and free now remember the team that lost the ball;
  - the gap context was fitted on 1% of hidden frames, so a gap-rich simulator variant was added;
  - 'nothing to go on' (22% of CAN-ISL's live frames) is neutral;
  - open-ball frames are tempered as one event;
  - the carrier context ends when the ball is seen away.
- **Disclosure:** a labeller's hand-back message summarised the held-out window W7 (90 dead ticks, 1 free) before the freeze. Nothing was tuned on it.

### Results
- **CAN-ISL, four development windows** (W1 W3 W6 W8, 96 team ticks; the constants were chosen on them):
  - coverage .615 at accuracy .932, with 4 of 105 named ticks the wrong team;
  - the old match record: .562 at .944.
- **CAN-ISL, four HELD-OUT windows** (W2 W4 W5 W7; blind labels; scored once after the freeze; 77 team ticks, 234 dead):
  - accuracy .962 at coverage .338; the old record on the same ticks: .429 at .273.
  - Wrong team: 1 of 73 named ticks.
  - Dead ticks counted as live possession: 39 of 234 (16.7%). These are stoppages the match state calls in play: an offside free kick and a goal kick being prepared.
  - Changes found: 0 of 7.
  - Pre-declared targets: accuracy (.90) and wrong team (5%) passed. Coverage (old + .20), dead leakage (15%) and change recall (.5) failed.
  - The long possessions left unknown are mostly a carrier 1.5-2.5 m from a seen ball, outside the 1 m zone.
- **The whole match's live time** (IN_PLAY and the state's UNKNOWN, 3,285 s):
  - unresolved .505 (old record) -> .430; a team named .438 -> .521; free .049;
  - team A 37.0% / B 63.0% (old record 37.9 / 62.1);
  - first half 33.6 / 66.4, second half 40.6 / 59.4;
  - in the 3 s after the 32 segment boundaries, unresolved .716 -> .61.
- **Broadcast Segment runs** (hand labels, old convention; possession v2 against possession-4, coverage / accuracy):
  - BC60: .907 / 1.0 against .992 / 1.0;
  - BC180: .813 / .988 against .975 / .981;
  - BC303: .854 / .943 against .932 / .958.
  - On broadcast possession v2 names less than possession-4.
- **Metrica game 3 through the simulated camera** (tactical_gaps), the second half (not tuned on):
  - team accuracy .975 at coverage .634; .913 at .464 where the ball was not placed;
  - named frames the truth's team .935;
  - changes within 2 s: precision .77, recall .28.

### Coaching and runtime
- **Key moments:** the report keeps a phase only where v2 and the segments' record agree. CAN-ISL went from 97 moments to 77 (60 unchanged). The replace mode would have changed 153 of 188.
- **Runtime:**
  - the match frame table: 45.6 s once (33 views);
  - the decode: 2-3 s;
  - the record: 331 KB;
  - segment mode: 0.57 s for 5400 frames.
- **Tests:**
  - tests/test_possession_v2.py: 16;
  - tests/test_possession_stats_panel.py: the page;
  - the full suite: 2460 OK (7 skipped) in the worktree; 2460 OK (6 skipped) in the checkout after the merge.


## Goals: what led to each one, what allowed it, and a cleaner timeline (2026-11-01)

The user asked for four things:
- recognise when a goal happens;
- analyse each goal for the team: before a goal it scored, what was good and led to it; before a goal it conceded,
  what went wrong and why it was allowed;
- a goal marker on the time control;
- a filter for what the time control shows (free kick, stoppage and the rest only when wanted).

Built in worktree `goal-analysis`, merged uncommitted. Nothing was re-run on the GPU for the analysis itself.

### What already existed
Goal recognition: the match state (`fullmatch/gamestate.py`) finds a GOAL from the kick-off after it (law 10). It found
CAN-ISL's four goals (4 of 4) and their teams. Its goal time was the last moment the ball was placed in the goal mouth.

### The reference: a blind narrator
An agent that saw only the video and a brief described each goal: when the ball crossed the line, how the ball was won,
the build-up, what went well, what went wrong, the foul for a penalty, and how sure it was of each point. It saw no
PitchIQ output. Its notes and labels are in session scratch `goals/narratives/`.
- Goal 1, 508.24 s, Iceland: a poke in Iceland's own half at 499.7 s; the ball ran loose; the forward won the race near
  halfway (about 501-503 s), ran 45 m and shot low from about 20 m at 507.4 s past a keeper who had come out to the
  penalty spot.
- Goal 2, 1276.0 s, Iceland: a tackle at the centre circle at 1271.2 s, a carry, a pass at 1273.4 s, the scorer dribbled
  between two defenders and shot from the edge of the box at 1275.5 s.
- Goal 3, 4180.59 s, Canada: a penalty for a challenge in the box at 4059.0-4059.65 s, taken two minutes later.
- Goal 4, 4684.75 s, Canada: a penalty for a challenge 11-12 m out at about 4591.5 s, taken at 4684.35 s.

### What the first version got wrong, and the fixes
Against the narrator, the first analysis:
- called goal 2 'a loss of the ball 1 s before the goal'. The state's goal time was 2 s late (1278.0): the shot was lost
  by the tracker, the ball seen again in the net, and possession after the goal was read as the attack's start;
- called goal 3 'a penalty or a direct free kick' (no ball placed on the mark);
- called goal 4 'open play' when nothing before it had been seen.
The fixes (`soccer/moments/goals.py`, `fullmatch/gamestate.py`):
- **The strike** ends every attack: the ball's last fast move towards that goal (12 m/s or more, from 35 m or closer).
  CAN-ISL: 507.39 s and 1275.39 s against the narrator's 507.4 and 1275.5.
- **The goal time** in the state is when the ball went in: the earliest of the last sight in the goal mouth, the start
  of that spell in the mouth and the crossing from the strike (`_goal_time`), only after the last restart of play
  before the goal's stop (the floor), and a penalty taken in that span is the goal (the same law-14 layout,
  `goals.penalty_layout`; state VERSION 5). CAN-ISL: 507.99 / 1275.91 / 4180.58 / 4684.78 against 508.24 / 1276.0 /
  4180.59 / 4684.75; before: 507.99 / 1277.99 / 4182.38 / 4683.38 (time uncertain). The goal analysis's own estimate:
  508.17 / 1276.16 / 4180.58 / 4684.78.
- **What the rerun showed** (match 20261101-035319-ba987c, the same video and settings, the merged code): it tracked
  more of the stretch before goal 4 and saw the ball in Iceland's goal mouth at 4553.6 s (a goal kick); the play after
  it, about 11 s up to the foul, was under the 12 s that says 'play went on', so the state made that the goal, two
  minutes early, with high confidence. The floor and the penalty rule above were added for it (tests: a ball in the
  goal mouth before a restart and 10 s of play, then a penalty, with and without the penalty).
- **Penalties by the players' layout** (law 14): one person on the goal line, one at the mark, everyone else out of the
  box and 9.15 m away, held 1 s or more. Over the whole of CAN-ISL it holds 1 s or more twice, at the two penalties
  (5.2 s and 1.2 s); the next longest is 0.8 s.
- **The play that won a penalty**: the scoring team's last loss of the ball before it (possession v2). CAN-ISL: 4059.54 s
  (narrator 4059.0-4059.65), the ball in the box 5 m out; 4590.90 s (narrator about 4591.5), 13 m out, right of centre.
- **A turnover from the possession runs** when possession v2 records no change between the two teams' runs, and the
  attack's start where the other team lost the ball (the ball loose 4 s or less).
- **Names as the page shows them**: the key-moments report named players by the assembly's labels (B-16, B-95), while
  the page shows the match's slots (B-06?, B-01?); 10 of 311 people agreed. The report now takes the slots' labels, for
  its goals and its moments. In the browser on the rerun the moment cards showed internal person ids ('A-P281 right
  midfielder'): the page's label map has them, but the cards went through the text pass, which rewrites only ids shaped
  like track ids ('A-12'). The cards now look each id up in the page's labels, then in report.labels (every person the
  report's moments involve).
- **Each team told the place in its own direction** (the conceding team 'lost the ball in its own third (left)').

### Result on CAN-ISL (the same four goals: development data, not held out)
20 findings: 14 right, 4 partly right, 2 the video cannot confirm, none wrong.
- Partly right: where and when the ball was won. Goal 1: the race at halfway, 3 s after the poke in Iceland's half.
  Goal 2: 2 s after the tackle and 8 m nearer goal (possession v2 still named a Canada player during Iceland's carry).
- Cannot confirm: the player on the ball (no shirt number legible) and the keeper 6 m off his line at goal 2.
- The match state labelled the play before goals 1 and 3 a POSSIBLE_STOPPAGE (Canada's 25 s of pressure before the
  penalty included): possession v2 then names nobody there, which is why goal 1's poke is not seen. Not changed here.

### The rerun (the user's request: the whole match again, into the checkout's runs/)
- Match `20261101-035319-ba987c`: the same video, the old match's three keyframes (437 / 530 / 580 s; the video's saved
  settings had lost them on 2026-10-30, a stopped Full Match start with none), halftime 2854-2885 s, 3 workers,
  discovery on (141 synthetic keyframes). 33 of 33 segments, no restart, 10,320 s (discovery 240, tracking 9,771,
  assembly 308; 15.2 fps; the CPU shared with test suites and agents, so not a benchmark). 642 MB.
- Calibrated samples 95.5% (the old match 90.9%). Its layers were built after it with the merged code (state version 5,
  possession v2, both teams' key moments at level 7): the four goals at 507.99 / 1275.91 / 4180.58 / 4684.78.
- Its goal analysis against the narrator: 19 findings, 11 right, 6 partly right (where and when the ball was won for
  goals 1-2 as before; goal 4's penalty placed at 4593.1 s, when play stopped, 1.6 s after the challenge, 9 m out
  against 11-12 m), 2 the video cannot confirm, none wrong.

### The page
- `coaching.js`: the GOALS section after the summary (agent; brief in session scratch `goals/FRONTEND_BRIEF.md`).
  Cards with the time (rounded like the timeline), the score after the goal, how it came about, findings with source,
  sureness and a time link, notes, WATCH GOAL on the moment player.
- `match.js` / `index.html`: goal markers with the running score; TIMELINE MARKERS (goals only at first, eleven kinds and
  the play-state band, remembered with try/catch). In the browser on the real match the popover was cut off under the
  sticky header on a 768 px window: it now opens below the seek bar when there is no room above, with its own scroll.
- Checked in the browser (worktree server, the CAN-ISL copy): the four markers and scores, the filter (free kicks on: 6
  more markers, '2 of 11', saved; Goals only: 4), both teams' Goals sections built from the page, WATCH GOAL on a
  penalty (from 69:32.38), a finding's link pausing at 67:39.54 on the challenge, no console errors.

### Tests
- tests/test_moments_goals.py: 24 (synthetic scenes and hand-made states).
- tests/test_intel_state.py: a goal's time when the ball went in; a ball in a goal mouth before a restart of play is not the
  goal; a penalty taken after it is.
- tests/test_moments_panel.py: the Goals section (the fixture regenerated from the backend's current output).
- tests/test_match_intel_panel.py: the timeline filter and the goal markers.

### Not done
- No held-out goals: another match with goals and a reference is needed before any number here is a claim beyond CAN-ISL.
- Shots, passes and the scorer are not measured. Whether a keeper's position was right is not judged.
- The match state does not classify the penalty restart itself (the timeline shows 'Unknown restart' at 69:40); the
  goal analysis tells it from the layout.
- The POSSIBLE_STOPPAGE spells over real play before goals 1 and 3 (above).
