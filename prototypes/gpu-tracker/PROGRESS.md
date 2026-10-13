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

## Phase 2 — identity coverage, roster and visibility

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
