# PitchIQ local GPU prototype

A separate, native Windows review tool for a local soccer video. The soccer
detector runs on CUDA; tracking, optical flow, pitch detection and appearance
embeddings use the CPU. Your footage stays on your computer. No account, upload,
or API credits are needed. The main PitchIQ application does not need to be running.

The tracker is soccer-specific. It does **not** track every person the detector
sees, and it does **not** treat a new tracker ID as a new player:

- Only people whose feet are on (or just at the edge of) the detected pitch are
  considered. Spectators, staff, bench, photographers and people behind the
  boards are rejected, with a reason you can see.
- Short-lived **local track IDs** (BoT-SORT) are separated from persistent
  **global identities** (`A-07`, `B-03`, `GK-1`, `REF-1`). A player who leaves the
  view and comes back is re-identified and keeps their global identity.
- When identity is uncertain, the person stays *identity uncertain* instead of
  being guessed. A wrong merge is worse than a short unknown.
- Identity, team and role are separate. Referees (`REF-1`, `REF-2`) are tracked
  separately and never counted as players; goalkeepers are recognised from
  position and kit and shown by team (`GK-A`, `GK-B`).
- One match ball is tracked. White spots, painted lines and socks are rejected;
  when the ball cannot be confirmed it is `BALL UNKNOWN`, never a guess.

## Run from Command Prompt

The environment and detector are already installed on this computer. From the
repository root:

```bat
cd /d C:\Users\some1\Documents\PitchIQ\PitchIQ
prototypes\gpu-tracker\start.cmd
```

Open http://127.0.0.1:8765, click **Upload video** and choose a match video
(.mp4, .mov, .mkv, .avi, .m4v or .webm). The file is copied into
`prototypes\gpu-tracker\uploads\` on this computer (ignored by Git) and never
leaves it; uploading the same file again reuses the copy. Videos uploaded earlier
can be reopened from **Earlier uploads**. Then click **Process on GPU**. Previous
results for the same video load automatically.

You can still pass a video on the command line instead:
`start.cmd "C:\Users\some1\Downloads\Video Project 1.mp4"`. With a video
given, add `--run` to start processing immediately. Use `start.cmd --port 8766`
(or add `--port 8766` after the video) for another port. Ctrl+C stops the server.
Accepted re-identifications, role changes, crossing corrections and sanity
warnings are also printed in the console.

Choose a segment up to 180 seconds long. For a longer match, process consecutive
segments and tick **Continue identities from the previous run**: the next run
starts from the saved global identities (and their appearance galleries) of the
latest completed run, so `A-07` at minute 5 is still `A-07` at minute 70 when
the evidence allows it. The initial 1280 setting combines 960 and 1280
whole-frame passes: the lower-resolution pass retains nearby players that this
checkpoint sometimes misses at higher resolutions. Choose 960 for a single,
faster pass. Extra small-ball scans inspect overlapping crops every other frame
and take longer.

## Pipeline

```
video frame
  -> person + ball detection ........ soccer YOLOv8x (CUDA), classes player / goalkeeper / referee / ball
  -> camera motion .................. soccer/camera.py      background optical flow; cuts start a new camera segment
  -> pitch detection ................ soccer/pitch.py       painted-line geometry (touchlines, goal lines) + grass, every 0.2 s
  -> outside-field filter ........... soccer/relevance.py   foot point in the pitch polygon, audience, size check
  -> local multi-object tracker ..... soccer/local.py       BoT-SORT with OSNet features (temporary track IDs)
  -> appearance ..................... soccer/appearance.py  OSNet embedding + jersey / shorts / socks colour, quality
  -> team kits + kit votes .......... soccer/teams.py       two team kits learned online, plus referee and goalkeeper kits
  -> position cues .................. soccer/goals.py       near a goal / in a penalty area, deepest, isolated
  -> role classification ............ soccer/roles.py       ~10 s of detector, kit and position evidence per track
  -> pitch coordinates .............. soccer/calibration.py optional landmark calibration, carried through pans
  -> GLOBAL IDENTITY MANAGER ........ soccer/identity.py    candidates, promotion, re-ID, role locking, swap guard, sanity
  -> ball tracker ................... soccer/ball.py        candidate scoring + one camera-compensated Kalman ball track
  -> persistent soccer observations . result.json           per frame: identity, team, role, confidences; the ball
```

`pipeline.py` wires the detector into `soccer/tracker.py`, which runs every step
after detection. The `soccer` package needs only NumPy and OpenCV (BoT-SORT is
imported lazily), so all of its logic is unit-tested without a GPU.

### 1. Only people on the pitch

- The playable region is re-detected from the image every 0.2 s and carried with
  the camera motion in between. It is never a fixed rectangle. Its edges come
  from the painted line geometry: white paint is found as thin bright ridges
  with grass on both sides (a 1–2 px far touchline blended with the grass
  included), straight lines are fitted to it, and the outermost line with grass
  on its pitch side and only a run-off strip beyond it becomes the boundary. Wide
  white advertising boards, their edges and the stands are not paint ridges, so
  the boundary sits on the white touchline, not on the boards. The front edge of
  a penalty or goal area is recognised and never taken for a goal line. Where
  no line is visible, the grass edge is used (painted lines and players are
  bridged, a band of advertising boards is not, so green seats behind the boards
  never join the pitch). Each side is followed over time: small changes are
  blended, a jump must be confirmed on the next analysis, and a line that briefly
  disappears is carried with the camera motion for up to 3 s.
- Each detection is tested at its **foot point** (bottom-centre of the box), not
  by box overlap: a spectator's box can overlap the pitch while their feet are in
  the stands. Zones are `inside`, `boundary` (within the touchline tolerance),
  `outside` and `unknown` (no reliable pitch, e.g. a crowd close-up).
- `outside` people are rejected as `REJECTED: AUDIENCE` (beyond the far touchline,
  in the stands, or in a crowd) or `REJECTED: OUTSIDE PITCH` (bench, technical
  area, behind a goal line). A perspective size model rejects implausibly large
  or small people (`REJECTED: SIZE`); a player lying on the ground is measured
  along the body. People just beyond a touchline may *continue* the track of a
  known nearby player (a throw-in) but never start a new one.
- **Touchline tolerance** (default 35% of the person's height, clamped to
  0.6–3% of the frame height) is adjustable. Raise it if real players at the
  line are rejected; lower it if bench staff are accepted.
- Optional **field boundary keyframes** (a polygon you draw) are combined with the
  automatic region; the stricter one wins. Use them for neighbouring pitches.
- Assistant referees patrol just outside the touchline. A lone person just
  beyond the near touchline whom the detector confidently calls a referee is
  kept (in the `boundary` zone); a track that stays there can only ever become
  an official, never a player.
- Turning off **Only track people on the pitch** keeps everyone (for review).

### 2. Candidates before players

A new local track starts as `CANDIDATE`. Evidence is gathered at 5 Hz
("observations"): field zone, detector class votes (player / goalkeeper /
referee, weighted by score), kit votes against the two learned team kits, and
appearance samples. A candidate is promoted only after at least 5 observations
(1 s) with at least 80% of its recent observations inside the pitch and a stable
role decision. A person first seen on the touchline needs 10 observations and
must also be seen inside. Tracks that stay mostly outside become
`REJECTED_OUTSIDE_FIELD`; tracks without a consistent role become `UNKNOWN`.
Nobody is promoted while the pitch is unreliable.

Identity, team and role are separate: `A-04` is an identity, `A` its team and
`PLAYER` its role. Roles come from about 10 s of accumulated evidence, never one
frame. Each observation adds the detector's class (player / goalkeeper /
referee, weighted by score), a kit vote (team A, team B, the referee kit, a
goalkeeper kit or neither; jersey and shorts are both checked, so a referee in a
jersey like one team's but with black shorts matches neither) and where the
person stands (in or near a penalty area, the deepest person towards a goal,
isolated from everyone, on the touchline):

- `PLAYER_TEAM_A/B`: at least 70% of kit votes for one team.
- `REFEREE`: the detector's referee votes; or a kit like the referee's (or
  consistently neither team's) while following play away from the goals; or a
  kit matching neither team running the touchline (an assistant referee). A kit
  that clearly matches a team holds a referee decision back unless the
  detector's votes are a clear majority.
- `GOALKEEPER`: the detector's goalkeeper votes; or time in or near a penalty
  area in a kit unlike the outfield kits (or like a known goalkeeper kit) while
  being the deepest or most isolated person. At least one strong cue (detector,
  goal area or goalkeeper kit) is required.
- Otherwise the person stays `CANDIDATE` (later `UNKNOWN`) and is never forced
  into team A or B.

Roles keep being re-evaluated after promotion, about once a second. A referee or
goalkeeper role with a role confidence of at least 0.7 is **locked**: it
survives leaving the goal area, odd frames or a detector that changes its mind,
and is never turned back into a team player. A player identity whose evidence
becomes clearly referee is converted: a matching missing referee identity, or a
new `REF-n`, takes over the track; if the player identity only ever followed
this person it is merged into the referee (`RETIRED`) and every observation
already labelled with it is relabelled, so a referee never stays `B-15`. A
player identity that turns out to be the goalkeeper keeps its ID, gets the
locked goalkeeper role and is shown as `GK-A` / `GK-B`, earlier observations
included. A returning referee or goalkeeper is matched by their own appearance
even when the detector or the kit vote calls them a player. Every change is
logged (`ROLE UPDATE`, `GOALKEEPER IDENTIFIED`, `GOALKEEPER TEAM`). An
established team only changes after 80% of the last 15 votes point to the other
kit (the identity is then released and re-identified).

Team kits are learned automatically from the torso (upper jersey) colour of people
the detector calls players, so referees and keepers do not pull a team prototype.
Two kits must each have enough support and be clearly separated; one kit under
changing light is not split into two teams. A/B naming is arbitrary but stays
fixed during a run and when identities are continued.

A goalkeeper's team is the team defending their goal. Whenever both teams are in
view, the two outfield players deepest towards each side are usually that
side's defenders (offside line) and vote for which team defends which side; a
goalkeeper at a goal gets that team after at least 15 consistent observations
with a 75% majority (shown as `GK-A` / `GK-B`), and stays `GK-1` with an unknown
team otherwise. Two goalkeepers for one team are flagged as a possible duplicate.

### 3. Local tracks vs global identities

BoT-SORT local tracks are temporary: an occlusion, an exit or a camera cut ends
them. The **global identity manager** keeps a registry of real people:

| Field | Meaning |
|---|---|
| `id`, `longId` | `A-07` / `TeamA_Player_07`, `GK-1` / `Goalkeeper_01`, `REF-1` / `Referee_01`: the persistent identity |
| `display` | what the overlay shows: `A-07`, `GK-A` / `GK-B` (goalkeepers by team), `REF-1` |
| `team`, `role`, `label` | `A` / `B` / none; `PLAYER`, `GOALKEEPER`, `REFEREE`; combined `PLAYER_TEAM_A`, `GOALKEEPER_TEAM_B`, `REFEREE` |
| `status` | `ACTIVE`, `MISSING`, `OFF_SCREEN` (left at an image edge), `RETIRED` (merged into the referee or goalkeeper it really was) |
| `roleLocked`, `roleHistory` | whether the referee / goalkeeper role is locked, and when and why the role changed |
| gallery | up to 6 diverse, high-quality samples (OSNet embedding + kit colours) |
| last seen | time, image box, pitch position, camera-compensated velocity, exit edge |
| confidences | identity, team, role, referee and goalkeeper confidence; recent history |

Identities are never deleted. When a promoted track appears, it is first compared
with every missing identity of the same team and role. The score combines
appearance (OSNet similarity against the whole gallery, best two pairs), shorts
and socks colour, team (a hard gate), spatial plausibility (where it left vs where
the new track appeared, within the same camera segment or on the calibrated
pitch, with a reach that grows with time missing), exit/entry edge, and time
missing. A reconnection needs several scored observations, a score of at least
0.72, and a 0.15 lead over the second-best identity **and** over "an unseen
teammate" (the chance this is someone not seen yet, estimated from how alike the
other known teammates look). Otherwise the decision is deferred and logged, and
the track shows as `IDENTITY_UNCERTAIN`. For example, 51% / 48% / 45% is
deferred; a later 93% / 41% / 38% reconnects. Once resolved, the observations
collected while uncertain are labelled retroactively (`evidence: reidentified`).

A new global identity is created only when no missing identity can be this
person. About 11 identities per team is a sanity reference, not a hard rule: at
the limit a new identity needs every missing teammate to be implausible and at
least 5 s of observation, and a sanity warning is logged. More identities than
expected usually means players were recreated instead of re-identified.

**Crossings.** When two identified players overlap, their motion is recorded and
the identities are compared again after they separate (KEEP vs SWAP). Only
appearance can swap them back; trajectories alone never do. If the appearance is
too similar to tell, both identities are kept, their confidence is capped, and a
crossing warning is logged. A track whose crops repeatedly stop matching its
identity (other kit, or very different OSNet appearance) releases it.

### 4. Pitch coordinates

There is no automatic metric calibration (no field keypoint model ships with
this prototype). For pitch coordinates, pause on a clear frame, use **Mark
landmarks**, pick named landmarks (corners, halfway line ends, centre spot,
penalty and goal box corners, penalty spots) and click them on the video; four
or more that are not on one line are needed. The calibration follows camera
motion and is dropped (with a warning) when motion becomes unreliable or after a
cut; add another keyframe then. While calibrated, each person gets
`pitch: [x, y]` (x 0..1 left goal line to right goal line, y 0..1 far touchline
to near touchline, assuming 105 × 68 m), the calibrated pitch outline becomes the
playable region, and spatial re-ID evidence uses metres on the pitch. Screen
coordinates (`box`) are always stored too.

### 5. Camera motion

Pan, zoom and small rotation are estimated once per frame from background features
(people and the top overlay band are masked) and shared with BoT-SORT's Kalman
prediction. Positions used for movement, spatial re-ID and crossings are
camera-compensated within a camera segment, so a pan is not read as player
movement. Unreliable motion is never extrapolated: stabilized positions are
marked as drifting (weakening spatial evidence) and boundaries/calibrations are
dropped until the next keyframe. A camera cut ends all local tracks (identities
become missing) and starts a new segment; re-identification then relies on
appearance.

### 6. The ball

The detector proposes ball candidates; most small white things on a pitch are
not the ball. Every candidate is scored on several properties at once: detector
confidence, size against the expected ball size at that depth (0.22 m, from the
perspective player-size model), a compact round shape, grass all around it, and
its location on the pitch. Candidates that are clearly something else are
rejected with a reason:

| Reason | Meaning |
|---|---|
| `FIELD LINE` | part of a long thin white structure: touchline, halfway line, box lines, centre circle. A ball lying on a line makes a bulge and is not rejected. |
| `STATIONARY` | a white spot that stays put in camera-compensated (or pitch) coordinates, such as the penalty and centre spots or debris, or a spot seen at the same time as the tracked ball |
| `SIZE` | far too large or too small for a ball at that depth |
| `OUTSIDE PITCH` | far outside the playable field |
| `PLAYER PART` | inside a person's body: white socks, shoes, shorts |
| `TRAJECTORY` | plausible, but off the tracked ball's path |

One ball track is kept with a constant-velocity Kalman filter that moves with
the camera. A detection on its predicted path continues it. A new track (at the
start, or after the ball was lost) needs several consistent hits with real
movement within 0.6 s, so a static spot or a single confident detection never
starts one. While the ball is hidden (a player in front of it) the track is
`MISSING` and its position is predicted for up to 1 s, or 2.5 s when it
disappeared at a player's feet (it then follows that player); when it reappears
near the prediction it reconnects to the same track (`BALL REACQUIRED`). After
that, or when its confidence drops below 0.35, the ball is `UNKNOWN` rather than
a guess. Only one ball is ever shown.

## Debug view (how to test)

- **Identity debug** labels every tracked person, e.g. `A-07 · Track 91 · Team A ·
  Role: PLAYER 92% · Identity: 94%`; goalkeepers `GK-A (GK-1)`, referees
  `REF-1`; unresolved people `A-?`, `REF-?`, `GK-?`, `UNK-3`, with the reason they
  are not yet identified. Colours: violet team A, orange team B, pink goalkeepers,
  yellow referees, dotted while the identity is uncertain, grey for unknown,
  white dashed for candidates, red dashed for people rejected outside the field.
- **Pitch boundary** shows the detected playable region in light blue (teal when
  calibrated); it should sit on the white touchlines and goal lines. With
  Identity debug on, it also shows the detected boundary lines (white; yellow
  dashed when carried by camera motion), goal-end hints (orange dashed), the raw
  grass hull (dashed) and which source each edge uses (`line`, `carried line`,
  `grass edge`), plus your boundary keyframes.
- The **ball** is drawn only once: `BALL` with `Confidence: 0.94` (dashed,
  "hidden, predicted", while it is missing; with a 1 s trail in Identity debug),
  or `BALL UNKNOWN` in the corner. **Ball debug** adds every rejected candidate
  with its reason (`BALL REJECTED: FIELD LINE`, `BALL REJECTED: STATIONARY`,
  `BALL REJECTED: SIZE`, `BALL REJECTED: TRAJECTORY`, …).
- **Rejected detections** shows each rejected person with its reason
  (`REJECTED: AUDIENCE`, `REJECTED: OUTSIDE PITCH`, `REJECTED: SIZE`,
  `REJECTED: LOW PLAYER CONFIDENCE`).
- **Identity events** lists every decision, newest first, with filters. Click one
  to jump to it. A reconnection reads:

```
RE-ID EVENT
Local Track: 91
Matched Global Player: TeamA_Player_07
Appearance similarity: 0.92 (OSNet cosine 0.93)
Team match: yes
Jersey similarity: 0.95
Spatial plausibility: 0.81
Time missing: 11.2 seconds
Final identity confidence: 0.94
Second best: A-04 0.61
```

  Rejected candidates (`RE-ID REJECTED`) and deferred decisions (`RE-ID DEFERRED`,
  with the reason) are listed too, and so are role and ball decisions (filters
  **Roles** and **Ball tracking**):

```
ROLE UPDATE
Global ID: B-15 -> REF-2 (B-15 retired: it was this referee)
Local Track: 37
Old role: PLAYER (team B)
New role: REFEREE
Team A similarity: 0.08
Team B similarity: 0.61
Kit matches neither team: 74% of observations
Referee kit: 70%
Detector referee / goalkeeper votes: 74% / 0%
Referee confidence: 0.83

GOALKEEPER IDENTIFIED
Global ID: A-01 (shown as GK-A)
Team: A
Old role: PLAYER
New role: GOALKEEPER
Goal proximity score: 0.86
Penalty-area residence: 86% of observations
Uniform difference score: 0.90
Deepest / isolated: 80% / 74%
Detector goalkeeper votes: 40%
Temporal confidence: 0.81

BALL TRACK UPDATE
Candidate: x = 812, y = 466 px
Detector confidence: 0.71
Motion consistency: 0.88
Shape score: 0.74
Trajectory score: 0.91
Final confidence: 0.86
Rejected candidates since last update: STATIONARY 48, FIELD_LINE 12
```

  The same log is written to `events.log` in the run folder.
- **Match identities** lists team A, team B, goalkeepers, referees and merged
  identities (for example `B-15 → REF-2`). Hover for the role history; click an
  identity to highlight it and jump to its first observation.

Suggested checks on a real clip: nobody in the stands or on the bench gets a
label; the light-blue boundary sits on the white lines, not on the boards;
referees (including the assistants on the near touchline) show as `REF-n`, never
as a team player; each goalkeeper shows as `GK-A` or `GK-B` and keeps it away
from the goal; a white spot or a line is never `BALL`; a player who leaves and
returns keeps their ID (look for a RE-ID EVENT) or stays uncertain, never a
different player's ID; no team grows far beyond 11 identities.

## Output

Runs are saved under `runs/<video-content-hash>/<run>/`: `config.json`,
`result.json`, `identities.json` (registry and team kits, used to continue
identities), `events.log` and the latest `preview.jpg`. The viewer downloads
`result.json`:

`result.json` has `version: 3`:

- `frames[]`: `time`, `frame`, `people[]`, `rejected[]`, `pitch` (every 0.2 s:
  polygon, detected lines, goal-end hints, edge sources), `boundary`,
  `cameraReliable`, `cut`, `segment`, `calibrated`, `ball`, `ballCandidates`.
- `people[]`: `id` (global identity or `null`), `display` (`A-07`, `GK-A`,
  `REF-1`, `A-?`, `UNK-3`), `track` (local), `team` (`A`, `B` or `null`), `role`
  (`PLAYER`, `GOALKEEPER`, `REFEREE`, `UNKNOWN`), `label` (combined:
  `PLAYER_TEAM_A`, `GOALKEEPER_TEAM_B`, `REFEREE`, `IDENTITY_UNCERTAIN`,
  `CANDIDATE`, …), `state` (`candidate`, `uncertain`, `confirmed`, `unknown`,
  `rejected`), `identityConfidence`, `roleConfidence`, `box` (normalized screen
  coordinates), optional `pitch`, `zone`, `cls` (detector class), `evidence`
  (`observed`, `reidentified` or `relabelled` after a role change), and `reason`
  while unresolved.
- `ball`: `state` (`TRACKED`, `MISSING`, `UNKNOWN`), `confidence`, and unless
  unknown `track`, `box`, `center`, `velocity`, optional `pitch`, `detector`,
  and `missingFor` / `nearTrack` while missing. `ballCandidates[]`: every
  candidate's `box`, `det`, `score`, `status` and rejection `reason` / `text`.
- `match.players`, `match.goalkeepers`, `match.referees`, `match.retired`: the
  registry, separated. **Tactical analysis should use only people with an `id`
  whose role is `PLAYER` or `GOALKEEPER`, and treat low `identityConfidence`
  values with care.** Referees are never players.
- `events`, `issues`, `teams`, `summary` (identity counts, re-identifications,
  deferred decisions, crossings, sanity warnings, rejected detections by reason,
  share of on-pitch observations with an identity, ball coverage and rejected
  ball candidates by reason, role conversions, which team defends which side,
  speed).

Score a run against hand-annotated frames with the existing evaluator (it reads
this format directly):

```bat
pnpm evaluate:soccer prototypes\gpu-tracker\runs\<hash>\<run>\result.json ground-truth.json
```

## Tests

From the repository root, after setup:

```bat
prototypes\gpu-tracker\test.cmd
```

`tests/` covers pitch detection on synthetic frames and perspective broadcast
renders (touchlines vs advertising boards, run-off, stands, green seats behind
boards, goal lines vs penalty-box fronts, temporal stability, zoomed views,
crowd close-ups), foot-point filtering and audience/size rejection, assistant
referees at the touchline, part descriptors and galleries, team and role
decisions, camera motion and cuts, landmark calibration, the ball tracker (a
moving ball among the centre spot, debris, a painted line and a white sock;
occlusion; loss), and the identity manager (promotion, exits and returns,
deferral of look-alikes, referees, a referee first labelled as a team player,
returning referees, assistant referees, goalkeepers found from position and
kit, role locking, touchline candidates, team limits, crossings, goalkeeper
team, continuation across runs). `test_integration.py` runs rendered frames
through the real BoT-SORT and OSNet with spectators, a coach, a referee, a pan
and a player who leaves and returns. These are synthetic checks of the logic,
not accuracy measurements on real footage.

## Install on another computer

Requires Windows, Python 3.12, an NVIDIA GPU, an appropriate NVIDIA driver,
and several GB of disk space. From the repository root:

```bat
prototypes\gpu-tracker\setup.cmd
```

Setup creates an isolated `.venv` and installs PyTorch 2.8/CUDA 12.8 for the
RTX 5080, then downloads the checksum-pinned soccer checkpoint. If Python 3.12
is not registered with the Python launcher, set `PITCHIQ_PYTHON` to its full
`python.exe` path. On this machine setup can also use Codex's bundled Python.
The existing `public/models/osnet-x025.onnx` must be present in the checkout.
No global Python packages are changed.

## Limits

- OSNet is a general pedestrian Re-ID model, not trained on broadcast soccer.
  Same-kit teammates can look alike, especially when small or blurred; then a
  returning player stays uncertain until spatial evidence, a better crop or the
  elimination of other candidates resolves it. Expect uncertain periods rather
  than wrong labels. There is no jersey-number OCR.
- Pitch detection is heuristic (grass colour and white lines). Unusual turf,
  heavy shadow, snow or close-ups can make it unreliable; then nobody new is
  promoted until a clear view returns. Draw a boundary keyframe if needed.
- Pitch coordinates need calibration keyframes; the 105 × 68 m size is assumed.
- Substitutions are not recognised automatically: a player who goes off stays
  `MISSING`, and an incoming substitute becomes a new identity only after the
  sanity checks above (a warning is logged).
- Role decisions are heuristic and need about 5–10 s of evidence; a referee whose
  kit, shorts and detector class all look like a team player's can stay a
  player until the evidence changes. Goalkeeper team inference stays unknown
  without enough evidence. Assistant referees on the far touchline are
  rejected with the audience.
- Processing speed depends on the CPU too: OSNet embeddings run on the CPU for
  every on-pitch detection, as before.
- Off-screen positions are unknown; no observations are fabricated. The ball
  tracker is heuristic: a ball high in the air, a long hidden spell or a crowded
  goalmouth can leave it `UNKNOWN`. No ball accuracy is claimed.

The server binds only to loopback. Environments, downloaded weights, settings,
uploaded videos and generated runs are ignored by Git.

## Model provenance and licensing

- Detector: [gianpaj/football-players-detection-1](https://huggingface.co/gianpaj/football-players-detection-1).
  Exact revision, size, and SHA-256 are recorded in `model.json` and verified
  before loading. Its model card declares AGPL-3.0. Its training data is the
  [Roboflow football-players-detection dataset](https://universe.roboflow.com/roboflow-jvuqo/football-players-detection-3zvbc).
- Detector framework and tracker: [Ultralytics v8.3.203](https://github.com/ultralytics/ultralytics/tree/v8.3.203),
  under its AGPL-3.0 / enterprise licensing options.
- Appearance model: existing [OSNet model notes](../../public/models/OSNET-MODEL.md)
  and [MIT notice](../../public/models/OSNET-LICENSE.txt).

Local use has no per-video API fee, but software/model license obligations still
apply. This prototype is not a blanket clearance to distribute these components
inside a closed-source commercial PitchIQ release.
