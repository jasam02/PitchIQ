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
- Referees are tracked separately and never counted as players.

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
Accepted re-identifications, crossing corrections and sanity warnings are also
printed in the console.

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
  -> pitch detection ................ soccer/pitch.py       grass region + touchline / goal-line detection, every 0.2 s
  -> outside-field filter ........... soccer/relevance.py   foot point in the pitch polygon, audience, size check
  -> local multi-object tracker ..... soccer/local.py       BoT-SORT with OSNet features (temporary track IDs)
  -> appearance ..................... soccer/appearance.py  OSNet embedding + jersey / shorts / socks colour, quality
  -> team + role classification ..... soccer/teams.py       two kits learned online; detector class votes over time
  -> pitch coordinates .............. soccer/calibration.py optional landmark calibration, carried through pans
  -> GLOBAL IDENTITY MANAGER ........ soccer/identity.py    candidates, promotion, re-ID, swap guard, sanity checks
  -> persistent soccer observations . result.json           per frame: global ID, local track, role, team, confidence
```

`pipeline.py` wires the detector into `soccer/tracker.py`, which runs every step
after detection. The `soccer` package needs only NumPy and OpenCV (BoT-SORT is
imported lazily), so all of its logic is unit-tested without a GPU.

### 1. Only people on the pitch

- The playable region is re-detected from the image every 0.2 s and carried with
  the camera motion in between. It is never a fixed rectangle. Grass is
  segmented (worn and yellow-green turf included), painted lines and players are
  bridged, but a band of advertising boards is not, so green seats behind the
  boards never join the pitch. Long white boundary lines with only a thin run-off
  strip beyond them clip the grass region, so a coach standing on the run-off
  grass is outside. A line that briefly disappears is carried for up to 2 s.
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

Roles come from accumulated votes, never one frame: `PLAYER_TEAM_A/B` needs at
least 70% of kit votes for one team; `REFEREE` and `GOALKEEPER` need a 60%
detector majority. A "referee" whose kit clearly matches a team, or a
"goalkeeper" in a team kit, is held back until the evidence agrees. An
established team only changes after 80% of the last 15 votes point to the other
kit (the identity is then released and re-identified).

Team kits are learned automatically from the torso (upper jersey) colour of people
the detector calls players, so referees and keepers do not pull a team prototype.
Two kits must each have enough support and be clearly separated; one kit under
changing light is not split into two teams. A/B naming is arbitrary but stays
fixed during a run and when identities are continued.

A goalkeeper's kit matches neither team. Their team is inferred from the two
outfield players deepest towards that goalkeeper's goal (usually that team's
defenders, because of the offside line); it is decided after at least 15
consistent observations with a 75% majority, and stays unknown otherwise.

### 3. Local tracks vs global identities

BoT-SORT local tracks are temporary: an occlusion, an exit or a camera cut ends
them. The **global identity manager** keeps a registry of real people:

| Field | Meaning |
|---|---|
| `id`, `longId` | `A-07` / `TeamA_Player_07`, `GK-1` / `Goalkeeper_01`, `REF-1` / `Referee_01` |
| `team`, `role` | `PLAYER_TEAM_A`, `GOALKEEPER_TEAM_B`, `REFEREE`, … |
| `status` | `ACTIVE`, `MISSING`, `OFF_SCREEN` (left at an image edge) |
| gallery | up to 6 diverse, high-quality samples (OSNet embedding + kit colours) |
| last seen | time, image box, pitch position, camera-compensated velocity, exit edge |
| confidences | identity, team and role confidence; recent history |

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

## Debug view (how to test)

- **Identity debug** labels every tracked person, e.g. `A-07 · Track 91 · TEAM A ·
  PLAYER · Identity: 94%`, and shows why unresolved people are not yet identified.
  Colours: team colours for players, magenta goalkeepers, yellow referees, grey
  dotted for identity uncertain / unknown, white dashed for candidates, red dashed
  for people rejected outside the field.
- **Pitch boundary** shows the detected playable region (green; cyan when
  calibrated), the detected boundary lines (white; yellow dashed when carried by
  camera motion), the raw grass hull (dashed, debug only) and your boundary
  keyframes.
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
  with the reason) are listed too. The same log is written to `events.log` in the
  run folder.
- **Match identities** lists team A, team B, goalkeepers and referees. Click an
  identity to highlight it and jump to its first observation.

Suggested checks on a real clip: nobody in the stands or on the bench gets a
label; referees show as `REF-n`; a player who leaves and returns keeps their ID
(look for a RE-ID EVENT) or stays uncertain, never a different player's ID; no
team grows far beyond 11 identities.

## Output

Runs are saved under `runs/<video-content-hash>/<run>/`: `config.json`,
`result.json`, `identities.json` (registry and team kits, used to continue
identities), `events.log` and the latest `preview.jpg`. The viewer downloads
`result.json`:

- `frames[]`: `time`, `frame`, `people[]`, `rejected[]`, `pitch` (every 0.2 s),
  `boundary`, `cameraReliable`, `cut`, `segment`, `calibrated`, `ball`,
  `ballCandidates`.
- `people[]`: `id` (global identity or `null`), `track` (local), `label`, `box`
  (normalized screen coordinates), optional `pitch`, `role`, `team`, `state`
  (`candidate`, `uncertain`, `confirmed`, `unknown`, `rejected`), `identity`
  (confidence), `zone`, `cls` (detector class), `evidence` (`observed` or
  `reidentified`), and `reason` while unresolved.
- `match.players`, `match.goalkeepers`, `match.referees`: the registry, separated.
  **Tactical analysis should use only people with an `id` whose role is
  `PLAYER_TEAM_*` or `GOALKEEPER_TEAM_*`, and treat low `identity` values with
  care.** Referees are never players.
- `events`, `issues`, `teams`, `summary` (identity counts, re-identifications,
  deferred decisions, crossings, sanity warnings, rejected detections by reason,
  share of on-pitch observations with an identity, speed).

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

`tests/` covers pitch detection on synthetic broadcast frames (touchlines,
run-off, stands, green seats behind boards, zoomed views, crowd close-ups),
foot-point filtering and audience/size rejection, part descriptors and
galleries, team and role decisions, camera motion and cuts, landmark
calibration, and the identity manager (promotion, exits and returns, deferral of
look-alikes, referees, touchline candidates, team limits, crossings, goalkeeper
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
- Goalkeeper team inference is a heuristic and stays unknown without enough
  evidence. Assistant referees standing outside the touchline are rejected like
  other people outside the field.
- Processing speed depends on the CPU too: OSNet embeddings run on the CPU for
  every on-pitch detection, as before.
- Off-screen positions are unknown; no observations are fabricated. Ball output
  is provisional.

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
