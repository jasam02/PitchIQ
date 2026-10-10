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
- One match ball is followed as a persistent track (`BALL-1`), never rediscovered
  frame by frame. Painted lines, static spots and things fixed to players (wrist
  tape, boots) are rejected; while hidden the ball is predicted; when it cannot
  be confirmed it is `BALL UNKNOWN`, never a guess.

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
  -> 2D match view .................. tactical.js          top-down recreation from the saved result, in step with the video
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

- `PLAYER_TEAM_A/B`: at least 70% of kit votes for one team, and the person does
  not look more like the known referees than like that team's players (then they
  stay a candidate rather than becoming a wrong player).
- `REFEREE`, by the strongest of four routes: the detector's referee votes; the
  person's **whole appearance** (OSNet embedding plus jersey, shorts, socks and
  body-layout colours, over the last few clean crops) looking more like the
  known referees than like either team's players; **touchline behaviour** (time
  on or up to 4 m outside one pitch edge, moving along it) together with
  referee appearance, the referee kit or detector votes (an assistant referee);
  or a kit like the learned referee kit, or matching neither team and backed by
  detector votes or referee appearance, while following play away from the
  goals. A kit matching neither team is never a referee by itself, on the pitch
  or on the touchline (a team kit in shadow, a coach pacing the line). A kit
  that matches a team holds the detector and position routes back unless the
  detector's votes are a clear majority or the match is loose (near the edge of
  that team's colour spread: a pink referee next to a garnet kit); it never
  holds back the appearance and touchline routes, which already weigh the
  person against that team's players. When that team already has its full
  count of identities, another "team player" is doubted and the referee
  threshold drops. Once a referee is locked, every crop in its gallery joins
  the referee appearance gallery, so the other officials in the same uniform
  are found by looking like them; a person the detector mostly calls a referee
  also seeds the referee kit. Central-referee behaviour (following the ball
  without playing it, sitting outside the supposed team's shape, moving with
  neither team) adds to the appearance route; it never decides alone.
- `GOALKEEPER`: the detector's goalkeeper votes; or time in or near a penalty
  area in a kit unlike the outfield kits (or like a known goalkeeper kit) while
  being the deepest or most isolated person. At least one strong cue (detector,
  goal area or goalkeeper kit) is required. This only makes the track a
  **goalkeeper candidate** (`GK-?` / `GK-B?`, label `GOALKEEPER_CANDIDATE`): a
  goalkeeper *identity* is confirmed separately, see below.
- Otherwise the person stays `CANDIDATE` (later `UNKNOWN`) and is never forced
  into team A or B.

Roles keep being re-evaluated after promotion, about once a second. A referee or
goalkeeper role with a role confidence of at least 0.7 is **locked**: it
survives leaving the goal area, odd frames or a detector that changes its mind,
and is never turned back into a team player by a few frames.

**Goalkeeper is a role with a place.** A goalkeeper candidate never creates,
converts or locks a goalkeeper identity (`GK-n`, shown as `GK-A` / `GK-B`) until
its evidence is *confirmed*: at least 3 s (15 observations) of goalkeeper
evidence with a temporal confidence of 0.7, a spatial route (goal-area residence
in at least half the observations; or consistently the deepest person, alone or
with detector / kit support; or sustained detector votes in a distinct kit plus
some goal-area or deepest-person evidence), and nothing that says sideline:
feet on or outside the touchline in more than 20% of observations, the touchline
corridor (on the line to 4 m out) in 30% or more, or referee evidence of 0.5 veto
it. The detector calling someone a goalkeeper is never enough on its own, and a
person whose feet say sideline is not even promoted as a candidate. Until then
the track shows `GK-?` and `GOALKEEPER CANDIDATE` events list the evidence
(detector votes, pitch state inside / on or outside the touchline, goal
proximity, penalty-area residence, deepest / isolated, temporal confidence and
the confirmation state); a candidate whose feet move to the sideline is dropped
(`GOALKEEPER CANDIDATE DROPPED`) and becomes `OUT` / `UNK` like anyone else.

Each team's goalkeeper **slot** is a live relationship, not a historical fact
(`goalkeeperSlots` in the result summary: `{"A": "GK-2", "B": null}`). It stays
empty until a confirmed candidate exists; `GK-?` is better than the wrong person
as `GK-B`. A confirmed goalkeeper is **revoked** only by sustained contradictory
evidence about that person: referee evidence of 0.7, outside the pitch for half
of the last 10 s away from any goal, or patrolling the touchline away from any
goal, on three consecutive checks (about 3 s) over a full 10 s window, and never
in the first 5 s after a track was bound. Walking upfield while the detector
calls them a player, or fetching a ball behind their own goal line, is not a
contradiction. Revocation (`GOALKEEPER REVOKED`) frees the slot at once: the
identity is kept, marked `revoked`, no longer counts, seeds a keeper kit, blocks
a team or matches returning keepers, its observations are relabelled
`GOALKEEPER_REVOKED`, and the track is re-decided. Candidates compete, judged by
what confirmed them (`keeperGoal`, the strongest goal-area residence ever seen),
not by the last few seconds: a visible owner never loses its status to a
candidate; a weakly confirmed owner (never seen at a goal) loses the slot to a
candidate confirmed at that goal (revoked when absent, de-teamed when visible);
an owner itself confirmed at a goal only loses the *team* once it has been out
of view for 10 s while another keeper stands in that goal (`GOALKEEPER TEAM
RELEASED`: it stays a goalkeeper, re-identifiable, re-teamed by the side votes).
The keeper-kit team vote is evidence, never ownership. A goalkeeper or referee
candidate in a clearly different jersey from a missing one is never held as
that identity's possible return, and a candidate confirmed at a goal for 10 s
is decided (re-identified at a lower bar or created) rather than deferred
forever behind a similar-looking stale owner. A track whose feet say sideline
(vetoed) is neither promoted as a goalkeeper candidate nor matched to any
missing goalkeeper identity. A confirmed player changes role
only after two consecutive checks (about 2 s) ask for the same new role, so a
label never flips back and forth. A player identity whose evidence
becomes clearly referee is converted: a matching missing referee identity, or a
new `REF-n`, takes over the track; if the player identity only ever followed
this person it is merged into the referee (`RETIRED`) and every observation
already labelled with it is relabelled, so a referee never stays `B-15`. A
player identity that turns out to be the goalkeeper keeps its ID, gets the
locked goalkeeper role and is shown as `GK-A` / `GK-B`, earlier observations
included. A returning referee or goalkeeper is matched by their own appearance
even when the detector or the kit vote calls them a player. Every change is
logged (`ROLE UPDATE`, `GOALKEEPER CANDIDATE`, `GOALKEEPER IDENTIFIED`,
`GOALKEEPER TEAM`, `GOALKEEPER REVOKED`), with the
evidence behind it: appearance against the known referees and both teams,
detector votes, touchline time and movement along the line, formation
consistency, ball-following behaviour, team count. The same evidence is kept
as `roleEvidence` on every identity and shown as a one-line `why` under each
person in the debug overlay. Referees also carry the kind of official the
evidence suggests (`ASSISTANT_REFEREE` on the touchline, `CENTER_REFEREE`
inside the pitch, `UNKNOWN_OFFICIAL`); all are shown as `REF-n`. An
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
team otherwise. Two goalkeepers for one team are flagged as a possible duplicate;
a confirmed keeper defending a goal whose team slot a weaker absent keeper holds
takes that slot (see the goalkeeper slot rules above).

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
| `official`, `roleEvidence` | referees: `ASSISTANT_REFEREE` / `CENTER_REFEREE` / `UNKNOWN_OFFICIAL`; the latest role decision's evidence scores |
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

The ball tracker never asks "which white blob looks most like a ball in this
frame?" but "which detection is the ball I am already following?". It is a
state machine with one active ball:

```
SEARCHING ─(several consistent, moving detections)─▶ LOCKED ─(no detection)─▶ OCCLUDED
   ▲                                                   ▲                         │ short gap: predicted,
   │                                                   └───(reconnect)───────────┤ kept at the player
   │                                                                             ▼ that hides it
   └──────────── LOST ◀─(1.5 s free / 3 s at a player)─ RECOVERING (search area grows, confirmation needed)
```

Every frame the active ball is **predicted** with a constant-velocity Kalman
filter that moves with the camera, a **search gate** is placed around the
prediction, and the candidates inside it are scored with **trajectory
consistency as the strongest term** (Mahalanobis distance from the prediction,
proximity, appearance, minus penalties). The gate is tight while `LOCKED`
(about 0.45 m plus a share of the speed), wider while `OCCLUDED`, grows with the
gap while `RECOVERING`, and becomes physically wide (a kick's worth of travel)
only when a player is within reach of the prediction, because a touch changes
the velocity at once; the filter's uncertainty is inflated at the same moment.
**Switching resistance**: the candidate on the predicted path keeps the track
unless another one scores at least 0.15 higher; a candidate outside the tight
gate, or one that reappears after a gap, needs a previous sighting in the same
place (its tracklet) and independent motion — one lucky detection never moves
the ball or brings it back. A ball hidden by a player is carried with that
player (the track follows their feet, unless the ball was moving away from
them) and reconnects when it reappears, keeping the same track id. A free ball
the detector loses sight of is predicted for 2.5 s before the track is given
up (broadcast detectors miss a small rolling ball for a while), and a lost
ball found again near where it vanished keeps its id. A new track needs
several consistent detections that moved at least half a metre; detections
that all lie inside one person's lower box (boots, socks, or a ball glued to
the feet) need 1.5 s of history first, because only time tells them apart.

Every candidate, not only the ball, is linked into a short **tracklet**, which
gives temporal evidence:

| Evidence | Meaning |
|---|---|
| `FIELD_LINE` | part of a long thin bright structure (touchline, halfway line, box lines, circle) or on a fitted pitch marking while the bright structure is stretched; a ball lying on a line makes a bulge and is kept |
| `STATIC` | standing still for 2 s in camera-compensated (or calibrated pitch) coordinates, at a spot anchored where it was first seen, or at a known painted spot; a track that never moved and rests on such a spot is dropped and the spot remembered as a false positive |
| `PLAYER_ATTACHED` | the same place inside one person's box for many frames: wrist tape, gloves, boots, socks. Inside the upper body 0.8 s of stability is enough; at the feet, where the ball lives, 2 s. A single frame inside the upper body is only a prior, and a tracklet that was attached a moment ago stays attached while an arm swings outside the box |
| `OTHER_OBJECT` | seen at the same time as the confidently tracked ball, elsewhere (there is one ball). Only a mature lock may say so (at least 0.5 s old, confident, and it has moved at least a metre like a ball), never about things within 1.5 m of the ball, and the flag lapses when that track is lost: a wrong lock on a boot cannot disqualify the real ball next to it |
| `SIZE`, `OUTSIDE_PITCH` | impossible for a ball at that depth (0.22 m scaled by the perspective player-size model), or far outside the playable field |
| `TRAJECTORY`, `LOW_SCORE` | plausible but off the tracked ball's path, or in the gate with too weak a score |

Independent motion is measured against the field (camera-compensated or pitch
coordinates) and against the player a candidate sits on; shape and colour are
only small terms, because the real ball is often blurred, elongated or a few
pixels wide. `tests/ball_eval.py` renders a panning broadcast sequence with all
of these distractors and prints the metrics that matter: frames the SAME ball
was followed, false positives, track switches, hidden frames bridged by
prediction, incorrect re-acquisitions and continuity:

```bat
.venv\Scripts\python prototypes\gpu-tracker\tests\ball_eval.py
```

## Debug view (how to test)

- **Identity debug** labels every tracked person, e.g. `A-07 · Track 91 · Team A ·
  Role: PLAYER 92% · Identity: 94%`; goalkeepers `GK-A (GK-1)`, referees
  `REF-1`; unresolved people `A-?`, `REF-?`, `GK-?`, `UNK-3`, with the reason they
  are not yet identified; and under each person the role evidence in one line
  (`looks ref 0.84 A 0.16 B 0.72 | touchline 94% along 0.91 | ASSISTANT REFEREE 0.98`). Colours: violet team A, orange team B, pink goalkeepers,
  yellow referees, dotted while the identity is uncertain, grey for unknown,
  white dashed for candidates, red dashed for people rejected outside the field.
- **Pitch boundary** shows the detected playable region in light blue (teal when
  calibrated); it should sit on the white touchlines and goal lines. With
  Identity debug on, it also shows the detected boundary lines (white; yellow
  dashed when carried by camera motion), goal-end hints (orange dashed), the raw
  grass hull (dashed) and which source each edge uses (`line`, `carried line`,
  `grass edge`), plus your boundary keyframes.
- The **ball** is drawn only once: `BALL 0.94`, or `BALL PREDICTED` (dashed)
  while it is hidden, with the track id, state and speed, the predicted next
  position (a cross) and a 1 s trail in Identity debug; otherwise `BALL UNKNOWN`
  in the corner. **Ball debug** adds every other candidate with its label and
  score: `CANDIDATE`, `FIELD_LINE`, `STATIC`, `PLAYER_ATTACHED`, `OTHER_OBJECT`,
  `TRAJECTORY_FAIL`, `LOW_SCORE`, `SIZE`, `OUTSIDE_PITCH`.
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
Appearance vs known referees / team A / team B: 0.84 / 0.16 / 0.52
Kit matches neither team: 74% of observations
Referee kit: 70%
Detector referee / goalkeeper votes: 74% / 0%
Team formation consistency: 0.31
Ball-following behaviour: 0.79
Referee movement behaviour: 0.68
Final: CENTRE REFEREE 0.83
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
Track BALL-1: LOCKED, confidence 0.94, speed 7.8 m/s, age 12.0 s
BALL CANDIDATE #1
Detector confidence: 0.84
Trajectory consistency: 0.93
Distance from prediction: 12 px
Independent motion: 0.88 (6.1 m/s)
Field line overlap: 0.03
Player attachment score: 0.04
Static object score: 0.00
Size / shape / isolation / on pitch: 0.95 / 0.80 / 1.00 / 1.00
FINAL SCORE: 0.91
ACCEPTED
BALL CANDIDATE #4
Detector confidence: 0.89
Distance from prediction: 430 px
Independent motion: 0.61 (1.8 m/s)
Field line overlap: 0.00
Player attachment score: 1.00 (same place on track 37 for 1.5 s)
...
REJECTED: PLAYER_ATTACHED
Rejected candidates since the last update: STATIC 48, FIELD_LINE 12, PLAYER_ATTACHED 25
```

  `BALL ACQUIRED`, `BALL REACQUIRED` (the same ball found again after a gap),
  `BALL TRACK SWITCH`, `BALL RECOVERING`, `BALL LOST` and `BALL CANDIDATE
  REJECTED` (a confident candidate in the gate turned down, with the same
  breakdown) explain every change of the ball track.

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

## 2D match view

Directly below the video, the review page recreates the match on a top-down
pitch from the saved result (`tactical.js`): team players and goalkeepers as
circles in the colours you pick (**Team A colour**, **Team B colour**; the
choice is remembered per video and never changes the tracking's A/B), `GK`
inside the goalkeeper markers, optional **Show player IDs** labels, and the ball
as a smaller white circle, dashed while its position is predicted. Referees,
bench, crowd and rejected detections are never drawn. The video is the only
clock: play, pause, seek, scrub and replay move both together.

Positions are the pipeline's pitch coordinates (`pitch` on each observed
person and on the ball: the feet, through the landmark calibration carried by
the camera-motion estimate, normalized on a 105 × 68 m pitch), never screen
coordinates, so the board stays fixed while the camera pans and zooms and
shows the whole pitch however little of it the camera sees. Without
calibration keyframes there are no positions and the board says so; where the
calibration is dropped (unreliable camera motion, a cut) nothing is placed
until the next keyframe. Who is drawn comes from the final registry, so a
player identity later found to be a referee is not shown as a player anywhere.

Between frames the markers are linearly interpolated, and positions are
smoothed with a short centred window (the whole history is known, so there is
no lag). A person the tracker lost stays for 0.6 s at their last position
(fading, dashed ring), then disappears: nobody is placed by guesswork. A gap
of up to 1 s between two observations of the same identity is bridged by
interpolation. A lost ball disappears (only a loss shorter than 0.15 s, a
dropped frame or two, is bridged); a hidden ball is drawn only while the ball
tracker still predicts it with confidence of at least 0.5. **Debug
positions** shows each marker's pitch x / y, position source (`OBSERVED`,
`INTERPOLATED`, `HELD`, `PREDICTED`) and confidence, plus the frame, time and
calibration state, to check that video and board correspond.

The renderer draws fixed layers (pitch, overlays, players, labels, ball, debug)
and exposes `tactical.overlays`, a list of `(ctx, geometry, state) => …` hooks
drawn between the pitch and the players, for trails, heat maps, formations or
passing lanes later. `tests/test_tactical.py` checks the view in a headless
browser against a synthetic calibrated clip when Playwright, a Chromium and
ffmpeg are installed (`pip install playwright` and `playwright install
chromium`, or point `PITCHIQ_CHROMIUM` at a browser).

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
  (`observed`, `reidentified` or `relabelled` after a role change), `reason`
  while unresolved, and `why` (the role evidence in one line) on evidence ticks.
- `ball`: `state` (`TRACKED`, `MISSING`, `UNKNOWN`), `phase` (`SEARCHING`,
  `LOCKED`, `OCCLUDED`, `RECOVERING`), `confidence`, and unless unknown
  `track` (`BALL-n`), `observed`, `box`, `center`, `predicted` (next frame),
  `velocity`, `speed` (m/s), `direction`, `acceleration`, `age`, `lastSeen`,
  `lastConfident`, optional `pitch`, `detector` and `score`, and `missingFrames`
  / `missingFor` / `nearTrack` while hidden. `ballCandidates[]`: every
  candidate's `box`, `det`, `score`, `status` (`ball`, `candidate`, `rejected`),
  `reason` / `text`, and `scores` (detector, size, shape, isolation, field, line,
  attachment, motion, static, trajectory, distance, penalty).
- `match.players`, `match.goalkeepers`, `match.referees`, `match.retired`: the
  registry, separated (referees with `official`, every identity with
  `roleEvidence`). **Tactical analysis should use only people with an `id`
  whose role is `PLAYER` or `GOALKEEPER`, and treat low `identityConfidence`
  values with care.** Referees are never players.
- `tactical`: what the 2D match view needs beyond the frames: the coordinate
  convention, how many frames carry pitch coordinates (`calibratedFrames`,
  `calibratedShare`) and `goalEnds` (which team defends the left / right goal
  line, when the tracking settled it).
- `events`, `issues`, `teams`, `summary` (identity counts, re-identifications,
  deferred decisions, crossings, sanity warnings, rejected detections by reason,
  share of on-pitch observations with an identity, ball coverage, ball tracks,
  reconnections and losses, time per ball state, rejected ball candidates by
  reason, role conversions, assistant and centre referees, which team defends
  which side, speed).

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
decisions, camera motion and cuts, landmark calibration, the ball tracker (the
rendered, panning sequence of `ball_eval.py`: a pass, a dribble, a long shot
across the halfway line and the centre circle, players overlapping the ball, a
wrist tape detected with a higher confidence than the ball, white boots, the
centre and penalty spots, debris on the ball's path; occlusions, a long hide at
the tape-wearer's feet, an undetected spell with re-acquisition of the same
ball, the ball leaving the scene, confident static spots, a calibrated pitch),
and the identity manager (promotion, exits and returns,
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
  The 2D match view therefore shows people only while the camera is calibrated
  and only those the camera sees; off-screen players are not placed.
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
  tracker is heuristic: it needs the ball to move before the first lock (a ball
  at rest cannot be told from a painted spot), and a ball high in the air, a
  long hidden spell or a crowded goalmouth can leave it `UNKNOWN`. Attachment to
  a player needs the local tracker's person boxes; white boots on a player the
  ball has not passed yet are told apart mainly by their short history. No ball
  accuracy is claimed on real footage; the evaluation is synthetic.

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
