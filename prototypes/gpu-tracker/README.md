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
  **global identities** (`A-07`, `B-03`, `GK-1`). A player who leaves the
  view and comes back is re-identified and keeps their global identity.
- When identity is uncertain, the person stays *identity uncertain* instead of
  being guessed. A wrong merge is worse than a short unknown.
- Identity, team and role are separate. Only match entities get identities: team
  players and goalkeepers (recognised from position and kit, shown by team as
  `GK-A`, `GK-B`). Referees, assistant referees, coaches and staff are **ignored
  people** (`NON_TARGET`): they keep a local box, so the tracker knows when
  someone overlaps a player, but never get an identity, a team, an appearance
  memory or a place in the 2D view, and stay unmarked in the normal overlay
  (debug shows them faintly as `IGNORE`). A confirmed player never becomes one.
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

Frames are decoded and detected ahead on a worker thread while the tracker works
on the current frame: the whole-frame detector can take several consecutive
frames per call (`detectBatch`, 1 by default: it bought nothing measurable on the
RTX 5080), the ball tile scans go to the detector in one call and run
only while the ball tracker is not following the ball (plus one safeguard scan a
second while it is, `ballScanEvery`), and the pitch fit is vectorized with the
same arithmetic as before (including its finite-difference Jacobian). The
detectors and the appearance model stay loaded in the server between runs, and
the offline ball decode costs every link into a frame in one go, with the same
path as before bit for bit. See PROGRESS.md (speed pass 3) for what each change
was verified against.

Choose a segment up to 180 seconds long. For a longer match, process consecutive
segments and tick **Continue identities from the previous run**: the next run
starts from the saved global identities (and their appearance galleries) of the
latest completed run, so `A-07` at minute 5 is still `A-07` at minute 70 when
the evidence allows it. The initial 1280 setting combines 960 and 1280
whole-frame passes: the lower-resolution pass retains nearby players that this
checkpoint sometimes misses at higher resolutions. Choose 960 for a single,
faster pass. Extra small-ball scans inspect overlapping crops every other frame
while the ball is not being followed (once a second while it is) and take longer.

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
  -> pitch coordinates .............. soccer/registration.py landmark keyframes carried both ways, re-registered to their views, remapped offline
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
- Every analysed frame gets a **pitch confidence** (0..1, shown on the video as
  `PITCH CONFIDENCE: 0.93 · CONFIRMED BY PAINTED LINES`) with the evidence
  behind it: both painted touchlines found is HIGH; a grass contour alone is
  LOW, because grass runs on beyond the touchline into run-offs, car parks and
  neighbouring pitches. The playable region comes from the first of these that
  is trusted: the tracked **pitch model** (section 4) at MEDIUM or better, the
  painted boundary lines at HIGH, else the grass region in **robust mode**,
  where feet just beyond the edge are held at the boundary instead of being
  rejected, new people must persist twice as long before they become
  identities, and after 3 s the stretch is reported as a section needing
  confirmation (*Pitch confirmation* in section 4). Broadcast footage with
  visible touchlines never leaves the automatic path. Mode and confidence are
  stored per frame (`pitchMode`, `pitchConfidence`).
- Far-away people on low, wide-angle cameras: the expected person size per image
  row comes from the heights actually seen there when the linear perspective
  model is clearly off (wide lenses compress the far field); people expected
  smaller than 36 px get a wider size tolerance and, when their box continues a
  tracked person, are kept even when cut short; and whenever people at the
  pitch's far edge are expected under 30 px (whatever the pitch mode), extra
  640 px tiles are run over that band (`farTilePasses` in the summary), with one
  suppression across the tiles' overlaps. Tile detections are provisional:
  they count only once they repeat on three consecutive frames or continue a
  tracked person (`REJECTED: FAR DETECTION NOT CONFIRMED` otherwise). A
  confirmed one is handed to the local tracker with a tracking score of 0.45
  (its own, from an upscaled tile, is below the tracker's new-track threshold)
  while the person keeps the true score, so a far player can start a track
  (`summary.detection.farStarts`). None of this triggers on broadcast footage,
  where the smallest players are about 40 px.
- A confirmed player who is small (under 60 px) and whose local track was just
  lost (under a second) is looked for again at full resolution: the detector
  runs on a crop of three times the predicted box (at least 256 px, at most
  four crops per frame, `localCropPasses`), and a detection found there counts
  only within an overlap of 0.3 with the prediction it was made for
  (`REJECTED: CROP DETECTION AWAY FROM THE PREDICTION` otherwise;
  `summary.detection.cropDetections`). No crop is made for a player lost next
  to another tracked person, and a crop detection overlapping another detected
  person is not accepted (`cropSkippedCrowded`, `cropRejectedCrowded`): a
  player lost behind someone is left to the identity manager's encounter
  handling, because a crop there finds the other person and would carry the
  lost identity onto them (measured on the broadcast clip, where it swapped
  two players' teams for 25 and 56 s). Nothing new is ever created from a crop.
- A BoT-SORT small-object setting (second-stage threshold 0.05 and a 90-frame
  buffer while the far band exists; `SMALL_OBJECT_TRACKING` in
  `soccer/tracker.py`) exists but is off: measured on the test clips it changed
  nothing on two windows and added a duplicate identity and fragmentation on
  the third (`PROGRESS.md`, Phase 3).
- Feet a little beyond (5% of the frame height) a far edge that is not a painted
  line (a predicted or grass edge), continuing a tracked person, stay in the
  `boundary` zone instead of being rejected as audience
  (`summary.detection.farBoundaryContinued`); a painted far touchline keeps its
  full meaning.
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
  an ignored person (`NON_TARGET`), never a player.
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
  not look more like the officials than like that team's players (then they
  stay a candidate rather than becoming a wrong player).
- `REFEREE` (a filter, not an identity: an unidentified person read this way
  becomes `NON_TARGET`, see below), by the strongest of four routes: the
  detector's referee votes; the person's **whole appearance** (OSNet embedding
  plus jersey, shorts, socks and body-layout colours, over the last few clean
  crops) looking more like the officials than like either team's players; **touchline behaviour** (time
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
  count of identities *on the pitch now*, another "team player" is doubted and
  the referee threshold drops. The **officials' look** is anonymous: the clean
  crops of people ignored as referee-like in the last minute whose reading had
  independent support (detector votes, the referee kit or the touchline), so the
  other officials in the same uniform are found by looking like them; nobody is
  identified by it. It never decides alone: a referee reading that nothing but
  the officials' look supports (no detector votes, referee kit, touchline,
  referee movement or kit matching neither team) is decided again without it. A
  person the detector mostly calls a referee also seeds the referee kit.
  Central-referee behaviour (following the ball without playing it, sitting
  outside the supposed team's shape, moving with neither team) adds to the
  appearance route.
- `GOALKEEPER`: the detector's goalkeeper votes; or time in or near a penalty
  area in a kit unlike the outfield kits (or like a known goalkeeper kit) while
  being the deepest or most isolated person. At least one strong cue (detector,
  goal area or goalkeeper kit) is required. This only makes the track a
  **goalkeeper candidate** (`GK-?` / `GK-B?`, label `GOALKEEPER_CANDIDATE`): a
  goalkeeper *identity* is confirmed separately, see below.
- Otherwise the person stays `CANDIDATE` (later `UNKNOWN`) and is never forced
  into team A or B.

Roles keep being re-evaluated after promotion, about once a second. A confirmed
goalkeeper role with a role confidence of at least 0.7 is **locked**: it
survives leaving the goal area, odd frames or a detector that changes its mind,
and is never turned back into a team player by a few frames.

**Referees are ignored people, not identities.** A person whose own evidence
reads referee-like becomes `NON_TARGET` (display `IGNORE`, a `hint` of
`REFEREE_LIKE` or `SIDELINE_OFFICIAL`; event `NON-TARGET PERSON`): no global
identity, team, gallery or 2D position, only the local box, which still counts
for overlaps (a player overlapping them has their memory frozen) and for the
ball's boot check. Identity comes first: before a referee-like reading ignores
someone, the missing players they may be are compared (a missing player scoring
near the reconnection threshold on their crops and place keeps them a player
candidate, `ROLE UPDATE DEFERRED`), and no new player identity is ever created
from referee-like evidence. Being ignored is not final: two seconds of clear
player or goalkeeper evidence make the person a candidate again (`NON-TARGET
PERSON RECONSIDERED`), and their frames are kept for retroactive labelling and
the offline pass. A **confirmed player is never turned into an ignored person**
by a role reading: the reading is logged as `ROLE UPDATE IGNORED` (raw role
`NON_TARGET`, the reason, "identity, team and role kept"); a local track that
moved to someone else is split by the switch check instead. The only exception
is a player identity that never became an *established* player — fewer than 25
clean kit votes, or under 60% of them clearly (not loosely) in its team's kit —
and that showed the same referee-like person from its start (its first crops
look like the crops now), read as referee-like on four clean checks in a row
with independent support: a non-target person promoted by mistake (a pink
referee next to a garnet kit, promoted before any official was seen). It is
**withdrawn** (`PLAYER IDENTITY WITHDRAWN`, status `WITHDRAWN`, kept in
`match.withdrawn`), every frame it labelled becomes `NON_TARGET` and leaves the
2D view, and the person is ignored from then on. Referee identities saved by an
earlier version are not loaded; a referee you corrected is recognised by
appearance and left out like someone you marked as not part of the match.

**Goalkeeper is a role with a place.** A goalkeeper candidate never creates,
converts or locks a goalkeeper identity (`GK-n`, shown as `GK-A` / `GK-B`) until
its evidence is *confirmed*: at least 3 s (15 observations) of goalkeeper
evidence with a temporal confidence of 0.7, a spatial route (goal-area residence
in at least half the observations; or consistently the deepest person, alone or
with detector / kit support; or sustained detector votes in a distinct kit plus
some goal-area or deepest-person evidence), and nothing that says sideline:
feet on or outside the touchline in more than 20% of observations, the touchline
corridor (on the line to 4 m out) in 30% or more, or referee-like evidence of 0.5
veto it. The detector calling someone a goalkeeper is never enough on its own, and a
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
evidence about that person: referee-like evidence of 0.7 away from any goal (a
keeper at their goal is never made non-target by a reading), outside the pitch for half
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
The keeper-kit team vote is evidence, never ownership. A goalkeeper
candidate in a clearly different jersey from a missing one is never held as
that identity's possible return, and a candidate confirmed at a goal for 10 s
is decided (re-identified at a lower bar or created) rather than deferred
forever behind a similar-looking stale owner. A track whose feet say sideline
(vetoed) is neither promoted as a goalkeeper candidate nor matched to any
missing goalkeeper identity. A confirmed player becomes a goalkeeper only after
four consecutive clean checks (about 4 s) ask for it, so a label never flips
back and forth. A player identity that turns out to be the goalkeeper keeps its
ID, gets the locked goalkeeper role and is shown as `GK-A` / `GK-B`, earlier
observations included. A returning goalkeeper is matched by their own
appearance even when the detector or the kit vote calls them a player. Every
change is logged (`ROLE UPDATE`, `GOALKEEPER CANDIDATE`, `GOALKEEPER
IDENTIFIED`, `GOALKEEPER TEAM`, `GOALKEEPER REVOKED`, `NON-TARGET PERSON`), with
the evidence behind it: appearance against the officials' look and both teams,
detector votes, touchline time and movement along the line, formation
consistency, ball-following behaviour, team count. The same evidence is kept
as `roleEvidence` on every identity and shown as a one-line `why` under each
person in the debug overlay. An established team only changes after 80% of the
last 15 votes point to the other kit (the identity is then released and
re-identified).

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
| `id` | `A-07`, `GK-1`: the persistent internal identity (players and goalkeepers only). Created once, never renumbered, never reused; everything keys on it |
| `display`, `longId` | what the overlay, the roster and the 2D view show: `A-07` / `TeamA_Player_07`, `GK-A` / `GK-B` (goalkeepers by team). Regenerated from the confirmed roster (below), so it can differ from `id` |
| `team`, `role`, `label` | `A` / `B` / none; `PLAYER`, `GOALKEEPER`; combined `PLAYER_TEAM_A`, `GOALKEEPER_TEAM_B` |
| `status` | `ACTIVE`, `MISSING`, `OFF_SCREEN` (left at an image edge), `RETIRED` (merged into the goalkeeper it really was), `WITHDRAWN` (a referee-like person promoted by mistake) |
| `roleLocked`, `roleHistory` | whether the goalkeeper role is locked, and when and why the role changed |
| `teamHistory`, `roleEvidence` | clean kit votes and how many clearly wore the team's kit (`established`); the latest role decision's evidence scores |
| gallery | up to 6 diverse, high-quality samples (OSNet embedding + kit colours) |
| last seen | time, image box, pitch position, camera-compensated velocity, exit edge |
| confidences | identity, team, role, non-target (referee-like) and goalkeeper confidence; recent history |

An ignored person in the per-frame output has `id: null`, `display: IGNORE`,
`role` and `label` `NON_TARGET`, `state: nontarget` and a `hint`
(`REFEREE_LIKE`, `SIDELINE_OFFICIAL`). `match.referees` stays in the result
(always empty) so pages and tools that read results of earlier versions keep
working.

Identities are never deleted. When a promoted track appears, it is first compared
with every missing identity of the same team and role. The score combines
appearance (OSNet similarity against the whole gallery, best two pairs), shorts
and socks colour, team (a hard gate), spatial plausibility (where the new track
appeared against where the identity is expected: its last clean position carried
on at its remembered speed, or held where it was, since a player hidden behind
someone has often stopped, whichever is closer; within the same camera segment or
on the calibrated pitch, with a reach that grows with the time since that clean
position; the speed is measured over the evidence tick from clean boxes only and
capped at a sprint), exit/entry edge, and time missing. A reconnection needs
several scored observations, a score of at least
0.72, and a 0.15 lead over the second-best identity **and** over "an unseen
teammate" (the chance this is someone not seen yet, estimated from how alike the
other known teammates look). Otherwise the decision is deferred and logged, and
the track shows as `IDENTITY_UNCERTAIN`. For example, 51% / 48% / 45% is
deferred; a later 93% / 41% / 38% reconnects. Once resolved, the observations
collected while uncertain are labelled retroactively (`evidence: reidentified`),
and so are those of a spell in which the kit read as neither team (`UNK-8`: a
small, distant player whose colours take seconds to settle): the local track
is the same person, so the frames of the last 12 s take the identity as soon
as it is decided.

**Display labels** (`soccer/labels.py`). The number a player is shown with is
not the internal identity. Per team, the confirmed outfield players, in the
order their identities were created, take `A-01`, `A-02`, ... without gaps, up
to the outfield slots a team fields at a time (10; `outfieldPerTeam` in the
settings). A player who left the match (substituted or sent off, a correction)
keeps their number and frees a slot, so a substitute takes the next number.
When a team holds more active identities than it fields, the slots go to the
identities with the most participation evidence over the whole segment
(seconds observed with the box clear of the frame edges and the person not
outside the pitch; ties in creation order), never to the first created: a
person seen for four seconds at the bottom edge among the staff does not keep
a slot from a player tracked for the whole minute on the far touchline whose
identity was decided late. The roster count is supporting evidence only:
nothing is merged, deleted or invented from it, and `PLAYER LABELS` says for
every unconfirmed identity how long it was seen, how much of that at an edge,
and how many teammates were in view with it. Identities beyond the active
slots are shown as `A-?` (unconfirmed) with all their tracking data kept for
re-identification and corrections (the roster slot says `confirmed: false`,
the timeline keeps their observed positions and estimates nothing for them),
until a slot frees up. Goalkeepers are `GK-A` / `GK-B` and are not counted. Labels are
regenerated at the end of a run and after every correction, never while
tracking: when you delete two false players shown as `B-07` and `B-08`, the
remaining players are renumbered `B-01`..`B-10`, the person who was `B-11`
is shown as `B-09`, and their `id`, appearance memory, trajectory and
timeline slot are unchanged (`PLAYER LABELS RENUMBERED` lists every move,
`summary.labelling` the counts). The review page shows `display · id` when
they differ.

A new global identity is created only when no missing identity can be this
person. A track that strongly resembles an identity lost less than 3 s ago
(appearance 0.65 or more) is never ruled out on the predicted position alone: it
waits as `IDENTITY_UNCERTAIN` (`waiting for clean frames` in its reason) for up
to 4 s of its own observation before the usual rules decide, so a local tracker
interruption at an occlusion does not mint a new identity. About 11 identities
per team is a sanity reference, not a hard rule: at
the limit a new identity needs every missing teammate to be implausible and at
least 5 s of observation, and a sanity warning is logged. More identities than
expected usually means players were recreated instead of re-identified.

**Crossings.** When two identified players overlap, their motion is recorded and
the identities are compared again after they separate (KEEP vs SWAP). Only
appearance can swap them back; trajectories alone never do. If the appearance is
too similar to tell, both identities are kept, their confidence is capped, and a
crossing warning is logged. A track whose crops repeatedly stop matching its
identity (other kit, or very different OSNet appearance) releases it.

**Close encounters and contaminated crops.** Local track id, global identity,
team and role are four different things, and a local tracker interruption must
not erase the real person. Every sample carries the IoU of its box with the
nearest other person's box (`occlusion`, `occludedWith`) and the share of its box
inside another person's box; above 0.15 IoU, or more than half inside another
box (a small player behind a near one keeps a small IoU), the crop is
contaminated and updates nothing: no kit vote, no detector vote (the position
cues still count), no appearance gallery, and the track's "first clean crops"
are left alone. A partial box, contaminated or next to someone and shorter than
80% of the track's recent clean boxes (the detector saw a sliver of a player
going behind another, its foot point too high), leaves the kinematic memory
alone as well: the identity's last position and speed, in the image and on the
pitch, stay those of the last clean box (`partial: true` on the frame), so the
re-identification after the occlusion predicts from where the player really
was. A crop joins an identity's gallery only when it is clean, of
decent quality, on the pitch, the binding has settled (0.6 s old, not after a
crossing) and it still looks like the gallery (OSNet cosine ≥ 0.45), so a local
track that silently moved to the person next door cannot write them into the
identity (`memory: updated / frozen: …` on each tick). A player (or goalkeeper)
and an ignored person overlapping at IoU ≥ 0.25 is logged as an `OCCLUSION
EVENT` with the freeze; the ignored person has no identity that could take the
player's.

**Identity first, role second.** A new local track that may be a missing player
is compared with that player before a referee-like reading may make it
non-target: while the best missing candidate still scores near the reconnection
threshold, or too few comparisons exist, or the recent evidence is mostly
contaminated, or a referee reading rests on one gallery resemblance that
neither the batch appearance, the detector nor the kit shares, the role stays
and `ROLE UPDATE DEFERRED` is logged. A track born as a second box on an
identity's own track (IoU ≥ 0.2 in its first 0.6 s, a split detection) is the
same person, not someone "seen together" with them, so the spatial evidence
counts. A missing player leaves room in a "full" team (only players on the
pitch now count), so the referee threshold is not lowered against a returning
teammate. No new player identity is created from referee-like evidence: once
no missing player fits, such a person is ignored instead.

**Role hysteresis.** `UNKNOWN → PLAYER` or `UNKNOWN → NON_TARGET` needs the
usual window; `NON_TARGET → candidate` needs two seconds of clear player or
goalkeeper evidence; `confirmed PLAYER → GOALKEEPER` needs four consecutive
clean checks (about 4 s) asking for it, at least 60% clean evidence in the
window, no overlap with another person in the last 3 s and no crossing in the
last 5 s; `confirmed PLAYER → NON_TARGET` does not exist (see the withdrawal
exception above, which needs the same four clean checks). Refused changes are
logged as `ROLE UPDATE IGNORED` with the raw reading and the reason; each
person carries `stableRole` (the identity's) next to `rawRole` (the latest
window reading, `NON_TARGET` for a referee-like one).

**Track switches.** The local tracker can hand a track from one person to
another when they cross (on the Veo clip, a B player's track continued on the
referee whose own track had just ended). A clean crop that looks like neither
the track's own first clean crops nor its identity's gallery (OSNet cosine
< 0.55; on these small crops another kit scores 0.36–0.5 and the same person
0.75–0.95), three ticks in a row that look like each other (≥ 0.7, so one odd
crop never counts), is such a switch: `TRACK SWITCH` is logged, the track is
re-born at the first of those crops (evidence, kit votes, first crops and
birth time restart, so the "seen together" rule no longer counts its past and
the person it now shows is decided from scratch: compared with the missing
players first, ignored if referee-like), the identity it carried goes missing
with its last sighting moved
back to the last crop that still matched, and the frames since then lose that
label (`?`) rather than showing the player on the referee's body.

**Offline tracklet refinement** (`soccer/tracklets.py`, after the causal pass).
Every finished local track is kept. Each track is cut where its identity label
changed (a mixed tracklet), where a `TRACK SWITCH` was found, and where its
clean crops stop looking like the ones before (the pairs across the cut
average below OSNet cosine 0.55 while each side agrees with itself at ≥ 0.6;
`TRACKLET SPLIT`; four or more tracks changing together within 0.75 s is a
scene change, a zoom or the light, and cuts nothing). A
fragment after such a cut is unidentified whatever label it still wears (a
team label like `B-?` becomes `?`), and every fragment without a confident
player identity — `?`, `B-?`, or `IGNORE` — is compared with the dormant player
identities around it in time: its own clean crops against the gallery
(appearance ≥ 0.5 required; ≥ 0.65, and never on position alone, for a fragment
that was ignored as referee-like), camera-compensated image distance from the
identity's last sighting before the fragment and first sighting after it (both
directions), the time gap (≤ 10 s; a fragment without usable crops only within
3 s, position alone says little after that) and team compatibility; an identity
seen elsewhere during the fragment is never a candidate. A fragment is connected at
a combined score ≥ 0.6 with a 0.1 lead, every frame takes the identity's id,
team and role (`evidence: connected offline`, `firstPassLabel` kept), and
`TRACKLET CONNECT` events give the scores. The local track is itself evidence:
an anonymous stretch of a track (`UNK-8`, or `B-?` before the identity was
decided) within 1 s of an identified stretch of the same track, with no track
switch or appearance cut between them, is that identity whatever OSNet makes
of thirty-pixel crops, unless the crops clearly contradict it (cosine under
0.55), the fragment voted the other team, or the identity was seen elsewhere
meanwhile. This runs before the appearance matching (the track itself outranks another
track's appearance match to the same identity, so a look-alike seen meanwhile
is never connected) and again after the global assignment (a fragment
identified there hands the identity to the anonymous stretches of its own
track). This is how a player first identified late, from the clearer
frames, gets his earlier frames back (future frames correct earlier labels),
and `TRACKLET UNRESOLVED` says why a fragment of a second or more stayed
anonymous (best candidate, score, floor, lead, or why its own track did not
carry the identity to it). There are no referee identities to
connect anything to. The 2D match view reads these finalized labels, so a
player who was B-02 → B-? → ignored in the causal pass stays B-02 throughout.
`summary.identity` reports id switches, mixed tracklets, track switches found
online, offline splits, confirmed-player tracks that continued as an ignored
person (and how many no track switch or release explains: the number that
should stay at zero), referee identities (always 0), ignored tracks and frames
by hint, withdrawn identities, ignored / deferred role updates, close
encounters with ignored people, how many players came through them and were
re-identified after them, new player identities, teams over their count,
fragments connected (ignored fragments included), frames restored to
players, and identities merged offline.

**The active roster and what the camera sees** (`soccer/roster.py`,
`soccer/visibility.py`; pure functions of the finished result, re-derived
whenever a correction is applied). `roster` lists one slot per global identity
of the match with its status at the end of the segment — ACTIVE (observed in
the last second), OFF_SCREEN (last seen leaving through a frame edge), UNKNOWN
(missing without an exit seen), LEFT (marked by you as substituted or sent off,
a correction choice) or RETIRED (merged) — its first and last observation,
observed frames, local tracks, last pitch position; the expected team sizes
(`maxPerTeam`), the actual counts and `overCap` (a team with more identities
than expected is reported, never trimmed), the anonymous observations:
tracked people with a team but no identity, and `mergeProposals`: pairs of
identities that may be one person (same team, never seen on the same frame, one
observed entirely before the other, galleries mutually the most similar at 0.7
or above, and a plausible move between the last sighting of one and the first
of the other). A proposal that describes a short occlusion is applied: the
later identity began within 3 s of the earlier one's last sighting, that
sighting was inside the frame (not an exit through an edge), the move is
plausible and measured, neither identity was set by you and the later one was
created in this run. The later identity is retired into the earlier one
(`match.retired`, `retiredInto`), every frame labelled with it takes the earlier
identity (`evidence: merged offline`, `firstPassLabel` kept), the galleries are
joined and `MERGE APPLIED` is logged (`summary.identity.mergedOffline`). Every
other proposal is listed for your review and logged as MERGE PROPOSED
(`applied: false`); longer gaps are not merged automatically until proposals
have been scored against corrections on enough videos. On every frame with a
registration,
`view` is the part of the pitch the camera sees (the frame rectangle cut at the
horizon, mapped to the ground, clipped to the pitch plus 5 m, as a normalized
polygon), and `visibility` classifies every roster slot that is not observed on
the frame from its last observed position and an uncertainty that grows with
the time since (2 m + 1.5 m/s, nothing older than 10 s): `C` expected in frame
but not seen (hidden behind someone or missed), `S` off-screen (outside the
view by more than the uncertainty, or last seen leaving through a frame edge
and not observed since), `U` unknown (no estimate, no view, or too close to the
edge of the view to say). Observed slots
are implied by their presence in `people`. This is evidence for the review and
for the later off-screen estimation; nothing is removed or invented because of
it (`summary.visibility` gives the shares).

**Where the unseen players probably are** (`soccer/imputation.py`, the
`timeline` block). For every roster slot and frame the result carries an
estimate with its kind: `O` observed, `I` interpolated across a gap of at most a
second between two observations, `P` predicted, `U` unknown (nothing
supportable: never observed, hidden for more than 25 s, or less certain than
15 m — the slot stays, no coordinates are invented). Between two observations
further apart than a second the estimate is the straight line between them
(kind `P`, with the sigma of both sides): on full-pitch data no formation model
beat it. Where only one side is observed (the player left the view and has not
come back, or before the first sighting) the estimate is the last observed
position **moved with the team**: the visible teammates' displacement since that
observation (at least four of them, each weighted by its uncertainty), ramped
in over 2 s so that the first seconds hold the last position, with a sigma
growing at 0.8 m/s. An observation always wins and a returning player never
jumps, because the line meets both observations exactly. Role-anchored
centroid voting (a dynamic formation learned from the segment, the team centre
voted by the visible players so that the camera's one-sided view does not bias
it) is implemented and benchmarked, but not used for the shipped estimate: on
the Metrica Sports sample games with simulated camera windows
(`tools/metrica_bench.py`, held-out game, 45 m view) the straight line errs by
3.2 m on average, the shipped method by 4.0 m over everything it estimates
(1.6 m within a second of the last sighting, 2.3 m at 1-3 s, 4.2 m at 3-8 s,
5.3-5.7 m beyond), holding the last position one-sided by 5.2-5.4 m, and the
voting by 7.2-7.3 m; the shipped sigma is calibrated (p68 of |error|/sigma
0.8-1.1 across views). Every estimate has a sigma in metres;
`summary.timeline.kinds` gives the shares. Details and the other views are in
`PROGRESS.md`.

**The 2D view and unidentified people.** A tracked person with a team and a
registered position but no identity (`B-?`) is a real observation: the 2D view
draws it as a hollow, dashed marker of the team's colour with a `?`, joined only
across gaps of 0.3 s and never interpolated further, and the roster strip under
the board says how many such people are shown and which roster slots are not
on the board, with their visibility. Identities are never invented for them.

### 4. Pitch coordinates and the pitch model

No field keypoint model ships with this prototype, so the pitch model is
anchored by your landmarks; a run without any keyframe near it falls back on
the automatic pinhole midfield anchor ("Amateur footage" below). Pause on a
clear frame, use **Mark landmarks**, pick named landmarks (corners, halfway line
ends, centre spot and circle, penalty and goal box corners, penalty spots) and
click them on the video; four or more that are not on one line are needed. Two
things are then kept apart (`soccer/pitch_model.py`; geometry in
`soccer/field.py`):

- The **match pitch model** is the physical field: a rectangle with standard
  markings whose length and width are learned once from all calibration
  keyframes together (`field.fit_keyframes`: only when the landmarks pin both
  dimensions down and explain them clearly better than the 105 × 68 m assumed
  otherwise) and never rebuilt per frame.
- The **camera homography** says where that rectangle lies in the current
  frame. A keyframe anchors it. Between keyframes it is carried by the camera
  motion (a full frame-to-frame homography when enough background points
  support one, the similarity otherwise, a whole-frame translation when the
  grass has no trackable texture); painted lines and the centre-circle paint
  that match the markings it predicts refine it (never by more than 2% of the
  frame width at once, and only once an anchor has aged) and restore its
  temporal confidence; a camera cut drops it (the physical pitch is kept); when
  most tracked people stand outside the predicted pitch three checks in a row
  it is doubted.

**Two confidences, kept apart.** The model's confidence is
`temporal × (0.25 + 0.75 × observability)`:

- *Temporal* confidence says how recently and how well the chain was anchored
  or confirmed: 1.0 at an anchor, falling 0.03 per second carried on measured
  flow, 0.12 per second on a coarse (whole-frame) pan estimate and 0.5 per
  second of unreliable motion, restored by painted markings that confirm it.
- *Observability* says how well the evidence constrains the geometry at all
  (`field.observability`): how many landmarks, how much of the frame they
  span and whether they fix both pitch axes; how many marking lines of each
  family (along the pitch: touchlines, box sides; across it: goal lines, the
  halfway line, box fronts — one family alone fixes neither the length
  direction nor where the pitch ends); how much of the centre circle is seen;
  the conditioning of the fit's Jacobian (a flat direction is a parameter the
  evidence cannot see); and the residual. Two touchlines alone score 0.1. A
  calibration below 0.15 is refused as an anchor: it would only be carried
  smoothly into a confident-looking wrong answer. Observability never rises by
  carrying; only new evidence raises it.

**Where the pitch ends.** Two parallel touchlines converge in the image towards
a vanishing point; that point is never a corner. The automatic detector marks
the converging side `unknown` when no goal line or box front was seen there
(`vanishing` on the pitch record): no edge cue is measured against it, nobody
is rejected beyond it, and a region bounded by it cannot score HIGH. The
tracked model reports a goal line that projects onto that apex as `AT THE
HORIZON` rather than predicted. Pitch ends come from transverse evidence, the
canonical dimensions and anchors only.

**Automatic midfield anchor** (`soccer/midfield.py`). A low sideline camera
often shows no corner and no goal line, but the centre circle (an ellipse in
the image), the halfway line and a touchline are enough: the halfway line is a
diameter of the circle, so its pole with respect to the ellipse is the
vanishing point of the touchline direction; the circle points on the halfway
line, the touchline crossings and the 1-D projectivity along it give the
centre spot; the diameter through it towards that vanishing point meets the
ellipse at the left and right circle points. Those derived points only start
a fit on the paint itself (halfway line, touchlines, the circle's ridge
pixels); the result is validated both ways (every visible marking projected
back must lie on paint), scored for observability including the fit's
conditioning, and refused below 0.2. It is attempted only where it is needed
(no model to carry and no confident painted boundary), and a weak automatic
anchor never replaces the detector's own confident region: broadcast footage
keeps its `automatic` mode. Centre-circle paint is found with RANSAC
over the connected arcs of the ridge mask around the likeliest halfway-line
candidates; a thin ellipse that only explains a straight line is rejected.
The circle also feeds the refinement of an existing model.

Runs without user keyframes use a different search and solver (a pinhole camera
fitted to the circle, the halfway line, people's heights and the other paint;
see "Pitch without keyframes" under "Amateur footage"). Runs with keyframes keep
the construction above unchanged.

**Direct re-registration to the keyframe's view** (`soccer/camera.py`
`AnchorViews`). A chain of frame-to-frame motions drifts: measured on the
test clips, a keyframe carried through 22 s of a low amateur camera's pan was
about 100 px (20 m on the far half of the pitch) off at the next keyframe, and
unusable after 5 s on a wide Veo lens. So the frame a keyframe (or an
automatic anchor) was taken on is remembered as SIFT features (people and the
top band masked, at 960 px width), and once a second the current frame is
matched to the remembered views directly (RANSAC homography, at least 40
inlier matches within 3 px, at least 30% of the frame covered). A good match
re-ties the model to that keyframe's geometry (`REGISTERED TO THE KEYFRAME
VIEW`), restores its temporal confidence and becomes a `reregistered` anchor of
the offline registration, rooted at the keyframe; a match that would move the
visible ground by more than 10% of the frame width while the chain is still
confident is refused and logged. Nothing is learned from the grass alone: with
too few features the chain simply continues. When a keyframe's view is leaving
the frame (fewer than 120 matches, or less than 60% of the frame covered), the
current re-registered frame is kept as a *relay* view rooted at the same
keyframe, and the following frames register to it: a pan is bridged by a few
relay hops of sub-pixel residual instead of hundreds of frame-to-frame steps.
Keyframe views are always tried first and used whenever they still match well
(80 matches or more); relays only when no keyframe view does. On the amateur
test clip this brought the error of carrying the 0 s keyframe to the one at
22.3 s from 101 px (18 m) to 49 px (5 m); the carry in the other direction is
unchanged, because relays exist only forward in time (`registrations` in
`summary.pitch.registration` counts tried, adopted, refused, root changes,
relays made and registrations via a relay).

**What a painted-line refinement is worth.** A refinement moves the live
model and restores its temporal confidence, but it anchors the offline
registration only when the paint it matched constrains the pitch on its own
(observability at least 0.15: both marking families, or the centre circle with
a transverse line). A single touchline match no longer restarts a chain at the
keyframe's certainty; such anchors used to carry a drifted view confidently.
A refinement that does anchor is worth the chain's observability, which its
paint just confirmed (or its own when that is higher): measured on the
broadcast clip, fading the chain's certainty with time instead halved the
share of confident frames although the hold-out errors were about a metre.

**Keyframes just outside the segment** are not lost: a keyframe up to 15 s
before the start is carried into the segment by processing the frames between
for camera motion and pitch evidence only (pre-roll, no detections, nothing
emitted), and one up to 15 s after the end is reached the same way so that the
backward sweep brings it in. Their pitch events say so.

**Keyframes with a wrong click.** Among six or more landmarks, one that lies
more than 3 m off the plane the others define, and three times further than
their median, is left out (a warning names it). Landmarks disagreeing by up to
4 m still anchor; the residual lowers the observability instead of a hard 1.5 m
limit dropping the keyframe silently. (A Jacobian condition number was tried for
keyframes and found uninformative: column-normalised, every real keyframe set
scored 55–175, and degenerate sets are refused by the general-position check
already; keyframes rest on their spread, axes and residual.)

**Offline passes, in order:** the pitch registration (below) comes first, so
the tracklet passes compare positions in metres where they exist; then the
tracklet refinement (`soccer/tracklets.py`: cuts and reconnections, section
3), then the global assignment of the fragments still anonymous: every run of
a local track without an identity (at least 0.4 s) is scored against the team
players not observed during it — appearance against the gallery, the distance
in metres on the pitch (or in body heights in the stabilised image where no
position is registered) from the identity's nearest sighting before and after,
the time gap — and fragments and identities are matched one-to-one per round
(Hungarian); a pair is taken only at the same thresholds as a reconnection
(0.6 with an appearance floor of 0.5, position alone only within 3 s) and with
a lead of 0.1 over the fragment's next identity and over the identity's next
fragment, so two teammates returning together take two identities or none,
never the same one twice (`TRACKLET ASSIGNED` events, `evidence: assigned
offline`, counts in `summary.identity.assignment`). Then the ball path.

**Offline registration passes** (`soccer/registration.py`, after tracking).
Pass 1 records, per frame, the camera motion and how it was measured, and
every validated anchor of the run (landmark keyframes, direct re-registrations,
painted markings that confirmed the model with evidence of their own,
automatic midfield anchors) with its observability and the keyframe its chain
started from (`root`). Pass 2 carries each anchor forward *and* backward
through the motion chain; between anchors the two directions are blended by
confidence and their disagreement lowers it. The disagreement is the median
distance, in metres, between the two mappings of a grid of image points over
the visible pitch, far rows included (it used to be measured at four near
points, where a pixel is worth centimetres, and so overstated the far-field
accuracy); points behind either horizon are left out. A cut or a stretch
without any motion estimate ends a chain. Pass 3 recomputes every person's and
the ball's pitch position from the final geometry: each position carries its
confidence (the calibration's times how much a few pixels of foot error are
worth in metres there) and `pitchSigma`, the error in metres along and across
the pitch (3 px of foot error through the local Jacobian of the mapping, 35%
of the box height instead when the box is cut by the bottom of the frame and
the feet are below it, `pitchClipped`, plus the frame's disagreement in
quadrature). No position is written on a frame whose registration confidence is
below 0.2 (`calibrated: false`), a foot above the horizon is `pitchStatus:
horizon`, a position that does not land on the pitch is `unresolved`, and
nothing is deleted — a registration failure can never end an image track.
Frames where many live image tracks have no pitch position are reported as
`MAPPING LOSS` calibration events (the image tracks are intact, the
registration is what failed), never as players leaving the match. A section
that asked for confirmation online is withdrawn when the passes cover it.

**Lens distortion as a measurement** (`soccer/lens.py`). On every evidence
tick the paint along the longest detected markings is tested for the
one-parameter division model that straightens it; `summary.pitch.registration.lens`
gives the video's median coefficient `k` (half-diagonal units), how many frames
measured it, the median share of the bend it removed and the sign agreement.
It is a diagnostic: the undistortion is not applied to the geometry yet.

While the model is trusted (MEDIUM or better) its projected rectangle is the
playable region — off-screen and faded edges included, predicted from the model
rather than read from the grass — each person gets `pitch: [x, y]` (x 0..1 left
goal line to right goal line, y 0..1 far touchline to near touchline) with
`pitchConfidence` and `pitchSource`, and spatial re-ID evidence uses metres on
the pitch. Screen coordinates (`box`) are always stored too. Without a usable
pitch on a frame, image evidence alone promotes people (after the touchline
requirement of 10 observations rather than 5) and nobody is rejected for a
zone that was never measured.

**Pitch confirmation.** When no painted boundary line is found and no model
covers a stretch of the video for 3 s or more, the stretch is reported once
(stretches closer than 3 s merged, and up to 10 s apart when one reliably
measured camera movement joins them: one pan asks once) in the *Pitch
confirmation* panel and in `pitchIssues`: its time range, the most useful frame
in it (the halfway line in view, as much grass as possible), the landmarks
worth marking there, with positions where the image already gives them (the
halfway line's ends at the grass edges), and a `hint` naming the landmark
family the nearby evidence lacked (a goal line, box front or the halfway line
when only touchlines were seen; the near touchline when the earlier keyframe
lies on one half only). *Confirm pitch* pauses on that frame with those points placed
(hollow: drag them onto the real lines), lets you add the rest, and saves an
ordinary calibration keyframe; the next run anchors the model there and carries
it through the section in both directions. The four corners work as a last
resort when nothing else is visible.

### 5. Camera motion

Pan, zoom and small rotation are estimated once per frame from background features
(people and the top overlay band are masked; each flow vector must survive a
backward check) and shared with BoT-SORT's Kalman prediction. When flat grass
offers too few features, a whole-frame phase correlation at 320 px width still
measures the pan (`cameraMotion: coarse`; the pitch model decays faster on it),
so a fast pan over an amateur field is not mistaken for a cut and does not end
every track. Positions used for movement, spatial re-ID and crossings are
camera-compensated within a camera segment, so a pan is not read as player
movement. Unreliable motion is never extrapolated: stabilized positions are
marked as drifting (weakening spatial evidence), boundary keyframes are dropped
until the next one and the pitch model's confidence falls quickly until it is
let go. A camera cut ends all local tracks (identities become missing) and
starts a new segment; re-identification then relies on appearance. The
sagitta of the longest touchline-family line (`curvature`, px per 1000 px) is
recorded as a lens-distortion diagnostic.

### 6. The ball

The ball tracker never asks "which white blob looks most like a ball in this
frame?" but "which detection is the ball I am already following?". It is a
state machine with one active ball:

```
SEARCHING ─(several consistent, moving detections)─▶ LOCKED ─(no detection)─▶ OCCLUDED
   ▲                                                   ▲                         │ short gap: predicted,
   │                                                   └───(reconnect)───────────┤ kept at the player
   │                                                                             ▼ that hides it
   └──────────── LOST ◀─(2.5 s free / 3 s at a player)─ RECOVERING (search area grows, confirmation needed)
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
unless another one scores at least 0.25 higher. **A detection never takes the
ball over on its own.** Beyond the gate (the gate itself stops growing with the
statistical uncertainty: it widens by 6 m/s of hidden time, no more) only a
*chain* of consistent sightings is considered, and it must pass the **entry
rule**: the speed the hidden ball would have needed to get from where it is
believed to be to where the chain started, in the unseen time, must be either
slow (under 8 m/s) or, up to a hard shot's 40 m/s, matched by the chain still
moving that fast and not sitting at a foot. Anything on **another player's
lower body** (not the player the ball was hidden at) is a boot or a sock until
proven otherwise: it counts only when the mean prediction itself reaches it or
it has moved relative to that player (0.6 m over 0.6 s counts as free). Every
such decision is logged as `BALL SWITCH WARNING` (old and new position,
elapsed time, required velocity, physical plausibility, previous trajectory
confidence, whether the candidate lies on a penalty or centre spot of a
trusted registration, the static field-marking evidence with its stillness
scores, detector confidence, player attachment and relative motion,
`ACCEPTED` / `REJECTED (JUMP | NO_ENTRY | UNCONFIRMED)`, with
`/ FIELD_MARKING` when the thing is a known marking). A thing fixed to the
field stays "settled" while a carried calibration drifts: its stillness is
judged in the camera-compensated image as well as on the pitch, and either
saying still is enough. The lock's "seen elsewhere while the ball was tracked" flags
lapse once the ball has been hidden for 0.4 s (a wrong lock must not keep
excluding the real ball). A ball hidden by a player is carried with that
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
| `OTHER_OBJECT` | seen at the same time as the confidently tracked ball, elsewhere (there is one ball). Only a mature lock may say so (at least 0.5 s old, confident, and it has moved at least a metre like a ball), never about things within 1.5 m of the ball, and the flag lapses once that ball has been hidden for 0.4 s or is lost: a wrong lock on a boot cannot disqualify the real ball next to it |
| `JUMP`, `NO_ENTRY`, `UNCONFIRMED` | beyond the gate while the ball is hidden: the hidden ball could not have got there in the unseen time; it sits on another player's lower body with no trajectory arriving and no motion of its own relative to them; it is not yet a chain of consistent sightings |
| `SIZE`, `OUTSIDE_PITCH` | impossible for a ball at that depth (0.22 m scaled by the perspective player-size model), or far outside the playable field |
| `TRAJECTORY`, `LOW_SCORE` | plausible but off the tracked ball's path, or in the gate with too weak a score |

Independent motion is measured against the field (camera-compensated or pitch
coordinates) and against the player a candidate sits on (`relative`: a boot
moves with its player, a dribbled ball does not); shape and colour are only
small terms, because the real ball is often blurred, elongated or a few pixels
wide. The output keeps detector confidence apart from identity: `identity` (the
track), `visual` (this observation) and `trajectory` (its consistency) are
reported separately, and the overlay reads `BALL LOCKED · identity 0.92 ·
visual 0.74 · trajectory 0.96`, `BALL OCCLUDED · likely interaction B-02 ·
hidden 0.42 s`, or `BALL RECOVERING · candidate paths: 4`. Every frame also
says where its position comes from (`origin`: `OBSERVATION` or `PREDICTION`)
and the motion state the evidence supports (`motion`: `STATIONARY` when the
sightings of the last 0.4 s show the ball at rest, `ROLLING` below 8 m/s,
`FAST` above, `OCCLUDED` while hidden, `UNKNOWN` without a ball). A ball
observed at rest carries no velocity into the next prediction; an unseen ball
is `OCCLUDED`, never assumed still.

**The final trajectory is decoded offline** (`soccer/ball_path.py`), because an
uploaded video is known end to end. Every frame is a layer of nodes — its
recorded candidates, scored for ball-likeness from the evidence above, plus an
explicit NOT VISIBLE node — and dynamic programming (Viterbi) finds the
cheapest path through the segment. A path pays per frame for the node it uses
(`-log` of the ball-likeness; NOT VISIBLE costs a fixed amount), for each link
between sightings (camera-compensated displacement as a speed, dearer the
longer the gap it had to cover unseen; cheap when the causal pass already
linked the two; +1.5 for arriving at a player's lower body unseen and +5,
fading over 2 s, when it comes from another player's feet or reach), for each
frame bridged inside a link (occluded on a known trajectory: 0.8 each, so a
mediocre true sighting is kept rather than skipped), and for starting (3.2;
+4 when the trajectory starts on a player's lower body: nothing arrived there)
or ending a trajectory. Links may skip up to 1.2 s, so a ball seen at frame 100
and 130 need not be "found" in between; those frames come out `MISSING` with
an interpolated position. A link whose implied speed says the ball stayed where
it was (under 1 m/s) may skip up to 3 s: a ball seen again where it vanished
most likely stayed there, hidden. A flight (8 m/s or more) into or out of a
sighting that stands still with no player within reach (+12: a ball does not
stop dead on the grass by itself, nor start from rest without a touch), and a
link faster than a hard shot (40 m/s) between sightings of two different
things (+12), cost more than ending and starting a trajectory. People rejected
as outside the pitch (ball boys, staff) are not players who can take, stop or
hide the ball. A candidate glued to the same spot on one player's lower body
loses ball-likeness (0.55; 0.3 after a second), and one the causal pass saw
while it confidently tracked the ball elsewhere keeps a fifth (there is one
ball: such a thing never pays for a frame by itself). The
effect: a boot that the first pass took for the ball while the real ball was
hidden is a short run that connects to nothing and never pays for its start;
the decoded path leaves those frames NOT VISIBLE, and a later sighting of the
real ball links back to the earlier ones. The result's per-frame `ball` is this
trajectory (`phase: PATH`, one identity `BALL-1`, `bridged` on interpolated
frames); the causal pass stays as `ball.firstPass`; `BALL PATH` events explain
every first-pass run the final path leaves out (the speed reaching it would
have needed). The 2D match view and the ball metrics use the final trajectory.

**Field markings.** The decoder knows the painted spots: a candidate within
1 m of a penalty spot or the centre spot of the offline registration (when
that registration is trusted, confidence 0.5 or more) keeps 40% of its
ball-likeness, unless the causal pass tracked the ball there (a penalty about
to be taken is still the ball). A tracklet whose sightings span 2 s or more
and all lie within 0.6 m of the first (on the registered pitch, else
camera-compensated in the image) is a thing standing still for seconds, a
marking, debris or a bag, and its sightings are discounted the same way,
again except those the causal pass took. Such sightings never pay for a frame
by themselves, so the path only goes to them when it rolled there; `summary.ball.markingNodes`
counts them. This is explicit penalty-spot handling with temporal evidence,
not a mask: a ball placed on the spot and tracked there stays on the path.

**Trajectory review.** The decoded path is then reviewed and decoded again
(up to eight rounds). A still stretch of the path (every sighting within
0.5 m of its first, camera-compensated, moving slower than 1 m/s over its
duration, at least four sightings, or any number when every one of them is a
known marking) whose sightings the causal pass ruled out (40% or more with
`JUMP`, `NO_ENTRY`, `OTHER_OBJECT`, `STATIC`, `FIELD_LINE` or
`PLAYER_ATTACHED`, or markings: the hidden ball could not have got there, it
was seen while the ball was confidently elsewhere, or it is a known spot,
paint or a boot) and that nothing was seen rolling to, judged from the last
sighting the causal pass accepted (a flight, 0.5 s or more unseen, or nothing
at all; rejected waypoints in between are not sightings), is a white thing
standing where the ball was known not to be, a ball boy's ball on the
touchline or the penalty spot for instance. Its sightings are cut out
(a marking's other sightings with them, so it cannot serve as the next
stepping stone across the same gap), `BALL TRAJECTORY REVIEW` is logged with
the frames, the last accepted sighting, the visual evidence, the motion over
the stretch, the causal pass's reasons and the decision
(`REJECTED_STATIC_UNANCHORED`), and the path is decoded without them, so a
ball that vanished at a player and was seen again there comes out as one
stationary trajectory, hidden in between, rather than an excursion to the
touchline and back. A rolling ball is never a still stretch, whatever the
causal pass said about it: the speed test keeps the review from chopping the
real ball into pieces when the causal pass was locked on a boot. An **island**,
a run shorter than 1 s that no ball could have reached from the run before it
nor left for the run after it (both faster than a hard shot, 40 m/s), is cut
too (`REJECTED_IMPOSSIBLE_JUMP`): whatever it was, the ball was somewhere else
just before and just after, and its frames stay unknown rather than a
teleport out and back. `UNCONFIRMED`, `LOW_SCORE` and
`TRAJECTORY` are not negatives: the real ball gets them while the causal pass
re-acquires it. A ball at rest the causal pass tracked, found nothing against,
or saw rolling there, is kept. Every frame of the final path says where its
position comes from (`origin`: `OBSERVATION` for a sighting with detector
confidence 0.3 and ball-likeness 0.5 or more, `WEAK_DETECTION` below,
`INTERPOLATION` for a bridged frame, `UNKNOWN`) and how the ball moved
(`motion`, from the link speeds: `STATIONARY`, `ROLLING`, `FAST`, `OCCLUDED`);
`summary.ball` counts the stretches the review cut (`staticRunsDropped`), the
links that are flights (`longUnseenLinks`) and the stationary links
(`stationaryLinks`). The decoder's per-frame costs are not confidences: a
weak sighting on the path is shown as such, and the ball-likeness it is
accepted with is never a measure of how sure the sighting is.
`ball_cleanup.py <result.json>` re-decodes an existing result without the
detector (camera motion from the run's `registration.json`, else `frames[].motion`,
an `.npz`, or re-estimated from the video; `--small-scene on|off` forces the
small-scene gate for the decode only, for results written before the frames
carried `profile.smallScene`). A ball candidate without a `source` key comes
from the whole-frame detector (`tiles`, `crop` and `proposal` are written).

## Amateur footage (small scenes)

Pole and tripod cameras on amateur pitches show players of 6-40 px, a ball the
detector often does not see at a dribbler's feet, a centre circle as a thin far
ellipse and frequently no user keyframe at all. Everything this section
describes sits behind one of three gates, and broadcast footage opens none of
them (`tools/exact_diff.py` compares a broadcast run with and without the code
bit for bit; see PROGRESS.md for the measurements):

- **`small_scene`** (`SoccerTracker.small_scene`, written per frame as
  `profile.smallScene`): the far band (where people are expected to be under
  30 px) active on at least half of the last 2 s, or a source under 720 px
  tall. It latches once open: a gate flipping with the view would switch the
  tracker's box convention, the ball rules and the identity rules back and
  forth. Broadcast 0-60 s: far band on 1 of 1800 frames, never open. A run
  counts as a **small-scene run** (`labels.small_scene_run`) when at least half
  of its frames are small-scene frames; the offline passes use that.
- **`auto_landmarks`**: no user keyframe within the segment ±15 s. Every
  regression window with keyframes keeps it closed.
- **`pitch_provisional`**: the pitch comes from an automatic anchor only.

### Pitch without keyframes

With `auto_landmarks` the centre circle is searched on the grass only
(`pitch.find_circle(on_grass=True)`: paint between grass, inside the grass
region, people's boxes masked, straight arcs kept, a minimum inlier density
and the ellipse's centre near the halfway line) and a **pinhole camera** is
solved from it (`midfield.solve(allow_pinhole=True)` → `pinhole_solve`): the
halfway line and the circle with one unknown focal length (principal point at
the image centre). When the camera stands in or near the halfway line's
vertical plane, as both pole and broadcast cameras do, that alone cannot fix
the camera (two mirror solutions, an unobservable focal length), so the fit
also uses people's box heights (they fix the horizon; a robust loss), the
other painted markings the solved camera predicts and the touchlines found
through it (`_ground_touchlines`, within 4 m of the model's line). Without
such evidence it refuses. An anchor whose only support is the circle, the
halfway line and people is **capped** (MEDIUM at most); only independent
painted lines it was not fitted to lift the cap (`pitch_model._lifts_cap`).
Field dimensions come from one place (`field.PitchSize`, 105 × 68 m assumed;
`registration.json` records `fieldSize` with the source of each dimension and
the width the anchors measured). A frame under 720 px whose turf the fixed
grass rule rejects (a screen recording's colour cast) learns the turf colour
from the lower frame (`pitch.ground_colour`).

Lens distortion is not what bends a panoramic follow camera's view: its output
is rectilinear (long paint is straight; `lens.estimate_k` finds nothing), and
the curved black corners are the edge of the panorama. On small scenes each
evidence tick also stores the outline of the part of the frame that carries
image (`pitch.imageArea`, `visibility.image_area`), so the view polygon does
not count the black as seen.

**Registered pitch (final)** (Advanced → Debug overlays) projects the frame's
final homography, the geometry the 2D positions use: cyan, a side solid where
the paint of a tick at most 0.2 s old lies on it under that tick's own final H,
dashed where predicted; other markings dashed; detected paint thin white; all
muted below 0.5 confidence.

### Detection and local tracks

On small-scene frames only:

- **Near pass**: an extra 640 px people pass that only *fills in* people of at
  least 60 px (scaled with a low-resolution source) that no row of the 960 and
  detail passes covers 30% of (`pipeline._fill_in`). The 960 and 1536 passes
  miss some near players of 60-120 px (amateur 3, 148.0 s: the ball carrier is
  found at 640 only). Merged into the joint suppression instead, it replaced
  the main passes' box on about three people per frame and doubled about two
  more, which cut their local tracks (awsrc: 357 → 524 local tracks).
- **Tiny far-tile rows** (under 30 px) are confirmed by the foot point (within
  half a body height, height within 1.67×) instead of IoU > .5, which never
  survives 1-2 px of jitter on a 6 × 13 px box; a tile row duplicating a
  whole-frame row that can start a track is dropped; one confirmed tile row
  replaces weak whole-frame rows of the same person.
- **Buffered IoU**: BoT-SORT sees boxes under 24 × 12 px inflated to that size
  about their centre (the identity layer keeps the true box); the live tracks
  are re-boxed when the gate opens.
- `dropped` lists the detections the local tracker did not use (debug).

### Identities

- **Kit change re-births the track** (no gate; a bug fix): when 12 of 15 clean
  kit votes name the other team, the track is re-born where the votes turned
  (at most 3 s back), the released identity loses the labels since then and
  is marked missing with its last sighting before the change. Before, the
  track kept its old birth time and ruled out every same-team identity as
  "seen together", creating a duplicate.
- **Size-aware kit votes** (small scenes): a vote counts alone only from a
  crop of quality .35 or more and at least 0.6 × the height the team was
  established at; otherwise the change needs the crops to look like someone
  else, else it is held (`team-hold` event).
- **Kit-consistent galleries** (small scenes): a crop whose own vote names the
  other team, or neither, is not stored.
- **Slots by kit establishment** (small-scene runs): the numbered display
  slots go first to identities with an established kit history.
- **Impossible side** (no gate; a bug fix): the offline connector scores a
  fragment by the better of the identity's sightings before and after it; an
  identity whose sighting on either side is farther than 12 m/s of the gap
  (counted as at least 0.5 s) is not that fragment (`tracklets.SIDE_MAX_SPEED`).
- **Small fragments** (small-scene runs, median box under 30 px or crop
  quality under .35): position decides and appearance only vetoes; the
  fragment's own kit votes must not name the other team; the identity's
  nearest sighting must lie within 2 m + 3 m/s of a gap of at most 2 s and
  lead every identity the position does not rule out.
- **Provisional persons**: anonymous local tracks that chain (end-to-start
  within 1.5 s and 1.5 body heights per half second, or 8 m/s on the pitch,
  leading every rival link) are reported as provisional persons; they are not
  given an identity.

### The ball

On small-scene frames only (`BallTracker.small_scene`):

- **Classical proposals** (`soccer/ball_proposals.py`): where the detector is
  blind, a compact, round, bright, unsaturated blob standing out from the grass
  around the last sighting and the prediction, the size of a ball at that row,
  not part of a line or a kit and not on a shirt or shins. A proposal carries
  detector score .12, needs a chain of two earlier sightings within .3 s,
  never wins over a detector candidate and never lets the tracker stop the
  ball-tile scan (`scan_locked`).
- **The hidden-ball gate at the candidate's depth**: the metres of a gap are
  converted to pixels with the scale at the candidate's row, not at the last
  sighting's (a near sighting made a 21 m gate at the far side).
- **Provisional lock**: a reacquisition beyond the gate locks provisionally and
  flags nothing as `OTHER_OBJECT` until it proves itself: 0.5 s old, moved a
  metre (image and pitch both) and passed within reach of a player (the causal
  camera compensation drifted 1.6-3.4 m/s on a still speck, so motion alone
  proves nothing there).
- **Size prior**: the score peaks at the measured box / prior ratio of real
  balls (1.4) and never expects a ball under 3 px.

In small-scene runs the offline decoder (`ball_path.py`) also audits a causal
lock on a still thing (still, unattended, nothing rolled into it, while a
strong, free, moving competitor was seen: the decoder stops deferring to it),
measures stillness within the camera-compensation drift for things nobody is
near, gives proposals no causal-acceptance bonus, and cuts a stretch that
stood still for seconds, unattended, that the causal pass never took for the
ball and that nothing it took for the ball left (a ball placed for a set piece
is kicked away: it stays).

### Measuring

`tools/diagnose.py RUN_DIR [--window S E] [--out DIR]` exports, read-only,
calibration.csv (the final registration per frame, its anchor, the stored-H
orientation, the final H checked against the detected paint), counts.csv
(people by state, rejected detections and those inside the final pitch),
ball_window.txt (final ball, first pass and every candidate) and
identity_fragments.csv (each identity's local-track segments and what bridged
each gap), plus a summary. `tools/check_ball_truth.py RUN_DIR` scores the
final ball against `tools/annotations/ball_positions.json` (hand-marked ball
centres; a hit is within max(2 diameters, 15 px)); `tools/check_pitch_truth.py RUN_DIR` scores the
final registration against `tools/annotations/pitch_landmarks.json`
(hand-marked points and lines, in pixels and metres).

## The review page

`index.html` is PitchIQ Studio in a dark graphite and forest-green theme:
app background #101715, surfaces #19221F / #202B26 / #28342D, a near-black
media stage #090E0C, text #F1F5F2 with #9AAA9F and #718278 for secondary and
muted text, #334038 borders, #19B96B green (#22CF79 on hover, #117B49 deep),
flat surfaces without gradients, glows or large shadows. The header (60 px)
carries the wordmark, a thin divider, "Match studio / <file>" and three
actions: **New match** (jumps to the source row of the dock below the media),
**Edit tracking** (the match identities and the corrections) and **Advanced**
(the result summary, debug overlays, identity events, pitch confirmation,
calibration and boundary keyframes); the two drawers open beside the
workspace on wide screens and over it on narrow ones. The match hero shows
the upload's name as the title with one line of real data: the analysis
state (No analysis yet, Processing N%, Analysis complete, Processing failed),
the length, resolution and frame rate, and the analysed stretch. Above the
stage a segmented switcher (Tracking, 2D match, Analytics; a sliding lighter
surface with a thin green indicator, arrows, the keyboard arrows) faces a
stage label and a status.

**The stage.** The video and the board sit on the near-black surface (12 px
radius, 1 px border, at most 1450 px wide) sized by the width available AND
by the height left under the stage (`--stage-h`, measured from the stage's
position, the transport, the visible controls and, on screens at least
860 px tall, the head of the dock), so the media, its controls and the dock's
title are in view on a desktop; shorter screens scroll. Switching views keeps
the carousel: the active view slides aside and shrinks while the next grows
in from the other side; all three stay mounted; reduced motion swaps
instantly.

**One match clock.** The `<video>` element is the only playback timeline: the
board and the analytics read its time every frame, and the one transport bar
under the media drives it from the Tracking and 2D match views (a green play
button, outline icons, frame steps, a seek slider with the played portion in
green, one time and duration readout, speed, full screen). Analytics has no
transport. Keyboard on the two replay views: space or `k`, arrows for a
frame (shift, 5 s), `f`, escape. **Full screen** keeps the controls, fading
while playing; where the browser refuses the API the stage takes the
viewport as a fixed layer.

**Analyze segment.** Under the media a dock (#19221F, 1 px #334038 border,
12 px radius) replaces the old upload panel; everything that panel had lives
here. The *source row* names the open video with its length, resolution and
frame rate, and offers **Change video** (the earlier uploads with Open) and
**Upload video**. The *segment timeline* is a video-editor trim strip over
the full source: a time ruler whose labels adapt to the width, a 64 px strip
of real frames of the source (drawn by a second muted video element, cached
per width), a start and an end handle with grips, a green translucent
selected region with a green outline, dimmed unselected regions and a white
playhead showing where the replay is. Drag a handle to move one end; drag
the middle to move the whole range at the same length; the handles are
keyboard sliders (arrows one frame, shift one second, Page Up/Down ten,
Home/End); touch works through pointer events. The range obeys 0 ≤ start <
end ≤ duration, snaps to whole frames at the video's real frame rate, is at
least 0.5 s and at most 3 minutes (the prototype's per-run limit), starts as
the first minute (or the whole clip) and restores the range of the previous
run of the same video. START and END are editable mm:ss.ff fields, DURATION
follows. **ANALYZE** sends start and end as start and length to the same
`/api/run` as before; while it runs the button is disabled, the dock and the
hero show the stage, the progress and the latest processed frame, **Stop**
cancels, and an error keeps the range. The analysis range and the playback
clock are separate: dragging never moves or starts the replay, seeking never
moves the range. **Processing settings** (collapsed by default) holds the
detection detail, expected identities per team, touchline tolerance, the
on-pitch filter, continuing identities, the extra small-ball scans, the
previous run's identities and the GPU. The dock collapses to its title (the
choice is remembered) and is hidden when no video is open.

**Analytics** stays a dashboard: the stage takes the panel surface, the
transport is gone, the pitch sits left and the insight panel right; under
1180 px the panel stacks under the pitch. **No match open** shows a welcome
stage with an upload action and, when earlier uploads exist, a way to open
one. Per-view display options sit under the stage as chips. The same
elements and script as before stand behind all of it; the correction popover
still opens on a player's box.

## Player analytics

Below the 2D match view, the **Player analytics** panel (`analytics.js`,
`soccer/analytics.py`, `GET /api/analytics`) shows, for one player and one time
range, where they spent their observed time (HEATMAP), their average position
(AVERAGE POSITION, alone or with every player of a team or both teams as a
formation-like picture) and their observed path (MOVEMENT TRAIL), on the same
pitch as the 2D view, in the same coordinates (x 0..1 left goal line to right
goal line, y 0..1 far touchline to near touchline), with the same team colours.
Nothing is recomputed from the video: the server extracts the observed samples
of the saved result once per result file (so a correction, which rewrites
the file, rebuilds them with the corrected roster) and answers each selection
from them; the page caches the answers per result.

Who is analysed: the final roster's team players and goalkeepers with a
confirmed slot. Referees, staff, excluded, retired and unconfirmed (`A-?`)
identities are not, and a relabelled player keeps only their own identity's
positions. What counts: an observation of that identity on a frame where the
person is confirmed, the identity confidence is 0.3 or more, the pitch
position lies on the pitch (within 2% of the lines; nothing is clamped), the
pitch confidence at that point (the frame's registration confidence times how
precisely that spot projects) is MEDIUM or better (0.35, the pitch model's own
level `pitch_model.MEDIUM`) and the box is not cut by the frame edge.
Estimates of the timeline (interpolated, predicted) are not observations and
never count, and the 2D view's stabilized positions are not used: the analytics
stay on the raw observations. The samples are smoothed with the 2D view's
0.14 s centred window (never across a gap).

The registration floor was 0.5 until 2026-10-18. A run without keyframes is
capped at MEDIUM by design (an automatic midfield anchor without an
independent painted line, "Pitch without keyframes" above), so on the amateur
run no observation reached 0.5 (pitch confidence p10/p50/p90 .36/.39/.40)
and on the Veo run 16% did: the Analytics view showed nothing while the 2D view
drew the players. The analytics now count what the system itself rates MEDIUM
or HIGH, as the playable region already does; LOW stays out. MEDIUM positions
are less certain, so every answer says how much of the selection's observed
time (time-weighted, like the coverage) rests on a HIGH (0.7 or more) and on a
MEDIUM registration (`quality`, and `averagesQuality` for the players the
average tab draws together), and when MEDIUM holds more than half of it the
status line says that the pitch registration places those positions with only
medium confidence and they are less certain, with the measured share and
nothing more (on the average tab, of the drawn players' time).
On the broadcast run 2.4% of the observations lie between 0.35 and 0.5: its
analytics barely change.

Time, not frames: a sample stands for half the distance to its neighbours,
at most 0.25 s a side (and half a frame on an open side), so denser sampling
never means more occupancy, and of a long absence only a quarter second at
each end is counted. Coverage is the union of those seconds over the selected
interval; under 50% the view says so, and a player without samples gets "Not
enough reliable tracking data for this analysis." Ranges: the processed
segment, the five-minute block the playhead is in (it follows the video), or a
custom interval.

- **Heatmap**: observed seconds accumulated on a 1 m grid, Gaussian-smoothed
  (sigma 2.5 m, the total preserved), drawn with a square-root ramp and a
  legend in seconds per cell.
- **Average position**: the time-weighted mean of the observed positions, a
  circle in the team colour with the label; low coverage is a dashed circle
  with the percentage. A mathematical summary, not a place the player stood,
  and not a role.
- **Movement trail**: consecutive observations joined; broken (nothing drawn)
  where more than 1 s passed unseen or where the position jumped more than
  2 m faster than 12 m/s (a registration glitch); START and END markers,
  direction arrowheads, and a white dot that follows the video along the
  observed path only.

Clicking a player in the Match identities list selects them for the
analytics too. `tests/test_analytics.py` checks the calculations on synthetic
trajectories: density proportional to observed time, sampling rate without
effect, absences not counted, ranges, validity rules, the time-weighted mean,
local tracks combined under one identity, the final roster and corrections,
trail breaks, the axis convention, and the registration floor (MEDIUM counts,
LOW does not, a MEDIUM-only run gets its analytics and says MEDIUM, the
quality shares are time-weighted and pooled for the average tab).
`tests/test_analytics_panels.py` runs `analytics.js` and `teamshape.js` in
node on reports made from synthetic results: the medium-registration note
appears on a MEDIUM run and not on a HIGH one.

`tests/ball_eval.py` renders a panning broadcast sequence with all of these
distractors — and two scenarios where the ball is hidden at a dribble while a
running player's white boots appear a few metres away — and prints the metrics
that matter for the first pass and the final trajectory: frames the SAME ball
was followed, false positives, frames on a boot, teleports (the shown ball
moving faster than any shot between consecutive frames), foot-to-foot switches
(the shown ball in one player's lower body, then another's within 0.3 s), track
switches, hidden frames bridged by prediction, correct / incorrect
re-acquisitions, continuity, state shares and the longest unknown interval:

```bat
.venv\Scripts\python prototypes\gpu-tracker\tests\ball_eval.py
```

What is *not* here: no second-stage ball/shoe classifier and no multi-frame
(TrackNet-style) detector — candidate generation is still the single-frame
YOLO ball class plus the full-resolution crop around the prediction, and the
decoder's node scores take any such verifier as one more factor whenever one
exists. The hard negatives for training one are already in every result: the
rejected candidates with their reasons, boxes and frames.

## Team shape analytics

Analytics has two categories: **Player** (heatmap, average position, movement
trail, as before) and **Team shape** (`teamshape.js`, `soccer/team_shape.py`,
`GET /api/team-shape`, `GET /api/team-shape/snapshot`), with three tabs:
WIDTH & DEPTH, DEFENSIVE LINE and FORMATION. They share the pitch, the
insight panel, the team selector, a range (the analysed segment or a custom
interval) and a SNAPSHOT slider that picks one sample of the analysed data (it
never moves the video; there is no playback on Analytics). A chart under the
pitch shows the measurement over the range (click it to choose the snapshot);
the formation tab shows the formation history instead. A custom range is
validated: an empty field asks for both times, a range given end first is
swapped and says so, and a range outside the analysed segment shows nothing
rather than a boundary sample.

**One shared timeline.** Everything is computed from the same validated
samples as the player analytics (`analytics.Samples`: the final roster's
confirmed team players and goalkeepers, observed, identity confidence 0.3 or
more, registration confidence MEDIUM or better (0.35, `pitch_model.MEDIUM`; a
run without keyframes is capped at MEDIUM, see "Player analytics"), on the
pitch, feet not cut by the frame), so identity corrections, exclusions and
relabelling apply. No team-shape rule depends on the registration confidence;
the report says how much of the team's observed outfield time in the range
rests on a HIGH and on a MEDIUM registration (`quality`), and when MEDIUM holds
more than half of it the status line says the pitch registration places those
positions with only medium confidence and the measurements are less certain.
On top:
5 samples a second on real frames, each standing for the time to its
neighbours; positions with a sigma above 3 m dropped; registration glitches
dropped (a run shorter than 0.5 s that a registration jump separates from the
track on one side, with a jump, a gap or the track's end on the other: a
mapped position that leaps 40 m and back, or leaps and disappears, changes
nothing); two identities of one team within 0.4 m of each other on 3 or more
consecutive samples on which both are seen counted once (the less certain
goes, the most certain identities first; a second detection that flickers on
and off is caught, two teammates side by side for a moment are not merged); a
player you marked as having left the match no longer counted after that time,
even when his identity carried on onto his substitute. Every sample keeps the players' mean
registration confidence and lowest identity confidence. Goalkeepers are kept
apart: shown on the pitch, never in width, depth, formation or the line.
Positions are OBSERVED by default; "Observed + estimated" adds the timeline's
interpolated and predicted positions (sigma at most 8 m), always counted and
labelled separately: an estimate on top of an observed teammate is taken for
the same person and dropped, of two estimates on top of each other only the
more certain counts, and a sample never has more positions than its lineup. Metres use the registration's pitch size;
when it is the standard assumption (no measured size) every metre value says
"est.".

**Coverage.** Each sample compares the observed outfield players with the
lineup: 10, or fewer for the whole video or from a time on (a dismissal),
set under Formation. Width and depth are *complete* with all of them,
*reliable* with at least 8 (all but one of a smaller lineup) when every
missing player's estimated position lies inside the measured extremes (only
estimates of distinct people count: a split identity's stale prediction on top
of an observed teammate is not the missing player), *partial* with 6 or more
otherwise (the observed spread, a lower bound: shown with "≥" and faded, never
in the averages), and *unavailable* below 6. More observed players than the
lineup is *excess* (a duplicate identity, or a player who left still tracked:
mark him as left), kept out of the averages. With estimated positions a sample
is complete only when enough of its players were observed; one made mostly of
estimates is *mostly estimated* (faded, not in the averages), and one where a
lineup player has no position at all stays partial. On the Metrica sample games with a
simulated side camera (a 60 m window following the ball), a whole team was in
view only about 0.1% of the time; those complete and reliable values were
within about 2 m of the true width and depth (90th percentile 1.3 to 3.5 m),
while partial values were 3.4 to 5.4 m short on average (90th percentile 11
to 16 m off), which is why they are labelled and kept out of the averages.

**Width and depth.** Width = max(y) − min(y), depth = max(x) − min(x) of the
measured outfield players, in metres (12 to 57 m across is 45 m; 28 to 79 m
along is 51 m). The players defining them are ringed; the indicators are drawn
along the pitch's own axes; the bounding rectangle is an outline only. The
10th to 90th percentile spread is reported separately as the central spread.
Aggregates over the range: time-weighted mean and median, minimum and
maximum, the share of time reliable, and the other team's averages. Long
ranges are thinned for the chart (at most 1500 points); the report says the
step, and the chart joins samples across it, so a full match draws lines, not
dots.

**Direction.** Which goal a team defends comes from your setting (team A's
goal and an optional time when the sides swap, e.g. the second half), else the
tracking's goal ends (one named end names the other), else the positions when
they agree. Every 30 s (and at the side swap) the positions are checked: the
goalkeeper's side and the goal-side order of the two teams' outfield centres;
the goalkeeper decides when he is seen, the order otherwise (on Metrica the
two disagreed in 1 to 3% of the windows, as often under the right direction as
under a wrong one). Where the evidence contradicts the direction, and in a
window without evidence next to such a window, no line or formation is
reported and the panel says so (an unknown side change is never believed
silently; it is caught within one 30 s window).
Without a setting or goal ends, the last seconds after the final 30 s window
take the side of the evidence just before them. Width and depth do not depend
on the direction.

**Defensive line.** The defensive unit is the back line of the formation
around the sample. The nearest formation window decides the period and the
formation: your label for that period when you set one (also for an "unclear"
or "transition" period), else the recognised formation. Over that period's
windows of that formation within 30 s, the players most often in its back line
form the unit, exactly as many as the back line has. A set piece or a
fullback's run does not change who the defenders are, a swap inside the back
line keeps both, a sustained swap (a centre-back and the holding midfielder
exchanging places for half a minute) is followed without a moment where
neither counts, and the neighbouring periods never outvote a short period or
your label. Back threes,
fours and fives are supported. Without a formation window within 30 s the
nearest one within 60 s is used ("last known roles"), then (with at least 8
players present) the 3 to 5 deepest players over 30 s behind a gap of 4 m or
more, found per defending direction. LINE HEIGHT = the median of the measured
defenders' distances from their own goal line (31, 32, 34, 33 m give 32.5 m);
DEEPEST DEFENDER = their minimum (31 m). Complete with every defender
measured, partial with one missing, unavailable with fewer, no direction or no
unit; *displaced* when the unit is ahead of most of its team (an attacking
corner): no line then. Quality from the defenders' spread along the pitch:
coherent (within 8 m), uneven (one defender stepped up, the others within 8 m,
for a back three too), stretched (within 15 m), disrupted (beyond: out of the
summary). The deepest defender is summarised from complete samples only (with
one defender unmeasured, he may be the deepest; the snapshot shows "≤"). With
estimated positions, a line standing mostly on predicted defenders is *mostly
estimated*: drawn faded, not summarised. The line is drawn across the
unit's lateral extent (not as a full-width offside line), solid when complete
and coherent, dashed otherwise, with the distance from the own goal line; the
chart shows height and deepest defender over time with gaps where there is no
value, and a trend (a weighted linear fit: advancing, retreating or steady).
Possession is not in the result, so the line is positional and labelled so.

**Formation.** Recognised from observed positions only, in 15 s windows
every 5 s, in the team's attacking frame (own goal at 0), with at most one
player of the lineup missing (9 of 10, 8 of 9), each observed for half the
window (on Metrica with a simulated camera, 8-player windows agreed with the
full-coverage label only 25 to 56% of the time). A window spanning a lineup
change is not classified. When more players than the lineup were observed (a
substitution), the previous window's players, then the most observed ones,
form the shape, and such a period is never called clear. A player's position in a
window is his mean offset from a voted team centre (each visible player's
position minus his own offset, so a camera on one side does not drag the
shape). Candidates: 4-4-2, 4-3-3, 4-2-3-1, 4-1-4-1, 3-5-2, 3-4-3, 5-3-2, 5-4-1
(4-4-1, 4-3-2, 5-3-1 for nine). Templates are the mean role positions of
SoccerCPD's 864 real formation periods (Kim et al., KDD 2022) for 4-4-2,
4-2-3-1, 4-3-3, 3-5-2 and 3-4-3, the others composed from the same lines; each
is stretched and shifted per axis (within 0.6 to 1.8 of its size) and the
players are assigned to its roles one to one by the Hungarian algorithm, a
player leaving the back line he held in the previous window paying a
continuity cost. The fit error is the RMS distance from the fitted roles
(shown in metres, compared relative to the team's spread). Stabilization is a
change-point search (Viterbi over the formations and an "unclear" state) over
each unbroken run of classified windows: a change costs as much as 0.45 of
relative fit error summed over windows, so a clear change takes 15 to 30 s and
a slight one over a minute; any window without the data for a formation
(coverage, direction, lineup) ends the run, so the label after a gap is
decided afresh. A formation period under 30 s between two periods of evidence
is a transition; next to a data gap it keeps its label. On Metrica (tuned on
game 1) that brought about 250 raw window-to-window label changes an hour
down to a handful per half on the held-out game 2. A period is *clear* (margin
to the runner-up at least 10%, 60% of its windows agreeing, every player of
the lineup, at least 3 windows, mean identity confidence 0.6 or more and at
least 70% of its players holding one role), *likely*, or *ambiguous* (margin
under 4%: "4-4-2 or 4-2-3-1"). The view shows the period's average shape from
observed positions with the goalkeeper's average position, its units joined
left to right from the team's own point of view, the fitted template's role
anchors as faint circles (players are never moved onto them), or a snapshot
with the roles of its window. The history strip shows only analysed periods;
click one to show it, and moving the snapshot selects the period it falls in.
Possession is not known, so shapes are observed positional structures over
both phases, labelled so.

**Corrections and settings** (`runs/<video>/team_shape.json`, apart from the
tracking): the defending direction and the side-swap time, the outfield lineup
per team (for the whole video, or from a time on), and formation labels for
periods (`POST /api/team-shape/settings`, `/api/team-shape/formation-label`,
`/remove`). A label leads the panel with "set by you" while PitchIQ's
prediction stays beside it unchanged; no position moves; the label decides
that period's defensive unit. A label belongs to a period when they share at
least half of each: it still finds its period after a re-run moves the
boundaries by a few seconds, a period now split in two keeps it on both
halves, and a short label never takes over a long neighbour. A label that no
period matches any more is kept and reported as not applied; saving a label
replaces, and removing removes, every label of that team belonging to the
period. Labels are kept with the
prediction they replaced, for evaluating the recognition later. Writes are
serialized and atomic (a unique temporary file, retried while Windows holds
the old one open), and a settings file that cannot be read is never taken as
empty by a save. Malformed settings or requests are refused or dropped, never
an error page.

**What is not claimed.** No formation accuracy is reported: the tests use
hand-placed layouts deliberately different from the templates, and Metrica's
sample games have no expert labels. `tools/team_shape_bench.py` runs the whole
analysis on the Metrica games (with `--camera WIDTH` for a simulated broadcast
camera and the errors by status against full coverage).

## Possession

The **2D match** view shows which team has the ball, who carries it and where it
changed hands (`soccer/possession.py`, `possession.js`, `GET /api/possession?run=<run>`):
a strip under the board in the board's team colours (solid while a player controls
the ball, fainter while the team's pass travels or the ball is loose, hatched where
the ball is interpolated, grey where nothing is named, dark while the ball is out of
play), a triangle for every change of possession (solid for a `high` change, hollow
for an `uncertain` one; untick **Show uncertain changes** to hide the hollow ones), a
one-line key saying that every change is an estimate, a live line for the moment
shown ("Team A in possession · A-03 on the ball", "· ball in flight", "Loose ball",
"Ball out of play · Team B to restart", "Possession unknown"), each team's share of
the time a team is named with the coverage, and a ring around the carrier on the
board (a thin ellipse at his feet on the tracking video when **Player boxes** is on).
A marker's tooltip names who lost and who won the ball, both times, the cause and the
confidence; a click pauses on the frame the ball was won, where the line names the
winner and the ring marks him. A click on the strip seeks there. The page reserves
110 px under the board for the panel (46 px for its one-line message), so the board
is smaller on this view (445 x 301 px instead of 608 x 411 at 1440 x 900).

**It is an estimate**, judged from the 2D positions of the players and the ball, not
a record of the match: the page says "estimated", "uncertain" and "unknown" where the
engine does, and a run it cannot judge says why in one line. Where it names a team it
was right 95-96% of the time on data nobody had tuned on (Metrica game 3 and a fresh
broadcast segment, below). When the ball changed hands is an approximate boundary
between attack and defence, not a counted turnover.

**Method.** Nothing here runs a model or reads the video. On every frame a team is
*at the ball* when one of its players is near it on the pitch (full evidence within
0.7 m, none from 1.5 m) and in the image (the `ball_path.within_reach` geometry,
1.0-2.0 m); the same player must be near on both, so a carrier is named only where
the two agree. A ball between two opponents, or at a person who has no team yet, is
evidence for nobody; referees and staff are neither at the ball nor in the way; a
goalkeeper the tracking has not given a team plays for the goal on his side
(`goalEnds`), and a ball in a goalkeeper's hands in his penalty area (its centre above
his lower body in his box) is his. An interpolated ball counts half, with its
`nearTrack` player only where the ball tracker's first pass followed the same ball,
within 2 m of the final one (on the broadcast's interpolated frames the nearTrack
person stood more than 2 m from the final ball on 50-66% of them, every time the
first pass had followed something else). A fast ball counts only where its path
bends (a touch), so a pass flying past a player is nobody's, and a ball knocked past
a defender is not his. A two-state decoder (Viterbi) then picks the possession
sequence: a change costs 0.15 s of evidence and can only start on a run of one
player's evidence that holds an observed sighting of the ball on both metrics, so it
lands on the winner's first control, and a deflection or a challenge that does not
keep the ball is no change. The winner is the new team's player with the most
evidence over the next 0.5 s. The decoder's forward-backward posterior says how sure
each frame is, and a team is named only where it is at least 0.8 (on Metrica game 1
naming every frame would give .949 at .986 coverage on full tracking instead of .973
at .940, and .928 instead of .969 on the camera).

Possession restarts after a dead ball, an unusable ball for 1.5 s or more and a
camera cut (then from the first observed control). *Out of play* is the ball seen
1 m beyond a line for 0.2 s and not back within 1 s, if it got there plausibly: from
the pitch at no more than 40 m/s, with the ball tracker's first pass not following
another ball and no other ball in play on the pitch on most of those frames;
otherwise that stretch is no ball (possession is carried over it, never dead). *A
stoppage* is the ball still for 4 s, or both teams slowed down (the 75th percentile
of the players' speeds under 2.2 m/s for 3 s) while the ball is unseen, still or
unplayed; a ball that a player got in play and keeps is never a stoppage. A stretch
between two camera cuts that shows another moment (the ball's pitch position jumps
at both cuts) is a possible replay and shows no ball. The restart belongs to the
team of the player who takes it. A result without `ball.origin` (runs from before
2026-10-15) takes every tracked ball as observed, and the record says so in a note.

**Conventions** (fixed before any engine code, with the metrics and the targets).
Team possession persists from a team's control through its passes, shots,
clearances and loose balls until the other team controls the ball (a change) or the
ball goes out of play (dead until the restart, which belongs to the restarting
team). A change is timed twice: `lostAt`, the losing team's last touch, and `wonAt`,
the winning team's first control. Timeline states are `control` (with the carrier),
`flight`, `loose`, `contested`, `dead` and `unknown`. Every time is absolute source
seconds, every frame an absolute index and every player an internal id; the page
shows the display labels of the result as it is now, so a correction relabels the
tooltips. The record (`compute(result)`, plain JSON, `version` `possession-4`):
`version`, `fps`, `t0`, `t1`, `available`, `reason`, `quality` (`level` good /
limited / unavailable, `coverage` = the share of the live frames with a team named,
`notes`), `shares` (seconds of A, B, dead, unknown), `timeline` (runs of constant
team, state and carrier: `t0`, `t1`, `f0`, `f1`, `team`, `state`, `carrier`,
`estimated`, `confidence` high / medium / low), `changes` (`id`
`pc-<won frame>-<team>`, `lostAt`, `wonAt`, `frame`, `fromTeam`, `toTeam`, `lostBy`,
`wonBy`, `cause` tackle_or_duel / interception / out_of_play / unclear, `confidence`
high / uncertain) and `deadSpells` (`t0`, `t1`, `reason` out_of_play / stoppage,
`restartTeam`). Right at a change the posterior is usually under 0.8 (5 of the 6 won
frames of the broadcast run 42-222 s are in an `unknown` run), which is why the page
names the winner on the won frame from the change itself. Any result gives a record,
a single frame included.

**Confidence.** A change is `high` when it is not undone within 2 s (and does not undo
one), the new team has 0.3 s of evidence over the next 2 s, the loser's last touch is
at most 5 s before and the new possession lasts 2 s; every other change is
`uncertain` (back-and-forth pairs, changes with little evidence, restarts after a
dead ball, changes across a gap or a cut, changes whose loss nobody saw).
possession-3 rated them `medium` and `low`; the two were merged because they were not
ordered (Metrica game 1, within 2 s: medium 67%, low 74%), and the page still draws a
possession-3 `low` change hollow and a `medium` one solid. Right within 2 s, pooled
over full tracking and four simulated cameras:

| change confidence | Metrica game 1 (tuned) | game 2 (seen) | game 3 (held out) | broadcast |
|---|---|---|---|---|
| high | 444 of 474 (.937) | 394 of 439 (.897) | 489 of 564 (.867; .83-.92 per variant) | 2 of 2 (42-222 s, dev); 1 of 1 (222-303 s, held out) |
| uncertain | 325 of 462 (.703) | 323 of 467 (.692) | 452 of 643 (.703) | 2 of 4; 0 of 1 |

So on data like Metrica's a `high` change is right about 85-90% of the time and an
`uncertain` one about 70%. Hiding the uncertain ones would hide 4 to 5 in 10 of the
right changes (game 1: 325 of 769; game 3: 452 of 941), which is why every change is
shown by default. A timeline run's confidence (`high`, `medium`, `low`) is how sure
the decoder is of that run's team. A run's `quality` is `limited` under 60% coverage
or under 85% agreement (below); on Metrica game 3 the `limited` windows were only
.004-.022 less accurate than the `good` ones, so it is a caution, not a measured
threshold.

**When it says nothing.** A run is not judged when half or more of its frames are a
small, far scene (amateur and wide cameras), when the ball is placed on under 20% of
the frames, when no team player is tracked near the ball, or when the pitch and the
image agree on the team at the ball on under 65% of the frames where one of them
puts one team near it (over at least 150 frames). The test runs fall clearly on
either side: broadcast 0-60 s and 42-222 s are a small scene on 0% and 0.9% of their
frames with agreement .877 and .869 (222-303 s: .81); amateur 3 and the Veo clip on
60% and 61% (agreement .277 and .571), and both abstain.

**Measured.** The data roles were declared before each round: *tuned* or *dev* data
chose the rules and constants, *seen* data had been read after an earlier freeze, and
*held-out* data was scored once with the engine frozen and is spent now. The truth of
Metrica sample games 1-3 is derived from their events (`tools/possession_truth.py`),
and the games are turned into PitchIQ results under full tracking or through a
simulated broadcast camera (`tools/possession_sim.py`, fitted to the ball audit of the
broadcast run, two seeds), in windows of 300 s, each a separate result. The broadcast
segments of the La Liga run have blind, adjudicated hand labels (two labellers, a
point every 0.5 s). Changes are matched on `wonAt` within 2 s with the same new team;
"without one-touch" leaves out Metrica's one-touch possessions (a first-time
interception or clearance that Metrica codes as a gain and the convention absorbs as
a deflection). possession-4, frozen on 2026-10-20, with the default settings:

| data (role) | which team: accuracy at coverage | when it changed hands: changes within 2 s | dead ball: recall / precision |
|---|---|---|---|
| Metrica game 3, full tracking (held out) | .964 at .942 | F1 .687 (precision .848, recall .578); .718 without one-touch | .897 / .965 |
| Metrica game 3, simulated camera, seed 0 / 1 (held out) | .956 at .881 / .959 at .883 | F1 .615 / .586 (precision .776 / .747); .636 / .609 without one-touch | .880 / .954, .864 / .955 |
| broadcast 222-303 s, 162 points (held out) | .949 at .924 (92 of 97 named team points right) | precision .50 (1 of 2 reported), recall .25 (1 of 4) | 28 of 55 dead points called dead; 28 of the engine's 30 dead points truly dead |
| Metrica game 2, full tracking (seen) | .968 at .933 | F1 .681; .767 without one-touch | .816 / .976 (.915 without its one abstaining window) |
| Metrica game 2, simulated camera, seed 0 / 1 (seen) | .965 at .888 / .972 at .884 | F1 .581 / .617; .644 / .665 without one-touch | .890 / .973, .906 / .973 |
| Metrica game 1, full tracking (tuned) | .973 at .940 | F1 .691 (precision .904, recall .559); .758 without one-touch | .929 / .985 |
| Metrica game 1, simulated camera, seed 0 / 1 (tuned) | .969 at .881 / .968 at .883 | F1 .607 / .571 (precision .840 on seed 0); .678 / .622 without one-touch | .872 / .979, .857 / .978 |
| broadcast 0-60 s, 120 points (dev) | 117 of 117 named team points right (coverage .992) | the clip has no change; none reported | 2 of 2 dead points |
| broadcast 42-222 s, 360 points (dev) | .977 at .975 | precision .667 (4 of 6 reported), recall .667; 2 of 6 within 1 s | 9 of 41 dead points; 9 of 9 |
| amateur 3 and Veo (dev) | abstain: no frame named | none | none |

Against the pre-declared targets (unchanged since possession-2), every accuracy and
coverage target is met on held-out data (full tracking .95 at .85, camera .90 at .65,
broadcast .90 at .60), and the amateur and Veo clips (dev data since possession-3)
abstain as they must; no change target is met (F1 .80 on full tracking: .687; .65 on
the camera: .615 and .586; precision .70 on the broadcast: .50). Resampling game 3's twenty windows puts full-tracking F1 within
.637-.739 (95%), and the change F1 has stayed at about .69 (full) and .60 (camera) on
three games and three engine versions, which suggests that the targets are out of
reach of this design rather than of its tuning. Who lost / who won the ball on the matched changes:
game 3 .854 / .883 on full tracking, .717 / .733 on the camera (seed 1 .735 / .740);
game 2 .894 / .856 and .710 / .703. Loose balls are the weakest frames: on game 3
.755 right on full tracking and .665-.687 on the camera, against .985 and .980 on
the other frames. For the broadcast numbers: always answering the main team scores
.810 on 222-303 s and .820 on 42-222 s, and the labellers agree on the team on 96%
and 315 of 316 of the points. None of possession-4's new rules fires on 222-303 s
(its record equals possession-3's apart from the confidence names), so that segment
measures the rules of possession-3.

**What it cannot do (yet).**
- *When the ball changed hands* is not reliable enough to count turnovers or to say
  who lost the ball: on held-out Metrica the change F1 is .69 (full tracking) and
  .59-.62 (camera), and on the fresh broadcast segment 1 of the 2 reported changes
  was right and 3 of the 4 real ones were missed. The false one (265.43 s) fell
  inside a throw-in stoppage, after an out-of-play spell had ended too early.
- *Timing.* On Metrica the matched changes are a median 0.03 s (won) / 0.08 s (lost)
  off on full tracking and 0.19 / 0.32 s on the camera, but within 1 s the F1 falls
  to .642 (full) and .49-.50 (camera). On the broadcast the ball is often hidden or
  misplaced at the duel: on 42-222 s the four right changes were 0.7-1.9 s off
  (median 1.05 s at the win), 2 of the 6 within 1 s.
- *One-touch and short possessions* are absorbed: a first-time interception or
  clearance, and possessions shorter than about 1-1.5 s (game 3, full tracking: 28 of
  the 101 changes into a one-touch possession found, 158 of the 190 changes outside
  them).
- *Dead balls on broadcast footage*: 28 of 55 dead points found on 222-303 s, 9 of 41
  on 42-222 s. 222-303 s has the first balls tracked over a line in PitchIQ footage
  that were checked: one became an out-of-play spell, the other (286.4-288.7 s) was
  thrown out as not the ball in play because the ball tracker's first pass was more
  than 2 m away (the check that keeps a steward's ball off the pitch). An out-of-play
  spell ends as soon as an interpolated ball comes back within 1 m of the line
  (253.07-262.17 s ended 11.5 s early); the restart team can come from a weak
  detection at a player's feet (274.6 s, the wrong team); a foul stoppage with the
  ball left rolling is missed (42-222 s). Dead time the engine misses is often named
  as a team's possession: 26 of the 55 dead points on 222-303 s (nearly all of those
  it missed) and 11 of 41 on 42-222 s. Restarts and set pieces should not be taken
  from it on broadcast footage yet.
- *A goalkeeper holding the ball*: the hands rule has never fired on a PitchIQ run
  (no ball in a keeper's box on the five runs checked) and the simulator never puts
  the ball in his hands, so its precision on real footage is unmeasured. On synthetic
  scenes a cross passing along the keeper's image column, or an attacker holding the
  ball just behind him as the camera sees it, gives a false change and change back. A
  hold the ball tracker does not see is still a stoppage or a gap.
- *Replays*: a replay is recognised only when the ball's pitch position jumps at both
  camera cuts; a replay of the same place is read as live play, and on a synthetic
  scene a replay shown during an out-of-play stoppage, with the live ball beyond the
  line afterwards, gave a false high-confidence change. PitchIQ set no camera cut on
  any of the runs checked, so none of this has been seen on real footage.
- *A ball held in play* is no stoppage since possession-4 (on held-out game 3 the
  in-play time called dead fell from 85.4 to 60.1 s on full tracking and from 91.6 to
  78.0 s on the camera), which costs .005-.020 of dead recall (on game 1, keepers
  fetching the ball for a goal kick look like a catch). On a synthetic scene where
  the fouling team stops the ball at the whistle and holds it until the free kick
  there is no dead ball and two false high-confidence changes. A ball unseen for
  1-3 s just after a reception, with the players walking, can become a false
  stoppage.
- Identity and team errors of the tracking become possession errors: an identity
  plays for its roster team (a frame's own team would override it, but PitchIQ never
  writes one that contradicts the roster). Correct them (`Correcting who someone
  is`): the record is computed again from the corrected result.
- Amateur and Veo footage get no possession, by design: with the gates switched off
  amateur 3 would have scored .544, below always answering its commoner team (.567).

**Speed.** On broadcast 42-222 s (5400 frames) the features take 0.66 s and the
decisions 0.09 s, after 1.05 s of reading the 111 MB result (peak memory 51 MB); on
the server the first request after the result changes takes 2.65 s and later ones
0.012 s from the cache (one record per run, kept until `result.json` changes). A
missing or failing engine answers 503 on this route only and the page says so in
one line; the other views never notice.

**Reproduce.** The Metrica sample data are downloaded files, not in the repository,
so the tools read them only under `python -I -B` (games 1-2: the tracking CSVs and
the events CSV; game 3: Metrica's EPTS release, the metadata XML, the tracking text
and the JSON events in one folder, passed as both `--tracking` and `--events`):

```bat
.venv\Scripts\python.exe -I -B tools\possession_bench.py --tracking DIR --events DIR [--json out.json]
.venv\Scripts\python.exe -I -B tools\possession_bench.py --tracking DIR --events DIR --heldout --frozen NOTE
.venv\Scripts\python.exe -I -B tools\possession_bench.py --tracking G3 --events G3 --heldout --game 3 --frozen NOTE
.venv\Scripts\python.exe -I -B tools\possession_bench.py --labels tools\annotations\possession --run BC180=<run>\result.json
```

The first prints every game-1 number (about 7 minutes; `--quick` gives the main
table), the second game 2 and the third game 3 (`--heldout` refuses to run unless
the freeze note named by `--frozen` exists; it does not check what the note says).
The truth is derived on the fly by `tools/possession_truth.py` (game 3's tracking
has Player6 and Player13 swapped over the second half: `EPTS_SWAPS` corrects it) and
the results are simulated by `tools/possession_sim.py`. `--labels` scores a run
against the hand labels of `tools/annotations/possession/` (broadcast 0-60 s and
42-222 s, amateur 3, Veo; its README says how they were made); the labels of
broadcast 222-303 s are kept with the freeze notes, outside the repository, in the
same format.

## Coaching report

The **Coaching report** tab (`coaching.js`, `soccer/coaching/`, the `/api/coaching*` routes) turns a finished run into
a report for one team, over the analysed segment or a custom range (source-video time). The report is adapted to the
team the coach describes: age group, playing level, match format and style.

- **Read-only.** It never changes the tracking, the result, Analytics, Team shape or Possession. A golden check
  (2026-10-22) hashed 170 outputs of `result.json`, `/api/analytics`, `/api/team-shape` and `/api/possession` on five
  runs, and all were byte-identical to the committed code's.
- **Not validated by a coach yet.** Every threshold, verdict rule, drill and progression rule is a PitchIQ proposal,
  stored as `origin: PITCHIQ-PROPOSAL, validated: false`.
- **No federation endorsement.** Federation guidance is paraphrased with its source. No federation reviewed or
  endorses PitchIQ.

**Setup (per video and team).** The coach sets:
- age group, U6 to Adult;
- playing level, 1 to 7: Foundation, Early development, Developing, Intermediate, Advanced youth, Competitive adult or
  academy, Professional or elite;
- match format: 4v4, 5v5, 7v7, 9v9 or 11v11, with a goalkeeper toggle for the small formats;
- federation, field size, intended formation, style, confirmed player roles, and a squad for progression.

Defaults:
- **Format:** the U.S. Soccer format for the age. With England or Canada chosen, that federation's own format where
  the research has it: England 5v5 at U7-U8, 7v7 at U9-U11 and 9v9 at U12-U13; Canada 5v5 at U8-U9 (4v4 without a
  goalkeeper is also played), 7v7 at U10-U11 and 9v9 at U12-U13. Elsewhere PitchIQ uses its nearest format and says so
  (Canada plays 3v3 at U6-U7, which PitchIQ does not assess); it never calls that the federation's format.
- **Heading:** the stricter of the U.S. Soccer and England rules always applies, whatever the federation setting.
- **Nothing set:** an adult level-6 11v11 team, with a notice saying these are defaults.

The settings are kept in `runs/<video>/coaching_settings.json`. A save that would drop or trim something is refused,
not silently cut.

**One measurement, many coaches.** The report is built in three cached tiers (`runs/<video>/<run>/coaching/v2/`):
- **Measurement** (`obs-*`): coaching series, situations, the six checks' opportunities and their evidence moments.
- **Role** (`roles-*`): the role book and the wide-defender advances.
- **Interpretation** (`report-*`): assessment, suggested areas, wording, drills.

What changes trigger a rebuild:
- **Level, age or style:** only the interpretation tier reruns. That took 11-33 ms on the test runs, inside the GET
  request.
- **A confirmed role or formation:** the role tier reruns.
- **Format, field size, the coach's possession marks or a tracking correction:** the measurement tier is rebuilt in
  the background (0.13-2.9 s on the test runs).

The observation files never change with the coaching profile. The evaluation re-measured every run copy under 14
profiles and every synthetic scene under 4: the bytes were identical every time.

**The six checks.** Which checks a level turns on, and which verdicts each may give, follow the profile table in
`soccer/coaching/profiles.json`. The verdicts are strength, work on, try this, review and describe.

| # | Check | What it measures (when) |
|---|---|---|
| D1 | Pressure and cover | Does the nearest defender press the ball carrier (within 5 m; at levels 1-2 also "goes to the ball")? Does a team-mate cover behind him? (the other team has the ball) |
| D2 | Far-side balance | Is someone goal-side in the far half when the ball is wide? (other team's ball) |
| D3 | Wide-defender advance | When a full-back or wing-back goes 15 m beyond the back line, does a centre-back shift, a midfielder drop or a team-mate take the channel? (our ball) |
| D4 | Behind the ball | Are there enough players goal-side of the ball with the ball in the final third? (our ball) |
| D5 | Defensive spacing | Gaps in the back line and between the units (other team's ball) |
| D6 | Attacking shape | Width, depth and channels used (our ball) |

All checks except D6's descriptions need to know who had the ball. That comes from one of two sources:
- the possession record (Possession, above), on broadcast footage;
- **phases the coach marks** while watching, under the video or the 2D match: "our ball / their ball / dead ball from
  here". The marks are kept per run in `runs/<video>/coach_spans.json`, shown on their own lane, and the report words
  their findings as coach-marked. The marking bar opens from the report ("Mark who had the ball"); its button in the
  tracking and 2D controls appears once the run has marks or the bar is open, so a run nobody marked keeps the earlier
  layout. Shares of who had the ball are said with their source: your marks, or your marks where you made them and
  PitchIQ's estimate elsewhere; with only one team's spells marked on footage PitchIQ cannot follow, no share is given
  and the report says which checks wait for the other team's spells.

Some things the report never assesses: recovery after losing the ball, passes, communication and outcomes. MATCH
OVERVIEW lists them under "Not measured by PitchIQ", with the reason.

**When it declines.** A finding needs at least the level's number of separate spells and opportunities. A failing
pattern with too little evidence becomes a review, never a work item. Nothing is criticised in these cases:
- an identity is in doubt at the moment;
- the formation changes within 15 s;
- the player is sprinting back;
- the calibration or the view does not support the measurement.

Short clips get their own rules. Under 3 minutes, the report never says "repeated" or "pattern", and a work item needs
3 separate moments. At levels 1-2:
- no player is named in a challenge;
- formation discipline is never judged;
- the report shows at most 1 work item at level 1, and 2 at level 2.

**Evidence and suggested areas.** Every finding cites key moments from the observation file:
- **The frame:** its exact frame, with seek = (frame + 0.5) / fps.
- **The ring:** a focus ring on the player's observed position at that frame.
- **The overlay:** drawn on the video where the calibration supports it, otherwise on the top-down diagram.

For a criticised moment, the report may also propose a **suggested area**: where one team-mate could have reached in
about 2 s. The area restores the relation the check found broken, opens no new gap, and stays 3 m clear of
team-mates. When no such area exists, the report says why.

**Words.** `explain.py` writes the text from structured claims, in a register for each level. A deterministic checker
rejects any sentence that:
- states a number, time or id that is not a fact of the finding;
- names an event PitchIQ does not measure (passes, turnovers, "lost the ball");
- uses a concept the level has not unlocked;
- uses a banned word for the level, or a generic sentence.

A rejected sentence is replaced by a shorter one, or the finding is withheld.

**Drills.** The library holds 14 drills. All are PitchIQ adaptations, credited to the sessions they adapt (sources in
`soccer/coaching/sources.json`: 122 sources, 90 read in full). A drill is offered only for a reported work item of its
own check, or once as an optional drill for the next objective. The safety filter runs before any ranking:
- no heading drill at U11 or younger;
- physical load capped by age;
- no tackling below U9.

**Progression.** Link a team to a squad, and its matches are followed per skill in `runs/_coaching/progress.json`.
- Rates count per judged moment, never per match total. Cutting Metrica game 1 into 60 s clips gave the whole
  match's counts exactly.
- The mastery states (LEARNING to ADAPTABLE) come from `progress_rules.json`, also unvalidated.

**PDF.** EXPORT PDF prints the same seven sections:
1. match overview;
2. what went well;
3. what needs work and what to review;
4. key moments, with zoomed frames;
5. tactical diagrams;
6. training drills;
7. development progression, with the sources.

The file is named `pitchiq-coaching-<video>-<run>-<team>-level<N>.pdf`.

**Feedback.** A coach's verdict on a finding (correct, partially correct, incorrect, not enough evidence, with a note)
and any area the coach moved are kept in `runs/<video>/coaching_feedback.json`, apart from the report.

**Footage.**
- **Broadcast footage** gives the most.
- **Amateur and Veo clips:** possession is not estimated, so every check that needs who had the ball reports "not
  assessed" until the coach marks the phases.
- **Small fields:** the pitch model knows only full-size markings, so on a small field a coach should declare the field
  size. Checks that need true metres are "not assessed (scale unknown)" while a small field's size is only assumed.

**The first (v1) report** (context levels development / competitive / advanced, judged against Metrica game 1's
professional reference) is deprecated. It is still served when a request names a `context`, and
`tools/coaching_report.py --context` still makes it.

**Evaluation (2026-10-22, `tools/coaching_eval.py`; scratch outputs, nothing under `runs/`).**

- **Synthetic counterexamples** (`scenes`; `tests/test_coaching_counterexamples.py`). The 30 variants of DESIGN
  13.1's scenes in `tools/coaching_synth.py` ran through the server's own measurement path and the real
  interpretation at levels 1-7:
  - 380 checks pass and 0 fail; 0 report invariants are broken on any report.
  - 1 known open defect: the identity-crossing scene gives a level-1 "try this", where DESIGN asks for at most a
    review.
  - At level 4, 6 of 6 planted unsupported-advance scenes became work items, and the covered twins drew no criticism
    at any level.
- **Same video, different levels** (`levels`; `tests/test_coaching_levels.py`):
  - The enabled checks and verdicts follow DESIGN 2.3's table, written out independently of `profiles.json`.
  - Over 30 scene variants x 14 profiles, the most similar explanations of one finding at two levels score a
    word-trigram Jaccard of .37 (the limit is .5).
  - Two open defects: the attacking-shape description repeats between levels 5 and 7 (.51-.80), and a few findings
    show different moments when a level judges a different failing set.
- **Real run copies** (`runs`): BC60, BC180, BC303, AM and VEO, both teams, 14 profiles, plus AM and VEO with a
  synthetic coach-marked phase made from the blind possession labels:
  - 0 invariant problems on 196 reports (7 run variants x 2 teams x 14 profiles).
  - No work item at levels 3-7. The only work items are BC180 Barcelona's attacking depth at levels 1-2 ("try
    this"); attacking shape never gets an area.
  - Every other real criticism is a review whose evidence guardrail failed, so no suggested area is drawn on the real
    clips.
  - Without marks, the amateur and Veo clips are "not assessed" for every phase-dependent check.
- **Pressure distance** (`statsbomb`): checked on StatsBomb open data (Euro 2020 matches with 360 freeze frames; data
  by StatsBomb, not copied here). Over 8,576 carrier events with a visible opponent:
  - Under pressure, the nearest outfield opponent was a median 2.1 m away, and 84% were within PitchIQ's 5 m.
  - Not under pressure, the median was 5.7 m, but 44% were still within 5 m (precision .32).
  - "Nobody within 5 m" agreed with StatsBomb's "not under pressure" on 94% of events (91% in the defending third).
  - So 5 m rarely calls an unpressed carrier pressed by mistake, but it counts many unpressed carriers as pressed.
- **Blind label sheets** (`sheets`, in `tools/annotations/coaching/labels/`; the scorer is
  `coaching_eval.py --labels DIR`):
  - MOMENT: 266 moments (200 broadcast, 47 amateur, 19 Veo).
  - ROLE: 60 players.
  - ADVANCE: 8 spells.
  - ZONE: 48 diagrams.
  - The sheets show the video at a time, or a clean diagram, and short questions; never PitchIQ's output. Until a coach
    fills them, no coaching judgement here is claimed validated.
- **Metrica game 3:** reserved for one held-out test on frozen rules, under a plan declared in advance.

```bat
.venv\Scripts\python.exe tools\coaching_report.py runs\<video>\<run>\result.json --team A --level 4 --age U14 ^
    --format 11v11 --out report.json --pdf report.pdf --video uploads\<video>.mp4
.venv\Scripts\python.exe tools\coaching_eval.py scenes --json scenes.json
.venv\Scripts\python.exe tools\coaching_eval.py runs --out DIR --run BC60=...\result.json --spans AM=tools\annotations\possession\AM.json
.venv\Scripts\python.exe tools\coaching_eval.py --labels FILLED_DIR --sheets tools\annotations\coaching\labels
.venv\Scripts\python.exe -I -B tools\coaching_eval.py statsbomb STATSBOMB_DIR
```

## Debug view (how to test)

- **Identity debug** labels every tracked person, e.g. `A-07 · Track 91 · Team A ·
  Role: PLAYER 92% · Identity: 94%`; goalkeepers `GK-A (GK-1)`; unresolved people
  `A-?`, `GK-?`, `UNK-3`, with the reason they are not yet identified; ignored
  people faintly as `IGNORE · referee like` (or `sideline official`) with their
  local track; and under each person the role evidence in one line (`looks ref
  0.84 A 0.16 B 0.72 | touchline 94% along 0.91 | ASSISTANT REFEREE 0.98`).
  Colours: violet team A, orange team B, pink goalkeepers, dotted while the
  identity is uncertain, grey for unknown and ignored people, white dashed for
  candidates, red dashed for people rejected outside the field. Without debug
  only the people that matter for the match analysis are drawn (players,
  goalkeepers, candidates, the ball): ignored, rejected and unknown people stay
  unmarked.
- **Pitch boundary** shows the playable region in light blue (teal when it comes
  from the pitch model: solid from a calibration, long dashes when predicted;
  red dashes while the confidence is LOW) with the status line, e.g.
  `PITCH CONFIDENCE: 0.43 · USING TEMPORAL PREDICTION (4.1s since confirmation)`
  or `... · GRASS REGION ONLY, NO PAINTED BOUNDARY (uncertain) · NEEDS
  CONFIRMATION`; it should sit on the white touchlines and goal lines. With
  Identity debug on, it also lists the evidence (`far touchline: FOUND (paint
  3 px from the prediction)`, `near touchline: PREDICTED (in view, no paint
  matched)`, `left goal line: OFF-SCREEN (predicted from the model)`, `tracked
  people on the predicted pitch: 18 of 20`), the detected boundary lines (white;
  yellow dashed when carried by camera motion), goal-end hints (orange dashed),
  the raw grass hull (dashed) and which source each edge uses (`line`, `carried
  line`, `grass edge`, `predicted`, `off-screen`), plus your boundary keyframes.
  The *Pitch model* event filter lists anchors, refinements, doubts and losses.
- The **ball** is drawn only once: `BALL 0.94`, or `BALL PREDICTED` (dashed)
  while it is hidden, with the track id, state and speed, the predicted next
  position (a cross) and a 1 s trail in Identity debug; otherwise `BALL UNKNOWN`
  in the corner. **Ball debug** adds every other candidate with its label and
  score: `CANDIDATE`, `FIELD_LINE`, `STATIC`, `PLAYER_ATTACHED`, `OTHER_OBJECT`,
  `TRAJECTORY_FAIL`, `LOW_SCORE`, `SIZE`, `OUTSIDE_PITCH`, `JUMP`, `NO_ENTRY`,
  `UNCONFIRMED`.
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
NON-TARGET PERSON
Local Track: 37
Was: CAND
Hint: REFEREE_LIKE
Why: referee-like from the start
Global identity: none (ignored; the local box still counts for overlaps and the ball)
Team A similarity: 0.08
Team B similarity: 0.61
Appearance vs the officials' look / team A / team B: 0.84 / 0.16 / 0.52
Kit matches neither team: 74% of observations
Referee kit: 70%
Detector referee / goalkeeper votes: 74% / 0%
Team formation consistency: 0.31
Ball-following behaviour: 0.79
Referee movement behaviour: 0.68
Final: CENTRE REFEREE 0.83
Non-target confidence: 0.83

ROLE UPDATE IGNORED
Global ID: B-02
Local Track: 191
Stable role: PLAYER (team B)
Raw role: NON_TARGET 0.78 (referee-like)
Reason: an established team player (61 of 70 clean kit votes clearly team B): a referee-like reading never changes who they are
Decision: identity, team and role kept

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
- **Match identities** lists team A, team B and goalkeepers (results of earlier
  versions also list their referees). Hover for the role history; click an
  identity to highlight it and jump to its first observation.

Suggested checks on a real clip: nobody in the stands or on the bench gets a
label; the light-blue boundary sits on the white lines, not on the boards;
referees (including the assistants on the near touchline) are never a team
player and never get an identity (turn on debug to see them as `IGNORE`); a
player who runs past the referee keeps their ID; each goalkeeper shows as `GK-A` or `GK-B` and keeps it away
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
until the next keyframe. Who is drawn comes from the final registry: team
players and goalkeepers only. Ignored people (referees, staff) never appear,
and a withdrawn identity (a referee-like person promoted by mistake) is not
shown anywhere.

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

**Kinds of position.** With a finished `timeline` (section 3) every roster slot
is drawn from it: an observed position as a solid marker; an interpolated one
(a gap of at most a second between two observations) solid with a thin inner
ring; a predicted one translucent with a dashed outline; a prediction less
certain than 8 m faded with a `~`, and hidden unless **Full lineup (every
estimate)** is on; an unknown position is never drawn (the roster strip under
the board names the slots that are not on it, with their visibility: expected
in frame but not seen, off-screen, unknown). Only observed samples are
smoothed; estimates are drawn as the timeline gives them, and a returning
player never jumps because the estimate meets the observation. **Camera
footprint** draws the part of the pitch the camera sees on the current frame.
The debug overlay adds the sigma of an estimate. **Download timeline CSV**
exports, per slot and frame, the position, its kind, its sigma along and across
the pitch and the visibility, for analytics that must never mistake an estimate
for a measurement. Results without a timeline (earlier versions) are drawn from
their frames as before.

## 2D replay stabilization

**Why the markers bounced.** A player's pitch position is the foot pixel (the
bottom-centre of the detector box, `soccer/geometry.py` `foot_point`) through
the frame's homography. The box's bottom edge is about as noisy on every
camera, but what a pixel is worth on the pitch is not: a low amateur or Veo
camera sees the ground at a grazing angle, so a vertical pixel is worth several
times what it is on a broadcast camera, almost entirely along the camera's
line of sight. The markers therefore bounce towards and away from the camera,
nearly in one dimension. Most of that comes from short box-extent events: the
box loses the legs behind a nearer player, ends at an overlapping player's
head, or splits into an upper body and legs; the bottom edge jumps while the
top stays, and the player is thrown metres away from the camera for a fraction
of a second. The renderer's short smoothing window cannot remove either. The
calibration adds little high-frequency jitter: between anchors the final
homography follows the camera motion exactly, but at an anchor frame
(`registration.build_timeline`) it snaps to the anchor, and every player steps
together (the median player up to 0.7 m on the amateur run, 1.1 and 2.3 m at
Veo's two keyframes, 0.6 m on the broadcast run). A registration that
crossfaded between anchors removed the snaps but also changed a tracklet
decision (the registration feeds the tracking), so the registration is
unchanged and the snaps are spread in the stabilized stream only
(**Calibration snaps**, below).

**What the stabilizer does.** `soccer/trajectory.py` `stabilize()` is a pure
function of the finished result. Per internal identity (never the display
label), and per anonymous `?` track, it takes the frames the timeline counts as
observed, fits the run's own pixel noise of the box edges, turns it into a
measurement uncertainty through the homography's Jacobian (so the depth
direction keeps its larger error), repairs the foot of a box that lost its legs
or reaches into a nearer player from the box top and the track's running height
(flagged), and smooths each identity with a robust constant-velocity Kalman
filter and Rauch-Tung-Striebel smoother whose process noise rises only for a
plausible change of motion. Gaps up to the timeline's interpolation limit are
bridged (kind `I`); a jump no player can run splits the trajectory (flagged)
instead of being glided across. Identities are never mixed or averaged with
teammates. See the module docstring for the constants.

**Calibration snaps.** Before smoothing, the stabilizer finds the frames where
the registration steps against the camera motion (the step of a ground point
from the previous frame's homography, carried by the frame's motion row, to this
frame's) by more than 3 cm and three times the steps around it, and spreads each
step over a ramp instead (`step_events`, `spread_steps`): as long as the median
person on the frame (everyone within 5 m of the lines, whatever their role: a
correction never changes a ramp) needs to glide it at 0.2 m/s, 1 to 6 s, cut
only by a keyframe, a camera cut or a registration start; overlapping ramps add,
so no frame jumps. Whose anchor it is decides the direction. A user keyframe is
shown exactly as registered: the step arriving at it is spread over the frames
before it, the step leaving it (into the blend towards the next anchor) over the
frames after it. A re-registration matched to a keyframe's view (directly or
through a relay view: source `reregistered`, its root a keyframe; 10 frames of
the broadcast run, 6 of Veo's, none of the amateur run's, whose re-registrations
are rooted at the automatic midfield anchor) is trusted: its steps are spread
the same way, but other ramps may run across it. Any other automatic anchor
(paint, auto-midfield, a re-registration rooted at one) is one event with the
steps within 0.2 s of it, whose net step is spread before it, so an anchor that
jumps away and back disappears; a trusted anchor that does that (an outlier
against both neighbours) is treated the same way, and so are the automatic steps
right after a keyframe or trusted anchor, which go with its leaving step. A ramp
its room squeezes turns the other way, or is not spread at all (a snap rather
than a fast glide). The trust is a rule, not a measurement: a trusted frame no
ramp crosses is shown as registered, good or bad (on the broadcast run, against
the keyframes' clicked landmarks carried by the camera motion, 24.6 and 34.4 s
agree with their keyframes to 0.12-0.13 m, 25.6 and 35.4 s are 0.39 and 0.42 m
off). Over +-3 s around the broadcast run's four keyframes the displayed
registration is worse than the raw one by more than 5 cm for 1.1 s and better
for 8.4 s (mean landmark error 0.62 / 0.36 / 0.28 / 1.04 m raw, 0.60 / 0.26 /
0.22 / 0.93 m displayed); 0.8 s of the 1.1 s lie in the 2 s before the paint
anchors at 42.8 and 43.2 s, whose net step is spread before them. The ramps are
display corrections of the 2D replay only: the raw positions and the
registration are untouched.

**Where it goes.** `soccer/stabilized.py` builds `result.stabilized` on the
finished report (`pipeline.run`, once the summary lists the registration's
anchors, whose sources and roots the step spreading reads: the block is the one
built from the saved `result.json`), and `corrections.apply_result` builds it
again after a correction (see below). It is a light module (numpy and
the soccer modules), so the corrections and the tools build the block without
loading torch; `pipeline.stabilized_block` and `pipeline.attach_stabilized` are
the same functions under their older names. Per roster slot (the keys of
`timeline.ids`) and frame: the normalized `x` / `y`, the `kind` (`O` / `I` from
the stabilizer; `P` / `U`, and imputation's own `I`, from `imputation.estimate`
run on the stabilized observations, so a prediction starts where the stabilized
player was), the `sigma` along and across in metres and the observation weight
`w` (1 normal, under 1 down-weighted, 0 rejected); the anonymous tracks with
their team, position, kind and sigma; the stabilizer's `flags` (every repaired,
down-weighted or rejected observation and every split, with the raw and the
used position in metres), its fitted `noise` model, `summary` and `constants`.
Imputation gets each stabilized observation with the raw observation's sigma
at least, so a prediction's sigma grows as the timeline's does and turns `P` or
`U` on the same frames (before that, 33, 12 and 0 frames of the amateur, Veo
and broadcast runs of Oct 18 were `P` where the timeline says `U`; now none). On
an unconfirmed slot (an identity beyond the team's ten outfield slots) the
stabilizer bridges a gap of at most a second between two observations (`I`)
where the timeline says `U`: the board's raw drawing joined exactly those gaps
already (its one-second interpolation; the longest is 0.47 s on the amateur run
and 0.83 s on the broadcast run), so nothing new is drawn. A frame without a
trusted registration places nobody, as in the timeline.

**What the block guarantees.** It keeps the timeline's observations: a frame
the timeline counts as observed (a slot's `O`; an anonymous track's
observation, by the same rule) that the stabilizer leaves unplaced (no
position, not finite, or beyond the drawable pitch) is drawn as the raw
observation and counted in `summary.stabilizer.rawFrames` (`rawSlots` names a
slot the stabilizer did not return at all); there are none on the three Oct 18
runs. More than 1 % of the observations drawn raw, or stabilized observations a
median of more than 2 m from the raw ones (0.17, 0.29 and 0.10 m on the three
runs), and the stabilizer is taken to have failed. A stabilizer that fails in
one of these ways or raises, or a block that would not serialize, never fails
the run or the correction: the block is left out, `summary.stabilizer` is
`{status: failed, error, seconds}`, the console has the traceback, and the 2D
view draws the raw positions as before. A result with no registered frame gets
no block (`status: skipped`, with the reason). The same result gives the same
block. Otherwise `summary.stabilizer` says where the noise model came from
(`noise`: `fitted`, `kept`, `refitted`), the kinds over the roster slots, and
under `slots` the roster slots' weighted observations, their rejected and
down-weighted (weight under 0.5, as the flags) shares and how far the
stabilized observations are from the raw ones (median, p90, max); `allTracks`
is the stabilizer's own count over every track, the anonymous ones included,
with the flags by reason; then the time and the block's size. In GPU runs of
the amateur, Veo and broadcast regression windows (Oct 18, nothing else on the
machine; the stabilizer before its last two fixes, which change the block by
under 0.05 % in size), the block adds 7.8 %, 9.0 % and 3.9 % to
`result.json` (2.3, 1.1 and 1.4 MB: the slots' columns most of it, then the
anonymous tracks and the flags) and takes 3.4, 1.1 and 3.9 s of the runs' 303,
121 and 232 s (the stabilizer itself 1.7, 0.7 and 1.6 s), with at most about
90 MB of allocations while it runs; the raw output is identical to the
committed code's, timings aside. The velocity is not stored (nothing drew it):
leaving it out and writing whole weights as integers made the block 19 %, 12 %
and 29 % smaller.

**Corrections.** A correction rebuilds the block only for a run that has one, so
it never switches the 2D view between raw and stabilized: a run analysed before
the stabilizer (the three Oct 18 regression runs among them) has no block and is
drawn the old way, corrected or not, until it is analysed again. The stabilizer
fits its noise per run, so a correction that changes the observations (excluding
the unconfirmed A-11 of the amateur run removes 232 of them) would fit it again
and move people the correction did not touch (535 samples of the others by up to
0.017 m there). The rebuild therefore hands the stabilizer the original block's
model (`noise`): the pixel noise, both passes of the geometric height's fit
(`geometric.firstPass`) and the footage's edge-fit level (`footage`), and the
registration's snaps are measured at every person on the frame, whatever their
role; with them the other identities' stabilized observations stay exactly where
they were (eleven exclusions on the three regression runs of Oct 18, and twenty
corrections of every kind in the verification: none moved), and only
imputation's predictions for the corrected player's team change, as they do in
the timeline. Each of the three was found missing on these runs: without the
replayed passes handing a run its own model back moved 1072 Veo samples by up to
0.25 m, without the level excluding B-08 moved 36 amateur samples by up to 0.08
m, and with the snaps measured at the observations only excluding B-03 moved
8048 by up to 0.23 m. `summary.stabilizer.noise` says `kept`, or `refitted` for
a stabilizer without a `noise` parameter, which fits the model again (and moves
the others as above).

**Raw and stabilized stay apart.** `frames[].people[].pitch`, `pitchSigma` and
`result.timeline` are untouched and stay what everything else reads: the player
analytics, team shape, visibility, the tracklet passes, the server endpoints,
the tools, the video overlay and the **Timeline CSV**. Only the 2D match view
draws `result.stabilized`, when its `frameTimes` match the frames, joined by
straight lines between frames with no further smoothing; a result without the
block (earlier runs) is drawn the old way.

**Comparing.** With **Debug positions** on, **Positions** chooses what the board
draws: **Raw** (the drawing before stabilization), **Stabilized** (the
default), or **Both** (the raw positions as small outlined markers, each tied by
a thin line to the stabilized marker of the same person). `tools/jitter.py
RESULT_JSON` measures the markers' jitter from a saved result, raw against
stabilized, as the renderer draws them (steps, high-frequency residual,
acceleration, reversals, jumps; identified players and `?` markers apart); it
builds the block (`soccer/stabilized.py`) when the result has none.
`tools/trajectory_bench.py` measures the smoothers' accuracy against the
Metrica sample games seen through a simulated pole or broadcast camera with the
measured pixel noise and box-extent events (downloaded data: run it with
`python -I`). `tools/check_foot_truth.py RUN_DIR [--stabilized]` scores the
raw box's foot pixel, the stored pitch position and the stabilized position of
the marked players against hand-marked ground contacts
(`tools/annotations/foot_positions.json`, on the three regression runs),
separately for normal boxes and box-extent events.

## Correcting who someone is

Click a person's box (or their label) on the video to change who they are: **Team A
player**, **Team B player**, **Goalkeeper · team A / B**, **Referee**, **Not part of
this match** or **Left the match here** (substituted or sent off: the identity's roster
slot is closed from the current video time on, the observations before it are kept, and
a later run of the video never continues that player). **Who** chooses between a new identity and *the same person as* an
existing one of that kind; only identities never on the video at the same time are
offered (two people at once cannot be one person). **Save** relabels every frame of
that identity in the result, and the roster, counts, 2D match view and events follow
(`CORRECTED BY YOU`). People you set are marked ✎. The **Corrections** panel lists
every change for the video with **Remove**; the popover of a corrected person offers
**Undo**.

A correction is remembered for the video (`runs/<video>/corrections.json`):

- The run it was made on is rebuilt from the untouched originals (`result.raw.json`,
  `identities.raw.json`, copied on the first correction), so removing a correction
  restores the original files exactly. Removing a correction also removes later ones
  that built on it (for example a correction of the goalkeeper it created).
- **Continue identities from the previous run** starts from the corrected identities.
- Every later run of the video starts with the people you corrected as *manual
  identities*: their saved appearance with the role and team you gave them (the
  log starts with `REMEMBERED FROM YOUR CORRECTIONS`). A track is recognised as one
  of them (`RECOGNISED: IDENTIFIED BY YOU`) when its latest crops look clearly like
  that person (appearance at least 0.7) and clearly more like them than like anyone
  the tracker knows (by 0.15), on two checks a second apart (three when the track
  already has an identity, since a track that just crossed someone briefly still
  carries that person's crops). Earlier frames of the track are relabelled. A
  manual identity's role and team are never changed by automatic evidence, and a
  goalkeeper you set keeps the team's slot. A track carrying someone you set keeps
  being compared with their appearance as you corrected it; three checks clearly
  unlike them (below 0.45) release the identity. Someone you marked as not part of
  the match is recognised the same way and shown as `OUT`
  (`EXCLUDED_BY_YOU`).

A person without a global identity yet (only a local track, `B-?` or `UNK-12`) has no
saved appearance: correcting them changes this result only, and the popover says
so. Team letters are only meaningful with the kit model they were given under, so a
run that does not continue identities starts from that run's kit model when a
remembered person has a team. Corrections can be made when no run is processing,
on the latest result.

## Output

Runs are saved under `runs/<video-content-hash>/<run>/`: `config.json`,
`result.json`, `identities.json` (registry and team kits, used to continue
identities), `registration.json` (what the offline registration used: the
per-frame camera-motion log, every anchor with its `root`, the keyframes with
the frame where each applied; `tools/holdout.py` rebuilds the registration
from it), `events.log` (the offline passes' events appended after the online
ones) and the latest `preview.jpg`. The viewer downloads `result.json`:

`result.json` has `version: 3`:

- `frames[]`: `time`, `frame`, `people[]`, `rejected[]`, `pitch` (every 0.2 s:
  polygon, detected lines, goal-end hints, edge sources incl. `predicted` /
  `off-screen` / `horizon`, `confidence`, `level`, `mode`, `status`,
  `evidence`, `vanishing`, `curvature`), `pitchModel` (on the same frames: the
  live model's `temporal`, `observability`, `details`, `state`, `source`,
  `cameraMotion`, the centre `circle` seen and the last automatic `midfield`
  attempt), `calibration` (the final registration of the frame after the
  offline passes: `confidence`, `temporal`, `observability`, `source` —
  `landmarks`, `reregistered`, `paint`, `auto-midfield`, `carried`,
  `interpolated`, `none` — `direction`, `anchor` frame, `anchorAge`,
  `disagreement` in metres over the visible pitch, `H` the frame's final
  homography, world metres to pixels, as nine numbers), `mapping`
  (`imageTracks`, `mapped`), `pitchMode` (`automatic`, `robust`, `assist`),
  `pitchConfidence`, `pitchCoordinates` (`calibration`, `predicted` or `null`),
  `boundary`, `cameraReliable`, `cut`, `segment`, `calibrated`, `ball`,
  `ballCandidates`.
- `people[]`: `id` (global identity or `null`), `display` (`A-07`, `GK-A`,
  `A-?`, `UNK-3`, `IGNORE`), `track` (local), `team` (`A`, `B` or `null`), `role`
  (`PLAYER`, `GOALKEEPER`, `NON_TARGET`, `UNKNOWN`), `label` (combined:
  `PLAYER_TEAM_A`, `GOALKEEPER_TEAM_B`, `NON_TARGET`, `IDENTITY_UNCERTAIN`,
  `CANDIDATE`, …), `state` (`candidate`, `uncertain`, `confirmed`, `nontarget`,
  `unknown`, `rejected`), `hint` for ignored people (`REFEREE_LIKE`,
  `SIDELINE_OFFICIAL`), `identityConfidence`, `roleConfidence`, `box` (normalized screen
  coordinates), optional `pitch` with `pitchConfidence`, `pitchSource`,
  `pitchSigma` (metres along and across the pitch) and `pitchClipped` when the
  box is cut by the bottom of the frame (or `pitchStatus: unresolved` when no
  registration of confidence 0.2 or more covers the frame or the feet do not
  land on the pitch, `horizon` when the foot is above the horizon), `zone`,
  `cls` (detector class), `evidence`
  (`observed`, `reidentified`, `relabelled` after a role change, `connected
  offline`, `assigned offline`), `reason` while unresolved, and `why` (the role
  evidence in one line) on evidence ticks.
- `ball`: `state` (`TRACKED`, `MISSING`, `UNKNOWN`), `phase` (`SEARCHING`,
  `LOCKED`, `OCCLUDED`, `RECOVERING`), `confidence`, and unless unknown
  `track` (`BALL-n`), `observed`, `box`, `center`, `predicted` (next frame),
  `velocity`, `speed` (m/s), `direction`, `acceleration`, `age`, `lastSeen`,
  `lastConfident`, `source` (`DETECTED`, `PREDICTED`, `INTERPOLATED` for a
  short gap bridged offline from both sides, `bridged: true`), optional `pitch`
  with `pitchConfidence`, `detector` and `score`, and `missingFrames` /
  `missingFor` / `nearTrack` while hidden. `ballCandidates[]`: every
  candidate's `box`, `det`, `score`, `status` (`ball`, `candidate`, `rejected`),
  `reason` / `text`, and `scores` (detector, size, shape, isolation, field, line,
  attachment, motion, static, trajectory, distance, penalty).
- `match.players`, `match.goalkeepers`, `match.referees`, `match.retired`: the
  registry, separated (referees with `official`, every identity with
  `roleEvidence`; a player marked as having left the match carries `left`
  with the time). **Tactical analysis should use only people with an `id`
  whose role is `PLAYER` or `GOALKEEPER`, and treat low `identityConfidence`
  values with care.** Referees are never players.
- `roster`: the active roster (section 3): `expected`, `counts`, `overCap`,
  `slots[]` (`id`, `display`, `team`, `role`, `status`, `firstSeen`,
  `lastSeen`, `observedFrames`, `tracks`, `lastPitch`, `exitEdge`, `left`),
  `anonymous` (fragments and frames of tracked people without an identity),
  `statuses`, `assignment`, `mergeProposals[]` (`earlier`, `later`, `team`,
  `appearance`, `gap`, `speed` or `imageSpatial`, `continuity`, `plausible`;
  never applied). Per frame: `view` (the visible pitch as a normalized
  polygon, or null) and `visibility` (`{slot id: C | S | U}` for the slots not
  observed on the frame). `summary.roster` and `summary.visibility` summarise
  them.
- `timeline`: `frameTimes[]` and, per roster slot, columns over the frames:
  `x`, `y` (normalized pitch position or null), `kind` (`O`, `I`, `P`, `U`),
  `sigma` (`[along, across]` metres or null); `method`, `constants`, `kinds`
  (shares). This is the analytics export: every coordinate says whether it was
  observed, interpolated or predicted.
- `stabilized` (when the stabilizer ran; see "2D replay stabilization"): what
  the 2D match view draws. `version`, `method`, `frameTimes[]` (the timeline's)
  and, per roster slot, the timeline's columns plus `w` (the observation's
  weight, null where the stabilizer placed nothing); `anonymous` (per local
  track: `team`, `x`, `y`, `kind`, `sigma`), `flags[]`, `noise`, `summary`,
  `constants`. `summary.stabilizer` says what it did, or why it is missing
  (`failed` with the error, `skipped` with the reason). Not an analytics input:
  the analyses read `timeline` and the frames.
- `corrections`: the corrections applied to this result (who, to what, whether it
  is remembered). Corrected people carry `manual: true` and `correction` on every
  frame, and `summary.remembered` lists the identities a run started with from your
  corrections.
- `tactical`: what the 2D match view needs beyond the frames: the coordinate
  convention, how many frames carry pitch coordinates (`calibratedFrames`,
  `calibratedShare`, `coordinateSources`), the `pitchSize` used (standard or
  learned from the landmarks) and `goalEnds` (which team defends the left /
  right goal line, when the tracking settled it).
- `pitchIssues[]`: sections whose pitch confidence stayed low without a model
  (`start`, `end`, the representative frame `at`, `suggested` landmarks with
  predicted positions where known, `reason`); `summary.pitch` has the mean
  confidence, the share of analysed frames per level and of frames per mode,
  the pitch size, the frames filled by the offline pass, far-tile passes, the
  number of sections needing confirmation and `registration` (calibrated and
  confident shares, mean observability, every anchor with its source and
  observability parts and `root`, frames per source, `mappingLossFrames`,
  `medianMappedShare`, ball frames bridged, `lensCurvature`,
  `vanishingPointShare`, mapping warnings, `registrations` — direct
  re-registrations tried, adopted, refused and chain root changes — and
  `lens`). `summary.timing` has the milliseconds per frame of each online
  stage (detector passes, far tiles, ball tiles, ball crops, tracking), the
  offline passes in seconds and the pre-roll/post-roll frames.
  `summary.detection` counts the adaptive detection triggers (far-tile passes
  and detections, far starts, crop passes and detections, far-edge
  continuations), the ball tile scans run (`ballTileScans`) and the detector
  batch used (`detectBatch`). `summary.ball.localSearches` counts the
  full-resolution crops searched around the predicted ball.
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

## Evaluation tools

`tools/run_segment.py --video <upload key or path> --start S --seconds N --out DIR`
runs a segment into a scratch directory with the video's saved settings
(`runs/<key>/settings.json`), never writing under `runs/`; `--drop-keyframe T`
leaves a keyframe out, `--batch jobs.json` runs several segments in one
process (the detectors and the embedder stay loaded from one job to the next,
as in the server; `--fresh-models` loads them anew per job), `--detect-batch K`
and `--ball-scan-every S` set the detector batch and the safeguard scan
interval. `tools/exact_diff.py A/result.json B/result.json` compares two runs
of the same segment frame by frame (people, pitch positions, ball, mapping,
roster, tracks, labels, events). `tools/check_ball_decode.py RUN_DIR` runs the
vectorized offline ball decode and the frozen copy of the old one on a run's
frames and says whether the paths are identical. `tools/ball_clips.py
result.json` prints the ball and identity records of the broadcast regression
clips. `tools/verify_speed_pass_3.cmd` runs the whole speed-pass-3 protocol
(see PROGRESS.md). `tools/compare.py A [B]` prints the metrics of a run or of two runs
side by side (people tracked and drawn per frame, identified share by box
height, identity, pitch, ball, timing) and how much B agrees with A (people
matched by box overlap, label agreement exact and under the best id mapping,
pitch position distance in metres). `tools/holdout.py RUN_DIR` rebuilds the
registration without each calibration keyframe (and without the anchors rooted
at it) and scores the rebuilt homography on that keyframe's landmarks, in
pixels and metres; it also reports the pan stability (ground jumps beyond the
measured camera motion) and the disagreement share. `tools/counts.py RUN_DIR`
compares the people tracked at the hand-counted frames of
`tools/annotations/counts.json` (people on the playing field, counted by
inspection with a stated uncertainty) with the counts: a recall proxy by frame
and overall, the small people among them, over-counts. `tools/diagnose.py`,
`tools/check_ball_truth.py` and `tools/check_pitch_truth.py` (the hand-marked
ball and pitch references of the amateur window) are described under
"Amateur footage" → "Measuring". `tools/possession_bench.py` scores the
possession engine on the Metrica truth ("Possession" → "Reproduce"). `PROGRESS.md` records
the runs and numbers of every phase.

## Tests

From the repository root, after setup:

```bat
prototypes\gpu-tracker\test.cmd
```

`tests/` covers pitch detection on synthetic frames and perspective broadcast
renders (touchlines vs advertising boards, run-off, stands, green seats behind
boards, goal lines vs penalty-box fronts, temporal stability, zoomed views,
crowd close-ups), the pitch model (`test_pitch_model.py`: exact homographies
from landmarks, painted lines pulling a drifted prior back, size learned only
when the landmarks pin it down, paint matched only to the marking it lies on,
HIGH confidence with painted touchlines and LOW on grass without paint,
carrying and decay, refinement, people outside the prediction, keyframes
learning the size, backwards propagation up to a cut, the size rules behind
scale conditions, the provisional boundary, broadcast frames staying automatic
with the detector's own region, faded lines asking for confirmation once per
section, far-tile detections counting only when confirmed, suggested
landmarks), foot-point filtering and audience/size rejection, assistant
referees at the touchline, part descriptors and galleries, team and role
decisions, camera motion and cuts, landmark calibration, the ball tracker (the
rendered, panning sequence of `ball_eval.py`: a pass, a dribble, a long shot
across the halfway line and the centre circle, players overlapping the ball, a
wrist tape detected with a higher confidence than the ball, white boots, the
centre and penalty spots, debris on the ball's path; occlusions, a long hide at
the tape-wearer's feet, an undetected spell with re-acquisition of the same
ball, the ball leaving the scene, confident static spots, a calibrated pitch),
and the identity manager (promotion, exits and returns,
deferral of look-alikes, referees ignored as NON_TARGET, a referee first
labelled as a team player and withdrawn, returning referees ignored again,
assistant referees, goalkeepers found from position and kit, role locking,
touchline candidates, team limits, crossings, goalkeeper team, continuation
across runs). `test_identity_encounters.py` covers a player running beside the
referee, a player lost and back on a new track, a track handed to the referee,
two tracks swapped with an ignored person, the officials' look never deciding
alone, and the offline pass (an ignored fragment reconnected to the player, or
never on position alone). `test_integration.py` runs rendered frames
through the real BoT-SORT and OSNet with spectators, a coach, a referee, a pan
and a player who leaves and returns. `test_registration.py` covers the low
sideline camera: observability ranking (spread landmarks vs one line,
touchlines alone, the centre circle, a flat Jacobian direction), the
temporal/geometric split and refused anchors, the vanishing-point guard (no
edge cue, no rejection, no HIGH confidence at the apex; a goal line at the
horizon), a fast pan over flat grass measured coarsely rather than taken for a
cut, the offline timeline (carried both ways up to a cut, blended anchors with
their disagreement, coarse motion fading faster, tracks remapped or left
unresolved, mapping-loss warnings), ball gap bridging, and the automatic
midfield anchor from exact and from detected paint on a rendered low view.
`test_team_shape.py` covers the team shape analytics on synthetic matches
whose hand-placed layouts differ from the formation templates: width and depth
(the max - min definitions, goalkeepers, referees and unconfirmed identities
left out, duplicates, glitches next to gaps and track ends, coverage and
estimated-mode statuses, players who left), the defending direction (tracking,
one named end, evidence alone and its tail, a side swap inside an evidence
window), the defensive line (median and deepest, the right goal, a back three
with a stepped-up defender, a fullback's overlap, a set piece, a sustained swap
into the back line, a coach's label on an unclear period, the depth fallback
across a side swap) and the formations (clear layouts, roles, orientation,
missing players, swaps, runs, sustained changes, gaps, transitions, ambiguous
shapes, half time, nine-player and changing lineups, identity confidence), the
settings and ranges (malformed input, oversized numbers, ranges outside or
reversed), and the cases a second review reproduced (flickering and chained
duplicates, a departed player's identity on his substitute, split-identity
estimates, short periods and short labels, more players than the lineup).
These are synthetic checks of the logic, not accuracy measurements on real
footage.

Possession is covered by `test_possession.py` (the engine on synthetic results: a
pass between teammates keeps possession, an interception changes it once, a
deflection and a half-second duel flicker do not, a ball out of play is dead until
the other team's restart, a missing ball is unknown, the carrier is near on both
metrics, the nearTrack check, a person without a team and a goalkeeper without a
team, a camera cut, the small-scene abstention, an empty result, stable change ids,
a team correction moving the carrier's team, the record's contract as plain JSON;
a change resting on an observed control, an image-only spell starting none, an
excursion reached by a jump or with another ball in play being no ball out of play,
a stoppage with the ball in view, who won the ball, back-and-forth pairs, player
speeds through the camera motion; results of 1 to 30 frames, a ball held in play
being no stoppage while a placed ball stays dead, a keeper's hands from the catch, a
ball knocked past a defender, a result without `ball.origin`, the first-pass check
with the image rows doubled, a ball placed after an unseen stretch, a replay between
two cuts, an identity on another team's player, high / uncertain changes),
`test_possession_server.py`
(the route: the record served byte for byte, cached per result version and computed
again after a correction, one computation at a time, 404 / 409 / 503 and the local
checks), `test_possession_panel.py` (`possession.js` in node with the real
`tactical.js` board: the live line, the shares, the strip and its colours, the
markers, their tooltips and seeking, the won frame, the rings, every change shown by
default with the uncertain ones hollow (possession-4's `uncertain`, possession-3's
`low`) and hidden when the option is unticked, labels after a correction, the
one-line unavailable and failed states, the page's fence) and
`test_possession_truth.py` (the Metrica truth tool on both of Metrica's formats: a
synthetic game written as CSV and as EPTS gives the same truth byte for byte, the
EPTS reader's substitutions, identity swap and refusals; with the Metrica data on
the machine, `PITCHIQ_METRICA_TRACKING`, `PITCHIQ_METRICA_EVENTS` and
`PITCHIQ_METRICA_GAME3`, also games 1-2's truth and simulation byte for byte as
before game 3 was added and game 3's truth as built).

The amateur-footage work ("Amateur footage" above) is covered by `test_midfield.py` (the pinhole midfield
solve on rendered pole and broadcast views, the mirror camera, capped anchors), `test_tracker.py` (the
small-scene gate and its latch, tiny far-tile rows, buffered boxes), `test_pipeline_run.py` (gates closed on
broadcast-like input, a low-resolution run), `test_prefetch.py` (the near pass only fills in tall people the
other passes missed), `test_ball_proposals.py`, `test_ball_small_scene.py`, `test_ball_path_small.py` and
`test_ball_cleanup.py` (the classical proposals, the hidden-ball gate at depth, the provisional lock, the
still-lock audit, a placed ball kept for its set piece), `test_identity_amateur.py` (the kit-change re-birth,
size-aware votes, kit-consistent galleries) and `test_identity_fragments.py`
(small fragments on position, an identity ruled out by an impossible side); each checks that the shipped
behaviour is unchanged with its gate closed.

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
- Pitch coordinates need calibration keyframes; the pitch model carries them
  through the video (forwards online, backwards in the offline pass) but
  cannot start without one, and the 105 × 68 m size is assumed unless the
  landmarks of your keyframes pin a different size down. The 2D match view
  therefore shows people only while the model is trusted and only those the
  camera sees; off-screen players are not placed.
- A homography describes a pinhole camera. Wide-angle action cameras (Veo and
  the like) bend straight lines, so the model's edges can be off by a few
  pixels at the frame borders, the frame-to-frame carry drifts over long pans,
  and the painted-line refinement can only correct what it can match (lines
  more than 1.5% of the frame width from the prediction are left alone). Add a
  keyframe where the overlay visibly disagrees with the lines. The recorded
  `curvature` says how bent the longest line is and `registration.lens` the
  division-model coefficient that would straighten the markings; the
  undistortion is not applied yet.
- A low sideline camera sees two converging touchlines, no corner and often
  no goal line. The automatic midfield anchor needs the centre circle, the
  halfway line and one touchline in view with legible paint; where the circle
  is too faint or too far, mark landmarks on a clear frame (centre spot, the
  circle's four points, the halfway line's ends): the offline passes carry
  that anchor both ways. Until then the image tracks and identities are kept
  with `pitchStatus: unresolved` and the 2D view shows nobody rather than
  guesses.
- Substitutions are not recognised automatically: a player who goes off stays
  `MISSING`, and an incoming substitute becomes a new identity only after the
  sanity checks above (a warning is logged).
- Role decisions are heuristic and need about 5–10 s of evidence; a referee whose
  kit, shorts and detector class all look like a team player's can stay a
  player until the evidence changes (then the identity is withdrawn, unless it
  had become an established team player, which a role reading never changes). Goalkeeper team inference stays unknown
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
- Possession is an estimate from the 2D tracking ("Possession" above): which team
  has the ball is right 95-96% of the time where it names one on held-out data, but
  the changes of possession reach an F1 of only .59-.69 on held-out Metrica games, on
  a fresh broadcast segment 1 of the 2 reported changes was right and half of the
  dead time was missed, and amateur and Veo footage get no possession at all.

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
