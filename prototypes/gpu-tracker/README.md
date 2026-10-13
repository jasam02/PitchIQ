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
| `id`, `longId` | `A-07` / `TeamA_Player_07`, `GK-1` / `Goalkeeper_01`: the persistent identity (players and goalkeepers only) |
| `display` | what the overlay shows: `A-07`, `GK-A` / `GK-B` (goalkeepers by team) |
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

**Close encounters and contaminated crops.** Local track id, global identity,
team and role are four different things, and a local tracker interruption must
not erase the real person. Every sample carries the IoU of its box with the
nearest other person's box (`occlusion`, `occludedWith`); above 0.15 the crop is
contaminated and updates nothing: no kit vote, no detector vote (the position
cues still count), no appearance gallery, and the track's "first clean crops"
are left alone. A crop joins an identity's gallery only when it is clean, of
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
`TRACKLET CONNECT` events give the scores. There are no referee identities to
connect anything to. The 2D match view reads these finalized labels, so a
player who was B-02 → B-? → ignored in the causal pass stays B-02 throughout.
`summary.identity` reports id switches, mixed tracklets, track switches found
online, offline splits, confirmed-player tracks that continued as an ignored
person (and how many no track switch or release explains: the number that
should stay at zero), referee identities (always 0), ignored tracks and frames
by hint, withdrawn identities, ignored / deferred role updates, close
encounters with ignored people, how many players came through them and were
re-identified after them, new player identities, teams over their count,
fragments connected (ignored fragments included) and frames restored to
players.

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
of the other). A proposal is listed for your review and logged as MERGE
PROPOSED; nothing is merged automatically until proposals have been scored
against corrections on enough videos. On every frame with a registration,
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

There is no automatic metric calibration (no field keypoint model ships with
this prototype), so the pitch model is anchored by your landmarks. Pause on a
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
required velocity, previous trajectory confidence, detector score, player
attachment and relative motion, `ACCEPTED` / `REJECTED (JUMP | NO_ENTRY |
UNCONFIRMED)`). The lock's "seen elsewhere while the ball was tracked" flags
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
hidden 0.42 s`, or `BALL RECOVERING · candidate paths: 4`.

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
an interpolated position. A candidate glued to the same spot on one player's
lower body loses ball-likeness (0.55; 0.3 after a second), and one the causal
pass saw while it confidently tracked the ball elsewhere loses half. The
effect: a boot that the first pass took for the ball while the real ball was
hidden is a short run that connects to nothing and never pays for its start;
the decoded path leaves those frames NOT VISIBLE, and a later sighting of the
real ball links back to the earlier ones. The result's per-frame `ball` is this
trajectory (`phase: PATH`, one identity `BALL-1`, `bridged` on interpolated
frames); the causal pass stays as `ball.firstPass`; `BALL PATH` events explain
every first-pass run the final path leaves out (the speed reaching it would
have needed). The 2D match view and the ball metrics use the final trajectory.
`ball_cleanup.py <result.json>` re-decodes an existing result without the
detector (camera motion from `frames[].motion`, an `.npz`, or re-estimated from
the video).

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
  continuations). `summary.ball.localSearches` counts the full-resolution crops
  searched around the predicted ball.
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
process. `tools/compare.py A [B]` prints the metrics of a run or of two runs
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
and overall, the small people among them, over-counts. `PROGRESS.md` records
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
These are synthetic checks of the logic, not accuracy measurements on real
footage.

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
