# Soccer identity tracking

Soccer identity tracking is the default automatic tracking mode in the match workspace (`/tracking/[videoId]`). It turns on with the **Soccer identity tracking (recommended)** switch under *Track automatically*, which only appears while *Use AI detection while tracking* is on. It follows the people who take part in the match, with one stable identity per roster slot, and it ignores everyone else the detector finds.

The rules behind it:

- **Do not track every person the detector sees.** People in the stands, on the bench, in the technical area or behind the boards are rejected before tracking.
- **A new tracker ID is not a new player.** Local track IDs change all the time (occlusions, exits, camera cuts). Global identities are roster slots (`A1`…`A11`, `B1`…`B11`, substitutes, `R1`…`R3` officials) and are never deleted.
- **Never guess.** A wrong merge is worse than an identity that stays unknown for a while. When the evidence is ambiguous the decision waits, and the track stays *identity uncertain* and is not saved.

The old automatic path (persistent tracker plus player recovery) is still available. Turn the switch off to use it.

## Pipeline

```
 video frame (every 0.2 s of video; ball every 0.1 s)
   │
   ▼
 person + ball detection ............ lib/player-detector.ts  scanFrame(pitchOnly:false)   YOLOX-Tiny, in the browser
   │  unfiltered detections + the RGBA frame they came from
   ▼
 pitch analysis ..................... lib/soccer/pitch.ts      grass hull, clipped by detected boundary lines
   │  PitchModel {polygon, lines, reliable}; lines carried with camera motion for ~2 s
   ▼
 field filter + size check .......... lib/soccer/relevance.ts  inside / boundary / outside; audience, bench, size
   │  accepted ScoredDetections            rejected → debug overlay (red, with reason)
   ▼
 local multi-object tracker ......... lib/soccer/local-tracker.ts (wraps persistent-tracker stepTracks)
   │  short-lived local tracks L1, L2, …  (camera cut → all local tracks end)
   ▼
 candidate evidence ................. hits, zone history, kit votes, descriptors, positions
   │
   ├─► role + team classification ... lib/soccer/classify.ts   temporal votes: player A/B, goalkeeper, referee
   ├─► appearance gallery + re-ID ... lib/soccer/appearance.ts part descriptors (jersey, shorts, socks, layout)
   └─► pitch coordinates ............ lib/soccer/homography.ts only when field marks give ≥4 usable points
   │
   ▼
 GLOBAL IDENTITY MANAGER ............ lib/soccer/identity.ts   promotion, re-identification, swap guard, sanity
   │
   ▼
 persistent observations ............ lib/soccer/pipeline.ts   SoccerTracker.step → points (playerId, track, conf, pitch)
   │
   ▼
 checkpoint every 5 s of video ...... app/tracking/workspace.tsx  points + identities + trackingLog → /api/tracking
```

Workspace integration (`runSoccer` in `app/tracking/workspace.tsx`):

1. The referee and both assistants are added to the roster as `R1`–`R3` (`ensureOfficials`) and saved with the first checkpoint.
2. The tracker is seeded from:
   - confirmed boxes on the start frame (anchors),
   - saved identities (`doc.identities`) with their appearance galleries,
   - each player's most recent confirmed labels before the start time. The workspace seeks to those frames and calls `rememberLabel`.
3. Each detection step passes the unfiltered detections, the camera motion since the previous detection step and the roster for that time. The camera motion is the chain of the 0.1 s estimates (so fast pans stay measurable); a cut at any 0.1 s step counts as a cut. Substitutions are resolved with `isActive`. The step also passes your confirmed boxes in its own time window, which always override automatic identities, plus the image→pitch homography when field marks exist.
4. Observations are saved only for roster slots that are active at the observation time. They are never saved on top of one of your confirmed boxes; confirmed labels and reviewed observations after the start time are kept, other experimental points after the start time are replaced. Each saved point carries the local track number (`track`), the identity confidence (`conf`) and pitch coordinates (`pitch`) when available. Re-identified tracks also get their earlier observations saved retroactively, marked `reidentified`.
5. Norfair (browser worker) smooths only *coasting* boxes, keyed by player ID. Its filter restarts when an identity is re-bound.
6. Camera cuts do not stop the pass. Local tracks end, identities become missing or off-screen, and re-identification runs on the new view. Losing the ball does not stop the pass either: it is logged, and the ball restarts from exactly one confident ball detection inside the pitch polygon.
7. You can start a pass with no labels at all; it then runs fully automatically.

## Statuses

Local track states (debug overlay):

| State | Meaning | Saved? |
|---|---|---|
| `CANDIDATE` (white, dashed) | Collecting evidence: hits, inside-pitch ratio, kit votes | No |
| `PLAYER` / `GOALKEEPER` / `REFEREE` | Confirmed and bound to a global identity | Yes |
| `IDENTITY UNCERTAIN` (grey) | A real participant whose identity is still being decided; observations are buffered | Only after it resolves |
| `UNKNOWN` (grey) | A participant that could not be classified after a long time | No |
| Rejected (red, dashed) | `REJECTED: OUTSIDE PITCH`, `REJECTED: AUDIENCE`, `REJECTED: SIZE`, `REJECTED: LOW PLAYER CONFIDENCE` | No |

Global identity statuses (Debug tab → *Global identities*, saved in `doc.identities`):

| Status | Meaning |
|---|---|
| `ACTIVE` | Bound to a visible local track |
| `MISSING` | Lost while inside the view (occlusion, detector miss) |
| `OFF SCREEN` | Last seen touching an image edge |
| `SUBSTITUTED` | Roster slot inactive at this time |
| `UNKNOWN` | Restored without enough information |

Identities are never deleted when a player disappears. A player who returns is matched against the missing and off-screen identities of the same team and role. The match combines appearance, jersey/kit, team consistency (a hard gate), spatial plausibility, exit/entry movement and time missing. A reconnection needs several scored steps, a minimum score (`reidMin`) and a clear margin over the second-best identity (`reidMargin`). For example, 45/48/51 % is deferred; a later 93/41/38 % reconnects.

## Debug mode (how to test)

1. Open a match and pause on a frame. Turn on **Debug overlay**, either below the video or in the **Debug** tab.
2. Click **Analyze this frame** in the Debug tab. This runs a stateless explanation with `explainFrame` and does not change any identity. It shows:
   - the detected playable area (green fill),
   - the grass hull (faint dashed),
   - boundary lines (white when observed, yellow dashed when carried by camera motion),
   - the people kept as possible participants,
   - red dashed boxes with the reason each rejected person was dropped.

   The status line summarises counts by reason and pitch reliability. After a tracking pass, the explanation uses that pass's team model.
3. Adjust **Touchline tolerance** (`boundaryMargin`, as a fraction of player height) if real players near a line are rejected (raise it) or bench staff are accepted (lower it). Analyze again to compare. The value is used by the next pass.
4. Start a pass with the overlay on. Every tracked person is labelled live, for example `A-07 · Track 91 · TEAM A · PLAYER · Identity 94%`. Colours: team colours for players, magenta for goalkeepers, orange on black for referees, grey for unknown or uncertain, white dashed for candidates.
5. After the pass, scrub or step through processed frames. The overlay replays the stored debug frame for that time. History is kept in memory only, for up to 3,000 detection steps, and is lost on reload.
6. Review **Identity events** in the Debug tab, newest first. Filters cover re-identifications, rejected or deferred matches, sanity and crossing warnings, and promotions, roles and labels. Click an event to jump to its time. A reconnection reads:

```
RE-ID EVENT
Local Track: 91
Matched Global Player: TeamA_Player_07
Appearance similarity: 94%
Team match: YES
Jersey similarity: 0.92
Spatial plausibility: 0.87
Time missing: 11.2 seconds
Final identity confidence: 93%
```

Suggested checks on a real clip:

- People in the stands and on the bench never get a label.
- Referees appear as `REF-n`, not as players.
- A player who leaves the frame and comes back keeps their ID (look for a `RE-ID EVENT`).
- Two teammates who cross keep their IDs, or a crossing warning asks you to check them.
- No team ever shows more than 11 identities. If it would, a sanity warning appears instead.

## Data model (`lib/tracking.ts`, still `version: 1`, all additions optional)

- `players[].team` adds `'ref'` for match officials: at most 4, not counted against the 11-starter limit. `players[].role` is optional: `player | goalkeeper | referee`, and `referee` is only allowed for officials. Set it in the roster editor (*Role*) or let the tracker infer goalkeepers.
- `points[]` adds optional `track` (local track number), `conf` (identity confidence 0–1) and `pitch` (`{x,y}`, normalized pitch coordinates).
- `identities` (≤ 40): saved global identities with status, role, team, confidences, last box and position, velocity, and an appearance gallery of up to 6 descriptors × 80 numbers.
- `trackingLog` (≤ 300 entries, message ≤ 240 characters): identity events with scores. This is the first thing trimmed when a checkpoint approaches the saved-clip size limit.
- Helpers:
  - `splitParticipants(doc)` returns `{players, goalkeepers, referees}`. Tactical analysis must use `players` (and `goalkeepers` where relevant) and never referees.
  - `ensureOfficials(doc)` adds `R1`–`R3` (idempotent).
  - `soccerRoster(doc, time)` builds the roster slots for the tracker.
  - `cleanIdentities` and `appendTrackingLog` bound and sanitize tracker output before saving.

Documents saved before this change still parse unchanged.

## Configuration

`SoccerOptions` (`lib/soccer/pipeline.ts`, `defaultSoccerOptions`):

| Option | Default | Effect |
|---|---|---|
| `field.enabled` | `true` | Turns the pitch filter off entirely; every zone then counts as `inside` |
| `field.boundaryMargin` | `0.35` | Tolerance beyond a boundary, as a fraction of box height (UI slider) |
| `field.minMargin` / `maxMargin` | `0.006` / `0.03` | Clamp on that tolerance, in normalized frame-height units |
| `maxPerTeam` | `11` | Sanity reference: no new identity is created beyond it; a sanity event is logged instead |
| `minHits` | `5` | Detection steps (1 s) before a candidate can be promoted |
| `boundaryHits` | `10` | Stronger requirement for candidates first seen on the boundary |
| `reidMin` | `0.72` | Minimum accumulated score to reconnect a returning player |
| `reidMargin` | `0.15` | Required lead over the second-best identity |
| `gallerySize` | `5` | Appearance samples kept per identity |

The workspace currently exposes only the touchline tolerance. Other values change in code.

## Honest limits

- **Same-kit teammates.** Colour descriptors cannot reliably tell teammates in the same kit apart. Re-identification between them depends on spatial and temporal evidence and roster elimination. When that evidence is missing (long absences, camera cuts), the identity stays uncertain instead of being guessed. Expect more `IDENTITY UNCERTAIN` periods than wrong labels. Confirm a box to resolve one immediately. There is no jersey-number OCR or learned re-ID embedding yet; the appearance module keeps a clean seam for adding one.
- **Pitch coordinates need field marks.** There is no automatic metric calibration. `pitch` coordinates are saved only when at least four usable, non-collinear corner or line-intersection points are available for that time from the Field tab marks, carried by the camera chain. Otherwise spatial reasoning uses camera-compensated image coordinates, with metres approximated from player height.
- **Automatic team kits need a wide enough view.** Without labels, the two kits are found only when at least three players of each kit are detected on the same frame (and the smaller kit is at least a quarter of the on-pitch people). In tight views, or when the detector misses most of one team, nobody is promoted until a wider view appears. Confirming one player of each team on any frame fixes the kits immediately.
- **Pitch detection is heuristic.** It relies on grass colour and white boundary lines. Close-ups, crowd shots, unusual turf colours or heavy shadows make the pitch *unreliable*. On those frames nothing new is promoted, and real players who first appear there are only picked up later.
- **Officials.** Referees are recognised by an outlier kit seen on the pitch. Assistant referees running the touchline need an established referee kit before promotion. A referee kit that looks like one team's kit can be misclassified; correct it with a confirmed box on `R1`–`R3`.
- **Goalkeepers.** Goalkeepers are inferred from a distinct kit seen repeatedly near a goal. Set *Role → Goalkeeper* in the roster to make this explicit.
- **Saved-clip capacity is unchanged.** About 3,400 saved points per match record; each point is now slightly larger. With ~25 identities this is roughly 25–30 seconds of footage per record. The pass pauses before the limit. Export the JSON before clearing experimental passes. Debug replay history is in memory only.
- **Speed.** Processing runs in the browser tab and is slower than real time. Keep the tab open.
- **Output is experimental.** Saved observations stay experimental (`reviewed:false`) until you confirm them. Tactical analysis should use `splitParticipants` and treat low `conf` values with care.

## Tests

- `tests/soccer-schema.test.cjs`: schema backward compatibility, new fields, official limits, `splitParticipants`, `ensureOfficials`, official labels and substitutions, identity/log bounds, and native-runner compatibility.
- Module tests for pitch, appearance and identity live in their own `tests/soccer-*.test.cjs` files.
- `tests/soccer-integration.test.cjs`: the workspace's per-step flow (roster and anchors from the document, `SoccerTracker.step`, `soccerPoints`, `appendTrackingLog`, `cleanIdentities`) checked against `trackingSchema`, then saved identities restored into a new tracker.
- `pnpm test` runs all of them. `node scripts/benchmark-soccer.cjs [steps] [width] [height]` times `analyzePitch`, `describe` (30 boxes) and `SoccerTracker.step` on a synthetic 1280×720 frame (detection time not included).
