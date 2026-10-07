# Persistent soccer tracking

This page describes the browser workspace. The local GPU prototype has its own
soccer identity layer (pitch detection, foot-point filtering, global identity
manager, debug overlay); see [prototypes/gpu-tracker/README.md](../prototypes/gpu-tracker/README.md).

This implementation replaces the browser run loop's single shirt-color memory
with separate local tracks and a saved global identity registry. Old match
documents remain readable. Confirmed labels can initialize learned galleries
when an older match is resumed. No sign-in is required; the upload limit remains
150 MiB.

## Original architecture and failure modes

- `player-detector.ts`: tiled YOLOX-Tiny COCO person/ball detections. A person
  category does not distinguish players, staff, spectators, or referees.
- `track-vision.ts`: torso histogram, grass mask, robust scale/translation camera
  estimate. The old mask joined entire row envelopes and admitted everyone if
  grass could not be isolated.
- `persistent-tracker.ts`: two-stage Hungarian matching with motion, area, IoU,
  shirt appearance, and ambiguity margins; local tracks expire after 1.4 seconds.
- `player-recovery.ts`: remembered shirt histograms and approximate re-entry
  locations. Same-kit teammates are indistinguishable using these features alone.
- Norfair: local Kalman filtering in Pyodide. It never provided match-long identity.
- The previous UI stopped at camera cuts and at approximately 3,400 saved points.

These modules are retained where useful. The primary browser pipeline now uses
`soccer-pipeline.ts`; the legacy recovery helpers remain for compatibility and
their existing tests. The optional Python native runner has not been upgraded.

## Automatic jersey color guide

The browser learns two shirt-color distributions from people on the pitch;
there are no preset team colors or manual color selections. Each mode needs
three separate, reasonably clear people in the same frame. Larger detections
inside the pitch carry more weight than tiny people near its boundary. Until
both modes have enough support, color does not reject any detections.

The central upper torso is sampled in three bands using soft hue, dark and
neutral bins. This allows shade changes, pale shirts, numbers and mixed kits;
green jerseys are not removed as grass. Matching either learned distribution
loosely is enough. Clear nearby people on the pitch survive even a poor color
match, and unreadable shirt crops remain visible. Only clear mismatches without
that foreground evidence are removed. Spectators in similar colors can remain.
This is a detection guide, not a team assignment or player identity.

The same guide is shared by scans and automatic tracking for the current video,
adapts slowly from clear matches, and resets when opening another video or
clicking **Relearn team colors**. The on/off preference is saved; color profiles
are relearned after reloading the page. Old saved manual color choices are
ignored. Rescan/retrack to apply the guide to existing footage; saved observations
are not relabeled or deleted. Disable the guide to review other kit colors.

## Processing and identity rules

1. **PersonDetector** (`player-detector.ts`) retains the existing detector,
   tiled scans, nonmaximum suppression and independent ball class.
2. **PitchDetector** (`pitch-detector.ts`) recomputes the largest connected grass
   region for each detection frame. It closes small holes only and tests a small
   neighborhood at the feet, allowing sideline tolerance. Dry and yellow-green turf is included, and narrow painted-line gaps are
   joined before selecting the connected field. The older second grass/touchline
   filter is no longer applied. A failed mask retains local person boxes for
   review while withholding automatic roster admission. Turning off the setup filter permits manual selection; automated
   admission still requires field evidence. The debug polygon is a visualization
   of the mask envelope; actual inclusion uses local mask occupancy.
3. **CameraMotionEstimator** reuses robust background patch matching and excludes
   known person boxes. Scale and translation compensate local motion. A cut
   resets local trajectories and invalidates the calibration chain, not identities.
4. **LocalTracker** reuses two-pass Hungarian association. Learned cosine
   similarity supplements motion, IoU, size and shirt appearance when available.
   Close competing assignments become gaps rather than forced matches. Norfair
   estimates can assist the next association in brief gaps; stored player
   observations remain detections rather than invented trajectories.
5. **TeamClassifier** finds two sizeable jersey-color modes, requiring at least
   three examples per mode and adequate separation, or uses manually labeled
   outfield examples. Repeated temporal votes are required. A single contrary
   frame cannot switch an established team. Auto-discovered A/B naming is arbitrary
   until manual examples establish the mapping.
6. **RoleClassifier** compares torso, shorts and lower-leg color evidence with
   confirmed referee/goalkeeper examples and accumulates role votes. Use **Track
   as referee** on a detected official and **Player role → Goalkeeper** for a keeper.
   Special roles require a confirmed kit example; uniform outliers stay UNKNOWN.
   This is not a trained referee classifier. Similar uniforms can still confuse
   appearance-based role inference, particularly before a referee is marked.
7. **PlayerReID** uses the bundled OSNet x0.25 MSMT17 model, RGB 256×128 crops,
   512-dimensional normalized descriptors and fixed batches of 16. It runs at
   one neural observation per processed second; detection/local association run
   at 5 Hz. When detailed scanning is enabled, tiled detection runs at every detection step so distant players remain observable between appearance checks. Disable it for faster wide-view scans. At most 32 high-quality person crops are embedded per scheduled frame.
   Edge truncation, overlaps, small boxes and low image gradient reject poor
   gallery crops. The gallery holds up to five distinct observations per identity,
   preserving the first trusted anchor. Quantized descriptors bound checkpoint size.
8. **GlobalIdentityManager** retains global IDs indefinitely within the match.
   A local track can expire and a new local ID can reconnect to that global ID.
   Restoration needs learned appearance, team/role agreement, spatial plausibility
   when calibrated, bidirectional ambiguity margins, and three independent neural
   sightings. Matching compares competing identities, including visible teammates.
   Shirt color or nearest image position alone never restores an identity.
9. **PitchCoordinateMapper** solves a projective mapping from four named real
   pitch corners. It rejects degenerate/concave configurations and uses only
   landmarks connected to the current camera segment. The normalized pitch
   position is the player's footpoint. No coordinates are fabricated from a
   grass bounding box. A conservative speed gate uses an assumed 105×68 m pitch,
   13 m/s maximum and 4 m tolerance; these are engineering defaults, not fitted
   probabilities or measured dimensions of the uploaded pitch.
10. **TrackHistoryManager** archives observations and recent camera transforms in
    five-second R2 objects. A revision-controlled D1 manifest commits references.
    The UI loads nearby chunks when seeking and all chunks when exporting. A
    manual correction invalidates that player's older-generation automatic
    history from that point onward. Substitution intervals filter archived data.

The registry records team, role, state, confidences, appearance gallery, kit,
last image/pitch positions, camera-compensated velocity and bounded recent
history. Active/hidden states are ACTIVE, OFF_FIELD (brief missing observation),
MISSING and SUBSTITUTED. Unresolved local tracks use UNKNOWN_PLAYER and do not
enter tactical player observations. Export separates `match.players`,
`match.goalkeepers` and `match.referees`.

New players require repeated field and team evidence, a usable learned crop, nearby people, and either compensated movement or a longer stationary observation.
If a same-team identity is missing, a new roster identity is withheld rather than
silently duplicating it. The existing active roster supplies identity slots;
extra local detections remain unresolved and generate a debug capacity warning.
Substitutes require the existing explicit substitution action and a fresh label;
an outgoing player is not automatically repurposed as the incoming player.

## Debugging and evaluation

Turn on **Identity debugging** before tracking. The overlay shows global/local
IDs, team, role, confidence, optional pitch coordinates and trajectories.
Referees have orange dashed boxes, goalkeepers yellow boxes, and unresolved
people purple dotted boxes. Unresolved local boxes also appear in normal mode; turn on name labels to see their status. The registry lists missing identities and their last
observed coordinates/time. Re-ID logs include appearance, spatial support,
missing duration, final score and rejection reason. Accepted events are archived;
debug rejections are bounded to avoid unbounded per-frame logs. Scores are
heuristic matching confidence, not calibrated probabilities.

Repeatable synthetic checks are provided, but were **not run**, per the user's
request to handle testing. They cover camera cuts and long absences, sparse
embedding cadence, same-kit ambiguity, absence of embeddings, role separation,
spectator filtering, impossible displacement, substitutions, homography and
archived correction invalidation. Run from Command Prompt:

```cmd
pnpm test:soccer
pnpm test
pnpm test:upload
```

For a real-footage evaluation:

1. Upload a representative clip with stands, referee crossings, player overlap,
   a camera pan/cut, and a known player's exit and return. Retain an untouched copy
   of your manual annotations for comparison.
2. Label stable identities and the referee/goalkeeper examples. Track with debug
   enabled, then use **Export tracking JSON**.
3. Independently annotate selected frames at the tracker’s 0.2-second sampling
   times. Include every visible player, referee and spectator in those frames:

   ```json
   {"frames":[{"time":12.0,"people":[
     {"id":"true-player-7","role":"PLAYER_TEAM_A","visible":true,
      "box":{"x":0.25,"y":0.4,"w":0.04,"h":0.15}},
     {"id":"true-referee","role":"REFEREE","visible":true,
      "box":{"x":0.5,"y":0.4,"w":0.04,"h":0.15}}
   ]}]}
   ```

4. Run:

   ```cmd
   pnpm evaluate:soccer pitchiq-tracking.json ground-truth.json
   ```

The report measures spectator admissions, referee-as-player errors, identity
switches, fragmentation, duplicate identities, correct/incorrect/unscored Re-ID
events, mean confidence, and per-player visible-frame coverage. It uses
one-to-one IoU matching (minimum 0.3) and nearest samples within 0.105 seconds.
It reports unannotated Re-ID events separately, never counts them as successes.
Truth IDs may differ from roster IDs; majority truth mapping is used for Re-ID
event scoring. Unmatched predictions are reported separately because incomplete
annotations must not be silently called spectator errors.

The repository contains a saved detection fixture, but **no sample MP4 or
independent identity/role ground truth**. That fixture cannot establish learned
Re-ID accuracy or full-match throughput. No empirical accuracy improvements or
real-time performance are claimed. TypeScript checking was performed separately.

## Limits and model provenance

OSNet is trained for general pedestrian Re-ID, not broadcast soccer. Tiny players,
same uniforms, compression, blur and occlusion remain difficult. A learned
embedding is more informative than a shirt histogram but does not guarantee
identification. The safe outcome is an unresolved track until a better crop or a
manual correction is available. Green technical areas connected to the pitch and
unusual turf/lighting are limitations of the lightweight grass segmentation.

The architecture bounds runtime state (64 simultaneous local tracks, 96 global
records, five embeddings per identity, 40 recent history entries per identity,
bounded debug events, and a 12-chunk playback cache). These are storage/compute
bounds, not a claim that a match has 64 players. The manifest permits 6,000 chunks
(over eight hours at five seconds per chunk); video timestamps remain capped at
four hours. Full-history export intentionally loads the requested history into
memory and can be large. Processing still stops when this browser tab closes.

Model source and preprocessing:

- [Torchreid / OSNet upstream](https://github.com/KaiyangZhou/deep-person-reid)
- [Upstream feature preprocessing](https://github.com/KaiyangZhou/deep-person-reid/blob/master/torchreid/utils/feature_extractor.py)
- [Bundled ONNX export publisher](https://huggingface.co/anriha/osnet_x0_25_msmt17)

The exact revision, SHA-256, graph shape and MIT notice are in
`public/models/OSNET-MODEL.md` and `public/models/OSNET-LICENSE.txt`.
