"""Possession from a finished result: which team has the ball on every frame, who carries it, and where the ball changed
hands (who lost it, who won it, how). Judged from the 2D positions of the players and the ball (the pitch metric) and
checked against the image (the reach geometry of ball_path.within_reach). Nothing here runs a model or reads the video.

Output (compute(result) -> JSON-safe dict, keyed on internal ids, never display labels; times are absolute source
seconds, frames absolute indices):
    version, fps, t0, t1, available, reason (why not, in one sentence), quality {level good|limited|unavailable,
    coverage (the share of the live, not-dead frames with a team named), notes}, shares {A, B, dead, unknown} (seconds),
    timeline [runs of constant (team, state, carrier): t0, t1, f0, f1, team, state control|flight|loose|contested|dead|
    unknown, carrier, estimated, confidence high|medium|low], changes [id 'pc-<won frame>-<new team>', lostAt, wonAt,
    frame, fromTeam, toTeam, lostBy, wonBy, cause tackle_or_duel|interception|out_of_play|unclear, confidence
    high|uncertain (CHANGE_LEVELS; possession-3 said high|medium|low)], deadSpells [t0, t1, reason out_of_play|stoppage,
    restartTeam].
The convention (PREDECLARED.md, the Metrica truth of tools/possession_truth.py): a team keeps possession from its
control through its passes, shots, clearances and loose balls until the other team controls the ball (a change, timed
twice: lostAt = the losing team's last touch, wonAt = the winning team's first control) or the ball goes out of play
(dead until the restart, which belongs to the restarting team). Between the loser's last touch and the winner's first
the ball is 'loose' and still the loser's.

Pipeline: result -> features() (per frame: the ball's state, origin, motion and pitch position in metres; per team, its
players nearest the ball in the image with their pitch distances and positions and which of them is the ball's
nearTrack player; the nearest person who may be a player of either team; the goalkeeper holding the ball in his hands;
camera cuts and motion; every team player's speed; a person's frame team where it contradicts his identity's) ->
excursions() (the ball path on something beyond a line that is not the ball in play) -> replays() (a stretch between two
camera cuts that shows another moment) -> evidence() (seconds of evidence per frame that a team is at the ball, and the
player behind it; keeper_hands(), knock_past()) -> boundaries() (dead balls and long unseen stretches, where possession
restarts with the first touch after them; live_holds(): no stoppage on a ball held in play) -> viterbi() between
boundaries (the possession sequence, changes starting only on a confirmed observed control) and posterior() (how sure
each frame is) -> the operating point (a team is named where its posterior >= min_posterior) -> record() (with
change_levels(), the confidence of each change).

Evidence. A team is at the ball on a frame when one of its players is near on the pitch (full evidence within r0 = .7 m,
none from r1 = 1.5 m) AND in the image (1.0 to 2.0 m): the frame counts as near as the less near metric of that player,
so a carrier is named only where both agree on him. Only one team may be near (the score is multiplied by one minus the
other team's): a ball between two players is contested, evidence for nobody; so is a ball at a person who may be a
player of either team (no team yet: an unresolved '?' track, a goalkeeper the tracking has not placed). Referees, staff
and people rejected as off the pitch are neither at the ball nor in the way. A goalkeeper without a team plays for the
team that defends the goal on his side (tactical.goalEnds). A bridged (interpolated) frame counts half, with the
nearTrack player (whose lower body hid the ball) near where the first pass followed this ball (its centre within
nt_agree = 2 m of the final one); a bridge longer than 1.2 s is no ball. A FAST ball counts only at the peak of a bend
of its path (a velocity change of 6 m/s or more over +-4 frames, the toucher near as on any frame): on game 1 a coded
pass, reception, recovery or loss bends the ball by a median 11-23 m/s while a pass in flight bends by .08 m/s (full
tracking; 2 m/s of noise on the camera), and the passes of Metrica's game are FAST on 80% of their frames. Blocking FAST
balls altogether lost the first-time passes and interceptions; treating them like any ball counted the balls flying past
a player (both measured at stage A, possession-1). Where the ball has an image position but no usable pitch position (a
registration hole: the broadcast's 190-215 s), or a player has none, the image alone counts half. possession-4 adds
three rules here (fixes 3, 4 and 9 below): a ball in a goalkeeper's hands is his team's, a ball running past a player is
not his, and an identity carried onto another team's player plays for the team its frames name.

Decoding. A two-state Viterbi between boundaries: staying with a team costs the other team's evidence, a change costs
switch = .15 s of evidence, and a team takes the ball over only on a frame where a change may start (below), so a change
lands on the winner's first control and a short touch by the other team is absorbed: an opponent alone on the ball for
less than 2 x switch (a deflection, a challenge with the carrier a metre away) is no change, while one who holds it for
longer, or touches it and is followed by a team-mate's touch (an interception, a first-time pass), is. The
forward-backward posterior of the same model (costs read as tau = .05 s per unit of log-likelihood, a change allowed on
any evidence frame) says how sure each frame is: frames below .8 were right only 40-60% of the time on game 1 (v2; .7-.8
now 72%), so they are not named (state unknown).

Where a change may start (fix 1 of possession-3, start_on 'confirmed'): on a run of one player's evidence on both
metrics (the ball and the player have pitch positions) that holds at least one observed frame, a TRACKED ball of
OBSERVATION origin; the run counts from its first frame, so the ball hidden at the feet of the player who is then seen
with it was his from the start. An interpolated run that is never observed (the bridged ball drifting to a defender), a
weak detection alone and image-only evidence (a registration hole) never start a change; all of them still weigh on the
decoder and on the posterior. Measured against the alternatives (game 1 camera seed 0, F1 against every change at +-2 s
/ accuracy at coverage): any evidence (v2) .608 / .961 at .887; 'confirmed' .607 / .968 at .871 (precision .82 -> .84,
median wonAt error .187 -> .160 s, wonBy .714 -> .775); observed frames only .585; observed frames only and the decoder
weighing nothing else (the literal rule) .524 / .970 at .854, seed 1 .520, visible .501, dead ball unseen .537. On
broadcast 42-222 s every variant from 'both metrics' on removes the same two false changes (image-only, inside the
registration hole) and the literal one also the 139.4 s pair (4 changes, all right, against 6 with 4 right): bought at
.05-.08 of F1 on every game-1 camera variant, so it is not the default; the pair is 'uncertain' instead. The 139.4 s
pair is not a bridged drift alone: the ball path followed a detection on the defender's boot for 0.4 s (TRACKED,
OBSERVATION origin, both metrics at the defender) while the real ball stayed at the dribbler's feet (checked on the
video). The coverage cost is the posterior's: changes start on fewer frames, the frames between a hidden first touch and
its observed confirmation are left unsure. Image-only evidence starting nothing costs the synthetic registration hole
(below) .025 of coverage and .03 of F1, and the independent-noise stress test .023 of F1: there the simulator's image is
exact and its pitch noisy, the reverse of the broadcast's hole. A result that records no ball.origin at all (fix 5)
takes every TRACKED ball as observed.

Boundaries. Out of play: the ball seen 1 m or more beyond a line for .2 s under a ball pitch confidence of .6 or more
(or on its last sighting before a long unseen stretch: Metrica blanks a dead ball), not back on the pitch within 1 s (a
lofted ball is projected beyond the far touchline: the ball audit), and consistent (fix 2 of possession-3,
excursions()): the path got there from the pitch (from the last ball on the pitch within out_look = 1 s it would not
have travelled faster than out_speed = 40 m/s, the ball tracker's MAX_SPEED), the first pass was not following another
ball on most of its frames (in ball_path.within_reach's metres, image rows doubled: fix 6), and no other ball candidate
was in play on the pitch on most of them (_live_candidate). An inconsistent excursion is no ball: possession is carried
over it (the same team on both sides, estimated, low confidence) or unknown, never dead, and it is never a stoppage cue.
On broadcast 42-222 s the path jumped 44 m in 0.2 s (and 57 m) to a steward's ball 1.5 m beyond the near touchline at
93.6-96.2 and 100.8-103.8 s while another candidate moved on the pitch on 83% / 76% of those frames (35% of all frames):
v2 made a 10.2 s out-of-play spell and a false change and change back of it; v3 none, and the 20 points there (all
Barcelona's) are carried and right. Game 1 has no such excursion (the simulator has no other objects). A dead spell runs
from the crossing to the restart.

A stoppage: the ball still for 4 s (within 1.5 m of its mean: a ball placed for a free kick), with a long unseen stretch
just before it; or (fix 3 of possession-3) both teams slowed down while nobody plays the ball: the players' activity
(the p75 of the team players' speeds over 1 s, at least six players, smoothed over 1 s; player_speeds: pitch positions,
else the feet carried through the camera motion) under slow_v = 2.2 m/s for slow_s = 3 s while the ball is not seen, or
stands still (within still_r of where it was a second before, on the pitch or in the camera-compensated image; a ball
not seen a second before is compared with where it was seen again since: fix 7), or nobody is at it (no evidence within
.5 s, the ball not FAST); dead from the start of that stretch to the touch that ends it (or the ball seen again: a kick
from a ball seen again has no evidence of its own). Neither rule makes a ball held in play dead (fix 2, live_holds()):
the stoppages possession-3 would find tell which controls began with the ball dead (a taker walking up to a placed
ball), then the rules run again without the controls a team got in play. The signature, measured: on Metrica game 1's
true tracking the p75 speed (over .5 s) is 1.31 m/s in dead time and 2.77 in play (medians, 1 s smoothing; a frame
threshold alone recalls 85% of the dead frames at 86% precision); through the simulated camera with the ball always
shown (dead_fill 'visible', 1 s lag) 1.75 against 3.12, and p75 < 2.2 for 3 s or more is dead on 90% of its frames alone
(recall .69). On the broadcast the activity alone does not separate them (42-222 s: dead time 1.48 m/s against 2.64 in
play, but Barcelona's build-up after the goal kick runs at 0.8-1.2, the first minute's build-up at 1.1-1.8): the ball
cue is what keeps slow live play live (no live point called dead by it on either broadcast run). Image-only evidence
counts as playing the ball (slow_played 'any'): counting only evidence with pitch positions finds the inferred foul
stoppage of 42-222 s (25 of its 41 dead points dead instead of 9, 25 of 26 right) but also stops slow live play in a
registration hole (BC60 with a synthetic 20-25 s hole, three placings: 102 of 140 points named and 2-3 false stoppages
each, against 137 of 140 and none), so it is not the default; the activity's median instead of its p75 recalls more of
game 1's dead time (.96 / .90 full / camera, with or without fix 2) and was not adopted for the broadcast's slow
build-up (possession-3's grids). A stretch between two boundaries where nobody is at the ball keeps the possession
before it (keep_prev), so a restart after an unseen stretch and a stoppage is still a change of the team that lost it.
An unusable ball for 1.5 s or more is a 'gap': possession restarts with the first touch after it (no restart bonus:
restart_gap), and the gap is carried when the same team is decoded on both sides (carry_by 'decoded'), else unknown. A
camera cut (frames[].cut, or a new frames[].segment) is a gap however short, never carried, and after it a team is named
only from the first observed control on (fix 8); a stretch between two cuts that shows another moment is a possible
replay (replays()) and shows no ball. The restart is when a placed ball leaves its spot (still .3 s), else when the ball
is first seen back on the pitch; its taker is the player within 1.5 m of the ball then (else the first team at it within
2 s) and gets 1 s of evidence after a dead ball: a set piece is a single kick, a goal kick leaves the keeper on its
first frame, too fast for a bend.

Who and how (fix 4 of possession-3). lostBy is the losing team's player at the ball on its last evidence frame before
the gain (lostAt: its last control or touch); wonBy the winning team's player with the most evidence over the .5 s after
wonAt and before the next change (the player who then holds the ball; v2 took its first evidence frame, which can be a
challenger who only poked it): game 1 wonBy .867 -> .888 (full) and .675 -> .775 (camera, with fix 1), lostBy .894 ->
.883 / .781 -> .776; over 1 s or 2 s it picks the receiver of the next pass (.832 / .726 full). The cause: out_of_play
across a ball out of play; interception when the ball travelled 3.5 m or more from the loss to the gain;
tackle_or_duel when it did not and a player of the winning team was within 2 m of the ball as it was lost, or both
teams were near it in between; unclear otherwise (a restart after a stoppage among them).

How sure a change is (change_levels). high: an open change, not in a back-and-forth pair (a change undone within PAIR_S
= 2 s, or one undoing a change made 2 s before), with .3 s of the new team's evidence over 2 s, the loser's last touch
at most HIGH_LOST = 5 s before, and the new possession lasting HIGH_HOLD = 2 s. uncertain: the rest (possession-3's
medium, restarts after a dead ball among them, and low: the pairs, an open change with less than .25 s of the new team's
evidence over the next 2 s, a change across a gap or a cut, one whose loss nobody saw). Fix 10 merged medium and low
because they were not ordered: pooled over game 1's five variants (+-2 s against every truth change) medium was right on
157 of 234 (67%, 95% interval 61-73%) and low on 168 of 228 (74%, 68-79%); +-1 s 62% against 61%; outside one-touch
possessions 43% against 54%; on broadcast 42-222 s medium 2 of 2 and low 0 of 2. 'high' is meant to be right 85% of the
time: on game 1 it is right on 97% of 102 (full) and 94% / 90% / 92% / 94% of 90 / 94 / 91 / 97 changes (camera seed 0 /
seed 1 / ball shown / ball never shown), on broadcast 42-222 s on 2 of 2; uncertain changes on 83% / 74% / 63% / 62% /
69% (BC180: 2 of 4). Hiding the uncertain changes would hide about 4 in 10 of the right ones on game 1 (325 of 769 at
+-2 s), so the page shows every change, the uncertain ones hollow.

One-touch possessions (fix 6 of possession-3). Metrica codes every first-time interception, header or clearance as a
gain (a RECOVERY of the player who loses the ball on the same frame): 79 possessions of game 1 (138 of its 320 changes
go into or out of one). The convention here absorbs a deflection, so the engine finds 18 of the 79 changes into one and
17 of the 59 out of one only (full tracking; camera 14 / 15) against 144 / 182 of the rest (camera 123): its F1 against
every change is .691 (full) / .607 (camera), against the changes outside one-touch possessions .758 / .678 (where its
changes into one count as false). switch .10 finds more of them at a precision of .885 (full, all) / .766 (camera);
switch .20 fewer. Both conventions are reported, the pre-declared metric is the first; the hybrid figure (precision
against all, recall outside one-touch: .844 / .749) was defined after the plan and is context.

Run-level abstention (the ball audit's section 6): not available on small-scene footage (frames[].profile.smallScene on
half the frames or more, ball_path.SMALL_RUN: amateur 3 had pitch-vs-image one-team agreement .30 and its ball on a
painted mark), when the ball is placed on fewer than 20% of the frames, or when the pitch and the image agree on the
team at the ball on fewer than 65% of the frames where one of them puts exactly one team near it (measured over 150
frames or more; the audit measured .87 on the broadcast and .58 on the Veo clip; the simulated camera's 5-minute windows
range .68-.93). Below .85 the quality is 'limited'. On the four PitchIQ runs (small-scene share / agreement): broadcast
0-60 s 0 / .877, broadcast 42-222 s .009 / .869, amateur 3 .60 / .277, Veo .607 / .571: amateur 3 and Veo abstain.

possession-4 (2026-10-20; the dev data of PREDECLARED_v4.md: game 1, BC60, BC180 and the v3 verifiers' synthetic
scenarios, v4/engine in the scratch; every rule can be switched off, DEFAULTS; with all of them off the records equal
possession-3's on the four run copies and the 120 game-1 windows of the bench, version field aside):
 1. Results shorter than the 1 s smoothing (30 frames at 30 fps) made possession-3 raise (np.convolve 'same' returns the
    kernel's length): _centred_sum, bit-identical wherever the result is long enough; 1 to 30 frames now give a record.
 2. A player in control of a ball he got in play is no stoppage (live_hold, live_holds()). Game 1's in-play seconds
    called dead: full 86.6 -> 37.6, camera 81.6 -> 48.2, seed 1 80.3 -> 51.5, ball shown 82.6 -> 49.2, ball unseen
    105.9 -> 73.5; dead recall / precision .934 / .966 -> .929 / .985 (full), .869 / .965 -> .872 / .979 (camera, with
    fix 7), .850 / .965 -> .857 / .978, .876 / .965 -> .871 / .979, .809 / .952 -> .808 / .966; dead seconds missed
    172.1 -> 185.5 (full: keepers fetching the ball for a goal kick after it left the pitch on bridged frames, which
    look like a catch), 343.1 -> 334.5, 393.1 -> 372.6, 323.6 -> 337.9, 498.7 -> 503.2; change F1 unchanged. The
    synthetic holds (a player shielding, waiting or carrying at 0-2 m/s for 3-6 s, others walking or jogging): 26 of 54
    were dead (136 s), now none; the placed balls of a clip's start (throw-in, free kick, pressed free kick, goal kick)
    stay dead. BC60 / BC180 unchanged.
 3. A ball in a goalkeeper's hands (its centre above his lower body in his box, him in his penalty area, HANDS_MIN
    frames running) is his team's on both metrics (keeper_hands). Synthetic catch at 3.9 s held 6 s: the change was
    dated at the release (10.0 s) and the hold was dead (10.1 s, others walking) or B's (6.1 s, jogging); now dated at
    the catch, A throughout, no dead ball (a 12 s hold too). The simulator never puts the ball in a keeper's hands and
    the four run copies have no such frame: game 1 and the broadcast are unchanged. A ball not seen in his hands is not
    found (a 6 s unseen hold with everyone walking is still a stoppage).
 4. A ball running past a player is not his (knock_past()): the synthetic knock-on dribble past three defenders 0.3-0.7
    m from the ball's line gave 6 false changes on 6 of 60 variants (36), now none; game 1 full one false pair fewer
    (F1 .6885 -> .6911, precision .895 -> .904), the cameras unchanged (lostBy .776 -> .783 on seed 0); BC180 one 0.4 s
    'loose' run is now 'flight'. Requiring the other team's touch next keeps a soft reception (Metrica's ball moving
    with its receiver at 3-4 m/s, no bend): without it the rule made a true change 2.9 s late.
 5. A result without ball.origin (the user's runs before 2026-10-15) takes TRACKED as observed and says so in a note
    (origin_fallback): BC180 with the field stripped gave 0 changes at coverage .743, now 6 (4 right within 2 s) at
    .908, as with the field; BC60 unchanged.
 6. The first-pass excursion check measures in ball_path.within_reach's metres (rows doubled, fp_rows): the synthetic
    ball boy's ball 3 m beyond the line (and 4.5 m), the first pass on the real ball, gave a 7.9 s out-of-play (or
    stoppage) spell, now none. Game 1 and the run copies unchanged.
 7. still_ball() compares a ball not seen a second before with where it was seen again (still_since): the synthetic
    clip that starts with the ball unseen 3 s and then placed 3 s had no dead ball, now dead 0.3-5.5 s (kick at 6.0);
    camera dead recall .869 -> .876 (seed 1 .850 -> .858) with this fix alone.
 8. Camera cuts (cut_fresh) and replays (replays()): a 6 s replay of the other team's attack was named for it with two
    'low' changes; now unknown, no change; a replay inside a stoppage is part of the dead spell; the frames after a cut
    and before the first observed control are unknown (0.6 s on the unseen-after-cut scenario). No PitchIQ run copy and
    no simulated game has a cut: nothing else changes.
 9. A person whose frames name another team than his identity's roster team, for KIT_S or more, plays for the frames'
    team (kit_team, kit_overrides()): the synthetic identity swap now gives both changes (10.0 and 16.0 s, high)
    instead of none. PitchIQ writes the identity's team on its frames, so no run copy has such a frame and nothing
    changes on them; the identity tracker's kit readings were tried and dropped (KIT_* above: misreads at crossings).
    Corrections and user identities keep their team.
10. Change confidence high / uncertain (above). Timeline runs keep high / medium / low: their named frames are right on
    97.5-98.1% (high), 70-81% (medium) and 93-100% of the few low ones (213 / 0 / 53 / 0 / 707 frames).

Measured on Metrica sample game 1 (TUNE: tools/possession_truth.py; tools/possession_sim.py's full tracking and
simulated broadcast camera), 20 windows of 300 s each a separate result, at the operating point; possession-3 in
brackets (tools/possession_bench.py reproduces every number; game 2 was the v3 validation and is seen data now; game 3
is held out for the v4 freeze):
                     accuracy / coverage       dead recall / precision   in-play called dead / dead missed, seconds
    full tracking    .973 / .940 (.972 / .924)  .93 / .98 (.93 / .97)     37.6 / 185.5 (86.6 / 172.1)
    camera           .969 / .881 (.968 / .871)  .87 / .98 (.87 / .97)     48.2 / 334.5 (81.6 / 343.1)
      second seed    .968 / .883 (.968 / .874)  .86 / .98 (.85 / .97)     51.5 / 372.6 (80.3 / 393.1)
      ball shown     .969 / .881 (.967 / .872)  .87 / .98 (.88 / .97)     49.2 / 337.9 (82.6 / 323.6)
      ball unseen    .963 / .864 (.963 / .854)  .81 / .97 (.81 / .95)     73.5 / 503.2 (105.9 / 498.7)
                     F1 +-2 s all / no 1-touch   F1 +-1 s all  median error won / lost   wonBy / lostBy
    full tracking    .691 / .758 (.689 / .754)   .633 (.631)   .04 / .07 s (same)        .888 / .883 (same)
    camera           .607 / .678 (same)          .523 (same)   .16 / .24 s (same)        .775 / .783 (.775 / .776)
      second seed    .571 / .622 (same)          .485 (same)   .20 / .24 s (same)        .733 / .767 (same)
      ball shown     .564 / .650 (same)          .488 (same)   .16 / .24 s (same)        .768 / .775 (.768 / .768)
      ball unseen    .598 / .665 (same)          .514 (same)   .24 / .25 s (.24 / .26)   .715 / .787 (.715 / .780)
Further game-1 tests: 60 s windows of the camera .970 / .858, dead .64, F1 .549 / .661, 11 of 97 windows abstaining (3
on the agreement, 8 with the ball placed on under 20% of the frames); a registration hole of 20 s in every 180 s .972 /
.849, F1 .555 / .626; the independent-noise stress test abstains on 20 of 20 windows, with its gate off .959 / .866,
F1 .484 / .536, 'high' changes right on 78% (far harsher than the broadcast, and gated).
Along min_posterior (accuracy / coverage):
    min_posterior   0           .5          .7          .8          .9          .95         .99
    full tracking   .949/.986   .960/.967   .969/.949   .973/.940   .979/.920   .982/.908   .985/.882
    camera          .928/.983   .950/.936   .962/.904   .969/.881   .975/.845   .981/.820   .988/.769
Precision / recall at +-2 s against every change: full .904 / .559, camera .840 / .475. Every accuracy and coverage
target is met (full .95 at .85, camera .90 at .65); the change targets are not (full F1 .80: .691, or .758 without
one-touch possessions; camera F1 .65: .607, or .678 without them, .622 on the second seed).

PitchIQ runs, hand labels in tools/annotations/possession (points every .5 s; changes within +-2 s, same new team; dev
data): possession-4 scores exactly as possession-3 on all four:
    BC60 (0-60 s)      117 of 118 team points named and right; 2 of 2 dead points dead; no change (none)
    BC180 (42-222 s)   accuracy .977 at coverage .975, without the BC60 overlap .975 at .979; dead points called dead 9
                       of 41, engine-dead points truly dead 9 of 9; the goal kick 192.85-198.7 s 70% covered, the
                       inferred foul stoppage 199.45-213.77 s not at all; changes: 6 reported, 4 right within 2 s
                       (precision .667, recall .667), within 1 s .333 / .333; high 2 of 2, uncertain 2 of 4 (the false
                       139.4 / 140.5 s pair); median error of the matched wonAt 1.05 s, lostAt 1.41 s; the 1.45 s
                       yellow possession 116.95-118.4 s is absorbed (missed)
    AM, VEO            not available: small scene on 60% / 61% of the frames
0.63 s of features and 0.09 s of decisions for the 5400 frames of 42-222 s (possession-3 0.51 / 0.05 s: v4 adds the
frame-team pass, the keeper's box test, the candidates' positions and a second pass of the stoppage rules).

Stage B notes kept from possession-2 (the PitchIQ adaptations, measured on BC60 and the runs' own data):

- The first pass's nearTrack is often another ball's: on bridged frames the nearTrack person stands more than 2 m from
  the final ball on 66% / 50% / 57% / 23% of the frames (BC60 / 42-222 s / amateur 3 / Veo), every time the first pass's
  own ball is more than 2 m from the final one, so nearTrack counts only where they agree (nt_agree).
- The carrier must be one player near on both metrics (BC60: one carrier in 13 of 14 labelled carries; 16 of 16 random
  control frames right on the video).
- A person who may be a player of either team near the ball is evidence for nobody; a goalkeeper without a team takes
  the team of the goal on his side; a keeper holding the ball in his hands is seen at it only through his box (fix 3:
  neither metric finds a ball off the ground at his feet).
- A registration hole (synthetic, BC60, every pitch position dropped over 5-25, 20-45 or 35-60 s): v2 named 139 of the
  140 labelled points inside, all right; v3 137, all right, with no change and no false stoppage.

What it cannot do (yet): a change on the broadcast is right about 2 times in 3 (4 of 6 on 42-222 s) and dated within 1 s
only 1 time in 3 (the ball hidden or misplaced at the moment of the duel: 72.4 s against 71.2, the path still at the
Barcelona dribbler; 175.07 s against 176.95, a yellow player at the ball on the tracking 1.9 s before the labellers'
control); short possessions of 1-1.5 s are absorbed; a stoppage with the ball rolling in a registration hole is not
found when players stand by the ball (42-222 s, 199.45-213.77 s: of its 29 dead points 21 unknown, 8 Barcelona's);
Metrica's one-touch possessions are mostly not changes here; a keeper's hold the ball tracker does not see, a ball boy's
ball the path is on from the clip's first frame without a first-pass record, a replay the registration cannot place and
a control that begins in a clip's first second are not judged by the possession-4 rules; the simulator has no other
objects, no identity errors, no cuts and no registration collapse, so fixes 3, 5, 6, 8 and 9 and the image fallback of
the activity are measured on the run copies and the synthetic scenarios only. Out of play has never been checked on a
real ball tracked over the line in PitchIQ footage (the v3 critic).
"""
import math

import numpy as np

from . import ball_path, pitch_model

VERSION = 'possession-4'
LENGTH, WIDTH = 105.0, 68.0          # metres, when the result names no pitch size

DEFAULTS = {
    # --- the ball frames that may decide who is at it
    'min_conf': pitch_model.MEDIUM,  # ball pitch confidence (the analytics floor)
    'max_sigma': 3.0,                # m: a ball position less certain than this decides nothing (ball audit rule 1)
    'long_bridge': 1.2,              # s: a bridged (interpolated) gap longer than this is no ball (ball_path.MAX_SKIP_S)
    # --- evidence that a team is at the ball
    'r0': .7,                        # m: full pitch evidence within this distance of a team's nearest player ...
    'r1': 1.5,                       # ... none from this one (linear between)
    'img_r0': 1.0,                   # m: the same in the image (ball_path.within_reach geometry) ...
    'img_r1': 2.0,                   # ... the frame counts only as near as the less near of the two metrics
    'use_image': True,               # require the image metric to agree with the pitch metric
    'image_only_w': .5,              # weight of image-only evidence: the ball or the player has no usable pitch position
    'near_track': True,              # on a bridged frame, the player whose lower body hid the ball (nearTrack) is near it ...
    'near_w': .5,                    # ... with this proximity score ...
    'nt_agree': 2.0,                 # ... where the first pass's ball was within this of the final ball (m, image metric)
    'bridge_w': .5,                  # evidence of a bridged frame counts this much
    'unlabelled': True,              # a person who may be a player of either team (no team yet) near the ball is evidence
                                     # for nobody, as an opponent is (the score is multiplied by one minus his)
    'fast': 'peak',                  # a FAST ball: evidence only at the peak of a bend ('peak'), never ('block'), as any ('none')
    'kink': 6.0,                     # m/s: velocity change of the ball that is a touch
    'kink_w': 4,                     # frames on each side over which the velocity is taken
    'touch_w': .15,                  # s of evidence of a touch on a fast ball
    # --- the decoder
    'switch': .15,                   # s of evidence a change of possession costs
    'start_on': 'confirmed',         # where a change may start (fix 1): 'confirmed' (a run of one player's evidence on
                                     # both metrics that holds an observed frame: TRACKED, OBSERVATION), 'observed' (only
                                     # such frames), 'tracked' (weak detections too), 'both' (any both-metric evidence),
                                     # 'any' (v2: any evidence, image-only too)
    'decode_on': 'all',              # the evidence the decoder weighs: 'all', or 'observed' (only where a change may start)
    'posterior_on': 'any',           # how sure a frame is: as if a change could start on 'any' evidence, or the 'same' model
    'tau': .05,                      # s of evidence per unit of log-likelihood: the posterior's temperature
    'min_posterior': .8,             # a team is named where its posterior is at least this (the operating point)
    # --- who won the ball, how sure a change is
    'won_by': 'most',                # wonBy: the new team's player with the 'most' evidence over won_w, or its 'first' (v2)
    'won_w': .5,                     # s after wonAt (and before the next change) wonBy is looked for
    'conf_rule': 'v4',               # change confidence: 'v4' (change_levels: high or uncertain), 'v3' (high / medium / low)
                                     # or 'v2' (_change_level)
    'high_ev': .3,                   # s of the new team's evidence over 2 s that a 'high' change needs (change_levels)
    # --- restarts, dead balls, unseen stretches
    'gap': 1.5,                      # s: the ball unusable this long restarts possession (the first touch after it decides)
    'carry_gap': True,               # an unseen stretch with the same team on both sides keeps that team (estimated) ...
    'carry_by': 'decoded',           # ... the team decoded after it ('decoded'), or the taker found at its end ('taker', v2)
    'keep_prev': True,               # a stretch between boundaries where nobody is at the ball keeps the possession before it
    'restart_w': 1.0,                # s of evidence for the restart's taker ...
    'restart_gap': False,            # ... after a dead ball only (True, v2: after an unseen stretch too)
    'restart_s': .3,                 # s after the restart within which the ball must be seen for that evidence
    'taker_r': 1.5,                  # m: the restart's taker is the player this near the ball as it is kicked or seen again ...
    'restart_look': 2.0,             # s: ... else the first team at the ball within this after the restart
    'out_margin': 1.0,               # m beyond a line: out of play ...
    'out_min': .2,                   # ... seen there this long ...
    'trusted': .6,                   # ... under a ball pitch confidence of at least this ...
    'out_confirm': 1.0,              # s: ... not back on the pitch within this (that was a lofted ball) ...
    'out_consistent': True,          # ... and consistent (fix 2, excursions()): else the frames show no ball
    'out_speed': 40.0,               # m/s: the path cannot have reached the excursion from the pitch faster (ball.MAX_SPEED)
    'out_look': 1.0,                 # s: how far back the last ball on the pitch is looked for
    'still_s': 4.0,                  # s: a ball standing still this long is a stoppage (placed for a set piece)
    'still_r': 1.5,                  # m: still means within this of the stretch's mean position
    'placed_s': .3,                  # s: a restart ball still this long was placed; the restart is when it leaves
    'slow_v': 2.2,                   # m/s: the players' activity under this ... (fix 3; inf slow_s: off)
    'slow_s': 3.0,                   # s: ... for this long while nobody plays the ball: a stoppage
    'slow_play': .5,                 # s: the ball is played when a team is at it within this ...
    'slow_played': 'any',            # ... on any evidence ('any'), or only on evidence with pitch positions ('pitch')
    'slow_back': 0.0,                # s: the stoppage starts after the last touch up to this before the slowdown
    'act_q': .75,                    # activity: this quantile of the team players' speeds (player_speeds) ...
    'act_min': 6,                    # ... over at least this many players ...
    'act_smooth': 1.0,               # ... smoothed over this many seconds ...
    'act_teams': 'all',              # ... of 'all' players together, or 'both' (each team's quantile under slow_v)
    # --- causes
    'duel_r': 2.0,                   # m: an opponent this close to the ball as it was lost: tackle or duel
    'travel_m': 3.5,                 # m: the ball travelled this far from the loss to the gain: interception
    # --- run-level abstention (ball audit section 6)
    'small_run': ball_path.SMALL_RUN,
    'agree_min': .65,                # pitch-vs-image one-team agreement below this: not available
    'agree_frames': 150,             # ... measured over at least this many frames
    'agree_r': 1.5,                  # m: the agreement's radius (the audit's)
    'min_usable': .2,                # share of frames with a usable ball below this: not available
    # --- possession-4 (each rule can be switched off; the v3 behaviour in brackets)
    'kit_team': True,                # fix 9: a frame whose own team contradicts the identity's roster team (KIT_*) plays
                                     # for the frame's team (False: the roster team always)
    'keeper_hands': True,            # fix 3: the ball in a goalkeeper's hands in his penalty area is his team's (HANDS_*)
    'knock_past': True,              # fix 4: a ball running past a player is not his (False: any proximity counts) ...
    'pass_m': 1.5,                   # ... m: it moved this far relative to him over his contact, with no touch of his team,
    'pass_after': 1.0,               # ... and the other team touched it next, within this many seconds after the contact
    'live_hold': True,               # fix 2: a player in control at the ball, who got it in play, is live play ...
    'hold_gap': .5,                  # ... s: his control may lapse this long without the other team at the ball ...
    'hold_look': 1.0,                # ... s: a stoppage cue this long before his control means he picked up a dead ball
    'hold_share': .5,                # ... a still ball held in play on this share of its frames is no placed ball
    'origin_fallback': True,         # fix 5: no ball.origin anywhere: a TRACKED ball counts as observed (False: no change)
    'fp_rows': 2.0,                  # fix 6: the first-pass excursion check's image rows weigh this much (v3 1.0)
    'still_since': True,             # fix 7: a ball unseen a second ago is compared with where it was seen again since
    'cut_fresh': True,               # fix 8: after a camera cut a team is named only from an observed control on
    'replays': True,                 # ... a possible replay between two cuts is unknown ...
    'replay_max': 20.0,              # ... s: at most this long
}

STATES = {'UNKNOWN': 0, 'TRACKED': 1, 'MISSING': 2}
MOTIONS = {'UNKNOWN': 0, 'STATIONARY': 1, 'ROLLING': 2, 'FAST': 3, 'OCCLUDED': 4}
LOOSE_STATES = ('rejected', 'nontarget')   # people who are not players: never at the ball, never in the way
TEAMS = ('A', 'B')
CANDIDATES = 3   # per team and frame, the players nearest the ball in the image whose pitch distance is kept (a carrier
                 # must be near on both metrics; more than three team-mates within the image reach of the ball is rare)
ORIGINS = {'UNKNOWN': 0, 'OBSERVATION': 1, 'WEAK_DETECTION': 2, 'INTERPOLATION': 3}
LIVE_SPEED = 1.0   # m/s: a ball candidate moving this fast ...
LIVE_AWAY = 5.0    # ... this many image metres from the final ball is another ball in play (_live_candidate: on the
                   # broadcast run 42-222 s on 35% of all frames, on 83% / 76% of the two excursions to a steward's ball)
PLAYER_H = 1.8     # m: a player's box height, the image scale of player_speeds (its image speeds are .85 of the pitch
                   # speeds of the same players and frames on both broadcast runs, 38744 / 99601 pairs)
ACT_LAG = 1.0      # s: player_speeds' lag (through the simulated camera a .5 s lag puts the dead-time p75 at 2.3 m/s
                   # against 3.5 live, the position noise dominating; 1 s: 1.75 against 3.12)
# fix 9 (kit_team): a person's frame team (frames[].people[].team, with a label of the same team) that names another
# team than his identity's roster team, on the frames of one track for KIT_S or more with none KIT_GAP apart and no
# frame of the roster team in between, plays for the frame's team (an identity carried onto another team's player). A
# person corrected or identified by the user ('manual', 'correction') keeps his team. The identity tracker's window
# readings (frames[].people[].why, '... | PLAYER TEAM B 1.00') are not used: on the broadcast run 42-222 s every run of
# them contradicting the roster at .9 or more for .5 s (5, 109-121 frames) was a misread kit while two players
# overlapped, the identity right on the video at 93.0, 121.5 and 198.5 s.
KIT_GAP = .5       # s
KIT_S = .5         # s
# fix 3 (keeper_hands): the ball's centre inside a goalkeeper's box above its lower body (ball_path.lower_body is v .6 to
# 1.08 of the box height, u -.1 to 1.1 of its width) and the keeper inside his penalty area (his team's goal by
# tactical.goalEnds, either goal when the ends are not known) for HANDS_MIN frames running: he holds the ball.
HANDS_U = (-.1, 1.1)
HANDS_V = (-.15, .6)
HANDS_MIN = 6      # frames (.2 s at 30 fps): a shot flying past at chest height crosses a box in a frame or two
PEN_DEPTH, PEN_HALF = 16.5, 20.16   # m: the penalty area, from the goal line and either side of the pitch's middle


def _num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _pair(v):
    return isinstance(v, (list, tuple)) and len(v) == 2 and _num(v[0]) and _num(v[1])


def _box(v):
    return isinstance(v, (list, tuple)) and len(v) == 4 and all(_num(x) for x in v)


def roster_teams(result):
    """{id: team} of the final roster's team players and goalkeepers (corrections rewrite it)."""
    out = {}
    match = result.get('match') or {}
    for key in ('players', 'goalkeepers'):
        for e in match.get(key) or []:
            if isinstance(e, dict) and isinstance(e.get('id'), str) and e.get('team') in TEAMS:
                out[e['id']] = e['team']
    return out


def _keeper_team(result):
    """(left, right): the team defending each goal (tactical.goalEnds; one end named gives the other), for a goalkeeper
    whom the tracking has not given a team."""
    ends = (result.get('tactical') or {}).get('goalEnds') or {}
    left, right = ends.get('left'), ends.get('right')
    if left in TEAMS and right not in TEAMS:
        right = TEAMS[1-TEAMS.index(left)]
    elif right in TEAMS and left not in TEAMS:
        left = TEAMS[1-TEAMS.index(right)]
    return (left, right) if left in TEAMS and right in TEAMS and left != right else (None, None)


def _in_penalty_area(x, y, L, Wd, side):
    """Pitch metres (x, y) inside the penalty area at the 'left' or 'right' goal (either, side None)."""
    if abs(y-Wd/2) > PEN_HALF:
        return False
    return (side != 'right' and x <= PEN_DEPTH) or (side != 'left' and x >= L-PEN_DEPTH)


def _motion_matrix(m):
    """frames[].motion (the camera motion from the previous frame: a 2 x 3 affine in full-resolution pixels) as 3 x 3,
    or None. (Measured on the broadcast run 42-222 s: carrying a static ball candidate through it leaves a median .33 px
    against .62 px uncarried, 0.77 against 4.4 px while the camera pans.)"""
    if not (isinstance(m, (list, tuple)) and len(m) == 6 and all(_num(v) for v in m)):
        return None
    M = np.array([[m[0], m[1], m[2]], [m[3], m[4], m[5]], [0.0, 0.0, 1.0]])
    return M if abs(np.linalg.det(M)) > 1e-9 else None


def _live_candidate(f, b, W, H, pxm):
    """A ball candidate other than the final ball that is in play somewhere else: on the pitch (field score .9 or more),
    not a static object, moving at LIVE_SPEED or more, LIVE_AWAY image metres or more from the final ball (the ball
    tracker's own scores; a PitchIQ result only: the simulator has no candidates)."""
    if not _pair(b.get('center')):
        return False
    cx, cy = b['center'][0]*W, b['center'][1]*H
    for c in f.get('ballCandidates') or []:
        if not isinstance(c, dict) or c.get('status') == 'ball' or not _box(c.get('box')):
            continue
        s = c.get('scores') or {}
        if not (_num(s.get('field')) and s['field'] >= .9 and _num(s.get('static')) and s['static'] < .5
                and _num(s.get('speed')) and s['speed'] >= LIVE_SPEED):
            continue
        x, y, w, h = c['box']
        if math.hypot((x+w/2)*W-cx, (y+h/2)*H-cy)/pxm >= LIVE_AWAY:
            return True
    return False


def kit_overrides(frames, teams_of, fps):
    """fix 9: {(frame index, id): frame team} where a track's frame team contradicts its identity's roster team for
    KIT_S or more (KIT_GAP; a corrected or user-identified person never). PitchIQ writes the identity's team on every
    frame of a confirmed person (identity.py _style), so on the four run copies no frame contradicts the roster (0 of
    37882 / 111503 / 30988 / 5830 identified person-frames of BC60 / BC180 / AM / VEO): the rule is for a result whose
    frames carry their own team (a pipeline that keeps the kit per frame, or an edited result)."""
    seq = {}
    for i, f in enumerate(frames):
        t = float(f['time']) if _num(f.get('time')) else i/fps
        for p in f.get('people') or []:
            if not isinstance(p, dict):
                continue
            pid = p.get('id') if isinstance(p.get('id'), str) else None
            R = teams_of.get(pid)
            if R is None or p.get('manual') or p.get('correction') is not None:
                continue
            ft = p.get('team')
            if ft == R or (ft in TEAMS and str(p.get('label') or '').endswith('_TEAM_'+ft)):
                seq.setdefault((p.get('track'), pid), []).append((i, t, ft))
    out = {}
    for (_, pid), rs in seq.items():
        R = teams_of[pid]
        run = []
        for i, t, k in rs+[(None, math.inf, R)]:
            if k != R and (not run or t-run[-1][1] <= KIT_GAP):
                run.append((i, t, k))
                continue
            if run and run[-1][1]-run[0][1] >= KIT_S:
                for j in range(run[0][0], run[-1][0]+1):
                    out[(j, pid)] = run[0][2]
            run = [(i, t, k)] if k != R else []
    return out


def player_speeds(n, fps, rows, J, cam, camseg, lag):
    """(n, J) speed (m/s) of every team player over `lag` seconds centred on each frame (NaN where unknown): from his
    pitch positions where both ends are usable, else from his feet pixel carried through the camera motion between the
    two frames (same motion run only), in image metres (box height = PLAYER_H, image rows doubled as in
    ball_path.within_reach)."""
    out = np.full((n, J), np.nan)
    if not rows or J == 0:
        return out
    A = np.array(rows, np.float64)
    ii, jj = A[:, 0].astype(np.int64), A[:, 1].astype(np.int64)
    P = [np.full((n, J), np.nan) for _ in range(5)]
    for c in range(5):
        P[c][ii, jj] = A[:, 3+c]
    PX, PY, FX, FY, BH = P
    k = max(1, int(round(lag*fps)))
    if n <= k:
        return out
    a = np.arange(0, n-k)
    b = a+k
    pitch = np.hypot(PX[b]-PX[a], PY[b]-PY[a])/(k/fps)
    T = np.einsum('nij,njk->nik', cam[b], np.linalg.inv(cam[a]))
    xa = T[:, 0, 0, None]*FX[a]+T[:, 0, 1, None]*FY[a]+T[:, 0, 2, None]
    ya = T[:, 1, 0, None]*FX[a]+T[:, 1, 1, None]*FY[a]+T[:, 1, 2, None]
    with np.errstate(invalid='ignore'):
        pxm = np.fmax(BH[a], BH[b])/PLAYER_H
        img = np.hypot(FX[b]-xa, 2.0*(FY[b]-ya))/np.maximum(pxm, 1e-3)/(k/fps)
    img = np.where((camseg[a] == camseg[b])[:, None], img, np.nan)
    out[a+k//2] = np.where(np.isfinite(pitch), pitch, img)
    return out


def features(result, settings=None):
    """Per-frame arrays the decisions read: the ball (state, origin, motion, pitch metres, confidence, how far the first
    pass's ball was from the final one, its image ground point and scale, whether another ball candidate is in play
    elsewhere) and, per team, its player nearest the ball on the pitch, its player nearest in the image, and its
    CANDIDATES players nearest in the image with their pitch distance and position and whether each is the ball's
    nearTrack player (a carrier is one player near on both metrics); the nearest person who may be a player of either
    team (no team: an unresolved track, a goalkeeper the tracking has not placed); the goalkeeper holding the ball in his
    hands (HANDS_*); camera cuts and the camera motion; every team player's speed (player_speeds). A goalkeeper without a
    team takes the team defending the goal nearer him (tactical.goalEnds). With settings' kit_team (fix 9) a frame whose
    own team contradicts the identity's roster team plays for the frame's team (kit_overrides)."""
    S = dict(DEFAULTS)
    S.update(settings or {})
    frames = result.get('frames') or []
    video = result.get('video') or {}
    size = ((result.get('tactical') or {}).get('pitchSize') or {})
    L = float(size['length']) if _num(size.get('length')) and size['length'] > 0 else LENGTH
    Wd = float(size['width']) if _num(size.get('width')) and size['width'] > 0 else WIDTH
    W = float(video['width']) if _num(video.get('width')) and video['width'] > 0 else 1920.0
    H = float(video['height']) if _num(video.get('height')) and video['height'] > 0 else 1080.0
    fps = float(video['fps']) if _num(video.get('fps')) and video['fps'] > 0 else 25.0
    n = len(frames)
    teams_of = roster_teams(result)
    keeper_left, keeper_right = _keeper_team(result)
    kit = kit_overrides(frames, teams_of, fps) if S['kit_team'] else {}
    ids, index = [], {}

    def idx(pid):
        if pid not in index:
            index[pid] = len(ids)
            ids.append(pid)
        return index[pid]

    K = CANDIDATES
    t = np.zeros(n)
    fr = np.zeros(n, np.int64)
    bstate = np.zeros(n, np.int8)
    borig = np.zeros(n, np.int8)
    bmotion = np.zeros(n, np.int8)
    bx = np.full(n, np.nan)
    by = np.full(n, np.nan)
    bconf = np.zeros(n)
    bsig = np.full(n, np.nan)
    bimg = np.zeros(n, bool)
    gpx = np.full((n, 2), np.nan)        # the ball's ground point in pixels
    bpxm = np.full(n, np.nan)            # pixels per metre there (ball_path's scale)
    livec = np.zeros(n, bool)            # another ball candidate in play elsewhere (_live_candidate)
    small = np.zeros(n, bool)
    cut = np.zeros(n, bool)
    cam = np.tile(np.eye(3), (n, 1, 1))  # the camera motion from the first frame of the motion run to this frame
    camseg = np.zeros(n, np.int64)       # motion runs: a frame without a motion row, or a cut, starts a new one
    dP = np.full((2, n), np.inf)
    iP = np.full((2, n), -1, np.int64)
    dI = np.full((2, n), np.inf)
    iI = np.full((2, n), -1, np.int64)
    cP = np.full((2, K, n), np.inf)      # pitch distance of team k's c-th player nearest in the image (inf: no usable position)
    cI = np.full((2, K, n), np.inf)      # ... his image distance
    cJ = np.full((2, K, n), -1, np.int64)
    cN = np.zeros((2, K, n), bool)       # ... he is the ball's nearTrack player
    cX = np.full((2, K, n), np.nan)      # ... his pitch position, metres (NaN: none usable)
    cY = np.full((2, K, n), np.nan)
    uP = np.full(n, np.inf)              # the nearest person who may be a player of either team: pitch distance ...
    uI = np.full(n, np.inf)              # ... image distance
    uN = np.full(n, np.inf)              # image distance of the nearTrack person when he is such a person
    fpd = np.full(n, np.nan)             # image metres between the first pass's ball centre and the final one (NaN: none)
    fpd2 = np.full(n, np.nan)            # ... the same with the image rows doubled (ball_path.within_reach's metric)
    kH = np.full(n, -1, np.int8)         # the team of the goalkeeper with the ball in his hands (HANDS_*), -1 none ...
    kJ = np.full(n, -1, np.int64)        # ... and his id index
    kit_frames = 0                       # person-frames that play for their kit team (fix 9)
    rows = []                            # (frame, player, team, pitch x, y m, feet x, y px, box height px): player_speeds
    seg_prev = None
    for i, f in enumerate(frames):
        t[i] = float(f['time']) if _num(f.get('time')) else i/fps
        fr[i] = int(f['frame']) if _num(f.get('frame')) else int(round(t[i]*fps))
        small[i] = bool((f.get('profile') or {}).get('smallScene', False))
        seg = f.get('segment')
        cut[i] = i > 0 and (f.get('cut') is True or (seg is not None and seg_prev is not None and seg != seg_prev))
        seg_prev = seg if seg is not None else seg_prev
        M = _motion_matrix(f.get('motion')) if i > 0 and not cut[i] else None
        if M is not None:
            cam[i] = M@cam[i-1]
            camseg[i] = camseg[i-1]
        elif i > 0:
            camseg[i] = camseg[i-1]+1
        people = [p for p in (f.get('people') or []) if isinstance(p, dict) and _box(p.get('box'))]
        players = []
        for p in people:
            if p.get('state') in LOOSE_STATES or p.get('role') not in ('PLAYER', 'GOALKEEPER', 'UNKNOWN'):
                continue
            pid = p.get('id') if isinstance(p.get('id'), str) else None
            team = teams_of.get(pid) or p.get('team')
            if kit and (i, pid) in kit:
                team = kit[(i, pid)]
                kit_frames += 1
            pos = p.get('pitch')
            usable_pos = _pair(pos) and _num(p.get('pitchConfidence')) and p['pitchConfidence'] >= pitch_model.MEDIUM
            if team not in TEAMS and p.get('role') == 'GOALKEEPER' and keeper_left is not None and usable_pos:
                team = keeper_left if pos[0] < .5 else keeper_right
            if team in TEAMS and p.get('role') == 'UNKNOWN':
                team = None
            players.append((p, pid, team, pos, usable_pos))
            if team in TEAMS and pid is not None:
                bxp, byp, bw, bh = p['box']
                rows.append((i, idx(pid), TEAMS.index(team), pos[0]*L if usable_pos else np.nan, pos[1]*Wd if usable_pos else np.nan,
                             (bxp+bw/2)*W, (byp+bh)*H, bh*H))
        b = f.get('ball') or {}
        st = STATES.get(b.get('state'), 0)
        bstate[i] = st
        if st == 0:
            continue
        borig[i] = ORIGINS.get(b.get('origin'), 0)
        bmotion[i] = MOTIONS.get(b.get('motion'), 0)
        if _pair(b.get('pitch')):
            bx[i], by[i] = b['pitch'][0]*L, b['pitch'][1]*Wd
            bconf[i] = float(b['pitchConfidence']) if _num(b.get('pitchConfidence')) else 0.0
            sg = b.get('pitchSigma')
            if isinstance(sg, (list, tuple)) and sg and all(_num(v) for v in sg):
                bsig[i] = float(max(sg))
        gx = gy = None
        if _pair(b.get('center')):
            # the ball's ground point in pixels, as the ball audit measured it
            gx = b['center'][0]*W
            gy = (b['center'][1]+(b['box'][3]/2 if _box(b.get('box')) else 0.0))*H
            pxm = max(ball_path._pixels_per_metre(people, gy, H), 1e-3)   # the final ball path's own scale
            bimg[i] = True
            gpx[i] = gx, gy
            bpxm[i] = pxm
            livec[i] = _live_candidate(f, b, W, H, pxm)
            fc = (b.get('firstPass') or {}).get('center')
            if _pair(fc):
                fpd[i] = math.hypot((fc[0]-b['center'][0])*W, (fc[1]-b['center'][1])*H)/pxm
                fpd2[i] = math.hypot((fc[0]-b['center'][0])*W, 2.0*(fc[1]-b['center'][1])*H)/pxm
        near = b.get('nearTrack')
        cands = ([], [])
        others = []
        for p, pid, team, pos, usable_pos in players:
            dp = di = math.inf
            if np.isfinite(bx[i]) and usable_pos:
                dp = math.hypot(pos[0]*L-bx[i], pos[1]*Wd-by[i])
            bxp, byp, bw, bh = p['box']
            if gx is not None and bw > 0 and bh > 0:
                # ball_path.within_reach: feet against the ball, image rows foreshortened 2x, metres from player heights
                di = math.hypot(gx-(bxp+bw/2)*W, 2.0*(gy-(byp+bh)*H))/pxm
            is_near = near is not None and p.get('track') == near
            if team not in TEAMS:
                others.append((di, dp, is_near))
                continue
            k = 0 if team == 'A' else 1
            j = idx(pid)
            if dp < dP[k, i]:
                dP[k, i], iP[k, i] = dp, j
            if di < dI[k, i]:
                dI[k, i], iI[k, i] = di, j
            cands[k].append((di, dp, j, is_near, pos[0]*L if usable_pos else np.nan, pos[1]*Wd if usable_pos else np.nan))
            if (kH[i] < 0 and p.get('role') == 'GOALKEEPER' and usable_pos and bimg[i] and bw > 0 and bh > 0
                    and HANDS_U[0] <= (b['center'][0]-bxp)/bw <= HANDS_U[1] and HANDS_V[0] <= (b['center'][1]-byp)/bh < HANDS_V[1]):
                goal = None if keeper_left is None else ('left' if team == keeper_left else 'right')
                if _in_penalty_area(pos[0]*L, pos[1]*Wd, L, Wd, goal):
                    kH[i], kJ[i] = k, j    # the ball in his hands (above his lower body) in his penalty area
        for k in range(2):
            cands[k].sort(key=lambda c: c[0])
            for c, (di, dp, j, is_near, px_, py_) in enumerate(cands[k][:K]):
                cI[k, c, i], cP[k, c, i], cJ[k, c, i], cN[k, c, i], cX[k, c, i], cY[k, c, i] = di, dp, j, is_near, px_, py_
        for di, dp, is_near in others:
            if di < uI[i] or (not np.isfinite(uI[i]) and dp < uP[i]):
                uI[i], uP[i] = di, dp
            if is_near:
                uN[i] = di
    spd_team = np.full(len(ids), -1, np.int8)
    for row in rows:
        spd_team[row[1]] = row[2]
    return {'fps': fps, 'length': L, 'width': Wd, 'n': n, 't': t, 'frame': fr, 'bstate': bstate, 'borig': borig, 'bmotion': bmotion,
            'bx': bx, 'by': by, 'bconf': bconf, 'bsig': bsig, 'bimg': bimg, 'gpx': gpx, 'bpxm': bpxm, 'livec': livec, 'small': small,
            'cut': cut, 'cam': cam, 'camseg': camseg,
            'dP': dP, 'iP': iP, 'dI': dI, 'iI': iI, 'cP': cP, 'cI': cI, 'cJ': cJ, 'cN': cN, 'cX': cX, 'cY': cY, 'uP': uP, 'uI': uI, 'uN': uN,
            'fpd': fpd, 'fpd2': fpd2, 'kH': kH, 'kJ': kJ, 'kitFrames': kit_frames,
            'ids': ids, 'sizeSource': size.get('source'),
            'spd': player_speeds(n, fps, rows, len(ids), cam, camseg, ACT_LAG).astype(np.float32), 'spdTeam': spd_team}


def _runs(mask):
    """(start, end) inclusive index runs where mask is True."""
    m = np.asarray(mask, bool)
    if not m.any():
        return []
    d = np.diff(np.concatenate([[0], m.astype(np.int8), [0]]))
    return list(zip(np.flatnonzero(d == 1).tolist(), (np.flatnonzero(d == -1)-1).tolist()))


def _bridge_lengths(F):
    """Per frame, the duration of the bridged (MISSING) stretch it lies in (0 outside one)."""
    out = np.zeros(F['n'])
    for a, b in _runs(F['bstate'] == 2):
        out[a:b+1] = F['t'][b]-F['t'][a]+1.0/F['fps']
    return out


def usable_ball(F, S):
    """(pitch, image): frames whose ball pitch position may decide who is at it, and frames whose ball has at least an
    image position (TRACKED or a short bridge)."""
    seen = (F['bstate'] > 0) & ~((F['bstate'] == 2) & (_bridge_lengths(F) > S['long_bridge']))
    with np.errstate(invalid='ignore'):
        pitch = seen & np.isfinite(F['bx']) & (F['bconf'] >= S['min_conf']) & ~(F['bsig'] > S['max_sigma'])
    return pitch, seen & F['bimg']


def kinks(F, usable, w):
    """Per frame, the change of the ball's pitch velocity (m/s): over the w frames before against the w frames after (both
    ends usable and contiguous). A touch bends the ball's path; a ball flying past a player does not."""
    n, t, fps, bx, by = F['n'], F['t'], F['fps'], F['bx'], F['by']
    out = np.zeros(n)
    if n <= 2*w:
        return out
    i = np.arange(w, n-w)
    ok = usable[i-w] & usable[i] & usable[i+w] & (t[i+w]-t[i-w] <= (2*w+.5)/fps)
    dt0 = np.maximum(t[i]-t[i-w], 1e-6)
    dt1 = np.maximum(t[i+w]-t[i], 1e-6)
    with np.errstate(invalid='ignore'):
        dvx = (bx[i+w]-bx[i])/dt1-(bx[i]-bx[i-w])/dt0
        dvy = (by[i+w]-by[i])/dt1-(by[i]-by[i-w])/dt0
    out[i] = np.where(ok, np.nan_to_num(np.hypot(dvx, dvy)), 0.0)
    return out


def touch_peaks(F, S, usable, radius=3):
    """Frames at the peak of a bend of at least S['kink'] m/s (the maximum over +-radius frames)."""
    d = kinks(F, usable, S['kink_w'])
    n = len(d)
    pad = np.concatenate([np.zeros(radius), d, np.zeros(radius)])
    local = np.max(np.stack([pad[j:j+n] for j in range(2*radius+1)]), axis=0)
    return (d >= S['kink']) & (d >= local)


def touch_frames(F, S, usable, radius=3):
    """Frames at (or one frame beside) the peak of a bend of at least S['kink'] m/s: where a fast ball was touched."""
    peaks = touch_peaks(F, S, usable, radius)
    touch = peaks.copy()
    touch[1:] |= peaks[:-1]
    touch[:-1] |= peaks[1:]
    return touch


def _score(S, dp, di, usable, img_only, bridged, is_near):
    """Proximity score (0..1) of one person to the ball per frame: near on the pitch (r0..r1) AND in the image (img_r0..
    img_r1), the less near of the two; image only (image_only_w) where the ball or the person has no usable pitch
    position; the nearTrack person on a bridged frame (is_near: the caller has checked that the first pass followed this
    ball) at least near_w; 0 where the ball is not usable at all."""
    image = np.clip((S['img_r1']-di)/(S['img_r1']-S['img_r0']), 0.0, 1.0)
    if S['use_image']:
        pitch = np.clip((S['r1']-dp)/(S['r1']-S['r0']), 0.0, 1.0)
        both = np.minimum(pitch, image)
        s = np.where(usable & np.isfinite(dp), both, S['image_only_w']*image)
    else:
        s = np.where(usable, np.clip((S['r1']-dp)/(S['r1']-S['r0']), 0.0, 1.0), S['image_only_w']*image)
    if S['near_track']:
        s = np.where(bridged & is_near & (usable | img_only), np.maximum(s, S['near_w']), s)
    return np.where(usable | img_only, s, 0.0)


def _same_runs(mask, who):
    """(start, end) inclusive runs of mask with one value of who throughout."""
    out = []
    for a, b in _runs(mask):
        cut = (np.flatnonzero(who[a+1:b+1] != who[a:b])+a+1).tolist()
        for s0, s1 in zip([a]+cut, [c-1 for c in cut]+[b]):
            out.append((s0, s1))
    return out


def keeper_hands(F, seen):
    """(n,) the team index of a goalkeeper holding the ball in his hands (-1 none): the ball's centre above his lower
    body inside his box, him inside his penalty area (features(): HANDS_*), the same keeper on HANDS_MIN frames or more
    in a row with the ball seen."""
    out = np.full(F['n'], -1, np.int8)
    kH = F.get('kH')
    if kH is None:
        return out
    for k in range(2):
        for a, b in _same_runs((kH == k) & seen, F['kJ']):
            if b-a+1 >= HANDS_MIN:
                out[a:b+1] = k
    return out


def knock_past(F, S, usable, p, who, px, py, peaks):
    """(2, n) frames of a ball running past a player of team k (the player behind the team's score): over his contact
    with it (and a frame each side) the ball is TRACKED with a pitch position; it goes through, from where it was
    relative to him on the contact's first frame to where it is on its last it moved pass_m or more (a ball he stops,
    or keeps and moves with, stays where he is); no touch of his team falls inside (a bend peak, touch_peaks, with a
    player of his team at least as near the ball as the other team's nearest: a deflection, a first-time pass, a stop);
    and the next touch is the other team's, inside the contact or within pass_after seconds of its end (the ball ran on
    to them). A ball does not stop at a player by itself: one that slows down beside him without a bend and is then
    touched by the dribbler is still running past him (the synthetic knock-on dribble). A soft reception with no bend
    at all (Metrica's ball moving with its receiver at 3-4 m/s) is left his: no touch follows from the other team."""
    n, t = F['n'], F['t']
    tracked = usable & (F['bstate'] == 1)
    out = np.zeros((2, n), bool)
    for k in range(2):
        mine = peaks & (F['dP'][k] <= F['dP'][1-k])
        theirs = peaks & ~mine
        for c0, c1 in _same_runs(p[k] > 0, who[k]):
            if c1 == c0 or c0 < 1 or c1 >= n-1 or not tracked[c0-1:c1+2].all() or mine[c0-1:c1+2].any():
                continue
            hi = int(np.searchsorted(t, t[c1]+S['pass_after'], side='right'))
            if not theirs[c0:hi].any():
                continue
            rx = F['bx'][[c0, c1]]-px[k, [c0, c1]]
            ry = F['by'][[c0, c1]]-py[k, [c0, c1]]
            if np.isfinite(rx).all() and np.isfinite(ry).all() and math.hypot(rx[1]-rx[0], ry[1]-ry[0]) >= S['pass_m']:
                out[k, c0:c1+1] = True
    return out


def evidence(F, S, usable, seen, extra=None):
    """(2, n) seconds of evidence that team k has the ball on each frame, (2, n) the proximity scores behind it, (2, n)
    the id index of the player behind each score (-1 none) and (2, n) whether that score rests on both metrics (the ball
    and the player have pitch positions: where a change may start, decide()). A slow ball near one team's player and not the other's is
    evidence for that team: the player near on the pitch AND in the image (the score is the less near of the two, for
    the same player), the nearTrack player near on a bridged frame (when he is within reach of the bridged ball), less
    for a bridged frame; a FAST ball only at the peak of a bend (a touch), never as it flies past. Image-only evidence
    where the pitch position is unusable (a registration hole, or the player's own position) counts image_only_w. A
    person who may be a player of either team (no team yet) near the ball takes the evidence away as an opponent does.
    possession-4: a ball in a goalkeeper's hands (keeper_hands(), fix 3) is his, on both metrics and nobody else's (the
    ball is off the ground, so neither metric finds it at his feet); a ball running past a player (knock_past(), fix 4)
    is not his and he is not in the way. extra (a dict) receives the 'hands' and 'past' masks."""
    n, dt = F['n'], 1.0/F['fps']
    bridged = F['bstate'] == 2
    fast = F['bmotion'] == MOTIONS['FAST']
    touch = touch_frames(F, S, usable) if S['fast'] == 'peak' else np.zeros(n, bool)
    if S['fast'] == 'none':
        fast = np.zeros(n, bool)
    img_only = seen & ~usable
    with np.errstate(invalid='ignore'):
        nt_ok = ~(F['fpd'] > S['nt_agree'])   # the first pass followed this ball (or there is no first pass to ask)
    p = np.zeros((2, n))
    who = np.full((2, n), -1, np.int64)
    both = np.zeros((2, n), bool)        # the score rests on both metrics (the ball and the player have a pitch position)
    px = np.full((2, n), np.nan)         # the pitch position of the player behind the score
    py = np.full((2, n), np.nan)
    cols = np.arange(n)
    for k in range(2):
        if S['use_image']:
            sc = np.stack([_score(S, F['cP'][k, c], F['cI'][k, c], usable, img_only, bridged, F['cN'][k, c] & nt_ok) for c in range(CANDIDATES)])
            best = np.argmax(sc, axis=0)
            p[k] = sc[best, cols]
            who[k] = np.where(p[k] > 0, F['cJ'][k][best, cols], -1)
            both[k] = usable & np.isfinite(F['cP'][k][best, cols])
            if 'cX' in F:
                px[k], py[k] = F['cX'][k][best, cols], F['cY'][k][best, cols]
        else:   # the pitch metric alone (an ablation): the team's player nearest on the pitch
            p[k] = _score(S, F['dP'][k], F['dI'][k], usable, img_only, bridged, (F['cN'][k] & nt_ok).any(axis=0))
            who[k] = np.where(p[k] > 0, F['iP'][k], -1)
            both[k] = usable & np.isfinite(F['dP'][k])
    pu = np.zeros(n)
    if S['unlabelled']:
        pu = _score(S, F['uP'], F['uI'], usable, img_only, bridged, np.isfinite(F['uN']) & (F['uN'] == F['uI']) & nt_ok)
    hands = keeper_hands(F, seen) if S['keeper_hands'] else np.full(n, -1, np.int8)
    if S['knock_past'] and 'cX' in F:
        past = knock_past(F, S, usable, p, who, px, py, touch_peaks(F, S, usable))
        # only where he is alone at it: a contested frame stays contested (taking a carrier out of a duel would give the
        # ball to the challenger: on game 1's camera both changes the rule moved by more than 2 s were such duels)
        past &= (hands < 0)[None, :] & (p[::-1] == 0)
        p = np.where(past, 0.0, p)
        who = np.where(past, -1, who)
        both &= ~past
    else:
        past = np.zeros((2, n), bool)
    for k in range(2):
        m = hands == k
        if m.any():
            p[k] = np.where(m, 1.0, p[k])
            who[k] = np.where(m, F['kJ'], who[k])
            both[k] |= m
            p[1-k] = np.where(m, 0.0, p[1-k])
            who[1-k] = np.where(m, -1, who[1-k])
            both[1-k] &= ~m
    if (hands >= 0).any():
        pu = np.where(hands >= 0, 0.0, pu)
        fast = fast & (hands < 0)                 # a caught ball moves with the keeper, whatever its speed was
    e = np.zeros((2, n))
    for k in range(2):
        excl = p[k]*(1.0-p[1-k])*(1.0-pu)
        slow = np.where(bridged, S['bridge_w'], 1.0)*excl*dt
        e[k] = np.where(fast, np.where(touch, S['touch_w']*excl/3.0, 0.0), slow)   # a touch spans its peak and a frame each side
    if extra is not None:
        extra.update(hands=hands, past=past)
    return e, p, who, both


def confirmed_runs(cand, who, obs):
    """(2, n): the frames of every run of one team's evidence by one player (cand, the same who frame to frame) that
    contains an observed frame (obs): the ball hidden at the feet of a player who is then seen with it was his from the
    start of the run; a run never seen (an interpolated ball drifting to someone) is nobody's start."""
    out = np.zeros_like(cand)
    for k in range(cand.shape[0]):
        c = cand[k]
        if not c.any():
            continue
        brk = np.ones(len(c), bool)
        brk[1:] = ~c[:-1] | (who[k, 1:] != who[k, :-1])
        for a, b in _runs(c):
            starts = [a]+[int(j) for j in np.flatnonzero(brk[a+1:b+1])+a+1]
            for s0, s1 in zip(starts, starts[1:]+[b+1]):
                if obs[k, s0:s1].any():
                    out[k, s0:s1] = True
    return out


def _placed_end(F, S, usable, j):
    """From frame j, the last frame of a still stretch: the ball within still_r of the stretch's mean position (two
    frames outside in a row end it; unusable frames up to half a second are skipped)."""
    t, bx, by, n = F['t'], F['bx'], F['by'], F['n']
    e = last = j
    sx, sy, m, away = bx[j], by[j], 1, 0
    for i in range(j+1, n):
        if not usable[i]:
            if t[i]-t[last] > .5:
                break
            continue
        if math.hypot(bx[i]-sx/m, by[i]-sy/m) > S['still_r']:
            away += 1
            if away >= 2:
                break
            continue
        away = 0
        sx, sy, m = sx+bx[i], sy+by[i], m+1
        e = last = i
    return e


def _beyond(F):
    """Metres the ball lies beyond the nearest line (negative inside the pitch; NaN without a pitch position)."""
    L, Wd, bx, by = F['length'], F['width'], F['bx'], F['by']
    with np.errstate(invalid='ignore'):
        return np.fmax.reduce([-bx, bx-L, -by, by-Wd])


def _row_quantile(x, q, min_count):
    """Per row of x (NaN = missing) the q-quantile (linear) of its finite values, NaN with fewer than min_count."""
    n = x.shape[0]
    if x.shape[1] == 0:
        return np.full(n, np.nan)
    c = np.isfinite(x).sum(axis=1)
    srt = np.sort(np.where(np.isfinite(x), x, np.inf), axis=1)
    pos = np.maximum(c-1, 0)*q
    lo = np.floor(pos).astype(np.int64)
    hi = np.minimum(lo+1, np.maximum(c-1, 0))
    rows = np.arange(n)
    a, b = srt[rows, lo], srt[rows, hi]
    with np.errstate(invalid='ignore'):
        v = a+(np.where(np.isfinite(b), b, a)-a)*(pos-lo)
    return np.where(c >= max(1, min_count), v, np.nan)


def _centred_sum(x, w):
    """Per frame, the sum of x over a centred window of w frames (frames [i - w//2, i + (w-1)//2]), as long as x.
    (np.convolve(x, ones(w), 'same') returns max(len(x), w) values: a result shorter than the window, 30 frames or
    fewer at 30 fps for the 1 s smoothing, made possession-3 raise; the centred slice of the full convolution is
    bit-identical to 'same' wherever x is at least w long, checked on random floats and lengths 30-1000.)"""
    off = (w-1)//2
    return np.convolve(np.asarray(x, float), np.ones(w), 'full')[off:off+len(x)]


def _moving_mean(x, w):
    """Centred moving mean over w frames of the finite values of x (NaN where fewer than a third are finite)."""
    if w <= 1 or not len(x):
        return x
    ok = np.isfinite(x)
    num = _centred_sum(np.where(ok, x, 0.0), w)
    den = _centred_sum(ok.astype(float), w)
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(den >= w/3.0, num/np.maximum(den, 1e-9), np.nan)


def activity(F, S):
    """Per frame how fast the players move: the act_q quantile of the team players' speeds (player_speeds) over at least
    act_min of them, smoothed over act_smooth seconds; with act_teams 'both' the faster of the two teams' quantiles (both
    teams must have slowed down). NaN where too few players are seen."""
    n, fps = F['n'], F['fps']
    spd = F.get('spd')
    if spd is None or not spd.size:
        return np.full(n, np.nan)
    spd = spd.astype(np.float64)
    if S['act_teams'] == 'both':
        team = F['spdTeam']
        qs = [_row_quantile(spd[:, team == k], S['act_q'], max(2, S['act_min']//2)) for k in range(2)]
        v = np.where(np.isfinite(qs[0]) & np.isfinite(qs[1]), np.fmax(qs[0], qs[1]), np.nan)
    else:
        v = _row_quantile(spd, S['act_q'], S['act_min'])
    return _moving_mean(v, int(round(S['act_smooth']*fps)))


def still_ball(F, S, usable, lag=1.0, seen=None):
    """Per frame: the ball is where it was lag seconds before, within still_r: on the pitch where both frames have a
    usable position, else in the image (its ground point carried through the camera motion, image metres). With
    still_since (fix 7) and seen given, a ball not seen lag seconds before is compared with where it was first seen
    again since, or, in the first half lag after it came back into view, with where it is half a lag after that: a ball
    placed for a goal kick after the broadcast showed something else is still from the frame it comes into view
    (possession-3 compared it with the unseen frame, so the first second after it reappeared broke the stoppage:
    synthetic clip start, 3 s unseen then 3 s placed, no dead ball at all)."""
    n, fps = F['n'], F['fps']
    out = np.zeros(n, bool)
    k = max(1, int(round(lag*fps)))
    if n <= k:
        return out
    a = np.arange(k, n)
    b = a-k
    if S.get('still_since') and seen is not None:
        shown = usable | seen
        idx = np.flatnonzero(shown)
        if len(idx):
            pos = np.searchsorted(idx, b)
            nb = np.where(pos < len(idx), idx[np.minimum(pos, len(idx)-1)], n)
            since = ~shown[b] & (nb <= a)
            fwd = np.minimum(nb+k//2, n-1)
            early = since & (nb > a-k//2) & shown[fwd]
            b = np.where(early, fwd, np.where(since & (nb <= a-k//2), nb, b))
    with np.errstate(invalid='ignore'):
        d = np.hypot(F['bx'][a]-F['bx'][b], F['by'][a]-F['by'][b])
        pitch = usable[a] & usable[b] & (d < S['still_r'])
        gpx, cam = F.get('gpx'), F.get('cam')
        if gpx is None or cam is None:
            out[a] = pitch
            return out
        T = np.einsum('nij,njk->nik', cam[a], np.linalg.inv(cam[b]))
        x0, y0 = gpx[b, 0], gpx[b, 1]
        xa = T[:, 0, 0]*x0+T[:, 0, 1]*y0+T[:, 0, 2]
        ya = T[:, 1, 0]*x0+T[:, 1, 1]*y0+T[:, 1, 2]
        di = np.hypot(gpx[a, 0]-xa, 2.0*(gpx[a, 1]-ya))/F['bpxm'][a]
        img = ~(usable[a] & usable[b]) & (F['camseg'][a] == F['camseg'][b]) & (di < S['still_r'])
    out[a] = pitch | img
    return out


def excursions(F, S, usable):
    """(n,) bool: runs of the ball seen beyond a line under a trusted registration that are not a ball out of play but
    the ball path on something else (a ball boy's ball, a steward's, a marking beyond the line): the path did not get
    there from the pitch (from the last ball on the pitch within out_look seconds it would have flown faster than
    out_speed), or the first pass was following another ball on most of its frames (farther than nt_agree from the
    final one), or another ball candidate was in play on the pitch on most of them (_live_candidate). Those frames show
    no ball: possession is carried or unknown there, never dead."""
    n, t, fps = F['n'], F['t'], F['fps']
    bad = np.zeros(n, bool)
    beyond = _beyond(F)
    with np.errstate(invalid='ignore'):
        trusted = (F['bstate'] == 1) & usable & (F['bconf'] >= S['trusted']) & np.isfinite(beyond)
        out = trusted & (beyond > S['out_margin'])
        inside = usable & np.isfinite(beyond) & (beyond <= S['out_margin'])
    tr = np.flatnonzero(trusted)
    livec = F.get('livec')
    i = 0
    while i < len(tr):
        if not out[tr[i]]:
            i += 1
            continue
        k = i
        while k+1 < len(tr) and out[tr[k+1]]:
            k += 1
        p, q = int(tr[i]), int(tr[k])
        i = k+1
        why = None
        lo = int(np.searchsorted(t, t[p]-S['out_look']))
        prev = np.flatnonzero(inside[lo:p])
        if len(prev):
            j = lo+int(prev[-1])
            v = math.hypot(F['bx'][p]-F['bx'][j], F['by'][p]-F['by'][j])/max(t[p]-t[j], 1.0/fps)
            if v > S['out_speed']:
                why = 'jump'
        fp = F['fpd2' if S['fp_rows'] == 2.0 and 'fpd2' in F else 'fpd'][p:q+1]   # fix 6: rows doubled as within_reach
        fp = fp[np.isfinite(fp)]
        if why is None and len(fp) and (fp > S['nt_agree']).mean() > .5:
            why = 'first pass'
        if why is None and livec is not None and livec[p:q+1][trusted[p:q+1]].mean() > .5:
            why = 'live candidate'
        if why is not None:
            bad[p:q+1] = True
    return bad


def live_holds(F, S, usable, e, who, both, spells, gaps, stops=(), bad=None):
    """(n,) fix 2: frames where a team is in control of a ball it got in play. Control: one team's evidence on both
    metrics with the other team not at the ball, lapses of up to hold_gap seconds (the other team still not at it; an
    excursion's frames, bad, do not count) joined, passes between team-mates included. It got the ball in play unless,
    within hold_look seconds before its control (or during it), the ball was out of play (spells: the out-of-play spells
    found so far), unseen for gap seconds or the camera cut (gaps), or it jumped to the player faster than out_speed (a
    new ball), or it lay still within still_r with nobody in control (a placed ball he walked up to), or the control
    began in a stoppage that had lasted hold_look already (stops, possession-3's still-ball and slowdown stoppages): the
    restart pattern, dead until the ball is played. A control that starts less than hold_look after the clip's first
    frame is judged from its first reception by a team-mate after that (none: not counted); one with the ball placed on
    the pitch on less than half the hold_look before it cannot tell and is not counted. Possession-3 made such controls
    dead:
    a player or a keeper holding the ball still for 4 s (the still-ball rule), or carrying it slowly while everyone
    walked (the slowdown rule): 86.6 s of game 1's in-play time on full tracking, 81.6 s through the simulated camera."""
    n, t, fps = F['n'], F['t'], F['fps']
    out = np.zeros(n, bool)
    if not n:
        return out
    stop = np.zeros(n, bool)
    for s in spells:
        stop[s['a']:min(s['r'], n)] = True
    for a, b in gaps:
        stop[a:b+1] = True
    if bad is not None:
        stop &= ~bad   # the ball path on another object while play went on around the real ball (excursions): no cue
    gapf = int(round(S['hold_gap']*fps))
    look = S['hold_look']
    anyheld = ((e[0] > 0) & both[0] & ~(e[1] > 0)) | ((e[1] > 0) & both[1] & ~(e[0] > 0))
    nb = np.concatenate([[0], np.cumsum(~bad)]) if bad is not None else np.arange(n+1)

    def in_play(h0, h1):
        """The control from h0 began in play (the lookback's checks above)."""
        lo = int(np.searchsorted(t, t[h0]-look))
        if stop[lo:h1+1].any():
            return False
        prev = np.flatnonzero(usable[lo:h0])+lo
        if 2*len(prev) < h0-lo or not len(prev):
            return False        # the ball mostly without a pitch position before: cannot tell
        j = int(prev[-1])
        if math.hypot(F['bx'][h0]-F['bx'][j], F['by'][h0]-F['by'][j])/max(t[h0]-t[j], 1.0/fps) > S['out_speed']:
            return False        # it jumped to him: a new ball
        if not anyheld[lo:h0].any():
            bx, by = F['bx'][prev], F['by'][prev]
            if np.hypot(bx-bx.mean(), by-by.mean()).max() <= S['still_r']:
                return False    # a placed ball nobody held, that he walked up to
        return not any(t[s['a']] <= t[h0]-look and s['r'] >= lo for s in stops)

    for k in range(2):
        held = (e[k] > 0) & both[k] & ~(e[1-k] > 0)
        idx = np.flatnonzero(held)
        if not len(idx):
            continue
        other = np.concatenate([[0], np.cumsum(e[1-k] > 0)])          # other[i]: frames before i with the other team at it
        lapse = nb[idx[1:]]-nb[idx[:-1]+1]
        other_in = other[idx[1:]]-other[idx[:-1]+1]
        brk = np.flatnonzero((lapse > gapf) | (other_in > 0))+1
        for s0, s1 in zip(np.concatenate([[0], brk]), np.concatenate([brk, [len(idx)]])):
            h0, h1 = int(idx[s0]), int(idx[s1-1])
            if t[h0]-t[0] < look:
                # the clip starts in this control: judge from its first reception by a team-mate far enough in
                run = idx[s0:s1]
                mate = who[k][run]
                later = [int(run[j]) for j in np.flatnonzero(mate[1:] != mate[:-1])+1 if t[run[j]]-t[0] >= look]
                if not later:
                    continue
                h0 = later[0]
            if in_play(h0, h1):
                out[h0:h1+1] = True
    return out


def boundaries(F, S, usable, seen, e, act=None, bad=None, both=None, extra=None, who=None):
    """Where play stopped or the ball was lost from view, so that possession restarts with the first touch after it:
    dead-ball spells and long unseen stretches. Each is {'a': first frame, 'r': first frame after it (the restart),
    'kind': 'dead'|'gap', 'reason': 'out_of_play'|'stoppage'|None, 'restart': team index of the restart's taker or None}.
    Out of play: the ball beyond a line by out_margin under a trusted registration for out_min seconds (or on its last
    sighting before a long unseen stretch), not back on the pitch within out_confirm; dead from the crossing to the
    restart (the excursions() that are not consistent were made no ball before: bad, never a stoppage cue either). A
    stoppage: the ball still for still_s seconds (a ball placed for a free kick), with a long unseen stretch just before
    it; dead to the kick. Or, with the players' activity act (activity()), both teams slower than slow_v for slow_s
    seconds while the ball is not seen, stands still (still_ball()) or nobody plays it (no evidence within slow_play,
    the ball not FAST; with slow_played 'pitch' only evidence on both metrics, both, counts); dead from the stretch's
    start to the touch, or the ball seen again, that ends it. The restart is when a placed ball leaves its spot, else
    when the ball is first seen back on the pitch. possession-4: neither stoppage rule fires on a ball a team holds in
    play (live_holds(), fix 2; who: the players behind the evidence), a still ball after an unseen stretch is compared
    with where it was seen again (still_ball(), fix 7), and every boundary says whether a camera cut lies in it ('cut',
    fix 8). extra (a dict) receives the 'live' mask."""
    n, t, fps = F['n'], F['t'], F['fps']
    beyond = _beyond(F)
    with np.errstate(invalid='ignore'):
        trusted = (F['bstate'] == 1) & usable & (F['bconf'] >= S['trusted']) & np.isfinite(beyond)
        out = trusted & (beyond > S['out_margin'])
        inside = usable & np.isfinite(beyond) & (beyond <= S['out_margin'])
    gaps = [(a, b) for a, b in _runs(~(usable | seen)) if t[b]-t[a]+1.0/fps >= S['gap']]
    # a camera cut (frames[].cut, or a new frames[].segment) is a gap however short: what follows may be another view
    # or a replay, so possession restarts with the first touch after it (none of the four PitchIQ runs has one)
    cuts = np.flatnonzero(F['cut']) if F.get('cut') is not None else []
    gaps = sorted(gaps+[(int(c), int(c)) for c in cuts])
    gap_starts = np.array([a for a, _ in gaps], np.int64)
    spells = []

    def restart_of(after):
        js = np.flatnonzero(inside[after:])
        if not len(js):
            return n, None
        j = after+int(js[0])
        end = _placed_end(F, S, usable, j)
        return (end+1 if t[end]-t[j] >= S['placed_s'] else j), j

    def crossing(p):
        a = p
        while a-1 >= 0 and t[p]-t[a-1] <= 1.0 and not (np.isfinite(beyond[a-1]) and beyond[a-1] <= 0):
            a -= 1
        return a

    tr = np.flatnonzero(trusted)
    i = 0
    while i < len(tr):
        if not out[tr[i]]:
            i += 1
            continue
        k = i
        while k+1 < len(tr) and out[tr[k+1]]:
            k += 1
        p, q = int(tr[i]), int(tr[k])
        i = k+1
        lost_after = bool(len(gap_starts)) and bool(((gap_starts > q) & (gap_starts <= q+.5*fps)).any())
        if t[q]-t[p]+1.0/fps < S['out_min'] and not lost_after:
            continue
        r, j = restart_of(q+1)
        if j is not None and t[j]-t[p] < S['out_confirm']:
            continue   # back on the pitch at once: a lofted ball projected beyond the line, not a ball out of play
        spells.append({'a': crossing(p), 'r': int(r), 'kind': 'dead', 'reason': 'out_of_play'})
        while i < len(tr) and tr[i] < r:
            i += 1
    busy = np.zeros(n+1, bool)
    for s in spells:
        busy[s['a']:s['r']] = True
    has_any = (e > 0).any(axis=0)
    h = max(0, int(round(S['slow_play']*fps)))
    shown = usable | seen
    slowdown = act is not None and math.isfinite(S['slow_s'])
    slow = cue = None
    if slowdown or S['live_hold']:
        # the slowdown rule's ball cues: nobody plays the ball (it is not seen, it stands still, or no player is at it)
        at = ((e > 0) & both).any(axis=0) if S['slow_played'] == 'pitch' and both is not None else has_any
        played = _centred_sum(at.astype(float), 2*h+1) > 0
        # (a ball flying fast was just kicked: that is play, not a ball nobody plays)
        cue = ~shown | still_ball(F, S, usable, seen=seen) | (~played & (F['bmotion'] != MOTIONS['FAST']))
        if bad is not None:
            cue &= ~bad    # the ball path on another object while play goes on around the real ball: no stoppage cue
    if slowdown:
        with np.errstate(invalid='ignore'):
            slow = act < S['slow_v']   # ... while both teams slowed down

    def stoppages(live, busy):
        """The still-ball and slowdown stoppages outside busy, never on a ball mostly held in play (live)."""
        found = []
        i = 0
        while i < n:
            if not usable[i] or busy[i]:
                i += 1
                continue
            end = _placed_end(F, S, usable, i)
            if t[end]-t[i] >= S['still_s'] and not busy[i:end+1].any() and not (live[i:end+1].mean() >= S['hold_share']):
                a, h_ = i, i-1
                while h_ >= 0 and not (usable[h_] or seen[h_]) and not busy[h_] and not (S['live_hold'] and bad is not None and bad[h_]):
                    h_ -= 1
                if h_ < i-1 and t[i-1]-t[h_+1] >= S['gap']:
                    a = h_+1   # the unseen stretch before the placed ball is the same stoppage
                found.append({'a': int(a), 'r': int(end+1), 'kind': 'dead', 'reason': 'stoppage'})
                busy[a:end+1] = True
                i = end+1
            else:
                i += 1
        if slowdown:
            # a stoppage with the ball in view: dead from the last touch before, to the first touch after
            for a, b in _runs(slow & cue & ~live & ~busy[:n]):
                if t[b]-t[a]+1.0/fps < S['slow_s']:
                    continue
                lo = int(np.searchsorted(t, t[a]-S['slow_back']))
                prior = np.flatnonzero(has_any[lo:a] | busy[lo:a])
                a0 = lo+int(prior[-1])+1 if len(prior) else lo
                # the restart: the touch (or the ball seen again) that ended it: from slow_play before the run's end,
                # since 'played' looks that far ahead; a kick from a ball seen again has no evidence of its own
                j0 = max(a+1, b+1-h)
                hi = int(np.searchsorted(t, t[b]+S['restart_look'], side='right'))
                nxt = np.flatnonzero(has_any[j0:hi] | (shown[j0:hi] & ~shown[j0-1:hi-1]))
                r = j0+int(nxt[0]) if len(nxt) else b+1
                found.append({'a': int(a0), 'r': int(r), 'kind': 'dead', 'reason': 'stoppage'})
                busy[a0:r] = True
        return found

    # (fix 2) a player in control of the ball he got in play is play, however slowly he and everyone else move: the
    # stoppages possession-3 would find tell which controls began with the ball dead (live_holds), then they are found
    # again without the controls that began in play
    live = np.zeros(n, bool)
    if S['live_hold'] and both is not None:
        live = live_holds(F, S, usable, e, who if who is not None else np.zeros(e.shape, np.int64), both, spells, gaps,
                          stoppages(live, busy.copy()), bad)
    if extra is not None:
        extra['live'] = live
    spells += stoppages(live, busy)
    cutset = set(int(c) for c in cuts)
    for a, b in gaps:
        if not busy[a:b+1].any():
            spells.append({'a': int(a), 'r': int(b+1), 'kind': 'gap', 'reason': None})
    for s in spells:
        s['cut'] = any(s['a'] <= c < s['r'] for c in cutset)   # (fix 8) the view changed inside it
    spells.sort(key=lambda s: s['a'])
    merged = []
    for s in spells:
        if merged and s['a'] <= merged[-1]['r']:
            m = merged[-1]
            m['r'] = max(m['r'], s['r'])
            m['cut'] = m['cut'] or s['cut']
            if m['kind'] == 'gap' and s['kind'] == 'dead':
                m.update(kind='dead', reason=s['reason'])
        else:
            merged.append(dict(s))
    # the restart's taker: the player nearest the ball (within taker_r) as it is kicked or first seen again, else the
    # first team at the ball within restart_look
    has = (e[0] > 0) | (e[1] > 0)
    near = np.fmin(F['dP'][0], F['dP'][1])
    for s in merged:
        s['restart'] = None
        r = s['r']
        if r >= n:
            continue
        for i in range(max(0, s['a'], r-2), min(n, r+int(S['restart_s']*fps)+1)):   # never from before a camera cut
            if (usable[i] or seen[i]) and near[i] <= S['taker_r']:
                s['restart'] = 0 if F['dP'][0, i] <= F['dP'][1, i] else 1
                break
        if s['restart'] is None:
            b = int(np.searchsorted(t, t[r]+S['restart_look'], side='right'))
            js = np.flatnonzero(has[r:b])
            if len(js):
                j = r+int(js[0])
                s['restart'] = 0 if e[0, j] >= e[1, j] else 1
    return merged


def viterbi(e, C, lo, hi, ok=None):
    """Two-state decode of possession over frames [lo, hi): staying with team k costs the other team's evidence, a
    switch costs C and is allowed only on a frame where the new team is at the ball (ok: where a change may start;
    default any evidence), either start is free. A change therefore lands on the winner's first touch: the convention,
    the ball belongs to the losing team until then. Returns (hi-lo,) int8."""
    m = hi-lo
    out = np.zeros(max(m, 0), np.int8)
    if m <= 0:
        return out
    D = np.zeros((m, 2))
    eA, eB = e[0, lo:hi].tolist(), e[1, lo:hi].tolist()
    if ok is None:
        okA, okB = [v > 0 for v in eA], [v > 0 for v in eB]
    else:
        okA, okB = ok[0, lo:hi].tolist(), ok[1, lo:hi].tolist()
    inf = math.inf
    a, b = eB[0], eA[0]
    D[0] = a, b
    for j in range(1, m):
        # a team takes the ball over only on a frame where it is at the ball (observed: ok)
        ca = C if okA[j] else inf
        cb = C if okB[j] else inf
        a, b = min(a, b+ca)+eB[j], min(b, a+cb)+eA[j]
        D[j, 0], D[j, 1] = a, b
    s = int(np.argmin(D[m-1]))
    for j in range(m-1, -1, -1):
        out[j] = s
        if j > 0:
            c = C if (okA if s == 0 else okB)[j] else inf
            if D[j-1, 1-s]+c < D[j-1, s]-1e-12:
                s = 1-s
    return out


def posterior(e, C, lo, hi, tau, ok=None):
    """Forward-backward over viterbi()'s model with the costs read as tau * -log-likelihood: per frame of [lo, hi), the
    probability that team 0 has the ball (ok as in viterbi())."""
    m = hi-lo
    if m <= 0:
        return np.zeros(0)
    LA, LB = (-e[1, lo:hi]/tau).tolist(), (-e[0, lo:hi]/tau).tolist()
    okA = (e[0, lo:hi] > 0) if ok is None else ok[0, lo:hi]
    okB = (e[1, lo:hi] > 0) if ok is None else ok[1, lo:hi]
    SA = [-C/tau if v else -math.inf for v in okA.tolist()]   # log weight of a switch to team 0 on the frame
    SB = [-C/tau if v else -math.inf for v in okB.tolist()]

    def lse(x, y):
        if x == -math.inf:
            return y
        if y == -math.inf:
            return x
        return (x if x > y else y)+math.log1p(math.exp(-abs(x-y)))

    fa, fb = [0.0]*m, [0.0]*m
    fa[0], fb[0] = LA[0], LB[0]
    for j in range(1, m):
        a, b = fa[j-1], fb[j-1]
        fa[j] = lse(a, b+SA[j])+LA[j]
        fb[j] = lse(b, a+SB[j])+LB[j]
    ba, bb = [0.0]*m, [0.0]*m
    for j in range(m-2, -1, -1):
        na, nb = ba[j+1]+LA[j+1], bb[j+1]+LB[j+1]
        ba[j] = lse(na, nb+SB[j+1])
        bb[j] = lse(nb, na+SA[j+1])
    d = (np.array(fb)+np.array(bb))-(np.array(fa)+np.array(ba))
    return 1.0/(1.0+np.exp(np.clip(d, -50, 50)))


def _player_at(F, D, k, i):
    """The id index of team k's player at the ball on frame i: the player behind the team's proximity score (near on
    both metrics, or the nearTrack player within reach on a bridged frame), else the nearest on the pitch, else in the
    image (-1: none). (Stage A took the nearest on the pitch and the nearTrack player on every bridged frame: on the
    broadcast's first minute the carrier then flickered between team-mates 3 to 10 m from the ball.)"""
    if D['who'][k, i] >= 0:
        return int(D['who'][k, i])
    if F['iP'][k, i] >= 0:
        return int(F['iP'][k, i])
    return int(F['iI'][k, i])


def replays(F, S, usable, look=1.0, settle=.2):
    """(n,) fix 8: the frames of a possible replay, a stretch between two camera cuts at most replay_max long where the
    ball's pitch position jumps at both cuts (from its last position within look seconds before the cut to one within
    settle seconds after it, faster than out_speed: the view shows another moment) and after the second cut it is back
    nearer where it was before the first than where it was before the second (the broadcast went back to live play).
    Its frames show no ball: unknown, never a team's. A replay the registration cannot place (no usable ball on either
    side of a cut) is not found, nor one shown during a stoppage whose ball does not move."""
    n, t, fps = F['n'], F['t'], F['fps']
    out = np.zeros(n, bool)
    cut = F.get('cut')
    if cut is None or cut.sum() < 2:
        return out
    bx, by = F['bx'], F['by']

    def last(lo, hi):
        js = np.flatnonzero(usable[lo:hi])
        return lo+int(js[-1]) if len(js) else None

    def dist(i, j):
        return math.hypot(bx[j]-bx[i], by[j]-by[i])

    def jumped(c, ref, hi):
        """The first usable frame from c (to settle after it, before hi) the ball could not have reached from ref."""
        for j in range(c, min(hi, int(np.searchsorted(t, t[c]+settle, side='right')))):
            if usable[j] and dist(ref, j)/max(t[j]-t[ref], 1.0/fps) > S['out_speed']:
                return j
        return None

    cs = np.flatnonzero(cut).tolist()
    for c1, c2 in zip(cs, cs[1:]):
        if t[c2]-t[c1] > S['replay_max']:
            continue
        b1 = last(int(np.searchsorted(t, t[c1]-look)), c1)
        j1 = jumped(c1, b1, c2) if b1 is not None else None
        b2 = last(j1, c2) if j1 is not None else None
        j2 = jumped(c2, b2, n) if b2 is not None else None
        if j2 is not None and dist(b1, j2) < dist(b2, j2):
            out[c1:c2] = True
    return out


def decide(F, settings=None):
    """The possession decisions on features F: per frame the team code (0 dead, 1 A, 2 B, -1 unknown) before and after
    the posterior gate, the posterior of the decoded team, the evidence, where a change may start (ok), the players'
    activity, the excursions; the boundaries; the changes (frame indices)."""
    S = dict(DEFAULTS)
    S.update(settings or {})
    n, t, fps = F['n'], F['t'], F['fps']
    usable, seen = usable_ball(F, S)
    bad = excursions(F, S, usable) if S['out_consistent'] else np.zeros(n, bool)
    rep = replays(F, S, usable & ~bad) if S['replays'] else np.zeros(n, bool)
    usable, seen = usable & ~bad & ~rep, seen & ~bad & ~rep
    extra = {}
    e, p, who, both = evidence(F, S, usable, seen, extra)
    act = activity(F, S) if math.isfinite(S['slow_s']) else None
    bounds = boundaries(F, S, usable, seen, e, act, bad, both, extra, who)
    for b in bounds:
        b['replay'] = bool(rep[b['a']:b['r']].any())
    # the restart's taker: a set piece is a single kick (a goal kick leaves the keeper on the first frame it is seen)
    if S['restart_w'] > 0:
        for b in bounds:
            r = b['r']
            js = np.flatnonzero((usable | seen)[r:min(n, r+int(S['restart_s']*fps)+1)])
            if not len(js):
                continue
            i = r+int(js[0])
            if b['restart'] is not None and (b['kind'] == 'dead' or S['restart_gap']):
                e[b['restart'], i] += S['restart_w']
    has = e > 0
    # a change rests on observed evidence of the new team: a TRACKED ball of OBSERVATION origin, near the same player on
    # both metrics; interpolated, weak or image-only frames may extend a control but never start one
    borig = F.get('borig')
    tracked = F['bstate'] == 1
    # (fix 5) a result that records no ball.origin at all (every run made before 2026-10-15: 45 of the user's 64): a
    # TRACKED ball counts as observed, with a quality note (possession-3 started no change there)
    no_origin = S['origin_fallback'] and bool(tracked.any()) and (borig is None or not (borig[tracked] > 0).any())
    if S['start_on'] != 'any':
        if S['start_on'] in ('observed', 'confirmed'):
            observed = tracked & ((borig == ORIGINS['OBSERVATION']) if borig is not None and not no_origin else True)
        elif S['start_on'] == 'tracked':
            observed = F['bstate'] == 1
        else:
            observed = np.ones(n, bool)
        ok = has & both & observed[None, :]
        if S['start_on'] == 'confirmed':
            ok = confirmed_runs(has & both, who, ok)
    else:
        ok = has
    ed = np.where(ok, e, 0.0) if S['decode_on'] == 'observed' else e
    raw = np.full(n, -1, np.int8)
    post = np.zeros(n)
    changes = []
    P_prev, last_prev = None, None
    edges = [None]+bounds+[None]
    for bi in range(len(edges)-1):
        b0, b1 = edges[bi], edges[bi+1]
        s0 = b0['r'] if b0 is not None else 0
        s1 = b1['a'] if b1 is not None else n
        if s0 >= n or s1 <= s0:
            continue
        st = viterbi(ed, S['switch'], s0, s1, ok)
        if S['posterior_on'] == 'any':
            pa = posterior(e, S['switch'], s0, s1, S['tau'])     # how sure: a change could have started on any evidence
        else:
            pa = posterior(ed, S['switch'], s0, s1, S['tau'], ok)
        post[s0:s1] = np.where(st == 0, pa, 1.0-pa)
        ev = np.flatnonzero(has[0, s0:s1] | has[1, s0:s1])
        if not len(ev):
            if not S['keep_prev']:
                P_prev, last_prev = None, None   # (v2) a stretch where nobody is at the ball forgets the possession
            continue                             # between two boundaries nobody took the ball over: it is still P_prev's
        first = s0+int(ev[0]) if b0 is None else s0   # the clip's first frames are unknown until the first evidence
        P0 = int(st[ev[0]])
        # (fix 8) after a camera cut (or a replay) a team is named only from an observed control on: what follows may
        # be another moment, and nobody saw the ball change hands over the cut
        fresh = S['cut_fresh'] and b0 is not None and (b0.get('cut') or b0.get('replay'))
        if fresh:
            okk = np.flatnonzero(ok[0, s0:s1] | ok[1, s0:s1])
            if not len(okk):
                continue
            first = s0+int(okk[0])
            P0 = int(st[first-s0])
        if (b0 is not None and b0['kind'] == 'gap' and S['carry_gap'] and S['carry_by'] == 'decoded' and P_prev is not None and P0 == P_prev
                and not fresh):
            raw[b0['a']:b0['r']] = P0+1   # the same team decoded on both sides of an unseen stretch keeps the ball
            post[b0['a']:b0['r']] = S['min_posterior']
        if b0 is not None and P_prev is not None and P0 != P_prev:
            changes.append({'won': first if fresh else s0, 'lost': last_prev if last_prev is not None else b0['a'],
                            'lostSeen': last_prev is not None, 'from': P_prev, 'to': P0, 'kind': 'restart', 'boundary': b0})
        raw[first:s1] = st[first-s0:]+1
        for j in (np.flatnonzero(np.diff(st))+1).tolist():
            i = s0+j
            if i < first:
                continue
            old, new = int(st[j-1]), int(st[j])
            prior = np.flatnonzero(has[old, s0:i])
            lost = s0+int(prior[-1]) if len(prior) else None
            changes.append({'won': i, 'lost': lost if lost is not None else s0, 'lostSeen': lost is not None,
                            'from': old, 'to': new, 'kind': 'open', 'boundary': None})
        P_prev = int(st[-1])
        prior = np.flatnonzero(has[P_prev, s0:s1])
        last_prev = s0+int(prior[-1]) if len(prior) else None
        if (b1 is not None and b1['kind'] == 'gap' and S['carry_gap'] and S['carry_by'] == 'taker' and b1['restart'] == P_prev
                and not (S['cut_fresh'] and (b1.get('cut') or b1.get('replay')))):
            raw[b1['a']:b1['r']] = P_prev+1   # the same team on both sides of an unseen stretch keeps the ball
            post[b1['a']:b1['r']] = S['min_posterior']
    carried = np.zeros(n, bool)
    for b in bounds:
        if b['kind'] == 'dead':
            raw[b['a']:b['r']] = 0
        elif S['carry_gap']:
            carried[b['a']:b['r']] = raw[b['a']:b['r']] > 0
    code = np.where((raw > 0) & (post < S['min_posterior']) & ~carried, -1, raw).astype(np.int8)
    D = {'S': S, 'raw': raw, 'code': code, 'post': post, 'e': e, 'p': p, 'who': who, 'usable': usable, 'seen': seen,
         'carried': carried, 'bounds': bounds, 'changes': changes, 'ok': ok, 'act': act, 'excursion': bad, 'replay': rep,
         'noOrigin': no_origin, 'hands': extra.get('hands'), 'past': extra.get('past'), 'live': extra.get('live')}
    wins = sorted(c['won'] for c in changes)
    for c in changes:
        c['lostBy'] = _player_at(F, D, c['from'], c['lost']) if c['lostSeen'] else -1
        hi = min(n, c['won']+int(round(S['won_w']*fps)))
        later = [w for w in wins if w > c['won']]
        hi = min(hi, later[0]) if later else hi
        nxt = np.flatnonzero(e[c['to'], c['won']:hi] > 0)
        c['wonBy'] = _player_at(F, D, c['to'], c['won']+int(nxt[0])) if len(nxt) else -1
        if S['won_by'] == 'most' and len(nxt):
            # the player who then holds the ball: the most evidence of the new team before the next change
            w = {}
            for i in (c['won']+nxt).tolist():
                j = int(who[c['to'], i])
                if j >= 0:
                    w[j] = w.get(j, 0.0)+float(e[c['to'], i])
            if w:
                c['wonBy'] = max(w, key=w.get)
    return D


# ---------------------------------------------------------------------------------------------- the record

LEVELS = ('low', 'medium', 'high')    # a frame's (a timeline run's) confidence
CHANGE_LEVELS = ('uncertain', 'high')   # a change's (fix 10): v3's medium and low were not ordered on game 1 (pooled over
                                        # the five variants, +-2 s against every change: medium 157 of 234 right, 67%, low
                                        # 168 of 228, 74%; +-1 s 62% / 61%; high 444 of 474, 94%), so they are one level
HIGH_POSTERIOR = .95   # a frame or change whose posterior reaches this is 'high' confidence; from min_posterior 'medium'
STATE_NAMES = ('control', 'flight', 'loose', 'contested', 'dead', 'unknown')
CARRIER_MIN = 4        # frames: a carrier's run shorter than this inside a control stretch is the carrier before it
LOW_EVIDENCE = .25     # s of the new team's evidence over the 2 s after an open change below which it is 'low' ...
HIGH_EVIDENCE = .5     # ... from which it is 'high' (game 1, camera: precision .66 below .21 s, about .95 from .48 s) ...
DUEL_GAP = .2          # ... unless the loser touched the ball within this many seconds of the gain (a duel: .74)
PAIR_S = 2.0           # s: a change undone within this (or undoing one made this before) is 'low' (change_levels)
HIGH_LOST = 5.0        # s: a 'high' change's loser touched the ball at most this before the gain ...
HIGH_HOLD = 2.0        # ... and the new team kept it this long
GOOD_COVERAGE = .6     # a run naming a team on less of its live time is 'limited' ...
GOOD_AGREEMENT = .85   # ... as is one whose pitch-vs-image agreement is below this (the broadcast measured .87)


def _level(post, S):
    """0 low, 1 medium, 2 high from a posterior."""
    return np.where(post >= HIGH_POSTERIOR, 2, np.where(post >= S['min_posterior'], 1, 0))


def agreement(F, S, usable):
    """The ball audit's pitch-vs-image one-team agreement on observed frames: of the frames where either metric puts
    exactly one team within agree_r of the ball, the share where both put the same one. (share or None, frames)."""
    obs = (F['bstate'] == 1) & usable & F['bimg']
    lab = []
    for d in (F['dP'], F['dI']):
        a, b = d[0] <= S['agree_r'], d[1] <= S['agree_r']
        lab.append(np.where(a & ~b, 1, np.where(b & ~a, 2, np.where(a & b, 3, 0))))
    lp, li = lab[0][obs], lab[1][obs]
    one = (lp == 1) | (lp == 2) | (li == 1) | (li == 2)
    m = int(one.sum())
    if not m:
        return None, 0
    return float(((lp == li) & one).sum()/m), m


def frame_states(F, S, D):
    """Per frame: the state (an index of STATE_NAMES), the carrier (id index, -1), estimated, the confidence level."""
    n = F['n']
    code, e, p, post = D['code'], D['e'], D['p'], D['post']
    has = (e[0] > 0) | (e[1] > 0)
    lead = np.where(e[0] >= e[1], 0, 1)
    state = np.full(n, STATE_NAMES.index('unknown'), np.int8)
    carrier = np.full(n, -1, np.int64)
    loose = np.zeros(n, bool)
    for c in D['changes']:
        if c['kind'] == 'open' and c['won'] > c['lost']+1:
            loose[c['lost']+1:c['won']] = True     # between the loser's last touch and the winner's first
    last = -1
    for i in range(n):
        if code[i] <= 0:
            last = -1
            state[i] = STATE_NAMES.index('dead' if code[i] == 0 else 'unknown')
            continue
        k = code[i]-1
        if has[i]:
            last = int(lead[i])
        if p[k, i] > 0 and p[1-k, i] > 0:
            state[i] = STATE_NAMES.index('contested')
        elif e[k, i] > 0:
            state[i] = STATE_NAMES.index('control')
            carrier[i] = _player_at(F, D, k, i)
        elif e[1-k, i] > 0 or loose[i] or last == 1-k:
            state[i] = STATE_NAMES.index('loose')
        else:
            state[i] = STATE_NAMES.index('flight')
    for a, b in _runs(state == STATE_NAMES.index('control')):
        seq = carrier[a:b+1]
        prev, s = None, 0
        for j in range(1, len(seq)+1):
            if j == len(seq) or seq[j] != seq[s]:
                if j-s < CARRIER_MIN and prev is not None:
                    seq[s:j] = prev
                else:
                    prev = seq[s]
                s = j
    estimated = (F['bstate'] == 2) | D['carried'] | (D['seen'] & ~D['usable'])
    level = _level(post, S)
    level[D['carried'] | (code == -1)] = 0
    level[code == 0] = 2
    return state, carrier, estimated, level


def _cause(F, S, D, c):
    """out_of_play across a ball out of play; interception when the ball travelled travel_m or more from the loser's last
    touch to the winner's first (a pass or a clearance that reached the other team); tackle_or_duel when it did not and
    a player of the winning team was within duel_r of the ball as it was lost, or both teams were near it in between;
    otherwise unclear."""
    if c['kind'] == 'restart':
        return 'out_of_play' if (c['boundary'] or {}).get('reason') == 'out_of_play' else 'unclear'
    if c['kind'] != 'open' or not c['lostSeen']:
        return 'unclear'
    t, bx, by = F['t'], F['bx'], F['by']
    lost, won, new = c['lost'], c['won'], c['to']
    if np.isfinite(bx[lost]) and np.isfinite(bx[won]) and math.hypot(bx[won]-bx[lost], by[won]-by[lost]) >= S['travel_m']:
        return 'interception'
    a = int(np.searchsorted(t, t[lost]-.2))
    if (F['dP'][new, a:lost+1] <= S['duel_r']).any() or ((D['p'][0, lost:won+1] > 0) & (D['p'][1, lost:won+1] > 0)).any():
        return 'tackle_or_duel'
    return 'unclear'


def _change_level(F, S, D, c):
    """high / medium / low. An open change: low when the new team has less than LOW_EVIDENCE s of evidence over the
    two seconds after it, high when it has HIGH_EVIDENCE s and the loser's last touch is more than DUEL_GAP s before
    the gain (not a duel's flicker), medium otherwise. A restart: medium (a ball out of play by one team and a restart
    by the other is clear, but the possession before the stoppage may be the wrong one); low across an unseen stretch
    or a camera cut (a 'gap': nobody saw the ball change hands) or without a seen loss."""
    if c['kind'] != 'open' or not c['lostSeen']:
        gap = (c.get('boundary') or {}).get('kind') == 'gap'
        return 'medium' if c['kind'] == 'restart' and c['lostSeen'] and not gap else 'low'
    a, b = c['won'], min(F['n'], c['won']+int(2*F['fps']))
    ev = float(D['e'][c['to'], a:b].sum())
    if ev < LOW_EVIDENCE:
        return 'low'
    if ev >= HIGH_EVIDENCE and F['t'][c['won']]-F['t'][c['lost']] > DUEL_GAP:
        return 'high'
    return 'medium'


def change_levels(F, S, D):
    """high / medium / low of every change (fix 5, calibrated on game 1 and the broadcast dev labels). low: a change
    undone within PAIR_S seconds, or one that undoes a change made PAIR_S before (a back-and-forth pair: on game 1 the
    pairs were right 50-81% of the time against 76-92% for the rest, on broadcast 42-222 s 0 of 2), an open change whose
    new team has less than LOW_EVIDENCE s of evidence over the 2 s after it, a change across an unseen stretch or a
    camera cut, or one whose loss nobody saw. high: an open change, not in a pair, with HIGH_EVIDENCE s of the new
    team's evidence over 2 s, the loser's last touch at most HIGH_LOST s before the gain, and the new possession lasting
    HIGH_HOLD s or more (to the next change or boundary). medium: the rest, restarts after a dead ball among them (on the
    simulated camera a restart's team or moment is right only 38-88% of the time). With conf_rule 'v2', the stage A rule
    (_change_level)."""
    chs = D['changes']
    if S['conf_rule'] == 'v2':
        return [_change_level(F, S, D, c) for c in chs]
    t, n, fps = F['t'], F['n'], F['fps']
    starts = sorted(b['a'] for b in D['bounds'])
    out = []
    for i, c in enumerate(chs):
        w = c['won']
        nxt = chs[i+1] if i+1 < len(chs) else None
        prv = chs[i-1] if i > 0 else None
        undone = nxt is not None and nxt['to'] == c['from'] and t[min(nxt['won'], n-1)]-t[w] <= PAIR_S
        undoes = prv is not None and prv['to'] == c['from'] and t[w]-t[min(prv['won'], n-1)] <= PAIR_S
        gap = (c.get('boundary') or {}).get('kind') == 'gap'
        ev = float(D['e'][c['to'], w:min(n, w+int(2*fps))].sum())
        if undone or undoes or gap or not c['lostSeen'] or (c['kind'] == 'open' and ev < LOW_EVIDENCE):
            out.append('low')
            continue
        end = min([nxt['won'] if nxt is not None else n]+[a for a in starts if a > w])
        hold = t[min(end, n-1)]-t[w]
        if c['kind'] == 'open' and ev >= S['high_ev'] and t[w]-t[c['lost']] <= HIGH_LOST and hold >= HIGH_HOLD:
            out.append('high')
        else:
            out.append('medium')
    if S['conf_rule'] == 'v4':
        return ['high' if lv == 'high' else 'uncertain' for lv in out]
    return out


def record(F, D):
    """The timeline runs, changes, dead spells, shares and coverage of decisions D on features F (JSON-safe)."""
    S = D['S']
    n, t, fr, fps, ids = F['n'], F['t'], F['frame'], F['fps'], F['ids']

    def pid(j):
        return ids[j] if j is not None and j >= 0 and ids[j] is not None else None

    state, carrier, estimated, level = frame_states(F, S, D)
    code = D['code']
    tm = np.where(code > 0, code-1, -1)
    runs = []
    i = 0
    while i < n:
        j = i
        while j+1 < n and tm[j+1] == tm[i] and state[j+1] == state[i] and carrier[j+1] == carrier[i]:
            j += 1
        runs.append({'t0': round(float(t[i]), 3), 't1': round(float(t[j]), 3), 'f0': int(fr[i]), 'f1': int(fr[j]),
                     'team': TEAMS[tm[i]] if tm[i] >= 0 else None, 'state': STATE_NAMES[state[i]], 'carrier': pid(int(carrier[i])),
                     'estimated': bool(estimated[i:j+1].mean() >= .5), 'confidence': LEVELS[int(np.median(level[i:j+1]))]})
        i = j+1
    changes = [{'id': f"pc-{int(fr[c['won']])}-{TEAMS[c['to']]}", 'lostAt': round(float(t[c['lost']]), 3), 'wonAt': round(float(t[c['won']]), 3),
                'frame': int(fr[c['won']]), 'fromTeam': TEAMS[c['from']], 'toTeam': TEAMS[c['to']], 'lostBy': pid(c['lostBy']),
                'wonBy': pid(c['wonBy']), 'cause': _cause(F, S, D, c), 'confidence': lv} for c, lv in zip(D['changes'], change_levels(F, S, D))]
    spells = [{'t0': round(float(t[b['a']]), 3), 't1': round(float(t[min(b['r'], n)-1]), 3), 'reason': b['reason'] or 'stoppage',
               'restartTeam': TEAMS[b['restart']] if b['restart'] is not None else None} for b in D['bounds'] if b['kind'] == 'dead']
    dt = 1.0/fps
    shares = {'A': round(float((code == 1).sum()*dt), 2), 'B': round(float((code == 2).sum()*dt), 2),
              'dead': round(float((code == 0).sum()*dt), 2), 'unknown': round(float((code == -1).sum()*dt), 2)}
    live = int((code != 0).sum())
    return {'runs': runs, 'changes': changes, 'deadSpells': spells, 'shares': shares,
            'coverage': round(float((code > 0).sum()/live), 3) if live else 0.0}


def compute(result, settings=None):
    """The possession record of a saved result (the output contract above): which team has the ball on every frame
    (absolute source seconds and frame indices, keyed on internal ids), the changes of possession with who lost and who
    won the ball, the dead-ball spells, the shares, and the run's quality, or why it is not available."""
    return from_features(features(result, settings), settings)


def from_features(F, settings=None):
    """compute() on features already extracted (tools/possession_bench.py extracts once and decides many times)."""
    S = dict(DEFAULTS)
    S.update(settings or {})
    n, fps = F['n'], F['fps']
    out = {'version': VERSION, 'fps': fps, 't0': round(float(F['t'][0]), 3) if n else 0.0, 't1': round(float(F['t'][-1]), 3) if n else 0.0}
    usable, seen = usable_ball(F, S)
    agree, agree_n = agreement(F, S, usable) if n else (None, 0)
    small = float(F['small'].mean()) if n else 0.0
    placed = float((usable | seen).mean()) if n else 0.0
    reason = None
    if not n:
        reason = 'The result has no frames.'
    elif small >= S['small_run']:
        reason = (f'The footage is a small, far scene on {100*small:.0f}% of its frames (amateur or wide camera): the ball and '
                  'the feet of the players are too uncertain to say who has the ball.')
    elif placed < S['min_usable']:
        reason = f'The ball is placed on only {100*placed:.0f}% of the frames: too little to say who has the ball.'
    elif not ((F['iP'] >= 0).any() or (F['iI'] >= 0).any()):
        reason = 'No player of team A or B is tracked near the ball.'
    elif agree is not None and agree_n >= S['agree_frames'] and agree < S['agree_min']:
        reason = (f'The pitch and the image disagree on which team is at the ball on {100*(1-agree):.0f}% of the frames where one '
                  'team is near it: the registration is not reliable enough to say who has the ball.')
    if reason is not None:
        dur = round(float(F['t'][-1]-F['t'][0]+1.0/fps), 2) if n else 0.0
        out.update(available=False, reason=reason, quality={'level': 'unavailable', 'coverage': 0.0, 'notes': [reason]},
                   shares={'A': 0.0, 'B': 0.0, 'dead': 0.0, 'unknown': dur},
                   timeline=[{'t0': out['t0'], 't1': out['t1'], 'f0': int(F['frame'][0]), 'f1': int(F['frame'][-1]), 'team': None,
                              'state': 'unknown', 'carrier': None, 'estimated': False, 'confidence': 'low'}] if n else [],
                   changes=[], deadSpells=[])
        return out
    D = decide(F, S)
    rec = record(F, D)
    notes = []
    named = D['code'] > 0
    bridged = float((F['bstate'][named] == 2).mean()) if named.any() else 0.0
    if bridged >= .3:
        notes.append(f'The ball is interpolated between sightings on {100*bridged:.0f}% of the frames where a team is named: '
                     'those runs are marked estimated.')
    if agree is not None:
        notes.append(f'The pitch and the image agree on the team at the ball on {100*agree:.0f}% of the frames where one team is near it.')
    gaps = sum(1 for b in D['bounds'] if b['kind'] == 'gap' and not (b.get('cut') and b['r']-b['a'] <= 1))
    if gaps:
        notes.append(f"The ball was out of view for {S['gap']:g} s or more {gaps} time{'s' if gaps != 1 else ''}: "
                     'possession restarts with the first touch after each.')
    cuts = int(F['cut'].sum()) if F.get('cut') is not None else 0
    if cuts:
        notes.append(f"The camera cut {cuts} time{'s' if cuts != 1 else ''}: after each, a team is named again only once one is "
                     'seen in control of the ball.' if S['cut_fresh'] else
                     f"The camera cut {cuts} time{'s' if cuts != 1 else ''}: possession restarts with the first touch after each.")
    rep = D.get('replay')
    if rep is not None and rep.any():
        notes.append(f'{rep.sum()/fps:.1f} s between two camera cuts look like a replay (the ball jumps away at the first cut and back '
                     'at the second): no team is named there.')
    if D.get('noOrigin'):
        notes.append('The result does not record where each ball position came from (ball.origin, results made before 2026-10-15): '
                     'every tracked ball is taken as observed, so changes of possession are less certain.')
    if F.get('kitFrames'):
        notes.append(f"On {F['kitFrames']} player-frames the frame's team contradicts the identity's team (an identity on another team's "
                     "player): those count for the frame's team.")
    excursion = D.get('excursion')
    if excursion is not None and excursion.any():
        notes.append(f'For {excursion.sum()/fps:.1f} s the ball path sat beyond a line on something that was not the ball in play '
                     '(it jumped there, or another ball was played on the pitch): no ball is shown there, and no out of play.')
    if F['sizeSource'] == 'standard':
        notes.append('The pitch size is assumed (105 x 68 m): the distances to the ball are estimates.')
    good = rec['coverage'] >= GOOD_COVERAGE and (agree is None or agree >= GOOD_AGREEMENT)
    out.update(available=True, reason=None, quality={'level': 'good' if good else 'limited', 'coverage': rec['coverage'], 'notes': notes},
               shares=rec['shares'], timeline=rec['runs'], changes=rec['changes'], deadSpells=rec['deadSpells'])
    return out
