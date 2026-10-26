# Synthetic coaching scenes

The scenes of DESIGN 13.1 (and the coach-marked possession scene of the user's decision of 2026-10-21) built by
`tools/coaching_synth.py`: deterministic PitchIQ-shaped results (`result.json` dicts) whose tactical truth is known, so
the coaching v2 pipeline can be tested on counterexamples no real clip provides. Tested by
`tests/test_coaching_synth.py`. Owner: EVAL (DESIGN 12.1).

## Use

```
python tools/coaching_synth.py list                               every scene/variant
python tools/coaching_synth.py check [SCENE[/VARIANT] ...] [--all]  load through analytics, team shape, possession, series
python tools/coaching_synth.py write SCENE[/VARIANT] --out DIR [--level N] [--fps F]
                                                                  result.json, truth.json, coaching_settings.json,
                                                                  coach_spans.json (refuses any runs/ or uploads/ path)
python tools/coaching_synth.py catalogue [--write]                the generated part of this file
```

In Python (tests put `tools/` on `sys.path`): `cs.build(scene, variant=None, **spec_overrides)` returns a `Built` with
`.result` (the result dict), `.truth` (truth.json), `.expected` (the expectations below), `.coach_spans` (or None);
`cs.settings_for(built, level=4, age=None, style='unknown', federation='auto', team='A', roles=True, declared=None)`
returns a complete, schema-valid `coaching_settings.json`; `cs.check(built)` runs the consumers; `cs.names()` lists the
catalogue. Overrides such as `fps=30.0` or `seed=1` rebuild the same script at another rate or noise draw.

## Conventions

- Times are absolute source seconds from 600.0 (frames[i].frame = round(time x fps): 6000 is the first frame at the
  default 10 fps). Team A defends the left goal (`tactical.goalEnds`), for the whole scene. Ids are internal: `A-01`..,
  `B-01`.., goalkeepers `GK-1` (A) and `GK-2` (B); displays are `GK-A`, `GK-B`.
- The coaching questions are about team A. 11v11 role names are team_shape's; small formats use the relative names
  `back|mid|front-k/n` (truth.roles has every player's role per period).
- Small fields are written the way PitchIQ writes them: positions normalized on the real field, `tactical.pitchSize`
  105 x 68 'standard' (the pitch model learns only 60-120 x 40-90 m fields). The real size is in `truth.field` and, when
  the scene says the coach declared it, in the settings (`field`). `kids_swarm/measured_size` writes the real size as
  'landmarks', which today's pitch model cannot produce.
- Expectations are DESIGN 13.1's (or the rule cited in `source`); they are what the coaching engine must do, not what it
  does today. Nothing in this file is a measurement of the coaching engine.

## Coach-marked possession spans (proposed layout)

`coach_spans.json` / `Built.coach_spans`, per run: `{"version": 1, "kind": "coach-possession-spans", "source": "coach",
"spans": [{"t0": 600.0, "t1": 632.0, "state": "A"}, {"t0": ..., "t1": ..., "state": "dead"}, ...]}`; `state` is `A` or
`B` (that team had the ball) or `dead`; spans are sorted, do not overlap and lie inside the analysed range. The stored
layout belongs to SERVER / CONTEXT (wave 2 and 7); this is the scenes' proposal (INTERFACES.md, EVAL-synth section).

## Known limits and DESIGN inconsistencies the scenes expose

1. DESIGN 13.1 plants "4 distinct spells", but DESIGN 4.9 asks 8-15 opportunities for any work item: with 4 spells the
   13.1 outcomes (try_this at L1, work_on at L4-L7) are unreachable. The default variants plant 16 spells over 256 s of
   own possession (L7: 15 opportunities, 180 s); `unsupported_advance/spells4` is the literal 4 with the 4.9 outcome.
2. DESIGN 13.1 gives the L1-L2 outcome of the unsupported advance only for variant (ii) (nobody behind the ball); none
   is asserted for (i).
3. A wing-back of a back three is "advanced" (15 m beyond the centre-backs) through most of his team's possession on
   his own (about 17-20 m in the layouts, as Metrica's team B: 20.6 m p50): his spells are one per possession span.
4. In the zoomed view (`three_visible`) the evidence tier is pinned: 2 paint segments are in view, world needs 3.
5. A 4v4 field written as 105 x 68 makes possession-4 abstain (distances read 3.5x too long): the swarm scene relies on
   the coach's spans for its phase.
6. Not simulated: detection misses inside the view, registration drift beyond .2 m of AR(1) noise, identity switches
   (a crossing is reported, never executed), referees, replays and cuts, lens bend in the image, bridged ball frames,
   the ball tracker's candidates, the stabilized 2D stream, the imputation timeline (TeamShape's estimated mode).

## Measured (2026-10-22, main venv, no GPU; `check --all` and truth.json)

Possession-4: available?, share of its named frames on the scripted team, at that coverage of the live frames, changes
found / scripted. Team shape: team A's stabilized formation periods. In view: median outfield players of A in the
camera's view. Plants (from truth.json, on the true positions): share of each spell's advanced own-possession samples
passing any cover test, share with the planted number behind the ball, share with an opponent in the vacated channel,
seconds advanced in own possession outside the change windows (ranges over the spells); for the defending scenes the
shares on the settled samples. Every scene: no player faster than 8.5 m/s, no ball faster than 25 m/s.

| Scene / variant | Frames | Possession-4 | Team shape (A) | In view | Plants |
|---|---|---|---|---|---|
| `covered_advance/cb_shift` | 3660 | yes, 0.998 at 0.996, 15/15 | 4-4-2 | 10 | 16 spells; cover 1.00; behind -; opponent 0.00; advanced own 6.4-6.8 s |
| `covered_advance/channel` | 3660 | yes, 0.998 at 0.997, 15/15 | 4-4-2 | 10 | 16 spells; cover 0.94-1.00; behind -; opponent 0.00; advanced own 6.4-6.8 s |
| `unsupported_advance/none_behind` | 3660 | yes, 0.999 at 0.996, 15/15 | 4-4-2 | 10 | 16 spells; cover 0.00; behind 1.00; opponent 1.00; advanced own 5.2-5.8 s |
| `unsupported_advance/one_behind` | 3660 | yes, 0.999 at 0.996, 15/15 | 4-4-2 | 10 | 16 spells; cover 0.00; behind 0.97-1.00; opponent 1.00; advanced own 6.0-6.4 s |
| `unsupported_advance/spells4` | 3660 | yes, 0.999 at 0.995, 15/15 | 4-4-2 | 10 | 4 spells; cover 0.00; behind 1.00; opponent 1.00; advanced own 5.4-5.6 s |
| `single_advance/default` | 3660 | yes, 0.998 at 0.996, 15/15 | 4-4-2 | 10 | 1 spell; cover 0.00; behind 1.00; opponent 1.00; advanced own 5.6 s |
| `identity_crossing/default` | 3660 | yes, 1.000 at 0.995, 15/15 | 4-4-2 | 10 | 16 spells; cover 0.00; behind 1.00; opponent 1.00; advanced own 5.4-6.0 s; crossing 0.30 m |
| `possession_unknown/default` | 3660 | no | 4-4-2 | 10 | 16 spells; cover 0.00; behind 1.00; opponent 1.00; advanced own 5.2-5.8 s |
| `three_visible/default` | 3660 | yes, 1.000 at 0.821, 15/15 | insufficient | 3 | 16 spells; cover 0.00; behind 0.97-1.00; opponent 1.00; advanced own 6.0-6.4 s |
| `low_calibration/default` | 3660 | yes, 0.999 at 0.996, 15/15 | 4-4-2 | 10 | 16 spells; cover 0.00; behind 1.00; opponent 1.00; advanced own 5.2-5.8 s |
| `sprint_back/default` | 3120 | yes, 0.999 at 0.992, 23/23 | 4-4-2 | 10 | 12 spells; cover 0.00; behind -; opponent 1.00; advanced own 0.2 s |
| `wrong_formation/default` | 3660 | yes, 0.998 at 0.996, 15/15 | 3-5-2 | 10 | 8 spells; cover 0.00-0.06; behind 0.93-1.00; opponent 1.00; advanced own 27.8-30.0 s |
| `wrong_formation/coach_roles` | 3660 | yes, 0.998 at 0.996, 15/15 | 3-5-2 | 10 | 8 spells; cover 0.00-0.06; behind 0.93-1.00; opponent 1.00; advanced own 27.8-30.0 s |
| `formation_change/default` | 3660 | yes, 0.999 at 0.996, 15/15 | 4-4-2 -> 3-5-2 | 10 | 12 spells; cover 0.00-0.20; behind 0.79-1.00; opponent 1.00; advanced own 6.0-28.4 s |
| `short_segment/default` | 600 | yes, 1.000 at 0.995, 1/1 | 4-4-2 | 10 | 3 spells; cover 0.00; behind 1.00; opponent 1.00; advanced own 5.6 s |
| `kids_swarm/default` | 2360 | no | lineup | 4 | pressed 1.00, covered 1.00, weak side empty -, gap open - |
| `kids_swarm/measured_size` | 2360 | yes, 0.983 at 0.700, 31/31 | lineup | 4 | pressed 1.00, covered 1.00, weak side empty -, gap open - |
| `small_field_scale/assumed` | 2950 | yes, 1.000 at 0.993, 15/15 | lineup | 6 | pressed 0.97, covered 1.00, weak side empty 0.00, gap open 0.00 |
| `small_field_scale/declared` | 2950 | yes, 1.000 at 0.993, 15/15 | lineup | 6 | pressed 0.97, covered 1.00, weak side empty 0.00, gap open 0.00 |
| `professional/default` | 3660 | yes, 1.000 at 0.995, 15/15 | 4-3-3 | 10 | 16 spells; cover 0.97-1.00; behind -; opponent 0.00; advanced own 6.4-6.8 s |
| `professional/no_advances` | 3660 | yes, 1.000 at 0.995, 15/15 | 4-3-3 | 10 | pressed 0.81, covered 1.00, weak side empty -, gap open - |
| `stacking/default` | 3660 | yes, 0.999 at 0.996, 15/15 | 4-4-2 | 10 | 16 spells; cover 0.00; behind 0.97-1.00; opponent 1.00; advanced own 6.0-6.4 s |
| `amateur_coach_spans/spans` | 3660 | no | lineup | 6 | 16 spells; cover 0.00; behind -; opponent 1.00; advanced own 6.2-6.6 s |
| `amateur_coach_spans/no_spans` | 3660 | no | lineup | 6 | 16 spells; cover 0.00; behind -; opponent 1.00; advanced own 6.2-6.6 s |
| `amateur_coach_spans/veo` | 3660 | no | lineup | 6 | 16 spells; cover 0.00; behind -; opponent 1.00; advanced own 6.2-6.6 s |
| `defending/press_cover` | 2790 | yes, 0.998 at 0.972, 15/15 | 4-4-2 | 10 | pressed 0.96, covered 0.95, weak side empty -, gap open - |
| `defending/press_no_cover` | 2790 | yes, 0.998 at 0.972, 15/15 | 4-4-2 | 10 | pressed 0.96, covered 0.12, weak side empty -, gap open - |
| `defending/no_press` | 2790 | yes, 0.998 at 0.991, 15/15 | 4-4-2 | 10 | pressed 0.01, covered 0.00, weak side empty -, gap open - |
| `defending/weak_empty` | 2790 | yes, 0.999 at 0.995, 15/15 | 4-4-2 | 10 | pressed 0.54, covered 0.97, weak side empty 1.00, gap open 0.00 |
| `defending/gap_open` | 2790 | yes, 0.999 at 0.946, 15/15 | 4-4-2 | 10 | pressed 0.55, covered 0.94, weak side empty 0.00, gap open 1.00 |

Wave 8 (2026-10-22): `defending/gap_open` keeps B's ball on one wing for each whole span (it switched wings mid-span
before, and the defence needs about 3.5 s to open the gap, so the gap was open on about half of every D5
opportunity: 7 of 25 failed at level 4). Its row above is re-measured; every other row is unchanged.

Fix round 1 (2026-10-23): the scenes the round added or changed, measured the same way (`check` on these variants and
truth.json). The channel opponent's depth is the deepest opponent in the vacated channel against the centre-back line,
the median over each spell's advanced own-possession samples (truth.json's new `channelOpponentVsCbM`); `opponent_at`
'high' puts him halfway up to the advanced left-back (`low_calibration`, `stacking/reachable`, `channel_threat/high`),
'deep' 4 m deeper than the centre-backs. The defending rows give A's shares: PRESS_D became 2.0 m for the three-state
pressure call (pressed within 3 m), so the first defender now stands at a median 2.44 m. `covered_buildup`, `box_edge`,
`low_block` and `unsupported_advance/final_third` were added by DETECT / ASSESS in the round; `channel_threat` and
`stacking/reachable` by EVAL.

| Scene / variant | Frames | Possession-4 | Team shape (A) | In view | Plants |
|---|---|---|---|---|---|
| `low_calibration/default` | 3660 | yes, 0.999 at 0.996, 15/15 | 4-4-2 | 10 | 16 spells; cover 0.00; behind 1.00; opponent 1.00; channel opponent vs CB line +6.5 to +9.9 m; advanced own 5.2-5.8 s |
| `stacking/default` | 3660 | yes, 0.999 at 0.996, 15/15 | 4-4-2 | 10 | 16 spells; cover 0.00; behind 0.97-1.00; opponent 1.00; channel opponent vs CB line -4.1 m; advanced own 6.0-6.4 s |
| `stacking/reachable` | 3660 | yes, 0.999 at 0.983, 15/15 | 4-4-2 | 10 | 16 spells; cover 0.00; behind 0.97-1.00; opponent 1.00; channel opponent vs CB line +9.9 m; advanced own 6.0-6.4 s |
| `channel_threat/deep` | 3660 | yes, 0.999 at 0.995, 15/15 | 4-4-2 | 10 | 16 spells; cover 0.00; behind 1.00; opponent 1.00; channel opponent vs CB line -4.1 m; advanced own 6.2-6.4 s |
| `channel_threat/high` | 3660 | yes, 0.999 at 0.995, 15/15 | 4-4-2 | 10 | 16 spells; cover 0.00; behind 1.00; opponent 1.00; channel opponent vs CB line +9.9 m; advanced own 6.2-6.4 s |
| `covered_buildup/default` | 3660 | yes, 0.997 at 0.995, 15/15 | 4-4-2 | 10 | 16 spells; cover 1.00; behind 1.00; opponent 0.00; advanced own 5.4-6.0 s |
| `box_edge/box` | 2790 | yes, 0.998 at 0.922, 15/15 | 4-4-2 | 10 | pressed 0.98, covered 0.37, median first defender 4.33 m, weak side empty 1.00, gap open 0.21 |
| `box_edge/flat_line` | 2790 | yes, 0.998 at 0.923, 15/15 | 4-4-2 | 10 | pressed 0.97, covered 0.17, median first defender 4.31 m, weak side empty 1.00, gap open 0.30 |
| `low_block/default` | 3660 | yes, 0.997 at 0.980, 15/15 | 4-4-2 | 10 | pressed 1.00, covered 0.15, median first defender 4.45 m, weak side empty -, gap open - |
| `low_block/compressed` | 3660 | yes, 0.997 at 0.982, 15/15 | 4-4-2 | 10 | pressed 1.00, covered 0.43, median first defender 4.44 m, weak side empty -, gap open - |
| `unsupported_advance/final_third` | 3660 | yes, 0.998 at 0.996, 15/15 | 4-4-2 | 10 | 16 spells; cover 0.00; behind 1.00; opponent 1.00; channel opponent vs CB line -27.7 to -27.0 m; advanced own 4.4-5.8 s |
| `defending/press_cover` | 2790 | yes, 0.997 at 0.949, 15/15 | 4-4-2 | 10 | pressed 0.96, covered 0.95, median first defender 2.44 m, weak side empty -, gap open - |
| `defending/press_no_cover` | 2790 | yes, 0.997 at 0.949, 15/15 | 4-4-2 | 10 | pressed 0.96, covered 0.11, median first defender 2.44 m, weak side empty -, gap open - |
| `defending/no_press` | 2790 | yes, 0.998 at 0.991, 15/15 | 4-4-2 | 10 | pressed 0.01, covered 0.00, median first defender 9.2 m, weak side empty -, gap open - |
| `defending/weak_empty` | 2790 | yes, 0.997 at 0.995, 15/15 | 4-4-2 | 10 | pressed 0.54, covered 0.97, median first defender 2.45 m, weak side empty 1.00, gap open 0.00 |
| `defending/gap_open` | 2790 | yes, 0.998 at 0.945, 15/15 | 4-4-2 | 10 | pressed 0.55, covered 0.94, median first defender 2.45 m, weak side empty 0.00, gap open 1.00 |

<!-- BEGIN GENERATED: python tools/coaching_synth.py catalogue -->

## Scenes

| Scene / variant | What is planted | Format, A's shape | Seconds (A has the ball) | Spells | Footage and settings |
|---|---|---|---|---|---|
| `covered_advance/cb_shift` | Left-back advances 20 m with cover (near centre-back shifts 6 m, or a team-mate takes the channel) | 11v11, 4-4-2 | 366 (256) | 16 | broadcast, full view |
| `covered_advance/channel` | Left-back advances 20 m with cover (near centre-back shifts 6 m, or a team-mate takes the channel) | 11v11, 4-4-2 | 366 (256) | 16 | broadcast, full view |
| `unsupported_advance/none_behind` | Left-back advances 20 m without cover, opponent in the vacated channel | 11v11, 4-4-2 | 366 (256) | 16 | broadcast, full view |
| `unsupported_advance/one_behind` | Left-back advances 20 m without cover, opponent in the vacated channel | 11v11, 4-4-2 | 366 (256) | 16 | broadcast, full view |
| `unsupported_advance/spells4` | Left-back advances 20 m without cover, opponent in the vacated channel | 11v11, 4-4-2 | 366 (256) | 4 | broadcast, full view |
| `unsupported_advance/final_third` | Left-back advances 20 m without cover, nobody behind, the ball in the final third | 11v11, 4-4-2 | 366 (256) | 16 | broadcast, full view |
| `single_advance/default` | One unsupported advance only | 11v11, 4-4-2 | 366 (256) | 1 | broadcast, full view |
| `identity_crossing/default` | Unsupported advances with an identity crossing at each | 11v11, 4-4-2 | 366 (256) | 16 | broadcast, full view |
| `possession_unknown/default` | Unsupported advances, possession unavailable | 11v11, 4-4-2 | 366 (256) | 16 | ball sparse |
| `three_visible/default` | Unsupported advances seen by a zoomed camera (about 3 team-mates in view) | 11v11, 4-4-2 | 366 (256) | 16 | follow camera |
| `low_calibration/default` | Unsupported advances under a MEDIUM registration everywhere | 11v11, 4-4-2 | 366 (256) | 16 | registration 0.55 |
| `sprint_back/default` | Left-back caught upfield at every loss, sprinting back | 11v11, 4-4-2 | 312 (192) | 12 | broadcast, full view |
| `wrong_formation/default` | Coach selects 4-4-2, the team plays 3-5-2 | 11v11, 3-5-2 | 366 (256) | 8 | broadcast, full view |
| `wrong_formation/coach_roles` | Coach selects 4-4-2, the team plays 3-5-2 | 11v11, 3-5-2 | 366 (256) | 8 | coach roles |
| `formation_change/default` | Formation change mid-segment (4-4-2 to 3-5-2) | 11v11, 4-4-2 -> 3-5-2 | 366 (256) | 12 | broadcast, full view |
| `short_segment/default` | A 60 s segment with three unsupported advances | 11v11, 4-4-2 | 60 (42) | 3 | broadcast, full view |
| `kids_swarm/default` | 4v4 (no keepers): everyone near the ball | 4v4 (no GK), 1-2-1 | 236 (112) | 0 | coach spans, field 30 x 20 m declared |
| `kids_swarm/measured_size` | 4v4 (no keepers): everyone near the ball | 4v4 (no GK), 1-2-1 | 236 (112) | 0 | field 30 x 20 m declared |
| `small_field_scale/assumed` | 7v7 on a 55 x 36 m field, with or without the field declared | 7v7, 1-3-2-1 | 295 (112) | 0 | field 55 x 36 m not declared |
| `small_field_scale/declared` | 7v7 on a 55 x 36 m field, with or without the field declared | 7v7, 1-3-2-1 | 295 (112) | 0 | field 55 x 36 m declared |
| `professional/default` | Professional-like team: covered advances, pressure with cover | 11v11, 4-3-3 | 366 (256) | 16 | broadcast, full view |
| `professional/no_advances` | Professional-like team: covered advances, pressure with cover | 11v11, 4-3-3 | 366 (256) | 0 | broadcast, full view |
| `stacking/default` | Unsupported advances with a team-mate where the cover zone would be | 11v11, 4-4-2 | 366 (256) | 16 | broadcast, full view |
| `stacking/reachable` | Unsupported advances with a team-mate where the cover zone would be, a reachable cover spot | 11v11, 4-4-2 | 366 (256) | 16 | broadcast, full view |
| `amateur_coach_spans/spans` | Amateur 7v7 clip: possession abstains, the coach marked the spans | 7v7, 1-3-2-1 | 366 (256) | 16 | amateur, registration 0.39, coach spans, field 55 x 36 m declared |
| `amateur_coach_spans/no_spans` | Amateur 7v7 clip: possession abstains, the coach marked the spans | 7v7, 1-3-2-1 | 366 (256) | 16 | amateur, registration 0.39, field 55 x 36 m declared |
| `amateur_coach_spans/veo` | Amateur 7v7 clip: possession abstains, the coach marked the spans | 7v7, 1-3-2-1 | 366 (256) | 16 | veo, registration 0.45, coach spans, field 55 x 36 m declared |
| `defending/press_cover` | Team A defending: pressure, cover, weak side, back-line gap | 11v11, 4-4-2 | 279 (96) | 0 | broadcast, full view |
| `defending/press_no_cover` | Team A defending: pressure, cover, weak side, back-line gap | 11v11, 4-4-2 | 279 (96) | 0 | broadcast, full view |
| `defending/no_press` | Team A defending: pressure, cover, weak side, back-line gap | 11v11, 4-4-2 | 279 (96) | 0 | broadcast, full view |
| `defending/weak_empty` | Team A defending: pressure, cover, weak side, back-line gap | 11v11, 4-4-2 | 279 (96) | 0 | broadcast, full view |
| `defending/gap_open` | Team A defending: pressure, cover, weak side, back-line gap | 11v11, 4-4-2 | 279 (96) | 0 | broadcast, full view |
| `covered_buildup/default` | Covered left-back advances while the near centre-back carries the ball in the own half | 11v11, 4-4-2 | 366 (256) | 16 | broadcast, full view |
| `box_edge/box` | B's wide carrier at the edge of A's box, 17.5 m from its goal line; A's back four level with the ball | 11v11, 4-4-2 | 279 (96) | 0 | broadcast, full view |
| `box_edge/flat_line` | B's wide carrier 27 m from A's goal line; A's back four level with the ball | 11v11, 4-4-2 | 279 (96) | 0 | broadcast, full view |
| `low_block/default` | Positional attack against a low block: forwards on the last line, about 24 m of length | 11v11, 4-4-2 | 366 (256) | 0 | broadcast, full view |
| `low_block/compressed` | Compressed attack against a low block: forwards 15 m short of the last line | 11v11, 4-4-2 | 366 (256) | 0 | broadcast, full view |
| `channel_threat/deep` | Uncovered left-back advances, the channel opponent deeper than the cover region | 11v11, 4-4-2 | 366 (256) | 16 | broadcast, full view |
| `channel_threat/high` | Uncovered left-back advances, the channel opponent upfield of the cover region | 11v11, 4-4-2 | 366 (256) | 16 | broadcast, full view |

## Expectations

Tokens (each expectation is `{levels, expect, detectors, value, source}` in truth.json and in `Built.expected`):

- `no_criticism`: no work_on, try_this or review finding of the detector(s) for team A.
- `criticised`: a work_on, try_this or review finding of the detector(s) for team A (level 1: the reframed advance included).
- `moments_review`: a 'moments worth a look' review of the detector (limitation moments_worth_a_look).
- `strength_allowed`: a strength of the detector may appear (a candidate: the scene alone does not require it).
- `no_strength`: no strength of the detector.
- `try_this_team`: a try_this finding phrased for the team (no player named).
- `team_level`: the findings name no player (team-level wording).
- `review_or_work_on`: the detector's criticism is a review or a work_on (DESIGN 4.9 decides which).
- `work_on`: a work_on finding of the detector.
- `at_most_review`: no work_on or try_this of the detector (review, describe or strength only).
- `review_one_instance`: a review that says it is one observed instance.
- `no_drill`: no drill linked to the detector's findings.
- `no_player_named`: the detector's findings name no player.
- `possession_unavailable`: possession-4 abstains for the run (available false).
- `not_assessed`: the detector's status is not_assessed with the reason in value.
- `assessed`: the detector's status is assessed (opportunities may still be 0).
- `describe_only`: only a describe finding (never a verdict).
- `no_team_claims`: team-level detectors are not assessed; local relations only.
- `no_video_zone`: no suggestedZone / moveArrow / channel element on any video overlay.
- `diagram_zone`: a suggested zone, when one is proposed, is drawn on the top-down diagram.
- `seek_exact`: every evidence seek equals (frame + 0.5) / fps.
- `notice`: the overview carries the notice named in value.
- `not_downgraded_by_mismatch`: the formation-mismatch rule does not downgrade the coach-confirmed role.
- `suppressed_near_formation_change`: no role-based criticism from opportunities within value s of a measured formation period boundary.
- `no_repetition_wording`: no "repeated", "pattern" or "often" in any text (range under 3 minutes).
- `work_items_need_spells`: every work item has at least value distinct failing spells.
- `no_work_items`: no work_on or try_this finding at all.
- `no_position_criticism`: no criticism of positions (D2-D5) for team A.
- `max_try_this`: at most value try_this findings.
- `strength_only_if_demonstrated`: every strength rests on demonstrated spells.
- `same_evidence_across_levels`: the findings of one opportunity family cite the same evidence ids at every level.
- `zone_avoids_stacking`: no suggested-zone point within 3 m of a team-mate, or no zone.
- `no_zone`: the detector's criticisms draw no suggested area and record the reason in value (vacuous without one).
- `zone_goal_side_of_threat`: every suggested area of the detector reaches at most 2 m upfield of the deepest opponent in the vacated channel at its moment (critic B-8), or no zone.
- `phase_from_coach`: opportunities carry phase.source "coach" where the coach marked the span.
- `planted_outcome`: the detector classifies the planted opportunities as value (DESIGN 4.8 step 2).

### `covered_advance` (DESIGN 13.1 row 1)

- **cb_shift**: Left-back advances 20 m with cover (near centre-back shifts 6 m, or a team-mate takes the channel).
  - all levels: `no_criticism` for wide_defender_advance [DESIGN 13.1]
  - L3,4,5,6,7: `strength_allowed` for wide_defender_advance [DESIGN 13.1]
  - Note: 16 covered advances in 8 possession spans of 32 s (256 s of own possession), 14 s apart.
  - Note: cb_shift passes cover test (c) and, with the centre-back inside the outer third, test (a); channel passes test (a) only.
- **channel**: Left-back advances 20 m with cover (near centre-back shifts 6 m, or a team-mate takes the channel).
  - all levels: `no_criticism` for wide_defender_advance [DESIGN 13.1]
  - L3,4,5,6,7: `strength_allowed` for wide_defender_advance [DESIGN 13.1]
  - Note: 16 covered advances in 8 possession spans of 32 s (256 s of own possession), 14 s apart.
  - Note: cb_shift passes cover test (c) and, with the centre-back inside the outer third, test (a); channel passes test (a) only.

### `unsupported_advance` (DESIGN 13.1 row 2)

- **none_behind**: Left-back advances 20 m without cover, opponent in the vacated channel.
  - L1: `try_this_team` for wide_defender_advance, behind_ball [critic B-1 (fix round 3)] (the centre-back carries the ball into midfield, but the advance is exposed: "someone ready to protect our goal" for the team, no player named (fix round 1 asserted no criticism here))
  - L2: `try_this_team` for wide_defender_advance [critic B-1 (fix round 3)] (level 2's team-level rule: exposed wherever the ball is)
  - L1,2: `criticised` for wide_defender_advance [critic B-1 (fix round 3)] (uncovered, an opponent in the vacated channel (exposed): judged wherever the ball is)
  - L3: `review_or_work_on` for wide_defender_advance [DESIGN 13.1] (per DESIGN 4.9)
  - L4,5,6: `work_on` for wide_defender_advance [DESIGN 13.1] (level-specific claims; at L6 because the channel is exposed)
  - L7: `review_or_work_on` for wide_defender_advance [DESIGN 2.3 L7, 2.6] (style unknown: a review)
  - L7: `at_most_review` for wide_defender_advance [DESIGN 2.3 L7, 2.6] (style unknown: criticisms stay review)
  - L7 (style possession declared): `work_on` for wide_defender_advance [DESIGN 2.3 L7, 2.6] (judged against the declared style; the channel is exposed)
  - all levels: `same_evidence_across_levels` for wide_defender_advance [DESIGN 13.1]
  - Note: No local cover test passes: the same-half team-mates tuck inside the outer third, the near centre-back does not shift; an opponent waits in the vacated channel in A's half (exposed).
  - Note: behind the ball: the near centre-back carries the ball into midfield while the far full-back steps up; (ii) the other centre-back too (nobody >= 10 m goal-side of the ball), (i) he stays (one player).
- **one_behind**: Left-back advances 20 m without cover, opponent in the vacated channel.
  - L1,2: `criticised` for wide_defender_advance [critic B-1 (fix round 3)] (uncovered, an opponent in the vacated channel (exposed): judged wherever the ball is)
  - L3: `review_or_work_on` for wide_defender_advance [DESIGN 13.1] (per DESIGN 4.9)
  - L4,5,6: `work_on` for wide_defender_advance [DESIGN 13.1] (level-specific claims; at L6 because the channel is exposed)
  - L7: `review_or_work_on` for wide_defender_advance [DESIGN 2.3 L7, 2.6] (style unknown: a review)
  - L7: `at_most_review` for wide_defender_advance [DESIGN 2.3 L7, 2.6] (style unknown: criticisms stay review)
  - L7 (style possession declared): `work_on` for wide_defender_advance [DESIGN 2.3 L7, 2.6] (judged against the declared style; the channel is exposed)
  - all levels: `same_evidence_across_levels` for wide_defender_advance [DESIGN 13.1]
  - Note: No local cover test passes: the same-half team-mates tuck inside the outer third, the near centre-back does not shift; an opponent waits in the vacated channel in A's half (exposed).
  - Note: behind the ball: the near centre-back carries the ball into midfield while the far full-back steps up; (ii) the other centre-back too (nobody >= 10 m goal-side of the ball), (i) he stays (one player).
  - Note: DESIGN 13.1 states L1-L2 outcomes only for variant (ii); none is asserted here for (i).
- **spells4**: Left-back advances 20 m without cover, opponent in the vacated channel.
  - all levels: `at_most_review` for wide_defender_advance [DESIGN 4.9] (4 spells are fewer than every level's minimum opportunities (8-15))
  - L3,4,5,6,7: `moments_review` for wide_defender_advance [critic S-2 (fix round 3)] (4 clear advances in 6.1 minutes: the wide-defender check counts its clear advances against its own floor (momentsMinSpells), not "the range is too short")
  - L1,2: `criticised` for wide_defender_advance [critic B-1 (fix round 3)] (exposed advances; 4 are fewer than the level's minimum, so a review)
  - Note: No local cover test passes: the same-half team-mates tuck inside the outer third, the near centre-back does not shift; an opponent waits in the vacated channel in A's half (exposed).
  - Note: behind the ball: the near centre-back carries the ball into midfield while the far full-back steps up; (ii) the other centre-back too (nobody >= 10 m goal-side of the ball), (i) he stays (one player).
  - Note: DESIGN 13.1 asks for 4 spells, but DESIGN 4.9 needs 8-15 opportunities for a work item: with 4 the 13.1 outcomes are unreachable. The default variants plant 16; this one is the literal 4.
- **final_third**: Left-back advances 20 m without cover, nobody behind, the ball in the final third.
  - L1: `try_this_team` for wide_defender_advance, behind_ball [DESIGN 4.3 L1, critic B-1 (fix round 1)] ("someone ready to protect our goal"; no player named)
  - L2: `try_this_team` for wide_defender_advance [DESIGN 4.3 L2, critic B-1 (fix round 1)]
  - L4,5,6: `review_or_work_on` for wide_defender_advance [DESIGN 13.1] (uncovered, an opponent in the vacated channel in our half (exposed))
  - L1,2,3: `no_player_named` for behind_ball [DESIGN 13.1] (levels 1-3 name nobody (naming never / relative / role words); from level 4 the evidence may label the player (naming evidence_only / yes))
  - Note: Possession high in B's half (block centre up to 80 m): the near centre-back carries the ball at about 74-76 m (the final third) while the left-back stands 20 m beyond the centre-back line; nobody of A is 10 m goal-side of the ball; no cover test passes; an opponent waits in the vacated channel in A's half. The advances start 8 and 19 s into each possession, once the block has moved up.

### `single_advance` (DESIGN 13.1 row 3)

- **default**: One unsupported advance only.
  - L3,4,5,6,7: `not_assessed` = too_few_situations for wide_defender_advance [critic S-1 (fix round 1), S-2 (fix round 3)] (one advance in 256 s of own possession: under half the level's minimum opportunities with enough of the phase, so too few situations to judge, not a short range (DESIGN 13.1 row 3 asked for a one-instance review))
  - all levels: `no_drill` for wide_defender_advance [DESIGN 13.1]
  - L1,2: `no_criticism` for wide_defender_advance, behind_ball [critic B-1 (fix rounds 1 and 3)] (one exposed advance is fewer situations than levels 1-2 judge (too few situations): no criticism (fix round 1: the ball was in midfield))
  - Note: The first spell of unsupported_advance/none_behind; the left-back stays home otherwise.

### `identity_crossing` (DESIGN 13.1 row 4)

- **default**: Unsupported advances with an identity crossing at each.
  - all levels: `no_player_named` for wide_defender_advance [DESIGN 13.1]
  - all levels: `at_most_review` for wide_defender_advance [DESIGN 13.1]
  - Note: An opponent runs through the left-back at the middle of every hold (closest .3 m); the result reports "Crossed a similar player" for both and a swap-uncertain event, and the left-back's identity confidence is .45 for 1.5 s either side.

### `possession_unknown` (DESIGN 13.1 row 5)

- **default**: Unsupported advances, possession unavailable.
  - all levels: `possession_unavailable` for report [DESIGN 3.2]
  - all levels: `not_assessed` = phase_unknown for wide_defender_advance, behind_ball [DESIGN 13.1]
  - all levels: `not_assessed` = phase_unknown for pressure_cover, weak_side_balance, defensive_spacing [DESIGN 3.2]
  - all levels: `describe_only` for attacking_shape [DESIGN 4.6]
  - Note: The ball is seen .5 s every 4 s (12.5% of the frames): possession-4 needs a usable ball on 20% and abstains for the whole run.

### `three_visible` (DESIGN 13.1 row 6)

- **default**: Unsupported advances seen by a zoomed camera (about 3 team-mates in view).
  - all levels: `no_team_claims` for report [DESIGN 13.1]
  - all levels: `not_assessed` = team_incomplete for weak_side_balance, behind_ball, attacking_shape [DESIGN 3.5]
  - Note: A camera 27 m wide follows the ball on A's left wing (left-biased by 10 m): the left-back, the ball carrier and one or two team-mates are in view; truth.inView gives the counts.
  - Note: Geometry of unsupported_advance/one_behind with the ball on the left.

### `low_calibration` (DESIGN 13.1 row 7)

- **default**: Unsupported advances under a MEDIUM registration everywhere.
  - L1,2: `criticised` for wide_defender_advance [critic B-1 (fix round 3)] (uncovered, an opponent in the vacated channel (exposed): judged wherever the ball is)
  - all levels: `no_video_zone` for wide_defender_advance [DESIGN 13.1]
  - L4,5,6,7: `diagram_zone` for wide_defender_advance [DESIGN 13.1]
  - all levels: `seek_exact` for report [DESIGN 13.1]
  - all levels: `zone_goal_side_of_threat` for wide_defender_advance [critic B-8 (fix round 1)]
  - Note: Registration confidence .55 on every frame and no paint ticks: the best evidence tier is pinned.
  - Note: Fix round 1: the channel opponent waits halfway between the centre-back line and the advanced left-back (opponent_at high). With him 4 m deeper than the centre-backs, as before, no cover spot goal-side of him is reachable in 2 s (critic B-8), no area is proposed at all, and the row's "diagram only, never on the video" had nothing left to test.

### `sprint_back` (DESIGN 13.1 row 8)

- **default**: Left-back caught upfield at every loss, sprinting back.
  - all levels: `no_criticism` for wide_defender_advance [DESIGN 13.1]
  - Note: Every A possession ends with an interception while the left-back is 18-20 m beyond the centre-back line; he then runs back at 5-8 m/s (>= 3 m/s goalwards) inside the change window.
  - Note: Advanced own-possession time outside the +-2 s change windows is under 3 s per spell (truth).

### `wrong_formation` (DESIGN 13.1 row 9)

- **default**: Coach selects 4-4-2, the team plays 3-5-2.
  - all levels: `notice` = formation_mismatch for report [DESIGN 13.1]
  - L2,3,4,5,6,7: `at_most_review` for wide_defender_advance [DESIGN 13.1] (role criticism without coach roles downgraded to review)
  - L1: `try_this_team` for wide_defender_advance, behind_ball [critic B-1 (fix round 3), DESIGN 3.4] (the exposed advance for the team; no player, role or formation named)
  - Note: A holds 3-5-2 throughout; its left wing-back (A-04) stands 20 m beyond the back three through each possession (one long spell per span: 8), uncovered, an opponent in the channel.
- **coach_roles**: Coach selects 4-4-2, the team plays 3-5-2.
  - all levels: `notice` = formation_mismatch for report [DESIGN 13.1]
  - all levels: `not_downgraded_by_mismatch` for wide_defender_advance [DESIGN 3.4] (the left wing-back is coach-confirmed)
  - Note: A holds 3-5-2 throughout; its left wing-back (A-04) stands 20 m beyond the back three through each possession (one long spell per span: 8), uncovered, an opponent in the channel.

### `formation_change` (DESIGN 13.1 row 10)

- **default**: Formation change mid-segment (4-4-2 to 3-5-2).
  - all levels: `suppressed_near_formation_change` = 15.0 for wide_defender_advance [DESIGN 13.1]
  - Note: A changes shape at 749.0 s (players walk to their new slots over 8 s; the left-back becomes the left wing-back). Before it: two left-back spells per span; after it: the wing-back held high through each span. truth.plants[].measured.advancedNearFormationChangeSeconds says which spells lie within 15 s of the scripted change; team_shape finds the boundary itself (a clear change takes it 15-30 s), so the suppression window is around the measured boundary.

### `short_segment` (DESIGN 13.1 row 11)

- **default**: A 60 s segment with three unsupported advances.
  - all levels: `no_repetition_wording` for report [DESIGN 13.1]
  - all levels: `work_items_need_spells` = 3 for report [DESIGN 13.1]
  - all levels: `no_work_items` for report [DESIGN 4.9] (42 s of own possession is under every level's minimum relevant phase time (60-180 s))
  - Note: Three distinct spells 13 s apart in 42 s of possession.

### `kids_swarm` (DESIGN 13.1 row 12)

- **default**: 4v4 (no keepers): everyone near the ball.
  - all levels: `possession_unavailable` for report [DESIGN 2.5 (assumed size)]
  - L1,2: `phase_from_coach` for report [user decision 1 (2026-10-21)]
  - L1,2: `no_position_criticism` for report [DESIGN 13.1]
  - L1,2: `max_try_this` = 1 for report [DESIGN 13.1]
  - L1,2: `strength_only_if_demonstrated` for report [DESIGN 13.1]
  - Note: A 30 x 20 m field (declared); all eight players within about 7 m of the ball; the ball changes team every 7 s.
  - Note: Written as PitchIQ writes a small field: 105 x 68 standard, so possession-4 reads every distance 3.5x too long and abstains (the pitch and the image disagree); the coach marked the spans.
- **measured_size**: 4v4 (no keepers): everyone near the ball.
  - L1,2: `no_position_criticism` for report [DESIGN 13.1]
  - L1,2: `max_try_this` = 1 for report [DESIGN 13.1]
  - L1,2: `strength_only_if_demonstrated` for report [DESIGN 13.1]
  - Note: A 30 x 20 m field (declared); all eight players within about 7 m of the ball; the ball changes team every 7 s.
  - Note: tactical.pitchSize is the real 30 x 20 m (landmarks), which today's pitch model cannot produce (soccer/field.py learns sizes of 60-120 x 40-90 m only): possession-4 then measures real metres.

### `small_field_scale` (DESIGN 13.1 row 13)

- **assumed**: 7v7 on a 55 x 36 m field, with or without the field declared.
  - all levels: `not_assessed` = scale_unknown for pressure_cover [DESIGN 13.1]
  - Note: A presses the ball with cover (D1 opportunities in B's 22 s spans). Without a declared field the scale is the assumed 105 x 68 (positions normalized on the real 55 x 36 m field): body-scale detectors must not be assessed.
- **declared**: 7v7 on a 55 x 36 m field, with or without the field declared.
  - all levels: `assessed` for pressure_cover [DESIGN 2.5]
  - Note: A presses the ball with cover (D1 opportunities in B's 22 s spans). Without a declared field the scale is the assumed 105 x 68 (positions normalized on the real 55 x 36 m field): body-scale detectors must not be assessed.

### `professional` (DESIGN 13.1 row 14 (+ 13.4 #11 mutation twin))

- **default**: Professional-like team: covered advances, pressure with cover.
  - all levels: `no_work_items` for report [DESIGN 13.1]
  - all levels: `strength_allowed` for pressure_cover [EVAL]
  - L3,4,5,6,7: `strength_allowed` for wide_defender_advance [DESIGN 13.1]
  - Note: A (4-3-3) alternates left-back (centre-back shift) and right-back (a midfielder takes the channel) advances, always covered; when defending its nearest player presses at 3 m with a second one 7 m behind him.
- **no_advances**: Professional-like team: covered advances, pressure with cover.
  - all levels: `no_work_items` for report [DESIGN 13.1]
  - all levels: `strength_allowed` for pressure_cover [EVAL]
  - all levels: `no_strength` for wide_defender_advance [DESIGN 13.4 #11] (mutation twin: the demonstrations removed)
  - Note: A (4-3-3) alternates left-back (centre-back shift) and right-back (a midfielder takes the channel) advances, always covered; when defending its nearest player presses at 3 m with a second one 7 m behind him.

### `stacking` (DESIGN 13.1 row 15)

- **default**: Unsupported advances with a team-mate where the cover zone would be.
  - L1,2: `criticised` for wide_defender_advance [critic B-1 (fix round 3)] (uncovered, an opponent in the vacated channel (exposed): judged wherever the ball is)
  - L4,5,6,7: `zone_avoids_stacking` for wide_defender_advance [DESIGN 13.1]
  - L4,5,6,7: `no_zone` = no_reachable_fix for wide_defender_advance [critic B-8 (fix round 1)] (the channel opponent is deeper than every spot reachable in 2 s: the principle only)
  - Note: The left centre-midfielder stands 2.5 m outside the outer third, 13 m behind the advanced left-back: not cover (test a), but inside the natural cover zone's 3 m disc.
  - Note: Fix round 1 (critic B-8): the channel opponent waits 4 m deeper than the centre-back line, so no cover spot level with or goal-side of him is reachable; variant reachable is the twin with a reachable spot.
- **reachable**: Unsupported advances with a team-mate where the cover zone would be, a reachable cover spot.
  - L1,2: `criticised` for wide_defender_advance [critic B-1 (fix round 3)] (uncovered, an opponent in the vacated channel (exposed): judged wherever the ball is)
  - L4,5,6,7: `zone_avoids_stacking` for wide_defender_advance [DESIGN 13.1]
  - all levels: `zone_goal_side_of_threat` for wide_defender_advance [critic B-8 (fix round 1)]
  - L4,5,6: `work_on` for wide_defender_advance [DESIGN 13.1] (the same plant as the default: uncovered, exposed)
  - Note: The left centre-midfielder stands 2.5 m outside the outer third, 13 m behind the advanced left-back: not cover (test a), but inside the natural cover zone's 3 m disc.
  - Note: Fix round 1: the channel opponent waits halfway between the centre-back line and the advanced left-back (opponent_at high), so a cover spot goal-side of him is reachable and the zone must keep 3 m off the stacking team-mate.

### `amateur_coach_spans` (user decision 1 (2026-10-21))

- **spans**: Amateur 7v7 clip: possession abstains, the coach marked the spans.
  - all levels: `possession_unavailable` for report [DESIGN 3.2]
  - all levels: `phase_from_coach` for report [user decision 1 (2026-10-21)]
  - all levels: `assessed` for wide_defender_advance, defensive_spacing, attacking_shape [user decision 1; DESIGN 4.3, 4.5, 4.6] (phase from the coach spans; these detectors need no ball)
  - all levels: `not_assessed` = no_carrier for pressure_cover [DESIGN 3.3]
  - L2,3,4,5,6,7: `not_assessed` = ball_unreliable for weak_side_balance, behind_ball [DESIGN 3.3 (ball vetoed on small scenes)]
  - L1: `criticised` for wide_defender_advance [critic B-1 (fix round 3)] (the exposed advance needs no ball depth)
  - L1: `assessed` for behind_ball [critic B-1 (fix round 3)] (the reframed advance is judged on the exposure, without the ball)
  - Note: Small-scene footage (frames[].profile.smallScene on every frame), a keyless MEDIUM registration (amateur) or a lens bend (veo): possession-4 abstains (small_run) and the series vetoes the ball.
  - Note: coach_spans.json marks every A, B and dead span of the script (state A|B|dead, t0, t1).
- **no_spans**: Amateur 7v7 clip: possession abstains, the coach marked the spans.
  - all levels: `possession_unavailable` for report [DESIGN 3.2]
  - all levels: `not_assessed` = phase_unknown for pressure_cover, weak_side_balance, wide_defender_advance, behind_ball, defensive_spacing [DESIGN 3.2]
  - all levels: `describe_only` for attacking_shape [DESIGN 4.6]
  - Note: Small-scene footage (frames[].profile.smallScene on every frame), a keyless MEDIUM registration (amateur) or a lens bend (veo): possession-4 abstains (small_run) and the series vetoes the ball.
  - Note: coach_spans.json marks every A, B and dead span of the script (state A|B|dead, t0, t1).
- **veo**: Amateur 7v7 clip: possession abstains, the coach marked the spans.
  - all levels: `possession_unavailable` for report [DESIGN 3.2]
  - all levels: `phase_from_coach` for report [user decision 1 (2026-10-21)]
  - all levels: `assessed` for wide_defender_advance, defensive_spacing, attacking_shape [user decision 1; DESIGN 4.3, 4.5, 4.6] (phase from the coach spans; these detectors need no ball)
  - all levels: `not_assessed` = no_carrier for pressure_cover [DESIGN 3.3]
  - L2,3,4,5,6,7: `not_assessed` = ball_unreliable for weak_side_balance, behind_ball [DESIGN 3.3 (ball vetoed on small scenes)]
  - L1: `criticised` for wide_defender_advance [critic B-1 (fix round 3)] (the exposed advance needs no ball depth)
  - L1: `assessed` for behind_ball [critic B-1 (fix round 3)] (the reframed advance is judged on the exposure, without the ball)
  - Note: Small-scene footage (frames[].profile.smallScene on every frame), a keyless MEDIUM registration (amateur) or a lens bend (veo): possession-4 abstains (small_run) and the series vetoes the ball.
  - Note: coach_spans.json marks every A, B and dead span of the script (state A|B|dead, t0, t1).

### `defending` (EVAL: D1 / D2 / D5 planted outcomes)

- **press_cover**: Team A defending: pressure, cover, weak side, back-line gap.
  - all levels: `planted_outcome` = demonstrated for pressure_cover [EVAL (DESIGN 4.1, 4.2, 4.5 definitions)]
  - Note: Not a row of DESIGN 13.1: planted D1 / D2 / D5 outcomes for the detectors "implemented first". B keeps the ball 22 s of every 34 s in A's half (wide variants: on one wing, then the other), carries last 3-5 s so the first defender arrives. truth.defending gives the shares on the settled samples.
- **press_no_cover**: Team A defending: pressure, cover, weak side, back-line gap.
  - L1: `planted_outcome` = demonstrated for pressure_cover [DESIGN 4.1 (level 1: demonstrated = pressure)]
  - L2,3,4,5,6,7: `planted_outcome` = not_demonstrated for pressure_cover [EVAL (DESIGN 4.1, 4.2, 4.5 definitions)]
  - Note: Not a row of DESIGN 13.1: planted D1 / D2 / D5 outcomes for the detectors "implemented first". B keeps the ball 22 s of every 34 s in A's half (wide variants: on one wing, then the other), carries last 3-5 s so the first defender arrives. truth.defending gives the shares on the settled samples.
- **no_press**: Team A defending: pressure, cover, weak side, back-line gap.
  - all levels: `planted_outcome` = not_demonstrated for pressure_cover [EVAL (DESIGN 4.1, 4.2, 4.5 definitions)]
  - Note: Not a row of DESIGN 13.1: planted D1 / D2 / D5 outcomes for the detectors "implemented first". B keeps the ball 22 s of every 34 s in A's half (wide variants: on one wing, then the other), carries last 3-5 s so the first defender arrives. truth.defending gives the shares on the settled samples.
- **weak_empty**: Team A defending: pressure, cover, weak side, back-line gap.
  - L3,4,5,6,7: `planted_outcome` = not_demonstrated for weak_side_balance [EVAL (DESIGN 4.1, 4.2, 4.5 definitions)]
  - L3,4,5,6,7: `review_or_work_on` for weak_side_balance [DESIGN 4.2, 4.9]
  - Note: Not a row of DESIGN 13.1: planted D1 / D2 / D5 outcomes for the detectors "implemented first". B keeps the ball 22 s of every 34 s in A's half (wide variants: on one wing, then the other), carries last 3-5 s so the first defender arrives. truth.defending gives the shares on the settled samples.
- **gap_open**: Team A defending: pressure, cover, weak side, back-line gap.
  - L3,4,5,6,7: `planted_outcome` = not_demonstrated for defensive_spacing [EVAL (DESIGN 4.1, 4.2, 4.5 definitions)]
  - L4,5,6,7: `review_or_work_on` for defensive_spacing [DESIGN 4.5, 4.9]
  - Note: Not a row of DESIGN 13.1: planted D1 / D2 / D5 outcomes for the detectors "implemented first". B keeps the ball 22 s of every 34 s in A's half (wide variants: on one wing, then the other), carries last 3-5 s so the first defender arrives. truth.defending gives the shares on the settled samples.

### `covered_buildup` (COACH verifier scene 1 (fix round 1, critic B-1))

- **default**: Covered left-back advances while the near centre-back carries the ball in the own half.
  - all levels: `no_criticism` for wide_defender_advance, behind_ball [COACH B1 / critic B-1 (fix round 1)]
  - L3,4,5,6,7: `strength_allowed` for wide_defender_advance [DESIGN 13.1]
  - Note: Build-up: the near centre-back carries the ball at about 31 m (the block held at 35 m), the other centre-back and the far full-back step up, nobody is 10 m goal-side of the ball; a midfielder takes the vacated channel 15 m behind the left-back (test a) on every advance.
  - Note: Before fix round 1 level 1 read these as "nobody stayed back to protect our goal" (the BC60 / BC303 Barcelona build-up the verifiers found).

### `box_edge` (COACH verifier scene 3 (fix round 1, critic B-5))

- **box**: B's wide carrier at the edge of A's box, 17.5 m from its goal line; A's back four level with the ball.
  - L3,4,5,6,7: `no_criticism` for weak_side_balance [COACH B3 / critic B-5 (fix round 1)]
  - Note: B keeps the ball 22 s of every 34 s on A's right wing at the edge of A's box, 17.5 m from its goal line (its block pushed up so its wide midfielder carries there); A's block is held so its centre-backs stand level with the ball's depth (the full-backs about 3 m higher): the far side is protected by players between the ball and the goal, never deeper than the ball.
  - Note: The plain depth rule (X <= ball) counts nobody goal-side there; the fixed rule counts the far full-back goal-side by his distance to the goal centre and calls the centre-backs level.
- **flat_line**: B's wide carrier 27 m from A's goal line; A's back four level with the ball.
  - L3,4,5,6,7: `no_criticism` for weak_side_balance [COACH B3 / critic B-5 (fix round 1)]
  - Note: B keeps the ball 22 s of every 34 s on A's right wing 27 m from A's goal line (its block pushed up so its wide midfielder carries there); A's block is held so its centre-backs stand level with the ball's depth (the full-backs about 3 m higher): the far side is protected by players between the ball and the goal, never deeper than the ball.
  - Note: The plain depth rule (X <= ball) counts nobody goal-side there; the fixed rule counts the far full-back goal-side by his distance to the goal centre and calls the centre-backs level.

### `low_block` (COACH verifier scene 4 (fix round 1, critic B-7))

- **default**: Positional attack against a low block: forwards on the last line, about 24 m of length.
  - L1,2,3: `no_criticism` for attacking_shape [COACH B4 / critic B-7 (fix round 1)] (tall and deep for the space available)
  - L1,2,3: `strength_allowed` for attacking_shape [DESIGN 13.1]
  - Note: B defends with its back line about 16.5 m from its goal (A frame 88.5 m); A attacks with its forwards on that line and its centre-backs about 24 m behind them: a field share of about .23, the whole available length.
- **compressed**: Compressed attack against a low block: forwards 15 m short of the last line.
  - L1,2,3: `no_criticism` for attacking_shape [LEAD decision (c), fix round 2: 11v11 depth described at levels 1-3] (about half of the available length used)
  - L1,2,3: `no_strength` for attacking_shape [LEAD decision (c), fix round 2] (a flat team is not praised for its shape either)
  - Note: The twin of low_block/default with A compressed: about 14 m from back to front, its forwards about 15 m short of B's last line (about half of the available length).

### `channel_threat` (COACH verifier scene 2 (fix round 1, critic B-8))

- **deep**: Uncovered left-back advances, the channel opponent deeper than the cover region.
  - L1,2: `criticised` for wide_defender_advance [critic B-1 (fix round 3)] (uncovered, an opponent in the vacated channel (exposed): judged wherever the ball is)
  - L4,5,6: `work_on` for wide_defender_advance [DESIGN 13.1] (uncovered, an opponent in the vacated channel (exposed))
  - L3,4,5,6,7: `no_zone` = no_reachable_fix for wide_defender_advance [COACH B2 / critic B-8 (fix round 1)] (nobody reaches a spot level with or goal-side of the opponent in 2 s: principle only)
  - all levels: `same_evidence_across_levels` for wide_defender_advance [DESIGN 13.1]
  - Note: The opponent waits 4 m deeper than A's centre-back line in the outer third the left-back left; the cover region (centre-back line - 5 m to the left-back - 10 m) lies upfield of him.
- **high**: Uncovered left-back advances, the channel opponent upfield of the cover region.
  - L1,2: `criticised` for wide_defender_advance [critic B-1 (fix round 3)] (uncovered, an opponent in the vacated channel (exposed): judged wherever the ball is)
  - L4,5,6: `work_on` for wide_defender_advance [DESIGN 13.1] (uncovered, an opponent in the vacated channel (exposed))
  - L4,5,6,7: `diagram_zone` for wide_defender_advance [COACH B2 / critic B-8 (fix round 1)]
  - all levels: `zone_goal_side_of_threat` for wide_defender_advance [COACH B2 / critic B-8 (fix round 1)]
  - all levels: `same_evidence_across_levels` for wide_defender_advance [DESIGN 13.1]
  - Note: The opponent waits halfway between A's centre-back line and the advanced left-back (a winger left near halfway): a cover spot goal-side of him is reachable.

<!-- END GENERATED -->
