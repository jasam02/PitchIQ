# Possession hand labels

Possession on four PitchIQ runs, labelled by hand from the video: `BC60` (La Liga broadcast, run 2bac432528d0e568,
0-60 s), `BC180` (the same match, 42.4-222.4 s), `AM` (amateur 3, run 6dfdc88ec6f2d100, 118.2-182.0 s) and `VEO`
(Veo, run f597d73d53f7b1fd, 1.0-31.0 s). Each file has a point every 0.5 s (`team` A / B / dead / unclear with a
confidence 1-3 and a note), the possession `changes` (`lostAt`, `wonAt`, `fromTeam`, `toTeam`, `cause`), dead spells or
restarts, the team-letter mapping, how disagreements were resolved and the labellers' agreement. Times are absolute
source seconds; the team letters are those of the run's result.json.

How they were made (2026-10-19/20): blind, from the video frames, with only the team letters of result.json boxes read
(never an engine output). BC60 and BC180: two labellers independently, then adjudicated frame by frame where they
differed (team agreement 118/118 and 315/316 points, kappa .989 on BC180; all 6 BC180 changes found by both, wonAt
within 0.45 s, lostAt within 2.0 s). AM and VEO: one labeller, then a seeded 20% re-check by the adjudicator (26/26 and
12/12 points agree) and every change re-checked. Roles: BC60 tuned possession-2; BC180, AM and VEO were held out for
possession-2 and are development data from possession-3 (labels are spent once read). Scored by
`python -I -B tools/possession_bench.py --labels tools/annotations/possession --run BC180=<run>/result.json`.
