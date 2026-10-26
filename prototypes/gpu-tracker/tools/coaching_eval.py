"""Coaching v2 evaluation (DESIGN 13, EVAL wave 8): the synthetic counterexamples through the REAL engine, the same video
at every level, the real run copies at all seven levels, the blind label sheets a coach fills in later, the scorer that
reads them, and an independent check of the pressure distance on StatsBomb's open data.

    python tools/coaching_eval.py scenes [SCENE[/VARIANT] ...] [--json OUT]
    python tools/coaching_eval.py levels [SCENE[/VARIANT] ...] [--run NAME=result.json ...] [--json OUT]
    python tools/coaching_eval.py runs --out DIR --run NAME=result.json ... [--spans NAME=labels.json ...]
    python tools/coaching_eval.py summary --reports DIR [--scenes J] [--levels J] [--statsbomb J] --out SUMMARY.md
    python tools/coaching_eval.py sheets --reports DIR --out DIR [--video NAME=video.mp4 ...]
    python tools/coaching_eval.py rekey --sheets DIR --reports DIR --out KEYS   (after the freeze: the same items' keys)
    python tools/coaching_eval.py --labels DIR [--sheets DIR] [--keys KEYS]    (the scorer: same as `score DIR`)
    python -I -B tools/coaching_eval.py statsbomb DATA_DIR [--json OUT]

Nothing here writes under a runs/ or uploads/ folder (refused), changes a result, or reads Metrica game 2 or 3.

Measured on Oct 22 (wave 8, CPU; outputs in the session scratch coaching2/build/eval/):
- scenes: 30 variants, 380 checks pass, 0 fail, 1 known open defect (KNOWN_DEFECTS), 43 not applicable; 0 report
  invariants broken on any report (139 s).
- levels: 30 variants x 14 profiles (4 re-measured): obs / roles bytes identical, 0 problems; known open defects
  d6_describe_repeats (184 level pairs, Jaccard .55-.76) and evidence_repicked (3 families; both repaired in fix round
  1, see KNOWN_LEVEL_DEFECTS); the most similar explanations of one family at two levels otherwise .368 (617 s).
- runs: BC60, BC180, BC303, AM, VEO and AM / VEO with the synthetic coach-marked phase, both teams, 14 profiles (196
  reports): 0 problems, obs / roles identical under all 14 re-measured profiles, 64 pairs of the D6 repetition; the
  only work items are BC180 Barcelona's attacking depth at levels 1-2; no suggested area drawn (every other criticism
  is a review whose evidence guardrail failed).
- statsbomb: 4 Euro 2020 matches, 8576 carrier events with a visible opponent (see statsbomb_check).
- sheets: MOMENT 266 (broadcast 200, amateur 47, Veo 19), ROLE 60, ADVANCE 8, ZONE 48 (45 Metrica game 1, 3 clips).

Fix round 1 (Oct 23, on the round's final code; outputs in scratch coaching2/build/eval_fix1/):
- report_problems also checks the round's rules (fix1_problems, game-3 G1+: presser, phase words, level-1/2 "someone
  back", strength runs, zone sides, reasons) and criterion 12 the drill's match format and 11v11 role words.
- scenes: 39 variants (+ channel_threat deep / high, stacking/reachable and the four scenes DETECT added), 451 checks
  pass, 0 fail, 0 known, 65 not applicable, 10 vacuous; 0 report invariants broken on any report (181 s).
- levels: 39 variants x 14 profiles (4 re-measured): obs / roles identical, 0 problems, KNOWN_LEVEL_DEFECTS empty;
  worst level pair of one family .444 (807 s).
- runs: the same 196 reports: 0 problems, 0 known defects; criticisms 18 -> 0 (15 reviews and 3 try_this before),
  strengths 17 -> 18, not-assessed checks without a reason 76 -> 0.
- B-9: the keys a sheet writes carry the frozen block; the scorer refuses any other (the committed wave-8 sheets
  included); `rekey` of the committed sheets on this code changes the engine fields of 67 of 266 MOMENT items, 4 of 8
  ADVANCE items and 0 of 60 ROLE items (eval_fix1/rekey_preview_diff.json). The sheets were not regenerated.

Fix round 2 (Oct 23, LEAD decisions (b)-(d); outputs in scratch coaching2/build/eval_fix2/ and coaching2/fix2/eval/):
- report_problems also runs fix2_problems (game-3 G1+ of the amended plan) on every 'moments worth a look' review,
  independently of assess and explain: its form (a review: no priority, drill, area; level >= momentsFromLevel; no
  failed guardrail but the evidence one; alone in its family), the cap (momentsMaxPerReport), its counts recomputed
  from the obs / roles files (decided, failing, clear opportunities and their distinct spells), every cited moment
  inside a clear failing run, the count of the judged ones (and the share where shares are shown) and every cited
  time stated, no verdict, prescription or blame words, and an overview that does not count it as a possible issue.
- on ASSESS's cached Metrica game-1 tiers (3- and 15-minute clips, full tracking and camera; the planted clips of the
  game-3 plan's G12 and of fix round 1) at the 14 PROFILES: 13734 reports, 350 moments reviews (11 of 2072 clip
  reports, 41 of 798 G12, 298 of 10864 fix-1 planted), 0 problems of any criterion; each of the 350 doctored in 7 ways
  (a count +1, the spells +1, a count of all opportunities, a cited time removed, "should" added, a cited
  opportunity's identity lowered, a second copy): 2446 of the 2446 applicable doctorings flagged with their criterion.
- scenes: 39 variants, 454 checks pass, 0 fail, 0 known, 65 not applicable, 10 vacuous; 0 report invariants broken
  (229 s); at the 14 profiles the scenes give 12 moments reviews in 546 reports, 0 problems.
- levels: 39 variants x 14 profiles (4 re-measured): obs / roles identical, 0 problems; worst level pair .444 (813 s).
- runs: the same 196 reports: 0 problems, 0 moments reviews; verdict lists identical to fix round 1 in 196 of 196;
  next objective 'none' 0 -> 182 (188 changed), optional drills 196 -> 14, the federation word in overviews 84 -> 0.

Fix round 3 (Oct 24, the round-2 critic's B-1 to B-5, S-5, S-6; LEAD decisions (1)-(3); outputs in scratch
coaching2/build/eval_fix3/ and coaching2/fix3/eval/):
- fix1_problems' G1+someone_back is amended with critic B-1 / N3: a level-1/2 failing advance sample is never covered;
  it is exposed (uncovered, opponentsInChannel >= 1) wherever the ball is, or, not exposed, has the ball in the final
  third with nobody (level 2) / too few (level 1) behind it. Before, every exposed failure off the final third was a
  problem: the level-1/2 reports of 14 scene variants failed the invariant test in TEXT's end-of-round suite
  (none_behind, one_behind, spells4, identity_crossing, low_calibration, wrong_formation x2, formation_change,
  stacking x2, channel_threat x2, amateur_coach_spans x2); 0 now.
- fix2_problems recounts the 'moments worth a look' with the round's clear-moment gates, written independently
  (unsettled_intervals / _unsettled: dead balls and the 2 s after them, 2 s around unnamed stretches; _sample_gates:
  momentsContestS from a loose or contested ball, the 11v11 D5 line of lineMinPlayers with lineStdM <= lineStdMaxM,
  a lateral-gap opponent free beyond markedM) and checks the base-rate test (G1+moments_base: binomial_tail, scipy's,
  of the recomputed clear spells over the decided opportunities against baseFailShare at baseTestAlpha; the stated
  baseShare / baseTail) and that the less clear failures are counted (FACTS F-4). small_field_scale/declared's level-4
  review: recount 10 clear in 8 spells -> 7 in 5, equal to assess's.
- fix3_problems (new, in report_problems): the first cited moment's own actor and gap pair (G1+moment_actor) and
  third (G1+moment_zone) from the obs / roles files, every aggregate recomputed (_family_judged) and no strength
  counted over all its opportunities (G1+aggregate), a count behind the ball with its margin (G1+behind_margin),
  TEXT's report.text_problems (G1+report_text), and DRILLS' rules (G1+drills: no drill below the fallback floor, no
  repetition word, no optional drill comparing with a match that decided nothing of its check). 0 problems on the
  scenes, the 196 run copies, the user's 9 runs (252 reports) and the game-1 dry run (784 reports); each rule flags a
  doctored report (tests/test_coaching_eval.Fix3Invariants).
- scene tokens 'criticised' and 'moments_review' (tools/coaching_synth.py's new expectations); levels_check and
  run_reports name the report's run (FACTS F-12: the 196 run-copy reports said 'synthetic'; now 'BC60' ... and
  'AM-coach-synthetic-spans' where the coach-marked phase is the eval's synthetic one);
  the label keys and manifests end with a newline (CODE L-7).
- scenes: 39 variants, 477 checks pass, 0 fail, 0 known, 65 not applicable, 10 vacuous; 0 report invariants broken
  (177 s). levels: 39 variants x 14 profiles (4 re-measured): obs / roles identical, 0 problems; worst level pair of
  one family .453 (.444 before; 778 s). runs: 196 reports, 0 problems; criticisms 0 -> 0; strengths 18 -> 30,
  mixed-evidence describes 18 -> 8 (verdict lists differ in 11 reports: BC180 B's far side and spacing praised,
  BC60 B's level-1 covered advance). Game-1 dry run of the game-3 plan: 784 reports, 0 G1 problems.

Why each part exists:
- `scenes`: every expectation of tools/coaching_synth.py (DESIGN 13.1 plus the coach-marked and defending scenes) is
  checked on the report coaching.interpret makes, through the server's own measurement path (store.measure: coach
  spans folded into the key, the evidence anchors merged), at levels 1-7. The expectations were written in wave 1,
  before any engine existed; DETECT's wave-3 run on assess() alone found 6 of 314 in conflict with rules other roles
  had adopted from DESIGN, and each conflict is settled in the scene file with its DESIGN section (see there).
- The report invariants (`report_problems`) are checked on EVERY report, synthetic or real: schema and cross_check
  (invariant 3), unmeasured events (invariant 4), seek = (frame + 0.5) / fps on the opportunity's own frames
  (criterion 8), the focus ring on the observed position at the key frame, a teammate relation behind every work item
  (criterion 7), the drill's detector and the age's safety (criterion 12), the level's banned and generic words and a
  per-report word-trigram Jaccard under .5 (criterion 13), and the level-1/2 caps (criterion 3).
- `levels` (DESIGN 13.2): the observation and role files are re-measured under every level, age band and style and
  must keep their bytes (invariant 2); a family's findings cite the same moments at every level; the enabled
  detectors follow DESIGN 2.3's table (written out here independently of profiles.json); explanations of one family
  differ between levels (trigram Jaccard < .5).
- `runs`: BC60, BC180, BC303, AM and VEO at all seven levels for both teams, AM and VEO also with a SYNTHETIC
  coach-marked phase (the blind possession hand labels turned into spans, labelled as such everywhere).
- `sheets` / `--labels`: DESIGN 13.3's ROLE, MOMENT, ADVANCE and ZONE sets as blind sheets (video timestamps and short
  questions, no PitchIQ output) and the scorer that computes the success criteria once a coach has filled them; until
  then nothing here is claimed validated.
- `statsbomb`: the nearest-defender distance at StatsBomb carrier events flagged under_pressure vs not, from the 360
  freeze frames, against PitchIQ's 5 m pressure rule (DESIGN Appendix A PRESSURE_M; data by StatsBomb, open data
  licence; not copied into the repository).
"""
import argparse
import csv
import hashlib
import html
import io
import json
import math
import os
import random
import re
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TOOLS = REPO/'tools'
for _p in (str(REPO), str(TOOLS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

ANNOTATIONS = REPO/'tools'/'annotations'/'coaching'
D1, D2, D3, D4, D5, D6 = ('pressure_cover', 'weak_side_balance', 'wide_defender_advance', 'behind_ball',
                          'defensive_spacing', 'attacking_shape')
DETECTORS = (D1, D2, D3, D4, D5, D6)
LEVELS = (1, 2, 3, 4, 5, 6, 7)
CRITICISM = ('work_on', 'try_this', 'review')
WORK = ('work_on', 'try_this')
FINDING_SECTIONS = ('strengths', 'needsWork', 'reviews', 'described')

# ---------- independent tables (DESIGN text, written out here, NOT read from profiles.json) ----------
# DESIGN 2.3 "Detectors [P]" per level, completed by the "Levels" rows of DESIGN 4.1-4.6: the detectors a level enables,
# the verdicts it names (required) and what else it allows: a strength wherever demonstrations can be counted (DESIGN
# 4.3 "candidate strength at L3+", 4.8 #5, 4.9's strength column), except the cells DESIGN calls "only" or describe
# (D2 review only and D5 describe at L3, D6 describe from L5, D4 describe at L5 U15-U16; D4 at L1-L3 was "try_this
# only" until fix round 1, where the critic's S-8 allowed its strength: a team that kept someone back is praised);
# review and describe for every criticising detector (PROFILE's deviation 2: DESIGN 4.8 needs them for guardrail
# failures and explained intent). Levels 6/7 are the style-unknown rows (DESIGN 2.6: L6 work_on only for D1/D3; L7
# criticisms review).
_ST = {'strength'}
DESIGN_TABLE = {
    1: {D1: ({'strength', 'try_this'}, set()), D4: ({'try_this'}, _ST), D6: ({'strength', 'try_this'}, set())},
    2: {D1: ({'strength', 'try_this', 'review'}, set()), D3: ({'try_this'}, set()), D4: ({'try_this'}, _ST),
        D6: ({'strength'}, {'try_this'})},
    3: {D1: ({'strength', 'work_on'}, set()), D2: ({'review'}, set()), D3: ({'review', 'work_on'}, _ST),
        D4: ({'try_this', 'review'}, _ST), D5: ({'describe'}, set()), D6: ({'strength', 'try_this'}, set())},
    4: {D1: ({'work_on'}, _ST), D2: ({'work_on'}, _ST), D3: ({'work_on'}, _ST), D4: ({'work_on'}, _ST),
        D5: ({'review', 'strength'}, set()), D6: ({'describe', 'strength'}, set())},
    5: {D1: ({'work_on'}, _ST), D2: ({'work_on'}, _ST), D3: ({'work_on'}, _ST), D4: ({'describe'}, set()),
        D5: ({'work_on'}, _ST), D6: ({'describe'}, set())},
    6: {D1: ({'work_on'}, _ST), D2: (set(), _ST), D3: ({'work_on'}, _ST), D4: (set(), _ST), D5: (set(), _ST),
        D6: ({'describe'}, set())},
    7: {D1: ({'review'}, _ST), D2: ({'review'}, _ST), D3: ({'review'}, _ST), D4: ({'review'}, _ST),
        D5: ({'review'}, _ST), D6: ({'describe'}, set())},
}
# DESIGN 4.9: work items shown at most, per level.
MAX_WORK = {1: 1, 2: 2, 3: 2, 4: 3, 5: 3, 6: 3, 7: 3}
# DESIGN 2.3 "Register [P]" banned words (plus "mistake" at every level: TEXT's deviation 3). Matched as whole words.
DESIGN_BANNED = {1: ('mistake', 'balance', 'rest defence', 'cover', 'press', 'pressing', 'transition'),
                 2: ('mistake', 'balance', 'rest defence', 'transition', 'pressing trigger'),
                 3: ('mistake', 'rest defence', 'pressing trigger')}
# Generic coaching phrases a report must never use (spec Part 10: specific, evidence-based language; criterion 13).
GENERIC = ('work on your positioning', 'work on positioning', 'keep your shape', 'keep the shape', 'stay compact',
           'be more compact', 'be more organised', 'be more organized', 'good job', 'well done', 'great job',
           'keep it up', 'needs to improve', 'need to improve', 'be better', 'focus more', 'try harder',
           'communicate better', 'talk more', 'play as a team', 'stay in formation', 'out of position',
           'poor positioning', 'bad positioning')
# Formation-discipline wording (criterion 3: beginners are never judged on formation slots).
FORMATION_DISCIPLINE = re.compile(r'\b(formation|out of position|in position|positional discipline|hold (?:your|the|his) '
                                  r'position|stay(?:ed)? in (?:your|his|their) (?:spot|slot|position)|slot)\b', re.I)
# A player singled out: an internal id (the page shows display labels; stored text carries ids) or a role word for one
# player ("the left back", "our left defender"). Team words ("our defenders") are not singling out.
ID_RE = re.compile(r'\b(?:[AB]-\d{1,3}|GK-\d{1,3})\b')
ROLE_RE = re.compile(r"\b(?:left|right)[- ](?:back|wing-back|centre-back|center-back|defender|midfielder|winger|"
                     r"forward)\b(?!s)|\b(?:centre|center)-back\b(?!s)|\bthe (?:six|holding midfielder|sweeper)\b", re.I)
REPETITION = re.compile(r'\b(repeated(?:ly)?|repeats?|patterns?|often)\b', re.I)
TRIGRAM_LIMIT = .5
# Same-video findings that miss DESIGN 13.2 because of a rule another role owns (kind, predicate, owner and why); the
# levels test asserts each is still seen, so a repair is noticed and the entry removed.
KNOWN_LEVEL_DEFECTS = {}
# Fix round 1 (TEXT, critic S-6 / W8 miss 1): explain.py's descriptions read a different relation at every level
# (lanes, rounded metres, metres and channels, shares, the length used up to the opponents' last line, the highest
# player against it), so 'd6_describe_repeats' is repaired and no longer listed: level pairs of one family at
# word-trigram Jaccard >= .5 went from 234 scene + 73 run-copy pairs (D6 and the D2 descriptions BC180 B now gets)
# to 0 (TEXT's fix1/text/check.py; coaching_eval levels re-run in fix1/text/levels_after.json).
# Fix round 1 (DETECT, critic S-5): assess.py ranks a family's moments by how many levels' rules fail (or
# demonstrate) each opportunity, so one family cites the same moments at every level; 'evidence_repicked' is
# repaired and no longer listed.
GENERIC_MIN_WORDS = 3       # a sentence that is a generic phrase plus fewer than this many other words says nothing specific
# The words a role is singled out by (team_shape role names; DESIGN 3.4), for "is the doubtful player named?".
ROLE_WORDS = {'L': (r'left[- ](?:back|full-back|wing-back|defender)',),
              'R': (r'right[- ](?:back|full-back|wing-back|defender)',),
              'C': (r'(?:centre|center)-back', r'central defender')}
# Checks of the scene expectations that fail because of a rule another role owns, recorded here so the test stays
# green while the defect is open and fails (as 'fixed') the day it is repaired: (scene/variant, level, token) -> why.
# Fix round 1 (DETECT): the level-1 identity defect of identity_crossing is repaired (critic B-6: the reframed
# advance keeps D3's identity guardrail), so no entry is open.
KNOWN_DEFECTS = {}
# Fix round 1 (critic N3): the stretches of the held-out clips the verification round studied moment by moment (BC60
# B's advances at 1.4-59.8 s and their level-1 reviews, BC303 B's at 222-245 s, the cited presser and far-side
# moments): their D3 and D4 items are 'seen', scored apart from the held-out ones. (run, t0, t1, detectors)
SEEN_SPANS = (('BC60', 0.0, 60.0, (D3, D4)), ('BC303', 222.0, 303.0, (D3, D4)))
# Independent copies of the fix-round-1 rules the G1+ invariants check (fix1_problems; critic section 6): a D3 area
# reaches at most LEVEL_M upfield of the deepest channel opponent (B-8, alternatives.LEVEL_M); a spell merges across
# interruptions of SPELL_MERGE_S (DESIGN 3.6); "someone back" is judged with the ball in the final third (B-1).
LEVEL_M = 2.0
SPELL_MERGE_S = 1.0
FINAL_THIRD = 2.0/3.0
# Possession words a phase-unknown headline or short label must not use (B-4); 11v11 role names a drill offered below
# 11v11 must not use (S-9, DRILLS fix round 1).
PHASE_WORDS = re.compile(r'\bwith the ball\b|\battack(?:s|ing|ed|ers?)?\b|\bin attack\b|\bin possession\b', re.I)
ELEVEN_ROLE_WORDS = re.compile(r'\b(?:full-?backs?|cent(?:re|er)[- ]backs?|holding midfielders?|wing[- ]?backs?|'
                               r'back (?:four|five)|sweepers?)\b', re.I)
# Fix round 2 (LEAD decision (b), game-3 G1+): a 'moments worth a look' review carries this limitation code; its text
# never uses verdict framing, a prescription or blame (an independent list: explain.VERDICT_WORDS_RE is TEXT's own),
# and it never says "possible issue". MOMENTS_SHARE_ENDS: a share this close to 0 or 1 is said as a count.
MOMENTS = 'moments_worth_a_look'
MOMENTS_COUNTS = ('clearMoments', 'clearSpells', 'decidedMoments', 'failingMoments')
MOMENTS_WORDS = re.compile(r'\b(?:work(?:ing)? on|try (?:this|to)|next step|should|must|needs? to|ha(?:ve|s) to|'
                           r'mistakes?|errors?|wrong|fail\w*|poor\w*|bad(?:ly)?|possible issues?|weakness\w*|'
                           r'drills?)\b', re.I)
MOMENTS_SHARE_ENDS = .005
NUMBER_MODES = ('none', 'counts', 'rounded_metres', 'metres', 'shares', 'rates')
TEAM_LEVEL = (D2, D4, D6)               # checks that name nobody: the whole team must be in view (DESIGN 4.8)
S_PHASE = {D1: 'opp', D2: 'opp', D3: 'own', D4: 'own', D5: 'opp', D6: 'own'}     # the phase each check judges


# ---------- small helpers ----------
def _r(x, d=3):
    return None if x is None else round(float(x), d)


def binomial_tail(k, n, p):
    """P(X >= k) for X ~ Binomial(n, p), from scipy (an implementation independent of assess.binom_tail, which the
    review and praise rules use: fix round 3, LEAD decisions (2) and (3)); 1 for k <= 0, 0 for k > n."""
    from scipy.stats import binom
    k, n = int(k), int(n)
    if k <= 0:
        return 1.0
    if k > n:
        return 0.0
    return float(binom.sf(k-1, n, float(p)))


def sha_json(obj):
    """sha256 of canonical JSON (sorted keys, no spaces): the byte identity of a tier file as the store writes it."""
    data = json.dumps(obj, sort_keys=True, separators=(',', ':'), allow_nan=False)
    return hashlib.sha256(data.encode('utf-8')).hexdigest()


def _safe_dir(path):
    p = Path(path).resolve()
    if any(part.lower() in ('runs', 'uploads') for part in p.parts):
        raise SystemExit(f'refusing to write under a runs/ or uploads/ folder: {p}')
    return p


def write_json(path, obj, indent=1):
    path = _safe_dir(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=indent, allow_nan=False, ensure_ascii=True), encoding='ascii')
    return path


def write_text(path, text):
    """ASCII text with CRLF line ends (the repository convention; the sheets open in Excel and Notepad alike)."""
    path = _safe_dir(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(text.replace('\r\n', '\n').replace('\n', '\r\n').encode('ascii', 'replace'))
    return path


def trigrams(text):
    words = re.findall(r"[a-z0-9']+", (text or '').lower())
    return {tuple(words[i:i+3]) for i in range(len(words)-2)}


def jaccard(a, b):
    ta, tb = trigrams(a), trigrams(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb)/len(ta | tb)


def words_in(text, phrases):
    """The phrases found in text as whole words (case-insensitive)."""
    low = (text or '').lower()
    return [p for p in phrases if re.search(r'(?<![a-z])'+re.escape(p.lower())+r'(?![a-z])', low)]


def findings_of(rep):
    return [f for k in FINDING_SECTIONS for f in rep[k]]


def family(f):
    """A finding's family: its key without the times (detector code, team, side/zone/team)."""
    return '|'.join(f['key'].split('|')[:3])


def finding_texts(f):
    return [(k, v) for k, v in (f.get('text') or {}).items() if isinstance(v, str) and v]


def report_texts(rep):
    """Every coach-facing text of a report with a path: finding texts, overview text and notices, possession caveat,
    key-moment and annotation captions, diagram captions, the next objective, drill fields. Source titles and the
    'not measured' category list are quotations or statements of absence (FRONTEND / PDF left them out of the word
    checks for that reason) and are not included."""
    out = []
    for f in findings_of(rep):
        for k, v in finding_texts(f):
            out.append((f'{f["id"]}.{k}', v, f))
    ov = rep['overview']
    if ov.get('text'):
        out.append(('overview.text', ov['text'], None))
    for n in ov['notices']:
        out.append((f'notice.{n["code"]}', n['text'], None))
    for k in ('caveat', 'reason'):
        if ov['possession'].get(k):
            out.append((f'possession.{k}', ov['possession'][k], None))
    for e in rep['keyMoments']:
        if e.get('caption'):
            out.append((f'{e["id"]}.caption', e['caption'], None))
        if e.get('annotation'):
            out.append((f'{e["id"]}.annotation', e['annotation']['caption'], None))
    for d in rep['diagrams']:
        out.append((f'{d["id"]}.caption', d['caption'], None))
    if rep.get('nextObjective'):
        out.append(('nextObjective', rep['nextObjective']['text'], None))
    for p in rep['drills']:
        for k in ('trains', 'why', 'easier', 'harder', 'progress'):
            if p['fields'].get(k):
                out.append((f'drill.{p["drill"]}.{k}', p['fields'][k], None))
    return out


def generic_sentences(text):
    """The sentences of a text that are a generic phrase with fewer than GENERIC_MIN_WORDS other words: "Keep it up."
    or "Stay compact!" say nothing the evidence supports; "Keep it up: going to the ball together is the next step."
    does (the phrase introduces a specific point after the colon)."""
    out = []
    for part in re.split(r'[.!?;]+', text or ''):
        low = ' '.join(re.findall(r"[a-z0-9']+", part.lower()))
        for g in GENERIC:
            rx = r'(?<![a-z])'+re.escape(g)+r'(?![a-z])'
            if re.search(rx, low) and len(re.sub(rx, ' ', low).split()) < GENERIC_MIN_WORDS:
                out.append(g)
    return out


def singled_out(text):
    """Player ids and single-player role words in a text."""
    return ID_RE.findall(text or '')+[m.group(0) for m in ROLE_RE.finditer(text or '')]


def moment_index(obs, roles):
    out = {m['id']: m for m in obs.get('moments') or ()}
    out.update({m['id']: m for m in (roles or {}).get('moments') or ()})
    return out


def opportunity_index(obs, roles):
    out = {o['id']: o for o in obs.get('opportunities') or ()}
    out.update({o['id']: o for o in (roles or {}).get('opportunities') or ()})
    return out


# ---------- the report invariants (every report, synthetic or real) ----------
def report_problems(rep, obs, roles, library=None):
    """[(criterion, problem)] of one v2 report against its own obs and roles files. Empty when it holds every
    invariant this module checks (see the module docstring)."""
    from soccer.coaching import drills as DR, schema as S
    out = []
    add = out.append
    prof = rep['profile']
    level = prof['level']
    for p in S.validate('report', rep):
        add(('schema', p))
    for p in S.cross_check(rep, obs, roles):
        add(('invariant3', p))
    moments, opps = moment_index(obs, roles), opportunity_index(obs, roles)
    fps = float(obs['segment']['fps'] or 25.0)
    texts = report_texts(rep)
    for path, text, _ in texts:
        for p in S.event_problems(text):
            add(('invariant4', f'{path}: {p}'))
    # Fix round 1 (TEXT, critic B-3 / B-4 / S-9 acceptance, game-3 G1+): the presser named is the moment's first
    # defender and never the cover; no possession words in a phase-unknown headline or short; no 11v11 role names
    # below 11v11 (explain.invariant_problems; 0 on the 576 + 196 fix1/text reports, 110 + 56 on the pre-round text)
    from soccer.coaching import explain as EX
    for f in findings_of(rep):
        for p in EX.invariant_problems(f):
            add(('text_invariants', f'{f["key"]}: {p}'))
    # Fix round 1 (EVAL, critic section 6 G1+): the rules the round introduced, re-checked here on the report against
    # its own obs and roles files (presser, phase words, level-1/2 "someone back", strength runs, zone sides, reasons)
    out.extend(fix1_problems(rep, obs, roles, moments, opps))
    # Fix round 2 (EVAL, LEAD decision (b) / (d), game-3 G1+): the 'moments worth a look' reviews, re-checked against
    # the obs and roles files (form, cap, counts, clear moments, count and share stated, no verdict words)
    out.extend(fix2_problems(rep, obs, roles, moments, opps))
    # Fix round 3 (EVAL, critic B-3 validators, game-3 G1+): the moment's own actor and third, the aggregates and
    # strength counts, the margin behind the ball, TEXT's report-level invariants and the drill rules, re-checked
    # against the obs and roles files
    out.extend(fix3_problems(rep, obs, roles, moments, opps, library))
    # criterion 8: every key moment is its opportunity's own frame, seek exact, the focus on the observed position
    for e in rep['keyMoments']:
        m = moments.get(e['id'])
        o = opps.get(e['opportunity'])
        if m is None or o is None:
            add(('criterion8', f'{e["id"]}: no moment / opportunity in the obs or roles file'))
            continue
        if e['frame'] != m['frame'] or abs(e['time']-m['t']) > 1e-3:
            add(('criterion8', f'{e["id"]}: frame/time {e["frame"]}/{e["time"]} != moment {m["frame"]}/{m["t"]}'))
        if not (o['frame0'] <= e['frame'] <= o['frame1']):
            add(('criterion8', f'{e["id"]}: frame {e["frame"]} outside its opportunity {o["frame0"]}-{o["frame1"]}'))
        if abs(e['seek']-round((e['frame']+.5)/e['fps'], 4)) > 1e-4 or abs(e['fps']-fps) > 1e-6:
            add(('criterion8', f'{e["id"]}: seek {e["seek"]} != (frame + 0.5) / fps'))
        scene = (m.get('scene') or {}).get('players') or {}
        for el in ((e.get('diagram') or {}).get('elements') or ()):
            if el.get('type') != 'focus':
                continue
            p = scene.get(el.get('id'))
            if p is None:
                add(('criterion8', f'{e["id"]}: focus {el.get("id")} is not an observed player of the moment'))
            elif p.get('kind') != 'O' or abs(p['x']-el['x']) > .006 or abs(p['y']-el['y']) > .006:
                add(('criterion8', f'{e["id"]}: focus {el.get("id")} at {el["x"]},{el["y"]} not the observed '
                                   f'{p["x"]},{p["y"]} ({p.get("kind")})'))
    # criterion 7: a work item rests on teammates (a relation or team-state claim) and passed the context guardrail
    for f in rep['needsWork']:
        kinds = {c['type'] for c in f['claims']}
        if not kinds & {'relation', 'team_state'}:
            add(('criterion7', f'{f["key"]}: no teammate relation or team-state claim'))
        if f['guardrails']['context'] != 'pass':
            add(('criterion7', f'{f["key"]}: context guardrail {f["guardrails"]["context"]}'))
    # criterion 11: a strength rests on the level's demonstrated spells, with evidence
    ev = prof['evidence']
    for f in rep['strengths']:
        a = f['aggregate']
        if a['distinctDemonstratedSpells'] < ev['strengthSpells'] or not f['timestamps']:
            add(('criterion11', f'{f["key"]}: {a["distinctDemonstratedSpells"]} demonstrated spells '
                                f'(level needs {ev["strengthSpells"]}), {len(f["timestamps"])} moments'))
    # criterion 12: a drill trains its finding's detector and fits the age's safety floor
    lib = library or DR.load_library()
    by_id = {f['id']: f for f in findings_of(rep)}
    for p in rep['drills']:
        d = DR.get(lib, p['drill'])
        if d is None:
            add(('criterion12', f'{p["drill"]}: not in the library'))
            continue
        f = by_id.get(p['finding']) if p['finding'] else None
        if f is not None and f['observation'] not in d['related_tactical_detector']:
            # level 1 reframes D3 as D4 (DESIGN 2.3): the reframed finding is a behind_ball finding already
            add(('criterion12', f'{p["drill"]}: trains {d["related_tactical_detector"]}, not {f["observation"]}'))
        if p['finding'] is None and not p['optional']:
            add(('criterion12', f'{p["drill"]}: a drill without a finding must be optional'))
        for q in DR.safety_problems(d, prof['ageGroup'], prof.get('safety')):
            add(('criterion12', f'{p["drill"]}: {q}'))
        # fix round 1 (DRILLS, critic S-9): never above the team's match format or its player count, and no 11v11
        # role names in a drill offered below 11v11
        for q in DR.format_problems(d, prof):
            add(('criterion12', f'{p["drill"]}: {q}'))
        if DR.team_players(prof) != 11:
            for k in ('trains', 'why', 'easier', 'harder', 'progress'):
                m_ = ELEVEN_ROLE_WORDS.search(p['fields'].get(k) or '')
                if m_:
                    add(('criterion12', f'{p["drill"]}.{k}: 11v11 role word "{m_.group(0)}" below 11v11'))
        if _age_years(prof['ageGroup']) <= 11 and (d.get('safety') or {}).get('heading') != 'none':
            add(('criterion12', f'{p["drill"]}: heading at {prof["ageGroup"]}'))
    # criterion 13: banned and generic words, repetition across the report's explanations
    banned = set(prof['register']['banned']) | set(DESIGN_BANNED.get(level, ())) | {'mistake'}
    for path, text, f in texts:
        for w in words_in(text, sorted(banned)):
            add(('criterion13', f'{path}: banned "{w}" at level {level}'))
        for w in generic_sentences(text):
            add(('criterion13', f'{path}: generic "{w}"'))
    expl = [(f['id'], (f.get('text') or {}).get('explanation') or '') for f in findings_of(rep)]
    for i in range(len(expl)):
        for j in range(i+1, len(expl)):
            jac = jaccard(expl[i][1], expl[j][1])
            if jac >= TRIGRAM_LIMIT:
                add(('criterion13', f'{expl[i][0]} / {expl[j][0]}: explanation trigram Jaccard {jac:.2f}'))
    # criterion 3 (and DESIGN 4.9 caps at every level)
    if len(rep['needsWork']) > MAX_WORK[level]:
        add(('criterion3', f'{len(rep["needsWork"])} work items at level {level} (cap {MAX_WORK[level]})'))
    if level <= 2:
        for f in rep['needsWork']+rep['reviews']:
            for k, v in finding_texts(f):
                named = ID_RE.findall(v)+([m.group(0) for m in ROLE_RE.finditer(v)] if level == 1 else [])
                if named:
                    add(('criterion3', f'{f["key"]}.{k}: names {named[0]} at level {level}'))
                if FORMATION_DISCIPLINE.search(v):
                    add(('criterion3', f'{f["key"]}.{k}: formation-discipline wording at level {level}'))
            if f['observation'] in (D2, D5):
                add(('criterion3', f'{f["key"]}: {f["observation"]} criticised at level {level}'))
            for t in f['timestamps']:
                e = next((x for x in rep['keyMoments'] if x['id'] == t['evidence']), None)
                for el in ((e or {}).get('diagram') or {}).get('elements') or ():
                    if el.get('type') == 'focus' and el.get('label') is True:
                        add(('criterion3', f'{f["key"]}: focus ring labelled at level {level}'))
    return out


def _attack_m(x, y, side, L, W):
    """Normalized pitch -> the team's attack frame in metres (defending the left goal keeps the axes)."""
    return (float(x)*L, float(y)*W) if side == 'left' else ((1.0-float(x))*L, (1.0-float(y))*W)


def demo_run_seconds(t, outs, step, merge_s=SPELL_MERGE_S):
    """Seconds of the longest run of +1 samples (t1 - t0 + step), merged across interruptions of at most merge_s
    (DESIGN 3.6), written independently of situations.spells."""
    ts = [float(t[k]) for k, o in enumerate(outs) if o == 1]
    if not ts:
        return 0.0
    best, start, prev = 0.0, ts[0], ts[0]
    for x in ts[1:]:
        if x-prev-step > merge_s+.5*step:
            best = max(best, prev-start+step)
            start = x
        prev = x
    return max(best, prev-start+step)


def _covered_sample(R, k, params, dm=False):
    """A wide-defender advance sample covered by (a) the channel, (c) the near centre-back's shift (the level's
    cbShiftM where the shift is measured, else the stored cbAdjusted) or (e) a geometric interchange (critic B-1 / S-4);
    dm: also (d) the holding midfielder dropping in where the level rotates him (useDmRotation)."""
    def at(key):
        v = R.get(key)
        return v[k] if v is not None and k < len(v) else None
    shift, cb_m = at('nearCbShiftM'), params.get('cbShiftM')
    cb = (shift >= float(cb_m)) if (shift is not None and cb_m is not None) else bool(at('cbAdjusted'))
    rot = bool(dm and params.get('useDmRotation') and at('dmDropped'))
    return bool(at('channelOccupied') or cb or at('interchangeCover') or rot)


def _behind_too_few(params, behind, margin):
    """Level 1's behind-the-ball rule on a reframed advance sample (DESIGN 4.4, written out independently): fewer than
    minBehind of ours behind the ball (or, with behindRule 'margin', a rest margin under minMargin); True when the
    count says too few, False when it says enough or is not measured."""
    if params.get('behindRule') == 'margin':
        return margin is not None and params.get('minMargin') is not None and margin < float(params['minMargin'])
    return behind is not None and params.get('minBehind') is not None and behind < int(params['minBehind'])


def zone_side_problems(rep, obs, roles, moments=None, opps=None):
    """[problem]: suggested areas on the wrong side of the threat they answer (critic B-8, game-3 G1+): an area for a
    wide-defender advance (D3, or level 1's reframed one) reaching more than LEVEL_M upfield of the deepest opponent in
    the vacated channel at its moment (his outer third, goal-side of the wide defender, our half), and a
    behind-the-ball area reaching ahead of the ball. Positions in the team's attack frame (alternatives.attack_side);
    a moment whose side, wide defender or ball cannot be read is skipped."""
    from soccer.coaching import alternatives as AL
    moments = moment_index(obs, roles) if moments is None else moments
    opps = opportunity_index(obs, roles) if opps is None else opps
    L, W = float(obs['timeline']['scale']['lengthM']), float(obs['timeline']['scale']['widthM'])
    by_id = {f['id']: f for f in findings_of(rep)}
    out = []
    for a in rep['alternatives']:
        poly = (a.get('zone') or {}).get('polygon')
        f, m = by_id.get(a['finding']), moments.get(a['evidence'])
        if not poly or f is None or m is None:
            continue
        side = AL.attack_side(obs, m)
        if side not in ('left', 'right'):
            continue
        top = max(_attack_m(q[0], q[1], side, L, W)[0] for q in poly)
        scene = m.get('scene') or {}
        players = scene.get('players') or {}
        o = opps.get(m.get('opportunity'))
        rel = (a.get('improves') or {}).get('relation')
        if o is not None and o['detector'] == D3:
            wd, lane = (o['actors']['primary'] or [None])[0], o.get('variant')
            pw = players.get(wd)
            if pw is not None and lane in ('left', 'right'):
                xw = _attack_m(pw['x'], pw['y'], side, L, W)[0]
                threat = []
                for pid, p in players.items():
                    if p.get('team') == rep['team'] or p.get('goalkeeper'):
                        continue
                    X, Y = _attack_m(p['x'], p['y'], side, L, W)
                    if (Y <= W/3.0 if lane == 'left' else Y >= 2.0*W/3.0) and X < xw and X < L/2.0:
                        threat.append((X, pid))
                if threat:
                    x0, pid = min(threat)
                    if top > x0+LEVEL_M+.05:
                        out.append(f'{a["id"]}: the area reaches {top-x0:.1f} m upfield of {pid}, the deepest opponent '
                                   f'in the vacated channel (at most {LEVEL_M:g} m)')
        if rel == 'behind_ball' and scene.get('ball') is not None:
            bx = _attack_m(scene['ball'][0], scene['ball'][1], side, L, W)[0]
            if top > bx+.05:
                out.append(f'{a["id"]}: a behind-the-ball area reaching {top-bx:.1f} m ahead of the ball')
    return out


def fix1_problems(rep, obs, roles, moments=None, opps=None):
    """[(criterion, problem)]: the rules fix round 1 introduced, checked on one report against its own obs and roles
    files, independently of the stage that applied them (critic section 6, game-3 G1+):
    - G1+presser (B-3): a pressure finding's presser is the first defender at its first cited moment (the
      relations' fd there) and is never named as the cover;
    - G1+phase (B-4): a finding whose phase is unknown says nothing of who had the ball in its headline or short;
    - G1+someone_back (B-1; amended in fix round 3, critic B-1 / N3): a level-1/2 "someone back" criticism (level
      1's reframed advance, level 2's team-level advance rule) never fails on a covered sample; a failing sample is
      exposed (an opponent in the vacated channel) wherever the ball is, or else has the ball in the final third with
      nobody (level 2) or too few (level 1) behind it;
    - G1+strength_run (B-2): every opportunity a strength cites has a demonstrated run of at least the level's
      spellMinS (merged as DESIGN 3.6);
    - G1+zone_side (B-8): zone_side_problems;
    - G1+reason (S-7): every check not assessed names its reason."""
    from soccer.coaching import contextual as CX
    out = []
    add = out.append
    prof = rep['profile']
    level = prof['level']
    tl = obs['timeline']
    scale = tl['scale']
    L = float(scale['lengthM'])
    step = float(tl['step'])
    moments = moment_index(obs, roles) if moments is None else moments
    opps = opportunity_index(obs, roles) if opps is None else opps
    fs = findings_of(rep)

    def cited(f):
        seen, got = set(), []
        for t in f['timestamps']:
            o = opps.get(t['opportunity'])
            if o is not None and o['id'] not in seen:
                seen.add(o['id'])
                got.append(o)
        return got
    for f in fs:
        claims = f.get('claims') or []
        # B-3
        if f['observation'] == D1 and f['timestamps']:
            actor = next((c['facts'].get('player') for c in claims if c['type'] == 'actor_state'
                          and (c.get('facts') or {}).get('player')), None)
            cover = next((c['facts'].get('player') for c in claims if c['type'] in ('relation', 'team_state')
                          and (c.get('facts') or {}).get('relation') == 'cover'
                          and c['facts'].get('scope') != 'demonstrated' and c['facts'].get('player')), None)
            t0 = f['timestamps'][0]
            m, o = moments.get(t0['evidence']), opps.get(t0['opportunity'])
            fd = None
            if m is not None and o is not None and isinstance(m.get('sample'), int):
                fds = o['relations'].get('fd') or []
                fd = fds[m['sample']] if 0 <= m['sample'] < len(fds) else None
            if actor and fd and actor != fd:
                add(('G1+presser', f'{f["key"]}: presser {actor} is not {fd}, the first defender at the first cited '
                                   f'moment'))
            if actor and cover and actor == cover:
                add(('G1+presser', f'{f["key"]}: {actor} is named as the presser and as the cover'))
        # B-4
        if ((f.get('context') or {}).get('phase') or {}).get('state') not in ('own', 'opp'):
            for k in ('headline', 'short'):
                m_ = PHASE_WORDS.search((f.get('text') or {}).get(k) or '')
                if m_:
                    add(('G1+phase', f'{f["key"]}.{k}: "{m_.group(0)}" while who had the ball is unknown'))
    # B-1, amended in fix round 3 (critic B-1 / N3, COACH F-1: the spec's EARLY DEVELOPMENT full-back example): a
    # level-1/2 failing advance sample is never covered; it is EXPOSED (uncovered with an opponent in the vacated
    # channel, opponentsInChannel >= 1) wherever the ball is, or, not exposed, it has the ball in the final third with
    # nobody behind it (level 2: behindN 0) or fewer than the level's minimum (level 1's behind-the-ball rule)
    if level <= 2:
        for f in rep['needsWork']+rep['reviews']:
            reframed = f['observation'] == D4 and f['key'].split('|')[2] == 'advance'
            params = prof['detectors'][f['observation']]['params']
            team_rule = f['observation'] == D3 and bool(params.get('requireNobodyBehind'))
            if not (reframed or team_rule):
                continue
            for o in cited(f):
                if o['detector'] != D3:
                    continue
                outs, _ = CX.sample_outcomes(o, params, scale, as_detector=D4 if reframed else None)
                R = o['relations']
                n = len(outs)
                bx = R.get('ballX') or [None]*n
                opp_in = R.get('opponentsInChannel') or [None]*n
                behind = R.get('behindN') or [None]*n
                margin = R.get('restMargin') or [None]*n
                off, cov, back = [], [], []
                for k, x in enumerate(outs):
                    if x != -1:
                        continue
                    if _covered_sample(R, k, params, dm=team_rule):
                        cov.append(k)
                        continue
                    if (opp_in[k] or 0) >= 1:
                        continue                # exposed: a failure wherever the ball is
                    if bx[k] is None or bx[k] < FINAL_THIRD*L-1e-6:
                        off.append(k)
                    elif team_rule and behind[k] != 0:
                        back.append(k)
                    elif reframed and not _behind_too_few(params, behind[k], margin[k]):
                        back.append(k)
                if off:
                    add(('G1+someone_back', f'{f["key"]}: {o["id"]} fails on {len(off)} samples with the ball outside '
                                            f'the final third and no opponent in the vacated channel (first at '
                                            f'{R["t"][off[0]]:.1f} s)'))
                if cov:
                    add(('G1+someone_back', f'{f["key"]}: {o["id"]} fails on {len(cov)} covered samples (first at '
                                            f'{R["t"][cov[0]]:.1f} s)'))
                if back:
                    add(('G1+someone_back', f'{f["key"]}: {o["id"]} fails on {len(back)} samples that are not exposed '
                                            f'and keep enough players behind the ball (first at '
                                            f'{R["t"][back[0]]:.1f} s)'))
    # B-2
    for f in rep['strengths']:
        params = prof['detectors'][f['observation']]['params']
        need = float(params.get('spellMinS', 2.0))
        for o in cited(f):
            outs, _ = CX.sample_outcomes(o, params, scale,
                                         as_detector=D4 if f['observation'] == D4 and o['detector'] == D3 else None)
            run = demo_run_seconds(o['relations']['t'], outs, step)
            if run < need-1e-6:
                add(('G1+strength_run', f'{f["key"]}: {o["id"]} demonstrated for {run:.1f} s at most (the level '
                                        f'needs {need:g} s)'))
    # B-8
    for p in zone_side_problems(rep, obs, roles, moments, opps):
        add(('G1+zone_side', p))
    # S-7
    for d in rep['overview']['detectors']:
        if d['status'] == 'not_assessed' and not d.get('reason'):
            add(('G1+reason', f'{d["detector"]}: not assessed without a reason'))
    return out


def is_moments_review(f):
    """A 'moments worth a look' review (fix round 2): the limitation code, whatever else the finding says."""
    return MOMENTS in (f.get('limitations') or ())


def unsettled_intervals(timeline, settle_s):
    """Fix round 3 (LEAD decision (2), FACTS F-3; written independently of assess.settle_index / settled): where a
    failing sample is NOT in settled play, from the timeline's phaseSpans (own / opp / dead, sorted by t0):
    {'dead': [(t0, t1)] the dead balls (a sample inside one, or less than settle_s after its end, is a restart),
    'gaps': [(t0, t1)] the unnamed stretches (a gap of more than 1.5 steps between two consecutive named spans: an
    engine change window or a run it could not name; a coach-marked direct switch between 'own' and 'opp' is a point at
    its middle; a sample closer than settle_s to one is not settled), 'settleS'}; None when settle_s is 0 (no gate)."""
    if not settle_s or settle_s <= 0:
        return None
    step = float(timeline.get('step') or .2)
    spans = sorted((float(s['t0']), float(s['t1']), s['state']) for s in timeline.get('phaseSpans') or ()
                   if s.get('state') in ('own', 'opp', 'dead'))
    gaps = []
    for (_, a1, sa), (b0, _, sb) in zip(spans, spans[1:]):
        if b0-a1 > 1.5*step+1e-9:
            gaps.append((a1, b0))
        elif {sa, sb} == {'own', 'opp'}:
            gaps.append(((a1+b0)/2.0, (a1+b0)/2.0))
    return {'dead': [(a0, a1) for a0, a1, st in spans if st == 'dead'], 'gaps': sorted(gaps),
            'deadStarts': [a0 for a0, _, st in spans if st == 'dead'], 'gapStarts': [a for a, _ in sorted(gaps)],
            'settleS': float(settle_s)}


def _unsettled(index, t):
    """Is time t outside settled play (unsettled_intervals)? Inside a dead ball or less than settleS after its end;
    less than settleS from an unnamed stretch (0 inside it). Spans of one kind do not overlap, so the latest dead ball
    starting at or before t and the unnamed stretches on either side of t decide."""
    import bisect
    if not index:
        return False
    s = index['settleS']
    j = bisect.bisect_right(index['deadStarts'], t+1e-9)-1
    if j >= 0 and t-index['dead'][j][1] < s-1e-9:
        return True
    j = bisect.bisect_right(index['gapStarts'], t+1e-9)-1
    for i in (j, j+1):
        if 0 <= i < len(index['gaps']):
            a, b = index['gaps'][i]
            d = a-t if t < a else (t-b if t > b else 0.0)
            if d < s-1e-9:
                return True
    return False


def _sample_gates(det, R, k, mode, params, unsettled, players):
    """Fix round 3 (LEAD decision (2), critic B-2 / N1): may failing sample k count toward a clear failure? In settled
    play (unsettled_intervals); at least momentsContestS (1 s) from a loose or contested ball (relations contestS, None
    passes); for D5 on 11v11 a recognised line of at least lineMinPlayers (4) with a depth spread lineStdM of at most
    lineStdMaxM (4 m) (None passes); a lateral-gap failure only while the opponent in the gap is free (gapOppFreeM above
    markedM, 4 m; None passes). Written independently of assess._settled_sample."""
    def at(key):
        v = R.get(key)
        return v[k] if v is not None and 0 <= k < len(v) else None
    if _unsettled(unsettled, float(R['t'][k])):
        return False
    cs = at('contestS')
    if params.get('momentsContestS') and cs is not None and cs < float(params['momentsContestS'])-1e-9:
        return False
    if det == D5:
        if players == 11:
            size, std = at('lineSize'), at('lineStdM')
            if params.get('lineMinPlayers') is not None and size is not None and size < int(params['lineMinPlayers']):
                return False
            if params.get('lineStdMaxM') is not None and std is not None and std > float(params['lineStdMaxM'])+1e-9:
                return False
        free = at('gapOppFreeM')
        if mode == 'lateral_gap' and params.get('markedM') is not None and free is not None and \
                free <= float(params['markedM'])+1e-9:
            return False
    return True


def _clear_failing_runs(o, params, scale, step, unsettled=None, players=11):
    """(outcome, runs) of one opportunity under the level's parameters, written independently of assess's
    _clear_moments (fix round 2, LEAD decision (b)): outcome is assess.classify's (the per-opportunity rule, measured
    by G6b and the scenes), runs the [(i0, i1)] of consecutive failing samples (no unclear, demonstrated, out-of-scope
    or missing sample: two samples are consecutive when at most 1.5 steps apart) lasting the level's failure spell
    (t1 - t0 + step >= spellMinS; D2's failMinS). At levels where 11v11 depth is described (params judgeDepth False,
    LEAD (c)) a sample failing on depth alone is not a failure. A D1 'nobody pressed' sample is a failure only while
    the gap from the carrier to our nearest player (d1, by its two neighbouring samples, both measured and at most 2.5
    steps apart) is not shrinking at pressureClosingMS (default 1 m/s): pressure arriving is not a carrier left
    alone. Fix round 3: a failing sample the clear-moment gates exclude (_sample_gates: settled play, the contested
    ball, the D5 line and the marked opponent) breaks a run like an unclear one; unsettled: unsettled_intervals of the
    obs timeline (None: no settle gate); players: the match format's players (the D5 line gate is 11v11's)."""
    from soccer.coaching import assess as AS, contextual as CX
    det = o['detector']
    c = AS.classify(o, params, scale, step)
    if c is None:
        return None, []
    outs, modes = CX.sample_outcomes(o, params, scale)
    R = o['relations']
    t = [float(x) for x in R['t']]
    bad = []
    closing = float(params.get('pressureClosingMS') or 1.0)
    d1 = R.get('d1') or [None]*len(t)
    for k, x in enumerate(outs):
        fail = x == -1
        if fail and det == D6 and params.get('judgeDepth') is False and modes[k] == 'depth':
            fail = False
        if fail and det == D1 and modes[k] == 'no_pressure':
            a, b = max(0, k-1), min(len(t)-1, k+1)
            if b == a or d1[a] is None or d1[b] is None or t[b]-t[a] > 2.5*step+1e-9:
                fail = False
            elif (d1[b]-d1[a])/(t[b]-t[a]) <= -closing+1e-9:
                fail = False
        if fail and not _sample_gates(det, R, k, modes[k], params, unsettled, players):
            fail = False
        bad.append(not fail)
    if det == D2:
        need = float(params.get('failMinS') or params.get('spellMinS') or 2.0)
    else:
        need = float(params.get('spellMinS') or 2.0)
    runs, k = [], 0
    while k < len(t):
        if bad[k]:
            k += 1
            continue
        i0 = k
        while k+1 < len(t) and not bad[k+1] and t[k+1]-t[k] <= 1.5*step+1e-9:
            k += 1
        if t[k]-t[i0]+step >= need-1e-6:
            runs.append((i0, k))
        k += 1
    return c['outcome'], runs


def _clear_opportunity(o, det, phase_want, players=11):
    """The opportunity-level gates of a clear failure (LEAD (b)): identity high and position medium / high, in view
    (the whole team for a team-level check), the relevant phase named by possession-4 or the coach and away from a
    change, no recovering run, interchange or formation change, and a role reading the role guardrail accepts for D3
    and for D5 below 11v11 (coach, stable or carried; a relative reading of medium or high confidence)."""
    q, ph, sg = o['quality'], o['phase'], o['intentSignals']
    if q['identity'] != 'high' or q['position'] not in ('high', 'medium'):
        return False
    if (q['visibility'] != 'complete') if det in TEAM_LEVEL else (q['visibility'] == 'none'):
        return False
    if ph['state'] != phase_want or ph['nearChange'] or ph['source'] not in ('possession-4', 'coach'):
        return False
    if sg['recovering'] or sg['interchange'] or sg['formationChangeNear']:
        return False
    if det == D3 or (det == D5 and players != 11):
        prim = (o['actors']['primary'] or [None])[0]
        ref = (o.get('roles') or {}).get(prim) if prim else None
        if ref is None or not (ref['source'] in ('coach', 'stable', 'carried') or
                               (ref['source'] == 'relative' and ref['confidence'] in ('medium', 'high'))):
            return False
    return True


def fix2_problems(rep, obs, roles, moments=None, opps=None):
    """[(criterion, problem)]: the 'moments worth a look' reviews of fix round 2 (LEAD decision (b), game-3 G1+),
    checked on one report against its own obs and roles files, independently of assess and explain:
    - G1+moments_form: a review (in the reviews section), no priority, no drill (its own or the report's), no
      suggested area or no-alternative note, only from the level's momentsFromLevel (3) and where the level names
      'review' for the check; every guardrail passes but the evidence one; the only finding of its family;
    - G1+moments_cap: at most the level's momentsMaxPerReport (1) in one report;
    - G1+moments_counts: its counts are the family's (the opportunities of the check, side or 'team', whose middle
      lies in the report's range): decidedMoments the demonstrated + not demonstrated ones, failingMoments the not
      demonstrated ones (assess.classify), clearMoments those with a clear failing run (_clear_failing_runs and
      _clear_opportunity, re-implemented here), clearSpells their distinct spells (distinctS apart), at least
      momentsMinSpells; the aggregate agrees;
    - G1+moments_clear: every cited moment lies inside a clear failing run of a clear opportunity, its player named;
    - G1+moments_text: the headline says "Moments worth a look"; the explanation states the clear moments of the
      judged ones ("N of (the) M" or "all M"), with the share ("about P%", P the rounded clear share) where the
      level's register shows shares, never a count "of" all opportunities, every cited time, and (fix round 3, FACTS
      F-4) the failing moments that were not clear ("N more fell short less clearly");
    - G1+moments_words: no verdict, prescription or blame word (MOMENTS_WORDS) in its texts or its key-moment
      captions, and the overview's possible issues count only the other reviews;
    - G1+moments_base (fix round 3, LEAD decision (2)): the level has a measured baseFailShare and baseTestAlpha, the
      recomputed clear spells are above the base by the one-sided binomial tail (binomial_tail <= alpha) over the
      recomputed decided opportunities, and the review's baseShare / baseTail are those.
    Fix round 3 (LEAD decision (2), critic B-2 / N1, FACTS F-3): a clear failing sample must also pass the clear-moment
    gates (_sample_gates: settled play away from dead balls and unnamed stretches by momentsSettleS, momentsContestS
    from a loose or contested ball, a D5 11v11 line of lineMinPlayers with lineStdM <= lineStdMaxM, a lateral-gap
    opponent free beyond markedM); an excluded sample breaks the clear run."""
    from soccer.coaching import assess as AS, language as LG
    out = []
    add = out.append
    prof = rep['profile']
    level = int(prof['level'])
    tl = obs['timeline']
    scale = tl['scale']
    step = float(tl.get('step') or .2)
    moments = moment_index(obs, roles) if moments is None else moments
    opps = opportunity_index(obs, roles) if opps is None else opps
    fs = findings_of(rep)
    mws = [f for f in fs if is_moments_review(f)]
    caps = [dp['params'].get('momentsMaxPerReport') for dp in prof['detectors'].values()
            if dp['enabled'] and dp['params'].get('momentsMaxPerReport') is not None]
    cap = int(max(caps)) if caps else 0
    if len(mws) > cap:
        add(('G1+moments_cap', f'{len(mws)} moments-worth-a-look reviews at level {level} (cap {cap})'))
    n_issues = sum(1 for f in rep['reviews'] if not is_moments_review(f))
    said = re.search(r'There (?:is (one)|are (\d+)) possible issues? to review', (rep['overview'].get('text') or ''))
    if said and int(said.group(2) or 1) != n_issues:
        add(('G1+moments_words', f'overview: {said.group(0)!r} with {n_issues} possible issues besides '
                                 f'{len(mws)} moments worth a look'))
    if not mws:
        return out
    lo = float(rep['range']['t0']) if rep.get('range') else float(obs['segment']['t0'])
    hi = float(rep['range']['t1']) if rep.get('range') else float(obs['segment']['t1'])
    shares_shown = NUMBER_MODES.index(prof['register']['numbers']) >= NUMBER_MODES.index('shares')
    all_opps = list(opps.values())
    by_opp = {}
    for mm in moments.values():
        by_opp.setdefault(mm.get('opportunity'), []).append(mm)
    players = int(str(tl.get('format') or '11v11').split('v')[0])
    for f in mws:
        key, det = f['key'], f['observation']
        dp = prof['detectors'][det]
        p = dp['params']
        fam = family(f)
        # ---- form
        if f['verdict'] != 'review' or not any(x['id'] == f['id'] for x in rep['reviews']):
            add(('G1+moments_form', f'{key}: verdict {f["verdict"]}, not a review'))
        if f.get('priority') is not None:
            add(('G1+moments_form', f'{key}: priority {f["priority"]}'))
        if f.get('drills') or any(d.get('finding') == f['id'] for d in rep['drills']):
            add(('G1+moments_form', f'{key}: a drill'))
        if any(a.get('finding') == f['id'] for a in rep['alternatives']):
            add(('G1+moments_form', f'{key}: a suggested area'))
        if p.get('momentsFromLevel') is None or level < int(p['momentsFromLevel']):
            add(('G1+moments_form', f'{key}: at level {level} (from level {p.get("momentsFromLevel")})'))
        if 'review' not in dp['verdicts']:
            add(('G1+moments_form', f'{key}: the level names no review for {det}'))
        g = f['guardrails']
        bad_g = [k for k, v in g.items() if k != 'evidence' and v == 'fail']
        if bad_g or g.get('evidence') == 'pass':
            add(('G1+moments_form', f'{key}: guardrails {bad_g or "evidence pass"}'))
        same = [x for x in fs if x['id'] != f['id'] and family(x) == fam]
        if same:
            add(('G1+moments_form', f'{key}: beside {same[0]["verdict"]} {same[0]["key"]} of the same family'))
        # ---- counts, recomputed from the obs / roles files
        m_ = f.get('measurements') or {}
        if not all(isinstance(m_.get(k), int) for k in MOMENTS_COUNTS):
            add(('G1+moments_counts', f'{key}: counts {[m_.get(k) for k in MOMENTS_COUNTS]}'))
            continue
        side = key.split('|')[2]
        want = S_PHASE[det]
        famo = []
        for o in all_opps:
            if o['detector'] != det or o.get('abstain') is not None:
                continue
            if not (lo-1e-6 <= (float(o['t0'])+float(o['t1']))/2.0 <= hi+1e-6):
                continue
            ofam = (o.get('variant') or 'team') if det == D3 else \
                ('unknown' if det == D6 and o.get('variant') == 'unknown' else 'team')
            if ofam == side:
                famo.append(o)
        judged = []
        unsettled = unsettled_intervals(tl, p.get('momentsSettleS'))
        for o in famo:
            outcome, runs = _clear_failing_runs(o, p, scale, step, unsettled, players)
            if outcome is not None:
                judged.append((o, outcome, runs))
        ids = sorted(o['id'] for o, _, _ in judged)
        if sorted(f.get('opportunities') or []) != ids:
            add(('G1+moments_counts', f'{key}: cites a family of {len(f.get("opportunities") or [])} opportunities, '
                                      f'{len(ids)} judged in the range'))
        dem = sum(1 for _, oc, _ in judged if oc == 'demonstrated')
        nd = [(o, runs) for o, oc, runs in judged if oc == 'not_demonstrated']
        clear = {}
        for o, runs in nd:
            if not runs or not _clear_opportunity(o, det, want, players):
                continue
            named = o['relations'].get('primaryNamed')
            ms = [mm for mm in by_opp.get(o['id'], ()) if isinstance(mm.get('sample'), int)
                  and any(i0 <= mm['sample'] <= i1 for i0, i1 in runs)
                  and not (named is not None and 0 <= mm['sample'] < len(named) and named[mm['sample']] is False)]
            if ms:
                clear[o['id']] = (o, runs)
        spells, last = 0, None
        apart = float(p.get('distinctS') or 10.0)
        for o, _ in sorted(clear.values(), key=lambda x: float(x[0]['t0'])):
            if last is None or float(o['t0'])-last >= apart-1e-9:
                spells += 1
                last = float(o['t0'])
        got = {'decidedMoments': dem+len(nd), 'failingMoments': len(nd), 'clearMoments': len(clear),
               'clearSpells': spells}
        for k_, v in got.items():
            if m_[k_] != v:
                add(('G1+moments_counts', f'{key}: {k_} {m_[k_]}, recomputed {v}'))
        need = int(p.get('momentsMinSpells') or 3)
        if m_['clearSpells'] < need or spells < need:
            add(('G1+moments_counts', f'{key}: {m_["clearSpells"]} clear spells stated, {spells} recomputed (the level '
                                      f'needs {need})'))
        # ---- the base-rate test (fix round 3, LEAD decision (2)): the clear spells significantly above what the
        # level's baseFailShare predicts for the decided opportunities (one-sided binomial tail at baseTestAlpha)
        base, alpha = p.get('baseFailShare'), p.get('baseTestAlpha')
        if base is None or alpha is None:
            add(('G1+moments_base', f'{key}: a review without a measured base share or test level (base {base}, '
                                    f'alpha {alpha})'))
        else:
            tail = binomial_tail(spells, dem+len(nd), float(base))
            if tail > float(alpha)+1e-12:
                add(('G1+moments_base', f'{key}: {spells} clear spells of {dem+len(nd)} decided against the base '
                                        f'{float(base):.3f}: tail {tail:.4f} above {float(alpha):g}'))
            if m_.get('baseShare') is None or abs(float(m_['baseShare'])-float(base)) > .0006:
                add(('G1+moments_base', f'{key}: baseShare {m_.get("baseShare")}, the level\'s base {base}'))
            if m_.get('baseTail') is None or abs(float(m_['baseTail'])-tail) > 6e-5:
                add(('G1+moments_base', f'{key}: baseTail {m_.get("baseTail")}, recomputed {tail:.5f}'))
        a_ = f.get('aggregate') or {}
        if (a_.get('demonstrated'), a_.get('notDemonstrated')) != (dem, len(nd)):
            add(('G1+moments_counts', f'{key}: aggregate {a_.get("demonstrated")} / {a_.get("notDemonstrated")}, '
                                      f'recomputed {dem} / {len(nd)}'))
        # ---- every cited moment is clear
        if not f['timestamps']:
            add(('G1+moments_clear', f'{key}: no moment cited'))
        for ts in f['timestamps']:
            mm, hit = moments.get(ts['evidence']), clear.get(ts['opportunity'])
            if mm is None or hit is None or not isinstance(mm.get('sample'), int) or \
                    not any(i0 <= mm['sample'] <= i1 for i0, i1 in hit[1]):
                add(('G1+moments_clear', f'{key}: the moment at {float(ts["t"]):.2f} s is not inside a clear failing '
                                         f'run'))
        # ---- text
        text = f.get('text') or {}
        exp = str(text.get('explanation') or '')
        if not str(text.get('headline') or '').startswith('Moments worth a look'):
            add(('G1+moments_text', f'{key}: headline {str(text.get("headline"))[:60]!r}'))
        c_, d_ = m_['clearMoments'], m_['decidedMoments']
        if c_ >= d_:
            stated = re.search(rf'\b[Aa]ll {d_}(?!\d)', exp)
        else:
            stated = re.search(rf'(?<![\d.]){c_} of (?:the )?{d_}(?![\d.])', exp)
        if not stated:
            add(('G1+moments_text', f'{key}: "{c_} of {d_}" (clear of the judged) is not stated'))
        elif c_ < d_ and shares_shown and MOMENTS_SHARE_ENDS < c_/d_ < 1.0-MOMENTS_SHARE_ENDS:
            # the share in the count's own sentence (levels 5-7 place it after the count or after its noun)
            pct = int(round(100.0*c_/d_))
            rest = re.split(r'\.(?:\s|$)', exp[stated.start():], maxsplit=1)[0]
            sh = re.search(r'about (\d+)%', rest)
            if sh is None:
                add(('G1+moments_text', f'{key}: the share of the judged ones (about {pct}%) is not stated'))
            elif int(sh.group(1)) != pct:
                add(('G1+moments_text', f'{key}: share stated {sh.group(1)}%, clear share {pct}%'))
        n_all = a_.get('opportunities')
        if isinstance(n_all, int) and n_all != d_ and re.search(rf'(?<![\d.])\d+ of (?:the )?{n_all}(?![\d.])', exp):
            add(('G1+moments_text', f'{key}: a count "of {n_all}" (all opportunities, not the judged ones)'))
        # fix round 3 (FACTS F-4 = COACH F-10): the failures that were not clear are counted, so clear + less clear +
        # met add up to the judged ones; the sentence that says "less clear" states failingMoments - clearMoments
        more = m_['failingMoments']-c_
        if more > 0:
            said_more = [s for s in re.split(r'(?<=[.;])\s+', exp) if re.search(r'\bless clear', s, re.I)]
            nums = {int(x) for s in said_more for x in re.findall(r'(?<![\d.:])(\d+)(?![\d.:])', s)}
            nums |= {1 for s in said_more if re.search(r'\bone\b', s, re.I)}
            if more not in nums:
                add(('G1+moments_text', f'{key}: the {more} failing moments that were not clear are not counted '
                                        f'("less clear" sentence: {said_more[:1]})'))
        for ts in f['timestamps']:
            if LG.fmt_time(ts['t']) not in exp:
                add(('G1+moments_text', f'{key}: the moment {LG.fmt_time(ts["t"])} is not given'))
        # ---- words
        own = [(k, v) for k, v in finding_texts(f)]
        own += [(f'{e["id"]}.caption', e.get('caption') or '') for e in rep['keyMoments']
                if e.get('finding') == f['id']]
        for k, v in own:
            w = MOMENTS_WORDS.search(v)
            if w:
                add(('G1+moments_words', f'{key}.{k}: "{w.group(0)}"'))
    return out


def _family_judged(f, rep, obs, opps):
    """[(opportunity, outcome)] of a finding's family recomputed from the obs / roles opportunities (written
    independently of assess._family / classify's callers): the check's opportunities (level 1's reframed advance: the
    wide-defender ones judged with the behind-the-ball rule) with no abstain, whose middle lies in the report's range,
    of the finding's family (D3 its side, D6 'unknown' apart, else 'team'), with an outcome under the level's
    parameters (assess.classify, the per-opportunity rule G6b measures)."""
    from soccer.coaching import assess as AS
    prof = rep['profile']
    det, side = f['observation'], f['key'].split('|')[2]
    p = prof['detectors'][det]['params']
    tl = obs['timeline']
    lo = float(rep['range']['t0']) if rep.get('range') else float(obs['segment']['t0'])
    hi = float(rep['range']['t1']) if rep.get('range') else float(obs['segment']['t1'])
    reframed = det == D4 and side == 'advance'
    out = []
    for o in opps.values():
        if o['detector'] != (D3 if reframed else det) or o.get('abstain') is not None:
            continue
        if not (lo-1e-6 <= (float(o['t0'])+float(o['t1']))/2.0 <= hi+1e-6):
            continue
        if reframed:
            fam = 'advance'
        elif det == D3:
            fam = o.get('variant') or 'team'
        else:
            fam = 'unknown' if det == D6 and o.get('variant') == 'unknown' else 'team'
        if fam != side:
            continue
        c = AS.classify(o, p, tl['scale'], float(tl.get('step') or .2), as_detector=D4 if reframed else None)
        if c is not None:
            out.append((o, c['outcome']))
    return out


def _third(x, length):
    """'own_third' / 'middle_third' / 'final_third' of a depth in metres from our goal line (None when unknown)."""
    if x is None or not math.isfinite(float(x)):
        return None
    return 'own_third' if x < length/3.0 else ('middle_third' if x < 2.0*length/3.0 else 'final_third')


COUNT_WORD = r'(?:\d+|none|no one|nobody|one|two|three|four|five|six|seven|eight|nine|ten|eleven|all)'
BEHIND_COUNT = re.compile(r'\b'+COUNT_WORD+r'\b[^.;:]*?\bbehind the ball\b', re.I)
BEHIND_MARGIN = re.compile(r'\bat least \d+(?:\.\d+)? m\b|\bwell\b', re.I)


def fix3_problems(rep, obs, roles, moments=None, opps=None, library=None):
    """[(criterion, problem)]: what the fix-round-3 texts rest on, checked on one report against its own obs and roles
    files independently of assess and explain (critic B-3 validators, game-3 G1+):
    - G1+moment_actor (FACTS F-2): a wide-defender finding's actorId is the advancing defender of its first cited
      opportunity, a back-line finding's gap pair (gapLeftId / gapRightId) the relations' pair at its first cited
      moment with actorId one of them; players.primary names actorId (and, for a gap, supporting the other);
    - G1+moment_zone (FACTS F-5): momentZone is the third of the first cited moment, recomputed from that sample (the
      carrier's zone for pressure and cover, the ball's depth for the far side, the advance and behind the ball);
    - G1+aggregate (FACTS F-6 / COACH F-3): a finding's aggregate (opportunities, demonstrated, not demonstrated,
      unclear) is its family recomputed (_family_judged); a strength never counts its demonstrations "of" all its
      opportunities when some were unclear;
    - G1+behind_margin (FACTS F-1 = COACH F-2): a count of ours behind the ball in a finding text states the margin it
      was measured with ("at least N m" or "well");
    - G1+report_text: TEXT's report-level invariants (report.text_problems: the objective's origin, the declared
      style, the capital, "professional" for a youth team or below level 5);
    - G1+drills (DRILLS fix round 3, coach F-6 / F-7 / F-8 = facts F-10): no drill from below the fallback floor (the
      level just below the profile's, or the age's typical level when that is lower, never below 1; written out
      here, not read from drills.py), no optional drill whose progress line compares with "this match" when the
      reported findings decided no situation of its check in the check's phase, no repetition word (REPETITION) in
      any drill field."""
    from soccer.coaching import drills as DR, profiles as PR, report as RP, schema as S
    out = []
    add = out.append
    prof = rep['profile']
    level = int(prof['level'])
    tl = obs['timeline']
    L = float(tl['scale']['lengthM'])
    moments = moment_index(obs, roles) if moments is None else moments
    opps = opportunity_index(obs, roles) if opps is None else opps
    for f in findings_of(rep):
        key, det = f['key'], f['observation']
        m_ = f.get('measurements') or {}
        ts0 = f['timestamps'][0] if f['timestamps'] else None
        mm = moments.get(ts0['evidence']) if ts0 else None
        o = opps.get(ts0['opportunity']) if ts0 else None
        k = mm.get('sample') if mm else None
        R = (o or {}).get('relations') or {}

        def at(name):
            v = R.get(name)
            return v[k] if isinstance(k, int) and v is not None and 0 <= k < len(v) else None
        # ---- F-2: the moment's own players
        prim = (f.get('players') or {}).get('primary') or []
        sup = (f.get('players') or {}).get('supporting') or []
        if o is not None and det == D3 and o['detector'] == D3 and m_.get('actorId') is not None:
            wd = (o['actors']['primary'] or [None])[0]
            if ID_RE.fullmatch(str(wd or '')) and m_['actorId'] != wd:
                add(('G1+moment_actor', f'{key}: actorId {m_["actorId"]}, the first cited advance is {wd}\'s'))
            if prim[:1] != [m_['actorId']]:
                add(('G1+moment_actor', f'{key}: players.primary {prim} is not the moment\'s {m_["actorId"]}'))
        if o is not None and det == D5 and ID_RE.fullmatch(str(at('gapLeft') or '')) and \
                ID_RE.fullmatch(str(at('gapRight') or '')):
            pair = (at('gapLeft'), at('gapRight'))
            if (m_.get('gapLeftId'), m_.get('gapRightId')) != pair:
                add(('G1+moment_actor', f'{key}: gap pair {m_.get("gapLeftId")} / {m_.get("gapRightId")}, the first '
                                        f'cited moment\'s is {pair[0]} / {pair[1]}'))
            if m_.get('actorId') not in pair:
                add(('G1+moment_actor', f'{key}: actorId {m_.get("actorId")} is not of the moment\'s gap pair'))
            elif prim[:1] != [m_['actorId']] or sup[:1] != [pair[1] if m_['actorId'] == pair[0] else pair[0]]:
                add(('G1+moment_actor', f'{key}: players {prim[:1]} / {sup[:1]} are not the moment\'s gap pair'))
        # ---- F-5: the moment's own third
        if o is not None and m_.get('momentZone') is not None:
            if det == D1 and at('carrierZone'):
                zone = at('carrierZone')
            elif det in (D2, D3, D4) and at('ballX') is not None:
                zone = _third(float(at('ballX')), L)
            else:
                zone = (o.get('ball') or {}).get('zone')
            if zone and zone != m_['momentZone']:
                add(('G1+moment_zone', f'{key}: momentZone {m_["momentZone"]}, the first cited moment was in {zone}'))
        # ---- F-6: the aggregate is the family's; a strength counts the judged ones
        a_ = f.get('aggregate') or {}
        judged = _family_judged(f, rep, obs, opps)
        got = (len(judged), sum(x == 'demonstrated' for _, x in judged),
               sum(x == 'not_demonstrated' for _, x in judged), sum(x == 'unclear' for _, x in judged))
        said = tuple(a_.get(x) for x in ('opportunities', 'demonstrated', 'notDemonstrated', 'unclear'))
        if said != got:
            add(('G1+aggregate', f'{key}: aggregate {said} (opportunities, demonstrated, not, unclear), recomputed '
                                 f'{got}'))
        if f['verdict'] == 'strength' and isinstance(a_.get('opportunities'), int):
            n, dem = a_['opportunities'], a_.get('demonstrated')
            dec = (a_.get('demonstrated') or 0)+(a_.get('notDemonstrated') or 0)
            exp = str((f.get('text') or {}).get('explanation') or '')
            if n != dec and re.search(rf'(?<![\d.]){dem} (?:times )?(?:of|out of) (?:the )?{n}(?![\d.])', exp):
                add(('G1+aggregate', f'{key}: the strength counts {dem} of all {n} opportunities, {n-dec} of them '
                                     f'unclear'))
        # ---- F-1: the margin on a count behind the ball
        for k_, v in finding_texts(f):
            for s in re.split(r'(?<=[.;!?])\s+', v):
                hit = BEHIND_COUNT.search(s)
                if hit and not BEHIND_MARGIN.search(s[:hit.end()]):
                    add(('G1+behind_margin', f'{key}.{k_}: "{hit.group(0)[:90]}" without the margin it was measured '
                                             f'with'))
    for p in RP.text_problems(rep):
        add(('G1+report_text', p))
    # ---- the drills (DRILLS fix round 3)
    typical = PR.typical_level(prof['ageGroup'])
    floor = level-1
    if isinstance(typical, int) and not isinstance(typical, bool):
        floor = min(floor, typical)
    floor = max(1, floor)
    lib = library
    no = rep.get('nextObjective') or {}
    obj_det = S.SKILLS.get(no.get('skill')) if no.get('skill') else None
    for p in rep['drills']:
        if lib is None:
            lib = DR.load_library()
        d = DR.get(lib, p['drill'])
        levels = [int(x) for x in (d or {}).get('development_level') or ()]
        if d is not None and not any(floor <= x <= level for x in levels):
            add(('G1+drills', f'{p["drill"]}: a drill of levels {levels} at level {level} (none from {floor} to '
                              f'{level})'))
        for k_, v in (p.get('fields') or {}).items():
            if isinstance(v, str) and REPETITION.search(v):
                add(('G1+drills', f'{p["drill"]}.{k_}: repetition word "{REPETITION.search(v).group(0)}"'))
        prog = str((p.get('fields') or {}).get('progress') or '')
        if p.get('optional') and p.get('finding') is None and obj_det and re.search(r'\bthis match\b', prog, re.I):
            decided = sum((x['aggregate'].get('demonstrated') or 0)+(x['aggregate'].get('notDemonstrated') or 0)
                          for x in findings_of(rep) if x['observation'] == obj_det and
                          ((x.get('context') or {}).get('phase') or {}).get('state') == S_PHASE[obj_det])
            if decided == 0:
                add(('G1+drills', f'{p["drill"]}.progress: compares with "this match", which decided no situation of '
                                  f'{obj_det} in its phase'))
    return out


def _age_years(age):
    m = re.fullmatch(r'U(\d+)', str(age or ''))
    return int(m.group(1)) if m else 99


# ---------- synthetic scenes through the server's measurement path ----------
_SCENES = {}


def scene_tiers(scene, variant=None, team='A', coach=True, prepared=None):
    """(built, settings, obs, roles) of a synthetic scene through store.measure (the server's path: coach spans folded
    into the measurement key, the evidence anchors merged into roleInputs) and store.build_role_tier. prepared: an
    optional callable (scene, variant) -> (built, settings, series, possession) whose series / possession are reused
    (tests share them with the other coaching tests in one process). Cached per process."""
    import coaching_synth as cs
    from soccer import team_shape
    from soccer.coaching import store as ST
    key = (scene, variant, team, coach)
    if key not in _SCENES:
        if prepared is not None:
            b, st, cd, poss = prepared(scene, variant)
        else:
            b = cs.build(scene, variant)
            st, cd, poss = cs.settings_for(b, level=4), None, None
        sha = ST.sha256_bytes(json.dumps(b.result, separators=(',', ':'), allow_nan=False).encode('utf-8'))
        spans = b.coach_spans if coach else None
        shape = team_shape.clean_settings({})
        obs = ST.measure(b.result, result_sha=sha, shape_settings=shape, settings=st, spans=spans, team=team,
                         series=cd, possession=poss)
        roles = ST.build_role_tier(obs, st, team)
        _SCENES[key] = (b, st, obs, roles)
    return _SCENES[key]


def settings_at(settings, team, level, age=None, style=None):
    """A copy of coaching settings with the team's level, age (default: the level's typical age, DESIGN 2.2) and
    style."""
    import copy
    from soccer.coaching import profiles as PR
    st = copy.deepcopy(settings)
    block = st['teams'][team]
    block['playingLevel'] = int(level)
    block['ageGroup'] = age or PR.default_age(level)
    if style:
        block['style'] = style
    return st


def interpret(obs, roles, settings, team, level, age=None, style=None, t0=None, t1=None, run='synthetic'):
    from soccer import coaching
    return coaching.interpret(obs, roles, settings_at(settings, team, level, age, style), team, t0=t0, t1=t1,
                              meta={'run': run, 'teamName': None})


def scene_report(scene, variant=None, level=4, team='A', age=None, style=None, prepared=None):
    b, st, obs, roles = scene_tiers(scene, variant, team, prepared=prepared)
    return interpret(obs, roles, st, team, level, age, style)


# ---------- the expectations of tools/coaching_synth.py ----------
def scoped(rep, detectors, level):
    """The findings an expectation about these detectors speaks of. Level 1 judges D3's advances with the behind-the-ball
    rule (DESIGN 2.3 L1: "D3 evidence is still used, reframed as D4"), as the 'advance' family of behind_ball: a D3
    expectation at level 1 covers those findings too."""
    dets = set(detectors or DETECTORS)
    out = [f for f in findings_of(rep) if f['observation'] in dets]
    if D3 in dets and level == 1:
        out += [f for f in findings_of(rep) if f['observation'] == D4 and f['key'].split('|')[2] == 'advance'
                and f not in out]
    return out


def _named(f, rep, strict=True):
    """The ids and single-player role words a finding's texts, captions and focus labels carry. strict: every role
    word for one player counts (team-level wording); else only ids, focus labels and role words of the finding's own
    primary actors (the player whose identity the guardrail is about: "the near centre-back did not move across" does
    not name the left-back whose identity crossed)."""
    if strict:
        pick = singled_out
    else:
        sides = {_role_side((r or {}).get('role')) for pid, r in (f.get('roles') or {}).items()
                 if pid in f['players']['primary']}
        pats = [w for k in sides for w in ROLE_WORDS.get(k, ())]
        rx = re.compile(r'\b(?:'+'|'.join(pats)+r')\b', re.I) if pats else None

        def pick(text):
            return ID_RE.findall(text or '')+([m.group(0) for m in rx.finditer(text or '')] if rx else [])
    found = []
    for _, v in finding_texts(f):
        found += pick(v)
    cited = {t['evidence'] for t in f['timestamps']}
    for e in rep['keyMoments']:
        if e['id'] in cited:
            for v in (e.get('caption'), (e.get('annotation') or {}).get('caption')):
                found += pick(v)
            for view in (e.get('diagram'), e.get('videoOverlay')):
                for el in (view or {}).get('elements') or ():
                    if el.get('type') == 'focus' and el.get('label') is True:
                        found.append(f'label:{el.get("id")}')
    return found


def _role_side(role):
    """'L' / 'R' / 'C' for a team_shape or relative back role (LB, LWB, back-1/3 -> L; RB, back-3/3 -> R; CB, LCB,
    RCB, back-2/3 -> C), else ''."""
    r = role or ''
    if r in ('LB', 'LWB'):
        return 'L'
    if r in ('RB', 'RWB'):
        return 'R'
    if r in ('CB', 'LCB', 'RCB'):
        return 'C'
    m = re.fullmatch(r'back-(\d)/(\d)', r)
    if m:
        return 'L' if m.group(1) == '1' else ('R' if m.group(1) == m.group(2) else 'C')
    return ''


def check_expectation(exp, level, rep, obs, roles, memo):
    """('pass' | 'fail' | 'n/a' | 'vacuous', detail) for one expectation at one level. memo collects what a
    cross-level token (same_evidence_across_levels) needs; finish_expectations() judges it."""
    tok, dets, val = exp['expect'], exp['detectors'], exp['value']
    fs = scoped(rep, dets, level)
    status = {d['detector']: d for d in rep['overview']['detectors']}
    verdicts = sorted(f['verdict'] for f in fs)
    crit = [f for f in fs if f['verdict'] in CRITICISM]

    def res(ok, detail=None):
        return ('pass' if ok else 'fail', detail if detail is not None else verdicts)
    if tok == 'strength_allowed':
        return ('n/a', 'a candidate only')
    if tok == 'no_criticism':
        return res(not crit, [f['key']+':'+f['verdict'] for f in crit])
    if tok == 'criticised':
        # fix round 3 (critic B-1): a work_on, try_this or review of the detector (level 1's reframed advance too)
        return res(bool(crit), [f['key']+':'+f['verdict'] for f in crit])
    if tok == 'moments_review':
        # fix round 3 (critic S-2): a 'moments worth a look' review of the detector
        return res(any(f['verdict'] == 'review' and is_moments_review(f) for f in fs),
                   [(f['verdict'], f['limitations']) for f in fs])
    if tok == 'no_strength':
        return res(not any(f['verdict'] == 'strength' for f in fs))
    if tok == 'try_this_team':
        hits = [f for f in fs if f['verdict'] == 'try_this']
        return res(bool(hits) and not any(_named(f, rep) for f in hits), [(f['key'], _named(f, rep)) for f in hits])
    if tok == 'team_level':
        return res(bool(fs) and not any(_named(f, rep) for f in crit), [(f['key'], _named(f, rep)) for f in crit])
    if tok == 'review_or_work_on':
        return res(any(f['verdict'] in ('review', 'work_on') for f in fs))
    if tok == 'work_on':
        return res(any(f['verdict'] == 'work_on' for f in fs))
    if tok == 'at_most_review':
        return res(not any(f['verdict'] in WORK for f in fs))
    if tok == 'review_one_instance':
        return res(any(f['verdict'] == 'review' and 'one_instance' in f['limitations'] for f in fs),
                   [(f['verdict'], f['limitations']) for f in fs])
    if tok == 'no_drill':
        ids = {f['id'] for f in fs}
        picks = [p['drill'] for p in rep['drills'] if p['finding'] in ids]
        return res(not picks and not any(f['drills'] for f in fs), picks)
    if tok == 'no_player_named':
        named = [(f['key'], _named(f, rep, strict=False)) for f in crit if _named(f, rep, strict=False)]
        return res(not named, named)
    if tok == 'possession_unavailable':
        pq = obs['timeline']['phaseQuality']
        return res(pq['available'] is False, pq.get('reason'))
    if tok == 'not_assessed':
        got = {d: (status[d]['status'], status[d]['reason']) for d in dets}
        return res(all(s == 'off' or (s == 'not_assessed' and r == val) for s, r in got.values()), got)
    if tok == 'assessed':
        got = {d: (status[d]['status'], status[d]['reason']) for d in dets}
        if all(s == 'off' for s, _ in got.values()):
            return ('vacuous', got)
        return res(all(s in ('assessed', 'off') for s, _ in got.values()), got)
    if tok == 'describe_only':
        return res(all(f['verdict'] == 'describe' for f in fs))
    if tok == 'no_team_claims':
        got = {d: status[d]['status'] for d in (D2, D4, D6)}
        return res(all(s != 'assessed' for s in got.values()), got)
    if tok == 'no_video_zone':
        bad = [e['id'] for e in rep['keyMoments']
               for el in ((e.get('videoOverlay') or {}).get('elements') or ())
               if el.get('type') in ('suggestedZone', 'moveArrow', 'channel')]
        return res(not bad, bad)
    if tok == 'diagram_zone':
        ids = {f['id'] for f in fs}
        alts = [a for a in rep['alternatives'] if a['finding'] in ids]
        if not alts:
            return ('vacuous', 'no zone proposed')
        drawn = {el.get('alternative') for e in rep['keyMoments'] for el in ((e.get('diagram') or {}).get('elements')
                                                                             or ()) if el.get('type') == 'suggestedZone'}
        missing = [a['id'] for a in alts if a['id'] not in drawn]
        return res(not missing, missing)
    if tok == 'seek_exact':
        bad = [e['id'] for e in rep['keyMoments'] if abs(e['seek']-round((e['frame']+.5)/e['fps'], 4)) > 1e-4]
        return res(not bad, bad)
    if tok == 'notice':
        codes = [n['code'] for n in rep['overview']['notices']]
        return res(val in codes, codes)
    if tok == 'not_downgraded_by_mismatch':
        bad = [f['key'] for f in fs if f['guardrails']['role'] == 'fail' or 'formation_mismatch' in f['limitations']]
        return res(not bad, bad)
    if tok == 'suppressed_near_formation_change':
        bounds = [p['t0'] for p in obs['timeline']['formationPeriods'] if p['index'] > 0]
        opps = opportunity_index(obs, roles)
        bad = []
        for f in crit:
            for t in f['timestamps']:
                o = opps[t['opportunity']]
                near = any(o['t0']-val <= b <= o['t1']+val for b in bounds)
                if o['intentSignals']['formationChangeNear'] or near:
                    bad.append((f['key'], t['t']))
        if not bounds:
            return ('fail', 'team_shape measured no formation change')
        return res(not bad, bad or bounds)
    if tok == 'no_repetition_wording':
        bad = [(p, m.group(0)) for p, v, _ in report_texts(rep) for m in REPETITION.finditer(v)]
        return res(not bad, bad)
    if tok == 'work_items_need_spells':
        bad = [(f['key'], f['aggregate']['distinctFailingSpells']) for f in rep['needsWork']
               if f['aggregate']['distinctFailingSpells'] < val]
        return res(not bad, bad)
    if tok == 'no_work_items':
        return res(not rep['needsWork'], [f['key'] for f in rep['needsWork']])
    if tok == 'no_position_criticism':
        bad = [f['key']+':'+f['verdict'] for f in findings_of(rep)
               if f['verdict'] in CRITICISM and f['observation'] in (D2, D3, D4, D5)]
        return res(not bad, bad)
    if tok == 'max_try_this':
        n = sum(1 for f in findings_of(rep) if f['verdict'] == 'try_this')
        return res(n <= val, n)
    if tok == 'strength_only_if_demonstrated':
        need = rep['profile']['evidence']['strengthSpells']
        bad = [(f['key'], f['aggregate']['distinctDemonstratedSpells']) for f in rep['strengths']
               if f['aggregate']['distinctDemonstratedSpells'] < max(1, need)]
        return res(not bad, bad)
    if tok == 'same_evidence_across_levels':
        memo.setdefault(('same_evidence', tuple(dets or ())), {})[level] = {
            family(f): tuple(t['evidence'] for t in f['timestamps']) for f in crit}
        return ('n/a', 'judged across levels')
    if tok == 'zone_avoids_stacking':
        return _stacking(rep, fs, obs, roles)
    if tok == 'no_zone':
        ids = {f['id'] for f in crit}
        if not ids:
            return ('vacuous', 'no criticism of the detector')
        drawn = [a['id'] for a in rep['alternatives'] if a['finding'] in ids]
        reasons = sorted({n['reason'] for n in rep['noAlternatives'] if n['finding'] in ids})
        return res(not drawn and (val is None or reasons == [val]), {'drawn': drawn, 'reasons': reasons})
    if tok == 'zone_goal_side_of_threat':
        ids = {f['id'] for f in fs}
        sub = dict(rep, alternatives=[a for a in rep['alternatives'] if a['finding'] in ids and a.get('zone')])
        if not sub['alternatives']:
            return ('vacuous', 'no zone proposed')
        bad = zone_side_problems(sub, obs, roles)
        return res(not bad, bad or f'{len(sub["alternatives"])} zones goal-side of the channel opponent')
    if tok == 'phase_from_coach':
        opps = list(opportunity_index(obs, roles).values())
        coach = (obs['timeline']['phaseQuality'].get('coach') or {})
        return res(any(o['phase']['source'] == 'coach' for o in opps) and bool(coach.get('spans')),
                   {'coachOpportunities': sum(o['phase']['source'] == 'coach' for o in opps), 'spans': coach.get('spans')})
    if tok == 'planted_outcome':
        if not fs:
            return ('fail', 'no finding of the detector')
        d = sum(f['aggregate']['demonstrated'] for f in fs)
        nd = sum(f['aggregate']['notDemonstrated'] for f in fs)
        ok = (d > nd) if val == 'demonstrated' else (nd > d)
        return res(ok, {'demonstrated': d, 'notDemonstrated': nd, 'unclear': sum(f['aggregate']['unclear'] for f in fs)})
    return ('n/a', f'unknown token {tok}')


def _stacking(rep, fs, obs, roles, stack_m=3.0):
    """No suggested-zone point within 3 m of a teammate other than the mover (DESIGN 5.1), checked on a 0.5 m grid
    inside each zone polygon against the moment's observed teammates (metres on the obs scale)."""
    ids = {f['id'] for f in fs}
    alts = [a for a in rep['alternatives'] if a['finding'] in ids and a.get('zone')]
    if not alts:
        return ('vacuous', 'no zone proposed')
    moments = moment_index(obs, roles)
    L, W = obs['timeline']['scale']['lengthM'], obs['timeline']['scale']['widthM']
    bad = []
    for a in alts:
        m = moments.get(a['evidence'])
        poly = a['zone']['polygon']
        mates = [(p['x']*L, p['y']*W) for pid, p in m['scene']['players'].items()
                 if p['team'] == rep['team'] and pid != a['mover']]
        xs, ys = [q[0]*L for q in poly], [q[1]*W for q in poly]
        x = min(xs)
        while x <= max(xs):
            y = min(ys)
            while y <= max(ys):
                if _inside(poly, (x/L, y/W)) and any(math.hypot(x-mx, y-my) < stack_m-.5 for mx, my in mates):
                    bad.append(a['id'])
                    break
                y += .5
            if bad and bad[-1] == a['id']:
                break
            x += .5
    return ('fail' if bad else 'pass', bad or f'{len(alts)} zones clear of team-mates')


def _centroid(poly):
    """The area centroid of a simple polygon [[x, y], ...] (shoelace); the vertex mean when its area is ~0."""
    a = cx = cy = 0.0
    n = len(poly)
    for i in range(n):
        (x0, y0), (x1, y1) = poly[i], poly[(i+1) % n]
        c = x0*y1-x1*y0
        a += c
        cx += (x0+x1)*c
        cy += (y0+y1)*c
    if abs(a) < 1e-12:
        return (sum(q[0] for q in poly)/n, sum(q[1] for q in poly)/n)
    return (cx/(3*a), cy/(3*a))


def _inside(poly, q):
    x, y = q
    c = False
    n = len(poly)
    for i in range(n):
        x0, y0 = poly[i]
        x1, y1 = poly[(i+1) % n]
        if (y0 > y) != (y1 > y) and x < (x1-x0)*(y-y0)/(y1-y0)+x0:
            c = not c
    return c


def finish_expectations(memo):
    """The cross-level checks gathered by check_expectation: [(token, status, detail)]."""
    out = []
    for (tok, dets), by_level in memo.items():
        if tok != 'same_evidence':
            continue
        fams = {}
        for lv, m in by_level.items():
            for fam, evs in m.items():
                fams.setdefault(fam, {})[lv] = evs
        diff = {fam: {lv: len(evs) for lv, evs in v.items()} for fam, v in fams.items() if len(set(v.values())) > 1}
        if not fams:
            out.append(('same_evidence_across_levels', 'fail', 'no criticism of the family at any level'))
        else:
            out.append(('same_evidence_across_levels', 'fail' if diff else 'pass',
                        diff or {fam: sorted(v) for fam, v in fams.items()}))
    return out


def run_scene(scene, variant=None, prepared=None, library=None):
    """Every expectation of one scene variant at levels 1-7 (an expectation with a 'style' is checked with that style
    declared) plus the report invariants of every report. -> dict (JSON-able)."""
    import coaching_synth as cs
    b, st, obs, roles = scene_tiers(scene, variant, prepared=prepared)
    out = {'scene': b.key, 'checks': [], 'levels': {}, 'invariants': []}
    memo = {}
    reports = {}
    for exp in b.expected:
        for lv in exp['levels']:
            style = exp.get('style')
            if (lv, style) not in reports:
                reports[(lv, style)] = interpret(obs, roles, st, 'A', lv, style=style)
            rep = reports[(lv, style)]
            status, detail = check_expectation(exp, lv, rep, obs, roles, memo)
            known = KNOWN_DEFECTS.get((b.key, lv, exp['expect']))
            if known:
                status, detail = ('known' if status == 'fail' else 'fixed'), {'got': detail, 'defect': known}
            out['checks'].append({'level': lv, 'style': style, 'expect': exp['expect'], 'detectors': exp['detectors'],
                                  'value': exp['value'], 'status': status, 'detail': _jsonable(detail)})
    for tok, status, detail in finish_expectations(memo):
        out['checks'].append({'level': 'all', 'style': None, 'expect': tok, 'detectors': [D3], 'value': None,
                              'status': status, 'detail': _jsonable(detail)})
    for lv in LEVELS:
        if (lv, None) not in reports:
            reports[(lv, None)] = interpret(obs, roles, st, 'A', lv)
    for (lv, style), rep in sorted(reports.items(), key=lambda kv: (kv[0][0], kv[0][1] or '')):
        tag = f'L{lv}' + (f'-{style}' if style else '')
        out['levels'][tag] = summary_row(rep)
        for crit, p in report_problems(rep, obs, roles, library):
            out['invariants'].append({'level': tag, 'criterion': crit, 'problem': p})
    out['obsSha'] = sha_json(obs)
    out['rolesSha'] = sha_json(roles)
    return out


def _jsonable(x):
    return json.loads(json.dumps(x, default=str))


def summary_row(rep):
    """One report in a line: the profile, each check's status, each finding's verdict and counts."""
    return {'profile': {k: rep['profile'][k] for k in ('ageGroup', 'level', 'style')},
            'detectors': {d['detector']: d['status'] if d['status'] != 'not_assessed' else f'not_assessed:{d["reason"]}'
                          for d in rep['overview']['detectors']},
            'findings': [{'key': f['key'], 'verdict': f['verdict'], 'opportunities': f['aggregate']['opportunities'],
                          'demonstrated': f['aggregate']['demonstrated'],
                          'notDemonstrated': f['aggregate']['notDemonstrated'],
                          'failingSpells': f['aggregate']['distinctFailingSpells'],
                          'limitations': f['limitations'], 'moments': [t['evidence'] for t in f['timestamps']],
                          'headline': (f.get('text') or {}).get('headline')} for f in findings_of(rep)],
            'notices': [n['code'] for n in rep['overview']['notices']],
            'drills': [{'drill': p['drill'], 'finding': p['finding'], 'optional': p['optional']} for p in rep['drills']],
            'zones': len(rep['alternatives']), 'noZones': len(rep['noAlternatives']),
            'keyMoments': len(rep['keyMoments'])}


def run_scenes(keys=None, prepared=None):
    import coaching_synth as cs
    from soccer.coaching import drills as DR
    lib = DR.load_library()
    names = cs.names()
    if keys:
        want = [k.partition('/') for k in keys]
        names = [(s, v) for s, v in names if any(s == a and (not c or v == c) for a, _, c in want)]
    return [run_scene(s, v, prepared=prepared, library=lib) for s, v in names]


# ---------- the same video at every level (DESIGN 13.2) ----------
# Representative profiles: each level at its typical age, the two level-5 bands, an adult academy, and a declared style
# at levels 6-7 (DESIGN 2.6: style changes interpretation only).
PROFILES = tuple([(lv, None, None) for lv in LEVELS] + [(5, 'U18', None), (6, 'U19', None), (6, 'Adult', 'possession'),
                                                        (7, 'Adult', 'possession'), (7, 'Adult', 'compact'),
                                                        (3, 'Adult', None), (1, 'U6', None)])


def profile_table_problems(profile):
    """The resolved profile against DESIGN 2.3's table (DESIGN_TABLE, written out independently): the enabled
    detectors are the table's, every verdict the table names is there (style unknown), nothing else is added but review
    and describe, and the 4.9 rules (no work_on at L1-L2, no try_this at L4+)."""
    out = []
    lv = profile['level']
    table = DESIGN_TABLE[lv]
    for d in DETECTORS:
        dp = profile['detectors'][d]
        if dp['enabled'] != (d in table):
            out.append(f'L{lv} {d}: enabled {dp["enabled"]}, DESIGN 2.3 {d in table}')
            continue
        if not dp['enabled']:
            continue
        got = set(dp['verdicts'])
        need, allowed = table[d]
        if profile['style'] == 'unknown' and not need <= got:
            out.append(f'L{lv} {d}: verdicts {sorted(got)} lack DESIGN {sorted(need-got)}')
        extra = got-need-allowed-{'review', 'describe'}
        if profile['style'] == 'unknown' and extra:
            out.append(f'L{lv} {d}: verdicts {sorted(extra)} beyond DESIGN 2.3')
        if lv <= 2 and 'work_on' in got:
            out.append(f'L{lv} {d}: work_on at level {lv}')
        if lv >= 4 and 'try_this' in got:
            out.append(f'L{lv} {d}: try_this at level {lv}')
    return out


def levels_check(obs, roles, settings, team, remeasure=None, profiles=PROFILES, remeasured=None, run='synthetic'):
    """DESIGN 13.2 on one team's tiers: -> {'problems': [...], 'reports': {tag: report}, 'pairs': {...}}.
    remeasure: callable(settings) -> (obs, roles) re-measuring under other interpretation settings (invariant 2: the
    bytes must not move); None skips that part. remeasured: the profiles re-measured (default: all of `profiles`; the
    measurement tier reads only format and field, so a few that differ in every interpretation input suffice). run:
    the report's meta run name (fix round 3, FACTS F-12: the run copies' reports carried 'synthetic')."""
    from soccer.coaching import profiles as PR
    problems = []
    obs_sha, roles_sha = sha_json(obs), sha_json(roles)
    reports = {}
    for lv, age, style in profiles:
        tag = f'L{lv}-{age or PR.default_age(lv)}-{style or "unknown"}'
        rep = interpret(obs, roles, settings, team, lv, age, style, run=run)
        reports[tag] = rep
        for p in profile_table_problems(rep['profile']):
            problems.append(('table', f'{tag}: {p}'))
        enabled = {d for d, dp in rep['profile']['detectors'].items() if dp['enabled']}
        for d in rep['overview']['detectors']:
            if (d['status'] == 'off') == (d['detector'] in enabled):
                problems.append(('table', f'{tag}: {d["detector"]} status {d["status"]} but enabled '
                                          f'{d["detector"] in enabled}'))
        for f in findings_of(rep):
            if f['observation'] not in enabled:
                problems.append(('table', f'{tag}: a {f["observation"]} finding while it is off'))
    if (sha_json(obs), sha_json(roles)) != (obs_sha, roles_sha):
        problems.append(('invariant2', 'interpreting changed the obs or roles object'))
    if remeasure is not None:
        for lv, age, style in (profiles if remeasured is None else remeasured):
            o2, r2 = remeasure(settings_at(settings, team, lv, age, style))
            if sha_json(o2) != obs_sha or sha_json(r2) != roles_sha:
                problems.append(('invariant2', f'L{lv}/{age}/{style}: re-measured obs/roles bytes differ'))
    # one family cites the same moments at every level where it is criticised (DESIGN 13.2 bullet 2): profiles with
    # the same detector parameters and evidence rules must cite exactly the same moments; profiles whose parameters
    # differ judge a different failing set and must still share at least one cited moment (else the known defect
    # 'evidence_repicked', see KNOWN_LEVEL_DEFECTS)
    fam_ev = {}
    for tag, rep in reports.items():
        for f in findings_of(rep):
            if f['verdict'] in CRITICISM and f['observation'] != D4:
                rules = sha_json([rep['profile']['detectors'][f['observation']]['params'], rep['profile']['evidence']])
                fam_ev.setdefault(family(f), {})[tag] = (rules, tuple(t['evidence'] for t in f['timestamps']))
    pairs = {'families': {}, 'jaccard': []}
    for fam, by in fam_ev.items():
        evs = {ev for _, ev in by.values()}
        pairs['families'][fam] = len(evs)
        groups = {}
        for tag, (rules, ev) in by.items():
            groups.setdefault(rules, {}).setdefault(ev, []).append(tag)
        for g in groups.values():
            if len(g) > 1:
                problems.append(('evidence', f'{fam}: different moments under the same rules: '
                                             f'{sorted(sorted(v) for v in g.values())}'))
        if len(evs) > 1 and not set.intersection(*[set(ev) for ev in evs]):
            known = [k for k, d in KNOWN_LEVEL_DEFECTS.items() if d['kind'] == 'evidence']
            problems.append((f'known:{known[0]}' if known else 'evidence',
                             f'{fam}: no moment shared across {len(groups)} rule sets {sorted(by)}'))
    # explanations of one family differ between levels (trigram Jaccard < .5)
    by_fam = {}
    for tag, rep in reports.items():
        for f in findings_of(rep):
            by_fam.setdefault((family(f), f['verdict']), []).append((tag, (f.get('text') or {}).get('explanation') or ''))
    for (fam, verdict), items in by_fam.items():
        for i in range(len(items)):
            for j in range(i+1, len(items)):
                la, lb = int(items[i][0][1]), int(items[j][0][1])
                if la == lb:
                    continue            # two profiles of one level may share a register
                jac = jaccard(items[i][1], items[j][1])
                known = [k for k, d in KNOWN_LEVEL_DEFECTS.items() if d['kind'] == 'text' and d['match'](fam, verdict, la, lb)]
                if not known:
                    pairs['jaccard'].append(jac)
                if jac >= TRIGRAM_LIMIT:
                    problems.append((f'known:{known[0]}' if known else 'text',
                                     f'{fam} {verdict}: {items[i][0]} vs {items[j][0]} Jaccard {jac:.2f}'))
    # level-1/2 rules and the per-report invariants
    for tag, rep in reports.items():
        for crit, p in report_problems(rep, obs, roles):
            problems.append((crit, f'{tag}: {p}'))
    return {'problems': problems, 'reports': reports, 'pairs': pairs}


REMEASURED = ((1, 'U6', None), (4, None, None), (5, 'U18', None), (7, 'Adult', 'possession'))


def scene_remeasure(scene, variant=None, team='A', prepared=None):
    """callable(settings) -> (obs, roles): the scene measured again through store.measure and store.build_role_tier
    under other coaching settings (invariant 2: the bytes must not move with the interpretation settings)."""
    import coaching_synth as cs
    from soccer import team_shape
    from soccer.coaching import store as ST
    if prepared is not None:
        b, _, cd, poss = prepared(scene, variant)
    else:
        b, cd, poss = cs.build(scene, variant), None, None
    sha = ST.sha256_bytes(json.dumps(b.result, separators=(',', ':'), allow_nan=False).encode('utf-8'))

    def again(settings):
        obs = ST.measure(b.result, result_sha=sha, shape_settings=team_shape.clean_settings({}), settings=settings,
                         spans=b.coach_spans, team=team, series=cd, possession=poss)
        return obs, ST.build_role_tier(obs, settings, team)
    return again


def levels_scenes(keys=None, prepared=None, remeasured=REMEASURED):
    """DESIGN 13.2 on every scene variant (or `keys`): {scene/variant: levels_check result}, each scene re-measured
    under `remeasured`."""
    import coaching_synth as cs
    names = cs.names()
    if keys:
        want = [k.partition('/') for k in keys]
        names = [(s, v) for s, v in names if any(s == a and (not c or v == c) for a, _, c in want)]
    out = {}
    for s, v in names:
        b, st, obs, roles = scene_tiers(s, v, prepared=prepared)
        out[b.key] = levels_check(obs, roles, st, 'A', remeasure=scene_remeasure(s, v, prepared=prepared),
                                  remeasured=remeasured)
    return out


def levels_digest(res):
    """A levels_check result without the reports (JSON-able): problems, the families' distinct evidence tuples,
    the trigram pairs, and one summary row per profile."""
    jac = res['pairs']['jaccard']
    return {'problems': [{'kind': k, 'problem': p} for k, p in res['problems']],
            'families': res['pairs']['families'], 'pairs': len(jac),
            'maxJaccard': _r(max(jac)) if jac else None, 'claims': claims_by_level(res['reports']),
            'profiles': {tag: summary_row(rep) for tag, rep in res['reports'].items()}}


def claims_by_level(reports, detector=D3):
    """{level: max relation + opponent claims of one detector's criticism} (criterion 5)."""
    out = {}
    for rep in reports.values():
        lv = rep['profile']['level']
        for f in findings_of(rep):
            if f['observation'] == detector and f['verdict'] in CRITICISM:
                n = sum(1 for c in f['claims'] if c['type'] in ('relation', 'opponent_state'))
                out[lv] = max(out.get(lv, 0), n)
    return out


# ---------- real runs ----------
def spans_from_possession_labels(path, t0=None, t1=None):
    """A SYNTHETIC coach-marked phase example from blind possession hand labels (tools/annotations/possession/*.json,
    0.5 s points): runs of one label (A, B, dead) of consecutive points, each point standing for +-0.25 s, runs under 1 s
    dropped, 'unclear' marking nothing (CONTEXT's wave-2 rule). It stands in for a coach's marks; nothing is tuned on
    it, and every output made with it says so."""
    pts = sorted(json.loads(Path(path).read_text(encoding='utf-8'))['points'], key=lambda p: p['t'])
    runs = []
    for p in pts:
        lab = p['team'] if p['team'] in ('A', 'B', 'dead') else None
        if runs and runs[-1]['state'] == lab and abs(p['t']-runs[-1]['last']-.5) < .05:
            runs[-1]['last'] = p['t']
        else:
            runs.append({'state': lab, 'first': p['t'], 'last': p['t']})
    spans = [{'t0': round(r['first']-.25, 2), 't1': round(r['last']+.25, 2), 'state': r['state']}
             for r in runs if r['state'] is not None and r['last']-r['first']+.5 >= 1.0]
    if t0 is not None:
        spans = [s for s in spans if s['t1'] > t0 and s['t0'] < t1]
    return {'version': 1, 'kind': 'coach-possession-spans', 'source': 'coach', 'spans': spans}


def run_settings(level=6, age=None, style=None, players=11):
    """Coaching settings for a real run copy: both teams at the footage's own format (11v11 with keepers: the broadcast,
    amateur and Veo clips are all 11-a-side), the given level and age, no roles, no intended formation."""
    from soccer.coaching import profiles as PR
    block = {'squad': None, 'ageGroup': age or PR.default_age(level), 'playingLevel': int(level),
             'format': {'players': players, 'goalkeeper': True}, 'field': None, 'federation': 'auto',
             'formation': {'intended': None, 'periods': []}, 'style': style or 'unknown', 'roles': {}, 'runLinks': []}
    return {'version': 1, 'teams': {'A': dict(block), 'B': dict(block)}, 'updated': None}


def measure_run(path, spans=None, settings=None, loaded=None):
    """(result, sha, obs {A, B}, roles {A, B}, spans as stored) of one run copy through the server's tiers (both teams
    in one pass). loaded: (result, sha) already read from path (a re-measurement skips the 1 s parse of BC180)."""
    from soccer import team_shape
    from soccer.coaching import situations as SI, store as ST
    if loaded is None:
        data = Path(path).read_bytes()
        result, sha = json.loads(data.decode('utf-8')), ST.sha256_bytes(data)
    else:
        result, sha = loaded
    st = settings or run_settings()
    if spans is not None:
        seg = ST.segment_of(result.get('frames'))
        kept, _ = SI.clean_coach_spans(spans, seg['t0'], seg['t1'])
        spans = dict(ST.empty_spans(), spans=kept) if kept else None
    obs = ST.measure(result, result_sha=sha, shape_settings=team_shape.clean_settings({}), settings=st, spans=spans)
    roles = {t: ST.build_role_tier(obs[t], st, t) for t in ('A', 'B')}
    return result, sha, obs, roles, spans


def run_reports(runs, out, spans=None, profiles=PROFILES, remeasure_levels=True, log=print):
    """Reports of each run copy at every profile for both teams (and, for runs in `spans`, again with the synthetic
    coach-marked phase), written under out/<RUN>[+coach]/<team>/; -> the summary dict (also out/summary.json).
    remeasure_levels: measure the run again under every profile's settings (both teams per pass) and compare the obs and
    roles bytes (invariant 2)."""
    out = _safe_dir(out)
    summary = {'generated': time.strftime('%Y-%m-%d %H:%M:%S'), 'runs': {}, 'notes': [
        'Runs marked +coach use a SYNTHETIC coach-marked phase made from the blind possession hand labels '
        '(tools/annotations/possession/<run>.json); no coach marked them.']}
    jobs = [(name, path, None) for name, path in runs.items()]
    jobs += [(f'{name}+coach', runs[name], lab) for name, lab in (spans or {}).items() if name in runs]
    for name, path, lab in jobs:
        t = time.perf_counter()
        sp = None
        if lab is not None:
            sp = spans_from_possession_labels(lab)
        result, sha, obs, roles, used = measure_run(path, spans=sp)
        took = time.perf_counter()-t
        if lab is not None:
            write_json(out/name/'coach_spans_SYNTHETIC.json', dict(used or {}, note=(
                'SYNTHETIC coach-marked phase: made by tools/coaching_eval.py from the blind possession hand labels '
                f'{Path(lab).name} (0.5 s points, runs of one label of at least 1 s). No coach marked these spans; '
                'every report in this folder rests on them.')))
            write_text(out/name/'README_SYNTHETIC.txt',
                       f'The reports under {name}/ use a SYNTHETIC coach-marked phase (coach_spans_SYNTHETIC.json),\n'
                       f'made from the blind possession hand labels {Path(lab).name}. They show what the coach-marking '
                       f'tool would unlock on this clip; they are not a coach\'s marks and nothing was tuned on them.\n')
        entry = {'result': str(path), 'resultSha': sha, 'coachSpans': (len(used['spans']) if used else 0),
                 'coachSpansSynthetic': lab is not None, 'measureSeconds': round(took, 2), 'teams': {},
                 'remeasured': 0}
        moved = {'A': [], 'B': []}
        if remeasure_levels:
            for lv_, age, style in profiles:
                st = run_settings(lv_, age, style)
                _, _, o2, r2, _ = measure_run(path, spans=sp, settings=st, loaded=(result, sha))
                entry['remeasured'] += 1
                for team in ('A', 'B'):
                    if (sha_json(o2[team]), sha_json(r2[team])) != (sha_json(obs[team]), sha_json(roles[team])):
                        moved[team].append(('invariant2', f'L{lv_}/{age}/{style}: re-measured obs/roles bytes differ'))
        for team in ('A', 'B'):
            o, r = obs[team], roles[team]
            write_json(out/name/team/'obs.json', o, indent=None)
            write_json(out/name/team/'roles.json', r, indent=None)
            # the report's run is the run copy's name in the schema's alphabet, and says when its coach-marked phase
            # is the eval's SYNTHETIC one (FACTS F-12 / LOW-9: 'AM+coach' -> 'AM-coach-synthetic-spans')
            lv = levels_check(o, r, run_settings(), team, remeasure=None, profiles=profiles,
                              run=re.sub(r'[^A-Za-z0-9_.-]', '-', name)+('-synthetic-spans' if lab is not None else ''))
            lv['problems'] += moved[team]
            rows = {}
            for tag, rep in lv['reports'].items():
                write_json(out/name/team/f'{tag}.json', rep)
                rows[tag] = summary_row(rep)
            entry['teams'][team] = {
                'obsSha': sha_json(o), 'rolesSha': sha_json(r), 'opportunities': len(o['opportunities']),
                'roleOpportunities': len(r['opportunities']), 'moments': len(o['moments'])+len(r['moments']),
                'phase': {k: o['timeline']['phaseSeconds'][k] for k in ('own', 'opp', 'dead', 'unknown')},
                'possession': {k: o['timeline']['phaseQuality'][k] for k in ('available', 'level', 'coverage')},
                'reports': rows, 'problems': [{'kind': k, 'problem': p} for k, p in lv['problems']],
                'claims': claims_by_level(lv['reports'])}
            real = [k for k, _ in lv['problems'] if not k.startswith('known:')]
            log(f'{name} {team}: {len(rows)} reports, {len(real)} problems, {len(lv["problems"])-len(real)} known open '
                f'defects, {sum(len(x["findings"]) for x in rows.values())} findings')
        entry['seconds'] = round(time.perf_counter()-t, 1)
        summary['runs'][name] = entry
    write_json(out/'summary.json', summary)
    write_text(out/'SUMMARY.md', runs_markdown(out))
    return summary


VERDICT_CODE = (('strength', 'S'), ('work_on', 'W'), ('try_this', 'T'), ('review', 'R'), ('describe', 'D'))
DET_CODE = {D1: 'D1', D2: 'D2', D3: 'D3', D4: 'D4', D5: 'D5', D6: 'D6'}
TABLE_TAGS = ('L1-U8-unknown', 'L2-U10-unknown', 'L3-U12-unknown', 'L4-U14-unknown', 'L5-U16-unknown',
              'L6-Adult-unknown', 'L7-Adult-unknown')


def report_stats(rep):
    """The numbers of one report the summary table shows: findings, verdict mix, zones drawn / declined (with the
    reasons), drills (for a work item / optional), coach-facing words, and every check not assessed with its reason."""
    fs = findings_of(rep)
    mix = {c: sum(f['verdict'] == v for f in fs) for v, c in VERDICT_CODE}
    words = sum(len(re.findall(r"[A-Za-z0-9']+", t)) for _, t, _ in report_texts(rep))
    expl = sorted(len(re.findall(r"[A-Za-z0-9']+", (f.get('text') or {}).get('explanation') or '')) for f in fs)
    declined = {}
    for n in rep['noAlternatives']:
        declined[n['reason']] = declined.get(n['reason'], 0)+1
    return {'findings': len(fs), 'mix': mix, 'zones': sum(1 for a in rep['alternatives'] if a.get('zone')),
            'points': sum(1 for a in rep['alternatives'] if not a.get('zone')), 'declined': declined,
            'drillsWork': sum(1 for p in rep['drills'] if p['finding']),
            'drillsOptional': sum(1 for p in rep['drills'] if not p['finding']),
            'words': words, 'explanationWords': expl[len(expl)//2] if expl else 0,
            # assess.py gives no reason when the phase never occurred in the range (the page then says "not enough
            # data"): shown here as the phase time it lacked
            'notAssessed': {DET_CODE[d['detector']]: d['reason'] or ('no phase time' if not d['phaseSeconds'] else
                                                                    'no reason given')
                            for d in rep['overview']['detectors'] if d['status'] == 'not_assessed'},
            'off': [DET_CODE[d['detector']] for d in rep['overview']['detectors'] if d['status'] == 'off'],
            'notices': [n['code'] for n in rep['overview']['notices']],
            'unavailable': [u['category'] for u in rep['overview']['unavailable']]}


def _mix_text(mix):
    return ' '.join(f'{c}{n}' for c, n in mix.items() if n) or '-'


def runs_markdown(reports_dir, scenes=None, levels=None, statsbomb=None):
    """The eval summary table (markdown) of a `runs` output folder: per run variant and team, one row per level at its
    typical age (DESIGN 2.2) with findings, verdict mix, zones drawn / declined, drills, words and the checks not
    assessed (reason), then the totals per level, then what the other profiles (age bands, declared styles) changed.
    scenes / levels / statsbomb: the JSON outputs of those commands, summarised when given."""
    reports_dir = Path(reports_dir)
    summary = json.loads((reports_dir/'summary.json').read_text(encoding='ascii'))
    L = ['# Coaching v2 evaluation: real run copies at all seven levels', '',
         f'Generated {summary["generated"]} by `tools/coaching_eval.py runs` (worktree code, uncommitted, from commit '
         '1262590). Every number below was produced by the real engine on the scratch run copies; nothing is '
         'validated by a coach (the blind label sheets are not filled yet).', '']
    for n in summary['notes']:
        L.append(f'- {n}')
    L += ['- Verdicts: S strength, W work on, T try this, R review, D describe. Zones: suggested areas drawn '
          '(zone / point) and declined (reason: count). Drills: for a work item / optional next-objective drill. '
          'Words: every coach-facing text of the report (finding texts, overview, notices, captions, drills); '
          'expl: median words of one explanation.',
          '- Not assessed: checks enabled at the level that the clip could not support, with the engine\'s reason '
          '(off = disabled by the level, not listed).', '']
    totals = {}
    for name, e in summary['runs'].items():
        L.append(f'## {name}')
        L.append('')
        for team, te in e['teams'].items():
            ph = te['phase']
            pos = te['possession']
            L.append(f'### {name} team {team}')
            L.append('')
            L.append(f'Possession: available {pos["available"]}, quality {pos["level"]}, coverage {pos["coverage"]}; '
                     f'phase seconds own {ph["own"]}, opp {ph["opp"]}, dead {ph["dead"]}, unknown {ph["unknown"]}; '
                     f'{te["opportunities"]} + {te["roleOpportunities"]} (role tier) opportunities, '
                     f'{te["moments"]} evidence moments' +
                     (f'; coach spans: {e["coachSpans"]} (SYNTHETIC: from the blind possession hand labels)'
                      if e.get('coachSpansSynthetic') else '') + '.')
            real = [p for p in te['problems'] if not p['kind'].startswith('known:')]
            known = len(te['problems'])-len(real)
            L.append(f'Checks: obs/roles bytes identical under {e["remeasured"]} re-measured profiles: '
                     f'{"yes" if not any(p["kind"] == "invariant2" for p in te["problems"]) else "NO"}; '
                     f'problems {len(real)}, known open defects {known}.')
            L.append('')
            L.append('| Profile | Findings | Verdicts | Zones drawn / declined | Drills (work / optional) | Words (expl) '
                     '| Not assessed (reason) |')
            L.append('|---|---|---|---|---|---|---|')
            for tag in TABLE_TAGS:
                path = reports_dir/name/team/f'{tag}.json'
                if not path.is_file():
                    continue
                st = report_stats(json.loads(path.read_text(encoding='ascii')))
                lv = tag.split('-')[0]
                tot = totals.setdefault(lv, {'reports': 0, 'findings': 0, 'mix': {c: 0 for _, c in VERDICT_CODE},
                                             'zones': 0, 'declined': 0, 'drillsWork': 0, 'drillsOptional': 0,
                                             'words': []})
                tot['reports'] += 1
                tot['findings'] += st['findings']
                for c, v in st['mix'].items():
                    tot['mix'][c] += v
                tot['zones'] += st['zones']+st['points']
                tot['declined'] += sum(st['declined'].values())
                tot['drillsWork'] += st['drillsWork']
                tot['drillsOptional'] += st['drillsOptional']
                tot['words'].append(st['words'])
                dec = ', '.join(f'{k} {v}' for k, v in sorted(st['declined'].items())) or '-'
                na = '; '.join(f'{k} {v}' for k, v in sorted(st['notAssessed'].items())) or '-'
                L.append(f'| {tag} | {st["findings"]} | {_mix_text(st["mix"])} | {st["zones"]}+{st["points"]} / '
                         f'{dec} | {st["drillsWork"]} / {st["drillsOptional"]} | {st["words"]} '
                         f'({st["explanationWords"]}) | {na} |')
            # the extra profiles: what an age band or a declared style changed against the level's typical row
            diffs = []
            for tag, row in te['reports'].items():
                if tag in TABLE_TAGS:
                    continue
                lv = tag.split('-')[0]
                base = next((t for t in TABLE_TAGS if t.startswith(lv+'-')), None)
                if base is None or base not in te['reports']:
                    continue
                a = sorted((f['key'].split('|')[0], f['verdict']) for f in te['reports'][base]['findings'])
                b = sorted((f['key'].split('|')[0], f['verdict']) for f in row['findings'])
                if a != b:
                    diffs.append(f'{tag}: {", ".join(f"{k} {v}" for k, v in b) or "no findings"} (vs {base})')
            if diffs:
                L.append('')
                L.append('Other profiles that changed the verdicts: ' + '; '.join(diffs) + '.')
            L.append('')
    L.append('## Totals per level (all run variants and teams above, typical age)')
    L.append('')
    L.append('| Level | Reports | Findings | Verdicts | Zones drawn | Zones declined | Drills (work / optional) '
             '| Words median |')
    L.append('|---|---|---|---|---|---|---|---|')
    for lv in sorted(totals):
        t = totals[lv]
        w = sorted(t['words'])
        L.append(f'| {lv} | {t["reports"]} | {t["findings"]} | {_mix_text(t["mix"])} | {t["zones"]} | '
                 f'{t["declined"]} | {t["drillsWork"]} / {t["drillsOptional"]} | {w[len(w)//2] if w else 0} |')
    L.append('')
    any_rep = next(reports_dir.glob('*/A/L4-U14-unknown.json'), None)
    if any_rep is not None:
        un = json.loads(any_rep.read_text(encoding='ascii'))['overview']['unavailable']
        L.append('Categories no report assesses (every run, every level; MATCH OVERVIEW "Not measured by PitchIQ"): ' +
                 '; '.join(f'{u["category"]} ({u["reason"]})' for u in un))
        L.append('')
    if scenes:
        tot = scenes['total']
        L += ['## Synthetic counterexamples (DESIGN 13.1, `coaching_eval.py scenes`)', '',
              f'{len(scenes["scenes"])} scene variants at levels 1-7 through the real engine: {tot["pass"]} checks '
              f'pass, {tot["fail"]} fail, {tot["known"]} known open defect, {tot["n/a"]} not applicable (a candidate '
              f'or a cross-level token judged once); report invariants violated on '
              f'{sum(len(r["invariants"]) for r in scenes["scenes"])} reports.', '']
    if levels:
        prob = sum(1 for d in levels['scenes'].values() for p in d['problems'] if not p['kind'].startswith('known:'))
        known = {}
        for d in levels['scenes'].values():
            for p in d['problems']:
                if p['kind'].startswith('known:'):
                    known[p['kind'][6:]] = known.get(p['kind'][6:], 0)+1
        mx = max((d['maxJaccard'] or 0) for d in levels['scenes'].values())
        L += ['## Same video, different levels on the scenes (DESIGN 13.2, `coaching_eval.py levels`)', '',
              f'{len(levels["scenes"])} scene variants x {len(PROFILES)} profiles ({len(REMEASURED)} re-measured): '
              f'{prob} problems; known open defects: ' +
              (', '.join(f'{k} {v}' for k, v in sorted(known.items())) or 'none') +
              f'. The most similar explanations of one family at two levels (outside the known defects) scored a '
              f'word-trigram Jaccard of {mx}.', '']
        for k, d in KNOWN_LEVEL_DEFECTS.items():
            L += [f'- {k} ({d["owner"]}): {d["why"]}']
        L.append('')
    if statsbomb:
        a = statsbomb['subsets']['all']
        d3 = statsbomb['subsets']['defendingThird']
        t5, t5d = a['thresholds']['5'], d3['thresholds']['5']
        L += ['## Pressure distance on StatsBomb open data', '',
              f'Data: StatsBomb open data, 360 freeze frames ({statsbomb["source"]}); matches '
              f'{", ".join(statsbomb["matches"])}. {a["events"]} carrier events (pass, carry, dribble, shot) with a '
              f'visible outfield opponent. Under pressure: nearest opponent median {a["pressured"]["p50"]} m '
              f'(p90 {a["pressured"]["p90"]}); not under pressure: {a["notPressured"]["p50"]} m. PitchIQ\'s 5 m: '
              f'{t5["pressuredWithin"]} of the pressured and {t5["notPressuredWithin"]} of the not-pressured events '
              f'have an opponent within 5 m (precision {t5["precision"]}, recall {t5["recall"]}); "nobody within '
              f'5 m" agrees with not-under-pressure on {t5["noPressureAgreement"]}. Carrier in the defending third '
              f'(where D1 criticises an unpressed carrier): {t5d["pressuredWithin"]} / {t5d["notPressuredWithin"]} '
              f'within 5 m, agreement {t5d["noPressureAgreement"]}.', '']
    return '\n'.join(L)+'\n'


# ---------- StatsBomb: an independent check of the pressure distance ----------
# StatsBomb open data (https://github.com/statsbomb/open-data; data by StatsBomb, used under its open-data terms and
# credited wherever a number from it is shown): events with an annotator's under_pressure flag, and 360 freeze frames
# (every visible player's position at the event, the camera's visible area). PitchIQ calls the carrier pressured when
# the first defender is within PRESSURE_M (DESIGN Appendix A, from the possession labeller's notes); this measures the
# nearest outfield opponent's distance at carrier events flagged under pressure and not, on data PitchIQ never saw.
SB_PITCH = (120.0, 80.0)            # StatsBomb's pitch units (x along, y across)
SB_METRES = (105.0, 68.0)           # metres they are mapped to (anisotropic: StatsBomb draws every pitch 120 x 80)
SB_CARRIER = ('Pass', 'Carry', 'Dribble', 'Shot')     # events whose actor has the ball
SB_THRESHOLDS = (2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0)
PRESSURE_M = 5.0                    # PitchIQ's rule (profiles.json D1 params; DESIGN Appendix A)
GO_TO_BALL_M = 8.0                  # level 1-2 "goes to the ball" distance (DESIGN Appendix A)


def _num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _sb_xy(loc):
    """StatsBomb [x, y] -> metres, or None for anything malformed (the files are untrusted: only numbers pass)."""
    if not isinstance(loc, list) or len(loc) < 2 or not (_num(loc[0]) and _num(loc[1])):
        return None
    return (float(loc[0])*SB_METRES[0]/SB_PITCH[0], float(loc[1])*SB_METRES[1]/SB_PITCH[1])


def _poly_edge_distance(poly, q):
    best = math.inf
    for i in range(len(poly)):
        (ax, ay), (bx, by) = poly[i], poly[(i+1) % len(poly)]
        dx, dy = bx-ax, by-ay
        L2 = dx*dx+dy*dy
        u = 0.0 if L2 == 0 else max(0.0, min(1.0, ((q[0]-ax)*dx+(q[1]-ay)*dy)/L2))
        best = min(best, math.hypot(q[0]-(ax+u*dx), q[1]-(ay+u*dy)))
    return best


def _quantiles(xs, qs=(.1, .25, .5, .75, .9)):
    xs = sorted(xs)
    if not xs:
        return {}
    out = {}
    for q in qs:
        k = q*(len(xs)-1)
        lo, hi = int(math.floor(k)), int(math.ceil(k))
        out[f'p{int(q*100)}'] = round(xs[lo]+(xs[hi]-xs[lo])*(k-lo), 2)
    return out


def statsbomb_rows(data_dir):
    """One row per carrier event with a 360 frame: {match, type, pressured, d (nearest outfield opponent, m, or None),
    certain (the disc of radius d around the carrier lies inside the camera's visible area: nobody unseen can be
    nearer, PitchIQ's own first-defender rule), opponents (visible), x (metres from the carrier's own goal line:
    StatsBomb orients every event as the actor's team attacking left to right, and the 360 actor sits on the event's
    location, median difference 1.5e-6 units on match 3795107)}. Reads events_<id>.json and three-sixty_<id>.json
    pairs; never executes anything from them."""
    data_dir = Path(data_dir)
    rows = []
    for ev_path in sorted(data_dir.glob('events_*.json')):
        mid = ev_path.stem.split('_', 1)[1]
        if not re.fullmatch(r'\d{1,12}', mid):
            continue
        ff_path = data_dir/f'three-sixty_{mid}.json'
        if not ff_path.is_file():
            continue
        events = json.loads(ev_path.read_text(encoding='utf-8'))
        frames = {}
        for f in json.loads(ff_path.read_text(encoding='utf-8')):
            if isinstance(f, dict) and isinstance(f.get('event_uuid'), str):
                frames[f['event_uuid']] = f
        for e in events if isinstance(events, list) else ():
            if not isinstance(e, dict) or (e.get('type') or {}).get('name') not in SB_CARRIER:
                continue
            f = frames.get(e.get('id'))
            if f is None or not isinstance(f.get('freeze_frame'), list):
                continue
            actor, opps = None, []
            for p in f['freeze_frame']:
                if not isinstance(p, dict):
                    continue
                xy = _sb_xy(p.get('location'))
                if xy is None:
                    continue
                if p.get('actor') is True:
                    actor = xy
                elif p.get('teammate') is False and p.get('keeper') is not True:
                    opps.append(xy)
            actor = actor or _sb_xy(e.get('location'))
            if actor is None:
                continue
            va = f.get('visible_area')
            poly = [_sb_xy(va[i:i+2]) for i in range(0, len(va)-1, 2)] if isinstance(va, list) else []
            poly = [q for q in poly if q is not None]
            d = min((math.hypot(o[0]-actor[0], o[1]-actor[1]) for o in opps), default=None)
            certain = False
            if d is not None and len(poly) >= 3 and _inside_xy(poly, actor):
                certain = _poly_edge_distance(poly, actor) >= d
            rows.append({'match': mid, 'type': e['type']['name'], 'pressured': e.get('under_pressure') is True,
                         'd': None if d is None else round(d, 2), 'certain': certain, 'opponents': len(opps),
                         'x': round(actor[0], 2)})
    return rows


def _inside_xy(poly, q):
    x, y = q
    c = False
    for i in range(len(poly)):
        (x0, y0), (x1, y1) = poly[i], poly[(i+1) % len(poly)]
        if (y0 > y) != (y1 > y) and x < (x1-x0)*(y-y0)/(y1-y0)+x0:
            c = not c
    return c


def statsbomb_check(data_dir):
    """The nearest-defender distance at carrier events under pressure vs not (all with a visible opponent, the
    'certain' subset, and the carrier in the defending team's half / third), and how well "d <= t" separates them for each t in SB_THRESHOLDS (precision, recall, accuracy,
    Youden's J, and the agreement of "nobody within t" with StatsBomb's not-under-pressure). -> dict (JSON-able).
    Measured Oct 22 on the four Euro 2020 matches with 360 data in the scratch download (8576 carrier events with a
    visible opponent): under pressure, nearest opponent median 2.12 m (84% within 5 m); not under pressure 5.69 m (44%
    within 5 m); 5 m precision .32 against a base rate of .20, and "nobody within 5 m" is StatsBomb's not-under-pressure
    on .94 of those events; carrier in the defending third: 89% / 64% within 5 m, agreement .91."""
    rows = statsbomb_rows(data_dir)
    out = {'source': 'StatsBomb open data (github.com/statsbomb/open-data), 360 freeze frames; data by StatsBomb',
           'matches': sorted({r['match'] for r in rows}), 'carrierEvents': len(rows),
           'noVisibleOpponent': sum(r['d'] is None for r in rows), 'pitchMetres': list(SB_METRES), 'subsets': {}}
    L = SB_METRES[0]
    subsets = (('all', lambda r: r['d'] is not None), ('certain', lambda r: r['d'] is not None and r['certain']),
               # where PitchIQ's D1 judges an unpressed carrier as failing (contextual.sample_outcomes: the carrier in
               # the defending team's own half at levels 1-2, own third from level 3)
               ('defendingHalf', lambda r: r['d'] is not None and r['x'] >= L/2.0),
               ('defendingThird', lambda r: r['d'] is not None and r['x'] >= 2.0*L/3.0))
    for name, keep in subsets:
        sel = [r for r in rows if keep(r)]
        pr = [r['d'] for r in sel if r['pressured']]
        un = [r['d'] for r in sel if not r['pressured']]
        sub = {'events': len(sel), 'pressured': {'n': len(pr), **_quantiles(pr)},
               'notPressured': {'n': len(un), **_quantiles(un)}, 'thresholds': {}}
        for t in SB_THRESHOLDS:
            tp = sum(d <= t for d in pr)
            fp = sum(d <= t for d in un)
            fn, tn = len(pr)-tp, len(un)-fp
            sub['thresholds'][f'{t:g}'] = {
                'pressuredWithin': _r(tp/len(pr)) if pr else None, 'notPressuredWithin': _r(fp/len(un)) if un else None,
                'precision': _r(tp/(tp+fp)) if tp+fp else None, 'recall': _r(tp/len(pr)) if pr else None,
                'accuracy': _r((tp+tn)/len(sel)) if sel else None,
                'youdenJ': _r(tp/len(pr)-fp/len(un)) if pr and un else None,
                # PitchIQ's criticism is the other side: "nobody within t" (a no_pressure failure in the zone); the
                # share of those events StatsBomb also calls not under pressure
                'noPressureAgreement': _r(tn/(tn+fn)) if tn+fn else None}
        by_type = {}
        for typ in SB_CARRIER:
            tsel = [r for r in sel if r['type'] == typ]
            tp_ = [r['d'] for r in tsel if r['pressured']]
            tu_ = [r['d'] for r in tsel if not r['pressured']]
            if tsel:
                by_type[typ] = {'pressured': {'n': len(tp_), **_quantiles(tp_, (.5,))},
                                'notPressured': {'n': len(tu_), **_quantiles(tu_, (.5,))},
                                'pressuredWithin5': _r(sum(d <= PRESSURE_M for d in tp_)/len(tp_)) if tp_ else None,
                                'notPressuredWithin5': _r(sum(d <= PRESSURE_M for d in tu_)/len(tu_)) if tu_ else None}
        sub['byType'] = by_type
        best = max(sub['thresholds'].items(), key=lambda kv: kv[1]['youdenJ'] or -1)
        sub['bestThreshold'] = {'metres': float(best[0]), **best[1]}
        out['subsets'][name] = sub
    return out


# ---------- blind label sheets (DESIGN 13.3) ----------
# Four sets a coach fills in WITHOUT seeing any PitchIQ coaching output: the video at a timestamp (or a clean top-down
# diagram for ZONE), a team or a player to look at, short questions. What PitchIQ measured is written to key/ files the
# sheets never show; the scorer joins the two after the labels exist. The tracking labels (A-07, ...) and the players'
# boxes are tracking output, not coaching output: they are shown so the coach knows whom a question is about.
SHEET_VERSION = 1
SHEET_SEED = 20261022               # every sample and the item order (re-running gives the same sheets)
RANDOM_SHARE = .3                   # DESIGN 13.3: ~200 moments per footage type, stratified, plus 30% random times
MOMENT_TARGET = 200
MOMENT_SPACING_S = 1.5              # two items of one team and video are at least this far apart
RANDOM_EVERY_S = 3.0                # small pools (amateur, Veo): at most one random time per team every 3 s
ROLE_MIN_S = 15.0                   # a player is on the ROLE sheet when observed at least this long
ADVANCE_M = 15.0                    # advanced: at least 15 m (x L/105) beyond the back line (DESIGN 4.3), role-free here
ADVANCE_MIN_S = 3.0                 # spell length (DESIGN 4.3)
BACK_PLAYERS = 5                    # a candidate wide defender is among the team's 5 deepest outfield players (median)
WINDOW = (3.0, 2.0)                 # the clip a row plays: from 3 s before the time to 2 s after (the report's window)
BANDS = {'youth': (2, 'U10'), 'intermediate': (4, 'U14'), 'senior': (6, 'Adult')}   # DESIGN 13.3 three bands
ROLE_OPTIONS = ('GK', 'LB', 'LCB', 'CB', 'RCB', 'RB', 'LWB', 'RWB', 'CDM', 'LCM', 'CM', 'RCM', 'LM', 'RM', 'CAM', 'LW',
                'RW', 'ST', 'unsure')
YN = ('yes', 'no', 'unclear')
YNA = ('yes', 'no', 'not applicable', 'unclear')
MOMENT_QUESTIONS = (
    ('ball', 'Who had the ball? (this team / the other team / nobody: dead ball or loose / unclear)',
     ('this team', 'other team', 'nobody', 'unclear')),
    ('pressure', 'If the other team had it: was the ball carrier under pressure from this team?', YNA),
    ('cover', 'If someone was pressing: did a team-mate cover behind the presser?', YNA),
    ('farside', 'If the ball was wide and the other team had it: was the far side protected (a defender goal-side in '
                'the far half)?', YNA),
    ('line', 'If the other team had it: was the back line organised (level, close enough to midfield, no big gap on '
             'the ball side)?', YNA),
    ('behind_youth', 'If this team had it: enough players behind the ball to protect the goal, for a youth team '
                     '(about U9-U12)?', YNA),
    ('behind_intermediate', '... for an intermediate team (about U13-U18)?', YNA),
    ('behind_senior', '... for a senior team (adult, academy)?', YNA),
    ('spread', 'If this team had it: was the team spread out (wide and deep)?', YNA),
    ('raise_youth', 'Would you raise a positioning problem of this team here with a youth team?', YN),
    ('raise_intermediate', '... with an intermediate team?', YN),
    ('raise_senior', '... with a senior team?', YN),
)
ROLE_QUESTIONS = (
    ('role', 'Main position in this clip', ROLE_OPTIONS),
    ('sure', 'How sure are you?', ('sure', 'fairly sure', 'unsure')),
    ('changed', 'Did he change position during the clip?', YN),
    ('role2', 'If yes: the other position', ROLE_OPTIONS+('none',)),
    ('change_at', 'If yes: from about which video time (m:ss)?', None),
)
ADVANCE_QUESTIONS = (
    ('wide_defender', 'Is the marked player a wide defender (full-back or wing-back) here?', YN),
    ('advanced', 'Did he go forward, well beyond the centre-backs?', YN),
    ('supported', 'Was the space behind him covered (a centre-back moved across, a midfielder dropped, or a team-mate '
                  'took the channel)?', ('covered', 'not covered', 'unclear')),
    ('intended', 'Did the run look like part of the team\'s plan?', YN),
    ('opponent', 'Was an opponent in the space he left (in this team\'s half)?', YN),
    ('raise', 'Would you raise it with the team as a problem?', ('yes', 'no', 'depends on the level')),
)
ZONE_PROBLEM = {
    ('pressure_cover', 'no_cover'): 'The ringed player of the blue team presses the ball carrier, but no blue '
                                    'team-mate covers behind him.',
    ('pressure_cover', 'no_pressure'): 'Nobody of the blue team puts pressure on the ball carrier (black ring) near '
                                       'the blue team\'s goal.',
    ('pressure_cover', None): 'The blue team is defending the ball carrier (black ring).',
    ('weak_side_balance', None): 'The ball is wide; nobody of the blue team protects the far side (goal-side, in the '
                                 'far half).',
    ('wide_defender_advance', None): 'The ringed blue wide defender has gone forward; nobody covers the channel '
                                     'behind him.',
    ('behind_ball', None): 'The blue team has the ball; too few blue players are behind the ball to protect their goal.',
    ('defensive_spacing', 'lateral_gap'): 'There is a big gap between two blue defenders on the ball side, with an '
                                          'opponent in it.',
    ('defensive_spacing', None): 'The blue team\'s back line is not organised here.',
}


def _split(src, t=None):
    """'tune' / 'holdout', or 'overlap' for a held-out clip's stretch that a tuning clip also covers (BC60 42.4-60 s)."""
    ov = src.get('overlap')
    if ov and t is not None and ov[0]-1e-6 <= t <= ov[1]+1e-6:
        return 'overlap'
    return src['split']


def _clock(t):
    t = max(0.0, float(t))
    m = int(t // 60)
    return f'{m}:{t-60*m:05.2f}'


def _hid(*parts):
    """A label-sheet item id: short, not ordered, carries no stratum (hash of its parts with the sheet seed)."""
    return hashlib.sha1('|'.join(str(p) for p in (SHEET_SEED,)+parts).encode('ascii')).hexdigest()[:10]


def frame_index(result):
    """{frame: {'time', 'people': {pid: (team, box)}}} of a result (boxes normalized [x, y, w, h])."""
    out = {}
    for f in result.get('frames') or ():
        ppl = {}
        for p in f.get('people') or ():
            pid, box = p.get('id'), p.get('box')
            if isinstance(pid, str) and isinstance(box, list) and len(box) == 4:
                team = pid.split('-')[0] if pid[:2] in ('A-', 'B-') else (p.get('team') or '')
                ppl[pid] = (team, [round(float(v), 4) for v in box])
        out[int(f['frame'])] = {'time': float(f['time']), 'people': ppl}
    return out


def _nearest_frame(index, t, fps):
    k = int(round(t*fps))
    if k in index:
        return k
    keys = sorted(index)
    if not keys:
        return None
    return min(keys, key=lambda x: abs(index[x]['time']-t))


def _team_boxes(index, frame, team):
    f = index.get(frame) or {}
    return [box for pid, (tm, box) in sorted((f.get('people') or {}).items()) if tm == team]


def _band_profiles(team):
    from soccer.coaching import profiles as PR
    return {b: PR.resolve({'format': {'players': 11, 'goalkeeper': True}, 'playingLevel': lv, 'ageGroup': age},
                          team=team) for b, (lv, age) in BANDS.items()}


def engine_at(obs, roles, team, t, bands):
    """What PitchIQ measured for a team at time t (the hidden key of a MOMENT item): the phase (engine / coach span)
    and, per opportunity covering t, the per-sample outcome under each band's parameters (+1 / -1 / 0 / None), D1's
    pressure and cover flags, at the opportunity's sample nearest t."""
    from soccer.coaching import contextual as CX
    tl = obs['timeline']
    phase = None
    for sp in tl['phaseSpans']:
        if sp['t0']-1e-6 <= t <= sp['t1']+1e-6:
            phase = {'state': sp['state'], 'source': sp.get('source')}
    out = {'phase': phase, 'detectors': {}}
    for o in (obs.get('opportunities') or [])+(roles.get('opportunities') or []):
        if not (o['t0']-.1 <= t <= o['t1']+.1) or o.get('abstain'):
            continue
        ts_ = o['relations']['t']
        k = min(range(len(ts_)), key=lambda i: abs(ts_[i]-t))
        rec = {'opportunity': o['id'], 't': ts_[k]}
        for b, prof in bands.items():
            dp = prof['detectors'][o['detector']]
            outs, modes = CX.sample_outcomes(o, dp['params'], tl['scale'])
            rec[b] = outs[k]
            if o['detector'] == D1:
                pc = CX.pressure_cover_flags(o['relations'], k, dp['params'], tl['scale'])
                rec[f'{b}Pressure'], rec[f'{b}Cover'] = bool(pc[0]), bool(pc[1])
        out['detectors'].setdefault(o['detector'], rec)
    return out


def _union_length(segments):
    """Seconds covered by a set of (t0, t1) segments."""
    tot, end = 0.0, None
    for t0, t1 in sorted(segments):
        if end is None or t0 > end:
            tot += t1-t0
            end = t1
        elif t1 > end:
            tot += t1-end
            end = t1
    return tot


def moment_items(sources, target=MOMENT_TARGET, seed=SHEET_SEED):
    """MOMENT items per footage type (DESIGN 13.3: ~200, stratified by detector outcome, plus 30% random times).
    The pool is every judged sample of every opportunity (5 Hz), stratum = (detector, outcome at level 4 / U14), the
    evidence moments (the samples a report would show) drawn first within their stratum; strata are taken round robin
    so rare outcomes are not crowded out. Items of one team and video are MOMENT_SPACING_S apart, so short footage
    cannot reach the target: the sheet then holds what the footage allows (capacity: 60% of the segment-union seconds
    x 2 teams / MOMENT_SPACING_S, the packing random spacing reaches), still 70 / 30 stratified / random. Random
    times come from the engine-only runs, weighted by segment length. The hidden key holds what PitchIQ measured at the
    item for every run variant of the same clip (an amateur clip with and without the synthetic coach-marked phase).
    -> (items, keys)."""
    from soccer.coaching import contextual as CX
    rng = random.Random(seed)
    names = {1: 'demonstrated', -1: 'not_demonstrated', 0: 'unclear'}
    by_type = {}
    for src in sources:
        by_type.setdefault(src['footage'], []).append(src)
    items, keys = [], []
    for footage, srcs in sorted(by_type.items()):
        videos = {}
        for src in srcs:
            videos.setdefault(src['video'], []).append(tuple(src['segment']))
        seconds = sum(max(0.0, _union_length(v)-sum(WINDOW)) for v in videos.values())
        total = min(target, int(seconds*2/MOMENT_SPACING_S*.6))
        n_strat = int(round(total*(1-RANDOM_SHARE)))
        strata = {}
        for src in srcs:
            for team, (obs, roles) in sorted(src['teams'].items()):
                prof4 = _band_profiles(team)['intermediate']
                ev = {(m['opportunity'], m['sample']): m['id'] for m in obs['moments']+roles['moments']}
                for o in obs['opportunities']+roles['opportunities']:
                    if o.get('abstain'):
                        continue
                    outs, _ = CX.sample_outcomes(o, prof4['detectors'][o['detector']]['params'],
                                                 obs['timeline']['scale'])
                    for k, t in enumerate(o['relations']['t']):
                        mid = ev.get((o['id'], k))
                        st = (o['detector'], names.get(outs[k], 'not_judged'))
                        strata.setdefault(st, []).append((0 if mid else 1, rng.random(), src, team, float(t), mid))
        for v in strata.values():
            v.sort(key=lambda e: (e[0], e[1]), reverse=True)        # pop() gives the evidence moments first
        chosen, taken = [], {}

        def free(src, team, t):
            return all(abs(t-x) >= MOMENT_SPACING_S for x in taken.get((src['video'], team), ()))
        while len(chosen) < n_strat and any(strata.values()):
            for k in sorted(strata):
                while strata[k] and len(chosen) < n_strat:
                    pr, _, src, team, t, mid = strata[k].pop()
                    if free(src, team, t):
                        chosen.append((src, team, round(t, 3), {'kind': 'outcome', 'detector': k[0], 'outcome': k[1],
                                                                'evidence': mid}))
                        taken.setdefault((src['video'], team), []).append(t)
                        break
        n_rand = total-len(chosen)
        base = [src for src in srcs if '+' not in src['run']] or srcs
        weights = [src['segment'][1]-src['segment'][0] for src in base]
        tries = 0
        while n_rand > 0 and tries < 50000:
            tries += 1
            src = rng.choices(base, weights=weights)[0]
            team = rng.choice(sorted(src['teams']))
            t = round(rng.uniform(src['segment'][0]+WINDOW[0], src['segment'][1]-WINDOW[1]), 2)
            if free(src, team, t):
                chosen.append((src, team, t, {'kind': 'random', 'detector': None, 'outcome': None, 'evidence': None}))
                taken.setdefault((src['video'], team), []).append(t)
                n_rand -= 1
        for src, team, t, stratum in chosen:
            obs, roles = src['teams'][team]
            fps = float(obs['segment']['fps'] or 25.0)
            frame = _nearest_frame(src['index'], t, fps)
            iid = 'M'+_hid('moment', src['run'], team, t)
            items.append({'id': iid, 'footage': footage, 'video': src['video'], 'time': t, 'clock': _clock(t),
                          'team': team, 'teamHint': f'team {team} (tracking labels {team}-..)',
                          'from': round(max(src['segment'][0], t-WINDOW[0]), 2),
                          'to': round(min(src['segment'][1], t+WINDOW[1]), 2),
                          'marks': _team_boxes(src['index'], frame, team) if frame is not None else []})
            keys.append({'id': iid, 'run': src['run'], 'split': _split(src, t), 'footage': footage, 'video': src['video'],
                         'team': team, 'time': t, 'frame': frame, 'stratum': stratum,
                         'engineByRun': _moment_engine(srcs, src['run'], team, t)})
    order = list(range(len(items)))
    rng.shuffle(order)
    return [items[i] for i in order], [keys[i] for i in order]


def _moment_engine(srcs, run, team, t):
    """{run variant: engine_at} of a MOMENT item: what PitchIQ measured at t in every run variant of the same clip
    (same team letters: AM and AM+coach) whose segment holds t."""
    bands = _band_profiles(team)
    by_run = {}
    for s2 in srcs:
        same = s2['run'].split('+')[0] == run.split('+')[0]
        if same and team in s2['teams'] and s2['segment'][0]-1e-6 <= t <= s2['segment'][1]+1e-6:
            by_run[s2['run']] = engine_at(*s2['teams'][team], team, t, bands)
    return by_run


def _engine_stratum(obs, roles, team, detector, t):
    """The rekeyed counterpart of a MOMENT item's sampling stratum: the outcome at level 4 (U14) of the opportunity
    of `detector` with a sample at t (+-2 ms), and the evidence moment there, under the code as it is now; outcome
    'not_judged' when no such sample is judged, None when no opportunity has a sample at t."""
    from soccer.coaching import contextual as CX
    names = {1: 'demonstrated', -1: 'not_demonstrated', 0: 'unclear'}
    prof4 = _band_profiles(team)['intermediate']
    ev = {(m['opportunity'], m['sample']): m['id'] for m in obs['moments']+roles['moments']}
    for o in obs['opportunities']+roles['opportunities']:
        if o['detector'] != detector or o.get('abstain'):
            continue
        ts_ = o['relations']['t']
        hits = [k for k, x in enumerate(ts_) if abs(float(x)-t) <= 2e-3]
        if hits:
            outs, _ = CX.sample_outcomes(o, prof4['detectors'][detector]['params'], obs['timeline']['scale'])
            k = hits[0]
            return {'kind': 'outcome', 'detector': detector, 'outcome': names.get(outs[k], 'not_judged'),
                    'evidence': ev.get((o['id'], k))}
    return {'kind': 'outcome', 'detector': detector, 'outcome': None, 'evidence': None}


def _role_records(roles, pid):
    return [{k: r[k] for k in ('t0', 't1', 'role', 'group', 'side', 'source', 'confidence')}
            for r in roles['rolebook']['records'] if r['id'] == pid]


def _advance_d3(obs, roles, pid, t0, t1, bands):
    """PitchIQ's own D3 spells of the player overlapping an ADVANCE candidate, with each band's outcome."""
    from soccer.coaching import assess as AS
    d3 = []
    for o in roles['opportunities']+obs['opportunities']:
        if o['detector'] != D3 or o['t1'] < t0-1 or o['t0'] > t1+1 or pid not in o['actors']['primary']:
            continue
        rec = {'opportunity': o['id'], 'primary': o['actors']['primary'], 't0': o['t0'], 't1': o['t1'],
               'abstain': o.get('abstain')}
        for b, prof in bands.items():
            c = None if o.get('abstain') else AS.classify(o, prof['detectors'][D3]['params'], obs['timeline']['scale'],
                                                          obs['timeline']['step'])
            rec[b] = (c or {}).get('outcome')
        d3.append(rec)
    return d3


def role_items(sources, min_s=ROLE_MIN_S):
    """ROLE items: every outfield player of the broadcast sources observed at least min_s, with a frame where his box
    is shown (the middle of his longest stretch in view). Key: the role book's records for him."""
    items, keys = [], []
    for src in sources:
        if src['footage'] != 'broadcast':
            continue
        for team, (obs, roles) in sorted(src['teams'].items()):
            players = obs['timeline']['players']
            for pid in sorted(players):
                p = players[pid]
                if p['team'] != team or p['goalkeeper'] or p['observedSeconds'] < min_s:
                    continue
                frames = sorted(f for f, rec in src['index'].items() if pid in rec['people'])
                if not frames:
                    continue
                best, cur = (frames[0], frames[0]), (frames[0], frames[0])
                for f in frames[1:]:
                    cur = (cur[0], f) if f-cur[1] <= 2 else (f, f)
                    if cur[1]-cur[0] > best[1]-best[0]:
                        best = cur
                mid = min(frames, key=lambda f: abs(f-(best[0]+best[1])/2))
                t = src['index'][mid]['time']
                iid = 'R'+_hid('role', src['run'], pid)
                items.append({'id': iid, 'footage': 'broadcast', 'video': src['video'], 'time': round(t, 2),
                              'clock': _clock(t), 'team': team, 'player': pid,
                              'clip': f'{_clock(src["segment"][0])}-{_clock(src["segment"][1])}',
                              'from': round(max(src['segment'][0], t-WINDOW[0]), 2),
                              'to': round(min(src['segment'][1], t+WINDOW[1]), 2),
                              'marks': [src['index'][mid]['people'][pid][1]]})
                keys.append({'id': iid, 'run': src['run'], 'split': src['split'], 'team': team, 'player': pid,
                             'observedSeconds': p['observedSeconds'], 'records': _role_records(roles, pid)})
    return items, keys


def advance_spells(obs, team):
    """Role-free wide-defender advance candidates from CONTEXT's 1 Hz attack-frame track (obs roleInputs.context): a
    player among the team's BACK_PLAYERS deepest outfield players (median depth), in an outer third of the width,
    at least ADVANCE_M (x L/105) beyond the mean depth of the three deepest other outfield players, while the team has
    the ball; consecutive points (1 s, one missing point bridged), at least ADVANCE_MIN_S. Written independently of the
    D3 detector so the sheet does not inherit its role reading (Barcelona's full-backs have no confident role)."""
    ctx = (obs.get('roleInputs') or {}).get('context') or {}
    tr = ctx.get('track') or {}
    ids, ts_ = ctx.get('ids') or [], tr.get('t') or []
    if not ids or not ts_:
        return []
    L, W = ctx['scale']['lengthM'], ctx['scale']['widthM']
    X, Y, ph = tr['x'], tr['y'], tr['phase']
    med = {}
    for j, pid in enumerate(ids):
        xs = sorted(v for v in X[j] if v is not None)
        if len(xs) >= 10:
            med[pid] = xs[len(xs)//2]
    backs = set(sorted(med, key=med.get)[:BACK_PLAYERS])
    out = []
    for j, pid in enumerate(ids):
        if pid not in backs:
            continue
        run = []
        for i, t in enumerate(ts_):
            x, y = X[j][i], Y[j][i]
            ok = False
            if ph[i] == 'own' and x is not None and y is not None and (y < W/3 or y > 2*W/3):
                others = sorted(X[k][i] for k in range(len(ids)) if k != j and X[k][i] is not None)
                if len(others) >= 3:
                    ok = x-sum(others[:3])/3 >= ADVANCE_M*L/105.0
            if ok:
                if run and i-run[-1] > 2:
                    if ts_[run[-1]]-ts_[run[0]] >= ADVANCE_MIN_S-1:
                        out.append((pid, ts_[run[0]], ts_[run[-1]]))
                    run = []
                run.append(i)
        if run and ts_[run[-1]]-ts_[run[0]] >= ADVANCE_MIN_S-1:
            out.append((pid, ts_[run[0]], ts_[run[-1]]))
    return out


def advance_candidates(obs, roles, team):
    """[(player, t0, t1, origin)]: the D3 opportunities PitchIQ made for the team ('engine': the role tier's advance
    spells, cut at 30 s) and the role-free spells of advance_spells ('role-free'); a role-free spell of the same player
    overlapping an engine spell is merged into it ('both'). On the broadcast copies (Oct 22) the role-free rule alone
    found 1 spell with BACK_PLAYERS = 4 (Barcelona's wing-backs stand high all clip and are not among the 4 deepest
    by median) and 5 with 5; the engine made 5 opportunities; the union is 8 spells (BC60 3, BC180 3, BC303 2)."""
    out = []
    for o in sorted((roles.get('opportunities') or [])+(obs.get('opportunities') or []), key=lambda o: o['t0']):
        if o['detector'] == D3 and o['actors']['primary']:
            out.append([o['actors']['primary'][0], float(o['t0']), float(o['t1']), 'engine'])
    for pid, t0, t1 in advance_spells(obs, team):
        hit = [c for c in out if c[0] == pid and c[3] != 'role-free' and c[1] <= t1+1 and t0 <= c[2]+1]
        for c in hit:
            c[3] = 'both'
        if not hit:
            out.append([pid, float(t0), float(t1), 'role-free'])
    return [tuple(c) for c in sorted(out, key=lambda c: (c[1], c[0]))]


def advance_items(sources):
    """ADVANCE items: the candidate spells on broadcast (advance_candidates: PitchIQ's own D3 spells plus the role-free
    ones, so the coach's answers measure both the precision of what PitchIQ judged and what it missed). Key: the
    overlapping D3 opportunities with their outcome in each band (levels 2, 4 and 6) and the candidate's origin; the
    sheet shows only the player, the clip and the questions."""
    items, keys = [], []
    for src in sources:
        if src['footage'] != 'broadcast' or '+' in src['run']:
            continue
        for team, (obs, roles) in sorted(src['teams'].items()):
            bands = _band_profiles(team)
            for pid, t0, t1, origin in advance_candidates(obs, roles, team):
                tm = (t0+t1)/2
                fps = float(obs['segment']['fps'] or 25.0)
                frame = _nearest_frame(src['index'], tm, fps)
                box = (src['index'].get(frame) or {}).get('people', {}).get(pid)
                iid = 'A'+_hid('advance', src['run'], pid, t0)
                items.append({'id': iid, 'footage': 'broadcast', 'video': src['video'], 'time': round(tm, 2),
                              'clock': f'{_clock(t0)}-{_clock(t1+1)}', 'team': team, 'player': pid,
                              'from': round(max(src['segment'][0], t0-2.0), 2),
                              'to': round(min(src['segment'][1], t1+2.0), 2), 'marks': [box[1]] if box else []})
                keys.append({'id': iid, 'run': src['run'], 'split': _split(src, tm), 'team': team, 'player': pid,
                             't0': t0, 't1': t1, 'origin': origin, 'd3': _advance_d3(obs, roles, pid, t0, t1, bands),
                             'detectorAbstain': (roles['detectors'].get(D3) or {}).get('dominantReason')})
    return items, keys


def zone_items(sources, extra=(), limit=48, seed=SHEET_SEED):
    """ZONE items: the zones PitchIQ proposes at level 4 for every failing evidence moment of the sources (cited alone
    by a pseudo work item with every guardrail passing: the real reports of these clips propose none, because every
    real criticism failed a guardrail), plus `extra` candidates (tools: Metrica game 1, see the module docstring),
    stratified by detector up to `limit`. Each item is a clean diagram: positions only, no zone."""
    from soccer.coaching import alternatives as AL, contextual as CX, schema as S
    rng = random.Random(seed)
    cands = []
    for src in sources:
        for team, (obs, roles) in sorted(src['teams'].items()):
            prof = _band_profiles(team)['intermediate']
            moments = {}
            for m in obs['moments']+roles['moments']:
                moments.setdefault(m['opportunity'], []).append(m)
            for o in obs['opportunities']+roles['opportunities']:
                det = o['detector']
                if det == D6 or not prof['detectors'][det]['enabled']:
                    continue
                outs, modes = CX.sample_outcomes(o, prof['detectors'][det]['params'], obs['timeline']['scale'])
                for m in moments.get(o['id'], ()):
                    if m['sample'] >= len(outs) or outs[m['sample']] != -1:
                        continue
                    key = S.finding_key(det, team, 'diag', o['t0'], o['t1'])
                    f = {'id': S.record_id('fnd', key), 'key': key, 'observation': det, 'team': team,
                         'verdict': 'work_on', 'guardrails': {g: 'pass' for g in S.GUARDRAILS},
                         'measurements': {'failureMode': modes[m['sample']]} if modes[m['sample']] else {},
                         'timestamps': [{'evidence': m['id'], 'opportunity': o['id'], 't': m['t'], 'frame': m['frame'],
                                         'seek': m['seek']}]}
                    alt = AL.propose(f, obs, roles, prof)
                    if alt is None or 'zone' not in alt:
                        continue
                    sc = m['scene']
                    cands.append({'source': src['run'], 'split': _split(src, m['t']), 'video': src['video'], 'team': team,
                                  'detector': det, 'failureMode': modes[m['sample']], 'level': 4,
                                  't': round(m['t'], 2), 'side': AL.attack_side(obs, m),
                                  'scale': {k: obs['timeline']['scale'][k] for k in ('lengthM', 'widthM')},
                                  'players': {pid: {'team': p['team'], 'x': round(p['x'], 4), 'y': round(p['y'], 4),
                                                    'goalkeeper': bool(p.get('goalkeeper'))}
                                              for pid, p in sc['players'].items()},
                                  'ball': sc.get('ball'), 'carrier': o['actors'].get('carrier'),
                                  'primary': o['actors']['primary'],
                                  'zone': {'polygon': alt['zone']['polygon'], 'areaM2': alt['zone']['areaM2'],
                                           'display': alt['display']},
                                  'mover': alt['mover'], 'from': alt['from'], 'principle': alt['principle'],
                                  'relation': alt['improves']['relation']})
    real = list(cands)
    pool = {}
    for c in extra:
        pool.setdefault(c['detector'], []).append(c)
    for v in pool.values():
        rng.shuffle(v)
    picked = list(real)
    while len(picked) < limit and any(pool.values()):
        for d in sorted(pool):
            if pool[d] and len(picked) < limit:
                picked.append(pool[d].pop())
    items, keys = [], []
    for c in picked:
        iid = 'Z'+_hid('zone', c['source'], c['team'], c['t'], c['detector'])
        problem = ZONE_PROBLEM.get((c['detector'], c.get('failureMode'))) or ZONE_PROBLEM.get((c['detector'], None))
        src_text = 'video' if c.get('video') else (
            'Metrica Sports sample data, game 1 (tracking only, no video)' if str(c['source']).startswith('metrica')
            else 'tracking only (no video)')
        items.append({'id': iid, 'source': src_text,
                      'video': c.get('video'), 'time': c['t'], 'clock': _clock(c['t']), 'team': c['team'],
                      'side': c['side'], 'scale': c['scale'], 'players': c['players'], 'ball': c['ball'],
                      'carrier': c['carrier'],
                      'ringed': c['primary'] if c['detector'] in (D1, D3) and c.get('failureMode') != 'no_pressure'
                      else [], 'problem': problem})
        keys.append({'id': iid, **{k: c[k] for k in ('source', 'split', 'team', 'detector', 'failureMode', 'level', 't',
                                                    'zone', 'mover', 'from', 'principle', 'relation')}})
    order = list(range(len(items)))
    rng.shuffle(order)
    return [items[i] for i in order], [keys[i] for i in order]


VIDEO_PAGE = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
:root{--bg:#f6f7f5;--fg:#1d2320;--muted:#5d6762;--line:#d5dad6;--accent:#1f7a4d;--mark:#ff2bd6}
@media (prefers-color-scheme: dark){:root{--bg:#161a18;--fg:#e8ece9;--muted:#9aa49f;--line:#323935;--accent:#5fd39a}}
body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,Segoe UI,Arial,sans-serif}
header{padding:12px 16px;border-bottom:1px solid var(--line)}
h1{font-size:18px;margin:0 0 6px}
details{margin:6px 0}.muted{color:var(--muted)}
#player{position:sticky;top:0;z-index:2;background:var(--bg);padding:8px 16px;border-bottom:1px solid var(--line)}
#stage{position:relative;max-width:900px}
#stage video{width:100%;display:block;background:#000}
#marks{position:absolute;inset:0;pointer-events:none}
.mk{position:absolute;border:2px solid var(--mark);border-radius:50%}
table{border-collapse:collapse;margin:8px 16px 40px}
td,th{border-bottom:1px solid var(--line);padding:4px 6px;vertical-align:top;text-align:left}
th{font-weight:600;position:sticky;top:0}
select,input[type=text]{max-width:150px;font:inherit}
tr.done td:first-child{color:var(--accent)}
button{font:inherit;padding:2px 8px;margin:1px}
.bar input{width:340px}
</style></head><body>
<header>
<h1>__TITLE__</h1>
<details open><summary>Instructions</summary>__INSTRUCTIONS__</details>
<div class="bar">Video folder: <input id="folder" type="text"> <button id="saveFolder">Use</button>
<span class="muted">(the folder that holds the videos named in the table; default: the PitchIQ uploads folder)</span></div>
<div class="bar"><button id="export">Download answers (CSV)</button> <label>Load answers: <input id="import" type="file"
accept=".csv"></label> <span id="progress" class="muted"></span></div>
</header>
<div id="player"><div id="stage"><video id="video" controls preload="metadata"></video><div id="marks"></div></div>
<div><button id="cont">Play on to the end of the clip</button> <span id="now" class="muted"></span></div></div>
<table id="sheet"></table>
<script id="data" type="application/json">__DATA__</script>
<script>
(function(){
var D=JSON.parse(document.getElementById('data').textContent);
var KEY='pitchiq-labels-'+D.set+'-'+D.sha;
var ans={};try{ans=JSON.parse(localStorage.getItem(KEY)||'{}')}catch(e){ans={}}
var folder=D.folder;try{folder=localStorage.getItem('pitchiq-label-folder')||D.folder}catch(e){}
var v=document.getElementById('video'),marks=document.getElementById('marks'),cur=null,stopAt=null;
document.getElementById('folder').value=folder;
document.getElementById('saveFolder').onclick=function(){folder=document.getElementById('folder').value;
 if(folder&&!/[\/\\]$/.test(folder))folder+='/';try{localStorage.setItem('pitchiq-label-folder',folder)}catch(e){}};
function save(){try{localStorage.setItem(KEY,JSON.stringify(ans))}catch(e){}progress()}
function progress(){var n=0;D.items.forEach(function(it){if(ans[it.id]&&D.questions.every(function(q){return !q.options||ans[it.id][q.name]}))n++});
 document.getElementById('progress').textContent=n+' of '+D.items.length+' rows complete';
 D.items.forEach(function(it){var r=document.getElementById('r'+it.id);if(r)r.className=(ans[it.id]&&D.questions.every(function(q){return !q.options||ans[it.id][q.name]}))?'done':''})}
function showMarks(it){marks.innerHTML='';if(!it)return;(it.marks||[]).forEach(function(b){var d=document.createElement('div');
 d.className='mk';var cx=b[0]+b[2]/2,by=b[1]+b[3];var w=Math.max(b[2]*2.2,.02);
 d.style.left=(100*(cx-w/2))+'%';d.style.width=(100*w)+'%';d.style.top=(100*(by-w*0.6))+'%';d.style.height=(100*w*1.2*16/9*0.5)+'%';marks.appendChild(d)})}
function play(it){cur=it;showMarks(null);var src=folder+it.video;if(v.getAttribute('src')!==src){v.setAttribute('src',src)}
 var go=function(){v.currentTime=it.from;stopAt=it.time;v.play().catch(function(){})};
 if(v.readyState>=1)go();else v.addEventListener('loadedmetadata',go,{once:true})}
v.addEventListener('timeupdate',function(){document.getElementById('now').textContent=clock(v.currentTime);
 if(stopAt!==null&&v.currentTime>=stopAt){v.pause();v.currentTime=stopAt;stopAt=null;showMarks(cur)}});
v.addEventListener('play',function(){if(stopAt===null)showMarks(null)});
document.getElementById('cont').onclick=function(){if(!cur)return;showMarks(null);stopAt=cur.to;v.play().catch(function(){})};
function clock(t){var m=Math.floor(t/60);var s=(t-60*m).toFixed(2);return m+':'+(s.length<5?'0':'')+s}
var tb=document.getElementById('sheet');var h=tb.insertRow();
['#','Clip'].concat(D.info.map(function(c){return c.label})).concat(D.questions.map(function(q){return q.label})).concat(['Notes'])
 .forEach(function(x){var th=document.createElement('th');th.textContent=x;h.appendChild(th)});
D.items.forEach(function(it,i){var r=tb.insertRow();r.id='r'+it.id;r.insertCell().textContent=(i+1)+' ('+it.id+')';
 var c=r.insertCell();var b=document.createElement('button');b.textContent='Play '+it.clock;b.onclick=function(){play(it)};c.appendChild(b);
 D.info.forEach(function(col){r.insertCell().textContent=it[col.key]===undefined?'':String(it[col.key])});
 var a=ans[it.id]||(ans[it.id]={});
 D.questions.forEach(function(q){var cell=r.insertCell();var el;
  if(q.options){el=document.createElement('select');var o0=document.createElement('option');o0.value='';o0.textContent='-';el.appendChild(o0);
   q.options.forEach(function(o){var op=document.createElement('option');op.value=o;op.textContent=o;el.appendChild(op)})}
  else{el=document.createElement('input');el.type='text'}
  el.value=a[q.name]||'';el.onchange=function(){a[q.name]=el.value;save()};cell.appendChild(el)});
 var n=document.createElement('input');n.type='text';n.value=a.notes||'';n.onchange=function(){a.notes=n.value;save()};r.insertCell().appendChild(n)});
function csvq(s){s=String(s==null?'':s);return /[",\n]/.test(s)?'"'+s.replace(/"/g,'""')+'"':s}
document.getElementById('export').onclick=function(){var cols=['id'].concat(D.questions.map(function(q){return q.name})).concat(['notes']);
 var lines=[cols.join(',')];D.items.forEach(function(it){var a=ans[it.id]||{};lines.push([it.id].concat(cols.slice(1).map(function(c){return csvq(a[c])})).join(','))});
 var blob=new Blob([lines.join('\r\n')+'\r\n'],{type:'text/csv'});var u=document.createElement('a');u.href=URL.createObjectURL(blob);
 u.download=D.set+'_labels.csv';document.body.appendChild(u);u.click();setTimeout(function(){URL.revokeObjectURL(u.href);u.remove()},500)};
document.getElementById('import').onchange=function(e){var f=e.target.files[0];if(!f)return;var rd=new FileReader();rd.onload=function(){
 var rows=parseCsv(rd.result);var head=rows.shift()||[];rows.forEach(function(r){var id=r[0];if(!id)return;var a=ans[id]||(ans[id]={});
 head.forEach(function(c,i){if(i>0&&r[i]!==undefined)a[c]=r[i]})});save();location.reload()};rd.readAsText(f)};
function parseCsv(t){var out=[],row=[],f='',q=false;for(var i=0;i<t.length;i++){var ch=t[i];
 if(q){if(ch=='"'&&t[i+1]=='"'){f+='"';i++}else if(ch=='"')q=false;else f+=ch}
 else if(ch=='"')q=true;else if(ch==','){row.push(f);f=''}else if(ch=='\n'){row.push(f.replace(/\r$/,''));out.push(row);row=[];f=''}else f+=ch}
 if(f||row.length){row.push(f);out.push(row)}return out}
progress();
})();
</script></body></html>
'''

ZONE_PAGE = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
:root{--bg:#f6f7f5;--fg:#1d2320;--muted:#5d6762;--line:#d5dad6;--pitch:#e7f0e9;--pl:#9fb7a6;--blue:#1d5fd1;--red:#c9342f;--draw:#ff8a00;--pi:#14a05a}
@media (prefers-color-scheme: dark){:root{--bg:#161a18;--fg:#e8ece9;--muted:#9aa49f;--line:#323935;--pitch:#1f2a23;--pl:#4b5f52}}
body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,Segoe UI,Arial,sans-serif}
main{max-width:980px;margin:0 auto;padding:12px 16px 40px}h1{font-size:18px;margin:0 0 6px}.muted{color:var(--muted)}
svg{width:100%;height:auto;background:var(--pitch);border:1px solid var(--line);touch-action:manipulation}
.row{margin:8px 0}button{font:inherit;padding:3px 10px;margin:2px}label{margin-right:10px}
.problem{font-weight:600}
</style></head><body><main>
<h1>__TITLE__</h1>
<details open><summary>Instructions</summary>__INSTRUCTIONS__</details>
<div class="row"><button id="prev">Previous</button> <button id="next">Next</button> <span id="pos" class="muted"></span>
 <button id="export">Download answers (JSON)</button></div>
<div class="row muted" id="meta"></div>
<div class="row problem" id="problem"></div>
<div class="row">Do you agree there is a problem here?
 <label><input type="radio" name="agree" value="yes">yes</label><label><input type="radio" name="agree" value="no">no</label>
 <label><input type="radio" name="agree" value="unclear">unclear</label></div>
<div class="row"><b>Mode:</b> <label><input type="radio" name="mode" value="draw" checked>draw the area (click corners)</label>
 <label><input type="radio" name="mode" value="pick">pick the blue player you would move</label>
 <button id="undo">Undo corner</button> <button id="clear">Clear area</button> <span id="picked" class="muted"></span></div>
<svg id="pitch" viewBox="0 0 1050 680" role="img" aria-label="pitch diagram"></svg>
<div class="row">Notes: <input id="notes" type="text" style="width:60%"></div>
<script id="data" type="application/json">__DATA__</script>
<script>
(function(){
var D=JSON.parse(document.getElementById('data').textContent);var KEY='pitchiq-labels-'+D.set+'-'+D.sha;
var ans={};try{ans=JSON.parse(localStorage.getItem(KEY)||'{}')}catch(e){ans={}}
var i=0,svg=document.getElementById('pitch'),NS='http://www.w3.org/2000/svg';
function save(){try{localStorage.setItem(KEY,JSON.stringify(ans))}catch(e){}}
function el(n,a,p){var e=document.createElementNS(NS,n);for(var k in a)e.setAttribute(k,a[k]);(p||svg).appendChild(e);return e}
function cur(){var it=D.items[i];return ans[it.id]||(ans[it.id]={polygon:[],mover:null,agree:null,notes:''})}
function draw(){var it=D.items[i],a=cur(),W=1050,H=680;svg.innerHTML='';
 el('rect',{x:0,y:0,width:W,height:H,fill:'none',stroke:'var(--pl)','stroke-width':3});
 el('line',{x1:W/2,y1:0,x2:W/2,y2:H,stroke:'var(--pl)','stroke-width':2});
 el('circle',{cx:W/2,cy:H/2,r:91.5,fill:'none',stroke:'var(--pl)','stroke-width':2});
 [0,1].forEach(function(s){var x0=s?W-165:0;el('rect',{x:x0,y:H/2-201.6,width:165,height:403.2,fill:'none',stroke:'var(--pl)','stroke-width':2});
  el('rect',{x:s?W-55:0,y:H/2-91.6,width:55,height:183.2,fill:'none',stroke:'var(--pl)','stroke-width':2})});
 var gx=it.side==='left'?14:W-14;el('text',{x:it.side==='left'?20:W-20,y:24,'text-anchor':it.side==='left'?'start':'end',fill:'var(--blue)','font-size':20}).textContent='blue defends this goal';
 el('rect',{x:it.side==='left'?0:W-8,y:H/2-36,width:8,height:72,fill:'var(--blue)'});
 if(a.polygon.length){el('polygon',{points:a.polygon.map(function(q){return q[0]*W+','+q[1]*H}).join(' '),fill:'rgba(255,138,0,.25)',stroke:'var(--draw)','stroke-width':3});
  a.polygon.forEach(function(q){el('circle',{cx:q[0]*W,cy:q[1]*H,r:5,fill:'var(--draw)'})})}
 Object.keys(it.players).sort().forEach(function(pid){var p=it.players[pid],blue=p.team===it.team,cx=p.x*W,cy=p.y*H;
  var c=el('circle',{cx:cx,cy:cy,r:11,fill:blue?'var(--blue)':'var(--red)',stroke:p.goalkeeper?'#ffd400':'#fff','stroke-width':p.goalkeeper?4:2});
  c.style.cursor='pointer';c.addEventListener('click',function(ev){if(mode()==='pick'&&blue){ev.stopPropagation();a.mover=pid;save();draw()}});
  if((it.ringed||[]).indexOf(pid)>=0)el('circle',{cx:cx,cy:cy,r:19,fill:'none',stroke:'var(--fg)','stroke-width':3});
  if(pid===it.carrier)el('circle',{cx:cx,cy:cy,r:19,fill:'none',stroke:'#000','stroke-width':4,'stroke-dasharray':'4 3'});
  if(pid===a.mover)el('circle',{cx:cx,cy:cy,r:24,fill:'none',stroke:'var(--draw)','stroke-width':4});
  el('text',{x:cx+13,y:cy-12,fill:'var(--fg)','font-size':14}).textContent=pid});
 if(it.ball)el('circle',{cx:it.ball[0]*W,cy:it.ball[1]*H,r:6,fill:'#000',stroke:'#fff','stroke-width':2});
 document.getElementById('pos').textContent=(i+1)+' of '+D.items.length+' ('+it.id+')';
 document.getElementById('meta').textContent=it.source+(it.video?(', video '+it.video+' at '+it.clock):(', time '+it.clock))+'; pitch '+it.scale.lengthM+' x '+it.scale.widthM+' m';
 document.getElementById('problem').textContent=it.problem;
 Array.prototype.forEach.call(document.getElementsByName('agree'),function(r){r.checked=(a.agree===r.value)});
 document.getElementById('notes').value=a.notes||'';document.getElementById('picked').textContent=a.mover?('player to move: '+a.mover):''}
function mode(){var m='draw';Array.prototype.forEach.call(document.getElementsByName('mode'),function(r){if(r.checked)m=r.value});return m}
svg.addEventListener('click',function(ev){if(mode()!=='draw')return;var pt=svg.createSVGPoint();pt.x=ev.clientX;pt.y=ev.clientY;
 var q=pt.matrixTransform(svg.getScreenCTM().inverse());cur().polygon.push([+(q.x/1050).toFixed(4),+(q.y/680).toFixed(4)]);save();draw()});
document.getElementById('undo').onclick=function(){cur().polygon.pop();save();draw()};
document.getElementById('clear').onclick=function(){cur().polygon=[];save();draw()};
Array.prototype.forEach.call(document.getElementsByName('agree'),function(r){r.onchange=function(){cur().agree=r.value;save()}});
document.getElementById('notes').onchange=function(){cur().notes=this.value;save()};
document.getElementById('prev').onclick=function(){if(i>0){i--;draw()}};
document.getElementById('next').onclick=function(){if(i<D.items.length-1){i++;draw()}};
document.getElementById('export').onclick=function(){var out={set:D.set,sha:D.sha,answers:ans};
 var blob=new Blob([JSON.stringify(out,null,1)],{type:'application/json'});var u=document.createElement('a');u.href=URL.createObjectURL(blob);
 u.download=D.set+'_labels.json';document.body.appendChild(u);u.click();setTimeout(function(){URL.revokeObjectURL(u.href);u.remove()},500)};
draw();
})();
</script></main></body></html>
'''

REVEAL_PAGE = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>ZONE reveal (after drawing)</title>
<style>
:root{--bg:#f6f7f5;--fg:#1d2320;--muted:#5d6762;--line:#d5dad6;--pitch:#e7f0e9;--pl:#9fb7a6;--blue:#1d5fd1;--red:#c9342f;--pi:#0f9d58}
@media (prefers-color-scheme: dark){:root{--bg:#161a18;--fg:#e8ece9;--muted:#9aa49f;--line:#323935;--pitch:#1f2a23;--pl:#4b5f52}}
body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,Segoe UI,Arial,sans-serif}
main{max-width:980px;margin:0 auto;padding:12px 16px 40px}.muted{color:var(--muted)}
svg{width:100%;height:auto;background:var(--pitch);border:1px solid var(--line)}button{font:inherit;padding:3px 10px}
</style></head><body><main>
<h1>ZONE reveal: PitchIQ's suggested area</h1>
<p><b>Open this page only after you have drawn every area on the ZONE sheet.</b> Green: the area PitchIQ suggests, with
the player it would move (green ring) and where he was. Rate whether it is a plausible place for one team-mate to have
been within about 2 seconds.</p>
<div><button id="prev">Previous</button> <button id="next">Next</button> <span id="pos" class="muted"></span>
 <button id="export">Download ratings (JSON)</button></div>
<p id="problem"></p>
<p>Plausible? <label><input type="radio" name="pl" value="yes">yes</label> <label><input type="radio" name="pl" value="no">no</label>
 <label><input type="radio" name="pl" value="unclear">unclear</label> &nbsp; A sensible player to move?
 <label><input type="radio" name="mv" value="yes">yes</label> <label><input type="radio" name="mv" value="no">no</label></p>
<svg id="pitch" viewBox="0 0 1050 680"></svg>
<script id="data" type="application/json">__DATA__</script>
<script>
(function(){var D=JSON.parse(document.getElementById('data').textContent);var KEY='pitchiq-reveal-'+D.sha;
var ans={};try{ans=JSON.parse(localStorage.getItem(KEY)||'{}')}catch(e){}var i=0,svg=document.getElementById('pitch'),NS='http://www.w3.org/2000/svg';
function el(n,a){var e=document.createElementNS(NS,n);for(var k in a)e.setAttribute(k,a[k]);svg.appendChild(e);return e}
function save(){try{localStorage.setItem(KEY,JSON.stringify(ans))}catch(e){}}
function draw(){var it=D.items[i],W=1050,H=680,a=ans[it.id]||(ans[it.id]={});svg.innerHTML='';
 el('rect',{x:0,y:0,width:W,height:H,fill:'none',stroke:'var(--pl)','stroke-width':3});el('line',{x1:W/2,y1:0,x2:W/2,y2:H,stroke:'var(--pl)','stroke-width':2});
 el('rect',{x:it.side==='left'?0:W-8,y:H/2-36,width:8,height:72,fill:'var(--blue)'});
 el('polygon',{points:it.zone.map(function(q){return q[0]*W+','+q[1]*H}).join(' '),fill:'rgba(15,157,88,.3)',stroke:'var(--pi)','stroke-width':3});
 Object.keys(it.players).sort().forEach(function(pid){var p=it.players[pid],blue=p.team===it.team;
  el('circle',{cx:p.x*W,cy:p.y*H,r:11,fill:blue?'var(--blue)':'var(--red)',stroke:'#fff','stroke-width':2});
  if(pid===it.mover)el('circle',{cx:p.x*W,cy:p.y*H,r:22,fill:'none',stroke:'var(--pi)','stroke-width':4});
  el('text',{x:p.x*W+13,y:p.y*H-12,fill:'var(--fg)','font-size':14}).textContent=pid});
 if(it.ball)el('circle',{cx:it.ball[0]*W,cy:it.ball[1]*H,r:6,fill:'#000',stroke:'#fff','stroke-width':2});
 document.getElementById('pos').textContent=(i+1)+' of '+D.items.length+' ('+it.id+')';document.getElementById('problem').textContent=it.problem;
 ['pl','mv'].forEach(function(n){Array.prototype.forEach.call(document.getElementsByName(n),function(r){r.checked=(a[n]===r.value);r.onchange=function(){a[n]=r.value;save()}})})}
document.getElementById('prev').onclick=function(){if(i>0){i--;draw()}};document.getElementById('next').onclick=function(){if(i<D.items.length-1){i++;draw()}};
document.getElementById('export').onclick=function(){var blob=new Blob([JSON.stringify({set:'zone_reveal',sha:D.sha,answers:ans},null,1)],{type:'application/json'});
 var u=document.createElement('a');u.href=URL.createObjectURL(blob);u.download='zone_reveal_labels.json';document.body.appendChild(u);u.click();setTimeout(function(){URL.revokeObjectURL(u.href);u.remove()},500)};
draw()})();
</script></main></body></html>
'''


INSTRUCTIONS = {
    'moment': '''<p>Each row is one moment of a match video. Press <b>Play</b>: the clip runs from 3 s before the moment
and stops on it; the pink rings then mark the players of the team the row asks about (their tracking labels are in the
"Team" column). <b>Play on</b> shows the next 2 s. Answer every question that applies; choose "not applicable" when the
question is about the other phase (for example "pressure" while this team had the ball), and "unclear" when the video
does not show enough. The three "raise" questions ask whether a coach of that age band would point out a positioning
problem of this team at this moment.</p><p>Do not open PitchIQ's coaching report for these videos before you finish:
the sheet is blind on purpose. Your answers save in this browser as you go; <b>Download answers</b> writes
moment_labels.csv (give it to the scorer).</p>''',
    'role': '''<p>Each row is one player. Press <b>Play</b>: the clip stops on a moment where the player is ringed in pink
(his tracking label is in the "Player" column). Watch as much of the clip as you need (the whole clip time is in the
"Clip" column) and choose the position he mainly played, how sure you are, and whether he changed position during the
clip. Use the team's real shape, not the formation announced before the match.</p><p>Blind sheet: do not open
PitchIQ's coaching report first. <b>Download answers</b> writes role_labels.csv.</p>''',
    'advance': '''<p>Each row is a spell in which a defender of the team was well forward on one side while his team had
the ball (found from the players' positions only, so some rows are not full-backs at all: say so). Press <b>Play</b>:
the clip starts 2 s before the spell and stops in its middle, with the player ringed; <b>Play on</b> runs to 2 s after
it. "Covered" means the space behind him was protected: a centre-back moved across, a midfielder dropped in, or a
team-mate stood in the channel behind him.</p><p>Blind sheet: do not open PitchIQ's coaching report first.
<b>Download answers</b> writes advance_labels.csv.</p>''',
    'zone': '''<p>Each page is one moment, drawn from above: blue is the team the question is about (it defends the
marked goal), red the opponents, the black dot the ball, a dashed black ring the ball carrier, a thick ring the player
the problem is about. Read the problem, say whether you agree it is a problem, then <b>click the corners of the area
where ONE blue team-mate could have stood to fix it</b> (he has about 2 seconds to get there), and pick the blue
player you would move. Only then, when every page is done, open <code>key/zone_reveal.html</code> to rate PitchIQ's
area.</p><p><b>Download answers</b> writes zone_labels.json.</p>''',
}


def _page(template, title, instructions, data):
    blob = json.dumps(data, ensure_ascii=True, separators=(',', ':'), allow_nan=False).replace('</', '<\\/')
    return template.replace('__TITLE__', html.escape(title)).replace('__INSTRUCTIONS__', instructions).replace(
        '__DATA__', blob)


def _csv(rows, cols):
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator='\n')
    w.writerow(cols)
    for r in rows:
        w.writerow(['' if r.get(c) is None else r.get(c) for c in cols])
    return buf.getvalue()


def _questions(qs):
    return [{'name': n, 'label': lab, 'options': list(opts) if opts else None} for n, lab, opts in qs]


def make_sheets(sources, out, zone_extra=(), folder='../../../../uploads/'):
    """Write the four blind sheets (HTML + CSV each), INSTRUCTIONS.txt, key/*.json (PitchIQ's side, never shown by a
    sheet), key/zone_reveal.html and manifest.json (sha256 of every file) into `out`. sources: [{'run', 'video',
    'footage' broadcast|amateur|veo, 'split' tune|holdout|overlap, 'segment' (t0, t1), 'index' (frame_index of the
    result), 'teams' {team: (obs, roles)}}]. -> the manifest."""
    out = _safe_dir(out)
    sets = {}
    sets['moment'] = moment_items(sources)
    sets['role'] = role_items(sources)
    sets['advance'] = advance_items(sources)
    sets['zone'] = zone_items(sources, extra=zone_extra)
    files = {}
    spec = {'moment': ('MOMENT sheet: what happened at this moment?', MOMENT_QUESTIONS,
                       [{'key': 'video', 'label': 'Video'}, {'key': 'clock', 'label': 'Time'},
                        {'key': 'teamHint', 'label': 'Team'}]),
            'role': ('ROLE sheet: which position did this player play?', ROLE_QUESTIONS,
                     [{'key': 'video', 'label': 'Video'}, {'key': 'clip', 'label': 'Clip'},
                      {'key': 'player', 'label': 'Player'}]),
            'advance': ('ADVANCE sheet: was the wide defender\'s forward run covered?', ADVANCE_QUESTIONS,
                        [{'key': 'video', 'label': 'Video'}, {'key': 'clock', 'label': 'Spell'},
                         {'key': 'player', 'label': 'Player'}])}
    frozen = frozen_state()
    for name, (items, keys) in sets.items():
        sha = sha_json(items)[:16]
        # fix round 1 (critic N1 / B-9): a key names the rules it was computed with; the scorer refuses it under others
        # (fix round 3, CODE round-2 L-7: every key ends with a newline; the manifest hashes the bytes written)
        files[f'key/{name}_key.json'] = json.dumps({'set': name, 'sheetSha': sha, 'version': SHEET_VERSION,
                                                   'frozen': frozen, 'keys': keys},
                                                  indent=None, ensure_ascii=True, allow_nan=False)+'\n'
        if name == 'zone':
            data = {'set': 'zone', 'sha': sha, 'items': items}
            files['zone_sheet.html'] = _page(ZONE_PAGE, 'ZONE sheet: where could one team-mate have been?',
                                             INSTRUCTIONS['zone'], data)
            kz = {k['id']: k for k in keys}
            reveal = [{'id': it['id'], 'team': it['team'], 'side': it['side'], 'players': it['players'],
                       'ball': it['ball'], 'problem': it['problem'], 'zone': kz[it['id']]['zone']['polygon'],
                       'mover': kz[it['id']]['mover']} for it in items]
            files['key/zone_reveal.html'] = REVEAL_PAGE.replace('__DATA__', json.dumps(
                {'sha': sha, 'items': reveal}, ensure_ascii=True, separators=(',', ':')).replace('</', '<\\/'))
            files['zone_items.csv'] = _csv(items, ['id', 'source', 'video', 'clock', 'team', 'problem'])
            continue
        title, qs, info = spec[name]
        data = {'set': name, 'sha': sha, 'folder': folder, 'items': items, 'questions': _questions(qs), 'info': info}
        files[f'{name}_sheet.html'] = _page(VIDEO_PAGE, title, INSTRUCTIONS[name], data)
        cols = ['id']+[c['key'] for c in info]+['time']+[n for n, _, _ in qs]+['notes']
        files[f'{name}_sheet.csv'] = _csv(items, cols)
    files['INSTRUCTIONS.txt'] = _instructions_text(sets, folder)
    files['key/README.txt'] = ('PitchIQ\'s side of the label sheets (what it measured at each item). Do NOT open these '
                               'files, or zone_reveal.html before the ZONE drawings are done: the sheets are blind.\n'
                               'The scorer (python tools/coaching_eval.py --labels DIR) reads them.\n')
    manifest = {'version': SHEET_VERSION, 'seed': SHEET_SEED, 'generated': time.strftime('%Y-%m-%d'),
                'sources': [{k: s[k] for k in ('run', 'video', 'footage', 'split')} | {'segment': list(s['segment'])}
                            for s in sources],
                'counts': {name: len(items) for name, (items, _) in sets.items()},
                'countsByFootage': {name: _count_by(items, 'footage') for name, (items, _) in sets.items()
                                    if name != 'zone'},
                'zoneSources': _count_by(sets['zone'][1], 'source'),
                'seen': {name: sum(1 for k in keys if _seen_of(k)) for name, (_, keys) in sets.items()},
                'seenSpans': [list(s[:3])+[list(s[3])] for s in SEEN_SPANS], 'files': {}}
    for rel, text in sorted(files.items()):
        write_text(out/rel, text)
        manifest['files'][rel] = hashlib.sha256(text.replace('\r\n', '\n').replace('\n', '\r\n').encode(
            'ascii', 'replace')).hexdigest()
    write_text(out/'manifest.json', json.dumps(manifest, indent=1, ensure_ascii=True)+'\n')
    return manifest


def _count_by(rows, key):
    out = {}
    for r in rows:
        out[str(r.get(key))] = out.get(str(r.get(key)), 0)+1
    return out


def _instructions_text(sets, folder):
    n = {k: len(v[0]) for k, v in sets.items()}
    return f'''PitchIQ coaching: blind label sheets (DESIGN 13.3)

Who: a coach who was not involved in tuning PitchIQ. What you see: the match video at a timestamp (or a clean diagram),
never PitchIQ's coaching report. Do not open the Coaching tab or a coaching PDF of these videos before you finish.

The sheets (open the .html files in a browser; or fill the .csv files in a spreadsheet, same columns):
  moment_sheet.html   {n['moment']} moments: who had the ball, pressure, cover, far side, back line, players behind the
                      ball (rated for youth / intermediate / senior), spread, and whether you would raise a problem.
  role_sheet.html     {n['role']} players: the position each played.
  advance_sheet.html  {n['advance']} spells: was a wide defender's forward run covered?
  zone_sheet.html     {n['zone']} moments on a diagram: draw where one team-mate could have stood to fix the problem.
                      AFTER finishing: key/zone_reveal.html (rate PitchIQ's own area).

Videos: the pages look for the videos in "{folder}" (the PitchIQ uploads folder seen from
tools/annotations/coaching/labels/); type another folder at the top of a page if yours differ. Times are seconds of
the source video.

Saving: answers save in the browser as you go; press "Download answers" on each page and put the files
(moment_labels.csv, role_labels.csv, advance_labels.csv, zone_labels.json, zone_reveal_labels.json) in one folder.
Then: python tools/coaching_eval.py --labels THAT_FOLDER --sheets tools/annotations/coaching/labels

About 6-8 hours in all (MOMENT about 4 of them); it can be done over several sittings, since the answers save as you
go. Nothing PitchIQ reports is called validated until these are scored.
'''


# ---------- the scorer ----------
FROZEN_FILES = ('soccer/coaching/profiles.json', 'soccer/coaching/drills.json', 'soccer/coaching/progress_rules.json',
                'soccer/coaching/sources.json')


def frozen_state():
    """sha256 (line endings normalized) of the data files and the coaching modules a score depends on: DESIGN 13.3
    scores held-out labels once, after freezing profiles.json, with the sha recorded; a later score whose 'frozen'
    differs was made on different rules and is not the held-out result."""
    rels = list(FROZEN_FILES)+sorted(p.relative_to(REPO).as_posix() for p in (REPO/'soccer'/'coaching').glob('*.py'))
    crlf, lf = bytes((13, 10)), bytes((10,))
    return {rel: hashlib.sha256((REPO/rel).read_bytes().replace(crlf, lf)).hexdigest() for rel in rels}


def frozen_diff(frozen):
    """The files whose sha differs between a key's frozen block and the code as it is now ('no frozen block' for a
    key made before fix round 1)."""
    if not isinstance(frozen, dict):
        return ['no frozen block (a key made before the freeze protocol: rekey it after the freeze)']
    now = frozen_state()
    return sorted(rel for rel in set(now) | set(frozen) if now.get(rel) != frozen.get(rel))


def _seen_dets(run, t):
    """The detectors for which an item at (run, t) is 'seen' data (SEEN_SPANS, critic N3)."""
    base = str(run).split('+')[0]
    out = set()
    for r, a, b, dets in SEEN_SPANS:
        if base == r and t is not None and a-1e-6 <= float(t) <= b+1e-6:
            out |= set(dets)
    return out


def _seen_of(k):
    t = k.get('time', k.get('t', (k.get('t0', 0.0)+k.get('t1', 0.0))/2 if 't0' in k else None))
    if 'detector' in k and 'engineByRun' not in k:
        return k['detector'] in _seen_dets(k.get('source') or k.get('run'), t)
    return bool(_seen_dets(k.get('run') or k.get('source'), t))


def _comparison_detector(name):
    n = name.split(':', 1)[-1]
    if n.startswith('strength.'):
        n = n.split('.', 1)[1]+'.'
    if n in ('pressure', 'cover'):
        return D1
    code = n.split('.', 1)[0]
    return {v: k for k, v in S_CODE.items()}.get(code)


def _base_split(k):
    return 'tune' if k['split'] == 'overlap' else k['split']


REKEY_SETS = ('moment', 'role', 'advance')


def rekey_keys(sheets_dir, sources, out):
    """B-9 (critic N1): PitchIQ's side of the MOMENT, ROLE and ADVANCE sheets recomputed for the SAME items (ids,
    times and players, which is what a coach labels) on `sources` measured with the code as it is now (make_sheets'
    own per-item helpers), written to out/<set>_key.json with that code's frozen block and out/rekey_manifest.json.
    The sampling stratum of a MOMENT item is kept (how it was drawn) and its rekeyed counterpart added
    (engineStratum). The ZONE sheet is not rekeyed: its items show PitchIQ's own problem text, so it is regenerated
    after the freeze (critic N4). -> the rekey manifest (with per-set counts of items whose engine fields changed)."""
    sheets_dir = Path(sheets_dir)
    out = _safe_dir(out)
    manifest = json.loads((sheets_dir/'manifest.json').read_text(encoding='ascii'))
    frozen = frozen_state()
    by_run = {s['run']: s for s in sources}
    res = {'version': SHEET_VERSION, 'frozen': frozen, 'sheets': str(sheets_dir), 'generated': time.strftime(
        '%Y-%m-%d %H:%M:%S'), 'files': {}, 'changed': {}, 'notRekeyed': {'zone': 'regenerate the ZONE sheet after the '
                                                                                  'freeze (critic N4)'}}
    for name in REKEY_SETS:
        rel = f'key/{name}_key.json'
        text = (sheets_dir/rel).read_bytes()
        if hashlib.sha256(text).hexdigest() != manifest['files'][rel]:
            raise SystemExit(f'{rel} differs from the manifest: the key changed after the sheets were made')
        old = json.loads(text.decode('ascii'))
        keys, changed = [], 0
        for k in old['keys']:
            src = by_run.get(k['run'])
            if src is None or k['team'] not in src['teams']:
                raise SystemExit(f'{name} {k["id"]}: no source {k["run"]}/{k["team"]} to rekey from')
            obs, roles = src['teams'][k['team']]
            k2 = dict(k)
            if name == 'moment':
                k2['engineByRun'] = _moment_engine(sources, k['run'], k['team'], k['time'])
                st = k.get('stratum') or {}
                if st.get('kind') == 'outcome':
                    k2['engineStratum'] = _engine_stratum(obs, roles, k['team'], st['detector'], k['time'])
                diff = _jsonable(k2['engineByRun']) != _jsonable(k['engineByRun']) or (
                    'engineStratum' in k2 and (k2['engineStratum']['outcome'], k2['engineStratum']['evidence'])
                    != (st.get('outcome'), st.get('evidence')))
            elif name == 'role':
                k2['records'] = _role_records(roles, k['player'])
                k2['observedSeconds'] = (obs['timeline']['players'].get(k['player']) or {}).get('observedSeconds')
                diff = k2['records'] != k['records']
            else:
                k2['d3'] = _advance_d3(obs, roles, k['player'], k['t0'], k['t1'], _band_profiles(k['team']))
                k2['detectorAbstain'] = (roles['detectors'].get(D3) or {}).get('dominantReason')
                diff = _jsonable(k2['d3']) != _jsonable(k['d3'])
            changed += int(bool(diff))
            keys.append(k2)
        doc = {'set': name, 'sheetSha': old['sheetSha'], 'version': SHEET_VERSION, 'frozen': frozen,
               'rekeyedFrom': manifest['files'][rel], 'keys': keys}
        text2 = json.dumps(doc, indent=None, ensure_ascii=True, allow_nan=False)+'\n'    # (L-7: a final newline)
        path = write_text(out/f'{name}_key.json', text2)
        res['files'][f'{name}_key.json'] = hashlib.sha256(path.read_bytes()).hexdigest()
        res['changed'][name] = {'items': len(keys), 'engineFieldsChanged': changed}
    write_text(out/'rekey_manifest.json', json.dumps(res, indent=1, ensure_ascii=True)+'\n')
    return res


def _read_csv(path):
    with open(path, newline='', encoding='utf-8-sig') as fh:
        return {r['id']: r for r in csv.DictReader(fh) if r.get('id')}


def _rate(hits, n):
    return None if not n else round(hits/n, 3)


def _yes(v):
    return {'yes': 1, 'no': -1, 'covered': 1, 'not covered': -1}.get((v or '').strip().lower())


def _moment_compare(engine, row, add, pre=''):
    """Add one MOMENT row's comparisons (PitchIQ's per-sample outcome vs the coach's answer, +1 / -1) under names
    prefixed by `pre`: the phase, D1 pressure and cover flags and its outcome per band, D2 far side, D5 back line,
    D6 spread at level 4, D4 behind the ball per band, and "would you raise it" per band against any failing check."""
    eng = engine['detectors']
    ph = (engine['phase'] or {}).get('state')
    coach_ph = {'this team': 'own', 'other team': 'opp', 'nobody': 'dead'}.get((row.get('ball') or '').strip())
    if ph in ('own', 'opp', 'dead') and coach_ph:
        add(pre+'phase', 1 if ph == coach_ph else -1, 1)
    d1 = eng.get(D1)
    if d1:
        cp, cc = _yes(row.get('pressure')), _yes(row.get('cover'))
        add(pre+'pressure', 1 if d1.get('intermediatePressure') else -1, cp)
        add(pre+'cover', 1 if d1.get('intermediateCover') else -1, cc)
        for b in BANDS:
            coach = None if cp is None else (-1 if cp == -1 or (cc == -1 and b != 'youth') else 1)
            add(pre+f'D1.{b}', d1.get(b) if d1.get(b) in (1, -1) else None, coach)
    for det, q in ((D2, 'farside'), (D5, 'line'), (D6, 'spread')):
        e = eng.get(det)
        if e:
            add(pre+f'{S_CODE[det]}.intermediate', e.get('intermediate') if e.get('intermediate') in (1, -1) else None,
                _yes(row.get(q)))
    e = eng.get(D4)
    if e:
        for b in BANDS:
            add(pre+f'D4.{b}', e.get(b) if e.get(b) in (1, -1) else None, _yes(row.get(f'behind_{b}')))
    for b in BANDS:
        fails = [x.get(b) for x in eng.values() if x.get(b) in (1, -1)]
        if fails:
            add(pre+f'raise.{b}', -1 if -1 in fails else 1,
                {'yes': -1, 'no': 1}.get((row.get(f'raise_{b}') or '').strip()))


def score_labels(labels_dir, sheets_dir=None, keys_dir=None):
    """The success criteria a coach's labels can measure (DESIGN 13.3/13.4), from the filled sheets in labels_dir and
    the keys in sheets_dir/key (default labels_dir), or for MOMENT, ROLE and ADVANCE the rekeyed ones in keys_dir
    (rekey_keys). Refuses (SystemExit) a key whose sha differs from its manifest (the sheets were regenerated after
    labelling), a rekeyed key of other items than the sheet's, and, since fix round 1 (critic N1 / B-9), any key whose
    frozen block is not the code's as it is now: a score always measures the rules it is stamped with. Items in
    SEEN_SPANS are scored apart ('seen', critic N3). -> dict; 'not measured' wherever a sheet is missing or too thin."""
    labels_dir = Path(labels_dir)
    sheets_dir = Path(sheets_dir or labels_dir)
    manifest = json.loads((sheets_dir/'manifest.json').read_text(encoding='ascii'))
    rekey = None
    if keys_dir is not None:
        rekey = json.loads((Path(keys_dir)/'rekey_manifest.json').read_text(encoding='ascii'))
    out = {'labels': str(labels_dir), 'sheets': str(sheets_dir), 'keys': str(keys_dir) if keys_dir else None,
           'scored': time.strftime('%Y-%m-%d %H:%M:%S'),
           'frozen': frozen_state(), 'labelFiles': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                                     for p in sorted(labels_dir.glob('*_labels.*'))},
           'criteria': {}, 'sets': {}}

    def key_of(name):
        rel = f'key/{name}_key.json'
        text = (sheets_dir/rel).read_bytes()
        if hashlib.sha256(text).hexdigest() != manifest['files'][rel]:
            raise SystemExit(f'{rel} differs from the manifest: the key changed after the sheets were made')
        doc = json.loads(text.decode('ascii'))
        if rekey is not None and name in REKEY_SETS:
            path = Path(keys_dir)/f'{name}_key.json'
            text2 = path.read_bytes()
            if hashlib.sha256(text2).hexdigest() != rekey['files'].get(f'{name}_key.json'):
                raise SystemExit(f'{path} differs from rekey_manifest.json')
            doc2 = json.loads(text2.decode('ascii'))
            if doc2['sheetSha'] != doc['sheetSha'] or sorted(k['id'] for k in doc2['keys']) != sorted(
                    k['id'] for k in doc['keys']):
                raise SystemExit(f'{path}: rekeyed for other items than the {name} sheet')
            doc = doc2
        bad = frozen_diff(doc.get('frozen'))
        if bad:
            raise SystemExit(f'{name} key: computed with other rules than the code being scored ({", ".join(bad[:6])}'
                             f'{" ..." if len(bad) > 6 else ""}): rekey after the freeze (critic N1)')
        return {k['id']: k for k in doc['keys']}
    # MOMENT
    path = labels_dir/'moment_labels.csv'
    if path.is_file():
        lab, key = _read_csv(path), key_of('moment')
        res = {'labelled': 0, 'bySplit': {}}
        for split in ('tune', 'holdout', 'seen', 'all'):
            acc = {}
            for iid, row in lab.items():
                k = key.get(iid)
                if k is None:
                    continue
                res['labelled'] += split == 'all'
                seen = _seen_dets(k['run'], k['time'])

                def add(name, engine, coach, k=k, seen=seen):
                    if engine is None or coach is None:
                        return
                    sp = 'seen' if _comparison_detector(name) in seen else _base_split(k)
                    if split != 'all' and sp != split:
                        return
                    a = acc.setdefault(name, {'n': 0, 'agree': 0, 'engineFail': 0, 'bothFail': 0, 'coachFail': 0})
                    a['n'] += 1
                    a['agree'] += int(engine == coach)
                    a['engineFail'] += int(engine == -1)
                    a['coachFail'] += int(coach == -1)
                    a['bothFail'] += int(engine == -1 and coach == -1)
                for run, engine in sorted(k['engineByRun'].items()):
                    _moment_compare(engine, row, add, 'coach:' if '+' in run else '')
                st = k.get('engineStratum') or k['stratum']
                if st['kind'] == 'outcome' and st['outcome'] == 'demonstrated' and st.get('evidence'):
                    q = {D1: 'cover', D2: 'farside', D4: 'behind_intermediate', D5: 'line', D6: 'spread'}.get(
                        st['detector'])
                    if q:
                        add(f'strength.{S_CODE[st["detector"]]}', 1, _yes(row.get(q)))
            sm = [v for n, v in acc.items() if n.startswith('strength.')]
            if sm:
                acc['strengthMoments'] = {'n': sum(v['n'] for v in sm), 'agree': sum(v['agree'] for v in sm),
                                          'engineFail': 0, 'bothFail': 0, 'coachFail': sum(v['coachFail'] for v in sm)}
            for a in acc.values():
                a['agreement'] = _rate(a['agree'], a['n'])
                a['precisionOfEngineFail'] = _rate(a['bothFail'], a['engineFail'])
                a['recallOfCoachFail'] = _rate(a['bothFail'], a['coachFail'])
            res['bySplit'][split] = acc
        out['sets']['moment'] = res
        allm = res['bySplit']['all']
        sm = allm.get('strengthMoments')
        out['criteria']['11 strength moments rated positive >= .8'] = (
            {'value': sm['agreement'], 'n': sm['n'], 'pass': sm['agreement'] >= .8} if sm and sm['n'] >= 20 else
            'not measured (fewer than 20 labelled evidence moments PitchIQ judged demonstrated)')
        for det in (D1, D2, D4, D5, D6):
            name = f'{S_CODE[det]}.intermediate'
            d = res['bySplit']['holdout'].get(name)
            out['criteria'][f'moment {name}: precision of "not demonstrated" >= .8 (held out)'] = (
                {'value': d['precisionOfEngineFail'], 'n': d['engineFail'],
                 'pass': (d['precisionOfEngineFail'] or 0) >= .8}
                if d and d['engineFail'] >= 10 else
                'not measured (fewer than 10 held-out moments PitchIQ judged not demonstrated)')
    else:
        out['criteria']['11 strength moments rated positive >= .8'] = 'not measured (no moment_labels.csv)'
    # ROLE
    path = labels_dir/'role_labels.csv'
    if path.is_file():
        lab, key = _read_csv(path), key_of('role')
        by_source = {}
        for iid, row in lab.items():
            k = key.get(iid)
            coach = (row.get('role') or '').strip()
            if k is None or not coach or coach == 'unsure':
                continue
            recs = sorted(k['records'], key=lambda r: r['t1']-r['t0'], reverse=True)
            got = recs[0] if recs else {'role': None, 'source': 'none'}
            src = got.get('source') or 'none'
            a = by_source.setdefault(src, {'n': 0, 'exact': 0})
            a['n'] += 1
            a['exact'] += int(got.get('role') == coach)
        for a in by_source.values():
            a['agreement'] = _rate(a['exact'], a['n'])
        out['sets']['role'] = by_source
    # ADVANCE
    path = labels_dir/'advance_labels.csv'
    if path.is_file():
        lab, key = _read_csv(path), key_of('advance')
        res = {'candidates': 0, 'coachAdvances': 0, 'withD3': 0, 'engineUnsupported': 0, 'bothUnsupported': 0,
               'heldOutEngineUnsupported': 0, 'heldOutBoth': 0, 'seenEngineUnsupported': 0, 'seenBoth': 0}
        for iid, row in lab.items():
            k = key.get(iid)
            if k is None:
                continue
            res['candidates'] += 1
            if row.get('wide_defender') != 'yes' or row.get('advanced') != 'yes':
                continue
            res['coachAdvances'] += 1
            judged = [d for d in k['d3'] if d.get('intermediate') in ('demonstrated', 'not_demonstrated')]
            if not judged:
                continue
            res['withD3'] += 1
            eng_bad = any(d['intermediate'] == 'not_demonstrated' for d in judged)
            coach_bad = row.get('supported') == 'not covered'
            res['engineUnsupported'] += eng_bad
            res['bothUnsupported'] += eng_bad and coach_bad
            if D3 in _seen_dets(k['run'], (k['t0']+k['t1'])/2):
                res['seenEngineUnsupported'] += eng_bad
                res['seenBoth'] += eng_bad and coach_bad
            elif k['split'] == 'holdout':
                res['heldOutEngineUnsupported'] += eng_bad
                res['heldOutBoth'] += eng_bad and coach_bad
        res['precision'] = _rate(res['bothUnsupported'], res['engineUnsupported'])
        res['heldOutPrecision'] = _rate(res['heldOutBoth'], res['heldOutEngineUnsupported'])
        out['sets']['advance'] = res
        out['criteria']['6 held-out precision of "unsupported" >= .8'] = (
            {'value': res['heldOutPrecision'], 'n': res['heldOutEngineUnsupported'],
             'pass': (res['heldOutPrecision'] or 0) >= .8} if res['heldOutEngineUnsupported'] >= 5 else
            f'not measured ({res["heldOutEngineUnsupported"]} held-out spells PitchIQ judged unsupported; 5 needed)')
    else:
        out['criteria']['6 held-out precision of "unsupported" >= .8'] = 'not measured (no advance_labels.csv)'
    # ZONE (DESIGN 13.4 #9 is judged on held-out items; the tuning items, mostly Metrica game 1, are shown apart)
    path = labels_dir/'zone_labels.json'
    if path.is_file():
        doc, key = json.loads(path.read_text(encoding='utf-8')), key_of('zone')
        rpath = labels_dir/'zone_reveal_labels.json'
        rv = (json.loads(rpath.read_text(encoding='utf-8')).get('answers') or {}) if rpath.is_file() else {}
        res = {}
        def zsplit(k):
            return 'seen' if k['detector'] in _seen_dets(k['source'], k['t']) else _base_split(k)
        for split in ('holdout', 'tune', 'seen', 'all'):
            n = inside = agree = 0
            for iid, a in (doc.get('answers') or {}).items():
                k = key.get(iid)
                poly = a.get('polygon') or []
                sp = None if k is None else zsplit(k)
                if k is None or len(poly) < 3 or (split != 'all' and sp != split):
                    continue
                n += 1
                inside += int(_inside(k['zone']['polygon'], _centroid(poly)))
                agree += int(a.get('mover') == k['mover'])
            rated = [v.get('pl') for iid, v in rv.items() if v.get('pl') in ('yes', 'no') and iid in key and
                     (split == 'all' or zsplit(key[iid]) == split)]
            res[split] = {'drawn': n, 'centroidInside': _rate(inside, n), 'sameMover': _rate(agree, n),
                          'rated': len(rated), 'plausible': _rate(sum(x == 'yes' for x in rated), len(rated))}
        out['sets']['zone'] = res
        h = res['holdout']
        out['criteria']['9 coach centroid inside PitchIQ zone >= .7 (held out)'] = (
            {'value': h['centroidInside'], 'n': h['drawn'], 'pass': (h['centroidInside'] or 0) >= .7}
            if h['drawn'] >= 10 else f'not measured ({h["drawn"]} held-out areas drawn; 10 needed; all items: '
                                     f'{res["all"]["centroidInside"]} of {res["all"]["drawn"]})')
        out['criteria']['9 PitchIQ zone rated plausible >= .8 (held out)'] = (
            {'value': h['plausible'], 'n': h['rated'], 'pass': (h['plausible'] or 0) >= .8}
            if h['rated'] >= 10 else f'not measured ({h["rated"]} held-out reveal ratings; 10 needed; all items: '
                                     f'{res["all"]["plausible"]} of {res["all"]["rated"]})')
    else:
        out['criteria']['9 coach centroid inside PitchIQ zone >= .7 (held out)'] = 'not measured (no zone_labels.json)'
    return out


S_CODE = {D1: 'D1', D2: 'D2', D3: 'D3', D4: 'D4', D5: 'D5', D6: 'D6'}


def sheet_sources(reports_dir, remeasure=False):
    """The label-sheet sources of a `runs` output folder (summary.json + each run's obs/roles and result): the
    broadcast clips BC60 (held out; its 42.4-60 s overlap BC180's start and are marked 'overlap'), BC180 (tuned on,
    DESIGN 13.3), BC303 (held out), AM and VEO (new footage, held out; their synthetic-coach variants only add
    opportunities, the spacing rule keeps one item per time)."""
    reports_dir = Path(reports_dir)
    summary = json.loads((reports_dir/'summary.json').read_text(encoding='ascii'))
    out = []
    for name, e in summary['runs'].items():
        result = json.loads(Path(e['result']).read_text(encoding='utf-8'))
        teams = {}
        if remeasure:
            # fix round 1 (B-9 rekey): the tiers measured again with the code as it is now (a runs folder may predate
            # the freeze); a +coach variant with the spans it was measured with (coach_spans_SYNTHETIC.json)
            sp = None
            if e.get('coachSpansSynthetic'):
                sp = json.loads((reports_dir/name/'coach_spans_SYNTHETIC.json').read_text(encoding='ascii'))
                sp = {k: v for k, v in sp.items() if k != 'note'}
            _, _, o2, r2, _ = measure_run(e['result'], spans=sp)
            teams = {team: (o2[team], r2[team]) for team in ('A', 'B')}
        for team in ('A', 'B'):
            if remeasure:
                continue
            obs = json.loads((reports_dir/name/team/'obs.json').read_text(encoding='ascii'))
            roles = json.loads((reports_dir/name/team/'roles.json').read_text(encoding='ascii'))
            teams[team] = (obs, roles)
        obs = teams['A'][0]
        base = name.split('+')[0]
        split = 'tune' if base == 'BC180' else 'holdout'
        out.append({'run': name, 'video': obs['video']['name'], 'footage': obs['timeline']['footage']['kind'],
                    'split': split, 'segment': (obs['segment']['t0'], obs['segment']['t1']),
                    'index': frame_index(result), 'teams': teams})
        del result
    # BC60's last 18 s are BC180's first: items there are not held out
    bc180 = next((s for s in out if s['run'] == 'BC180'), None)
    if bc180 is not None:
        for s in out:
            if s['run'] == 'BC60':
                s['overlap'] = (bc180['segment'][0], s['segment'][1])
    return out


# ---------- command line ----------
def _pairs(items, what):
    out = {}
    for it in items or ():
        name, sep, path = it.partition('=')
        if not sep or not name or not path:
            raise SystemExit(f'{what}: NAME=PATH, got {it!r}')
        out[name] = path
    return out


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ['--labels']:
        argv = ['score']+argv[1:]
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    sc = sub.add_parser('scenes', help='the synthetic counterexamples through the real engine at levels 1-7')
    sc.add_argument('keys', nargs='*')
    sc.add_argument('--json')
    lv = sub.add_parser('levels', help='DESIGN 13.2 on the scenes (and on --run copies)')
    lv.add_argument('keys', nargs='*', help='scene[/variant] (default: every scene variant)')
    lv.add_argument('--run', action='append')
    lv.add_argument('--json')
    rr = sub.add_parser('runs', help='real-run reports at every level into --out')
    rr.add_argument('--run', action='append', required=True)
    rr.add_argument('--spans', action='append', help='NAME=possession labels json: a synthetic coach-marked phase')
    rr.add_argument('--out', required=True)
    sm = sub.add_parser('summary', help='the markdown summary table of a `runs` output folder')
    sm.add_argument('--reports', required=True)
    sm.add_argument('--scenes', help='the JSON of `scenes --json`')
    sm.add_argument('--levels', help='the JSON of `levels --json`')
    sm.add_argument('--statsbomb', help='the JSON of `statsbomb --json`')
    sm.add_argument('--out', required=True)
    sh = sub.add_parser('sheets', help='write the blind label sheets (DESIGN 13.3) from a `runs` output folder')
    sh.add_argument('--reports', required=True)
    sh.add_argument('--out', required=True, action='append', help='may repeat: the same sheets into several folders')
    sh.add_argument('--zone-extra', help='a JSON {"items": [...]} of extra ZONE candidates (e.g. Metrica game 1)')
    sh.add_argument('--folder', action='append', help='the video folder each --out page plays from, in order (default '
                                                     '../../../../uploads/, the uploads folder seen from '
                                                     'tools/annotations/coaching/labels/)')
    rk = sub.add_parser('rekey', help='PitchIQ\'s side of the MOMENT / ROLE / ADVANCE sheets recomputed with the code '
                                      'as it is now, for the same items (after the freeze: critic N1)')
    rk.add_argument('--sheets', required=True, help='the folder with manifest.json and key/')
    rk.add_argument('--reports', required=True, help='the `runs` output folder the sheets were made from')
    rk.add_argument('--out', required=True)
    so = sub.add_parser('score', help='score filled label sheets (also: --labels DIR)')
    so.add_argument('labels')
    so.add_argument('--sheets', help='the folder with manifest.json and key/ (default: the labels folder)')
    so.add_argument('--keys', help='the folder `rekey` wrote (MOMENT, ROLE and ADVANCE keys of the frozen rules)')
    so.add_argument('--json')
    sb = sub.add_parser('statsbomb', help='pressure distance on StatsBomb 360 data (run with python -I -B)')
    sb.add_argument('data')
    sb.add_argument('--json')
    a = ap.parse_args(argv)
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')
    if a.cmd == 'scenes':
        t = time.perf_counter()
        rows = run_scenes(a.keys)
        tot = {'pass': 0, 'fail': 0, 'n/a': 0, 'vacuous': 0, 'known': 0, 'fixed': 0}
        for r in rows:
            for c in r['checks']:
                tot[c['status']] += 1
            bad = [c for c in r['checks'] if c['status'] in ('fail', 'known', 'fixed')]
            print(f"{r['scene']:<36} checks {sum(c['status'] == 'pass' for c in r['checks'])}/"
                  f"{sum(c['status'] in ('pass', 'fail') for c in r['checks'])} invariants {len(r['invariants'])}")
            for c in bad:
                print(f"   {c['status'].upper()} L{c['level']}{'/'+c['style'] if c['style'] else ''} {c['expect']} {c['detectors']} "
                      f"{c['value']}: {json.dumps(c['detail'])[:300]}")
            for p in r['invariants'][:6]:
                print(f"   INVARIANT {p['level']} {p['criterion']}: {p['problem'][:200]}")
        print(f'total {tot} in {time.perf_counter()-t:.1f} s')
        if a.json:
            write_json(a.json, {'scenes': rows, 'total': tot})
        return 1 if tot['fail'] or tot['fixed'] or any(r['invariants'] for r in rows) else 0
    if a.cmd == 'levels':
        t = time.perf_counter()
        out = {'scenes': {}, 'runs': {}}
        for key, res in levels_scenes(a.keys or None).items():
            out['scenes'][key] = levels_digest(res)
        for name, path in _pairs(a.run, '--run').items():
            result, sha, obs, roles, _ = measure_run(path)
            for team in ('A', 'B'):
                def remeasure(settings, team=team, path=path, loaded=(result, sha)):
                    _, _, o2, r2, _ = measure_run(path, settings=settings, loaded=loaded)
                    return o2[team], r2[team]
                res = levels_check(obs[team], roles[team], run_settings(), team, remeasure=remeasure,
                                   remeasured=REMEASURED, run=re.sub(r'[^A-Za-z0-9_.-]', '-', name))
                out['runs'][f'{name}/{team}'] = levels_digest(res)
        bad = 0
        for group in ('scenes', 'runs'):
            for key, d in out[group].items():
                real = [p for p in d['problems'] if not p['kind'].startswith('known:')]
                bad += len(real)
                print(f"{key:<36} profiles {len(d['profiles'])} problems {len(real)} known "
                      f"{len(d['problems'])-len(real)} max level-pair Jaccard {d['maxJaccard']}")
                for p in real[:6]:
                    print(f"   {p['kind']}: {p['problem'][:200]}")
        print(f'{bad} problems in {time.perf_counter()-t:.1f} s')
        if a.json:
            write_json(a.json, out)
        return 1 if bad else 0
    if a.cmd == 'summary':
        def load(path):
            return json.loads(Path(path).read_text(encoding='ascii')) if path else None
        text = runs_markdown(a.reports, scenes=load(a.scenes), levels=load(a.levels), statsbomb=load(a.statsbomb))
        print(f'{write_text(a.out, text)}: {len(text.splitlines())} lines')
        return 0
    if a.cmd == 'sheets':
        extra = json.loads(Path(a.zone_extra).read_text(encoding='utf-8'))['items'] if a.zone_extra else ()
        srcs = sheet_sources(a.reports)
        folders = list(a.folder or ())
        for i, out in enumerate(a.out):
            folder = folders[i] if i < len(folders) else '../../../../uploads/'
            man = make_sheets(srcs, out, zone_extra=extra, folder=folder)
            print(f'{out}: {man["counts"]}; by footage {man["countsByFootage"]}; zones {man["zoneSources"]}')
        return 0
    if a.cmd == 'rekey':
        res = rekey_keys(a.sheets, sheet_sources(a.reports, remeasure=True), a.out)
        print(f'{a.out}: {json.dumps(res["changed"])}; not rekeyed: {res["notRekeyed"]}')
        return 0
    if a.cmd == 'score':
        res = score_labels(a.labels, a.sheets, a.keys)
        for k, v in res['criteria'].items():
            print(f'{k}: {v}')
        if a.json:
            write_json(a.json, res)
        return 0
    if a.cmd == 'statsbomb':
        res = statsbomb_check(a.data)
        for name, sub_ in res['subsets'].items():
            t5 = sub_['thresholds'][f'{PRESSURE_M:g}']
            print(f"{name}: {sub_['events']} events; pressured n {sub_['pressured']['n']} median "
                  f"{sub_['pressured'].get('p50')} m, not pressured n {sub_['notPressured']['n']} median "
                  f"{sub_['notPressured'].get('p50')} m; <= {PRESSURE_M:g} m: {t5['pressuredWithin']} vs "
                  f"{t5['notPressuredWithin']} (precision {t5['precision']}, recall {t5['recall']}); best "
                  f"{sub_['bestThreshold']['metres']:g} m (J {sub_['bestThreshold']['youdenJ']})")
        print(res['source'])
        if a.json:
            write_json(a.json, res)
        return 0
    if a.cmd == 'runs':
        runs = _pairs(a.run, '--run')
        summary = run_reports(runs, a.out, spans=_pairs(a.spans, '--spans'))
        probs = [p for e in summary['runs'].values() for t in e['teams'].values() for p in t['problems']]
        bad = sum(1 for p in probs if not p['kind'].startswith('known:'))
        print(f'{len(summary["runs"])} run variants, {bad} problems, {len(probs)-bad} known open defects; summary in '
              f'{Path(a.out)/"summary.json"} and SUMMARY.md')
        return 1 if bad else 0
    return 2


if __name__ == '__main__':
    sys.exit(main())
