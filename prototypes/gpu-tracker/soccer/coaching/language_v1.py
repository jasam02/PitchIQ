"""DEPRECATED (coaching v2, DESIGN v2 11.2): the v1 report's words, moved here unchanged from language.py on
Oct 22 (TEXT wave 5). explain.py replaces them for the v2 report; they stay only while the server serves the
v1 report (wave 7 switches it), and language.py forwards their names so old callers keep working. The fact
check (validate_text, facts_for, wording_problems) stays in language.py and is imported here.

The original module notes follow.

The words of the coaching report: templates per recommendation key x classification x context level, filled only
from an observation's own fields, and a fact check that every number, time, formation and player in a text comes
from those fields.

Why templates and a fact check, not free text: the report's reader acts on it, and "a visually attractive report
containing incorrect coaching advice is worse than a smaller report with trustworthy observations". So
- every slot is computed from the observation, its evidence or the linked drill record; a sentence whose slot has no
  value is left out, never filled with a default, and a finding whose first sentence cannot be filled is not shown;
- validate_text(text, facts) checks any engine's output: each number (at its shown precision), percentage, mm:ss.ss
  time, formation label and player reference must be derivable from the facts (facts_for: the record's numbers, shares
  as percentages, durations, seconds as minutes, absolute values). The default TemplateEngine passes it by
  construction; a future model-written engine (TextEngine protocol) is held to the same check by report.py;
- wording_problems(text) rejects exaggeration, blame, claims about intent ("chose", "deliberately") and any phase of
  play the tracking cannot see: the result has no possession record (team_shape's formation.phases.available is
  False), so no finding says possession, pass, pressing or turnover. Every compactness and spacing finding carries the
  phase hedge (PHASE_HEDGE: it matters most when the opponents have the ball, which the tracking does not record) and
  the video moments are where the coach checks the phase;
- counts carry their measured time and the professional count over the same measured time. Below
  SHORT_SEGMENT_SECONDS (3 min) of measured play nothing is called repeated or recurring, one occurrence is "one
  observed instance", and a potential improvement says that a short count can vary by chance;
- a strength says "maintained ..." and states the team's median beside the professional median of the same context
  (the strength rule, detectors.strength_check: median on the favourable side of the professional p50 and little time
  beyond the professional p75), never "excellent" and never "within the professional upper level": the first
  reports said that for broadcast 0-60 s team A's length (17.8 m against a professional median of 31.5 m), which hid
  how far below the professional range it was. A pooled observation names its strictest context comparison
  (broadcast 42-222 s team B's width: 0% of the time beyond the all-context p90, 9% beyond its own third's). A
  development-level headline names no professional yardstick, and every strength ends with a "to build on it" line;
- grouped findings (group_render): width, length and area that are all strengths are one "compact team shape"
  finding, and both unit gaps one "compact distances between the lines" finding (broadcast 42-222 s team B had five
  strengths describing one compact block, with one "why it matters" sentence printed three times);
- a neutral note says which state it is in (detectors.neutral_state): no spell beyond the level, one spell (with its
  times and peak: one observed instance), spells within the professional count, or measured too briefly. The first
  reports used one sentence ("no strength or improvement met the evidence rules") for all three;
- the run's data-quality limits (small players, no pitch keyframes, a camera that shows a few players per team) are
  stated once, with what would help (data_quality), and each not-assessed metric gets one short line with its own
  reasons (limitation). The first amateur 118-182 s team B report repeated "no pitch keyframes were set" 9 times in
  11 limitation lines (734 words); now it says it once, in 389 words. The Veo clip gets one block (12 limitation
  lines and 570 words before, 148 words now); over the 24 reports of the four runs the text fell from 28,300 to
  20,200 words while every strength gained its professional median and a "to build on it" line;
- a lateral lag is "measured in continuous stretches of at least" the window length (broadcast 42-222 s team B: 70 s
  measured, longest stretch 25 s), and a lag of at most one 5 Hz sample step (the lag resolution, 0.2 s) reads "at
  about the same time";
- context levels (development: plain words, "typical" for the median, no standard deviation; competitive; advanced:
  tactical terms) change the words only: every level states the same numbers;
- players appear by their internal id (A-07, never the display label): display labels are renumbered by
  corrections (soccer/labels.py), so the page (coaching.js labelNodes) and the PDF (pdf.py say) show each id as the
  current display label in one pass (resolve_players here), and an id no longer in the roster is marked as such
  rather than left bare, where it could read as the renumbered player who now has that label;
- plain printable ASCII only (the PDF writer uses base-14 Helvetica with WinAnsi); tests encode every template.
"""
import math
import re
from dataclasses import dataclass, field
from typing import Optional

from soccer.coaching.language import (  # noqa: F401  (the fact check and its tokens: used by the v1 code)
    SHORT_SEGMENT_SECONDS, TONE_BANNED, PHASE_BANNED, SHORT_BANNED, NUMBER_WORDS, SMALL_NUMBER_WORDS, TIME_KEYS,
    TIME_TOKEN, ID_TOKEN, ID_RE, FORMATION_TOKEN, FORMATION_RE, PERCENT_TOKEN, NUMBER_TOKEN, SLOT, PRINTABLE,
    NOT_IN_ROSTER, fmt_time, fmt_duration, fmt_percent, fmt_rate, player_token, resolve_players, TEXT_KEYS,
    resolve_report, Facts, facts_for, _walk, _close, validate_text, wording_problems, TextEngine)

LEVELS = ('development', 'competitive', 'advanced')
DEFAULT_LEVEL = 'competitive'
CLASSIFICATIONS = ('strength', 'improvement', 'review', 'neutral', 'insufficient')
FINDING_CLASSES = ('strength', 'improvement', 'review')
LABELS = {'observed_strength': 'Observed strength', 'review_opportunity': 'Review opportunity',
          'potential_improvement': 'Potential improvement', 'high_confidence_concern': 'High-confidence concern'}
DEFAULT_SEVERITY = {'strength': 'observed_strength', 'improvement': 'potential_improvement',
                    'review': 'review_opportunity'}
MAX_TIMES = 3                   # occurrence times quoted in a sentence; more read as "... and later"
MAX_PLAYERS = 5                 # player references in a sentence; more read as "... and others"
MAX_CLIPS = 3                   # evidence moments in the VIDEO line (evidence.py selects 1-3 per observation)
MAX_CLIP_PLAYERS = 3            # players named per moment in the VIDEO line; more read "... and others"
                                # (broadcast 0-60 s team A's back four lost A-10 without a word)
MAX_LIMIT_REASONS = 3           # reasons on one not-assessed line (the first amateur lines listed up to 10)
LAG_SAME_SECONDS = .2           # s: a lateral lag of at most one 5 Hz sample step (the lag resolution) reads "at about
                                # the same time" (broadcast 42-222 s team B: the engine's -0.2 s was 0.0 s in an
                                # independent re-measurement)
KEYFRAME_ADVICE = (2, 3)        # pitch keyframes the data-quality note suggests (the calibration audit: two or three
                                # landmark frames anchor a keyless registration); report.py adds them to its facts
PRO_STRENGTH_SHARE = .25        # the professional share of time beyond its own p75 (by construction of the
                                # quantile), the yardstick of the strength rule (detectors.STRENGTH_PRO_SHARE); the text
                                # says "a quarter", never a number

# Metric -> unit. Metres are estimates while the pitch size is assumed (tactical.pitchSize.source 'standard').
UNITS = {'width': 'm', 'depth': 'm', 'hull': 'm2', 'spread': 'm', 'gapDM': 'm', 'gapMF': 'm', 'lineStd': 'm',
         'lineHeight': 'm', 'nearestTeammate': 'm', 'lagDM': 's', 'corrDM': 'ratio'}

D_M = {'development': ('the defenders', 'the midfielders'), 'competitive': ('the back line', 'midfield'),
       'advanced': ('the defensive line', 'the first midfield line')}
M_F = {'development': ('the midfielders', 'the forwards'), 'competitive': ('midfield', 'the front line'),
       'advanced': ('the midfield line', 'the forward line')}

# Pattern vocabulary: what the detector's pattern code measures, in each level's words. 'units' are the (first,
# second) unit of a two-unit pattern; 'measure' follows a possessive ("Team A's ..."), 'short' names the measure in a
# headline, 'strengthName' is what a strength maintained, 'compare' the (favourable, unfavourable) comparative against
# the professional median; 'definition' a string or a dict per level.
PATTERNS = {
    'space_between_defence_and_midfield': {
        'key': 'compact_lines', 'metric': 'gapDM', 'units': D_M,
        'noun': {'development': 'the space between the defenders and the midfielders',
                 'competitive': 'the distance between the back line and midfield',
                 'advanced': 'the vertical gap between the defensive line and the first midfield line'},
        'measure': {'development': 'space between the defenders and the midfielders',
                    'competitive': 'distance from the back line to midfield',
                    'advanced': 'gap from the defensive line to the first midfield line'},
        'short': {'development': 'space', 'competitive': 'distance', 'advanced': 'gap'},
        'topic': {'development': 'Space between defenders and midfielders',
                  'competitive': 'Back line to midfield distance', 'advanced': 'Defence to midfield line gap'},
        'compare': ('shorter', 'longer'),
        'definition': {'development': 'It is the distance along the pitch between the average positions of both '
                                      'groups.',
                       'competitive': 'It is measured between the average positions of both units along the length '
                                      'of the pitch.',
                       'advanced': 'It is measured between the average positions of both units along the length of '
                                   'the pitch.'}},
    'space_between_midfield_and_forwards': {
        'key': 'compact_lines', 'metric': 'gapMF', 'units': M_F,
        'noun': {'development': 'the space between the midfielders and the forwards',
                 'competitive': 'the distance between midfield and the front line',
                 'advanced': 'the vertical gap between the midfield and forward lines'},
        'measure': {'development': 'space between the midfielders and the forwards',
                    'competitive': 'distance from midfield to the front line',
                    'advanced': 'gap from the midfield line to the forward line'},
        'short': {'development': 'space', 'competitive': 'distance', 'advanced': 'gap'},
        'topic': {'development': 'Space between midfielders and forwards',
                  'competitive': 'Midfield to front line distance', 'advanced': 'Midfield to forward line gap'},
        'compare': ('shorter', 'longer'),
        'definition': {'development': 'It is the distance along the pitch between the average positions of both '
                                      'groups.',
                       'competitive': 'It is measured between the average positions of both units along the length '
                                      'of the pitch.',
                       'advanced': 'It is measured between the average positions of both units along the length of '
                                   'the pitch.'}},
    'team_width': {
        'key': 'compact_shape', 'metric': 'width',
        'noun': {'development': 'the width of the team', 'competitive': "the team's width",
                 'advanced': "the team's width (horizontal compactness)"},
        'measure': {'development': 'width', 'competitive': 'width', 'advanced': 'width'},
        'short': {'development': 'width', 'competitive': 'width', 'advanced': 'width'},
        'topic': {'development': 'Team width', 'competitive': 'Team width', 'advanced': 'Horizontal compactness'},
        'stretch': {'development': 'spread out wide', 'competitive': 'stretched wide',
                    'advanced': 'stretched horizontally'},
        'strengthName': {'development': 'a narrow team shape', 'competitive': 'a narrow team shape',
                         'advanced': 'horizontal compactness'},
        'compare': ('narrower', 'wider'),
        'definition': {'development': 'Width is the distance across the pitch from the widest player on one side to '
                                      'the widest on the other.',
                       'competitive': 'Width is measured across the pitch between the outermost outfield players.',
                       'advanced': 'Width is measured across the pitch between the outermost outfield players.'}},
    'team_depth': {
        'key': 'compact_shape', 'metric': 'depth',
        'noun': {'development': 'the length of the team from back to front',
                 'competitive': "the team's length from back to front",
                 'advanced': "the team's length (vertical compactness)"},
        'measure': {'development': 'length from back to front', 'competitive': 'length from back to front',
                    'advanced': 'length'},
        'short': {'development': 'length', 'competitive': 'length', 'advanced': 'length'},
        'topic': {'development': 'Team length', 'competitive': 'Team length', 'advanced': 'Vertical compactness'},
        'stretch': {'development': 'stretched out from back to front', 'competitive': 'stretched long',
                    'advanced': 'stretched vertically'},
        'strengthName': {'development': 'a short team from back to front',
                         'competitive': 'a short team from back to front', 'advanced': 'vertical compactness'},
        'compare': ('shorter', 'longer'),
        'definition': {'development': 'Length is the distance from the deepest outfield player to the most advanced '
                                      'one.',
                       'competitive': 'Length is measured along the pitch between the deepest and the most advanced '
                                      'outfield players.',
                       'advanced': 'Length is measured along the pitch between the deepest and the most advanced '
                                   'outfield players.'}},
    'team_area': {
        'key': 'compact_shape', 'metric': 'hull',
        'noun': {'development': 'the area the team covered', 'competitive': "the area covered by the team's shape",
                 'advanced': "the team's surface area (convex hull)"},
        'measure': {'development': 'area', 'competitive': 'area', 'advanced': 'surface area'},
        'short': {'development': 'area', 'competitive': 'area', 'advanced': 'area'},
        'topic': {'development': 'Area covered by the team', 'competitive': 'Team area',
                  'advanced': 'Area compactness'},
        'stretch': {'development': 'spread out over a large area', 'competitive': 'covered a large area',
                    'advanced': 'expanded its surface area'},
        'strengthName': {'development': 'a small team area', 'competitive': 'a compact team area',
                         'advanced': 'a compact surface area'},
        'compare': ('smaller', 'larger'),
        'definition': {'development': 'The area is the space inside a line drawn around all the outfield players.',
                       'competitive': 'The area is that of the smallest shape enclosing all outfield players.',
                       'advanced': 'The area is that of the smallest shape enclosing all outfield players.'}},
    'uneven_back_line': {
        'key': 'back_line_coordination', 'metric': 'lineStd', 'units': D_M,
        'noun': {'development': 'the spread of the defenders from front to back',
                 'competitive': 'the spread of the back line from front to back',
                 'advanced': "the back line's depth staggering (spread of the defenders' depth)"},
        'measure': {'development': 'difference in depth between the defenders',
                    'competitive': 'back-line spread from front to back', 'advanced': 'back-line depth spread'},
        'short': {'development': 'spread', 'competitive': 'spread', 'advanced': 'depth spread'},
        'topic': {'development': 'Defenders level with each other', 'competitive': 'Back-line evenness',
                  'advanced': 'Back-line coordination'},
        'compare': ('more even', 'less even'),
        'definition': {'development': 'It shows how far the defenders stand from each other front to back; one '
                                      'defender out of line makes it bigger.',
                       'competitive': "The spread is the typical front-to-back difference between the defenders' "
                                      "distances from their own goal, so a single defender out of line raises it.",
                       'advanced': "The spread is the standard deviation of the defenders' distances from their own "
                                   "goal, so a single defender out of line raises it."}},
    'high_line_with_gap': {
        'key': 'line_height_gap', 'metric': 'gapDM', 'units': D_M,
        'noun': {'development': 'the space between the defenders and the midfielders',
                 'competitive': 'the distance between the back line and midfield',
                 'advanced': 'the gap between the defensive line and the first midfield line'},
        'topic': {'development': 'High defenders with space in front', 'competitive': 'Line height and gap',
                  'advanced': 'Line height against the gap to midfield'}},
    'formation_stable': {
        'key': 'formation_organisation', 'metric': None,
        'topic': {'development': "The team's shape", 'competitive': 'Team organisation',
                  'advanced': 'Structural organisation'}},
    'formation_change': {
        'key': 'formation_organisation', 'metric': None,
        'topic': {'development': "The team's shape", 'competitive': 'Team organisation',
                  'advanced': 'Structural organisation'},
        'formationHeadline': {'development': "the team's shape changed", 'competitive': 'the organisation changed',
                              'advanced': 'structural changes'}},
    'formation_ambiguous': {
        'key': 'formation_organisation', 'metric': None,
        'topic': {'development': "The team's shape", 'competitive': 'Team organisation',
                  'advanced': 'Structural organisation'},
        'formationHeadline': {'development': "the team's shape was hard to read",
                              'competitive': 'the organisation was not clear', 'advanced': 'ambiguous structure'}},
    'lateral_lag': {
        'key': 'lateral_shift', 'metric': 'lagDM',
        'units': {'development': ('the defenders', 'the midfielders'), 'competitive': ('the back line', 'midfield'),
                  'advanced': ('the defensive line', 'the midfield line')},
        'noun': {'development': 'how the defenders and midfielders moved across together',
                 'competitive': 'the sideways shift of the back line and midfield',
                 'advanced': 'the lateral shift of the defensive and midfield lines'},
        'topic': {'development': 'Moving across together', 'competitive': 'Sideways shift of back line and midfield',
                  'advanced': 'Lateral shift of the lines'},
        'definition': {'development': 'The delay is the time between the midfielders moving sideways and the '
                                      'defenders following.',
                       'competitive': 'The delay is the time shift that best lines up the sideways movements of both '
                                      'lines.',
                       'advanced': 'The delay is the time shift that best lines up the sideways movements of both '
                                   'lines.'}},
    'player_isolation': {
        'key': 'support_distances', 'metric': 'nearestTeammate',
        'noun': {'development': 'the distance to the nearest team-mate', 'competitive': 'the support distance',
                 'advanced': 'the support distance (nearest team-mate)'},
        'measure': {'development': 'distance to the nearest team-mate',
                    'competitive': 'distance to the nearest team-mate', 'advanced': 'support distance'},
        'short': {'development': 'distance', 'competitive': 'distance', 'advanced': 'support distance'},
        'topic': {'development': 'Distance to team-mates', 'competitive': 'Support distances',
                  'advanced': 'Support distances'},
        'compare': ('closer', 'further')},
    'ball_location': {
        'key': 'ball_context', 'metric': None,
        'topic': {'development': 'Where the ball was', 'competitive': 'Ball location',
                  'advanced': 'Ball location'}},
}

# Detector registry (contract section 5) -> the recommendation key, category and default pattern this module words.
DETECTORS = {
    'compactness': ('compact_shape', 'spacing', 'team_width'),
    'unit_gap_dm': ('compact_lines', 'spacing', 'space_between_defence_and_midfield'),
    'unit_gap_mf': ('compact_lines', 'spacing', 'space_between_midfield_and_forwards'),
    'back_line_coordination': ('back_line_coordination', 'defensive_line', 'uneven_back_line'),
    'line_height_and_gap': ('line_height_gap', 'defensive_line', 'high_line_with_gap'),
    'formation_stability': ('formation_organisation', 'formation', 'formation_stable'),
    'lateral_shift': ('lateral_shift', 'movement', 'lateral_lag'),
    'isolation': ('support_distances', 'support', 'player_isolation'),
    'ball_context': ('ball_context', 'ball_context', 'ball_location'),
}

KEY_TOPIC = {key: PATTERNS[pat]['topic'] for key, _, pat in DETECTORS.values()}

# The phase hedge of every compactness and spacing finding: the result has no possession record, and a stretched
# shape is normal with the ball (broadcast 42-222 s: team A, 55.5 m wide with its back line 54.6 m from goal while team
# B's stood 27.9 m from its own, was most likely the team with the ball).
PHASE_HEDGE = {
    'development': 'This matters most when the other team has the ball. The tracking does not record who had the '
                   'ball, so {checkWhere}.',
    'competitive': 'This matters most when the opponents have the ball; the tracking does not record who had it, so '
                   '{checkWhere}.',
    'advanced': 'This matters most when the opponents have the ball; the tracking does not record who had it, so '
                '{checkWhere}.',
}
HEDGED_KEYS = ('compact_shape', 'compact_lines', 'back_line_coordination', 'lateral_shift')

# The measured sentences of a strength (compact shape, compact lines, back line, support distance): the lead (the
# median against the professional median of the same context), for a pooled observation the strictest context
# comparison, the definition, then the share of time beyond the professional p75 (the strength rule's level).
LEAD = {
    'development': "Over {measured} that could be measured{context}, {owner}'s typical {measure} was {median}, "
                   "against {proMedian} for a typical {proOne}.",
    'competitive': "Over {measured} of measured play{context}, {owner}'s {measure} had a median of {median}, "
                   "against a professional median of {proMedian}.",
    'advanced': "Over {measured} of measured play{context}, {owner}'s {measure} had a median of {median} "
                "(professional median {proMedian}).",
}
LEAD_PLAIN = {      # a record without the professional median (never from detectors.strength_check): no comparison
    'development': "Over {measured} that could be measured{context}, {owner}'s typical {measure} was {median}.",
    'competitive': "Over {measured} of measured play{context}, {owner}'s {measure} had a median of {median}.",
    'advanced': "Over {measured} of measured play{context}, {owner}'s {measure} had a median of {median}.",
}
LEAD_POOLED = {
    'development': "Over {measured} that could be measured, {owner}'s typical {measure} was {pooledMedian}.",
    'competitive': "Over {measured} of measured play, {owner}'s {measure} had a median of {pooledMedian}.",
    'advanced': "Over {measured} of measured play, {owner}'s {measure} had a median of {pooledMedian}.",
}
COMPARISON = {
    'development': 'Looking only at the moments {strictContext} ({comparisonSeconds}), it was typically {median}, '
                   'against {proMedian} for a typical {proOne} there.',
    'competitive': 'The strictest comparison is the moments {strictContext} ({comparisonSeconds}): a median of '
                   '{median} against a professional median of {proMedian}.',
    'advanced': 'Strictest context, {strictContext} ({comparisonSeconds}): median {median} against a professional '
                'median of {proMedian}.',
}
BEYOND = {
    'development': 'It went past {proUpper} for {shareUpper} of {thatTime}; {proMany} go past it about a quarter '
                   'of the time.',
    'competitive': 'It was beyond {proUpper}, a level {proMany} exceed about a quarter of the time, for {shareUpper} '
                   'of {thatTime}.',
    'advanced': 'Time beyond the professional upper quartile ({proUpper}): {shareUpper} of {thatTime}, against a '
                'quarter for {proMany}.',
}
STRENGTH_HAPPENED = ['{lead}', '{comparison}', '{definition}', '{beyond}']

# Counted findings: the measured count against the professional count over the same measured time (the improvement
# rule, detectors.classify: more than the professional window p99 count).
RATE = 'That is {rate} times per measured minute; in the professional reference this happens {proRate}.'
WINDOW = '{proMany} rarely showed more than {proCount} such spells over the same measured time.'
HIGHLIGHT = ('It is marked as a high-confidence concern because the measurement was reliable and the count was clearly '
             'above the professional range.')
POSITIONS_ONLY = "This reading comes from the positions only; it has not been checked against a coach's view."

# Templates: TEMPLATES[recommendationKey][classification] with 'headline', 'whyItMatters', 'whatToImprove' and
# 'priority' per level, 'whatHappened' (sentences, shared by every level or per level), rendered in order, the first
# 'required' of which must fill, and 'byPattern' overrides of any field for one pattern. A strength's whatToImprove is
# its "to build on it" line.
TEMPLATES = {
    'compact_lines': {
        'improvement': {
            'headline': {'development': 'Space opened up between {unitA} and {unitB}',
                         'competitive': 'Large distance between {unitA} and {unitB}',
                         'advanced': 'Gap between {unitA} and {unitB} beyond the reference range'},
            'whatHappened': ['In {measured} of measured play{context}, {noun} went beyond {threshold} '
                             '{countPhrase}.', '{definition}', 'The largest was {peak} at {peakTime}.', '{window}',
                             RATE, '{highlight}', 'Players involved: {players}.', '{censored}', '{shortCaution}'],
            'whyItMatters': {
                'development': 'When {unitA} and {unitB} are far apart, there is space between them where opponents '
                               'can get the ball and turn, and it is harder for team-mates to help each other. '
                               '{hedge}',
                'competitive': 'A large space between the lines may let opponents receive and turn there, and it '
                               'lengthens the distances over which {unitA} and {unitB} support each other. {hedge}',
                'advanced': 'Without vertical compactness the space between the lines may open for opponents to '
                            'receive facing forward, and the distances for cover and support between {unitA} and '
                            '{unitB} grow. {hedge}'},
            'whatToImprove': {
                'development': 'Start with the basics: the team moves up and back together. When {unitA} move, '
                               '{unitB} move the same way, so the space between them stays small. Check the video '
                               'moments to see which team had the ball.',
                'competitive': 'Work on {unitA} and {unitB} moving up and back as one unit, so the distance between '
                               'them stays short when the ball moves. Review the listed moments first: if the '
                               'opponents had the ball, these are the situations to coach.',
                'advanced': 'Coach the distance between the lines as a shared reference: when {unitB} steps out or '
                            'drops, {unitA} adjusts its height to keep vertical compactness. Confirm the phase of '
                            'play on the listed moments before drawing conclusions.'},
            'priority': {'development': 'Keep {unitA} and {unitB} closer together',
                         'competitive': 'Shorten the distance between {unitA} and {unitB}',
                         'advanced': 'Restore vertical compactness between {unitA} and {unitB}'}},
        'strength': {
            'headline': {'development': 'Maintained a short distance between {unitA} and {unitB}',
                         'competitive': 'Maintained a short distance between {unitA} and {unitB}: median {median}, '
                                        'professional median {proMedian}',
                         'advanced': 'Maintained vertical compactness between {unitA} and {unitB} (median {median} '
                                     'against {proMedian})'},
            'whatHappened': STRENGTH_HAPPENED,
            'whyItMatters': {
                'development': 'Staying close together means team-mates can help each other quickly, and there is '
                               'less space between them for opponents. {hedge}',
                'competitive': 'Short distances between the lines keep support close and leave less room between '
                               'them. {hedge}',
                'advanced': 'Vertical compactness limits the space between the lines and keeps the distances for '
                            'cover and support short. {hedge}'},
            'whatToImprove': {
                'development': 'To build on it: keep the groups this close when the team moves up and back, also '
                               'when the ball goes over the top.',
                'competitive': 'To build on it: keep the lines this close as the team steps up and drops, also after '
                               'long balls, and check on the video that it holds when the opponents have the ball.',
                'advanced': 'To build on it: hold the vertical distances while the line steps and drops at match '
                            'tempo, including after balls played in behind.'}}},
    'compact_shape': {
        'improvement': {
            'headline': {'development': 'The team {stretch}', 'competitive': 'Team shape {stretch}',
                         'advanced': 'Team shape {stretch} beyond the reference range'},
            'whatHappened': ['In {measured} of measured play{context}, {noun} went beyond {threshold} '
                             '{countPhrase}.', '{definition}', 'The largest was {peak} at {peakTime}.', '{window}',
                             RATE, '{highlight}', '{censored}', '{shortCaution}'],
            'whyItMatters': {
                'development': 'When the team is spread out, players are further apart, so there are bigger spaces '
                               'between them and it takes longer to help each other. Spreading out can be right when '
                               'your team has the ball. {hedge}',
                'competitive': 'A stretched shape may leave gaps between players and between the units; with the '
                               'ball, a wider or longer shape can be useful. {hedge}',
                'advanced': 'When defending, a stretched shape increases the distances between units and players and '
                            'may open spaces to play through; with the ball it can be a way to create space. {hedge}'},
            'whatToImprove': {
                'development': 'Practise staying close to the nearest team-mates when the other team has the ball, '
                               'and moving together towards the ball.',
                'competitive': 'Work on the team becoming compact again quickly when the opponents have the ball: '
                               'the far-side players move in and the lines stay close.',
                'advanced': 'Coach the compaction cues: far-side players narrow towards the ball side and the lines '
                            'close vertically as soon as the opponents have the ball.'},
            'priority': {'development': 'Stay closer together as a team',
                         'competitive': "Keep the team's shape compact",
                         'advanced': 'Improve {topicLower}'}},
        'strength': {
            'headline': {'development': 'Maintained {strengthName}',
                         'competitive': 'Maintained {strengthName}: median {short} {median}, professional median '
                                        '{proMedian}',
                         'advanced': 'Maintained {strengthName}: median {short} {median} against a professional '
                                     'median of {proMedian}'},
            'whatHappened': STRENGTH_HAPPENED,
            'whyItMatters': {
                'development': 'A compact team leaves less space between players and makes it easier to help each '
                               'other. {hedge}',
                'competitive': 'A compact shape keeps the distances between players short, so support and cover are '
                               'close. {hedge}',
                'advanced': 'Compactness reduces the space available between and around the units and keeps cover '
                            'distances short. {hedge}'},
            'whatToImprove': {
                'development': 'To build on it: keep moving together as a group, and check on the video that the '
                               'team stays this close when the other team has the ball.',
                'competitive': 'To build on it: keep these distances when the ball is switched quickly from side to '
                               'side, and check on the video that the shape stays this compact when the opponents '
                               'have the ball.',
                'advanced': 'To build on it: test the compactness against quicker switches of play and check that '
                            'the far-side players keep narrowing when the opponents have the ball.'}}},
    'back_line_coordination': {
        'improvement': {
            'headline': {'development': 'Defenders at different depths', 'competitive': 'Back line uneven',
                         'advanced': 'Staggered back line beyond the reference range'},
            'whatHappened': ['In {measured} in which the whole back line was measured{context}, {noun} went beyond '
                             '{threshold} {countPhrase}.', '{definition}', 'The largest was {peak} at {peakTime}.',
                             '{window}', RATE, '{highlight}', 'Defenders involved: {players}.', '{censored}',
                             '{shortCaution}'],
            'whyItMatters': {
                'development': 'When one defender is much deeper or higher than the others, it is harder for the '
                               'defenders to move up and back together, and space can open behind or between them. '
                               '{hedge}',
                'competitive': 'An uneven line is harder to move as a unit, and a defender out of line may leave '
                               'space behind or between the defenders. {hedge}',
                'advanced': 'A staggered line complicates collective stepping and dropping, can leave space in behind '
                            'and between the defenders, and makes the line a weaker reference for the midfield. '
                            '{hedge}'},
            'whatToImprove': {
                'development': "First, the defenders look across at each other and stay level. One defender calls "
                               "'up' or 'back' and everyone moves together.",
                'competitive': 'Work on one organiser setting the height of the line and the other defenders '
                               'adjusting across to stay level, stepping up and dropping together.',
                'advanced': 'Coach collective line cues from one organiser (step up when the ball is under pressure, '
                            'drop when it is not), so depth adjustments are made by the whole line, not by '
                            'individuals.'},
            'priority': {'development': 'Keep the defenders level with each other',
                         'competitive': 'Keep the back line level and moving together',
                         'advanced': "Coordinate the back line's stepping and dropping"}},
        'strength': {
            'headline': {'development': 'Maintained a level line of defenders',
                         'competitive': 'Maintained an even back line: median spread {median}, professional median '
                                        '{proMedian}',
                         'advanced': 'Maintained a coordinated back line (depth spread {median} against a '
                                     'professional median of {proMedian})'},
            'whatHappened': ['{leadLine}', '{comparison}', '{definition}', '{beyond}'],
            'whyItMatters': {
                'development': 'A level line of defenders can move up and back together and is easier to organise. '
                               '{hedge}',
                'competitive': 'An even back line is easier to move as a unit and leaves fewer gaps behind it. '
                               '{hedge}',
                'advanced': 'An even line keeps collective stepping and dropping coherent and gives the midfield a '
                            'clear reference. {hedge}'},
            'whatToImprove': {
                'development': 'To build on it: keep looking across at each other and moving together, also when the '
                               'ball is close to your own goal.',
                'competitive': "To build on it: keep stepping up and dropping together on one organiser's call, also "
                               "when the ball is near your own box.",
                'advanced': "To build on it: keep the line's stepping and dropping collective when the ball is near "
                            "the box and under pressure."}}},
    'line_height_gap': {
        'review': {
            'headline': {'development': 'To review: defenders high up the pitch with space in front of them',
                         'competitive': 'To review: high back line with a large gap to midfield',
                         'advanced': 'To review: high line height combined with a large gap to the first midfield '
                                     'line'},
            'whatHappened': ["{UnitA} stood {height} from the team's own goal while the distance to {unitB} reached "
                             "{gap}{context}.", '{occurrence}', 'Players in the back line: {players}.',
                             '{censored}'],
            'whyItMatters': {
                'development': 'Defenders standing high up the pitch is not a problem by itself. With a big space '
                               'between them and the midfielders, though, opponents may find room there, so it is '
                               'worth a look.',
                'competitive': 'A high line is not a problem in itself. Combined with a large gap to midfield it may '
                               'leave space between the lines, so it is worth checking on the video.',
                'advanced': 'A high line height is a valid choice in many game models; combined with a large gap to '
                            'the first midfield line it may leave the space between the lines unprotected. Its value '
                            'depends on the pressure on the ball, which the tracking does not record.'},
            'whatToImprove': {
                'development': 'To review with the players: was there space between the defenders and the '
                               'midfielders, and could they have been closer together?',
                'competitive': 'To review: was the space between the lines covered at these moments, and could '
                               'midfield have dropped or the back line held a little deeper?',
                'advanced': 'To review: was the ball under pressure when the line stood this high, and was the space '
                            'between the lines protected by the midfield?'},
            'priority': {'development': 'Review: defenders high with space in front of them',
                         'competitive': 'Review the high line together with the gap to midfield',
                         'advanced': 'Review line height against the gap to the first midfield line'}}},
    'formation_organisation': {
        'strength': {
            'headline': {'development': "Maintained a stable team shape ({label})",
                         'competitive': 'Maintained a stable {label} organisation',
                         'advanced': 'Maintained a stable {label} structure'},
            'whatHappened': ['Over {measured} of measured play, the tracked positions read as {label} '
                             '({labelConfidence}).', 'The next closest reading was {alternative}.', '{roles}',
                             '{sameBackLine}', POSITIONS_ONLY],
            'whyItMatters': {
                'development': 'When everyone knows their position, the team keeps its distances and players know '
                               'where to help.',
                'competitive': 'A stable organisation keeps distances predictable and makes it clear who covers '
                               'whom.',
                'advanced': 'A stable structure keeps role relationships and distances predictable, which supports '
                            'collective movement.'},
            'whatToImprove': {
                'development': 'To build on it: practise moving into a second, simple shape on a call, keeping the '
                               'same distances.',
                'competitive': 'To build on it: rehearse the change to an alternative shape on a call, with the same '
                               'distances between the lines.',
                'advanced': 'To build on it: rehearse planned rotations from this structure and check that cover and '
                            'balance are kept.'}},
        'review': {
            'headline': {'development': 'To review: {formationHeadline}',
                         'competitive': 'To review: {formationHeadline}',
                         'advanced': 'To review: {formationHeadline}'},
            'whatHappened': ['{reading}', '{roles}', 'The group of players forming the back line changed '
                             '{setChanges} times between complete readings, with {distinctSets} different groups.',
                             POSITIONS_ONLY],
            'whyItMatters': {
                'development': 'Changing shape is not wrong, but when it is not planned players may be unsure where '
                               'to stand and whom to help.',
                'competitive': 'A change of shape is not wrong in itself, but one the players did not plan can make '
                               'distances and responsibilities less clear, so it is worth checking.',
                'advanced': 'Fluid structures can be part of a game model, but unplanned role changes blur the '
                            'responsibilities for cover and balance; the video shows whether the changes were '
                            'organised.'},
            'whatToImprove': {
                'development': 'To review: watch the moments with the players and ask where each of them should '
                               'stand. Shadow play without opponents is a simple way to practise positions.',
                'competitive': "To review: check on the video which shape the team meant to play and whether the "
                               "players held it. If not, rehearse the team's shape and each player's position "
                               "relative to the ball and team-mates.",
                'advanced': 'To review: separate planned rotations from unplanned role changes, then rehearse the '
                            'structure and its triggers in shadow play.'},
            # shared by a change and an ambiguous reading, so it names no change
            'priority': {'development': "Review where each player stands in the team's shape",
                         'competitive': "Review the team's organisation",
                         'advanced': "Review the structure and the players' roles in it"},
            'byPattern': {'formation_ambiguous': {'whyItMatters': {
                'development': 'When the shape is hard to read, players may be unsure of their positions; the video '
                               'shows whether the positions were clear to them.',
                'competitive': 'A shape that reads as unrelated formations at the same time may mean the players were '
                               'not sure of their positions, or that the team mixed shapes; the video shows which.',
                'advanced': 'An ambiguous structure between unrelated families may be a deliberate hybrid or unclear '
                            'roles; the listed moments show which, and whether cover and balance were kept.'}}}}},
    'lateral_shift': {
        'improvement': {
            'headline': {'development': 'The defenders moved across later than the midfielders',
                         'competitive': 'Back line shifted across later than midfield',
                         'advanced': 'Delayed lateral shift of the defensive line relative to midfield'},
            'whatHappened': ['In {measured} of measurement{stretches}{context}, {unitA} moved sideways {lagPhrase} '
                             '{unitB} on average.', '{definition}', '{resolution}',
                             'In the professional reference this delay is rarely more than {threshold}.',
                             '{occurrence}', '{window}', '{highlight}', '{censored}', '{shortCaution}'],
            'whyItMatters': {
                'development': 'When the defenders move across later than the midfielders, gaps can open between '
                               'them on the side where the ball is. {hedge}',
                'competitive': 'If the back line shifts later than midfield, gaps may open between and inside the '
                               'lines on the ball side. {hedge}',
                'advanced': 'A delayed shift of the defensive line breaks the horizontal and vertical distances to '
                            'midfield as the ball moves across, which may open channels between the lines on the '
                            'ball side. {hedge}'},
            'whatToImprove': {
                'development': 'Practise moving across together: when the ball goes from one side to the other, '
                               'everyone slides the same way at the same time.',
                'competitive': 'Work on the back line and midfield sliding across together as the ball moves, with '
                               'the players nearest the ball moving first and the others following straight away.',
                'advanced': 'Coach a synchronised lateral shift: both lines slide towards the ball side together, '
                            'keeping their horizontal and vertical distances, while the far-side players narrow.'},
            'priority': {'development': 'Move across together as the ball moves',
                         'competitive': 'Shift across together with midfield',
                         'advanced': 'Synchronise the lateral shift of both lines'}},
        'strength': {
            'headline': {'development': 'Maintained moving across together',
                         'competitive': 'Maintained a coordinated sideways shift',
                         'advanced': 'Maintained a synchronised lateral shift'},
            'whatHappened': ['Over {measured} of measurement{stretches}{context}, {unitA} moved sideways '
                             '{lagPhrase} {unitB} on average.', '{definition}', '{resolution}',
                             '{proLagSentence}'],
            'whyItMatters': {
                'development': "Moving across together keeps the team's distances the same wherever the ball goes. "
                               "{hedge}",
                'competitive': 'Shifting together keeps the distances between and within the lines intact as the '
                               'ball moves. {hedge}',
                'advanced': 'A synchronised shift preserves the horizontal and vertical distances of the unit as play '
                            'moves across. {hedge}'},
            'whatToImprove': {
                'development': 'To build on it: keep sliding across together, also when the ball moves quickly from '
                               'one side to the other.',
                'competitive': 'To build on it: keep shifting together with midfield when the ball is switched '
                               'quickly, with the far-side players narrowing.',
                'advanced': 'To build on it: keep the shift synchronised against faster switches of play and check '
                            'the far-side narrowing.'}}},
    'support_distances': {
        'improvement': {
            'headline': {'development': '{player} was far from the nearest team-mate',
                         'competitive': '{player} separated from the nearest team-mate',
                         'advanced': '{player}: support distance beyond the reference range'},
            'whatHappened': ['In {measured} in which the area around {player} was in camera view{context}, the '
                             'distance to the nearest team-mate went beyond {threshold} {countPhrase}.',
                             'The largest distance was {peak} at {peakTime}.', '{window}', RATE, '{highlight}',
                             '{censored}', '{shortCaution}'],
            'whyItMatters': {
                'development': 'A player who is far from team-mates is harder to help and has fewer team-mates '
                               'nearby to play with. For some positions, such as a forward, being further away can '
                               'be part of the job.',
                'competitive': "Long distances to the nearest team-mate reduce the support and cover around the "
                               "player. For a forward or a player holding the width this can be part of the role, so "
                               "check it against the player's task.",
                'advanced': 'Large support distances reduce the options and cover around the player; for a player '
                            'with a width or depth-holding role this may be part of the task, so judge it against '
                            'the role.'},
            'whatToImprove': {
                'development': 'Help {player} and the nearest team-mates stay a short distance from each other, so '
                               'someone is close enough to help.',
                'competitive': 'Work on {player} and the nearest team-mates adjusting their distances as the ball '
                               'moves, so support is close enough when it is needed.',
                'advanced': "Coach the support distances and angles around {player}'s position, so the nearest "
                            "team-mates stay within supporting range as play develops."},
            'priority': {'development': 'Keep {player} close to team-mates',
                         'competitive': 'Shorten the support distances around {player}',
                         'advanced': 'Improve the support distances around {player}'}},
        # A player far from team-mates more than the professional players of his unit over the same time is a role
        # check, never an improvement (detectors.detect_isolation): a winger holding the width does exactly this.
        'review': {
            'headline': {'development': 'To review: {player} far from team-mates (check the role)',
                         'competitive': 'To review: distance between {player} and the nearest team-mate (role check)',
                         'advanced': 'To review: support distance around {player} (role check)'},
            'whatHappened': ['In {measured} in which the area around {player} was in camera view{context}, the '
                             'distance to the nearest team-mate went beyond {threshold} {countPhrase}.',
                             'The largest distance was {peak} at {peakTime}.', '{window}', '{censored}'],
            'whyItMatters': {
                'development': 'A player far from team-mates is harder to help. For some positions, such as a winger '
                               'staying wide or a forward staying high, this is the job itself, so first check what '
                               'the player was asked to do.',
                'competitive': "Long distances to the nearest team-mate reduce the support and cover around the "
                               "player. For a forward, or a player told to hold the width, this can be the role "
                               "itself, so check it against the player's task before treating it as a problem.",
                'advanced': 'Large support distances reduce options and cover around the player, but a width- or '
                            'depth-holding role produces exactly this, so judge it against the role first.'},
            'whatToImprove': {
                'development': "To review: watch the moments and ask whether {player} was where the position asks. "
                               "If not, help the nearest team-mates stay closer.",
                'competitive': "To review: check on the video whether {player}'s position matched the role; if it "
                               "did not, work on the nearest team-mates adjusting their distances as the ball moves.",
                'advanced': "To review: compare {player}'s positions with the role's reference points; if they do not "
                            "match, coach the support distances and angles around that position."},
            'priority': {'development': "Check {player}'s position and distance to team-mates",
                         'competitive': "Check {player}'s role and support distances",
                         'advanced': "Check {player}'s role against the support distances"}},
        'strength': {
            'headline': {'development': 'Maintained short distances around {player}',
                         'competitive': 'Maintained close support around {player}',
                         'advanced': 'Maintained support distances around {player}'},
            'whatHappened': STRENGTH_HAPPENED,
            'whyItMatters': {
                'development': 'Team-mates close by can help quickly.',
                'competitive': 'Close support gives the player options and cover nearby.',
                'advanced': 'Short support distances keep options and cover available around the player.'},
            'whatToImprove': {
                'development': 'To build on it: keep staying close to team-mates as the ball moves.',
                'competitive': 'To build on it: keep these support distances as the ball moves from side to side.',
                'advanced': 'To build on it: keep these support distances and angles at match tempo.'}}},
}

# Findings grouped into one (report.py): several strengths that describe the same compact block.
GROUPS = {
    'compact_team_shape': {
        'key': 'compact_shape', 'patterns': ('team_width', 'team_depth', 'team_area'), 'min': 2,
        'headline': {'development': 'Maintained a compact team shape',
                     'competitive': 'Maintained a compact team shape: {members} below the professional median',
                     'advanced': 'Maintained a compact block: {members} below the professional median'},
        'lead': {'development': 'Over {measured} that could be measured{context}, {team} stayed compact.',
                 'competitive': 'Over {measured} of measured play{context}, {team} kept a compact shape.',
                 'advanced': 'Over {measured} of measured play{context}, {team} stayed compact.'}},
    'compact_lines_both': {
        'key': 'compact_lines', 'min': 2,
        'patterns': ('space_between_defence_and_midfield', 'space_between_midfield_and_forwards'),
        'headline': {'development': 'Maintained short distances between the lines',
                     'competitive': 'Maintained compact distances between the lines: both gaps below the '
                                    'professional median',
                     'advanced': 'Maintained vertical compactness from the defensive line to the forward line'},
        'lead': {'development': 'Over {measured} that could be measured{context}, {team} kept its groups close '
                                'together.',
                 'competitive': 'Over {measured} of measured play{context}, {team} kept its lines close together.',
                 'advanced': 'Over {measured} of measured play{context}, {team} stayed vertically compact.'}},
}
MEMBER = {
    'development': '{Measure}: typically {median}, against {proMedian} for a typical professional team; above '
                   '{proUpper} for {shareUpper} of the time.',
    'competitive': '{Measure}: median {median} against a professional median of {proMedian}; beyond {proUpper} for '
                   '{shareUpper} of the time.',
    'advanced': '{Measure}: median {median} (professional median {proMedian}), {shareUpper} of the time beyond the '
                'upper quartile ({proUpper}).',
}
MEMBER_POOLED = {
    'development': '{Measure}: typically {pooledMedian} over {measured}; looking only at the moments {strictContext} '
                   '({comparisonSeconds}), {median} against {proMedian} for a typical professional team, and above '
                   '{proUpper} for {shareUpper} of those moments.',
    'competitive': '{Measure}: median {pooledMedian} over {measured}; in the strictest comparison, {strictContext} '
                   '({comparisonSeconds}), median {median} against a professional median of {proMedian}, beyond '
                   '{proUpper} for {shareUpper} of that time.',
    'advanced': '{Measure}: median {pooledMedian} over {measured}; strictest context {strictContext} '
                '({comparisonSeconds}): median {median} against {proMedian}, {shareUpper} beyond the upper quartile '
                '({proUpper}).',
}
GROUP_CLOSE = {
    'development': 'Professional teams go above those levels about a quarter of the time.',
    'competitive': 'Professional teams go beyond each of those levels about a quarter of the time.',
    'advanced': 'Professional teams spend a quarter of their time beyond the upper quartile.',
}

# Formation families (team_shape.RELATED): one shape read in different ways, with what tells them apart.
RELATED_SHAPES = {
    frozenset(('4-3-3', '4-1-4-1')): 'both have the same back line and one holding midfielder in front of it; they '
                                     'differ only in how high the other central midfielders and the wide players '
                                     'stand',
    frozenset(('3-5-2', '5-3-2')): 'both have central defenders with wing-backs outside them; they differ only in '
                                   'whether the wing-backs stand level with midfield or drop into the back line',
    frozenset(('3-4-3', '5-4-1')): 'they are the same structure with the wide players higher or deeper: in the deeper '
                                   'reading the wing-backs join the back line and the wide forwards drop into '
                                   'midfield',
    frozenset(('4-4-2', '4-2-3-1')): 'both have the same back line and a pair of central midfielders; they differ '
                                     'only in whether the second forward plays beside the striker or just behind him, '
                                     'between the wide players',
}

# One-sentence notes for measured observations without a finding (neutral), by state (detectors.neutral_state).
NEUTRAL = {
    'metric': {
        'development': '{Topic}: typically {median} over {measured}{context}{versus}.',
        'competitive': '{Topic}: median {median} over {measured}{context}{versus}.',
        'advanced': '{Topic}: median {median} over {measured}{context}{versus}.'},
    'metricPooled': {
        'development': '{Topic}: typically {pooledMedian} over {measured}; looking only at the moments '
                       '{strictContext} ({comparisonSeconds}), {median}{versus}.',
        'competitive': '{Topic}: median {pooledMedian} over {measured}; {strictContext} ({comparisonSeconds}) the '
                       'median was {median}{versus}.',
        'advanced': '{Topic}: median {pooledMedian} over {measured}; {strictContext} ({comparisonSeconds}) median '
                    '{median}{versus}.'},
    'absent': 'No spell beyond {threshold}.',
    'one_spell': ('One spell beyond {threshold}, from {spellFrom} to {spellTo} (peak {spellPeak}): one observed '
                  'instance, so it says nothing about the team\'s play in general.'),
    'within_range': '{count} spells beyond {threshold}, within what {proMany} showed over the same measured time.',
    'one_team': ('{count} spells beyond {threshold}; the professional reference for this has one team only, so the '
                 'count is not compared.'),
    'under_measured': 'Measured too briefly to call it a strength.',
    'evidence_bound': 'Too few independent moments in this time to call it a strength.',
    'context_fails': 'It did not meet the strength rule in every third of the pitch it was measured in.',
    'wide': ('A {stretchWord} shape is typical for a team with the ball; without it, it leaves space between players. '
             'Check the phase on the video.'),
    'formation': {
        'development': '{Topic}: read as {label} ({labelConfidence}) over {measured} of play, from the tracked '
                       'positions only.',
        'competitive': '{Topic}: the tracked positions read as {label} ({labelConfidence}) over {measured} of measured '
                       'play.',
        'advanced': '{Topic}: the tracked positions read as {label} ({labelConfidence}) over {measured} of measured '
                    'play.'},
    'formationThroughout': {
        'development': '{Topic}: read as {label} ({labelConfidence}) throughout the {measured} measured, from the '
                       'tracked positions only.',
        'competitive': '{Topic}: read as {label} ({labelConfidence}) throughout the {measured} measured, from the '
                       'tracked positions only.',
        'advanced': '{Topic}: read as {label} ({labelConfidence}) throughout the {measured} measured, from the '
                    'tracked positions only.'},
    'formationRelated': ('{Topic}: read as {label}, with {alternative} almost as close: {pairNote}. Read from the '
                         'tracked positions over {measured}; not a finding, so no drill.'),
    'formationRelatedChange': ('{Topic}: read as {labelList} at different times: {pairNote}, so this is one shape read '
                               'in different ways, not a change of shape ({measured} measured).'),
    'formationShort': 'A shape is only called a strength over at least {strengthSeconds} of measured play.',
    'lag': '{Topic}: measured for {measured}{stretches}; {unitA} moved sideways {lagPhrase} {unitB} on average.',
    'lagSpell': ('One stretch with a delay beyond {threshold}, from {spellFrom} to {spellTo}: one observed instance, '
                 'so it says nothing about the team\'s play in general.'),
    'support': {
        'development': ('{Topic}: each player was typically {median} from the nearest team-mate over {measured}; '
                        'nobody was far from team-mates for longer than professional players in the same position '
                        'are.'),
        'competitive': ('{Topic}: the distance to the nearest team-mate had a median of {median} over {measured}; no '
                        'player stood out against professional players of the same unit over the same measured '
                        'time.'),
        'advanced': ('{Topic}: median support distance {median} over {measured}; no player stood out against '
                     'professional players of the same unit over the same measured time.')},
    'ball_context': ("Ball location, for context, over {measured} with a reliable ball position: {team}'s own "
                     "third {own}, middle third {middle}, final third {final}. This shows where the ball was, not "
                     "which team had it."),
    'block': {
        'development': ("How deep the teams stood: {team}'s defenders were typically {lineHeight} from their own goal "
                        "over {lineHeightSeconds} (a typical professional team: {proLineHeight}); the other team's "
                        "defenders ({other}) were {opponentLineHeight} from theirs over {opponentSeconds}."),
        'competitive': ("Block height: {team}'s back line stood a median {lineHeight} from its own goal over "
                        "{lineHeightSeconds} (professional median {proLineHeight}); the other team's back line "
                        "({other}) stood {opponentLineHeight} from its own goal over {opponentSeconds}."),
        'advanced': ("Block height: {team}'s defensive line stood a median {lineHeight} from its own goal over "
                     "{lineHeightSeconds} (professional median {proLineHeight}); the opponent's line ({other}) stood "
                     "{opponentLineHeight} from its own goal over {opponentSeconds}.")},
    'blockOwn': {
        'development': ("How deep the team stood: {team}'s defenders were typically {lineHeight} from their own goal "
                        "over {lineHeightSeconds} (a typical professional team: {proLineHeight})."),
        'competitive': ("Block height: {team}'s back line stood a median {lineHeight} from its own goal over "
                        "{lineHeightSeconds} (professional median {proLineHeight})."),
        'advanced': ("Block height: {team}'s defensive line stood a median {lineHeight} from its own goal over "
                     "{lineHeightSeconds} (professional median {proLineHeight}).")},
    'blockUncertain': 'The back line was read from an uncertain reading of the units, so treat these heights as rough.',
}
NEUTRAL_HEDGE_PATTERNS = ('team_width', 'team_depth', 'team_area')

# Reason codes (series.py eligibility, detectors' confidence reasons) in plain words.
REASONS = {
    'not_calibrated': 'parts of the segment had no pitch registration',
    'low_calibration': 'the pitch registration was not confident enough on part of the segment',
    'disagreement': 'the pitch registration was inconsistent on part of the segment',
    'incomplete_team': 'not every outfield player was in view',
    'incomplete_units': 'the players of the units involved were rarely all in view at the same time',
    'no_formation': 'no formation could be read from the positions',
    'no_direction': 'the playing direction was not known',
    'small_scene_veto': 'players appear too small in this footage for this measurement',
    'small_scene': 'players appear small in this footage',
    'short_measured': 'too little of the play could be measured',
    'short_stretch': 'no continuous measured stretch was long enough',
    'medium_registration': 'the pitch registration was of medium confidence',
    'low_registration': 'the pitch registration was of low confidence',
    'keyless': 'no pitch keyframes were set, so distances rely on the automatic registration',
    'identity': 'some player identities were uncertain',
    'identity_switches': 'some player identities may have switched',
    'crossings': 'players crossed each other many times, which makes identities less certain',
    'stabilizer_flags': 'some player boxes were unreliable',
    'few_players': 'only a few players of each team were in view',
    'out_of_view': 'the area around the player was out of camera view for much of the time',
    'no_reference': 'no professional reference was available for this measurement',
    'broadcast_only': 'this measurement is only made on broadcast-style footage',
    'no_ball': 'the ball position was not reliable enough',
    'low_confidence': 'the measurement confidence was low',
    'no_wording': 'no reviewed wording exists for this observation yet',
    'unverified_text': 'its wording could not be checked against the measurements',
    # series.py eligibility and detectors.py confidence codes that had no words (each was dropped silently from the
    # limitation lines, so on broadcast 42-222 s team A the decisive reason, the ambiguous formation, never showed).
    'ambiguous_formation': 'the units come from a formation reading that was ambiguous',
    'formation_unverified': 'formation readings from tracked positions are not validated, so their confidence is at '
                            'most medium',
    'excess_players': 'more outfield players of the team were tracked than it has on the pitch',
    'line_displaced': 'the back line stood ahead of most of its team (as at a set piece)',
    'line_disrupted': 'the defenders were spread too far along the pitch to form one line',
    'not_observed': 'the player was not observed',
    'no_teammate': 'no team-mate had a position',
    'unplaced_teammate': 'a team-mate in view had no usable position',
    'ball_uncertain': 'the ball position was too uncertain',
    'ball_off_pitch': 'the ball was off the pitch',
    # coaching v2 wave 1 (series.py, CONTEXT): the interleaved-units gate (critic B2; broadcast 0-60 s team B had a
    # midfielder ahead of the forwards' mean depth on 203 of 287 samples), the per-defender line offsets and the
    # velocities. The v1 report still lists series codes in its limitation lines until wave 7.
    'units_interleaved': 'players of one line stood among the next line, so the distance between the lines was not '
                         'measured',
    'not_in_line': "the player was not one of the back line's measured defenders",
    'short_track': "the player's positions were tracked too briefly to measure a speed",
}
# The words of a not-assessed line where REASONS overstates (broadcast 42-222 s team A's back line was measured 63% of
# the time: "rarely all in view" was too strong) or names a cause that was never tested (amateur: the ball had 43.0 s
# of usable positions; the cause is the small-scene rule, series.SMALL_SCENE).
LIMIT_WORDS = {
    'incomplete_units': 'the players of the lines involved were not all in view together for part of the time',
    'incomplete_team': 'not every outfield player was in view at the same time',
}
VETO_WORDS = {
    'ball_context': "the ball's location is only used on broadcast-style footage, a fixed rule for this footage type "
                    "rather than a check of the ball tracking in this clip",
    'isolation': 'support distances are only measured on broadcast-style footage, where players appear large '
                 'enough',
}
# Codes that describe the run, not one measurement: stated once by data_quality, left off the per-metric lines.
RUN_REASONS = ('keyless', 'small_scene', 'few_players', 'no_reference')
# Codes a not-assessed line does not repeat: its numbers say it ('short_measured', 'short_stretch'), or it is a
# confidence aspect rather than why the metric was not assessed ('medium_registration', 'formation_unverified').
QUIET_REASONS = ('short_measured', 'short_stretch', 'medium_registration', 'formation_unverified', 'low_confidence')
DECISIVE_REASONS = ('ambiguous_formation', 'crossings', 'identity')   # codes that alone make an aspect low: named first
ASPECT_OF = {'short_stretch': 'tracking', 'short_measured': 'tracking', 'small_scene': 'tracking',
             'stabilizer_flags': 'tracking', 'ambiguous_formation': 'tracking', 'medium_registration': 'calibration',
             'keyless': 'calibration', 'crossings': 'identity', 'identity': 'identity'}
ASPECTS = {'tracking': 'player tracking', 'calibration': 'pitch registration', 'identity': 'player identification',
           'ball': 'ball position'}


RECOMMENDATION_KEYS = tuple(sorted(set(TEMPLATES) | {'ball_context'}))


class NoWording(Exception):
    """The engine has no approved wording for this observation (or its required facts are missing): the report
    withholds it rather than inventing text."""


def fmt_value(value, unit, estimated=True, plain=False):
    """A metric value with its unit; metres carry ' (est.)' (' (estimated)' in plain words) while the pitch size is
    assumed. Professional reference values are real metres (Metrica's pitch is measured): estimated=False."""
    v = float(value)
    mark = (' (estimated)' if plain else ' (est.)') if estimated else ''
    if unit == 'm':
        return f'{v:.1f} m'+mark
    if unit == 'm2':
        return f'{v:.0f} square metres'+mark
    if unit == 's':
        return f'{v:.1f} s'
    if unit == 'ratio':
        return f'{v:.2f}'
    return f'{v:.1f}'


def team_label(team, team_name=None):
    return team_name if team_name else f'Team {team}'


def team_full(team, team_name=None):
    """First mention of the team: team letters are arbitrary per run (result.teams.meaning)."""
    return team_name if team_name else f'Team {team} (as labelled in this run)'


def title(kind):
    return 'MATCH COACHING REPORT' if kind == 'match' else 'SEGMENT COACHING REPORT'


def _join(items, more=False, word='and'):
    items = list(items)
    if more:
        return ', '.join(items)+' and later' if len(items) > 1 else items[0]+' and later'
    if len(items) <= 1:
        return ''.join(items)
    return ', '.join(items[:-1])+f' {word} '+items[-1]


def _players(ids):
    ids = [i for i in ids if isinstance(i, str)]
    if not ids:
        return None
    shown = [player_token(i) for i in ids[:MAX_PLAYERS]]
    return ', '.join(shown)+' and others' if len(ids) > MAX_PLAYERS else _join(shown)


def _cap(s):
    return s[:1].upper()+s[1:] if s and s[:1].islower() else s


def _lower(s):
    return s[:1].lower()+s[1:] if s else s


def _num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))


def _get(d, *path):
    for k in path:
        if not isinstance(d, dict):
            return None
        d = d.get(k)
    return d


def _level_text(x, level):
    """A template field that is one string for every level or a dict per level."""
    return x.get(level) if isinstance(x, dict) else x


# ---------------------------------------------------------------------------------------------------------------
# Engines

@dataclass
class FindingContext:
    """Everything an engine may use to word one observation: the record, its evidence records, the context level,
    the team (letter and optional name), the report kind, the linked drill (library record) and whether metres are
    estimates. Engines must not use anything else."""
    observation: dict
    evidence: tuple = ()
    level: str = DEFAULT_LEVEL
    team: str = 'A'
    team_name: Optional[str] = None
    kind: str = 'segment'
    drill: Optional[dict] = None
    metres_estimated: bool = True
    label: Optional[str] = None      # the label report.py settled on (it may downgrade a severity)
    run_codes: tuple = ()            # run-level reason codes the report states once (data_quality)


class TemplateEngine:
    """The default engine: TEMPLATES filled from the observation's fields."""
    name = 'template'

    def render(self, ctx):
        o = ctx.observation
        cls = o.get('classification')
        level = ctx.level if ctx.level in LEVELS else DEFAULT_LEVEL
        slots = _slots(ctx, level)
        if cls == 'insufficient':
            return {'limitation': limitation(ctx, level)}
        if cls == 'neutral':
            note = _neutral(o, slots, level)
            if not note:
                raise NoWording('no_wording')
            return {'note': note}
        key = o.get('recommendationKey')
        tpl = (TEMPLATES.get(key) or {}).get(cls)
        if tpl is None:
            raise NoWording('no_wording')
        over = (tpl.get('byPattern') or {}).get(o.get('pattern')) or {}

        def part(name):
            return over.get(name, tpl.get(name))
        happened = _level_text(part('whatHappened'), level)
        sentences = [_fill(s, slots) for s in happened]
        required = tpl.get('required', 1)
        if any(s is None for s in sentences[:required]):
            raise NoWording('no_wording')
        happened = ' '.join(s for s in sentences if s)
        headline = _fill(part('headline')[level], slots)
        if headline is None and cls == 'strength':
            headline = _fill(part('headline')['development'], slots)    # the plain headline names no number
        why = _fill(part('whyItMatters')[level], slots)
        if not headline or not why:
            raise NoWording('no_wording')
        out = {'headline': headline, 'whatHappened': happened, 'whyItMatters': why,
               'whatToImprove': None, 'howToPractice': None, 'video': _video(ctx), 'priority': None,
               'confidenceNote': _confidence_note(o, ctx.run_codes)}
        if part('whatToImprove'):
            out['whatToImprove'] = _fill(part('whatToImprove')[level], slots)
        if cls in ('improvement', 'review'):
            out['priority'] = _fill(part('priority')[level], slots)
            out['howToPractice'] = _practice(ctx.drill, cls)
        return out


def _fill(template, slots):
    """The template with its slots filled, or None when any slot has no value."""
    if template is None:
        return None
    missing = False

    def rep(m):
        nonlocal missing
        v = slots.get(m.group(1))
        if v is None:
            missing = True
            return ''
        return str(v)
    out = SLOT.sub(rep, template)
    if missing:
        return None
    return _cap(re.sub(r' {2,}', ' ', out).strip()) or None


def _pattern(o):
    pat = o.get('pattern')
    if pat in PATTERNS:
        return pat, PATTERNS[pat]
    det = DETECTORS.get(o.get('detector'))
    if det:
        return det[2], PATTERNS[det[2]]
    return None, {}


def _metric(o, pinfo):
    m = o.get('metric') or pinfo.get('metric')
    if not m:
        for occ in o.get('occurrences') or ():
            keys = list((occ.get('metrics') or {}).keys())
            if keys:
                return keys[0]
    return m


THIRD_PHRASE = {'third:own': "with the team's centre in its own third",
                'third:middle': "with the team's centre in the middle third",
                'third:final': "with the team's centre in the final third"}


def _context_phrase(ctx_name, lead=' '):
    """' with the team's centre in the middle third' for a context code ('' for 'all' or an unknown one)."""
    if not ctx_name or ctx_name == 'all':
        return ''
    if ctx_name in THIRD_PHRASE:
        return lead+THIRD_PHRASE[ctx_name]
    m = re.fullmatch(r'backline:(\d)', ctx_name)
    if m and int(m.group(1)) in SMALL_NUMBER_WORDS:
        return lead+'with a back line of '+SMALL_NUMBER_WORDS[int(m.group(1))]
    return ''


def _occurrence_sentence(occ, count, measured, short):
    if not count or not occ:
        return None
    starts = [fmt_time(x['t0']) for x in occ if _num(x.get('t0'))]
    if count == 1 and len(occ) == 1 and _num(occ[0].get('t0')) and _num(occ[0].get('t1')):
        return f'This was one observed instance, from {fmt_time(occ[0]["t0"])} to {fmt_time(occ[0]["t1"])}.'
    if count < 2 or not starts or measured is None:
        return None
    times = _join(starts[:MAX_TIMES], more=len(starts) > MAX_TIMES or count > len(starts))
    if short:
        return f'It happened {count} times in {measured} of measured play, at {times}.'
    return f'This was a repeated pattern: {count} times in {measured} of measured play, at {times}.'


def _count_phrase(occ, count, short):
    """'once, from t0 to t1 (one observed instance)', 'N times, at t, t and t', or for a long measurement
    'N times, at ...; this is a repeated pattern'. None without the occurrence records."""
    if count == 1 and len(occ) == 1 and _num(occ[0].get('t0')) and _num(occ[0].get('t1')):
        return f'once, from {fmt_time(occ[0]["t0"])} to {fmt_time(occ[0]["t1"])} (one observed instance)'
    starts = [fmt_time(x['t0']) for x in occ if _num(x.get('t0'))]
    if count < 2 or not starts:
        return None
    text = f'{count} times, at {_join(starts[:MAX_TIMES], more=len(starts) > MAX_TIMES or count > len(starts))}'
    return text if short else text+'; this is a repeated pattern'


def _lag_phrase(lag):
    if not _num(lag):
        return None
    if abs(lag) <= LAG_SAME_SECONDS+1e-9:
        return 'at about the same time as'
    return f'{abs(lag):.1f} s '+('later than' if lag >= 0 else 'earlier than')


def _slots(ctx, level):
    o = ctx.observation
    pat, pinfo = _pattern(o)
    metric = _metric(o, pinfo)
    unit = UNITS.get(metric)
    est = ctx.metres_estimated
    plain = level == 'development'
    team = team_label(ctx.team, ctx.team_name)
    s = {'team': team, 'teamFull': team_full(ctx.team, ctx.team_name),
         'span': 'this match' if ctx.kind == 'match' else 'this segment'}
    for name in ('noun', 'topic', 'stretch', 'formationHeadline', 'measure', 'short', 'strengthName'):
        v = (pinfo.get(name) or {}).get(level)
        s[name] = v
    s['definition'] = _level_text(pinfo.get('definition'), level)
    support = o.get('recommendationKey') == 'support_distances'
    s['proSubject'] = 'professional players' if support else 'professional teams'
    s['proMany'] = s['proSubject']
    s['proOne'] = 'professional player' if support else 'professional team'
    if s['topic'] is None:
        s['topic'] = (KEY_TOPIC.get(o.get('recommendationKey')) or {}).get(level)
    s['topicLower'] = _lower(s['topic']) if s['topic'] else None
    units = (pinfo.get('units') or {}).get(level)
    s['unitA'], s['unitB'] = units if units else (None, None)
    base = o.get('baseline') or {}
    thr = base.get('threshold') or {}
    metrics = o.get('metrics') or {}
    s['context'] = _context_phrase(base.get('context'))
    measured_s = o.get('measuredSeconds')
    s['measured'] = fmt_duration(measured_s) if _num(measured_s) and measured_s > 0 else None
    s['eligible'] = fmt_percent(o['eligibleShare']) if _num(o.get('eligibleShare')) else None
    s['longest'] = fmt_duration(o['longestStretchSeconds']) if _num(o.get('longestStretchSeconds')) else None
    count = o.get('occurrenceCount')
    count = int(count) if _num(count) else len(o.get('occurrences') or [])
    s['count'] = str(count)
    short = not _num(measured_s) or measured_s < SHORT_SEGMENT_SECONDS
    occ = sorted([x for x in o.get('occurrences') or [] if isinstance(x, dict)], key=lambda x: x.get('t0') or 0)
    s['occurrence'] = _occurrence_sentence(occ, count, s['measured'], short)
    s['countPhrase'] = _count_phrase(occ, count, short)
    s['censored'] = ('At least one instance was cut off by a gap in the measurement, so its full length is not known.'
                     if any(x.get('censored') or x.get('censoredStart') or x.get('censoredEnd') for x in occ)
                     else None)
    is_improvement = o.get('classification') == 'improvement'
    s['shortCaution'] = ('In a segment this short a count can vary by chance, so treat this as a point to check on '
                         'the video rather than a conclusion.') if is_improvement and short else None
    s['highlight'] = HIGHLIGHT if ctx.label == LABELS['high_confidence_concern'] else None
    val = (lambda x: fmt_value(x, unit, est, plain) if unit and _num(x) else None)
    pro = (lambda x: fmt_value(x, unit, False) if unit and _num(x) else None)     # professional: real metres
    s['threshold'] = pro(thr.get('enter'))
    s['refLevel'] = pro(thr.get('p90', thr.get('exit')))
    s['median'] = val(metrics.get('median'))
    s['pooledMedian'] = val(metrics.get('pooledMedian'))
    s['proMedian'] = pro(metrics.get('professionalP50', thr.get('p50')))
    s['proUpper'] = pro(metrics.get('professionalP75', thr.get('p75')))
    s['proHigh'] = pro(thr.get('p90'))
    share75 = metrics.get('shareAboveProfessionalP75')
    s['shareUpper'] = fmt_percent(share75) if _num(share75) else None
    s['shareAbove'] = fmt_percent(metrics['shareAboveProfessionalP90']) \
        if _num(metrics.get('shareAboveProfessionalP90')) else None
    s['proShare'] = fmt_percent(base['professionalShareAbove']) if _num(base.get('professionalShareAbove')) else None
    p99 = metrics.get('professionalWindowP99')
    s['proCount'] = (str(int(round(p99))) if abs(p99-round(p99)) < 1e-9 else f'{p99:.1f}') if _num(p99) else None
    s['window'] = _fill(WINDOW, s) if s['proCount'] is not None else None
    rate = o.get('ratePerMeasuredMinute')
    s['rate'] = fmt_rate(rate) if _num(rate) and count >= 2 else None
    rates = base.get('professionalRatePerMinute') or {}
    p50r, p90r = rates.get('p50'), rates.get('p90')
    if _num(p90r) and _num(p50r) and p50r >= .005:
        s['proRate'] = f'about {fmt_rate(p50r)} times per minute, and {fmt_rate(p90r)} at the upper end'
    elif _num(p90r):
        s['proRate'] = f'up to {fmt_rate(p90r)} times per minute'
    else:
        s['proRate'] = None
    peaks = [(x['metrics'][metric]['peak'], x) for x in occ
             if metric and _num(_get(x, 'metrics', metric, 'peak'))]
    if peaks:
        pk, px = max(peaks, key=lambda p: p[0])
        s['peak'] = val(pk)
        s['peakTime'] = fmt_time(px['peakT']) if _num(px.get('peakT')) else None
    else:
        s['peak'] = s['peakTime'] = None
    players = o.get('players') or []
    s['players'] = _players(players)
    s['player'] = player_token(players[0]) if players and isinstance(players[0], str) else None
    s['owner'] = s['player'] if support else team
    s['Owner'] = s['owner']
    # A strength's lead, its strictest-context comparison (pooled observations) and the share beyond the p75.
    pooled = bool(metrics.get('strictestContext')) and s['pooledMedian'] is not None
    strict = _context_phrase(metrics.get('strictestContext'), lead='') if pooled else ''
    s['strictContext'] = strict or None
    comp = metrics.get('comparisonSeconds')
    s['comparisonSeconds'] = fmt_duration(comp) if _num(comp) and comp > 0 else None
    if pooled and strict:
        s['context'] = ''
        s['lead'] = _fill(LEAD_POOLED[level], s)
        s['comparison'] = _fill(COMPARISON[level], s)
        s['thatTime'] = 'those moments'
    else:
        s['lead'] = _fill(LEAD[level], s) or _fill(LEAD_PLAIN[level], s)
        s['comparison'] = ''
        s['thatTime'] = 'the time'
    s['beyond'] = _fill(BEYOND[level], s)
    # back line: "in which the whole back line was measured"
    if pat == 'uneven_back_line':
        lead = s['lead']
        s['leadLine'] = lead.replace(' of measured play', ' in which the whole back line was measured', 1) \
            .replace(' that could be measured', ' in which the whole back line could be measured', 1) if lead else None
    else:
        s['leadLine'] = s['lead']
    lag = metrics.get('lag', metrics.get('median')) if metric == 'lagDM' else None
    s['lagPhrase'] = _lag_phrase(lag)
    window = metrics.get('windowSeconds') if metric == 'lagDM' else None
    s['stretches'] = f' in continuous stretches of at least {fmt_duration(window)}' if _num(window) and window > 0 \
        else ''
    res = metrics.get('lagResolutionSeconds') if metric == 'lagDM' else None
    s['resolution'] = f'Delays are measured in steps of {res:.1f} s.' if _num(res) and res > 0 else None
    p50lag = metrics.get('professionalP50') if metric == 'lagDM' else None
    pro_phrase = _lag_phrase(p50lag)
    if metric == 'lagDM' and pro_phrase and s['threshold']:
        typical = 'usually moves' if plain else 'moves, at the median,'
        s['proLagSentence'] = (f'In the professional reference the back line {typical} {pro_phrase} its midfield, '
                               f'and the delay is rarely more than {s["threshold"]}.')
    elif metric == 'lagDM' and s['threshold']:
        s['proLagSentence'] = f'In the professional reference this delay is rarely more than {s["threshold"]}.'
    else:
        s['proLagSentence'] = None
    # The phase hedge: where to check it.
    s['checkWhere'] = 'check the moments below' if ctx.evidence else 'check the phase on the video'
    s['hedge'] = _fill(PHASE_HEDGE[level], s) if o.get('recommendationKey') in HEDGED_KEYS else ''
    # Line height with gap (review): the largest-gap occurrence, else the observation's medians.
    if pat == 'high_line_with_gap':
        gaps = [x for x in occ if _num(_get(x, 'metrics', 'gapDM', 'peak'))]
        best = max(gaps, key=lambda x: x['metrics']['gapDM']['peak']) if gaps else None
        height = _get(best, 'metrics', 'lineHeight', 'mean') if best else None
        if not _num(height):
            height = metrics.get('lineHeight')
        gap = best['metrics']['gapDM']['peak'] if best else metrics.get('gapDM')
        s['height'] = fmt_value(height, 'm', est, plain) if _num(height) else None
        s['gap'] = fmt_value(gap, 'm', est, plain) if _num(gap) else None
        if count == 1 and len(occ) == 1:
            s['occurrence'] = (f'This was one observed moment, from {fmt_time(occ[0]["t0"])} to '
                               f'{fmt_time(occ[0]["t1"])}.') if _num(occ[0].get('t0')) and _num(occ[0].get('t1')) \
                else None
    # Block heights (the line-height detector's context note).
    for k in ('lineHeight', 'opponentLineHeight'):
        s[k] = fmt_value(metrics[k], 'm', est, plain) if _num(metrics.get(k)) else None
    for k, name in (('lineHeightSeconds', 'lineHeightSeconds'), ('opponentLineHeightSeconds', 'opponentSeconds')):
        s[name] = fmt_duration(metrics[k]) if _num(metrics.get(k)) and metrics[k] > 0 else None
    s['proLineHeight'] = fmt_value(metrics['professionalLineHeightP50'], 'm', False) \
        if _num(metrics.get('professionalLineHeightP50')) else None
    s['other'] = 'Team '+('B' if ctx.team == 'A' else 'A')
    # Formation.
    label = metrics.get('label')
    s['label'] = label if isinstance(label, str) and FORMATION_RE.fullmatch(label) else None
    conf = metrics.get('labelConfidence')
    s['labelConfidence'] = conf if conf in ('clear', 'likely', 'ambiguous') else None
    alt = metrics.get('alternative')
    s['alternative'] = alt if isinstance(alt, str) and FORMATION_RE.fullmatch(alt) else None
    labels = [x for x in metrics.get('labels') or [] if isinstance(x, str) and FORMATION_RE.fullmatch(x)]
    s['labelList'] = _join(labels) if len(labels) >= 2 else None
    pair = None
    if s['label'] and s['alternative']:
        pair = RELATED_SHAPES.get(frozenset((s['label'], s['alternative'])))
    if pair is None and len(labels) == 2:
        pair = RELATED_SHAPES.get(frozenset(labels))
    s['pairNote'] = pair
    s['roleStability'] = fmt_percent(metrics['roleStability']) if _num(metrics.get('roleStability')) else None
    kept, assessed = metrics.get('playersKeptRole'), metrics.get('playersAssessed')
    cut = metrics.get('roleShareThreshold')
    if _num(kept) and _num(assessed) and assessed > 0 and _num(cut):
        s['roles'] = (f'{int(kept)} of {int(assessed)} players kept one role at least {fmt_percent(cut)} of the '
                      f'time.')
    else:
        s['roles'] = None
    changes = metrics.get('defenderSetChanges')
    s['setChanges'] = str(int(changes)) if _num(changes) and changes >= 2 else None
    distinct = metrics.get('distinctDefenderSets')
    s['distinctSets'] = str(int(distinct)) if _num(distinct) and distinct >= 2 else None
    s['sameBackLine'] = 'The same players formed the back line throughout.' if _num(changes) and changes == 0 \
        else None
    secs = metrics.get('strengthSeconds') if o.get('recommendationKey') == 'formation_organisation' else None
    s['strengthSeconds'] = fmt_duration(secs) if _num(secs) and secs > 0 else None
    s['reading'] = None
    if s['measured']:
        held = metrics.get('labelSeconds') or {}
        if pat == 'formation_change' and len(labels) >= 2:
            parts = [f'{x} for {fmt_duration(held[x])}' if _num(held.get(x)) else x for x in labels]
            s['reading'] = (f'During {s["measured"]} of measured play, the tracked positions read as '
                            f'{_join(parts)}.')
        elif s['label'] and s['alternative'] and s['labelConfidence']:
            s['reading'] = (f'During {s["measured"]} of measured play, the tracked positions read most like '
                            f'{s["label"]}, with {s["alternative"]} fitting almost as well (reading: '
                            f'{s["labelConfidence"]}).')
    # Neutral states.
    spell = metrics.get('spell') or {}
    s['spellFrom'] = fmt_time(spell['t0']) if _num(spell.get('t0')) else None
    s['spellTo'] = fmt_time(spell['t1']) if _num(spell.get('t1')) else None
    s['spellPeak'] = val(spell.get('peak'))
    s['stretchWord'] = {'team_width': 'wide', 'team_depth': 'long', 'team_area': 'spread-out'}.get(pat)
    s['versus'] = _versus(pinfo, metrics.get('median'), metrics.get('professionalP50'), unit, level)
    # Ball context.
    thirds = metrics.get('thirds') or {}
    for k in ('own', 'middle', 'final'):
        s[k] = fmt_percent(thirds[k]) if _num(thirds.get(k)) else None
    for k in list(s):
        if isinstance(s[k], str) and k[:1].islower():
            s.setdefault(k[:1].upper()+k[1:], _cap(s[k]))
    return s


def _versus(pinfo, median, p50, unit, level):
    """', narrower than the professional median of 43.0 m' (', the same as ...' at the shown precision); '' without
    a comparison."""
    words = pinfo.get('compare')
    if not words or not _num(median) or not _num(p50) or not unit:
        return ''
    shown = fmt_value(p50, unit, False)
    same = fmt_value(median, unit, False) == shown
    if level == 'development':
        what = 'a typical professional team' if pinfo.get('key') != 'support_distances' else 'a typical professional'
        return f', the same as {what} ({shown})' if same else \
            f', {words[0] if median < p50 else words[1]} than {what} ({shown})'
    return f', the same as the professional median of {shown}' if same else \
        f', {words[0] if median < p50 else words[1]} than the professional median of {shown}'


def context_fails_note(metrics):
    """Why a pooled observation is no strength although its pooled numbers would be one: the thirds measured long
    enough for a strength of their own that did not pass on their own (detectors.pooled_strength; the observation's
    metrics.contexts), named ("... with the team's centre in its own third"), else NEUTRAL['context_fails']."""
    contexts = metrics.get('contexts') if isinstance(metrics.get('contexts'), dict) else {}
    failing = [THIRD_PHRASE[c] for c in ('third:own', 'third:middle', 'third:final')
               if isinstance(contexts.get(c), dict) and contexts[c].get('enoughForStrength')
               and contexts[c].get('passesStrength') is False]
    if not failing:
        return NEUTRAL['context_fails']
    return f'It did not meet the strength rule {_join(failing, word="or")}.'


def _neutral(o, s, level):
    """The note of a neutral observation, by recommendation key and state (detectors.neutral_state)."""
    key = o.get('recommendationKey')
    state = o.get('state')
    metrics = o.get('metrics') or {}
    if key == 'ball_context':
        return _fill(NEUTRAL['ball_context'], s)
    if key == 'line_height_gap':
        return block_note(o, s, level)
    if key == 'formation_organisation':
        return _formation_note(o, s, level)
    if key == 'lateral_shift':
        # a signed median delay ("-0.1 s") says nothing to a reader; the lag phrase says which line moved first
        note = _fill(NEUTRAL['lag'], s)
        if not note:
            return None
        extra = None
        if state == 'one_spell':
            extra = _fill(NEUTRAL['lagSpell'], s)
        elif state == 'under_measured':
            extra = NEUTRAL['under_measured']
        elif state == 'absent' and 'evidence_bound' in (metrics.get('strengthFailed') or ()):
            extra = NEUTRAL['evidence_bound']
        return note+(' '+extra if extra else '')
    if key == 'support_distances':
        note = _fill(NEUTRAL['support'][level], s)
        if note and state == 'under_measured':
            note += ' '+NEUTRAL['under_measured']
        return note
    pooled = s.get('strictContext') and s.get('pooledMedian') and s.get('comparisonSeconds')
    note = _fill(NEUTRAL['metricPooled' if pooled else 'metric'][level], s)
    if not note:
        return None
    extra = []
    failed = list(metrics.get('strengthFailed') or ())
    favourable = metrics.get('favourableSide')
    count = int(o.get('occurrenceCount') or 0) if _num(o.get('occurrenceCount')) else len(o.get('occurrences') or [])
    if state == 'one_spell':
        extra.append(_fill(NEUTRAL['one_spell'], s))
    elif state == 'within_range' or (state is None and count >= 2):
        blocked = metrics.get('improvementBlocked') == 'one_team_baseline'
        extra.append(_fill(NEUTRAL['one_team' if blocked else 'within_range'], s))
    elif state in ('absent', 'under_measured') or (state is None and count == 0):
        extra.append(_fill(NEUTRAL['absent'], s))
    if state == 'under_measured' or 'short_for_strength' in failed:
        extra.append(NEUTRAL['under_measured'])
    elif favourable is not False and 'evidence_bound' in failed:
        extra.append(NEUTRAL['evidence_bound'])
    elif favourable is not False and 'context_fails' in failed:
        extra.append(context_fails_note(metrics))
    if o.get('pattern') in NEUTRAL_HEDGE_PATTERNS and (favourable is False or state == 'one_spell'):
        extra.append(_fill(NEUTRAL['wide'], s))
    return ' '.join([note]+[x for x in extra if x])


def _formation_note(o, s, level):
    state = o.get('state')
    metrics = o.get('metrics') or {}
    pattern = o.get('pattern')
    if state == 'related_shapes' and s.get('pairNote'):
        if pattern == 'formation_change' and s.get('labelList'):
            note = _fill(NEUTRAL['formationRelatedChange'], s)
        else:
            note = _fill(NEUTRAL['formationRelated'], s)
    elif len(metrics.get('labels') or []) == 1 and state == 'under_measured':
        note = _fill(NEUTRAL['formationThroughout'][level], s)
    else:
        note = _fill(NEUTRAL['formation'][level], s)
    if not note:
        return None
    extra = [s.get('roles')]
    if state == 'under_measured' and s.get('strengthSeconds'):
        extra.append(_fill(NEUTRAL['formationShort'], s))
    return ' '.join([note]+[x for x in extra if x])


def block_note(o, s, level):
    """The block-height sentence of the line-height detector's context note (the team's and the opponent's median
    back-line height), for the report's context paragraph."""
    if not s.get('lineHeight') or not s.get('lineHeightSeconds'):
        return None
    if s.get('opponentLineHeight') and s.get('opponentSeconds'):
        note = _fill(NEUTRAL['block'][level], s)
    else:
        note = _fill(NEUTRAL['blockOwn'][level], s)
    if note and (o.get('confidence') or {}).get('overall') == 'low':
        note += ' '+NEUTRAL['blockUncertain']
    return note


def reason_phrases(observation):
    """The plain-words reasons behind an observation's confidence: its reason codes, else its medium or low aspects.
    Unknown codes are left out (never printed raw)."""
    conf = observation.get('confidence') or {}
    codes = list(observation.get('insufficientReasons') or [])+list(conf.get('reasons') or [])
    if 'small_scene_veto' in codes:     # "players appear too small ... for this measurement" already says it
        codes = [c for c in codes if c != 'small_scene']
    out = []
    for c in codes:
        text = REASONS.get(c) if isinstance(c, str) else None
        if text and text not in out:
            out.append(text)
    if not out:
        for aspect, words in ASPECTS.items():
            v = conf.get(aspect)
            if v in ('low', 'medium'):
                out.append(f'the {words} was of {v} confidence')
    return out


def limitation_reasons(observation, run_codes=()):
    """The reasons of one not-assessed line: the observation's own (the series' not-measured codes, and for a low
    confidence the reasons of its low aspects), without the run-level ones data_quality states once and those its
    numbers already say; at most MAX_LIMIT_REASONS, unknown codes left out. A small-scene veto names its rule, per
    detector (VETO_WORDS), never a measured cause."""
    conf = observation.get('confidence') or {}
    codes = [c for c in observation.get('insufficientReasons') or [] if isinstance(c, str)]
    if 'low_confidence' in codes:
        # the decisive reasons first: those of the aspects that made the confidence low (all of them when no aspect
        # says which)
        low = {a for a in ('tracking', 'calibration', 'identity') if conf.get(a) == 'low'}
        own = [c for c in conf.get('reasons') or [] if isinstance(c, str) and (not low or ASPECT_OF.get(c) in low)]
        own.sort(key=lambda c: c not in DECISIVE_REASONS)
        codes = own+[c for c in codes if c not in own]
    veto = 'small_scene_veto' in codes
    out = []
    if veto:
        out.append(VETO_WORDS.get(observation.get('detector'), REASONS['small_scene_veto']))
    for c in codes:
        if c in run_codes or c in QUIET_REASONS or c in ('small_scene', 'small_scene_veto', 'broadcast_only') or \
                (veto and c == 'no_ball'):
            continue
        text = LIMIT_WORDS.get(c) or REASONS.get(c)
        if text and text not in out:
            out.append(text)
    if not out and 'low_confidence' in codes:
        out.append(REASONS['low_confidence'])
    return out[:MAX_LIMIT_REASONS]


def _confidence_reasons(o, run_codes=()):
    """(this observation's own confidence reasons in words, whether run-level ones were left to the data-quality
    note): the codes the note states (run_codes) are left out, and on a keyless run so is the medium registration
    the keyless note explains."""
    conf = o.get('confidence') or {}
    codes = [c for c in list(o.get('insufficientReasons') or [])+list(conf.get('reasons') or []) if isinstance(c, str)]
    run = [c for c in codes if c in run_codes or (c == 'medium_registration' and 'keyless' in run_codes)]
    reasons = []
    for c in codes:
        text = REASONS.get(c) if c not in run else None
        if text and text not in reasons:
            reasons.append(text)
    if not reasons and not run:
        reasons = reason_phrases(o)
    return reasons, bool(run)


def _confidence_note(o, run_codes=()):
    """'Medium confidence: <this finding's reasons>.' The run-level reasons are stated once in the data-quality note;
    a finding that has only those points there."""
    if (o.get('confidence') or {}).get('overall') != 'medium':
        return None
    return _note_text(*_confidence_reasons(o, run_codes))


def _note_text(reasons, run):
    if not reasons:
        return 'Medium confidence because of the footage limits in the data-quality note.' if run else \
            'Medium confidence: the measurement conditions were not ideal.'
    return 'Medium confidence: '+'; '.join(reasons)+(' (and the footage limits in the data-quality note)' if run
                                                      else '')+'.'


def limitation(ctx, level=None):
    """The one short line of an observation that could not be assessed: topic, what could be measured and its own
    reasons (limitation_reasons; the run-level ones are in the data-quality note)."""
    o = ctx.observation
    level = level or ctx.level
    slots = _slots(ctx, level)
    topic = slots.get('topic') or 'This measurement'
    reasons = limitation_reasons(o, ctx.run_codes)
    veto = 'small_scene_veto' in (o.get('insufficientReasons') or ())
    if veto and not slots.get('measured'):
        return f'{topic}: not measured on this footage; '+'; '.join(reasons)+'.'
    if slots.get('measured'):
        part = f'measured {slots["measured"]}'
        if slots.get('eligible'):
            part += f' ({slots["eligible"]} of the time)'
        if slots.get('longest'):
            part += f', longest continuous stretch {slots["longest"]}'
    else:
        part = 'nothing could be measured'
    text = f'{topic}: not assessed; {part}'
    if reasons:
        text += '; '+'; '.join(reasons)
    return text+'.'


def withheld(ctx, reason, level=None):
    """The limitation line of a finding the report withheld (low confidence, no wording, unverified text); a low
    confidence names its own reasons (limitation_reasons), not the run-level ones."""
    level = level or ctx.level
    slots = _slots(ctx, level)
    topic = slots.get('topic') or 'An observation'
    text = f'{topic}: not shown because {REASONS.get(reason, REASONS["low_confidence"])}'
    reasons = []
    if reason == 'low_confidence':
        o = dict(ctx.observation, insufficientReasons=['low_confidence'])
        reasons = [r for r in limitation_reasons(o, ctx.run_codes) if r != REASONS['low_confidence']]
    return text+('; '+'; '.join(reasons) if reasons else '')+'.'


def _practice(drill, cls):
    if not drill:
        return None
    pl, du = drill.get('players') or {}, drill.get('durationMinutes') or {}
    size = ''
    if _num(pl.get('min')) and _num(pl.get('max')) and _num(du.get('min')) and _num(du.get('max')):
        size = f' ({pl["min"]} to {pl["max"]} players, {du["min"]} to {du["max"]} minutes)'
    lead = 'If the review confirms it, use this exercise: ' if cls == 'review' else 'Use this exercise: '
    return f'{lead}{drill["name"]}{size}. Aim: {_cap(drill["purpose"])} See Training.'


def _video(ctx, evidence=None):
    clips = []
    for e in list(ctx.evidence if evidence is None else evidence)[:MAX_CLIPS]:
        if not _num(e.get('time')):
            continue
        ids = [p for p in (e.get('players') or []) if isinstance(p, str)]
        who = [player_token(p) for p in ids[:MAX_CLIP_PLAYERS]]
        names = ', '.join(who)+(' and others' if len(ids) > MAX_CLIP_PLAYERS else '')
        clips.append(fmt_time(e['time'])+(f' ({names})' if who else ''))
    return 'Moments to watch: '+'; '.join(clips)+'.' if clips else None


# ---------------------------------------------------------------------------------------------------------------
# Grouped findings (report.py): one finding from several strengths of the same compact block.

def group_of(observation):
    """The GROUPS key an observation's strength belongs to, or None."""
    if observation.get('classification') != 'strength':
        return None
    for name, g in GROUPS.items():
        if observation.get('recommendationKey') == g['key'] and observation.get('pattern') in g['patterns']:
            return name
    return None


def group_render(name, contexts, level=DEFAULT_LEVEL):
    """The text fields of one grouped finding from the FindingContexts of its members (in GROUPS order): a headline
    naming the members, one lead sentence, one sentence per member (its median against the professional median of
    the same context, the strictest context for a pooled member, and its share beyond the professional p75), why it
    matters with the phase hedge, the "to build on it" line, the moments of all members and one confidence note."""
    g = GROUPS[name]
    level = level if level in LEVELS else DEFAULT_LEVEL
    order = {p: i for i, p in enumerate(g['patterns'])}
    contexts = sorted(contexts, key=lambda c: order.get(c.observation.get('pattern'), 9))
    slots = [_slots(c, level) for c in contexts]
    first = slots[0]
    pooled = [bool(s.get('strictContext') and s.get('pooledMedian') and s.get('comparisonSeconds')) for s in slots]
    named = {s.get('context') or '' for s, p in zip(slots, pooled) if not p}
    common = named.pop() if len(named) == 1 and not any(pooled) else ''
    members = []
    for s, p in zip(slots, pooled):
        own = '' if common or p else (s.get('context') or '')
        line = _fill(MEMBER_POOLED[level] if p else MEMBER[level],
                     dict(s, Measure=_cap((s.get('measure') or '')+own) or None))
        if line is None:
            raise NoWording('no_wording')
        members.append(line)
    shorts = [s.get('short') or s.get('measure') for s in slots]
    names = _join([x for x in shorts if x])
    headline = _fill(g['headline'][level], dict(first, members=names))
    measured = max(float(c.observation.get('measuredSeconds') or 0.0) for c in contexts)
    lead = _fill(g['lead'][level], dict(first, measured=fmt_duration(measured) if measured > 0 else None,
                                        context=common))
    definitions = list(dict.fromkeys(s.get('definition') for s in slots if s.get('definition')))
    happened = ' '.join(x for x in [lead]+members+definitions+[GROUP_CLOSE[level]] if x)
    tpl = TEMPLATES[g['key']]['strength']
    evidence = [e for c in contexts for e in c.evidence]
    hedge_slots = dict(first, checkWhere='check the moments below' if evidence else 'check the phase on the video')
    hedge_slots['hedge'] = _fill(PHASE_HEDGE[level], hedge_slots)
    why = _fill(tpl['whyItMatters'][level], hedge_slots)
    build = _fill(tpl['whatToImprove'][level], first)
    note = None
    medium = [c for c in contexts if (c.observation.get('confidence') or {}).get('overall') == 'medium']
    if medium:
        reasons, run = [], False
        for c in medium:
            own, r = _confidence_reasons(c.observation, c.run_codes)
            reasons += [x for x in own if x not in reasons]
            run = run or r
        note = _note_text(reasons, run)
    if not headline or not lead or not why:
        raise NoWording('no_wording')
    video_ctx = contexts[0]
    order_ev = _interleave([list(c.evidence) for c in contexts])
    return {'headline': headline, 'whatHappened': happened, 'whyItMatters': why, 'whatToImprove': build,
            'howToPractice': None, 'video': _video(video_ctx, order_ev), 'priority': None, 'confidenceNote': note,
            'evidence': [e.get('id') for e in order_ev]}


def _interleave(lists):
    """The members' evidence moments taken in turn, so the VIDEO line shows one moment of each member first."""
    out = []
    for k in range(max((len(x) for x in lists), default=0)):
        for x in lists:
            if k < len(x) and x[k] not in out:
                out.append(x[k])
    return out


# ---------------------------------------------------------------------------------------------------------------
# Report-level sentences (report.py): summary, notes and caveats, all number-checked against the report's facts.

def summary_sentences(kind, team, team_name, t0, t1, seconds, counts, main=None, main_kind=None,
                      confidence='low', limited=0, strength=None, wide=False, no_reference=False):
    """The report summary: scope, what met the evidence rules, the main point (and the main strength), the
    short-segment caution and the confidence. counts = {'strength': n, 'improvement': n, 'review': n}; wide: the
    footage shows only a few players per team (team analysis not possible); no_reference: no professional reference
    was available, which is then why nothing was compared (broadcast 0-60 s team A without one: 99-100% of the time
    measured, so "too little of the play could be measured" would be false)."""
    out = [f'{"Match" if kind == "match" else "Segment"} report for {team_full(team, team_name)}, '
           f'{fmt_time(t0)} to {fmt_time(t1)} ({fmt_duration(seconds)} of video).']
    words = {'strength': ('observed strength', 'observed strengths'),
             'improvement': ('potential improvement', 'potential improvements'),
             'review': ('moment to review', 'moments to review')}
    parts = [f'{n} {words[k][0] if n == 1 else words[k][1]}' for k in ('strength', 'improvement', 'review')
             for n in [counts.get(k, 0)] if n]
    span = 'this match' if kind == 'match' else 'this segment'
    if parts:
        out.append(f'{_cap(_join(parts))} met the evidence rules.')
    else:
        out.append(f'No strength, potential improvement or moment to review met the evidence rules in {span}.')
        if wide:
            out.append('Team analysis was not possible on this footage: only a few players of each team were in view '
                       'at a time (see the limitations).')
        elif no_reference and limited:
            out.append('No professional reference was available, so the measurements could not be compared; see the '
                       'limitations.')
        elif limited:
            out.append('Too little of the play could be measured reliably; see the limitations.')
    if main and main_kind == 'improvement':
        out.append(f'Main point to work on: {main}.')
    elif main and main_kind == 'review':
        out.append(f'Main point to review: {main}.')
    if strength:
        out.append(f'Main strength: {strength}.')
    if kind != 'match' and seconds < SHORT_SEGMENT_SECONDS:
        out.append("This is a short segment, so the findings describe what was seen in it and say nothing yet about "
                   "the team's play in general.")
    if parts and confidence != 'high':
        out.append(f'Overall confidence is {confidence}; see the notes on data quality.')
    return out


def data_quality(meta, codes=(), team='A', team_name=None, observed=None, help_spell=False, vetoed=()):
    """The run's data-quality limits, stated once at the head of the limitations with what would help: few players
    per team in view (a camera-view limit, not a judgement on the team), small players, no pitch keyframes, no
    professional reference. codes: the run-level reason codes of the report's observations and meta; observed:
    {team: mean observed outfield players}; help_spell: some metric missed members of its lines; vetoed: the
    detectors a small-scene rule left out ('isolation', 'ball_context')."""
    codes = set(codes)
    footage = meta.get('footage')
    out, helps = [], []
    if footage == 'wide' or 'few_players' in codes:
        a, b = (observed or {}).get('A'), (observed or {}).get('B')
        if _num(a) and _num(b):
            out.append(f'Team analysis was not possible: on average only {a:.1f} players of Team A and {b:.1f} of '
                       f'Team B were in view, and the team measurements need every outfield player of a team in view '
                       f'at the same time. This is a camera-view limit, not a judgement on the team.')
        else:
            out.append('Team analysis was not possible: only a few players of each team were in view at a time, and '
                       'the team measurements need every outfield player of a team in view at the same time. This is '
                       'a camera-view limit, not a judgement on the team.')
        helps.append("use a higher or wider camera view that keeps both teams' lines in the picture")
    elif meta.get('smallScene') or footage == 'amateur' or 'small_scene' in codes:
        what = {'isolation': 'support distances', 'ball_context': 'ball location'}
        left = [what[d] for d in ('isolation', 'ball_context') if d in vetoed]
        out.append('Players appear small in this footage, so positions and identities are less certain than on '
                   'broadcast footage' + (f'; {_join(left)} are only measured on broadcast-style footage.' if left
                                          else '.'))
        helps.append('use a higher or wider camera view')
    if meta.get('keyless') or 'keyless' in codes:
        out.append('No pitch keyframes were set, so the pitch registration is automatic and stays at medium '
                   'confidence; positions near the far touchline and the ends of the pitch are the least reliable.')
        helps.insert(0, f'set {KEYFRAME_ADVICE[0]}-{KEYFRAME_ADVICE[1]} pitch keyframes on frames where the pitch '
                        f'lines are clearly visible')
    if help_spell and footage != 'wide':
        helps.insert(1 if helps and helps[0].startswith('set ') else 0,
                     'choose a spell where both lines of the team are in view')
    if 'no_reference' in codes:
        out.append('No professional reference was available, so nothing in this report is compared with '
                   'professional play.')
    if helps:
        out.append(('What would help: ' if out else 'To measure more: ')+_join(helps, word='or')+'.')
    return out


def caveat_notes(meta, team='A'):
    """The method notes every report carries (confidence.notes). The footage limits are not here: data_quality
    states them once, at the head of the limitations."""
    notes = []
    pitch = meta.get('pitch') or {}
    if meta.get('metresEstimated', True):
        if _num(pitch.get('length')) and _num(pitch.get('width')):
            notes.append(f'Distances are estimates: the pitch was assumed to be {pitch["length"]:.0f} x '
                         f'{pitch["width"]:.0f} m because its real size was not measured.')
        else:
            notes.append('Distances are estimates: the pitch was assumed to be of standard size because its real '
                         'size was not measured.')
    notes.append('The tracking result does not record which team had the ball, so this report does not separate '
                 'attacking from defending phases; check the phase on the video moments.')
    reg = meta.get('registration') or {}
    if all(_num(reg.get(k)) for k in ('high', 'medium', 'low')):
        notes.append(f'Pitch registration: high confidence for {fmt_percent(reg["high"])} of the segment, medium for '
                     f'{fmt_percent(reg["medium"])}, low or none for {fmt_percent(reg["low"])}.')
    if _num(meta.get('corrections')) and meta['corrections'] > 0:
        n = int(meta['corrections'])
        notes.append(f'This report reflects {n} manual correction{"s" if n != 1 else ""} to the tracking.')
    notes.append(f'Team letters are assigned per run: the team labelled {team} here may carry another letter in '
                 f'another run of the same video.')
    if 'no_reference' not in (meta.get('limitations') or ()):     # without one, the data-quality note says so
        notes.append('Comparisons are with professional tracking data from a single sample match, used as a reference '
                     'range rather than a standard to copy'+("; professional distances are real metres, this team's "
                                                              "are estimates." if meta.get('metresEstimated', True)
                                                             else '.'))
    return notes


SECTION_REASONS = {
    'attackingStructure': "The tracking result does not record which team had the ball, so the team's attacking "
                          "structure cannot be separated from its defending shape.",
    'default': 'No measurement of this kind could be made in this segment.',
}


def section_measured_note(measured_seconds, range_seconds):
    return f'Measured for up to {fmt_duration(measured_seconds)} of the {fmt_duration(range_seconds)} analysed.'


def section_pointer(topics, available=True, no_reference=False):
    """The breakdown's pointer to the limitations: 'Not assessed here: team width and team area (see the
    limitations).' Each limitation line is printed once, under the limitations. no_reference: nothing could be
    compared, which is then the reason given (not a lack of reliable measurement)."""
    topics = list(dict.fromkeys(_lower(t) for t in topics if t))
    if not topics:
        return None
    names = _join(topics)
    if available:
        return f'Not assessed here: {names} (see the limitations).'
    if no_reference:
        return f'No professional reference was available: {names} could not be compared (see the limitations).'
    return f'Not enough reliable measurement: {names} could not be assessed (see the limitations).'


def section_unavailable(section, limitations=(), topics=(), no_reference=False):
    """Why a breakdown section has no finding: the possession gap for attacking structure, else a pointer to its
    not-assessed measurements (section_pointer), else the generic reason."""
    if section in SECTION_REASONS and section != 'default':
        return SECTION_REASONS[section]
    if topics:
        return section_pointer(topics, available=False, no_reference=no_reference)
    if limitations:
        return 'Not enough reliable measurement. '+' '.join(limitations)
    return SECTION_REASONS['default']


def meta_reason(reason):
    """A reason the engine passed in meta (a code from REASONS, or plain printable text without digits)."""
    if isinstance(reason, str) and reason in REASONS:
        return _cap(REASONS[reason])+'.'
    if isinstance(reason, str) and PRINTABLE.fullmatch(reason) and not re.search(r'\d', reason) and ' ' in reason:
        return reason.strip()
    return None


def all_template_strings():
    """Every fixed string of this module's wording, with a path, for the ASCII and tone checks."""
    out = []

    def walk(x, path):
        if isinstance(x, str):
            out.append((path, x))
        elif isinstance(x, dict):
            for k, v in x.items():
                walk(v, f'{path}.{k}')
        elif isinstance(x, (list, tuple)):
            for i, v in enumerate(x):
                walk(v, f'{path}[{i}]')
    walk(TEMPLATES, 'TEMPLATES')
    walk({p: {k: v for k, v in d.items() if k not in ('key', 'metric')} for p, d in PATTERNS.items()}, 'PATTERNS')
    walk(NEUTRAL, 'NEUTRAL')
    walk(REASONS, 'REASONS')
    walk(LIMIT_WORDS, 'LIMIT_WORDS')
    walk(VETO_WORDS, 'VETO_WORDS')
    walk(SECTION_REASONS, 'SECTION_REASONS')
    walk({k: {f: v for f, v in g.items() if f not in ('key', 'patterns', 'min')} for k, g in GROUPS.items()}, 'GROUPS')
    walk([MEMBER, MEMBER_POOLED, GROUP_CLOSE], 'GROUP_SENTENCES')
    walk({' / '.join(sorted(k)): v for k, v in RELATED_SHAPES.items()}, 'RELATED_SHAPES')
    walk([LEAD, LEAD_PLAIN, LEAD_POOLED, COMPARISON, BEYOND, PHASE_HEDGE], 'STRENGTH_SENTENCES')
    walk([RATE, WINDOW, HIGHLIGHT, POSITIONS_ONLY, NOT_IN_ROSTER], 'SENTENCES')
    return out
