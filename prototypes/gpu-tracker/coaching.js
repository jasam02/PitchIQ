/* PitchIQ · Coaching report: the coaching panel of the fourth workspace tab (the coaching setup, the report, its key
   moments), the evidence of a moment (its frame with the overlay, the top-down diagram) and the drawing of an
   annotation on the 2D board and over the video.

   Two report engines. A coaching v2 report (soccer/coaching/schema.py: kind 'coaching-report', version 2) is shown in
   the seven sections of DESIGN 9.2: MATCH OVERVIEW, WHAT WENT WELL, WHAT NEEDS WORK, KEY MOMENTS, TACTICAL DIAGRAMS,
   TRAINING DRILLS, DEVELOPMENT PROGRESSION. A server without the v2 routes (GET /api/coaching/profiles fails) builds the
   v1 report (one professional reference, three wordings), which keeps its v1 rendering below ("legacy" mode); the
   setup's level then picks the nearest v1 wording, and the setup stays on the page.

   Setup (DESIGN 9.1). Age group and playing level are kept apart and never guessed (unset until chosen). The format
   follows the age group's default (U.S. Soccer formats by age, the user's decision of Oct 21; England and Canada differ
   at U11 and U13) until the coach picks another one; the formations and the roles a coach may confirm follow the format
   (small formats get relative slots such as 'back-2/3', never 11v11 role names). Saving posts the team's block (schema
   TeamSettings) and the server says which cache tiers went stale: a level, age or style change re-interprets the same
   measurements at once; a format or field change needs Generate (no new tracking).

   Possession marked by the coach (wave 7, user decision 1 of Oct 21; PhaseMarks below). possession-4 abstains on
   amateur and Veo footage, so every phase-dependent check read "not assessed" there. While watching (a bar under the
   video and the 2D match, opened from a button in their controls) the coach presses "Our ball / Their ball / Dead ball
   from here" and "End"; the newest mark wins where it overlaps an older one, a list edits or removes phases, and every
   change saves the run's whole list (POST /api/coaching/spans, schema 'coach_spans', team letters: "ours" is the report's
   team). The marks are drawn on a lane of their own under the possession strip (possession.js, the engine's estimate, is
   untouched), and the report names them "marked by the coach" wherever it uses them. Measured on the scratch server
   (Oct 22, CPU): on the amateur clip (AM, 63.8 s) four marked phases turned "Go to the ball", "Support behind an
   advancing wide defender" and "Someone ready to protect our goal" from "the ball and who had it could not be followed"
   into their own reasons (no carrier, roles unconfirmed, ball unreliable) and made "Spread out" checkable (1 moment);
   on BC60 one marked span changed the phase words to "(estimated, partly marked by the coach)". The marks are a
   measurement input: the report asks for Generate after a change (2.0 s on BC60, 4.0 s on BC180, no new tracking).

   Server (wave 7). Every route of the setup, report, marks, feedback, squads, match date and progress detail was run
   against SERVER's server.py on a scratch server root (fe_w7/srv/launch.py: the store in scratch, the video read only).
   A level or age change answered with the re-interpreted report in 117 ms (save to report shown, BC60), no Generate.

   Register. The page writes no coaching prose: a finding carries its own validated text (explain.py). The page only adds
   counts, measurements and phase words where the level's register allows them (profile.register.numbers: none at level
   1, counts at level 2, metres from level 3, shares from level 5), so a level-1 report shows no number and adds no word
   level 1 bans (tested on a schema-valid mock: 0 banned words, 0 unmeasured events in the rendered sections).

   Annotations (DESIGN 5.4-5.6). New elements: focus (the player reviewed or praised), support, relation, channel,
   suggestedZone, moveArrow, caption. Each is drawn only at the tiers TIER_ELEMENTS allows (a suggested area over the
   video only at 'world'; an image-tier moment opens on the 2D match), at most ELEMENT_BUDGET of each. Their colours are
   chosen per pair of team colours (annotationPalette: CIEDE2000 and WCAG contrast). A key moment's annotation lives in
   its window (3 s before to 2 s after the key time): the key frame draws the whole overlay, the other frames of the
   window only the people it follows, at their live boxes or board markers (index.html, coachMark).

   Data. Nothing is measured here. The server builds the report from the saved result (soccer/coaching; v1: observations
   measured over the segment and compared with a professional reference, worded by templates; v2: situations measured
   once per run and interpreted for the coaching profile) and caches it under a fingerprint of everything it was
   computed from (soccer/coaching/store.py). This view asks for the report of one selection (team, range; v1 also the
   context level), shows it, offers to build it when there is
   none and to rebuild it when the result, the team-shape settings or the engine changed since (the banner lists the
   reasons; the earlier report stays readable meanwhile), and posts the coach's verdict on each finding. The report keys
   players by their internal id; every id in its text and drawings is shown as the player's display label in the result
   the page shows now (labels renumber after corrections), with the id in the tooltip; an identity a correction excluded
   shows as "excluded (was A-03)", never as a bare label that now belongs to someone else.

   Requests. Each selection (team, range, level) is one status request; a newer one, also one answered from the cache,
   drops the late answer of an older one, so the report shown and the one Export PDF saves are always the selection's.
   A build in progress (this selection's, or another one's: busy) is asked again every second and never cached. Feedback
   belongs to one run: a verdict sits on its finding only in the run it was given in; verdicts from earlier runs of the
   video are listed apart (team letters are arbitrary per run).

   Time. Every report time is absolute source seconds (the video's currentTime), shown as mm:ss.ss with the offset into
   the analysed segment. An evidence record carries the absolute source frame and seek = (frame + 0.5)/fps: the browser
   then shows exactly that frame (20 of 20 targets per video in Chromium; seeking to the frame's own time showed the frame
   before on 1 of 20). The screenshot is that frame's raw pixels (/api/coaching/frame); its overlay is drawn here in the
   source video's pixels, so labels follow corrections.

   Drawings never claim more than the data: the evidence's tier says what may be drawn over the video (world: pitch
   lines through a paint-checked HIGH registration; pinned: shapes from the players' feet on that frame; image: boxes and
   labels only), diagrams draw the observed positions the record lists (an estimate would be dashed and hollow), and a
   suggested area is dotted, labelled, and drawn only where an alternative was validated (the v1 'suggested' circle is
   still drawn for old reports). */
(function (global) {
'use strict';

const SVG_NS = 'http://www.w3.org/2000/svg';
const BOARD = {length: 105, width: 68, margin: 4.5};   // the board is drawn 105 x 68 m (tactical.js PITCH); spec coordinates are normalized
const KINDS = {'Observed strength': 'strength', 'Review opportunity': 'review', 'Potential improvement': 'improvement', 'High-confidence concern': 'concern'};
// The strength and improvement rules as soccer/coaching/detectors.py applies them: a strength's typical value sits on the
// favourable side of the professional median and rarely beyond the professional upper quarter over enough measured time; an
// improvement has more spells beyond the professional level than 99% of professional stretches of the same measured length.
const GROUPS = [
  ['strengths', 'Strengths', 'What the team kept on the favourable side of the professional median, and rarely beyond the professional upper quarter, over enough measured time.',
   'No strength met the evidence bar in this range (a typical value on the favourable side of the professional median, rarely beyond its upper quarter, over enough measured time).'],
  ['improvements', 'Areas to improve', 'What went beyond the professional level more often than professional teams do over the same measured time, in the same context.',
   'Nothing went beyond the professional level more often than professional teams do over the same measured time.'],
  ['reviews', 'Worth reviewing', 'Context-dependent moments: worth a look on the video with the players, not a verdict.',
   'Nothing context-dependent to review in this range.']];
const SECTIONS = [['defensiveOrganization', 'Defensive organisation'], ['spacing', 'Spacing and compactness'], ['formation', 'Formation'],
                  ['attackingStructure', 'Attacking structure'], ['movementSupport', 'Movement and support']];
const VERDICTS = [['correct', 'Correct'], ['partial', 'Partially correct'], ['incorrect', 'Incorrect'], ['insufficient', 'Not enough evidence']];
const CONTEXTS = {development: 'Development', competitive: 'Competitive', advanced: 'Advanced'};
const TIERS = {world: 'Pitch lines drawn through this frame’s registration (high confidence, checked against the painted lines).',
               pinned: 'Shapes drawn from the players’ feet on this frame; pitch geometry is approximate.',
               image: 'Boxes and labels only: the registration is not reliable enough on this frame for pitch lines.'};
const METRICS = {width: ['Team width', 'm'], depth: ['Team depth', 'm'], hull: ['Team area', 'm2'], spread: ['Spread', 'm'],
                 gapDM: ['Defence to midfield gap', 'm'], gapMF: ['Midfield to attack gap', 'm'], lineStd: ['Back-line unevenness', 'm'],
                 lineHeight: ['Back-line height', 'm'], nearestTeammate: ['Nearest teammate', 'm'], lagDM: ['Midfield lag behind the defence', 's'],
                 corrDM: ['Defence and midfield moving together', 'ratio']};
// The page's tokens (--text, --text-2, --amber, --slate, --danger) and the board's neutral and label halo.
const TOKENS = {text: '#F1F5F2', muted: '#9AAA9F', halo: '#07131bd9', amber: '#D9A961', slate: '#8FB3D9', danger: '#D96B6B', neutral: '#8d99a6'};
const ID_PATTERN = /(?<![\w-])(?:[AB]-(?:\d{1,3}|\?)|GK-[A-Z0-9]{1,3})(?!\w)/g;   // ids in report text: language.ID_TOKEN, one pass
const POLL_MS = 1000;
const NARROW = '(max-width: 1180px)';                    // the grid's one-column breakpoint: evidence goes under its finding

// Rounded to the hundredth before the minutes are split (critic N8): 59.999 s is "01:00.00", never "00:60.00".
const clock = t => { const c = Math.round(Math.max(0, Number(t) || 0) * 100), m = Math.floor(c / 6000); return `${String(m).padStart(2, '0')}:${((c - m * 6000) / 100).toFixed(2).padStart(5, '0')}`; };
const parseClock = text => { const m = String(text).trim().match(/^(?:(\d+):)?(\d+(?:\.\d+)?)$/); return m ? (m[1] ? Number(m[1]) * 60 : 0) + Number(m[2]) : null; };
const seconds = v => !Number.isFinite(v) ? '–' : v >= 90 ? `${(v / 60).toFixed(1)} min` : `${v.toFixed(1)} s`;
const words = key => String(key).replace(/([a-z0-9])([A-Z])/g, '$1 $2').replace(/_/g, ' ').toLowerCase().replace(/^./, c => c.toUpperCase());
const isNum = v => typeof v === 'number' && Number.isFinite(v);
const r2 = v => Math.round(v * 100) / 100;
// The one rounding of a typed range time: the status request sends toFixed(2) of this value and the build request the
// value itself, so both name the same hundredth (rounding twice, toFixed and Python's round, asked for 00:10.125 as 10.13
// and built it as 10.12, and the report was never found).
const hundredths = r2;
const UNPLACED = ['excluded', 'withdrawn'];             // match groups of identities that are no longer players (store.UNPLACED)

// ---------- coaching v2: the contract is soccer/coaching/schema.py ----------
const isV2 = r => !!r && typeof r === 'object' && r.kind === 'coaching-report' && r.version === 2;
const ALL_TIERS = ['world', 'pinned', 'image', 'diagram'];
// Where each element may be drawn (schema.TIER_ELEMENTS, DESIGN 5.4; a test compares the two tables): the video's tier
// is the evidence's (world: a paint-checked registration, pinned: the players' feet on that frame, image: boxes only),
// 'diagram' the top-down drawing. A suggested area is drawn over the video only at world: anything less would place a
// precise-looking shape the registration cannot support (spec Part 9), so the moment then opens on the 2D diagram.
const TIER_ELEMENTS = {
  focus: ALL_TIERS, support: ALL_TIERS, relation: ['world', 'pinned', 'diagram'], channel: ['world', 'diagram'],
  suggestedZone: ['world', 'diagram'], moveArrow: ['world', 'diagram'], caption: ALL_TIERS,
  player: ALL_TIERS, unit: ['world', 'pinned', 'diagram'], measure: ['world', 'pinned', 'diagram'], zone: ['world', 'diagram'],
  band: ['world', 'diagram'], arrow: ALL_TIERS, label: ALL_TIERS, suggested: ['diagram']};
const V2_TYPES = new Set(['focus', 'support', 'relation', 'channel', 'suggestedZone', 'moveArrow', 'caption']);
// At most this many of each new element per annotation (schema.ELEMENT_BUDGET, DESIGN 5.5: "do not overload the
// video"); the schema refuses more, and the drawing caps them again so a damaged record cannot clutter the frame.
const ELEMENT_BUDGET = {focus: 2, support: 3, relation: 3, channel: 1, suggestedZone: 1, moveArrow: 1, caption: 1};
// The annotation window around a key moment (DESIGN 5.6, [P]: key time - 3.0 s to + 2.0 s) is the record's
// annotation.window; index.html keeps the mark while the video is inside it (one frame of tolerance at either edge).
const CAPTION_STRIP = 4.6;          // metres of dark strip above a v2 diagram's pitch that hold its caption

// ---------- annotation colours (DESIGN 5.5): chosen per pair of team colours, never fixed ----------
// The spec asks for a red/orange outline and a green dotted area that stay distinguishable from the team colours. Fixed
// colours cannot: Team B's default orange #f6b564 is "orange", and green is the pitch. So each colour is picked from three
// candidates by CIEDE2000 distance (Sharma, Wu and Dalal 2005; the 14 published test pairs are reproduced to 4 decimals in
// test_coaching_panel.py) and WCAG contrast. With the default teams (#b98cff, #f6b564) the review outline is rose
// #ff4d6d (nearest colour 32.1 away: team A) and the zone mint #7dffb0. Lime would win on contrast, but lime is the
// video's ball marker (#c6ff3d, 1.2 away), so a candidate closer than BALL_MIN_DE to the ball is never the zone colour
// (a deviation from DESIGN 5.5, which did not list the ball). render.py must use the same rule (INTERFACES.md).
const PALETTE = {
  review: ['#ff4d6d', '#ff2fd1', '#ffffff'],            // rose red, magenta, white
  zone: ['#7dffb0', '#c6ff4d', '#5cf2ff'],              // mint, lime, aqua
  pitch: ['#2b8a46', '#30944d'],                        // the board's two grass stripes (tactical.js; also the grass proxy on video)
  ball: '#c6ff3d',                                      // index.html COLORS.BALL: the ball marker over the video
  minDE: 20,                                            // PALETTE_MIN_DE [P]: CIEDE2000 distance a colour keeps from every team colour
  ballMinDE: 10,                                        // [P]: a zone colour is never the ball's own colour
  minContrast: 3,                                       // WCAG 2 non-text contrast of the zone against both stripes
  relation: '#ffffff', relationOpacity: .45,            // relation lines: white at 45%, dashed when the relation failed
  halo: '#07131bd9'};
const hexRgb = hex => {
  const m = /^#?([0-9a-f]{3}|[0-9a-f]{6})$/i.exec(String(hex || '').trim());
  if (!m) return null;
  const h = m[1].length === 3 ? m[1].split('').map(c => c + c).join('') : m[1];
  return [0, 2, 4].map(i => parseInt(h.slice(i, i + 2), 16));
};
const srgbLinear = v => { v /= 255; return v <= .04045 ? v / 12.92 : Math.pow((v + .055) / 1.055, 2.4); };
function labOf(hex) {
  // sRGB (D65) -> CIE L*a*b*.
  const rgb = hexRgb(hex);
  if (!rgb) return null;
  const [r, g, b] = rgb.map(srgbLinear);
  const x = (.4124564 * r + .3575761 * g + .1804375 * b) / .95047, y = .2126729 * r + .7151522 * g + .072175 * b, z = (.0193339 * r + .119192 * g + .9503041 * b) / 1.08883;
  const f = t => t > 216 / 24389 ? Math.cbrt(t) : (24389 / 27 * t + 16) / 116;
  return [116 * f(y) - 16, 500 * (f(x) - f(y)), 200 * (f(y) - f(z))];
}
function ciede2000(lab1, lab2) {
  // CIEDE2000 colour difference of two L*a*b* colours (kL = kC = kH = 1), as Sharma, Wu and Dalal (2005) give it.
  const [L1, a1, b1] = lab1, [L2, a2, b2] = lab2, rad = d => d * Math.PI / 180, p7 = Math.pow(25, 7);
  const C1 = Math.hypot(a1, b1), C2 = Math.hypot(a2, b2), Cb7 = Math.pow((C1 + C2) / 2, 7), G = .5 * (1 - Math.sqrt(Cb7 / (Cb7 + p7)));
  const a1p = (1 + G) * a1, a2p = (1 + G) * a2, C1p = Math.hypot(a1p, b1), C2p = Math.hypot(a2p, b2);
  const hue = (b, a) => { if (b === 0 && a === 0) return 0; const h = Math.atan2(b, a) * 180 / Math.PI; return h < 0 ? h + 360 : h; };
  const h1 = hue(b1, a1p), h2 = hue(b2, a2p), zero = C1p * C2p === 0;
  let dh = zero ? 0 : h2 - h1;
  if (dh > 180) dh -= 360; else if (dh < -180) dh += 360;
  const dL = L2 - L1, dC = C2p - C1p, dH = 2 * Math.sqrt(C1p * C2p) * Math.sin(rad(dh / 2)), Lb = (L1 + L2) / 2, Cb = (C1p + C2p) / 2;
  const hb = zero ? h1 + h2 : Math.abs(h1 - h2) > 180 ? (h1 + h2 < 360 ? (h1 + h2 + 360) / 2 : (h1 + h2 - 360) / 2) : (h1 + h2) / 2;
  const T = 1 - .17 * Math.cos(rad(hb - 30)) + .24 * Math.cos(rad(2 * hb)) + .32 * Math.cos(rad(3 * hb + 6)) - .2 * Math.cos(rad(4 * hb - 63));
  const Cbp7 = Math.pow(Cb, 7), Rt = -Math.sin(rad(2 * 30 * Math.exp(-Math.pow((hb - 275) / 25, 2)))) * 2 * Math.sqrt(Cbp7 / (Cbp7 + p7));
  const Sl = 1 + .015 * Math.pow(Lb - 50, 2) / Math.sqrt(20 + Math.pow(Lb - 50, 2)), Sc = 1 + .045 * Cb, Sh = 1 + .015 * Cb * T;
  return Math.sqrt(Math.pow(dL / Sl, 2) + Math.pow(dC / Sc, 2) + Math.pow(dH / Sh, 2) + Rt * (dC / Sc) * (dH / Sh));
}
const deltaE = (p, q) => { const a = labOf(p), b = labOf(q); return a && b ? ciede2000(a, b) : Infinity; };
const luminance = hex => { const [r, g, b] = (hexRgb(hex) || [0, 0, 0]).map(srgbLinear); return .2126 * r + .7152 * g + .0722 * b; };
const contrastRatio = (p, q) => { const a = luminance(p), b = luminance(q); return (Math.max(a, b) + .05) / (Math.min(a, b) + .05); };
function annotationPalette(teamA, teamB, pitch) {
  // The colours of one annotation for these team colours (DESIGN 5.5; render.py implements the same rule):
  // - review outline: the review candidate with the largest smallest CIEDE2000 distance to both team colours and the
  //   grass; below minDE it is white with a dashed dark halo (fallback);
  // - suggested zone: among the zone candidates with WCAG contrast >= minContrast against both stripes, >= minDE from
  //   both team colours and >= ballMinDE from the ball marker, the best contrast; when none passes, the candidate
  //   farthest from the team colours (fallback: drawn over a heavier dark halo).
  const greens = Array.isArray(pitch) && pitch.length ? pitch : PALETTE.pitch, teams = [teamA, teamB].filter(c => hexRgb(c));
  const minTo = (c, others) => Math.min(...others.map(o => deltaE(c, o)));
  const reviews = PALETTE.review.map(c => ({c, d: minTo(c, teams.concat(greens))})).sort((p, q) => q.d - p.d || PALETTE.review.indexOf(p.c) - PALETTE.review.indexOf(q.c));
  const reviewOk = reviews[0].d >= PALETTE.minDE;
  const zones = PALETTE.zone.map(c => ({c, contrast: Math.min(...greens.map(g => contrastRatio(c, g))), team: teams.length ? minTo(c, teams) : Infinity, ball: deltaE(c, PALETTE.ball)}));
  const passing = zones.filter(z => z.contrast >= PALETTE.minContrast && z.team >= PALETTE.minDE && z.ball >= PALETTE.ballMinDE).sort((p, q) => q.contrast - p.contrast);
  const zone = passing[0] || zones.filter(z => z.ball >= PALETTE.ballMinDE).sort((p, q) => q.team - p.team || q.contrast - p.contrast)[0];
  return {review: reviewOk ? reviews[0].c : '#ffffff', reviewFallback: !reviewOk, zone: zone.c, zoneFallback: !passing.length,
          relation: PALETTE.relation, relationOpacity: PALETTE.relationOpacity, halo: PALETTE.halo,
          scores: {review: Math.round(reviews[0].d * 10) / 10, zoneContrast: Math.round(zone.contrast * 100) / 100, zoneTeam: Math.round(zone.team * 10) / 10}};
}

// ---------- coaching v2 vocabulary shown on the page ----------
const VERDICT_UI = {strength: ['Strength', 'strength'], try_this: ['Try this', 'try'], work_on: ['Work on', 'improvement'],
                    review: ['Possible issue to review', 'review'], describe: ['Context', 'describe']};
// Fix round 2 (LEAD decision (b); report.REVIEW_PARTS / review_part, TEXT's request): a 'moments worth a look' review
// (assess limitation 'moments_worth_a_look': repeated clear failures below the level's verdict bar) is a review in its
// own part of WHAT NEEDS WORK, after the possible issues, with its own badge, "What to look for." and its clear counts;
// it fails the evidence guardrail by construction, so that note is not shown, and it has no area, drill or objective.
const MOMENTS_CODE = 'moments_worth_a_look';
const isMoments = (f) => !!f && Array.isArray(f.limitations) && f.limitations.includes(MOMENTS_CODE);
const MOMENTS_UI = ['Moments worth a look', 'review'];
const REVIEW_PARTS_UI = {issues: ['Possible issues to review', 'PitchIQ could not confirm everything about these: worth a look on the video, not a verdict.'],
                         moments: ['Moments worth a look', 'Clear moments, in separate spells, that stayed below this level’s bar for a verdict: no verdict and no drill. Watch them with the players and decide whether they matter for how you want to play.']};
const verdictUi = (f) => isMoments(f) ? MOMENTS_UI : VERDICT_UI[f.verdict] || ['Finding', 'review'];
// Polish round (Oct 23; assess.LEVEL_FOCUS_LABEL, the PDF's copy, a test keeps them equal): with no objective from the
// match the NEXT DEVELOPMENT OBJECTIVE is the level's suggested focus from its guidance, "<label>: <check>. <guidance>
// Source: <ids>."; the page shows the label on its own line, the focus, and each source's title with its id.
const LEVEL_FOCUS_LABEL = 'Suggested focus for this level - from the level\'s guidance, not from this match';
const ORIGIN_NAMED = /from this match|from the level|this match['’]s|level['’]s guidance/i;   // an objective text that says where it comes from (pdf.ORIGIN_NAMED)
const splitFocus = (text) => {
  if (typeof text !== 'string' || !text.startsWith(LEVEL_FOCUS_LABEL + ': ')) return null;
  let body = text.slice(LEVEL_FOCUS_LABEL.length + 2), ids = [];
  const m = /\s*Source: ([A-Z0-9][A-Za-z0-9-]*(?:, [A-Z0-9][A-Za-z0-9-]*)*)\.$/.exec(body);
  if (m) { ids = m[1].split(', '); body = body.slice(0, m.index); }
  return {label: LEVEL_FOCUS_LABEL, body: body.trim(), ids};
};
// Pre-freeze (Oct 24, critic S-3: DRILLS / PAGE; pdf.OPTIONAL_LABELS / pdf.optional_label, a test keeps them equal):
// an optional drill read "not from this match’s evidence" also when the next objective was chosen from this match’s
// checks (a skill, not from a work item, not the level’s suggested focus: report.S8_LABEL), where it was false.
const OPTIONAL_LABELS = {match: 'Optional: for the next objective, chosen from this match’s checks (not a work item).',
                         level: 'Optional: for the next objective, not from this match’s evidence.'};
const optionalLabel = (r) => {
  const n = r && r.nextObjective && typeof r.nextObjective === 'object' ? r.nextObjective : null;
  return n && n.skill !== null && n.skill !== undefined && !n.fromMatch && !splitFocus(n.text) ? OPTIONAL_LABELS.match : OPTIONAL_LABELS.level;
};
const TIER_UI = {world: ['Pitch-accurate', 'Drawn through this frame’s pitch registration (paint-checked): areas and lines are placed on the pitch.'],
                 pinned: ['From the players’ feet', 'Drawn from the players’ feet on this frame; pitch areas are shown on the 2D diagram only.'],
                 image: ['Boxes only', 'The registration is not reliable on this frame: only the players are marked on the video; areas are on the 2D diagram.']};
// Why a check was not made (schema.ABSTAIN_REASONS), in the coach's words, with what would unlock it where the coach can.
const REASON_UI = {
  phase_unknown: 'the ball and who had it could not be followed reliably on this footage',
  possession_unavailable: 'who had the ball could not be estimated on this footage',
  ball_unreliable: 'the ball was not tracked reliably enough', no_carrier: 'the player on the ball was not known',
  out_of_view: 'the players involved were often out of the camera view', team_incomplete: 'too few teammates were in view',
  identity_uncertain: 'player identities were uncertain at those moments',
  role_unconfirmed: 'the players’ roles are not confirmed (confirm them under Player roles in the coaching setup)',
  scale_unknown: 'the field size is not known (enter it in the coaching setup)', format_not_supported: 'this check does not apply to this match format',
  registration_low: 'the camera-to-pitch registration was not reliable enough', short_range: 'the range is too short',
  recovering: 'players were running back towards their goal', units_interleaved: 'the team’s lines overlapped, so its units could not be told apart',
  near_change: 'it was close to a change of who had the ball', off: 'not part of this level’s checks',
  // fix round 1 (DETECT's new reasons, schema ABSTAIN_REASONS; critic S-7, N6, S-3)
  phase_absent: 'the phase this check needs did not happen in this range',
  coach_marks_one_side: 'you marked who had the ball for one team only: mark the other team’s spells too to run this check',
  formation_changing: 'the team’s formation was changing at those moments',
  // fix round 3 (ASSESS; critic S-2, COACH F-14): enough play, too few of the check's situations ("the range is too
  // short" was said of 4 clear advances in 6.1 minutes); with a count the page says how many (fewSituations)
  too_few_situations: 'too few of these situations happened in this range to judge'};
const fewSituations = n => `only ${n} ${n === 1 ? 'situation' : 'situations'} of this kind happened in this range, too few to judge`;
const PHASE_REASONS = ['phase_unknown', 'possession_unavailable'];   // the abstain reasons a coach's possession marks can lift (coach_marks_one_side says so itself)
const GUARDRAIL_UI = {identity: 'who the player was', position: 'the player’s pitch position', role: 'the player’s tactical role',
                      context: 'the teammates around the moment', phase: 'who had the ball', intent: 'whether the movement was intended',
                      alternative: 'a reachable better position', evidence: 'enough repeated evidence'};
// Fallback names of the checks; the report's profile.skills gives the level's own names (DESIGN 8.3) and wins.
const DETECTOR_UI = {pressure_cover: 'Going to the ball and helping a teammate', weak_side_balance: 'Protecting the far side',
                     wide_defender_advance: 'Support behind an advancing wide defender', behind_ball: 'Players ready behind the ball when attacking',
                     defensive_spacing: 'Back-line spacing', attacking_shape: 'Width and depth when attacking'};
const SKILL_OF = {pressure_cover: 'pressure_and_cover', behind_ball: 'someone_back_when_attacking', wide_defender_advance: 'wide_defender_support',
                  weak_side_balance: 'far_side_protection', defensive_spacing: 'back_line_coordination', attacking_shape: 'spread_when_attacking'};
// A finding's measurements on its card (DESIGN 9.2 "measurements in the level's register"), by the key DETECT gives them
// (explain.MEASUREMENT_KEYS): [label, unit, lowest level that shows it]. The lowest level is the one from which
// explain.py may state the same fact (centralCount from 4, the shares and the opponents on our side from 5, the
// margin from 6, the midfielder-to-centre-back spacing at 7), so a chip never says more than its level's text may;
// below level 3 the register shows no measurement at all. The first real reports (Oct 22) printed raw keys ("Threat
// n 1", "Weak count 0", "Open width m 34.0 m") because the wave-1 table had only the mock's keys. Strings (side,
// failure mode, block height, the phase source), ids and unknown keys are never chips.
const MEASURE_UI = {
  d1M: ['Nearest player to the ball', 'm', 3], d1Median: ['Nearest player to the ball', 'm', 3], coverDistM: ['Cover distance', 'm', 4],
  coverAngleDeg: ['Cover angle', 'deg', 6], pressurePresent: ['Pressure on the ball', 'bool', 3], coverPresent: ['Cover behind', 'bool', 3],
  coverPresentShare: ['Cover present', 'share', 5], goalSideShare: ['Goal-side', 'share', 5],
  weakCount: ['Defenders on the far side', 'count', 3], centralCount: ['Defenders in the middle', 'count', 4], openWidthM: ['Open width on the far side', 'm', 5],
  advanceM: ['Advance', 'm', 3], behindN: ['Behind the ball', 'count', 3], minBehind: ['Needed behind the ball', 'count', 3],
  channelOccupied: ['Channel behind covered', 'bool', 3], interchange: ['Positions swapped', 'bool', 3], opponentsInChannel: ['Opponents in the channel', 'count', 4],
  cbAdjusted: ['Centre-back moved across', 'bool', 5], nearCbShiftM: ['Near centre-back shift', 'm', 5], dmDropped: ['Holding midfielder dropped', 'bool', 5],
  threatN: ['Their players on our side of the ball', 'count', 5], restMargin: ['Margin behind the ball', 'signed', 6], dmCbSpacingM: ['Midfielder to centre-backs', 'm', 7],
  lateralGapM: ['Widest gap in the back line', 'm', 3], gapBallSide: ['Gap on the ball side', 'bool', 4], opponentInGap: ['Opponent in the gap', 'bool', 4],
  dmGapM: ['Defence to midfield gap', 'm', 5], lineStdM: ['Back-line unevenness', 'm', 5], dmGapOpponents: ['Opponents between defence and midfield on the ball’s side', 'count', 4],
  widthM: ['Width', 'm', 3], depthM: ['Depth', 'm', 3], channelsOccupied: ['Lanes occupied', 'count', 3],
  widthShare: ['Width of the pitch used', 'share', 5], depthShare: ['Length of the pitch used', 'share', 5]};
// Back-line spacing cited for the gap between defence and midfield (failureMode 'dm_gap') shows that gap's own chips from
// the level its text states them (the distance from 3, the opponents in it from 4, as opponentInGap) and none of the
// lateral gap's yes/no chips: a level-4 'moments worth a look' review cited for a 12.2 m gap with an opponent in it
// showed only "Gap on the ball side no; Opponent in the gap no" (CODE L-4, round 2). The PDF's pdf.chip_levels.
const DM_GAP_CHIPS = {dmGapM: 3, dmGapOpponents: 4}, LATERAL_CHIPS = ['gapBallSide', 'opponentInGap'];
function chipLevels(f) {
  const m = (f && f.measurements) || {}, out = {};
  for (const [k, ui] of Object.entries(MEASURE_UI)) out[k] = ui[2];
  const dm = f && f.observation === 'defensive_spacing' && f.verdict !== 'strength' && f.verdict !== 'describe'
    && (m.failureMode === 'dm_gap' || ((m.failureMode === undefined || m.failureMode === null) && isNum(m.dmGapM) && !isNum(m.lateralGapM)));
  if (dm) { Object.assign(out, DM_GAP_CHIPS); for (const k of LATERAL_CHIPS) out[k] = null; } else out.dmGapOpponents = null;
  return out;
}
const MASTERY_UI = {'NOT ENOUGH EVIDENCE': 'Not enough evidence yet', LEARNING: 'Learning', DEVELOPING: 'Developing', CONSISTENT: 'Consistent', ADAPTABLE: 'Adaptable'};
const TREND_UI = {up: 'improving', down: 'fewer successes', flat: 'steady', unknown: '–'};
// Why a skill reads as it does (progress.py mastery 'reason'), in the coach's words.
const MASTERY_WHY = {few_opportunities: 'fewer judged moments than the rules need so far', few_matches: 'too few matches with enough judged moments so far',
                     recent_matches: 'shown in the recent matches, each of them', contexts: 'shown in the recent matches, in more than one kind of situation',
                     lower_bound: 'read from how often it was shown over the recent matches'};
const PRINCIPLE_UI = {cover_first_defender: 'cover behind the player on the ball', cover_behind_advancing_defender: 'cover behind an advancing defender'};
const ALT_REASON_UI = {no_reachable_fix: 'no teammate could have reached a better area in time', would_open_new_gap: 'every reachable area would have opened another gap',
                       mover_unseen: 'the teammates who could help were not in view just before', context_missing: 'the players around were not all in view',
                       scale_unknown: 'the field size is not known', guardrail_failed: 'the moment could not be confirmed', no_zone_for_detector: 'this check describes the team’s shape, not one position'};
const DRILL_SOURCE_UI = {adapted_from: 'Adapted from', inspired_by: 'Inspired by', pitchiq_original: 'PitchIQ original, informed by'};

// ---------- the coaching setup catalogue (DESIGN 2.1-2.6) ----------
// GET /api/coaching/profiles answers this (SERVER); the copy here keeps the setup usable when that route is missing
// and is compared with schema.py by test_coaching_panel.py (ages, formats, formations, styles, federations).
const CATALOGUE = {
  ages: ['U6', 'U7', 'U8', 'U9', 'U10', 'U11', 'U12', 'U13', 'U14', 'U15', 'U16', 'U17', 'U18', 'U19', 'U21', 'U23', 'Adult'],
  levels: [
    {level: 1, name: 'FOUNDATION', typical: 'U7-U8', summary: 'Spreading out, going to the ball, helping a teammate; everyone tries every position.'},
    {level: 2, name: 'EARLY DEVELOPMENT', typical: 'U9-U10', summary: 'Pressure on the ball, a teammate covering, someone staying back.'},
    {level: 3, name: 'DEVELOPING', typical: 'U11-U12', summary: 'Width and depth, recovering towards goal, supporting the player on the ball.'},
    {level: 4, name: 'INTERMEDIATE', typical: 'U13-U14', summary: 'Positional responsibilities, the defensive line, the space behind an advancing defender.'},
    {level: 5, name: 'ADVANCED YOUTH', typical: 'U15-U18', summary: 'Collective defensive organisation, cover behind attacking full-backs, rotations.'},
    {level: 6, name: 'COMPETITIVE ADULT / ACADEMY', typical: 'Adult, or U17-U23 academy', summary: 'Tactical roles, players kept behind the ball, block compactness.'},
    {level: 7, name: 'PROFESSIONAL / ELITE', typical: 'Adult or U19+', summary: 'Analyst view, read against your declared game plan and style.'}],
  formats: [{name: '4v4', players: 4, goalkeeper: false, choice: true}, {name: '5v5', players: 5, goalkeeper: true, choice: true, note: 'assessed as 4v4'},
            {name: '7v7', players: 7, goalkeeper: true, choice: false}, {name: '9v9', players: 9, goalkeeper: true, choice: false},
            {name: '11v11', players: 11, goalkeeper: true, choice: false}],
  formations: {4: ['1-2-1'], 5: ['1-2-1'], 7: ['1-3-2-1', '1-2-3-1', '1-3-1-2'], 9: ['1-3-2-3', '1-3-3-2', '1-4-3-1'],
               11: ['4-4-2', '4-3-3', '4-2-3-1', '4-1-4-1', '3-5-2', '3-4-3', '5-3-2', '5-4-1', '4-4-1', '4-3-2', '5-3-1']},
  // DESIGN 2.5: shapes offered although no source describes them ([P]); the 9-outfield 11v11 shapes are for a dismissal.
  formationNotes: {'1-2-3-1': 'common shape, not from the sources', '1-3-1-2': 'common shape, not from the sources', '1-3-3-2': 'common shape, not from the sources',
                   '1-4-3-1': 'common shape, not from the sources', '4-4-1': 'with a player sent off', '4-3-2': 'with a player sent off', '5-3-1': 'with a player sent off'},
  styles: [{id: 'unknown', name: 'Unknown / no preference'}, {id: 'possession', name: 'Possession-oriented'}, {id: 'counter', name: 'Counter-attacking'},
           {id: 'high_press', name: 'High pressing'}, {id: 'compact', name: 'Compact defending'}],
  federations: [{id: 'auto', name: 'Automatic (U.S. formats)'}, {id: 'US', name: 'U.S. Soccer'}, {id: 'England', name: 'England'}, {id: 'Canada', name: 'Canada'}],
  // The template roles of 11v11 (schema.TEMPLATE_ROLES); small formats use relative slots (DESIGN 2.5).
  roles: ['GK', 'LB', 'LCB', 'CB', 'RCB', 'RB', 'LWB', 'RWB', 'CDM', 'LDM', 'RDM', 'LM', 'LCM', 'CM', 'RCM', 'RM', 'CAM', 'LW', 'RW', 'CF', 'LCF', 'RCF'],
  sourcesNote: 'PitchIQ’s checks and numbers are its own proposals; the sources describe coaching objectives, not PitchIQ’s thresholds. No federation reviewed or endorses PitchIQ.'};
const AGE_INDEX = a => CATALOGUE.ages.indexOf(a);
function formatDefault(age, federation) {
  // The match format a new age group starts with (DESIGN 2.2 "PitchIQ default"; the user's decision of Oct 21: U.S.
  // Soccer formats by age for auto and US, the coach can override), profiles.json formatDefaults when the server does
  // not send it. With England or Canada chosen, their own formats where the research has them: England 5v5 at U7-U8
  // (FA-PITCH-U78) and 7v7 at U11, Canada 5v5 at U8-U9 (4v4 without a keeper is also played; CS-STANDARDS-2026), both
  // 9v9 at U13; elsewhere the U.S. one (FORMAT_BASIS says where that is not the federation's). Fix round 1 (SOURCES F1):
  // England U7-U8 and Canada U8-U9 were 4v4 without a keeper and Canada U9 7v7.
  const k = AGE_INDEX(age), us = !federation || federation === 'auto' || federation === 'US';
  if (k < 0) return {players: 11, goalkeeper: true};
  if ((federation === 'England' && (age === 'U7' || age === 'U8')) || (federation === 'Canada' && (age === 'U8' || age === 'U9')))
    return {players: 5, goalkeeper: true};
  if (k <= AGE_INDEX('U8')) return {players: 4, goalkeeper: false};
  if (k <= AGE_INDEX('U10')) return {players: 7, goalkeeper: true};
  if (age === 'U11') return {players: us ? 9 : 7, goalkeeper: true};
  if (age === 'U12') return {players: 9, goalkeeper: true};
  if (age === 'U13') return {players: us ? 11 : 9, goalkeeper: true};
  return {players: 11, goalkeeper: true};
}
function typicalLevel(age) {
  // DESIGN 2.2 typical age per level, read backwards (adults: level 6).
  const k = AGE_INDEX(age);
  if (k < 0) return null;
  if (k <= AGE_INDEX('U8')) return 1;
  if (k <= AGE_INDEX('U10')) return 2;
  if (k <= AGE_INDEX('U12')) return 3;
  if (k <= AGE_INDEX('U14')) return 4;
  if (k <= AGE_INDEX('U18')) return 5;
  return 6;
}
const formatName = f => f && Number.isInteger(f.players) ? `${f.players}v${f.players}` : '11v11';
const outfieldOf = f => f.players - (f.goalkeeper ? 1 : 0);
// The relative blocks (back, mid, front) of a small format without a usable formation, by outfield count: the same
// table as profiles.OUTFIELD_BLOCKS, so the setup offers exactly the slots the server accepts.
const OUTFIELD_BLOCKS = {3: [1, 1, 1], 4: [1, 2, 1], 6: [3, 2, 1], 8: [3, 2, 3]};
function blocksOf(format, formation) {
  // profiles.blocks_of: the formation's blocks when it is offered for the format and describes its outfield players
  // (7v7 and 9v9 names count the keeper first, '1-3-2-1' = 6 outfield; the 4v4 / 5v5 '1-2-1' is outfield only), else
  // the generic blocks of the outfield count.
  const players = format.players, out = outfieldOf(format);
  if (formation && (CATALOGUE.formations[players] || []).includes(formation)) {
    const parts = formation.split('-').map(Number), sum = parts.reduce((a, b) => a + b, 0);
    if ((players === 7 || players === 9 ? sum - 1 : sum) === out) {
      const blocks = players === 7 || players === 9 ? parts.slice(1) : parts;
      if (blocks.length === 3) return blocks;
    }
  }
  if (OUTFIELD_BLOCKS[out]) return OUTFIELD_BLOCKS[out];
  const third = Math.max(1, Math.floor(out / 3));
  return [third, Math.max(1, out - 2 * third), third];
}
function relativeSlots(format, formation) {
  // The relative role slots of a small format (schema RELATIVE_ROLE 'back-2/3': block, k-th from the left of n).
  const out = [];
  blocksOf(format, formation).forEach((n, b) => { for (let k = 1; k <= n; k++) out.push(`${['back', 'mid', 'front'][b]}-${k}/${n}`); });
  return out;
}
function roleText(role) {
  const m = /^(back|mid|front)-(\d)\/(\d)$/.exec(String(role || ''));
  if (!m) return String(role || '');
  const block = {back: 'back', mid: 'middle', front: 'front'}[m[1]];
  return m[3] === '1' ? `${block} (only player)` : `${block} ${m[2]} of ${m[3]} from the left`;
}

// ---------- coaching setup helpers ----------
const titleCase = s => { const t = String(s || '').toLowerCase(); return t.charAt(0).toUpperCase() + t.slice(1); };
const pct = v => `${Math.round(v * 100)}%`;
const plural = (n, w) => n === 1 ? w : w + 's';
// Local time, as the server stamps its own records (store._iso: datetime.now()): a role confirmed at 05:14 local read
// "set 2026-10-24T11:14:43" (UTC) beside the feedback's "made 2026-10-24T05:14:50" (CODE L-7, round 2).
const isoNow = () => { const d = new Date(), two = n => String(n).padStart(2, '0'); return `${d.getFullYear()}-${two(d.getMonth() + 1)}-${two(d.getDate())}T${two(d.getHours())}:${two(d.getMinutes())}:${two(d.getSeconds())}`; };
// A finding's action as the card prints it: a strength's without its own opening "To build on it, " (explain.py opens
// all 20 v2 build-on texts so), since the lead "To build on it." says it: BC60 L5 read "To build on it. To build on it,
// hold the same far-side position deeper in our half." (CODE L-3, round 2). The PDF does the same (pdf.without_lead).
const BUILD_ON_LEAD = /^To build on it[,:.]?\s+/;
const actionText = f => { const a = String((f && f.text && f.text.action) || ''), m = f && f.verdict === 'strength' ? BUILD_ON_LEAD.exec(a) : null; if (!m) return a; const rest = a.slice(m[0].length); return rest.charAt(0).toUpperCase() + rest.slice(1); };
// Why the drills section is empty (the PDF's pdf.no_drill_text). Since the polish round a report without a work item
// offers one optional drill for the next objective, so "a drill is suggested only for a work item found in this range,
// never at random" misstated the rule where no drill fitted the objective (CODE L-5, round 2: SFS L4).
const noDrillText = r => {
  const work = ((r && r.needsWork) || []).filter(f => f && typeof f === 'object'), nxt = r && r.nextObjective && typeof r.nextObjective === 'object' ? r.nextObjective : null;
  if (work.length) return 'No drill: PitchIQ has none that fits the work items in this range for this team’s age, level and format.';
  if (!nxt || typeof nxt.text !== 'string' || !nxt.text.trim()) return 'No drill: there is no work item in this range and no next objective.';
  if (!nxt.skill) return 'No drill: there is no work item in this range, and the next objective is a general one with no drill tied to it.';
  return 'No drill: there is no work item in this range, and PitchIQ has none that fits the next objective for this team’s age, level, format and range.';
};
// An api() error in words: a body that is not JSON (an HTML error page) is named by its HTTP status, not by the parser.
const failText = e => !e ? 'no answer' : e.name === 'SyntaxError' && Number.isInteger(e.status) ? `the server answered HTTP ${e.status} without a readable message` : (e.message || 'no answer');
const clone = x => JSON.parse(JSON.stringify(x));
const clampN = v => Math.min(1.1, Math.max(-.1, Math.round(v * 10000) / 10000));   // schema NORM_MARGIN
// An adjusted area's note reaches the PDF, which SERVER keeps to printable ASCII: smart punctuation is folded to ASCII.
const ASCII_FOLD = [[/[\u2018\u2019\u201a\u2032]/g, "'"], [/[\u201c\u201d\u201e\u2033]/g, '"'], [/[\u2010-\u2015\u2212]/g, '-'], [/\u2026/g, '...'], [/[\u00a0\u2007\u202f]/g, ' ']];
const asciiNote = text => ASCII_FOLD.reduce((t, [re, to]) => t.replace(re, to), String(text || '')).replace(/\s+/g, ' ').trim();
const uniqueBy = (items, key) => { const seen = new Set(); return items.filter(x => { const k = key(x); if (!x || seen.has(k)) return false; seen.add(k); return true; }); };
// A server without the v2 routes builds the v1 report, whose only option is one of three wordings; the setup's level
// picks the nearest (levels 1-3 development, 4-5 competitive, 6-7 advanced; unset: competitive, the v1 default).
const legacyContext = level => !Number.isInteger(level) ? 'competitive' : level <= 3 ? 'development' : level <= 5 ? 'competitive' : 'advanced';
const sameFormat = (a, b) => !!a && !!b && a.players === b.players && !!a.goalkeeper === !!b.goalkeeper;
function blankBlock() {
  // schema.TIER_DEFAULTS as one team's settings: 11v11 with a keeper, no field, style unknown, federation auto. Age
  // group and playing level stay unset until the coach chooses them (PitchIQ never guesses a team's level).
  return {squad: null, ageGroup: null, playingLevel: null, format: {players: 11, goalkeeper: true}, field: null, federation: 'auto',
          formation: {intended: null, periods: []}, style: 'unknown', roles: {}, runLinks: []};
}
function cleanBlock(raw, cat) {
  // A team's coaching settings (schema TeamSettings) as the setup edits them; malformed parts become the defaults.
  const c = cat || CATALOGUE, b = blankBlock();
  if (!raw || typeof raw !== 'object') return b;
  if (c.ages.includes(raw.ageGroup)) b.ageGroup = raw.ageGroup;
  if (Number.isInteger(raw.playingLevel) && raw.playingLevel >= 1 && raw.playingLevel <= 7) b.playingLevel = raw.playingLevel;
  const f = raw.format;
  if (f && [4, 5, 7, 9, 11].includes(f.players)) b.format = {players: f.players, goalkeeper: f.players >= 7 ? true : typeof f.goalkeeper === 'boolean' ? f.goalkeeper : f.players === 5};
  if (raw.field && typeof raw.field === 'object') b.field = {length: isNum(raw.field.length) ? raw.field.length : NaN, width: isNum(raw.field.width) ? raw.field.width : NaN};
  if (c.federations.some(x => x.id === raw.federation)) b.federation = raw.federation;
  const fm = raw.formation, offered = c.formations[b.format.players] || [];
  if (fm && typeof fm === 'object') {
    b.formation.intended = offered.includes(fm.intended) ? fm.intended : null;
    b.formation.periods = Array.isArray(fm.periods) ? fm.periods.filter(p => p && offered.includes(p.formation) && isNum(p.t0) && isNum(p.t1)).map(p => ({t0: p.t0, t1: p.t1, formation: p.formation})) : [];
  }
  if (c.styles.some(x => x.id === raw.style)) b.style = raw.style;
  if (raw.roles && typeof raw.roles === 'object' && !Array.isArray(raw.roles)) {
    for (const [id, spans] of Object.entries(raw.roles)) {
      const ok = Array.isArray(spans) ? spans.filter(s => s && typeof s.role === 'string') : [];
      if (PLAYER_ID.test(id) && ok.length) b.roles[id] = ok.map(s => ({role: s.role, from: isNum(s.from) ? s.from : null, to: isNum(s.to) ? s.to : null, set: typeof s.set === 'string' ? s.set : isoNow()}));
    }
  }
  if (raw.squad && typeof raw.squad.id === 'string' && typeof raw.squad.name === 'string') b.squad = {id: raw.squad.id, name: raw.squad.name};
  if (Array.isArray(raw.runLinks)) b.runLinks = raw.runLinks.filter(l => l && typeof l.run === 'string' && (l.team === 'A' || l.team === 'B')).map(l => ({run: l.run, team: l.team, confirmed: !!l.confirmed}));
  return b;
}
function setupProblems(d) {
  // What keeps a setup from being saved (schema TeamSettings: age group and level required; field 15-130 x 10-100 m).
  const out = [], f = d.field;
  if (!d.ageGroup) out.push('Choose an age group.');
  if (!d.playingLevel) out.push('Choose a playing level.');
  if (f && !(isNum(f.length) && isNum(f.width) && f.length >= 15 && f.length <= 130 && f.width >= 10 && f.width <= 100 && f.width <= f.length))
    out.push('Field size: a length of 15 to 130 m and a width of 10 to 100 m, the width not larger than the length (or leave both empty).');
  return out;
}
function linkRun(d, run, team) {
  // This run's team is the squad (schema RunLink); one link per run.
  if (!run) return;
  d.runLinks = d.runLinks.filter(l => l.run !== run).concat({run, team, confirmed: true});
}
const isList = (v, ok) => Array.isArray(v) && v.length > 0 && v.every(ok);
const strings = v => isList(v, x => typeof x === 'string');
function mergeCatalogue(c) {
  // The server's catalogue (GET /api/coaching/profiles, profiles.catalog()) wherever a part is well formed, else this
  // file's copy. profiles.catalog() gives levels with their guidance, evidence rules and typical ages; ageGroups with
  // the typical level and the heading rule; formats with the goalkeeper default and choice, formations (tag, note) and
  // the roles a coach may confirm per formation; formatDefaults by age then federation; styles; federations as ids;
  // sources as a list; the disclaimer.
  const out = {...CATALOGUE, levelInfo: {}, typicalByAge: {}, headingByAge: {}, rolesByFormat: {}, formatNotes: {}};
  if (!c || typeof c !== 'object') return out;
  if (strings(c.ages)) out.ages = c.ages.slice();
  if (isList(c.ageGroups, a => a && typeof a.id === 'string')) {
    out.ages = c.ageGroups.map(a => a.id);
    for (const a of c.ageGroups) {
      if (Number.isInteger(a.typicalLevel)) out.typicalByAge[a.id] = a.typicalLevel;
      if (a.heading && typeof a.heading.text === 'string') out.headingByAge[a.id] = a.heading.text;
    }
  }
  if (Array.isArray(c.levels) && c.levels.length === 7 && c.levels.every((l, k) => l && l.level === k + 1 && typeof l.name === 'string')) {
    out.levels = c.levels.map(l => {
      const ages = strings(l.typicalAges) ? l.typicalAges : null;
      return {level: l.level, name: l.name, summary: typeof l.summary === 'string' ? l.summary : CATALOGUE.levels[l.level - 1].summary,
              typical: typeof l.typical === 'string' ? l.typical : ages ? (ages.length > 1 ? `${ages[0]}-${ages[ages.length - 1]}` : ages[0]) : CATALOGUE.levels[l.level - 1].typical};
    });
    for (const l of c.levels) if (Array.isArray(l.objectives) || l.evidence) {
      const ids = [];
      for (const part of ['objectives', 'principles', 'notExpected']) for (const g of l[part] || []) for (const id of (g && g.sources) || []) if (!ids.includes(id)) ids.push(id);
      out.levelInfo[l.level] = {objectives: l.objectives || [], principles: l.principles || [], notExpected: l.notExpected || [], evidence: l.evidence || null,
                                sources: strings(l.sources) ? l.sources : ids};
    }
  }
  if (isList(c.formats, f => f && typeof f.name === 'string' && [4, 5, 7, 9, 11].includes(f.players))) {
    out.formats = c.formats.map(f => ({name: f.name, players: f.players, goalkeeper: typeof f.goalkeeperDefault === 'boolean' ? f.goalkeeperDefault : f.goalkeeper !== false && f.players !== 4,
                                       choice: typeof f.goalkeeperChoice === 'boolean' ? f.goalkeeperChoice : f.players <= 5,
                                       note: f.aliasOf ? `assessed as ${f.aliasOf}v${f.aliasOf}` : typeof f.note === 'string' && !f.formations ? f.note : null}));
    const formations = {}, notes = {...CATALOGUE.formationNotes};
    for (const f of c.formats) {
      if (typeof f.note === 'string' && f.formations) out.formatNotes[f.players] = f.note;
      if (Array.isArray(f.formations)) {
        formations[f.players] = f.formations.map(x => typeof x === 'string' ? x : x && x.name).filter(x => typeof x === 'string');
        for (const x of f.formations) if (x && x.name && x.tag === 'PITCHIQ-PROPOSAL' && !notes[x.name]) notes[x.name] = 'common shape, not from the sources';
      }
      if (f.roles && typeof f.roles === 'object') out.rolesByFormat[f.players] = clone(f.roles);
    }
    if ([4, 5, 7, 9, 11].every(n => formations[n] && formations[n].length)) { out.formations = formations; out.formationNotes = notes; }
  }
  if (c.formations && typeof c.formations === 'object' && [4, 5, 7, 9, 11].every(n => Array.isArray(c.formations[n]))) out.formations = clone(c.formations);
  if (isList(c.styles, x => x && typeof x.id === 'string' && typeof x.name === 'string')) out.styles = c.styles.map(x => ({...x}));
  if (isList(c.federations, x => x && typeof x.id === 'string' && typeof x.name === 'string')) out.federations = c.federations.map(x => ({...x}));
  else if (strings(c.federations)) out.federations = c.federations.map(id => CATALOGUE.federations.find(x => x.id === id) || {id, name: id});
  if (strings(c.roles)) out.roles = c.roles.slice();
  if (c.formatDefaults && typeof c.formatDefaults === 'object' && !Array.isArray(c.formatDefaults)) out.formatDefaults = clone(c.formatDefaults);
  if (c.formatDefaultBasis && typeof c.formatDefaultBasis === 'object' && !Array.isArray(c.formatDefaultBasis)) out.formatDefaultBasis = clone(c.formatDefaultBasis);
  if (c.levelInfo && typeof c.levelInfo === 'object' && !Array.isArray(c.levelInfo)) Object.assign(out.levelInfo, clone(c.levelInfo));
  if (isList(c.sources, s => s && typeof s.id === 'string')) out.sources = Object.fromEntries(c.sources.map(s => [s.id, {...s}]));
  else if (c.sources && typeof c.sources === 'object' && !Array.isArray(c.sources)) out.sources = clone(c.sources);
  if (c.formationNotes && typeof c.formationNotes === 'object') out.formationNotes = {...out.formationNotes, ...c.formationNotes};
  for (const key of ['sourcesNote', 'disclaimer']) if (typeof c[key] === 'string' && c[key]) out.sourcesNote = c[key];
  return out;
}
function formatDefaultOf(cat, age, federation) {
  // The server's table when it has one (profiles.catalog(): formatDefaults[age][federation]; also accepted the other
  // way round), else DESIGN 2.2 (formatDefault).
  const t = cat && cat.formatDefaults, fed = federation || 'auto';
  const f = t && ((t[age] && (t[age][fed] || t[age].auto)) || (t[fed] && t[fed][age]) || (t.auto && t.auto[age]));
  return f && [4, 5, 7, 9, 11].includes(f.players) ? {players: f.players, goalkeeper: f.players >= 7 ? true : !!f.goalkeeper} : formatDefault(age, federation);
}
const fmtText = f => formatName(f) + (f.players <= 5 ? (f.goalkeeper ? ' with goalkeeper' : ' without goalkeeper') : '');
// profiles.json formatDefaults.basis (catalogue: formatDefaultBasis): the England and Canada defaults that are not the
// federation's own format.
const FORMAT_BASIS = {U6: {England: 'the U.S. Soccer format: the FA format below U7 was not part of the research',
                           Canada: 'the nearest PitchIQ format: Canada plays 3v3 at U6-U7 (CS-STANDARDS-2026), which PitchIQ does not assess'},
                      U7: {Canada: 'the nearest PitchIQ format: Canada plays 3v3 at U6-U7 (CS-STANDARDS-2026), which PitchIQ does not assess'},
                      U14: {Canada: 'the U.S. Soccer format: Canada’s Grassroots Standards stop at U13, so its U14 format was not verified'}};
function defaultSource(cat, age, federation) {
  // Whose format the default is, for the setup's note: U.S. Soccer for auto and US; with England or Canada chosen the
  // federation's own unless the basis (the server's, else this file's copy) says it is PitchIQ's nearest format. Before
  // fix round 1 a server table named nobody, so "4v4 without goalkeeper is the default for U8" read as England's.
  const fed = federation || 'auto';
  if (fed === 'auto' || fed === 'US' || (fed !== 'England' && fed !== 'Canada')) return 'U.S. Soccer';
  const server = cat && cat.formatDefaultBasis && typeof cat.formatDefaultBasis === 'object' ? cat.formatDefaultBasis : null;
  const b = (server || FORMAT_BASIS)[age], basis = b && typeof b[fed] === 'string' ? b[fed] : null;
  return basis || (fed === 'England' ? 'England' : 'Canada Soccer');
}
function selectEl(options, value, onchange, aria) {
  const s = el('select');
  for (const [v, text] of options) { const o = el('option', null, text); o.value = v; s.append(o); }
  s.value = value === null || value === undefined ? '' : String(value);
  if (aria) s.setAttribute('aria-label', aria);
  s.addEventListener('change', () => onchange(s.value));
  return s;
}
function fieldEl(label, ...nodes) {
  const l = el('label', 'cr-field');
  l.append(el('span', null, label), ...nodes);
  return l;
}
const evidenceRule = ev => `A work item needs at least ${ev.minOpportunities} moments and ${ev.minFailingSpells} separate spells` +
  (isNum(ev.minFailureShare) ? `, with the behaviour missing in at least ${pct(ev.minFailureShare)} of the moments` : ', judged against your declared style') +
  `, over at least ${Math.round(ev.minPhaseSeconds)} s of the relevant phase; at most ${ev.maxWorkItems} ${ev.maxWorkItems === 1 ? 'is' : 'are'} shown. ` +
  `A strength needs ${ev.strengthSpells} separate spells where the behaviour was shown, in at least ${pct(ev.strengthShare)} of the moments.`;

function el(tag, cls, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined && text !== null) e.textContent = String(text);
  return e;
}
async function saveFile(url) {
  // The PDF is fetched first, so a refusal (no report, no PDF writer) is shown as a message instead of downloaded.
  const response = await fetch(url);
  if (!response.ok) {
    let message = response.statusText || `HTTP ${response.status}`;
    try { message = (await response.json()).error || message; } catch (e) { /* not JSON */ }
    throw new Error(message);
  }
  const blob = await response.blob(), name = (/filename="?([^";]+)"?/.exec(response.headers.get('Content-Disposition') || '') || [])[1] || 'pitchiq-coaching.pdf';
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob); a.download = name; document.body.append(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}
const drillName = d => (d && (d.name || d.title)) || (d && d.id ? words(d.id) : 'Drill');
const verdictName = v => (VERDICTS.find(x => x[0] === v) || [0, String(v)])[1];
function button(text, cls, onclick, title) {
  const b = el('button', cls, text);
  b.type = 'button';
  if (title) b.title = title;
  if (onclick) b.onclick = onclick;
  return b;
}

// ---------- display labels ----------
function labelMap(result) {
  // id -> display of the result as it is now (store.labels_of): an identity a correction excluded or the tracker
  // withdrew shows as 'excluded (was A-03)', since after the renumbering its bare label belongs to another player (on a
  // BC60 copy with A-03 excluded, A-04 shows as A-03); a merged identity shows the label of the one it was merged into.
  const out = new Map();
  if (!result) return out;
  const m = result.match || {};
  for (const key of ['players', 'goalkeepers', 'referees']) for (const e of m[key] || []) if (e && e.id) out.set(e.id, e.display || e.id);
  for (const key of UNPLACED) {
    for (const e of m[key] || []) {
      if (!e || typeof e.id !== 'string' || out.has(e.id)) continue;
      out.set(e.id, `${key} (was ${typeof e.display === 'string' && e.display && e.display !== 'OUT' ? e.display : e.id})`);
    }
  }
  for (const s of ((result.roster || {}).slots) || []) if (s && s.id && !out.has(s.id)) out.set(s.id, s.display || s.id);
  for (const e of m.retired || []) if (e && e.id && !out.has(e.id)) out.set(e.id, out.get(e.retiredInto) || e.retiredInto || e.id);
  return out;
}
function plainLabels(text, labels) {
  return String(text === undefined || text === null ? '' : text).replace(ID_PATTERN, id => labels.has(id) ? labels.get(id) : id);
}
function labelNodes(text, labels) {
  // The text as nodes, every known id shown as its display label (the id in the tooltip).
  const s = String(text === undefined || text === null ? '' : text), out = [];
  let last = 0;
  for (const m of s.matchAll(ID_PATTERN)) {
    if (!labels.has(m[0])) continue;
    if (m.index > last) out.push(document.createTextNode(s.slice(last, m.index)));
    const shown = labels.get(m[0]), span = el('span', 'cr-id', shown);
    span.title = shown === m[0] ? m[0] : `${shown} (internal id ${m[0]})`;
    out.push(span);
    last = m.index + m[0].length;
  }
  if (last < s.length || !out.length) out.push(document.createTextNode(s.slice(last)));
  return out;
}

// ---------- drawing: one painter, two pens (SVG for the report, canvas for the board and the video) ----------
function svgPen(svg) {
  const add = (tag, attrs) => {
    const e = document.createElementNS(SVG_NS, tag);
    for (const [k, v] of Object.entries(attrs)) if (v !== undefined && v !== null) e.setAttribute(k, String(v));
    svg.append(e);
    return e;
  };
  const paint = s => ({fill: s.fill || 'none', 'fill-opacity': s.fillOpacity, stroke: s.stroke || 'none', 'stroke-opacity': s.strokeOpacity, 'stroke-width': s.width ? r2(s.width) : undefined,
                       'stroke-dasharray': s.dash ? s.dash.map(r2).join(' ') : undefined, 'stroke-linecap': 'round', 'stroke-linejoin': 'round'});
  return {
    circle: (x, y, r, s) => add('circle', {cx: r2(x), cy: r2(y), r: r2(r), ...paint(s)}),
    ellipse: (x, y, rx, ry, s) => add('ellipse', {cx: r2(x), cy: r2(y), rx: r2(rx), ry: r2(ry), ...paint(s)}),
    rect: (x, y, w, h, s) => add('rect', {x: r2(x), y: r2(y), width: r2(w), height: r2(h), ...paint(s)}),
    poly: (pts, s) => pts.length < 2 ? null : add(s.close ? 'polygon' : 'polyline', {points: pts.map(p => `${r2(p[0])},${r2(p[1])}`).join(' '), ...paint(s)}),
    text: (x, y, str, s) => {
      const e = add('text', {x: r2(x), y: r2(y), 'font-size': r2(s.size), 'text-anchor': s.anchor || 'middle', fill: s.fill || TOKENS.text, 'font-weight': s.weight || 600,
                             'font-family': 'Inter, Segoe UI, system-ui, sans-serif', stroke: s.halo, 'stroke-width': s.halo ? r2(s.size * .28) : undefined,
                             'paint-order': s.halo ? 'stroke' : undefined, 'stroke-linejoin': s.halo ? 'round' : undefined});
      e.textContent = str;
      return e;
    },
  };
}
function canvasPen(ctx) {
  const run = (s, path) => {
    ctx.save();
    ctx.lineCap = 'round'; ctx.lineJoin = 'round';
    ctx.beginPath(); path();
    if (s.fill && s.fill !== 'none') { ctx.globalAlpha = isNum(s.fillOpacity) ? s.fillOpacity : 1; ctx.fillStyle = s.fill; ctx.fill(); ctx.globalAlpha = 1; }
    if (s.stroke && s.stroke !== 'none') { ctx.setLineDash(s.dash || []); ctx.lineWidth = s.width || 1; ctx.strokeStyle = s.stroke; ctx.globalAlpha = isNum(s.strokeOpacity) ? s.strokeOpacity : 1; ctx.stroke(); ctx.globalAlpha = 1; }
    ctx.restore();
  };
  return {
    circle: (x, y, r, s) => run(s, () => ctx.arc(x, y, r, 0, Math.PI * 2)),
    ellipse: (x, y, rx, ry, s) => run(s, () => ctx.ellipse(x, y, rx, ry, 0, 0, Math.PI * 2)),
    rect: (x, y, w, h, s) => run(s, () => ctx.rect(x, y, w, h)),
    poly: (pts, s) => { if (pts.length < 2) return; run(s, () => { pts.forEach(([x, y], i) => i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)); if (s.close) ctx.closePath(); }); },
    text: (x, y, str, s) => {
      ctx.save();
      ctx.font = `${s.weight || 600} ${Math.round(s.size)}px Inter, Segoe UI, system-ui, sans-serif`;
      ctx.textAlign = s.anchor === 'start' ? 'left' : s.anchor === 'end' ? 'right' : 'center'; ctx.textBaseline = 'alphabetic';
      if (s.halo) { ctx.lineWidth = s.size * .28; ctx.strokeStyle = s.halo; ctx.lineJoin = 'round'; ctx.strokeText(str, x, y); }
      ctx.fillStyle = s.fill || TOKENS.text; ctx.fillText(str, x, y);
      ctx.restore();
    },
  };
}
const arc = (cx, cy, r, a0, a1, n = 24) => Array.from({length: n + 1}, (_, k) => { const a = a0 + (a1 - a0) * k / n; return [cx + r * Math.cos(a), cy + r * Math.sin(a)]; });

function alongAxis(points) {
  // Points in the order of their main direction: a unit's line runs across its players, not in id order.
  if (points.length < 3) return points.slice();
  const n = points.length, mx = points.reduce((a, p) => a + p[0], 0) / n, my = points.reduce((a, p) => a + p[1], 0) / n;
  let sxx = 0, syy = 0, sxy = 0;
  for (const [x, y] of points) { sxx += (x - mx) ** 2; syy += (y - my) ** 2; sxy += (x - mx) * (y - my); }
  const a = .5 * Math.atan2(2 * sxy, sxx - syy), ux = Math.cos(a), uy = Math.sin(a);
  return points.slice().sort((p, q) => ((p[0] - mx) * ux + (p[1] - my) * uy) - ((q[0] - mx) * ux + (q[1] - my) * uy));
}

// Drawing order: areas under lines, lines under markers, text last. The v2 annotation types (DESIGN 5.5) slot in so
// that a suggested area never hides the players and the relation lines run under the markers they join.
const ORDER = ['band', 'zone', 'channel', 'suggestedZone', 'unit', 'measure', 'bracket', 'relation', 'arrow', 'moveArrow', 'suggested', 'player', 'support',
               'focus', 'label', 'caption'];
const centroid = q => q.length ? [q.reduce((a, p) => a + p[0], 0) / q.length, q.reduce((a, p) => a + p[1], 0) / q.length] : null;
function hatch(q, spacing) {
  // Diagonal hatch lines (45 deg) inside a polygon as segments: each line x - y = c is cut at the polygon's edges and the
  // inside pieces are kept (even-odd), so the SVG and the canvas pen draw the same hatch without clipping paths.
  if (!(spacing > 0) || q.length < 3) return [];
  const d = q.map(([x, y]) => x - y), lo = Math.min(...d), hi = Math.max(...d), out = [];
  for (let c = lo + spacing / 2; c < hi && out.length < 400; c += spacing) {
    const xs = [];
    for (let i = 0; i < q.length; i++) {
      const [x0, y0] = q[i], [x1, y1] = q[(i + 1) % q.length], f0 = x0 - y0 - c, f1 = x1 - y1 - c;
      if ((f0 < 0) !== (f1 < 0)) { const t = f0 / (f0 - f1); xs.push([x0 + (x1 - x0) * t, y0 + (y1 - y0) * t]); }
    }
    xs.sort((a, b) => a[0] - b[0]);
    for (let k = 0; k + 1 < xs.length; k += 2) out.push([xs[k], xs[k + 1]]);
  }
  return out;
}

function paint(pen, spec, o) {
  // spec: a DiagramSpec (o.mode 'pitch', normalized pitch coordinates) or a VideoOverlaySpec (o.mode 'video', source
  // pixels). o: {toPx, line, font, radius (px), colours, labels (Map id -> display), positions (Map id -> spec
  // coordinates: the board's drawn markers, or the live boxes' feet), boxes (Map id -> box in spec pixels), box (spec
  // box -> px), players (draw the markers; the board draws its own), palette (annotationPalette), captions (false: the
  // page shows the caption as a chip instead), frame ([x0, y0, x1, y1] px: where a caption may stand), adjustments
  // ([{polygon, label}]: the coach's own areas, drawn as a thin solid outline beside PitchIQ's)}.
  // Elements a tier may not carry are skipped (TIER_ELEMENTS), and at most ELEMENT_BUDGET of each new type are drawn.
  const els = spec && Array.isArray(spec.elements) ? spec.elements.filter(e => e && typeof e.type === 'string') : [];
  const video = o.mode === 'video', P = (x, y) => o.toPx(x, y), labels = o.labels || new Map(), label = id => labels.get(id) || id;
  const own = new Map(), teams = new Map(), boxes = new Map();
  for (const e of els) if (e.type === 'player' && isNum(e.x) && isNum(e.y)) { own.set(e.id, [e.x, e.y]); teams.set(e.id, e.team); }
  for (const e of els) {
    if ((e.type === 'focus' || e.type === 'support') && isNum(e.x) && isNum(e.y) && !own.has(e.id)) { own.set(e.id, [e.x, e.y]); if (e.team) teams.set(e.id, e.team); }
    if (Array.isArray(e.box) && e.box.length === 4 && e.box.every(isNum) && typeof e.id === 'string' && !boxes.has(e.id)) boxes.set(e.id, e.box);
  }
  const at = id => (o.positions && o.positions.get(id)) || own.get(id) || null;
  const teamOf = id => teams.get(id) || (/^[AB]-/.test(String(id)) ? String(id)[0] : spec.team);
  const colour = team => (o.colours && o.colours[team]) || TOKENS.neutral;
  const tone = style => style === 'space' ? TOKENS.slate : TOKENS.amber;
  const pt = p => Array.isArray(p) && isNum(p[0]) && isNum(p[1]) ? P(p[0], p[1]) : null;
  const halo = {stroke: TOKENS.halo};
  const pal = o.palette || annotationPalette(colour('A'), colour('B'));
  const boxPx = b => (o.box ? o.box(b) : b);
  const boxOf = e => { const b = (o.boxes && o.boxes.get(e.id)) || (Array.isArray(e.box) && e.box.length === 4 && e.box.every(isNum) ? e.box : boxes.get(e.id)); return b ? boxPx(b) : null; };
  // The tier this drawing is gated by: the overlay's own (or the caller's), 'diagram' for top-down drawings. A video
  // overlay of unknown tier keeps the v1 elements (made tier-aware by the v1 engine) and only the image-tier v2 ones.
  const gate = video ? (['world', 'pinned', 'image'].includes(spec.tier) ? spec.tier : ['world', 'pinned', 'image'].includes(o.tier) ? o.tier : null) : 'diagram';
  const allowed = type => { const t = TIER_ELEMENTS[type]; return !t || (gate ? t.includes(gate) : !V2_TYPES.has(type) || t.includes('image')); };
  const v2spec = els.some(e => e.type === 'focus' || e.type === 'support'), used = {};
  const haloDash = pal.reviewFallback ? [o.line * 1.6, o.line * 1.2] : null;
  const marker = (id, x, y, faint) => {
    // A player's marker on a diagram (the 'player' element's look): the people a v2 annotation is about carry it.
    pen.circle(x, y, o.radius, {fill: colour(teamOf(id)), fillOpacity: faint ? .5 : undefined, stroke: TOKENS.halo, width: o.line * .5});
    pen.text(x, y - o.radius * 1.85 - o.font * .2, label(id), {size: o.font * .75, halo: TOKENS.halo});
  };
  const placed = e => at(e.id) || (isNum(e.x) && isNum(e.y) ? [e.x, e.y] : null);
  const focus = e => {
    // The player being reviewed (or praised): the review colour around the feet (world, pinned), around the box
    // (image), or as a ring around the marker (diagram, board). A white fallback outline gets a dashed dark halo.
    const p = placed(e);
    if (!p) return;
    const [x, y] = P(p[0], p[1]), c = pal.review, w = e.secondary ? .7 : 1;
    if (video) {
      const box = boxOf(e);
      if (gate === 'image' && box) {
        const pad = o.line * 2, r = [box[0] - pad, box[1] - pad, box[2] + 2 * pad, box[3] + 2 * pad];
        pen.rect(r[0], r[1], r[2], r[3], {stroke: pal.halo, width: o.line * 3.2 * w, dash: haloDash});
        pen.rect(r[0], r[1], r[2], r[3], {stroke: c, width: o.line * 1.6 * w});
        if (e.label !== false) pen.text(r[0] + r[2] / 2, r[1] - o.font * .45, label(e.id), {size: o.font, fill: c, halo: TOKENS.halo});
      } else {
        const rx = box ? Math.max(box[2] * .85, o.radius * .7) : o.radius * 1.1, ry = rx * .34;
        pen.ellipse(x, y, rx, ry, {stroke: pal.halo, width: o.line * 3.4 * w, dash: haloDash});
        pen.ellipse(x, y, rx, ry, {stroke: c, width: o.line * 1.8 * w});
        if (e.label !== false) pen.text(x, (box ? box[1] : y - o.radius * 3.4) - o.font * .45, label(e.id), {size: o.font, fill: c, halo: TOKENS.halo});
      }
      return;
    }
    if (o.players) marker(e.id, x, y, false);
    const r = o.radius * (o.players ? 1.6 : 1.95);
    pen.circle(x, y, r, {stroke: pal.halo, width: o.line * 2.2 * w, dash: haloDash});
    pen.circle(x, y, r, {stroke: c, width: o.line * 1.1 * w});
  };
  const support = e => {
    // A teammate or opponent the annotation relies on: a subtle white mark, never the review colour.
    const p = placed(e);
    if (!p) return;
    const [x, y] = P(p[0], p[1]);
    if (video) {
      const box = boxOf(e);
      if (gate === 'image' && box) pen.rect(box[0] - o.line, box[1] - o.line, box[2] + 2 * o.line, box[3] + 2 * o.line, {stroke: '#ffffff', strokeOpacity: .7, width: o.line * .9});
      else {
        const rx = box ? Math.max(box[2] * .75, o.radius * .6) : o.radius, ry = rx * .32;
        pen.ellipse(x, y, rx, ry, {stroke: pal.halo, width: o.line * 2});
        pen.ellipse(x, y, rx, ry, {stroke: '#ffffff', strokeOpacity: .8, width: o.line * .9, dash: [o.line * 2, o.line * 1.4]});
      }
      return;
    }
    if (o.players) marker(e.id, x, y, false);
    pen.circle(x, y, o.radius * 1.45, {stroke: '#ffffff', strokeOpacity: .75, width: o.line * .55, dash: [o.line * 1.4, o.line * 1.1]});
  };
  const endpoint = (id, p) => {
    if (Array.isArray(p) && isNum(p[0]) && isNum(p[1])) return P(p[0], p[1]);
    if (id === 'zone') { const z = els.find(x => x.type === 'suggestedZone' && Array.isArray(x.polygon)), c = z && centroid(z.polygon.filter(q => Array.isArray(q) && isNum(q[0]) && isNum(q[1]))); return c ? P(c[0], c[1]) : null; }
    if (id === 'goal') return !video && (spec.attack === 'left' || spec.attack === 'right') ? P(spec.attack === 'right' ? 0 : 1, .5) : null;   // the team's own goal
    const q = typeof id === 'string' && id !== 'ball' ? at(id) : null;
    return q ? P(q[0], q[1]) : null;
  };
  const relation = e => {
    // A relationship that mattered (cover, gap, channel, pressure, behind the ball): a subtle white line, dashed when
    // the relation was missing, with its short value.
    const a = endpoint(e.a, e.pa), b = endpoint(e.b, e.pb);
    if (!a || !b) return;
    pen.poly([a, b], {stroke: pal.halo, strokeOpacity: .5, width: o.line * 1.8});
    pen.poly([a, b], {stroke: pal.relation, strokeOpacity: pal.relationOpacity, width: o.line * .9, dash: e.ok === false ? [o.line * 2.2, o.line * 1.8] : null});
    if (e.text) {
      const len = Math.hypot(b[0] - a[0], b[1] - a[1]) || 1, nx = -(b[1] - a[1]) / len * o.font * .9, ny = (b[0] - a[0]) / len * o.font * .9, mx = (a[0] + b[0]) / 2, my = (a[1] + b[1]) / 2;
      const shown = plainLabels(e.text, labels), [x, y] = freeSpot([[mx + nx, my + ny + o.font * .25, 'middle'], [mx - nx, my - ny + o.font * .25, 'middle']], shown, o.font * .72);
      pen.text(x, y, shown, {size: o.font * .72, fill: TOKENS.text, halo: TOKENS.halo});
    }
  };
  const arrowLine = (q, c, dashed) => {
    const [x1, y1] = q[q.length - 1], [x0, y0] = q[q.length - 2], ang = Math.atan2(y1 - y0, x1 - x0), h = o.font * .75;
    pen.poly(q, {...halo, width: o.line * 2.6});
    pen.poly(q, {stroke: c, width: o.line * 1.3, dash: dashed ? [o.line * 2.4, o.line * 1.8] : null});
    pen.poly([[x1, y1], [x1 - h * Math.cos(ang - .45), y1 - h * Math.sin(ang - .45)], [x1 - h * Math.cos(ang + .45), y1 - h * Math.sin(ang + .45)]], {close: true, fill: c, stroke: TOKENS.halo, width: o.line * .5});
  };
  // Labels of areas and relations stand where they hide the fewest players: of a few candidate spots, the one farthest
  // from every drawn marker and earlier label (in the preview of the example report the area's label above the area sat on
  // the reviewed player, who stands right above it).
  const spots = [];
  const freeSpot = (cands, text, size) => {
    // cands: [x, y (baseline), anchor]; a label is a segment of its estimated width at mid-height, and the spot whose
    // segment stays farthest from the markers and the earlier labels wins.
    const pts = spots.flatMap(s => [[s[0], s[2]], [s[1], s[2]], [(s[0] + s[1]) / 2, s[2]]]);
    for (const id of new Set([...own.keys(), ...(o.positions ? o.positions.keys() : [])])) { const q = at(id); if (q) pts.push(P(q[0], q[1])); }
    const width = String(text).length * size * .52;
    const seg = ([x, y, anchor]) => { const x0 = anchor === 'start' ? x : anchor === 'end' ? x - width : x - width / 2; return [x0, x0 + width, y - size * .35]; };
    const gap = (s, p) => Math.hypot(p[0] - Math.min(s[1], Math.max(s[0], p[0])), p[1] - s[2]);
    // A spot whose label would leave the drawing (o.bounds) comes last: the preview cut 'possible covering area' at the
    // left edge of a diagram whose area touched the touchline.
    const b = o.bounds, outside = s => !!b && (s[0] < b[0] || s[1] > b[2] || s[2] - size < b[1] || s[2] + size > b[3]);
    let best = cands[0], far = -Infinity;
    for (const c of cands) {
      const s = seg(c), d = (pts.length ? Math.min(...pts.map(p => gap(s, p))) : 1e6) - (outside(s) ? 1e7 : 0);
      if (d > far) { far = d; best = c; }
    }
    spots.push(seg(best));
    return best;
  };
  const zoneLabel = (text, q, c) => {
    if (!text) return;
    const xs = q.map(p => p[0]), ys = q.map(p => p[1]), x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(...ys), y1 = Math.max(...ys), cx = (x0 + x1) / 2, cy = (y0 + y1) / 2;
    const size = o.font * .66, shown = plainLabels(text, labels);
    const [x, y, anchor] = freeSpot([[cx, y0 - size * .6, 'middle'], [cx, y1 + size * 1.25, 'middle'], [x0 - size * .5, cy + size * .35, 'end'], [x1 + size * .5, cy + size * .35, 'start']], shown, size);
    pen.text(x, y, shown, {size, fill: c, halo: TOKENS.halo, anchor});
  };
  const area = (polygon, kind, c, heavy, text, solid) => {
    // A suggested area (DESIGN 5.3): a dotted outline over a dark halo with a light hatch; a 'point' is a small dashed
    // circle, never a dot. The coach's own area (solid) is a thin solid outline.
    const q = (Array.isArray(polygon) ? polygon : []).map(pt).filter(Boolean);
    if (q.length < 3) return;
    const [cx, cy] = centroid(q);
    if (solid) {
      pen.poly(q, {close: true, stroke: pal.halo, width: o.line * 1.6});
      pen.poly(q, {close: true, stroke: c, width: o.line * .7});
      zoneLabel(text, q, c);
      return;
    }
    if (kind === 'point') {
      const r = Math.max(o.radius * .8, ...q.map(p => Math.hypot(p[0] - cx, p[1] - cy)));
      pen.circle(cx, cy, r, {stroke: pal.halo, width: o.line * 2.2 * heavy});
      pen.circle(cx, cy, r, {stroke: c, width: o.line * 1.1, dash: [o.line * 1.8, o.line * 1.4]});
      zoneLabel(text, [[cx - r, cy - r], [cx + r, cy + r]], c);
      return;
    }
    for (const seg of hatch(q, o.line * 5)) pen.poly(seg, {stroke: c, strokeOpacity: .45, width: o.line * .55});
    pen.poly(q, {close: true, stroke: pal.halo, width: o.line * 2.2 * heavy});
    pen.poly(q, {close: true, fill: c, fillOpacity: .1, stroke: c, width: o.line * 1.1, dash: [o.line * .35, o.line * 1.35]});
    zoneLabel(text, q, c);
  };
  const caption = e => {
    // The short caption inside the drawing (top-down diagrams and the evidence screenshot); a strength gets a check.
    const box = o.frame;
    if (!box || o.captions === false) return;
    const strong = els.some(x => x.type === 'focus' && x.tone === 'strength'), text = (strong ? '\u2713 ' : '') + plainLabels(e.text, labels);
    const right = /right$/.test(e.anchor || ''), bottom = /^bottom/.test(e.anchor || ''), size = o.font * .8, pad = o.font * .6;
    pen.text(right ? box[2] - pad : box[0] + pad, bottom ? box[3] - pad : box[1] + pad + size, text, {size, fill: TOKENS.text, halo: TOKENS.halo, anchor: right ? 'end' : 'start'});
  };
  const bracket = e => {
    // A team width or length as a bracket beside the team (soccer/coaching/render.py _bracket, the PDF's drawing): end
    // ticks pointing back at the team, dashed leaders to the two players that set the extent, the value outside. A
    // measure with bracket true already stands where it goes (evidence.py places it BRACKET_OFFSET outside the team); a
    // 'bracket' element's ends are the extent's own and it is moved beyond the team on its side (or the roomier one).
    let a = pt(e.a), b = pt(e.b);
    if (!a || !b) return;
    const len = Math.hypot(b[0] - a[0], b[1] - a[1]);
    if (len < 1e-6) return;
    const ux = (b[0] - a[0]) / len, uy = (b[1] - a[1]) / len, team = [];
    for (const id of own.keys()) if (teamOf(id) === spec.team) { const q = pt(at(id)); if (q) team.push(q); }
    let nx = -uy, ny = ux;
    const side = (q, sx, sy) => (q[0] - a[0]) * sx + (q[1] - a[1]) * sy;
    if (e.type === 'bracket') {
      const want = {left: [-1, 0], right: [1, 0], top: [0, -1], bottom: [0, 1]}[e.side];
      const ext = s => Math.max(0, ...team.map(q => side(q, nx * s, ny * s)));
      let sign = want ? Math.sign(want[0] * nx + want[1] * ny) : 0;
      if (!sign) sign = ext(1) <= ext(-1) ? 1 : -1;      // the side the team reaches least far beyond the extent's ends
      nx *= sign; ny *= sign;
      const d = ext(1) + o.radius * 2.4;
      a = [a[0] + nx * d, a[1] + ny * d]; b = [b[0] + nx * d, b[1] + ny * d];
    } else if (team.length) {
      const cx = team.reduce((s, q) => s + q[0], 0) / team.length, cy = team.reduce((s, q) => s + q[1], 0) / team.length;
      if (side([cx, cy], nx, ny) > 0) { nx = -nx; ny = -ny; }   // outward: away from the team's centre
    }
    const tick = o.font * .45;
    for (const s of [{...halo, width: o.line * 2.4}, {stroke: TOKENS.text, width: o.line}]) {
      pen.poly([a, b], s);
      for (const p of [a, b]) pen.poly([p, [p[0] - nx * tick, p[1] - ny * tick]], s);
    }
    const ids = (Array.isArray(e.ids) ? e.ids : []).filter(id => typeof id === 'string' && at(id));
    for (const p of [a, b]) {
      // the end's player: the named one standing nearest along the bracket
      const near = ids.map(id => pt(at(id))).filter(Boolean).sort((q, r) => Math.abs(side(q, ux, uy) - side(p, ux, uy)) - Math.abs(side(r, ux, uy) - side(p, ux, uy)))[0];
      if (!near) continue;
      const from = [p[0] - nx * tick, p[1] - ny * tick], reach = (from[0] - near[0]) * nx + (from[1] - near[1]) * ny - o.radius * 1.4;
      if (reach > o.line) pen.poly([from, [from[0] - nx * reach, from[1] - ny * reach]], {stroke: TOKENS.muted, width: o.line * .6, dash: [o.line * 1.6, o.line * 1.3]});
    }
    if (e.text) {
      const mx = (a[0] + b[0]) / 2, my = (a[1] + b[1]) / 2, across = Math.abs(nx) > .7, txt = plainLabels(e.text, labels), size = o.font * .85;
      if (across) pen.text(mx + nx * o.font * .6, my + size * .35, txt, {size, halo: TOKENS.halo, anchor: nx > 0 ? 'start' : 'end'});
      else pen.text(mx, my + (ny > 0 ? size * 1.15 : -size * .45), txt, {size, halo: TOKENS.halo});
    }
  };
  for (const type of ORDER) for (const e of els) {
    if (e.type !== type || !allowed(type)) continue;
    if (ELEMENT_BUDGET[type]) { used[type] = (used[type] || 0) + 1; if (used[type] > ELEMENT_BUDGET[type]) continue; }
    if (type === 'band' && !video && isNum(e.from) && isNum(e.to)) {
      const c = tone(e.style), q = e.axis === 'y' ? [[0, e.from], [1, e.from], [1, e.to], [0, e.to]] : [[e.from, 0], [e.to, 0], [e.to, 1], [e.from, 1]];
      pen.poly(q.map(p => P(p[0], p[1])), {close: true, fill: c, fillOpacity: .16, stroke: c, width: o.line * .7, dash: [o.line * 3, o.line * 2.5]});
    } else if (type === 'zone' && Array.isArray(e.polygon)) {
      const q = e.polygon.map(pt).filter(Boolean), c = tone(e.style);
      if (q.length >= 3) pen.poly(q, {close: true, fill: c, fillOpacity: .2, stroke: c, width: o.line * .8, dash: [o.line * 2.5, o.line * 2]});
    } else if (type === 'unit' && Array.isArray(e.ids)) {
      const q = alongAxis(e.ids.map(at).filter(Boolean)).map(p => P(p[0], p[1]));
      if (q.length >= 2) { pen.poly(q, {...halo, width: o.line * 2.6}); pen.poly(q, {stroke: colour(teamOf(e.ids[0])), width: o.line * 1.4}); }
    } else if ((type === 'measure' && e.bracket) || type === 'bracket') {
      if (!video) bracket(e);
      else {      // on the video a bracket sits where its ends say (render.py's video overlay): the line and its value
        const a = pt(e.a), b = pt(e.b);
        if (!a || !b) continue;
        for (const s of [{...halo, width: o.line * 2.4}, {stroke: TOKENS.text, width: o.line}]) pen.poly([a, b], s);
        if (e.text) pen.text((a[0] + b[0]) / 2, (a[1] + b[1]) / 2 - o.font * .3, plainLabels(e.text, labels), {size: o.font * .85, halo: TOKENS.halo});
      }
    } else if (type === 'measure') {
      const a = pt(e.a), b = pt(e.b);
      if (!a || !b) continue;
      const len = Math.hypot(b[0] - a[0], b[1] - a[1]) || 1, nx = -(b[1] - a[1]) / len, ny = (b[0] - a[0]) / len, tick = o.font * .45;
      for (const s of [{...halo, width: o.line * 2.4}, {stroke: TOKENS.text, width: o.line}]) {
        pen.poly([a, b], s);
        for (const p of [a, b]) pen.poly([[p[0] - nx * tick, p[1] - ny * tick], [p[0] + nx * tick, p[1] + ny * tick]], s);
      }
      if (e.text) pen.text((a[0] + b[0]) / 2 + nx * o.font * .9, (a[1] + b[1]) / 2 + ny * o.font * .9 + o.font * .35, plainLabels(e.text, labels), {size: o.font * .85, halo: TOKENS.halo});
    } else if (type === 'arrow' && Array.isArray(e.points)) {
      const q = e.points.map(pt).filter(Boolean);
      if (q.length < 2) continue;
      const c = colour(teamOf(e.id)), [x1, y1] = q[q.length - 1], [x0, y0] = q[q.length - 2], ang = Math.atan2(y1 - y0, x1 - x0), h = o.font * .75;
      pen.poly(q, {...halo, width: o.line * 2.6}); pen.poly(q, {stroke: c, width: o.line * 1.3});
      pen.poly([[x1, y1], [x1 - h * Math.cos(ang - .45), y1 - h * Math.sin(ang - .45)], [x1 - h * Math.cos(ang + .45), y1 - h * Math.sin(ang + .45)]], {close: true, fill: c, stroke: TOKENS.halo, width: o.line * .5});
    } else if (type === 'suggested' && isNum(e.x) && isNum(e.y)) {
      const [x, y] = P(e.x, e.y), r = video ? o.radius * 1.2 : o.radius * 1.35;
      pen.circle(x, y, r, {stroke: TOKENS.slate, width: o.line, dash: [o.line * 2, o.line * 1.6]});
      pen.text(x, y - r - o.font * .35, e.label || 'COACHING ILLUSTRATION', {size: o.font * .62, fill: TOKENS.slate, halo: TOKENS.halo});
    } else if (type === 'channel' && Array.isArray(e.polygon)) {
      // The area the finding is about (a vacated channel, a gap, the far side): the review colour, faint.
      const q = e.polygon.map(pt).filter(Boolean);
      if (q.length >= 3) pen.poly(q, {close: true, fill: pal.review, fillOpacity: .12, stroke: pal.review, strokeOpacity: .75, width: o.line * .8, dash: [o.line * 2.5, o.line * 2]});
    } else if (type === 'suggestedZone' && Array.isArray(e.polygon)) {
      area(e.polygon, e.kind, pal.zone, pal.zoneFallback ? 1.6 : 1, e.label);
      for (const adj of o.adjustments || []) if (adj && (!adj.alternative || adj.alternative === e.alternative)) area(adj.polygon, 'zone', '#ffffff', 1, adj.label || 'coach\u2019s area', true);
    } else if (type === 'relation') {
      relation(e);
    } else if (type === 'moveArrow') {
      const a = pt(e.from), b = pt(e.to);
      if (a && b) arrowLine([a, b], pal.zone, true);
    } else if (type === 'support') {
      support(e);
    } else if (type === 'focus') {
      focus(e);
    } else if (type === 'caption' && typeof e.text === 'string' && e.text) {
      caption(e);
    } else if (type === 'player' && isNum(e.x) && isNum(e.y)) {
      const p = at(e.id) || [e.x, e.y], [x, y] = P(p[0], p[1]), c = colour(e.team || teamOf(e.id)), observed = !e.kind || e.kind === 'O';
      if (video) {
        const box = boxOf(e), rx = box ? Math.max(box[2] * .75, o.radius * .6) : o.radius, ry = rx * .32;
        pen.ellipse(x, y, rx, ry, {...halo, width: o.line * 2.2});
        pen.ellipse(x, y, rx, ry, {stroke: c, width: o.line * (e.emphasis ? 1.4 : .9), dash: observed ? null : [o.line * 2, o.line * 1.6]});
        if (e.emphasis) pen.text(x, (box ? box[1] : y - o.radius * 3.4) - o.font * .4, label(e.id), {size: o.font, fill: TOKENS.text, halo: TOKENS.halo});
      } else if (o.players) {
        // In a v2 diagram the people the annotation is not about are drawn faint (DESIGN 5.5).
        pen.circle(x, y, o.radius, observed ? {fill: c, fillOpacity: v2spec && !e.emphasis ? .5 : undefined, stroke: TOKENS.halo, width: o.line * .5} : {stroke: c, width: o.line * .6, dash: [o.line * 1.5, o.line * 1.2]});
        if (e.emphasis) pen.circle(x, y, o.radius * 1.65, {stroke: TOKENS.text, width: o.line * .7});
        pen.text(x, y - o.radius * (e.emphasis ? 1.75 : 1.25) - o.font * .2, label(e.id), {size: o.font * .75, halo: TOKENS.halo});
      } else if (e.emphasis) {
        pen.circle(x, y, o.radius * 1.9, {stroke: TOKENS.text, width: o.line * .8});
      }
    } else if (type === 'label' && pt(e.at) && e.text) {
      const [x, y] = pt(e.at);
      pen.text(x, y, plainLabels(e.text, labels), {size: o.font * .8, halo: TOKENS.halo, fill: TOKENS.text});
    }
  }
}

function pitchLines(pen, L, W, M) {
  // The board's pitch (tactical.js drawPitch), in metres.
  const P = (global.Tactical && global.Tactical.PITCH) || {penaltyDepth: 16.5, penaltyHalf: 20.16, goalDepth: 5.5, goalHalf: 9.16, spot: 11, circle: 9.15, goalWidth: 7.32, goalNet: 2.2, cornerArc: 1};
  pen.rect(0, 0, L + 2 * M, W + 2 * M, {fill: '#1e6b36'});
  for (let i = 0; i < 10; i++) pen.rect(M + L * i / 10, M, L / 10 + .05, W, {fill: i % 2 ? '#2b8a46' : '#30944d'});
  const s = {stroke: '#f4f7f4', width: .22};
  pen.rect(M, M, L, W, s);
  pen.poly([[M + L / 2, M], [M + L / 2, M + W]], s);
  pen.poly(arc(M + L / 2, M + W / 2, P.circle, 0, Math.PI * 2, 48), s);
  pen.circle(M + L / 2, M + W / 2, .35, {fill: '#f4f7f4'});
  for (const [x0, sign] of [[0, 1], [L, -1]]) {
    for (const [depth, half] of [[P.penaltyDepth, P.penaltyHalf], [P.goalDepth, P.goalHalf]]) {
      const fx = x0 + sign * depth;
      pen.poly([[M + x0, M + W / 2 - half], [M + fx, M + W / 2 - half], [M + fx, M + W / 2 + half], [M + x0, M + W / 2 + half]], s);
    }
    pen.circle(M + x0 + sign * P.spot, M + W / 2, .3, {fill: '#f4f7f4'});
    const spread = Math.acos((P.penaltyDepth - P.spot) / P.circle), cx = M + x0 + sign * P.spot;
    pen.poly(sign > 0 ? arc(cx, M + W / 2, P.circle, -spread, spread) : arc(cx, M + W / 2, P.circle, Math.PI - spread, Math.PI + spread), s);
    pen.rect(sign > 0 ? M - P.goalNet : M + L, M + W / 2 - P.goalWidth / 2, P.goalNet, P.goalWidth, {...s, fill: '#ffffff', fillOpacity: .13});
  }
}

function svgRoot(viewBox, cls, title) {
  const svg = document.createElementNS(SVG_NS, 'svg');
  svg.setAttribute('viewBox', viewBox);
  svg.setAttribute('class', cls);
  svg.setAttribute('role', 'img');
  if (title) svg.setAttribute('aria-label', title);
  return svg;
}

function svgDiagram(spec, colours, labels, opts) {
  // The top-down diagram of a DiagramSpec as SVG (in metres; the board's 105 x 68 m with its margin). opts: {palette,
  // adjustments (the coach's areas)}.
  // A v2 caption stands in a strip above the pitch (CAPTION_STRIP metres), never over it: in the preview a caption over
  // the pitch ran under the reviewed player standing near the touchline.
  const L = BOARD.length, W = BOARD.width, M = BOARD.margin, ls = labels || new Map(), op = opts || {};
  const strip = (spec.elements || []).some(e => e && e.type === 'caption' && e.text) ? CAPTION_STRIP : 0;
  const svg = svgRoot(`0 ${-strip} ${L + 2 * M} ${W + 2 * M + 4 + strip}`, 'cr-diagram', plainLabels(spec.title || 'Diagram', ls)), pen = svgPen(svg);
  pen.rect(0, -strip, L + 2 * M, W + 2 * M + 4 + strip, {fill: '#14211b'});
  pitchLines(pen, L, W, M);
  paint(pen, spec, {mode: 'pitch', toPx: (x, y) => [M + x * L, M + y * W], line: .35, font: 3.1, radius: 1.6, colours, labels: ls, players: true,
                    palette: op.palette, adjustments: op.adjustments, frame: strip ? [M, -strip - .9, M + L, M] : [M, M, M + L, M + W],
                    bounds: [0, -strip, L + 2 * M, W + 2 * M]});
  if (spec.attack === 'left' || spec.attack === 'right') {
    const y = W + 2 * M + 1.8, right = spec.attack === 'right';
    pen.text(L / 2 + M, y, `Team ${spec.team || ''} attacks ${right ? '→' : '←'}`.replace('  ', ' '), {size: 2.6, fill: TOKENS.muted, weight: 500});
  }
  return svg;
}

function svgOverlay(spec, colours, labels, opts) {
  // A VideoOverlaySpec over the screenshot: the SVG has the source video's pixel size, stretched over the image.
  const W = spec.width, H = spec.height, ls = labels || new Map(), u = W / 640, op = opts || {};
  const svg = svgRoot(`0 0 ${W} ${H}`, 'cr-overlay', 'Overlay'), pen = svgPen(svg);
  svg.setAttribute('preserveAspectRatio', 'none');
  paint(pen, spec, {mode: 'video', toPx: (x, y) => [x, y], line: 1.6 * u, font: 13 * u, radius: 11 * u, colours, labels: ls, players: true,
                    palette: op.palette, frame: [0, 0, W, H], bounds: [0, 0, W, H]});
  return svg;
}

function validOverlay(spec) {
  return !!spec && isNum(spec.width) && isNum(spec.height) && spec.width > 0 && spec.height > 0 && Array.isArray(spec.elements);
}

// For the page: a finding's diagram on the 2D board (a tactical.overlays layer, under the board's own markers: the
// units run through the drawn markers, the players involved get a ring) and its overlay on the video canvas.
const boardPositions = state => {
  const positions = new Map();
  for (const p of (state && state.players) || []) if (p.id && isNum(p.x) && isNum(p.y)) positions.set(p.id, [p.x, p.y]);
  return positions;
};
const boardScale = g => ({line: Math.max(1.5, g.scale * .3), font: Math.max(11, g.playerRadius * 1.15), radius: g.playerRadius});
function drawBoard(ctx, g, spec, colours, state, labels, opts) {
  // The page shows the caption as a chip over the stage, so the canvas layers draw none (captions: false).
  if (!spec || !Array.isArray(spec.elements)) return;
  const op = opts || {};
  paint(canvasPen(ctx), spec, {mode: 'pitch', toPx: g.toPx, ...boardScale(g), colours, labels: labels || new Map(), positions: boardPositions(state), players: false,
                               palette: op.palette, adjustments: op.adjustments, captions: false, bounds: [0, 0, g.width, g.height]});
}
function drawVideo(ctx, spec, width, height, colours, labels, opts) {
  if (!validOverlay(spec)) return;
  const s = width / spec.width, sy = height / spec.height, u = spec.width / 640 * s, op = opts || {};
  paint(canvasPen(ctx), spec, {mode: 'video', toPx: (x, y) => [x * s, y * sy], box: b => [b[0] * s, b[1] * sy, b[2] * s, b[3] * sy], line: 1.6 * u, font: 13 * u, radius: 11 * u,
                               colours, labels: labels || new Map(), players: true, palette: op.palette, captions: false, bounds: [0, 0, width, height]});
}

// ---------- the annotation while a moment replays (DESIGN 5.6) ----------
const PLAYER_ID = /^(?:[AB]-\d{1,3}|GK-\d{1,3})$/;
function followOf(ev) {
  // The people a moment's annotation follows inside its window: its focus, support and player-to-player relations
  // (from the moment's diagram, else its overlay), within the element budget. A suggested area, a channel and a
  // movement arrow describe the key instant only and are not followed.
  const specs = [ev && ev.diagram, ev && ev.videoOverlay].filter(s => s && Array.isArray(s.elements));
  const out = {focus: [], support: [], relations: []}, seen = new Set();
  for (const spec of specs) for (const e of spec.elements) {
    if (!e || typeof e.type !== 'string') continue;
    if (e.type === 'focus' && PLAYER_ID.test(e.id) && !seen.has('f' + e.id) && out.focus.length < ELEMENT_BUDGET.focus) {
      seen.add('f' + e.id); out.focus.push({id: e.id, tone: e.tone === 'strength' ? 'strength' : 'review', secondary: !!e.secondary});
    } else if (e.type === 'support' && PLAYER_ID.test(e.id) && !seen.has('s' + e.id) && out.support.length < ELEMENT_BUDGET.support) {
      seen.add('s' + e.id); out.support.push({id: e.id, team: e.team === 'B' ? 'B' : 'A'});
    } else if (e.type === 'relation' && PLAYER_ID.test(e.a) && PLAYER_ID.test(e.b) && !seen.has('r' + e.a + '|' + e.b) && out.relations.length < ELEMENT_BUDGET.relation) {
      seen.add('r' + e.a + '|' + e.b); out.relations.push({a: e.a, b: e.b, kind: e.kind, ok: e.ok !== false});
    }
  }
  return out;
}
function followSpec(follow) {
  // A follow record as elements without coordinates: paint() places them at the live positions it is given. A
  // relation's value is not drawn while following (it was measured at the key instant only).
  const f = follow || {};
  return [].concat((f.relations || []).map(r => ({type: 'relation', a: r.a, b: r.b, kind: r.kind, ok: r.ok})),
                   (f.support || []).map(s => ({type: 'support', id: s.id, team: s.team})),
                   (f.focus || []).map(x => ({type: 'focus', id: x.id, tone: x.tone, secondary: x.secondary, label: true})));
}
function drawFollow(ctx, follow, o) {
  // o.mode 'video': o.people (Map id -> the frame's normalized box), o.width, o.height, o.tier;
  // o.mode 'board': o.g (board geometry), o.state (the board's drawn markers). Plus colours, labels, palette.
  if (!follow) return;
  const elements = followSpec(follow);
  if (!elements.length) return;
  if (o.mode === 'board') {
    paint(canvasPen(ctx), {elements}, {mode: 'pitch', toPx: o.g.toPx, ...boardScale(o.g), colours: o.colours, labels: o.labels || new Map(),
                                       positions: boardPositions(o.state), players: false, palette: o.palette, captions: false});
    return;
  }
  const W = o.width, H = o.height, u = W / 640, positions = new Map(), boxes = new Map();
  for (const [id, b] of o.people || []) {
    if (!Array.isArray(b) || b.length !== 4 || !b.every(isNum)) continue;
    const px = [b[0] * W, b[1] * H, b[2] * W, b[3] * H];
    boxes.set(id, px); positions.set(id, [px[0] + px[2] / 2, px[1] + px[3]]);
  }
  paint(canvasPen(ctx), {elements, tier: o.tier}, {mode: 'video', toPx: (x, y) => [x, y], line: 1.6 * u, font: 13 * u, radius: 11 * u, colours: o.colours,
                                                 labels: o.labels || new Map(), positions, boxes, players: true, palette: o.palette, captions: false});
}

function metricText(name, value, unit, estimated) {
  const known = METRICS[name], u = unit || (known ? known[1] : '');
  let v;
  if (!isNum(value)) v = value === undefined || value === null ? '–' : String(value);
  else if (u === 'm') v = `${value.toFixed(1)} m${estimated ? ' (est.)' : ''}`;
  else if (u === 'm2') v = `${Math.round(value)} m²${estimated ? ' (est.)' : ''}`;
  else if (u === 's') v = value >= 90 ? `${(value / 60).toFixed(1)} min` : `${value.toFixed(1)} s`;
  else if (u === 'share' || u === '%') v = `${(value * 100).toFixed(value * 100 >= 10 || Number.isInteger(value * 100) ? 0 : 1)}%`;
  else if (u === 'per_min') v = `${value.toFixed(2)} per min`;
  else if (u === 'count') v = String(Math.round(value));
  else if (u === 'ratio' || u === 'label' || !u) v = Number.isInteger(value) ? String(value) : value.toFixed(2);
  else v = `${Number.isInteger(value) ? value : value.toFixed(1)} ${u}`;
  return [known ? known[0] : words(name), v];
}

// ---------- coach-marked possession (user decision 1, Oct 21) ----------
// While watching, the coach says who had the ball: "our ball from here", "their ball from here", "dead ball from here",
// "end". possession-4 abstains on amateur and Veo footage (the clips most youth teams have), so every check that needs
// the phase was "not assessed" there; a coach span decides the phase where it is (situations.py: trusted as given, no
// change window at its edges, inside the analysed range). The spans are stored apart from the tracking (GET / POST
// /api/coaching/spans, schema kind 'coach_spans' per run: team letters, since "ours" depends on the team chosen),
// drawn on a lane of their own under the 2D match's possession strip (possession.js, the engine's estimate, is not
// touched: the two are never mixed), and named "marked by the coach" wherever the report uses them. They are a
// measurement input, so the report asks for Generate after a change (no new tracking).
const SPAN_STATES = ['A', 'B', 'dead'];
const SPAN_EDGE_S = .5;      // situations.COACH_EDGE_S: a span may reach this far outside the segment, and is clipped to it
const SPAN_MIN_S = .2;       // shorter phases are dropped: two clicks in a row, or what is left of a phase a newer mark covered
const SPAN_DEAD = '#050807'; // dead ball: near black, as the possession strip draws it (possession.js SHADE.dead)
const MARKS_KEY = 'pitchiq.coachMarks';   // localStorage: the marking bar open or closed (a per-viewer convenience)
const spanRecord = spans => ({version: 1, kind: 'coach-possession-spans', source: 'coach',
                              spans: spans.map(s => ({t0: r2(s.t0), t1: r2(s.t1), state: s.state}))});
function cleanSpans(raw, lo, hi) {
  // situations.clean_coach_spans in the page: numbers with t0 < t1, a known state, inside [lo, hi] give or take
  // SPAN_EDGE_S (then clipped), sorted by start; a span overlapping one kept before it is dropped. lo / hi null: no clip.
  const list = raw && Array.isArray(raw.spans) ? raw.spans : Array.isArray(raw) ? raw : [];
  const has = isNum(lo) && isNum(hi);
  const cand = list.filter(s => s && isNum(s.t0) && isNum(s.t1) && s.t1 > s.t0 && SPAN_STATES.includes(s.state) && (!has || (s.t0 >= lo - SPAN_EDGE_S && s.t1 <= hi + SPAN_EDGE_S)))
    .map(s => ({t0: has ? Math.max(s.t0, lo) : s.t0, t1: has ? Math.min(s.t1, hi) : s.t1, state: s.state}))
    .filter(s => s.t1 > s.t0).sort((a, b) => a.t0 - b.t0 || a.t1 - b.t1);
  const kept = [];
  for (const s of cand) if (!kept.length || s.t0 >= kept[kept.length - 1].t1 - 1e-9) kept.push(s);
  return kept;
}
function placeSpan(spans, span) {
  // The newest mark wins where it overlaps: an earlier span is cut around it (split in two when it covers it), pieces
  // shorter than SPAN_MIN_S are dropped, and touching spans of the same state are joined. So marking over a mistake
  // corrects it, and the list never holds an overlap (schema 'coach_spans').
  const out = [];
  for (const s of spans) {
    if (s.t1 <= span.t0 || s.t0 >= span.t1) { out.push({t0: s.t0, t1: s.t1, state: s.state}); continue; }
    if (s.t0 < span.t0) out.push({t0: s.t0, t1: span.t0, state: s.state});
    if (s.t1 > span.t1) out.push({t0: span.t1, t1: s.t1, state: s.state});
  }
  out.push({t0: span.t0, t1: span.t1, state: span.state});
  out.sort((a, b) => a.t0 - b.t0);
  const joined = [];
  for (const s of out) {
    if (s.t1 - s.t0 < SPAN_MIN_S - 1e-9) continue;
    const last = joined[joined.length - 1];
    if (last && last.state === s.state && Math.abs(last.t1 - s.t0) < 1e-6) last.t1 = s.t1; else joined.push(s);
  }
  return joined;
}

class PhaseMarks {
  // o: {bar (the marking row under the video), toggles (a host in each view's controls row for the open/close button),
  // lane, track, note (the coach lane of the possession panel), host (#viewBoard: reserves the lane's height), now()
  // (the video's time), seek(t), fit() (the page re-measures the stage), show(view)}. Every element is optional.
  constructor(view, o) {
    this.view = view; this.o = o || {};
    this.available = null; this.run = undefined; this.ask = 0; this.segment = null; this.fps = 25;
    this.spans = []; this.saved = null; this.open = null; this.version = 0; this.dirty = false; this.busy = false; this.again = false;
    this.message = ''; this.tone = null; this.listOpen = false; this.confirmAll = false; this.nowEl = null; this.openEl = null; this.lastTick = null;
    this.savedNow = false;          // a save of this page went through (the status then says so)
    this.expanded = false;
    try { this.expanded = !!global.localStorage && global.localStorage.getItem(MARKS_KEY) === '1'; } catch (error) { /* storage blocked: closed */ }
    if (this.o.track && typeof this.o.track.addEventListener === 'function') this.o.track.addEventListener('click', e => this.laneClick(e));
  }

  // ---------- data ----------
  load(result, info) {
    // Every showResult: a new run starts empty and asks the server; the same run (a correction) keeps unsaved marks.
    const run = this.view.runName(), frames = result && Array.isArray(result.frames) ? result.frames : [];
    if (run !== this.run) {
      this.run = run; this.spans = []; this.saved = null; this.open = null; this.version++; this.dirty = false; this.message = ''; this.tone = null; this.savedNow = false;
      this.listOpen = false; this.confirmAll = false;
    }
    this.segment = frames.length && isNum(frames[0].time) && isNum(frames[frames.length - 1].time) ? [frames[0].time, frames[frames.length - 1].time] : null;
    const fps = info && info.video && info.video.fps;
    this.fps = isNum(fps) && fps > 0 ? fps : 25;
    if (!result) { this.available = null; this.ask++; this.render(); return Promise.resolve(); }
    return this.fetch();
  }
  async fetch() {
    const seq = ++this.ask, run = this.run;
    let a = null;
    try { a = await this.view.api('/api/coaching/spans?' + new URLSearchParams({run: run || ''}).toString()); } catch (error) { a = null; }
    if (seq !== this.ask) return;
    const rec = a && typeof a === 'object' ? a.spans : null;
    // No such route (a server without coach spans) or no record: the tool stays hidden; nothing is guessed.
    if (!rec || typeof rec !== 'object' || !Array.isArray(rec.spans)) { this.available = false; this.render(); return; }
    this.available = true;
    if (a.segment && isNum(a.segment.t0) && isNum(a.segment.t1) && a.segment.t1 > a.segment.t0) this.segment = [a.segment.t0, a.segment.t1];
    if (!this.dirty) { this.spans = cleanSpans(rec, ...this.bounds()); this.saved = typeof a.saved === 'string' ? a.saved : null; }
    this.render();
  }
  bounds() { return this.segment ? this.segment.slice() : [null, null]; }
  team() { return this.view.teamKey(); }
  stateFor(kind) { const us = this.team(); return kind === 'own' ? us : kind === 'opp' ? (us === 'A' ? 'B' : 'A') : 'dead'; }
  stateName(state) { return state === 'dead' ? 'Dead ball' : state === this.team() ? `Our ball (Team ${state})` : `Their ball (Team ${state})`; }
  colourOf(state) { const c = this.view.colours(); return state === 'dead' ? SPAN_DEAD : c[state] || TOKENS.neutral; }
  here() {
    // The video's time as a mark: rounded to the hundredth, inside the analysed segment (a time just past either end is
    // taken as the end, as situations.py clips it); outside it nothing is marked.
    const t = Number(this.o.now ? this.o.now() : NaN), [lo, hi] = this.bounds();
    if (!isNum(t)) return {t: null, why: 'The video time is not known yet.'};
    if (lo !== null && (t < lo - SPAN_EDGE_S || t > hi + SPAN_EDGE_S))
      return {t: null, why: `The video is outside the analysed segment (${clock(lo)}–${clock(hi)}): who had the ball is marked only inside it.`};
    return {t: r2(lo === null ? t : Math.min(hi, Math.max(lo, t)))};
  }
  say(text, tone) { this.message = text || ''; this.tone = tone || null; }

  // ---------- marking ----------
  mark(kind) {
    // "... from here": closes the phase being marked (kept when it has some length) and starts the next one.
    const h = this.here(), state = this.stateFor(kind);
    if (h.t === null) { this.say(h.why, 'error'); this.render(); return; }
    if (this.open && this.open.state === state && h.t >= this.open.t0) {
      this.say(`Already marking ${this.stateName(state).toLowerCase()} since ${clock(this.open.t0)}.`); this.render(); return;
    }
    let kept = false;
    if (this.open && h.t - this.open.t0 >= SPAN_MIN_S) { this.spans = placeSpan(this.spans, {t0: this.open.t0, t1: h.t, state: this.open.state}); kept = true; }
    const dropped = this.open && !kept && h.t < this.open.t0;
    this.open = {t0: h.t, state};
    this.say(`${this.stateName(state)} from ${clock(h.t)}${dropped ? ' (the phase marked later in the video was dropped)' : ''}: press End, or mark what happens next, to keep it.`);
    if (kept) this.changed(); else this.render();
  }
  end() {
    if (!this.open) { this.say('Nothing is being marked: press one of the buttons when the phase starts.'); this.render(); return; }
    const h = this.here();
    if (h.t === null) { this.say(h.why, 'error'); this.render(); return; }
    const open = this.open;
    this.open = null;
    if (h.t - open.t0 < SPAN_MIN_S) {
      this.say(h.t < open.t0 ? `The video is before the start of that phase (${clock(open.t0)}), so it was not kept.` : 'That phase was too short to keep.');
      this.render(); return;
    }
    this.spans = placeSpan(this.spans, {t0: open.t0, t1: h.t, state: open.state});
    this.say(`${this.stateName(open.state)} ${clock(open.t0)}–${clock(h.t)} kept.`);
    this.changed();
  }
  editSpan(k, field, value) {
    // A row of the list: who had the ball, or a time typed as mm:ss.ss (source-video time). The edited phase wins where
    // it now overlaps another, as a new mark does.
    const s = this.spans[k];
    if (!s) return;
    const next = {t0: s.t0, t1: s.t1, state: s.state};
    if (field === 'state') { if (!SPAN_STATES.includes(value)) return; next.state = value; }
    else {
      const t = parseClock(value);
      if (t === null) { this.say('Type the time as mm:ss.ss (source-video time, as on the timestamps).', 'error'); this.render(); return; }
      next[field === 't1' ? 't1' : 't0'] = r2(t);
    }
    const [lo, hi] = this.bounds();
    if (lo !== null) { next.t0 = Math.max(lo, next.t0); next.t1 = Math.min(hi, next.t1); }
    if (!(next.t1 - next.t0 >= SPAN_MIN_S - 1e-9)) {
      this.say(`A phase ends after it starts (at least ${SPAN_MIN_S} s), inside the analysed segment${lo !== null ? ` (${clock(lo)}–${clock(hi)})` : ''}.`, 'error');
      this.render(); return;
    }
    this.spans = placeSpan(this.spans.filter((_, i) => i !== k), next);
    this.say('');
    this.changed();
  }
  removeSpan(k) {
    if (!this.spans[k]) return;
    this.spans = this.spans.filter((_, i) => i !== k);
    this.say('');
    this.changed();
  }
  removeAll() {
    // Two clicks: the first asks, the second removes every phase of this run.
    if (!this.confirmAll) { this.confirmAll = true; this.render(); return; }
    this.spans = []; this.open = null;
    this.say('Every phase you marked for this run was removed.');
    this.changed();
  }
  changed() { this.dirty = true; this.version++; this.confirmAll = false; this.render(); this.save(); }
  async save() {
    // The whole list, every time (a span edit can split or join others). One save at a time; a change made while one
    // is on its way is saved right after it. A failed save keeps the marks on the page and offers to try again.
    if (this.busy) { this.again = true; return; }
    if (this.available !== true) return;
    this.busy = true; this.again = false;
    const v = this.version, run = this.run, body = {run, spans: spanRecord(this.spans)};
    this.render();
    let a = null, failed = null;
    try { a = await this.view.post('/api/coaching/spans', body); } catch (error) { failed = error; }
    this.busy = false;
    if (run !== this.run) return;                    // another run was opened meanwhile: its own marks are shown
    if (failed) { this.say(`Not saved: ${failed.message}. Your marks stay on this page.`, 'error'); this.render(); return; }
    if (v === this.version && !this.again) {
      const rec = a && a.spans && Array.isArray(a.spans.spans) ? a.spans : null;
      if (rec) this.spans = cleanSpans(rec, ...this.bounds());
      this.dirty = false; this.saved = isoNow(); this.savedNow = true;
    }
    if (this.tone === 'error') this.say('');
    this.view.spansChanged(a);
    if (this.again || v !== this.version) { this.save(); return; }
    this.render();
  }

  // ---------- drawing ----------
  render() {
    const on = this.available === true && !!this.view.result;
    for (const host of this.o.toggles || []) this.renderToggle(host, on);
    const bar = this.o.bar;
    if (bar) {
      const show = on && this.expanded;
      bar.classList.toggle('off', !show);
      bar.replaceChildren();
      if (show) this.renderBar(bar);
    }
    this.renderLane(on);
    if (this.o.fit) this.o.fit();
  }
  setExpanded(on) {
    this.expanded = !!on;
    try { if (global.localStorage) global.localStorage.setItem(MARKS_KEY, this.expanded ? '1' : '0'); } catch (error) { /* storage blocked */ }
    this.render();
  }
  renderToggle(host, on) {
    // The toggle sits in the tracking and 2D controls rows only once the coach uses the marks (the bar is open, or this
    // run has marked phases); before, the report's "Mark who had the ball" buttons open the bar. A toggle that is always
    // there wrapped those rows onto another line where they were nearly full (critic S-11, measured on BC60: tracking
    // stage 408 instead of 442 px at 1440x900, 2D 318 instead of 356 px at 800x900), so a run nobody marked keeps the
    // checkout's layout at every width.
    host.replaceChildren();
    const show = on && (this.expanded || this.spans.length > 0 || !!this.open);
    host.hidden = !show;
    if (!show) return;
    const n = this.spans.length, b = button(this.expanded ? 'Hide possession marks' : 'Mark who has the ball', 'cm-toggle' + (this.expanded ? ' on' : ''), () => this.setExpanded(!this.expanded),
                                            'Mark who had the ball while you watch; used instead of PitchIQ’s estimate where you mark it');
    b.setAttribute('aria-pressed', String(this.expanded));
    b.dataset.marks = 'toggle';
    if (n && !this.expanded) b.append(el('span', 'cm-count', ` · ${n}`));
    host.append(b);
  }
  renderBar(bar) {
    const team = this.team(), cue = el('div', 'group');
    cue.append(el('span', 'label', 'Who has the ball'));
    const us = selectEl([['A', 'Team A is us'], ['B', 'Team B is us']], team, v => this.view.setTeam(v), 'Our team (the coaching report’s team)');
    us.dataset.marks = 'team';
    cue.append(us);
    const acts = el('div', 'group cm-acts');
    for (const [kind, text] of [['own', 'Our ball from here'], ['opp', 'Their ball from here'], ['dead', 'Dead ball from here']]) {
      const state = this.stateFor(kind), b = button('', 'cm-btn', () => this.mark(kind), `${this.stateName(state)} from the video’s current time`), dot = el('i', 'cm-dot');
      dot.style.background = this.colourOf(state);
      b.append(dot, document.createTextNode(text));
      b.dataset.mark = kind;
      if (this.open && this.open.state === state) { b.classList.add('on'); b.setAttribute('aria-pressed', 'true'); }
      acts.append(b);
    }
    const end = button('End', 'cm-btn', () => this.end(), 'End the phase being marked at the video’s current time');
    end.dataset.mark = 'end'; end.disabled = !this.open;
    acts.append(end);
    const info = el('div', 'group cm-info'), status = el('span', 'cm-status' + (this.tone === 'error' ? ' error' : ''));
    status.setAttribute('role', 'status');
    status.textContent = this.statusText();
    const list = button(`List (${this.spans.length})`, 'quiet', () => { this.listOpen = !this.listOpen; this.confirmAll = false; this.render(); }, 'Edit or remove the phases you marked');
    list.dataset.marks = 'list'; list.setAttribute('aria-expanded', String(this.listOpen));
    info.append(status, list);
    if (this.dirty && !this.busy && this.tone === 'error') { const again = button('Save again', 'primary', () => this.save()); again.dataset.marks = 'retry'; info.append(again); }
    bar.append(cue, acts, info);
    if (this.listOpen) bar.append(this.renderList());
  }
  statusText() {
    // What was just done (or what is being marked, or how many phases there are), then whether it is saved.
    const n = this.spans.length, parts = [];
    if (this.message) parts.push(this.message);
    else if (this.open) parts.push(`Marking ${this.stateName(this.open.state).toLowerCase()} since ${clock(this.open.t0)}: press End, or mark what happens next.`);
    else if (!n) parts.push('Nothing marked yet: press a button as the play changes. Your marks are used instead of PitchIQ’s estimate where you make them.');
    else parts.push(`${n} ${plural(n, 'phase')} marked by you.`);
    if (this.busy) parts.push('Saving…');
    else if (this.dirty && this.tone !== 'error') parts.push('Not saved yet.');
    else if (!this.dirty && this.savedNow) parts.push('Saved with this video: the report uses your marks after Generate (no new tracking).');
    return parts.join(' ');
  }
  renderList() {
    const box = el('div', 'cm-list');
    box.dataset.marks = 'rows';
    if (!this.spans.length) { box.append(el('p', 'cr-none', 'No phase marked for this run yet.')); return box; }
    const rows = el('div', 'cm-rows'), total = {A: 0, B: 0, dead: 0};
    this.spans.forEach((s, k) => {
      total[s.state] += s.t1 - s.t0;
      const row = el('div', 'cm-row'), sw = el('i', 'cm-dot');
      row.dataset.span = String(k);
      sw.style.background = this.colourOf(s.state);
      const st = selectEl(SPAN_STATES.map(x => [x, this.stateName(x)]), s.state, v => this.editSpan(k, 'state', v), `Who had the ball from ${clock(s.t0)}`);
      st.dataset.field = 'state';
      const from = el('input'), to = el('input');
      for (const [inp, key, label] of [[from, 't0', 'From'], [to, 't1', 'To']]) {
        inp.type = 'text'; inp.inputMode = 'decimal'; inp.value = clock(s[key]); inp.dataset.field = key;
        inp.setAttribute('aria-label', `${label}, source-video time (mm:ss.ss)`);
        inp.addEventListener('change', () => this.editSpan(k, key, inp.value));
      }
      const go = button('Go', 'quiet', () => { if (this.o.seek) this.o.seek(s.t0); }, `Show ${clock(s.t0)} on the video`);
      go.dataset.marks = 'go';
      const rm = button('Remove', 'quiet', () => this.removeSpan(k), 'Remove this phase');
      rm.dataset.marks = 'remove';
      const times = el('span', 'cm-times');        // from and to wrap together on a phone
      times.append(from, el('span', 'cm-sep', '–'), to);
      row.append(sw, st, times, el('span', 'cm-len', seconds(s.t1 - s.t0)), go, rm);
      rows.append(row);
    });
    const us = this.team(), them = us === 'A' ? 'B' : 'A', foot = el('div', 'cm-foot');
    foot.append(el('span', null, `Ours ${seconds(total[us])} · theirs ${seconds(total[them])} · dead ball ${seconds(total.dead)} · entered by you, kept apart from the tracking.`));
    const all = button(this.confirmAll ? `Remove all ${this.spans.length}? Click again` : 'Remove all', this.confirmAll ? 'danger' : 'quiet', () => this.removeAll());
    all.dataset.marks = 'remove-all';
    foot.append(all);
    box.append(rows, foot);
    return box;
  }
  laneSpan() {
    // The possession strip's time axis (possession.js: the analysed segment, its last frame one frame wide).
    const [lo, hi] = this.bounds();
    return lo === null ? null : [lo, Math.max(lo + 1 / this.fps, hi + 1 / this.fps)];
  }
  pct(t) { const s = this.laneSpan(); return s && isNum(t) ? Math.max(0, Math.min(100, 100 * (t - s[0]) / (s[1] - s[0]))) : null; }
  renderLane(on) {
    const lane = this.o.lane, track = this.o.track;
    if (!lane || !track) return;
    const show = on && !!this.laneSpan() && (this.spans.length > 0 || !!this.open);
    lane.hidden = !show;
    if (this.o.host && this.o.host.classList) this.o.host.classList.toggle('pz-coach-on', show);
    track.replaceChildren(); this.nowEl = null; this.openEl = null; this.lastTick = null;
    if (this.o.note) this.o.note.replaceChildren();
    if (!show) return;
    const team = this.team();
    for (const s of this.spans) {
      const i = el('i', `pz-cspan s-${s.state === 'dead' ? 'dead' : s.state === team ? 'own' : 'opp'}`), a = this.pct(s.t0), b = this.pct(s.t1);
      i.style.left = `${a.toFixed(3)}%`; i.style.width = `${Math.max(.15, b - a).toFixed(3)}%`; i.style.background = this.colourOf(s.state);
      i.title = `${this.stateName(s.state)} ${clock(s.t0)}–${clock(s.t1)}, marked by the coach`;
      i.dataset.state = s.state;
      track.append(i);
    }
    if (this.open) {
      this.openEl = el('i', 'pz-cspan open');
      this.openEl.style.left = `${this.pct(this.open.t0).toFixed(3)}%`; this.openEl.style.borderColor = this.colourOf(this.open.state);
      this.openEl.title = `${this.stateName(this.open.state)} being marked since ${clock(this.open.t0)}`;
      track.append(this.openEl);
    }
    this.nowEl = el('b', 'pz-cnow');
    track.append(this.nowEl);
    track.setAttribute('aria-label', `Possession marked by the coach: ${this.spans.length} ${plural(this.spans.length, 'phase')}; click to seek`);
    if (this.o.note) {
      const n = this.o.note, item = (state, text) => { const s = el('span', 'cm-key'), i = el('i'); i.style.background = this.colourOf(state); s.append(i, document.createTextNode(text)); return s; };
      n.append(el('span', 'pz-k', 'Coach'), el('span', 'cm-what', 'who had the ball, as you marked it (used instead of PitchIQ’s estimate where marked)'),
               item(team, `Team ${team} (us)`), item(team === 'A' ? 'B' : 'A', `Team ${team === 'A' ? 'B' : 'A'}`), item('dead', 'dead ball'));
    }
    this.tick(this.o.now ? this.o.now() : null, true);
  }
  tick(t, force) {
    // Every drawn frame: the lane's playhead and the phase being marked follow the video (a style write only when the
    // time moved by 0.04 s or more).
    if ((!this.nowEl && !this.openEl) || !isNum(t)) return;
    if (!force && this.lastTick !== null && Math.abs(t - this.lastTick) < .04) return;
    this.lastTick = t;
    const p = this.pct(t);
    if (this.nowEl) { this.nowEl.hidden = p === null; if (p !== null) this.nowEl.style.left = `${p.toFixed(3)}%`; }
    if (this.openEl && this.open) { const a = this.pct(this.open.t0); this.openEl.style.width = `${Math.max(0, (p === null ? a : p) - a).toFixed(3)}%`; }
  }
  laneClick(e) {
    const track = this.o.track, s = this.laneSpan(), r = track && typeof track.getBoundingClientRect === 'function' ? track.getBoundingClientRect() : null;
    if (!s || !r || !(r.width > 0) || !isNum(e && e.clientX) || !this.o.seek) return;
    const u = Math.max(0, Math.min(1, (e.clientX - r.left) / r.width));
    this.o.seek(Math.min(s[1] - .5 / this.fps, s[0] + u * (s[1] - s[0])));
  }
}

// ---------- the view ----------
class CoachingView {
  constructor(options) {
    this.main = options.main; this.side = options.side; this.banner = options.banner; this.c = options.controls;
    this.api = options.api; this.post = options.post; this.video = options.video; this.board = options.board;
    this.runName = options.run || (() => null);
    this.jumpTo = options.jump || (() => false);
    this.download = options.download || saveFile;
    this.media = options.media !== undefined ? options.media : (typeof global.matchMedia === 'function' ? global.matchMedia(NARROW) : null);
    this.home = this.side.parentElement || null;     // the grid column the evidence lives in on wide screens
    this.result = null; this.info = null; this.labels = new Map(); this.loadedRun = undefined;
    this.cache = new Map(); this.answer = null; this.seq = 0; this.gen = 0; this.dataGen = -1; this.active = false; this.loading = false;
    // The selection the shown answer belongs to (Export PDF saves that one), its cache key and JSON, and the gen the
    // report was last drawn for: an answer identical to the one drawn redraws only the bar and the banner.
    this.answerOptions = null; this.answerKey = null; this.answerSig = null; this.renderedGen = -1;
    this.selected = null; this.selectedEvidence = null; this.cards = new Map(); this.boxes = new Map(); this.drafts = new Map(); this.notes = new Map();
    this.timer = null; this.message = '';
    // Coaching v2. mode: 'v2' once the server answered GET /api/coaching/profiles, 'legacy' when it has no such route
    // (the v1 engine: one professional reference, three wordings), null before asking. The setup of each team is the
    // saved block (GET /api/coaching/settings), or the coach's unsaved draft; a legacy server keeps it on the page.
    this.mode = null; this.catalogue = CATALOGUE; this.catalogueAsk = null; this.catalogueError = null;
    this.setupAnswer = null; this.setupAsk = null; this.setupRun = undefined; this.setupDraft = {A: null, B: null}; this.localSaved = {A: null, B: null};
    this.setupOpen = null; this.setupPanel = null; this.setupMessage = ''; this.setupBusy = false; this.squadName = null;
    this.selectedMoment = null; this.momentRows = []; this.adjusting = null; this.adjustMessage = ''; this.drillBoxes = new Map();
    this.progressAsk = new Map(); this.progressOpen = null; this.spansSaved = false;
    // The coach-marked possession tool (index.html gives its elements; without them there is none).
    this.marks = options.marks ? new PhaseMarks(this, options.marks) : null;
    this.c.team.addEventListener('change', () => this.teamChanged());
    this.c.range.addEventListener('change', () => { this.syncRange(true); this.request(); });
    for (const input of [this.c.t0, this.c.t1]) input.addEventListener('change', () => this.request());
    if (this.c.context) this.c.context.addEventListener('change', () => this.request());
    this.c.generate.addEventListener('click', () => this.generate());
    this.c.pdf.addEventListener('click', () => this.exportPdf());
    for (const input of options.colourInputs || []) input.addEventListener('input', () => { this.renderSide(); if (this.marks) this.marks.render(); });
    if (this.media && typeof this.media.addEventListener === 'function') this.media.addEventListener('change', () => this.placeSide());
    this.syncRange(false);
    this.render();
  }

  label(id) { return this.labels.get(id) || id; }
  teamChanged() {
    // The report's team is also "us" for the possession marks.
    this.setupPanel = null; this.setupMessage = '';
    if (this.marks) this.marks.render();
    this.request();
  }
  setTeam(team) {
    if (team !== 'A' && team !== 'B') return;
    this.c.team.value = team;
    this.teamChanged();
  }
  tick(t) { if (this.marks) this.marks.tick(t); }
  spansChanged() {
    // Coach spans are a measurement input: every cached answer is dropped, and the report says to Generate.
    this.spansSaved = true; this.cache.clear(); this.gen++;
    if (this.active && this.result) this.request(); else this.renderSetup();
  }
  openMarks() {
    // From the report: open the marking bar under the video and show the video.
    if (!this.marks || this.marks.available !== true) return false;
    this.marks.setExpanded(true);
    if (this.marks.o.show) this.marks.o.show('tracking');
    return true;
  }
  colours() { return this.board && this.board.colours ? this.board.colours : {A: '#b98cff', B: '#f6b564'}; }
  fps() { return (this.info && this.info.video && this.info.video.fps) || 25; }
  bounds() { const f = this.result && this.result.frames; return f && f.length ? [f[0].time, f[f.length - 1].time] : [0, 0]; }

  // ---------- selection and data ----------
  load(result, info) {
    // Every showResult (a new run, or the same run after a correction): the answers cached for the earlier result are
    // dropped; unsaved reviews survive a correction of the same run, not a new run.
    const run = this.runName();
    if (run !== this.loadedRun) {
      this.drafts.clear(); this.notes.clear(); this.selected = null; this.selectedEvidence = null; this.loadedRun = run;
      this.setupAnswer = null; this.setupAsk = null; this.setupRun = undefined; this.setupDraft = {A: null, B: null}; this.setupOpen = null;
      this.setupPanel = null; this.setupMessage = ''; this.squadName = null; this.selectedMoment = null; this.adjusting = null;
    }
    this.result = result; this.info = info; this.labels = labelMap(result);
    if (run !== this.spansRun) { this.spansRun = run; this.spansSaved = false; this.progressAsk.clear(); this.progressOpen = null; }
    this.marksAsk = this.marks ? this.marks.load(result, info) : null;
    this.cache.clear(); this.answer = null; this.answerOptions = null; this.answerKey = null; this.answerSig = null;
    this.gen++; this.seq++; this.loading = false; this.stopPoll();     // an answer still in flight belongs to the earlier result
    this.syncRange(false);
    if (this.active && result) this.request(); else this.render();
  }

  setActive(on) {
    // Called when the coaching view is shown or hidden, and a no-op otherwise: a window resize re-marks the views, and
    // re-asking then rebuilt the whole report (5 resizes replaced the note a coach was typing in and every screenshot).
    // Shown again, the report is asked again, since a team-shape save on the Analytics tab (which reloads no result)
    // can have made it stale; the status request is cheap (the server hashes each result version once).
    on = !!on;
    if (on === this.active) return;
    this.active = on;
    if (!on) { this.stopPoll(); return; }
    this.cache.clear();
    if (this.result) this.request(); else this.render();
  }

  syncRange(fill) {
    const custom = this.c.range.value === 'custom';
    this.c.custom.hidden = !custom;
    if (this.result) {
      const [a, b] = this.bounds();
      this.c.custom.title = `Source-video time (the video’s own clock, as on the timestamps). The analysed segment runs ${clock(a)}–${clock(b)}.`;
      if (custom && fill) { this.c.t0.value = clock(a); this.c.t1.value = clock(b); }
    }
  }

  options() {
    const team = this.c.team.value === 'B' ? 'B' : 'A', out = {team, context: null, t0: null, t1: null};
    if (this.c.context) out.context = CONTEXTS[this.c.context.value] ? this.c.context.value : 'competitive';
    else if (this.mode !== 'v2') out.context = legacyContext((this.savedSetup(team) || {}).playingLevel);
    if (this.c.range.value === 'custom') {
      const a = parseClock(this.c.t0.value), b = parseClock(this.c.t1.value);
      if (a !== null && b !== null) { out.t0 = hundredths(Math.min(a, b)); out.t1 = hundredths(Math.max(a, b)); } else out.invalid = true;
    }
    return out;
  }
  key(o) { return `${o.team}|${o.context}|${o.t0}|${o.t1}`; }
  params(o, extra) {
    const p = new URLSearchParams({run: this.runName() || '', team: o.team});
    if (o.context) p.set('context', o.context);
    if (o.t0 !== null) { p.set('t0', o.t0.toFixed(2)); p.set('t1', o.t1.toFixed(2)); }
    for (const [k, v] of Object.entries(extra || {})) p.set(k, v);
    return p;
  }

  async request() {
    // Every request supersedes the ones still in flight, also one answered at once (from the cache, or an unreadable
    // range): their late answers fail the seq check. Before, a cache hit left an older request current: with team B
    // asked and A then shown from the cache, B's late answer replaced A's report while the selector said A, and the
    // stuck loading flag kept Generate disabled.
    this.stopPoll(); this.message = '';
    const seq = ++this.seq, gen = this.gen;
    if (!this.result) { this.loading = false; this.show(null, null, null, gen); return; }
    await this.prepare();
    if (seq !== this.seq || gen !== this.gen) return;
    const o = this.options(), key = this.key(o);
    if (o.invalid) {
      this.loading = false;
      this.show({state: 'error', error: 'Enter the custom range as two times (mm:ss.ss, source-video time) inside the analysed segment.', report: null, feedback: [], reasons: []}, o, null, gen);
      return;
    }
    if (this.cache.has(key)) { this.loading = false; this.show(this.cache.get(key), o, key, gen); return; }
    this.loading = true; this.renderBar();
    let data, failed = false;
    try {
      data = await this.api('/api/coaching?' + this.params(o).toString());
    } catch (error) {
      data = {state: 'error', error: failText(error), report: null, feedback: [], reasons: []}; failed = true;
    }
    if (seq !== this.seq || gen !== this.gen) return;     // a newer selection, result or save superseded this one
    // A build in progress, this selection's or another one's (busy), is asked again until it ends and never cached: a
    // cached 'busy' kept Generate disabled after the other build had finished, and nothing asked again.
    const waiting = !failed && (data.state === 'building' || !!data.busy);
    if (!failed && !waiting) this.cache.set(key, data);
    this.loading = false;
    this.show(data, o, key, gen);
    if (waiting && this.active) this.timer = setTimeout(() => { this.timer = null; this.request(); }, POLL_MS);
  }
  show(answer, options, key, gen) {
    // Shows an answer; one identical to the answer drawn for the same selection and result (a poll during a build, the
    // tab shown again) redraws only the bar and the banner, so a note being typed and the loaded screenshots stay.
    const sig = answer ? JSON.stringify(answer) : null;
    const same = key !== null && key === this.answerKey && sig === this.answerSig && this.renderedGen === gen;
    this.answer = answer; this.answerOptions = options; this.answerKey = key; this.answerSig = sig; this.dataGen = gen;
    const rolesNew = this.takeInferred(options && options.team, answer && answer.inferred);
    if (same) { this.renderBar(); this.renderBanner(); if (rolesNew) this.renderSetup(); } else this.render();
  }
  takeInferred(team, list) {
    // The measured roles the report answer carries once the team's roles file is stored (SERVER: answer.inferred). The
    // roles editor read them from GET /api/coaching/settings once per run, before the first Generate had measured
    // anything, and said "Measured: no stable role" for every player until the page was reloaded (CODE M3).
    if (this.mode !== 'v2' || (team !== 'A' && team !== 'B') || !Array.isArray(list)) return false;
    const now = Object.assign({}, (this.setupAnswer && this.setupAnswer.inferred) || {});
    if (JSON.stringify(now[team] || null) === JSON.stringify(list)) return false;
    now[team] = list.filter(r => r && typeof r.id === 'string');
    this.setupAnswer = Object.assign({}, this.setupAnswer || {}, {inferred: now});
    return true;
  }
  stopPoll() { if (this.timer) { clearTimeout(this.timer); this.timer = null; } }

  async generate() {
    const a = this.answer;
    if (!this.result || this.loading || (a && (a.state === 'building' || a.busy))) return;
    const o = this.options(), body = {run: this.runName(), team: o.team};
    if (o.context) body.context = o.context;
    if (o.invalid) return;
    if (o.t0 !== null) { body.t0 = o.t0; body.t1 = o.t1; }     // the hundredths the status request asked for
    this.message = ''; this.c.generate.disabled = true;
    this.setupMessage = '';          // a "press Generate" note of the last save is answered by this build
    try {
      await this.post('/api/coaching/build', body);
    } catch (error) {
      this.message = error.message; this.render(); return;
    }
    this.cache.clear(); this.gen++;
    await this.request();
  }

  async exportPdf() {
    // The PDF of the report on screen: the selection its answer was asked for, never one still loading.
    if (!this.answer || !this.answer.report || this.loading) return;
    const o = this.answerOptions && !this.answerOptions.invalid ? this.answerOptions : this.options();
    this.message = ''; this.c.pdf.disabled = true;
    try {
      const c = this.colours(), extra = isV2(this.answer.report) ? {colourA: c.A, colourB: c.B} : null;
      await this.download('/api/coaching/pdf?' + this.params(o, extra).toString());
    } catch (error) {
      this.message = `The PDF could not be made: ${error.message}`;
    }
    this.renderBar(); this.renderBanner();
  }

  async sendFeedback(observation) {
    const a = this.answer, draft = this.drafts.get(observation);
    if (!a || !a.report || !draft || !draft.verdict) return;
    this.notes.set(observation, 'Saving\u2026'); this.refreshFeedback(observation);
    try {
      if (isV2(a.report)) {
        const range = a.report.range || {};
        await this.post('/api/coaching/feedback', {run: this.runName(), team: a.report.team, findingKey: observation, verdict: draft.verdict, text: draft.text || '',
                                                   range: {t0: range.t0, t1: range.t1}});
      } else {
        await this.post('/api/coaching/feedback', {run: this.runName(), observation, verdict: draft.verdict, text: draft.text || '', fingerprint: a.fingerprint, team: a.report.team});
      }
    } catch (error) {
      this.notes.set(observation, error.message); this.refreshFeedback(observation); return;
    }
    this.drafts.delete(observation); this.notes.set(observation, 'Saved. Your review is kept with this video and never changes the measurements.');
    this.cache.clear(); this.gen++;
    await this.request();
  }

  async removeFeedback(entry) {
    try {
      await this.post('/api/coaching/feedback/remove', {id: entry.id});
    } catch (error) {
      this.notes.set(entry.attached, error.message); this.refreshFeedback(entry.attached); return;
    }
    this.drafts.delete(entry.attached); this.notes.set(entry.attached, 'Review removed.');
    this.cache.clear(); this.gen++;
    await this.request();
  }

  // ---------- lookups ----------
  report() { return this.answer && this.answer.report ? this.answer.report : null; }
  evidenceOf(id) { const r = this.report(); return r && Array.isArray(r.evidence) ? r.evidence.find(e => e && e.id === id) || null : null; }
  observationOf(id) { const r = this.report(); return r && Array.isArray(r.observations) ? r.observations.find(o => o && o.id === id) || null : null; }
  findings() { const r = this.report(); return r ? GROUPS.flatMap(([key]) => (Array.isArray(r[key]) ? r[key] : []).filter(f => f && f.observation)) : []; }
  findingOf(id) { return this.findings().find(f => f.observation === id) || null; }
  estimated() { const r = this.report(); return !r || (r.metresEstimated !== false && ((r.overview || {}).coverage || {}).metresEstimated !== false); }   // metres from a homography on an assumed pitch: estimates unless the report says otherwise
  feedbackFor(id) { return ((this.answer && this.answer.feedback) || []).filter(f => f.attached === id); }

  // ---------- rendering ----------
  render() {
    this.renderSetup(); this.renderBar(); this.renderBanner(); this.renderMain(); this.renderSide();
  }

  renderBar() {
    const a = this.answer, st = a ? a.state : null, b = this.c.generate;
    b.disabled = !this.result || this.loading || st === 'building' || st === 'ready' || !!(a && a.busy);
    b.textContent = st === 'building' ? 'Building…' : st === 'ready' ? 'Report up to date' : st === 'stale' ? 'Regenerate' : st === 'error' && a && a.report ? 'Regenerate' : 'Generate report';
    b.classList.toggle('primary', st !== 'ready');
    b.title = st === 'stale' ? 'Build the report again from the result as it is now' : st === 'ready' ? 'The report matches the result, the settings and the engine' : 'Build the report from this run’s tracking';
    this.c.pdf.disabled = !(a && a.report) || st === 'building' || this.loading;
  }

  renderBanner() {
    const a = this.answer, box = this.banner;
    box.replaceChildren(); box.className = 'cr-banner';
    let tone = null;
    if (this.message) { tone = 'error'; box.append(el('span', null, this.message)); }
    else if (!a || !this.result) tone = null;
    else if (a.state === 'stale') {
      tone = 'stale';
      // v2: a stored report whose measurement inputs changed (tracking, team-shape settings, match format, field size,
      // the possession the coach marked); a level, age or style change never makes one stale (the server re-reads it).
      box.append(el('b', null, 'This report is out of date.'), el('span', null, this.mode === 'v2'
        ? ' The findings below were made before these changes; Regenerate measures this run’s situations again (no new tracking):'
        : ' The findings below describe the result as it was when the report was made. Regenerate to update them:'));
      const list = el('ul');
      for (const r of a.reasons || []) list.append(el('li', null, r));
      box.append(list);
    } else if (a.state === 'building') {
      tone = 'info';
      box.append(el('span', null, a.report ? 'Building the report again from this run’s tracking; the earlier version stays shown until it is ready.' : 'Building the report from this run’s tracking…'));
    } else if (a.state === 'error') {
      tone = 'error';
      box.append(el('span', null, a.error || 'The report could not be built.'));
    } else if (a.busy && a.state !== 'ready') {
      tone = 'info';
      box.append(el('span', null, 'Another coaching report is being built; this one can be generated when it has finished (checked every second).'));
    } else if (this.mode === 'v2' && a.state === 'missing' && a.tiers && a.tiers.measurement && a.tiers.measurement !== 'ready'
               && (this.spansSaved || (a.reasons || []).length)) {
      // No report for the current measurement inputs (the match format, the field size, the possession the coach
      // marked). A run never generated is 'missing' too, and the empty report says what Generate does: this banner is
      // only for a change the server names, or a save of the coach's marks on this page.
      tone = 'info';
      const why = (a.reasons || []).filter(x => typeof x === 'string' && x);
      box.append(el('span', null, why.length ? `${why.join(' ')} This run’s situations are measured again: Generate (no new tracking).`
        : 'The possession you marked changed, so this run’s situations are measured again: Generate (no new tracking).'));
    }
    const note = this.rangeNote();
    if (note) { tone = tone || 'info'; box.append(el('span', 'cr-range-note', note)); }
    box.hidden = !tone;
    if (tone) box.classList.add(tone);
  }

  rangeNote() {
    // A custom range is typed in source-video time and the server clamps it to the analysed segment: say so when that
    // changed it (on BC180, 00:30-01:30 silently became 00:42.43-01:30.00).
    const a = this.answer, o = this.answerOptions, given = a && a.options;
    if (!this.result || !o || o.invalid || o.t0 === null || !given || typeof given !== 'object') return null;
    const [lo, hi] = this.bounds(), typed = `${clock(o.t0)}–${clock(o.t1)}`, segment = `${clock(lo)}–${clock(hi)}, source-video time`;
    if (given.t0 === null || given.t0 === undefined) return `The range ${typed} covers the whole analysed segment (${segment}), so this is the whole-segment report.`;
    if (isNum(given.t0) && isNum(given.t1) && (Math.abs(given.t0 - o.t0) > .005 || Math.abs(given.t1 - o.t1) > .005))
      return `The range ${typed} reaches outside the analysed segment (${segment}), so the report covers ${clock(given.t0)}–${clock(given.t1)}.`;
    return null;
  }

  section(key, title, intro) {
    const s = el('section', 'cr-sec');
    s.dataset.section = key;
    const h = el('h3', null, title);
    s.append(h);
    if (intro) s.append(el('p', 'cr-intro', intro));
    return s;
  }
  para(lead, text, cls) {
    const p = el('p', cls || null);
    if (lead) p.append(el('b', 'lead', lead + ' '));
    p.append(...labelNodes(text, this.labels));
    return p;
  }

  renderMain() {
    const main = this.main;
    main.replaceChildren(); this.cards = new Map(); this.boxes = new Map(); this.momentRows = []; this.renderedGen = this.gen;
    const a = this.answer, r = this.report();
    if (!this.result) { main.append(this.empty('Coaching reports appear here after a run.')); return; }
    if (!r && this.mode === 'v2') { this.emptyV2(a); return; }
    if (isV2(r)) { this.renderV2(r); return; }
    if (!r) {
      if (!a || this.loading) main.append(this.empty('Loading the coaching report…'));
      else if (a.state === 'building') main.append(this.empty('Building the report: the observations are measured from this run’s tracking and compared with the professional reference. This takes a few seconds.'));
      else if (a.state === 'error') main.append(this.empty(a.error || 'The report could not be built.'));
      else {
        // The range as the server took it (clamped to the analysed segment), else as typed.
        const o = this.options(), given = a.options && typeof a.options === 'object' && 't0' in a.options ? a.options : o;
        const box = this.empty(`No coaching report yet for Team ${o.team} (${given.t0 === null ? 'whole analysed segment' : `${clock(given.t0)}–${clock(given.t1)}`}, ${CONTEXTS[o.context].toLowerCase()} level).`);
        box.append(el('p', 'cr-intro', 'Generate builds it from this run’s tracking: team shape, spacing and line patterns measured over the segment and compared with a professional reference. Only what the tracking measured is reported; anything measured too briefly is listed as not assessed.'));
        main.append(box);
      }
      return;
    }
    // The PDF prints the same order (soccer/coaching/pdf.py section_order): overview, summary, the three groups of
    // findings, breakdown, priorities, training, coverage and confidence, then the feedback that sits on no finding.
    main.append(this.overview(r), this.summary(r));
    for (const [key, title, intro, none] of GROUPS) main.append(this.group(r, key, title, intro, none));
    main.append(this.breakdown(r), this.priorities(r), this.training(r), this.confidence(r));
    const apart = this.feedbackSection(r);
    if (apart) main.append(apart);
    if (!this.selected || !this.findingOf(this.selected)) {
      const first = ['improvements', 'reviews', 'strengths'].map(k => (Array.isArray(r[k]) ? r[k] : []).find(f => f && (f.evidence || []).length)).find(Boolean);
      this.selected = first ? first.observation : null; this.selectedEvidence = null;
    }
    for (const [id, card] of this.cards) card.classList.toggle('selected', id === this.selected);
  }

  empty(text) { const box = el('div', 'cr-sec cr-empty'); box.dataset.section = 'empty'; box.append(el('p', null, text)); return box; }

  overview(r) {
    const s = this.section('overview', 'Overview'), ov = r.overview || {}, range = r.range || {};
    const head = el('div', 'cr-ov-head');
    head.append(el('div', 'kicker', ov.title || (r.kind === 'match' ? 'MATCH COACHING REPORT' : 'SEGMENT COACHING REPORT')),
                el('div', 'cr-team', r.teamName || `Team ${r.team} (as labelled in this run)`));
    s.append(head);
    const rows = [];
    if (isNum(range.t0) && isNum(range.t1)) rows.push(['Range', `${clock(range.t0)}–${clock(range.t1)}` + (isNum(range.segmentT0) ? ` (segment ${clock(range.segmentT0)}–${clock(range.segmentT1)})` : '')]);
    if (isNum(range.seconds)) rows.push(['Length', seconds(range.seconds)]);
    rows.push(['Report', r.kind === 'match' ? 'Match report' : 'Segment report (under 20 minutes measured: no recurring-weakness claims)']);
    rows.push(['Context level', `${CONTEXTS[r.context] || words(r.context || 'competitive')} (changes the wording, never the measurements)`]);
    const value = (k, v) => isNum(v) ? (/seconds$/i.test(k) ? seconds(v) : /share$/i.test(k) ? `${Math.round(v * 100)}%` : String(r2(v)))
      : typeof v === 'boolean' ? (v ? 'yes' : 'no') : String(v);
    for (const [k, v] of Object.entries(ov.coverage || {})) {
      if (v === null || v === undefined || Array.isArray(v) || k === 'rangeSeconds') continue;   // the range's length is its own row
      if (typeof v === 'object') {
        // A group of counts or times (observations, measured seconds per category): one row.
        const parts = Object.entries(v).filter(([, x]) => isNum(x) || typeof x === 'string').map(([kk, x]) => `${words(kk).toLowerCase()} ${value(/seconds$/i.test(k) ? k : kk, x)}`);
        if (parts.length) rows.push([words(k), parts.join(' \u00b7 ')]);
        continue;
      }
      rows.push([words(k), value(k, v)]);
    }
    if (this.answer && this.answer.saved) rows.push(['Generated', String(this.answer.saved).replace('T', ' ')]);
    for (const [k, v] of rows) s.append(this.kv(k, v));
    // The run's data-quality note (soccer/coaching/report.py: the first overview.dataQualityLines limitations, stated once
    // with what would help) belongs with the overview: it says how far every finding below can be read. The rest of the
    // limitations, one line per measurement, stay under Confidence and limitations.
    const quality = this.qualityLines(r);
    if (quality.length) { s.append(el('h4', 'cr-sub-h', 'Data quality')); s.append(this.list(quality, 'cr-list cr-quality')); }
    // The context paragraph: block height of both teams, then where the ball was (descriptive, never a finding).
    const context = Array.isArray(ov.context) ? ov.context : [];
    if (context.length) { s.append(el('h4', 'cr-sub-h', 'Context')); s.append(this.list(context)); }
    return s;
  }

  qualityLines(r) {
    const ov = (r && r.overview) || {}, lim = Array.isArray(ov.limitations) ? ov.limitations : [];
    const n = Number.isInteger(ov.dataQualityLines) && ov.dataQualityLines > 0 ? Math.min(ov.dataQualityLines, lim.length) : 0;
    return lim.slice(0, n).filter(t => typeof t === 'string' && t);
  }

  limitationLines(r) {
    // The limitations after the data-quality note, with the bare "X: not assessed on this footage." lines of a footage that
    // shows a few players per team (the Veo clip: ten of them under one data-quality block) joined into one line.
    const ov = (r && r.overview) || {}, lim = Array.isArray(ov.limitations) ? ov.limitations : [];
    // (pdf.py join_bare does the same, so page and PDF list the same lines)
    const rest = lim.slice(this.qualityLines(r).length).filter(t => typeof t === 'string' && t), BARE = /^(.+): not assessed on this footage\.$/;
    const bare = rest.map(t => BARE.exec(t)).filter(Boolean).map(m => m[1]);
    if (bare.length < 2) return rest;
    const joined = `Not assessed on this footage: ${bare.map((x, i) => i ? x.charAt(0).toLowerCase() + x.slice(1) : x).join(', ')}.`;
    return rest.filter((t, i) => !BARE.test(t) || rest.findIndex(x => BARE.test(x)) === i).map(t => BARE.test(t) ? joined : t);
  }

  kv(label, value, low) {
    const row = el('div', 'kv'), v = el('b', low ? 'low' : null);
    v.append(...labelNodes(value, this.labels));
    row.append(el('span', null, label), v);
    return row;
  }
  list(items, cls) {
    const ul = el('ul', cls || 'cr-list');
    for (const t of items) { const li = el('li'); li.append(...labelNodes(t, this.labels)); ul.append(li); }
    return ul;
  }

  summary(r) {
    const s = this.section('summary', 'Summary'), items = Array.isArray(r.summary) ? r.summary : [];
    s.append(items.length ? this.list(items) : el('p', 'cr-intro', 'Nothing in this range met the evidence bar for a summary.'));
    return s;
  }

  group(r, key, title, intro, none) {
    const items = (Array.isArray(r[key]) ? r[key] : []).filter(f => f && f.observation);
    const s = this.section(key, `${title} · ${items.length}`, intro);
    if (!items.length) s.append(el('p', 'cr-none', none));
    for (const f of items) s.append(this.card(f));
    return s;
  }

  card(f) {
    const kind = KINDS[f.label] || 'review', obs = this.observationOf(f.observation) || {}, card = el('article', `cr-finding k-${kind}`);
    card.dataset.observation = f.observation;
    const head = el('div', 'cr-f-head'), conf = f.confidence || (obs.confidence && obs.confidence.overall);
    head.append(el('span', 'cr-badge', f.label || 'Finding'));
    if (conf) head.append(el('span', 'cr-conf', `Confidence ${conf}`));
    card.append(head);
    const h = el('h4'); h.append(...labelNodes(f.headline || words(obs.pattern || 'finding'), this.labels)); card.append(h);
    if (f.whatHappened) card.append(this.para('What happened.', f.whatHappened));
    if (f.whyItMatters) card.append(this.para('Why it matters.', f.whyItMatters));
    if (f.whatToImprove) {
      // A strength's whatToImprove is how to build on it (language.py starts it "To build on it:"): its own lead.
      const build = kind === 'strength' && /^To build on it:\s*/.exec(f.whatToImprove), rest = build ? f.whatToImprove.slice(build[0].length) : f.whatToImprove;
      card.append(this.para(kind === 'strength' ? 'To build on it.' : 'What to improve.', build ? rest.charAt(0).toUpperCase() + rest.slice(1) : rest));
    }
    if (f.howToPractice) card.append(this.para('How to practise.', f.howToPractice));
    const drill = this.drillOf(f.drill);
    if (drill) card.append(this.para('Drill.', drillName(drill), 'cr-drill-ref'));
    if (f.video) card.append(this.para('On the video.', f.video));
    if (f.confidenceNote) card.append(this.para('Confidence.', f.confidenceNote, 'cr-intro'));
    const metrics = Array.isArray(f.metrics) ? f.metrics : [];
    if (metrics.length) {
      const box = el('div', 'cr-metrics');
      for (const m of metrics) { const [name, value] = metricText(m.name, m.value, m.unit, m.estimated === true || (m.estimated === undefined && this.estimated())), chip = el('span', 'cr-metric'); chip.append(document.createTextNode(name + ' '), el('b', null, value)); box.append(chip); }
      card.append(box);
    }
    const evidence = (f.evidence || []).map(id => this.evidenceOf(id)).filter(Boolean), actions = el('div', 'cr-actions');
    actions.append(button(evidence.length ? `Evidence (${evidence.length})` : 'Details', 'quiet cr-show', () => this.select(f.observation), 'Show the evidence of this finding'));
    for (const ev of evidence) {
      const t = button(clock(ev.time), 'cr-time', () => { this.select(f.observation, ev.id); this.jump(ev, 'tracking'); }, `Show ${clock(ev.time)} on the video (segment ${clock(ev.segmentTime)})`);
      t.dataset.evidence = ev.id;
      actions.append(t);
    }
    card.append(actions, this.feedbackBlock(f));
    this.cards.set(f.observation, card);
    return card;
  }

  drillOf(id) {
    const r = this.report();
    if (!id || !r) return null;
    for (const t of r.training || []) { const d = t && t.drill; if (d && (d.id === id || d === id)) return typeof d === 'object' ? d : {id: d}; }
    return {id};
  }

  feedbackBlock(f) {
    const box = el('div', 'cr-fb'), id = isV2(this.report()) ? f.key : f.observation;
    box.dataset.feedback = id;
    this.boxes.set(id, box);
    this.fillFeedback(box, f);
    return box;
  }
  refreshFeedback(id) {
    const box = this.boxes.get(id), f = isV2(this.report()) ? this.findingByKey(id) : this.findingOf(id);
    if (box && f) this.fillFeedback(box, f);
  }
  fillFeedback(box, f) {
    // The verdict given on this finding in this run (the server attaches no other run's verdict to a finding). A v2
    // finding is keyed by its engine-independent key (DESIGN 6.1), so a verdict survives an engine update.
    const id = isV2(this.report()) ? f.key : f.observation, mine = this.feedbackFor(id).filter(e => e.thisRun && e.verdict).sort((a, b) => String(b.made).localeCompare(String(a.made)))[0] || null;
    const draft = this.drafts.get(id) || {verdict: mine ? mine.verdict : null, text: mine ? mine.text || '' : ''};
    box.replaceChildren(el('span', 'label', 'Your review'));
    const buttons = VERDICTS.map(([v, text]) => {
      const b = button(text, 'cr-verdict', () => {
        const d = this.drafts.get(id) || {...draft};
        d.verdict = v; this.drafts.set(id, d);
        for (const x of buttons) x.setAttribute('aria-pressed', String(x.dataset.verdict === v));
        save.disabled = false;
      });
      b.dataset.verdict = v; b.setAttribute('aria-pressed', String(draft.verdict === v));
      return b;
    });
    box.append(...buttons);
    const note = el('textarea', 'cr-note-input');
    note.value = draft.text || ''; note.maxLength = 2000; note.rows = 2;
    note.placeholder = 'Optional note: what the tracking got right or missed';
    note.setAttribute('aria-label', 'Review note');
    note.addEventListener('input', () => { const d = this.drafts.get(id) || {...draft}; d.text = note.value; this.drafts.set(id, d); save.disabled = !d.verdict; });
    const save = button(mine ? 'Update review' : 'Save review', 'primary cr-save', () => this.sendFeedback(id));
    save.disabled = !this.drafts.has(id) || !(this.drafts.get(id) || {}).verdict;
    box.append(note, save);
    if (mine) box.append(button('Remove', 'quiet cr-remove', () => this.removeFeedback(mine), 'Remove your review of this finding'));
    const lines = [];
    if (mine) lines.push(`Saved ${String(mine.made || '').replace('T', ' ')}: ${verdictName(mine.verdict)}${mine.text ? ` · “${mine.text}”` : ''}`);
    if (this.notes.has(id)) lines.push(this.notes.get(id));
    if (lines.length) box.append(el('p', 'msg', lines.join('\n')));
  }

  link(id) {
    const f = this.findingOf(id), obs = this.observationOf(id);
    const b = button('', 'cr-link', () => { this.select(id); const card = this.cards.get(id); if (card && card.scrollIntoView) card.scrollIntoView({behavior: 'smooth', block: 'center'}); });
    b.append(...labelNodes(f ? f.headline : obs ? words(obs.pattern || obs.detector || id) : id, this.labels));
    return b;
  }

  breakdown(r) {
    const s = this.section('breakdown', 'Breakdown'), b = r.breakdown || {};
    for (const [key, title] of SECTIONS) {
      const sec = b[key] || {available: false, reason: 'Not part of this report.'}, row = el('div', 'cr-part');
      row.dataset.part = key;
      row.append(el('h4', null, title));
      if (!sec.available) row.append(this.para(null, sec.reason || 'Not available for this footage.', 'cr-none'));
      else {
        const ids = Array.isArray(sec.findings) ? sec.findings : [];
        if (ids.length) { const list = el('div', 'cr-links'); for (const id of ids) list.append(this.link(id)); row.append(list); }
        else row.append(el('p', 'cr-none', 'Measured; nothing here met the evidence bar.'));
      }
      for (const n of Array.isArray(sec.notes) ? sec.notes : []) row.append(this.para(null, n, 'cr-intro'));
      s.append(row);
    }
    return s;
  }

  priorities(r) {
    const items = (Array.isArray(r.priorities) ? r.priorities : []).slice(0, 3), s = this.section('priorities', 'Priorities');
    if (!items.length) { s.append(el('p', 'cr-none', 'No priority: nothing in this range calls for a training focus.')); return s; }
    const ol = el('ol', 'cr-list');
    for (const p of items) {
      const li = el('li');
      li.append(...labelNodes(p.title || '', this.labels));
      if (p.finding) { li.append(document.createTextNode(' · ')); li.append(this.link(p.finding)); }
      const d = this.drillOf(p.drill);
      if (p.drill && d) li.append(document.createTextNode(` \u00b7 drill: ${drillName(d)}`));
      ol.append(li);
    }
    s.append(ol);
    return s;
  }

  training(r) {
    // Drills as soccer/coaching/drills.drill_ref carries them: the library record and the variation for the level.
    const items = Array.isArray(r.training) ? r.training : [], s = this.section('training', 'Training plan');
    if (!items.length) { s.append(el('p', 'cr-none', 'No drill is suggested for this range.')); return s; }
    const span = (x, unit) => x && typeof x === 'object' ? [isNum(x.min) && isNum(x.max) ? (x.min === x.max ? `${x.min}${unit}` : `${x.min}\u2013${x.max}${unit}`) : '', x.note || ''].filter(Boolean).join('. ') : typeof x === 'string' || isNum(x) ? String(x) : '';
    for (const t of items) {
      const d = t && typeof t.drill === 'object' && t.drill ? t.drill : {id: t && t.drill}, box = el('div', 'cr-drill');
      box.dataset.drill = d.id || '';
      box.append(el('h4', null, drillName(d)));
      for (const [key, lead] of [['purpose', 'Purpose.'], ['objective', 'Objective.'], ['tacticalIssue', 'Addresses.'], ['summary', null], ['description', null]])
        if (typeof d[key] === 'string' && d[key]) box.append(this.para(lead, d[key]));
      const players = span(d.players, ' players'), minutes = span(d.durationMinutes || d.duration, ' min');
      if (players) box.append(this.para('Players.', players));
      if (minutes) box.append(this.para('Duration.', minutes));
      if (typeof d.area === 'string' && d.area) box.append(this.para('Area.', d.area));
      if (typeof d.setup === 'string' && d.setup) box.append(this.para('Set-up.', d.setup));
      // drills.json's fields, in the PDF's order (soccer/coaching/pdf.py DRILL_FIELDS): equipment, how it runs, coaching
      // points, progressions, regressions, success criteria (one in the exercise, one naming the report measurement it
      // should move), safety.
      for (const [key, title] of [['equipment', 'Equipment'], ['instructions', 'How it runs'], ['coachingPoints', 'Coaching points'], ['progressions', 'Progressions'],
                                  ['regressions', 'Regressions'], ['successCriteria', 'Success looks like'], ['safety', 'Safety']]) {
        const v = d[key];
        if (Array.isArray(v) && v.length) { box.append(el('div', 'cr-sub-h', title)); box.append(this.list(v.map(String))); }
        else if (typeof v === 'string' && v) box.append(this.para(title + '.', v));
      }
      if (typeof d.variation === 'string' && d.variation) box.append(this.para(`At the ${CONTEXTS[d.level] ? CONTEXTS[d.level].toLowerCase() : 'chosen'} level.`, d.variation));
      if (typeof d.source === 'string' && d.source) box.append(this.para(null, d.source, 'cr-intro'));
      const ids = Array.isArray(t.findings) ? t.findings : [];
      if (ids.length) { const list = el('div', 'cr-links'); list.append(el('span', 'cr-intro', 'For: ')); for (const id of ids) list.append(this.link(id)); box.append(list); }
      s.append(box);
    }
    return s;
  }

  confidence(r) {
    const s = this.section('confidence', 'Confidence and limitations'), c = r.confidence || {};
    if (c.overall) s.append(this.kv('Overall confidence', String(c.overall), c.overall === 'low'));
    if (Array.isArray(c.notes) && c.notes.length) s.append(this.list(c.notes));
    // What could not be assessed and why, as the report words it (an observation measured too briefly is a
    // limitation, never a strength or a weakness); the data-quality note is in the overview, shown once.
    const lim = this.limitationLines(r);
    if (lim.length) { s.append(el('h4', 'cr-sub-h', 'Limitations')); s.append(this.list(lim)); }
    if (this.qualityLines(r).length) s.append(el('p', 'cr-intro', 'The data-quality note of this footage is in the overview.'));
    s.append(el('p', 'cr-intro', 'Metres assume a standard 105 \u00d7 68 m pitch and carry the registration\u2019s error; no possession, passes or pressing are recorded, so nothing here is tied to a phase of play.'));
    return s;
  }

  feedbackSection(r) {
    // Reviews kept with this video that sit on no finding of this report, where the PDF has its Expert feedback: this
    // run's verdicts on findings this report does not have (another range, or a finding that changed since), and
    // verdicts from earlier runs, which never sit on a finding (observation ids leave the run out and team letters are
    // arbitrary per run: the verifier saw a run-r1 "Incorrect" printed on run r2's finding). Those name players by that
    // run's ids, so their text is shown as stored, not as this run's labels.
    const all = ((this.answer && this.answer.feedback) || []).filter(f => f && typeof f === 'object');
    const teamOf = f => (f.observation && f.observation.team) || null;
    const loose = all.filter(f => f.thisRun && !f.attached && (!teamOf(f) || teamOf(f) === r.team));
    const earlier = all.filter(f => !f.thisRun);
    if (!loose.length && !earlier.length) return null;
    const s = this.section('feedback', 'Expert feedback', 'Reviews kept with this video that are not on a finding of this report. They never change the measurements.');
    const line = f => `${(f.observation && f.observation.headline) || 'A finding'} \u00b7 ${verdictName(f.verdict)}${f.text ? ` \u00b7 \u201c${f.text}\u201d` : ''}`;
    if (loose.length) { s.append(el('h4', 'cr-sub-h', 'Your reviews of this run that match no finding here')); s.append(this.list(loose.map(line))); }
    if (earlier.length) {
      s.append(el('h4', 'cr-sub-h', 'From an earlier run'));
      const ul = el('ul', 'cr-list');
      for (const f of earlier) ul.append(el('li', null, `${line(f)} (run ${f.run}${teamOf(f) ? `, Team ${teamOf(f)} as labelled in that run` : ''}; players named by that run\u2019s ids)`));
      s.append(ul);
    }
    return s;
  }

  select(observation, evidence) {
    this.selected = observation; this.selectedEvidence = evidence || null;
    for (const [id, card] of this.cards) card.classList.toggle('selected', id === observation);
    this.renderSide();
  }

  jump(ev, view) {
    const fps = isNum(ev.fps) && ev.fps > 0 ? ev.fps : this.fps();
    const seek = isNum(ev.seek) ? ev.seek : Number.isInteger(ev.frame) ? (ev.frame + .5) / fps : ev.time;
    return this.jumpTo({frame: ev.frame, seek, time: ev.time, view: view === 'board' ? 'board' : 'tracking', focus: (ev.players || []).filter(p => typeof p === 'string'),
                        diagram: ev.diagram || null, overlay: validOverlay(ev.videoOverlay) ? ev.videoOverlay : null, evidence: ev.id, observation: ev.observation});
  }

  renderSide() {
    if (isV2(this.report())) { this.renderSideV2(); return; }
    const side = this.side, r = this.report();
    side.replaceChildren(el('h3', null, 'Evidence'));
    side.hidden = !r;
    const f = this.selected ? this.findingOf(this.selected) : null;
    if (!r || !f) { side.append(el('p', 'an-empty', r ? 'Select a finding to see its evidence: the frame, the measured shape and a diagram.' : 'The evidence of a finding appears here: its frame, the measured shape and a diagram.')); this.placeSide(); return; }
    const head = el('div', `cr-side-head k-${KINDS[f.label] || 'review'}`), h = el('div', 'cr-side-title');
    h.append(...labelNodes(f.headline || '', this.labels));
    head.append(el('span', 'cr-badge', f.label || 'Finding'), h);
    side.append(head);
    const list = (f.evidence || []).map(id => this.evidenceOf(id)).filter(Boolean);
    if (!list.length) side.append(el('p', 'an-empty', 'No evidence frame for this finding: it rests on the measured totals listed with it.'));
    for (const ev of list) side.append(this.figure(ev));
    this.placeSide();
  }

  figure(ev) {
    const fig = el('figure', 'cr-ev'), colours = this.colours(), video = (this.info && this.info.video) || {};
    fig.dataset.evidence = ev.id;
    if (ev.id === this.selectedEvidence) fig.classList.add('selected');
    const times = el('div', 'cr-times');
    const at = button(clock(ev.time), 'cr-time', () => this.jump(ev, 'tracking'), `Show this frame on the video (segment ${clock(ev.segmentTime)})`);
    at.dataset.evidence = ev.id;
    const board = button('Show on 2D', 'quiet cr-board', () => this.jump(ev, 'board'), 'Show this moment on the 2D match view');
    board.dataset.evidence = ev.id;
    times.append(at, el('span', 'cr-seg', `segment ${clock(ev.segmentTime)}`), board);
    fig.append(times);
    if (Number.isInteger(ev.frame)) {
      const shot = el('div', 'cr-shot'), img = el('img');
      if (video.width && video.height) shot.style.aspectRatio = `${video.width} / ${video.height}`;
      img.alt = `Frame ${ev.frame} at ${clock(ev.time)}`; img.loading = 'lazy'; img.decoding = 'async';
      img.onerror = () => shot.replaceChildren(el('p', 'cr-noshot', 'The screenshot could not be loaded.'));
      img.src = '/api/coaching/frame?' + new URLSearchParams({run: this.runName() || '', frame: String(ev.frame), w: '640'}).toString();
      shot.append(img);
      if (validOverlay(ev.videoOverlay)) shot.append(svgOverlay(ev.videoOverlay, colours, this.labels));
      fig.append(shot);
      const tier = (ev.videoOverlay && ev.videoOverlay.tier) || ev.tier;
      if (TIERS[tier]) fig.append(el('p', 'cr-note', TIERS[tier]));
    }
    if (ev.caption) fig.append(this.para(null, ev.caption, 'cr-caption'));
    if (ev.diagram && Array.isArray(ev.diagram.elements)) {
      if (ev.diagram.title) fig.append(this.para(null, ev.diagram.title, 'cr-dtitle'));
      fig.append(svgDiagram(ev.diagram, colours, this.labels));
      if (ev.diagram.metresEstimated !== false) fig.append(el('p', 'cr-note', 'Top-down from the observed positions at this frame; distances estimated (assumed 105 × 68 m pitch).'));
    }
    for (const [k, v] of Object.entries(ev.metrics || {})) { const [name, value] = metricText(k, v, null, this.estimated()); fig.append(this.kv(name, value)); }
    if (ev.confidence) fig.append(this.kv('Confidence', String(ev.confidence), ev.confidence === 'low'));
    const cal = ev.calibration || {};
    if (isNum(cal.confidence)) fig.append(this.kv('Registration confidence', cal.confidence.toFixed(2), cal.confidence < .7));
    if (Array.isArray(ev.players) && ev.players.length) fig.append(this.kv('Players', ev.players.join(', ')));   // ids: kv shows their labels
    return fig;
  }

  // ================= coaching v2: the setup (DESIGN 9.1) =================
  async prepare() {
    // The catalogue once per page (no such route: a v1 server, legacy mode) and the run's coaching settings once per run.
    if (!this.catalogueAsk) {
      // Legacy (the v1 report) only for a server without the route: an HTTP 404 (index.html's api() puts the status on
      // its error, also when the body is Flask's HTML page and does not parse). Any other failure (a JSON 503 when a v2
      // module is broken or a coaching table does not validate, an HTML 500, the network) is a v2 server that could not
      // answer: its error is shown and the catalogue asked again on the next request, never the professional-reference
      // v1 report in its place. CODE L2 (round 1), M-1 (round 2): a profiles.json the table's own bounds refuse
      // answered an HTML 500, which failed to parse (a SyntaxError), and that was taken for a missing route.
      this.catalogueAsk = Promise.resolve().then(() => this.api('/api/coaching/profiles'))
        .then(c => { this.mode = 'v2'; this.catalogue = mergeCatalogue(c); this.catalogueError = null; },
              e => {
                if (!!e && e.status === 404) { this.mode = 'legacy'; this.catalogue = CATALOGUE; this.catalogueError = null; return; }
                this.mode = 'v2'; this.catalogue = CATALOGUE; this.catalogueAsk = null;
                this.catalogueError = `The coaching setup catalogue could not be read (${failText(e)}); PitchIQ’s built-in copy is shown.`;
              });
    }
    await this.catalogueAsk;
    const run = this.runName();
    if (this.mode === 'v2' && (!this.setupAsk || this.setupRun !== run)) {
      this.setupRun = run;
      this.setupAsk = Promise.resolve().then(() => this.api('/api/coaching/settings?' + new URLSearchParams({run: run || ''}).toString()))
        .then(a => { this.setupAnswer = a && typeof a === 'object' ? a : null; },
              e => { this.setupAnswer = null; this.setupMessage = `The saved coaching setup could not be read (${e.message}); the defaults are shown.`; });
    }
    if (this.setupAsk) await this.setupAsk;
    if (this.marksAsk) await this.marksAsk;
  }
  teamKey() { return this.c.team.value === 'B' ? 'B' : 'A'; }
  marksReady() { return !!this.marks && this.marks.available === true && this.mode === 'v2'; }
  marksCall(text) {
    const p = el('p', 'cr-intro cr-marks-call'), b = button('Mark who had the ball', 'quiet', () => this.openMarks(), 'Open the marking bar under the video');
    b.dataset.marks = 'open';
    p.append(document.createTextNode(text + ' '), b);
    return p;
  }
  savedSetup(team) {
    if (this.mode !== 'v2') return this.localSaved[team];
    const s = this.setupAnswer && this.setupAnswer.settings, b = s && s.teams && s.teams[team];
    return b && typeof b === 'object' ? b : null;
  }
  setupBlock(team) {
    // The block the setup shows: the coach's draft, else the saved settings.
    if (!this.setupDraft[team]) this.setupDraft[team] = cleanBlock(this.savedSetup(team), this.catalogue);
    return this.setupDraft[team];
  }
  setupDirty(team) {
    return JSON.stringify(cleanBlock(this.savedSetup(team), this.catalogue)) !== JSON.stringify(this.setupBlock(team)) || !!(this.squadName && this.squadName.trim());
  }
  editSetup(change) {
    change(this.setupBlock(this.teamKey()));
    this.setupMessage = '';
    this.renderSetup();
  }
  setAge(d, age) {
    // The format follows the age group's default (DESIGN 2.2) unless the coach chose another one: a format equal to the
    // previous age's default (or to 11v11 before any age) is a default, anything else the coach's choice.
    const before = d.ageGroup ? formatDefaultOf(this.catalogue, d.ageGroup, d.federation) : {players: 11, goalkeeper: true}, follows = sameFormat(d.format, before);
    d.ageGroup = age || null;
    if (follows && d.ageGroup) this.setFormat(d, formatDefaultOf(this.catalogue, d.ageGroup, d.federation));
  }
  setFederation(d, fed) {
    const before = d.ageGroup ? formatDefaultOf(this.catalogue, d.ageGroup, d.federation) : null, follows = before && sameFormat(d.format, before);
    d.federation = fed;
    if (follows) this.setFormat(d, formatDefaultOf(this.catalogue, d.ageGroup, d.federation));
  }
  setFormat(d, f) {
    // A format change drops an intended formation and confirmed roles the new format does not offer (never apply 11v11
    // positions to a small-sided match: spec Part 3).
    d.format = {players: f.players, goalkeeper: f.players >= 7 ? true : !!f.goalkeeper};
    const offered = this.catalogue.formations[d.format.players] || [];
    if (d.formation.intended && !offered.includes(d.formation.intended)) d.formation.intended = null;
    d.formation.periods = d.formation.periods.filter(p => offered.includes(p.formation));
    this.dropRoles(d);
  }
  dropRoles(d) {
    const roles = this.roleOptions(d);
    for (const [id, spans] of Object.entries(d.roles)) if (spans.some(s => !roles.includes(s.role))) delete d.roles[id];
  }
  roleOptions(d) {
    // The roles a coach may confirm (profiles.roles_for): the server's list for the format and formation when it sent
    // one, else team_shape's role names for 11v11 and relative slots (plus GK) for smaller formats.
    const byFormat = this.catalogue.rolesByFormat && this.catalogue.rolesByFormat[d.format.players], listed = byFormat && d.formation.intended ? byFormat[d.formation.intended] : null;
    const usable = strings(listed) && (d.format.players === 11 || listed.includes('GK') === !!d.format.goalkeeper);
    if (usable) return listed.slice();
    return d.format.players === 11 ? this.catalogue.roles.slice() : (d.format.goalkeeper ? ['GK'] : []).concat(relativeSlots(d.format, d.formation.intended));
  }
  teamPlayers(team) {
    // The team's identities the roles editor lists: [{id, keeper}], the tracked goalkeepers first. An identity beyond
    // the numbered slots (labels.py: confirmed false, shown as A-?) is left out, its identity is not settled (CODE L9).
    const m = (this.result && this.result.match) || {}, out = [];
    for (const key of ['goalkeepers', 'players']) for (const e of m[key] || []) {
      if (e && e.team === team && typeof e.id === 'string' && PLAYER_ID.test(e.id) && e.confirmed !== false) out.push({id: e.id, keeper: key === 'goalkeepers'});
    }
    return out;
  }
  unsettledPlayers(team) {
    const m = (this.result && this.result.match) || {};
    return (m.players || []).filter(e => e && e.team === team && typeof e.id === 'string' && PLAYER_ID.test(e.id) && e.confirmed === false).length;
  }
  playerRoles(options, keeper, team) {
    // A tracked goalkeeper is offered the goalkeeper only, and an outfield player the outfield roles when the team has
    // a tracked goalkeeper (the editor offered every outfield role on the GK-A row: CODE L9).
    if (keeper) return options.filter(r => r === 'GK');
    return this.teamPlayers(team).some(p => p.keeper) ? options.filter(r => r !== 'GK') : options.slice();
  }
  setupSummary(d) {
    const lv = this.catalogue.levels.find(l => l.level === d.playingLevel), style = this.catalogue.styles.find(s => s.id === d.style);
    return [d.ageGroup || 'age group not set', lv ? `level ${lv.level} ${titleCase(lv.name)}` : 'level not set',
            formatName(d.format) + (d.format.players <= 5 && !d.format.goalkeeper ? ' without goalkeeper' : ''),
            d.formation.intended || 'formation not set', style && d.style !== 'unknown' ? style.name.toLowerCase() : 'style not set'].join(' · ');
  }
  measuredField() {
    const ps = this.result && this.result.tactical && this.result.tactical.pitchSize;
    return ps && ps.source === 'landmarks' && isNum(ps.length) && isNum(ps.width) ? ps : null;
  }

  renderSetup() {
    const host = this.c.setup;
    if (!host) return;
    host.replaceChildren();
    host.hidden = !this.result || this.mode === null;
    if (host.hidden) return;
    const team = this.teamKey(), d = this.setupBlock(team), saved = this.savedSetup(team), dirty = this.setupDirty(team);
    const open = this.setupOpen === null ? !saved || dirty : this.setupOpen;
    const head = el('div', 'cr-setup-head'), title = el('div', 'cr-setup-title');
    title.append(el('h3', null, 'Coaching setup'), el('span', 'cr-setup-sum', `Team ${team} · ${this.setupSummary(d)}`));
    const toggle = button(open ? 'Hide' : 'Edit setup', 'quiet cr-setup-toggle', () => { this.setupOpen = !open; this.renderSetup(); });
    toggle.setAttribute('aria-expanded', String(open)); toggle.dataset.setup = 'toggle';
    head.append(title, toggle);
    host.append(head);
    if (!saved && !dirty) {
      host.append(el('p', 'cr-setup-note', this.mode === 'v2'
        ? 'Not saved for this team yet: the report uses PitchIQ’s defaults. Choose the age group and the playing level so it speaks to your players; the measurements never change with them.'
        : 'This server still builds the earlier report (one professional reference, three wordings): the setup is kept on this page and only the level picks the wording.'));
    }
    if (open) host.append(this.setupBody(team, d, dirty));
    if (this.catalogueError) host.append(el('p', 'msg cr-setup-msg', this.catalogueError));
    if (this.setupMessage) host.append(el('p', 'msg cr-setup-msg', this.setupMessage));
  }

  setupBody(team, d, dirty) {
    const cat = this.catalogue, body = el('div', 'cr-setup-body'), grid = el('div', 'cr-setup-grid');
    const tag = (node, name) => { node.dataset.setup = name; return node; };
    const age = tag(selectEl([['', 'Choose…'], ...cat.ages.map(a => [a, a === 'Adult' ? 'Adult' : a])], d.ageGroup, v => this.editSetup(x => this.setAge(x, v || null)), 'Age group'), 'age');
    const level = tag(selectEl([['', 'Choose…'], ...cat.levels.map(l => [String(l.level), `${l.level} · ${titleCase(l.name)}`])], d.playingLevel,
                               v => this.editSetup(x => { x.playingLevel = Number(v) || null; }), 'Playing level'), 'level');
    const format = tag(selectEl(cat.formats.map(f => [f.name, f.name + (f.note && f.note.length <= 40 ? ` (${f.note})` : '')]), formatName(d.format),
                                v => this.editSetup(x => { const f = cat.formats.find(q => q.name === v); if (f) this.setFormat(x, {players: f.players, goalkeeper: f.goalkeeper}); }), 'Match format'), 'format');
    const formatField = fieldEl('Match format', format);
    const fmtInfo = cat.formats.find(f => f.players === d.format.players) || {};
    if (fmtInfo.choice !== undefined ? fmtInfo.choice : d.format.players <= 5) {
      // U.S. 4v4 has no goalkeeper; the coach can add one (and take it out of a 5v5).
      const gk = tag(el('input'), 'goalkeeper'), chip = el('label', 'chip cr-gk');
      gk.type = 'checkbox'; gk.checked = !!d.format.goalkeeper;
      gk.addEventListener('change', () => this.editSetup(x => { x.format = {players: x.format.players, goalkeeper: !!gk.checked}; this.dropRoles(x); }));
      chip.append(gk, document.createTextNode(' Goalkeeper'));
      formatField.append(chip);
    }
    const notes = cat.formationNotes || {}, offered = cat.formations[d.format.players] || [];
    const formation = tag(selectEl([['', 'Other / not sure'], ...offered.map(n => [n, n + (notes[n] ? ` (${notes[n]})` : '')])], d.formation.intended,
                                   v => this.editSetup(x => { x.formation.intended = v || null; this.dropRoles(x); }), 'Intended formation'), 'formation');
    const style = tag(selectEl(cat.styles.map(s => [s.id, s.name]), d.style, v => this.editSetup(x => { x.style = v; }), 'Style of play'), 'style');
    const chosen = cat.styles.find(s => s.id === d.style);
    if (chosen && typeof chosen.description === 'string') style.title = chosen.description;
    const fed = tag(selectEl(cat.federations.map(f => [f.id, f.name]), d.federation, v => this.editSetup(x => this.setFederation(x, v)), 'Federation'), 'federation');
    fed.title = 'Sets the age-group format defaults and which heading-safety text is shown; the stricter safety rule always applies.';
    const len = tag(el('input'), 'field-length'), wid = tag(el('input'), 'field-width');
    for (const [inp, key, lo, hi] of [[len, 'length', 15, 130], [wid, 'width', 10, 100]]) {
      inp.type = 'number'; inp.min = String(lo); inp.max = String(hi); inp.step = '0.5'; inp.placeholder = key;
      inp.value = d.field && isNum(d.field[key]) ? String(d.field[key]) : '';
      inp.setAttribute('aria-label', `Field ${key} in metres (optional)`);
      inp.addEventListener('change', () => this.editSetup(x => {
        const a = String(len.value).trim(), b = String(wid.value).trim();
        x.field = !a && !b ? null : {length: a ? Number(a) : NaN, width: b ? Number(b) : NaN};
      }));
    }
    const size = el('span', 'cr-size');
    size.append(len, el('span', null, '×'), wid, el('span', null, 'm'));
    grid.append(fieldEl('Age group', age), fieldEl('Playing level', level), formatField, fieldEl('Intended formation', formation), fieldEl('Style of play', style),
                fieldEl('Federation', fed), fieldEl('Field size (optional)', size));
    body.append(grid);
    const lines = this.setupNotes(d);
    if (lines.length) { const ul = el('ul', 'cr-list cr-setup-notes'); for (const t of lines) ul.append(el('li', null, t)); body.append(ul); }
    const actions = el('div', 'cr-setup-actions');
    const info = tag(button('Sources for this level', 'quiet', () => { this.setupPanel = this.setupPanel === 'info' ? null : 'info'; this.renderSetup(); }), 'info');
    const confirmed = Object.keys(d.roles).length;
    const roles = tag(button(`Player roles (optional)${confirmed ? ` · ${confirmed} confirmed` : ''}`, 'quiet', () => { this.setupPanel = this.setupPanel === 'roles' ? null : 'roles'; this.renderSetup(); }), 'roles');
    info.setAttribute('aria-expanded', String(this.setupPanel === 'info')); roles.setAttribute('aria-expanded', String(this.setupPanel === 'roles'));
    actions.append(info, roles);
    if (this.mode === 'v2') actions.append(this.squadControls(team, d));
    const discard = tag(button('Discard changes', 'quiet', () => { this.setupDraft[team] = null; this.squadName = null; this.setupMessage = ''; this.renderSetup(); }), 'discard');
    const save = tag(button(this.setupBusy ? 'Saving…' : 'Save setup', dirty ? 'primary' : null, () => this.saveSetup()), 'save');
    discard.disabled = !dirty || this.setupBusy; save.disabled = !dirty || this.setupBusy;
    actions.append(el('span', 'cr-spacer'), discard, save);
    body.append(actions);
    if (this.marksReady()) {
      const n = this.marks.spans.length, row = this.marksCall(n ? `Who had the ball: ${n} ${plural(n, 'phase')} marked by you for this run (both teams; "ours" is the team chosen above).`
                                                                : 'Who had the ball: not marked by you for this run. Marks you make while watching are used instead of PitchIQ\u2019s estimate where you make them.');
      row.classList.add('cr-setup-marks');
      body.append(row);
    }
    if (this.setupPanel === 'roles') body.append(this.rolesPanel(team, d));
    if (this.setupPanel === 'info') body.append(this.infoPanel(d));
    return body;
  }

  setupNotes(d) {
    // The non-blocking notes of DESIGN 2.2 and 2.5: the level, an unusual age and level pair, the format default, small
    // formats, the field scale.
    const out = [], cat = this.catalogue, lv = cat.levels.find(l => l.level === d.playingLevel);
    if (lv) out.push(`Level ${lv.level} ${titleCase(lv.name)}${/^typical\b/i.test(lv.summary) ? '' : ` (typical ${lv.typical})`}: ${lv.summary}`);   // the server's summaries start with the typical ages
    const typical = d.ageGroup ? (cat.typicalByAge && cat.typicalByAge[d.ageGroup]) || typicalLevel(d.ageGroup) : null;
    if (lv && typical && lv.level > typical + 2)
      out.push(`Level ${lv.level} is well above the typical level for ${d.ageGroup} (level ${typical}): its checks are applied, while safety and tone still follow the age group.`);
    if (lv && d.ageGroup === 'Adult' && lv.level <= 3) out.push(`Adult players at level ${lv.level}: level ${lv.level} ideas, worded for adults.`);
    if (d.ageGroup) {
      const def = formatDefaultOf(cat, d.ageGroup, d.federation), src = defaultSource(cat, d.ageGroup, d.federation), by = src ? ` (${src})` : '';
      out.push(sameFormat(def, d.format) ? `${fmtText(def)} is the default for ${d.ageGroup}${by}; you can change it.`
                                         : `${d.ageGroup} usually plays ${fmtText(def)}${by}; your choice, ${fmtText(d.format)}, is used.`);
    }
    if (cat.formatNotes && cat.formatNotes[d.format.players]) out.push(cat.formatNotes[d.format.players]);
    if (d.format.players < 11) out.push('Small-sided matches are never judged against 11v11 positions. PitchIQ’s pitch registration knows full-size markings only: check the 2D match view before trusting distances on a small field.');
    out.push('The formation is a starting model, not proof of where every player must stand.');
    const measured = this.measuredField();
    if (measured) out.push(`Field measured by calibration: about ${Math.round(measured.length)} × ${Math.round(measured.width)} m.`);
    else if (!d.field && d.format.players < 11) out.push('Without a field size, distances between players are not judged on a small-sided field.');
    return out;
  }

  squadControls(team, d) {
    // The squad this team is (progress across matches, DESIGN 8.1): an existing one, a new one, or none. A squad saved
    // for this video in another run asks to be linked to this run first (team letters change between runs).
    const wrap = el('span', 'cr-squad'), run = this.runName();
    const squads = (this.setupAnswer && Array.isArray(this.setupAnswer.squads) ? this.setupAnswer.squads : []).filter(s => s && typeof s.id === 'string' && typeof s.name === 'string');
    const options = [['', 'No squad'], ...squads.map(s => [s.id, s.name])];
    if (d.squad && !squads.some(s => s.id === d.squad.id)) options.splice(1, 0, [d.squad.id, d.squad.name]);
    options.push(['new', 'New squad…']);
    const sel = selectEl(options, this.squadName !== null ? 'new' : d.squad ? d.squad.id : '', v => {
      if (v === 'new') { this.squadName = this.squadName || ''; this.renderSetup(); return; }
      this.squadName = null;
      this.editSetup(x => {
        if (!v) { x.squad = null; x.runLinks = x.runLinks.filter(l => l.run !== run); return; }
        const s = squads.find(q => q.id === v) || (x.squad && x.squad.id === v ? x.squad : null);
        if (s) { x.squad = {id: s.id, name: s.name}; linkRun(x, run, team); }
      });
    }, 'Squad, to follow progress across matches');
    sel.dataset.setup = 'squad';
    wrap.append(fieldEl('Squad', sel));
    if (this.squadName !== null) {
      const name = el('input');
      name.type = 'text'; name.maxLength = 60; name.placeholder = 'Squad name, e.g. U10 Blue'; name.value = this.squadName; name.dataset.setup = 'squad-name';
      name.setAttribute('aria-label', 'New squad name');
      name.addEventListener('input', () => { this.squadName = name.value; const save = this.c.setup && this.findSetup('save'); if (save) { save.disabled = !this.setupDirty(team); } });
      wrap.append(name);
    } else if (d.squad && !d.runLinks.some(l => l.run === run && l.team === team && l.confirmed)) {
      const link = button(`Link Team ${team} of this run to ${d.squad.name}`, 'quiet', () => this.editSetup(x => linkRun(x, run, team)));
      link.dataset.setup = 'link';
      wrap.append(link);
    }
    if (d.squad && this.setupAnswer && 'matchDate' in this.setupAnswer) {
      // The day of the match, for the squad's progress (else the video file's date is used); saved at once.
      const day = el('input');
      day.type = 'date'; day.value = typeof this.setupAnswer.matchDate === 'string' ? this.setupAnswer.matchDate : ''; day.dataset.setup = 'match-date';
      day.setAttribute('aria-label', 'Match date, for progress across matches');
      day.addEventListener('change', () => this.saveMatchDate(day.value));
      wrap.append(fieldEl('Match date', day));
    }
    return wrap;
  }
  async saveMatchDate(value) {
    const day = /^\d{4}-\d{2}-\d{2}$/.test(String(value || '')) ? value : null;
    try {
      const a = await this.post('/api/coaching/match-date', {run: this.runName(), date: day});
      if (this.setupAnswer) this.setupAnswer.matchDate = a && typeof a.date === 'string' ? a.date : null;
      this.setupMessage = day ? `Match date saved: ${day}. The squad’s progress dates this match by it.` : 'Match date removed: the video file’s date is used.';
    } catch (error) {
      this.setupMessage = `The match date could not be saved: ${error.message}`;
    }
    this.renderSetup();
  }
  findSetup(name) {
    const walk = n => { if (!n) return null; if (n.dataset && n.dataset.setup === name) return n; for (const c of n.children || []) { const hit = walk(c); if (hit) return hit; } return null; };
    return walk(this.c.setup);
  }

  rolesPanel(team, d) {
    const box = el('div', 'cr-roles'), inferred = new Map();
    for (const r of ((this.setupAnswer && this.setupAnswer.inferred && this.setupAnswer.inferred[team]) || [])) if (r && typeof r.id === 'string') inferred.set(r.id, r);
    box.append(el('h4', 'cr-sub-h', 'Player roles (optional)'),
               el('p', 'cr-intro', 'Confirm a player’s role when you know it. PitchIQ never takes a role from a shirt number or a label. A confirmed role is used as given; when the measured positions contradict it, the report asks you to check it.'));
    const options = this.roleOptions(d);
    if (d.format.players < 11 && !d.formation.intended) box.append(el('p', 'cr-none', 'Positions are named by line (back, middle, front) and counted from the left; choose an intended formation to match your shape.'));
    const players = this.teamPlayers(team), unsettled = this.unsettledPlayers(team);
    if (!players.length) { box.append(el('p', 'cr-none', `No Team ${team} players in this run.`)); return box; }
    const list = el('div', 'cr-role-list');
    for (const {id, keeper} of players) {
      const row = el('div', 'cr-role'), name = el('span', 'cr-role-name'), inf = inferred.get(id), spans = d.roles[id] || [];
      const offered = this.playerRoles(options, keeper, team);
      row.dataset.player = id;
      name.append(...labelNodes(id, this.labels));
      const how = inf && inf.role ? `Measured: ${roleText(inf.role)}${inf.source ? ` (${inf.source}${inf.confidence ? `, ${inf.confidence}` : ''})` : ''}` : 'Measured: no stable role';
      row.append(name, el('span', 'cr-role-inf', how));
      if (spans.some(s => s.from !== null || s.to !== null)) row.append(el('span', 'cr-note', 'Roles for parts of the match are set: kept as they are.'));
      else {
        const current = spans.length ? spans[0].role : '', shown = current && !offered.includes(current) && options.includes(current) ? [current, ...offered] : offered;
        const opts = [['', 'Not confirmed'], ...shown.map(r => [r, roleText(r)])];
        const sel = selectEl(opts, current, v => this.editSetup(x => { if (v) x.roles[id] = [{role: v, from: null, to: null, set: isoNow()}]; else delete x.roles[id]; }), `Role of ${this.label(id)}`);
        sel.dataset.role = id;
        row.append(sel);
      }
      list.append(row);
    }
    box.append(list);
    if (unsettled) box.append(el('p', 'cr-note', `${unsettled} Team ${team} ${unsettled === 1 ? 'identity' : 'identities'} without a numbered slot (shown as ${team}-?) ${unsettled === 1 ? 'is' : 'are'} not listed: who ${unsettled === 1 ? 'it is' : 'they are'} is not settled yet.`));
    return box;
  }

  infoPanel(d) {
    // What the level looks for, PitchIQ's own rules for it and the sources behind them (DESIGN 2.7): from the report's
    // resolved profile when it matches the setup, else from the server's catalogue.
    const box = el('div', 'cr-info'), r = this.report(), cat = this.catalogue, lv = cat.levels.find(l => l.level === d.playingLevel);
    if (!lv) { box.append(el('p', 'cr-none', 'Choose a playing level to see what it looks for and the sources behind it.')); return box; }
    const p = isV2(r) && r.profile && r.profile.level === d.playingLevel && r.profile.ageGroup === d.ageGroup ? r.profile : null;
    const info = p || (cat.levelInfo && cat.levelInfo[lv.level]) || null;
    box.append(el('h4', 'cr-sub-h', `Level ${lv.level} · ${titleCase(lv.name)}${p && p.band ? ` (${p.band})` : ''}`), el('p', 'cr-intro', `Typical ${lv.typical}. ${lv.summary}`));
    if (!info) box.append(el('p', 'cr-none', 'Generate the report with this level and age group to see the level’s objectives, PitchIQ’s rules for it and the sources behind them.'));
    else {
      for (const [key, title] of [['objectives', 'Objectives'], ['principles', 'Principles'], ['notExpected', 'Not expected at this level']]) {
        const items = (Array.isArray(info[key]) ? info[key] : []).filter(g => g && typeof g.text === 'string');
        if (!items.length) continue;
        const ul = el('ul', 'cr-list');
        for (const g of items) {
          const li = el('li', null, g.text + ' ');
          li.append(el('span', `cr-tag ${g.tag === 'FEDERATION' ? 'f' : 'p'}`, g.tag === 'FEDERATION' ? `Federation: ${(g.sources || []).join(', ')}` : 'PitchIQ proposal'));
          ul.append(li);
        }
        box.append(el('div', 'cr-sub-h', title), ul);
      }
      const ev = info.evidence;
      if (ev && isNum(ev.minOpportunities)) box.append(el('div', 'cr-sub-h', 'PitchIQ’s rules for this level (proposals, not validated)'), el('p', null, evidenceRule(ev)));
      const sf = info.safety, heading = d.ageGroup && cat.headingByAge ? cat.headingByAge[d.ageGroup] : null;
      if (!sf && heading) box.append(el('div', 'cr-sub-h', 'Safety'), this.list([heading]));
      if (sf && typeof sf === 'object') {
        const lines = [sf.heading === 'none' ? 'No heading practice and no heading coaching points at this age.' : 'Heading practice only within the stricter of the U.S. Soccer and England limits; PitchIQ suggests no heading drill.'];
        if (sf.maxLoad) lines.push(`Physical load of drills: ${sf.maxLoad} at most.`);
        for (const n of sf.notes || []) if (typeof n === 'string') lines.push(n);
        box.append(el('div', 'cr-sub-h', 'Safety'), this.list(lines));
      }
    }
    const ids = info && Array.isArray(info.sources) ? info.sources : [];
    const refs = p && Array.isArray(r.sources) ? r.sources : ids.map(id => ({id, ...((cat.sources && cat.sources[id]) || {})}));
    if (refs.length) {
      const ul = el('ul', 'cr-list cr-sources');
      for (const s of refs.filter(x => x && x.id)) {
        const li = el('li');
        if (typeof s.url === 'string' && /^https?:\/\//.test(s.url)) {
          const a = el('a', null, s.title || s.id);
          a.href = s.url; a.target = '_blank'; a.rel = 'noopener noreferrer';
          li.append(a);
        } else li.append(document.createTextNode(s.title || s.id));
        li.append(document.createTextNode(`${s.publisher ? ` · ${s.publisher}` : ''} [${s.id}]`));
        ul.append(li);
      }
      box.append(el('div', 'cr-sub-h', 'Sources that informed this level'), ul);
    }
    box.append(el('p', 'cr-note', cat.sourcesNote || CATALOGUE.sourcesNote));
    return box;
  }

  async saveSetup() {
    const team = this.teamKey(), d = clone(this.setupBlock(team)), problems = setupProblems(d), run = this.runName();
    const name = this.squadName !== null ? asciiNote(this.squadName) : '';
    if (this.squadName !== null && !name) problems.push('Enter the new squad’s name, or choose No squad.');
    // The server keeps squad names to printable ASCII (they reach the PDF): said before the save, not as its 400
    // (CODE L11); smart quotes and dashes are folded to plain ones first.
    if (name && !/^[\x20-\x7e]+$/.test(name)) problems.push('The squad name can use plain letters, digits, spaces and punctuation only (no accents or symbols).');
    if (problems.length) { this.setupMessage = problems.join(' '); this.renderSetup(); return; }
    if (this.mode !== 'v2') {
      this.localSaved[team] = d; this.setupDraft[team] = null; this.setupOpen = false;
      this.setupMessage = 'Kept on this page: this server builds the earlier report, so only the level picks its wording.';
      this.cache.clear(); this.gen++;
      await this.request();
      return;
    }
    this.setupBusy = true; this.setupMessage = ''; this.renderSetup();
    let answer;
    try {
      if (name) {
        const made = await this.post('/api/coaching/squads', {name, ageGroup: d.ageGroup, level: d.playingLevel, format: d.format});
        const s = made && made.squad;
        if (!s || typeof s.id !== 'string') throw new Error('the server did not return the new squad');
        d.squad = {id: s.id, name: typeof s.name === 'string' ? s.name : name};
        linkRun(d, run, team);
        if (this.setupAnswer) this.setupAnswer.squads = (Array.isArray(this.setupAnswer.squads) ? this.setupAnswer.squads : []).concat(s);
      }
      answer = await this.post('/api/coaching/settings', {run, team, settings: d});
    } catch (error) {
      this.setupBusy = false; this.setupMessage = `The setup could not be saved: ${error.message}`; this.renderSetup();
      return;
    }
    this.setupBusy = false; this.squadName = null;
    const settings = answer && answer.settings && typeof answer.settings === 'object' ? answer.settings : null;
    const teams = Object.assign({A: null, B: null}, (this.setupAnswer && this.setupAnswer.settings && this.setupAnswer.settings.teams) || {}, {[team]: d});
    this.setupAnswer = Object.assign({}, this.setupAnswer || {}, {settings: settings || {version: 1, teams, updated: null}});
    this.setupDraft[team] = null; this.setupOpen = false;
    const stale = answer && Array.isArray(answer.stale) ? answer.stale : [];
    const note = answer && typeof answer.note === 'string' && answer.note ? ` ${answer.note}` : '';
    this.setupMessage = 'Saved.' + note;
    this.cache.clear(); this.gen++;
    await this.request();
    // What the save did, from the report answer that followed it (CODE L3: a format change back to a measurement made
    // before said "press Generate" beside "Report up to date", and a first save said "worked out again from the same
    // measurements" before anything was measured).
    const a = this.answer, ready = !!a && a.state === 'ready', measured = !!a && a.tiers && a.tiers.measurement === 'ready';
    let text = 'Saved.';
    if (ready) text = stale.includes('measurement') ? 'Saved. This run was already measured with this match format and field size: the report below is up to date.'
      : stale.includes('role') ? 'Saved. The positional checks were worked out again from the same measurements: the report below is up to date.'
      : stale.includes('interpretation') ? 'Saved. The report below was re-interpreted for this profile from the same measurements.' : 'Saved.';
    else if (a && a.state === 'missing' && !measured) text = 'Saved. Press Generate to measure this run’s situations with this setup (no new tracking).';
    else if (a && a.state === 'stale') text = 'Saved. The report below is from earlier settings (see why above): press Regenerate to measure this run’s situations again (no new tracking).';
    this.setupMessage = text + note;
    this.renderSetup();
  }

  // ================= coaching v2: the report (DESIGN 9.2) =================
  findingsV2(r) { return ['strengths', 'needsWork', 'reviews', 'described'].flatMap(k => (Array.isArray(r[k]) ? r[k] : []).filter(f => f && typeof f.id === 'string')); }
  findingById(id) { const r = this.report(); return id && isV2(r) ? this.findingsV2(r).find(f => f.id === id) || null : null; }
  findingByKey(key) { const r = this.report(); return key && isV2(r) ? this.findingsV2(r).find(f => f.key === key) || null : null; }
  momentOf(id) { const r = this.report(); return id && isV2(r) && Array.isArray(r.keyMoments) ? r.keyMoments.find(m => m && m.id === id) || null : null; }
  momentsOf(f) { return (Array.isArray(f.timestamps) ? f.timestamps : []).map(ts => ts && this.momentOf(ts.evidence)).filter(Boolean); }
  alternativeOf(id) { const r = this.report(); return id && isV2(r) && Array.isArray(r.alternatives) ? r.alternatives.find(a => a && a.id === id) || null : null; }
  noAlternativeOf(ev) {
    const r = this.report(), list = isV2(r) && Array.isArray(r.noAlternatives) ? r.noAlternatives.filter(n => n && n.reason) : [];
    return list.find(n => n.evidence === ev.id) || list.find(n => n.finding === ev.finding && !n.evidence) || null;
  }
  profileV2() { const r = this.report(); return isV2(r) && r.profile && typeof r.profile === 'object' ? r.profile : {}; }
  numbers() { const reg = this.profileV2().register; return reg && typeof reg.numbers === 'string' ? reg.numbers : 'metres'; }
  skillName(detector) {
    // The level's own name of a check (profile.skills, DESIGN 8.3), else a plain fallback.
    const p = this.profileV2(), name = p.skills && p.skills[SKILL_OF[detector]];
    return typeof name === 'string' && name ? name : DETECTOR_UI[detector] || words(detector || 'check');
  }
  phaseWords(ph, f) {
    // DESIGN 3.2 rule 8: "when we had the ball (estimated)"; a coach-marked span is said to be the coach's, and a
    // finding resting on both (measurements.phaseSource 'mixed', DETECT) says so.
    if (!ph || typeof ph !== 'object') return null;
    const w = {own: 'when we had the ball', opp: 'when the other team had the ball', dead: 'while the ball was out of play'}[ph.state];
    const source = f && f.measurements ? f.measurements.phaseSource : null;
    return w ? `${w} (${ph.estimated === false || source === 'coach' ? 'marked by the coach' : source === 'mixed' ? 'estimated, partly marked by the coach' : 'estimated'})` : null;
  }
  drillName(id) { const lib = this.answer && this.answer.drills, d = lib && typeof lib === 'object' ? lib[id] : null; return d && typeof d.name === 'string' && d.name ? d.name : words(id); }
  sourceTitle(id) {
    const r = this.report(), s = isV2(r) && Array.isArray(r.sources) ? r.sources.find(x => x && x.id === id) : null, c = this.catalogue.sources && this.catalogue.sources[id];
    return (s && s.title) || (c && c.title) || id;
  }

  emptyV2(a) {
    const main = this.main, o = this.options(), d = this.setupBlock(o.team);
    if (!a || this.loading) { main.append(this.empty('Loading the coaching report…')); return; }
    if (a.state === 'building') { main.append(this.empty('Building the report: the situations are measured from this run’s tracking (no new tracking). This takes a few seconds.')); return; }
    if (a.state === 'error') { main.append(this.empty(a.error || 'The report could not be built.')); return; }
    const given = a.options && typeof a.options === 'object' && 't0' in a.options ? a.options : o, whole = given.t0 === null || given.t0 === undefined;
    const box = this.empty(`No coaching report yet for Team ${o.team} (${whole ? 'whole analysed segment' : `${clock(given.t0)}–${clock(given.t1)}`}; ${this.setupSummary(d)}).`);
    box.append(el('p', 'cr-intro', 'Generate measures this run’s situations once, from the tracking you already have. Changing the age group, the level or the style afterwards re-interprets the same measurements; only the match format, the field size or the possession you mark measures them again. Moments PitchIQ cannot fully confirm are shown as possible issues to review.'));
    main.append(box);
  }

  renderV2(r) {
    this.drillBoxes = new Map();
    this.main.append(this.overviewV2(r), this.wentWellV2(r), this.needsWorkV2(r), this.keyMomentsV2(r), this.diagramsV2(r), this.drillsV2(r), this.progressionV2(r));
    const apart = this.feedbackSectionV2(r);
    if (apart) this.main.append(apart);
    if (!this.selected || !this.findingById(this.selected)) {
      const first = [].concat(r.needsWork || [], r.reviews || [], r.strengths || [], r.described || []).find(f => f && Array.isArray(f.timestamps) && f.timestamps.length);
      this.selected = first ? first.id : null;
      this.selectedMoment = first ? first.timestamps[0].evidence : null;
    }
    if (this.selectedMoment && !this.momentOf(this.selectedMoment)) this.selectedMoment = null;
    if (this.adjusting && this.adjusting.evidence !== this.selectedMoment) this.adjusting = null;
    this.markSelection();
  }
  markSelection() {
    for (const [id, card] of this.cards) card.classList.toggle('selected', id === this.selected);
    for (const row of this.momentRows) row.classList.toggle('selected', row.dataset.evidence === this.selectedMoment);
  }

  overviewV2(r) {
    const s = this.section('overview', 'Match overview'), p = r.profile || {}, ov = r.overview || {}, range = r.range || {}, f = p.format || {};
    const head = el('div', 'cr-ov-head');
    head.append(el('div', 'cr-team', r.teamName || `Team ${r.team} (as labelled in this run)`));
    s.append(head);
    const style = this.catalogue.styles.find(x => x.id === p.style), rows = [];
    rows.push(['Coaching profile', `${p.ageGroup || '–'} · level ${p.level || '–'} ${titleCase(p.levelName || '')}${p.band ? ` (${p.band})` : ''}`]);
    rows.push(['Match format', `${f.name || '–'}${f.goalkeeper === false ? ' without goalkeeper' : ''}${isNum(f.outfield) ? ` · ${f.outfield} outfield players` : ''}`]);
    rows.push(['Intended formation', p.formation ? `${p.formation} (a starting model, not where every player must stand)` : 'Other / not sure']);
    rows.push(['Style of play', !style || p.style === 'unknown' ? 'Unknown: no intended style is assumed' : style.name]);
    if (isNum(range.t0) && isNum(range.t1)) rows.push(['Range', `${clock(range.t0)}–${clock(range.t1)} (${seconds(range.seconds)}${range.whole ? ', the whole analysed segment' : ''})`]);
    rows.push(['Footage', this.footageText(ov.footage || {})]);
    rows.push(['Pitch scale', this.scaleText(ov.scale || {})]);
    rows.push(['Who had the ball', this.possessionText(ov.possession || {})]);
    const dq = this.qualityText(ov.dataQuality || {});
    if (dq) rows.push(['Data quality', dq]);
    for (const [k, v] of rows) s.append(this.kv(k, v));
    if (ov.possession && ov.possession.available && ov.possession.caveat) s.append(this.para(null, ov.possession.caveat, 'cr-intro'));
    const notices = uniqueBy([].concat(ov.notices || [], p.notices || []).filter(n => n && typeof n.text === 'string'), n => n.code);
    if (notices.length) { s.append(el('h4', 'cr-sub-h', 'Notes on this profile')); s.append(this.list(notices.map(n => n.text))); }
    const phaseOff = notices.some(n => n.code === 'phase_unavailable') || (ov.possession && ov.possession.available === false);
    const oneSided = this.oneSidedNote(r);
    if (oneSided) s.append(this.marksReady() ? this.marksCall(oneSided) : this.para(null, oneSided, 'cr-intro'));
    else if (phaseOff && this.marksReady()) s.append(this.marksCall('You can mark who had the ball yourself while you watch: our ball, their ball, dead ball. The checks that need it are then made from your marks.'));
    if (typeof ov.text === 'string' && ov.text) s.append(this.para(null, ov.text));
    const periods = (ov.formations || []).filter(x => x && isNum(x.t0) && isNum(x.t1));
    if (periods.length) {
      s.append(el('h4', 'cr-sub-h', 'Shape measured (described, not judged)'));
      s.append(this.list(periods.map(x => `${clock(x.t0)}–${clock(x.t1)}: ${x.label || 'no clear shape'}${x.confidence ? ` (${x.confidence})` : ''}`)));
    }
    const dets = (ov.detectors || []).filter(x => x && x.detector);
    if (dets.length) {
      s.append(el('h4', 'cr-sub-h', 'What was checked'));
      const ul = el('ul', 'cr-list cr-checks');
      for (const d of dets) {
        const li = el('li');
        li.dataset.detector = d.detector; li.dataset.status = d.status;
        const side = d.reason === 'coach_marks_one_side' ? this.markedSide() : null;
        const few = d.reason === 'too_few_situations' && this.numbers() !== 'none' && Number.isInteger(d.opportunities) && d.opportunities > 0 ? fewSituations(d.opportunities) : null;
        const reason = side === 'A' || side === 'B' ? `you marked only Team ${side}’s possession: mark Team ${side === 'A' ? 'B' : 'A'}’s spells too to run this check` : few || REASON_UI[d.reason];
        let why = d.status === 'assessed' ? this.assessedText(d) : d.status === 'off' ? REASON_UI.off : `not assessed: ${reason || words(d.reason || 'not enough data').toLowerCase()}`;
        if (d.status === 'not_assessed' && PHASE_REASONS.includes(d.reason) && this.marksReady()) why += ' (marking who had the ball under the video would unlock it)';
        li.append(el('b', null, `${this.skillName(d.detector)}: `), document.createTextNode(why));
        ul.append(li);
      }
      s.append(ul);
    }
    const un = (ov.unavailable || []).filter(x => x && x.category && x.reason);
    if (un.length) {
      // What PitchIQ does not measure, named with its reason (DESIGN 4.7): the one place passes or duels may be named,
      // as things not measured (marked, so the page's word checks leave it out).
      const list = this.list(un.map(x => `${words(x.category)}: ${x.reason}${x.unlock ? ` What would make it possible: ${x.unlock}` : ''}`));
      list.dataset.unmeasured = 'categories';
      s.append(el('h4', 'cr-sub-h', 'Not measured by PitchIQ'), list);
    }
    const described = (r.described || []).filter(x => x && x.id);
    if (described.length) { s.append(el('h4', 'cr-sub-h', 'Context from this match')); for (const x of described) s.append(this.cardV2(x)); }
    const held = (r.withheld || []).length;
    if (held) s.append(el('p', 'cr-intro', `${held} ${held === 1 ? 'finding was' : 'findings were'} held back: ${held === 1 ? 'its' : 'their'} wording could not be checked against the measurements.`));
    return s;
  }
  assessedText(d) {
    if (this.numbers() === 'none' || !Number.isInteger(d.opportunities)) return 'checked';
    return `checked in ${d.opportunities} ${plural(d.opportunities, 'moment')}${isNum(d.eligibleSeconds) ? ` (${seconds(d.eligibleSeconds)} usable)` : ''}`;
  }
  footageText(fo) {
    const kind = {broadcast: 'Broadcast camera', amateur: 'Amateur camera', veo: 'Automatic camera (Veo type)'}[fo.kind] || 'Camera type not known', more = [];
    if (isNum(fo.smallSceneShare) && fo.smallSceneShare >= .2) more.push(`players appear small in ${pct(fo.smallSceneShare)} of it`);
    if (fo.keyless) more.push('no pitch keyframes were set');
    if (fo.lensBent) more.push('a curved lens');
    return kind + (more.length ? `: ${more.join(', ')}` : '');
  }
  scaleText(sc) {
    if (!isNum(sc.lengthM) || !isNum(sc.widthM)) return 'not known';
    const how = {measured: 'measured by calibration', declared: 'entered in the coaching setup', assumed: 'assumed'}[sc.source] || 'assumed';
    const tail = sc.metresUsable === false ? '; distances between players are not judged (enter the field size in the setup)' : sc.source === 'assumed' ? '; distances are estimates' : '';
    return `${Math.round(sc.lengthM)} × ${Math.round(sc.widthM)} m, ${how}${tail}`;
  }
  markedSide() {
    // Whose possession the coach marked on this run: null (none), 'A' or 'B' (one team only) or 'both'. From the marks
    // on this page; when it holds none (marks saved elsewhere, not loaded yet), from the report itself: shares of 100%
    // to 0% from the coach's marks alone, or a check left out for coach_marks_one_side (browser check on the AM copy:
    // spans posted outside the page left the page's list empty, and 100% / 0% was shown).
    const spans = this.marks && Array.isArray(this.marks.spans) ? this.marks.spans : [];
    const teams = new Set(spans.map(s => s && s.state).filter(s => s === 'A' || s === 'B'));
    if (teams.size) return teams.size === 2 ? 'both' : [...teams][0];
    const r = this.report(), ov = (isV2(r) && r.overview) || {}, po = ov.possession || {}, rs = po.roughShares;
    if (!(this.answer && this.answer.coachSpans) || !rs || !isNum(rs.A) || !isNum(rs.B)) return null;
    const full = rs.A >= .999 ? 'A' : rs.B >= .999 ? 'B' : null;
    const told = (ov.detectors || []).some(d => d && d.reason === 'coach_marks_one_side');
    return full && (told || this.possessionSource(po) === 'coach') ? full : null;
  }
  possessionSource(po) {
    // Where the shares come from: the report's own word when it gives one (po.source, requested of TEXT), else the
    // run's coach marks (SERVER: answer.coachSpans) and the caveat report.py writes when only the coach's marks name
    // the phase ("PitchIQ could not follow it on this footage").
    if (po.source === 'engine' || po.source === 'coach' || po.source === 'mixed') return po.source;
    if (!(this.answer && this.answer.coachSpans)) return 'engine';
    return /could not follow/i.test(String(po.caveat || '')) ? 'coach' : 'mixed';
  }
  possessionText(po) {
    // Who had the ball, said with where it comes from (critic B-11): shares from the coach's marks are called the
    // coach's, never an estimate (on the amateur clip they came only from the marks); marks laid over PitchIQ's estimate
    // say both; and no share is given from marks of one team only when PitchIQ could not follow the ball, where it
    // read 100% / 0% (critic N6 on the AM copy: Team A's two spells only gave "A 1.0, B 0.0").
    if (!po.available) return `Not available on this footage${po.reason ? `: ${po.reason}` : ''}`;
    const source = this.possessionSource(po), side = this.markedSide(), one = side === 'A' || side === 'B';
    const cover = isNum(po.coverage) ? `${pct(po.coverage)} of the time` : null;
    const how = source === 'coach' ? `from your marks${cover ? `, ${cover}` : ''}`
      : source === 'mixed' ? `your marks where you made them, PitchIQ’s estimate elsewhere${cover ? `; from ${cover}` : ''}`
      : cover ? `estimated from ${cover}` : 'estimated';
    const rs = po.roughShares, shares = rs && isNum(rs.A) && isNum(rs.B) && !(source === 'coach' && one);
    let text = shares ? `Team A about ${pct(rs.A)}, Team B about ${pct(rs.B)} (${how})` : how.charAt(0).toUpperCase() + how.slice(1);
    if (!shares && source === 'coach' && one) text = `No share given: you marked only Team ${side}’s possession${cover ? ` (${cover})` : ''}; mark Team ${side === 'A' ? 'B' : 'A'}’s spells too for a share`;
    return text + (po.quality === 'limited' ? '; limited quality' : '');
  }
  oneSidedNote(r) {
    // The coach marked one team's possession and PitchIQ could not follow the ball: the checks played while the other
    // team had it cannot be run, and the report's phase notice is gone (the phase is known where marked), so the
    // overview says what is missing (critic N6).
    const side = this.markedSide(), po = ((r && r.overview) || {}).possession || {};
    if (side !== 'A' && side !== 'B') return null;
    const dets = (((r && r.overview) || {}).detectors || []);
    if (this.possessionSource(po) !== 'coach' && !dets.some(d => d && d.reason === 'coach_marks_one_side')) return null;
    const other = side === 'A' ? 'B' : 'A';
    return `You marked only Team ${side}’s possession. PitchIQ could not follow the ball on this footage, so the checks played while Team ${other} had the ball are not assessed until you mark Team ${other}’s spells too.`;
  }
  qualityText(dq) {
    const parts = [];
    if (isNum(dq.positioned)) parts.push(`about ${dq.positioned.toFixed(1)} players per team placed on the pitch`);
    if (Number.isInteger(dq.lineup)) parts.push(`lineup of ${dq.lineup}`);
    const cause = {registration: 'mostly limited by the pitch registration', identity: 'mostly limited by player identities', out_of_view: 'mostly limited by players out of the camera view'}[dq.dominantCause];
    if (cause) parts.push(cause);
    return parts.join('; ');
  }

  wentWellV2(r) {
    const items = (r.strengths || []).filter(f => f && f.id), s = this.section('wentWell', `What went well · ${items.length}`);
    if (!items.length) s.append(el('p', 'cr-none', 'No strength met this level’s evidence bar in this range. A strength needs the behaviour actually shown in separate moments, never only the absence of problems.'));
    for (const f of items) s.append(this.cardV2(f));
    return s;
  }
  needsWorkV2(r) {
    const work = (r.needsWork || []).filter(f => f && f.id), reviews = (r.reviews || []).filter(f => f && f.id), p = this.profileV2();
    const issues = reviews.filter(f => !isMoments(f)), moments = reviews.filter(isMoments);
    const s = this.section('needsWork', `What needs work · ${work.length + issues.length}`), max = p.evidence && Number.isInteger(p.evidence.maxWorkItems) ? p.evidence.maxWorkItems : null;
    if (max !== null) s.append(el('p', 'cr-intro', `At most ${max} ${max === 1 ? 'item is' : 'items are'} shown at this level, the most important first; nothing is added to fill the section.`));
    if (!work.length) s.append(el('p', 'cr-none', 'Nothing met this level’s evidence bar for a work item in this range.'));
    for (const f of work) s.append(this.cardV2(f));
    if (issues.length) {
      s.append(el('h4', 'cr-sub-h', REVIEW_PARTS_UI.issues[0]), el('p', 'cr-intro', REVIEW_PARTS_UI.issues[1]));
      for (const f of issues) s.append(this.cardV2(f));
    }
    if (moments.length) {
      const h = el('h4', 'cr-sub-h', `${REVIEW_PARTS_UI.moments[0]} · ${moments.length}`);
      h.dataset.part = 'moments';
      s.append(h, el('p', 'cr-intro', REVIEW_PARTS_UI.moments[1]));
      for (const f of moments) s.append(this.cardV2(f));
    }
    return s;
  }

  cardV2(f) {
    const mw = isMoments(f), [badge, kind] = verdictUi(f), card = el('article', `cr-finding k-${kind}`), t = f.text || {};
    card.dataset.finding = f.id; card.dataset.verdict = f.verdict;
    if (mw) card.dataset.part = 'moments';
    const head = el('div', 'cr-f-head');
    head.append(el('span', 'cr-badge', badge), el('span', 'cr-conf', this.skillName(f.observation)));
    if (Number.isInteger(f.priority) && (f.verdict === 'work_on' || f.verdict === 'try_this')) head.append(el('span', 'cr-prio', `Priority ${f.priority}`));
    card.append(head);
    const h = el('h4');
    h.append(...labelNodes(t.headline || this.skillName(f.observation), this.labels));
    card.append(h);
    if (t.explanation) card.append(this.para(null, t.explanation));
    if (t.action) card.append(this.para(f.verdict === 'strength' ? 'To build on it.' : f.verdict === 'try_this' ? null : mw ? 'What to look for.' : 'What could help.', actionText(f), f.verdict === 'try_this' ? 'cr-try' : null));
    const facts = this.factLine(f);
    if (facts) card.append(el('p', 'cr-facts', facts));
    const chips = this.measureChips(f);
    if (chips) card.append(chips);
    for (const n of this.guardNotes(f)) card.append(this.para(null, n, 'cr-intro'));
    const act = mw ? null : this.actionNote(f);
    if (act) card.append(this.para(null, act, 'cr-intro'));
    const moments = this.momentsOf(f), actions = el('div', 'cr-actions');
    if (moments.length) actions.append(button(`Moments (${moments.length})`, 'quiet cr-show', () => this.selectMoment(moments[0].id), 'Show the key moments of this finding beside the report'));
    for (const ev of moments) {
      const b = button(clock(ev.time), 'cr-time', () => this.showMoment(ev, 'show'), `Show ${clock(ev.time)} with its annotation (segment ${clock(ev.segmentTime)})`);
      b.dataset.evidence = ev.id;
      actions.append(b);
    }
    for (const id of Array.isArray(f.drills) ? f.drills : []) {
      const b = button(`Drill: ${this.drillName(id)}`, 'quiet cr-drill-ref', () => { const box = this.drillBoxes && this.drillBoxes.get(id); if (box && box.scrollIntoView) box.scrollIntoView({behavior: 'smooth', block: 'center'}); });
      b.dataset.drill = id;
      actions.append(b);
    }
    if (actions.children.length) card.append(actions);
    if (f.verdict !== 'describe') card.append(this.feedbackBlock(f));
    this.cards.set(f.id, card);
    return card;
  }
  factLine(f) {
    // The counts behind a finding in the level's register (profile.register.numbers): none at level 1, counts at
    // level 2, seconds and rates further up.
    const n = this.numbers(), a = f.aggregate || {}, ph = this.phaseWords(f.context && f.context.phase, f), m = f.measurements || {};
    const mw = isMoments(f) && ['clearMoments', 'clearSpells', 'decidedMoments'].every(k => Number.isInteger(m[k]));
    if (!Number.isInteger(a.opportunities) || a.opportunities === 0) return null;
    if (n === 'none') return ph ? `Looked at ${ph}.` : null;
    let text;
    if (f.verdict === 'strength') text = `Shown in ${a.demonstrated} of ${a.opportunities} moments (${a.distinctDemonstratedSpells} separate ${plural(a.distinctDemonstratedSpells, 'spell')})`;
    else if (f.verdict === 'describe') text = `${a.opportunities} ${plural(a.opportunities, 'moment')} looked at`;
    else if (mw) {
      // The counts add up (critic B-3, FACTS F-4 / COACH F-10 of round 2): the failures that were not clear are said too.
      text = `Clear in ${m.clearMoments >= m.decidedMoments ? `all ${m.decidedMoments}` : `${m.clearMoments} of the ${m.decidedMoments}`} moments PitchIQ could judge (${m.clearSpells} separate ${plural(m.clearSpells, 'spell')})`;
      if (Number.isInteger(m.failingMoments) && m.failingMoments > m.clearMoments) text += `, ${m.failingMoments - m.clearMoments} more fell short less clearly`;
    }
    else text = `Missing in ${a.notDemonstrated} of ${a.opportunities} moments (${a.distinctFailingSpells} separate ${plural(a.distinctFailingSpells, 'spell')})`;
    if (a.unclear) text += `, ${a.unclear} unclear`;
    if (ph) text += `, ${ph}`;
    if (n !== 'counts' && isNum(a.phaseSeconds) && a.phaseSeconds > 0) text += `, over ${seconds(a.phaseSeconds)} of that time`;
    if (n === 'rates' && isNum(a.ratePer10) && !mw) text += ` (${a.ratePer10.toFixed(1)} per 10 minutes of it)`;
    return text + '.';
  }
  measureChips(f) {
    // The finding's measurements, only from level 3 up (metres, rounded at level 3; shares from level 5), each from the
    // level MEASURE_UI gives it; metres only where the scale is usable (m.metres 'unusable': a small field without a
    // declared size), marked as estimates on an assumed pitch.
    const n = this.numbers(), level = Number.isInteger(f.level) ? f.level : this.profileV2().level;
    if (n === 'none' || n === 'counts') return null;
    const m = f.measurements || {}, est = m.metres === 'estimated', box = el('div', 'cr-metrics'), first = chipLevels(f);
    for (const [k, v] of Object.entries(m)) {
      const ui = MEASURE_UI[k];
      if (!ui || v === null || v === undefined || first[k] === null || (Number.isInteger(level) && level < first[k])) continue;
      let text = null;
      if (ui[1] === 'bool') text = typeof v === 'boolean' ? (v ? 'yes' : 'no') : null;
      else if (!isNum(v)) text = null;
      else if (ui[1] === 'share') text = n === 'shares' || n === 'rates' ? pct(v) : null;
      else if (ui[1] === 'm') text = m.metres === 'unusable' ? null : `${n === 'rounded_metres' ? Math.round(v) : v.toFixed(1)} m${est ? ' (est.)' : ''}`;
      else if (ui[1] === 'deg') text = `${Math.round(v)}°`;
      else if (ui[1] === 'signed') text = `${v > 0 ? '+' : ''}${Math.round(v)}`;
      else text = String(Math.round(v));
      if (text === null) continue;
      const chip = el('span', 'cr-metric');
      chip.dataset.measure = k;
      chip.append(document.createTextNode(ui[0] + ' '), el('b', null, text));
      box.append(chip);
    }
    if (m.phaseSource === 'coach' || m.phaseSource === 'mixed') box.append(el('span', 'cr-metric cr-coach-tag', m.phaseSource === 'coach' ? 'Who had the ball: marked by the coach' : 'Who had the ball: partly marked by the coach'));
    return box.children.length ? box : null;
  }
  guardNotes(f) {
    // What PitchIQ could not confirm (DESIGN 4.8 guardrails), the limits, and a style the reading depends on.
    const out = [], g = f.guardrails || {};
    const failed = Object.keys(GUARDRAIL_UI).filter(k => g[k] === 'fail' && !(k === 'evidence' && isMoments(f)));
    if (failed.length) out.push(`Could not be confirmed: ${failed.map(k => GUARDRAIL_UI[k]).join(', ')}.`);
    const lim = (Array.isArray(f.limitations) ? f.limitations : []).filter(c => REASON_UI[c]).map(c => REASON_UI[c]);
    if (lim.length) out.push(`Limits: ${lim.join('; ')}.`);
    if (f.context && f.context.styleEffect) {
      const st = this.catalogue.styles.find(x => x.id === f.context.style);
      out.push(`Read with the style you chose${st ? ` (${st.name.toLowerCase()})` : ''}.`);
    }
    return out;
  }
  actionNote(f) {
    const sa = f.suggestedAction;
    if (!sa || f.verdict === 'strength' || f.verdict === 'describe') return null;
    if (sa.mode === 'zone' || sa.mode === 'point') return 'A possible better area is drawn on the key moment: one player moved, everyone else kept where they were.';
    if (sa.mode === 'direction') return 'The direction of a possible movement is drawn on the key moment; the area itself was too large to show as one place.';
    return 'No reachable better position could be confirmed, so only the principle is given.';
  }

  keyMomentsV2(r) {
    const list = (r.keyMoments || []).filter(m => m && m.id);
    const s = this.section('keyMoments', `Key moments · ${list.length}`, list.length ? 'Each opens the video (or the 2D match) at its exact frame with its annotation; Replay plays it from three seconds before with the usual playback controls.' : null);
    if (!list.length) { s.append(el('p', 'cr-none', 'No key moment in this range.')); return s; }
    const ol = el('ol', 'cr-km');
    for (const ev of list) {
      const f = this.findingById(ev.finding), [badge, kind] = f ? verdictUi(f) : ['', 'describe'], row = el('li', `cr-km-row k-${kind}`);
      row.dataset.evidence = ev.id;
      const time = button(clock(ev.time), 'cr-time', () => this.showMoment(ev, 'show'), `Show ${clock(ev.time)} with its annotation (segment ${clock(ev.segmentTime)})`);
      time.dataset.evidence = ev.id;
      const body = el('div', 'cr-km-body'), title = el('div', 'cr-km-title'), head = f && f.text ? f.text.short || f.text.headline : this.skillName(ev.detector);
      title.append(...labelNodes(head, this.labels));
      if (badge) title.append(el('span', 'cr-km-badge', badge));
      body.append(title);
      // The report's captions are the findings' short texts (evidence.apply_captions), the row's title already: a
      // caption is shown only when it says something else (on BC180 every caption repeated its title).
      if (ev.caption && ev.caption !== head) body.append(this.para(null, ev.caption, 'cr-km-cap'));
      const meta = el('div', 'cr-km-meta'), tier = TIER_UI[ev.tier];
      meta.append(el('span', `cr-tier t-${ev.tier}`, tier ? tier[0] : String(ev.tier)));
      if (Array.isArray(ev.players) && ev.players.length) { const who = el('span', 'cr-km-who'); who.append(...labelNodes(ev.players.join(', '), this.labels)); meta.append(who); }
      meta.append(el('span', 'cr-seg', `segment ${clock(ev.segmentTime)}`));
      body.append(meta);
      const acts = el('div', 'cr-km-acts');
      const replay = button('Replay', 'quiet', () => this.showMoment(ev, 'replay'), 'Play this moment from three seconds before, with its annotation');
      replay.dataset.replay = ev.id; replay.disabled = !(ev.annotation && ev.annotation.window);
      const board = button('2D', 'quiet', () => this.showMoment(ev, 'board'), 'Show this moment on the 2D match view');
      board.dataset.board = ev.id;
      acts.append(replay, board);
      row.append(time, body, acts);
      row.addEventListener('click', e => { if (e && e.target && e.target.tagName === 'BUTTON') return; this.selectMoment(ev.id); });
      this.momentRows.push(row);
      ol.append(row);
    }
    s.append(ol);
    return s;
  }

  diagramsV2(r) {
    const list = (r.diagrams || []).filter(d => d && d.diagram && Array.isArray(d.diagram.elements)), s = this.section('diagrams', `Tactical diagrams · ${list.length}`);
    if (!list.length) { s.append(el('p', 'cr-none', 'No diagram in this range.')); return s; }
    const grid = el('div', 'cr-dg-grid'), colours = this.colours(), pal = annotationPalette(colours.A, colours.B);
    for (const d of list) {
      const fig = el('figure', 'cr-dg'), cap = el('figcaption');
      fig.dataset.diagram = d.id || '';
      fig.append(svgDiagram(d.diagram, colours, this.labels, {palette: pal, adjustments: this.adjustmentsFor(d.evidence)}));
      cap.append(el('span', 'cr-dg-kind', {finding: 'Finding', formation: 'Shape', movement: 'Movement'}[d.kind] || 'Diagram'), ...labelNodes(` ${d.caption || d.diagram.title || ''}`, this.labels));
      fig.append(cap);
      const ev = d.evidence ? this.momentOf(d.evidence) : null;
      if (ev) { const b = button(`Show ${clock(ev.time)}`, 'quiet cr-time', () => this.showMoment(ev, 'show')); b.dataset.evidence = ev.id; fig.append(b); }
      grid.append(fig);
    }
    s.append(grid);
    if (list.some(d => d.diagram.elements.some(e => e && e.type === 'suggestedZone')))
      s.append(el('p', 'cr-note', 'A dotted area is a possible better position, with a dashed arrow for the movement towards it: an illustration with one player moved and everyone else kept where they were, never an exact spot.'));
    if (list.some(d => d.diagram.metresEstimated !== false)) s.append(el('p', 'cr-note', 'Positions are the observed ones at each frame; distances are estimates.'));
    return s;
  }

  drillsV2(r) {
    const picks = (r.drills || []).filter(d => d && typeof d.drill === 'string'), s = this.section('drills', `Training drills · ${picks.length}`);
    if (!picks.length) { s.append(el('p', 'cr-none', noDrillText(r))); return s; }
    const lib = this.answer && this.answer.drills && typeof this.answer.drills === 'object' ? this.answer.drills : {};
    for (const pick of picks) s.append(this.drillV2(pick, lib[pick.drill] && typeof lib[pick.drill] === 'object' ? lib[pick.drill] : null));
    return s;
  }
  drillV2(pick, d) {
    // DESIGN 7.3: what it trains, why it was chosen, the moments it addresses, easier, harder, what progress looks like;
    // then the library record (when the server sent it) with its source credit.
    const box = el('div', 'cr-drill'), f = pick.finding ? this.findingById(pick.finding) : null, fields = pick.fields || {};
    box.dataset.drill = pick.drill;
    if (this.drillBoxes && !this.drillBoxes.has(pick.drill)) this.drillBoxes.set(pick.drill, box);
    box.append(el('h4', null, d && d.name ? d.name : words(pick.drill)));
    if (f && !pick.optional) { const p = el('div', 'cr-links'); p.append(el('span', 'cr-intro', 'For: '), this.findingLink(f)); box.append(p); }
    else box.append(el('p', 'cr-intro', optionalLabel(this.report())));
    const field = (key, lead) => { if (typeof fields[key] === 'string' && fields[key]) box.append(this.para(lead, fields[key])); };
    field('trains', 'What it trains.');
    field('why', 'Why it was chosen.');
    const moments = (Array.isArray(fields.moments) ? fields.moments : []).map(id => this.momentOf(id)).filter(Boolean);
    if (moments.length) {
      const row = el('div', 'cr-links');
      row.append(el('span', 'cr-intro', 'Moments it addresses: '));
      for (const ev of moments) { const b = button(clock(ev.time), 'cr-time', () => this.showMoment(ev, 'show')); b.dataset.evidence = ev.id; row.append(b); }
      box.append(row);
    }
    field('easier', 'Make it easier.');
    field('harder', 'Make it harder.');
    field('progress', 'What progress looks like next match.');
    if (d) {
      const pc = d.player_count, du = d.duration, fs = d.field_setup, dim = v => isNum(v) ? v : v && isNum(v.value) ? v.value : null;
      if (pc && isNum(pc.min) && isNum(pc.max)) box.append(this.para('Players.', `${pc.min === pc.max ? pc.min : `${pc.min}–${pc.max}`} players${pc.note ? `. ${pc.note}` : ''}`));
      if (du && isNum(du.minutes)) box.append(this.para('Duration.', `${du.minutes} min${du.blocks ? ` (${du.blocks})` : ''}`));
      if (fs && typeof fs === 'object') {
        const L = dim(fs.length), W = dim(fs.width), area = [L !== null && W !== null ? `About ${L} × ${W} m` : '', fs.notes || ''].filter(Boolean).join('. ');
        if (area) box.append(this.para('Area.', area));
      }
      for (const [key, title] of [['equipment', 'Equipment'], ['instructions', 'How it runs'], ['coaching_points', 'Coaching points']])
        if (Array.isArray(d[key]) && d[key].length) box.append(el('div', 'cr-sub-h', title), this.list(d[key].map(String)));
      const sf = d.safety;
      if (sf && typeof sf === 'object')
        box.append(this.para('Safety.', [sf.heading === 'none' ? 'no heading' : 'heading only within the age limits', `${sf.physicalLoad || 'low'} physical load`, sf.contact === 'no_tackle' ? 'no tackling' : null].filter(Boolean).join(', ') + '.'));
      const refs = (Array.isArray(d.source_reference) ? d.source_reference : []).filter(x => x && x.source);
      if (refs.length) {
        // A source is credited by its own title (a quotation of the federation's document, so a level's banned words
        // can appear in it, e.g. "Pressing and covering session" under a level-1 drill); marked as a credit.
        const credit = this.para('Source.', refs.map(x => `${DRILL_SOURCE_UI[x.relation] || 'From'} ${this.sourceTitle(x.source)}${x.note ? ` (${x.note})` : ''}`).join('; ') + '.', 'cr-intro cr-credit');
        credit.dataset.credit = 'source';
        box.append(credit);
      }
    }
    return box;
  }
  findingLink(f) {
    const b = button('', 'cr-link', () => {
      const ev = this.momentsOf(f)[0];
      if (ev) this.selectMoment(ev.id); else { this.selected = f.id; this.markSelection(); this.renderSide(); }
      const card = this.cards.get(f.id);
      if (card && card.scrollIntoView) card.scrollIntoView({behavior: 'smooth', block: 'center'});
    });
    b.append(...labelNodes(f.text ? f.text.headline : this.skillName(f.observation), this.labels));
    b.dataset.finding = f.id;
    return b;
  }

  progressionV2(r) {
    const s = this.section('progression', 'Development progression'), pr = r.progression, skills = pr && Array.isArray(pr.skills) ? pr.skills.filter(x => x && x.skill) : [];
    if (skills.length) {
      const table = el('div', 'cr-prog'), headRow = el('div', 'cr-prog-row head');
      for (const t of ['Skill', 'State', 'Moments', 'Matches', 'Trend']) headRow.append(el('span', null, t));
      table.append(headRow);
      const squad = typeof pr.squad === 'string' ? pr.squad : pr.squad && typeof pr.squad.id === 'string' ? pr.squad.id : null;
      for (const k of skills) {
        const row = el('div', 'cr-prog-row'), name = el('span', 'cr-prog-name');
        row.dataset.skill = k.skill;
        name.append(document.createTextNode(k.name || words(k.skill)));
        const holder = el('div', 'cr-prog-detail');
        holder.hidden = true; holder.dataset.progressDetail = k.skill;
        if (squad) {
          // The skill's detail (PROGRESS mastery(): why the state, the recent matches, coach-marked moments), asked once.
          const more = button('Details', 'quiet cr-prog-more', () => this.toggleProgress(squad, k.skill, holder, more), 'Why this state, match by match');
          more.dataset.progress = k.skill; more.setAttribute('aria-expanded', 'false');
          name.append(more);
        }
        row.append(name, el('span', `cr-state s-${String(k.state).toLowerCase().replace(/ /g, '-')}`, MASTERY_UI[k.state] || String(k.state)),
                   el('span', null, String(k.opportunities)), el('span', null, String(k.matches)), el('span', null, TREND_UI[k.trend] || '–'));
        table.append(row, holder);
      }
      s.append(table);
      if (pr.note) s.append(el('p', 'cr-intro', pr.note));
      s.append(el('p', 'cr-note', 'States follow PitchIQ’s own rules (proposals, not validated: review them with a coach). They count the moments where each skill was needed, so a short clip and a full match are compared fairly.'));
    } else {
      s.append(el('p', 'cr-none', this.mode === 'v2' ? 'Link this team to a squad in the coaching setup to follow its skills across matches.' : 'Progress across matches needs the new coaching engine on the server.'));
    }
    const n = r.nextObjective;
    if (n && typeof n.text === 'string' && n.text) {
      const box = el('div', 'cr-next'), f = n.finding ? this.findingById(n.finding) : null, focus = n.fromMatch ? null : splitFocus(n.text);
      box.append(el('div', 'cr-sub-h', 'Next development objective'));
      if (focus) {
        // the level's suggested focus (polish round): its label, the focus, its sources by title and id (as the PDF)
        box.dataset.origin = 'level';
        box.append(el('p', 'cr-intro', focus.label + '.'), this.para(null, focus.body));
        if (focus.ids.length) box.append(el('p', 'cr-note', `Source: ${focus.ids.map(id => `${this.sourceTitle(id)} (${id})`).join('; ')}.`));
      } else box.append(this.para(null, n.text));
      // The report's text names where the objective comes from (TEXT wave 5: "From this match: ..." or "From the
      // level's objectives, not from this match: ..."); the page adds the link to the finding, or the plain origin
      // only when the text does not say it.
      // Fix round 3: the plain origin only for a skill not from this match whose text names no origin itself; an
      // objective chosen from this match's checks (critic S-3, COACH F-4 / FACTS F-9), or from a work item whose
      // finding is not shown, never reads "not from this match" (the PDF's pdf.origin_line).
      if (n.fromMatch && f) { const p = el('div', 'cr-links'); p.append(el('span', 'cr-intro', 'The finding: '), this.findingLink(f)); box.append(p); }
      else if (!n.fromMatch && n.skill !== null && n.skill !== undefined && !ORIGIN_NAMED.test(n.text)) box.append(el('p', 'cr-intro', 'From the level’s objectives, not from this match.'));
      s.append(box);
    }
    return s;
  }

  progressDetail(squad) {
    // GET /api/coaching/progress?squad= (PROGRESS's mastery() for the squad under the run's settings), once per squad
    // and report version; a failure is asked again next time.
    const key = `${squad}|${this.gen}`;
    if (!this.progressAsk.has(key)) {
      const ask = Promise.resolve().then(() => this.api('/api/coaching/progress?' + new URLSearchParams({squad, run: this.runName() || '', team: this.teamKey()}).toString()));
      ask.catch(() => this.progressAsk.delete(key));
      this.progressAsk.set(key, ask);
    }
    return this.progressAsk.get(key);
  }
  async toggleProgress(squad, skill, holder, btn) {
    const open = holder.hidden;
    holder.hidden = !open; btn.textContent = open ? 'Hide' : 'Details'; btn.setAttribute('aria-expanded', String(open));
    if (!open || holder.dataset.filled) return;
    holder.replaceChildren(el('p', 'cr-none', 'Reading the squad\u2019s matches\u2026'));
    let detail = null;
    try { detail = await this.progressDetail(squad); } catch (error) {
      holder.replaceChildren(el('p', 'cr-none', `The details could not be read (${error.message}).`)); return;
    }
    // SERVER answers {squad, progress: mastery(), entries, rederived}; a bare mastery() record is read the same way.
    const m = detail && detail.progress && typeof detail.progress === 'object' ? detail.progress : detail;
    const d = m && m.skills && typeof m.skills === 'object' ? m.skills[skill] : null;
    if (!d || typeof d !== 'object') { holder.replaceChildren(el('p', 'cr-none', 'No detail for this skill yet.')); return; }
    holder.dataset.filled = '1';
    holder.replaceChildren(...this.masteryNodes(d));
  }
  masteryNodes(d) {
    // The detail of one skill: why its state, the judged moments, the matches (most recent last), the coach-marked
    // share; shares only where the level's register shows them (DESIGN 6.3).
    const n = this.numbers(), out = [], shares = n === 'shares' || n === 'rates';
    const why = MASTERY_WHY[d.reason];
    out.push(this.para('Why this state.', `${MASTERY_UI[d.state] || d.state}${why ? `: ${why}` : ''}.`));
    if (d.opportunities === 0) out.push(this.para('Moments.', 'No moment of this skill has been judged in the squad’s matches yet.'));
    else if (Number.isInteger(d.opportunities)) {
      let text = `Shown in ${d.demonstrated} of ${d.opportunities} judged ${plural(d.opportunities, 'moment')} over ${d.matches} ${d.matches === 1 ? 'match' : 'matches'}`;
      if (Number.isInteger(d.unclear) && d.unclear) text += `; ${d.unclear} more could not be judged clearly`;
      if (shares && isNum(d.lower) && isNum(d.upper) && d.opportunities) text += ` (likely between ${pct(d.lower)} and ${pct(d.upper)} of such moments)`;
      out.push(this.para('Moments.', text + '.'));
    }
    if (isNum(d.coachMarkedShare) && d.coachMarkedShare > 0) out.push(el('p', 'cr-intro', `${pct(d.coachMarkedShare)} of these moments use possession marked by the coach.`));
    const hist = (Array.isArray(d.history) ? d.history : []).filter(h => h && typeof h === 'object');
    if (hist.length) {
      const list = el('div', 'cr-prog-hist');
      for (const h of hist) {
        const row = el('div', 'cr-prog-hrow' + (h.inWindow === false ? ' old' : ''));
        row.append(el('span', null, typeof h.matchDate === 'string' ? h.matchDate : '–'),
                   el('span', null, h.opportunities ? `${h.demonstrated} of ${h.opportunities} shown` : 'no judged moment'),
                   el('span', null, isNum(h.seconds) ? seconds(h.seconds) : ''),
                   el('span', 'cr-note', h.qualifies ? (h.inWindow === false ? 'older than the matches read' : 'counted') : 'too few moments to count'));
        list.append(row);
      }
      out.push(el('div', 'cr-sub-h', 'Match by match'), list);
    }
    return out;
  }

  feedbackSectionV2(r) {
    // Reviews kept with this video that sit on no finding of this report: this run's on findings it does not have,
    // earlier runs' (team letters are arbitrary per run), and the previous engine's (DESIGN 10).
    const all = ((this.answer && this.answer.feedback) || []).filter(f => f && typeof f === 'object' && f.verdict);
    const previous = all.filter(f => !f.findingKey);
    const loose = all.filter(f => f.findingKey && f.thisRun && !f.attached), earlier = all.filter(f => f.findingKey && !f.thisRun);
    if (!previous.length && !loose.length && !earlier.length) return null;
    const s = this.section('feedback', 'Coach feedback', 'Reviews kept with this video that are not on a finding of this report. They never change the measurements.');
    const line = f => `${(f.finding && f.finding.headline) || (f.observation && f.observation.headline) || 'A finding'} · ${verdictName(f.verdict)}${f.text ? ` · “${f.text}”` : ''}`;
    const plain = items => { const ul = el('ul', 'cr-list'); for (const f of items) ul.append(el('li', null, line(f) + (f.run && !f.thisRun ? ` (run ${f.run}; players named by that run’s ids)` : ''))); return ul; };
    if (loose.length) s.append(el('h4', 'cr-sub-h', 'Your reviews of this run that match no finding here'), plain(loose));
    if (earlier.length) s.append(el('h4', 'cr-sub-h', 'From an earlier run'), plain(earlier));
    if (previous.length) s.append(el('h4', 'cr-sub-h', 'Feedback from the previous report engine'), plain(previous));
    return s;
  }

  // ---------- the selected key moment: side column, jump, replay ----------
  selectMoment(id) {
    const ev = this.momentOf(id);
    if (!ev) return;
    this.selectedMoment = id;
    if (ev.finding) this.selected = ev.finding;
    if (this.adjusting && this.adjusting.evidence !== id) { this.adjusting = null; this.adjustMessage = ''; }
    this.markSelection();
    this.renderSide();
  }
  showMoment(ev, how) {
    this.selectMoment(ev.id);
    return this.jumpV2(ev, how);
  }
  jumpV2(ev, how) {
    // DESIGN 9.3: the moment's view (the video at tier world or pinned, else the 2D match), paused at the frame's
    // centre with the primary player selected, the annotation drawn within its window and the caption shown; Replay
    // starts at the window's start and plays through the page's own transport.
    const fps = isNum(ev.fps) && ev.fps > 0 ? ev.fps : this.fps(), seek = isNum(ev.seek) ? ev.seek : Number.isInteger(ev.frame) ? (ev.frame + .5) / fps : ev.time;
    const view = how === 'board' || ev.tier === 'image' ? 'board' : 'tracking', f = this.findingById(ev.finding), seen = (ev.players || []).filter(p => typeof p === 'string');
    const primary = f && f.players && Array.isArray(f.players.primary) ? f.players.primary.filter(id => seen.includes(id)) : [];
    const a = ev.annotation && ev.annotation.window && isNum(ev.annotation.window.t0) && isNum(ev.annotation.window.t1) ? ev.annotation : null;
    // DESIGN 5.4: a suggested area the video's tier may not carry is on the 2D diagram, and the caption says so.
    const elsewhere = view === 'tracking' && ev.tier !== 'world' && this.zoneOf(ev) ? ' Suggested area shown on the 2D diagram.' : '';
    return this.jumpTo({frame: ev.frame, seek, time: ev.time, view, focus: primary.length ? primary.concat(seen.filter(id => !primary.includes(id))) : seen,
                        diagram: ev.diagram || null, overlay: validOverlay(ev.videoOverlay) ? ev.videoOverlay : null, evidence: ev.id, finding: ev.finding || null,
                        tier: ev.tier, caption: plainLabels((a && a.caption) || ev.caption || '', this.labels) + elsewhere, window: a ? {t0: a.window.t0, t1: a.window.t1} : null,
                        follow: followOf(ev), replay: how === 'replay' && !!a, explanation: f && f.text && typeof f.text.explanation === 'string' ? plainLabels(f.text.explanation, this.labels) : null});   // ids shown as the result's display labels
  }
  zoneOf(ev) {
    const els = ev && ev.diagram && Array.isArray(ev.diagram.elements) ? ev.diagram.elements : [];
    return els.find(e => e && e.type === 'suggestedZone' && Array.isArray(e.polygon) && e.polygon.length >= 3) || null;
  }
  savedAreas(evId) {
    // The coach's saved areas for a moment: feedback entries of this run carrying an adjustment (SERVER: one per finding,
    // moment, range and team; id adj-...).
    return ((this.answer && this.answer.feedback) || []).filter(f => f && f.thisRun && f.adjustment && f.adjustment.evidence === evId && Array.isArray(f.adjustment.polygon));
  }
  async removeArea(entry) {
    this.adjustMessage = 'Removing…'; this.renderSide();
    try {
      await this.post('/api/coaching/feedback/remove', {id: entry.id});
    } catch (error) {
      this.adjustMessage = `The area could not be removed: ${error.message}`; this.renderSide();
      return;
    }
    this.adjustMessage = 'Your area was removed.';
    this.cache.clear(); this.gen++;
    await this.request();
  }
  adjustmentsFor(evId) {
    // The coach's own areas for a moment (feedback entries of this run carrying an adjustment) and the one being edited.
    const out = [];
    if (!evId) return out;
    for (const f of this.savedAreas(evId)) out.push({polygon: f.adjustment.polygon, label: 'coach’s area'});
    if (this.adjusting && this.adjusting.evidence === evId) out.push({polygon: this.adjusting.polygon, label: 'your adjusted area'});
    return out;
  }
  alternativeText(alt) {
    const n = this.numbers(), reach = alt.reach || {}, imp = alt.improves || {};
    const base = `The dotted area is one possibility for ${alt.mover}, with everyone else kept where they were: an illustration, not an instruction.`;
    if (n === 'none' || n === 'counts' || !isNum(reach.horizonS) || !isNum(reach.speedMS)) return base;
    return `${base} It is within reach of where ${alt.mover} stood ${reach.horizonS.toFixed(1)} s earlier at up to ${reach.speedMS.toFixed(1)} m/s${isNum(imp.shareOfZone) ? `, and the ${words(imp.relation || 'relation').toLowerCase()} holds in ${pct(imp.shareOfZone)} of it` : ''}.`;
  }

  renderSideV2() {
    const side = this.side, r = this.report(), ev = this.selectedMoment ? this.momentOf(this.selectedMoment) : null;
    side.replaceChildren(el('h3', null, 'Key moment'));
    side.hidden = !r;
    const f = ev && ev.finding ? this.findingById(ev.finding) : this.selected ? this.findingById(this.selected) : null;
    if (!ev) {
      side.append(el('p', 'an-empty', f ? 'This finding has no key moment of its own: it rests on the counts listed with it.' : 'Select a key moment to see its frame, its annotation and the diagram.'));
      this.placeSide();
      return;
    }
    const colours = this.colours(), pal = annotationPalette(colours.A, colours.B), video = (this.info && this.info.video) || {};
    if (f) {
      const [badge, kind] = verdictUi(f), head = el('div', `cr-side-head k-${kind}`), h = el('div', 'cr-side-title');
      h.append(...labelNodes(f.text ? f.text.headline : this.skillName(f.observation), this.labels));
      head.append(el('span', 'cr-badge', badge), h);
      side.append(head);
    }
    const fig = el('figure', 'cr-ev selected'), times = el('div', 'cr-times');
    fig.dataset.evidence = ev.id;
    const show = button(clock(ev.time), 'cr-time', () => this.showMoment(ev, 'show'), `Show this frame with its annotation (segment ${clock(ev.segmentTime)})`);
    show.dataset.evidence = ev.id;
    const replay = button('Replay', 'quiet', () => this.showMoment(ev, 'replay'), 'Play this moment from three seconds before, with its annotation');
    replay.dataset.replay = ev.id; replay.disabled = !(ev.annotation && ev.annotation.window);
    const board = button('Show on 2D', 'quiet cr-board', () => this.showMoment(ev, 'board'), 'Show this moment on the 2D match view');
    board.dataset.evidence = ev.id;
    times.append(show, replay, board, el('span', 'cr-seg', `segment ${clock(ev.segmentTime)}`));
    fig.append(times);
    if (Number.isInteger(ev.frame)) {
      const shot = el('div', 'cr-shot'), img = el('img');
      if (video.width && video.height) shot.style.aspectRatio = `${video.width} / ${video.height}`;
      img.alt = `Frame ${ev.frame} at ${clock(ev.time)}`; img.loading = 'lazy'; img.decoding = 'async';
      img.onerror = () => shot.replaceChildren(el('p', 'cr-noshot', 'The screenshot could not be loaded.'));
      img.src = '/api/coaching/frame?' + new URLSearchParams({run: this.runName() || '', frame: String(ev.frame), w: '640'}).toString();
      shot.append(img);
      if (validOverlay(ev.videoOverlay)) shot.append(svgOverlay(ev.videoOverlay, colours, this.labels, {palette: pal}));
      fig.append(shot);
    }
    const tier = TIER_UI[ev.tier];
    if (tier) fig.append(el('p', 'cr-note', `${tier[0]}: ${tier[1]}`));
    const zone = this.zoneOf(ev);
    if (zone && ev.tier !== 'world') fig.append(el('p', 'cr-note cr-zone-note', 'Suggested area shown on the 2D diagram.'));
    const cap = ev.annotation && ev.annotation.caption;
    if (cap) fig.append(this.para(null, cap, 'cr-caption'));
    if (ev.caption && ev.caption !== cap) fig.append(this.para(null, ev.caption, 'cr-caption'));
    if (ev.diagram && Array.isArray(ev.diagram.elements)) {
      if (ev.diagram.title) fig.append(this.para(null, ev.diagram.title, 'cr-dtitle'));
      fig.append(this.adjustDiagram(ev, colours, pal));
      if (ev.diagram.metresEstimated !== false) fig.append(el('p', 'cr-note', 'Top-down from the observed positions at this frame; distances are estimates.'));
    }
    if (f && f.text && f.text.explanation) fig.append(this.para(null, f.text.explanation));
    if (f && f.text && f.text.action) fig.append(this.para(f.verdict === 'strength' ? 'To build on it.' : f.verdict === 'try_this' ? null : isMoments(f) ? 'What to look for.' : 'What could help.', actionText(f)));
    const alt = this.alternativeOf(ev.alternative), none = alt || isMoments(f) ? null : this.noAlternativeOf(ev);
    if (alt) fig.append(this.para(null, this.alternativeText(alt), 'cr-intro'));
    else if (none) fig.append(this.para(null, `No suggested area: ${ALT_REASON_UI[none.reason] || words(none.reason).toLowerCase()}. The principle to review is given instead.`, 'cr-intro'));
    if (zone && this.mode === 'v2') fig.append(this.adjustControls(ev));
    if (Array.isArray(ev.players) && ev.players.length) fig.append(this.kv('Players', ev.players.join(', ')));
    if (ev.confidence) fig.append(this.kv('Confidence', String(ev.confidence), ev.confidence === 'low'));
    const calib = ev.calibration || {};
    if (isNum(calib.confidence)) fig.append(this.kv('Registration confidence', calib.confidence.toFixed(2), calib.confidence < .7));
    side.append(fig);
    this.placeSide();
  }

  // ---------- adjusting a suggested area (DESIGN 10): kept beside PitchIQ's, never changes the measurements ----------
  adjustDiagram(ev, colours, pal) {
    // The moment's diagram; while the coach adjusts its area, dragging on it moves the area (pointer events on the
    // wrapper, so the capture survives the diagram being redrawn under the pointer).
    const wrap = el('div', 'cr-adjust-wrap');
    const draw = () => wrap.replaceChildren(svgDiagram(ev.diagram, colours, this.labels, {palette: pal, adjustments: this.adjustmentsFor(ev.id)}));
    draw();
    let drag = null;
    wrap.addEventListener('pointerdown', e => {
      const svg = wrap.firstElementChild || (wrap.children && wrap.children[0]), m = svg && typeof svg.getScreenCTM === 'function' ? svg.getScreenCTM() : null;
      if (!this.adjusting || this.adjusting.evidence !== ev.id || !m || !m.a) return;
      drag = {x: e.clientX, y: e.clientY, k: m.a, poly: this.adjusting.polygon.map(p => p.slice())};
      try { wrap.setPointerCapture(e.pointerId); } catch (error) { /* capture unavailable: the drag still follows inside */ }
      e.preventDefault();
    });
    wrap.addEventListener('pointermove', e => {
      if (!drag || !this.adjusting) return;
      const dx = (e.clientX - drag.x) / drag.k / BOARD.length, dy = (e.clientY - drag.y) / drag.k / BOARD.width;
      this.adjusting.polygon = drag.poly.map(([x, y]) => [clampN(x + dx), clampN(y + dy)]);
      draw();
    });
    for (const type of ['pointerup', 'pointercancel']) wrap.addEventListener(type, () => { drag = null; });
    if (this.adjusting && this.adjusting.evidence === ev.id) wrap.classList.add('adjusting');
    return wrap;
  }
  adjustControls(ev) {
    const box = el('div', 'cr-adjust'), a = this.adjusting && this.adjusting.evidence === ev.id ? this.adjusting : null;
    if (!a) {
      const b = button('Adjust the area', 'quiet', () => this.startAdjust(ev), 'Move or resize the suggested area to where you would want it; PitchIQ’s area and the measurements stay as they are');
      b.dataset.adjust = 'start';
      box.append(b);
      for (const saved of this.savedAreas(ev.id)) {
        const rm = button('Remove my area', 'quiet', () => this.removeArea(saved), 'Remove the area you saved for this moment');
        rm.dataset.adjust = 'remove';
        box.append(el('span', 'cr-note', `Your area saved ${String(saved.made || '').replace('T', ' ')}${saved.text ? ` (${saved.text})` : ''}.`), rm);
      }
      if (this.adjustMessage) box.append(el('p', 'msg', this.adjustMessage));
      return box;
    }
    box.append(el('p', 'cr-intro', 'Drag the area on the diagram, or move it a metre at a time. Your area is kept with this video beside PitchIQ’s and never changes the measurements.'));
    const row = el('div', 'cr-adjust-row');
    for (const [name, text, dx, dy, label] of [['left', '←', -1, 0, 'Move 1 m left'], ['right', '→', 1, 0, 'Move 1 m right'], ['up', '↑', 0, -1, 'Move 1 m up'], ['down', '↓', 0, 1, 'Move 1 m down']]) {
      const b = button(text, 'quiet', () => this.moveAdjust(dx, dy), label);
      b.setAttribute('aria-label', label); b.dataset.adjust = name;
      row.append(b);
    }
    for (const [name, text, k] of [['smaller', 'Smaller', .85], ['larger', 'Larger', 1 / .85]]) { const b = button(text, 'quiet', () => this.scaleAdjust(k)); b.dataset.adjust = name; row.append(b); }
    const reset = button('Reset', 'quiet', () => { a.polygon = a.base.map(p => p.slice()); this.renderSide(); });
    reset.dataset.adjust = 'reset';
    row.append(reset);
    const note = el('textarea', 'cr-note-input');
    note.value = a.note || ''; note.maxLength = 2000; note.rows = 2; note.placeholder = 'Optional note: why this area';
    note.setAttribute('aria-label', 'Note on the adjusted area');
    note.addEventListener('input', () => { a.note = note.value; });
    const actions = el('div', 'cr-adjust-row');
    const save = button('Save my area', 'primary', () => this.saveAdjust()), cancel = button('Cancel', 'quiet', () => { this.adjusting = null; this.adjustMessage = ''; this.renderSide(); });
    save.dataset.adjust = 'save'; cancel.dataset.adjust = 'cancel';
    actions.append(save, cancel);
    box.append(row, note, actions);
    if (this.adjustMessage) box.append(el('p', 'msg', this.adjustMessage));
    return box;
  }
  startAdjust(ev) {
    const z = this.zoneOf(ev);
    if (!z) return;
    const poly = z.polygon.filter(p => Array.isArray(p) && isNum(p[0]) && isNum(p[1])).map(p => [p[0], p[1]]);
    this.adjusting = {evidence: ev.id, finding: ev.finding, alternative: z.alternative || null, mover: z.mover || null, base: poly.map(p => p.slice()), polygon: poly, note: ''};
    this.adjustMessage = '';
    this.renderSide();
  }
  pitchOf(evId) { const ev = this.momentOf(evId), p = ev && ev.diagram && ev.diagram.pitch; return p && isNum(p.length) && isNum(p.width) ? p : {length: BOARD.length, width: BOARD.width}; }
  moveAdjust(dxM, dyM) {
    const a = this.adjusting;
    if (!a) return;
    const p = this.pitchOf(a.evidence);
    a.polygon = a.polygon.map(([x, y]) => [clampN(x + dxM / p.length), clampN(y + dyM / p.width)]);
    this.renderSide();
  }
  scaleAdjust(k) {
    const a = this.adjusting;
    if (!a) return;
    const [cx, cy] = centroid(a.polygon);
    a.polygon = a.polygon.map(([x, y]) => [clampN(cx + (x - cx) * k), clampN(cy + (y - cy) * k)]);
    this.renderSide();
  }
  async saveAdjust() {
    const a = this.adjusting, r = this.report(), f = a ? this.findingById(a.finding) : null;
    if (!a || !isV2(r) || !f) return;
    const range = r.range || {};
    this.adjustMessage = 'Saving…'; this.renderSide();
    try {
      await this.post('/api/coaching/feedback', {run: this.runName(), team: r.team, findingKey: f.key, range: {t0: range.t0, t1: range.t1},
                                                 adjustment: {evidence: a.evidence, polygon: a.polygon, mover: a.mover, note: asciiNote(a.note) || null}});
    } catch (error) {
      this.adjustMessage = `The area could not be saved: ${error.message}`; this.renderSide();
      return;
    }
    this.adjusting = null; this.adjustMessage = 'Saved: your area is kept with this video beside PitchIQ’s.';
    this.cache.clear(); this.gen++;
    await this.request();
  }

  placeSide() {
    // Wide screens: the evidence column beside the report, moved down to the selected finding. Narrow screens (one
    // column): the evidence goes right under the selected finding.
    const narrow = !!(this.media && this.media.matches), card = this.selected ? this.cards.get(this.selected) : null;
    if (narrow && card && typeof card.after === 'function') {
      card.after(this.side);
      this.side.classList.add('inline'); this.side.style.marginTop = '';
      return;
    }
    if (this.home && this.side.parentElement !== this.home) this.home.append(this.side);
    this.side.classList.remove('inline');
    const top = card && isNum(card.offsetTop) && isNum(this.main.offsetTop) ? Math.max(0, card.offsetTop - this.main.offsetTop) : 0;
    this.side.style.marginTop = top ? `${Math.round(top)}px` : '';
  }
}

global.Coaching = {create: (options) => new CoachingView(options), drawBoard, drawVideo, drawFollow, followOf, svgDiagram, svgOverlay, labelMap, plainLabels, clock,
                   annotationPalette, ciede2000, labOf, contrastRatio, TIER_ELEMENTS, ELEMENT_BUDGET, CATALOGUE, formatDefault, typicalLevel, relativeSlots, cleanBlock,
                   cleanSpans, placeSpan, spanRecord, MEASURE_UI};
})(window);
