"""Evidence for coaching: the moments a coach should watch, each with its exact source frame, a top-down diagram of
the observed positions there and, where the frame's registration supports it, a video overlay. Two engines live here
while the v2 work lands (DESIGN v2 12.2; the v1 callers move in wave 7): v1 below, and v2 (moments,
moments_from_obs, annotate; DESIGN v2 5.4-5.6) after it. Both use the same overlay tiers (Frames.tier, paint_check)
and the same time convention.

v1 (attach, select, build: the evidence of the v1 observations, CONTRACT section 6; unchanged). Drawn by
soccer/coaching/render.py (PDF, PNG) and coaching.js.

Which moments. Per observation at most MAX_EVIDENCE occurrences (detectors.py), at least MIN_SEPARATION_SECONDS
apart so they show different situations, ranked by the overlay tier the frame allows, the share of the team observed
there, the registration confidence (in steps of 0.1: the error changes slowly, between samples 0.55 s apart by a
median 0.6-1.4 px), whether the stabilizer flagged an involved player's box within +-0.5 s (a flagged box foot is
15-32 px off against 2-3 px normally), and the occurrence's peak. Within an occurrence the frame is its peak sample
unless a frame of a better tier (or the same tier without a flagged box) exists inside it. A strength or a neutral
measurement has no occurrence; up to TYPICAL_EVIDENCE typical moments (the sample nearest the median in a stretch,
only among the samples inside the measurement's interquartile range: broadcast 0-60 s team A's area showed a 600 m2
"typical" moment against a 490 m2 median; within that, moments within TYPICAL_BAND of the IQR from the median first,
then by the same ranking) show what was maintained. A formation moment is taken where the team's whole lineup is in
view (the amateur clip's team B drew a back three under a "4-1-4-1" title because B-12 was not in view on that frame);
an occurrence without such a sample keeps its frame but gets no diagram. The players an evidence record names are only
those observed on its frame (they are the page's focus). An insufficient observation has no evidence.

Time. Every record carries the absolute source frame index (frames[i].frame), its time (frames[i].time, the video's
currentTime), the segment time (time - frames[0].time) and the browser seek time (frame + 0.5) / fps: seeking to
the exact frame / fps shows the previous frame on about one frame in three in Chromium, (n + 0.5) / fps was right
29 of 29 times. The server grabs the frame by index (cv2 CAP_PROP_POS_FRAMES), identical to the pipeline's decode.

Overlay tiers (audit of the registration against hand-marked truth, broadcast and amateur):
  world   calibration confidence >= .7, the registration chains within WORLD_MAX_DISAGREEMENT (3 m) or not measured
          (the contract's rule for every distance claim), and every paint tick within +-PAINT_WINDOW_SECONDS has a
          wide residual of at most PAINT_MAX_RESIDUAL_PX (at 1920 px; scaled with the frame width) over at least
          PAINT_MIN_SEGMENTS matched segments: broadcast median 4.1-5.5 px, worst landmark p90 9.5-11.6 px, 0% over
          20 px, 0.3-0.4 m. Confidence alone is not enough: frames at .71-.85 were 130-240 px off for 5 s on broadcast
          42-222 s, and every one had a wide residual of 35-47 px. Never on small-scene runs, keyless (provisional)
          registrations or footage with a measured lens bend (Veo frames rated MEDIUM are 50-60 px off; a single
          homography cannot fit that lens).
          Pitch geometry may be drawn (zones, measures between unit centres).
  pinned  calibrated and confidence >= .35: shapes whose corners are the current frame's foot pixels (box bottom
          centre), exact at the feet and 3-4 px off within 10 m on broadcast, 1.5 px on amateur, 4.7 px on Veo. Unit
          polylines and lines between players; no pitch-space zones.
  image   otherwise: boxes and labels only, no metre value on the video.
Top-down diagrams are always available (but for a formation moment without the full lineup). Positions there are the
measurement's own observed positions (series positions: analytics.Samples smoothed over 0.14 s); an estimate is never
drawn as a fact, goalkeepers are left out (their identities are not in the series), and arrows follow observed samples
only, cut at any gap over ARROW_GAP_SECONDS. A team width or length is drawn as a band over the extent with a bracket
(a measure with end ticks) between the two extreme players, outside the team.

v2 (adaptive coaching). The detectors (contextual.py) emit profile-independent opportunities; this module gives each
one evidence MOMENT in the same cache tier (obs-*.json / roles-*.json), and at interpretation time ANNOTATES the
moments the findings cite.

  Which sample (key_run, _choose). One moment per opportunity, at a sample of its DOMINANT outcome under
  contextual.REFERENCE_PARAMS (DESIGN Appendix A, the level-4 values): the longest run of failing samples when they are
  at least as many as demonstrated ones, else the longest demonstrated run. contextual.key_sample prefers any failing
  run, so a demonstrated opportunity with a short failing stretch showed that stretch under a strength. Measured on
  EVAL's 30 synthetic scene variants x levels 1-7, counting only the opportunities whose own outcome under the level
  matches the finding: the moment's sample shows that outcome for 4035 of 4060 demonstrated opportunities cited by
  strengths (key_sample: 3767) and for all 1468 not-demonstrated ones cited by criticisms and reviews (key_sample:
  1468); of the moments the findings actually cite, 322 of 322 and 778 of 786.
  Inside the run: where the primary's identity is tainted when it is tainted anywhere in the run (a clean frame must
  not hide a crossing from assess.py's identity guardrail, which reads the moment's sample: counterexample "identity
  crossing"), then without a stabilizer flag on an involved player within +-FLAG_WINDOW_S, then the best overlay tier
  (at most MAX_RANKED candidates checked), then nearest the run's middle. The moment never depends on the coaching
  profile (invariant 2); assess reads its 'sample' for the finding's measurements.
  What a moment holds (schema 'moment'): t, the exact grid frame, seek = (frame + 0.5) / fps, the tier and its
  calibration facts, the involved players observed there, stabilizer flags, the SCENE (every person passing the
  position gate: kind O, registration, sigma <= 3 m; the single keeper of each team; the reliable ball; the frame's
  normalized boxes of our players and of the involved opponents, the only people an overlay outlines) and 'before':
  our own players REACH_H = 2.0 s earlier (the movers of a suggested zone are team-mates, DESIGN 5.1 step 5). The
  calibration map also carries 'side' (the goal the team defends there: attack frame <-> pitch), the result's pitch
  size and, on world-tier frames only, the oriented homography h0..h8 (pitch metres -> source pixels): the
  interpretation tier never reads the result. A moment is about 4 KB of compact JSON (scene 2.4, before 1.1).
  Role tier (moments_from_obs). D3 and small-format D5 opportunities are made from obs['roleInputs']['detect'], so
  their scenes are rebuilt from it (own outfield players, the opponents stored there, the ball) and their tier comes
  from obs['frameTiers']: the tier runs of exactly the samples the role tier stores, computed in the measurement tier
  (broadcast 42-222 s team A: 26 runs, 2.6 KB). Boxes and homographies come from the pixel anchors role_anchors()
  keeps every ANCHOR_STRIDE-th stored sample (about 0.47 KB each; Metrica game 1, 90 minutes: 1103-1263 anchors,
  0.46-0.65 MB per team) once the orchestrator stores them as obs['roleInputs']['evidence']; without them a role-tier
  moment has no video overlay (its diagram is complete).

  Annotation (annotate, evidence_of; DESIGN 5.4-5.6). Per cited moment: the reviewed player's outline ('focus', tone
  review or strength; labelled only where the level names players in evidence: naming evidence_only / yes, identity
  guardrail passed, not a team-level-only check), up to three relevant team-mates or opponents ('support'), up to three
  relation lines (pressure, cover, gap, behind the ball; dashed where the relation was missing under the level's own
  parameters; a value only where the level's register allows numbers), the channel or gap area, the validated
  suggested zone (only on the moment ZONES chose) with one movement arrow, and a short caption. Elements per detector
  follow DESIGN 5.2; D6 has no zone (no lane model) and draws the team's width or depth.
  Projection by tier (S.TIER_ELEMENTS). Focus and support stand on the box feet (the box itself outlined at image
  tier); relations join feet at world and pinned; channel, zone and arrow go on the video only at world, through the
  frame's paint-checked homography, and only when the disagreement rule holds: the involved players' pitch positions,
  projected, land within PROJECT_MAX_PX (20 px at 1920) of their box feet. Measured on every world-tier frame of
  broadcast 0-60 s and 42-222 s (267 frames, 5241 player feet): per-frame median 0.7 px, p99 4.2 px, worst 4.7 px, so
  the rule only catches a broken registration. The area also needs every corner in front of the camera and its centre
  in the picture. Otherwise the overlay is downgraded and the area lives on the 2D diagram, which always carries every
  element; the evidence caption then says so (spec Part 9: no falsely precise area over the footage; the link still
  seeks the exact frame).
  Lifetime: annotation.window = key time - 3 s .. + 2 s (clamped to the segment), keyFrame = the moment's frame.
  Captions: annotate runs before explain, so a caption is the level's own name of the behaviour (profiles.json skills)
  and what the moment is for, with no number, player or relation claim; apply_captions() swaps in the finding's
  validated text.short once explain has run.
  Measured (CPU, Oct 22). EVAL's 30 scene variants through the real tiers: obs, roles and every evidence record
  schema-valid at levels 1-7, and the verdicts, guardrails and limitations identical to DETECT's runs with stand-in
  moments (0 of 210 scene-levels differ). ZONES' propose through annotate (4 scenes x 7 levels): 23 zones, each drawn
  on its own moment only, on the video for the 17 on world frames, on the diagram for all 23. The five run copies
  (engine phase, and coach spans on AM / VEO / BC180): 205 of 205 moments on their exact frame; broadcast 0-60 s and
  42-222 s: 41 world and 62 pinned moments (engine phase); 303 s and the amateur clip: pinned (keyless registration,
  small scene). The stage costs 0.05-0.6 s per team on broadcast (mostly paint checks, shared by both teams); annotate
  at most 1.3 ms per level. Metrica game 1 (90 minutes, simulated, python -I -B): 819-848 moments under full tracking
  (3.5-3.6 MB of the 10.1-10.3 MB obs per team, default JSON), 438-439 under the camera (1.7 MB), stage 0.2-0.5 s,
  0 schema problems, the strengths' cited moments showing a demonstrated sample 185 of 186.
"""
import copy
import math

import numpy as np

from .. import field, pitch_model, registration
from . import contextual as CX
from . import detectors as det
from . import episodes as ep
from . import schema as S
from . import situations as SI
from . import tactics_geom as G

MAX_EVIDENCE = 3                    # occurrences shown per observation (contract: 1-3)
TYPICAL_EVIDENCE = 2                # typical moments of a strength or neutral measurement
TYPICAL_BAND = .5                   # a typical moment's value is within this share of the interquartile range of the
                                    # median (ranked before the overlay tier)
MIN_SEPARATION_SECONDS = 10.0       # s between the evidence moments of one observation (contract)
MAX_CANDIDATES = 120                # samples of one occurrence or measurement whose tier is evaluated (evenly spread)
WORLD_MIN_CONFIDENCE = pitch_model.HIGH         # .7
WORLD_MAX_DISAGREEMENT = 3.0                    # m: chains further apart than this never draw pitch geometry (the
                                                # contract's distance rule; series.MAX_DISAGREEMENT)
PINNED_MIN_CONFIDENCE = pitch_model.MEDIUM      # .35
UNCALIBRATED = registration.MIN_POSITION        # .2: below it the frame has no positions
PAINT_WINDOW_SECONDS = .5           # s: paint ticks (every 6 frames) this close to the frame are checked
PAINT_REFERENCE_WIDTH = 1920.0      # px: the frame width the pixel thresholds were measured at
PAINT_MAX_RESIDUAL_PX = 6.0         # px at 1920: the wide paint residual of a world-tier tick (0.3% of the width)
PAINT_MIN_SEGMENTS = 3              # matched paint segments a world-tier tick needs
PAINT_REACH_PX = 150.0              # px at 1920: a detected segment further from every parallel marking is not used
PAINT_ANGLE_DEGREES = 8.0           # a predicted marking within this angle of a segment is parallel to it
BRACKET_OFFSET = .045               # a width / length bracket stands this far (normalized) outside the team ...
BRACKET_MARGIN = .02                # ... and at least this far inside the pitch edge
ARROW_SECONDS = 4.0                 # s of a player's observed trail drawn before the evidence moment
ARROW_GAP_SECONDS = 1.0             # s: an arrow never bridges a longer gap between observed samples (contract)
TIER_RANK = {'world': 2, 'pinned': 1, 'image': 0}
UNIT_VALUES = {'D': ('DX', 'DY'), 'M1': ('M1X', 'M1Y'), 'M': ('MX', 'MY'), 'F': ('FX', 'FY')}
TITLES = {'team_width': 'Team width', 'team_depth': 'Team length', 'team_area': 'Team area',
          'space_between_defence_and_midfield': 'Back line to midfield distance',
          'space_between_midfield_and_forwards': 'Midfield to front line distance',
          'uneven_back_line': 'Back-line evenness', 'high_line_with_gap': 'Line height and gap',
          'formation_stable': 'Team shape', 'formation_change': 'Team shape', 'formation_ambiguous': 'Team shape',
          'lateral_lag': 'Sideways shift of back line and midfield', 'player_isolation': 'Distance to the nearest team-mate'}
CAPTIONS = {'width': 'Team width at this moment', 'depth': 'Team length at this moment',
            'hull': 'Area covered by the team at this moment', 'gapDM': 'Distance between back line and midfield here',
            'gapMF': 'Distance between midfield and front line here',
            'lineStd': "Spread of the defenders' depth (standard deviation) here",
            'nearestTeammate': 'Distance to the nearest team-mate here'}


def _num(x):
    return isinstance(x, (int, float, np.floating, np.integer)) and not isinstance(x, bool) and math.isfinite(float(x))


def _r(x, d=4):
    return round(float(x), d)


def _metres(v, unit, estimated):
    if unit == 'm2':
        return f'{float(v):.0f} square metres'+(' (est.)' if estimated else '')
    if unit == 's':
        return f'{float(v):.1f} s'
    return f'{float(v):.1f} m'+(' (est.)' if estimated else '')


# ---------- the paint check and the tiers ----------
def paint_check(H, size, frame, width, height):
    """The final registration H of a paint tick against the paint detected on it (frames[j].pitch.markings): the
    stock assignment (field.match_segments) and the wide residual: per detected segment the distance to the nearest
    visible predicted marking parallel within PAINT_ANGLE_DEGREES and within PAINT_REACH_PX, median over the segments.
    Returns {'segments', 'matched', 'wide' (px or None), 'wideN'}."""
    p = frame.get('pitch') if isinstance(frame.get('pitch'), dict) else {}
    segs = [{'a': (m[0], m[1]), 'b': (m[2], m[3])} for m in p.get('markings') or [] if isinstance(m, (list, tuple))
            and len(m) == 4 and all(_num(v) for v in m)]
    if not segs:
        return {'segments': 0, 'matched': 0, 'wide': None, 'wideN': 0}
    matched = field.match_segments(H, size, segs, width, height)
    predicted = [i['segment'] for i in field.image_lines(H, size, width, height).values() if i['visible']]
    reach = PAINT_REACH_PX*width/PAINT_REFERENCE_WIDTH
    cos_tol = math.cos(math.radians(PAINT_ANGLE_DEGREES))
    wide = []
    for seg in segs:
        a = (seg['a'][0]*width, seg['a'][1]*height)
        b = (seg['b'][0]*width, seg['b'][1]*height)
        n = math.hypot(b[0]-a[0], b[1]-a[1])
        if n < 1e-6:
            continue
        u = ((b[0]-a[0])/n, (b[1]-a[1])/n)
        best = None
        for qa, qb in predicted:
            dx, dy = qb[0]-qa[0], qb[1]-qa[1]
            pl = math.hypot(dx, dy)
            if pl < 1e-6:
                continue
            ux, uy = dx/pl, dy/pl
            if abs(ux*u[0]+uy*u[1]) < cos_tol:
                continue
            mid = ((a[0]+b[0])/2, (a[1]+b[1])/2)
            along = (mid[0]-qa[0])*ux+(mid[1]-qa[1])*uy
            if along < -.25*pl or along > 1.25*pl:
                continue
            d = max(abs(ux*(q[1]-qa[1])-uy*(q[0]-qa[0])) for q in (a, b))
            if d <= reach and (best is None or d < best):
                best = d
        if best is not None:
            wide.append(best)
    return {'segments': len(segs), 'matched': len(matched), 'wide': float(np.median(wide)) if wide else None,
            'wideN': len(wide)}


def oriented_h(frame, width, height):
    """frames[i].calibration.H oriented (field.orient), or None."""
    cal = frame.get('calibration') if isinstance(frame.get('calibration'), dict) else {}
    h = cal.get('H')
    if not (isinstance(h, (list, tuple)) and len(h) == 9 and all(_num(v) for v in h)) or not width or not height:
        return None
    return field.orient(np.array(h, float).reshape(3, 3), width, height)


class Frames:
    """The result's frames as the evidence sees them: per grid sample its frame, the overlay tier (cached), the people
    boxes; per paint tick its check (cached)."""

    def __init__(self, cd):
        self.cd = cd
        result = cd.result or {}
        self.frames = result.get('frames') or []
        self.index = np.asarray(cd.shape.observed.frame_index) if cd.shape is not None else np.arange(len(cd.t))
        video = result.get('video') or {}
        self.width = video.get('width') if _num(video.get('width')) else None
        self.height = video.get('height') if _num(video.get('height')) else None
        self.size = field.PitchSize.from_json((result.get('tactical') or {}).get('pitchSize'))
        self.run = det.run_facts(cd)
        self.fps = float(cd.fps or 0.0)
        self.segment_start = float(self.frames[0]['time']) if self.frames else float(cd.start)
        ticks = [i for i, f in enumerate(self.frames) if isinstance(f.get('pitch'), dict) and 'markings' in f['pitch']]
        self.ticks = np.array(ticks, int)
        self.tick_times = np.array([float(self.frames[i]['time']) for i in ticks], float)
        self.world_allowed = bool(self.width and self.height and not self.run['smallScene'] and not self.run['keyless']
                                  and not self.run['lens'])
        self._paint, self._tier = {}, {}

    def frame(self, s):
        return self.frames[int(self.index[s])] if len(self.frames) else {}

    def paint(self, j):
        if j not in self._paint:
            f = self.frames[j]
            H = oriented_h(f, self.width, self.height)
            self._paint[j] = paint_check(H, self.size, f, self.width, self.height) if H is not None else None
        return self._paint[j]

    def tier(self, s):
        """(tier, calibration info) of grid sample s (module docstring)."""
        if s in self._tier:
            return self._tier[s]
        f = self.frame(s)
        cal = f.get('calibration') if isinstance(f.get('calibration'), dict) else None
        c = cal.get('confidence') if cal else None
        info = {'confidence': _r(c, 3) if _num(c) else None,
                'disagreement': _r(cal['disagreement'], 2) if cal and _num(cal.get('disagreement')) else None,
                'paintResidualPx': None, 'paintSegments': None, 'paintTicks': 0}
        calibrated = f.get('calibrated') is not False and _num(c) and c >= UNCALIBRATED
        if not calibrated or c < PINNED_MIN_CONFIDENCE:
            tier = 'image'
        else:
            tier = 'pinned'
            dis = info['disagreement']
            chains_agree = dis is None or dis <= WORLD_MAX_DISAGREEMENT
            if c >= WORLD_MIN_CONFIDENCE and self.world_allowed and chains_agree and \
                    oriented_h(f, self.width, self.height) is not None:
                t = float(f['time'])
                near = self.ticks[np.abs(self.tick_times-t) <= PAINT_WINDOW_SECONDS+1e-9] if len(self.ticks) else []
                checks = [self.paint(int(j)) for j in near]
                limit = PAINT_MAX_RESIDUAL_PX*self.width/PAINT_REFERENCE_WIDTH
                ok = bool(checks) and all(ch is not None and ch['wide'] is not None and ch['wide'] <= limit
                                          and ch['matched'] >= PAINT_MIN_SEGMENTS for ch in checks)
                walls = [ch['wide'] for ch in checks if ch and ch['wide'] is not None]
                info.update(paintResidualPx=_r(max(walls), 2) if walls else None,
                            paintSegments=min((ch['matched'] for ch in checks if ch), default=0) if checks else 0,
                            paintTicks=len(checks))
                if ok:
                    tier = 'world'
        self._tier[s] = (tier, info)
        return self._tier[s]

    def people(self, s):
        """{id: person} of the frame of sample s (people with an id and a box)."""
        out = {}
        for p in self.frame(s).get('people') or ():
            if isinstance(p.get('id'), str) and isinstance(p.get('box'), (list, tuple)) and len(p['box']) == 4:
                out.setdefault(p['id'], p)
        return out


# ---------- selection ----------
def _spread(cands, values, limit=MAX_CANDIDATES):
    if len(cands) <= limit:
        return list(cands), list(values)
    keep = np.unique(np.linspace(0, len(cands)-1, limit).round().astype(int))
    return [cands[i] for i in keep], [values[i] for i in keep]


def _coverage(series, s):
    exp = series.expected[s] if len(series.expected) else 0
    return float(series.observed[s])/exp if exp else 0.0


def _flagged(series, s, players):
    return [p for p in players if p in series.col and bool(series.values['flagged'][s, series.col[p]])]


def _choose_in(fx, series, cands, values, peak, players, prefer_target=None):
    """The evidence sample inside one occurrence (or one measurement): the peak (or the sample nearest the target)
    unless a better tier, or the same tier without a flagged box, exists."""
    cands, values = _spread(cands, values)
    if peak is not None and peak not in cands:
        cands, values = cands+[peak], values+[None]
    best, best_key = None, None
    for s, v in zip(cands, values):
        tier, info = fx.tier(s)
        if prefer_target is not None:
            close = (-abs(float(v)-float(prefer_target)) if v is not None else -math.inf,)
        elif peak is not None:
            close = (s == peak, float(v) if v is not None else -math.inf)
        else:
            close = (0.0,)
        key = (TIER_RANK[tier], not _flagged(series, s, players))+close
        if best_key is None or key > best_key:
            best, best_key = s, key
    return best


def _rank_key(fx, series, s, players, strength):
    tier, info = fx.tier(s)
    c = info.get('confidence')
    return (TIER_RANK[tier], round(_coverage(series, s), 1), round(float(c), 1) if _num(c) else 0.0,
            not _flagged(series, s, players), float(strength) if _num(strength) else 0.0)


def _pick(items, t, limit):
    """Greedy: in rank order, keep items at least MIN_SEPARATION_SECONDS from the kept ones."""
    kept = []
    for it in items:
        if all(abs(float(t[it['s']])-float(t[k['s']])) >= MIN_SEPARATION_SECONDS-1e-9 for k in kept):
            kept.append(it)
        if len(kept) >= limit:
            break
    return kept


def _complete(series, s):
    """The team's whole lineup is observed at sample s (a formation diagram needs every player in view)."""
    exp = int(series.expected[s]) if len(series.expected) else 0
    return exp > 0 and int(series.observed[s]) >= exp


def _full_lineup(series, cands, values):
    """The candidates where the whole lineup is in view, or (if none) all of them flagged as incomplete."""
    keep = [(c, v) for c, v in zip(cands, values) if _complete(series, c)]
    if keep:
        return [c for c, _ in keep], [v for _, v in keep], True
    return list(cands), list(values), False


def select(fx, series, hint):
    """[(sample, occurrence id or None, players)] for one observation's evidence hint. A formation moment is taken
    where the whole lineup is in view when the occurrence has such a sample; otherwise it gets no diagram
    ('complete' False). A typical moment lies inside the interquartile range of the measured values (TYPICAL_BAND of
    it around the median ranks first); a measurement with no sample there has no typical moment."""
    items = []
    formation = hint.get('preset') == 'formation_shape'
    for h in hint.get('occurrences') or ():
        cands, values, complete = h['candidates'], h['values'], True
        if formation:
            cands, values, complete = _full_lineup(series, cands, values)
        peak = h['peak'] if h['peak'] in cands else None
        s = _choose_in(fx, series, cands, values, peak, h['players'])
        if s is None:
            continue
        items.append({'s': s, 'occurrence': h['occurrence'], 'players': list(h['players']), 'complete': complete,
                      'key': _rank_key(fx, series, s, h['players'], h.get('strength'))})
    limit = MAX_EVIDENCE
    typ = hint.get('typical')
    if not items and typ and typ.get('candidates'):
        limit = TYPICAL_EVIDENCE
        cands, values, complete = typ['candidates'], typ['values'], True
        if formation:
            cands, values, complete = _full_lineup(series, cands, values)
        finite = [float(v) for v in values if _num(v)]
        target = typ.get('target')
        if target is not None and len(finite) >= 4:
            # A typical moment of a measured value is inside its interquartile range (broadcast 0-60 s team A's area:
            # median 490 m2, but a 600 m2 moment, beyond its own p75, was shown as typical).
            q25, q75 = np.percentile(finite, [25, 75])
            keep = [(c, v) for c, v in zip(cands, values) if _num(v) and q25-1e-9 <= float(v) <= q75+1e-9]
            cands, values = [c for c, _ in keep], [v for _, v in keep]
            band = TYPICAL_BAND*float(q75-q25)
        else:
            band = math.inf
        # One candidate per continuous stretch (the longest first), nearest the target there.
        t = fx.cd.t
        runs, cur = [], []
        for s, v in zip(cands, values):
            if cur and t[s]-t[cur[-1][0]] > 1.5*fx.cd.step:
                runs.append(cur)
                cur = []
            cur.append((s, v))
        if cur:
            runs.append(cur)
        runs.sort(key=lambda r: -len(r))
        for run in runs[:6]:
            ss = [x[0] for x in run]
            vv = [x[1] for x in run]
            goal = target
            if goal is None:
                goal = max(vv) if vv else None        # e.g. the most players observed
            players = _players_at(fx, series, hint, ss[len(ss)//2])
            s = _choose_in(fx, series, ss, vv, None, players, prefer_target=goal)
            if s is None:
                continue
            players = _players_at(fx, series, hint, s)
            v = dict(zip(ss, vv)).get(s)
            near = target is None or not _num(v) or abs(float(v)-float(target)) <= band+1e-9
            key = _rank_key(fx, series, s, players, None)[:4]+(len(run),)
            items.append({'s': s, 'occurrence': None, 'players': players, 'complete': complete, 'key': (near,)+key})
    items.sort(key=lambda it: it['key'], reverse=True)
    return _pick(items, fx.cd.t, limit)


def _players_at(fx, series, hint, s):
    """The players a typical moment involves: the hint's units at s, or the hint's player."""
    if hint.get('player'):
        return [hint['player']]
    if hint.get('preset') == 'width_depth':
        axes = ('width', 'depth') if hint.get('metric') == 'hull' else (hint.get('metric'),)
        out = []
        for ax in axes:
            for key in ('From', 'To'):
                arr = series.values.get(f'{ax}{key}')
                p = arr[s] if arr is not None else None
                if isinstance(p, str) and p and p not in out:
                    out.append(p)
        return out
    u = series.units[s] if s < len(series.units) else None
    names = hint.get('units') or ()
    if u and names:
        out = []
        for n in names:
            for p in u.get(n) or ():
                if p not in out:
                    out.append(p)
        return out
    return []


# ---------- diagram ----------
def _to_norm(X, Y, side, L, W):
    if side == 'left':
        return X/L, Y/W
    return 1-X/L, 1-Y/W


def _player_elements(series, other, s, emphasis, roles=True):
    els = []
    group = series.values['group'][s]
    for pid, (x, y, kind) in series.positions(s).items():
        el = {'type': 'player', 'id': pid, 'team': series.team, 'x': _r(x), 'y': _r(y), 'kind': kind,
              'emphasis': pid in emphasis}
        g = group[series.col[pid]] if roles else ''
        if g:
            el['role'] = str(g)
        els.append(el)
    if other is not None:
        for pid, (x, y, kind) in other.positions(s).items():
            els.append({'type': 'player', 'id': pid, 'team': other.team, 'x': _r(x), 'y': _r(y), 'kind': kind,
                        'emphasis': False})
    return els


def _unit_centre(series, s, name, L, W):
    kx, ky = UNIT_VALUES[name]
    X, Y = series.values[kx][s], series.values[ky][s]
    side = str(series.side[s])
    if not (_num(X) and _num(Y)) or side not in ('left', 'right'):
        return None
    return _to_norm(float(X), float(Y), side, L, W)


def _trail(series, pid, s, seconds=ARROW_SECONDS):
    """[[x, y, t], ...] of a player's observed positions over the last `seconds` up to sample s, the latest run
    without a gap over ARROW_GAP_SECONDS."""
    j = series.col.get(pid)
    if j is None:
        return []
    t = series.t
    lo = int(np.searchsorted(t, t[s]-seconds, side='left'))
    pts = []
    for k in range(s, lo-1, -1):
        if series.values['kind'][k, j] != 'O' or not np.isfinite(series.values['x'][k, j]):
            continue
        if pts and pts[-1][2]-float(t[k]) > ARROW_GAP_SECONDS+1e-9:
            break
        pts.append([_r(series.values['x'][k, j]), _r(series.values['y'][k, j]), _r(t[k], 3)])
    pts.reverse()
    return pts if len(pts) >= 2 else []


def _hull(points):
    pts = sorted(set(points))
    if len(pts) < 3:
        return []

    def cross(o, a, b):
        return (a[0]-o[0])*(b[1]-o[1])-(a[1]-o[1])*(b[0]-o[0])
    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1]+upper[:-1]


def diagram(fx, obs, hint, s, players):
    """The DiagramSpec (contract section 6) of an evidence moment: normalized pitch coordinates, real tracked ids and
    positions at that frame (observed only)."""
    cd = fx.cd
    team = obs['team']
    series = cd.teams[team]
    other = cd.teams['B' if team == 'A' else 'A']
    L, W = float(cd.pitch.get('length') or 105.0), float(cd.pitch.get('width') or 68.0)
    est = bool(cd.metres_estimated)
    side = str(series.side[s])
    preset = hint.get('preset')
    metric = hint.get('metric')
    f = fx.frame(s)
    spec = {'preset': preset, 'title': TITLES.get(obs.get('pattern'), 'Coaching moment'), 't': _r(f.get('time', cd.t[s]), 3),
            'frame': int(cd.frame[s]), 'team': team, 'attack': 'right' if side == 'left' else ('left' if side == 'right'
                                                                                                else None),
            'metresEstimated': est, 'pitch': {'length': L, 'width': W}, 'elements': []}
    els = _player_elements(series, other, s, set(players))
    pos = series.positions(s)
    value = series.values[metric][s] if metric in series.values and series.values[metric].ndim == 1 else None
    unit = det.UNITS_OF.get(metric, 'm')
    if preset == 'defensive_gap':
        a_name, b_name = (hint.get('units') or ('D', 'M1'))[:2]
        u = series.units[s] or {}
        for name in (a_name, b_name):
            ids = [p for p in u.get(name) or () if p in pos]
            if len(ids) >= 2:
                els.append({'type': 'unit', 'unit': name, 'ids': ids})
        ca, cb = _unit_centre(series, s, a_name, L, W), _unit_centre(series, s, b_name, L, W)
        members = [pos[p] for n in (a_name, b_name) for p in u.get(n) or () if p in pos]
        if ca and cb and members:
            ys = [m[1] for m in members]
            y0, y1 = min(ys), max(ys)
            ym = (ca[1]+cb[1])/2
            els.append({'type': 'zone', 'polygon': [[_r(ca[0]), _r(y0)], [_r(cb[0]), _r(y0)], [_r(cb[0]), _r(y1)],
                                                    [_r(ca[0]), _r(y1)]], 'style': 'gap'})
            gap = series.values['gapDM' if (a_name, b_name) == ('D', 'M1') else 'gapMF'][s]
            text = _metres(gap, 'm', est) if _num(gap) else None
            els.append({'type': 'measure', 'a': [_r(ca[0]), _r(ym)], 'b': [_r(cb[0]), _r(ym)],
                        **({'text': text} if text else {})})
    elif preset == 'width_depth':
        xs = [p[0] for p in pos.values()]
        ys = [p[1] for p in pos.values()]
        if metric in ('width', 'depth'):
            frm, to = series.values[metric+'From'][s], series.values[metric+'To'][s]
            if frm in pos and to in pos and xs:
                # A bracket between the two extreme players, drawn outside the team (beyond its most advanced player
                # for the width, beside its touchline-most player for the length) over a band the extent spans, so the
                # measurement reads at a glance instead of as a small label among the players.
                axis = 1 if metric == 'width' else 0
                lo, hi = sorted((pos[frm][axis], pos[to][axis]))
                spread = xs if axis == 1 else ys
                out = max(spread)+BRACKET_OFFSET
                if out > 1-BRACKET_MARGIN:
                    out = min(spread)-BRACKET_OFFSET
                out = min(max(out, BRACKET_MARGIN), 1-BRACKET_MARGIN)
                els.append({'type': 'band', 'axis': 'y' if axis == 1 else 'x', 'from': _r(lo), 'to': _r(hi),
                            'style': 'space'})
                a, b = ([_r(out), _r(lo)], [_r(out), _r(hi)]) if axis == 1 else ([_r(lo), _r(out)], [_r(hi), _r(out)])
                el = {'type': 'measure', 'a': a, 'b': b, 'bracket': True, 'ids': [frm, to]}
                if _num(value):
                    el['text'] = _metres(value, unit, est)
                els.append(el)
        elif metric == 'hull' and len(pos) >= 3:
            hull = _hull([(p[0], p[1]) for p in pos.values()])
            if len(hull) >= 3:
                els.append({'type': 'zone', 'polygon': [[_r(x), _r(y)] for x, y in hull], 'style': 'space'})
                if _num(value):
                    els.append({'type': 'label', 'at': [_r(np.mean(xs)), _r(np.mean(ys))],
                                'text': _metres(value, 'm2', est)})
    elif preset == 'line_movement':
        u = series.units[s] or {}
        names = hint.get('units') or ('D',)
        defenders = [p for p in (series.values['lineDefenders'][s] if metric == 'lineStd' else u.get('D') or ())
                     if p in pos]
        if len(defenders) >= 2:
            els.append({'type': 'unit', 'unit': 'D', 'ids': defenders})
        for name in names:
            if name == 'D':
                continue
            ids = [p for p in u.get(name) or () if p in pos]
            if len(ids) >= 2:
                els.append({'type': 'unit', 'unit': name, 'ids': ids})
        if metric == 'lineStd' and len(defenders) >= 2:
            dx = [pos[p][0] for p in defenders]
            lo, hi = min(dx), max(dx)
            ym = float(np.mean([pos[p][1] for p in defenders]))
            depth_range = abs(hi-lo)*L
            els.append({'type': 'measure', 'a': [_r(lo), _r(ym)], 'b': [_r(hi), _r(ym)],
                        'text': 'depth range '+_metres(depth_range, 'm', est)})
        if hint.get('arrows'):
            for p in players:
                pts = _trail(series, p, s)
                if pts:
                    els.append({'type': 'arrow', 'id': p, 'points': pts})
    elif preset == 'player_movement':
        pid = hint.get('player') or (players[0] if players else None)
        if pid in pos:
            near = series.values['nearestTeammate'][s, series.col[pid]]
            px, py, _ = pos[pid]
            best, bd = None, None
            for q, (qx, qy, _) in pos.items():
                if q == pid:
                    continue
                d = math.hypot((qx-px)*L, (qy-py)*W)
                if bd is None or d < bd:
                    best, bd = q, d
            if best is not None and _num(near) and abs(bd-float(near)) <= .5:
                els.append({'type': 'measure', 'a': [_r(px), _r(py)], 'b': [_r(pos[best][0]), _r(pos[best][1])],
                            'text': _metres(near, 'm', est)})
                for el in els:
                    if el['type'] == 'player' and el['id'] == best:
                        el['emphasis'] = True
            pts = _trail(series, pid, s)
            if pts:
                els.append({'type': 'arrow', 'id': pid, 'points': pts})
    elif preset == 'formation_shape':
        u = series.units[s] or {}
        lines = u.get('lines') or []
        for k, ids in enumerate(lines):
            ids = [p for p in ids if p in pos]
            if len(ids) >= 2:
                name = 'D' if k == 0 else ('F' if k == len(lines)-1 else 'M')
                els.append({'type': 'unit', 'unit': name, 'ids': ids})
        label = (obs.get('metrics') or {}).get('label')
        conf = (obs.get('metrics') or {}).get('labelConfidence')
        if isinstance(label, str):
            els.append({'type': 'label', 'at': [.5, .04], 'text': label+(f' ({conf})' if conf else '')})
    spec['elements'] = els
    return spec


# ---------- video overlay ----------
def _foot(box, width, height):
    x, y, w, h = (float(v) for v in box)
    return (x+w/2)*width, (y+h)*height


def video_overlay(fx, obs, hint, s, players, tier):
    """The VideoOverlaySpec of an evidence moment in source-image pixels, or None when the frame size or the
    involved players' boxes are missing. world: + pitch-space zone and measure through the frame's H; pinned: + unit
    polylines between the feet and a measure between two feet; image: boxes and labels only."""
    W, Hh = fx.width, fx.height
    if not (W and Hh):
        return None
    cd = fx.cd
    team = obs['team']
    series = cd.teams[team]
    people = fx.people(s)
    involved = [p for p in players if p in people]
    hint_units = hint.get('units') or ()
    u = series.units[s] or {}
    members = []
    if hint.get('preset') in ('defensive_gap', 'line_movement', 'formation_shape'):
        for name in hint_units:
            for p in u.get(name) or ():
                if p in people and p not in members:
                    members.append(p)
    ids = list(dict.fromkeys(involved+members))
    if not ids:
        return None
    els = []
    for pid in ids:
        p = people[pid]
        bx, by, bw, bh = (float(v) for v in p['box'])
        fxp, fyp = _foot(p['box'], W, Hh)
        el = {'type': 'player', 'id': pid, 'team': p.get('team') or team, 'x': _r(fxp, 1), 'y': _r(fyp, 1),
              'box': [_r(bx*W, 1), _r(by*Hh, 1), _r(bw*W, 1), _r(bh*Hh, 1)], 'emphasis': pid in involved}
        if isinstance(p.get('pitch'), (list, tuple)) and len(p['pitch']) == 2 and all(_num(v) for v in p['pitch']):
            el['pitch'] = [_r(p['pitch'][0]), _r(p['pitch'][1])]
        els.append(el)
    est = bool(cd.metres_estimated)
    if tier in ('pinned', 'world'):
        for name in hint_units:
            unit_ids = [p for p in (u.get(name) or ()) if p in people]
            if len(unit_ids) >= 2:
                els.append({'type': 'unit', 'unit': name, 'ids': unit_ids})
        if hint.get('preset') == 'player_movement' and involved:
            pid = involved[0]
            near = series.values['nearestTeammate'][s, series.col[pid]] if pid in series.col else None
            pos = series.positions(s)
            L, Wd = float(cd.pitch.get('length') or 105.0), float(cd.pitch.get('width') or 68.0)
            if pid in pos and _num(near):
                cands = [(math.hypot((q[0]-pos[pid][0])*L, (q[1]-pos[pid][1])*Wd), k) for k, q in pos.items()
                         if k != pid and k in people]
                if cands:
                    d, best = min(cands)
                    if abs(d-float(near)) <= .5:
                        els.append({'type': 'measure', 'a': list(_foot(people[pid]['box'], W, Hh)),
                                    'b': list(_foot(people[best]['box'], W, Hh)), 'text': _metres(near, 'm', est)})
    if tier == 'world' and hint.get('preset') == 'defensive_gap':
        H = oriented_h(fx.frame(s), W, Hh)
        a_name, b_name = (hint_units or ('D', 'M1'))[:2]
        L, Wd = float(cd.pitch.get('length') or 105.0), float(cd.pitch.get('width') or 68.0)
        ca, cb = _unit_centre(series, s, a_name, L, Wd), _unit_centre(series, s, b_name, L, Wd)
        pos = series.positions(s)
        ys = [pos[p][1] for n in (a_name, b_name) for p in u.get(n) or () if p in pos]
        if H is not None and ca and cb and ys:
            corners = np.array([[ca[0]*L, min(ys)*Wd], [cb[0]*L, min(ys)*Wd], [cb[0]*L, max(ys)*Wd],
                                [ca[0]*L, max(ys)*Wd]])
            ym = (ca[1]+cb[1])/2*Wd
            ends = np.array([[ca[0]*L, ym], [cb[0]*L, ym]])
            if (field.depth(H, corners) > 1e-9).all() and (field.depth(H, ends) > 1e-9).all():
                q = field.project(H, corners)
                e = field.project(H, ends)
                els.append({'type': 'zone', 'polygon': [[_r(x, 1), _r(y, 1)] for x, y in q], 'style': 'gap'})
                gap = series.values['gapDM' if (a_name, b_name) == ('D', 'M1') else 'gapMF'][s]
                el = {'type': 'measure', 'a': [_r(e[0][0], 1), _r(e[0][1], 1)], 'b': [_r(e[1][0], 1), _r(e[1][1], 1)]}
                if _num(gap):
                    el['text'] = _metres(gap, 'm', est)
                els.append(el)
    return {'tier': tier, 'width': int(W), 'height': int(Hh), 'elements': els}


# ---------- records ----------
def _caption(obs, hint, series, s, estimated):
    metric = hint.get('metric')
    if obs.get('detector') == 'formation_stability':
        m = obs.get('metrics') or {}
        if isinstance(m.get('label'), str):
            return f"Positions at this moment; the period reads as {m['label']}" + \
                (f" ({m['labelConfidence']})." if m.get('labelConfidence') else '.')
        return 'Positions at this moment.'
    if metric == 'lagDM':
        return 'Back line and midfield at this moment of the measured window; the arrows show their last seconds.'
    v = None
    if metric in series.values:
        arr = series.values[metric]
        if arr.ndim == 1:
            v = arr[s]
        elif hint.get('player') in series.col:
            v = arr[s, series.col[hint['player']]]
    if _num(v) and metric in CAPTIONS:
        return f'{CAPTIONS[metric]}: {_metres(v, det.UNITS_OF.get(metric, "m"), estimated)}.'
    return TITLES.get(obs.get('pattern'), 'Coaching moment')+' at this moment.'


def _evidence_confidence(obs, tier, info):
    level = (obs.get('confidence') or {}).get('overall') or 'low'
    c = info.get('confidence')
    if _num(c):
        if c < PINNED_MIN_CONFIDENCE:
            level = det._worst(level, 'low')
        elif c < WORLD_MIN_CONFIDENCE:
            level = det._worst(level, 'medium')
    return level


def build(fx, obs, hint, item):
    """One evidence record (contract section 6)."""
    cd = fx.cd
    s = int(item['s'])
    series = cd.teams[obs['team']]
    f = fx.frame(s)
    frame = int(cd.frame[s])
    t = float(f.get('time', cd.t[s]))
    first = fx.segment_start
    last = float(fx.frames[-1]['time']) if fx.frames else float(cd.end)
    t = min(max(t, first), last)
    tier, info = fx.tier(s)
    seen = series.positions(s)
    players = [p for p in item['players'] if p in seen]       # only those observed at this frame
    metrics = {}
    metric = hint.get('metric')
    if metric in series.values:
        arr = series.values[metric]
        v = arr[s] if arr.ndim == 1 else (arr[s, series.col[hint['player']]] if hint.get('player') in series.col
                                          else None)
        if _num(v):
            metrics[metric] = det._val(metric, v)
    if obs.get('pattern') == 'high_line_with_gap' and _num(series.values['lineHeight'][s]):
        metrics['lineHeight'] = det._val('lineHeight', series.values['lineHeight'][s])
    occ = next((o for o in obs.get('occurrences') or () if o['id'] == item['occurrence']), None)
    if metric == 'lagDM' and occ:
        metrics['lagDM'] = (occ.get('metrics') or {}).get('lagDM', {}).get('peak')
    fps = round(fx.fps or float(((cd.result or {}).get('video') or {}).get('fps') or 0.0), 6)
    ev = {'id': ep.stable_id('ev', obs['id'], item['occurrence'] or 'typical', frame), 'observation': obs['id'],
          'occurrence': item['occurrence'], 'kind': 'occurrence' if item['occurrence'] else 'typical',
          'time': _r(t, 3), 'frame': frame, 'segmentTime': _r(t-first, 3),
          'seek': _r((frame+.5)/fps, 4) if fps else _r(t, 3), 'fps': fps,
          'players': players, 'metrics': metrics, 'confidence': _evidence_confidence(obs, tier, info), 'tier': tier,
          'calibration': info, 'flagged': _flagged(series, s, players),
          'diagram': diagram(fx, obs, hint, s, players) if item.get('complete', True) else None,
          'videoOverlay': video_overlay(fx, obs, hint, s, players, tier),
          'caption': _caption(obs, hint, series, s, bool(cd.metres_estimated))}
    return ev


def attach(observations, cd):
    """Select and build the evidence of every observation of the series cd (detectors.detect records): fills each
    observation's 'evidence' with the ids, removes the private hints, and returns the evidence records in the order
    of the observations."""
    fx = Frames(cd)
    out = []
    for obs in observations:
        hint = obs.pop('_ev', None)
        obs['evidence'] = []
        if not hint or obs.get('classification') == 'insufficient' or obs.get('status') == 'suppressed':
            continue
        series = cd.teams[obs['team']]
        for item in select(fx, series, hint):
            ev = build(fx, obs, hint, item)
            if ev['id'] in obs['evidence']:
                continue
            obs['evidence'].append(ev['id'])
            out.append(ev)
    return out


# =====================================================================================================================
# v2 (DESIGN v2 5.4-5.6): moments of opportunities (measurement and role tiers) and the annotations of findings
# =====================================================================================================================
REACH_H = 2.0                       # s: a moment's 'before' scene is taken this long before it (DESIGN 5.1 H: the
                                    # suggested zone's reach is measured from the mover's position then)
WINDOW_BEFORE_S = 3.0               # s: an annotation lives from this long before its key time ... (DESIGN 5.6 [P])
WINDOW_AFTER_S = 2.0                # ... to this long after it [P]
FLAG_WINDOW_S = .5                  # s: a stabilizer flag on an involved player this close keeps a sample from being
                                    # the evidence frame while another sample of the run has none (DESIGN 3.5; v1
                                    # measured a flagged box foot 15-32 px off against 2-3 px)
PROJECT_MAX_PX = 20.0               # px at 1920: the disagreement rule (module docstring); measured median 0.7 px
ANCHOR_STRIDE = 5                   # grid samples (1 s at 5 Hz) between the role tier's pixel anchors
MAX_RANKED = 15                     # candidate samples of a run whose overlay tier is evaluated (evenly spread, the
                                    # middle always in): a paint check costs 0.5-2.6 ms and tiers change slowly
ANCHOR_VERSION = 1                  # obs['roleInputs']['evidence'] layout
ARROW_MIN_M = 1.0                   # m: a mover already this close to the area's centre gets no arrow
GAP_DEPTH_M = 3.0                   # m: a back-line gap is drawn at the line's depth +-3 m (DESIGN 5.2)
DM_LINE_M = 6.0                     # m: own players this close to the line height form the back line of a dm-gap area
TONE = {'strength': 'strength', 'try_this': 'review', 'work_on': 'review', 'review': 'review', 'describe': None}
CAPTION_SUFFIX = {'strength': 'a good example', 'try_this': 'something to try', 'work_on': 'a moment to work on',
                  'review': 'a possible issue to review', 'describe': 'what was measured'}
DEFAULT_TITLE = {'pressure_cover': 'Going to the ball', 'weak_side_balance': 'The far side',
                 'wide_defender_advance': 'Space behind a wide defender', 'behind_ball': 'Players behind the ball',
                 'defensive_spacing': 'Spacing of the back players', 'attacking_shape': 'Spread when we have the ball'}
ZONE_LABEL = {'pressure_cover': 'possible covering area', 'weak_side_balance': 'possible far-side area',
              'wide_defender_advance': 'possible covering area', 'behind_ball': 'possible area behind the ball',
              'defensive_spacing': 'possible area to close the gap'}
ZONE_LABEL_PLAIN = 'possible area'
WHY = {'carrier': 'player on the ball', 'cover': 'nearest teammate', 'cb': 'nearest central defender',
       'channel': 'teammate in the space', 'channel_opp': 'opponent in the space', 'behind': 'teammate behind the ball',
       'gap': 'defender beside the gap', 'gap_opp': 'opponent in the gap', 'mover': 'could move to the area'}
DIAGRAM_NOTE = 'Suggested area shown on the 2D diagram.'
CALIBRATION_KEYS = ('confidence', 'disagreement', 'paintResidualPx', 'paintSegments', 'paintTicks')
LEVEL_RANK = {'high': 0, 'medium': 1, 'low': 2}


def _rn(x, d=4):
    return round(float(x), d)


def _clip_norm(v):
    return min(max(float(v), -S.NORM_MARGIN), 1.0+S.NORM_MARGIN)


def _attack(x, y, side, L, W):
    """Normalized pitch -> the team's attack frame in metres (X from its own goal line, Y from its left touchline)."""
    if side == 'left':
        return x*L, y*W
    return (1.0-x)*L, (1.0-y)*W


def _norm_poly(points, side, L, W):
    return [[_rn(_clip_norm(a)), _rn(_clip_norm(b))] for a, b in (_to_norm(X, Y, side, L, W) for X, Y in points)]


def _sig(v):
    """A homography entry for the JSON files: 10 significant digits (deterministic bytes)."""
    return float(f'{float(v):.10g}')


# ---------- which sample of an opportunity is its moment ----------
def outcomes(opp, params, scale, as_detector=None):
    """contextual.sample_outcomes, or no outcome at all (None per sample) when the opportunity lacks a relation the
    rule reads (a hand-made or older record): evidence never invents an outcome."""
    n = len(opp['relations'].get('t') or ())
    try:
        return CX.sample_outcomes(opp, params, scale, as_detector=as_detector)
    except (KeyError, IndexError, TypeError):
        return [None]*n, [None]*n


def _runs(outs, want):
    runs, cur = [], None
    for k, v in enumerate(outs):
        if v == want:
            cur = (cur[0], k) if cur else (k, k)
        elif cur:
            runs.append(cur)
            cur = None
    if cur:
        runs.append(cur)
    return runs


def key_run(opp, scale):
    """(outcome, i0, i1): the run of an opportunity's samples its moment is taken from, profile-independently (under
    contextual.REFERENCE_PARAMS): the longest run of its DOMINANT outcome (failing when failing samples are at least as
    many as demonstrated ones, else demonstrated), the other outcome's longest run when it has none, else the whole
    opportunity (outcome 0). contextual.key_sample prefers any failing run; a demonstrated opportunity with a short
    failing stretch would then show a failing frame under a strength (module docstring, measured)."""
    outs, _ = outcomes(opp, None, scale)
    fail = sum(1 for v in outs if v == -1)
    demo = sum(1 for v in outs if v == 1)
    for want in ((-1, 1) if fail >= demo else (1, -1)):
        runs = _runs(outs, want)
        if runs:
            i0, i1 = max(runs, key=lambda r: (r[1]-r[0], -r[0]))
            return want, i0, i1
    return 0, 0, max(0, len(outs)-1)


def _choose(cands, mid, named, flagged, rank):
    """The evidence sample among a run's candidates: where the primary's identity is in doubt when it is anywhere in
    the run (a clean frame must not hide a crossing from the identity guardrail: counterexample 'identity crossing'),
    then without a stabilizer flag when possible, then the best rank (tier), then nearest the run's middle."""
    doubt = [k for k in cands if named(k) is False]
    if doubt:
        cands = doubt
    clean = [k for k in cands if not flagged(k)]
    if clean:
        cands = clean
    if len(cands) > MAX_RANKED:
        middle = min(cands, key=lambda k: (abs(k-mid), k))
        keep = np.unique(np.linspace(0, len(cands)-1, MAX_RANKED-1).round().astype(int))
        cands = sorted({cands[int(j)] for j in keep} | {middle})
    return max(cands, key=lambda k: (rank(k), -abs(k-mid), -k))


def _involved(opp, k):
    """The players an opportunity's moment involves: its actors and the ids its relations name at sample k (first
    defender, cover, weak-side defender, deepest and behind-the-ball players, gap defenders, channel coverer, near
    centre-back), primary first."""
    a, R = opp['actors'], opp['relations']
    out = list(a['primary'])
    for key in ('fd', 'coverId', 'nearestWeak', 'deepest', 'gapLeft', 'gapRight', 'nearCb', 'channelBy'):
        v = R.get(key)
        if v is not None and 0 <= k < len(v) and isinstance(v[k], str):
            out.append(v[k])
    b = R.get('behindIds')
    if b is not None and 0 <= k < len(b) and isinstance(b[k], str):
        out += b[k].split('|')
    out += list(a['supporting'])+list(a['opponents'])+([a['carrier']] if a['carrier'] else [])
    return [p for p in dict.fromkeys(out) if CX.valid_id(p)]


def _grid_index(t_grid, times):
    """The grid sample of each relation time (-1 when none lies within a millisecond)."""
    times = np.round(np.asarray(times, float), 3)
    pos = np.clip(np.searchsorted(t_grid, times), 0, max(0, len(t_grid)-1))
    out = []
    for p, t in zip(pos, times):
        best = -1
        for c in (p-1, p):
            if 0 <= c < len(t_grid) and abs(t_grid[c]-t) <= 1e-3:
                best = int(c)
        out.append(best)
    return out


def _nearest(t_grid, t, step):
    """The grid sample nearest t when it lies within half a step, else None."""
    if not len(t_grid):
        return None
    p = int(np.clip(np.searchsorted(t_grid, t), 0, len(t_grid)-1))
    best = min((c for c in (p-1, p) if 0 <= c < len(t_grid)), key=lambda c: abs(t_grid[c]-t))
    return best if abs(t_grid[best]-t) <= step/2.0+1e-6 else None


# ---------- measurement tier: evidence.moments ----------
def frames_of(cd):
    """The Frames of a CoachingData, made once and kept on it: the measurement tier runs once per team on the same
    series, and the paint checks (most of this stage's cost) are the same for both."""
    fx = cd.__dict__.get('_evidence_frames')
    if fx is None or fx.cd is not cd:
        fx = Frames(cd)
        cd.__dict__['_evidence_frames'] = fx
    return fx


def _keepers(tl):
    """{team: keeper id} where team_shape knows exactly one keeper of the team (GK-n)."""
    out = {}
    data = tl.cd.shape.observed if getattr(tl, 'cd', None) is not None and tl.cd.shape is not None else None
    if data is None:
        return out
    for team in (tl.team, tl.other):
        gks = (data.teams.get(team) or {}).get('gks') or []
        if len(gks) == 1 and CX.valid_id(str(gks[0])):
            out[team] = str(gks[0])
    return out


def _box(person):
    b = person.get('box') if person else None
    if isinstance(b, (list, tuple)) and len(b) == 4 and all(_num(v) for v in b):
        return [_rn(v) for v in b]
    return None


def _entry(team, x, y, sigma, goalkeeper, box):
    return {'team': team, 'x': _rn(_clip_norm(x)), 'y': _rn(_clip_norm(y)), 'kind': 'O',
            'sigma': _rn(min(float(sigma), 100.0), 2) if _num(sigma) else None, 'goalkeeper': bool(goalkeeper),
            'box': box}


def _scene_tl(tl, fx, s, keepers, boxed=()):
    """The scene at grid sample s of a timeline: every person observed there (the position gate: kind O, registration,
    sigma <= 3 m), keepers apart, the reliable ball; the frame's boxes of the team's own players and of the opponents
    in `boxed` (the moment's involved players: the only ones an overlay outlines)."""
    people = fx.people(s)
    players = {}
    for ids, x, y, ok, sig, team in ((tl.ids, tl.x, tl.y, tl.posOk, tl.sigma, tl.team),
                                     (tl.oppIds, tl.oppx, tl.oppy, tl.oppPosOk, tl.oppSigma, tl.other)):
        for j, pid in enumerate(ids):
            if not (ok[s, j] and np.isfinite(x[s, j]) and np.isfinite(y[s, j]) and CX.valid_id(pid)):
                continue
            box = _box(people.get(pid)) if team == tl.team or pid in boxed else None
            players[pid] = _entry(team, x[s, j], y[s, j], sig[s, j], False, box)
    if str(tl.regGate[s]) == '':
        for team, key in ((tl.team, 'gk'), (tl.other, 'oppGk')):
            gid = keepers.get(team)
            g = getattr(tl, key)[s] if gid else None
            if gid and g is not None and np.isfinite(g[0]) and np.isfinite(g[1]):
                players[gid] = _entry(team, g[0], g[1], None, True,
                                      _box(people.get(gid)) if team == tl.team or gid in boxed else None)
    ok_ball = bool(tl.ballReliable[s] and np.isfinite(tl.ballX[s]) and np.isfinite(tl.ballY[s]))
    return {'t': _rn(tl.t[s], 3), 'frame': int(tl.frame[s]), 'players': players,
            'ball': [_rn(_clip_norm(tl.ballX[s])), _rn(_clip_norm(tl.ballY[s]))] if ok_ball else None,
            'ballReliable': ok_ball}


def _own_only(scene, team):
    """A 'before' scene: the team's own players only (the movers of a suggested zone are team-mates; DESIGN 5.1 step
    5 reads only the mover's position then), without boxes; None when none of them was observed."""
    if scene is None:
        return None
    own = {p: dict(e, box=None) for p, e in scene['players'].items() if e['team'] == team}
    return dict(scene, players=own) if own else None


def _flagged_tl(tl, s, pids, t_grid):
    """The players of pids (own team) the stabilizer flagged within FLAG_WINDOW_S of sample s."""
    lo = int(np.searchsorted(t_grid, t_grid[s]-FLAG_WINDOW_S-1e-6, side='left'))
    hi = int(np.searchsorted(t_grid, t_grid[s]+FLAG_WINDOW_S+1e-6, side='right'))
    return [p for p in pids if p in tl.col and bool(np.any(tl.stabFlag[lo:hi, tl.col[p]]))]


def _fps(fx, cd):
    return round(float(fx.fps or float(((cd.result or {}).get('video') or {}).get('fps') or 0.0)), 6)


def _seek(frame, fps, t):
    return S.seek_of(frame, fps) if fps > 0 else _rn(t, 4)


def _homography(fx, s):
    H = oriented_h(fx.frame(s), fx.width, fx.height)
    return None if H is None else [_sig(v) for v in np.asarray(H, float).ravel()]


def _calibration(tier_info, side, pitch, H):
    cal = {k: tier_info.get(k) for k in CALIBRATION_KEYS}
    if side in ('left', 'right'):
        cal['side'] = side
    if pitch:
        cal['pitchLengthM'], cal['pitchWidthM'] = float(pitch[0]), float(pitch[1])
    if H is not None:
        cal.update({f'h{i}': v for i, v in enumerate(H)})
    return cal


def _role_mask(tl):
    """The grid samples the role tier stores (contextual.role_geometry): our phase (and on small formats the other
    team's) in spans of 3 s or more on frames passing the registration gates; the role tier's moments lie there."""
    small = tl.format['players'] != 11
    phase = np.asarray(tl.phase)
    with np.errstate(invalid='ignore'):
        long_span = np.asarray(tl.phaseSpan, float) >= SI.MIN_SHAPE_SPAN_S-1e-6
    return ((phase == 'own') | ((phase == 'opp') & small)) & long_span & (np.asarray(tl.regGate) == '')


def frame_tiers(tl, fx, mask=None):
    """obs['frameTiers']: the evidence tier (Frames.tier) of the samples the role tier may cite, as runs of
    consecutive grid samples of one tier [{t0, t1, frame0, frame1, tier, confidence (the run's lowest registration
    confidence) | None}]."""
    idx = np.flatnonzero(_role_mask(tl) if mask is None else mask)
    out, cur = [], None
    for s in idx:
        s = int(s)
        tier, info = fx.tier(s)
        c = info.get('confidence')
        if cur and cur['tier'] == tier and s == cur['_s']+1:
            cur.update(_s=s, t1=_rn(tl.t[s], 3), frame1=int(tl.frame[s]))
            if _num(c):
                cur['confidence'] = c if cur['confidence'] is None else min(cur['confidence'], c)
            continue
        if cur:
            out.append(cur)
        cur = {'_s': s, 't0': _rn(tl.t[s], 3), 't1': _rn(tl.t[s], 3), 'frame0': int(tl.frame[s]),
               'frame1': int(tl.frame[s]), 'tier': tier, 'confidence': float(c) if _num(c) else None}
    if cur:
        out.append(cur)
    for r in out:
        r.pop('_s')
        if r['confidence'] is not None:
            r['confidence'] = _rn(min(max(r['confidence'], 0.0), 1.0), 3)
    return out


def role_anchors(tl, fx, mask=None):
    """obs['roleInputs']['evidence'] (ANCHOR_VERSION): what the role tier needs to draw its moments over the video,
    since it never reads the result: every ANCHOR_STRIDE-th stored sample's people boxes (own outfield; opponents in
    the outer thirds of our half on 11v11 as contextual.role_geometry keeps them, all on small formats) and, on world
    frames, the oriented homography."""
    idx = [int(s) for s in np.flatnonzero(_role_mask(tl) if mask is None else mask) if int(s) % ANCHOR_STRIDE == 0]
    L, W = tl.scale['lengthM'], tl.scale['widthM']
    small = tl.format['players'] != 11
    frames = {}
    for s in idx:
        people = fx.people(s)
        boxes = {}
        for j, pid in enumerate(tl.ids):
            if tl.posOk[s, j] and _box(people.get(pid)):
                boxes[pid] = _box(people[pid])
        for j, pid in enumerate(tl.oppIds):
            if not tl.oppPosOk[s, j] or not _box(people.get(pid)):
                continue
            X, Y = float(tl.oppX[s, j]), float(tl.oppY[s, j])
            if small or (X < L/2.0 and (Y <= W/3.0 or Y >= 2.0*W/3.0)):
                boxes[pid] = _box(people[pid])
        tier, _ = fx.tier(s)
        frames[str(int(tl.frame[s]))] = {'h': _homography(fx, s) if tier == 'world' else None, 'boxes': boxes}
    pitch = (tl.cd.pitch or {}) if getattr(tl, 'cd', None) is not None else {}
    return {'version': ANCHOR_VERSION, 'stride': ANCHOR_STRIDE,
            'pitch': {'length': float(pitch.get('length') or 105.0), 'width': float(pitch.get('width') or 68.0)},
            'image': {'width': int(fx.width), 'height': int(fx.height)} if fx.width and fx.height else None,
            'frames': frames}


def moments(cd, timeline, opportunities):
    """The measurement tier's 'moments' stage (INTERFACES.md LEAD 4): one evidence moment per opportunity (schema
    'moment'; module docstring) and the frame tiers the role tier reads. Also returns 'roleInputs': {'evidence':
    role_anchors()} for the orchestrator to store beside DETECT's role inputs (a request to LEAD: until it is stored
    the role tier's moments carry no boxes and no homography)."""
    tl = timeline
    fx = frames_of(cd)
    fps = _fps(fx, cd)
    t_grid = np.round(np.asarray(tl.t, float), 3)
    step = float(tl.step)
    keepers = _keepers(tl)
    pitch = (float((cd.pitch or {}).get('length') or 105.0), float((cd.pitch or {}).get('width') or 68.0))
    out = []
    for o in opportunities:
        if o.get('abstain') is not None:
            continue
        R = o['relations']
        idx = _grid_index(t_grid, R['t'])
        _, i0, i1 = key_run(o, tl.scale)
        cands = [k for k in range(i0, i1+1) if idx[k] >= 0] or [k for k in range(len(idx)) if idx[k] >= 0]
        if not cands:
            continue
        named = R.get('primaryNamed')

        def flagged(k):
            return bool(_flagged_tl(tl, idx[k], [p for p in _involved(o, k) if p in tl.col], t_grid))
        k = _choose(cands, (i0+i1)/2.0, (lambda k: named[k]) if named is not None else (lambda k: None), flagged,
                    lambda k: TIER_RANK[fx.tier(idx[k])[0]])
        s = idx[k]
        frame, t = int(tl.frame[s]), _rn(tl.t[s], 3)
        scene = _scene_tl(tl, fx, s, keepers, boxed=set(_involved(o, k)))
        sb = _nearest(t_grid, t-REACH_H, step)
        before = _own_only(_scene_tl(tl, fx, sb, keepers), o['team']) if sb is not None and sb < s else None
        tier, info = fx.tier(s)
        players = [p for p in _involved(o, k) if p in scene['players']]
        out.append({'id': S.record_id('ev', o['id'], frame), 'opportunity': o['id'], 'detector': o['detector'],
                    'team': o['team'], 'sample': int(k), 't': t, 'frame': frame, 'seek': _seek(frame, fps, t),
                    'fps': fps, 'tier': tier,
                    'calibration': _calibration(info, str(tl.side[s]), pitch,
                                                _homography(fx, s) if tier == 'world' else None),
                    'players': players, 'flagged': _flagged_tl(tl, s, players, t_grid), 'scene': scene,
                    'before': before,
                    'image': {'width': int(fx.width), 'height': int(fx.height)} if fx.width and fx.height else None})
    mask = _role_mask(tl)
    return {'moments': out, 'frameTiers': frame_tiers(tl, fx, mask),
            'roleInputs': {'evidence': role_anchors(tl, fx, mask)}}


# ---------- role tier: evidence.moments_from_obs ----------
def _tier_at(tiers, frame):
    for r in tiers:
        if r['frame0'] <= frame <= r['frame1']:
            return r['tier'], r['confidence']
    return 'image', None


def _scene_geo(geo, i, team, anchor):
    """The scene of stored sample i of obs['roleInputs']['detect'] (attack frame -> normalized with the side the team
    defended): own outfield players, the opponents stored there, the reliable ball; boxes from the anchor when one
    exists for the frame."""
    smp = geo['samples']
    side = smp['side'][i]
    if side not in ('left', 'right'):
        return None
    L, W = float(geo['scale']['lengthM']), float(geo['scale']['widthM'])
    boxes = (anchor or {}).get('boxes') or {}
    other = 'B' if team == 'A' else 'A'
    players = {}
    ok = smp['ok'][i]
    for j, pid in enumerate(geo['ids']):
        X, Y = smp['X'][j][i], smp['Y'][j][i]
        if j < len(ok) and ok[j] == '1' and X is not None and Y is not None and CX.valid_id(pid):
            players[pid] = _entry(team, *_to_norm(X, Y, side, L, W), None, False, boxes.get(pid))
    if smp.get('oX'):
        opp = [(pid, smp['oX'][j][i], smp['oY'][j][i]) for j, pid in enumerate(geo['oppIds'])]
    else:
        opp = [tuple(e) for e in (smp['oppL'][i] or [])+(smp['oppR'][i] or [])]
    for pid, X, Y in opp:
        if X is not None and Y is not None and CX.valid_id(pid):
            players[pid] = _entry(other, *_to_norm(X, Y, side, L, W), None, False, boxes.get(pid))
    bx, by = smp['ballX'][i], smp['ballY'][i]
    ball = None
    if bx is not None and by is not None:
        x, y = _to_norm(bx, by, side, L, W)
        ball = [_rn(_clip_norm(x)), _rn(_clip_norm(y))]
    return {'t': _rn(smp['t'][i], 3), 'frame': int(smp['frame'][i]), 'players': players, 'ball': ball,
            'ballReliable': ball is not None}


def moments_from_obs(obs, opportunities):
    """The role tier's 'role_moments' stage (INTERFACES.md LEAD 4): one moment per role-tier opportunity, built from
    the obs file only (roleInputs['detect'] samples, frameTiers, and roleInputs['evidence'] anchors when stored)."""
    ri = obs.get('roleInputs') or {}
    geo = ri.get('detect') or {}
    smp = geo.get('samples') or {}
    anchors = ri.get('evidence') if isinstance(ri.get('evidence'), dict) else None
    aframes = (anchors or {}).get('frames') or {}
    t_grid = np.round(np.asarray(smp.get('t') or [], float), 3)
    step = float(geo.get('step') or obs['timeline'].get('step') or .2)
    scale = obs['timeline']['scale']
    fps = round(float(obs['segment']['fps'] or 0.0), 6)
    tiers = obs.get('frameTiers') or []
    image = (anchors or {}).get('image') or next((m['image'] for m in obs.get('moments') or () if m.get('image')),
                                                 None)
    apitch = (anchors or {}).get('pitch')
    team = obs['team']
    out = []
    for o in opportunities:
        if o.get('abstain') is not None or not len(t_grid):
            continue
        R = o['relations']
        idx = _grid_index(t_grid, R['t'])
        _, i0, i1 = key_run(o, scale)
        cands = [k for k in range(i0, i1+1) if idx[k] >= 0] or [k for k in range(len(idx)) if idx[k] >= 0]
        if not cands:
            continue
        named = R.get('primaryNamed')

        def rank(k):
            fr = int(smp['frame'][idx[k]])
            return str(fr) in aframes, TIER_RANK[_tier_at(tiers, fr)[0]]
        k = _choose(cands, (i0+i1)/2.0, (lambda k: named[k]) if named is not None else (lambda k: None),
                    lambda k: False, rank)
        i = idx[k]
        frame, t = int(smp['frame'][i]), _rn(smp['t'][i], 3)
        anchor = aframes.get(str(frame))
        scene = _scene_geo(geo, i, team, anchor)
        if scene is None:
            continue
        ib = _nearest(t_grid, t-REACH_H, step)
        before = _own_only(_scene_geo(geo, ib, team, None), team) if ib is not None and ib < i else None
        tier, conf = _tier_at(tiers, frame)
        cal_conf = smp['cal'][i] if smp['cal'][i] is not None else conf      # the frame's own, else its run's
        H = (anchor or {}).get('h')
        cal = _calibration({'confidence': cal_conf}, smp['side'][i],
                           (apitch['length'], apitch['width']) if apitch else None,
                           H if isinstance(H, list) and len(H) == 9 else None)
        players = [p for p in _involved(o, k) if p in scene['players']]
        out.append({'id': S.record_id('ev', o['id'], frame), 'opportunity': o['id'], 'detector': o['detector'],
                    'team': o['team'], 'sample': int(k), 't': t, 'frame': frame, 'seek': _seek(frame, fps, t),
                    'fps': fps, 'tier': tier, 'calibration': cal, 'players': players, 'flagged': [], 'scene': scene,
                    'before': before, 'image': dict(image) if image else None})
    return {'moments': out}


# ---------- interpretation tier: evidence.annotate ----------
def _register_numbers(profile):
    return ((profile or {}).get('register') or {}).get('numbers') or 'none'


def _metres_word(scale):
    if not scale.get('metresUsable'):
        return 'unusable'
    return {'measured': 'measured', 'declared': 'declared'}.get(scale.get('source'), 'estimated')


def _metres_text(v, profile, scale):
    """A metre value as the level's register allows it on a drawing (DESIGN 6.3): none below level 3, 'about N m' at
    rounded_metres or with estimated metres, 'N m' otherwise; nothing when the scale is unusable."""
    mode = _register_numbers(profile)
    word = _metres_word(scale)
    if not _num(v) or mode in ('none', 'counts') or word == 'unusable':
        return None
    if mode == 'rounded_metres' or word == 'estimated':
        return f'about {int(round(float(v)))} m'
    return f'{int(round(float(v)))} m'


def _count_text(n, what, profile):
    if not _num(n) or _register_numbers(profile) == 'none':
        return None
    return f'{int(n)} {what}'


def _banned(text, profile):
    low = text.lower()
    return any(b and b.lower() in low for b in ((profile or {}).get('register') or {}).get('banned') or ())


def _title(f, profile):
    skill = next((k for k, d in S.SKILLS.items() if d == f['observation']), None)
    name = ((profile or {}).get('skills') or {}).get(skill) if skill else None
    if not isinstance(name, str) or not name or _banned(name, profile):
        name = DEFAULT_TITLE[f['observation']]
    return name


def caption_of(f, profile):
    """The annotation caption of a finding when explain's short text is not there yet (annotate runs before explain):
    the level's own name of the checked behaviour (profiles.json skills) and what the moment is for. No number, no
    player, no relation claim, so nothing in it can contradict the measurement; the report stage may replace it with
    the finding's validated text.short (apply_captions)."""
    text = f.get('text') or {}
    short = text.get('short') if isinstance(text, dict) else None
    if isinstance(short, str) and 0 < len(short) <= S.CAPTION_MAX:
        return short
    return f'{_title(f, profile)}: {CAPTION_SUFFIX[f["verdict"]]}'[:S.CAPTION_MAX]


def _labels_allowed(f, profile):
    """May the focus ring carry the player's label? Only where the level names players in evidence (naming
    evidence_only / yes, DESIGN 4.9), the identity guardrail did not fail and the check is not team-level only."""
    naming = ((profile or {}).get('evidence') or {}).get('naming')
    params = (((profile or {}).get('detectors') or {}).get(f['observation']) or {}).get('params') or {}
    return (naming in ('evidence_only', 'yes') and f['guardrails'].get('identity') != 'fail'
            and not params.get('teamLevelOnly') and f['verdict'] != 'describe')


class _Annotation:
    """The elements of one evidence moment's annotation in normalized pitch coordinates, before tier gating."""

    def __init__(self, f, m, o, alt, obs, profile):
        self.f, self.m, self.o, self.alt, self.profile = f, m, o, alt, profile
        self.scale = obs['timeline']['scale']
        self.L, self.W = float(self.scale['lengthM']), float(self.scale['widthM'])
        self.kl = self.L/G.REF_LENGTH
        self.side = (m.get('calibration') or {}).get('side')
        self.scene = m['scene']['players']
        self.k = int(m['sample'])
        self.R = o['relations']
        self.tone = TONE[f['verdict']]
        self.label = _labels_allowed(f, profile)
        dp = (profile.get('detectors') or {}).get(f['observation']) or {}
        self.params = dp.get('params') or {}
        as_det = f['observation'] if f['observation'] != o['detector'] else None
        self.outs, self.modes = outcomes(o, self.params, self.scale, as_detector=as_det)
        self.focus, self.support, self.relations, self.measures = [], [], [], []
        self.channel = self.zone = self.arrow = None
        self.emphasis_team = None
        self.caption = caption_of(f, profile)
        getattr(self, '_'+m['detector'])()
        if alt is not None:
            self._alternative()

    # helpers
    def rel(self, key):
        v = self.R.get(key)
        return v[self.k] if v is not None and 0 <= self.k < len(v) else None

    def seen(self, pid):
        return isinstance(pid, str) and pid in self.scene

    def at(self, pid):
        e = self.scene[pid]
        return _attack(e['x'], e['y'], self.side, self.L, self.W)

    def add_focus(self, pid, secondary=False):
        if self.tone and self.seen(pid) and pid not in [p for p, _ in self.focus] and \
                len(self.focus) < S.ELEMENT_BUDGET['focus']:
            self.focus.append((pid, secondary))

    def add_support(self, pid, why, first=False):
        if not self.seen(pid) or pid in [p for p, _ in self.focus] or pid in [p for p, _ in self.support]:
            return
        if first:
            self.support.insert(0, (pid, why))
        else:
            self.support.append((pid, why))
        del self.support[S.ELEMENT_BUDGET['support']:]

    def add_relation(self, a, b, kind, ok, text=None, pb=None):
        if len(self.relations) >= S.ELEMENT_BUDGET['relation'] or not self.seen(a):
            return
        if b in ('ball', 'goal', 'zone') or self.seen(b):
            self.relations.append({'a': a, 'b': b, 'kind': kind, 'ok': bool(ok), 'text': text, 'pb': pb})

    def poly(self, points):
        if self.side not in ('left', 'right'):
            return None
        return _norm_poly(points, self.side, self.L, self.W)

    def judged_ok(self):
        v = self.outs[self.k] if 0 <= self.k < len(self.outs) else None
        return v != -1

    # detectors (elements per DESIGN 5.2 / 5.5)
    def _pressure_cover(self):
        fd = self.rel('fd') or (self.o['actors']['primary'] or [None])[0]
        carrier = self.o['actors']['carrier']
        self.add_focus(fd)
        self.add_support(carrier, WHY['carrier'])
        if self.rel('d1') is None or not self.seen(fd):
            return
        try:
            pressure, cover = CX.pressure_cover_flags(self.R, self.k, self.params, self.scale)
        except (KeyError, IndexError, TypeError):
            return
        self.add_relation(fd, carrier, 'pressure', pressure, _metres_text(self.rel('d1'), self.profile, self.scale))
        mate = self.rel('coverId')
        if self.params.get('coverRequired', True) and mate:
            self.add_support(mate, WHY['cover'])
            self.add_relation(fd, mate, 'cover', cover, _metres_text(self.rel('coverDist'), self.profile, self.scale))

    def _weak_side_balance(self):
        prim = self.rel('nearestWeak') or (self.o['actors']['primary'] or [None])[0]
        self.add_focus(prim)
        self.add_support(self.o['actors']['carrier'], WHY['carrier'])
        bx, side = self.rel('ballX'), self.rel('ballSide')
        if _num(bx) and side in ('left', 'right') and bx > 1.0:
            y0, y1 = (self.W/2.0, self.W) if side == 'left' else (0.0, self.W/2.0)
            p = self.poly([(0.0, y0), (bx, y0), (bx, y1), (0.0, y1)])
            if p:
                self.channel = (p, 'weak_side')

    def _wide_defender_advance(self):
        wd = (self.o['actors']['primary'] or [None])[0]
        self.add_focus(wd)
        near, by = self.rel('nearCb'), self.rel('channelBy')
        covered = bool(self.rel('channelOccupied') or self.rel('cbAdjusted') or self.rel('interchange') or
                       (self.params.get('useDmRotation') and self.rel('dmDropped')))
        if self.seen(near):
            self.add_support(near, WHY['cb'])
            self.add_relation(wd, near, 'cover', covered)
        if by and by != near:
            self.add_support(by, WHY['channel'])
        side, adv = self.o.get('variant'), self.rel('advanceM')
        if self.seen(wd) and side in ('left', 'right') and _num(adv) and self.side in ('left', 'right'):
            wx, _ = self.at(wd)
            x0, x1, y0, y1 = G.vacated_channel(wx-float(adv), wx, side, self.W, G.CHANNEL_BEHIND_M*self.kl)
            if x1-x0 >= 1.0:
                self.channel = (self.poly([(x0, y0), (x1, y0), (x1, y1), (x0, y1)]), 'vacated')
        for pid in self.o['actors']['opponents']:
            self.add_support(pid, WHY['channel_opp'])

    def _behind_ball(self):
        deep = self.rel('deepest') or (self.o['actors']['primary'] or [None])[0]
        self.add_focus(deep)
        for pid in (self.rel('behindIds') or '').split('|'):
            if pid and pid != deep:
                self.add_support(pid, WHY['behind'])
        ball = self.m['scene'].get('ball')
        if ball and self.seen(deep):
            self.add_relation(deep, 'ball', 'behind_ball', self.judged_ok(),
                              _count_text(self.rel('behindN'), 'behind', self.profile), pb=ball)

    def _defensive_spacing(self):
        a, b = self.rel('gapLeft'), self.rel('gapRight')
        prim = (self.o['actors']['primary'] or [None])[0]
        prim = prim if prim in (a, b) else a
        other = b if prim == a else a
        self.add_focus(prim)
        self.add_support(other, WHY['gap'])
        mode = self.modes[self.k] if 0 <= self.k < len(self.modes) else None
        lateral_fail = self.outs[self.k] == -1 and mode == 'lateral_gap' if 0 <= self.k < len(self.outs) else False
        if self.seen(a) and self.seen(b):
            self.add_relation(prim, other, 'gap', not lateral_fail,
                              _metres_text(self.rel('lateralGapM'), self.profile, self.scale))
        if self.side not in ('left', 'right'):
            return
        if mode == 'dm_gap' and _num(self.rel('lineHeightM')) and _num(self.rel('dmGapM')):
            h, g = float(self.rel('lineHeightM')), float(self.rel('dmGapM'))
            ys = [self.at(p)[1] for p, e in self.scene.items() if e['team'] == self.o['team'] and not e['goalkeeper']
                  and abs(self.at(p)[0]-h) <= DM_LINE_M]
            if len(ys) >= 2 and g >= 1.0:
                self.channel = (self.poly([(h, min(ys)), (h+g, min(ys)), (h+g, max(ys)), (h, max(ys))]), 'gap')
        elif self.seen(a) and self.seen(b):
            (ax, ay), (bx, by) = self.at(a), self.at(b)
            xm = (ax+bx)/2.0
            lo, hi = sorted((ay, by))
            if hi-lo >= 1.0:
                self.channel = (self.poly([(xm-GAP_DEPTH_M, lo), (xm+GAP_DEPTH_M, lo), (xm+GAP_DEPTH_M, hi),
                                           (xm-GAP_DEPTH_M, hi)]), 'gap')
                lo_x = min(ax, bx)-G.GAP_BEHIND_M*self.kl
                hi_x = max(ax, bx)+G.GAP_AHEAD_M*self.kl
                for pid, e in self.scene.items():
                    if e['team'] != self.o['team'] and not e['goalkeeper']:
                        X, Y = self.at(pid)
                        if lo < Y < hi and lo_x <= X <= hi_x:
                            self.add_support(pid, WHY['gap_opp'])

    def _attacking_shape(self):
        team = self.o['team']
        own = [(p, self.at(p)) for p, e in self.scene.items() if e['team'] == team and not e['goalkeeper']] \
            if self.side in ('left', 'right') else []
        self.emphasis_team = team
        if len(own) < 2:
            return
        ws, ds = self.rel('widthShare'), self.rel('depthShare')
        width_fail = _num(ws) and self.params.get('judgeWidth', True) is not False and \
            ws < float(self.params.get('widthShare', CX.REFERENCE_PARAMS['attacking_shape']['widthShare']))
        if width_fail or not _num(ds):
            lo = min(own, key=lambda q: q[1][1])
            hi = max(own, key=lambda q: q[1][1])
            xm = sum(q[1][0] for q in own)/len(own)
            a, b = (xm, lo[1][1]), (xm, hi[1][1])
            text = _metres_text(self.rel('widthM'), self.profile, self.scale)
        else:
            lo = min(own, key=lambda q: q[1][0])
            hi = max(own, key=lambda q: q[1][0])
            ym = sum(q[1][1] for q in own)/len(own)
            a, b = (lo[1][0], ym), (hi[1][0], ym)
            text = _metres_text(self.rel('depthM'), self.profile, self.scale)
        pa, pb = _norm_poly([a, b], self.side, self.L, self.W)
        self.measures.append({'a': pa, 'b': pb, 'text': text, 'ids': [lo[0], hi[0]]})

    def _alternative(self):
        alt = self.alt
        poly = [[_rn(_clip_norm(x)), _rn(_clip_norm(y))] for x, y in alt['zone']['polygon']]
        if len(poly) < 3:
            return
        label = ZONE_LABEL.get(self.f['observation'], ZONE_LABEL_PLAIN)
        if _banned(label, self.profile):
            label = ZONE_LABEL_PLAIN
        if alt['display'] in ('zone', 'point'):
            self.zone = (poly, 'point' if alt['display'] == 'point' else alt['zone']['kind'], alt['id'], alt['mover'],
                         label)
        mover = alt['mover']
        start = [self.scene[mover]['x'], self.scene[mover]['y']] if self.seen(mover) else list(alt['from'])
        cx = sum(p[0] for p in poly)/len(poly)
        cy = sum(p[1] for p in poly)/len(poly)
        dx, dy = (cx-start[0])*self.L, (cy-start[1])*self.W
        dist = math.hypot(dx, dy)
        if dist >= ARROW_MIN_M:
            if alt['display'] == 'direction':
                cap = float(alt['reach']['radiusM'])
                if cap > 0 and dist > cap:
                    cx, cy = start[0]+dx*cap/dist/self.L, start[1]+dy*cap/dist/self.W
            self.arrow = (mover, [_rn(start[0]), _rn(start[1])], [_rn(_clip_norm(cx)), _rn(_clip_norm(cy))])
        if self.seen(mover):
            self.add_support(mover, WHY['mover'], first=True)


def _diagram(an, obs, f, m):
    scale = an.scale
    side = an.side
    drawn = {p for p, _ in an.focus} | {p for p, _ in an.support}
    els = []
    for pid, e in sorted(an.scene.items()):
        if pid in drawn:
            continue
        el = {'type': 'player', 'id': pid, 'team': e['team'], 'x': e['x'], 'y': e['y'], 'kind': 'O',
              'emphasis': bool(an.emphasis_team and e['team'] == an.emphasis_team and not e['goalkeeper'])}
        els.append(el)
    if an.channel and an.channel[0]:
        els.append({'type': 'channel', 'polygon': an.channel[0], 'kind': an.channel[1]})
    if an.zone:
        poly, kind, aid, mover, label = an.zone
        els.append({'type': 'suggestedZone', 'polygon': poly, 'kind': kind, 'alternative': aid, 'mover': mover,
                    'label': label})
    for r in an.relations:
        el = {'type': 'relation', 'a': r['a'], 'b': r['b'], 'kind': r['kind'], 'ok': r['ok']}
        if r['text']:
            el['text'] = r['text']
        if r['pb'] is not None:
            el['pb'] = list(r['pb'])
        els.append(el)
    for ms in an.measures:
        el = {'type': 'measure', 'a': ms['a'], 'b': ms['b'], 'ids': ms['ids']}
        if ms['text']:
            el['text'] = ms['text']
        els.append(el)
    if an.arrow:
        els.append({'type': 'moveArrow', 'id': an.arrow[0], 'from': an.arrow[1], 'to': an.arrow[2]})
    for pid, why in an.support:
        e = an.scene[pid]
        els.append({'type': 'support', 'id': pid, 'team': e['team'], 'x': e['x'], 'y': e['y'], 'why': why})
    for pid, secondary in an.focus:
        e = an.scene[pid]
        el = {'type': 'focus', 'id': pid, 'x': e['x'], 'y': e['y'], 'tone': an.tone, 'label': an.label}
        if secondary:
            el['secondary'] = True
        els.append(el)
    els.append({'type': 'caption', 'text': an.caption, 'anchor': 'top-left'})
    L = min(max(float(scale['lengthM']), 15.0), 130.0)
    W = min(max(float(scale['widthM']), 10.0), 100.0)
    return {'preset': m['detector'], 'title': _title(f, an.profile)[:120], 't': m['t'], 'frame': m['frame'],
            'team': m['team'], 'attack': {'left': 'right', 'right': 'left'}.get(side), 'metresEstimated':
            _metres_word(scale) in ('estimated', 'unusable'), 'pitch': {'length': L, 'width': W}, 'elements': els}


def _project(H, pts):
    """Pitch metres -> source pixels through the oriented homography, None when a point is behind the camera."""
    pts = np.asarray(pts, float).reshape(-1, 2)
    if not (field.depth(H, pts) > 1e-9).all():
        return None
    return field.project(H, pts)


def _overlay(an, m):
    """(VideoOverlaySpec | None, tier, area drawn on the video): the annotation in the source pixels of the moment's
    frame, gated by tier (DESIGN 5.4). focus / support stand on the box feet (the box outlined at image tier);
    relations join feet (world, pinned); the channel, the suggested zone and the arrow are projected through the
    frame's homography at world only, after the disagreement rule (the involved players' projected positions within
    PROJECT_MAX_PX of their feet), and only when every corner is in front of the camera and the centre is in the
    picture. Nothing placeable: None (and the moment's own tier)."""
    img = m.get('image')
    tier = m['tier']
    if not img:
        return None, tier, False
    Wp, Hp = float(img['width']), float(img['height'])
    cal = m.get('calibration') or {}
    H = None
    if all(_num(cal.get(f'h{i}')) for i in range(9)) and _num(cal.get('pitchLengthM')) and _num(cal.get('pitchWidthM')):
        H = np.array([float(cal[f'h{i}']) for i in range(9)]).reshape(3, 3)
    PL, PW = float(cal.get('pitchLengthM') or 0.0), float(cal.get('pitchWidthM') or 0.0)
    scene = an.scene

    def box_px(pid):
        b = scene[pid].get('box') if pid in scene else None
        return [b[0]*Wp, b[1]*Hp, b[2]*Wp, b[3]*Hp] if b else None

    def foot(pid):
        b = box_px(pid)
        return (b[0]+b[2]/2.0, b[1]+b[3]) if b else None

    def proj(pt):
        if H is None:
            return None
        q = _project(H, [[float(pt[0])*PL, float(pt[1])*PW]])
        return None if q is None else (float(q[0][0]), float(q[0][1]))
    ids = [p for p, _ in an.focus]+[p for p, _ in an.support]+[r['a'] for r in an.relations] + \
        [r['b'] for r in an.relations if r['b'] in scene]+[p for ms in an.measures for p in ms['ids']]
    if tier == 'world':
        ds = []
        for pid in dict.fromkeys(ids):
            fp, q = foot(pid), proj((scene[pid]['x'], scene[pid]['y']))
            if fp and q:
                ds.append(math.hypot(q[0]-fp[0], q[1]-fp[1]))
        if H is None or not ds or float(np.median(ds)) > PROJECT_MAX_PX*Wp/PAINT_REFERENCE_WIDTH:
            tier = 'pinned'

    def place(pid):
        fp = foot(pid)
        if fp:
            return fp
        return proj((scene[pid]['x'], scene[pid]['y'])) if tier == 'world' and pid in scene else None
    els, people = [], 0

    def r1(p):
        return [_rn(p[0], 1), _rn(p[1], 1)]
    for pid, why in an.support:
        p, b = place(pid), box_px(pid)
        if p is None or (tier == 'image' and b is None):
            continue
        el = {'type': 'support', 'id': pid, 'team': scene[pid]['team'], 'x': _rn(p[0], 1), 'y': _rn(p[1], 1),
              'why': why}
        if b:
            el['box'] = [_rn(v, 1) for v in b]
        els.append(el)
        people += 1
    for pid, secondary in an.focus:
        p, b = place(pid), box_px(pid)
        if p is None or (tier == 'image' and b is None):
            continue
        el = {'type': 'focus', 'id': pid, 'x': _rn(p[0], 1), 'y': _rn(p[1], 1), 'tone': an.tone, 'label': an.label}
        if secondary:
            el['secondary'] = True
        if b:
            el['box'] = [_rn(v, 1) for v in b]
        els.append(el)
        people += 1
    area = False
    if tier == 'world':
        def poly_px(poly):
            q = _project(H, [[x*PL, y*PW] for x, y in poly])
            if q is None:
                return None
            c = q.mean(axis=0)
            if not (0 <= c[0] <= Wp and 0 <= c[1] <= Hp):
                return None
            return [r1(p) for p in q]
        if an.channel and an.channel[0]:
            q = poly_px(an.channel[0])
            if q:
                els.insert(0, {'type': 'channel', 'polygon': q, 'kind': an.channel[1]})
        if an.zone:
            poly, kind, aid, mover, label = an.zone
            q = poly_px(poly)
            if q:
                els.insert(0, {'type': 'suggestedZone', 'polygon': q, 'kind': kind, 'alternative': aid,
                               'mover': mover, 'label': label})
                area = True
        if an.arrow:
            a, b = proj(an.arrow[1]), proj(an.arrow[2])
            if a and b and (an.zone is None or area):
                if an.arrow[0] in scene and foot(an.arrow[0]):
                    a = foot(an.arrow[0])
                els.append({'type': 'moveArrow', 'id': an.arrow[0], 'from': r1(a), 'to': r1(b)})
    if tier in ('world', 'pinned'):
        for ms in an.measures:              # attacking shape: the width or depth between the two extreme players' feet
            a, b = foot(ms['ids'][0]), foot(ms['ids'][1])
            if a and b:
                el = {'type': 'measure', 'a': r1(a), 'b': r1(b), 'ids': list(ms['ids'])}
                if ms['text']:
                    el['text'] = ms['text']
                els.append(el)
                people += 1
        goal = None
        if an.side in ('left', 'right'):
            goal = (0.0, .5) if an.side == 'left' else (1.0, .5)
        for r in an.relations:
            pa = place(r['a'])
            if r['b'] == 'ball':
                pb = proj(r['pb']) if tier == 'world' and r['pb'] is not None else None
            elif r['b'] == 'goal':
                pb = proj(goal) if tier == 'world' and goal else None
            elif r['b'] == 'zone':
                pb = None
            else:
                pb = place(r['b'])
            if pa is None or pb is None:
                continue
            el = {'type': 'relation', 'a': r['a'], 'b': r['b'], 'kind': r['kind'], 'ok': r['ok'], 'pa': r1(pa),
                  'pb': r1(pb)}
            if r['text']:
                el['text'] = r['text']
            els.append(el)
            people += 1
    if not people:
        return None, m['tier'], False       # nothing to draw over the video: the moment's tier stands
    els.append({'type': 'caption', 'text': an.caption, 'anchor': 'top-left'})
    return {'tier': tier, 'width': int(Wp), 'height': int(Hp), 'elements': els}, tier, area


def _confidence(o, tier):
    q = o['quality']
    level = max((q.get('identity'), q.get('position')), key=lambda v: LEVEL_RANK.get(v, 2))
    level = level if level in LEVEL_RANK else 'low'
    if tier == 'image':
        return 'low'
    if tier == 'pinned' and level == 'high':
        return 'medium'
    return level


def _with_note(caption):
    """The evidence caption of a moment whose suggested area is on the diagram only (DESIGN 5.4)."""
    return f'{caption}{"" if caption.endswith(".") else "."} {DIAGRAM_NOTE}'


def evidence_of(f, m, o, alt, obs, profile):
    """One Evidence record (schema 'evidence') of a finding at one of its moments."""
    an = _Annotation(f, m, o, alt, obs, profile)
    diagram = _diagram(an, obs, f, m) if an.scene else None
    overlay, tier, area = _overlay(an, m)
    t = float(m['t'])
    seg = obs['segment']
    t0 = max(float(seg['t0']), t-WINDOW_BEFORE_S)
    t1 = min(float(seg['t1']), t+WINDOW_AFTER_S)
    caption = _with_note(an.caption) if an.zone and not area else an.caption
    cal = m.get('calibration')
    return {'id': m['id'], 'opportunity': m['opportunity'], 'finding': f['id'], 'detector': m['detector'],
            'team': m['team'], 'time': _rn(t, 3), 'frame': int(m['frame']), 'seek': m['seek'], 'fps': m['fps'],
            'segmentTime': _rn(max(0.0, t-float(seg['t0'])), 3), 'tier': tier,
            'calibration': {k: cal.get(k) for k in CALIBRATION_KEYS if k in cal} if cal else None,
            'players': list(m['players']), 'flagged': list(m['flagged']), 'confidence': _confidence(o, tier),
            'alternative': an.zone[2] if an.zone else (alt['id'] if alt is not None else None),
            'diagram': diagram, 'videoOverlay': overlay,
            'annotation': {'finding': f['id'], 'window': {'t0': _rn(min(t0, t), 3), 't1': _rn(max(t1, t), 3)},
                           'keyFrame': int(m['frame']), 'keyTime': _rn(t, 3), 'caption': an.caption},
            'caption': caption[:200]}


def annotate(findings, obs, roles, profile, alternatives=None):
    """The interpretation tier's 'annotate' stage (INTERFACES.md LEAD 4): one Evidence record per moment the findings
    cite (chronological; id = the moment's id), with its top-down diagram, its tier-gated video overlay and its
    annotation window. alternatives: ZONES' Alternative records; a finding's zone is drawn only on the evidence the
    alternative names (alt['evidence']). Raises ValueError when a finding cites a moment or an opportunity the files do
    not hold (invariant 3)."""
    moments_by = {m['id']: m for m in list(obs.get('moments') or [])+list((roles or {}).get('moments') or [])}
    opps = {o['id']: o for o in list(obs.get('opportunities') or [])+list((roles or {}).get('opportunities') or [])}
    alts = {}
    for a in alternatives or ():
        if isinstance(a, dict) and 'zone' in a:
            alts.setdefault(a['finding'], a)
    out = {}
    for f in findings:
        alt = alts.get(f['id'])
        for ts in f['timestamps']:
            mid = ts['evidence']
            if mid in out:
                continue
            m = moments_by.get(mid)
            if m is None or m['opportunity'] not in opps:
                raise ValueError(f'finding {f["id"]} cites {mid}, which is not a measured moment (invariant 3)')
            out[mid] = evidence_of(f, m, opps[m['opportunity']], alt if alt is not None and alt['evidence'] == mid
                                   else None, obs, profile)
    return sorted(out.values(), key=lambda e: (e['time'], e['id']))


def apply_captions(evidence, findings):
    """Copies of the evidence records with each annotation caption (and the drawings' caption elements) replaced by
    its finding's validated text.short when the finding has one (explain runs after annotate; DESIGN 5.5 'short
    text'). The report stage may call it; evidence of findings without text keeps its caption."""
    shorts = {}
    for f in findings:
        text = f.get('text') if isinstance(f, dict) else None
        short = text.get('short') if isinstance(text, dict) else None
        if isinstance(short, str) and 0 < len(short) <= S.CAPTION_MAX:
            shorts[f['id']] = short
    out = []
    for ev in evidence:
        ev = copy.deepcopy(ev)
        short = shorts.get(ev.get('finding'))
        if short and ev.get('annotation'):
            ev['annotation']['caption'] = short
            for spec in (ev.get('diagram'), ev.get('videoOverlay')):
                for el in (spec or {}).get('elements') or ():
                    if el.get('type') == 'caption':
                        el['text'] = short
            ev['caption'] = (_with_note(short) if ev['caption'].endswith(DIAGRAM_NOTE) else short)[:200]
        out.append(ev)
    return out


TIER_ELEMENTS = S.TIER_ELEMENTS     # DESIGN 5.4: shared with render.py and coaching.js (same table)
