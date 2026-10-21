"""Evidence for the coaching observations: the moments a coach should watch, each with its exact source frame, a
top-down diagram of the observed positions there (DiagramSpec) and, where the frame's registration supports it, a
video overlay (VideoOverlaySpec). Contract section 6; drawn by soccer/coaching/render.py (PDF, PNG) and coaching.js.

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
"""
import math

import numpy as np

from .. import field, pitch_model, registration
from . import detectors as det
from . import episodes as ep

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
