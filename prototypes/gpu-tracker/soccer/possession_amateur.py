"""Amateur possession: an image-based estimate of which team has the ball, for the footage possession-4 declines to
judge (soccer/possession.py, frozen: a small, far scene on half the frames or more, or the pitch and the image
disagreeing on the team at the ball). It never runs a model or reads the video either.

compute(result, settings=None) is the dispatch every consumer calls (server /api/possession, the coaching measurement
tier): possession-4's own record, byte for byte, wherever possession-4 judges the footage (it is computed by the same
two calls compute() makes); else, where possession-4 declines for the 'small_scene' or the 'agreement' reason
(dispatch_reason), estimate(): possession-4's record shape (version, fps, t0, t1, available, reason, quality, shares,
timeline, changes, deadSpells) with version 'possession-amateur-1', 'basis' 'image', 'fallbackFrom' {version, reason,
gate} quoting possession-4, quality level 'limited' with notes saying it is an image-based estimate and
quality.abstained {reason: seconds}, and a 'why' on every unknown timeline run (REASONS). A ball placed on too few
frames or no player near it is possession-4's own record (its reason). settings 'engine': 'possession-4' never gives
the estimate, 'amateur' always; settings this module alone reads (OWN) never reach possession-4.

Why a second engine. The possession audit (scratch amball/possession_audit, DEV data): forced past its gates,
possession-4 named amateur 3 (118.2-182 s) at .574 on .600 of the 90 labelled team points, against .567 for always
answering the commoner team; its image metric (ball_path.within_reach: rows doubled) is depth-blind on a low camera,
where an image metre in depth (rows doubled) is 4.7 (amateur 3) / 3.9 (Veo) pitch metres against 1.7-1.9 on the
broadcast; and
its wrong calls came from the people layer (an undetected carrier near the camera, teamless goalkeepers on reversed
tactical.goalEnds, an identity sitting on a player in the other kit) and from a ball path on a painted arc tip.

The evidence (features(), decide()). A team is at the ball on a frame when one of its ELIGIBLE carriers is within
reach of a TRUSTED ball, and nobody else is:
- Reach: the feet (box bottom centre) to the ball's ground point, through the local shape of the frame's
  calibration.H at the ball (the Jacobian of image -> pitch divided by its column scale, _shape): a displacement across
  the image keeps its length, one along the rows is stretched by the depth factor, and pixels become metres through
  the people's box heights there (ball_path._pixels_per_metre, 1.8 m), so the assumed 105 x 68 pitch and the players'
  real height cancel out of the shape. Full evidence within reach_r0 = 1.0 m, none from reach_r1 = 2.0 m (possession-4's
  image radii). The depth factor at the ball: median 10.5 (amateur 3) / 10.7 (Veo), p90 13.5 / 16.1, max 14.8 / 22.7,
  against 3.1 / 3.3 on the broadcast runs 0-60 / 42-222 s; it is capped at max_depth = 25. Without a calibration at the
  ball the rows weigh the run's own median depth factor (row_frames = 30 calibrated frames or more), else row_weight =
  10.5 on a small-scene run (the amateur medians) and row_weight_wide = 3.2 elsewhere (the broadcast medians). The
  row noise the depth factor multiplies is small (label-free, residuals of 7-frame line fits per track): feet rows
  p90 1.5 / 3.3 px, the ball's ground row 1.3 / 1.5 px (amateur 3 / Veo).
- Trusted ball: TRACKED, or interpolated inside a bridge of at most max_bridge = .5 s; where the ball tracker's first
  pass is more than fp_agree = 2 m (the reach metric) from the final path, both must point to the same team (one
  team's eligible carrier within reach_r1 and nobody else), else the frame is not trusted. possession-4's checks of a
  path on another object (excursions, replays) remove their frames. A FAST ball gives no evidence (in flight).
- Eligible carrier: a confirmed identity of a team (state 'confirmed', role PLAYER) whose kit readings on the same
  track within kit_s = 1 s (people[].why 'looks ref R A a B b', the identity tracker's) favour its team (mean A - B of
  the right sign); with no reading in that second it is accepted (kit_none 'accept'; 'reject' is the stricter
  option). A goalkeeper plays for his roster team, his frame's, or, teamless, the team defending the goal on his side
  (tactical.goalEnds, one end named gives the other; keeper_ends 'ignore' makes him nobody's), with or without an
  identity (PitchIQ gives teamless keepers none). A ball in a keeper's hands (possession-4's HANDS_* test, HANDS_MIN
  frames running) is his team's. Anyone else who may be a player of either team (an unconfirmed or unteamed track, an
  identity on the other kit) near the ball takes the evidence away, as an opponent does:
  e = p_k (1 - p_other) (1 - p_unidentified) dt, a bridged frame bridge_w.
- Contested: both teams at the ball is evidence for nobody.
Decoding: possession-4's two-state Viterbi and posterior (switch .15 s, tau .05, a team named from posterior .8; a
change starts only on a run of one carrier's evidence that holds an observed sighting, P4.confirmed_runs) between
boundaries: a stretch without evidence of either team that holds nocarry = 1.5 s of trusted ball is never carried
(a placed ball nobody of either team is seen at: on amateur footage an undetected carrier is likelier than a long
pass, so a 1.5 s pass breaks possession too); a stretch of gap = 1.5 s without a trusted ball otherwise is carried
when the same team is decoded on both sides (possession-4's rule, estimated, low); a camera cut restarts from an
observed control. Dead balls are possession-4's stoppage and out-of-play spells on this evidence (P4.boundaries), kept
only where a touchline or goal line within line_near = 10 m of the ball lies inside the camera's view footprint
(frames[].view) on half or more of the spell's frames with a ball pitch position (line_in_view), else unknown
('registration'). Confidence is never 'high': timeline runs 'medium' (named, dead) or 'low' (carried, unknown);
every change is 'low', reported as the boundary of two named spans (lostAt: the last named frame of the old team,
wonAt: the first of the new; cause out_of_play across a dead spell, else unclear), since no variant of the audit
dated amateur 3's changes within 1 s.
Abstention is per span: every unknown frame gets a reason (ball_untrusted, carrier_unidentified, contested,
nobody_at_ball, registration, uncertain: the posterior under .8, cut), an unknown stretch is cut where it changes, and
a piece shorter than REASON_MIN_S = .5 s keeps its neighbour's.

Measured (DEV only: tools/annotations/possession AM.json / VEO.json were used for development since possession-3;
amateur 3 0-60 s and 182-242 s are held out for blind labels and were not looked at; tools/possession_amateur_bench.py
reproduces every labelled-run number below):
    amateur 3, 118.2-182 s (run 20261018-174919-2e3174), 90 team points, base rate .567:
        possession-4: abstains (small scene .600); forced .574 at .600 (31/54); the audit's image-only variant .733 at
        .667 (44/60)
        this estimate: 1.0 at .356 (32/32), with the stored and with the corrected goal ends alike; 2 changes
        reported, 2 of the 4 labelled ones within 2 s, 1 within 1 s (161.65 s against 161.83); 20 of the 22 dead points
        dead (the throw-in 172-181.7 s), 34 points dead of which 20 labelled dead and 14 unclear (164.8-171.7 s, the
        ball out at an unseen moment); of the 23 points forced possession-4 had wrong, 22 are unknown and 1 right
        (arc-tip path 5, the undetected near-camera carrier 9, teamless keeper 5, identity on the other kit 4)
    Veo, 1-31 s (run 20261018-143758-8b0bab), 51 team points, all A (base rate 1.0): possession-4 abstains (.607);
        the image-only variant .957 at .902; this estimate 1.0 at .196 (10/10), no change (the clip's one change has
        no wonAt), 0 of the 8 dead points (the goal kick after the ball rolled out over the goal line: no stoppage
        found, unknown)
    the BALL agent's interim reruns (scratch amball/ball/out_p1, small scene from the first frame): amateur 3 .968 at
        .344 (30/31; 125.7 s: an A forward at the ball while the B keeper who has it is not detected), Veo 1.0 at .235
    one rule off at a time (amateur 3 / Veo): first-pass rule .886 at .389 (153.7-155.2 s wrong) / unchanged;
        no-carry .648 at .600 (19 wrong: 125.7-134.7 s, the keeper's hold under the arc tip carried) / 1.0 at .588;
        bridges to 1.2 s .947 at .422; reach 1.5-3 m .750 at .622, 2-4 m .738 at .678 / .692 at .255; depth capped at
        4 .678 at .656, at 2 (possession-4's rows) .952 at .467 / .917 at .471; kit 'reject' 1.0 at .222 / 1.0 at .176
    forced on broadcast, where it is never dispatched: 0-60 s 1.0 at .898 (possession-4 117 of 117 at .992), 42-222 s
        .974 at .852 (possession-4 .977 at .975)
So where it names a team it was right on every DEV point but one, and it names far less than possession-4 does on
broadcast. What it leaves unknown is mostly the ball (no trusted ball on 65% of amateur 3's frames: the path unseen,
bridged for longer than .5 s, or the first pass on another team): 49 of the 58 unknown team points of amateur 3 and
24 of the 41 of the Veo clip are in 'ball_untrusted' spans, 8 / 16 in 'nobody_at_ball' ones (a carrier not detected,
or a track without a confirmed identity). Coverage follows the ball and the people layers, not this module.
0.4-0.6 s for amateur 3's 1915 frames and 2.5 s for broadcast 42-222 s (5400), possession-4's features included.

What it cannot do: name a team where the carrier is not detected or not confirmed (the Veo clip: about 4 in 5 team
points unknown); a pass of 1.5 s or more breaks possession; a free kick or a foul in midfield is never dead (no line
near the ball); a goalkeeper's team rests on tactical.goalEnds, which is reversed on amateur 3 as stored (no named
frame of the DEV clips rests on a teamless keeper); an opponent pressing at the ball while the carrier is undetected is
named (125.7 s above); the DEV labels are 141 team points on two clips (32 of 32 right has a two-sided 95% lower
bound of .89). The coaching layer uses it only outside coach-marked spans that it agreed with
(soccer/coaching/situations.py image_use).
"""
import math
import re

import numpy as np

from . import possession as P4

VERSION = 'possession-amateur-1'
TEAMS = P4.TEAMS
DISPATCH_REASONS = ('small_scene', 'agreement')

DEFAULTS = {
    # --- dispatch (compute())
    'engine': 'auto',          # 'auto': possession-4, this estimate only where possession-4 declines for a small scene
                               # or the pitch-vs-image agreement; 'possession-4': never this; 'amateur': always this
    # --- the trusted ball
    'max_bridge': .5,          # s: an interpolated ball inside a longer bridge is not trusted
    'fp_agree': 2.0,           # m (reach metric): where the first pass's ball is farther than this from the final one,
                               # both must point to the same team
    # --- the reach
    'reach_r0': 1.0,           # m (body-height normalised): full evidence within this reach ...
    'reach_r1': 2.0,           # ... none from this one
    'max_depth': 25.0,         # the depth factor of the calibration at the ball is capped here (the horizon)
    'row_weight': 10.5,        # image rows weigh this much without a calibration at the ball and no run median, on a
                               # small-scene run ...
    'row_weight_wide': 3.2,    # ... and on any other run
    'row_frames': 30,          # frames with a valid calibration at the ball for the run's own median row weight
    # --- the carrier
    'kit_s': 1.0,              # s: the kit readings of a person's track within this of the frame
    # --- evidence and decoding
    'bridge_w': .5,            # an interpolated (trusted) ball's evidence counts this much
    'switch': .15,             # s of evidence a change costs (possession-4)
    'tau': .05,                # the posterior's temperature (possession-4)
    'min_posterior': .8,       # a team is named where its posterior is at least this (possession-4)
    'nocarry': 1.5,            # s: nobody of either team at a trusted ball this long breaks possession (never carried)
    'gap': 1.5,                # s: no trusted ball this long restarts possession (carried when the same team follows)
    'kit_none': 'accept',      # a confirmed player whose track has no kit reading within kit_s: 'accept' or 'reject'
    'keeper_ends': 'use',      # a teamless goalkeeper plays for the goal on his side by tactical.goalEnds ('use'), or is
                               # nobody's ('ignore': where the ends are known to be wrong)
    # --- dead balls
    'line_near': 10.0,         # m: a dead ball needs a touchline or goal line this near the ball in view
}

LOOKS = re.compile(r'looks ref ([0-9.]+) A ([0-9.]+) B ([0-9.]+)')
STATES = P4.STATE_NAMES
REASONS = ('ball_untrusted', 'carrier_unidentified', 'contested', 'nobody_at_ball', 'registration', 'uncertain', 'cut')


OWN = frozenset(DEFAULTS)-frozenset(P4.DEFAULTS)   # settings only this module reads (the shared names mean the same)


def _settings(settings):
    S = dict(DEFAULTS)
    S.update({k: v for k, v in (settings or {}).items() if k in DEFAULTS})
    return S


def _p4_settings(settings):
    """possession-4's settings: every key but this module's own (None when none is left, as compute(result) passes)."""
    return {k: v for k, v in (settings or {}).items() if k not in OWN} or None


def _shape(Hm, gx, gy, L, Wd, cap):
    """The local shape of the frame's calibration at the ball's ground point (pixels): the Jacobian of image -> pitch
    there, divided by its column scale, so a displacement across the image keeps its length and one along the rows is
    stretched by the depth factor; None where the calibration is missing or does not place the point on the ground in
    front of the camera."""
    if not (isinstance(Hm, (list, tuple)) and len(Hm) == 9 and all(P4._num(v) for v in Hm)):
        return None
    Hh = np.array(Hm, float).reshape(3, 3)
    try:
        M = np.linalg.inv(Hh)
    except np.linalg.LinAlgError:
        return None
    u = np.array([gx, gy, 1.0])
    w = float(M[2]@u)
    if not math.isfinite(w) or abs(w) < 1e-12:
        return None
    X, Y = float(M[0]@u)/w, float(M[1]@u)/w
    if not (-L <= X <= 2*L and -Wd <= Y <= 2*Wd) or not float(Hh[2]@np.array([X, Y, 1.0])) > 0:
        return None
    J = np.array([[(M[0, 0]-X*M[2, 0])/w, (M[0, 1]-X*M[2, 1])/w], [(M[1, 0]-Y*M[2, 0])/w, (M[1, 1]-Y*M[2, 1])/w]])
    c0 = math.hypot(J[0, 0], J[1, 0])
    if not (math.isfinite(c0) and c0 > 0):
        return None
    A = J/c0
    k = math.hypot(A[0, 1], A[1, 1])
    if not math.isfinite(k):
        return None
    if k > cap:
        A[:, 1] *= cap/k
        k = cap
    return A, k


def _reach(A, gx, gy, fx, fy, pxm):
    dx, dy = gx-fx, gy-fy
    return math.hypot(A[0, 0]*dx+A[0, 1]*dy, A[1, 0]*dx+A[1, 1]*dy)/pxm


def features(result, settings=None, F4=None):
    """Per-frame arrays of the image-space estimate (on top of possession-4's features F4): per team the nearest
    eligible carrier's reach to the final ball and to the first pass's ball, the nearest person who may be a player of
    either team (unidentified), the goalkeeper holding the ball, and the depth factor used."""
    S = _settings(settings)
    if F4 is None:
        F4 = P4.features(result, _p4_settings(settings))
    frames = result.get('frames') or []
    video = result.get('video') or {}
    W = float(video['width']) if P4._num(video.get('width')) and video['width'] > 0 else 1920.0
    Hh = float(video['height']) if P4._num(video.get('height')) and video['height'] > 0 else 1080.0
    n, t, L, Wd = F4['n'], F4['t'], F4['length'], F4['width']
    teams_of = P4.roster_teams(result)
    match = result.get('match') or {}
    gk_roster = {e['id'] for e in match.get('goalkeepers') or [] if isinstance(e, dict) and isinstance(e.get('id'), str)
                 and e.get('team') in TEAMS}
    left, right = P4._keeper_team(result) if S['keeper_ends'] == 'use' else (None, None)
    ids, index = [], {}

    def idx(pid):
        if pid not in index:
            index[pid] = len(ids)
            ids.append(pid)
        return index[pid]

    gpx = F4['gpx']
    shapes = [None]*n
    gfp = np.full((n, 2), np.nan)
    rows = []          # (frame, key, id index or -1, team index or -1, kind, fx, fy, box)
    readings = {}      # key -> [(t, A - B kit reading)]
    for i, f in enumerate(frames[:n]):
        b = f.get('ball') or {}
        gx, gy = gpx[i]
        if np.isfinite(gx):
            shapes[i] = _shape((f.get('calibration') or {}).get('H'), gx, gy, L, Wd, S['max_depth'])
            fc = (b.get('firstPass') or {}).get('center')
            if P4._pair(fc):
                gfp[i] = fc[0]*W, (fc[1]+(b['box'][3]/2 if P4._box(b.get('box')) else 0.0))*Hh
        for p in f.get('people') or []:
            if not isinstance(p, dict) or not P4._box(p.get('box')):
                continue
            if p.get('state') in P4.LOOSE_STATES or p.get('role') not in ('PLAYER', 'GOALKEEPER', 'UNKNOWN'):
                continue
            pid = p.get('id') if isinstance(p.get('id'), str) else None
            role = p.get('role')
            team = teams_of.get(pid) or p.get('team')
            keeper = None
            if role == 'GOALKEEPER' and team not in TEAMS and left is not None and P4._pair(p.get('pitch')):
                team = left if p['pitch'][0] < .5 else right
                keeper = 'ends'
            elif role == 'GOALKEEPER' and team in TEAMS:
                keeper = 'roster' if pid in gk_roster else 'frame'
            if team in TEAMS and role == 'UNKNOWN':
                team = None
            key = (p.get('track'), pid)
            m = LOOKS.search(str(p.get('why') or ''))
            if m:
                readings.setdefault(key, []).append((float(t[i]), float(m.group(2))-float(m.group(3))))
            if team in TEAMS and keeper is not None:
                kind = 'keeper'     # a goalkeeper plays for his team even without an identity (PitchIQ gives teamless
                                    # keepers none: id None on every keeper frame of amateur 3 and the Veo clip)
            elif team not in TEAMS or pid is None:
                kind = 'no_team'
            elif role == 'PLAYER' and p.get('state') == 'confirmed':
                kind = 'player'
            else:
                kind = 'unconfirmed'
            bx, by, bw, bh = p['box']
            rows.append((i, key, idx(pid) if pid is not None and team in TEAMS else -1, TEAMS.index(team) if team in TEAMS else -1,
                         kind, (bx+bw/2)*W, (by+bh)*Hh, (bx*W, by*Hh, bw*W, bh*Hh)))
    # the kit readings of each track within kit_s of each frame: their mean (A - B) must favour the identity's team
    kit = {}
    for key, rs in readings.items():
        rs.sort()
        kit[key] = (np.array([r[0] for r in rs]), np.concatenate([[0.0], np.cumsum([r[1] for r in rs])]))
    valid = [s[1] for s in shapes if s is not None]
    small = n > 0 and float(F4['small'].mean()) >= P4.DEFAULTS['small_run']
    row_w = float(np.median(valid)) if len(valid) >= S['row_frames'] else S['row_weight' if small else 'row_weight_wide']
    fallback = (np.array([[1.0, 0.0], [0.0, row_w]]), row_w)
    depth = np.full(n, np.nan)
    calibrated = np.zeros(n, bool)
    for i in range(n):
        if shapes[i] is not None:
            calibrated[i] = True
        elif np.isfinite(gpx[i, 0]):
            shapes[i] = fallback
        if shapes[i] is not None:
            depth[i] = shapes[i][1]
    dE = np.full((2, n), np.inf)
    jE = np.full((2, n), -1, np.int64)
    dF = np.full((2, n), np.inf)          # the same at the first pass's ball
    dU = np.full(n, np.inf)               # the nearest person who may be a player of either team
    dUF = np.full(n, np.inf)
    uKind = np.full(n, '', dtype='<U12')  # why that person is not an eligible carrier
    kH = np.full(n, -1, np.int8)
    kJ = np.full(n, -1, np.int64)
    centre = np.full((n, 2), np.nan)
    for i, f in enumerate(frames[:n]):
        c = (f.get('ball') or {}).get('center')
        if P4._pair(c):
            centre[i] = c[0]*W, c[1]*Hh
    kit_frames = {'agree': 0, 'contradict': 0, 'none': 0}
    for i, key, j, k, kind, fx, fy, box in rows:
        sh = shapes[i]
        if sh is None:
            continue
        A = sh[0]
        pxm = float(F4['bpxm'][i])
        gx, gy = gpx[i]
        d = _reach(A, gx, gy, fx, fy, pxm)
        dfp = _reach(A, gfp[i, 0], gfp[i, 1], fx, fy, pxm) if np.isfinite(gfp[i, 0]) else math.inf
        eligible = kind == 'keeper'
        why = kind
        if kind == 'player':
            ts, cs = kit.get(key, (None, None))
            if ts is None:
                why = 'no_kit'
                eligible = S['kit_none'] == 'accept'
            else:
                a = int(np.searchsorted(ts, t[i]-S['kit_s']-1e-9))
                b = int(np.searchsorted(ts, t[i]+S['kit_s']+1e-9, side='right'))
                if b <= a:
                    why = 'no_kit'
                    eligible = S['kit_none'] == 'accept'
                else:
                    mean = (cs[b]-cs[a])/(b-a)
                    eligible = mean > 0 if k == 0 else mean < 0
                    why = 'player' if eligible else 'kit'
            kit_frames[{'player': 'agree', 'kit': 'contradict'}.get(why, 'none')] += 1
        if eligible:
            if d < dE[k, i]:
                dE[k, i], jE[k, i] = d, j
            if dfp < dF[k, i]:
                dF[k, i] = dfp
            bx, by, bw, bh = box
            if (kind == 'keeper' and kH[i] < 0 and np.isfinite(centre[i, 0]) and bw > 0 and bh > 0
                    and P4.HANDS_U[0] <= (centre[i, 0]-bx)/bw <= P4.HANDS_U[1]
                    and P4.HANDS_V[0] <= (centre[i, 1]-by)/bh < P4.HANDS_V[1]):
                kH[i], kJ[i] = k, j
        else:
            if d < dU[i]:
                dU[i], uKind[i] = d, why
            dUF[i] = min(dUF[i], dfp)
    dfp_ball = np.full(n, np.nan)
    for i in range(n):
        if shapes[i] is not None and np.isfinite(gfp[i, 0]):
            dfp_ball[i] = _reach(shapes[i][0], gpx[i, 0], gpx[i, 1], gfp[i, 0], gfp[i, 1], float(F4['bpxm'][i]))
    return {'F4': F4, 'n': n, 't': t, 'frame': F4['frame'], 'fps': F4['fps'], 'ids': ids, 'dE': dE, 'jE': jE, 'dF': dF,
            'dU': dU, 'dUF': dUF, 'uKind': uKind, 'kH': kH, 'kJ': kJ, 'depth': depth, 'calibrated': calibrated,
            'rowWeight': row_w, 'dFP': dfp_ball, 'kitFrames': kit_frames,
            'view': [f.get('view') for f in frames[:n]]}


def _prox(d, S):
    return np.clip((S['reach_r1']-d)/(S['reach_r1']-S['reach_r0']), 0.0, 1.0)


def _points(dA, dB, dU, r1):
    """The team a ball position points to: one team's eligible carrier within r1 and nobody else (-1: none)."""
    a, b, u = dA < r1, dB < r1, dU < r1
    return np.where(a & ~b & ~u, 0, np.where(b & ~a & ~u, 1, -1))


def trusted_ball(G, S):
    """(placed, trusted): frames whose ball has an image position (TRACKED, or interpolated inside a bridge of at most
    max_bridge), and of those the frames where the final path and the ball tracker's first pass agree (within fp_agree
    in the reach metric) or both point to the same team."""
    F4 = G['F4']
    st = F4['bstate']
    bridge = P4._bridge_lengths(F4)
    placed = F4['bimg'] & np.isfinite(G['depth']) & ((st == 1) | ((st == 2) & (bridge <= S['max_bridge']+1e-9)))
    with np.errstate(invalid='ignore'):
        apart = G['dFP'] > S['fp_agree']
    final = _points(G['dE'][0], G['dE'][1], G['dU'], S['reach_r1'])
    first = _points(G['dF'][0], G['dF'][1], G['dUF'], S['reach_r1'])
    return placed, placed & (~apart | ((final >= 0) & (final == first)))


def keeper_hands(G):
    out = np.full(G['n'], -1, np.int8)
    for k in range(2):
        for a, b in P4._same_runs(G['kH'] == k, G['kJ']):
            if b-a+1 >= P4.HANDS_MIN:
                out[a:b+1] = k
    return out


def _in_polygon(poly, x, y):
    inside = False
    m = len(poly)
    for a in range(m):
        x0, y0 = poly[a]
        x1, y1 = poly[(a+1) % m]
        if (y0 > y) != (y1 > y) and x < x0+(y-y0)*(x1-x0)/(y1-y0):
            inside = not inside
    return inside


def line_in_view(G, S, i):
    """Is a touchline or goal line within line_near of the ball in the camera's view on frame i (frames[].view, the
    view's footprint in normalised pitch coordinates; the ball's pitch position whatever its confidence)? None where
    the frame cannot tell (no view, no ball pitch position)."""
    F4 = G['F4']
    v = G['view'][i]
    bx, by = F4['bx'][i], F4['by'][i]
    if not (isinstance(v, (list, tuple)) and len(v) >= 3 and all(P4._pair(q) for q in v)) or not (np.isfinite(bx) and np.isfinite(by)):
        return None
    L, Wd = F4['length'], F4['width']
    x, y = min(max(bx, 0.0), L), min(max(by, 0.0), Wd)
    for dist, px, py in ((by, x, 0.0), (Wd-by, x, Wd), (bx, 0.0, y), (L-bx, L, y)):
        if dist <= S['line_near'] and _in_polygon(v, px/L, py/Wd):
            return True
    return False


def decide(G, settings=None):
    """The image-space decisions on features G: per frame the evidence, the team code (0 dead, 1 A, 2 B, -1 unknown),
    the posterior, the abstention reason of every unknown frame, the boundaries and the changes."""
    S = _settings(settings)
    S4 = dict(P4.DEFAULTS)
    S4.update(_p4_settings(settings) or {})
    F4 = G['F4']
    n, t, fps = G['n'], G['t'], G['fps']
    dt = 1.0/fps
    st = F4['bstate']
    placed, trusted = trusted_ball(G, S)
    # possession-4's checks of a ball path on something else: an excursion beyond a line the path jumped to (or with
    # another ball in play), a replay between two camera cuts; those frames show no ball
    usable4 = P4.usable_ball(F4, S4)[0]
    bad = P4.excursions(F4, S4, usable4) if S4['out_consistent'] else np.zeros(n, bool)
    rep = P4.replays(F4, S4, usable4 & ~bad) if S4['replays'] else np.zeros(n, bool)
    placed, trusted = placed & ~bad & ~rep, trusted & ~bad & ~rep
    hands = keeper_hands(G)
    p = np.stack([_prox(G['dE'][0], S), _prox(G['dE'][1], S)])
    pu = _prox(G['dU'], S)
    who = np.where(p > 0, G['jE'], -1)
    for k in range(2):
        m = hands == k
        p[k] = np.where(m, 1.0, np.where(hands == 1-k, 0.0, p[k]))
        who[k] = np.where(m, G['kJ'], np.where(hands == 1-k, -1, who[k]))
    pu = np.where(hands >= 0, 0.0, pu)
    fast = (F4['bmotion'] == P4.MOTIONS['FAST']) & (hands < 0)
    use = trusted & ~fast
    w = np.where(st == 2, S['bridge_w'], 1.0)
    e = np.zeros((2, n))
    for k in range(2):
        e[k] = np.where(use, w*p[k]*(1.0-p[1-k])*(1.0-pu)*dt, 0.0)
    who = np.where(e > 0, who, -1)
    has = e > 0
    borig = F4['borig']
    tracked = st == 1
    no_origin = bool(tracked.any()) and not (borig[tracked] > 0).any()
    observed = tracked & ((borig == P4.ORIGINS['OBSERVATION']) if not no_origin else True)
    ok = P4.confirmed_runs(has, who, has & observed[None, :])
    # boundaries: nobody of either team at a trusted ball (or no trusted ball) for nocarry seconds, camera cuts, and
    # possession-4's dead-ball spells on this evidence, kept only where a line near the ball is in view
    none = ~(has[0] | has[1])
    bounds = []
    for a, b in P4._runs(none):
        if float(trusted[a:b+1].sum())*dt >= S['nocarry']-1e-9:
            kind = 'nocarry'    # a trusted ball nobody of either team was at: never carried across
        elif t[b]-t[a]+dt >= S['gap']-1e-9:
            kind = 'gap'        # the ball unseen or untrusted: carried when the same team is on both sides
        else:
            continue
        bounds.append({'a': int(a), 'r': int(b+1), 'kind': kind, 'reason': None, 'cut': False})
    cuts = np.flatnonzero(F4['cut']) if F4.get('cut') is not None else np.zeros(0, np.int64)
    bounds += [{'a': int(c), 'r': int(c)+1, 'kind': 'gap', 'reason': None, 'cut': True} for c in cuts]
    act = P4.activity(F4, S4) if math.isfinite(S4['slow_s']) else None
    dead = [b for b in P4.boundaries(F4, S4, usable4 & placed, placed, e, act, bad, has & tracked[None, :], None, who)
            if b['kind'] == 'dead']
    for b in dead:
        seen = [line_in_view(G, S, i) for i in range(b['a'], min(b['r'], n))]
        told = [v for v in seen if v is not None]
        b['verified'] = bool(told) and sum(told) >= .5*len(told)
        b['kind'] = 'dead' if b['verified'] else 'unverified'
        b['cut'] = bool(F4['cut'][b['a']:b['r']].any())
    spells = sorted(bounds+dead, key=lambda b: (b['a'], -b['r']))
    merged = []
    rank = {'dead': 3, 'unverified': 2, 'nocarry': 1, 'gap': 0}
    for b in spells:
        if merged and b['a'] <= merged[-1]['r']:
            m = merged[-1]
            m['r'] = max(m['r'], b['r'])
            m['cut'] = m['cut'] or b['cut']
            if rank[b['kind']] > rank[m['kind']]:
                m.update(kind=b['kind'], reason=b['reason'])
        else:
            merged.append(dict(b))
    raw = np.full(n, -1, np.int8)
    post = np.zeros(n)
    carried = np.zeros(n, bool)
    P_prev = None
    edges = [None]+merged+[None]
    for bi in range(len(edges)-1):
        b0, b1 = edges[bi], edges[bi+1]
        s0 = b0['r'] if b0 is not None else 0
        s1 = b1['a'] if b1 is not None else n
        if s0 >= n or s1 <= s0:
            continue
        stt = P4.viterbi(e, S['switch'], s0, s1, ok)
        pa = P4.posterior(e, S['switch'], s0, s1, S['tau'])
        post[s0:s1] = np.where(stt == 0, pa, 1.0-pa)
        ev = np.flatnonzero(has[0, s0:s1] | has[1, s0:s1])
        if b0 is not None and b0['cut']:
            ev = np.flatnonzero(ok[0, s0:s1] | ok[1, s0:s1])   # after a cut: from an observed control on
        if not len(ev):
            continue
        first = s0+int(ev[0])
        P0 = int(stt[first-s0])
        if b0 is not None and b0['kind'] == 'gap' and not b0['cut'] and P_prev is not None and P0 == P_prev:
            raw[b0['a']:b0['r']] = P0+1        # the same team on both sides of the stretch nobody was at: carried
            post[b0['a']:b0['r']] = S['min_posterior']
            carried[b0['a']:b0['r']] = True
        raw[first:s1] = stt[first-s0:]+1
        P_prev = int(stt[-1])
    for b in merged:
        if b['kind'] == 'dead':
            raw[b['a']:b['r']] = 0
        elif b['kind'] == 'unverified':
            raw[b['a']:b['r']] = -1
            carried[b['a']:b['r']] = False
    code = np.where((raw > 0) & (post < S['min_posterior']) & ~carried, -1, raw).astype(np.int8)
    # why each unknown frame is unknown (one reason per frame; a span takes its commonest)
    why = np.full(n, '', dtype='<U20')
    unk = code == -1
    contested = (p[0] > 0) & (p[1] > 0)
    why[unk] = 'uncertain'
    why[unk & trusted & ~has[0] & ~has[1] & (pu == 0) & ~contested] = 'nobody_at_ball'
    why[unk & trusted & (pu > 0) & ~has[0] & ~has[1] & ~contested] = 'carrier_unidentified'
    why[unk & trusted & contested] = 'contested'
    why[unk & ~trusted] = 'ball_untrusted'
    for b in merged:
        if b['kind'] == 'unverified':
            why[b['a']:b['r']] = 'registration'
        if b['cut']:
            m = np.zeros(n, bool)
            m[b['a']:b['r']] = True
            why[m & unk & (why == 'uncertain')] = 'cut'
    return {'S': S, 'e': e, 'p': p, 'pu': pu, 'who': who, 'ok': ok, 'placed': placed, 'trusted': trusted, 'hands': hands,
            'raw': raw, 'code': code, 'post': post, 'carried': carried, 'bounds': merged, 'why': why,
            'contested': contested, 'noOrigin': no_origin}


# ---------------------------------------------------------------------------------------------- the record

CARRIER_MIN = P4.CARRIER_MIN
REASON_MIN_S = .5      # s: an unknown span's piece with another reason shorter than this keeps its neighbour's
WON_W = .5             # s after the first named frame of a new team over which wonBy is the most evidence
RESTART_LOOK = 2.0     # s after a dead spell within which the first named team is its restart team


def frame_states(G, D):
    """Per frame: the state (an index of STATE_NAMES), the carrier (id index, -1), estimated, the confidence level
    (0 low, 1 medium: never high)."""
    n = G['n']
    code, e, p, who = D['code'], D['e'], D['p'], D['who']
    state = np.full(n, STATES.index('unknown'), np.int8)
    carrier = np.full(n, -1, np.int64)
    for i in range(n):
        if code[i] <= 0:
            state[i] = STATES.index('dead' if code[i] == 0 else 'unknown')
            continue
        k = code[i]-1
        if p[k, i] > 0 and p[1-k, i] > 0:
            state[i] = STATES.index('contested')
        elif e[k, i] > 0:
            state[i] = STATES.index('control')
            carrier[i] = int(who[k, i])
        elif e[1-k, i] > 0 or p[1-k, i] > 0:
            state[i] = STATES.index('loose')
        else:
            state[i] = STATES.index('flight')
    for a, b in P4._runs(state == STATES.index('control')):
        seq = carrier[a:b+1]
        prev, s = None, 0
        for j in range(1, len(seq)+1):
            if j == len(seq) or seq[j] != seq[s]:
                if j-s < CARRIER_MIN and prev is not None:
                    seq[s:j] = prev
                else:
                    prev = seq[s]
                s = j
    estimated = (G['F4']['bstate'] == 2) | D['carried']
    level = np.where((code > 0) & ~D['carried'], 1, 0)
    level[code == 0] = 1
    return state, carrier, estimated, level


def span_reasons(D, fps):
    """The abstention reason of every unknown frame, per span: an unknown stretch is cut where its frames' reason
    changes, and a piece shorter than REASON_MIN_S takes the reason of the piece before it (of the one after it at the
    stretch's start), so a span of the timeline says why in one code."""
    code, why = D['code'], D['why']
    out = np.full(len(code), '', dtype='<U20')
    least = max(1, int(round(REASON_MIN_S*fps)))
    for a, b in P4._runs(code == -1):
        parts = [[a+s0, a+s1, str(why[a+s0])] for s0, s1 in P4._same_runs(np.ones(b-a+1, bool), why[a:b+1])]
        keep = [p for p in parts if p[1]-p[0]+1 >= least] or [max(parts, key=lambda p: p[1]-p[0])]
        for p in parts:
            if p in keep:
                continue
            before = [k for k in keep if k[1] < p[0]]
            p[2] = before[-1][2] if before else next(k for k in keep if k[0] > p[1])[2]
        for s0, s1, r in parts:
            out[s0:s1+1] = r
    return out


def record(G, D):
    """The timeline runs, changes, dead spells, shares and coverage of decisions D on features G (JSON-safe; the
    possession-4 record's shape, unknown runs with their abstention reason in 'why')."""
    n, t, fr, fps, ids = G['n'], G['t'], G['frame'], G['fps'], G['ids']
    code, e, who = D['code'], D['e'], D['who']

    def pid(j):
        return ids[j] if j is not None and j >= 0 and ids[j] is not None else None

    state, carrier, estimated, level = frame_states(G, D)
    reason = span_reasons(D, fps)
    tm = np.where(code > 0, code-1, -1)
    runs = []
    i = 0
    while i < n:
        j = i
        while (j+1 < n and tm[j+1] == tm[i] and state[j+1] == state[i] and carrier[j+1] == carrier[i]
               and reason[j+1] == reason[i]):
            j += 1
        run = {'t0': round(float(t[i]), 3), 't1': round(float(t[j]), 3), 'f0': int(fr[i]), 'f1': int(fr[j]),
               'team': TEAMS[tm[i]] if tm[i] >= 0 else None, 'state': STATES[state[i]], 'carrier': pid(int(carrier[i])),
               'estimated': bool(estimated[i:j+1].mean() >= .5), 'confidence': P4.LEVELS[int(np.median(level[i:j+1]))]}
        if reason[i]:
            run['why'] = str(reason[i])
        runs.append(run)
        i = j+1
    # changes: the boundaries of the named spans (a run of one team's named frames, unknown and dead frames between
    # two of them left out), never a moment: lostAt is the last named frame of the old team, wonAt the first of the new
    spans = [(int(code[a])-1, a, b) for a, b in P4._same_runs(code > 0, code)]
    dead = [b for b in D['bounds'] if b['kind'] == 'dead']
    changes = []
    for (k0, a0, b0), (k1, a1, b1) in zip(spans, spans[1:]):
        if k0 == k1:
            continue
        last = [i for i in range(b0, a0-1, -1) if e[k0, i] > 0]
        lost_by = pid(int(who[k0, last[0]])) if last else None
        hi = min(b1+1, a1+int(round(WON_W*fps)))
        wsum = {}
        for i in range(a1, hi):
            j = int(who[k1, i])
            if j >= 0 and e[k1, i] > 0:
                wsum[j] = wsum.get(j, 0.0)+float(e[k1, i])
        if not wsum:
            for i in range(a1, b1+1):
                if e[k1, i] > 0 and who[k1, i] >= 0:
                    wsum[int(who[k1, i])] = 1.0
                    break
        won_by = pid(max(wsum, key=wsum.get)) if wsum else None
        between = [d for d in dead if d['a'] >= b0 and d['a'] <= a1]
        cause = 'out_of_play' if any(d.get('reason') == 'out_of_play' for d in between) else 'unclear'
        changes.append({'id': f"pc-{int(fr[a1])}-{TEAMS[k1]}", 'lostAt': round(float(t[b0]), 3), 'wonAt': round(float(t[a1]), 3),
                        'frame': int(fr[a1]), 'fromTeam': TEAMS[k0], 'toTeam': TEAMS[k1], 'lostBy': lost_by, 'wonBy': won_by,
                        'cause': cause, 'confidence': 'low'})
    spells = []
    for b in dead:
        r = min(b['r'], n)
        hi = int(np.searchsorted(t, t[min(r, n-1)]+RESTART_LOOK, side='right'))
        nxt = np.flatnonzero(code[r:hi] > 0) if r < n else []
        spells.append({'t0': round(float(t[b['a']]), 3), 't1': round(float(t[r-1]), 3), 'reason': b.get('reason') or 'stoppage',
                       'restartTeam': TEAMS[int(code[r+int(nxt[0])])-1] if len(nxt) else None})
    dt = 1.0/fps
    shares = {'A': round(float((code == 1).sum()*dt), 2), 'B': round(float((code == 2).sum()*dt), 2),
              'dead': round(float((code == 0).sum()*dt), 2), 'unknown': round(float((code == -1).sum()*dt), 2)}
    live = int((code != 0).sum())
    abstained = {}
    for r_ in REASONS:
        s = float(((code == -1) & (reason == r_)).sum()*dt)
        if s > 0:
            abstained[r_] = round(s, 2)
    return {'runs': runs, 'changes': changes, 'deadSpells': spells, 'shares': shares, 'abstained': abstained,
            'coverage': round(float((code > 0).sum()/live), 3) if live else 0.0}


def dispatch_reason(F4, settings=None):
    """Why possession-4 declines a result, in from_features' order: 'no_frames', 'small_scene', 'ball' (placed on
    too few frames), 'players' (no team player near the ball), 'agreement' (pitch vs image), or None (it judges it)."""
    S4 = dict(P4.DEFAULTS)
    S4.update(settings or {})
    n = F4['n']
    if not n:
        return 'no_frames'
    usable, seen = P4.usable_ball(F4, S4)
    if float(F4['small'].mean()) >= S4['small_run']:
        return 'small_scene'
    if float((usable | seen).mean()) < S4['min_usable']:
        return 'ball'
    if not ((F4['iP'] >= 0).any() or (F4['iI'] >= 0).any()):
        return 'players'
    agree, m = P4.agreement(F4, S4, usable)
    if agree is not None and m >= S4['agree_frames'] and agree < S4['agree_min']:
        return 'agreement'
    return None


REASON_TEXT = {'ball_untrusted': 'the ball was not placed or not trusted', 'carrier_unidentified': 'the person at the ball had no confirmed team',
               'contested': 'both teams were at the ball', 'nobody_at_ball': 'nobody was at the ball',
               'registration': 'a dead ball could not be checked against a line in view', 'uncertain': 'the estimate was unsure',
               'cut': 'the camera cut'}


def estimate(result, settings=None, F4=None, fallback=None):
    """The image-space estimate of a result (always, whatever possession-4 says): possession-4's record shape with
    version possession-amateur-1. fallback: possession-4's record of the same result (its reason is quoted)."""
    s4 = _p4_settings(settings)
    if F4 is None:
        F4 = P4.features(result, s4)
    n, fps = F4['n'], F4['fps']
    out = {'version': VERSION, 'fps': fps, 't0': round(float(F4['t'][0]), 3) if n else 0.0,
           't1': round(float(F4['t'][-1]), 3) if n else 0.0, 'basis': 'image'}
    if fallback is not None:
        out['fallbackFrom'] = {'version': fallback.get('version'), 'reason': fallback.get('reason'),
                               'gate': dispatch_reason(F4, s4)}
    G = features(result, settings, F4) if n else None
    D = decide(G, settings) if n else None
    rec = record(G, D) if n else None
    if not n or not ((D['code'] > 0).any() or rec['deadSpells']):
        reason = 'The result has no frames.' if not n else (
            (str(fallback['reason'])+' ' if fallback is not None and fallback.get('reason') else '') +
            'The image-based estimate could not name a team on any frame either.')
        dur = round(float(F4['t'][-1]-F4['t'][0]+1.0/fps), 2) if n else 0.0
        out.update(available=False, reason=reason, quality={'level': 'unavailable', 'coverage': 0.0, 'notes': [reason]},
                   shares={'A': 0.0, 'B': 0.0, 'dead': 0.0, 'unknown': dur},
                   timeline=[{'t0': out['t0'], 't1': out['t1'], 'f0': int(F4['frame'][0]), 'f1': int(F4['frame'][-1]), 'team': None,
                              'state': 'unknown', 'carrier': None, 'estimated': False, 'confidence': 'low'}] if n else [],
                   changes=[], deadSpells=[])
        return out
    notes = ['An image-based estimate (possession-amateur-1): who had the ball is judged from the players nearest the ball in '
             'the image, never with high confidence, and every change is a boundary between two named spans, not a moment.']
    if fallback is not None and fallback.get('reason'):
        notes.append('possession-4 does not judge this footage: ' + str(fallback['reason']))
    top = sorted(rec['abstained'].items(), key=lambda kv: -kv[1])[:3]
    if top:
        notes.append('No team is named where ' + '; '.join(f'{REASON_TEXT[k]} ({v:.1f} s)' for k, v in top) + '.')
    if D['noOrigin']:
        notes.append('The result does not record where each ball position came from: every tracked ball is taken as observed.')
    out.update(available=True, reason=None, quality={'level': 'limited', 'coverage': rec['coverage'], 'notes': notes,
                                                     'abstained': rec['abstained']},
               shares=rec['shares'], timeline=rec['runs'], changes=rec['changes'], deadSpells=rec['deadSpells'])
    return out


def compute(result, settings=None):
    """The possession record of a saved result: possession-4's (soccer/possession.py, unchanged and byte for byte the
    same) wherever it judges the footage, else, where it declines for a small scene or the pitch-vs-image agreement,
    this image-based estimate (settings 'engine': 'possession-4' never gives it, 'amateur' always)."""
    S = _settings(settings)
    s4 = _p4_settings(settings)
    F4 = P4.features(result, s4)
    rec = P4.from_features(F4, s4)
    if S['engine'] == 'possession-4':
        return rec
    if S['engine'] != 'amateur' and (rec['available'] or dispatch_reason(F4, s4) not in DISPATCH_REASONS):
        return rec
    return estimate(result, settings, F4=F4, fallback=rec)
