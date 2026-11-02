"""Possession v2: which team controls the ball, frame by frame, over a whole match (or one analysed segment), through
the frames where the ball is not seen, with the uncertainty kept. Nothing here runs a model or reads the video; it
reads what the tracking saved.

Four things are kept apart (the spec's separation):
    BALL OBSERVATION   what the ball tracker saw on a frame: ball_states() (OBSERVED, OCCLUDED, PREDICTED, REACQUIRING,
                       UNKNOWN), with its provenance and confidence. Read, never changed.
    BALL POSITION      the tracker's own position (observed, or interpolated across a short gap). Possession never
                       writes one: a ball hidden at a player is not drawn at his feet.
    PLAYER CONTROL     the player most likely on the ball (possessor()), a separate track with its own confidence; the
                       team can be known while the player is not.
    TEAM POSSESSION    the decoded state of the frame: A, B, contested, free ball, unknown, or excluded by the game state
                       (a confirmed stoppage, halftime, footage that is not the match, a possible stoppage).

Definition (CONTROLLED IN-PLAY POSSESSION, the analytics' default, written into every record as `definition`): a team
has the ball while one of its players controls it and while the ball travels on its pass to a team-mate who controls it
next. A duel nobody controls is CONTESTED; a ball in play that nobody controls and that is not a completed pass (a
clearance, a shot, a deflection, a loose ball, an intercepted pass before the gain) is FREE. A team's share is its
resolved controlled time over both teams' resolved controlled time; contested, free and unresolved time are shown
beside it and never handed to a team to make the shares add up.

The model. Every frame gets soft memberships of a few OBSERVATION CLASSES (classes()):
    OBS_A / OBS_B      a seen ball (TRACKED, an OBSERVATION or a weak detection) within control reach of exactly one
                       team's player (the pitch distance with the ball's own uncertainty, checked in the image: the less
                       near of the two metrics, possession-4's rule; the keeper with the ball in his hands)
    OBS_DUEL           a seen ball within reach of both teams' players
    OBS_OPEN           a seen ball nobody is at (in flight or loose); a FAST ball counts as at a player only where its
                       path bends (a touch), so a pass flying past a defender is OPEN; OBS_OPEN_LONG once nobody has been
                       at it for open_long seconds (a completed pass is rarely in the air that long; a clearance is)
    BR_*               the same for a ball interpolated across a short gap (MISSING), at its interpolated position, the
                       player whose body hid it (nearTrack) counting as at it
    GAP_k_FREE / GAP_k_PRESS   the ball not placed at all, the last player of team k seen on it visible with no opponent
                       near him / with one at him (the situation, not a fixed timeout, sets how fast the belief fades)
    GAP_FLIGHT         the ball not placed, last seen flying with nobody at it
    NONE               nothing to go on
A six-state hidden Markov model reads them: A, B, and contested and free each remembering the team that had the ball
before (contested_A / free_A: a duel or a loose ball after A's control), so a loose ball ends with the other team's gain
and a pass between team-mates never passes through 'free'. It reads them through each class's probability under each state
(EMISSION, team-symmetric), the rates of change between the states (GENERATOR, per second) and a tempering exponent
(GAMMA: 25 frames a second are not 25 independent looks), fitted on Metrica's game 3 through tools/possession_sim.py's
camera (tools/possession_v2_bench.py fits and reports them). forward-backward gives every frame the probability of each
state using the whole sequence, past and future (a ball reacquired at a team-mate's feet resolves the gap before it);
Viterbi gives the single most likely sequence. The sequence runs over the whole match: segment boundaries are not
breaks; confirmed stoppages, halftime and footage that is not the match are (the next sequence starts from the restart,
the restarting team a prior); a camera cut restarts the belief. A frame is named a team where that team's probability
is at least min_posterior, contested or free where the two together are and the team ones are not; else unknown.
OBSERVED control is a team frame whose ball was seen at that team's player on that frame (OBS class); every other team
frame is INFERRED (a hidden ball, a pass in flight, a gap carried by the sequence).

Output (compute() / decode_table()): per-frame codes (CODES), the team posterior, the designation and the possessor;
intervals() runs of constant code; stats() the shares, the coverage and the possession sequences of any time range.
"""
import math
from collections import Counter

import numpy as np

from . import pitch_model

VERSION = 'possession-v2-1'
LENGTH, WIDTH = 105.0, 68.0
TEAMS = ('A', 'B')
STATE_NAMES = ('A', 'B', 'contested_A', 'contested_B', 'free_A', 'free_B')
A, B, CA, CB, FA, FB = 0, 1, 2, 3, 4, 5
NS = 6

# per-frame codes (decode_table / stats / intervals)
CODES = ('A', 'B', 'contested', 'free', 'unknown', 'possible_stoppage', 'stoppage', 'halftime', 'non_match')
LIVE_CODES = (0, 1, 2, 3, 4)
POSSIBLE, STOPPAGE, HALFTIME, NON_MATCH = 5, 6, 7, 8

DEFINITION = ('Controlled in-play possession: a team has the ball while one of its players controls it and while the '
              'ball travels on its pass to a team-mate who controls it next. Contested (a duel nobody controls), free '
              '(in play, nobody controls it: clearances, deflections, loose balls, intercepted passes) and unresolved '
              'time are counted apart. The team shares are each team\'s resolved controlled time over both teams\' '
              'resolved controlled time, in play only: confirmed stoppages, halftime and footage that is not the match '
              'are left out, possible stoppages are shown apart.')

# observation classes
CLASSES = ('NONE', 'OBS_A', 'OBS_B', 'OBS_DUEL', 'OBS_OPEN', 'OBS_OPEN_LONG', 'BR_A', 'BR_B', 'BR_DUEL', 'BR_OPEN',
           'BR_OPEN_LONG', 'GAP_A_FREE', 'GAP_A_PRESS', 'GAP_B_FREE', 'GAP_B_PRESS', 'GAP_FLIGHT')
K = len(CLASSES)
CI = {c: i for i, c in enumerate(CLASSES)}
MIRROR_CLASS = np.array([CI[c.replace('_A', '_X').replace('_B', '_A').replace('_X', '_B')] for c in CLASSES])
MIRROR_STATE = np.array([B, A, CB, CA, FB, FA])

# ball observation states (ball_states)
BALL_STATES = ('OBSERVED', 'OCCLUDED', 'PREDICTED', 'REACQUIRING', 'UNKNOWN')
PROVENANCE = ('detector', 'weak_detection', 'interpolation', 'none')

DEFAULTS = {
    # --- control reach (the evidence)
    'r_ctrl': 1.0,          # m: a ball within this of a player's feet is at him (Vidal-Codina et al.'s possession zone is
                            # .5-1 m; possession-4 counts .7 m fully and nothing from 1.5 m) ...
    'r_soft': .25,          # ... softly (logistic width, m) ...
    'sigma_w': .5,          # ... widened by this share of the ball's own pitch uncertainty (pitchSigma, max axis)
    'bridge_extra': .3,     # m added for a ball interpolated across a gap
    'min_conf': pitch_model.MEDIUM,   # a pitch position (ball or player) below this confidence is not used
    'max_sigma': 3.0,       # m: a ball pitch position less certain than this is not used
    'image_check': True,    # the image reach (depth-aware, possession_amateur's metric) must agree: the less near counts
    'image_only_w': .5,     # memberships of a frame judged on the image alone (no usable pitch position)
    'weak_w': .6,           # memberships of a weak detection
    'fast': 8.0,            # m/s: a ball this fast is in flight ...
    'kink': 6.0,            # ... unless its path bends by this much (m/s, over +-kink_w frames): a touch
    'kink_w': 4,
    'touch_r': 2,           # frames either side of a bend peak that count as the touch
    'long_bridge': 1.2,     # s: a longer interpolation is not used as a position
    # --- the ball not placed: the situation around the last player seen on it
    'gap_age': 4.0,         # s: the last player on the ball counts this long after he was last seen on it
    'free_r': 2.5,          # m: no opponent within this of him: unpressed ...
    'press_r': 1.5,         # ... one within this: pressed (softly between)
    'flight_age': 2.0,      # s: a ball last seen flying with nobody at it is still in flight this long
    'open_long': 1.0,       # s: a ball nobody has been at for this long is a long flight or a loose ball (the _LONG classes)
    # --- decoding
    'min_posterior': .7,    # a frame is named where its state (team, or contested and free together) is this sure: the
                            # lowest of .5 / .6 / .7 / .8 whose CAN-ISL dev team accuracy is .92 or more with at most 5% of
                            # the named frames the other team's AND at which the spec's own cases pass (tests/
                            # test_possession_v2.py: at .5 a duel and an 8 s hidden ball were named a team, at .6 one case
                            # failed): dev .932 at .615 coverage (.8: .938 at .50; the old record .944 at .562); scratch
                            # possv2/PREDECLARED.md addenda 4 and 4b
    'cut_reset': True,      # a camera cut restarts the belief (a replay or another camera: no carry across it)
    'max_dt': 1.0,          # s: a longer jump between consecutive frames is a break
    # --- the possessor
    'carry_s': 2.0,         # s: an inferred frame takes the player seen on the ball this recently (same team, same run)
}

# ----- the fitted model (tools/possession_v2_bench.py --fit; see its report for the data and the fit) -----------------
# EMISSION[s, k] = P(class k | state s), rows sum to 1, team-symmetric (EMISSION[B] is EMISSION[A] mirrored): soft counts
# of the classes under the truth state, mirrored and pooled, shrunk toward the pooled class frequencies by .5 (a class rare
# in the simulation must not make a state impossible). Fitted 2026-10-31 by tools/possession_v2_bench.py --fit on Metrica
# game 3's first half through possession_sim's camera, ball variant 'tactical_gaps' (seed 0, 25 fps: the ball tracked on
# .368 of live frames, bridged on .545, unplaced on .087; CAN-ISL's live play .461 / .447 / .092).
EMISSION = [
    [0.01613, 0.09407, 0.02089, 0.00701, 0.12572, 0.07863, 0.19150, 0.04339, 0.02404, 0.18296, 0.10811, 0.03974, 0.00507, 0.00834, 0.00199, 0.05242],
    [0.01613, 0.02089, 0.09407, 0.00701, 0.12572, 0.07863, 0.04339, 0.19150, 0.02404, 0.18296, 0.10811, 0.00834, 0.00199, 0.03974, 0.00507, 0.05242],
    [0.03118, 0.07372, 0.04846, 0.04949, 0.08832, 0.02876, 0.12745, 0.09667, 0.11414, 0.12189, 0.12278, 0.00778, 0.00619, 0.00759, 0.00113, 0.07444],
    [0.03118, 0.04846, 0.07372, 0.04949, 0.08832, 0.02876, 0.09667, 0.12745, 0.11414, 0.12189, 0.12278, 0.00759, 0.00113, 0.00778, 0.00619, 0.07444],
    [0.01781, 0.03003, 0.02580, 0.00370, 0.13839, 0.16128, 0.06530, 0.05344, 0.01439, 0.20524, 0.23005, 0.01590, 0.00231, 0.00759, 0.00171, 0.02707],
    [0.01781, 0.02580, 0.03003, 0.00370, 0.13839, 0.16128, 0.05344, 0.06530, 0.01439, 0.20524, 0.23005, 0.00759, 0.00171, 0.01590, 0.00231, 0.02707],
]
# GENERATOR: the truth's transition counts per second spent in a state (mirrored), rows sum to 0; A <-> B directly are
# tackles and first-time gains, A -> free_A a lost ball or a clearance, free_A -> B the other team's recovery (never back
# to A: a ball A's team-mate then controls was a pass, A's own).
GENERATOR = [
    [-0.07431, 0.00995, 0.01287, 0.00000, 0.05149, 0.00000],
    [0.00995, -0.07431, 0.00000, 0.01287, 0.00000, 0.05149],
    [0.00000, 0.78740, -4.72441, 0.00000, 3.93701, 0.00000],
    [0.78740, 0.00000, 0.00000, -4.72441, 0.00000, 3.93701],
    [0.00000, 0.55848, 0.14236, 0.00000, -0.70083, 0.00000],
    [0.55848, 0.00000, 0.00000, 0.14236, 0.00000, -0.70083],
]
# GAMMA, RATE_SCALE, OPEN_GAMMA: chosen by the procedure declared before the runs (scratch possv2/PREDECLARED.md, addendum
# 2): the grid gamma {.2 .3 .5} x rate_scale {3 10} x open_gamma {1 .3 .1}; Metrica first-half floors (team accuracy
# >= .97, named frames the truth's team >= .95); among those the best CAN-ISL dev score (W1 W3 W6 W8, 96 team ticks):
# dev coverage 0.5, accuracy 0.938 (3 wrong team), Metrica first half coverage
# 0.565 accuracy 0.9848 change F1 0.48.
GAMMA = 0.2
RATE_SCALE = 10.0
OPEN_GAMMA = 0.1      # the open-ball classes' ratios raised to this before gamma (likelihoods())
INIT_RESTART = [0.9545, 0.0091, 0.0091, 0.0091, 0.0091, 0.0091]   # the restarting team (A) has the ball first
INIT_UNKNOWN = [0.1667, 0.1667, 0.1667, 0.1667, 0.1667, 0.1667]
MODEL_PROVENANCE = {'fitted': '2026-10-31', 'data': 'Metrica game 3 (first half), simulated camera, ball variant tactical_gaps; operating constants chosen on CAN-ISL dev windows W1 W3 W6 W8', 'tool': 'tools/possession_v2_bench.py --fit'}


def _num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _pair(v):
    return isinstance(v, (list, tuple)) and len(v) >= 2 and _num(v[0]) and _num(v[1])


def _box(v):
    return isinstance(v, (list, tuple)) and len(v) == 4 and all(_num(x) for x in v)


def _runs(mask):
    """(start, end) inclusive index runs where mask is True."""
    m = np.asarray(mask, bool)
    if not m.any():
        return []
    d = np.diff(np.concatenate([[0], m.astype(np.int8), [0]]))
    return list(zip(np.flatnonzero(d == 1).tolist(), (np.flatnonzero(d == -1)-1).tolist()))


def _sigmoid(x):
    return 1.0/(1.0+np.exp(-np.clip(x, -40, 40)))


# ================================================================================================= the frame table
FRAME_KEYS = ('t', 'frame', 'seg', 'cut', 'bst', 'borig', 'bmot', 'bx', 'by', 'bsig', 'bconf', 'bspeed', 'bimg')
NEAR_KEYS = ('nr_raw', 'nr_team', 'nr_dp', 'nr_di', 'nr_x', 'nr_y', 'nr_isnear', 'nr_hands')
ROW_KEYS = ('rw_raw', 'rw_team', 'rw_x', 'rw_y')
CAND = 3          # per team and frame, the nearest eligible players resolve() keeps
NEAR_M = 6.0      # m (either metric): the people a frame keeps as possibly at the ball


def table(result, owned=None, seg=0, raw_ids=None):
    """The raw per-frame arrays of a result (a segment run's result.json, or a match view) for its frames with owned[0]
    <= frame < owned[1] (all when owned is None), independent of anyone's corrections (resolve() applies the people):
    the ball (state, origin, motion, pitch metres and uncertainty, speed, image ground point), the people who may be at
    it (within NEAR_M on either metric: their raw id, the team their frame names, the pitch and the image distance to
    the ball, their pitch position, whether the ball tracker's nearTrack is theirs, whether a goalkeeper's box holds the
    ball above his lower body inside a penalty area), and every player's pitch position (for the situation around a
    hidden ball). raw_ids: a shared list raw index -> raw id, extended in place (a match's tables share one)."""
    from . import ball_path
    from .possession_amateur import _shape, _reach
    frames = result.get('frames') or []
    video = result.get('video') or {}
    W = float(video['width']) if _num(video.get('width')) and video['width'] > 0 else 1920.0
    H = float(video['height']) if _num(video.get('height')) and video['height'] > 0 else 1080.0
    size = (result.get('tactical') or {}).get('pitchSize') or {}
    L = float(size['length']) if _num(size.get('length')) and size['length'] > 0 else LENGTH
    Wd = float(size['width']) if _num(size.get('width')) and size['width'] > 0 else WIDTH
    if owned is not None:
        frames = [f for f in frames if owned[0] <= int(f.get('frame', -1)) < owned[1]]
    if raw_ids is None:
        raw_ids = []
    index = {pid: i for i, pid in enumerate(raw_ids)}
    roster_team = {e['id']: e['team'] for e in (result.get('roster') or [])
                   if isinstance(e, dict) and isinstance(e.get('id'), str) and e.get('team') in TEAMS}

    def idx(pid):
        if pid not in index:
            index[pid] = len(raw_ids)
            raw_ids.append(pid)
        return index[pid]

    n = len(frames)
    t = np.zeros(n)
    fr = np.zeros(n, np.int64)
    cut = np.zeros(n, bool)
    bst = np.zeros(n, np.int8)
    borig = np.zeros(n, np.int8)
    bmot = np.zeros(n, np.int8)
    bx = np.full(n, np.nan)
    by = np.full(n, np.nan)
    bsig = np.full(n, np.nan)
    bconf = np.zeros(n)
    bspeed = np.full(n, np.nan)
    bimg = np.zeros(n, bool)
    nr = {k: [] for k in NEAR_KEYS}
    rw = {k: [] for k in ROW_KEYS}
    nr_start = np.zeros(n+1, np.int64)
    rw_start = np.zeros(n+1, np.int64)
    ST = {'UNKNOWN': 0, 'TRACKED': 1, 'MISSING': 2}
    OR = {'UNKNOWN': 0, 'OBSERVATION': 1, 'WEAK_DETECTION': 2, 'INTERPOLATION': 3}
    MO = {'UNKNOWN': 0, 'STATIONARY': 1, 'ROLLING': 2, 'FAST': 3, 'OCCLUDED': 4}
    seg_prev = None
    for i, f in enumerate(frames):
        t[i] = float(f['time']) if _num(f.get('time')) else 0.0
        fr[i] = int(f['frame']) if _num(f.get('frame')) else i
        s_ = f.get('segment')
        cut[i] = i > 0 and (f.get('cut') is True or (s_ is not None and seg_prev is not None and s_ != seg_prev))
        seg_prev = s_ if s_ is not None else seg_prev
        b = f.get('ball') or {}
        st = ST.get(b.get('state'), 0)
        bst[i] = st
        gx = gy = pxm = shape = None
        people = [p for p in (f.get('people') or []) if isinstance(p, dict) and _box(p.get('box'))]
        if st:
            borig[i] = OR.get(b.get('origin'), 0)
            bmot[i] = MO.get(b.get('motion'), 0)
            if _num(b.get('speed')):
                bspeed[i] = float(b['speed'])
            if _pair(b.get('pitch')) and _num(b.get('pitchConfidence')):
                bconf[i] = float(b['pitchConfidence'])
                sg = b.get('pitchSigma')
                sig = float(max(sg)) if isinstance(sg, (list, tuple)) and sg and all(_num(v) for v in sg) else 1.0
                if bconf[i] >= DEFAULTS['min_conf'] and sig <= DEFAULTS['max_sigma']:
                    bx[i], by[i], bsig[i] = b['pitch'][0]*L, b['pitch'][1]*Wd, sig
            if _pair(b.get('center')):
                gx = b['center'][0]*W
                gy = (b['center'][1]+(b['box'][3]/2 if _box(b.get('box')) else 0.0))*H
                pxm = max(ball_path._pixels_per_metre(people, gy, H), 1e-3)
                bimg[i] = True
                shape = _shape((f.get('calibration') or {}).get('H'), gx, gy, L, Wd, 25.0)
        nt = b.get('nearTrack') if st else None
        nr_start[i] = len(nr['nr_raw'])
        rw_start[i] = len(rw['rw_raw'])
        for p in people:
            if p.get('state') in ('rejected', 'nontarget') or p.get('role') not in ('PLAYER', 'GOALKEEPER', 'UNKNOWN'):
                continue
            raw = idx(p['id']) if isinstance(p.get('id'), str) else -1
            team = roster_team.get(p.get('id')) if isinstance(p.get('id'), str) else None
            team = team or p.get('team')
            tcode = -2 if p.get('role') == 'UNKNOWN' else (TEAMS.index(team) if team in TEAMS else -1)
            gk = p.get('role') == 'GOALKEEPER'
            pos = p.get('pitch')
            usable = _pair(pos) and _num(p.get('pitchConfidence')) and p['pitchConfidence'] >= DEFAULTS['min_conf']
            px_, py_ = (pos[0]*L, pos[1]*Wd) if usable else (math.nan, math.nan)
            if usable:
                rw['rw_raw'].append(raw)
                rw['rw_team'].append(tcode if not gk else (tcode if tcode >= 0 else -3))
                rw['rw_x'].append(px_)
                rw['rw_y'].append(py_)
            if not st:
                continue
            dp = math.hypot(px_-bx[i], py_-by[i]) if (usable and np.isfinite(bx[i])) else math.inf
            di = math.inf
            bxp, byp, bw, bh = p['box']
            if gx is not None and bw > 0 and bh > 0:
                fx, fy = (bxp+bw/2)*W, (byp+bh)*H
                di = _reach(shape[0], gx, gy, fx, fy, pxm) if shape is not None else math.hypot(gx-fx, 2.0*(gy-fy))/pxm
            if min(dp, di) > NEAR_M:
                continue
            hands = bool(gk and gx is not None and bw > 0 and bh > 0 and usable
                         and -.1 <= (b['center'][0]-bxp)/bw <= 1.1 and -.15 <= (b['center'][1]-byp)/bh < .6
                         and abs(py_-Wd/2) <= 22.16 and (px_ <= 18.5 or px_ >= L-18.5))
            nr['nr_raw'].append(raw)
            nr['nr_team'].append(tcode if not gk else (tcode if tcode >= 0 else -3))
            nr['nr_dp'].append(dp)
            nr['nr_di'].append(di)
            nr['nr_x'].append(px_)
            nr['nr_y'].append(py_)
            nr['nr_isnear'].append(nt is not None and p.get('track') == nt)
            nr['nr_hands'].append(hands)
    nr_start[n] = len(nr['nr_raw'])
    rw_start[n] = len(rw['rw_raw'])
    out = {'t': t, 'frame': fr, 'seg': np.full(n, seg, np.int32), 'cut': cut, 'bst': bst, 'borig': borig, 'bmot': bmot,
           'bx': bx, 'by': by, 'bsig': bsig, 'bconf': bconf, 'bspeed': bspeed, 'bimg': bimg,
           'nr_start': nr_start, 'rw_start': rw_start,
           'nr_raw': np.asarray(nr['nr_raw'], np.int32), 'nr_team': np.asarray(nr['nr_team'], np.int8),
           'nr_dp': np.asarray(nr['nr_dp'], np.float32), 'nr_di': np.asarray(nr['nr_di'], np.float32),
           'nr_x': np.asarray(nr['nr_x'], np.float32), 'nr_y': np.asarray(nr['nr_y'], np.float32),
           'nr_isnear': np.asarray(nr['nr_isnear'], bool), 'nr_hands': np.asarray(nr['nr_hands'], bool),
           'rw_raw': np.asarray(rw['rw_raw'], np.int32), 'rw_team': np.asarray(rw['rw_team'], np.int8),
           'rw_x': np.asarray(rw['rw_x'], np.float32), 'rw_y': np.asarray(rw['rw_y'], np.float32)}
    out['n'] = n
    out['raw_ids'] = raw_ids
    out['length'], out['width'] = L, Wd
    out['fps'] = float(video['fps']) if _num(video.get('fps')) and video['fps'] > 0 else 25.0
    return out


def concat(tables):
    """One raw table from consecutive raw tables sharing one raw_ids list (a match's segments, each trimmed to its owned
    frames); a frame further than max_dt from the previous one starts a new sequence (a cut)."""
    tables = [T for T in tables if T and T['n']]
    if not tables:
        return None
    out = {k: np.concatenate([T[k] for T in tables]) for k in FRAME_KEYS+NEAR_KEYS+ROW_KEYS}
    for start in ('nr_start', 'rw_start'):
        parts, offset = [], 0
        for T in tables:
            parts.append(T[start][:-1]+offset)
            offset += int(T[start][-1])
        out[start] = np.concatenate(parts+[np.array([offset], np.int64)])
    out['n'] = int(sum(T['n'] for T in tables))
    out['raw_ids'] = tables[-1]['raw_ids']
    out['length'], out['width'], out['fps'] = tables[0]['length'], tables[0]['width'], tables[0]['fps']
    first = np.concatenate([[0], np.cumsum([T['n'] for T in tables])[:-1]])
    for a in first[1:]:
        if out['t'][a]-out['t'][a-1] > DEFAULTS['max_dt']:
            out['cut'][a] = True
    return out


def resolve(T, person_of=None, team_of=None, excluded=(), ends=None, S=None):
    """The raw table under the people: raw ids -> person ids (person_of), a person's team (team_of; else the team his
    frame names), people who are not players dropped (excluded raw ids), a goalkeeper the tracking has not given a team
    playing for the goal on his side (ends: the teams defending the left and the right goal). Adds per team its CAND
    eligible players nearest the ball (cand_id: person index into T['ids'], cand_dp / cand_di metres, cand_x / cand_y,
    cand_near), the nearest person who may be a player of either team (u_d), the keeper holding the ball (kH team, kJ
    person), the nearTrack person (near) and every player's mapped rows (row_*). Returns a new dict."""
    S = dict(DEFAULTS, **(S or {}))
    person_of = person_of or {}
    team_of = team_of or {}
    excluded = set(excluded or ())
    raw_ids = T['raw_ids']
    ids, pindex = [], {}
    pmap = np.full(len(raw_ids)+1, -1, np.int64)     # raw index -> person index (-1: none / excluded); [-1] for raw -1
    ptm = np.full(len(raw_ids)+1, -1, np.int8)       # raw index -> overriding team (-1: the frame's)
    drop = np.zeros(len(raw_ids)+1, bool)
    for r, raw in enumerate(raw_ids):
        if raw in excluded:
            drop[r] = True
            continue
        pid = person_of.get(raw, raw)
        if pid not in pindex:
            pindex[pid] = len(ids)
            ids.append(pid)
        pmap[r] = pindex[pid]
        tm = team_of.get(pid)
        if tm in TEAMS:
            ptm[r] = TEAMS.index(tm)
    n = T['n']
    out = {k: T[k] for k in FRAME_KEYS}
    out.update(n=n, length=T['length'], width=T['width'], fps=T['fps'], ids=ids)

    def team_code(raw, frame_team, x):
        t_ = np.where(ptm[raw] >= 0, ptm[raw], frame_team)
        gk_free = t_ == -3
        if ends and ends[0] in TEAMS and ends[1] in TEAMS:
            left, right = TEAMS.index(ends[0]), TEAMS.index(ends[1])
            t_ = np.where(gk_free & np.isfinite(x), np.where(x < T['length']/2, left, right), t_)
        return np.where(t_ == -3, -1, t_).astype(np.int8)

    nraw = T['nr_raw']
    keep = ~drop[nraw]
    nteam = team_code(nraw, T['nr_team'], T['nr_x'])
    npers = pmap[nraw]
    # per frame, per team the CAND nearest by the effective distance
    dp, di = T['nr_dp'].astype(float), T['nr_di'].astype(float)
    both = np.isfinite(dp) & np.isfinite(di)
    deff = np.where(both, np.maximum(dp, di), np.minimum(dp, di)) if S['image_check'] else np.where(np.isfinite(dp), dp, di)
    cid = np.full((2, CAND, n), -1, np.int32)
    cdp = np.full((2, CAND, n), np.inf, np.float32)
    cdi = np.full((2, CAND, n), np.inf, np.float32)
    cx = np.full((2, CAND, n), np.nan, np.float32)
    cy = np.full((2, CAND, n), np.nan, np.float32)
    cnear = np.zeros((2, CAND, n), bool)
    ud = np.full(n, np.inf, np.float32)
    kH = np.full(n, -1, np.int8)
    kJ = np.full(n, -1, np.int32)
    near = np.full(n, -1, np.int32)
    starts = T['nr_start']
    for i in range(n):
        a, b = starts[i], starts[i+1]
        if a == b:
            continue
        sl = np.arange(a, b)
        sl = sl[keep[a:b]]
        if not len(sl):
            continue
        order = sl[np.argsort(deff[sl], kind='stable')]
        fill = [0, 0]
        for r in order:
            tm = nteam[r]
            if T['nr_isnear'][r] and npers[r] >= 0:
                near[i] = npers[r]
            if tm < 0:
                ud[i] = min(ud[i], deff[r])
                continue
            if T['nr_hands'][r] and kH[i] < 0:
                kH[i], kJ[i] = tm, npers[r]
            c = fill[tm]
            if c < CAND:
                cid[tm, c, i], cdp[tm, c, i], cdi[tm, c, i] = npers[r], dp[r], di[r]
                cx[tm, c, i], cy[tm, c, i], cnear[tm, c, i] = T['nr_x'][r], T['nr_y'][r], T['nr_isnear'][r]
                fill[tm] = c+1
    rraw = T['rw_raw']
    rkeep = ~drop[rraw] & (pmap[rraw] >= 0)
    rteam = team_code(rraw, T['rw_team'], T['rw_x'])
    rkeep &= rteam >= 0
    kept = np.concatenate([[0], np.cumsum(rkeep.astype(np.int64))])
    counts = kept[T['rw_start'][1:]]-kept[T['rw_start'][:-1]]
    out.update(cand_id=cid, cand_dp=cdp, cand_di=cdi, cand_x=cx, cand_y=cy, cand_near=cnear, u_d=ud, kH=kH, kJ=kJ, near=near,
               row_start=np.concatenate([[0], np.cumsum(counts)]).astype(np.int64),
               row_id=pmap[rraw][rkeep].astype(np.int32), row_team=rteam[rkeep], row_x=T['rw_x'][rkeep], row_y=T['rw_y'][rkeep])
    return out


def weights(T, max_gap=None):
    """Seconds each frame stands for: half the gaps to its neighbours, each half capped at one frame's duration (a gap
    in the frames, a halftime break, is nobody's time). Frames counted once; the sum is the covered time."""
    t = T['t']
    n = len(t)
    if n == 0:
        return np.zeros(0)
    step = 1.0/float(T.get('fps') or 25.0)
    cap = step if max_gap is None else max_gap
    gaps = np.diff(t)
    left = np.concatenate([[step], np.minimum(gaps, cap)])
    right = np.concatenate([np.minimum(gaps, cap), [step]])
    return (left+right)/2.0


# ================================================================================================= the ball's states
def ball_states(T, reacquire_s=.5, lost_s=1.0):
    """Per frame: the ball observation state (index into BALL_STATES), its confidence and provenance (PROVENANCE).
    OBSERVED: seen on this frame (TRACKED; a weak detection keeps its own provenance and a lower confidence);
    OCCLUDED: interpolated across a short gap while a player's body hid it (nearTrack, or the gap begins and ends at the
    same player); PREDICTED: interpolated across a gap otherwise (a ball in flight between two sightings);
    REACQUIRING: seen, within reacquire_s of the end of an unplaced stretch of lost_s or more (a new trajectory not yet
    established); UNKNOWN: not placed. Interpolated positions are never OBSERVED."""
    n = T['n']
    st = np.full(n, 4, np.int8)
    conf = np.zeros(n)
    prov = np.full(n, 3, np.int8)
    bst, borig, t = T['bst'], T['borig'], T['t']
    seen = bst == 1
    st[seen] = 0
    prov[seen & (borig == 2)] = 1
    prov[seen & (borig != 2)] = 0
    conf[seen] = np.where(borig[seen] == 2, .5, .9)
    bridged = bst == 2
    prov[bridged] = 2
    for a, b in _runs(bridged):
        first = T['cand_id'][:, 0, max(a-1, 0)]
        last = T['cand_id'][:, 0, min(b+1, n-1)]
        same = any(first[k] >= 0 and first[k] == last[k] and T['cand_di'][k, 0, max(a-1, 0)] < 2.0 and
                   T['cand_di'][k, 0, min(b+1, n-1)] < 2.0 for k in range(2))
        occ = same or bool((T['near'][a:b+1] >= 0).any())
        st[a:b+1] = 1 if occ else 2
        span = t[b]-t[a]
        conf[a:b+1] = max(.2, .7-.4*span) if occ else max(.1, .6-.4*span)
    for a, b in _runs(bst == 0):
        if t[b]-t[a] >= lost_s and b+1 < n:
            j = b+1
            while j < n and bst[j] == 1 and t[j]-t[b+1] < reacquire_s:
                st[j] = 3
                conf[j] = .6
                j += 1
    return st, conf, prov


# ================================================================================================= the evidence
def _kinks(T, w):
    """Per frame, the change of the ball's pitch velocity (m/s) over +-w frames (possession-4's touch measure)."""
    t, bx, by = T['t'], T['bx'], T['by']
    n = len(t)
    out = np.zeros(n)
    if n <= 2*w:
        return out
    ok_ = np.isfinite(bx) & (T['bst'] > 0)
    i = np.arange(w, n-w)
    fps = float(T.get('fps') or 25.0)
    ok = ok_[i-w] & ok_[i] & ok_[i+w] & (t[i+w]-t[i-w] <= (2*w+.5)/fps)
    dt0 = np.maximum(t[i]-t[i-w], 1e-6)
    dt1 = np.maximum(t[i+w]-t[i], 1e-6)
    with np.errstate(invalid='ignore'):
        dvx = (bx[i+w]-bx[i])/dt1-(bx[i]-bx[i-w])/dt0
        dvy = (by[i+w]-by[i])/dt1-(by[i]-by[i-w])/dt0
    out[i] = np.where(ok, np.nan_to_num(np.hypot(dvx, dvy)), 0.0)
    return out


def touches(T, S):
    """Frames at (or within touch_r of) the peak of a bend of the ball's path of at least S['kink'] m/s."""
    d = _kinks(T, S['kink_w'])
    n = len(d)
    r = 3
    pad = np.concatenate([np.zeros(r), d, np.zeros(r)])
    local = np.max(np.stack([pad[j:j+n] for j in range(2*r+1)]), axis=0) if n else d
    peaks = (d >= S['kink']) & (d >= local)
    out = peaks.copy()
    for s in range(1, S['touch_r']+1):
        out[s:] |= peaks[:-s]
        out[:-s] |= peaks[s:]
    return out


def _effective(T, S):
    """Per team and candidate the effective distance (m) and whether it rests on the image alone."""
    dp, di = T['cand_dp'].astype(float), T['cand_di'].astype(float)
    if S['image_check']:
        both = np.isfinite(dp) & np.isfinite(di)
        d = np.where(both, np.maximum(dp, di), np.minimum(dp, di))
    else:
        d = np.where(np.isfinite(dp), dp, di)
    img_only = ~np.isfinite(dp) & np.isfinite(di)
    return d, img_only


def proximity(T, S):
    """(pA, pB, pU, who, img_only): per frame how surely each team's nearest eligible player is at the ball (0..1), the
    same for a person who may be a player of either team, the nearest player of each team (person index, -1) and
    whether that rests on the image only."""
    n = T['n']
    d, img_only = _effective(T, S)
    sig = np.where(np.isfinite(T['bsig']), T['bsig'], 1.0)
    bridged = T['bst'] == 2
    r = S['r_ctrl']+S['sigma_w']*sig+np.where(bridged, S['bridge_extra'], 0.0)
    p = _sigmoid((r[None, None, :]-d)/S['r_soft'])
    p = np.where(np.isfinite(d), p, 0.0)
    # the player whose body hid the ball across a gap is at it
    p = np.where(T['cand_near'] & bridged[None, None, :], np.maximum(p, .8), p)
    k0 = np.argmax(p, axis=1)                     # the nearest (most surely at it) candidate per team
    pt = np.take_along_axis(p, k0[:, None, :], axis=1)[:, 0, :]
    who = np.take_along_axis(T['cand_id'], k0[:, None, :], axis=1)[:, 0, :]
    io = np.take_along_axis(img_only, k0[:, None, :], axis=1)[:, 0, :]
    ru = S['r_ctrl']+S['sigma_w']*sig
    pu = np.where(np.isfinite(T['u_d']), _sigmoid((ru-T['u_d'])/S['r_soft']), 0.0)
    # the keeper with the ball in his hands
    for k in range(2):
        hold = T['kH'] == k
        pt[k] = np.where(hold, 1.0, pt[k])
        pt[1-k] = np.where(hold, 0.0, pt[1-k])
        who[k] = np.where(hold, T['kJ'], who[k])
    pt = np.where(pt < 1e-3, 0.0, pt)
    return pt[0], pt[1], np.where(pu < 1e-3, 0.0, pu), who, io


def classes(T, S=None):
    """(M, aux): per frame the soft memberships of the observation classes (n x K, rows sum to 1) and what the decoder's
    output reads besides (the nearest player per team, the last player seen on the ball and his team)."""
    S = dict(DEFAULTS, **(S or {}))
    n = T['n']
    M = np.zeros((n, K))
    pA, pB, pU, who, img_only = proximity(T, S)
    bst, borig = T['bst'], T['borig']
    seen = bst == 1
    t = T['t']
    # a bridged frame of a long interpolation is no position
    bridged = np.zeros(n, bool)
    for a, b in _runs(bst == 2):
        if t[b]-t[a] <= S['long_bridge']:
            bridged[a:b+1] = True
    placed = seen | bridged
    fast = np.nan_to_num(T['bspeed'], nan=0.0) >= S['fast']
    touch = touches(T, S)
    flying = fast & ~touch
    pa = np.where(flying, 0.0, pA)
    pb = np.where(flying, 0.0, pB)
    pu = np.where(flying, 0.0, pU)
    oneA = pa*(1-pb)*(1-pu)
    oneB = pb*(1-pa)*(1-pu)
    duel = pa*pb
    open_ = (1-pa)*(1-pb)*(1-pu)
    w = np.ones(n)
    w = np.where(seen & (borig == 2), S['weak_w'], w)
    io = img_only[0] & img_only[1] | ~np.isfinite(T['bx'])
    w = np.where(io & placed, w*S['image_only_w'], w)
    # how long nobody has been at the ball (a placed ball within reach of a player, not flying past him)
    at_someone = placed & ((pa >= .5) | (pb >= .5) | (pu >= .5))
    since = np.zeros(n)
    last = -np.inf
    for i in range(n):
        if T['cut'][i]:
            last = -np.inf
        if at_someone[i]:
            last = t[i]
        since[i] = t[i]-last
    long_ = since >= S['open_long']
    for cls, base in (('OBS', seen), ('BR', bridged)):
        m = base.astype(float)*w
        M[:, CI[cls+'_A']] = m*oneA
        M[:, CI[cls+'_B']] = m*oneB
        M[:, CI[cls+'_DUEL']] = m*duel
        M[:, CI[cls+'_OPEN']] = m*open_*~long_
        M[:, CI[cls+'_OPEN_LONG']] = m*open_*long_
    # the ball not placed: the situation around the last player seen on it (past evidence only; the decoder's backward
    # pass brings the future)
    last_k = np.full(n, -1, np.int8)
    last_j = np.full(n, -1, np.int64)
    last_t = np.full(n, -np.inf)
    flight_t = np.full(n, -np.inf)
    cur_k, cur_j, cur_t, cur_f = -1, -1, -np.inf, -np.inf
    strong = (oneA >= .5) | (oneB >= .5)
    for i in range(n):
        if T['cut'][i]:
            cur_k, cur_j, cur_t, cur_f = -1, -1, -np.inf, -np.inf
        if placed[i] and strong[i]:
            k = 0 if oneA[i] >= oneB[i] else 1
            cur_k, cur_j, cur_t = k, int(who[k, i]), t[i]
        elif placed[i] and open_[i] >= .5:
            cur_k, cur_j = -1, -1     # the ball was seen away from him: what he does next says nothing about it
        if seen[i] and flying[i] and open_[i] >= .5:
            cur_f = t[i]
        elif seen[i] and not flying[i]:
            cur_f = -np.inf
        last_k[i], last_j[i], last_t[i], flight_t[i] = cur_k, cur_j, cur_t, cur_f
    gap = ~placed
    rs, rid, rteam, rx, ry = T['row_start'], T['row_id'], T['row_team'], T['row_x'], T['row_y']
    for i in np.flatnonzero(gap):
        if t[i]-flight_t[i] <= S['flight_age']:
            M[i, CI['GAP_FLIGHT']] = 1.0
            continue
        if last_k[i] < 0 or t[i]-last_t[i] > S['gap_age'] or last_j[i] < 0:
            continue
        a, b = rs[i], rs[i+1]
        js = rid[a:b]
        me = np.flatnonzero(js == last_j[i])
        if not len(me):
            continue
        mx, my = rx[a+me[0]], ry[a+me[0]]
        opp = rteam[a:b] != last_k[i]
        if not opp.any() or not np.isfinite(mx):
            dmin = math.inf
        else:
            dmin = float(np.nanmin(np.hypot(rx[a:b][opp]-mx, ry[a:b][opp]-my)))
        press = float(_sigmoid((S['press_r']+(S['free_r']-S['press_r'])/2-dmin)/.3)) if np.isfinite(dmin) else 0.0
        k = int(last_k[i])
        M[i, CI['GAP_%s_FREE' % TEAMS[k]]] = 1-press
        M[i, CI['GAP_%s_PRESS' % TEAMS[k]]] = press
    M[:, CI['NONE']] = np.clip(1.0-M[:, 1:].sum(axis=1), 0.0, 1.0)
    aux = {'who': who, 'pA': pA, 'pB': pB, 'last_k': last_k, 'last_j': last_j, 'last_t': last_t, 'seen': seen,
           'bridged': bridged, 'flying': flying, 'oneA': oneA, 'oneB': oneB}
    return M, aux


# ================================================================================================= the model
def model(params=None):
    """The fitted model as arrays: EMISSION (4 x K), GENERATOR (4 x 4 per s), GAMMA, the initial distributions."""
    p = params or {}
    E = np.asarray(p.get('emission', EMISSION), float)
    G = np.asarray(p.get('generator', GENERATOR), float)*float(p.get('rate_scale', RATE_SCALE))
    np.fill_diagonal(G, 0.0)
    np.fill_diagonal(G, -G.sum(axis=1))
    return {'E': E, 'G': G, 'gamma': float(p.get('gamma', GAMMA)), 'open_gamma': float(p.get('open_gamma', OPEN_GAMMA)),
            'init_restart': np.asarray(p.get('init_restart', INIT_RESTART), float),
            'init_unknown': np.asarray(p.get('init_unknown', INIT_UNKNOWN), float)}


def _expm(G, dt):
    """exp(G dt) for a generator (eigen-free: scaling and squaring of a Taylor series)."""
    A_ = G*dt
    m = len(G)
    norm = np.abs(A_).sum(axis=1).max()
    s = max(0, int(math.ceil(math.log2(norm)))+1) if norm > .5 else 0
    A_ = A_/(2**s)
    out = np.eye(m)
    term = np.eye(m)
    for k in range(1, 12):
        term = term@A_/k
        out = out+term
    for _ in range(s):
        out = out@out
    out = np.clip(out, 0.0, None)
    return out/out.sum(axis=1, keepdims=True)


OPEN_CLASSES = ('OBS_OPEN', 'OBS_OPEN_LONG', 'BR_OPEN', 'BR_OPEN_LONG', 'GAP_FLIGHT')


def likelihoods(M, mod, dt):
    """Per frame and state the tempered emission likelihood (n x states): sum_k m_k R(s, k) with R = P(k | s) / Q(k),
    raised to gamma * dt * 25 (the exponent was fitted at 25 frames a second). The open-ball classes' ratios are first
    raised to open_gamma: a ball nobody is at is one event over its whole flight, so its frames repeat one piece of
    evidence (a 3 s pass is 75 frames) far more than the frames at a player do."""
    E = mod['E']
    Q = E.mean(axis=0)   # the classes' marginal under equal states (only ratios between states matter)
    R = E/np.maximum(Q, 1e-12)[None, :]
    og = mod.get('open_gamma', 1.0)
    if og != 1.0:
        cols = [CI[c] for c in OPEN_CLASSES]
        R[:, cols] = R[:, cols]**og
    # nothing to go on says nothing: on the simulated camera an unplaced ball is mostly a loose one (.7% of its frames),
    # on CAN-ISL a fifth of the live frames have no usable evidence for reasons the simulation has not (an unidentified
    # carrier, an image-only position, a long stationary bridge), so the class is neutral by construction
    R[:, CI['NONE']] = 1.0
    L = M@R.T
    L = np.maximum(L, 1e-12)
    expo = mod['gamma']*np.clip(dt, 0.0, .2)*25.0
    return np.exp(np.log(L)*expo[:, None])


def _forward_backward(Lk, P, init):
    """Posterior (n x 4) of one sequence: Lk the per-frame likelihoods, P a list (or one) of transition matrices
    (P[i] from frame i-1 to i), init the first frame's prior."""
    n, m = Lk.shape
    alpha = np.zeros((n, m))
    c = np.zeros(n)
    a = init*Lk[0]
    c[0] = a.sum() or 1e-300
    alpha[0] = a/c[0]
    same = not isinstance(P, list)
    for i in range(1, n):
        Pi = P if same else P[i]
        a = (alpha[i-1]@Pi)*Lk[i]
        s = a.sum()
        if not s > 0:
            a = Lk[i].copy()
            s = a.sum() or 1e-300
        c[i] = s
        alpha[i] = a/s
    beta = np.ones((n, m))
    for i in range(n-2, -1, -1):
        Pi = P if same else P[i+1]
        b = Pi@(Lk[i+1]*beta[i+1])
        s = b.sum()
        beta[i] = b/s if s > 0 else np.ones(m)/m
    post = alpha*beta
    post /= np.maximum(post.sum(axis=1, keepdims=True), 1e-300)
    return post


def _viterbi(Lk, P, init):
    n, m = Lk.shape
    logL = np.log(np.maximum(Lk, 1e-300))
    same = not isinstance(P, list)
    logP = np.log(np.maximum(P, 1e-300)) if same else None
    back = np.zeros((n, m), np.int8)
    v = np.log(np.maximum(init, 1e-300))+logL[0]
    for i in range(1, n):
        lp = logP if same else np.log(np.maximum(P[i], 1e-300))
        cand = v[:, None]+lp
        back[i] = np.argmax(cand, axis=0)
        v = cand[back[i], np.arange(m)]+logL[i]
    path = np.zeros(n, np.int8)
    path[-1] = int(np.argmax(v))
    for i in range(n-1, 0, -1):
        path[i-1] = back[i, path[i]]
    return path


def decode(T, M, live, mod, starts=None, S=None):
    """(post, path): per frame the probability of each state and the most likely sequence, over the frames where live
    is True, in sequences broken where live is False, at camera cuts (when cut_reset) and where consecutive frames are
    more than max_dt apart. starts: {frame index: team index or None}, the restarting team of a sequence that begins
    there (the init_restart prior mirrored to it); other sequences begin from init_unknown."""
    S = dict(DEFAULTS, **(S or {}))
    n = T['n']
    post = np.full((n, NS), np.nan)
    path = np.full(n, -1, np.int8)
    t = T['t']
    dt = np.concatenate([[1.0/float(T.get('fps') or 25.0)], np.diff(t)])
    brk = ~live.copy()
    starts = starts or {}
    cache = {}
    Lall = likelihoods(M, mod, dt)
    seq_start = np.zeros(n, bool)
    seq_start[1:] = (dt[1:] > S['max_dt']) | (T['cut'][1:] & S['cut_reset'])
    for a, b in _runs(live):
        cuts = [a]+[i for i in range(a+1, b+1) if seq_start[i]]+[b+1]
        for lo, hi in zip(cuts[:-1], cuts[1:]):
            idx = np.arange(lo, hi)
            if not len(idx):
                continue
            team = starts.get(lo)
            init = mod['init_unknown'] if team is None else (mod['init_restart'] if team == 0 else mod['init_restart'][MIRROR_STATE])
            Ps = []
            for i in idx:
                key = round(float(min(dt[i], S['max_dt'])), 3)
                if key not in cache:
                    cache[key] = _expm(mod['G'], key)
                Ps.append(cache[key])
            Lk = Lall[idx]
            common = len(set(round(float(min(dt[i], S['max_dt'])), 3) for i in idx)) == 1
            P = Ps[0] if common else Ps
            post[idx] = _forward_backward(Lk, P, init)
            path[idx] = _viterbi(Lk, P, init)
    return post, path


# ================================================================================================= labels and output
def direct(M, T):
    """Per team, frames where the ball is placed at that team's player (OBS or BR class at least .5, the keeper's hands)."""
    dA = (M[:, CI['OBS_A']]+M[:, CI['BR_A']] >= .5) | (T['kH'] == 0)
    dB = (M[:, CI['OBS_B']]+M[:, CI['BR_B']] >= .5) | (T['kH'] == 1)
    return dA, dB


def label(T, post, path, M, aux, live_code, S=None):
    """Per frame: the code (CODES index), the designation (0 none, 1 observed, 2 inferred), the possessor (person index,
    -1) with its confidence (0 none, 1 low, 2 medium, 3 high), and the possession changes.

    The team comes from the posterior (a team is named at min_posterior); contested and free come from the definition:
    between the losing team's last touch (the last frame its player was at the placed ball) and the winning team's first
    control (the first frame after the decoded switch its player is), and between a team's last touch and a stoppage,
    a placed ball nobody is at is FREE and one both teams are at is CONTESTED; the frames there without a placed ball
    keep the posterior (unknown unless it is sure). A pass to a team-mate is never inside such a window, so its flight
    stays the team's. live_code: per frame the code the game state gives where play is not live (STOPPAGE / HALFTIME /
    NON_MATCH / POSSIBLE) and -1 where it is; POSSIBLE frames are decoded too (their decoded code is `inner`)."""
    S = dict(DEFAULTS, **(S or {}))
    n = T['n']
    th = S['min_posterior']
    t = T['t']
    code = np.full(n, 4, np.int8)
    ok = np.isfinite(post[:, 0])
    pa, pb = post[:, A], post[:, B]
    pc, pf = post[:, CA]+post[:, CB], post[:, FA]+post[:, FB]
    code[ok & (pa >= th)] = 0
    code[ok & (pb >= th)] = 1
    neu = ok & (pa < th) & (pb < th) & (pc+pf >= th)
    code[neu & (pc >= pf)] = 2
    code[neu & (pc < pf)] = 3
    dA, dB = direct(M, T)
    dteam = (dA, dB)
    openm = M[:, CI['OBS_OPEN']]+M[:, CI['OBS_OPEN_LONG']]+M[:, CI['BR_OPEN']]+M[:, CI['BR_OPEN_LONG']]
    duelm = M[:, CI['OBS_DUEL']]+M[:, CI['BR_DUEL']]
    who = aux['who']
    changes = []

    def window(a, b):
        """Frames a..b (inclusive) lie between a loss and a gain: free / contested from their own evidence."""
        if b < a:
            return
        sl = slice(a, b+1)
        named = (code[sl] == 0) | (code[sl] == 1)
        code[sl] = np.where(named, code[sl], np.where(duelm[sl] >= .5, 2, np.where(openm[sl] >= .5, 3, code[sl])))
    # the decoded sequences: runs of decoded frames (path >= 0) between breaks
    seq = path >= 0
    for lo, hi in _runs(seq):
        # breaks inside a run of decoded frames (a camera cut restarting the belief): decode() restarted there, and the
        # team path is read per piece
        cuts = [lo]+[i for i in range(lo+1, hi+1) if T['cut'][i] and S['cut_reset']]+[hi+1]
        for a0, b0 in zip(cuts[:-1], cuts[1:]):
            b0 -= 1
            tp = path[a0:b0+1].astype(int)
            teams_idx = [i for i in range(len(tp)) if tp[i] in (A, B)]
            prev = run_start = None
            for q, i in enumerate(teams_idx):
                k = tp[i]
                if prev is None:
                    run_start = i
                elif k != tp[prev]:
                    kp = tp[prev]
                    # the loser's last touch inside its own run, the winner's first control inside its new run
                    lt = next((j for j in range(a0+prev, a0+run_start-1, -1) if dteam[kp][j]), None)
                    nxt_other = next((teams_idx[r] for r in range(q+1, len(teams_idx)) if tp[teams_idx[r]] != k), len(tp))
                    fc = next((j for j in range(a0+i, a0+nxt_other) if dteam[k][j]), None)
                    w0 = (lt+1) if lt is not None else a0+prev+1
                    w1 = (fc-1) if fc is not None else a0+i-1
                    window(w0, w1)
                    won = fc if fc is not None else a0+i
                    fps_ = float(T.get('fps') or 25.0)
                    high = (fc is not None and lt is not None and t[fc]-t[lt] <= 5.0 and
                            float(post[fc:min(fc+int(.5*fps_)+1, n), k].mean()) >= .9)
                    changes.append({'from': TEAMS[kp], 'to': TEAMS[k], 'kind': 'open',
                                    'lostAt': round(float(t[lt]), 3) if lt is not None else None,
                                    'wonAt': round(float(t[won]), 3), 'lostFrame': int(T['frame'][lt]) if lt is not None else None,
                                    'wonFrame': int(T['frame'][won]), '_lost': lt, '_won': won,
                                    'supported': fc is not None, 'confidence': 'high' if high else 'uncertain'})
                    run_start = i
                prev = i
            # a stoppage after the last team's last touch: a placed ball nobody is at before it is free
            if teams_idx:
                kl = tp[teams_idx[-1]]
                after = b0+1 < n and live_code[b0+1] in (STOPPAGE, HALFTIME, NON_MATCH)
                if after:
                    lt = None
                    for j in range(b0, a0-1, -1):
                        if dteam[kl][j]:
                            lt = j
                            break
                    if lt is not None:
                        sl = slice(lt+1, b0+1)
                        named = (code[sl] == 0) | (code[sl] == 1)
                        code[sl] = np.where(~named & (openm[sl] >= .5), 3, code[sl])
    inner = code.copy()
    excl = live_code >= 0
    code[excl] = live_code[excl]
    # observed vs inferred
    design = np.zeros(n, np.int8)
    for k, dk in ((0, dA), (1, dB)):
        team = inner == k
        obs = (M[:, CI['OBS_A' if k == 0 else 'OBS_B']] >= .5) | (T['kH'] == k)
        design[team & obs] = 1
        design[team & ~obs] = 2
    # the possessor: the team's player at the ball, else the last (or next) one seen on it within carry_s, same team run
    poss = np.full(n, -1, np.int64)
    pconf = np.zeros(n, np.int8)
    for k, obs_cls, br_cls in ((0, 'OBS_A', 'BR_A'), (1, 'OBS_B', 'BR_B')):
        team = inner == k
        direct_ = team & ((M[:, CI[obs_cls]] >= .5) | (T['kH'] == k))
        bridged = team & (M[:, CI[br_cls]] >= .5) & ~direct_
        poss[direct_] = np.where(T['kH'][direct_] == k, T['kJ'][direct_], who[k, direct_])
        pconf[direct_] = 3
        poss[bridged] = who[k, bridged]
        pconf[bridged] = 2
        for a, b in _runs(team):
            known = np.array([i for i in range(a, b+1) if poss[i] >= 0])
            if not len(known):
                continue
            for i in range(a, b+1):
                if poss[i] >= 0:
                    continue
                j = int(np.searchsorted(known, i))
                cands = [known[j-1]] if j > 0 else []
                if j < len(known):
                    cands.append(known[j])
                best = min(cands, key=lambda q: abs(t[q]-t[i]))
                if abs(t[best]-t[i]) <= S['carry_s']:
                    poss[i] = poss[best]
                    pconf[i] = 1
    for c in changes:
        lt, fc = c.pop('_lost'), c.pop('_won')
        c['lostBy'] = int(poss[lt]) if lt is not None and poss[lt] >= 0 else (int(who[TEAMS.index(c['from']), lt]) if lt is not None else None)
        c['wonBy'] = int(poss[fc]) if poss[fc] >= 0 else int(who[TEAMS.index(c['to']), fc])
        if c['lostBy'] is not None and c['lostBy'] < 0:
            c['lostBy'] = None
        if c['wonBy'] is not None and c['wonBy'] < 0:
            c['wonBy'] = None
    return {'code': code, 'inner': inner, 'design': design, 'possessor': poss, 'pconf': pconf, 'changes': changes}


def intervals(T, out, post, ids=None, names=None, bstate=None):
    """Runs of constant (code, designation): t0, t1 (each frame its weights() time, so the runs tile the frames),
    state, team, designation (observed / inferred / None), confidence (the mean probability of the named state; None for
    unknown and the excluded kinds), possessor (the commonest player of the run, with his shown label), the share of its
    frames the ball was seen on (ballSeen), and under a possible stoppage the state decoded there."""
    code, design, inner = out['code'], out['design'], out['inner']
    w = weights(T)
    t = T['t']
    n = T['n']
    res = []
    if n == 0:
        return res
    key = code.astype(np.int32)*4+design
    step = 1.0/float(T.get('fps') or 25.0)
    # a run never spans a hole in the frames (a gap is nobody's time: weights() gives each side at most one frame)
    change = np.flatnonzero((np.diff(key) != 0) | (np.diff(t) > 1.5*step))+1
    starts = np.concatenate([[0], change])
    ends = np.concatenate([change-1, [n-1]])
    for a, b in zip(starts.tolist(), ends.tolist()):
        c = int(code[a])
        r = {'t0': round(float(t[a]-w[a]/2), 3), 't1': round(float(t[b]+w[b]/2), 3), 'state': CODES[c],
             'team': TEAMS[c] if c in (0, 1) else None,
             'designation': {1: 'observed', 2: 'inferred'}.get(int(design[a])) if c in (0, 1) else None}
        if c in (0, 1, 2, 3) and np.isfinite(post[a, 0]):
            col = {0: [A], 1: [B], 2: [CA, CB, FA, FB], 3: [CA, CB, FA, FB]}[c]
            r['confidence'] = round(float(post[a:b+1][:, col].sum(axis=1).mean()), 3)
        if c == POSSIBLE:
            r['decoded'] = CODES[Counter(int(x) for x in inner[a:b+1]).most_common(1)[0][0]]
        if bstate is not None:
            r['ballSeen'] = round(float((bstate[a:b+1] == 0).mean()), 2)
        ps = out['possessor'][a:b+1]
        ps = ps[ps >= 0]
        if len(ps) and ids is not None:
            pid = ids[int(Counter(ps.tolist()).most_common(1)[0][0])]
            r['possessor'] = pid
            if names and pid in names and names[pid].get('display'):
                r['possessorLabel'] = names[pid]['display']
        res.append(r)
    return res


def stats(T, out, post=None, lo=None, hi=None, min_resolved=30.0):
    """Shares of the frames in [lo, hi) by time (each frame its weights() seconds, counted once): see stats_from_runs."""
    t = T['t']
    w = weights(T)
    sel = np.ones(T['n'], bool)
    if lo is not None:
        sel &= t >= lo
    if hi is not None:
        sel &= t < hi
    runs = []
    code, design = out['code'], out['design']
    idx = np.flatnonzero(sel)
    for i in idx:
        runs.append((float(t[i]-w[i]/2), float(t[i]+w[i]/2), int(code[i]), int(design[i])))
    return stats_from_runs(runs, min_resolved=min_resolved)


def stats_from_runs(runs, lo=None, hi=None, min_resolved=30.0, min_sequence=1.0):
    """The analytics of runs (t0, t1, code, designation) clipped to [lo, hi):
    seconds per code; observed / inferred seconds per team; live (A + B + contested + free + unknown) and resolved
    (A + B) time; the team percentages over A + B (None when A + B is under min_resolved seconds: 'insufficient', never
    50/50); coverage = (A + B + contested + free) / live; the possession sequences per team (maximal stretches named
    that team that only unknown time interrupts and that hold min_sequence seconds of its time or more: count, mean
    duration of the team's time in them)."""
    sec = Counter()
    obs, inf = Counter(), Counter()
    seqs = []
    cur = None
    for t0, t1, c, d in runs:
        if lo is not None:
            t0 = max(t0, lo)
        if hi is not None:
            t1 = min(t1, hi)
        if t1 <= t0:
            continue
        dur = t1-t0
        sec[CODES[c]] += dur
        if c in (0, 1):
            (obs if d == 1 else inf)[TEAMS[c]] += dur
            if cur is None or cur[0] != c:
                if cur is not None:
                    seqs.append(cur)
                cur = [c, t0, t1, 0.0]
            cur[2] = t1
            cur[3] += dur
        elif c == 4:
            continue
        else:
            if cur is not None:
                seqs.append(cur)
            cur = None
    if cur is not None:
        seqs.append(cur)
    seqs = [s for s in seqs if s[3] >= min_sequence]
    resolved = sec['A']+sec['B']
    live = sum(sec[CODES[i]] for i in LIVE_CODES)
    pct = None
    if resolved >= min_resolved:
        a = round(100.0*sec['A']/resolved, 1)
        pct = {'A': a, 'B': round(100.0-a, 1)}
    per = {k: [s for s in seqs if s[0] == i] for i, k in enumerate(TEAMS)}
    return {'seconds': {k: round(sec.get(k, 0.0), 2) for k in CODES},
            'live': round(live, 2), 'resolved': round(resolved, 2),
            'observed': {k: round(obs.get(k, 0.0), 2) for k in TEAMS}, 'inferred': {k: round(inf.get(k, 0.0), 2) for k in TEAMS},
            'percent': pct, 'insufficient': pct is None,
            'coverage': round((resolved+sec['contested']+sec['free'])/live, 3) if live > 0 else None,
            'shareOfLive': ({k: round(v/live, 3) for k, v in (('observed', sum(obs.values())), ('inferred', sum(inf.values())),
                                                               ('contested', sec['contested']), ('free', sec['free']),
                                                               ('unknown', sec['unknown']))} if live > 0 else None),
            'sequences': {k: {'count': len(v), 'meanSeconds': round(float(np.mean([s[3] for s in v])), 2) if v else None}
                          for k, v in per.items()},
            'minResolved': min_resolved}


def model_id():
    """A short fingerprint of the fitted model (caches are keyed by it)."""
    import hashlib
    import json
    key = [EMISSION, GENERATOR, GAMMA, RATE_SCALE, OPEN_GAMMA, INIT_RESTART, INIT_UNKNOWN, sorted(DEFAULTS.items())]
    return hashlib.sha256(json.dumps(key, default=str).encode()).hexdigest()[:12]


def record(T, out, post, path, M, bstate, halves=(), names=None, settings=None, min_resolved=30.0):
    """The JSON-safe record of a decoded timeline: the definition, the intervals, the changes, the stats of the whole
    range and of each half, the ball's observation states over live play, notes."""
    S = dict(DEFAULTS, **(settings or {}))
    ids = T.get('ids') or []
    runs = intervals(T, out, post, ids=ids, names=names, bstate=bstate)
    t = T['t']
    lo, hi = (float(t[0]), float(t[-1])) if T['n'] else (0.0, 0.0)
    rr = [(r['t0'], r['t1'], CODES.index(r['state']), {'observed': 1, 'inferred': 2}.get(r['designation'], 0)) for r in runs]
    st = {'match': stats_from_runs(rr, min_resolved=min_resolved)}
    for h in halves or ():
        st[h['name']] = stats_from_runs(rr, lo=h['t0'], hi=h['t1'], min_resolved=min_resolved) | {'t0': h['t0'], 't1': h['t1']}
    live = np.isin(out['code'], LIVE_CODES)
    w = weights(T)
    bs = {}
    if live.any():
        tot = float(w[live].sum())
        for k, name in enumerate(BALL_STATES):
            bs[name] = round(float(w[live & (bstate == k)].sum())/tot, 3)
    changes = []
    for c in out['changes']:
        c = dict(c)
        for key in ('lostBy', 'wonBy'):
            if c.get(key) is not None:
                pid = ids[c[key]]
                c[key] = pid
                if names and pid in names and names[pid].get('display'):
                    c[key+'Label'] = names[pid]['display']
        changes.append(c)
    notes = []
    m = st['match']
    if m['live'] > 0 and m['shareOfLive'] and m['shareOfLive']['inferred'] > m['shareOfLive']['observed']:
        notes.append('More of the controlled time is inferred than directly observed: the shares are an estimate.')
    if m['insufficient']:
        notes.append(f'Less than {min_resolved:.0f} s of resolved controlled time: no percentages.')
    return {'version': VERSION, 'model': {'id': model_id(), 'provenance': MODEL_PROVENANCE}, 'definition': DEFINITION,
            'settings': {'minPosterior': S['min_posterior'], 'minResolved': min_resolved}, 't0': lo, 't1': hi,
            'fps': float(T.get('fps') or 25.0), 'halves': list(halves or ()), 'codes': list(CODES),
            'timeline': columns(runs), 'changes': changes, 'stats': st, 'ballStates': bs, 'notes': notes}


def columns(runs):
    """intervals() as columns (what the page loads: about 40 bytes a run): t0, t1, state (CODES index), designation (0
    none, 1 observed, 2 inferred), confidence (percent, -1 none), possessor (index into people, -1 none), ballSeen
    (percent, -1 unknown), decoded (CODES index under a possible stoppage, -1 otherwise), and people [{id, label}]."""
    people, pidx = [], {}
    cols = {k: [] for k in ('t0', 't1', 'state', 'designation', 'confidence', 'possessor', 'ballSeen', 'decoded')}
    for r in runs:
        cols['t0'].append(r['t0'])
        cols['t1'].append(r['t1'])
        cols['state'].append(CODES.index(r['state']))
        cols['designation'].append({'observed': 1, 'inferred': 2}.get(r.get('designation'), 0))
        cols['confidence'].append(int(round(100*r['confidence'])) if r.get('confidence') is not None else -1)
        pid = r.get('possessor')
        if pid is not None and pid not in pidx:
            pidx[pid] = len(people)
            people.append({'id': pid, 'label': r.get('possessorLabel') or pid})
        cols['possessor'].append(pidx[pid] if pid is not None else -1)
        cols['ballSeen'].append(int(round(100*r['ballSeen'])) if r.get('ballSeen') is not None else -1)
        cols['decoded'].append(CODES.index(r['decoded']) if r.get('decoded') else -1)
    cols['people'] = people
    return cols


def runs_of(record):
    """(t0, t1, code, designation) of a record's timeline columns (stats_from_runs reads them)."""
    c = record['timeline']
    return list(zip(c['t0'], c['t1'], c['state'], c['designation']))


def range_stats(record, lo=None, hi=None):
    """stats_from_runs over a record's timeline clipped to [lo, hi) (a custom range of the page)."""
    return stats_from_runs(runs_of(record), lo=lo, hi=hi, min_resolved=(record.get('settings') or {}).get('minResolved', 30.0))


def stoppage_codes(T, stoppages):
    """Per frame STOPPAGE inside the given intervals [{'t0', 't1'}], -1 elsewhere (live)."""
    t = T['t']
    out = np.full(T['n'], -1, np.int8)
    for s in stoppages or []:
        try:
            a = int(np.searchsorted(t, float(s['t0'])-1e-9))
            b = int(np.searchsorted(t, float(s['t1'])+1e-9, side='right'))
        except (KeyError, TypeError, ValueError):
            continue
        out[a:b] = STOPPAGE
    return out


def analyse(result, stoppages=None, settings=None, params=None, restarts=None, live_code=None):
    """(T, out, post, path, M, aux, bstate) of one result decoded as one timeline: the resolved table, label()'s output,
    the posterior, the Viterbi path, the classes, their aux arrays and the ball's observation states. stoppages
    (intervals [{'t0', 't1'}]) break the timeline and are left out; live_code (per frame, label()'s convention) replaces
    them when given; restarts {time: 'A'|'B'} start the sequence after a stoppage with that team as a prior."""
    ends = (result.get('tactical') or {}).get('goalEnds') or {}
    left, right = ends.get('left'), ends.get('right')
    if left in TEAMS and right not in TEAMS:
        right = TEAMS[1-TEAMS.index(left)]
    elif right in TEAMS and left not in TEAMS:
        left = TEAMS[1-TEAMS.index(right)]
    T = resolve(table(result), ends=(left, right) if left in TEAMS and right in TEAMS and left != right else None, S=settings)
    M, aux = classes(T, settings)
    codes = stoppage_codes(T, stoppages) if live_code is None else np.asarray(live_code, np.int8)
    decoded = (codes == -1) | (codes == POSSIBLE)
    starts = {}
    t = T['t']
    rt = sorted((float(k), v) for k, v in (restarts or {}).items() if v in TEAMS)
    for a, b in _runs(decoded):
        starts[a] = None
        for tk, team in rt:
            if abs(tk-t[a]) <= 2.0:
                starts[a] = TEAMS.index(team)
    post, path = decode(T, M, decoded, model(params), starts=starts, S=settings)
    out = label(T, post, path, M, aux, codes, settings)
    bst, _, _ = ball_states(T)
    return T, out, post, path, M, aux, bst


def compute(result, stoppages=None, settings=None, params=None, restarts=None):
    """The record of one result (a Segment run): its frames decoded as one timeline, broken at the given stoppages
    (intervals [{'t0', 't1'}]: in Segment mode the dead-ball spells of the run's possession record, which possession-4
    measured at dead precision .98 on Metrica) and at camera cuts; restarts {time: 'A'|'B'} start the sequence after a
    stoppage with that team as a prior. Returns the record (record()), with source notes."""
    T, out, post, path, M, aux, bst = analyse(result, stoppages, settings, params, restarts)
    rec = record(T, out, post, path, M, bst, settings=settings)
    rec['source'] = {'kind': 'segment', 'state': "the run's dead-ball spells" if stoppages else 'none (no stoppages known)',
                     'people': 'segment identities'}
    if not stoppages:
        rec['notes'].insert(0, 'No stoppage is known for this run: every frame counts as live play.')
    return rec
