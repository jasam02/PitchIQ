"""Synthetic matches for the team shape tests: results in PitchIQ's format built from hand-placed formations.

The layouts below are placed by hand in metres on a 105 x 68 m pitch (attacking frame: X from the own goal line, Y from
the team's left touchline), with spacings that deliberately differ from the formation templates in soccer/team_shape.py,
so a test that recognises them is not the templates recognising themselves. A team moves as a block (forward and back,
side to side, stretching), every player wanders around his spot, and observations carry noise."""
import math

import numpy as np

LENGTH, WIDTH = 105.0, 68.0

LAYOUTS = {
    '4-4-2': [(28, 7), (24, 24), (24, 44), (28, 61), (43, 9), (39, 26), (39, 42), (43, 59), (55, 28), (56, 41)],
    '4-3-3': [(27, 8), (24, 25), (24, 43), (27, 60), (38, 21), (34, 34), (38, 47), (53, 9), (58, 33), (53, 59)],
    '4-2-3-1': [(28, 7), (24, 25), (24, 43), (28, 61), (35, 27), (35, 41), (47, 11), (48, 34), (47, 57), (58, 34)],
    '3-5-2': [(25, 17), (23, 34), (25, 51), (41, 4), (38, 22), (35, 34), (38, 46), (41, 64), (55, 28), (55, 40)],
    '4-1-4-1': [(27, 8), (24, 25), (24, 43), (27, 60), (33, 34), (43, 10), (41, 26), (41, 42), (43, 58), (57, 34)],
}
# The line (from the back) of each layout's players, for checks of the roles.
LINES = {'4-4-2': [0]*4+[1]*4+[2]*2, '4-3-3': [0]*4+[1]*3+[2]*3, '4-2-3-1': [0]*4+[1]*2+[2]*3+[3],
         '3-5-2': [0]*3+[1]*5+[2]*2, '4-1-4-1': [0]*4+[1]+[2]*4+[3]}


def to_pitch(X, Y, side):
    """Attacking-frame metres to normalized pitch coordinates for a team defending `side`."""
    x, y = np.asarray(X)/LENGTH, np.asarray(Y)/WIDTH
    return (x, y) if side == 'left' else (1-x, 1-y)


def team_motion(times, seed, block=10.0, lateral=6.0, stretch=.12, wander=1.2):
    """Block movement (dx, dy, sx) per time and a per-player wander generator."""
    rng = np.random.default_rng(seed)
    p1, p2, p3 = rng.uniform(0, 2*math.pi, 3)
    dx = block*np.sin(2*math.pi*times/41+p1)
    dy = lateral*np.sin(2*math.pi*times/29+p2)
    sx = 1+stretch*np.sin(2*math.pi*times/23+p3)

    def walk(n):
        steps = rng.normal(0, 1, (len(times), n, 2))
        out = np.zeros_like(steps)
        for k in range(1, len(times)):
            out[k] = .97*out[k-1]+.25*steps[k]
        return wander*out/max(1e-9, out.std())
    return dx, dy, sx, walk


def team_track(times, layout, side, seed=0, overrides=None, **motion):
    """Normalized (T, 10, 2) positions of a team holding `layout` (a name or a list per time via overrides: a function
    (times) -> list of layouts per sample) while moving as a block."""
    dx, dy, sx, walk = team_motion(times, seed, **motion)
    base = np.array(LAYOUTS[layout] if isinstance(layout, str) else layout, np.float64)
    T = len(times)
    pos = np.repeat(base[None], T, axis=0)
    if overrides is not None:
        pos = overrides(times, pos)
    centre_x = pos[:, :, 0].mean(axis=1, keepdims=True)
    X = centre_x+(pos[:, :, 0]-centre_x)*sx[:, None]+dx[:, None]
    Y = pos[:, :, 1]+dy[:, None]
    jitter = walk(pos.shape[1])
    X, Y = X+jitter[..., 0], Y+jitter[..., 1]
    x, y = to_pitch(X, Y, side)
    return np.stack([x, y], axis=2)


def goalkeeper_track(times, side, seed=0):
    rng = np.random.default_rng(seed+99)
    X = 6+1.5*np.sin(times/7)+rng.normal(0, .3, len(times))
    Y = 34+3*np.sin(times/5)
    x, y = to_pitch(X, Y, side)
    return np.stack([x, y], axis=1)


def make_result(times, teams, goal_ends=None, size_source='standard', visible=None, pitch_confidence=.9, sigma=.4,
                identity_confidence=.85, extra_people=None, timeline=None, expected=11):
    """A result from {'A': (outfield (T, N, 2), goalkeeper (T, 2) or None), 'B': ...}. visible: {'A': bool (T, N)} hides
    observations. extra_people: per frame index a list of person dicts appended as they are."""
    frames = []
    players, gks, slots = [], [], []
    for team, (out, gk) in teams.items():
        for j in range(out.shape[1]):
            pid = f'{team}-{j+1:02d}'
            players.append({'id': pid, 'display': pid, 'team': team, 'confirmed': True})
            slots.append({'id': pid, 'team': team, 'role': 'PLAYER', 'confirmed': True, 'firstSeen': float(times[0]), 'left': None})
        if gk is not None:
            gks.append({'id': f'GK-{team}', 'display': f'GK-{team}', 'team': team, 'confirmed': True})
    for k, t in enumerate(times):
        people = []
        for team, (out, gk) in teams.items():
            for j in range(out.shape[1]):
                if visible is not None and team in visible and not visible[team][k, j]:
                    continue
                x, y = out[k, j]
                if not (np.isfinite(x) and np.isfinite(y)):
                    continue
                people.append({'id': f'{team}-{j+1:02d}', 'team': team, 'role': 'PLAYER', 'state': 'confirmed',
                               'identityConfidence': identity_confidence, 'pitch': [float(x), float(y)], 'pitchConfidence': pitch_confidence,
                               'pitchSigma': [sigma, sigma]})
            if gk is not None and np.isfinite(gk[k, 0]):
                people.append({'id': f'GK-{team}', 'team': team, 'role': 'GOALKEEPER', 'state': 'confirmed', 'identityConfidence': .9,
                               'pitch': [float(gk[k, 0]), float(gk[k, 1])], 'pitchConfidence': pitch_confidence, 'pitchSigma': [sigma, sigma]})
        if extra_people and k in extra_people:
            people += extra_people[k]
        frames.append({'time': float(t), 'frame': k, 'people': people, 'calibrated': True})
    result = {'version': 3, 'video': {'fps': float(1/np.median(np.diff(times))) if len(times) > 1 else 25.0}, 'frames': frames,
              'match': {'players': players, 'goalkeepers': gks},
              'roster': {'expected': {'A': expected, 'B': expected}, 'slots': slots},
              'tactical': {'goalEnds': goal_ends if goal_ends is not None else {'left': 'A', 'right': 'B'},
                           'pitchSize': {'length': LENGTH, 'width': WIDTH, 'source': size_source}}}
    if timeline is not None:
        result['timeline'] = timeline
    return result


def match(layout_a, layout_b, seconds=90.0, fps=10.0, seed=0, side_a='left', **kw):
    """Two teams holding layouts for `seconds`, team A defending side_a."""
    times = np.arange(0, seconds, 1/fps)
    side_b = 'right' if side_a == 'left' else 'left'
    teams = {'A': (team_track(times, layout_a, side_a, seed), goalkeeper_track(times, side_a, seed)),
             'B': (team_track(times, layout_b, side_b, seed+1), goalkeeper_track(times, side_b, seed+1))}
    ends = {side_a: 'A', side_b: 'B'}
    return times, teams, ends
