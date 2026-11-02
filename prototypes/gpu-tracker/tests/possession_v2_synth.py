"""Synthetic PitchIQ results for soccer/possession_v2.py's tests: players and a ball on paths in pitch metres, seen
through a simple camera (1 m = 16 px across, 8 px down the image: the image reach metric with rows doubled is then
the pitch distance), at 25 fps. A ball path returns the position, or None where the tracker has no position; its state
per frame (TRACKED, MISSING = interpolated across a gap, UNKNOWN) is given separately."""
import math

W, H, FPS = 1920, 1080, 25.0
L, WD = 105.0, 68.0
SX, SY, X0, Y0 = 16.0, 8.0, 200.0, 300.0
PLAYER_H = 1.8*SX


def image(x, y):
    return X0+SX*x, Y0+SY*y


def lerp(points):
    """A path through (t, x, y) points, straight between them, held after the last and before the first."""
    pts = sorted(points)

    def f(t):
        if t <= pts[0][0]:
            return pts[0][1], pts[0][2]
        for (t0, x0, y0), (t1, x1, y1) in zip(pts, pts[1:]):
            if t <= t1:
                u = (t-t0)/max(t1-t0, 1e-9)
                return x0+u*(x1-x0), y0+u*(y1-y0)
        return pts[-1][1], pts[-1][2]
    return f


class Scene:
    def __init__(self, seconds, fps=FPS, start=0.0):
        self.n = int(round(seconds*fps))
        self.fps = fps
        self.start = start
        self.players = []
        self.ball_path = lambda t: None
        self.ball_state = lambda t: 'TRACKED'
        self.near = lambda t: None
        self.cuts = set()

    def player(self, pid, team, path, role='PLAYER', visible=None):
        self.players.append({'id': pid, 'team': team, 'path': path, 'role': role, 'visible': visible or (lambda t: True),
                             'track': len(self.players)+1})
        return self

    def ball(self, path, state=None, near=None):
        self.ball_path = path
        if state is not None:
            self.ball_state = state
        if near is not None:
            self.near = near
        return self

    def time(self, i):
        return self.start+i/self.fps

    def result(self, frames=None):
        out = []
        idx = range(self.n) if frames is None else frames
        for i in idx:
            t = self.time(i)
            people = []
            for p in self.players:
                if not p['visible'](t):
                    continue
                x, y = p['path'](t)
                fx, fy = image(x, y)
                h = PLAYER_H
                people.append({'id': p['id'], 'track': p['track'], 'team': p['team'], 'role': p['role'], 'state': 'confirmed',
                               'box': [(fx-.2*h)/W, (fy-h)/H, .4*h/W, h/H], 'pitch': [x/L, y/WD], 'pitchConfidence': .9,
                               'pitchSigma': [.3, .3], 'display': p['id']})
            pos = self.ball_path(t)
            st = self.ball_state(t) if pos is not None else 'UNKNOWN'
            if st == 'UNKNOWN' or pos is None:
                ball = {'state': 'UNKNOWN', 'origin': 'UNKNOWN', 'motion': 'UNKNOWN'}
            else:
                x, y = pos
                bx, by = image(x, y)
                p0, p1 = self.ball_path(t-.1), self.ball_path(t+.1)
                speed = math.hypot(p1[0]-p0[0], p1[1]-p0[1])/.2 if p0 is not None and p1 is not None else 0.0
                motion = 'OCCLUDED' if st == 'MISSING' else ('FAST' if speed >= 8 else ('ROLLING' if speed >= .5 else 'STATIONARY'))
                ball = {'state': st, 'origin': 'INTERPOLATION' if st == 'MISSING' else 'OBSERVATION', 'motion': motion,
                        'center': [bx/W, (by-3)/H], 'box': [(bx-3)/W, (by-6)/H, 6/W, 6/H], 'pitch': [x/L, y/WD],
                        'pitchConfidence': .9, 'pitchSigma': [.3, .3], 'speed': round(speed, 2)}
                nt = self.near(t)
                if nt is not None:
                    track = next((p['track'] for p in self.players if p['id'] == nt), None)
                    ball['nearTrack'] = track
            out.append({'frame': int(round(t*self.fps)), 'time': round(t, 6), 'people': people, 'ball': ball,
                        'cut': i in self.cuts})
        return {'version': 3, 'video': {'width': W, 'height': H, 'fps': self.fps}, 'frames': out,
                'tactical': {'goalEnds': {'left': 'A', 'right': 'B'}, 'pitchSize': {'length': L, 'width': WD}},
                'roster': [{'id': p['id'], 'team': p['team']} for p in self.players]}


def at(path, dx=0.4, dy=0.0):
    """A ball path at a player's feet (just in front of him)."""
    return lambda t: (path(t)[0]+dx, path(t)[1]+dy)


def switch(*parts):
    """A path made of (until time, path) parts: the first part whose time is not past applies."""
    def f(t):
        for until, p in parts:
            if t < until:
                return p(t)
        return parts[-1][1](t)
    return f
