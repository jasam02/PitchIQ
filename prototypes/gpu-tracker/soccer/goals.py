"""Where the goals are, and position cues for role classification.

Goal ends come from the pitch detector (a visible goal line, or the front edge of a penalty or goal
area) and are remembered in camera-stabilized coordinates of the current camera segment, so a goal
that pans out of view is still known. With a pitch calibration the penalty areas are exact.
"""
import math

from .geometry import x_at

PENALTY_DEPTH = 16.5     # metres from the goal line
PLAYER_HEIGHT = 1.8


class GoalMemory:
    def __init__(self):
        self.segment = None
        self.hints = {}      # side -> {'kind', 'a', 'b', 'time'} in stabilized normalized coordinates

    def update(self, goals, stabilize_point, segment, time):
        if segment != self.segment:
            self.hints, self.segment = {}, segment
        for g in goals or []:
            previous = self.hints.get(g['side'])
            # A goal line is better evidence than a box front; keep the freshest of the same kind.
            if previous is None or g['kind'] == 'goal-line' or previous['kind'] != 'goal-line' or time-previous['time'] > 30:
                self.hints[g['side']] = {'kind': g['kind'], 'a': stabilize_point(*g['a']), 'b': stabilize_point(*g['b']), 'time': time}

    def side_near(self, stab, aspect, pitch_point=None):
        """'left' / 'right' when the person stands in (or just in front of) a penalty area, else None.
        stab: camera-stabilized foot point and box height (x, y, h)."""
        if pitch_point is not None:
            x, y = pitch_point
            depth = (PENALTY_DEPTH+2)/105
            if .15 <= y <= .85:
                if x <= depth:
                    return 'left'
                if x >= 1-depth:
                    return 'right'
            return None
        for side, hint in self.hints.items():
            line_x = x_at((hint['a'], hint['b']), stab.y)
            inward = (stab.x-line_x) if side == 'left' else (line_x-stab.x)
            metres = inward*aspect/max(stab.h, 1e-3)*PLAYER_HEIGHT
            if hint['kind'] == 'goal-line' and -2 <= metres <= PENALTY_DEPTH+2:
                return side
            if hint['kind'] == 'box-front' and metres <= 2 and metres >= -PENALTY_DEPTH-4:
                return side
        return None


def position_cues(samples, goals, aspect):
    """Per sample: near a goal (and which), the deepest person towards a side, isolated from others."""
    out = {}
    people = [s for s in samples if s['zone'] in ('inside', 'boundary')]
    ordered = sorted(people, key=lambda s: s['stab'].x)
    deepest = set()
    if len(ordered) >= 4:
        for first, second in ((ordered[0], ordered[1]), (ordered[-1], ordered[-2])):
            h = max(first['stab'].h, 1e-3)
            if abs(first['stab'].x-second['stab'].x)*aspect >= 1.5*h:
                deepest.add(first['id'])
    for s in samples:
        st = s['stab']
        nearest = min((math.hypot((st.x-o['stab'].x)*aspect, st.y-o['stab'].y) for o in people if o is not s), default=math.inf)
        side = goals.side_near(st, aspect, s.get('pitch'))
        out[s['id']] = {'near_goal': side is not None, 'goal_side': side, 'extreme': s['id'] in deepest,
                        'isolated': nearest >= 3*max(st.h, 1e-3)}
    return out
