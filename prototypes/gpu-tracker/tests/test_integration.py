"""End-to-end check of the soccer layer with the real local tracker (BoT-SORT) and the real OSNet model
on rendered frames. Detections come from the rendered ground truth (the CUDA detector is not used).
Skipped when Ultralytics or ONNX Runtime is not installed."""
import importlib.util
import os
import unittest

import numpy as np

from scene import GRASS, GRASS_DARK, H, W, WHITE, draw_person, person_box, row

AVAILABLE = all(importlib.util.find_spec(m) for m in ('ultralytics', 'onnxruntime')) and \
    os.path.exists(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'public', 'models', 'osnet-x025.onnx'))
FPS = 15
PANORAMA = int(W*1.25)


def panorama():
    rng = np.random.default_rng(5)
    img = rng.integers(40, 200, size=(H, PANORAMA, 3), dtype=np.uint8)
    ys = np.arange(H)/H
    xs = np.arange(PANORAMA)/W
    img[(ys >= .27) & (ys < .30)] = (150, 60, 20)
    stripes = (xs*12).astype(int) % 2 == 0
    img[ys >= .30] = np.where(stripes[:, None], GRASS, GRASS_DARK)
    img[ys >= .96] = (70, 70, 80)
    for y in (.34, .9):
        img[int(y*H)-1:int(y*H)+2, int(.03*W):int(1.22*W)] = WHITE
    img[int(.34*H):int(.9*H), int(.625*W)-1:int(.625*W)+2] = WHITE
    return img


class Actor:
    def __init__(self, kit, cls, path, skin, hair):
        self.kit, self.cls, self.path, self.skin, self.hair = kit, cls, path, skin, hair


def world_actors():
    """Positions in panorama-normalized x (0..1.25) as functions of time."""
    rng = np.random.default_rng(11)
    tone = lambda: tuple(int(v) for v in rng.integers(60, 220, 3))
    actors = {}
    starts = {'A1': (.25, .5), 'A2': (.35, .7), 'A3': (.45, .55), 'B1': (.55, .6), 'B2': (.65, .45), 'B3': (.75, .75), 'B4': (.4, .82)}
    for name, (x, y) in starts.items():
        actors[name] = Actor(name[0], 2, (lambda t, x=x, y=y: (x+.01*np.sin(t), y+.01*np.cos(t))), tone(), tone())
    actors['REF'] = Actor('REF', 3, lambda t: (.5+.02*np.sin(t/2), .62), tone(), tone())
    # A4 runs off the right edge of the view, waits outside it and comes back.
    def runner(t):
        if t < 3:
            return .9+.05*t, .5
        if t < 9:
            return 1.4, .5
        return max(.95, 1.25-.1*(t-9)), .5
    actors['A4'] = Actor('A', 2, runner, tone(), tone())
    fans = [Actor('FAN', 2, (lambda t, x=x: (x, .2)), tone(), tone()) for x in (.3, .36, .42, .48)]
    coach = Actor('FAN', 2, lambda t: (.6, .99), tone(), tone())
    return actors, fans, coach


def render(base, actors, t, pan):
    frame = np.ascontiguousarray(base[:, pan:pan+W])
    rows, truth = [], []
    for name, a in sorted(actors.items(), key=lambda kv: kv[1].path(t)[1]):
        x, y = a.path(t)
        fx = x*W/W-pan/W
        h = .1+.08*(y-.3)
        box = person_box(fx, y, h)
        if box[0] < -box[2]*.3 or box[0]+box[2] > 1+box[2]*.3:
            continue
        draw_person(frame, box, a.kit, a.skin)
        frame[int(box[1]*H):int((box[1]+box[3]*.08)*H), max(0, int((box[0]+box[2]*.3)*W)):max(0, int((box[0]+box[2]*.7)*W))] = a.hair
        clipped = [max(0, box[0]), box[1], min(1, box[0]+box[2])-max(0, box[0]), box[3]]
        rows.append(row(clipped, .85, a.cls))
        truth.append(name)
    return frame, np.asarray(rows, np.float32).reshape(-1, 6), truth


@unittest.skipUnless(AVAILABLE, 'needs ultralytics, onnxruntime and public/models/osnet-x025.onnx')
class IntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from soccer.local import BotSortTracker
        from soccer.tracker import SoccerTracker, config_from
        from vision import Appearance
        actors, fans, coach = world_actors()
        everyone = dict(actors, **{f'FAN{i}': f for i, f in enumerate(fans)}, COACH=coach)
        tracker = SoccerTracker(W, H, config_from({'maxPerTeam': 4}), BotSortTracker(FPS), Appearance().embed)
        base = panorama()
        cls.assigned = {}
        for i in range(int(16*FPS)):
            t = i/FPS
            pan = int(min(PANORAMA-W, 8*t))  # slow pan to the right
            frame, rows, truth = render(base, everyone, t, pan)
            observation, _, _ = tracker.step(frame, t, rows)
            for person in observation['people']:
                best = max(range(len(rows)), key=lambda k: _iou(rows[k], person['box']), default=None)
                if best is not None and _iou(rows[best], person['box']) > .5:
                    cls.assigned.setdefault(truth[best], []).append((t, person))
        cls.tracker = tracker

    def ids(self, name, after=0.0):
        return {p['id'] for t, p in self.assigned.get(name, []) if t >= after and p['id']}

    def test_audience_and_staff_never_get_an_identity(self):
        for name in ('FAN0', 'FAN1', 'FAN2', 'FAN3', 'COACH'):
            self.assertEqual(self.ids(name), set(), name)

    def test_referee_is_tracked_separately(self):
        self.assertEqual({i[:3] for i in self.ids('REF')}, {'REF'})
        self.assertEqual(len(self.tracker.ids.summary()['referees']), 1)

    def test_each_player_has_one_identity(self):
        for name in ('A1', 'A2', 'A3', 'B1', 'B2', 'B3', 'B4'):
            self.assertEqual(len(self.ids(name)), 1, (name, self.ids(name)))
        teams = {}
        for g in self.tracker.ids.summary()['players']:
            teams[g['team']] = teams.get(g['team'], 0)+1
        self.assertEqual(sorted(teams.values()), [4, 4])

    def test_returning_player_gets_the_old_identity(self):
        before, after = self.ids('A4', 0), self.ids('A4', 9)
        self.assertEqual(len(before), 1, before)
        self.assertTrue(after <= before, 'never a different identity after the return')
        reid = [e for e in self.tracker.events if e['kind'] == 'reid']
        self.assertTrue(any(e['playerId'] in before for e in reid), [e['message'] for e in self.tracker.events if e['kind'] in ('deferred', 'reid')][-4:])


def _iou(r, box):
    a = [r[0]/W, r[1]/H, (r[2]-r[0])/W, (r[3]-r[1])/H]
    ix = max(0, min(a[0]+a[2], box[0]+box[2])-max(a[0], box[0]))
    iy = max(0, min(a[1]+a[3], box[1]+box[3])-max(a[1], box[1]))
    inter = ix*iy
    return inter/(a[2]*a[3]+box[2]*box[3]-inter)


if __name__ == '__main__':
    unittest.main()
