"""Ball-tracking evaluation on a rendered, panning broadcast sequence (synthetic and deterministic).

Twelve seconds of play with everything that trips ball trackers up: painted lines (touchlines,
halfway line, centre circle, boxes), the centre and penalty spots, debris on the path of the ball, a
player's white wrist tape swinging as they run, white boots, players overlapping the ball, a pass, a
long shot with a hop, the ball crossing the halfway line and the centre circle, the ball hidden
behind players and under a dribble, a camera following the play and a short burst of unreliable
camera motion. The stub detector proposes the ball (with a modest, varying confidence) AND every
distractor, the tape with a higher confidence than the ball.

Run it directly for a report:

    python tests/ball_eval.py

The metrics ask "did the system keep following the SAME ball?", not "did the detector find a ball".
"""
import math
import sys

import cv2
import numpy as np

import broadcast as B
from scene import draw_person
from soccer.ball import BallTracker
from soccer.geometry import to_normalized
from soccer.pitch import PitchDetector, warp_model

FPS = 25
W, H = B.W, B.H
SPAN = 45.0
S2 = np.diag([2.0, 2.0, 1.0])


def camera_at(cx):
    return B.camera(centre_x=cx, span=SPAN)


def px_per_m(P, x, y):
    return B._px_per_m(P, x, y)


def image_point(P, x, y, height=0.0):
    cx, cy = B._project(P, [(x, y)])[0]
    return float(cx), float(cy-height*px_per_m(P, x, y))


def person_box(P, x, y):
    cx, cy = B._project(P, [(x, y)])[0]
    h = 1.8*px_per_m(P, x, y)
    w = h*.38
    return [float((cx-w/2)/W), float((cy-h)/H), float(w/W), float(h/H)]


def kick(p0, p1, v0, decel):
    """The ball rolling from p0 towards p1 at v0 m/s, slowing by `decel` m/s^2: (position(t'), arrival)."""
    dx, dy = p1[0]-p0[0], p1[1]-p0[1]
    length = math.hypot(dx, dy)
    ux, uy = dx/length, dy/length
    arrival = (v0-math.sqrt(max(0.0, v0*v0-2*decel*length)))/decel

    def at(tp):
        s = min(length, v0*tp-.5*decel*tp*tp)
        return p0[0]+ux*s, p0[1]+uy*s
    return at, arrival


class Match:
    """Scripted play in pitch metres. Players 1..7; player 4 wears white wrist tape, player 5 white boots."""

    def __init__(self):
        self.pass1, a1 = kick((32.4, 30.6), (46, 42.5), 12.5, 2.5)          # t = 1.0: player 1 passes to player 2
        self.t_pass1, self.t_arrive1 = 1.0, 1.0+a1
        self.t_dribble = (3.0, 4.2)                                           # player 2 dribbles, the ball in and out of view
        start2 = self.dribble_ball(4.2)
        self.shot, a2 = kick(start2, (70, 24), 24.0, 3.0)                     # t = 4.2: long shot to player 3, with a hop
        self.t_shot, self.t_arrive2 = 4.2, 4.2+a2
        self.pass2, a3 = kick((70, 24.5), (79.05, 44.9), 11.0, 2.0)          # t = 6.6: pass to the running player 4
        self.t_pass2, self.t_arrive3 = 6.6, 6.6+a3
        self.t_dribble4 = (self.t_arrive3, self.t_arrive3+1.0)
        start4 = self.dribble4_ball(self.t_dribble4[1])
        self.pass3, _ = kick(start4, (70, 58), 10.0, 2.0)                     # player 4 passes down the line
        self.t_pass3 = self.t_dribble4[1]

    def p2(self, t):
        f = min(1.0, max(0.0, (t-self.t_dribble[0])/(self.t_dribble[1]-self.t_dribble[0])))
        return 46+5*f, 42+f

    def p4(self, t):
        return 62+22*t/12, 48-4*t/12

    def dribble_ball(self, t):
        x, y = self.p2(t)
        phase = 2*math.pi*(t-self.t_dribble[0])/.6
        return x+1.0*(.5+.5*math.sin(phase)), y+.6*math.cos(phase)

    def dribble4_ball(self, t):
        x, y = self.p4(t)
        phase = 2*math.pi*(t-self.t_dribble4[0])/.6
        return x+1.0*(.5+.5*math.sin(phase)), y+.6*math.cos(phase)

    def players(self, t):
        sway = lambda k: (.3*math.sin(t/1.3+k), .2*math.cos(t/1.7+k))
        out = {1: (32+sway(1)[0], 30+sway(1)[1]), 2: self.p2(t), 3: (70+sway(3)[0], 24+sway(3)[1]), 4: self.p4(t),
               5: (58+.4*math.sin(t/.9), 36.8+.3*math.cos(t/1.1)), 6: (44+sway(6)[0], 26+sway(6)[1]), 7: (30+sway(7)[0], 50+sway(7)[1])}
        return out

    def ball(self, t):
        """(x, y, height) of the ball."""
        if t < self.t_pass1:
            return 32.4, 30.6, 0.0
        if t < self.t_arrive1:
            x, y = self.pass1(t-self.t_pass1)
            return x, y, 0.0
        if t < self.t_dribble[0]:
            return 46.0, 42.5, 0.0
        if t < self.t_shot:
            x, y = self.dribble_ball(t)
            return x, y, 0.0
        if t < self.t_arrive2:
            x, y = self.shot(t-self.t_shot)
            s = (t-self.t_shot)/(self.t_arrive2-self.t_shot)
            return x, y, 1.2*4*s*(1-s)
        if t < self.t_pass2:
            return 70.0, 24.5, 0.0
        if t < self.t_arrive3:
            x, y = self.pass2(t-self.t_pass2)
            return x, y, 0.0
        if t < self.t_pass3:
            x, y = self.dribble4_ball(t)
            return x, y, 0.0
        x, y = self.pass3(t-self.t_pass3)
        return x, y, 0.0


DISTRACTORS = {'centre spot': (52.5, 34.0, .45), 'penalty spot': (94.0, 34.0, .45), 'penalty spot left': (11.0, 34.0, .45),
               'halfway line': (52.5, 48.0, .3), 'circle 1': (52.5+9.15*math.cos(.5), 34+9.15*math.sin(.5), .35),
               'circle 2': (52.5+9.15*math.cos(3.6), 34+9.15*math.sin(3.6), .35), 'debris': (74.0, 33.6, .45)}


class Sequence:
    """Renders the frames, proposes detector candidates and keeps the ground truth."""

    def __init__(self, seconds=12.0, calibrated=False, ball_until=math.inf, unreliable=(7.0, 7.2), noise=True, seed=0,
                 spot_conf=.45, tape_conf=.78, hidden=(), undetected=()):
        """ball_until: the ball leaves the scene at this time. hidden: [(t0, t1)] when the ball is behind a
        player whatever the geometry says. undetected: [(t0, t1)] when the detector misses the visible ball."""
        self.match = Match()
        self.seconds, self.calibrated, self.ball_until, self.unreliable, self.noise = seconds, calibrated, ball_until, unreliable, noise
        self.spot_conf, self.tape_conf, self.hidden, self.undetected = spot_conf, tape_conf, hidden, undetected
        self.rng = np.random.default_rng(seed)
        self.base_P = S2 @ B.camera(centre_x=52.5, span=125)
        self.base = B.render(self.base_P, extra_white_dots=[DISTRACTORS['debris'][:2]], size=(2*W, 2*H))
        self.centre = None

    def camera(self, t):
        bx = self.match.ball(t)[0]
        if self.centre is None:
            self.centre = bx
        self.centre += (bx-self.centre)*(1-math.exp(-(1/FPS)/.5))  # follows the ball with a lag
        return camera_at(self.centre)

    def frame(self, i):
        t = i/FPS
        P = self.camera(t)
        frame = cv2.warpPerspective(self.base, P @ np.linalg.inv(self.base_P), (W, H), flags=cv2.INTER_LINEAR)
        players = self.match.players(t)
        bx, by, bh = self.match.ball(t)
        in_scene = t < self.ball_until
        ball_px = image_point(P, bx, by, bh)
        d = max(4.0, .22*px_per_m(P, bx, by))
        rows, people, truth_objects = [], [], {}
        forced_hidden = any(t0 <= t < t1 for t0, t1 in self.hidden)
        drawn = sorted([(y, 'p', k) for k, (x, y) in players.items()]+([(by, 'b', 0)] if in_scene and not forced_hidden else []))
        visible = in_scene and not forced_hidden
        for depth, kind, k in drawn:
            if kind == 'b':
                cv2.circle(frame, (int(round(ball_px[0])), int(round(ball_px[1]))), max(2, int(round(d/2))), (240, 240, 240), -1, cv2.LINE_AA)
                continue
            x, y = players[k]
            box = person_box(P, x, y)
            draw_person(frame, box, 'A' if k in (1, 3, 5, 7) else 'B')
            if in_scene and depth > by and box[0]+.12*box[2] <= ball_px[0]/W <= box[0]+.88*box[2] and box[1]+.02*box[3] <= ball_px[1]/H <= box[1]+.98*box[3]:
                visible = False
            if k == 4:   # white wrist tape swinging with the arm
                u, v = .05+.2*math.sin(2*math.pi*t/.7), .45+.04*math.sin(2*math.pi*t/.7+1)
                tx, ty = (box[0]+u*box[2])*W, (box[1]+v*box[3])*H
                s = max(3.0, .1*px_per_m(P, x, y))
                cv2.rectangle(frame, (int(tx-s/2), int(ty-s/2)), (int(tx+s/2), int(ty+s/2)), (238, 238, 238), -1)
                rows.append([tx-s/2, ty-s/2, tx+s/2, ty+s/2, self.tape_conf, 0])
                truth_objects['wrist tape'] = (tx, ty)
            if k == 5:   # white boots
                for j, u in enumerate((.32, .68)):
                    if math.sin(2*math.pi*t/.6+j*math.pi) > -.3:
                        fx, fy = (box[0]+u*box[2])*W, (box[1]+.965*box[3])*H
                        sw, sh = max(3.0, .14*px_per_m(P, x, y)), max(2.0, .06*px_per_m(P, x, y))
                        cv2.rectangle(frame, (int(fx-sw/2), int(fy-sh/2)), (int(fx+sw/2), int(fy+sh/2)), (236, 236, 236), -1)
                        rows.append([fx-sw/2, fy-sh/2, fx+sw/2, fy+sh/2, .4, 0])
                        truth_objects[f'boot {j+1}'] = (fx, fy)
            people.append((k, box))
        missed = any(t0 <= t < t1 for t0, t1 in self.undetected)
        if visible and not missed:
            jitter = self.rng.normal(0, .4, 2)
            conf = .5+.15*math.sin(t*7)
            rows.append([ball_px[0]-d/2+jitter[0], ball_px[1]-d/2+jitter[1], ball_px[0]+d/2+jitter[0], ball_px[1]+d/2+jitter[1], conf, 0])
        for name, (x, y, conf) in DISTRACTORS.items():
            cx, cy = image_point(P, x, y)
            if 0 <= cx < W and 0 <= cy < H:
                dd = max(4.0, .22*px_per_m(P, x, y))
                rows.append([cx-dd/2, cy-dd/2, cx+dd/2, cy+dd/2, self.spot_conf if 'spot' in name or name == 'debris' else conf, 0])
                truth_objects[name] = (cx, cy)
        if self.noise and i % 7 == 0:
            nx, ny = self.rng.uniform(.1, .9), self.rng.uniform(.35, .95)
            cv2.circle(frame, (int(nx*W), int(ny*H)), 2, (185, 185, 185), -1)
            rows.append([nx*W-3, ny*H-3, nx*W+3, ny*H+3, .2, 0])
        truth = {'time': t, 'ball': ball_px, 'd': d, 'visible': visible and not missed, 'inScene': in_scene, 'px_m': px_per_m(P, bx, by), 'objects': truth_objects}
        return t, P, frame, np.asarray(rows, np.float32).reshape(-1, 6), people, truth

    def run(self, tracker=None, pitch_every=5):
        tracker = tracker or BallTracker(W, H, FPS)
        detector = PitchDetector()
        P0 = prev_P = None
        pitch = None
        records = []
        for i in range(int(self.seconds*FPS)):
            t, P, frame, rows, people, truth = self.frame(i)
            if P0 is None:
                P0 = P
            M = np.eye(3) if prev_P is None else P @ np.linalg.inv(prev_P)
            reliable = not (self.unreliable[0] <= t < self.unreliable[1])
            T = P0 @ np.linalg.inv(P)
            P_inv = np.linalg.inv(P)

            def stabilize(nx, ny, T=T):
                p = T @ np.array([nx*W, ny*H, 1.0])
                return float(p[0]/p[2]/W), float(p[1]/p[2]/H)

            def scale(y_norm, P=P, P_inv=P_inv):
                p = P_inv @ np.array([W/2, y_norm*H, 1.0])
                return 1.8*px_per_m(P, p[0]/p[2], p[1]/p[2])/H

            def to_pitch(point, P_inv=P_inv):
                p = P_inv @ np.array([point[0]*W, point[1]*H, 1.0])
                return float(p[0]/p[2]/105), float(p[1]/p[2]/68)
            motion = to_normalized(M, W, H)
            if pitch is None or i % pitch_every == 0:
                pitch = detector.analyze(frame, t, motion, reliable, False)
            else:
                pitch = warp_model(pitch, motion, t, reliable, False)
            out, cands = tracker.step(frame, t, rows, M, reliable, False, pitch, people, scale, stabilize, 0, to_pitch if self.calibrated else None)
            records.append({'truth': truth, 'out': out, 'cands': cands})
            prev_P = P
        return records, tracker


def evaluate(records, events):
    """Did the tracker keep following the actual ball? Frame-level accuracy plus continuity."""
    n = len(records)
    stats = {'frames': n, 'visibleFrames': 0, 'hiddenFrames': 0, 'correct': 0, 'falsePositives': 0, 'missed': 0, 'bridged': 0,
             'unknownWhileVisible': 0, 'trackSwitches': 0, 'incorrectAcquisitions': 0, 'acquisitions': 0,
             'stolenBy': {}, 'continuityRuns': [], 'ballUnknownShare': 0.0}
    previous_track, run_length, prev_state = None, 0, 'UNKNOWN'
    for r in records:
        truth, out = r['truth'], r['out']
        tol = max(1.5*truth['d'], 6.0)
        near = out['state'] != 'UNKNOWN' and math.hypot(out['center'][0]*W-truth['ball'][0], out['center'][1]*H-truth['ball'][1]) <= tol
        good = False
        if truth['visible']:
            stats['visibleFrames'] += 1
            if out['state'] == 'TRACKED' and near:
                stats['correct'] += 1
                good = True
            elif out['state'] == 'TRACKED':
                stats['falsePositives'] += 1
            else:
                stats['missed'] += 1
                stats['unknownWhileVisible'] += out['state'] == 'UNKNOWN'
        else:
            if truth['inScene']:
                stats['hiddenFrames'] += 1
            if out['state'] == 'TRACKED':
                stats['falsePositives'] += 1
            elif out['state'] == 'MISSING' and truth['inScene'] and \
                    math.hypot(out['center'][0]*W-truth['ball'][0], out['center'][1]*H-truth['ball'][1]) <= 2.0*truth['px_m']:
                stats['bridged'] += 1
                good = True
        if out['state'] == 'TRACKED' and not good:
            for name, (ox, oy) in truth['objects'].items():
                if math.hypot(out['center'][0]*W-ox, out['center'][1]*H-oy) <= tol:
                    stats['stolenBy'][name] = stats['stolenBy'].get(name, 0)+1
        if out['state'] != 'UNKNOWN':
            if previous_track is not None and out['track'] != previous_track:
                stats['trackSwitches'] += 1
            if prev_state == 'UNKNOWN' or (previous_track is not None and out['track'] != previous_track):
                stats['acquisitions'] += 1
                if truth['visible'] and not near:
                    stats['incorrectAcquisitions'] += 1
            previous_track = out['track']
        prev_state = out['state']
        if good:
            run_length += 1
        elif run_length:
            stats['continuityRuns'].append(run_length)
            run_length = 0
    if run_length:
        stats['continuityRuns'].append(run_length)
    runs = stats['continuityRuns']
    stats['meanContinuitySeconds'] = round(sum(runs)/len(runs)/FPS, 2) if runs else 0.0
    stats['longestContinuitySeconds'] = round(max(runs)/FPS, 2) if runs else 0.0
    stats['accuracy'] = round(stats['correct']/stats['visibleFrames'], 3) if stats['visibleFrames'] else 0.0
    stats['bridgedShare'] = round(stats['bridged']/stats['hiddenFrames'], 3) if stats['hiddenFrames'] else None
    stats['ballUnknownShare'] = round(sum(1 for r in records if r['out']['state'] == 'UNKNOWN')/n, 3)
    stats['events'] = {k: sum(1 for e in events if e['message'].startswith(k)) for k in ('BALL ACQUIRED', 'BALL REACQUIRED', 'BALL TRACK SWITCH', 'BALL LOST', 'BALL CANDIDATE PREFERRED')}
    stats['trackIds'] = sorted({r['out']['track'] for r in records if r['out']['state'] != 'UNKNOWN'})
    return stats


def report(stats):
    lines = [f"frames {stats['frames']} (ball visible {stats['visibleFrames']}, hidden {stats['hiddenFrames']})",
             f"correctly tracked: {stats['correct']} of {stats['visibleFrames']} visible frames ({stats['accuracy']:.0%})",
             f"false positives: {stats['falsePositives']} frames   missed: {stats['missed']} (unknown {stats['unknownWhileVisible']})",
             f"hidden frames bridged by prediction: {stats['bridged']} of {stats['hiddenFrames']}",
             f"track switches: {stats['trackSwitches']}   acquisitions: {stats['acquisitions']} (incorrect {stats['incorrectAcquisitions']})   track ids {stats['trackIds']}",
             f"mean continuity: {stats['meanContinuitySeconds']} s   longest: {stats['longestContinuitySeconds']} s   unknown share: {stats['ballUnknownShare']:.0%}",
             f"frames stolen by distractors: {stats['stolenBy'] or 'none'}", f"events: {stats['events']}"]
    return '\n'.join(lines)


if __name__ == '__main__':
    for name, kwargs in (('panning camera', {}), ('calibrated pitch', {'calibrated': True})):
        records, tracker = Sequence(**kwargs).run()
        print(f'== {name}')
        print(report(evaluate(records, tracker.events)))
        if '-v' in sys.argv:
            for e in tracker.events:
                print(f"[{e['time']:6.2f}] " + e['message'].replace('\n', ' | ')[:400])
