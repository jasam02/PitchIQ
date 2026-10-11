"""ACTIVE_PITCH: the one pitch this match is played on, tracked through the video with an explicit confidence.

Two things are kept apart. The MATCH PITCH MODEL is the physical field: a rectangle with standard
markings whose length and width are learned once (field.PitchSize) and never rebuilt per frame. The
CURRENT CAMERA HOMOGRAPHY says where that rectangle projects into the present frame: it is anchored by
calibration keyframes, carried by the camera motion between them, and refined only by painted lines
that agree with its prediction. Off-screen edges are predicted from the model, never read from the
grass. Its confidence falls while it is carried without confirmation, falls fast on unreliable camera
motion and when the tracked people stand outside the predicted pitch, and is restored by keyframes and
confirmed paint; the levels HIGH / MEDIUM / LOW decide whether the model drives the playable area, is
used as a temporal prediction, or is not trusted at all.
"""
import math

import numpy as np

from . import field

HIGH, MEDIUM = .7, .35       # confidence levels
LOW_LEVEL = 'LOW'
DECAY = .03                  # confidence lost per second carried on reliable camera motion without confirmation
DECAY_LOOSE = .5             # ... per second of unreliable camera motion
DROP_BELOW = .1              # the camera homography is let go below this confidence
REFINE_BELOW = .95           # painted lines move the model only once a keyframe's exact anchor has aged
MAX_SHIFT = .02              # fraction of the frame width a refinement may move the visible ground
PEOPLE_MIN = .5              # share of tracked people that must stand on the predicted pitch
PEOPLE_STRIKES = 3           # consecutive checks failing that before the model is doubted
MARGIN = .06                 # pitch-fraction margin for "on the pitch" (a touchline player is on it)
REFINE_TOL = .008            # fraction of the frame width: the paint must agree with the fit this well


def levels(confidence):
    return 'HIGH' if confidence >= HIGH else 'MEDIUM' if confidence >= MEDIUM else 'LOW'


class PitchTracker:
    def __init__(self, width, height, size=None):
        self.width, self.height = width, height
        self.size = size or field.PitchSize()
        self.H = None                 # world metres -> image pixels, or None
        self.confidence = 0.0
        self.source = 'none'          # calibration | refined | none: where the current chain started
        self.state = 'none'           # none | anchored | carried | refined | lost
        self.carried = 0.0            # seconds since the last anchor or confirmation
        self.anchored_at = None
        self.confirmed_at = None
        self.lost = None              # why the model was let go, until the next anchor
        self.matched = []
        self.evidence = []
        self.events = []              # [(time, kind, message)]
        self.strikes = 0
        self.inside = None            # (inside, total) of the last people check

    # ---------- state ----------
    @property
    def level(self):
        return levels(self.confidence) if self.H is not None else 'LOW'

    def drop(self, time, why):
        if self.H is not None:
            self.events.append((time, 'lost', f'camera homography dropped: {why}'))
        self.H, self.confidence, self.state, self.lost = None, 0.0, 'lost', why
        self.matched, self.strikes = [], 0

    def anchor(self, H_image_to_world, time, rms=None, note='calibration keyframe'):
        """Adopt a calibration (image pixels -> metres, as calibration.solve returns it)."""
        try:
            H = field.orient(np.linalg.inv(np.asarray(H_image_to_world, np.float64)), self.width, self.height)
        except np.linalg.LinAlgError:
            H = None
        if H is None or field.sanity(H, self.size, self.width, self.height):
            return False
        self.H, self.confidence, self.source, self.state = H, 1.0, 'calibration', 'anchored'
        self.carried, self.anchored_at, self.confirmed_at, self.lost = 0.0, time, time, None
        self.strikes = 0
        self.events.append((time, 'anchored', f'camera homography anchored by a {note}' + (f' (landmarks agree within {rms:.2f} m)' if rms is not None else '')))
        return True

    def carry(self, M_px, reliable, cut, dt, time):
        """Move the homography with the camera (M_px: previous -> current frame, pixels)."""
        if self.H is None:
            return
        if cut:
            self.drop(time, 'camera cut (the physical pitch is kept; the view must be re-anchored)')
            return
        if reliable and M_px is not None:
            moved = field.orient(np.asarray(M_px, np.float64) @ self.H, self.width, self.height)
            if moved is None:
                self.drop(time, 'the camera motion made the homography unusable')
                return
            self.H = moved
            self.confidence -= DECAY*dt
        else:
            self.confidence -= DECAY_LOOSE*dt
        self.carried += dt
        if self.state in ('anchored', 'refined'):
            self.state = 'carried'
        if self.confidence < DROP_BELOW:
            self.drop(time, 'carried too long without confirmation' if reliable else 'unreliable camera motion')

    # ---------- evidence ----------
    def refine(self, segs, time, people=()):
        """Compare this frame's painted segments (normalized, from pitch.detect_lines) with the markings
        the model predicts; confirm or refine the homography. people: normalized foot points of the
        tracked people, which must mostly stand on the predicted pitch."""
        self.evidence = []
        if self.H is None:
            self.matched = []
            return
        W, Hh = self.width, self.height
        self.matched = field.match_segments(self.H, self.size, segs, W, Hh)
        if self.matched:
            if self.confidence < REFINE_BELOW:
                fit = field.fit({'lines': self.matched}, W, Hh, self.size, prior=self.H, max_rms_px=REFINE_TOL*W)
                shift = self._shift(fit.H) if fit.ok else math.inf
                if fit.ok and shift <= MAX_SHIFT*W:
                    before = self.confidence
                    self.H, self.state = fit.H, 'refined'
                    self.confidence = max(self.confidence, min(.96, .75+.07*len(self.matched)))
                    self.carried, self.confirmed_at = 0.0, time
                    if self.source == 'none':
                        self.source = 'refined'
                    names = sorted({m['name'] for m in self.matched})
                    self.events.append((time, 'refined', f'{len(names)} painted line(s) confirm the pitch ({", ".join(names)}); '
                                                          f'moved {shift:.1f} px, confidence {before:.2f} -> {self.confidence:.2f}'))
                else:
                    self.evidence.append(f'paint disagrees with the prediction ({fit.reason if not fit.ok else f"would move the view {shift:.0f} px"}); not applied')
            else:
                self.confirmed_at, self.carried = time, 0.0
        self._check_people(people, time)
        self.evidence = self._explain()+self.evidence

    def _shift(self, H_new):
        """Mean displacement (pixels) of ground points spread over the image between the current and a
        new homography."""
        xs, ys = np.linspace(.1, .9, 4)*self.width, np.linspace(.15, .95, 4)*self.height
        px = np.array([(x, y) for y in ys for x in xs], np.float64)
        px = px[field.image_depth(self.H, px) > 1e-9]
        if len(px) < 4:
            return math.inf
        world = field.unproject(self.H, px)
        moved = field.project(H_new, world)
        return float(np.hypot(*(moved-px).T).mean())

    def _check_people(self, people, time):
        feet = [(x*self.width, y*self.height) for x, y in people]
        if len(feet) < 4:
            self.inside = None
            return
        inside = 0
        for p in feet:
            q = field.to_pitch(self.H, self.size, p)
            inside += q is not None and -MARGIN <= q[0] <= 1+MARGIN and -MARGIN <= q[1] <= 1+MARGIN
        self.inside = (inside, len(feet))
        if inside/len(feet) < PEOPLE_MIN:
            self.strikes += 1
            if self.strikes >= PEOPLE_STRIKES and self.confidence > MEDIUM-.05:
                self.confidence = MEDIUM-.05
                self.events.append((time, 'doubt', f'most tracked people ({len(feet)-inside} of {len(feet)}) stand outside the predicted pitch; confidence lowered'))
        else:
            self.strikes = 0

    def _explain(self):
        lines = []
        predicted = field.image_lines(self.H, self.size, self.width, self.height)
        found = {m['name']: m for m in self.matched}
        for side, name in field.SIDES.items():
            if name in found:
                lines.append(f'{name}: FOUND (paint {found[name]["error"]:.0f} px from the prediction)')
            elif predicted[name]['visible']:
                lines.append(f'{name}: PREDICTED (in view, no paint matched)')
            else:
                lines.append(f'{name}: OFF-SCREEN (predicted from the model)')
        others = [n for n in found if n not in field.SIDES.values()]
        if others:
            lines.append('also found: '+', '.join(others))
        if self.inside is not None:
            lines.append(f'tracked people on the predicted pitch: {self.inside[0]} of {self.inside[1]}')
        return lines

    # ---------- queries ----------
    def to_pitch(self, point_px):
        return field.to_pitch(self.H, self.size, point_px) if self.H is not None else None

    def outline(self):
        """Normalized polygon of the pitch rectangle in this frame, or None."""
        return field.outline(self.H, self.size, self.width, self.height) if self.H is not None else None

    def sides(self):
        return field.side_status(self.H, self.size, self.width, self.height, self.matched) if self.H is not None else {}

    def status_text(self):
        if self.H is None:
            return 'NO PITCH MODEL' + (f' ({self.lost})' if self.lost else '')
        if self.state == 'anchored':
            return 'ANCHORED BY CALIBRATION'
        if self.state == 'refined':
            return 'CONFIRMED BY PAINTED LINES'
        return f'USING TEMPORAL PREDICTION ({self.carried:.1f}s since confirmation)'

    def to_json(self):
        return {'confidence': round(float(self.confidence), 3), 'level': self.level, 'state': self.state, 'source': self.source,
                'carried': round(self.carried, 2), 'size': self.size.to_json(), 'evidence': list(self.evidence),
                'status': self.status_text()}


def propagate(log, width, height, size, decay=DECAY):
    """Offline: fill the camera homography backwards in time through the recorded camera motion.

    log: per frame {'dt', 'M' (pixels, previous -> this frame), 'reliable', 'cut', 'H' (or None),
    'confidence'}. A frame without a homography takes the next frame's one moved back through that
    frame's motion (while the motion is reliable and there is no cut), with the confidence decaying
    as when carried forward; the fill stops when it would fall below MEDIUM. Returns a list of
    (H, confidence, filled) per frame."""
    n = len(log)
    out = [(e['H'], e['confidence'], False) for e in log]
    for i in range(n-2, -1, -1):
        H_here, _, _ = out[i]
        if H_here is not None:
            continue
        H_next, conf_next, _ = out[i+1]
        nxt = log[i+1]
        if H_next is None or not nxt['reliable'] or nxt['cut'] or nxt['M'] is None:
            continue
        conf = conf_next-decay*max(0.0, nxt['dt'])
        if conf < MEDIUM:
            continue
        try:
            H = field.orient(np.linalg.inv(np.asarray(nxt['M'], np.float64)) @ H_next, width, height)
        except np.linalg.LinAlgError:
            continue
        if H is None or field.sanity(H, size, width, height):
            continue
        out[i] = (H, conf, True)
    return out
