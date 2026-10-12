"""ACTIVE_PITCH: the one pitch this match is played on, tracked through the video with an explicit confidence.

Two things are kept apart. The MATCH PITCH MODEL is the physical field: a rectangle with standard
markings whose length and width are learned once (field.PitchSize) and never rebuilt per frame. The
CURRENT CAMERA HOMOGRAPHY says where that rectangle projects into the present frame: it is anchored by
calibration keyframes, carried by the camera motion between them, and refined only by painted lines
that agree with its prediction. Off-screen edges are predicted from the model, never read from the
grass. Its confidence is two things kept apart. TEMPORAL confidence falls while the model is carried without
confirmation, falls fast on unreliable camera motion and when the tracked people stand outside the
predicted pitch, and is restored by keyframes and confirmed paint. OBSERVABILITY is the geometric
certainty of the evidence the chain rests on (field.observability: landmarks spanning both axes, both
line families, the centre circle, conditioning, residual); a smoothly carried weak anchor stays weak.
The reported confidence is the temporal one scaled by the observability, so it can only be HIGH when the
geometry is actually constrained; the levels HIGH / MEDIUM / LOW decide whether the model drives the
playable area, is used as a temporal prediction, or is not trusted at all.
"""
import math

import numpy as np

from . import field

HIGH, MEDIUM = .7, .35       # confidence levels
LOW_LEVEL = 'LOW'
DECAY = .03                  # confidence lost per second carried on reliable camera motion without confirmation
DECAY_COARSE = .12           # ... per second carried on a coarse (whole-frame translation) estimate
DECAY_LOOSE = .5             # ... per second of unreliable camera motion
MIN_ANCHOR = .15             # observability below which a calibration is refused as an anchor
DROP_BELOW = .1              # the camera homography is let go below this confidence
REFINE_BELOW = .95           # painted lines move the model only once a keyframe's exact anchor has aged
MAX_SHIFT = .02              # fraction of the frame width a refinement may move the visible ground
PEOPLE_MIN = .5              # share of tracked people that must stand on the predicted pitch
PEOPLE_STRIKES = 3           # consecutive checks failing that before the model is doubted
MARGIN = .06                 # pitch-fraction margin for "on the pitch" (a touchline player is on it)
REFINE_TOL = .008            # fraction of the frame width: the paint must agree with the fit this well


def levels(confidence):
    return 'HIGH' if confidence >= HIGH else 'MEDIUM' if confidence >= MEDIUM else 'LOW'


def combine(temporal, observability):
    """The reported confidence: temporal certainty scaled by geometric certainty. A perfectly carried
    anchor of observability 0 is worth 0.25 (LOW); observability 0.6 can reach HIGH."""
    return max(0.0, temporal)*(.25+.75*max(0.0, min(1.0, observability)))


class PitchTracker:
    def __init__(self, width, height, size=None):
        self.width, self.height = width, height
        self.size = size or field.PitchSize()
        self.H = None                 # world metres -> image pixels, or None
        self.temporal = 0.0           # how recently and how well the chain was anchored or confirmed
        self.observability = 0.0      # geometric certainty of the evidence the chain rests on
        self.details = {}             # the observability's parts, for the overlay
        self.source = 'none'          # calibration | refined | auto-midfield | none: where the current chain started
        self.state = 'none'           # none | anchored | carried | refined | lost
        self.carried = 0.0            # seconds since the last anchor or confirmation
        self.anchored_at = None
        self.anchored_index = None    # frame index of the anchor the current chain started from (registration.Anchor.root)
        self.confirmed_at = None
        self.lost = None              # why the model was let go, until the next anchor
        self.matched = []
        self.evidence = []
        self.events = []              # [(time, kind, message)]
        self.strikes = 0
        self.inside = None            # (inside, total) of the last people check

    # ---------- state ----------
    @property
    def confidence(self):
        return combine(self.temporal, self.observability) if self.H is not None else 0.0

    @property
    def level(self):
        return levels(self.confidence) if self.H is not None else 'LOW'

    def drop(self, time, why):
        if self.H is not None:
            self.events.append((time, 'lost', f'camera homography dropped: {why}'))
        self.H, self.temporal, self.observability, self.state, self.lost = None, 0.0, 0.0, 'lost', why
        self.details, self.matched, self.strikes = {}, [], 0
        self.anchored_index = None

    def anchor(self, H_image_to_world, time, rms=None, note='calibration keyframe', observability=None, details=None, source='calibration', index=None):
        """Adopt a calibration (image pixels -> metres, as calibration.solve returns it). observability:
        the geometric certainty of its evidence (field.observability); without one the anchor is taken
        as fully constrained. An anchor below MIN_ANCHOR is refused: it would only be carried smoothly."""
        try:
            H = field.orient(np.linalg.inv(np.asarray(H_image_to_world, np.float64)), self.width, self.height)
        except np.linalg.LinAlgError:
            H = None
        if H is None or field.sanity(H, self.size, self.width, self.height):
            return False
        observability = 1.0 if observability is None else float(observability)
        if observability < MIN_ANCHOR:
            self.events.append((time, 'refused', f'{note} refused as an anchor: observability {observability:.2f} (its evidence does not constrain the pitch)'))
            return False
        self.H, self.temporal, self.observability, self.source, self.state = H, 1.0, observability, source, 'anchored'
        self.details = dict(details or {})
        self.carried, self.anchored_at, self.confirmed_at, self.lost = 0.0, time, time, None
        self.anchored_index = index
        self.strikes = 0
        self.events.append((time, 'anchored', f'camera homography anchored by a {note}' + (f' (landmarks agree within {rms:.2f} m)' if rms is not None else '') +
                            f'; observability {observability:.2f}, confidence {self.confidence:.2f}'))
        return True

    def carry(self, M_px, reliable, cut, dt, time, quality='flow'):
        """Move the homography with the camera (M_px: previous -> current frame, pixels). quality: how
        the motion was measured; a coarse estimate decays the temporal confidence faster."""
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
            self.temporal -= (DECAY_COARSE if quality == 'coarse' else DECAY)*dt
        else:
            self.temporal -= DECAY_LOOSE*dt
        self.carried += dt
        if self.state in ('anchored', 'refined'):
            self.state = 'carried'
        if self.temporal < DROP_BELOW:
            self.drop(time, 'carried too long without confirmation' if reliable else 'unreliable camera motion')

    # ---------- evidence ----------
    def refine(self, segs, time, people=(), circle=None):
        """Compare this frame's painted segments (normalized, from pitch.detect_lines) and centre-circle
        paint (pixels, from pitch.circle_points) with the markings the model predicts; confirm or refine
        the homography. people: normalized foot points of the tracked people, which must mostly stand on
        the predicted pitch."""
        self.evidence = []
        if self.H is None:
            self.matched = []
            return
        W, Hh = self.width, self.height
        self.matched = field.match_segments(self.H, self.size, segs, W, Hh)
        circle = list(circle or ())
        if self.matched or circle:
            if self.temporal < REFINE_BELOW:
                obs = {'lines': self.matched, 'circle': circle}
                fit = field.fit(obs, W, Hh, self.size, prior=self.H, max_rms_px=REFINE_TOL*W)
                shift = self._shift(fit.H) if fit.ok else math.inf
                if fit.ok and shift <= MAX_SHIFT*W:
                    before = self.confidence
                    self.H, self.state = fit.H, 'refined'
                    # Temporal certainty is restored by confirmation; geometric certainty only rises when
                    # this evidence constrains the pitch better than what the chain rested on.
                    self.temporal = max(self.temporal, min(.96, .75+.07*len(self.matched)+.1*bool(circle)))
                    found = field.observability(dict(obs, H=fit.H), W, Hh, self.size, rms_px=fit.rms_px, jac=fit.jac)
                    if found['score'] > self.observability:
                        self.observability, self.details = found['score'], found
                    self.carried, self.confirmed_at = 0.0, time
                    if self.source == 'none':
                        self.source = 'refined'
                    names = sorted({m['name'] for m in self.matched})+(['centre circle'] if circle else [])
                    self.events.append((time, 'refined', f'{len(names)} marking(s) confirm the pitch ({", ".join(names)}); '
                                                          f'moved {shift:.1f} px, confidence {before:.2f} -> {self.confidence:.2f} '
                                                          f'(observability {self.observability:.2f}, this evidence alone {found["score"]:.2f})'))
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
                self.temporal = min(self.temporal, (MEDIUM-.05)/max(1e-6, .25+.75*self.observability))
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
            elif side in ('left', 'right') and predicted[name]['visible'] and field.at_horizon(self.H, self.size, self.width, self.height, side):
                lines.append(f'{name}: AT THE HORIZON (the touchlines converge there: a vanishing point, not a measured pitch end)')
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
        geometry = f' · observability {self.observability:.2f}'
        if self.state == 'anchored':
            return ('ANCHORED BY CALIBRATION' if self.source == 'calibration' else 'ANCHORED BY THE CENTRE CIRCLE AND HALFWAY LINE')+geometry
        if self.state == 'refined':
            return 'CONFIRMED BY PAINTED MARKINGS'+geometry
        return f'USING TEMPORAL PREDICTION ({self.carried:.1f}s since confirmation)'+geometry

    def to_json(self):
        return {'confidence': round(float(self.confidence), 3), 'temporal': round(float(max(0.0, self.temporal)), 3),
                'observability': round(float(self.observability), 3), 'level': self.level, 'state': self.state, 'source': self.source,
                'carried': round(self.carried, 2), 'size': self.size.to_json(), 'evidence': list(self.evidence),
                'status': self.status_text(), 'details': {k: v for k, v in self.details.items() if not isinstance(v, (list, dict))}}


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
